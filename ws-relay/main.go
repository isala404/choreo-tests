package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin:     func(r *http.Request) bool { return true },
}

type Client struct {
	room *Room
	conn *websocket.Conn
	send chan []byte
	id   string
}

type Room struct {
	id         string
	clients    map[*Client]bool
	broadcast  chan []byte
	register   chan *Client
	unregister chan *Client
	mu         sync.RWMutex
}

type Hub struct {
	rooms map[string]*Room
	mu    sync.RWMutex
}

type Message struct {
	Type    string          `json:"type"`
	Room    string          `json:"room,omitempty"`
	From    string          `json:"from,omitempty"`
	Payload json.RawMessage `json:"payload,omitempty"`
}

var hub = &Hub{rooms: make(map[string]*Room)}

func (h *Hub) getOrCreateRoom(id string) *Room {
	h.mu.Lock()
	defer h.mu.Unlock()

	if room, ok := h.rooms[id]; ok {
		return room
	}

	room := &Room{
		id:         id,
		clients:    make(map[*Client]bool),
		broadcast:  make(chan []byte, 256),
		register:   make(chan *Client),
		unregister: make(chan *Client),
	}
	h.rooms[id] = room
	go room.run()
	log.Printf("[room:%s] created", id)
	return room
}

func (h *Hub) removeRoom(id string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	delete(h.rooms, id)
	log.Printf("[room:%s] removed", id)
}

func (r *Room) run() {
	for {
		select {
		case client := <-r.register:
			r.mu.Lock()
			r.clients[client] = true
			count := len(r.clients)
			r.mu.Unlock()
			log.Printf("[room:%s] client %s joined (%d clients)", r.id, client.id, count)

		case client := <-r.unregister:
			r.mu.Lock()
			if _, ok := r.clients[client]; ok {
				delete(r.clients, client)
				close(client.send)
			}
			count := len(r.clients)
			r.mu.Unlock()
			log.Printf("[room:%s] client %s left (%d clients)", r.id, client.id, count)

			if count == 0 {
				hub.removeRoom(r.id)
				return
			}

		case msg := <-r.broadcast:
			r.mu.RLock()
			for client := range r.clients {
				select {
				case client.send <- msg:
				default:
					close(client.send)
					delete(r.clients, client)
				}
			}
			r.mu.RUnlock()
		}
	}
}

func (c *Client) readPump() {
	defer func() {
		c.room.unregister <- c
		c.conn.Close()
	}()

	c.conn.SetReadLimit(65536)
	c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	c.conn.SetPongHandler(func(string) error {
		c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	for {
		_, data, err := c.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("[room:%s] client %s read error: %v", c.room.id, c.id, err)
			}
			break
		}

		var msg Message
		if err := json.Unmarshal(data, &msg); err != nil {
			continue
		}

		msg.From = c.id
		msg.Room = c.room.id

		out, _ := json.Marshal(msg)
		c.room.broadcast <- out
	}
}

func (c *Client) writePump() {
	ticker := time.NewTicker(30 * time.Second)
	defer func() {
		ticker.Stop()
		c.conn.Close()
	}()

	for {
		select {
		case msg, ok := <-c.send:
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if !ok {
				c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}
			if err := c.conn.WriteMessage(websocket.TextMessage, msg); err != nil {
				return
			}

		case <-ticker.C:
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

func wsHandler(w http.ResponseWriter, r *http.Request) {
	roomID := r.URL.Query().Get("room")
	if roomID == "" {
		roomID = "default"
	}

	clientID := r.URL.Query().Get("id")
	if clientID == "" {
		clientID = r.RemoteAddr
	}

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("upgrade error: %v", err)
		return
	}

	room := hub.getOrCreateRoom(roomID)
	client := &Client{
		room: room,
		conn: conn,
		send: make(chan []byte, 256),
		id:   clientID,
	}

	room.register <- client

	go client.writePump()
	client.readPump()
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	hub.mu.RLock()
	roomCount := len(hub.rooms)
	hub.mu.RUnlock()
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "healthy",
		"rooms":  roomCount,
	})
}

func main() {
	http.HandleFunc("/ws", wsHandler)
	http.HandleFunc("/healthz", healthHandler)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("ws-relay starting on port %s", port)
	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatal(err)
	}
}
