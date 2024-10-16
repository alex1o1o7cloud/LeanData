import Mathlib

namespace NUMINAMATH_CALUDE_cost_of_groceries_l610_61022

/-- The cost of groceries problem -/
theorem cost_of_groceries
  (mango_cost : ℝ → ℝ)  -- Cost function for mangos (kg → $)
  (rice_cost : ℝ → ℝ)   -- Cost function for rice (kg → $)
  (flour_cost : ℝ → ℝ)  -- Cost function for flour (kg → $)
  (h1 : mango_cost 10 = rice_cost 10)  -- 10 kg mangos cost same as 10 kg rice
  (h2 : flour_cost 6 = rice_cost 2)    -- 6 kg flour costs same as 2 kg rice
  (h3 : ∀ x, flour_cost x = 21 * x)    -- Flour costs $21 per kg
  : mango_cost 4 + rice_cost 3 + flour_cost 5 = 546 := by
  sorry

#check cost_of_groceries

end NUMINAMATH_CALUDE_cost_of_groceries_l610_61022


namespace NUMINAMATH_CALUDE_triangle_problem_l610_61034

open Real

noncomputable def f (x : ℝ) := 2 * sin x * cos (x - π/3) - sqrt 3 / 2

theorem triangle_problem (A B C : ℝ) (a b c R : ℝ) :
  (0 < A ∧ A < π/2) →
  (0 < B ∧ B < π/2) →
  (0 < C ∧ C < π/2) →
  A + B + C = π →
  a * cos B - b * cos A = R →
  f A = 1 →
  a = 2 * R * sin A →
  b = 2 * R * sin B →
  c = 2 * R * sin C →
  (B = π/4 ∧ -1 < (R - c) / b ∧ (R - c) / b < 0) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l610_61034


namespace NUMINAMATH_CALUDE_parallel_line_equation_l610_61096

/-- A line in the plane is represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Check if a point lies on a line given by an equation ax + by + c = 0 -/
def pointOnLine (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

/-- Two lines are parallel if they have the same slope -/
def parallel (l₁ l₂ : Line) : Prop :=
  l₁.slope = l₂.slope

theorem parallel_line_equation (p : ℝ × ℝ) :
  let l₁ : Line := { slope := 2, point := (0, 0) }  -- y = 2x
  let l₂ : Line := { slope := 2, point := p }       -- parallel line through p
  parallel l₁ l₂ →
  p = (1, -2) →
  pointOnLine 2 (-1) (-4) p.1 p.2 :=
by
  sorry

#check parallel_line_equation

end NUMINAMATH_CALUDE_parallel_line_equation_l610_61096


namespace NUMINAMATH_CALUDE_sum_of_squares_l610_61003

theorem sum_of_squares (x y : ℝ) (h1 : x * (x + y) = 35) (h2 : y * (x + y) = 77) :
  (x + y)^2 = 112 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l610_61003


namespace NUMINAMATH_CALUDE_triangle_angle_identity_l610_61098

theorem triangle_angle_identity (α : Real) 
  (h1 : 0 < α ∧ α < Real.pi)  -- α is an internal angle of a triangle
  (h2 : Real.sin α * Real.cos α = 1/8) :  -- given condition
  Real.cos α + Real.sin α = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_identity_l610_61098


namespace NUMINAMATH_CALUDE_sales_difference_l610_61005

/-- Represents a company selling bottled milk -/
structure Company where
  big_bottle_price : ℝ
  small_bottle_price : ℝ
  big_bottle_discount : ℝ
  small_bottle_discount : ℝ
  big_bottles_sold : ℕ
  small_bottles_sold : ℕ

def tax_rate : ℝ := 0.07

def company_A : Company := {
  big_bottle_price := 4
  small_bottle_price := 2
  big_bottle_discount := 0.1
  small_bottle_discount := 0
  big_bottles_sold := 300
  small_bottles_sold := 400
}

def company_B : Company := {
  big_bottle_price := 3.5
  small_bottle_price := 1.75
  big_bottle_discount := 0
  small_bottle_discount := 0.05
  big_bottles_sold := 350
  small_bottles_sold := 600
}

def calculate_total_sales (c : Company) : ℝ :=
  let big_bottle_revenue := c.big_bottle_price * c.big_bottles_sold
  let small_bottle_revenue := c.small_bottle_price * c.small_bottles_sold
  let total_before_discount := big_bottle_revenue + small_bottle_revenue
  let big_bottle_discount := if c.big_bottles_sold ≥ 10 then c.big_bottle_discount * big_bottle_revenue else 0
  let small_bottle_discount := if c.small_bottles_sold > 20 then c.small_bottle_discount * small_bottle_revenue else 0
  let total_after_discount := total_before_discount - big_bottle_discount - small_bottle_discount
  let total_after_tax := total_after_discount * (1 + tax_rate)
  total_after_tax

theorem sales_difference : 
  calculate_total_sales company_B - calculate_total_sales company_A = 366.475 := by
  sorry

end NUMINAMATH_CALUDE_sales_difference_l610_61005


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l610_61008

theorem partial_fraction_decomposition :
  ∀ (A B C : ℝ),
    (∀ (x : ℝ), x ≠ 0 →
      (x^3 - 2*x^2 + x - 5) / (x^4 + x^2) = A / x^2 + (B*x + C) / (x^2 + 1)) ↔
    (A = -5 ∧ B = 1 ∧ C = 3) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l610_61008


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l610_61068

/-- Given two lines a²x + y + 7 = 0 and x - 2ay + 1 = 0 that are perpendicular,
    prove that a = 0 or a = 2 -/
theorem perpendicular_lines_a_value (a : ℝ) :
  (∀ x y : ℝ, a^2*x + y + 7 = 0 ∧ x - 2*a*y + 1 = 0 → 
    (a^2 : ℝ) * (1 / (-2*a)) = -1) →
  a = 0 ∨ a = 2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l610_61068


namespace NUMINAMATH_CALUDE_range_of_x_range_of_a_l610_61013

-- Define the conditions
def p (x : ℝ) : Prop := (x + 1) / (x - 2) > 2
def q (x a : ℝ) : Prop := x^2 - a*x + 5 > 0

-- Theorem 1: If p is true, then x is in the open interval (2,5)
theorem range_of_x (x : ℝ) : p x → x ∈ Set.Ioo 2 5 := by sorry

-- Theorem 2: If p is a sufficient but not necessary condition for q,
-- then a is in the interval (-∞, 2√5)
theorem range_of_a (a : ℝ) :
  (∀ x, p x → q x a) ∧ (∃ x, q x a ∧ ¬p x) →
  a ∈ Set.Iio (2 * Real.sqrt 5) := by sorry

end NUMINAMATH_CALUDE_range_of_x_range_of_a_l610_61013


namespace NUMINAMATH_CALUDE_sequence_properties_l610_61029

/-- Given a sequence {a_n}, where S_n is the sum of the first n terms,
    a_1 = a (a ≠ 4), and a_{n+1} = 2S_n + 4^n for n ∈ ℕ* -/
def Sequence (a : ℝ) (a_n : ℕ+ → ℝ) (S_n : ℕ+ → ℝ) : Prop :=
  a ≠ 4 ∧
  a_n 1 = a ∧
  ∀ n : ℕ+, a_n (n + 1) = 2 * S_n n + 4^(n : ℕ)

/-- Definition of b_n -/
def b_n (S_n : ℕ+ → ℝ) : ℕ+ → ℝ :=
  λ n => S_n n - 4^(n : ℕ)

theorem sequence_properties {a : ℝ} {a_n : ℕ+ → ℝ} {S_n : ℕ+ → ℝ}
    (h : Sequence a a_n S_n) :
    /- 1. {b_n} forms a geometric progression with common ratio 3 -/
    (∀ n : ℕ+, b_n S_n (n + 1) = 3 * b_n S_n n) ∧
    /- 2. General formula for {a_n} -/
    (∀ n : ℕ+, n = 1 → a_n n = a) ∧
    (∀ n : ℕ+, n ≥ 2 → a_n n = 3 * 4^(n - 1 : ℕ) + 2 * (a - 4) * 3^(n - 2 : ℕ)) ∧
    /- 3. Range of a that satisfies a_{n+1} ≥ a_n for n ∈ ℕ* -/
    (∀ n : ℕ+, a_n (n + 1) ≥ a_n n ↔ a ∈ Set.Icc (-4 : ℝ) 4 ∪ Set.Ioi 4) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l610_61029


namespace NUMINAMATH_CALUDE_transportation_is_car_l610_61020

/-- Represents different modes of transportation -/
inductive TransportMode
  | Walking
  | Bicycle
  | Car

/-- Definition of a transportation mode with its speed -/
structure Transportation where
  mode : TransportMode
  speed : ℝ  -- Speed in kilometers per hour

/-- Theorem stating that a transportation with speed 70 km/h is a car -/
theorem transportation_is_car (t : Transportation) (h : t.speed = 70) : t.mode = TransportMode.Car := by
  sorry


end NUMINAMATH_CALUDE_transportation_is_car_l610_61020


namespace NUMINAMATH_CALUDE_triangle_angle_not_all_greater_than_60_l610_61024

theorem triangle_angle_not_all_greater_than_60 :
  ¬ ∃ (a b c : ℝ),
    (0 < a ∧ 0 < b ∧ 0 < c) ∧
    (a + b + c = 180) ∧
    (60 < a ∧ 60 < b ∧ 60 < c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_not_all_greater_than_60_l610_61024


namespace NUMINAMATH_CALUDE_complex_quadrant_implies_m_range_l610_61049

def z (m : ℝ) : ℂ := Complex.mk (m + 1) (3 - m)

def in_second_or_fourth_quadrant (z : ℂ) : Prop :=
  z.re * z.im > 0

theorem complex_quadrant_implies_m_range (m : ℝ) :
  in_second_or_fourth_quadrant (z m) → m ∈ Set.Ici 3 ∪ Set.Iic (-1) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_quadrant_implies_m_range_l610_61049


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l610_61089

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem equal_area_rectangles (carol_rect jordan_rect : Rectangle) 
  (h1 : carol_rect.length = 5)
  (h2 : carol_rect.width = 24)
  (h3 : jordan_rect.length = 4)
  (h4 : area carol_rect = area jordan_rect) :
  jordan_rect.width = 30 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l610_61089


namespace NUMINAMATH_CALUDE_first_player_winning_strategy_l610_61036

/-- Represents the state of the game -/
structure GameState :=
  (stones : ℕ)

/-- Represents a valid move in the game -/
inductive Move : Type
  | take_one : Move
  | take_two : Move

/-- Applies a move to the game state -/
def apply_move (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.take_one => ⟨state.stones - 1⟩
  | Move.take_two => ⟨state.stones - 2⟩

/-- Checks if a move is valid given the current game state -/
def is_valid_move (state : GameState) (move : Move) : Prop :=
  match move with
  | Move.take_one => state.stones ≥ 1
  | Move.take_two => state.stones ≥ 2

/-- Defines the winning strategy sequence for the first player -/
def winning_sequence : List ℕ := [21, 18, 15, 12, 9, 6, 3, 0]

/-- Theorem: The first player has a winning strategy in the 22-stone game -/
theorem first_player_winning_strategy :
  ∃ (strategy : GameState → Move),
    ∀ (opponent_move : Move),
      let initial_state := GameState.mk 22
      let first_move := strategy initial_state
      let state_after_first_move := apply_move initial_state first_move
      let state_after_opponent := apply_move state_after_first_move opponent_move
      is_valid_move initial_state first_move ∧
      is_valid_move state_after_first_move opponent_move ∧
      state_after_opponent.stones ∈ winning_sequence :=
by
  sorry

end NUMINAMATH_CALUDE_first_player_winning_strategy_l610_61036


namespace NUMINAMATH_CALUDE_cos_two_alpha_zero_l610_61004

theorem cos_two_alpha_zero (α : Real) (h : Real.sin (π/6 - α) = Real.cos (π/6 + α)) : 
  Real.cos (2 * α) = 0 := by
sorry

end NUMINAMATH_CALUDE_cos_two_alpha_zero_l610_61004


namespace NUMINAMATH_CALUDE_range_of_a_l610_61056

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0) →
  a ≤ -2 ∨ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l610_61056


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l610_61033

-- Problem 1
theorem problem_1 : (1 * (1/6 - 5/7 + 2/3)) * (-42) = -5 := by sorry

-- Problem 2
theorem problem_2 : -(2^2) + (-3)^2 * (-2/3) - 4^2 / |(-4)| = -14 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l610_61033


namespace NUMINAMATH_CALUDE_pen_sales_problem_l610_61095

theorem pen_sales_problem (d : ℕ) : 
  (96 + 44 * d) / (d + 1) = 48 → d = 12 := by
  sorry

end NUMINAMATH_CALUDE_pen_sales_problem_l610_61095


namespace NUMINAMATH_CALUDE_race_distance_l610_61040

/-- The race problem -/
theorem race_distance (time_A time_B : ℝ) (lead : ℝ) (distance : ℝ) : 
  time_A = 36 →
  time_B = 45 →
  lead = 20 →
  (distance / time_A) * time_B = distance + lead →
  distance = 80 := by
sorry

end NUMINAMATH_CALUDE_race_distance_l610_61040


namespace NUMINAMATH_CALUDE_spending_ratio_l610_61080

def initial_amount : ℕ := 200
def spent_on_books : ℕ := 30
def spent_on_clothes : ℕ := 55
def spent_on_snacks : ℕ := 25
def spent_on_gift : ℕ := 20
def spent_on_electronics : ℕ := 40

def total_spent : ℕ := spent_on_books + spent_on_clothes + spent_on_snacks + spent_on_gift + spent_on_electronics
def unspent : ℕ := initial_amount - total_spent

theorem spending_ratio : 
  (total_spent : ℚ) / (unspent : ℚ) = 17 / 3 := by sorry

end NUMINAMATH_CALUDE_spending_ratio_l610_61080


namespace NUMINAMATH_CALUDE_inscribed_hexagon_area_l610_61086

/-- The area of a regular hexagon inscribed in a circle with area 16π -/
theorem inscribed_hexagon_area : 
  ∀ (circle_area : ℝ) (hexagon_area : ℝ),
  circle_area = 16 * Real.pi →
  hexagon_area = (6 * Real.sqrt 3 * circle_area) / (2 * Real.pi) →
  hexagon_area = 24 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_area_l610_61086


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l610_61094

theorem opposite_of_negative_three : -(- 3) = 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l610_61094


namespace NUMINAMATH_CALUDE_first_road_workers_approx_30_man_hours_proportional_to_length_l610_61072

/-- Represents the details of a road construction project -/
structure RoadProject where
  length : ℝ  -- Road length in km
  workers : ℝ  -- Number of workers
  days : ℝ    -- Number of days worked
  hoursPerDay : ℝ  -- Hours worked per day

/-- Calculates the total man-hours for a road project -/
def manHours (project : RoadProject) : ℝ :=
  project.workers * project.days * project.hoursPerDay

/-- The first road project -/
def road1 : RoadProject := {
  length := 1,
  workers := 30,  -- This is what we're trying to prove
  days := 12,
  hoursPerDay := 8
}

/-- The second road project -/
def road2 : RoadProject := {
  length := 2,
  workers := 20,
  days := 20.571428571428573,
  hoursPerDay := 14
}

/-- Theorem stating that the number of workers on the first road is approximately 30 -/
theorem first_road_workers_approx_30 :
  ∃ ε > 0, ε < 1 ∧ |road1.workers - 30| < ε :=
by sorry

/-- Theorem showing the relationship between man-hours and road length -/
theorem man_hours_proportional_to_length :
  2 * manHours road1 = manHours road2 :=
by sorry

end NUMINAMATH_CALUDE_first_road_workers_approx_30_man_hours_proportional_to_length_l610_61072


namespace NUMINAMATH_CALUDE_principal_amount_l610_61087

/-- Given a principal amount P and an interest rate R, if the amount after 2 years is 850
    and after 7 years is 1020, then the principal amount P is 782. -/
theorem principal_amount (P R : ℚ) : 
  P + (P * R * 2) / 100 = 850 →
  P + (P * R * 7) / 100 = 1020 →
  P = 782 := by
sorry

end NUMINAMATH_CALUDE_principal_amount_l610_61087


namespace NUMINAMATH_CALUDE_target_hit_probability_l610_61017

theorem target_hit_probability (prob_A prob_B : ℝ) : 
  prob_A = 1/2 → 
  prob_B = 1/3 → 
  1 - (1 - prob_A) * (1 - prob_B) = 2/3 := by
sorry

end NUMINAMATH_CALUDE_target_hit_probability_l610_61017


namespace NUMINAMATH_CALUDE_floor_abs_negative_real_l610_61059

theorem floor_abs_negative_real : ⌊|(-45.7 : ℝ)|⌋ = 45 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_real_l610_61059


namespace NUMINAMATH_CALUDE_at_most_one_tiling_l610_61063

/-- Represents a polyomino -/
structure Polyomino where
  squares : Set (ℕ × ℕ)
  nonempty : squares.Nonempty

/-- An L-shaped polyomino consisting of three squares -/
def l_shape : Polyomino := {
  squares := {(0,0), (0,1), (1,0)}
  nonempty := by simp
}

/-- Another polyomino with at least two squares -/
def other_polyomino : Polyomino := {
  squares := {(0,0), (0,1)}  -- Minimal example with two squares
  nonempty := by simp
}

/-- Represents a tiling of a board -/
def Tiling (n : ℕ) (p1 p2 : Polyomino) :=
  ∃ (t : Set (ℕ × ℕ × ℕ × ℕ)), 
    (∀ x y, x < n ∧ y < n → ∃ a b dx dy, (a, b, dx, dy) ∈ t ∧
      ((dx, dy) ∈ p1.squares ∨ (dx, dy) ∈ p2.squares) ∧
      (a + dx = x ∧ b + dy = y)) ∧
    (∀ (a b dx dy : ℕ), (a, b, dx, dy) ∈ t →
      (dx, dy) ∈ p1.squares ∨ (dx, dy) ∈ p2.squares)

/-- The main theorem -/
theorem at_most_one_tiling (n m : ℕ) (h : Nat.Coprime n m) :
  ¬(Tiling n l_shape other_polyomino ∧ Tiling m l_shape other_polyomino) := by
  sorry

end NUMINAMATH_CALUDE_at_most_one_tiling_l610_61063


namespace NUMINAMATH_CALUDE_system_solution_l610_61001

theorem system_solution : 
  ∀ x y : ℝ, 
  (x^3 + y^3 = 19 ∧ x^2 + y^2 + 5*x + 5*y + x*y = 12) ↔ 
  ((x = 3 ∧ y = -2) ∨ (x = -2 ∧ y = 3)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l610_61001


namespace NUMINAMATH_CALUDE_x_fourth_minus_reciprocal_l610_61093

theorem x_fourth_minus_reciprocal (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 723 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_minus_reciprocal_l610_61093


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l610_61060

-- Define the rectangular prism
structure RectangularPrism :=
  (height : ℝ)
  (base_length : ℝ)
  (base_width : ℝ)

-- Define the slant representation
structure SlantRepresentation :=
  (angle : ℝ)
  (long_side : ℝ)
  (short_side : ℝ)

-- Define the theorem
theorem rectangular_prism_volume
  (prism : RectangularPrism)
  (slant : SlantRepresentation)
  (h1 : prism.height = 1)
  (h2 : slant.angle = 45)
  (h3 : slant.long_side = 2)
  (h4 : slant.long_side = 2 * slant.short_side)
  (h5 : prism.base_length = slant.long_side)
  (h6 : prism.base_width = slant.long_side) :
  prism.height * prism.base_length * prism.base_width = 4 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l610_61060


namespace NUMINAMATH_CALUDE_a_17_value_l610_61012

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) / a n = a 2 / a 1

-- State the theorem
theorem a_17_value (a : ℕ → ℝ) (h_geo : geometric_sequence a) 
  (h_a1 : a 1 = 1) (h_a2a8 : a 2 * a 8 = 16) : a 17 = 256 := by
  sorry

end NUMINAMATH_CALUDE_a_17_value_l610_61012


namespace NUMINAMATH_CALUDE_bus_catch_probability_l610_61077

/-- The probability of catching a bus within 5 minutes -/
theorem bus_catch_probability 
  (p3 : ℝ) -- Probability of bus No. 3 arriving
  (p6 : ℝ) -- Probability of bus No. 6 arriving
  (h1 : p3 = 0.20) -- Given probability for bus No. 3
  (h2 : p6 = 0.60) -- Given probability for bus No. 6
  (h3 : 0 ≤ p3 ∧ p3 ≤ 1) -- p3 is a valid probability
  (h4 : 0 ≤ p6 ∧ p6 ≤ 1) -- p6 is a valid probability
  : p3 + p6 = 0.80 := by
  sorry

end NUMINAMATH_CALUDE_bus_catch_probability_l610_61077


namespace NUMINAMATH_CALUDE_odd_periodic_function_property_l610_61079

open Real

-- Define the properties of the function f
def is_odd_and_periodic (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 3) = -f x)

-- State the theorem
theorem odd_periodic_function_property 
  (f : ℝ → ℝ) 
  (α : ℝ) 
  (h_f : is_odd_and_periodic f) 
  (h_α : tan α = 2) : 
  f (15 * sin α * cos α) = 0 := by
sorry

end NUMINAMATH_CALUDE_odd_periodic_function_property_l610_61079


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l610_61041

theorem polynomial_evaluation :
  ∃ x : ℝ, x > 0 ∧ x^2 - 2*x - 15 = 0 ∧ x^3 - 2*x^2 - 8*x + 16 = 51 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l610_61041


namespace NUMINAMATH_CALUDE_min_trees_for_three_types_l610_61048

/-- Represents the four types of trees in the grove -/
inductive TreeType
  | Birch
  | Spruce
  | Pine
  | Aspen

/-- Represents the grove of trees -/
structure Grove :=
  (trees : Finset ℕ)
  (type : ℕ → TreeType)
  (total_trees : trees.card = 100)
  (four_types_in_85 : ∀ s : Finset ℕ, s ⊆ trees → s.card = 85 → 
    (∃ i ∈ s, type i = TreeType.Birch) ∧
    (∃ i ∈ s, type i = TreeType.Spruce) ∧
    (∃ i ∈ s, type i = TreeType.Pine) ∧
    (∃ i ∈ s, type i = TreeType.Aspen))

/-- The main theorem stating the minimum number of trees to guarantee at least three types -/
theorem min_trees_for_three_types (g : Grove) :
  ∀ s : Finset ℕ, s ⊆ g.trees → s.card ≥ 69 →
    (∃ t1 t2 t3 : TreeType, t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧
      (∃ i ∈ s, g.type i = t1) ∧
      (∃ i ∈ s, g.type i = t2) ∧
      (∃ i ∈ s, g.type i = t3)) :=
by sorry

end NUMINAMATH_CALUDE_min_trees_for_three_types_l610_61048


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l610_61045

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is a three-digit number -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∀ n : ℕ, is_three_digit n → n % 9 = 0 → digit_sum n = 27 → n ≤ 999 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l610_61045


namespace NUMINAMATH_CALUDE_total_money_calculation_l610_61047

-- Define the proportions
def prop1 : ℚ := 1/2
def prop2 : ℚ := 1/3
def prop3 : ℚ := 3/4

-- Define the value of the second part
def second_part : ℝ := 164.6315789473684

-- Theorem statement
theorem total_money_calculation (total : ℝ) :
  (total * (prop2 / (prop1 + prop2 + prop3)) = second_part) →
  total = 65.1578947368421 := by
sorry

end NUMINAMATH_CALUDE_total_money_calculation_l610_61047


namespace NUMINAMATH_CALUDE_equation_solution_l610_61069

theorem equation_solution (a b c : ℤ) :
  (∀ x : ℝ, (x - a)*(x - 5) + 2 = (x + b)*(x + c)) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l610_61069


namespace NUMINAMATH_CALUDE_window_purchase_savings_l610_61007

/-- Represents the window purchase scenario --/
structure WindowPurchase where
  regularPrice : ℕ
  freeWindows : ℕ
  purchaseThreshold : ℕ
  daveNeeds : ℕ
  dougNeeds : ℕ

/-- Calculates the cost for a given number of windows --/
def calculateCost (wp : WindowPurchase) (windows : ℕ) : ℕ :=
  let freeGroups := windows / wp.purchaseThreshold
  let paidWindows := windows - (freeGroups * wp.freeWindows)
  paidWindows * wp.regularPrice

/-- Calculates the savings when purchasing together vs separately --/
def calculateSavings (wp : WindowPurchase) : ℕ :=
  let separateCost := calculateCost wp wp.daveNeeds + calculateCost wp wp.dougNeeds
  let jointCost := calculateCost wp (wp.daveNeeds + wp.dougNeeds)
  separateCost - jointCost

/-- The main theorem stating the savings amount --/
theorem window_purchase_savings (wp : WindowPurchase) 
  (h1 : wp.regularPrice = 120)
  (h2 : wp.freeWindows = 2)
  (h3 : wp.purchaseThreshold = 6)
  (h4 : wp.daveNeeds = 12)
  (h5 : wp.dougNeeds = 9) :
  calculateSavings wp = 360 := by
  sorry


end NUMINAMATH_CALUDE_window_purchase_savings_l610_61007


namespace NUMINAMATH_CALUDE_inverse_g_sum_l610_61016

-- Define the function g
def g (x : ℝ) : ℝ := x^3 * |x|

-- State the theorem
theorem inverse_g_sum : 
  ∃ (a b : ℝ), g a = 8 ∧ g b = -64 ∧ a + b = Real.sqrt 2 - 2 :=
sorry

end NUMINAMATH_CALUDE_inverse_g_sum_l610_61016


namespace NUMINAMATH_CALUDE_tetrahedral_toys_probability_l610_61027

-- Define the face values of the tetrahedral toys
def face_values : Finset ℕ := {1, 2, 3, 5}

-- Define the sample space of all possible outcomes
def sample_space : Finset (ℕ × ℕ) := face_values.product face_values

-- Define m as the sum of the two face values
def m (outcome : ℕ × ℕ) : ℕ := outcome.1 + outcome.2

-- Define the event where m is not less than 6
def event_m_ge_6 : Finset (ℕ × ℕ) := sample_space.filter (λ x => m x ≥ 6)

-- Define the event where m is odd
def event_m_odd : Finset (ℕ × ℕ) := sample_space.filter (λ x => m x % 2 = 1)

-- Define the event where m is even
def event_m_even : Finset (ℕ × ℕ) := sample_space.filter (λ x => m x % 2 = 0)

theorem tetrahedral_toys_probability :
  (event_m_ge_6.card : ℚ) / sample_space.card = 1/2 ∧
  (event_m_odd.card : ℚ) / sample_space.card = 3/8 ∧
  (event_m_even.card : ℚ) / sample_space.card = 5/8 :=
sorry

end NUMINAMATH_CALUDE_tetrahedral_toys_probability_l610_61027


namespace NUMINAMATH_CALUDE_sharp_composition_10_l610_61062

def sharp (N : ℕ) : ℕ := N^2 - N + 2

theorem sharp_composition_10 : sharp (sharp (sharp 10)) = 70123304 := by
  sorry

end NUMINAMATH_CALUDE_sharp_composition_10_l610_61062


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l610_61015

theorem min_distance_between_curves (a : ℝ) (h : a > 0) : 
  ∃ (min_val : ℝ), min_val = 12 ∧ ∀ x > 0, |16 / x + x^2| ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l610_61015


namespace NUMINAMATH_CALUDE_leftover_seashells_proof_l610_61090

/-- The number of leftover seashells after packaging -/
def leftover_seashells (derek_shells : ℕ) (emily_shells : ℕ) (fiona_shells : ℕ) (package_size : ℕ) : ℕ :=
  (derek_shells + emily_shells + fiona_shells) % package_size

theorem leftover_seashells_proof (derek_shells emily_shells fiona_shells package_size : ℕ) 
  (h_package_size : package_size > 0) :
  leftover_seashells derek_shells emily_shells fiona_shells package_size = 
  (derek_shells + emily_shells + fiona_shells) % package_size :=
by
  sorry

#eval leftover_seashells 58 73 31 10

end NUMINAMATH_CALUDE_leftover_seashells_proof_l610_61090


namespace NUMINAMATH_CALUDE_probability_not_distinct_roots_greater_than_two_l610_61065

def is_valid_pair (a c : ℤ) : Prop :=
  |a| ≤ 6 ∧ |c| ≤ 6

def has_distinct_roots_greater_than_two (a c : ℤ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 2 ∧ x₂ > 2 ∧ a * x₁^2 - 3 * a * x₁ + c = 0 ∧ a * x₂^2 - 3 * a * x₂ + c = 0

def total_pairs : ℕ := 169

def valid_pairs : ℕ := 2

theorem probability_not_distinct_roots_greater_than_two :
  (total_pairs - valid_pairs) / total_pairs = 167 / 169 :=
sorry

end NUMINAMATH_CALUDE_probability_not_distinct_roots_greater_than_two_l610_61065


namespace NUMINAMATH_CALUDE_fuel_cost_savings_l610_61097

theorem fuel_cost_savings (old_efficiency : ℝ) (old_fuel_cost : ℝ) 
  (trip_distance : ℝ) (efficiency_improvement : ℝ) (fuel_cost_increase : ℝ) :
  old_efficiency > 0 → old_fuel_cost > 0 → trip_distance > 0 →
  efficiency_improvement = 0.6 → fuel_cost_increase = 0.25 → trip_distance = 300 →
  let new_efficiency := old_efficiency * (1 + efficiency_improvement)
  let new_fuel_cost := old_fuel_cost * (1 + fuel_cost_increase)
  let old_trip_cost := (trip_distance / old_efficiency) * old_fuel_cost
  let new_trip_cost := (trip_distance / new_efficiency) * new_fuel_cost
  let savings_percentage := (old_trip_cost - new_trip_cost) / old_trip_cost * 100
  savings_percentage = 21.875 := by
sorry

end NUMINAMATH_CALUDE_fuel_cost_savings_l610_61097


namespace NUMINAMATH_CALUDE_concatenatedDecimal_irrational_l610_61031

/-- The infinite decimal formed by concatenating all natural numbers after the decimal point -/
noncomputable def concatenatedDecimal : ℝ :=
  sorry

/-- Function that generates the n-th digit of the concatenatedDecimal -/
def nthDigit (n : ℕ) : ℕ :=
  sorry

theorem concatenatedDecimal_irrational : Irrational concatenatedDecimal :=
  sorry

end NUMINAMATH_CALUDE_concatenatedDecimal_irrational_l610_61031


namespace NUMINAMATH_CALUDE_exists_perfect_square_sum_twelve_satisfies_condition_l610_61010

theorem exists_perfect_square_sum : ∃ x : ℕ, ∃ y : ℕ, 2^x + 2^8 + 2^11 = y^2 := by
  sorry

theorem twelve_satisfies_condition : ∃ y : ℕ, 2^12 + 2^8 + 2^11 = y^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_perfect_square_sum_twelve_satisfies_condition_l610_61010


namespace NUMINAMATH_CALUDE_set_condition_implies_range_l610_61042

theorem set_condition_implies_range (a : ℝ) : 
  let A := {x : ℝ | x > 5}
  let B := {x : ℝ | x > a}
  (∀ x, x ∈ A → x ∈ B) ∧ (∃ x, x ∈ B ∧ x ∉ A) → a < 5 := by
  sorry

end NUMINAMATH_CALUDE_set_condition_implies_range_l610_61042


namespace NUMINAMATH_CALUDE_grasshopper_visits_all_integers_l610_61099

def grasshopper_jump (k : ℕ) : ℤ :=
  if k % 2 = 0 then -k else k + 1

def grasshopper_position (n : ℕ) : ℤ :=
  (List.range n).foldl (λ acc k => acc + grasshopper_jump k) 0

theorem grasshopper_visits_all_integers :
  ∀ (z : ℤ), ∃ (n : ℕ), grasshopper_position n = z :=
sorry

end NUMINAMATH_CALUDE_grasshopper_visits_all_integers_l610_61099


namespace NUMINAMATH_CALUDE_decreasing_before_vertex_l610_61085

/-- The quadratic function f(x) = (x - 4)² + 3 -/
def f (x : ℝ) : ℝ := (x - 4)^2 + 3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 2 * (x - 4)

theorem decreasing_before_vertex :
  ∀ x : ℝ, x < 4 → f' x < 0 :=
by
  sorry

end NUMINAMATH_CALUDE_decreasing_before_vertex_l610_61085


namespace NUMINAMATH_CALUDE_trigonometric_equation_solutions_l610_61046

theorem trigonometric_equation_solutions (x : ℝ) :
  (5.14 * Real.sin (3 * x) + Real.sin (5 * x) = 2 * ((Real.cos (2 * x))^2 - (Real.sin (3 * x))^2)) ↔
  (∃ k : ℤ, x = π / 2 * (2 * k + 1) ∨ x = π / 18 * (4 * k + 1)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solutions_l610_61046


namespace NUMINAMATH_CALUDE_team_selection_l610_61082

theorem team_selection (m : ℕ) : 
  (0 ≤ 14 - m) ∧ 
  (14 - m ≤ 2 * m) ∧ 
  (0 ≤ 5 * m - 11) ∧ 
  (5 * m - 11 ≤ 3 * m) →
  (m = 5) ∧ 
  (Nat.choose 10 9 * Nat.choose 15 14 = 150) := by
sorry


end NUMINAMATH_CALUDE_team_selection_l610_61082


namespace NUMINAMATH_CALUDE_greatest_k_value_l610_61018

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    x^2 + k*x + 7 = 0 ∧ 
    y^2 + k*y + 7 = 0 ∧ 
    |x - y| = Real.sqrt 85) →
  k ≤ Real.sqrt 113 :=
by sorry

end NUMINAMATH_CALUDE_greatest_k_value_l610_61018


namespace NUMINAMATH_CALUDE_chocolate_bars_in_large_box_l610_61058

theorem chocolate_bars_in_large_box :
  let small_boxes : ℕ := 15
  let bars_per_small_box : ℕ := 25
  let total_bars : ℕ := small_boxes * bars_per_small_box
  total_bars = 375 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_in_large_box_l610_61058


namespace NUMINAMATH_CALUDE_order_of_sqrt_differences_l610_61025

/-- Given a = √3 - √2, b = √6 - √5, and c = √7 - √6, prove that a > b > c -/
theorem order_of_sqrt_differences :
  let a := Real.sqrt 3 - Real.sqrt 2
  let b := Real.sqrt 6 - Real.sqrt 5
  let c := Real.sqrt 7 - Real.sqrt 6
  a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_order_of_sqrt_differences_l610_61025


namespace NUMINAMATH_CALUDE_magician_trick_l610_61044

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem magician_trick (numbers : Finset ℕ) (a d : ℕ) :
  numbers = Finset.range 16 →
  a ∈ numbers →
  d ∈ numbers →
  is_even a →
  is_even d →
  ∃ (b c : ℕ), b ∈ numbers ∧ c ∈ numbers ∧
    (b < c ∨ (b > c ∧ a < d)) ∧
    (c < d ∨ (c > d ∧ b < a)) →
  a * d = 120 := by
  sorry

end NUMINAMATH_CALUDE_magician_trick_l610_61044


namespace NUMINAMATH_CALUDE_r_amount_l610_61043

def total_amount : ℕ := 1210
def num_persons : ℕ := 3

def ratio_p_q : Rat := 5 / 4
def ratio_q_r : Rat := 9 / 10

theorem r_amount (p q r : ℕ) (h1 : p + q + r = total_amount) 
  (h2 : (p : ℚ) / q = ratio_p_q) (h3 : (q : ℚ) / r = ratio_q_r) : r = 400 := by
  sorry

end NUMINAMATH_CALUDE_r_amount_l610_61043


namespace NUMINAMATH_CALUDE_gravel_cost_theorem_l610_61081

/-- Calculates the cost of gravelling a path inside a rectangular plot -/
def gravel_cost (length width path_width gravel_cost_per_sqm : ℝ) : ℝ :=
  let total_area := length * width
  let inner_area := (length - 2 * path_width) * (width - 2 * path_width)
  let path_area := total_area - inner_area
  path_area * gravel_cost_per_sqm

/-- Theorem stating the cost of gravelling the path -/
theorem gravel_cost_theorem :
  gravel_cost 100 70 2.5 0.9 = 742.5 := by
sorry

end NUMINAMATH_CALUDE_gravel_cost_theorem_l610_61081


namespace NUMINAMATH_CALUDE_bad_carrots_l610_61055

/-- The number of bad carrots in Carol and her mother's carrot picking scenario -/
theorem bad_carrots (carol_carrots : ℕ) (mother_carrots : ℕ) (good_carrots : ℕ) : 
  carol_carrots = 29 → mother_carrots = 16 → good_carrots = 38 →
  carol_carrots + mother_carrots - good_carrots = 7 := by
  sorry

end NUMINAMATH_CALUDE_bad_carrots_l610_61055


namespace NUMINAMATH_CALUDE_total_votes_is_102000_l610_61009

/-- The number of votes that switched from the first to the second candidate -/
def votes_switched_to_second : ℕ := 16000

/-- The number of votes that switched from the first to the third candidate -/
def votes_switched_to_third : ℕ := 8000

/-- The ratio of votes between the winner and the second place in the second round -/
def winner_ratio : ℕ := 5

/-- Represents the election results -/
structure ElectionResult where
  first_round_votes : ℕ
  second_round_first : ℕ
  second_round_second : ℕ
  second_round_third : ℕ

/-- Checks if the election result satisfies all conditions -/
def is_valid_result (result : ElectionResult) : Prop :=
  -- First round: all candidates have equal votes
  result.first_round_votes * 3 = result.second_round_first + result.second_round_second + result.second_round_third
  -- Vote transfers in second round
  ∧ result.second_round_first = result.first_round_votes - votes_switched_to_second - votes_switched_to_third
  ∧ result.second_round_second = result.first_round_votes + votes_switched_to_second
  ∧ result.second_round_third = result.first_round_votes + votes_switched_to_third
  -- Winner has 5 times as many votes as the second place
  ∧ (result.second_round_second = winner_ratio * result.second_round_first
     ∨ result.second_round_second = winner_ratio * result.second_round_third
     ∨ result.second_round_third = winner_ratio * result.second_round_first
     ∨ result.second_round_third = winner_ratio * result.second_round_second)

/-- The main theorem: prove that the total number of votes is 102000 -/
theorem total_votes_is_102000 :
  ∃ (result : ElectionResult), is_valid_result result ∧ result.first_round_votes * 3 = 102000 :=
sorry

end NUMINAMATH_CALUDE_total_votes_is_102000_l610_61009


namespace NUMINAMATH_CALUDE_no_special_primes_l610_61037

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def digit_swap (n : ℕ) : ℕ := (n % 10) * 10 + (n / 10)

theorem no_special_primes :
  ∀ n : ℕ, 13 ≤ n → n < 100 →
    is_prime n →
    is_prime (digit_swap n) →
    is_prime (digit_sum n) →
    False :=
sorry

end NUMINAMATH_CALUDE_no_special_primes_l610_61037


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l610_61038

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 - x)^(1/3 : ℝ) = 4 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l610_61038


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l610_61084

/-- An isosceles triangle with side lengths 3 and 8 has a perimeter of 19. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  (a = b ∧ (c = 3 ∨ c = 8)) ∨ (a = c ∧ (b = 3 ∨ b = 8)) ∨ (b = c ∧ (a = 3 ∨ a = 8)) →
  a + b > c → b + c > a → a + c > b →
  a + b + c = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l610_61084


namespace NUMINAMATH_CALUDE_sum_of_digits_62_l610_61075

def is_two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def reverse_digits (n : ℕ) : ℕ := (n % 10) * 10 + (n / 10)

theorem sum_of_digits_62 :
  ∀ n : ℕ,
  is_two_digit_number n →
  n = 62 →
  reverse_digits n + 36 = n →
  digit_sum n = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_62_l610_61075


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_tangent_to_circle_l610_61070

theorem hyperbola_asymptotes_tangent_to_circle (m : ℝ) : 
  m > 0 →
  (∀ x y : ℝ, y^2 - x^2 / m^2 = 1) →
  (∀ x y : ℝ, x^2 + y^2 - 4*y + 3 = 0) →
  (∃ x y : ℝ, y = m*x ∧ x^2 + y^2 - 4*y + 3 = 0) →
  m = Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_tangent_to_circle_l610_61070


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l610_61000

/-- Given a quadratic inequality x^2 + bx - a < 0 with solution set {x | -2 < x < 3}, prove that a + b = 5 -/
theorem quadratic_inequality_solution (a b : ℝ) 
  (h : ∀ x, x^2 + b*x - a < 0 ↔ -2 < x ∧ x < 3) : 
  a + b = 5 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l610_61000


namespace NUMINAMATH_CALUDE_intersection_A_B_l610_61026

-- Define set A
def A : Set ℝ := {x | (x + 1) * (4 - x) > 0}

-- Define set B
def B : Set ℝ := {x | 0 < x ∧ x < 9}

-- Define the open interval (0, 4)
def open_interval_0_4 : Set ℝ := {x | 0 < x ∧ x < 4}

-- Theorem statement
theorem intersection_A_B : A ∩ B = open_interval_0_4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l610_61026


namespace NUMINAMATH_CALUDE_min_l_shapes_5x5_grid_l610_61057

/-- Represents a 5x5 grid --/
def Grid := Fin 5 → Fin 5 → Bool

/-- An L-shaped figure made of 3 cells --/
structure LShape where
  x : Fin 5
  y : Fin 5
  orientation : Fin 4

/-- Check if an L-shape is within the grid bounds --/
def LShape.isValid (l : LShape) : Bool :=
  match l.orientation with
  | 0 => l.x < 4 ∧ l.y < 4
  | 1 => l.x > 0 ∧ l.y < 4
  | 2 => l.x < 4 ∧ l.y > 0
  | 3 => l.x > 0 ∧ l.y > 0

/-- Check if two L-shapes overlap --/
def LShape.overlaps (l1 l2 : LShape) : Bool :=
  sorry

/-- Check if a set of L-shapes is valid (non-overlapping and within bounds) --/
def isValidPlacement (shapes : List LShape) : Bool :=
  sorry

/-- Check if no more L-shapes can be added to a given set of shapes --/
def isMaximalPlacement (shapes : List LShape) : Bool :=
  sorry

/-- The main theorem --/
theorem min_l_shapes_5x5_grid :
  ∃ (shapes : List LShape),
    shapes.length = 4 ∧
    isValidPlacement shapes ∧
    isMaximalPlacement shapes ∧
    ∀ (otherShapes : List LShape),
      isValidPlacement otherShapes ∧ isMaximalPlacement otherShapes →
      otherShapes.length ≥ 4 :=
  sorry

end NUMINAMATH_CALUDE_min_l_shapes_5x5_grid_l610_61057


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l610_61035

theorem polynomial_division_theorem (x : ℝ) : 
  (4 * x^3 + x^2 + 2 * x + 3) * (3 * x - 2) + 11 = 
  12 * x^4 - 9 * x^3 + 6 * x^2 + 11 * x - 3 := by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l610_61035


namespace NUMINAMATH_CALUDE_prism_volume_l610_61019

/-- The volume of a right rectangular prism with given face areas -/
theorem prism_volume (x y z : ℝ) 
  (h₁ : x * y = 20)  -- side face area
  (h₂ : y * z = 12)  -- front face area
  (h₃ : x * z = 8)   -- bottom face area
  : x * y * z = 8 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l610_61019


namespace NUMINAMATH_CALUDE_exactly_five_cheaper_values_l610_61014

/-- The cost function for books, including the discount -/
def C (n : ℕ) : ℚ :=
  let base := if n ≤ 20 then 15 * n
              else if n ≤ 40 then 14 * n - 5
              else 13 * n
  base - 10 * (n / 10 : ℚ)

/-- Predicate for when it's cheaper to buy n+1 books than n books -/
def cheaper_to_buy_more (n : ℕ) : Prop :=
  C (n + 1) < C n

/-- The main theorem stating there are exactly 5 values where it's cheaper to buy more -/
theorem exactly_five_cheaper_values :
  (∃ (s : Finset ℕ), s.card = 5 ∧ ∀ n, n ∈ s ↔ cheaper_to_buy_more n) :=
sorry

end NUMINAMATH_CALUDE_exactly_five_cheaper_values_l610_61014


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l610_61039

theorem quadratic_two_distinct_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 - 3*x₁ + 0 = 0 ∧ x₂^2 - 3*x₂ + 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l610_61039


namespace NUMINAMATH_CALUDE_combined_discount_rate_l610_61050

/-- Calculate the combined rate of discount for three items -/
theorem combined_discount_rate
  (bag_marked : ℝ) (shoes_marked : ℝ) (hat_marked : ℝ)
  (bag_discounted : ℝ) (shoes_discounted : ℝ) (hat_discounted : ℝ)
  (h_bag : bag_marked = 150 ∧ bag_discounted = 120)
  (h_shoes : shoes_marked = 100 ∧ shoes_discounted = 80)
  (h_hat : hat_marked = 50 ∧ hat_discounted = 40) :
  let total_marked := bag_marked + shoes_marked + hat_marked
  let total_discounted := bag_discounted + shoes_discounted + hat_discounted
  let discount_rate := (total_marked - total_discounted) / total_marked
  discount_rate = 0.2 := by
sorry

end NUMINAMATH_CALUDE_combined_discount_rate_l610_61050


namespace NUMINAMATH_CALUDE_zero_power_is_zero_l610_61052

theorem zero_power_is_zero (n : ℕ) : 0^n = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_power_is_zero_l610_61052


namespace NUMINAMATH_CALUDE_tv_show_average_episodes_l610_61021

theorem tv_show_average_episodes (total_years : ℕ) (seasons_15 : ℕ) (seasons_20 : ℕ) (seasons_12 : ℕ)
  (h1 : total_years = 14)
  (h2 : seasons_15 = 8)
  (h3 : seasons_20 = 4)
  (h4 : seasons_12 = 2) :
  (seasons_15 * 15 + seasons_20 * 20 + seasons_12 * 12) / total_years = 16 := by
  sorry

end NUMINAMATH_CALUDE_tv_show_average_episodes_l610_61021


namespace NUMINAMATH_CALUDE_train_length_l610_61076

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 30 → time = 24 → ∃ length : ℝ, abs (length - 199.92) < 0.01 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l610_61076


namespace NUMINAMATH_CALUDE_largest_quantity_l610_61061

def X : ℚ := 2010 / 2009 + 2010 / 2011
def Y : ℚ := 2010 / 2011 + 2012 / 2011
def Z : ℚ := 2011 / 2010 + 2011 / 2012

theorem largest_quantity : X > Y ∧ X > Z := by
  sorry

end NUMINAMATH_CALUDE_largest_quantity_l610_61061


namespace NUMINAMATH_CALUDE_max_values_for_constrained_expressions_l610_61067

theorem max_values_for_constrained_expressions (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 1) :
  (∃ (max_ab : ℝ), ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 → x * y ≤ max_ab) ∧
  (∃ (max_sqrt : ℝ), ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 → Real.sqrt x + Real.sqrt y ≤ max_sqrt) ∧
  (∀ (M : ℝ), ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 ∧ x^2 + y^2 > M) ∧
  (∀ (M : ℝ), ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 ∧ 1/x + 4/y > M) :=
by sorry

end NUMINAMATH_CALUDE_max_values_for_constrained_expressions_l610_61067


namespace NUMINAMATH_CALUDE_triangle_problem_l610_61051

theorem triangle_problem (A B C : Real) (a b c : Real) (D : Real × Real) :
  -- Given conditions
  0 < A ∧ A < π/2 ∧
  0 < B ∧ B < π/2 ∧
  0 < C ∧ C < π/2 ∧
  A + B + C = π ∧
  a = 3 ∧
  b = Real.sqrt 13 ∧
  a * Real.sin (2 * B) = b * Real.sin A ∧
  -- Definition of point D
  D = ((1/3) * (Real.cos A, Real.sin A) + (2/3) * (Real.cos C, Real.sin C)) →
  -- Conclusions
  B = π/3 ∧
  Real.sqrt ((D.1 - Real.cos B)^2 + (D.2 - Real.sin B)^2) = (2 * Real.sqrt 19) / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l610_61051


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l610_61088

theorem fraction_sum_equality (p q r : ℝ) 
  (h : p / (30 - p) + q / (70 - q) + r / (50 - r) = 8) :
  6 / (30 - p) + 14 / (70 - q) + 10 / (50 - r) = 2.2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l610_61088


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l610_61028

/-- An arithmetic sequence with first term a₁ > 0 and common ratio q -/
structure ArithmeticSequence where
  a₁ : ℝ
  q : ℝ
  h₁ : a₁ > 0

/-- Sum of first n terms of an arithmetic sequence -/
def S (as : ArithmeticSequence) (n : ℕ) : ℝ := sorry

/-- Statement: q > 1 is sufficient but not necessary for S₃ + S₅ > 2S₄ -/
theorem sufficient_not_necessary (as : ArithmeticSequence) :
  (∀ as, as.q > 1 → S as 3 + S as 5 > 2 * S as 4) ∧
  ¬(∀ as, S as 3 + S as 5 > 2 * S as 4 → as.q > 1) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l610_61028


namespace NUMINAMATH_CALUDE_eight_cos_squared_25_minus_tan_40_minus_4_equals_sqrt_3_l610_61091

theorem eight_cos_squared_25_minus_tan_40_minus_4_equals_sqrt_3 :
  8 * (Real.cos (25 * π / 180))^2 - Real.tan (40 * π / 180) - 4 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_eight_cos_squared_25_minus_tan_40_minus_4_equals_sqrt_3_l610_61091


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l610_61006

-- Define the inequality function
def f (m : ℝ) (x : ℝ) : Prop := m * x^2 + (2 * m - 1) * x - 2 > 0

-- Define the solution set for each case
def solution_set (m : ℝ) : Set ℝ :=
  if m = 0 then { x | x < -2 }
  else if m > 0 then { x | x < -2 ∨ x > 1/m }
  else if -1/2 < m ∧ m < 0 then { x | 1/m < x ∧ x < -2 }
  else if m = -1/2 then ∅
  else { x | -2 < x ∧ x < 1/m }

-- State the theorem
theorem inequality_solution_sets (m : ℝ) :
  { x : ℝ | f m x } = solution_set m := by sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l610_61006


namespace NUMINAMATH_CALUDE_isabella_hair_growth_l610_61083

def monthly_growth : List Float := [0.5, 1, 0.75, 1.25, 1, 0.5, 1.5, 1, 0.25, 1.5, 1.25, 0.75]

theorem isabella_hair_growth :
  monthly_growth.sum = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_isabella_hair_growth_l610_61083


namespace NUMINAMATH_CALUDE_no_integer_solution_l610_61011

theorem no_integer_solution : ¬ ∃ (m n : ℤ), 5 * m^2 - 6 * m * n + 7 * n^2 = 2005 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l610_61011


namespace NUMINAMATH_CALUDE_Q_zeros_count_l610_61092

noncomputable def Q (x : ℝ) : ℂ :=
  2 + Complex.exp (Complex.I * x) - 2 * Complex.exp (2 * Complex.I * x) + Complex.exp (3 * Complex.I * x)

theorem Q_zeros_count : ∃! (s : Finset ℝ), s.card = 2 ∧ (∀ x ∈ s, 0 ≤ x ∧ x < 4 * Real.pi ∧ Q x = 0) ∧ (∀ x, 0 ≤ x → x < 4 * Real.pi → Q x = 0 → x ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_Q_zeros_count_l610_61092


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l610_61066

/-- The number of diagonals in a regular polygon with n sides -/
def numDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular nine-sided polygon contains 27 diagonals -/
theorem nonagon_diagonals : numDiagonals 9 = 27 := by sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l610_61066


namespace NUMINAMATH_CALUDE_fourteenth_root_of_unity_l610_61074

theorem fourteenth_root_of_unity : 
  ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 13 ∧ 
  (Complex.tan (Real.pi / 7) + Complex.I) / (Complex.tan (Real.pi / 7) - Complex.I) = 
  Complex.exp (Complex.I * (2 * Real.pi * n / 14)) := by
  sorry

end NUMINAMATH_CALUDE_fourteenth_root_of_unity_l610_61074


namespace NUMINAMATH_CALUDE_shaded_region_area_l610_61073

/-- Given a shaded region consisting of congruent squares, proves that the total area is 40 cm² --/
theorem shaded_region_area (n : ℕ) (d : ℝ) (A : ℝ) :
  n = 20 →  -- Total number of congruent squares
  d = 8 →   -- Diagonal of the square formed by 16 smaller squares
  A = d^2 / 2 →  -- Area of the square formed by 16 smaller squares
  A / 16 * n = 40 :=
by sorry

end NUMINAMATH_CALUDE_shaded_region_area_l610_61073


namespace NUMINAMATH_CALUDE_find_N_l610_61023

theorem find_N : ∃ N : ℝ, (0.2 * N = 0.6 * 2500) ∧ (N = 7500) := by
  sorry

end NUMINAMATH_CALUDE_find_N_l610_61023


namespace NUMINAMATH_CALUDE_linear_combination_harmonic_l610_61054

/-- A function is harmonic if its value at each point is the average of its values at the four neighboring points. -/
def IsHarmonic (f : ℤ → ℤ → ℝ) : Prop :=
  ∀ x y : ℤ, f x y = (f (x + 1) y + f (x - 1) y + f x (y + 1) + f x (y - 1)) / 4

/-- The theorem states that a linear combination of two harmonic functions is also harmonic. -/
theorem linear_combination_harmonic
    (f g : ℤ → ℤ → ℝ) (hf : IsHarmonic f) (hg : IsHarmonic g) (a b : ℝ) :
    IsHarmonic (fun x y ↦ a * f x y + b * g x y) := by
  sorry

end NUMINAMATH_CALUDE_linear_combination_harmonic_l610_61054


namespace NUMINAMATH_CALUDE_sum_a_d_equals_negative_one_l610_61053

theorem sum_a_d_equals_negative_one
  (a b c d : ℤ)
  (eq1 : a + b = 11)
  (eq2 : b + c = 9)
  (eq3 : c + d = 3) :
  a + d = -1 := by sorry

end NUMINAMATH_CALUDE_sum_a_d_equals_negative_one_l610_61053


namespace NUMINAMATH_CALUDE_base9_726_to_base3_l610_61002

/-- Converts a base-9 digit to its two-digit base-3 representation -/
def base9ToBase3Digit (d : Nat) : Nat × Nat :=
  ((d / 3), (d % 3))

/-- Converts a base-9 number to its base-3 representation -/
def base9ToBase3 (n : Nat) : List Nat :=
  let digits := n.digits 9
  List.join (digits.map (fun d => let (a, b) := base9ToBase3Digit d; [a, b]))

theorem base9_726_to_base3 :
  base9ToBase3 726 = [2, 1, 0, 2, 2, 0] :=
sorry

end NUMINAMATH_CALUDE_base9_726_to_base3_l610_61002


namespace NUMINAMATH_CALUDE_window_offer_savings_l610_61030

/-- Represents the store's window offer -/
structure WindowOffer where
  regularPrice : ℕ
  buyQuantity : ℕ
  freeQuantity : ℕ

/-- Calculates the cost for a given number of windows under the offer -/
def costUnderOffer (offer : WindowOffer) (windowsNeeded : ℕ) : ℕ :=
  let fullSets := windowsNeeded / (offer.buyQuantity + offer.freeQuantity)
  let remainder := windowsNeeded % (offer.buyQuantity + offer.freeQuantity)
  (fullSets * offer.buyQuantity + min remainder offer.buyQuantity) * offer.regularPrice

/-- Calculates the savings for a given number of windows under the offer -/
def savingsUnderOffer (offer : WindowOffer) (windowsNeeded : ℕ) : ℕ :=
  windowsNeeded * offer.regularPrice - costUnderOffer offer windowsNeeded

/-- The main theorem to prove -/
theorem window_offer_savings (offer : WindowOffer) 
  (h1 : offer.regularPrice = 100)
  (h2 : offer.buyQuantity = 8)
  (h3 : offer.freeQuantity = 2)
  (dave_windows : ℕ) (doug_windows : ℕ)
  (h4 : dave_windows = 9)
  (h5 : doug_windows = 10) :
  savingsUnderOffer offer (dave_windows + doug_windows) - 
  (savingsUnderOffer offer dave_windows + savingsUnderOffer offer doug_windows) = 100 := by
  sorry

end NUMINAMATH_CALUDE_window_offer_savings_l610_61030


namespace NUMINAMATH_CALUDE_inversion_property_l610_61032

/-- Circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Point in 2D plane -/
def Point := ℝ × ℝ

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Inversion of a point with respect to a circle -/
def inversion (c : Circle) (p : Point) : Point := sorry

/-- Theorem: Inversion property -/
theorem inversion_property (c : Circle) (p p' : Point) : 
  p' = inversion c p → 
  distance c.center p * distance c.center p' = c.radius ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_inversion_property_l610_61032


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_range_l610_61071

/-- A cubic function with three distinct real roots -/
structure CubicFunction where
  m : ℝ
  n : ℝ
  p : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  has_three_roots : a ≠ b ∧ b ≠ c ∧ a ≠ c
  is_root_a : a^3 + m*a^2 + n*a + p = 0
  is_root_b : b^3 + m*b^2 + n*b + p = 0
  is_root_c : c^3 + m*c^2 + n*c + p = 0
  neg_one_eq_two : ((-1)^3 + m*(-1)^2 + n*(-1) + p) = (2^3 + m*2^2 + n*2 + p)
  one_eq_four : (1^3 + m*1^2 + n*1 + p) = (4^3 + m*4^2 + n*4 + p)
  neg_one_neg : ((-1)^3 + m*(-1)^2 + n*(-1) + p) < 0
  one_pos : (1^3 + m*1^2 + n*1 + p) > 0

/-- The main theorem stating the range of the sum of reciprocals of roots -/
theorem sum_of_reciprocals_range (f : CubicFunction) :
  -(3/4) < (1/f.a + 1/f.b + 1/f.c) ∧ (1/f.a + 1/f.b + 1/f.c) < -(3/14) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_range_l610_61071


namespace NUMINAMATH_CALUDE_division_problem_l610_61078

theorem division_problem (x : ℝ) : 75 / x = 1500 → x = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l610_61078


namespace NUMINAMATH_CALUDE_max_sin_a_is_one_l610_61064

theorem max_sin_a_is_one (a b : ℝ) (h : Real.sin (a + b) = Real.sin a + Real.sin b) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (x : ℝ), Real.sin x ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_sin_a_is_one_l610_61064
