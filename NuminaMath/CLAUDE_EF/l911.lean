import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l911_91177

-- Define constants a and b
noncomputable def a : ℝ := (10/11) * Real.exp (1/11)
noncomputable def b : ℝ := 11 * Real.log 1.1

-- Theorem statement
theorem inequality_proof : 1 < a * b ∧ a * b < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l911_91177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_leg_speed_l911_91116

/-- Given a car journey with two legs, this theorem proves the speed of the second leg. -/
theorem second_leg_speed
  (total_distance : ℝ)
  (first_leg_distance : ℝ)
  (first_leg_speed : ℝ)
  (average_speed : ℝ)
  (h1 : total_distance = 320)
  (h2 : first_leg_distance = 160)
  (h3 : first_leg_speed = 64)
  (h4 : average_speed = 71.11111111111111)
  (h5 : first_leg_distance * 2 = total_distance) :
  let second_leg_distance := total_distance - first_leg_distance
  let total_time := total_distance / average_speed
  let first_leg_time := first_leg_distance / first_leg_speed
  let second_leg_time := total_time - first_leg_time
  second_leg_distance / second_leg_time = 80 := by
  sorry

-- Use #eval to check the result
#eval Float.round ((320 - 160) / ((320 / 71.11111111111111) - (160 / 64)))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_leg_speed_l911_91116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_of_negative_three_equals_fourteen_l911_91188

-- Define the functions u and p
def u (x : ℝ) : ℝ := 4 * x + 5

noncomputable def p (x : ℝ) : ℝ := 
  let y := (x - 5) / 4
  y^2 - 2 * y + 6

-- State the theorem
theorem p_of_negative_three_equals_fourteen : p (-3) = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_of_negative_three_equals_fourteen_l911_91188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_m_inclination_l911_91119

-- Define the parallel lines
def l1 (x y : ℝ) : Prop := x - y + 1 = 0
def l2 (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the line m (we don't know its equation, but we know it exists)
def m : Set (ℝ × ℝ) := sorry

-- Define the length of the segment cut by m between l1 and l2
noncomputable def segment_length : ℝ := 2 * Real.sqrt 2

-- Define the inclination angle of a line
noncomputable def inclination_angle (line : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem line_m_inclination :
  (∀ x y, l1 x y ↔ x - y + 1 = 0) →
  (∀ x y, l2 x y ↔ x - y - 1 = 0) →
  (∃ p q : ℝ × ℝ, p ∈ m ∧ q ∈ m ∧ p ≠ q) →
  (∃ p q : ℝ × ℝ, p ∈ m ∧ (l1 p.1 p.2 ∨ l2 p.1 p.2) ∧
                  q ∈ m ∧ (l1 q.1 q.2 ∨ l2 q.1 q.2) ∧
                  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = segment_length) →
  inclination_angle m = 15 ∨ inclination_angle m = 75 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_m_inclination_l911_91119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_constraint_l911_91151

/-- The curve function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * x^2 + a * x + 3

/-- The derivative of f(x) -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x^2 - x + a

theorem tangent_line_constraint (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ f' a x₁ = 3 ∧ f' a x₂ = 3) →
  3 < a ∧ a < 13/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_constraint_l911_91151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oranges_for_150_cents_profit_l911_91155

/-- The minimum number of oranges needed to be sold to make a profit of at least 150 cents -/
def min_oranges_for_profit (cost_per_4 : ℚ) (sell_per_6 : ℚ) (target_profit : ℚ) : ℕ :=
  let cost_per_orange := cost_per_4 / 4
  let sell_per_orange := sell_per_6 / 6
  let profit_per_orange := sell_per_orange - cost_per_orange
  (target_profit / profit_per_orange).ceil.toNat

/-- Theorem stating the minimum number of oranges needed to be sold for the given problem -/
theorem oranges_for_150_cents_profit :
  min_oranges_for_profit 15 25 150 = 358 :=
by sorry

#eval min_oranges_for_profit 15 25 150

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oranges_for_150_cents_profit_l911_91155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_angles_l911_91134

open Real

theorem triangle_abc_angles (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  (Real.sqrt 2 * a = 2 * b * Real.sin A → (B = π/4 ∨ B = 3*π/4)) ∧
  (a^2 + b^2 + Real.sqrt 2*a*b = c^2 → C = 3*π/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_angles_l911_91134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monkey_peach_theorem_l911_91126

/-- Represents a pile of peaches -/
structure PeachPile where
  count : ℕ

/-- Represents the state of the peach piles -/
structure PeachState where
  piles : List PeachPile
  eaten : ℕ

/-- Defines a valid move by the monkey -/
def validMove (f t : PeachPile) : Prop :=
  f.count > t.count ∧ f.count - t.count ≤ 5

/-- Defines the result of a move -/
def moveResult (f t : PeachPile) : PeachState :=
  { piles := [{ count := f.count - (f.count - t.count) },
              { count := t.count + (f.count - t.count) - 1 }],
    eaten := 1 }

/-- Checks if all piles have equal counts -/
def allEqual (state : PeachState) : Prop :=
  ∀ p q, p ∈ state.piles → q ∈ state.piles → p.count = q.count

/-- The main theorem to be proved -/
theorem monkey_peach_theorem (initialState : PeachState) :
  initialState.piles = [{ count := 93 }, { count := 70 }, { count := 63 }] →
  ∃ (finalState : PeachState),
    allEqual finalState ∧
    finalState.eaten = 4 ∧
    (∀ otherState : PeachState, allEqual otherState →
      otherState.eaten ≥ finalState.eaten) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monkey_peach_theorem_l911_91126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_volume_is_864_l911_91150

/-- Represents a right triangle formed by halving a square --/
structure HalfSquareTriangle where
  side_length : ℝ
  is_positive : side_length > 0

/-- Represents the polyhedron formed by folding six HalfSquareTriangles --/
structure Polyhedron where
  triangle : HalfSquareTriangle
  num_triangles : ℕ
  is_six_triangles : num_triangles = 6

/-- Calculates the volume of the polyhedron --/
noncomputable def volume (p : Polyhedron) : ℝ :=
  (1 / 2) * p.triangle.side_length ^ 3

/-- Theorem: The volume of the polyhedron is 864 cm³ when the square side length is 12 cm --/
theorem polyhedron_volume_is_864 (p : Polyhedron) 
    (h : p.triangle.side_length = 12) : volume p = 864 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_volume_is_864_l911_91150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_with_distinct_digits_l911_91159

def has_distinct_digits (n : ℕ) : Prop :=
  ∀ d₁ d₂, d₁ ∈ n.digits 10 → d₂ ∈ n.digits 10 → d₁ = d₂

theorem sum_with_distinct_digits :
  ∃ (a b c d e : ℕ),
    a + b + c + d + e = 2017 ∧
    has_distinct_digits a ∧
    has_distinct_digits b ∧
    has_distinct_digits c ∧
    has_distinct_digits d ∧
    has_distinct_digits e ∧
    (∀ (x y : ℕ), x ∈ [a, b, c, d, e] → y ∈ [a, b, c, d, e] → x ≠ y →
      (∀ dx dy, dx ∈ x.digits 10 → dy ∈ y.digits 10 → dx ≠ dy)) :=
by sorry

#check sum_with_distinct_digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_with_distinct_digits_l911_91159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_extensions_l911_91166

def is_valid_extension (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n ≤ 9999 ∧  -- 4-digit number
  n % 2 = 0 ∧  -- even number
  ∃ (a b c d : ℕ), n = 1000 * a + 100 * b + 10 * c + d ∧
    ({a, b, c, d} : Finset ℕ) = {1, 2, 3, 8}  -- uses all digits 1, 2, 3, 8

theorem max_valid_extensions :
  (∃ (s : Finset ℕ), (∀ n ∈ s, is_valid_extension n) ∧ s.card = 12) ∧
  (∀ (t : Finset ℕ), (∀ n ∈ t, is_valid_extension n) → t.card ≤ 12) :=
by sorry

#check max_valid_extensions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_extensions_l911_91166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_is_three_l911_91105

/-- The limit of (6t^3 - 9t + 1) / (2t^3 - 3t) as t approaches infinity is 3 -/
theorem limit_is_three :
  ∀ ε > 0, ∃ N : ℝ, ∀ t : ℝ, t ≥ N → 
    |((6 * t^3 - 9 * t + 1) / (2 * t^3 - 3 * t)) - 3| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_is_three_l911_91105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_101_l911_91112

def a (n : ℕ) : ℕ := n^2 + 1

theorem tenth_term_is_101 : a 10 = 101 := by
  rw [a]
  norm_num

#eval a 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_101_l911_91112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_after_9_minutes_l911_91162

/-- Represents the robot's position on a 2D grid -/
structure Position where
  x : ℤ
  y : ℤ
deriving Repr

/-- Represents the possible directions the robot can face -/
inductive Direction
  | North
  | East
  | South
  | West
deriving Repr

/-- Represents a turn the robot can make -/
inductive Turn
  | Left
  | Right
  | None
deriving Repr

/-- Calculates the new direction after a turn -/
def newDirection (d : Direction) (t : Turn) : Direction :=
  match d, t with
  | Direction.North, Turn.Left => Direction.West
  | Direction.North, Turn.Right => Direction.East
  | Direction.East, Turn.Left => Direction.North
  | Direction.East, Turn.Right => Direction.South
  | Direction.South, Turn.Left => Direction.East
  | Direction.South, Turn.Right => Direction.West
  | Direction.West, Turn.Left => Direction.South
  | Direction.West, Turn.Right => Direction.North
  | _, Turn.None => d

/-- Calculates the new position after moving in a given direction -/
def move (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.North => ⟨p.x, p.y + 10⟩
  | Direction.East => ⟨p.x + 10, p.y⟩
  | Direction.South => ⟨p.x, p.y - 10⟩
  | Direction.West => ⟨p.x - 10, p.y⟩

/-- Calculates the distance between two positions -/
noncomputable def distance (p1 p2 : Position) : ℝ :=
  Real.sqrt (((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2) : ℝ)

/-- Represents a sequence of turns the robot makes -/
def TurnSequence := List Turn

/-- Simulates the robot's movement given a sequence of turns -/
def simulateMovement (turns : TurnSequence) : Position :=
  let initialPos : Position := ⟨10, 0⟩
  let initialDir : Direction := Direction.East
  let (finalPos, _) := turns.foldl
    (fun (acc : Position × Direction) t =>
      let (pos, dir) := acc
      let newDir := newDirection dir t
      (move pos newDir, newDir))
    (initialPos, initialDir)
  finalPos

/-- The main theorem: The minimum distance from the starting point after 9 minutes is 10 meters -/
theorem min_distance_after_9_minutes :
  ∀ (turns : TurnSequence),
    turns.length = 8 →
    distance (simulateMovement turns) ⟨0, 0⟩ ≥ 10 := by
  sorry

#eval simulateMovement [Turn.Left, Turn.Left, Turn.Left, Turn.Left, Turn.Left, Turn.Left, Turn.Left, Turn.Left]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_after_9_minutes_l911_91162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_program_profit_l911_91127

/-- Calculates the desired profit for a college football program -/
theorem football_program_profit
  (production_cost : ℚ)
  (ad_revenue : ℚ)
  (copies_sold : ℕ)
  (selling_price : ℚ)
  (h1 : production_cost = 7/10)
  (h2 : ad_revenue = 15000)
  (h3 : copies_sold = 35000)
  (h4 : selling_price = 1/2) :
  (copies_sold : ℚ) * selling_price + ad_revenue - (copies_sold : ℚ) * production_cost = 8000 := by
  sorry

-- Remove the #eval line as it's causing issues
-- #eval football_program_profit (7/10) 15000 35000 (1/2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_program_profit_l911_91127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_n_l911_91186

open BigOperators

theorem existence_of_n : ∃ n : ℕ, ∀ x y : ℝ, ∃ a : Fin n → ℝ, 
  x = ∑ i, a i ∧ y = ∑ i, (1 / (a i)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_n_l911_91186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coins_processable_l911_91195

/-- The maximum number of coins that can be processed in n weighings -/
def f (n : ℕ) : ℕ := 2 * n^2 + 1

/-- A weighing strategy for identifying the counterfeit coin -/
def strategy_identifies_counterfeit (n m : ℕ) : Prop := sorry

/-- A strategy uses each coin at most twice -/
def strategy_uses_each_coin_at_most_twice (strategy : ℕ → ℕ → Prop) : Prop := sorry

/-- The theorem stating the maximum number of coins that can be processed -/
theorem max_coins_processable (n : ℕ) : 
  f n = 2 * n^2 + 1 ∧
  ∀ m : ℕ, m > f n → ¬ (∃ (strategy : ℕ → ℕ → Prop), 
    strategy_identifies_counterfeit n m ∧
    strategy_uses_each_coin_at_most_twice strategy) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coins_processable_l911_91195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_length_is_24_emily_steps_l911_91130

/-- The length of Emily's step -/
noncomputable def emily_step : ℝ := 1

/-- The length of the ship's movement during one of Emily's steps -/
noncomputable def ship_step : ℝ := emily_step / 5

/-- The number of steps Emily takes when walking in the same direction as the ship -/
def steps_same_direction : ℕ := 150

/-- The number of steps Emily takes when walking against the ship's direction -/
def steps_opposite_direction : ℕ := 90

/-- The length of the ship in terms of Emily's steps -/
noncomputable def ship_length : ℝ := 24 * emily_step

theorem ship_length_is_24_emily_steps :
  ship_length = steps_same_direction * emily_step - steps_same_direction * ship_step ∧
  ship_length = steps_opposite_direction * emily_step + steps_opposite_direction * ship_step :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_length_is_24_emily_steps_l911_91130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dvd_to_cd_ratio_is_8_to_5_l911_91118

def total_items : ℕ := 273
def dvds_sold : ℕ := 168

def cds_sold : ℕ := total_items - dvds_sold

def dvd_to_cd_ratio : ℚ :=
  (dvds_sold : ℚ) / (cds_sold : ℚ)

theorem dvd_to_cd_ratio_is_8_to_5 : dvd_to_cd_ratio = 8 / 5 := by
  -- Expand the definition of dvd_to_cd_ratio
  unfold dvd_to_cd_ratio
  -- Simplify the fraction
  simp [total_items, dvds_sold, cds_sold]
  -- The proof is completed
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dvd_to_cd_ratio_is_8_to_5_l911_91118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weekly_social_media_usage_l911_91191

/-- Charlotte's daily phone usage in hours -/
noncomputable def daily_phone_usage : ℝ := 16

/-- The fraction of phone time spent on social media -/
noncomputable def social_media_fraction : ℝ := 1 / 2

/-- Number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: Charlotte's weekly social media usage is 56 hours -/
theorem weekly_social_media_usage :
  (daily_phone_usage * social_media_fraction * (days_in_week : ℝ)) = 56 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_weekly_social_media_usage_l911_91191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_Q_to_EH_l911_91160

-- Define the square EFGH
noncomputable def E : ℝ × ℝ := (0, 6)
noncomputable def F : ℝ × ℝ := (6, 6)
noncomputable def G : ℝ × ℝ := (6, 0)
noncomputable def H : ℝ × ℝ := (0, 0)

-- Define N as the midpoint of GH
noncomputable def N : ℝ × ℝ := (3, 0)

-- Define the circles
def circle_N (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9
def circle_E (x y : ℝ) : Prop := x^2 + (y - 6)^2 = 36

-- Define Q as the intersection point (other than H)
noncomputable def Q : ℝ × ℝ := (24/5, 12/5)

-- Define the line EH
def line_EH (y : ℝ) : Prop := y = 6

-- Theorem statement
theorem distance_Q_to_EH : 
  abs (6 - Q.2) = 18/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_Q_to_EH_l911_91160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_equilibrium_angle_l911_91103

noncomputable def angle_of_equilibrium (γ : ℝ) (h : γ > 0) : ℝ := 
  Real.arctan (0.6823)

theorem cube_equilibrium_angle (γ : ℝ) (h : γ > 0) :
  let φ := angle_of_equilibrium γ h
  (Real.tan φ)^3 + Real.tan φ - 1 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_equilibrium_angle_l911_91103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l911_91111

noncomputable def f (a b c d : ℝ) : ℝ := 
  Real.sqrt (5*a + 9) + Real.sqrt (5*b + 9) + Real.sqrt (5*c + 9) + Real.sqrt (5*d + 9)

theorem f_bounds (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) (hsum : a + b + c + d = 32) :
  22 ≤ f a b c d ∧ f a b c d ≤ 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l911_91111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_x_minus_2_defined_l911_91175

theorem sqrt_x_minus_2_defined (x : ℝ) : x ≥ 2 ↔ ∃ y : ℝ, y^2 = x - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_x_minus_2_defined_l911_91175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_approximation_l911_91196

theorem expression_approximation : 
  let x : ℝ := 0.76
  let numerator : ℝ := x * x * x - 0.008
  let denominator : ℝ := x * x + x * 0.2 + 0.04
  ∃ ε : ℝ, ε > 0 ∧ abs ((numerator / denominator) - 0.560) < ε ∧ ε < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_approximation_l911_91196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_partition_size_l911_91106

theorem smallest_partition_size (r : ℕ) (hr : r ≥ 1) :
  ∃ (h : ℕ),
    h ≥ 1 ∧
    (∀ (partition : Fin h → Fin r),
      ∃ (a x y : ℕ),
        x ≥ 1 ∧ y ≥ x ∧
        x < h ∧ y < h ∧ (a + x) < h ∧ (a + y) < h ∧ (a + x + y) < h ∧
        partition (⟨a + x, by sorry⟩) = partition (⟨a + y, by sorry⟩) ∧
        partition (⟨a + x, by sorry⟩) = partition (⟨a + x + y, by sorry⟩)) ∧
    (∀ (h' : ℕ),
      h' < h →
      ∃ (partition : Fin h' → Fin r),
        ∀ (a x y : ℕ),
          x ≥ 1 → y ≥ x → x < h' ∧ y < h' ∧ (a + x) < h' ∧ (a + y) < h' ∧ (a + x + y) < h' →
          (partition (⟨a + x, by sorry⟩) = partition (⟨a + y, by sorry⟩) ∧
           partition (⟨a + x, by sorry⟩) = partition (⟨a + x + y, by sorry⟩) →
           False)) ∧
    h = 2 * r :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_partition_size_l911_91106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_cars_return_to_start_l911_91157

/-- Represents the state of a car on the circular track -/
structure Car where
  position : ℝ  -- Position on the circular track (0 ≤ position < 1)
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- Represents the state of the race at a given time -/
def RaceState := List Car

/-- The number of cars in the race -/
def n : ℕ := sorry

/-- Initial state of the race -/
def initialState : RaceState := sorry

/-- Axiom: All cars have distinct initial positions -/
axiom distinct_initial_positions :
  ∀ i j, i ≠ j → (initialState.get? i).map Car.position ≠ (initialState.get? j).map Car.position

/-- Axiom: The number of cars in the initial state is n -/
axiom initial_state_size : initialState.length = n

/-- Function that evolves the race state over time -/
noncomputable def evolveRace (t : ℝ) : RaceState → RaceState := sorry

/-- Theorem: There exists a time when all cars return to their initial positions -/
theorem all_cars_return_to_start :
  ∃ t : ℝ, evolveRace t initialState = initialState := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_cars_return_to_start_l911_91157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_lids_per_box_l911_91144

theorem can_lids_per_box 
  (initial_lids : ℕ) 
  (num_boxes : ℕ) 
  (total_lids : ℕ) 
  (h1 : initial_lids = 14)
  (h2 : num_boxes = 3)
  (h3 : total_lids = 53) :
  (total_lids - initial_lids) / num_boxes = 13 := by
  sorry

#check can_lids_per_box

end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_lids_per_box_l911_91144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rainy_days_probability_l911_91141

noncomputable def june_days : ℕ := 30
noncomputable def daily_rain_prob : ℝ := 1/5

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1-p)^(n-k)

noncomputable def at_most_three_rainy_days : ℝ :=
  (binomial_probability june_days 0 daily_rain_prob) +
  (binomial_probability june_days 1 daily_rain_prob) +
  (binomial_probability june_days 2 daily_rain_prob) +
  (binomial_probability june_days 3 daily_rain_prob)

theorem rainy_days_probability :
  |at_most_three_rainy_days - 0.855| < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rainy_days_probability_l911_91141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_triangle_area_l911_91102

theorem largest_inscribed_triangle_area (r : ℝ) (h : r = 8) :
  ∃ (A : ℝ), A = r^2 * 2 ∧ 
  ∀ (B : ℝ), B ≤ A → 
  ∃ (a b c : ℝ), a^2 + b^2 = c^2 ∧ c = 2*r ∧ B = (a * b) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_triangle_area_l911_91102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_range_with_acceptable_deviation_l911_91139

/-- The exact conversion formula from Celsius to Fahrenheit -/
noncomputable def exact_formula (C : ℝ) : ℝ := (9/5) * C + 32

/-- The approximate conversion formula from Celsius to Fahrenheit -/
noncomputable def approx_formula (C : ℝ) : ℝ := 2 * C + 30

/-- The deviation between the exact and approximate formulas -/
noncomputable def deviation (C : ℝ) : ℝ := |approx_formula C - exact_formula C| / |exact_formula C|

/-- The theorem stating the range of temperatures where the deviation is within 5% -/
theorem temperature_range_with_acceptable_deviation :
  ∀ C : ℝ, (40/29 ≤ C ∧ C ≤ 360/11) ↔ deviation C ≤ 0.05 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_range_with_acceptable_deviation_l911_91139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_is_irrational_l911_91107

-- Define the given real numbers
noncomputable def a : ℝ := Real.rpow 8 (1/3)
noncomputable def b : ℝ := 2/3
noncomputable def c : ℝ := Real.pi / 2
noncomputable def d : ℝ := |(-3)|

-- State the properties of the given numbers
axiom pi_irrational : Irrational Real.pi
axiom a_rational : ¬ Irrational a
axiom b_rational : ¬ Irrational b
axiom d_rational : ¬ Irrational d

-- Theorem to prove
theorem c_is_irrational : Irrational c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_is_irrational_l911_91107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_vector_exists_l911_91133

def a : ℝ × ℝ × ℝ := (3, 4, 2)
def b : ℝ × ℝ × ℝ := (-1, 1, 1)

theorem bisecting_vector_exists : ∃ v : ℝ × ℝ × ℝ, 
  ‖v‖ = 1 ∧ ∃ k : ℝ, b = k • ((a + ‖a‖ • v) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_vector_exists_l911_91133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_m_min_value_sum_squares_l911_91114

-- Define the functions f and g
noncomputable def f (x θ : ℝ) : ℝ := |x + Real.sin θ ^ 2|
noncomputable def g (x θ : ℝ) : ℝ := 2 * |x - Real.cos θ ^ 2|

-- Define the condition for a
def condition_a (a : ℝ) : Prop :=
  ∀ x θ, θ ∈ Set.Icc 0 (2 * Real.pi) → 2 * f x θ ≥ a - g x θ

-- Define m as the maximum value of a satisfying the condition
noncomputable def m : ℝ := sSup {a | condition_a a}

-- State the theorem for the first part
theorem max_value_m : m = 2 := by sorry

-- State the theorem for the second part
theorem min_value_sum_squares :
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + 2*b + 3*c = 2*m →
  a^2 + b^2 + c^2 ≥ 8/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_m_min_value_sum_squares_l911_91114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_fraction_l911_91110

theorem pure_imaginary_fraction (a : ℝ) : 
  (Complex.I * 3 = Complex.mk (a - 2) 3) → 
  (Complex.mk a 1) / (Complex.mk 1 a) = Complex.mk (4/5) (-3/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_fraction_l911_91110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_for_equilateral_triangle_l911_91190

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with vertex at the origin -/
structure Parabola where
  p : ℝ

/-- Check if a point lies on a parabola -/
def pointOnParabola (point : Point) (parabola : Parabola) : Prop :=
  point.y^2 = 2 * parabola.p * point.x

/-- The equation of the parabola passing through points A and B -/
def parabolaEquation (A B : Point) : ℝ → Prop :=
  fun p => pointOnParabola A ⟨p⟩ ∧ pointOnParabola B ⟨p⟩

/-- Theorem stating the equation of the parabola -/
theorem parabola_equation_for_equilateral_triangle 
  (A B : Point)
  (h_equilateral : A.x^2 + A.y^2 = 1 ∧ B.x^2 + B.y^2 = 1 ∧ (A.x - B.x)^2 + (A.y - B.y)^2 = 1)
  (h_perpendicular : A.x = B.x)
  (h_symmetric : A.y = -B.y)
  : ∃ p : ℝ, p = Real.sqrt 3 / 12 ∨ p = -Real.sqrt 3 / 12 ∧ parabolaEquation A B p := by
  sorry

#check parabola_equation_for_equilateral_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_for_equilateral_triangle_l911_91190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_power_n_plus_inverse_l911_91152

theorem x_power_n_plus_inverse (θ : ℝ) (x : ℂ) (n : ℕ+) 
  (h1 : 0 < θ) (h2 : θ < Real.pi / 2) 
  (h3 : x + x⁻¹ = 2 * Real.sin θ) : 
  x^(n : ℕ) + (x^(n : ℕ))⁻¹ = 2 * Real.sin (n * θ) := by
  sorry

#check x_power_n_plus_inverse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_power_n_plus_inverse_l911_91152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_darts_game_score_l911_91120

/-- Represents the score of a single dart throw -/
def DartScore := ℕ

/-- Calculates the score of the fifth dart with the 'Golden Bullseye' rule -/
def goldenBullseyeScore (previousTotal : ℕ) : ℕ :=
  50 + previousTotal

/-- Represents the darts game with given conditions -/
structure DartsGame where
  bullseyeScore : ℕ
  firstDartScore : ℕ
  secondDartScore : ℕ
  thirdDartScore : ℕ
  fourthDartScore : ℕ
  fifthDartScore : ℕ → ℕ

/-- Calculates the total score of the darts game -/
def totalScore (game : DartsGame) : ℕ :=
  let previousTotal := game.firstDartScore + game.secondDartScore + game.thirdDartScore + game.fourthDartScore
  game.firstDartScore + game.secondDartScore + game.thirdDartScore + game.fourthDartScore + game.fifthDartScore previousTotal

/-- Theorem stating that the final total score is 600 points -/
theorem darts_game_score :
  ∃ (game : DartsGame),
    game.bullseyeScore = 50 ∧
    game.firstDartScore = 3 * game.bullseyeScore ∧
    game.secondDartScore = 0 ∧
    game.thirdDartScore = game.bullseyeScore / 2 ∧
    game.fourthDartScore = 2 * game.bullseyeScore ∧
    game.fifthDartScore = goldenBullseyeScore ∧
    totalScore game = 600 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_darts_game_score_l911_91120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_powers_of_two_l911_91128

theorem two_digit_powers_of_two : 
  ∃! (count : ℕ), ∃ (S : Finset ℕ), 
    (∀ n ∈ S, 10 ≤ 2^n ∧ 2^n ≤ 99) ∧ 
    (∀ n ∉ S, 2^n < 10 ∨ 99 < 2^n) ∧ 
    S.card = count ∧
    count = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_powers_of_two_l911_91128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lost_card_number_l911_91171

theorem lost_card_number (n : ℕ) (h1 : n > 0) (h2 : ∃ x : ℕ, x ≤ n ∧ n * (n + 1) / 2 - 101 = x) : 
  n * (n + 1) / 2 - 101 = 4 := by
  sorry

#check lost_card_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lost_card_number_l911_91171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_deriv_tangent_at_e_tangent_through_origin_l911_91149

/-- The exponential function -/
noncomputable def exp (x : ℝ) : ℝ := Real.exp x

/-- The derivative of the exponential function -/
theorem exp_deriv (x : ℝ) : deriv exp x = exp x := by sorry

/-- Tangent line equation at (e, e^e) -/
theorem tangent_at_e (x y : ℝ) :
  (exp (Real.exp 1) * x - y + exp (Real.exp 1) - exp (Real.exp 1 + 1) = 0) ↔
  (y - exp (exp (Real.exp 1)) = exp (exp (Real.exp 1)) * (x - Real.exp 1)) := by sorry

/-- Tangent line equation through origin -/
theorem tangent_through_origin (x y : ℝ) :
  ((Real.exp 1) * x - y = 0) ↔
  (∃ x₀ : ℝ, x₀ = 1 ∧ y = exp x₀ ∧ y = (exp x₀ / x₀) * x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_deriv_tangent_at_e_tangent_through_origin_l911_91149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_equivalent_to_parametric_l911_91147

/-- Curve C in parametric form -/
noncomputable def C (α : Real) : Real × Real := (Real.cos α + 1, Real.sin α)

/-- Polar equation of curve C -/
noncomputable def polar_equation (θ : Real) : Real := 2 * Real.cos θ

/-- Theorem stating that the polar equation represents the same curve as the parametric equations -/
theorem polar_equation_equivalent_to_parametric :
  ∀ θ : Real, ∃ α : Real, 
    (polar_equation θ * Real.cos θ, polar_equation θ * Real.sin θ) = C α :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_equivalent_to_parametric_l911_91147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_quotient_division_l911_91121

/-- Given a polynomial division of x⁵ by (x - 1/2), prove that the remainder
    when the quotient q₁(x) is divided by (x - 1) is equal to 2.9375 -/
theorem remainder_of_quotient_division (q₁ : ℝ → ℝ) (r₁ : ℝ) :
  (∀ x, x^5 = (x - 1/2) * q₁ x + r₁) →
  (q₁ 1 = 2.9375) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_quotient_division_l911_91121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_is_line_l911_91117

/-- A curve in polar coordinates defined by θ = π/4 is a line -/
theorem polar_curve_is_line : 
  ∀ (r : ℝ) (θ : ℝ), θ = π/4 → 
  ∃ (a b : ℝ), ∀ (x y : ℝ), x = r * Real.cos θ ∧ y = r * Real.sin θ → y = a * x + b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_is_line_l911_91117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_visits_implies_infinite_visits_l911_91148

/-- A circular billiard table -/
structure CircularBilliardTable where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A ball moving on a circular billiard table -/
structure Ball where
  position : ℝ × ℝ
  velocity : ℝ × ℝ

/-- The path of a ball on a circular billiard table -/
def BallPath (table : CircularBilliardTable) (ball : Ball) : Set (ℝ × ℝ) :=
  sorry

/-- A point is visited by the ball if it's in the ball's path -/
def PointVisited (table : CircularBilliardTable) (ball : Ball) (point : ℝ × ℝ) : Prop :=
  point ∈ BallPath table ball

/-- A point is visited infinitely many times if for any number of visits, 
    there exists a future time when the point will be visited again -/
def InfinitelyVisited (table : CircularBilliardTable) (ball : Ball) (point : ℝ × ℝ) : Prop :=
  ∀ n : ℕ, ∃ t : ℝ, t > 0 ∧ PointVisited table 
    (Ball.mk (ball.position.1 + t * ball.velocity.1, ball.position.2 + t * ball.velocity.2) ball.velocity) 
    point

/-- The main theorem: if a point is visited three times, it's visited infinitely many times -/
theorem three_visits_implies_infinite_visits 
  (table : CircularBilliardTable) (ball : Ball) (point : ℝ × ℝ) :
  (∃ t₁ t₂ t₃ : ℝ, t₁ < t₂ ∧ t₂ < t₃ ∧ 
    PointVisited table (Ball.mk (ball.position.1 + t₁ * ball.velocity.1, ball.position.2 + t₁ * ball.velocity.2) ball.velocity) point ∧
    PointVisited table (Ball.mk (ball.position.1 + t₂ * ball.velocity.1, ball.position.2 + t₂ * ball.velocity.2) ball.velocity) point ∧
    PointVisited table (Ball.mk (ball.position.1 + t₃ * ball.velocity.1, ball.position.2 + t₃ * ball.velocity.2) ball.velocity) point) →
  InfinitelyVisited table ball point :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_visits_implies_infinite_visits_l911_91148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_l911_91185

/-- Given a line l with direction vector (1, -2, 0) and a plane α with normal vector (m, 3, 6),
    if l is parallel to α, then m = 6. -/
theorem line_parallel_to_plane (m : ℝ) : 
  let d : Fin 3 → ℝ := ![1, -2, 0]
  let n : Fin 3 → ℝ := ![m, 3, 6]
  (∀ (t : ℝ), (d 0) * (n 0) + (d 1) * (n 1) + (d 2) * (n 2) = 0) →
  m = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_l911_91185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l911_91182

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.sin x - Real.cos x)

theorem domain_of_f :
  ∀ x : ℝ, f x ∈ Set.range f ↔ 
    ∃ k : ℤ, x ∈ Set.Icc (2 * k * Real.pi + Real.pi / 4) (2 * k * Real.pi + 5 * Real.pi / 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l911_91182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_monotonic_intervals_condition_l911_91108

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * a * x^3 + x^2 + a * x + 1

-- Define the derivative of f(x)
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x + a

-- Theorem statement
theorem three_monotonic_intervals_condition (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧
    (∀ x ∈ Set.Ioo x₁ x₂, f_derivative a x > 0) ∧
    (∀ x ∈ Set.Ioo x₂ x₃, f_derivative a x < 0) ∧
    (∀ x, x < x₁ ∨ x > x₃ → f_derivative a x > 0)) ↔
  a ∈ Set.Ioo (-1) 0 ∪ Set.Ioo 0 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_monotonic_intervals_condition_l911_91108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_is_6_l911_91178

-- Define constants
noncomputable def π : ℝ := Real.pi
def semicircle_circumference : ℝ := 28.27
def rectangle_breadth : ℝ := 16

-- Define the side length of the square
noncomputable def square_side : ℝ := 
  (semicircle_circumference / (π / 2 + 1))

-- Define the perimeter of the square
noncomputable def square_perimeter : ℝ := 4 * square_side

-- Define the length of the rectangle
noncomputable def rectangle_length : ℝ := 
  (square_perimeter - 2 * rectangle_breadth) / 2

-- Theorem statement
theorem rectangle_length_is_6 : 
  ‖rectangle_length - 6‖ < 0.01 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_is_6_l911_91178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_passes_through_A_perpendicular_to_BC_l911_91173

-- Define the points A, B, and C
def A : Fin 3 → ℝ := ![0, -8, 10]
def B : Fin 3 → ℝ := ![-5, 5, 7]
def C : Fin 3 → ℝ := ![-8, 0, 4]

-- Define the vector BC
def BC : Fin 3 → ℝ := ![C 0 - B 0, C 1 - B 1, C 2 - B 2]

-- Define the plane equation
def plane_equation (x y z : ℝ) : Prop := 3 * x + 5 * y + 3 * z + 10 = 0

-- Theorem statement
theorem plane_passes_through_A_perpendicular_to_BC :
  plane_equation (A 0) (A 1) (A 2) ∧
  (∀ (x y z : ℝ), plane_equation x y z → 
    (x - A 0) * BC 0 + (y - A 1) * BC 1 + (z - A 2) * BC 2 = 0) :=
by sorry

#check plane_passes_through_A_perpendicular_to_BC

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_passes_through_A_perpendicular_to_BC_l911_91173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eventual_stability_l911_91158

/-- Represents the state of a man's response (Yes or No) -/
inductive Response
  | Yes
  | No
deriving DecidableEq

/-- Represents the circular arrangement of 25 men -/
def CircularArrangement := Fin 25 → Response

/-- The rule for updating a man's response based on his and his neighbors' responses -/
def updateRule (prev: CircularArrangement) (i: Fin 25) : Response :=
  let left := prev ((i - 1) % 25)
  let right := prev ((i + 1) % 25)
  let current := prev i
  if current = left ∨ current = right then current else
    match current with
    | Response.Yes => Response.No
    | Response.No => Response.Yes

/-- The next state of the entire arrangement after applying the update rule -/
def nextState (current: CircularArrangement) : CircularArrangement :=
  fun i => updateRule current i

/-- A sequence of states represents the evolution of the system over time -/
def StateSequence := ℕ → CircularArrangement

/-- A state sequence is stable if it reaches a point where no further changes occur -/
def isStable (seq: StateSequence) : Prop :=
  ∃ n : ℕ, ∀ m : ℕ, m ≥ n → seq m = seq n

/-- The main theorem: Any initial configuration will lead to a stable state -/
theorem eventual_stability (initial: CircularArrangement) :
  isStable (fun n => (nextState^[n]) initial) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eventual_stability_l911_91158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steve_nickels_l911_91137

-- Define the number of each type of coin
variable (nickels dimes quarters : ℕ)

-- Define the relationships between coins
axiom dimes_relation : dimes = nickels + 4
axiom quarters_relation : quarters = dimes + 3

-- Define the total value of coins in cents
def total_value : ℕ := 5 * nickels + 10 * dimes + 25 * quarters + 5

-- State the theorem
theorem steve_nickels : 
  total_value nickels dimes quarters = 380 → nickels = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_steve_nickels_l911_91137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_store_profit_theorem_l911_91131

/-- Represents the furniture store's inventory and pricing strategy -/
structure FurnitureStore where
  a : ℚ  -- Changed to ℚ (rational numbers) for better computability
  tablePurchasePrice : ℚ := a
  chairPurchasePrice : ℚ := a - 110
  tableRetailPrice : ℚ := 270
  chairRetailPrice : ℚ := 70
  setPrice : ℚ := 500
  numTables : ℕ
  numChairs : ℕ
  (chair_table_relation : numChairs = 5 * numTables + 20)
  (total_quantity : numTables + numChairs ≤ 200)
  (purchase_equality : 600 / a = 160 / (a - 110))

/-- Calculates the profit for the store -/
def calculateProfit (store : FurnitureStore) : ℚ :=
  let setProfit := store.setPrice - store.tablePurchasePrice - 4 * store.chairPurchasePrice
  let tableRetailProfit := store.tableRetailPrice - store.tablePurchasePrice
  let chairRetailProfit := store.chairRetailPrice - store.chairPurchasePrice
  (store.numTables / 2 : ℚ) * setProfit + 
  (store.numTables / 2 : ℚ) * tableRetailProfit + 
  (store.numChairs - 2 * store.numTables : ℚ) * chairRetailProfit

/-- Represents the store after price increase -/
structure FurnitureStoreAfterIncrease extends FurnitureStore where
  newTablePurchasePrice : ℚ := tablePurchasePrice + 10
  newChairPurchasePrice : ℚ := chairPurchasePrice + 10
  numSets : ℕ

/-- Calculates the profit after price increase -/
def calculateProfitAfterIncrease (store : FurnitureStoreAfterIncrease) : ℚ :=
  let newSetProfit := store.setPrice - store.newTablePurchasePrice - 4 * store.newChairPurchasePrice
  let newTableRetailProfit := store.tableRetailPrice - store.newTablePurchasePrice
  let newChairRetailProfit := store.chairRetailPrice - store.newChairPurchasePrice
  (store.numSets : ℚ) * newSetProfit + 
  (store.numTables - store.numSets : ℚ) * newTableRetailProfit + 
  (store.numChairs - 4 * store.numSets : ℚ) * newChairRetailProfit

/-- The main theorem to be proved -/
theorem furniture_store_profit_theorem (store : FurnitureStore) 
  (storeAfter : FurnitureStoreAfterIncrease) :
  store.a = 150 →
  store.numTables = 30 →
  calculateProfit store = 7050 ∧ 
  storeAfter.numSets = 20 ∧
  calculateProfitAfterIncrease storeAfter = 4800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_store_profit_theorem_l911_91131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_less_than_one_necessary_not_sufficient_l911_91122

-- Define the circle C: x^2 + y^2 - 2mx = 0
def circle_equation (m : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2*m*x = 0

-- Define what it means for a point to be outside the circle
def outside_circle (m : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2*m*x > 0

-- Define the point P(1,1)
def point_P : ℝ × ℝ := (1, 1)

-- Theorem statement
theorem m_less_than_one_necessary_not_sufficient :
  (∀ m : ℝ, outside_circle m point_P.1 point_P.2 → m < 1) ∧
  ¬(∀ m : ℝ, m < 1 → outside_circle m point_P.1 point_P.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_less_than_one_necessary_not_sufficient_l911_91122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_negative_5_l911_91161

def sequence_a : ℕ → ℤ
  | 0 => 1  -- Add this case for 0
  | 1 => 1
  | (n + 1) => sequence_a n - n

theorem a_4_equals_negative_5 : sequence_a 4 = -5 := by
  -- Unfold the definition for each step
  calc
    sequence_a 4 = sequence_a 3 - 3 := rfl
    _ = (sequence_a 2 - 2) - 3 := rfl
    _ = ((sequence_a 1 - 1) - 2) - 3 := rfl
    _ = ((1 - 1) - 2) - 3 := rfl
    _ = (0 - 2) - 3 := by simp
    _ = -2 - 3 := by simp
    _ = -5 := by simp

#eval sequence_a 4  -- This will evaluate the function for n = 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_negative_5_l911_91161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_walking_speed_l911_91164

/-- Calculates the walking speed in km/hr given the bridge length in meters and crossing time in minutes -/
noncomputable def walking_speed (bridge_length : ℝ) (crossing_time : ℝ) : ℝ :=
  (bridge_length / crossing_time) * (60 / 1000)

/-- Theorem stating that a man crossing a 1250-meter bridge in 15 minutes has a walking speed of 5 km/hr -/
theorem man_walking_speed :
  walking_speed 1250 15 = 5 := by
  -- Unfold the definition of walking_speed
  unfold walking_speed
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_walking_speed_l911_91164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_l911_91113

/-- A convex pentagon with rational sides and equal angles is regular -/
theorem regular_pentagon (v₁ v₂ v₃ v₄ v₅ : ℂ) : 
  let z : Fin 5 → ℂ := fun i => 
    match i with
    | ⟨0, _⟩ => v₂ - v₁
    | ⟨1, _⟩ => v₃ - v₂
    | ⟨2, _⟩ => v₄ - v₃
    | ⟨3, _⟩ => v₅ - v₄
    | ⟨4, _⟩ => v₁ - v₅
  (∀ i : Fin 5, ∃ q : ℚ, (z i).re = q ∧ (z i).im = q) →  -- rational sides
  (z 0 + z 1 + z 2 + z 3 + z 4 = 0) →  -- closed figure
  (∀ i j : Fin 5, i ≠ j → Complex.arg (z j / z i) = 2 * π / 5) →  -- equal angles
  (∀ i j : Fin 5, Complex.abs (z i) = Complex.abs (z j)) -- regular pentagon
  := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_l911_91113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_k_negative_one_f_greater_than_exp_negative_x_implies_k_positive_l911_91163

-- Define the function f(x)
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (2 : ℝ)^x + k * (2 : ℝ)^(-x)

-- Theorem 1: If f is an odd function, then k = -1
theorem odd_function_implies_k_negative_one (k : ℝ) :
  (∀ x, f k x = -f k (-x)) → k = -1 := by
  sorry

-- Theorem 2: If f(x) > 2^(-x) for all x ≥ 0, then k > 0
theorem f_greater_than_exp_negative_x_implies_k_positive (k : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f k x > (2 : ℝ)^(-x)) → k > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_k_negative_one_f_greater_than_exp_negative_x_implies_k_positive_l911_91163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_pairing_exists_S_8_no_perfect_pairing_S_5_many_perfect_pairings_exist_l911_91140

-- Define the set S_n
def S (n : ℕ) : Set ℕ := {i | 1 ≤ i ∧ i ≤ 2*n}

-- Define a perfect pairing
def is_perfect_pairing (n : ℕ) (p : List (ℕ × ℕ)) : Prop :=
  (∀ pair ∈ p, pair.1 ∈ S n ∧ pair.2 ∈ S n) ∧
  (∀ pair ∈ p, ∃ k : ℕ, pair.1 + pair.2 = k^2) ∧
  p.length = n ∧
  (∀ i ∈ S n, ∃ pair ∈ p, i = pair.1 ∨ i = pair.2)

-- Theorem statements
theorem perfect_pairing_exists_S_8 : ∃ p : List (ℕ × ℕ), is_perfect_pairing 8 p := by
  sorry

theorem no_perfect_pairing_S_5 : ¬∃ p : List (ℕ × ℕ), is_perfect_pairing 5 p := by
  sorry

theorem many_perfect_pairings_exist :
  ∃ n : ℕ, ∃ ps : List (List (ℕ × ℕ)), 
    (∀ p ∈ ps, is_perfect_pairing n p) ∧ 
    ps.length ≥ 2017 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_pairing_exists_S_8_no_perfect_pairing_S_5_many_perfect_pairings_exist_l911_91140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_property_l911_91179

-- Define an even function on [-5,5]
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc (-5 : ℝ) 5 → f (-x) = f x

-- State the theorem
theorem even_function_property (f : ℝ → ℝ) 
  (h_even : EvenFunction f) (h_inequality : f 3 > f 1) : 
  f (-1) < f 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_property_l911_91179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_theorem_l911_91145

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def isConvexQuadrilateral (q : Quadrilateral) : Prop := sorry

def segmentLength (p1 p2 : ℝ × ℝ) : ℝ := sorry

def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

noncomputable def area (q : Quadrilateral) : ℝ := sorry

-- Define the theorem
theorem quadrilateral_area_theorem (q : Quadrilateral) :
  isConvexQuadrilateral q →
  segmentLength q.A q.B = 6 →
  segmentLength q.B q.C = 5 →
  segmentLength q.C q.D = 13 →
  segmentLength q.D q.A = 13 →
  angle q.C q.D q.A = 90 →
  ∃ (a b c : ℝ), 
    area q = 84.5 + Real.sqrt 2783 / 2 ∧
    area q = Real.sqrt a + b * Real.sqrt c ∧
    (∀ n : ℕ, n > 1 → ¬(IsSquare (↑n * ↑n * a))) ∧
    (∀ n : ℕ, n > 1 → ¬(IsSquare (↑n * ↑n * c))) ∧
    a + b + c = 2783.5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_theorem_l911_91145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l911_91183

/-- The function f(x) = a*ln(x+1) - x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log (x + 1) - x

/-- The derivative of f(x) -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := a / (x + 1) - 1

theorem f_properties (a : ℝ) :
  (∀ x > -1, a ≤ 0 → f_deriv a x < 0) ∧
  (a > 0 → (∀ x, -1 < x ∧ x < a - 1 → f_deriv a x > 0) ∧
           (∀ x > a - 1, f_deriv a x < 0)) ∧
  (a > 0 → ∀ x > -1, f a x < (a - 1) * Real.log a + a^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l911_91183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_implies_sum_l911_91124

theorem set_equality_implies_sum (a b : ℝ) (ha : a ≠ 0) 
  (h : ({1, b/a, a} : Set ℝ) = {0, a+b, a^2}) : a^2 + b^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_implies_sum_l911_91124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l911_91165

/-- Predicate to check if a line is the directrix of a parabola -/
def IsDirectrix (x y : ℝ) : Prop :=
  ∃ (p : ℝ), p > 0 ∧ x^2 = 4*p*y ∧ y = -p

/-- The directrix of a parabola given by x² = y -/
theorem parabola_directrix (x y : ℝ) : x^2 = y → (4*y + 1 = 0 ↔ IsDirectrix x y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l911_91165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_don_arrival_time_l911_91138

-- Define the speeds and times
noncomputable def ambulance_speed : ℝ := 60
noncomputable def don_speed : ℝ := 30
noncomputable def mary_time : ℝ := 15 / 60  -- 15 minutes converted to hours

-- Define the theorem
theorem don_arrival_time :
  let distance := ambulance_speed * mary_time
  don_speed * (distance / don_speed) = 30 / 60 := by
  -- Unfold the definitions
  unfold ambulance_speed don_speed mary_time
  -- Simplify the expression
  simp [mul_div_cancel']
  -- The proof is complete
  sorry

#check don_arrival_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_don_arrival_time_l911_91138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_and_area_l911_91132

/-- A right rectangular prism with given face areas -/
structure RectPrism where
  front_area : ℝ
  bottom_area : ℝ
  side_area : ℝ
  front_area_pos : 0 < front_area
  bottom_area_pos : 0 < bottom_area
  side_area_pos : 0 < side_area

/-- The volume of a right rectangular prism -/
noncomputable def volume (p : RectPrism) : ℝ :=
  Real.sqrt (p.front_area * p.bottom_area * p.side_area)

/-- The total surface area of a right rectangular prism -/
def surface_area (p : RectPrism) : ℝ :=
  2 * (p.front_area + p.bottom_area + p.side_area)

theorem prism_volume_and_area (p : RectPrism) 
  (h1 : p.front_area = 24) 
  (h2 : p.bottom_area = 18) 
  (h3 : p.side_area = 12) : 
  volume p = 72 ∧ surface_area p = 108 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_and_area_l911_91132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_vertex_l911_91156

def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 2

theorem quadratic_vertex :
  ∃ (a b c : ℝ), (∀ x, f x = a * (x - b)^2 + c) ∧ f b = c ∧ b = 1 ∧ c = -7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_vertex_l911_91156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_l911_91129

theorem angle_B_measure (A B C : ℝ) 
  (h : Real.sin B ^ 2 = Real.sin A ^ 2 + Real.sqrt 3 * Real.sin A * Real.sin C + Real.sin C ^ 2) :
  B = 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_l911_91129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_completion_time_l911_91100

/-- The number of days A takes to complete the entire work -/
noncomputable def a_days : ℝ := 15

/-- The number of days B takes to complete the entire work -/
noncomputable def b_days : ℝ := 15

/-- The number of days B worked before leaving -/
noncomputable def b_worked : ℝ := 10

/-- The fraction of work completed by B before leaving -/
noncomputable def b_completed : ℝ := b_worked / b_days

/-- The fraction of work remaining after B left -/
noncomputable def remaining_work : ℝ := 1 - b_completed

/-- The number of days A takes to complete the remaining work -/
noncomputable def a_remaining_days : ℝ := remaining_work * a_days

/-- Theorem stating that A can complete the remaining work in 5 days -/
theorem a_completion_time : a_remaining_days = 5 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_completion_time_l911_91100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_towel_bleaching_l911_91109

theorem towel_bleaching (original_length original_breadth : ℝ) 
  (h1 : original_length > 0) (h2 : original_breadth > 0) : 
  let new_length := 0.9 * original_length
  let new_area := 0.72 * (original_length * original_breadth)
  ∃ new_breadth : ℝ, 
    new_breadth = 0.8 * original_breadth ∧ 
    new_length * new_breadth = new_area :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_towel_bleaching_l911_91109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_new_students_count_l911_91174

/-- Calculates the number of new students joining a class given the original and new average ages -/
def new_students_count (original_avg : ℚ) (new_students_avg : ℚ) (avg_decrease : ℚ) (original_strength : ℕ) : ℚ :=
  let new_avg := original_avg - avg_decrease
  (original_strength : ℚ) * (original_avg - new_avg) / (new_avg - new_students_avg)

/-- Proves that 2 new students joined the class given the specific conditions -/
theorem prove_new_students_count : 
  new_students_count 40 32 4 2 = 2 := by
  unfold new_students_count
  simp
  -- The proof goes here
  sorry

#eval new_students_count 40 32 4 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_new_students_count_l911_91174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_products_l911_91123

def X : Finset Nat := Finset.range 9 \ {0}

theorem min_sum_of_products (A B : Finset Nat) : 
  A ⊆ X → B ⊆ X → A ∪ B = X → A ∩ B = ∅ → A.Nonempty → B.Nonempty →
  (A.prod id + B.prod id) ≥ 402 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_products_l911_91123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_monotone_in_triangle_tan_sum_product_in_oblique_triangle_l911_91170

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  angle_sum : A + B + C = Real.pi
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0

-- Statement A
theorem sin_monotone_in_triangle (t : Triangle) : 
  t.A > t.B → Real.sin t.A > Real.sin t.B := by
  sorry

-- Statement D
theorem tan_sum_product_in_oblique_triangle (t : Triangle) 
  (h : t.A ≠ Real.pi/2 ∧ t.B ≠ Real.pi/2 ∧ t.C ≠ Real.pi/2) : 
  Real.tan t.A + Real.tan t.B + Real.tan t.C = 
  Real.tan t.A * Real.tan t.B * Real.tan t.C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_monotone_in_triangle_tan_sum_product_in_oblique_triangle_l911_91170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_max_min_values_monotonicity_intervals_axes_of_symmetry_centers_of_symmetry_l911_91187

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * (Real.cos x)^2 + (1/2) * Real.sin (2*x)

-- Theorem for the smallest positive period
theorem smallest_positive_period : ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ ∀ S : ℝ, S > 0 → (∀ x : ℝ, f (x + S) = f x) → T ≤ S := by
  sorry

-- Theorem for maximum and minimum values on the given interval
theorem max_min_values : ∃ max min : ℝ, 
  (∀ x : ℝ, -π/6 ≤ x ∧ x ≤ π/4 → f x ≤ max) ∧
  (∃ x : ℝ, -π/6 ≤ x ∧ x ≤ π/4 ∧ f x = max) ∧
  (∀ x : ℝ, -π/6 ≤ x ∧ x ≤ π/4 → min ≤ f x) ∧
  (∃ x : ℝ, -π/6 ≤ x ∧ x ≤ π/4 ∧ f x = min) := by
  sorry

-- Theorem for monotonicity intervals
theorem monotonicity_intervals : ∀ k : ℤ, 
  (∀ x y : ℝ, k*π - 5*π/12 ≤ x ∧ x < y ∧ y ≤ k*π + π/12 → f x < f y) ∧
  (∀ x y : ℝ, k*π + π/12 ≤ x ∧ x < y ∧ y ≤ k*π + 7*π/12 → f y < f x) := by
  sorry

-- Theorem for axes of symmetry
theorem axes_of_symmetry : ∀ k : ℤ, ∀ x : ℝ, 
  f (k*π/2 + π/12 + x) = f (k*π/2 + π/12 - x) := by
  sorry

-- Theorem for centers of symmetry
theorem centers_of_symmetry : ∀ k : ℤ, ∃ y : ℝ, 
  f (k*π/2 - π/6 + x) + f (k*π/2 - π/6 - x) = 2*y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_max_min_values_monotonicity_intervals_axes_of_symmetry_centers_of_symmetry_l911_91187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_perp_AD_C_is_correct_cosine_diagonals_l911_91125

-- Define the points
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (3, 2)
def D : ℝ × ℝ := (-1, 4)

-- Define vector operations
def vec (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem 1: AB is perpendicular to AD
theorem AB_perp_AD : dot_product (vec A B) (vec A D) = 0 := by sorry

-- Define C based on the rectangle property
def C : ℝ × ℝ := (0, 5)

-- Theorem 2: C is correctly defined
theorem C_is_correct : vec A B = vec D C := by sorry

-- Define the diagonals
def AC : ℝ × ℝ := vec A C
def BD : ℝ × ℝ := vec B D

-- Define the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Theorem 3: Cosine of the acute angle between diagonals
theorem cosine_diagonals : 
  dot_product AC BD / (magnitude AC * magnitude BD) = 4/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_perp_AD_C_is_correct_cosine_diagonals_l911_91125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joanne_main_job_wage_l911_91172

/-- Joanne's work schedule and earnings --/
structure WorkSchedule where
  main_job_hours_per_day : ℕ
  part_time_job_hours_per_day : ℕ
  part_time_job_hourly_rate : ℚ
  days_per_week : ℕ
  total_weekly_earnings : ℚ

/-- Calculate Joanne's hourly wage at her main job --/
noncomputable def main_job_hourly_wage (w : WorkSchedule) : ℚ :=
  let part_time_daily_earnings := w.part_time_job_hours_per_day * w.part_time_job_hourly_rate
  let part_time_weekly_earnings := part_time_daily_earnings * w.days_per_week
  let main_job_weekly_earnings := w.total_weekly_earnings - part_time_weekly_earnings
  let main_job_weekly_hours := w.main_job_hours_per_day * w.days_per_week
  main_job_weekly_earnings / main_job_weekly_hours

/-- Theorem: Joanne's hourly wage at her main job is $16 --/
theorem joanne_main_job_wage :
  let w : WorkSchedule := {
    main_job_hours_per_day := 8
    part_time_job_hours_per_day := 2
    part_time_job_hourly_rate := 27/2
    days_per_week := 5
    total_weekly_earnings := 775
  }
  main_job_hourly_wage w = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joanne_main_job_wage_l911_91172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_divisibility_l911_91154

theorem sum_of_powers_divisibility (n : ℕ) :
  (1^n + 2^n + 3^n + 4^n) % 5 = 0 ↔ n % 4 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_divisibility_l911_91154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_example_4_good_max_k_is_4_l911_91169

/-- A quadruplet is k-good if it satisfies the given conditions -/
def is_k_good (a b c d k : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧  -- distinct
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧  -- positive
  ¬(2 * a = b + c ∨ 2 * a = b + d ∨ 2 * a = c + d ∨
    2 * b = a + c ∨ 2 * b = a + d ∨ 2 * b = c + d ∨
    2 * c = a + b ∨ 2 * c = a + d ∨ 2 * c = b + d ∨
    2 * d = a + b ∨ 2 * d = a + c ∨ 2 * d = b + c) ∧  -- no three form arithmetic progression
  ∃ (x y z w : ℕ), (Finset.toSet {x, y, z, w} ⊆ Finset.toSet {a + b, a + c, a + d, b + c, b + d, c + d} ∧
                    x < y ∧ y < z ∧ z < w ∧
                    y - x = z - y ∧ z - y = w - z ∧
                    Finset.card {x, y, z, w} = k)  -- k sums form arithmetic progression

/-- The quadruplet (1, 2, 4, 5) is 4-good -/
theorem example_4_good : is_k_good 1 2 4 5 4 := by sorry

/-- There does not exist a k-good quadruplet for any k > 4 -/
theorem max_k_is_4 : ∀ k > 4, ¬∃ a b c d, is_k_good a b c d k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_example_4_good_max_k_is_4_l911_91169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_neg_properties_l911_91176

-- Define the function f(x) = tan(-x)
noncomputable def f (x : ℝ) : ℝ := Real.tan (-x)

-- State the theorem
theorem tan_neg_properties :
  (∃ (p : ℝ), p > 0 ∧ p = π ∧ ∀ (x : ℝ), f (x + p) = f x) ∧ 
  (∀ (x y : ℝ), π/2 < x ∧ x < y ∧ y < π → f y < f x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_neg_properties_l911_91176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_unity_ratio_theorem_l911_91115

noncomputable def ω : ℂ := -1/2 + Complex.I * (Real.sqrt 3 / 2)

theorem cube_root_unity_ratio_theorem (a b c : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a / b = b / c) (h2 : b / c = c / a) :
  (a + b - c) / (a - b + c) ∈ ({1, ω, ω^2} : Set ℂ) :=
by
  sorry

#check cube_root_unity_ratio_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_unity_ratio_theorem_l911_91115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_count_64n_cubed_l911_91192

/-- Given a positive integer n where 150n² has 150 positive integer divisors,
    prove that 64n³ has 160 positive integer divisors -/
theorem divisor_count_64n_cubed (n : ℕ+) 
  (h : (Nat.divisors (150 * n^2)).card = 150) :
  (Nat.divisors (64 * n^3)).card = 160 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_count_64n_cubed_l911_91192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_bouquet_carnations_l911_91146

theorem first_bouquet_carnations : ℕ := by
  let total_bouquets : ℕ := 3
  let second_bouquet : ℕ := 14
  let third_bouquet : ℕ := 13
  let average_carnations : ℕ := 12
  let first_bouquet : ℕ := total_bouquets * average_carnations - second_bouquet - third_bouquet

  have h : first_bouquet = 9 := by
    -- Proof steps would go here
    sorry

  exact 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_bouquet_carnations_l911_91146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l911_91189

noncomputable section

variable (a b c A B C : ℝ)

def is_triangle (a b c A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  A > 0 ∧ B > 0 ∧ C > 0 ∧
  A + B + C = Real.pi

def triangle_area (a b c A B C : ℝ) : ℝ :=
  (1/2) * a * b * Real.sin C

theorem triangle_theorem 
  (h1 : b^2 + Real.sqrt 2 * a * c = a^2 + c^2)
  (h2 : a * Real.cos B = b * Real.sin A)
  (h3 : Real.sin B + Real.cos B = Real.sqrt 2)
  (h4 : A = Real.pi/3)
  (h5 : b = Real.sqrt 2)
  (h6 : is_triangle a b c A B C) :
  B = Real.pi/4 ∧ triangle_area a b c A B C = (3 + Real.sqrt 3)/4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l911_91189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_reflection_theorem_l911_91143

noncomputable def spherical_to_rectangular (ρ θ φ : Real) : Real × Real × Real :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

def reflect_across_y_eq_x (x y z : Real) : Real × Real × Real := (y, x, z)

theorem spherical_reflection_theorem :
  let (x, y, z) := spherical_to_rectangular 3 (3 * Real.pi / 2) (Real.pi / 3)
  let (x', y', z') := reflect_across_y_eq_x x y z
  (x', y', z') = (-3 * Real.sqrt 3 / 2, 0, 3 / 2) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_reflection_theorem_l911_91143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l911_91142

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : t.b * Real.cos t.C - t.c * Real.cos (t.A + t.C) = 3 * t.a * Real.cos t.B)
  (h2 : t.a * t.c * Real.cos t.B = 2)
  (h3 : t.a = Real.sqrt 6) :
  Real.cos t.B = 1/3 ∧ t.b = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l911_91142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_systems_l911_91167

-- Define a structure for a system of two equations
structure EquationSystem :=
  (eq1 : ℝ → ℝ → ℝ)
  (eq2 : ℝ → ℝ → ℝ)

-- Define a predicate for linear equations
def isLinear (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, ∃ a b c : ℝ, f x y = a * x + b * y + c

-- Define the four equation systems
def system1 : EquationSystem :=
  ⟨λ x y ↦ x - 2, λ x y ↦ y - 3⟩

def system2 : EquationSystem :=
  ⟨λ x y ↦ 4 * x - 15, λ x y ↦ 3 * y - 4 * x + 3⟩

def system3 : EquationSystem :=
  ⟨λ x y ↦ x + y - 16, λ x y ↦ x^2 - y - 4⟩

def system4 : EquationSystem :=
  ⟨λ x y ↦ x + y - 35, λ x y ↦ 2 * x + 4 * y - 94⟩

-- Define a predicate for a system of two linear equations
def isLinearSystem (s : EquationSystem) : Prop :=
  isLinear s.eq1 ∧ isLinear s.eq2

-- Theorem statement
theorem linear_systems :
  isLinearSystem system1 ∧
  isLinearSystem system2 ∧
  isLinearSystem system4 ∧
  ¬isLinearSystem system3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_systems_l911_91167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangement_l911_91153

/-- A function representing the number of people in each row -/
noncomputable def people_in_row : ℕ → ℕ := sorry

/-- Theorem about seating arrangement -/
theorem seating_arrangement (total_people : ℕ) (rows_with_ten : ℕ) (total_rows : ℕ) : 
  total_people = 58 →
  (∀ row, row ≤ total_rows → (9 ≤ people_in_row row ∧ people_in_row row ≤ 10)) →
  (10 * rows_with_ten + 9 * (total_rows - rows_with_ten) = total_people) →
  rows_with_ten = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangement_l911_91153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l911_91136

theorem circle_line_intersection (r : ℝ) (A B : ℝ × ℝ) :
  r > 0 →
  (∀ x y, x + Real.sqrt 3 * y - 2 = 0 → x^2 + y^2 = r^2) →
  (A.1^2 + A.2^2 = r^2) →
  (B.1^2 + B.2^2 = r^2) →
  (A.1 + Real.sqrt 3 * A.2 - 2 = 0) →
  (B.1 + Real.sqrt 3 * B.2 - 2 = 0) →
  Real.arccos ((A.1 * B.1 + A.2 * B.2) / r^2) = 2 * π / 3 →
  r = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l911_91136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_squared_l911_91181

theorem cube_root_sum_squared : 
  (Real.rpow (8 + 3 * Real.sqrt 21) (1/3) + Real.rpow (8 - 3 * Real.sqrt 21) (1/3)) ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_squared_l911_91181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_x_axis_intersection_l911_91180

/-- The point where the line 4y - 5x = 15 intersects the x-axis is (-3, 0). -/
theorem line_x_axis_intersection :
  ∃ (x : ℝ), 4 * 0 - 5 * x = 15 ∧ x = -3 := by
  use -3
  constructor
  · simp
    norm_num
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_x_axis_intersection_l911_91180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l911_91199

/-- An arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_properties
  (a : ℕ → ℝ) (h : is_arithmetic_sequence a) :
  (is_arithmetic_sequence (λ n ↦ a n + 3)) ∧
  (is_arithmetic_sequence (λ n ↦ 2 * a n)) ∧
  (is_arithmetic_sequence (λ n ↦ a (n + 1) - a n)) ∧
  (is_arithmetic_sequence (λ n ↦ 2 * a n + n)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l911_91199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_octagon_area_is_one_sixth_l911_91193

/-- A parallelogram with area 1 -/
structure Parallelogram :=
  (area : ℝ)
  (area_eq_one : area = 1)

/-- The octagon formed by connecting midpoints to opposite sides -/
def midpoint_octagon (p : Parallelogram) : Set (ℝ × ℝ) := sorry

/-- The area of the midpoint octagon -/
def midpoint_octagon_area (p : Parallelogram) : ℝ := sorry

/-- Theorem: The area of the midpoint octagon is 1/6 of the parallelogram's area -/
theorem midpoint_octagon_area_is_one_sixth (p : Parallelogram) :
  midpoint_octagon_area p = 1/6 * p.area := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_octagon_area_is_one_sixth_l911_91193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_have_two_common_tangents_l911_91197

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y - 15 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 4 = 0

-- Define the centers and radii of the circles
def center₁ : ℝ × ℝ := (-1, 3)
def center₂ : ℝ × ℝ := (2, -1)
def radius₁ : ℝ := 5
def radius₂ : ℝ := 1

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt 25

-- Define the number of common tangents (this is just a placeholder)
def number_of_common_tangents (C₁ C₂ : ℝ → ℝ → Prop) : ℕ := sorry

-- Theorem statement
theorem circles_have_two_common_tangents :
  distance_between_centers > radius₁ - radius₂ ∧
  distance_between_centers < radius₁ + radius₂ →
  ∃! (n : ℕ), n = 2 ∧ n = number_of_common_tangents C₁ C₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_have_two_common_tangents_l911_91197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sum_constant_sequence_l911_91184

-- Define a sequence type
def Sequence := ℕ → ℝ

-- Define what it means for a sequence to be arithmetic
def is_arithmetic (S : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, S (n + 1) - S n = d

-- Define what it means for a sequence to be constant
def is_constant (a : Sequence) : Prop :=
  ∃ c : ℝ, ∀ n : ℕ, a n = c

-- Define the relationship between a sequence and its partial sums
def partial_sum (a : Sequence) : Sequence :=
  λ n => (Finset.range n).sum a

-- State the theorem
theorem arithmetic_sum_constant_sequence :
  (∀ a : Sequence, is_arithmetic (partial_sum a) → is_constant a) ∧
  (∃ a : Sequence, is_constant a ∧ ¬is_arithmetic (partial_sum a)) := by
  sorry

#check arithmetic_sum_constant_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sum_constant_sequence_l911_91184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_gain_calculation_l911_91135

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (compounds_per_year : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / compounds_per_year) ^ (compounds_per_year * time)

def loan_AB_principal : ℝ := 3150
def loan_AB_rate : ℝ := 0.075
def loan_AB_compounds : ℝ := 4
def loan_AB_time : ℝ := 2.7

def loan_BC_principal : ℝ := 3150
def loan_BC_rate : ℝ := 0.1325
def loan_BC_compounds : ℝ := 12
def loan_BC_time : ℝ := 3.2

theorem B_gain_calculation :
  let AB_total := compound_interest loan_AB_principal loan_AB_rate loan_AB_compounds loan_AB_time
  let BC_total := compound_interest loan_BC_principal loan_BC_rate loan_BC_compounds loan_BC_time
  let AB_interest := AB_total - loan_AB_principal
  let BC_interest := BC_total - loan_BC_principal
  let B_gain := BC_interest - AB_interest
  ∃ ε > 0, |B_gain - 1110.14| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_gain_calculation_l911_91135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l911_91104

noncomputable section

variable (f : ℝ → ℝ)

-- Define the conditions for the function f
axiom f_pos (x : ℝ) : x > 0 → f x > 1
axiom f_mult (x y : ℝ) : f (x + y) = f x * f y
axiom f_inj (x y : ℝ) : x ≠ y → f x ≠ f y

-- Define sets A and B
def A : Set (ℝ × ℝ) := {p | f (p.1^2) * f (p.2^2) < f 1}
def B (a b c : ℝ) : Set (ℝ × ℝ) := {p | f (a * p.1 + b * p.2 + c) = 1}

-- State the theorem
theorem function_properties :
  (∀ x > 0, f x > 1) →
  (∀ x y, f (x + y) = f x * f y) →
  (∀ x y, x ≠ y → f x ≠ f y) →
  (f 0 = 1) ∧
  (∀ x y, x < y → f x < f y) ∧
  (∀ a b c, a ≠ 0 → (A f ∩ B f a b c = ∅ ↔ a^2 + b^2 ≤ c^2)) :=
by
  intros h1 h2 h3
  constructor
  · sorry -- Proof for f 0 = 1
  constructor
  · sorry -- Proof for monotonicity of f
  · sorry -- Proof for the condition on a, b, c

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l911_91104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_radius_l911_91198

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the endpoints of the diameter
def endpoint1 : ℝ × ℝ := (2, -7)
def endpoint2 : ℝ × ℝ := (8, 5)

-- Theorem statement
theorem circle_center_and_radius :
  ∃ (O : Set (ℝ × ℝ)),
    O = Circle (5, -1) (3 * Real.sqrt 5) ∧
    endpoint1 ∈ O ∧ endpoint2 ∈ O ∧
    (∀ (p : ℝ × ℝ), p ∈ O → (p.1 - 5)^2 + (p.2 + 1)^2 ≤ (3 * Real.sqrt 5)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_radius_l911_91198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_intersect_N_l911_91194

open Set Real

-- Define the sets
def U : Set ℝ := univ
def M : Set ℝ := {x | log (x - 1) / log (1/2) > -1}
def N : Set ℝ := {x | 1 < (2:ℝ)^x ∧ (2:ℝ)^x < 4}

-- State the theorem
theorem complement_M_intersect_N :
  (U \ M) ∩ N = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_intersect_N_l911_91194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_a_l911_91168

-- Define the regression line equation
noncomputable def regression_line (x : ℝ) : ℝ := 12.6 * x + 0.6

-- Define the data points
def data_points : List (ℝ × ℝ) := [(2, 26), (3, 38), (3.5, 43), (4.5, 60), (7, 88)]

-- Calculate the mean of x values
noncomputable def mean_x : ℝ := (data_points.map (λ p => p.fst)).sum / data_points.length

-- Calculate the mean of y values
noncomputable def mean_y : ℝ := (data_points.map (λ p => p.snd)).sum / data_points.length

-- Theorem: The value of a that satisfies the equation is 88
theorem value_of_a : 
  let a := (data_points.getLast?).map Prod.snd |>.getD 0
  regression_line mean_x = mean_y ∧ a = 88 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_a_l911_91168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_arrangement_exists_l911_91101

def Vertex := Fin 8

structure Cube where
  vertices : Fin 8 → Nat
  adjacent : Vertex → Vertex → Bool

def has_sum_property (c : Cube) (v1 v2 v3 v4 : Vertex) : Prop :=
  c.vertices v1 = c.vertices v2 + c.vertices v3 + c.vertices v4

def valid_cube (c : Cube) : Prop :=
  (∀ v, c.vertices v ∈ Finset.range 9 \ {0}) ∧
  (∃ f1 f2 f3 : Fin 4 → Vertex,
    has_sum_property c (f1 0) (f1 1) (f1 2) (f1 3) ∧
    has_sum_property c (f2 0) (f2 1) (f2 2) (f2 3) ∧
    has_sum_property c (f3 0) (f3 1) (f3 2) (f3 3))

theorem cube_arrangement_exists :
  ∃ (c : Cube) (v6 v2 v3 v5 : Vertex),
    valid_cube c ∧
    c.vertices v6 = 6 ∧
    c.vertices v2 = 2 ∧
    c.vertices v3 = 3 ∧
    c.vertices v5 = 5 ∧
    c.adjacent v6 v2 ∧
    c.adjacent v6 v3 ∧
    c.adjacent v6 v5 :=
by sorry

#check cube_arrangement_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_arrangement_exists_l911_91101
