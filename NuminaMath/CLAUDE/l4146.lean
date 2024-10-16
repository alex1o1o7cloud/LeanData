import Mathlib

namespace NUMINAMATH_CALUDE_prob_divisible_by_five_prob_divisible_by_five_is_one_l4146_414693

/-- A three-digit positive integer with a ones digit of 5 -/
def ThreeDigitEndingIn5 : Type := { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ n % 10 = 5 }

/-- The probability that a number in ThreeDigitEndingIn5 is divisible by 5 -/
theorem prob_divisible_by_five (n : ThreeDigitEndingIn5) : ℚ :=
  1

/-- The probability that a number in ThreeDigitEndingIn5 is divisible by 5 is 1 -/
theorem prob_divisible_by_five_is_one : 
  ∀ n : ThreeDigitEndingIn5, prob_divisible_by_five n = 1 :=
sorry

end NUMINAMATH_CALUDE_prob_divisible_by_five_prob_divisible_by_five_is_one_l4146_414693


namespace NUMINAMATH_CALUDE_perpendicular_vectors_parallel_vectors_l4146_414621

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (-2, 1)

-- Define dot product for 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector subtraction
def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

-- Define scalar multiplication
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Define vector addition
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Theorem 1: k*a - b is perpendicular to a + 3*b when k = -13/5
theorem perpendicular_vectors (k : ℝ) : 
  dot_product (vec_sub (scalar_mul k a) b) (vec_add a (scalar_mul 3 b)) = 0 ↔ k = -13/5 := by
  sorry

-- Theorem 2: k*a - b is parallel to a + 3*b when k = -1/3
theorem parallel_vectors (k : ℝ) : 
  ∃ (t : ℝ), vec_sub (scalar_mul k a) b = scalar_mul t (vec_add a (scalar_mul 3 b)) ↔ k = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_parallel_vectors_l4146_414621


namespace NUMINAMATH_CALUDE_waiting_skill_players_count_l4146_414667

/-- Represents the football team's water cooler situation during practice. -/
structure WaterCoolerProblem where
  linemen : Nat
  skillPlayers : Nat
  linemenConsumption : Nat
  skillPlayerConsumption : Nat
  initialWater : Nat
  refillWater : Nat

/-- Calculates the number of skill position players who must wait for a drink. -/
def waitingSkillPlayers (p : WaterCoolerProblem) : Nat :=
  let totalLinemenConsumption := p.linemen * p.linemenConsumption
  let remainingWater := p.initialWater + p.refillWater - totalLinemenConsumption
  let drinkingSkillPlayers := remainingWater / p.skillPlayerConsumption
  p.skillPlayers - min p.skillPlayers drinkingSkillPlayers

/-- Theorem stating the number of waiting skill position players for the given problem. -/
theorem waiting_skill_players_count :
  let problem := WaterCoolerProblem.mk 20 18 12 10 190 120
  waitingSkillPlayers problem = 11 := by
  sorry

end NUMINAMATH_CALUDE_waiting_skill_players_count_l4146_414667


namespace NUMINAMATH_CALUDE_tangent_perpendicular_to_line_l4146_414601

-- Define the curve
def curve (x : ℝ) : ℝ := 2 * x^2

-- Define the line perpendicular to the tangent
def perp_line (x y : ℝ) : Prop := x + 4*y - 8 = 0

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 4*x - y - 2 = 0

-- Theorem statement
theorem tangent_perpendicular_to_line :
  ∀ (x₀ y₀ : ℝ),
  y₀ = curve x₀ →
  (∃ (m : ℝ), ∀ (x y : ℝ), y - y₀ = m * (x - x₀) → 
    (perp_line x y ↔ (x - x₀) * 1 + (y - y₀) * 4 = 0)) →
  tangent_line x₀ y₀ :=
sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_to_line_l4146_414601


namespace NUMINAMATH_CALUDE_win_in_four_moves_cannot_win_in_ten_moves_min_moves_for_2018_l4146_414660

/-- Represents the state of the game --/
structure GameState where
  coinsA : ℕ  -- Coins in box A
  coinsB : ℕ  -- Coins in box B

/-- Defines a single move in the game --/
inductive Move
  | MoveToB    : Move  -- Move a coin from A to B
  | RemoveFromA : Move  -- Remove coins from A equal to coins in B

/-- Applies a move to the current game state --/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.MoveToB => 
      { coinsA := state.coinsA - 1, coinsB := state.coinsB + 1 }
  | Move.RemoveFromA => 
      { coinsA := state.coinsA - state.coinsB, coinsB := state.coinsB }

/-- Checks if the game is won (i.e., box A is empty) --/
def isGameWon (state : GameState) : Prop := state.coinsA = 0

/-- Theorem: For 6 initial coins, the game can be won in 4 moves --/
theorem win_in_four_moves (initialCoins : ℕ) (h : initialCoins = 6) : 
  ∃ (moves : List Move), moves.length = 4 ∧ 
    isGameWon (moves.foldl applyMove { coinsA := initialCoins, coinsB := 0 }) := by
  sorry

/-- Theorem: For 31 initial coins, the game cannot be won in 10 moves --/
theorem cannot_win_in_ten_moves (initialCoins : ℕ) (h : initialCoins = 31) :
  ∀ (moves : List Move), moves.length = 10 → 
    ¬isGameWon (moves.foldl applyMove { coinsA := initialCoins, coinsB := 0 }) := by
  sorry

/-- Theorem: For 2018 initial coins, the minimum number of moves to win is 89 --/
theorem min_moves_for_2018 (initialCoins : ℕ) (h : initialCoins = 2018) :
  (∃ (moves : List Move), moves.length = 89 ∧ 
    isGameWon (moves.foldl applyMove { coinsA := initialCoins, coinsB := 0 })) ∧
  (∀ (moves : List Move), moves.length < 89 → 
    ¬isGameWon (moves.foldl applyMove { coinsA := initialCoins, coinsB := 0 })) := by
  sorry

end NUMINAMATH_CALUDE_win_in_four_moves_cannot_win_in_ten_moves_min_moves_for_2018_l4146_414660


namespace NUMINAMATH_CALUDE_x_range_l4146_414662

theorem x_range (x : ℝ) 
  (h : ∀ a b : ℝ, a^2 + b^2 = 1 → a + Real.sqrt 3 * b ≤ |x^2 - 1|) : 
  x ≤ -Real.sqrt 3 ∨ x ≥ Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_x_range_l4146_414662


namespace NUMINAMATH_CALUDE_shifted_parabola_properties_l4146_414688

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := x^2 + 2*x - 1

/-- The shifted parabola function -/
def shifted_parabola (x : ℝ) : ℝ := x^2 + 2*x + 3

/-- Theorem stating that the shifted parabola is a vertical translation of the original parabola
    and passes through the point (0, 3) -/
theorem shifted_parabola_properties :
  (∃ k : ℝ, ∀ x : ℝ, shifted_parabola x = original_parabola x + k) ∧
  shifted_parabola 0 = 3 := by
  sorry


end NUMINAMATH_CALUDE_shifted_parabola_properties_l4146_414688


namespace NUMINAMATH_CALUDE_cycle_gain_percent_l4146_414631

theorem cycle_gain_percent (cost_price selling_price : ℝ) : 
  cost_price = 675 →
  selling_price = 1080 →
  ((selling_price - cost_price) / cost_price) * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_cycle_gain_percent_l4146_414631


namespace NUMINAMATH_CALUDE_function_periodicity_l4146_414615

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem function_periodicity (f : ℝ → ℝ) 
  (h1 : ∀ x, |f x| ≤ 1)
  (h2 : ∀ x, f (x + 13/42) + f x = f (x + 1/6) + f (x + 1/7)) :
  is_periodic f 1 := by
sorry

end NUMINAMATH_CALUDE_function_periodicity_l4146_414615


namespace NUMINAMATH_CALUDE_school_boys_count_l4146_414651

theorem school_boys_count :
  ∀ (boys girls : ℕ),
  (boys : ℚ) / (girls : ℚ) = 5 / 13 →
  girls = boys + 64 →
  boys = 40 := by
sorry

end NUMINAMATH_CALUDE_school_boys_count_l4146_414651


namespace NUMINAMATH_CALUDE_line_segment_param_sum_squares_l4146_414604

/-- Given a line segment connecting (1,3) and (4,9) parameterized by x = pt + q and y = rt + s,
    where 0 ≤ t ≤ 1 and t = 0 corresponds to (1,3), prove that p^2 + q^2 + r^2 + s^2 = 55 -/
theorem line_segment_param_sum_squares (p q r s : ℝ) : 
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ∃ x y : ℝ, x = p * t + q ∧ y = r * t + s) → -- parameterization
  (q = 1 ∧ s = 3) → -- t = 0 corresponds to (1,3)
  (p + q = 4 ∧ r + s = 9) → -- endpoint (4,9)
  p^2 + q^2 + r^2 + s^2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_param_sum_squares_l4146_414604


namespace NUMINAMATH_CALUDE_intersection_A_B_l4146_414647

-- Define set A
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 + 2*x ≤ 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l4146_414647


namespace NUMINAMATH_CALUDE_johns_remaining_money_l4146_414643

/-- Calculates the amount of money John has left after walking his dog and spending money on books and his sister. -/
theorem johns_remaining_money (total_days : Nat) (sundays : Nat) (daily_pay : Nat) (book_cost : Nat) (sister_gift : Nat) :
  total_days = 30 →
  sundays = 4 →
  daily_pay = 10 →
  book_cost = 50 →
  sister_gift = 50 →
  (total_days - sundays) * daily_pay - (book_cost + sister_gift) = 160 := by
  sorry

end NUMINAMATH_CALUDE_johns_remaining_money_l4146_414643


namespace NUMINAMATH_CALUDE_roots_sum_problem_l4146_414650

theorem roots_sum_problem (a b : ℝ) : 
  a^2 - 5*a + 6 = 0 → b^2 - 5*b + 6 = 0 → a^4 + a^5*b^3 + a^3*b^5 + b^4 = 2905 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_problem_l4146_414650


namespace NUMINAMATH_CALUDE_banana_production_l4146_414634

theorem banana_production (x : ℕ) : 
  x + 10 * x = 99000 → x = 9000 := by
  sorry

end NUMINAMATH_CALUDE_banana_production_l4146_414634


namespace NUMINAMATH_CALUDE_set_intersection_and_complement_l4146_414696

def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2 < x ∧ x ≤ 5}

theorem set_intersection_and_complement :
  (A ∩ B = {x | 2 < x ∧ x < 3}) ∧
  ((Set.univ \ A) ∩ B = {x | 3 ≤ x ∧ x ≤ 5}) := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_and_complement_l4146_414696


namespace NUMINAMATH_CALUDE_sqrt_b_minus_3_domain_l4146_414606

theorem sqrt_b_minus_3_domain : {b : ℝ | ∃ x : ℝ, x^2 = b - 3} = {b : ℝ | b ≥ 3} := by sorry

end NUMINAMATH_CALUDE_sqrt_b_minus_3_domain_l4146_414606


namespace NUMINAMATH_CALUDE_sine_function_omega_l4146_414657

/-- Given a function f(x) = sin(ωx + π/3) where ω > 0, 
    if f(π/6) = f(π/3) and f(x) has a maximum value but no minimum value 
    in the interval (π/6, π/3), then ω = 2/3 -/
theorem sine_function_omega (ω : ℝ) (h_pos : ω > 0) : 
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x + π / 3)
  (f (π / 6) = f (π / 3)) → 
  (∃ (x : ℝ), x ∈ Set.Ioo (π / 6) (π / 3) ∧ 
    (∀ (y : ℝ), y ∈ Set.Ioo (π / 6) (π / 3) → f y ≤ f x)) →
  (∀ (x : ℝ), x ∈ Set.Ioo (π / 6) (π / 3) → 
    ∃ (y : ℝ), y ∈ Set.Ioo (π / 6) (π / 3) ∧ f y < f x) →
  ω = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_sine_function_omega_l4146_414657


namespace NUMINAMATH_CALUDE_snickers_cost_calculation_l4146_414630

/-- The cost of a single piece of Snickers -/
def snickers_cost : ℚ := 1.5

/-- The number of Snickers pieces Julia bought -/
def snickers_count : ℕ := 2

/-- The number of M&M's packs Julia bought -/
def mm_count : ℕ := 3

/-- The cost of a pack of M&M's in terms of Snickers pieces -/
def mm_cost_in_snickers : ℕ := 2

/-- The total amount Julia gave to the cashier -/
def total_paid : ℚ := 20

/-- The change Julia received -/
def change_received : ℚ := 8

theorem snickers_cost_calculation :
  snickers_cost * (snickers_count + mm_count * mm_cost_in_snickers) = total_paid - change_received :=
by sorry

end NUMINAMATH_CALUDE_snickers_cost_calculation_l4146_414630


namespace NUMINAMATH_CALUDE_third_piece_coverage_l4146_414668

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents a piece that covers some number of squares -/
structure Piece :=
  (squares_covered : ℕ)

/-- The theorem stating that if two pieces cover 3 squares in a 4x4 grid, 
    the third piece must cover 13 squares -/
theorem third_piece_coverage 
  (grid : Grid) 
  (piece1 piece2 : Piece) :
  grid.size = 4 →
  piece1.squares_covered = 2 →
  piece2.squares_covered = 1 →
  (∃ (piece3 : Piece), 
    piece1.squares_covered + piece2.squares_covered + piece3.squares_covered = grid.size * grid.size ∧
    piece3.squares_covered = 13) :=
by sorry

end NUMINAMATH_CALUDE_third_piece_coverage_l4146_414668


namespace NUMINAMATH_CALUDE_silly_bills_game_l4146_414644

theorem silly_bills_game (x : ℕ) : 
  x + (x + 11) + (x - 18) > 0 →  -- Ensure positive number of bills
  x + 2 * (x + 11) + 3 * (x - 18) = 100 →
  x = 22 := by
sorry

end NUMINAMATH_CALUDE_silly_bills_game_l4146_414644


namespace NUMINAMATH_CALUDE_vector_AB_l4146_414632

/-- Given points A(1, -1) and B(1, 2), prove that the vector AB is (0, 3) -/
theorem vector_AB (A B : ℝ × ℝ) (hA : A = (1, -1)) (hB : B = (1, 2)) :
  B.1 - A.1 = 0 ∧ B.2 - A.2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_AB_l4146_414632


namespace NUMINAMATH_CALUDE_investment_plans_count_l4146_414623

/-- The number of ways to distribute 3 distinct objects into 4 distinct containers,
    with no container holding more than 2 objects. -/
def investmentPlans : ℕ :=
  let numProjects : ℕ := 3
  let numCities : ℕ := 4
  let maxProjectsPerCity : ℕ := 2
  -- The actual calculation is not implemented here
  60

/-- Theorem stating that the number of investment plans is 60 -/
theorem investment_plans_count : investmentPlans = 60 := by
  sorry

end NUMINAMATH_CALUDE_investment_plans_count_l4146_414623


namespace NUMINAMATH_CALUDE_parallel_transitivity_false_l4146_414681

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the parallel relation between lines
variable (parallel_line_line : Line → Line → Prop)

-- State the theorem
theorem parallel_transitivity_false 
  (l m : Line) (α : Plane) : 
  (parallel_line_plane l α ∧ parallel_line_plane m α) → 
  parallel_line_line l m :=
sorry

end NUMINAMATH_CALUDE_parallel_transitivity_false_l4146_414681


namespace NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l4146_414698

/-- Given a regular pentagon with perimeter 60 inches and a rectangle with perimeter 80 inches
    where the length is twice the width, the ratio of the pentagon's side length to the rectangle's
    width is 9/10. -/
theorem pentagon_rectangle_ratio : 
  ∀ (pentagon_side rectangle_width : ℝ),
    pentagon_side * 5 = 60 →
    rectangle_width * 6 = 80 →
    pentagon_side / rectangle_width = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l4146_414698


namespace NUMINAMATH_CALUDE_correct_num_schedules_l4146_414659

/-- Represents a subject in the school schedule -/
inductive Subject
| Chinese
| Mathematics
| English
| ScienceComprehensive

/-- Represents a class period -/
inductive ClassPeriod
| First
| Second
| Third

/-- A schedule is a function that assigns subjects to class periods -/
def Schedule := ClassPeriod → List Subject

/-- Checks if a schedule is valid according to the problem constraints -/
def isValidSchedule (s : Schedule) : Prop :=
  (∀ subject : Subject, ∃ period : ClassPeriod, subject ∈ s period) ∧
  (∀ period : ClassPeriod, s period ≠ []) ∧
  (∀ period : ClassPeriod, Subject.Mathematics ∈ s period → Subject.ScienceComprehensive ∉ s period) ∧
  (∀ period : ClassPeriod, Subject.ScienceComprehensive ∈ s period → Subject.Mathematics ∉ s period)

/-- The number of valid schedules -/
def numValidSchedules : ℕ := sorry

theorem correct_num_schedules : numValidSchedules = 30 := by sorry

end NUMINAMATH_CALUDE_correct_num_schedules_l4146_414659


namespace NUMINAMATH_CALUDE_remainder_theorem_l4146_414607

theorem remainder_theorem : ∃ q : ℕ, 2^202 + 202 = (2^101 + 2^51 + 1) * q + 201 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l4146_414607


namespace NUMINAMATH_CALUDE_carter_road_trip_l4146_414665

/-- The duration of Carter's road trip without pit stops -/
def road_trip_duration : ℝ := 13.33

/-- The theorem stating the duration of Carter's road trip without pit stops -/
theorem carter_road_trip :
  let stop_interval : ℝ := 2 -- Hours between leg-stretching stops
  let food_stops : ℕ := 2 -- Number of additional food stops
  let gas_stops : ℕ := 3 -- Number of additional gas stops
  let pit_stop_duration : ℝ := 1/3 -- Duration of each pit stop in hours (20 minutes)
  let total_trip_duration : ℝ := 18 -- Total trip duration including pit stops in hours
  
  road_trip_duration = total_trip_duration - 
    (⌊total_trip_duration / stop_interval⌋ + food_stops + gas_stops) * pit_stop_duration :=
by
  sorry

end NUMINAMATH_CALUDE_carter_road_trip_l4146_414665


namespace NUMINAMATH_CALUDE_fred_weekend_earnings_l4146_414619

/-- Fred's earnings from car washing over the weekend -/
def fred_earnings (initial_amount : ℝ) (final_amount : ℝ) (percentage_cars_washed : ℝ) : ℝ :=
  final_amount - initial_amount

/-- Theorem stating Fred's earnings over the weekend -/
theorem fred_weekend_earnings :
  fred_earnings 19 40 0.35 = 21 := by
  sorry

end NUMINAMATH_CALUDE_fred_weekend_earnings_l4146_414619


namespace NUMINAMATH_CALUDE_right_triangle_with_consecutive_sides_l4146_414600

theorem right_triangle_with_consecutive_sides (a b c : ℕ) : 
  a = 11 → b + 1 = c → a^2 + b^2 = c^2 → c = 61 := by sorry

end NUMINAMATH_CALUDE_right_triangle_with_consecutive_sides_l4146_414600


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l4146_414679

/-- A geometric sequence with fifth term 48 and sixth term 72 has second term 384/27 -/
theorem geometric_sequence_second_term :
  ∀ (a : ℚ) (r : ℚ),
  a * r^4 = 48 →
  a * r^5 = 72 →
  a * r = 384/27 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l4146_414679


namespace NUMINAMATH_CALUDE_vasya_always_wins_l4146_414639

/-- Represents a player in the game -/
inductive Player : Type
| Petya : Player
| Vasya : Player

/-- Represents a move in the game -/
inductive Move : Type
| Positive : Move
| Negative : Move

/-- Represents the game state -/
structure GameState :=
(moves : List Move)
(current_player : Player)

/-- The number of divisions on each side of the triangle -/
def n : Nat := 2008

/-- The total number of cells in the triangle -/
def total_cells : Nat := n * n

/-- Determines the winner based on the final game state -/
def winner (final_state : GameState) : Player :=
  sorry

/-- The main theorem stating that Vasya always wins -/
theorem vasya_always_wins :
  ∀ (game : GameState),
  game.moves.length = total_cells →
  game.current_player = Player.Vasya →
  winner game = Player.Vasya :=
sorry

end NUMINAMATH_CALUDE_vasya_always_wins_l4146_414639


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l4146_414673

theorem regular_polygon_sides : ∃ n : ℕ, 
  n > 0 ∧ 
  (360 : ℝ) / n = n - 9 ∧
  n = 24 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l4146_414673


namespace NUMINAMATH_CALUDE_circle_equation_l4146_414640

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point is on the circle if its distance from the center equals the radius -/
def onCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- A circle is tangent to the y-axis if its distance from the y-axis equals its radius -/
def tangentToYAxis (c : Circle) : Prop :=
  |c.center.1| = c.radius

theorem circle_equation (c : Circle) (h : tangentToYAxis c) (h2 : c.center = (-2, 3)) :
  ∀ (x y : ℝ), onCircle c (x, y) ↔ (x + 2)^2 + (y - 3)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l4146_414640


namespace NUMINAMATH_CALUDE_solve_F_equation_l4146_414603

-- Define the function F
def F (a b c : ℚ) : ℚ := a * b^3 + c

-- Theorem statement
theorem solve_F_equation :
  ∃ a : ℚ, F a 3 10 = F a 5 20 ∧ a = -5/49 := by
  sorry

end NUMINAMATH_CALUDE_solve_F_equation_l4146_414603


namespace NUMINAMATH_CALUDE_simple_interest_principal_l4146_414687

/-- Simple interest calculation -/
theorem simple_interest_principal
  (rate : ℝ) (interest : ℝ) (time : ℝ)
  (h_rate : rate = 15)
  (h_interest : interest = 120)
  (h_time : time = 2) :
  (interest * 100) / (rate * time) = 400 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l4146_414687


namespace NUMINAMATH_CALUDE_total_pears_picked_l4146_414691

theorem total_pears_picked (jason_pears keith_pears mike_pears : ℕ)
  (h1 : jason_pears = 46)
  (h2 : keith_pears = 47)
  (h3 : mike_pears = 12) :
  jason_pears + keith_pears + mike_pears = 105 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_picked_l4146_414691


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4146_414608

theorem complex_equation_solution (a : ℝ) : 
  (2 + a * Complex.I) / (1 + Complex.I) = (3 : ℂ) + Complex.I → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4146_414608


namespace NUMINAMATH_CALUDE_fraction_simplification_l4146_414699

theorem fraction_simplification (a b m : ℝ) (hb : b ≠ 0) (hm : m ≠ 0) :
  (a * m) / (b * m) = a / b :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4146_414699


namespace NUMINAMATH_CALUDE_ellipse_equation_l4146_414628

/-- Represents an ellipse with specific properties -/
structure Ellipse where
  /-- The sum of distances from any point on the ellipse to the two foci -/
  focal_distance_sum : ℝ
  /-- The eccentricity of the ellipse -/
  eccentricity : ℝ

/-- Theorem stating the equation of an ellipse with given properties -/
theorem ellipse_equation (e : Ellipse) 
  (h1 : e.focal_distance_sum = 6)
  (h2 : e.eccentricity = 1/3) :
  ∃ (x y : ℝ), x^2/9 + y^2/8 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l4146_414628


namespace NUMINAMATH_CALUDE_julios_age_l4146_414645

/-- Proves that Julio's current age is 36 years old, given the conditions of the problem -/
theorem julios_age (james_age : ℕ) (future_years : ℕ) (julio_age : ℕ) : 
  james_age = 11 →
  future_years = 14 →
  julio_age + future_years = 2 * (james_age + future_years) →
  julio_age = 36 :=
by sorry

end NUMINAMATH_CALUDE_julios_age_l4146_414645


namespace NUMINAMATH_CALUDE_quadratic_equations_solution_l4146_414675

theorem quadratic_equations_solution (m n k : ℝ) : 
  (∃ x : ℝ, m * x^2 + n = 0) ∧
  (∃ x : ℝ, n * x^2 + k = 0) ∧
  (∃ x : ℝ, k * x^2 + m = 0) →
  m = 0 ∧ n = 0 ∧ k = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equations_solution_l4146_414675


namespace NUMINAMATH_CALUDE_valid_numbers_l4146_414611

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  (n / 1000 = (n / 100) % 10) ∧
  ((n / 100) % 10 = n % 10) ∧
  (n ^ 2) % ((n / 1000) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)) = 0

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {1111, 1212, 1515, 2424, 3636} :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l4146_414611


namespace NUMINAMATH_CALUDE_largest_triangle_perimeter_l4146_414692

/-- The largest perimeter of a triangle with two sides of 7 and 8 units, and the third side being an integer --/
theorem largest_triangle_perimeter : 
  ∀ x : ℤ, 
  (7 : ℝ) + 8 > x ∧ 
  (7 : ℝ) + x > 8 ∧ 
  8 + x > 7 →
  (∀ y : ℤ, 
    (7 : ℝ) + 8 > y ∧ 
    (7 : ℝ) + y > 8 ∧ 
    8 + y > 7 →
    7 + 8 + x ≥ 7 + 8 + y) →
  7 + 8 + x = 29 :=
by sorry

end NUMINAMATH_CALUDE_largest_triangle_perimeter_l4146_414692


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l4146_414697

theorem polynomial_division_theorem :
  let dividend : Polynomial ℤ := 4 * X^5 - 5 * X^4 + 3 * X^3 - 7 * X^2 + 6 * X - 1
  let divisor : Polynomial ℤ := X^2 + 2 * X + 3
  let quotient : Polynomial ℤ := 4 * X^3 - 13 * X^2 + 35 * X - 104
  let remainder : Polynomial ℤ := 87
  dividend = divisor * quotient + remainder := by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l4146_414697


namespace NUMINAMATH_CALUDE_factorization_1_l4146_414672

theorem factorization_1 (a b x y : ℝ) : a * (x - y) + b * (y - x) = (a - b) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_1_l4146_414672


namespace NUMINAMATH_CALUDE_article_cost_l4146_414613

/-- 
Given an article that can be sold at two different prices, prove that the cost of the article
is 140 if the higher price yields a 5% greater gain than the lower price.
-/
theorem article_cost (selling_price_high selling_price_low : ℕ) 
  (h1 : selling_price_high = 350)
  (h2 : selling_price_low = 340)
  (h3 : selling_price_high - selling_price_low = 10) :
  ∃ (cost gain : ℕ),
    selling_price_low = cost + gain ∧
    selling_price_high = cost + gain + (gain * 5 / 100) ∧
    cost = 140 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_l4146_414613


namespace NUMINAMATH_CALUDE_described_method_is_analogical_thinking_l4146_414646

/-- A learning method in mathematics -/
structure LearningMethod where
  compare_objects : Bool
  find_similarities : Bool
  deduce_similar_properties : Bool

/-- Analogical thinking in mathematics -/
def analogical_thinking : LearningMethod :=
  { compare_objects := true,
    find_similarities := true,
    deduce_similar_properties := true }

/-- The described learning method -/
def described_method : LearningMethod :=
  { compare_objects := true,
    find_similarities := true,
    deduce_similar_properties := true }

/-- Theorem stating that the described learning method is equivalent to analogical thinking -/
theorem described_method_is_analogical_thinking : described_method = analogical_thinking :=
  sorry

end NUMINAMATH_CALUDE_described_method_is_analogical_thinking_l4146_414646


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l4146_414663

/-- The quadratic equation ax² + 10x + c = 0 has exactly one solution,
    a + c = 12, and a < c. Prove that a = 6 - √11 and c = 6 + √11. -/
theorem unique_quadratic_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 10 * x + c = 0) →
  (a + c = 12) →
  (a < c) →
  (a = 6 - Real.sqrt 11 ∧ c = 6 + Real.sqrt 11) := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l4146_414663


namespace NUMINAMATH_CALUDE_madelines_score_l4146_414605

theorem madelines_score (madeline_mistakes : ℕ) (leo_mistakes : ℕ) (brent_score : ℕ) (brent_mistakes : ℕ) 
  (h1 : madeline_mistakes = 2)
  (h2 : madeline_mistakes * 2 = leo_mistakes)
  (h3 : brent_score = 25)
  (h4 : brent_mistakes = leo_mistakes + 1) :
  30 - madeline_mistakes = 28 := by
  sorry

end NUMINAMATH_CALUDE_madelines_score_l4146_414605


namespace NUMINAMATH_CALUDE_tangent_slope_circle_l4146_414674

/-- Slope of the tangent line to a circle -/
theorem tangent_slope_circle (center_x center_y tangent_x tangent_y : ℝ) :
  center_x = 2 →
  center_y = 3 →
  tangent_x = 7 →
  tangent_y = 8 →
  (tangent_y - center_y) / (tangent_x - center_x) = 1 →
  -(((tangent_y - center_y) / (tangent_x - center_x))⁻¹) = -1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_circle_l4146_414674


namespace NUMINAMATH_CALUDE_intersection_count_l4146_414671

/-- The number of points on the positive x-axis -/
def num_x_points : ℕ := 15

/-- The number of points on the positive y-axis -/
def num_y_points : ℕ := 10

/-- The maximum number of intersection points in the first quadrant -/
def max_intersections : ℕ := (num_x_points.choose 2) * (num_y_points.choose 2)

/-- Theorem stating the maximum number of intersection points -/
theorem intersection_count :
  max_intersections = 4725 := by sorry

end NUMINAMATH_CALUDE_intersection_count_l4146_414671


namespace NUMINAMATH_CALUDE_expand_product_l4146_414622

theorem expand_product (x : ℝ) : 5 * (x - 3) * (x + 6) = 5 * x^2 + 15 * x - 90 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l4146_414622


namespace NUMINAMATH_CALUDE_distance_QR_l4146_414652

-- Define the triangle
def Triangle (D E F : ℝ × ℝ) : Prop :=
  let de := Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2)
  let ef := Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2)
  let df := Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2)
  de = 5 ∧ ef = 12 ∧ df = 13

-- Define the circles
def Circle (Q : ℝ × ℝ) (D E F : ℝ × ℝ) : Prop :=
  let qe := Real.sqrt ((E.1 - Q.1)^2 + (E.2 - Q.2)^2)
  let qd := Real.sqrt ((D.1 - Q.1)^2 + (D.2 - Q.2)^2)
  qe = qd ∧ (E.1 - Q.1) * (F.1 - E.1) + (E.2 - Q.2) * (F.2 - E.2) = 0

def Circle' (R : ℝ × ℝ) (D E F : ℝ × ℝ) : Prop :=
  let rd := Real.sqrt ((D.1 - R.1)^2 + (D.2 - R.2)^2)
  let re := Real.sqrt ((E.1 - R.1)^2 + (E.2 - R.2)^2)
  rd = re ∧ (D.1 - R.1) * (F.1 - D.1) + (D.2 - R.2) * (F.2 - D.2) = 0

-- Theorem statement
theorem distance_QR (D E F Q R : ℝ × ℝ) :
  Triangle D E F →
  Circle Q D E F →
  Circle' R D E F →
  Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2) = 25/12 := by
  sorry

end NUMINAMATH_CALUDE_distance_QR_l4146_414652


namespace NUMINAMATH_CALUDE_cube_face_sum_l4146_414682

theorem cube_face_sum (a b c d e f g h : ℕ+) : 
  (a * b * c + a * e * c + a * b * f + a * e * f + 
   d * b * c + d * e * c + d * b * f + d * e * f) = 2107 →
  a + b + c + d + e + f + g + h = 57 := by
sorry

end NUMINAMATH_CALUDE_cube_face_sum_l4146_414682


namespace NUMINAMATH_CALUDE_folded_paper_properties_l4146_414658

/-- Represents a folded rectangular paper with specific properties -/
structure FoldedPaper where
  short_edge : ℝ
  long_edge : ℝ
  fold_length : ℝ
  congruent_triangles : Prop

/-- Theorem stating the properties of the folded paper -/
theorem folded_paper_properties (paper : FoldedPaper) 
  (h1 : paper.short_edge = 12)
  (h2 : paper.long_edge = 18)
  (h3 : paper.congruent_triangles)
  : paper.fold_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_folded_paper_properties_l4146_414658


namespace NUMINAMATH_CALUDE_uncovered_area_of_overlapping_squares_l4146_414684

theorem uncovered_area_of_overlapping_squares :
  ∀ (large_side small_side : ℝ),
    large_side = 10 →
    small_side = 4 →
    large_side > 0 →
    small_side > 0 →
    large_side ≥ small_side →
    (large_side ^ 2 - small_side ^ 2) = 84 := by
  sorry

end NUMINAMATH_CALUDE_uncovered_area_of_overlapping_squares_l4146_414684


namespace NUMINAMATH_CALUDE_frequency_of_eighth_group_l4146_414612

theorem frequency_of_eighth_group 
  (num_rectangles : ℕ) 
  (sample_size : ℕ) 
  (area_last_rectangle : ℝ) 
  (sum_area_other_rectangles : ℝ) :
  num_rectangles = 8 →
  sample_size = 200 →
  area_last_rectangle = (1/4 : ℝ) * sum_area_other_rectangles →
  (area_last_rectangle / (area_last_rectangle + sum_area_other_rectangles)) * sample_size = 40 :=
by sorry

end NUMINAMATH_CALUDE_frequency_of_eighth_group_l4146_414612


namespace NUMINAMATH_CALUDE_sqrt_three_seven_plus_four_sqrt_three_l4146_414609

theorem sqrt_three_seven_plus_four_sqrt_three :
  Real.sqrt (3 * (7 + 4 * Real.sqrt 3)) = 2 * Real.sqrt 3 + 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_seven_plus_four_sqrt_three_l4146_414609


namespace NUMINAMATH_CALUDE_arithmetic_geometric_relation_l4146_414649

/-- An arithmetic sequence with first term 1 and common difference d -/
def arithmetic_seq (d : ℝ) (n : ℕ) : ℝ := 1 + (n - 1) * d

theorem arithmetic_geometric_relation (d : ℝ) (h : d ≠ 0) :
  (arithmetic_seq d 2) ^ 2 = (arithmetic_seq d 1) * (arithmetic_seq d 5) → d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_relation_l4146_414649


namespace NUMINAMATH_CALUDE_kim_no_tests_probability_l4146_414678

theorem kim_no_tests_probability 
  (p_math : ℝ) 
  (p_history : ℝ) 
  (h_math : p_math = 5/8) 
  (h_history : p_history = 1/3) 
  (h_independent : True)  -- Represents the independence of events
  : 1 - p_math - p_history + p_math * p_history = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_kim_no_tests_probability_l4146_414678


namespace NUMINAMATH_CALUDE_unpainted_cubes_in_four_cube_l4146_414666

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ
  total_cubes : ℕ
  painted_corners : ℕ
  painted_edges : ℕ

/-- Properties of a 4x4x4 cube with painted corners -/
def four_cube : Cube 4 :=
  { side_length := 4
  , total_cubes := 64
  , painted_corners := 8
  , painted_edges := 12 }

/-- The number of unpainted cubes in a cube with painted corners -/
def unpainted_cubes (c : Cube n) : ℕ :=
  c.total_cubes - (c.painted_corners + c.painted_edges)

/-- Theorem: The number of unpainted cubes in a 4x4x4 cube with painted corners is 44 -/
theorem unpainted_cubes_in_four_cube :
  unpainted_cubes four_cube = 44 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_in_four_cube_l4146_414666


namespace NUMINAMATH_CALUDE_max_sum_distances_in_unit_square_l4146_414636

theorem max_sum_distances_in_unit_square :
  ∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 →
  let PA := Real.sqrt (x^2 + y^2)
  let PB := Real.sqrt ((1-x)^2 + y^2)
  let PC := Real.sqrt ((1-x)^2 + (1-y)^2)
  let PD := Real.sqrt (x^2 + (1-y)^2)
  PA + PB + PC + PD ≤ 2 + Real.sqrt 2 ∧
  ∃ (x₀ y₀ : ℝ), 0 ≤ x₀ ∧ x₀ ≤ 1 ∧ 0 ≤ y₀ ∧ y₀ ≤ 1 ∧
    let PA₀ := Real.sqrt (x₀^2 + y₀^2)
    let PB₀ := Real.sqrt ((1-x₀)^2 + y₀^2)
    let PC₀ := Real.sqrt ((1-x₀)^2 + (1-y₀)^2)
    let PD₀ := Real.sqrt (x₀^2 + (1-y₀)^2)
    PA₀ + PB₀ + PC₀ + PD₀ = 2 + Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_distances_in_unit_square_l4146_414636


namespace NUMINAMATH_CALUDE_emily_walks_farther_l4146_414620

/-- The distance Troy walks to school (in meters) -/
def troy_distance : ℕ := 75

/-- The distance Emily walks to school (in meters) -/
def emily_distance : ℕ := 98

/-- The number of days -/
def days : ℕ := 5

/-- The additional distance Emily walks compared to Troy over the given number of days -/
def additional_distance : ℕ := 
  days * (2 * emily_distance - 2 * troy_distance)

theorem emily_walks_farther : additional_distance = 230 := by
  sorry

end NUMINAMATH_CALUDE_emily_walks_farther_l4146_414620


namespace NUMINAMATH_CALUDE_expression_value_l4146_414614

theorem expression_value : 
  (2024^3 - 2 * 2024^2 * 2025 + 3 * 2024 * 2025^2 - 2025^3 + 4) / (2024 * 2025) = 2022 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4146_414614


namespace NUMINAMATH_CALUDE_cone_apex_angle_l4146_414616

theorem cone_apex_angle (r : ℝ) (h : ℝ) (l : ℝ) (θ : ℝ) : 
  r > 0 → h > 0 → l > 0 →
  l = 2 * r →  -- ratio of lateral area to base area is 2
  h = r * Real.sqrt 3 →  -- derived from Pythagorean theorem
  θ = 2 * Real.arctan (1 / Real.sqrt 3) →  -- definition of apex angle
  θ = π / 3  -- 60 degrees in radians
:= by sorry

end NUMINAMATH_CALUDE_cone_apex_angle_l4146_414616


namespace NUMINAMATH_CALUDE_ball_probability_l4146_414680

theorem ball_probability (total white green yellow red purple black blue pink : ℕ) 
  (h_total : total = 500)
  (h_white : white = 200)
  (h_green : green = 80)
  (h_yellow : yellow = 70)
  (h_red : red = 57)
  (h_purple : purple = 33)
  (h_black : black = 30)
  (h_blue : blue = 16)
  (h_pink : pink = 14)
  (h_sum : white + green + yellow + red + purple + black + blue + pink = total) :
  (total - (red + purple + black)) / total = 19 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l4146_414680


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l4146_414629

theorem max_value_on_ellipse :
  ∃ (max : ℝ), max = 2 * Real.sqrt 10 ∧
  (∀ x y : ℝ, x^2/9 + y^2/4 = 1 → 2*x - y ≤ max) ∧
  (∃ x y : ℝ, x^2/9 + y^2/4 = 1 ∧ 2*x - y = max) := by
sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l4146_414629


namespace NUMINAMATH_CALUDE_odd_function_property_l4146_414627

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) (h_odd : is_odd_function f) 
  (h_neg : ∀ x < 0, f x = x * (1 + x)) : 
  ∀ x > 0, f x = x * (1 - x) := by
sorry

end NUMINAMATH_CALUDE_odd_function_property_l4146_414627


namespace NUMINAMATH_CALUDE_line_intersection_x_axis_l4146_414624

/-- A line passing through two points (2, 3) and (6, 7) intersects the x-axis at (-1, 0). -/
theorem line_intersection_x_axis :
  let line := (fun x => x + 1)  -- Define the line equation y = x + 1
  ∀ x y : ℝ,
    (x = 2 ∧ y = 3) ∨ (x = 6 ∧ y = 7) →  -- The line passes through (2, 3) and (6, 7)
    y = line x →  -- The point (x, y) is on the line
    (line (-1) = 0)  -- The line intersects the x-axis at x = -1
    ∧ (∀ t : ℝ, t ≠ -1 → line t ≠ 0)  -- The intersection point is unique
    := by sorry

end NUMINAMATH_CALUDE_line_intersection_x_axis_l4146_414624


namespace NUMINAMATH_CALUDE_andrea_sod_rectangles_l4146_414602

/-- Calculates the number of sod rectangles needed for a given area -/
def sodRectanglesNeeded (length width : ℕ) : ℕ :=
  (length * width + 11) / 12

/-- The total number of sod rectangles needed for Andrea's backyard -/
def totalSodRectangles : ℕ :=
  sodRectanglesNeeded 35 42 +
  sodRectanglesNeeded 55 86 +
  sodRectanglesNeeded 20 50 +
  sodRectanglesNeeded 48 66

theorem andrea_sod_rectangles :
  totalSodRectangles = 866 := by
  sorry

end NUMINAMATH_CALUDE_andrea_sod_rectangles_l4146_414602


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l4146_414618

theorem complex_magnitude_problem (m : ℂ) :
  (((4 : ℂ) + m * Complex.I) / ((1 : ℂ) + 2 * Complex.I)).im = 0 →
  Complex.abs (m + 6 * Complex.I) = 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l4146_414618


namespace NUMINAMATH_CALUDE_reduce_piles_to_zero_reduce_table_to_zero_l4146_414686

/-- Represents the state of three piles of stones -/
structure ThreePiles :=
  (pile1 pile2 pile3 : Nat)

/-- Represents the state of an 8x5 table of natural numbers -/
def Table := Fin 8 → Fin 5 → Nat

/-- Allowed operations on three piles of stones -/
inductive PileOperation
  | removeOne : PileOperation
  | doubleOne : Fin 3 → PileOperation

/-- Allowed operations on the table -/
inductive TableOperation
  | doubleColumn : Fin 5 → TableOperation
  | subtractRow : Fin 8 → TableOperation

/-- Applies a pile operation to a ThreePiles state -/
def applyPileOp (s : ThreePiles) (op : PileOperation) : ThreePiles :=
  match op with
  | PileOperation.removeOne => ⟨s.pile1 - 1, s.pile2 - 1, s.pile3 - 1⟩
  | PileOperation.doubleOne i =>
      match i with
      | 0 => ⟨s.pile1 * 2, s.pile2, s.pile3⟩
      | 1 => ⟨s.pile1, s.pile2 * 2, s.pile3⟩
      | 2 => ⟨s.pile1, s.pile2, s.pile3 * 2⟩

/-- Applies a table operation to a Table state -/
def applyTableOp (t : Table) (op : TableOperation) : Table :=
  match op with
  | TableOperation.doubleColumn j => fun i k => if k = j then t i k * 2 else t i k
  | TableOperation.subtractRow i => fun j k => if j = i then t j k - 1 else t j k

/-- Theorem stating that any ThreePiles state can be reduced to zero -/
theorem reduce_piles_to_zero (s : ThreePiles) :
  ∃ (ops : List PileOperation), (ops.foldl applyPileOp s).pile1 = 0 ∧
                                (ops.foldl applyPileOp s).pile2 = 0 ∧
                                (ops.foldl applyPileOp s).pile3 = 0 :=
  sorry

/-- Theorem stating that any Table state can be reduced to zero -/
theorem reduce_table_to_zero (t : Table) :
  ∃ (ops : List TableOperation), ∀ i j, (ops.foldl applyTableOp t) i j = 0 :=
  sorry

end NUMINAMATH_CALUDE_reduce_piles_to_zero_reduce_table_to_zero_l4146_414686


namespace NUMINAMATH_CALUDE_crayon_count_l4146_414683

theorem crayon_count (small_left medium_left large_left : ℕ) 
  (h_small : small_left = 60)
  (h_medium : medium_left = 98)
  (h_large : large_left = 168) :
  ∃ (small_initial medium_initial large_initial : ℕ),
    small_initial = 100 ∧
    medium_initial = 392 ∧
    large_initial = 294 ∧
    small_left = (3 : ℚ) / 5 * small_initial ∧
    medium_left = (1 : ℚ) / 4 * medium_initial ∧
    large_left = (4 : ℚ) / 7 * large_initial ∧
    (2 : ℚ) / 5 * small_initial + 
    (3 : ℚ) / 4 * medium_initial + 
    (3 : ℚ) / 7 * large_initial = 460 := by
  sorry


end NUMINAMATH_CALUDE_crayon_count_l4146_414683


namespace NUMINAMATH_CALUDE_unique_solution_l4146_414633

/-- A function satisfying the given conditions -/
def SatisfiesConditions (f : ℝ → ℝ) : Prop :=
  (∀ x, x > 0 → f x > 0) ∧ 
  (∀ x y, x > 0 → y > 0 → f (x * f y) = y * f x) ∧
  (Filter.Tendsto f Filter.atTop (nhds 0))

/-- The theorem stating that the function f(x) = 1/x is the unique solution -/
theorem unique_solution (f : ℝ → ℝ) (h : SatisfiesConditions f) : 
  ∀ x, x > 0 → f x = 1 / x := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l4146_414633


namespace NUMINAMATH_CALUDE_system_solution_ratio_l4146_414626

theorem system_solution_ratio (a b x y : ℝ) (h1 : 4 * x - 2 * y = a) 
  (h2 : 9 * y - 18 * x = b) (h3 : b ≠ 0) : a / b = -2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l4146_414626


namespace NUMINAMATH_CALUDE_equation_solution_l4146_414635

theorem equation_solution :
  let f : ℝ → ℝ := fun x => 0.05 * x + 0.09 * (30 + x)
  ∃! x : ℝ, f x = 15.3 - 3.3 ∧ x = 465 / 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4146_414635


namespace NUMINAMATH_CALUDE_problem_solution_l4146_414625

theorem problem_solution :
  (∃ x : ℝ, x > 0 ∧ x + 4/x = 6) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2*y = 1 → 2/x + 1/y ≥ 8) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4146_414625


namespace NUMINAMATH_CALUDE_smallest_gcd_multiple_l4146_414610

theorem smallest_gcd_multiple (m n : ℕ+) (h : Nat.gcd m n = 10) :
  (∃ (m' n' : ℕ+), Nat.gcd m' n' = 10 ∧ Nat.gcd (8 * m') (12 * n') = 40) ∧
  (∀ (m'' n'' : ℕ+), Nat.gcd m'' n'' = 10 → Nat.gcd (8 * m'') (12 * n'') ≥ 40) :=
sorry

end NUMINAMATH_CALUDE_smallest_gcd_multiple_l4146_414610


namespace NUMINAMATH_CALUDE_original_number_is_509_l4146_414695

theorem original_number_is_509 (subtracted_number : ℕ) : 
  (509 - subtracted_number) % 9 = 0 →
  subtracted_number ≥ 5 →
  ∀ n < subtracted_number, (509 - n) % 9 ≠ 0 →
  509 = 509 :=
by
  sorry

end NUMINAMATH_CALUDE_original_number_is_509_l4146_414695


namespace NUMINAMATH_CALUDE_unsatisfactory_tests_l4146_414677

theorem unsatisfactory_tests (n : ℕ) (k : ℕ) : 
  n < 50 →
  n % 7 = 0 →
  n % 3 = 0 →
  n % 2 = 0 →
  n / 7 + n / 3 + n / 2 + k = n →
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_unsatisfactory_tests_l4146_414677


namespace NUMINAMATH_CALUDE_line_through_5_2_slope_neg1_l4146_414638

/-- The point-slope form equation of a line passing through a given point with a given slope -/
def point_slope_form (x₀ y₀ m : ℝ) (x y : ℝ) : Prop :=
  y - y₀ = m * (x - x₀)

/-- Theorem: The point-slope form equation of the line passing through (5, 2) with slope -1 -/
theorem line_through_5_2_slope_neg1 (x y : ℝ) :
  point_slope_form 5 2 (-1) x y ↔ y - 2 = -(x - 5) :=
by sorry

end NUMINAMATH_CALUDE_line_through_5_2_slope_neg1_l4146_414638


namespace NUMINAMATH_CALUDE_bus_seating_capacity_l4146_414694

theorem bus_seating_capacity : 
  let left_seats : ℕ := 15
  let right_seats : ℕ := left_seats - 3
  let people_per_seat : ℕ := 3
  let back_seat_capacity : ℕ := 11
  
  left_seats * people_per_seat + right_seats * people_per_seat + back_seat_capacity = 92 :=
by sorry

end NUMINAMATH_CALUDE_bus_seating_capacity_l4146_414694


namespace NUMINAMATH_CALUDE_sum_smallest_largest_prime_1_to_50_l4146_414617

theorem sum_smallest_largest_prime_1_to_50 : 
  ∃ (p q : ℕ), 
    p.Prime ∧ q.Prime ∧ 
    1 < p ∧ p ≤ 50 ∧ 
    1 < q ∧ q ≤ 50 ∧ 
    (∀ r : ℕ, r.Prime → 1 < r → r ≤ 50 → p ≤ r ∧ r ≤ q) ∧ 
    p + q = 49 :=
by sorry

end NUMINAMATH_CALUDE_sum_smallest_largest_prime_1_to_50_l4146_414617


namespace NUMINAMATH_CALUDE_translation_property_l4146_414653

-- Define a translation as a function from ℂ to ℂ
def Translation := ℂ → ℂ

-- Define the property of a translation taking one point to another
def TranslatesTo (T : Translation) (z w : ℂ) : Prop := T z = w

theorem translation_property (T : Translation) :
  TranslatesTo T (1 - 2*I) (4 + 3*I) →
  TranslatesTo T (2 + 4*I) (5 + 9*I) := by
  sorry

end NUMINAMATH_CALUDE_translation_property_l4146_414653


namespace NUMINAMATH_CALUDE_triangle_problem_l4146_414685

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  a > b →
  a = 5 →
  c = 6 →
  Real.sin B = 3/5 →
  (b = Real.sqrt 13 ∧ 
   Real.sin A = (3 * Real.sqrt 13) / 13) ∧
  Real.sin (2*A + π/4) = (7 * Real.sqrt 2) / 26 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l4146_414685


namespace NUMINAMATH_CALUDE_at_op_difference_l4146_414641

-- Define the @ operation
def at_op (x y : ℤ) : ℤ := x * y - y * x - 3 * x + 2 * y

-- State the theorem
theorem at_op_difference : at_op 9 5 - at_op 5 9 = -20 := by
  sorry

end NUMINAMATH_CALUDE_at_op_difference_l4146_414641


namespace NUMINAMATH_CALUDE_beatrix_books_l4146_414664

theorem beatrix_books (beatrix alannah queen : ℕ) 
  (h1 : alannah = beatrix + 20)
  (h2 : queen = alannah + alannah / 5)
  (h3 : beatrix + alannah + queen = 140) : 
  beatrix = 30 := by
  sorry

end NUMINAMATH_CALUDE_beatrix_books_l4146_414664


namespace NUMINAMATH_CALUDE_min_value_of_a_l4146_414670

theorem min_value_of_a (a b c : ℝ) (ha : a > 0) (hroots : ∃ x y : ℝ, 0 < x ∧ x < 2 ∧ 0 < y ∧ y < 2 ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) (hineq : ∀ c' : ℝ, c' ≥ 1 → 25 * a + 10 * b + 4 * c' ≥ 4) : a ≥ 16/25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_a_l4146_414670


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l4146_414654

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ y : ℝ, (3 - 5*I) * (a + b*I) = y*I) : a/b = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l4146_414654


namespace NUMINAMATH_CALUDE_rush_order_cost_rush_order_cost_is_five_l4146_414661

/-- Calculate the extra amount paid for a rush order given the following conditions:
  * There are 4 people ordering dinner
  * Each main meal costs $12.0
  * 2 appetizers are ordered at $6.00 each
  * A 20% tip is included
  * The total amount spent is $77
-/
theorem rush_order_cost (num_people : ℕ) (main_meal_cost : ℚ) (num_appetizers : ℕ) 
  (appetizer_cost : ℚ) (tip_percentage : ℚ) (total_spent : ℚ) : ℚ :=
  let subtotal := num_people * main_meal_cost + num_appetizers * appetizer_cost
  let tip := subtotal * tip_percentage
  let total_before_rush := subtotal + tip
  total_spent - total_before_rush

/-- The extra amount paid for the rush order is $5.0 -/
theorem rush_order_cost_is_five : 
  rush_order_cost 4 12 2 6 (1/5) 77 = 5 := by
  sorry

end NUMINAMATH_CALUDE_rush_order_cost_rush_order_cost_is_five_l4146_414661


namespace NUMINAMATH_CALUDE_total_goats_is_320_l4146_414642

/-- The number of goats Washington has -/
def washington_goats : ℕ := 140

/-- The number of additional goats Paddington has compared to Washington -/
def additional_goats : ℕ := 40

/-- The total number of goats Paddington and Washington have together -/
def total_goats : ℕ := washington_goats + (washington_goats + additional_goats)

/-- Theorem stating the total number of goats is 320 -/
theorem total_goats_is_320 : total_goats = 320 := by
  sorry

end NUMINAMATH_CALUDE_total_goats_is_320_l4146_414642


namespace NUMINAMATH_CALUDE_saree_price_calculation_l4146_414689

theorem saree_price_calculation (final_price : ℝ) : 
  final_price = 248.625 → 
  ∃ (original_price : ℝ), 
    original_price * (1 - 0.15) * (1 - 0.25) = final_price ∧ 
    original_price = 390 := by
  sorry

end NUMINAMATH_CALUDE_saree_price_calculation_l4146_414689


namespace NUMINAMATH_CALUDE_evaluate_expression_l4146_414676

theorem evaluate_expression :
  (2^1501 + 5^1502)^2 - (2^1501 - 5^1502)^2 = 20 * 10^1501 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l4146_414676


namespace NUMINAMATH_CALUDE_select_defective_theorem_l4146_414648

/-- The number of ways to select at least 2 defective products -/
def select_defective (total : ℕ) (defective : ℕ) (selected : ℕ) : ℕ :=
  Nat.choose defective 2 * Nat.choose (total - defective) (selected - 2) +
  Nat.choose defective 3 * Nat.choose (total - defective) (selected - 3)

/-- Theorem stating the number of ways to select at least 2 defective products
    from 5 randomly selected products out of 200 total products with 3 defective products -/
theorem select_defective_theorem :
  select_defective 200 3 5 = Nat.choose 3 2 * Nat.choose 197 3 + Nat.choose 3 3 * Nat.choose 197 2 := by
  sorry

end NUMINAMATH_CALUDE_select_defective_theorem_l4146_414648


namespace NUMINAMATH_CALUDE_opposite_leg_length_l4146_414637

/-- Represents a right triangle with a 30° angle -/
structure RightTriangle30 where
  /-- Length of the hypotenuse -/
  hypotenuse : ℝ
  /-- Length of the leg opposite to the 30° angle -/
  opposite_leg : ℝ
  /-- Constraint that the hypotenuse is twice the opposite leg -/
  hyp_constraint : hypotenuse = 2 * opposite_leg

/-- 
Theorem: In a right triangle with a 30° angle and hypotenuse of 18 inches, 
the leg opposite to the 30° angle is 9 inches long.
-/
theorem opposite_leg_length (triangle : RightTriangle30) 
  (h : triangle.hypotenuse = 18) : triangle.opposite_leg = 9 := by
  sorry

end NUMINAMATH_CALUDE_opposite_leg_length_l4146_414637


namespace NUMINAMATH_CALUDE_garden_breadth_l4146_414655

/-- Given a rectangular garden with perimeter 680 m and length 258 m, its breadth is 82 m -/
theorem garden_breadth (perimeter length breadth : ℝ) : 
  perimeter = 680 ∧ length = 258 ∧ perimeter = 2 * (length + breadth) → breadth = 82 := by
  sorry

end NUMINAMATH_CALUDE_garden_breadth_l4146_414655


namespace NUMINAMATH_CALUDE_det_sin_matrix_zero_l4146_414669

theorem det_sin_matrix_zero :
  let A : Matrix (Fin 3) (Fin 3) ℝ := λ i j =>
    match i, j with
    | 0, 0 => Real.sin 1
    | 0, 1 => Real.sin 2
    | 0, 2 => Real.sin 3
    | 1, 0 => Real.sin 4
    | 1, 1 => Real.sin 5
    | 1, 2 => Real.sin 6
    | 2, 0 => Real.sin 7
    | 2, 1 => Real.sin 8
    | 2, 2 => Real.sin 9
  Matrix.det A = 0 :=
by sorry

end NUMINAMATH_CALUDE_det_sin_matrix_zero_l4146_414669


namespace NUMINAMATH_CALUDE_courier_strategy_l4146_414690

-- Define the probability of a courier being robbed
variable (p : ℝ) 

-- Define the probability of failure for each strategy
def P2 : ℝ := p^2
def P3 : ℝ := p^2 * (3 - 2*p)
def P4 : ℝ := p^3 * (4 - 3*p)

-- Define the theorem
theorem courier_strategy (h1 : 0 < p) (h2 : p < 1) :
  (0 < p ∧ p < 1/3 → 1 - P4 p > max (1 - P2 p) (1 - P3 p)) ∧
  (1/3 ≤ p ∧ p < 1 → 1 - P2 p ≥ max (1 - P3 p) (1 - P4 p)) :=
sorry

end NUMINAMATH_CALUDE_courier_strategy_l4146_414690


namespace NUMINAMATH_CALUDE_minimize_quadratic_l4146_414656

/-- The quadratic function f(x) = x^2 + 8x + 7 -/
def f (x : ℝ) : ℝ := x^2 + 8*x + 7

/-- Theorem stating that -4 minimizes the quadratic function f(x) = x^2 + 8x + 7 for all real x -/
theorem minimize_quadratic :
  ∀ x : ℝ, f (-4) ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_minimize_quadratic_l4146_414656
