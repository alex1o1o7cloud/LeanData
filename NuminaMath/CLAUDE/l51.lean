import Mathlib

namespace NUMINAMATH_CALUDE_smallest_rectangle_area_is_768_l51_5126

/-- The side length of each square in centimeters -/
def square_side : ℝ := 8

/-- The number of squares in the height of the L-shape -/
def height_squares : ℕ := 3

/-- The number of squares in the width of the L-shape -/
def width_squares : ℕ := 4

/-- The height of the L-shape in centimeters -/
def l_shape_height : ℝ := square_side * height_squares

/-- The width of the L-shape in centimeters -/
def l_shape_width : ℝ := square_side * width_squares

/-- The smallest possible area of a rectangle that can completely contain the L-shape -/
def smallest_rectangle_area : ℝ := l_shape_height * l_shape_width

theorem smallest_rectangle_area_is_768 : smallest_rectangle_area = 768 := by
  sorry

end NUMINAMATH_CALUDE_smallest_rectangle_area_is_768_l51_5126


namespace NUMINAMATH_CALUDE_f_of_two_equals_five_l51_5196

-- Define the function f
def f : ℝ → ℝ := fun x ↦ 2 * (x - 1) + 3

-- State the theorem
theorem f_of_two_equals_five : f 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_of_two_equals_five_l51_5196


namespace NUMINAMATH_CALUDE_angle_D_measure_l51_5178

/-- 
Given a geometric figure with angles A, B, C, D, and E, where:
1. The sum of angles A and B is 140 degrees
2. Angle C is equal to angle D
3. The sum of angles C, D, and E is 180 degrees

This theorem proves that the measure of angle D is 20 degrees.
-/
theorem angle_D_measure (A B C D E : ℝ) 
  (sum_AB : A + B = 140)
  (C_eq_D : C = D)
  (sum_CDE : C + D + E = 180) :
  D = 20 := by sorry

end NUMINAMATH_CALUDE_angle_D_measure_l51_5178


namespace NUMINAMATH_CALUDE_same_terminal_side_angles_l51_5185

theorem same_terminal_side_angles (π : ℝ) : 
  {β : ℝ | ∃ k : ℤ, β = π / 3 + 2 * k * π ∧ -2 * π ≤ β ∧ β < 4 * π} = 
  {-5 * π / 3, π / 3, 7 * π / 3} := by sorry

end NUMINAMATH_CALUDE_same_terminal_side_angles_l51_5185


namespace NUMINAMATH_CALUDE_quadratic_equation_real_root_l51_5139

theorem quadratic_equation_real_root (p : ℝ) : 
  ((-p)^2 - 4 * (3*(p+2)) * (-(4*p+7))) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_root_l51_5139


namespace NUMINAMATH_CALUDE_binary_101_equals_5_l51_5160

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101_equals_5 :
  binary_to_decimal [true, false, true] = 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_101_equals_5_l51_5160


namespace NUMINAMATH_CALUDE_max_value_m_l51_5109

theorem max_value_m (a b m : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ a b, (3/a + 1/b ≥ m/(a + 3*b))) → m ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_max_value_m_l51_5109


namespace NUMINAMATH_CALUDE_weighted_average_score_l51_5163

-- Define the scores and weights
def interview_score : ℕ := 90
def computer_score : ℕ := 85
def design_score : ℕ := 80

def interview_weight : ℕ := 5
def computer_weight : ℕ := 2
def design_weight : ℕ := 3

-- Define the total weighted score
def total_weighted_score : ℕ := 
  interview_score * interview_weight + 
  computer_score * computer_weight + 
  design_score * design_weight

-- Define the sum of weights
def sum_of_weights : ℕ := 
  interview_weight + computer_weight + design_weight

-- Theorem to prove
theorem weighted_average_score : 
  total_weighted_score / sum_of_weights = 86 := by
  sorry

end NUMINAMATH_CALUDE_weighted_average_score_l51_5163


namespace NUMINAMATH_CALUDE_equation_solution_l51_5114

theorem equation_solution : 
  ∃! x : ℝ, (x - 60) / 3 = (4 - 3*x) / 6 ∧ x = 24.8 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l51_5114


namespace NUMINAMATH_CALUDE_boat_journey_distance_l51_5103

/-- Calculates the total distance covered by a man rowing a boat in a river with varying currents. -/
theorem boat_journey_distance
  (man_speed : ℝ)
  (current1_speed : ℝ)
  (current1_time : ℝ)
  (current2_speed : ℝ)
  (current2_time : ℝ)
  (h1 : man_speed = 15)
  (h2 : current1_speed = 2.5)
  (h3 : current1_time = 2)
  (h4 : current2_speed = 3)
  (h5 : current2_time = 1.5) :
  (man_speed + current1_speed) * current1_time +
  (man_speed - current2_speed) * current2_time = 53 := by
sorry


end NUMINAMATH_CALUDE_boat_journey_distance_l51_5103


namespace NUMINAMATH_CALUDE_bill_calculation_l51_5164

def original_bill : ℝ := 500

def first_late_charge_rate : ℝ := 0.02

def second_late_charge_rate : ℝ := 0.03

def final_bill_amount : ℝ := original_bill * (1 + first_late_charge_rate) * (1 + second_late_charge_rate)

theorem bill_calculation :
  final_bill_amount = 525.30 := by
  sorry

end NUMINAMATH_CALUDE_bill_calculation_l51_5164


namespace NUMINAMATH_CALUDE_impossible_to_reach_all_threes_l51_5173

/-- A game state consisting of a list of stone piles --/
structure GameState where
  piles : List Nat

/-- A move in the game --/
inductive Move where
  | remove_and_split (i j : Nat) : Move

/-- Apply a move to a game state --/
def apply_move (state : GameState) (move : Move) : GameState :=
  sorry

/-- The initial game state with a single pile of 1001 stones --/
def initial_state : GameState :=
  { piles := [1001] }

/-- A predicate that checks if all piles in a game state contain exactly three stones --/
def all_piles_three (state : GameState) : Prop :=
  sorry

/-- The main theorem stating that it's impossible to reach a state with only piles of three stones --/
theorem impossible_to_reach_all_threes :
  ¬∃ (moves : List Move), all_piles_three (moves.foldl apply_move initial_state) :=
  sorry

end NUMINAMATH_CALUDE_impossible_to_reach_all_threes_l51_5173


namespace NUMINAMATH_CALUDE_hyperbola_canonical_equation_l51_5108

/-- Represents a hyperbola with foci on the x-axis -/
structure Hyperbola where
  ε : ℝ  -- eccentricity
  f : ℝ  -- focal distance

/-- The canonical equation of a hyperbola -/
def canonical_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1

/-- Theorem: For a hyperbola with ε = 1.5 and focal distance 6, the canonical equation is x²/4 - y²/5 = 1 -/
theorem hyperbola_canonical_equation (h : Hyperbola) 
    (h_ε : h.ε = 1.5) 
    (h_f : h.f = 6) :
    ∀ x y : ℝ, canonical_equation h x y :=
  sorry

end NUMINAMATH_CALUDE_hyperbola_canonical_equation_l51_5108


namespace NUMINAMATH_CALUDE_log_powers_sum_l51_5121

theorem log_powers_sum (a b : ℝ) (ha : a = Real.log 25) (hb : b = Real.log 49) :
  (5 : ℝ) ^ (a / b) + (7 : ℝ) ^ (b / a) = 12 := by
  sorry

end NUMINAMATH_CALUDE_log_powers_sum_l51_5121


namespace NUMINAMATH_CALUDE_grass_seed_price_five_pound_bag_price_l51_5143

/-- Represents the price of a bag of grass seed -/
structure BagPrice where
  weight : ℕ
  price : ℚ

/-- Represents the customer's purchase -/
structure Purchase where
  bags5lb : ℕ
  bags10lb : ℕ
  bags25lb : ℕ

def total_weight (p : Purchase) : ℕ :=
  5 * p.bags5lb + 10 * p.bags10lb + 25 * p.bags25lb

def total_cost (p : Purchase) (price5lb : ℚ) : ℚ :=
  price5lb * p.bags5lb + 20.42 * p.bags10lb + 32.25 * p.bags25lb

def is_valid_purchase (p : Purchase) : Prop :=
  65 ≤ total_weight p ∧ total_weight p ≤ 80

theorem grass_seed_price (price5lb : ℚ) : Prop :=
  ∃ (p : Purchase),
    is_valid_purchase p ∧
    total_cost p price5lb = 98.77 ∧
    ∀ (q : Purchase), is_valid_purchase q → total_cost q price5lb ≥ 98.77 →
    price5lb = 2.02

/-- The main theorem stating that the price of the 5-pound bag is $2.02 -/
theorem five_pound_bag_price : ∃ (price5lb : ℚ), grass_seed_price price5lb :=
  sorry

end NUMINAMATH_CALUDE_grass_seed_price_five_pound_bag_price_l51_5143


namespace NUMINAMATH_CALUDE_solution_set_implies_a_values_l51_5181

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

-- State the theorem
theorem solution_set_implies_a_values :
  (∀ x : ℝ, f a x ≤ 0 ↔ x ≤ -1) →
  (a = 2 ∨ a = -4) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_values_l51_5181


namespace NUMINAMATH_CALUDE_caterer_sundae_order_l51_5179

/-- Represents the problem of determining the number of sundaes ordered by a caterer --/
theorem caterer_sundae_order (total_price : ℚ) (ice_cream_bars : ℕ) (ice_cream_price : ℚ) (sundae_price : ℚ)
  (h1 : total_price = 200)
  (h2 : ice_cream_bars = 225)
  (h3 : ice_cream_price = 60/100)
  (h4 : sundae_price = 52/100) :
  ∃ (sundaes : ℕ), sundaes = 125 ∧ total_price = ice_cream_bars * ice_cream_price + sundaes * sundae_price :=
by sorry

end NUMINAMATH_CALUDE_caterer_sundae_order_l51_5179


namespace NUMINAMATH_CALUDE_player_A_wins_l51_5167

/-- Represents a player in the game -/
inductive Player : Type
| A : Player
| B : Player

/-- Represents the state of the blackboard -/
def BoardState : Type := List Nat

/-- Checks if a move is valid according to the game rules -/
def isValidMove (current : BoardState) (next : BoardState) : Prop :=
  sorry

/-- Represents a strategy for playing the game -/
def Strategy : Type := BoardState → BoardState

/-- Checks if a strategy is winning for a given player -/
def isWinningStrategy (player : Player) (strat : Strategy) : Prop :=
  sorry

/-- The initial state of the board -/
def initialState : BoardState := [10^2007]

/-- The main theorem stating that Player A has a winning strategy -/
theorem player_A_wins :
  ∃ (strat : Strategy), isWinningStrategy Player.A strat :=
sorry

end NUMINAMATH_CALUDE_player_A_wins_l51_5167


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l51_5180

theorem simplify_sqrt_expression : 
  (Real.sqrt 450 / Real.sqrt 200) + (Real.sqrt 392 / Real.sqrt 98) = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l51_5180


namespace NUMINAMATH_CALUDE_condition_property_l51_5193

theorem condition_property (x y : ℝ) :
  (∀ x y : ℝ, x ≥ 1 ∧ y ≥ 1 → x^2 + y^2 ≥ 2) ∧
  (∃ x y : ℝ, x^2 + y^2 ≥ 2 ∧ ¬(x ≥ 1 ∧ y ≥ 1)) :=
by sorry

end NUMINAMATH_CALUDE_condition_property_l51_5193


namespace NUMINAMATH_CALUDE_equation_solution_set_l51_5172

theorem equation_solution_set : 
  let f : ℝ → ℝ := λ x => 1 / (x^2 + 13*x - 12) + 1 / (x^2 + 4*x - 12) + 1 / (x^2 - 15*x - 12)
  {x : ℝ | f x = 0} = {1, -12, 12, -1} := by
sorry

end NUMINAMATH_CALUDE_equation_solution_set_l51_5172


namespace NUMINAMATH_CALUDE_g_satisfies_conditions_g_value_at_neg_two_l51_5183

/-- A cubic polynomial -/
def CubicPolynomial (α : Type*) [Field α] := α → α

/-- The polynomial f(x) = x^3 - 2x^2 + 5 -/
def f : CubicPolynomial ℝ := λ x ↦ x^3 - 2*x^2 + 5

/-- The polynomial g, which is defined by the problem conditions -/
noncomputable def g : CubicPolynomial ℝ := sorry

/-- The roots of f -/
noncomputable def roots_f : Finset ℝ := sorry

theorem g_satisfies_conditions :
  (g 0 = 1) ∧ 
  (∀ r ∈ roots_f, ∃ s, g s = 0 ∧ s = r^2) ∧
  (∀ s, g s = 0 → ∃ r ∈ roots_f, s = r^2) := sorry

theorem g_value_at_neg_two : g (-2) = 24.2 := sorry

end NUMINAMATH_CALUDE_g_satisfies_conditions_g_value_at_neg_two_l51_5183


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l51_5166

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_geo : is_geometric_sequence a)
    (h_a1 : a 1 = 8) (h_a4 : a 4 = 64) : 
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l51_5166


namespace NUMINAMATH_CALUDE_mp3_song_count_l51_5137

theorem mp3_song_count (x y : ℕ) : 
  (15 : ℕ) - x + y = 2 * 15 → y = x + 15 := by
sorry

end NUMINAMATH_CALUDE_mp3_song_count_l51_5137


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l51_5112

theorem simplify_sqrt_expression :
  (Real.sqrt 450 / Real.sqrt 200) + (Real.sqrt 245 / Real.sqrt 175) = (15 + 2 * Real.sqrt 7) / 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l51_5112


namespace NUMINAMATH_CALUDE_number_division_problem_l51_5144

theorem number_division_problem : ∃ x : ℝ, x / 5 = 75 + x / 6 ∧ x = 2250 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l51_5144


namespace NUMINAMATH_CALUDE_janna_weekly_sleep_l51_5131

/-- Represents the number of hours Janna sleeps in a week. -/
def weekly_sleep_hours (weekday_sleep : ℕ) (weekend_sleep : ℕ) : ℕ :=
  5 * weekday_sleep + 2 * weekend_sleep

/-- Proves that Janna sleeps 51 hours in a week. -/
theorem janna_weekly_sleep :
  weekly_sleep_hours 7 8 = 51 :=
by sorry

end NUMINAMATH_CALUDE_janna_weekly_sleep_l51_5131


namespace NUMINAMATH_CALUDE_fishbowl_count_l51_5194

theorem fishbowl_count (total_fish : ℕ) (fish_per_bowl : ℕ) (h1 : total_fish = 6003) (h2 : fish_per_bowl = 23) :
  total_fish / fish_per_bowl = 261 :=
by sorry

end NUMINAMATH_CALUDE_fishbowl_count_l51_5194


namespace NUMINAMATH_CALUDE_system_solution_l51_5159

theorem system_solution : ∃! (x y : ℝ), x + y = 5 ∧ x - y = 3 ∧ x = 4 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l51_5159


namespace NUMINAMATH_CALUDE_f_sum_negative_l51_5113

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_odd : ∀ x, f x + f (-x) = 0
axiom f_increasing_neg : ∀ x y, x < y → y ≤ 0 → f x < f y

-- Define the theorem
theorem f_sum_negative (x₁ x₂ : ℝ) 
  (h₁ : x₁ + x₂ < 0) (h₂ : x₁ * x₂ < 0) : 
  f x₁ + f x₂ < 0 := by sorry

end NUMINAMATH_CALUDE_f_sum_negative_l51_5113


namespace NUMINAMATH_CALUDE_triangle_theorem_l51_5140

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a * Real.sin (2 * t.B) = Real.sqrt 3 * t.b * Real.sin t.A)
  (h2 : Real.cos t.A = 1 / 3) :
  t.B = π / 6 ∧ Real.sin t.C = (2 * Real.sqrt 6 + 1) / 6 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l51_5140


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_odd_integers_l51_5171

/-- Given three consecutive odd integers where the largest is -47, their sum is -141 -/
theorem sum_of_three_consecutive_odd_integers :
  ∀ (a b c : ℤ),
  (a < b ∧ b < c) →                   -- a, b, c are in ascending order
  (∃ k : ℤ, a = 2*k + 1) →            -- a is odd
  (∃ k : ℤ, b = 2*k + 1) →            -- b is odd
  (∃ k : ℤ, c = 2*k + 1) →            -- c is odd
  (b = a + 2) →                       -- b is the next consecutive odd integer after a
  (c = b + 2) →                       -- c is the next consecutive odd integer after b
  (c = -47) →                         -- the largest number is -47
  (a + b + c = -141) :=               -- their sum is -141
by sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_odd_integers_l51_5171


namespace NUMINAMATH_CALUDE_original_profit_percentage_l51_5138

theorem original_profit_percentage (cost_price selling_price : ℝ) :
  cost_price > 0 →
  selling_price > cost_price →
  (2 * selling_price - cost_price) / cost_price = 3.2 →
  (selling_price - cost_price) / cost_price = 1.1 := by
  sorry

end NUMINAMATH_CALUDE_original_profit_percentage_l51_5138


namespace NUMINAMATH_CALUDE_acute_triangle_condition_l51_5125

/-- A triangle is represented by its incircle radius and circumcircle radius -/
structure Triangle where
  r : ℝ  -- radius of the incircle
  R : ℝ  -- radius of the circumcircle

/-- A triangle is acute if all its angles are less than 90 degrees -/
def Triangle.isAcute (t : Triangle) : Prop :=
  sorry  -- definition of an acute triangle

/-- The main theorem: if R < r(√2 + 1), then the triangle is acute -/
theorem acute_triangle_condition (t : Triangle) 
  (h : t.R < t.r * (Real.sqrt 2 + 1)) : t.isAcute :=
sorry

end NUMINAMATH_CALUDE_acute_triangle_condition_l51_5125


namespace NUMINAMATH_CALUDE_logarithm_expression_equals_two_l51_5150

-- Define the base 10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_expression_equals_two :
  lg 4 + lg 9 + 2 * Real.sqrt ((lg 6)^2 - lg 36 + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equals_two_l51_5150


namespace NUMINAMATH_CALUDE_least_common_meeting_time_l51_5101

def prime_lap_times : List Nat := [2, 3, 5, 7, 11, 13, 17]

def is_divisible_by_at_least_four (n : Nat) : Bool :=
  (prime_lap_times.filter (fun p => n % p = 0)).length ≥ 4

theorem least_common_meeting_time :
  ∃ T : Nat, T > 0 ∧ is_divisible_by_at_least_four T ∧
  ∀ t : Nat, 0 < t ∧ t < T → ¬is_divisible_by_at_least_four t :=
by sorry

end NUMINAMATH_CALUDE_least_common_meeting_time_l51_5101


namespace NUMINAMATH_CALUDE_triangle_smallest_side_l51_5192

theorem triangle_smallest_side (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_inequality : a^2 + b^2 > 5 * c^2) : c < a ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_triangle_smallest_side_l51_5192


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l51_5130

/-- Given an arithmetic sequence with the first three terms x-y, x+y, and x/y,
    the fourth term is (10 - (2√13)/3) / (√13 - 1) -/
theorem arithmetic_sequence_fourth_term (x y : ℝ) (h : x ≠ 0) :
  let a₁ : ℝ := x - y
  let a₂ : ℝ := x + y
  let a₃ : ℝ := x / y
  let d : ℝ := a₂ - a₁
  let a₄ : ℝ := a₃ + d
  a₄ = (10 - (2 * Real.sqrt 13) / 3) / (Real.sqrt 13 - 1) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l51_5130


namespace NUMINAMATH_CALUDE_cos_squared_difference_eq_sqrt_three_half_l51_5146

theorem cos_squared_difference_eq_sqrt_three_half :
  (Real.cos (π / 12))^2 - (Real.cos (5 * π / 12))^2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_difference_eq_sqrt_three_half_l51_5146


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_average_l51_5188

theorem consecutive_even_numbers_sum_average (x y z : ℤ) : 
  (∃ k : ℤ, x = 2*k ∧ y = 2*k + 2 ∧ z = 2*k + 4) →  -- consecutive even numbers
  (x + y + z = (x + y + z) / 3 + 44) →               -- sum is 44 more than average
  z = 24 :=                                          -- largest number is 24
by sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_average_l51_5188


namespace NUMINAMATH_CALUDE_notebook_redistribution_l51_5176

theorem notebook_redistribution (total_notebooks : ℕ) (initial_boxes : ℕ) (new_notebooks_per_box : ℕ) :
  total_notebooks = 1200 →
  initial_boxes = 30 →
  new_notebooks_per_box = 35 →
  total_notebooks % new_notebooks_per_box = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_notebook_redistribution_l51_5176


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersections_l51_5134

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The number of distinct interior intersection points of diagonals in a regular decagon -/
def intersection_points (n : ℕ) : ℕ := Nat.choose n 4

theorem decagon_diagonal_intersections :
  intersection_points n = 210 :=
sorry

end NUMINAMATH_CALUDE_decagon_diagonal_intersections_l51_5134


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l51_5151

theorem sin_2alpha_value (α : Real) 
  (h : Real.cos (π / 4 - α) = 3 / 5) : 
  Real.sin (2 * α) = -7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l51_5151


namespace NUMINAMATH_CALUDE_polynomial_factorization_l51_5120

theorem polynomial_factorization (a : ℝ) : a^2 - a = a * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l51_5120


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l51_5157

/-- Proves that a train with given length and speed takes the calculated time to cross a bridge of given length. -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 90) 
  (h2 : train_speed_kmh = 45) 
  (h3 : bridge_length = 285) : 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

#check train_bridge_crossing_time

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l51_5157


namespace NUMINAMATH_CALUDE_number_fraction_problem_l51_5110

theorem number_fraction_problem (x : ℝ) : (1/3 : ℝ) * (1/4 : ℝ) * x = 15 → (3/10 : ℝ) * x = 54 := by
  sorry

end NUMINAMATH_CALUDE_number_fraction_problem_l51_5110


namespace NUMINAMATH_CALUDE_complex_root_sum_square_l51_5199

theorem complex_root_sum_square (p q : ℝ) : 
  (6 * (p + q * I)^3 + 5 * (p + q * I)^2 - (p + q * I) + 14 = 0) →
  (6 * (p - q * I)^3 + 5 * (p - q * I)^2 - (p - q * I) + 14 = 0) →
  p + q^2 = 21/4 := by
sorry

end NUMINAMATH_CALUDE_complex_root_sum_square_l51_5199


namespace NUMINAMATH_CALUDE_rationalization_factor_l51_5111

theorem rationalization_factor (a b : ℝ) :
  (Real.sqrt a - Real.sqrt b) * (Real.sqrt a + Real.sqrt b) = a - b :=
by sorry

end NUMINAMATH_CALUDE_rationalization_factor_l51_5111


namespace NUMINAMATH_CALUDE_minimum_value_sqrt_plus_reciprocal_l51_5189

theorem minimum_value_sqrt_plus_reciprocal (x : ℝ) (hx : x > 0) :
  3 * Real.sqrt x + 1 / x ≥ 4 ∧
  (3 * Real.sqrt x + 1 / x = 4 ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_sqrt_plus_reciprocal_l51_5189


namespace NUMINAMATH_CALUDE_reseat_twelve_women_l51_5156

/-- Number of ways to reseat n women under given conditions -/
def T : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | n + 3 => T (n + 2) + T (n + 1) + T n

/-- Theorem stating that the number of ways to reseat 12 women is 927 -/
theorem reseat_twelve_women : T 12 = 927 := by
  sorry

end NUMINAMATH_CALUDE_reseat_twelve_women_l51_5156


namespace NUMINAMATH_CALUDE_consecutive_integers_product_360_l51_5149

theorem consecutive_integers_product_360 :
  ∃ (n m : ℤ), 
    n * (n + 1) = 360 ∧ 
    (m - 1) * m * (m + 1) = 360 ∧ 
    n + (n + 1) + (m - 1) + m + (m + 1) = 55 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_360_l51_5149


namespace NUMINAMATH_CALUDE_min_reciprocal_81_l51_5104

/-- The reciprocal function -/
def reciprocal (x : ℚ) : ℚ := 1 / x

/-- Apply the reciprocal function n times -/
def apply_reciprocal (x : ℚ) (n : ℕ) : ℚ :=
  match n with
  | 0 => x
  | n + 1 => reciprocal (apply_reciprocal x n)

/-- Theorem: The minimum number of times to apply the reciprocal function to 81 to return to 81 is 2 -/
theorem min_reciprocal_81 :
  (∃ n : ℕ, apply_reciprocal 81 n = 81 ∧ n > 0) ∧
  (∀ m : ℕ, m > 0 ∧ m < 2 → apply_reciprocal 81 m ≠ 81) ∧
  apply_reciprocal 81 2 = 81 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_81_l51_5104


namespace NUMINAMATH_CALUDE_surface_area_of_sliced_prism_l51_5122

/-- A right prism with equilateral triangular bases -/
structure RightPrism where
  base_side_length : ℝ
  height : ℝ

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The solid CXYZ formed by slicing the prism -/
structure SolidCXYZ where
  prism : RightPrism
  C : Point3D
  X : Point3D
  Y : Point3D
  Z : Point3D

/-- Function to calculate the surface area of SolidCXYZ -/
def surface_area_CXYZ (solid : SolidCXYZ) : ℝ :=
  sorry

/-- Theorem statement -/
theorem surface_area_of_sliced_prism (solid : SolidCXYZ) 
  (h1 : solid.prism.base_side_length = 12)
  (h2 : solid.prism.height = 16)
  (h3 : solid.X.x = (solid.C.x + solid.prism.base_side_length / 2))
  (h4 : solid.Y.x = (solid.C.x + solid.prism.base_side_length))
  (h5 : solid.Z.z = (solid.C.z + solid.prism.height / 2)) :
  surface_area_CXYZ solid = 48 + 9 * Real.sqrt 3 + 3 * Real.sqrt 91 :=
sorry

end NUMINAMATH_CALUDE_surface_area_of_sliced_prism_l51_5122


namespace NUMINAMATH_CALUDE_possible_values_of_y_l51_5153

theorem possible_values_of_y (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 90) :
  let y := ((x - 3)^2 * (x + 4)) / (2 * x - 5)
  y = 0 ∨ y = 144 ∨ y = -24 := by sorry

end NUMINAMATH_CALUDE_possible_values_of_y_l51_5153


namespace NUMINAMATH_CALUDE_smallest_covering_rectangles_l51_5106

/-- Represents a rectangle with width and height. -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a rectangular region to be covered. -/
structure Region where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle. -/
def rectangleArea (r : Rectangle) : ℕ := r.width * r.height

/-- Calculates the area of a region. -/
def regionArea (r : Region) : ℕ := r.width * r.height

/-- Theorem: The smallest number of 3-by-5 rectangles needed to cover a 15-by-20 region is 20. -/
theorem smallest_covering_rectangles :
  let coveringRectangle : Rectangle := { width := 3, height := 5 }
  let regionToCover : Region := { width := 15, height := 20 }
  (regionArea regionToCover) / (rectangleArea coveringRectangle) = 20 ∧
  (regionToCover.width % coveringRectangle.width = 0) ∧
  (regionToCover.height % coveringRectangle.height = 0) := by
  sorry

#check smallest_covering_rectangles

end NUMINAMATH_CALUDE_smallest_covering_rectangles_l51_5106


namespace NUMINAMATH_CALUDE_range_of_f_l51_5190

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the theorem
theorem range_of_f :
  (∀ x : ℝ, (1 + x > 0 ∧ 1 - x > 0) → (3/4 ≤ f x ∧ f x ≤ 57)) →
  Set.range f = Set.Ici 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l51_5190


namespace NUMINAMATH_CALUDE_angle_relation_in_triangle_l51_5187

theorem angle_relation_in_triangle (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C) (h_sin : Real.sin A > Real.sin B) : A > B := by
  sorry

end NUMINAMATH_CALUDE_angle_relation_in_triangle_l51_5187


namespace NUMINAMATH_CALUDE_angle_sum_around_point_l51_5168

theorem angle_sum_around_point (x : ℝ) : 
  (3 * x + 6 * x + x + 2 * x = 360) → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_around_point_l51_5168


namespace NUMINAMATH_CALUDE_perception_arrangements_l51_5195

def word_length : ℕ := 10

def repeating_letters : List (Char × ℕ) := [('E', 2), ('P', 2), ('I', 2)]

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem perception_arrangements : 
  (factorial word_length) / ((repeating_letters.map (λ (_, count) => factorial count)).prod) = 453600 := by
  sorry

end NUMINAMATH_CALUDE_perception_arrangements_l51_5195


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l51_5116

theorem arithmetic_simplification : (-18) + (-12) - (-33) + 17 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l51_5116


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l51_5182

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem f_satisfies_conditions :
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l51_5182


namespace NUMINAMATH_CALUDE_cubic_equation_geometric_progression_solution_l51_5107

/-- Given a cubic equation ax^3 + bx^2 + cx + d = 0 where the coefficients form
    a geometric progression with ratio q, prove that x = -q is a solution. -/
theorem cubic_equation_geometric_progression_solution
  (a b c d q : ℝ) (hq : q ≠ 0) (ha : a ≠ 0)
  (hb : b = a * q) (hc : c = a * q^2) (hd : d = a * q^3) :
  a * (-q)^3 + b * (-q)^2 + c * (-q) + d = 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_geometric_progression_solution_l51_5107


namespace NUMINAMATH_CALUDE_scientific_notation_15510000_l51_5127

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_15510000 :
  toScientificNotation 15510000 = ScientificNotation.mk 1.551 7 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_15510000_l51_5127


namespace NUMINAMATH_CALUDE_red_markers_count_l51_5177

def total_markers : ℕ := 105
def blue_markers : ℕ := 64

theorem red_markers_count : 
  ∃ (red_markers : ℕ), red_markers = total_markers - blue_markers ∧ red_markers = 41 := by
  sorry

end NUMINAMATH_CALUDE_red_markers_count_l51_5177


namespace NUMINAMATH_CALUDE_wuhan_spring_temp_difference_l51_5132

/-- The average daily high temperature in spring in the Wuhan area -/
def average_high : ℝ := 15

/-- The lowest temperature in spring in the Wuhan area -/
def lowest_temp : ℝ := 7

/-- The difference between the average daily high temperature and the lowest temperature -/
def temp_difference : ℝ := average_high - lowest_temp

/-- Theorem stating that the temperature difference is 8°C -/
theorem wuhan_spring_temp_difference : temp_difference = 8 := by
  sorry

end NUMINAMATH_CALUDE_wuhan_spring_temp_difference_l51_5132


namespace NUMINAMATH_CALUDE_nigels_money_theorem_l51_5147

/-- Represents the amount of money Nigel has at different stages --/
structure NigelsMoney where
  initial : ℕ
  afterFirstGiveaway : ℕ
  afterMotherGift : ℕ
  final : ℕ

/-- Theorem stating Nigel's final amount is $10 more than twice his initial amount --/
theorem nigels_money_theorem (n : NigelsMoney) (h1 : n.initial = 45)
  (h2 : n.afterMotherGift = n.afterFirstGiveaway + 80)
  (h3 : n.final = n.afterMotherGift - 25)
  (h4 : n.afterFirstGiveaway < n.initial) :
  n.final = 2 * n.initial + 10 := by
  sorry

end NUMINAMATH_CALUDE_nigels_money_theorem_l51_5147


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solution_equation_three_solutions_equation_four_solutions_l51_5175

-- Equation 1
theorem equation_one_solutions (x : ℝ) : 
  (x + 3)^2 = (1 - 2*x)^2 ↔ x = 4 ∨ x = -2/3 := by sorry

-- Equation 2
theorem equation_two_solution (x : ℝ) :
  (x + 1)^2 = 4*x ↔ x = 1 := by sorry

-- Equation 3
theorem equation_three_solutions (x : ℝ) :
  2*x^2 - 5*x + 1 = 0 ↔ x = (5 + Real.sqrt 17) / 4 ∨ x = (5 - Real.sqrt 17) / 4 := by sorry

-- Equation 4
theorem equation_four_solutions (x : ℝ) :
  (2*x - 1)^2 = x*(3*x + 2) - 7 ↔ x = 4 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solution_equation_three_solutions_equation_four_solutions_l51_5175


namespace NUMINAMATH_CALUDE_snake_shedding_decimal_l51_5155

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : ℕ) : ℕ :=
  let ones := octal % 10
  let eights := (octal / 10) % 10
  let sixty_fours := octal / 100
  ones + 8 * eights + 64 * sixty_fours

/-- The number of ways a snake can shed its skin in octal --/
def snake_shedding_octal : ℕ := 453

theorem snake_shedding_decimal :
  octal_to_decimal snake_shedding_octal = 299 := by
  sorry

end NUMINAMATH_CALUDE_snake_shedding_decimal_l51_5155


namespace NUMINAMATH_CALUDE_a_alone_time_equals_b_alone_time_l51_5136

/-- Two workers finishing a job -/
structure WorkerPair where
  total_time : ℝ
  b_alone_time : ℝ
  work : ℝ

/-- The time it takes for worker a to finish the job alone -/
def a_alone_time (w : WorkerPair) : ℝ :=
  w.b_alone_time

theorem a_alone_time_equals_b_alone_time (w : WorkerPair)
  (h1 : w.total_time = 10)
  (h2 : w.b_alone_time = 20) :
  a_alone_time w = w.b_alone_time :=
by
  sorry

#check a_alone_time_equals_b_alone_time

end NUMINAMATH_CALUDE_a_alone_time_equals_b_alone_time_l51_5136


namespace NUMINAMATH_CALUDE_total_pay_calculation_l51_5184

/-- Calculates the total pay for a worker given regular and overtime hours --/
def total_pay (regular_rate : ℝ) (regular_hours : ℝ) (overtime_hours : ℝ) : ℝ :=
  let overtime_rate := 2 * regular_rate
  regular_rate * regular_hours + overtime_rate * overtime_hours

/-- Theorem stating that under the given conditions, the total pay is $192 --/
theorem total_pay_calculation :
  let regular_rate := 3
  let regular_hours := 40
  let overtime_hours := 12
  total_pay regular_rate regular_hours overtime_hours = 192 := by
  sorry

end NUMINAMATH_CALUDE_total_pay_calculation_l51_5184


namespace NUMINAMATH_CALUDE_otimes_properties_l51_5197

def otimes (a b : ℝ) : ℝ := a * b + a + b

theorem otimes_properties :
  (∀ a b : ℝ, otimes a b = otimes b a) ∧
  (∃ a b c : ℝ, otimes (otimes a b) c ≠ otimes a (otimes b c)) ∧
  (∃ a b c : ℝ, otimes (a + b) c ≠ otimes a c + otimes b c) := by
  sorry

end NUMINAMATH_CALUDE_otimes_properties_l51_5197


namespace NUMINAMATH_CALUDE_henry_twice_jills_age_l51_5191

theorem henry_twice_jills_age (henry_age jill_age : ℕ) (years_ago : ℕ) : 
  henry_age + jill_age = 43 →
  henry_age = 27 →
  jill_age = 16 →
  henry_age - years_ago = 2 * (jill_age - years_ago) →
  years_ago = 5 := by
  sorry

end NUMINAMATH_CALUDE_henry_twice_jills_age_l51_5191


namespace NUMINAMATH_CALUDE_phase_shift_of_sine_function_l51_5174

/-- The phase shift of the function y = 5 sin(3x - π/3) is π/9. -/
theorem phase_shift_of_sine_function : 
  let f : ℝ → ℝ := λ x => 5 * Real.sin (3 * x - π / 3)
  ∃ (shift : ℝ), shift = π / 9 ∧ 
    ∀ (x : ℝ), f (x + shift) = 5 * Real.sin (3 * x) :=
by
  sorry

end NUMINAMATH_CALUDE_phase_shift_of_sine_function_l51_5174


namespace NUMINAMATH_CALUDE_average_of_middle_two_l51_5128

theorem average_of_middle_two (n₁ n₂ n₃ n₄ n₅ n₆ : ℝ) : 
  (n₁ + n₂ + n₃ + n₄ + n₅ + n₆) / 6 = 3.95 →
  (n₁ + n₂) / 2 = 3.6 →
  (n₅ + n₆) / 2 = 4.400000000000001 →
  (n₃ + n₄) / 2 = 3.85 :=
by sorry

end NUMINAMATH_CALUDE_average_of_middle_two_l51_5128


namespace NUMINAMATH_CALUDE_inequality_proof_l51_5154

theorem inequality_proof (a b : ℝ) (h : a < b) : 7 * a - 7 * b < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l51_5154


namespace NUMINAMATH_CALUDE_games_played_calculation_l51_5123

/-- Represents the number of games played by a baseball team --/
def GamesPlayed : ℕ → ℕ → ℕ → ℕ
  | games_won, games_left, games_to_win_more => games_won + games_left - games_to_win_more

/-- Represents the total number of games in a season --/
def TotalGames : ℕ → ℕ → ℕ
  | games_played, games_left => games_played + games_left

theorem games_played_calculation (games_won : ℕ) (games_left : ℕ) (games_to_win_more : ℕ) :
  games_won = 12 →
  games_left = 10 →
  games_to_win_more = 8 →
  (3 * (games_won + games_to_win_more) = 2 * TotalGames (GamesPlayed games_won games_left games_to_win_more) games_left) →
  GamesPlayed games_won games_left games_to_win_more = 20 := by
  sorry

end NUMINAMATH_CALUDE_games_played_calculation_l51_5123


namespace NUMINAMATH_CALUDE_grid_rectangles_l51_5129

/-- The number of rectangles formed in a grid of parallel lines -/
def rectangles_in_grid (lines1 : ℕ) (lines2 : ℕ) : ℕ :=
  (lines1 - 1) * (lines2 - 1)

/-- Theorem: In a grid formed by 8 parallel lines intersected by 10 parallel lines, 
    the total number of rectangles formed is 63 -/
theorem grid_rectangles :
  rectangles_in_grid 8 10 = 63 := by
  sorry

end NUMINAMATH_CALUDE_grid_rectangles_l51_5129


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l51_5105

def vector_a : Fin 2 → ℝ := ![1, -2]
def vector_b (x : ℝ) : Fin 2 → ℝ := ![x, 4]

theorem parallel_vectors_magnitude (x : ℝ) :
  (∃ k : ℝ, vector_a = k • vector_b x) →
  Real.sqrt ((vector_a 0 - vector_b x 0)^2 + (vector_a 1 - vector_b x 1)^2) = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l51_5105


namespace NUMINAMATH_CALUDE_total_oranges_count_l51_5141

/-- The number of oranges picked by Joan -/
def joan_oranges : ℕ := 37

/-- The number of oranges picked by Sara -/
def sara_oranges : ℕ := 10

/-- The total number of oranges picked -/
def total_oranges : ℕ := joan_oranges + sara_oranges

theorem total_oranges_count : total_oranges = 47 := by
  sorry

end NUMINAMATH_CALUDE_total_oranges_count_l51_5141


namespace NUMINAMATH_CALUDE_root_exists_and_bisection_applicable_l51_5124

-- Define the function f(x) = x^2 - 2x - 3
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Define the interval (-2, 2)
def interval : Set ℝ := Set.Ioo (-2) 2

-- Theorem statement
theorem root_exists_and_bisection_applicable :
  ∃ (x : ℝ), x ∈ interval ∧ f x = 0 ∧
  (∃ (a b : ℝ), a ∈ interval ∧ b ∈ interval ∧ f a * f b ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_root_exists_and_bisection_applicable_l51_5124


namespace NUMINAMATH_CALUDE_equation_solution_l51_5169

theorem equation_solution : ∃ m : ℤ, 3^4 - m = 4^3 + 2 ∧ m = 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l51_5169


namespace NUMINAMATH_CALUDE_mass_of_ccl4_produced_l51_5162

-- Define the chemical equation
def balanced_equation : String := "CaC2 + 4 Cl2O → CCl4 + CaCl2O"

-- Define the number of moles of reaction
def reaction_moles : ℝ := 8

-- Define molar masses
def molar_mass_carbon : ℝ := 12.01
def molar_mass_chlorine : ℝ := 35.45

-- Define the molar mass of CCl4
def molar_mass_ccl4 : ℝ := molar_mass_carbon + 4 * molar_mass_chlorine

-- Theorem statement
theorem mass_of_ccl4_produced : 
  reaction_moles * molar_mass_ccl4 = 1230.48 := by sorry

end NUMINAMATH_CALUDE_mass_of_ccl4_produced_l51_5162


namespace NUMINAMATH_CALUDE_unique_albums_count_l51_5118

/-- The number of albums in either Andrew's, John's, or Sarah's collection, but not in all three -/
def unique_albums (shared_albums andrew_total john_unique sarah_unique : ℕ) : ℕ :=
  (andrew_total - shared_albums) + john_unique + sarah_unique

/-- Theorem stating the number of unique albums across the three collections -/
theorem unique_albums_count :
  unique_albums 10 20 5 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_unique_albums_count_l51_5118


namespace NUMINAMATH_CALUDE_min_value_of_f_l51_5119

/-- The polynomial function in two variables -/
def f (x y : ℝ) : ℝ := x^2 + y^2 - 6*x + 8*y + 7

/-- The minimum value of the polynomial function -/
theorem min_value_of_f :
  ∀ x y : ℝ, f x y ≥ -18 ∧ ∃ a b : ℝ, f a b = -18 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l51_5119


namespace NUMINAMATH_CALUDE_multiply_decimals_l51_5115

theorem multiply_decimals : 3.6 * 0.05 = 0.18 := by
  sorry

end NUMINAMATH_CALUDE_multiply_decimals_l51_5115


namespace NUMINAMATH_CALUDE_f_19_equals_zero_l51_5133

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem f_19_equals_zero
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_period : ∀ x, f (x + 2) = -f x) :
  f 19 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_19_equals_zero_l51_5133


namespace NUMINAMATH_CALUDE_chords_intersection_concyclic_l51_5186

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define a point on the ellipse
structure PointOnEllipse (a b : ℝ) where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse a b x y

-- Define the theorem
theorem chords_intersection_concyclic 
  (a b : ℝ) 
  (A B C D S : PointOnEllipse a b) 
  (h1 : S.x ≠ A.x ∨ S.y ≠ A.y) 
  (h2 : S.x ≠ B.x ∨ S.y ≠ B.y)
  (h3 : S.x ≠ C.x ∨ S.y ≠ C.y)
  (h4 : S.x ≠ D.x ∨ S.y ≠ D.y)
  (h5 : (A.y - S.y) * (C.x - S.x) = (A.x - S.x) * (C.y - S.y)) -- AB and CD intersect at S
  (h6 : (B.y - S.y) * (D.x - S.x) = (B.x - S.x) * (D.y - S.y)) -- AB and CD intersect at S
  (h7 : (A.y - S.y) * (C.x - S.x) = (C.y - S.y) * (D.x - S.x)) -- ∠ASC = ∠BSD
  : ∃ (center : ℝ × ℝ) (radius : ℝ),
    (A.x - center.1)^2 + (A.y - center.2)^2 = radius^2 ∧
    (B.x - center.1)^2 + (B.y - center.2)^2 = radius^2 ∧
    (C.x - center.1)^2 + (C.y - center.2)^2 = radius^2 ∧
    (D.x - center.1)^2 + (D.y - center.2)^2 = radius^2 := by
  sorry

end NUMINAMATH_CALUDE_chords_intersection_concyclic_l51_5186


namespace NUMINAMATH_CALUDE_range_of_a_l51_5165

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x_0 : ℝ, x_0^2 + 2*a*x_0 + 2 - a = 0) → 
  a ≤ -2 ∨ a = 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l51_5165


namespace NUMINAMATH_CALUDE_roberts_chocolates_l51_5158

theorem roberts_chocolates (nickel_chocolates : ℕ) (difference : ℕ) : 
  nickel_chocolates = 3 → difference = 9 → nickel_chocolates + difference = 12 := by
  sorry

end NUMINAMATH_CALUDE_roberts_chocolates_l51_5158


namespace NUMINAMATH_CALUDE_expected_rainfall_theorem_l51_5117

/-- Represents the daily rainfall probabilities and amounts -/
structure DailyRainfall where
  sun_prob : ℝ
  light_rain_prob : ℝ
  heavy_rain_prob : ℝ
  light_rain_amount : ℝ
  heavy_rain_amount : ℝ

/-- Calculates the expected total rainfall over a given number of days -/
def expected_total_rainfall (daily : DailyRainfall) (days : ℕ) : ℝ :=
  days * (daily.light_rain_prob * daily.light_rain_amount + daily.heavy_rain_prob * daily.heavy_rain_amount)

/-- The main theorem stating the expected total rainfall over 10 days -/
theorem expected_rainfall_theorem (daily : DailyRainfall)
  (h1 : daily.sun_prob = 0.5)
  (h2 : daily.light_rain_prob = 0.3)
  (h3 : daily.heavy_rain_prob = 0.2)
  (h4 : daily.light_rain_amount = 3)
  (h5 : daily.heavy_rain_amount = 6)
  : expected_total_rainfall daily 10 = 21 := by
  sorry

end NUMINAMATH_CALUDE_expected_rainfall_theorem_l51_5117


namespace NUMINAMATH_CALUDE_student_rank_from_right_l51_5170

theorem student_rank_from_right 
  (total_students : Nat) 
  (rank_from_left : Nat) 
  (h1 : total_students = 21)
  (h2 : rank_from_left = 6) :
  total_students - rank_from_left + 1 = 17 :=
by sorry

end NUMINAMATH_CALUDE_student_rank_from_right_l51_5170


namespace NUMINAMATH_CALUDE_cubic_equation_root_l51_5142

theorem cubic_equation_root (a b : ℚ) : 
  (2 - 3 * Real.sqrt 3) ^ 3 + a * (2 - 3 * Real.sqrt 3) ^ 2 + b * (2 - 3 * Real.sqrt 3) - 37 = 0 →
  a = -55/23 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l51_5142


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l51_5145

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x, a * x^2 + 2 * a * x + 1 > 0) → 
  (0 < a ∧ a < 1) → 
  ¬ ((0 < a ∧ a < 1) ↔ (∀ x, a * x^2 + 2 * a * x + 1 > 0)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l51_5145


namespace NUMINAMATH_CALUDE_inequality_solution_set_sum_of_coordinates_l51_5102

-- Problem 1
theorem inequality_solution_set (x : ℝ) :
  x + |2*x - 1| < 3 ↔ -2 < x ∧ x < 4/3 := by sorry

-- Problem 2
theorem sum_of_coordinates (x y z : ℝ) 
  (h1 : x^2 + y^2 + z^2 = 1) 
  (h2 : x + 2*y + 3*z = Real.sqrt 14) : 
  x + y + z = 3 * Real.sqrt 14 / 7 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_sum_of_coordinates_l51_5102


namespace NUMINAMATH_CALUDE_difference_of_squares_l51_5198

theorem difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l51_5198


namespace NUMINAMATH_CALUDE_rectangle_area_l51_5161

/-- Proves that a rectangle with width 4 inches and perimeter 30 inches has an area of 44 square inches -/
theorem rectangle_area (width : ℝ) (perimeter : ℝ) (height : ℝ) (area : ℝ) : 
  width = 4 →
  perimeter = 30 →
  perimeter = 2 * (width + height) →
  area = width * height →
  area = 44 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l51_5161


namespace NUMINAMATH_CALUDE_triangle_345_not_triangle_123_not_triangle_384_not_triangle_5510_l51_5152

/-- A function that checks if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that line segments of lengths 3, 4, and 5 can form a triangle -/
theorem triangle_345 : can_form_triangle 3 4 5 := by sorry

/-- Theorem stating that line segments of lengths 1, 2, and 3 cannot form a triangle -/
theorem not_triangle_123 : ¬can_form_triangle 1 2 3 := by sorry

/-- Theorem stating that line segments of lengths 3, 8, and 4 cannot form a triangle -/
theorem not_triangle_384 : ¬can_form_triangle 3 8 4 := by sorry

/-- Theorem stating that line segments of lengths 5, 5, and 10 cannot form a triangle -/
theorem not_triangle_5510 : ¬can_form_triangle 5 5 10 := by sorry

end NUMINAMATH_CALUDE_triangle_345_not_triangle_123_not_triangle_384_not_triangle_5510_l51_5152


namespace NUMINAMATH_CALUDE_polynomial_expansion_l51_5100

theorem polynomial_expansion (t : ℝ) :
  (3 * t^3 - 4 * t^2 + 5 * t - 3) * (4 * t^2 - 2 * t + 1) =
  12 * t^5 - 22 * t^4 + 31 * t^3 - 26 * t^2 + 11 * t - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l51_5100


namespace NUMINAMATH_CALUDE_circle_center_problem_circle_center_l51_5148

/-- The equation of a circle in the form x² + y² + 2ax + 2by + c = 0 
    has center (-a, -b) -/
theorem circle_center (a b c : ℝ) :
  let circle_eq := fun (x y : ℝ) => x^2 + y^2 + 2*a*x + 2*b*y + c = 0
  let center := (-a, -b)
  ∀ x y, circle_eq x y ↔ (x - center.1)^2 + (y - center.2)^2 = a^2 + b^2 - c :=
by sorry

/-- The center of the circle x² + y² + 2x + 4y - 3 = 0 is (-1, -2) -/
theorem problem_circle_center :
  let circle_eq := fun (x y : ℝ) => x^2 + y^2 + 2*x + 4*y - 3 = 0
  let center := (-1, -2)
  ∀ x y, circle_eq x y ↔ (x - center.1)^2 + (y - center.2)^2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_problem_circle_center_l51_5148


namespace NUMINAMATH_CALUDE_small_box_tape_length_l51_5135

theorem small_box_tape_length (large_seal : ℕ) (medium_seal : ℕ) (label_tape : ℕ)
  (large_count : ℕ) (medium_count : ℕ) (small_count : ℕ) (total_tape : ℕ)
  (h1 : large_seal = 4)
  (h2 : medium_seal = 2)
  (h3 : label_tape = 1)
  (h4 : large_count = 2)
  (h5 : medium_count = 8)
  (h6 : small_count = 5)
  (h7 : total_tape = 44)
  (h8 : total_tape = large_count * large_seal + medium_count * medium_seal + 
        small_count * label_tape + large_count * label_tape + 
        medium_count * label_tape + small_count * label_tape + 
        small_count * small_seal) :
  small_seal = 1 :=
by sorry

end NUMINAMATH_CALUDE_small_box_tape_length_l51_5135
