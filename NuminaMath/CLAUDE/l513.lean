import Mathlib

namespace NUMINAMATH_CALUDE_largest_value_l513_51349

theorem largest_value : 
  (4^2 : ℝ) ≥ 4 * 2 ∧ 
  (4^2 : ℝ) ≥ 4 - 2 ∧ 
  (4^2 : ℝ) ≥ 4 / 2 ∧ 
  (4^2 : ℝ) ≥ 4 + 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_value_l513_51349


namespace NUMINAMATH_CALUDE_solution_in_interval_l513_51343

theorem solution_in_interval (x₀ : ℝ) (h : Real.exp x₀ + x₀ = 2) : 0 < x₀ ∧ x₀ < 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_in_interval_l513_51343


namespace NUMINAMATH_CALUDE_reading_time_difference_l513_51364

def xanthia_speed : ℝ := 120
def molly_speed : ℝ := 60
def book_pages : ℝ := 300

theorem reading_time_difference : 
  (book_pages / molly_speed - book_pages / xanthia_speed) * 60 = 150 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_difference_l513_51364


namespace NUMINAMATH_CALUDE_player_jump_height_to_dunk_l513_51309

/-- Represents the height of a basketball player in feet -/
def player_height : ℝ := 6

/-- Represents the additional reach of the player above their head in inches -/
def player_reach : ℝ := 22

/-- Represents the height of the basketball rim in feet -/
def rim_height : ℝ := 10

/-- Represents the additional height above the rim needed to dunk in inches -/
def dunk_clearance : ℝ := 6

/-- Converts feet to inches -/
def feet_to_inches (feet : ℝ) : ℝ := feet * 12

/-- Calculates the jump height required for the player to dunk -/
def required_jump_height : ℝ :=
  feet_to_inches rim_height + dunk_clearance - (feet_to_inches player_height + player_reach)

theorem player_jump_height_to_dunk :
  required_jump_height = 32 := by sorry

end NUMINAMATH_CALUDE_player_jump_height_to_dunk_l513_51309


namespace NUMINAMATH_CALUDE_quadratic_root_value_l513_51312

theorem quadratic_root_value (m : ℝ) : 
  m^2 + m - 1 = 0 → 2*m^2 + 2*m + 2025 = 2027 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l513_51312


namespace NUMINAMATH_CALUDE_ladies_walking_ratio_l513_51300

/-- Given two ladies walking in Central Park, prove that the ratio of their distances is 2:1 -/
theorem ladies_walking_ratio :
  ∀ (distance1 distance2 : ℝ),
  distance2 = 4 →
  distance1 + distance2 = 12 →
  distance1 / distance2 = 2 := by
sorry

end NUMINAMATH_CALUDE_ladies_walking_ratio_l513_51300


namespace NUMINAMATH_CALUDE_expansion_equality_l513_51399

theorem expansion_equality (a b : ℝ) : (a - b) * (-a - b) = b^2 - a^2 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equality_l513_51399


namespace NUMINAMATH_CALUDE_sequence_correctness_l513_51337

def a (n : ℕ) : ℤ := (-1 : ℤ)^(n + 1) * n^2

theorem sequence_correctness : 
  (a 1 = 1) ∧ (a 2 = -4) ∧ (a 3 = 9) ∧ (a 4 = -16) ∧ (a 5 = 25) := by
  sorry

end NUMINAMATH_CALUDE_sequence_correctness_l513_51337


namespace NUMINAMATH_CALUDE_function_inequality_condition_l513_51371

open Real

/-- For the function f(x) = ln x - ax, where a ∈ ℝ and x ∈ (1, +∞),
    the inequality f(x) + a < 0 holds for all x in (1, +∞) if and only if a ≥ 1 -/
theorem function_inequality_condition (a : ℝ) :
  (∀ x : ℝ, x > 1 → (log x - a * x + a < 0)) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_condition_l513_51371


namespace NUMINAMATH_CALUDE_factorization_equality_l513_51327

theorem factorization_equality (m n : ℝ) : m^2 * n - n = n * (m + 1) * (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l513_51327


namespace NUMINAMATH_CALUDE_player1_wins_l513_51307

/-- Represents the state of the game -/
structure GameState :=
  (coins : ℕ)

/-- Represents a player's move -/
structure Move :=
  (coins_taken : ℕ)

/-- Defines a valid move for Player 1 -/
def valid_move_player1 (m : Move) : Prop :=
  m.coins_taken % 2 = 1 ∧ 1 ≤ m.coins_taken ∧ m.coins_taken ≤ 99

/-- Defines a valid move for Player 2 -/
def valid_move_player2 (m : Move) : Prop :=
  m.coins_taken % 2 = 0 ∧ 2 ≤ m.coins_taken ∧ m.coins_taken ≤ 100

/-- Defines the game transition for a player's move -/
def make_move (s : GameState) (m : Move) : GameState :=
  ⟨s.coins - m.coins_taken⟩

/-- Defines a winning strategy for Player 1 -/
def winning_strategy (initial_coins : ℕ) : Prop :=
  ∃ (strategy : GameState → Move),
    (∀ s : GameState, valid_move_player1 (strategy s)) ∧
    (∀ s : GameState, ∀ m : Move, 
      valid_move_player2 m → 
      ∃ (next_move : Move), 
        valid_move_player1 next_move ∧
        make_move (make_move s m) next_move = ⟨0⟩)

theorem player1_wins : winning_strategy 2001 := by
  sorry


end NUMINAMATH_CALUDE_player1_wins_l513_51307


namespace NUMINAMATH_CALUDE_first_three_decimal_digits_l513_51331

theorem first_three_decimal_digits (n : ℕ) (x : ℝ) : 
  n = 2003 → x = (10^n + 1)^(11/7) → 
  ∃ (y : ℝ), x = 10^2861 + y ∧ 0.571 < y/10^858 ∧ y/10^858 < 0.572 :=
by sorry

end NUMINAMATH_CALUDE_first_three_decimal_digits_l513_51331


namespace NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l513_51348

theorem sum_of_x_solutions_is_zero (y : ℝ) (h1 : y = 10) (h2 : ∃ x : ℝ, x^2 + y^2 = 169) : 
  ∃ x1 x2 : ℝ, x1^2 + y^2 = 169 ∧ x2^2 + y^2 = 169 ∧ x1 + x2 = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l513_51348


namespace NUMINAMATH_CALUDE_travel_time_calculation_l513_51360

/-- Travel time calculation given distance and average speed -/
theorem travel_time_calculation 
  (distance : ℝ) 
  (average_speed : ℝ) 
  (h1 : distance = 790) 
  (h2 : average_speed = 50) :
  distance / average_speed = 15.8 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_calculation_l513_51360


namespace NUMINAMATH_CALUDE_probability_is_two_ninety_one_l513_51358

/-- Represents the number of jellybeans of each color in the basket -/
structure JellyBeanBasket where
  red : Nat
  blue : Nat
  yellow : Nat

/-- Calculates the probability of picking exactly 2 red and 2 blue jellybeans -/
def probability_two_red_two_blue (basket : JellyBeanBasket) : Rat :=
  let total := basket.red + basket.blue + basket.yellow
  let favorable := Nat.choose basket.red 2 * Nat.choose basket.blue 2
  let total_combinations := Nat.choose total 4
  favorable / total_combinations

/-- The main theorem stating the probability is 2/91 -/
theorem probability_is_two_ninety_one :
  probability_two_red_two_blue ⟨5, 3, 7⟩ = 2 / 91 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_two_ninety_one_l513_51358


namespace NUMINAMATH_CALUDE_trip_cost_equals_bills_cost_l513_51370

/-- Proves that the cost of Liam's trip to Paris is equal to the cost of his bills. -/
theorem trip_cost_equals_bills_cost (
  monthly_savings : ℕ)
  (savings_duration_years : ℕ)
  (bills_cost : ℕ)
  (money_left_after_bills : ℕ)
  (h1 : monthly_savings = 500)
  (h2 : savings_duration_years = 2)
  (h3 : bills_cost = 3500)
  (h4 : money_left_after_bills = 8500)
  : bills_cost = monthly_savings * savings_duration_years * 12 - money_left_after_bills :=
by sorry

end NUMINAMATH_CALUDE_trip_cost_equals_bills_cost_l513_51370


namespace NUMINAMATH_CALUDE_tangent_line_x_ln_x_at_1_l513_51334

/-- The equation of the tangent line to y = x ln x at x = 1 is x - y - 1 = 0 -/
theorem tangent_line_x_ln_x_at_1 : 
  let f : ℝ → ℝ := λ x => x * Real.log x
  let tangent_line : ℝ → ℝ := λ x => x - 1
  (∀ x, x > 0 → HasDerivAt f (Real.log x + 1) x) ∧ 
  HasDerivAt f 1 1 ∧
  f 1 = 0 →
  ∀ x y, y = tangent_line x ↔ x - y - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_x_ln_x_at_1_l513_51334


namespace NUMINAMATH_CALUDE_sibling_ages_sum_l513_51395

/-- Given four positive integers representing ages of siblings, prove that their sum is 24 --/
theorem sibling_ages_sum (x y z : ℕ) (h1 : 2 * x^2 + y^2 + z^2 = 194) (h2 : x > y) (h3 : y > z) :
  x + x + y + z = 24 := by
  sorry

end NUMINAMATH_CALUDE_sibling_ages_sum_l513_51395


namespace NUMINAMATH_CALUDE_negation_of_universal_positive_square_not_equal_l513_51355

theorem negation_of_universal_positive_square_not_equal (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, x > 0 → x^2 ≠ x) ↔ (∃ x : ℝ, x > 0 ∧ x^2 = x) :=
sorry

end NUMINAMATH_CALUDE_negation_of_universal_positive_square_not_equal_l513_51355


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l513_51366

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  4 * a^3 + 8 * b^3 + 18 * c^3 + 1 / (9 * a * b * c) ≥ 8 / Real.sqrt 3 :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (4 * a^3 + 8 * b^3 + 18 * c^3 + 1 / (9 * a * b * c) = 8 / Real.sqrt 3) ↔
  (4 * a^3 = 8 * b^3 ∧ 8 * b^3 = 18 * c^3 ∧ 24 * a * b * c = 1 / (9 * a * b * c)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l513_51366


namespace NUMINAMATH_CALUDE_min_steps_to_remove_zeros_l513_51354

/-- Represents the state of the blackboard -/
structure BoardState where
  zeros : Nat
  ones : Nat

/-- Defines a step operation on the board state -/
def step (s : BoardState) : BoardState :=
  { zeros := s.zeros - 1, ones := s.ones + 1 }

/-- Theorem: Minimum steps to remove all zeroes -/
theorem min_steps_to_remove_zeros (initial : BoardState) 
  (h1 : initial.zeros = 150) 
  (h2 : initial.ones = 151) : 
  ∃ (n : Nat), n = 150 ∧ (step^[n] initial).zeros = 0 :=
sorry

end NUMINAMATH_CALUDE_min_steps_to_remove_zeros_l513_51354


namespace NUMINAMATH_CALUDE_equation_unique_solution_l513_51326

theorem equation_unique_solution :
  ∃! x : ℝ, (Real.sqrt x + 3 * Real.sqrt (x^2 + 9*x) + Real.sqrt (x + 9) = 45 - 3*x) ∧ 
  (x = 400/49) := by
  sorry

end NUMINAMATH_CALUDE_equation_unique_solution_l513_51326


namespace NUMINAMATH_CALUDE_investment_growth_l513_51381

/-- Compound interest function -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Problem statement -/
theorem investment_growth :
  let principal : ℝ := 8000
  let rate : ℝ := 0.04
  let time : ℕ := 10
  abs (compound_interest principal rate time - 11841.92) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_investment_growth_l513_51381


namespace NUMINAMATH_CALUDE_stripe_area_theorem_l513_51306

/-- Represents a cylindrical silo -/
structure Cylinder where
  diameter : ℝ
  height : ℝ

/-- Represents a stripe wrapped around a cylinder -/
structure Stripe where
  width : ℝ
  revolutions : ℕ

/-- Calculates the area of a stripe wrapped around a cylinder -/
def stripeArea (c : Cylinder) (s : Stripe) : ℝ :=
  s.width * c.height

theorem stripe_area_theorem (c : Cylinder) (s : Stripe) :
  stripeArea c s = s.width * c.height := by sorry

end NUMINAMATH_CALUDE_stripe_area_theorem_l513_51306


namespace NUMINAMATH_CALUDE_soap_brand_ratio_l513_51356

def total_households : ℕ := 240
def households_neither : ℕ := 80
def households_only_A : ℕ := 60
def households_both : ℕ := 25

theorem soap_brand_ratio :
  ∃ (households_only_B : ℕ),
    households_only_A + households_only_B + households_both + households_neither = total_households ∧
    households_only_B / households_both = 3 := by
  sorry

end NUMINAMATH_CALUDE_soap_brand_ratio_l513_51356


namespace NUMINAMATH_CALUDE_ellipse_b_value_l513_51394

/-- Definition of an ellipse with foci and a point on it -/
structure Ellipse where
  a : ℝ
  b : ℝ
  F1 : ℝ × ℝ
  F2 : ℝ × ℝ
  P : ℝ × ℝ
  h1 : a > b
  h2 : b > 0
  h3 : (P.1^2 / a^2) + (P.2^2 / b^2) = 1

/-- The vectors PF1 and PF2 are perpendicular -/
def vectors_perpendicular (e : Ellipse) : Prop :=
  let PF1 := (e.F1.1 - e.P.1, e.F1.2 - e.P.2)
  let PF2 := (e.F2.1 - e.P.1, e.F2.2 - e.P.2)
  PF1.1 * PF2.1 + PF1.2 * PF2.2 = 0

/-- The area of triangle PF1F2 is 9 -/
def triangle_area_is_9 (e : Ellipse) : Prop :=
  let PF1 := (e.F1.1 - e.P.1, e.F1.2 - e.P.2)
  let PF2 := (e.F2.1 - e.P.1, e.F2.2 - e.P.2)
  abs (PF1.1 * PF2.2 - PF1.2 * PF2.1) / 2 = 9

/-- Main theorem -/
theorem ellipse_b_value (e : Ellipse) 
  (h_perp : vectors_perpendicular e) 
  (h_area : triangle_area_is_9 e) : 
  e.b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_b_value_l513_51394


namespace NUMINAMATH_CALUDE_doctors_visit_insurance_coverage_percentage_l513_51386

def doctors_visit_cost : ℝ := 300
def cats_visit_cost : ℝ := 120
def pet_insurance_coverage : ℝ := 60
def total_paid_after_insurance : ℝ := 135

theorem doctors_visit_insurance_coverage_percentage :
  let total_cost := doctors_visit_cost + cats_visit_cost
  let total_insurance_coverage := total_cost - total_paid_after_insurance
  let doctors_visit_coverage := total_insurance_coverage - pet_insurance_coverage
  doctors_visit_coverage / doctors_visit_cost = 0.75 := by sorry

end NUMINAMATH_CALUDE_doctors_visit_insurance_coverage_percentage_l513_51386


namespace NUMINAMATH_CALUDE_min_value_of_function_l513_51347

theorem min_value_of_function (x : ℝ) (h : x > 2) :
  (x^2 - 4*x + 8) / (x - 2) ≥ 4 ∧ ∃ y > 2, (y^2 - 4*y + 8) / (y - 2) = 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l513_51347


namespace NUMINAMATH_CALUDE_octagon_handshake_distance_l513_51383

theorem octagon_handshake_distance (n : ℕ) (r : ℝ) (h1 : n = 8) (h2 : r = 50) :
  let points := n
  let radius := r
  let connections_per_point := n - 3
  let angle_between_points := 2 * Real.pi / n
  let distance_to_third := radius * Real.sqrt (2 - Real.sqrt 2)
  let total_distance := n * connections_per_point * distance_to_third
  total_distance = 1600 * Real.sqrt (2 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_octagon_handshake_distance_l513_51383


namespace NUMINAMATH_CALUDE_expression_simplification_l513_51314

theorem expression_simplification (q : ℚ) : 
  ((7 * q - 2) + 2 * q * 3) * 4 + (5 + 2 / 2) * (4 * q - 6) = 76 * q - 44 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l513_51314


namespace NUMINAMATH_CALUDE_five_digit_subtraction_l513_51377

theorem five_digit_subtraction (a b c d e : ℕ) : 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 →
  a > 0 →
  (a * 10000 + b * 1000 + c * 100 + d * 10 + e) - 
  (e * 10000 + d * 1000 + c * 100 + b * 10 + a) = 
  (10072 : ℕ) →
  a > e →
  (∀ a' e' : ℕ, a' < 10 ∧ e' < 10 ∧ a' > e' → a' - e' ≥ a - e) →
  a = 9 ∧ e = 7 := by
sorry

end NUMINAMATH_CALUDE_five_digit_subtraction_l513_51377


namespace NUMINAMATH_CALUDE_integral_extrema_l513_51315

open Real MeasureTheory

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ∫ t in (x - a)..(x + a), sqrt (4 * a^2 - t^2)

theorem integral_extrema (a : ℝ) (ha : a > 0) :
  (∀ x : ℝ, |x| ≤ a → f a x ≤ 2 * π * a^2) ∧
  (∀ x : ℝ, |x| ≤ a → f a x ≥ π * a^2) ∧
  (∃ x : ℝ, |x| ≤ a ∧ f a x = 2 * π * a^2) ∧
  (∃ x : ℝ, |x| ≤ a ∧ f a x = π * a^2) :=
sorry

end NUMINAMATH_CALUDE_integral_extrema_l513_51315


namespace NUMINAMATH_CALUDE_tuesday_boot_sales_l513_51342

/-- Represents the sales data for a day -/
structure DailySales where
  shoes : ℕ
  boots : ℕ
  total : ℚ

/-- Represents the pricing and sales information for the shoe store -/
structure ShoeStore where
  shoe_price : ℚ
  boot_price : ℚ
  monday : DailySales
  tuesday : DailySales
  price_difference : boot_price = shoe_price + 15

theorem tuesday_boot_sales (store : ShoeStore) : store.tuesday.boots = 24 :=
  by sorry

end NUMINAMATH_CALUDE_tuesday_boot_sales_l513_51342


namespace NUMINAMATH_CALUDE_car_speed_comparison_l513_51365

/-- Given two cars A and B that travel the same distance, where:
    - Car A travels 1/3 of the distance at u mph, 1/3 at v mph, and 1/3 at w mph
    - Car B travels 1/3 of the time at u mph, 1/3 at v mph, and 1/3 at w mph
    - Average speed of Car A is x mph
    - Average speed of Car B is y mph
    This theorem proves that the average speed of Car A is less than or equal to the average speed of Car B. -/
theorem car_speed_comparison 
  (u v w : ℝ) 
  (hu : u > 0) (hv : v > 0) (hw : w > 0) 
  (x y : ℝ) 
  (hx : x = 3 / (1/u + 1/v + 1/w)) 
  (hy : y = (u + v + w) / 3) : 
  x ≤ y := by
sorry

end NUMINAMATH_CALUDE_car_speed_comparison_l513_51365


namespace NUMINAMATH_CALUDE_x_plus_y_equals_negative_eight_l513_51373

theorem x_plus_y_equals_negative_eight 
  (h1 : |x| + x - y = 16) 
  (h2 : x - |y| + y = -8) : 
  x + y = -8 := by sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_negative_eight_l513_51373


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l513_51392

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, x^2 + b*x - 2*a > 0 ↔ 2 < x ∧ x < 3) → a + b = -8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l513_51392


namespace NUMINAMATH_CALUDE_kit_savings_percentage_l513_51339

/-- The price of the camera lens filter kit -/
def kit_price : ℚ := 75.50

/-- The number of filters in the kit -/
def num_filters : ℕ := 5

/-- The price of the first type of filter -/
def filter_price1 : ℚ := 7.35

/-- The number of filters of the first type -/
def num_filters1 : ℕ := 3

/-- The price of the second type of filter -/
def filter_price2 : ℚ := 12.05

/-- The number of filters of the second type (only 2 are used in the kit) -/
def num_filters2 : ℕ := 2

/-- The price of the third type of filter -/
def filter_price3 : ℚ := 12.50

/-- The number of filters of the third type -/
def num_filters3 : ℕ := 1

/-- The total price of filters if purchased individually -/
def total_individual_price : ℚ :=
  filter_price1 * num_filters1 + filter_price2 * num_filters2 + filter_price3 * num_filters3

/-- The amount saved by purchasing the kit -/
def amount_saved : ℚ := total_individual_price - kit_price

/-- The percentage saved by purchasing the kit -/
def percentage_saved : ℚ := (amount_saved / total_individual_price) * 100

theorem kit_savings_percentage :
  percentage_saved = 28.72 := by sorry

end NUMINAMATH_CALUDE_kit_savings_percentage_l513_51339


namespace NUMINAMATH_CALUDE_train_passing_time_l513_51389

theorem train_passing_time (fast_train_length slow_train_length : ℝ)
  (fast_train_passing_time : ℝ) (h1 : fast_train_length = 315)
  (h2 : slow_train_length = 300) (h3 : fast_train_passing_time = 21) :
  slow_train_length / (fast_train_length / fast_train_passing_time) = 20 :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_l513_51389


namespace NUMINAMATH_CALUDE_right_triangle_incircle_area_ratio_l513_51328

theorem right_triangle_incircle_area_ratio 
  (h r : ℝ) 
  (h_pos : h > 0) 
  (r_pos : r > 0) : 
  ∃ (x y : ℝ), 
    x > 0 ∧ y > 0 ∧ 
    x^2 + y^2 = h^2 ∧ 
    (π * r^2) / ((1/2) * x * y) = π * r / (h + r) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_incircle_area_ratio_l513_51328


namespace NUMINAMATH_CALUDE_sqrt_plus_reciprocal_inequality_l513_51316

theorem sqrt_plus_reciprocal_inequality (x : ℝ) (hx : x > 0) : 
  Real.sqrt x + 1 / Real.sqrt x ≥ 2 ∧ 
  (Real.sqrt x + 1 / Real.sqrt x = 2 ↔ x = 1) := by
sorry

end NUMINAMATH_CALUDE_sqrt_plus_reciprocal_inequality_l513_51316


namespace NUMINAMATH_CALUDE_garden_walkway_area_l513_51357

theorem garden_walkway_area :
  let flower_bed_width : ℕ := 4
  let flower_bed_height : ℕ := 3
  let flower_bed_rows : ℕ := 4
  let flower_bed_columns : ℕ := 3
  let walkway_width : ℕ := 2
  let pond_width : ℕ := 3
  let pond_height : ℕ := 2

  let total_width : ℕ := flower_bed_width * flower_bed_columns + walkway_width * (flower_bed_columns + 1)
  let total_height : ℕ := flower_bed_height * flower_bed_rows + walkway_width * (flower_bed_rows + 1)
  let total_area : ℕ := total_width * total_height

  let pond_area : ℕ := pond_width * pond_height
  let adjusted_area : ℕ := total_area - pond_area

  let flower_bed_area : ℕ := flower_bed_width * flower_bed_height
  let total_flower_bed_area : ℕ := flower_bed_area * flower_bed_rows * flower_bed_columns

  let walkway_area : ℕ := adjusted_area - total_flower_bed_area

  walkway_area = 290 := by sorry

end NUMINAMATH_CALUDE_garden_walkway_area_l513_51357


namespace NUMINAMATH_CALUDE_focus_of_hyperbola_l513_51335

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := y^2 / 3 - x^2 / 6 = 1

/-- Definition of a focus for this hyperbola -/
def is_focus (x y : ℝ) : Prop :=
  x^2 + y^2 = (3 : ℝ)^2 ∧ hyperbola x y

/-- Theorem: (0, 3) is a focus of the given hyperbola -/
theorem focus_of_hyperbola : is_focus 0 3 := by sorry

end NUMINAMATH_CALUDE_focus_of_hyperbola_l513_51335


namespace NUMINAMATH_CALUDE_largest_whole_number_less_than_150_over_9_l513_51336

theorem largest_whole_number_less_than_150_over_9 :
  ∀ x : ℕ, x ≤ 16 ↔ 9 * x < 150 :=
by sorry

end NUMINAMATH_CALUDE_largest_whole_number_less_than_150_over_9_l513_51336


namespace NUMINAMATH_CALUDE_enlarged_parallelepiped_volume_equals_l513_51351

/-- The volume of the set of points that are inside or within one unit of a rectangular parallelepiped with dimensions 4 by 5 by 6 units -/
def enlarged_parallelepiped_volume : ℝ := sorry

/-- The dimensions of the original parallelepiped -/
def original_dimensions : Fin 3 → ℕ
| 0 => 4
| 1 => 5
| 2 => 6
| _ => 0

theorem enlarged_parallelepiped_volume_equals : 
  enlarged_parallelepiped_volume = (1884 + 139 * Real.pi) / 3 := by sorry

end NUMINAMATH_CALUDE_enlarged_parallelepiped_volume_equals_l513_51351


namespace NUMINAMATH_CALUDE_expression_simplification_l513_51384

theorem expression_simplification (a : ℝ) (h : a^2 - a - (7/2) = 0) :
  a^2 - (a - (2*a)/(a+1)) / ((a^2 - 2*a + 1)/(a^2 - 1)) = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l513_51384


namespace NUMINAMATH_CALUDE_speedster_convertibles_count_l513_51317

theorem speedster_convertibles_count (total : ℕ) (speedsters : ℕ) (convertibles : ℕ) : 
  (2 : ℚ) / 3 * total = speedsters →
  (4 : ℚ) / 5 * speedsters = convertibles →
  total - speedsters = 40 →
  convertibles = 64 := by
  sorry

end NUMINAMATH_CALUDE_speedster_convertibles_count_l513_51317


namespace NUMINAMATH_CALUDE_barbara_colored_paper_bundles_l513_51318

/-- Represents the number of sheets in different paper units -/
structure PaperUnits where
  sheets_per_bunch : ℕ
  sheets_per_bundle : ℕ
  sheets_per_heap : ℕ

/-- Represents the quantities of different types of paper -/
structure PaperQuantities where
  bunches_of_white : ℕ
  heaps_of_scrap : ℕ
  total_sheets_removed : ℕ

/-- Calculates the number of bundles of colored paper -/
def bundles_of_colored_paper (units : PaperUnits) (quantities : PaperQuantities) : ℕ :=
  let white_sheets := quantities.bunches_of_white * units.sheets_per_bunch
  let scrap_sheets := quantities.heaps_of_scrap * units.sheets_per_heap
  let colored_sheets := quantities.total_sheets_removed - (white_sheets + scrap_sheets)
  colored_sheets / units.sheets_per_bundle

/-- Theorem stating that Barbara found 3 bundles of colored paper -/
theorem barbara_colored_paper_bundles :
  let units := PaperUnits.mk 4 2 20
  let quantities := PaperQuantities.mk 2 5 114
  bundles_of_colored_paper units quantities = 3 := by
  sorry

end NUMINAMATH_CALUDE_barbara_colored_paper_bundles_l513_51318


namespace NUMINAMATH_CALUDE_three_times_a_plus_square_of_b_l513_51303

/-- The algebraic expression "three times a plus the square of b" is equivalent to 3a + b² -/
theorem three_times_a_plus_square_of_b (a b : ℝ) : 3 * a + b^2 = 3 * a + b^2 := by
  sorry

end NUMINAMATH_CALUDE_three_times_a_plus_square_of_b_l513_51303


namespace NUMINAMATH_CALUDE_system_solution_l513_51320

theorem system_solution (x y : ℝ) : 
  x^2 = 4*y^2 + 19 ∧ x*y + 2*y^2 = 18 → 
  (x = 55 / Real.sqrt 91 ∨ x = -55 / Real.sqrt 91) ∧
  (y = 18 / Real.sqrt 91 ∨ y = -18 / Real.sqrt 91) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l513_51320


namespace NUMINAMATH_CALUDE_division_of_A_by_1001_l513_51308

/-- A number consisting of 1001 sevens -/
def A : ℕ := (10 ^ 1001 - 1) / 9 * 7

/-- The expected quotient when A is divided by 1001 -/
def expected_quotient : ℕ := (10 ^ 1001 - 1) / (9 * 1001) * 777

/-- The expected remainder when A is divided by 1001 -/
def expected_remainder : ℕ := 700

theorem division_of_A_by_1001 :
  (A / 1001 = expected_quotient) ∧ (A % 1001 = expected_remainder) :=
sorry

end NUMINAMATH_CALUDE_division_of_A_by_1001_l513_51308


namespace NUMINAMATH_CALUDE_bus_passengers_l513_51382

theorem bus_passengers (initial : ℕ) (got_on : ℕ) (got_off : ℕ) : 
  initial = 28 → got_on = 7 → got_off = 9 → 
  initial + got_on - got_off = 26 := by
  sorry

end NUMINAMATH_CALUDE_bus_passengers_l513_51382


namespace NUMINAMATH_CALUDE_equation_solution_l513_51338

theorem equation_solution : ∃ m : ℚ, (24 / (3 / 2) = (24 / 3) / m) ∧ m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l513_51338


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l513_51380

theorem partial_fraction_decomposition :
  ∀ (A B C : ℝ),
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 5 →
    (x^2 - 7) / ((x - 2) * (x - 3) * (x - 5)) = A / (x - 2) + B / (x - 3) + C / (x - 5)) ↔
  A = -1 ∧ B = -1 ∧ C = 3 :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l513_51380


namespace NUMINAMATH_CALUDE_mystery_number_multiple_of_four_l513_51346

def mystery_number (k : ℕ) : ℕ := (2*k+2)^2 - (2*k)^2

theorem mystery_number_multiple_of_four (k : ℕ) :
  ∃ m : ℕ, mystery_number k = 4 * m :=
sorry

end NUMINAMATH_CALUDE_mystery_number_multiple_of_four_l513_51346


namespace NUMINAMATH_CALUDE_tan_double_alpha_l513_51390

theorem tan_double_alpha (α : Real) 
  (h : (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1/2) : 
  Real.tan (2 * α) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_alpha_l513_51390


namespace NUMINAMATH_CALUDE_aerith_win_probability_l513_51333

def coin_game (p_heads : ℚ) : ℚ :=
  (1 - p_heads) / (2 - p_heads)

theorem aerith_win_probability :
  let p_heads : ℚ := 4/7
  coin_game p_heads = 7/11 := by sorry

end NUMINAMATH_CALUDE_aerith_win_probability_l513_51333


namespace NUMINAMATH_CALUDE_investment_problem_investment_problem_proof_l513_51374

/-- The investment problem -/
theorem investment_problem (a_investment : ℕ) (b_join_time : ℚ) (profit_ratio : ℚ × ℚ) : ℕ :=
  let a_investment := 27000
  let b_join_time := 7.5
  let profit_ratio := (2, 1)
  let total_months := 12
  let b_investment := a_investment * (total_months / (total_months - b_join_time)) * (profit_ratio.2 / profit_ratio.1)
  36000

/-- Proof of the investment problem -/
theorem investment_problem_proof : investment_problem 27000 (15/2) (2, 1) = 36000 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_investment_problem_proof_l513_51374


namespace NUMINAMATH_CALUDE_anoop_investment_l513_51363

/-- Calculates the investment amount of the second partner in a business partnership --/
def calculate_second_partner_investment (first_partner_investment : ℕ) (first_partner_months : ℕ) (second_partner_months : ℕ) : ℕ :=
  (first_partner_investment * first_partner_months) / second_partner_months

/-- Proves that Anoop's investment is 40,000 given the problem conditions --/
theorem anoop_investment :
  let arjun_investment : ℕ := 20000
  let total_months : ℕ := 12
  let anoop_months : ℕ := 6
  calculate_second_partner_investment arjun_investment total_months anoop_months = 40000 := by
  sorry

#eval calculate_second_partner_investment 20000 12 6

end NUMINAMATH_CALUDE_anoop_investment_l513_51363


namespace NUMINAMATH_CALUDE_functional_equation_solution_l513_51372

/-- A bounded real-valued function satisfying a specific functional equation. -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  (∃ M : ℝ, ∀ x, |f x| ≤ M) ∧ 
  (∀ x y, f (x * f y) + y * f x = x * f y + f (x * y))

/-- The theorem stating the only possible forms of f satisfying the functional equation. -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  (∀ x, f x = 0) ∨ 
  (∀ x, x < 0 → f x = -2*x) ∧ (∀ x, x ≥ 0 → f x = 0) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l513_51372


namespace NUMINAMATH_CALUDE_certain_number_proof_l513_51330

theorem certain_number_proof (x : ℝ) : 
  (x + 40 + 60) / 3 = (10 + 80 + 15) / 3 + 5 → x = 20 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l513_51330


namespace NUMINAMATH_CALUDE_pumpkin_spiderweb_ratio_l513_51305

/-- Represents the Halloween decorations problem --/
def halloween_decorations (total : ℕ) (skulls : ℕ) (broomsticks : ℕ) (spiderwebs : ℕ) 
  (cauldron : ℕ) (budget : ℕ) (left_to_put : ℕ) : Prop :=
  ∃ (pumpkins : ℕ),
    total = skulls + broomsticks + spiderwebs + pumpkins + cauldron + budget + left_to_put ∧
    pumpkins = 2 * spiderwebs

/-- The ratio of pumpkins to spiderwebs is 2:1 given the specified conditions --/
theorem pumpkin_spiderweb_ratio :
  halloween_decorations 83 12 4 12 1 20 10 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_spiderweb_ratio_l513_51305


namespace NUMINAMATH_CALUDE_box_two_three_l513_51323

/-- Define the box operation -/
def box (a b : ℝ) : ℝ := a * (b^2 + 3) - b + 1

/-- Theorem: The value of (2) □ (3) is 22 -/
theorem box_two_three : box 2 3 = 22 := by
  sorry

end NUMINAMATH_CALUDE_box_two_three_l513_51323


namespace NUMINAMATH_CALUDE_remainder_theorem_l513_51322

theorem remainder_theorem : 2^9 * 3^10 + 14 ≡ 2 [ZMOD 25] := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l513_51322


namespace NUMINAMATH_CALUDE_min_value_theorem_l513_51391

/-- A quadratic function f(x) = ax^2 + bx + c with certain properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  second_deriv_positive : 2 * a > 0
  nonnegative : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0

/-- The theorem stating the minimum value of f(1) / f''(0) for the given quadratic function -/
theorem min_value_theorem (f : QuadraticFunction) :
  (∀ g : QuadraticFunction, (g.a + g.b + g.c) / (2 * g.a) ≥ (f.a + f.b + f.c) / (2 * f.a)) →
  (f.a + f.b + f.c) / (2 * f.a) = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l513_51391


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l513_51319

theorem geometric_sequence_first_term :
  ∀ (a r : ℝ),
    a * r^2 = 720 →
    a * r^5 = 5040 →
    a = 720 / 7^(2/3) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l513_51319


namespace NUMINAMATH_CALUDE_ayen_extra_minutes_friday_l513_51341

/-- Represents Ayen's jogging routine for a week --/
structure JoggingRoutine where
  regularMinutes : ℕ  -- Regular minutes jogged per weekday
  weekdays : ℕ        -- Number of weekdays
  tuesdayExtra : ℕ    -- Extra minutes jogged on Tuesday
  totalHours : ℕ      -- Total hours jogged in the week

/-- Calculates the extra minutes jogged on Friday --/
def extraMinutesFriday (routine : JoggingRoutine) : ℕ :=
  routine.totalHours * 60 - (routine.regularMinutes * routine.weekdays + routine.tuesdayExtra)

/-- Theorem stating that Ayen jogged an extra 25 minutes on Friday --/
theorem ayen_extra_minutes_friday :
  ∀ (routine : JoggingRoutine),
    routine.regularMinutes = 30 ∧
    routine.weekdays = 5 ∧
    routine.tuesdayExtra = 5 ∧
    routine.totalHours = 3 →
    extraMinutesFriday routine = 25 := by
  sorry

end NUMINAMATH_CALUDE_ayen_extra_minutes_friday_l513_51341


namespace NUMINAMATH_CALUDE_sum_of_integers_l513_51344

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x^2 + y^2 = 90) 
  (h2 : x * y = 27) : 
  x + y = 12 := by sorry

end NUMINAMATH_CALUDE_sum_of_integers_l513_51344


namespace NUMINAMATH_CALUDE_total_students_eq_920_l513_51375

/-- The number of students in the third school -/
def students_third_school : ℕ := 200

/-- The number of students in the second school -/
def students_second_school : ℕ := students_third_school + 40

/-- The number of students in the first school -/
def students_first_school : ℕ := 2 * students_second_school

/-- The total number of students from all three schools -/
def total_students : ℕ := students_first_school + students_second_school + students_third_school

theorem total_students_eq_920 : total_students = 920 := by
  sorry

end NUMINAMATH_CALUDE_total_students_eq_920_l513_51375


namespace NUMINAMATH_CALUDE_evaluate_expression_l513_51376

theorem evaluate_expression : (47^2 - 28^2) + 100 = 1525 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l513_51376


namespace NUMINAMATH_CALUDE_compound_interest_problem_l513_51369

/-- Calculate the compound interest given principal, rate, and time -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

/-- Calculate the total amount returned after compound interest -/
def total_amount (principal : ℝ) (interest : ℝ) : ℝ :=
  principal + interest

theorem compound_interest_problem (P : ℝ) :
  compound_interest P 0.05 2 = 492 →
  total_amount P 492 = 5292 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l513_51369


namespace NUMINAMATH_CALUDE_min_value_divisible_by_72_l513_51361

def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = b * k

theorem min_value_divisible_by_72 (x y : ℕ) (h1 : x ≥ 4) 
  (h2 : is_divisible_by (98348 * 10 + x * 10 + y) 72) : y = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_divisible_by_72_l513_51361


namespace NUMINAMATH_CALUDE_librarian_crates_l513_51379

theorem librarian_crates (novels comics documentaries albums : ℕ) 
  (items_per_crate : ℕ) (h1 : novels = 145) (h2 : comics = 271) 
  (h3 : documentaries = 419) (h4 : albums = 209) (h5 : items_per_crate = 9) : 
  (novels + comics + documentaries + albums + items_per_crate - 1) / items_per_crate = 117 := by
  sorry

end NUMINAMATH_CALUDE_librarian_crates_l513_51379


namespace NUMINAMATH_CALUDE_digit_sum_proof_l513_51301

theorem digit_sum_proof (P Q R : ℕ) : 
  P ∈ Finset.range 9 → 
  Q ∈ Finset.range 9 → 
  R ∈ Finset.range 9 → 
  P + P + P = 2022 → 
  P + Q + R = 15 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_proof_l513_51301


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l513_51345

theorem part_to_whole_ratio (N : ℝ) (part : ℝ) : 
  (1/4 : ℝ) * part * (2/5 : ℝ) * N = 20 →
  (40/100 : ℝ) * N = 240 →
  part / ((2/5 : ℝ) * N) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l513_51345


namespace NUMINAMATH_CALUDE_exactly_one_correct_statement_l513_51385

/-- Rules of the oblique projection drawing method -/
structure ObliqueProjectionRules where
  parallelism_preserved : Bool
  x_axis_length_preserved : Bool
  y_axis_length_halved : Bool

/-- Statements about intuitive diagrams -/
structure IntuitiveDiagramStatements where
  equal_angles_preserved : Bool
  equal_segments_preserved : Bool
  longest_segment_preserved : Bool
  midpoint_preserved : Bool

/-- Theorem: Exactly one statement is correct given the oblique projection rules -/
theorem exactly_one_correct_statement 
  (rules : ObliqueProjectionRules)
  (statements : IntuitiveDiagramStatements) :
  rules.parallelism_preserved ∧
  rules.x_axis_length_preserved ∧
  rules.y_axis_length_halved →
  (statements.equal_angles_preserved = false) ∧
  (statements.equal_segments_preserved = false) ∧
  (statements.longest_segment_preserved = false) ∧
  (statements.midpoint_preserved = true) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_correct_statement_l513_51385


namespace NUMINAMATH_CALUDE_f_properties_l513_51393

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 - a*x + a)

theorem f_properties (a : ℝ) (h : a > 2) :
  let f' := deriv (f a)
  ∃ (S₁ S₂ S₃ : Set ℝ),
    (f' 0 = a) ∧
    (S₁ = Set.Iio 0) ∧
    (S₂ = Set.Ioi (a - 2)) ∧
    (S₃ = Set.Ioo 0 (a - 2)) ∧
    (StrictMonoOn (f a) S₁) ∧
    (StrictMonoOn (f a) S₂) ∧
    (StrictAntiOn (f a) S₃) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l513_51393


namespace NUMINAMATH_CALUDE_distribute_five_objects_l513_51313

/-- The number of ways to distribute n distinguishable objects into 2 indistinguishable containers -/
def distribute_objects (n : ℕ) : ℕ :=
  (2^n - 2) / 2 + 2

/-- Theorem: There are 17 ways to distribute 5 distinguishable objects into 2 indistinguishable containers -/
theorem distribute_five_objects : distribute_objects 5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_objects_l513_51313


namespace NUMINAMATH_CALUDE_sufficient_material_l513_51311

-- Define the surface area of a rectangular box
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

-- Define the volume of a rectangular box
def volume (l w h : ℝ) : ℝ := l * w * h

-- Theorem statement
theorem sufficient_material :
  ∃ (l w h : ℝ), l > 0 ∧ w > 0 ∧ h > 0 ∧ 
  surface_area l w h = 958 ∧ 
  volume l w h ≥ 1995 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_material_l513_51311


namespace NUMINAMATH_CALUDE_ranch_minimum_animals_l513_51353

theorem ranch_minimum_animals : ∀ (ponies horses : ℕ),
  ponies > 0 →
  horses = ponies + 4 →
  (3 * ponies) % 10 = 0 →
  (5 * ((3 * ponies) / 10)) % 8 = 0 →
  ponies + horses ≥ 36 :=
by
  sorry

end NUMINAMATH_CALUDE_ranch_minimum_animals_l513_51353


namespace NUMINAMATH_CALUDE_always_even_l513_51397

theorem always_even (m n : ℤ) : 
  ∃ k : ℤ, (2*m + 1)^2 + 3*(2*m + 1)*(2*n + 1) = 2*k := by
sorry

end NUMINAMATH_CALUDE_always_even_l513_51397


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l513_51368

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (W : Point)
  (X : Point)
  (Y : Point)
  (Z : Point)

/-- Checks if a quadrilateral is convex -/
def is_convex (q : Quadrilateral) : Prop :=
  sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

/-- Checks if two line segments intersect at right angles -/
def intersect_at_right_angle (p1 p2 p3 p4 : Point) : Prop :=
  sorry

/-- Checks if two line segments bisect each other -/
def bisect_each_other (p1 p2 p3 p4 : Point) : Prop :=
  sorry

theorem quadrilateral_diagonal_length 
  (q : Quadrilateral)
  (h1 : is_convex q)
  (h2 : distance q.W q.Y = 15)
  (h3 : distance q.X q.Z = 20)
  (h4 : distance q.W q.X = 18)
  (P : Point)
  (h5 : intersect_at_right_angle q.W q.X q.Y q.Z)
  (h6 : bisect_each_other q.W q.X q.Y q.Z) :
  distance q.W P = 9 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l513_51368


namespace NUMINAMATH_CALUDE_raisin_cost_fraction_nut_to_dried_fruit_ratio_dried_fruit_percentage_l513_51362

/-- Represents the trail mix problem with raisins, nuts, and dried fruit. -/
structure TrailMix where
  x : ℝ
  raisin_cost : ℝ
  raisin_weight : ℝ := 3 * x
  nut_weight : ℝ := 4 * x
  dried_fruit_weight : ℝ := 5 * x
  nut_cost : ℝ := 3 * raisin_cost
  dried_fruit_cost : ℝ := 1.5 * raisin_cost

/-- The total cost of raisins is 1/7.5 of the total cost of the mixture. -/
theorem raisin_cost_fraction (mix : TrailMix) :
  (mix.raisin_weight * mix.raisin_cost) / 
  (mix.raisin_weight * mix.raisin_cost + mix.nut_weight * mix.nut_cost + mix.dried_fruit_weight * mix.dried_fruit_cost) = 1 / 7.5 := by
  sorry

/-- The ratio of the cost of nuts to the cost of dried fruit is 2:1. -/
theorem nut_to_dried_fruit_ratio (mix : TrailMix) :
  mix.nut_cost / mix.dried_fruit_cost = 2 := by
  sorry

/-- The total cost of dried fruit is 50% of the total cost of raisins and nuts combined. -/
theorem dried_fruit_percentage (mix : TrailMix) :
  (mix.dried_fruit_weight * mix.dried_fruit_cost) / 
  (mix.raisin_weight * mix.raisin_cost + mix.nut_weight * mix.nut_cost) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_raisin_cost_fraction_nut_to_dried_fruit_ratio_dried_fruit_percentage_l513_51362


namespace NUMINAMATH_CALUDE_opposites_and_reciprocals_l513_51396

theorem opposites_and_reciprocals (a b c d : ℝ) 
  (h1 : a = -b) -- a and b are opposites
  (h2 : c * d = 1) -- c and d are reciprocals
  : 3 * (a + b) - 4 * c * d = -4 := by
  sorry

end NUMINAMATH_CALUDE_opposites_and_reciprocals_l513_51396


namespace NUMINAMATH_CALUDE_parabola_coefficients_l513_51352

/-- A parabola with equation y = ax^2 + bx + c, vertex at (4, 5), and passing through (2, 3) has coefficients (a, b, c) = (-1/2, 4, -3) -/
theorem parabola_coefficients :
  ∀ (a b c : ℝ),
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (5 : ℝ) = a * 4^2 + b * 4 + c →
  (∀ x : ℝ, a * (x - 4)^2 + 5 = a * x^2 + b * x + c) →
  (3 : ℝ) = a * 2^2 + b * 2 + c →
  (a = -1/2 ∧ b = 4 ∧ c = -3) :=
by sorry

end NUMINAMATH_CALUDE_parabola_coefficients_l513_51352


namespace NUMINAMATH_CALUDE_min_occupied_seats_l513_51332

/-- Represents the seating arrangement problem --/
def SeatingArrangement (total_seats : ℕ) (pattern : List ℕ) (occupied : ℕ) : Prop :=
  -- The total number of seats is 150
  total_seats = 150 ∧
  -- The pattern alternates between 4 and 3 empty seats
  pattern = [4, 3] ∧
  -- The occupied seats ensure the next person must sit next to someone
  occupied ≥ 
    -- Calculate the minimum number of occupied seats
    let full_units := total_seats / (pattern.sum + pattern.length)
    let remaining_seats := total_seats % (pattern.sum + pattern.length)
    let seats_in_full_units := full_units * pattern.length
    let additional_seats := if remaining_seats ≥ pattern.head! then 2 else 0
    seats_in_full_units + additional_seats

/-- The theorem stating the minimum number of occupied seats --/
theorem min_occupied_seats :
  ∃ (occupied : ℕ), SeatingArrangement 150 [4, 3] occupied ∧ occupied = 50 := by
  sorry

end NUMINAMATH_CALUDE_min_occupied_seats_l513_51332


namespace NUMINAMATH_CALUDE_water_bottle_theorem_l513_51340

def water_bottle_problem (water_A : ℝ) (extra_B : ℝ) (extra_C : ℝ) : Prop :=
  let water_B : ℝ := water_A + extra_B
  let water_C_ml : ℝ := (water_B / 10) * 1000 + extra_C
  let water_C_L : ℝ := water_C_ml / 1000
  water_C_L = 4.94

theorem water_bottle_theorem :
  water_bottle_problem 3.8 8.4 3720 := by
  sorry

end NUMINAMATH_CALUDE_water_bottle_theorem_l513_51340


namespace NUMINAMATH_CALUDE_hyperbola_dimensions_l513_51387

/-- A hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0
  foci_to_asymptote : ℝ
  asymptote_slope : ℝ
  foci_distance : foci_to_asymptote = 2
  asymptote_parallel : asymptote_slope = 1/2

/-- The theorem stating the specific dimensions of the hyperbola -/
theorem hyperbola_dimensions (h : Hyperbola) : h.a = 4 ∧ h.b = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_dimensions_l513_51387


namespace NUMINAMATH_CALUDE_martha_butterflies_l513_51388

/-- The number of black butterflies in Martha's collection --/
def black_butterflies (total blue yellow : ℕ) : ℕ :=
  total - blue - yellow

/-- Theorem stating the number of black butterflies in Martha's collection --/
theorem martha_butterflies :
  ∀ (total blue yellow : ℕ),
    total = 11 →
    blue = 4 →
    blue = 2 * yellow →
    black_butterflies total blue yellow = 5 := by
  sorry

end NUMINAMATH_CALUDE_martha_butterflies_l513_51388


namespace NUMINAMATH_CALUDE_triangle_area_l513_51398

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the area of the triangle is √2 under the following conditions:
    1. b = a*cos(C) + c*cos(B)
    2. CA · CB = 1 (dot product)
    3. c = 2 -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  b = a * Real.cos C + c * Real.cos B →
  a * c * Real.cos B = 1 →
  c = 2 →
  (1/2) * a * b * Real.sin C = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l513_51398


namespace NUMINAMATH_CALUDE_f_odd_f_max_on_interval_l513_51359

/-- A function satisfying the given conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The function f satisfies the additive property -/
axiom f_additive (x y : ℝ) : f (x + y) = f x + f y

/-- The function f is negative for positive inputs -/
axiom f_neg_for_pos (x : ℝ) (h : x > 0) : f x < 0

/-- The value of f at 1 is -2 -/
axiom f_one : f 1 = -2

/-- f is an odd function -/
theorem f_odd : ∀ x, f (-x) = -f x := by sorry

/-- The maximum value of f on [-3, 3] is 6 -/
theorem f_max_on_interval : ∃ x ∈ Set.Icc (-3) 3, ∀ y ∈ Set.Icc (-3) 3, f y ≤ f x ∧ f x = 6 := by sorry

end NUMINAMATH_CALUDE_f_odd_f_max_on_interval_l513_51359


namespace NUMINAMATH_CALUDE_cost_price_determination_l513_51321

theorem cost_price_determination (selling_price_profit selling_price_loss : ℝ) 
  (h : selling_price_profit = 54 ∧ selling_price_loss = 40) :
  ∃ cost_price : ℝ, 
    selling_price_profit - cost_price = cost_price - selling_price_loss ∧ 
    cost_price = 47 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_determination_l513_51321


namespace NUMINAMATH_CALUDE_melanie_attended_games_l513_51367

theorem melanie_attended_games 
  (total_games : ℕ) 
  (missed_games : ℕ) 
  (attended_games : ℕ) 
  (h1 : total_games = 64) 
  (h2 : missed_games = 32) 
  (h3 : attended_games = total_games - missed_games) : 
  attended_games = 32 :=
by sorry

end NUMINAMATH_CALUDE_melanie_attended_games_l513_51367


namespace NUMINAMATH_CALUDE_real_part_of_z_l513_51350

theorem real_part_of_z (z : ℂ) (h : z * (2 + Complex.I) = 1) : 
  Complex.re z = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l513_51350


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_complement_B_l513_51378

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 3*x - 10 < 0}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≥ 0}

-- Define the universal set R (real numbers)
def R : Type := ℝ

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x : ℝ | -5 < x ∧ x ≤ -1} := by sorry

-- Theorem for A ∪ (∁ₖ B)
theorem union_A_complement_B : A ∪ (Set.univ \ B) = {x : ℝ | -5 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_complement_B_l513_51378


namespace NUMINAMATH_CALUDE_triangle_properties_l513_51325

-- Define a triangle with given properties
structure Triangle where
  a : ℝ  -- side BC
  m : ℝ  -- altitude from B to AC
  k : ℝ  -- median to side AC
  a_pos : 0 < a
  m_pos : 0 < m
  k_pos : 0 < k

-- Define the theorem
theorem triangle_properties (t : Triangle) :
  let b := 2 * Real.sqrt (t.k^2 + t.a * (t.a - Real.sqrt (4 * t.k^2 - t.m^2)))
  let c := 2 * Real.sqrt (t.k^2 + (t.a/2) * ((t.a/2) - Real.sqrt (4 * t.k^2 - t.m^2)))
  (∃ (γ β : ℝ),
    b > 0 ∧
    c > 0 ∧
    Real.sin γ = t.m / b ∧
    Real.sin β = t.m / c) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l513_51325


namespace NUMINAMATH_CALUDE_fourth_rectangle_area_l513_51302

/-- Represents a rectangle divided into four smaller rectangles -/
structure DividedRectangle where
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ

/-- The theorem stating that if three of the areas are 24, 15, and 9, then the fourth is 15 -/
theorem fourth_rectangle_area (rect : DividedRectangle) 
  (h1 : rect.area1 = 24)
  (h2 : rect.area2 = 15)
  (h3 : rect.area3 = 9) :
  rect.area4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_fourth_rectangle_area_l513_51302


namespace NUMINAMATH_CALUDE_fence_perimeter_l513_51304

/-- The number of posts in the fence -/
def total_posts : ℕ := 24

/-- The width of each post in inches -/
def post_width : ℚ := 5

/-- The space between adjacent posts in feet -/
def post_spacing : ℚ := 6

/-- The number of posts on each side of the square fence -/
def posts_per_side : ℕ := 7

/-- The length of one side of the square fence in feet -/
def side_length : ℚ := post_spacing * 6 + posts_per_side * (post_width / 12)

/-- The outer perimeter of the square fence in feet -/
def outer_perimeter : ℚ := 4 * side_length

theorem fence_perimeter : outer_perimeter = 156 := by sorry

end NUMINAMATH_CALUDE_fence_perimeter_l513_51304


namespace NUMINAMATH_CALUDE_dividend_calculation_l513_51324

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17) 
  (h2 : quotient = 9) 
  (h3 : remainder = 10) : 
  divisor * quotient + remainder = 163 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l513_51324


namespace NUMINAMATH_CALUDE_base_eight_addition_sum_l513_51310

/-- Given distinct non-zero digits S, H, and E less than 8 that satisfy the base-8 addition
    SEH₈ + EHS₈ = SHE₈, prove that their sum in base 10 is 6. -/
theorem base_eight_addition_sum (S H E : ℕ) : 
  S ≠ 0 → H ≠ 0 → E ≠ 0 →
  S < 8 → H < 8 → E < 8 →
  S ≠ H → S ≠ E → H ≠ E →
  S * 64 + E * 8 + H + E * 64 + H * 8 + S = S * 64 + H * 8 + E →
  S + H + E = 6 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_addition_sum_l513_51310


namespace NUMINAMATH_CALUDE_angela_problems_count_l513_51329

def total_problems : ℕ := 20
def martha_problems : ℕ := 2
def jenna_problems : ℕ := 4 * martha_problems - 2
def mark_problems : ℕ := jenna_problems / 2

theorem angela_problems_count : 
  total_problems - (martha_problems + jenna_problems + mark_problems) = 9 := by
  sorry

end NUMINAMATH_CALUDE_angela_problems_count_l513_51329
