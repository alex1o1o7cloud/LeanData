import Mathlib

namespace NUMINAMATH_GPT_timothy_total_cost_l1449_144936

-- Define the costs of the individual items
def costOfLand (acres : Nat) (cost_per_acre : Nat) : Nat :=
  acres * cost_per_acre

def costOfHouse : Nat :=
  120000

def costOfCows (number_of_cows : Nat) (cost_per_cow : Nat) : Nat :=
  number_of_cows * cost_per_cow

def costOfChickens (number_of_chickens : Nat) (cost_per_chicken : Nat) : Nat :=
  number_of_chickens * cost_per_chicken

def installationCost (hours : Nat) (cost_per_hour : Nat) (equipment_fee : Nat) : Nat :=
  (hours * cost_per_hour) + equipment_fee

-- Define the total cost function
def totalCost : Nat :=
  costOfLand 30 20 +
  costOfHouse +
  costOfCows 20 1000 +
  costOfChickens 100 5 +
  installationCost 6 100 6000

-- Theorem to state the total cost
theorem timothy_total_cost : totalCost = 147700 :=
by
  -- Placeholder for the proof, for now leave it as sorry
  sorry

end NUMINAMATH_GPT_timothy_total_cost_l1449_144936


namespace NUMINAMATH_GPT_regular_polygon_sides_and_interior_angle_l1449_144959

theorem regular_polygon_sides_and_interior_angle (n : ℕ) (H : (n - 2) * 180 = 3 * 360 + 180) :
  n = 9 ∧ (n - 2) * 180 / n = 140 :=
by
-- This marks the start of the proof, but the proof is omitted.
sorry

end NUMINAMATH_GPT_regular_polygon_sides_and_interior_angle_l1449_144959


namespace NUMINAMATH_GPT_sum_of_two_squares_l1449_144957

theorem sum_of_two_squares (n : ℕ) (h : ∀ m, m = n → n = 2 ∨ (n = 2 * 10 + m) → n % 8 = m) :
  (∃ a b : ℕ, n = a^2 + b^2) ↔ n = 2 := by
  sorry

end NUMINAMATH_GPT_sum_of_two_squares_l1449_144957


namespace NUMINAMATH_GPT_next_perfect_square_l1449_144998

theorem next_perfect_square (n : ℤ) (hn : Even n) (x : ℤ) (hx : x = n^2) : 
  ∃ y : ℤ, y = x + 2 * n + 1 ∧ (∃ m : ℤ, y = m^2) ∧ m > n :=
by
  sorry

end NUMINAMATH_GPT_next_perfect_square_l1449_144998


namespace NUMINAMATH_GPT_evaluate_expression_l1449_144961

theorem evaluate_expression (a : ℚ) (h : a = 3/2) : 
  ((5 * a^2 - 13 * a + 4) * (2 * a - 3)) = 0 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1449_144961


namespace NUMINAMATH_GPT_binders_can_bind_books_l1449_144902

theorem binders_can_bind_books :
  (∀ (binders books days : ℕ), binders * days * books = 18 * 10 * 900 → 
    11 * binders * 12 = 660) :=
sorry

end NUMINAMATH_GPT_binders_can_bind_books_l1449_144902


namespace NUMINAMATH_GPT_quadratic_eq_coeffs_l1449_144962

theorem quadratic_eq_coeffs (x : ℝ) : 
  ∃ a b c : ℝ, 3 * x^2 + 1 - 6 * x = a * x^2 + b * x + c ∧ a = 3 ∧ b = -6 ∧ c = 1 :=
by sorry

end NUMINAMATH_GPT_quadratic_eq_coeffs_l1449_144962


namespace NUMINAMATH_GPT_rachel_age_when_emily_half_age_l1449_144940

-- Conditions
def Emily_current_age : ℕ := 20
def Rachel_current_age : ℕ := 24

-- Proof statement
theorem rachel_age_when_emily_half_age :
  ∃ x : ℕ, (Emily_current_age - x = (Rachel_current_age - x) / 2) ∧ (Rachel_current_age - x = 8) := 
sorry

end NUMINAMATH_GPT_rachel_age_when_emily_half_age_l1449_144940


namespace NUMINAMATH_GPT_find_horizontal_length_l1449_144933

variable (v h : ℝ)

-- Conditions
def is_horizontal_length_of_rectangle_perimeter_54_and_vertical_plus_3 (v h : ℝ) : Prop :=
  2 * h + 2 * v = 54 ∧ h = v + 3

-- The proof we aim to show
theorem find_horizontal_length (v h : ℝ) :
  is_horizontal_length_of_rectangle_perimeter_54_and_vertical_plus_3 v h → h = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_horizontal_length_l1449_144933


namespace NUMINAMATH_GPT_triangle_area_l1449_144951

theorem triangle_area (a b c p : ℕ) (h_ratio : a = 5 * p) (h_ratio2 : b = 12 * p) (h_ratio3 : c = 13 * p) (h_perimeter : a + b + c = 300) : 
  (1 / 4) * Real.sqrt ((a + b + c) * (a + b - c) * (a + c - b) * (b + c - a)) = 3000 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_area_l1449_144951


namespace NUMINAMATH_GPT_relationship_among_abc_l1449_144972

noncomputable def a : ℝ := Real.log (1/4) / Real.log 2
noncomputable def b : ℝ := 2.1^(1/3)
noncomputable def c : ℝ := (4/5)^2

theorem relationship_among_abc : a < c ∧ c < b :=
by
  -- Definitions
  have ha : a = Real.log (1/4) / Real.log 2 := rfl
  have hb : b = 2.1^(1/3) := rfl
  have hc : c = (4/5)^2 := rfl
  sorry

end NUMINAMATH_GPT_relationship_among_abc_l1449_144972


namespace NUMINAMATH_GPT_find_line_equation_l1449_144971

-- Define the first line equation
def line1 (x y : ℝ) : Prop := 2 * x - y - 5 = 0

-- Define the second line equation
def line2 (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the parallel line equation with a variable constant term
def line_parallel (x y m : ℝ) : Prop := 3 * x + y + m = 0

-- State the intersection point
def intersect_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- The desired equation of the line passing through the intersection point
theorem find_line_equation (x y : ℝ) (h : intersect_point x y) : ∃ m, line_parallel x y m := by
  sorry

end NUMINAMATH_GPT_find_line_equation_l1449_144971


namespace NUMINAMATH_GPT_number_of_goats_l1449_144943

-- Mathematical definitions based on the conditions
def number_of_hens : ℕ := 10
def total_cost : ℤ := 2500
def price_per_hen : ℤ := 50
def price_per_goat : ℤ := 400

-- Prove the number of goats
theorem number_of_goats (G : ℕ) : 
  number_of_hens * price_per_hen + G * price_per_goat = total_cost ↔ G = 5 := 
by
  sorry

end NUMINAMATH_GPT_number_of_goats_l1449_144943


namespace NUMINAMATH_GPT_cyclist_speed_ratio_l1449_144979

-- Define the conditions
def speeds_towards_each_other (v1 v2 : ℚ) : Prop :=
  v1 + v2 = 25

def speeds_apart_with_offset (v1 v2 : ℚ) : Prop :=
  v1 - v2 = 10 / 3

-- The proof problem to show the required ratio of speeds
theorem cyclist_speed_ratio (v1 v2 : ℚ) (h1 : speeds_towards_each_other v1 v2) (h2 : speeds_apart_with_offset v1 v2) :
  v1 / v2 = 17 / 13 :=
sorry

end NUMINAMATH_GPT_cyclist_speed_ratio_l1449_144979


namespace NUMINAMATH_GPT_expression_equals_a5_l1449_144985

theorem expression_equals_a5 (a : ℝ) : a^4 * a = a^5 := 
by sorry

end NUMINAMATH_GPT_expression_equals_a5_l1449_144985


namespace NUMINAMATH_GPT_point_cannot_exist_on_line_l1449_144917

theorem point_cannot_exist_on_line (m k : ℝ) (h : m * k > 0) : ¬ (2000 * m + k = 0) :=
sorry

end NUMINAMATH_GPT_point_cannot_exist_on_line_l1449_144917


namespace NUMINAMATH_GPT_correct_operation_l1449_144915

theorem correct_operation : ∀ (m : ℤ), (-m + 2) * (-m - 2) = m^2 - 4 :=
by
  intro m
  sorry

end NUMINAMATH_GPT_correct_operation_l1449_144915


namespace NUMINAMATH_GPT_annual_salary_is_20_l1449_144950

-- Define the conditions
variable (months_worked : ℝ) (total_received : ℝ) (turban_price : ℝ)
variable (S : ℝ)

-- Actual values from the problem
axiom h1 : months_worked = 9 / 12
axiom h2 : total_received = 55
axiom h3 : turban_price = 50

-- Define the statement to prove
theorem annual_salary_is_20 : S = 20 := by
  -- Conditions derived from the problem
  have cash_received := total_received - turban_price
  have fraction_of_salary := months_worked * S
  -- Given the servant worked 9 months and received Rs. 55 including Rs. 50 turban
  have : cash_received = fraction_of_salary := by sorry
  -- Solving the equation 3/4 S = 5 for S
  have : S = 20 := by sorry
  sorry -- Final proof step

end NUMINAMATH_GPT_annual_salary_is_20_l1449_144950


namespace NUMINAMATH_GPT_find_a_l1449_144912

theorem find_a (a x : ℝ) (h : x = -1) (heq : -2 * (x - a) = 4) : a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1449_144912


namespace NUMINAMATH_GPT_total_amount_paid_l1449_144946

def jacket_price : ℝ := 150
def sale_discount : ℝ := 0.25
def coupon_discount : ℝ := 10
def sales_tax : ℝ := 0.10

theorem total_amount_paid : 
  (jacket_price * (1 - sale_discount) - coupon_discount) * (1 + sales_tax) = 112.75 := 
by
  sorry

end NUMINAMATH_GPT_total_amount_paid_l1449_144946


namespace NUMINAMATH_GPT_sequence_general_term_l1449_144996

theorem sequence_general_term (a : ℕ → ℕ) 
  (h₀ : a 1 = 4) 
  (h₁ : ∀ n : ℕ, a (n + 1) = 2 * a n + n^2) : 
  ∀ n : ℕ, a n = 5 * 2^n - n^2 - 2*n - 3 :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l1449_144996


namespace NUMINAMATH_GPT_temperature_difference_l1449_144907

def h : ℤ := 10
def l : ℤ := -5
def d : ℤ := 15

theorem temperature_difference : h - l = d :=
by
  rw [h, l, d]
  sorry

end NUMINAMATH_GPT_temperature_difference_l1449_144907


namespace NUMINAMATH_GPT_flat_fee_rate_l1449_144982

-- Definitions for the variables
variable (F n : ℝ)

-- Conditions based on the problem statement
axiom mark_cost : F + 4.6 * n = 310
axiom lucy_cost : F + 6.2 * n = 410

-- Problem Statement
theorem flat_fee_rate : F = 22.5 ∧ n = 62.5 :=
by
  sorry

end NUMINAMATH_GPT_flat_fee_rate_l1449_144982


namespace NUMINAMATH_GPT_range_of_m_l1449_144927

def positive_numbers (a b : ℝ) : Prop := a > 0 ∧ b > 0

def equation_condition (a b : ℝ) : Prop := 9 * a + b = a * b

def inequality_for_any_x (a b m : ℝ) : Prop := ∀ x : ℝ, a + b ≥ -x^2 + 2 * x + 18 - m

theorem range_of_m :
  ∀ (a b m : ℝ),
    positive_numbers a b →
    equation_condition a b →
    inequality_for_any_x a b m →
    m ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1449_144927


namespace NUMINAMATH_GPT_bus_profit_problem_l1449_144944

def independent_variable := "number of passengers per month"
def dependent_variable := "monthly profit"

-- Given monthly profit equation
def monthly_profit (x : ℕ) : ℤ := 2 * x - 4000

-- 1. Independent and Dependent variables
def independent_variable_defined_correctly : Prop :=
  independent_variable = "number of passengers per month"

def dependent_variable_defined_correctly : Prop :=
  dependent_variable = "monthly profit"

-- 2. Minimum passenger volume to avoid losses
def minimum_passenger_volume_no_loss : Prop :=
  ∀ x : ℕ, (monthly_profit x >= 0) → (x >= 2000)

-- 3. Monthly profit prediction for 4230 passengers
def monthly_profit_prediction_4230 (x : ℕ) : Prop :=
  x = 4230 → monthly_profit x = 4460

theorem bus_profit_problem :
  independent_variable_defined_correctly ∧
  dependent_variable_defined_correctly ∧
  minimum_passenger_volume_no_loss ∧
  monthly_profit_prediction_4230 4230 :=
by
  sorry

end NUMINAMATH_GPT_bus_profit_problem_l1449_144944


namespace NUMINAMATH_GPT_percentage_difference_liliane_alice_l1449_144923

theorem percentage_difference_liliane_alice :
  let J := 200
  let L := 1.30 * J
  let A := 1.15 * J
  (L - A) / A * 100 = 13.04 :=
by
  sorry

end NUMINAMATH_GPT_percentage_difference_liliane_alice_l1449_144923


namespace NUMINAMATH_GPT_find_magical_points_on_specific_quad_find_t_for_unique_magical_point_l1449_144963

-- Define what it means to be a "magical point"
def is_magical_point (m : ℝ) (A : ℝ × ℝ) : Prop :=
  A = (m, 2 * m)

-- Specialize for the specific quadratic function y = x^2 - x - 4
def on_specific_quadratic (m : ℝ) (A : ℝ × ℝ) : Prop :=
  A = (m, m^2 - m - 4)

-- Theorem for part 1: Find the magical points on y = x^2 - x - 4
theorem find_magical_points_on_specific_quad (m : ℝ) (A : ℝ × ℝ) :
  is_magical_point m A ∧ on_specific_quadratic m A →
  (A = (4, 8) ∨ A = (-1, -2)) :=
sorry

-- Define the quadratic function for part 2
def on_general_quadratic (t m : ℝ) (A : ℝ × ℝ) : Prop :=
  A = (m, t * m^2 + (t-2) * m - 4)

-- Theorem for part 2: Find the t values for unique magical points
theorem find_t_for_unique_magical_point (t m : ℝ) (A : ℝ × ℝ) :
  ( ∀ m, is_magical_point m A ∧ on_general_quadratic t m A → 
    (t * m^2 + (t-4) * m - 4 = 0) ) → 
  ( ∃! m, is_magical_point m A ∧ on_general_quadratic t m A ) →
  t = -4 :=
sorry

end NUMINAMATH_GPT_find_magical_points_on_specific_quad_find_t_for_unique_magical_point_l1449_144963


namespace NUMINAMATH_GPT_bottle_caps_sum_l1449_144990

theorem bottle_caps_sum : 
  let starting_caps := 91
  let found_caps := 88
  starting_caps + found_caps = 179 :=
by
  sorry

end NUMINAMATH_GPT_bottle_caps_sum_l1449_144990


namespace NUMINAMATH_GPT_spaceship_not_moving_time_l1449_144913

-- Definitions based on the conditions given
def total_journey_time : ℕ := 3 * 24  -- 3 days in hours

def first_travel_time : ℕ := 10
def first_break_time : ℕ := 3
def second_travel_time : ℕ := 10
def second_break_time : ℕ := 1

def subsequent_travel_period : ℕ := 11  -- 11 hours traveling, then 1 hour break

-- Function to compute total break time
def total_break_time (total_travel_time : ℕ) : ℕ :=
  let remaining_time := total_journey_time - (first_travel_time + first_break_time + second_travel_time + second_break_time)
  let subsequent_breaks := remaining_time / subsequent_travel_period
  first_break_time + second_break_time + subsequent_breaks

theorem spaceship_not_moving_time : total_break_time total_journey_time = 8 := by
  sorry

end NUMINAMATH_GPT_spaceship_not_moving_time_l1449_144913


namespace NUMINAMATH_GPT_tangent_line_and_point_l1449_144945

theorem tangent_line_and_point (x0 y0 k: ℝ) (hx0 : x0 ≠ 0) 
  (hC : y0 = x0^3 - 3 * x0^2 + 2 * x0) (hl : y0 = k * x0) 
  (hk_tangent : k = 3 * x0^2 - 6 * x0 + 2) : 
  (k = -1/4) ∧ (x0 = 3/2) ∧ (y0 = -3/8) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_and_point_l1449_144945


namespace NUMINAMATH_GPT_increase_in_area_l1449_144949

theorem increase_in_area :
  let original_side := 6
  let increase := 1
  let new_side := original_side + increase
  let original_area := original_side * original_side
  let new_area := new_side * new_side
  let area_increase := new_area - original_area
  area_increase = 13 :=
by
  let original_side := 6
  let increase := 1
  let new_side := original_side + increase
  let original_area := original_side * original_side
  let new_area := new_side * new_side
  let area_increase := new_area - original_area
  sorry

end NUMINAMATH_GPT_increase_in_area_l1449_144949


namespace NUMINAMATH_GPT_average_calls_per_day_l1449_144988

def calls_Monday : ℕ := 35
def calls_Tuesday : ℕ := 46
def calls_Wednesday : ℕ := 27
def calls_Thursday : ℕ := 61
def calls_Friday : ℕ := 31

def total_calls : ℕ := calls_Monday + calls_Tuesday + calls_Wednesday + calls_Thursday + calls_Friday
def number_of_days : ℕ := 5

theorem average_calls_per_day : (total_calls / number_of_days) = 40 := 
by 
  -- calculations and proof steps go here.
  sorry

end NUMINAMATH_GPT_average_calls_per_day_l1449_144988


namespace NUMINAMATH_GPT_inequality_l1449_144934

noncomputable def a : ℝ := (1 / 3) ^ (2 / 5)
noncomputable def b : ℝ := 2 ^ (4 / 3)
noncomputable def c : ℝ := Real.log 1 / 3 / Real.log 2

theorem inequality : c < a ∧ a < b := 
by 
  sorry

end NUMINAMATH_GPT_inequality_l1449_144934


namespace NUMINAMATH_GPT_identical_digits_satisfy_l1449_144975

theorem identical_digits_satisfy (n : ℕ) (hn : n ≥ 2) (x y z : ℕ) :
  (∃ (x y z : ℕ),
     (∃ (x y z : ℕ), 
         x = 3 ∧ y = 2 ∧ z = 1) ∨
     (∃ (x y z : ℕ), 
         x = 6 ∧ y = 8 ∧ z = 4) ∨
     (∃ (x y z : ℕ), 
         x = 8 ∧ y = 3 ∧ z = 7)) :=
by sorry

end NUMINAMATH_GPT_identical_digits_satisfy_l1449_144975


namespace NUMINAMATH_GPT_solve_for_n_l1449_144952

theorem solve_for_n (n : ℕ) : (3^n * 3^n * 3^n * 3^n = 81^2) → n = 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_n_l1449_144952


namespace NUMINAMATH_GPT_max_cookies_without_ingredients_l1449_144922

-- Defining the number of cookies and their composition
def total_cookies : ℕ := 36
def peanuts : ℕ := (2 * total_cookies) / 3
def chocolate_chips : ℕ := total_cookies / 3
def raisins : ℕ := total_cookies / 4
def oats : ℕ := total_cookies / 8

-- Proving the largest number of cookies without any ingredients
theorem max_cookies_without_ingredients : (total_cookies - (max (max peanuts chocolate_chips) raisins)) = 12 := by
    sorry

end NUMINAMATH_GPT_max_cookies_without_ingredients_l1449_144922


namespace NUMINAMATH_GPT_probability_A_wins_championship_distribution_and_expectation_B_l1449_144941

noncomputable def prob_event_1 : ℝ := 0.5
noncomputable def prob_event_2 : ℝ := 0.4
noncomputable def prob_event_3 : ℝ := 0.8

noncomputable def prob_A_wins_all : ℝ := prob_event_1 * prob_event_2 * prob_event_3
noncomputable def prob_A_wins_exactly_2 : ℝ :=
  prob_event_1 * prob_event_2 * (1 - prob_event_3) +
  prob_event_1 * (1 - prob_event_2) * prob_event_3 +
  (1 - prob_event_1) * prob_event_2 * prob_event_3

noncomputable def prob_A_wins_champ : ℝ := prob_A_wins_all + prob_A_wins_exactly_2

theorem probability_A_wins_championship : prob_A_wins_champ = 0.6 := by
  sorry

noncomputable def prob_B_wins_0 : ℝ := prob_A_wins_all
noncomputable def prob_B_wins_1 : ℝ := prob_event_1 * (1 - prob_event_2) * (1 - prob_event_3) +
                                        (1 - prob_event_1) * prob_event_2 * (1 - prob_event_3) +
                                        (1 - prob_event_1) * (1 - prob_event_2) * prob_event_3
noncomputable def prob_B_wins_2 : ℝ := (1 - prob_event_1) * prob_event_2 * prob_event_3 +
                                        prob_event_1 * (1 - prob_event_2) * prob_event_3 + 
                                        prob_event_1 * prob_event_2 * (1 - prob_event_3)
noncomputable def prob_B_wins_3 : ℝ := (1 - prob_event_1) * (1 - prob_event_2) * (1 - prob_event_3)

noncomputable def expected_score_B : ℝ :=
  0 * prob_B_wins_0 + 10 * prob_B_wins_1 +
  20 * prob_B_wins_2 + 30 * prob_B_wins_3

theorem distribution_and_expectation_B : 
  prob_B_wins_0 = 0.16 ∧
  prob_B_wins_1 = 0.44 ∧
  prob_B_wins_2 = 0.34 ∧
  prob_B_wins_3 = 0.06 ∧
  expected_score_B = 13 := by
  sorry

end NUMINAMATH_GPT_probability_A_wins_championship_distribution_and_expectation_B_l1449_144941


namespace NUMINAMATH_GPT_value_of_a_l1449_144987

theorem value_of_a (a : ℝ) (h : (2 : ℝ)^a = (1 / 2 : ℝ)) : a = -1 := 
sorry

end NUMINAMATH_GPT_value_of_a_l1449_144987


namespace NUMINAMATH_GPT_log_101600_l1449_144901

noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_101600 (h : log_base_10 102 = 0.3010) : log_base_10 101600 = 2.3010 :=
by
  sorry

end NUMINAMATH_GPT_log_101600_l1449_144901


namespace NUMINAMATH_GPT_outfits_without_matching_color_l1449_144995

theorem outfits_without_matching_color (red_shirts green_shirts pairs_pants green_hats red_hats : ℕ) 
  (h_red_shirts : red_shirts = 5) 
  (h_green_shirts : green_shirts = 5) 
  (h_pairs_pants : pairs_pants = 6) 
  (h_green_hats : green_hats = 8) 
  (h_red_hats : red_hats = 8) : 
  (red_shirts * pairs_pants * green_hats) + (green_shirts * pairs_pants * red_hats) = 480 := 
by 
  sorry

end NUMINAMATH_GPT_outfits_without_matching_color_l1449_144995


namespace NUMINAMATH_GPT_correctLikeTermsPair_l1449_144989

def areLikeTerms (term1 term2 : String) : Bool :=
  -- Define the criteria for like terms (variables and their respective powers)
  sorry

def pairA : (String × String) := ("-2x^3", "-2x")
def pairB : (String × String) := ("-1/2ab", "18ba")
def pairC : (String × String) := ("x^2y", "-xy^2")
def pairD : (String × String) := ("4m", "4mn")

theorem correctLikeTermsPair :
  areLikeTerms pairA.1 pairA.2 = false ∧
  areLikeTerms pairB.1 pairB.2 = true ∧
  areLikeTerms pairC.1 pairC.2 = false ∧
  areLikeTerms pairD.1 pairD.2 = false :=
sorry

end NUMINAMATH_GPT_correctLikeTermsPair_l1449_144989


namespace NUMINAMATH_GPT_area_of_farm_l1449_144905

theorem area_of_farm (W L : ℝ) (hW : W = 30) 
  (hL_fence_cost : 14 * (L + W + Real.sqrt (L^2 + W^2)) = 1680) : 
  W * L = 1200 :=
by
  sorry -- Proof not required

end NUMINAMATH_GPT_area_of_farm_l1449_144905


namespace NUMINAMATH_GPT_original_selling_price_l1449_144908

theorem original_selling_price (P : ℝ) (h1 : ∀ P, 1.17 * P = 1.10 * P + 42) :
    1.10 * P = 660 := by
  sorry

end NUMINAMATH_GPT_original_selling_price_l1449_144908


namespace NUMINAMATH_GPT_arithmetic_sequence_a3_l1449_144920

theorem arithmetic_sequence_a3 (a : ℕ → ℤ) (h1 : a 1 = 4) (h10 : a 10 = 22) (d : ℤ) (hd : ∀ n, a n = a 1 + (n - 1) * d) :
  a 3 = 8 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a3_l1449_144920


namespace NUMINAMATH_GPT_intersection_M_N_complement_N_U_l1449_144924

-- Definitions for the sets and the universal set
def U := Set ℝ
def M : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }
def N : Set ℝ := { x | ∃ y, y = Real.sqrt (1 - x) } -- Simplified domain interpretation for N

-- Intersection and complement calculations
theorem intersection_M_N (x : ℝ) : x ∈ M ∧ x ∈ N ↔ x ∈ { x | -2 ≤ x ∧ x ≤ 1 } := by sorry

theorem complement_N_U (x : ℝ) : x ∉ N ↔ x ∈ { x | x > 1 } := by sorry

end NUMINAMATH_GPT_intersection_M_N_complement_N_U_l1449_144924


namespace NUMINAMATH_GPT_patrons_per_golf_cart_l1449_144991

theorem patrons_per_golf_cart (patrons_from_cars patrons_from_bus golf_carts total_patrons patrons_per_cart : ℕ) 
  (h1 : patrons_from_cars = 12)
  (h2 : patrons_from_bus = 27)
  (h3 : golf_carts = 13)
  (h4 : total_patrons = patrons_from_cars + patrons_from_bus)
  (h5 : patrons_per_cart = total_patrons / golf_carts) : 
  patrons_per_cart = 3 := 
by
  sorry

end NUMINAMATH_GPT_patrons_per_golf_cart_l1449_144991


namespace NUMINAMATH_GPT_expand_expression_l1449_144903

open Nat

theorem expand_expression (x : ℝ) : (7 * x - 3) * (3 * x^2) = 21 * x^3 - 9 * x^2 :=
by
  sorry

end NUMINAMATH_GPT_expand_expression_l1449_144903


namespace NUMINAMATH_GPT_partition_subset_sum_l1449_144968

variable {p k : ℕ}

def V_p (p : ℕ) := {k : ℕ | p ∣ (k * (k + 1) / 2) ∧ k ≥ 2 * p - 1}

theorem partition_subset_sum (p : ℕ) (hp : Nat.Prime p) (k : ℕ) : k ∈ V_p p := sorry

end NUMINAMATH_GPT_partition_subset_sum_l1449_144968


namespace NUMINAMATH_GPT_algebraic_expression_evaluation_l1449_144909

theorem algebraic_expression_evaluation (x : ℝ) (h : x^2 + 3 * x - 5 = 2) : 2 * x^2 + 6 * x - 3 = 11 :=
sorry

end NUMINAMATH_GPT_algebraic_expression_evaluation_l1449_144909


namespace NUMINAMATH_GPT_jack_sugar_final_l1449_144954

-- Conditions
def initial_sugar := 65
def sugar_used := 18
def sugar_bought := 50

-- Question and proof goal
theorem jack_sugar_final : initial_sugar - sugar_used + sugar_bought = 97 := by
  sorry

end NUMINAMATH_GPT_jack_sugar_final_l1449_144954


namespace NUMINAMATH_GPT_supplement_of_complementary_angle_l1449_144973

theorem supplement_of_complementary_angle (α β : ℝ) 
  (h1 : α + β = 90) (h2 : α = 30) : 180 - β = 120 :=
by sorry

end NUMINAMATH_GPT_supplement_of_complementary_angle_l1449_144973


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1449_144964

def A : Set ℝ := {y | y > 1}
def B : Set ℝ := {x | Real.log x ≥ 0}
def Intersect : Set ℝ := {x | x > 1}

theorem intersection_of_A_and_B : A ∩ B = Intersect :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1449_144964


namespace NUMINAMATH_GPT_simplify_expression_l1449_144914

theorem simplify_expression (x : ℤ) : 120 * x - 55 * x = 65 * x := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1449_144914


namespace NUMINAMATH_GPT_sum_of_coefficients_is_225_l1449_144967

theorem sum_of_coefficients_is_225 :
  let C4 := 1
  let C41 := 4
  let C42 := 6
  let C43 := 4
  (C4 + C41 + C42 + C43)^2 = 225 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_is_225_l1449_144967


namespace NUMINAMATH_GPT_canoe_upstream_speed_l1449_144938

namespace canoe_speed

def V_c : ℝ := 12.5            -- speed of the canoe in still water in km/hr
def V_downstream : ℝ := 16     -- speed of the canoe downstream in km/hr

theorem canoe_upstream_speed :
  ∃ (V_upstream : ℝ), V_upstream = V_c - (V_downstream - V_c) ∧ V_upstream = 9 := by
  sorry

end canoe_speed

end NUMINAMATH_GPT_canoe_upstream_speed_l1449_144938


namespace NUMINAMATH_GPT_trebled_resultant_l1449_144984

theorem trebled_resultant (n : ℕ) (h : n = 20) : 3 * ((2 * n) + 5) = 135 := 
by
  sorry

end NUMINAMATH_GPT_trebled_resultant_l1449_144984


namespace NUMINAMATH_GPT_solve_percentage_increase_length_l1449_144969

def original_length (L : ℝ) : Prop := true
def original_breadth (B : ℝ) : Prop := true

def new_breadth (B' : ℝ) (B : ℝ) : Prop := B' = 1.25 * B

def new_length (L' : ℝ) (L : ℝ) (x : ℝ) : Prop := L' = L * (1 + x / 100)

def original_area (L : ℝ) (B : ℝ) (A : ℝ) : Prop := A = L * B

def new_area (A' : ℝ) (A : ℝ) : Prop := A' = 1.375 * A

def percentage_increase_length (x : ℝ) : Prop := x = 10

theorem solve_percentage_increase_length (L B A A' L' B' x : ℝ)
  (hL : original_length L)
  (hB : original_breadth B)
  (hB' : new_breadth B' B)
  (hL' : new_length L' L x)
  (hA : original_area L B A)
  (hA' : new_area A' A)
  (h_eqn : L' * B' = A') :
  percentage_increase_length x :=
by
  sorry

end NUMINAMATH_GPT_solve_percentage_increase_length_l1449_144969


namespace NUMINAMATH_GPT_scientific_notation_correct_l1449_144955

theorem scientific_notation_correct :
  52000000 = 5.2 * 10^7 :=
sorry

end NUMINAMATH_GPT_scientific_notation_correct_l1449_144955


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1449_144960

theorem solution_set_of_inequality :
  {x : ℝ | |x^2 - 2| < 2} = {x : ℝ | (x > -2 ∧ x < 0) ∨ (x > 0 ∧ x < 2)} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1449_144960


namespace NUMINAMATH_GPT_solve_equation_l1449_144956

theorem solve_equation : ∀ x : ℝ, (2 * x - 1)^2 - (1 - 3 * x)^2 = 5 * (1 - x) * (x + 1) → x = 5 / 2 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_equation_l1449_144956


namespace NUMINAMATH_GPT_new_average_weight_l1449_144928

def average_weight (A B C D E : ℝ) : Prop :=
  (A + B + C) / 3 = 70 ∧
  (A + B + C + D) / 4 = 70 ∧
  E = D + 3 ∧
  A = 81

theorem new_average_weight (A B C D E : ℝ) (h: average_weight A B C D E) : 
  (B + C + D + E) / 4 = 68 :=
by
  sorry

end NUMINAMATH_GPT_new_average_weight_l1449_144928


namespace NUMINAMATH_GPT_alice_meeting_distance_l1449_144994

noncomputable def distanceAliceWalks (t : ℝ) : ℝ :=
  6 * t

theorem alice_meeting_distance :
  ∃ t : ℝ, 
    distanceAliceWalks t = 
      (900 * Real.sqrt 2 - Real.sqrt 630000) / 11 ∧
    (5 * t) ^ 2 =
      (6 * t) ^ 2 + 150 ^ 2 - 2 * 6 * t * 150 * Real.cos (Real.pi / 4) :=
sorry

end NUMINAMATH_GPT_alice_meeting_distance_l1449_144994


namespace NUMINAMATH_GPT_solve_for_x_l1449_144999

theorem solve_for_x (x : ℝ) (h : 3^(3 * x - 2) = (1 : ℝ) / 27) : x = -(1 : ℝ) / 3 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1449_144999


namespace NUMINAMATH_GPT_johns_disposable_income_increase_l1449_144978

noncomputable def percentage_increase_of_johns_disposable_income
  (weekly_income_before : ℝ) (weekly_income_after : ℝ)
  (tax_rate_before : ℝ) (tax_rate_after : ℝ)
  (monthly_expense : ℝ) : ℝ :=
  let disposable_income_before := (weekly_income_before * (1 - tax_rate_before) * 4 - monthly_expense)
  let disposable_income_after := (weekly_income_after * (1 - tax_rate_after) * 4 - monthly_expense)
  (disposable_income_after - disposable_income_before) / disposable_income_before * 100

theorem johns_disposable_income_increase :
  percentage_increase_of_johns_disposable_income 60 70 0.15 0.18 100 = 24.62 :=
  by
  sorry

end NUMINAMATH_GPT_johns_disposable_income_increase_l1449_144978


namespace NUMINAMATH_GPT_teacher_works_days_in_month_l1449_144947

theorem teacher_works_days_in_month (P : ℕ) (W : ℕ) (M : ℕ) (T : ℕ) (H1 : P = 5) (H2 : W = 5) (H3 : M = 6) (H4 : T = 3600) : 
  (T / M) / (P * W) = 24 :=
by
  sorry

end NUMINAMATH_GPT_teacher_works_days_in_month_l1449_144947


namespace NUMINAMATH_GPT_part1_part2_l1449_144919

-- Define all given conditions
variable {A B C AC BC : ℝ}
variable (A_in_range : 0 < A ∧ A < π/2)
variable (B_in_range : 0 < B ∧ B < π/2)
variable (C_in_range : 0 < C ∧ C < π/2)
variable (m_perp_n : (Real.cos (A + π/3) * Real.cos B) + (Real.sin (A + π/3) * Real.sin B) = 0)
variable (cos_B : Real.cos B = 3/5)
variable (AC_value : AC = 8)

-- First part: Prove A - B = π/6
theorem part1 : A - B = π / 6 :=
by
  sorry

-- Second part: Prove BC = 4√3 + 3 given additional conditions
theorem part2 : BC = 4 * Real.sqrt 3 + 3 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1449_144919


namespace NUMINAMATH_GPT_smallest_y_l1449_144900

theorem smallest_y (y : ℕ) : (27^y > 3^24) ↔ (y ≥ 9) :=
sorry

end NUMINAMATH_GPT_smallest_y_l1449_144900


namespace NUMINAMATH_GPT_cut_square_into_rectangles_l1449_144937

theorem cut_square_into_rectangles :
  ∃ x y : ℕ, 3 * x + 4 * y = 25 :=
by
  -- Given that the total area is 25 and we are using rectangles of areas 3 and 4
  -- we need to verify the existence of integers x and y such that 3x + 4y = 25
  existsi 7
  existsi 1
  sorry

end NUMINAMATH_GPT_cut_square_into_rectangles_l1449_144937


namespace NUMINAMATH_GPT_conditional_probability_A_given_B_l1449_144926

noncomputable def P (A B : Prop) : ℝ := sorry -- Placeholder for the probability function

variables (A B : Prop)

axiom P_A_def : P A = 4/15
axiom P_B_def : P B = 2/15
axiom P_AB_def : P (A ∧ B) = 1/10

theorem conditional_probability_A_given_B : P (A ∧ B) / P B = 3/4 :=
by
  rw [P_AB_def, P_B_def]
  norm_num
  sorry

end NUMINAMATH_GPT_conditional_probability_A_given_B_l1449_144926


namespace NUMINAMATH_GPT_average_temperature_l1449_144948

def highTemps : List ℚ := [51, 60, 56, 55, 48, 63, 59]
def lowTemps : List ℚ := [42, 50, 44, 43, 41, 46, 45]

def dailyAverage (high low : ℚ) : ℚ :=
  (high + low) / 2

def averageOfAverages (tempsHigh tempsLow : List ℚ) : ℚ :=
  (List.sum (List.zipWith dailyAverage tempsHigh tempsLow)) / tempsHigh.length

theorem average_temperature :
  averageOfAverages highTemps lowTemps = 50.2 :=
  sorry

end NUMINAMATH_GPT_average_temperature_l1449_144948


namespace NUMINAMATH_GPT_rectangle_area_l1449_144906

theorem rectangle_area (x : ℝ) (w : ℝ) (h1 : (3 * w)^2 + w^2 = x^2) : (3 * w) * w = 3 * x^2 / 10 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1449_144906


namespace NUMINAMATH_GPT_workers_number_l1449_144910

theorem workers_number (W A : ℕ) (h1 : W * 25 = A) (h2 : (W + 10) * 15 = A) : W = 15 :=
by
  sorry

end NUMINAMATH_GPT_workers_number_l1449_144910


namespace NUMINAMATH_GPT_a_10_value_l1449_144930

-- Definitions for the initial conditions and recurrence relation.
def seq (a : ℕ → ℝ) : Prop :=
  a 0 = 0 ∧
  ∀ n, a (n + 1) = (8 / 5) * a n + (6 / 5) * (Real.sqrt (4 ^ n - a n ^ 2))

-- Statement that proves a_10 = 24576 / 25 given the conditions.
theorem a_10_value (a : ℕ → ℝ) (h : seq a) : a 10 = 24576 / 25 :=
by
  sorry

end NUMINAMATH_GPT_a_10_value_l1449_144930


namespace NUMINAMATH_GPT_roots_product_eq_three_l1449_144942

theorem roots_product_eq_three
  (p q r : ℝ)
  (h : (3:ℝ) * p ^ 3 - 8 * p ^ 2 + p - 9 = 0 ∧
       (3:ℝ) * q ^ 3 - 8 * q ^ 2 + q - 9 = 0 ∧
       (3:ℝ) * r ^ 3 - 8 * r ^ 2 + r - 9 = 0) :
  p * q * r = 3 :=
sorry

end NUMINAMATH_GPT_roots_product_eq_three_l1449_144942


namespace NUMINAMATH_GPT_smallest_n_square_partition_l1449_144925

theorem smallest_n_square_partition (n : ℕ) (h : ∃ a b : ℕ, a ≥ 1 ∧ b ≥ 1 ∧ n = 40 * a + 49 * b) : n ≥ 2000 :=
by sorry

end NUMINAMATH_GPT_smallest_n_square_partition_l1449_144925


namespace NUMINAMATH_GPT_kendra_minivans_l1449_144953

theorem kendra_minivans (afternoon: ℕ) (evening: ℕ) (h1: afternoon = 4) (h2: evening = 1) : afternoon + evening = 5 :=
by sorry

end NUMINAMATH_GPT_kendra_minivans_l1449_144953


namespace NUMINAMATH_GPT_three_digit_numbers_divisible_by_5_l1449_144981

theorem three_digit_numbers_divisible_by_5 : 
  let first_term := 100
  let last_term := 995
  let common_difference := 5 
  (last_term - first_term) / common_difference + 1 = 180 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_numbers_divisible_by_5_l1449_144981


namespace NUMINAMATH_GPT_tan_x_eq_sqrt3_intervals_of_monotonic_increase_l1449_144935

noncomputable def m (x : ℝ) : ℝ × ℝ :=
  (Real.sin (x - Real.pi / 6), 1)

noncomputable def n (x : ℝ) : ℝ × ℝ :=
  (Real.cos x, 1)

noncomputable def f (x : ℝ) : ℝ :=
  (m x).1 * (n x).1 + (m x).2 * (n x).2

-- Proof for part 1
theorem tan_x_eq_sqrt3 (x : ℝ) (h₀ : m x = n x) : Real.tan x = Real.sqrt 3 :=
sorry

-- Proof for part 2
theorem intervals_of_monotonic_increase (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ Real.pi) :
  (0 ≤ x ∧ x ≤ Real.pi / 3) ∨ (5 * Real.pi / 6 ≤ x ∧ x ≤ Real.pi) ↔ 
  (0 ≤ x ∧ x ≤ Real.pi / 3) ∨ (5 * Real.pi / 6 ≤ x ∧ x ≤ Real.pi) :=
sorry

end NUMINAMATH_GPT_tan_x_eq_sqrt3_intervals_of_monotonic_increase_l1449_144935


namespace NUMINAMATH_GPT_sixth_equation_l1449_144958

theorem sixth_equation :
  (6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 + 14 + 15 + 16 = 121) :=
by
  sorry

end NUMINAMATH_GPT_sixth_equation_l1449_144958


namespace NUMINAMATH_GPT_min_sum_of_a_and_b_l1449_144993

theorem min_sum_of_a_and_b (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > 4 * b) : a + b ≥ 6 :=
by
  sorry

end NUMINAMATH_GPT_min_sum_of_a_and_b_l1449_144993


namespace NUMINAMATH_GPT_extreme_point_l1449_144986

noncomputable def f (x : ℝ) : ℝ := (x^4 / 4) - (x^3 / 3)
noncomputable def f_prime (x : ℝ) : ℝ := deriv f x

theorem extreme_point (x : ℝ) : f_prime 1 = 0 ∧
  (∀ y, y < 1 → f_prime y < 0) ∧
  (∀ z, z > 1 → f_prime z > 0) :=
by
  sorry

end NUMINAMATH_GPT_extreme_point_l1449_144986


namespace NUMINAMATH_GPT_sticker_price_l1449_144911

theorem sticker_price (x : ℝ) (h : 0.85 * x - 90 = 0.75 * x - 15) : x = 750 := 
sorry

end NUMINAMATH_GPT_sticker_price_l1449_144911


namespace NUMINAMATH_GPT_solve_686_l1449_144997

theorem solve_686 : ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x^2 + y^2 + z^2 = 686 := 
by
  sorry

end NUMINAMATH_GPT_solve_686_l1449_144997


namespace NUMINAMATH_GPT_Mikaela_savings_l1449_144966

theorem Mikaela_savings
  (hourly_rate : ℕ)
  (first_month_hours : ℕ)
  (additional_hours_second_month : ℕ)
  (spending_fraction : ℚ)
  (earnings_first_month := hourly_rate * first_month_hours)
  (hours_second_month := first_month_hours + additional_hours_second_month)
  (earnings_second_month := hourly_rate * hours_second_month)
  (total_earnings := earnings_first_month + earnings_second_month)
  (amount_spent := spending_fraction * total_earnings)
  (amount_saved := total_earnings - amount_spent) :
  hourly_rate = 10 →
  first_month_hours = 35 →
  additional_hours_second_month = 5 →
  spending_fraction = 4 / 5 →
  amount_saved = 150 :=
by
  sorry

end NUMINAMATH_GPT_Mikaela_savings_l1449_144966


namespace NUMINAMATH_GPT_midpoint_of_interception_l1449_144974

theorem midpoint_of_interception (x1 x2 y1 y2 : ℝ) 
  (h1 : y1^2 = 4 * x1) 
  (h2 : y2^2 = 4 * x2) 
  (h3 : y1 = x1 - 1) 
  (h4 : y2 = x2 - 1) : 
  ( (x1 + x2) / 2, (y1 + y2) / 2 ) = (3, 2) :=
by 
  sorry

end NUMINAMATH_GPT_midpoint_of_interception_l1449_144974


namespace NUMINAMATH_GPT_exposed_sides_correct_l1449_144904

-- Define the number of sides of each polygon
def sides_triangle := 3
def sides_square := 4
def sides_pentagon := 5
def sides_hexagon := 6
def sides_heptagon := 7

-- Total sides from all polygons
def total_sides := sides_triangle + sides_square + sides_pentagon + sides_hexagon + sides_heptagon

-- Number of shared sides
def shared_sides := 4

-- Final number of exposed sides
def exposed_sides := total_sides - shared_sides

-- Statement to prove
theorem exposed_sides_correct : exposed_sides = 21 :=
by {
  -- This part will contain the proof which we do not need. Replace with 'sorry' for now.
  sorry
}

end NUMINAMATH_GPT_exposed_sides_correct_l1449_144904


namespace NUMINAMATH_GPT_intersection_nonempty_iff_m_lt_one_l1449_144977

open Set Real

variable {m : ℝ}

theorem intersection_nonempty_iff_m_lt_one 
  (A : Set ℝ) (B : Set ℝ) (U : Set ℝ := univ) 
  (hA : A = {x | x + m >= 0}) 
  (hB : B = {x | -1 < x ∧ x < 5}) : 
  (U \ A ∩ B ≠ ∅) ↔ m < 1 := by
  sorry

end NUMINAMATH_GPT_intersection_nonempty_iff_m_lt_one_l1449_144977


namespace NUMINAMATH_GPT_intersection_product_of_circles_l1449_144983

theorem intersection_product_of_circles :
  (∀ x y : ℝ, (x^2 + 2 * x + y^2 + 4 * y + 5 = 0) ∧ (x^2 + 6 * x + y^2 + 4 * y + 9 = 0) →
  x * y = 2) :=
sorry

end NUMINAMATH_GPT_intersection_product_of_circles_l1449_144983


namespace NUMINAMATH_GPT_polynomial_sum_l1449_144929

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end NUMINAMATH_GPT_polynomial_sum_l1449_144929


namespace NUMINAMATH_GPT_apple_cost_is_2_l1449_144939

def total_spent (hummus_cost chicken_cost bacon_cost vegetable_cost : ℕ) : ℕ :=
  2 * hummus_cost + chicken_cost + bacon_cost + vegetable_cost

theorem apple_cost_is_2 :
  ∀ (hummus_cost chicken_cost bacon_cost vegetable_cost total_money apples_cost : ℕ),
    hummus_cost = 5 →
    chicken_cost = 20 →
    bacon_cost = 10 →
    vegetable_cost = 10 →
    total_money = 60 →
    apples_cost = 5 →
    (total_money - total_spent hummus_cost chicken_cost bacon_cost vegetable_cost) / apples_cost = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_apple_cost_is_2_l1449_144939


namespace NUMINAMATH_GPT_sum_of_remainders_l1449_144916

theorem sum_of_remainders (n : ℤ) (h : n % 18 = 11) : (n % 2 + n % 9) = 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_remainders_l1449_144916


namespace NUMINAMATH_GPT_proof_problem_l1449_144921

variable (m : ℝ)

theorem proof_problem 
  (h1 : ∀ x, (m / (x - 2) + 1 = x / (2 - x)) → x ≠ 2 ∧ x ≥ 0) :
  m ≤ 2 ∧ m ≠ -2 := by
  sorry

end NUMINAMATH_GPT_proof_problem_l1449_144921


namespace NUMINAMATH_GPT_mabel_petals_remaining_l1449_144931

/-- Mabel has 5 daisies, each with 8 petals. If she gives 2 daisies to her teacher,
how many petals does she have on the remaining daisies in her garden? -/
theorem mabel_petals_remaining :
  (5 - 2) * 8 = 24 :=
by
  sorry

end NUMINAMATH_GPT_mabel_petals_remaining_l1449_144931


namespace NUMINAMATH_GPT_west_for_200_is_neg_200_l1449_144918

-- Given a definition for driving east
def driving_east (d : Int) : Int := d

-- Driving east for 80 km is +80 km
def driving_east_80 : Int := driving_east 80

-- Driving west should be the negative of driving east
def driving_west (d : Int) : Int := -d

-- Driving west for 200 km is -200 km
def driving_west_200 : Int := driving_west 200

-- Theorem to prove the given condition and expected result
theorem west_for_200_is_neg_200 : driving_west_200 = -200 :=
by
  -- Proof step is skipped
  sorry

end NUMINAMATH_GPT_west_for_200_is_neg_200_l1449_144918


namespace NUMINAMATH_GPT_orange_balls_count_l1449_144976

theorem orange_balls_count :
  ∀ (total red blue orange pink : ℕ), 
  total = 50 → red = 20 → blue = 10 → 
  total = red + blue + orange + pink → 3 * orange = pink → 
  orange = 5 :=
by
  intros total red blue orange pink h_total h_red h_blue h_total_eq h_ratio
  sorry

end NUMINAMATH_GPT_orange_balls_count_l1449_144976


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1449_144980

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (d : ℤ) :
  (∀ n, a n = a 1 + (n - 1) * d) → 
  (∀ n, S n = n * (a 1 + a n) / 2) → 
  (a 3 + 4 = a 2 + a 7) → 
  S 11 = 44 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1449_144980


namespace NUMINAMATH_GPT_find_n_in_permutation_combination_equation_l1449_144932

-- Lean statement for the proof problem

theorem find_n_in_permutation_combination_equation :
  ∃ (n : ℕ), (n > 0) ∧ (Nat.factorial 8 / Nat.factorial (8 - n) = 2 * (Nat.factorial 8 / (Nat.factorial 2 * Nat.factorial 6)))
  := sorry

end NUMINAMATH_GPT_find_n_in_permutation_combination_equation_l1449_144932


namespace NUMINAMATH_GPT_tan_alpha_value_l1449_144970

open Real

theorem tan_alpha_value
  (α : ℝ)
  (h₀ : 0 < α)
  (h₁ : α < π / 2)
  (h₂ : cos (2 * α) = (2 * sqrt 5 / 5) * sin (α + π / 4)) :
  tan α = 1 / 3 :=
sorry

end NUMINAMATH_GPT_tan_alpha_value_l1449_144970


namespace NUMINAMATH_GPT_find_y_l1449_144992

theorem find_y (t : ℚ) (x y : ℚ) (h1 : x = 3 - 2 * t) (h2 : y = 3 * t + 10) (hx : x = -4) : y = 41 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1449_144992


namespace NUMINAMATH_GPT_rectangle_area_proof_l1449_144965

variable (x y : ℕ) -- Declaring the variables to represent length and width of the rectangle.

-- Declaring the conditions as hypotheses.
def condition1 := (x + 3) * (y - 1) = x * y
def condition2 := (x - 3) * (y + 2) = x * y
def condition3 := (x + 4) * (y - 2) = x * y

-- The theorem to prove the area is 36 given the above conditions.
theorem rectangle_area_proof (h1 : condition1 x y) (h2 : condition2 x y) (h3 : condition3 x y) : x * y = 36 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_proof_l1449_144965
