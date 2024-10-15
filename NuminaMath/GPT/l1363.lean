import Mathlib

namespace NUMINAMATH_GPT_rebecca_income_percentage_l1363_136337

-- Define Rebecca's initial income
def rebecca_initial_income : ℤ := 15000
-- Define Jimmy's income
def jimmy_income : ℤ := 18000
-- Define the increase in Rebecca's income
def rebecca_income_increase : ℤ := 7000

-- Define the new income for Rebecca after increase
def rebecca_new_income : ℤ := rebecca_initial_income + rebecca_income_increase
-- Define the new combined income
def new_combined_income : ℤ := rebecca_new_income + jimmy_income

-- State the theorem to prove that Rebecca's new income is 55% of the new combined income
theorem rebecca_income_percentage : 
  (rebecca_new_income * 100) / new_combined_income = 55 :=
sorry

end NUMINAMATH_GPT_rebecca_income_percentage_l1363_136337


namespace NUMINAMATH_GPT_mall_incur_1_percent_loss_l1363_136339

theorem mall_incur_1_percent_loss
  (a b x : ℝ)
  (ha : x = a * 1.1)
  (hb : x = b * 0.9) :
  (2 * x - (a + b)) / (a + b) = -0.01 :=
sorry

end NUMINAMATH_GPT_mall_incur_1_percent_loss_l1363_136339


namespace NUMINAMATH_GPT_sum_of_first_three_terms_l1363_136316

theorem sum_of_first_three_terms (a : ℕ → ℤ) 
  (h4 : a 4 = 8) 
  (h5 : a 5 = 12) 
  (h6 : a 6 = 16) : 
  a 1 + a 2 + a 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_three_terms_l1363_136316


namespace NUMINAMATH_GPT_jose_julia_completion_time_l1363_136355

variable (J N L : ℝ)

theorem jose_julia_completion_time :
  J + N + L = 1/4 ∧
  J * (1/3) = 1/18 ∧
  N = 1/9 ∧
  L * (1/3) = 1/18 →
  1/J = 6 ∧ 1/L = 6 := sorry

end NUMINAMATH_GPT_jose_julia_completion_time_l1363_136355


namespace NUMINAMATH_GPT_haley_marble_distribution_l1363_136342

theorem haley_marble_distribution (total_marbles : ℕ) (num_boys : ℕ) (h1 : total_marbles = 20) (h2 : num_boys = 2) : (total_marbles / num_boys) = 10 := 
by 
  sorry

end NUMINAMATH_GPT_haley_marble_distribution_l1363_136342


namespace NUMINAMATH_GPT_work_b_alone_l1363_136311

theorem work_b_alone (a b : ℕ) (h1 : 2 * b = a) (h2 : a + b = 3) (h3 : (a + b) * 11 = 33) : 33 = 33 :=
by
  -- sorry is used here because we are skipping the actual proof
  sorry

end NUMINAMATH_GPT_work_b_alone_l1363_136311


namespace NUMINAMATH_GPT_patio_perimeter_is_100_feet_l1363_136323

theorem patio_perimeter_is_100_feet
  (rectangle : Prop)
  (length : ℝ)
  (width : ℝ)
  (length_eq_40 : length = 40)
  (length_eq_4_times_width : length = 4 * width) :
  2 * length + 2 * width = 100 := 
by
  sorry

end NUMINAMATH_GPT_patio_perimeter_is_100_feet_l1363_136323


namespace NUMINAMATH_GPT_simplify_expression_l1363_136348

-- Define the variables a and b as real numbers
variables (a b : ℝ)

-- Define the expression and the simplified expression
def original_expr := -a^2 * (-2 * a * b) + 3 * a * (a^2 * b - 1)
def simplified_expr := 5 * a^3 * b - 3 * a

-- Statement that the original expression is equal to the simplified expression
theorem simplify_expression : original_expr a b = simplified_expr a b :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1363_136348


namespace NUMINAMATH_GPT_range_of_t_l1363_136392

noncomputable def f : ℝ → ℝ := sorry

axiom f_symmetric (x : ℝ) : f (x - 3) = f (-x - 3)
axiom f_ln_definition (x : ℝ) (h : x ≤ -3) : f x = Real.log (-x)

theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, f (Real.sin x - t) > f (3 * Real.sin x - 1)) ↔ (t < -1 ∨ t > 9) := sorry

end NUMINAMATH_GPT_range_of_t_l1363_136392


namespace NUMINAMATH_GPT_pythagorean_numbers_b_l1363_136330

-- Define Pythagorean numbers and conditions
variable (a b c m : ℕ)
variable (h1 : a = 1/2 * m^2 - 1/2)
variable (h2 : c = 1/2 * m^2 + 1/2)
variable (h3 : m > 1 ∧ ¬ even m)

theorem pythagorean_numbers_b (h4 : c^2 = a^2 + b^2) : b = m :=
sorry

end NUMINAMATH_GPT_pythagorean_numbers_b_l1363_136330


namespace NUMINAMATH_GPT_min_value_of_expression_l1363_136320

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 4/y = 1) : 
  x + 2 * y ≥ 9 + 4 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_min_value_of_expression_l1363_136320


namespace NUMINAMATH_GPT_add_fractions_l1363_136369

theorem add_fractions : (1 : ℚ) / 4 + (3 : ℚ) / 8 = 5 / 8 :=
by
  sorry

end NUMINAMATH_GPT_add_fractions_l1363_136369


namespace NUMINAMATH_GPT_isosceles_trapezoid_AB_length_l1363_136354

theorem isosceles_trapezoid_AB_length (BC AD : ℝ) (r : ℝ) (a : ℝ) (h_isosceles : BC = a) (h_ratio : AD = 3 * a) (h_area : 4 * a * r = Real.sqrt 3 / 2) (h_radius : r = a * Real.sqrt 3 / 2) :
  2 * a = 1 :=
by
 sorry

end NUMINAMATH_GPT_isosceles_trapezoid_AB_length_l1363_136354


namespace NUMINAMATH_GPT_coffee_maker_capacity_l1363_136379

theorem coffee_maker_capacity (x : ℝ) (h : 0.36 * x = 45) : x = 125 :=
sorry

end NUMINAMATH_GPT_coffee_maker_capacity_l1363_136379


namespace NUMINAMATH_GPT_reyn_pieces_l1363_136318

-- Define the conditions
variables (total_pieces : ℕ) (pieces_each : ℕ) (pieces_left : ℕ)
variables (R : ℕ) (Rhys : ℕ) (Rory : ℕ)

-- Initial Conditions
def mrs_young_conditions :=
  total_pieces = 300 ∧
  pieces_each = total_pieces / 3 ∧
  Rhys = 2 * R ∧
  Rory = 3 * R ∧
  6 * R + pieces_left = total_pieces ∧
  pieces_left = 150

-- The statement of our proof goal
theorem reyn_pieces (h : mrs_young_conditions total_pieces pieces_each pieces_left R Rhys Rory) : R = 25 :=
sorry

end NUMINAMATH_GPT_reyn_pieces_l1363_136318


namespace NUMINAMATH_GPT_largest_integer_same_cost_l1363_136358

def cost_base_10 (n : ℕ) : ℕ :=
  (n.digits 10).sum

def cost_base_2 (n : ℕ) : ℕ :=
  (n.digits 2).sum

theorem largest_integer_same_cost : ∃ n < 1000, 
  cost_base_10 n = cost_base_2 n ∧
  ∀ m < 1000, cost_base_10 m = cost_base_2 m → n ≥ m :=
sorry

end NUMINAMATH_GPT_largest_integer_same_cost_l1363_136358


namespace NUMINAMATH_GPT_binary_to_decimal_l1363_136378

theorem binary_to_decimal : 
  (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 1 * 2^3 + 1 * 2^4) = 27 :=
by
  sorry

end NUMINAMATH_GPT_binary_to_decimal_l1363_136378


namespace NUMINAMATH_GPT_transformed_inequality_l1363_136376

theorem transformed_inequality (x : ℝ) : 
  (x - 3) / 3 < (2 * x + 1) / 2 - 1 ↔ 2 * (x - 3) < 3 * (2 * x + 1) - 6 :=
by
  sorry

end NUMINAMATH_GPT_transformed_inequality_l1363_136376


namespace NUMINAMATH_GPT_train_crossing_time_l1363_136335

noncomputable def length_first_train : ℝ := 200  -- meters
noncomputable def speed_first_train_kmph : ℝ := 72  -- km/h
noncomputable def speed_first_train : ℝ := speed_first_train_kmph * (1000 / 3600)  -- m/s

noncomputable def length_second_train : ℝ := 300  -- meters
noncomputable def speed_second_train_kmph : ℝ := 36  -- km/h
noncomputable def speed_second_train : ℝ := speed_second_train_kmph * (1000 / 3600)  -- m/s

noncomputable def relative_speed : ℝ := speed_first_train - speed_second_train -- m/s
noncomputable def total_length : ℝ := length_first_train + length_second_train  -- meters
noncomputable def time_to_cross : ℝ := total_length / relative_speed  -- seconds

theorem train_crossing_time :
  time_to_cross = 50 := by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l1363_136335


namespace NUMINAMATH_GPT_certain_number_l1363_136382

theorem certain_number (a b : ℕ) (n : ℕ) 
  (h1: a % n = 0) (h2: b % n = 0) 
  (h3: b = a + 9 * n)
  (h4: b = a + 126) : n = 14 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_l1363_136382


namespace NUMINAMATH_GPT_probability_blue_or_purple_l1363_136366

def total_jelly_beans : ℕ := 35
def blue_jelly_beans : ℕ := 7
def purple_jelly_beans : ℕ := 10

theorem probability_blue_or_purple : (blue_jelly_beans + purple_jelly_beans: ℚ) / total_jelly_beans = 17 / 35 := 
by sorry

end NUMINAMATH_GPT_probability_blue_or_purple_l1363_136366


namespace NUMINAMATH_GPT_common_chord_eqn_l1363_136300

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 12 * x - 2 * y - 13 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 + 12 * x + 16 * y - 25 = 0

-- Define the proposition stating the common chord equation
theorem common_chord_eqn : ∀ x y : ℝ, C1 x y ∧ C2 x y → 4 * x + 3 * y - 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_common_chord_eqn_l1363_136300


namespace NUMINAMATH_GPT_largest_n_is_253_l1363_136388

-- Define the triangle property for a set
def triangle_property (s : Set ℕ) : Prop :=
∀ (a b c : ℕ), a ∈ s → b ∈ s → c ∈ s → a < b → b < c → c < a + b

-- Define the problem statement
def largest_possible_n (n : ℕ) : Prop :=
∀ (s : Finset ℕ), (∀ (x : ℕ), x ∈ s → 4 ≤ x ∧ x ≤ n) → (s.card = 10 → triangle_property s)

-- The given proof problem
theorem largest_n_is_253 : largest_possible_n 253 :=
by
  sorry

end NUMINAMATH_GPT_largest_n_is_253_l1363_136388


namespace NUMINAMATH_GPT_range_of_a_l1363_136394

noncomputable def f (x a : ℝ) : ℝ := 
  x * (a - 1 / Real.exp x)

noncomputable def gx (x : ℝ) : ℝ :=
  (1 + x) / Real.exp x

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 a = 0 ∧ f x2 a = 0) →
  a < 2 / Real.exp 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1363_136394


namespace NUMINAMATH_GPT_g_at_neg2_eq_8_l1363_136307

-- Define the functions f and g
def f (x : ℤ) : ℤ := 4 * x - 6
def g (y : ℤ) : ℤ := 3 * (y + 6/4)^2 + 4 * (y + 6/4) + 1

-- Statement of the math proof problem:
theorem g_at_neg2_eq_8 : g (-2) = 8 := 
by 
  sorry

end NUMINAMATH_GPT_g_at_neg2_eq_8_l1363_136307


namespace NUMINAMATH_GPT_relationship_of_variables_l1363_136344

theorem relationship_of_variables
  (a b c d : ℚ)
  (h : (a + b) / (b + c) = (c + d) / (d + a)) :
  a = c ∨ a + b + c + d = 0 :=
by sorry

end NUMINAMATH_GPT_relationship_of_variables_l1363_136344


namespace NUMINAMATH_GPT_find_a_value_l1363_136364

namespace Proof

-- Define the context and variables
variables (a b c : ℝ)
variables (h1 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1))
variables (h2 : a * 15 * 2 = 4)

-- State the theorem we want to prove
theorem find_a_value: a = 6 :=
by
  sorry

end Proof

end NUMINAMATH_GPT_find_a_value_l1363_136364


namespace NUMINAMATH_GPT_inequality_holds_for_all_real_numbers_l1363_136370

theorem inequality_holds_for_all_real_numbers (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + k * x - 3 / 4 < 0) ↔ (k ∈ Set.Icc (-3 : ℝ) 0) := 
sorry

end NUMINAMATH_GPT_inequality_holds_for_all_real_numbers_l1363_136370


namespace NUMINAMATH_GPT_largest_number_of_cakes_without_ingredients_l1363_136325

theorem largest_number_of_cakes_without_ingredients :
  let total_cakes := 60
  let cakes_with_strawberries := (1 / 3) * total_cakes
  let cakes_with_blueberries := (1 / 2) * total_cakes
  let cakes_with_raspberries := (3 / 5) * total_cakes
  let cakes_with_coconut := (1 / 10) * total_cakes
  ∃ (max_no_ingredients : ℕ), max_no_ingredients = 24 :=
by
  let total_cakes := 60
  let cakes_with_strawberries := (1 / 3) * total_cakes
  let cakes_with_blueberries := (1 / 2) * total_cakes
  let cakes_with_raspberries := (3 / 5) * total_cakes
  let cakes_with_coconut := (1 / 10) * total_cakes
  existsi (60 - max 20 (max 30 (max 36 6))) -- max value should be used to reflect maximum coverage content
  sorry -- Proof to be completed

end NUMINAMATH_GPT_largest_number_of_cakes_without_ingredients_l1363_136325


namespace NUMINAMATH_GPT_same_terminal_side_l1363_136375

open Real

theorem same_terminal_side (k : ℤ) : (∃ k : ℤ, k * 360 - 315 = 9 / 4 * 180) :=
by
  sorry

end NUMINAMATH_GPT_same_terminal_side_l1363_136375


namespace NUMINAMATH_GPT_original_average_age_older_l1363_136365

theorem original_average_age_older : 
  ∀ (n : ℕ) (T : ℕ), (T = n * 40) →
  (T + 408) / (n + 12) = 36 →
  40 - 36 = 4 :=
by
  intros n T hT hNewAvg
  sorry

end NUMINAMATH_GPT_original_average_age_older_l1363_136365


namespace NUMINAMATH_GPT_total_pizzas_served_l1363_136324

-- Define the conditions
def pizzas_lunch : Nat := 9
def pizzas_dinner : Nat := 6

-- Define the theorem to prove
theorem total_pizzas_served : pizzas_lunch + pizzas_dinner = 15 := by
  sorry

end NUMINAMATH_GPT_total_pizzas_served_l1363_136324


namespace NUMINAMATH_GPT_find_y_value_l1363_136360

noncomputable def y_value (y : ℝ) : Prop :=
  let side1_sq_area := 9 * y^2
  let side2_sq_area := 36 * y^2
  let triangle_area := 9 * y^2
  (side1_sq_area + side2_sq_area + triangle_area = 1000)

theorem find_y_value (y : ℝ) : y_value y → y = 10 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_GPT_find_y_value_l1363_136360


namespace NUMINAMATH_GPT_cost_to_fill_pool_l1363_136381

-- Definitions based on conditions

def hours_to_fill_pool : ℕ := 50
def hose_rate : ℕ := 100  -- hose runs at 100 gallons per hour
def water_cost_per_10_gallons : ℕ := 1 -- cost is 1 cent for 10 gallons
def cents_to_dollars (cents : ℕ) : ℕ := cents / 100 -- Conversion from cents to dollars

-- Prove the cost to fill the pool is 5 dollars
theorem cost_to_fill_pool : 
  (hours_to_fill_pool * hose_rate / 10 * water_cost_per_10_gallons) / 100 = 5 :=
by sorry

end NUMINAMATH_GPT_cost_to_fill_pool_l1363_136381


namespace NUMINAMATH_GPT_find_x_for_fraction_equality_l1363_136390

theorem find_x_for_fraction_equality (x : ℝ) : 
  (4 + 2 * x) / (7 + x) = (2 + x) / (3 + x) ↔ (x = -2 ∨ x = 1) := by
  sorry

end NUMINAMATH_GPT_find_x_for_fraction_equality_l1363_136390


namespace NUMINAMATH_GPT_simplify_evaluate_l1363_136310

theorem simplify_evaluate (x y : ℝ) (h : (x - 2)^2 + |1 + y| = 0) : 
  ((x - y) * (x + 2 * y) - (x + y)^2) / y = 1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_evaluate_l1363_136310


namespace NUMINAMATH_GPT_sam_morning_run_distance_l1363_136380

variable (x : ℝ) -- The distance of Sam's morning run in miles

theorem sam_morning_run_distance (h1 : ∀ y, y = 2 * x) (h2 : 12 = 12) (h3 : x + 2 * x + 12 = 18) : x = 2 :=
by sorry

end NUMINAMATH_GPT_sam_morning_run_distance_l1363_136380


namespace NUMINAMATH_GPT_bd_le_q2_l1363_136333

theorem bd_le_q2 (a b c d p q : ℝ) (h1 : a * b + c * d = 2 * p * q) (h2 : a * c ≥ p^2 ∧ p^2 > 0) : b * d ≤ q^2 :=
sorry

end NUMINAMATH_GPT_bd_le_q2_l1363_136333


namespace NUMINAMATH_GPT_simplify_expression_l1363_136372

variable (x : ℝ) (hx : x ≠ 0)

theorem simplify_expression : 
  ( (x + 3)^2 + (x + 3) * (x - 3) ) / (2 * x) = x + 3 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1363_136372


namespace NUMINAMATH_GPT_kendra_fish_count_l1363_136308

variable (K : ℕ) -- Number of fish Kendra caught
variable (Ken_fish : ℕ) -- Number of fish Ken brought home

-- Conditions
axiom twice_as_many : Ken_fish = 2 * K - 3
axiom total_fish : K + Ken_fish = 87

-- The theorem we need to prove
theorem kendra_fish_count : K = 30 :=
by
  -- Lean proof goes here
  sorry

end NUMINAMATH_GPT_kendra_fish_count_l1363_136308


namespace NUMINAMATH_GPT_turtle_reaches_waterhole_28_minutes_after_meeting_l1363_136351

theorem turtle_reaches_waterhole_28_minutes_after_meeting (x : ℝ) (distance_lion1 : ℝ := 5 * x) 
  (speed_lion2 : ℝ := 1.5 * x) (distance_turtle : ℝ := 30) (speed_turtle : ℝ := 1/30) : 
  ∃ t_meeting : ℝ, t_meeting = 2 ∧ (distance_turtle - speed_turtle * t_meeting) / speed_turtle = 28 :=
by 
  sorry

end NUMINAMATH_GPT_turtle_reaches_waterhole_28_minutes_after_meeting_l1363_136351


namespace NUMINAMATH_GPT_factors_of_180_multiple_of_15_count_l1363_136352

theorem factors_of_180_multiple_of_15_count :
  ∃ n : Nat, n = 6 ∧ ∀ x : Nat, x ∣ 180 → 15 ∣ x → x > 0 :=
by
  sorry

end NUMINAMATH_GPT_factors_of_180_multiple_of_15_count_l1363_136352


namespace NUMINAMATH_GPT_eggs_per_basket_l1363_136306

theorem eggs_per_basket (n : ℕ) (total_eggs_red total_eggs_orange min_eggs_per_basket : ℕ) (h_red : total_eggs_red = 20) (h_orange : total_eggs_orange = 30) (h_min : min_eggs_per_basket = 5) (h_div_red : total_eggs_red % n = 0) (h_div_orange : total_eggs_orange % n = 0) (h_at_least : n ≥ min_eggs_per_basket) : n = 5 :=
sorry

end NUMINAMATH_GPT_eggs_per_basket_l1363_136306


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1363_136346

theorem arithmetic_sequence_sum (a : ℕ → ℚ) (S_9 : ℚ) 
  (h_arith : ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_a2_a8 : a 2 + a 8 = 4 / 3) :
  S_9 = 6 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1363_136346


namespace NUMINAMATH_GPT_brady_work_hours_l1363_136331

theorem brady_work_hours (A : ℕ) :
    (A * 30 + 5 * 30 + 8 * 30 = 3 * 190) → 
    A = 6 :=
by sorry

end NUMINAMATH_GPT_brady_work_hours_l1363_136331


namespace NUMINAMATH_GPT_pure_alcohol_addition_problem_l1363_136303

-- Define the initial conditions
def initial_volume := 6
def initial_concentration := 0.30
def final_concentration := 0.50

-- Define the amount of pure alcohol to be added
def x := 2.4

-- Proof problem statement
theorem pure_alcohol_addition_problem (initial_volume initial_concentration final_concentration x : ℝ) :
  initial_volume * initial_concentration + x = final_concentration * (initial_volume + x) :=
by
  -- Initial condition values definition
  let initial_volume := 6
  let initial_concentration := 0.30
  let final_concentration := 0.50
  let x := 2.4
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_pure_alcohol_addition_problem_l1363_136303


namespace NUMINAMATH_GPT_minimize_quadratic_l1363_136393

theorem minimize_quadratic : ∃ x : ℝ, x = -4 ∧ ∀ y : ℝ, x^2 + 8*x + 7 ≤ y^2 + 8*y + 7 :=
by 
  use -4
  sorry

end NUMINAMATH_GPT_minimize_quadratic_l1363_136393


namespace NUMINAMATH_GPT_unknown_number_is_10_l1363_136384

def operation_e (x y : ℕ) : ℕ := 2 * x * y

theorem unknown_number_is_10 (n : ℕ) (h : operation_e 8 (operation_e n 5) = 640) : n = 10 :=
by
  sorry

end NUMINAMATH_GPT_unknown_number_is_10_l1363_136384


namespace NUMINAMATH_GPT_refrigerator_profit_l1363_136353

theorem refrigerator_profit 
  (marked_price : ℝ) 
  (cost_price : ℝ) 
  (profit_margin : ℝ ) 
  (discount1 : ℝ) 
  (profit1 : ℝ)
  (discount2 : ℝ):
  profit_margin = 0.1 → 
  profit1 = 200 → 
  cost_price = 2000 → 
  discount1 = 0.8 → 
  discount2 = 0.85 → 
  discount1 * marked_price - cost_price = profit1 → 

  (discount2 * marked_price - cost_price) = 337.5 := 
by 
  intros; 
  let marked_price := 2750; 
  sorry

end NUMINAMATH_GPT_refrigerator_profit_l1363_136353


namespace NUMINAMATH_GPT_exists_xi_l1363_136396

variable (f : ℝ → ℝ)
variable (hf_diff : ∀ x, DifferentiableAt ℝ f x)
variable (hf_twice_diff : ∀ x, DifferentiableAt ℝ (deriv f) x)
variable (hf₀ : f 0 = 2)
variable (hf_prime₀ : deriv f 0 = -2)
variable (hf₁ : f 1 = 1)

theorem exists_xi (h0 : f 0 = 2) (h1 : deriv f 0 = -2) (h2 : f 1 = 1) :
  ∃ ξ ∈ Set.Ioo 0 1, f ξ * deriv f ξ + deriv (deriv f) ξ = 0 :=
sorry

end NUMINAMATH_GPT_exists_xi_l1363_136396


namespace NUMINAMATH_GPT_remainder_when_sum_divided_l1363_136371

theorem remainder_when_sum_divided (p q : ℕ) (m n : ℕ) (hp : p = 80 * m + 75) (hq : q = 120 * n + 115) :
  (p + q) % 40 = 30 := 
by sorry

end NUMINAMATH_GPT_remainder_when_sum_divided_l1363_136371


namespace NUMINAMATH_GPT_Dave_earning_l1363_136362

def action_games := 3
def adventure_games := 2
def role_playing_games := 3

def price_action := 6
def price_adventure := 5
def price_role_playing := 7

def earning_from_action_games := action_games * price_action
def earning_from_adventure_games := adventure_games * price_adventure
def earning_from_role_playing_games := role_playing_games * price_role_playing

def total_earning := earning_from_action_games + earning_from_adventure_games + earning_from_role_playing_games

theorem Dave_earning : total_earning = 49 := by
  show total_earning = 49
  sorry

end NUMINAMATH_GPT_Dave_earning_l1363_136362


namespace NUMINAMATH_GPT_abs_eq_imp_b_eq_2_l1363_136361

theorem abs_eq_imp_b_eq_2 (b : ℝ) (h : |1 - b| = |3 - b|) : b = 2 := 
sorry

end NUMINAMATH_GPT_abs_eq_imp_b_eq_2_l1363_136361


namespace NUMINAMATH_GPT_find_tangent_point_l1363_136321

theorem find_tangent_point (x : ℝ) (y : ℝ) (h_curve : y = x^2) (h_slope : 2 * x = 1) : 
    (x, y) = (1/2, 1/4) :=
sorry

end NUMINAMATH_GPT_find_tangent_point_l1363_136321


namespace NUMINAMATH_GPT_fish_count_when_james_discovers_l1363_136340

def fish_in_aquarium (initial_fish : ℕ) (bobbit_worm_eats : ℕ) (predatory_fish_eats : ℕ)
  (reproduction_rate : ℕ × ℕ) (days_1 : ℕ) (added_fish: ℕ) (days_2 : ℕ) : ℕ :=
  let predation_rate := bobbit_worm_eats + predatory_fish_eats
  let total_eaten_in_14_days := predation_rate * days_1
  let reproduction_events_in_14_days := days_1 / reproduction_rate.snd
  let fish_born_in_14_days := reproduction_events_in_14_days * reproduction_rate.fst
  let fish_after_14_days := initial_fish - total_eaten_in_14_days + fish_born_in_14_days
  let fish_after_14_days_non_negative := max fish_after_14_days 0
  let fish_after_addition := fish_after_14_days_non_negative + added_fish
  let total_eaten_in_7_days := predation_rate * days_2
  let reproduction_events_in_7_days := days_2 / reproduction_rate.snd
  let fish_born_in_7_days := reproduction_events_in_7_days * reproduction_rate.fst
  let fish_after_7_days := fish_after_addition - total_eaten_in_7_days + fish_born_in_7_days
  max fish_after_7_days 0

theorem fish_count_when_james_discovers :
  fish_in_aquarium 60 2 3 (2, 3) 14 8 7 = 4 :=
sorry

end NUMINAMATH_GPT_fish_count_when_james_discovers_l1363_136340


namespace NUMINAMATH_GPT_x4_value_l1363_136386

/-- Define x_n sequence based on given initial value and construction rules -/
def x_n (n : ℕ) : ℕ :=
  if n = 1 then 27
  else if n = 2 then 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2 + 1
  else if n = 3 then 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2 + 1 -- Need to generalize for actual sequence definition
  else if n = 4 then 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 1 * 2 + 1
  else 0 -- placeholder for general case (not needed here)

/-- Prove that x_4 = 23 given x_1=27 and the sequence construction criteria --/
theorem x4_value : x_n 4 = 23 :=
by
  -- Proof not required, hence sorry is used
  sorry

end NUMINAMATH_GPT_x4_value_l1363_136386


namespace NUMINAMATH_GPT_length_of_DC_l1363_136374

theorem length_of_DC (AB : ℝ) (angle_ADB : ℝ) (sin_A : ℝ) (sin_C : ℝ)
  (h1 : AB = 30) (h2 : angle_ADB = pi / 2) (h3 : sin_A = 3 / 5) (h4 : sin_C = 1 / 4) :
  ∃ DC : ℝ, DC = 18 * Real.sqrt 15 :=
by
  sorry

end NUMINAMATH_GPT_length_of_DC_l1363_136374


namespace NUMINAMATH_GPT_fraction_of_canvas_painted_blue_l1363_136343

noncomputable def square_canvas_blue_fraction : ℚ :=
  sorry

theorem fraction_of_canvas_painted_blue :
  square_canvas_blue_fraction = 3 / 8 :=
  sorry

end NUMINAMATH_GPT_fraction_of_canvas_painted_blue_l1363_136343


namespace NUMINAMATH_GPT_sum_series_l1363_136319

noncomputable def series_sum := (∑' n : ℕ, (4 * (n + 1) - 2) / 3^(n + 1))

theorem sum_series : series_sum = 4 := by
  sorry

end NUMINAMATH_GPT_sum_series_l1363_136319


namespace NUMINAMATH_GPT_minimum_value_l1363_136399

open Real

theorem minimum_value (x : ℝ) (hx : x > 2) : 
  ∃ y ≥ 4 * Real.sqrt 2, ∀ z, (z = (x + 6) / (Real.sqrt (x - 2)) → y ≤ z) := 
sorry

end NUMINAMATH_GPT_minimum_value_l1363_136399


namespace NUMINAMATH_GPT_cos_half_pi_plus_alpha_l1363_136373

theorem cos_half_pi_plus_alpha (α : ℝ) (h : Real.sin (π - α) = 1 / 3) : Real.cos (π / 2 + α) = - (1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_cos_half_pi_plus_alpha_l1363_136373


namespace NUMINAMATH_GPT_rods_in_one_mile_l1363_136363

theorem rods_in_one_mile (mile_to_furlong : ℕ) (furlong_to_rod : ℕ) (mile_eq : 1 = 8 * mile_to_furlong) (furlong_eq: 1 = 50 * furlong_to_rod) : 
  (1 * 8 * 50 = 400) :=
by
  sorry

end NUMINAMATH_GPT_rods_in_one_mile_l1363_136363


namespace NUMINAMATH_GPT_calc_total_push_ups_correct_l1363_136377

-- Definitions based on conditions
def sets : ℕ := 9
def push_ups_per_set : ℕ := 12
def reduced_push_ups : ℕ := 8

-- Calculate total push-ups considering the reduction in the ninth set
def total_push_ups (sets : ℕ) (push_ups_per_set : ℕ) (reduced_push_ups : ℕ) : ℕ :=
  (sets - 1) * push_ups_per_set + (push_ups_per_set - reduced_push_ups)

-- Theorem statement
theorem calc_total_push_ups_correct :
  total_push_ups sets push_ups_per_set reduced_push_ups = 100 :=
by
  sorry

end NUMINAMATH_GPT_calc_total_push_ups_correct_l1363_136377


namespace NUMINAMATH_GPT_minimum_cost_for_28_apples_l1363_136314

/--
Conditions:
  - apples can be bought at a rate of 4 for 15 cents,
  - apples can be bought at a rate of 7 for 30 cents,
  - you need to buy exactly 28 apples.
Prove that the minimum total cost to buy exactly 28 apples is 120 cents.
-/
theorem minimum_cost_for_28_apples : 
  let cost_4_for_15 := 15
  let cost_7_for_30 := 30
  let apples_needed := 28
  ∃ (n m : ℕ), n * 4 + m * 7 = apples_needed ∧ n * cost_4_for_15 + m * cost_7_for_30 = 120 := sorry

end NUMINAMATH_GPT_minimum_cost_for_28_apples_l1363_136314


namespace NUMINAMATH_GPT_determine_amount_of_substance_l1363_136341

noncomputable def amount_of_substance 
  (A : ℝ) (R : ℝ) (delta_T : ℝ) : ℝ :=
  (2 * A) / (R * delta_T)

theorem determine_amount_of_substance 
  (A : ℝ := 831) 
  (R : ℝ := 8.31) 
  (delta_T : ℝ := 100) 
  (nu : ℝ := amount_of_substance A R delta_T) :
  nu = 2 := by
  -- Conditions rewritten as definitions
  -- Definition: A = 831 J
  -- Definition: R = 8.31 J/(mol * K)
  -- Definition: delta_T = 100 K
  -- The correct answer to be proven: nu = 2 mol
  sorry

end NUMINAMATH_GPT_determine_amount_of_substance_l1363_136341


namespace NUMINAMATH_GPT_age_in_1988_equals_sum_of_digits_l1363_136357

def birth_year (x y : ℕ) : ℕ := 1900 + 10 * x + y

def age_in_1988 (birth_year : ℕ) : ℕ := 1988 - birth_year

def sum_of_digits (x y : ℕ) : ℕ := 1 + 9 + x + y

theorem age_in_1988_equals_sum_of_digits (x y : ℕ) (h0 : 0 ≤ x) (h1 : x ≤ 9) (h2 : 0 ≤ y) (h3 : y ≤ 9) 
  (h4 : age_in_1988 (birth_year x y) = sum_of_digits x y) :
  age_in_1988 (birth_year x y) = 22 :=
by {
  sorry
}

end NUMINAMATH_GPT_age_in_1988_equals_sum_of_digits_l1363_136357


namespace NUMINAMATH_GPT_arithmetic_expression_evaluation_l1363_136347

theorem arithmetic_expression_evaluation :
  4 * 6 + 8 * 3 - 28 / 2 = 34 := by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_evaluation_l1363_136347


namespace NUMINAMATH_GPT_original_cost_of_each_bag_l1363_136313

theorem original_cost_of_each_bag (C : ℕ) (hC : C % 13 = 0) (h4 : (85 * C) % 400 = 0) : C / 5 = 208 := by
  sorry

end NUMINAMATH_GPT_original_cost_of_each_bag_l1363_136313


namespace NUMINAMATH_GPT_solve_problem_l1363_136338

theorem solve_problem : 
  ∃ p q : ℝ, 
    (p ≠ q) ∧ 
    ((∀ x : ℝ, (x = p ∨ x = q) ↔ (x-4)*(x+4) = 24*x - 96)) ∧ 
    (p > q) ∧ 
    (p - q = 16) :=
by
  sorry

end NUMINAMATH_GPT_solve_problem_l1363_136338


namespace NUMINAMATH_GPT_rate_per_square_meter_l1363_136336

theorem rate_per_square_meter 
  (L : ℝ) (W : ℝ) (C : ℝ)
  (hL : L = 5.5) 
  (hW : W = 3.75)
  (hC : C = 20625)
  : C / (L * W) = 1000 :=
by
  sorry

end NUMINAMATH_GPT_rate_per_square_meter_l1363_136336


namespace NUMINAMATH_GPT_class_size_l1363_136356

theorem class_size (n : ℕ) (h1 : 85 - 33 + 90 - 40 = 102) (h2 : (102 : ℚ) / n = 1.5): n = 68 :=
by
  sorry

end NUMINAMATH_GPT_class_size_l1363_136356


namespace NUMINAMATH_GPT_sample_mean_and_variance_l1363_136326

def sample : List ℕ := [10, 12, 9, 14, 13]
def n : ℕ := 5

-- Definition of sample mean
noncomputable def sampleMean : ℝ := (sample.sum / n)

-- Definition of sample variance using population formula
noncomputable def sampleVariance : ℝ := (sample.map (λ x_i => (x_i - sampleMean)^2)).sum / n

theorem sample_mean_and_variance :
  sampleMean = 11.6 ∧ sampleVariance = 3.44 := by
  sorry

end NUMINAMATH_GPT_sample_mean_and_variance_l1363_136326


namespace NUMINAMATH_GPT_arrangement_of_letters_l1363_136301

-- Define the set of letters with subscripts
def letters : Finset String := {"B", "A₁", "B₁", "A₂", "B₂", "A₃"}

-- Define the number of ways to arrange 6 distinct letters
theorem arrangement_of_letters : letters.card.factorial = 720 := 
by {
  sorry
}

end NUMINAMATH_GPT_arrangement_of_letters_l1363_136301


namespace NUMINAMATH_GPT_elena_pens_l1363_136317

theorem elena_pens (X Y : ℕ) (h1 : X + Y = 12) (h2 : 4*X + 22*Y = 420) : X = 9 := by
  sorry

end NUMINAMATH_GPT_elena_pens_l1363_136317


namespace NUMINAMATH_GPT_each_child_plays_equally_l1363_136312

theorem each_child_plays_equally (total_time : ℕ) (num_children : ℕ)
  (play_group_size : ℕ) (play_time : ℕ) :
  num_children = 6 ∧ play_group_size = 3 ∧ total_time = 120 ∧ play_time = (total_time * play_group_size) / num_children →
  play_time = 60 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_each_child_plays_equally_l1363_136312


namespace NUMINAMATH_GPT_number_of_bass_caught_l1363_136350

/-
Statement:
Given:
1. An eight-pound trout.
2. Two twelve-pound salmon.
3. They need to feed 22 campers with two pounds of fish each.
Prove that the number of two-pound bass caught is 6.
-/

theorem number_of_bass_caught
  (weight_trout : ℕ := 8)
  (weight_salmon : ℕ := 12)
  (num_salmon : ℕ := 2)
  (num_campers : ℕ := 22)
  (required_per_camper : ℕ := 2)
  (weight_bass : ℕ := 2) :
  (num_campers * required_per_camper - (weight_trout + num_salmon * weight_salmon)) / weight_bass = 6 :=
by
  sorry  -- Proof to be completed

end NUMINAMATH_GPT_number_of_bass_caught_l1363_136350


namespace NUMINAMATH_GPT_find_reggie_long_shots_l1363_136383

-- Define the constants used in the problem
def layup_points : ℕ := 1
def free_throw_points : ℕ := 2
def long_shot_points : ℕ := 3

-- Define Reggie's shooting results
def reggie_layups : ℕ := 3
def reggie_free_throws : ℕ := 2
def reggie_long_shots : ℕ := sorry -- we need to find this

-- Define Reggie's brother's shooting results
def brother_long_shots : ℕ := 4

-- Given conditions
def reggie_total_points := reggie_layups * layup_points + reggie_free_throws * free_throw_points + reggie_long_shots * long_shot_points
def brother_total_points := brother_long_shots * long_shot_points

def reggie_lost_by_2_points := reggie_total_points + 2 = brother_total_points

-- The theorem we need to prove
theorem find_reggie_long_shots : reggie_long_shots = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_reggie_long_shots_l1363_136383


namespace NUMINAMATH_GPT_fuel_calculation_l1363_136367

def total_fuel_needed (empty_fuel_per_mile people_fuel_per_mile bag_fuel_per_mile num_passengers num_crew bags_per_person miles : ℕ) : ℕ :=
  let total_people := num_passengers + num_crew
  let total_bags := total_people * bags_per_person
  let total_fuel_per_mile := empty_fuel_per_mile + people_fuel_per_mile * total_people + bag_fuel_per_mile * total_bags
  total_fuel_per_mile * miles

theorem fuel_calculation :
  total_fuel_needed 20 3 2 30 5 2 400 = 106000 :=
by
  sorry

end NUMINAMATH_GPT_fuel_calculation_l1363_136367


namespace NUMINAMATH_GPT_find_side_b_in_triangle_l1363_136349

theorem find_side_b_in_triangle (A B C : ℝ) (a b c : ℝ) 
  (h1 : A + C = 2 * B) 
  (h2 : A + B + C = 180) 
  (h3 : a + c = 8) 
  (h4 : a * c = 15) 
  (h5 : (b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos (B * Real.pi / 180))) : 
  b = Real.sqrt 19 := 
  by sorry

end NUMINAMATH_GPT_find_side_b_in_triangle_l1363_136349


namespace NUMINAMATH_GPT_tank_full_capacity_l1363_136391

-- Define the conditions
def gas_tank_initially_full_fraction : ℚ := 4 / 5
def gas_tank_after_usage_fraction : ℚ := 1 / 3
def used_gallons : ℚ := 18

-- Define the statement that translates to "How many gallons does this tank hold when it is full?"
theorem tank_full_capacity (x : ℚ) : 
  gas_tank_initially_full_fraction * x - gas_tank_after_usage_fraction * x = used_gallons → 
  x = 270 / 7 :=
sorry

end NUMINAMATH_GPT_tank_full_capacity_l1363_136391


namespace NUMINAMATH_GPT_least_positive_integer_fac_6370_factorial_l1363_136328

theorem least_positive_integer_fac_6370_factorial :
  ∃ (n : ℕ), (∀ m : ℕ, (6370 ∣ m.factorial) → m ≥ n) ∧ n = 14 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_fac_6370_factorial_l1363_136328


namespace NUMINAMATH_GPT_find_all_good_sets_l1363_136359

def is_good_set (A : Finset ℕ) : Prop :=
  (∀ (a b c : ℕ), a ∈ A → b ∈ A → c ∈ A → a ≠ b → b ≠ c → a ≠ c → Nat.gcd a (Nat.gcd b c) = 1) ∧
  (∀ (b c : ℕ), b ∈ A → c ∈ A → b ≠ c → ∃ (a : ℕ), a ∈ A ∧ a ≠ b ∧ a ≠ c ∧ (b * c) % a = 0)

theorem find_all_good_sets : ∀ (A : Finset ℕ), is_good_set A ↔ 
  (A = {a, b, a * b} ∧ Nat.gcd a b = 1) ∨ 
  ∃ (p q r : ℕ), Nat.gcd p q = 1 ∧ Nat.gcd q r = 1 ∧ Nat.gcd r p = 1 ∧ A = {p * q, q * r, r * p} :=
by
  sorry

end NUMINAMATH_GPT_find_all_good_sets_l1363_136359


namespace NUMINAMATH_GPT_find_kids_l1363_136305

theorem find_kids (A K : ℕ) (h1 : A + K = 12) (h2 : 3 * A = 15) : K = 7 :=
sorry

end NUMINAMATH_GPT_find_kids_l1363_136305


namespace NUMINAMATH_GPT_arithmetic_seq_sum_equidistant_l1363_136322

variable (a : ℕ → ℤ)

theorem arithmetic_seq_sum_equidistant :
  (∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) → a 4 = 12 → a 1 + a 7 = 24 :=
by
  intros h_seq h_a4
  sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_equidistant_l1363_136322


namespace NUMINAMATH_GPT_find_a_l1363_136334

noncomputable def M (a : ℤ) : Set ℤ := {a, 0}
noncomputable def N : Set ℤ := { x : ℤ | 2 * x^2 - 3 * x < 0 }

theorem find_a (a : ℤ) (h : (M a ∩ N).Nonempty) : a = 1 := sorry

end NUMINAMATH_GPT_find_a_l1363_136334


namespace NUMINAMATH_GPT_ratio_of_beef_to_pork_l1363_136309

/-- 
James buys 20 pounds of beef. 
James buys an unknown amount of pork. 
James uses 1.5 pounds of meat to make each meal. 
Each meal sells for $20. 
James made $400 from selling meals.
The ratio of the amount of beef to the amount of pork James bought is 2:1.
-/
theorem ratio_of_beef_to_pork (beef pork : ℝ) (meal_weight : ℝ) (meal_price : ℝ) (total_revenue : ℝ)
  (h_beef : beef = 20)
  (h_meal_weight : meal_weight = 1.5)
  (h_meal_price : meal_price = 20)
  (h_total_revenue : total_revenue = 400) :
  (beef / pork) = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_beef_to_pork_l1363_136309


namespace NUMINAMATH_GPT_equal_sets_l1363_136385

def M : Set ℝ := {x | x^2 + 16 = 0}
def N : Set ℝ := {x | x^2 + 6 = 0}

theorem equal_sets : M = N := by
  sorry

end NUMINAMATH_GPT_equal_sets_l1363_136385


namespace NUMINAMATH_GPT_initial_apples_correct_l1363_136304

-- Define the conditions
def apples_handout : Nat := 5
def pies_made : Nat := 9
def apples_per_pie : Nat := 5

-- Calculate the number of apples used for pies
def apples_for_pies := pies_made * apples_per_pie

-- Define the total number of apples initially
def apples_initial := apples_for_pies + apples_handout

-- State the theorem to prove
theorem initial_apples_correct : apples_initial = 50 :=
by
  sorry

end NUMINAMATH_GPT_initial_apples_correct_l1363_136304


namespace NUMINAMATH_GPT_Tim_total_money_l1363_136315

theorem Tim_total_money :
  let nickels_amount := 3 * 0.05
  let dimes_amount_shoes := 13 * 0.10
  let shining_shoes := nickels_amount + dimes_amount_shoes
  let dimes_amount_tip_jar := 7 * 0.10
  let half_dollars_amount := 9 * 0.50
  let tip_jar := dimes_amount_tip_jar + half_dollars_amount
  let total := shining_shoes + tip_jar
  total = 6.65 :=
by
  sorry

end NUMINAMATH_GPT_Tim_total_money_l1363_136315


namespace NUMINAMATH_GPT_interest_rate_is_10_percent_l1363_136398

theorem interest_rate_is_10_percent (P : ℝ) (t : ℝ) (d : ℝ) (r : ℝ) 
  (hP : P = 9999.99999999988) 
  (ht : t = 1) 
  (hd : d = 25)
  : P * (1 + r / 2)^(2 * t) - P - (P * r * t) = d → r = 0.1 :=
by
  intros h
  rw [hP, ht, hd] at h
  sorry

end NUMINAMATH_GPT_interest_rate_is_10_percent_l1363_136398


namespace NUMINAMATH_GPT_has_zero_when_a_gt_0_l1363_136397

noncomputable def f (x a : ℝ) : ℝ :=
  x * Real.log (x - 1) - a

theorem has_zero_when_a_gt_0 (a : ℝ) (h : a > 0) :
  ∃ x0 : ℝ, f x0 a = 0 ∧ 2 < x0 :=
sorry

end NUMINAMATH_GPT_has_zero_when_a_gt_0_l1363_136397


namespace NUMINAMATH_GPT_find_matrix_A_l1363_136329

-- Let A be a 2x2 matrix such that 
def A (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![a, b], ![c, d]]

theorem find_matrix_A :
  ∃ A : Matrix (Fin 2) (Fin 2) ℝ,
  (A.mulVec ![4, 1] = ![8, 14]) ∧ (A.mulVec ![2, -3] = ![-2, 11]) ∧
  A = ![![2, 1/2], ![-1, -13/3]] :=
by
  sorry

end NUMINAMATH_GPT_find_matrix_A_l1363_136329


namespace NUMINAMATH_GPT_cubic_inches_needed_l1363_136332

/-- The dimensions of each box are 20 inches by 20 inches by 12 inches. -/
def box_length : ℝ := 20
def box_width : ℝ := 20
def box_height : ℝ := 12

/-- The cost of each box is $0.40. -/
def box_cost : ℝ := 0.40

/-- The minimum spending required by the university on boxes is $200. -/
def min_spending : ℝ := 200

/-- Given the above conditions, the total cubic inches needed to package the collection is 2,400,000 cubic inches. -/
theorem cubic_inches_needed :
  (min_spending / box_cost) * (box_length * box_width * box_height) = 2400000 := by
  sorry

end NUMINAMATH_GPT_cubic_inches_needed_l1363_136332


namespace NUMINAMATH_GPT_part1_part2_l1363_136327

noncomputable def f (x : ℝ) := |x - 3| + |x - 4|

theorem part1 (a : ℝ) (h : ∃ x : ℝ, f x < a) : a > 1 :=
sorry

theorem part2 (x : ℝ) : f x ≥ 7 + 7 * x - x ^ 2 ↔ x ≤ 0 ∨ 7 ≤ x :=
sorry

end NUMINAMATH_GPT_part1_part2_l1363_136327


namespace NUMINAMATH_GPT_cistern_empty_time_without_tap_l1363_136395

noncomputable def leak_rate (L : ℕ) : Prop :=
  let tap_rate := 4
  let cistern_volume := 480
  let empty_time_with_tap := 24
  let empty_rate_net := cistern_volume / empty_time_with_tap
  L - tap_rate = empty_rate_net

theorem cistern_empty_time_without_tap (L : ℕ) (h : leak_rate L) :
  480 / L = 20 := by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_cistern_empty_time_without_tap_l1363_136395


namespace NUMINAMATH_GPT_probability_each_person_selected_l1363_136389

-- Define the number of initial participants
def initial_participants := 2007

-- Define the number of participants to exclude
def exclude_participants := 7

-- Define the final number of participants remaining after exclusion
def remaining_participants := initial_participants - exclude_participants

-- Define the number of participants to select
def select_participants := 50

-- Define the probability of each participant being selected
def selection_probability : ℚ :=
  select_participants * remaining_participants / (initial_participants * remaining_participants)

theorem probability_each_person_selected :
  selection_probability = (50 / 2007 : ℚ) :=
sorry

end NUMINAMATH_GPT_probability_each_person_selected_l1363_136389


namespace NUMINAMATH_GPT_mass_percentage_H_in_NH4I_is_correct_l1363_136387

noncomputable def molar_mass_NH4I : ℝ := 1 * 14.01 + 4 * 1.01 + 1 * 126.90

noncomputable def mass_H_in_NH4I : ℝ := 4 * 1.01

noncomputable def mass_percentage_H_in_NH4I : ℝ := (mass_H_in_NH4I / molar_mass_NH4I) * 100

theorem mass_percentage_H_in_NH4I_is_correct :
  abs (mass_percentage_H_in_NH4I - 2.79) < 0.01 := by
  sorry

end NUMINAMATH_GPT_mass_percentage_H_in_NH4I_is_correct_l1363_136387


namespace NUMINAMATH_GPT_race_problem_l1363_136345

theorem race_problem 
    (d : ℕ) (a1 : ℕ) (a2 : ℕ) 
    (h1 : d = 60)
    (h2 : a1 = 10)
    (h3 : a2 = 20) 
    (const_speed : ∀ (x y z : ℕ), x * y = z → y ≠ 0 → x = z / y) :
  (d - d * (d - a1) / (d - a2) = 12) := 
by {
  sorry
}

end NUMINAMATH_GPT_race_problem_l1363_136345


namespace NUMINAMATH_GPT_three_consecutive_arithmetic_l1363_136368

def seq (n : ℕ) : ℝ := 
  if n % 2 = 1 then (n : ℝ)
  else 2 * 3^(n / 2 - 1)

theorem three_consecutive_arithmetic (m : ℕ) (h_m : seq m + seq (m+2) = 2 * seq (m+1)) : m = 1 :=
  sorry

end NUMINAMATH_GPT_three_consecutive_arithmetic_l1363_136368


namespace NUMINAMATH_GPT_savings_by_december_l1363_136302

-- Define the basic conditions
def initial_savings : ℕ := 1147240
def total_income : ℕ := (55000 + 45000 + 10000 + 17400) * 4
def total_expenses : ℕ := (40000 + 20000 + 5000 + 2000 + 2000) * 4

-- Define the final savings calculation
def final_savings : ℕ := initial_savings + total_income - total_expenses

-- The theorem to be proved
theorem savings_by_december : final_savings = 1340840 := by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_savings_by_december_l1363_136302
