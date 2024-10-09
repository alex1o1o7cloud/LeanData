import Mathlib

namespace pump_A_time_l954_95482

theorem pump_A_time (B C A : ℝ) (hB : B = 1/3) (hC : C = 1/6)
(h : (A + B - C) * 0.75 = 0.5) : 1 / A = 2 :=
by
sorry

end pump_A_time_l954_95482


namespace sum_of_squares_l954_95485

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 72) : x^2 + y^2 = 180 :=
sorry

end sum_of_squares_l954_95485


namespace sophie_total_spend_l954_95473

-- Definitions based on conditions
def cost_cupcakes : ℕ := 5 * 2
def cost_doughnuts : ℕ := 6 * 1
def cost_apple_pie : ℕ := 4 * 2
def cost_cookies : ℕ := 15 * 6 / 10 -- since 0.60 = 6/10

-- Total cost
def total_cost : ℕ := cost_cupcakes + cost_doughnuts + cost_apple_pie + cost_cookies

-- Prove the total cost
theorem sophie_total_spend : total_cost = 33 := by
  sorry

end sophie_total_spend_l954_95473


namespace roots_polynomial_d_l954_95407

theorem roots_polynomial_d (c d u v : ℝ) (ru rpush rv rpush2 : ℝ) :
    (u + v + ru = 0) ∧ (u+3 + v-2 + rpush2 = 0) ∧
    (d + 153 = -(u + 3) * (v - 2) * (ru)) ∧ (d + 153 = s) ∧ (s = -(u + 3) * (v - 2) * (rpush2 - 1)) →
    d = 0 :=
by
  sorry

end roots_polynomial_d_l954_95407


namespace converse_even_sum_l954_95403

def is_even (n : Int) : Prop := ∃ k : Int, n = 2 * k

theorem converse_even_sum (a b : Int) :
  (is_even a ∧ is_even b → is_even (a + b)) →
  (is_even (a + b) → is_even a ∧ is_even b) :=
by
  sorry

end converse_even_sum_l954_95403


namespace july14_2030_is_sunday_l954_95472

-- Define the given condition that July 3, 2030 is a Wednesday. 
def july3_2030_is_wednesday : Prop := true -- Assume the existence and correctness of this statement.

-- Define the proof problem that July 14, 2030 is a Sunday given the above condition.
theorem july14_2030_is_sunday : july3_2030_is_wednesday → (14 % 7 = 0) := 
sorry

end july14_2030_is_sunday_l954_95472


namespace correct_articles_l954_95415

-- Definitions based on conditions provided in the problem
def sentence := "Traveling in ____ outer space is quite ____ exciting experience."
def first_blank_article := "no article"
def second_blank_article := "an"

-- Statement of the proof problem
theorem correct_articles : 
  (first_blank_article = "no article" ∧ second_blank_article = "an") :=
by
  sorry

end correct_articles_l954_95415


namespace determine_a_for_unique_solution_of_quadratic_l954_95418

theorem determine_a_for_unique_solution_of_quadratic :
  {a : ℝ | ∃! x : ℝ, a * x^2 - 4 * x + 2 = 0} = {0, 2} :=
sorry

end determine_a_for_unique_solution_of_quadratic_l954_95418


namespace tens_digit_of_13_pow_2021_l954_95474

theorem tens_digit_of_13_pow_2021 :
  let p := 2021
  let base := 13
  let mod_val := 100
  let digit := (base^p % mod_val) / 10
  digit = 1 := by
  sorry

end tens_digit_of_13_pow_2021_l954_95474


namespace john_speed_first_part_l954_95488

theorem john_speed_first_part (S : ℝ) (h1 : 2 * S + 3 * 55 = 255) : S = 45 :=
by
  sorry

end john_speed_first_part_l954_95488


namespace smallest_x_satisfying_abs_eq_l954_95433

theorem smallest_x_satisfying_abs_eq (x : ℝ) 
  (h : |2 * x^2 + 3 * x - 1| = 33) : 
  x = (-3 - Real.sqrt 281) / 4 := 
sorry

end smallest_x_satisfying_abs_eq_l954_95433


namespace min_value_expression_l954_95417

theorem min_value_expression (x y : ℝ) : (x^2 * y - 1)^2 + (x + y - 1)^2 ≥ 1 :=
sorry

end min_value_expression_l954_95417


namespace calculate_expression_l954_95406

theorem calculate_expression : (Real.sqrt 4) + abs (3 - Real.pi) + (1 / 3)⁻¹ = 2 + Real.pi :=
by 
  sorry

end calculate_expression_l954_95406


namespace sum_of_first_twelve_terms_l954_95443

section ArithmeticSequence

variables (a : ℕ → ℚ) (d : ℚ) (a₁ : ℚ)

-- General definition of the nth term in arithmetic progression
def arithmetic_term (n : ℕ) : ℚ := a₁ + (n - 1) * d

-- Given conditions in the problem
axiom fifth_term : arithmetic_term a₁ d 5 = 1
axiom seventeenth_term : arithmetic_term a₁ d 17 = 18

-- Define the sum of the first n terms in arithmetic sequence
def sum_arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

-- Statement of the proof problem
theorem sum_of_first_twelve_terms : 
  sum_arithmetic_sequence a₁ d 12 = 37.5 := 
sorry

end ArithmeticSequence

end sum_of_first_twelve_terms_l954_95443


namespace solve_for_nabla_l954_95483

theorem solve_for_nabla (nabla : ℤ) (h : 3 * (-2) = nabla + 2) : nabla = -8 :=
by
  sorry

end solve_for_nabla_l954_95483


namespace heartsuit_ratio_l954_95491

def k : ℝ := 3

def heartsuit (n m : ℕ) : ℝ := k * n^3 * m^2

theorem heartsuit_ratio : (heartsuit 3 5) / (heartsuit 5 3) = 3 / 5 := 
by
  sorry

end heartsuit_ratio_l954_95491


namespace bricks_required_l954_95414

-- Definitions
def courtyard_length : ℕ := 20  -- in meters
def courtyard_breadth : ℕ := 16  -- in meters
def brick_length : ℕ := 20  -- in centimeters
def brick_breadth : ℕ := 10  -- in centimeters

-- Statement to prove
theorem bricks_required :
  ((courtyard_length * 100) * (courtyard_breadth * 100)) / (brick_length * brick_breadth) = 16000 :=
sorry

end bricks_required_l954_95414


namespace find_c_find_cos_2B_minus_pi_over_4_l954_95452

variable (A B C : Real) (a b c : Real)

-- Given conditions
def conditions (a b c : Real) (A : Real) : Prop :=
  a = 4 * Real.sqrt 3 ∧
  b = 6 ∧
  Real.cos A = -1 / 3

-- Proof of question 1
theorem find_c (h : conditions a b c A) : c = 2 :=
sorry

-- Proof of question 2
theorem find_cos_2B_minus_pi_over_4 (h : conditions a b c A) (B : Real) :
  (angle_opp_b : b = Real.sin B) → -- This is to ensure B is the angle opposite to side b
  Real.cos (2 * B - Real.pi / 4) = (4 - Real.sqrt 2) / 6 :=
sorry

end find_c_find_cos_2B_minus_pi_over_4_l954_95452


namespace problem1_l954_95481

variable (x y : ℝ)
variable (h1 : x = Real.sqrt 3 + Real.sqrt 5)
variable (h2 : y = Real.sqrt 3 - Real.sqrt 5)

theorem problem1 : 2 * x^2 - 4 * x * y + 2 * y^2 = 40 :=
by sorry

end problem1_l954_95481


namespace remainder_7n_mod_4_l954_95497

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
by sorry

end remainder_7n_mod_4_l954_95497


namespace gcd_number_between_75_and_90_is_5_l954_95460

theorem gcd_number_between_75_and_90_is_5 :
  ∃ n : ℕ, 75 ≤ n ∧ n ≤ 90 ∧ Nat.gcd 15 n = 5 :=
sorry

end gcd_number_between_75_and_90_is_5_l954_95460


namespace per_capita_income_ratio_l954_95412

theorem per_capita_income_ratio
  (PL_10 PZ_10 PL_now PZ_now : ℝ)
  (h1 : PZ_10 = 0.4 * PL_10)
  (h2 : PZ_now = 0.8 * PL_now)
  (h3 : PL_now = 3 * PL_10) :
  PZ_now / PZ_10 = 6 := by
  -- Proof to be filled
  sorry

end per_capita_income_ratio_l954_95412


namespace part1_part2_part3_l954_95430

def climbing_function_1_example (x : ℝ) : Prop :=
  ∃ a : ℝ, a^2 = -8 / a

theorem part1 (x : ℝ) : climbing_function_1_example x ↔ (x = -2) := sorry

def climbing_function_2_example (m : ℝ) : Prop :=
  ∃ a : ℝ, (a^2 = m*a + m) ∧ ∀ d: ℝ, ((d^2 = m*d + m) → d = a)

theorem part2 (m : ℝ) : (m = -4) ∧ climbing_function_2_example m := sorry

def climbing_function_3_example (m n p q : ℝ) (h1 : m ≥ 2) (h2 : p^2 = 3*q) : Prop :=
  ∃ a1 a2 : ℝ, ((a1 + a2 = n/(1-m)) ∧ (a1*a2 = 1/(m-1)) ∧ (|a1 - a2| = p)) ∧ 
  (∀ x : ℝ, (m * x^2 + n * x + 1) ≥ q) 

theorem part3 (m n p q : ℝ) (h1 : m ≥ 2) (h2 : p^2 = 3*q) : climbing_function_3_example m n p q h1 h2 ↔ (0 < q) ∧ (q ≤ 4/11) := sorry

end part1_part2_part3_l954_95430


namespace CH4_reaction_with_Cl2_l954_95455

def balanced_chemical_equation (CH4 Cl2 CH3Cl HCl : ℕ) : Prop :=
  CH4 + Cl2 = CH3Cl + HCl

theorem CH4_reaction_with_Cl2
  (CH4 Cl2 CH3Cl HCl : ℕ)
  (balanced_eq : balanced_chemical_equation 1 1 1 1)
  (reaction_cl2 : Cl2 = 2) :
  CH4 = 2 :=
by
  sorry

end CH4_reaction_with_Cl2_l954_95455


namespace incorrect_options_l954_95475

variable (a b : ℚ) (h : a / b = 5 / 6)

theorem incorrect_options :
  (2 * a - b ≠ b * 6 / 4) ∧
  (a + 3 * b ≠ 2 * a * 19 / 10) :=
by
  sorry

end incorrect_options_l954_95475


namespace find_small_pack_size_l954_95446

-- Define the conditions of the problem
def soymilk_sold_in_packs (pack_size : ℕ) : Prop :=
  pack_size = 2 ∨ ∃ L : ℕ, pack_size = L

def cartons_bought (total_cartons : ℕ) (large_pack_size : ℕ) (num_large_packs : ℕ) (small_pack_size : ℕ) : Prop :=
  total_cartons = num_large_packs * large_pack_size + small_pack_size

-- The problem statement as a Lean theorem
theorem find_small_pack_size (total_cartons : ℕ) (num_large_packs : ℕ) (large_pack_size : ℕ) :
  soymilk_sold_in_packs 2 →
  soymilk_sold_in_packs large_pack_size →
  cartons_bought total_cartons large_pack_size num_large_packs 2 →
  total_cartons = 17 →
  num_large_packs = 3 →
  large_pack_size = 5 →
  ∃ S : ℕ, soymilk_sold_in_packs S ∧ S = 2 :=
by
  sorry

end find_small_pack_size_l954_95446


namespace rectangle_area_3650_l954_95409

variables (L B : ℕ)

-- Conditions given in the problem
def condition1 : Prop := L - B = 23
def condition2 : Prop := 2 * (L + B) = 246

-- Prove that the area of the rectangle is 3650 m² given the conditions
theorem rectangle_area_3650 (h1 : condition1 L B) (h2 : condition2 L B) : L * B = 3650 := by
  sorry

end rectangle_area_3650_l954_95409


namespace Barbara_spent_46_22_on_different_goods_l954_95464

theorem Barbara_spent_46_22_on_different_goods :
  let tuna_cost := (5 * 2) -- Total cost of tuna
  let water_cost := (4 * 1.5) -- Total cost of water
  let total_before_discount := 56 / 0.9 -- Total before discount, derived from the final amount paid after discount
  let total_tuna_water_cost := 10 + 6 -- Total cost of tuna and water together
  let different_goods_cost := total_before_discount - total_tuna_water_cost
  different_goods_cost = 46.22 := 
sorry

end Barbara_spent_46_22_on_different_goods_l954_95464


namespace village_foods_sales_l954_95404

-- Definitions based on conditions
def customer_count : Nat := 500
def lettuce_per_customer : Nat := 2
def tomato_per_customer : Nat := 4
def price_per_lettuce : Nat := 1
def price_per_tomato : Nat := 1 / 2 -- Note: Handling decimal requires careful type choice

-- Main statement to prove
theorem village_foods_sales : 
  customer_count * (lettuce_per_customer * price_per_lettuce + tomato_per_customer * price_per_tomato) = 2000 := 
by
  sorry

end village_foods_sales_l954_95404


namespace weigh_grain_with_inaccurate_scales_l954_95439

theorem weigh_grain_with_inaccurate_scales
  (inaccurate_scales : ℕ → ℕ → Prop)
  (correct_weight : ℕ)
  (bag_of_grain : ℕ → Prop)
  (balanced : ∀ a b : ℕ, inaccurate_scales a b → a = b := sorry)
  : ∃ grain_weight : ℕ, bag_of_grain grain_weight ∧ grain_weight = correct_weight :=
sorry

end weigh_grain_with_inaccurate_scales_l954_95439


namespace complex_number_quadrant_l954_95426

def i_squared : ℂ := -1

def z (i : ℂ) : ℂ := (-2 + i) * i^5

def in_quadrant_III (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

theorem complex_number_quadrant 
  (i : ℂ) (hi : i^2 = -1) (z_val : z i = (-2 + i) * i^5) :
  in_quadrant_III (z i) :=
sorry

end complex_number_quadrant_l954_95426


namespace line_shift_up_l954_95490

theorem line_shift_up (x y : ℝ) (k : ℝ) (h : y = -2 * x - 4) : 
    y + k = -2 * x - 1 := by
  sorry

end line_shift_up_l954_95490


namespace pupils_in_class_l954_95471

theorem pupils_in_class (n : ℕ) (wrong_entry_increase : n * (1/2) = 13) : n = 26 :=
sorry

end pupils_in_class_l954_95471


namespace B_gain_correct_l954_95420

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def gain_of_B : ℝ :=
  let principal : ℝ := 3150
  let interest_rate_A_to_B : ℝ := 0.08
  let annual_compound : ℕ := 1
  let time_A_to_B : ℝ := 3

  let interest_rate_B_to_C : ℝ := 0.125
  let semiannual_compound : ℕ := 2
  let time_B_to_C : ℝ := 2.5

  let amount_A_to_B := compound_interest principal interest_rate_A_to_B annual_compound time_A_to_B
  let amount_B_to_C := compound_interest principal interest_rate_B_to_C semiannual_compound time_B_to_C

  amount_B_to_C - amount_A_to_B

theorem B_gain_correct : gain_of_B = 282.32 :=
  sorry

end B_gain_correct_l954_95420


namespace solve_inequality_l954_95428

theorem solve_inequality :
  {x : ℝ | 0 ≤ x ∧ x ≤ 1 } = {x : ℝ | x * (x - 1) ≤ 0} :=
by sorry

end solve_inequality_l954_95428


namespace at_least_one_woman_selected_l954_95492

noncomputable def probability_at_least_one_woman_selected (men women : ℕ) (total_selected : ℕ) : ℚ :=
  let total_people := men + women
  let prob_no_woman := (men / total_people) * ((men - 1) / (total_people - 1)) * ((men - 2) / (total_people - 2))
  1 - prob_no_woman

theorem at_least_one_woman_selected (men women : ℕ) (total_selected : ℕ) :
  men = 5 → women = 5 → total_selected = 3 → 
  probability_at_least_one_woman_selected men women total_selected = 11 / 12 := by
  intros hmen hwomen hselected
  rw [hmen, hwomen, hselected]
  unfold probability_at_least_one_woman_selected
  sorry

end at_least_one_woman_selected_l954_95492


namespace smallest_sum_B_c_l954_95454

theorem smallest_sum_B_c : 
  ∃ (B : ℕ) (c : ℕ), (0 ≤ B ∧ B ≤ 4) ∧ (c ≥ 6) ∧ 31 * B = 4 * (c + 1) ∧ B + c = 8 := 
sorry

end smallest_sum_B_c_l954_95454


namespace sin_minus_cos_value_l954_95434

open Real

noncomputable def tan_alpha := sqrt 3
noncomputable def alpha_condition (α : ℝ) := π < α ∧ α < (3 / 2) * π

theorem sin_minus_cos_value (α : ℝ) (h1 : tan α = tan_alpha) (h2 : alpha_condition α) : 
  sin α - cos α = -((sqrt 3) - 1) / 2 := 
by 
  sorry

end sin_minus_cos_value_l954_95434


namespace line_and_circle_condition_l954_95468

theorem line_and_circle_condition (P Q : ℝ × ℝ) (radius : ℝ) 
  (x y m : ℝ) (n : ℝ) (l : ℝ × ℝ → Prop)
  (hPQ : P = (4, -2)) 
  (hPQ' : Q = (-1, 3)) 
  (hC : ∀ (x y : ℝ), (x - 1)^2 + y^2 = radius) 
  (hr : radius < 5) 
  (h_y_segment : ∃ (k : ℝ), |k - 0| = 4 * Real.sqrt 3) 
  : (∀ (x y : ℝ), x + y = 2) ∧ 
    ((∀ (x y : ℝ), l (x, y) ↔ x + y + m = 0 ∨ x + y = 0) 
    ∧ (m = 3 ∨ m = -4) 
    ∧ (∀ A B : ℝ × ℝ, l A → l B → (A.1 - B.1)^2 + (A.2 - B.2)^2 = radius)) := 
  by
  sorry

end line_and_circle_condition_l954_95468


namespace bread_products_wasted_l954_95453

theorem bread_products_wasted :
  (50 * 8 - (20 * 5 + 15 * 4 + 10 * 10 * 1.5)) / 1.5 = 60 := by
  -- The proof steps are omitted here
  sorry

end bread_products_wasted_l954_95453


namespace smallest_n_for_at_least_64_candies_l954_95445

theorem smallest_n_for_at_least_64_candies :
  ∃ n : ℕ, (n > 0) ∧ (n * (n + 1) / 2 ≥ 64) ∧ (∀ m : ℕ, (m > 0) ∧ (m * (m + 1) / 2 ≥ 64) → n ≤ m) := 
sorry

end smallest_n_for_at_least_64_candies_l954_95445


namespace hyperbola_eccentricity_range_l954_95476

theorem hyperbola_eccentricity_range {a b : ℝ} (h₀ : a > 0) (h₁ : b > 0) (h₂ : a > b) :
  ∃ e : ℝ, e = (Real.sqrt (a^2 + b^2)) / a ∧ 1 < e ∧ e < Real.sqrt 2 :=
by
  -- Proof would go here
  sorry

end hyperbola_eccentricity_range_l954_95476


namespace sum_of_squares_of_rates_l954_95493

variable (b j s : ℕ)

theorem sum_of_squares_of_rates
  (h1 : 3 * b + 2 * j + 3 * s = 82)
  (h2 : 5 * b + 3 * j + 2 * s = 99) :
  b^2 + j^2 + s^2 = 314 := by
  sorry

end sum_of_squares_of_rates_l954_95493


namespace compare_log_exp_l954_95480

theorem compare_log_exp (x y z : ℝ) 
  (hx : x = Real.log 2 / Real.log 5) 
  (hy : y = Real.log 2) 
  (hz : z = Real.sqrt 2) : 
  x < y ∧ y < z := 
sorry

end compare_log_exp_l954_95480


namespace increasing_decreasing_intervals_l954_95487

noncomputable def f (x : ℝ) : ℝ := Real.sin (-2 * x + 3 * Real.pi / 4)

theorem increasing_decreasing_intervals : (∀ k : ℤ, 
    ∀ x, 
      ((k : ℝ) * Real.pi + 5 * Real.pi / 8 ≤ x ∧ x ≤ (k : ℝ) * Real.pi + 9 * Real.pi / 8) 
      → 0 < f x ∧ f x < 1) 
  ∧ 
    (∀ k : ℤ, 
    ∀ x, 
      ((k : ℝ) * Real.pi + Real.pi / 8 ≤ x ∧ x ≤ (k : ℝ) * Real.pi + 5 * Real.pi / 8) 
      → -1 < f x ∧ f x < 0) :=
by
  sorry

end increasing_decreasing_intervals_l954_95487


namespace solve_for_y_l954_95441

theorem solve_for_y (x y : ℝ) (h : 2 * y - 4 * x + 5 = 0) : y = 2 * x - 2.5 :=
sorry

end solve_for_y_l954_95441


namespace maximum_value_of_x_minus_y_is_sqrt8_3_l954_95478

variable {x y z : ℝ}

noncomputable def maximum_value_of_x_minus_y (x y z : ℝ) : ℝ :=
  x - y

theorem maximum_value_of_x_minus_y_is_sqrt8_3 (h1 : x + y + z = 2) (h2 : x * y + y * z + z * x = 1) : 
  maximum_value_of_x_minus_y x y z = Real.sqrt (8 / 3) :=
sorry

end maximum_value_of_x_minus_y_is_sqrt8_3_l954_95478


namespace largest_value_among_l954_95431

theorem largest_value_among (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hneq : a ≠ b) :
  max (a + b) (max (2 * Real.sqrt (a * b)) ((a^2 + b^2) / (2 * a * b))) = a + b :=
sorry

end largest_value_among_l954_95431


namespace tan_double_angle_l954_95489

theorem tan_double_angle (α : ℝ) (x y : ℝ) (hxy : y / x = -2) : 
  2 * y / (1 - (y / x)^2) = (4 : ℝ) / 3 :=
by sorry

end tan_double_angle_l954_95489


namespace john_pushups_less_l954_95444

theorem john_pushups_less (zachary david john : ℕ) 
  (h1 : zachary = 19)
  (h2 : david = zachary + 39)
  (h3 : david = 58)
  (h4 : john < david) : 
  david - john = 0 :=
sorry

end john_pushups_less_l954_95444


namespace total_amount_paid_l954_95469

theorem total_amount_paid (cost_lunch : ℝ) (sales_tax_rate : ℝ) (tip_rate : ℝ) (sales_tax : ℝ) (tip : ℝ) 
  (h1 : cost_lunch = 100) 
  (h2 : sales_tax_rate = 0.04) 
  (h3 : tip_rate = 0.06) 
  (h4 : sales_tax = cost_lunch * sales_tax_rate) 
  (h5 : tip = cost_lunch * tip_rate) :
  cost_lunch + sales_tax + tip = 110 :=
by
  sorry

end total_amount_paid_l954_95469


namespace shadow_area_correct_l954_95410

noncomputable def shadow_area (R : ℝ) : ℝ := 3 * Real.pi * R^2

theorem shadow_area_correct (R r d R' : ℝ)
  (h1 : r = (Real.sqrt 3) * R / 2)
  (h2 : d = (3 * R) / 2)
  (h3 : R' = ((3 * R * r) / d)) :
  shadow_area R = Real.pi * R' ^ 2 :=
by
  sorry

end shadow_area_correct_l954_95410


namespace team_E_speed_l954_95419

noncomputable def average_speed_team_E (d t_E t_A v_A v_E : ℝ) : Prop :=
  d = 300 ∧
  t_A = t_E - 3 ∧
  v_A = v_E + 5 ∧
  d = v_E * t_E ∧
  d = v_A * t_A →
  v_E = 20

theorem team_E_speed : ∃ (v_E : ℝ), average_speed_team_E 300 t_E (t_E - 3) (v_E + 5) v_E :=
by
  sorry

end team_E_speed_l954_95419


namespace typing_problem_l954_95413

theorem typing_problem (a b m n : ℕ) (h1 : 60 = a * b) (h2 : 540 = 75 * n) (h3 : n = 3 * m) :
  a = 25 :=
by {
  -- sorry placeholder where the proof would go
  sorry
}

end typing_problem_l954_95413


namespace quadratic_root_value_k_l954_95477

theorem quadratic_root_value_k (k : ℝ) :
  (
    ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = -4 / 3 ∧
    (∀ x : ℝ, x^2 * k - 8 * x - 18 = 0 ↔ (x = x₁ ∨ x = x₂))
  ) → k = 4.5 :=
by
  sorry

end quadratic_root_value_k_l954_95477


namespace minimum_workers_needed_to_make_profit_l954_95421

-- Given conditions
def fixed_maintenance_fee : ℝ := 550
def setup_cost : ℝ := 200
def wage_per_hour : ℝ := 18
def widgets_per_worker_per_hour : ℝ := 6
def sell_price_per_widget : ℝ := 3.5
def work_hours_per_day : ℝ := 8

-- Definitions derived from conditions
def daily_wage_per_worker := wage_per_hour * work_hours_per_day
def daily_revenue_per_worker := widgets_per_worker_per_hour * work_hours_per_day * sell_price_per_widget
def total_daily_cost (n : ℝ) := fixed_maintenance_fee + setup_cost + n * daily_wage_per_worker

-- Prove that the number of workers needed to make a profit is at least 32
theorem minimum_workers_needed_to_make_profit (n : ℕ) (h : (total_daily_cost (n : ℝ)) < n * daily_revenue_per_worker) :
  n ≥ 32 := by
  -- We fill the sorry for proof to pass Lean check
  sorry

end minimum_workers_needed_to_make_profit_l954_95421


namespace minimum_value_8m_n_l954_95423

noncomputable def min_value (m n : ℝ) : ℝ :=
  8 * m + n

theorem minimum_value_8m_n (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : (1 / m) + (8 / n) = 4) : 
  min_value m n = 8 :=
sorry

end minimum_value_8m_n_l954_95423


namespace complete_laps_l954_95401

-- Definitions based on conditions
def total_distance := 3.25  -- total distance Lexi wants to run
def lap_distance := 0.25    -- distance of one lap

-- Proof statement: Total number of complete laps to cover the given distance
theorem complete_laps (h1 : total_distance = 3 + 1/4) (h2 : lap_distance = 1/4) :
  (total_distance / lap_distance) = 13 :=
by 
  sorry

end complete_laps_l954_95401


namespace circle_area_l954_95416

theorem circle_area (r : ℝ) (h : 3 * (1 / (2 * π * r)) = r) : π * r^2 = 3 / 2 :=
by
  -- We leave this place for computations and derivations.
  sorry

end circle_area_l954_95416


namespace book_page_count_l954_95465

def total_pages_in_book (pages_three_nights_ago pages_two_nights_ago pages_last_night pages_tonight total_pages : ℕ) : Prop :=
  pages_three_nights_ago = 15 ∧
  pages_two_nights_ago = 2 * pages_three_nights_ago ∧
  pages_last_night = pages_two_nights_ago + 5 ∧
  pages_tonight = 20 ∧
  total_pages = pages_three_nights_ago + pages_two_nights_ago + pages_last_night + pages_tonight

theorem book_page_count : total_pages_in_book 15 30 35 20 100 :=
by {
  sorry
}

end book_page_count_l954_95465


namespace pages_revised_once_l954_95411

-- Definitions
def total_pages : ℕ := 200
def pages_revised_twice : ℕ := 20
def total_cost : ℕ := 1360
def cost_first_time : ℕ := 5
def cost_revision : ℕ := 3

theorem pages_revised_once (x : ℕ) (h1 : total_cost = 1000 + 3 * x + 120) : x = 80 := by
  sorry

end pages_revised_once_l954_95411


namespace B_pow_2048_l954_95438

open Real Matrix

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![cos (π / 4), 0, -sin (π / 4)],
    ![0, 1, 0],
    ![sin (π / 4), 0, cos (π / 4)]]

theorem B_pow_2048 :
  B ^ 2048 = (1 : Matrix (Fin 3) (Fin 3) ℝ) :=
by
  sorry

end B_pow_2048_l954_95438


namespace derivative_of_y_l954_95435

noncomputable def y (x : ℝ) : ℝ :=
  -1/4 * Real.arcsin ((5 + 3 * Real.cosh x) / (3 + 5 * Real.cosh x))

theorem derivative_of_y (x : ℝ) :
  deriv y x = 1 / (3 + 5 * Real.cosh x) :=
sorry

end derivative_of_y_l954_95435


namespace single_elimination_games_l954_95462

theorem single_elimination_games (n : ℕ) (h : n = 23) : 
  ∃ games : ℕ, games = n - 1 :=
by
  use 22
  sorry

end single_elimination_games_l954_95462


namespace eggs_left_l954_95425

theorem eggs_left (x : ℕ) : (47 - 5 - x) = (42 - x) :=
  by
  sorry

end eggs_left_l954_95425


namespace total_cost_is_17_l954_95440

def taco_shells_cost : ℝ := 5
def bell_pepper_cost_per_unit : ℝ := 1.5
def bell_pepper_quantity : ℕ := 4
def meat_cost_per_pound : ℝ := 3
def meat_quantity : ℕ := 2

def total_spent : ℝ :=
  taco_shells_cost + (bell_pepper_cost_per_unit * bell_pepper_quantity) + (meat_cost_per_pound * meat_quantity)

theorem total_cost_is_17 : total_spent = 17 := 
  by sorry

end total_cost_is_17_l954_95440


namespace complex_square_l954_95461

theorem complex_square (z : ℂ) (i : ℂ) (h1 : z = 2 - 3 * i) (h2 : i^2 = -1) : z^2 = -5 - 12 * i :=
sorry

end complex_square_l954_95461


namespace geom_seq_sum_of_terms_l954_95442

theorem geom_seq_sum_of_terms
  (a : ℕ → ℝ) (q : ℝ) (n : ℕ)
  (h_geometric: ∀ n, a (n + 1) = a n * q)
  (h_q : q = 2)
  (h_sum : a 0 + a 1 + a 2 = 21)
  (h_pos : ∀ n, a n > 0) :
  a 2 + a 3 + a 4 = 84 :=
by
  sorry

end geom_seq_sum_of_terms_l954_95442


namespace line_intersects_parabola_at_vertex_l954_95405

theorem line_intersects_parabola_at_vertex :
  ∃ (a : ℝ), (∀ x : ℝ, -x + a = x^2 + a^2) ↔ a = 0 ∨ a = 1 :=
by
  sorry

end line_intersects_parabola_at_vertex_l954_95405


namespace min_ratio_of_integers_l954_95436

theorem min_ratio_of_integers (x y : ℕ) (hx : 50 < x) (hy : 50 < y) (h_mean : x + y = 130) : 
  x = 51 → y = 79 → x / y = 51 / 79 := by
  sorry

end min_ratio_of_integers_l954_95436


namespace future_years_l954_95400

theorem future_years (P A F : ℝ) (Y : ℝ) 
  (h1 : P = 50)
  (h2 : P = 1.25 * A)
  (h3 : P = 5 / 6 * F)
  (h4 : A + 10 + Y = F) : 
  Y = 10 := sorry

end future_years_l954_95400


namespace cos_sum_equals_fraction_sqrt_13_minus_1_div_4_l954_95450

noncomputable def cos_sum : ℝ :=
  (Real.cos (2 * Real.pi / 17) +
   Real.cos (6 * Real.pi / 17) +
   Real.cos (8 * Real.pi / 17))

theorem cos_sum_equals_fraction_sqrt_13_minus_1_div_4 :
  cos_sum = (Real.sqrt 13 - 1) / 4 := 
sorry

end cos_sum_equals_fraction_sqrt_13_minus_1_div_4_l954_95450


namespace algebraic_expression_value_l954_95424

noncomputable def a : ℝ := Real.sqrt 6 + 1
noncomputable def b : ℝ := Real.sqrt 6 - 1

theorem algebraic_expression_value :
  a^2 + a * b = 12 + 2 * Real.sqrt 6 :=
sorry

end algebraic_expression_value_l954_95424


namespace min_length_intersection_l954_95437

theorem min_length_intersection
  (m n : ℝ)
  (hM0 : 0 ≤ m)
  (hM1 : m + 3/4 ≤ 1)
  (hN0 : n - 1/3 ≥ 0)
  (hN1 : n ≤ 1) :
  ∃ x, 0 ≤ x ∧ x ≤ 1 ∧
  x = ((m + 3/4) + (n - 1/3)) - 1 :=
sorry

end min_length_intersection_l954_95437


namespace b3_b8_product_l954_95456

-- Definitions based on conditions
def is_arithmetic_seq (b : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

-- The problem statement
theorem b3_b8_product (b : ℕ → ℤ) (h_seq : is_arithmetic_seq b) (h4_7 : b 4 * b 7 = 24) : 
  b 3 * b 8 = 200 / 9 :=
sorry

end b3_b8_product_l954_95456


namespace class_sizes_l954_95427

theorem class_sizes
  (finley_students : ℕ)
  (johnson_students : ℕ)
  (garcia_students : ℕ)
  (smith_students : ℕ)
  (h1 : finley_students = 24)
  (h2 : johnson_students = 10 + finley_students / 2)
  (h3 : garcia_students = 2 * johnson_students)
  (h4 : smith_students = finley_students / 3) :
  finley_students = 24 ∧ johnson_students = 22 ∧ garcia_students = 44 ∧ smith_students = 8 :=
by
  sorry

end class_sizes_l954_95427


namespace train_speed_in_km_hr_l954_95422

noncomputable def train_length : ℝ := 110
noncomputable def bridge_length : ℝ := 132
noncomputable def crossing_time : ℝ := 9.679225661947045
noncomputable def distance_covered : ℝ := train_length + bridge_length
noncomputable def speed_m_s : ℝ := distance_covered / crossing_time
noncomputable def speed_km_hr : ℝ := speed_m_s * 3.6

theorem train_speed_in_km_hr : speed_km_hr = 90.0216 := by
  sorry

end train_speed_in_km_hr_l954_95422


namespace find_sixth_term_l954_95496

noncomputable def first_term : ℝ := Real.sqrt 3
noncomputable def fifth_term : ℝ := Real.sqrt 243
noncomputable def common_ratio (q : ℝ) : Prop := fifth_term = first_term * q^4
noncomputable def sixth_term (b6 : ℝ) (q : ℝ) : Prop := b6 = fifth_term * q

theorem find_sixth_term (q : ℝ) (b6 : ℝ) : 
  first_term = Real.sqrt 3 ∧
  fifth_term = Real.sqrt 243 ∧
  common_ratio q ∧ 
  sixth_term b6 q → 
  b6 = 27 ∨ b6 = -27 := 
by
  intros
  sorry

end find_sixth_term_l954_95496


namespace contrapositive_proposition_l954_95495

theorem contrapositive_proposition (x y : ℝ) :
  (¬ (x = 0 ∧ y = 0)) → (x^2 + y^2 ≠ 0) :=
sorry

end contrapositive_proposition_l954_95495


namespace unique_solution_range_l954_95402
-- import relevant libraries

-- define the functions
def f (a x : ℝ) : ℝ := 2 * a * x ^ 3 + 3
def g (x : ℝ) : ℝ := 3 * x ^ 2 + 2

-- state and prove the main theorem (statement only)
theorem unique_solution_range (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ f a x = g x ∧ ∀ y : ℝ, y > 0 → f a y = g y → y = x) ↔ a ∈ Set.Iio (-1) :=
sorry

end unique_solution_range_l954_95402


namespace cos_of_angle_sum_l954_95470

variable (θ : ℝ)

-- Given condition
axiom sin_theta : Real.sin θ = 1 / 4

-- To prove
theorem cos_of_angle_sum : Real.cos (3 * Real.pi / 2 + θ) = -1 / 4 :=
by
  sorry

end cos_of_angle_sum_l954_95470


namespace scientific_notation_of_170000_l954_95466

-- Define the concept of scientific notation
def is_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  (1 ≤ a) ∧ (a < 10) ∧ (x = a * 10^n)

-- The main statement to prove
theorem scientific_notation_of_170000 : is_scientific_notation 1.7 5 170000 :=
by sorry

end scientific_notation_of_170000_l954_95466


namespace comic_books_stacking_order_l954_95479

-- Definitions of the conditions
def num_spiderman_books : ℕ := 6
def num_archie_books : ℕ := 5
def num_garfield_books : ℕ := 4

-- Calculations of factorials
def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

-- Grouping and order calculation
def ways_to_arrange_group_books : ℕ :=
  factorial num_spiderman_books *
  factorial num_archie_books *
  factorial num_garfield_books

def num_groups : ℕ := 3

def ways_to_arrange_groups : ℕ :=
  factorial num_groups

def total_ways_to_stack_books : ℕ :=
  ways_to_arrange_group_books * ways_to_arrange_groups

-- Theorem stating the total number of different orders
theorem comic_books_stacking_order :
  total_ways_to_stack_books = 12441600 :=
by
  sorry

end comic_books_stacking_order_l954_95479


namespace tan_alpha_eq_one_then_expr_value_l954_95484

theorem tan_alpha_eq_one_then_expr_value (α : ℝ) (h : Real.tan α = 1) :
  1 / (Real.cos α ^ 2 + Real.sin (2 * α)) = 2 / 3 :=
by
  sorry

end tan_alpha_eq_one_then_expr_value_l954_95484


namespace perfect_square_representation_l954_95459

theorem perfect_square_representation :
  29 - 12*Real.sqrt 5 = (2*Real.sqrt 5 - 3*Real.sqrt 5 / 5)^2 :=
sorry

end perfect_square_representation_l954_95459


namespace solve_sqrt_eq_l954_95499

theorem solve_sqrt_eq (x : ℝ) :
  (Real.sqrt ((3 + 2 * Real.sqrt 2)^x) + Real.sqrt ((3 - 2 * Real.sqrt 2)^x) = 5) ↔ (x = 2 ∨ x = -2) := by
  sorry

end solve_sqrt_eq_l954_95499


namespace max_interval_length_l954_95463

def m (x : ℝ) : ℝ := x^2 - 3 * x + 4
def n (x : ℝ) : ℝ := 2 * x - 3

def are_close_functions (m n : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |m x - n x| ≤ 1

theorem max_interval_length
  (h : are_close_functions m n 2 3) :
  3 - 2 = 1 :=
sorry

end max_interval_length_l954_95463


namespace seeds_germination_l954_95458

theorem seeds_germination (seed_plot1 seed_plot2 : ℕ) (germ_rate2 total_germ_rate : ℝ) (germinated_total_pct : ℝ)
  (h1 : seed_plot1 = 300)
  (h2 : seed_plot2 = 200)
  (h3 : germ_rate2 = 0.35)
  (h4 : germinated_total_pct = 28.999999999999996 / 100) :
  (germinated_total_pct * (seed_plot1 + seed_plot2) - germ_rate2 * seed_plot2) / seed_plot1 * 100 = 25 :=
by sorry  -- Proof not required

end seeds_germination_l954_95458


namespace second_year_growth_rate_l954_95457

variable (initial_investment : ℝ) (first_year_growth : ℝ) (additional_investment : ℝ) (final_value : ℝ) (second_year_growth : ℝ)

def calculate_portfolio_value_after_first_year (initial_investment first_year_growth : ℝ) : ℝ :=
  initial_investment * (1 + first_year_growth)

def calculate_new_value_after_addition (value_after_first_year additional_investment : ℝ) : ℝ :=
  value_after_first_year + additional_investment

def calculate_final_value_after_second_year (new_value second_year_growth : ℝ) : ℝ :=
  new_value * (1 + second_year_growth)

theorem second_year_growth_rate 
  (h1 : initial_investment = 80) 
  (h2 : first_year_growth = 0.15) 
  (h3 : additional_investment = 28) 
  (h4 : final_value = 132) : 
  calculate_final_value_after_second_year
    (calculate_new_value_after_addition
      (calculate_portfolio_value_after_first_year initial_investment first_year_growth)
      additional_investment)
    0.1 = final_value := 
  by
  sorry

end second_year_growth_rate_l954_95457


namespace maximize_total_profit_maximize_average_annual_profit_l954_95447

-- Define the profit function
def total_profit (x : ℤ) : ℤ := -x^2 + 18*x - 36

-- Define the average annual profit function
def average_annual_profit (x : ℤ) : ℤ :=
  let y := total_profit x
  y / x

-- Prove the maximum total profit
theorem maximize_total_profit : 
  ∃ x : ℤ, (total_profit x = 45) ∧ (x = 9) := 
  sorry

-- Prove the maximum average annual profit
theorem maximize_average_annual_profit : 
  ∃ x : ℤ, (average_annual_profit x = 6) ∧ (x = 6) :=
  sorry

end maximize_total_profit_maximize_average_annual_profit_l954_95447


namespace ratio_of_groups_l954_95432

variable (x : ℚ)

-- The total number of people in the calligraphy group
def calligraphy_group (x : ℚ) := x + (2 / 7) * x

-- The total number of people in the recitation group
def recitation_group (x : ℚ) := x + (1 / 5) * x

theorem ratio_of_groups (x : ℚ) (hx : x ≠ 0) : 
    (calligraphy_group x) / (recitation_group x) = (3 : ℚ) / (4 : ℚ) := by
  sorry

end ratio_of_groups_l954_95432


namespace number_of_buildings_l954_95448

theorem number_of_buildings (studio_apartments : ℕ) (two_person_apartments : ℕ) (four_person_apartments : ℕ)
    (occupancy_percentage : ℝ) (current_occupancy : ℕ)
    (max_occupancy_building : ℕ) (max_occupancy_complex : ℕ) (num_buildings : ℕ)
    (h_studio : studio_apartments = 10)
    (h_two_person : two_person_apartments = 20)
    (h_four_person : four_person_apartments = 5)
    (h_occupancy_percentage : occupancy_percentage = 0.75)
    (h_current_occupancy : current_occupancy = 210)
    (h_max_occupancy_building : max_occupancy_building = 10 * 1 + 20 * 2 + 5 * 4)
    (h_max_occupancy_complex : max_occupancy_complex = current_occupancy / occupancy_percentage)
    (h_num_buildings : num_buildings = max_occupancy_complex / max_occupancy_building) :
    num_buildings = 4 :=
by
  sorry

end number_of_buildings_l954_95448


namespace cannot_form_right_triangle_l954_95498

theorem cannot_form_right_triangle (a b c : ℝ) (h₁ : a = 2) (h₂ : b = 2) (h₃ : c = 3) :
  a^2 + b^2 ≠ c^2 :=
by
  rw [h₁, h₂, h₃]
  -- Next step would be to simplify and show the inequality, but we skip the proof
  -- 2^2 + 2^2 = 4 + 4 = 8 
  -- 3^2 = 9 
  -- 8 ≠ 9
  sorry

end cannot_form_right_triangle_l954_95498


namespace product_mod_m_l954_95408

-- Define the constants
def a : ℕ := 2345
def b : ℕ := 1554
def m : ℕ := 700

-- Definitions derived from the conditions
def a_mod_m : ℕ := a % m
def b_mod_m : ℕ := b % m

-- The proof problem
theorem product_mod_m (a b m : ℕ) (h1 : a % m = 245) (h2 : b % m = 154) :
  (a * b) % m = 630 := by sorry

end product_mod_m_l954_95408


namespace tan_beta_tan_alpha_eq_m_minus_n_over_m_plus_n_l954_95451

/-- Given the trigonometric identity and the ratio, we want to prove the relationship between the tangents of the angles. -/
theorem tan_beta_tan_alpha_eq_m_minus_n_over_m_plus_n
  (α β m n : ℝ)
  (h : (Real.sin (α + β)) / (Real.sin (α - β)) = m / n) :
  (Real.tan β) / (Real.tan α) = (m - n) / (m + n) :=
  sorry

end tan_beta_tan_alpha_eq_m_minus_n_over_m_plus_n_l954_95451


namespace max_value_neg7s_squared_plus_56s_plus_20_l954_95467

theorem max_value_neg7s_squared_plus_56s_plus_20 :
  ∃ s : ℝ, s = 4 ∧ ∀ t : ℝ, -7 * t^2 + 56 * t + 20 ≤ 132 := 
by
  sorry

end max_value_neg7s_squared_plus_56s_plus_20_l954_95467


namespace greatest_integer_gcd_l954_95486

theorem greatest_integer_gcd (n : ℕ) (h1 : n < 200) (h2 : gcd n 18 = 6) : n = 192 :=
sorry

end greatest_integer_gcd_l954_95486


namespace sin_sum_leq_3_sqrt3_over_2_l954_95494

theorem sin_sum_leq_3_sqrt3_over_2 
  (A B C : ℝ) 
  (h₁ : A + B + C = Real.pi) 
  (h₂ : 0 < A ∧ A < Real.pi)
  (h₃ : 0 < B ∧ B < Real.pi)
  (h₄ : 0 < C ∧ C < Real.pi) :
  Real.sin A + Real.sin B + Real.sin C ≤ 3 * Real.sqrt 3 / 2 :=
sorry

end sin_sum_leq_3_sqrt3_over_2_l954_95494


namespace evaluate_complex_power_expression_l954_95429

theorem evaluate_complex_power_expression : (i : ℂ)^23 + ((i : ℂ)^105 * (i : ℂ)^17) = -i - 1 := by
  sorry

end evaluate_complex_power_expression_l954_95429


namespace Lagrange_interpol_equiv_x_squared_l954_95449

theorem Lagrange_interpol_equiv_x_squared (a b c x : ℝ)
    (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
    c^2 * ((x - a) * (x - b)) / ((c - a) * (c - b)) +
    b^2 * ((x - a) * (x - c)) / ((b - a) * (b - c)) +
    a^2 * ((x - b) * (x - c)) / ((a - b) * (a - c)) = x^2 := 
    sorry

end Lagrange_interpol_equiv_x_squared_l954_95449
