import Mathlib

namespace NUMINAMATH_GPT_product_ABC_sol_l1642_164232

theorem product_ABC_sol (A B C : ℚ) : 
  (∀ x : ℚ, x^2 - 20 = A * (x + 2) * (x - 3) + B * (x - 2) * (x - 3) + C * (x - 2) * (x + 2)) → 
  A * B * C = 2816 / 35 := 
by 
  intro h
  sorry

end NUMINAMATH_GPT_product_ABC_sol_l1642_164232


namespace NUMINAMATH_GPT_money_inequalities_l1642_164207

theorem money_inequalities (a b : ℝ) (h₁ : 5 * a + b > 51) (h₂ : 3 * a - b = 21) : a > 9 ∧ b > 6 := 
by
  sorry

end NUMINAMATH_GPT_money_inequalities_l1642_164207


namespace NUMINAMATH_GPT_at_least_one_genuine_l1642_164228

theorem at_least_one_genuine (batch : Finset ℕ) 
  (h_batch_size : batch.card = 12) 
  (genuine_items : Finset ℕ)
  (h_genuine_size : genuine_items.card = 10)
  (defective_items : Finset ℕ)
  (h_defective_size : defective_items.card = 2)
  (h_disjoint : genuine_items ∩ defective_items = ∅)
  (drawn_items : Finset ℕ)
  (h_draw_size : drawn_items.card = 3)
  (h_subset : drawn_items ⊆ batch)
  (h_union : genuine_items ∪ defective_items = batch) :
  (∃ (x : ℕ), x ∈ drawn_items ∧ x ∈ genuine_items) :=
sorry

end NUMINAMATH_GPT_at_least_one_genuine_l1642_164228


namespace NUMINAMATH_GPT_always_exists_triangle_l1642_164203

variable (a1 a2 a3 a4 d : ℕ)

def arithmetic_sequence (a1 a2 a3 a4 d : ℕ) :=
  a2 = a1 + d ∧ a3 = a1 + 2 * d ∧ a4 = a1 + 3 * d

def positive_terms (a1 a2 a3 a4 : ℕ) :=
  a1 > 0 ∧ a2 > 0 ∧ a3 > 0 ∧ a4 > 0

theorem always_exists_triangle (a1 a2 a3 a4 d : ℕ)
  (h1 : arithmetic_sequence a1 a2 a3 a4 d)
  (h2 : d > 0)
  (h3 : positive_terms a1 a2 a3 a4) :
  a2 + a3 > a4 ∧ a2 + a4 > a3 ∧ a3 + a4 > a2 :=
sorry

end NUMINAMATH_GPT_always_exists_triangle_l1642_164203


namespace NUMINAMATH_GPT_negation_proposition_l1642_164251

theorem negation_proposition : (¬ ∀ x : ℝ, (1 < x) → x - 1 ≥ Real.log x) ↔ (∃ x_0 : ℝ, (1 < x_0) ∧ x_0 - 1 < Real.log x_0) :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_l1642_164251


namespace NUMINAMATH_GPT_total_revenue_is_correct_l1642_164230

def category_a_price : ℝ := 65
def category_b_price : ℝ := 45
def category_c_price : ℝ := 25

def category_a_discounted_price : ℝ := category_a_price - 0.55 * category_a_price
def category_b_discounted_price : ℝ := category_b_price - 0.35 * category_b_price
def category_c_discounted_price : ℝ := category_c_price - 0.20 * category_c_price

def category_a_full_price_quantity : ℕ := 100
def category_b_full_price_quantity : ℕ := 50
def category_c_full_price_quantity : ℕ := 60

def category_a_discounted_quantity : ℕ := 20
def category_b_discounted_quantity : ℕ := 30
def category_c_discounted_quantity : ℕ := 40

def revenue_from_category_a : ℝ :=
  category_a_discounted_quantity * category_a_discounted_price +
  category_a_full_price_quantity * category_a_price

def revenue_from_category_b : ℝ :=
  category_b_discounted_quantity * category_b_discounted_price +
  category_b_full_price_quantity * category_b_price

def revenue_from_category_c : ℝ :=
  category_c_discounted_quantity * category_c_discounted_price +
  category_c_full_price_quantity * category_c_price

def total_revenue : ℝ :=
  revenue_from_category_a + revenue_from_category_b + revenue_from_category_c

theorem total_revenue_is_correct :
  total_revenue = 12512.50 :=
by
  unfold total_revenue
  unfold revenue_from_category_a
  unfold revenue_from_category_b
  unfold revenue_from_category_c
  unfold category_a_discounted_price
  unfold category_b_discounted_price
  unfold category_c_discounted_price
  sorry

end NUMINAMATH_GPT_total_revenue_is_correct_l1642_164230


namespace NUMINAMATH_GPT_garden_fencing_cost_l1642_164219

theorem garden_fencing_cost (x y : ℝ) (h1 : x^2 + y^2 = 900) (h2 : x * y = 200)
    (cost_per_meter : ℝ) (h3 : cost_per_meter = 15) : 
    cost_per_meter * (2 * x + y) = 300 * Real.sqrt 7 + 150 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_garden_fencing_cost_l1642_164219


namespace NUMINAMATH_GPT_boys_in_class_l1642_164279

theorem boys_in_class (r : ℕ) (g b : ℕ) (h1 : g/b = 4/3) (h2 : g + b = 35) : b = 15 :=
  sorry

end NUMINAMATH_GPT_boys_in_class_l1642_164279


namespace NUMINAMATH_GPT_parabola_equation_l1642_164267

theorem parabola_equation (P : ℝ × ℝ) :
  let d1 := dist P (-3, 0)
  let d2 := abs (P.1 - 2)
  (d1 = d2 + 1 ↔ P.2^2 = -12 * P.1) :=
by
  intro d1 d2
  sorry

end NUMINAMATH_GPT_parabola_equation_l1642_164267


namespace NUMINAMATH_GPT_find_y_l1642_164256

theorem find_y :
  (∃ y : ℝ, (4 * Real.arctan (1/5) + Real.arctan (1/25) + Real.arctan (1/y) = π/4) ∧ y = 1251) :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1642_164256


namespace NUMINAMATH_GPT_lilies_per_centerpiece_l1642_164289

theorem lilies_per_centerpiece (centerpieces roses orchids cost total_budget price_per_flower number_of_lilies_per_centerpiece : ℕ) 
  (h0 : centerpieces = 6)
  (h1 : roses = 8)
  (h2 : orchids = 2 * roses)
  (h3 : cost = total_budget)
  (h4 : total_budget = 2700)
  (h5 : price_per_flower = 15)
  (h6 : cost = (centerpieces * roses * price_per_flower) + (centerpieces * orchids * price_per_flower) + (centerpieces * number_of_lilies_per_centerpiece * price_per_flower))
  : number_of_lilies_per_centerpiece = 6 := 
by 
  sorry

end NUMINAMATH_GPT_lilies_per_centerpiece_l1642_164289


namespace NUMINAMATH_GPT_find_n_l1642_164275

theorem find_n (m n : ℝ) (h1 : m + 2 * n = 1.2) (h2 : 0.1 + m + n + 0.1 = 1) : n = 0.4 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l1642_164275


namespace NUMINAMATH_GPT_sum_max_min_ratio_l1642_164292

def ellipse_eq (x y : ℝ) : Prop :=
  5 * x^2 + x * y + 4 * y^2 - 15 * x - 24 * y + 56 = 0

theorem sum_max_min_ratio (p q : ℝ) (x y : ℝ) :
  (∀ x y : ℝ, ellipse_eq x y → y / x = p ∨ y / x = q) → 
  p + q = 31 / 34 :=
by
  sorry

end NUMINAMATH_GPT_sum_max_min_ratio_l1642_164292


namespace NUMINAMATH_GPT_root_of_equation_l1642_164250

theorem root_of_equation (x : ℝ) :
  (∃ u : ℝ, u = Real.sqrt (x + 15) ∧ u - 7 / u = 6) → x = 34 :=
by
  sorry

end NUMINAMATH_GPT_root_of_equation_l1642_164250


namespace NUMINAMATH_GPT_average_prime_numbers_l1642_164291

-- Definitions of the visible numbers.
def visible1 : ℕ := 51
def visible2 : ℕ := 72
def visible3 : ℕ := 43

-- Definitions of the hidden numbers as prime numbers.
def hidden1 : ℕ := 2
def hidden2 : ℕ := 23
def hidden3 : ℕ := 31

-- Common sum of the numbers on each card.
def common_sum : ℕ := 74

-- Establishing the conditions given in the problem.
def condition1 : hidden1 + visible2 = common_sum := by sorry
def condition2 : hidden2 + visible1 = common_sum := by sorry
def condition3 : hidden3 + visible3 = common_sum := by sorry

-- Calculate the average of the hidden prime numbers.
def average_hidden_primes : ℚ := (hidden1 + hidden2 + hidden3) / 3

-- The proof statement that the average of the hidden prime numbers is 56/3.
theorem average_prime_numbers : average_hidden_primes = 56 / 3 := by
  sorry

end NUMINAMATH_GPT_average_prime_numbers_l1642_164291


namespace NUMINAMATH_GPT_correct_statement_l1642_164263

/-- Given the following statements:
 1. Seeing a rainbow after rain is a random event.
 2. To check the various equipment before a plane takes off, a random sampling survey should be conducted.
 3. When flipping a coin 20 times, it will definitely land heads up 10 times.
 4. The average precipitation in two cities, A and B, over the past 5 years is 800 millimeters. The variances are 3.4 for city A and 4.3 for city B. The city with the most stable annual precipitation is city B.

 Prove that the correct statement is: Seeing a rainbow after rain is a random event.
-/
theorem correct_statement : 
  let statement_A := "Seeing a rainbow after rain is a random event"
  let statement_B := "To check the various equipment before a plane takes off, a random sampling survey should be conducted"
  let statement_C := "When flipping a coin 20 times, it will definitely land heads up 10 times"
  let statement_D := "The average precipitation in two cities, A and B, over the past 5 years is 800 millimeters. The variances are 3.4 for city A and 4.3 for city B. The city with the most stable annual precipitation is city B"
  statement_A = "Seeing a rainbow after rain is a random event" := by
sorry

end NUMINAMATH_GPT_correct_statement_l1642_164263


namespace NUMINAMATH_GPT_discriminant_zero_geometric_progression_l1642_164274

variable (a b c : ℝ)

theorem discriminant_zero_geometric_progression
  (h : b^2 = 4 * a * c) : (b / (2 * a)) = (2 * c / b) :=
by
  sorry

end NUMINAMATH_GPT_discriminant_zero_geometric_progression_l1642_164274


namespace NUMINAMATH_GPT_distinct_solutions_diff_l1642_164220

theorem distinct_solutions_diff {r s : ℝ} (h_eq_r : (6 * r - 18) / (r^2 + 3 * r - 18) = r + 3)
  (h_eq_s : (6 * s - 18) / (s^2 + 3 * s - 18) = s + 3)
  (h_distinct : r ≠ s) (h_r_gt_s : r > s) : r - s = 11 := 
sorry

end NUMINAMATH_GPT_distinct_solutions_diff_l1642_164220


namespace NUMINAMATH_GPT_darry_full_ladder_climbs_l1642_164281

-- Definitions and conditions
def full_ladder_steps : ℕ := 11
def smaller_ladder_steps : ℕ := 6
def smaller_ladder_climbs : ℕ := 7
def total_steps_climbed_today : ℕ := 152

-- Question: How many times did Darry climb his full ladder?
theorem darry_full_ladder_climbs (x : ℕ) 
  (H : 11 * x + smaller_ladder_steps * 7 = total_steps_climbed_today) : 
  x = 10 := by
  -- proof steps omitted, so we write
  sorry

end NUMINAMATH_GPT_darry_full_ladder_climbs_l1642_164281


namespace NUMINAMATH_GPT_average_length_l1642_164270

def length1 : ℕ := 2
def length2 : ℕ := 3
def length3 : ℕ := 7

theorem average_length : (length1 + length2 + length3) / 3 = 4 :=
by
  sorry

end NUMINAMATH_GPT_average_length_l1642_164270


namespace NUMINAMATH_GPT_sheets_of_paper_per_week_l1642_164254

theorem sheets_of_paper_per_week
  (sheets_per_class_per_day : ℕ)
  (num_classes : ℕ)
  (school_days_per_week : ℕ)
  (total_sheets_per_week : ℕ) 
  (h1 : sheets_per_class_per_day = 200)
  (h2 : num_classes = 9)
  (h3 : school_days_per_week = 5)
  (h4 : total_sheets_per_week = sheets_per_class_per_day * num_classes * school_days_per_week) :
  total_sheets_per_week = 9000 :=
sorry

end NUMINAMATH_GPT_sheets_of_paper_per_week_l1642_164254


namespace NUMINAMATH_GPT_at_least_2020_distinct_n_l1642_164255

theorem at_least_2020_distinct_n : 
  ∃ (N : Nat), N ≥ 2020 ∧ ∃ (a : Fin N → ℕ), 
  Function.Injective a ∧ ∀ i, ∃ k : ℚ, (a i : ℚ) + 0.25 = (k + 1/2)^2 := 
sorry

end NUMINAMATH_GPT_at_least_2020_distinct_n_l1642_164255


namespace NUMINAMATH_GPT_coins_distribution_l1642_164242

theorem coins_distribution :
  ∃ (x y z : ℕ), x + y + z = 1000 ∧ x + 2 * y + 5 * z = 2000 ∧ Nat.Prime x ∧ x = 3 ∧ y = 996 ∧ z = 1 :=
by
  sorry

end NUMINAMATH_GPT_coins_distribution_l1642_164242


namespace NUMINAMATH_GPT_midpoint_C_is_either_l1642_164259

def A : ℝ := -7
def dist_AB : ℝ := 5

theorem midpoint_C_is_either (C : ℝ) (h : C = (A + (A + dist_AB / 2)) / 2 ∨ C = (A + (A - dist_AB / 2)) / 2) : 
  C = -9 / 2 ∨ C = -19 / 2 := 
sorry

end NUMINAMATH_GPT_midpoint_C_is_either_l1642_164259


namespace NUMINAMATH_GPT_seven_divides_n_iff_seven_divides_q_minus_2r_seven_divides_2023_thirteen_divides_n_iff_thirteen_divides_q_plus_4r_l1642_164296

-- Problem 1
theorem seven_divides_n_iff_seven_divides_q_minus_2r (n q r : ℕ) (h : n = 10 * q + r) :
  (7 ∣ n) ↔ (7 ∣ (q - 2 * r)) := sorry

-- Problem 2
theorem seven_divides_2023 : 7 ∣ 2023 :=
  let q := 202
  let r := 3
  have h : 2023 = 10 * q + r := by norm_num
  have h1 : (7 ∣ 2023) ↔ (7 ∣ (q - 2 * r)) :=
    seven_divides_n_iff_seven_divides_q_minus_2r 2023 q r h
  sorry -- Here you would use h1 and prove the statement using it

-- Problem 3
theorem thirteen_divides_n_iff_thirteen_divides_q_plus_4r (n q r : ℕ) (h : n = 10 * q + r) :
  (13 ∣ n) ↔ (13 ∣ (q + 4 * r)) := sorry

end NUMINAMATH_GPT_seven_divides_n_iff_seven_divides_q_minus_2r_seven_divides_2023_thirteen_divides_n_iff_thirteen_divides_q_plus_4r_l1642_164296


namespace NUMINAMATH_GPT_apartments_in_each_complex_l1642_164225

variable {A : ℕ}

theorem apartments_in_each_complex
    (h1 : ∀ (locks_per_apartment : ℕ), locks_per_apartment = 3)
    (h2 : ∀ (num_complexes : ℕ), num_complexes = 2)
    (h3 : 3 * 2 * A = 72) :
    A = 12 :=
by
  sorry

end NUMINAMATH_GPT_apartments_in_each_complex_l1642_164225


namespace NUMINAMATH_GPT_operation_positive_l1642_164288

theorem operation_positive (op : ℤ → ℤ → ℤ) (is_pos : op 1 (-2) > 0) : op = Int.sub :=
by
  sorry

end NUMINAMATH_GPT_operation_positive_l1642_164288


namespace NUMINAMATH_GPT_no_non_congruent_right_triangles_l1642_164209

theorem no_non_congruent_right_triangles (a b : ℝ) (c : ℝ) (h_right_triangle : c = Real.sqrt (a^2 + b^2)) (h_perimeter : a + b + Real.sqrt (a^2 + b^2) = 2 * Real.sqrt (a^2 + b^2)) : a = 0 ∨ b = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_non_congruent_right_triangles_l1642_164209


namespace NUMINAMATH_GPT_t_mobile_additional_line_cost_l1642_164236

variable (T : ℕ)

def t_mobile_cost (n : ℕ) : ℕ :=
  if n ≤ 2 then 50 else 50 + (n - 2) * T

def m_mobile_cost (n : ℕ) : ℕ :=
  if n ≤ 2 then 45 else 45 + (n - 2) * 14

theorem t_mobile_additional_line_cost
  (h : t_mobile_cost 5 = m_mobile_cost 5 + 11) :
  T = 16 :=
by
  sorry

end NUMINAMATH_GPT_t_mobile_additional_line_cost_l1642_164236


namespace NUMINAMATH_GPT_more_triangles_with_perimeter_2003_than_2000_l1642_164261

theorem more_triangles_with_perimeter_2003_than_2000 :
  (∃ (count_2003 count_2000 : ℕ), 
   count_2003 > count_2000 ∧ 
   (∀ (a b c : ℕ), a + b + c = 2000 → a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a) ∧ 
   (∀ (a b c : ℕ), a + b + c = 2003 → a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a))
  := 
sorry

end NUMINAMATH_GPT_more_triangles_with_perimeter_2003_than_2000_l1642_164261


namespace NUMINAMATH_GPT_year_with_greatest_temp_increase_l1642_164239

def avg_temp (year : ℕ) : ℝ :=
  match year with
  | 2000 => 2.0
  | 2001 => 2.3
  | 2002 => 2.5
  | 2003 => 2.7
  | 2004 => 3.9
  | 2005 => 4.1
  | 2006 => 4.2
  | 2007 => 4.4
  | 2008 => 3.9
  | 2009 => 3.1
  | _    => 0.0

theorem year_with_greatest_temp_increase : ∃ year, year = 2004 ∧
  (∀ y, 2000 < y ∧ y ≤ 2009 → avg_temp y - avg_temp (y - 1) ≤ avg_temp 2004 - avg_temp 2003) := by
  sorry

end NUMINAMATH_GPT_year_with_greatest_temp_increase_l1642_164239


namespace NUMINAMATH_GPT_counterexample_proof_l1642_164245

theorem counterexample_proof :
  ∃ a : ℝ, |a - 1| > 1 ∧ ¬ (a > 2) :=
  sorry

end NUMINAMATH_GPT_counterexample_proof_l1642_164245


namespace NUMINAMATH_GPT_GroundBeefSalesTotalRevenue_l1642_164283

theorem GroundBeefSalesTotalRevenue :
  let price_regular := 3.50
  let price_lean := 4.25
  let price_extra_lean := 5.00

  let monday_revenue := 198.5 * price_regular +
                        276.2 * price_lean +
                        150.7 * price_extra_lean

  let tuesday_revenue := 210 * (price_regular * 0.90) +
                         420 * (price_lean * 0.90) +
                         150 * (price_extra_lean * 0.90)
  
  let wednesday_revenue := 230 * price_regular +
                           324.6 * 3.75 +
                           120.4 * price_extra_lean

  monday_revenue + tuesday_revenue + wednesday_revenue = 8189.35 :=
by
  sorry

end NUMINAMATH_GPT_GroundBeefSalesTotalRevenue_l1642_164283


namespace NUMINAMATH_GPT_stone_105_is_3_l1642_164262

def stone_numbered_at_105 (n : ℕ) := (15 + (n - 1) % 28)

theorem stone_105_is_3 :
  stone_numbered_at_105 105 = 3 := by
  sorry

end NUMINAMATH_GPT_stone_105_is_3_l1642_164262


namespace NUMINAMATH_GPT_value_of_a_l1642_164252

noncomputable def f : ℝ → ℝ 
| x => if x > 0 then 2^x else x + 1

theorem value_of_a (a : ℝ) (h : f a + f 1 = 0) : a = -3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l1642_164252


namespace NUMINAMATH_GPT_method1_three_sessions_cost_method2_more_cost_effective_for_nine_sessions_method1_allows_more_sessions_l1642_164293

/-- Method 1: Membership card costs 200 yuan + 10 yuan per swim session. -/
def method1_cost (num_sessions : ℕ) : ℕ := 200 + 10 * num_sessions

/-- Method 2: Each swim session costs 30 yuan. -/
def method2_cost (num_sessions : ℕ) : ℕ := 30 * num_sessions

/-- Problem (1): Total cost for 3 swim sessions using Method 1 is 230 yuan. -/
theorem method1_three_sessions_cost : method1_cost 3 = 230 := by
  sorry

/-- Problem (2): Method 2 is more cost-effective than Method 1 for 9 swim sessions. -/
theorem method2_more_cost_effective_for_nine_sessions : method2_cost 9 < method1_cost 9 := by
  sorry

/-- Problem (3): Method 1 allows more sessions than Method 2 within a budget of 600 yuan. -/
theorem method1_allows_more_sessions : (600 - 200) / 10 > 600 / 30 := by
  sorry

end NUMINAMATH_GPT_method1_three_sessions_cost_method2_more_cost_effective_for_nine_sessions_method1_allows_more_sessions_l1642_164293


namespace NUMINAMATH_GPT_campers_afternoon_l1642_164224

noncomputable def campers_morning : ℕ := 35
noncomputable def campers_total : ℕ := 62

theorem campers_afternoon :
  campers_total - campers_morning = 27 :=
by
  sorry

end NUMINAMATH_GPT_campers_afternoon_l1642_164224


namespace NUMINAMATH_GPT_sum_g_values_l1642_164271

noncomputable def g (x : ℝ) : ℝ :=
if x > 3 then x^2 - 1 else
if x >= -3 then 3 * x + 2 else 4

theorem sum_g_values : g (-4) + g 0 + g 4 = 21 :=
by
  sorry

end NUMINAMATH_GPT_sum_g_values_l1642_164271


namespace NUMINAMATH_GPT_find_number_l1642_164200

theorem find_number (a p x : ℕ) (h1 : p = 36) (h2 : 6 * a = 6 * (2 * p + x)) : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1642_164200


namespace NUMINAMATH_GPT_find_three_numbers_l1642_164240

-- Define the conditions
def condition1 (X : ℝ) : Prop := X = 0.35 * X + 60
def condition2 (X Y : ℝ) : Prop := X = 0.7 * (1 / 2) * Y + (1 / 2) * Y
def condition3 (Y Z : ℝ) : Prop := Y = 2 * Z ^ 2

-- Define the final result that we need to prove
def final_result (X Y Z : ℝ) : Prop := X = 92 ∧ Y = 108 ∧ Z = 7

-- The main theorem statement
theorem find_three_numbers :
  ∃ (X Y Z : ℝ), condition1 X ∧ condition2 X Y ∧ condition3 Y Z ∧ final_result X Y Z :=
by
  sorry

end NUMINAMATH_GPT_find_three_numbers_l1642_164240


namespace NUMINAMATH_GPT_blue_pill_cost_l1642_164243

variable (cost_blue_pill : ℕ) (cost_red_pill : ℕ) (daily_cost : ℕ) 
variable (num_days : ℕ) (total_cost : ℕ)
variable (cost_diff : ℕ)

theorem blue_pill_cost :
  num_days = 21 ∧
  total_cost = 966 ∧
  cost_diff = 4 ∧
  daily_cost = total_cost / num_days ∧
  daily_cost = cost_blue_pill + cost_red_pill ∧
  cost_blue_pill = cost_red_pill + cost_diff ∧
  daily_cost = 46 →
  cost_blue_pill = 25 := by
  sorry

end NUMINAMATH_GPT_blue_pill_cost_l1642_164243


namespace NUMINAMATH_GPT_remainder_when_divided_by_l1642_164238

def P (x : ℤ) : ℤ := 5 * x^8 - 2 * x^7 - 8 * x^6 + 3 * x^4 + 5 * x^3 - 13
def D (x : ℤ) : ℤ := 3 * (x - 3)

theorem remainder_when_divided_by (x : ℤ) : P 3 = 23364 :=
by {
  -- This is where the calculation steps would go, but we're omitting them.
  sorry
}

end NUMINAMATH_GPT_remainder_when_divided_by_l1642_164238


namespace NUMINAMATH_GPT_line_translation_upwards_units_l1642_164229

theorem line_translation_upwards_units:
  ∀ (x : ℝ), (y = x / 3) → (y = (x + 5) / 3) → (y' = y + 5 / 3) :=
by
  sorry

end NUMINAMATH_GPT_line_translation_upwards_units_l1642_164229


namespace NUMINAMATH_GPT_number_of_female_students_l1642_164223

noncomputable def total_students : ℕ := 1600
noncomputable def sample_size : ℕ := 200
noncomputable def sampled_males : ℕ := 110
noncomputable def sampled_females := sample_size - sampled_males
noncomputable def total_males := (sampled_males * total_students) / sample_size
noncomputable def total_females := total_students - total_males

theorem number_of_female_students : total_females = 720 := 
sorry

end NUMINAMATH_GPT_number_of_female_students_l1642_164223


namespace NUMINAMATH_GPT_find_quadruplets_l1642_164278

theorem find_quadruplets :
  ∃ (x y z w : ℝ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w ∧
  (xyz + 1) / (x + 1) = (yzw + 1) / (y + 1) ∧
  (yzw + 1) / (y + 1) = (zwx + 1) / (z + 1) ∧
  (zwx + 1) / (z + 1) = (wxy + 1) / (w + 1) ∧
  x + y + z + w = 48 ∧
  x = 12 ∧ y = 12 ∧ z = 12 ∧ w = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_quadruplets_l1642_164278


namespace NUMINAMATH_GPT_polly_age_is_33_l1642_164241

theorem polly_age_is_33 
  (x : ℕ) 
  (h1 : ∀ y, y = 20 → x - y = x - 20)
  (h2 : ∀ y, y = 22 → x - y = x - 22)
  (h3 : ∀ y, y = 24 → x - y = x - 24) : 
  x = 33 :=
by 
  sorry

end NUMINAMATH_GPT_polly_age_is_33_l1642_164241


namespace NUMINAMATH_GPT_meiosis_and_fertilization_outcome_l1642_164237

-- Definitions corresponding to the conditions:
def increases_probability_of_genetic_mutations (x : Type) := 
  ∃ (p : x), false -- Placeholder for the actual mutation rate being low

def inherits_all_genetic_material (x : Type) :=
  ∀ (p : x), false -- Parents do not pass all genes to offspring

def receives_exactly_same_genetic_information (x : Type) :=
  ∀ (p : x), false -- Offspring do not receive exact genetic information from either parent

def produces_genetic_combination_different (x : Type) :=
  ∃ (o : x), true -- The offspring has different genetic information from either parent

-- The main statement to be proven:
theorem meiosis_and_fertilization_outcome (x : Type) 
  (cond1 : ¬ increases_probability_of_genetic_mutations x)
  (cond2 : ¬ inherits_all_genetic_material x)
  (cond3 : ¬ receives_exactly_same_genetic_information x) :
  produces_genetic_combination_different x :=
sorry

end NUMINAMATH_GPT_meiosis_and_fertilization_outcome_l1642_164237


namespace NUMINAMATH_GPT_Darnel_sprinted_further_l1642_164235

-- Define the distances sprinted and jogged
def sprinted : ℝ := 0.88
def jogged : ℝ := 0.75

-- State the theorem to prove the main question
theorem Darnel_sprinted_further : sprinted - jogged = 0.13 :=
by
  sorry

end NUMINAMATH_GPT_Darnel_sprinted_further_l1642_164235


namespace NUMINAMATH_GPT_expected_value_of_problems_l1642_164284

-- Define the setup
def num_pairs : ℕ := 5
def num_shoes : ℕ := num_pairs * 2
def prob_same_color : ℚ := 1 / (num_shoes - 1)
def days : ℕ := 5

-- Define the expected value calculation using linearity of expectation
def expected_problems_per_day : ℚ := prob_same_color
def expected_total_problems : ℚ := days * expected_problems_per_day

-- Prove the expected number of practice problems Sandra gets to do over 5 days
theorem expected_value_of_problems : expected_total_problems = 5 / 9 := 
by 
  rw [expected_total_problems, expected_problems_per_day, prob_same_color]
  norm_num
  sorry

end NUMINAMATH_GPT_expected_value_of_problems_l1642_164284


namespace NUMINAMATH_GPT_find_x_value_l1642_164234

theorem find_x_value (x : ℝ) (hx : x ≠ 0) : 
    (1/x) + (3/x) / (6/x) = 1 → x = 2 := 
by 
    intro h
    sorry

end NUMINAMATH_GPT_find_x_value_l1642_164234


namespace NUMINAMATH_GPT_cubes_with_odd_red_faces_l1642_164295

-- Define the dimensions and conditions of the block
def block_length : ℕ := 6
def block_width: ℕ := 6
def block_height : ℕ := 2

-- The block is painted initially red on all sides
-- Then the bottom face is painted blue
-- The block is cut into 1-inch cubes
-- 

noncomputable def num_cubes_with_odd_red_faces (length width height : ℕ) : ℕ :=
  -- Only edge cubes have odd number of red faces in this configuration
  let corner_count := 8  -- 4 on top + 4 on bottom (each has 4 red faces)
  let edge_count := 40   -- 20 on top + 20 on bottom (each has 3 red faces)
  let face_only_count := 32 -- 16 on top + 16 on bottom (each has 2 red faces)
  -- The resulting total number of cubes with odd red faces
  edge_count

-- The theorem we need to prove
theorem cubes_with_odd_red_faces : num_cubes_with_odd_red_faces block_length block_width block_height = 40 :=
  by 
    -- Proof goes here
    sorry

end NUMINAMATH_GPT_cubes_with_odd_red_faces_l1642_164295


namespace NUMINAMATH_GPT_smallest_value_of_reciprocal_sums_l1642_164297

theorem smallest_value_of_reciprocal_sums (r1 r2 s p : ℝ) 
  (h1 : r1 + r2 = s)
  (h2 : r1^2 + r2^2 = s)
  (h3 : r1^3 + r2^3 = s)
  (h4 : r1^4 + r2^4 = s)
  (h1004 : r1^1004 + r2^1004 = s)
  (h_r1_r2_roots : ∀ x, x^2 - s * x + p = 0) :
  (1 / r1^1005 + 1 / r2^1005) = 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_of_reciprocal_sums_l1642_164297


namespace NUMINAMATH_GPT_initial_money_is_10_l1642_164276

-- Definition for the initial amount of money
def initial_money (X : ℝ) : Prop :=
  let spent_on_cupcakes := (1 / 5) * X
  let remaining_after_cupcakes := X - spent_on_cupcakes
  let spent_on_milkshake := 5
  let remaining_after_milkshake := remaining_after_cupcakes - spent_on_milkshake
  remaining_after_milkshake = 3

-- The statement proving that Ivan initially had $10
theorem initial_money_is_10 (X : ℝ) (h : initial_money X) : X = 10 :=
by sorry

end NUMINAMATH_GPT_initial_money_is_10_l1642_164276


namespace NUMINAMATH_GPT_parabola_focus_directrix_distance_l1642_164217

theorem parabola_focus_directrix_distance :
  ∀ (x y : ℝ), y = (1 / 4) * x^2 → 
  (∃ p : ℝ, p = 2 ∧ x^2 = 4 * p * y) →
  ∃ d : ℝ, d = 2 :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_directrix_distance_l1642_164217


namespace NUMINAMATH_GPT_allyn_total_expense_in_june_l1642_164231

/-- We have a house with 40 bulbs, each using 60 watts of power daily.
Allyn pays 0.20 dollars per watt used. June has 30 days.
We need to calculate Allyn's total monthly expense on electricity in June,
which should be \$14400. -/
theorem allyn_total_expense_in_june
    (daily_watt_per_bulb : ℕ := 60)
    (num_bulbs : ℕ := 40)
    (cost_per_watt : ℝ := 0.20)
    (days_in_june : ℕ := 30)
    : num_bulbs * daily_watt_per_bulb * days_in_june * cost_per_watt = 14400 := 
by
  sorry

end NUMINAMATH_GPT_allyn_total_expense_in_june_l1642_164231


namespace NUMINAMATH_GPT_profit_increase_l1642_164218

theorem profit_increase (x y : ℝ) (a : ℝ)
  (h1 : x = (57 / 20) * y)
  (h2 : (x - y) / y = a / 100)
  (h3 : (x - 0.95 * y) / (0.95 * y) = (a + 15) / 100) :
  a = 185 := sorry

end NUMINAMATH_GPT_profit_increase_l1642_164218


namespace NUMINAMATH_GPT_difference_mean_median_is_neg_half_l1642_164285

-- Definitions based on given conditions
def scoreDistribution : List (ℕ × ℚ) :=
  [(65, 0.05), (75, 0.25), (85, 0.4), (95, 0.2), (105, 0.1)]

-- Defining the total number of students as 100 for easier percentage calculations
def totalStudents := 100

-- Definition to compute mean
def mean : ℚ :=
  scoreDistribution.foldl (λ acc (score, percentage) => acc + (↑score * percentage)) 0

-- Median score based on the distribution conditions
def median : ℚ := 85

-- Proving the proposition that the difference between the mean and the median is -0.5
theorem difference_mean_median_is_neg_half :
  median - mean = -0.5 :=
sorry

end NUMINAMATH_GPT_difference_mean_median_is_neg_half_l1642_164285


namespace NUMINAMATH_GPT_num_ways_to_pay_l1642_164210

theorem num_ways_to_pay (n : ℕ) : 
  ∃ a_n : ℕ, a_n = (n / 2) + 1 :=
sorry

end NUMINAMATH_GPT_num_ways_to_pay_l1642_164210


namespace NUMINAMATH_GPT_cost_price_of_apple_l1642_164260

-- Define the given conditions SP = 20, and the relation between SP and CP.
variables (SP CP : ℝ)
axiom h1 : SP = 20
axiom h2 : SP = CP - (1/6) * CP

-- Statement to be proved.
theorem cost_price_of_apple : CP = 24 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_apple_l1642_164260


namespace NUMINAMATH_GPT_father_l1642_164294

noncomputable def father's_current_age : ℕ :=
  let S : ℕ := 40 -- Sebastian's current age
  let Si : ℕ := S - 10 -- Sebastian's sister's current age
  let sum_five_years_ago := (S - 5) + (Si - 5) -- Sum of their ages five years ago
  let father_age_five_years_ago := (4 * sum_five_years_ago) / 3 -- From the given condition
  father_age_five_years_ago + 5 -- Their father's current age

theorem father's_age_is_85 : father's_current_age = 85 :=
  sorry

end NUMINAMATH_GPT_father_l1642_164294


namespace NUMINAMATH_GPT_minimum_radius_part_a_minimum_radius_part_b_l1642_164268

-- Definitions for Part (a)
def a := 7
def b := 8
def c := 9
def R1 := 6

-- Statement for Part (a)
theorem minimum_radius_part_a : (c / 2) = R1 := by sorry

-- Definitions for Part (b)
def a' := 9
def b' := 15
def c' := 16
def R2 := 9

-- Statement for Part (b)
theorem minimum_radius_part_b : (c' / 2) = R2 := by sorry

end NUMINAMATH_GPT_minimum_radius_part_a_minimum_radius_part_b_l1642_164268


namespace NUMINAMATH_GPT_group_size_l1642_164298

-- Define the conditions
variables (N : ℕ)
variable (h1 : (1 / 5 : ℝ) * N = (N : ℝ) * 0.20)
variable (h2 : 128 ≤ N)
variable (h3 : (1 / 5 : ℝ) * N - 128 = 0.04 * (N : ℝ))

-- Prove that the number of people in the group is 800
theorem group_size : N = 800 :=
by
  sorry

end NUMINAMATH_GPT_group_size_l1642_164298


namespace NUMINAMATH_GPT_initial_pepper_amount_l1642_164266
-- Import the necessary libraries.

-- Declare the problem as a theorem.
theorem initial_pepper_amount (used left : ℝ) (h₁ : used = 0.16) (h₂ : left = 0.09) :
  used + left = 0.25 :=
by
  -- The proof is not required here.
  sorry

end NUMINAMATH_GPT_initial_pepper_amount_l1642_164266


namespace NUMINAMATH_GPT_assume_dead_heat_race_l1642_164265

variable {Va Vb L H : ℝ}

theorem assume_dead_heat_race (h1 : Va = (51 / 44) * Vb) :
  H = (7 / 51) * L :=
sorry

end NUMINAMATH_GPT_assume_dead_heat_race_l1642_164265


namespace NUMINAMATH_GPT_percent_in_second_part_l1642_164247

-- Defining the conditions and the proof statement
theorem percent_in_second_part (x y P : ℝ) 
  (h1 : 0.25 * (x - y) = (P / 100) * (x + y))
  (h2 : y = 0.25 * x) : 
  P = 15 :=
by
  sorry

end NUMINAMATH_GPT_percent_in_second_part_l1642_164247


namespace NUMINAMATH_GPT_average_weight_of_children_l1642_164222

theorem average_weight_of_children (avg_weight_boys avg_weight_girls : ℕ)
                                   (num_boys num_girls : ℕ)
                                   (h1 : avg_weight_boys = 160)
                                   (h2 : avg_weight_girls = 110)
                                   (h3 : num_boys = 8)
                                   (h4 : num_girls = 5) :
                                   (num_boys * avg_weight_boys + num_girls * avg_weight_girls) / (num_boys + num_girls) = 141 :=
by
    sorry

end NUMINAMATH_GPT_average_weight_of_children_l1642_164222


namespace NUMINAMATH_GPT_years_ago_l1642_164202

theorem years_ago (M D X : ℕ) (hM : M = 41) (hD : D = 23) 
  (h_eq : M - X = 2 * (D - X)) : X = 5 := by 
  sorry

end NUMINAMATH_GPT_years_ago_l1642_164202


namespace NUMINAMATH_GPT_find_y_coordinate_of_P_l1642_164233

theorem find_y_coordinate_of_P (P Q : ℝ × ℝ)
  (h1 : ∀ x, y = 0.8 * x) -- line equation
  (h2 : P.1 = 4) -- x-coordinate of P
  (h3 : P = Q) -- P and Q are equidistant from the line
  : P.2 = 3.2 := sorry

end NUMINAMATH_GPT_find_y_coordinate_of_P_l1642_164233


namespace NUMINAMATH_GPT_binomial_coefficient_7_5_permutation_7_5_l1642_164246

-- Define function for binomial coefficient
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define function for permutation calculation
def permutation (n k : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - k)

theorem binomial_coefficient_7_5 : binomial_coefficient 7 5 = 21 :=
by
  sorry

theorem permutation_7_5 : permutation 7 5 = 2520 :=
by
  sorry

end NUMINAMATH_GPT_binomial_coefficient_7_5_permutation_7_5_l1642_164246


namespace NUMINAMATH_GPT_problem_statement_l1642_164211

def f (x : ℕ) : ℕ := x^2 + x + 4
def g (x : ℕ) : ℕ := 3 * x^3 + 2

theorem problem_statement : g (f 3) = 12290 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1642_164211


namespace NUMINAMATH_GPT_unique_root_of_increasing_l1642_164212

variable {R : Type} [LinearOrderedField R] [DecidableEq R]

def increasing (f : R → R) : Prop :=
  ∀ x1 x2 : R, x1 < x2 → f x1 < f x2

theorem unique_root_of_increasing (f : R → R)
  (h_inc : increasing f) :
  ∃! x : R, f x = 0 :=
sorry

end NUMINAMATH_GPT_unique_root_of_increasing_l1642_164212


namespace NUMINAMATH_GPT_B_subsetneq_A_l1642_164201

def A : Set ℝ := { x : ℝ | x^2 - x - 2 < 0 }
def B : Set ℝ := { x : ℝ | 1 - x^2 > 0 }

theorem B_subsetneq_A : B ⊂ A :=
by
  sorry

end NUMINAMATH_GPT_B_subsetneq_A_l1642_164201


namespace NUMINAMATH_GPT_sin_double_angle_l1642_164221

theorem sin_double_angle (θ : ℝ) (h : Real.tan θ = 2) : Real.sin (2 * θ) = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l1642_164221


namespace NUMINAMATH_GPT_total_amount_paid_correct_l1642_164204

-- Define variables for prices of the pizzas
def first_pizza_price : ℝ := 8
def second_pizza_price : ℝ := 12
def third_pizza_price : ℝ := 10

-- Define variables for discount rate and tax rate
def discount_rate : ℝ := 0.20
def sales_tax_rate : ℝ := 0.05

-- Define the total amount paid by Mrs. Hilt
def total_amount_paid : ℝ :=
  let total_cost := first_pizza_price + second_pizza_price + third_pizza_price
  let discount := total_cost * discount_rate
  let discounted_total := total_cost - discount
  let sales_tax := discounted_total * sales_tax_rate
  discounted_total + sales_tax

-- Prove that the total amount paid is $25.20
theorem total_amount_paid_correct : total_amount_paid = 25.20 := 
  by
  sorry

end NUMINAMATH_GPT_total_amount_paid_correct_l1642_164204


namespace NUMINAMATH_GPT_toms_score_l1642_164215

theorem toms_score (T J : ℝ) (h1 : T = J + 30) (h2 : (T + J) / 2 = 90) : T = 105 := by
  sorry

end NUMINAMATH_GPT_toms_score_l1642_164215


namespace NUMINAMATH_GPT_squares_difference_sum_l1642_164249

theorem squares_difference_sum : 
  19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 200 :=
by
  sorry

end NUMINAMATH_GPT_squares_difference_sum_l1642_164249


namespace NUMINAMATH_GPT_b_product_l1642_164290

variable (a : ℕ → ℝ) (b : ℕ → ℝ)

-- All terms in the arithmetic sequence \{aₙ\} are non-zero.
axiom a_nonzero : ∀ n, a n ≠ 0

-- The sequence satisfies the given condition.
axiom a_cond : a 3 - (a 7)^2 / 2 + a 11 = 0

-- The sequence \{bₙ\} is a geometric sequence with ratio r.
axiom b_geometric : ∃ r, ∀ n, b (n + 1) = r * b n

-- And b₇ = a₇
axiom b_7 : b 7 = a 7

-- Prove that b₁ * b₁₃ = 16
theorem b_product : b 1 * b 13 = 16 :=
sorry

end NUMINAMATH_GPT_b_product_l1642_164290


namespace NUMINAMATH_GPT_number_of_trees_planted_l1642_164269

theorem number_of_trees_planted (initial_trees final_trees trees_planted : ℕ) 
  (h_initial : initial_trees = 22)
  (h_final : final_trees = 77)
  (h_planted : trees_planted = final_trees - initial_trees) : 
  trees_planted = 55 := by
  sorry

end NUMINAMATH_GPT_number_of_trees_planted_l1642_164269


namespace NUMINAMATH_GPT_ratio_of_perimeters_of_similar_triangles_l1642_164205

theorem ratio_of_perimeters_of_similar_triangles (A1 A2 P1 P2 : ℝ) (h : A1 / A2 = 16 / 9) : P1 / P2 = 4 / 3 :=
sorry

end NUMINAMATH_GPT_ratio_of_perimeters_of_similar_triangles_l1642_164205


namespace NUMINAMATH_GPT_average_time_to_win_permit_l1642_164282

theorem average_time_to_win_permit :
  let p n := (9/10)^(n-1) * (1/10)
  ∑' n, n * p n = 10 :=
sorry

end NUMINAMATH_GPT_average_time_to_win_permit_l1642_164282


namespace NUMINAMATH_GPT_Alex_hula_hoop_duration_l1642_164253

-- Definitions based on conditions
def Nancy_duration := 10
def Casey_duration := Nancy_duration - 3
def Morgan_duration := Casey_duration * 3
def Alex_duration := Casey_duration + Morgan_duration - 2

-- The theorem we need to prove
theorem Alex_hula_hoop_duration : Alex_duration = 26 := by
  -- proof to be provided
  sorry

end NUMINAMATH_GPT_Alex_hula_hoop_duration_l1642_164253


namespace NUMINAMATH_GPT_eval_fraction_product_l1642_164257

theorem eval_fraction_product :
  ((1 + (1 / 3)) * (1 + (1 / 4)) = (5 / 3)) :=
by
  sorry

end NUMINAMATH_GPT_eval_fraction_product_l1642_164257


namespace NUMINAMATH_GPT_max_point_h_l1642_164248

-- Definitions of the linear functions f and g
def f (x : ℝ) : ℝ := 2 * x + 2
def g (x : ℝ) : ℝ := -x - 3

-- The product of f(x) and g(x)
def h (x : ℝ) : ℝ := f x * g x

-- Statement: Prove that x = -2 is the maximum point of h(x)
theorem max_point_h : ∃ x_max : ℝ, h x_max = (-2) :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_max_point_h_l1642_164248


namespace NUMINAMATH_GPT_lana_trip_longer_by_25_percent_l1642_164214

-- Define the dimensions of the rectangular field
def length_field : ℕ := 3
def width_field : ℕ := 1

-- Define Tom's path distance
def tom_path_distance : ℕ := length_field + width_field

-- Define Lana's path distance
def lana_path_distance : ℕ := 2 + 1 + 1 + 1

-- Define the percentage increase calculation
def percentage_increase (initial final : ℕ) : ℕ :=
  (final - initial) * 100 / initial

-- Define the theorem to be proven
theorem lana_trip_longer_by_25_percent :
  percentage_increase tom_path_distance lana_path_distance = 25 :=
by
  sorry

end NUMINAMATH_GPT_lana_trip_longer_by_25_percent_l1642_164214


namespace NUMINAMATH_GPT_fraction_to_decimal_l1642_164213

theorem fraction_to_decimal : (7 / 12 : ℝ) = 0.5833 + (3 / 10000) * (1 / (1 - (1 / 10))) := 
by sorry

end NUMINAMATH_GPT_fraction_to_decimal_l1642_164213


namespace NUMINAMATH_GPT_exist_directed_graph_two_step_l1642_164227

theorem exist_directed_graph_two_step {n : ℕ} (h : n > 4) :
  ∃ G : SimpleGraph (Fin n), 
    (∀ u v : Fin n, u ≠ v → 
      (G.Adj u v ∨ (∃ w : Fin n, u ≠ w ∧ w ≠ v ∧ G.Adj u w ∧ G.Adj w v))) :=
sorry

end NUMINAMATH_GPT_exist_directed_graph_two_step_l1642_164227


namespace NUMINAMATH_GPT_fraction_doubling_unchanged_l1642_164226

theorem fraction_doubling_unchanged (x y : ℝ) (h : x ≠ y) : 
  (3 * (2 * x)) / (2 * x - 2 * y) = (3 * x) / (x - y) :=
by
  sorry

end NUMINAMATH_GPT_fraction_doubling_unchanged_l1642_164226


namespace NUMINAMATH_GPT_gift_card_amount_l1642_164244

theorem gift_card_amount (original_price final_price : ℝ) 
  (discount1 discount2 : ℝ) 
  (discounted_price1 discounted_price2 : ℝ) :
  original_price = 2000 →
  discount1 = 0.15 →
  discount2 = 0.10 →
  discounted_price1 = original_price - (discount1 * original_price) →
  discounted_price2 = discounted_price1 - (discount2 * discounted_price1) →
  final_price = 1330 →
  discounted_price2 - final_price = 200 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_gift_card_amount_l1642_164244


namespace NUMINAMATH_GPT_copy_pages_15_dollars_l1642_164299

theorem copy_pages_15_dollars (cost_per_page : ℕ) (total_dollars : ℕ) (cents_per_dollar : ℕ) : 
  cost_per_page = 3 → total_dollars = 15 → cents_per_dollar = 100 → 
  (total_dollars * cents_per_dollar) / cost_per_page = 500 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_copy_pages_15_dollars_l1642_164299


namespace NUMINAMATH_GPT_max_value_ineq_l1642_164216

theorem max_value_ineq (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (x + y)^2 / (x^2 + y^2 + xy) ≤ 4 / 3 :=
sorry

end NUMINAMATH_GPT_max_value_ineq_l1642_164216


namespace NUMINAMATH_GPT_kerosene_cost_is_024_l1642_164277

-- Definitions from the conditions
def dozen_eggs_cost := 0.36 -- Cost of a dozen eggs is the same as 1 pound of rice which is $0.36
def pound_of_rice_cost := 0.36
def kerosene_cost := 8 * (0.36 / 12) -- Cost of kerosene is the cost of 8 eggs

-- Theorem to prove
theorem kerosene_cost_is_024 : kerosene_cost = 0.24 := by
  sorry

end NUMINAMATH_GPT_kerosene_cost_is_024_l1642_164277


namespace NUMINAMATH_GPT_max_parallelograms_in_hexagon_l1642_164264

theorem max_parallelograms_in_hexagon (side_hexagon side_parallelogram1 side_parallelogram2 : ℝ)
                                        (angle_parallelogram : ℝ) :
  side_hexagon = 3 ∧ side_parallelogram1 = 1 ∧ side_parallelogram2 = 2 ∧ angle_parallelogram = (π / 3) →
  ∃ n : ℕ, n = 12 :=
by 
  sorry

end NUMINAMATH_GPT_max_parallelograms_in_hexagon_l1642_164264


namespace NUMINAMATH_GPT_train_time_to_pass_platform_l1642_164286

noncomputable def train_length : ℝ := 360
noncomputable def platform_length : ℝ := 140
noncomputable def train_speed_km_per_hr : ℝ := 45

noncomputable def train_speed_m_per_s : ℝ :=
  train_speed_km_per_hr * (1000 / 3600)

noncomputable def total_distance : ℝ :=
  train_length + platform_length

theorem train_time_to_pass_platform :
  (total_distance / train_speed_m_per_s) = 40 := by
  sorry

end NUMINAMATH_GPT_train_time_to_pass_platform_l1642_164286


namespace NUMINAMATH_GPT_find_f_105_5_l1642_164258

noncomputable def f : ℝ → ℝ :=
sorry -- Definition of f

-- Hypotheses
axiom even_function (x : ℝ) : f x = f (-x)
axiom functional_equation (x : ℝ) : f (x + 2) = -f x
axiom function_values (x : ℝ) (hx : 2 ≤ x ∧ x ≤ 3) : f x = x

-- Goal
theorem find_f_105_5 : f 105.5 = 2.5 :=
sorry

end NUMINAMATH_GPT_find_f_105_5_l1642_164258


namespace NUMINAMATH_GPT_original_average_marks_l1642_164280

theorem original_average_marks (n : ℕ) (A : ℝ) (new_avg : ℝ) 
  (h1 : n = 30) 
  (h2 : new_avg = 90)
  (h3 : ∀ new_avg, new_avg = 2 * A → A = 90 / 2) : 
  A = 45 :=
by
  sorry

end NUMINAMATH_GPT_original_average_marks_l1642_164280


namespace NUMINAMATH_GPT_measure_15_minutes_l1642_164206

/-- Given a timer setup with a 7-minute hourglass and an 11-minute hourglass, show that we can measure exactly 15 minutes. -/
theorem measure_15_minutes (h7 : ∃ t : ℕ, t = 7) (h11 : ∃ t : ℕ, t = 11) : ∃ t : ℕ, t = 15 := 
  by 
    sorry

end NUMINAMATH_GPT_measure_15_minutes_l1642_164206


namespace NUMINAMATH_GPT_boy_usual_time_to_school_l1642_164273

theorem boy_usual_time_to_school
  (S : ℝ) -- Usual speed
  (T : ℝ) -- Usual time
  (D : ℝ) -- Distance, D = S * T
  (hD : D = S * T)
  (h1 : 3/4 * D / (7/6 * S) + 1/4 * D / (5/6 * S) = T - 2) : 
  T = 35 :=
by
  sorry

end NUMINAMATH_GPT_boy_usual_time_to_school_l1642_164273


namespace NUMINAMATH_GPT_point_always_outside_circle_l1642_164287

theorem point_always_outside_circle (a : ℝ) : a^2 + (2 - a)^2 > 1 :=
by sorry

end NUMINAMATH_GPT_point_always_outside_circle_l1642_164287


namespace NUMINAMATH_GPT_value_of_a_l1642_164272

def A := { x : ℝ | x^2 - 8*x + 15 = 0 }
def B (a : ℝ) := { x : ℝ | x * a - 1 = 0 }

theorem value_of_a (a : ℝ) : (∀ x, x ∈ B a → x ∈ A) ↔ (a = 0 ∨ a = 1/3 ∨ a = 1/5) :=
by sorry

end NUMINAMATH_GPT_value_of_a_l1642_164272


namespace NUMINAMATH_GPT_sin_A_plus_B_eq_max_area_eq_l1642_164208

-- Conditions for problem 1 and 2
variables (A B C a b c : ℝ)
variable (h_A_B_C : A + B + C = Real.pi)
variable (h_sin_C_div_2 : Real.sin (C / 2) = 2 * Real.sqrt 2 / 3)

noncomputable def sin_A_plus_B := Real.sin (A + B)

-- Problem 1: Prove that sin(A + B) = 4 * sqrt 2 / 9
theorem sin_A_plus_B_eq : sin_A_plus_B A B = 4 * Real.sqrt 2 / 9 :=
by sorry

-- Adding additional conditions for problem 2
variable (h_a_b_sum : a + b = 2 * Real.sqrt 2)

noncomputable def area (a b C : ℝ) := (1 / 2) * a * b * (2 * Real.sin (C / 2) * (Real.cos (C / 2)))

-- Problem 2: Prove that the maximum value of the area S of triangle ABC is 4 * sqrt 2 / 9
theorem max_area_eq : ∃ S, S = area a b C ∧ S ≤ 4 * Real.sqrt 2 / 9 :=
by sorry

end NUMINAMATH_GPT_sin_A_plus_B_eq_max_area_eq_l1642_164208
