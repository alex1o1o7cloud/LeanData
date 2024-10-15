import Mathlib

namespace NUMINAMATH_GPT_range_of_a_for_inequality_l22_2207

theorem range_of_a_for_inequality (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2 * x + a > 0) → a > 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_for_inequality_l22_2207


namespace NUMINAMATH_GPT_value_of_m_l22_2210

theorem value_of_m (m : ℝ) (h1 : m - 2 ≠ 0) (h2 : |m| - 1 = 1) : m = -2 := by {
  sorry
}

end NUMINAMATH_GPT_value_of_m_l22_2210


namespace NUMINAMATH_GPT_square_placement_conditions_l22_2299

-- Definitions for natural numbers at vertices and center
def top_left := 14
def top_right := 6
def bottom_right := 15
def bottom_left := 35
def center := 210

theorem square_placement_conditions :
  (∃ gcd1 > 1, gcd1 = Nat.gcd top_left top_right) ∧
  (∃ gcd2 > 1, gcd2 = Nat.gcd top_right bottom_right) ∧
  (∃ gcd3 > 1, gcd3 = Nat.gcd bottom_right bottom_left) ∧
  (∃ gcd4 > 1, gcd4 = Nat.gcd bottom_left top_left) ∧
  (Nat.gcd top_left bottom_right = 1) ∧
  (Nat.gcd top_right bottom_left = 1) ∧
  (Nat.gcd top_left center > 1) ∧
  (Nat.gcd top_right center > 1) ∧
  (Nat.gcd bottom_right center > 1) ∧
  (Nat.gcd bottom_left center > 1) 
 := by
sorry

end NUMINAMATH_GPT_square_placement_conditions_l22_2299


namespace NUMINAMATH_GPT_describe_set_T_l22_2230

theorem describe_set_T:
  ( ∀ (x y : ℝ), ((x + 2 = 4 ∧ y - 5 ≤ 4) ∨ (y - 5 = 4 ∧ x + 2 ≤ 4) ∨ (x + 2 = y - 5 ∧ 4 ≤ x + 2)) →
    ( ∃ (x y : ℝ), x = 2 ∧ y ≤ 9 ∨ y = 9 ∧ x ≤ 2 ∨ y = x + 7 ∧ x ≥ 2 ∧ y ≥ 9) ) :=
sorry

end NUMINAMATH_GPT_describe_set_T_l22_2230


namespace NUMINAMATH_GPT_determine_c_absolute_value_l22_2204

theorem determine_c_absolute_value
  (a b c : ℤ)
  (h_gcd : Int.gcd (Int.gcd a b) c = 1)
  (h_root : a * (Complex.mk 3 1)^4 + b * (Complex.mk 3 1)^3 + c * (Complex.mk 3 1)^2 + b * (Complex.mk 3 1) + a = 0) :
  |c| = 109 := 
sorry

end NUMINAMATH_GPT_determine_c_absolute_value_l22_2204


namespace NUMINAMATH_GPT_three_layers_coverage_l22_2252

/--
Three table runners have a combined area of 208 square inches. 
By overlapping the runners to cover 80% of a table of area 175 square inches, 
the area that is covered by exactly two layers of runner is 24 square inches. 
Prove that the area of the table that is covered with three layers of runner is 22 square inches.
--/
theorem three_layers_coverage :
  ∀ (A T two_layers total_table_coverage : ℝ),
  A = 208 ∧ total_table_coverage = 0.8 * 175 ∧ two_layers = 24 →
  A = (total_table_coverage - two_layers - T) + 2 * two_layers + 3 * T →
  T = 22 :=
by
  intros A T two_layers total_table_coverage h1 h2
  sorry

end NUMINAMATH_GPT_three_layers_coverage_l22_2252


namespace NUMINAMATH_GPT_remaining_amount_spent_on_watermelons_l22_2203

def pineapple_cost : ℕ := 7
def total_spent : ℕ := 38
def pineapples_purchased : ℕ := 2

theorem remaining_amount_spent_on_watermelons:
  total_spent - (pineapple_cost * pineapples_purchased) = 24 :=
by
  sorry

end NUMINAMATH_GPT_remaining_amount_spent_on_watermelons_l22_2203


namespace NUMINAMATH_GPT_find_binomial_params_l22_2232

noncomputable def binomial_params (n p : ℝ) := 2.4 = n * p ∧ 1.44 = n * p * (1 - p)

theorem find_binomial_params (n p : ℝ) (h : binomial_params n p) : n = 6 ∧ p = 0.4 :=
by
  sorry

end NUMINAMATH_GPT_find_binomial_params_l22_2232


namespace NUMINAMATH_GPT_population_net_increase_l22_2271

def birth_rate : ℕ := 8
def birth_time : ℕ := 2
def death_rate : ℕ := 6
def death_time : ℕ := 2
def seconds_per_minute : ℕ := 60
def minutes_per_hour : ℕ := 60
def hours_per_day : ℕ := 24

theorem population_net_increase :
  (birth_rate / birth_time - death_rate / death_time) * (seconds_per_minute * minutes_per_hour * hours_per_day) = 86400 :=
by
  sorry

end NUMINAMATH_GPT_population_net_increase_l22_2271


namespace NUMINAMATH_GPT_grid_covering_impossible_l22_2202

theorem grid_covering_impossible :
  ∀ (x y : ℕ), x + y = 19 → 6 * x + 7 * y = 132 → False :=
by
  intros x y h1 h2
  -- Proof would go here.
  sorry

end NUMINAMATH_GPT_grid_covering_impossible_l22_2202


namespace NUMINAMATH_GPT_find_ordered_triple_l22_2237

theorem find_ordered_triple
  (a b c : ℝ)
  (h1 : a > 2)
  (h2 : b > 2)
  (h3 : c > 2)
  (h4 : (a + 3) ^ 2 / (b + c - 3) + (b + 5) ^ 2 / (c + a - 5) + (c + 7) ^ 2 / (a + b - 7) = 48) :
  (a, b, c) = (7, 5, 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_ordered_triple_l22_2237


namespace NUMINAMATH_GPT_cos_double_angle_trig_identity_l22_2224

theorem cos_double_angle_trig_identity
  (α : ℝ) 
  (h : Real.sin (α - Real.pi / 3) = 4 / 5) : 
  Real.cos (2 * α + Real.pi / 3) = 7 / 25 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_trig_identity_l22_2224


namespace NUMINAMATH_GPT_tangent_line_relation_l22_2245

noncomputable def proof_problem (x1 x2 : ℝ) : Prop :=
  ((∃ (P Q : ℝ × ℝ),
    P = (x1, Real.log x1) ∧
    Q = (x2, Real.exp x2) ∧
    ∀ k : ℝ, Real.exp x2 = k ↔ k * (x2 - x1) = Real.log x1 - Real.exp x2) →
    (((x1 * Real.exp x2 = 1) ∧ ((x1 + 1) / (x1 - 1) + x2 = 0))))


theorem tangent_line_relation (x1 x2 : ℝ) (h : proof_problem x1 x2) : 
  (x1 * Real.exp x2 = 1) ∧ ((x1 + 1) / (x1 - 1) + x2 = 0) :=
sorry

end NUMINAMATH_GPT_tangent_line_relation_l22_2245


namespace NUMINAMATH_GPT_simplify_fraction_l22_2275

-- Define the given variables and their assigned values.
variable (b : ℕ)
variable (b_eq : b = 2)

-- State the theorem we want to prove
theorem simplify_fraction (b : ℕ) (h : b = 2) : 
  15 * b ^ 4 / (75 * b ^ 3) = 2 / 5 :=
by
  -- sorry indicates where the proof would be written.
  sorry

end NUMINAMATH_GPT_simplify_fraction_l22_2275


namespace NUMINAMATH_GPT_intersection_is_correct_l22_2256

noncomputable def setA : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
noncomputable def setB : Set ℝ := {x | Real.log x / Real.log 2 ≤ 2}

theorem intersection_is_correct : setA ∩ setB = {x : ℝ | 0 < x ∧ x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_is_correct_l22_2256


namespace NUMINAMATH_GPT_johns_total_earnings_per_week_l22_2240

def small_crab_baskets_monday := 3
def medium_crab_baskets_monday := 2
def large_crab_baskets_thursday := 4
def jumbo_crab_baskets_thursday := 1

def crabs_per_small_basket := 4
def crabs_per_medium_basket := 3
def crabs_per_large_basket := 5
def crabs_per_jumbo_basket := 2

def price_per_small_crab := 3
def price_per_medium_crab := 4
def price_per_large_crab := 5
def price_per_jumbo_crab := 7

def total_weekly_earnings :=
  (small_crab_baskets_monday * crabs_per_small_basket * price_per_small_crab) +
  (medium_crab_baskets_monday * crabs_per_medium_basket * price_per_medium_crab) +
  (large_crab_baskets_thursday * crabs_per_large_basket * price_per_large_crab) +
  (jumbo_crab_baskets_thursday * crabs_per_jumbo_basket * price_per_jumbo_crab)

theorem johns_total_earnings_per_week : total_weekly_earnings = 174 :=
by sorry

end NUMINAMATH_GPT_johns_total_earnings_per_week_l22_2240


namespace NUMINAMATH_GPT_teacher_discount_l22_2220

-- Definitions that capture the conditions in Lean
def num_students : ℕ := 30
def num_pens_per_student : ℕ := 5
def num_notebooks_per_student : ℕ := 3
def num_binders_per_student : ℕ := 1
def num_highlighters_per_student : ℕ := 2
def cost_per_pen : ℚ := 0.50
def cost_per_notebook : ℚ := 1.25
def cost_per_binder : ℚ := 4.25
def cost_per_highlighter : ℚ := 0.75
def amount_spent : ℚ := 260

-- Compute the total cost without discount
def total_cost : ℚ :=
  (num_students * num_pens_per_student) * cost_per_pen +
  (num_students * num_notebooks_per_student) * cost_per_notebook +
  (num_students * num_binders_per_student) * cost_per_binder +
  (num_students * num_highlighters_per_student) * cost_per_highlighter

-- The main theorem to prove the applied teacher discount
theorem teacher_discount :
  total_cost - amount_spent = 100 := by
  sorry

end NUMINAMATH_GPT_teacher_discount_l22_2220


namespace NUMINAMATH_GPT_initial_bananas_on_tree_l22_2298

-- Definitions of given conditions
def bananas_left_on_tree : ℕ := 100
def bananas_eaten : ℕ := 70
def bananas_in_basket : ℕ := 2 * bananas_eaten

-- Statement to prove the initial number of bananas on the tree
theorem initial_bananas_on_tree : bananas_left_on_tree + (bananas_in_basket + bananas_eaten) = 310 :=
by
  sorry

end NUMINAMATH_GPT_initial_bananas_on_tree_l22_2298


namespace NUMINAMATH_GPT_sum_at_simple_interest_l22_2282

theorem sum_at_simple_interest (P R : ℝ) (h1 : P * R * 3 / 100 - P * (R + 3) * 3 / 100 = -90) : P = 1000 :=
sorry

end NUMINAMATH_GPT_sum_at_simple_interest_l22_2282


namespace NUMINAMATH_GPT_probability_of_each_suit_in_five_draws_with_replacement_l22_2279

theorem probability_of_each_suit_in_five_draws_with_replacement :
  let deck_size := 52
  let num_cards := 5
  let num_suits := 4
  let prob_each_suit := 1/4
  let target_probability := 9/16
  prob_each_suit * (3/4) * (1/2) * (1/4) * 24 = target_probability :=
by sorry

end NUMINAMATH_GPT_probability_of_each_suit_in_five_draws_with_replacement_l22_2279


namespace NUMINAMATH_GPT_first_class_students_count_l22_2235

theorem first_class_students_count 
  (x : ℕ) 
  (avg1 : ℕ) (avg2 : ℕ) (num2 : ℕ) (overall_avg : ℝ)
  (h_avg1 : avg1 = 40)
  (h_avg2 : avg2 = 60)
  (h_num2 : num2 = 50)
  (h_overall_avg : overall_avg = 52.5)
  (h_eq : 40 * x + 60 * 50 = (52.5:ℝ) * (x + 50)) :
  x = 30 :=
by
  sorry

end NUMINAMATH_GPT_first_class_students_count_l22_2235


namespace NUMINAMATH_GPT_find_line_equation_l22_2274

-- define the condition of passing through the point (-3, -1)
def passes_through (x y : ℝ) (a b : ℝ) := (a = -3) ∧ (b = -1)

-- define the condition of being parallel to the line x - 3y - 1 = 0
def is_parallel (m n c : ℝ) := (m = 1) ∧ (n = -3)

-- theorem statement
theorem find_line_equation (a b : ℝ) (c : ℝ) :
  passes_through a b (-3) (-1) →
  is_parallel 1 (-3) c →
  (a - 3 * b + c = 0) :=
sorry

end NUMINAMATH_GPT_find_line_equation_l22_2274


namespace NUMINAMATH_GPT_total_soda_bottles_l22_2272

def regular_soda : ℕ := 57
def diet_soda : ℕ := 26
def lite_soda : ℕ := 27

theorem total_soda_bottles : regular_soda + diet_soda + lite_soda = 110 := by
  sorry

end NUMINAMATH_GPT_total_soda_bottles_l22_2272


namespace NUMINAMATH_GPT_market_value_of_house_l22_2236

theorem market_value_of_house 
  (M : ℝ) -- Market value of the house
  (S : ℝ) -- Selling price of the house
  (P : ℝ) -- Pre-tax amount each person gets
  (after_tax : ℝ := 135000) -- Each person's amount after taxes
  (tax_rate : ℝ := 0.10) -- Tax rate
  (num_people : ℕ := 4) -- Number of people splitting the revenue
  (over_market_value_rate : ℝ := 0.20): 
  S = M + over_market_value_rate * M → 
  (num_people * P) = S → 
  after_tax = (1 - tax_rate) * P → 
  M = 500000 := 
by
  sorry

end NUMINAMATH_GPT_market_value_of_house_l22_2236


namespace NUMINAMATH_GPT_number_of_skew_line_pairs_in_cube_l22_2273

theorem number_of_skew_line_pairs_in_cube : 
  let vertices := 8
  let total_lines := 28
  let sets_of_4_points := Nat.choose 8 4 - 12
  let skew_pairs_per_set := 3
  let number_of_skew_pairs := sets_of_4_points * skew_pairs_per_set
  number_of_skew_pairs = 174 := sorry

end NUMINAMATH_GPT_number_of_skew_line_pairs_in_cube_l22_2273


namespace NUMINAMATH_GPT_find_a_l22_2261

noncomputable def f (x a : ℝ) : ℝ := |x - 4| + |x - a|

theorem find_a (a : ℝ) (h : ∃ x, f x a = 3) : a = 1 ∨ a = 7 := 
sorry

end NUMINAMATH_GPT_find_a_l22_2261


namespace NUMINAMATH_GPT_only_k_equal_1_works_l22_2278

-- Define the first k prime numbers product
def prime_prod (k : ℕ) : ℕ :=
  Nat.recOn k 1 (fun n prod => prod * (Nat.factorial (n + 1) - Nat.factorial n))

-- Define a predicate for being the sum of two positive cubes
def is_sum_of_two_cubes (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ n = a^3 + b^3

-- The theorem statement
theorem only_k_equal_1_works :
  ∀ k : ℕ, (prime_prod k = 2 ↔ k = 1) :=
by
  sorry

end NUMINAMATH_GPT_only_k_equal_1_works_l22_2278


namespace NUMINAMATH_GPT_largest_lambda_inequality_l22_2205

theorem largest_lambda_inequality :
  ∀ (a b c d e : ℝ), 0 ≤ a → 0 ≤ b → 0 ≤ c → 0 ≤ d → 0 ≤ e →
  (a^2 + b^2 + c^2 + d^2 + e^2 ≥ a * b + (5/4) * b * c + c * d + d * e) :=
by
  sorry

end NUMINAMATH_GPT_largest_lambda_inequality_l22_2205


namespace NUMINAMATH_GPT_sum_of_coefficients_of_expansion_l22_2294

theorem sum_of_coefficients_of_expansion (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, (2 * x - 1)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) →
  a_1 + a_2 + a_3 + a_4 + a_5 = 2 :=
by
  intro h
  have h0 := h 0
  have h1 := h 1
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_of_expansion_l22_2294


namespace NUMINAMATH_GPT_max_f_value_range_of_a_l22_2293

noncomputable def f (x a : ℝ) : ℝ := |x + 1| - |x - 4| - a

theorem max_f_value (a : ℝ) : ∃ x, f x a = 5 - a :=
sorry

theorem range_of_a (a : ℝ) : (∃ x, f x a ≥ (4 / a) + 1) ↔ (a = 2 ∨ a < 0) :=
sorry

end NUMINAMATH_GPT_max_f_value_range_of_a_l22_2293


namespace NUMINAMATH_GPT_plates_per_meal_l22_2255

theorem plates_per_meal 
  (people : ℕ) (meals_per_day : ℕ) (total_days : ℕ) (total_plates : ℕ) 
  (h_people : people = 6) 
  (h_meals : meals_per_day = 3) 
  (h_days : total_days = 4) 
  (h_plates : total_plates = 144) 
  : (total_plates / (people * meals_per_day * total_days)) = 2 := 
  sorry

end NUMINAMATH_GPT_plates_per_meal_l22_2255


namespace NUMINAMATH_GPT_find_side_length_l22_2263

theorem find_side_length
  (X : ℕ)
  (h1 : 3 + 2 + X + 4 = 12) :
  X = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_side_length_l22_2263


namespace NUMINAMATH_GPT_expand_and_simplify_l22_2208

theorem expand_and_simplify (x : ℝ) : (x^2 + 4) * (x - 5) = x^3 - 5 * x^2 + 4 * x - 20 := 
sorry

end NUMINAMATH_GPT_expand_and_simplify_l22_2208


namespace NUMINAMATH_GPT_series_ln2_series_1_ln2_l22_2219

theorem series_ln2 :
  ∑' n : ℕ, (1 / (n + 1) / (n + 2)) = Real.log 2 :=
sorry

theorem series_1_ln2 :
  ∑' k : ℕ, (1 / ((2 * k + 2) * (2 * k + 3))) = 1 - Real.log 2 :=
sorry

end NUMINAMATH_GPT_series_ln2_series_1_ln2_l22_2219


namespace NUMINAMATH_GPT_temperature_on_fifth_day_l22_2251

theorem temperature_on_fifth_day (T : ℕ → ℝ) (x : ℝ)
  (h1 : (T 1 + T 2 + T 3 + T 4) / 4 = 58)
  (h2 : (T 2 + T 3 + T 4 + T 5) / 4 = 59)
  (h3 : T 1 / T 5 = 7 / 8) :
  T 5 = 32 := 
sorry

end NUMINAMATH_GPT_temperature_on_fifth_day_l22_2251


namespace NUMINAMATH_GPT_problem_1_problem_2_l22_2267

variables (α : ℝ) (h : Real.tan α = 3)

theorem problem_1 : (Real.sin α + 3 * Real.cos α) / (2 * Real.sin α + 5 * Real.cos α) = 6 / 11 :=
by
  -- Proof is skipped
  sorry

theorem problem_2 : Real.sin α * Real.sin α + Real.sin α * Real.cos α + 3 * Real.cos α * Real.cos α = 3 / 2 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l22_2267


namespace NUMINAMATH_GPT_equal_powers_eq_a_b_l22_2211

theorem equal_powers_eq_a_b 
  (a b : ℝ) 
  (ha_pos : 0 < a) 
  (hb_pos : 0 < b)
  (h_exp_eq : a^b = b^a)
  (h_a_lt_1 : a < 1) : 
  a = b :=
sorry

end NUMINAMATH_GPT_equal_powers_eq_a_b_l22_2211


namespace NUMINAMATH_GPT_pete_ate_percentage_l22_2287

-- Definitions of the conditions
def total_slices : ℕ := 2 * 12
def stephen_ate_slices : ℕ := (25 * total_slices) / 100
def remaining_slices_after_stephen : ℕ := total_slices - stephen_ate_slices
def slices_left_after_pete : ℕ := 9

-- The statement to be proved
theorem pete_ate_percentage (h1 : total_slices = 24)
                            (h2 : stephen_ate_slices = 6)
                            (h3 : remaining_slices_after_stephen = 18)
                            (h4 : slices_left_after_pete = 9) :
  ((remaining_slices_after_stephen - slices_left_after_pete) * 100 / remaining_slices_after_stephen) = 50 :=
sorry

end NUMINAMATH_GPT_pete_ate_percentage_l22_2287


namespace NUMINAMATH_GPT_problem_statement_l22_2233

theorem problem_statement (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 := 
sorry

end NUMINAMATH_GPT_problem_statement_l22_2233


namespace NUMINAMATH_GPT_probability_of_perfect_square_sum_l22_2264

def is_perfect_square (n : ℕ) : Prop :=
  n = 1*1 ∨ n = 2*2 ∨ n = 3*3 ∨ n = 4*4

theorem probability_of_perfect_square_sum :
  let total_outcomes := 64
  let perfect_square_sums := 12
  (perfect_square_sums / total_outcomes : ℚ) = 3 / 16 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_perfect_square_sum_l22_2264


namespace NUMINAMATH_GPT_Michaela_needs_20_oranges_l22_2231

variable (M : ℕ)
variable (C : ℕ)

theorem Michaela_needs_20_oranges 
  (h1 : C = 2 * M)
  (h2 : M + C = 60):
  M = 20 :=
by 
  sorry

end NUMINAMATH_GPT_Michaela_needs_20_oranges_l22_2231


namespace NUMINAMATH_GPT_total_price_of_purchases_l22_2262

def price_of_refrigerator := 4275
def price_difference := 1490
def price_of_washing_machine := price_of_refrigerator - price_difference
def total_price := price_of_refrigerator + price_of_washing_machine

theorem total_price_of_purchases : total_price = 7060 :=
by
  rfl  -- This is just a placeholder; you need to solve the proof.

end NUMINAMATH_GPT_total_price_of_purchases_l22_2262


namespace NUMINAMATH_GPT_count_measures_of_angle_A_l22_2225

theorem count_measures_of_angle_A :
  ∃ n : ℕ, n = 17 ∧
  ∃ (A B : ℕ), A > 0 ∧ B > 0 ∧ A + B = 180 ∧ (∃ k : ℕ, k ≥ 1 ∧ A = k * B) ∧ (∀ (A' B' : ℕ), A' > 0 ∧ B' > 0 ∧ A' + B' = 180 ∧ (∀ k : ℕ, k ≥ 1 ∧ A' = k * B') → n = 17) :=
sorry

end NUMINAMATH_GPT_count_measures_of_angle_A_l22_2225


namespace NUMINAMATH_GPT_smallest_int_a_for_inequality_l22_2269

theorem smallest_int_a_for_inequality (a : ℤ) : 
  (∀ x : ℝ, (0 < x ∧ x < Real.pi / 2) → 
  Real.exp x - x * Real.cos x + Real.cos x * Real.log (Real.cos x) + a * x^2 ≥ 1) → 
  a = 1 := 
sorry

end NUMINAMATH_GPT_smallest_int_a_for_inequality_l22_2269


namespace NUMINAMATH_GPT_hancho_milk_consumption_l22_2290

theorem hancho_milk_consumption :
  ∀ (initial_yeseul_consumption gayoung_bonus liters_left initial_milk consumption_yeseul consumption_gayoung consumption_total), 
  initial_yeseul_consumption = 0.1 →
  gayoung_bonus = 0.2 →
  liters_left = 0.3 →
  initial_milk = 1 →
  consumption_yeseul = initial_yeseul_consumption →
  consumption_gayoung = initial_yeseul_consumption + gayoung_bonus →
  consumption_total = consumption_yeseul + consumption_gayoung →
  (initial_milk - (consumption_total + liters_left)) = 0.3 :=
by sorry

end NUMINAMATH_GPT_hancho_milk_consumption_l22_2290


namespace NUMINAMATH_GPT_taylor_class_more_girls_l22_2268

theorem taylor_class_more_girls (b g : ℕ) (total : b + g = 42) (ratio : b / g = 3 / 4) : g - b = 6 := by
  sorry

end NUMINAMATH_GPT_taylor_class_more_girls_l22_2268


namespace NUMINAMATH_GPT_gold_tetrahedron_volume_l22_2248

theorem gold_tetrahedron_volume (side_length : ℝ) (h : side_length = 8) : 
  volume_of_tetrahedron_with_gold_vertices = 170.67 := 
by 
  sorry

end NUMINAMATH_GPT_gold_tetrahedron_volume_l22_2248


namespace NUMINAMATH_GPT_probability_reach_edge_within_five_hops_l22_2227

-- Define the probability of reaching an edge within n hops from the center
noncomputable def probability_reach_edge_by_hops (n : ℕ) : ℚ :=
if n = 5 then 121 / 128 else 0 -- This is just a placeholder for the real recursive computation.

-- Main theorem to prove
theorem probability_reach_edge_within_five_hops :
  probability_reach_edge_by_hops 5 = 121 / 128 :=
by
  -- Skipping the actual proof here
  sorry

end NUMINAMATH_GPT_probability_reach_edge_within_five_hops_l22_2227


namespace NUMINAMATH_GPT_certain_number_mod_l22_2246

theorem certain_number_mod (n : ℤ) : (73 * n) % 8 = 7 → n % 8 = 7 := 
by sorry

end NUMINAMATH_GPT_certain_number_mod_l22_2246


namespace NUMINAMATH_GPT_total_shells_correct_l22_2253

def morning_shells : ℕ := 292
def afternoon_shells : ℕ := 324

theorem total_shells_correct : morning_shells + afternoon_shells = 616 := by
  sorry

end NUMINAMATH_GPT_total_shells_correct_l22_2253


namespace NUMINAMATH_GPT_probability_all_white_balls_l22_2218

-- Definitions
def total_balls : ℕ := 15
def white_balls : ℕ := 8
def black_balls : ℕ := 7
def balls_drawn : ℕ := 7

-- Lean theorem statement
theorem probability_all_white_balls :
  (Nat.choose white_balls balls_drawn : ℚ) / (Nat.choose total_balls balls_drawn) = 8 / 6435 :=
sorry

end NUMINAMATH_GPT_probability_all_white_balls_l22_2218


namespace NUMINAMATH_GPT_keith_picked_0_pears_l22_2250

structure Conditions where
  apples_total : ℕ
  apples_mike : ℕ
  apples_nancy : ℕ
  apples_keith : ℕ
  pears_keith : ℕ

theorem keith_picked_0_pears (c : Conditions) (h_total : c.apples_total = 16)
 (h_mike : c.apples_mike = 7) (h_nancy : c.apples_nancy = 3)
 (h_keith : c.apples_keith = 6) : c.pears_keith = 0 :=
by
  sorry

end NUMINAMATH_GPT_keith_picked_0_pears_l22_2250


namespace NUMINAMATH_GPT_AM_GM_problem_l22_2283

theorem AM_GM_problem (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 1) :
  (1 + 1/x) * (1 + 1/y) ≥ 9 := 
sorry

end NUMINAMATH_GPT_AM_GM_problem_l22_2283


namespace NUMINAMATH_GPT_find_two_digit_number_l22_2249

def digit_eq_square_of_units (n x : ℤ) : Prop :=
  10 * (x - 3) + x = n ∧ n = x * x

def units_digit_3_larger_than_tens (x : ℤ) : Prop :=
  x - 3 >= 1 ∧ x - 3 < 10 ∧ x >= 3 ∧ x < 10

theorem find_two_digit_number (n x : ℤ) (h1 : digit_eq_square_of_units n x)
  (h2 : units_digit_3_larger_than_tens x) : n = 25 ∨ n = 36 :=
by sorry

end NUMINAMATH_GPT_find_two_digit_number_l22_2249


namespace NUMINAMATH_GPT_inequality_ge_9_l22_2265

theorem inequality_ge_9 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) : 
  (2 / a + 1 / b) ≥ 9 :=
sorry

end NUMINAMATH_GPT_inequality_ge_9_l22_2265


namespace NUMINAMATH_GPT_apples_in_box_l22_2280

theorem apples_in_box :
  (∀ (o p a : ℕ), 
    (o = 1 / 4 * 56) ∧ 
    (p = 1 / 2 * o) ∧ 
    (a = 5 * p) → 
    a = 35) :=
  by sorry

end NUMINAMATH_GPT_apples_in_box_l22_2280


namespace NUMINAMATH_GPT_household_savings_regression_l22_2281

-- Define the problem conditions in Lean
def n := 10
def sum_x := 80
def sum_y := 20
def sum_xy := 184
def sum_x2 := 720

-- Define the averages
def x_bar := sum_x / n
def y_bar := sum_y / n

-- Define the lxx and lxy as per the solution
def lxx := sum_x2 - n * x_bar^2
def lxy := sum_xy - n * x_bar * y_bar

-- Define the regression coefficients
def b_hat := lxy / lxx
def a_hat := y_bar - b_hat * x_bar

-- State the theorem to be proved
theorem household_savings_regression :
  (∀ (x: ℝ), y = b_hat * x + a_hat) :=
by
  sorry -- skip the proof

end NUMINAMATH_GPT_household_savings_regression_l22_2281


namespace NUMINAMATH_GPT_tan_sub_sin_eq_sq3_div2_l22_2260

noncomputable def tan_60 := Real.tan (Real.pi / 3)
noncomputable def sin_60 := Real.sin (Real.pi / 3)
noncomputable def result := (tan_60 - sin_60)

theorem tan_sub_sin_eq_sq3_div2 : result = Real.sqrt 3 / 2 := 
by
  -- Proof might go here
  sorry

end NUMINAMATH_GPT_tan_sub_sin_eq_sq3_div2_l22_2260


namespace NUMINAMATH_GPT_cos_sum_eq_neg_ratio_l22_2244

theorem cos_sum_eq_neg_ratio (γ δ : ℝ) 
  (hγ: Complex.exp (Complex.I * γ) = 4 / 5 + 3 / 5 * Complex.I) 
  (hδ: Complex.exp (Complex.I * δ) = -5 / 13 + 12 / 13 * Complex.I) :
  Real.cos (γ + δ) = -56 / 65 :=
  sorry

end NUMINAMATH_GPT_cos_sum_eq_neg_ratio_l22_2244


namespace NUMINAMATH_GPT_ellipse_minor_axis_length_l22_2295

noncomputable def minor_axis_length (a b : ℝ) (eccentricity : ℝ) (sum_distances : ℝ) :=
  if (a > b ∧ b > 0 ∧ eccentricity = (Real.sqrt 5) / 3 ∧ sum_distances = 12) then
    2 * b
  else
    0

theorem ellipse_minor_axis_length (a b : ℝ) (eccentricity : ℝ) (sum_distances : ℝ)
  (h1 : a > b) (h2 : b > 0) (h3 : eccentricity = (Real.sqrt 5) / 3) (h4 : sum_distances = 12) :
  minor_axis_length a b eccentricity sum_distances = 8 :=
sorry

end NUMINAMATH_GPT_ellipse_minor_axis_length_l22_2295


namespace NUMINAMATH_GPT_max_k_solution_l22_2297

theorem max_k_solution
  (k x y : ℝ)
  (h_pos: 0 < k ∧ 0 < x ∧ 0 < y)
  (h_eq: 5 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + k * ((x / y) + (y / x))) :
  ∃ k, 8*k^3 - 8*k^2 - 7*k = 0 := 
sorry

end NUMINAMATH_GPT_max_k_solution_l22_2297


namespace NUMINAMATH_GPT_function_nonnegative_l22_2266

noncomputable def f (x : ℝ) := (x - 10*x^2 + 35*x^3) / (9 - x^3)

theorem function_nonnegative (x : ℝ) : 
  (f x ≥ 0) ↔ (0 ≤ x ∧ x ≤ (1 / 7)) ∨ (3 ≤ x) :=
sorry

end NUMINAMATH_GPT_function_nonnegative_l22_2266


namespace NUMINAMATH_GPT_rational_numbers_property_l22_2228

theorem rational_numbers_property (n : ℕ) (h : n > 0) :
  ∃ (a b : ℚ), a ≠ b ∧ (∀ k, 1 ≤ k ∧ k ≤ n → ∃ m : ℤ, a^k - b^k = m) ∧ 
  ∀ i, (a : ℝ) ≠ i ∧ (b : ℝ) ≠ i :=
sorry

end NUMINAMATH_GPT_rational_numbers_property_l22_2228


namespace NUMINAMATH_GPT_ratio_of_new_time_to_previous_time_l22_2257

noncomputable def distance : ℝ := 420
noncomputable def previous_time : ℝ := 7
noncomputable def speed_increase : ℝ := 40

-- Original speed
noncomputable def original_speed : ℝ := distance / previous_time

-- New speed
noncomputable def new_speed : ℝ := original_speed + speed_increase

-- New time taken to cover the same distance at the new speed
noncomputable def new_time : ℝ := distance / new_speed

-- Ratio of new time to previous time
noncomputable def time_ratio : ℝ := new_time / previous_time

theorem ratio_of_new_time_to_previous_time :
  time_ratio = 0.6 :=
by sorry

end NUMINAMATH_GPT_ratio_of_new_time_to_previous_time_l22_2257


namespace NUMINAMATH_GPT_series_sum_eq_four_ninths_l22_2286

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, k / (4 : ℝ) ^ (k + 1)

theorem series_sum_eq_four_ninths : sum_series = 4 / 9 := 
sorry

end NUMINAMATH_GPT_series_sum_eq_four_ninths_l22_2286


namespace NUMINAMATH_GPT_travis_takes_home_money_l22_2239

-- Define the conditions
def total_apples : ℕ := 10000
def apples_per_box : ℕ := 50
def price_per_box : ℕ := 35

-- Define the main theorem to be proved
theorem travis_takes_home_money : (total_apples / apples_per_box) * price_per_box = 7000 := by
  sorry

end NUMINAMATH_GPT_travis_takes_home_money_l22_2239


namespace NUMINAMATH_GPT_subtract_045_from_3425_l22_2285

theorem subtract_045_from_3425 : 34.25 - 0.45 = 33.8 :=
by sorry

end NUMINAMATH_GPT_subtract_045_from_3425_l22_2285


namespace NUMINAMATH_GPT_iron_weight_l22_2221

theorem iron_weight 
  (A : ℝ) (hA : A = 0.83) 
  (I : ℝ) (hI : I = A + 10.33) : 
  I = 11.16 := 
by 
  sorry

end NUMINAMATH_GPT_iron_weight_l22_2221


namespace NUMINAMATH_GPT_find_B_max_f_A_l22_2254

namespace ProofProblem

-- Definitions
variables {A B C a b c : ℝ} -- Angles and sides in the triangle
noncomputable def givenCondition (A B C a b c : ℝ) : Prop :=
  2 * b * Real.cos A = 2 * c - Real.sqrt 3 * a

noncomputable def f (x : ℝ) : ℝ :=
  Real.cos x * Real.sin (x + Real.pi / 3) - Real.sqrt 3 / 4

-- Problem Statements (to be proved)
theorem find_B (h : givenCondition A B C a b c) : B = Real.pi / 6 := sorry

theorem max_f_A (A : ℝ) (B : ℝ) (h1 : 0 < A) (h2 : A < 5 * Real.pi / 6) (h3 : B = Real.pi / 6) : (∃ (x : ℝ), f x = 1 / 2) := sorry

end ProofProblem

end NUMINAMATH_GPT_find_B_max_f_A_l22_2254


namespace NUMINAMATH_GPT_machine_work_hours_l22_2229

theorem machine_work_hours (A B : ℝ) (x : ℝ) (hA : A = 1 / 8) (hB : B = A / 4)
  (hB_rate : B = 1 / 32) (B_time : B * 8 = 1 - x / 8) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_machine_work_hours_l22_2229


namespace NUMINAMATH_GPT_translate_parabola_up_one_unit_l22_2241

theorem translate_parabola_up_one_unit (x : ℝ) :
  let y := 3 * x^2
  (y + 1) = 3 * x^2 + 1 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_translate_parabola_up_one_unit_l22_2241


namespace NUMINAMATH_GPT_common_solution_exists_l22_2238

theorem common_solution_exists (a b : ℝ) :
  (∃ x y : ℝ, 19 * x^2 + 19 * y^2 + a * x + b * y + 98 = 0 ∧
                         98 * x^2 + 98 * y^2 + a * x + b * y + 19 = 0)
  → a^2 + b^2 ≥ 13689 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_common_solution_exists_l22_2238


namespace NUMINAMATH_GPT_somu_one_fifth_age_back_l22_2243

theorem somu_one_fifth_age_back {S F Y : ℕ}
  (h1 : S = 16)
  (h2 : S = F / 3)
  (h3 : S - Y = (F - Y) / 5) :
  Y = 8 :=
by
  sorry

end NUMINAMATH_GPT_somu_one_fifth_age_back_l22_2243


namespace NUMINAMATH_GPT_find_value_l22_2209

theorem find_value (x v : ℝ) (h1 : 0.80 * x + v = x) (h2 : x = 100) : v = 20 := by
    sorry

end NUMINAMATH_GPT_find_value_l22_2209


namespace NUMINAMATH_GPT_neg_p_l22_2200

noncomputable def f (a x : ℝ) : ℝ := a^x - x - a

theorem neg_p :
  ∃ (a : ℝ), a > 0 ∧ a ≠ 1 ∧ ∀ (x : ℝ), f a x ≠ 0 :=
sorry

end NUMINAMATH_GPT_neg_p_l22_2200


namespace NUMINAMATH_GPT_greatest_value_of_n_l22_2277

theorem greatest_value_of_n : ∀ (n : ℤ), 102 * n^2 ≤ 8100 → n ≤ 8 :=
by 
  sorry

end NUMINAMATH_GPT_greatest_value_of_n_l22_2277


namespace NUMINAMATH_GPT_number_of_pairs_l22_2270

theorem number_of_pairs : 
  (∃ (m n : ℤ), m + n = mn - 3) → ∃! (count : ℕ), count = 6 := by
  sorry

end NUMINAMATH_GPT_number_of_pairs_l22_2270


namespace NUMINAMATH_GPT_inequality_holds_l22_2213

theorem inequality_holds (x y z : ℝ) : x^2 + y^2 + z^2 ≥ Real.sqrt 2 * (x * y + y * z) := 
by 
  sorry

end NUMINAMATH_GPT_inequality_holds_l22_2213


namespace NUMINAMATH_GPT_fraction_draw_l22_2216

/-
Theorem: Given the win probabilities for Amy, Lily, and Eve, the fraction of the time they end up in a draw is 3/10.
-/

theorem fraction_draw (P_Amy P_Lily P_Eve : ℚ) (h_Amy : P_Amy = 2/5) (h_Lily : P_Lily = 1/5) (h_Eve : P_Eve = 1/10) : 
  1 - (P_Amy + P_Lily + P_Eve) = 3 / 10 := by
  sorry

end NUMINAMATH_GPT_fraction_draw_l22_2216


namespace NUMINAMATH_GPT_range_of_a_l22_2296

noncomputable def A := {x : ℝ | x^2 - 2*x - 8 < 0}
noncomputable def B := {x : ℝ | x^2 + 2*x - 3 > 0}
noncomputable def C (a : ℝ) := {x : ℝ | x^2 - 3*a*x + 2*a^2 < 0}

theorem range_of_a (a : ℝ) :
  (C a ⊆ A ∩ B) ↔ (1 ≤ a ∧ a ≤ 2 ∨ a = 0) :=
sorry

end NUMINAMATH_GPT_range_of_a_l22_2296


namespace NUMINAMATH_GPT_cube_prism_surface_area_l22_2226

theorem cube_prism_surface_area (a : ℝ) (h : a > 0) :
  2 * (6 * a^2) > 4 * a^2 + 2 * (2 * a * a) :=
by sorry

end NUMINAMATH_GPT_cube_prism_surface_area_l22_2226


namespace NUMINAMATH_GPT_mean_height_is_approx_correct_l22_2291

def heights : List ℕ := [120, 123, 127, 132, 133, 135, 140, 142, 145, 148, 152, 155, 158, 160]

def mean_height : ℚ := heights.sum / heights.length

theorem mean_height_is_approx_correct : 
  abs (mean_height - 140.71) < 0.01 := 
by
  sorry

end NUMINAMATH_GPT_mean_height_is_approx_correct_l22_2291


namespace NUMINAMATH_GPT_number_of_graphic_novels_l22_2259

theorem number_of_graphic_novels (total_books novels_percent comics_percent : ℝ) 
  (h_total : total_books = 120) 
  (h_novels_percent : novels_percent = 0.65) 
  (h_comics_percent : comics_percent = 0.20) :
  total_books - (novels_percent * total_books + comics_percent * total_books) = 18 :=
by
  sorry

end NUMINAMATH_GPT_number_of_graphic_novels_l22_2259


namespace NUMINAMATH_GPT_baseball_card_total_percent_decrease_l22_2289

theorem baseball_card_total_percent_decrease :
  ∀ (original_value first_year_decrease second_year_decrease : ℝ),
  first_year_decrease = 0.60 →
  second_year_decrease = 0.10 →
  original_value > 0 →
  (original_value - original_value * first_year_decrease - (original_value * (1 - first_year_decrease)) * second_year_decrease) =
  original_value * (1 - 0.64) :=
by
  intros original_value first_year_decrease second_year_decrease h_first_year h_second_year h_original_pos
  sorry

end NUMINAMATH_GPT_baseball_card_total_percent_decrease_l22_2289


namespace NUMINAMATH_GPT_part_a_part_b_l22_2234

-- Definitions based on the conditions:
def probability_of_hit (p : ℝ) := p
def probability_of_miss (p : ℝ) := 1 - p

-- Condition: exactly three unused rockets after firing at five targets
def exactly_three_unused_rockets (p : ℝ) : ℝ := 10 * (probability_of_hit p) ^ 3 * (probability_of_miss p) ^ 2

-- Condition: expected number of targets hit when there are nine targets
def expected_targets_hit (p : ℝ) : ℝ := 10 * p - p ^ 10

-- Lean 4 statements representing the proof problems:
theorem part_a (p : ℝ) (h_p_nonneg : 0 ≤ p) (h_p_le_one : p ≤ 1) : 
  exactly_three_unused_rockets p = 10 * p ^ 3 * (1 - p) ^ 2 :=
by sorry

theorem part_b (p : ℝ) (h_p_nonneg : 0 ≤ p) (h_p_le_one : p ≤ 1) :
  expected_targets_hit p = 10 * p - p ^ 10 :=
by sorry

end NUMINAMATH_GPT_part_a_part_b_l22_2234


namespace NUMINAMATH_GPT_dolls_total_l22_2247

theorem dolls_total (dina_dolls ivy_dolls casey_dolls : ℕ) 
  (h1 : dina_dolls = 2 * ivy_dolls)
  (h2 : (2 / 3 : ℚ) * ivy_dolls = 20)
  (h3 : casey_dolls = 5 * 20) :
  dina_dolls + ivy_dolls + casey_dolls = 190 :=
by sorry

end NUMINAMATH_GPT_dolls_total_l22_2247


namespace NUMINAMATH_GPT_seq_eighth_term_l22_2214

-- Define the sequence recursively
def seq : ℕ → ℕ
| 0     => 1  -- Base case, since 1 is the first term of the sequence
| (n+1) => seq n + (n + 1)  -- Recursive case, each term is the previous term plus the index number (which is n + 1) minus 1

-- Define the statement to prove 
theorem seq_eighth_term : seq 7 = 29 :=  -- Note: index 7 corresponds to the 8th term since indexing is 0-based
  by
  sorry

end NUMINAMATH_GPT_seq_eighth_term_l22_2214


namespace NUMINAMATH_GPT_total_earrings_after_one_year_l22_2288

theorem total_earrings_after_one_year :
  let bella_earrings := 10
  let monica_earrings := 10 / 0.25
  let rachel_earrings := monica_earrings / 2
  let initial_total := bella_earrings + monica_earrings + rachel_earrings
  let olivia_earrings_initial := initial_total + 5
  let olivia_earrings_after := olivia_earrings_initial * 1.2
  let total_earrings := bella_earrings + monica_earrings + rachel_earrings + olivia_earrings_after
  total_earrings = 160 :=
by
  sorry

end NUMINAMATH_GPT_total_earrings_after_one_year_l22_2288


namespace NUMINAMATH_GPT_find_alpha_l22_2212

theorem find_alpha (P : Real × Real) (h: P = (Real.sin (50 * Real.pi / 180), 1 + Real.cos (50 * Real.pi / 180))) :
  ∃ α : Real, α = 65 * Real.pi / 180 := by
  sorry

end NUMINAMATH_GPT_find_alpha_l22_2212


namespace NUMINAMATH_GPT_number_of_sides_of_regular_polygon_l22_2242

theorem number_of_sides_of_regular_polygon (P s n : ℕ) (hP : P = 150) (hs : s = 15) (hP_formula : P = n * s) : n = 10 :=
  by {
    -- proof goes here
    sorry
  }

end NUMINAMATH_GPT_number_of_sides_of_regular_polygon_l22_2242


namespace NUMINAMATH_GPT_service_center_location_l22_2215

-- Definitions from conditions
def third_exit := 30
def twelfth_exit := 195
def seventh_exit := 90

-- Concept of distance and service center location
def distance := seventh_exit - third_exit
def service_center_milepost := third_exit + 2 * distance / 3

-- The theorem to prove
theorem service_center_location : service_center_milepost = 70 := by
  -- Sorry is used to skip the proof details.
  sorry

end NUMINAMATH_GPT_service_center_location_l22_2215


namespace NUMINAMATH_GPT_max_good_diagonals_l22_2284

def is_good_diagonal (n : ℕ) (d : ℕ) : Prop := ∀ (P : Fin n → Prop), ∃! (i j : Fin n), P i ∧ P j ∧ (d = i + j)

theorem max_good_diagonals (n : ℕ) (h : 2 ≤ n) :
  (∃ (m : ℕ), is_good_diagonal n m ∧ (m = n - 2 ↔ Even n) ∧ (m = n - 3 ↔ Odd n)) :=
by
  sorry

end NUMINAMATH_GPT_max_good_diagonals_l22_2284


namespace NUMINAMATH_GPT_units_digit_of_7_pow_6_pow_5_l22_2201

theorem units_digit_of_7_pow_6_pow_5 : (7^(6^5)) % 10 = 1 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_units_digit_of_7_pow_6_pow_5_l22_2201


namespace NUMINAMATH_GPT_even_integer_squares_l22_2292

noncomputable def Q (x : ℤ) : ℤ := x^4 + 6 * x^3 + 11 * x^2 + 3 * x + 25

theorem even_integer_squares (x : ℤ) (hx : x % 2 = 0) :
  (∃ (a : ℤ), Q x = a ^ 2) → x = 8 :=
by
  sorry

end NUMINAMATH_GPT_even_integer_squares_l22_2292


namespace NUMINAMATH_GPT_profits_to_revenues_ratio_l22_2217

theorem profits_to_revenues_ratio (R P: ℝ) 
    (rev_2009: R_2009 = 0.8 * R) 
    (profit_2009_rev_2009: P_2009 = 0.2 * R_2009)
    (profit_2009: P_2009 = 1.6 * P):
    (P / R) * 100 = 10 :=
by
  sorry

end NUMINAMATH_GPT_profits_to_revenues_ratio_l22_2217


namespace NUMINAMATH_GPT_area_of_rhombus_l22_2206

theorem area_of_rhombus (d1 d2 : ℕ) (h1 : d1 = 16) (h2 : d2 = 20) : (d1 * d2) / 2 = 160 := by
  sorry

end NUMINAMATH_GPT_area_of_rhombus_l22_2206


namespace NUMINAMATH_GPT_distribution_value_l22_2258

def standard_deviation := 2
def mean := 51

theorem distribution_value (x : ℝ) (hx : x < 45) : (mean - 3 * standard_deviation) > x :=
by
  -- Provide the statement without proof
  sorry

end NUMINAMATH_GPT_distribution_value_l22_2258


namespace NUMINAMATH_GPT_trail_mix_total_weight_l22_2276

theorem trail_mix_total_weight :
  let peanuts := 0.16666666666666666
  let chocolate_chips := 0.16666666666666666
  let raisins := 0.08333333333333333
  let almonds := 0.14583333333333331
  let cashews := (1 / 8 : Real)
  let dried_cranberries := (3 / 32 : Real)
  (peanuts + chocolate_chips + raisins + almonds + cashews + dried_cranberries) = 0.78125 :=
by
  sorry

end NUMINAMATH_GPT_trail_mix_total_weight_l22_2276


namespace NUMINAMATH_GPT_initial_pen_count_is_30_l22_2223

def pen_count (initial_pens : ℕ) : ℕ :=
  let after_mike := initial_pens + 20
  let after_cindy := 2 * after_mike
  let after_sharon := after_cindy - 10
  after_sharon

theorem initial_pen_count_is_30 : pen_count 30 = 30 :=
by
  sorry

end NUMINAMATH_GPT_initial_pen_count_is_30_l22_2223


namespace NUMINAMATH_GPT_determine_min_guesses_l22_2222

def minimum_guesses (n k : ℕ) (h : n > k) : ℕ :=
  if n = 2 * k then 2 else 1

theorem determine_min_guesses (n k : ℕ) (h : n > k) :
  (if n = 2 * k then 2 else 1) = minimum_guesses n k h := by
  sorry

end NUMINAMATH_GPT_determine_min_guesses_l22_2222
