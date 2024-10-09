import Mathlib

namespace transistor_length_scientific_notation_l381_38102

theorem transistor_length_scientific_notation :
  0.000000006 = 6 * 10^(-9) := 
sorry

end transistor_length_scientific_notation_l381_38102


namespace even_function_derivative_at_zero_l381_38141

variable (f : ℝ → ℝ)
variable (hf_even : ∀ x, f x = f (-x))
variable (hf_diff : Differentiable ℝ f)

theorem even_function_derivative_at_zero : deriv f 0 = 0 :=
by 
  -- proof omitted
  sorry

end even_function_derivative_at_zero_l381_38141


namespace problem_solution_l381_38186

variable {a b x y : ℝ}

-- Define the conditions as Lean assumptions
axiom cond1 : a * x + b * y = 3
axiom cond2 : a * x^2 + b * y^2 = 7
axiom cond3 : a * x^3 + b * y^3 = 16
axiom cond4 : a * x^4 + b * y^4 = 42

-- The main theorem statement: under these conditions, prove a * x^5 + b * y^5 = 99
theorem problem_solution : a * x^5 + b * y^5 = 99 := 
sorry -- proof omitted

end problem_solution_l381_38186


namespace sum_opposite_sign_zero_l381_38198

def opposite_sign (a b : ℝ) : Prop :=
(a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0)

theorem sum_opposite_sign_zero {a b : ℝ} (h : opposite_sign a b) : a + b = 0 :=
sorry

end sum_opposite_sign_zero_l381_38198


namespace find_quadrant_372_degrees_l381_38111

theorem find_quadrant_372_degrees : 
  ∃ q : ℕ, q = 1 ↔ (372 % 360 = 12 ∧ (0 ≤ 12 ∧ 12 < 90)) :=
by
  sorry

end find_quadrant_372_degrees_l381_38111


namespace range_of_a_l381_38175

noncomputable def f (a x : ℝ) : ℝ := x^3 - a * x^2 + 4

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x > 0 → f a x = 0 → (f a x = 0 → x > 0)) ↔ a > 3 := sorry

end range_of_a_l381_38175


namespace find_m_value_l381_38173

def vector := (ℝ × ℝ)

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def is_perpendicular (v1 v2 : vector) : Prop :=
  dot_product v1 v2 = 0

theorem find_m_value (a b : vector) (m : ℝ) (h: a = (2, -1)) (h2: b = (1, 3))
  (h3: is_perpendicular a (a.1 + m * b.1, a.2 + m * b.2)) : m = 5 :=
sorry

end find_m_value_l381_38173


namespace cost_of_fencing_l381_38126

theorem cost_of_fencing (d : ℝ) (rate : ℝ) (C : ℝ) (cost : ℝ) : 
  d = 22 → rate = 3 → C = Real.pi * d → cost = C * rate → cost = 207 :=
by
  intros
  sorry

end cost_of_fencing_l381_38126


namespace min_value_of_sum_eq_l381_38147

theorem min_value_of_sum_eq : ∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a + b = a * b - 1 → a + 2 * b = 5 + 2 * Real.sqrt 6 :=
by
  intros a b h
  sorry

end min_value_of_sum_eq_l381_38147


namespace square_same_area_as_rectangle_l381_38129

theorem square_same_area_as_rectangle (l w : ℝ) (rect_area sq_side : ℝ) :
  l = 25 → w = 9 → rect_area = l * w → sq_side^2 = rect_area → sq_side = 15 :=
by
  intros h_l h_w h_rect_area h_sq_area
  rw [h_l, h_w] at h_rect_area
  sorry

end square_same_area_as_rectangle_l381_38129


namespace sticker_price_of_laptop_l381_38140

variable (x : ℝ)

-- Conditions
noncomputable def price_store_A : ℝ := 0.90 * x - 100
noncomputable def price_store_B : ℝ := 0.80 * x
noncomputable def savings : ℝ := price_store_B x - price_store_A x

-- Theorem statement
theorem sticker_price_of_laptop (x : ℝ) (h : savings x = 20) : x = 800 :=
by
  sorry

end sticker_price_of_laptop_l381_38140


namespace sequence_remainder_4_l381_38133

def sequence_of_numbers (n : ℕ) : ℕ :=
  7 * n + 4

theorem sequence_remainder_4 (n : ℕ) : (sequence_of_numbers n) % 7 = 4 := by
  sorry

end sequence_remainder_4_l381_38133


namespace company_l381_38118

-- Define conditions
def initial_outlay : ℝ := 10000

def material_cost_per_set_first_300 : ℝ := 20
def material_cost_per_set_beyond_300 : ℝ := 15

def exchange_rate : ℝ := 1.1

def import_tax_rate : ℝ := 0.10

def sales_price_per_set_first_400 : ℝ := 50
def sales_price_per_set_beyond_400 : ℝ := 45

def export_tax_threshold : ℕ := 500
def export_tax_rate : ℝ := 0.05

def production_and_sales : ℕ := 800

-- Helper functions for the problem
def material_cost_first_300_sets : ℝ :=
  300 * material_cost_per_set_first_300 * exchange_rate

def material_cost_next_500_sets : ℝ :=
  (production_and_sales - 300) * material_cost_per_set_beyond_300 * exchange_rate

def total_material_cost : ℝ :=
  material_cost_first_300_sets + material_cost_next_500_sets

def import_tax : ℝ := total_material_cost * import_tax_rate

def total_manufacturing_cost : ℝ :=
  initial_outlay + total_material_cost + import_tax

def sales_revenue_first_400_sets : ℝ :=
  400 * sales_price_per_set_first_400

def sales_revenue_next_400_sets : ℝ :=
  (production_and_sales - 400) * sales_price_per_set_beyond_400

def total_sales_revenue_before_export_tax : ℝ :=
  sales_revenue_first_400_sets + sales_revenue_next_400_sets

def sales_revenue_beyond_threshold : ℝ :=
  (production_and_sales - export_tax_threshold) * sales_price_per_set_beyond_400

def export_tax : ℝ := sales_revenue_beyond_threshold * export_tax_rate

def total_sales_revenue_after_export_tax : ℝ :=
  total_sales_revenue_before_export_tax - export_tax

def profit : ℝ :=
  total_sales_revenue_after_export_tax - total_manufacturing_cost

-- Lean 4 statement for the proof problem
theorem company's_profit_is_10990 :
  profit = 10990 := by
  sorry

end company_l381_38118


namespace fg_value_l381_38119

def g (x : ℕ) : ℕ := 4 * x + 10
def f (x : ℕ) : ℕ := 6 * x - 12

theorem fg_value : f (g 10) = 288 := by
  sorry

end fg_value_l381_38119


namespace two_packs_remainder_l381_38124

theorem two_packs_remainder (m : ℕ) (h : m % 7 = 5) : (2 * m) % 7 = 3 :=
by {
  sorry
}

end two_packs_remainder_l381_38124


namespace simplify_expression_l381_38159

-- Define the variables x and y
variables (x y : ℝ)

-- State the theorem
theorem simplify_expression (x y : ℝ) (hy : y ≠ 0) :
  ((x + 3 * y)^2 - (x + y) * (x - y)) / (2 * y) = 3 * x + 5 * y := 
by 
  -- skip the proof
  sorry

end simplify_expression_l381_38159


namespace tangent_line_circle_l381_38164

theorem tangent_line_circle (m : ℝ) :
  (∀ x y : ℝ, x - 2*y + m = 0 ↔ (x^2 + y^2 - 4*x + 6*y + 8 = 0)) →
  m = -3 ∨ m = -13 :=
sorry

end tangent_line_circle_l381_38164


namespace lily_pad_cover_entire_lake_l381_38105

-- Definitions per the conditions
def doublesInSizeEveryDay (P : ℕ → ℝ) : Prop :=
  ∀ n, P (n + 1) = 2 * P n

-- The initial state that it takes 36 days to cover the lake
def coversEntireLakeIn36Days (P : ℕ → ℝ) (L : ℝ) : Prop :=
  P 36 = L

-- The main theorem to prove
theorem lily_pad_cover_entire_lake (P : ℕ → ℝ) (L : ℝ) (h1 : doublesInSizeEveryDay P) (h2 : coversEntireLakeIn36Days P L) :
  ∃ n, n = 36 := 
by
  sorry

end lily_pad_cover_entire_lake_l381_38105


namespace arithmetic_sequence_sum_S9_l381_38193

variable {a : ℕ → ℝ} -- Define the arithmetic sequence
variable {S : ℕ → ℝ} -- Define the sum sequence

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := ∀ n, a (n + 1) = a n + d
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop := ∀ n, S n = n * (a 1 + a n) / 2

-- Problem statement in Lean
theorem arithmetic_sequence_sum_S9 (h_seq : ∃ d, arithmetic_sequence a d) (h_a2 : a 2 = -2) (h_a8 : a 8 = 6) (h_S_def : sum_of_first_n_terms a S) : S 9 = 18 := 
by {
  sorry
}

end arithmetic_sequence_sum_S9_l381_38193


namespace train_length_correct_l381_38142

noncomputable def train_length (time : ℝ) (platform_length : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  let total_distance := speed_mps * time
  total_distance - platform_length

theorem train_length_correct :
  train_length 17.998560115190784 200 90 = 249.9640028797696 :=
by
  sorry

end train_length_correct_l381_38142


namespace last_passenger_probability_l381_38151

noncomputable def probability_last_passenger_gets_seat {n : ℕ} (h : n > 0) : ℚ :=
  if n = 1 then 1 else 1/2

theorem last_passenger_probability
  (n : ℕ) (h : n > 0) :
  probability_last_passenger_gets_seat h = 1/2 :=
  sorry

end last_passenger_probability_l381_38151


namespace proportion_of_capacity_filled_l381_38104

noncomputable def milk_proportion_8cup_bottle : ℚ := 16 / 3
noncomputable def total_milk := 8

theorem proportion_of_capacity_filled :
  ∃ p : ℚ, (8 * p = milk_proportion_8cup_bottle) ∧ (4 * p = total_milk - milk_proportion_8cup_bottle) ∧ (p = 2 / 3) :=
by
  sorry

end proportion_of_capacity_filled_l381_38104


namespace intersection_eq_set_l381_38158

-- Define set A based on the inequality
def A : Set ℝ := {x | x^2 + x - 2 < 0}

-- Define set B based on the inequality
def B : Set ℝ := {x | 0 ≤ Real.log (x + 1) / Real.log 2 ∧ Real.log (x + 1) / Real.log 2 < 2}

-- Translate the question to a lean theorem
theorem intersection_eq_set : (A ∩ B) = {x | 0 ≤ x ∧ x < 1} := 
sorry

end intersection_eq_set_l381_38158


namespace number_of_dimes_l381_38117

theorem number_of_dimes (x : ℕ) (h1 : 10 * x + 25 * x + 50 * x = 2040) : x = 24 :=
by {
  -- The proof will go here if you need to fill it out.
  sorry
}

end number_of_dimes_l381_38117


namespace p_necessary_for_q_l381_38176

-- Definitions
def p (a b : ℝ) : Prop := (a + b = 2) ∨ (a + b = -2)
def q (a b : ℝ) : Prop := a + b = 2

-- Statement of the problem
theorem p_necessary_for_q (a b : ℝ) : (p a b → q a b) ∧ ¬(q a b → p a b) := 
sorry

end p_necessary_for_q_l381_38176


namespace three_x_minus_five_y_l381_38168

noncomputable def F : ℝ × ℝ :=
  let D := (15, 3)
  let E := (6, 8)
  ((D.1 + E.1) / 2, (D.2 + E.2) / 2)

theorem three_x_minus_five_y : (3 * F.1 - 5 * F.2) = 4 := by
  sorry

end three_x_minus_five_y_l381_38168


namespace cost_of_each_sale_puppy_l381_38107

-- Conditions
def total_cost (total: ℚ) : Prop := total = 800
def non_sale_puppy_cost (cost: ℚ) : Prop := cost = 175
def num_puppies (num: ℕ) : Prop := num = 5

-- Question to Prove
theorem cost_of_each_sale_puppy (total cost : ℚ) (num: ℕ):
  total_cost total →
  non_sale_puppy_cost cost →
  num_puppies num →
  (total - 2 * cost) / (num - 2) = 150 := 
sorry

end cost_of_each_sale_puppy_l381_38107


namespace savings_calculation_l381_38137

noncomputable def weekly_rate_peak : ℕ := 10
noncomputable def weekly_rate_non_peak : ℕ := 8
noncomputable def monthly_rate_peak : ℕ := 40
noncomputable def monthly_rate_non_peak : ℕ := 35
noncomputable def non_peak_duration_weeks : ℝ := 17.33
noncomputable def peak_duration_weeks : ℝ := 52 - non_peak_duration_weeks
noncomputable def non_peak_duration_months : ℕ := 4
noncomputable def peak_duration_months : ℕ := 12 - non_peak_duration_months

noncomputable def total_weekly_cost := (non_peak_duration_weeks * weekly_rate_non_peak) 
                                     + (peak_duration_weeks * weekly_rate_peak)

noncomputable def total_monthly_cost := (non_peak_duration_months * monthly_rate_non_peak) 
                                      + (peak_duration_months * monthly_rate_peak)

noncomputable def savings := total_weekly_cost - total_monthly_cost

theorem savings_calculation 
  : savings = 25.34 := by
  sorry

end savings_calculation_l381_38137


namespace total_people_in_church_l381_38188

def c : ℕ := 80
def m : ℕ := 60
def f : ℕ := 60

theorem total_people_in_church : c + m + f = 200 :=
by
  sorry

end total_people_in_church_l381_38188


namespace area_of_triangle_l381_38113

theorem area_of_triangle (a b c : ℝ) (C : ℝ) 
  (h1 : c^2 = (a - b)^2 + 6)
  (h2 : C = Real.pi / 3) 
  : (1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2) :=
sorry

end area_of_triangle_l381_38113


namespace problem_statement_l381_38150

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4, 5}

theorem problem_statement : ((U \ A) ∪ (U \ B)) = {0, 1, 3, 4, 5} := by
  sorry

end problem_statement_l381_38150


namespace min_value_fraction_l381_38185

theorem min_value_fraction (x y : ℝ) 
  (h1 : x - 1 ≥ 0)
  (h2 : x - y + 1 ≤ 0)
  (h3 : x + y - 4 ≤ 0) : 
  ∃ a, (∀ x y, (x - 1 ≥ 0) ∧ (x - y + 1 ≤ 0) ∧ (x + y - 4 ≤ 0) → (x / (y + 1)) ≥ a) ∧ 
      (a = 1 / 4) :=
sorry

end min_value_fraction_l381_38185


namespace shopkeeper_percentage_above_cost_l381_38167

theorem shopkeeper_percentage_above_cost (CP MP SP : ℚ) 
  (h1 : CP = 100) 
  (h2 : SP = CP * 1.02)
  (h3 : SP = MP * 0.85) : 
  (MP - CP) / CP * 100 = 20 :=
by sorry

end shopkeeper_percentage_above_cost_l381_38167


namespace math_problem_proof_l381_38125

theorem math_problem_proof (a b x y : ℝ) 
  (h1: x = a) 
  (h2: y = b)
  (h3: a + a = b * a)
  (h4: y = a)
  (h5: a * a = a + a)
  (h6: b = 3) : 
  x * y = 4 := 
by 
  sorry

end math_problem_proof_l381_38125


namespace erica_pie_percentage_l381_38121

theorem erica_pie_percentage (a c : ℚ) (ha : a = 1/5) (hc : c = 3/4) : 
  (a + c) * 100 = 95 := 
sorry

end erica_pie_percentage_l381_38121


namespace friends_gcd_l381_38163

theorem friends_gcd {a b : ℤ} (h : ∃ n : ℤ, a * b = n * n) : 
  ∃ m : ℤ, a * Int.gcd a b = m * m :=
sorry

end friends_gcd_l381_38163


namespace sufficient_but_not_necessary_condition_l381_38172

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (m = -2) → (∀ x y, ((m + 2) * x + m * y + 1 = 0) ∧ ((m - 2) * x + (m + 2) * y - 3 = 0) → (m = 1) ∨ (m = -2)) → (m = -2) → (∀ x y, ((m + 2) * x + m * y + 1 = 0) ∧ ((m - 2) * x + (m + 2) * y - 3 = 0) → false) :=
by
  intros hm h_perp h
  sorry

end sufficient_but_not_necessary_condition_l381_38172


namespace determine_functions_l381_38179

noncomputable def functional_eq_condition (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f x + f y) = y + (f x) ^ 2

theorem determine_functions (f : ℝ → ℝ) (h : functional_eq_condition f) : 
  (∀ x, f x = x) ∨ (∀ x, f x = -x) :=
sorry

end determine_functions_l381_38179


namespace trig_product_identity_l381_38122

theorem trig_product_identity :
  (1 + Real.sin (Real.pi / 12)) * (1 + Real.sin (5 * Real.pi / 12)) *
  (1 + Real.sin (7 * Real.pi / 12)) * (1 + Real.sin (11 * Real.pi / 12)) =
  (1 + Real.sin (Real.pi / 12))^2 * (1 + Real.sin (5 * Real.pi / 12))^2 :=
by
  sorry

end trig_product_identity_l381_38122


namespace podium_height_l381_38191

theorem podium_height (l w h : ℝ) (r s : ℝ) (H1 : r = l + h - w) (H2 : s = w + h - l) 
  (Hr : r = 40) (Hs : s = 34) : h = 37 :=
by
  sorry

end podium_height_l381_38191


namespace ratio_of_books_sold_l381_38109

theorem ratio_of_books_sold
  (T W R : ℕ)
  (hT : T = 7)
  (hW : W = 3 * T)
  (hTotal : T + W + R = 91) :
  R / W = 3 :=
by
  sorry

end ratio_of_books_sold_l381_38109


namespace broken_seashells_count_l381_38157

def total_seashells : ℕ := 7
def unbroken_seashells : ℕ := 3

theorem broken_seashells_count : (total_seashells - unbroken_seashells) = 4 := by
  sorry

end broken_seashells_count_l381_38157


namespace necessary_but_not_sufficient_for_inequality_l381_38145

variables (a b : ℝ)

theorem necessary_but_not_sufficient_for_inequality (h : a ≠ b) (hab_pos : a * b > 0) :
  (b/a + a/b > 2) :=
sorry

end necessary_but_not_sufficient_for_inequality_l381_38145


namespace bucket_volume_l381_38103

theorem bucket_volume :
  ∃ (V : ℝ), -- The total volume of the bucket
    (∀ (rate_A rate_B rate_combined : ℝ),
      rate_A = 3 ∧ 
      rate_B = V / 60 ∧ 
      rate_combined = V / 10 ∧ 
      rate_A + rate_B = rate_combined) →
    V = 36 :=
by
  sorry

end bucket_volume_l381_38103


namespace no_positive_integer_solution_l381_38190

/-- Let \( p \) be a prime greater than 3 and \( x \) be an integer such that \( p \) divides \( x \).
    Then the equation \( x^2 - 1 = y^p \) has no positive integer solutions for \( y \). -/
theorem no_positive_integer_solution {p x y : ℕ} (hp : Nat.Prime p) (hgt : 3 < p) (hdiv : p ∣ x) :
  ¬∃ y : ℕ, (x^2 - 1 = y^p) ∧ (0 < y) :=
by
  sorry

end no_positive_integer_solution_l381_38190


namespace linear_coefficient_of_quadratic_term_is_negative_five_l381_38183

theorem linear_coefficient_of_quadratic_term_is_negative_five (a b c : ℝ) (x : ℝ) :
  (2 * x^2 = 5 * x - 3) →
  (a = 2) →
  (b = -5) →
  (c = 3) →
  (a * x^2 + b * x + c = 0) :=
by
  sorry

end linear_coefficient_of_quadratic_term_is_negative_five_l381_38183


namespace min_value_correct_l381_38181

noncomputable def min_value (a b x y : ℝ) [Fact (a > 0)] [Fact (b > 0)] [Fact (x > 0)] [Fact (y > 0)] : ℝ :=
  if x + y = 1 then (a / x + b / y) else 0

theorem min_value_correct (a b x y : ℝ) [Fact (a > 0)] [Fact (b > 0)] [Fact (x > 0)] [Fact (y > 0)]
  (h : x + y = 1) : min_value a b x y = (Real.sqrt a + Real.sqrt b)^2 :=
by
  sorry

end min_value_correct_l381_38181


namespace inequality_solution_set_l381_38184

theorem inequality_solution_set (x : ℝ) (h : x ≠ 0) : 
  (1 / x > 3) ↔ (0 < x ∧ x < 1 / 3) := 
by 
  sorry

end inequality_solution_set_l381_38184


namespace price_per_ton_max_tons_l381_38169

variable (x y m : ℝ)

def conditions := x = y + 100 ∧ 2 * x + y = 1700

theorem price_per_ton (h : conditions x y) : x = 600 ∧ y = 500 :=
  sorry

def budget_conditions := 10 * (600 - 100) + 1 * 500 ≤ 5600

theorem max_tons (h : budget_conditions) : 600 * m + 500 * (10 - m) ≤ 5600 → m ≤ 6 :=
  sorry

end price_per_ton_max_tons_l381_38169


namespace arithmetic_expression_evaluation_l381_38132

theorem arithmetic_expression_evaluation :
  4 * (7 * 24) / 3 + 5 * (13 * 15) - 2 * (6 * 28) + 7 * (3 * 19) / 2 = 1062.5 := 
by
  -- Skipping the proof.
  sorry

end arithmetic_expression_evaluation_l381_38132


namespace negative_number_zero_exponent_l381_38156

theorem negative_number_zero_exponent (a : ℤ) (h : a ≠ 0) : a^0 = 1 :=
by sorry

end negative_number_zero_exponent_l381_38156


namespace intersection_M_N_l381_38149

def M (x : ℝ) : Prop := -2 < x ∧ x < 2
def N (x : ℝ) : Prop := |x - 1| ≤ 2

theorem intersection_M_N :
  {x : ℝ | M x ∧ N x} = {x : ℝ | -1 ≤ x ∧ x < 2} :=
sorry

end intersection_M_N_l381_38149


namespace student_most_stable_l381_38197

theorem student_most_stable (A B C : ℝ) (hA : A = 0.024) (hB : B = 0.08) (hC : C = 0.015) : C < A ∧ C < B := by
  sorry

end student_most_stable_l381_38197


namespace max_ratio_xy_l381_38162

def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem max_ratio_xy (x y : ℕ) (hx : two_digit x) (hy : two_digit y) (hmean : (x + y) / 2 = 60) : x / y ≤ 33 / 7 :=
by
  sorry

end max_ratio_xy_l381_38162


namespace equilibrium_table_n_max_l381_38195

theorem equilibrium_table_n_max (table : Fin 2010 → Fin 2010 → ℕ) :
  (∃ n, ∀ (i j k l : Fin 2010),
      table i j + table k l = table i l + table k j ∧
      ∀ m ≤ n, (m = 0 ∨ m = 1)
  ) → n = 1 ∧ table (Fin.mk 0 (by norm_num)) (Fin.mk 0 (by norm_num)) = 2 :=
by
  sorry

end equilibrium_table_n_max_l381_38195


namespace simon_age_is_10_l381_38108

def alvin_age : ℕ := 30

def simon_age (alvin_age : ℕ) : ℕ :=
  (alvin_age / 2) - 5

theorem simon_age_is_10 : simon_age alvin_age = 10 := by
  sorry

end simon_age_is_10_l381_38108


namespace arctan_addition_formula_l381_38135

noncomputable def arctan_add : ℝ :=
  Real.arctan (1 / 3) + Real.arctan (3 / 8)

theorem arctan_addition_formula :
  arctan_add = Real.arctan (17 / 21) :=
by
  sorry

end arctan_addition_formula_l381_38135


namespace sum_eq_neg_20_div_3_l381_38100
-- Import the necessary libraries

-- The main theoretical statement
theorem sum_eq_neg_20_div_3
    (a b c d : ℝ)
    (h : a + 2 = b + 4 ∧ b + 4 = c + 6 ∧ c + 6 = d + 8 ∧ d + 8 = a + b + c + d + 10) :
    a + b + c + d = -20 / 3 :=
by
  sorry

end sum_eq_neg_20_div_3_l381_38100


namespace union_of_sets_l381_38110

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

-- State the proof problem
theorem union_of_sets : A ∪ B = {x | -1 < x ∧ x < 3} := by
  sorry

end union_of_sets_l381_38110


namespace subtract_from_sum_base8_l381_38116

def add_in_base_8 (a b : ℕ) : ℕ :=
  ((a % 8) + (b % 8)) % 8
  + (((a / 8) % 8 + (b / 8) % 8 + ((a % 8) + (b % 8)) / 8) % 8) * 8
  + (((((a / 8) % 8 + (b / 8) % 8 + ((a % 8) + (b % 8)) / 8) / 8) + ((a / 64) % 8 + (b / 64) % 8)) % 8) * 64

def subtract_in_base_8 (a b : ℕ) : ℕ :=
  ((a % 8) - (b % 8) + 8) % 8
  + (((a / 8) % 8 - (b / 8) % 8 - if (a % 8) < (b % 8) then 1 else 0 + 8) % 8) * 8
  + (((a / 64) - (b / 64) - if (a / 8) % 8 < (b / 8) % 8 then 1 else 0) % 8) * 64

theorem subtract_from_sum_base8 :
  subtract_in_base_8 (add_in_base_8 652 147) 53 = 50 := by
  sorry

end subtract_from_sum_base8_l381_38116


namespace heath_plants_per_hour_l381_38182

theorem heath_plants_per_hour (rows : ℕ) (plants_per_row : ℕ) (hours : ℕ) (total_plants : ℕ) :
  rows = 400 ∧ plants_per_row = 300 ∧ hours = 20 ∧ total_plants = rows * plants_per_row →
  total_plants / hours = 6000 :=
by
  sorry

end heath_plants_per_hour_l381_38182


namespace range_of_a_l381_38136

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (x ∈ {x : ℝ | x ≥ 3 ∨ x ≤ -1} ∩ {x : ℝ | x ≤ a} ↔ x ∈ {x : ℝ | x ≤ a})) ↔ a ≤ -1 :=
by sorry

end range_of_a_l381_38136


namespace smallest_sum_l381_38192

-- First, we define the conditions as assumptions:
def is_arithmetic_sequence (x y z : ℕ) : Prop :=
  2 * y = x + z

def is_geometric_sequence (x y z : ℕ) : Prop :=
  y ^ 2 = x * z

-- Given conditions
variables (A B C D : ℕ)
variables (hABC : is_arithmetic_sequence A B C) (hBCD : is_geometric_sequence B C D)
variables (h_ratio : 4 * C = 7 * B)

-- The main theorem to prove
theorem smallest_sum : A + B + C + D = 97 :=
sorry

end smallest_sum_l381_38192


namespace space_mission_contribution_l381_38112

theorem space_mission_contribution 
  (mission_cost_million : ℕ := 30000) 
  (combined_population_million : ℕ := 350) : 
  mission_cost_million / combined_population_million = 86 := by
  sorry

end space_mission_contribution_l381_38112


namespace transformed_curve_eq_l381_38154

-- Define the original ellipse curve
def ellipse (x y : ℝ) : Prop := (x^2) / 4 + y^2 = 1

-- Define the transformation
def transform (x y x' y' : ℝ) : Prop := x' = x ∧ y' = 2 * y

-- Prove the transformed curve satisfies x'^2 + y'^2 = 4
theorem transformed_curve_eq :
  ∀ (x y x' y' : ℝ), ellipse x y → transform x y x' y' → (x'^2 + y'^2 = 4) :=
by
  intros x y x' y' h_ellipse h_transform
  simp [ellipse, transform] at *
  sorry

end transformed_curve_eq_l381_38154


namespace cleaning_task_sequences_correct_l381_38170

section ChemistryClass

-- Total number of students
def total_students : ℕ := 15

-- Number of classes in a week
def classes_per_week : ℕ := 5

-- Calculate the number of valid sequences of task assignments
def num_valid_sequences : ℕ := total_students * (total_students - 1) * (total_students - 2) * (total_students - 3) * (total_students - 4)

theorem cleaning_task_sequences_correct :
  num_valid_sequences = 360360 :=
by
  unfold num_valid_sequences
  norm_num
  sorry

end ChemistryClass

end cleaning_task_sequences_correct_l381_38170


namespace fixed_costs_16699_50_l381_38139

noncomputable def fixed_monthly_costs (production_cost shipping_cost units_sold price_per_unit : ℝ) : ℝ :=
  let total_variable_cost := (production_cost + shipping_cost) * units_sold
  let total_revenue := price_per_unit * units_sold
  total_revenue - total_variable_cost

theorem fixed_costs_16699_50 :
  fixed_monthly_costs 80 7 150 198.33 = 16699.5 :=
by
  sorry

end fixed_costs_16699_50_l381_38139


namespace magnitude_of_z_l381_38134

theorem magnitude_of_z (z : ℂ) (h : z * (2 - Complex.I) = Complex.I) : Complex.abs z = 1 / Real.sqrt 5 := by
  sorry

end magnitude_of_z_l381_38134


namespace find_p_when_q_is_1_l381_38196

-- Define the proportionality constant k and the relationship
variables {k p q : ℝ}
def inversely_proportional (k q p : ℝ) : Prop := p = k / (q + 2)

-- Given conditions
theorem find_p_when_q_is_1 (h1 : inversely_proportional k 4 1) : 
  inversely_proportional k 1 2 :=
by 
  sorry

end find_p_when_q_is_1_l381_38196


namespace triangle_area_eq_e_div_4_l381_38177

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

noncomputable def P : ℝ × ℝ := (1, Real.exp 1)

noncomputable def tangent_line (x : ℝ) : ℝ :=
  let k := (Real.exp 1) * (x + 1)
  k * (x - 1) + Real.exp 1

theorem triangle_area_eq_e_div_4 :
  let area := (1 / 2) * Real.exp 1 * (1 / 2)
  area = (Real.exp 1) / 4 :=
by
  sorry

end triangle_area_eq_e_div_4_l381_38177


namespace star_four_three_l381_38131

def star (x y : ℕ) : ℕ := x^2 - x*y + y^2

theorem star_four_three : star 4 3 = 13 := by
  sorry

end star_four_three_l381_38131


namespace factorize_expression_l381_38101

variable (a x : ℝ)

theorem factorize_expression : a * x^2 - 4 * a * x + 4 * a = a * (x - 2)^2 := 
by 
  sorry

end factorize_expression_l381_38101


namespace intersection_of_A_B_C_l381_38114

-- Define the sets A, B, and C as given conditions:
def A : Set ℕ := { x | ∃ n : ℕ, x = 2 * n }
def B : Set ℕ := { x | ∃ n : ℕ, x = 3 * n }
def C : Set ℕ := { x | ∃ n : ℕ, x = n ^ 2 }

-- Prove that A ∩ B ∩ C = { x | ∃ n : ℕ, x = 36 * n ^ 2 }
theorem intersection_of_A_B_C :
  (A ∩ B ∩ C) = { x | ∃ n : ℕ, x = 36 * n ^ 2 } :=
sorry

end intersection_of_A_B_C_l381_38114


namespace cos_75_degree_identity_l381_38143

theorem cos_75_degree_identity :
  Real.cos (75 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 :=
by
  sorry

end cos_75_degree_identity_l381_38143


namespace perpendicular_vectors_m_solution_l381_38123

theorem perpendicular_vectors_m_solution (m : ℝ) 
  (a : ℝ × ℝ := (1, 2)) 
  (b : ℝ × ℝ := (m, 1)) 
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) : m = -2 := by
  sorry

end perpendicular_vectors_m_solution_l381_38123


namespace simplify_expression_l381_38130

theorem simplify_expression (r : ℝ) : 100*r - 48*r + 10 = 52*r + 10 :=
by
  sorry

end simplify_expression_l381_38130


namespace michelle_scored_30_l381_38155

-- Define the total team points
def team_points : ℕ := 72

-- Define the number of other players
def num_other_players : ℕ := 7

-- Define the average points scored by the other players
def avg_points_other_players : ℕ := 6

-- Calculate the total points scored by the other players
def total_points_other_players : ℕ := num_other_players * avg_points_other_players

-- Define the points scored by Michelle
def michelle_points : ℕ := team_points - total_points_other_players

-- Prove that the points scored by Michelle is 30
theorem michelle_scored_30 : michelle_points = 30 :=
by
  -- Here would be the proof, but we skip it with sorry.
  sorry

end michelle_scored_30_l381_38155


namespace solve_equation_l381_38178

theorem solve_equation (x : ℝ) :
  x * (x + 3)^2 * (5 - x) = 0 ∧ x^2 + 3 * x + 2 > 0 ↔ x = -3 ∨ x = 0 ∨ x = 5 :=
by
  sorry

end solve_equation_l381_38178


namespace total_amount_l381_38165

theorem total_amount (A B C : ℤ) (S : ℤ) (h_ratio : 100 * B = 45 * A ∧ 100 * C = 30 * A) (h_B : B = 6300) : S = 24500 := by
  sorry

end total_amount_l381_38165


namespace cylinder_volume_triple_radius_quadruple_height_l381_38138

open Real

theorem cylinder_volume_triple_radius_quadruple_height (r h : ℝ) (V : ℝ) (hV : V = π * r^2 * h) :
  (3 * r) ^ 2 * 4 * h * π = 360 :=
by
  sorry

end cylinder_volume_triple_radius_quadruple_height_l381_38138


namespace solution_set_of_inequalities_l381_38146

theorem solution_set_of_inequalities :
  {x : ℝ | 2 ≤ x / (3 * x - 5) ∧ x / (3 * x - 5) < 9} = {x : ℝ | x > 45 / 26} :=
by sorry

end solution_set_of_inequalities_l381_38146


namespace dhoni_spent_300_dollars_l381_38194

theorem dhoni_spent_300_dollars :
  ∀ (L S X : ℝ),
  L = 6 →
  S = L - 2 →
  (X / S) - (X / L) = 25 →
  X = 300 :=
by
intros L S X hL hS hEquation
sorry

end dhoni_spent_300_dollars_l381_38194


namespace sum_of_two_integers_eq_sqrt_466_l381_38153

theorem sum_of_two_integers_eq_sqrt_466
  (x y : ℝ)
  (hx : x^2 + y^2 = 250)
  (hy : x * y = 108) :
  x + y = Real.sqrt 466 :=
sorry

end sum_of_two_integers_eq_sqrt_466_l381_38153


namespace eagles_points_l381_38171

theorem eagles_points (x y : ℕ) (h₁ : x + y = 82) (h₂ : x - y = 18) : y = 32 :=
sorry

end eagles_points_l381_38171


namespace distance_between_towns_l381_38180

variables (x y z : ℝ)

theorem distance_between_towns
  (h1 : x / 24 + y / 16 + z / 12 = 2)
  (h2 : x / 12 + y / 16 + z / 24 = 2.25) :
  x + y + z = 34 :=
sorry

end distance_between_towns_l381_38180


namespace hyperbola_asymptote_y_eq_1_has_m_neg_3_l381_38187

theorem hyperbola_asymptote_y_eq_1_has_m_neg_3
    (m : ℝ)
    (h1 : ∀ x y, (x^2 / (2 * m)) - (y^2 / m) = 1)
    (h2 : ∀ x, 1 = (x^2 / (2 * m))): m = -3 :=
by
  sorry

end hyperbola_asymptote_y_eq_1_has_m_neg_3_l381_38187


namespace determine_point_T_l381_38128

noncomputable def point : Type := ℝ × ℝ

def is_square (O P Q R : point) : Prop :=
  O.1 = 0 ∧ O.2 = 0 ∧
  Q.1 = 3 ∧ Q.2 = 3 ∧
  P.1 = 3 ∧ P.2 = 0 ∧
  R.1 = 0 ∧ R.2 = 3

def twice_area_square_eq_area_triangle (O P Q T : point) : Prop :=
  2 * (3 * 3) = abs ((P.1 * Q.2 + Q.1 * T.2 + T.1 * P.2 - P.2 * Q.1 - Q.2 * T.1 - T.2 * P.1) / 2)

theorem determine_point_T (O P Q R T : point) (h1 : is_square O P Q R) : 
  twice_area_square_eq_area_triangle O P Q T ↔ T = (3, 12) :=
sorry

end determine_point_T_l381_38128


namespace seq_product_l381_38152

theorem seq_product (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hSn : ∀ n, S n = 2^n - 1)
  (ha : ∀ n, a n = if n = 1 then 1 else 2^(n-1)) :
  a 2 * a 6 = 64 :=
by 
  sorry

end seq_product_l381_38152


namespace animals_percentage_monkeys_l381_38166

theorem animals_percentage_monkeys (initial_monkeys : ℕ) (initial_birds : ℕ) (birds_eaten : ℕ) (final_monkeys : ℕ) (final_birds : ℕ) : 
  initial_monkeys = 6 → 
  initial_birds = 6 → 
  birds_eaten = 2 → 
  final_monkeys = initial_monkeys → 
  final_birds = initial_birds - birds_eaten → 
  (final_monkeys * 100 / (final_monkeys + final_birds) = 60) := 
by intros
   sorry

end animals_percentage_monkeys_l381_38166


namespace shortest_distance_midpoint_parabola_chord_l381_38120

theorem shortest_distance_midpoint_parabola_chord
  (A B : ℝ × ℝ)
  (hA : A.1 ^ 2 = 4 * A.2)
  (hB : B.1 ^ 2 = 4 * B.2)
  (cord_length : dist A B = 6)
  : dist ((A.1 + B.1) / 2, (A.2 + B.2) / 2) (0, 0) = 2 :=
sorry

end shortest_distance_midpoint_parabola_chord_l381_38120


namespace eval_36_pow_five_over_two_l381_38115

theorem eval_36_pow_five_over_two : (36 : ℝ)^(5/2) = 7776 := by
  sorry

end eval_36_pow_five_over_two_l381_38115


namespace limit_sum_perimeters_l381_38199

theorem limit_sum_perimeters (a : ℝ) : ∑' n : ℕ, (4 * a) * (1 / 2) ^ n = 8 * a :=
by sorry

end limit_sum_perimeters_l381_38199


namespace simple_interest_rate_l381_38189

-- Define the conditions
def S : ℚ := 2500
def P : ℚ := 5000
def T : ℚ := 5

-- Define the proof problem
theorem simple_interest_rate (R : ℚ) (h : S = P * R * T / 100) : R = 10 := by
  sorry

end simple_interest_rate_l381_38189


namespace muffins_sugar_l381_38106

theorem muffins_sugar (cups_muffins_ratio : 24 * 3 = 72 * s / 9) : s = 9 := by
  sorry

end muffins_sugar_l381_38106


namespace three_legged_extraterrestrials_l381_38148

-- Define the conditions
variables (x y : ℕ)

-- Total number of heads
def heads_equation := x + y = 300

-- Total number of legs
def legs_equation := 3 * x + 4 * y = 846

theorem three_legged_extraterrestrials : heads_equation x y ∧ legs_equation x y → x = 246 :=
by
  sorry

end three_legged_extraterrestrials_l381_38148


namespace min_area_OBX_l381_38161

structure Point : Type :=
  (x : ℤ)
  (y : ℤ)

def O : Point := ⟨0, 0⟩
def B : Point := ⟨11, 8⟩

def area_triangle (A B C : Point) : ℚ :=
  1 / 2 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

def in_rectangle (X : Point) : Prop :=
  0 ≤ X.x ∧ X.x ≤ 11 ∧ 0 ≤ X.y ∧ X.y ≤ 8

theorem min_area_OBX : ∃ (X : Point), in_rectangle X ∧ area_triangle O B X = 1 / 2 :=
sorry

end min_area_OBX_l381_38161


namespace cost_of_child_ticket_is_4_l381_38144

def cost_of_child_ticket (cost_adult cost_total tickets_sold tickets_child receipts_total : ℕ) : ℕ :=
  let tickets_adult := tickets_sold - tickets_child
  let receipts_adult := tickets_adult * cost_adult
  let receipts_child := receipts_total - receipts_adult
  receipts_child / tickets_child

theorem cost_of_child_ticket_is_4 (cost_adult : ℕ) (cost_total : ℕ)
  (tickets_sold : ℕ) (tickets_child : ℕ) (receipts_total : ℕ) :
  cost_of_child_ticket 12 4 130 90 840 = 4 := by
  sorry

end cost_of_child_ticket_is_4_l381_38144


namespace B_is_345_complement_U_A_inter_B_is_3_l381_38160

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define set A
def A : Set ℕ := {2, 4, 5}

-- Define set B as given in the conditions
def B : Set ℕ := {x ∈ U | 2 < x ∧ x < 6}

-- Prove that B is {3, 4, 5}
theorem B_is_345 : B = {3, 4, 5} := by
  sorry

-- Define the complement of A with respect to U
def complement_U_A : Set ℕ := U \ A

-- Prove the intersection of the complement of A and B is {3}
theorem complement_U_A_inter_B_is_3 : (complement_U_A ∩ B) = {3} := by
  sorry

end B_is_345_complement_U_A_inter_B_is_3_l381_38160


namespace find_angle_and_perimeter_l381_38174

open Real

variables {A B C a b c : ℝ}

/-- If (2a - c)sinA + (2c - a)sinC = 2bsinB in triangle ABC -/
theorem find_angle_and_perimeter
  (h1 : (2 * a - c) * sin A + (2 * c - a) * sin C = 2 * b * sin B)
  (acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
  (b_eq : b = 1) :
  B = π / 3 ∧ (sqrt 3 + 1 < a + b + c ∧ a + b + c ≤ 3) :=
sorry

end find_angle_and_perimeter_l381_38174


namespace solve_for_x_l381_38127

theorem solve_for_x (x : ℝ) (h : 12 - 2 * x = 6) : x = 3 :=
sorry

end solve_for_x_l381_38127
