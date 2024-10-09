import Mathlib

namespace max_dot_product_of_points_on_ellipses_l342_34231

theorem max_dot_product_of_points_on_ellipses :
  let C1 (M : ℝ × ℝ) := M.1^2 / 25 + M.2^2 / 9 = 1
  let C2 (N : ℝ × ℝ) := N.1^2 / 9 + N.2^2 / 25 = 1
  ∃ M N : ℝ × ℝ,
    C1 M ∧ C2 N ∧
    (∀ M N, C1 M ∧ C2 N → M.1 * N.1 + M.2 * N.2 ≤ 15 ∧ 
      (∃ θ φ, M = (5 * Real.cos θ, 3 * Real.sin θ) ∧ N = (3 * Real.cos φ, 5 * Real.sin φ) ∧ (M.1 * N.1 + M.2 * N.2 = 15))) :=
by
  sorry

end max_dot_product_of_points_on_ellipses_l342_34231


namespace maximum_a_for_monotonically_increasing_interval_l342_34246

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)
noncomputable def g (x : ℝ) : ℝ := f (x - (Real.pi / 4))

theorem maximum_a_for_monotonically_increasing_interval :
  ∀ a : ℝ, (∀ x y : ℝ, 0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ a ∧ x < y → g x < g y) → a ≤ Real.pi / 4 := 
by
  sorry

end maximum_a_for_monotonically_increasing_interval_l342_34246


namespace sufficient_conditions_for_x_squared_lt_one_l342_34299

variable (x : ℝ)

theorem sufficient_conditions_for_x_squared_lt_one :
  (∀ x, (0 < x ∧ x < 1) → (x^2 < 1)) ∧
  (∀ x, (-1 < x ∧ x < 0) → (x^2 < 1)) ∧
  (∀ x, (-1 < x ∧ x < 1) → (x^2 < 1)) :=
by
  sorry

end sufficient_conditions_for_x_squared_lt_one_l342_34299


namespace sufficient_but_not_necessary_l342_34281

theorem sufficient_but_not_necessary (a : ℝ) :
  0 < a ∧ a < 1 → (∀ x : ℝ, ax^2 + 2 * a * x + 1 > 0) ∧ ¬ (∀ a, (∀ x : ℝ, ax^2 + 2 * a * x + 1 > 0) → 0 < a ∧ a < 1) :=
by
  sorry

end sufficient_but_not_necessary_l342_34281


namespace ratio_square_areas_l342_34278

theorem ratio_square_areas (r : ℝ) (h1 : r > 0) :
  let s1 := 2 * r / Real.sqrt 5
  let area1 := (s1) ^ 2
  let h := r * Real.sqrt 3
  let s2 := r
  let area2 := (s2) ^ 2
  area1 / area2 = 4 / 5 := by
  sorry

end ratio_square_areas_l342_34278


namespace inequality_solution_l342_34228

theorem inequality_solution (x : ℝ) : 3 * x ^ 2 + x - 2 < 0 ↔ -1 < x ∧ x < 2 / 3 :=
by
  -- The proof should factor the quadratic expression and apply the rule for solving strict inequalities
  sorry

end inequality_solution_l342_34228


namespace last_four_digits_of_m_smallest_l342_34263

theorem last_four_digits_of_m_smallest (m : ℕ) (h1 : m > 0)
  (h2 : m % 6 = 0) (h3 : m % 8 = 0)
  (h4 : ∀ d, d ∈ (m.digits 10) → d = 2 ∨ d = 7)
  (h5 : 2 ∈ (m.digits 10)) (h6 : 7 ∈ (m.digits 10)) :
  (m % 10000) = 2722 :=
sorry

end last_four_digits_of_m_smallest_l342_34263


namespace product_of_primes_impossible_l342_34267

theorem product_of_primes_impossible (q : ℕ) (hq1 : Nat.Prime q) (hq2 : q % 2 = 1) :
  ¬ ∀ i ∈ Finset.range (q-1), ∃ p1 p2 : ℕ, Nat.Prime p1 ∧ Nat.Prime p2 ∧ (i^2 + i + q = p1 * p2) :=
sorry

end product_of_primes_impossible_l342_34267


namespace find_g_5_l342_34249

theorem find_g_5 (g : ℝ → ℝ) (h : ∀ x : ℝ, g x + 3 * g (1 - x) = 2 * x ^ 2 + 1) : g 5 = 8 :=
sorry

end find_g_5_l342_34249


namespace mindy_messages_total_l342_34236

theorem mindy_messages_total (P : ℕ) (h1 : 83 = 9 * P - 7) : 83 + P = 93 :=
  by
    sorry

end mindy_messages_total_l342_34236


namespace simplify_expression_l342_34290

variable (a b : ℤ) -- Define variables a and b

theorem simplify_expression : 
  (35 * a + 70 * b + 15) + (15 * a + 54 * b + 5) - (20 * a + 85 * b + 10) =
  30 * a + 39 * b + 10 := 
by sorry

end simplify_expression_l342_34290


namespace martha_children_l342_34293

noncomputable def num_children (total_cakes : ℕ) (cakes_per_child : ℕ) : ℕ :=
  total_cakes / cakes_per_child

theorem martha_children : num_children 18 6 = 3 := by
  sorry

end martha_children_l342_34293


namespace set_intersections_l342_34221

open Set Nat

def I : Set ℕ := univ

def A : Set ℕ := { x | ∃ n, x = 3 * n ∧ ∃ k, n = 2 * k }

def B : Set ℕ := { y | ∃ m, y = m ∧ 24 % m = 0 }

theorem set_intersections :
  A ∩ B = {6, 12, 24} ∧ (I \ A) ∩ B = {1, 2, 3, 4, 8} :=
by
  sorry

end set_intersections_l342_34221


namespace mildred_heavier_than_carol_l342_34212

def mildred_weight : ℕ := 59
def carol_weight : ℕ := 9

theorem mildred_heavier_than_carol : mildred_weight - carol_weight = 50 := 
by
  sorry

end mildred_heavier_than_carol_l342_34212


namespace min_value_expression_l342_34219

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  8 * x^3 + 27 * y^3 + 64 * z^3 + (1 / (8 * x * y * z)) ≥ 4 :=
by
  sorry

end min_value_expression_l342_34219


namespace price_of_soda_l342_34289

theorem price_of_soda (regular_price_per_can : ℝ) (case_discount : ℝ) (bulk_discount : ℝ) (num_cases : ℕ) (num_cans : ℕ) :
  regular_price_per_can = 0.15 →
  case_discount = 0.12 →
  bulk_discount = 0.05 →
  num_cases = 3 →
  num_cans = 75 →
  (num_cans * ((regular_price_per_can * (1 - case_discount)) * (1 - bulk_discount))) = 9.405 :=
by
  intros h1 h2 h3 h4 h5
  -- normal price per can
  have hp1 : ℝ := regular_price_per_can
  -- price after case discount
  have hp2 : ℝ := hp1 * (1 - case_discount)
  -- price after bulk discount
  have hp3 : ℝ := hp2 * (1 - bulk_discount)
  -- total price
  have total_price : ℝ := num_cans * hp3
  -- goal
  sorry -- skip the proof, as only the statement is needed.

end price_of_soda_l342_34289


namespace cube_volume_fourth_power_l342_34271

theorem cube_volume_fourth_power (s : ℝ) (h : 6 * s^2 = 864) : s^4 = 20736 :=
sorry

end cube_volume_fourth_power_l342_34271


namespace find_total_grade10_students_l342_34222

/-
Conditions:
1. The school has a total of 1800 students in grades 10 and 11.
2. 90 students are selected as a sample for a survey.
3. The sample contains 42 grade 10 students.
-/

variables (total_students sample_size sample_grade10 total_grade10 : ℕ)

axiom total_students_def : total_students = 1800
axiom sample_size_def : sample_size = 90
axiom sample_grade10_def : sample_grade10 = 42

theorem find_total_grade10_students : total_grade10 = 840 :=
by
  have h : (sample_size : ℚ) / (total_students : ℚ) = (sample_grade10 : ℚ) / (total_grade10 : ℚ) :=
    sorry
  sorry

end find_total_grade10_students_l342_34222


namespace det_of_matrix_l342_34201

variable {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)

theorem det_of_matrix (h1 : 1 ≤ n)
  (h2 : A ^ 7 + A ^ 5 + A ^ 3 + A - 1 = 0) :
  0 < Matrix.det A :=
sorry

end det_of_matrix_l342_34201


namespace distance_walked_east_l342_34259

-- Definitions for distances
def s1 : ℕ := 25   -- distance walked south
def s2 : ℕ := 20   -- distance walked east
def s3 : ℕ := 25   -- distance walked north
def final_distance : ℕ := 35   -- final distance from the starting point

-- Proof problem: Prove that the distance walked east in the final step is as expected
theorem distance_walked_east (d : Real) :
  d = Real.sqrt (final_distance ^ 2 - s2 ^ 2) :=
sorry

end distance_walked_east_l342_34259


namespace david_tips_l342_34232

noncomputable def avg_tips_resort (tips_other_months : ℝ) (months : ℕ) := tips_other_months / months

theorem david_tips 
  (tips_march_to_july_september : ℝ)
  (tips_august_resort : ℝ)
  (total_tips_delivery_driver : ℝ)
  (total_tips_resort : ℝ)
  (total_tips : ℝ)
  (fraction_august : ℝ)
  (avg_tips := avg_tips_resort tips_march_to_july_september 6):
  tips_august_resort = 4 * avg_tips →
  total_tips_delivery_driver = 2 * avg_tips →
  total_tips_resort = tips_march_to_july_september + tips_august_resort →
  total_tips = total_tips_resort + total_tips_delivery_driver →
  fraction_august = tips_august_resort / total_tips →
  fraction_august = 1 / 2 :=
by
  sorry

end david_tips_l342_34232


namespace problem_correct_statements_l342_34239

def T (a b x y : ℚ) : ℚ := a * x * y + b * x - 4

theorem problem_correct_statements (a b : ℚ) (h₁ : T a b 2 1 = 2) (h₂ : T a b (-1) 2 = -8) :
  (a = 1 ∧ b = 2) ∧
  (∀ m n : ℚ, T 1 2 m n = 0 ∧ n ≠ -2 → m = 4 / (n + 2)) ∧
  ¬ (∃ m n : ℤ, T 1 2 m n = 0 ∧ n ≠ -2 ∧ m + n = 3) ∧
  (∀ k x y : ℚ, T 1 2 (k * x) y = T 1 2 (k * x) y → y = -2) ∧
  (∀ k x y : ℚ, x ≠ y → T 1 2 (k * x) y = T 1 2 (k * y) x → k = 0) :=
by
  sorry

end problem_correct_statements_l342_34239


namespace binomial_expansion_a5_l342_34264

theorem binomial_expansion_a5 (x : ℝ) 
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ) 
  (h : (x - 1) ^ 8 = a_0 + a_1 * (1 + x) + a_2 * (1 + x) ^ 2 + a_3 * (1 + x) ^ 3 + a_4 * (1 + x) ^ 4 + a_5 * (1 + x) ^ 5 + a_6 * (1 + x) ^ 6 + a_7 * (1 + x) ^ 7 + a_8 * (1 + x) ^ 8) : 
  a_5 = -448 := 
sorry

end binomial_expansion_a5_l342_34264


namespace curve_focus_x_axis_l342_34280

theorem curve_focus_x_axis : 
    (x^2 - y^2 = 1)
    ∨ (∃ a b : ℝ, a ≠ 0 ∧ a ≠ b ∧ 
        (a*x^2 + b*y^2 = 1 → False)
    )
    ∨ (∃ a b : ℝ, a ≠ 0 ∧ a ≠ b ∧ 
        (b*y^2 - a*x^2 = 1 → False)
    )
    ∨ (∃ c : ℝ, c ≠ 0 ∧ 
        (y = c*x^2 → False)
    ) :=
sorry

end curve_focus_x_axis_l342_34280


namespace sheela_monthly_income_l342_34282

theorem sheela_monthly_income (d : ℝ) (p : ℝ) (income : ℝ) (h1 : d = 4500) (h2 : p = 0.28) (h3 : d = p * income) : 
  income = 16071.43 :=
by
  sorry

end sheela_monthly_income_l342_34282


namespace full_price_ticket_revenue_correct_l342_34277

-- Define the constants and assumptions
variables (f t : ℕ) (p : ℝ)

-- Total number of tickets sold
def total_tickets := (f + t = 180)

-- Total revenue from ticket sales
def total_revenue := (f * p + t * (p / 3) = 2600)

-- Full price ticket revenue
def full_price_revenue := (f * p = 975)

-- The theorem combines the above conditions to prove the correct revenue from full-price tickets
theorem full_price_ticket_revenue_correct :
  total_tickets f t →
  total_revenue f t p →
  full_price_revenue f p :=
by
  sorry

end full_price_ticket_revenue_correct_l342_34277


namespace equation_of_chord_l342_34247

open Real

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6 * x = 0

def is_midpoint_of_chord (P M N : ℝ × ℝ) : Prop :=
  ∃ (C : ℝ × ℝ), circle_eq (C.1) (C.2) ∧ (P.1, P.2) = ((M.1 + N.1) / 2, (M.2 + N.2) / 2)

theorem equation_of_chord (P : ℝ × ℝ) (M N : ℝ × ℝ) (h : P = (4, 2)) (h_mid : is_midpoint_of_chord P M N) :
  ∀ (x y : ℝ), (2 * y) - (8 : ℝ) = (-(1 / 2) * (x - 4)) →
  x + 2 * y - 8 = 0 :=
by
  intro x y H
  sorry

end equation_of_chord_l342_34247


namespace inequality_pow4_geq_sum_l342_34223

theorem inequality_pow4_geq_sum (a b c d e : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0) :
  (a / b) ^ 4 + (b / c) ^ 4 + (c / d) ^ 4 + (d / e) ^ 4 + (e / a) ^ 4 ≥ 
  (a / b) + (b / c) + (c / d) + (d / e) + (e / a) :=
by
  sorry

end inequality_pow4_geq_sum_l342_34223


namespace triangle_ABC_AC_l342_34257

-- Defining the relevant points and lengths in the triangle
variables {A B C D : Type} 
variables (AB CD : ℝ)
variables (AD BC AC : ℝ)

-- Given constants
axiom hAB : AB = 3
axiom hCD : CD = Real.sqrt 3
axiom hAD_BC : AD = BC

-- The final theorem statement that needs to be proved
theorem triangle_ABC_AC :
  (AD = BC) ∧ (CD = Real.sqrt 3) ∧ (AB = 3) → AC = Real.sqrt 7 :=
by
  intros h
  sorry

end triangle_ABC_AC_l342_34257


namespace least_number_subtracted_l342_34295

theorem least_number_subtracted (n : ℕ) (h : n = 2361) : 
  ∃ k, (n - k) % 23 = 0 ∧ k = 15 := 
by
  sorry

end least_number_subtracted_l342_34295


namespace divides_of_exponentiation_l342_34205

theorem divides_of_exponentiation (n : ℕ) : 7 ∣ 3^(12 * n + 1) + 2^(6 * n + 2) := 
  sorry

end divides_of_exponentiation_l342_34205


namespace inequality_proof_l342_34269

variable (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
variable (h_ab_bc_ca : a * b + b * c + c * a = 1)

theorem inequality_proof :
  (3 / Real.sqrt (a^2 + 1)) + (4 / Real.sqrt (b^2 + 1)) + (12 / Real.sqrt (c^2 + 1)) < (39 / 2) :=
by
  sorry

end inequality_proof_l342_34269


namespace remaining_apps_eq_files_plus_more_initial_apps_eq_16_l342_34250

-- Defining the initial number of files
def initial_files: ℕ := 9

-- Defining the remaining number of files and apps
def remaining_files: ℕ := 5
def remaining_apps: ℕ := 12

-- Given: Dave has 7 more apps than files left
def apps_more_than_files: ℕ := 7

-- Equating the given condition 12 = 5 + 7
theorem remaining_apps_eq_files_plus_more :
  remaining_apps = remaining_files + apps_more_than_files := by
  sorry -- This would trivially prove as 12 = 5+7

-- Proving the number of initial apps
theorem initial_apps_eq_16 (A: ℕ) (h1: initial_files = 9) (h2: remaining_files = 5) (h3: remaining_apps = 12) (h4: apps_more_than_files = 7):
  A - remaining_apps = initial_files - remaining_files → A = 16 := by
  sorry

end remaining_apps_eq_files_plus_more_initial_apps_eq_16_l342_34250


namespace problem_solution_exists_l342_34291

theorem problem_solution_exists (a b n : ℕ) (p : ℕ) [hp : Fact (Nat.Prime p)]
  (h : a > 0 ∧ b > 0 ∧ n > 0 ∧ a ^ 2013 + b ^ 2013 = p ^ n) :
  ∃ k : ℕ, a = 2^k ∧ b = 2^k ∧ n = 2013 * k + 1 ∧ p = 2 := by
  sorry

end problem_solution_exists_l342_34291


namespace max_median_cans_per_customer_l342_34238

theorem max_median_cans_per_customer : 
    ∀ (total_cans : ℕ) (total_customers : ℕ), 
    total_cans = 252 → total_customers = 100 →
    (∀ (cans_per_customer : ℕ),
    1 ≤ cans_per_customer) →
    (∃ (max_median : ℝ),
    max_median = 3.5) :=
by
  sorry

end max_median_cans_per_customer_l342_34238


namespace point_below_line_l342_34287

theorem point_below_line (a : ℝ) (h : 2 * a - 3 > 3) : a > 3 :=
sorry

end point_below_line_l342_34287


namespace triangle_is_isosceles_l342_34234

theorem triangle_is_isosceles (A B C : ℝ) (a b c : ℝ) 
  (h1 : c = 2 * a * Real.cos B) 
  (h2 : a = b) :
  ∃ (isIsosceles : Bool), isIsosceles := 
sorry

end triangle_is_isosceles_l342_34234


namespace technicans_permanent_50pct_l342_34200

noncomputable def percentage_technicians_permanent (p : ℝ) : Prop :=
  let technicians := 0.5
  let non_technicians := 0.5
  let temporary := 0.5
  (0.5 * (1 - 0.5)) + (technicians * p) = 0.5 ->
  p = 0.5

theorem technicans_permanent_50pct (p : ℝ) :
  percentage_technicians_permanent p :=
sorry

end technicans_permanent_50pct_l342_34200


namespace problem_statement_l342_34260

noncomputable def proposition_p (x : ℝ) : Prop := ∃ x0 : ℝ, x0 - 2 > 0
noncomputable def proposition_q (x : ℝ) : Prop := ∀ x : ℝ, (2:ℝ)^x > x^2

theorem problem_statement : ∃ (p q : Prop), (∃ x0 : ℝ, x0 - 2 > 0) ∧ (¬ (∀ x : ℝ, (2:ℝ)^x > x^2)) :=
by
  sorry

end problem_statement_l342_34260


namespace exactly_one_absent_l342_34233

variables (B K Z : Prop)

theorem exactly_one_absent (h1 : B ∨ K) (h2 : K ∨ Z) (h3 : Z ∨ B)
    (h4 : ¬B ∨ ¬K ∨ ¬Z) : (¬B ∧ K ∧ Z) ∨ (B ∧ ¬K ∧ Z) ∨ (B ∧ K ∧ ¬Z) :=
by
  sorry

end exactly_one_absent_l342_34233


namespace eval_f_at_neg_twenty_three_sixth_pi_l342_34243

noncomputable def f (α : ℝ) : ℝ := 
    (2 * (Real.sin (2 * Real.pi - α)) * (Real.cos (2 * Real.pi + α)) - Real.cos (-α)) / 
    (1 + Real.sin α ^ 2 + Real.sin (2 * Real.pi + α) - Real.cos (4 * Real.pi - α) ^ 2)

theorem eval_f_at_neg_twenty_three_sixth_pi : 
  f (-23 / 6 * Real.pi) = -Real.sqrt 3 :=
  sorry

end eval_f_at_neg_twenty_three_sixth_pi_l342_34243


namespace tensor_example_l342_34256
-- Import the necessary library

-- Define the binary operation ⊗
def tensor (a b : ℚ) : ℚ := (a + b) / (a - b)

-- State the main theorem
theorem tensor_example : tensor (tensor 8 6) 2 = 9 / 5 := by
  sorry

end tensor_example_l342_34256


namespace percentage_increase_is_200_l342_34244

noncomputable def total_cost : ℝ := 300
noncomputable def rate_per_sq_m : ℝ := 5
noncomputable def length : ℝ := 13.416407864998739
noncomputable def area : ℝ := total_cost / rate_per_sq_m
noncomputable def breadth : ℝ := area / length
noncomputable def percentage_increase : ℝ := (length - breadth) / breadth * 100

theorem percentage_increase_is_200 :
  percentage_increase = 200 :=
by
  sorry

end percentage_increase_is_200_l342_34244


namespace find_original_expenditure_l342_34217

def original_expenditure (x : ℝ) := 35 * x
def new_expenditure (x : ℝ) := 42 * (x - 1)

theorem find_original_expenditure :
  ∃ x, 35 * x + 42 = 42 * (x - 1) ∧ original_expenditure x = 420 :=
by
  sorry

end find_original_expenditure_l342_34217


namespace area_of_right_triangle_l342_34224

variables {x y : ℝ} (r : ℝ)

theorem area_of_right_triangle (hx : ∀ r, r * (x + y + r) = x * y) :
  1 / 2 * (x + r) * (y + r) = x * y :=
by sorry

end area_of_right_triangle_l342_34224


namespace new_bag_marbles_l342_34284

open Nat

theorem new_bag_marbles 
  (start_marbles : ℕ)
  (lost_marbles : ℕ)
  (given_marbles : ℕ)
  (received_back_marbles : ℕ)
  (end_marbles : ℕ)
  (h_start : start_marbles = 40)
  (h_lost : lost_marbles = 3)
  (h_given : given_marbles = 5)
  (h_received_back : received_back_marbles = 2 * given_marbles)
  (h_end : end_marbles = 54) :
  (end_marbles = (start_marbles - lost_marbles - given_marbles + received_back_marbles + new_bag) ∧ new_bag = 12) :=
by
  sorry

end new_bag_marbles_l342_34284


namespace find_age_of_b_l342_34209

variable (a b : ℤ)

-- Conditions
axiom cond1 : a + 10 = 2 * (b - 10)
axiom cond2 : a = b + 9

-- Goal
theorem find_age_of_b : b = 39 :=
sorry

end find_age_of_b_l342_34209


namespace greatest_divisible_by_13_l342_34279

theorem greatest_divisible_by_13 (A B C : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 0 ≤ B ∧ B ≤ 9) (hC : 0 ≤ C ∧ C ≤ 9)
  (distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) : (10000 * A + 1000 * B + 100 * C + 10 * B + A = 96769) 
  ↔ (10000 * A + 1000 * B + 100 * C + 10 * B + A) % 13 = 0 :=
sorry

end greatest_divisible_by_13_l342_34279


namespace money_brought_to_store_l342_34210

theorem money_brought_to_store : 
  let sheet_cost := 42
  let rope_cost := 18
  let propane_and_burner_cost := 14
  let helium_cost_per_ounce := 1.5
  let height_per_ounce := 113
  let max_height := 9492
  let total_item_cost := sheet_cost + rope_cost + propane_and_burner_cost
  let helium_needed := max_height / height_per_ounce
  let helium_total_cost := helium_needed * helium_cost_per_ounce
  total_item_cost + helium_total_cost = 200 :=
by
  sorry

end money_brought_to_store_l342_34210


namespace perfect_square_sequence_l342_34208

theorem perfect_square_sequence (x : ℕ → ℤ) (h₀ : x 0 = 0) (h₁ : x 1 = 3) 
  (h₂ : ∀ n, x (n + 1) + x (n - 1) = 4 * x n) : 
  ∀ n, ∃ k : ℤ, x (n + 1) * x (n - 1) + 9 = k^2 :=
by 
  sorry

end perfect_square_sequence_l342_34208


namespace rational_roots_of_quadratic_l342_34203

theorem rational_roots_of_quadratic (k : ℤ) (h : k > 0) :
  (∃ x : ℚ, k * x^2 + 12 * x + k = 0) ↔ (k = 3 ∨ k = 6) :=
by
  sorry

end rational_roots_of_quadratic_l342_34203


namespace combination_sum_l342_34272

theorem combination_sum : Nat.choose 10 3 + Nat.choose 10 4 = 330 := 
by
  sorry

end combination_sum_l342_34272


namespace remainder_of_division_l342_34297

theorem remainder_of_division :
  ∃ R : ℕ, 176 = (19 * 9) + R ∧ R = 5 :=
by
  sorry

end remainder_of_division_l342_34297


namespace negative_solution_range_l342_34218

theorem negative_solution_range (m x : ℝ) (h : (2 * x + m) / (x - 1) = 1) (hx : x < 0) : m > -1 :=
  sorry

end negative_solution_range_l342_34218


namespace lcm_140_225_is_6300_l342_34253

def lcm_140_225 : ℕ := Nat.lcm 140 225

theorem lcm_140_225_is_6300 : lcm_140_225 = 6300 :=
by
  sorry

end lcm_140_225_is_6300_l342_34253


namespace physics_students_l342_34248

variable (B : Nat) (G : Nat) (Biology : Nat) (Physics : Nat)

axiom h1 : B = 25
axiom h2 : G = 3 * B
axiom h3 : Biology = B + G
axiom h4 : Physics = 2 * Biology

theorem physics_students : Physics = 200 :=
by
  sorry

end physics_students_l342_34248


namespace minimum_value_of_sum_l342_34215

theorem minimum_value_of_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) : 
    1 / (2 * a + b) + 1 / (2 * b + c) + 1 / (2 * c + a) >= 3 :=
by
  sorry

end minimum_value_of_sum_l342_34215


namespace train_crossing_time_l342_34227

theorem train_crossing_time
    (length_of_train : ℕ)
    (speed_of_train_kmph : ℕ)
    (length_of_bridge : ℕ)
    (h_train_length : length_of_train = 160)
    (h_speed_kmph : speed_of_train_kmph = 45)
    (h_bridge_length : length_of_bridge = 215)
  : length_of_train + length_of_bridge / ((speed_of_train_kmph * 1000) / 3600) = 30 :=
by
  rw [h_train_length, h_speed_kmph, h_bridge_length]
  norm_num
  sorry

end train_crossing_time_l342_34227


namespace minimum_g_a_l342_34283

noncomputable def f (x a : ℝ) : ℝ := x ^ 2 + 2 * a * x + 3

noncomputable def g (a : ℝ) : ℝ := 3 * a ^ 2 + 2 * a

theorem minimum_g_a : ∀ a : ℝ, a ≤ -1 → g a = 3 * a ^ 2 + 2 * a → g a ≥ 1 := by
  sorry

end minimum_g_a_l342_34283


namespace train_length_l342_34211

theorem train_length (speed_kmh : ℕ) (cross_time : ℕ) (h_speed : speed_kmh = 54) (h_time : cross_time = 9) :
  let speed_ms := speed_kmh * (1000 / 3600)
  let length_m := speed_ms * cross_time
  length_m = 135 := by
  sorry

end train_length_l342_34211


namespace expression_evaluates_to_one_l342_34240

noncomputable def a := Real.sqrt 2 + 0.8
noncomputable def b := Real.sqrt 2 - 0.2

theorem expression_evaluates_to_one : 
  ( (2 - b) / (b - 1) + 2 * (a - 1) / (a - 2) ) / ( b * (a - 1) / (b - 1) + a * (2 - b) / (a - 2) ) = 1 :=
by
  sorry

end expression_evaluates_to_one_l342_34240


namespace find_weight_per_square_inch_l342_34286

-- Define the TV dimensions and other given data
def bill_tv_width : ℕ := 48
def bill_tv_height : ℕ := 100
def bob_tv_width : ℕ := 70
def bob_tv_height : ℕ := 60
def weight_difference_pounds : ℕ := 150
def ounces_per_pound : ℕ := 16

-- Compute areas
def bill_tv_area := bill_tv_width * bill_tv_height
def bob_tv_area := bob_tv_width * bob_tv_height

-- Assume weight per square inch
def weight_per_square_inch : ℕ := 4

-- Total weight computation given in ounces
def bill_tv_weight := bill_tv_area * weight_per_square_inch
def bob_tv_weight := bob_tv_area * weight_per_square_inch
def weight_difference_ounces := weight_difference_pounds * ounces_per_pound

-- The theorem to prove
theorem find_weight_per_square_inch : 
  bill_tv_weight - bob_tv_weight = weight_difference_ounces → weight_per_square_inch = 4 :=
by
  intros
  /- Proof by computation -/
  sorry

end find_weight_per_square_inch_l342_34286


namespace gabrielle_peaches_l342_34237

theorem gabrielle_peaches (B G : ℕ) 
  (h1 : 16 = 2 * B + 6)
  (h2 : B = G / 3) :
  G = 15 :=
by
  sorry

end gabrielle_peaches_l342_34237


namespace find_cost_per_sq_foot_l342_34202

noncomputable def monthly_rent := 2800 / 2
noncomputable def old_annual_rent (C : ℝ) := 750 * C * 12
noncomputable def new_annual_rent := monthly_rent * 12
noncomputable def annual_savings := old_annual_rent - new_annual_rent

theorem find_cost_per_sq_foot (C : ℝ):
    (750 * C * 12 - 2800 / 2 * 12 = 1200) ↔ (C = 2) :=
sorry

end find_cost_per_sq_foot_l342_34202


namespace range_of_a_l342_34220

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x < 1 then (2 - a) * x - 3 * a + 3 
  else Real.log x / Real.log a

-- Main statement to prove
theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (5 / 4 ≤ a ∧ a < 2) :=
sorry

end range_of_a_l342_34220


namespace simplify_expr1_simplify_expr2_l342_34270

-- Definition for the expression (2x - 3y)²
def expr1 (x y : ℝ) : ℝ := (2 * x - 3 * y) ^ 2

-- Theorem to prove that (2x - 3y)² = 4x² - 12xy + 9y²
theorem simplify_expr1 (x y : ℝ) : expr1 x y = 4 * (x ^ 2) - 12 * x * y + 9 * (y ^ 2) := 
sorry

-- Definition for the expression (x + y) * (x + y) * (x² + y²)
def expr2 (x y : ℝ) : ℝ := (x + y) * (x + y) * (x ^ 2 + y ^ 2)

-- Theorem to prove that (x + y) * (x + y) * (x² + y²) = x⁴ + 2x²y² + y⁴ + 2x³y + 2xy³
theorem simplify_expr2 (x y : ℝ) : expr2 x y = x ^ 4 + 2 * (x ^ 2) * (y ^ 2) + y ^ 4 + 2 * (x ^ 3) * y + 2 * x * (y ^ 3) := 
sorry

end simplify_expr1_simplify_expr2_l342_34270


namespace principal_invested_years_l342_34204

-- Define the given conditions
def principal : ℕ := 9200
def rate : ℕ := 12
def interest_deficit : ℤ := 5888

-- Define the time to be proved
def time_invested : ℤ := 3

-- Define the simple interest formula
def simple_interest (P R t : ℕ) : ℕ :=
  (P * R * t) / 100

-- Define the problem statement
theorem principal_invested_years :
  ∃ t : ℕ, principal - interest_deficit = simple_interest principal rate t ∧ t = time_invested := 
by
  sorry

end principal_invested_years_l342_34204


namespace number_of_rolls_l342_34276

theorem number_of_rolls (p : ℚ) (h : p = 1 / 9) : (2 : ℕ) = 2 :=
by 
  have h1 : 2 = 2 := rfl
  exact h1

end number_of_rolls_l342_34276


namespace systematic_sampling_l342_34262

theorem systematic_sampling (N n : ℕ) (hN : N = 1650) (hn : n = 35) :
  let E := 5 
  let segments := 35 
  let individuals_per_segment := 47 
  1650 % 35 = E ∧ 
  (1650 - E) / 35 = individuals_per_segment :=
by 
  sorry

end systematic_sampling_l342_34262


namespace compute_9_times_one_seventh_pow_4_l342_34266

theorem compute_9_times_one_seventh_pow_4 :
  9 * (1 / 7) ^ 4 = 9 / 2401 :=
by
  -- The actual proof would go here.
  sorry

end compute_9_times_one_seventh_pow_4_l342_34266


namespace problem_f_2010_l342_34206

noncomputable def f : ℝ → ℝ := sorry

axiom f_1 : f 1 = 1 / 4
axiom f_eq : ∀ x y : ℝ, 4 * f x * f y = f (x + y) + f (x - y)

theorem problem_f_2010 : f 2010 = 1 / 2 :=
sorry

end problem_f_2010_l342_34206


namespace find_percentage_of_other_investment_l342_34229

theorem find_percentage_of_other_investment
  (total_investment : ℝ) (specific_investment : ℝ) (specific_rate : ℝ) (total_interest : ℝ) 
  (other_investment : ℝ) (other_interest : ℝ) (P : ℝ) :
  total_investment = 17000 ∧
  specific_investment = 12000 ∧
  specific_rate = 0.04 ∧
  total_interest = 1380 ∧
  other_investment = total_investment - specific_investment ∧
  other_interest = total_interest - specific_rate * specific_investment ∧ 
  other_interest = (P / 100) * other_investment
  → P = 18 :=
by
  intros
  sorry

end find_percentage_of_other_investment_l342_34229


namespace chocolate_distribution_l342_34292

theorem chocolate_distribution (n : ℕ) 
  (h1 : 12 * 2 ≤ n * 2 ∨ n * 2 ≤ 12 * 2) 
  (h2 : ∃ d : ℚ, (12 / n) = d ∧ d * n = 12) : 
  n = 15 :=
by 
  sorry

end chocolate_distribution_l342_34292


namespace average_candies_correct_l342_34216

noncomputable def Eunji_candies : ℕ := 35
noncomputable def Jimin_candies : ℕ := Eunji_candies + 6
noncomputable def Jihyun_candies : ℕ := Eunji_candies - 3
noncomputable def Total_candies : ℕ := Eunji_candies + Jimin_candies + Jihyun_candies
noncomputable def Average_candies : ℚ := Total_candies / 3

theorem average_candies_correct :
  Average_candies = 36 := by
  sorry

end average_candies_correct_l342_34216


namespace water_park_admission_l342_34251

def adult_admission_charge : ℝ := 1
def child_admission_charge : ℝ := 0.75
def children_accompanied : ℕ := 3
def total_admission_charge (adults : ℝ) (children : ℝ) : ℝ := adults + children

theorem water_park_admission :
  let adult_charge := adult_admission_charge
  let children_charge := children_accompanied * child_admission_charge
  total_admission_charge adult_charge children_charge = 3.25 :=
by sorry

end water_park_admission_l342_34251


namespace max_value_fraction_l342_34225

theorem max_value_fraction (x : ℝ) : 
  ∃ (n : ℤ), n = 3 ∧ 
  ∃ (y : ℝ), y = (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 9) ∧ 
  y ≤ n := 
sorry

end max_value_fraction_l342_34225


namespace zack_traveled_to_18_countries_l342_34273

-- Defining the conditions
variables (countries_traveled_by_george countries_traveled_by_joseph 
           countries_traveled_by_patrick countries_traveled_by_zack : ℕ)

-- Set the conditions as per the problem statement
axiom george_traveled : countries_traveled_by_george = 6
axiom joseph_traveled : countries_traveled_by_joseph = countries_traveled_by_george / 2
axiom patrick_traveled : countries_traveled_by_patrick = 3 * countries_traveled_by_joseph
axiom zack_traveled : countries_traveled_by_zack = 2 * countries_traveled_by_patrick

-- The theorem to prove Zack traveled to 18 countries
theorem zack_traveled_to_18_countries : countries_traveled_by_zack = 18 :=
by
  -- Adding the proof here is unnecessary as per the instructions
  sorry

end zack_traveled_to_18_countries_l342_34273


namespace hyperbola_asymptote_m_l342_34288

def isAsymptote (x y : ℝ) (m : ℝ) : Prop :=
  y = m * x ∨ y = -m * x

theorem hyperbola_asymptote_m (m : ℝ) : 
  (∀ x y, (x^2 / 25 - y^2 / 16 = 1 → isAsymptote x y m)) ↔ m = 4 / 5 := 
by
  sorry

end hyperbola_asymptote_m_l342_34288


namespace fraction_zero_value_l342_34261

theorem fraction_zero_value (x : ℝ) (h : (3 - x) ≠ 0) : (x+2)/(3-x) = 0 ↔ x = -2 := by
  sorry

end fraction_zero_value_l342_34261


namespace incorrect_value_l342_34296

theorem incorrect_value:
  ∀ (n : ℕ) (initial_mean corrected_mean : ℚ) (correct_value incorrect_value : ℚ),
  n = 50 →
  initial_mean = 36 →
  corrected_mean = 36.5 →
  correct_value = 48 →
  incorrect_value = correct_value - (corrected_mean * n - initial_mean * n) →
  incorrect_value = 23 :=
by
  intros n initial_mean corrected_mean correct_value incorrect_value
  intros h1 h2 h3 h4 h5
  sorry

end incorrect_value_l342_34296


namespace sum_of_three_numbers_l342_34241

def a : ℚ := 859 / 10
def b : ℚ := 531 / 100
def c : ℚ := 43 / 2

theorem sum_of_three_numbers : a + b + c = 11271 / 100 := by
  sorry

end sum_of_three_numbers_l342_34241


namespace helmet_price_for_given_profit_helmet_price_for_max_profit_l342_34255

section helmet_sales

-- Define the conditions
variable (original_price : ℝ := 80) (initial_sales : ℝ := 200) (cost_price : ℝ := 50) 
variable (price_reduction_unit : ℝ := 1) (additional_sales_per_reduction : ℝ := 10)
variable (minimum_price_reduction : ℝ := 10)

-- Profits
def profit (x : ℝ) : ℝ :=
  (original_price - x - cost_price) * (initial_sales + additional_sales_per_reduction * x)

-- Prove the selling price when profit is 5250 yuan
theorem helmet_price_for_given_profit (GDP : profit 15 = 5250) : (original_price - 15) = 65 :=
by
  sorry

-- Prove the price for maximum profit
theorem helmet_price_for_max_profit : 
  ∃ x, x = 10 ∧ (original_price - x = 70) ∧ (profit x = 6000) :=
by 
  sorry

end helmet_sales

end helmet_price_for_given_profit_helmet_price_for_max_profit_l342_34255


namespace candy_bars_total_l342_34245

theorem candy_bars_total :
  let people : ℝ := 3.0;
  let candy_per_person : ℝ := 1.66666666699999;
  people * candy_per_person = 5.0 :=
by
  let people : ℝ := 3.0
  let candy_per_person : ℝ := 1.66666666699999
  show people * candy_per_person = 5.0
  sorry

end candy_bars_total_l342_34245


namespace power_mod_equivalence_l342_34252

theorem power_mod_equivalence : (7^700) % 100 = 1 := 
by 
  -- Given that (7^4) % 100 = 1
  have h : 7^4 % 100 = 1 := by sorry
  -- Use this equivalence to prove the statement
  sorry

end power_mod_equivalence_l342_34252


namespace composite_prime_fraction_l342_34275

theorem composite_prime_fraction :
  let P1 : ℕ := 4 * 6 * 8 * 9 * 10 * 12 * 14 * 15
  let P2 : ℕ := 16 * 18 * 20 * 21 * 22 * 24 * 25 * 26
  let first_prime : ℕ := 2
  let second_prime : ℕ := 3
  (P1 + first_prime) / (P2 + second_prime) =
    (4 * 6 * 8 * 9 * 10 * 12 * 14 * 15 + 2) / (16 * 18 * 20 * 21 * 22 * 24 * 25 * 26 + 3) := by
  sorry

end composite_prime_fraction_l342_34275


namespace simplify_expression_l342_34268

theorem simplify_expression : 
  (3.875 * (1 / 5) + (38 + 3 / 4) * 0.09 - 0.155 / 0.4) / 
  (2 + 1 / 6 + (((4.32 - 1.68 - (1 + 8 / 25)) * (5 / 11) - 2 / 7) / (1 + 9 / 35)) + (1 + 11 / 24))
  = 1 := sorry

end simplify_expression_l342_34268


namespace simplify_expression_l342_34298

theorem simplify_expression :
  (-2) ^ 2006 + (-1) ^ 3007 + 1 ^ 3010 - (-2) ^ 2007 = -2 ^ 2006 := 
sorry

end simplify_expression_l342_34298


namespace A_iff_B_l342_34265

-- Define Proposition A: ab > b^2
def PropA (a b : ℝ) : Prop := a * b > b ^ 2

-- Define Proposition B: 1/b < 1/a < 0
def PropB (a b : ℝ) : Prop := 1 / b < 1 / a ∧ 1 / a < 0

theorem A_iff_B (a b : ℝ) : (PropA a b) ↔ (PropB a b) := sorry

end A_iff_B_l342_34265


namespace hearty_buys_red_packages_l342_34285

-- Define the conditions
def packages_of_blue := 3
def beads_per_package := 40
def total_beads := 320

-- Calculate the number of blue beads
def blue_beads := packages_of_blue * beads_per_package

-- Calculate the number of red beads
def red_beads := total_beads - blue_beads

-- Prove that the number of red packages is 5
theorem hearty_buys_red_packages : (red_beads / beads_per_package) = 5 := by
  sorry

end hearty_buys_red_packages_l342_34285


namespace range_of_x_l342_34274

variable {p : ℝ} {x : ℝ}

theorem range_of_x (h : 0 ≤ p ∧ p ≤ 4) : x^2 + p * x > 4 * x + p - 3 ↔ (x ≤ -1 ∨ x ≥ 3) :=
sorry

end range_of_x_l342_34274


namespace like_terms_exponents_l342_34242

theorem like_terms_exponents (m n : ℤ) 
  (h1 : 3 = m - 2) 
  (h2 : n + 1 = 2) : m - n = 4 := 
by
  sorry

end like_terms_exponents_l342_34242


namespace parabola_directrix_standard_eq_l342_34207

theorem parabola_directrix_standard_eq (p : ℝ) (h : p = 2) :
  ∀ y x : ℝ, (x = -1) → (y^2 = 4 * x) :=
by
  sorry

end parabola_directrix_standard_eq_l342_34207


namespace work_rate_B_l342_34235

theorem work_rate_B :
  (∀ A B : ℝ, A = 30 → (1 / A + 1 / B = 1 / 19.411764705882355) → B = 55) := by 
    intro A B A_cond combined_rate
    have hA : A = 30 := A_cond
    rw [hA] at combined_rate
    sorry

end work_rate_B_l342_34235


namespace SarahsScoreIs135_l342_34214

variable (SarahsScore GregsScore : ℕ)

-- Conditions
def ScoreDifference (SarahsScore GregsScore : ℕ) : Prop := SarahsScore = GregsScore + 50
def AverageScore (SarahsScore GregsScore : ℕ) : Prop := (SarahsScore + GregsScore) / 2 = 110

-- Theorem statement
theorem SarahsScoreIs135 (h1 : ScoreDifference SarahsScore GregsScore) (h2 : AverageScore SarahsScore GregsScore) : SarahsScore = 135 :=
sorry

end SarahsScoreIs135_l342_34214


namespace compute_expression_l342_34254

theorem compute_expression : 45 * 1313 - 10 * 1313 = 45955 := by
  sorry

end compute_expression_l342_34254


namespace remainder_is_one_l342_34226

theorem remainder_is_one (dividend divisor quotient remainder : ℕ) 
  (h1 : dividend = 222) 
  (h2 : divisor = 13)
  (h3 : quotient = 17)
  (h4 : dividend = divisor * quotient + remainder) : remainder = 1 :=
sorry

end remainder_is_one_l342_34226


namespace remainder_div_8_l342_34230

theorem remainder_div_8 (x : ℤ) (h : ∃ k : ℤ, x = 63 * k + 27) : x % 8 = 3 :=
by
  sorry

end remainder_div_8_l342_34230


namespace yogurt_price_l342_34294

theorem yogurt_price (x y : ℝ) (h1 : 4 * x + 4 * y = 14) (h2 : 2 * x + 8 * y = 13) : x = 2.5 :=
by
  sorry

end yogurt_price_l342_34294


namespace sin_75_is_sqrt_6_add_sqrt_2_div_4_l342_34213

noncomputable def sin_75_angle (a : Real) (b : Real) : Real :=
  Real.sin (75 * Real.pi / 180)

theorem sin_75_is_sqrt_6_add_sqrt_2_div_4 :
  sin_75_angle π (π / 6) = (Real.sqrt 6 + Real.sqrt 2) / 4 :=
by
  sorry

end sin_75_is_sqrt_6_add_sqrt_2_div_4_l342_34213


namespace probability_A_not_lose_l342_34258

-- Define the probabilities
def P_A_wins : ℝ := 0.30
def P_draw : ℝ := 0.25
def P_A_not_lose : ℝ := 0.55

-- Statement to prove
theorem probability_A_not_lose : P_A_wins + P_draw = P_A_not_lose :=
by 
  sorry

end probability_A_not_lose_l342_34258
