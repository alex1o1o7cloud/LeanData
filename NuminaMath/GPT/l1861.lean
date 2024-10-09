import Mathlib

namespace calculate_initial_money_l1861_186147

noncomputable def initial_money (remaining_money: ℝ) (spent_percent: ℝ) : ℝ :=
  remaining_money / (1 - spent_percent)

theorem calculate_initial_money :
  initial_money 3500 0.30 = 5000 := 
by
  rw [initial_money]
  sorry

end calculate_initial_money_l1861_186147


namespace find_height_l1861_186128

-- Definitions from the problem conditions
def Area : ℕ := 442
def width : ℕ := 7
def length : ℕ := 8

-- The statement to prove
theorem find_height (h : ℕ) (H : 2 * length * width + 2 * length * h + 2 * width * h = Area) : h = 11 := 
by
  sorry

end find_height_l1861_186128


namespace arithmetic_expression_l1861_186132

theorem arithmetic_expression : (4 + 6 + 4) / 3 - 4 / 3 = 10 / 3 := by
  sorry

end arithmetic_expression_l1861_186132


namespace polynomial_coefficient_product_identity_l1861_186140

theorem polynomial_coefficient_product_identity (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ)
  (h1 : a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 0)
  (h2 : a_0 - a_1 + a_2 - a_3 + a_4 - a_5 = 32) :
  (a_0 + a_2 + a_4) * (a_1 + a_3 + a_5) = -256 := 
by {
  sorry
}

end polynomial_coefficient_product_identity_l1861_186140


namespace ball_box_arrangement_l1861_186154

-- Given n distinguishable balls and m distinguishable boxes,
-- prove that the number of ways to place the n balls into the m boxes is m^n.
-- Specifically for n = 6 and m = 3.

theorem ball_box_arrangement : (3^6 = 729) :=
by
  sorry

end ball_box_arrangement_l1861_186154


namespace sofa_price_is_correct_l1861_186145

def price_sofa (invoice_total armchair_price table_price : ℕ) (armchair_count : ℕ) : ℕ :=
  invoice_total - (armchair_price * armchair_count + table_price)

theorem sofa_price_is_correct
  (invoice_total : ℕ)
  (armchair_price : ℕ)
  (table_price : ℕ)
  (armchair_count : ℕ)
  (sofa_price : ℕ)
  (h_invoice : invoice_total = 2430)
  (h_armchair_price : armchair_price = 425)
  (h_table_price : table_price = 330)
  (h_armchair_count : armchair_count = 2)
  (h_sofa_price : sofa_price = 1250) :
  price_sofa invoice_total armchair_price table_price armchair_count = sofa_price :=
by
  sorry

end sofa_price_is_correct_l1861_186145


namespace work_done_in_a_day_l1861_186188

noncomputable def A : ℕ := sorry
noncomputable def B_days : ℕ := A / 2

theorem work_done_in_a_day (h : 1 / A + 2 / A = 1 / 6) : A = 18 := 
by 
  -- skipping the proof as instructed
  sorry

end work_done_in_a_day_l1861_186188


namespace sphere_to_cube_volume_ratio_l1861_186144

noncomputable def volume_ratio (s : ℝ) : ℝ :=
  let r := s / 4
  let V_s := (4/3:ℝ) * Real.pi * r^3 
  let V_c := s^3
  V_s / V_c

theorem sphere_to_cube_volume_ratio (s : ℝ) (h : s > 0) : volume_ratio s = Real.pi / 48 := by
  sorry

end sphere_to_cube_volume_ratio_l1861_186144


namespace find_interval_n_l1861_186166

theorem find_interval_n 
  (n : ℕ) 
  (h1 : n < 500)
  (h2 : (∃ abcde : ℕ, 0 < abcde ∧ abcde < 99999 ∧ n * abcde = 99999))
  (h3 : (∃ uvw : ℕ, 0 < uvw ∧ uvw < 999 ∧ (n + 3) * uvw = 999)) 
  : 201 ≤ n ∧ n ≤ 300 := 
sorry

end find_interval_n_l1861_186166


namespace factorize_quadratic_l1861_186148

theorem factorize_quadratic (x : ℝ) : 2*x^2 - 4*x + 2 = 2*(x-1)^2 :=
by
  sorry

end factorize_quadratic_l1861_186148


namespace yura_catches_up_l1861_186103

theorem yura_catches_up (a : ℕ) (x : ℕ) (h1 : 2 * a * x = a * (x + 5)) : x = 5 :=
by
  sorry

end yura_catches_up_l1861_186103


namespace cost_of_largest_pot_l1861_186157

theorem cost_of_largest_pot
  (total_cost : ℝ)
  (n : ℕ)
  (a b : ℝ)
  (h_total_cost : total_cost = 7.80)
  (h_n : n = 6)
  (h_b : b = 0.25)
  (h_small_cost : ∃ x : ℝ, ∃ is_odd : ℤ → Prop, (∃ c: ℤ, x = c / 100 ∧ is_odd c) ∧
                  total_cost = x + (x + b) + (x + 2 * b) + (x + 3 * b) + (x + 4 * b) + (x + 5 * b)) :
  ∃ y, y = (x + 5*b) ∧ y = 1.92 :=
  sorry

end cost_of_largest_pot_l1861_186157


namespace teal_bluish_count_l1861_186146

theorem teal_bluish_count (n G Bg N B : ℕ) (h1 : n = 120) (h2 : G = 80) (h3 : Bg = 35) (h4 : N = 20) :
  B = 55 :=
by
  sorry

end teal_bluish_count_l1861_186146


namespace rate_of_Y_l1861_186171

noncomputable def rate_X : ℝ := 2
noncomputable def time_to_cross : ℝ := 0.5

theorem rate_of_Y (rate_Y : ℝ) : rate_X * time_to_cross = 1 → rate_Y * time_to_cross = 1 → rate_Y = rate_X :=
by
    intros h_rate_X h_rate_Y
    sorry

end rate_of_Y_l1861_186171


namespace solve_absolute_value_equation_l1861_186108

theorem solve_absolute_value_equation (y : ℝ) :
  (|y - 8| + 3 * y = 11) → (y = 1.5) :=
by
  sorry

end solve_absolute_value_equation_l1861_186108


namespace divide_numbers_into_consecutive_products_l1861_186104

theorem divide_numbers_into_consecutive_products :
  ∃ (A B : Finset ℕ), A ∪ B = {2, 3, 5, 7, 11, 13, 17} ∧ A ∩ B = ∅ ∧ 
  (A.prod id = 714 ∧ B.prod id = 715 ∨ A.prod id = 715 ∧ B.prod id = 714) :=
sorry

end divide_numbers_into_consecutive_products_l1861_186104


namespace range_f_l1861_186185

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 1)

theorem range_f : Set.Ioo 0 1 ∪ {1} = {y : ℝ | ∃ x : ℝ, f x = y} :=
by 
  sorry

end range_f_l1861_186185


namespace bob_speed_l1861_186176

theorem bob_speed (j_speed : ℝ) (b_headstart : ℝ) (t : ℝ) (j_catches_up : t = 20 / 60 ∧ j_speed = 9 ∧ b_headstart = 1) : 
  ∃ b_speed : ℝ, b_speed = 6 := 
by
  sorry

end bob_speed_l1861_186176


namespace factors_of_P_factorization_of_P_factorize_expression_l1861_186119

noncomputable def P (a b c : ℝ) : ℝ :=
  a^2 * (b - c) + b^2 * (c - a) + c^2 * (a - b)

theorem factors_of_P (a b c : ℝ) :
  (a - b ∣ P a b c) ∧ (b - c ∣ P a b c) ∧ (c - a ∣ P a b c) :=
sorry

theorem factorization_of_P (a b c : ℝ) :
  P a b c = -(a - b) * (b - c) * (c - a) :=
sorry

theorem factorize_expression (x y z : ℝ) :
  (x + y + z)^3 - x^3 - y^3 - z^3 = 3 * (x + y) * (y + z) * (z + x) :=
sorry

end factors_of_P_factorization_of_P_factorize_expression_l1861_186119


namespace hypotenuse_length_l1861_186137

theorem hypotenuse_length
  (a b c : ℝ)
  (h1 : a + b + c = 40)
  (h2 : (1 / 2) * a * b = 24)
  (h3 : a^2 + b^2 = c^2) :
  c = 18.8 :=
by sorry

end hypotenuse_length_l1861_186137


namespace sum_S16_over_S4_l1861_186196

variable {α : Type*} [LinearOrderedField α]

def geometric_sequence (a q : α) (n : ℕ) := a * q^n

def sum_of_first_n_terms (a q : α) (n : ℕ) : α :=
if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem sum_S16_over_S4
  (a q : α)
  (hq : q ≠ 1)
  (h8_over_4 : sum_of_first_n_terms a q 8 / sum_of_first_n_terms a q 4 = 3) :
  sum_of_first_n_terms a q 16 / sum_of_first_n_terms a q 4 = 15 :=
sorry

end sum_S16_over_S4_l1861_186196


namespace volume_range_l1861_186182

theorem volume_range (a b c : ℝ) (h1 : a + b + c = 9)
  (h2 : a * b + b * c + a * c = 24) : 16 ≤ a * b * c ∧ a * b * c ≤ 20 :=
by {
  -- Proof would go here
  sorry
}

end volume_range_l1861_186182


namespace product_divisible_by_eight_l1861_186141

theorem product_divisible_by_eight (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 96) : 
  8 ∣ n * (n + 1) * (n + 2) := 
sorry

end product_divisible_by_eight_l1861_186141


namespace p_work_alone_time_l1861_186149

variable (Wp Wq : ℝ)
variable (x : ℝ)

-- Conditions
axiom h1 : Wp = 1.5 * Wq
axiom h2 : (1 / x) + (Wq / Wp) * (1 / x) = 1 / 15

-- Proof of the question (p alone can complete the work in x days)
theorem p_work_alone_time : x = 25 :=
by
  -- Add your proof here
  sorry

end p_work_alone_time_l1861_186149


namespace work_done_l1861_186127

theorem work_done (m : ℕ) : 18 * 30 = m * 36 → m = 15 :=
by
  intro h  -- assume the equality condition
  have h1 : m = 15 := by
    -- We would solve for m here similarly to the solution given to derive 15
    sorry
  exact h1

end work_done_l1861_186127


namespace solve_x_l1861_186163

theorem solve_x (x : ℝ) (h : (30 * x + 15)^(1/3) = 15) : x = 112 := by
  sorry

end solve_x_l1861_186163


namespace axis_of_symmetry_r_minus_2s_zero_l1861_186184

/-- 
Prove that if y = x is an axis of symmetry for the curve 
y = (2 * p * x + q) / (r * x - 2 * s) with p, q, r, s nonzero, 
then r - 2s = 0. 
-/
theorem axis_of_symmetry_r_minus_2s_zero
  (p q r s : ℝ) (h_p : p ≠ 0) (h_q : q ≠ 0) (h_r : r ≠ 0) (h_s : s ≠ 0) 
  (h_sym : ∀ (a b : ℝ), (b = (2 * p * a + q) / (r * a - 2 * s)) ↔ (a = (2 * p * b + q) / (r * b - 2 * s))) :
  r - 2 * s = 0 :=
sorry

end axis_of_symmetry_r_minus_2s_zero_l1861_186184


namespace shift_gives_f_l1861_186178

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 6)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x)

theorem shift_gives_f :
  (∀ x, f x = g (x + Real.pi / 3)) :=
  by
  sorry

end shift_gives_f_l1861_186178


namespace not_obtain_other_than_given_set_l1861_186112

theorem not_obtain_other_than_given_set : 
  ∀ (x : ℝ), x = 1 → 
  ∃ (n : ℕ → ℝ), (n 0 = 1) ∧ 
  (∀ k, n (k + 1) = n k + 1 ∨ n (k + 1) = -1 / n k) ∧
  (x = -2 ∨ x = 1/2 ∨ x = 5/3 ∨ x = 7) → 
  ∃ k, x = n k :=
sorry

end not_obtain_other_than_given_set_l1861_186112


namespace smaller_part_volume_l1861_186164

noncomputable def volume_of_smaller_part (a : ℝ) : ℝ :=
  (25 / 144) * (a^3)

theorem smaller_part_volume (a : ℝ) (h_pos : 0 < a) :
  ∃ v : ℝ, v = volume_of_smaller_part a :=
  sorry

end smaller_part_volume_l1861_186164


namespace a_greater_than_1_and_b_less_than_1_l1861_186189

theorem a_greater_than_1_and_b_less_than_1
  (a b c : ℝ) 
  (h1 : a ≥ b)
  (h2 : b ≥ c)
  (h3 : c > 0)
  (h4 : a * b * c = 1)
  (h5 : a + b + c > 1/a + 1/b + 1/c) : a > 1 ∧ b < 1 :=
by
  sorry

end a_greater_than_1_and_b_less_than_1_l1861_186189


namespace bottle_caps_per_visit_l1861_186169

-- Define the given conditions
def total_bottle_caps : ℕ := 25
def number_of_visits : ℕ := 5

-- The statement we want to prove
theorem bottle_caps_per_visit :
  total_bottle_caps / number_of_visits = 5 :=
sorry

end bottle_caps_per_visit_l1861_186169


namespace bill_take_home_salary_l1861_186111

-- Define the parameters
def property_taxes : ℝ := 2000
def sales_taxes : ℝ := 3000
def gross_salary : ℝ := 50000
def income_tax_rate : ℝ := 0.10

-- Define income tax calculation
def income_tax : ℝ := income_tax_rate * gross_salary

-- Define total taxes calculation
def total_taxes : ℝ := property_taxes + sales_taxes + income_tax

-- Define the take-home salary calculation
def take_home_salary : ℝ := gross_salary - total_taxes

-- Statement of the theorem
theorem bill_take_home_salary : take_home_salary = 40000 := by
  -- Sorry is used to skip the proof.
  sorry

end bill_take_home_salary_l1861_186111


namespace quincy_monthly_payment_l1861_186186

-- Definitions based on the conditions:
def car_price : ℕ := 20000
def down_payment : ℕ := 5000
def loan_years : ℕ := 5
def months_in_year : ℕ := 12

-- The mathematical problem to be proven:
theorem quincy_monthly_payment :
  let amount_to_finance := car_price - down_payment
  let total_months := loan_years * months_in_year
  amount_to_finance / total_months = 250 := by
  sorry

end quincy_monthly_payment_l1861_186186


namespace determinant_zero_implies_sum_l1861_186122

open Matrix

noncomputable def matrix_example (a b : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![2, 5, 8],
    ![4, a, b],
    ![4, b, a]
  ]

theorem determinant_zero_implies_sum (a b : ℝ) (h : a ≠ b) (h_det : det (matrix_example a b) = 0) : a + b = 26 :=
by
  sorry

end determinant_zero_implies_sum_l1861_186122


namespace sprinkler_days_needed_l1861_186125

-- Definitions based on the conditions
def morning_water : ℕ := 4
def evening_water : ℕ := 6
def daily_water : ℕ := morning_water + evening_water
def total_water_needed : ℕ := 50

-- The proof statement
theorem sprinkler_days_needed : total_water_needed / daily_water = 5 := by
  sorry

end sprinkler_days_needed_l1861_186125


namespace amit_work_days_l1861_186123

variable (x : ℕ)

theorem amit_work_days
  (ananthu_rate : ℚ := 1/30) -- Ananthu's work rate is 1/30
  (amit_days : ℕ := 3) -- Amit worked for 3 days
  (ananthu_days : ℕ := 24) -- Ananthu worked for remaining 24 days
  (total_days : ℕ := 27) -- Total work completed in 27 days
  (amit_work: ℚ := amit_days * 1/x) -- Amit's work rate
  (ananthu_work: ℚ := ananthu_days * ananthu_rate) -- Ananthu's work rate
  (total_work : ℚ := 1) -- Total work completed  
  : 3 * (1/x) + 24 * (1/30) = 1 ↔ x = 15 := 
by
  sorry

end amit_work_days_l1861_186123


namespace largest_common_term_arith_progressions_l1861_186134

theorem largest_common_term_arith_progressions (a : ℕ) : 
  (∃ n m : ℕ, a = 4 + 5 * n ∧ a = 3 + 9 * m ∧ a < 1000) → a = 984 := by
  -- Proof is not required, so we add sorry.
  sorry

end largest_common_term_arith_progressions_l1861_186134


namespace store_total_income_l1861_186107

def pencil_with_eraser_cost : ℝ := 0.8
def regular_pencil_cost : ℝ := 0.5
def short_pencil_cost : ℝ := 0.4

def pencils_with_eraser_sold : ℕ := 200
def regular_pencils_sold : ℕ := 40
def short_pencils_sold : ℕ := 35

noncomputable def total_money_made : ℝ :=
  (pencil_with_eraser_cost * pencils_with_eraser_sold) +
  (regular_pencil_cost * regular_pencils_sold) +
  (short_pencil_cost * short_pencils_sold)

theorem store_total_income : total_money_made = 194 := by
  sorry

end store_total_income_l1861_186107


namespace syllogism_major_minor_premise_l1861_186101

theorem syllogism_major_minor_premise
(people_of_Yaan_strong_unyielding : Prop)
(people_of_Yaan_Chinese : Prop)
(all_Chinese_strong_unyielding : Prop) :
  all_Chinese_strong_unyielding ∧ people_of_Yaan_Chinese → (all_Chinese_strong_unyielding = all_Chinese_strong_unyielding ∧ people_of_Yaan_Chinese = people_of_Yaan_Chinese) :=
by
  intros h
  exact ⟨rfl, rfl⟩

end syllogism_major_minor_premise_l1861_186101


namespace simplify_expression_l1861_186135

variable (a b : ℤ)

theorem simplify_expression :
  (30 * a + 45 * b) + (15 * a + 40 * b) - (20 * a + 55 * b) + (5 * a - 10 * b) = 30 * a + 20 * b :=
by
  sorry

end simplify_expression_l1861_186135


namespace only_three_A_l1861_186187

def student := Type
variable (Alan Beth Carlos Diana Eliza : student)

variable (gets_A : student → Prop)

variable (H1 : gets_A Alan → gets_A Beth)
variable (H2 : gets_A Beth → gets_A Carlos)
variable (H3 : gets_A Carlos → gets_A Diana)
variable (H4 : gets_A Diana → gets_A Eliza)
variable (H5 : gets_A Eliza → gets_A Alan)
variable (H6 : ∃ a b c : student, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ gets_A a ∧ gets_A b ∧ gets_A c ∧ ∀ d : student, gets_A d → d = a ∨ d = b ∨ d = c)

theorem only_three_A : gets_A Carlos ∧ gets_A Diana ∧ gets_A Eliza :=
by
  sorry

end only_three_A_l1861_186187


namespace smallest_integer_l1861_186198

theorem smallest_integer (M : ℕ) :
  (M % 4 = 3) ∧ (M % 5 = 4) ∧ (M % 6 = 5) ∧ (M % 7 = 6) ∧
  (M % 8 = 7) ∧ (M % 9 = 8) → M = 2519 :=
by sorry

end smallest_integer_l1861_186198


namespace price_decrease_proof_l1861_186130

-- Definitions based on the conditions
def original_price (C : ℝ) : ℝ := C
def new_price (C : ℝ) : ℝ := 0.76 * C

theorem price_decrease_proof (C : ℝ) : new_price C = 421.05263157894734 :=
by
  sorry

end price_decrease_proof_l1861_186130


namespace compute_value_l1861_186124

theorem compute_value (a b c : ℕ) (h : a = 262 ∧ b = 258 ∧ c = 150) : 
  (a^2 - b^2) + c = 2230 := 
by
  sorry

end compute_value_l1861_186124


namespace fuel_tank_capacity_l1861_186105

theorem fuel_tank_capacity (C : ℝ) 
  (h1 : 0.12 * 98 + 0.16 * (C - 98) = 30) : 
  C = 212 :=
by
  sorry

end fuel_tank_capacity_l1861_186105


namespace smallest_four_digit_divisible_by_primes_l1861_186120

theorem smallest_four_digit_divisible_by_primes :
  ∃ n, 1000 ≤ n ∧ n ≤ 9999 ∧ (∀ p ∈ [2, 3, 5, 7], p ∣ n) ∧ n = 1050 :=
by
  sorry

end smallest_four_digit_divisible_by_primes_l1861_186120


namespace mnp_sum_correct_l1861_186192

noncomputable def mnp_sum : ℕ :=
  let m := 1032
  let n := 40
  let p := 3
  m + n + p

theorem mnp_sum_correct : mnp_sum = 1075 := by
  -- Given the conditions, the established value for m, n, and p should sum to 1075
  sorry

end mnp_sum_correct_l1861_186192


namespace average_weight_of_all_girls_l1861_186172

theorem average_weight_of_all_girls (avg1 : ℝ) (n1 : ℕ) (avg2 : ℝ) (n2 : ℕ) :
  avg1 = 50.25 → n1 = 16 → avg2 = 45.15 → n2 = 8 → 
  ((n1 * avg1 + n2 * avg2) / (n1 + n2)) = 48.55 := 
by
  intros h1 h2 h3 h4
  sorry

end average_weight_of_all_girls_l1861_186172


namespace find_number_l1861_186117

theorem find_number (n : ℝ) (x : ℕ) (h1 : x = 4) (h2 : n^(2*x) = 3^(12-x)) : n = 3 := by
  sorry

end find_number_l1861_186117


namespace students_in_class_l1861_186126

theorem students_in_class (S : ℕ) 
  (h1 : chess_students = S / 3)
  (h2 : tournament_students = chess_students / 2)
  (h3 : tournament_students = 4) : 
  S = 24 :=
by
  sorry

end students_in_class_l1861_186126


namespace exercise_l1861_186193

theorem exercise (n : ℕ) : 900 ∣ (6 ^ (2 * (n + 1)) - 2 ^ (n + 3) * 3 ^ (n + 2) + 36) :=
sorry

end exercise_l1861_186193


namespace number_division_l1861_186177

theorem number_division (m k n : ℤ) (h : n = m * k + 1) : n = m * k + 1 :=
by
  exact h

end number_division_l1861_186177


namespace sum_six_smallest_multiples_of_12_is_252_l1861_186159

-- Define the six smallest positive distinct multiples of 12
def six_smallest_multiples_of_12 := [12, 24, 36, 48, 60, 72]

-- Define the sum problem
def sum_of_six_smallest_multiples_of_12 : Nat :=
  six_smallest_multiples_of_12.foldr (· + ·) 0

-- Main proof statement
theorem sum_six_smallest_multiples_of_12_is_252 :
  sum_of_six_smallest_multiples_of_12 = 252 :=
by
  sorry

end sum_six_smallest_multiples_of_12_is_252_l1861_186159


namespace combined_age_of_Jane_and_John_in_future_l1861_186181

def Justin_age : ℕ := 26
def Jessica_age_when_Justin_born : ℕ := 6
def James_older_than_Jessica : ℕ := 7
def Julia_younger_than_Justin : ℕ := 8
def Jane_older_than_James : ℕ := 25
def John_older_than_Jane : ℕ := 3
def years_later : ℕ := 12

theorem combined_age_of_Jane_and_John_in_future :
  let Jessica_age := Justin_age + Jessica_age_when_Justin_born
  let James_age := Jessica_age + James_older_than_Jessica
  let Julia_age := Justin_age - Julia_younger_than_Justin
  let Jane_age := James_age + Jane_older_than_James
  let John_age := Jane_age + John_older_than_Jane
  let Jane_age_after_years := Jane_age + years_later
  let John_age_after_years := John_age + years_later
  Jane_age_after_years + John_age_after_years = 155 :=
by
  sorry

end combined_age_of_Jane_and_John_in_future_l1861_186181


namespace ellipse_hyperbola_foci_l1861_186151

theorem ellipse_hyperbola_foci (a b : ℝ) 
  (h1 : ∃ (a b : ℝ), b^2 - a^2 = 25 ∧ a^2 + b^2 = 64) : 
  |a * b| = (Real.sqrt 3471) / 2 :=
by
  sorry

end ellipse_hyperbola_foci_l1861_186151


namespace limit_of_R_l1861_186113

noncomputable def R (m b : ℝ) : ℝ :=
  let x := ((-b) + Real.sqrt (b^2 + 4 * m)) / 2
  m * x + 3 

theorem limit_of_R (b : ℝ) (hb : b ≠ 0) : 
  (∀ m : ℝ, m < 3) → 
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 0) < δ → abs ((R x (-b) - R x b) / x - b) < ε) :=
by
  sorry

end limit_of_R_l1861_186113


namespace cos_210_eq_neg_sqrt3_div_2_l1861_186180

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (Real.pi + Real.pi / 6) = -Real.sqrt 3 / 2 := sorry

end cos_210_eq_neg_sqrt3_div_2_l1861_186180


namespace division_of_expressions_l1861_186190

theorem division_of_expressions : 
  (2 * 3 + 4) / (2 + 3) = 2 :=
by
  sorry

end division_of_expressions_l1861_186190


namespace fraction_of_liars_l1861_186110

theorem fraction_of_liars (n : ℕ) (villagers : Fin n → Prop) (right_neighbor : ∀ i, villagers i ↔ ∀ j : Fin n, j = (i + 1) % n → villagers j) :
  ∃ (x : ℚ), x = 1 / 2 :=
by 
  sorry

end fraction_of_liars_l1861_186110


namespace product_greater_than_sum_l1861_186168

variable {a b : ℝ}

theorem product_greater_than_sum (ha : a > 2) (hb : b > 2) : a * b > a + b := 
  sorry

end product_greater_than_sum_l1861_186168


namespace find_p1_plus_q1_l1861_186173

noncomputable def p (x : ℤ) := x^4 + 14 * x^2 + 1
noncomputable def q (x : ℤ) := x^4 - 14 * x^2 + 1

theorem find_p1_plus_q1 :
  (p 1) + (q 1) = 4 :=
sorry

end find_p1_plus_q1_l1861_186173


namespace value_of_f_neg_2_l1861_186183

section
variable {f : ℝ → ℝ}
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_pos : ∀ x : ℝ, 0 < x → f x = 2 ^ x + 1)

theorem value_of_f_neg_2 (h_odd : ∀ x, f (-x) = -f x) (h_pos : ∀ x, 0 < x → f x = 2^x + 1) :
  f (-2) = -5 :=
by
  sorry
end

end value_of_f_neg_2_l1861_186183


namespace ratio_expression_value_l1861_186179

variable {A B C : ℚ}

theorem ratio_expression_value (h : A / B = 3 / 2 ∧ A / C = 3 / 6) : (4 * A - 3 * B) / (5 * C + 2 * A) = 1 / 4 := 
sorry

end ratio_expression_value_l1861_186179


namespace vasya_tolya_badges_l1861_186100

theorem vasya_tolya_badges (x y : ℤ)
    (h1 : y = x + 5) -- Vasya initially had 5 more badges than Tolya
    (h2 : (y - (6 * (y / 25) / 25) + (4 * x) / 25) = (x - (4 * x) / 5 + 6 * (y / 25) / 5 - 1)) : -- equation balancing after exchange
    x = 45 ∧ y = 50 := 
sorry

end vasya_tolya_badges_l1861_186100


namespace correct_option_C_l1861_186152

def number_of_stamps : String := "the number of the stamps"
def number_of_people : String := "a number of people"

def is_singular (subject : String) : Prop := subject = number_of_stamps
def is_plural (subject : String) : Prop := subject = number_of_people

def correct_sentence (verb1 verb2 : String) : Prop :=
  verb1 = "is" ∧ verb2 = "want"

theorem correct_option_C : correct_sentence "is" "want" :=
by
  show correct_sentence "is" "want"
  -- Proof is omitted
  sorry

end correct_option_C_l1861_186152


namespace not_in_M_4n2_l1861_186106

def M : Set ℤ := {a | ∃ x y : ℤ, a = x^2 - y^2}

theorem not_in_M_4n2 (n : ℤ) : ¬ (4 * n + 2 ∈ M) :=
by
sorry

end not_in_M_4n2_l1861_186106


namespace quadratic_real_roots_l1861_186114

theorem quadratic_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
  sorry

end quadratic_real_roots_l1861_186114


namespace interior_edges_sum_l1861_186153

-- Definitions based on conditions
def frame_width : ℕ := 2
def frame_area : ℕ := 32
def outer_edge_length : ℕ := 8

-- Mathematically equivalent proof problem
theorem interior_edges_sum :
  ∃ (y : ℕ),  (frame_width * 2) * (y - frame_width * 2) = 32 ∧ (outer_edge_length * y - (outer_edge_length - 2 * frame_width) * (y - 2 * frame_width)) = 32 -> 4 + 4 + 0 + 0 = 8 :=
sorry

end interior_edges_sum_l1861_186153


namespace find_monic_polynomial_of_shifted_roots_l1861_186161

theorem find_monic_polynomial_of_shifted_roots (a b c : ℝ) (h : ∀ x : ℝ, (x - a) * (x - b) * (x - c) = x^3 - 5 * x + 7) : 
  (x : ℝ) → (x - (a - 3)) * (x - (b - 3)) * (x - (c - 3)) = x^3 + 9 * x^2 + 22 * x + 19 :=
by
  -- Proof will be provided here.
  sorry

end find_monic_polynomial_of_shifted_roots_l1861_186161


namespace problem_solve_l1861_186121

theorem problem_solve (n : ℕ) (h_pos : 0 < n) 
    (h_eq : Real.sin (Real.pi / (3 * n)) + Real.cos (Real.pi / (3 * n)) = Real.sqrt (2 * n) / 3) : 
    n = 6 := 
  sorry

end problem_solve_l1861_186121


namespace abs_sum_lt_abs_sum_of_neg_product_l1861_186131

theorem abs_sum_lt_abs_sum_of_neg_product 
  (a b : ℝ) : ab < 0 ↔ |a + b| < |a| + |b| := 
by 
  sorry

end abs_sum_lt_abs_sum_of_neg_product_l1861_186131


namespace time_to_cut_womans_hair_l1861_186156

theorem time_to_cut_womans_hair 
  (WL : ℕ) (WM : ℕ) (WK : ℕ) (total_time : ℕ) 
  (num_women : ℕ) (num_men : ℕ) (num_kids : ℕ) 
  (men_haircut_time : ℕ) (kids_haircut_time : ℕ) 
  (overall_time : ℕ) :
  men_haircut_time = 15 →
  kids_haircut_time = 25 →
  num_women = 3 →
  num_men = 2 →
  num_kids = 3 →
  overall_time = 255 →
  overall_time = (num_women * WL + num_men * men_haircut_time + num_kids * kids_haircut_time) →
  WL = 50 :=
by
  sorry

end time_to_cut_womans_hair_l1861_186156


namespace largest_multiple_of_7_less_than_neg_30_l1861_186195

theorem largest_multiple_of_7_less_than_neg_30 (m : ℤ) (h1 : m % 7 = 0) (h2 : m < -30) : m = -35 :=
sorry

end largest_multiple_of_7_less_than_neg_30_l1861_186195


namespace jewelry_store_gross_profit_l1861_186133

theorem jewelry_store_gross_profit (purchase_price selling_price new_selling_price gross_profit : ℝ)
    (h1 : purchase_price = 240)
    (h2 : markup = 0.25 * selling_price)
    (h3 : selling_price = purchase_price + markup)
    (h4 : decrease = 0.20 * selling_price)
    (h5 : new_selling_price = selling_price - decrease)
    (h6 : gross_profit = new_selling_price - purchase_price) :
    gross_profit = 16 :=
by
    sorry

end jewelry_store_gross_profit_l1861_186133


namespace line_equation_l1861_186194

theorem line_equation (P : ℝ × ℝ) (slope : ℝ) (hP : P = (-2, 0)) (hSlope : slope = 3) :
    ∃ (a b : ℝ), ∀ x y : ℝ, y = a * x + b ↔ P.1 = -2 ∧ P.2 = 0 ∧ slope = 3 ∧ y = 3 * x + 6 :=
by
  sorry

end line_equation_l1861_186194


namespace email_sending_ways_l1861_186116

theorem email_sending_ways (n k : ℕ) (hn : n = 3) (hk : k = 5) : n^k = 243 := 
by
  sorry

end email_sending_ways_l1861_186116


namespace new_energy_vehicle_price_l1861_186165

theorem new_energy_vehicle_price (x : ℝ) :
  (5000 / (x + 1)) = (5000 * (1 - 0.2)) / x :=
sorry

end new_energy_vehicle_price_l1861_186165


namespace number_of_common_tangents_l1861_186197

noncomputable def circle1_center : ℝ × ℝ := (-3, 0)
noncomputable def circle1_radius : ℝ := 4

noncomputable def circle2_center : ℝ × ℝ := (0, 3)
noncomputable def circle2_radius : ℝ := 6

theorem number_of_common_tangents 
  (center1 center2 : ℝ × ℝ)
  (radius1 radius2 : ℝ)
  (h_center1: center1 = (-3, 0))
  (h_radius1: radius1 = 4)
  (h_center2: center2 = (0, 3))
  (h_radius2: radius2 = 6) :
  -- The sought number of common tangents between the two circles
  2 = 2 :=
by
  sorry

end number_of_common_tangents_l1861_186197


namespace max_tan_B_l1861_186102

theorem max_tan_B (A B : ℝ) (C : Prop) 
  (sin_pos_A : 0 < Real.sin A) 
  (sin_pos_B : 0 < Real.sin B) 
  (angle_condition : Real.sin B / Real.sin A = Real.cos (A + B)) :
  Real.tan B ≤ Real.sqrt 2 / 4 :=
by
  sorry

end max_tan_B_l1861_186102


namespace repeated_process_pure_alcohol_l1861_186199

theorem repeated_process_pure_alcohol : 
  ∃ n : ℕ, n ≥ 4 ∧ ∀ m < 4, 2 * (1 / 2 : ℝ)^(m : ℝ) ≥ 0.2 := by
  sorry

end repeated_process_pure_alcohol_l1861_186199


namespace sqrt_abs_eq_zero_imp_power_eq_neg_one_l1861_186160

theorem sqrt_abs_eq_zero_imp_power_eq_neg_one (m n : ℤ) (h : (Real.sqrt (m - 2) + abs (n + 3) = 0)) : (m + n) ^ 2023 = -1 := by
  sorry

end sqrt_abs_eq_zero_imp_power_eq_neg_one_l1861_186160


namespace g_of_f_at_3_eq_1902_l1861_186155

def f (x : ℤ) : ℤ := x^3 - 2
def g (x : ℤ) : ℤ := 3 * x^2 + x + 2

theorem g_of_f_at_3_eq_1902 : g (f 3) = 1902 := by
  sorry

end g_of_f_at_3_eq_1902_l1861_186155


namespace determine_k_l1861_186139

theorem determine_k (a b c k : ℝ) (h : a + b + c = 1) (h_eq : k * (a + bc) = (a + b) * (a + c)) : k = 1 :=
sorry

end determine_k_l1861_186139


namespace scout_hours_worked_l1861_186143

variable (h : ℕ) -- number of hours worked on Saturday
variable (base_pay : ℕ) -- base pay per hour
variable (tip_per_customer : ℕ) -- tip per customer
variable (saturday_customers : ℕ) -- customers served on Saturday
variable (sunday_hours : ℕ) -- hours worked on Sunday
variable (sunday_customers : ℕ) -- customers served on Sunday
variable (total_earnings : ℕ) -- total earnings over the weekend

theorem scout_hours_worked {h : ℕ} (base_pay : ℕ) (tip_per_customer : ℕ) (saturday_customers : ℕ) (sunday_hours : ℕ) (sunday_customers : ℕ) (total_earnings : ℕ) :
  base_pay = 10 → 
  tip_per_customer = 5 → 
  saturday_customers = 5 → 
  sunday_hours = 5 → 
  sunday_customers = 8 → 
  total_earnings = 155 → 
  10 * h + 5 * 5 + 10 * 5 + 5 * 8 = 155 → 
  h = 4 :=
by
  intros
  sorry

end scout_hours_worked_l1861_186143


namespace fair_tickets_sold_l1861_186115

theorem fair_tickets_sold (F : ℕ) (number_of_baseball_game_tickets : ℕ) 
  (h1 : F = 2 * number_of_baseball_game_tickets + 6) (h2 : number_of_baseball_game_tickets = 56) :
  F = 118 :=
by
  sorry

end fair_tickets_sold_l1861_186115


namespace stock_worth_l1861_186174

theorem stock_worth (profit_part loss_part total_loss : ℝ) 
  (h1 : profit_part = 0.10) 
  (h2 : loss_part = 0.90) 
  (h3 : total_loss = 400) 
  (profit_rate : ℝ := 0.20) 
  (loss_rate : ℝ := 0.05)
  (profit_value := profit_rate * profit_part)
  (loss_value := loss_rate * loss_part)
  (overall_loss := total_loss)
  (h4 : loss_value - profit_value = overall_loss) :
  ∃ X : ℝ, X = 16000 :=
by
  sorry

end stock_worth_l1861_186174


namespace coefficient_x_is_five_l1861_186175

theorem coefficient_x_is_five (x y a : ℤ) (h1 : a * x + y = 19) (h2 : x + 3 * y = 1) (h3 : 3 * x + 2 * y = 10) : a = 5 :=
by sorry

end coefficient_x_is_five_l1861_186175


namespace distance_from_A_to_O_is_3_l1861_186136

-- Define polar coordinates with the given conditions
def point_A : ℝ × ℝ := (3, -4)

-- Define the distance function in terms of polar coordinates
def distance_to_pole_O (coords : ℝ × ℝ) : ℝ := coords.1

-- The main theorem to be proved
theorem distance_from_A_to_O_is_3 : distance_to_pole_O point_A = 3 := by
  sorry

end distance_from_A_to_O_is_3_l1861_186136


namespace pyramid_volume_l1861_186142

-- Define the given conditions
def regular_octagon (A B C D E F G H : Point) : Prop := sorry
def right_pyramid (P A B C D E F G H : Point) : Prop := sorry
def equilateral_triangle (P A D : Point) (side_length : ℝ) : Prop := sorry

-- Define the specific pyramid problem with all the given conditions
noncomputable def volume_pyramid (P A B C D E F G H : Point) (height : ℝ) (base_area : ℝ) : ℝ :=
  (1 / 3) * base_area * height

-- The main theorem to prove the volume of the pyramid
theorem pyramid_volume (A B C D E F G H P : Point) 
(h1 : regular_octagon A B C D E F G H)
(h2 : right_pyramid P A B C D E F G H)
(h3 : equilateral_triangle P A D 10) :
  volume_pyramid P A B C D E F G H (5 * Real.sqrt 3) (50 * Real.sqrt 3) = 250 := 
sorry

end pyramid_volume_l1861_186142


namespace initial_thickness_of_blanket_l1861_186191

theorem initial_thickness_of_blanket (T : ℝ)
  (h : ∀ n, n = 4 → T * 2^n = 48) : T = 3 :=
by
  have h4 := h 4 rfl
  sorry

end initial_thickness_of_blanket_l1861_186191


namespace inverse_h_l1861_186150

-- Definitions from the problem conditions
def f (x : ℝ) : ℝ := 4 * x + 2
def g (x : ℝ) : ℝ := 3 * x - 5
def h (x : ℝ) : ℝ := f (g x)

-- Statement of the theorem for the inverse of h
theorem inverse_h : ∀ x : ℝ, h⁻¹ x = (x + 18) / 12 :=
sorry

end inverse_h_l1861_186150


namespace total_cost_of_motorcycle_l1861_186129

-- Definitions from conditions
def total_cost (x : ℝ) := 0.20 * x = 400

-- The theorem to prove
theorem total_cost_of_motorcycle (x : ℝ) (h : total_cost x) : x = 2000 := 
by
  sorry

end total_cost_of_motorcycle_l1861_186129


namespace prime_triple_l1861_186170

theorem prime_triple (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r)
  (h1 : p ∣ (q * r - 1)) (h2 : q ∣ (p * r - 1)) (h3 : r ∣ (p * q - 1)) :
  (p = 2 ∧ q = 3 ∧ r = 5) ∨ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 3 ∧ q = 2 ∧ r = 5) ∨ (p = 3 ∧ q = 5 ∧ r = 2) ∨ (p = 5 ∧ q = 2 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) :=
sorry

end prime_triple_l1861_186170


namespace total_weight_of_fruits_l1861_186162

/-- Define the given conditions in Lean -/
def weight_of_orange_bags (n : ℕ) : ℝ :=
  if n = 12 then 24 else 0

def weight_of_apple_bags (n : ℕ) : ℝ :=
  if n = 8 then 30 else 0

/-- Prove that the total weight of 5 bags of oranges and 4 bags of apples is 25 pounds given the conditions -/
theorem total_weight_of_fruits :
  weight_of_orange_bags 12 / 12 * 5 + weight_of_apple_bags 8 / 8 * 4 = 25 :=
by sorry

end total_weight_of_fruits_l1861_186162


namespace combined_bus_capacity_l1861_186118

-- Define conditions
def train_capacity : ℕ := 120
def bus_capacity : ℕ := train_capacity / 6
def number_of_buses : ℕ := 2

-- Define theorem for the combined capacity of two buses
theorem combined_bus_capacity : number_of_buses * bus_capacity = 40 := by
  -- We declare that the proof is skipped here
  sorry

end combined_bus_capacity_l1861_186118


namespace page_added_twice_l1861_186109

theorem page_added_twice (n k : ℕ) (h1 : (n * (n + 1)) / 2 + k = 1986) : k = 33 :=
sorry

end page_added_twice_l1861_186109


namespace solve_fraction_equation_l1861_186167

theorem solve_fraction_equation (x : ℝ) (h : (x + 5) / (x - 3) = 4) : x = 17 / 3 :=
by
  sorry

end solve_fraction_equation_l1861_186167


namespace regular_polygon_sides_l1861_186158

theorem regular_polygon_sides (exterior_angle : ℕ) (h : exterior_angle = 30) : (360 / exterior_angle) = 12 := by
  sorry

end regular_polygon_sides_l1861_186158


namespace minimum_trees_with_at_least_three_types_l1861_186138

theorem minimum_trees_with_at_least_three_types 
    (total_trees : ℕ)
    (birches spruces pines aspens : ℕ)
    (h_total : total_trees = 100)
    (h_any_85 : ∀ (S : Finset ℕ), S.card = 85 → 
                  (∃ (b s p a : ℕ), b ∈ S ∧ s ∈ S ∧ p ∈ S ∧ a ∈ S)) :
  ∃ (n : ℕ), n = 69 ∧ ∀ (T : Finset ℕ), T.card = n → 
                  ∃ (b s p : ℕ), b ∈ T ∧ s ∈ T ∧ p ∈ T :=
  sorry

end minimum_trees_with_at_least_three_types_l1861_186138
