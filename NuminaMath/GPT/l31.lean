import Mathlib

namespace min_point_transformed_graph_l31_3189

noncomputable def original_eq (x : ℝ) : ℝ := 2 * |x| - 4

noncomputable def translated_eq (x : ℝ) : ℝ := 2 * |x - 3| - 8

theorem min_point_transformed_graph : translated_eq 3 = -8 :=
by
  -- Solution steps would go here
  sorry

end min_point_transformed_graph_l31_3189


namespace sum_of_sides_l31_3119

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (cosB cosC : ℝ)
variable (sinB : ℝ)
variable (area : ℝ)

-- Given conditions
axiom h1 : b = 2
axiom h2 : b * cosC + c * cosB = 3 * a * cosB
axiom h3 : area = 3 * Real.sqrt 2 / 2
axiom h4 : sinB = Real.sqrt (1 - cosB ^ 2)

-- Prove the desired result
theorem sum_of_sides (A B C a b c cosB cosC sinB : ℝ) (area : ℝ)
  (h1 : b = 2)
  (h2 : b * cosC + c * cosB = 3 * a * cosB)
  (h3 : area = 3 * Real.sqrt 2 / 2)
  (h4 : sinB = Real.sqrt (1 - cosB ^ 2)) :
  a + c = 4 := 
sorry

end sum_of_sides_l31_3119


namespace gcd_765432_654321_l31_3117

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 9 := by
  sorry

end gcd_765432_654321_l31_3117


namespace base_329_digits_even_l31_3155

noncomputable def base_of_four_digit_even_final : ℕ := 5

theorem base_329_digits_even (b : ℕ) (h1 : b^3 ≤ 329) (h2 : 329 < b^4)
  (h3 : ∀ d, 329 % b = d → d % 2 = 0) : b = base_of_four_digit_even_final :=
by sorry

end base_329_digits_even_l31_3155


namespace width_of_crate_l31_3161

theorem width_of_crate
  (r : ℝ) (h : ℝ) (w : ℝ)
  (h_crate : h = 6 ∨ h = 10 ∨ w = 6 ∨ w = 10)
  (r_tank : r = 4)
  (height_longest_crate : h > w)
  (maximize_volume : ∃ d : ℝ, d = 2 * r ∧ w = d) :
  w = 8 := 
sorry

end width_of_crate_l31_3161


namespace walking_speed_l31_3102

theorem walking_speed (W : ℝ) : (1 / (1 / W + 1 / 8)) * 6 = 2.25 * (12 / 2) -> W = 4 :=
by
  intro h
  sorry

end walking_speed_l31_3102


namespace log_expression_simplifies_to_zero_l31_3133

theorem log_expression_simplifies_to_zero : 
  (1/2 : ℝ) * (Real.log 4) + Real.log 5 - Real.exp (0 * Real.log (Real.pi + 1)) = 0 := 
by
  sorry

end log_expression_simplifies_to_zero_l31_3133


namespace arithmetic_sequence_a15_value_l31_3164

variables {a : ℕ → ℤ}

noncomputable def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a15_value
  (h1 : is_arithmetic_sequence a)
  (h2 : a 3 + a 13 = 20)
  (h3 : a 2 = -2) : a 15 = 24 :=
by sorry

end arithmetic_sequence_a15_value_l31_3164


namespace num_divisors_not_divisible_by_2_of_360_l31_3165

def is_divisor (n d : ℕ) : Prop := d ∣ n

def is_prime (p : ℕ) : Prop := Nat.Prime p

noncomputable def prime_factors (n : ℕ) : List ℕ := sorry -- To be implemented if needed

def count_divisors_not_divisible_by_2 (n : ℕ) : ℕ :=
  let factors : List ℕ := prime_factors 360
  let a := 0
  let b_choices := [0, 1, 2]
  let c_choices := [0, 1]
  (b_choices.length) * (c_choices.length)

theorem num_divisors_not_divisible_by_2_of_360 :
  count_divisors_not_divisible_by_2 360 = 6 :=
by sorry

end num_divisors_not_divisible_by_2_of_360_l31_3165


namespace Freddie_ratio_l31_3103

noncomputable def Veronica_distance : ℕ := 1000

noncomputable def Freddie_distance (F : ℕ) : Prop :=
  1000 + 12000 = 5 * F - 2000

theorem Freddie_ratio (F : ℕ) (h : Freddie_distance F) :
  F / Veronica_distance = 3 := by
  sorry

end Freddie_ratio_l31_3103


namespace speed_of_stream_l31_3127

theorem speed_of_stream (b s : ℝ) (h1 : 75 = 5 * (b + s)) (h2 : 45 = 5 * (b - s)) : s = 3 :=
by
  have eq1 : b + s = 15 := by linarith [h1]
  have eq2 : b - s = 9 := by linarith [h2]
  have b_val : b = 12 := by linarith [eq1, eq2]
  linarith 

end speed_of_stream_l31_3127


namespace find_EQ_l31_3179

open Real

noncomputable def Trapezoid_EFGH (EF FG GH HE EQ QF : ℝ) : Prop :=
  EF = 110 ∧
  FG = 60 ∧
  GH = 23 ∧
  HE = 75 ∧
  EQ + QF = EF ∧
  EQ = 250 / 3

theorem find_EQ (EF FG GH HE EQ QF : ℝ) (h : Trapezoid_EFGH EF FG GH HE EQ QF) :
  EQ = 250 / 3 :=
by
  sorry

end find_EQ_l31_3179


namespace compare_sums_l31_3136

open Classical

-- Define the necessary sequences and their properties
variable {α : Type*} [LinearOrderedField α]

-- Arithmetic Sequence {a_n}
noncomputable def arith_seq (a_1 d : α) : ℕ → α
| 0     => a_1
| (n+1) => (arith_seq a_1 d n) + d

-- Geometric Sequence {b_n}
noncomputable def geom_seq (b_1 q : α) : ℕ → α
| 0     => b_1
| (n+1) => (geom_seq b_1 q n) * q

-- Sum of the first n terms of an arithmetic sequence
noncomputable def arith_sum (a_1 d : α) (n : ℕ) : α :=
(n + 1) * (a_1 + arith_seq a_1 d n) / 2

-- Sum of the first n terms of a geometric sequence
noncomputable def geom_sum (b_1 q : α) (n : ℕ) : α :=
if q = 1 then (n + 1) * b_1
else b_1 * (1 - q^(n + 1)) / (1 - q)

theorem compare_sums
  (a_1 b_1 : α) (d q : α)
  (hd : d ≠ 0) (hq : q > 0) (hq1 : q ≠ 1)
  (h_eq1 : a_1 = b_1)
  (h_eq2 : arith_seq a_1 d 1011 = geom_seq b_1 q 1011) :
  arith_sum a_1 d 2022 < geom_sum b_1 q 2022 :=
sorry

end compare_sums_l31_3136


namespace sum_of_faces_of_rectangular_prism_l31_3148

/-- Six positive integers are written on the faces of a rectangular prism.
Each vertex is labeled with the product of the three numbers on the faces adjacent to that vertex.
If the sum of the numbers on the eight vertices is equal to 720, 
prove that the sum of the numbers written on the faces is equal to 27. -/
theorem sum_of_faces_of_rectangular_prism (a b c d e f : ℕ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0)
(h_vertex_sum : a * b * c + a * e * c + a * b * f + a * e * f + d * b * c + d * e * c + d * b * f + d * e * f = 720) :
  (a + d) + (b + e) + (c + f) = 27 :=
by
  sorry

end sum_of_faces_of_rectangular_prism_l31_3148


namespace train_length_l31_3131

theorem train_length (speed_kmph : ℝ) (time_s : ℝ) (length_m : ℝ) :
  speed_kmph = 60 →
  time_s = 3 →
  length_m = 50.01 :=
by
  sorry

end train_length_l31_3131


namespace domain_f_l31_3104

def domain (f : ℝ → ℝ) : Set ℝ := {x | ∃ y, f x = y}

noncomputable def f (x : ℝ) : ℝ := abs (Real.log (x-1) / Real.log 2) + 1

theorem domain_f : domain f = {x | 1 < x} :=
by {
  sorry
}

end domain_f_l31_3104


namespace replace_asterisks_l31_3159

theorem replace_asterisks (x : ℕ) (h : (x / 20) * (x / 180) = 1) : x = 60 := by
  sorry

end replace_asterisks_l31_3159


namespace functional_equation_solution_l31_3186

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), (∀ x y : ℝ, f (x + f y) + f (y + f x) = 2 * f (x * f y)) → (∀ x : ℝ, f x = 0) :=
by
  intro f h
  sorry

end functional_equation_solution_l31_3186


namespace arccos_cos_10_l31_3160

theorem arccos_cos_10 : Real.arccos (Real.cos 10) = 2 := by
  sorry

end arccos_cos_10_l31_3160


namespace max_value_of_expression_l31_3193

theorem max_value_of_expression (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : x₁^2 / 4 + 9 * y₁^2 / 4 = 1) 
  (h₂ : x₂^2 / 4 + 9 * y₂^2 / 4 = 1) 
  (h₃ : x₁ * x₂ + 9 * y₁ * y₂ = -2) :
  (|2 * x₁ + 3 * y₁ - 3| + |2 * x₂ + 3 * y₂ - 3|) ≤ 6 + 2 * Real.sqrt 5 :=
sorry

end max_value_of_expression_l31_3193


namespace max_subjects_per_teacher_l31_3120

theorem max_subjects_per_teacher
  (math_teachers : ℕ := 7)
  (physics_teachers : ℕ := 6)
  (chemistry_teachers : ℕ := 5)
  (min_teachers_required : ℕ := 6)
  (total_subjects : ℕ := 18) :
  ∀ (x : ℕ), x ≥ 3 ↔ 6 * x ≥ total_subjects := by
  sorry

end max_subjects_per_teacher_l31_3120


namespace jackson_holidays_l31_3108

theorem jackson_holidays (holidays_per_month : ℕ) (months_per_year : ℕ) (total_holidays : ℕ) :
  holidays_per_month = 3 → months_per_year = 12 → total_holidays = holidays_per_month * months_per_year →
  total_holidays = 36 :=
by
  intros
  sorry

end jackson_holidays_l31_3108


namespace sum_of_variables_l31_3140

theorem sum_of_variables (a b c d : ℤ)
  (h1 : a - b + 2 * c = 7)
  (h2 : b - c + d = 8)
  (h3 : c - d + a = 5)
  (h4 : d - a + b = 4) : a + b + c + d = 20 :=
by
  sorry

end sum_of_variables_l31_3140


namespace xy_range_l31_3178

theorem xy_range (x y : ℝ) (h1 : y = 3 * (⌊x⌋) + 2) (h2 : y = 4 * (⌊x - 3⌋) + 6) (h3 : (⌊x⌋ : ℝ) ≠ x) :
  34 < x + y ∧ x + y < 35 := 
by 
  sorry

end xy_range_l31_3178


namespace find_monthly_income_l31_3130

-- Define the percentages spent on various categories
def household_items_percentage : ℝ := 0.35
def clothing_percentage : ℝ := 0.18
def medicines_percentage : ℝ := 0.06
def entertainment_percentage : ℝ := 0.11
def transportation_percentage : ℝ := 0.12
def mutual_fund_percentage : ℝ := 0.05
def taxes_percentage : ℝ := 0.07

-- Define the savings amount
def savings_amount : ℝ := 12500

-- Total spent percentage
def total_spent_percentage := household_items_percentage + clothing_percentage + medicines_percentage + entertainment_percentage + transportation_percentage + mutual_fund_percentage + taxes_percentage

-- Percentage saved
def savings_percentage := 1 - total_spent_percentage

-- Prove that Ajay's monthly income is Rs. 208,333.33
theorem find_monthly_income (I : ℝ) (h : I * savings_percentage = savings_amount) : I = 208333.33 := by
  sorry

end find_monthly_income_l31_3130


namespace value_of_a_l31_3144

theorem value_of_a (a : ℝ) (h : (a, 0) ∈ {p : ℝ × ℝ | p.2 = p.1 + 8}) : a = -8 :=
sorry

end value_of_a_l31_3144


namespace prime_solution_l31_3101

theorem prime_solution (p q : ℕ) (hp : Prime p) (hq : Prime q) (h : q^5 - 2 * p^2 = 1) : p + q = 14 :=
sorry

end prime_solution_l31_3101


namespace simplify_polynomial_l31_3167

theorem simplify_polynomial (x : ℝ) :
  (5 - 5 * x - 10 * x^2 + 10 + 15 * x - 20 * x^2 - 10 + 20 * x + 30 * x^2) = 5 + 30 * x :=
  by sorry

end simplify_polynomial_l31_3167


namespace keith_total_cost_correct_l31_3146

noncomputable def total_cost_keith_purchases : Real :=
  let discount_toy := 6.51
  let price_toy := discount_toy / 0.90
  let pet_food := 5.79
  let cage_price := 12.51
  let tax_rate := 0.08
  let cage_tax := cage_price * tax_rate
  let price_with_tax := cage_price + cage_tax
  let water_bottle := 4.99
  let bedding := 7.65
  let discovered_money := 1.0
  let total_cost := discount_toy + pet_food + price_with_tax + water_bottle + bedding
  total_cost - discovered_money

theorem keith_total_cost_correct :
  total_cost_keith_purchases = 37.454 :=
by
  sorry -- Proof of the theorem will go here

end keith_total_cost_correct_l31_3146


namespace prove_a3_l31_3126

variable (a1 a2 a3 a4 : ℕ)
variable (q : ℕ)

-- Definition of the geometric sequence
def geom_seq (n : ℕ) : ℕ :=
  a1 * q^(n-1)

-- Given conditions
def cond1 := geom_seq 4 = 8
def cond2 := (geom_seq 2 + geom_seq 3) / (geom_seq 1 + geom_seq 2) = 2

-- Proving the required condition
theorem prove_a3 : cond1 ∧ cond2 → geom_seq 3 = 4 :=
by
sorry

end prove_a3_l31_3126


namespace katrina_cookies_left_l31_3191

/-- Katrina’s initial number of cookies. -/
def initial_cookies : ℕ := 120

/-- Cookies sold in the morning. 
    1 dozen is 12 cookies, so 3 dozen is 36 cookies. -/
def morning_sales : ℕ := 3 * 12

/-- Cookies sold during the lunch rush. -/
def lunch_sales : ℕ := 57

/-- Cookies sold in the afternoon. -/
def afternoon_sales : ℕ := 16

/-- Calculate the number of cookies left after all sales. -/
def cookies_left : ℕ :=
  initial_cookies - morning_sales - lunch_sales - afternoon_sales

/-- Prove that the number of cookies left for Katrina to take home is 11. -/
theorem katrina_cookies_left : cookies_left = 11 := by
  sorry

end katrina_cookies_left_l31_3191


namespace percentage_students_enrolled_in_bio_l31_3151

-- Problem statement
theorem percentage_students_enrolled_in_bio (total_students : ℕ) (students_not_in_bio : ℕ) 
    (h1 : total_students = 880) (h2 : students_not_in_bio = 462) : 
    ((total_students - students_not_in_bio : ℚ) / total_students) * 100 = 47.5 := by 
  -- Proof is omitted
  sorry

end percentage_students_enrolled_in_bio_l31_3151


namespace train_speed_equivalent_l31_3110

def length_train1 : ℝ := 180
def length_train2 : ℝ := 160
def speed_train1 : ℝ := 60 
def crossing_time_sec : ℝ := 12.239020878329734

noncomputable def speed_train2 (length1 length2 speed1 time : ℝ) : ℝ :=
  let total_length_km := (length1 + length2) / 1000
  let time_hr := time / 3600
  let relative_speed := total_length_km / time_hr
  relative_speed - speed1

theorem train_speed_equivalent :
  speed_train2 length_train1 length_train2 speed_train1 crossing_time_sec = 40 :=
by
  simp [length_train1, length_train2, speed_train1, crossing_time_sec, speed_train2]
  sorry

end train_speed_equivalent_l31_3110


namespace Anil_profit_in_rupees_l31_3121

def cost_scooter (C : ℝ) : Prop := 0.10 * C = 500
def profit (C P : ℝ) : Prop := P = 0.20 * C

theorem Anil_profit_in_rupees (C P : ℝ) (h1 : cost_scooter C) (h2 : profit C P) : P = 1000 :=
by
  sorry

end Anil_profit_in_rupees_l31_3121


namespace total_playtime_l31_3113

-- Conditions
def lena_playtime_hours : ℝ := 3.5
def minutes_per_hour : ℝ := 60
def lena_playtime_minutes : ℝ := lena_playtime_hours * minutes_per_hour
def brother_playtime_extra_minutes : ℝ := 17
def brother_playtime_minutes : ℝ := lena_playtime_minutes + brother_playtime_extra_minutes

-- Proof problem
theorem total_playtime : lena_playtime_minutes + brother_playtime_minutes = 437 := by
  sorry

end total_playtime_l31_3113


namespace minimize_expression_10_l31_3125

theorem minimize_expression_10 (n : ℕ) (h : 0 < n) : 
  (∃ m : ℕ, 0 < m ∧ (∀ k : ℕ, 0 < k → (n = k) → (n = 10))) :=
by
  sorry

end minimize_expression_10_l31_3125


namespace proof_problem_l31_3115

noncomputable def p (a : ℝ) : Prop :=
∀ x : ℝ, x^2 + a * x + a^2 ≥ 0

noncomputable def q : Prop :=
∃ x₀ : ℕ, 0 < x₀ ∧ 2 * x₀^2 - 1 ≤ 0

theorem proof_problem (a : ℝ) (hp : p a) (hq : q) : p a ∨ q :=
by
  sorry

end proof_problem_l31_3115


namespace prove_Φ_eq_8_l31_3170

-- Define the structure of the problem.
def condition (Φ : ℕ) : Prop := 504 / Φ = 40 + 3 * Φ

-- Define the main proof question.
theorem prove_Φ_eq_8 (Φ : ℕ) (h : condition Φ) : Φ = 8 := 
sorry

end prove_Φ_eq_8_l31_3170


namespace min_value_2x_plus_y_l31_3188

theorem min_value_2x_plus_y (x y : ℝ) (h1 : |y| ≤ 2 - x) (h2 : x ≥ -1) : 
  ∃ (x y : ℝ), |y| ≤ 2 - x ∧ x ≥ -1 ∧ (∀ y : ℝ, |y| ≤ 2 - x → x ≥ -1 → 2 * x + y ≥ -5) ∧ (2 * x + y = -5) :=
by
  sorry

end min_value_2x_plus_y_l31_3188


namespace evaluate_expression_l31_3190

variable (g : ℕ → ℕ)
variable (g_inv : ℕ → ℕ)
variable (h1 : g 4 = 7)
variable (h2 : g 6 = 2)
variable (h3 : g 3 = 6)
variable (h4 : ∀ x, g (g_inv x) = x)
variable (h5 : ∀ x, g_inv (g x) = x)

theorem evaluate_expression : g_inv (g_inv 6 + g_inv 7) = 4 :=
by
  -- The proof is omitted
  sorry

end evaluate_expression_l31_3190


namespace total_students_l31_3109

theorem total_students (x : ℕ) (h1 : (x + 6) / (2*x + 6) = 2 / 3) : 2 * x + 6 = 18 :=
sorry

end total_students_l31_3109


namespace best_fit_model_l31_3157

-- Define the coefficients of determination for each model
noncomputable def R2_Model1 : ℝ := 0.75
noncomputable def R2_Model2 : ℝ := 0.90
noncomputable def R2_Model3 : ℝ := 0.45
noncomputable def R2_Model4 : ℝ := 0.65

-- State the theorem 
theorem best_fit_model : 
  R2_Model2 ≥ R2_Model1 ∧ 
  R2_Model2 ≥ R2_Model3 ∧ 
  R2_Model2 ≥ R2_Model4 :=
by
  sorry

end best_fit_model_l31_3157


namespace sum_of_fractions_l31_3185

-- Definitions (Conditions)
def frac1 : ℚ := 5 / 13
def frac2 : ℚ := 9 / 11

-- Theorem (Equivalent Proof Problem)
theorem sum_of_fractions : frac1 + frac2 = 172 / 143 := 
by
  -- Proof skipped
  sorry

end sum_of_fractions_l31_3185


namespace negation_of_p_l31_3162

-- Define the original predicate
def p (x₀ : ℝ) : Prop := x₀^2 > 1

-- Define the negation of the predicate
def not_p : Prop := ∀ x : ℝ, x^2 ≤ 1

-- Prove the negation of the proposition
theorem negation_of_p : (∃ x₀ : ℝ, p x₀) ↔ not_p := by
  sorry

end negation_of_p_l31_3162


namespace trig_identity_l31_3177

open Real

theorem trig_identity (α : ℝ) (h : tan α = -1/2) : 1 - sin (2 * α) = 9/5 := 
  sorry

end trig_identity_l31_3177


namespace sum_first_two_integers_l31_3147

/-- Prove that the sum of the first two integers n > 1 such that 3^n is divisible by n 
and 3^n - 1 is divisible by n - 1 is equal to 30. -/
theorem sum_first_two_integers (n : ℕ) (h1 : n > 1) (h2 : 3 ^ n % n = 0) (h3 : (3 ^ n - 1) % (n - 1) = 0) : 
  n = 3 ∨ n = 27 → n + 3 + 27 = 30 :=
sorry

end sum_first_two_integers_l31_3147


namespace speed_first_hour_l31_3197

theorem speed_first_hour (x : ℝ) :
  (∃ x, (x + 45) / 2 = 65) → x = 85 :=
by
  intro h
  obtain ⟨x, hx⟩ := h
  sorry

end speed_first_hour_l31_3197


namespace cube_volume_given_surface_area_l31_3158

theorem cube_volume_given_surface_area (SA : ℝ) (a V : ℝ) (h : SA = 864) (h1 : 6 * a^2 = SA) (h2 : V = a^3) : 
  V = 1728 := 
by 
  sorry

end cube_volume_given_surface_area_l31_3158


namespace negation_proof_l31_3192

theorem negation_proof :
  (¬ ∀ x : ℝ, x < 0 → 1 - x > Real.exp x) ↔ (∃ x_0 : ℝ, x_0 < 0 ∧ 1 - x_0 ≤ Real.exp x_0) :=
by
  sorry

end negation_proof_l31_3192


namespace quadratic_one_solution_l31_3100

theorem quadratic_one_solution (p : ℝ) : (3 * (1 : ℝ) ^ 2 - 6 * (1 : ℝ) + p = 0) 
  → ((-6) ^ 2 - 4 * 3 * p = 0) 
  → p = 3 :=
by
  intro h1 h2
  have h1' : 3 * (1 : ℝ) ^ 2 - 6 * (1 : ℝ) + p = 0 := h1
  have h2' : (-6) ^ 2 - 4 * 3 * p = 0 := h2
  sorry

end quadratic_one_solution_l31_3100


namespace game_cost_l31_3132

theorem game_cost
    (total_earnings : ℕ)
    (expenses : ℕ)
    (games_bought : ℕ)
    (remaining_money := total_earnings - expenses)
    (cost_per_game := remaining_money / games_bought)
    (h1 : total_earnings = 104)
    (h2 : expenses = 41)
    (h3 : games_bought = 7) :
    cost_per_game = 9 := by
  sorry

end game_cost_l31_3132


namespace find_ABC_l31_3149

theorem find_ABC (A B C : ℝ) (h : ∀ n : ℕ, n > 0 → 2 * n^3 + 3 * n^2 = A * (n * (n - 1) * (n - 2)) / 6 + B * (n * (n - 1)) / 2 + C * n) :
  A = 12 ∧ B = 18 ∧ C = 5 :=
by {
  sorry
}

end find_ABC_l31_3149


namespace min_x_squared_plus_y_squared_l31_3114

theorem min_x_squared_plus_y_squared (x y : ℝ) (h : (x + 5) * (y - 5) = 0) :
  x^2 + y^2 ≥ 50 :=
by
  sorry

end min_x_squared_plus_y_squared_l31_3114


namespace car_dealership_sales_l31_3181

theorem car_dealership_sales (trucks_ratio suvs_ratio trucks_expected suvs_expected : ℕ)
  (h_ratio : trucks_ratio = 5 ∧ suvs_ratio = 8)
  (h_expected : trucks_expected = 35 ∧ suvs_expected = 56) :
  (trucks_ratio : ℚ) / suvs_ratio = (trucks_expected : ℚ) / suvs_expected :=
by
  sorry

end car_dealership_sales_l31_3181


namespace find_a_l31_3176

noncomputable def triangle_side (a b c : ℝ) (A : ℝ) (area : ℝ) : ℝ :=
if b + c = 2 * Real.sqrt 3 ∧ A = Real.pi / 3 ∧ area = Real.sqrt 3 / 2 then
  Real.sqrt (b^2 + c^2 - 2 * b * c * Real.cos A)
else 0

theorem find_a (b c : ℝ) (h1 : b + c = 2 * Real.sqrt 3) (h2 : Real.cos (Real.pi / 3) = 1 / 2) (area : ℝ)
  (h3 : area = Real.sqrt 3 / 2)
  (a := triangle_side (Real.sqrt 6) b c (Real.pi / 3) (Real.sqrt 3 / 2)) :
  a = Real.sqrt 6 :=
sorry

end find_a_l31_3176


namespace stacy_days_to_complete_paper_l31_3173

def total_pages : ℕ := 66
def pages_per_day : ℕ := 11

theorem stacy_days_to_complete_paper :
  total_pages / pages_per_day = 6 := by
  sorry

end stacy_days_to_complete_paper_l31_3173


namespace geometric_sequence_common_ratio_l31_3116

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : ∀ n, 0 < a n)
  (h3 : 2 * (1 / 2 * a 3) = 3 * a 1 + 2 * a 2) : q = 3 := by
  sorry

end geometric_sequence_common_ratio_l31_3116


namespace parabola_c_value_l31_3129

theorem parabola_c_value (b c : ℝ)
  (h1 : 3 = 2^2 + b * 2 + c)
  (h2 : 6 = 5^2 + b * 5 + c) :
  c = -13 :=
by
  -- Proof would follow here
  sorry

end parabola_c_value_l31_3129


namespace max_total_toads_l31_3107

variable (x y : Nat)
variable (frogs total_frogs : Nat)
variable (total_toads : Nat)

def pond1_frogs := 3 * x
def pond1_toads := 4 * x
def pond2_frogs := 5 * y
def pond2_toads := 6 * y

def all_frogs := pond1_frogs x + pond2_frogs y
def all_toads := pond1_toads x + pond2_toads y

theorem max_total_toads (h_frogs : all_frogs x y = 36) : all_toads x y = 46 := 
sorry

end max_total_toads_l31_3107


namespace Cassini_l31_3163

-- Define the Fibonacci sequence
def Fibonacci : ℕ → ℤ
| 0       => 0
| 1       => 1
| (n + 2) => Fibonacci (n + 1) + Fibonacci n

-- State Cassini's Identity theorem
theorem Cassini (n : ℕ) : Fibonacci (n + 1) * Fibonacci (n - 1) - (Fibonacci n) ^ 2 = (-1) ^ n := 
by sorry

end Cassini_l31_3163


namespace total_dinners_sold_203_l31_3150

def monday_dinners : ℕ := 40
def tuesday_dinners : ℕ := monday_dinners + 40
def wednesday_dinners : ℕ := tuesday_dinners / 2
def thursday_dinners : ℕ := wednesday_dinners + 3

def total_dinners_sold : ℕ := monday_dinners + tuesday_dinners + wednesday_dinners + thursday_dinners

theorem total_dinners_sold_203 : total_dinners_sold = 203 := by
  sorry

end total_dinners_sold_203_l31_3150


namespace triangle_side_length_l31_3154

theorem triangle_side_length 
  (side1 : ℕ) (side2 : ℕ) (side3 : ℕ) (P : ℕ)
  (h_side1 : side1 = 5)
  (h_side3 : side3 = 30)
  (h_P : P = 55) :
  side1 + side2 + side3 = P → side2 = 20 :=
by
  intros h
  sorry 

end triangle_side_length_l31_3154


namespace group_photo_arrangements_grouping_methods_selection_methods_with_at_least_one_male_l31_3111

-- Question 1
theorem group_photo_arrangements {M F : ℕ} (hM : M = 3) (hF : F = 5) :
  ∃ arrangements : ℕ, arrangements = 14400 := 
sorry

-- Question 2
theorem grouping_methods {N : ℕ} (hN : N = 8) :
  ∃ methods : ℕ, methods = 2520 := 
sorry

-- Question 3
theorem selection_methods_with_at_least_one_male {M F : ℕ} (hM : M = 3) (hF : F = 5) :
  ∃ methods : ℕ, methods = 1560 := 
sorry

end group_photo_arrangements_grouping_methods_selection_methods_with_at_least_one_male_l31_3111


namespace min_balls_to_guarantee_18_l31_3112

noncomputable def min_balls_needed {red green yellow blue white black : ℕ}
    (h_red : red = 30) 
    (h_green : green = 23) 
    (h_yellow : yellow = 21) 
    (h_blue : blue = 17) 
    (h_white : white = 14) 
    (h_black : black = 12) : ℕ :=
  95

theorem min_balls_to_guarantee_18 {red green yellow blue white black : ℕ}
    (h_red : red = 30) 
    (h_green : green = 23) 
    (h_yellow : yellow = 21) 
    (h_blue : blue = 17) 
    (h_white : white = 14) 
    (h_black : black = 12) :
  min_balls_needed h_red h_green h_yellow h_blue h_white h_black = 95 :=
  by
  -- Placeholder for the actual proof
  sorry

end min_balls_to_guarantee_18_l31_3112


namespace smallest_solution_is_39_over_8_l31_3139

noncomputable def smallest_solution (x : ℝ) : Prop :=
  (3 * x / (x - 3) + (3 * x^2 - 27) / x = 14) ∧ (x ≠ 0) ∧ (x ≠ 3)

theorem smallest_solution_is_39_over_8 : ∃ x > 0, smallest_solution x ∧ x = 39 / 8 :=
by
  sorry

end smallest_solution_is_39_over_8_l31_3139


namespace discount_percentage_for_two_pairs_of_jeans_l31_3142

theorem discount_percentage_for_two_pairs_of_jeans
  (price_per_pair : ℕ := 40)
  (price_for_three_pairs : ℕ := 112)
  (discount : ℕ := 8)
  (original_price_for_two_pairs : ℕ := price_per_pair * 2)
  (discount_percentage : ℕ := (discount * 100) / original_price_for_two_pairs) :
  discount_percentage = 10 := 
by
  sorry

end discount_percentage_for_two_pairs_of_jeans_l31_3142


namespace amoeba_population_after_5_days_l31_3128

theorem amoeba_population_after_5_days 
  (initial : ℕ)
  (split_factor : ℕ)
  (days : ℕ)
  (h_initial : initial = 2)
  (h_split : split_factor = 3)
  (h_days : days = 5) :
  (initial * split_factor ^ days) = 486 :=
by sorry

end amoeba_population_after_5_days_l31_3128


namespace solve_for_k_l31_3194

variable {x y k : ℝ}

theorem solve_for_k (h1 : 2 * x + y = 1) (h2 : x + 2 * y = k - 2) (h3 : x - y = 2) : k = 1 :=
by
  sorry

end solve_for_k_l31_3194


namespace product_consecutive_natural_not_equal_even_l31_3141

theorem product_consecutive_natural_not_equal_even (n m : ℕ) (h : m % 2 = 0 ∧ m > 0) : n * (n + 1) ≠ m * (m + 2) :=
sorry

end product_consecutive_natural_not_equal_even_l31_3141


namespace first_course_cost_l31_3153

theorem first_course_cost (x : ℝ) (h1 : 60 - (x + (x + 5) + 0.25 * (x + 5)) = 20) : x = 15 :=
by sorry

end first_course_cost_l31_3153


namespace sum_op_two_triangles_l31_3123

def op (a b c : ℕ) : ℕ := 2 * a - b + c

theorem sum_op_two_triangles : op 3 7 5 + op 6 2 8 = 22 := by
  sorry

end sum_op_two_triangles_l31_3123


namespace correct_option_D_l31_3134

theorem correct_option_D (x y : ℝ) : (x - y) ^ 2 = (y - x) ^ 2 := by
  sorry

end correct_option_D_l31_3134


namespace find_original_price_l31_3145

-- Definitions based on the conditions
def original_price_increased (x : ℝ) : ℝ := 1.25 * x
def loan_payment (total_cost : ℝ) : ℝ := 0.75 * total_cost
def own_funds (total_cost : ℝ) : ℝ := 0.25 * total_cost

-- Condition values
def new_home_cost : ℝ := 500000
def loan_amount := loan_payment new_home_cost
def funds_paid := own_funds new_home_cost

-- Proof statement
theorem find_original_price : 
  ∃ x : ℝ, original_price_increased x = funds_paid ↔ x = 100000 :=
by
  -- Placeholder for actual proof
  sorry

end find_original_price_l31_3145


namespace probability_at_most_one_red_light_l31_3195

def probability_of_no_red_light (p : ℚ) (n : ℕ) : ℚ := (1 - p) ^ n

def probability_of_exactly_one_red_light (p : ℚ) (n : ℕ) : ℚ :=
  (n.choose 1) * p ^ 1 * (1 - p) ^ (n - 1)

theorem probability_at_most_one_red_light (p : ℚ) (n : ℕ) (h : p = 1/3 ∧ n = 4) :
  probability_of_no_red_light p n + probability_of_exactly_one_red_light p n = 16 / 27 :=
by
  rw [h.1, h.2]
  sorry

end probability_at_most_one_red_light_l31_3195


namespace product_of_solutions_eq_zero_l31_3175

theorem product_of_solutions_eq_zero :
  (∃ x : ℝ, (x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) →
  (∀ x : ℝ, ((x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) → (x = 0 ∨ x = -4/7)) →
  (0 * (-4/7) = 0) :=
by
  sorry

end product_of_solutions_eq_zero_l31_3175


namespace find_b_l31_3187

theorem find_b (b : ℕ) (h1 : 40 < b) (h2 : b < 120) 
    (h3 : b % 4 = 3) (h4 : b % 5 = 3) (h5 : b % 6 = 3) : 
    b = 63 := by
  sorry

end find_b_l31_3187


namespace evaluate_expression_l31_3152

theorem evaluate_expression :
  -(12 * 2) - (3 * 2) + ((-18 / 3) * -4) = -6 := 
by
  sorry

end evaluate_expression_l31_3152


namespace min_value_of_expression_min_value_achieved_l31_3156

noncomputable def f (x : ℝ) : ℝ :=
  (15 - x) * (8 - x) * (15 + x) * (8 + x)

theorem min_value_of_expression : ∀ x : ℝ, (15 - x) * (8 - x) * (15 + x) * (8 + x) ≥ -6452.25 :=
by sorry

theorem min_value_achieved : ∃ x : ℝ, (15 - x) * (8 - x) * (15 + x) * (8 + x) = -6452.25 :=
by sorry

end min_value_of_expression_min_value_achieved_l31_3156


namespace theta_in_third_quadrant_l31_3169

-- Define the mathematical conditions
variable (θ : ℝ)
axiom cos_theta_neg : Real.cos θ < 0
axiom cos_minus_sin_eq_sqrt : Real.cos θ - Real.sin θ = Real.sqrt (1 - 2 * Real.sin θ * Real.cos θ)

-- Prove that θ is in the third quadrant
theorem theta_in_third_quadrant : 
  (∀ θ : ℝ, Real.cos θ < 0 → Real.cos θ - Real.sin θ = Real.sqrt (1 - 2 * Real.sin θ * Real.cos θ) → 
    Real.sin θ < 0 ∧ Real.cos θ < 0) :=
by sorry

end theta_in_third_quadrant_l31_3169


namespace candles_used_l31_3171

theorem candles_used (starting_candles used_candles remaining_candles : ℕ) (h1 : starting_candles = 44) (h2 : remaining_candles = 12) : used_candles = 32 :=
by
  sorry

end candles_used_l31_3171


namespace find_value_in_box_l31_3122

theorem find_value_in_box (x : ℕ) :
  10 * 20 * 30 * 40 * 50 = 100 * 2 * 300 * 4 * x ↔ x = 50 := by
  sorry

end find_value_in_box_l31_3122


namespace positive_number_property_l31_3137

-- Define the problem conditions and the goal
theorem positive_number_property (y : ℝ) (hy : y > 0) (h : y^2 / 100 = 9) : y = 30 := by
  sorry

end positive_number_property_l31_3137


namespace sampled_students_within_interval_l31_3135

/-- Define the conditions for the student's problem --/
def student_count : ℕ := 1221
def sampled_students : ℕ := 37
def sampling_interval : ℕ := student_count / sampled_students
def interval_lower_bound : ℕ := 496
def interval_upper_bound : ℕ := 825
def interval_range : ℕ := interval_upper_bound - interval_lower_bound + 1

/-- State the goal within the above conditions --/
theorem sampled_students_within_interval :
  interval_range / sampling_interval = 10 :=
sorry

end sampled_students_within_interval_l31_3135


namespace work_ratio_l31_3105

theorem work_ratio (M B : ℝ) 
  (h1 : 5 * (12 * M + 16 * B) = 1)
  (h2 : 4 * (13 * M + 24 * B) = 1) : 
  M / B = 2 := 
  sorry

end work_ratio_l31_3105


namespace proof_problem_l31_3198

noncomputable def a : ℝ := 0.85 * 250
noncomputable def b : ℝ := 0.75 * 180
noncomputable def c : ℝ := 0.90 * 320

theorem proof_problem :
  (a - b = 77.5) ∧ (77.5 < c) :=
by
  sorry

end proof_problem_l31_3198


namespace sophie_buys_six_doughnuts_l31_3166

variable (num_doughnuts : ℕ)

theorem sophie_buys_six_doughnuts 
  (h1 : 5 * 2 = 10)
  (h2 : 4 * 2 = 8)
  (h3 : 15 * 0.60 = 9)
  (h4 : 10 + 8 + 9 = 27)
  (h5 : 33 - 27 = 6)
  (h6 : num_doughnuts * 1 = 6) :
  num_doughnuts = 6 := 
  by
    sorry

end sophie_buys_six_doughnuts_l31_3166


namespace sara_picked_6_pears_l31_3182

def total_pears : ℕ := 11
def tim_pears : ℕ := 5
def sara_pears : ℕ := total_pears - tim_pears

theorem sara_picked_6_pears : sara_pears = 6 := by
  sorry

end sara_picked_6_pears_l31_3182


namespace largest_lcm_l31_3174

def lcm_list : List ℕ := [
  Nat.lcm 15 3,
  Nat.lcm 15 5,
  Nat.lcm 15 9,
  Nat.lcm 15 10,
  Nat.lcm 15 12,
  Nat.lcm 15 15
]

theorem largest_lcm : List.maximum lcm_list = 60 := by
  sorry

end largest_lcm_l31_3174


namespace avg_of_first_three_groups_prob_of_inspection_l31_3124
  
-- Define the given frequency distribution as constants
def freq_40_50 : ℝ := 0.04
def freq_50_60 : ℝ := 0.06
def freq_60_70 : ℝ := 0.22
def freq_70_80 : ℝ := 0.28
def freq_80_90 : ℝ := 0.22
def freq_90_100 : ℝ := 0.18

-- Calculate the midpoint values for the first three groups
def mid_40_50 : ℝ := 45
def mid_50_60 : ℝ := 55
def mid_60_70 : ℝ := 65

-- Define the probabilities interpreted from the distributions
def prob_poor : ℝ := freq_40_50 + freq_50_60
def prob_avg : ℝ := freq_60_70 + freq_70_80
def prob_good : ℝ := freq_80_90 + freq_90_100

-- Define the main theorem for the average score of the first three groups
theorem avg_of_first_three_groups :
  (mid_40_50 * freq_40_50 + mid_50_60 * freq_50_60 + mid_60_70 * freq_60_70) /
  (freq_40_50 + freq_50_60 + freq_60_70) = 60.625 := 
by { sorry }

-- Define the theorem for the probability of inspection
theorem prob_of_inspection :
  1 - (3 * (prob_good * prob_avg * prob_avg) + 3 * (prob_avg * prob_avg * prob_good) + (prob_good * prob_good * prob_good)) = 0.396 :=
by { sorry }

end avg_of_first_three_groups_prob_of_inspection_l31_3124


namespace boys_down_slide_l31_3143

theorem boys_down_slide (boys_1 boys_2 : ℕ) (h : boys_1 = 22) (h' : boys_2 = 13) : boys_1 + boys_2 = 35 := by
  sorry

end boys_down_slide_l31_3143


namespace angle_at_630_is_15_degrees_l31_3180

-- Definitions for positions of hour and minute hands at 6:30 p.m.
def angle_per_hour : ℝ := 30
def minute_hand_position_630 : ℝ := 180
def hour_hand_position_630 : ℝ := 195

-- The angle between the hour hand and minute hand at 6:30 p.m.
def angle_between_hands_630 : ℝ := |hour_hand_position_630 - minute_hand_position_630|

-- Statement to prove
theorem angle_at_630_is_15_degrees :
  angle_between_hands_630 = 15 := by
  sorry

end angle_at_630_is_15_degrees_l31_3180


namespace remaining_standby_time_l31_3196

variable (fully_charged_standby : ℝ) (fully_charged_gaming : ℝ)
variable (standby_time : ℝ) (gaming_time : ℝ)

theorem remaining_standby_time
  (h1 : fully_charged_standby = 10)
  (h2 : fully_charged_gaming = 2)
  (h3 : standby_time = 4)
  (h4 : gaming_time = 1.5) :
  (10 - ((standby_time * (1 / fully_charged_standby)) + (gaming_time * (1 / fully_charged_gaming)))) * 10 = 1 :=
by
  sorry

end remaining_standby_time_l31_3196


namespace vertex_of_quadratic1_vertex_of_quadratic2_l31_3168

theorem vertex_of_quadratic1 :
  ∃ x y : ℝ, 
  (∀ x', 2 * x'^2 - 4 * x' - 1 = 2 * (x' - x)^2 + y) ∧ 
  (x = 1 ∧ y = -3) :=
by sorry

theorem vertex_of_quadratic2 :
  ∃ x y : ℝ, 
  (∀ x', -3 * x'^2 + 6 * x' - 2 = -3 * (x' - x)^2 + y) ∧ 
  (x = 1 ∧ y = 1) :=
by sorry

end vertex_of_quadratic1_vertex_of_quadratic2_l31_3168


namespace sum_of_abc_is_40_l31_3118

theorem sum_of_abc_is_40 (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h1 : a * b + c = 55) (h2 : b * c + a = 55) (h3 : c * a + b = 55) :
    a + b + c = 40 :=
by
  sorry

end sum_of_abc_is_40_l31_3118


namespace hyperbola_eccentricity_l31_3106

-- Definition of the parabola C1: y^2 = 2px with p > 0.
def parabola (p : ℝ) (p_pos : 0 < p) (x y : ℝ) : Prop := y^2 = 2 * p * x

-- Definition of the hyperbola C2: x^2 / a^2 - y^2 / b^2 = 1 with a > 0 and b > 0.
def hyperbola (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (x y : ℝ) : Prop := 
  x^2 / a^2 - y^2 / b^2 = 1

-- Definition of having a common focus F at (p / 2, 0).
def common_focus (p a b c : ℝ) (p_pos : 0 < p) (a_pos : 0 < a) (b_pos : 0 < b) : Prop := 
  c = p / 2 ∧ c^2 = a^2 + b^2

-- Definition for points A and B on parabola C1 and point M on hyperbola C2.
def points_A_B_M (c a b : ℝ) (x1 y1 x2 y2 yM : ℝ) : Prop := 
  x1 = c ∧ y1 = 2 * c ∧ x2 = c ∧ y2 = -2 * c ∧ yM = b^2 / a

-- Condition for OM, OA, and OB relation and mn = 1/8.
def OM_OA_OB_relation (m n : ℝ) : Prop := 
  m * n = 1 / 8

-- Theorem statement: Given the conditions, the eccentricity of hyperbola C2 is √6 + √2 / 2.
theorem hyperbola_eccentricity (p a b c m n : ℝ) (p_pos : 0 < p) (a_pos : 0 < a) (b_pos : 0 < b) :
  parabola p p_pos c (2 * c) → 
  hyperbola a b a_pos b_pos c (b^2 / a) → 
  common_focus p a b c p_pos a_pos b_pos →
  points_A_B_M c a b c (2 * c) c (-2 * c) (b^2 / a) →
  OM_OA_OB_relation m n → 
  m * n = 1 / 8 →
  ∃ e : ℝ, e = (Real.sqrt 6 + Real.sqrt 2) / 2 :=
sorry

end hyperbola_eccentricity_l31_3106


namespace prob1_prob2_odd_prob2_monotonic_prob3_l31_3172

variable (a : ℝ) (f : ℝ → ℝ)
variable (hf : ∀ x : ℝ, f (log a x) = a / (a^2 - 1) * (x - 1 / x))
variable (ha : 0 < a ∧ a < 1)

-- Problem 1: Prove the expression for f(x)
theorem prob1 (x : ℝ) : f x = a / (a^2 - 1) * (a^x - a^(-x)) := sorry

-- Problem 2: Prove oddness and monotonicity of f(x)
theorem prob2_odd : ∀ x, f (-x) = -f x := sorry
theorem prob2_monotonic : ∀ x₁ x₂ : ℝ, (x₁ < x₂) → (f x₁ < f x₂) := sorry

-- Problem 3: Determine the range of k
theorem prob3 (k : ℝ) : (∀ t : ℝ, 1 ≤ t ∧ t ≤ 3 → f (3 * t^2 - 1) + f (4 * t - k) > 0) → (k < 6) := sorry

end prob1_prob2_odd_prob2_monotonic_prob3_l31_3172


namespace flower_count_l31_3199

theorem flower_count (roses carnations : ℕ) (h₁ : roses = 5) (h₂ : carnations = 5) : roses + carnations = 10 :=
by
  sorry

end flower_count_l31_3199


namespace range_of_m_l31_3184

noncomputable def quadratic_function : Type := ℝ → ℝ

variable (f : quadratic_function)

axiom quadratic : ∃ a b : ℝ, ∀ x : ℝ, f x = a * (x-2)^2 + b
axiom symmetry : ∀ x : ℝ, f (x + 2) = f (-x + 2)
axiom condition1 : f 0 = 3
axiom condition2 : f 2 = 1
axiom max_value : ∀ m : ℝ, ∀ (x : ℝ) (hx : 0 ≤ x ∧ x ≤ m), f x ≤ 3
axiom min_value : ∀ m : ℝ, ∀ (x : ℝ) (hx : 0 ≤ x ∧ x ≤ m), 1 ≤ f x

theorem range_of_m : ∀ m : ℝ, (∀ (x : ℝ) (hx : 0 ≤ x ∧ x ≤ m), 1 ≤ f x ∧ f x ≤ 3) → 2 ≤ m ∧ m ≤ 4 :=
by
  intro m
  intro h
  sorry

end range_of_m_l31_3184


namespace set_intersection_l31_3138

   -- Define set A
   def A : Set ℝ := {x : ℝ | (x - 3) / (x + 1) ≥ 0 }
   
   -- Define set B
   def B : Set ℝ := {x : ℝ | Real.log x / Real.log 2 < 2}

   -- Define the relative complement of A in the real numbers
   def complement_R (A : Set ℝ) : Set ℝ := {x : ℝ | ¬ (A x)}

   -- The main statement that needs to be proven
   theorem set_intersection :
     (complement_R A) ∩ B = {x : ℝ | 0 < x ∧ x < 3} := by
     sorry
   
end set_intersection_l31_3138


namespace find_a_from_conditions_l31_3183

theorem find_a_from_conditions
  (a b : ℤ)
  (h₁ : 2584 * a + 1597 * b = 0)
  (h₂ : 1597 * a + 987 * b = -1) :
  a = 1597 :=
by sorry

end find_a_from_conditions_l31_3183
