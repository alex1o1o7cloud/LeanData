import Mathlib

namespace trigonometric_identity_l221_221865

theorem trigonometric_identity : (1 / 4) * Real.sin (15 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) = 1 / 16 := by
  sorry

end trigonometric_identity_l221_221865


namespace no_natural_n_divisible_by_2019_l221_221007

theorem no_natural_n_divisible_by_2019 :
  ∀ n : ℕ, ¬ 2019 ∣ (n^2 + n + 2) :=
by sorry

end no_natural_n_divisible_by_2019_l221_221007


namespace necessary_but_not_sufficient_condition_l221_221442

noncomputable def p (x : ℝ) : Prop := (1 - x^2 < 0 ∧ |x| - 2 > 0) ∨ (1 - x^2 > 0 ∧ |x| - 2 < 0)
noncomputable def q (x : ℝ) : Prop := x^2 + x - 6 > 0

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (q x → p x) ∧ ¬(p x → q x) :=
sorry

end necessary_but_not_sufficient_condition_l221_221442


namespace probability_of_B_l221_221632

theorem probability_of_B (P : Set ℕ → ℝ) (A B : Set ℕ) (hA : P A = 0.25) (hAB : P (A ∩ B) = 0.15) (hA_complement_B_complement : P (Aᶜ ∩ Bᶜ) = 0.5) : P B = 0.4 :=
by
  sorry

end probability_of_B_l221_221632


namespace emails_difference_l221_221420

def morning_emails : ℕ := 6
def afternoon_emails : ℕ := 8

theorem emails_difference :
  afternoon_emails - morning_emails = 2 := 
by
  sorry

end emails_difference_l221_221420


namespace greatest_integer_l221_221129

theorem greatest_integer (y : ℤ) : (8 / 11 : ℝ) > (y / 17 : ℝ) → y ≤ 12 :=
by sorry

end greatest_integer_l221_221129


namespace travel_distance_l221_221453

noncomputable def distance_traveled (AB BC : ℝ) : ℝ :=
  let BD := Real.sqrt (AB^2 + BC^2)
  let arc1 := (2 * Real.pi * BD) / 4
  let arc2 := (2 * Real.pi * AB) / 4
  arc1 + arc2

theorem travel_distance (hAB : AB = 3) (hBC : BC = 4) : 
  distance_traveled AB BC = 4 * Real.pi := by
    sorry

end travel_distance_l221_221453


namespace line_passes_through_first_and_fourth_quadrants_l221_221882

theorem line_passes_through_first_and_fourth_quadrants (b k : ℝ) (H : b * k < 0) :
  (∃x₁, k * x₁ + b > 0) ∧ (∃x₂, k * x₂ + b < 0) :=
by
  sorry

end line_passes_through_first_and_fourth_quadrants_l221_221882


namespace ratio_problem_l221_221732

theorem ratio_problem 
  (A B C : ℚ) 
  (h : A / B = 3 / 2 ∧ B / C = 2 / 5 ∧ A / C = 3 / 5) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by 
  sorry

end ratio_problem_l221_221732


namespace smallest_x_plus_y_l221_221381

theorem smallest_x_plus_y 
  (x y : ℕ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y) 
  (hxy_neq : x ≠ y)
  (h_eq : (1/x + 1/y = 1/10)) : x + y = 45 :=
sorry

end smallest_x_plus_y_l221_221381


namespace converse_of_statement_l221_221622

theorem converse_of_statement (x y : ℝ) :
  (¬ (x = 0 ∧ y = 0)) → (x^2 + y^2 ≠ 0) :=
by {
  sorry
}

end converse_of_statement_l221_221622


namespace cookies_left_l221_221189

theorem cookies_left (total_cookies : ℕ) (total_neighbors : ℕ) (cookies_per_neighbor : ℕ) (sarah_cookies : ℕ)
  (h1 : total_cookies = 150)
  (h2 : total_neighbors = 15)
  (h3 : cookies_per_neighbor = 10)
  (h4 : sarah_cookies = 12) :
  total_cookies - ((total_neighbors - 1) * cookies_per_neighbor + sarah_cookies) = 8 :=
by
  simp [h1, h2, h3, h4]
  sorry

end cookies_left_l221_221189


namespace concentric_circles_ratio_l221_221798

theorem concentric_circles_ratio (d1 d2 d3 : ℝ) (h1 : d1 = 2) (h2 : d2 = 4) (h3 : d3 = 6) :
  let r1 := d1 / 2
  let r2 := d2 / 2
  let r3 := d3 / 2
  let A_red := π * r1 ^ 2
  let A_middle := π * r2 ^ 2
  let A_large := π * r3 ^ 2
  let A_blue := A_middle - A_red
  let A_green := A_large - A_middle
  (A_green / A_blue) = 5 / 3 := 
by
  sorry

end concentric_circles_ratio_l221_221798


namespace shaded_area_of_rotated_semicircle_l221_221538

noncomputable def area_of_shaded_region (R : ℝ) (α : ℝ) : ℝ :=
  (1 / 2) * (2 * R) ^ 2 * (α / (2 * Real.pi))

theorem shaded_area_of_rotated_semicircle (R : ℝ) (α : ℝ) (h : α = Real.pi / 9) :
  area_of_shaded_region R α = 2 * Real.pi * R ^ 2 / 9 :=
by
  sorry

end shaded_area_of_rotated_semicircle_l221_221538


namespace tan_neq_sqrt3_sufficient_but_not_necessary_l221_221621

-- Definition of the condition: tan(α) ≠ √3
def condition_tan_neq_sqrt3 (α : ℝ) : Prop := Real.tan α ≠ Real.sqrt 3

-- Definition of the statement: α ≠ π/3
def statement_alpha_neq_pi_div_3 (α : ℝ) : Prop := α ≠ Real.pi / 3

-- The theorem to be proven
theorem tan_neq_sqrt3_sufficient_but_not_necessary {α : ℝ} :
  condition_tan_neq_sqrt3 α → statement_alpha_neq_pi_div_3 α :=
sorry

end tan_neq_sqrt3_sufficient_but_not_necessary_l221_221621


namespace original_number_is_28_l221_221808

theorem original_number_is_28 (N : ℤ) :
  (∃ k : ℤ, N - 11 = 17 * k) → N = 28 :=
by
  intro h
  obtain ⟨k, h₁⟩ := h
  have h₂: N = 17 * k + 11 := by linarith
  have h₃: k = 1 := sorry
  linarith [h₃]
 
end original_number_is_28_l221_221808


namespace cost_of_cookbook_l221_221058

def cost_of_dictionary : ℕ := 11
def cost_of_dinosaur_book : ℕ := 19
def amount_saved : ℕ := 8
def amount_needed : ℕ := 29

theorem cost_of_cookbook :
  let total_cost := amount_saved + amount_needed
  let accounted_cost := cost_of_dictionary + cost_of_dinosaur_book
  total_cost - accounted_cost = 7 :=
by
  sorry

end cost_of_cookbook_l221_221058


namespace tony_income_l221_221294

-- Definitions for the given conditions
def investment : ℝ := 3200
def purchase_price : ℝ := 85
def dividend : ℝ := 6.640625

-- Theorem stating Tony's income based on the conditions
theorem tony_income : (investment / purchase_price) * dividend = 250 :=
by
  sorry

end tony_income_l221_221294


namespace t_shirts_to_buy_l221_221924

variable (P T : ℕ)

def condition1 : Prop := 3 * P + 6 * T = 750
def condition2 : Prop := P + 12 * T = 750

theorem t_shirts_to_buy (h1 : condition1 P T) (h2 : condition2 P T) :
  400 / T = 8 :=
by
  sorry

end t_shirts_to_buy_l221_221924


namespace parabola_equation_l221_221041

open Classical

noncomputable def circle_center : ℝ × ℝ := (2, 0)

theorem parabola_equation (vertex : ℝ × ℝ) (focus : ℝ × ℝ) :
  vertex = (0, 0) ∧ focus = circle_center → ∀ x y : ℝ, y^2 = 8 * x := by
  intro h
  sorry

end parabola_equation_l221_221041


namespace complement_A_in_U_l221_221066

open Set

variable {𝕜 : Type*} [LinearOrderedField 𝕜]

def A (x : 𝕜) : Prop := |x - (1 : 𝕜)| > 2
def U : Set 𝕜 := univ

theorem complement_A_in_U : (U \ {x : 𝕜 | A x}) = {x : 𝕜 | -1 ≤ x ∧ x ≤ 3} := by
  sorry

end complement_A_in_U_l221_221066


namespace maximum_sum_l221_221818

theorem maximum_sum (a b c d : ℕ) (h₀ : a < b ∧ b < c ∧ c < d)
  (h₁ : (c + d) + (a + b + c) = 2017) : a + b + c + d ≤ 806 :=
sorry

end maximum_sum_l221_221818


namespace intersection_correct_l221_221207

def set_A : Set ℤ := {-1, 1, 2, 4}
def set_B : Set ℤ := {x | |x - 1| ≤ 1}

theorem intersection_correct :
  set_A ∩ set_B = {1, 2} :=
  sorry

end intersection_correct_l221_221207


namespace first_nonzero_digit_one_over_137_l221_221652

noncomputable def first_nonzero_digit_right_of_decimal (n : ℚ) : ℕ := sorry

theorem first_nonzero_digit_one_over_137 : first_nonzero_digit_right_of_decimal (1 / 137) = 7 := sorry

end first_nonzero_digit_one_over_137_l221_221652


namespace robin_total_cost_l221_221079

def num_letters_in_name (name : String) : Nat := name.length

def calculate_total_cost (names : List String) (cost_per_bracelet : Nat) : Nat :=
  let total_bracelets := names.foldl (fun acc name => acc + num_letters_in_name name) 0
  total_bracelets * cost_per_bracelet

theorem robin_total_cost : 
  calculate_total_cost ["Jessica", "Tori", "Lily", "Patrice"] 2 = 44 :=
by
  sorry

end robin_total_cost_l221_221079


namespace composite_sum_l221_221606

open Nat

theorem composite_sum (a b c d : ℕ) (h1 : c > b) (h2 : a + b + c + d = a * b - c * d) : ∃ x y : ℕ, x > 1 ∧ y > 1 ∧ a + c = x * y :=
by
  sorry

end composite_sum_l221_221606


namespace sum_of_three_iterated_digits_of_A_is_7_l221_221172

def sum_of_digits (n : ℕ) : ℕ :=
  (Nat.digits 10 n).sum

def A : ℕ := 4444 ^ 4444

theorem sum_of_three_iterated_digits_of_A_is_7 :
  sum_of_digits (sum_of_digits (sum_of_digits A)) = 7 :=
by
  -- We'll skip the actual proof here
  sorry

end sum_of_three_iterated_digits_of_A_is_7_l221_221172


namespace day_200th_of_year_N_minus_1_is_Wednesday_l221_221755

-- Define the basic conditions given in the problem
def day_of_year_N (d : ℕ) : nat := (d % 7)
def day_of_week (day : nat) : Prop :=
  day_of_year_N day = 1   -- 1 represents Wednesday

-- Assume the given conditions
axiom condition_400th_day_of_N_is_Wednesday : day_of_week 400
axiom condition_300th_day_of_N_plus_2_is_Wednesday : day_of_week (300 + 2 * 365 + 1) -- considering 1 leap year

-- Define the year calculations as derived and reasoned in the problem
def day_200th_of_N_minus_1 (d : ℕ) : nat :=
  (d - 365) % 7

-- The statement to prove
theorem day_200th_of_year_N_minus_1_is_Wednesday :
  day_of_week (day_200th_of_N_minus_1 1) :=
sorry

end day_200th_of_year_N_minus_1_is_Wednesday_l221_221755


namespace infinitely_many_n_l221_221774

-- Definition capturing the condition: equation \( (x + y + z)^3 = n^2 xyz \)
def equation (x y z n : ℕ) : Prop := (x + y + z)^3 = n^2 * x * y * z

-- The main statement: proving the existence of infinitely many positive integers n such that the equation has a solution
theorem infinitely_many_n :
  ∃ᶠ n : ℕ in at_top, ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ equation x y z n :=
sorry

end infinitely_many_n_l221_221774


namespace minimize_sum_of_reciprocals_l221_221702

def dataset : List ℝ := [2, 4, 6, 8]

def mean : ℝ := 5
def variance: ℝ := 5

theorem minimize_sum_of_reciprocals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : mean * a + variance * b = 1) : 
  (1 / a + 1 / b) = 20 :=
sorry

end minimize_sum_of_reciprocals_l221_221702


namespace inequality_solution_l221_221778

theorem inequality_solution (a x : ℝ) (h₁ : 0 < a) : 
  (0 < a ∧ a < 1 → 2 < x ∧ x < (a-2)/(a-1) → (a * (x - 1)) / (x-2) > 1) ∧ 
  (a = 1 → 2 < x → (a * (x - 1)) / (x-2) > 1 ∧ true) ∧ 
  (a > 1 → (2 < x ∨ x < (a-2)/(a-1)) → (a * (x - 1)) / (x-2) > 1) := 
sorry

end inequality_solution_l221_221778


namespace arithmetic_sequence_term_count_l221_221580

theorem arithmetic_sequence_term_count (a d n an : ℕ) (h₀ : a = 5) (h₁ : d = 7) (h₂ : an = 126) (h₃ : an = a + (n - 1) * d) : n = 18 := by
  sorry

end arithmetic_sequence_term_count_l221_221580


namespace cookies_in_box_l221_221559

/-- Graeme is weighing cookies to see how many he can fit in his box. His box can only hold
    40 pounds of cookies. If each cookie weighs 2 ounces, how many cookies can he fit in the box? -/
theorem cookies_in_box (box_capacity_pounds : ℕ) (cookie_weight_ounces : ℕ) (pound_to_ounces : ℕ)
  (h_box_capacity : box_capacity_pounds = 40)
  (h_cookie_weight : cookie_weight_ounces = 2)
  (h_pound_to_ounces : pound_to_ounces = 16) :
  (box_capacity_pounds * pound_to_ounces) / cookie_weight_ounces = 320 := by 
  sorry

end cookies_in_box_l221_221559


namespace quadratic_z_and_u_l221_221042

variables (a b c α β γ : ℝ)
variable (d : ℝ)
variable (δ : ℝ)
variables (x₁ x₂ y₁ y₂ z₁ z₂ u₁ u₂ : ℝ)

-- Given conditions
variable (h_nonzero : a * α ≠ 0)
variable (h_discriminant1 : b^2 - 4 * a * c ≥ 0)
variable (h_discriminant2 : β^2 - 4 * α * γ ≥ 0)
variable (hx_roots_order : x₁ ≤ x₂)
variable (hy_roots_order : y₁ ≤ y₂)
variable (h_eq_discriminant1 : b^2 - 4 * a * c = d^2)
variable (h_eq_discriminant2 : β^2 - 4 * α * γ = δ^2)

-- Translate into mathematical constraints for the roots
variable (hx1 : x₁ = (-b - d) / (2 * a))
variable (hx2 : x₂ = (-b + d) / (2 * a))
variable (hy1 : y₁ = (-β - δ) / (2 * α))
variable (hy2 : y₂ = (-β + δ) / (2 * α))

-- Variables for polynomial equations roots
axiom h_z1 : z₁ = x₁ + y₁
axiom h_z2 : z₂ = x₂ + y₂
axiom h_u1 : u₁ = x₁ + y₂
axiom h_u2 : u₂ = x₂ + y₁

theorem quadratic_z_and_u :
  (2 * a * α) * z₂ * z₂ + 2 * (a * β + α * b) * z₁ + (2 * a * γ + 2 * α * c + b * β - d * δ) = 0 ∧
  (2 * a * α) * u₂ * u₂ + 2 * (a * β + α * b) * u₁ + (2 * a * γ + 2 * α * c + b * β + d * δ) = 0 := sorry

end quadratic_z_and_u_l221_221042


namespace sum_of_ages_of_alex_and_allison_is_47_l221_221302

theorem sum_of_ages_of_alex_and_allison_is_47 (diane_age_now : ℕ)
  (diane_age_at_30_alex_relation : diane_age_now + 14 = 30 ∧ diane_age_now + 14 = 60 / 2)
  (diane_age_at_30_allison_relation : diane_age_now + 14 = 30 ∧ 30 = 2 * (diane_age_now + 14 - (30 - 15)))
  : (60 - (30 - 16)) + (15 - (30 - 16)) = 47 :=
by
  sorry

end sum_of_ages_of_alex_and_allison_is_47_l221_221302


namespace fraction_of_fraction_l221_221123

theorem fraction_of_fraction (a b c d : ℚ) (h1 : a = 2) (h2 : b = 9) (h3 : c = 3) (h4 : d = 4) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by
  rw [div_mul_div, mul_comm c _] -- Using properties of divisions and multiplications.
  sorry

end fraction_of_fraction_l221_221123


namespace solve_system_l221_221086

theorem solve_system :
  ∃ x y : ℚ, (4 * x - 7 * y = -20) ∧ (9 * x + 3 * y = -21) ∧ (x = -69 / 25) ∧ (y = 32 / 25) := by
  sorry

end solve_system_l221_221086


namespace set_intersection_example_l221_221203

theorem set_intersection_example (A : Set ℝ) (B : Set ℝ):
  A = { -1, 1, 2, 4 } → 
  B = { x | |x - 1| ≤ 1 } → 
  A ∩ B = {1, 2} :=
by
  intros hA hB
  sorry

end set_intersection_example_l221_221203


namespace find_AD_l221_221989

-- Define the geometrical context and constraints
variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (AB AC AD BD CD : ℝ) (x : ℝ)

-- Assume the given conditions
def problem_conditions := 
  (AB = 50) ∧
  (AC = 41) ∧
  (BD = 10 * x) ∧
  (CD = 3 * x) ∧
  (AB^2 = AD^2 + BD^2) ∧
  (AC^2 = AD^2 + CD^2)

-- Formulate the problem question and the correct answer
theorem find_AD (h : problem_conditions AB AC AD BD CD x) : AD = 40 :=
sorry

end find_AD_l221_221989


namespace sin_x_sin_y_eq_sin_beta_sin_gamma_l221_221894

theorem sin_x_sin_y_eq_sin_beta_sin_gamma
  (A B C M : Type)
  (AM BM CM : ℝ)
  (alpha beta gamma x y : ℝ)
  (h1 : AM * AM = BM * CM)
  (h2 : BM ≠ 0)
  (h3 : CM ≠ 0)
  (hx : AM / BM = Real.sin beta / Real.sin x)
  (hy : AM / CM = Real.sin gamma / Real.sin y) :
  Real.sin x * Real.sin y = Real.sin beta * Real.sin gamma := 
sorry

end sin_x_sin_y_eq_sin_beta_sin_gamma_l221_221894


namespace largest_three_digit_multiple_of_4_and_5_l221_221299

theorem largest_three_digit_multiple_of_4_and_5 : 
  ∃ (n : ℕ), n < 1000 ∧ n ≥ 100 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ n = 980 :=
by
  sorry

end largest_three_digit_multiple_of_4_and_5_l221_221299


namespace robin_total_cost_l221_221078

def num_letters_in_name (name : String) : Nat := name.length

def calculate_total_cost (names : List String) (cost_per_bracelet : Nat) : Nat :=
  let total_bracelets := names.foldl (fun acc name => acc + num_letters_in_name name) 0
  total_bracelets * cost_per_bracelet

theorem robin_total_cost : 
  calculate_total_cost ["Jessica", "Tori", "Lily", "Patrice"] 2 = 44 :=
by
  sorry

end robin_total_cost_l221_221078


namespace boys_and_girls_at_bus_stop_l221_221824

theorem boys_and_girls_at_bus_stop (H M : ℕ) 
  (h1 : H = 2 * (M - 15)) 
  (h2 : M - 15 = 5 * (H - 45)) : 
  H = 50 ∧ M = 40 := 
by 
  sorry

end boys_and_girls_at_bus_stop_l221_221824


namespace evaluate_f_diff_l221_221570

def f (x : ℝ) : ℝ := x^4 + 3 * x^3 + 2 * x^2 + 7 * x

theorem evaluate_f_diff:
  f 6 - f (-6) = 1380 := by
  sorry

end evaluate_f_diff_l221_221570


namespace analysis_hours_l221_221616

-- Define the conditions: number of bones and minutes per bone
def number_of_bones : Nat := 206
def minutes_per_bone : Nat := 45

-- Define the conversion factor: minutes per hour
def minutes_per_hour : Nat := 60

-- Define the total minutes spent analyzing all bones
def total_minutes (number_of_bones minutes_per_bone : Nat) : Nat :=
  number_of_bones * minutes_per_bone

-- Define the total hours required for analysis
def total_hours (total_minutes minutes_per_hour : Nat) : Float :=
  total_minutes.toFloat / minutes_per_hour.toFloat

-- Prove that total_hours equals 154.5 hours
theorem analysis_hours : total_hours (total_minutes number_of_bones minutes_per_bone) minutes_per_hour = 154.5 := by
  sorry

end analysis_hours_l221_221616


namespace sin_identity_l221_221360

theorem sin_identity (α : ℝ) (h : Real.sin (π * α) = 4 / 5) : 
  Real.sin (π / 2 + 2 * α) = -24 / 25 :=
by
  sorry

end sin_identity_l221_221360


namespace ratio_expression_value_l221_221717

theorem ratio_expression_value (A B C : ℚ) (hA : A = 3 * B / 2) (hC : C = 5 * B / 2) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by sorry

end ratio_expression_value_l221_221717


namespace smallest_b_for_factorization_l221_221024

-- Let us state the problem conditions and the objective
theorem smallest_b_for_factorization :
  ∃ (b : ℕ), b = 92 ∧ ∃ (p q : ℤ), (p + q = b) ∧ (p * q = 2016) :=
begin
  sorry
end

end smallest_b_for_factorization_l221_221024


namespace belongs_to_one_progression_l221_221878

-- Define the arithmetic progression and membership property
def is_arith_prog (P : ℕ → Prop) : Prop :=
  ∃ a d, ∀ n, P (a + n * d)

-- Define the given conditions
def condition (P1 P2 P3 : ℕ → Prop) : Prop :=
  is_arith_prog P1 ∧ is_arith_prog P2 ∧ is_arith_prog P3 ∧
  (P1 1 ∨ P2 1 ∨ P3 1) ∧
  (P1 2 ∨ P2 2 ∨ P3 2) ∧
  (P1 3 ∨ P2 3 ∨ P3 3) ∧
  (P1 4 ∨ P2 4 ∨ P3 4) ∧
  (P1 5 ∨ P2 5 ∨ P3 5) ∧
  (P1 6 ∨ P2 6 ∨ P3 6) ∧
  (P1 7 ∨ P2 7 ∨ P3 7) ∧
  (P1 8 ∨ P2 8 ∨ P3 8)

-- Statement to prove
theorem belongs_to_one_progression (P1 P2 P3 : ℕ → Prop) (h : condition P1 P2 P3) : 
  P1 1980 ∨ P2 1980 ∨ P3 1980 := 
by
sorry

end belongs_to_one_progression_l221_221878


namespace battery_change_month_battery_change_in_november_l221_221047

theorem battery_change_month :
  (119 % 12) = 11 := by
  sorry

theorem battery_change_in_november (n : Nat) (h1 : n = 18) :
  let month := ((n - 1) * 7) % 12
  month = 11 := by
  sorry

end battery_change_month_battery_change_in_november_l221_221047


namespace taqeesha_grade_l221_221109

theorem taqeesha_grade (s : ℕ → ℕ) (h1 : (s 16) = 77) (h2 : (s 17) = 78) : s 17 - s 16 = 94 :=
by
  -- Add definitions and sorry to skip the proof
  sorry

end taqeesha_grade_l221_221109


namespace total_tickets_sold_l221_221291

theorem total_tickets_sold (A C : ℕ) (total_revenue : ℝ) (cost_adult cost_child : ℝ) :
  (cost_adult = 6.00) →
  (cost_child = 4.50) →
  (total_revenue = 2100.00) →
  (C = 200) →
  (cost_adult * ↑A + cost_child * ↑C = total_revenue) →
  A + C = 400 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof omitted
  sorry

end total_tickets_sold_l221_221291


namespace math_club_total_members_l221_221582

theorem math_club_total_members (female_count : ℕ) (h_female : female_count = 6) (h_male_ratio : ∃ male_count : ℕ, male_count = 2 * female_count) :
  ∃ total_members : ℕ, total_members = female_count + classical.some h_male_ratio :=
by
  let male_count := classical.some h_male_ratio
  have h_male_count : male_count = 12 := by sorry
  existsi (female_count + male_count)
  rw [h_female, h_male_count]
  exact rfl

end math_club_total_members_l221_221582


namespace maximum_value_of_piecewise_function_l221_221279

noncomputable def piecewise_function (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x + 3 else 
  if 0 < x ∧ x ≤ 1 then x + 3 else 
  -x + 5

theorem maximum_value_of_piecewise_function : ∃ M, ∀ x, piecewise_function x ≤ M ∧ (∀ y, (∀ x, piecewise_function x ≤ y) → M ≤ y) := 
by
  use 4
  sorry

end maximum_value_of_piecewise_function_l221_221279


namespace focal_chord_length_perpendicular_l221_221709

theorem focal_chord_length_perpendicular (x1 y1 x2 y2 : ℝ)
  (h_parabola : y1^2 = 4 * x1 ∧ y2^2 = 4 * x2)
  (h_perpendicular : x1 = x2) :
  abs (y1 - y2) = 4 :=
by sorry

end focal_chord_length_perpendicular_l221_221709


namespace unique_function_l221_221020

noncomputable def f (x : ℝ) : ℝ := sorry -- We will prove that f(x) = A * x^(1 + sqrt(2))

theorem unique_function (A : ℝ) (f : ℝ → ℝ)
  (h_cont : ContinuousOn f (set.Ici 0)) 
  (h_pos : ∀ x > 0, 0 < f x) 
  (h_centroid : ∀ x0 > 0,
    (1 / x0) * ∫ t in 0..x0, t * f t = (1 / (x0 * ∫ t in 0..x0, f t)) * (∫ t in 0..x0, f t)^2) :
  ∃ (A : ℝ), ∀ (x : ℝ), f x = A * x^(1 + Real.sqrt 2) :=
begin
  -- Proof omitted
  sorry
end

end unique_function_l221_221020


namespace avery_egg_cartons_filled_l221_221002

-- Definitions (conditions identified in step a)
def total_chickens : ℕ := 20
def eggs_per_chicken : ℕ := 6
def eggs_per_carton : ℕ := 12

-- Theorem statement (equivalent to the problem statement)
theorem avery_egg_cartons_filled : (total_chickens * eggs_per_chicken) / eggs_per_carton = 10 :=
by
  -- Proof omitted; sorry used to denote unfinished proof
  sorry

end avery_egg_cartons_filled_l221_221002


namespace largest_n_unique_k_l221_221942

-- Defining the main theorem statement
theorem largest_n_unique_k :
  ∃ (n : ℕ), (n = 63) ∧ (∃! (k : ℤ), (9 / 17 : ℚ) < (n : ℚ) / ((n + k) : ℚ) ∧ (n : ℚ) / ((n + k) : ℚ) < (8 / 15 : ℚ)) :=
sorry

end largest_n_unique_k_l221_221942


namespace manuscript_pages_l221_221847

theorem manuscript_pages (P : ℝ)
  (h1 : 10 * (0.05 * P) + 10 * 5 = 250) : P = 400 :=
sorry

end manuscript_pages_l221_221847


namespace quadratic_has_real_solutions_l221_221096

theorem quadratic_has_real_solutions (m : ℝ) : 
  (∃ x : ℝ, (m - 2) * x^2 - 2 * x + 1 = 0) → m ≤ 3 := 
by
  sorry

end quadratic_has_real_solutions_l221_221096


namespace missing_number_unique_l221_221860

theorem missing_number_unique (x : ℤ) 
  (h : |9 - x * (3 - 12)| - |5 - 11| = 75) : 
  x = 8 :=
sorry

end missing_number_unique_l221_221860


namespace last_digit_of_2_pow_2004_l221_221449

theorem last_digit_of_2_pow_2004 : (2 ^ 2004) % 10 = 6 := 
by {
  sorry
}

end last_digit_of_2_pow_2004_l221_221449


namespace soda_preference_respondents_l221_221579

noncomputable def fraction_of_soda (angle_soda : ℝ) (total_angle : ℝ) : ℝ :=
  angle_soda / total_angle

noncomputable def number_of_soda_preference (total_people : ℕ) (fraction : ℝ) : ℝ :=
  total_people * fraction

theorem soda_preference_respondents (total_people : ℕ) (angle_soda : ℝ) (total_angle : ℝ) : 
  total_people = 520 → angle_soda = 298 → total_angle = 360 → 
  number_of_soda_preference total_people (fraction_of_soda angle_soda total_angle) = 429 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  unfold fraction_of_soda number_of_soda_preference
  -- further calculation steps
  sorry

end soda_preference_respondents_l221_221579


namespace part_a_part_b_l221_221308

open Nat

theorem part_a (n: ℕ) (h_pos: 0 < n) : (2^n - 1) % 7 = 0 ↔ ∃ k : ℕ, k > 0 ∧ n = 3 * k :=
sorry

theorem part_b (n: ℕ) (h_pos: 0 < n) : (2^n + 1) % 7 ≠ 0 :=
sorry

end part_a_part_b_l221_221308


namespace part_I_part_II_l221_221555

noncomputable def f (x a : ℝ) : ℝ := x - 1 - a * Real.log x

theorem part_I (a : ℝ) (h1 : 0 < a) (h2 : ∀ x : ℝ, 0 < x → f x a ≥ 0) : a = 1 := 
sorry

theorem part_II (n : ℕ) (hn : 0 < n) : 
  let an := (1 + 1 / (n : ℝ)) ^ n
  let bn := (1 + 1 / (n : ℝ)) ^ (n + 1)
  an < Real.exp 1 ∧ Real.exp 1 < bn := 
sorry

end part_I_part_II_l221_221555


namespace sasha_fractions_l221_221775

theorem sasha_fractions (x y z t : ℕ) 
  (hx : x ≠ y) (hxy : x ≠ z) (hxz : x ≠ t)
  (hyz : y ≠ z) (hyt : y ≠ t) (hzt : z ≠ t) :
  ∃ (q1 q2 : ℚ), (q1 ≠ q2) ∧ 
    (q1 = x / y ∨ q1 = x / z ∨ q1 = x / t ∨ q1 = y / x ∨ q1 = y / z ∨ q1 = y / t ∨ q1 = z / x ∨ q1 = z / y ∨ q1 = z / t ∨ q1 = t / x ∨ q1 = t / y ∨ q1 = t / z) ∧ 
    (q2 = x / y ∨ q2 = x / z ∨ q2 = x / t ∨ q2 = y / x ∨ q2 = y / z ∨ q2 = y / t ∨ q2 = z / x ∨ q2 = z / y ∨ q2 = z / t ∨ q2 = t / x ∨ q2 = t / y ∨ q2 = t / z) ∧ 
    |q1 - q2| ≤ 11 / 60 := by 
  sorry

end sasha_fractions_l221_221775


namespace each_person_pays_12_10_l221_221466

noncomputable def total_per_person : ℝ :=
  let taco_salad := 10
  let daves_single := 6 * 5
  let french_fries := 5 * 2.5
  let peach_lemonade := 7 * 2
  let apple_pecan_salad := 4 * 6
  let chocolate_frosty := 5 * 3
  let chicken_sandwiches := 3 * 4
  let chili := 2 * 3.5
  let subtotal := taco_salad + daves_single + french_fries + peach_lemonade + apple_pecan_salad + chocolate_frosty + chicken_sandwiches + chili
  let discount := 0.10
  let tax := 0.08
  let subtotal_after_discount := subtotal * (1 - discount)
  let total_after_tax := subtotal_after_discount * (1 + tax)
  total_after_tax / 10

theorem each_person_pays_12_10 :
  total_per_person = 12.10 :=
by
  -- omitted proof
  sorry

end each_person_pays_12_10_l221_221466


namespace symmetric_circle_eq_l221_221097

/-- Define the equation of the circle C -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Define the equation of the line l -/
def line_equation (x y : ℝ) : Prop := x + y - 1 = 0

/-- 
The symmetric circle to C with respect to line l 
has the equation (x - 1)^2 + (y - 1)^2 = 4.
-/
theorem symmetric_circle_eq (x y : ℝ) :
  (∃ x y : ℝ, circle_equation x y) → 
  (∃ x y : ℝ, line_equation x y) →
  (∃ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 4) :=
by
  sorry

end symmetric_circle_eq_l221_221097


namespace value_of_x_plus_2y_l221_221271

theorem value_of_x_plus_2y :
  let x := 3
  let y := 1
  x + 2 * y = 5 :=
by
  sorry

end value_of_x_plus_2y_l221_221271


namespace johnPaysPerYear_l221_221431

-- Define the conditions
def epiPenCost : ℝ := 500
def insuranceCoverage : ℝ := 0.75
def epiPenFrequencyPerYear : ℝ := 2 -- Twice a year since 12 months / 6 months per EpiPen

-- Calculate the cost after insurance
def costAfterInsurance (cost : ℝ) (coverage : ℝ) : ℝ :=
  cost * (1 - coverage)

-- Calculate the annual cost
def annualCost (freq : ℝ) (cost : ℝ) : ℝ :=
  freq * cost

-- The statement asserting the proof
theorem johnPaysPerYear (cost : ℝ) (coverage : ℝ) (freq : ℝ) : 
  epiPenCost = cost → 
  insuranceCoverage = coverage → 
  epiPenFrequencyPerYear = freq → 
  annualCost freq (costAfterInsurance cost coverage) = 250 := 
by 
  intros _ _ _ 
  sorry

end johnPaysPerYear_l221_221431


namespace circle_chords_intersect_radius_square_l221_221502

theorem circle_chords_intersect_radius_square
  (r : ℝ) -- The radius of the circle
  (AB CD BP : ℝ) -- The lengths of chords AB, CD, and segment BP
  (angle_APD : ℝ) -- The angle ∠APD in degrees
  (AB_len : AB = 8)
  (CD_len : CD = 12)
  (BP_len : BP = 10)
  (angle_APD_val : angle_APD = 60) :
  r^2 = 91 := 
sorry

end circle_chords_intersect_radius_square_l221_221502


namespace binary_operation_l221_221023

theorem binary_operation : 
  let a := 0b11011
  let b := 0b1101
  let c := 0b1010
  let result := 0b110011101  
  ((a * b) - c) = result := by
  sorry

end binary_operation_l221_221023


namespace flu_epidemic_infection_rate_l221_221349

theorem flu_epidemic_infection_rate : 
  ∃ x : ℝ, 1 + x + x * (1 + x) = 100 ∧ x = 9 := 
by
  sorry

end flu_epidemic_infection_rate_l221_221349


namespace one_third_greater_than_333_l221_221181

theorem one_third_greater_than_333 :
  (1 : ℝ) / 3 > (333 : ℝ) / 1000 - 1 / 3000 :=
sorry

end one_third_greater_than_333_l221_221181


namespace afternoon_sales_l221_221665

variable (x y : ℕ)

theorem afternoon_sales (hx : y = 2 * x) (hy : x + y = 390) : y = 260 := by
  sorry

end afternoon_sales_l221_221665


namespace no_passing_quadrant_III_l221_221869

def y (k x : ℝ) : ℝ := k * x - k

theorem no_passing_quadrant_III (k : ℝ) (h : k < 0) :
  ¬(∃ x y : ℝ, x < 0 ∧ y < 0 ∧ y = k * x - k) :=
sorry

end no_passing_quadrant_III_l221_221869


namespace regular_polygon_sides_l221_221163

theorem regular_polygon_sides (ex_angle : ℝ) (hne_zero : ex_angle ≠ 0)
  (sum_ext_angles : ∀ (n : ℕ), n > 2 → n * ex_angle = 360) :
  ∃ (n : ℕ), n * 15 = 360 ∧ n = 24 :=
by 
  sorry

end regular_polygon_sides_l221_221163


namespace income_expenditure_ratio_l221_221629

theorem income_expenditure_ratio
  (I E : ℕ)
  (h1 : I = 18000)
  (S : ℕ)
  (h2 : S = 2000)
  (h3 : S = I - E) :
  I.gcd E = 2000 ∧ I / I.gcd E = 9 ∧ E / I.gcd E = 8 :=
by sorry

end income_expenditure_ratio_l221_221629


namespace sum_of_coefficients_l221_221870

noncomputable def u : ℕ → ℕ
| 0       => 5
| (n + 1) => u n + (3 + 4 * (n - 1))

theorem sum_of_coefficients :
  (2 + -3 + 6) = 5 :=
by {
  sorry
}

end sum_of_coefficients_l221_221870


namespace value_of_a_l221_221746

theorem value_of_a (x a : ℤ) (h : x = 3 ∧ x^2 = a) : a = 9 :=
sorry

end value_of_a_l221_221746


namespace evaluate_expression_l221_221637

theorem evaluate_expression : (5 + 2) + (8 + 6) + (4 + 7) + (3 + 2) = 37 := 
sorry

end evaluate_expression_l221_221637


namespace magnitude_z_is_sqrt_2_l221_221368

open Complex

noncomputable def z (x y : ℝ) : ℂ := x + y * I

theorem magnitude_z_is_sqrt_2 (x y : ℝ) (h1 : (2 * x) / (1 - I) = 1 + y * I) : abs (z x y) = Real.sqrt 2 :=
by
  -- You would fill in the proof steps here based on the problem's solution.
  sorry

end magnitude_z_is_sqrt_2_l221_221368


namespace value_of_MN_l221_221404

theorem value_of_MN (M N : ℝ) (log : ℝ → ℝ → ℝ)
    (h1 : log (M ^ 2) N = log N (M ^ 2))
    (h2 : M ≠ N)
    (h3 : M * N > 0)
    (h4 : M ≠ 1)
    (h5 : N ≠ 1) :
    M * N = N^(1/2) :=
  sorry

end value_of_MN_l221_221404


namespace base8_357_plus_base13_4CD_eq_1084_l221_221694

def C := 12
def D := 13

def base8_357 := 3 * (8^2) + 5 * (8^1) + 7 * (8^0)
def base13_4CD := 4 * (13^2) + C * (13^1) + D * (13^0)

theorem base8_357_plus_base13_4CD_eq_1084 :
  base8_357 + base13_4CD = 1084 :=
by
  sorry

end base8_357_plus_base13_4CD_eq_1084_l221_221694


namespace fraction_of_sophomores_attending_fair_l221_221964

theorem fraction_of_sophomores_attending_fair
  (s j n : ℕ)
  (h1 : s = j)
  (h2 : j = n)
  (soph_attend : ℚ)
  (junior_attend : ℚ)
  (senior_attend : ℚ)
  (fraction_s : soph_attend = 4/5 * s)
  (fraction_j : junior_attend = 3/4 * j)
  (fraction_n : senior_attend = 1/3 * n) :
  soph_attend / (soph_attend + junior_attend + senior_attend) = 240 / 565 :=
by
  sorry

end fraction_of_sophomores_attending_fair_l221_221964


namespace problem_l221_221598

theorem problem (a b c : ℕ) (h : 28 * a + 30 * b + 31 * c = 365) : a + b + c = 12 :=
  sorry

end problem_l221_221598


namespace ratio_problem_l221_221742

theorem ratio_problem (A B C : ℚ) (h : A / B = 3 / 2) (h' : B / C = 2 / 5) : (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := 
by
  sorry

end ratio_problem_l221_221742


namespace evaluate_expression_l221_221982

noncomputable def expression_equal : Prop :=
  let a := (11: ℝ)
  let b := (11 : ℝ)^((1 : ℝ) / 6)
  let c := (11 : ℝ)^((1 : ℝ) / 5)
  (b / c = a^(-((1 : ℝ) / 30)))

theorem evaluate_expression :
  expression_equal :=
sorry

end evaluate_expression_l221_221982


namespace find_point_N_l221_221997

-- Definition of symmetrical reflection across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Given condition
def point_M : ℝ × ℝ := (1, 3)

-- Theorem statement
theorem find_point_N : reflect_x point_M = (1, -3) :=
by
  sorry

end find_point_N_l221_221997


namespace max_dn_eq_401_l221_221101

open BigOperators

def a (n : ℕ) : ℕ := 100 + n^2

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_dn_eq_401 : ∃ n, d n = 401 ∧ ∀ m, d m ≤ 401 := by
  -- Proof will be filled here
  sorry

end max_dn_eq_401_l221_221101


namespace value_of_expression_l221_221738

variables {A B C : ℚ}

def conditions (A B C : ℚ) : Prop := A / B = 3 / 2 ∧ B / C = 2 / 5

theorem value_of_expression (h : conditions A B C) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
sorry

end value_of_expression_l221_221738


namespace find_m_plus_n_l221_221761

theorem find_m_plus_n (AB AC BC : ℕ) (RS : ℚ) (m n : ℕ) 
  (hmn_rel_prime : Nat.gcd m n = 1)
  (hAB : AB = 1995)
  (hAC : AC = 1994)
  (hBC : BC = 1993)
  (hRS : RS = m / n) :
  m + n = 997 :=
sorry

end find_m_plus_n_l221_221761


namespace apprentice_daily_output_l221_221831

namespace Production

variables (x y : ℝ)

theorem apprentice_daily_output
  (h1 : 4 * x + 7 * y = 765)
  (h2 : 6 * x + 2 * y = 765) :
  y = 45 :=
sorry

end Production

end apprentice_daily_output_l221_221831


namespace rowing_upstream_speed_l221_221504

theorem rowing_upstream_speed (Vm Vdown : ℝ) (H1 : Vm = 20) (H2 : Vdown = 33) :
  ∃ Vup Vs : ℝ, Vup = Vm - Vs ∧ Vs = Vdown - Vm ∧ Vup = 7 := 
by {
  sorry
}

end rowing_upstream_speed_l221_221504


namespace maximize_sum_l221_221583

def a_n (n : ℕ): ℤ := 11 - 2 * (n - 1)

theorem maximize_sum (n : ℕ) (S : ℕ → ℤ → Prop) :
  (∀ n, S n (a_n n)) → (a_n n ≥ 0) → n = 6 :=
by
  sorry

end maximize_sum_l221_221583


namespace k_value_l221_221231

theorem k_value {x y k : ℝ} (h : ∃ c : ℝ, (x ^ 2 + k * x * y + 49 * y ^ 2) = c ^ 2) : k = 14 ∨ k = -14 :=
by sorry

end k_value_l221_221231


namespace largest_n_unique_k_l221_221939

theorem largest_n_unique_k :
  ∃ (n : ℕ), (∀ (k1 k2 : ℕ), 
    (9 / 17 < n / (n + k1) → n / (n + k1) < 8 / 15 → 9 / 17 < n / (n + k2) → n / (n + k2) < 8 / 15 → k1 = k2) ∧ 
    n = 72) :=
sorry

end largest_n_unique_k_l221_221939


namespace automotive_test_l221_221813

noncomputable def total_distance (D : ℝ) (t : ℝ) : ℝ := 3 * D

theorem automotive_test (D : ℝ) (h_time : (D / 4 + D / 5 + D / 6 = 37)) : total_distance D 37 = 180 :=
  by
    -- This skips the proof, only the statement is given
    sorry

end automotive_test_l221_221813


namespace fourth_bus_people_difference_l221_221285

def bus1_people : Nat := 12
def bus2_people : Nat := 2 * bus1_people
def bus3_people : Nat := bus2_people - 6
def total_people : Nat := 75
def bus4_people : Nat := total_people - (bus1_people + bus2_people + bus3_people)
def difference_people : Nat := bus4_people - bus1_people

theorem fourth_bus_people_difference : difference_people = 9 := by
  -- Proof logic here
  sorry

end fourth_bus_people_difference_l221_221285


namespace son_l221_221326

theorem son's_age (S M : ℕ) 
  (h1 : M = S + 35)
  (h2 : M + 2 = 2 * (S + 2)) : 
  S = 33 := 
by
  sorry

end son_l221_221326


namespace log_base_8_of_512_l221_221977

theorem log_base_8_of_512 : log 8 512 = 3 :=
by {
  -- math proof here
  sorry
}

end log_base_8_of_512_l221_221977


namespace f_eq_f_at_neg_one_f_at_neg_500_l221_221465

noncomputable def f : ℝ → ℝ := sorry

theorem f_eq : ∀ x y : ℝ, f (x * y) + x = x * f y + f x := sorry
theorem f_at_neg_one : f (-1) = 1 := sorry

theorem f_at_neg_500 : f (-500) = 999 := sorry

end f_eq_f_at_neg_one_f_at_neg_500_l221_221465


namespace lola_pop_tarts_baked_l221_221770

theorem lola_pop_tarts_baked :
  ∃ P : ℕ, (13 + P + 8) + (16 + 12 + 14) = 73 ∧ P = 10 := by
  sorry

end lola_pop_tarts_baked_l221_221770


namespace brownies_in_pan_l221_221672

theorem brownies_in_pan : 
    ∀ (pan_length pan_width brownie_length brownie_width : ℕ), 
    pan_length = 24 -> 
    pan_width = 20 -> 
    brownie_length = 3 -> 
    brownie_width = 2 -> 
    (pan_length * pan_width) / (brownie_length * brownie_width) = 80 := 
by
  intros pan_length pan_width brownie_length brownie_width h1 h2 h3 h4
  sorry

end brownies_in_pan_l221_221672


namespace age_composition_is_decline_l221_221152

-- Define the population and age groups
variable (P : Type)
variable (Y E : P → ℕ) -- Functions indicating the number of young and elderly individuals

-- Assumptions as per the conditions
axiom fewer_young_more_elderly (p : P) : Y p < E p

-- Conclusion: Prove that the population is of Decline type.
def age_composition_decline (p : P) : Prop :=
  Y p < E p

theorem age_composition_is_decline (p : P) : age_composition_decline P Y E p := by
  sorry

end age_composition_is_decline_l221_221152


namespace ferris_wheel_time_l221_221454

noncomputable def radius : ℝ := 30
noncomputable def revolution_time : ℝ := 90
noncomputable def desired_height : ℝ := 15

theorem ferris_wheel_time :
  ∃ t : ℝ, 0 <= t ∧ t <= revolution_time / 2 ∧ 30 * real.cos ((real.pi / 45) * t) + 30 = 15 ∧ t = 30 :=
by
  sorry

end ferris_wheel_time_l221_221454


namespace square_free_condition_l221_221695

/-- Define square-free integer -/
def square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m * m ∣ n → m = 1

/-- Define the problem in Lean -/
theorem square_free_condition (p : ℕ) (hp : p ≥ 3 ∧ Nat.Prime p) :
  (∀ q : ℕ, Nat.Prime q ∧ q < p → square_free (p - (p / q) * q)) ↔
  p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 13 := by
  sorry

end square_free_condition_l221_221695


namespace caitlin_draws_pairs_probability_l221_221685

def caitlin_probability : ℚ :=
  let total_ways := nat.choose 10 6 in
  let favorable_ways := nat.choose 5 2 * nat.choose 3 2 * 1 * 1 in
  favorable_ways / total_ways

theorem caitlin_draws_pairs_probability :
  caitlin_probability = 1 / 7 :=
by {
  sorry
}

end caitlin_draws_pairs_probability_l221_221685


namespace percentage_increase_l221_221433

theorem percentage_increase (old_earnings new_earnings : ℝ) (h_old : old_earnings = 50) (h_new : new_earnings = 70) :
  ((new_earnings - old_earnings) / old_earnings) * 100 = 40 :=
by
  rw [h_old, h_new]
  -- Simplification and calculation steps would go here
  sorry

end percentage_increase_l221_221433


namespace janet_owes_wages_and_taxes_l221_221898

theorem janet_owes_wages_and_taxes :
  (∀ (workdays : ℕ) (hours : ℕ) (warehouse_workers : ℕ) (manager_workers : ℕ) (warehouse_wage : ℕ) (manager_wage : ℕ) (tax_rate : ℚ),
    workdays = 25 →
    hours = 8 →
    warehouse_workers = 4 →
    manager_workers = 2 →
    warehouse_wage = 15 →
    manager_wage = 20 →
    tax_rate = 0.1 →
    let total_hours := workdays * hours
        warehouse_monthly := total_hours * warehouse_wage
        manager_monthly := total_hours * manager_wage
        total_wage := warehouse_monthly * warehouse_workers + manager_monthly * manager_workers
        total_taxes := total_wage * tax_rate in
    total_wage + total_taxes = 22000) :=
begin
  intros,
  rw [← mul_assoc, mul_comm 25 8, mul_assoc],
  have h1 : 25 * 8 = 200, {norm_num},
  rw h1,
  have h2 : 200 * 15 * 4 = 12000, {norm_num},
  have h3 : 200 * 20 * 2 = 8000, {norm_num},
  rw [h2, h3],
  have h4 : 12000 + 8000 = 20000, {norm_num},
  have h5 : 20000 * 0.1 = 2000, {norm_num},
  rw [h4, h5],
  norm_num,
end

end janet_owes_wages_and_taxes_l221_221898


namespace smaller_number_is_neg_five_l221_221795

theorem smaller_number_is_neg_five (x y : ℤ) (h1 : x + y = 30) (h2 : x - y = 40) : y = -5 :=
by
  sorry

end smaller_number_is_neg_five_l221_221795


namespace quadratic_eq_solution_trig_expression_calc_l221_221951

-- Part 1: Proof for the quadratic equation solution
theorem quadratic_eq_solution : ∀ (x : ℝ), x^2 - 4 * x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7 :=
by
  sorry

-- Part 2: Proof for trigonometric expression calculation
theorem trig_expression_calc : (-1 : ℝ) ^ 2 + 2 * Real.sin (Real.pi / 3) - Real.tan (Real.pi / 4) = Real.sqrt 3 :=
by
  sorry

end quadratic_eq_solution_trig_expression_calc_l221_221951


namespace log8_512_eq_3_l221_221973

theorem log8_512_eq_3 : ∃ x : ℝ, 8^x = 512 ∧ x = 3 :=
by
  use 3
  have h1 : 8 = 2^3 := by norm_num
  have h2 : 512 = 2^9 := by norm_num
  calc
    8^3 = (2^3)^3 := by rw h1
    ... = 2^(3*3) := by rw [pow_mul]
    ... = 2^9    := by norm_num
    ... = 512    := by rw h2

  sorry

end log8_512_eq_3_l221_221973


namespace find_k_values_l221_221418

theorem find_k_values (k : ℝ) : 
  ((2 * 1 + 3 * k = 0) ∨
   (1 * 2 + (3 - k) * 3 = 0) ∨
   (1 * 1 + (3 - k) * k = 0)) →
   (k = -2/3 ∨ k = 11/3 ∨ k = (3 + Real.sqrt 3)/2 ∨ k = (3 - Real.sqrt 3)/2) := 
by
  sorry

end find_k_values_l221_221418


namespace simplify_2M_minus_N_value_at_neg_1_M_gt_N_l221_221990

-- Definitions of M and N
def M (x : ℝ) : ℝ := 4 * x^2 - 2 * x - 1
def N (x : ℝ) : ℝ := 3 * x^2 - 2 * x - 5

-- The simplified expression for 2M - N
theorem simplify_2M_minus_N {x : ℝ} : 2 * M x - N x = 5 * x^2 - 2 * x + 3 :=
by sorry

-- Value of the simplified expression when x = -1
theorem value_at_neg_1 : (5 * (-1)^2 - 2 * (-1) + 3) = 10 :=
by sorry

-- Relationship between M and N
theorem M_gt_N {x : ℝ} : M x > N x :=
by
  have h : M x - N x = x^2 + 4 := by sorry
  -- x^2 >= 0 for all x, so x^2 + 4 > 0 => M > N
  have nonneg : x^2 >= 0 := by sorry
  have add_pos : x^2 + 4 > 0 := by sorry
  sorry

end simplify_2M_minus_N_value_at_neg_1_M_gt_N_l221_221990


namespace mural_width_l221_221758

theorem mural_width (l p r c t w : ℝ) (h₁ : l = 6) (h₂ : p = 4) (h₃ : r = 1.5) (h₄ : c = 10) (h₅ : t = 192) :
  4 * 6 * w + 10 * (6 * w / 1.5) = 192 → w = 3 :=
by
  intros
  sorry

end mural_width_l221_221758


namespace combinatorial_problem_correct_l221_221956

def combinatorial_problem : Prop :=
  let boys := 4
  let girls := 3
  let chosen_boys := 3
  let chosen_girls := 2
  let num_ways_select := Nat.choose boys chosen_boys * Nat.choose girls chosen_girls
  let arrangements_no_consecutive_girls := 6 * Nat.factorial 4 / Nat.factorial 2
  num_ways_select * arrangements_no_consecutive_girls = 864

theorem combinatorial_problem_correct : combinatorial_problem := 
  by 
  -- proof to be provided
  sorry

end combinatorial_problem_correct_l221_221956


namespace john_pays_per_year_l221_221425

-- Define the costs and insurance parameters.
def cost_per_epipen : ℝ := 500
def insurance_coverage : ℝ := 0.75

-- Number of months in a year.
def months_in_year : ℕ := 12

-- Number of months each EpiPen lasts.
def months_per_epipen : ℕ := 6

-- Amount covered by insurance for each EpiPen.
def insurance_amount (cost : ℝ) (coverage : ℝ) : ℝ :=
  cost * coverage

-- Amount John pays after insurance for each EpiPen.
def amount_john_pays_per_epipen (cost : ℝ) (covered: ℝ) : ℝ :=
  cost - covered

-- Number of EpiPens John needs per year.
def epipens_per_year (months_in_year : ℕ) (months_per_epipen : ℕ) : ℕ :=
  months_in_year / months_per_epipen

-- Total amount John pays per year.
def total_amount_john_pays_per_year (amount_per_epipen : ℝ) (epipens_per_year : ℕ) : ℝ :=
  amount_per_epipen * epipens_per_year

-- Theorem to prove the correct answer.
theorem john_pays_per_year :
  total_amount_john_pays_per_year (amount_john_pays_per_epipen cost_per_epipen (insurance_amount cost_per_epipen insurance_coverage)) (epipens_per_year months_in_year months_per_epipen) = 250 := 
by
  sorry

end john_pays_per_year_l221_221425


namespace exists_a_func_max_on_interval_eq_zero_l221_221348

noncomputable def func (a x : ℝ) : ℝ :=
  cos x ^ 2 + a * sin x + 5 * a / 8 - 5 / 2

theorem exists_a_func_max_on_interval_eq_zero :
  ∃ (a : ℝ), a = 3 / 2 ∧
    ∃ (x ∈ Icc (0:ℝ) (π)), 
      ∀ (t ∈ Icc (0:ℝ) (π)), func a t ≤ func a x ∧ func a x = 0 :=
by
  sorry

end exists_a_func_max_on_interval_eq_zero_l221_221348


namespace greatest_y_least_y_greatest_integer_y_l221_221135

theorem greatest_y (y : ℤ) (H : (8 : ℝ) / 11 > y / 17) : y ≤ 12 :=
sorry

theorem least_y (y : ℤ) (H : (8 : ℝ) / 11 > y / 17) : y ≥ 12 :=
sorry

theorem greatest_integer_y : ∀ (y : ℤ), ((8 : ℝ) / 11 > y / 17) → y = 12 :=
by
  intro y H
  apply le_antisymm
  apply greatest_y y H
  apply least_y y H

end greatest_y_least_y_greatest_integer_y_l221_221135


namespace locus_of_midpoint_of_tangents_l221_221523

theorem locus_of_midpoint_of_tangents 
  (P Q Q1 Q2 : ℝ × ℝ)
  (L : P.2 = P.1 + 2)
  (C : ∀ p, p = Q1 ∨ p = Q2 → p.2 ^ 2 = 4 * p.1)
  (Q_is_midpoint : Q = ((Q1.1 + Q2.1) / 2, (Q1.2 + Q2.2) / 2)) :
  ∃ x y, (y - 1)^2 = 2 * (x - 3 / 2) := sorry

end locus_of_midpoint_of_tangents_l221_221523


namespace tailor_cut_difference_l221_221332

def dress_silk_cut : ℝ := 0.75
def dress_satin_cut : ℝ := 0.60
def dress_chiffon_cut : ℝ := 0.55
def pants_cotton_cut : ℝ := 0.50
def pants_polyester_cut : ℝ := 0.45

theorem tailor_cut_difference :
  (dress_silk_cut + dress_satin_cut + dress_chiffon_cut) - (pants_cotton_cut + pants_polyester_cut) = 0.95 :=
by
  sorry

end tailor_cut_difference_l221_221332


namespace exists_congruent_triangle_covering_with_parallel_side_l221_221393

variable {Point : Type}
variable [MetricSpace Point]
variable {Triangle : Type}
variable {Polygon : Type}

-- Definitions of triangle and polygon covering relationships.
def covers (T : Triangle) (P : Polygon) : Prop := sorry 
def congruent (T1 T2 : Triangle) : Prop := sorry
def side_parallel_or_coincident (T : Triangle) (P : Polygon) : Prop := sorry

-- Statement: Given a triangle covering a polygon, there exists a congruent triangle which covers the polygon 
-- and has one side parallel to or coincident with a side of the polygon.
theorem exists_congruent_triangle_covering_with_parallel_side 
  (ABC : Triangle) (M : Polygon) 
  (h_cover : covers ABC M) : 
  ∃ Δ : Triangle, congruent Δ ABC ∧ covers Δ M ∧ side_parallel_or_coincident Δ M := 
sorry

end exists_congruent_triangle_covering_with_parallel_side_l221_221393


namespace circle_value_in_grid_l221_221892

theorem circle_value_in_grid :
  ∃ (min_circle_val : ℕ), min_circle_val = 21 ∧ (∀ (max_circle_val : ℕ), ∃ (L : ℕ), L > max_circle_val) :=
by
  sorry

end circle_value_in_grid_l221_221892


namespace total_age_in_3_years_l221_221458

theorem total_age_in_3_years (Sam Sue Kendra : ℕ)
  (h1 : Kendra = 18)
  (h2 : Kendra = 3 * Sam)
  (h3 : Sam = 2 * Sue) :
  Sam + Sue + Kendra + 3 * 3 = 36 :=
by
  sorry

end total_age_in_3_years_l221_221458


namespace find_t_l221_221408

-- Define sets M and N
def M (t : ℝ) : Set ℝ := {1, t^2}
def N (t : ℝ) : Set ℝ := {-2, t + 2}

-- Goal: prove that t = 2 given M ∩ N ≠ ∅
theorem find_t (t : ℝ) (h : (M t ∩ N t).Nonempty) : t = 2 :=
sorry

end find_t_l221_221408


namespace necessary_condition_x_squared_minus_x_lt_zero_l221_221700

theorem necessary_condition_x_squared_minus_x_lt_zero (x : ℝ) :
  (x^2 - x < 0) → (-1 < x ∧ x < 1) ∧ ((-1 < x ∧ x < 1) → ¬ (x^2 - x < 0)) :=
by
  sorry

end necessary_condition_x_squared_minus_x_lt_zero_l221_221700


namespace relationship_abc_l221_221554

variables {a b c : ℝ}

-- Given conditions
def condition1 (a b c : ℝ) : Prop := 0 < a ∧ 0 < b ∧ 0 < c ∧ (11/6 : ℝ) * c < a + b ∧ a + b < 2 * c
def condition2 (a b c : ℝ) : Prop := (3/2 : ℝ) * a < b + c ∧ b + c < (5/3 : ℝ) * a
def condition3 (a b c : ℝ) : Prop := (5/2 : ℝ) * b < a + c ∧ a + c < (11/4 : ℝ) * b

-- Proof statement
theorem relationship_abc (a b c : ℝ) (h1 : condition1 a b c) (h2 : condition2 a b c) (h3 : condition3 a b c) :
  b < c ∧ c < a :=
by
  sorry

end relationship_abc_l221_221554


namespace sqrt_expression_l221_221969

open Real

theorem sqrt_expression :
  3 * sqrt 12 / (3 * sqrt (1 / 3)) - 2 * sqrt 3 = 6 - 2 * sqrt 3 :=
by
  sorry

end sqrt_expression_l221_221969


namespace evaluate_expression_l221_221210

theorem evaluate_expression (a b c : ℝ) (h : a / (30 - a) + b / (70 - b) + c / (55 - c) = 8) : 
  6 / (30 - a) + 14 / (70 - b) + 11 / (55 - c) = 2.2 :=
by 
  sorry

end evaluate_expression_l221_221210


namespace ratio_expression_value_l221_221718

theorem ratio_expression_value (A B C : ℚ) (hA : A = 3 * B / 2) (hC : C = 5 * B / 2) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by sorry

end ratio_expression_value_l221_221718


namespace inequality_solution_set_l221_221213

theorem inequality_solution_set (a b : ℝ) (h : ∀ x : ℝ, -3 < x ∧ x < 2 ↔ ax^2 - 5 * x + b > 0) :
  ∀ x : ℝ, x < -1/3 ∨ x > 1/2 ↔ b * x^2 - 5 * x + a > 0 :=
sorry

end inequality_solution_set_l221_221213


namespace value_of_expression_l221_221736

variables {A B C : ℚ}

def conditions (A B C : ℚ) : Prop := A / B = 3 / 2 ∧ B / C = 2 / 5

theorem value_of_expression (h : conditions A B C) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
sorry

end value_of_expression_l221_221736


namespace probability_of_snowing_at_least_once_l221_221478

theorem probability_of_snowing_at_least_once (p : ℚ) (h : p = 3 / 4) :
  let q := 1 - p in
  let not_snowing_five_days := q ^ 5 in
  let at_least_once := 1 - not_snowing_five_days in
  at_least_once = 1023 / 1024 :=
by
  have q_def : q = 1 - p := rfl,
  have not_snowing_five_days_def : not_snowing_five_days = q ^ 5 := rfl,
  have at_least_once_def : at_least_once = 1 - not_snowing_five_days := rfl,
  sorry

end probability_of_snowing_at_least_once_l221_221478


namespace problem_rewrite_equation_l221_221182

theorem problem_rewrite_equation :
  ∃ a b c : ℤ, a > 0 ∧ (64*(x^2) + 96*x - 81 = 0) → ((a*x + b)^2 = c) ∧ (a + b + c = 131) :=
sorry

end problem_rewrite_equation_l221_221182


namespace product_of_fractions_l221_221171

-- Define the fractions
def one_fourth : ℚ := 1 / 4
def one_half : ℚ := 1 / 2
def one_eighth : ℚ := 1 / 8

-- State the theorem we are proving
theorem product_of_fractions :
  one_fourth * one_half = one_eighth :=
by
  sorry

end product_of_fractions_l221_221171


namespace geometric_sum_4_terms_l221_221754

theorem geometric_sum_4_terms 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : a 2 = 9) 
  (h2 : a 5 = 243) 
  (hq : ∀ n, a (n + 1) = a n * q) 
  : a 1 * (1 - q^4) / (1 - q) = 120 := 
sorry

end geometric_sum_4_terms_l221_221754


namespace area_R3_l221_221834

-- Define the initial dimensions of rectangle R1
def length_R1 := 8
def width_R1 := 4

-- Define the dimensions of rectangle R2 after bisecting R1
def length_R2 := length_R1 / 2
def width_R2 := width_R1

-- Define the dimensions of rectangle R3 after bisecting R2
def length_R3 := length_R2 / 2
def width_R3 := width_R2

-- Prove that the area of R3 is 8
theorem area_R3 : (length_R3 * width_R3) = 8 := by
  -- Calculation for the theorem
  sorry

end area_R3_l221_221834


namespace probability_top_card_is_star_l221_221832

theorem probability_top_card_is_star :
  let total_cards := 65
  let suits := 5
  let ranks_per_suit := 13
  let star_cards := 13
  (star_cards / total_cards) = 1 / 5 :=
by
  sorry

end probability_top_card_is_star_l221_221832


namespace ratio_problem_l221_221741

theorem ratio_problem (A B C : ℚ) (h : A / B = 3 / 2) (h' : B / C = 2 / 5) : (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := 
by
  sorry

end ratio_problem_l221_221741


namespace polynomial_equivalence_l221_221304

variable (x : ℝ) -- Define variable x

-- Define the expressions.
def expr1 := (3 * x^2 + 5 * x + 8) * (x + 2)
def expr2 := (x + 2) * (x^2 + 5 * x - 72)
def expr3 := (4 * x - 15) * (x + 2) * (x + 6)

-- Define the expression to be proved.
def original_expr := expr1 - expr2 + expr3
def simplified_expr := 6 * x^3 + 21 * x^2 + 18 * x

-- The theorem to prove the equivalence of the original and simplified expressions.
theorem polynomial_equivalence : original_expr = simplified_expr :=
by sorry -- proof to be filled in

end polynomial_equivalence_l221_221304


namespace triangle_inequality_x_not_2_l221_221548

theorem triangle_inequality_x_not_2 (x : ℝ) (h1 : 2 < x) (h2 : x < 8) : x ≠ 2 :=
by 
  sorry

end triangle_inequality_x_not_2_l221_221548


namespace avery_egg_cartons_l221_221003

theorem avery_egg_cartons 
  (num_chickens : ℕ) (eggs_per_chicken : ℕ) (carton_capacity : ℕ)
  (h1 : num_chickens = 20) (h2 : eggs_per_chicken = 6) (h3 : carton_capacity = 12) :
  (num_chickens * eggs_per_chicken) / carton_capacity = 10 :=
by sorry

end avery_egg_cartons_l221_221003


namespace sum_and_product_of_three_numbers_l221_221140

variables (a b c : ℝ)

-- Conditions
axiom h1 : a + b = 35
axiom h2 : b + c = 47
axiom h3 : c + a = 52

-- Prove the sum and product
theorem sum_and_product_of_three_numbers : a + b + c = 67 ∧ a * b * c = 9600 :=
by {
  sorry
}

end sum_and_product_of_three_numbers_l221_221140


namespace sum_of_consecutive_even_numbers_l221_221139

theorem sum_of_consecutive_even_numbers (n : ℕ) (h : (n + 2)^2 - n^2 = 84) :
  n + (n + 2) = 42 :=
sorry

end sum_of_consecutive_even_numbers_l221_221139


namespace max_subway_riders_l221_221751

theorem max_subway_riders:
  ∃ (P F : ℕ), P + F = 251 ∧ (1 / 11) * P + (1 / 13) * F = 22 := sorry

end max_subway_riders_l221_221751


namespace truck_left_1_hour_later_l221_221314

theorem truck_left_1_hour_later (v_car v_truck : ℝ) (time_to_pass : ℝ) : 
  v_car = 55 ∧ v_truck = 65 ∧ time_to_pass = 6.5 → 
  1 = time_to_pass - (time_to_pass * (v_car / v_truck)) := 
by
  intros h
  sorry

end truck_left_1_hour_later_l221_221314


namespace more_girls_than_boys_l221_221890

variables (boys girls : ℕ)

def ratio_condition : Prop := (3 * girls = 4 * boys)
def total_students_condition : Prop := (boys + girls = 42)

theorem more_girls_than_boys (h1 : ratio_condition boys girls) (h2 : total_students_condition boys girls) :
  (girls - boys = 6) :=
sorry

end more_girls_than_boys_l221_221890


namespace find_p_q_l221_221566

theorem find_p_q (p q : ℤ) (h : ∀ x : ℤ, (x - 5) * (x + 2) = x^2 + p * x + q) :
  p = -3 ∧ q = -10 :=
by {
  -- The proof would go here, but for now we'll use sorry to indicate it's incomplete.
  sorry
}

end find_p_q_l221_221566


namespace area_of_common_part_geq_3484_l221_221331

theorem area_of_common_part_geq_3484 :
  ∀ (R : ℝ) (S T : ℝ → Prop), 
  (R = 1) →
  (∀ x y, S x ↔ (x * x + y * y = R * R) ∧ T y) →
  ∃ (S_common : ℝ) (T_common : ℝ),
    (S_common + T_common > 3.484) :=
by
  sorry

end area_of_common_part_geq_3484_l221_221331


namespace balls_in_boxes_l221_221402

/-- Prove that the number of ways to put 6 distinguishable balls in 3 distinguishable boxes is 729 (which is 3^6). -/
theorem balls_in_boxes : (3 ^ 6) = 729 :=
by 
  sorry

end balls_in_boxes_l221_221402


namespace expected_value_max_l221_221701

def E_max_x_y_z (x y z : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 10) : ℚ :=
  (4 * (1/6) + 5 * (1/3) + 6 * (1/4) + 7 * (1/6) + 8 * (1/12))

theorem expected_value_max (x y z : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 10) :
  E_max_x_y_z x y z h1 h2 h3 h4 = 17 / 3 := 
sorry

end expected_value_max_l221_221701


namespace greatest_integer_l221_221133

theorem greatest_integer (y : ℤ) (h : (8 : ℚ) / 11 > y / 17) : y ≤ 12 :=
by
  have h₁ : (8 : ℚ) / 11 * 17 > y := by exact (div_mul_cancel _ (by norm_num : 17 ≠ 0))
  have h₂ : 136 / 11 > y := by rwa mul_comm _ 17 at h₁
  exact_mod_cast le_of_lt h₂

end greatest_integer_l221_221133


namespace blue_sequins_per_row_l221_221756

theorem blue_sequins_per_row : 
  ∀ (B : ℕ),
  (6 * B) + (5 * 12) + (9 * 6) = 162 → B = 8 :=
by
  intro B
  sorry

end blue_sequins_per_row_l221_221756


namespace evaluate_expression_at_3_l221_221300

-- Define the expression
def expression (x : ℕ) : ℕ := x^2 - 3*x + 2

-- Statement of the problem
theorem evaluate_expression_at_3 : expression 3 = 2 := by
    sorry -- Proof is omitted

end evaluate_expression_at_3_l221_221300


namespace total_area_of_rectangles_l221_221342

/-- The combined area of two adjacent rectangular regions given their conditions -/
theorem total_area_of_rectangles (u v w z : ℝ) 
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) (hz : w < z) : 
  (u + v) * z = (u + v) * w + (u + v) * (z - w) :=
by
  sorry

end total_area_of_rectangles_l221_221342


namespace smallest_value_expression_l221_221985

theorem smallest_value_expression
    (a b c : ℝ) 
    (h1 : c > b)
    (h2 : b > a)
    (h3 : c ≠ 0) : 
    ∃ z : ℝ, z = 0 ∧ z = (a + b)^2 / c^2 + (b - c)^2 / c^2 + (c - b)^2 / c^2 :=
by
  sorry

end smallest_value_expression_l221_221985


namespace berries_count_l221_221821

theorem berries_count (total_berries : ℕ)
  (h1 : total_berries = 42)
  (h2 : total_berries / 2 = 21)
  (h3 : total_berries / 3 = 14) :
  total_berries - (total_berries / 2 + total_berries / 3) = 7 :=
by
  rw [h1, h2, h3]
  norm_num
  exact rfl

end berries_count_l221_221821


namespace find_number_l221_221822

theorem find_number (x : ℝ) : 
  0.05 * x = 0.20 * 650 + 190 → x = 6400 :=
by
  intro h
  sorry

end find_number_l221_221822


namespace num_baskets_l221_221643

axiom num_apples_each_basket : ℕ
axiom total_apples : ℕ

theorem num_baskets (h1 : num_apples_each_basket = 17) (h2 : total_apples = 629) : total_apples / num_apples_each_basket = 37 :=
  sorry

end num_baskets_l221_221643


namespace volume_of_solid_of_revolution_l221_221250

noncomputable def piecewise_f (x : ℝ) : ℝ :=
  if x < 0 then real.sqrt (4 - x ^ 2) else 2 - x

theorem volume_of_solid_of_revolution :
  let f := piecewise_f in
  ∫ x in -2..2, π * (f x) ^ 2 = 8 * π :=
by
  sorry

end volume_of_solid_of_revolution_l221_221250


namespace intersection_M_P_l221_221769

variable {x a : ℝ}

def M (a : ℝ) : Set ℝ := { x | x > a ∧ a^2 - 12*a + 20 < 0 }
def P : Set ℝ := { x | x ≤ 10 }

theorem intersection_M_P (a : ℝ) (h : 2 < a ∧ a < 10) : 
  M a ∩ P = { x | a < x ∧ x ≤ 10 } :=
sorry

end intersection_M_P_l221_221769


namespace regular_ticket_price_l221_221121

variable (P : ℝ) -- Define the regular ticket price as a real number

-- Condition: Travis pays $1400 for his ticket after a 30% discount on a regular price P
axiom h : 0.70 * P = 1400

-- Theorem statement: Proving that the regular ticket price P equals $2000
theorem regular_ticket_price : P = 2000 :=
by 
  sorry

end regular_ticket_price_l221_221121


namespace problem_statement_l221_221993

def f (x : ℝ) : ℝ := x^2 - 3 * x + 4
def g (x : ℝ) : ℝ := x - 2

theorem problem_statement : f (g 5) - g (f 5) = -8 := by sorry

end problem_statement_l221_221993


namespace find_number_l221_221505

def initial_condition (x : ℝ) : Prop :=
  ((x + 7) * 3 - 12) / 6 = -8

theorem find_number (x : ℝ) (h : initial_condition x) : x = -19 := by
  sorry

end find_number_l221_221505


namespace socks_ratio_l221_221519

-- Definitions based on the conditions
def initial_black_socks : ℕ := 6
def initial_white_socks (B : ℕ) : ℕ := 4 * B
def remaining_white_socks (B : ℕ) : ℕ := B + 6

-- The theorem to prove the ratio is 1/2
theorem socks_ratio (B : ℕ) (hB : B = initial_black_socks) :
  ((initial_white_socks B - remaining_white_socks B) : ℚ) / initial_white_socks B = 1 / 2 :=
by
  sorry

end socks_ratio_l221_221519


namespace general_term_of_sequence_l221_221030

theorem general_term_of_sequence
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_pos_a : ∀ n, 0 < a n)
  (h_pos_b : ∀ n, 0 < b n)
  (h_arith : ∀ n, 2 * b n = a n + a (n + 1))
  (h_geom : ∀ n, (a (n + 1))^2 = b n * b (n + 1))
  (h_a1 : a 1 = 1)
  (h_a2 : a 2 = 3)
  : ∀ n, a n = (n^2 + n) / 2 :=
by
  sorry

end general_term_of_sequence_l221_221030


namespace largest_int_lt_100_div_9_rem_5_l221_221541

theorem largest_int_lt_100_div_9_rem_5 :
  ∃ a, a < 100 ∧ (a % 9 = 5) ∧ ∀ b, b < 100 ∧ (b % 9 = 5) → b ≤ 95 := by
sorry

end largest_int_lt_100_div_9_rem_5_l221_221541


namespace degree_to_radian_l221_221528

theorem degree_to_radian : (855 : ℝ) * (Real.pi / 180) = (59 / 12) * Real.pi :=
by
  sorry

end degree_to_radian_l221_221528


namespace percent_problem_l221_221883

variable (x : ℝ)

theorem percent_problem (h : 0.30 * 0.15 * x = 27) : 0.15 * 0.30 * x = 27 :=
by sorry

end percent_problem_l221_221883


namespace ball_hits_ground_l221_221468

noncomputable def ball_height (t : ℝ) : ℝ := -9 * t^2 + 15 * t + 72

theorem ball_hits_ground :
  (∃ t : ℝ, t = (5 + Real.sqrt 313) / 6 ∧ ball_height t = 0) :=
sorry

end ball_hits_ground_l221_221468


namespace probability_of_snowing_at_least_once_l221_221479

theorem probability_of_snowing_at_least_once (p : ℚ) (h : p = 3 / 4) :
  let q := 1 - p in
  let not_snowing_five_days := q ^ 5 in
  let at_least_once := 1 - not_snowing_five_days in
  at_least_once = 1023 / 1024 :=
by
  have q_def : q = 1 - p := rfl,
  have not_snowing_five_days_def : not_snowing_five_days = q ^ 5 := rfl,
  have at_least_once_def : at_least_once = 1 - not_snowing_five_days := rfl,
  sorry

end probability_of_snowing_at_least_once_l221_221479


namespace increase_in_average_l221_221150

variable (A : ℝ)
variable (new_avg : ℝ := 44)
variable (score_12th_inning : ℝ := 55)
variable (total_runs_after_11 : ℝ := 11 * A)

theorem increase_in_average :
  ((total_runs_after_11 + score_12th_inning) / 12 - A = 1) :=
by
  sorry

end increase_in_average_l221_221150


namespace determinant_expression_l221_221535

noncomputable def matrixDet (α β : ℝ) : ℝ :=
  Matrix.det ![
    ![Real.sin α * Real.cos β, -Real.sin α * Real.sin β, Real.cos α],
    ![-Real.sin β, -Real.cos β, 0],
    ![Real.cos α * Real.cos β, Real.cos α * Real.sin β, Real.sin α]]

theorem determinant_expression (α β: ℝ) : matrixDet α β = Real.sin α ^ 3 := 
by 
  sorry

end determinant_expression_l221_221535


namespace no_b_satisfies_condition_l221_221854

noncomputable def f (b x : ℝ) : ℝ :=
  x^2 + 3 * b * x + 5 * b

theorem no_b_satisfies_condition :
  ∀ b : ℝ, ¬ (∃ x : ℝ, ∀ y : ℝ, |f b y| ≤ 5 → y = x) :=
by
  sorry

end no_b_satisfies_condition_l221_221854


namespace cookout_kids_2006_l221_221051

theorem cookout_kids_2006 :
  let kids_2004 := 60
  let kids_2005 := kids_2004 / 2
  let kids_2006 := (kids_2005 / 3) * 2
  in kids_2006 = 20 :=
by
  let kids_2004 := 60
  let kids_2005 := kids_2004 / 2
  let kids_2006 := (kids_2005 / 3) * 2
  have h : kids_2006 = 20 := sorry
  exact h

end cookout_kids_2006_l221_221051


namespace distinct_powers_exist_l221_221327

theorem distinct_powers_exist :
  ∃ (a1 a2 b1 b2 c1 c2 d1 d2 : ℕ),
    (∃ n, a1 = n^2) ∧ (∃ m, a2 = m^2) ∧
    (∃ p, b1 = p^3) ∧ (∃ q, b2 = q^3) ∧
    (∃ r, c1 = r^5) ∧ (∃ s, c2 = s^5) ∧
    (∃ t, d1 = t^7) ∧ (∃ u, d2 = u^7) ∧
    a1 - a2 = b1 - b2 ∧ b1 - b2 = c1 - c2 ∧ c1 - c2 = d1 - d2 ∧
    a1 ≠ b1 ∧ a1 ≠ c1 ∧ a1 ≠ d1 ∧ b1 ≠ c1 ∧ b1 ≠ d1 ∧ c1 ≠ d1 := 
sorry

end distinct_powers_exist_l221_221327


namespace min_value_expression_l221_221356

theorem min_value_expression (x : ℝ) (hx : x > 0) : 9 * x + 1 / x^3 ≥ 10 :=
sorry

end min_value_expression_l221_221356


namespace smallest_sum_l221_221377

theorem smallest_sum (x y : ℕ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x ≠ y) (eq : 1/x + 1/y = 1/10) : x + y = 45 :=
by
  sorry

end smallest_sum_l221_221377


namespace smallest_sum_of_xy_l221_221374

theorem smallest_sum_of_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y)
  (hcond : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 10) : x + y = 45 :=
sorry

end smallest_sum_of_xy_l221_221374


namespace find_WZ_length_l221_221417

noncomputable def WZ_length (XY YZ XZ WX : ℝ) (theta : ℝ) : ℝ :=
  Real.sqrt ((WX^2 + XZ^2 - 2 * WX * XZ * (-1 / 2)))

-- Define the problem within the context of the provided lengths and condition
theorem find_WZ_length :
  WZ_length 3 5 7 8.5 (-1 / 2) = Real.sqrt 180.75 :=
by 
  -- This "by sorry" is used to indicate the proof is omitted
  sorry

end find_WZ_length_l221_221417


namespace triangle_area_is_24_l221_221125

-- Defining the vertices of the triangle
def A := (2, 2)
def B := (8, 2)
def C := (4, 10)

-- Calculate the area of the triangle
def area_of_triangle (A B C : ℕ × ℕ) : ℕ := 
  let base := |B.1 - A.1| 
  let height := |C.2 - A.2| 
  ((base * height) / 2)

-- Statement to prove
theorem triangle_area_is_24 : area_of_triangle A B C = 24 := 
by
  sorry

end triangle_area_is_24_l221_221125


namespace component_unqualified_l221_221317

theorem component_unqualified :
  ∀ (φ : ℝ), (19.98 ≤ φ ∧ φ ≤ 20.02) → ¬(φ = 19.9) → True :=
by
  intro φ
  intro h
  intro h'
  -- skip proof
  sorry

end component_unqualified_l221_221317


namespace each_child_play_time_l221_221684

theorem each_child_play_time (n_children : ℕ) (game_time : ℕ) (children_per_game : ℕ)
  (h1 : n_children = 8) (h2 : game_time = 120) (h3 : children_per_game = 2) :
  ((children_per_game * game_time) / n_children) = 30 :=
  sorry

end each_child_play_time_l221_221684


namespace box_volume_l221_221161

theorem box_volume (a b c : ℝ) (H1 : a * b = 15) (H2 : b * c = 10) (H3 : c * a = 6) : a * b * c = 30 := 
sorry

end box_volume_l221_221161


namespace unique_solution_integer_equation_l221_221971

theorem unique_solution_integer_equation : 
  ∃! (x y z : ℤ), x^2 + y^2 + z^2 = x^2 * y^2 :=
by sorry

end unique_solution_integer_equation_l221_221971


namespace intersection_A_B_l221_221206

def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℝ := {x | abs (x - 1) ≤ 1}

theorem intersection_A_B : A ∩ B = {1, 2} :=
  sorry

end intersection_A_B_l221_221206


namespace sum_abs_values_l221_221232

theorem sum_abs_values (a b : ℝ) (h₁ : abs a = 4) (h₂ : abs b = 7) (h₃ : a < b) : a + b = 3 ∨ a + b = 11 :=
by
  sorry

end sum_abs_values_l221_221232


namespace range_of_a_plus_b_l221_221992

variable {a b : ℝ}

-- Assumptions
def are_positive_and_unequal (a b : ℝ) : Prop := a > 0 ∧ b > 0 ∧ a ≠ b
def equation_holds (a b : ℝ) : Prop := a^2 - a + b^2 - b + a * b = 0

-- Problem Statement
theorem range_of_a_plus_b (h₁ : are_positive_and_unequal a b) (h₂ : equation_holds a b) : 1 < a + b ∧ a + b < 4 / 3 :=
sorry

end range_of_a_plus_b_l221_221992


namespace stick_horisontal_fall_position_l221_221564

-- Definitions based on the conditions
def stick_length : ℝ := 120 -- length of the stick in cm
def projection_distance : ℝ := 70 -- distance between projections of the ends of the stick on the floor

-- The main theorem to prove
theorem stick_horisontal_fall_position :
  ∀ (L d : ℝ), L = stick_length ∧ d = projection_distance → 
  ∃ x : ℝ, x = 25 :=
by
  intros L d h
  have h1 : L = stick_length := h.1
  have h2 : d = projection_distance := h.2
  -- The detailed proof steps will be here
  sorry

end stick_horisontal_fall_position_l221_221564


namespace linear_function_difference_l221_221088

noncomputable def linear_function (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x : ℝ, f x = m * x + b

theorem linear_function_difference (f : ℝ → ℝ) 
  (h_linear : linear_function f)
  (h_cond1 : f 10 - f 5 = 20)
  (h_cond2 : f 0 = 3) :
  f 15 - f 5 = 40 :=
sorry

end linear_function_difference_l221_221088


namespace least_faces_triangular_pyramid_l221_221642

def triangular_prism_faces : ℕ := 5
def quadrangular_prism_faces : ℕ := 6
def triangular_pyramid_faces : ℕ := 4
def quadrangular_pyramid_faces : ℕ := 5
def truncated_quadrangular_pyramid_faces : ℕ := 5 -- assuming the minimum possible value

theorem least_faces_triangular_pyramid :
  triangular_pyramid_faces < triangular_prism_faces ∧
  triangular_pyramid_faces < quadrangular_prism_faces ∧
  triangular_pyramid_faces < quadrangular_pyramid_faces ∧
  triangular_pyramid_faces ≤ truncated_quadrangular_pyramid_faces :=
by
  sorry

end least_faces_triangular_pyramid_l221_221642


namespace sum_of_abc_is_12_l221_221595

theorem sum_of_abc_is_12 (a b c : ℕ) (h : 28 * a + 30 * b + 31 * c = 365) : a + b + c = 12 :=
by
  sorry

end sum_of_abc_is_12_l221_221595


namespace expression_not_equal_l221_221493

variable (a b c : ℝ)

theorem expression_not_equal :
  (a - (b - c)) ≠ (a - b - c) :=
by sorry

end expression_not_equal_l221_221493


namespace complex_exp_cos_l221_221367

theorem complex_exp_cos (z : ℂ) (α : ℂ) (n : ℕ) (h : z + z⁻¹ = 2 * Complex.cos α) : 
  z^n + z⁻¹^n = 2 * Complex.cos (n * α) :=
by
  sorry

end complex_exp_cos_l221_221367


namespace probability_first_card_heart_second_king_l221_221935

theorem probability_first_card_heart_second_king :
  ∀ (deck : Finset ℕ) (is_heart : ℕ → Prop) (is_king : ℕ → Prop),
  deck.card = 52 →
  (∀ card ∈ deck, is_heart card ∨ ¬ is_heart card) →
  (∀ card ∈ deck, is_king card ∨ ¬ is_king card) →
  (∃ p : ℚ, p = 1/52) :=
by
  intros deck is_heart is_king h_card h_heart h_king,
  sorry

end probability_first_card_heart_second_king_l221_221935


namespace ratio_of_speeds_l221_221811

-- Define the speeds V1 and V2
variable {V1 V2 : ℝ}

-- Given the initial conditions
def bike_ride_time_min := 10 -- in minutes
def subway_ride_time_min := 40 -- in minutes
def total_bike_only_time_min := 210 -- 3.5 hours in minutes

-- Prove the ratio of subway speed to bike speed is 5:1
theorem ratio_of_speeds (h : bike_ride_time_min * V1 + subway_ride_time_min * V2 = total_bike_only_time_min * V1) :
  V2 = 5 * V1 :=
by
  sorry

end ratio_of_speeds_l221_221811


namespace jordan_rectangle_length_l221_221174

variables (L : ℝ)

-- Condition: Carol's rectangle measures 12 inches by 15 inches.
def carol_area : ℝ := 12 * 15

-- Condition: Jordan's rectangle has the same area as Carol's rectangle.
def jordan_area : ℝ := carol_area

-- Condition: Jordan's rectangle is 20 inches wide.
def jordan_width : ℝ := 20

-- Proposition: Length of Jordan's rectangle == 9 inches.
theorem jordan_rectangle_length : L * jordan_width = jordan_area → L = 9 := 
by
  intros h
  sorry

end jordan_rectangle_length_l221_221174


namespace plane_divided_by_n_lines_l221_221802

-- Definition of the number of regions created by n lines in a plane
def regions (n : ℕ) : ℕ :=
  if n = 0 then 1 else (n * (n + 1)) / 2 + 1 -- Using the given formula directly

-- Theorem statement to prove the formula holds
theorem plane_divided_by_n_lines (n : ℕ) : 
  regions n = (n * (n + 1)) / 2 + 1 :=
sorry

end plane_divided_by_n_lines_l221_221802


namespace johns_age_is_25_l221_221900

variable (JohnAge DadAge SisterAge : ℕ)

theorem johns_age_is_25
    (h1 : JohnAge = DadAge - 30)
    (h2 : JohnAge + DadAge = 80)
    (h3 : SisterAge = JohnAge - 5) :
    JohnAge = 25 := 
sorry

end johns_age_is_25_l221_221900


namespace probability_of_die_showing_1_after_5_steps_l221_221180

def prob_showing_1 (steps : ℕ) : ℚ :=
  if steps = 5 then 37 / 192 else 0

theorem probability_of_die_showing_1_after_5_steps :
  prob_showing_1 5 = 37 / 192 :=
sorry

end probability_of_die_showing_1_after_5_steps_l221_221180


namespace base9_digit_divisible_by_13_l221_221698

theorem base9_digit_divisible_by_13 :
    ∃ (d : ℕ), (0 ≤ d ∧ d ≤ 8) ∧ (13 ∣ (2 * 9^4 + d * 9^3 + 6 * 9^2 + d * 9 + 4)) :=
by
  sorry

end base9_digit_divisible_by_13_l221_221698


namespace stack_height_difference_l221_221289

theorem stack_height_difference :
  ∃ S : ℕ,
    (7 + S + (S - 6) + (S + 4) + 2 * S = 55) ∧ (S - 7 = 3) := 
by 
  sorry

end stack_height_difference_l221_221289


namespace find_integer_values_of_a_l221_221531

theorem find_integer_values_of_a
  (x a b c : ℤ)
  (h : (x - a) * (x - 10) + 5 = (x + b) * (x + c)) :
  a = 4 ∨ a = 16 := by
    sorry

end find_integer_values_of_a_l221_221531


namespace probability_closer_to_eight_l221_221159

noncomputable def probability_point_closer_to_eight (x : ℝ) : ℚ :=
if 0 ≤ x ∧ x ≤ 8 then 
  if x > 4 then 1 else 0
else 0

theorem probability_closer_to_eight : 
  (∫ x in 0..8, probability_point_closer_to_eight x) / ∫ x in 0..8, 1 = (1 : ℚ) / 2 :=
sorry

end probability_closer_to_eight_l221_221159


namespace box_volume_l221_221301

theorem box_volume (l w h V : ℝ) 
  (h1 : l * w = 30) 
  (h2 : w * h = 18) 
  (h3 : l * h = 10) 
  : V = l * w * h → V = 90 :=
by 
  intro volume_eq
  sorry

end box_volume_l221_221301


namespace compound_interest_second_year_l221_221921

theorem compound_interest_second_year
  (P : ℝ) (r : ℝ) (CI_3 : ℝ) (CI_2 : ℝ) 
  (h1 : r = 0.08) 
  (h2 : CI_3 = 1512)
  (h3 : CI_3 = CI_2 * (1 + r)) :
  CI_2 = 1400 :=
by
  rw [h1, h2] at h3
  sorry

end compound_interest_second_year_l221_221921


namespace inequality_solution_l221_221265

theorem inequality_solution (x : ℝ) 
  (hx1 : x ≠ 1) 
  (hx2 : x ≠ 2) 
  (hx3 : x ≠ 3) 
  (hx4 : x ≠ 4) :
  (1 / (x - 1) - 3 / (x - 2) + 5 / (x - 3) - 1 / (x - 4) < 1 / 24) ↔ (x ∈ Set.Ioo (-7 : ℝ) 1 ∪ Set.Ioo 3 4) := 
sorry

end inequality_solution_l221_221265


namespace common_ratio_of_geometric_series_l221_221962

noncomputable def geometric_series_common_ratio (a S : ℝ) : ℝ := 1 - (a / S)

theorem common_ratio_of_geometric_series :
  geometric_series_common_ratio 520 3250 = 273 / 325 :=
by
  sorry

end common_ratio_of_geometric_series_l221_221962


namespace add_in_base14_l221_221958

-- Define symbols A, B, C, D in base 10 as they are used in the base 14 representation
def base14_A : ℕ := 10
def base14_B : ℕ := 11
def base14_C : ℕ := 12
def base14_D : ℕ := 13

-- Define the numbers given in base 14
def num1_base14 : ℕ := 9 * 14^2 + base14_C * 14 + 7
def num2_base14 : ℕ := 4 * 14^2 + base14_B * 14 + 3

-- Define the expected result in base 14
def result_base14 : ℕ := 1 * 14^2 + 0 * 14 + base14_A

-- The theorem statement that needs to be proven
theorem add_in_base14 : num1_base14 + num2_base14 = result_base14 := by
  sorry

end add_in_base14_l221_221958


namespace two_layers_area_zero_l221_221290

theorem two_layers_area_zero (A X Y Z : ℕ)
  (h1 : A = 212)
  (h2 : X + Y + Z = 140)
  (h3 : Y + Z = 24)
  (h4 : Z = 24) : Y = 0 :=
by
  sorry

end two_layers_area_zero_l221_221290


namespace find_angle_A_l221_221749

theorem find_angle_A (A B : ℝ) (a b : ℝ) (h1 : b = 2 * a * Real.sin B) (h2 : a ≠ 0) :
  A = 30 ∨ A = 150 :=
by
  sorry

end find_angle_A_l221_221749


namespace find_x_l221_221609

-- Definitions corresponding to conditions a)
def rectangle (AB CD BC AD x : ℝ) := AB = 2 ∧ CD = 2 ∧ BC = 1 ∧ AD = 1 ∧ x = 0

-- Define the main statement to be proven
theorem find_x (AB CD BC AD x k m: ℝ) (h: rectangle AB CD BC AD x) : 
  x = (0 : ℝ) ∧ k = 0 ∧ m = 0 ∧ x = (Real.sqrt k - m) ∧ k + m = 0 :=
by
  cases h
  sorry

end find_x_l221_221609


namespace tan_a6_of_arithmetic_sequence_l221_221881

noncomputable def arithmetic_sequence (a : ℕ → ℝ) := 
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) := 
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

theorem tan_a6_of_arithmetic_sequence
  (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (H1 : arithmetic_sequence a)
  (H2 : sum_of_first_n_terms a S)
  (H3 : S 11 = 22 * Real.pi / 3) : 
  Real.tan (a 6) = -Real.sqrt 3 :=
sorry

end tan_a6_of_arithmetic_sequence_l221_221881


namespace find_a_plus_b_l221_221784

-- Define the constants and conditions
variables (a b c : ℤ)
variables (a_cond : 0 ≤ a ∧ a < 5) (b_cond : 0 ≤ b ∧ b < 13)
variables (frac_decomp : (1 : ℚ) / 2015 = (a : ℚ) / 5 + (b : ℚ) / 13 + (c : ℚ) / 31)

-- State the theorem
theorem find_a_plus_b (a b c : ℤ) (a_cond : 0 ≤ a ∧ a < 5) (b_cond : 0 ≤ b ∧ b < 13) (frac_decomp : (1 : ℚ) / 2015 = (a : ℚ) / 5 + (b : ℚ) / 13 + (c : ℚ) / 31) :
  a + b = 14 := 
sorry

end find_a_plus_b_l221_221784


namespace ratio_problem_l221_221743

theorem ratio_problem (A B C : ℚ) (h : A / B = 3 / 2) (h' : B / C = 2 / 5) : (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := 
by
  sorry

end ratio_problem_l221_221743


namespace evaluate_expression_l221_221693

theorem evaluate_expression :
  -2 ^ 2005 + (-2) ^ 2006 + 2 ^ 2007 - 2 ^ 2008 = 2 ^ 2005 :=
by
  -- The following proof is left as an exercise.
  sorry

end evaluate_expression_l221_221693


namespace sequence_property_l221_221571

theorem sequence_property (n : ℕ) (hn : 1 ≤ n ∧ n ≤ 9) : 7 * n * 15873 = n * 111111 :=
by sorry

end sequence_property_l221_221571


namespace expected_intersections_100gon_l221_221063

noncomputable def expected_intersections : ℝ :=
  let n := 100
  let total_pairs := (n * (n - 3) / 2)
  total_pairs * (1/3)

theorem expected_intersections_100gon :
  expected_intersections = 4850 / 3 :=
by
  sorry

end expected_intersections_100gon_l221_221063


namespace triangle_inequality_sum_2_l221_221335

theorem triangle_inequality_sum_2 (a b c : ℝ) (h_triangle : a + b + c = 2) (h_side_ineq : a + c > b ∧ a + b > c ∧ b + c > a):
  1 ≤ a * b + b * c + c * a - a * b * c ∧ a * b + b * c + c * a - a * b * c ≤ 1 + 1 / 27 :=
by
  sorry

end triangle_inequality_sum_2_l221_221335


namespace probability_of_first_heart_second_king_l221_221938

noncomputable def probability_first_heart_second_king : ℚ :=
  1 / 52 * 3 / 51 + 12 / 52 * 4 / 51

theorem probability_of_first_heart_second_king :
  probability_first_heart_second_king = 1 / 52 :=
by
  sorry

end probability_of_first_heart_second_king_l221_221938


namespace curve_cartesian_equation_chord_length_l221_221584
noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * θ.cos, ρ * θ.sin)

noncomputable def line_parametric (t : ℝ) : ℝ × ℝ :=
  (2 + 1/2 * t, (Real.sqrt 3) / 2 * t)

theorem curve_cartesian_equation :
  ∀ (ρ θ : ℝ), 
    ρ * θ.sin * θ.sin = 8 * θ.cos →
    (ρ * θ.cos) ^ 2 + (ρ * θ.sin) ^ 2 = 
    8 * (ρ * θ.cos) :=
by sorry

theorem chord_length :
  ∀ (t₁ t₂ : ℝ),
    (3 * t₁^2 - 16 * t₁ - 64 = 0) →
    (3 * t₂^2 - 16 * t₂ - 64 = 0) →
    |t₁ - t₂| = (32 / 3) :=
by sorry

end curve_cartesian_equation_chord_length_l221_221584


namespace symmetric_points_sum_l221_221996

theorem symmetric_points_sum
  (a b : ℝ)
  (h1 : a = -3)
  (h2 : b = 2) :
  a + b = -1 := by
  sorry

end symmetric_points_sum_l221_221996


namespace proof_problem_l221_221098

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)

-- Definition for statement 1
def statement1 := f 0 = 0

-- Definition for statement 2
def statement2 := (∃ x > 0, ∀ y > 0, f x ≥ f y) → (∃ x < 0, ∀ y < 0, f x ≤ f y)

-- Definition for statement 3
def statement3 := (∀ x ≥ 1, ∀ y ≥ 1, x < y → f x < f y) → (∀ x ≤ -1, ∀ y ≤ -1, x < y → f y < f x)

-- Definition for statement 4
def statement4 := (∀ x > 0, f x = x^2 - 2 * x) → (∀ x < 0, f x = -x^2 - 2 * x)

-- Combined proof problem
theorem proof_problem :
  (statement1 f) ∧ (statement2 f) ∧ (statement4 f) ∧ ¬ (statement3 f) :=
by sorry

end proof_problem_l221_221098


namespace prove_f_three_eq_neg_three_l221_221573

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * Real.sin (2 * x) + b * Real.tan x + 1

theorem prove_f_three_eq_neg_three (a b : ℝ) (h : f (-3) a b = 5) : f 3 a b = -3 := by
  sorry

end prove_f_three_eq_neg_three_l221_221573


namespace ratio_of_administrators_to_teachers_l221_221415

-- Define the conditions
def graduates : ℕ := 50
def parents_per_graduate : ℕ := 2
def teachers : ℕ := 20
def total_chairs : ℕ := 180

-- Calculate intermediate values
def parents : ℕ := graduates * parents_per_graduate
def graduates_and_parents_chairs : ℕ := graduates + parents
def total_graduates_parents_teachers_chairs : ℕ := graduates_and_parents_chairs + teachers
def administrators : ℕ := total_chairs - total_graduates_parents_teachers_chairs

-- Specify the theorem to prove the ratio of administrators to teachers
theorem ratio_of_administrators_to_teachers : administrators / teachers = 1 / 2 :=
by
  -- Proof is omitted; placeholder 'sorry'
  sorry

end ratio_of_administrators_to_teachers_l221_221415


namespace least_possible_students_l221_221447

def TotalNumberOfStudents : ℕ := 35
def NumberOfStudentsWithBrownEyes : ℕ := 15
def NumberOfStudentsWithLunchBoxes : ℕ := 25
def NumberOfStudentsWearingGlasses : ℕ := 10

theorem least_possible_students (TotalNumberOfStudents NumberOfStudentsWithBrownEyes NumberOfStudentsWithLunchBoxes NumberOfStudentsWearingGlasses : ℕ) :
  ∃ n, n = 5 :=
sorry

end least_possible_students_l221_221447


namespace snow_probability_at_least_once_l221_221474

theorem snow_probability_at_least_once :
  let p := 3 / 4
  let prob_no_snow_single_day := 1 - p
  let prob_no_snow_all_days := prob_no_snow_single_day ^ 5
  let prob_snow_at_least_once := 1 - prob_no_snow_all_days
  prob_snow_at_least_once = 1023 / 1024 :=
by
  sorry

end snow_probability_at_least_once_l221_221474


namespace tangential_quadrilateral_perpendicular_diagonals_l221_221198

-- Define what it means for a quadrilateral to be tangential
def is_tangential_quadrilateral (a b c d : ℝ) : Prop :=
  a + c = b + d

-- Define what it means for a quadrilateral to be a kite
def is_kite (a b c d : ℝ) : Prop :=
  a = b ∧ c = d

-- Define what it means for the diagonals of a quadrilateral to be perpendicular
def diagonals_perpendicular (a b c d : ℝ) : Prop :=
  sorry -- Actual geometric definition needs to be elaborated

-- Main statement to prove
theorem tangential_quadrilateral_perpendicular_diagonals (a b c d : ℝ) :
  is_tangential_quadrilateral a b c d → 
  (diagonals_perpendicular a b c d ↔ is_kite a b c d) := 
sorry

end tangential_quadrilateral_perpendicular_diagonals_l221_221198


namespace exponent_equality_l221_221226

theorem exponent_equality (n : ℕ) : (4^8 = 4^n) → (n = 8) := by
  intro h
  sorry

end exponent_equality_l221_221226


namespace value_of_y_l221_221691

theorem value_of_y (x y : ℤ) (h1 : 1.5 * (x : ℝ) = 0.25 * (y : ℝ)) (h2 : x = 24) : y = 144 :=
  sorry

end value_of_y_l221_221691


namespace john_pays_per_year_l221_221424

-- Define the costs and insurance parameters.
def cost_per_epipen : ℝ := 500
def insurance_coverage : ℝ := 0.75

-- Number of months in a year.
def months_in_year : ℕ := 12

-- Number of months each EpiPen lasts.
def months_per_epipen : ℕ := 6

-- Amount covered by insurance for each EpiPen.
def insurance_amount (cost : ℝ) (coverage : ℝ) : ℝ :=
  cost * coverage

-- Amount John pays after insurance for each EpiPen.
def amount_john_pays_per_epipen (cost : ℝ) (covered: ℝ) : ℝ :=
  cost - covered

-- Number of EpiPens John needs per year.
def epipens_per_year (months_in_year : ℕ) (months_per_epipen : ℕ) : ℕ :=
  months_in_year / months_per_epipen

-- Total amount John pays per year.
def total_amount_john_pays_per_year (amount_per_epipen : ℝ) (epipens_per_year : ℕ) : ℝ :=
  amount_per_epipen * epipens_per_year

-- Theorem to prove the correct answer.
theorem john_pays_per_year :
  total_amount_john_pays_per_year (amount_john_pays_per_epipen cost_per_epipen (insurance_amount cost_per_epipen insurance_coverage)) (epipens_per_year months_in_year months_per_epipen) = 250 := 
by
  sorry

end john_pays_per_year_l221_221424


namespace sum_first_19_terms_l221_221369

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (d : ℝ)
variable (a₀ a₃ a₁₇ a₁₀ : ℝ)

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ a₀ d, ∀ n, a n = a₀ + n * d

noncomputable def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))

theorem sum_first_19_terms (h1 : is_arithmetic_sequence a)
                          (h2 : a 3 + a 17 = 10)
                          (h3 : sum_first_n_terms S a) :
  S 19 = 95 :=
sorry

end sum_first_19_terms_l221_221369


namespace ratio_problem_l221_221744

theorem ratio_problem (A B C : ℚ) (h : A / B = 3 / 2) (h' : B / C = 2 / 5) : (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := 
by
  sorry

end ratio_problem_l221_221744


namespace monotonicity_intervals_number_of_zeros_l221_221444

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - k / 2 * x^2

theorem monotonicity_intervals (k : ℝ) :
  (k ≤ 0 → (∀ x, x < 0 → f k x < 0) ∧ (∀ x, x ≥ 0 → f k x > 0)) ∧
  (0 < k ∧ k < 1 → 
    (∀ x, x < Real.log k → f k x < 0) ∧ (∀ x, x ≥ Real.log k ∧ x < 0 → f k x > 0) ∧ 
    (∀ x, x > 0 → f k x > 0)) ∧
  (k = 1 → ∀ x, f k x > 0) ∧
  (k > 1 → 
    (∀ x, x < 0 → f k x < 0) ∧ 
    (∀ x, x ≥ 0 ∧ x < Real.log k → f k x > 0) ∧ 
    (∀ x, x > Real.log k → f k x > 0)) :=
sorry

theorem number_of_zeros (k : ℝ) (h_nonpos : k ≤ 0) :
  (k < 0 → (∃ a b : ℝ, a < 0 ∧ b > 0 ∧ f k a = 0 ∧ f k b = 0)) ∧
  (k = 0 → f k 1 = 0 ∧ (∀ x, x ≠ 1 → f k x ≠ 0)) :=
sorry

end monotonicity_intervals_number_of_zeros_l221_221444


namespace balls_in_boxes_l221_221401

/-- Prove that the number of ways to put 6 distinguishable balls in 3 distinguishable boxes is 729 (which is 3^6). -/
theorem balls_in_boxes : (3 ^ 6) = 729 :=
by 
  sorry

end balls_in_boxes_l221_221401


namespace m₁_m₂_relationship_l221_221069

-- Defining the conditions
variables {Point Line : Type}
variables (intersect : Line → Line → Prop)
variables (coplanar : Line → Line → Prop)

-- Assumption that lines l₁ and l₂ are non-coplanar.
variables {l₁ l₂ : Line} (h_non_coplanar : ¬ coplanar l₁ l₂)

-- Assuming m₁ and m₂ both intersect with l₁ and l₂.
variables {m₁ m₂ : Line}
variables (h_intersect_m₁_l₁ : intersect m₁ l₁)
variables (h_intersect_m₁_l₂ : intersect m₁ l₂)
variables (h_intersect_m₂_l₁ : intersect m₂ l₁)
variables (h_intersect_m₂_l₂ : intersect m₂ l₂)

-- Statement to prove that m₁ and m₂ are either intersecting or non-coplanar.
theorem m₁_m₂_relationship :
  (¬ coplanar m₁ m₂) ∨ (∃ p : Point, (intersect m₁ m₂ ∧ intersect m₂ m₁)) :=
sorry

end m₁_m₂_relationship_l221_221069


namespace express_set_l221_221190

open Set

/-- Define the set of natural numbers for which an expression is also a natural number. -/
theorem express_set : {x : ℕ | ∃ y : ℕ, 6 = y * (5 - x)} = {2, 3, 4} :=
by
  sorry

end express_set_l221_221190


namespace gcd_factorial_eight_nine_eq_8_factorial_l221_221022

theorem gcd_factorial_eight_nine_eq_8_factorial : Nat.gcd (Nat.factorial 8) (Nat.factorial 9) = Nat.factorial 8 := 
by 
  sorry

end gcd_factorial_eight_nine_eq_8_factorial_l221_221022


namespace root_interval_k_l221_221040

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem root_interval_k (k : ℤ) (h_cont : Continuous f) (h_mono : Monotone f)
  (h1 : f 2 < 0) (h2 : f 3 > 0) : k = 4 :=
by
  -- The proof part is omitted as per instruction.
  sorry

end root_interval_k_l221_221040


namespace inequalities_of_function_nonneg_l221_221765

theorem inequalities_of_function_nonneg (a b A B : ℝ)
  (h : ∀ θ : ℝ, 1 - a * Real.cos θ - b * Real.sin θ - A * Real.sin (2 * θ) - B * Real.cos (2 * θ) ≥ 0) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := sorry

end inequalities_of_function_nonneg_l221_221765


namespace ratio_expression_value_l221_221722

theorem ratio_expression_value (A B C : ℚ) (h_ratio : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
by
  sorry

end ratio_expression_value_l221_221722


namespace parallelogram_area_twice_quadrilateral_area_l221_221091

theorem parallelogram_area_twice_quadrilateral_area (S : ℝ) (LMNP_area : ℝ) 
  (h : LMNP_area = 2 * S) : LMNP_area = 2 * S := 
by {
  sorry
}

end parallelogram_area_twice_quadrilateral_area_l221_221091


namespace min_value_x_plus_inv_x_l221_221248

open Real

theorem min_value_x_plus_inv_x (x : ℝ) (hx : 0 < x) : x + 1/x ≥ 2 := by
  sorry

end min_value_x_plus_inv_x_l221_221248


namespace log_eight_of_five_twelve_l221_221980

theorem log_eight_of_five_twelve : log 8 512 = 3 :=
by
  -- Definitions from the problem conditions
  have h₁ : 8 = 2^3 := rfl
  have h₂ : 512 = 2^9 := rfl
  sorry

end log_eight_of_five_twelve_l221_221980


namespace uncle_dave_ice_cream_sandwiches_l221_221488

theorem uncle_dave_ice_cream_sandwiches (n : ℕ) (s : ℕ) (total : ℕ) 
  (h1 : n = 11) (h2 : s = 13) (h3 : total = n * s) : total = 143 := by
  sorry

end uncle_dave_ice_cream_sandwiches_l221_221488


namespace tan_alpha_solution_l221_221390

variable (α : ℝ)
variable (h₀ : 0 < α ∧ α < π)
variable (h₁ : Real.sin α + Real.cos α = 7 / 13)

theorem tan_alpha_solution : Real.tan α = -12 / 5 := 
by
  sorry

end tan_alpha_solution_l221_221390


namespace cages_used_l221_221679

theorem cages_used (total_puppies sold_puppies puppies_per_cage remaining_puppies needed_cages additional_cage total_cages: ℕ) 
  (h1 : total_puppies = 36) 
  (h2 : sold_puppies = 7) 
  (h3 : puppies_per_cage = 4) 
  (h4 : remaining_puppies = total_puppies - sold_puppies) 
  (h5 : needed_cages = remaining_puppies / puppies_per_cage) 
  (h6 : additional_cage = if (remaining_puppies % puppies_per_cage = 0) then 0 else 1) 
  (h7 : total_cages = needed_cages + additional_cage) : 
  total_cages = 8 := 
by 
  sorry

end cages_used_l221_221679


namespace connect_four_no_win_probability_l221_221416

-- Definitions based on the conditions
def connect_four := {grid : array (7*6) (option (sum unit unit)) // 
  ∀ (row column : ℕ), (∀ direction : ℤ × ℤ, direction ≠ (0, 0) →  
  (0 ≤ row + 3 * direction.1 ∧ row + 3 * direction.1 < 6) ∧ 
  (0 ≤ column + 3 * direction.2 ∧ column + 3 * direction.2 < 7) → 
  (1 ≤ row ∧ row < 6) ∧ (1 ≤ column ∧ column < 7) → 
  ¬ (grid.get! (row + 3 * direction.1) (column + 3 * direction.2) = 
    some (sum.inl ())))}

def random_play (players_turn : ℕ) (grid: array (7*6) (option (sum unit unit))) :
    array (7*6) (option (sum unit unit)) :=
  sorry -- definition of a random play will be complex and is not provided here
  
def probability_no_win : ℝ :=
  sorry -- simulated or empirical estimation of the probability

theorem connect_four_no_win_probability :
  probability_no_win ≈ 0.0025632817 :=
sorry

end connect_four_no_win_probability_l221_221416


namespace area_of_enclosed_shape_l221_221917

noncomputable def areaEnclosedByCurves : ℝ :=
  ∫ x in (-2:ℝ)..(1:ℝ), (2 - x^2 - x)

theorem area_of_enclosed_shape :
  areaEnclosedByCurves = 9 / 2 :=
by
  sorry

end area_of_enclosed_shape_l221_221917


namespace feathers_per_flamingo_l221_221905

theorem feathers_per_flamingo (num_boa : ℕ) (feathers_per_boa : ℕ) (num_flamingoes : ℕ) (pluck_rate : ℚ)
  (total_feathers : ℕ) (feathers_per_flamingo : ℕ) :
  num_boa = 12 →
  feathers_per_boa = 200 →
  num_flamingoes = 480 →
  pluck_rate = 0.25 →
  total_feathers = num_boa * feathers_per_boa →
  total_feathers = num_flamingoes * feathers_per_flamingo * pluck_rate →
  feathers_per_flamingo = 20 :=
by
  intros h_num_boa h_feathers_per_boa h_num_flamingoes h_pluck_rate h_total_feathers h_feathers_eq
  sorry

end feathers_per_flamingo_l221_221905


namespace min_value_of_f_l221_221631

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then (Real.log x / Real.log 2) * (Real.log (2 * x) / Real.log 2) else 0

theorem min_value_of_f : ∃ x > 0, f x = -1/4 :=
sorry

end min_value_of_f_l221_221631


namespace common_difference_of_arithmetic_sequence_l221_221106

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence_sum (n : ℕ) (an : ℕ → α) : α :=
  (n : α) * an 1 + (n * (n - 1) / 2 * (an 2 - an 1))

theorem common_difference_of_arithmetic_sequence (S : ℕ → ℕ) (d : ℕ) (a1 a2 : ℕ)
  (h1 : ∀ n, S n = 4 * n ^ 2 - n)
  (h2 : a1 = S 1)
  (h3 : a2 = S 2 - S 1) :
  d = a2 - a1 → d = 8 := by
  sorry

end common_difference_of_arithmetic_sequence_l221_221106


namespace stone_105_is_3_l221_221191

def stone_numbered_at_105 (n : ℕ) := (15 + (n - 1) % 28)

theorem stone_105_is_3 :
  stone_numbered_at_105 105 = 3 := by
  sorry

end stone_105_is_3_l221_221191


namespace lattice_intersections_l221_221966

theorem lattice_intersections (squares : ℕ) (circles : ℕ) 
        (line_segment : ℤ × ℤ → ℤ × ℤ) 
        (radius : ℚ) (side_length : ℚ) : 
        line_segment (0, 0) = (1009, 437) → 
        radius = 1/8 → side_length = 1/4 → 
        (squares + circles = 430) :=
by
  sorry

end lattice_intersections_l221_221966


namespace math_club_total_members_l221_221581

   theorem math_club_total_members:
     ∀ (num_females num_males total_members : ℕ),
     num_females = 6 →
     num_males = 2 * num_females →
     total_members = num_females + num_males →
     total_members = 18 :=
   by
     intros num_females num_males total_members
     intros h_females h_males h_total
     rw [h_females, h_males] at h_total
     exact h_total
   
end math_club_total_members_l221_221581


namespace smallest_sum_l221_221378

theorem smallest_sum (x y : ℕ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x ≠ y) (eq : 1/x + 1/y = 1/10) : x + y = 45 :=
by
  sorry

end smallest_sum_l221_221378


namespace number_of_subsets_l221_221194

def num_subsets (n : ℕ) : ℕ := 2 ^ n

theorem number_of_subsets (A : Finset α) (n : ℕ) (h : A.card = n) : A.powerset.card = num_subsets n :=
by
  have : A.powerset.card = 2 ^ A.card := sorry -- Proof omitted
  rw [h] at this
  exact this

end number_of_subsets_l221_221194


namespace train_crossing_time_l221_221512

def speed_kmph : ℝ := 90
def length_train : ℝ := 225

noncomputable def speed_mps : ℝ := speed_kmph * (1000 / 3600)

theorem train_crossing_time : (length_train / speed_mps) = 9 := by
  sorry

end train_crossing_time_l221_221512


namespace arithmetic_sequence_properties_l221_221706

noncomputable def arithmeticSeq (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem arithmetic_sequence_properties (a₁ d : ℕ) (n : ℕ) (h1 : d = 2)
  (h2 : (a₁ + d)^2 = a₁ * (a₁ + 3 * d)) :
  (a₁ = 2) ∧ (∃ S, S = (n * (2 * a₁ + (n - 1) * d)) / 2 ∧ S = n^2 + n) :=
by 
  sorry

end arithmetic_sequence_properties_l221_221706


namespace moving_circle_passes_through_focus_l221_221311

-- Given conditions
def is_on_parabola (x y : ℝ) : Prop :=
  y^2 = 8 * x

def is_tangent_to_line (circle_center_x : ℝ) : Prop :=
  circle_center_x + 2 = 0

-- Prove that the point (2,0) lies on the moving circle
theorem moving_circle_passes_through_focus (circle_center_x circle_center_y : ℝ) :
  is_on_parabola circle_center_x circle_center_y →
  is_tangent_to_line circle_center_x →
  (circle_center_x - 2)^2 + circle_center_y^2 = (circle_center_x + 2)^2 :=
by
  -- Proof skipped with sorry.
  sorry

end moving_circle_passes_through_focus_l221_221311


namespace annulus_area_l221_221970

theorem annulus_area (B C RW : ℝ) (h1 : B > C)
  (h2 : B^2 - (C + 5)^2 = RW^2) : 
  π * RW^2 = π * (B^2 - (C + 5)^2) :=
by
  sorry

end annulus_area_l221_221970


namespace total_ages_l221_221852

variable (Craig_age Mother_age : ℕ)

theorem total_ages (h1 : Craig_age = 16) (h2 : Mother_age = Craig_age + 24) : Craig_age + Mother_age = 56 := by
  sorry

end total_ages_l221_221852


namespace range_of_fx_a_eq_2_range_of_a_increasing_fx_l221_221217

-- Part (1)
theorem range_of_fx_a_eq_2 (x : ℝ) (h : x ∈ Set.Icc (-2 : ℝ) (3 : ℝ)) :
  ∃ y ∈ Set.Icc (-21 / 4 : ℝ) (15 : ℝ), y = x^2 + 3 * x - 3 :=
sorry

-- Part (2)
theorem range_of_a_increasing_fx (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) (3 : ℝ) → 2 * x + 2 * a - 1 ≥ 0) ↔ a ∈ Set.Ici (3 / 2 : ℝ) :=
sorry

end range_of_fx_a_eq_2_range_of_a_increasing_fx_l221_221217


namespace sofia_running_time_l221_221085

theorem sofia_running_time :
  let distance_first_section := 100 -- meters
  let speed_first_section := 5 -- meters per second
  let distance_second_section := 300 -- meters
  let speed_second_section := 4 -- meters per second
  let num_laps := 6
  let time_first_section := distance_first_section / speed_first_section -- in seconds
  let time_second_section := distance_second_section / speed_second_section -- in seconds
  let time_per_lap := time_first_section + time_second_section -- in seconds
  let total_time_seconds := num_laps * time_per_lap -- in seconds
  let total_time_minutes := total_time_seconds / 60 -- integer division for minutes
  let remaining_seconds := total_time_seconds % 60 -- modulo for remaining seconds
  total_time_minutes = 9 ∧ remaining_seconds = 30 := 
  by
  sorry

end sofia_running_time_l221_221085


namespace area_of_paper_l221_221496

theorem area_of_paper (L W : ℕ) (h1 : 2 * L + W = 34) (h2 : L + 2 * W = 38) : 
  L * W = 140 := 
by sorry

end area_of_paper_l221_221496


namespace product_of_four_consecutive_integers_l221_221242

theorem product_of_four_consecutive_integers (n : ℤ) : ∃ k : ℤ, k^2 = (n-1) * n * (n+1) * (n+2) + 1 :=
by
  sorry

end product_of_four_consecutive_integers_l221_221242


namespace not_all_inequalities_hold_l221_221387

theorem not_all_inequalities_hold (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ¬ ((a + b < c + d) ∧ ((a + b) * (c + d) < a * b + c * d) ∧ ((a + b) * c * d < a * b * (c + d))) :=
by
  sorry

end not_all_inequalities_hold_l221_221387


namespace side_length_square_l221_221928

-- Define the length and width of the rectangle
def length_rect := 10 -- cm
def width_rect := 8 -- cm

-- Define the perimeter of the rectangle
def perimeter_rect := 2 * (length_rect + width_rect)

-- Define the perimeter of the square
def perimeter_square (s : ℕ) := 4 * s

-- The theorem to prove
theorem side_length_square : ∃ s : ℕ, perimeter_rect = perimeter_square s ∧ s = 9 :=
by
  sorry

end side_length_square_l221_221928


namespace pool_people_count_l221_221472

theorem pool_people_count (P : ℕ) (total_money : ℝ) (cost_per_person : ℝ) (leftover_money : ℝ) 
  (h1 : total_money = 30) 
  (h2 : cost_per_person = 2.50) 
  (h3 : leftover_money = 5) 
  (h4 : total_money - leftover_money = cost_per_person * P) : 
  P = 10 :=
sorry

end pool_people_count_l221_221472


namespace exists_square_in_interval_l221_221435

def x_k (k : ℕ) : ℕ := k * (k + 1) / 2

noncomputable def sum_x (n : ℕ) : ℕ := (List.range n).map x_k |>.sum

theorem exists_square_in_interval (n : ℕ) (hn : n ≥ 10) :
  ∃ m, (sum_x n - x_k n ≤ m^2 ∧ m^2 ≤ sum_x n) :=
by sorry

end exists_square_in_interval_l221_221435


namespace largest_n_unique_k_l221_221941

-- Defining the main theorem statement
theorem largest_n_unique_k :
  ∃ (n : ℕ), (n = 63) ∧ (∃! (k : ℤ), (9 / 17 : ℚ) < (n : ℚ) / ((n + k) : ℚ) ∧ (n : ℚ) / ((n + k) : ℚ) < (8 / 15 : ℚ)) :=
sorry

end largest_n_unique_k_l221_221941


namespace intersection_A_B_l221_221205

def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℝ := {x | abs (x - 1) ≤ 1}

theorem intersection_A_B : A ∩ B = {1, 2} :=
  sorry

end intersection_A_B_l221_221205


namespace cookie_ratio_l221_221686

theorem cookie_ratio (K : ℕ) (h1 : K / 2 + K + 24 = 33) : 24 / K = 4 :=
by {
  sorry
}

end cookie_ratio_l221_221686


namespace range_of_a_l221_221236

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a + Real.cos (2 * x) < 5 - 4 * Real.sin x + Real.sqrt (5 * a - 4)) :
  a ∈ Set.Icc (4 / 5) 8 :=
sorry

end range_of_a_l221_221236


namespace possible_box_dimensions_l221_221953

-- Define the initial conditions
def edge_length_original_box := 4
def edge_length_dice := 1
def total_cubes := (edge_length_original_box * edge_length_original_box * edge_length_original_box)

-- Prove that these are the possible dimensions of boxes with square bases that fit all the dice
theorem possible_box_dimensions :
  ∃ (len1 len2 len3 : ℕ), 
  total_cubes = (len1 * len2 * len3) ∧ 
  (len1 = len2) ∧ 
  ((len1, len2, len3) = (1, 1, 64) ∨ (len1, len2, len3) = (2, 2, 16) ∨ (len1, len2, len3) = (4, 4, 4) ∨ (len1, len2, len3) = (8, 8, 1)) :=
by {
  sorry -- The proof would be placed here
}

end possible_box_dimensions_l221_221953


namespace greatest_integer_y_l221_221127

theorem greatest_integer_y (y : ℤ) : (8 : ℚ) / 11 > y / 17 ↔ y ≤ 12 := 
sorry

end greatest_integer_y_l221_221127


namespace max_diagonal_intersections_l221_221347

theorem max_diagonal_intersections (n : ℕ) (h : n ≥ 4) : 
    ∃ k, k = n * (n - 1) * (n - 2) * (n - 3) / 24 :=
by
    sorry

end max_diagonal_intersections_l221_221347


namespace investment_duration_l221_221021

theorem investment_duration 
  (P : ℝ) (A : ℝ) (r : ℝ) (t : ℝ)
  (h1 : P = 939.60)
  (h2 : A = 1120)
  (h3 : r = 8) :
  t = 2.4 :=
by
  sorry

end investment_duration_l221_221021


namespace range_of_m_l221_221202

open Set

-- Definitions and conditions
def p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 10
def q (x m : ℝ) : Prop := (x + m - 1) * (x - m - 1) ≤ 0
def neg_p (x : ℝ) : Prop := ¬ p x
def neg_q (x m : ℝ) : Prop := ¬ q x m

-- Theorem statement
theorem range_of_m (x m : ℝ) (h₁ : ¬ p x → ¬ q x m) (h₂ : m > 0) : m ≥ 9 :=
  sorry

end range_of_m_l221_221202


namespace probability_first_spade_second_king_l221_221934

/--
In a standard deck of 52 cards, the probability of drawing the first card as a ♠ and the second card as a king is 1/52.
-/
theorem probability_first_spade_second_king : 
  let deck_size := 52 in
  let hearts_count := 13 in
  let kings_count := 4 in
  let prob := (1 / deck_size : ℚ) * (kings_count / (deck_size - 1)) + ((hearts_count - 1) / deck_size) * (kings_count / (deck_size - 1)) 
  in 
  prob = 1 / deck_size :=
by
  sorry

end probability_first_spade_second_king_l221_221934


namespace brown_house_number_l221_221842

-- Defining the problem conditions
def sum_arithmetic_series (k : ℕ) := k * (k + 1) / 2

theorem brown_house_number (t n : ℕ) (h1 : 20 < t) (h2 : t < 500)
    (h3 : sum_arithmetic_series n = sum_arithmetic_series t / 2) : n = 84 := by
  sorry

end brown_house_number_l221_221842


namespace area_of_rectangular_plot_l221_221099

theorem area_of_rectangular_plot (breadth : ℝ) (length : ℝ) 
    (h1 : breadth = 17) 
    (h2 : length = 3 * breadth) : 
    length * breadth = 867 := 
by
  sorry

end area_of_rectangular_plot_l221_221099


namespace son_l221_221325

theorem son's_age (S M : ℕ) 
  (h1 : M = S + 35)
  (h2 : M + 2 = 2 * (S + 2)) : 
  S = 33 := 
by
  sorry

end son_l221_221325


namespace cookout_2006_kids_l221_221052

def kids_2004 : ℕ := 60
def kids_2005 : ℕ := kids_2004 / 2
def kids_2006 : ℕ := (2 * kids_2005) / 3

theorem cookout_2006_kids : kids_2006 = 20 := by
  sorry

end cookout_2006_kids_l221_221052


namespace greatest_y_least_y_greatest_integer_y_l221_221136

theorem greatest_y (y : ℤ) (H : (8 : ℝ) / 11 > y / 17) : y ≤ 12 :=
sorry

theorem least_y (y : ℤ) (H : (8 : ℝ) / 11 > y / 17) : y ≥ 12 :=
sorry

theorem greatest_integer_y : ∀ (y : ℤ), ((8 : ℝ) / 11 > y / 17) → y = 12 :=
by
  intro y H
  apply le_antisymm
  apply greatest_y y H
  apply least_y y H

end greatest_y_least_y_greatest_integer_y_l221_221136


namespace baking_dish_to_recipe_book_ratio_is_2_l221_221601

-- Definitions of costs
def cost_recipe_book : ℕ := 6
def cost_ingredient : ℕ := 3
def num_ingredients : ℕ := 5
def cost_apron : ℕ := cost_recipe_book + 1
def total_spent : ℕ := 40

-- Definition to calculate the total cost excluding the baking dish
def cost_excluding_baking_dish : ℕ :=
  cost_recipe_book + cost_apron + cost_ingredient * num_ingredients

-- Definition of cost of baking dish
def cost_baking_dish : ℕ := total_spent - cost_excluding_baking_dish

-- Definition of the ratio
def ratio_baking_dish_to_recipe_book : ℕ := cost_baking_dish / cost_recipe_book

-- Theorem stating that the ratio is 2
theorem baking_dish_to_recipe_book_ratio_is_2 :
  ratio_baking_dish_to_recipe_book = 2 :=
sorry

end baking_dish_to_recipe_book_ratio_is_2_l221_221601


namespace company_A_profit_l221_221179

-- Define the conditions
def total_profit (x : ℝ) : ℝ := x
def company_B_share (x : ℝ) : Prop := 0.4 * x = 60000
def company_A_percentage : ℝ := 0.6

-- Define the statement to be proved
theorem company_A_profit (x : ℝ) (h : company_B_share x) : 0.6 * x = 90000 := sorry

end company_A_profit_l221_221179


namespace curve_symmetry_l221_221187

-- Define the curve equation
def curve_eq (x y : ℝ) : Prop := x * y^2 - x^2 * y = -2

-- Define the symmetry condition about the line y = -x
def symmetry_about_y_equals_neg_x (x y : ℝ) : Prop :=
  curve_eq (-y) (-x)

-- Define the original curve equation
def original_curve (x y : ℝ) : Prop := curve_eq x y

-- Proof statement: The curve xy^2 - x^2y = -2 is symmetric about the line y = -x.
theorem curve_symmetry : ∀ (x y : ℝ), original_curve x y ↔ symmetry_about_y_equals_neg_x x y :=
by
  sorry

end curve_symmetry_l221_221187


namespace find_m_l221_221395

theorem find_m {A B : Set ℝ} (m : ℝ) :
  (A = {x : ℝ | x^2 + x - 12 = 0}) →
  (B = {x : ℝ | mx + 1 = 0}) →
  (A ∩ B = {3}) →
  m = -1 / 3 := 
by
  intros hA hB h_inter
  sorry

end find_m_l221_221395


namespace find_m_l221_221710

theorem find_m (x p q m : ℝ) 
    (h1 : 4 * p^2 + 9 * q^2 = 2) 
    (h2 : (1/2) * x + 3 * p * q = 1) 
    (h3 : ∀ x, x^2 + 2 * m * x - 3 * m + 1 ≥ 1) :
    m = -3 ∨ m = 1 :=
sorry

end find_m_l221_221710


namespace translate_parabola_l221_221295

theorem translate_parabola :
  (∀ x, y = 1/2 * x^2 + 1 → y = 1/2 * (x - 1)^2 - 2) :=
by
  sorry

end translate_parabola_l221_221295


namespace ratio_of_inscribed_to_circumscribed_l221_221705

theorem ratio_of_inscribed_to_circumscribed (a : ℝ) :
  let r' := a * Real.sqrt 6 / 12
  let R' := a * Real.sqrt 6 / 4
  r' / R' = 1 / 3 := by
  sorry

end ratio_of_inscribed_to_circumscribed_l221_221705


namespace second_integer_value_l221_221485

-- Definitions of conditions directly from a)
def consecutive_integers (a b c : ℤ) : Prop :=
  b = a + 1 ∧ c = b + 1

def sum_of_first_and_third (a c : ℤ) (sum : ℤ) : Prop :=
  a + c = sum

-- Translated proof problem
theorem second_integer_value (n: ℤ) (h1: consecutive_integers (n - 1) n (n + 1))
  (h2: sum_of_first_and_third (n - 1) (n + 1) 118) : 
  n = 59 :=
by
  sorry

end second_integer_value_l221_221485


namespace simplify_complex_fraction_l221_221613

theorem simplify_complex_fraction : 
  (6 - 3 * Complex.I) / (-2 + 5 * Complex.I) = (-27 / 29) - (24 / 29) * Complex.I := 
by 
  sorry

end simplify_complex_fraction_l221_221613


namespace part1_a_eq_zero_part2_range_of_a_l221_221556

noncomputable def f (x : ℝ) := abs (x + 1)
noncomputable def g (x : ℝ) (a : ℝ) := 2 * abs x + a

theorem part1_a_eq_zero :
  ∀ x, 0 < x + 1 → 0 < 2 * abs x → a = 0 →
  f x ≥ g x a ↔ (-1 / 3 : ℝ) ≤ x ∧ x ≤ 1 :=
sorry

theorem part2_range_of_a :
  ∃ x, f x ≥ g x a ↔ a ≤ 1 :=
sorry

end part1_a_eq_zero_part2_range_of_a_l221_221556


namespace train_length_is_50_meters_l221_221297

theorem train_length_is_50_meters
  (L : ℝ)
  (equal_length : ∀ (a b : ℝ), a = L ∧ b = L → a + b = 2 * L)
  (speed_faster_train : ℝ := 46) -- km/hr
  (speed_slower_train : ℝ := 36) -- km/hr
  (relative_speed : ℝ := speed_faster_train - speed_slower_train)
  (relative_speed_km_per_sec : ℝ := relative_speed / 3600) -- converting km/hr to km/sec
  (time : ℝ := 36) -- seconds
  (distance_covered : ℝ := 2 * L)
  (distance_eq : distance_covered = relative_speed_km_per_sec * time):
  L = 50 / 1000 :=
by 
  -- We will prove it as per the derived conditions
  sorry

end train_length_is_50_meters_l221_221297


namespace functional_equation_solution_l221_221018

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ),
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → f(x) + f(1 / (1 - x)) = x) →
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → f(x) = 1 / 2 * (x + 1 - 1 / x - 1 / (1 - x))) :=
by
  intros f h x hx,
  sorry

end functional_equation_solution_l221_221018


namespace egg_cartons_l221_221000

theorem egg_cartons (chickens eggs_per_chicken eggs_per_carton : ℕ) (h_chickens : chickens = 20) (h_eggs_per_chicken : eggs_per_chicken = 6) (h_eggs_per_carton : eggs_per_carton = 12) : 
  (chickens * eggs_per_chicken) / eggs_per_carton = 10 :=
by
  rw [h_chickens, h_eggs_per_chicken, h_eggs_per_carton] -- Replace the variables with the given values
  -- Calculate the number of eggs
  have h_eggs := 20 * 6
  -- Apply the number of eggs to find the number of cartons
  rw [show 20 * 6 = 120, from rfl, show 120 / 12 = 10, from rfl]
  sorry -- Placeholder for the detailed proof

end egg_cartons_l221_221000


namespace prime_product_correct_l221_221027

theorem prime_product_correct 
    (p1 : Nat := 1021031) (pr1 : Prime p1)
    (p2 : Nat := 237019) (pr2 : Prime p2) :
    p1 * p2 = 241940557349 :=
by
  sorry

end prime_product_correct_l221_221027


namespace tan_half_angle_l221_221551

theorem tan_half_angle {α β : ℝ} (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) (h1 : Real.tan α = 2) (h2 : Real.tan β = 3) :
  Real.tan ((α + β) / 2) = 1 + Real.sqrt 2 := 
sorry

end tan_half_angle_l221_221551


namespace smallest_q_for_5_in_range_l221_221026

theorem smallest_q_for_5_in_range : ∃ q, (q = 9) ∧ (∃ x, (x^2 - 4 * x + q = 5)) := 
by 
  sorry

end smallest_q_for_5_in_range_l221_221026


namespace max_distance_line_l221_221470

noncomputable def equation_of_line (x y : ℝ) : ℝ := x + 2 * y - 5

theorem max_distance_line (x y : ℝ) : 
  equation_of_line 1 2 = 0 ∧ 
  (∀ (a b c : ℝ), c ≠ 0 → (x = 1 ∧ y = 2 → equation_of_line x y = 0)) ∧ 
  (∀ (L : ℝ → ℝ → ℝ), L 1 2 = 0 → (L = equation_of_line)) :=
sorry

end max_distance_line_l221_221470


namespace height_of_right_triangle_on_parabola_equals_one_l221_221960

theorem height_of_right_triangle_on_parabola_equals_one 
    (x0 x1 x2 : ℝ) 
    (h0 : x0 ≠ x1)
    (h1 : x0 ≠ x2) 
    (h2 : x1 ≠ x2) 
    (h3 : x0^2 = x1^2) 
    (h4 : x0^2 < x2^2):
    x2^2 - x0^2 = 1 := by
  sorry

end height_of_right_triangle_on_parabola_equals_one_l221_221960


namespace range_of_function_l221_221791

theorem range_of_function : ∀ x : ℝ, 1 ≤ abs (Real.sin x) + 2 * abs (Real.cos x) ∧ abs (Real.sin x) + 2 * abs (Real.cos x) ≤ Real.sqrt 5 :=
by
  intro x
  sorry

end range_of_function_l221_221791


namespace func_passes_through_fixed_point_l221_221627

theorem func_passes_through_fixed_point (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : 
  a^(2 * (1 / 2) - 1) = 1 :=
by
  sorry

end func_passes_through_fixed_point_l221_221627


namespace probability_snow_at_least_once_l221_221477

noncomputable def probability_at_least_once_snow : ℚ :=
  1 - (↑((1:ℚ) / 4) ^ 5)

theorem probability_snow_at_least_once (p : ℚ) (h : p = 3 / 4) :
  probability_at_least_once_snow = 1023 / 1024 := by
  sorry

end probability_snow_at_least_once_l221_221477


namespace distribute_6_balls_in_3_boxes_l221_221400

def number_of_ways_to_distribute_balls (balls boxes : Nat) : Nat :=
  boxes ^ balls

theorem distribute_6_balls_in_3_boxes : number_of_ways_to_distribute_balls 6 3 = 729 := by
  sorry

end distribute_6_balls_in_3_boxes_l221_221400


namespace mowing_ratio_is_sqrt2_l221_221510

noncomputable def mowing_ratio (s w : ℝ) (hw_half_area : w * (s * Real.sqrt 2) = s^2) : ℝ :=
  s / w

theorem mowing_ratio_is_sqrt2 (s w : ℝ) (hs_positive : s > 0) (hw_positive : w > 0)
  (hw_half_area : w * (s * Real.sqrt 2) = s^2) : mowing_ratio s w hw_half_area = Real.sqrt 2 :=
by
  sorry

end mowing_ratio_is_sqrt2_l221_221510


namespace octahedron_vertex_probability_l221_221963

/-- An octahedron consists of two square-based pyramids glued together along their square bases. 
    This forms a polyhedron with eight faces.
    An ant starts walking from the bottom vertex and randomly picks one of the four adjacent vertices 
    (middle ring) and calls it vertex A. 
    From vertex A, the ant then randomly selects one of its four adjacent vertices and calls it vertex B. 
    Prove that the probability that vertex B is the top vertex of the octahedron is 1/4. -/
theorem octahedron_vertex_probability : 
  let bottom_vertex := "initial vertex", 
      mid_ring := Set.of_list ["v1", "v2", "v3", "v4"], 
      top_vertex := "top vertex" in 
  ∀ A ∈ mid_ring, (cond_prob (λ v, v = top_vertex) (λ v, v ∈ {bottom_vertex} ∪ mid_ring ∪ {top_vertex})) = 1/4 :=
sorry

end octahedron_vertex_probability_l221_221963


namespace number_of_multiples_of_15_between_35_and_200_l221_221397

theorem number_of_multiples_of_15_between_35_and_200 : ∃ n : ℕ, n = 11 ∧ ∃ k : ℕ, k ≤ 200 ∧ k ≥ 35 ∧ (∃ m : ℕ, m < n ∧ 45 + m * 15 = k) :=
by
  sorry

end number_of_multiples_of_15_between_35_and_200_l221_221397


namespace max_students_per_class_l221_221111

theorem max_students_per_class
    (total_students : ℕ)
    (total_classes : ℕ)
    (bus_count : ℕ)
    (bus_seats : ℕ)
    (students_per_class : ℕ)
    (total_students = 920)
    (bus_count = 16)
    (bus_seats = 71)
    (∀ c < total_classes, students_per_class ≤ bus_seats) : 
    students_per_class ≤ 17 := 
by
    sorry

end max_students_per_class_l221_221111


namespace unique_8_tuple_real_l221_221193

theorem unique_8_tuple_real (x : Fin 8 → ℝ) :
  (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + (x 6 - x 7)^2 + x 7^2 = 1 / 8 →
  ∃! (y : Fin 8 → ℝ), (1 - y 0)^2 + (y 0 - y 1)^2 + (y 1 - y 2)^2 + (y 2 - y 3)^2 + (y 3 - y 4)^2 + (y 4 - y 5)^2 + (y 5 - y 6)^2 + (y 6 - y 7)^2 + y 7^2 = 1 / 8 :=
by
  sorry

end unique_8_tuple_real_l221_221193


namespace find_b_plus_m_l221_221341

open Matrix

noncomputable def X (b : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, 3, b], ![0, 1, 5], ![0, 0, 1]]

noncomputable def Y : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, 27, 8085], ![0, 1, 45], ![0, 0, 1]]

theorem find_b_plus_m (b m : ℝ)
    (h1 : X b ^ m = Y) : b + m = 847 := sorry

end find_b_plus_m_l221_221341


namespace smallest_b_factors_l221_221025

theorem smallest_b_factors (b p q : ℤ) (H : p * q = 2016) : 
  (∀ k₁ k₂ : ℤ, k₁ * k₂ = 2016 → k₁ + k₂ ≥ p + q) → 
  b = 90 :=
by
  -- Here, we assume the premises stated for integers p, q such that their product is 2016.
  -- We need to fill in the proof steps which will involve checking all appropriate (p, q) pairs.
  sorry

end smallest_b_factors_l221_221025


namespace sphere_volume_given_surface_area_l221_221887

theorem sphere_volume_given_surface_area (r : ℝ) (V : ℝ) (S : ℝ)
  (hS : S = 36 * Real.pi)
  (h_surface_area : 4 * Real.pi * r^2 = S)
  (h_volume : V = (4/3) * Real.pi * r^3) : V = 36 * Real.pi := by
  sorry

end sphere_volume_given_surface_area_l221_221887


namespace cuboid_can_form_square_projection_l221_221162

-- Definitions and conditions based directly on the problem
def length1 := 3
def length2 := 4
def length3 := 6

-- Statement to prove
theorem cuboid_can_form_square_projection (x y : ℝ) :
  (4 * x * x + y * y = 36) ∧ (x + y = 4) → True :=
by sorry

end cuboid_can_form_square_projection_l221_221162


namespace scientific_notation_l221_221967

theorem scientific_notation :
  0.000000014 = 1.4 * 10^(-8) :=
sorry

end scientific_notation_l221_221967


namespace find_multiplier_l221_221809

theorem find_multiplier (N x : ℕ) (h₁ : N = 12) (h₂ : N * x - 3 = (N - 7) * 9) : x = 4 :=
by
  sorry

end find_multiplier_l221_221809


namespace cheryl_bill_cost_correct_l221_221009

def cheryl_electricity_bill_cost : Prop :=
  ∃ (E : ℝ), 
    (E + 400) + 0.20 * (E + 400) = 1440 ∧ 
    E = 800

theorem cheryl_bill_cost_correct : cheryl_electricity_bill_cost :=
by
  sorry

end cheryl_bill_cost_correct_l221_221009


namespace infinite_solutions_l221_221563

theorem infinite_solutions (x : ℕ) :
  15 < 2 * x + 10 ↔ ∃ n : ℕ, x = n + 3 :=
by {
  sorry
}

end infinite_solutions_l221_221563


namespace marble_game_solution_l221_221901

theorem marble_game_solution (B R : ℕ) (h1 : B + R = 21) (h2 : (B * (B - 1)) / (21 * 20) = 1 / 2) : B^2 + R^2 = 261 :=
by
  sorry

end marble_game_solution_l221_221901


namespace least_number_divisible_increased_by_seven_l221_221498

theorem least_number_divisible_increased_by_seven : 
  ∃ n : ℕ, (∀ k ∈ [24, 32, 36, 54], (n + 7) % k = 0) ∧ n = 857 := 
by
  sorry

end least_number_divisible_increased_by_seven_l221_221498


namespace d_is_rth_power_of_integer_l221_221767

theorem d_is_rth_power_of_integer 
  (d r : ℤ) 
  (a b : ℤ) 
  (hr : r ≠ 0) 
  (hab : (a, b) ≠ (0, 0)) 
  (h_eq : a ^ r = d * b ^ r) : 
  ∃ (δ : ℤ), d = δ ^ r :=
sorry

end d_is_rth_power_of_integer_l221_221767


namespace geometric_sequence_sum_2018_l221_221214

noncomputable def geometric_sum (n : ℕ) (a1 q : ℝ) : ℝ :=
  if q = 1 then n * a1 else a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_2018 :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ),
    (∀ n, S n = geometric_sum n (a 1) 2) →
    a 1 = 1 / 2 →
    (a 1 * 2^2)^2 = 8 * a 1 * 2^3 - 16 →
    S 2018 = 2^2017 - 1 / 2 :=
by sorry

end geometric_sequence_sum_2018_l221_221214


namespace percentage_of_two_is_point_eight_l221_221676

theorem percentage_of_two_is_point_eight (p : ℝ) : (p / 100) * 2 = 0.8 ↔ p = 40 := 
by
  sorry

end percentage_of_two_is_point_eight_l221_221676


namespace sequence_integers_l221_221286

theorem sequence_integers (a : ℕ → ℤ) (h1 : a 1 = 1) (h2 : a 2 = 1) 
  (h3 : ∀ n ≥ 3, a n = (a (n - 1))^2 + 2 / a (n - 2)) : ∀ n, ∃ k : ℤ, a n = k :=
sorry

end sequence_integers_l221_221286


namespace probability_different_suits_l221_221255

theorem probability_different_suits (h : ∀ (c1 c2 c3 : ℕ), c1 ≠ c2 ∧ c1 ≠ c3 ∧ c2 ≠ c3 ∧ 
                                    ∀ {x}, x ∈ {c1, c2, c3} → x ∈ finset.range 52) : 
  let prob := (13 / 17) * (13 / 25) in
  prob = (169 / 425) := 
by
  sorry

end probability_different_suits_l221_221255


namespace polycarp_error_l221_221909

def three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

theorem polycarp_error (a b n : ℕ) (ha : three_digit a) (hb : three_digit b)
  (h : 10000 * a + b = n * a * b) : n = 73 :=
by
  sorry

end polycarp_error_l221_221909


namespace probability_of_first_heart_second_king_l221_221937

noncomputable def probability_first_heart_second_king : ℚ :=
  1 / 52 * 3 / 51 + 12 / 52 * 4 / 51

theorem probability_of_first_heart_second_king :
  probability_first_heart_second_king = 1 / 52 :=
by
  sorry

end probability_of_first_heart_second_king_l221_221937


namespace high_school_total_students_l221_221828

theorem high_school_total_students (N_seniors N_sample N_freshmen_sample N_sophomores_sample N_total : ℕ)
  (h_seniors : N_seniors = 1000)
  (h_sample : N_sample = 185)
  (h_freshmen_sample : N_freshmen_sample = 75)
  (h_sophomores_sample : N_sophomores_sample = 60)
  (h_proportion : N_seniors * (N_sample - (N_freshmen_sample + N_sophomores_sample)) = N_total * (N_sample - N_freshmen_sample - N_sophomores_sample)) :
  N_total = 3700 :=
by
  sorry

end high_school_total_students_l221_221828


namespace minimum_value_of_2x_plus_y_l221_221547

theorem minimum_value_of_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y + 6 = x * y) : 2 * x + y ≥ 12 :=
  sorry

end minimum_value_of_2x_plus_y_l221_221547


namespace binomial_coefficient_12_4_l221_221687

noncomputable def binomial_coefficient (n k : ℕ) : ℕ := n.choose k

theorem binomial_coefficient_12_4 : binomial_coefficient 12 4 = 495 := by
  sorry

end binomial_coefficient_12_4_l221_221687


namespace sum_SHE_equals_6_l221_221617

-- Definitions for conditions
variables {S H E : ℕ}

-- Conditions as stated in the problem
def distinct_non_zero_digits (S H E : ℕ) : Prop :=
  S ≠ H ∧ H ≠ E ∧ S ≠ E ∧ 1 ≤ S ∧ S < 8 ∧ 1 ≤ H ∧ H < 8 ∧ 1 ≤ E ∧ E < 8

-- Base 8 addition problem
def addition_holds_in_base8 (S H E : ℕ) : Prop :=
  (E + H + (S + E + H) / 8) % 8 = S ∧    -- First column carry
  (H + S + (E + H + S) / 8) % 8 = E ∧    -- Second column carry
  (S + E + (H + S + E) / 8) % 8 = H      -- Third column carry

-- Final statement
theorem sum_SHE_equals_6 :
  distinct_non_zero_digits S H E → addition_holds_in_base8 S H E → S + H + E = 6 :=
by sorry

end sum_SHE_equals_6_l221_221617


namespace greatest_possible_value_of_x_l221_221806

theorem greatest_possible_value_of_x :
  ∃ (x : ℚ), x = 9 / 5 ∧ 
  (\left(5 * x - 20) / (4 * x - 5)) ^ 2 + \left((5 * x - 20) / (4 * x - 5)) = 20 ∧ x ≥ 0 :=
begin
  existsi (9 / 5 : ℚ),
  split,
  { refl },
  split,
  { sorry },
  { sorry }
end

end greatest_possible_value_of_x_l221_221806


namespace max_value_of_f_l221_221280

noncomputable def f (x: ℝ) := (Real.sqrt x) / (x + 1)

theorem max_value_of_f :
  (∀ x ≥ 0, f x ≤ 1 / 2) ∧ (f 1 = 1 / 2) := 
begin
  sorry
end

end max_value_of_f_l221_221280


namespace fib_math_competition_l221_221861

theorem fib_math_competition :
  ∃ (n9 n8 n7 : ℕ), 
    n9 * 4 = n8 * 7 ∧ 
    n9 * 3 = n7 * 10 ∧ 
    n9 + n8 + n7 = 131 :=
sorry

end fib_math_competition_l221_221861


namespace investment_time_p_l221_221792

theorem investment_time_p (p_investment q_investment p_profit q_profit : ℝ) (p_invest_time : ℝ) (investment_ratio_pq : p_investment / q_investment = 7 / 5.00001) (profit_ratio_pq : p_profit / q_profit = 7.00001 / 10) (q_invest_time : q_invest_time = 9.999965714374696) : p_invest_time = 50 :=
sorry

end investment_time_p_l221_221792


namespace smallest_x_y_sum_l221_221383

theorem smallest_x_y_sum (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) (h4 : 1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 10) : x + y = 45 :=
by
  sorry

end smallest_x_y_sum_l221_221383


namespace find_s_l221_221246

noncomputable def is_monic (p : Polynomial ℝ) : Prop :=
  p.leadingCoeff = 1

variables (f g : Polynomial ℝ) (s : ℝ)
variables (r1 r2 r3 r4 r5 r6 : ℝ)

-- Conditions
def conditions : Prop :=
  is_monic f ∧ is_monic g ∧
  (f.roots = [s + 2, s + 8, r1] ∨ f.roots = [s + 8, s + 2, r1] ∨ f.roots = [s + 2, r1, s + 8] ∨
   f.roots = [r1, s + 2, s + 8] ∨ f.roots = [r1, s + 8, s + 2]) ∧
  (g.roots = [s + 4, s + 10, r2] ∨ g.roots = [s + 10, s + 4, r2] ∨ g.roots = [s + 4, r2, s + 10] ∨
   g.roots = [r2, s + 4, s + 10] ∨ g.roots = [r2, s + 10, s + 4]) ∧
  ∀ (x : ℝ), f.eval x - g.eval x = 2 * s

-- Theorem statement

theorem find_s (h : conditions f g r1 r2 s) : s = 288 / 14 :=
sorry

end find_s_l221_221246


namespace pairs_divisible_by_4_l221_221771

-- Define the set of valid pairs of digits from 00 to 99
def valid_pairs : List (Fin 100) := List.filter (λ n => n % 4 = 0) (List.range 100)

-- State the theorem
theorem pairs_divisible_by_4 : valid_pairs.length = 25 := by
  sorry

end pairs_divisible_by_4_l221_221771


namespace rehabilitation_centers_l221_221945

def Lisa : ℕ := 6 
def Jude : ℕ := Lisa / 2
def Han : ℕ := 2 * Jude - 2
def Jane : ℕ := 27 - Lisa - Jude - Han
def x : ℕ := 2

theorem rehabilitation_centers:
  Jane = x * Han + 6 := 
by
  -- Proof goes here (not required)
  sorry

end rehabilitation_centers_l221_221945


namespace ratio_problem_l221_221729

theorem ratio_problem 
  (A B C : ℚ) 
  (h : A / B = 3 / 2 ∧ B / C = 2 / 5 ∧ A / C = 3 / 5) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by 
  sorry

end ratio_problem_l221_221729


namespace Simon_has_72_legos_l221_221612

theorem Simon_has_72_legos 
  (Kent_legos : ℕ)
  (h1 : Kent_legos = 40) 
  (Bruce_legos : ℕ) 
  (h2 : Bruce_legos = Kent_legos + 20) 
  (Simon_legos : ℕ) 
  (h3 : Simon_legos = Bruce_legos + (Bruce_legos/5)) :
  Simon_legos = 72 := 
  by
    -- Begin proof (not required for the problem)
    -- Proof steps would follow here
    sorry

end Simon_has_72_legos_l221_221612


namespace max_angle_C_l221_221888

-- Define the necessary context and conditions
variable {a b c : ℝ}

-- Condition that a^2 + b^2 = 2c^2 in a triangle
axiom triangle_condition : a^2 + b^2 = 2 * c^2

-- Theorem statement
theorem max_angle_C (h : a^2 + b^2 = 2 * c^2) : ∃ C : ℝ, C = Real.pi / 3 := sorry

end max_angle_C_l221_221888


namespace b_car_usage_hours_l221_221666

theorem b_car_usage_hours (h : ℕ) (total_cost_a_b_c : ℕ) 
  (a_usage : ℕ) (b_payment : ℕ) (c_usage : ℕ) 
  (total_cost : total_cost_a_b_c = 720)
  (usage_a : a_usage = 9) 
  (usage_c : c_usage = 13)
  (payment_b : b_payment = 225) 
  (cost_per_hour : ℝ := total_cost_a_b_c / (a_usage + h + c_usage)) :
  b_payment = cost_per_hour * h → h = 10 := 
by
  sorry

end b_car_usage_hours_l221_221666


namespace set_intersection_example_l221_221204

theorem set_intersection_example (A : Set ℝ) (B : Set ℝ):
  A = { -1, 1, 2, 4 } → 
  B = { x | |x - 1| ≤ 1 } → 
  A ∩ B = {1, 2} :=
by
  intros hA hB
  sorry

end set_intersection_example_l221_221204


namespace continuity_sum_l221_221592

noncomputable def piecewise_function (x : ℝ) (a b c : ℝ) : ℝ :=
if h : x > 1 then a * (2 * x + 1) + 2
else if h' : -1 <= x && x <= 1 then b * x + 3
else 3 * x - c

theorem continuity_sum (a b c : ℝ) (h_cont1 : 3 * a = b + 1) (h_cont2 : c = 3 * a + 1) :
  a + c = 4 * a + 1 :=
by
  sorry

end continuity_sum_l221_221592


namespace sum_of_abc_is_12_l221_221596

theorem sum_of_abc_is_12 (a b c : ℕ) (h : 28 * a + 30 * b + 31 * c = 365) : a + b + c = 12 :=
by
  sorry

end sum_of_abc_is_12_l221_221596


namespace find_a_given_star_l221_221529

def star (a b : ℤ) : ℤ := 2 * a - b^3

theorem find_a_given_star : ∃ a : ℤ, star a 3 = 15 ∧ a = 21 :=
by
  use 21
  simp [star]
  split
  · rfl
  · omega -- or use linarith in older versions

end find_a_given_star_l221_221529


namespace multiples_of_15_between_35_and_200_l221_221398

theorem multiples_of_15_between_35_and_200 : 
  ∃ n : ℕ, ∀ k : ℕ, 35 < k * 15 ∧ k * 15 < 200 ↔ k = n :=
begin
  sorry,
end

end multiples_of_15_between_35_and_200_l221_221398


namespace trapezoid_median_l221_221682

noncomputable def median_trapezoid (base₁ base₂ height : ℝ) : ℝ :=
(base₁ + base₂) / 2

theorem trapezoid_median (b_t : ℝ) (a_t : ℝ) (h_t : ℝ) (a_tp : ℝ) 
  (h_eq : h_t = 16) (a_eq : a_t = 192) (area_tp_eq : a_tp = a_t) : median_trapezoid h_t h_t h_t = 12 :=
by
  have h_t_eq : h_t = 16 := by sorry
  have a_t_eq : a_t = 192 := by sorry
  have area_tp : a_tp = 192 := by sorry
  sorry

end trapezoid_median_l221_221682


namespace max_distance_unit_circle_l221_221999

open Complex

theorem max_distance_unit_circle : 
  ∀ (z : ℂ), abs z = 1 → ∃ M : ℝ, M = abs (z - (1 : ℂ) - I) ∧ ∀ w : ℂ, abs w = 1 → abs (w - 1 - I) ≤ M :=
by
  sorry

end max_distance_unit_circle_l221_221999


namespace ratio_problem_l221_221745

theorem ratio_problem (A B C : ℚ) (h : A / B = 3 / 2) (h' : B / C = 2 / 5) : (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := 
by
  sorry

end ratio_problem_l221_221745


namespace total_pages_in_book_l221_221604

theorem total_pages_in_book 
    (pages_read : ℕ) (pages_left : ℕ) 
    (h₁ : pages_read = 11) 
    (h₂ : pages_left = 6) : 
    pages_read + pages_left = 17 := 
by 
    sorry

end total_pages_in_book_l221_221604


namespace cylinder_height_l221_221626

theorem cylinder_height (r h : ℝ) (SA : ℝ) (h₀ : r = 3) (h₁ : SA = 36 * Real.pi) (h₂ : SA = 2 * Real.pi * r^2 + 2 * Real.pi * r * h) : h = 3 :=
by
  -- The proof will be constructed here
  sorry

end cylinder_height_l221_221626


namespace intersection_equality_l221_221220

def setA := {x : ℝ | (x - 1) * (3 - x) < 0}
def setB := {x : ℝ | -3 ≤ x ∧ x ≤ 3}

theorem intersection_equality : setA ∩ setB = {x : ℝ | -3 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_equality_l221_221220


namespace problem_l221_221597

theorem problem (a b c : ℕ) (h : 28 * a + 30 * b + 31 * c = 365) : a + b + c = 12 :=
  sorry

end problem_l221_221597


namespace no_such_function_exists_l221_221912

theorem no_such_function_exists :
  ¬ ∃ (f : ℕ → ℕ), ∀ n : ℕ, f (f n) = n + 1987 := 
sorry

end no_such_function_exists_l221_221912


namespace z_pow_n_add_inv_pow_n_eq_two_cos_n_alpha_l221_221365

-- Given conditions and definitions
variables {α : ℝ} {z : ℂ} 
  (hz : z + 1/z = 2 * Real.cos α)

-- The target statement
theorem z_pow_n_add_inv_pow_n_eq_two_cos_n_alpha (n : ℕ) (hz : z + 1/z = 2 * Real.cos α) : 
  z ^ n + 1 / (z ^ n) = 2 * Real.cos (n * α) := 
  sorry

end z_pow_n_add_inv_pow_n_eq_two_cos_n_alpha_l221_221365


namespace first_nonzero_digit_fraction_l221_221655

theorem first_nonzero_digit_fraction :
  (∃ n: ℕ, 0 < n ∧ n < 10 ∧ (n / 137 % 1) * 10 < 10 ∧ ((n / 137 % 1) * 10).floor = 2) :=
sorry

end first_nonzero_digit_fraction_l221_221655


namespace z_pow_n_add_inv_pow_n_eq_two_cos_n_alpha_l221_221364

-- Given conditions and definitions
variables {α : ℝ} {z : ℂ} 
  (hz : z + 1/z = 2 * Real.cos α)

-- The target statement
theorem z_pow_n_add_inv_pow_n_eq_two_cos_n_alpha (n : ℕ) (hz : z + 1/z = 2 * Real.cos α) : 
  z ^ n + 1 / (z ^ n) = 2 * Real.cos (n * α) := 
  sorry

end z_pow_n_add_inv_pow_n_eq_two_cos_n_alpha_l221_221364


namespace S6_is_48_l221_221370

-- Define the first term and common difference
def a₁ : ℕ := 3
def d : ℕ := 2

-- Define the formula for sum of the first n terms of an arithmetic sequence
def sum_of_arithmetic_sequence (n : ℕ) : ℕ :=
  n / 2 * (2 * a₁ + (n - 1) * d)

-- Prove that the sum of the first 6 terms is 48
theorem S6_is_48 : sum_of_arithmetic_sequence 6 = 48 := by
  sorry

end S6_is_48_l221_221370


namespace max_value_l221_221065

open Real

theorem max_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + 5 * y < 75) : 
  xy * (75 - 2 * x - 5 * y) ≤ 1562.5 := 
sorry

end max_value_l221_221065


namespace component_unqualified_l221_221316

/-- 
    The specified diameter range for a component is within [19.98, 20.02].
    The measured diameter of the component is 19.9.
    Prove that the component is unqualified.
-/
def is_unqualified (diameter_measured : ℝ) : Prop :=
    diameter_measured < 19.98 ∨ diameter_measured > 20.02

theorem component_unqualified : is_unqualified 19.9 :=
by
  -- Proof goes here
  sorry

end component_unqualified_l221_221316


namespace intersection_complement_P_CUQ_l221_221904

universe U

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5}
def CUQ : Set ℕ := U \ Q

theorem intersection_complement_P_CUQ : 
  (P ∩ CUQ) = {1, 2} :=
by 
  sorry

end intersection_complement_P_CUQ_l221_221904


namespace goods_train_length_is_280_l221_221156

noncomputable def length_of_goods_train (passenger_speed passenger_speed_kmh: ℝ) 
                                       (goods_speed goods_speed_kmh: ℝ) 
                                       (time_to_pass: ℝ) : ℝ :=
  let kmh_to_ms := (1000 : ℝ) / (3600 : ℝ)
  let passenger_speed_ms := passenger_speed * kmh_to_ms
  let goods_speed_ms     := goods_speed * kmh_to_ms
  let relative_speed     := passenger_speed_ms + goods_speed_ms
  relative_speed * time_to_pass

theorem goods_train_length_is_280 :
  length_of_goods_train 70 70 42 42 9 = 280 :=
by
  sorry

end goods_train_length_is_280_l221_221156


namespace sum_of_discounts_l221_221031

theorem sum_of_discounts
  (price_fox : ℝ)
  (price_pony : ℝ)
  (savings : ℝ)
  (discount_pony : ℝ) :
  (3 * price_fox * (F / 100) + 2 * price_pony * (discount_pony / 100) = savings) →
  (F + discount_pony = 22) :=
sorry


end sum_of_discounts_l221_221031


namespace product_of_solutions_l221_221195

theorem product_of_solutions :
  (∀ x : ℝ, |3 * x - 2| + 5 = 23 → x = 20 / 3 ∨ x = -16 / 3) →
  (20 / 3 * -16 / 3 = -320 / 9) :=
by
  intros h
  have h₁ : 20 / 3 * -16 / 3 = -320 / 9 := sorry
  exact h₁

end product_of_solutions_l221_221195


namespace interest_rate_l221_221624

theorem interest_rate (P CI SI: ℝ) (r: ℝ) : P = 5100 → CI = P * (1 + r)^2 - P → SI = P * r * 2 → (CI - SI = 51) → r = 0.1 :=
by
  intros
  -- skipping the proof
  sorry

end interest_rate_l221_221624


namespace average_age_of_new_men_is_30_l221_221273

noncomputable def average_age_of_two_new_men (A : ℝ) : ℝ :=
  let total_age_before : ℝ := 8 * A
  let total_age_after : ℝ := 8 * (A + 2)
  let age_of_replaced_men : ℝ := 21 + 23
  let total_age_of_new_men : ℝ := total_age_after - total_age_before + age_of_replaced_men
  total_age_of_new_men / 2

theorem average_age_of_new_men_is_30 (A : ℝ) : 
  average_age_of_two_new_men A = 30 :=
by 
  sorry

end average_age_of_new_men_is_30_l221_221273


namespace evaluate_expression_l221_221857

theorem evaluate_expression : 8^3 + 3 * 8^2 + 3 * 8 + 1 = 729 := by
  sorry

end evaluate_expression_l221_221857


namespace find_a_l221_221995

theorem find_a (a : ℝ) (h : ∃ x : ℝ, x = 2 ∧ x^2 + a * x - 2 = 0) : a = -1 := 
by 
  sorry

end find_a_l221_221995


namespace greatest_integer_y_l221_221126

theorem greatest_integer_y (y : ℤ) : (8 : ℚ) / 11 > y / 17 ↔ y ≤ 12 := 
sorry

end greatest_integer_y_l221_221126


namespace solve_fractional_eq_l221_221777

noncomputable def fractional_eq (x : ℝ) : Prop := 
  (3 / (x^2 - 3 * x) + (x - 1) / (x - 3) = 1)

noncomputable def not_zero_denom (x : ℝ) : Prop := 
  (x^2 - 3 * x ≠ 0) ∧ (x - 3 ≠ 0)

theorem solve_fractional_eq : fractional_eq (-3/2) ∧ not_zero_denom (-3/2) :=
by
  sorry

end solve_fractional_eq_l221_221777


namespace greatest_integer_l221_221130

theorem greatest_integer (y : ℤ) : (8 / 11 : ℝ) > (y / 17 : ℝ) → y ≤ 12 :=
by sorry

end greatest_integer_l221_221130


namespace sin_phi_value_l221_221923

theorem sin_phi_value 
  (φ α : ℝ)
  (hφ : φ = 2 * α)
  (hα1 : Real.sin α = (Real.sqrt 5) / 5)
  (hα2 : Real.cos α = 2 * (Real.sqrt 5) / 5) 
  : Real.sin φ = 4 / 5 := 
by 
  sorry

end sin_phi_value_l221_221923


namespace arithmetic_sequence_properties_l221_221961

-- Defining the arithmetic sequence and the conditions
variable {a : ℕ → ℤ}
variable {d : ℤ}
noncomputable def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = d

-- Given conditions
variable (h1 : a 5 = 10)
variable (h2 : a 1 + a 2 + a 3 = 3)

-- The theorem to prove
theorem arithmetic_sequence_properties :
  is_arithmetic_sequence a d → a 1 = -2 ∧ d = 3 :=
sorry

end arithmetic_sequence_properties_l221_221961


namespace john_annual_payment_l221_221427

open Real

-- Definitions extracted from the problem:
def epipen_cost : ℝ := 500
def insurance_coverage : ℝ := 0.75
def epipen_frequency_per_year : ℕ := 2
def john_payment_per_epipen : ℝ := epipen_cost * (1 - insurance_coverage)

-- The statement to be proved:
theorem john_annual_payment :
  john_payment_per_epipen * epipen_frequency_per_year = 250 :=
by
  sorry

end john_annual_payment_l221_221427


namespace parabolic_arch_height_l221_221506

/-- Define the properties of the parabolic arch -/
def parabolic_arch (a k x : ℝ) : ℝ := a * x^2 + k

/-- Define the conditions of the problem -/
def conditions (a k : ℝ) : Prop :=
  (parabolic_arch a k 25 = 0) ∧ (parabolic_arch a k 0 = 20)

theorem parabolic_arch_height (a k : ℝ) (condition_a_k : conditions a k) :
  parabolic_arch a k 10 = 16.8 :=
by
  unfold conditions at condition_a_k
  cases' condition_a_k with h1 h2
  sorry

end parabolic_arch_height_l221_221506


namespace parking_savings_l221_221328

theorem parking_savings (weekly_cost : ℕ) (monthly_cost : ℕ) (weeks_in_year : ℕ) (months_in_year : ℕ)
  (h_weekly_cost : weekly_cost = 10)
  (h_monthly_cost : monthly_cost = 42)
  (h_weeks_in_year : weeks_in_year = 52)
  (h_months_in_year : months_in_year = 12) :
  weekly_cost * weeks_in_year - monthly_cost * months_in_year = 16 := 
by
  sorry

end parking_savings_l221_221328


namespace determinant_expression_l221_221436

theorem determinant_expression (a b c p q : ℝ) 
  (h_root : ∀ x : ℝ, (x = a ∨ x = b ∨ x = c) → (Polynomial.eval x (Polynomial.X ^ 3 - 3 * Polynomial.C p * Polynomial.X + 2 * Polynomial.C q) = 0)) :
  Matrix.det ![![2 + a, 1, 1], ![1, 2 + b, 1], ![1, 1, 2 + c]] = -3 * p - 2 * q + 4 :=
by {
  sorry
}

end determinant_expression_l221_221436


namespace country_x_income_l221_221851

theorem country_x_income (I : ℝ) (h1 : I > 40000) (_ : 0.15 * 40000 + 0.20 * (I - 40000) = 8000) : I = 50000 :=
sorry

end country_x_income_l221_221851


namespace not_divisible_l221_221173

theorem not_divisible (n k : ℕ) : ¬ (5 ^ n + 1) ∣ (5 ^ k - 1) :=
sorry

end not_divisible_l221_221173


namespace problem_1_l221_221168

theorem problem_1 :
  (-7/4) - (19/3) - 9/4 + 10/3 = -7 := by
  sorry

end problem_1_l221_221168


namespace bugs_eat_flowers_l221_221072

-- Define the problem conditions
def number_of_bugs : ℕ := 3
def flowers_per_bug : ℕ := 2

-- Define the expected outcome
def total_flowers_eaten : ℕ := 6

-- Prove that total flowers eaten is equal to the product of the number of bugs and flowers per bug
theorem bugs_eat_flowers : number_of_bugs * flowers_per_bug = total_flowers_eaten :=
by
  sorry

end bugs_eat_flowers_l221_221072


namespace quadratic_function_range_l221_221043

noncomputable def quadratic_range : Set ℝ := {y | -2 ≤ y ∧ y < 2}

theorem quadratic_function_range :
  ∀ y : ℝ, 
    (∃ x : ℝ, -2 < x ∧ x < 1 ∧ y = x^2 + 2 * x - 1) ↔ (y ∈ quadratic_range) :=
by
  sorry

end quadratic_function_range_l221_221043


namespace charlie_has_32_cards_l221_221968

variable (Chris_cards Charlie_cards : ℕ)

def chris_has_18_cards : Chris_cards = 18 := sorry
def chris_has_14_fewer_cards_than_charlie : Chris_cards + 14 = Charlie_cards := sorry

theorem charlie_has_32_cards (h18 : Chris_cards = 18) (h14 : Chris_cards + 14 = Charlie_cards) : Charlie_cards = 32 := 
sorry

end charlie_has_32_cards_l221_221968


namespace person_B_correct_probability_l221_221955

-- Define probabilities
def P_A_correct : ℝ := 0.4
def P_A_incorrect : ℝ := 1 - P_A_correct
def P_B_correct_if_A_incorrect : ℝ := 0.5
def P_B_correct : ℝ := P_A_incorrect * P_B_correct_if_A_incorrect

-- Theorem statement
theorem person_B_correct_probability : P_B_correct = 0.3 :=
by
  -- Problem conditions implicitly used in definitions
  sorry

end person_B_correct_probability_l221_221955


namespace original_fraction_l221_221102

def fraction (a b c : ℕ) := 10 * a + b / 10 * c + a

theorem original_fraction (a b c : ℕ) (ha: a < 10) (hb : b < 10) (hc : c < 10) (h : b ≠ c):
  (fraction a b c = b / c) →
  (fraction 6 4 1 = 64 / 16) ∨ (fraction 9 8 4 = 98 / 49) ∨
  (fraction 9 5 1 = 95 / 19) ∨ (fraction 6 5 2 = 65 / 26) :=
sorry

end original_fraction_l221_221102


namespace annual_interest_rate_l221_221192

noncomputable def compound_interest_rate (P A : ℝ) (n t : ℕ) : ℝ :=
  ((A / P) ^ (1 / t)) - 1

-- Define the given parameters
def P : ℝ := 1200
def A : ℝ := 2488.32
def n : ℕ := 1
def t : ℕ := 4

theorem annual_interest_rate : compound_interest_rate P A n t = 0.25 :=
by
  sorry

end annual_interest_rate_l221_221192


namespace include_both_male_and_female_l221_221089

noncomputable def probability_includes_both_genders (total_students male_students female_students selected_students : ℕ) : ℚ :=
  let total_ways := Nat.choose total_students selected_students
  let all_female_ways := Nat.choose female_students selected_students
  (total_ways - all_female_ways) / total_ways

theorem include_both_male_and_female :
  probability_includes_both_genders 6 2 4 4 = 14 / 15 := 
by
  sorry

end include_both_male_and_female_l221_221089


namespace first_nonzero_digit_right_decimal_l221_221654

/--
  To prove that the first nonzero digit to the right of the decimal point of the fraction 1/137 is 9
-/
theorem first_nonzero_digit_right_decimal (n : ℕ) (h1 : n = 137) :
  ∃ d, d = 9 ∧ (∀ k, 10 ^ k * 1 / 137 < 10^(k+1)) → the_first_nonzero_digit_right_of_decimal_is 9 := 
sorry

end first_nonzero_digit_right_decimal_l221_221654


namespace luncheon_cost_l221_221322

variables (s c p : ℝ)

def eq1 := 5 * s + 8 * c + 2 * p = 5.10
def eq2 := 6 * s + 11 * c + 2 * p = 6.45

theorem luncheon_cost (h₁ : 5 * s + 8 * c + 2 * p = 5.10) (h₂ : 6 * s + 11 * c + 2 * p = 6.45) : 
  s + c + p = 1.35 :=
  sorry

end luncheon_cost_l221_221322


namespace number_of_bracelets_l221_221906

-- Define the conditions as constants
def metal_beads_nancy := 40
def pearl_beads_nancy := 60
def crystal_beads_rose := 20
def stone_beads_rose := 40
def beads_per_bracelet := 2

-- Define the number of sets each person can make
def sets_of_metal_beads := metal_beads_nancy / beads_per_bracelet
def sets_of_pearl_beads := pearl_beads_nancy / beads_per_bracelet
def sets_of_crystal_beads := crystal_beads_rose / beads_per_bracelet
def sets_of_stone_beads := stone_beads_rose / beads_per_bracelet

-- Define the theorem to prove
theorem number_of_bracelets : min sets_of_metal_beads (min sets_of_pearl_beads (min sets_of_crystal_beads sets_of_stone_beads)) = 10 := by
  -- Placeholder for the proof
  sorry

end number_of_bracelets_l221_221906


namespace spoons_in_set_l221_221957

def number_of_spoons_in_set (total_cost_set : ℕ) (cost_five_spoons : ℕ) : ℕ :=
  let c := cost_five_spoons / 5
  let s := total_cost_set / c
  s

theorem spoons_in_set (total_cost_set : ℕ) (cost_five_spoons : ℕ) (h1 : total_cost_set = 21) (h2 : cost_five_spoons = 15) : 
  number_of_spoons_in_set total_cost_set cost_five_spoons = 7 :=
by
  sorry

end spoons_in_set_l221_221957


namespace n_five_minus_n_divisible_by_30_l221_221262

theorem n_five_minus_n_divisible_by_30 (n : ℤ) : 30 ∣ (n^5 - n) :=
sorry

end n_five_minus_n_divisible_by_30_l221_221262


namespace gcd_xyz_square_of_diff_l221_221249

theorem gcd_xyz_square_of_diff {x y z : ℕ} 
    (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) : 
    ∃ n : ℕ, Nat.gcd x (Nat.gcd y z) * (y - x) = n ^ 2 :=
by
  sorry

end gcd_xyz_square_of_diff_l221_221249


namespace option_c_correct_l221_221662

theorem option_c_correct (α x1 x2 : ℝ) (hα1 : 0 < α) (hα2 : α < π) (hx1 : 0 < x1) (hx2 : x1 < x2) : 
  (x2 / x1) ^ Real.sin α > 1 :=
by
  sorry

end option_c_correct_l221_221662


namespace product_evaluation_l221_221351

theorem product_evaluation : (3 - 1) * 3 * (3 + 1) * (3 + 2) * (3 + 3) * (3 + 4) = 5040 := by
  -- sorry
  exact rfl  -- This is just a placeholder. The proof would go here.

end product_evaluation_l221_221351


namespace cheryl_tournament_cost_is_1440_l221_221011

noncomputable def cheryl_electricity_bill : ℝ := 800
noncomputable def additional_for_cell_phone : ℝ := 400
noncomputable def cheryl_cell_phone_expenses : ℝ := cheryl_electricity_bill + additional_for_cell_phone
noncomputable def tournament_cost_percentage : ℝ := 0.2
noncomputable def additional_tournament_cost : ℝ := tournament_cost_percentage * cheryl_cell_phone_expenses
noncomputable def total_tournament_cost : ℝ := cheryl_cell_phone_expenses + additional_tournament_cost

theorem cheryl_tournament_cost_is_1440 : total_tournament_cost = 1440 := by
  sorry

end cheryl_tournament_cost_is_1440_l221_221011


namespace min_value_fraction_l221_221628

theorem min_value_fraction (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0)
  (h₂ : ∃ x₀, (2 * x₀ - 2) * (-2 * x₀ + a) = -1) : 
  ∃ a b, a + b = 5 / 2 → a > 0 → b > 0 → 
  (∀ a b, (1 / a + 4 / b) ≥ 18 / 5) :=
by
  sorry

end min_value_fraction_l221_221628


namespace mrs_hilt_total_spent_l221_221605

def kids_ticket_usual_cost : ℕ := 1 -- $1 for 4 tickets
def adults_ticket_usual_cost : ℕ := 2 -- $2 for 3 tickets

def kids_ticket_deal_cost : ℕ := 4 -- $4 for 20 tickets
def adults_ticket_deal_cost : ℕ := 8 -- $8 for 15 tickets

def kids_tickets_purchased : ℕ := 24
def adults_tickets_purchased : ℕ := 18

def total_kids_ticket_cost : ℕ :=
  let kids_deal_tickets := kids_ticket_deal_cost
  let remaining_kids_tickets := kids_ticket_usual_cost
  kids_deal_tickets + remaining_kids_tickets

def total_adults_ticket_cost : ℕ :=
  let adults_deal_tickets := adults_ticket_deal_cost
  let remaining_adults_tickets := adults_ticket_usual_cost
  adults_deal_tickets + remaining_adults_tickets

def total_cost (kids_cost adults_cost : ℕ) : ℕ :=
  kids_cost + adults_cost

theorem mrs_hilt_total_spent : total_cost total_kids_ticket_cost total_adults_ticket_cost = 15 := by
  sorry

end mrs_hilt_total_spent_l221_221605


namespace savings_are_equal_and_correct_l221_221482

-- Definitions of the given conditions
variables (I1 I2 E1 E2 : ℝ)
variables (S1 S2 : ℝ)
variables (rI : ℝ := 5/4) -- ratio of incomes
variables (rE : ℝ := 3/2) -- ratio of expenditures
variables (I1_val : ℝ := 3000) -- P1's income

-- Given conditions
def given_conditions : Prop :=
  I1 = I1_val ∧
  I1 / I2 = rI ∧
  E1 / E2 = rE ∧
  S1 = S2

-- Required proof
theorem savings_are_equal_and_correct (I2_val : I2 = (I1_val * 4/5)) (x : ℝ) (h1 : E1 = 3 * x) (h2 : E2 = 2 * x) (h3 : S1 = 1200) :
  S1 = S2 ∧ S1 = 1200 := by
  sorry

end savings_are_equal_and_correct_l221_221482


namespace floor_value_correct_l221_221859

def calc_floor_value : ℤ :=
  let a := (15 : ℚ) / 8
  let b := a^2
  let c := (225 : ℚ) / 64
  let d := 4
  let e := (19 : ℚ) / 5
  let f := d + e
  ⌊f⌋

theorem floor_value_correct : calc_floor_value = 7 := by
  sorry

end floor_value_correct_l221_221859


namespace solution_set_of_inequality_l221_221763

variable {α : Type*} [LinearOrder α]

def is_decreasing (f : α → α) : Prop :=
  ∀ ⦃x y⦄, x < y → f y < f x

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h_decreasing : is_decreasing f)
  (domain_cond : ∀ x, x ∈ Set.Ioo (-2 : ℝ) 2 → x ∈ Set.Ioo (-2 : ℝ) 2)
  : { x | x > 0 ∧ x < 1 } = { x | f x > f (2 - x) } :=
by {
  sorry
}

end solution_set_of_inequality_l221_221763


namespace john_pays_per_year_l221_221423

-- Define the costs and insurance parameters.
def cost_per_epipen : ℝ := 500
def insurance_coverage : ℝ := 0.75

-- Number of months in a year.
def months_in_year : ℕ := 12

-- Number of months each EpiPen lasts.
def months_per_epipen : ℕ := 6

-- Amount covered by insurance for each EpiPen.
def insurance_amount (cost : ℝ) (coverage : ℝ) : ℝ :=
  cost * coverage

-- Amount John pays after insurance for each EpiPen.
def amount_john_pays_per_epipen (cost : ℝ) (covered: ℝ) : ℝ :=
  cost - covered

-- Number of EpiPens John needs per year.
def epipens_per_year (months_in_year : ℕ) (months_per_epipen : ℕ) : ℕ :=
  months_in_year / months_per_epipen

-- Total amount John pays per year.
def total_amount_john_pays_per_year (amount_per_epipen : ℝ) (epipens_per_year : ℕ) : ℝ :=
  amount_per_epipen * epipens_per_year

-- Theorem to prove the correct answer.
theorem john_pays_per_year :
  total_amount_john_pays_per_year (amount_john_pays_per_epipen cost_per_epipen (insurance_amount cost_per_epipen insurance_coverage)) (epipens_per_year months_in_year months_per_epipen) = 250 := 
by
  sorry

end john_pays_per_year_l221_221423


namespace total_boxes_moved_l221_221513

-- Define a truck's capacity and number of trips
def truck_capacity : ℕ := 4
def trips : ℕ := 218

-- Prove that the total number of boxes is 872
theorem total_boxes_moved : truck_capacity * trips = 872 := by
  sorry

end total_boxes_moved_l221_221513


namespace neither_jia_nor_yi_has_winning_strategy_l221_221590

/-- 
  There are 99 points, each marked with a number from 1 to 99, placed 
  on 99 equally spaced points on a circle. Jia and Yi take turns 
  placing one piece at a time, with Jia going first. The player who 
  first makes the numbers on three consecutive points form an 
  arithmetic sequence wins. Prove that neither Jia nor Yi has a 
  guaranteed winning strategy, and both possess strategies to avoid 
  losing.
-/
theorem neither_jia_nor_yi_has_winning_strategy :
  ∀ (points : Fin 99 → ℕ), -- 99 points on the circle
  (∀ i, 1 ≤ points i ∧ points i ≤ 99) → -- Each point is numbered between 1 and 99
  ¬(∃ (player : Fin 99 → ℕ) (h : ∀ (i : Fin 99), player i ≠ 0 ∧ (player i = 1 ∨ player i = 2)),
    ∃ i : Fin 99, (points i + points (i + 1) + points (i + 2)) / 3 = points i)
:=
by
  sorry

end neither_jia_nor_yi_has_winning_strategy_l221_221590


namespace decimal_to_fraction_l221_221803

theorem decimal_to_fraction (x : ℚ) (h : x = 3.68) : x = 92 / 25 := by
  sorry

end decimal_to_fraction_l221_221803


namespace three_number_product_l221_221814

theorem three_number_product
  (x y z : ℝ)
  (h1 : x + y = 18)
  (h2 : x ^ 2 + y ^ 2 = 220)
  (h3 : z = x - y) :
  x * y * z = 104 * Real.sqrt 29 :=
sorry

end three_number_product_l221_221814


namespace range_of_m_l221_221218

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 + 6 * m * x + m + 8 ≥ 0) ↔ (-8 / 9 ≤ m ∧ m ≤ 1) :=
sorry

end range_of_m_l221_221218


namespace perimeter_of_square_l221_221092

theorem perimeter_of_square (A : ℝ) (hA : A = 400) : exists P : ℝ, P = 80 :=
by
  sorry

end perimeter_of_square_l221_221092


namespace age_ratio_l221_221411

theorem age_ratio (B_age : ℕ) (H1 : B_age = 34) (A_age : ℕ) (H2 : A_age = B_age + 4) :
  (A_age + 10) / (B_age - 10) = 2 :=
by
  sorry

end age_ratio_l221_221411


namespace deepak_present_age_l221_221105

-- We start with the conditions translated into Lean definitions.

variables (R D : ℕ)

-- Condition 1: The ratio between Rahul's and Deepak's ages is 4:3.
def age_ratio := R * 3 = D * 4

-- Condition 2: After 6 years, Rahul's age will be 38 years.
def rahul_future_age := R + 6 = 38

-- The goal is to prove that D = 24 given the above conditions.
theorem deepak_present_age 
  (h1: age_ratio R D) 
  (h2: rahul_future_age R) : D = 24 :=
sorry

end deepak_present_age_l221_221105


namespace value_of_expression_l221_221737

variables {A B C : ℚ}

def conditions (A B C : ℚ) : Prop := A / B = 3 / 2 ∧ B / C = 2 / 5

theorem value_of_expression (h : conditions A B C) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
sorry

end value_of_expression_l221_221737


namespace distribute_6_balls_in_3_boxes_l221_221399

def number_of_ways_to_distribute_balls (balls boxes : Nat) : Nat :=
  boxes ^ balls

theorem distribute_6_balls_in_3_boxes : number_of_ways_to_distribute_balls 6 3 = 729 := by
  sorry

end distribute_6_balls_in_3_boxes_l221_221399


namespace expected_left_handed_l221_221908

theorem expected_left_handed (p : ℚ) (n : ℕ) (h : p = 1/6) (hs : n = 300) : n * p = 50 :=
by 
  -- Proof goes here
  sorry

end expected_left_handed_l221_221908


namespace ratio_a_b_eq_neg_one_fifth_l221_221396

theorem ratio_a_b_eq_neg_one_fifth (x y a b : ℝ) (hb_ne_zero : b ≠ 0) 
    (h1 : 4 * x - 2 * y = a) (h2 : 5 * y - 10 * x = b) : a / b = -1 / 5 :=
by {
  sorry
}

end ratio_a_b_eq_neg_one_fifth_l221_221396


namespace Rover_has_46_spots_l221_221046

theorem Rover_has_46_spots (G C R : ℕ) 
  (h1 : G = 5 * C)
  (h2 : C = (1/2 : ℝ) * R - 5)
  (h3 : G + C = 108) : 
  R = 46 :=
by
  sorry

end Rover_has_46_spots_l221_221046


namespace smallest_x_plus_y_l221_221379

theorem smallest_x_plus_y 
  (x y : ℕ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y) 
  (hxy_neq : x ≠ y)
  (h_eq : (1/x + 1/y = 1/10)) : x + y = 45 :=
sorry

end smallest_x_plus_y_l221_221379


namespace find_zebras_last_year_l221_221282

def zebras_last_year (current : ℕ) (born : ℕ) (died : ℕ) : ℕ :=
  current - born + died

theorem find_zebras_last_year :
  zebras_last_year 725 419 263 = 569 :=
by
  sorry

end find_zebras_last_year_l221_221282


namespace max_candy_remainder_l221_221222

theorem max_candy_remainder (x : ℕ) : x % 11 < 11 ∧ (∀ r : ℕ, r < 11 → x % 11 ≤ r) → x % 11 = 10 := 
sorry

end max_candy_remainder_l221_221222


namespace total_clients_correct_l221_221835

-- Define the number of each type of cars and total cars
def num_cars : ℕ := 12
def num_sedans : ℕ := 4
def num_coupes : ℕ := 4
def num_suvs : ℕ := 4

-- Define the number of selections per car and total selections required
def selections_per_car : ℕ := 3

-- Define the number of clients per type of car
def num_clients_who_like_sedans : ℕ := (num_sedans * selections_per_car) / 2
def num_clients_who_like_coupes : ℕ := (num_coupes * selections_per_car) / 2
def num_clients_who_like_suvs : ℕ := (num_suvs * selections_per_car) / 2

-- Compute total number of clients
def total_clients : ℕ := num_clients_who_like_sedans + num_clients_who_like_coupes + num_clients_who_like_suvs

-- Prove that the total number of clients is 18
theorem total_clients_correct : total_clients = 18 := by
  sorry

end total_clients_correct_l221_221835


namespace not_necessarily_divisible_by_66_l221_221268

theorem not_necessarily_divisible_by_66 (m : ℤ) (h1 : ∃ k : ℤ, m = k * (k + 1) * (k + 2) * (k + 3) * (k + 4)) (h2 : 11 ∣ m) : ¬ (66 ∣ m) :=
sorry

end not_necessarily_divisible_by_66_l221_221268


namespace distinct_real_roots_iff_l221_221038

-- Define f(x, a) := |x^2 - a| - x + 2
noncomputable def f (x a : ℝ) : ℝ := abs (x^2 - a) - x + 2

-- The proposition we need to prove
theorem distinct_real_roots_iff (a : ℝ) (h : 0 < a) : 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ 4 < a :=
by
  sorry

end distinct_real_roots_iff_l221_221038


namespace max_notebooks_l221_221244

-- Definitions based on the conditions
def joshMoney : ℕ := 1050
def notebookCost : ℕ := 75

-- Statement to prove
theorem max_notebooks (x : ℕ) : notebookCost * x ≤ joshMoney → x ≤ 14 := by
  -- Placeholder for the proof
  sorry

end max_notebooks_l221_221244


namespace instantaneous_velocity_at_t3_l221_221633

open Real

noncomputable def displacement (t : ℝ) : ℝ := 4 - 2 * t + t ^ 2

theorem instantaneous_velocity_at_t3 : deriv displacement 3 = 4 := 
by
  sorry

end instantaneous_velocity_at_t3_l221_221633


namespace fruit_basket_count_l221_221225

/-- We have seven identical apples and twelve identical oranges.
    A fruit basket must contain at least one piece of fruit.
    Prove that the number of different fruit baskets we can make
    is 103. -/
theorem fruit_basket_count :
  let apples := 7
  let oranges := 12
  let total_possible_baskets := (apples + 1) * (oranges + 1) - 1
  total_possible_baskets = 103 :=
by
  let apples := 7
  let oranges := 12
  let total_possible_baskets := (apples + 1) * (oranges + 1) - 1
  show total_possible_baskets = 103
  sorry

end fruit_basket_count_l221_221225


namespace common_root_unique_k_l221_221219

theorem common_root_unique_k (k : ℝ) (x : ℝ) 
  (h₁ : x^2 + k * x - 12 = 0) 
  (h₂ : 3 * x^2 - 8 * x - 3 * k = 0) 
  : k = 1 :=
sorry

end common_root_unique_k_l221_221219


namespace initial_packs_l221_221175

def num_invitations_per_pack := 3
def num_friends := 9
def extra_invitations := 3
def total_invitations := num_friends + extra_invitations

theorem initial_packs (h : total_invitations = 12) : (total_invitations / num_invitations_per_pack) = 4 :=
by
  have h1 : total_invitations = 12 := by exact h
  have h2 : num_invitations_per_pack = 3 := by exact rfl
  have H_pack : total_invitations / num_invitations_per_pack = 4 := by sorry
  exact H_pack

end initial_packs_l221_221175


namespace jane_waiting_time_l221_221421

-- Given conditions as constants for readability
def base_coat_drying_time := 2
def first_color_coat_drying_time := 3
def second_color_coat_drying_time := 3
def top_coat_drying_time := 5

-- Total drying time calculation
def total_drying_time := base_coat_drying_time 
                       + first_color_coat_drying_time 
                       + second_color_coat_drying_time 
                       + top_coat_drying_time

-- The theorem to prove
theorem jane_waiting_time : total_drying_time = 13 := 
by
  sorry

end jane_waiting_time_l221_221421


namespace area_difference_triangles_l221_221055

theorem area_difference_triangles
  (A B C F D : Type)
  (angle_FAB_right : true) 
  (angle_ABC_right : true) 
  (AB : Real) (hAB : AB = 5)
  (BC : Real) (hBC : BC = 3)
  (AF : Real) (hAF : AF = 7)
  (area_triangle : A -> B -> C -> Real)
  (angle_bet : A -> D -> F) 
  (angle_bet : B -> D -> C)
  (area_ADF : Real)
  (area_BDC : Real) : (area_ADF - area_BDC = 10) :=
sorry

end area_difference_triangles_l221_221055


namespace solve_inequality_l221_221087

theorem solve_inequality (a x : ℝ) :
  (a > 0 → (a - 1) / a < x ∧ x < 1) ∧ 
  (a = 0 → x < 1) ∧ 
  (a < 0 → x > (a - 1) / a ∨ x < 1) ↔ 
  (ax / (x - 1) < (a - 1) / (x - 1)) :=
sorry

end solve_inequality_l221_221087


namespace smallest_w_l221_221948

theorem smallest_w (w : ℕ) (h1 : 2^4 ∣ 1452 * w) (h2 : 3^3 ∣ 1452 * w) (h3 : 13^3 ∣ 1452 * w) : w = 79132 :=
by
  sorry

end smallest_w_l221_221948


namespace product_remainder_31_l221_221409

theorem product_remainder_31 (m n : ℕ) (h₁ : m % 31 = 7) (h₂ : n % 31 = 12) : (m * n) % 31 = 22 :=
by
  sorry

end product_remainder_31_l221_221409


namespace diagonal_length_of_quadrilateral_l221_221354

theorem diagonal_length_of_quadrilateral 
  (area : ℝ) (m n : ℝ) (d : ℝ) 
  (h_area : area = 210) 
  (h_m : m = 9) 
  (h_n : n = 6) 
  (h_formula : area = 0.5 * d * (m + n)) : 
  d = 28 :=
by 
  sorry

end diagonal_length_of_quadrilateral_l221_221354


namespace negation_of_universal_l221_221699

theorem negation_of_universal (P : ∀ x : ℤ, x^3 < 1) : ¬ (∀ x : ℤ, x^3 < 1) ↔ ∃ x : ℤ, x^3 ≥ 1 :=
by
  sorry

end negation_of_universal_l221_221699


namespace trapezoid_segment_length_l221_221274

theorem trapezoid_segment_length (a b : ℝ) : 
  ∃ x : ℝ, x = Real.sqrt ((a^2 + b^2) / 2) :=
sorry

end trapezoid_segment_length_l221_221274


namespace probability_same_carriage_l221_221930

theorem probability_same_carriage (num_carriages num_people : ℕ) (h1 : num_carriages = 10) (h2 : num_people = 3) : 
  ∃ p : ℚ, p = 7/25 ∧ p = 1 - (10 * 9 * 8) / (10^3) :=
by
  sorry

end probability_same_carriage_l221_221930


namespace typhoon_probabilities_l221_221074

-- Defining the conditions
def probAtLeastOneHit : ℝ := 0.36

-- Defining the events and probabilities
def probOfHit (p : ℝ) := p
def probBothHit (p : ℝ) := p^2

def probAtLeastOne (p : ℝ) : ℝ := p^2 + 2 * p * (1 - p)

-- Defining the variable X as the number of cities hit by the typhoon
def P_X_0 (p : ℝ) : ℝ := (1 - p)^2
def P_X_1 (p : ℝ) : ℝ := 2 * p * (1 - p)
def E_X (p : ℝ) : ℝ := 2 * p

-- Main theorem
theorem typhoon_probabilities :
  ∀ (p : ℝ),
    probAtLeastOne p = probAtLeastOneHit → 
    p = 0.2 ∧ P_X_0 p = 0.64 ∧ P_X_1 p = 0.32 ∧ E_X p = 0.4 :=
by
  intros p h
  sorry

end typhoon_probabilities_l221_221074


namespace conference_min_duration_l221_221827

theorem conference_min_duration : Nat.gcd 9 11 = 1 ∧ Nat.gcd 9 12 = 3 ∧ Nat.gcd 11 12 = 1 ∧ Nat.lcm 9 (Nat.lcm 11 12) = 396 := by
  sorry

end conference_min_duration_l221_221827


namespace sum_of_k_values_l221_221490

-- Conditions
def P (x : ℝ) : ℝ := x^2 - 4 * x + 3
def Q (x k : ℝ) : ℝ := x^2 - 6 * x + k

-- Statement of the mathematical problem
theorem sum_of_k_values (k1 k2 : ℝ) (h1 : P 1 = 0) (h2 : P 3 = 0) 
  (h3 : Q 1 k1 = 0) (h4 : Q 3 k2 = 0) : k1 + k2 = 14 := 
by
  -- Here we would proceed with the proof steps corresponding to the solution
  sorry

end sum_of_k_values_l221_221490


namespace michael_truck_meetings_2_times_l221_221256

/-- Michael walks at a rate of 6 feet per second on a straight path. 
Trash pails are placed every 240 feet along the path. 
A garbage truck traveling at 12 feet per second in the same direction stops for 40 seconds at each pail. 
When Michael passes a pail, he sees the truck, which is 240 feet ahead, just leaving the next pail. 
Prove that Michael and the truck will meet exactly 2 times. -/

def michael_truck_meetings (v_michael v_truck d_pail t_stop init_michael init_truck : ℕ) : ℕ := sorry

theorem michael_truck_meetings_2_times :
  michael_truck_meetings 6 12 240 40 0 240 = 2 := 
  sorry

end michael_truck_meetings_2_times_l221_221256


namespace squares_area_ratios_l221_221850

noncomputable def squareC_area (x : ℝ) : ℝ := x ^ 2
noncomputable def squareD_area (x : ℝ) : ℝ := 3 * x ^ 2
noncomputable def squareE_area (x : ℝ) : ℝ := 6 * x ^ 2

theorem squares_area_ratios (x : ℝ) (h : x ≠ 0) :
  (squareC_area x / squareE_area x = 1 / 36) ∧ (squareD_area x / squareE_area x = 1 / 4) := by
  sorry

end squares_area_ratios_l221_221850


namespace jason_total_spent_l221_221056

theorem jason_total_spent (h_shorts : ℝ) (h_jacket : ℝ) (h1 : h_shorts = 14.28) (h2 : h_jacket = 4.74) : h_shorts + h_jacket = 19.02 :=
by
  rw [h1, h2]
  norm_num

end jason_total_spent_l221_221056


namespace bicycle_saves_time_l221_221895

-- Define the conditions
def time_to_walk : ℕ := 98
def time_saved_by_bicycle : ℕ := 34

-- Prove the question equals the answer
theorem bicycle_saves_time :
  time_saved_by_bicycle = 34 := 
by
  sorry

end bicycle_saves_time_l221_221895


namespace max_students_per_class_l221_221113

-- Definitions used in Lean 4 statement:
def num_students := 920
def seats_per_bus := 71
def num_buses := 16

-- The main statement, showing this is the maximum value such that each class stays together within the given constraints.
theorem max_students_per_class : ∃ k, (∀ k' : ℕ, k' > k → 
  ¬∃ (classes : ℕ), classes * k' + (num_students - classes * k') ≤ seats_per_bus * num_buses ∧ k' <= seats_per_bus) ∧ k = 17 := 
by sorry

end max_students_per_class_l221_221113


namespace three_pow_2023_mod_17_l221_221661

theorem three_pow_2023_mod_17 : (3 ^ 2023) % 17 = 7 := by
  sorry

end three_pow_2023_mod_17_l221_221661


namespace complex_multiplication_l221_221994

variable (i : ℂ)
axiom imaginary_unit : i^2 = -1

theorem complex_multiplication :
  i * (2 * i - 1) = -2 - i :=
  sorry

end complex_multiplication_l221_221994


namespace donation_calculation_l221_221257

/-- Patricia's initial hair length -/
def initial_length : ℕ := 14

/-- Patricia's hair growth -/
def growth_length : ℕ := 21

/-- Desired remaining hair length after donation -/
def remaining_length : ℕ := 12

/-- Calculate the donation length -/
def donation_length (L G R : ℕ) : ℕ := (L + G) - R

-- Theorem stating the donation length required for Patricia to achieve her goal.
theorem donation_calculation : donation_length initial_length growth_length remaining_length = 23 :=
by
  -- Proof omitted
  sorry

end donation_calculation_l221_221257


namespace intersection_complement_l221_221083

open Set

def A : Set ℝ := {x | x < -1 ∨ x > 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem intersection_complement :
  A ∩ (univ \ B) = {x : ℝ | x < -1 ∨ x > 2} :=
by
  sorry

end intersection_complement_l221_221083


namespace time_to_cross_pole_correct_l221_221511

-- Definitions based on problem conditions
def speed_km_per_hr := 90 -- Speed of the train in km/hr
def train_length_meters := 225 -- Length of the train in meters

-- Meters per second conversion factor for km/hr
def km_to_m_conversion := 1000.0 / 3600.0

-- The speed of the train in m/s calculated from the given speed in km/hr
def speed_m_per_s := speed_km_per_hr * km_to_m_conversion

-- Time to cross the pole calculated using distance / speed formula
def time_to_cross_pole (distance speed : ℝ) := distance / speed

-- Theorem to prove the time it takes for the train to cross the pole is 9 seconds
theorem time_to_cross_pole_correct :
  time_to_cross_pole train_length_meters speed_m_per_s = 9 :=
by
  sorry

end time_to_cross_pole_correct_l221_221511


namespace n_five_minus_n_divisible_by_30_l221_221263

theorem n_five_minus_n_divisible_by_30 (n : ℤ) : 30 ∣ (n^5 - n) :=
sorry

end n_five_minus_n_divisible_by_30_l221_221263


namespace log_base_8_of_512_is_3_l221_221979

theorem log_base_8_of_512_is_3 (a b : ℕ) (h1 : a = 2^3) (h2 : b = 2^9) : log b a = 3 :=
sorry

end log_base_8_of_512_is_3_l221_221979


namespace scarves_per_box_l221_221451

theorem scarves_per_box (S M : ℕ) (h1 : S = M) (h2 : 6 * (S + M) = 60) : S = 5 :=
by
  sorry

end scarves_per_box_l221_221451


namespace ratio_doctors_to_lawyers_l221_221094

-- Definitions based on conditions
def average_age_doctors := 35
def average_age_lawyers := 50
def combined_average_age := 40

-- Define variables
variables (d l : ℕ) -- d is number of doctors, l is number of lawyers

-- Hypothesis based on the problem statement
axiom h : (average_age_doctors * d + average_age_lawyers * l) = combined_average_age * (d + l)

-- The theorem we need to prove is the ratio of doctors to lawyers is 2:1
theorem ratio_doctors_to_lawyers : d = 2 * l :=
by sorry

end ratio_doctors_to_lawyers_l221_221094


namespace find_n_l221_221817

theorem find_n (n : ℕ) : (Nat.lcm n 10 = 36) ∧ (Nat.gcd n 10 = 5) → n = 18 :=
by
  -- The proof will be provided here
  sorry

end find_n_l221_221817


namespace sum_of_cubes_ages_l221_221635

theorem sum_of_cubes_ages (d t h : ℕ) 
  (h1 : 4 * d + t = 3 * h) 
  (h2 : 4 * h ^ 2 = 2 * d ^ 2 + t ^ 2) 
  (h3 : Nat.gcd d (Nat.gcd t h) = 1)
  : d ^ 3 + t ^ 3 + h ^ 3 = 155557 :=
sorry

end sum_of_cubes_ages_l221_221635


namespace area_of_regions_l221_221522

noncomputable def g (x : ℝ) := 1 - real.sqrt (1 - (x - 0.5) ^ 2)

theorem area_of_regions :
  let x_intersection := 1 - 1 / real.sqrt 2,
      area_1 := 2 * ∫ x in -1..x_intersection, abs (g x - x),
      area_2 := 2 * ∫ x in x_intersection..1, abs (g x - x) in
  area_1 ≈ 0.64 ∧ area_2 ≈ 0.22 :=
by
  sorry

end area_of_regions_l221_221522


namespace find_a_l221_221801

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := x^4 - 18 * x^3 + ((86 : ℝ)) * x^2 + 200 * x - 1984

-- Define the condition and statement
theorem find_a (α β γ δ : ℝ) (hαβγδ : α * β * γ * δ = -1984)
  (hαβ : α * β = -32) (hγδ : γ * δ = 62) :
  (∀ a : ℝ, a = 86) :=
  sorry

end find_a_l221_221801


namespace max_strings_cut_volleyball_net_l221_221849

-- Define the structure of a volleyball net with 10x20 cells where each cell is divided into 4 triangles.
structure VolleyballNet : Type where
  -- The dimensions of the volleyball net
  rows : ℕ
  cols : ℕ
  -- Number of nodes (vertices + centers)
  nodes : ℕ
  -- Maximum number of strings (edges) connecting neighboring nodes that can be cut without disconnecting the net
  max_cut_without_disconnection : ℕ

-- Define the specific volleyball net in question
def volleyball_net : VolleyballNet := 
  { rows := 10, 
    cols := 20, 
    nodes := (11 * 21) + (10 * 20), -- vertices + center nodes
    max_cut_without_disconnection := 800 
  }

-- The main theorem stating that we can cut these strings without the net falling apart
theorem max_strings_cut_volleyball_net (net : VolleyballNet) 
    (h_dim : net.rows = 10) 
    (h_dim2 : net.cols = 20) :
  net.max_cut_without_disconnection = 800 :=
sorry -- The proof is omitted

end max_strings_cut_volleyball_net_l221_221849


namespace train_length_l221_221144

noncomputable def speed_km_hr : ℝ := 60
noncomputable def time_sec : ℝ := 3
noncomputable def speed_m_s := speed_km_hr * 1000 / 3600
noncomputable def length_of_train := speed_m_s * time_sec

theorem train_length :
  length_of_train = 50.01 := by
  sorry

end train_length_l221_221144


namespace decimal_representation_of_7_div_12_l221_221524

theorem decimal_representation_of_7_div_12 : (7 / 12 : ℚ) = 0.58333333 := 
sorry

end decimal_representation_of_7_div_12_l221_221524


namespace sugar_left_correct_l221_221608

-- Define the total amount of sugar bought by Pamela
def total_sugar : ℝ := 9.8

-- Define the amount of sugar spilled by Pamela
def spilled_sugar : ℝ := 5.2

-- Define the amount of sugar left after spilling
def sugar_left : ℝ := total_sugar - spilled_sugar

-- State that the amount of sugar left should be equivalent to the correct answer
theorem sugar_left_correct : sugar_left = 4.6 :=
by
  sorry

end sugar_left_correct_l221_221608


namespace square_three_times_side_length_l221_221012

theorem square_three_times_side_length (a : ℝ) : 
  ∃ s, s = a * Real.sqrt 3 ∧ s ^ 2 = 3 * a ^ 2 := 
by 
  sorry

end square_three_times_side_length_l221_221012


namespace total_weight_of_carrots_and_cucumbers_is_875_l221_221797

theorem total_weight_of_carrots_and_cucumbers_is_875 :
  ∀ (carrots : ℕ) (cucumbers : ℕ),
    carrots = 250 →
    cucumbers = (5 * carrots) / 2 →
    carrots + cucumbers = 875 := 
by
  intros carrots cucumbers h_carrots h_cucumbers
  rw [h_carrots, h_cucumbers]
  sorry

end total_weight_of_carrots_and_cucumbers_is_875_l221_221797


namespace solve_system_of_equations_l221_221779

theorem solve_system_of_equations (x y z t : ℤ) :
  (3 * x - 2 * y + 4 * z + 2 * t = 19) ∧ (5 * x + 6 * y - 2 * z + 3 * t = 23) →
  (x = 16 * z - 18 * y - 11) ∧ (t = 28 * y - 26 * z + 26) :=
by {
  sorry
}

end solve_system_of_equations_l221_221779


namespace tan_neg_3pi_over_4_eq_one_l221_221017

theorem tan_neg_3pi_over_4_eq_one : Real.tan (-3 * Real.pi / 4) = 1 := 
by 
  sorry

end tan_neg_3pi_over_4_eq_one_l221_221017


namespace negation_of_all_men_are_tall_l221_221788

variable {α : Type}
variable (man : α → Prop) (tall : α → Prop)

theorem negation_of_all_men_are_tall :
  (¬ ∀ x, man x → tall x) ↔ ∃ x, man x ∧ ¬ tall x :=
sorry

end negation_of_all_men_are_tall_l221_221788


namespace avery_egg_cartons_filled_l221_221001

-- Definitions (conditions identified in step a)
def total_chickens : ℕ := 20
def eggs_per_chicken : ℕ := 6
def eggs_per_carton : ℕ := 12

-- Theorem statement (equivalent to the problem statement)
theorem avery_egg_cartons_filled : (total_chickens * eggs_per_chicken) / eggs_per_carton = 10 :=
by
  -- Proof omitted; sorry used to denote unfinished proof
  sorry

end avery_egg_cartons_filled_l221_221001


namespace distance_between_parallel_lines_l221_221355

theorem distance_between_parallel_lines
  (line1 : ∀ (x y : ℝ), 3*x - 2*y - 1 = 0)
  (line2 : ∀ (x y : ℝ), 3*x - 2*y + 1 = 0) :
  ∃ d : ℝ, d = (2 * Real.sqrt 13) / 13 :=
by
  sorry

end distance_between_parallel_lines_l221_221355


namespace initial_yellow_hard_hats_count_l221_221891

noncomputable def initial_yellow_hard_hats := 24

theorem initial_yellow_hard_hats_count
  (initial_pink: ℕ)
  (initial_green: ℕ)
  (carl_pink: ℕ)
  (john_pink: ℕ)
  (john_green: ℕ)
  (total_remaining: ℕ)
  (remaining_pink: ℕ)
  (remaining_green: ℕ)
  (initial_yellow: ℕ) :
  initial_pink = 26 →
  initial_green = 15 →
  carl_pink = 4 →
  john_pink = 6 →
  john_green = 2 * john_pink →
  total_remaining = 43 →
  remaining_pink = initial_pink - carl_pink - john_pink →
  remaining_green = initial_green - john_green →
  initial_yellow = total_remaining - remaining_pink - remaining_green →
  initial_yellow = initial_yellow_hard_hats :=
by
  intros
  sorry

end initial_yellow_hard_hats_count_l221_221891


namespace selling_price_correct_l221_221562

-- Define the conditions
def boxes := 3
def face_masks_per_box := 20
def cost_price := 15  -- in dollars
def profit := 15      -- in dollars

-- Define the total number of face masks
def total_face_masks := boxes * face_masks_per_box

-- Define the total amount he wants after selling all face masks
def total_amount := cost_price + profit

-- Prove that the selling price per face mask is $0.50
noncomputable def selling_price_per_face_mask : ℚ :=
  total_amount / total_face_masks

theorem selling_price_correct : selling_price_per_face_mask = 0.50 := by
  sorry

end selling_price_correct_l221_221562


namespace find_m_l221_221926

theorem find_m 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h_arith_seq : ∀ n, a (n - 1) + a (n + 1) = 2 * a n)
  (h_cond1 : a (m - 1) + a (m + 1) - a m ^ 2 = 0)
  (h_cond2 : S (2 * m - 1) = 38) 
  : m = 10 :=
sorry

end find_m_l221_221926


namespace find_g_neg2_l221_221872

-- Definitions of the conditions
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x 

variables (f : ℝ → ℝ) (g : ℝ → ℝ)
variables (h_even_f : even_function f)
variables (h_g_def : ∀ x, g x = f x + x^3)
variables (h_g_2 : g 2 = 10)

-- Statement to prove
theorem find_g_neg2 : g (-2) = -6 :=
sorry

end find_g_neg2_l221_221872


namespace incorrect_statement_D_l221_221998

noncomputable def f : ℝ → ℝ := sorry

axiom A1 : ∃ x : ℝ, f x ≠ 0
axiom A2 : ∀ x : ℝ, f (x + 1) = -f (2 - x)
axiom A3 : ∀ x : ℝ, f (x + 3) = f (x - 3)

theorem incorrect_statement_D :
  ¬ (∀ x : ℝ, f (3 + x) + f (3 - x) = 0) :=
sorry

end incorrect_statement_D_l221_221998


namespace dice_probability_l221_221448

def first_die_prob : ℚ := 3 / 8
def second_die_prob : ℚ := 3 / 4
def combined_prob : ℚ := first_die_prob * second_die_prob

theorem dice_probability :
  combined_prob = 9 / 32 :=
by
  -- Here we write the proof steps.
  sorry

end dice_probability_l221_221448


namespace find_radius_l221_221235

-- Define the given values
def arc_length : ℝ := 4
def central_angle : ℝ := 2

-- We need to prove this statement
theorem find_radius (radius : ℝ) : arc_length = radius * central_angle → radius = 2 := 
by
  sorry

end find_radius_l221_221235


namespace axis_of_symmetry_of_f_l221_221919

noncomputable def f (x : ℝ) : ℝ := (x - 3) * (x + 1)

theorem axis_of_symmetry_of_f : (axis_of_symmetry : ℝ) = -1 :=
by
  sorry

end axis_of_symmetry_of_f_l221_221919


namespace number_of_solutions_decrease_l221_221483

-- Define the conditions and the main theorem
theorem number_of_solutions_decrease (a : ℝ) :
  (∃ x y : ℝ, x^2 - y^2 = 0 ∧ (x - a)^2 + y^2 = 1) → 
  (∀ x y : ℝ, x^2 - x^2 = 0 ∧ (x - a)^2 + x^2 = 1) →
  a = 1 ∨ a = -1 := 
sorry

end number_of_solutions_decrease_l221_221483


namespace math_proof_problem_l221_221768

/-- Given three real numbers a, b, and c such that a ≥ b ≥ 1 ≥ c ≥ 0 and a + b + c = 3.

Part (a): Prove that 2 ≤ ab + bc + ca ≤ 3.
Part (b): Prove that (24 / (a^3 + b^3 + c^3)) + (25 / (ab + bc + ca)) ≥ 14.
--/
theorem math_proof_problem (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ 1) (h3 : 1 ≥ c)
  (h4 : c ≥ 0) (h5 : a + b + c = 3) :
  (2 ≤ a * b + b * c + c * a ∧ a * b + b * c + c * a ≤ 3) ∧ 
  (24 / (a^3 + b^3 + c^3) + 25 / (a * b + b * c + c * a) ≥ 14) 
  :=
by
  sorry

end math_proof_problem_l221_221768


namespace f_of_x_l221_221903

theorem f_of_x (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x-1) = 3*x - 1) : ∀ x : ℤ, f x = 3*x + 2 :=
by
  sorry

end f_of_x_l221_221903


namespace runs_twice_l221_221412

-- Definitions of the conditions
def game_count : ℕ := 6
def runs_one : ℕ := 1
def runs_five : ℕ := 5
def average_runs : ℕ := 4

-- Assuming the number of runs scored twice is x
variable (x : ℕ)

-- Definition of total runs scored based on the conditions
def total_runs : ℕ := runs_one + 2 * x + 3 * runs_five

-- Statement to prove the number of runs scored twice
theorem runs_twice :
  (total_runs x) / game_count = average_runs → x = 4 :=
by
  sorry

end runs_twice_l221_221412


namespace club_leadership_team_selection_l221_221954

theorem club_leadership_team_selection :
  let n := 20 in let k := 2 in let m := 1 in 
  (nat.choose n k) * (nat.choose (n - k) m) = 3420 :=
  by sorry

end club_leadership_team_selection_l221_221954


namespace Cheryl_golf_tournament_cost_l221_221010

theorem Cheryl_golf_tournament_cost :
  let electricity_bill := 800 in
  let cell_phone_expenses := electricity_bill + 400 in
  let tournament_extra_cost := 0.20 * cell_phone_expenses in
  let total_tournament_cost := cell_phone_expenses + tournament_extra_cost in
  total_tournament_cost = 1440 :=
by
  sorry

end Cheryl_golf_tournament_cost_l221_221010


namespace led_message_count_l221_221640

theorem led_message_count : 
  let n := 7
  let colors := 2
  let lit_leds := 3
  let non_adjacent_combinations := 10
  (non_adjacent_combinations * (colors ^ lit_leds)) = 80 :=
by
  sorry

end led_message_count_l221_221640


namespace f_2012_l221_221853

noncomputable def f : ℝ → ℝ := sorry -- provided as a 'sorry' to be determined

axiom odd_function (hf : ℝ → ℝ) : ∀ x : ℝ, hf (-x) = -hf (x)

axiom f_shift : ∀ x : ℝ, f (x + 3) = -f (x)
axiom f_one : f 1 = 2

theorem f_2012 : f 2012 = 2 :=
by
  -- proofs would go here, but 'sorry' is enough to define the theorem statement
  sorry

end f_2012_l221_221853


namespace kiyiv_first_problem_kiyiv_second_problem_l221_221460

/-- Let x and y be positive real numbers such that xy ≥ 1.
Prove that x^3 + y^3 + 4xy ≥ x^2 + y^2 + x + y + 2. -/
theorem kiyiv_first_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 1 ≤ x * y) :
  x^3 + y^3 + 4 * x * y ≥ x^2 + y^2 + x + y + 2 :=
sorry

/-- Let x and y be positive real numbers such that xy ≥ 1.
Prove that 2(x^3 + y^3 + xy + x + y) ≥ 5(x^2 + y^2). -/
theorem kiyiv_second_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 1 ≤ x * y) :
  2 * (x^3 + y^3 + x * y + x + y) ≥ 5 * (x^2 + y^2) :=
sorry

end kiyiv_first_problem_kiyiv_second_problem_l221_221460


namespace findDivisor_l221_221158

def addDivisorProblem : Prop :=
  ∃ d : ℕ, ∃ n : ℕ, n = 172835 + 21 ∧ d ∣ n ∧ d = 21

theorem findDivisor : addDivisorProblem :=
by
  sorry

end findDivisor_l221_221158


namespace find_n_values_l221_221353

theorem find_n_values (n : ℕ) (h1 : 0 < n) : 
  (∃ (a : ℕ), n * 2^n + 1 = a * a) ↔ (n = 2 ∨ n = 3) := 
by
  sorry

end find_n_values_l221_221353


namespace smallest_sum_of_xy_l221_221373

theorem smallest_sum_of_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y)
  (hcond : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 10) : x + y = 45 :=
sorry

end smallest_sum_of_xy_l221_221373


namespace isosceles_right_triangle_square_ratio_l221_221688

noncomputable def x : ℝ := 1 / 2
noncomputable def y : ℝ := Real.sqrt 2 / 2

theorem isosceles_right_triangle_square_ratio :
  x / y = Real.sqrt 2 := by
  sorry

end isosceles_right_triangle_square_ratio_l221_221688


namespace find_initial_population_l221_221146

theorem find_initial_population
  (birth_rate : ℕ)
  (death_rate : ℕ)
  (net_growth_rate_percent : ℝ)
  (net_growth_rate_per_person : ℕ)
  (h1 : birth_rate = 32)
  (h2 : death_rate = 11)
  (h3 : net_growth_rate_percent = 2.1)
  (h4 : net_growth_rate_per_person = birth_rate - death_rate)
  (h5 : (net_growth_rate_per_person : ℝ) / 100 = net_growth_rate_percent / 100) :
  P = 1000 :=
by
  sorry

end find_initial_population_l221_221146


namespace decreasing_interval_ln_quadratic_l221_221281

theorem decreasing_interval_ln_quadratic :
  ∀ x : ℝ, (x < 1 ∨ x > 3) → (∀ a b : ℝ, (a ≤ b) → (a < 1 ∨ a > 3) → (b < 1 ∨ b > 3) → (a ≤ x ∧ x ≤ b → (x^2 - 4 * x + 3) ≥ (b^2 - 4 * b + 3))) :=
by
  sorry

end decreasing_interval_ln_quadratic_l221_221281


namespace probability_hits_10_ring_l221_221678

-- Definitions based on conditions
def total_shots : ℕ := 10
def hits_10_ring : ℕ := 2

-- Theorem stating the question and answer equivalence.
theorem probability_hits_10_ring : (hits_10_ring : ℚ) / total_shots = 0.2 := by
  -- We are skipping the proof with 'sorry'
  sorry

end probability_hits_10_ring_l221_221678


namespace smallest_c_d_sum_l221_221034

theorem smallest_c_d_sum : ∃ (c d : ℕ), 2^12 * 7^6 = c^d ∧  (∀ (c' d' : ℕ), 2^12 * 7^6 = c'^d'  → (c + d) ≤ (c' + d')) ∧ c + d = 21954 := by
  sorry

end smallest_c_d_sum_l221_221034


namespace ratio_problem_l221_221731

theorem ratio_problem 
  (A B C : ℚ) 
  (h : A / B = 3 / 2 ∧ B / C = 2 / 5 ∧ A / C = 3 / 5) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by 
  sorry

end ratio_problem_l221_221731


namespace sandwiches_per_day_l221_221987

theorem sandwiches_per_day (S : ℕ) 
  (h1 : ∀ n, n = 4 * S)
  (h2 : 7 * 4 * S = 280) : S = 10 := 
by
  sorry

end sandwiches_per_day_l221_221987


namespace factor_expression_l221_221352

theorem factor_expression (z : ℂ) : 
  75 * z^12 + 162 * z^24 + 27 = 3 * (9 + z^12 * (25 + 54 * z^12)) :=
sorry

end factor_expression_l221_221352


namespace john_annual_payment_l221_221426

open Real

-- Definitions extracted from the problem:
def epipen_cost : ℝ := 500
def insurance_coverage : ℝ := 0.75
def epipen_frequency_per_year : ℕ := 2
def john_payment_per_epipen : ℝ := epipen_cost * (1 - insurance_coverage)

-- The statement to be proved:
theorem john_annual_payment :
  john_payment_per_epipen * epipen_frequency_per_year = 250 :=
by
  sorry

end john_annual_payment_l221_221426


namespace rectangular_solid_surface_area_l221_221310

theorem rectangular_solid_surface_area
  (length : ℕ) (width : ℕ) (depth : ℕ)
  (h_length : length = 9) (h_width : width = 8) (h_depth : depth = 5) :
  2 * (length * width + width * depth + length * depth) = 314 := 
  by
  sorry

end rectangular_solid_surface_area_l221_221310


namespace viewers_watching_program_A_l221_221450

theorem viewers_watching_program_A (T : ℕ) (hT : T = 560) (x : ℕ)
  (h_ratio : 1 * x + (2 * x - x) + (3 * x - x) = T) : 2 * x = 280 :=
by
  -- by solving the given equation, we find x = 140
  -- substituting x = 140 in 2 * x gives 2 * x = 280
  sorry

end viewers_watching_program_A_l221_221450


namespace num_trucks_l221_221120

variables (T : ℕ) (num_cars : ℕ := 13) (total_wheels : ℕ := 100) (wheels_per_vehicle : ℕ := 4)

theorem num_trucks :
  (num_cars * wheels_per_vehicle + T * wheels_per_vehicle = total_wheels) -> T = 12 :=
by
  intro h
  -- skipping the proof implementation
  sorry

end num_trucks_l221_221120


namespace pow_2023_eq_one_or_neg_one_l221_221772

theorem pow_2023_eq_one_or_neg_one (x : ℂ) (h : (x - 1) * (x^5 + x^4 + x^3 + x^2 + x + 1) = 0) : 
  x^2023 = 1 ∨ x^2023 = -1 := 
by 
{
  sorry
}

end pow_2023_eq_one_or_neg_one_l221_221772


namespace ratio_a_to_d_l221_221885

theorem ratio_a_to_d (a b c d : ℕ) 
  (h1 : a * 4 = b * 3) 
  (h2 : b * 9 = c * 7) 
  (h3 : c * 7 = d * 5) : 
  a * 3 = d := 
sorry

end ratio_a_to_d_l221_221885


namespace min_absolute_sum_value_l221_221659

def absolute_sum (x : ℝ) : ℝ :=
  abs (x + 3) + abs (x + 6) + abs (x + 7)

theorem min_absolute_sum_value : ∃ x, absolute_sum x = 4 :=
sorry

end min_absolute_sum_value_l221_221659


namespace largest_int_less_100_remainder_5_l221_221539

theorem largest_int_less_100_remainder_5 (a : ℕ) (h1 : a < 100) (h2 : a % 9 = 5) :
  a = 95 :=
sorry

end largest_int_less_100_remainder_5_l221_221539


namespace odd_sol_exists_l221_221076

theorem odd_sol_exists (n : ℕ) (hn : n > 0) : 
  ∃ (x_n y_n : ℕ), (x_n % 2 = 1) ∧ (y_n % 2 = 1) ∧ (x_n^2 + 7 * y_n^2 = 2^n) := 
sorry

end odd_sol_exists_l221_221076


namespace smallest_n_modulo_l221_221138

theorem smallest_n_modulo :
  ∃ (n : ℕ), 0 < n ∧ 1031 * n % 30 = 1067 * n % 30 ∧ ∀ (m : ℕ), 0 < m ∧ 1031 * m % 30 = 1067 * m % 30 → n ≤ m :=
by
  sorry

end smallest_n_modulo_l221_221138


namespace mean_of_three_l221_221782

theorem mean_of_three (a b c : ℝ) (h : (a + b + c + 105) / 4 = 92) : (a + b + c) / 3 = 87.7 :=
by
  sorry

end mean_of_three_l221_221782


namespace maximum_value_cosine_sine_combination_l221_221186

noncomputable def max_cosine_sine_combination : Real :=
  let g (θ : Real) := (Real.cos (θ / 2)) * (1 + Real.sin θ)
  have h₁ : ∃ θ : Real, -Real.pi / 2 < θ ∧ θ < Real.pi / 2 :=
    sorry -- Existence of such θ is trivial
  Real.sqrt 2

theorem maximum_value_cosine_sine_combination :
  ∀ θ : Real, -Real.pi / 2 < θ ∧ θ < Real.pi / 2 →
  (Real.cos (θ / 2)) * (1 + Real.sin θ) ≤ Real.sqrt 2 :=
by
  intros θ h
  let y := (Real.cos (θ / 2)) * (1 + Real.sin θ)
  have hy : y ≤ Real.sqrt 2 := sorry
  exact hy

end maximum_value_cosine_sine_combination_l221_221186


namespace dan_remaining_marbles_l221_221345

-- Define the initial number of marbles Dan has
def initial_marbles : ℕ := 64

-- Define the number of marbles Dan gave to Mary
def marbles_given : ℕ := 14

-- Define the number of remaining marbles
def remaining_marbles : ℕ := initial_marbles - marbles_given

-- State the theorem
theorem dan_remaining_marbles : remaining_marbles = 50 := by
  -- Placeholder for the proof
  sorry

end dan_remaining_marbles_l221_221345


namespace probability_at_least_two_meters_l221_221507

def rope_length : ℝ := 6
def num_nodes : ℕ := 5
def equal_parts : ℕ := 6
def min_length : ℝ := 2

theorem probability_at_least_two_meters (h_rope_division : rope_length / equal_parts = 1) :
  let favorable_cuts := 3
  let total_cuts := num_nodes
  (favorable_cuts : ℝ) / total_cuts = 3 / 5 :=
by
  sorry

end probability_at_least_two_meters_l221_221507


namespace compare_exponents_l221_221545

noncomputable def a : ℝ := 20 ^ 22
noncomputable def b : ℝ := 21 ^ 21
noncomputable def c : ℝ := 22 ^ 20

theorem compare_exponents : a > b ∧ b > c :=
by {
  sorry
}

end compare_exponents_l221_221545


namespace polynomial_binomial_square_l221_221229

theorem polynomial_binomial_square (b : ℝ) : 
  (∃ c : ℝ, (3*X + c)^2 = 9*X^2 - 24*X + b) → b = 16 :=
by
  sorry

end polynomial_binomial_square_l221_221229


namespace max_students_per_class_l221_221112

-- Definitions used in Lean 4 statement:
def num_students := 920
def seats_per_bus := 71
def num_buses := 16

-- The main statement, showing this is the maximum value such that each class stays together within the given constraints.
theorem max_students_per_class : ∃ k, (∀ k' : ℕ, k' > k → 
  ¬∃ (classes : ℕ), classes * k' + (num_students - classes * k') ≤ seats_per_bus * num_buses ∧ k' <= seats_per_bus) ∧ k = 17 := 
by sorry

end max_students_per_class_l221_221112


namespace decimal_representation_of_7_div_12_l221_221525

theorem decimal_representation_of_7_div_12 : (7 / 12 : ℚ) = 0.58333333 := 
sorry

end decimal_representation_of_7_div_12_l221_221525


namespace polygon_area_144_l221_221240

-- Given definitions
def polygon (n : ℕ) : Prop := -- definition to capture n squares arrangement
  n = 36

def is_perpendicular (sides : ℕ) : Prop := -- every pair of adjacent sides is perpendicular
  sides = 4

def all_sides_congruent (length : ℕ) : Prop := -- all sides have the same length
  true

def total_perimeter (perimeter : ℕ) : Prop := -- total perimeter of the polygon
  perimeter = 72

-- The side length s leading to polygon's perimeter
def side_length (s perimeter : ℕ) : Prop :=
  perimeter = 36 * s / 2 

-- Prove the area of polygon is 144
theorem polygon_area_144 (n sides length perimeter s: ℕ) 
    (h1 : polygon n) 
    (h2 : is_perpendicular sides) 
    (h3 : all_sides_congruent length) 
    (h4 : total_perimeter perimeter) 
    (h5 : side_length s perimeter) : 
    n * s * s = 144 := 
sorry

end polygon_area_144_l221_221240


namespace present_age_of_son_l221_221324

-- Define variables for the current ages of the son and the man (father).
variables (S M : ℕ)

-- Define the conditions:
-- The man is 35 years older than his son.
def condition1 : Prop := M = S + 35

-- In two years, the man's age will be twice the age of his son.
def condition2 : Prop := M + 2 = 2 * (S + 2)

-- The theorem that we need to prove:
theorem present_age_of_son : condition1 S M ∧ condition2 S M → S = 33 :=
by
  -- Add sorry to skip the proof.
  sorry

end present_age_of_son_l221_221324


namespace log_base_8_of_512_l221_221976

theorem log_base_8_of_512 : log 8 512 = 3 := by
  have h₁ : 8 = 2^3 := by rfl
  have h₂ : 512 = 2^9 := by rfl
  rw [h₂, h₁]
  sorry

end log_base_8_of_512_l221_221976


namespace total_cost_is_160_l221_221050

-- Define the costs of each dress
def CostOfPaulineDress := 30
def CostOfJeansDress := CostOfPaulineDress - 10
def CostOfIdasDress := CostOfJeansDress + 30
def CostOfPattysDress := CostOfIdasDress + 10

-- The total cost
def TotalCost := CostOfPaulineDress + CostOfJeansDress + CostOfIdasDress + CostOfPattysDress

-- Prove the total cost is $160
theorem total_cost_is_160 : TotalCost = 160 := by
  -- skipping the proof steps
  sorry

end total_cost_is_160_l221_221050


namespace avg_weight_B_correct_l221_221796

-- Definitions of the conditions
def students_A : ℕ := 24
def students_B : ℕ := 16
def avg_weight_A : ℝ := 40
def avg_weight_class : ℝ := 38

-- Definition of the total weight calculation for sections A and B
def total_weight_A : ℝ := students_A * avg_weight_A
def total_weight_class : ℝ := (students_A + students_B) * avg_weight_class

-- Defining the average weight of section B as the unknown to be proven
noncomputable def avg_weight_B : ℝ := 35

-- The theorem to prove that the average weight of section B is 35 kg
theorem avg_weight_B_correct : 
  total_weight_A + students_B * avg_weight_B = total_weight_class :=
by
  sorry

end avg_weight_B_correct_l221_221796


namespace number_of_sarees_l221_221104

-- Define variables representing the prices of one saree and one shirt
variables (X S T : ℕ)

-- Define the conditions 
def condition1 := X * S + 4 * T = 1600
def condition2 := S + 6 * T = 1600
def condition3 := 12 * T = 2400

-- The proof problem (statement only, without proof)
theorem number_of_sarees (X S T : ℕ) (h1 : condition1 X S T) (h2 : condition2 S T) (h3 : condition3 T) : X = 2 := by
  sorry

end number_of_sarees_l221_221104


namespace probability_ratio_l221_221534

theorem probability_ratio (bins balls n1 n2 n3 n4 : Nat)
  (h_balls : balls = 18)
  (h_bins : bins = 4)
  (scenarioA : n1 = 6 ∧ n2 = 2 ∧ n3 = 5 ∧ n4 = 5)
  (scenarioB : n1 = 5 ∧ n2 = 5 ∧ n3 = 4 ∧ n4 = 4) :
  ((Nat.choose bins 1) * (Nat.choose (bins - 1) 1) * Nat.factorial balls /
  (Nat.factorial n1 * Nat.factorial n2 * Nat.factorial n3 * Nat.factorial n4)) /
  ((Nat.choose bins 2) * Nat.factorial balls /
  (Nat.factorial n1 * Nat.factorial n2 * Nat.factorial n3 * Nat.factorial n4)) = 10 / 3 :=
by
  sorry

end probability_ratio_l221_221534


namespace bears_on_each_shelf_l221_221333

theorem bears_on_each_shelf (initial_bears : ℕ) (additional_bears : ℕ) (shelves : ℕ) (total_bears : ℕ) (bears_per_shelf : ℕ) :
  initial_bears = 5 → additional_bears = 7 → shelves = 2 → total_bears = initial_bears + additional_bears → bears_per_shelf = total_bears / shelves → bears_per_shelf = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end bears_on_each_shelf_l221_221333


namespace solve_eq1_solve_eq2_l221_221615

variable (x : ℝ)

theorem solve_eq1 : (2 * x - 3 * (2 * x - 3) = x + 4) → (x = 1) :=
by
  intro h
  sorry

theorem solve_eq2 : ((3 / 4 * x - 1 / 4) - 1 = (5 / 6 * x - 7 / 6)) → (x = -1) :=
by
  intro h
  sorry

end solve_eq1_solve_eq2_l221_221615


namespace probability_first_spade_second_king_l221_221933

/--
In a standard deck of 52 cards, the probability of drawing the first card as a ♠ and the second card as a king is 1/52.
-/
theorem probability_first_spade_second_king : 
  let deck_size := 52 in
  let hearts_count := 13 in
  let kings_count := 4 in
  let prob := (1 / deck_size : ℚ) * (kings_count / (deck_size - 1)) + ((hearts_count - 1) / deck_size) * (kings_count / (deck_size - 1)) 
  in 
  prob = 1 / deck_size :=
by
  sorry

end probability_first_spade_second_king_l221_221933


namespace minimum_value_of_expression_l221_221211

noncomputable def min_value (a b : ℝ) : ℝ :=
  a^2 + (1 / (a * b)) + (1 / (a * (a - b)))

theorem minimum_value_of_expression (a b : ℝ) (h1 : a > b) (h2 : b > 0) : min_value a b >= 4 := by
  sorry

end minimum_value_of_expression_l221_221211


namespace bus_driver_total_compensation_l221_221675

-- Define the regular rate
def regular_rate : ℝ := 16

-- Define the number of regular hours
def regular_hours : ℕ := 40

-- Define the overtime rate as 75% higher than the regular rate
def overtime_rate : ℝ := regular_rate * 1.75

-- Define the total hours worked in the week
def total_hours_worked : ℕ := 48

-- Calculate the overtime hours
def overtime_hours : ℕ := total_hours_worked - regular_hours

-- Calculate the total compensation
def total_compensation : ℝ :=
  (regular_hours * regular_rate) + (overtime_hours * overtime_rate)

-- Theorem to prove that the total compensation is $864
theorem bus_driver_total_compensation : total_compensation = 864 := by
  -- Proof is omitted
  sorry

end bus_driver_total_compensation_l221_221675


namespace sum_two_angles_greater_third_l221_221258

-- Definitions of the angles and the largest angle condition
variables {P A B C} -- Points defining the trihedral angle
variables {α β γ : ℝ} -- Angles α, β, γ
variables (h1 : γ ≥ α) (h2 : γ ≥ β)

-- Statement of the theorem
theorem sum_two_angles_greater_third (P A B C : Type*) (α β γ : ℝ)
  (h1 : γ ≥ α) (h2 : γ ≥ β) : α + β > γ :=
sorry  -- Proof is omitted

end sum_two_angles_greater_third_l221_221258


namespace quadratic_solutions_l221_221287

theorem quadratic_solutions (x : ℝ) : x * (x - 1) = 1 - x ↔ x = 1 ∨ x = -1 :=
by
  sorry

end quadratic_solutions_l221_221287


namespace value_of_expression_l221_221735

variables {A B C : ℚ}

def conditions (A B C : ℚ) : Prop := A / B = 3 / 2 ∧ B / C = 2 / 5

theorem value_of_expression (h : conditions A B C) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
sorry

end value_of_expression_l221_221735


namespace find_xy_l221_221575

-- Define the conditions as constants for clarity
def condition1 (x : ℝ) : Prop := 0.60 / x = 6 / 2
def condition2 (x y : ℝ) : Prop := x / y = 8 / 12

theorem find_xy (x y : ℝ) (hx : condition1 x) (hy : condition2 x y) : 
  x = 0.20 ∧ y = 0.30 :=
by
  sorry

end find_xy_l221_221575


namespace initial_worth_is_30_l221_221610

-- Definitions based on conditions
def numberOfCoinsLeft := 2
def amountLeft := 12

-- Definition of the value of each gold coin based on amount left and number of coins left
def valuePerCoin : ℕ := amountLeft / numberOfCoinsLeft

-- Define the total worth of sold coins
def soldCoinsWorth (coinsSold : ℕ) : ℕ := coinsSold * valuePerCoin

-- The total initial worth of Roman's gold coins
def totalInitialWorth : ℕ := amountLeft + soldCoinsWorth 3

-- The proof goal
theorem initial_worth_is_30 : totalInitialWorth = 30 :=
by
  sorry

end initial_worth_is_30_l221_221610


namespace complement_of_A_cap_B_l221_221251

def set_A (x : ℝ) : Prop := x ≤ -4 ∨ x ≥ 2
def set_B (x : ℝ) : Prop := |x - 1| ≤ 3

def A_cap_B (x : ℝ) : Prop := set_A x ∧ set_B x

def complement_A_cap_B (x : ℝ) : Prop := ¬A_cap_B x

theorem complement_of_A_cap_B :
  {x : ℝ | complement_A_cap_B x} = {x : ℝ | x < 2 ∨ x > 4} :=
by
  sorry

end complement_of_A_cap_B_l221_221251


namespace two_point_form_eq_l221_221108

theorem two_point_form_eq (x y : ℝ) : 
  let A := (5, 6)
  let B := (-1, 2)
  (y - 6) / (2 - 6) = (x - 5) / (-1 - 5) := 
  sorry

end two_point_form_eq_l221_221108


namespace intersection_line_constant_l221_221320

-- Definitions based on conditions provided:
def circle1_eq (x y : ℝ) : Prop := (x + 6)^2 + (y - 2)^2 = 144
def circle2_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 9)^2 = 65

-- The theorem statement
theorem intersection_line_constant (c : ℝ) : 
  (∃ x y : ℝ, circle1_eq x y ∧ circle2_eq x y ∧ x + y = c) ↔ c = 6 :=
by
  sorry

end intersection_line_constant_l221_221320


namespace bus_carrying_capacity_l221_221151

variables (C : ℝ)

theorem bus_carrying_capacity (h1 : ∀ x : ℝ, x = (3 / 5) * C) 
                              (h2 : ∀ y : ℝ, y = 50 - 18)
                              (h3 : ∀ z : ℝ, x + y = C) : C = 80 :=
by
  sorry

end bus_carrying_capacity_l221_221151


namespace minimum_value_of_nS_n_l221_221358

noncomputable def a₁ (d : ℝ) : ℝ := -9/2 * d

noncomputable def S (n : ℕ) (d : ℝ) : ℝ :=
  n / 2 * (2 * a₁ d + (n - 1) * d)

theorem minimum_value_of_nS_n :
  S 10 (2/3) = 0 → S 15 (2/3) = 25 → ∃ (n : ℕ), (n * S n (2/3)) = -48 :=
by 
  intros h10 h15
  sorry

end minimum_value_of_nS_n_l221_221358


namespace total_koalas_l221_221638

namespace KangarooKoalaProof

variables {P Q R S T U V p q r s t u v : ℕ}
variables (h₁ : P = q + r + s + t + u + v)
variables (h₂ : Q = p + r + s + t + u + v)
variables (h₃ : R = p + q + s + t + u + v)
variables (h₄ : S = p + q + r + t + u + v)
variables (h₅ : T = p + q + r + s + u + v)
variables (h₆ : U = p + q + r + s + t + v)
variables (h₇ : V = p + q + r + s + t + u)
variables (h_total : P + Q + R + S + T + U + V = 2022)

theorem total_koalas : p + q + r + s + t + u + v = 337 :=
by
  sorry

end KangarooKoalaProof

end total_koalas_l221_221638


namespace largest_int_less_100_remainder_5_l221_221540

theorem largest_int_less_100_remainder_5 (a : ℕ) (h1 : a < 100) (h2 : a % 9 = 5) :
  a = 95 :=
sorry

end largest_int_less_100_remainder_5_l221_221540


namespace ratio_expression_value_l221_221716

theorem ratio_expression_value (A B C : ℚ) (hA : A = 3 * B / 2) (hC : C = 5 * B / 2) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by sorry

end ratio_expression_value_l221_221716


namespace find_g3_l221_221277

variable (g : ℝ → ℝ)

axiom condition_g :
  ∀ x : ℝ, x ≠ 1 / 2 → g x + g ((x + 2) / (2 - 4 * x)) = 2 * x

theorem find_g3 : g 3 = 9 / 2 :=
  by
    sorry

end find_g3_l221_221277


namespace complete_the_square_l221_221013

theorem complete_the_square (y : ℤ) : y^2 + 14 * y + 60 = (y + 7)^2 + 11 :=
by
  sorry

end complete_the_square_l221_221013


namespace value_of_bill_used_to_pay_l221_221243

-- Definitions of the conditions
def num_games : ℕ := 6
def cost_per_game : ℕ := 15
def num_change_bills : ℕ := 2
def change_per_bill : ℕ := 5
def total_cost : ℕ := num_games * cost_per_game
def total_change : ℕ := num_change_bills * change_per_bill

-- Proof statement: What was the value of the bill Jed used to pay
theorem value_of_bill_used_to_pay : 
  total_value = (total_cost + total_change) :=
by
  sorry

end value_of_bill_used_to_pay_l221_221243


namespace determine_radius_l221_221825

variable (R r : ℝ)

theorem determine_radius (h1 : R = 10) (h2 : π * R^2 = 2 * (π * R^2 - π * r^2)) : r = 5 * Real.sqrt 2 :=
  sorry

end determine_radius_l221_221825


namespace area_of_each_small_concave_quadrilateral_l221_221677

noncomputable def inner_diameter : ℝ := 8
noncomputable def outer_diameter : ℝ := 10
noncomputable def total_area_covered_by_annuli : ℝ := 112.5
noncomputable def pi : ℝ := 3.14

theorem area_of_each_small_concave_quadrilateral (inner_diameter outer_diameter total_area_covered_by_annuli pi: ℝ)
    (h1 : inner_diameter = 8)
    (h2 : outer_diameter = 10)
    (h3 : total_area_covered_by_annuli = 112.5)
    (h4 : pi = 3.14) :
    (π * (outer_diameter / 2) ^ 2 - π * (inner_diameter / 2) ^ 2) * 5 - total_area_covered_by_annuli / 4 = 7.2 := 
sorry

end area_of_each_small_concave_quadrilateral_l221_221677


namespace cookies_fit_in_box_l221_221561

variable (box_capacity_pounds : ℕ)
variable (cookie_weight_ounces : ℕ)
variable (ounces_per_pound : ℕ)

theorem cookies_fit_in_box (h1 : box_capacity_pounds = 40)
                           (h2 : cookie_weight_ounces = 2)
                           (h3 : ounces_per_pound = 16) :
                           box_capacity_pounds * (ounces_per_pound / cookie_weight_ounces) = 320 := by
  sorry

end cookies_fit_in_box_l221_221561


namespace correct_option_l221_221946

theorem correct_option : ∀ (x y : ℝ), 10 * x * y - 10 * y * x = 0 :=
by 
  intros x y
  sorry

end correct_option_l221_221946


namespace fraction_of_is_l221_221122

theorem fraction_of_is (a b c d e : ℚ) (h1 : a = 2) (h2 : b = 9) (h3 : c = 3) (h4 : d = 4) (h5 : e = 8/27) :
  (a / b) = e * (c / d) := 
sorry

end fraction_of_is_l221_221122


namespace gh_of_2_l221_221884

def g (x : ℝ) : ℝ := 3 * x^2 + 2
def h (x : ℝ) : ℝ := 4 * x^3 + 1

theorem gh_of_2 :
  g (h 2) = 3269 :=
by
  sorry

end gh_of_2_l221_221884


namespace sally_jolly_money_sum_l221_221913

/-- Prove the combined amount of money of Sally and Jolly is $150 given the conditions. -/
theorem sally_jolly_money_sum (S J x : ℝ) (h1 : S - x = 80) (h2 : J + 20 = 70) (h3 : S + J = 150) : S + J = 150 :=
by
  sorry

end sally_jolly_money_sum_l221_221913


namespace original_price_of_dish_l221_221497

theorem original_price_of_dish :
  let P : ℝ := 40
  (0.9 * P + 0.15 * P) - (0.9 * P + 0.15 * 0.9 * P) = 0.60 → P = 40 := by
  intros P h
  sorry

end original_price_of_dish_l221_221497


namespace three_different_suits_probability_l221_221254

def probability_three_different_suits := (39 / 51) * (35 / 50) = 91 / 170

theorem three_different_suits_probability (deck : Finset (Fin 52)) (h : deck.card = 52) :
  probability_three_different_suits :=
sorry

end three_different_suits_probability_l221_221254


namespace find_g_at_1_l221_221153

noncomputable def g (x : ℝ) : ℝ := x^2 - 2*x + 4

theorem find_g_at_1 : 
  (∀ x : ℝ, g (2*x + 3) = x^2 - 2*x + 4) → 
  g 1 = 7 := 
by
  intro h
  -- Proof goes here
  sorry

end find_g_at_1_l221_221153


namespace intersection_points_lie_on_circle_l221_221544

variables (u x y : ℝ)

theorem intersection_points_lie_on_circle :
  (∃ u : ℝ, 3 * u - 4 * y + 2 = 0 ∧ 2 * x - 3 * u * y - 4 = 0) →
  ∃ r : ℝ, (x^2 + y^2 = r^2) :=
by 
  sorry

end intersection_points_lie_on_circle_l221_221544


namespace hyperbola_foci_coords_l221_221623

theorem hyperbola_foci_coords :
  let a := 5
  let b := 2
  let c := Real.sqrt (a^2 + b^2)
  ∀ x y : ℝ, 4 * y^2 - 25 * x^2 = 100 →
  (x = 0 ∧ (y = c ∨ y = -c)) := by
  intros a b c x y h
  have h1 : 4 * y^2 = 100 + 25 * x^2 := by linarith
  have h2 : y^2 = 25 + 25/4 * x^2 := by linarith
  have h3 : x = 0 := by sorry
  have h4 : y = c ∨ y = -c := by sorry
  exact ⟨h3, h4⟩

end hyperbola_foci_coords_l221_221623


namespace find_a5_over_T9_l221_221991

-- Define arithmetic sequences and their sums
variables {a_n : ℕ → ℚ} {b_n : ℕ → ℚ}
variables {S_n : ℕ → ℚ} {T_n : ℕ → ℚ}

-- Conditions
def arithmetic_seq_a (a_n : ℕ → ℚ) : Prop :=
  ∀ n, a_n n = a_n 1 + (n - 1) * (a_n 2 - a_n 1)

def arithmetic_seq_b (b_n : ℕ → ℚ) : Prop :=
  ∀ n, b_n n = b_n 1 + (n - 1) * (b_n 2 - b_n 1)

def sum_a (S_n : ℕ → ℚ) (a_n : ℕ → ℚ) : Prop :=
  ∀ n, S_n n = n * (a_n 1 + a_n n) / 2

def sum_b (T_n : ℕ → ℚ) (b_n : ℕ → ℚ) : Prop :=
  ∀ n, T_n n = n * (b_n 1 + b_n n) / 2

def given_condition (S_n : ℕ → ℚ) (T_n : ℕ → ℚ) : Prop :=
  ∀ n, S_n n / T_n n = (n + 3) / (2 * n - 1)

-- Goal statement
theorem find_a5_over_T9 (h_a : arithmetic_seq_a a_n) (h_b : arithmetic_seq_b b_n)
  (sum_a_S : sum_a S_n a_n) (sum_b_T : sum_b T_n b_n) (cond : given_condition S_n T_n) :
  a_n 5 / T_n 9 = 4 / 51 :=
  sorry

end find_a5_over_T9_l221_221991


namespace snow_probability_at_least_once_l221_221475

theorem snow_probability_at_least_once :
  let p := 3 / 4
  let prob_no_snow_single_day := 1 - p
  let prob_no_snow_all_days := prob_no_snow_single_day ^ 5
  let prob_snow_at_least_once := 1 - prob_no_snow_all_days
  prob_snow_at_least_once = 1023 / 1024 :=
by
  sorry

end snow_probability_at_least_once_l221_221475


namespace hot_dogs_leftover_l221_221233

theorem hot_dogs_leftover :
  36159782 % 6 = 2 :=
by
  sorry

end hot_dogs_leftover_l221_221233


namespace complement_of_M_in_U_is_1_4_l221_221252

-- Define U
def U : Set ℕ := {x | x < 5 ∧ x ≠ 0}

-- Define M
def M : Set ℕ := {x | x^2 - 5 * x + 6 = 0}

-- The complement of M in U
def complement_U_M : Set ℕ := U \ M

-- The theorem to prove
theorem complement_of_M_in_U_is_1_4 : complement_U_M = {1, 4} := 
by sorry

end complement_of_M_in_U_is_1_4_l221_221252


namespace tshirt_cost_l221_221176

theorem tshirt_cost (initial_amount sweater_cost shoes_cost amount_left spent_on_tshirt : ℕ) 
  (h_initial : initial_amount = 91) 
  (h_sweater : sweater_cost = 24) 
  (h_shoes : shoes_cost = 11) 
  (h_left : amount_left = 50)
  (h_spent : spent_on_tshirt = initial_amount - amount_left - sweater_cost - shoes_cost) :
  spent_on_tshirt = 6 :=
sorry

end tshirt_cost_l221_221176


namespace probability_not_orange_not_white_l221_221577

theorem probability_not_orange_not_white (num_orange num_black num_white : ℕ)
    (h_orange : num_orange = 8) (h_black : num_black = 7) (h_white : num_white = 6) :
    (num_black : ℚ) / (num_orange + num_black + num_white : ℚ) = 1 / 3 :=
  by
    -- Solution will be here.
    sorry

end probability_not_orange_not_white_l221_221577


namespace white_marbles_count_l221_221641

section Marbles

variable (total_marbles black_marbles red_marbles green_marbles white_marbles : Nat)

theorem white_marbles_count
  (h_total: total_marbles = 60)
  (h_black: black_marbles = 32)
  (h_red: red_marbles = 10)
  (h_green: green_marbles = 5)
  (h_color: total_marbles = black_marbles + red_marbles + green_marbles + white_marbles) : 
  white_marbles = 13 := 
by
  sorry 

end Marbles

end white_marbles_count_l221_221641


namespace vacuum_total_time_l221_221587

theorem vacuum_total_time (x : ℕ) (hx : 2 * x + 5 = 27) :
  27 + x = 38 :=
by
  sorry

end vacuum_total_time_l221_221587


namespace rectangle_area_l221_221927

theorem rectangle_area (a b k : ℕ)
  (h1 : k = 6 * (a + b) + 36)
  (h2 : k = 114)
  (h3 : a / b = 8 / 5) :
  a * b = 40 :=
by {
  sorry
}

end rectangle_area_l221_221927


namespace circles_intersect_l221_221284

theorem circles_intersect :
  ∀ (x y : ℝ),
    ((x^2 + y^2 - 2 * x + 4 * y + 1 = 0) →
    (x^2 + y^2 - 6 * x + 2 * y + 9 = 0) →
    (∃ c1 c2 r1 r2 d : ℝ,
      (x - 1)^2 + (y + 2)^2 = r1 ∧ r1 = 4 ∧
      (x - 3)^2 + (y + 1)^2 = r2 ∧ r2 = 1 ∧
      d = Real.sqrt ((3 - 1)^2 + (-1 + 2)^2) ∧
      d > abs (r1 - r2) ∧ d < (r1 + r2))) :=
sorry

end circles_intersect_l221_221284


namespace sum_of_ages_l221_221692

theorem sum_of_ages (a b c : ℕ) (h1 : a = b) (h2 : a * b * c = 72) : a + b + c = 14 :=
sorry

end sum_of_ages_l221_221692


namespace sum_of_powers_of_2_and_mersenne_primes_is_sum_of_squares_l221_221037

theorem sum_of_powers_of_2_and_mersenne_primes_is_sum_of_squares 
  (n : ℕ)
  (a b c d : ℕ) 
  (h1 : n = 2^a + 2^b) 
  (h2 : a ≠ b) 
  (h3 : n = (2^c - 1) + (2^d - 1)) 
  (h4 : c ≠ d)
  (h5 : Nat.Prime (2^c - 1)) 
  (h6 : Nat.Prime (2^d - 1)) : 
  ∃ x y : ℕ, x ≠ y ∧ n = x^2 + y^2 := 
by
  sorry

end sum_of_powers_of_2_and_mersenne_primes_is_sum_of_squares_l221_221037


namespace number_of_apples_and_erasers_l221_221082

def totalApplesAndErasers (a e : ℕ) : Prop :=
  a + e = 84

def applesPerFriend (a : ℕ) : ℕ :=
  a / 3

def erasersPerTeacher (e : ℕ) : ℕ :=
  e / 2

theorem number_of_apples_and_erasers (a e : ℕ) (h : totalApplesAndErasers a e) :
  applesPerFriend a = a / 3 ∧ erasersPerTeacher e = e / 2 :=
by
  sorry

end number_of_apples_and_erasers_l221_221082


namespace molecular_weight_of_compound_l221_221656

-- Given atomic weights in g/mol
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_O  : ℝ := 15.999
def atomic_weight_H  : ℝ := 1.008

-- Given number of atoms in the compound
def num_atoms_Ca : ℕ := 1
def num_atoms_O  : ℕ := 2
def num_atoms_H  : ℕ := 2

-- Definition of the molecular weight
def molecular_weight : ℝ :=
  (num_atoms_Ca * atomic_weight_Ca) +
  (num_atoms_O * atomic_weight_O) +
  (num_atoms_H * atomic_weight_H)

-- The theorem to prove
theorem molecular_weight_of_compound : molecular_weight = 74.094 :=
by
  sorry

end molecular_weight_of_compound_l221_221656


namespace sufficient_but_not_necessary_condition_l221_221950

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a > 4 → a^2 > 16) ∧ (∃ a, (a < -4) ∧ (a^2 > 16)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l221_221950


namespace stack_glasses_opacity_l221_221288

-- Define the main problem's parameters and conditions
def num_glass_pieces : Nat := 5
def rotations := [0, 90, 180, 270] -- Possible rotations

-- Define the main theorem to state the problem in Lean
theorem stack_glasses_opacity :
  (∃ count : Nat, count = 7200 ∧
   -- There are 5 glass pieces
   ∀ (g : Fin num_glass_pieces), 
     -- Each piece is divided into 4 triangles
     ∀ (parts : Fin 4),
     -- There exists a unique painting configuration for each piece, can one prove it is exactly 7200 ways
     True
  ) :=
  sorry

end stack_glasses_opacity_l221_221288


namespace largest_int_lt_100_div_9_rem_5_l221_221542

theorem largest_int_lt_100_div_9_rem_5 :
  ∃ a, a < 100 ∧ (a % 9 = 5) ∧ ∀ b, b < 100 ∧ (b % 9 = 5) → b ≤ 95 := by
sorry

end largest_int_lt_100_div_9_rem_5_l221_221542


namespace candy_distribution_l221_221014

theorem candy_distribution :
  (∑ r in finset.range 7 \ finset.range 2, 
     ∑ w in (finset.range (8 - r)).filter (λ w, w ≥ 2), 
       nat.choose 8 r * nat.choose (8 - r) w) = 120 :=
by
  sorry

end candy_distribution_l221_221014


namespace fraction_to_decimal_l221_221536

theorem fraction_to_decimal : (58 : ℚ) / 125 = 0.464 := by
  sorry

end fraction_to_decimal_l221_221536


namespace complex_transformation_result_l221_221298

theorem complex_transformation_result :
  let z := -1 - 2 * Complex.I 
  let rotation := (1 / 2 : ℂ) + (Complex.I * (Real.sqrt 3) / 2)
  let dilation := 2
  (z * (rotation * dilation)) = (2 * Real.sqrt 3 - 1 - (2 + Real.sqrt 3) * Complex.I) :=
by
  sorry

end complex_transformation_result_l221_221298


namespace temperature_on_friday_l221_221815

variables {M T W Th F : ℝ}

theorem temperature_on_friday
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (h3 : M = 41) :
  F = 33 :=
  sorry

end temperature_on_friday_l221_221815


namespace union_of_intervals_l221_221394

open Set

theorem union_of_intervals :
  let M := { x : ℝ | 1 < x ∧ x ≤ 3 }
  let N := { x : ℝ | 2 < x ∧ x ≤ 5 }
  M ∪ N = { x : ℝ | 1 < x ∧ x ≤ 5 } :=
by
  let M := { x : ℝ | 1 < x ∧ x ≤ 3 }
  let N := { x : ℝ | 2 < x ∧ x ≤ 5 }
  sorry

end union_of_intervals_l221_221394


namespace ratio_expression_value_l221_221727

theorem ratio_expression_value (A B C : ℚ) (h_ratio : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
by
  sorry

end ratio_expression_value_l221_221727


namespace sqrt_meaningful_iff_l221_221237

theorem sqrt_meaningful_iff (x: ℝ) : (6 - 2 * x ≥ 0) ↔ (x ≤ 3) :=
by
  sorry

end sqrt_meaningful_iff_l221_221237


namespace license_plate_increase_l221_221925

def old_license_plates : ℕ := 26 * (10^5)

def new_license_plates : ℕ := 26^2 * (10^4)

theorem license_plate_increase :
  (new_license_plates / old_license_plates : ℝ) = 2.6 := by
  sorry

end license_plate_increase_l221_221925


namespace choose_three_positive_or_two_negative_l221_221947

theorem choose_three_positive_or_two_negative (n : ℕ) (hn : n ≥ 3) (a : Fin n → ℝ) :
  ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ (0 < a i + a j + a k) ∨ ∃ (i j : Fin n), i ≠ j ∧ (a i + a j < 0) := sorry

end choose_three_positive_or_two_negative_l221_221947


namespace min_boys_needed_l221_221783

theorem min_boys_needed
  (T : ℕ) -- total apples
  (n : ℕ) -- total number of boys
  (x : ℕ) -- number of boys collecting 20 apples each
  (y : ℕ) -- number of boys collecting 20% of total apples each
  (h1 : n = x + y)
  (h2 : T = 20 * x + Nat.div (T * 20 * y) 100)
  (hx_pos : x > 0) 
  (hy_pos : y > 0) : n ≥ 2 :=
sorry

end min_boys_needed_l221_221783


namespace side_length_of_S2_is_1001_l221_221259

-- Definitions and Conditions
variables (R1 R2 : Type) (S1 S2 S3 : Type)
variables (r s : ℤ)
variables (h_total_width : 2 * r + 3 * s = 4422)
variables (h_total_height : 2 * r + s = 2420)

theorem side_length_of_S2_is_1001 (R1 R2 S1 S2 S3 : Type) (r s : ℤ)
  (h_total_width : 2 * r + 3 * s = 4422)
  (h_total_height : 2 * r + s = 2420) : s = 1001 :=
by
  sorry -- proof to be provided

end side_length_of_S2_is_1001_l221_221259


namespace fraction_equality_l221_221846

theorem fraction_equality (x y : ℚ) (hx : x = 4 / 7) (hy : y = 5 / 11) : 
  (7 * x + 11 * y) / (77 * x * y) = 9 / 20 :=
by
  -- proof can be provided here.
  sorry

end fraction_equality_l221_221846


namespace smallest_rectangle_area_contains_L_shape_l221_221657

-- Condition: Side length of each square
def side_length : ℕ := 8

-- Condition: Number of squares
def num_squares : ℕ := 6

-- The correct answer (to be proven equivalent)
def expected_area : ℕ := 768

-- The main theorem stating the expected proof problem
theorem smallest_rectangle_area_contains_L_shape 
  (side_length : ℕ) (num_squares : ℕ) (h_shape : side_length = 8 ∧ num_squares = 6) : 
  ∃area, area = expected_area :=
by
  sorry

end smallest_rectangle_area_contains_L_shape_l221_221657


namespace part_I_part_II_l221_221876

noncomputable def f (x a : ℝ) := 2 * |x - 1| - a
noncomputable def g (x m : ℝ) := - |x + m|

theorem part_I (a : ℝ) : 
  (∃! x : ℤ, x = -3 ∧ g x 3 > -1) → m = 3 := 
sorry

theorem part_II (m : ℝ) : 
  (∀ x : ℝ, f x a > g x m) → a < 4 := 
sorry

end part_I_part_II_l221_221876


namespace train_length_proof_l221_221495

-- Defining the conditions
def speed_kmph : ℕ := 72
def platform_length : ℕ := 250  -- in meters
def time_seconds : ℕ := 26

-- Conversion factor from kmph to m/s
def kmph_to_mps (v : ℕ) : ℕ := (v * 1000) / 3600

-- The main goal: the length of the train
def train_length (speed_kmph : ℕ) (platform_length : ℕ) (time_seconds : ℕ) : ℕ :=
  let speed_mps := kmph_to_mps speed_kmph
  let total_distance := speed_mps * time_seconds
  total_distance - platform_length

theorem train_length_proof : train_length speed_kmph platform_length time_seconds = 270 := 
by 
  unfold train_length kmph_to_mps
  sorry

end train_length_proof_l221_221495


namespace dream_miles_driven_l221_221533

theorem dream_miles_driven (x : ℕ) (h : 4 * x + 4 * (x + 200) = 4000) : x = 400 :=
by
  sorry

end dream_miles_driven_l221_221533


namespace fraction_to_decimal_l221_221537

theorem fraction_to_decimal : (58 : ℚ) / 125 = 0.464 := by
  sorry

end fraction_to_decimal_l221_221537


namespace digits_partition_impossible_l221_221008

theorem digits_partition_impossible : 
  ¬ ∃ (A B : Finset ℕ), 
    A.card = 4 ∧ B.card = 4 ∧ A ∪ B = {1, 2, 3, 4, 5, 7, 8, 9} ∧ A ∩ B = ∅ ∧ 
    A.sum id = B.sum id := 
by
  sorry

end digits_partition_impossible_l221_221008


namespace smallest_sum_of_xy_l221_221372

theorem smallest_sum_of_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y)
  (hcond : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 10) : x + y = 45 :=
sorry

end smallest_sum_of_xy_l221_221372


namespace log_difference_example_l221_221848

theorem log_difference_example :
  ∀ (log : ℕ → ℝ),
    log 3 * 24 - log 3 * 8 = 1 := 
by
sorry

end log_difference_example_l221_221848


namespace train_length_approx_l221_221334

noncomputable def speed_kmh_to_ms (v: ℝ) : ℝ :=
  v * (1000 / 3600)

noncomputable def length_of_train (v_kmh: ℝ) (time_s: ℝ) : ℝ :=
  (speed_kmh_to_ms v_kmh) * time_s

theorem train_length_approx (v_kmh: ℝ) (time_s: ℝ) (L: ℝ) 
  (h1: v_kmh = 58) 
  (h2: time_s = 9) 
  (h3: L = length_of_train v_kmh time_s) : 
  |L - 145| < 1 :=
  by sorry

end train_length_approx_l221_221334


namespace part1_part2_l221_221032

def first_order_ratio_increasing (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), 0 < x → x < y → (f x) / x < (f y) / y

def second_order_ratio_increasing (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), 0 < x → x < y → (f x) / x^2 < (f y) / y^2

noncomputable def f (h : ℝ) (x : ℝ) : ℝ :=
  x^3 - 2 * h * x^2 - h * x

theorem part1 (h : ℝ) (h1 : first_order_ratio_increasing (f h)) (h2 : ¬ second_order_ratio_increasing (f h)) :
  h < 0 :=
sorry

theorem part2 (f : ℝ → ℝ) (h : second_order_ratio_increasing f) (h2 : ∃ k > 0, ∀ x > 0, f x < k) :
  ∃ k, k = 0 ∧ ∀ x > 0, f x < k :=
sorry

end part1_part2_l221_221032


namespace max_S_value_l221_221419

noncomputable def max_S (A C : ℝ) [DecidableEq ℝ] : ℝ :=
  if h : 0 < A ∧ A < 2 * Real.pi / 3 ∧ A + C = 2 * Real.pi / 3 then
    (Real.sqrt 3 / 6) * Real.sin (2 * A - Real.pi / 3) + (Real.sqrt 3 / 12)
  else
    0

theorem max_S_value :
  ∃ (A C : ℝ), A + C = 2 * Real.pi / 3 ∧
    (S = (Real.sqrt 3 / 3) * Real.sin A * Real.sin C) ∧
    (max_S A C = Real.sqrt 3 / 4) := 
sorry

end max_S_value_l221_221419


namespace smallest_x_plus_y_l221_221380

theorem smallest_x_plus_y 
  (x y : ℕ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y) 
  (hxy_neq : x ≠ y)
  (h_eq : (1/x + 1/y = 1/10)) : x + y = 45 :=
sorry

end smallest_x_plus_y_l221_221380


namespace polarBearDailyFish_l221_221350

-- Define the conditions
def polarBearDailyTrout : ℝ := 0.2
def polarBearDailySalmon : ℝ := 0.4

-- Define the statement to be proven
theorem polarBearDailyFish : polarBearDailyTrout + polarBearDailySalmon = 0.6 :=
by
  sorry

end polarBearDailyFish_l221_221350


namespace find_annual_interest_rate_l221_221057

noncomputable def compound_interest (P A : ℝ) (r : ℝ) (n t : ℕ) :=
  A = P * (1 + r / n) ^ (n * t)

theorem find_annual_interest_rate
  (P A : ℝ) (t n : ℕ) (r : ℝ)
  (hP : P = 6000)
  (hA : A = 6615)
  (ht : t = 2)
  (hn : n = 1)
  (hr : compound_interest P A r n t) :
  r = 0.05 :=
sorry

end find_annual_interest_rate_l221_221057


namespace weight_of_each_bag_of_food_l221_221899

theorem weight_of_each_bag_of_food
  (horses : ℕ)
  (feedings_per_day : ℕ)
  (pounds_per_feeding : ℕ)
  (days : ℕ)
  (bags : ℕ)
  (total_food_in_pounds : ℕ)
  (h1 : horses = 25)
  (h2 : feedings_per_day = 2)
  (h3 : pounds_per_feeding = 20)
  (h4 : days = 60)
  (h5 : bags = 60)
  (h6 : total_food_in_pounds = horses * (feedings_per_day * pounds_per_feeding) * days) :
  total_food_in_pounds / bags = 1000 :=
by
  sorry

end weight_of_each_bag_of_food_l221_221899


namespace arithmetic_progression_conditions_l221_221696

theorem arithmetic_progression_conditions (a d : ℝ) :
  let x := a
  let y := a + d
  let z := a + 2 * d
  (y^2 = (x^2 * z^2)^(1/2)) ↔ (d = 0 ∨ d = a * (-2 + Real.sqrt 2) ∨ d = a * (-2 - Real.sqrt 2)) :=
by
  intros
  sorry

end arithmetic_progression_conditions_l221_221696


namespace count_paths_l221_221715

theorem count_paths (m n : ℕ) : (n + m).choose m = (n + m).choose n :=
by
  sorry

end count_paths_l221_221715


namespace symmetric_line_equation_l221_221625

theorem symmetric_line_equation : 
  ∀ (P : ℝ × ℝ) (L : ℝ × ℝ × ℝ), 
  P = (1, 1) → 
  L = (2, 3, -6) → 
  (∃ (a b c : ℝ), a * 1 + b * 1 + c = 0 → a * x + b * y + c = 0 ↔ 2 * x + 3 * y - 4 = 0) 
:= 
sorry

end symmetric_line_equation_l221_221625


namespace ratio_problem_l221_221728

theorem ratio_problem 
  (A B C : ℚ) 
  (h : A / B = 3 / 2 ∧ B / C = 2 / 5 ∧ A / C = 3 / 5) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by 
  sorry

end ratio_problem_l221_221728


namespace max_a_l221_221039

noncomputable def f (a x : ℝ) : ℝ := 2 * Real.log x - a * x^2 + 3

theorem max_a (a m n : ℝ) (h₀ : 1 ≤ m ∧ m ≤ 5)
                      (h₁ : 1 ≤ n ∧ n ≤ 5)
                      (h₂ : n - m ≥ 2)
                      (h_eq : f a m = f a n) :
  a ≤ Real.log 3 / 4 :=
sorry

end max_a_l221_221039


namespace combined_score_of_three_students_left_l221_221272

variable (T S : ℕ) (avg16 avg13 : ℝ) (N16 N13 : ℕ)

theorem combined_score_of_three_students_left (h_avg16 : avg16 = 62.5) 
  (h_avg13 : avg13 = 62.0) (h_N16 : N16 = 16) (h_N13 : N13 = 13) 
  (h_total16 : T = avg16 * N16) (h_total13 : T - S = avg13 * N13) :
  S = 194 :=
by
  sorry

end combined_score_of_three_students_left_l221_221272


namespace ratio_expression_value_l221_221719

theorem ratio_expression_value (A B C : ℚ) (hA : A = 3 * B / 2) (hC : C = 5 * B / 2) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by sorry

end ratio_expression_value_l221_221719


namespace initial_men_l221_221486

/-- Initial number of men M being catered for. 
Proof that the initial number of men M is equal to 760 given the conditions. -/
theorem initial_men (M : ℕ)
  (H1 : 22 * M = 20 * M)
  (H2 : 2 * (M + 3040) = M) : M = 760 := 
sorry

end initial_men_l221_221486


namespace initial_men_count_l221_221464

-- Definitions based on problem conditions
def initial_days : ℝ := 18
def extra_men : ℝ := 400
def final_days : ℝ := 12.86

-- Proposition to show the initial number of men based on conditions
theorem initial_men_count (M : ℝ) (h : M * initial_days = (M + extra_men) * final_days) : M = 1000 := by
  sorry

end initial_men_count_l221_221464


namespace nine_b_value_l221_221568

theorem nine_b_value (a b : ℚ) (h1 : 8 * a + 3 * b = 0) (h2 : a = b - 3) : 
  9 * b = 216 / 11 :=
by
  sorry

end nine_b_value_l221_221568


namespace initial_amount_l221_221060

theorem initial_amount (H P L : ℝ) (C : ℝ) (n : ℕ) (T M : ℝ) 
  (hH : H = 10) 
  (hP : P = 2) 
  (hC : C = 1.25) 
  (hn : n = 4) 
  (hL : L = 3) 
  (hT : T = H + P + n * C) 
  (hM : M = T + L) : 
  M = 20 := 
sorry

end initial_amount_l221_221060


namespace total_students_in_high_school_l221_221647

theorem total_students_in_high_school (sample_size first_year third_year second_year : ℕ) (total_students : ℕ) 
  (h1 : sample_size = 45) 
  (h2 : first_year = 20) 
  (h3 : third_year = 10) 
  (h4 : second_year = 300)
  (h5 : sample_size = first_year + third_year + (sample_size - first_year - third_year)) :
  total_students = 900 :=
by
  sorry

end total_students_in_high_school_l221_221647


namespace find_B_l221_221812

theorem find_B (A B C : ℝ) (h1 : A = B + C) (h2 : A + B = 1/25) (h3 : C = 1/35) : B = 1/175 :=
by
  sorry

end find_B_l221_221812


namespace sum_of_integers_is_23_l221_221481

theorem sum_of_integers_is_23
  (x y : ℕ) (x_pos : 0 < x) (y_pos : 0 < y) (h : x * y + x + y = 155) 
  (rel_prime : Nat.gcd x y = 1) (x_lt_30 : x < 30) (y_lt_30 : y < 30) :
  x + y = 23 :=
by
  sorry

end sum_of_integers_is_23_l221_221481


namespace range_of_m_l221_221550

-- Definitions of vectors a and b
def a : ℝ × ℝ := (1, 3)
def b (m : ℝ) : ℝ × ℝ := (m, 4)

-- Dot product function for two 2D vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Condition for acute angle
def is_acute (m : ℝ) : Prop := dot_product a (b m) > 0

-- Definition of the range of m
def m_range : Set ℝ := {m | m > -12 ∧ m ≠ 4/3}

-- The theorem to prove
theorem range_of_m (m : ℝ) : is_acute m → m ∈ m_range :=
by
  sorry

end range_of_m_l221_221550


namespace find_a_l221_221867

noncomputable def f (a x : ℝ) : ℝ := (a * x^2 + 1) / (x + 1)

theorem find_a (a : ℝ) (h1 : ∃ t, t = (f a 1 - 1) / (1 - 0) ∧ t = ((3 * a - 1) / 4)) : a = -1 :=
by
  -- Auxiliary steps to frame the Lean theorem precisely
  let f1 := f a 1
  have h2 : f1 = (a + 1) / 2 := sorry
  have slope_tangent : ∀ t : ℝ, t = (3 * a - 1) / 4 := sorry
  have tangent_eq : (∀ (x y : ℝ), y - f1 = ((3 * a - 1) / 4) * (x - 1)) := sorry
  have pass_point : ∀ (x y : ℝ), (x, y) = (0, 1) -> (1 : ℝ) - ((a + 1) / 2) = ((1 - 3 * a) / 4) := sorry
  exact sorry

end find_a_l221_221867


namespace equal_cost_per_copy_l221_221142

theorem equal_cost_per_copy 
    (x : ℕ) 
    (h₁ : 2000 % x = 0) 
    (h₂ : 3000 % (x + 50) = 0) 
    (h₃ : 2000 / x = 3000 / (x + 50)) :
    (2000 : ℕ) / x = (3000 : ℕ) / (x + 50) :=
by
  sorry

end equal_cost_per_copy_l221_221142


namespace average_weight_l221_221413

theorem average_weight {w : ℝ} 
  (h1 : 62 < w) 
  (h2 : w < 72) 
  (h3 : 60 < w) 
  (h4 : w < 70) 
  (h5 : w ≤ 65) : w = 63.5 :=
by
  sorry

end average_weight_l221_221413


namespace arithmetic_sqrt_of_4_l221_221093

theorem arithmetic_sqrt_of_4 : ∃ x : ℚ, x^2 = 4 ∧ x > 0 → x = 2 :=
by {
  sorry
}

end arithmetic_sqrt_of_4_l221_221093


namespace find_parabola_equation_l221_221874

noncomputable def given_conditions (A B : Point) (O : Point) (line_eq : Line) (p : ℝ) (b : ℝ) :=
  let yx := λ x => x + b in
  let y2 := λ y => y^2 - 2*p*x in
  let OA := vector O A in
  let OB := vector O B in
  (p > 0) ∧  -- condition for p
  (line_eq = yx) ∧  -- line equation
  (y2 = 2 * p) ∧  -- parabola equation
  (OA ⟂ OB) ∧  -- perpendicular condition
  (triangle_area O A B = 2 * sqrt 5) -- area condition

theorem find_parabola_equation (A B : Point) (O : Point) (line_eq : Line) (p : ℝ) (b : ℝ) :
  given_conditions A B O line_eq p b -> (p = 1) :=
by
sorry

end find_parabola_equation_l221_221874


namespace division_by_fraction_l221_221845

theorem division_by_fraction : 5 / (1 / 5) = 25 := by
  sorry

end division_by_fraction_l221_221845


namespace coin_and_die_probability_l221_221224

theorem coin_and_die_probability :
  let coin_outcomes := 2
  let die_outcomes := 8
  let total_outcomes := coin_outcomes * die_outcomes
  let successful_outcomes := 1 in
  let P := (successful_outcomes : ℚ) / total_outcomes in
  P = 1 / 16 :=
by
  sorry

end coin_and_die_probability_l221_221224


namespace arithmetic_expression_l221_221307

theorem arithmetic_expression :
  (((15 - 2) + (4 / (1 / 2)) - (6 * 8)) * (100 - 24)) / 38 = -54 := by
  sorry

end arithmetic_expression_l221_221307


namespace binary_to_decimal_l221_221343

theorem binary_to_decimal : (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 0 * 2^3 + 1 * 2^4 + 1 * 2^5) = 51 :=
by
  sorry

end binary_to_decimal_l221_221343


namespace sufficient_but_not_necessary_condition_l221_221216

variable (x : ℝ)

def p : Prop := (x - 1) / (x + 2) ≥ 0
def q : Prop := (x - 1) * (x + 2) ≥ 0

theorem sufficient_but_not_necessary_condition : (p x) → (q x) ∧ ¬((q x) → (p x)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l221_221216


namespace johnPaysPerYear_l221_221430

-- Define the conditions
def epiPenCost : ℝ := 500
def insuranceCoverage : ℝ := 0.75
def epiPenFrequencyPerYear : ℝ := 2 -- Twice a year since 12 months / 6 months per EpiPen

-- Calculate the cost after insurance
def costAfterInsurance (cost : ℝ) (coverage : ℝ) : ℝ :=
  cost * (1 - coverage)

-- Calculate the annual cost
def annualCost (freq : ℝ) (cost : ℝ) : ℝ :=
  freq * cost

-- The statement asserting the proof
theorem johnPaysPerYear (cost : ℝ) (coverage : ℝ) (freq : ℝ) : 
  epiPenCost = cost → 
  insuranceCoverage = coverage → 
  epiPenFrequencyPerYear = freq → 
  annualCost freq (costAfterInsurance cost coverage) = 250 := 
by 
  intros _ _ _ 
  sorry

end johnPaysPerYear_l221_221430


namespace max_x_possible_value_l221_221805

theorem max_x_possible_value : ∃ x : ℚ, 
  (∃ y : ℚ, y = (5 * x - 20) / (4 * x - 5) ∧ (y^2 + y = 20)) ∧
  x = 9 / 5 :=
begin
  sorry
end

end max_x_possible_value_l221_221805


namespace range_of_x_l221_221708

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^3

theorem range_of_x (x : ℝ) (h : f (x^2) < f (3*x - 2)) : 1 < x ∧ x < 2 :=
by
  sorry

end range_of_x_l221_221708


namespace jeff_cats_count_l221_221757

theorem jeff_cats_count :
  let initial_cats := 20
  let found_monday := 2 + 3
  let found_tuesday := 1 + 2
  let adopted_wednesday := 4 * 2
  let adopted_thursday := 3
  let found_friday := 3
  initial_cats + found_monday + found_tuesday - adopted_wednesday - adopted_thursday + found_friday = 20 := by
  sorry

end jeff_cats_count_l221_221757


namespace kates_discount_is_8_percent_l221_221841

-- Definitions based on the problem's conditions
def bobs_bill : ℤ := 30
def kates_bill : ℤ := 25
def total_paid : ℤ := 53
def total_without_discount : ℤ := bobs_bill + kates_bill
def discount_received : ℤ := total_without_discount - total_paid
def kates_discount_percentage : ℚ := (discount_received : ℚ) / kates_bill * 100

-- The theorem to prove
theorem kates_discount_is_8_percent : kates_discount_percentage = 8 :=
by
  sorry

end kates_discount_is_8_percent_l221_221841


namespace cookies_in_box_l221_221558

/-- Graeme is weighing cookies to see how many he can fit in his box. His box can only hold
    40 pounds of cookies. If each cookie weighs 2 ounces, how many cookies can he fit in the box? -/
theorem cookies_in_box (box_capacity_pounds : ℕ) (cookie_weight_ounces : ℕ) (pound_to_ounces : ℕ)
  (h_box_capacity : box_capacity_pounds = 40)
  (h_cookie_weight : cookie_weight_ounces = 2)
  (h_pound_to_ounces : pound_to_ounces = 16) :
  (box_capacity_pounds * pound_to_ounces) / cookie_weight_ounces = 320 := by 
  sorry

end cookies_in_box_l221_221558


namespace flamingoes_needed_l221_221603

def feathers_per_flamingo : ℕ := 20
def safe_pluck_percentage : ℚ := 0.25
def boas_needed : ℕ := 12
def feathers_per_boa : ℕ := 200
def total_feathers_needed : ℕ := boas_needed * feathers_per_boa

theorem flamingoes_needed :
  480 = total_feathers_needed / (feathers_per_flamingo * safe_pluck_percentage).toNat :=
by sorry

end flamingoes_needed_l221_221603


namespace small_circles_sixth_figure_l221_221780

-- Defining the function to calculate the number of circles in the nth figure
def small_circles (n : ℕ) : ℕ :=
  n * (n + 1) + 4

-- Statement of the theorem
theorem small_circles_sixth_figure :
  small_circles 6 = 46 :=
by sorry

end small_circles_sixth_figure_l221_221780


namespace units_digit_33_exp_l221_221690

def units_digit_of_power_cyclic (base exponent : ℕ) (cycle : List ℕ) : ℕ :=
  cycle.get! (exponent % cycle.length)

theorem units_digit_33_exp (n : ℕ) (h1 : 33 = 1 + 4 * 8) (h2 : 44 = 4 * 11) :
  units_digit_of_power_cyclic 33 (33 * 44 ^ 44) [3, 9, 7, 1] = 3 :=
by
  sorry

end units_digit_33_exp_l221_221690


namespace q_value_l221_221440

-- Define the conditions and the problem statement
theorem q_value (a b m p q : ℚ) (h1 : a * b = 3) 
  (h2 : (a + 1 / b) * (b + 1 / a) = q) : 
  q = 16 / 3 :=
by
  sorry

end q_value_l221_221440


namespace value_of_T_l221_221646

-- Define the main variables and conditions
variables {M T : ℝ}

-- State the conditions given in the problem
def condition1 (M T : ℝ) := 2 * M + T = 7000
def condition2 (M T : ℝ) := M + 2 * T = 9800

-- State the theorem to be proved
theorem value_of_T : 
  ∀ (M T : ℝ), condition1 M T ∧ condition2 M T → T = 4200 :=
by 
  -- Proof would go here; for now, we use "sorry" to skip it
  sorry

end value_of_T_l221_221646


namespace diana_age_is_8_l221_221972

noncomputable def age_of_grace_last_year : ℕ := 3
noncomputable def age_of_grace_today : ℕ := age_of_grace_last_year + 1
noncomputable def age_of_diana_today : ℕ := 2 * age_of_grace_today

theorem diana_age_is_8 : age_of_diana_today = 8 :=
by
  -- The proof would go here
  sorry

end diana_age_is_8_l221_221972


namespace susie_vacuums_each_room_in_20_minutes_l221_221916

theorem susie_vacuums_each_room_in_20_minutes
  (total_time_hours : ℕ)
  (number_of_rooms : ℕ)
  (total_time_minutes : ℕ)
  (time_per_room : ℕ)
  (h1 : total_time_hours = 2)
  (h2 : number_of_rooms = 6)
  (h3 : total_time_minutes = total_time_hours * 60)
  (h4 : time_per_room = total_time_minutes / number_of_rooms) :
  time_per_room = 20 :=
by
  sorry

end susie_vacuums_each_room_in_20_minutes_l221_221916


namespace probability_both_truth_l221_221145

noncomputable def probability_A_truth : ℝ := 0.75
noncomputable def probability_B_truth : ℝ := 0.60

theorem probability_both_truth : 
  (probability_A_truth * probability_B_truth) = 0.45 :=
by sorry

end probability_both_truth_l221_221145


namespace smallest_x_y_sum_l221_221386

theorem smallest_x_y_sum (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) (h4 : 1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 10) : x + y = 45 :=
by
  sorry

end smallest_x_y_sum_l221_221386


namespace count_males_not_in_orchestra_l221_221618

variable (females_band females_orchestra females_choir females_all
          males_band males_orchestra males_choir males_all total_students : ℕ)
variable (males_band_not_in_orchestra : ℕ)

theorem count_males_not_in_orchestra :
  females_band = 120 ∧ females_orchestra = 90 ∧ females_choir = 50 ∧ females_all = 30 ∧
  males_band = 90 ∧ males_orchestra = 120 ∧ males_choir = 40 ∧ males_all = 20 ∧
  total_students = 250 ∧ males_band_not_in_orchestra = (males_band - (males_band + males_orchestra + males_choir - males_all - total_students)) 
  → males_band_not_in_orchestra = 20 :=
by
  intros
  sorry

end count_males_not_in_orchestra_l221_221618


namespace plane_intersects_unit_cubes_l221_221829

-- Definitions:
def isLargeCube (cube : ℕ × ℕ × ℕ) : Prop := cube = (4, 4, 4)
def isUnitCube (size : ℕ) : Prop := size = 1

-- The main theorem we want to prove:
theorem plane_intersects_unit_cubes :
  ∀ (cube : ℕ × ℕ × ℕ) (plane : (ℝ × ℝ × ℝ) → ℝ),
  isLargeCube cube →
  (∀ point : ℝ × ℝ × ℝ, plane point = 0 → 
       ∃ (x y z : ℕ), x < 4 ∧ y < 4 ∧ z < 4 ∧ 
                     (x, y, z) ∈ { coords : ℕ × ℕ × ℕ | true }) →
  (∃ intersects : ℕ, intersects = 16) :=
by
  intros cube plane Hcube Hplane
  sorry

end plane_intersects_unit_cubes_l221_221829


namespace clever_question_l221_221645

-- Define the conditions as predicates
def inhabitants_truthful (city : String) : Prop := 
  city = "Mars-Polis"

def inhabitants_lying (city : String) : Prop := 
  city = "Mars-City"

def responses (question : String) (city : String) : String :=
  if question = "Are we in Mars-City?" then
    if city = "Mars-City" then "No" else "Yes"
  else if question = "Do you live here?" then
    if city = "Mars-City" then "No" else "Yes"
  else "Unknown"

-- Define the main theorem
theorem clever_question (city : String) (initial_response : String) :
  (inhabitants_truthful city ∨ inhabitants_lying city) →
  responses "Are we in Mars-City?" city = initial_response →
  responses "Do you live here?" city = "Yes" ∨ responses "Do you live here?" city = "No" :=
by
  sorry

end clever_question_l221_221645


namespace not_all_sets_of_10_segments_form_triangle_l221_221585

theorem not_all_sets_of_10_segments_form_triangle :
  ¬ ∀ (segments : Fin 10 → ℝ), ∃ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (segments a + segments b > segments c) ∧
    (segments a + segments c > segments b) ∧
    (segments b + segments c > segments a) :=
by
  sorry

end not_all_sets_of_10_segments_form_triangle_l221_221585


namespace composite_expression_l221_221359

theorem composite_expression (n : ℕ) (h : n > 1) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 3^(2*n+1) - 2^(2*n+1) - 6^n = a * b :=
sorry

end composite_expression_l221_221359


namespace number_of_balls_is_fifty_l221_221952

variable (x : ℝ)
variable (h : x - 40 = 60 - x)

theorem number_of_balls_is_fifty : x = 50 :=
by
  have : 2 * x = 100 := by
    linarith
  linarith

end number_of_balls_is_fifty_l221_221952


namespace cost_of_tax_free_item_D_l221_221422

theorem cost_of_tax_free_item_D 
  (P_A P_B P_C : ℝ)
  (H1 : 0.945 * P_A + 1.064 * P_B + 1.18 * P_C = 225)
  (H2 : 0.045 * P_A + 0.12 * P_B + 0.18 * P_C = 30) :
  250 - (0.945 * P_A + 1.064 * P_B + 1.18 * P_C) = 25 := 
by
  -- The proof steps would go here.
  sorry

end cost_of_tax_free_item_D_l221_221422


namespace max_students_per_class_l221_221110

theorem max_students_per_class
    (total_students : ℕ)
    (total_classes : ℕ)
    (bus_count : ℕ)
    (bus_seats : ℕ)
    (students_per_class : ℕ)
    (total_students = 920)
    (bus_count = 16)
    (bus_seats = 71)
    (∀ c < total_classes, students_per_class ≤ bus_seats) : 
    students_per_class ≤ 17 := 
by
    sorry

end max_students_per_class_l221_221110


namespace reciprocal_of_sum_of_fractions_l221_221793

theorem reciprocal_of_sum_of_fractions :
  (1 / (1 / 4 + 1 / 6)) = 12 / 5 :=
by
  sorry

end reciprocal_of_sum_of_fractions_l221_221793


namespace fraction_four_or_older_l221_221319

theorem fraction_four_or_older (total_students : ℕ) (under_three : ℕ) (not_between_three_and_four : ℕ)
  (h_total : total_students = 300) (h_under_three : under_three = 20) (h_not_between_three_and_four : not_between_three_and_four = 50) :
  (not_between_three_and_four - under_three) / total_students = 1 / 10 :=
by
  sorry

end fraction_four_or_older_l221_221319


namespace kit_costs_more_l221_221674

-- Defining the individual prices of the filters and the kit price
def price_filter1 := 16.45
def price_filter2 := 14.05
def price_filter3 := 19.50
def kit_price := 87.50

-- Calculating the total price of the filters if bought individually
def total_individual_price := (2 * price_filter1) + (2 * price_filter2) + price_filter3

-- Calculate the amount saved
def amount_saved := total_individual_price - kit_price

-- The theorem to show the amount saved 
theorem kit_costs_more : amount_saved = -7.00 := by
  sorry

end kit_costs_more_l221_221674


namespace smallest_m_divisible_by_15_l221_221439

noncomputable def largest_prime_with_2023_digits : ℕ := sorry

theorem smallest_m_divisible_by_15 :
  ∃ m : ℕ, m > 0 ∧ (largest_prime_with_2023_digits ^ 2 - m) % 15 = 0 ∧ m = 1 :=
  sorry

end smallest_m_divisible_by_15_l221_221439


namespace James_wait_weeks_l221_221588

def JamesExercising (daysPainSubside : ℕ) (healingMultiplier : ℕ) (delayAfterHealing : ℕ) (totalDaysUntilHeavyLift : ℕ) : ℕ :=
  let healingTime := daysPainSubside * healingMultiplier
  let startWorkingOut := healingTime + delayAfterHealing
  let waitingPeriodDays := totalDaysUntilHeavyLift - startWorkingOut
  waitingPeriodDays / 7

theorem James_wait_weeks : 
  JamesExercising 3 5 3 39 = 3 :=
by
  sorry

end James_wait_weeks_l221_221588


namespace log_base_8_of_512_l221_221974

theorem log_base_8_of_512 :
  log 8 512 = 3 :=
by
  /-
    We know that:
    - 8 = 2^3
    - 512 = 2^9

    Using the change of base formula we get:
    log_8 512 = log_2 512 / log_2 8
    
    Since log_2 512 = 9 and log_2 8 = 3:
    log_8 512 = 9 / 3 = 3
  -/
  sorry

end log_base_8_of_512_l221_221974


namespace articles_produced_l221_221230

theorem articles_produced (x y z w : ℕ) :
  (x ≠ 0) → (y ≠ 0) → (z ≠ 0) → (w ≠ 0) →
  ((x * x * x * (1 / x^2) = x) →
  y * z * w * (1 / x^2) = y * z * w / x^2) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end articles_produced_l221_221230


namespace tangent_identity_l221_221911

theorem tangent_identity :
  Real.tan (55 * Real.pi / 180) * 
  Real.tan (65 * Real.pi / 180) * 
  Real.tan (75 * Real.pi / 180) = 
  Real.tan (85 * Real.pi / 180) :=
sorry

end tangent_identity_l221_221911


namespace intersection_correct_l221_221208

def set_A : Set ℤ := {-1, 1, 2, 4}
def set_B : Set ℤ := {x | |x - 1| ≤ 1}

theorem intersection_correct :
  set_A ∩ set_B = {1, 2} :=
  sorry

end intersection_correct_l221_221208


namespace sum_of_ages_is_50_l221_221634

def youngest_child_age : ℕ := 4

def age_intervals : ℕ := 3

def ages_sum (n : ℕ) : ℕ :=
  youngest_child_age + (youngest_child_age + age_intervals) +
  (youngest_child_age + 2 * age_intervals) +
  (youngest_child_age + 3 * age_intervals) +
  (youngest_child_age + 4 * age_intervals)

theorem sum_of_ages_is_50 : ages_sum 5 = 50 :=
by
  sorry

end sum_of_ages_is_50_l221_221634


namespace reconstruct_points_l221_221607

noncomputable def symmetric (x y : ℝ) := 2 * y - x

theorem reconstruct_points (A' B' C' D' B C D : ℝ) :
  (∃ (A B C D : ℝ),
     B = (A + A') / 2 ∧  -- B is the midpoint of line segment AA'
     C = (B + B') / 2 ∧  -- C is the midpoint of line segment BB'
     D = (C + C') / 2 ∧  -- D is the midpoint of line segment CC'
     A = (D + D') / 2)   -- A is the midpoint of line segment DD'
  ↔ (∃ (A : ℝ), A = symmetric D D') → True := sorry

end reconstruct_points_l221_221607


namespace ratio_problem_l221_221730

theorem ratio_problem 
  (A B C : ℚ) 
  (h : A / B = 3 / 2 ∧ B / C = 2 / 5 ∧ A / C = 3 / 5) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by 
  sorry

end ratio_problem_l221_221730


namespace area_correct_l221_221670

noncomputable def area_bounded_curves : ℝ := sorry

theorem area_correct :
  ∃ S, S = area_bounded_curves ∧ S = 12 * pi + 16 := sorry

end area_correct_l221_221670


namespace winning_candidate_percentage_l221_221671

theorem winning_candidate_percentage (v1 v2 v3 : ℕ) (h1 : v1 = 1136) (h2 : v2 = 7636) (h3 : v3 = 11628) :
  ((v3: ℝ) / (v1 + v2 + v3)) * 100 = 57 := by
  sorry

end winning_candidate_percentage_l221_221671


namespace polynomial_remainder_l221_221543

def f (r : ℝ) : ℝ := r^15 - r + 3

theorem polynomial_remainder :
  f 2 = 32769 := by
  sorry

end polynomial_remainder_l221_221543


namespace cosine_identity_l221_221201

theorem cosine_identity
  (α : ℝ)
  (h : Real.sin (π / 6 + α) = (Real.sqrt 3) / 3) :
  Real.cos (π / 3 - α) = (Real.sqrt 3) / 2 :=
by
  sorry

end cosine_identity_l221_221201


namespace binomial_product_l221_221855

theorem binomial_product (x : ℝ) : (4 * x + 3) * (x - 6) = 4 * x ^ 2 - 21 * x - 18 := 
sorry

end binomial_product_l221_221855


namespace augmented_matrix_solution_l221_221747

theorem augmented_matrix_solution (m n : ℝ) (x y : ℝ)
  (h1 : m * x = 6) (h2 : 3 * y = n) (hx : x = -3) (hy : y = 4) :
  m + n = 10 :=
by
  sorry

end augmented_matrix_solution_l221_221747


namespace smallest_sum_l221_221375

theorem smallest_sum (x y : ℕ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x ≠ y) (eq : 1/x + 1/y = 1/10) : x + y = 45 :=
by
  sorry

end smallest_sum_l221_221375


namespace pay_per_task_l221_221799

def tasks_per_day : ℕ := 100
def days_per_week : ℕ := 6
def weekly_pay : ℕ := 720

theorem pay_per_task :
  (weekly_pay : ℚ) / (tasks_per_day * days_per_week) = 1.20 := 
sorry

end pay_per_task_l221_221799


namespace slowest_time_l221_221253

open Real

def time_lola (stories : ℕ) (run_time : ℝ) : ℝ := stories * run_time

def time_sam (stories_run stories_elevator : ℕ) (run_time elevate_time stop_time : ℝ) (wait_time : ℝ) : ℝ :=
  let run_part  := stories_run * run_time
  let wait_part := wait_time
  let elevator_part := stories_elevator * elevate_time + (stories_elevator - 1) * stop_time
  run_part + wait_part + elevator_part

def time_tara (stories : ℕ) (elevate_time stop_time : ℝ) : ℝ :=
  stories * elevate_time + (stories - 1) * stop_time

theorem slowest_time 
  (build_stories : ℕ) (lola_run_time sam_run_time elevate_time stop_time wait_time : ℝ)
  (h_build : build_stories = 50)
  (h_lola_run : lola_run_time = 12) (h_sam_run : sam_run_time = 15)
  (h_elevate : elevate_time = 10) (h_stop : stop_time = 4) (h_wait : wait_time = 20) :
  max (time_lola build_stories lola_run_time) 
    (max (time_sam 25 25 sam_run_time elevate_time stop_time wait_time) 
         (time_tara build_stories elevate_time stop_time)) = 741 := by
  sorry

end slowest_time_l221_221253


namespace summation_eq_16_implies_x_eq_3_over_4_l221_221567

theorem summation_eq_16_implies_x_eq_3_over_4 (x : ℝ) (h : ∑' n : ℕ, (n + 1) * x^n = 16) : x = 3 / 4 :=
sorry

end summation_eq_16_implies_x_eq_3_over_4_l221_221567


namespace ratio_problem_l221_221733

theorem ratio_problem 
  (A B C : ℚ) 
  (h : A / B = 3 / 2 ∧ B / C = 2 / 5 ∧ A / C = 3 / 5) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by 
  sorry

end ratio_problem_l221_221733


namespace decimal_representation_of_7_over_12_eq_0_point_5833_l221_221527

theorem decimal_representation_of_7_over_12_eq_0_point_5833 : (7 : ℝ) / 12 = 0.5833 :=
by
  sorry

end decimal_representation_of_7_over_12_eq_0_point_5833_l221_221527


namespace scientific_notation_correct_l221_221461

-- Define the given condition
def average_daily_users : ℝ := 2590000

-- The proof problem
theorem scientific_notation_correct :
  average_daily_users = 2.59 * 10^6 :=
sorry

end scientific_notation_correct_l221_221461


namespace qin_jiushao_operations_required_l221_221489

def polynomial (x : ℝ) : ℝ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 1

theorem qin_jiushao_operations_required : 
  (∃ x : ℝ, polynomial x = (((((5 * x + 4) * x + 3) * x + 2) * x + 1) * x + 1)) →
  (∃ m a : ℕ, m = 5 ∧ a = 5) := by
  sorry

end qin_jiushao_operations_required_l221_221489


namespace combination_divisible_by_30_l221_221261

theorem combination_divisible_by_30 (n : ℤ) : ∃ k : ℤ, n^5 - n = 30 * k :=
by
  sorry

end combination_divisible_by_30_l221_221261


namespace line_intersects_ellipse_max_chord_length_l221_221875

theorem line_intersects_ellipse (m : ℝ) : 
  (∃ x y : ℝ, (y = (3/2 : ℝ) * x + m) ∧ (x^2 / 4 + y^2 / 9 = 1)) ↔ 
  (-3 * Real.sqrt 2 ≤ m ∧ m ≤ 3 * Real.sqrt 2) := 
by sorry

theorem max_chord_length : 
  (∃ m : ℝ, (m = 0) ∧ 
    (∀ x y x1 y1 : ℝ, (y = (3/2 : ℝ) * x + m) ∧ (x^2 / 4 + y^2 / 9 = 1) ∧ 
     (y1 = (3/2 : ℝ) * x1 + m) ∧ (x1^2 / 4 + y1^2 / 9 = 1) ∧ 
     (x ≠ x1 ∨ y ≠ y1) → 
     (Real.sqrt (13 / 9) * Real.sqrt (18 - m^2) = Real.sqrt 26))) := 
by sorry

end line_intersects_ellipse_max_chord_length_l221_221875


namespace john_spent_at_candy_store_l221_221713

noncomputable def johns_allowance : ℝ := 2.40
noncomputable def arcade_spending : ℝ := (3 / 5) * johns_allowance
noncomputable def remaining_after_arcade : ℝ := johns_allowance - arcade_spending
noncomputable def toy_store_spending : ℝ := (1 / 3) * remaining_after_arcade
noncomputable def remaining_after_toy_store : ℝ := remaining_after_arcade - toy_store_spending
noncomputable def candy_store_spending : ℝ := remaining_after_toy_store

theorem john_spent_at_candy_store : candy_store_spending = 0.64 := by sorry

end john_spent_at_candy_store_l221_221713


namespace determine_rectangle_R_area_l221_221321

def side_length_large_square (s : ℕ) : Prop :=
  s = 4

def area_rectangle_R (s : ℕ) (area_R : ℕ) : Prop :=
  s * s - (1 * 4 + 1 * 1) = area_R

theorem determine_rectangle_R_area :
  ∃ (s : ℕ) (area_R : ℕ), side_length_large_square s ∧ area_rectangle_R s area_R :=
by {
  sorry
}

end determine_rectangle_R_area_l221_221321


namespace part_I_part_II_l221_221441

-- Translate the conditions and questions to Lean definition statements.

-- First part of the problem: proving the value of a
theorem part_I (a : ℝ) (f : ℝ → ℝ) (Hf : ∀ x, f x = |a * x - 1|) 
(Hsol : ∀ x, f x ≤ 2 ↔ -6 ≤ x ∧ x ≤ 2) : a = -1 / 2 :=
sorry

-- Second part of the problem: proving the range of m
theorem part_II (m : ℝ) 
(H : ∃ x : ℝ, |4 * x + 1| - |2 * x - 3| ≤ 7 - 3 * m) : m ≤ 7 / 2 :=
sorry

end part_I_part_II_l221_221441


namespace find_third_number_l221_221833

theorem find_third_number (x : ℕ) (h : (6 + 16 + x) / 3 = 13) : x = 17 :=
by
  sorry

end find_third_number_l221_221833


namespace paint_cost_per_quart_l221_221572

theorem paint_cost_per_quart
  (total_cost : ℝ)
  (coverage_per_quart : ℝ)
  (side_length : ℝ)
  (cost_per_quart : ℝ) 
  (h1 : total_cost = 192)
  (h2 : coverage_per_quart = 10)
  (h3 : side_length = 10) 
  (h4 : cost_per_quart = total_cost / ((6 * side_length ^ 2) / coverage_per_quart))
  : cost_per_quart = 3.20 := 
by 
  sorry

end paint_cost_per_quart_l221_221572


namespace system_of_equations_solution_l221_221264

theorem system_of_equations_solution :
  ∃ (x y : ℚ), 
    (2 * x - 3 * y = 1) ∧ 
    (5 * x + 4 * y = 6) ∧ 
    (x + 2 * y = 2) ∧
    x = 2 / 3 ∧ y = 2 / 3 :=
by {
  sorry
}

end system_of_equations_solution_l221_221264


namespace y_intercept_of_line_l221_221650

theorem y_intercept_of_line (x y : ℝ) (h : 2 * x - 3 * y = 6) : y = -2 :=
by
  sorry

end y_intercept_of_line_l221_221650


namespace least_pos_int_solution_l221_221863

theorem least_pos_int_solution (x : ℤ) : x + 4609 ≡ 2104 [ZMOD 12] → x = 3 := by
  sorry

end least_pos_int_solution_l221_221863


namespace log8_512_is_3_l221_221975

def log_base_8_of_512 : Prop :=
  ∀ (log8 : ℝ → ℝ),
    (log8 8 = 1 / 3 * log8 2) →
    (log8 512 = 9 * log8 2) →
    log8 8 = 3 → log8 512 = 3

theorem log8_512_is_3 : log_base_8_of_512 :=
by
  intros log8 H1 H2 H3
  -- here you would normally provide the detailed steps to solve this.
  -- however, we directly proclaim the result due to the proof being non-trivial.
  sorry

end log8_512_is_3_l221_221975


namespace find_prime_p_l221_221902

def f (x : ℕ) : ℕ :=
  (x^4 + 2 * x^3 + 4 * x^2 + 2 * x + 1)^5

theorem find_prime_p : ∃! p, Nat.Prime p ∧ f p = 418195493 := by
  sorry

end find_prime_p_l221_221902


namespace find_difference_l221_221100

variables (x y : ℝ)

theorem find_difference (h1 : x * (y + 2) = 100) (h2 : y * (x + 2) = 60) : x - y = 20 :=
sorry

end find_difference_l221_221100


namespace Deepak_age_l221_221309

theorem Deepak_age (A D : ℕ) (h1 : A / D = 4 / 3) (h2 : A + 6 = 26) : D = 15 :=
by
  sorry

end Deepak_age_l221_221309


namespace ratio_expression_value_l221_221724

theorem ratio_expression_value (A B C : ℚ) (h_ratio : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
by
  sorry

end ratio_expression_value_l221_221724


namespace geom_seq_sum_seven_terms_l221_221864

-- Defining the conditions
def a0 : ℚ := 1 / 3
def r : ℚ := 1 / 3
def n : ℕ := 7

-- Definition for the sum of the first n terms in a geometric series
def geom_series_sum (a r : ℚ) (n : ℕ) : ℚ := a * (1 - r^n) / (1 - r)

-- Statement to prove the sum of the first seven terms equals 1093/2187
theorem geom_seq_sum_seven_terms : geom_series_sum a0 r n = 1093 / 2187 := 
by 
  sorry

end geom_seq_sum_seven_terms_l221_221864


namespace average_of_pqrs_l221_221048

theorem average_of_pqrs (p q r s : ℚ) (h : (5/4) * (p + q + r + s) = 20) : ((p + q + r + s) / 4) = 4 :=
sorry

end average_of_pqrs_l221_221048


namespace f_2007_eq_0_l221_221035

-- Define even function and odd function properties
def is_even (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Define functions f and g
variables (f g : ℝ → ℝ)

-- Assume the given conditions
axiom even_f : is_even f
axiom odd_g : is_odd g
axiom g_def : ∀ x, g x = f (x - 1)

-- Prove that f(2007) = 0
theorem f_2007_eq_0 : f 2007 = 0 :=
sorry

end f_2007_eq_0_l221_221035


namespace find_a_l221_221530

def star (a b : ℝ) : ℝ := 2 * a - b^3

theorem find_a (a : ℝ) : star a 3 = 15 → a = 21 :=
by
  intro h
  sorry

end find_a_l221_221530


namespace number_of_color_copies_l221_221029

def charge_shop_X (n : ℕ) : ℝ := 1.20 * n
def charge_shop_Y (n : ℕ) : ℝ := 1.70 * n
def difference := 20

theorem number_of_color_copies (n : ℕ) (h : charge_shop_Y n = charge_shop_X n + difference) : n = 40 :=
by {
  sorry
}

end number_of_color_copies_l221_221029


namespace minimum_value_of_fractions_l221_221557

theorem minimum_value_of_fractions (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : 1 / a + 1 / b = 1) : 
  ∃ a b, (0 < a) ∧ (0 < b) ∧ (1 / a + 1 / b = 1) ∧ (∃ t, ∀ x y, (0 < x) ∧ (0 < y) ∧ (1 / x + 1 / y = 1) -> t = (1 / (x - 1) + 4 / (y - 1))) := 
sorry

end minimum_value_of_fractions_l221_221557


namespace base_conversion_b_l221_221920

-- Define the problem in Lean
theorem base_conversion_b (b : ℕ) : 
  (b^2 + 2 * b - 16 = 0) → b = 4 := 
by
  intro h
  sorry

end base_conversion_b_l221_221920


namespace johnPaysPerYear_l221_221429

-- Define the conditions
def epiPenCost : ℝ := 500
def insuranceCoverage : ℝ := 0.75
def epiPenFrequencyPerYear : ℝ := 2 -- Twice a year since 12 months / 6 months per EpiPen

-- Calculate the cost after insurance
def costAfterInsurance (cost : ℝ) (coverage : ℝ) : ℝ :=
  cost * (1 - coverage)

-- Calculate the annual cost
def annualCost (freq : ℝ) (cost : ℝ) : ℝ :=
  freq * cost

-- The statement asserting the proof
theorem johnPaysPerYear (cost : ℝ) (coverage : ℝ) (freq : ℝ) : 
  epiPenCost = cost → 
  insuranceCoverage = coverage → 
  epiPenFrequencyPerYear = freq → 
  annualCost freq (costAfterInsurance cost coverage) = 250 := 
by 
  intros _ _ _ 
  sorry

end johnPaysPerYear_l221_221429


namespace floor_ceil_expression_l221_221858

theorem floor_ceil_expression :
  (Int.floor ∘ (λ x => x + ↑(19/5)) ∘ Int.ceil ∘ λ x => x^2) (15/8) = 7 := 
by 
  sorry

end floor_ceil_expression_l221_221858


namespace sum_of_numbers_with_lcm_and_ratio_l221_221269

theorem sum_of_numbers_with_lcm_and_ratio (a b : ℕ) (h_lcm : Nat.lcm a b = 60) (h_ratio : a = 2 * b / 3) : a + b = 50 := 
by
  sorry

end sum_of_numbers_with_lcm_and_ratio_l221_221269


namespace area_of_triangle_ABC_is_25_l221_221599

/-- Define the coordinates of points A, B, C given OA and the angle BAC.
    Calculate the area of triangle ABC -/
noncomputable def area_of_triangle_ABC : ℝ :=
  let OA := real.cbrt 50 in
  let A := (OA, 0, 0) in
  let b := 1 in
  let c := 1 in
  let B := (0, b, 0) in
  let C := (0, 0, c) in
  let angle_BAC := real.pi / 4 in
  let AB := real.sqrt ((OA)^2 + (b)^2) in
  let AC := real.sqrt ((OA)^2 + (c)^2) in
  let cos_BAC := real.cos angle_BAC in
  let sin_BAC := real.sin angle_BAC in
  0.5 * AB * AC * sin_BAC

theorem area_of_triangle_ABC_is_25 : area_of_triangle_ABC = 25 :=
by sorry

end area_of_triangle_ABC_is_25_l221_221599


namespace aaron_erasers_l221_221514

theorem aaron_erasers (initial_erasers erasers_given_to_Doris erasers_given_to_Ethan erasers_given_to_Fiona : ℕ) 
  (h1 : initial_erasers = 225) 
  (h2 : erasers_given_to_Doris = 75) 
  (h3 : erasers_given_to_Ethan = 40) 
  (h4 : erasers_given_to_Fiona = 50) : 
  initial_erasers - (erasers_given_to_Doris + erasers_given_to_Ethan + erasers_given_to_Fiona) = 60 :=
by sorry

end aaron_erasers_l221_221514


namespace part_I_distribution_and_expectation_part_II_conditional_probability_l221_221090

/-- Define the context for the problem, including players and their match winning probabilities.
    Define the score distribution for player A and its expectation. -/
open ProbabilityTheory

-- Define probabilities of A, B, C winning their respective matches.
def P_A_wins_B : ℝ := 2 / 3
def P_A_wins_C : ℝ := 2 / 3
def P_A_wins_D : ℝ := 2 / 3
def P_B_wins_C : ℝ := 3 / 5
def P_B_wins_D : ℝ := 3 / 5
def P_C_wins_D : ℝ := 1 / 2

-- Assume independence of individual match results.
axiom independence {A B : Prop} : Prob (A ∧ B) = Prob A * Prob B

-- Define the distribution table for A's score X.
def distribution_table : Π (X : ℕ), ℝ
| 0 := (1 / 3) ^ 3
| 1 := 3 * (2 / 3) * (1 / 3) ^ 2
| 2 := 3 * (2 / 3) ^ 2 * (1 / 3)
| 3 := (2 / 3) ^ 3
| _ := 0

-- Define the expectation of X for player A.
def expectation_X : ℝ := 0 * distribution_table 0 + 1 * distribution_table 1 + 2 * distribution_table 2 + 3 * distribution_table 3

-- Main theorem statement for Part (I): distribution and expectation of A's score.
theorem part_I_distribution_and_expectation :
  distribution_table 0 = 1 / 27 ∧
  distribution_table 1 = 2 / 9 ∧
  distribution_table 2 = 4 / 9 ∧
  distribution_table 3 = 8 / 27 ∧
  expectation_X = 2 := by sorry

-- Define the probability of A winning the championship.
-- Define the conditional probability that B wins given A wins.
def P_A_wins_championship : ℝ := (2 / 3) ^ 3 + (1 / 3) * (2 / 3) ^ 2 * (1 - (3 / 5) ^ 2) + 2 * (1 / 3) * (2 / 3) ^ 2 * (1 - (2 / 5) * (1 / 2))

def P_A_and_B_wins_championship : ℝ := (1 / 3) * (2 / 3) ^ 2 * (2 / 5) * (3 / 5) * 2 + 2 * (1 / 3) * (2 / 3) ^ 2 * (3 / 5) ^ 2

def P_B_given_A_wins_championship : ℝ := P_A_and_B_wins_championship / P_A_wins_championship

-- Main theorem statement for Part (Ⅱ): conditional probability B wins given A wins.
theorem part_II_conditional_probability :
  P_B_given_A_wins_championship = 15 / 53 := by sorry

end part_I_distribution_and_expectation_part_II_conditional_probability_l221_221090


namespace ratio_problem_l221_221740

theorem ratio_problem (A B C : ℚ) (h : A / B = 3 / 2) (h' : B / C = 2 / 5) : (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := 
by
  sorry

end ratio_problem_l221_221740


namespace initial_employees_l221_221107

theorem initial_employees (E : ℕ)
  (salary_per_employee : ℕ)
  (laid_off_fraction : ℚ)
  (total_paid_remaining : ℕ)
  (remaining_employees : ℕ) :
  salary_per_employee = 2000 →
  laid_off_fraction = 1 / 3 →
  total_paid_remaining = 600000 →
  remaining_employees = total_paid_remaining / salary_per_employee →
  (2 / 3 : ℚ) * E = remaining_employees →
  E = 450 := by
  sorry

end initial_employees_l221_221107


namespace shape_is_cone_l221_221986

-- Define spherical coordinates
structure SphericalCoordinates where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the positive constant c
def c : ℝ := sorry

-- Assume c is positive
axiom c_positive : c > 0

-- Define the shape equation in spherical coordinates
def shape_equation (p : SphericalCoordinates) : Prop :=
  p.ρ = c * Real.sin p.φ

-- The theorem statement
theorem shape_is_cone (p : SphericalCoordinates) : shape_equation p → 
  ∃ z : ℝ, (z = p.ρ * Real.cos p.φ) ∧ (p.ρ ^ 2 = (c * Real.sin p.φ) ^ 2 + z ^ 2) :=
sorry

end shape_is_cone_l221_221986


namespace find_gross_salary_l221_221520

open Real

noncomputable def bill_take_home_salary : ℝ := 40000
noncomputable def property_tax : ℝ := 2000
noncomputable def sales_tax : ℝ := 3000
noncomputable def income_tax_rate : ℝ := 0.10

theorem find_gross_salary (gross_salary : ℝ) :
  bill_take_home_salary = gross_salary - (income_tax_rate * gross_salary + property_tax + sales_tax) →
  gross_salary = 50000 :=
by
  sorry

end find_gross_salary_l221_221520


namespace solve_for_z_l221_221234

theorem solve_for_z :
  ∃ z : ℤ, (∀ x y : ℤ, x = 11 → y = 8 → 2 * x + 3 * z = 5 * y) → z = 6 :=
by
  sorry

end solve_for_z_l221_221234


namespace radius_of_circle_B_l221_221177

-- Definitions of circles and their properties
noncomputable def circle_tangent_externally (r1 r2 : ℝ) := ∃ d : ℝ, d = r1 + r2
noncomputable def circle_tangent_internally (r1 r2 : ℝ) := ∃ d : ℝ, d = r2 - r1

-- Problem statement in Lean 4
theorem radius_of_circle_B
  (rA rB rC rD centerA centerB centerC centerD : ℝ)
  (h_rA : rA = 2)
  (h_congruent_B_C : rB = rC)
  (h_circle_A_tangent_to_B : circle_tangent_externally rA rB)
  (h_circle_A_tangent_to_C : circle_tangent_externally rA rC)
  (h_circle_B_C_tangent_e : circle_tangent_externally rB rC)
  (h_circle_B_D_tangent_i : circle_tangent_internally rB rD)
  (h_center_A_passes_D : centerA = centerD)
  (h_rD : rD = 4) : 
  rB = 1 := sorry

end radius_of_circle_B_l221_221177


namespace f_eq_91_for_all_n_leq_100_l221_221247

noncomputable def f : ℤ → ℝ := sorry

theorem f_eq_91_for_all_n_leq_100 (n : ℤ) (h : n ≤ 100) : f n = 91 := sorry

end f_eq_91_for_all_n_leq_100_l221_221247


namespace solve_floor_equation_l221_221148

noncomputable def x_solution_set : Set ℚ := 
  {x | x = 1 ∨ ∃ k : ℕ, 16 ≤ k ∧ k ≤ 22 ∧ x = (k : ℚ)/23 }

theorem solve_floor_equation (x : ℚ) (hx : x ∈ x_solution_set) : 
  (⌊20*x + 23⌋ : ℚ) = 20 + 23*x :=
sorry

end solve_floor_equation_l221_221148


namespace sum_primes_1_to_50_mod_4_and_6_l221_221196

axiom problem_condition (p : ℕ) : 
  (prime p) ∧ p ∈ Icc 1 50 ∧ (p % 4 = 3) ∧ (p % 6 = 1)

theorem sum_primes_1_to_50_mod_4_and_6 : 
  (∑ p in (finset.filter (λ p, prime p ∧ p ∈ Icc 1 50 ∧ p % 4 = 3 ∧ p % 6 = 1) finset.range(51)), p) = 38 :=
by
  sorry

end sum_primes_1_to_50_mod_4_and_6_l221_221196


namespace polygon_diagonals_l221_221215

-- Definitions of the conditions
def sum_of_angles (n : ℕ) : ℝ := (n - 2) * 180 + 360

def num_diagonals (n : ℕ) : ℤ := n * (n - 3) / 2

-- Theorem statement
theorem polygon_diagonals (n : ℕ) (h : sum_of_angles n = 2160) : num_diagonals n = 54 :=
sorry

end polygon_diagonals_l221_221215


namespace friday_vs_tuesday_l221_221073

def tuesday_amount : ℝ := 8.5
def wednesday_amount : ℝ := 5.5 * tuesday_amount
def thursday_amount : ℝ := wednesday_amount + 0.10 * wednesday_amount
def friday_amount : ℝ := 0.75 * thursday_amount

theorem friday_vs_tuesday :
  friday_amount - tuesday_amount = 30.06875 :=
sorry

end friday_vs_tuesday_l221_221073


namespace smallest_stamps_l221_221492

theorem smallest_stamps : ∃ S, 1 < S ∧ (S % 9 = 1) ∧ (S % 10 = 1) ∧ (S % 11 = 1) ∧ S = 991 :=
by
  sorry

end smallest_stamps_l221_221492


namespace first_pipe_fills_in_10_hours_l221_221487

def pipe_equation (x : ℝ) : Prop :=
  1/x + 1/12 - 1/20 = 1/7.5

theorem first_pipe_fills_in_10_hours : pipe_equation 10 :=
by
  -- Statement of the theorem
  sorry

end first_pipe_fills_in_10_hours_l221_221487


namespace price_increase_and_decrease_l221_221681

theorem price_increase_and_decrease (P : ℝ) (x : ℝ) 
  (h1 : 0 < P) 
  (h2 : (P * (1 - (x / 100) ^ 2)) = 0.81 * P) : 
  abs (x - 44) < 1 :=
by
  sorry

end price_increase_and_decrease_l221_221681


namespace principal_is_400_l221_221944

-- Define the conditions
def rate_of_interest : ℚ := 12.5
def simple_interest : ℚ := 100
def time_in_years : ℚ := 2

-- Define the formula for principal amount based on the given conditions
def principal_amount (SI R T : ℚ) : ℚ := SI * 100 / (R * T)

-- Prove that the principal amount is 400
theorem principal_is_400 :
  principal_amount simple_interest rate_of_interest time_in_years = 400 := 
by
  simp [principal_amount, simple_interest, rate_of_interest, time_in_years]
  sorry

end principal_is_400_l221_221944


namespace conjectured_equation_l221_221988

theorem conjectured_equation (n : ℕ) (h : 0 < n) : 
  ∑ k in finset.range (2n-1), (n + k) = (2n-1)^2 := 
sorry

end conjectured_equation_l221_221988


namespace greatest_integer_l221_221132

theorem greatest_integer (y : ℤ) (h : (8 : ℚ) / 11 > y / 17) : y ≤ 12 :=
by
  have h₁ : (8 : ℚ) / 11 * 17 > y := by exact (div_mul_cancel _ (by norm_num : 17 ≠ 0))
  have h₂ : 136 / 11 > y := by rwa mul_comm _ 17 at h₁
  exact_mod_cast le_of_lt h₂

end greatest_integer_l221_221132


namespace hitting_first_third_fifth_probability_hitting_exactly_three_out_of_five_probability_l221_221157

noncomputable def probability_hitting_first_third_fifth (P : ℚ) : ℚ :=
  P * (1 - P) * P * (1 - P) * P

noncomputable def binomial_coefficient (n k : ℕ) : ℕ := 
  Nat.choose n k

noncomputable def probability_hitting_exactly_three_out_of_five (P : ℚ) : ℚ :=
  binomial_coefficient 5 3 * P^3 * (1 - P)^2

theorem hitting_first_third_fifth_probability :
  probability_hitting_first_third_fifth (3/5) = 108/3125 := by
  sorry

theorem hitting_exactly_three_out_of_five_probability :
  probability_hitting_exactly_three_out_of_five (3/5) = 216/625 := by
  sorry

end hitting_first_third_fifth_probability_hitting_exactly_three_out_of_five_probability_l221_221157


namespace max_value_l221_221312

theorem max_value (a b c : ℕ) (h1 : a = 2^35) (h2 : b = 26) (h3 : c = 1) : max a (max b c) = 2^35 :=
by
  -- This is where the proof would go
  sorry

end max_value_l221_221312


namespace exists_root_in_interval_l221_221707

theorem exists_root_in_interval
    (a b c x₁ x₂ : ℝ)
    (h₁ : a * x₁^2 + b * x₁ + c = 0)
    (h₂ : -a * x₂^2 + b * x₂ + c = 0) :
    ∃ x₃ : ℝ, (a / 2) * x₃^2 + b * x₃ + c = 0 ∧ ((x₁ ≤ x₃ ∧ x₃ ≤ x₂) ∨ (x₁ ≥ x₃ ∧ x₃ ≥ x₂)) :=
sorry

end exists_root_in_interval_l221_221707


namespace solve_for_x_l221_221776

theorem solve_for_x : ∀ x : ℝ, (x - 5) ^ 3 = (1 / 27)⁻¹ → x = 8 := by
  intro x
  intro h
  sorry

end solve_for_x_l221_221776


namespace max_students_per_class_l221_221115

theorem max_students_per_class (num_students : ℕ) (seats_per_bus : ℕ) (num_buses : ℕ) (k : ℕ) 
  (h_num_students : num_students = 920) 
  (h_seats_per_bus : seats_per_bus = 71) 
  (h_num_buses : num_buses = 16) 
  (h_class_size_bound : ∀ c, c ≤ k) : 
  k = 17 :=
sorry

end max_students_per_class_l221_221115


namespace find_three_digit_number_l221_221407

theorem find_three_digit_number :
  ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧
    (x - 6) % 7 = 0 ∧
    (x - 7) % 8 = 0 ∧
    (x - 8) % 9 = 0 ∧
    x = 503 :=
by
  sorry

end find_three_digit_number_l221_221407


namespace max_valid_subset_cardinality_l221_221711

def set_S : Finset ℕ := Finset.range 1998 \ {0}

def is_valid_subset (A : Finset ℕ) : Prop :=
  ∀ (x y : ℕ), x ≠ y → x ∈ A → y ∈ A → (x + y) % 117 ≠ 0

theorem max_valid_subset_cardinality :
  ∃ (A : Finset ℕ), is_valid_subset A ∧ 995 = A.card :=
sorry

end max_valid_subset_cardinality_l221_221711


namespace van_distance_covered_l221_221683

noncomputable def distance_covered (V : ℝ) := 
  let D := V * 6
  D

theorem van_distance_covered : ∃ (D : ℝ), ∀ (V : ℝ), 
  (D = 288) ∧ (D = distance_covered V) ∧ (D = 32 * 9) :=
by
  sorry

end van_distance_covered_l221_221683


namespace probability_a_plus_ab_plus_abc_divisible_by_3_l221_221910

theorem probability_a_plus_ab_plus_abc_divisible_by_3 :
  let S := finset.range (2013 + 1)  -- the set {1, 2, ..., 2013}
  let count_multiples_of_3 (n : ℕ) : ℕ := finset.card (finset.filter (λ x, x % 3 = 0) (finset.range (n + 1)))
  let probability_div_by_3 (n : ℕ) : ℚ := (count_multiples_of_3 n).to_rat / n.to_rat
  ∃ (P : ℚ), P = (probability_div_by_3 2013) + (2/3 * (2/9)) :=
  P = (13 / 27) :=
by
  -- Proof Steps and Calculations would go here
  sorry

end probability_a_plus_ab_plus_abc_divisible_by_3_l221_221910


namespace decimal_representation_of_7_over_12_eq_0_point_5833_l221_221526

theorem decimal_representation_of_7_over_12_eq_0_point_5833 : (7 : ℝ) / 12 = 0.5833 :=
by
  sorry

end decimal_representation_of_7_over_12_eq_0_point_5833_l221_221526


namespace zoey_finishes_on_monday_l221_221664

def total_reading_days (books : ℕ) : ℕ :=
  (books * (books + 1)) / 2 + books

def day_of_week (start_day : ℕ) (days : ℕ) : ℕ :=
  (start_day + days) % 7

theorem zoey_finishes_on_monday : 
  day_of_week 2 (total_reading_days 20) = 1 :=
by
  -- Definitions
  let books := 20
  let start_day := 2 -- Corresponding to Tuesday
  let days := total_reading_days books
  
  -- Prove day_of_week 2 (total_reading_days 20) = 1
  sorry

end zoey_finishes_on_monday_l221_221664


namespace find_triple_l221_221862
-- Import necessary libraries

-- Define the required predicates and conditions
def satisfies_conditions (x y z : ℕ) : Prop :=
  x ≤ y ∧ y ≤ z ∧ x^3 * (y^3 + z^3) = 2012 * (x * y * z + 2)

-- The main theorem statement
theorem find_triple : 
  ∀ (x y z : ℕ), satisfies_conditions x y z → (x, y, z) = (2, 251, 252) :=
by
  sorry

end find_triple_l221_221862


namespace tom_total_expenditure_l221_221644

noncomputable def tom_spent_total : ℝ :=
  let skateboard_price := 9.46
  let skateboard_discount := 0.10 * skateboard_price
  let discounted_skateboard := skateboard_price - skateboard_discount

  let marbles_price := 9.56
  let marbles_discount := 0.10 * marbles_price
  let discounted_marbles := marbles_price - marbles_discount

  let shorts_price := 14.50

  let figures_price := 12.60
  let figures_discount := 0.20 * figures_price
  let discounted_figures := figures_price - figures_discount

  let puzzle_price := 6.35
  let puzzle_discount := 0.15 * puzzle_price
  let discounted_puzzle := puzzle_price - puzzle_discount

  let game_price_eur := 20.50
  let game_discount_eur := 0.05 * game_price_eur
  let discounted_game_eur := game_price_eur - game_discount_eur
  let exchange_rate := 1.12
  let discounted_game_usd := discounted_game_eur * exchange_rate

  discounted_skateboard + discounted_marbles + shorts_price + discounted_figures + discounted_puzzle + discounted_game_usd

theorem tom_total_expenditure : abs (tom_spent_total - 68.91) < 0.01 :=
by norm_num1; sorry

end tom_total_expenditure_l221_221644


namespace find_number_l221_221667

theorem find_number (x : ℚ) (h : 0.15 * 0.30 * 0.50 * x = 108) : x = 4800 :=
by
  sorry

end find_number_l221_221667


namespace days_to_complete_work_together_l221_221313

theorem days_to_complete_work_together :
  (20 * 35) / (20 + 35) = 140 / 11 :=
by
  sorry

end days_to_complete_work_together_l221_221313


namespace jenny_change_l221_221586

-- Definitions for the conditions
def single_sided_cost_per_page : ℝ := 0.10
def double_sided_cost_per_page : ℝ := 0.17
def pages_per_essay : ℕ := 25
def single_sided_copies : ℕ := 5
def double_sided_copies : ℕ := 2
def pen_cost_before_tax : ℝ := 1.50
def number_of_pens : ℕ := 7
def sales_tax_rate : ℝ := 0.10
def payment_amount : ℝ := 2 * 20.00

-- Hypothesis for the total costs and calculations
noncomputable def total_single_sided_cost : ℝ := single_sided_copies * pages_per_essay * single_sided_cost_per_page
noncomputable def total_double_sided_cost : ℝ := double_sided_copies * pages_per_essay * double_sided_cost_per_page
noncomputable def total_pen_cost_before_tax : ℝ := number_of_pens * pen_cost_before_tax
noncomputable def total_sales_tax : ℝ := sales_tax_rate * total_pen_cost_before_tax
noncomputable def total_pen_cost : ℝ := total_pen_cost_before_tax + total_sales_tax
noncomputable def total_printing_cost : ℝ := total_single_sided_cost + total_double_sided_cost
noncomputable def total_cost : ℝ := total_printing_cost + total_pen_cost
noncomputable def change : ℝ := payment_amount - total_cost

-- The proof statement
theorem jenny_change : change = 7.45 := by
  sorry

end jenny_change_l221_221586


namespace probability_snow_at_least_once_l221_221476

noncomputable def probability_at_least_once_snow : ℚ :=
  1 - (↑((1:ℚ) / 4) ^ 5)

theorem probability_snow_at_least_once (p : ℚ) (h : p = 3 / 4) :
  probability_at_least_once_snow = 1023 / 1024 := by
  sorry

end probability_snow_at_least_once_l221_221476


namespace forgotten_angle_l221_221178

theorem forgotten_angle {n : ℕ} (h₁ : 2070 = (n - 2) * 180 - angle) : angle = 90 :=
by
  sorry

end forgotten_angle_l221_221178


namespace smallest_x_y_sum_l221_221385

theorem smallest_x_y_sum (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) (h4 : 1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 10) : x + y = 45 :=
by
  sorry

end smallest_x_y_sum_l221_221385


namespace combination_divisible_by_30_l221_221260

theorem combination_divisible_by_30 (n : ℤ) : ∃ k : ℤ, n^5 - n = 30 * k :=
by
  sorry

end combination_divisible_by_30_l221_221260


namespace burgers_ordered_l221_221959

theorem burgers_ordered (H : ℕ) (Ht : H + 2 * H = 45) : 2 * H = 30 := by
  sorry

end burgers_ordered_l221_221959


namespace price_of_large_pizza_l221_221283

variable {price_small_pizza : ℕ}
variable {total_revenue : ℕ}
variable {small_pizzas_sold : ℕ}
variable {large_pizzas_sold : ℕ}
variable {price_large_pizza : ℕ}

theorem price_of_large_pizza
  (h1 : price_small_pizza = 2)
  (h2 : total_revenue = 40)
  (h3 : small_pizzas_sold = 8)
  (h4 : large_pizzas_sold = 3) :
  price_large_pizza = 8 :=
by
  sorry

end price_of_large_pizza_l221_221283


namespace equilibrium_force_l221_221044

def f1 : ℝ × ℝ := (-2, -1)
def f2 : ℝ × ℝ := (-3, 2)
def f3 : ℝ × ℝ := (4, -3)
def expected_f4 : ℝ × ℝ := (1, 2)

theorem equilibrium_force :
  (1, 2) = -(f1 + f2 + f3) := 
by
  sorry

end equilibrium_force_l221_221044


namespace inequality_2n_squared_plus_3n_plus_1_l221_221075

theorem inequality_2n_squared_plus_3n_plus_1 (n : ℕ) (h: n > 0) : (2 * n^2 + 3 * n + 1)^n ≥ 6^n * (n! * n!) := 
by sorry

end inequality_2n_squared_plus_3n_plus_1_l221_221075


namespace cake_remaining_l221_221292

theorem cake_remaining (T J: ℝ) (h1: T = 0.60) (h2: J = 0.25) :
  (1 - ((1 - T) * J + T)) = 0.30 :=
by
  sorry

end cake_remaining_l221_221292


namespace miles_in_one_hour_eq_8_l221_221518

-- Parameters as given in the conditions
variables (x : ℕ) (h1 : ∀ t : ℕ, t >= 6 → t % 6 = 0 ∨ t % 6 < 6)
variables (miles_in_one_hour : ℕ)
-- Given condition: The car drives 88 miles in 13 hours.
variable (miles_in_13_hours : miles_in_one_hour * 11 = 88)

-- Statement to prove: The car can drive 8 miles in one hour.
theorem miles_in_one_hour_eq_8 : miles_in_one_hour = 8 :=
by {
  -- Proof goes here
  sorry
}

end miles_in_one_hour_eq_8_l221_221518


namespace ratio_expression_value_l221_221720

theorem ratio_expression_value (A B C : ℚ) (hA : A = 3 * B / 2) (hC : C = 5 * B / 2) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by sorry

end ratio_expression_value_l221_221720


namespace cyclist_go_south_speed_l221_221296

noncomputable def speed_of_cyclist_go_south (v : ℝ) : Prop :=
  let north_speed := 10 -- speed of cyclist going north in kmph
  let time := 2 -- time in hours
  let distance := 50 -- distance apart in km
  (north_speed + v) * time = distance

theorem cyclist_go_south_speed (v : ℝ) : speed_of_cyclist_go_south v → v = 15 :=
by
  intro h
  -- Proof part is skipped
  sorry

end cyclist_go_south_speed_l221_221296


namespace count_integers_l221_221437

def Q (x : ℝ) : ℝ := (x - 1) * (x - 4) * (x - 9) * (x - 16) * (x - 25) * (x - 36) * (x - 49) * (x - 64) * (x - 81)

theorem count_integers (Q_le_0 : ∀ n : ℤ, Q n ≤ 0 → ∃ k : ℕ, k = 53) : ∃ k : ℕ, k = 53 := by
  sorry

end count_integers_l221_221437


namespace max_leftover_candies_l221_221223

-- Given conditions as definitions
def pieces_of_candy := ℕ
def num_bags := 11

-- Statement of the problem
theorem max_leftover_candies (x : pieces_of_candy) (h : x % num_bags ≠ 0) :
  x % num_bags = 10 :=
sorry

end max_leftover_candies_l221_221223


namespace geometric_sequence_a6a7_l221_221753

theorem geometric_sequence_a6a7 (a : ℕ → ℝ) (q : ℝ)
  (h_geometric : ∀ n, a (n+1) = q * a n)
  (h1 : a 4 * a 5 = 1)
  (h2 : a 8 * a 9 = 16) : a 6 * a 7 = 4 :=
sorry

end geometric_sequence_a6a7_l221_221753


namespace ratio_expression_value_l221_221725

theorem ratio_expression_value (A B C : ℚ) (h_ratio : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
by
  sorry

end ratio_expression_value_l221_221725


namespace ice_cream_maker_completion_time_l221_221517

def start_time := 9
def time_to_half := 3
def end_time := start_time + 2 * time_to_half

theorem ice_cream_maker_completion_time :
  end_time = 15 :=
by
  -- Definitions: 9:00 AM -> 9, 12:00 PM -> 12, 3:00 PM -> 15
  -- Calculation: end_time = 9 + 2 * 3 = 15
  sorry

end ice_cream_maker_completion_time_l221_221517


namespace contrapositive_proposition_l221_221922

theorem contrapositive_proposition (a b : ℝ) :
  (¬ ((a - b) * (a + b) = 0) → ¬ (a - b = 0)) :=
sorry

end contrapositive_proposition_l221_221922


namespace repair_cost_l221_221611

theorem repair_cost (purchase_price transport_charges selling_price profit_percentage R : ℝ)
  (h1 : purchase_price = 10000)
  (h2 : transport_charges = 1000)
  (h3 : selling_price = 24000)
  (h4 : profit_percentage = 0.5)
  (h5 : selling_price = (1 + profit_percentage) * (purchase_price + R + transport_charges)) :
  R = 5000 :=
by
  sorry

end repair_cost_l221_221611


namespace hundredth_number_is_100_l221_221591

/-- Define the sequence of numbers said by Jo, Blair, and Parker following the conditions described. --/
def next_number (turn : ℕ) : ℕ :=
  -- Each turn increments by one number starting from 1
  turn

-- Prove that the 100th number in the sequence is 100
theorem hundredth_number_is_100 :
  next_number 100 = 100 := 
by sorry

end hundredth_number_is_100_l221_221591


namespace days_in_month_l221_221473

-- The number of days in the month
variable (D : ℕ)

-- The conditions provided in the problem
def mean_daily_profit (D : ℕ) := 350
def mean_first_fifteen_days := 225
def mean_last_fifteen_days := 475
def total_profit := mean_first_fifteen_days * 15 + mean_last_fifteen_days * 15

-- The Lean statement to prove the number of days in the month
theorem days_in_month : D = 30 :=
by
  -- mean_daily_profit(D) * D should be equal to total_profit
  have h : mean_daily_profit D * D = total_profit := sorry
  -- solve for D
  sorry

end days_in_month_l221_221473


namespace quadratic_factorization_l221_221276

theorem quadratic_factorization (a b : ℕ) (h1 : x^2 - 20 * x + 96 = (x - a) * (x - b)) (h2 : a > b) : 2 * b - a = 4 :=
sorry

end quadratic_factorization_l221_221276


namespace lowest_two_digit_number_whose_digits_product_is_12_l221_221943

def is_valid_two_digit_number (n : ℕ) : Prop :=
  10 <= n ∧ n < 100 ∧ ∃ d1 d2 : ℕ, 1 ≤ d1 ∧ d1 < 10 ∧ 1 ≤ d2 ∧ d2 < 10 ∧ n = 10 * d1 + d2 ∧ d1 * d2 = 12

theorem lowest_two_digit_number_whose_digits_product_is_12 :
  ∃ n : ℕ, is_valid_two_digit_number n ∧ ∀ m : ℕ, is_valid_two_digit_number m → n ≤ m ∧ n = 26 :=
sorry

end lowest_two_digit_number_whose_digits_product_is_12_l221_221943


namespace greatest_possible_value_of_x_l221_221804

theorem greatest_possible_value_of_x : 
  (∀ x : ℚ, ((5 * x - 20) / (4 * x - 5))^2 + ((5 * x - 20) / (4 * x - 5)) = 20) → 
  x ≤ 9/5 := sorry

end greatest_possible_value_of_x_l221_221804


namespace parametric_circle_eqn_l221_221363

variables (t x y : ℝ)

theorem parametric_circle_eqn (h1 : y = t * x) (h2 : x^2 + y^2 - 4 * y = 0) :
  x = 4 * t / (1 + t^2) ∧ y = 4 * t^2 / (1 + t^2) :=
by
  sorry

end parametric_circle_eqn_l221_221363


namespace perimeter_square_III_l221_221015

theorem perimeter_square_III (perimeter_I perimeter_II : ℕ) (hI : perimeter_I = 12) (hII : perimeter_II = 24) : 
  let side_I := perimeter_I / 4 
  let side_II := perimeter_II / 4 
  let side_III := side_I + side_II 
  4 * side_III = 36 :=
by
  sorry

end perimeter_square_III_l221_221015


namespace constant_speed_l221_221266

open Real

def total_trip_time := 50.0
def total_distance := 2790.0
def break_interval := 5.0
def break_duration := 0.5
def hotel_search_time := 0.5

theorem constant_speed :
  let number_of_breaks := total_trip_time / break_interval
  let total_break_time := number_of_breaks * break_duration
  let actual_driving_time := total_trip_time - total_break_time - hotel_search_time
  let constant_speed := total_distance / actual_driving_time
  constant_speed = 62.7 :=
by
  -- Provide proof here
  sorry

end constant_speed_l221_221266


namespace blood_flow_scientific_notation_l221_221781

theorem blood_flow_scientific_notation (blood_flow : ℝ) (h : blood_flow = 4900) : 
  4900 = 4.9 * (10 ^ 3) :=
by
  sorry

end blood_flow_scientific_notation_l221_221781


namespace problem_statement_l221_221212

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := λ x => f (x - 1)

theorem problem_statement :
  (∀ x : ℝ, f (-x) = f x) →  -- Condition: f is an even function.
  (∀ x : ℝ, g (-x) = -g x) → -- Condition: g is an odd function.
  (g 1 = 3) →                -- Condition: g passes through (1,3).
  (f 2012 + g 2013 = 6) :=   -- Statement to prove.
by
  sorry

end problem_statement_l221_221212


namespace custom_op_diff_l221_221569

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x + y

theorem custom_op_diff : custom_op 8 5 - custom_op 5 8 = -12 :=
by
  sorry

end custom_op_diff_l221_221569


namespace robin_total_spending_l221_221080

def jelly_bracelets_total_cost : ℕ :=
  let names := ["Jessica", "Tori", "Lily", "Patrice"]
  let total_letters := names.foldl (λ acc name => acc + name.length) 0
  total_letters * 2

theorem robin_total_spending : jelly_bracelets_total_cost = 44 := by
  sorry

end robin_total_spending_l221_221080


namespace automobile_travel_distance_l221_221516

theorem automobile_travel_distance 
  (a r : ℝ) 
  (travel_rate : ℝ) (h1 : travel_rate = a / 6)
  (time_in_seconds : ℝ) (h2 : time_in_seconds = 180):
  (3 * time_in_seconds * travel_rate) * (1 / r) * (1 / 3) = 10 * a / r :=
by
  sorry

end automobile_travel_distance_l221_221516


namespace first_digit_one_over_137_l221_221653

-- Define the main problem in terms of first nonzero digit.
def first_nonzero_digit_right_of_decimal (n : ℕ) : ℕ :=
  let frac := 1 / (Rat.of_int n)
  let shifted_frac := frac * 10 ^ 3
  let integer_part := shifted_frac.to_nat
  integer_part % 10

theorem first_digit_one_over_137 :
  first_nonzero_digit_right_of_decimal 137 = 7 :=
by
  sorry

end first_digit_one_over_137_l221_221653


namespace ratio_problem_l221_221227

theorem ratio_problem (a b c d : ℝ) (h1 : a / b = 5) (h2 : b / c = 1 / 2) (h3 : c / d = 6) : 
  d / a = 1 / 15 :=
by sorry

end ratio_problem_l221_221227


namespace broken_line_AEC_correct_l221_221059

noncomputable def length_of_broken_line_AEC 
  (side_length : ℝ)
  (height_of_pyramid : ℝ)
  (radius_of_equiv_circle : ℝ) 
  (length_AE : ℝ)
  (length_AEC : ℝ) : Prop :=
  side_length = 230.0 ∧
  height_of_pyramid = 146.423 ∧
  radius_of_equiv_circle = height_of_pyramid ∧
  length_AE = ((230.0 * 186.184) / 218.837) ∧
  length_AEC = 2 * length_AE ∧
  round (length_AEC * 100) = 39136

theorem broken_line_AEC_correct :
  length_of_broken_line_AEC 230 146.423 (146.423) 195.681 391.362 :=
by
  sorry

end broken_line_AEC_correct_l221_221059


namespace females_with_advanced_degrees_eq_90_l221_221750

-- define the given constants
def total_employees : ℕ := 360
def total_females : ℕ := 220
def total_males : ℕ := 140
def advanced_degrees : ℕ := 140
def college_degrees : ℕ := 160
def vocational_training : ℕ := 60
def males_with_college_only : ℕ := 55
def females_with_vocational_training : ℕ := 25

-- define the main theorem to prove the number of females with advanced degrees
theorem females_with_advanced_degrees_eq_90 :
  ∃ (females_with_advanced_degrees : ℕ), females_with_advanced_degrees = 90 :=
by
  sorry

end females_with_advanced_degrees_eq_90_l221_221750


namespace circumscribed_circles_intersect_l221_221209

noncomputable def circumcircle (a b c : Point) : Set Point := sorry

noncomputable def intersect_at_single_point (circles : List (Set Point)) : Option Point := sorry

variables {A1 A2 A3 B1 B2 B3 : Point}

theorem circumscribed_circles_intersect
  (h1 : ∃ P, ∀ circle ∈ [
    circumcircle A1 A2 B3, 
    circumcircle A1 B2 A3, 
    circumcircle B1 A2 A3
  ], P ∈ circle) :
  ∃ Q, ∀ circle ∈ [
    circumcircle B1 B2 A3, 
    circumcircle B1 A2 B3, 
    circumcircle A1 B2 B3
  ], Q ∈ circle :=
sorry

end circumscribed_circles_intersect_l221_221209


namespace min_ab_12_min_rec_expression_2_l221_221361

noncomputable def condition1 (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (1 / a + 3 / b = 1)

theorem min_ab_12 {a b : ℝ} (h : condition1 a b) : 
  a * b = 12 :=
sorry

theorem min_rec_expression_2 {a b : ℝ} (h : condition1 a b) :
  (1 / (a - 1)) + (3 / (b - 3)) = 2 :=
sorry

end min_ab_12_min_rec_expression_2_l221_221361


namespace find_m_for_asymptotes_l221_221183

-- Definition of the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  (y - 1)^2 / 16 - (x + 2)^2 / 9 = 1

-- Definition of the asymptotes form
def asymptote_form (m : ℝ) (x y : ℝ) : Prop :=
  y - 1 = m * x + 2 * m ∨ y - 1 = -m * x - 2 * m

-- The main theorem to prove
theorem find_m_for_asymptotes :
  (∀ x y : ℝ, hyperbola x y → asymptote_form (4 / 3) x y) :=
sorry

end find_m_for_asymptotes_l221_221183


namespace gwen_spending_l221_221028

theorem gwen_spending : 
    ∀ (initial_amount spent remaining : ℕ), 
    initial_amount = 7 → remaining = 5 → initial_amount - remaining = 2 :=
by
    sorry

end gwen_spending_l221_221028


namespace area_of_regular_octagon_l221_221509

/-- The perimeters of a square and a regular octagon are equal.
    The area of the square is 16.
    Prove that the area of the regular octagon is 8 + 8 * sqrt 2. -/
theorem area_of_regular_octagon (a b : ℝ) (h1 : 4 * a = 8 * b) (h2 : a^2 = 16) :
  2 * (1 + Real.sqrt 2) * b^2 = 8 + 8 * Real.sqrt 2 :=
by
  sorry

end area_of_regular_octagon_l221_221509


namespace max_students_distribution_l221_221499

theorem max_students_distribution (pens toys : ℕ) (h_pens : pens = 451) (h_toys : toys = 410) :
  Nat.gcd pens toys = 41 :=
by
  sorry

end max_students_distribution_l221_221499


namespace max_students_per_class_l221_221114

theorem max_students_per_class (num_students : ℕ) (seats_per_bus : ℕ) (num_buses : ℕ) (k : ℕ) 
  (h_num_students : num_students = 920) 
  (h_seats_per_bus : seats_per_bus = 71) 
  (h_num_buses : num_buses = 16) 
  (h_class_size_bound : ∀ c, c ≤ k) : 
  k = 17 :=
sorry

end max_students_per_class_l221_221114


namespace residual_at_sample_point_l221_221877

theorem residual_at_sample_point :
  ∀ (x y : ℝ), (8 * x - 70 = 10) → (x = 10) → (y = 13) → (13 - (8 * x - 70) = 3) :=
by
  intros x y h1 h2 h3
  sorry

end residual_at_sample_point_l221_221877


namespace large_font_pages_l221_221432

theorem large_font_pages (L S : ℕ) (h1 : L + S = 21) (h2 : 3 * L = 2 * S) : L = 8 :=
by {
  sorry -- Proof can be filled in Lean; this ensures the statement aligns with problem conditions.
}

end large_font_pages_l221_221432


namespace probability_first_card_heart_second_king_l221_221936

theorem probability_first_card_heart_second_king :
  ∀ (deck : Finset ℕ) (is_heart : ℕ → Prop) (is_king : ℕ → Prop),
  deck.card = 52 →
  (∀ card ∈ deck, is_heart card ∨ ¬ is_heart card) →
  (∀ card ∈ deck, is_king card ∨ ¬ is_king card) →
  (∃ p : ℚ, p = 1/52) :=
by
  intros deck is_heart is_king h_card h_heart h_king,
  sorry

end probability_first_card_heart_second_king_l221_221936


namespace cookies_fit_in_box_l221_221560

variable (box_capacity_pounds : ℕ)
variable (cookie_weight_ounces : ℕ)
variable (ounces_per_pound : ℕ)

theorem cookies_fit_in_box (h1 : box_capacity_pounds = 40)
                           (h2 : cookie_weight_ounces = 2)
                           (h3 : ounces_per_pound = 16) :
                           box_capacity_pounds * (ounces_per_pound / cookie_weight_ounces) = 320 := by
  sorry

end cookies_fit_in_box_l221_221560


namespace range_of_sum_of_two_l221_221388

theorem range_of_sum_of_two (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a^2 + b^2 + c^2 = 1) : 
  0 ≤ a + b ∧ a + b ≤ 4 / 3 :=
by
  -- Proof goes here.
  sorry

end range_of_sum_of_two_l221_221388


namespace avery_egg_cartons_l221_221004

theorem avery_egg_cartons 
  (num_chickens : ℕ) (eggs_per_chicken : ℕ) (carton_capacity : ℕ)
  (h1 : num_chickens = 20) (h2 : eggs_per_chicken = 6) (h3 : carton_capacity = 12) :
  (num_chickens * eggs_per_chicken) / carton_capacity = 10 :=
by sorry

end avery_egg_cartons_l221_221004


namespace smallest_x_y_sum_l221_221384

theorem smallest_x_y_sum (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) (h4 : 1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 10) : x + y = 45 :=
by
  sorry

end smallest_x_y_sum_l221_221384


namespace area_percentage_decrease_l221_221197

theorem area_percentage_decrease {a b : ℝ} 
  (h1 : 2 * b = 0.1 * 4 * a) :
  ((b^2) / (a^2) * 100 = 4) :=
by
  sorry

end area_percentage_decrease_l221_221197


namespace component_unqualified_l221_221315

/-- 
    The specified diameter range for a component is within [19.98, 20.02].
    The measured diameter of the component is 19.9.
    Prove that the component is unqualified.
-/
def is_unqualified (diameter_measured : ℝ) : Prop :=
    diameter_measured < 19.98 ∨ diameter_measured > 20.02

theorem component_unqualified : is_unqualified 19.9 :=
by
  -- Proof goes here
  sorry

end component_unqualified_l221_221315


namespace maya_additional_cars_l221_221445

theorem maya_additional_cars : 
  ∃ n : ℕ, 29 + n ≥ 35 ∧ (29 + n) % 7 = 0 ∧ n = 6 :=
by
  sorry

end maya_additional_cars_l221_221445


namespace candy_store_problem_l221_221053

variable (S : ℝ)
variable (not_caught_percentage : ℝ) (sample_percentage : ℝ)
variable (caught_percentage : ℝ := 1 - not_caught_percentage)

theorem candy_store_problem
  (h1 : not_caught_percentage = 0.15)
  (h2 : sample_percentage = 25.88235294117647) :
  caught_percentage * sample_percentage = 22 := by
  sorry

end candy_store_problem_l221_221053


namespace parallel_line_with_y_intercept_l221_221469

theorem parallel_line_with_y_intercept (x y : ℝ) (m : ℝ) : 
  ((x + y + 4 = 0) → (x + y + m = 0)) ∧ (m = 1)
 := by sorry

end parallel_line_with_y_intercept_l221_221469


namespace all_values_achievable_all_values_achievable_1_all_values_achievable_2_all_values_achievable_3_all_values_achievable_4_l221_221463

def coin_values : Set ℤ := {1, 5, 10, 25}

theorem all_values_achievable (a b c d: ℕ) (h: a + b + c + d = 6) (h_a: a * 1 + b * 5 + c * 10 + d * 25 = 30) 
  (coins: Set ℤ := coin_values) : 
  ∃ (x y z w: ℕ), x + y + z + w = 6 ∧ x * 1 + y * 5 + z * 10 + w * 25 = a * 1 + b * 5 + c * 10 + d * 25 :=
by sorry

theorem all_values_achievable_1 (a b c d: ℕ) (h: a + b + c + d = 6) (h_a: a * 1 + b * 5 + c * 10 + d * 25 = 40) 
  (coins: Set ℤ := coin_values) : 
  ∃ (x y z w: ℕ), x + y + z + w = 6 ∧ x * 1 + y * 5 + z * 10 + w * 25 = a * 1 + b * 5 + c * 10 + d * 25 :=
by sorry

theorem all_values_achievable_2 (a b c d: ℕ) (h: a + b + c + d = 6) (h_a: a * 1 + b * 5 + c * 10 + d * 25 = 50) 
  (coins: Set ℤ := coin_values) : 
  ∃ (x y z w: ℕ), x + y + z + w = 6 ∧ x * 1 + y * 5 + z * 10 + w * 25 = a * 1 + b * 5 + c * 10 + d * 25 :=
by sorry

theorem all_values_achievable_3 (a b c d: ℕ) (h: a + b + c + d = 6) (h_a: a * 1 + b * 5 + c * 10 + d * 25 = 60) 
  (coins: Set ℤ := coin_values) : 
  ∃ (x y z w: ℕ), x + y + z + w = 6 ∧ x * 1 + y * 5 + z * 10 + w * 25 = a * 1 + b * 5 + c * 10 + d * 25 :=
by sorry

theorem all_values_achievable_4 (a b c d: ℕ) (h: a + b + c + d = 6) (h_a: a * 1 + b * 5 + c * 10 + d * 25 = 70) 
  (coins: Set ℤ := coin_values) : 
  ∃ (x y z w: ℕ), x + y + z + w = 6 ∧ x * 1 + y * 5 + z * 10 + w * 25 = a * 1 + b * 5 + c * 10 + d * 25 :=
by sorry

end all_values_achievable_all_values_achievable_1_all_values_achievable_2_all_values_achievable_3_all_values_achievable_4_l221_221463


namespace num_outfits_l221_221267

def num_shirts := 6
def num_ties := 4
def num_pants := 3
def outfits : ℕ := num_shirts * num_pants * (num_ties + 1)

theorem num_outfits: outfits = 90 :=
by 
  -- sorry will be removed when proof is provided
  sorry

end num_outfits_l221_221267


namespace integral_calculation_l221_221521

noncomputable def integral_value : ℝ :=
  ∫ x in 0..1, (exp (sqrt ((1-x) / (1+x)))) / ((1+x) * sqrt(1 - x^2))

theorem integral_calculation : integral_value = real.exp 1 - 1 :=
by
  sorry

end integral_calculation_l221_221521


namespace ratio_expression_value_l221_221721

theorem ratio_expression_value (A B C : ℚ) (hA : A = 3 * B / 2) (hC : C = 5 * B / 2) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by sorry

end ratio_expression_value_l221_221721


namespace y_intercept_of_line_l221_221651

theorem y_intercept_of_line (x y : ℝ) (h : 2 * x - 3 * y = 6) : y = -2 :=
by
  sorry

end y_intercept_of_line_l221_221651


namespace pet_food_total_weight_l221_221446

theorem pet_food_total_weight:
  let cat_food_bags := 3
  let weight_per_cat_food_bag := 3 -- pounds
  let dog_food_bags := 4 
  let weight_per_dog_food_bag := 5 -- pounds
  let bird_food_bags := 5
  let weight_per_bird_food_bag := 2 -- pounds
  let total_weight_pounds := (cat_food_bags * weight_per_cat_food_bag) + (dog_food_bags * weight_per_dog_food_bag) + (bird_food_bags * weight_per_bird_food_bag)
  let total_weight_ounces := total_weight_pounds * 16
  total_weight_ounces = 624 :=
by
  let cat_food_bags := 3
  let weight_per_cat_food_bag := 3
  let dog_food_bags := 4
  let weight_per_dog_food_bag := 5
  let bird_food_bags := 5
  let weight_per_bird_food_bag := 2
  let total_weight_pounds := (cat_food_bags * weight_per_cat_food_bag) + (dog_food_bags * weight_per_dog_food_bag) + (bird_food_bags * weight_per_bird_food_bag)
  let total_weight_ounces := total_weight_pounds * 16
  show total_weight_ounces = 624
  sorry

end pet_food_total_weight_l221_221446


namespace triangle_area_is_24_l221_221124

structure Point where
  x : ℝ
  y : ℝ

def distance_x (A B : Point) : ℝ :=
  abs (B.x - A.x)

def distance_y (A C : Point) : ℝ :=
  abs (C.y - A.y)

def triangle_area (A B C : Point) : ℝ :=
  0.5 * distance_x A B * distance_y A C

noncomputable def A : Point := ⟨2, 2⟩
noncomputable def B : Point := ⟨8, 2⟩
noncomputable def C : Point := ⟨4, 10⟩

theorem triangle_area_is_24 : triangle_area A B C = 24 := 
  sorry

end triangle_area_is_24_l221_221124


namespace units_digit_product_of_four_consecutive_integers_l221_221532

theorem units_digit_product_of_four_consecutive_integers (n : ℕ) (h : n % 2 = 1) : (n * (n + 1) * (n + 2) * (n + 3)) % 10 = 0 := 
by 
  sorry

end units_digit_product_of_four_consecutive_integers_l221_221532


namespace pieces_per_package_l221_221455

-- Define Robin's packages
def numGumPackages := 28
def numCandyPackages := 14

-- Define total number of pieces
def totalPieces := 7

-- Define the total number of packages
def totalPackages := numGumPackages + numCandyPackages

-- Define the expected number of pieces per package as the theorem to prove
theorem pieces_per_package : (totalPieces / totalPackages) = 1/6 := by
  sorry

end pieces_per_package_l221_221455


namespace factorize_expression_equilateral_triangle_of_sides_two_p_eq_m_plus_n_l221_221452

-- Problem 1: Factorize x^2 - y^2 + 2x - 2y
theorem factorize_expression (x y : ℝ) : x^2 - y^2 + 2 * x - 2 * y = (x - y) * (x + y + 2) := 
by sorry

-- Problem 2: Determine the shape of a triangle given a^2 + c^2 - 2b(a - b + c) = 0
theorem equilateral_triangle_of_sides (a b c : ℝ) (h : a^2 + c^2 - 2 * b * (a - b + c) = 0) : a = b ∧ b = c :=
by sorry

-- Problem 3: Prove that 2p = m + n given (1/4)(m - n)^2 = (p - n)(m - p)
theorem two_p_eq_m_plus_n (m n p : ℝ) (h : (1/4) * (m - n)^2 = (p - n) * (m - p)) : 2 * p = m + n := 
by sorry

end factorize_expression_equilateral_triangle_of_sides_two_p_eq_m_plus_n_l221_221452


namespace sum_of_polynomials_l221_221245

open Polynomial

noncomputable def f : ℚ[X] := -4 * X^2 + 2 * X - 5
noncomputable def g : ℚ[X] := -6 * X^2 + 4 * X - 9
noncomputable def h : ℚ[X] := 6 * X^2 + 6 * X + 2

theorem sum_of_polynomials :
  f + g + h = -4 * X^2 + 12 * X - 12 :=
by sorry

end sum_of_polynomials_l221_221245


namespace simplify_and_evaluate_expression_l221_221914

theorem simplify_and_evaluate_expression (x : ℤ) (h1 : -2 < x) (h2 : x < 3) :
    (x ≠ 1) → (x ≠ -1) → (x ≠ 0) → 
    ((x / (x + 1) - (3 * x) / (x - 1)) / (x / (x^2 - 1))) = -8 :=
by 
  intro h3 h4 h5
  sorry

end simplify_and_evaluate_expression_l221_221914


namespace minimum_days_bacteria_count_exceeds_500_l221_221578

theorem minimum_days_bacteria_count_exceeds_500 :
  ∃ n : ℕ, 4 * 3^n > 500 ∧ ∀ m : ℕ, m < n → 4 * 3^m ≤ 500 :=
by
  sorry

end minimum_days_bacteria_count_exceeds_500_l221_221578


namespace problem_intersection_point_l221_221949

open Real
open EuclideanGeometry

noncomputable def proofProblem :=
  let ABC := triangle
  let A  := point
  let B  := point
  let C  := point
  
  let C1 := foot_of_perpendicular A B C
  let B1 := foot_of_perpendicular A C B
  let A0 := midpoint B C
  let A1 := foot_of_perpendicular A B C
  
  let PQ  := line_through A (parallel_line B C)
  let P   := intersection PQ C1
  let Q   := intersection PQ B1
  
  let K := intersection (line_through A0 C1) PQ
  let L := intersection (line_through A0 B1) PQ
  
  -- Circumcircles of triangles PQA1, KLA0, A1B1C1, and the circle with diameter AA1 intersect at T
  let omega1 := circumscribed_circle P Q A1
  let omega2 := circumscribed_circle K L A0
  let omega3 := circumscribed_circle A1 B1 C1
  let omega4 := circumscribed_circle_on_diameter A A1

  ∃ T : point, T ∈ omega1.circle ∧ T ∈ omega2.circle ∧ T ∈ omega3.circle ∧ T ∈ omega4.circle

theorem problem_intersection_point : ∃ (T : point), ∃ ω1 ω2 ω3 ω4 (circle T), 
    (T ∈ ω1 ∧ T ∈ ω2 ∧ T ∈ ω3 ∧ T ∈ ω4) :=
by
  sorry

end problem_intersection_point_l221_221949


namespace find_grade_2_l221_221339

-- Definitions for the problem
def grade_1 := 78
def weight_1 := 20
def weight_2 := 30
def grade_3 := 90
def weight_3 := 10
def grade_4 := 85
def weight_4 := 40
def overall_average := 83

noncomputable def calc_weighted_average (G : ℕ) : ℝ :=
  (grade_1 * weight_1 + G * weight_2 + grade_3 * weight_3 + grade_4 * weight_4) / (weight_1 + weight_2 + weight_3 + weight_4)

theorem find_grade_2 (G : ℕ) : calc_weighted_average G = overall_average → G = 81 := sorry

end find_grade_2_l221_221339


namespace unique_integer_n_l221_221880

theorem unique_integer_n (n : ℤ) (h : ⌊(n^2 : ℚ) / 5⌋ - ⌊(n / 2 : ℚ)⌋^2 = 3) : n = 5 :=
  sorry

end unique_integer_n_l221_221880


namespace quadratic_has_two_distinct_real_roots_iff_l221_221748

theorem quadratic_has_two_distinct_real_roots_iff (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 6 * x - m = 0 ∧ y^2 - 6 * y - m = 0) ↔ m > -9 :=
by 
  sorry

end quadratic_has_two_distinct_real_roots_iff_l221_221748


namespace problem1_l221_221170

theorem problem1 : (- (1 / 12) - (1 / 16) + (3 / 4) - (1 / 6)) * (-48) = -21 :=
by
  sorry

end problem1_l221_221170


namespace smallest_sum_of_xy_l221_221371

theorem smallest_sum_of_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y)
  (hcond : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 10) : x + y = 45 :=
sorry

end smallest_sum_of_xy_l221_221371


namespace calc1_calc2_calc3_l221_221965

theorem calc1 : 1 - 2 + 3 - 4 + 5 = 3 := by sorry
theorem calc2 : - (4 / 7) / (8 / 49) = - (7 / 2) := by sorry
theorem calc3 : ((1 / 2) - (3 / 5) + (2 / 3)) * (-15) = - (17 / 2) := by sorry

end calc1_calc2_calc3_l221_221965


namespace greatest_integer_l221_221134

theorem greatest_integer (y : ℤ) (h : (8 : ℚ) / 11 > y / 17) : y ≤ 12 :=
by
  have h₁ : (8 : ℚ) / 11 * 17 > y := by exact (div_mul_cancel _ (by norm_num : 17 ≠ 0))
  have h₂ : 136 / 11 > y := by rwa mul_comm _ 17 at h₁
  exact_mod_cast le_of_lt h₂

end greatest_integer_l221_221134


namespace cs_competition_hits_l221_221330

theorem cs_competition_hits :
  (∃ x y z : ℕ, 5 * x + 4 * y + 3 * z = 15 ∧ x + y + z ≥ 1)
  ∧ (∀ x y z : ℕ, 5 * x + 4 * y + 3 * z = 15 ∧ x + y + z ≥ 1 → (x = 1 ∧ y = 1 ∧ z = 2) ∨ (x = 0 ∧ y = 3 ∧ z = 1)) :=
by
  sorry

end cs_competition_hits_l221_221330


namespace sequence_either_increases_or_decreases_l221_221546

theorem sequence_either_increases_or_decreases {x : ℕ → ℝ} (x1_pos : 0 < x 1) (x1_ne_one : x 1 ≠ 1) 
    (recurrence : ∀ n : ℕ, x (n + 1) = x n * (x n ^ 2 + 3) / (3 * x n ^ 2 + 1)) :
    (∀ n : ℕ, x n < x (n + 1)) ∨ (∀ n : ℕ, x n > x (n + 1)) :=
sorry

end sequence_either_increases_or_decreases_l221_221546


namespace hexagon_largest_angle_l221_221270

variable (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ)
theorem hexagon_largest_angle (h : a₁ = 3)
                             (h₀ : a₂ = 3)
                             (h₁ : a₃ = 3)
                             (h₂ : a₄ = 4)
                             (h₃ : a₅ = 5)
                             (h₄ : a₆ = 6)
                             (sum_angles : 3*a₁ + 3*a₀ + 3*a₁ + 4*a₂ + 5*a₃ + 6*a₄ = 720) :
                             6 * 30 = 180 := by
    sorry

end hexagon_largest_angle_l221_221270


namespace total_weight_is_28_87_l221_221456

def blue_ball_weight : ℝ := 6
def brown_ball_weight : ℝ := 3.12
def green_ball_weight : ℝ := 4.25

def red_ball_weight : ℝ := 2 * green_ball_weight
def yellow_ball_weight : ℝ := red_ball_weight - 1.5

def total_weight : ℝ := blue_ball_weight + brown_ball_weight + green_ball_weight + red_ball_weight + yellow_ball_weight

theorem total_weight_is_28_87 : total_weight = 28.87 :=
by
  /- proof goes here -/
  sorry

end total_weight_is_28_87_l221_221456


namespace solve_modular_equation_l221_221614

theorem solve_modular_equation (x : ℤ) :
  (15 * x + 2) % 18 = 7 % 18 ↔ x % 6 = 1 % 6 := by
  sorry

end solve_modular_equation_l221_221614


namespace balls_in_base_l221_221070

theorem balls_in_base (n k : ℕ) (h1 : 165 = (n * (n + 1) * (n + 2)) / 6) (h2 : k = n * (n + 1) / 2) : k = 45 := 
by 
  sorry

end balls_in_base_l221_221070


namespace percentage_HNO3_final_l221_221501

-- Define the initial conditions
def initial_volume_solution : ℕ := 60 -- 60 liters of solution
def initial_percentage_HNO3 : ℝ := 0.45 -- 45% HNO3
def added_pure_HNO3 : ℕ := 6 -- 6 liters of pure HNO3

-- Define the volume of HNO3 in the initial solution
def hno3_initial := initial_percentage_HNO3 * initial_volume_solution

-- Define the total volume of the final solution
def total_volume_final := initial_volume_solution + added_pure_HNO3

-- Define the total amount of HNO3 in the final solution
def total_hno3_final := hno3_initial + added_pure_HNO3

-- The main theorem: prove the final percentage is 50%
theorem percentage_HNO3_final :
  (total_hno3_final / total_volume_final) * 100 = 50 :=
by
  -- proof is omitted
  sorry

end percentage_HNO3_final_l221_221501


namespace sin_B_value_cos_A_value_l221_221410

theorem sin_B_value (A B C S : Real)
  (h1: ∃ (a b c : Real), 
    (a * c * Real.cos (π - B) = (3/2) * (1/2) * a * c * Real.sin B) ∧ 
    (S = (1/2) * a * c * Real.sin B)) : 
  Real.sin B = 4/5 :=
sorry

theorem cos_A_value (A B C : Real)
  (h1: ∃ (a b c : Real), 
    (a * c * Real.cos (π - B) = (3/2) * (1/2) * a * c * Real.sin B) ∧ 
    (S = (1/2) * a * c * Real.sin B)) 
  (h2: A - C = π/4)
  (h3: Real.sin B = 4/5) 
  (h4: Real.cos B = -3/5): 
  Real.cos A = Real.sqrt (50 + 5 * Real.sqrt 2) / 10 :=
sorry

end sin_B_value_cos_A_value_l221_221410


namespace smallest_x_plus_y_l221_221382

theorem smallest_x_plus_y 
  (x y : ℕ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y) 
  (hxy_neq : x ≠ y)
  (h_eq : (1/x + 1/y = 1/10)) : x + y = 45 :=
sorry

end smallest_x_plus_y_l221_221382


namespace Intersection_A_B_l221_221443

open Set

theorem Intersection_A_B :
  let A := {x : ℝ | 2 * x + 1 < 3}
  let B := {x : ℝ | -3 < x ∧ x < 2}
  A ∩ B = {x : ℝ | -3 < x ∧ x < 1} := by
  let A := {x : ℝ | 2 * x + 1 < 3}
  let B := {x : ℝ | -3 < x ∧ x < 2}
  show A ∩ B = {x : ℝ | -3 < x ∧ x < 1}
  sorry

end Intersection_A_B_l221_221443


namespace sum_square_geq_one_third_l221_221762

variable (a b c : ℝ)

theorem sum_square_geq_one_third (h : a + b + c = 1) : 
  a^2 + b^2 + c^2 ≥ 1 / 3 := 
sorry

end sum_square_geq_one_third_l221_221762


namespace positivity_of_xyz_l221_221819

variable {x y z : ℝ}

theorem positivity_of_xyz
  (h1 : x + y + z > 0)
  (h2 : xy + yz + zx > 0)
  (h3 : xyz > 0) :
  x > 0 ∧ y > 0 ∧ z > 0 := 
sorry

end positivity_of_xyz_l221_221819


namespace gcd_of_repeated_three_digit_numbers_l221_221185

theorem gcd_of_repeated_three_digit_numbers :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → Int.gcd 1001001 n = 1001001 :=
by
  -- proof omitted
  sorry

end gcd_of_repeated_three_digit_numbers_l221_221185


namespace area_of_shaded_region_l221_221241

theorem area_of_shaded_region 
  (r R : ℝ)
  (hR : R = 9)
  (h : 2 * r = R) :
  π * R^2 - 3 * (π * r^2) = 20.25 * π :=
by
  sorry

end area_of_shaded_region_l221_221241


namespace isosceles_triangle_sides_l221_221574

theorem isosceles_triangle_sides (P : ℝ) (a b c : ℝ) (h₀ : P = 26) (h₁ : a = 11) (h₂ : a = b ∨ a = c)
  (h₃ : a + b + c = P) : 
  (b = 11 ∧ c = 4) ∨ (b = 7.5 ∧ c = 7.5) :=
by
  sorry

end isosceles_triangle_sides_l221_221574


namespace find_a_plus_b_l221_221278

theorem find_a_plus_b (a b : ℝ) 
  (h1 : ∃ x y : ℝ, (y = a * x + 1) ∧ (x^2 + y^2 + b*x - y = 1))
  (h2 : ∀ x y : ℝ, (y = a * x + 1) ∧ (x^2 + y^2 + b*x - y = 1) → x + y = 0) : 
  a + b = 2 :=
sorry

end find_a_plus_b_l221_221278


namespace candy_profit_l221_221508

theorem candy_profit :
  let num_bars := 800
  let cost_per_4_bars := 3
  let sell_per_3_bars := 2
  let cost_price := (cost_per_4_bars / 4) * num_bars
  let sell_price := (sell_per_3_bars / 3) * num_bars
  let profit := sell_price - cost_price
  profit = -66.67 :=
by
  sorry

end candy_profit_l221_221508


namespace omega_range_monotonically_decreasing_l221_221199

-- Definition of the function f(x)
def f (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x + π / 4)

-- The theorem to be proved
theorem omega_range_monotonically_decreasing (ω : ℝ) :
  ω > 0 →
  (∀ x, π / 2 < x ∧ x < π → f ω x ≤ f ω (x + ε))) →
  1 / 2 ≤ ω ∧ ω ≤ 5 / 4 :=
sorry

end omega_range_monotonically_decreasing_l221_221199


namespace remainder_zero_l221_221438

theorem remainder_zero {n : ℕ} (h : n > 0) : 
  (2013^n - 1803^n - 1781^n + 1774^n) % 203 = 0 :=
by {
  sorry
}

end remainder_zero_l221_221438


namespace branches_on_fourth_tree_l221_221344

theorem branches_on_fourth_tree :
  ∀ (height_1 branches_1 height_2 branches_2 height_3 branches_3 height_4 avg_branches_per_foot : ℕ),
    height_1 = 50 →
    branches_1 = 200 →
    height_2 = 40 →
    branches_2 = 180 →
    height_3 = 60 →
    branches_3 = 180 →
    height_4 = 34 →
    avg_branches_per_foot = 4 →
    (height_4 * avg_branches_per_foot = 136) :=
by
  intros height_1 branches_1 height_2 branches_2 height_3 branches_3 height_4 avg_branches_per_foot
  intros h1_eq_50 b1_eq_200 h2_eq_40 b2_eq_180 h3_eq_60 b3_eq_180 h4_eq_34 avg_eq_4
  -- We assume the conditions of the problem are correct, so add them to the context
  have height1 := h1_eq_50
  have branches1 := b1_eq_200
  have height2 := h2_eq_40
  have branches2 := b2_eq_180
  have height3 := h3_eq_60
  have branches3 := b3_eq_180
  have height4 := h4_eq_34
  have avg_branches := avg_eq_4
  -- Now prove the desired result
  sorry

end branches_on_fourth_tree_l221_221344


namespace largest_n_unique_k_l221_221940

theorem largest_n_unique_k :
  ∃ (n : ℕ), (∀ (k1 k2 : ℕ), 
    (9 / 17 < n / (n + k1) → n / (n + k1) < 8 / 15 → 9 / 17 < n / (n + k2) → n / (n + k2) < 8 / 15 → k1 = k2) ∧ 
    n = 72) :=
sorry

end largest_n_unique_k_l221_221940


namespace lunch_people_count_l221_221103

theorem lunch_people_count
  (C : ℝ)   -- total lunch cost including gratuity
  (G : ℝ)   -- gratuity rate
  (P : ℝ)   -- average price per person excluding gratuity
  (n : ℕ)   -- number of people
  (h1 : C = 207.0)  -- condition: total cost with gratuity
  (h2 : G = 0.15)   -- condition: gratuity rate of 15%
  (h3 : P = 12.0)   -- condition: average price per person
  (h4 : C = (1 + G) * n * P) -- condition: total cost with gratuity is (1 + gratuity rate) * number of people * average price per person
  : n = 15 :=       -- conclusion: number of people
sorry

end lunch_people_count_l221_221103


namespace robin_total_spending_l221_221081

def jelly_bracelets_total_cost : ℕ :=
  let names := ["Jessica", "Tori", "Lily", "Patrice"]
  let total_letters := names.foldl (λ acc name => acc + name.length) 0
  total_letters * 2

theorem robin_total_spending : jelly_bracelets_total_cost = 44 := by
  sorry

end robin_total_spending_l221_221081


namespace y_intercept_of_line_l221_221649

theorem y_intercept_of_line :
  ∃ y : ℝ, (∃ x : ℝ, x = 0 ∧ 2 * x - 3 * y = 6) ∧ y = -2 :=
sorry

end y_intercept_of_line_l221_221649


namespace general_term_b_sum_inequality_l221_221067

variable (a : ℕ → ℝ) (T : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ)

-- Conditions
axiom condition1 : ∀ n : ℕ, T n = ∏ i in Finset.range n.succ, a (i + 1)
axiom condition2 : ∀ n : ℕ, 2 * a (n + 1) + T n.succ = 1
axiom condition3 : ∀ n : ℕ, b n = 1 + 1 / T n
axiom condition4 : ∀ n : ℕ, S n = ∑ i in Finset.range n.succ, a (i + 1)

-- Question 1: Find the general term formula for b_n
theorem general_term_b (n : ℕ) : b n = 2 ^ (n + 1) :=
sorry

-- Question 2: Prove the inequality for S_n
theorem sum_inequality (n : ℕ) : S n < (↑n / 2) + (0.5) * Real.log (T n + 1) - 0.25 :=
sorry

end general_term_b_sum_inequality_l221_221067


namespace inequality_solution_l221_221984

theorem inequality_solution (x : ℝ) :
  (3 / 20 + |x - 13 / 60| < 7 / 30) ↔ (2 / 15 < x ∧ x < 3 / 10) :=
sorry

end inequality_solution_l221_221984


namespace snow_at_Brecknock_l221_221337

theorem snow_at_Brecknock (hilt_snow brecknock_snow : ℕ) (h1 : hilt_snow = 29) (h2 : hilt_snow = brecknock_snow + 12) : brecknock_snow = 17 :=
by
  sorry

end snow_at_Brecknock_l221_221337


namespace truck_travel_yards_l221_221166

variables (b t : ℝ)

theorem truck_travel_yards : 
  (2 * (2 * b / 7) / (2 * t)) * 240 / 3 = (80 * b) / (7 * t) :=
by 
  sorry

end truck_travel_yards_l221_221166


namespace arithmetic_sequence_a5_value_l221_221893

variable {a_n : ℕ → ℝ}

theorem arithmetic_sequence_a5_value
  (h : a_n 2 + a_n 8 = 15 - a_n 5) :
  a_n 5 = 5 :=
sorry

end arithmetic_sequence_a5_value_l221_221893


namespace range_of_a_l221_221362

noncomputable def p (x : ℝ) : Prop := abs (3 * x - 4) > 2
noncomputable def q (x : ℝ) : Prop := 1 / (x^2 - x - 2) > 0
noncomputable def r (x a : ℝ) : Prop := (x - a) * (x - a - 1) < 0

theorem range_of_a {a : ℝ} :
  (∀ x : ℝ, ¬ r x a → ¬ p x) → (a ≥ 2 ∨ a ≤ -1/3) :=
sorry

end range_of_a_l221_221362


namespace fraction_of_300_greater_than_3_fifths_of_125_l221_221844

theorem fraction_of_300_greater_than_3_fifths_of_125 (f : ℚ)
    (h : f * 300 = 3 / 5 * 125 + 45) : 
    f = 2 / 5 :=
sorry

end fraction_of_300_greater_than_3_fifths_of_125_l221_221844


namespace y_intercept_of_line_l221_221648

theorem y_intercept_of_line :
  ∃ y : ℝ, (∃ x : ℝ, x = 0 ∧ 2 * x - 3 * y = 6) ∧ y = -2 :=
sorry

end y_intercept_of_line_l221_221648


namespace multiple_of_interest_rate_l221_221095

theorem multiple_of_interest_rate (P r m : ℝ) (h1 : P * r^2 = 40) (h2 : P * (m * r)^2 = 360) : m = 3 :=
by
  sorry

end multiple_of_interest_rate_l221_221095


namespace find_x_parallel_l221_221045

def m : ℝ × ℝ := (-2, 4)
def n (x : ℝ) : ℝ × ℝ := (x, -1)

def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ u.1 = k * v.1 ∧ u.2 = k * v.2

theorem find_x_parallel :
  parallel m (n x) → x = 1 / 2 := by 
sorry

end find_x_parallel_l221_221045


namespace not_lt_neg_version_l221_221663

theorem not_lt_neg_version (a b : ℝ) (h : a < b) : ¬ (-3 * a < -3 * b) :=
by 
  -- This is where the proof would go
  sorry

end not_lt_neg_version_l221_221663


namespace negation_equivalence_l221_221787

-- Definition of the original proposition
def proposition (x : ℝ) : Prop := x > 1 → Real.log x > 0

-- Definition of the negated proposition
def negation (x : ℝ) : Prop := ¬ (x > 1 → Real.log x > 0)

-- The mathematically equivalent proof problem as Lean statement
theorem negation_equivalence (x : ℝ) : 
  (¬ (x > 1 → Real.log x > 0)) ↔ (x ≤ 1 → Real.log x ≤ 0) := 
by 
  sorry

end negation_equivalence_l221_221787


namespace sophia_fraction_of_pie_l221_221915

theorem sophia_fraction_of_pie
  (weight_fridge : ℕ) (weight_eaten : ℕ)
  (h1 : weight_fridge = 1200)
  (h2 : weight_eaten = 240) :
  (weight_eaten : ℚ) / ((weight_fridge + weight_eaten : ℚ)) = (1 / 6) :=
by
  sorry

end sophia_fraction_of_pie_l221_221915


namespace chick_hit_count_l221_221810

theorem chick_hit_count :
  ∃ x y z : ℕ,
    9 * x + 5 * y + 2 * z = 61 ∧
    x + y + z = 10 ∧
    x ≥ 1 ∧
    y ≥ 1 ∧
    z ≥ 1 ∧
    x = 5 :=
by
  sorry

end chick_hit_count_l221_221810


namespace inequality_abcd_l221_221868

theorem inequality_abcd (a b c d : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : 0 < d) :
    (a / (b + 2*c + 3*d) + b / (c + 2*d + 3*a) + c / (d + 2*a + 3*b) + d / (a + 2*b + 3*c) >= 2 / 3) :=
by
  sorry

end inequality_abcd_l221_221868


namespace number_of_seats_in_classroom_l221_221843

theorem number_of_seats_in_classroom 
    (seats_per_row_condition : 7 + 13 = 19) 
    (rows_condition : 8 + 14 = 21) : 
    19 * 21 = 399 := 
by 
    sorry

end number_of_seats_in_classroom_l221_221843


namespace tom_splitting_slices_l221_221293

theorem tom_splitting_slices :
  ∃ S : ℕ, (∃ t, t = 3/8 * S) → 
          (∃ u, u = 1/2 * (S - t)) → 
          (∃ v, v = u + t) → 
          (v = 5) → 
          (S / 2 = 8) :=
sorry

end tom_splitting_slices_l221_221293


namespace greatest_integer_y_l221_221128

theorem greatest_integer_y (y : ℤ) : (8 : ℚ) / 11 > y / 17 ↔ y ≤ 12 := 
sorry

end greatest_integer_y_l221_221128


namespace gcd_gx_x_l221_221873

noncomputable def g (x : ℕ) := (5 * x + 3) * (11 * x + 2) * (6 * x + 7) * (3 * x + 8)

theorem gcd_gx_x {x : ℕ} (hx : 36000 ∣ x) : Nat.gcd (g x) x = 144 := by
  sorry

end gcd_gx_x_l221_221873


namespace triangle_abc_l221_221576

/-!
# Problem Statement
In triangle ABC with side lengths a, b, and c opposite to vertices A, B, and C respectively, we are given that ∠A = 2 * ∠B. We need to prove that a² = b * (b + c).
-/

variables (A B C : Type) -- Define vertices of the triangle
variables (α β γ : ℝ) -- Define angles at vertices A, B, and C respectively.

-- Define sides of the triangle
variables (a b c x y : ℝ) -- Define sides opposite to the corresponding angles

-- Main statement to prove in Lean 4
theorem triangle_abc (h1 : α = 2 * β) (h2 : a = b * (2 * β)) :
  a^2 = b * (b + c) :=
sorry

end triangle_abc_l221_221576


namespace inequality_proof_l221_221462

theorem inequality_proof (x y : ℝ) (hx : x > -1) (hy : y > -1) (hxy : x + y = 1) : 
    (x / (y + 1) + y / (x + 1) ≥ 2 / 3) := 
  sorry

end inequality_proof_l221_221462


namespace inscribed_sphere_radius_l221_221871

theorem inscribed_sphere_radius {V S1 S2 S3 S4 R : ℝ} :
  (1/3) * R * (S1 + S2 + S3 + S4) = V → 
  R = 3 * V / (S1 + S2 + S3 + S4) :=
by
  intro h
  sorry

end inscribed_sphere_radius_l221_221871


namespace nishita_common_shares_l221_221907

def annual_dividend_preferred_shares (num_preferred_shares : ℕ) (par_value : ℕ) (dividend_rate_preferred : ℕ) : ℕ :=
  (dividend_rate_preferred * par_value * num_preferred_shares) / 100

def annual_dividend_common_shares (total_dividend : ℕ) (dividend_preferred : ℕ) : ℕ :=
  total_dividend - dividend_preferred

def number_of_common_shares (annual_dividend_common : ℕ) (par_value : ℕ) (annual_rate_common : ℕ) : ℕ :=
  annual_dividend_common / ((annual_rate_common * par_value) / 100)

theorem nishita_common_shares (total_annual_dividend : ℕ) (num_preferred_shares : ℕ)
                             (par_value : ℕ) (dividend_rate_preferred : ℕ)
                             (semi_annual_rate_common : ℕ) : 
                             (number_of_common_shares (annual_dividend_common_shares total_annual_dividend 
                             (annual_dividend_preferred_shares num_preferred_shares par_value dividend_rate_preferred)) 
                             par_value (semi_annual_rate_common * 2)) = 3000 :=
by
  -- Provide values specific to the problem
  let total_annual_dividend := 16500
  let num_preferred_shares := 1200
  let par_value := 50
  let dividend_rate_preferred := 10
  let semi_annual_rate_common := 3.5
  sorry

end nishita_common_shares_l221_221907


namespace find_cost_price_per_meter_l221_221143

/-- Given that a shopkeeper sells 200 meters of cloth for Rs. 12000 at a loss of Rs. 6 per meter,
we want to find the cost price per meter of cloth. Specifically, we need to prove that the
cost price per meter is Rs. 66. -/
theorem find_cost_price_per_meter
  (total_meters : ℕ := 200)
  (selling_price : ℕ := 12000)
  (loss_per_meter : ℕ := 6) :
  (selling_price + total_meters * loss_per_meter) / total_meters = 66 :=
sorry

end find_cost_price_per_meter_l221_221143


namespace scout_troop_profit_l221_221680

-- Defining the basic conditions as Lean definitions
def num_bars : ℕ := 1500
def cost_rate : ℚ := 3 / 4 -- rate in dollars per bar
def sell_rate : ℚ := 2 / 3 -- rate in dollars per bar

-- Calculate total cost, total revenue, and profit
def total_cost : ℚ := num_bars * cost_rate
def total_revenue : ℚ := num_bars * sell_rate
def profit : ℚ := total_revenue - total_cost

-- The final theorem to be proved
theorem scout_troop_profit : profit = -125 := by
  sorry

end scout_troop_profit_l221_221680


namespace dot_product_zero_l221_221221

-- Define vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (4, 3)

-- Define the dot product operation for two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the scalar multiplication and vector subtraction for 2D vectors
def scalar_mul_vec (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (c * v.1, c * v.2)

def vec_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

-- Now we state the theorem we want to prove
theorem dot_product_zero : dot_product a (vec_sub (scalar_mul_vec 2 a) b) = 0 := 
by
  sorry

end dot_product_zero_l221_221221


namespace log_equation_l221_221981

theorem log_equation :
  (3 / (Real.log 1000^4 / Real.log 8)) + (4 / (Real.log 1000^4 / Real.log 9)) = 3 :=
by
  sorry

end log_equation_l221_221981


namespace not_possible_127_points_l221_221054

theorem not_possible_127_points (n_correct n_unanswered n_incorrect : ℕ) :
  n_correct + n_unanswered + n_incorrect = 25 →
  127 ≠ 5 * n_correct + 2 * n_unanswered - n_incorrect :=
by
  intro h_total
  sorry

end not_possible_127_points_l221_221054


namespace total_cost_l221_221836

def c_teacher : ℕ := 60
def c_student : ℕ := 40

theorem total_cost (x : ℕ) : ∃ y : ℕ, y = c_student * x + c_teacher := by
  sorry

end total_cost_l221_221836


namespace greatest_y_least_y_greatest_integer_y_l221_221137

theorem greatest_y (y : ℤ) (H : (8 : ℝ) / 11 > y / 17) : y ≤ 12 :=
sorry

theorem least_y (y : ℤ) (H : (8 : ℝ) / 11 > y / 17) : y ≥ 12 :=
sorry

theorem greatest_integer_y : ∀ (y : ℤ), ((8 : ℝ) / 11 > y / 17) → y = 12 :=
by
  intro y H
  apply le_antisymm
  apply greatest_y y H
  apply least_y y H

end greatest_y_least_y_greatest_integer_y_l221_221137


namespace sugar_mixture_problem_l221_221714

theorem sugar_mixture_problem :
  ∃ x : ℝ, (9 * x + 7 * (63 - x) = 0.9 * (9.24 * 63)) ∧ x = 41.724 :=
by
  sorry

end sugar_mixture_problem_l221_221714


namespace kim_gets_change_of_5_l221_221434

noncomputable def meal_cost : ℝ := 10
noncomputable def drink_cost : ℝ := 2.5
noncomputable def tip_rate : ℝ := 0.20
noncomputable def payment : ℝ := 20
noncomputable def total_cost_before_tip := meal_cost + drink_cost
noncomputable def tip := tip_rate * total_cost_before_tip
noncomputable def total_cost_with_tip := total_cost_before_tip + tip
noncomputable def change := payment - total_cost_with_tip

theorem kim_gets_change_of_5 : change = 5 := by
  sorry

end kim_gets_change_of_5_l221_221434


namespace propositions_correct_l221_221494

variable {R : Type} [LinearOrderedField R] {A B : Set R}

theorem propositions_correct :
  (¬ ∃ x : R, x^2 + x + 1 = 0) ∧
  (¬ (∃ x : R, x + 1 ≤ 2) → ∀ x : R, x + 1 > 2) ∧
  (∀ x : R, x ∈ A ∩ B → x ∈ A) ∧
  (∀ x : R, x > 3 → x^2 > 9 ∧ ∃ y : R, y^2 > 9 ∧ y < 3) :=
by
  sorry

end propositions_correct_l221_221494


namespace floss_per_student_l221_221188

theorem floss_per_student
  (students : ℕ)
  (yards_per_packet : ℕ)
  (floss_left_over : ℕ)
  (total_packets : ℕ)
  (total_floss : ℕ)
  (total_floss_bought : ℕ)
  (smallest_multiple_of_35 : ℕ)
  (each_student_needs : ℕ)
  (hs1 : students = 20)
  (hs2 : yards_per_packet = 35)
  (hs3 : floss_left_over = 5)
  (hs4 : total_floss = total_packets * yards_per_packet)
  (hs5 : total_floss_bought = total_floss + floss_left_over)
  (hs6 : total_floss_bought % 35 = 0)
  (hs7 : smallest_multiple_of_35 > total_packets * yards_per_packet - floss_left_over)
  (hs8 : 20 * each_student_needs + 5 = smallest_multiple_of_35)
  : each_student_needs = 5 :=
by
  sorry

end floss_per_student_l221_221188


namespace convex_2k_vertices_l221_221752

theorem convex_2k_vertices (k : ℕ) (h1 : 2 ≤ k) (h2 : k ≤ 50)
    (P : Finset (EuclideanSpace ℝ (Fin 2)))
    (hP : P.card = 100) (M : Finset (EuclideanSpace ℝ (Fin 2)))
    (hM : M.card = k) : 
  ∃ V : Finset (EuclideanSpace ℝ (Fin 2)), V.card = 2 * k ∧ ∀ m ∈ M, m ∈ convexHull ℝ V :=
by
  sorry

end convex_2k_vertices_l221_221752


namespace thomas_saves_40_per_month_l221_221929

variables (T J : ℝ) (months : ℝ := 72) 

theorem thomas_saves_40_per_month 
  (h1 : J = (3/5) * T)
  (h2 : 72 * T + 72 * J = 4608) : 
  T = 40 :=
by sorry

end thomas_saves_40_per_month_l221_221929


namespace min_sticks_cover_200cm_l221_221306

def length_covered (n6 n7 : ℕ) : ℕ :=
  6 * n6 + 7 * n7

theorem min_sticks_cover_200cm :
  ∃ (n6 n7 : ℕ), length_covered n6 n7 = 200 ∧ (∀ (m6 m7 : ℕ), (length_covered m6 m7 = 200 → m6 + m7 ≥ n6 + n7)) ∧ (n6 + n7 = 29) :=
sorry

end min_sticks_cover_200cm_l221_221306


namespace complement_union_l221_221068

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {1, 2, 4}
def N : Set ℕ := {2, 3}

theorem complement_union (U : Set ℕ) (M : Set ℕ) (N : Set ℕ) (hU : U = {0, 1, 2, 3, 4}) (hM : M = {1, 2, 4}) (hN : N = {2, 3}) :
  (U \ M) ∪ N = {0, 2, 3} :=
by
  rw [hU, hM, hN] -- Substitute U, M, N definitions
  sorry -- Proof omitted

end complement_union_l221_221068


namespace metal_bar_weight_loss_l221_221149

theorem metal_bar_weight_loss :
  ∃ T S : ℝ, 
  T + S = 50 ∧ 
  T / S = 2 / 3 ∧ 
  ((T / 10) * 1.375) + ((S / 5) * 0.375) = 5 :=
begin
  sorry
end

end metal_bar_weight_loss_l221_221149


namespace parker_total_stamps_l221_221118

-- Definitions based on conditions
def original_stamps := 430
def addie_stamps := 1890
def addie_fraction := 3 / 7
def stamps_added_by_addie := addie_fraction * addie_stamps

-- Theorem statement to prove the final number of stamps
theorem parker_total_stamps : original_stamps + stamps_added_by_addie = 1240 :=
by
  -- definitions instantiated above
  sorry  -- proof required

end parker_total_stamps_l221_221118


namespace infinitely_many_coprime_binomials_l221_221033

theorem infinitely_many_coprime_binomials (k l : ℕ) (hk : 0 < k) (hl : 0 < l) :
  ∃ᶠ n in at_top, n > k ∧ Nat.gcd (Nat.choose n k) l = 1 := by
  sorry

end infinitely_many_coprime_binomials_l221_221033


namespace cost_of_child_ticket_l221_221167

-- Define the conditions
def adult_ticket_cost : ℕ := 60
def total_people : ℕ := 280
def total_collected_dollars : ℕ := 140
def total_collected_cents : ℕ := total_collected_dollars * 100
def children_attended : ℕ := 80
def adults_attended : ℕ := total_people - children_attended
def total_collected_from_adults : ℕ := adults_attended * adult_ticket_cost

-- State the theorem to prove the cost of a child ticket
theorem cost_of_child_ticket (x : ℕ) :
  total_collected_from_adults + children_attended * x = total_collected_cents →
  x = 25 :=
by
  sorry

end cost_of_child_ticket_l221_221167


namespace circle_radius_l221_221500

theorem circle_radius (x y : ℝ) :
  y = (x - 2)^2 ∧ x - 3 = (y + 1)^2 →
  (∃ c d r : ℝ, (c, d) = (3/2, -1/2) ∧ r^2 = 25/4) :=
by
  sorry

end circle_radius_l221_221500


namespace sum_of_roots_of_quadratic_l221_221491

theorem sum_of_roots_of_quadratic (a b c : ℝ) (h_eq : 2 * (X^2) - 8 * X + 6 = 0) : 
  (-b / a) = 4 :=
sorry

end sum_of_roots_of_quadratic_l221_221491


namespace quadratic_has_real_root_l221_221238

theorem quadratic_has_real_root (a : ℝ) : 
  ¬(∀ x : ℝ, x^2 + a * x + a - 1 ≠ 0) :=
sorry

end quadratic_has_real_root_l221_221238


namespace ratio_expression_value_l221_221723

theorem ratio_expression_value (A B C : ℚ) (h_ratio : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
by
  sorry

end ratio_expression_value_l221_221723


namespace imo_42nd_inequality_l221_221064

theorem imo_42nd_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (a^2 + 8 * b * c)) + (b / Real.sqrt (b^2 + 8 * c * a)) + (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 := by
  sorry

end imo_42nd_inequality_l221_221064


namespace functional_equation_solution_l221_221019

theorem functional_equation_solution (f : ℝ → ℝ) (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) :
  (f x = (1/2) * (x + 1 - 1/x - 1/(1-x))) →
  (f x + f (1 / (1 - x)) = x) :=
sorry

end functional_equation_solution_l221_221019


namespace greatest_integer_l221_221131

theorem greatest_integer (y : ℤ) : (8 / 11 : ℝ) > (y / 17 : ℝ) → y ≤ 12 :=
by sorry

end greatest_integer_l221_221131


namespace mutually_exclusive_not_complementary_l221_221239

def group : Finset (String × String) := {("boy1", "boy"), ("boy2", "boy"), ("boy3", "boy"), ("girl1", "girl"), ("girl2", "girl")}
def selection_size : ℕ := 2

def event_at_least_one_boy (s : Finset (String × String)) : Prop :=
  ∃ x ∈ s, x.2 = "boy"

def event_all_girls (s : Finset (String × String)) : Prop :=
  ∀ x ∈ s, x.2 = "girl"

theorem mutually_exclusive_not_complementary : 
  ∃ (s : Finset (String × String)), s.card = selection_size ∧ event_at_least_one_boy s ∧ event_all_girls s :=
sorry

end mutually_exclusive_not_complementary_l221_221239


namespace monotonically_increasing_interval_l221_221471

def f (x : ℝ) (a : ℝ) : ℝ := |2 * x + a| + 3

theorem monotonically_increasing_interval (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f x₁ a ≤ f x₂ a) → a ≥ -2 :=
by
  sorry

end monotonically_increasing_interval_l221_221471


namespace mr_jones_loss_l221_221071

theorem mr_jones_loss :
  ∃ (C_1 C_2 : ℝ), 
    (1.2 = 1.2 * C_1 / 1.2) ∧ 
    (1.2 = 0.8 * C_2) ∧ 
    ((C_1 + C_2) - (2 * 1.2)) = -0.1 :=
by
  sorry

end mr_jones_loss_l221_221071


namespace identify_conic_section_hyperbola_l221_221565

-- Defining the variables and constants in the Lean environment
variable (x y : ℝ)

-- The given equation in function form
def conic_section_eq : Prop := (x - 3) ^ 2 = 4 * (y + 2) ^ 2 + 25

-- The expected type of conic section (Hyperbola)
def is_hyperbola : Prop := 
  ∃ (a b c d e f : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * x^2 - b * y^2 + c * x + d * y + e = f

-- The theorem statement to prove
theorem identify_conic_section_hyperbola (h : conic_section_eq x y) : is_hyperbola x y := by
  sorry

end identify_conic_section_hyperbola_l221_221565


namespace rhombus_area_l221_221630

theorem rhombus_area (d₁ d₂ : ℕ) (h₁ : d₁ = 6) (h₂ : d₂ = 8) : 
  (1 / 2 : ℝ) * d₁ * d₂ = 24 := 
by
  sorry

end rhombus_area_l221_221630


namespace cars_parked_l221_221790

def front_parking_spaces : ℕ := 52
def back_parking_spaces : ℕ := 38
def filled_back_spaces : ℕ := back_parking_spaces / 2
def available_spaces : ℕ := 32
def total_parking_spaces : ℕ := front_parking_spaces + back_parking_spaces
def filled_spaces : ℕ := total_parking_spaces - available_spaces

theorem cars_parked : 
  filled_spaces = 58 := by
  sorry

end cars_parked_l221_221790


namespace KochCurve_MinkowskiDimension_l221_221169

noncomputable def minkowskiDimensionOfKochCurve : ℝ :=
  let N (n : ℕ) := 3 * (4 ^ (n - 1))
  (Real.log 4) / (Real.log 3)

theorem KochCurve_MinkowskiDimension : minkowskiDimensionOfKochCurve = (Real.log 4) / (Real.log 3) := by
  sorry

end KochCurve_MinkowskiDimension_l221_221169


namespace triangle_A_l221_221760

variables {a b c : ℝ}
variables (A B C : ℝ) -- Represent vertices
variables (C1 C2 A1 A2 B1 B2 A' B' C' : ℝ)

-- Definition of equilateral triangle
def is_equilateral_trig (x y z : ℝ) : Prop :=
  dist x y = dist y z ∧ dist y z = dist z x

-- Given conditions
axiom ABC_equilateral : is_equilateral_trig A B C
axiom length_cond_1 : dist A1 A2 = a ∧ dist C B1 = a ∧ dist B C2 = a
axiom length_cond_2 : dist B1 B2 = b ∧ dist A C1 = b ∧ dist C A2 = b
axiom length_cond_3 : dist C1 C2 = c ∧ dist B A1 = c ∧ dist A B2 = c

-- Additional constructions
axiom A'_construction : is_equilateral_trig A' B2 C1
axiom B'_construction : is_equilateral_trig B' C2 A1
axiom C'_construction : is_equilateral_trig C' A2 B1

-- The final proof goal
theorem triangle_A'B'C'_equilateral : is_equilateral_trig A' B' C' :=
sorry

end triangle_A_l221_221760


namespace sqrt_meaningful_range_iff_l221_221228

noncomputable def sqrt_meaningful_range (x : ℝ) : Prop :=
  (∃ r : ℝ, r ≥ 0 ∧ r * r = x - 2023)

theorem sqrt_meaningful_range_iff {x : ℝ} : sqrt_meaningful_range x ↔ x ≥ 2023 :=
by
  sorry

end sqrt_meaningful_range_iff_l221_221228


namespace value_of_expression_l221_221739

variables {A B C : ℚ}

def conditions (A B C : ℚ) : Prop := A / B = 3 / 2 ∧ B / C = 2 / 5

theorem value_of_expression (h : conditions A B C) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
sorry

end value_of_expression_l221_221739


namespace range_of_a_l221_221886

open Real

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) → -1 ≤ a ∧ a ≤ 3 :=
by
  intro h
  -- insert the actual proof here
  sorry

end range_of_a_l221_221886


namespace expected_rolls_in_leap_year_l221_221840

theorem expected_rolls_in_leap_year :
  let E := (3/4) * 1 + (1/4) * (1 + E) in  -- Expected value equation
  E = 4/3 →
  let E_total := E * 366 in
  E_total = 488 :=
by
  sorry

end expected_rolls_in_leap_year_l221_221840


namespace problem1_problem2_l221_221340

-- Proof problem 1
theorem problem1 : (-3)^2 / 3 + abs (-7) + 3 * (-1/3) = 3 :=
by
  sorry

-- Proof problem 2
theorem problem2 : (-1) ^ 2022 - ( (-1/4) - (-1/3) ) / (-1/12) = 2 :=
by
  sorry

end problem1_problem2_l221_221340


namespace compute_expression_l221_221117

theorem compute_expression (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1^3 - 3 * x1 * y1^2 = 2017)
  (h2 : y1^3 - 3 * x1^2 * y1 = 2016)
  (h3 : x2^3 - 3 * x2 * y2^2 = 2017)
  (h4 : y2^3 - 3 * x2^2 * y2 = 2016)
  (h5 : x3^3 - 3 * x3 * y3^2 = 2017)
  (h6 : y3^3 - 3 * x3^2 * y3 = 2016) :
  (2 - x1 / y1) * (2 - x2 / y2) * (2 - x3 / y3) = 26219 / 2016 := 
by
  sorry

end compute_expression_l221_221117


namespace trapezoid_area_l221_221165

theorem trapezoid_area (x y : ℝ) (hx : y^2 + x^2 = 625) (hy : y^2 + (25 - x)^2 = 900) :
  1 / 2 * (11 + 36) * 24 = 564 :=
by
  sorry

end trapezoid_area_l221_221165


namespace females_count_l221_221837

-- Defining variables and constants
variables (P M F : ℕ)
-- The condition given the total population
def town_population := P = 600
-- The condition given the proportion of males
def proportion_of_males := M = P / 3
-- The condition determining the number of females
def number_of_females := F = P - M

-- The theorem stating the number of females is 400
theorem females_count (P M F : ℕ) (h1 : town_population P)
  (h2 : proportion_of_males P M) 
  (h3 : number_of_females P M F) : 
  F = 400 := 
sorry

end females_count_l221_221837


namespace roots_polynomial_expression_l221_221594

theorem roots_polynomial_expression (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : a * b + a * c + b * c = -1)
  (h3 : a * b * c = -2) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = 0 :=
by
  sorry

end roots_polynomial_expression_l221_221594


namespace number_of_sets_satisfying_conditions_l221_221403

open Finset

theorem number_of_sets_satisfying_conditions :
  let M := {M : Finset (Fin 5) // 
              M ⊆ {0, 1, 2, 3, 4} ∧
              M ∩ {0, 1, 2} = {0, 1}} 
  in M.card = 4 :=
by sorry

end number_of_sets_satisfying_conditions_l221_221403


namespace exists_coprime_integers_divisible_l221_221766

theorem exists_coprime_integers_divisible {a b p : ℤ} : ∃ k l : ℤ, gcd k l = 1 ∧ p ∣ (a * k + b * l) :=
by
  sorry

end exists_coprime_integers_divisible_l221_221766


namespace snowfall_rate_in_Hamilton_l221_221338

theorem snowfall_rate_in_Hamilton 
  (initial_depth_Kingston : ℝ := 12.1)
  (rate_Kingston : ℝ := 2.6)
  (initial_depth_Hamilton : ℝ := 18.6)
  (duration : ℕ := 13)
  (final_depth_equal : initial_depth_Kingston + rate_Kingston * duration = initial_depth_Hamilton + duration * x)
  (x : ℝ) :
  x = 2.1 :=
sorry

end snowfall_rate_in_Hamilton_l221_221338


namespace framed_painting_ratio_l221_221823

-- Definitions and conditions
def painting_width : ℕ := 20
def painting_height : ℕ := 30
def frame_side_width (x : ℕ) : ℕ := x
def frame_top_bottom_width (x : ℕ) : ℕ := 3 * x

-- Overall dimensions of the framed painting
def framed_painting_width (x : ℕ) : ℕ := painting_width + 2 * frame_side_width x
def framed_painting_height (x : ℕ) : ℕ := painting_height + 2 * frame_top_bottom_width x

-- Area of the painting
def painting_area : ℕ := painting_width * painting_height

-- Area of the frame
def frame_area (x : ℕ) : ℕ := framed_painting_width x * framed_painting_height x - painting_area

-- Condition that frame area equals painting area
def frame_area_condition (x : ℕ) : Prop := frame_area x = painting_area

-- Theoretical ratio of smaller to larger dimension of the framed painting
def dimension_ratio (x : ℕ) : ℚ := (framed_painting_width x : ℚ) / (framed_painting_height x)

-- The mathematical problem to prove
theorem framed_painting_ratio : ∃ x : ℕ, frame_area_condition x ∧ dimension_ratio x = (4 : ℚ) / 7 :=
by
  sorry

end framed_painting_ratio_l221_221823


namespace minimum_yellow_balls_l221_221503

theorem minimum_yellow_balls (g o y : ℕ) :
  (o ≥ (1/3:ℝ) * g) ∧ (o ≤ (1/4:ℝ) * y) ∧ (g + o ≥ 75) → y ≥ 76 :=
sorry

end minimum_yellow_balls_l221_221503


namespace x_cubed_plus_y_cubed_l221_221406

theorem x_cubed_plus_y_cubed (x y : ℝ) (h1 : x + y = 8) (h2 : x * y = 14) : x^3 + y^3 = 176 :=
sorry

end x_cubed_plus_y_cubed_l221_221406


namespace proposition_contradiction_l221_221866

-- Define the proposition P for natural numbers.
def P (n : ℕ+) : Prop := sorry

theorem proposition_contradiction (h1 : ∀ k : ℕ+, P k → P (k + 1)) (h2 : ¬ P 5) : ¬ P 4 :=
by
  sorry

end proposition_contradiction_l221_221866


namespace fractions_order_l221_221141

theorem fractions_order :
  (21 / 17) < (18 / 13) ∧ (18 / 13) < (16 / 11) := by
  sorry

end fractions_order_l221_221141


namespace b_2023_value_l221_221593

noncomputable def seq (b : ℕ → ℝ) : Prop := 
  ∀ n ≥ 2, b n = b (n - 1) * b (n + 1)

theorem b_2023_value (b : ℕ → ℝ) (h1 : seq b) (h2 : b 1 = 2 + Real.sqrt 5) (h3 : b 1984 = 12 + Real.sqrt 5) : 
  b 2023 = -4/3 + 10 * Real.sqrt 5 / 3 :=
sorry

end b_2023_value_l221_221593


namespace bus_system_carry_per_day_l221_221467

theorem bus_system_carry_per_day (total_people : ℕ) (weeks : ℕ) (days_in_week : ℕ) (people_per_day : ℕ) :
  total_people = 109200000 →
  weeks = 13 →
  days_in_week = 7 →
  people_per_day = total_people / (weeks * days_in_week) →
  people_per_day = 1200000 :=
by
  intros htotal hweeks hdays hcalc
  sorry

end bus_system_carry_per_day_l221_221467


namespace position_1011th_square_l221_221786

-- Define the initial position and transformations
inductive SquarePosition
| ABCD : SquarePosition
| DABC : SquarePosition
| BADC : SquarePosition
| DCBA : SquarePosition

open SquarePosition

def R1 (p : SquarePosition) : SquarePosition :=
  match p with
  | ABCD => DABC
  | DABC => BADC
  | BADC => DCBA
  | DCBA => ABCD

def R2 (p : SquarePosition) : SquarePosition :=
  match p with
  | ABCD => DCBA
  | DCBA => ABCD
  | DABC => BADC
  | BADC => DABC

def transform : ℕ → SquarePosition
| 0 => ABCD
| n + 1 => if n % 2 = 0 then R1 (transform n) else R2 (transform n)

theorem position_1011th_square : transform 1011 = DCBA :=
by {
  sorry
}

end position_1011th_square_l221_221786


namespace regression_lines_intersect_at_average_l221_221119

theorem regression_lines_intersect_at_average
  {x_vals1 x_vals2 : List ℝ} {y_vals1 y_vals2 : List ℝ}
  (n1 : x_vals1.length = 100) (n2 : x_vals2.length = 150)
  (mean_x1 : (List.sum x_vals1 / 100) = s) (mean_x2 : (List.sum x_vals2 / 150) = s)
  (mean_y1 : (List.sum y_vals1 / 100) = t) (mean_y2 : (List.sum y_vals2 / 150) = t)
  (regression_line1 : ℝ → ℝ)
  (regression_line2 : ℝ → ℝ)
  (on_line1 : ∀ x, regression_line1 x = (a1 * x + b1))
  (on_line2 : ∀ x, regression_line2 x = (a2 * x + b2))
  (sample_center1 : regression_line1 s = t)
  (sample_center2 : regression_line2 s = t) :
  regression_line1 s = regression_line2 s := sorry

end regression_lines_intersect_at_average_l221_221119


namespace lcm_pair_eq_sum_l221_221983

theorem lcm_pair_eq_sum (x y : ℕ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : Nat.lcm x y = 1 + 2 * x + 3 * y) :
  (x = 4 ∧ y = 9) ∨ (x = 9 ∧ y = 4) :=
by {
  sorry
}

end lcm_pair_eq_sum_l221_221983


namespace number_of_nonempty_proper_subsets_l221_221789

open Finset

theorem number_of_nonempty_proper_subsets :
  let S := {y ∈ range 7 | ∃ x ∈ range 3, y = 6 - x^2} in
  S = {2, 5, 6} ∧ (card (powerset S) - 2) = 6 :=
by 
  let S := {y ∈ range 7 | ∃ x ∈ range 3, y = 6 - x^2}
  have hs : S = {2, 5, 6} := sorry
  have h_subsets : card (powerset S) = 8 := sorry
  exact ⟨hs, by rw [h_subsets, Nat.sub_eq_of_eq_add]; exact rfl⟩

end number_of_nonempty_proper_subsets_l221_221789


namespace tetrahedrons_from_triangular_prism_l221_221879

theorem tetrahedrons_from_triangular_prism : 
  let n := 6
  let choose4 := Nat.choose n 4
  let coplanar_cases := 3
  choose4 - coplanar_cases = 12 := by
  sorry

end tetrahedrons_from_triangular_prism_l221_221879


namespace women_per_table_l221_221838

theorem women_per_table 
  (total_tables : ℕ)
  (men_per_table : ℕ)
  (total_customers : ℕ) 
  (h_total_tables : total_tables = 6)
  (h_men_per_table : men_per_table = 5)
  (h_total_customers : total_customers = 48) :
  (total_customers - (men_per_table * total_tables)) / total_tables = 3 :=
by
  subst h_total_tables
  subst h_men_per_table
  subst h_total_customers
  sorry

end women_per_table_l221_221838


namespace john_annual_payment_l221_221428

open Real

-- Definitions extracted from the problem:
def epipen_cost : ℝ := 500
def insurance_coverage : ℝ := 0.75
def epipen_frequency_per_year : ℕ := 2
def john_payment_per_epipen : ℝ := epipen_cost * (1 - insurance_coverage)

-- The statement to be proved:
theorem john_annual_payment :
  john_payment_per_epipen * epipen_frequency_per_year = 250 :=
by
  sorry

end john_annual_payment_l221_221428


namespace find_f_of_1_over_3_l221_221405

theorem find_f_of_1_over_3
  (g : ℝ → ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x, g x = 1 - x^2)
  (h2 : ∀ x, x ≠ 0 → f (g x) = (1 - x^2) / x^2) :
  f (1 / 3) = 1 / 2 := by
  sorry -- Proof goes here

end find_f_of_1_over_3_l221_221405


namespace ages_of_Mel_and_Lexi_l221_221062

theorem ages_of_Mel_and_Lexi (M L K : ℤ)
  (h1 : M = K - 3)
  (h2 : L = M + 2)
  (h3 : K = 60) :
  M = 57 ∧ L = 59 :=
  by
    -- Proof steps are omitted.
    sorry

end ages_of_Mel_and_Lexi_l221_221062


namespace find_n_l221_221016

theorem find_n (a b : ℕ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h1 : ∃ n : ℕ, n - 76 = a^3) (h2 : ∃ n : ℕ, n + 76 = b^3) : ∃ n : ℕ, n = 140 :=
by 
  sorry

end find_n_l221_221016


namespace rancher_steers_cows_solution_l221_221329

theorem rancher_steers_cows_solution :
  ∃ (s c : ℕ), s > 0 ∧ c > 0 ∧ (30 * s + 31 * c = 1200) ∧ (s = 9) ∧ (c = 30) :=
by
  sorry

end rancher_steers_cows_solution_l221_221329


namespace stratified_sampling_community_A_l221_221553

theorem stratified_sampling_community_A :
  let A_households := 360
  let B_households := 270
  let C_households := 180
  let total_households := A_households + B_households + C_households
  let total_units := 90
  (A_households : ℕ) / total_households * total_units = 40 :=
by
  let A_households := 360
  let B_households := 270
  let C_households := 180
  let total_households := A_households + B_households + C_households
  let total_units := 90
  have : total_households = 810 := by sorry
  have : (A_households : ℕ) / total_households * total_units = 40 := by sorry
  exact this

end stratified_sampling_community_A_l221_221553


namespace log8_512_l221_221978

theorem log8_512 : log 8 512 = 3 :=
by
  -- Given conditions
  have h1 : 8 = 2^3 := by rfl
  have h2 : 512 = 2^9 := by rfl
  -- Logarithmic statement to solve
  rw [h1, h2]
  -- Power rule application
  have h3 : (2^3)^3 = 2^9 := by exact congr_arg (λ n, 2^n) (by linarith)
  -- Final equality
  exact congr_arg log h3

end log8_512_l221_221978


namespace sunny_ahead_in_second_race_l221_221414

theorem sunny_ahead_in_second_race
  (s w : ℝ)
  (h1 : s / w = 8 / 7) :
  let sunny_new_speed := 0.9 * s
  let distance_sunny_runs := 450
  let distance_windy_runs := 400
  let time_sunny := distance_sunny_runs / sunny_new_speed
  let distance_windy_in_time_sunny := w * time_sunny
  450 - distance_windy_in_time_sunny = 12.5 :=
by
  let sunny_new_speed := 0.9 * s
  let distance_sunny_runs := 450
  let distance_windy_runs := 400
  let time_sunny := distance_sunny_runs / sunny_new_speed
  let distance_windy_in_time_sunny := w * time_sunny
  sorry

end sunny_ahead_in_second_race_l221_221414


namespace quadrilateral_area_l221_221773

-- Define the dimensions of the rectangles
variables (AB BC EF FG : ℝ)
variables (AFCH_area : ℝ)

-- State the conditions explicitly
def conditions : Prop :=
  (AB = 9) ∧ 
  (BC = 5) ∧ 
  (EF = 3) ∧ 
  (FG = 10)

-- State the theorem to prove
theorem quadrilateral_area (h: conditions AB BC EF FG) : 
  AFCH_area = 52.5 := 
sorry

end quadrilateral_area_l221_221773


namespace complex_power_six_l221_221061

theorem complex_power_six (i : ℂ) (hi : i * i = -1) : (1 + i)^6 = -8 * i :=
by
  sorry

end complex_power_six_l221_221061


namespace alpha_beta_sum_eq_l221_221392

theorem alpha_beta_sum_eq (a : ℝ) (h : 1 < a) (α β : ℝ) 
  (hα : α ∈ Set.Ioo (-π / 2) (π / 2))
  (hβ : β ∈ Set.Ioo (-π / 2) (π / 2)) 
  (h_roots : (∀ x, x^2 + 3 * a * x + (3 * a + 1) = 0 → x = Real.tan α ∨ x = Real.tan β)) :
  α + β = -3 * π / 4 := 
sorry

end alpha_beta_sum_eq_l221_221392


namespace Shyam_money_l221_221147

theorem Shyam_money (r g k s : ℕ) 
  (h1 : 7 * g = 17 * r) 
  (h2 : 7 * k = 17 * g)
  (h3 : 11 * s = 13 * k)
  (hr : r = 735) : 
  s = 2119 := 
by
  sorry

end Shyam_money_l221_221147


namespace number_of_blueberries_l221_221820

def total_berries : ℕ := 42
def raspberries : ℕ := total_berries / 2
def blackberries : ℕ := total_berries / 3
def blueberries : ℕ := total_berries - (raspberries + blackberries)

theorem number_of_blueberries :
  blueberries = 7 :=
by
  sorry

end number_of_blueberries_l221_221820


namespace rotate_circle_sectors_l221_221889

theorem rotate_circle_sectors (n : ℕ) (h : n > 0) :
  (∀ i, i < n → ∃ θ : ℝ, θ < (π / (n^2 - n + 1))) →
  ∃ θ : ℝ, 0 < θ ∧ θ < 2 * π ∧
  (∀ i : ℕ, i < n → (θ * i) % (2 * π) > (π / (n^2 - n + 1))) :=
sorry

end rotate_circle_sectors_l221_221889


namespace remainder_sequences_mod_1000_l221_221697

theorem remainder_sequences_mod_1000 :
  ∃ m, (m = 752) ∧ (m % 1000 = 752) ∧ 
  (∃ (a : ℕ → ℕ) (h : ∀ i, 1 ≤ i ∧ i ≤ 6 → (a i) - i % 2 = 1), 
    (∀ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ 6 → a i ≤ a j) ∧ 
    (∀ i, 1 ≤ i ∧ i ≤ 6 → 1 ≤ a i ∧ a i ≤ 1500)
  ) := by
    -- proof would go here
    sorry

end remainder_sequences_mod_1000_l221_221697


namespace smallest_sum_l221_221376

theorem smallest_sum (x y : ℕ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x ≠ y) (eq : 1/x + 1/y = 1/10) : x + y = 45 :=
by
  sorry

end smallest_sum_l221_221376


namespace binom_25_5_l221_221389

theorem binom_25_5 :
  (Nat.choose 23 3 = 1771) ∧
  (Nat.choose 23 4 = 8855) ∧
  (Nat.choose 23 5 = 33649) → 
  Nat.choose 25 5 = 53130 := by
sorry

end binom_25_5_l221_221389


namespace smallest_tree_height_l221_221636

theorem smallest_tree_height (tallest middle smallest : ℝ)
  (h1 : tallest = 108)
  (h2 : middle = (tallest / 2) - 6)
  (h3 : smallest = middle / 4) : smallest = 12 :=
by
  sorry

end smallest_tree_height_l221_221636


namespace minimize_sum_of_squares_at_mean_l221_221549

-- Definitions of the conditions
def P1 (x1 : ℝ) : ℝ := x1
def P2 (x2 : ℝ) : ℝ := x2
def P3 (x3 : ℝ) : ℝ := x3
def P4 (x4 : ℝ) : ℝ := x4
def P5 (x5 : ℝ) : ℝ := x5

-- Definition of the function we want to minimize
def s (P : ℝ) (x1 x2 x3 x4 x5 : ℝ) : ℝ :=
  (P - x1)^2 + (P - x2)^2 + (P - x3)^2 + (P - x4)^2 + (P - x5)^2

-- Proof statement
theorem minimize_sum_of_squares_at_mean (x1 x2 x3 x4 x5 : ℝ) :
  ∃ P : ℝ, P = (x1 + x2 + x3 + x4 + x5) / 5 ∧ 
           ∀ x : ℝ, s P x1 x2 x3 x4 x5 ≤ s x x1 x2 x3 x4 x5 := 
by
  sorry

end minimize_sum_of_squares_at_mean_l221_221549


namespace solution_set_of_inequality_l221_221484

theorem solution_set_of_inequality (x : ℝ) : (∃ x, (0 ≤ x ∧ x < 1) ↔ (x-2)/(x-1) ≥ 2) :=
sorry

end solution_set_of_inequality_l221_221484


namespace options_necessarily_positive_l221_221336

variable (x y z : ℝ)

theorem options_necessarily_positive (h₁ : -1 < x) (h₂ : x < 0) (h₃ : 0 < y) (h₄ : y < 1) (h₅ : 2 < z) (h₆ : z < 3) :
  y + x^2 * z > 0 ∧
  y + x^2 > 0 ∧
  y + y^2 > 0 ∧
  y + 2 * z > 0 := 
  sorry

end options_necessarily_positive_l221_221336


namespace circle_radius_is_six_l221_221480

open Real

theorem circle_radius_is_six
  (r : ℝ)
  (h : 2 * 3 * 2 * π * r = 2 * π * r^2) :
  r = 6 := sorry

end circle_radius_is_six_l221_221480


namespace ratio_expression_value_l221_221726

theorem ratio_expression_value (A B C : ℚ) (h_ratio : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
by
  sorry

end ratio_expression_value_l221_221726


namespace exists_root_in_interval_l221_221689

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x - 3

theorem exists_root_in_interval : ∃ c ∈ Set.Ioo (2 : ℝ) (3 : ℝ), f c = 0 :=
by
  sorry

end exists_root_in_interval_l221_221689


namespace exists_monomial_l221_221303

variables (x y : ℕ) -- Define x and y as natural numbers

theorem exists_monomial :
  ∃ (c : ℕ) (e_x e_y : ℕ), c = 3 ∧ e_x + e_y = 3 ∧ (c * x ^ e_x * y ^ e_y) = (3 * x ^ e_x * y ^ e_y) :=
by
  sorry

end exists_monomial_l221_221303


namespace minimum_order_amount_to_get_discount_l221_221084

theorem minimum_order_amount_to_get_discount 
  (cost_quiche : ℝ) (cost_croissant : ℝ) (cost_biscuit : ℝ) (n_quiches : ℝ) (n_croissants : ℝ) (n_biscuits : ℝ)
  (discount_percent : ℝ) (total_with_discount : ℝ) (min_order_amount : ℝ) :
  cost_quiche = 15.0 → cost_croissant = 3.0 → cost_biscuit = 2.0 →
  n_quiches = 2 → n_croissants = 6 → n_biscuits = 6 →
  discount_percent = 0.10 → total_with_discount = 54.0 →
  (n_quiches * cost_quiche + n_croissants * cost_croissant + n_biscuits * cost_biscuit) * (1 - discount_percent) = total_with_discount →
  min_order_amount = 60.0 :=
by
  sorry

end minimum_order_amount_to_get_discount_l221_221084


namespace sum_of_solutions_l221_221357

theorem sum_of_solutions (x : ℝ) (hx : x + 36 / x = 12) : x = 6 ∨ x = -6 := sorry

end sum_of_solutions_l221_221357


namespace sum_of_a6_and_a7_l221_221391

theorem sum_of_a6_and_a7 (a : ℕ → ℝ) (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (h_sum : a 2 + a 3 + a 10 + a 11 = 48) : a 6 + a 7 = 24 :=
by
  sorry

end sum_of_a6_and_a7_l221_221391


namespace quadratic_condition_l221_221275

theorem quadratic_condition (a : ℝ) :
  (∃ x : ℝ, (a - 1) * x^2 + 4 * x - 3 = 0) → a ≠ 1 :=
by
  sorry

end quadratic_condition_l221_221275


namespace complex_exp_cos_l221_221366

theorem complex_exp_cos (z : ℂ) (α : ℂ) (n : ℕ) (h : z + z⁻¹ = 2 * Complex.cos α) : 
  z^n + z⁻¹^n = 2 * Complex.cos (n * α) :=
by
  sorry

end complex_exp_cos_l221_221366


namespace computation_problem_points_l221_221164

/-- A teacher gives out a test of 30 problems. Each computation problem is worth some points, and
each word problem is worth 5 points. The total points you can receive on the test is 110 points,
and there are 20 computation problems. How many points is each computation problem worth? -/

theorem computation_problem_points (x : ℕ) (total_problems : ℕ := 30) (word_problem_points : ℕ := 5)
    (total_points : ℕ := 110) (computation_problems : ℕ := 20) :
    20 * x + (total_problems - computation_problems) * word_problem_points = total_points → x = 3 :=
by
  intro h
  sorry

end computation_problem_points_l221_221164


namespace min_abs_sum_l221_221660

open Real

theorem min_abs_sum : ∃ (x : ℝ), (∀ y : ℝ, ∑ z in [| y + 3, y + 6, y + 7].toFinset, abs z ≥ -2) :=
by
  sorry

end min_abs_sum_l221_221660


namespace pure_imaginary_iff_a_eq_2_l221_221764

theorem pure_imaginary_iff_a_eq_2 (a : ℝ) : (∃ k : ℝ, (∃ x : ℝ, (2-a) / 2 = x ∧ x = 0) ∧ (2+a)/2 = k ∧ k ≠ 0) ↔ a = 2 :=
by
  sorry

end pure_imaginary_iff_a_eq_2_l221_221764


namespace find_explicit_formula_range_of_k_l221_221600

variable (a b x k : ℝ)

def f (x : ℝ) : ℝ := a * x ^ 3 - b * x + 4

theorem find_explicit_formula (h_extremum_at_2 : f a b 2 = -4 / 3 ∧ (3 * a * 4 - b = 0)) :
  ∃ a b, f a b x = (1 / 3) * x ^ 3 - 4 * x + 4 :=
sorry

theorem range_of_k (h_extremum_at_2 : f (1 / 3) 4 2 = -4 / 3) :
  ∃ k, -4 / 3 < k ∧ k < 8 / 3 :=
sorry

end find_explicit_formula_range_of_k_l221_221600


namespace average_salary_8800_l221_221918

theorem average_salary_8800 
  (average_salary_start : ℝ)
  (salary_jan : ℝ)
  (salary_may : ℝ)
  (total_salary : ℝ)
  (avg_specific_months : ℝ)
  (jan_salary_rate : average_salary_start * 4 = total_salary)
  (may_salary_rate : total_salary - salary_jan = total_salary - 3300)
  (final_salary_rate : total_salary - salary_jan + salary_may = 35200)
  (specific_avg_calculation : 35200 / 4 = avg_specific_months)
  : avg_specific_months = 8800 :=
sorry -- Proof steps will be filled in later

end average_salary_8800_l221_221918


namespace exists_pairs_of_stops_l221_221639

def problem := ∃ (A1 B1 A2 B2 : Fin 6) (h1 : A1 < B1) (h2 : A2 < B2),
  (A1 ≠ A2 ∧ A1 ≠ B2 ∧ B1 ≠ A2 ∧ B1 ≠ B2) ∧
  ¬(∃ (a b : Fin 6), A1 = a ∧ B1 = b ∧ A2 = a ∧ B2 = b) -- such that no passenger boards at A1 and alights at B1
                                                              -- and no passenger boards at A2 and alights at B2.

theorem exists_pairs_of_stops (n : ℕ) (stops : Fin n) (max_passengers : ℕ) 
  (h : n = 6 ∧ max_passengers = 5 ∧ 
  ∀ (a b : Fin n), a < b → a < stops ∧ b < stops) : problem :=
sorry

end exists_pairs_of_stops_l221_221639


namespace milly_needs_flamingoes_l221_221602

theorem milly_needs_flamingoes
  (flamingo_feathers : ℕ)
  (pluck_percent : ℚ)
  (num_boas : ℕ)
  (feathers_per_boa : ℕ)
  (pluckable_feathers_per_flamingo : ℕ)
  (total_feathers_needed : ℕ)
  (num_flamingoes : ℕ)
  (h1 : flamingo_feathers = 20)
  (h2 : pluck_percent = 0.25)
  (h3 : num_boas = 12)
  (h4 : feathers_per_boa = 200)
  (h5 : pluckable_feathers_per_flamingo = flamingo_feathers * pluck_percent)
  (h6 : total_feathers_needed = num_boas * feathers_per_boa)
  (h7 : num_flamingoes = total_feathers_needed / pluckable_feathers_per_flamingo)
  : num_flamingoes = 480 := 
by
  sorry

end milly_needs_flamingoes_l221_221602


namespace janet_total_owed_l221_221897

def warehouseHourlyWage : ℝ := 15
def managerHourlyWage : ℝ := 20
def numWarehouseWorkers : ℕ := 4
def numManagers : ℕ := 2
def workDaysPerMonth : ℕ := 25
def workHoursPerDay : ℕ := 8
def ficaTaxRate : ℝ := 0.10

theorem janet_total_owed : 
  let warehouseWorkerMonthlyWage := warehouseHourlyWage * workDaysPerMonth * workHoursPerDay
  let managerMonthlyWage := managerHourlyWage * workDaysPerMonth * workHoursPerDay
  let totalMonthlyWages := (warehouseWorkerMonthlyWage * numWarehouseWorkers) + (managerMonthlyWage * numManagers)
  let ficaTaxes := totalMonthlyWages * ficaTaxRate
  let totalAmountOwed := totalMonthlyWages + ficaTaxes
  totalAmountOwed = 22000 := by
  sorry

end janet_total_owed_l221_221897


namespace cycle_final_selling_price_l221_221155

-- Lean 4 statement capturing the problem definition and final selling price
theorem cycle_final_selling_price (original_price : ℝ) (initial_discount_rate : ℝ) 
  (loss_rate : ℝ) (exchange_discount_rate : ℝ) (final_price : ℝ) :
  original_price = 1400 →
  initial_discount_rate = 0.05 →
  loss_rate = 0.25 →
  exchange_discount_rate = 0.10 →
  final_price = 
    (original_price * (1 - initial_discount_rate) * (1 - loss_rate) * (1 - exchange_discount_rate)) →
  final_price = 897.75 :=
by
  sorry

end cycle_final_selling_price_l221_221155


namespace slope_range_l221_221830

theorem slope_range {A : ℝ × ℝ} (k : ℝ) : 
  A = (1, 1) → (0 < 1 - k ∧ 1 - k < 2) → -1 < k ∧ k < 1 :=
by
  sorry

end slope_range_l221_221830


namespace calc_log_expression_l221_221006

theorem calc_log_expression : 2 * Real.log 5 + Real.log 4 = 2 :=
by
  sorry

end calc_log_expression_l221_221006


namespace last_two_digits_A_pow_20_l221_221759

/-- 
Proof that for any even number A not divisible by 10, 
the last two digits of A^20 are 76.
--/
theorem last_two_digits_A_pow_20 (A : ℕ) (h_even : A % 2 = 0) (h_not_div_by_10 : A % 10 ≠ 0) : 
  (A ^ 20) % 100 = 76 :=
by
  sorry

end last_two_digits_A_pow_20_l221_221759


namespace exists_q_no_zero_in_decimal_l221_221077

theorem exists_q_no_zero_in_decimal : ∃ q : ℕ, ∀ (d : ℕ), q * 2 ^ 1967 ≠ 10 * d := 
sorry

end exists_q_no_zero_in_decimal_l221_221077


namespace length_of_field_l221_221668

variable (w l : ℝ)
variable (H1 : l = 2 * w)
variable (pond_area : ℝ := 64)
variable (field_area : ℝ := l * w)
variable (H2 : pond_area = (1 / 98) * field_area)

theorem length_of_field : l = 112 :=
by
  sorry

end length_of_field_l221_221668


namespace min_abs_sum_value_l221_221658

def abs_sum (x : ℝ) := |x + 3| + |x + 6| + |x + 7|

theorem min_abs_sum_value : ∃ x : ℝ, abs_sum x = 4 ∧ ∀ y : ℝ, abs_sum y ≥ abs_sum x := 
by 
  use -6
  have abs_sum_eq : abs_sum (-6) = 4 := by
    simp [abs_sum]
  -- Other conditions ensuring it is the minimum
  sorry

end min_abs_sum_value_l221_221658


namespace present_age_of_son_l221_221323

-- Define variables for the current ages of the son and the man (father).
variables (S M : ℕ)

-- Define the conditions:
-- The man is 35 years older than his son.
def condition1 : Prop := M = S + 35

-- In two years, the man's age will be twice the age of his son.
def condition2 : Prop := M + 2 = 2 * (S + 2)

-- The theorem that we need to prove:
theorem present_age_of_son : condition1 S M ∧ condition2 S M → S = 33 :=
by
  -- Add sorry to skip the proof.
  sorry

end present_age_of_son_l221_221323


namespace determine_x_2y_l221_221049

theorem determine_x_2y (x y : ℝ) (h1 : 2 * x + y = 7) (h2 : (x + y) / 3 = 5 / 3) : x + 2 * y = 8 :=
sorry

end determine_x_2y_l221_221049


namespace main_theorem_l221_221552

-- Define even functions
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define odd functions
def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x, g x = -g (-x)

-- Given conditions
variable (f g : ℝ → ℝ)
variable (h1 : is_even_function f)
variable (h2 : is_odd_function g)
variable (h3 : ∀ x, g x = f (x - 1))

-- Theorem to prove
theorem main_theorem : f 2017 + f 2019 = 0 := sorry

end main_theorem_l221_221552


namespace value_of_expression_l221_221734

variables {A B C : ℚ}

def conditions (A B C : ℚ) : Prop := A / B = 3 / 2 ∧ B / C = 2 / 5

theorem value_of_expression (h : conditions A B C) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
sorry

end value_of_expression_l221_221734


namespace unique_solution_for_2_pow_m_plus_1_eq_n_square_l221_221346

theorem unique_solution_for_2_pow_m_plus_1_eq_n_square (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  2 ^ m + 1 = n ^ 2 → (m = 3 ∧ n = 3) :=
by {
  sorry
}

end unique_solution_for_2_pow_m_plus_1_eq_n_square_l221_221346


namespace omega_range_for_monotonically_decreasing_l221_221200

noncomputable def f (ω x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4)

theorem omega_range_for_monotonically_decreasing
  (ω : ℝ)
  (hω : ω > 0)
  (h_decreasing : ∀ x ∈ Set.Ioo (Real.pi / 2) Real.pi, f ω x < f ω (x + 1e-6)) :
  1/2 ≤ ω ∧ ω ≤ 5/4 :=
by
  sorry

end omega_range_for_monotonically_decreasing_l221_221200


namespace apples_given_by_Susan_l221_221459

theorem apples_given_by_Susan (x y final_apples : ℕ) (h1 : y = 9) (h2 : final_apples = 17) (h3: final_apples = y + x) : x = 8 := by
  sorry

end apples_given_by_Susan_l221_221459


namespace probability_heart_king_l221_221932

theorem probability_heart_king :
  let total_cards := 52
  let total_kings := 4
  let hearts_count := 13
  let king_of_hearts := 1 in
  let prob_king_of_hearts_first := (1 : ℚ) / total_cards
  let prob_other_heart_first := (hearts_count - king_of_hearts : ℚ) / total_cards
  let prob_king_second_if_king_heart_first := (total_kings - king_of_hearts : ℚ) / (total_cards - 1)
  let prob_king_second_if_other_heart_first := (total_kings : ℚ) / (total_cards - 1) in
  prob_king_of_hearts_first * prob_king_second_if_king_heart_first +
  prob_other_heart_first * prob_king_second_if_other_heart_first = (1 : ℚ) / total_cards :=
by sorry

end probability_heart_king_l221_221932


namespace find_c_for_two_zeros_l221_221785

noncomputable def f (x c : ℝ) : ℝ := x^3 - 3*x + c

theorem find_c_for_two_zeros (c : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 c = 0 ∧ f x2 c = 0) ↔ c = -2 ∨ c = 2 :=
sorry

end find_c_for_two_zeros_l221_221785


namespace sum_of_numbers_is_216_l221_221816

-- Define the conditions and what needs to be proved.
theorem sum_of_numbers_is_216 
  (x : ℕ) 
  (h_lcm : Nat.lcm (2 * x) (Nat.lcm (3 * x) (7 * x)) = 126) : 
  2 * x + 3 * x + 7 * x = 216 :=
by
  sorry

end sum_of_numbers_is_216_l221_221816


namespace cannot_use_diff_of_squares_l221_221515

def diff_of_squares (a b : ℤ) : ℤ := a^2 - b^2

theorem cannot_use_diff_of_squares (x y : ℤ) : 
  ¬ ( ((-x + y) * (x - y)) = diff_of_squares (x - y) (0) ) :=
by {
  sorry
}

end cannot_use_diff_of_squares_l221_221515


namespace Jane_shopping_oranges_l221_221896

theorem Jane_shopping_oranges 
  (o a : ℕ)
  (h1 : a + o = 5)
  (h2 : 30 * a + 45 * o + 20 = n)
  (h3 : ∃ k : ℕ, n = 100 * k) : 
  o = 2 :=
by
  sorry

end Jane_shopping_oranges_l221_221896


namespace flag_count_l221_221619

def colors := 3

def stripes := 3

noncomputable def number_of_flags (colors stripes : ℕ) : ℕ :=
  colors ^ stripes

theorem flag_count : number_of_flags colors stripes = 27 :=
by
  -- sorry is used to skip the actual proof steps
  sorry

end flag_count_l221_221619


namespace triangle_problem_l221_221704

variables {a b c A B C : ℝ}

-- The conditions
def triangle_conditions (a b c A B C : ℝ) : Prop :=
  c * Real.cos A + (√3) * c * Real.sin A - b - a = 0

-- The problem statement to prove
theorem triangle_problem (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
    (h4 : triangle_conditions a b c A B C) :
  C = 60 * Real.pi / 180 ∧ (c = 1 → 0.5 * a * b * Real.sin C ≤ sqrt 3 / 4) :=
by
  sorry

end triangle_problem_l221_221704


namespace average_speed_is_65_l221_221794

-- Definitions based on the problem's conditions
def speed_first_hour : ℝ := 100 -- 100 km in the first hour
def speed_second_hour : ℝ := 30 -- 30 km in the second hour
def total_distance : ℝ := speed_first_hour + speed_second_hour -- total distance
def total_time : ℝ := 2 -- total time in hours (1 hour + 1 hour)

-- Problem: prove that the average speed is 65 km/h
theorem average_speed_is_65 : (total_distance / total_time) = 65 := by
  sorry

end average_speed_is_65_l221_221794


namespace min_fence_posts_needed_l221_221160

-- Definitions for the problem conditions
def area_length : ℕ := 72
def regular_side : ℕ := 30
def sloped_side : ℕ := 33
def interval : ℕ := 15

-- The property we want to prove
theorem min_fence_posts_needed : 3 * ((sloped_side + interval - 1) / interval) + 3 * ((regular_side + interval - 1) / interval) = 6 := 
by
  sorry

end min_fence_posts_needed_l221_221160


namespace cost_of_fencing_is_8750_rsquare_l221_221669

variable (l w : ℝ)
variable (area : ℝ := 7500)
variable (cost_per_meter : ℝ := 0.25)
variable (ratio_lw : ℝ := 4/3)

theorem cost_of_fencing_is_8750_rsquare :
  (l / w = ratio_lw) → 
  (l * w = area) → 
  (2 * (l + w) * cost_per_meter = 87.50) :=
by 
  intros h1 h2
  sorry

end cost_of_fencing_is_8750_rsquare_l221_221669


namespace original_price_eq_36_l221_221826

-- Definitions for the conditions
def first_cup_price (x : ℕ) : ℕ := x
def second_cup_price (x : ℕ) : ℕ := x / 2
def third_cup_price : ℕ := 3
def total_cost (x : ℕ) : ℕ := x + (x / 2) + third_cup_price
def average_price (total : ℕ) : ℕ := total / 3

-- The proof statement
theorem original_price_eq_36 (x : ℕ) (h : total_cost x = 57) : x = 36 :=
  sorry

end original_price_eq_36_l221_221826


namespace coffee_ounces_per_cup_l221_221589

theorem coffee_ounces_per_cup
  (persons : ℕ)
  (cups_per_person_per_day : ℕ)
  (cost_per_ounce : ℝ)
  (total_spent_per_week : ℝ)
  (total_cups_per_day : ℕ)
  (total_cups_per_week : ℕ)
  (total_ounces : ℝ)
  (ounces_per_cup : ℝ) :
  persons = 4 →
  cups_per_person_per_day = 2 →
  cost_per_ounce = 1.25 →
  total_spent_per_week = 35 →
  total_cups_per_day = persons * cups_per_person_per_day →
  total_cups_per_week = total_cups_per_day * 7 →
  total_ounces = total_spent_per_week / cost_per_ounce →
  ounces_per_cup = total_ounces / total_cups_per_week →
  ounces_per_cup = 0.5 :=
by
  sorry

end coffee_ounces_per_cup_l221_221589


namespace find_vertical_shift_l221_221005

theorem find_vertical_shift (A B C D : ℝ) (h1 : ∀ x, -3 ≤ A * Real.cos (B * x + C) + D ∧ A * Real.cos (B * x + C) + D ≤ 5) :
  D = 1 :=
by
  -- Here's where the proof would go
  sorry

end find_vertical_shift_l221_221005


namespace min_distance_from_origin_l221_221703

-- Define the condition of the problem
def condition (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x + 6 * y + 4 = 0

-- Statement of the problem in Lean 4
theorem min_distance_from_origin (x y : ℝ) (h : condition x y) : 
  ∃ m : ℝ, m = Real.sqrt (x^2 + y^2) ∧ m = Real.sqrt 13 - 3 := 
sorry

end min_distance_from_origin_l221_221703


namespace num_rows_of_gold_bars_l221_221457

-- Definitions from the problem conditions
def num_bars_per_row : ℕ := 20
def total_worth : ℕ := 1600000

-- Statement to prove
theorem num_rows_of_gold_bars :
  (total_worth / (total_worth / num_bars_per_row)) = 1 := 
by sorry

end num_rows_of_gold_bars_l221_221457


namespace find_x_plus_y_l221_221036

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.cos y = 3005) (h2 : x + 3005 * Real.sin y = 3004) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) : x + y = 3004 :=
by 
  sorry

end find_x_plus_y_l221_221036


namespace money_raised_is_correct_l221_221154

noncomputable def total_money_raised : ℝ :=
  let ticket_sales := 120 * 2.50 + 80 * 4.50 + 40 * 8.00 + 15 * 14.00
  let donations := 3 * 20.00 + 2 * 55.00 + 75.00 + 95.00 + 150.00
  ticket_sales + donations

theorem money_raised_is_correct :
  total_money_raised = 1680 := by
  sorry

end money_raised_is_correct_l221_221154


namespace minimum_area_of_Archimedean_triangle_l221_221620

-- Define the problem statement with necessary conditions
theorem minimum_area_of_Archimedean_triangle (p : ℝ) (hp : p > 0) :
  ∃ (ABQ_area : ℝ), ABQ_area = p^2 ∧ 
    (∀ (A B Q : ℝ × ℝ), 
      (A.2 ^ 2 = 2 * p * A.1) ∧
      (B.2 ^ 2 = 2 * p * B.1) ∧
      (0, 0) = (p / 2, p / 2) ∧
      (Q.2 = 0) → 
      ABQ_area = p^2) :=
sorry

end minimum_area_of_Archimedean_triangle_l221_221620


namespace complement_union_correct_l221_221712

-- Defining the sets U, A, and B
def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

-- Theorem statement
theorem complement_union_correct : (U \ A) ∪ B = {2, 3, 4} := by
  sorry

end complement_union_correct_l221_221712


namespace probability_heart_king_l221_221931

theorem probability_heart_king :
  let total_cards := 52
  let total_kings := 4
  let hearts_count := 13
  let king_of_hearts := 1 in
  let prob_king_of_hearts_first := (1 : ℚ) / total_cards
  let prob_other_heart_first := (hearts_count - king_of_hearts : ℚ) / total_cards
  let prob_king_second_if_king_heart_first := (total_kings - king_of_hearts : ℚ) / (total_cards - 1)
  let prob_king_second_if_other_heart_first := (total_kings : ℚ) / (total_cards - 1) in
  prob_king_of_hearts_first * prob_king_second_if_king_heart_first +
  prob_other_heart_first * prob_king_second_if_other_heart_first = (1 : ℚ) / total_cards :=
by sorry

end probability_heart_king_l221_221931


namespace tips_fraction_l221_221839

-- Define the conditions
variables (S T : ℝ) (h : T = (2 / 4) * S)

-- The statement to be proved
theorem tips_fraction : (T / (S + T)) = 1 / 3 :=
by
  sorry

end tips_fraction_l221_221839


namespace find_fake_coin_l221_221116

def coin_value (n : Nat) : Nat :=
  match n with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 5
  | _ => 0

def coin_weight (n : Nat) : Nat :=
  match n with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 5
  | _ => 0

def is_fake (weight : Nat) : Prop :=
  weight ≠ coin_weight 1 ∧ weight ≠ coin_weight 2 ∧ weight ≠ coin_weight 3 ∧ weight ≠ coin_weight 4

theorem find_fake_coin :
  ∃ (n : Nat) (w : Nat), (is_fake w) → ∃! (m : Nat), m ≠ w ∧ (m = coin_weight 1 ∨ m = coin_weight 2 ∨ m = coin_weight 3 ∨ m = coin_weight 4) := 
sorry

end find_fake_coin_l221_221116


namespace basketball_tournament_l221_221800

theorem basketball_tournament (x : ℕ) 
  (h1 : ∀ n, ((n * (n - 1)) / 2) = 28 -> n = x) 
  (h2 : (x * (x - 1)) / 2 = 28) : 
  (1 / 2 : ℚ) * x * (x - 1) = 28 :=
by 
  sorry

end basketball_tournament_l221_221800


namespace profit_percentage_before_decrease_l221_221184

-- Defining the conditions as Lean definitions
def newManufacturingCost : ℝ := 50
def oldManufacturingCost : ℝ := 80
def profitPercentageNew : ℝ := 0.5

-- Defining the problem as a theorem in Lean
theorem profit_percentage_before_decrease
  (P : ℝ)
  (hP : profitPercentageNew * P = P - newManufacturingCost) :
  ((P - oldManufacturingCost) / P) * 100 = 20 := 
by
  sorry

end profit_percentage_before_decrease_l221_221184


namespace system_of_equations_l221_221305

theorem system_of_equations (x y : ℝ) (h1 : 3 * x + 210 = 5 * y) (h2 : 10 * y - 10 * x = 100) :
    (3 * x + 210 = 5 * y) ∧ (10 * y - 10 * x = 100) := by
  sorry

end system_of_equations_l221_221305


namespace probability_two_red_two_blue_l221_221673

theorem probability_two_red_two_blue :
  let total_marbles := 20
  let red_marbles := 12
  let blue_marbles := 8
  let total_ways_to_choose_4 := Nat.choose total_marbles 4
  let ways_to_choose_2_red := Nat.choose red_marbles 2
  let ways_to_choose_2_blue := Nat.choose blue_marbles 2
  (ways_to_choose_2_red * ways_to_choose_2_blue : ℚ) / total_ways_to_choose_4 = 56 / 147 := 
by {
  sorry
}

end probability_two_red_two_blue_l221_221673


namespace roots_of_equation_l221_221856

theorem roots_of_equation :
  ∀ x : ℝ, (x^4 + x^2 - 20 = 0) ↔ (x = 2 ∨ x = -2) :=
by
  -- This will be the proof.
  -- We are claiming that x is a root of the polynomial if and only if x = 2 or x = -2.
  sorry

end roots_of_equation_l221_221856


namespace largest_digit_for_divisibility_l221_221807

theorem largest_digit_for_divisibility (N : ℕ) (h1 : N % 2 = 0) (h2 : (3 + 6 + 7 + 2 + N) % 3 = 0) : N = 6 :=
sorry

end largest_digit_for_divisibility_l221_221807


namespace component_unqualified_l221_221318

theorem component_unqualified :
  ∀ (φ : ℝ), (19.98 ≤ φ ∧ φ ≤ 20.02) → ¬(φ = 19.9) → True :=
by
  intro φ
  intro h
  intro h'
  -- skip proof
  sorry

end component_unqualified_l221_221318
