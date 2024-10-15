import Mathlib

namespace NUMINAMATH_GPT_term_with_largest_binomial_coeffs_and_largest_coefficient_l1365_136540

theorem term_with_largest_binomial_coeffs_and_largest_coefficient :
  ∀ x : ℝ,
    (∀ k : ℕ, k = 2 → (Nat.choose 5 k) * (x ^ (2 / 3)) ^ (5 - k) * (3 * x ^ 2) ^ k = 90 * x ^ 6) ∧
    (∀ k : ℕ, k = 3 → (Nat.choose 5 k) * (x ^ (2 / 3)) ^ (5 - k) * (3 * x ^ 2) ^ k = 270 * x ^ (22 / 3)) ∧
    (∀ r : ℕ, r = 4 → (Nat.choose 5 4) * (x ^ (2 / 3)) ^ (5 - 4) * (3 * x ^ 2) ^ 4 = 405 * x ^ (26 / 3)) :=
by sorry

end NUMINAMATH_GPT_term_with_largest_binomial_coeffs_and_largest_coefficient_l1365_136540


namespace NUMINAMATH_GPT_factor_expression_l1365_136599

theorem factor_expression (a : ℝ) :
  (8 * a^3 + 105 * a^2 + 7) - (-9 * a^3 + 16 * a^2 - 14) = a^2 * (17 * a + 89) + 21 :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1365_136599


namespace NUMINAMATH_GPT_perfect_square_trinomial_l1365_136586

theorem perfect_square_trinomial (a : ℝ) :
  (∃ m : ℝ, (x^2 + (a-1)*x + 9) = (x + m)^2) → (a = 7 ∨ a = -5) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l1365_136586


namespace NUMINAMATH_GPT_symmetry_of_transformed_graphs_l1365_136558

noncomputable def y_eq_f_x_symmetric_line (f : ℝ → ℝ) : Prop :=
∀ (x : ℝ), f (x - 19) = f (99 - x) ↔ x = 59

theorem symmetry_of_transformed_graphs (f : ℝ → ℝ) :
  y_eq_f_x_symmetric_line f :=
by {
  sorry
}

end NUMINAMATH_GPT_symmetry_of_transformed_graphs_l1365_136558


namespace NUMINAMATH_GPT_technical_class_average_age_l1365_136509

noncomputable def average_age_in_technical_class : ℝ :=
  let average_age_arts := 21
  let num_arts_classes := 8
  let num_technical_classes := 5
  let overall_average_age := 19.846153846153847
  let total_classes := num_arts_classes + num_technical_classes
  let total_age_university := overall_average_age * total_classes
  ((total_age_university - (average_age_arts * num_arts_classes)) / num_technical_classes)

theorem technical_class_average_age :
  average_age_in_technical_class = 990.4 :=
by
  sorry  -- Proof to be provided

end NUMINAMATH_GPT_technical_class_average_age_l1365_136509


namespace NUMINAMATH_GPT_angle_BAC_eq_angle_DAE_l1365_136531

-- Define types and points A, B, C, D, E
variables (A B C D E : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]
variables (P Q R S T : Point)

-- Define angles
variable {α β γ δ θ ω : Angle}

-- Establish the conditions
axiom angle_ABC_eq_angle_ADE : α = θ
axiom angle_AEC_eq_angle_ADB : β = ω

-- State the theorem
theorem angle_BAC_eq_angle_DAE
  (h1 : α = θ) -- Given \(\angle ABC = \angle ADE\)
  (h2 : β = ω) -- Given \(\angle AEC = \angle ADB\)
  : γ = δ := sorry

end NUMINAMATH_GPT_angle_BAC_eq_angle_DAE_l1365_136531


namespace NUMINAMATH_GPT_clock_angle_230_l1365_136539

theorem clock_angle_230 (h12 : ℕ := 12) (deg360 : ℕ := 360) 
  (hour_mark_deg : ℕ := deg360 / h12) (hour_halfway : ℕ := hour_mark_deg / 2)
  (hour_deg_230 : ℕ := hour_mark_deg * 3) (total_angle : ℕ := hour_halfway + hour_deg_230) :
  total_angle = 105 := 
by
  sorry

end NUMINAMATH_GPT_clock_angle_230_l1365_136539


namespace NUMINAMATH_GPT_citizen_income_l1365_136545

theorem citizen_income (I : ℝ) 
  (h1 : I > 0)
  (h2 : 0.12 * 40000 + 0.20 * (I - 40000) = 8000) : 
  I = 56000 := 
sorry

end NUMINAMATH_GPT_citizen_income_l1365_136545


namespace NUMINAMATH_GPT_increasing_interval_of_f_l1365_136556

noncomputable def f (x : ℝ) : ℝ := x^2 - 6 * x

theorem increasing_interval_of_f :
  ∀ x : ℝ, 3 ≤ x → ∀ y : ℝ, 3 ≤ y → x < y → f x < f y := 
sorry

end NUMINAMATH_GPT_increasing_interval_of_f_l1365_136556


namespace NUMINAMATH_GPT_same_terminal_side_l1365_136534

theorem same_terminal_side : ∃ k : ℤ, 36 + k * 360 = -324 :=
by
  use -1
  linarith

end NUMINAMATH_GPT_same_terminal_side_l1365_136534


namespace NUMINAMATH_GPT_smallest_possible_value_of_c_l1365_136591

/-- 
Given three integers \(a, b, c\) with \(a < b < c\), 
such that they form an arithmetic progression (AP) with the property that \(2b = a + c\), 
and form a geometric progression (GP) with the property that \(c^2 = ab\), 
prove that \(c = 2\) is the smallest possible value of \(c\).
-/
theorem smallest_possible_value_of_c :
  ∃ a b c : ℤ, a < b ∧ b < c ∧ 2 * b = a + c ∧ c^2 = a * b ∧ c = 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_value_of_c_l1365_136591


namespace NUMINAMATH_GPT_ratio_P_S_l1365_136502

theorem ratio_P_S (S N P : ℝ) 
  (hN : N = S / 4) 
  (hP : P = N / 4) : 
  P / S = 1 / 16 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_P_S_l1365_136502


namespace NUMINAMATH_GPT_teresa_spends_40_dollars_l1365_136518

-- Definitions of the conditions
def sandwich_cost : ℝ := 7.75
def num_sandwiches : ℝ := 2

def salami_cost : ℝ := 4.00

def brie_cost : ℝ := 3 * salami_cost

def olives_cost_per_pound : ℝ := 10.00
def amount_of_olives : ℝ := 0.25

def feta_cost_per_pound : ℝ := 8.00
def amount_of_feta : ℝ := 0.5

def french_bread_cost : ℝ := 2.00

-- Total cost calculation
def total_cost : ℝ :=
  num_sandwiches * sandwich_cost + salami_cost + brie_cost + olives_cost_per_pound * amount_of_olives + feta_cost_per_pound * amount_of_feta + french_bread_cost

-- Proof statement
theorem teresa_spends_40_dollars :
  total_cost = 40.0 :=
by
  sorry

end NUMINAMATH_GPT_teresa_spends_40_dollars_l1365_136518


namespace NUMINAMATH_GPT_range_of_m_l1365_136584

noncomputable def p (m : ℝ) : Prop :=
  (m^2 - 4 > 0) ∧ (m < 0)

noncomputable def q (m : ℝ) : Prop :=
  (16*(m-2)^2 - 16 < 0)

theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) → (m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_m_l1365_136584


namespace NUMINAMATH_GPT_soda_cans_purchase_l1365_136524

noncomputable def cans_of_soda (S Q D : ℕ) : ℕ :=
  10 * D * S / Q

theorem soda_cans_purchase (S Q D : ℕ) :
  (1 : ℕ) * 10 * D / Q = (10 * D * S) / Q := by
  sorry

end NUMINAMATH_GPT_soda_cans_purchase_l1365_136524


namespace NUMINAMATH_GPT_perimeter_of_floor_l1365_136598

-- Define the side length of the room's floor
def side_length : ℕ := 5

-- Define the formula for the perimeter of a square
def perimeter_of_square (side : ℕ) : ℕ := 4 * side

-- State the theorem: the perimeter of the floor of the room is 20 meters
theorem perimeter_of_floor : perimeter_of_square side_length = 20 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_floor_l1365_136598


namespace NUMINAMATH_GPT_max_candy_leftover_l1365_136557

theorem max_candy_leftover (x : ℕ) : (∃ k : ℕ, x = 12 * k + 11) → (x % 12 = 11) :=
by
  sorry

end NUMINAMATH_GPT_max_candy_leftover_l1365_136557


namespace NUMINAMATH_GPT_add_decimals_l1365_136579

theorem add_decimals :
  5.467 + 3.92 = 9.387 :=
by
  sorry

end NUMINAMATH_GPT_add_decimals_l1365_136579


namespace NUMINAMATH_GPT_C_plus_D_l1365_136581

theorem C_plus_D (C D : ℝ) (h : ∀ x : ℝ, x ≠ 3 → C / (x - 3) + D * (x + 2) = (-4 * x^2 + 18 * x + 32) / (x - 3)) : 
  C + D = 28 := sorry

end NUMINAMATH_GPT_C_plus_D_l1365_136581


namespace NUMINAMATH_GPT_largest_possible_b_l1365_136574

theorem largest_possible_b (a b c : ℕ) (h₁ : 1 < c) (h₂ : c < b) (h₃ : b < a) (h₄ : a * b * c = 360): b = 12 :=
by
  sorry

end NUMINAMATH_GPT_largest_possible_b_l1365_136574


namespace NUMINAMATH_GPT_find_f_five_thirds_l1365_136527

variable {R : Type*} [LinearOrderedField R]

-- Define the odd function and its properties
variable {f : R → R}
variable (oddf : ∀ x : R, f (-x) = -f x)
variable (propf : ∀ x : R, f (1 + x) = f (-x))
variable (val : f (- (1 / 3 : R)) = 1 / 3)

theorem find_f_five_thirds : f (5 / 3 : R) = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_find_f_five_thirds_l1365_136527


namespace NUMINAMATH_GPT_tom_bought_8_kg_of_apples_l1365_136596

/-- 
   Given:
   - The cost of apples is 70 per kg.
   - 9 kg of mangoes at a rate of 55 per kg.
   - Tom paid a total of 1055.

   Prove that Tom purchased 8 kg of apples.
 -/
theorem tom_bought_8_kg_of_apples 
  (A : ℕ) 
  (h1 : 70 * A + 55 * 9 = 1055) : 
  A = 8 :=
sorry

end NUMINAMATH_GPT_tom_bought_8_kg_of_apples_l1365_136596


namespace NUMINAMATH_GPT_air_conditioner_usage_l1365_136504

-- Define the given data and the theorem to be proven
theorem air_conditioner_usage (h : ℝ) (rate : ℝ) (days : ℝ) (total_consumption : ℝ) :
  rate = 0.9 → days = 5 → total_consumption = 27 → (days * h * rate = total_consumption) → h = 6 :=
by
  intros hr dr tc h_eq
  sorry

end NUMINAMATH_GPT_air_conditioner_usage_l1365_136504


namespace NUMINAMATH_GPT_same_type_monomials_l1365_136588

theorem same_type_monomials (a b : ℤ) (h1 : 1 = a - 2) (h2 : b + 1 = 3) : (a - b) ^ 2023 = 1 := by
  sorry

end NUMINAMATH_GPT_same_type_monomials_l1365_136588


namespace NUMINAMATH_GPT_tan_75_eq_2_plus_sqrt_3_l1365_136580

theorem tan_75_eq_2_plus_sqrt_3 : Real.tan (75 * Real.pi / 180) = 2 + Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_tan_75_eq_2_plus_sqrt_3_l1365_136580


namespace NUMINAMATH_GPT_focus_of_parabola_x_squared_eq_4y_is_0_1_l1365_136521

theorem focus_of_parabola_x_squared_eq_4y_is_0_1 :
  ∃ (x y : ℝ), (0, 1) = (x, y) ∧ (∀ a b : ℝ, a^2 = 4 * b → (x, y) = (0, 1)) :=
sorry

end NUMINAMATH_GPT_focus_of_parabola_x_squared_eq_4y_is_0_1_l1365_136521


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1365_136523

variable {a : ℕ → ℝ}
variable {q : ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem necessary_but_not_sufficient_condition
    (a1_pos : a 1 > 0)
    (geo_seq : geometric_sequence a q)
    (a3_lt_a6 : a 3 < a 6) :
  (a 1 < a 3) ↔ ∃ k : ℝ, k > 1 ∧ a 1 * k^2 < a 1 * k^5 :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1365_136523


namespace NUMINAMATH_GPT_solution_set_inequality_l1365_136568

theorem solution_set_inequality (x : ℝ) : |3 * x + 1| - |x - 1| < 0 ↔ -1 < x ∧ x < 0 := 
sorry

end NUMINAMATH_GPT_solution_set_inequality_l1365_136568


namespace NUMINAMATH_GPT_sum_remainders_l1365_136519

theorem sum_remainders (n : ℤ) (h : n % 20 = 13) : (n % 4 + n % 5 = 4) :=
by
  sorry

end NUMINAMATH_GPT_sum_remainders_l1365_136519


namespace NUMINAMATH_GPT_domain_of_f_l1365_136549

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / Real.sqrt (x^2 - 4)

theorem domain_of_f :
  {x : ℝ | x^2 - 4 >= 0 ∧ x^2 - 4 ≠ 0} = {x : ℝ | x ≤ -2 ∨ x ≥ 2} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l1365_136549


namespace NUMINAMATH_GPT_construct_1_degree_l1365_136585

def canConstruct1DegreeUsing19Degree : Prop :=
  ∃ (n : ℕ), n * 19 = 360 + 1

theorem construct_1_degree (h : ∃ (x : ℕ), x * 19 = 360 + 1) : canConstruct1DegreeUsing19Degree := by
  sorry

end NUMINAMATH_GPT_construct_1_degree_l1365_136585


namespace NUMINAMATH_GPT_stationery_sales_other_l1365_136590

theorem stationery_sales_other (p e n : ℝ) (h_p : p = 25) (h_e : e = 30) (h_n : n = 20) :
    100 - (p + e + n) = 25 :=
by
  sorry

end NUMINAMATH_GPT_stationery_sales_other_l1365_136590


namespace NUMINAMATH_GPT_wrapping_paper_area_l1365_136583

theorem wrapping_paper_area 
  (l w h : ℝ) :
  (l + 4 + 2 * h) ^ 2 = l^2 + 8 * l + 16 + 4 * l * h + 16 * h + 4 * h^2 := 
by 
  sorry

end NUMINAMATH_GPT_wrapping_paper_area_l1365_136583


namespace NUMINAMATH_GPT_find_d_l1365_136587

theorem find_d (a b c d : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0)
    (h5 : a^2 = c * (d + 29)) (h6 : b^2 = c * (d - 29)) :
    d = 421 :=
    sorry

end NUMINAMATH_GPT_find_d_l1365_136587


namespace NUMINAMATH_GPT_ice_cream_cost_l1365_136562

theorem ice_cream_cost
  (num_pennies : ℕ) (num_nickels : ℕ) (num_dimes : ℕ) (num_quarters : ℕ) 
  (leftover_cents : ℤ) (num_family_members : ℕ)
  (h_pennies : num_pennies = 123)
  (h_nickels : num_nickels = 85)
  (h_dimes : num_dimes = 35)
  (h_quarters : num_quarters = 26)
  (h_leftover : leftover_cents = 48)
  (h_members : num_family_members = 5) :
  (123 * 0.01 + 85 * 0.05 + 35 * 0.1 + 26 * 0.25 - 0.48) / 5 = 3 :=
by
  sorry

end NUMINAMATH_GPT_ice_cream_cost_l1365_136562


namespace NUMINAMATH_GPT_min_employees_birthday_Wednesday_l1365_136582

theorem min_employees_birthday_Wednesday (W D : ℕ) (h_eq : W + 6 * D = 50) (h_gt : W > D) : W = 8 :=
sorry

end NUMINAMATH_GPT_min_employees_birthday_Wednesday_l1365_136582


namespace NUMINAMATH_GPT_sufficient_not_necessary_l1365_136592

variables {a : ℕ → ℝ} {S : ℕ → ℝ}
variables (h_seq : ∀ n, a (n + 1) = a n + (a 1 - a 0))
variables (h_sum : ∀ n, S n = n * (a 1 + a n) / 2)
variables (h_condition : 3 * a 2 = a 5 + 4)

theorem sufficient_not_necessary (h1 : a 1 < 1) : S 4 < 10 :=
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l1365_136592


namespace NUMINAMATH_GPT_minimum_soldiers_to_add_l1365_136560

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  ∃ k : ℕ, 84 * k + 2 - N = 82 := 
by 
  sorry

end NUMINAMATH_GPT_minimum_soldiers_to_add_l1365_136560


namespace NUMINAMATH_GPT_no_solution_l1365_136578

theorem no_solution : ∀ x y z t : ℕ, 16^x + 21^y + 26^z ≠ t^2 :=
by
  intro x y z t
  sorry

end NUMINAMATH_GPT_no_solution_l1365_136578


namespace NUMINAMATH_GPT_ceiling_and_floor_calculation_l1365_136529

theorem ceiling_and_floor_calculation : 
  let a := (15 : ℚ) / 8
  let b := (-34 : ℚ) / 4
  Int.ceil (a * b) - Int.floor (a * Int.floor b) = 2 :=
by
  sorry

end NUMINAMATH_GPT_ceiling_and_floor_calculation_l1365_136529


namespace NUMINAMATH_GPT_second_discount_percentage_l1365_136536

theorem second_discount_percentage
  (original_price : ℝ)
  (first_discount : ℝ)
  (final_price : ℝ)
  (second_discount : ℝ) :
  original_price = 10000 →
  first_discount = 0.20 →
  final_price = 6840 →
  second_discount = 14.5 :=
by
  sorry

end NUMINAMATH_GPT_second_discount_percentage_l1365_136536


namespace NUMINAMATH_GPT_loan_amount_needed_l1365_136572

-- Define the total cost of tuition.
def total_tuition : ℝ := 30000

-- Define the amount Sabina has saved.
def savings : ℝ := 10000

-- Define the grant coverage rate.
def grant_coverage_rate : ℝ := 0.4

-- Define the remainder of the tuition after using savings.
def remaining_tuition : ℝ := total_tuition - savings

-- Define the amount covered by the grant.
def grant_amount : ℝ := grant_coverage_rate * remaining_tuition

-- Define the loan amount Sabina needs to apply for.
noncomputable def loan_amount : ℝ := remaining_tuition - grant_amount

-- State the theorem to prove the loan amount needed.
theorem loan_amount_needed : loan_amount = 12000 := by
  sorry

end NUMINAMATH_GPT_loan_amount_needed_l1365_136572


namespace NUMINAMATH_GPT_find_g_product_l1365_136520

theorem find_g_product 
  (x1 x2 x3 x4 x5 : ℝ)
  (h_root1 : x1^5 - x1^3 + 1 = 0)
  (h_root2 : x2^5 - x2^3 + 1 = 0)
  (h_root3 : x3^5 - x3^3 + 1 = 0)
  (h_root4 : x4^5 - x4^3 + 1 = 0)
  (h_root5 : x5^5 - x5^3 + 1 = 0)
  (g : ℝ → ℝ) 
  (hg : ∀ x, g x = x^2 - 3) :
  g x1 * g x2 * g x3 * g x4 * g x5 = 107 := 
sorry

end NUMINAMATH_GPT_find_g_product_l1365_136520


namespace NUMINAMATH_GPT_solution_set_l1365_136576

theorem solution_set (x y : ℝ) : (x - 2 * y = 1) ∧ (x^3 - 6 * x * y - 8 * y^3 = 1) ↔ y = (x - 1) / 2 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l1365_136576


namespace NUMINAMATH_GPT_digit_x_base_7_l1365_136559

theorem digit_x_base_7 (x : ℕ) : 
    (4 * 7^3 + 5 * 7^2 + x * 7 + 2) % 9 = 0 → x = 4 := 
by {
    sorry
}

end NUMINAMATH_GPT_digit_x_base_7_l1365_136559


namespace NUMINAMATH_GPT_kyle_vs_parker_l1365_136541

-- Define the distances thrown by Parker, Grant, and Kyle.
def parker_distance : ℕ := 16
def grant_distance : ℕ := (125 * parker_distance) / 100
def kyle_distance : ℕ := 2 * grant_distance

-- Prove that Kyle threw the ball 24 yards farther than Parker.
theorem kyle_vs_parker : kyle_distance - parker_distance = 24 := 
by
  -- Sorry for proof
  sorry

end NUMINAMATH_GPT_kyle_vs_parker_l1365_136541


namespace NUMINAMATH_GPT_percentage_of_male_students_solved_l1365_136597

variable (M F : ℝ)
variable (M_25 F_25 : ℝ)
variable (prob_less_25 : ℝ)

-- Conditions from the problem
def graduation_class_conditions (M F M_25 F_25 prob_less_25 : ℝ) : Prop :=
  M + F = 100 ∧
  M_25 = 0.50 * M ∧
  F_25 = 0.30 * F ∧
  (1 - 0.50) * M + (1 - 0.30) * F = prob_less_25 * 100

-- Theorem to prove
theorem percentage_of_male_students_solved (M F : ℝ) (M_25 F_25 prob_less_25 : ℝ) :
  graduation_class_conditions M F M_25 F_25 prob_less_25 → prob_less_25 = 0.62 → M = 40 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_male_students_solved_l1365_136597


namespace NUMINAMATH_GPT_length_of_train_is_400_meters_l1365_136522

noncomputable def relative_speed (speed_train speed_man : ℝ) : ℝ :=
  speed_train - speed_man

noncomputable def km_per_hr_to_m_per_s (speed_km_per_hr : ℝ) : ℝ :=
  speed_km_per_hr * (1000 / 3600)

noncomputable def length_of_train (relative_speed_m_per_s time_seconds : ℝ) : ℝ :=
  relative_speed_m_per_s * time_seconds

theorem length_of_train_is_400_meters :
  let speed_train := 30 -- km/hr
  let speed_man := 6 -- km/hr
  let time_to_cross := 59.99520038396929 -- seconds
  let rel_speed := km_per_hr_to_m_per_s (relative_speed speed_train speed_man)
  length_of_train rel_speed time_to_cross = 400 :=
by
  sorry

end NUMINAMATH_GPT_length_of_train_is_400_meters_l1365_136522


namespace NUMINAMATH_GPT_parabola_y_intercepts_l1365_136501

theorem parabola_y_intercepts : 
  ∃ (n : ℕ), n = 2 ∧ 
  ∀ (x : ℝ), x = 0 → 
  ∃ (y : ℝ), 3 * y^2 - 5 * y - 2 = 0 :=
sorry

end NUMINAMATH_GPT_parabola_y_intercepts_l1365_136501


namespace NUMINAMATH_GPT_find_max_z_l1365_136550

theorem find_max_z :
  ∃ (x y : ℝ), abs x + abs y ≤ 4 ∧ 2 * x + y ≤ 4 ∧ (2 * x - y) = (20 / 3) :=
by
  sorry

end NUMINAMATH_GPT_find_max_z_l1365_136550


namespace NUMINAMATH_GPT_find_a2_plus_b2_l1365_136551

theorem find_a2_plus_b2 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h: 8 * a^a * b^b = 27 * a^b * b^a) : a^2 + b^2 = 117 := by
  sorry

end NUMINAMATH_GPT_find_a2_plus_b2_l1365_136551


namespace NUMINAMATH_GPT_a_plus_b_is_24_l1365_136542

theorem a_plus_b_is_24 (a b : ℤ) (h1 : 0 < b) (h2 : b < a) (h3 : a * (a + 3 * b) = 550) : a + b = 24 :=
sorry

end NUMINAMATH_GPT_a_plus_b_is_24_l1365_136542


namespace NUMINAMATH_GPT_log_27_gt_point_53_l1365_136547

open Real

theorem log_27_gt_point_53 :
  log 27 > 0.53 :=
by
  sorry

end NUMINAMATH_GPT_log_27_gt_point_53_l1365_136547


namespace NUMINAMATH_GPT_pieces_eaten_first_night_l1365_136533

-- Define the initial numbers of candies
def debby_candies : Nat := 32
def sister_candies : Nat := 42
def candies_left : Nat := 39

-- Calculate the initial total number of candies
def initial_total_candies : Nat := debby_candies + sister_candies

-- Define the number of candies eaten the first night
def candies_eaten : Nat := initial_total_candies - candies_left

-- The problem statement with the proof goal
theorem pieces_eaten_first_night : candies_eaten = 35 := by
  sorry

end NUMINAMATH_GPT_pieces_eaten_first_night_l1365_136533


namespace NUMINAMATH_GPT_line_parabola_intersect_l1365_136512

theorem line_parabola_intersect {k : ℝ} 
    (h1: ∀ x y : ℝ, y = k*x - 2 → y^2 = 8*x → x ≠ y)
    (h2: ∀ x1 x2 y1 y2 : ℝ, y1 = k*x1 - 2 → y2 = k*x2 - 2 → y1^2 = 8*x1 → y2^2 = 8*x2 → (x1 + x2) / 2 = 2) : 
    k = 2 := 
sorry

end NUMINAMATH_GPT_line_parabola_intersect_l1365_136512


namespace NUMINAMATH_GPT_num_real_roots_l1365_136565

theorem num_real_roots (f : ℝ → ℝ)
  (h_eq : ∀ x, f x = 2 * x ^ 3 - 6 * x ^ 2 + 7)
  (h_interval : ∀ x, 0 < x ∧ x < 2 → f x < 0 ∧ f (2 - x) > 0) : 
  ∃! x, 0 < x ∧ x < 2 ∧ f x = 0 :=
sorry

end NUMINAMATH_GPT_num_real_roots_l1365_136565


namespace NUMINAMATH_GPT_person_age_in_1954_l1365_136506

theorem person_age_in_1954 
  (x : ℤ)
  (cond1 : ∃ k1 : ℤ, 7 * x = 13 * k1 + 11)
  (cond2 : ∃ k2 : ℤ, 13 * x = 11 * k2 + 7)
  (input_year : ℤ) :
  input_year = 1954 → x = 1868 → input_year - x = 86 :=
by
  sorry

end NUMINAMATH_GPT_person_age_in_1954_l1365_136506


namespace NUMINAMATH_GPT_sin_angle_add_pi_over_4_l1365_136555

open Real

theorem sin_angle_add_pi_over_4 (α : ℝ) (h1 : (cos α = -3/5) ∧ (sin α = 4/5)) : sin (α + π / 4) = sqrt 2 / 10 :=
by
  sorry

end NUMINAMATH_GPT_sin_angle_add_pi_over_4_l1365_136555


namespace NUMINAMATH_GPT_A_speed_ratio_B_speed_l1365_136526

-- Define the known conditions
def B_speed : ℚ := 1 / 12
def total_speed : ℚ := 1 / 4

-- Define the problem statement
theorem A_speed_ratio_B_speed : ∃ (A_speed : ℚ), A_speed + B_speed = total_speed ∧ (A_speed / B_speed = 2) :=
by
  sorry

end NUMINAMATH_GPT_A_speed_ratio_B_speed_l1365_136526


namespace NUMINAMATH_GPT_radius_of_spheres_in_cone_l1365_136595

theorem radius_of_spheres_in_cone :
  ∀ (r : ℝ),
    let base_radius := 6
    let height := 15
    let distance_from_vertex := (2 * Real.sqrt 3 / 3) * r
    let total_height := height - r
    (total_height = distance_from_vertex) →
    r = 27 - 6 * Real.sqrt 3 :=
by
  intros r base_radius height distance_from_vertex total_height H
  sorry -- The proof of the theorem will be filled here.

end NUMINAMATH_GPT_radius_of_spheres_in_cone_l1365_136595


namespace NUMINAMATH_GPT_geometric_sequence_sum_q_value_l1365_136511

theorem geometric_sequence_sum_q_value (q : ℝ) (a S : ℕ → ℝ) :
  a 1 = 4 →
  (∀ n, a (n+1) = a n * q ) →
  (∀ n, S n = a 1 * (1 - q^n) / (1 - q)) →
  (∀ n, (S n + 2) = (S 1 + 2) * (q ^ (n - 1))) →
  q = 3
:= 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_q_value_l1365_136511


namespace NUMINAMATH_GPT_exponential_inequality_l1365_136573

theorem exponential_inequality (a x1 x2 : ℝ) (h1 : 1 < a) (h2 : x1 < x2) :
  |a ^ ((1 / 2) * (x1 + x2)) - a ^ x1| < |a ^ x2 - a ^ ((1 / 2) * (x1 + x2))| :=
by
  sorry

end NUMINAMATH_GPT_exponential_inequality_l1365_136573


namespace NUMINAMATH_GPT_max_value_fraction_sum_l1365_136594

theorem max_value_fraction_sum (a b c : ℝ) (h : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h_sum : a + b + c = 3) :
  (ab / (a + b + 1) + ac / (a + c + 1) + bc / (b + c + 1) ≤ 3 / 2) :=
sorry

end NUMINAMATH_GPT_max_value_fraction_sum_l1365_136594


namespace NUMINAMATH_GPT_part_I_part_II_l1365_136517

noncomputable def f (x : ℝ) : ℝ := abs x

theorem part_I (x : ℝ) : f (x-1) > 2 ↔ x < -1 ∨ x > 3 := 
by sorry

theorem part_II (x y z : ℝ) (h : f x ^ 2 + y ^ 2 + z ^ 2 = 9) : ∃ (min_val : ℝ), min_val = -9 ∧ ∀ (a b c : ℝ), f a ^ 2 + b ^ 2 + c ^ 2 = 9 → (a + 2 * b + 2 * c) ≥ min_val := 
by sorry

end NUMINAMATH_GPT_part_I_part_II_l1365_136517


namespace NUMINAMATH_GPT_dots_per_ladybug_l1365_136516

-- Define the conditions as variables
variables (m t : ℕ) (total_dots : ℕ) (d : ℕ)

-- Setting actual values for the variables based on the given conditions
def m_val : ℕ := 8
def t_val : ℕ := 5
def total_dots_val : ℕ := 78

-- Defining the total number of ladybugs and the average dots per ladybug
def total_ladybugs : ℕ := m_val + t_val

-- To prove: Each ladybug has 6 dots on average
theorem dots_per_ladybug : total_dots_val / total_ladybugs = 6 :=
by
  have m := m_val
  have t := t_val
  have total_dots := total_dots_val
  have d := 6
  sorry

end NUMINAMATH_GPT_dots_per_ladybug_l1365_136516


namespace NUMINAMATH_GPT_probability_same_color_l1365_136561

/-
Problem statement:
Given a bag contains 6 green balls and 7 white balls,
if two balls are drawn simultaneously, prove that the probability 
that both balls are the same color is 6/13.
-/

theorem probability_same_color
  (total_balls : ℕ := 6 + 7)
  (green_balls : ℕ := 6)
  (white_balls : ℕ := 7)
  (two_balls_drawn_simultaneously : Prop := true) :
  ((green_balls / total_balls) * ((green_balls - 1) / (total_balls - 1))) +
  ((white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1))) = 6 / 13 :=
sorry

end NUMINAMATH_GPT_probability_same_color_l1365_136561


namespace NUMINAMATH_GPT_veronica_pre_selected_photos_l1365_136569

-- Definition: Veronica needs to include 3 or 4 of her pictures
def needs_3_or_4_photos : Prop := True

-- Definition: Veronica has pre-selected a certain number of photos
def pre_selected_photos : ℕ := 15

-- Definition: She has 15 choices
def choices : ℕ := 15

-- The proof statement
theorem veronica_pre_selected_photos : needs_3_or_4_photos → choices = pre_selected_photos :=
by
  intros
  sorry

end NUMINAMATH_GPT_veronica_pre_selected_photos_l1365_136569


namespace NUMINAMATH_GPT_fractional_sum_l1365_136507

noncomputable def greatest_integer (t : ℝ) : ℝ := ⌊t⌋
noncomputable def fractional_part (t : ℝ) : ℝ := t - greatest_integer t

theorem fractional_sum (x : ℝ) (h : x^3 + (1/x)^3 = 18) : 
  fractional_part x + fractional_part (1/x) = 1 :=
sorry

end NUMINAMATH_GPT_fractional_sum_l1365_136507


namespace NUMINAMATH_GPT_daragh_initial_bears_l1365_136567

variables (initial_bears eden_initial_bears eden_final_bears favorite_bears shared_bears_per_sister : ℕ)
variables (sisters : ℕ)

-- Given conditions
axiom h1 : eden_initial_bears = 10
axiom h2 : eden_final_bears = 14
axiom h3 : favorite_bears = 8
axiom h4 : sisters = 3

-- Derived condition
axiom h5 : shared_bears_per_sister = eden_final_bears - eden_initial_bears
axiom h6 : initial_bears = favorite_bears + (shared_bears_per_sister * sisters)

-- The theorem to prove
theorem daragh_initial_bears : initial_bears = 20 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_daragh_initial_bears_l1365_136567


namespace NUMINAMATH_GPT_books_and_games_left_to_experience_l1365_136538

def booksLeft (B_total B_read : Nat) : Nat := B_total - B_read
def gamesLeft (G_total G_played : Nat) : Nat := G_total - G_played
def totalLeft (B_total B_read G_total G_played : Nat) : Nat := booksLeft B_total B_read + gamesLeft G_total G_played

theorem books_and_games_left_to_experience :
  totalLeft 150 74 50 17 = 109 := by
  sorry

end NUMINAMATH_GPT_books_and_games_left_to_experience_l1365_136538


namespace NUMINAMATH_GPT_average_remaining_ropes_l1365_136563

theorem average_remaining_ropes 
  (n : ℕ) 
  (m : ℕ) 
  (l_avg : ℕ) 
  (l1_avg : ℕ) 
  (l2_avg : ℕ) 
  (h1 : n = 6)
  (h2 : m = 2)
  (hl_avg : l_avg = 80)
  (hl1_avg : l1_avg = 70)
  (htotal : l_avg * n = 480)
  (htotal1 : l1_avg * m = 140)
  (htotal2 : l_avg * n - l1_avg * m = 340):
  (340 : ℕ) / (4 : ℕ) = 85 := by
  sorry

end NUMINAMATH_GPT_average_remaining_ropes_l1365_136563


namespace NUMINAMATH_GPT_find_additional_student_number_l1365_136544

def classSize : ℕ := 52
def sampleSize : ℕ := 4
def sampledNumbers : List ℕ := [5, 31, 44]
def additionalStudentNumber : ℕ := 18

theorem find_additional_student_number (classSize sampleSize : ℕ) 
    (sampledNumbers : List ℕ) : additionalStudentNumber ∈ (5 :: 31 :: 44 :: []) →
    (sampledNumbers = [5, 31, 44]) →
    (additionalStudentNumber = 18) := by
  sorry

end NUMINAMATH_GPT_find_additional_student_number_l1365_136544


namespace NUMINAMATH_GPT_quadrilateral_inequality_l1365_136593

-- Definitions based on conditions in a)
variables {A B C D : Type}
variables (AB AC AD BC CD : ℝ)
variable (angleA angleC: ℝ)
variable (convex := angleA + angleC < 180)

-- Lean statement that encodes the problem
theorem quadrilateral_inequality 
  (Hconvex : convex = true)
  : AB * CD + AD * BC < AC * (AB + AD) := 
sorry

end NUMINAMATH_GPT_quadrilateral_inequality_l1365_136593


namespace NUMINAMATH_GPT_felix_chopped_down_trees_l1365_136543

theorem felix_chopped_down_trees
  (sharpening_cost : ℕ)
  (trees_per_sharpening : ℕ)
  (total_spent : ℕ)
  (times_sharpened : ℕ)
  (trees_chopped_down : ℕ)
  (h1 : sharpening_cost = 5)
  (h2 : trees_per_sharpening = 13)
  (h3 : total_spent = 35)
  (h4 : times_sharpened = total_spent / sharpening_cost)
  (h5 : trees_chopped_down = trees_per_sharpening * times_sharpened) :
  trees_chopped_down ≥ 91 :=
by
  sorry

end NUMINAMATH_GPT_felix_chopped_down_trees_l1365_136543


namespace NUMINAMATH_GPT_gem_stone_necklaces_sold_l1365_136589

theorem gem_stone_necklaces_sold (total_earned total_cost number_bead number_gem total_necklaces : ℕ) 
    (h1 : total_earned = 36) 
    (h2 : total_cost = 6) 
    (h3 : number_bead = 3) 
    (h4 : total_necklaces = total_earned / total_cost) 
    (h5 : total_necklaces = number_bead + number_gem) : 
    number_gem = 3 := 
sorry

end NUMINAMATH_GPT_gem_stone_necklaces_sold_l1365_136589


namespace NUMINAMATH_GPT_no_two_digit_number_divisible_l1365_136515

theorem no_two_digit_number_divisible (a b : ℕ) (distinct : a ≠ b)
  (h₁ : 1 ≤ a ∧ a ≤ 9) (h₂ : 1 ≤ b ∧ b ≤ 9)
  : ¬ ∃ k : ℕ, (1 < k ∧ k ≤ 9) ∧ (10 * a + b = k * (10 * b + a)) :=
by
  sorry

end NUMINAMATH_GPT_no_two_digit_number_divisible_l1365_136515


namespace NUMINAMATH_GPT_g_18_equals_5832_l1365_136510

noncomputable def g (n : ℕ) : ℕ := sorry

axiom cond1 : ∀ (n : ℕ), (0 < n) → g (n + 1) > g n
axiom cond2 : ∀ (m n : ℕ), (0 < m ∧ 0 < n) → g (m * n) = g m * g n
axiom cond3 : ∀ (m n : ℕ), (0 < m ∧ 0 < n ∧ m ≠ n ∧ m^2 = n^3) → (g m = n ∨ g n = m)

theorem g_18_equals_5832 : g 18 = 5832 :=
by sorry

end NUMINAMATH_GPT_g_18_equals_5832_l1365_136510


namespace NUMINAMATH_GPT_problem_solution_l1365_136500

theorem problem_solution : (121^2 - 110^2) / 11 = 231 := 
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1365_136500


namespace NUMINAMATH_GPT_solve_system_of_equations_l1365_136537

theorem solve_system_of_equations (x y : Real) : 
  (3 * x^2 + 3 * y^2 - 2 * x^2 * y^2 = 3) ∧ 
  (x^4 + y^4 + (2/3) * x^2 * y^2 = 17) ↔
  ( (x = Real.sqrt 2 ∧ (y = Real.sqrt 3 ∨ y = -Real.sqrt 3 )) ∨ 
    (x = -Real.sqrt 2 ∧ (y = Real.sqrt 3 ∨ y = -Real.sqrt 3)) ∨ 
    (x = Real.sqrt 3 ∧ (y = Real.sqrt 2 ∨ y = -Real.sqrt 2 )) ∨ 
    (x = -Real.sqrt 3 ∧ (y = Real.sqrt 2 ∨ y = -Real.sqrt 2 )) ) := 
  by
    sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1365_136537


namespace NUMINAMATH_GPT_quadrilateral_with_equal_angles_is_parallelogram_l1365_136532

axiom Quadrilateral (a b c d : Type) : Prop
axiom Parallelogram (a b c d : Type) : Prop
axiom equal_angles (a b c d : Type) : Prop

theorem quadrilateral_with_equal_angles_is_parallelogram 
  (a b c d : Type) 
  (q : Quadrilateral a b c d)
  (h : equal_angles a b c d) : Parallelogram a b c d := 
sorry

end NUMINAMATH_GPT_quadrilateral_with_equal_angles_is_parallelogram_l1365_136532


namespace NUMINAMATH_GPT_inscribed_circle_radius_l1365_136508

theorem inscribed_circle_radius (r : ℝ) (R : ℝ) (angle : ℝ):
  R = 6 → angle = 2 * Real.pi / 3 → r = (6 * Real.sqrt 3) / 5 :=
by
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_l1365_136508


namespace NUMINAMATH_GPT_laura_change_l1365_136546

-- Define the cost of a pair of pants and a shirt.
def cost_of_pants := 54
def cost_of_shirts := 33

-- Define the number of pants and shirts Laura bought.
def num_pants := 2
def num_shirts := 4

-- Define the amount Laura gave to the cashier.
def amount_given := 250

-- Calculate the total cost.
def total_cost := num_pants * cost_of_pants + num_shirts * cost_of_shirts

-- Define the expected change.
def expected_change := 10

-- The main theorem stating the problem and its solution.
theorem laura_change :
  amount_given - total_cost = expected_change :=
by
  sorry

end NUMINAMATH_GPT_laura_change_l1365_136546


namespace NUMINAMATH_GPT_number_of_pairs_satisfying_l1365_136505

theorem number_of_pairs_satisfying (h1 : 2 ^ 2013 < 5 ^ 867) (h2 : 5 ^ 867 < 2 ^ 2014) :
  ∃ k, k = 279 ∧ ∀ (m n : ℕ), 1 ≤ m ∧ m ≤ 2012 ∧ 5 ^ n < 2 ^ m ∧ 2 ^ (m + 2) < 5 ^ (n + 1) → 
  ∃ (count : ℕ), count = 279 :=
by
  sorry

end NUMINAMATH_GPT_number_of_pairs_satisfying_l1365_136505


namespace NUMINAMATH_GPT_area_enclosed_by_curve_l1365_136575

theorem area_enclosed_by_curve :
  ∃ (area : ℝ), (∀ (x y : ℝ), |x - 1| + |y - 1| = 1 → area = 2) :=
sorry

end NUMINAMATH_GPT_area_enclosed_by_curve_l1365_136575


namespace NUMINAMATH_GPT_length_of_cable_l1365_136503

-- Conditions
def condition1 (x y z : ℝ) : Prop := x + y + z = 8
def condition2 (x y z : ℝ) : Prop := x * y + y * z + x * z = -18

-- Conclusion we want to prove
theorem length_of_cable (x y z : ℝ) (h1 : condition1 x y z) (h2 : condition2 x y z) :
  4 * π * Real.sqrt (59 / 3) = 4 * π * (Real.sqrt ((x^2 + y^2 + z^2 - ((x + y + z)^2 - 4*(x*y + y*z + x*z))) / 3)) :=
sorry

end NUMINAMATH_GPT_length_of_cable_l1365_136503


namespace NUMINAMATH_GPT_dress_assignment_l1365_136577

variables {Girl : Type} [Finite Girl]
variables (Katya Olya Liza Rita Pink Green Yellow Blue : Girl)
variables (standing_between : Girl → Girl → Girl → Prop)

-- Conditions
variable (cond1 : Katya ≠ Pink ∧ Katya ≠ Blue)
variable (cond2 : standing_between Green Liza Yellow)
variable (cond3 : Rita ≠ Green ∧ Rita ≠ Blue)
variable (cond4 : standing_between Olya Rita Pink)

-- Theorem statement
theorem dress_assignment :
  Katya = Green ∧ Olya = Blue ∧ Liza = Pink ∧ Rita = Yellow := 
sorry

end NUMINAMATH_GPT_dress_assignment_l1365_136577


namespace NUMINAMATH_GPT_angle_measure_x_l1365_136513

theorem angle_measure_x
    (angle_CBE : ℝ)
    (angle_EBD : ℝ)
    (angle_ABE : ℝ)
    (sum_angles_TRIA : ∀ a b c : ℝ, a + b + c = 180)
    (sum_straight_ANGLE : ∀ a b : ℝ, a + b = 180) :
    angle_CBE = 124 → angle_EBD = 33 → angle_ABE = 19 → x = 91 :=
by
    sorry

end NUMINAMATH_GPT_angle_measure_x_l1365_136513


namespace NUMINAMATH_GPT_sum_of_cubes_l1365_136548

theorem sum_of_cubes (x y : ℝ) (h_sum : x + y = 3) (h_prod : x * y = 2) : x^3 + y^3 = 9 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l1365_136548


namespace NUMINAMATH_GPT_sum_mean_median_mode_l1365_136528

def numbers : List ℕ := [3, 5, 3, 0, 2, 5, 0, 2]

def mode (l : List ℕ) : ℝ := 4

def median (l : List ℕ) : ℝ := 2.5

def mean (l : List ℕ) : ℝ := 2.5

theorem sum_mean_median_mode : mean numbers + median numbers + mode numbers = 9 := by
  sorry

end NUMINAMATH_GPT_sum_mean_median_mode_l1365_136528


namespace NUMINAMATH_GPT_max_candies_per_student_l1365_136553

theorem max_candies_per_student (n_students : ℕ) (mean_candies : ℕ) (min_candies : ℕ) (max_candies : ℕ) :
  n_students = 50 ∧
  mean_candies = 7 ∧
  min_candies = 1 ∧
  max_candies = 20 →
  ∃ m : ℕ, m ≤ max_candies :=
by
  intro h
  use 20
  sorry

end NUMINAMATH_GPT_max_candies_per_student_l1365_136553


namespace NUMINAMATH_GPT_dot_product_PA_PB_l1365_136514

theorem dot_product_PA_PB (x_0 : ℝ) (h : x_0 > 0):
  let P := (x_0, x_0 + 2/x_0)
  let A := ((x_0 + 2/x_0) / 2, (x_0 + 2/x_0) / 2)
  let B := (0, x_0 + 2/x_0)
  let vector_PA := ((x_0 + 2/x_0) / 2 - x_0, (x_0 + 2/x_0) / 2 - (x_0 + 2/x_0))
  let vector_PB := (0 - x_0, (x_0 + 2/x_0) - (x_0 + 2/x_0))
  vector_PA.1 * vector_PB.1 + vector_PA.2 * vector_PB.2 = -1 := by
  sorry

end NUMINAMATH_GPT_dot_product_PA_PB_l1365_136514


namespace NUMINAMATH_GPT_height_of_parallelogram_l1365_136570

-- Define the problem statement
theorem height_of_parallelogram (A : ℝ) (b : ℝ) (h : ℝ) (h_eq : A = b * h) (A_val : A = 384) (b_val : b = 24) : h = 16 :=
by
  -- Skeleton proof, include the initial conditions and proof statement
  sorry

end NUMINAMATH_GPT_height_of_parallelogram_l1365_136570


namespace NUMINAMATH_GPT_louis_never_reaches_target_l1365_136566

def stable (p : ℤ × ℤ) : Prop :=
  (p.1 + p.2) % 7 ≠ 0

def move1 (p : ℤ × ℤ) : ℤ × ℤ :=
  (p.2, p.1)

def move2 (p : ℤ × ℤ) : ℤ × ℤ :=
  (3 * p.1, -4 * p.2)

def move3 (p : ℤ × ℤ) : ℤ × ℤ :=
  (-2 * p.1, 5 * p.2)

def move4 (p : ℤ × ℤ) : ℤ × ℤ :=
  (p.1 + 1, p.2 + 6)

def move5 (p : ℤ × ℤ) : ℤ × ℤ :=
  (p.1 - 7, p.2)

-- Define the start and target points
def start : ℤ × ℤ := (0, 1)
def target : ℤ × ℤ := (0, 0)

theorem louis_never_reaches_target :
  ∀ p, (p = start → ¬ ∃ k, move1^[k] p = target) ∧
       (p = start → ¬ ∃ k, move2^[k] p = target) ∧
       (p = start → ¬ ∃ k, move3^[k] p = target) ∧
       (p = start → ¬ ∃ k, move4^[k] p = target) ∧
       (p = start → ¬ ∃ k, move5^[k] p = target) :=
by {
  sorry
}

end NUMINAMATH_GPT_louis_never_reaches_target_l1365_136566


namespace NUMINAMATH_GPT_sqrt_difference_l1365_136554

theorem sqrt_difference :
  (Real.sqrt 63 - 7 * Real.sqrt (1 / 7)) = 2 * Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_difference_l1365_136554


namespace NUMINAMATH_GPT_symm_diff_complement_l1365_136530

variable {U : Type} -- Universal set U
variable (A B : Set U) -- Sets A and B

-- Definition of symmetric difference
def symm_diff (X Y : Set U) : Set U := (X ∪ Y) \ (X ∩ Y)

theorem symm_diff_complement (A B : Set U) :
  (symm_diff A B) = (symm_diff (Aᶜ) (Bᶜ)) :=
sorry

end NUMINAMATH_GPT_symm_diff_complement_l1365_136530


namespace NUMINAMATH_GPT_triangle_inequality_l1365_136571

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (triangle_cond : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2*(b + c - a) + b^2*(c + a - b) + c^2*(a + b - c) ≤ 3*a*b*c :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l1365_136571


namespace NUMINAMATH_GPT_man_work_alone_in_5_days_l1365_136552

theorem man_work_alone_in_5_days (d : ℕ) (h1 : ∀ m : ℕ, (1 / (m : ℝ)) + 1 / 20 = 1 / 4):
  d = 5 := by
  sorry

end NUMINAMATH_GPT_man_work_alone_in_5_days_l1365_136552


namespace NUMINAMATH_GPT_find_second_number_l1365_136564

theorem find_second_number 
  (h1 : (20 + 40 + 60) / 3 = (10 + x + 45) / 3 + 5) :
  x = 50 :=
sorry

end NUMINAMATH_GPT_find_second_number_l1365_136564


namespace NUMINAMATH_GPT_vector_addition_in_triangle_l1365_136525

theorem vector_addition_in_triangle
  (A B C D : Type)
  [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] 
  [Module ℝ A] [Module ℝ B] [Module ℝ C] [Module ℝ D]
  (AB AC AD BD DC : A)
  (h1 : BD = 2 • DC) :
  AD = (1/3 : ℝ) • AB + (2/3 : ℝ) • AC :=
sorry

end NUMINAMATH_GPT_vector_addition_in_triangle_l1365_136525


namespace NUMINAMATH_GPT_minimum_inhabitants_to_ask_l1365_136535

def knights_count : ℕ := 50
def civilians_count : ℕ := 15

theorem minimum_inhabitants_to_ask (knights civilians : ℕ) (h_knights : knights = knights_count) (h_civilians : civilians = civilians_count) :
  ∃ n, (∀ asked : ℕ, (asked ≥ n) → asked - civilians > civilians) ∧ n = 31 :=
by
  sorry

end NUMINAMATH_GPT_minimum_inhabitants_to_ask_l1365_136535
