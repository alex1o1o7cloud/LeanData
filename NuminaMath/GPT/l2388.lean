import Mathlib

namespace NUMINAMATH_GPT_probability_is_zero_l2388_238811

noncomputable def probability_same_number (b d : ℕ) (h_b : b < 150) (h_d : d < 150)
    (h_b_multiple: b % 15 = 0) (h_d_multiple: d % 20 = 0) (h_square: b * b = b ∨ d * d = d) : ℝ :=
  0

theorem probability_is_zero (b d : ℕ) (h_b : b < 150) (h_d : d < 150)
    (h_b_multiple: b % 15 = 0) (h_d_multiple: d % 20 = 0) (h_square: b * b = b ∨ d * d = d) : 
    probability_same_number b d h_b h_d h_b_multiple h_d_multiple h_square = 0 :=
  sorry

end NUMINAMATH_GPT_probability_is_zero_l2388_238811


namespace NUMINAMATH_GPT_fraction_of_ponies_with_horseshoes_l2388_238889

theorem fraction_of_ponies_with_horseshoes 
  (P H : ℕ) 
  (h1 : H = P + 4) 
  (h2 : H + P ≥ 164) 
  (x : ℚ)
  (h3 : ∃ (n : ℕ), n = (5 / 8) * (x * P)) :
  x = 1 / 10 := by
  sorry

end NUMINAMATH_GPT_fraction_of_ponies_with_horseshoes_l2388_238889


namespace NUMINAMATH_GPT_sqrt_frac_meaningful_l2388_238800

theorem sqrt_frac_meaningful (x : ℝ) (h : 1 / (x - 1) > 0) : x > 1 :=
sorry

end NUMINAMATH_GPT_sqrt_frac_meaningful_l2388_238800


namespace NUMINAMATH_GPT_solve_system_l2388_238821

theorem solve_system (x y : ℝ) (h1 : 4 * x - y = 2) (h2 : 3 * x - 2 * y = -1) : x - y = -1 := 
by
  sorry

end NUMINAMATH_GPT_solve_system_l2388_238821


namespace NUMINAMATH_GPT_rainfall_wednesday_correct_l2388_238829

def monday_rainfall : ℝ := 0.9
def tuesday_rainfall : ℝ := monday_rainfall - 0.7
def wednesday_rainfall : ℝ := 2 * (monday_rainfall + tuesday_rainfall)

theorem rainfall_wednesday_correct : wednesday_rainfall = 2.2 := by
sorry

end NUMINAMATH_GPT_rainfall_wednesday_correct_l2388_238829


namespace NUMINAMATH_GPT_negation_of_existence_l2388_238858

theorem negation_of_existence (p : Prop) (h : ∃ (c : ℝ), c > 0 ∧ (∃ (x : ℝ), x^2 - x + c = 0)) : 
  ¬ (∃ (c : ℝ), c > 0 ∧ (∃ (x : ℝ), x^2 - x + c = 0)) ↔ 
  ∀ (c : ℝ), c > 0 → ¬ (∃ (x : ℝ), x^2 - x + c = 0) :=
by 
  sorry

end NUMINAMATH_GPT_negation_of_existence_l2388_238858


namespace NUMINAMATH_GPT_simplify_fraction_l2388_238827

theorem simplify_fraction : (1 / (2 + (2/3))) = (3 / 8) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2388_238827


namespace NUMINAMATH_GPT_sin_180_degrees_l2388_238817

theorem sin_180_degrees : Real.sin (180 * Real.pi / 180) = 0 := 
by
  sorry

end NUMINAMATH_GPT_sin_180_degrees_l2388_238817


namespace NUMINAMATH_GPT_symmetric_point_yoz_l2388_238869

theorem symmetric_point_yoz (x y z : ℝ) (hx : x = 2) (hy : y = 3) (hz : z = 4) :
  (-x, y, z) = (-2, 3, 4) :=
by
  -- The proof is skipped
  sorry

end NUMINAMATH_GPT_symmetric_point_yoz_l2388_238869


namespace NUMINAMATH_GPT_bicycle_cost_price_l2388_238813

-- Definitions of conditions
def profit_22_5_percent (x : ℝ) : ℝ := 1.225 * x
def loss_14_3_percent (x : ℝ) : ℝ := 0.857 * x
def profit_32_4_percent (x : ℝ) : ℝ := 1.324 * x
def loss_7_8_percent (x : ℝ) : ℝ := 0.922 * x
def discount_5_percent (x : ℝ) : ℝ := 0.95 * x
def tax_6_percent (x : ℝ) : ℝ := 1.06 * x

theorem bicycle_cost_price (CP_A : ℝ) (TP_E : ℝ) (h : TP_E = 295.88) : 
  CP_A = 295.88 / 1.29058890594 :=
by
  sorry

end NUMINAMATH_GPT_bicycle_cost_price_l2388_238813


namespace NUMINAMATH_GPT_yang_hui_problem_l2388_238852

theorem yang_hui_problem (x : ℝ) :
  x * (x + 12) = 864 :=
sorry

end NUMINAMATH_GPT_yang_hui_problem_l2388_238852


namespace NUMINAMATH_GPT_max_pages_l2388_238837

/-- Prove that the maximum number of pages the book has is 208 -/
theorem max_pages (pages: ℕ) (h1: pages ≥ 16 * 12 + 1) (h2: pages ≤ 13 * 16) 
(h3: pages ≥ 20 * 10 + 1) (h4: pages ≤ 11 * 20) : 
  pages ≤ 208 :=
by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_max_pages_l2388_238837


namespace NUMINAMATH_GPT_find_f_2017_l2388_238853

noncomputable def f (x : ℤ) (a α b β : ℝ) : ℝ :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β)

theorem find_f_2017
(x : ℤ)
(a α b β : ℝ)
(h : f 4 a α b β = 3) :
f 2017 a α b β = -3 := 
sorry

end NUMINAMATH_GPT_find_f_2017_l2388_238853


namespace NUMINAMATH_GPT_hypotenuse_length_l2388_238879

-- Let a and b be the lengths of the non-hypotenuse sides of a right triangle.
-- We are given that a = 6 and b = 8, and we need to prove that the hypotenuse c is 10.
theorem hypotenuse_length (a b c : ℕ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c ^ 2 = a ^ 2 + b ^ 2) : c = 10 :=
by
  -- The proof goes here.
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l2388_238879


namespace NUMINAMATH_GPT_plane_through_A_perpendicular_to_BC_l2388_238815

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def vector_between_points (P Q : Point3D) : Point3D :=
  { x := Q.x - P.x, y := Q.y - P.y, z := Q.z - P.z }

def plane_eq (n : Point3D) (P : Point3D) (x y z : ℝ) : ℝ :=
  n.x * (x - P.x) + n.y * (y - P.y) + n.z * (z - P.z)

def A := Point3D.mk 0 (-2) 8
def B := Point3D.mk 4 3 2
def C := Point3D.mk 1 4 3

def n := vector_between_points B C
def plane := plane_eq n A

theorem plane_through_A_perpendicular_to_BC :
  ∀ x y z : ℝ, plane x y z = 0 ↔ -3 * x + y + z - 6 = 0 :=
by
  sorry

end NUMINAMATH_GPT_plane_through_A_perpendicular_to_BC_l2388_238815


namespace NUMINAMATH_GPT_probability_longer_piece_at_least_x_squared_l2388_238847

noncomputable def probability_longer_piece (x : ℝ) : ℝ :=
  if x = 0 then 1 else (2 / (x^2 + 1))

theorem probability_longer_piece_at_least_x_squared (x : ℝ) :
  probability_longer_piece x = (2 / (x^2 + 1)) :=
sorry

end NUMINAMATH_GPT_probability_longer_piece_at_least_x_squared_l2388_238847


namespace NUMINAMATH_GPT_jake_buys_packages_l2388_238810

theorem jake_buys_packages:
  ∀ (pkg_weight cost_per_pound total_paid : ℕ),
    pkg_weight = 2 →
    cost_per_pound = 4 →
    total_paid = 24 →
    (total_paid / (pkg_weight * cost_per_pound)) = 3 :=
by
  intros pkg_weight cost_per_pound total_paid hw_cp ht
  sorry

end NUMINAMATH_GPT_jake_buys_packages_l2388_238810


namespace NUMINAMATH_GPT_complement_intersection_l2388_238825

open Set

theorem complement_intersection
  (U : Set ℝ) (A B : Set ℝ) 
  (hU : U = univ) 
  (hA : A = { x : ℝ | x ≤ -2 }) 
  (hB : B = { x : ℝ | x < 1 }) :
  (U \ A) ∩ B = { x : ℝ | -2 < x ∧ x < 1 } :=
by
  rw [hU, hA, hB]
  sorry

end NUMINAMATH_GPT_complement_intersection_l2388_238825


namespace NUMINAMATH_GPT_total_interest_l2388_238883

variable (P R : ℝ)

-- Given condition: Simple interest on sum of money is Rs. 700 after 10 years
def interest_10_years (P R : ℝ) : Prop := (P * R * 10) / 100 = 700

-- Principal is trebled after 5 years
def interest_5_years_treble (P R : ℝ) : Prop := (15 * P * R) / 100 = 105

-- The final interest is the sum of interest for the first 10 years and next 5 years post trebling the principal
theorem total_interest (P R : ℝ) (h1: interest_10_years P R) (h2: interest_5_years_treble P R) : 
  (700 + 105 = 805) := 
  by 
  sorry

end NUMINAMATH_GPT_total_interest_l2388_238883


namespace NUMINAMATH_GPT_second_largest_of_five_consecutive_is_19_l2388_238830

theorem second_largest_of_five_consecutive_is_19 (n : ℕ) 
  (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 90): 
  n + 3 = 19 :=
by sorry

end NUMINAMATH_GPT_second_largest_of_five_consecutive_is_19_l2388_238830


namespace NUMINAMATH_GPT_paintable_sum_l2388_238808

theorem paintable_sum :
  ∃ (h t u v : ℕ), h > 0 ∧ t > 0 ∧ u > 0 ∧ v > 0 ∧
  (∀ k, k % h = 1 ∨ k % t = 2 ∨ k % u = 3 ∨ k % v = 4) ∧
  (∀ k k', k ≠ k' → (k % h ≠ k' % h ∧ k % t ≠ k' % t ∧ k % u ≠ k' % u ∧ k % v ≠ k' % v)) ∧
  1000 * h + 100 * t + 10 * u + v = 4536 :=
by
  sorry

end NUMINAMATH_GPT_paintable_sum_l2388_238808


namespace NUMINAMATH_GPT_back_seat_tickets_sold_l2388_238840

variable (M B : ℕ)

theorem back_seat_tickets_sold:
  M + B = 20000 ∧ 55 * M + 45 * B = 955000 → B = 14500 :=
by
  sorry

end NUMINAMATH_GPT_back_seat_tickets_sold_l2388_238840


namespace NUMINAMATH_GPT_find_p_l2388_238876

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 5
def parabola_eq (p x y : ℝ) : Prop := y^2 = 2 * p * x
def quadrilateral_is_rectangle (A B C D : ℝ × ℝ) : Prop := 
  A.1 = C.1 ∧ B.1 = D.1 ∧ A.2 = D.2 ∧ B.2 = C.2

theorem find_p (A B C D : ℝ × ℝ) (p : ℝ) (h1 : ∃ x y, circle_eq x y ∧ parabola_eq p x y) 
  (h2 : ∃ x y, circle_eq x y ∧ x = 0) 
  (h3 : quadrilateral_is_rectangle A B C D) 
  (h4 : 0 < p) : 
  p = 2 := 
sorry

end NUMINAMATH_GPT_find_p_l2388_238876


namespace NUMINAMATH_GPT_max_grain_mass_l2388_238890

def platform_length : ℝ := 10
def platform_width : ℝ := 5
def grain_density : ℝ := 1200
def angle_of_repose : ℝ := 45
def max_mass : ℝ := 175000

theorem max_grain_mass :
  let height_of_pile := platform_width / 2
  let volume_of_prism := platform_length * platform_width * height_of_pile
  let volume_of_pyramid := (1 / 3) * (platform_width * height_of_pile) * height_of_pile
  let total_volume := volume_of_prism + 2 * volume_of_pyramid
  let calculated_mass := total_volume * grain_density
  calculated_mass = max_mass :=
by {
  sorry
}

end NUMINAMATH_GPT_max_grain_mass_l2388_238890


namespace NUMINAMATH_GPT_masha_nonnegative_l2388_238807

theorem masha_nonnegative (a b c d : ℝ) (h1 : a + b = c * d) (h2 : a * b = c + d) : 
  (a + 1) * (b + 1) * (c + 1) * (d + 1) ≥ 0 := 
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_masha_nonnegative_l2388_238807


namespace NUMINAMATH_GPT_total_dog_legs_l2388_238814

theorem total_dog_legs (total_animals cats dogs: ℕ) (h1: total_animals = 300) 
  (h2: cats = 2 / 3 * total_animals) 
  (h3: dogs = 1 / 3 * total_animals): (dogs * 4) = 400 :=
by
  sorry

end NUMINAMATH_GPT_total_dog_legs_l2388_238814


namespace NUMINAMATH_GPT_task_force_combinations_l2388_238824

theorem task_force_combinations :
  (Nat.choose 10 4) * (Nat.choose 7 3) = 7350 :=
by
  sorry

end NUMINAMATH_GPT_task_force_combinations_l2388_238824


namespace NUMINAMATH_GPT_equation_one_solution_equation_two_solution_l2388_238866

theorem equation_one_solution (x : ℝ) : ((x + 3) ^ 2 - 9 = 0) ↔ (x = 0 ∨ x = -6) := by
  sorry

theorem equation_two_solution (x : ℝ) : (x ^ 2 - 4 * x + 1 = 0) ↔ (x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) := by
  sorry

end NUMINAMATH_GPT_equation_one_solution_equation_two_solution_l2388_238866


namespace NUMINAMATH_GPT_lisa_children_l2388_238872

theorem lisa_children (C : ℕ) 
  (h1 : 5 * 52 = 260)
  (h2 : (2 * C + 3 + 2) * 260 = 3380) : 
  C = 4 := 
by
  sorry

end NUMINAMATH_GPT_lisa_children_l2388_238872


namespace NUMINAMATH_GPT_expression_evaluation_l2388_238818

-- Definitions of the expressions
def expr (x y : ℤ) : ℤ :=
  ((x - 2 * y) ^ 2 + (3 * x - y) * (3 * x + y) - 3 * y ^ 2) / (-2 * x)

-- Proof that the expression evaluates to -11 when x = 1 and y = -3
theorem expression_evaluation : expr 1 (-3) = -11 :=
by
  -- Declarations
  let x := 1
  let y := -3
  -- The core calculation
  show expr x y = -11
  sorry

end NUMINAMATH_GPT_expression_evaluation_l2388_238818


namespace NUMINAMATH_GPT_problem1_problem2_l2388_238850

open Real

/-- Problem 1: Simplify trigonometric expression. -/
theorem problem1 : 
  (sqrt (1 - 2 * sin (10 * pi / 180) * cos (10 * pi / 180)) /
  (sin (170 * pi / 180) - sqrt (1 - sin (170 * pi / 180)^2))) = -1 :=
sorry

/-- Problem 2: Given tan(θ) = 2, find the value.
  Required to prove: 2 + sin(θ) * cos(θ) - cos(θ)^2 equals 11/5 -/
theorem problem2 (θ : ℝ) (h : tan θ = 2) :
  2 + sin θ * cos θ - cos θ^2 = 11 / 5 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l2388_238850


namespace NUMINAMATH_GPT_total_weight_on_scale_l2388_238846

def weight_blue_ball : ℝ := 6
def weight_brown_ball : ℝ := 3.12

theorem total_weight_on_scale :
  weight_blue_ball + weight_brown_ball = 9.12 :=
by sorry

end NUMINAMATH_GPT_total_weight_on_scale_l2388_238846


namespace NUMINAMATH_GPT_find_b_of_sin_l2388_238894

theorem find_b_of_sin (a b c d : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
                       (h_period : (2 * Real.pi) / b = Real.pi / 2) : b = 4 := by
  sorry

end NUMINAMATH_GPT_find_b_of_sin_l2388_238894


namespace NUMINAMATH_GPT_smallest_not_prime_nor_square_no_prime_factor_lt_60_correct_l2388_238893

noncomputable def smallest_not_prime_nor_square_no_prime_factor_lt_60 : ℕ :=
  4087

theorem smallest_not_prime_nor_square_no_prime_factor_lt_60_correct :
  ∀ n : ℕ, 
    (n > 0) → 
    (¬ Prime n) →
    (¬ ∃ k : ℕ, k * k = n) →
    (∀ p : ℕ, Prime p → p ∣ n → p ≥ 60) →
    n ≥ 4087 :=
sorry

end NUMINAMATH_GPT_smallest_not_prime_nor_square_no_prime_factor_lt_60_correct_l2388_238893


namespace NUMINAMATH_GPT_sequence_formula_l2388_238845

def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 3
  | 2 => 6
  | 3 => 10
  | _ => sorry  -- The pattern is more general

theorem sequence_formula (n : ℕ) : a n = (n * (n + 1)) / 2 := 
  sorry

end NUMINAMATH_GPT_sequence_formula_l2388_238845


namespace NUMINAMATH_GPT_prove_interest_rates_equal_l2388_238899

noncomputable def interest_rates_equal : Prop :=
  let initial_savings := 1000
  let savings_simple := initial_savings / 2
  let savings_compound := initial_savings / 2
  let simple_interest_earned := 100
  let compound_interest_earned := 105
  let time := 2
  let r_s := simple_interest_earned / (savings_simple * time)
  let r_c := (compound_interest_earned / savings_compound + 1) ^ (1 / time) - 1
  r_s = r_c

theorem prove_interest_rates_equal : interest_rates_equal :=
  sorry

end NUMINAMATH_GPT_prove_interest_rates_equal_l2388_238899


namespace NUMINAMATH_GPT_x_minus_q_eq_three_l2388_238895

theorem x_minus_q_eq_three (x q : ℝ) (h1 : |x - 3| = q) (h2 : x > 3) : x - q = 3 :=
by 
  sorry

end NUMINAMATH_GPT_x_minus_q_eq_three_l2388_238895


namespace NUMINAMATH_GPT_sheena_completes_in_37_weeks_l2388_238826

-- Definitions based on the conditions
def hours_per_dress : List Nat := [15, 18, 20, 22, 24, 26, 28]
def hours_cycle : List Nat := [5, 3, 6, 4]
def finalize_hours : Nat := 10

-- The total hours needed to sew all dresses
def total_dress_hours : Nat := hours_per_dress.sum

-- The total hours needed including finalizing hours
def total_hours : Nat := total_dress_hours + finalize_hours

-- Total hours sewed in each 4-week cycle
def hours_per_cycle : Nat := hours_cycle.sum

-- Total number of weeks it will take to complete all dresses
def weeks_needed : Nat := 4 * ((total_hours + hours_per_cycle - 1) / hours_per_cycle)
def additional_weeks : Nat := if total_hours % hours_per_cycle == 0 then 0 else 1

theorem sheena_completes_in_37_weeks : weeks_needed + additional_weeks = 37 := by
  sorry

end NUMINAMATH_GPT_sheena_completes_in_37_weeks_l2388_238826


namespace NUMINAMATH_GPT_undefined_expression_iff_l2388_238856

theorem undefined_expression_iff (x : ℝ) :
  (x^2 - 24 * x + 144 = 0) ↔ (x = 12) := 
sorry

end NUMINAMATH_GPT_undefined_expression_iff_l2388_238856


namespace NUMINAMATH_GPT_ratio_of_women_to_men_l2388_238832

theorem ratio_of_women_to_men (M W : ℕ) 
  (h1 : M + W = 72) 
  (h2 : M - 16 = W + 8) : 
  W / M = 1 / 2 :=
sorry

end NUMINAMATH_GPT_ratio_of_women_to_men_l2388_238832


namespace NUMINAMATH_GPT_john_total_expense_l2388_238874

-- Define variables
variables (M D : ℝ)

-- Define the conditions
axiom cond1 : M = 20 * D
axiom cond2 : M = 24 * (D - 3)

-- State the theorem to prove
theorem john_total_expense : M = 360 :=
by
  -- Add the proof steps here
  sorry

end NUMINAMATH_GPT_john_total_expense_l2388_238874


namespace NUMINAMATH_GPT_triangle_angle_proof_l2388_238802

theorem triangle_angle_proof (α β γ : ℝ) (hα : α > 60) (hβ : β > 60) (hγ : γ > 60) (h_sum : α + β + γ = 180) : false :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_proof_l2388_238802


namespace NUMINAMATH_GPT_evaluate_expression_l2388_238878

theorem evaluate_expression : 
  3 * (-4) - ((5 * (-5)) * (-2)) + 6 = -56 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2388_238878


namespace NUMINAMATH_GPT_albert_mary_age_ratio_l2388_238860

theorem albert_mary_age_ratio
  (A M B : ℕ)
  (h1 : A = 4 * B)
  (h2 : M = A - 14)
  (h3 : B = 7)
  :
  A / M = 2 := 
by sorry

end NUMINAMATH_GPT_albert_mary_age_ratio_l2388_238860


namespace NUMINAMATH_GPT_line_of_intersecting_circles_l2388_238880

theorem line_of_intersecting_circles
  (A B : ℝ × ℝ)
  (hAB1 : A.1^2 + A.2^2 + 4 * A.1 - 4 * A.2 = 0)
  (hAB2 : B.1^2 + B.2^2 + 4 * B.1 - 4 * B.2 = 0)
  (hAB3 : A.1^2 + A.2^2 + 2 * A.1 - 12 = 0)
  (hAB4 : B.1^2 + B.2^2 + 2 * B.1 - 12 = 0) :
  ∃ (a b c : ℝ), a * A.1 + b * A.2 + c = 0 ∧ a * B.1 + b * B.2 + c = 0 ∧
                  a = 1 ∧ b = -2 ∧ c = 6 :=
sorry

end NUMINAMATH_GPT_line_of_intersecting_circles_l2388_238880


namespace NUMINAMATH_GPT_even_function_f_l2388_238877

noncomputable def f (x : ℝ) : ℝ := if 0 < x ∧ x < 10 then Real.log x else 0

theorem even_function_f (x : ℝ) (h : f (-x) = f x) (h1 : ∀ x, 0 < x ∧ x < 10 → f x = Real.log x) :
  f (-Real.exp 1) + f (Real.exp 2) = 3 := by
  sorry

end NUMINAMATH_GPT_even_function_f_l2388_238877


namespace NUMINAMATH_GPT_vector_sum_magnitude_eq_2_or_5_l2388_238836

noncomputable def a : ℝ := 1
noncomputable def b : ℝ := 1
noncomputable def c : ℝ := 3
def equal_angles (θ : ℝ) := θ = 120 ∨ θ = 0

theorem vector_sum_magnitude_eq_2_or_5
  (a_mag : ℝ := a)
  (b_mag : ℝ := b)
  (c_mag : ℝ := c)
  (θ : ℝ)
  (Hθ : equal_angles θ) :
  (|a_mag| = 1) ∧ (|b_mag| = 1) ∧ (|c_mag| = 3) →
  (|a_mag + b_mag + c_mag| = 2 ∨ |a_mag + b_mag + c_mag| = 5) :=
by
  sorry

end NUMINAMATH_GPT_vector_sum_magnitude_eq_2_or_5_l2388_238836


namespace NUMINAMATH_GPT_P_cubed_plus_7_is_composite_l2388_238884

theorem P_cubed_plus_7_is_composite (P : ℕ) (h_prime_P : Nat.Prime P) (h_prime_P3_plus_5 : Nat.Prime (P^3 + 5)) : ¬ Nat.Prime (P^3 + 7) ∧ (P^3 + 7).factors.length > 1 :=
by
  sorry

end NUMINAMATH_GPT_P_cubed_plus_7_is_composite_l2388_238884


namespace NUMINAMATH_GPT_frank_reads_pages_per_day_l2388_238862

-- Define the conditions and problem statement
def total_pages : ℕ := 450
def total_chapters : ℕ := 41
def total_days : ℕ := 30

-- The derived value we need to prove
def pages_per_day : ℕ := total_pages / total_days

-- The theorem to prove
theorem frank_reads_pages_per_day : pages_per_day = 15 :=
  by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_frank_reads_pages_per_day_l2388_238862


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_l2388_238812

def equation1 (x : ℝ) : Prop := 3 * x^2 + 2 * x - 1 = 0
def equation2 (x : ℝ) : Prop := (x + 2) * (x - 1) = 2 - 2 * x

theorem solve_equation1 :
  (equation1 (-1) ∨ equation1 (1 / 3)) ∧ 
  (∀ x, equation1 x → x = -1 ∨ x = 1 / 3) :=
sorry

theorem solve_equation2 :
  (equation2 1 ∨ equation2 (-4)) ∧ 
  (∀ x, equation2 x → x = 1 ∨ x = -4) :=
sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_l2388_238812


namespace NUMINAMATH_GPT_geometric_series_sum_l2388_238875

theorem geometric_series_sum :
  ∑' n : ℕ, (2 : ℝ) * (1 / 4) ^ n = 8 / 3 := by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l2388_238875


namespace NUMINAMATH_GPT_maria_chairs_l2388_238841

variable (C : ℕ) -- Number of chairs Maria bought
variable (tables : ℕ := 2) -- Number of tables Maria bought is 2
variable (time_per_furniture : ℕ := 8) -- Time spent on each piece of furniture in minutes
variable (total_time : ℕ := 32) -- Total time spent assembling furniture

theorem maria_chairs :
  (time_per_furniture * C + time_per_furniture * tables = total_time) → C = 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_maria_chairs_l2388_238841


namespace NUMINAMATH_GPT_Tina_independent_work_hours_l2388_238805

-- Defining conditions as Lean constants
def Tina_work_rate := 1 / 12
def Ann_work_rate := 1 / 9
def Ann_work_hours := 3

-- Declaring the theorem to be proven
theorem Tina_independent_work_hours : 
  (Ann_work_hours * Ann_work_rate = 1/3) →
  ((1 : ℚ) - (Ann_work_hours * Ann_work_rate)) / Tina_work_rate = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_Tina_independent_work_hours_l2388_238805


namespace NUMINAMATH_GPT_floor_value_correct_l2388_238823

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

end NUMINAMATH_GPT_floor_value_correct_l2388_238823


namespace NUMINAMATH_GPT_problems_completed_l2388_238857

theorem problems_completed (p t : ℕ) (h1 : p ≥ 15) (h2 : p * t = (2 * p - 10) * (t - 1)) : p * t = 60 := sorry

end NUMINAMATH_GPT_problems_completed_l2388_238857


namespace NUMINAMATH_GPT_smallest_number_among_bases_l2388_238831

noncomputable def convert_base_9 (n : ℕ) : ℕ :=
match n with
| 85 => 8 * 9 + 5
| _ => 0

noncomputable def convert_base_4 (n : ℕ) : ℕ :=
match n with
| 1000 => 1 * 4^3
| _ => 0

noncomputable def convert_base_2 (n : ℕ) : ℕ :=
match n with
| 111111 => 1 * 2^6 - 1
| _ => 0

theorem smallest_number_among_bases:
  min (min (convert_base_9 85) (convert_base_4 1000)) (convert_base_2 111111) = convert_base_2 111111 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_number_among_bases_l2388_238831


namespace NUMINAMATH_GPT_fractions_product_l2388_238839

theorem fractions_product : 
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 3 / 7 := 
by 
  sorry

end NUMINAMATH_GPT_fractions_product_l2388_238839


namespace NUMINAMATH_GPT_area_of_square_with_perimeter_40_l2388_238892

theorem area_of_square_with_perimeter_40 (P : ℝ) (s : ℝ) (A : ℝ) 
  (hP : P = 40) (hs : s = P / 4) (hA : A = s^2) : A = 100 :=
by
  sorry

end NUMINAMATH_GPT_area_of_square_with_perimeter_40_l2388_238892


namespace NUMINAMATH_GPT_remainder_when_2x_div_8_is_1_l2388_238849

theorem remainder_when_2x_div_8_is_1 (x y : ℤ) 
  (h1 : x = 11 * y + 4)
  (h2 : ∃ r : ℤ, 2 * x = 8 * (3 * y) + r)
  (h3 : 13 * y - x = 3) : ∃ r : ℤ, r = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_2x_div_8_is_1_l2388_238849


namespace NUMINAMATH_GPT_inverse_variation_l2388_238835

theorem inverse_variation (w : ℝ) (h1 : ∃ (c : ℝ), ∀ (x : ℝ), x^4 * w^(1/4) = c)
  (h2 : (3 : ℝ)^4 * (16 : ℝ)^(1/4) = (6 : ℝ)^4 * w^(1/4)) : 
  w = 1 / 4096 :=
by
  sorry

end NUMINAMATH_GPT_inverse_variation_l2388_238835


namespace NUMINAMATH_GPT_fish_population_estimate_l2388_238855

theorem fish_population_estimate :
  (∀ (x : ℕ),
    ∃ (m n k : ℕ), 
      m = 30 ∧
      k = 2 ∧
      n = 30 ∧
      ((k : ℚ) / n = m / x) → x = 450) :=
by
  sorry

end NUMINAMATH_GPT_fish_population_estimate_l2388_238855


namespace NUMINAMATH_GPT_arithmetic_expression_eval_l2388_238867

theorem arithmetic_expression_eval : 
  (1000 * 0.09999) / 10 * 999 = 998001 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_expression_eval_l2388_238867


namespace NUMINAMATH_GPT_value_of_expression_l2388_238888

variable (x1 x2 : ℝ)

def sum_roots (x1 x2 : ℝ) : Prop := x1 + x2 = 3
def product_roots (x1 x2 : ℝ) : Prop := x1 * x2 = -4

theorem value_of_expression (h1 : sum_roots x1 x2) (h2 : product_roots x1 x2) : 
  x1^2 - 4*x1 - x2 + 2*x1*x2 = -7 :=
by sorry

end NUMINAMATH_GPT_value_of_expression_l2388_238888


namespace NUMINAMATH_GPT_bike_route_length_l2388_238891

theorem bike_route_length (u1 u2 u3 l1 l2 : ℕ) (h1 : u1 = 4) (h2 : u2 = 7) (h3 : u3 = 2) (h4 : l1 = 6) (h5 : l2 = 7) :
  u1 + u2 + u3 + u1 + u2 + u3 + l1 + l2 + l1 + l2 = 52 := 
by
  sorry

end NUMINAMATH_GPT_bike_route_length_l2388_238891


namespace NUMINAMATH_GPT_bakery_profit_l2388_238819

noncomputable def revenue_per_piece : ℝ := 4
noncomputable def pieces_per_pie : ℕ := 3
noncomputable def pies_per_hour : ℕ := 12
noncomputable def cost_per_pie : ℝ := 0.5

theorem bakery_profit (pieces_per_pie_pos : 0 < pieces_per_pie) 
                      (pies_per_hour_pos : 0 < pies_per_hour) 
                      (cost_per_pie_pos : 0 < cost_per_pie) :
  pies_per_hour * (pieces_per_pie * revenue_per_piece) - (pies_per_hour * cost_per_pie) = 138 := 
sorry

end NUMINAMATH_GPT_bakery_profit_l2388_238819


namespace NUMINAMATH_GPT_imaginary_part_of_z_l2388_238870

-- Define the complex number z
def z : Complex := Complex.mk 3 (-4)

-- State the proof goal
theorem imaginary_part_of_z : z.im = -4 :=
by
  sorry

end NUMINAMATH_GPT_imaginary_part_of_z_l2388_238870


namespace NUMINAMATH_GPT_triangle_at_most_one_obtuse_l2388_238897

theorem triangle_at_most_one_obtuse (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A > 90 → B + C < 90) (h3 : B > 90 → A + C < 90) (h4 : C > 90 → A + B < 90) :
  ¬ (A > 90 ∧ B > 90 ∨ B > 90 ∧ C > 90 ∨ A > 90 ∧ C > 90) :=
by
  sorry

end NUMINAMATH_GPT_triangle_at_most_one_obtuse_l2388_238897


namespace NUMINAMATH_GPT_cost_function_discrete_points_l2388_238806

def cost (n : ℕ) : ℕ :=
  if n <= 10 then 20 * n
  else if n <= 25 then 18 * n
  else 0

theorem cost_function_discrete_points :
  (∀ n, 1 ≤ n ∧ n ≤ 25 → ∃ y, cost n = y) ∧
  (∀ m n, 1 ≤ m ∧ m ≤ 25 ∧ 1 ≤ n ∧ n ≤ 25 ∧ m ≠ n → cost m ≠ cost n) :=
sorry

end NUMINAMATH_GPT_cost_function_discrete_points_l2388_238806


namespace NUMINAMATH_GPT_find_a_l2388_238851

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  Real.log (1 - x) - Real.log (1 + x) + a

theorem find_a 
  (M : ℝ) (N : ℝ) (a : ℝ)
  (h1 : M = f a (-1/2))
  (h2 : N = f a (1/2))
  (h3 : M + N = 1) :
  a = 1 / 2 := 
sorry

end NUMINAMATH_GPT_find_a_l2388_238851


namespace NUMINAMATH_GPT_find_number_l2388_238871

theorem find_number (x : ℤ) (h : 33 + 3 * x = 48) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2388_238871


namespace NUMINAMATH_GPT_jasmine_percentage_is_approx_l2388_238886

noncomputable def initial_solution_volume : ℝ := 80
noncomputable def initial_jasmine_percent : ℝ := 0.10
noncomputable def initial_lemon_percent : ℝ := 0.05
noncomputable def initial_orange_percent : ℝ := 0.03
noncomputable def added_jasmine_volume : ℝ := 8
noncomputable def added_water_volume : ℝ := 12
noncomputable def added_lemon_volume : ℝ := 6
noncomputable def added_orange_volume : ℝ := 7

noncomputable def initial_jasmine_volume := initial_solution_volume * initial_jasmine_percent
noncomputable def initial_lemon_volume := initial_solution_volume * initial_lemon_percent
noncomputable def initial_orange_volume := initial_solution_volume * initial_orange_percent
noncomputable def initial_water_volume := initial_solution_volume - (initial_jasmine_volume + initial_lemon_volume + initial_orange_volume)

noncomputable def new_jasmine_volume := initial_jasmine_volume + added_jasmine_volume
noncomputable def new_water_volume := initial_water_volume + added_water_volume
noncomputable def new_lemon_volume := initial_lemon_volume + added_lemon_volume
noncomputable def new_orange_volume := initial_orange_volume + added_orange_volume
noncomputable def new_total_volume := new_jasmine_volume + new_water_volume + new_lemon_volume + new_orange_volume

noncomputable def new_jasmine_percent := (new_jasmine_volume / new_total_volume) * 100

theorem jasmine_percentage_is_approx :
  abs (new_jasmine_percent - 14.16) < 0.01 := sorry

end NUMINAMATH_GPT_jasmine_percentage_is_approx_l2388_238886


namespace NUMINAMATH_GPT_meeting_at_centroid_l2388_238803

theorem meeting_at_centroid :
  let A := (2, 9)
  let B := (-3, -4)
  let C := (6, -1)
  let centroid := ((2 - 3 + 6) / 3, (9 - 4 - 1) / 3)
  centroid = (5 / 3, 4 / 3) := sorry

end NUMINAMATH_GPT_meeting_at_centroid_l2388_238803


namespace NUMINAMATH_GPT_arithmetic_progression_exists_l2388_238859

theorem arithmetic_progression_exists (a_1 a_2 a_3 a_4 : ℕ) (d : ℕ) :
  a_2 = a_1 + d →
  a_3 = a_1 + 2 * d →
  a_4 = a_1 + 3 * d →
  a_1 * a_2 * a_3 = 6 →
  a_1 * a_2 * a_3 * a_4 = 24 →
  a_1 = 1 ∧ a_2 = 2 ∧ a_3 = 3 ∧ a_4 = 4 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_progression_exists_l2388_238859


namespace NUMINAMATH_GPT_find_extra_factor_l2388_238881

theorem find_extra_factor (w : ℕ) (h1 : w > 0) (h2 : w = 156) (h3 : ∃ (k : ℕ), (2^5 * 13^2) ∣ (936 * w))
  : 3 ∣ w := sorry

end NUMINAMATH_GPT_find_extra_factor_l2388_238881


namespace NUMINAMATH_GPT_molecular_weight_proof_l2388_238842

def atomic_weight_Al : Float := 26.98
def atomic_weight_O : Float := 16.00
def atomic_weight_H : Float := 1.01

def molecular_weight_AlOH3 : Float :=
  (1 * atomic_weight_Al) + (3 * atomic_weight_O) + (3 * atomic_weight_H)

def moles : Float := 7.0

def molecular_weight_7_moles_AlOH3 : Float :=
  moles * molecular_weight_AlOH3

theorem molecular_weight_proof : molecular_weight_7_moles_AlOH3 = 546.07 :=
by
  /- Here we calculate the molecular weight of Al(OH)3 and multiply it by 7.
     molecular_weight_AlOH3 = (1 * 26.98) + (3 * 16.00) + (3 * 1.01) = 78.01
     molecular_weight_7_moles_AlOH3 = 7 * 78.01 = 546.07 -/
  sorry

end NUMINAMATH_GPT_molecular_weight_proof_l2388_238842


namespace NUMINAMATH_GPT_standard_equation_of_ellipse_l2388_238882

theorem standard_equation_of_ellipse :
  ∀ (m n : ℝ), 
    (m > 0 ∧ n > 0) →
    (∃ (c : ℝ), c^2 = m^2 - n^2 ∧ c = 2) →
    (∃ (e : ℝ), e = c / m ∧ e = 1 / 2) →
    (m = 4 ∧ n = 2 * Real.sqrt 3) →
    (∀ x y : ℝ, (x^2 / 16 + y^2 / 12 = 1)) :=
by
  intros m n hmn hc he hm_eq hn_eq
  sorry

end NUMINAMATH_GPT_standard_equation_of_ellipse_l2388_238882


namespace NUMINAMATH_GPT_quadratic_roots_solve_equation_l2388_238820

theorem quadratic_roots (a b c : ℝ) (x1 x2 : ℝ) (h : a ≠ 0)
  (root_eq : x1 = (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
            ∧ x2 = (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a))
  (h_eq : a*x^2 + b*x + c = 0) :
  ∀ x, a*x^2 + b*x + c = 0 → x = x1 ∨ x = x2 :=
by
  sorry -- Proof not given

theorem solve_equation (x : ℝ) :
  7*x*(5*x + 2) = 6*(5*x + 2) ↔ x = -2 / 5 ∨ x = 6 / 7 :=
by
  sorry -- Proof not given

end NUMINAMATH_GPT_quadratic_roots_solve_equation_l2388_238820


namespace NUMINAMATH_GPT_vasya_days_without_purchase_l2388_238801

variables (x y z w : ℕ)

-- Given conditions as assumptions
def total_days : Prop := x + y + z + w = 15
def total_marshmallows : Prop := 9 * x + 4 * z = 30
def total_meat_pies : Prop := 2 * y + z = 9

-- Prove w = 7
theorem vasya_days_without_purchase (h1 : total_days x y z w) 
                                     (h2 : total_marshmallows x z) 
                                     (h3 : total_meat_pies y z) : 
  w = 7 :=
by
  -- Code placeholder to satisfy the theorem's syntax
  sorry

end NUMINAMATH_GPT_vasya_days_without_purchase_l2388_238801


namespace NUMINAMATH_GPT_jenna_filter_change_15th_is_March_l2388_238833

def month_of_nth_change (startMonth interval n : ℕ) : ℕ :=
  ((interval * (n - 1)) % 12 + startMonth) % 12

theorem jenna_filter_change_15th_is_March :
  month_of_nth_change 1 7 15 = 3 := 
  sorry

end NUMINAMATH_GPT_jenna_filter_change_15th_is_March_l2388_238833


namespace NUMINAMATH_GPT_parabola_reflection_translation_l2388_238822

open Real

noncomputable def f (a b c x : ℝ) : ℝ := a * (x - 4)^2 + b * (x - 4) + c
noncomputable def g (a b c x : ℝ) : ℝ := -a * (x + 4)^2 - b * (x + 4) - c
noncomputable def fg_x (a b c x : ℝ) : ℝ := f a b c x + g a b c x

theorem parabola_reflection_translation (a b c x : ℝ) (ha : a ≠ 0) :
  fg_x a b c x = -16 * a * x :=
by
  sorry

end NUMINAMATH_GPT_parabola_reflection_translation_l2388_238822


namespace NUMINAMATH_GPT_min_value_of_n_l2388_238873

theorem min_value_of_n : 
  ∃ (n : ℕ), (∃ r : ℕ, 4 * n - 7 * r = 0) ∧ n = 7 := 
sorry

end NUMINAMATH_GPT_min_value_of_n_l2388_238873


namespace NUMINAMATH_GPT_starting_player_ensures_non_trivial_solution_l2388_238861

theorem starting_player_ensures_non_trivial_solution :
  ∀ (a1 b1 c1 a2 b2 c2 a3 b3 c3 : ℚ), 
    ∃ (x y z : ℚ), 
    ((a1 * x + b1 * y + c1 * z = 0) ∧ 
     (a2 * x + b2 * y + c2 * z = 0) ∧ 
     (a3 * x + b3 * y + c3 * z = 0)) 
    ∧ ((a1 * (b2 * c3 - b3 * c2) - b1 * (a2 * c3 - a3 * c2) + c1 * (a2 * b3 - a3 * b2) = 0) ∧ 
         (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) :=
by
  intros a1 b1 c1 a2 b2 c2 a3 b3 c3
  sorry

end NUMINAMATH_GPT_starting_player_ensures_non_trivial_solution_l2388_238861


namespace NUMINAMATH_GPT_inequality_proof_l2388_238843

theorem inequality_proof (a b c d : ℝ) 
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) 
  (h_sum : a * b + b * c + c * d + d * a = 1) : 
  (a^3 / (b + c + d)) + (b^3 / (a + c + d)) + (c^3 / (a + b + d)) + (d^3 / (a + b + c)) ≥ (1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l2388_238843


namespace NUMINAMATH_GPT_average_age_in_club_l2388_238864

theorem average_age_in_club (women men children : ℕ) 
    (avg_age_women avg_age_men avg_age_children : ℤ)
    (hw : women = 12) (hm : men = 18) (hc : children = 20)
    (haw : avg_age_women = 32) (ham : avg_age_men = 36) (hac : avg_age_children = 10) :
    (12 * 32 + 18 * 36 + 20 * 10) / (12 + 18 + 20) = 24 := by
  sorry

end NUMINAMATH_GPT_average_age_in_club_l2388_238864


namespace NUMINAMATH_GPT_f_is_even_f_range_l2388_238863

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (|x| + 2) / (1 - |x|)

-- Prove that f(x) is an even function
theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

-- Prove the range of f(x) is (-∞, -1) ∪ [2, +∞)
theorem f_range : ∀ y : ℝ, ∃ x : ℝ, y = f x ↔ y ≥ 2 ∨ y < -1 := by
  sorry

end NUMINAMATH_GPT_f_is_even_f_range_l2388_238863


namespace NUMINAMATH_GPT_max_slope_tangent_eqn_l2388_238848

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem max_slope_tangent_eqn (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi) :
    (∃ m b, m = Real.sqrt 2 ∧ b = -Real.sqrt 2 * (Real.pi / 4) ∧ 
    (∀ y, y = m * x + b)) :=
sorry

end NUMINAMATH_GPT_max_slope_tangent_eqn_l2388_238848


namespace NUMINAMATH_GPT_number_of_schools_l2388_238838

theorem number_of_schools (total_students d : ℕ) (S : ℕ) (ellen frank : ℕ) (d_median : total_students = 2 * d - 1)
    (d_highest : ellen < d) (ellen_position : ellen = 29) (frank_position : frank = 50) (team_size : ∀ S, total_students = 3 * S) : 
    S = 19 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_schools_l2388_238838


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l2388_238868

variable (a : ℕ → ℝ)
variable (d : ℝ)
variable (a1 : ℝ)

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (a1 : ℝ) (d : ℝ) :=
  ∀ n, a n = a1 + n * d

-- Given condition
variable (h1 : a 3 + a 4 + a 5 = 36)

-- The goal is to prove that a 0 + a 8 = 24
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (a1 : ℝ) (d : ℝ) :
  arithmetic_sequence a a1 d →
  a 3 + a 4 + a 5 = 36 →
  a 0 + a 8 = 24 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l2388_238868


namespace NUMINAMATH_GPT_train_speed_in_km_per_hr_l2388_238885

variables (L : ℕ) (t : ℕ) (train_speed : ℕ)

-- Conditions
def length_of_train : ℕ := 1050
def length_of_platform : ℕ := 1050
def crossing_time : ℕ := 1

-- Given calculation of speed in meters per minute
def speed_in_m_per_min : ℕ := (length_of_train + length_of_platform) / crossing_time

-- Conversion units
def meters_to_kilometers (m : ℕ) : ℕ := m / 1000
def minutes_to_hours (min : ℕ) : ℕ := min / 60

-- Speed in km/hr
def speed_in_km_per_hr : ℕ := speed_in_m_per_min * (meters_to_kilometers 1000) * (minutes_to_hours 60)

theorem train_speed_in_km_per_hr : speed_in_km_per_hr = 35 :=
by {
  -- We will include the proof steps here, but for now, we just assert with sorry.
  sorry
}

end NUMINAMATH_GPT_train_speed_in_km_per_hr_l2388_238885


namespace NUMINAMATH_GPT_inequality_abc_l2388_238898

variable {a b c : ℝ}

theorem inequality_abc (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : a * b * c = 8) :
  (a - 2) / (a + 1) + (b - 2) / (b + 1) + (c - 2) / (c + 1) ≤ 0 := by
  sorry

end NUMINAMATH_GPT_inequality_abc_l2388_238898


namespace NUMINAMATH_GPT_combined_average_age_l2388_238854

theorem combined_average_age 
    (avgA : ℕ → ℕ → ℕ) -- defines the average function
    (avgA_cond : avgA 6 240 = 40) 
    (avgB : ℕ → ℕ → ℕ)
    (avgB_cond : avgB 4 100 = 25) 
    (combined_total_age : ℕ := 340) 
    (total_people : ℕ := 10) : avgA (total_people) (combined_total_age) = 34 := 
by
  sorry

end NUMINAMATH_GPT_combined_average_age_l2388_238854


namespace NUMINAMATH_GPT_sin_C_eq_sqrt14_div_8_area_triangle_eq_sqrt7_div_4_l2388_238809

theorem sin_C_eq_sqrt14_div_8 (b c : ℝ) (cosB : ℝ) (h1 : b = Real.sqrt 2) (h2 : c = 1) (h3 : cosB = 3 / 4) : 
  let sinB := Real.sqrt (1 - cosB^2)
  let sinC := c * sinB / b
  sinC = Real.sqrt 14 / 8 := 
by
  -- Proof is omitted
  sorry

theorem area_triangle_eq_sqrt7_div_4 (b c : ℝ) (cosB : ℝ) (h1 : b = Real.sqrt 2) (h2 : c = 1) (h3 : cosB = 3 / 4) : 
  let sinB := Real.sqrt (1 - cosB^2)
  let sinC := c * sinB / b
  let cosC := Real.sqrt (1 - sinC^2)
  let sinA := sinB * cosC + cosB * sinC
  let area := 1 / 2 * b * c * sinA
  area = Real.sqrt 7 / 4 := 
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_sin_C_eq_sqrt14_div_8_area_triangle_eq_sqrt7_div_4_l2388_238809


namespace NUMINAMATH_GPT_solve_floor_sum_eq_125_l2388_238844

def floorSum (x : ℕ) : ℕ :=
  (x - 1) * x * (4 * x + 1) / 6

theorem solve_floor_sum_eq_125 (x : ℕ) (h_pos : 0 < x) : floorSum x = 125 → x = 6 := by
  sorry

end NUMINAMATH_GPT_solve_floor_sum_eq_125_l2388_238844


namespace NUMINAMATH_GPT_shopkeeper_percentage_gain_l2388_238834

theorem shopkeeper_percentage_gain (false_weight true_weight : ℝ) 
    (h_false_weight : false_weight = 930)
    (h_true_weight : true_weight = 1000) : 
    (true_weight - false_weight) / false_weight * 100 = 7.53 := 
by
  rw [h_false_weight, h_true_weight]
  sorry

end NUMINAMATH_GPT_shopkeeper_percentage_gain_l2388_238834


namespace NUMINAMATH_GPT_smallest_four_digits_valid_remainder_l2388_238865

def isFourDigit (x : ℕ) : Prop := 1000 ≤ x ∧ x ≤ 9999 

def validRemainder (x : ℕ) : Prop := 
  ∀ k ∈ [2, 3, 4, 5, 6], x % k = 1

theorem smallest_four_digits_valid_remainder :
  ∃ x1 x2 x3 x4 : ℕ,
    isFourDigit x1 ∧ validRemainder x1 ∧
    isFourDigit x2 ∧ validRemainder x2 ∧
    isFourDigit x3 ∧ validRemainder x3 ∧
    isFourDigit x4 ∧ validRemainder x4 ∧
    x1 = 1021 ∧ x2 = 1081 ∧ x3 = 1141 ∧ x4 = 1201 := 
sorry

end NUMINAMATH_GPT_smallest_four_digits_valid_remainder_l2388_238865


namespace NUMINAMATH_GPT_joe_used_paint_total_l2388_238816

theorem joe_used_paint_total :
  let first_airport_paint := 360
  let second_airport_paint := 600
  let first_week_first_airport := (1/4 : ℝ) * first_airport_paint
  let remaining_first_airport := first_airport_paint - first_week_first_airport
  let second_week_first_airport := (1/6 : ℝ) * remaining_first_airport
  let total_first_airport := first_week_first_airport + second_week_first_airport
  let first_week_second_airport := (1/3 : ℝ) * second_airport_paint
  let remaining_second_airport := second_airport_paint - first_week_second_airport
  let second_week_second_airport := (1/5 : ℝ) * remaining_second_airport
  let total_second_airport := first_week_second_airport + second_week_second_airport
  total_first_airport + total_second_airport = 415 :=
by
  let first_airport_paint := 360
  let second_airport_paint := 600
  let first_week_first_airport := (1/4 : ℝ) * first_airport_paint
  let remaining_first_airport := first_airport_paint - first_week_first_airport
  let second_week_first_airport := (1/6 : ℝ) * remaining_first_airport
  let total_first_airport := first_week_first_airport + second_week_first_airport
  let first_week_second_airport := (1/3 : ℝ) * second_airport_paint
  let remaining_second_airport := second_airport_paint - first_week_second_airport
  let second_week_second_airport := (1/5 : ℝ) * remaining_second_airport
  let total_second_airport := first_week_second_airport + second_week_second_airport
  show total_first_airport + total_second_airport = 415
  sorry

end NUMINAMATH_GPT_joe_used_paint_total_l2388_238816


namespace NUMINAMATH_GPT_least_number_subtracted_l2388_238896

theorem least_number_subtracted (n m k : ℕ) (h1 : n = 3830) (h2 : k = 15) (h3 : n % k = m) (h4 : m = 5) : 
  (n - m) % k = 0 :=
by
  sorry

end NUMINAMATH_GPT_least_number_subtracted_l2388_238896


namespace NUMINAMATH_GPT_temperature_problem_l2388_238828

theorem temperature_problem
  (M L N : ℝ)
  (h1 : M = L + N)
  (h2 : M - 9 = M - 9)
  (h3 : L + 5 = L + 5)
  (h4 : abs (M - 9 - (L + 5)) = 1) :
  (N = 15 ∨ N = 13) → (N = 15 ∧ N = 13 → 15 * 13 = 195) :=
by
  sorry

end NUMINAMATH_GPT_temperature_problem_l2388_238828


namespace NUMINAMATH_GPT_part1_part2_l2388_238887

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := abs (2 * x - a) + a
def g (x : ℝ) : ℝ := abs (2 * x - 1)

theorem part1 (x : ℝ) : f x 2 ≤ 6 ↔ -1 ≤ x ∧ x ≤ 3 := by
  sorry

theorem part2 (a : ℝ) : 2 ≤ a ↔ ∀ (x : ℝ), f x a + g x ≥ 3 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l2388_238887


namespace NUMINAMATH_GPT_faster_train_passes_slower_l2388_238804

theorem faster_train_passes_slower (v_fast v_slow : ℝ) (length_fast : ℝ) 
  (hv_fast : v_fast = 50) (hv_slow : v_slow = 32) (hl_length_fast : length_fast = 75) :
  ∃ t : ℝ, t = 15 := 
by
  sorry

end NUMINAMATH_GPT_faster_train_passes_slower_l2388_238804
