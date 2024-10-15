import Mathlib

namespace NUMINAMATH_GPT_trigonometric_identity_l808_80898

theorem trigonometric_identity (α : ℝ) 
  (h : Real.cos (5 * Real.pi / 12 - α) = Real.sqrt 2 / 3) :
  Real.sqrt 3 * Real.cos (2 * α) - Real.sin (2 * α) = 10 / 9 := sorry

end NUMINAMATH_GPT_trigonometric_identity_l808_80898


namespace NUMINAMATH_GPT_expenses_denoted_as_negative_l808_80839

theorem expenses_denoted_as_negative (income_yuan expenses_yuan : Int) (h : income_yuan = 6) : 
  expenses_yuan = -4 :=
by
  sorry

end NUMINAMATH_GPT_expenses_denoted_as_negative_l808_80839


namespace NUMINAMATH_GPT_compare_neg_fractions_l808_80815

theorem compare_neg_fractions : (-2 / 3 : ℚ) < -3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_compare_neg_fractions_l808_80815


namespace NUMINAMATH_GPT_evaluate_expression_l808_80836

theorem evaluate_expression : 
  (Int.ceil ((Int.floor ((15 / 8 : Rat) ^ 2) : Rat) - (19 / 5 : Rat) : Rat) : Int) = 0 :=
sorry

end NUMINAMATH_GPT_evaluate_expression_l808_80836


namespace NUMINAMATH_GPT_find_set_A_l808_80819

def M : Set ℤ := {1, 3, 5, 7, 9}

def satisfiesCondition (A : Set ℤ) : Prop :=
  A ≠ ∅ ∧
  (∀ a ∈ A, a + 4 ∈ M) ∧
  (∀ a ∈ A, a - 4 ∈ M)

theorem find_set_A : ∃ A : Set ℤ, satisfiesCondition A ∧ A = {5} :=
  by
    sorry

end NUMINAMATH_GPT_find_set_A_l808_80819


namespace NUMINAMATH_GPT_percent_sugar_in_resulting_solution_l808_80821

theorem percent_sugar_in_resulting_solution (W : ℝ) (hW : W > 0) :
  let original_sugar_percent := 22 / 100
  let second_solution_sugar_percent := 74 / 100
  let remaining_original_weight := (3 / 4) * W
  let removed_weight := (1 / 4) * W
  let sugar_from_remaining_original := (original_sugar_percent * remaining_original_weight)
  let sugar_from_added_second_solution := (second_solution_sugar_percent * removed_weight)
  let total_sugar := sugar_from_remaining_original + sugar_from_added_second_solution
  let resulting_sugar_percent := total_sugar / W
  resulting_sugar_percent = 35 / 100 :=
by
  sorry

end NUMINAMATH_GPT_percent_sugar_in_resulting_solution_l808_80821


namespace NUMINAMATH_GPT_tan_theta_plus_pi_over_eight_sub_inv_l808_80854

/-- Given the trigonometric identity, we can prove the tangent calculation -/
theorem tan_theta_plus_pi_over_eight_sub_inv (θ : ℝ)
  (h : 3 * Real.sin θ + Real.cos θ = Real.sqrt 10) :
  Real.tan (θ + Real.pi / 8) - 1 / Real.tan (θ + Real.pi / 8) = -14 := 
sorry

end NUMINAMATH_GPT_tan_theta_plus_pi_over_eight_sub_inv_l808_80854


namespace NUMINAMATH_GPT_magnitude_of_root_of_quadratic_eq_l808_80852

open Complex

theorem magnitude_of_root_of_quadratic_eq (z : ℂ) 
  (h : z^2 - (2 : ℂ) * z + 2 = 0) : abs z = Real.sqrt 2 :=
by 
  sorry

end NUMINAMATH_GPT_magnitude_of_root_of_quadratic_eq_l808_80852


namespace NUMINAMATH_GPT_flower_shop_types_l808_80888

variable (C V T R F : ℕ)

-- Define the conditions
def condition1 : Prop := V = C / 3
def condition2 : Prop := T = V / 4
def condition3 : Prop := R = T
def condition4 : Prop := C = (2 / 3) * F

-- The main statement we need to prove: the shop stocks 4 types of flowers
theorem flower_shop_types
  (h1 : condition1 C V)
  (h2 : condition2 V T)
  (h3 : condition3 T R)
  (h4 : condition4 C F) :
  4 = 4 :=
by 
  sorry

end NUMINAMATH_GPT_flower_shop_types_l808_80888


namespace NUMINAMATH_GPT_zack_marbles_number_l808_80840

-- Define the conditions as Lean definitions
def zack_initial_marbles (x : ℕ) :=
  (∃ k : ℕ, x = 3 * k + 5) ∧ (3 * 20 + 5 = 65)

-- State the theorem using the conditions
theorem zack_marbles_number : ∃ x : ℕ, zack_initial_marbles x ∧ x = 65 :=
by
  sorry

end NUMINAMATH_GPT_zack_marbles_number_l808_80840


namespace NUMINAMATH_GPT_line_perpendicular_l808_80807

theorem line_perpendicular (m : ℝ) : 
  -- Conditions
  (∀ x y : ℝ, x - 2 * y + 5 = 0 → y = 1/2 * x + 5/2) →  -- Slope of the first line
  (∀ x y : ℝ, 2 * x + m * y - 6 = 0 → y = -2/m * x + 6/m) →  -- Slope of the second line
  -- Perpendicular condition
  ((1/2) * (-2/m) = -1) →
  -- Conclusion
  m = 1 := 
sorry

end NUMINAMATH_GPT_line_perpendicular_l808_80807


namespace NUMINAMATH_GPT_maximize_integral_l808_80899
open Real

noncomputable def integral_to_maximize (a b : ℝ) : ℝ :=
  ∫ x in a..b, exp (cos x) * (380 - x - x^2)

theorem maximize_integral :
  ∀ (a b : ℝ), a ≤ b → integral_to_maximize a b ≤ integral_to_maximize (-20) 19 :=
by
  intros a b h
  sorry

end NUMINAMATH_GPT_maximize_integral_l808_80899


namespace NUMINAMATH_GPT_expression_is_minus_two_l808_80847

noncomputable def A : ℝ := (Real.sqrt 6 + Real.sqrt 2) * (Real.sqrt 3 - 2) * Real.sqrt (Real.sqrt 3 + 2)

theorem expression_is_minus_two : A = -2 := by
  sorry

end NUMINAMATH_GPT_expression_is_minus_two_l808_80847


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l808_80853

noncomputable def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

def z (a : ℝ) : ℂ := ⟨a^2 - 4, a + 1⟩

theorem sufficient_but_not_necessary (a : ℝ) (h : a = -2) : 
  is_purely_imaginary (z a) ∧ ¬(∀ a, is_purely_imaginary (z a) → a = -2) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l808_80853


namespace NUMINAMATH_GPT_four_digit_positive_integers_count_l808_80802

theorem four_digit_positive_integers_count :
  let p := 17
  let a := 4582 % p
  let b := 902 % p
  let c := 2345 % p
  ∃ (n : ℕ), 
    (1000 ≤ 14 + p * n ∧ 14 + p * n ≤ 9999) ∧ 
    (4582 * (14 + p * n) + 902 ≡ 2345 [MOD p]) ∧ 
    n = 530 := sorry

end NUMINAMATH_GPT_four_digit_positive_integers_count_l808_80802


namespace NUMINAMATH_GPT_find_z_solutions_l808_80831

theorem find_z_solutions (r : ℚ) (z : ℤ) (h : 2^z + 2 = r^2) : 
  (r = 2 ∧ z = 1) ∨ (r = -2 ∧ z = 1) ∨ (r = 3/2 ∧ z = -2) ∨ (r = -3/2 ∧ z = -2) :=
sorry

end NUMINAMATH_GPT_find_z_solutions_l808_80831


namespace NUMINAMATH_GPT_solve_xyz_l808_80882

theorem solve_xyz (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 3 * y + x) : x + y + z = 14 * x :=
by
  sorry

end NUMINAMATH_GPT_solve_xyz_l808_80882


namespace NUMINAMATH_GPT_gain_percent_40_l808_80842

theorem gain_percent_40 (cost_price selling_price : ℝ) (h1 : cost_price = 900) (h2 : selling_price = 1260) :
  ((selling_price - cost_price) / cost_price) * 100 = 40 :=
by
  sorry

end NUMINAMATH_GPT_gain_percent_40_l808_80842


namespace NUMINAMATH_GPT_line_through_point_area_T_l808_80895

variable (a T : ℝ)

def equation_of_line (x y : ℝ) : Prop := 2 * T * x - a^2 * y + 2 * a * T = 0

theorem line_through_point_area_T :
  ∃ (x y : ℝ), equation_of_line a T x y ∧ x = -a ∧ y = (2 * T) / a :=
by
  sorry

end NUMINAMATH_GPT_line_through_point_area_T_l808_80895


namespace NUMINAMATH_GPT_quadratic_has_real_roots_b_3_c_1_l808_80804

theorem quadratic_has_real_roots_b_3_c_1 :
  ∃ x₁ x₂ : ℝ, (∀ x : ℝ, x * x + 3 * x + 1 = 0 ↔ x = x₁ ∨ x = x₂) ∧
  x₁ = (-3 + Real.sqrt 5) / 2 ∧
  x₂ = (-3 - Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_real_roots_b_3_c_1_l808_80804


namespace NUMINAMATH_GPT_calc_derivative_at_pi_over_2_l808_80872

noncomputable def f (x: ℝ) : ℝ := Real.exp x * Real.cos x

theorem calc_derivative_at_pi_over_2 : (deriv f) (Real.pi / 2) = -Real.exp (Real.pi / 2) :=
by
  sorry

end NUMINAMATH_GPT_calc_derivative_at_pi_over_2_l808_80872


namespace NUMINAMATH_GPT_balloon_totals_l808_80845

-- Definitions
def Joan_blue := 40
def Joan_red := 30
def Joan_green := 0
def Joan_yellow := 0

def Melanie_blue := 41
def Melanie_red := 0
def Melanie_green := 20
def Melanie_yellow := 0

def Eric_blue := 0
def Eric_red := 25
def Eric_green := 0
def Eric_yellow := 15

-- Total counts
def total_blue := Joan_blue + Melanie_blue + Eric_blue
def total_red := Joan_red + Melanie_red + Eric_red
def total_green := Joan_green + Melanie_green + Eric_green
def total_yellow := Joan_yellow + Melanie_yellow + Eric_yellow

-- Statement of the problem
theorem balloon_totals :
  total_blue = 81 ∧
  total_red = 55 ∧
  total_green = 20 ∧
  total_yellow = 15 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_balloon_totals_l808_80845


namespace NUMINAMATH_GPT_problem1_problem2_l808_80851

-- Define the function f(x)
def f (x m : ℝ) : ℝ := abs (x - m) - abs (x + 3 * m)

-- Condition that m must be greater than 0
variable {m : ℝ} (hm : m > 0)

-- First problem statement: When m=1, the solution set for f(x) ≥ 1 is x ≤ -3/2.
theorem problem1 (x : ℝ) (h : f x 1 ≥ 1) : x ≤ -3 / 2 :=
sorry

-- Second problem statement: The range of values for m such that f(x) < |2 + t| + |t - 1| holds for all x and t is 0 < m < 3/4.
theorem problem2 (m : ℝ) : (∀ (x t : ℝ), f x m < abs (2 + t) + abs (t - 1)) ↔ (0 < m ∧ m < 3 / 4) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l808_80851


namespace NUMINAMATH_GPT_number_of_friends_l808_80825

-- Let n be the number of friends
-- Given the conditions:
-- 1. 9 chicken wings initially.
-- 2. 7 more chicken wings cooked.
-- 3. Each friend gets 4 chicken wings.

theorem number_of_friends :
  let initial_wings := 9
  let additional_wings := 7
  let wings_per_friend := 4
  let total_wings := initial_wings + additional_wings
  let n := total_wings / wings_per_friend
  n = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_friends_l808_80825


namespace NUMINAMATH_GPT_problem_solving_example_l808_80869

theorem problem_solving_example (α β : ℝ) (h1 : α + β = 3) (h2 : α * β = 1) (h3 : α^2 - 3 * α + 1 = 0) (h4 : β^2 - 3 * β + 1 = 0) :
  7 * α^5 + 8 * β^4 = 1448 :=
sorry

end NUMINAMATH_GPT_problem_solving_example_l808_80869


namespace NUMINAMATH_GPT_find_b_l808_80857

-- Define the curve and the line equations
def curve (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x + 1
def line (k : ℝ) (b : ℝ) (x : ℝ) : ℝ := k * x + b

-- Define the conditions in the problem
def passes_through_point (a : ℝ) : Prop := curve a 2 = 3
def is_tangent_at_point (a k b : ℝ) : Prop :=
  ∀ x : ℝ, curve a x = 3 → line k b 2 = 3

-- Main theorem statement
theorem find_b (a k b : ℝ) (h1 : passes_through_point a) (h2 : is_tangent_at_point a k b) : b = -15 :=
by sorry

end NUMINAMATH_GPT_find_b_l808_80857


namespace NUMINAMATH_GPT_missed_questions_l808_80866

theorem missed_questions (F U : ℕ) (h1 : U = 5 * F) (h2 : F + U = 216) : U = 180 :=
by
  sorry

end NUMINAMATH_GPT_missed_questions_l808_80866


namespace NUMINAMATH_GPT_intersection_unique_one_point_l808_80810

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 7 * x + a
noncomputable def g (x : ℝ) : ℝ := -3 * x^2 + 5 * x - 6

theorem intersection_unique_one_point (a : ℝ) :
  (∃ x y, y = f x a ∧ y = g x) ↔ a = 3 := by
  sorry

end NUMINAMATH_GPT_intersection_unique_one_point_l808_80810


namespace NUMINAMATH_GPT_combined_area_is_correct_l808_80820

def tract1_length := 300
def tract1_width  := 500
def tract2_length := 250
def tract2_width  := 630
def tract3_length := 350
def tract3_width  := 450
def tract4_length := 275
def tract4_width  := 600
def tract5_length := 325
def tract5_width  := 520

def area (length width : ℕ) : ℕ := length * width

theorem combined_area_is_correct :
  area tract1_length tract1_width +
  area tract2_length tract2_width +
  area tract3_length tract3_width +
  area tract4_length tract4_width +
  area tract5_length tract5_width = 799000 :=
by
  sorry

end NUMINAMATH_GPT_combined_area_is_correct_l808_80820


namespace NUMINAMATH_GPT_xiao_ming_final_score_correct_l808_80890

/-- Xiao Ming's scores in image, content, and effect are 9, 8, and 8 points, respectively.
    The weights (ratios) for these scores are 3:4:3.
    Prove that Xiao Ming's final competition score is 8.3 points. -/
def xiao_ming_final_score : Prop :=
  let image_score := 9
  let content_score := 8
  let effect_score := 8
  let image_weight := 3
  let content_weight := 4
  let effect_weight := 3
  let total_weight := image_weight + content_weight + effect_weight
  let weighted_score := (image_score * image_weight) + (content_score * content_weight) + (effect_score * effect_weight)
  weighted_score / total_weight = 8.3

theorem xiao_ming_final_score_correct : xiao_ming_final_score := by
  sorry

end NUMINAMATH_GPT_xiao_ming_final_score_correct_l808_80890


namespace NUMINAMATH_GPT_lightest_pumpkin_weight_l808_80863

theorem lightest_pumpkin_weight 
  (A B C : ℕ)
  (h₁ : A + B = 12)
  (h₂ : B + C = 15)
  (h₃ : A + C = 13) :
  A = 5 :=
by
  sorry

end NUMINAMATH_GPT_lightest_pumpkin_weight_l808_80863


namespace NUMINAMATH_GPT_total_boxes_l808_80859

theorem total_boxes (w1 w2 : ℕ) (h1 : w1 = 400) (h2 : w1 = 2 * w2) : w1 + w2 = 600 := 
by
  sorry

end NUMINAMATH_GPT_total_boxes_l808_80859


namespace NUMINAMATH_GPT_first_number_in_proportion_is_60_l808_80850

theorem first_number_in_proportion_is_60 : 
  ∀ (x : ℝ), (x / 6 = 2 / 0.19999999999999998) → x = 60 :=
by
  intros x hx
  sorry

end NUMINAMATH_GPT_first_number_in_proportion_is_60_l808_80850


namespace NUMINAMATH_GPT_original_number_is_two_over_three_l808_80801

theorem original_number_is_two_over_three (x : ℚ) (h : 1 + 1/x = 5/2) : x = 2/3 :=
sorry

end NUMINAMATH_GPT_original_number_is_two_over_three_l808_80801


namespace NUMINAMATH_GPT_ben_paints_area_l808_80834

variable (allen_ratio : ℕ) (ben_ratio : ℕ) (total_area : ℕ)
variable (total_ratio : ℕ := allen_ratio + ben_ratio)
variable (part_size : ℕ := total_area / total_ratio)

theorem ben_paints_area 
  (h1 : allen_ratio = 2)
  (h2 : ben_ratio = 6)
  (h3 : total_area = 360) : 
  ben_ratio * part_size = 270 := sorry

end NUMINAMATH_GPT_ben_paints_area_l808_80834


namespace NUMINAMATH_GPT_bike_ride_time_good_l808_80885

theorem bike_ride_time_good (x : ℚ) :
  (20 * x + 12 * (8 - x) = 122) → x = 13 / 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_bike_ride_time_good_l808_80885


namespace NUMINAMATH_GPT_problem_statement_l808_80841

noncomputable def x : ℕ := 4
noncomputable def y : ℤ := 3  -- alternatively, we could define y as -3 and the equality would still hold

theorem problem_statement : x^2 + y^2 + x + 2023 = 2052 := by
  sorry  -- Proof goes here

end NUMINAMATH_GPT_problem_statement_l808_80841


namespace NUMINAMATH_GPT_find_angle_C_find_side_c_l808_80827

noncomputable section

-- Definitions and conditions for Part 1
def vectors_dot_product_sin_2C (A B C : ℝ) (m : ℝ × ℝ) (n : ℝ × ℝ) : Prop :=
  m = (Real.sin A, Real.cos A) ∧ n = (Real.cos B, Real.sin B) ∧ 
  ((m.1 * n.1 + m.2 * n.2) = Real.sin (2 * C))

def angles_of_triangle (A B C : ℝ) : Prop := 
  A + B + C = Real.pi

theorem find_angle_C (A B C : ℝ) (m n : ℝ × ℝ) :
  vectors_dot_product_sin_2C A B C m n → angles_of_triangle A B C → C = Real.pi / 3 :=
sorry

-- Definitions and conditions for Part 2
def sin_in_arithmetic_sequence (x y z : ℝ) : Prop :=
  x + z = 2 * y

def product_of_sides_cos_C (a b c : ℝ) (C : ℝ) : Prop :=
  (a * b * Real.cos C = 18) ∧ (Real.cos C = 1 / 2)

theorem find_side_c (A B C a b c : ℝ) (m n : ℝ × ℝ) :
  sin_in_arithmetic_sequence (Real.sin A) (Real.sin C) (Real.sin B) → 
  angles_of_triangle A B C → 
  product_of_sides_cos_C a b c C → 
  C = Real.pi / 3 → 
  c = 6 :=
sorry

end NUMINAMATH_GPT_find_angle_C_find_side_c_l808_80827


namespace NUMINAMATH_GPT_images_per_memory_card_l808_80823

-- Define the constants based on the conditions given in the problem
def daily_pictures : ℕ := 10
def years : ℕ := 3
def days_per_year : ℕ := 365
def cost_per_card : ℕ := 60
def total_spent : ℕ := 13140

-- Define the properties to be proved
theorem images_per_memory_card :
  (years * days_per_year * daily_pictures) / (total_spent / cost_per_card) = 50 :=
by
  sorry

end NUMINAMATH_GPT_images_per_memory_card_l808_80823


namespace NUMINAMATH_GPT_chord_length_l808_80848

theorem chord_length
  (a b c A B C : ℝ)
  (h₁ : c * Real.sin C = 3 * a * Real.sin A + 3 * b * Real.sin B)
  (O : ℝ → ℝ → Prop)
  (hO : ∀ x y, O x y ↔ x^2 + y^2 = 12)
  (l : ℝ → ℝ → Prop)
  (hl : ∀ x y, l x y ↔ a * x - b * y + c = 0) :
  (2 * Real.sqrt ( (2 * Real.sqrt 3)^2 - (Real.sqrt 3)^2 )) = 6 :=
by
  sorry

end NUMINAMATH_GPT_chord_length_l808_80848


namespace NUMINAMATH_GPT_student_weight_loss_l808_80879

theorem student_weight_loss {S R L : ℕ} (h1 : S = 90) (h2 : S + R = 132) (h3 : S - L = 2 * R) : L = 6 := by
  sorry

end NUMINAMATH_GPT_student_weight_loss_l808_80879


namespace NUMINAMATH_GPT_min_ratio_area_of_incircle_circumcircle_rt_triangle_l808_80893

variables (a b: ℝ)
variables (a' b' c: ℝ)

-- Conditions
def area_of_right_triangle (a b : ℝ) : ℝ := 
    0.5 * a * b

def incircle_radius (a' b' c : ℝ) : ℝ := 
    0.5 * (a' + b' - c)

def circumcircle_radius (c : ℝ) : ℝ := 
    0.5 * c

-- Condition of the problem
def condition (a b a' b' c : ℝ) : Prop :=
    incircle_radius a' b' c = circumcircle_radius c ∧ 
    a' + b' = 2 * c

-- The final proof problem
theorem min_ratio_area_of_incircle_circumcircle_rt_triangle (a b a' b' c : ℝ)
    (h_area_a : a = area_of_right_triangle a' b')
    (h_area_b : b = area_of_right_triangle a b)
    (h_condition : condition a b a' b' c) :
    (a / b ≥ 3 + 2 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_min_ratio_area_of_incircle_circumcircle_rt_triangle_l808_80893


namespace NUMINAMATH_GPT_abs_neg_one_over_2023_l808_80860

theorem abs_neg_one_over_2023 : abs (-1 / 2023) = 1 / 2023 :=
by
  sorry

end NUMINAMATH_GPT_abs_neg_one_over_2023_l808_80860


namespace NUMINAMATH_GPT_part1_part2_l808_80829

section Problem

open Real

noncomputable def f (x : ℝ) := exp x
noncomputable def g (x : ℝ) := log x - 2

theorem part1 (x : ℝ) (hx : x > 0) : g x ≥ - (exp 1) / x :=
by sorry

theorem part2 (a : ℝ) (h : ∀ x, x ≥ 0 → f x - 1 / (f x) ≥ a * x) : a ≤ 2 :=
by sorry

end Problem

end NUMINAMATH_GPT_part1_part2_l808_80829


namespace NUMINAMATH_GPT_books_loaned_l808_80891

theorem books_loaned (L : ℕ)
  (initial_books : ℕ := 150)
  (end_year_books : ℕ := 100)
  (return_rate : ℝ := 0.60)
  (loan_rate : ℝ := 0.40)
  (returned_books : ℕ := (initial_books - end_year_books)) :
  loan_rate * (L : ℝ) = (returned_books : ℝ) → L = 125 := by
  intro h
  sorry

end NUMINAMATH_GPT_books_loaned_l808_80891


namespace NUMINAMATH_GPT_chicken_coop_problem_l808_80894

-- Definitions of conditions
def available_area : ℝ := 240
def area_per_chicken : ℝ := 4
def area_per_chick : ℝ := 2
def max_daily_feed : ℝ := 8000
def feed_per_chicken : ℝ := 160
def feed_per_chick : ℝ := 40

-- Variables representing the number of chickens and chicks
variables (x y : ℕ)

-- Condition expressions
def space_condition (x y : ℕ) : Prop := 
  (2 * x + y = (available_area / area_per_chick))

def feed_condition (x y : ℕ) : Prop := 
  ((4 * x + y) * feed_per_chick <= max_daily_feed / feed_per_chick)

-- Given conditions and queries proof problem
theorem chicken_coop_problem : 
  (∃ x y : ℕ, space_condition x y ∧ feed_condition x y ∧ (x = 20 ∧ y = 80)) 
  ∧
  (¬ ∃ x y : ℕ, space_condition x y ∧ feed_condition x y ∧ (x = 30 ∧ y = 100))
  ∧
  (∃ x y : ℕ, space_condition x y ∧ feed_condition x y ∧ (x = 40 ∧ y = 40))
  ∧
  (∃ x y : ℕ, space_condition x y ∧ feed_condition x y ∧ (x = 0 ∧ y = 120)) :=
by
  sorry  -- The proof will be provided here.


end NUMINAMATH_GPT_chicken_coop_problem_l808_80894


namespace NUMINAMATH_GPT_find_other_solution_l808_80846

theorem find_other_solution (x : ℚ) (hx : 45 * (2 / 5 : ℚ)^2 + 22 = 56 * (2 / 5 : ℚ) - 9) : x = 7 / 9 :=
by 
  sorry

end NUMINAMATH_GPT_find_other_solution_l808_80846


namespace NUMINAMATH_GPT_rhombus_area_l808_80811

theorem rhombus_area (d1 d2 : ℕ) (h1 : d1 = 18) (h2 : d2 = 14) : 
  (d1 * d2) / 2 = 126 := 
  by sorry

end NUMINAMATH_GPT_rhombus_area_l808_80811


namespace NUMINAMATH_GPT_sin_double_angle_ineq_l808_80877

theorem sin_double_angle_ineq (α : ℝ) (h1 : Real.sin (π - α) = 1 / 3) (h2 : π / 2 ≤ α) (h3 : α ≤ π) :
  Real.sin (2 * α) = - (4 * Real.sqrt 2) / 9 := 
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_ineq_l808_80877


namespace NUMINAMATH_GPT_students_neither_art_nor_music_l808_80867

def total_students := 75
def art_students := 45
def music_students := 50
def both_art_and_music := 30

theorem students_neither_art_nor_music : 
  total_students - (art_students - both_art_and_music + music_students - both_art_and_music + both_art_and_music) = 10 :=
by 
  sorry

end NUMINAMATH_GPT_students_neither_art_nor_music_l808_80867


namespace NUMINAMATH_GPT_vans_needed_l808_80873

-- Given Conditions
def van_capacity : ℕ := 4
def students : ℕ := 2
def adults : ℕ := 6
def total_people : ℕ := students + adults

-- Theorem to prove
theorem vans_needed : total_people / van_capacity = 2 :=
by
  -- Proof will be added here
  sorry

end NUMINAMATH_GPT_vans_needed_l808_80873


namespace NUMINAMATH_GPT_total_cost_of_plates_and_cups_l808_80803

theorem total_cost_of_plates_and_cups 
  (P C : ℝ)
  (h : 100 * P + 200 * C = 7.50) :
  20 * P + 40 * C = 1.50 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_plates_and_cups_l808_80803


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l808_80805

def M := {x : ℝ | abs x ≤ 2}
def N := {x : ℝ | x^2 - 3 * x = 0}

theorem intersection_of_M_and_N : M ∩ N = {0} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l808_80805


namespace NUMINAMATH_GPT_loss_percent_l808_80878

theorem loss_percent (CP SP : ℝ) (h₁ : CP = 600) (h₂ : SP = 300) : 
  (CP - SP) / CP * 100 = 50 :=
by
  rw [h₁, h₂]
  norm_num

end NUMINAMATH_GPT_loss_percent_l808_80878


namespace NUMINAMATH_GPT_retailer_profit_percentage_l808_80887

theorem retailer_profit_percentage
  (wholesale_price : ℝ)
  (retail_price : ℝ)
  (discount_rate : ℝ)
  (h_wholesale_price : wholesale_price = 108)
  (h_retail_price : retail_price = 144)
  (h_discount_rate : discount_rate = 0.10) :
  (retail_price * (1 - discount_rate) - wholesale_price) / wholesale_price * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_retailer_profit_percentage_l808_80887


namespace NUMINAMATH_GPT_total_monthly_bill_working_from_home_l808_80837

def original_monthly_bill : ℝ := 60
def percentage_increase : ℝ := 0.30

theorem total_monthly_bill_working_from_home :
  original_monthly_bill + (original_monthly_bill * percentage_increase) = 78 := by
  sorry

end NUMINAMATH_GPT_total_monthly_bill_working_from_home_l808_80837


namespace NUMINAMATH_GPT_find_x_l808_80844

-- Define the initial point A with coordinates A(x, -2)
def A (x : ℝ) : ℝ × ℝ := (x, -2)

-- Define the transformation of moving 5 units up and 3 units to the right to obtain point B
def transform (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 3, p.2 + 5)

-- Define the final point B with coordinates B(1, y)
def B (y : ℝ) : ℝ × ℝ := (1, y)

-- Define the proof problem
theorem find_x (x y : ℝ) (h : transform (A x) = B y) : x = -2 :=
by sorry

end NUMINAMATH_GPT_find_x_l808_80844


namespace NUMINAMATH_GPT_watermelon_cost_l808_80816

-- Define the problem conditions
def container_full_conditions (w m : ℕ) : Prop :=
  w + m = 150 ∧ (w / 160) + (m / 120) = 1

def equal_total_values (w m w_value m_value : ℕ) : Prop :=
  w * w_value = m * m_value ∧ w * w_value + m * m_value = 24000

-- Define the proof problem
theorem watermelon_cost (w m w_value m_value : ℕ) (hw : container_full_conditions w m) (hv : equal_total_values w m w_value m_value) :
  w_value = 100 :=
by
  -- precise proof goes here
  sorry

end NUMINAMATH_GPT_watermelon_cost_l808_80816


namespace NUMINAMATH_GPT_triangle_side_ineq_l808_80838

theorem triangle_side_ineq (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) : 
  a^2 * c + b^2 * a + c^2 * b < 1 / 8 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_side_ineq_l808_80838


namespace NUMINAMATH_GPT_sequence_geometric_proof_l808_80861

theorem sequence_geometric_proof (a : ℕ → ℕ) (h1 : a 1 = 5) (h2 : ∀ n, a (n + 1) = 2 * a n) :
  ∀ n, a n = 5 * 2 ^ (n - 1) :=
by
  sorry

end NUMINAMATH_GPT_sequence_geometric_proof_l808_80861


namespace NUMINAMATH_GPT_cake_remaining_after_4_trips_l808_80830

theorem cake_remaining_after_4_trips :
  ∀ (cake_portion_left_after_trip : ℕ → ℚ), 
    cake_portion_left_after_trip 0 = 1 ∧
    (∀ n, cake_portion_left_after_trip (n + 1) = cake_portion_left_after_trip n / 2) →
    cake_portion_left_after_trip 4 = 1 / 16 :=
by
  intros cake_portion_left_after_trip h
  have h0 : cake_portion_left_after_trip 0 = 1 := h.1
  have h1 : ∀ n, cake_portion_left_after_trip (n + 1) = cake_portion_left_after_trip n / 2 := h.2
  sorry

end NUMINAMATH_GPT_cake_remaining_after_4_trips_l808_80830


namespace NUMINAMATH_GPT_average_speed_of_car_l808_80812

noncomputable def avgSpeed (Distance_uphill Speed_uphill Distance_downhill Speed_downhill : ℝ) : ℝ :=
  let Time_uphill := Distance_uphill / Speed_uphill
  let Time_downhill := Distance_downhill / Speed_downhill
  let Total_time := Time_uphill + Time_downhill
  let Total_distance := Distance_uphill + Distance_downhill
  Total_distance / Total_time

theorem average_speed_of_car:
  avgSpeed 100 30 50 60 = 36 := by
  sorry

end NUMINAMATH_GPT_average_speed_of_car_l808_80812


namespace NUMINAMATH_GPT_kanul_cash_spending_percentage_l808_80880

theorem kanul_cash_spending_percentage :
  ∀ (spent_raw_materials spent_machinery total_amount spent_cash : ℝ),
    spent_raw_materials = 500 →
    spent_machinery = 400 →
    total_amount = 1000 →
    spent_cash = total_amount - (spent_raw_materials + spent_machinery) →
    (spent_cash / total_amount) * 100 = 10 :=
by
  intros spent_raw_materials spent_machinery total_amount spent_cash
  intro h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_kanul_cash_spending_percentage_l808_80880


namespace NUMINAMATH_GPT_compute_product_l808_80828

variables (x1 y1 x2 y2 x3 y3 : ℝ)

def condition1 (x y : ℝ) : Prop :=
  x^3 - 3 * x * y^2 = 2017

def condition2 (x y : ℝ) : Prop :=
  y^3 - 3 * x^2 * y = 2016

theorem compute_product :
  condition1 x1 y1 → condition2 x1 y1 →
  condition1 x2 y2 → condition2 x2 y2 →
  condition1 x3 y3 → condition2 x3 y3 →
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = 1 / 1008 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_compute_product_l808_80828


namespace NUMINAMATH_GPT_initial_avg_income_l808_80884

theorem initial_avg_income (A : ℝ) :
  (4 * A - 990 = 3 * 650) → (A = 735) :=
by
  sorry

end NUMINAMATH_GPT_initial_avg_income_l808_80884


namespace NUMINAMATH_GPT_solve_for_a_l808_80864

theorem solve_for_a (a : ℕ) (h : a^3 = 21 * 25 * 35 * 63) : a = 105 :=
sorry

end NUMINAMATH_GPT_solve_for_a_l808_80864


namespace NUMINAMATH_GPT_books_sold_on_friday_l808_80800

theorem books_sold_on_friday
  (total_books : ℕ)
  (books_sold_mon : ℕ)
  (books_sold_tue : ℕ)
  (books_sold_wed : ℕ)
  (books_sold_thu : ℕ)
  (pct_unsold : ℚ)
  (initial_stock : total_books = 1400)
  (sold_mon : books_sold_mon = 62)
  (sold_tue : books_sold_tue = 62)
  (sold_wed : books_sold_wed = 60)
  (sold_thu : books_sold_thu = 48)
  (percentage_unsold : pct_unsold = 0.8057142857142857) :
  total_books - (books_sold_mon + books_sold_tue + books_sold_wed + books_sold_thu + 40) = total_books * pct_unsold :=
by
  sorry

end NUMINAMATH_GPT_books_sold_on_friday_l808_80800


namespace NUMINAMATH_GPT_value_of_ab_l808_80874

theorem value_of_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 25) : ab = 8 :=
by
  sorry

end NUMINAMATH_GPT_value_of_ab_l808_80874


namespace NUMINAMATH_GPT_trigonometric_identity_l808_80822

theorem trigonometric_identity :
  let cos60 := (1 / 2)
  let sin30 := (1 / 2)
  let tan45 := (1 : ℝ)
  4 * cos60 + 8 * sin30 - 5 * tan45 = 1 :=
by
  let cos60 := (1 / 2 : ℝ)
  let sin30 := (1 / 2 : ℝ)
  let tan45 := (1 : ℝ)
  show 4 * cos60 + 8 * sin30 - 5 * tan45 = 1
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l808_80822


namespace NUMINAMATH_GPT_cost_of_each_television_l808_80881

-- Define the conditions
def number_of_televisions : Nat := 5
def number_of_figurines : Nat := 10
def cost_per_figurine : Nat := 1
def total_spent : Nat := 260

-- Define the proof problem
theorem cost_of_each_television (T : Nat) :
  (number_of_televisions * T + number_of_figurines * cost_per_figurine = total_spent) → (T = 50) :=
by
  sorry

end NUMINAMATH_GPT_cost_of_each_television_l808_80881


namespace NUMINAMATH_GPT_find_unique_digit_sets_l808_80832

theorem find_unique_digit_sets (a b c : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c)
 (h4 : 22 * (a + b + c) = 462) :
  (a = 4 ∧ b = 8 ∧ c = 9) ∨ 
  (a = 4 ∧ b = 9 ∧ c = 8) ∨ 
  (a = 8 ∧ b = 4 ∧ c = 9) ∨
  (a = 8 ∧ b = 9 ∧ c = 4) ∨ 
  (a = 9 ∧ b = 4 ∧ c = 8) ∨ 
  (a = 9 ∧ b = 8 ∧ c = 4) ∨
  (a = 5 ∧ b = 7 ∧ c = 9) ∨ 
  (a = 5 ∧ b = 9 ∧ c = 7) ∨ 
  (a = 7 ∧ b = 5 ∧ c = 9) ∨
  (a = 7 ∧ b = 9 ∧ c = 5) ∨ 
  (a = 9 ∧ b = 5 ∧ c = 7) ∨ 
  (a = 9 ∧ b = 7 ∧ c = 5) ∨
  (a = 6 ∧ b = 7 ∧ c = 8) ∨ 
  (a = 6 ∧ b = 8 ∧ c = 7) ∨ 
  (a = 7 ∧ b = 6 ∧ c = 8) ∨
  (a = 7 ∧ b = 8 ∧ c = 6) ∨ 
  (a = 8 ∧ b = 6 ∧ c = 7) ∨ 
  (a = 8 ∧ b = 7 ∧ c = 6) :=
sorry

end NUMINAMATH_GPT_find_unique_digit_sets_l808_80832


namespace NUMINAMATH_GPT_anne_cleaning_time_l808_80843

theorem anne_cleaning_time (B A : ℝ)
  (h1 : 4 * (B + A) = 1)
  (h2 : 3 * (B + 2 * A) = 1) : 1 / A = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_anne_cleaning_time_l808_80843


namespace NUMINAMATH_GPT_shaded_region_area_l808_80813

noncomputable def area_of_shaded_region (r : ℝ) (oa : ℝ) (ab_length : ℝ) : ℝ :=
  18 * (Real.sqrt 2) - 9 - (9 * Real.pi / 4)

theorem shaded_region_area (r : ℝ) (oa : ℝ) (ab_length : ℝ) : 
  r = 3 ∧ oa = 3 * Real.sqrt 2 ∧ ab_length = 6 * Real.sqrt 2 → 
  area_of_shaded_region r oa ab_length = 18 * (Real.sqrt 2) - 9 - (9 * Real.pi / 4) :=
by
  intro h
  obtain ⟨hr, hoa, hab⟩ := h
  rw [hr, hoa, hab]
  exact rfl

end NUMINAMATH_GPT_shaded_region_area_l808_80813


namespace NUMINAMATH_GPT_solve_system_l808_80892

variable (x y z : ℝ)

def equation1 : Prop := x^2 + 25 * y + 19 * z = -471
def equation2 : Prop := y^2 + 23 * x + 21 * z = -397
def equation3 : Prop := z^2 + 21 * x + 21 * y = -545

theorem solve_system : equation1 (-22) (-23) (-20) ∧ equation2 (-22) (-23) (-20) ∧ equation3 (-22) (-23) (-20) := by
  sorry

end NUMINAMATH_GPT_solve_system_l808_80892


namespace NUMINAMATH_GPT_Jovana_final_addition_l808_80817

theorem Jovana_final_addition 
  (initial_amount added_initial removed final_amount x : ℕ)
  (h1 : initial_amount = 5)
  (h2 : added_initial = 9)
  (h3 : removed = 2)
  (h4 : final_amount = 28) :
  final_amount = initial_amount + added_initial - removed + x → x = 16 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_Jovana_final_addition_l808_80817


namespace NUMINAMATH_GPT_validate_assignment_l808_80865

-- Define the statements as conditions
def S1 := "x = x + 1"
def S2 := "b ="
def S3 := "x = y = 10"
def S4 := "x + y = 10"

-- A function to check if a statement is a valid assignment
def is_valid_assignment (s : String) : Prop :=
  s = S1

-- The theorem statement proving that S1 is the only valid assignment
theorem validate_assignment : is_valid_assignment S1 ∧
                              ¬is_valid_assignment S2 ∧
                              ¬is_valid_assignment S3 ∧
                              ¬is_valid_assignment S4 :=
by
  sorry

end NUMINAMATH_GPT_validate_assignment_l808_80865


namespace NUMINAMATH_GPT_factorize_expr_l808_80826

theorem factorize_expr (a b : ℝ) : a^2 - 2 * a * b = a * (a - 2 * b) := 
by 
  sorry

end NUMINAMATH_GPT_factorize_expr_l808_80826


namespace NUMINAMATH_GPT_positive_difference_l808_80858

theorem positive_difference:
  let a := (7^3 + 7^3) / 7
  let b := (7^3)^2 / 7
  b - a = 16709 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_l808_80858


namespace NUMINAMATH_GPT_last_donation_on_saturday_l808_80806

def total_amount : ℕ := 2010
def daily_donation : ℕ := 10
def first_day_donation : ℕ := 0 -- where 0 represents Monday, 6 represents Sunday

def total_days : ℕ := total_amount / daily_donation

def last_donation_day_of_week : ℕ := (total_days % 7 + first_day_donation) % 7

theorem last_donation_on_saturday : last_donation_day_of_week = 5 := by
  -- Prove it by calculation
  sorry

end NUMINAMATH_GPT_last_donation_on_saturday_l808_80806


namespace NUMINAMATH_GPT_carolyn_fewer_stickers_l808_80886

theorem carolyn_fewer_stickers :
  let belle_stickers := 97
  let carolyn_stickers := 79
  carolyn_stickers < belle_stickers →
  belle_stickers - carolyn_stickers = 18 :=
by
  intros
  sorry

end NUMINAMATH_GPT_carolyn_fewer_stickers_l808_80886


namespace NUMINAMATH_GPT_cone_volume_l808_80818

theorem cone_volume (d h : ℝ) (V : ℝ) (hd : d = 12) (hh : h = 8) :
  V = (1 / 3) * Real.pi * (d / 2) ^ 2 * h → V = 96 * Real.pi :=
by
  rw [hd, hh]
  sorry

end NUMINAMATH_GPT_cone_volume_l808_80818


namespace NUMINAMATH_GPT_julias_total_spending_l808_80814

def adoption_fee : ℝ := 20.00
def dog_food_cost : ℝ := 20.00
def treat_cost_per_bag : ℝ := 2.50
def num_treat_bags : ℝ := 2
def toy_box_cost : ℝ := 15.00
def crate_cost : ℝ := 20.00
def bed_cost : ℝ := 20.00
def collar_leash_cost : ℝ := 15.00
def discount_rate : ℝ := 0.20

def total_items_cost : ℝ :=
  dog_food_cost + (treat_cost_per_bag * num_treat_bags) + toy_box_cost +
  crate_cost + bed_cost + collar_leash_cost

def discount_amount : ℝ := total_items_cost * discount_rate
def discounted_items_cost : ℝ := total_items_cost - discount_amount
def total_expenditure : ℝ := adoption_fee + discounted_items_cost

theorem julias_total_spending :
  total_expenditure = 96.00 := by
  sorry

end NUMINAMATH_GPT_julias_total_spending_l808_80814


namespace NUMINAMATH_GPT_number_exceeds_fraction_l808_80883

theorem number_exceeds_fraction (x : ℝ) (hx : x = 0.45 * x + 1000) : x = 1818.18 := 
by
  sorry

end NUMINAMATH_GPT_number_exceeds_fraction_l808_80883


namespace NUMINAMATH_GPT_boxes_contain_fruits_l808_80856

-- Define the weights of the boxes
def box_weights : List ℕ := [15, 16, 18, 19, 20, 31]

-- Define the weight requirement for apples and pears
def weight_rel (apple_weight pear_weight : ℕ) : Prop := apple_weight = pear_weight / 2

-- Define the statement with the constraints, given conditions and assignments.
theorem boxes_contain_fruits (h1 : box_weights = [15, 16, 18, 19, 20, 31])
                             (h2 : ∃ apple_weight pear_weight, 
                                   weight_rel apple_weight pear_weight ∧ 
                                   pear_weight ∈ box_weights ∧ apple_weight ∈ box_weights)
                             (h3 : ∃ orange_weight, orange_weight ∈ box_weights ∧ 
                                   ∀ w, w ∈ box_weights → w ≠ orange_weight)
                             : (15 = 2 ∧ 19 = 3 ∧ 20 = 1 ∧ 31 = 3) := 
                             sorry

end NUMINAMATH_GPT_boxes_contain_fruits_l808_80856


namespace NUMINAMATH_GPT_club_population_after_five_years_l808_80855

noncomputable def a : ℕ → ℕ
| 0     => 18
| (n+1) => 3 * (a n - 5) + 5

theorem club_population_after_five_years : a 5 = 3164 := by
  sorry

end NUMINAMATH_GPT_club_population_after_five_years_l808_80855


namespace NUMINAMATH_GPT_stratified_sampling_l808_80862

theorem stratified_sampling (teachers male_students female_students total_pop sample_female_students proportion_total n : ℕ)
    (h_teachers : teachers = 200)
    (h_male_students : male_students = 1200)
    (h_female_students : female_students = 1000)
    (h_total_pop : total_pop = teachers + male_students + female_students)
    (h_sample_female_students : sample_female_students = 80)
    (h_proportion_total : proportion_total = female_students / total_pop)
    (h_proportion_equation : sample_female_students = proportion_total * n) :
  n = 192 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_l808_80862


namespace NUMINAMATH_GPT_sophomores_selected_l808_80870

variables (total_students freshmen sophomores juniors selected_students : ℕ)
def high_school_data := total_students = 2800 ∧ freshmen = 970 ∧ sophomores = 930 ∧ juniors = 900 ∧ selected_students = 280

theorem sophomores_selected (h : high_school_data total_students freshmen sophomores juniors selected_students) :
  (930 / 2800 : ℚ) * 280 = 93 := by
  sorry

end NUMINAMATH_GPT_sophomores_selected_l808_80870


namespace NUMINAMATH_GPT_smallest_nonprime_greater_than_with_no_prime_factors_less_than_15_l808_80896

-- Definitions for the given conditions.
def is_prime (p : ℕ) : Prop := (p > 1) ∧ ∀ d : ℕ, d ∣ p → (d = 1 ∨ d = p)

def has_no_prime_factors_less_than (n : ℕ) (m : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ m

def is_nonprime (n : ℕ) : Prop := ¬ is_prime n

-- The main theorem based on the proof problem.
theorem smallest_nonprime_greater_than_with_no_prime_factors_less_than_15 
  (n : ℕ) (h1 : n > 1) (h2 : has_no_prime_factors_less_than n 15) (h3 : is_nonprime n) : 
  280 < n ∧ n ≤ 290 :=
by
  sorry

end NUMINAMATH_GPT_smallest_nonprime_greater_than_with_no_prime_factors_less_than_15_l808_80896


namespace NUMINAMATH_GPT_find_b_minus_c_l808_80876

variable (a b c: ℤ)

theorem find_b_minus_c (h1: a - b - c = 1) (h2: a - (b - c) = 13) (h3: (b - c) - a = -9) : b - c = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_b_minus_c_l808_80876


namespace NUMINAMATH_GPT_last_two_digits_28_l808_80849

theorem last_two_digits_28 (n : ℕ) (h1 : n > 0) (h2 : n % 2 = 1) : 
  (2^(2*n) * (2^(2*n+1) - 1)) % 100 = 28 :=
by
  sorry

end NUMINAMATH_GPT_last_two_digits_28_l808_80849


namespace NUMINAMATH_GPT_bees_lost_each_day_l808_80808

theorem bees_lost_each_day
    (initial_bees : ℕ)
    (daily_hatch : ℕ)
    (days : ℕ)
    (total_bees_after_days : ℕ)
    (bees_lost_each_day : ℕ) :
    initial_bees = 12500 →
    daily_hatch = 3000 →
    days = 7 →
    total_bees_after_days = 27201 →
    (initial_bees + days * (daily_hatch - bees_lost_each_day) = total_bees_after_days) →
    bees_lost_each_day = 899 :=
by
  intros h_initial h_hatch h_days h_total h_eq
  sorry

end NUMINAMATH_GPT_bees_lost_each_day_l808_80808


namespace NUMINAMATH_GPT_universal_inequality_l808_80824

theorem universal_inequality (x y : ℝ) : x^2 + y^2 ≥ 2 * x * y := 
by 
  sorry

end NUMINAMATH_GPT_universal_inequality_l808_80824


namespace NUMINAMATH_GPT_sin_maximum_value_l808_80871

theorem sin_maximum_value (c : ℝ) :
  (∀ x : ℝ, x = -π/4 → 3 * Real.sin (2 * x + c) = 3) → c = π :=
by
 sorry

end NUMINAMATH_GPT_sin_maximum_value_l808_80871


namespace NUMINAMATH_GPT_find_d_l808_80809

noncomputable def median (x : ℕ) : ℕ := x + 4
noncomputable def mean (x d : ℕ) : ℕ := x + (13 + d) / 5

theorem find_d (x d : ℕ) (h : mean x d = median x + 5) : d = 32 := by
  sorry

end NUMINAMATH_GPT_find_d_l808_80809


namespace NUMINAMATH_GPT_geometric_sum_of_first_five_terms_l808_80868

theorem geometric_sum_of_first_five_terms (a_1 l : ℝ)
  (h₁ : ∀ r : ℝ, (2 * l = a_1 * (r - 1) ^ 2)) 
  (h₂ : ∀ (r : ℝ), a_1 * r ^ 3 = 8 * a_1):
  (a_1 + a_1 * (2 : ℝ) + a_1 * (2 : ℝ)^2 + a_1 * (2 : ℝ)^3 + a_1 * (2 : ℝ)^4) = 62 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sum_of_first_five_terms_l808_80868


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l808_80835

theorem sufficient_but_not_necessary_condition (h1 : 1^2 - 1 = 0) (h2 : ∀ x, x^2 - 1 = 0 → (x = 1 ∨ x = -1)) :
  (∀ x, x = 1 → x^2 - 1 = 0) ∧ ¬ (∀ x, x^2 - 1 = 0 → x = 1) := by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l808_80835


namespace NUMINAMATH_GPT_max_pairs_of_corner_and_squares_l808_80875

def rectangle : ℕ := 3 * 100
def unit_squares_per_pair : ℕ := 4 + 3

-- Given conditions
def conditions := rectangle = 300 ∧ unit_squares_per_pair = 7

-- Proof statement
theorem max_pairs_of_corner_and_squares (h: conditions) : ∃ n, n = 33 ∧ n * unit_squares_per_pair ≤ rectangle := 
sorry

end NUMINAMATH_GPT_max_pairs_of_corner_and_squares_l808_80875


namespace NUMINAMATH_GPT_riley_pawns_lost_l808_80889

theorem riley_pawns_lost (initial_pawns : ℕ) (kennedy_lost : ℕ) (total_pawns_left : ℕ)
  (kennedy_initial_pawns : ℕ) (riley_initial_pawns : ℕ) : 
  kennedy_initial_pawns = initial_pawns ∧
  riley_initial_pawns = initial_pawns ∧
  kennedy_lost = 4 ∧
  total_pawns_left = 11 →
  riley_initial_pawns - (total_pawns_left - (kennedy_initial_pawns - kennedy_lost)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_riley_pawns_lost_l808_80889


namespace NUMINAMATH_GPT_simplify_expression_l808_80833

theorem simplify_expression (x : ℝ) : 
  (x^3 * x^2 * x + (x^3)^2 + (-2 * x^2)^3) = -6 * x^6 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l808_80833


namespace NUMINAMATH_GPT_algebraic_expression_value_l808_80897

theorem algebraic_expression_value
  (a b x y : ℤ)
  (h1 : x = a)
  (h2 : y = b)
  (h3 : x - 2 * y = 7) :
  -a + 2 * b + 1 = -6 :=
by
  -- the proof steps are omitted as instructed
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l808_80897
