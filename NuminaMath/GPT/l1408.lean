import Mathlib

namespace NUMINAMATH_GPT_recipe_flour_amount_l1408_140846

theorem recipe_flour_amount
  (cups_of_sugar : ℕ) (cups_of_salt : ℕ) (cups_of_flour_added : ℕ)
  (additional_cups_of_flour : ℕ)
  (h1 : cups_of_sugar = 2)
  (h2 : cups_of_salt = 80)
  (h3 : cups_of_flour_added = 7)
  (h4 : additional_cups_of_flour = cups_of_sugar + 1) :
  cups_of_flour_added + additional_cups_of_flour = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_recipe_flour_amount_l1408_140846


namespace NUMINAMATH_GPT_solve_equation_l1408_140853

theorem solve_equation : ∃ x : ℝ, (1 + x) / (2 - x) - 1 = 1 / (x - 2) ↔ x = 0 := 
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1408_140853


namespace NUMINAMATH_GPT_range_of_a_l1408_140822

noncomputable def f (x : ℝ) : ℝ := Real.exp (x - 1) + x - 2

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x - a + 3

theorem range_of_a :
  (∃ x1 x2 : ℝ, f x1 = 0 ∧ g x2 a = 0 ∧ |x1 - x2| ≤ 1) ↔ (a ∈ Set.Icc 2 3) := sorry

end NUMINAMATH_GPT_range_of_a_l1408_140822


namespace NUMINAMATH_GPT_wine_cost_today_l1408_140870

theorem wine_cost_today (C : ℝ) (h1 : ∀ (new_tariff : ℝ), new_tariff = 0.25) (h2 : ∀ (total_increase : ℝ), total_increase = 25) (h3 : C = 20) : 5 * (1.25 * C - C) = 25 :=
by
  sorry

end NUMINAMATH_GPT_wine_cost_today_l1408_140870


namespace NUMINAMATH_GPT_constant_term_expanded_eq_neg12_l1408_140884

theorem constant_term_expanded_eq_neg12
  (a w c d : ℤ)
  (h_eq : (a * x + w) * (c * x + d) = 6 * x ^ 2 + x - 12)
  (h_abs_sum : abs a + abs w + abs c + abs d = 12) :
  w * d = -12 := by
  sorry

end NUMINAMATH_GPT_constant_term_expanded_eq_neg12_l1408_140884


namespace NUMINAMATH_GPT_beef_weight_after_processing_l1408_140803

def original_weight : ℝ := 861.54
def weight_loss_percentage : ℝ := 0.35
def retained_percentage : ℝ := 1 - weight_loss_percentage
def weight_after_processing (w : ℝ) := retained_percentage * w

theorem beef_weight_after_processing :
  weight_after_processing original_weight = 560.001 :=
by
  sorry

end NUMINAMATH_GPT_beef_weight_after_processing_l1408_140803


namespace NUMINAMATH_GPT_simplify_expression_l1408_140858

theorem simplify_expression (a b : ℝ) (h : a ≠ b) : 
  ((a^3 - b^3) / (a * b)) - ((a * b^2 - b^3) / (a * b - a^3)) = (2 * a * (a - b)) / b :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1408_140858


namespace NUMINAMATH_GPT_characterization_of_points_l1408_140810

def satisfies_eq (x : ℝ) (y : ℝ) : Prop :=
  max x (x^2) + min y (y^2) = 1

theorem characterization_of_points :
  ∀ x y : ℝ,
  satisfies_eq x y ↔
  ((x < 0 ∨ x > 1) ∧ (y < 0 ∨ y > 1) ∧ y ≤ 0 ∧ y = 1 - x^2) ∨
  ((x < 0 ∨ x > 1) ∧ (0 < y ∧ y < 1) ∧ x^2 + y^2 = 1 ∧ x ≤ -1 ∧ x > 0) ∨
  ((0 < x ∧ x < 1) ∧ (y < 0 ∨ y > 1) ∧ false) ∨
  ((0 < x ∧ x < 1) ∧ (0 < y ∧ y < 1) ∧ y^2 = 1 - x) :=
sorry

end NUMINAMATH_GPT_characterization_of_points_l1408_140810


namespace NUMINAMATH_GPT_find_savings_l1408_140859

-- Definitions and conditions from the problem
def income : ℕ := 36000
def ratio_income_to_expenditure : ℚ := 9 / 8
def expenditure : ℚ := 36000 * (8 / 9)
def savings : ℚ := income - expenditure

-- The theorem statement to prove
theorem find_savings : savings = 4000 := by
  sorry

end NUMINAMATH_GPT_find_savings_l1408_140859


namespace NUMINAMATH_GPT_jodi_third_week_miles_l1408_140806

theorem jodi_third_week_miles (total_miles : ℕ) (first_week : ℕ) (second_week : ℕ) (fourth_week : ℕ) (days_per_week : ℕ) (third_week_miles_per_day : ℕ) 
  (H1 : first_week * days_per_week + second_week * days_per_week + third_week_miles_per_day * days_per_week + fourth_week * days_per_week = total_miles)
  (H2 : first_week = 1) 
  (H3 : second_week = 2) 
  (H4 : fourth_week = 4)
  (H5 : total_miles = 60)
  (H6 : days_per_week = 6) :
    third_week_miles_per_day = 3 :=
by sorry

end NUMINAMATH_GPT_jodi_third_week_miles_l1408_140806


namespace NUMINAMATH_GPT_even_function_f_l1408_140839

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem even_function_f (hx : ∀ x : ℝ, f (-x) = f x) 
  (hg : ∀ x : ℝ, g (-x) = -g x)
  (h_pass : g (-1) = 1)
  (hg_eq_f : ∀ x : ℝ, g x = f (x - 1)) 
  : f 7 + f 8 = -1 := 
by
  sorry

end NUMINAMATH_GPT_even_function_f_l1408_140839


namespace NUMINAMATH_GPT_sandy_spent_percentage_l1408_140813

theorem sandy_spent_percentage (I R : ℝ) (hI : I = 200) (hR : R = 140) : 
  ((I - R) / I) * 100 = 30 :=
by
  sorry

end NUMINAMATH_GPT_sandy_spent_percentage_l1408_140813


namespace NUMINAMATH_GPT_train_B_time_to_destination_l1408_140866

-- Definitions (conditions)
def speed_train_A := 60  -- Train A travels at 60 kmph
def speed_train_B := 90  -- Train B travels at 90 kmph
def time_train_A_after_meeting := 9 -- Train A takes 9 hours after meeting train B

-- Theorem statement
theorem train_B_time_to_destination 
  (speed_A : ℝ)
  (speed_B : ℝ)
  (time_A_after_meeting : ℝ)
  (time_B_to_destination : ℝ) :
  speed_A = speed_train_A ∧
  speed_B = speed_train_B ∧
  time_A_after_meeting = time_train_A_after_meeting →
  time_B_to_destination = 4.5 :=
by
  sorry

end NUMINAMATH_GPT_train_B_time_to_destination_l1408_140866


namespace NUMINAMATH_GPT_sum_of_roots_is_zero_l1408_140886

-- Definitions
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Problem Statement
theorem sum_of_roots_is_zero (f : ℝ → ℝ) (h_even : is_even f) (h_intersects : ∃ x1 x2 x3 x4 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ f x4 = 0) : 
  x1 + x2 + x3 + x4 = 0 :=
by 
  sorry -- Proof can be provided here

end NUMINAMATH_GPT_sum_of_roots_is_zero_l1408_140886


namespace NUMINAMATH_GPT_water_height_in_tank_l1408_140823

noncomputable def cone_radius := 10 -- in cm
noncomputable def cone_height := 15 -- in cm
noncomputable def tank_width := 20 -- in cm
noncomputable def tank_length := 30 -- in cm
noncomputable def cone_volume := (1/3:ℝ) * Real.pi * (cone_radius^2) * cone_height
noncomputable def tank_volume (h:ℝ) := tank_width * tank_length * h

theorem water_height_in_tank : ∃ h : ℝ, tank_volume h = cone_volume ∧ h = 5 * Real.pi / 6 := 
by 
  sorry

end NUMINAMATH_GPT_water_height_in_tank_l1408_140823


namespace NUMINAMATH_GPT_remainder_of_fractions_l1408_140809

theorem remainder_of_fractions : 
  ∀ (x y : ℚ), x = 5/7 → y = 3/4 → (x - y * ⌊x / y⌋) = 5/7 :=
by
  intros x y hx hy
  rw [hx, hy]
  -- Additional steps can be filled in here, if continuing with the proof.
  sorry

end NUMINAMATH_GPT_remainder_of_fractions_l1408_140809


namespace NUMINAMATH_GPT_bianca_points_l1408_140855

theorem bianca_points : 
  let a := 5; let b := 8; let c := 10;
  let A1 := 10; let P1 := 5; let G1 := 5;
  let A2 := 3; let P2 := 2; let G2 := 1;
  (A1 * a - A2 * a) + (P1 * b - P2 * b) + (G1 * c - G2 * c) = 99 := 
by
  sorry

end NUMINAMATH_GPT_bianca_points_l1408_140855


namespace NUMINAMATH_GPT_tan_of_angle_in_third_quadrant_l1408_140830

theorem tan_of_angle_in_third_quadrant (α : ℝ) (h1 : Real.sin α = -12 / 13) (h2 : π < α ∧ α < 3 * π / 2) : Real.tan α = 12 / 5 := 
sorry

end NUMINAMATH_GPT_tan_of_angle_in_third_quadrant_l1408_140830


namespace NUMINAMATH_GPT_problem_solution_l1408_140827

def p : Prop := ∀ x : ℝ, |x| ≥ 0
def q : Prop := ∃ x : ℝ, x = 2 ∧ x + 2 = 0

theorem problem_solution : p ∧ ¬q :=
by
  -- Here we would provide the proof to show that p ∧ ¬q is true
  sorry

end NUMINAMATH_GPT_problem_solution_l1408_140827


namespace NUMINAMATH_GPT_max_distance_on_curve_and_ellipse_l1408_140857

noncomputable def max_distance_between_P_and_Q : ℝ :=
  6 * Real.sqrt 2

theorem max_distance_on_curve_and_ellipse :
  ∃ P Q, (P ∈ { p : ℝ × ℝ | p.1^2 + (p.2 - 6)^2 = 2 }) ∧ 
         (Q ∈ { q : ℝ × ℝ | q.1^2 / 10 + q.2^2 = 1 }) ∧ 
         (dist P Q = max_distance_between_P_and_Q) := 
sorry

end NUMINAMATH_GPT_max_distance_on_curve_and_ellipse_l1408_140857


namespace NUMINAMATH_GPT_pascal_triangle_ratio_l1408_140860

theorem pascal_triangle_ratio (n r : ℕ) :
  (r + 1 = (4 * (n - r)) / 5) ∧ (r + 2 = (5 * (n - r - 1)) / 6) → n = 53 :=
by sorry

end NUMINAMATH_GPT_pascal_triangle_ratio_l1408_140860


namespace NUMINAMATH_GPT_factor_polynomial_int_l1408_140898

theorem factor_polynomial_int : 
  ∀ x : ℤ, 5 * (x + 3) * (x + 7) * (x + 9) * (x + 11) - 4 * x^2 = 
           (5 * x^2 + 81 * x + 315) * (x^2 + 16 * x + 213) := 
by
  intros
  norm_num
  sorry

end NUMINAMATH_GPT_factor_polynomial_int_l1408_140898


namespace NUMINAMATH_GPT_triangle_equilateral_if_arithmetic_sequences_l1408_140890

theorem triangle_equilateral_if_arithmetic_sequences
  (A B C : ℝ) (a b c : ℝ)
  (h_angles : A + B + C = 180)
  (h_angle_seq : ∃ (N : ℝ), A = B - N ∧ C = B + N)
  (h_sides : ∃ (n : ℝ), a = b - n ∧ c = b + n) :
  A = B ∧ B = C ∧ a = b ∧ b = c :=
sorry

end NUMINAMATH_GPT_triangle_equilateral_if_arithmetic_sequences_l1408_140890


namespace NUMINAMATH_GPT_remainder_when_divided_by_5_l1408_140893

theorem remainder_when_divided_by_5 (n : ℕ) (h1 : n^2 % 5 = 1) (h2 : n^3 % 5 = 4) : n % 5 = 4 :=
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_5_l1408_140893


namespace NUMINAMATH_GPT_A_time_to_cover_distance_is_45_over_y_l1408_140856

variable (y : ℝ)
variable (h0 : y > 0)
variable (h1 : (45 : ℝ) / (y - 2 / 3) - (45 : ℝ) / y = 3 / 4)

theorem A_time_to_cover_distance_is_45_over_y :
  45 / y = 45 / y :=
by
  sorry

end NUMINAMATH_GPT_A_time_to_cover_distance_is_45_over_y_l1408_140856


namespace NUMINAMATH_GPT_range_of_a_l1408_140820

theorem range_of_a (f : ℝ → ℝ) (h_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2) (h_ineq : f (1 - a) < f (2 * a - 1)) : a < 2 / 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1408_140820


namespace NUMINAMATH_GPT_product_x_z_l1408_140841

-- Defining the variables x, y, z as positive integers and the given conditions.
theorem product_x_z (x y z : ℕ) (h1 : x = 4 * y) (h2 : z = 2 * x) (h3 : x + y + z = 3 * y ^ 2) : 
    x * z = 5408 / 9 := 
  sorry

end NUMINAMATH_GPT_product_x_z_l1408_140841


namespace NUMINAMATH_GPT_diagonals_in_convex_polygon_l1408_140807

-- Define the number of sides for the polygon
def polygon_sides : ℕ := 15

-- The main theorem stating the number of diagonals in a convex polygon with 15 sides
theorem diagonals_in_convex_polygon : polygon_sides = 15 → ∃ d : ℕ, d = 90 :=
by
  intro h
  -- sorry is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_diagonals_in_convex_polygon_l1408_140807


namespace NUMINAMATH_GPT_general_formula_arithmetic_sequence_l1408_140880

def f (x : ℝ) : ℝ := x^2 - 4*x + 2

theorem general_formula_arithmetic_sequence (x : ℝ) (a : ℕ → ℝ) 
  (h1 : a 1 = f (x + 1))
  (h2 : a 2 = 0)
  (h3 : a 3 = f (x - 1)) :
  ∀ n : ℕ, (a n = 2 * n - 4) ∨ (a n = 4 - 2 * n) :=
by
  sorry

end NUMINAMATH_GPT_general_formula_arithmetic_sequence_l1408_140880


namespace NUMINAMATH_GPT_landmark_postcards_probability_l1408_140833

theorem landmark_postcards_probability :
  let total_postcards := 12
  let landmark_postcards := 4
  let total_arrangements := Nat.factorial total_postcards
  let favorable_arrangements := Nat.factorial (total_postcards - landmark_postcards + 1) * Nat.factorial landmark_postcards
  favorable_arrangements / total_arrangements = (1:ℝ) / 55 :=
by
  sorry

end NUMINAMATH_GPT_landmark_postcards_probability_l1408_140833


namespace NUMINAMATH_GPT_exists_quad_root_l1408_140801

theorem exists_quad_root (a b c d : ℝ) (h : a * c = 2 * b + 2 * d) :
  (∃ x, x^2 + a * x + b = 0) ∨ (∃ x, x^2 + c * x + d = 0) :=
sorry

end NUMINAMATH_GPT_exists_quad_root_l1408_140801


namespace NUMINAMATH_GPT_solve_for_m_l1408_140889

noncomputable def operation (a b c x y : ℝ) := a * x + b * y + c * x * y

theorem solve_for_m (a b c : ℝ) (h1 : operation a b c 1 2 = 3)
                              (h2 : operation a b c 2 3 = 4) 
                              (h3 : ∃ (m : ℝ), m ≠ 0 ∧ ∀ (x : ℝ), operation a b c x m = x) :
  ∃ (m : ℝ), m = 4 :=
sorry

end NUMINAMATH_GPT_solve_for_m_l1408_140889


namespace NUMINAMATH_GPT_roundness_720_eq_7_l1408_140835

def roundness (n : ℕ) : ℕ :=
  if h : n > 1 then
    let factors := n.factorization
    factors.sum (λ _ k => k)
  else 0

theorem roundness_720_eq_7 : roundness 720 = 7 := by
  sorry

end NUMINAMATH_GPT_roundness_720_eq_7_l1408_140835


namespace NUMINAMATH_GPT_CaitlinIs24_l1408_140840

-- Definition using the given conditions
def AuntAnnaAge : ℕ := 45
def BriannaAge : ℕ := (2 * AuntAnnaAge) / 3
def CaitlinAge : ℕ := BriannaAge - 6

-- Statement to be proved
theorem CaitlinIs24 : CaitlinAge = 24 :=
by
  sorry

end NUMINAMATH_GPT_CaitlinIs24_l1408_140840


namespace NUMINAMATH_GPT_solve_for_w_l1408_140896

theorem solve_for_w (w : ℂ) (i : ℂ) (i_squared : i^2 = -1) 
  (h : 3 - i * w = 1 + 2 * i * w) : 
  w = -2 * i / 3 := 
sorry

end NUMINAMATH_GPT_solve_for_w_l1408_140896


namespace NUMINAMATH_GPT_sum_of_coordinates_l1408_140869

theorem sum_of_coordinates (x y : ℚ) (h₁ : y = 5) (h₂ : (y - 0) / (x - 0) = 3/4) : x + y = 35/3 :=
by sorry

end NUMINAMATH_GPT_sum_of_coordinates_l1408_140869


namespace NUMINAMATH_GPT_sean_bought_3_sodas_l1408_140851

def soda_cost (S : ℕ) : ℕ := S * 1
def soup_cost (S : ℕ) (C : ℕ) : Prop := C = S
def sandwich_cost (C : ℕ) (X : ℕ) : Prop := X = 3 * C
def total_cost (S C X : ℕ) : Prop := S + 2 * C + X = 18

theorem sean_bought_3_sodas (S C X : ℕ) (h1 : soup_cost S C) (h2 : sandwich_cost C X) (h3 : total_cost S C X) : S = 3 :=
by
  sorry

end NUMINAMATH_GPT_sean_bought_3_sodas_l1408_140851


namespace NUMINAMATH_GPT_original_houses_count_l1408_140805

namespace LincolnCounty

-- Define the constants based on the conditions
def houses_built_during_boom : ℕ := 97741
def houses_now : ℕ := 118558

-- Statement of the theorem
theorem original_houses_count : houses_now - houses_built_during_boom = 20817 := 
by sorry

end LincolnCounty

end NUMINAMATH_GPT_original_houses_count_l1408_140805


namespace NUMINAMATH_GPT_welders_that_left_first_day_l1408_140892

-- Definitions of conditions
def welders := 12
def days_to_complete_order := 3
def days_remaining_work_after_first_day := 8
def work_done_first_day (r : ℝ) := welders * r * 1
def total_work (r : ℝ) := welders * r * days_to_complete_order

-- Theorem statement
theorem welders_that_left_first_day (r : ℝ) : 
  ∃ x : ℝ, 
    (welders - x) * r * days_remaining_work_after_first_day = total_work r - work_done_first_day r 
    ∧ x = 9 :=
by
  sorry

end NUMINAMATH_GPT_welders_that_left_first_day_l1408_140892


namespace NUMINAMATH_GPT_train_speed_l1408_140864

theorem train_speed (length_m : ℝ) (time_s : ℝ) (h_length : length_m = 133.33333333333334) (h_time : time_s = 8) : 
  let length_km := length_m / 1000
  let time_hr := time_s / 3600
  length_km / time_hr = 60 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l1408_140864


namespace NUMINAMATH_GPT_inequality_proof_l1408_140879

variable (x y z : ℝ)
variable (hx : 0 < x)
variable (hy : 0 < y)
variable (hz : 0 < z)

theorem inequality_proof :
  (x + 1) / (y + 1) + (y + 1) / (z + 1) + (z + 1) / (x + 1) ≤ x / y + y / z + z / x :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1408_140879


namespace NUMINAMATH_GPT_nondegenerate_ellipse_iff_l1408_140861

theorem nondegenerate_ellipse_iff (k : ℝ) :
  (∃ x y : ℝ, x^2 + 9*y^2 - 6*x + 27*y = k) ↔ k > -117/4 :=
by
  sorry

end NUMINAMATH_GPT_nondegenerate_ellipse_iff_l1408_140861


namespace NUMINAMATH_GPT_g_inv_f_five_l1408_140843

-- Declare the existence of functions f and g and their inverses
variables (f g : ℝ → ℝ)

-- Given condition from the problem
axiom inv_cond : ∀ x, f⁻¹ (g x) = 4 * x - 1

-- Define the specific problem to solve
theorem g_inv_f_five : g⁻¹ (f 5) = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_g_inv_f_five_l1408_140843


namespace NUMINAMATH_GPT_find_quotient_l1408_140808

-- Constants representing the given conditions
def dividend : ℕ := 690
def divisor : ℕ := 36
def remainder : ℕ := 6

-- Theorem statement
theorem find_quotient : ∃ (quotient : ℕ), dividend = (divisor * quotient) + remainder ∧ quotient = 19 := 
by
  sorry

end NUMINAMATH_GPT_find_quotient_l1408_140808


namespace NUMINAMATH_GPT_estimate_3_sqrt_2_range_l1408_140899

theorem estimate_3_sqrt_2_range :
  4 < 3 * Real.sqrt 2 ∧ 3 * Real.sqrt 2 < 5 :=
by
  sorry

end NUMINAMATH_GPT_estimate_3_sqrt_2_range_l1408_140899


namespace NUMINAMATH_GPT_exists_natural_pairs_a_exists_natural_pair_b_l1408_140847

open Nat

-- Part (a) Statement
theorem exists_natural_pairs_a (x y : ℕ) :
  x^2 - y^2 = 105 → (x, y) = (53, 52) ∨ (x, y) = (19, 16) ∨ (x, y) = (13, 8) ∨ (x, y) = (11, 4) :=
sorry

-- Part (b) Statement
theorem exists_natural_pair_b (x y : ℕ) :
  2*x^2 + 5*x*y - 12*y^2 = 28 → (x, y) = (8, 5) :=
sorry

end NUMINAMATH_GPT_exists_natural_pairs_a_exists_natural_pair_b_l1408_140847


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1408_140895

variables (a b : ℚ)

theorem simplify_and_evaluate_expression : 
  (4 * (a^2 - 2 * a * b) - (3 * a^2 - 5 * a * b + 1)) = 5 :=
by
  let a := -2
  let b := (1 : ℚ) / 3
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1408_140895


namespace NUMINAMATH_GPT_median_length_YN_perimeter_triangle_XYZ_l1408_140882

-- Definitions for conditions
noncomputable def length_XY : ℝ := 5
noncomputable def length_XZ : ℝ := 12
noncomputable def is_right_angle_XYZ : Prop := true
noncomputable def midpoint_N : ℝ := length_XZ / 2

-- Theorem statement for the length of the median YN
theorem median_length_YN (XY XZ : ℝ) (right_angle : is_right_angle_XYZ) :
  XY = 5 ∧ XZ = 12 → (XY^2 + XZ^2) = 169 → (13 / 2) = 6.5 := by
  sorry

-- Theorem statement for the perimeter of triangle XYZ
theorem perimeter_triangle_XYZ (XY XZ : ℝ) (right_angle : is_right_angle_XYZ) :
  XY = 5 ∧ XZ = 12 → (XY^2 + XZ^2) = 169 → (XY + XZ + 13) = 30 := by
  sorry

end NUMINAMATH_GPT_median_length_YN_perimeter_triangle_XYZ_l1408_140882


namespace NUMINAMATH_GPT_gray_area_correct_l1408_140844

-- Define the side lengths of the squares
variable (a b : ℝ)

-- Define the areas of the larger and smaller squares
def area_large_square : ℝ := (a + b) * (a + b)
def area_small_square : ℝ := a * a

-- Define the gray area
def gray_area : ℝ := area_large_square a b - area_small_square a

-- The proof statement
theorem gray_area_correct (a b : ℝ) : gray_area a b = 2 * a * b + b ^ 2 := by
  sorry

end NUMINAMATH_GPT_gray_area_correct_l1408_140844


namespace NUMINAMATH_GPT_hyperbola_equation_l1408_140837

theorem hyperbola_equation (a b c : ℝ) (e : ℝ) 
  (h1 : e = (Real.sqrt 6) / 2)
  (h2 : a > 0)
  (h3 : b > 0)
  (h4 : (c / a) = e)
  (h5 : (b * c) / (Real.sqrt (b^2 + a^2)) = 1) :
  (∃ a b : ℝ, (a = Real.sqrt 2) ∧ (b = 1) ∧ (∀ x y : ℝ, (x^2 / 2) - y^2 = 1)) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l1408_140837


namespace NUMINAMATH_GPT_tan_alpha_half_l1408_140821

theorem tan_alpha_half (α: ℝ) (h: Real.tan α = 1/2) :
  (1 + 2 * Real.sin (Real.pi - α) * Real.cos (-2 * Real.pi - α)) / (Real.sin (-α)^2 - Real.sin (5 * Real.pi / 2 - α)^2) = -3 := 
by
  sorry

end NUMINAMATH_GPT_tan_alpha_half_l1408_140821


namespace NUMINAMATH_GPT_first_donor_amount_l1408_140825

theorem first_donor_amount
  (x second third fourth : ℝ)
  (h1 : second = 2 * x)
  (h2 : third = 3 * second)
  (h3 : fourth = 4 * third)
  (h4 : x + second + third + fourth = 132)
  : x = 4 := 
by 
  -- Simply add this line to make the theorem complete without proof.
  sorry

end NUMINAMATH_GPT_first_donor_amount_l1408_140825


namespace NUMINAMATH_GPT_convert_rect_to_polar_l1408_140883

theorem convert_rect_to_polar (y x : ℝ) (h : y = x) : ∃ θ : ℝ, θ = π / 4 :=
by
  sorry

end NUMINAMATH_GPT_convert_rect_to_polar_l1408_140883


namespace NUMINAMATH_GPT_number_of_girls_at_camp_l1408_140868

theorem number_of_girls_at_camp (total_people : ℕ) (difference_boys_girls : ℕ) (nb_girls : ℕ) :
  total_people = 133 ∧ difference_boys_girls = 33 ∧ 2 * nb_girls + 33 = total_people → nb_girls = 50 := 
by
  intros
  sorry

end NUMINAMATH_GPT_number_of_girls_at_camp_l1408_140868


namespace NUMINAMATH_GPT_combined_work_time_l1408_140842

def ajay_completion_time : ℕ := 8
def vijay_completion_time : ℕ := 24

theorem combined_work_time (T_A T_V : ℕ) (h1 : T_A = ajay_completion_time) (h2 : T_V = vijay_completion_time) :
  1 / (1 / (T_A : ℝ) + 1 / (T_V : ℝ)) = 6 :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_combined_work_time_l1408_140842


namespace NUMINAMATH_GPT_initial_deadline_in_days_l1408_140816

theorem initial_deadline_in_days
  (men_initial : ℕ)
  (days_initial : ℕ)
  (hours_per_day_initial : ℕ)
  (fraction_work_initial : ℚ)
  (additional_men : ℕ)
  (hours_per_day_additional : ℕ)
  (fraction_work_additional : ℚ)
  (total_work : ℚ := men_initial * days_initial * hours_per_day_initial)
  (remaining_days : ℚ := (men_initial * days_initial * hours_per_day_initial) / (additional_men * hours_per_day_additional * fraction_work_additional))
  (total_days : ℚ := days_initial + remaining_days) :
  men_initial = 100 →
  days_initial = 25 →
  hours_per_day_initial = 8 →
  fraction_work_initial = 1 / 3 →
  additional_men = 160 →
  hours_per_day_additional = 10 →
  fraction_work_additional = 2 / 3 →
  total_days = 37.5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_initial_deadline_in_days_l1408_140816


namespace NUMINAMATH_GPT_arc_length_one_radian_l1408_140871

-- Given definitions and conditions
def radius : ℝ := 6370
def angle : ℝ := 1

-- Arc length formula
def arc_length (R α : ℝ) : ℝ := R * α

-- Statement to prove
theorem arc_length_one_radian : arc_length radius angle = 6370 := 
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_arc_length_one_radian_l1408_140871


namespace NUMINAMATH_GPT_problem1_problem2_l1408_140873

-- Problem 1: Simplify the calculation: 6.9^2 + 6.2 * 6.9 + 3.1^2
theorem problem1 : 6.9^2 + 6.2 * 6.9 + 3.1^2 = 100 := 
by
  sorry

-- Problem 2: Simplify and find the value of the expression with given conditions
theorem problem2 (a b : ℝ) (h1 : a = 1) (h2 : b = 0.5) :
  (a^2 * b^3 + 2 * a^3 * b) / (2 * a * b) - (a + 2 * b) * (a - 2 * b) = 9 / 8 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1408_140873


namespace NUMINAMATH_GPT_combined_resistance_parallel_l1408_140802

theorem combined_resistance_parallel (x y : ℝ) (r : ℝ) (hx : x = 3) (hy : y = 5) 
  (h : 1 / r = 1 / x + 1 / y) : r = 15 / 8 :=
by
  sorry

end NUMINAMATH_GPT_combined_resistance_parallel_l1408_140802


namespace NUMINAMATH_GPT_minute_hand_distance_traveled_l1408_140881

noncomputable def radius : ℝ := 8
noncomputable def minutes_in_one_revolution : ℝ := 60
noncomputable def total_minutes : ℝ := 45

theorem minute_hand_distance_traveled :
  (total_minutes / minutes_in_one_revolution) * (2 * Real.pi * radius) = 12 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_minute_hand_distance_traveled_l1408_140881


namespace NUMINAMATH_GPT_original_mixture_acid_percent_l1408_140875

-- Definitions of conditions as per the original problem
def original_acid_percentage (a w : ℕ) (h1 : 4 * a = a + w + 2) (h2 : 5 * (a + 2) = 2 * (a + w + 4)) : Prop :=
  (a * 100) / (a + w) = 100 / 3

-- Main theorem statement
theorem original_mixture_acid_percent (a w : ℕ) 
  (h1 : 4 * a = a + w + 2)
  (h2 : 5 * (a + 2) = 2 * (a + w + 4)) : original_acid_percentage a w h1 h2 :=
sorry

end NUMINAMATH_GPT_original_mixture_acid_percent_l1408_140875


namespace NUMINAMATH_GPT_sum_of_reciprocals_l1408_140829

theorem sum_of_reciprocals (a b : ℝ) (h_sum : a + b = 15) (h_prod : a * b = 225) :
  (1 / a) + (1 / b) = 1 / 15 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l1408_140829


namespace NUMINAMATH_GPT_isosceles_right_triangle_side_length_l1408_140824

theorem isosceles_right_triangle_side_length
  (a b : ℝ)
  (h_triangle : a = b ∨ b = a)
  (h_hypotenuse : xy > yz)
  (h_area : (1 / 2) * a * b = 9) :
  xy = 6 :=
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_isosceles_right_triangle_side_length_l1408_140824


namespace NUMINAMATH_GPT_van_distance_l1408_140850

theorem van_distance
  (D : ℝ)  -- distance the van needs to cover
  (S : ℝ)  -- original speed
  (h1 : D = S * 5)  -- the van takes 5 hours to cover the distance D
  (h2 : D = 62 * 7.5)  -- the van should maintain a speed of 62 kph to cover the same distance in 7.5 hours
  : D = 465 :=         -- prove that the distance D is 465 kilometers
by
  sorry

end NUMINAMATH_GPT_van_distance_l1408_140850


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1408_140836

def vectors_parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

def vector_a (x : ℝ) : ℝ × ℝ := (2, x - 1)
def vector_b (x : ℝ) : ℝ × ℝ := (x + 1, 4)

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  x = 3 → vectors_parallel (vector_a x) (vector_b x) ∧
  vectors_parallel (vector_a 3) (vector_b 3) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1408_140836


namespace NUMINAMATH_GPT_different_result_l1408_140832

theorem different_result :
  let A := -2 - (-3)
  let B := 2 - 3
  let C := -3 + 2
  let D := -3 - (-2)
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B = C ∧ B = D :=
by
  sorry

end NUMINAMATH_GPT_different_result_l1408_140832


namespace NUMINAMATH_GPT_value_of_a_l1408_140831

theorem value_of_a (a : ℝ) : (-2)^2 + 3*(-2) + a = 0 → a = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_a_l1408_140831


namespace NUMINAMATH_GPT_problem_1_problem_2_l1408_140865

theorem problem_1 (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a^3 + b^3 = 2) : (a + b) * (a^5 + b^5) ≥ 4 :=
sorry

theorem problem_2 (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a^3 + b^3 = 2) : a + b ≤ 2 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1408_140865


namespace NUMINAMATH_GPT_Stuart_reward_points_l1408_140888

theorem Stuart_reward_points (reward_points_per_unit : ℝ) (spending : ℝ) (unit_amount : ℝ) : 
  reward_points_per_unit = 5 → 
  spending = 200 → 
  unit_amount = 25 → 
  (spending / unit_amount) * reward_points_per_unit = 40 :=
by 
  intros h_points h_spending h_unit
  sorry

end NUMINAMATH_GPT_Stuart_reward_points_l1408_140888


namespace NUMINAMATH_GPT_second_smallest_palindromic_prime_l1408_140867

-- Three digit number definition
def three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Palindromic number definition
def is_palindromic (n : ℕ) : Prop := 
  let hundreds := n / 100
  let tens := (n % 100) / 10
  let ones := n % 10
  hundreds = ones 

-- Prime number definition
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Second-smallest three-digit palindromic prime
theorem second_smallest_palindromic_prime :
  ∃ n : ℕ, three_digit_number n ∧ is_palindromic n ∧ is_prime n ∧ 
  ∃ m : ℕ, three_digit_number m ∧ is_palindromic m ∧ is_prime m ∧ m > 101 ∧ m < n ∧ 
  n = 131 := 
by
  sorry

end NUMINAMATH_GPT_second_smallest_palindromic_prime_l1408_140867


namespace NUMINAMATH_GPT_domain_ln_x_squared_minus_2_l1408_140815

theorem domain_ln_x_squared_minus_2 (x : ℝ) : 
  x^2 - 2 > 0 ↔ (x < -Real.sqrt 2 ∨ x > Real.sqrt 2) := 
by 
  sorry

end NUMINAMATH_GPT_domain_ln_x_squared_minus_2_l1408_140815


namespace NUMINAMATH_GPT_correct_transformation_l1408_140887

theorem correct_transformation (a b c : ℝ) (h : a / c = b / c) (hc : c ≠ 0) : a = b :=
by
  sorry

end NUMINAMATH_GPT_correct_transformation_l1408_140887


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l1408_140897

theorem isosceles_triangle_base_length (a b : ℕ) (h1 : a = 7) (h2 : b + 2 * a = 25) : b = 11 := by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l1408_140897


namespace NUMINAMATH_GPT_trapezoid_circle_center_l1408_140848

theorem trapezoid_circle_center 
  (EF GH : ℝ)
  (FG HE : ℝ)
  (p q : ℕ) 
  (rel_prime : Nat.gcd p q = 1)
  (EQ GH : ℝ)
  (h1 : EF = 105)
  (h2 : FG = 57)
  (h3 : GH = 22)
  (h4 : HE = 80)
  (h5 : EQ = p / q)
  (h6 : p = 10)
  (h7 : q = 1) :
  p + q = 11 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_circle_center_l1408_140848


namespace NUMINAMATH_GPT_sixth_term_is_sixteen_l1408_140854

-- Definition of the conditions
def first_term : ℝ := 512
def eighth_term (r : ℝ) : Prop := 512 * r^7 = 2

-- Proving the 6th term is 16 given the conditions
theorem sixth_term_is_sixteen (r : ℝ) (hr : eighth_term r) :
  512 * r^5 = 16 :=
by
  sorry

end NUMINAMATH_GPT_sixth_term_is_sixteen_l1408_140854


namespace NUMINAMATH_GPT_extreme_value_of_f_range_of_a_l1408_140845

noncomputable def f (x a : ℝ) := Real.exp x - a * x - 1

theorem extreme_value_of_f (a : ℝ) (ha : 0 < a) : ∃ x, f x a = a - a * Real.log a - 1 :=
sorry

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 ≤ 2 ∧ 0 ≤ x2 ∧ x2 ≤ 2 ∧ f x1 a = f x2 a ∧ abs (x1 - x2) ≥ 1 ) →
  (e - 1 < a ∧ a < Real.exp 2 - Real.exp 1) :=
sorry

end NUMINAMATH_GPT_extreme_value_of_f_range_of_a_l1408_140845


namespace NUMINAMATH_GPT_two_a_minus_two_d_eq_zero_l1408_140800

noncomputable def g (a b c d x : ℝ) : ℝ := (2 * a * x - b) / (c * x - 2 * d)

theorem two_a_minus_two_d_eq_zero (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
  (h5 : ∀ x : ℝ, (g a a c d (g a b c d x)) = x) : 2 * a - 2 * d = 0 :=
sorry

end NUMINAMATH_GPT_two_a_minus_two_d_eq_zero_l1408_140800


namespace NUMINAMATH_GPT_floor_sqrt_72_l1408_140818

theorem floor_sqrt_72 : ⌊Real.sqrt 72⌋ = 8 :=
by
  -- Proof required here
  sorry

end NUMINAMATH_GPT_floor_sqrt_72_l1408_140818


namespace NUMINAMATH_GPT_common_ratio_half_l1408_140838

-- Definitions based on conditions
def geom_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, a (n+1) = a n * q
def arith_seq (x y z : ℝ) := x + z = 2 * y

-- Theorem statement
theorem common_ratio_half (a : ℕ → ℝ) (q : ℝ) (h_geom : geom_seq a q)
  (h_arith : arith_seq (a 5) (a 6 + a 8) (a 7)) : q = 1 / 2 := 
sorry

end NUMINAMATH_GPT_common_ratio_half_l1408_140838


namespace NUMINAMATH_GPT_largest_divisor_of_n4_minus_n2_is_12_l1408_140812

theorem largest_divisor_of_n4_minus_n2_is_12 : ∀ n : ℤ, 12 ∣ (n^4 - n^2) :=
by
  intro n
  -- Placeholder for proof; the detailed steps of the proof go here
  sorry

end NUMINAMATH_GPT_largest_divisor_of_n4_minus_n2_is_12_l1408_140812


namespace NUMINAMATH_GPT_eval_operation_l1408_140863

-- Definition of the * operation based on the given table
def op (a b : ℕ) : ℕ :=
  match a, b with
  | 1, 1 => 4
  | 1, 2 => 1
  | 1, 3 => 2
  | 1, 4 => 3
  | 2, 1 => 1
  | 2, 2 => 3
  | 2, 3 => 4
  | 2, 4 => 2
  | 3, 1 => 2
  | 3, 2 => 4
  | 3, 3 => 1
  | 3, 4 => 3
  | 4, 1 => 3
  | 4, 2 => 2
  | 4, 3 => 3
  | 4, 4 => 4
  | _, _ => 0 -- Default case (not needed as per the given problem definition)

-- Statement of the problem in Lean 4
theorem eval_operation : op (op 3 1) (op 4 2) = 3 :=
by {
  sorry -- Proof to be provided
}

end NUMINAMATH_GPT_eval_operation_l1408_140863


namespace NUMINAMATH_GPT_purely_periodic_denominator_l1408_140849

theorem purely_periodic_denominator :
  ∀ q : ℕ, (∃ a : ℕ, (∃ b : ℕ, q = 99 ∧ (a < 10) ∧ (b < 10) ∧ (∃ f : ℝ, f = ↑a / (10 * q) ∧ ∃ g : ℝ, g = (0.01 * ↑b / (10 * (99 / q))))) → q = 11 ∨ q = 33 ∨ q = 99) :=
by sorry

end NUMINAMATH_GPT_purely_periodic_denominator_l1408_140849


namespace NUMINAMATH_GPT_recorder_price_new_l1408_140804

theorem recorder_price_new (a b : ℕ) (h1 : 10 * a + b < 50) (h2 : 10 * b + a = (10 * a + b) * 12 / 10) :
  10 * b + a = 54 :=
by
  sorry

end NUMINAMATH_GPT_recorder_price_new_l1408_140804


namespace NUMINAMATH_GPT_markers_per_box_l1408_140819

theorem markers_per_box
  (students : ℕ) (boxes : ℕ) (group1_students : ℕ) (group1_markers : ℕ)
  (group2_students : ℕ) (group2_markers : ℕ) (last_group_markers : ℕ)
  (h_students : students = 30)
  (h_boxes : boxes = 22)
  (h_group1_students : group1_students = 10)
  (h_group1_markers : group1_markers = 2)
  (h_group2_students : group2_students = 15)
  (h_group2_markers : group2_markers = 4)
  (h_last_group_markers : last_group_markers = 6) :
  (110 = students * ((group1_students * group1_markers + group2_students * group2_markers + (students - group1_students - group2_students) * last_group_markers)) / boxes) :=
by
  sorry

end NUMINAMATH_GPT_markers_per_box_l1408_140819


namespace NUMINAMATH_GPT_inequality_proof_l1408_140874

-- Define the inequality problem in Lean 4
theorem inequality_proof (x y : ℝ) (h1 : x ≠ -1) (h2 : y ≠ -1) (h3 : x * y = 1) : 
  ( (2 + x) / (1 + x) )^2 + ( (2 + y) / (1 + y) )^2 ≥ 9 / 2 := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1408_140874


namespace NUMINAMATH_GPT_required_words_to_learn_l1408_140876

def total_words : ℕ := 500
def required_percentage : ℕ := 85

theorem required_words_to_learn (x : ℕ) :
  (x : ℚ) / total_words ≥ (required_percentage : ℚ) / 100 ↔ x ≥ 425 := 
sorry

end NUMINAMATH_GPT_required_words_to_learn_l1408_140876


namespace NUMINAMATH_GPT_find_side_y_l1408_140877

noncomputable def side_length_y : ℝ :=
  let AB := 10 / Real.sqrt 2
  let AD := 10
  let CD := AD / 2
  CD * Real.sqrt 3

theorem find_side_y : side_length_y = 5 * Real.sqrt 3 := by
  let AB : ℝ := 10 / Real.sqrt 2
  let AD : ℝ := 10
  let CD : ℝ := AD / 2
  have h1 : CD * Real.sqrt 3 = 5 * Real.sqrt 3 := by sorry
  exact h1

end NUMINAMATH_GPT_find_side_y_l1408_140877


namespace NUMINAMATH_GPT_first_competitor_hotdogs_l1408_140852

theorem first_competitor_hotdogs (x y z : ℕ) (h1 : y = 3 * x) (h2 : z = 2 * y) (h3 : z * 5 = 300) : x = 10 :=
sorry

end NUMINAMATH_GPT_first_competitor_hotdogs_l1408_140852


namespace NUMINAMATH_GPT_find_some_number_l1408_140834

theorem find_some_number :
  ∃ (some_number : ℝ), (0.0077 * 3.6) / (some_number * 0.1 * 0.007) = 990.0000000000001 ∧ some_number = 0.04 :=
  sorry

end NUMINAMATH_GPT_find_some_number_l1408_140834


namespace NUMINAMATH_GPT_triangle_lattice_points_l1408_140814

theorem triangle_lattice_points :
  ∀ (A B C : ℕ) (AB AC BC : ℕ), 
    AB = 2016 → AC = 1533 → BC = 1533 → 
    ∃ lattice_points: ℕ, lattice_points = 1165322 := 
by
  sorry

end NUMINAMATH_GPT_triangle_lattice_points_l1408_140814


namespace NUMINAMATH_GPT_triangle_area_correct_l1408_140872

-- Define the points (vertices) of the triangle
def point1 : ℝ × ℝ := (2, 1)
def point2 : ℝ × ℝ := (8, -3)
def point3 : ℝ × ℝ := (2, 7)

-- Function to calculate the area of the triangle given three points (shoelace formula)
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - B.2 * C.1 - C.2 * A.1 - A.2 * B.1)

-- Prove that the area of the triangle with the given vertices is 18 square units
theorem triangle_area_correct : triangle_area point1 point2 point3 = 18 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_correct_l1408_140872


namespace NUMINAMATH_GPT_total_travel_time_correct_l1408_140828

-- Define the conditions
def highway_distance : ℕ := 100 -- miles
def mountain_distance : ℕ := 15 -- miles
def break_time : ℕ := 30 -- minutes
def time_on_mountain_road : ℕ := 45 -- minutes
def speed_ratio : ℕ := 5

-- Define the speeds using the given conditions.
def mountain_speed := mountain_distance / time_on_mountain_road -- miles per minute
def highway_speed := speed_ratio * mountain_speed -- miles per minute

-- Prove that total trip time equals 240 minutes
def total_trip_time : ℕ := 2 * (time_on_mountain_road + (highway_distance / highway_speed)) + break_time

theorem total_travel_time_correct : total_trip_time = 240 := 
by
  -- to be proved
  sorry

end NUMINAMATH_GPT_total_travel_time_correct_l1408_140828


namespace NUMINAMATH_GPT_find_12th_term_l1408_140891

noncomputable def geometric_sequence (a r : ℝ) : ℕ → ℝ
| 0 => a
| (n+1) => r * geometric_sequence a r n

theorem find_12th_term : ∃ a r, geometric_sequence a r 4 = 5 ∧ geometric_sequence a r 7 = 40 ∧ geometric_sequence a r 11 = 640 :=
by
  -- statement only, no proof provided
  sorry

end NUMINAMATH_GPT_find_12th_term_l1408_140891


namespace NUMINAMATH_GPT_fraction_evaluation_l1408_140811

theorem fraction_evaluation : (20 + 24) / (20 - 24) = -11 := by
  sorry

end NUMINAMATH_GPT_fraction_evaluation_l1408_140811


namespace NUMINAMATH_GPT_bacteria_initial_count_l1408_140826

noncomputable def initial_bacteria (b_final : ℕ) (q : ℕ) : ℕ :=
  b_final / 4^q

theorem bacteria_initial_count : initial_bacteria 262144 4 = 1024 := by
  sorry

end NUMINAMATH_GPT_bacteria_initial_count_l1408_140826


namespace NUMINAMATH_GPT_Jessica_victory_l1408_140817

def bullseye_points : ℕ := 10
def other_possible_scores : Set ℕ := {0, 2, 5, 8, 10}
def minimum_score_per_shot : ℕ := 2
def shots_taken : ℕ := 40
def remaining_shots : ℕ := 40
def jessica_advantage : ℕ := 30

def victory_condition (n : ℕ) : Prop :=
  8 * n + 80 > 370

theorem Jessica_victory :
  ∃ n, victory_condition n ∧ n = 37 :=
by
  use 37
  sorry

end NUMINAMATH_GPT_Jessica_victory_l1408_140817


namespace NUMINAMATH_GPT_cosine_value_l1408_140885

theorem cosine_value (α : ℝ) (h : Real.sin (α - Real.pi / 3) = 1 / 3) :
  Real.cos (α + Real.pi / 6) = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_cosine_value_l1408_140885


namespace NUMINAMATH_GPT_problem_solution_l1408_140894

theorem problem_solution (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 3 = 21 :=
by 
  sorry

end NUMINAMATH_GPT_problem_solution_l1408_140894


namespace NUMINAMATH_GPT_total_beads_needed_l1408_140878

-- Condition 1: Number of members in the crafts club
def members := 9

-- Condition 2: Number of necklaces each member makes
def necklaces_per_member := 2

-- Condition 3: Number of beads each necklace requires
def beads_per_necklace := 50

-- Total number of beads needed
theorem total_beads_needed :
  (members * (necklaces_per_member * beads_per_necklace)) = 900 := 
by
  sorry

end NUMINAMATH_GPT_total_beads_needed_l1408_140878


namespace NUMINAMATH_GPT_julia_played_more_kids_on_monday_l1408_140862

def n_monday : ℕ := 6
def n_tuesday : ℕ := 5

theorem julia_played_more_kids_on_monday : n_monday - n_tuesday = 1 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_julia_played_more_kids_on_monday_l1408_140862
