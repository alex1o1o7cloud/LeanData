import Mathlib

namespace problem1_l227_22733

theorem problem1 (x y : ℤ) (h : |x + 2| + |y - 3| = 0) : x - y + 1 = -4 :=
sorry

end problem1_l227_22733


namespace votes_cast_l227_22710

theorem votes_cast (V : ℝ) (h1 : ∃ (x : ℝ), x = 0.35 * V) (h2 : ∃ (y : ℝ), y = x + 2100) : V = 7000 :=
by sorry

end votes_cast_l227_22710


namespace sqrt_of_neg_five_squared_l227_22797

theorem sqrt_of_neg_five_squared : Real.sqrt ((-5 : Real) ^ 2) = 5 := 
by 
  sorry

end sqrt_of_neg_five_squared_l227_22797


namespace find_q_l227_22756

open Real

noncomputable def q := (9 + 3 * Real.sqrt 5) / 2

theorem find_q (p q : ℝ) (hp : p > 1) (hq : q > 1) 
  (h1 : 1 / p + 1 / q = 1) (h2 : p * q = 9) : q = (9 + 3 * Real.sqrt 5) / 2 :=
by
  sorry

end find_q_l227_22756


namespace number_of_boys_in_class_l227_22705

theorem number_of_boys_in_class (n : ℕ)
  (avg_height : ℕ) (incorrect_height : ℕ) (actual_height : ℕ)
  (actual_avg_height : ℕ)
  (h1 : avg_height = 180)
  (h2 : incorrect_height = 156)
  (h3 : actual_height = 106)
  (h4 : actual_avg_height = 178)
  : n = 25 :=
by 
  -- We have the following conditions:
  -- Incorrect total height = avg_height * n
  -- Difference due to incorrect height = incorrect_height - actual_height
  -- Correct total height = avg_height * n - (incorrect_height - actual_height)
  -- Total height according to actual average = actual_avg_height * n
  -- Equating both, we have:
  -- avg_height * n - (incorrect_height - actual_height) = actual_avg_height * n
  -- We know avg_height, incorrect_height, actual_height, actual_avg_height from h1, h2, h3, h4
  -- Substituting these values and solving:
  -- 180n - (156 - 106) = 178n
  -- 180n - 50 = 178n
  -- 2n = 50
  -- n = 25
  sorry

end number_of_boys_in_class_l227_22705


namespace smallest_positive_period_of_f_extreme_values_of_f_on_interval_l227_22740

noncomputable def f (x : ℝ) : ℝ :=
  let a : ℝ × ℝ := (2 * Real.cos x, Real.sqrt 3 * Real.cos x)
  let b : ℝ × ℝ := (Real.cos x, 2 * Real.sin x)
  a.1 * b.1 + a.2 * b.2

theorem smallest_positive_period_of_f :
  ∃ p > 0, ∀ x : ℝ, f (x + p) = f x ∧ p = Real.pi := sorry

theorem extreme_values_of_f_on_interval :
  ∃ max_val min_val, (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ max_val) ∧
                     (∀ x ∈ Set.Icc 0 (Real.pi / 2), min_val ≤ f x) ∧
                     max_val = 3 ∧ min_val = 0 := sorry

end smallest_positive_period_of_f_extreme_values_of_f_on_interval_l227_22740


namespace three_digit_integer_one_more_than_LCM_l227_22730

theorem three_digit_integer_one_more_than_LCM:
  ∃ (n : ℕ), (n > 99 ∧ n < 1000) ∧ (∃ (k : ℕ), n = k + 1 ∧ (∃ m, k = 3 * 4 * 5 * 7 * 2^m)) :=
  sorry

end three_digit_integer_one_more_than_LCM_l227_22730


namespace gcf_3465_10780_l227_22724

theorem gcf_3465_10780 : Nat.gcd 3465 10780 = 385 := by
  sorry

end gcf_3465_10780_l227_22724


namespace product_of_repeating_decimal_l227_22717

theorem product_of_repeating_decimal (p : ℝ) (h : p = 0.6666666666666667) : p * 6 = 4 :=
sorry

end product_of_repeating_decimal_l227_22717


namespace least_number_to_subtract_l227_22758

theorem least_number_to_subtract (x : ℕ) (h : x = 1234567890) : ∃ n, x - n = 5 := 
  sorry

end least_number_to_subtract_l227_22758


namespace min_colors_correct_l227_22787

def min_colors (n : Nat) : Nat :=
  if n = 1 then 1
  else if n = 2 then 2
  else 3

theorem min_colors_correct (n : Nat) : min_colors n = 
  if n = 1 then 1
  else if n = 2 then 2
  else 3 := by
  sorry

end min_colors_correct_l227_22787


namespace sin_ineq_l227_22785

open Real

theorem sin_ineq (n : ℕ) (h : n > 0) : sin (π / (4 * n)) ≥ (sqrt 2) / (2 * n) :=
sorry

end sin_ineq_l227_22785


namespace find_n_for_geom_sum_l227_22799

-- Define the first term and the common ratio
def first_term := 1
def common_ratio := 1 / 2

-- Define the sum function of the first n terms of the geometric sequence
def geom_sum (n : ℕ) : ℚ := first_term * (1 - (common_ratio)^n) / (1 - common_ratio)

-- Define the target sum
def target_sum := 31 / 16

-- State the theorem to prove
theorem find_n_for_geom_sum : ∃ n : ℕ, geom_sum n = target_sum := 
    by
    sorry

end find_n_for_geom_sum_l227_22799


namespace original_water_amount_l227_22720

theorem original_water_amount (W : ℝ) 
    (evap_rate : ℝ := 0.03) 
    (days : ℕ := 22) 
    (evap_percent : ℝ := 0.055) 
    (total_evap : ℝ := evap_rate * days) 
    (evap_condition : evap_percent * W = total_evap) : W = 12 :=
by sorry

end original_water_amount_l227_22720


namespace angle_C_in_triangle_l227_22725

theorem angle_C_in_triangle (A B C : ℝ)
  (hA : A = 60)
  (hAC : C = 2 * B)
  (hSum : A + B + C = 180) : C = 80 :=
sorry

end angle_C_in_triangle_l227_22725


namespace find_y_l227_22793

theorem find_y (y : ℝ) (h : 9 * y^3 = y * 81) : y = 3 * Real.sqrt 3 :=
by
  sorry

end find_y_l227_22793


namespace final_net_earnings_l227_22726

-- Declare constants representing the problem conditions
def connor_hourly_rate : ℝ := 7.20
def connor_hours_worked : ℝ := 8.0
def emily_hourly_rate : ℝ := 2 * connor_hourly_rate
def sarah_hourly_rate : ℝ := 5 * connor_hourly_rate
def emily_hours_worked : ℝ := 10.0
def connor_deduction_rate : ℝ := 0.05
def emily_deduction_rate : ℝ := 0.08
def sarah_deduction_rate : ℝ := 0.10

-- Combined final net earnings for the day
def combined_final_net_earnings (connor_hourly_rate emily_hourly_rate sarah_hourly_rate
                                  connor_hours_worked emily_hours_worked
                                  connor_deduction_rate emily_deduction_rate sarah_deduction_rate : ℝ) : ℝ :=
  let connor_gross := connor_hourly_rate * connor_hours_worked
  let emily_gross := emily_hourly_rate * emily_hours_worked
  let sarah_gross := sarah_hourly_rate * connor_hours_worked
  let connor_net := connor_gross * (1 - connor_deduction_rate)
  let emily_net := emily_gross * (1 - emily_deduction_rate)
  let sarah_net := sarah_gross * (1 - sarah_deduction_rate)
  connor_net + emily_net + sarah_net

-- The theorem statement proving their combined final net earnings
theorem final_net_earnings : 
  combined_final_net_earnings 7.20 14.40 36.00 8.0 10.0 0.05 0.08 0.10 = 498.24 :=
by sorry

end final_net_earnings_l227_22726


namespace speed_of_stream_l227_22735

-- Definitions
variable (b s : ℝ)
def downstream_distance : ℝ := 120
def downstream_time : ℝ := 4
def upstream_distance : ℝ := 90
def upstream_time : ℝ := 6

-- Equations
def downstream_eq : Prop := downstream_distance = (b + s) * downstream_time
def upstream_eq : Prop := upstream_distance = (b - s) * upstream_time

-- Main statement
theorem speed_of_stream (h₁ : downstream_eq b s) (h₂ : upstream_eq b s) : s = 7.5 :=
by
  sorry

end speed_of_stream_l227_22735


namespace flag_design_l227_22762

/-- Given three colors and a flag with three horizontal stripes where no adjacent stripes can be the 
same color, there are exactly 12 different possible flags. -/
theorem flag_design {colors : Finset ℕ} (h_colors : colors.card = 3) : 
  ∃ n : ℕ, n = 12 ∧ (∃ f : ℕ → ℕ, (∀ i, f i ∈ colors) ∧ (∀ i < 2, f i ≠ f (i + 1))) :=
sorry

end flag_design_l227_22762


namespace elisa_math_books_l227_22768

theorem elisa_math_books (N M L : ℕ) (h₀ : 24 + M + L + 1 = N + 1) (h₁ : (N + 1) % 9 = 0) (h₂ : (N + 1) % 4 = 0) (h₃ : N < 100) : M = 7 :=
by
  sorry

end elisa_math_books_l227_22768


namespace total_tape_area_l227_22757

theorem total_tape_area 
  (long_side_1 short_side_1 : ℕ) (boxes_1 : ℕ)
  (long_side_2 short_side_2 : ℕ) (boxes_2 : ℕ)
  (long_side_3 short_side_3 : ℕ) (boxes_3 : ℕ)
  (overlap : ℕ) (tape_width : ℕ) :
  long_side_1 = 30 → short_side_1 = 15 → boxes_1 = 5 →
  long_side_2 = 40 → short_side_2 = 40 → boxes_2 = 2 →
  long_side_3 = 50 → short_side_3 = 20 → boxes_3 = 3 →
  overlap = 2 → tape_width = 2 →
  let total_length_1 := boxes_1 * (long_side_1 + overlap + 2 * (short_side_1 + overlap))
  let total_length_2 := boxes_2 * 3 * (long_side_2 + overlap)
  let total_length_3 := boxes_3 * (long_side_3 + overlap + 2 * (short_side_3 + overlap))
  let total_length := total_length_1 + total_length_2 + total_length_3
  let total_area := total_length * tape_width
  total_area = 1740 :=
  by
  -- Add the proof steps here
  -- sorry can be used to skip the proof
  sorry

end total_tape_area_l227_22757


namespace sufficient_not_necessary_condition_parallel_lines_l227_22783

theorem sufficient_not_necessary_condition_parallel_lines :
  ∀ (a : ℝ), (a = 1/2 → (∀ x y : ℝ, x + 2*a*y = 1 ↔ (x - x + 1) ≠ 0) 
            ∧ ((∃ a', a' ≠ 1/2 ∧ (∀ x y : ℝ, x + 2*a'*y = 1 ↔ (x - x + 1) ≠ 0)) → (a ≠ 1/2))) :=
by
  intro a
  sorry

end sufficient_not_necessary_condition_parallel_lines_l227_22783


namespace pqrs_product_l227_22772

theorem pqrs_product :
  let P := (Real.sqrt 2010 + Real.sqrt 2009 + Real.sqrt 2008)
  let Q := (-Real.sqrt 2010 - Real.sqrt 2009 + Real.sqrt 2008)
  let R := (Real.sqrt 2010 - Real.sqrt 2009 - Real.sqrt 2008)
  let S := (-Real.sqrt 2010 + Real.sqrt 2009 - Real.sqrt 2008)
  P * Q * R * S = 1 := by
{
  sorry -- Proof is omitted as per the provided instructions.
}

end pqrs_product_l227_22772


namespace largest_value_is_E_l227_22770

theorem largest_value_is_E :
  let A := 3 + 1 + 2 + 9
  let B := 3 * 1 + 2 + 9
  let C := 3 + 1 * 2 + 9
  let D := 3 + 1 + 2 * 9
  let E := 3 * 1 * 2 * 9
  E > A ∧ E > B ∧ E > C ∧ E > D := 
by
  let A := 3 + 1 + 2 + 9
  let B := 3 * 1 + 2 + 9
  let C := 3 + 1 * 2 + 9
  let D := 3 + 1 + 2 * 9
  let E := 3 * 1 * 2 * 9
  sorry

end largest_value_is_E_l227_22770


namespace complement_union_l227_22721

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {3, 4, 5}
def complementU (A B : Set ℕ) : Set ℕ := U \ (A ∪ B)

theorem complement_union :
  complementU A B = {2, 6} := by
  sorry

end complement_union_l227_22721


namespace spadesuit_evaluation_l227_22773

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spadesuit_evaluation : spadesuit 3 (spadesuit 4 5) = -72 := by
  sorry

end spadesuit_evaluation_l227_22773


namespace first_division_percentage_l227_22784

theorem first_division_percentage (total_students : ℕ) (second_division_percentage just_passed_students : ℕ) 
  (h1 : total_students = 300) (h2 : second_division_percentage = 54) (h3 : just_passed_students = 60) : 
  (100 - second_division_percentage - ((just_passed_students * 100) / total_students)) = 26 :=
by
  sorry

end first_division_percentage_l227_22784


namespace simplify_fraction_l227_22709

theorem simplify_fraction (a b : ℤ) (h : a = 2^6 + 2^4) (h1 : b = 2^5 - 2^2) : 
  (a / b : ℚ) = 20 / 7 := by
  sorry

end simplify_fraction_l227_22709


namespace solve_equation_l227_22704

theorem solve_equation (n m : ℤ) : 
  n^4 + 2*n^3 + 2*n^2 + 2*n + 1 = m^2 ↔ (n = 0 ∧ (m = 1 ∨ m = -1)) ∨ (n = -1 ∧ m = 0) :=
by sorry

end solve_equation_l227_22704


namespace sample_size_l227_22794

variable (num_classes : ℕ) (papers_per_class : ℕ)

theorem sample_size (h_classes : num_classes = 8) (h_papers : papers_per_class = 12) : 
  num_classes * papers_per_class = 96 := 
by 
  sorry

end sample_size_l227_22794


namespace arithmetic_mean_of_normal_distribution_l227_22752

theorem arithmetic_mean_of_normal_distribution
  (σ : ℝ) (hσ : σ = 1.5)
  (value : ℝ) (hvalue : value = 11.5)
  (hsd : value = μ - 2 * σ) :
  μ = 14.5 :=
by
  sorry

end arithmetic_mean_of_normal_distribution_l227_22752


namespace sum_first_seven_terms_is_28_l227_22706

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Define the arithmetic sequence 
def is_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop := 
  ∀ n, a (n + 1) = a n + d

-- Given conditions
axiom a2_a4_a6_sum : a 2 + a 4 + a 6 = 12

-- Prove that the sum of the first seven terms is 28
theorem sum_first_seven_terms_is_28 (h : is_arithmetic_seq a d) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 := 
sorry

end sum_first_seven_terms_is_28_l227_22706


namespace enclosed_area_of_curve_l227_22723

noncomputable def radius_of_arcs := 1

noncomputable def arc_length := (1 / 2) * Real.pi

noncomputable def side_length_of_octagon := 3

noncomputable def area_of_octagon (s : ℝ) := 
  2 * (1 + Real.sqrt 2) * s ^ 2

noncomputable def area_of_sectors (n : ℕ) (arc_radius : ℝ) (arc_theta : ℝ) := 
  n * (1 / 4) * Real.pi

theorem enclosed_area_of_curve : 
  area_of_octagon side_length_of_octagon + area_of_sectors 12 radius_of_arcs arc_length 
  = 54 + 54 * Real.sqrt 2 + 3 * Real.pi := 
by
  sorry

end enclosed_area_of_curve_l227_22723


namespace CA_inter_B_l227_22796

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 5, 7}

theorem CA_inter_B :
  (U \ A) ∩ B = {2, 7} := by
  sorry

end CA_inter_B_l227_22796


namespace rainy_days_last_week_l227_22719

-- All conditions in Lean definitions
def even_integer (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k
def cups_of_tea_n (n : ℤ) : ℤ := 3
def total_drinks (R NR : ℤ) (m : ℤ) : Prop := 2 * m * R + 3 * NR = 36
def more_tea_than_hot_chocolate (R NR : ℤ) (m : ℤ) : Prop := 3 * NR - 2 * m * R = 12
def odd_number_of_rainy_days (R : ℤ) : Prop := R % 2 = 1
def total_days_in_week (R NR : ℤ) : Prop := R + NR = 7

-- Main statement
theorem rainy_days_last_week : ∃ R m NR : ℤ, 
  odd_number_of_rainy_days R ∧ 
  total_days_in_week R NR ∧ 
  total_drinks R NR m ∧ 
  more_tea_than_hot_chocolate R NR m ∧
  R = 3 :=
by
  sorry

end rainy_days_last_week_l227_22719


namespace sum_of_f_l227_22769

noncomputable def f (x : ℝ) : ℝ := 1 / (2^x + Real.sqrt 2)

theorem sum_of_f :
  f (-5) + f (-4) + f (-3) + f (-2) + f (-1) + f 0 + f 1 + f 2 + f 3 + f 4 + f 5 + f 6 = 3 * Real.sqrt 2 :=
by
  sorry

end sum_of_f_l227_22769


namespace perp_case_parallel_distance_l227_22737

open Real

-- Define the line equations
def l1 (x y : ℝ) := 2 * x + y + 4 = 0
def l2 (a x y : ℝ) := a * x + 4 * y + 1 = 0

-- Perpendicular condition between l1 and l2
def perpendicular (a : ℝ) := (∃ x y : ℝ, l1 x y ∧ l2 a x y ∧ (2 * -a) / 4 = -1)

-- Parallel condition between l1 and l2
def parallel (a : ℝ) := (∃ x y : ℝ, l1 x y ∧ l2 a x y ∧ a = 8)

noncomputable def intersection_point : (ℝ × ℝ) := (-3/2, -1)

noncomputable def distance_between_lines : ℝ := (3 * sqrt 5) / 4

-- Statement for the intersection point when perpendicular
theorem perp_case (a : ℝ) : perpendicular a → ∃ x y, l1 x y ∧ l2 (-2) x y := 
by
  sorry

-- Statement for the distance when parallel
theorem parallel_distance {a : ℝ} : parallel a → distance_between_lines = (3 * sqrt 5) / 4 :=
by
  sorry

end perp_case_parallel_distance_l227_22737


namespace smaller_angle_measure_l227_22707

theorem smaller_angle_measure (x : ℝ) (h1 : 3 * x + 2 * x = 90) : 2 * x = 36 :=
by {
  sorry
}

end smaller_angle_measure_l227_22707


namespace alpha_beta_identity_l227_22764

open Real

theorem alpha_beta_identity 
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2)
  (h : cos β = tan α * (1 + sin β)) : 
  2 * α + β = π / 2 :=
by
  sorry

end alpha_beta_identity_l227_22764


namespace square_vectors_l227_22731

theorem square_vectors (AB CD AD : ℝ × ℝ)
  (side_length: ℝ)
  (M N : ℝ × ℝ)
  (x y: ℝ)
  (MN : ℝ × ℝ):
  side_length = 2 →
  M = ((AB.1 + CD.1) / 2, (AB.2 + CD.2) / 2) →
  N = ((CD.1 + AD.1) / 2, (CD.2 + AD.2) / 2) →
  MN = (x * AB.1 + y * AD.1, x * AB.2 + y * AD.2) →
  (x = -1/2) ∧ (y = 1/2) →
  (x * y = -1/4) ∧ ((N.1 - M.1) * AD.1 + (N.2 - M.2) * AD.2 - (N.1 - M.1) * AB.1 - (N.2 - M.2) * AB.2 = -1) :=
by
  intros side_length_cond M_cond N_cond MN_cond xy_cond
  sorry

end square_vectors_l227_22731


namespace enclosed_area_of_curve_l227_22780

/-
  The closed curve in the figure is made up of 9 congruent circular arcs each of length \(\frac{\pi}{2}\),
  where each of the centers of the corresponding circles is among the vertices of a regular hexagon of side 3.
  We want to prove that the area enclosed by the curve is \(\frac{27\sqrt{3}}{2} + \frac{9\pi}{8}\).
-/

theorem enclosed_area_of_curve :
  let side_length := 3
  let arc_length := π / 2
  let num_arcs := 9
  let hexagon_area := (3 * Real.sqrt 3 / 2) * side_length^2
  let radius := 1 / 2
  let sector_area := (π * radius^2) / 4
  let total_sector_area := num_arcs * sector_area
  let enclosed_area := hexagon_area + total_sector_area
  enclosed_area = (27 * Real.sqrt 3) / 2 + (9 * π) / 8 :=
by
  sorry

end enclosed_area_of_curve_l227_22780


namespace complex_eq_sub_l227_22715

open Complex

theorem complex_eq_sub {a b : ℝ} (h : (a : ℂ) + 2 * I = I * ((b : ℂ) - I)) : a - b = -3 := by
  sorry

end complex_eq_sub_l227_22715


namespace sin_120_eq_sqrt3_div_2_l227_22728

theorem sin_120_eq_sqrt3_div_2 : Real.sin (120 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_120_eq_sqrt3_div_2_l227_22728


namespace cost_of_one_dozen_pens_l227_22734

-- Define the initial conditions
def cost_pen : ℕ := 65
def cost_pencil := cost_pen / 5
def total_cost (pencils : ℕ) := 3 * cost_pen + pencils * cost_pencil

-- State the theorem
theorem cost_of_one_dozen_pens (pencils : ℕ) (h : total_cost pencils = 260) :
  12 * cost_pen = 780 :=
by
  -- Preamble to show/conclude that the proofs are given
  sorry

end cost_of_one_dozen_pens_l227_22734


namespace explicit_form_l227_22779

-- Define the functional equation
def f (x : ℝ) : ℝ := sorry

-- Define the condition that f(x) satisfies
axiom functional_equation (x : ℝ) (h : x ≠ 0) : f x = 2 * f (1 / x) + 3 * x

-- State the theorem that we need to prove
theorem explicit_form (x : ℝ) (h : x ≠ 0) : f x = -x - (2 / x) :=
by
  sorry

end explicit_form_l227_22779


namespace abc_product_le_two_l227_22792

theorem abc_product_le_two (a b c : ℝ) (ha : 0 ≤ a) (ha2 : a ≤ 2) (hb : 0 ≤ b) (hb2 : b ≤ 2) (hc : 0 ≤ c) (hc2 : c ≤ 2) :
  (a - b) * (b - c) * (a - c) ≤ 2 :=
sorry

end abc_product_le_two_l227_22792


namespace pioneer_ages_l227_22713

def pioneer_data (Burov Gridnev Klimenko Kolya Petya Grisha : String) (Burov_age Gridnev_age Klimenko_age Petya_age Grisha_age : ℕ) :=
  Burov ≠ Kolya ∧
  Petya_age = 12 ∧
  Gridnev_age = Petya_age + 1 ∧
  Grisha_age = Petya_age + 1 ∧
  Burov_age = Grisha_age ∧
-- defining the names corresponding to conditions given in problem
  Burov = Grisha ∧ Gridnev = Kolya ∧ Klimenko = Petya 

theorem pioneer_ages (Burov Gridnev Klimenko Kolya Petya Grisha : String) (Burov_age Gridnev_age Klimenko_age Petya_age Grisha_age : ℕ)
  (h : pioneer_data Burov Gridnev Klimenko Kolya Petya Grisha Burov_age Gridnev_age Klimenko_age Petya_age Grisha_age) :
  (Burov, Burov_age) = (Grisha, 13) ∧ 
  (Gridnev, Gridnev_age) = (Kolya, 13) ∧ 
  (Klimenko, Klimenko_age) = (Petya, 12) :=
by
  sorry

end pioneer_ages_l227_22713


namespace arithmetic_sequence_common_difference_l227_22744

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arith : ∀ n m, a (n + 1) - a n = a (m + 1) - a m)
  (h_a2 : a 2 = 3)
  (h_a7 : a 7 = 13) : 
  ∃ d, ∀ n, a n = a 1 + (n - 1) * d ∧ d = 2 := 
by
  sorry

end arithmetic_sequence_common_difference_l227_22744


namespace polygon_triangle_existence_l227_22739

theorem polygon_triangle_existence (n : ℕ) (h₁ : n > 1)
  (h₂ : ∀ (k₁ k₂ : ℕ), k₁ ≠ k₂ → (4 ≤ k₁) → (4 ≤ k₂) → k₁ ≠ k₂) :
  ∃ k, k = 3 :=
by
  sorry

end polygon_triangle_existence_l227_22739


namespace max_value_t_min_value_y_l227_22736

-- 1. Prove that the maximum value of t given |2x+5| + |2x-1| - t ≥ 0 is s = 6.
theorem max_value_t (t : ℝ) (x : ℝ) :
  (abs (2*x + 5) + abs (2*x - 1) - t ≥ 0) → (t ≤ 6) :=
by sorry

-- 2. Given s = 6 and 4a + 5b = s, prove that the minimum value of y = 1/(a+2b) + 4/(3a+3b) is y = 3/2.
theorem min_value_y (a b : ℝ) (s : ℝ) :
  s = 6 → (4*a + 5*b = s) → (a > 0) → (b > 0) → 
  (1/(a + 2*b) + 4/(3*a + 3*b) ≥ 3/2) :=
by sorry

end max_value_t_min_value_y_l227_22736


namespace Benjamin_skating_time_l227_22765

-- Definitions based on the conditions in the problem
def distance : ℕ := 80 -- Distance skated in kilometers
def speed : ℕ := 10 -- Speed in kilometers per hour

-- Theorem to prove that the skating time is 8 hours
theorem Benjamin_skating_time : distance / speed = 8 := by
  -- Proof goes here, we skip it with sorry
  sorry

end Benjamin_skating_time_l227_22765


namespace complement_correct_l227_22771

-- Define the universal set U
def U : Set ℤ := {x | -2 < x ∧ x ≤ 3}

-- Define the set A
def A : Set ℤ := {3}

-- Define the complement of A with respect to U
def complement_U_A : Set ℤ := {x | x ∈ U ∧ x ∉ A}

theorem complement_correct : complement_U_A = { -1, 0, 1, 2 } :=
by
  sorry

end complement_correct_l227_22771


namespace parabola_standard_equation_l227_22760

/-- Given that the directrix of a parabola coincides with the line on which the circles 
    x^2 + y^2 - 4 = 0 and x^2 + y^2 + y - 3 = 0 lie, the standard equation of the parabola 
    is x^2 = 4y.
-/
theorem parabola_standard_equation :
  (∀ x y : ℝ, x^2 + y^2 - 4 = 0 → x^2 + y^2 + y - 3 = 0 → y = -1) →
  ∀ p : ℝ, 4 * (p / 2) = 4 → x^2 = 4 * p * y :=
by
  sorry

end parabola_standard_equation_l227_22760


namespace meal_cost_is_seven_l227_22767

-- Defining the given conditions
def total_cost : ℕ := 21
def number_of_meals : ℕ := 3

-- The amount each meal costs
def meal_cost : ℕ := total_cost / number_of_meals

-- Prove that each meal costs 7 dollars given the conditions
theorem meal_cost_is_seven : meal_cost = 7 :=
by
  -- The result follows directly from the definition of meal_cost
  unfold meal_cost
  have h : 21 / 3 = 7 := by norm_num
  exact h


end meal_cost_is_seven_l227_22767


namespace barbed_wire_cost_l227_22738

theorem barbed_wire_cost
  (A : ℕ)          -- Area of the square field (sq m)
  (cost_per_meter : ℕ)  -- Cost per meter for the barbed wire (Rs)
  (gate_width : ℕ)      -- Width of each gate (m)
  (num_gates : ℕ)       -- Number of gates
  (side_length : ℕ)     -- Side length of the square field (m)
  (perimeter : ℕ)       -- Perimeter of the square field (m)
  (total_length : ℕ)    -- Total length of the barbed wire needed (m)
  (total_cost : ℕ)      -- Total cost of drawing the barbed wire (Rs)
  (h1 : A = 3136)       -- Given: Area = 3136 sq m
  (h2 : cost_per_meter = 1)  -- Given: Cost per meter = 1 Rs/m
  (h3 : gate_width = 1)      -- Given: Width of each gate = 1 m
  (h4 : num_gates = 2)       -- Given: Number of gates = 2
  (h5 : side_length * side_length = A)  -- Side length calculated from the area
  (h6 : perimeter = 4 * side_length)    -- Perimeter of the square field
  (h7 : total_length = perimeter - (num_gates * gate_width))  -- Actual barbed wire length after gates
  (h8 : total_cost = total_length * cost_per_meter)           -- Total cost calculation
  : total_cost = 222 :=      -- The result we need to prove
sorry

end barbed_wire_cost_l227_22738


namespace probability_of_red_second_given_red_first_l227_22786

-- Define the conditions as per the problem.
def total_balls := 5
def red_balls := 3
def yellow_balls := 2
def first_draw_red : ℚ := (red_balls : ℚ) / (total_balls : ℚ)
def both_draws_red : ℚ := (red_balls * (red_balls - 1)) / (total_balls * (total_balls - 1))

-- Define the probability of drawing a red ball in the second draw given the first was red.
def conditional_probability_red_second_given_first : ℚ :=
  both_draws_red / first_draw_red

-- The main statement to be proved.
theorem probability_of_red_second_given_red_first :
  conditional_probability_red_second_given_first = 1 / 2 :=
by
  sorry

end probability_of_red_second_given_red_first_l227_22786


namespace find_x_l227_22778

theorem find_x (x : ℝ) (h_pos : 0 < x) (h_eq : x * ⌊x⌋ = 48) : x = 8 :=
sorry

end find_x_l227_22778


namespace proposition_D_correct_l227_22700

theorem proposition_D_correct :
  ∀ x : ℝ, x^2 + x + 2 > 0 :=
by
  sorry

end proposition_D_correct_l227_22700


namespace inflation_two_years_real_rate_of_return_l227_22701

-- Proof Problem for Question 1
theorem inflation_two_years :
  ((1 + 0.015)^2 - 1) * 100 = 3.0225 :=
by
  sorry

-- Proof Problem for Question 2
theorem real_rate_of_return :
  ((1.07 * 1.07) / (1 + 0.030225) - 1) * 100 = 11.13 :=
by
  sorry

end inflation_two_years_real_rate_of_return_l227_22701


namespace solve_linear_system_l227_22790

theorem solve_linear_system :
  ∃ x y : ℚ, 7 * x = -10 - 3 * y ∧ 4 * x = 5 * y - 32 ∧ 
  x = -219 / 88 ∧ y = 97 / 22 :=
by
  sorry

end solve_linear_system_l227_22790


namespace basic_astrophysics_degrees_l227_22776

-- Define the given percentages
def microphotonics_percentage : ℝ := 14
def home_electronics_percentage : ℝ := 24
def food_additives_percentage : ℝ := 10
def gmo_percentage : ℝ := 29
def industrial_lubricants_percentage : ℝ := 8
def total_circle_degrees : ℝ := 360

-- Define a proof problem to show that basic astrophysics research occupies 54 degrees in the circle
theorem basic_astrophysics_degrees :
  total_circle_degrees - (microphotonics_percentage + home_electronics_percentage + food_additives_percentage + gmo_percentage + industrial_lubricants_percentage) = 15 ∧
  0.15 * total_circle_degrees = 54 :=
by
  sorry

end basic_astrophysics_degrees_l227_22776


namespace count_valid_n_l227_22774

theorem count_valid_n (n : ℕ) (h₁ : (n % 2015) ≠ 0) :
  (n^3 + 3^n) % 5 = 0 :=
by
  sorry

end count_valid_n_l227_22774


namespace arith_seq_a4_a10_l227_22788

variable {a : ℕ → ℕ}
axiom hp1 : a 1 + a 2 + a 3 = 32
axiom hp2 : a 11 + a 12 + a 13 = 118

theorem arith_seq_a4_a10 :
  a 4 + a 10 = 50 :=
by
  have h1 : a 2 = 32 / 3 := sorry
  have h2 : a 12 = 118 / 3 := sorry
  have h3 : a 2 + a 12 = 50 := sorry
  exact sorry

end arith_seq_a4_a10_l227_22788


namespace inequality_true_for_all_real_l227_22749

theorem inequality_true_for_all_real (a : ℝ) : 
  3 * (1 + a^2 + a^4) ≥ (1 + a + a^2)^2 :=
sorry

end inequality_true_for_all_real_l227_22749


namespace contrapositive_equivalence_l227_22742
-- Importing the necessary libraries

-- Declaring the variables P and Q as propositions
variables (P Q : Prop)

-- The statement that we need to prove
theorem contrapositive_equivalence :
  (P → ¬ Q) ↔ (Q → ¬ P) :=
sorry

end contrapositive_equivalence_l227_22742


namespace simplify_expression_l227_22727

theorem simplify_expression :
  2 + 3 / (4 + 5 / (6 + 7 / 8)) = 137 / 52 :=
by
  sorry

end simplify_expression_l227_22727


namespace coefficient_of_c_l227_22791

theorem coefficient_of_c (f c : ℝ) (h₁ : f = (9/5) * c + 32)
                         (h₂ : f + 25 = (9/5) * (c + 13.88888888888889) + 32) :
  (5/9) = (9/5) := sorry

end coefficient_of_c_l227_22791


namespace man_fraction_ownership_l227_22763

theorem man_fraction_ownership :
  ∀ (F : ℚ), (3 / 5 * F = 15000) → (75000 = 75000) → (F / 75000 = 1 / 3) :=
by
  intros F h1 h2
  sorry

end man_fraction_ownership_l227_22763


namespace expression_value_l227_22753

theorem expression_value (x y z : ℤ) (hx : x = -2) (hy : y = 1) (hz : z = 1) : 
  x^2 * y * z - x * y * z^2 = 6 :=
by
  rw [hx, hy, hz]
  rfl

end expression_value_l227_22753


namespace solve_for_angle_B_solutions_l227_22743

noncomputable def number_of_solutions_for_angle_B (BC AC : ℝ) (angle_A : ℝ) : ℕ :=
  if (BC = 6 ∧ AC = 8 ∧ angle_A = 40) then 2 else 0

theorem solve_for_angle_B_solutions : number_of_solutions_for_angle_B 6 8 40 = 2 :=
  by sorry

end solve_for_angle_B_solutions_l227_22743


namespace parabola_vertex_l227_22722

theorem parabola_vertex (x y : ℝ) : y^2 + 6*y + 2*x + 5 = 0 → (x, y) = (2, -3) :=
sorry

end parabola_vertex_l227_22722


namespace probability_of_event_correct_l227_22782

def within_interval (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ Real.pi

def tan_in_range (x : ℝ) : Prop :=
  -1 ≤ Real.tan x ∧ Real.tan x ≤ Real.sqrt 3

def valid_subintervals (x : ℝ) : Prop :=
  within_interval x ∧ tan_in_range x

def interval_length (a b : ℝ) : ℝ :=
  b - a

noncomputable def probability_of_event : ℝ :=
  (interval_length 0 (Real.pi / 3) + interval_length (3 * Real.pi / 4) Real.pi) / Real.pi

theorem probability_of_event_correct :
  probability_of_event = 7 / 12 := sorry

end probability_of_event_correct_l227_22782


namespace solve_equation_l227_22703

theorem solve_equation : ∀ (x : ℝ), 2 * (x - 1) = 2 - (5 * x - 2) → x = 6 / 7 :=
by
  sorry

end solve_equation_l227_22703


namespace method_1_more_cost_effective_l227_22711

open BigOperators

def racket_price : ℕ := 20
def shuttlecock_price : ℕ := 5
def rackets_bought : ℕ := 4
def shuttlecocks_bought : ℕ := 30
def discount_rate : ℚ := 0.92

def total_price (rackets shuttlecocks : ℕ) := racket_price * rackets + shuttlecock_price * shuttlecocks

def method_1_cost (rackets shuttlecocks : ℕ) := 
  total_price rackets shuttlecocks - shuttlecock_price * rackets

def method_2_cost (total : ℚ) :=
  total * discount_rate

theorem method_1_more_cost_effective :
  method_1_cost rackets_bought shuttlecocks_bought
  <
  method_2_cost (total_price rackets_bought shuttlecocks_bought) :=
by
  sorry

end method_1_more_cost_effective_l227_22711


namespace most_significant_price_drop_l227_22718

noncomputable def price_change (month : ℕ) : ℝ :=
  match month with
  | 1 => -1.00
  | 2 => 0.50
  | 3 => -3.00
  | 4 => 2.00
  | 5 => -1.50
  | 6 => -0.75
  | _ => 0.00 -- For any invalid month, we assume no price change

theorem most_significant_price_drop :
  ∀ m : ℕ, (m = 1 ∨ m = 2 ∨ m = 3 ∨ m = 4 ∨ m = 5 ∨ m = 6) →
  (∀ n : ℕ, (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6) →
  price_change m ≤ price_change n) → m = 3 :=
by
  intros m hm H
  sorry

end most_significant_price_drop_l227_22718


namespace total_students_l227_22748

-- Definition of variables and conditions
def M := 50
def E := 4 * M - 3

-- Statement of the theorem to prove
theorem total_students : E + M = 247 := by
  sorry

end total_students_l227_22748


namespace gcd_division_steps_l227_22702

theorem gcd_division_steps (a b : ℕ) (h₁ : a = 1813) (h₂ : b = 333) : 
  ∃ steps : ℕ, steps = 3 ∧ (Nat.gcd a b = 37) :=
by
  have h₁ : a = 1813 := h₁
  have h₂ : b = 333 := h₂
  sorry

end gcd_division_steps_l227_22702


namespace find_m_l227_22741

theorem find_m (m : ℕ) : (11 - m + 1 = 5) → m = 7 :=
by
  sorry

end find_m_l227_22741


namespace find_k_l227_22708

theorem find_k (a b c : ℝ) :
    (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) + (-1) * a * b * c :=
by
  sorry

end find_k_l227_22708


namespace least_positive_integer_special_property_l227_22754

theorem least_positive_integer_special_property : ∃ (N : ℕ) (a b c : ℕ), 
  N = 100 * a + 10 * b + c ∧ a ≠ 0 ∧ 10 * b + c = N / 29 ∧ N = 725 :=
by
  sorry

end least_positive_integer_special_property_l227_22754


namespace brady_earns_181_l227_22789

def bradyEarnings (basic_count : ℕ) (gourmet_count : ℕ) (total_cards : ℕ) : ℕ :=
  let basic_earnings := basic_count * 70
  let gourmet_earnings := gourmet_count * 90
  let total_earnings := basic_earnings + gourmet_earnings
  let total_bonus := (total_cards / 100) * 10 + ((total_cards / 100) - 1) * 5
  total_earnings + total_bonus

theorem brady_earns_181 :
  bradyEarnings 120 80 200 = 181 :=
by 
  sorry

end brady_earns_181_l227_22789


namespace proof_l227_22798

noncomputable def problem_statement : Prop :=
  ( ( (Real.sqrt 1.21 * Real.sqrt 1.44) / (Real.sqrt 0.81 * Real.sqrt 0.64)
    + (Real.sqrt 1.0 * Real.sqrt 3.24) / (Real.sqrt 0.49 * Real.sqrt 2.25) ) ^ 3 
  = 44.6877470366 )

theorem proof : problem_statement := 
  by
  sorry

end proof_l227_22798


namespace minimum_value_of_y_l227_22755

theorem minimum_value_of_y (x : ℝ) (h : x > 0) : (∃ y, y = (x^2 + 1) / x ∧ y ≥ 2) ∧ (∃ y, y = (x^2 + 1) / x ∧ y = 2) :=
by
  sorry

end minimum_value_of_y_l227_22755


namespace midpoint_coordinate_sum_l227_22714

theorem midpoint_coordinate_sum
  (x1 y1 x2 y2 : ℝ)
  (h1 : x1 = 10)
  (h2 : y1 = 3)
  (h3 : x2 = 4)
  (h4 : y2 = -3) :
  let xm := (x1 + x2) / 2
  let ym := (y1 + y2) / 2
  xm + ym =  7 := by
  sorry

end midpoint_coordinate_sum_l227_22714


namespace num_ordered_triples_pos_int_l227_22729

theorem num_ordered_triples_pos_int
  (lcm_ab: lcm a b = 180)
  (lcm_ac: lcm a c = 450)
  (lcm_bc: lcm b c = 1200)
  (gcd_abc: gcd (gcd a b) c = 3) :
  ∃ n: ℕ, n = 4 :=
sorry

end num_ordered_triples_pos_int_l227_22729


namespace quadratic_solution_sum_l227_22759

theorem quadratic_solution_sum (m n p : ℕ) (h : m.gcd (n.gcd p) = 1)
  (h₀ : ∀ x, x * (5 * x - 11) = -6 ↔ x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p) :
  m + n + p = 70 :=
sorry

end quadratic_solution_sum_l227_22759


namespace right_triangle_hypotenuse_l227_22747

-- Define the right triangle conditions and hypotenuse calculation
theorem right_triangle_hypotenuse (a b c : ℝ) (h1 : b = a + 3) (h2 : 1 / 2 * a * b = 120) :
  c^2 = 425 :=
by
  sorry

end right_triangle_hypotenuse_l227_22747


namespace number_of_k_solutions_l227_22777

theorem number_of_k_solutions :
  ∃ (n : ℕ), n = 1006 ∧
  (∀ k, (∃ a b : ℕ+, (a ≠ b) ∧ (k * (a + b) = 2013 * Nat.lcm a b)) ↔ k ≤ n ∧ 0 < k) :=
by
  sorry

end number_of_k_solutions_l227_22777


namespace laura_needs_to_buy_flour_l227_22716

/--
Laura is baking a cake and needs to buy ingredients.
Flour costs $4, sugar costs $2, butter costs $2.5, and eggs cost $0.5.
The cake is cut into 6 slices. Her mother ate 2 slices.
The dog ate the remaining cake, costing $6.
Prove that Laura needs to buy flour worth $4.
-/
theorem laura_needs_to_buy_flour
  (flour_cost sugar_cost butter_cost eggs_cost dog_ate_cost : ℝ)
  (cake_slices mother_ate_slices dog_ate_slices : ℕ)
  (H_flour : flour_cost = 4)
  (H_sugar : sugar_cost = 2)
  (H_butter : butter_cost = 2.5)
  (H_eggs : eggs_cost = 0.5)
  (H_dog_ate : dog_ate_cost = 6)
  (total_slices : cake_slices = 6)
  (mother_slices : mother_ate_slices = 2)
  (dog_slices : dog_ate_slices = 4) :
  flour_cost = 4 :=
by {
  sorry
}

end laura_needs_to_buy_flour_l227_22716


namespace minimum_value_of_n_l227_22732

open Int

theorem minimum_value_of_n (n d : ℕ) (h1 : n > 0) (h2 : d > 0) (h3 : d % n = 0)
    (h4 : 10 * n - 20 = 90) : n = 11 :=
by
  sorry

end minimum_value_of_n_l227_22732


namespace isosceles_triangle_perimeter_l227_22750

theorem isosceles_triangle_perimeter (a b : ℝ) (h_iso : a = 4 ∨ b = 4) (h_iso2 : a = 8 ∨ b = 8) : 
  (a = 4 ∧ b = 8 ∧ 4 + a + b = 16 ∨ 
  a = 4 ∧ b = 8 ∧ b + a + a = 20 ∨ 
  a = 8 ∧ b = 4 ∧ a + a + b = 20) :=
by sorry

end isosceles_triangle_perimeter_l227_22750


namespace car_quotient_div_15_l227_22751

/-- On a straight, one-way, single-lane highway, cars all travel at the same speed
    and obey a modified safety rule: the distance from the back of the car ahead
    to the front of the car behind is exactly two car lengths for each 20 kilometers
    per hour of speed. A sensor by the road counts the number of cars that pass in
    one hour. Each car is 5 meters long. 
    Let N be the maximum whole number of cars that can pass the sensor in one hour.
    Prove that when N is divided by 15, the quotient is 266. -/
theorem car_quotient_div_15 
  (speed : ℕ) 
  (d : ℕ) 
  (sensor_time : ℕ) 
  (car_length : ℕ)
  (N : ℕ)
  (h1 : ∀ m, speed = 20 * m)
  (h2 : d = 2 * car_length)
  (h3 : car_length = 5)
  (h4 : sensor_time = 1)
  (h5 : N = 4000) : 
  N / 15 = 266 := 
sorry

end car_quotient_div_15_l227_22751


namespace find_certain_number_l227_22795

theorem find_certain_number (d q r : ℕ) (HD : d = 37) (HQ : q = 23) (HR : r = 16) :
    ∃ n : ℕ, n = d * q + r ∧ n = 867 := by
  sorry

end find_certain_number_l227_22795


namespace system_of_equations_solution_l227_22775

theorem system_of_equations_solution (x y z u v : ℤ) 
  (h1 : x + y + z + u = 5)
  (h2 : y + z + u + v = 1)
  (h3 : z + u + v + x = 2)
  (h4 : u + v + x + y = 0)
  (h5 : v + x + y + z = 4) :
  v = -2 ∧ x = 2 ∧ y = 1 ∧ z = 3 ∧ u = -1 := 
by 
  sorry

end system_of_equations_solution_l227_22775


namespace age_ratio_in_4_years_l227_22781

variable {p k x : ℕ}

theorem age_ratio_in_4_years (h₁ : p - 8 = 2 * (k - 8)) (h₂ : p - 14 = 3 * (k - 14)) : x = 4 :=
by
  sorry

end age_ratio_in_4_years_l227_22781


namespace program_output_l227_22766

-- Define the initial conditions
def initial_a := 1
def initial_b := 3

-- Define the program transformations
def a_step1 (a b : ℕ) := a + b
def b_step2 (a b : ℕ) := a - b

-- Define the final values after program execution
def final_a := a_step1 initial_a initial_b
def final_b := b_step2 final_a initial_b

-- Statement to prove
theorem program_output :
  final_a = 4 ∧ final_b = 1 :=
by
  -- Placeholder for the actual proof
  sorry

end program_output_l227_22766


namespace solve_for_x_l227_22761

theorem solve_for_x (x y : ℝ) (h1 : 3 * x - 2 * y = 7) (h2 : x + 3 * y = 6) : x = 3 := 
by 
  sorry

end solve_for_x_l227_22761


namespace initial_range_calculation_l227_22745

variable (initial_range telescope_range : ℝ)
variable (increased_by : ℝ)
variable (h_telescope : telescope_range = increased_by * initial_range)

theorem initial_range_calculation 
  (h_telescope_range : telescope_range = 150)
  (h_increased_by : increased_by = 3)
  (h_telescope : telescope_range = increased_by * initial_range) :
  initial_range = 50 :=
  sorry

end initial_range_calculation_l227_22745


namespace dirk_profit_l227_22712

theorem dirk_profit 
  (days : ℕ) 
  (amulets_per_day : ℕ) 
  (sale_price : ℕ) 
  (cost_price : ℕ) 
  (cut_percentage : ℕ) 
  (profit : ℕ) : 
  days = 2 → amulets_per_day = 25 → sale_price = 40 → cost_price = 30 → cut_percentage = 10 → profit = 300 :=
by
  intros h_days h_amulets_per_day h_sale_price h_cost_price h_cut_percentage
  -- Placeholder for the proof
  sorry

end dirk_profit_l227_22712


namespace richard_remaining_distance_l227_22746

noncomputable def remaining_distance : ℝ :=
  let d1 := 45
  let d2 := d1 / 2 - 8
  let d3 := 2 * d2 - 4
  let d4 := (d1 + d2 + d3) / 3 + 3
  let d5 := 0.7 * d4
  let total_walked := d1 + d2 + d3 + d4 + d5
  635 - total_walked

theorem richard_remaining_distance : abs (remaining_distance - 497.5166) < 0.0001 :=
by
  sorry

end richard_remaining_distance_l227_22746
