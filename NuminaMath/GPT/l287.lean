import Mathlib

namespace NUMINAMATH_GPT_product_of_two_large_integers_l287_28789

theorem product_of_two_large_integers :
  ∃ a b : ℕ, a > 2009^182 ∧ b > 2009^182 ∧ 3^2008 + 4^2009 = a * b :=
by { sorry }

end NUMINAMATH_GPT_product_of_two_large_integers_l287_28789


namespace NUMINAMATH_GPT_graph_represents_two_intersecting_lines_l287_28777

theorem graph_represents_two_intersecting_lines (x y : ℝ) :
  (x - 1) * (x + y + 2) = (y - 1) * (x + y + 2) → 
  (x + y + 2 = 0 ∨ x = y) ∧ 
  (∃ (x y : ℝ), (x = -1 ∧ y = -1 ∧ x = y ∨ x = -y - 2) ∧ (y = x ∨ y = -x - 2)) :=
by
  sorry

end NUMINAMATH_GPT_graph_represents_two_intersecting_lines_l287_28777


namespace NUMINAMATH_GPT_cone_volume_l287_28787

theorem cone_volume (d : ℝ) (h : ℝ) (π : ℝ) (volume : ℝ) 
  (hd : d = 10) (hh : h = 0.8 * d) (hπ : π = Real.pi) : 
  volume = (200 / 3) * π :=
by
  sorry

end NUMINAMATH_GPT_cone_volume_l287_28787


namespace NUMINAMATH_GPT_completing_the_square_step_l287_28731

theorem completing_the_square_step (x : ℝ) : 
  x^2 + 4 * x + 2 = 0 → x^2 + 4 * x = -2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_completing_the_square_step_l287_28731


namespace NUMINAMATH_GPT_rectangle_perimeter_ratio_l287_28779

theorem rectangle_perimeter_ratio
    (initial_height : ℕ)
    (initial_width : ℕ)
    (H_initial_height : initial_height = 2)
    (H_initial_width : initial_width = 4)
    (fold1_height : ℕ)
    (fold1_width : ℕ)
    (H_fold1_height : fold1_height = initial_height / 2)
    (H_fold1_width : fold1_width = initial_width)
    (fold2_height : ℕ)
    (fold2_width : ℕ)
    (H_fold2_height : fold2_height = fold1_height)
    (H_fold2_width : fold2_width = fold1_width / 2)
    (cut_height : ℕ)
    (cut_width : ℕ)
    (H_cut_height : cut_height = fold2_height)
    (H_cut_width : cut_width = fold2_width) :
    (2 * (cut_height + cut_width)) / (2 * (fold1_height + fold1_width)) = 3 / 5 := 
    by sorry

end NUMINAMATH_GPT_rectangle_perimeter_ratio_l287_28779


namespace NUMINAMATH_GPT_log_inequality_l287_28794

theorem log_inequality (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) :
  2 * ((Real.log a / Real.log b) / (a + b) + (Real.log b / Real.log c) / (b + c) + (Real.log c / Real.log a) / (c + a)) 
    ≥ 9 / (a + b + c) :=
by
  sorry

end NUMINAMATH_GPT_log_inequality_l287_28794


namespace NUMINAMATH_GPT_math_problem_l287_28759

noncomputable def a : ℝ := (0.96)^3 
noncomputable def b : ℝ := (0.1)^3 
noncomputable def c : ℝ := (0.96)^2 
noncomputable def d : ℝ := (0.1)^2 

theorem math_problem : a - b / c + 0.096 + d = 0.989651 := 
by 
  -- skip proof 
  sorry

end NUMINAMATH_GPT_math_problem_l287_28759


namespace NUMINAMATH_GPT_insert_zeros_between_digits_is_cube_l287_28733

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n

theorem insert_zeros_between_digits_is_cube (k b : ℕ) (h_b : b ≥ 4) 
  : is_perfect_cube (1 * b^(3*(1+k)) + 3 * b^(2*(1+k)) + 3 * b^(1+k) + 1) :=
sorry

end NUMINAMATH_GPT_insert_zeros_between_digits_is_cube_l287_28733


namespace NUMINAMATH_GPT_directrix_of_parabola_l287_28766

theorem directrix_of_parabola (x y : ℝ) : y = 3 * x^2 - 6 * x + 1 → y = -25 / 12 :=
sorry

end NUMINAMATH_GPT_directrix_of_parabola_l287_28766


namespace NUMINAMATH_GPT_smallest_positive_period_l287_28795

open Real

-- Define conditions
def max_value_condition (b a : ℝ) : Prop := b + a = -1
def min_value_condition (b a : ℝ) : Prop := b - a = -5

-- Define the period of the function
def period (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

-- Main theorem
theorem smallest_positive_period (a b : ℝ) (h1 : a < 0) 
  (h2 : max_value_condition b a) 
  (h3 : min_value_condition b a) : 
  period (fun x => tan ((3 * a + b) * x)) (π / 9) :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_period_l287_28795


namespace NUMINAMATH_GPT_max_ab_bc_ca_a2_div_bc_plus_b2_div_ca_plus_c2_div_ab_geq_1_over_2_l287_28769

variable (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
variable (h : a + b + c = 1)

theorem max_ab_bc_ca : ab + bc + ca ≤ 1 / 3 :=
by sorry

theorem a2_div_bc_plus_b2_div_ca_plus_c2_div_ab_geq_1_over_2 :
  (a^2 / (b + c)) + (b^2 / (c + a)) + (c^2 / (a + b)) ≥ 1 / 2 :=
by sorry

end NUMINAMATH_GPT_max_ab_bc_ca_a2_div_bc_plus_b2_div_ca_plus_c2_div_ab_geq_1_over_2_l287_28769


namespace NUMINAMATH_GPT_find_expression_max_value_min_value_l287_28703

namespace MathProblem

-- Define the function f(x) with parameters a and b
def f (a b x : ℝ) : ℝ := a * x^2 + a^2 * x + 2 * b - a^3

-- Hypotheses based on problem conditions
lemma a_neg (a b : ℝ) : a < 0 := sorry
lemma root_neg2 (a b : ℝ) : f a b (-2) = 0 := sorry
lemma root_6 (a b : ℝ) : f a b 6 = 0 := sorry

-- Proving the explicit expression for f(x)
theorem find_expression (a b : ℝ) (x : ℝ) : 
  a = -4 → 
  b = -8 → 
  f a b x = -4 * x^2 + 16 * x + 48 :=
sorry

-- Maximum value of f(x) on the interval [1, 10]
theorem max_value (x : ℝ) (h : 1 ≤ x ∧ x ≤ 10) : 
  f (-4) (-8) 2 = 64 :=
sorry

-- Minimum value of f(x) on the interval [1, 10]
theorem min_value (x : ℝ) (h : 1 ≤ x ∧ x ≤ 10) : 
  f (-4) (-8) 10 = -192 :=
sorry

end MathProblem

end NUMINAMATH_GPT_find_expression_max_value_min_value_l287_28703


namespace NUMINAMATH_GPT_closest_perfect_square_to_315_l287_28706

theorem closest_perfect_square_to_315 : ∃ n : ℤ, n^2 = 324 ∧
  (∀ m : ℤ, m ≠ n → (abs (315 - m^2) > abs (315 - n^2))) := 
sorry

end NUMINAMATH_GPT_closest_perfect_square_to_315_l287_28706


namespace NUMINAMATH_GPT_Sawyer_cleans_in_6_hours_l287_28773

theorem Sawyer_cleans_in_6_hours (N : ℝ) (S : ℝ) (h1 : S = (2/3) * N) 
                                 (h2 : 1/S + 1/N = 1/3.6) : S = 6 :=
by
  sorry

end NUMINAMATH_GPT_Sawyer_cleans_in_6_hours_l287_28773


namespace NUMINAMATH_GPT_events_related_with_99_confidence_l287_28738

theorem events_related_with_99_confidence (K_squared : ℝ) (h : K_squared > 6.635) : 
  events_A_B_related_with_99_confidence :=
sorry

end NUMINAMATH_GPT_events_related_with_99_confidence_l287_28738


namespace NUMINAMATH_GPT_find_a_l287_28764

theorem find_a (a : ℝ) :
  (∀ x, x < 2 → 0 < a - 3 * x) ↔ (a = 6) :=
by
  sorry

end NUMINAMATH_GPT_find_a_l287_28764


namespace NUMINAMATH_GPT_yogurt_amount_l287_28714

-- Conditions
def total_ingredients : ℝ := 0.5
def strawberries : ℝ := 0.2
def orange_juice : ℝ := 0.2

-- Question and Answer (Proof Goal)
theorem yogurt_amount : total_ingredients - strawberries - orange_juice = 0.1 := by
  -- Since calculation involves specifics, we add sorry to indicate the proof is skipped
  sorry

end NUMINAMATH_GPT_yogurt_amount_l287_28714


namespace NUMINAMATH_GPT_solution_is_correct_l287_28782

noncomputable def satisfies_inequality (x y : ℝ) : Prop := 
  x + 3 * y + 14 ≤ 0

noncomputable def satisfies_equation (x y : ℝ) : Prop := 
  x^4 + 2 * x^2 * y^2 + y^4 + 64 - 20 * x^2 - 20 * y^2 = 8 * x * y

theorem solution_is_correct : satisfies_inequality (-2) (-4) ∧ satisfies_equation (-2) (-4) :=
  by sorry

end NUMINAMATH_GPT_solution_is_correct_l287_28782


namespace NUMINAMATH_GPT_total_seeds_l287_28774

theorem total_seeds (seeds_per_watermelon : ℕ) (number_of_watermelons : ℕ) 
(seeds_each : seeds_per_watermelon = 100)
(watermelons_count : number_of_watermelons = 4) :
(seeds_per_watermelon * number_of_watermelons) = 400 := by
  sorry

end NUMINAMATH_GPT_total_seeds_l287_28774


namespace NUMINAMATH_GPT_find_base_k_l287_28707

theorem find_base_k (k : ℕ) (h1 : 1 + 3 * k + 2 * k^2 = 30) : k = 4 :=
by sorry

end NUMINAMATH_GPT_find_base_k_l287_28707


namespace NUMINAMATH_GPT_product_remainder_mod_5_l287_28742

theorem product_remainder_mod_5 :
  (1024 * 1455 * 1776 * 2018 * 2222) % 5 = 0 := 
sorry

end NUMINAMATH_GPT_product_remainder_mod_5_l287_28742


namespace NUMINAMATH_GPT_calc_product_l287_28734

def x : ℝ := 150.15
def y : ℝ := 12.01
def z : ℝ := 1500.15
def w : ℝ := 12

theorem calc_product :
  x * y * z * w = 32467532.8227 :=
by
  sorry

end NUMINAMATH_GPT_calc_product_l287_28734


namespace NUMINAMATH_GPT_find_g_values_l287_28770

open Function

-- Defining the function g and its properties
axiom g : ℝ → ℝ
axiom g_domain : ∀ x, 0 ≤ x → 0 ≤ g x
axiom g_proper : ∀ x, 0 ≤ x → 0 ≤ g (g x)
axiom g_func : ∀ x, 0 ≤ x → g (g x) = 3 * x / (x + 3)
axiom g_interval : ∀ x, 2 ≤ x ∧ x ≤ 3 → g x = (x + 1) / 2

-- Problem statement translating to Lean
theorem find_g_values :
  g 2021 = 2021.5 ∧ g (1 / 2021) = 6 := by {
  sorry 
}

end NUMINAMATH_GPT_find_g_values_l287_28770


namespace NUMINAMATH_GPT_year_2023_ad_is_written_as_positive_2023_l287_28741

theorem year_2023_ad_is_written_as_positive_2023 :
  (∀ (year : Int), year = -500 → year = -500) → -- This represents the given condition that year 500 BC is -500
  (∀ (year : Int), year > 0) → -- This represents the condition that AD years are postive
  2023 = 2023 := -- The problem conclusion

by
  intros
  trivial -- The solution is quite trivial due to the conditions.

end NUMINAMATH_GPT_year_2023_ad_is_written_as_positive_2023_l287_28741


namespace NUMINAMATH_GPT_simplify_expression_and_find_ratio_l287_28785

theorem simplify_expression_and_find_ratio:
  ∀ (k : ℤ), (∃ (a b : ℤ), (a = 1 ∧ b = 3) ∧ (6 * k + 18 = 6 * (a * k + b))) →
  (1 : ℤ) / (3 : ℤ) = (1 : ℤ) / (3 : ℤ) :=
by
  intro k
  intro h
  sorry

end NUMINAMATH_GPT_simplify_expression_and_find_ratio_l287_28785


namespace NUMINAMATH_GPT_ants_on_track_l287_28722

/-- Given that ants move on a circular track of length 60 cm at a speed of 1 cm/s
and that there are 48 pairwise collisions in a minute, prove that the possible 
total number of ants on the track is 10, 11, 14, or 25. -/
theorem ants_on_track (x y : ℕ) (h : x * y = 24) : x + y = 10 ∨ x + y = 11 ∨ x + y = 14 ∨ x + y = 25 :=
by sorry

end NUMINAMATH_GPT_ants_on_track_l287_28722


namespace NUMINAMATH_GPT_area_of_circle_with_endpoints_l287_28736

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

noncomputable def radius (d : ℝ) : ℝ :=
  d / 2

noncomputable def area_of_circle (r : ℝ) : ℝ :=
  Real.pi * r^2

theorem area_of_circle_with_endpoints :
  area_of_circle (radius (distance (5, 9) (13, 17))) = 32 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_area_of_circle_with_endpoints_l287_28736


namespace NUMINAMATH_GPT_total_corn_cobs_l287_28780

-- Definitions for the conditions
def rows_first_field : ℕ := 13
def rows_second_field : ℕ := 16
def cobs_per_row : ℕ := 4

-- Statement to prove
theorem total_corn_cobs : (rows_first_field * cobs_per_row + rows_second_field * cobs_per_row) = 116 :=
by sorry

end NUMINAMATH_GPT_total_corn_cobs_l287_28780


namespace NUMINAMATH_GPT_product_of_roots_eq_negative_forty_nine_l287_28719

theorem product_of_roots_eq_negative_forty_nine (t : ℝ) (h : t^2 = 49) : (7 * -7) = -49 :=
by
  sorry

end NUMINAMATH_GPT_product_of_roots_eq_negative_forty_nine_l287_28719


namespace NUMINAMATH_GPT_intersection_A_B_l287_28757

open Set Real

def A := { x : ℝ | x ^ 2 - 6 * x + 5 ≤ 0 }
def B := { x : ℝ | ∃ y : ℝ, y = log (x - 2) / log 2 }

theorem intersection_A_B : A ∩ B = { x : ℝ | 2 < x ∧ x ≤ 5 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l287_28757


namespace NUMINAMATH_GPT_friends_total_l287_28717

-- Define the conditions as constants
def can_go : Nat := 8
def can't_go : Nat := 7

-- Define the total number of friends and the correct answer
def total_friends : Nat := can_go + can't_go
def correct_answer : Nat := 15

-- Prove that the total number of friends is 15
theorem friends_total : total_friends = correct_answer := by
  -- We use the definitions and the conditions directly here
  sorry

end NUMINAMATH_GPT_friends_total_l287_28717


namespace NUMINAMATH_GPT_tree_count_l287_28726

theorem tree_count (m N : ℕ) 
  (h1 : 12 ≡ (33 - m) [MOD N])
  (h2 : (105 - m) ≡ 8 [MOD N]) :
  N = 76 := 
sorry

end NUMINAMATH_GPT_tree_count_l287_28726


namespace NUMINAMATH_GPT_total_number_of_bricks_l287_28753

/-- Given bricks of volume 80 unit cubes and 42 unit cubes,
 and a box of volume 1540 unit cubes,
 prove the total number of bricks that can fill the box exactly is 24. -/
theorem total_number_of_bricks (x y : ℕ) (vol_a vol_b total_vol : ℕ)
  (vol_a_def : vol_a = 80)
  (vol_b_def : vol_b = 42)
  (total_vol_def : total_vol = 1540)
  (volume_filled : x * vol_a + y * vol_b = total_vol) :
  x + y = 24 :=
  sorry

end NUMINAMATH_GPT_total_number_of_bricks_l287_28753


namespace NUMINAMATH_GPT_average_marks_l287_28746

-- Define the conditions
variables (M P C : ℝ)
variables (h1 : M + P = 60) (h2 : C = P + 10)

-- Define the theorem statement
theorem average_marks : (M + C) / 2 = 35 :=
by {
  sorry -- Placeholder for the proof.
}

end NUMINAMATH_GPT_average_marks_l287_28746


namespace NUMINAMATH_GPT_compute_r_l287_28798

noncomputable def r (side_length : ℝ) : ℝ :=
  let a := (0.5 * side_length, 0.5 * side_length)
  let b := (1.5 * side_length, 2.5 * side_length)
  let c := (2.5 * side_length, 1.5 * side_length)
  let ab := Real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)
  let ac := Real.sqrt ((c.1 - a.1)^2 + (c.2 - a.2)^2)
  let bc := Real.sqrt ((c.1 - b.1)^2 + (c.2 - b.2)^2)
  let s := (ab + ac + bc) / 2
  let area_ABC := Real.sqrt (s * (s - ab) * (s - ac) * (s - bc))
  let circumradius := ab * ac * bc / (4 * area_ABC)
  circumradius - (side_length / 2)

theorem compute_r :
  r 1 = (5 * Real.sqrt 2 - 3) / 6 :=
by
  unfold r
  sorry

end NUMINAMATH_GPT_compute_r_l287_28798


namespace NUMINAMATH_GPT_complex_multiplication_l287_28710

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (2 - i) = 1 + 2 * i :=
by
  sorry

end NUMINAMATH_GPT_complex_multiplication_l287_28710


namespace NUMINAMATH_GPT_largest_integral_value_l287_28792

theorem largest_integral_value (x : ℤ) : (1 / 3 : ℚ) < x / 5 ∧ x / 5 < 5 / 8 → x = 3 :=
by
  sorry

end NUMINAMATH_GPT_largest_integral_value_l287_28792


namespace NUMINAMATH_GPT_find_u_l287_28763

theorem find_u 
    (a b c p q u : ℝ) 
    (H₁: (∀ x, x^3 + 2*x^2 + 5*x - 8 = 0 → x = a ∨ x = b ∨ x = c))
    (H₂: (∀ x, x^3 + p*x^2 + q*x + u = 0 → x = a+b ∨ x = b+c ∨ x = c+a)) :
    u = 18 :=
by 
    sorry

end NUMINAMATH_GPT_find_u_l287_28763


namespace NUMINAMATH_GPT_no_such_number_exists_l287_28702

-- Definitions for conditions
def base_5_digit_number (x : ℕ) : Prop := 
  ∀ n, 0 ≤ n ∧ n < 2023 → x / 5^n % 5 < 5

def odd_plus_one (n m : ℕ) : Prop :=
  (∀ k < 1012, (n / 5^(2*k) % 25 / 5 = m / 5^(2*k) % 25 / 5 + 1)) ∧
  (∀ k < 1011, (n / 5^(2*k+1) % 25 / 5 = m / 5^(2*k+1) % 25 / 5 - 1))

def has_two_prime_factors_that_differ_by_two (x : ℕ) : Prop :=
  ∃ u v, u * v = x ∧ Prime u ∧ Prime v ∧ v = u + 2

-- Combined conditions for the hypothesized number x
def hypothesized_number (x : ℕ) : Prop := 
  base_5_digit_number x ∧
  odd_plus_one x x ∧
  has_two_prime_factors_that_differ_by_two x

-- The proof statement that the hypothesized number cannot exist
theorem no_such_number_exists : ¬ ∃ x, hypothesized_number x :=
by
  sorry

end NUMINAMATH_GPT_no_such_number_exists_l287_28702


namespace NUMINAMATH_GPT_polygon_edges_of_set_S_l287_28716

variable (a : ℝ)

def in_set_S(x y : ℝ) : Prop :=
  (a / 2 ≤ x ∧ x ≤ 2 * a) ∧
  (a / 2 ≤ y ∧ y ≤ 2 * a) ∧
  (x + y ≥ a) ∧
  (x + a ≥ y) ∧
  (y + a ≥ x)

theorem polygon_edges_of_set_S (a : ℝ) (h : 0 < a) :
  (∃ n, ∀ x y, in_set_S a x y → n = 6) :=
sorry

end NUMINAMATH_GPT_polygon_edges_of_set_S_l287_28716


namespace NUMINAMATH_GPT_problem1_problem2_l287_28729

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x + a^2 / x
noncomputable def g (x : ℝ) : ℝ := x + Real.log x
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := f x a + g x

-- Problem 1: Prove that a = sqrt(3) given that x = 1 is an extremum point for h(x, a)
theorem problem1 (a : ℝ) (h_extremum : ∀ x : ℝ, x = 1 → 0 = (2 - a^2 / x^2 + 1 / x)) : a = Real.sqrt 3 := sorry

-- Problem 2: Prove the range of a is [ (e + 1) / 2, +∞ ) such that for any x1, x2 ∈ [1, e], f(x1, a) ≥ g(x2)
theorem problem2 (a : ℝ) :
  (∀ x1 x2 : ℝ, 1 ≤ x1 ∧ x1 ≤ Real.exp 1 ∧ 1 ≤ x2 ∧ x2 ≤ Real.exp 1 → f x1 a ≥ g x2) →
  (Real.exp 1 + 1) / 2 ≤ a :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l287_28729


namespace NUMINAMATH_GPT_total_number_of_water_filled_jars_l287_28781

theorem total_number_of_water_filled_jars : 
  ∃ (x : ℕ), 28 = x * (1/4 + 1/2 + 1) ∧ 3 * x = 48 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_water_filled_jars_l287_28781


namespace NUMINAMATH_GPT_tan_double_angle_l287_28728

theorem tan_double_angle (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin (α - π / 2) = 3 / 5) : 
  Real.tan (2 * α) = 24 / 7 :=
by
  sorry

end NUMINAMATH_GPT_tan_double_angle_l287_28728


namespace NUMINAMATH_GPT_pies_sold_in_week_l287_28721

def daily_pies := 8
def days_in_week := 7
def total_pies := 56

theorem pies_sold_in_week : daily_pies * days_in_week = total_pies :=
by
  sorry

end NUMINAMATH_GPT_pies_sold_in_week_l287_28721


namespace NUMINAMATH_GPT_max_blocks_that_fit_l287_28720

noncomputable def box_volume : ℕ :=
  3 * 4 * 2

noncomputable def block_volume : ℕ :=
  2 * 1 * 2

noncomputable def max_blocks (box_volume : ℕ) (block_volume : ℕ) : ℕ :=
  box_volume / block_volume

theorem max_blocks_that_fit : max_blocks box_volume block_volume = 6 :=
by
  sorry

end NUMINAMATH_GPT_max_blocks_that_fit_l287_28720


namespace NUMINAMATH_GPT_range_of_m_l287_28767

-- Define the conditions based on the problem statement
def equation (x m : ℝ) : Prop := (2 * x + m) = (x - 1)

-- The goal is to prove that if there exists a positive solution x to the equation, then m < -1
theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, equation x m ∧ x > 0) → m < -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l287_28767


namespace NUMINAMATH_GPT_quadratic_to_vertex_form_l287_28735

-- Define the given quadratic function.
def quadratic_function (x : ℝ) : ℝ := x^2 - 2 * x + 3

-- Define the vertex form of the quadratic function.
def vertex_form (x : ℝ) : ℝ := (x - 1)^2 + 2

-- State the equivalence we want to prove.
theorem quadratic_to_vertex_form :
  ∀ x : ℝ, quadratic_function x = vertex_form x :=
by
  intro x
  show quadratic_function x = vertex_form x
  sorry

end NUMINAMATH_GPT_quadratic_to_vertex_form_l287_28735


namespace NUMINAMATH_GPT_maria_total_cost_l287_28747

-- Define the costs of the items
def pencil_cost : ℕ := 8
def pen_cost : ℕ := pencil_cost / 2
def eraser_cost : ℕ := 2 * pen_cost

-- Define the total cost
def total_cost : ℕ := pen_cost + pencil_cost + eraser_cost

-- The theorem to prove
theorem maria_total_cost : total_cost = 20 := by
  sorry

end NUMINAMATH_GPT_maria_total_cost_l287_28747


namespace NUMINAMATH_GPT_simplify_expression_l287_28732

variable (a b : ℤ)

theorem simplify_expression : 
  (32 * a + 45 * b) + (15 * a + 36 * b) - (27 * a + 41 * b) = 20 * a + 40 * b := 
by sorry

end NUMINAMATH_GPT_simplify_expression_l287_28732


namespace NUMINAMATH_GPT_remainder_of_division_l287_28799

theorem remainder_of_division (d : ℝ) (q : ℝ) (r : ℝ) : 
  d = 187.46067415730337 → q = 89 → 16698 = (d * q) + r → r = 14 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  simp at h3
  sorry

end NUMINAMATH_GPT_remainder_of_division_l287_28799


namespace NUMINAMATH_GPT_probability_neither_square_nor_cube_l287_28739

theorem probability_neither_square_nor_cube :
  let count_squares := 14
  let count_cubes := 5
  let overlap := 2
  let total_range := 200
  let neither_count := total_range - (count_squares + count_cubes - overlap)
  let probability := (neither_count : ℚ) / total_range
  probability = 183 / 200 :=
by {
  sorry
}

end NUMINAMATH_GPT_probability_neither_square_nor_cube_l287_28739


namespace NUMINAMATH_GPT_greater_number_l287_28705

theorem greater_number (x y : ℕ) (h1 : x * y = 2048) (h2 : x + y - (x - y) = 64) : x = 64 :=
by
  sorry

end NUMINAMATH_GPT_greater_number_l287_28705


namespace NUMINAMATH_GPT_phone_price_is_correct_l287_28718

-- Definition of the conditions
def monthly_cost := 7
def months := 4
def total_cost := 30

-- Definition to be proven
def phone_price := total_cost - (monthly_cost * months)

theorem phone_price_is_correct : phone_price = 2 :=
by
  sorry

end NUMINAMATH_GPT_phone_price_is_correct_l287_28718


namespace NUMINAMATH_GPT_carter_reading_pages_l287_28751

theorem carter_reading_pages (c l o : ℕ)
  (h1: c = l / 2)
  (h2: l = o + 20)
  (h3: o = 40) : c = 30 := by
  sorry

end NUMINAMATH_GPT_carter_reading_pages_l287_28751


namespace NUMINAMATH_GPT_area_of_folded_shape_is_two_units_squared_l287_28750

/-- 
A square piece of paper with each side of length 2 units is divided into 
four equal squares along both its length and width. From the top left corner to 
bottom right corner, a line is drawn through the center dividing the square diagonally.
The paper is folded along this line to form a new shape.
We prove that the area of the folded shape is 2 units².
-/
theorem area_of_folded_shape_is_two_units_squared
  (side_len : ℝ)
  (area_original : ℝ)
  (area_folded : ℝ)
  (h1 : side_len = 2)
  (h2 : area_original = side_len * side_len)
  (h3 : area_folded = area_original / 2) :
  area_folded = 2 := by
  -- Place proof here
  sorry

end NUMINAMATH_GPT_area_of_folded_shape_is_two_units_squared_l287_28750


namespace NUMINAMATH_GPT_scientific_notation_of_0_000815_l287_28748

theorem scientific_notation_of_0_000815 :
  (∃ (c : ℝ) (n : ℤ), 0.000815 = c * 10^n ∧ 1 ≤ c ∧ c < 10) ∧ (0.000815 = 8.15 * 10^(-4)) :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_0_000815_l287_28748


namespace NUMINAMATH_GPT_final_score_is_80_l287_28768

def adam_final_score : ℕ :=
  let first_half := 8
  let second_half := 2
  let points_per_question := 8
  (first_half + second_half) * points_per_question

theorem final_score_is_80 : adam_final_score = 80 := by
  sorry

end NUMINAMATH_GPT_final_score_is_80_l287_28768


namespace NUMINAMATH_GPT_sector_area_correct_l287_28790

-- Define the initial conditions
def arc_length := 4 -- Length of the arc in cm
def central_angle := 2 -- Central angle in radians
def radius := arc_length / central_angle -- Radius of the circle

-- Define the formula for the area of the sector
def sector_area := (1 / 2) * radius * arc_length

-- The statement of our theorem
theorem sector_area_correct : sector_area = 4 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sector_area_correct_l287_28790


namespace NUMINAMATH_GPT_remainder_when_A_divided_by_9_l287_28737

theorem remainder_when_A_divided_by_9 (A B : ℕ) (h1 : A = B * 9 + 13) : A % 9 = 4 := 
by {
  sorry
}

end NUMINAMATH_GPT_remainder_when_A_divided_by_9_l287_28737


namespace NUMINAMATH_GPT_sin_double_angle_l287_28756

theorem sin_double_angle (α : ℝ) (h : Real.sin (α + Real.pi / 4) = 1 / 2) : Real.sin (2 * α) = -1 / 2 :=
sorry

end NUMINAMATH_GPT_sin_double_angle_l287_28756


namespace NUMINAMATH_GPT_gadgets_selling_prices_and_total_amount_l287_28760

def cost_price_mobile : ℕ := 16000
def cost_price_laptop : ℕ := 25000
def cost_price_camera : ℕ := 18000

def loss_percentage_mobile : ℕ := 20
def gain_percentage_laptop : ℕ := 15
def loss_percentage_camera : ℕ := 10

def selling_price_mobile : ℕ := cost_price_mobile - (cost_price_mobile * loss_percentage_mobile / 100)
def selling_price_laptop : ℕ := cost_price_laptop + (cost_price_laptop * gain_percentage_laptop / 100)
def selling_price_camera : ℕ := cost_price_camera - (cost_price_camera * loss_percentage_camera / 100)

def total_amount_received : ℕ := selling_price_mobile + selling_price_laptop + selling_price_camera

theorem gadgets_selling_prices_and_total_amount :
  selling_price_mobile = 12800 ∧
  selling_price_laptop = 28750 ∧
  selling_price_camera = 16200 ∧
  total_amount_received = 57750 := by
  sorry

end NUMINAMATH_GPT_gadgets_selling_prices_and_total_amount_l287_28760


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_squares_l287_28700

theorem sum_of_reciprocals_of_squares (a b : ℕ) (h : a * b = 19) : 1 / (a * a : ℚ) + 1 / (b * b : ℚ) = 362 / 361 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_squares_l287_28700


namespace NUMINAMATH_GPT_find_A_l287_28725

def hash_rel (A B : ℝ) := A^2 + B^2

theorem find_A (A : ℝ) (h : hash_rel A 7 = 196) : A = 7 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_GPT_find_A_l287_28725


namespace NUMINAMATH_GPT_A_inter_complement_B_is_empty_l287_28730

open Set Real

noncomputable def U : Set Real := univ

noncomputable def A : Set Real := { x : Real | ∃ (y : Real), y = sqrt (Real.log x) }

noncomputable def B : Set Real := { y : Real | ∃ (x : Real), y = sqrt x }

theorem A_inter_complement_B_is_empty :
  A ∩ (U \ B) = ∅ :=
by
    sorry

end NUMINAMATH_GPT_A_inter_complement_B_is_empty_l287_28730


namespace NUMINAMATH_GPT_one_sofa_in_room_l287_28758

def num_sofas_in_room : ℕ :=
  let num_4_leg_tables := 4
  let num_4_leg_chairs := 2
  let num_3_leg_tables := 3
  let num_1_leg_table := 1
  let num_2_leg_rocking_chairs := 1
  let total_legs := 40

  let legs_of_4_leg_tables := num_4_leg_tables * 4
  let legs_of_4_leg_chairs := num_4_leg_chairs * 4
  let legs_of_3_leg_tables := num_3_leg_tables * 3
  let legs_of_1_leg_table := num_1_leg_table * 1
  let legs_of_2_leg_rocking_chairs := num_2_leg_rocking_chairs * 2

  let accounted_legs := legs_of_4_leg_tables + legs_of_4_leg_chairs + legs_of_3_leg_tables + legs_of_1_leg_table + legs_of_2_leg_rocking_chairs

  let remaining_legs := total_legs - accounted_legs

  let sofa_legs := 4
  remaining_legs / sofa_legs

theorem one_sofa_in_room : num_sofas_in_room = 1 :=
  by
    unfold num_sofas_in_room
    rfl

end NUMINAMATH_GPT_one_sofa_in_room_l287_28758


namespace NUMINAMATH_GPT_scale_down_multiplication_l287_28713

theorem scale_down_multiplication (h : 14.97 * 46 = 688.62) : 1.497 * 4.6 = 6.8862 :=
by
  -- here we assume the necessary steps to justify the statement.
  sorry

end NUMINAMATH_GPT_scale_down_multiplication_l287_28713


namespace NUMINAMATH_GPT_no_adjacent_stand_up_probability_l287_28723

noncomputable def coin_flip_prob_adjacent_people_stand_up : ℚ :=
  123 / 1024

theorem no_adjacent_stand_up_probability :
  let num_people := 10
  let total_outcomes := 2^num_people
  (123 : ℚ) / total_outcomes = coin_flip_prob_adjacent_people_stand_up :=
by
  sorry

end NUMINAMATH_GPT_no_adjacent_stand_up_probability_l287_28723


namespace NUMINAMATH_GPT_jane_wins_l287_28765

/-- Define the total number of possible outcomes and the number of losing outcomes -/
def total_outcomes := 64
def losing_outcomes := 12

/-- Define the probability that Jane wins -/
def jane_wins_probability := (total_outcomes - losing_outcomes) / total_outcomes

/-- Problem: Jane wins with a probability of 13/16 given the conditions -/
theorem jane_wins :
  jane_wins_probability = 13 / 16 :=
sorry

end NUMINAMATH_GPT_jane_wins_l287_28765


namespace NUMINAMATH_GPT_vertex_on_x_axis_l287_28771

theorem vertex_on_x_axis (c : ℝ) : (∃ (h : ℝ), (h, 0) = ((-(-8) / (2 * 1)), c - (-8)^2 / (4 * 1))) → c = 16 :=
by
  sorry

end NUMINAMATH_GPT_vertex_on_x_axis_l287_28771


namespace NUMINAMATH_GPT_ratio_a_c_l287_28752

-- Define variables and conditions
variables (a b c d : ℚ)

-- Conditions
def ratio_a_b : Prop := a / b = 5 / 4
def ratio_c_d : Prop := c / d = 4 / 3
def ratio_d_b : Prop := d / b = 1 / 5

-- Theorem statement
theorem ratio_a_c (h1 : ratio_a_b a b)
                  (h2 : ratio_c_d c d)
                  (h3 : ratio_d_b d b) : 
  (a / c = 75 / 16) :=
sorry

end NUMINAMATH_GPT_ratio_a_c_l287_28752


namespace NUMINAMATH_GPT_parabola_equation_l287_28724

theorem parabola_equation (P : ℝ × ℝ) (hP : P = (-4, -2)) :
  (∃ p : ℝ, P.1^2 = -2 * p * P.2 ∧ p = -4 ∧ x^2 = -8*y) ∨ 
  (∃ p : ℝ, P.2^2 = -2 * p * P.1 ∧ p = -1/2 ∧ y^2 = -x) :=
by
  sorry

end NUMINAMATH_GPT_parabola_equation_l287_28724


namespace NUMINAMATH_GPT_pennies_on_friday_l287_28708

-- Define the initial number of pennies and the function for doubling
def initial_pennies : Nat := 3
def double (n : Nat) : Nat := 2 * n

-- Prove the number of pennies on Friday
theorem pennies_on_friday : double (double (double (double initial_pennies))) = 48 := by
  sorry

end NUMINAMATH_GPT_pennies_on_friday_l287_28708


namespace NUMINAMATH_GPT_stratified_sampling_middle_schools_l287_28776

theorem stratified_sampling_middle_schools (high_schools : ℕ) (middle_schools : ℕ) (elementary_schools : ℕ) (total_selected : ℕ) 
    (h_high_schools : high_schools = 10) (h_middle_schools : middle_schools = 30) (h_elementary_schools : elementary_schools = 60)
    (h_total_selected : total_selected = 20) : 
    middle_schools * (total_selected / (high_schools + middle_schools + elementary_schools)) = 6 := 
by 
  sorry

end NUMINAMATH_GPT_stratified_sampling_middle_schools_l287_28776


namespace NUMINAMATH_GPT_units_digit_7_pow_3_l287_28783

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_7_pow_3 : units_digit (7^3) = 3 :=
by
  -- Proof of the theorem would go here
  sorry

end NUMINAMATH_GPT_units_digit_7_pow_3_l287_28783


namespace NUMINAMATH_GPT_evaluate_trig_expression_l287_28712

theorem evaluate_trig_expression :
  (Real.tan (π / 18) - Real.sqrt 3) * Real.sin (2 * π / 9) = -1 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_trig_expression_l287_28712


namespace NUMINAMATH_GPT_total_acorns_proof_l287_28786

variable (x y : ℝ)

def total_acorns (x y : ℝ) : ℝ :=
  let shawna := x
  let sheila := 5.3 * x
  let danny := 5.3 * x + y
  let ella := 2 * (4.3 * x + y)
  shawna + sheila + danny + ella

theorem total_acorns_proof (x y : ℝ) :
  total_acorns x y = 20.2 * x + 3 * y :=
by
  unfold total_acorns
  sorry

end NUMINAMATH_GPT_total_acorns_proof_l287_28786


namespace NUMINAMATH_GPT_solution_inequality_l287_28796

open Set

theorem solution_inequality (x : ℝ) : (x > 3 ∨ x < -3) ↔ (x > 9 / x) := by
  sorry

end NUMINAMATH_GPT_solution_inequality_l287_28796


namespace NUMINAMATH_GPT_javiers_household_legs_l287_28709

-- Definitions given the problem conditions
def humans : ℕ := 6
def human_legs : ℕ := 2

def dogs : ℕ := 2
def dog_legs : ℕ := 4

def cats : ℕ := 1
def cat_legs : ℕ := 4

def parrots : ℕ := 1
def parrot_legs : ℕ := 2

def lizards : ℕ := 1
def lizard_legs : ℕ := 4

def stool_legs : ℕ := 3
def table_legs : ℕ := 4
def cabinet_legs : ℕ := 6

-- Problem statement
theorem javiers_household_legs :
  (humans * human_legs) + (dogs * dog_legs) + (cats * cat_legs) + (parrots * parrot_legs) +
  (lizards * lizard_legs) + stool_legs + table_legs + cabinet_legs = 43 := by
  -- We leave the proof as an exercise for the reader
  sorry

end NUMINAMATH_GPT_javiers_household_legs_l287_28709


namespace NUMINAMATH_GPT_find_b_l287_28788

-- Define the problem based on the conditions identified
theorem find_b (b : ℕ) (h₁ : b > 0) (h₂ : (b : ℝ)/(b+15) = 0.75) : b = 45 := 
  sorry

end NUMINAMATH_GPT_find_b_l287_28788


namespace NUMINAMATH_GPT_quadratic_not_proposition_l287_28775

def is_proposition (P : Prop) : Prop := ∃ (b : Bool), (b = true ∨ b = false)

theorem quadratic_not_proposition : ¬ is_proposition (∃ x : ℝ, x^2 + 2*x - 3 < 0) :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_not_proposition_l287_28775


namespace NUMINAMATH_GPT_find_symmetric_sequence_l287_28749

noncomputable def symmetric_sequence (b : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n → b k = b (n - k + 1)

noncomputable def arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d, b 2 = b 1 + d ∧ b 3 = b 2 + d ∧ b 4 = b 3 + d

theorem find_symmetric_sequence :
  ∃ b : ℕ → ℤ, symmetric_sequence b 7 ∧ arithmetic_sequence b ∧ b 1 = 2 ∧ b 2 + b 4 = 16 ∧
  (b 1 = 2 ∧ b 2 = 5 ∧ b 3 = 8 ∧ b 4 = 11 ∧ b 5 = 8 ∧ b 6 = 5 ∧ b 7 = 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_symmetric_sequence_l287_28749


namespace NUMINAMATH_GPT_ratio_of_children_l287_28761

theorem ratio_of_children (C H : ℕ) 
  (hC1 : C / 8 = 16)
  (hC2 : C * (C / 8) = 512)
  (hH : H * 16 = 512) :
  H / C = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_children_l287_28761


namespace NUMINAMATH_GPT_ax5_by5_eq_neg1065_l287_28727

theorem ax5_by5_eq_neg1065 (a b x y : ℝ) 
  (h1 : a*x + b*y = 5) 
  (h2 : a*x^2 + b*y^2 = 9) 
  (h3 : a*x^3 + b*y^3 = 20) 
  (h4 : a*x^4 + b*y^4 = 48) 
  (h5 : x + y = -15) 
  (h6 : x^2 + y^2 = 55) : 
  a * x^5 + b * y^5 = -1065 := 
sorry

end NUMINAMATH_GPT_ax5_by5_eq_neg1065_l287_28727


namespace NUMINAMATH_GPT_number_of_oranges_l287_28762

def apples : ℕ := 14
def more_oranges : ℕ := 10

theorem number_of_oranges (o : ℕ) (apples_eq : apples = 14) (more_oranges_eq : more_oranges = 10) :
  o = apples + more_oranges :=
by
  sorry

end NUMINAMATH_GPT_number_of_oranges_l287_28762


namespace NUMINAMATH_GPT_total_running_duration_l287_28743

-- Conditions
def speed1 := 15 -- speed during the first part in mph
def time1 := 3 -- time during the first part in hours
def speed2 := 19 -- speed during the second part in mph
def distance2 := 190 -- distance during the second part in miles

-- Initialize
def distance1 := speed1 * time1 -- distance covered in the first part in miles

def time2 := distance2 / speed2 -- time to cover the distance in the second part in hours

-- Total duration
def total_duration := time1 + time2

-- Proof statement
theorem total_running_duration : total_duration = 13 :=
by
  sorry

end NUMINAMATH_GPT_total_running_duration_l287_28743


namespace NUMINAMATH_GPT_odd_square_mod_eight_l287_28784

theorem odd_square_mod_eight (k : ℤ) : ((2 * k + 1) ^ 2) % 8 = 1 := 
sorry

end NUMINAMATH_GPT_odd_square_mod_eight_l287_28784


namespace NUMINAMATH_GPT_proof_problem_l287_28772

def M : Set ℝ := { x | x > -1 }

theorem proof_problem : {0} ⊆ M := by
  sorry

end NUMINAMATH_GPT_proof_problem_l287_28772


namespace NUMINAMATH_GPT_domain_of_h_l287_28740

noncomputable def h (x : ℝ) : ℝ := (5 * x - 2) / (2 * x - 10)

theorem domain_of_h :
  {x : ℝ | 2 * x - 10 ≠ 0} = {x : ℝ | x ≠ 5} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_h_l287_28740


namespace NUMINAMATH_GPT_six_people_acquaintance_or_strangers_l287_28704

theorem six_people_acquaintance_or_strangers (p : Fin 6 → Prop) :
  ∃ (A B C : Fin 6), (p A ∧ p B ∧ p C) ∨ (¬p A ∧ ¬p B ∧ ¬p C) :=
sorry

end NUMINAMATH_GPT_six_people_acquaintance_or_strangers_l287_28704


namespace NUMINAMATH_GPT_find_fx_l287_28793

def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x

theorem find_fx (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_neg : ∀ x : ℝ, x < 0 → f x = x * (x - 1)) :
  ∀ x : ℝ, x > 0 → f x = x * (x + 1) :=
by
  sorry

end NUMINAMATH_GPT_find_fx_l287_28793


namespace NUMINAMATH_GPT_find_first_week_customers_l287_28745

def commission_per_customer := 1
def first_week_customers (C : ℕ) := C
def second_week_customers (C : ℕ) := 2 * C
def third_week_customers (C : ℕ) := 3 * C
def salary := 500
def bonus := 50
def total_earnings := 760

theorem find_first_week_customers (C : ℕ) (H : salary + bonus + commission_per_customer * (first_week_customers C + second_week_customers C + third_week_customers C) = total_earnings) : 
  C = 35 :=
by
  sorry

end NUMINAMATH_GPT_find_first_week_customers_l287_28745


namespace NUMINAMATH_GPT_min_sum_of_abc_conditions_l287_28791

theorem min_sum_of_abc_conditions
  (a b c d : ℕ)
  (hab : a + b = 2)
  (hac : a + c = 3)
  (had : a + d = 4)
  (hbc : b + c = 5)
  (hbd : b + d = 6)
  (hcd : c + d = 7) :
  a + b + c + d = 9 :=
sorry

end NUMINAMATH_GPT_min_sum_of_abc_conditions_l287_28791


namespace NUMINAMATH_GPT_hector_gumballs_remaining_l287_28701

def gumballs_remaining (gumballs : ℕ) (given_todd : ℕ) (given_alisha : ℕ) (given_bobby : ℕ) : ℕ :=
  gumballs - (given_todd + given_alisha + given_bobby)

theorem hector_gumballs_remaining :
  let gumballs := 45
  let given_todd := 4
  let given_alisha := 2 * given_todd
  let given_bobby := 4 * given_alisha - 5
  gumballs_remaining gumballs given_todd given_alisha given_bobby = 6 :=
by 
  let gumballs := 45
  let given_todd := 4
  let given_alisha := 2 * given_todd
  let given_bobby := 4 * given_alisha - 5
  show gumballs_remaining gumballs given_todd given_alisha given_bobby = 6
  sorry

end NUMINAMATH_GPT_hector_gumballs_remaining_l287_28701


namespace NUMINAMATH_GPT_add_percentages_10_30_15_50_l287_28754

-- Define the problem conditions:
def ten_percent (x : ℝ) : ℝ := 0.10 * x
def fifteen_percent (y : ℝ) : ℝ := 0.15 * y
def add_percentages (x y : ℝ) : ℝ := ten_percent x + fifteen_percent y

theorem add_percentages_10_30_15_50 :
  add_percentages 30 50 = 10.5 :=
by
  sorry

end NUMINAMATH_GPT_add_percentages_10_30_15_50_l287_28754


namespace NUMINAMATH_GPT_max_successful_free_throws_l287_28744

theorem max_successful_free_throws (a b : ℕ) 
  (h1 : a + b = 105) 
  (h2 : a > 0)
  (h3 : b > 0)
  (ha : a % 3 = 0)
  (hb : b % 5 = 0)
  : (a / 3 + 3 * (b / 5)) ≤ 59 := sorry

end NUMINAMATH_GPT_max_successful_free_throws_l287_28744


namespace NUMINAMATH_GPT_evaluate_g_at_3_l287_28797

def g (x : ℝ) : ℝ := 7 * x^3 - 5 * x^2 - 7 * x + 3

theorem evaluate_g_at_3 : g 3 = 126 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_g_at_3_l287_28797


namespace NUMINAMATH_GPT_calculation_l287_28755

theorem calculation : 120 / 5 / 3 * 2 = 16 := by
  sorry

end NUMINAMATH_GPT_calculation_l287_28755


namespace NUMINAMATH_GPT_how_many_raisins_did_bryce_receive_l287_28711

def raisins_problem : Prop :=
  ∃ (B C : ℕ), B = C + 8 ∧ C = B / 3 ∧ B + C = 44 ∧ B = 33

theorem how_many_raisins_did_bryce_receive : raisins_problem :=
sorry

end NUMINAMATH_GPT_how_many_raisins_did_bryce_receive_l287_28711


namespace NUMINAMATH_GPT_harmonic_mean_1999_2001_is_2000_l287_28715

def harmonic_mean (a b : ℕ) : ℚ := (2 * a * b) / (a + b)

theorem harmonic_mean_1999_2001_is_2000 :
  abs (harmonic_mean 1999 2001 - 2000 : ℚ) < 1 := by
  -- Actual proof omitted
  sorry

end NUMINAMATH_GPT_harmonic_mean_1999_2001_is_2000_l287_28715


namespace NUMINAMATH_GPT_smallest_x_of_quadratic_eqn_l287_28778

theorem smallest_x_of_quadratic_eqn : ∃ x : ℝ, (12*x^2 - 44*x + 40 = 0) ∧ x = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_of_quadratic_eqn_l287_28778
