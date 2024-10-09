import Mathlib

namespace initial_percentage_of_managers_l1005_100589

theorem initial_percentage_of_managers (P : ℕ) (h : 0 ≤ P ∧ P ≤ 100)
  (total_employees initial_managers : ℕ) 
  (h1 : total_employees = 500) 
  (h2 : initial_managers = P * total_employees / 100) 
  (remaining_employees remaining_managers : ℕ)
  (h3 : remaining_employees = total_employees - 250)
  (h4 : remaining_managers = initial_managers - 250)
  (h5 : remaining_managers * 100 = 98 * remaining_employees) :
  P = 99 := 
by
  sorry

end initial_percentage_of_managers_l1005_100589


namespace distinguishable_squares_count_is_70_l1005_100502

def count_distinguishable_squares : ℕ :=
  let total_colorings : ℕ := 2^9
  let rotation_90_270_fixed : ℕ := 2^3
  let rotation_180_fixed : ℕ := 2^5
  let average_fixed_colorings : ℕ :=
    (total_colorings + rotation_90_270_fixed + rotation_90_270_fixed + rotation_180_fixed) / 4
  let distinguishable_squares : ℕ := average_fixed_colorings / 2
  distinguishable_squares

theorem distinguishable_squares_count_is_70 :
  count_distinguishable_squares = 70 := by
  sorry

end distinguishable_squares_count_is_70_l1005_100502


namespace player_A_winning_probability_l1005_100527

theorem player_A_winning_probability :
  let P_draw := 1 / 2
  let P_B_wins := 1 / 3
  let P_total := 1
  P_total - P_draw - P_B_wins = 1 / 6 :=
by
  let P_draw := 1 / 2
  let P_B_wins := 1 / 3
  let P_total := 1
  sorry

end player_A_winning_probability_l1005_100527


namespace find_m_eq_l1005_100559

theorem find_m_eq : 
  (∀ (m : ℝ),
    ((m + 2)^2 + (m + 3)^2 = m^2 + 16 + 4 + (m - 1)^2) →
    m = 2 / 3 ) :=
by
  intros m h
  sorry

end find_m_eq_l1005_100559


namespace find_tan_of_cos_in_4th_quadrant_l1005_100530

-- Given conditions
variable (α : ℝ) (h1 : Real.cos α = 3/5) (h2 : α > 3*Real.pi/2 ∧ α < 2*Real.pi)

-- Lean statement to prove the question
theorem find_tan_of_cos_in_4th_quadrant : Real.tan α = - (4 / 3) := 
by
  sorry

end find_tan_of_cos_in_4th_quadrant_l1005_100530


namespace math_problem_proof_l1005_100526

def eight_to_zero : ℝ := 1
def log_base_10_of_100 : ℝ := 2

theorem math_problem_proof : eight_to_zero - log_base_10_of_100 = -1 :=
by sorry

end math_problem_proof_l1005_100526


namespace tourists_originally_in_group_l1005_100562

theorem tourists_originally_in_group (x : ℕ) (h₁ : 220 / x - 220 / (x + 1) = 2) : x = 10 := 
by
  sorry

end tourists_originally_in_group_l1005_100562


namespace part1_prove_BD_eq_b_part2_prove_cos_ABC_l1005_100573

-- Definition of the problem setup
variables {a b c : ℝ}
variables {A B C : ℝ}    -- angles
variables {D : ℝ}        -- point on side AC

-- Conditions
axiom b_squared_eq_ac : b^2 = a * c
axiom BD_sin_ABC_eq_a_sin_C : D * Real.sin B = a * Real.sin C
axiom AD_eq_2DC : A = 2 * C

-- Part (1)
theorem part1_prove_BD_eq_b : D = b :=
by
  sorry

-- Part (2)
theorem part2_prove_cos_ABC :
  Real.cos B = 7 / 12 :=
by
  sorry

end part1_prove_BD_eq_b_part2_prove_cos_ABC_l1005_100573


namespace min_number_of_bags_l1005_100580

theorem min_number_of_bags (a b : ℕ) : 
  ∃ K : ℕ, K = a + b - Nat.gcd a b :=
by
  sorry

end min_number_of_bags_l1005_100580


namespace andy_time_correct_l1005_100529

-- Define the conditions
def time_dawn_wash_dishes : ℕ := 20
def time_andy_put_laundry : ℕ := 2 * time_dawn_wash_dishes + 6

-- The theorem to prove
theorem andy_time_correct : time_andy_put_laundry = 46 :=
by
  -- Proof goes here
  sorry

end andy_time_correct_l1005_100529


namespace quotient_remainder_base5_l1005_100579

theorem quotient_remainder_base5 (n m : ℕ) 
    (hn : n = 3 * 5^3 + 2 * 5^2 + 3 * 5^1 + 2)
    (hm : m = 2 * 5^1 + 1) :
    n / m = 40 ∧ n % m = 2 :=
by
  sorry

end quotient_remainder_base5_l1005_100579


namespace remaining_shoes_to_sell_l1005_100553

def shoes_goal : Nat := 80
def shoes_sold_last_week : Nat := 27
def shoes_sold_this_week : Nat := 12

theorem remaining_shoes_to_sell : shoes_goal - (shoes_sold_last_week + shoes_sold_this_week) = 41 :=
by
  sorry

end remaining_shoes_to_sell_l1005_100553


namespace inequality_condition_l1005_100507

-- Define the inequality (x - 2) * (x + 2) > 0
def inequality_holds (x : ℝ) : Prop := (x - 2) * (x + 2) > 0

-- The sufficient and necessary condition for the inequality to hold is x > 2 or x < -2
theorem inequality_condition (x : ℝ) : inequality_holds x ↔ (x > 2 ∨ x < -2) :=
  sorry

end inequality_condition_l1005_100507


namespace find_functions_l1005_100533

theorem find_functions (M N : ℝ × ℝ)
  (hM : M.fst = -4) (hM_quad2 : 0 < M.snd)
  (hN : N = (-6, 0))
  (h_area : 1 / 2 * 6 * M.snd = 15) :
  (∃ k, ∀ x, (M = (-4, 5) → N = (-6, 0) → x * k = -5 / 4 * x)) ∧ 
  (∃ a b, ∀ x, (M = (-4, 5) → N = (-6, 0) → x * a + b = 5 / 2 * x + 15)) := 
sorry

end find_functions_l1005_100533


namespace non_isosceles_count_l1005_100572

def n : ℕ := 20

def total_triangles : ℕ := Nat.choose n 3

def isosceles_triangles_per_vertex : ℕ := 9

def total_isosceles_triangles : ℕ := n * isosceles_triangles_per_vertex

def non_isosceles_triangles : ℕ := total_triangles - total_isosceles_triangles

theorem non_isosceles_count :
  non_isosceles_triangles = 960 := 
  by 
    -- proof details would go here
    sorry

end non_isosceles_count_l1005_100572


namespace trigonometric_expression_evaluation_l1005_100541

theorem trigonometric_expression_evaluation :
  1 / Real.sin (70 * Real.pi / 180) - Real.sqrt 3 / Real.cos (70 * Real.pi / 180) = -4 :=
by
  sorry

end trigonometric_expression_evaluation_l1005_100541


namespace correct_system_of_equations_l1005_100581

theorem correct_system_of_equations (x y : ℕ) :
  (x / 3 = y - 2) ∧ ((x - 9) / 2 = y) ↔
  (∃ x y, (x / 3 = y - 2) ∧ ((x - 9) / 2 = y)) :=
by
  sorry

end correct_system_of_equations_l1005_100581


namespace power_identity_l1005_100586

theorem power_identity :
  (3 ^ 12) * (3 ^ 8) = 243 ^ 4 :=
sorry

end power_identity_l1005_100586


namespace area_shaded_region_l1005_100568

-- Define the conditions in Lean

def semicircle_radius_ADB : ℝ := 2
def semicircle_radius_BEC : ℝ := 2
def midpoint_arc_ADB (D : ℝ) : Prop := D = semicircle_radius_ADB
def midpoint_arc_BEC (E : ℝ) : Prop := E = semicircle_radius_BEC
def semicircle_radius_DFE : ℝ := 1
def midpoint_arc_DFE (F : ℝ) : Prop := F = semicircle_radius_DFE

-- Given the mentioned conditions, we want to show the area of the shaded region is 8 square units
theorem area_shaded_region 
  (D E F : ℝ) 
  (hD : midpoint_arc_ADB D)
  (hE : midpoint_arc_BEC E)
  (hF : midpoint_arc_DFE F) : 
  ∃ (area : ℝ), area = 8 := 
sorry

end area_shaded_region_l1005_100568


namespace solve_for_y_l1005_100534

-- Define the conditions and the goal to prove in Lean 4
theorem solve_for_y
  (x y : ℤ) 
  (h1 : x + y = 250) 
  (h2 : x - y = 200) : 
  y = 25 :=
by
  sorry

end solve_for_y_l1005_100534


namespace base3_to_base10_l1005_100598

theorem base3_to_base10 (d0 d1 d2 d3 d4 : ℕ)
  (h0 : d4 = 2)
  (h1 : d3 = 1)
  (h2 : d2 = 0)
  (h3 : d1 = 2)
  (h4 : d0 = 1) :
  d4 * 3^4 + d3 * 3^3 + d2 * 3^2 + d1 * 3^1 + d0 * 3^0 = 196 := by
  sorry

end base3_to_base10_l1005_100598


namespace correct_calculation_result_l1005_100522

theorem correct_calculation_result (x : ℤ) (h : x + 44 - 39 = 63) : x + 39 - 44 = 53 := by
  sorry

end correct_calculation_result_l1005_100522


namespace contradiction_proof_example_l1005_100545

theorem contradiction_proof_example (a b : ℝ) (h: a ≤ b → False) : a > b :=
by sorry

end contradiction_proof_example_l1005_100545


namespace second_set_length_is_20_l1005_100565

-- Define the lengths
def length_first_set : ℕ := 4
def length_second_set : ℕ := 5 * length_first_set

-- Formal proof statement
theorem second_set_length_is_20 : length_second_set = 20 :=
by
  sorry

end second_set_length_is_20_l1005_100565


namespace polygon_sides_arithmetic_progression_l1005_100510

theorem polygon_sides_arithmetic_progression 
  (n : ℕ) 
  (h1 : ∀ n, ∃ a_1, ∃ a_n, ∀ i, a_n = 172 ∧ (a_i = a_1 + (i - 1) * 4) ∧ (i ≤ n))
  (h2 : ∀ S, S = 180 * (n - 2)) 
  (h3 : ∀ S, S = n * ((172 - 4 * (n - 1) + 172) / 2)) 
  : n = 12 := 
by 
  sorry

end polygon_sides_arithmetic_progression_l1005_100510


namespace min_f_value_l1005_100511

noncomputable def f (a b : ℝ) := 
  Real.sqrt (2 * a^2 - 8 * a + 10) + 
  Real.sqrt (b^2 - 6 * b + 10) + 
  Real.sqrt (2 * a^2 - 2 * a * b + b^2)

theorem min_f_value : ∃ a b : ℝ, f a b = 2 * Real.sqrt 5 :=
sorry

end min_f_value_l1005_100511


namespace axis_of_symmetry_compare_m_n_range_t_max_t_l1005_100556

-- Condition: Definition of the parabola
def parabola (t x : ℝ) := x^2 - 2 * t * x + 1

-- Problem 1: Axis of symmetry
theorem axis_of_symmetry (t : ℝ) : 
  ∀ (y x : ℝ), parabola t x = y -> x = t :=
sorry

-- Problem 2: Comparing m and n
theorem compare_m_n (t m n : ℝ) :
  parabola t (t - 2) = m ∧ parabola t (t + 3) = n -> n > m := 
sorry

-- Problem 3: Range of t for y₁ ≤ y₂
theorem range_t (t x₁ y₁ y₂ : ℝ) :
  -1 ≤ x₁ ∧ x₁ < 3 ∧ parabola t x₁ = y₁ ∧ parabola t 3 = y₂ -> y₁ ≤ y₂ → t ≤ 1 :=
sorry

-- Problem 4: Maximum t for y₁ ≥ y₂
theorem max_t (t y₁ y₂ : ℝ) :
  (parabola t (t + 1) = y₁ ∧ parabola t (2 * t - 4) = y₂) → y₁ ≥ y₂ → t ≤ 5 :=
sorry

end axis_of_symmetry_compare_m_n_range_t_max_t_l1005_100556


namespace avg_integer_N_between_fractions_l1005_100571

theorem avg_integer_N_between_fractions (N : ℕ) (h1 : (2 : ℚ) / 5 < N / 42) (h2 : N / 42 < 1 / 3) : 
  N = 15 := 
by
  sorry

end avg_integer_N_between_fractions_l1005_100571


namespace unique_solution_otimes_l1005_100535

def otimes (x y : ℝ) : ℝ := 5 * x - 2 * y + 3 * x * y

theorem unique_solution_otimes : 
  (∃! y : ℝ, otimes 2 y = 20) := 
by
  sorry

end unique_solution_otimes_l1005_100535


namespace global_maximum_condition_l1005_100592

noncomputable def f (x m : ℝ) : ℝ :=
if x ≤ m then -x^2 - 2 * x else -x + 2

theorem global_maximum_condition (m : ℝ) (h : ∃ (x0 : ℝ), ∀ (x : ℝ), f x m ≤ f x0 m) : m ≥ 1 :=
sorry

end global_maximum_condition_l1005_100592


namespace ratio_diff_l1005_100560

theorem ratio_diff (x : ℕ) (h1 : 7 * x = 56) : 56 - 3 * x = 32 :=
by
  sorry

end ratio_diff_l1005_100560


namespace carl_olivia_cookie_difference_l1005_100543

-- Defining the various conditions
def Carl_cookies : ℕ := 7
def Olivia_cookies : ℕ := 2

-- Stating the theorem we need to prove
theorem carl_olivia_cookie_difference : Carl_cookies - Olivia_cookies = 5 :=
by sorry

end carl_olivia_cookie_difference_l1005_100543


namespace first_number_is_seven_l1005_100542

variable (x y : ℝ)

theorem first_number_is_seven (h1 : x + y = 10) (h2 : 2 * x = 3 * y + 5) : x = 7 :=
sorry

end first_number_is_seven_l1005_100542


namespace even_function_m_eq_neg_one_l1005_100531

theorem even_function_m_eq_neg_one (m : ℝ) :
  (∀ x : ℝ, (m - 1)*x^2 - (m^2 - 1)*x + (m + 2) = (m - 1)*(-x)^2 - (m^2 - 1)*(-x) + (m + 2)) →
  m = -1 :=
  sorry

end even_function_m_eq_neg_one_l1005_100531


namespace vector_x_value_l1005_100516

open Real

noncomputable def a (x : ℝ) : ℝ × ℝ := (x, x + 1)
def b : ℝ × ℝ := (1, 2)

def perpendicular (v1 v2 : ℝ × ℝ) : Prop := 
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem vector_x_value (x : ℝ) : (perpendicular (a x) b) → x = -2 / 3 := by
  intro h
  sorry

end vector_x_value_l1005_100516


namespace shaded_area_in_design_l1005_100563

theorem shaded_area_in_design (side_length : ℝ) (radius : ℝ)
  (h1 : side_length = 30) (h2 : radius = side_length / 6)
  (h3 : 6 * (π * radius^2) = 150 * π) :
  (side_length^2) - 6 * (π * radius^2) = 900 - 150 * π := 
by
  sorry

end shaded_area_in_design_l1005_100563


namespace treaty_signed_on_thursday_l1005_100575

def initial_day : ℕ := 0  -- 0 representing Monday, assuming a week cycle from 0 (Monday) to 6 (Sunday)
def days_in_week : ℕ := 7

def treaty_day (n : ℕ) : ℕ :=
(n + initial_day) % days_in_week

theorem treaty_signed_on_thursday :
  treaty_day 1000 = 4 :=  -- 4 representing Thursday
by
  sorry

end treaty_signed_on_thursday_l1005_100575


namespace ratio_of_larger_to_smaller_l1005_100578

theorem ratio_of_larger_to_smaller (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) (h4 : a + b = 7 * (a - b)) : a / b = 2 := 
by
  sorry

end ratio_of_larger_to_smaller_l1005_100578


namespace arcsin_one_eq_pi_div_two_l1005_100504

theorem arcsin_one_eq_pi_div_two : Real.arcsin 1 = (Real.pi / 2) :=
by
  sorry

end arcsin_one_eq_pi_div_two_l1005_100504


namespace inequality_example_l1005_100521

open Real

theorem inequality_example 
    (x y z : ℝ) 
    (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1):
    (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := 
by 
  sorry

end inequality_example_l1005_100521


namespace ice_cream_cones_sixth_day_l1005_100548

theorem ice_cream_cones_sixth_day (cones_day1 cones_day2 cones_day3 cones_day4 cones_day5 cones_day7 : ℝ)
  (mean : ℝ) (h1 : cones_day1 = 100) (h2 : cones_day2 = 92) 
  (h3 : cones_day3 = 109) (h4 : cones_day4 = 96) 
  (h5 : cones_day5 = 103) (h7 : cones_day7 = 105) 
  (h_mean : mean = 100.1) : 
  ∃ cones_day6 : ℝ, cones_day6 = 95.7 :=
by 
  sorry

end ice_cream_cones_sixth_day_l1005_100548


namespace cabin_price_correct_l1005_100597

noncomputable def cabin_price 
  (cash : ℤ)
  (cypress_trees : ℤ) (pine_trees : ℤ) (maple_trees : ℤ)
  (price_cypress : ℤ) (price_pine : ℤ) (price_maple : ℤ)
  (remaining_cash : ℤ)
  (expected_price : ℤ) : Prop :=
   cash + (cypress_trees * price_cypress + pine_trees * price_pine + maple_trees * price_maple) - remaining_cash = expected_price

theorem cabin_price_correct :
  cabin_price 150 20 600 24 100 200 300 350 130000 :=
by
  sorry

end cabin_price_correct_l1005_100597


namespace multiplication_of_monomials_l1005_100523

-- Define the constants and assumptions
def a : ℝ := -2
def b : ℝ := 4
def e1 : ℤ := 4
def e2 : ℤ := 5
def result : ℝ := -8
def result_exp : ℤ := 9

-- State the theorem to be proven
theorem multiplication_of_monomials :
  (a * 10^e1) * (b * 10^e2) = result * 10^result_exp := 
by
  sorry

end multiplication_of_monomials_l1005_100523


namespace total_pikes_l1005_100512

theorem total_pikes (x : ℝ) (h : x = 4 + (1/2) * x) : x = 8 :=
sorry

end total_pikes_l1005_100512


namespace angle_sum_eq_180_l1005_100577

theorem angle_sum_eq_180 (A B C D E F G : ℝ) 
  (h1 : A + B + C + D + E + F = 360) : 
  A + B + C + D + E + F + G = 180 :=
by
  sorry

end angle_sum_eq_180_l1005_100577


namespace circle_radius_eq_l1005_100576

theorem circle_radius_eq (r : ℝ) (AB : ℝ) (BC : ℝ) (hAB : AB = 10) (hBC : BC = 12) : r = 25 / 4 := by
  sorry

end circle_radius_eq_l1005_100576


namespace value_of_f5_f_neg5_l1005_100536

-- Define the function f
def f (x a b : ℝ) : ℝ := x^5 - a * x^3 + b * x + 2

-- Given conditions
variable (a b : ℝ)
axiom h1 : f (-5) a b = 3

-- The proposition to prove
theorem value_of_f5_f_neg5 : f 5 a b + f (-5) a b = 4 :=
by
  -- Include the result of the proof
  sorry

end value_of_f5_f_neg5_l1005_100536


namespace prism_faces_same_color_l1005_100585

structure PrismColoring :=
  (A : Fin 5 → Fin 5 → Bool)
  (B : Fin 5 → Fin 5 → Bool)
  (A_to_B : Fin 5 → Fin 5 → Bool)

def all_triangles_diff_colors (pc : PrismColoring) : Prop :=
  ∀ i j k : Fin 5, i ≠ j → j ≠ k → k ≠ i →
    (pc.A i j = !pc.A i k ∨ pc.A i j = !pc.A j k) ∧
    (pc.B i j = !pc.B i k ∨ pc.B i j = !pc.B j k) ∧
    (pc.A_to_B i j = !pc.A_to_B i k ∨ pc.A_to_B i j = !pc.A_to_B j k)

theorem prism_faces_same_color (pc : PrismColoring) (h : all_triangles_diff_colors pc) :
  (∀ i j : Fin 5, pc.A i j = pc.A 0 1) ∧ (∀ i j : Fin 5, pc.B i j = pc.B 0 1) :=
sorry

end prism_faces_same_color_l1005_100585


namespace slope_of_parallel_lines_l1005_100590

theorem slope_of_parallel_lines (m : ℝ)
  (y1 y2 y3 : ℝ)
  (h1 : y1 = 2) 
  (h2 : y2 = 3) 
  (h3 : y3 = 4)
  (sum_of_x_intercepts : (-2 / m) + (-3 / m) + (-4 / m) = 36) :
  m = -1 / 4 := by
  sorry

end slope_of_parallel_lines_l1005_100590


namespace Fred_earned_4_dollars_l1005_100554

-- Conditions are translated to definitions
def initial_amount_Fred : ℕ := 111
def current_amount_Fred : ℕ := 115

-- Proof problem in Lean 4 statement
theorem Fred_earned_4_dollars : current_amount_Fred - initial_amount_Fred = 4 := by
  sorry

end Fred_earned_4_dollars_l1005_100554


namespace greatest_product_l1005_100595

theorem greatest_product (x : ℤ) (h : x + (1998 - x) = 1998) : 
  x * (1998 - x) ≤ 998001 :=
  sorry

end greatest_product_l1005_100595


namespace problem_l1005_100594

def f (a : ℕ) : ℕ := a + 3
def F (a b : ℕ) : ℕ := b^2 + a

theorem problem : F 4 (f 5) = 68 := by sorry

end problem_l1005_100594


namespace inequality_for_large_n_l1005_100540

theorem inequality_for_large_n (n : ℕ) (hn : n > 1) : 
  (1 / Real.exp 1 - 1 / (n * Real.exp 1)) < (1 - 1 / n) ^ n ∧ (1 - 1 / n) ^ n < (1 / Real.exp 1 - 1 / (2 * n * Real.exp 1)) :=
sorry

end inequality_for_large_n_l1005_100540


namespace range_of_x_for_positive_y_l1005_100584

theorem range_of_x_for_positive_y (x : ℝ) : 
  (-1 < x ∧ x < 3) ↔ (-x^2 + 2*x + 3 > 0) :=
sorry

end range_of_x_for_positive_y_l1005_100584


namespace number_of_balls_l1005_100561

noncomputable def frequency_of_yellow (n : ℕ) : ℚ := 9 / n

theorem number_of_balls (n : ℕ) (h1 : frequency_of_yellow n = 0.30) : n = 30 :=
by sorry

end number_of_balls_l1005_100561


namespace prime_1002_n_count_l1005_100587

theorem prime_1002_n_count :
  ∃! n : ℕ, n ≥ 2 ∧ Prime (n^3 + 2) :=
by
  sorry

end prime_1002_n_count_l1005_100587


namespace total_signs_at_intersections_l1005_100547

-- Definitions based on the given conditions
def first_intersection_signs : ℕ := 40
def second_intersection_signs : ℕ := first_intersection_signs + first_intersection_signs / 4
def third_intersection_signs : ℕ := 2 * second_intersection_signs
def fourth_intersection_signs : ℕ := third_intersection_signs - 20

-- Prove the total number of signs at the four intersections is 270
theorem total_signs_at_intersections :
  first_intersection_signs + second_intersection_signs + third_intersection_signs + fourth_intersection_signs = 270 := by
  sorry

end total_signs_at_intersections_l1005_100547


namespace equal_poly_terms_l1005_100515

theorem equal_poly_terms (p q : ℝ) (hpq : p + q = 1) (hp : 0 < p) (hq : 0 < q) : 
  (7 * p^6 * q = 21 * p^5 * q^2) -> p = 3 / 4 :=
by
  sorry

end equal_poly_terms_l1005_100515


namespace problem_solution_l1005_100501

theorem problem_solution (x y : ℝ) (h1 : x + y = 500) (h2 : x / y = 0.8) : y - x = 500 / 9 :=
by
  sorry

end problem_solution_l1005_100501


namespace parabola_vertex_intercept_l1005_100550

variable (a b c p : ℝ)

theorem parabola_vertex_intercept (h_vertex : ∀ x : ℝ, (a * (x - p) ^ 2 + p) = a * x^2 + b * x + c)
                                  (h_intercept : a * p^2 + p = 2 * p)
                                  (hp : p ≠ 0) : b = -2 :=
sorry

end parabola_vertex_intercept_l1005_100550


namespace g_five_eq_thirteen_sevenths_l1005_100558

def g (x : ℚ) : ℚ := (3 * x - 2) / (x + 2)

theorem g_five_eq_thirteen_sevenths : g 5 = 13 / 7 := by
  sorry

end g_five_eq_thirteen_sevenths_l1005_100558


namespace profit_calculation_l1005_100519

def Initial_Value : ℕ := 100
def Multiplier : ℕ := 3
def New_Value : ℕ := Initial_Value * Multiplier
def Profit : ℕ := New_Value - Initial_Value

theorem profit_calculation : Profit = 200 := by
  sorry

end profit_calculation_l1005_100519


namespace Mike_changed_64_tires_l1005_100557

def tires_changed (motorcycles: ℕ) (cars: ℕ): ℕ := 
  (motorcycles * 2) + (cars * 4)

theorem Mike_changed_64_tires:
  (tires_changed 12 10) = 64 :=
by
  sorry

end Mike_changed_64_tires_l1005_100557


namespace largest_inscribed_triangle_area_l1005_100570

theorem largest_inscribed_triangle_area 
  (radius : ℝ) 
  (diameter : ℝ)
  (base : ℝ)
  (height : ℝ) 
  (area : ℝ)
  (h1 : radius = 10)
  (h2 : diameter = 2 * radius)
  (h3 : base = diameter)
  (h4 : height = radius) 
  (h5 : area = (1/2) * base * height) : 
  area  = 100 :=
by 
  have h_area := (1/2) * 20 * 10
  sorry

end largest_inscribed_triangle_area_l1005_100570


namespace cost_of_jam_l1005_100582

theorem cost_of_jam (N B J : ℕ) (hN : N > 1) (h_total_cost : N * (5 * B + 6 * J) = 348) :
    6 * N * J = 348 := by
  sorry

end cost_of_jam_l1005_100582


namespace gina_tom_goals_l1005_100505

theorem gina_tom_goals :
  let g_day1 := 2
  let t_day1 := g_day1 + 3
  let t_day2 := 6
  let g_day2 := t_day2 - 2
  let g_total := g_day1 + g_day2
  let t_total := t_day1 + t_day2
  g_total + t_total = 17 := by
  sorry

end gina_tom_goals_l1005_100505


namespace true_statement_given_conditions_l1005_100506

theorem true_statement_given_conditions (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a < b) :
  |1| / |a| > |1| / |b| := 
by
  sorry

end true_statement_given_conditions_l1005_100506


namespace k3_to_fourth_equals_81_l1005_100588

theorem k3_to_fourth_equals_81
  (h k : ℝ → ℝ)
  (h_cond : ∀ x, x ≥ 1 → h (k x) = x^3)
  (k_cond : ∀ x, x ≥ 1 → k (h x) = x^4)
  (k_81 : k 81 = 81) :
  k 3 ^ 4 = 81 :=
sorry

end k3_to_fourth_equals_81_l1005_100588


namespace fraction_ratios_l1005_100569

theorem fraction_ratios (m n p q : ℕ) (h1 : (m : ℚ) / n = 18) (h2 : (p : ℚ) / n = 6) (h3 : (p : ℚ) / q = 1 / 15) :
  (m : ℚ) / q = 1 / 5 :=
sorry

end fraction_ratios_l1005_100569


namespace curve_passes_through_fixed_point_l1005_100551

theorem curve_passes_through_fixed_point (m n : ℝ) :
  (2:ℝ)^2 + (-2:ℝ)^2 - 2 * m * (2:ℝ) - 2 * n * (-2:ℝ) + 4 * (m - n - 2) = 0 :=
by sorry

end curve_passes_through_fixed_point_l1005_100551


namespace simplify_expression_l1005_100524

theorem simplify_expression (x y m : ℤ) 
  (h1 : (x-5)^2 = -|m-1|)
  (h2 : y + 1 = 5) :
  (2 * x^2 - 3 * x * y - 4 * y^2) - m * (3 * x^2 - x * y + 9 * y^2) = -273 :=
sorry

end simplify_expression_l1005_100524


namespace probability_one_instrument_l1005_100593

theorem probability_one_instrument (total_people : ℕ) (at_least_one_instrument_ratio : ℚ) (two_or_more_instruments : ℕ)
  (h1 : total_people = 800) (h2 : at_least_one_instrument_ratio = 1 / 5) (h3 : two_or_more_instruments = 128) :
  (160 - 128) / 800 = 1 / 25 :=
by
  sorry

end probability_one_instrument_l1005_100593


namespace sin_cos_product_l1005_100532

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := by
  sorry

end sin_cos_product_l1005_100532


namespace find_a1_l1005_100509

-- Defining the conditions
variables (a : ℕ → ℝ)
variable (q : ℝ)
variable (h_monotone : ∀ n, a n ≥ a (n + 1)) -- Monotonically decreasing

-- Specific values from the problem
axiom h_a3 : a 3 = 1
axiom h_a2_a4 : a 2 + a 4 = 5 / 2
axiom h_geom_seq : ∀ n, a (n + 1) = a n * q  -- Geometric sequence property

-- The goal is to prove that a 1 = 4
theorem find_a1 : a 1 = 4 :=
by
  -- Insert proof here
  sorry

end find_a1_l1005_100509


namespace total_over_or_underweight_is_8kg_total_selling_price_is_5384_point_8_yuan_l1005_100591

-- Definitions based on conditions
def standard_weight : ℝ := 25
def weight_diffs : List ℝ := [-3, -2, -2, -2, -2, -1.5, -1.5, 0, 0, 0, 1, 1, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
def price_per_kg : ℝ := 10.6

-- Problem 1
theorem total_over_or_underweight_is_8kg :
  (weight_diffs.sum = 8) := 
  sorry

-- Problem 2
theorem total_selling_price_is_5384_point_8_yuan :
  (20 * standard_weight + 8) * price_per_kg = 5384.8 :=
  sorry

end total_over_or_underweight_is_8kg_total_selling_price_is_5384_point_8_yuan_l1005_100591


namespace right_triangle_cos_B_l1005_100520

theorem right_triangle_cos_B (A B C : ℝ) (hC : C = 90) (hSinA : Real.sin A = 2 / 3) :
  Real.cos B = 2 / 3 :=
sorry

end right_triangle_cos_B_l1005_100520


namespace inequality_ab_equals_bc_l1005_100555

-- Define the given conditions and state the theorem as per the proof problem
theorem inequality_ab_equals_bc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    a^b * b^c * c^a ≤ a^a * b^b * c^c :=
by
  sorry

end inequality_ab_equals_bc_l1005_100555


namespace speed_of_stream_l1005_100503

theorem speed_of_stream (v_s : ℝ) (D : ℝ) (h1 : D / (78 - v_s) = 2 * (D / (78 + v_s))) : v_s = 26 :=
by
  sorry

end speed_of_stream_l1005_100503


namespace product_prs_l1005_100537

open Real

theorem product_prs (p r s : ℕ) 
  (h1 : 4 ^ p + 64 = 272) 
  (h2 : 3 ^ r = 81)
  (h3 : 6 ^ s = 478) : 
  p * r * s = 64 :=
by
  sorry

end product_prs_l1005_100537


namespace div_mul_fraction_eq_neg_81_over_4_l1005_100546

theorem div_mul_fraction_eq_neg_81_over_4 : 
  -4 / (4 / 9) * (9 / 4) = - (81 / 4) := 
by
  sorry

end div_mul_fraction_eq_neg_81_over_4_l1005_100546


namespace expected_heads_l1005_100508

def coin_flips : Nat := 64

def prob_heads (tosses : ℕ) : ℚ :=
  1 / 2^(tosses + 1)

def total_prob_heads : ℚ :=
  prob_heads 0 + prob_heads 1 + prob_heads 2 + prob_heads 3

theorem expected_heads : (coin_flips : ℚ) * total_prob_heads = 60 := by
  sorry

end expected_heads_l1005_100508


namespace percentage_liked_B_l1005_100574

-- Given conditions
def percent_liked_A (X : ℕ) : Prop := X ≥ 0 ∧ X ≤ 100 -- X percent of respondents liked product A
def percent_liked_both : ℕ := 23 -- 23 percent liked both products.
def percent_liked_neither : ℕ := 23 -- 23 percent liked neither product.
def min_surveyed_people : ℕ := 100 -- The minimum number of people surveyed by the company.

-- Required proof
theorem percentage_liked_B (X : ℕ) (h : percent_liked_A X):
  100 - X = Y :=
sorry

end percentage_liked_B_l1005_100574


namespace sum_of_possible_values_l1005_100538

theorem sum_of_possible_values (x y z w : ℝ) 
  (h : (x - y) * (z - w) / ((y - z) * (w - x)) = 3 / 7) :
  (x - z) * (y - w) / ((x - y) * (z - w)) = -4 / 3 := 
sorry

end sum_of_possible_values_l1005_100538


namespace length_of_goods_train_l1005_100514

-- Define the given data
def speed_kmph := 72
def platform_length_m := 250
def crossing_time_s := 36

-- Convert speed from kmph to m/s
def speed_mps := speed_kmph * (5 / 18)

-- Define the total distance covered while crossing the platform
def distance_covered_m := speed_mps * crossing_time_s

-- Define the length of the train
def train_length_m := distance_covered_m - platform_length_m

-- The theorem to be proven
theorem length_of_goods_train : train_length_m = 470 := by
  sorry

end length_of_goods_train_l1005_100514


namespace cricket_initial_overs_l1005_100566

theorem cricket_initial_overs
  (target_runs : ℚ) (initial_run_rate : ℚ) (remaining_run_rate : ℚ) (remaining_overs : ℕ)
  (total_runs_needed : target_runs = 282)
  (run_rate_initial : initial_run_rate = 3.4)
  (run_rate_remaining : remaining_run_rate = 6.2)
  (overs_remaining : remaining_overs = 40) :
  ∃ (initial_overs : ℕ), initial_overs = 10 :=
by
  sorry

end cricket_initial_overs_l1005_100566


namespace concyclic_iff_ratio_real_l1005_100552

noncomputable def concyclic_condition (z1 z2 z3 z4 : ℂ) : Prop :=
  (∃ c : ℂ, c ≠ 0 ∧ ∀ (w : ℂ), (w - z1) * (w - z3) / ((w - z2) * (w - z4)) = c)

noncomputable def ratio_real (z1 z2 z3 z4 : ℂ) : Prop :=
  ∃ r : ℝ, (z1 - z3) * (z2 - z4) / ((z1 - z4) * (z2 - z3)) = r

theorem concyclic_iff_ratio_real (z1 z2 z3 z4 : ℂ) :
  concyclic_condition z1 z2 z3 z4 ↔ ratio_real z1 z2 z3 z4 :=
sorry

end concyclic_iff_ratio_real_l1005_100552


namespace min_value_le_one_l1005_100518

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x
noncomputable def g (a : ℝ) : ℝ := a - a * Real.log a

theorem min_value_le_one (a : ℝ) (ha : a > 0) :
  (∀ x : ℝ, f x a ≥ g a) ∧ g a ≤ 1 := sorry

end min_value_le_one_l1005_100518


namespace correct_calculation_result_l1005_100544

theorem correct_calculation_result :
  ∀ (A B D : ℝ),
  C = 6 →
  E = 5 →
  (A * 10 + B) * 6 + D * E = 39.6 ∨ (A * 10 + B) * 6 * D * E = 36.9 →
  (A * 10 + B) * 6 + D * E = 26.1 :=
by
  intros A B D C_eq E_eq errors
  sorry

end correct_calculation_result_l1005_100544


namespace line_equation_l1005_100513

theorem line_equation (x y : ℝ) (c : ℝ)
  (h1 : 2 * x - y + 3 = 0)
  (h2 : 4 * x + 3 * y + 1 = 0)
  (h3 : 3 * x + 2 * y + c = 0) :
  c = 1 := sorry

end line_equation_l1005_100513


namespace simplify_expression_l1005_100599

theorem simplify_expression (i : ℂ) (h : i^2 = -1) : 
  3 * (4 - 2 * i) + 2 * i * (3 + i) + 5 * (-1 + i) = 5 + 5 * i :=
by
  sorry

end simplify_expression_l1005_100599


namespace find_cd_minus_dd_base_d_l1005_100528

namespace MathProof

variables (d C D : ℤ)

def digit_sum (C D : ℤ) (d : ℤ) : ℤ := d * C + D
def digit_sum_same (C : ℤ) (d : ℤ) : ℤ := d * C + C

theorem find_cd_minus_dd_base_d (h_d : d > 8) (h_eq : digit_sum C D d + digit_sum_same C d = d^2 + 8 * d + 4) :
  C - D = 1 :=
by
  sorry

end MathProof

end find_cd_minus_dd_base_d_l1005_100528


namespace lily_pad_growth_rate_l1005_100525

theorem lily_pad_growth_rate 
  (day_37_covers_full : ℕ → ℝ)
  (day_36_covers_half : ℕ → ℝ)
  (exponential_growth : day_37_covers_full = 2 * day_36_covers_half) :
  (2 - 1) / 1 * 100 = 100 :=
by sorry

end lily_pad_growth_rate_l1005_100525


namespace max_marked_cells_no_shared_vertices_l1005_100539

theorem max_marked_cells_no_shared_vertices (N : ℕ) (cube_side : ℕ) (total_cells : ℕ) (total_vertices : ℕ) :
  cube_side = 3 →
  total_cells = cube_side ^ 3 →
  total_vertices = 8 + 12 * 2 + 6 * 4 →
  ∀ (max_cells : ℕ), (4 * max_cells ≤ total_vertices) → (max_cells ≤ 14) :=
by
  sorry

end max_marked_cells_no_shared_vertices_l1005_100539


namespace problem1_problem2_l1005_100567

-- Definitions
def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

-- Statement 1: If a = 1 and p ∧ q is true, then the range of x is 2 < x < 3
theorem problem1 (x : ℝ) (h : 1 = 1) (hpq : p x 1 ∧ q x) : 2 < x ∧ x < 3 :=
sorry

-- Statement 2: If ¬p is a sufficient but not necessary condition for ¬q, then the range of a is 1 < a ≤ 2
theorem problem2 (a : ℝ) (h1 : 1 < a) (h2 : a ≤ 2) (h3 : ¬ (∃ x, p x a) → ¬ (∃ x, q x)) : 1 < a ∧ a ≤ 2 :=
sorry

end problem1_problem2_l1005_100567


namespace lines_intersect_l1005_100564

-- Define the parameterizations of the two lines
def line1 (t : ℚ) : ℚ × ℚ := ⟨2 + 3 * t, 3 - 4 * t⟩
def line2 (u : ℚ) : ℚ × ℚ := ⟨4 + 5 * u, 1 + 3 * u⟩

theorem lines_intersect :
  ∃ t u : ℚ, line1 t = line2 u ∧ line1 t = ⟨26 / 11, 19 / 11⟩ :=
by
  sorry

end lines_intersect_l1005_100564


namespace correct_answer_B_l1005_100517

def point_slope_form (k : ℝ) (x y : ℝ) : Prop := y + 1 = k * (x - 2)

def proposition_2 (k : ℝ) (x y : ℝ) : Prop :=
  ∃ k : ℝ, @point_slope_form k x y

def proposition_3 (k : ℝ) : Prop := point_slope_form k 2 (-1)

def proposition_4 (k : ℝ) : Prop := k ≠ 0

theorem correct_answer_B : 
  (∃ k : ℝ, @point_slope_form k 2 (-1)) ∧ 
  (∀ k : ℝ, @point_slope_form k 2 (-1)) ∧
  (∀ k : ℝ, k ≠ 0) → true := 
by
  intro h
  sorry

end correct_answer_B_l1005_100517


namespace mat_length_is_correct_l1005_100549

noncomputable def mat_length (r : ℝ) (w : ℝ) : ℝ :=
  let θ := 2 * Real.pi / 5
  let side := 2 * r * Real.sin (θ / 2)
  let D := r * Real.cos (Real.pi / 5)
  let x := ((Real.sqrt (r^2 - ((w / 2) ^ 2))) - D + (w / 2))
  x

theorem mat_length_is_correct :
  mat_length 5 1 = 1.4 :=
by
  sorry

end mat_length_is_correct_l1005_100549


namespace movie_tickets_ratio_l1005_100500

theorem movie_tickets_ratio (R H : ℕ) (hR : R = 25) (hH : H = 93) : 
  (H / R : ℚ) = 93 / 25 :=
by
  sorry

end movie_tickets_ratio_l1005_100500


namespace perimeter_of_grid_l1005_100596

theorem perimeter_of_grid (area: ℕ) (side_length: ℕ) (perimeter: ℕ) 
  (h1: area = 144) 
  (h2: 4 * side_length * side_length = area) 
  (h3: perimeter = 4 * 2 * side_length) : 
  perimeter = 48 :=
by
  sorry

end perimeter_of_grid_l1005_100596


namespace find_a_l1005_100583

noncomputable def f (a x : ℝ) : ℝ := Real.log (a * x + 1)

theorem find_a {a : ℝ} (h : (deriv (f a) 0 = 1)) : a = 1 :=
by
  -- Proof goes here
  sorry

end find_a_l1005_100583
