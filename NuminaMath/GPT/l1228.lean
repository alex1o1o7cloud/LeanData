import Mathlib

namespace NUMINAMATH_GPT_solve_for_x_l1228_122833

theorem solve_for_x (x : ℚ) (h : (1 / 2 - 1 / 3) = 3 / x) : x = 18 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1228_122833


namespace NUMINAMATH_GPT_problem_conditions_l1228_122858

theorem problem_conditions (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 3) :
  (x * y ≤ 9 / 8) ∧ (4 ^ x + 2 ^ y ≥ 4 * Real.sqrt 2) ∧ (x / y + 1 / x ≥ 2 / 3 + 2 * Real.sqrt 3 / 3) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_problem_conditions_l1228_122858


namespace NUMINAMATH_GPT_count_valid_three_digit_numbers_l1228_122839

theorem count_valid_three_digit_numbers : 
  let total_numbers := 900
  let excluded_numbers := 9 * 10 * 9
  let valid_numbers := total_numbers - excluded_numbers
  valid_numbers = 90 :=
by
  let total_numbers := 900
  let excluded_numbers := 9 * 10 * 9
  let valid_numbers := total_numbers - excluded_numbers
  have h1 : valid_numbers = 900 - 810 := by rfl
  have h2 : 900 - 810 = 90 := by norm_num
  exact h1.trans h2

end NUMINAMATH_GPT_count_valid_three_digit_numbers_l1228_122839


namespace NUMINAMATH_GPT_ray_nickels_left_l1228_122862

theorem ray_nickels_left (h1 : 285 % 5 = 0) (h2 : 55 % 5 = 0) (h3 : 3 * 55 % 5 = 0) (h4 : 45 % 5 = 0) : 
  285 / 5 - ((55 / 5) + (3 * 55 / 5) + (45 / 5)) = 4 := sorry

end NUMINAMATH_GPT_ray_nickels_left_l1228_122862


namespace NUMINAMATH_GPT_length_of_AB_l1228_122844

theorem length_of_AB (V : ℝ) (r : ℝ) :
  V = 216 * Real.pi →
  r = 3 →
  ∃ (len_AB : ℝ), len_AB = 20 :=
by
  intros hV hr
  have volume_cylinder := V - 36 * Real.pi
  have height_cylinder := volume_cylinder / (Real.pi * r^2)
  exists height_cylinder
  exact sorry

end NUMINAMATH_GPT_length_of_AB_l1228_122844


namespace NUMINAMATH_GPT_figures_can_be_drawn_l1228_122866

structure Figure :=
  (degrees : List ℕ) -- List of degrees of the vertices in the graph associated with the figure.

-- Define a predicate to check if a figure can be drawn without lifting the pencil and without retracing
def canBeDrawnWithoutLifting (fig : Figure) : Prop :=
  let odd_degree_vertices := fig.degrees.filter (λ d => d % 2 = 1)
  odd_degree_vertices.length = 0 ∨ odd_degree_vertices.length = 2

-- Define the figures A, B, C, D with their degrees (examples, these should match the problem's context)
def figureA : Figure := { degrees := [2, 2, 2, 2] }
def figureB : Figure := { degrees := [2, 2, 2, 2, 4] }
def figureC : Figure := { degrees := [3, 3, 3, 3] }
def figureD : Figure := { degrees := [4, 4, 2, 2] }

-- State the theorem that figures A, B, and D can be drawn without lifting the pencil
theorem figures_can_be_drawn :
  canBeDrawnWithoutLifting figureA ∧ canBeDrawnWithoutLifting figureB ∧ canBeDrawnWithoutLifting figureD :=
  by sorry -- Proof to be completed

end NUMINAMATH_GPT_figures_can_be_drawn_l1228_122866


namespace NUMINAMATH_GPT_cannot_reach_target_l1228_122874

def initial_price : ℕ := 1
def annual_increment : ℕ := 1
def tripling_year (n : ℕ) : ℕ := 3 * n
def total_years : ℕ := 99
def target_price : ℕ := 152
def incremental_years : ℕ := 98

noncomputable def final_price (x : ℕ) : ℕ := 
  initial_price + incremental_years * annual_increment + tripling_year x - annual_increment

theorem cannot_reach_target (p : ℕ) (h : p = final_price p) : p ≠ target_price :=
sorry

end NUMINAMATH_GPT_cannot_reach_target_l1228_122874


namespace NUMINAMATH_GPT_ellipse_foci_condition_l1228_122804

theorem ellipse_foci_condition {m : ℝ} :
  (1 < m ∧ m < 2) ↔ (∃ (x y : ℝ), (x^2 / (m - 1) + y^2 / (3 - m) = 1) ∧ (3 - m > m - 1) ∧ (m - 1 > 0) ∧ (3 - m > 0)) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_foci_condition_l1228_122804


namespace NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l1228_122888

-- Definitions for the quadratic equation coefficients
def a : ℝ := 3
def b : ℝ := -4
def c : ℝ := 1

-- Definition of the discriminant
def Δ : ℝ := b^2 - 4 * a * c

-- Statement of the problem: Prove that the quadratic equation has two distinct real roots
theorem quadratic_has_distinct_real_roots (hΔ : Δ = 4) : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l1228_122888


namespace NUMINAMATH_GPT_no_rectangular_prism_equal_measures_l1228_122812

theorem no_rectangular_prism_equal_measures (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0): 
  ¬ (4 * (a + b + c) = 2 * (a * b + b * c + c * a) ∧ 2 * (a * b + b * c + c * a) = a * b * c) :=
by
  sorry

end NUMINAMATH_GPT_no_rectangular_prism_equal_measures_l1228_122812


namespace NUMINAMATH_GPT_square_adjacent_to_multiple_of_5_l1228_122815

theorem square_adjacent_to_multiple_of_5 (n : ℤ) (h : n % 5 ≠ 0) : (∃ k : ℤ, n^2 = 5 * k + 1) ∨ (∃ k : ℤ, n^2 = 5 * k - 1) := 
by
  sorry

end NUMINAMATH_GPT_square_adjacent_to_multiple_of_5_l1228_122815


namespace NUMINAMATH_GPT_technicians_count_l1228_122806

theorem technicians_count (avg_all : ℕ) (avg_tech : ℕ) (avg_other : ℕ) (total_workers : ℕ)
  (h1 : avg_all = 750) (h2 : avg_tech = 900) (h3 : avg_other = 700) (h4 : total_workers = 20) :
  ∃ T O : ℕ, (T + O = total_workers) ∧ ((T * avg_tech + O * avg_other) = total_workers * avg_all) ∧ (T = 5) :=
by
  sorry

end NUMINAMATH_GPT_technicians_count_l1228_122806


namespace NUMINAMATH_GPT_g_value_at_8_l1228_122811

noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 5*x + 6

theorem g_value_at_8 (g : ℝ → ℝ) (h1 : ∀ x : ℝ, g x = (1/216) * (x - (a^3)) * (x - (b^3)) * (x - (c^3))) 
  (h2 : g 0 = 1) 
  (h3 : ∀ a b c : ℝ, f (a) = 0 ∧ f (b) = 0 ∧ f (c) = 0) : 
  g 8 = 0 :=
sorry

end NUMINAMATH_GPT_g_value_at_8_l1228_122811


namespace NUMINAMATH_GPT_months_to_survive_l1228_122851

theorem months_to_survive (P_survive : ℝ) (initial_population : ℕ) (expected_survivors : ℝ) (n : ℕ)
  (h1 : P_survive = 5 / 6)
  (h2 : initial_population = 200)
  (h3 : expected_survivors = 115.74)
  (h4 : initial_population * (P_survive ^ n) = expected_survivors) :
  n = 3 :=
sorry

end NUMINAMATH_GPT_months_to_survive_l1228_122851


namespace NUMINAMATH_GPT_cassie_and_brian_meet_at_1111am_l1228_122823

theorem cassie_and_brian_meet_at_1111am :
  ∃ t : ℕ, t = 11*60 + 11 ∧
    (∃ x : ℚ, x = 51/16 ∧ 
      14 * x + 18 * (x - 1) = 84) :=
sorry

end NUMINAMATH_GPT_cassie_and_brian_meet_at_1111am_l1228_122823


namespace NUMINAMATH_GPT_bridget_gave_erasers_l1228_122829

variable (p_start : ℕ) (p_end : ℕ) (e_b : ℕ)

theorem bridget_gave_erasers (h1 : p_start = 8) (h2 : p_end = 11) (h3 : p_end = p_start + e_b) :
  e_b = 3 := by
  sorry

end NUMINAMATH_GPT_bridget_gave_erasers_l1228_122829


namespace NUMINAMATH_GPT_find_z_l1228_122813

open Complex

theorem find_z (z : ℂ) (h : z * (2 - I) = 5 * I) : z = -1 + 2 * I :=
sorry

end NUMINAMATH_GPT_find_z_l1228_122813


namespace NUMINAMATH_GPT_Jolene_charge_per_car_l1228_122863

theorem Jolene_charge_per_car (babysitting_families cars_washed : ℕ) (charge_per_family total_raised babysitting_earnings car_charge : ℕ) :
  babysitting_families = 4 →
  charge_per_family = 30 →
  cars_washed = 5 →
  total_raised = 180 →
  babysitting_earnings = babysitting_families * charge_per_family →
  car_charge = (total_raised - babysitting_earnings) / cars_washed →
  car_charge = 12 :=
by
  intros
  sorry

end NUMINAMATH_GPT_Jolene_charge_per_car_l1228_122863


namespace NUMINAMATH_GPT_problem_1_problem_2_l1228_122807

variable {m n x : ℝ}

theorem problem_1 (m n : ℝ) : (m + n) * (2 * m + n) + n * (m - n) = 2 * m ^ 2 + 4 * m * n := 
by
  sorry

theorem problem_2 (x : ℝ) (h : x ≠ 0) : ((x + 3) / x - 2) / ((x ^ 2 - 9) / (4 * x)) = -(4  / (x + 3)) :=
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1228_122807


namespace NUMINAMATH_GPT_solution_is_correct_l1228_122832

-- Initial conditions
def initial_volume : ℝ := 6
def initial_concentration : ℝ := 0.40
def target_concentration : ℝ := 0.50

-- Given that we start with 2.4 liters of pure alcohol in a 6-liter solution
def initial_pure_alcohol : ℝ := initial_volume * initial_concentration

-- Expected result after adding x liters of pure alcohol
def final_solution_volume (x : ℝ) : ℝ := initial_volume + x
def final_pure_alcohol (x : ℝ) : ℝ := initial_pure_alcohol + x

-- Equation to prove
theorem solution_is_correct (x : ℝ) :
  (final_pure_alcohol x) / (final_solution_volume x) = target_concentration ↔ 
  x = 1.2 := 
sorry

end NUMINAMATH_GPT_solution_is_correct_l1228_122832


namespace NUMINAMATH_GPT_find_a3_l1228_122805

open Nat

def seq (a : ℕ → ℕ) : Prop := 
  (a 1 = 1) ∧ (∀ n : ℕ, n > 0 → a (n + 1) - a n = n)

theorem find_a3 (a : ℕ → ℕ) (h : seq a) : a 3 = 4 := by
  sorry

end NUMINAMATH_GPT_find_a3_l1228_122805


namespace NUMINAMATH_GPT_payment_correct_l1228_122853

def total_payment (hours1 hours2 hours3 : ℕ) (rate_per_hour : ℕ) (num_men : ℕ) : ℕ :=
  (hours1 + hours2 + hours3) * rate_per_hour * num_men

theorem payment_correct :
  total_payment 10 8 15 10 2 = 660 :=
by
  -- We skip the proof here
  sorry

end NUMINAMATH_GPT_payment_correct_l1228_122853


namespace NUMINAMATH_GPT_percentage_increase_area_l1228_122830

theorem percentage_increase_area (L W : ℝ) :
  let A := L * W
  let L' := 1.20 * L
  let W' := 1.20 * W
  let A' := L' * W'
  let percentage_increase := (A' - A) / A * 100
  L > 0 → W > 0 → percentage_increase = 44 := 
by
  sorry

end NUMINAMATH_GPT_percentage_increase_area_l1228_122830


namespace NUMINAMATH_GPT_quadratic_to_vertex_form_l1228_122852

theorem quadratic_to_vertex_form :
  ∀ (x a h k : ℝ), (x^2 - 7*x = a*(x - h)^2 + k) → k = -49 / 4 :=
by
  intros x a h k
  sorry

end NUMINAMATH_GPT_quadratic_to_vertex_form_l1228_122852


namespace NUMINAMATH_GPT_octopus_leg_count_l1228_122836

theorem octopus_leg_count :
  let num_initial_octopuses := 5
  let legs_per_normal_octopus := 8
  let num_removed_octopuses := 2
  let legs_first_mutant := 10
  let legs_second_mutant := 6
  let legs_third_mutant := 2 * legs_per_normal_octopus
  let num_initial_legs := num_initial_octopuses * legs_per_normal_octopus
  let num_removed_legs := num_removed_octopuses * legs_per_normal_octopus
  let num_mutant_legs := legs_first_mutant + legs_second_mutant + legs_third_mutant
  num_initial_legs - num_removed_legs + num_mutant_legs = 56 :=
by
  -- proof to be filled in later
  sorry

end NUMINAMATH_GPT_octopus_leg_count_l1228_122836


namespace NUMINAMATH_GPT_third_dimension_of_box_l1228_122854

theorem third_dimension_of_box (h : ℕ) (H : (151^2 - 150^2) * h + 151^2 = 90000) : h = 223 :=
sorry

end NUMINAMATH_GPT_third_dimension_of_box_l1228_122854


namespace NUMINAMATH_GPT_inequality_proof_l1228_122826

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 3) :
    (a^2 + 6) / (2*a^2 + 2*b^2 + 2*c^2 + 2*a - 1) +
    (b^2 + 6) / (2*a^2 + 2*b^2 + 2*c^2 + 2*b - 1) +
    (c^2 + 6) / (2*a^2 + 2*b^2 + 2*c^2 + 2*c - 1) ≤
    3 := by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1228_122826


namespace NUMINAMATH_GPT_range_of_fx1_l1228_122881

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2*x + 1 + a * Real.log x

theorem range_of_fx1 (a x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) (h3 : f x1 a = 0) (h4 : f x2 a = 0) :
    f x1 a > (1 - 2 * Real.log 2) / 4 :=
sorry

end NUMINAMATH_GPT_range_of_fx1_l1228_122881


namespace NUMINAMATH_GPT_pastries_sold_is_correct_l1228_122825

-- Definitions of the conditions
def initial_pastries : ℕ := 56
def remaining_pastries : ℕ := 27

-- Statement of the theorem
theorem pastries_sold_is_correct : initial_pastries - remaining_pastries = 29 :=
by
  sorry

end NUMINAMATH_GPT_pastries_sold_is_correct_l1228_122825


namespace NUMINAMATH_GPT_prime_factor_difference_duodecimal_l1228_122876

theorem prime_factor_difference_duodecimal (A B : ℕ) (hA : 0 ≤ A ∧ A ≤ 11) (hB : 0 ≤ B ∧ B ≤ 11) (h : A ≠ B) : 
  ∃ k : ℤ, (12 * A + B - (12 * B + A)) = 11 * k := 
by sorry

end NUMINAMATH_GPT_prime_factor_difference_duodecimal_l1228_122876


namespace NUMINAMATH_GPT_TruckY_average_speed_is_63_l1228_122821

noncomputable def average_speed_TruckY (initial_gap : ℕ) (extra_distance : ℕ) (hours : ℕ) (distance_X_per_hour : ℕ) : ℕ :=
  let distance_X := distance_X_per_hour * hours
  let total_distance_Y := distance_X + initial_gap + extra_distance
  total_distance_Y / hours

theorem TruckY_average_speed_is_63 
  (initial_gap : ℕ := 14) 
  (extra_distance : ℕ := 4) 
  (hours : ℕ := 3)
  (distance_X_per_hour : ℕ := 57) : 
  average_speed_TruckY initial_gap extra_distance hours distance_X_per_hour = 63 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_TruckY_average_speed_is_63_l1228_122821


namespace NUMINAMATH_GPT_convert_cylindrical_to_rectangular_l1228_122842

theorem convert_cylindrical_to_rectangular (r θ z x y : ℝ) (h_r : r = 5) (h_θ : θ = (3 * Real.pi) / 2) (h_z : z = 4)
    (h_x : x = r * Real.cos θ) (h_y : y = r * Real.sin θ) :
    (x, y, z) = (0, -5, 4) :=
by
    sorry

end NUMINAMATH_GPT_convert_cylindrical_to_rectangular_l1228_122842


namespace NUMINAMATH_GPT_coordinates_of_point_l1228_122849

theorem coordinates_of_point (x y : ℝ) (hx : x < 0) (hy : y > 0) (dx : |x| = 3) (dy : |y| = 2) :
  (x, y) = (-3, 2) := 
sorry

end NUMINAMATH_GPT_coordinates_of_point_l1228_122849


namespace NUMINAMATH_GPT_train_length_is_135_l1228_122870

noncomputable def length_of_train (speed_kmh : ℕ) (time_sec : ℕ) : ℕ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * time_sec

theorem train_length_is_135 :
  length_of_train 54 9 = 135 := 
by
  -- Conditions: 
  -- speed_kmh = 54
  -- time_sec = 9
  sorry

end NUMINAMATH_GPT_train_length_is_135_l1228_122870


namespace NUMINAMATH_GPT_xy_yz_zx_value_l1228_122887

namespace MathProof

theorem xy_yz_zx_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : x^2 + x * y + y^2 = 147) 
  (h2 : y^2 + y * z + z^2 = 16) 
  (h3 : z^2 + x * z + x^2 = 163) :
  x * y + y * z + z * x = 56 := 
sorry      

end MathProof

end NUMINAMATH_GPT_xy_yz_zx_value_l1228_122887


namespace NUMINAMATH_GPT_range_of_f_x_minus_2_l1228_122880

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x + 1 else if x > 0 then -(x + 1) else 0

theorem range_of_f_x_minus_2 :
  ∀ x : ℝ, f (x - 2) < 0 ↔ x ∈ Set.union (Set.Iio 1) (Set.Ioo 2 3) := by
sorry

end NUMINAMATH_GPT_range_of_f_x_minus_2_l1228_122880


namespace NUMINAMATH_GPT_additional_dividend_amount_l1228_122893

theorem additional_dividend_amount
  (E : ℝ) (Q : ℝ) (expected_extra_per_earnings : ℝ) (half_of_extra_per_earnings_to_dividend : ℝ) 
  (expected : E = 0.80) (quarterly_earnings : Q = 1.10)
  (extra_per_earnings : expected_extra_per_earnings = 0.30)
  (half_dividend : half_of_extra_per_earnings_to_dividend = 0.15):
  Q - E = expected_extra_per_earnings ∧ 
  expected_extra_per_earnings / 2 = half_of_extra_per_earnings_to_dividend :=
by sorry

end NUMINAMATH_GPT_additional_dividend_amount_l1228_122893


namespace NUMINAMATH_GPT_cats_after_purchasing_l1228_122894

/-- Mrs. Sheridan's total number of cats after purchasing more -/
theorem cats_after_purchasing (a b : ℕ) (h₀ : a = 11) (h₁ : b = 43) : a + b = 54 := by
  sorry

end NUMINAMATH_GPT_cats_after_purchasing_l1228_122894


namespace NUMINAMATH_GPT_second_horse_revolutions_l1228_122840

theorem second_horse_revolutions (r1 r2 d1: ℝ) (n1 n2: ℕ) 
  (h1: r1 = 30) (h2: d1 = 36) (h3: r2 = 10) 
  (h4: 2 * Real.pi * r1 * d1 = 2 * Real.pi * r2 * n2) : 
  n2 = 108 := 
by
   sorry

end NUMINAMATH_GPT_second_horse_revolutions_l1228_122840


namespace NUMINAMATH_GPT_initial_women_count_l1228_122877

-- Let x be the initial number of women.
-- Let y be the initial number of men.

theorem initial_women_count (x y : ℕ) (h1 : y = 2 * (x - 15)) (h2 : (y - 45) * 5 = (x - 15)) :
  x = 40 :=
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_initial_women_count_l1228_122877


namespace NUMINAMATH_GPT_range_of_4x_2y_l1228_122860

theorem range_of_4x_2y (x y : ℝ) 
  (h1 : 1 ≤ x + y) (h2 : x + y ≤ 3) 
  (h3 : -1 ≤ x - y) (h4 : x - y ≤ 1) :
  2 ≤ 4 * x + 2 * y ∧ 4 * x + 2 * y ≤ 10 := 
sorry

end NUMINAMATH_GPT_range_of_4x_2y_l1228_122860


namespace NUMINAMATH_GPT_grassy_plot_width_l1228_122879

/-- A rectangular grassy plot has a length of 100 m and a certain width. 
It has a gravel path 2.5 m wide all round it on the inside. The cost of gravelling 
the path at 0.90 rupees per square meter is 742.5 rupees. 
Prove that the width of the grassy plot is 60 meters. -/
theorem grassy_plot_width 
  (length : ℝ)
  (path_width : ℝ)
  (cost_per_sq_meter : ℝ)
  (total_cost : ℝ)
  (width : ℝ) : 
  length = 100 ∧ 
  path_width = 2.5 ∧ 
  cost_per_sq_meter = 0.9 ∧ 
  total_cost = 742.5 → 
  width = 60 := 
by sorry

end NUMINAMATH_GPT_grassy_plot_width_l1228_122879


namespace NUMINAMATH_GPT_find_line_eq_l1228_122814

noncomputable def line_eq (x y : ℝ) : Prop :=
  (∃ a : ℝ, a ≠ 0 ∧ (a * x - y = 0 ∨ x + y - a = 0)) 

theorem find_line_eq : line_eq 2 3 :=
by
  sorry

end NUMINAMATH_GPT_find_line_eq_l1228_122814


namespace NUMINAMATH_GPT_problem_range_of_a_l1228_122890

theorem problem_range_of_a (a : ℝ) :
  (∀ x : ℝ, |2 - x| + |3 + x| ≥ a^2 - 4 * a) ↔ -1 ≤ a ∧ a ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_problem_range_of_a_l1228_122890


namespace NUMINAMATH_GPT_problem_given_conditions_l1228_122867

theorem problem_given_conditions (x y z : ℝ) 
  (h : x / 3 = y / (-4) ∧ y / (-4) = z / 7) : (3 * x + y + z) / y = -3 := 
by 
  sorry

end NUMINAMATH_GPT_problem_given_conditions_l1228_122867


namespace NUMINAMATH_GPT_number_of_students_l1228_122838

theorem number_of_students (N : ℕ) (T : ℕ)
  (h1 : T = 80 * N)
  (h2 : (T - 160) / (N - 8) = 90) :
  N = 56 :=
sorry

end NUMINAMATH_GPT_number_of_students_l1228_122838


namespace NUMINAMATH_GPT_pyramid_total_blocks_l1228_122878

-- Define the number of layers in the pyramid
def num_layers : ℕ := 8

-- Define the block multiplier for each subsequent layer
def block_multiplier : ℕ := 5

-- Define the number of blocks in the top layer
def top_layer_blocks : ℕ := 3

-- Define the total number of sandstone blocks
def total_blocks_pyramid : ℕ :=
  let rec total_blocks (layer : ℕ) (blocks : ℕ) :=
    if layer = 0 then blocks
    else blocks + total_blocks (layer - 1) (blocks * block_multiplier)
  total_blocks (num_layers - 1) top_layer_blocks

theorem pyramid_total_blocks :
  total_blocks_pyramid = 312093 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_pyramid_total_blocks_l1228_122878


namespace NUMINAMATH_GPT_required_large_loans_l1228_122809

-- We start by introducing the concepts of the number of small, medium, and large loans
def small_loans : Type := ℕ
def medium_loans : Type := ℕ
def large_loans : Type := ℕ

-- Definition of the conditions as two scenarios
def Scenario1 (m s b : ℕ) : Prop := (m = 9 ∧ s = 6 ∧ b = 1)
def Scenario2 (m s b : ℕ) : Prop := (m = 3 ∧ s = 2 ∧ b = 3)

-- Definition of the problem
theorem required_large_loans (m s b : ℕ) (H1 : Scenario1 m s b) (H2 : Scenario2 m s b) :
  b = 4 :=
sorry

end NUMINAMATH_GPT_required_large_loans_l1228_122809


namespace NUMINAMATH_GPT_rhombus_side_length_l1228_122861

-- Define the conditions including the diagonals and area of the rhombus
def diagonal_ratio (d1 d2 : ℝ) : Prop := d1 = 3 * d2
def area_rhombus (b : ℝ) (K : ℝ) : Prop := K = (1 / 2) * b * (3 * b)

-- Define the side length of the rhombus in terms of K
noncomputable def side_length (K : ℝ) : ℝ := Real.sqrt (5 * K / 3)

-- The main theorem statement
theorem rhombus_side_length (K : ℝ) (b : ℝ) (h1 : diagonal_ratio (3 * b) b) (h2 : area_rhombus b K) : 
  side_length K = Real.sqrt (5 * K / 3) := 
sorry

end NUMINAMATH_GPT_rhombus_side_length_l1228_122861


namespace NUMINAMATH_GPT_figure_100_squares_l1228_122847

-- Define the initial conditions as given in the problem
def squares_in_figure (n : ℕ) : ℕ :=
  match n with
  | 0 => 3
  | 1 => 11
  | 2 => 25
  | 3 => 45
  | _ => sorry

-- Define the quadratic formula assumed from the problem conditions
def quadratic_formula (n : ℕ) : ℕ :=
  3 * n^2 + 5 * n + 3

-- Theorem: For figure 100, the number of squares is 30503
theorem figure_100_squares :
  squares_in_figure 100 = quadratic_formula 100 :=
by
  sorry

end NUMINAMATH_GPT_figure_100_squares_l1228_122847


namespace NUMINAMATH_GPT_lisa_score_is_85_l1228_122899

def score_formula (c w : ℕ) : ℕ := 30 + 4 * c - w

theorem lisa_score_is_85 (c w : ℕ) 
  (score_equality : 85 = score_formula c w)
  (non_neg_w : w ≥ 0)
  (total_questions : c + w ≤ 30) :
  (c = 14 ∧ w = 1) :=
by
  sorry

end NUMINAMATH_GPT_lisa_score_is_85_l1228_122899


namespace NUMINAMATH_GPT_find_C_l1228_122898

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

noncomputable def A : ℕ := sum_of_digits (4568 ^ 7777)
noncomputable def B : ℕ := sum_of_digits A
noncomputable def C : ℕ := sum_of_digits B

theorem find_C : C = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_C_l1228_122898


namespace NUMINAMATH_GPT_domain_of_sqrt_expression_l1228_122808

theorem domain_of_sqrt_expression :
  {x : ℝ | x^2 - 5 * x - 6 ≥ 0} = {x : ℝ | x ≤ -1} ∪ {x : ℝ | x ≥ 6} := by
sorry

end NUMINAMATH_GPT_domain_of_sqrt_expression_l1228_122808


namespace NUMINAMATH_GPT_factorize_expression_l1228_122827

theorem factorize_expression (x : ℝ) : 
  x^3 - 5 * x^2 + 4 * x = x * (x - 1) * (x - 4) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l1228_122827


namespace NUMINAMATH_GPT_interval_of_x_l1228_122810

theorem interval_of_x (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ (1 / 2 < x ∧ x < 3 / 5) :=
by
  sorry

end NUMINAMATH_GPT_interval_of_x_l1228_122810


namespace NUMINAMATH_GPT_cloak_change_in_silver_l1228_122850

theorem cloak_change_in_silver :
  (∀ c : ℤ, (20 = c + 4) → (15 = c + 1)) →
  (5 * g = 3) →
  14 * gold / exchange_rate = 10 := 
sorry

end NUMINAMATH_GPT_cloak_change_in_silver_l1228_122850


namespace NUMINAMATH_GPT_determine_range_of_k_l1228_122800

noncomputable def inequality_holds_for_all_x (k : ℝ) : Prop :=
  ∀ (x : ℝ), x^4 + (k - 1) * x^2 + 1 ≥ 0

theorem determine_range_of_k (k : ℝ) : inequality_holds_for_all_x k ↔ k ≥ 1 := sorry

end NUMINAMATH_GPT_determine_range_of_k_l1228_122800


namespace NUMINAMATH_GPT_bailey_rawhide_bones_l1228_122886

variable (dog_treats : ℕ) (chew_toys : ℕ) (total_items : ℕ)
variable (credit_cards : ℕ) (items_per_card : ℕ)

theorem bailey_rawhide_bones :
  (dog_treats = 8) →
  (chew_toys = 2) →
  (credit_cards = 4) →
  (items_per_card = 5) →
  (total_items = credit_cards * items_per_card) →
  (total_items - (dog_treats + chew_toys) = 10) :=
by
  intros
  sorry

end NUMINAMATH_GPT_bailey_rawhide_bones_l1228_122886


namespace NUMINAMATH_GPT_relationship_between_y1_y2_l1228_122843

theorem relationship_between_y1_y2 (y1 y2 : ℝ)
  (h1 : y1 = -2 * (-2) + 3)
  (h2 : y2 = -2 * 3 + 3) :
  y1 > y2 := by
  sorry

end NUMINAMATH_GPT_relationship_between_y1_y2_l1228_122843


namespace NUMINAMATH_GPT_number_of_boys_and_girls_l1228_122897

theorem number_of_boys_and_girls (b g : ℕ) 
    (h1 : ∀ n : ℕ, (n ≥ 1) → ∃ (a_n : ℕ), a_n = 2 * n + 1)
    (h2 : (2 * b + 1 = g))
    : b = (g - 1) / 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_boys_and_girls_l1228_122897


namespace NUMINAMATH_GPT_contrapositive_example_l1228_122891

theorem contrapositive_example (a b : ℝ) : (a ≠ 0 ∨ b ≠ 0) → (a^2 + b^2 ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_example_l1228_122891


namespace NUMINAMATH_GPT_triangle_inequalities_l1228_122802

theorem triangle_inequalities (a b c : ℝ) (h : a < b + c) : b < a + c ∧ c < a + b := 
  sorry

end NUMINAMATH_GPT_triangle_inequalities_l1228_122802


namespace NUMINAMATH_GPT_zero_point_interval_l1228_122872

variable (f : ℝ → ℝ)
variable (f_deriv : ℝ → ℝ)
variable (e : ℝ)
variable (monotonic_f : MonotoneOn f (Set.Ioi 0))

noncomputable def condition1_property (x : ℝ) (h : 0 < x) : f (f x - Real.log x) = Real.exp 1 + 1 := sorry
noncomputable def derivative_property (x : ℝ) (h : 0 < x) : f_deriv x = (deriv f) x := sorry

theorem zero_point_interval :
  ∃ x ∈ Set.Ioo 1 2, f x - f_deriv x - e = 0 := sorry

end NUMINAMATH_GPT_zero_point_interval_l1228_122872


namespace NUMINAMATH_GPT_river_depth_conditions_l1228_122883

noncomputable def depth_beginning_may : ℝ := 15
noncomputable def depth_increase_june : ℝ := 11.25

theorem river_depth_conditions (d k : ℝ)
  (h1 : ∃ d, d = depth_beginning_may) 
  (h2 : 1.5 * d + k = 45)
  (h3 : k = 0.75 * d) :
  d = depth_beginning_may ∧ k = depth_increase_june :=
by
  have H : d = 15 := sorry
  have K : k = 11.25 := sorry
  exact ⟨H, K⟩

end NUMINAMATH_GPT_river_depth_conditions_l1228_122883


namespace NUMINAMATH_GPT_max_length_interval_l1228_122895

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := ((m ^ 2 + m) * x - 1) / (m ^ 2 * x)

theorem max_length_interval (a b m : ℝ) (h1 : m ≠ 0) (h2 : ∀ x, f m x = x → x ∈ Set.Icc a b) :
  |b - a| = (2 * Real.sqrt 3) / 3 := sorry

end NUMINAMATH_GPT_max_length_interval_l1228_122895


namespace NUMINAMATH_GPT_johns_watermelon_weight_l1228_122822

-- Michael's largest watermelon weighs 8 pounds
def michael_weight : ℕ := 8

-- Clay's watermelon weighs three times the size of Michael's watermelon
def clay_weight : ℕ := 3 * michael_weight

-- John's watermelon weighs half the size of Clay's watermelon
def john_weight : ℕ := clay_weight / 2

-- Prove that John's watermelon weighs 12 pounds
theorem johns_watermelon_weight : john_weight = 12 := by
  sorry

end NUMINAMATH_GPT_johns_watermelon_weight_l1228_122822


namespace NUMINAMATH_GPT_common_number_in_lists_l1228_122831

theorem common_number_in_lists (nums : List ℚ) (h_len : nums.length = 9)
  (h_first_five_avg : (nums.take 5).sum / 5 = 7)
  (h_last_five_avg : (nums.drop 4).sum / 5 = 9)
  (h_total_avg : nums.sum / 9 = 73/9) :
  ∃ x, x ∈ nums.take 5 ∧ x ∈ nums.drop 4 ∧ x = 7 := 
sorry

end NUMINAMATH_GPT_common_number_in_lists_l1228_122831


namespace NUMINAMATH_GPT_find_third_number_l1228_122803

-- Define the conditions
def equation1_valid : Prop := (5 * 3 = 15) ∧ (5 * 2 = 10) ∧ (2 * 1000 + 3 * 100 + 5 = 1022)
def equation2_valid : Prop := (9 * 2 = 18) ∧ (9 * 4 = 36) ∧ (4 * 1000 + 2 * 100 + 9 = 3652)

-- The theorem to prove
theorem find_third_number (h1 : equation1_valid) (h2 : equation2_valid) : (7 * 2 = 14) ∧ (7 * 5 = 35) ∧ (5 * 1000 + 2 * 100 + 7 = 547) :=
by 
  sorry

end NUMINAMATH_GPT_find_third_number_l1228_122803


namespace NUMINAMATH_GPT_eval_fraction_power_l1228_122828

theorem eval_fraction_power : (0.5 ^ 4 / 0.05 ^ 3) = 500 := by
  sorry

end NUMINAMATH_GPT_eval_fraction_power_l1228_122828


namespace NUMINAMATH_GPT_chord_length_through_focus_l1228_122868

theorem chord_length_through_focus (x y : ℝ) (h : x^2 / 4 + y^2 / 3 = 1)
  (h_perp : (x = 1) ∨ (x = -1)) : abs (2 * y) = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_chord_length_through_focus_l1228_122868


namespace NUMINAMATH_GPT_shooter_hit_rate_l1228_122889

noncomputable def shooter_prob := 2 / 3

theorem shooter_hit_rate:
  ∀ (x : ℚ), (1 - x)^4 = 1 / 81 → x = shooter_prob :=
by
  intro x h
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_shooter_hit_rate_l1228_122889


namespace NUMINAMATH_GPT_total_time_is_three_hours_l1228_122875

-- Define the conditions of the problem in Lean
def time_uber_house := 10
def time_uber_airport := 5 * time_uber_house
def time_check_bag := 15
def time_security := 3 * time_check_bag
def time_boarding := 20
def time_takeoff := 2 * time_boarding

-- Total time in minutes
def total_time_minutes := time_uber_house + time_uber_airport + time_check_bag + time_security + time_boarding + time_takeoff

-- Conversion from minutes to hours
def total_time_hours := total_time_minutes / 60

-- The theorem to prove
theorem total_time_is_three_hours : total_time_hours = 3 := by
  sorry

end NUMINAMATH_GPT_total_time_is_three_hours_l1228_122875


namespace NUMINAMATH_GPT_Mark_jump_rope_hours_l1228_122864

theorem Mark_jump_rope_hours 
  (record : ℕ) 
  (jump_rate : ℕ) 
  (seconds_per_hour : ℕ) 
  (h_record : record = 54000) 
  (h_jump_rate : jump_rate = 3) 
  (h_seconds_per_hour : seconds_per_hour = 3600) 
  : (record / jump_rate) / seconds_per_hour = 5 := 
by
  sorry

end NUMINAMATH_GPT_Mark_jump_rope_hours_l1228_122864


namespace NUMINAMATH_GPT_MrMartinBought2Cups_l1228_122845

theorem MrMartinBought2Cups (c b : ℝ) (x : ℝ) (h1 : 3 * c + 2 * b = 12.75)
                             (h2 : x * c + 5 * b = 14.00)
                             (hb : b = 1.5) :
  x = 2 :=
sorry

end NUMINAMATH_GPT_MrMartinBought2Cups_l1228_122845


namespace NUMINAMATH_GPT_cos2theta_sin2theta_l1228_122892

theorem cos2theta_sin2theta (θ : ℝ) (h : 2 * Real.cos θ + Real.sin θ = 0) :
  Real.cos (2 * θ) + (1 / 2) * Real.sin (2 * θ) = -1 :=
sorry

end NUMINAMATH_GPT_cos2theta_sin2theta_l1228_122892


namespace NUMINAMATH_GPT_david_pushups_difference_l1228_122848

-- Definitions based on conditions
def zachary_pushups : ℕ := 44
def total_pushups : ℕ := 146

-- The number of push-ups David did more than Zachary
def david_more_pushups_than_zachary (D : ℕ) := D - zachary_pushups

-- The theorem we need to prove
theorem david_pushups_difference :
  ∃ D : ℕ, D > zachary_pushups ∧ D + zachary_pushups = total_pushups ∧ david_more_pushups_than_zachary D = 58 :=
by
  -- We leave the proof as an exercise or for further filling.
  sorry

end NUMINAMATH_GPT_david_pushups_difference_l1228_122848


namespace NUMINAMATH_GPT_number_of_divisors_of_2018_or_2019_is_7_l1228_122871

theorem number_of_divisors_of_2018_or_2019_is_7 (h1 : Prime 673) (h2 : Prime 1009) : 
  Nat.card {d : Nat | d ∣ 2018 ∨ d ∣ 2019} = 7 := 
  sorry

end NUMINAMATH_GPT_number_of_divisors_of_2018_or_2019_is_7_l1228_122871


namespace NUMINAMATH_GPT_value_added_to_number_l1228_122873

theorem value_added_to_number (n v : ℤ) (h1 : n = 9)
  (h2 : 3 * (n + 2) = v + n) : v = 24 :=
by
  sorry

end NUMINAMATH_GPT_value_added_to_number_l1228_122873


namespace NUMINAMATH_GPT_place_synthetic_method_l1228_122855

theorem place_synthetic_method :
  "Synthetic Method" = "Direct Proof" :=
sorry

end NUMINAMATH_GPT_place_synthetic_method_l1228_122855


namespace NUMINAMATH_GPT_sheila_earning_per_hour_l1228_122824

def sheila_hours_per_day_mwf : ℕ := 8
def sheila_days_mwf : ℕ := 3
def sheila_hours_per_day_tt : ℕ := 6
def sheila_days_tt : ℕ := 2
def sheila_total_earnings : ℕ := 432

theorem sheila_earning_per_hour : (sheila_total_earnings / (sheila_hours_per_day_mwf * sheila_days_mwf + sheila_hours_per_day_tt * sheila_days_tt)) = 12 := by
  sorry

end NUMINAMATH_GPT_sheila_earning_per_hour_l1228_122824


namespace NUMINAMATH_GPT_fraction_subtraction_simplified_l1228_122834

theorem fraction_subtraction_simplified :
  (8 / 21 - 3 / 63) = 1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_fraction_subtraction_simplified_l1228_122834


namespace NUMINAMATH_GPT_greatest_even_integer_leq_z_l1228_122882

theorem greatest_even_integer_leq_z (z : ℝ) (z_star : ℝ → ℝ)
  (h1 : ∀ z, z_star z = z_star (z - (z - z_star z))) -- (This is to match the definition given)
  (h2 : 6.30 - z_star 6.30 = 0.2999999999999998) : z_star 6.30 ≤ 6.30 := by
sorry

end NUMINAMATH_GPT_greatest_even_integer_leq_z_l1228_122882


namespace NUMINAMATH_GPT_inversely_proportional_example_l1228_122884

theorem inversely_proportional_example (x y k : ℝ) (h₁ : x * y = k) (h₂ : x = 30) (h₃ : y = 8) :
  y = 24 → x = 10 :=
by
  sorry

end NUMINAMATH_GPT_inversely_proportional_example_l1228_122884


namespace NUMINAMATH_GPT_correct_options_l1228_122896

-- Given conditions
def f : ℝ → ℝ := sorry -- We will assume there is some function f that satisfies the conditions

axiom xy_identity (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x * y) = x * f y + y * f x
axiom f_positive (x : ℝ) (hx : 1 < x) : 0 < f x

-- Proof of the required conclusion
theorem correct_options (h1 : f 1 = 0) (h2 : ∀ x y, f (x * y) ≠ f x * f y)
  (h3 : ∀ x, 1 < x → ∀ y, 1 < y → x < y → f x < f y)
  (h4 : ∀ x, 2 ≤ x → x * f (x - 3 / 2) ≥ (3 / 2 - x) * f x) : 
  f 1 = 0 ∧ (∀ x y, f (x * y) ≠ f x * f y) ∧ (∀ x, 1 < x → ∀ y, 1 < y → x < y → f x < f y) ∧ (∀ x, 2 ≤ x → x * f (x - 3 / 2) ≥ (3 / 2 - x) * f x) :=
sorry

end NUMINAMATH_GPT_correct_options_l1228_122896


namespace NUMINAMATH_GPT_sum_of_digits_of_10_pow_30_minus_36_l1228_122841

def sum_of_digits (n : ℕ) : ℕ := 
  (n.digits 10).sum

theorem sum_of_digits_of_10_pow_30_minus_36 : 
  sum_of_digits (10^30 - 36) = 11 := 
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_10_pow_30_minus_36_l1228_122841


namespace NUMINAMATH_GPT_strongest_correlation_l1228_122819

variables (r1 : ℝ) (r2 : ℝ) (r3 : ℝ) (r4 : ℝ)
variables (abs_r3 : ℝ)

-- Define conditions as hypotheses
def conditions :=
  r1 = 0 ∧ r2 = -0.95 ∧ abs_r3 = 0.89 ∧ r4 = 0.75 ∧ abs r3 = abs_r3

-- Theorem stating the correct answer
theorem strongest_correlation (hyp : conditions r1 r2 r3 r4 abs_r3) : 
  abs r2 > abs r1 ∧ abs r2 > abs r3 ∧ abs r2 > abs r4 :=
by sorry

end NUMINAMATH_GPT_strongest_correlation_l1228_122819


namespace NUMINAMATH_GPT_work_done_by_6_men_and_11_women_l1228_122818

-- Definitions based on conditions
def work_completed_by_men (men : ℕ) (days : ℕ) : ℚ := men / (8 * days)
def work_completed_by_women (women : ℕ) (days : ℕ) : ℚ := women / (12 * days)
def combined_work_rate (men : ℕ) (women : ℕ) (days : ℕ) : ℚ := 
  work_completed_by_men men days + work_completed_by_women women days

-- Problem statement
theorem work_done_by_6_men_and_11_women :
  combined_work_rate 6 11 12 = 1 := by
  sorry

end NUMINAMATH_GPT_work_done_by_6_men_and_11_women_l1228_122818


namespace NUMINAMATH_GPT_set_inter_complement_eq_l1228_122859

-- Given conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x^2 < 1}
def B : Set ℝ := {x | x^2 - 2 * x > 0}

-- Question translated to proof problem statement
theorem set_inter_complement_eq :
  A ∩ (U \ B) = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_set_inter_complement_eq_l1228_122859


namespace NUMINAMATH_GPT_job_planned_completion_days_l1228_122816

noncomputable def initial_days_planned (W D : ℝ) := 6 * (W / D) = (W - 3 * (W / D)) / 3

theorem job_planned_completion_days (W : ℝ ) : 
  ∃ D : ℝ, initial_days_planned W D ∧ D = 6 := 
sorry

end NUMINAMATH_GPT_job_planned_completion_days_l1228_122816


namespace NUMINAMATH_GPT_prove_a_pow_minus_b_l1228_122846

-- Definitions of conditions
variables (x a b : ℝ)

def condition_1 : Prop := x - a > 2
def condition_2 : Prop := 2 * x - b < 0
def solution_set_condition : Prop := -1 < x ∧ x < 1
def derived_a : Prop := a + 2 = -1
def derived_b : Prop := b / 2 = 1

-- The main theorem to prove
theorem prove_a_pow_minus_b (h1 : condition_1 x a) (h2 : condition_2 x b) (h3 : solution_set_condition x) (ha : derived_a a) (hb : derived_b b) : a^(-b) = (1 / 9) :=
by
  sorry

end NUMINAMATH_GPT_prove_a_pow_minus_b_l1228_122846


namespace NUMINAMATH_GPT_max_value_function_max_value_expression_l1228_122857

theorem max_value_function (x a : ℝ) (hx : x > 0) (ha : a > 2 * x) : ∃ y : ℝ, y = (a^2) / 8 :=
by
  sorry

theorem max_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^2 + b^2 + c^2 = 4) : 
   ∃ m : ℝ, m = 4 :=
by
  sorry

end NUMINAMATH_GPT_max_value_function_max_value_expression_l1228_122857


namespace NUMINAMATH_GPT_range_of_a_l1228_122817

theorem range_of_a (a : ℝ) (hx : ∀ x : ℝ, x ≥ 1 → x^2 + a * x + 9 ≥ 0) : a ≥ -6 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1228_122817


namespace NUMINAMATH_GPT_hidden_prime_average_correct_l1228_122869

noncomputable def hidden_prime_average : ℚ :=
  (13 + 17 + 59) / 3

theorem hidden_prime_average_correct :
  hidden_prime_average = 29.6 :=
by
  sorry

end NUMINAMATH_GPT_hidden_prime_average_correct_l1228_122869


namespace NUMINAMATH_GPT_find_m_l1228_122835

open Real

noncomputable def a : ℝ × ℝ := (1, sqrt 3)
noncomputable def b (m : ℝ) : ℝ × ℝ := (3, m)
noncomputable def dot_prod (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem find_m (m : ℝ) (h : dot_prod a (b m) / magnitude a = 3) : m = sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1228_122835


namespace NUMINAMATH_GPT_rotameter_percentage_l1228_122837

theorem rotameter_percentage (l_inch_flow : ℝ) (l_liters_flow : ℝ) (g_inch_flow : ℝ) (g_liters_flow : ℝ) :
  l_inch_flow = 2.5 → l_liters_flow = 60 → g_inch_flow = 4 → g_liters_flow = 192 → 
  (g_liters_flow / g_inch_flow) / (l_liters_flow / l_inch_flow) * 100 = 200 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_rotameter_percentage_l1228_122837


namespace NUMINAMATH_GPT_unit_digit_of_15_pow_l1228_122885

-- Define the conditions
def base_number : ℕ := 15
def base_unit_digit : ℕ := 5

-- State the question and objective in Lean 4
theorem unit_digit_of_15_pow (X : ℕ) (h : 0 < X) : (15^X) % 10 = 5 :=
sorry

end NUMINAMATH_GPT_unit_digit_of_15_pow_l1228_122885


namespace NUMINAMATH_GPT_system_of_equations_solution_non_negative_system_of_equations_solution_positive_sum_l1228_122865

theorem system_of_equations_solution_non_negative (x y : ℝ) (h1 : x^3 + y^3 + 3 * x * y = 1) (h2 : x^2 - y^2 = 1) (h3 : x ≥ 0) (h4 : y ≥ 0) : x = 1 ∧ y = 0 :=
sorry

theorem system_of_equations_solution_positive_sum (x y : ℝ) (h1 : x^3 + y^3 + 3 * x * y = 1) (h2 : x^2 - y^2 = 1) (h3 : x + y > 0) : x = 1 ∧ y = 0 :=
sorry

end NUMINAMATH_GPT_system_of_equations_solution_non_negative_system_of_equations_solution_positive_sum_l1228_122865


namespace NUMINAMATH_GPT_sticks_in_100th_stage_l1228_122856

theorem sticks_in_100th_stage : 
  ∀ (n a₁ d : ℕ), a₁ = 5 → d = 4 → n = 100 → a₁ + (n - 1) * d = 401 :=
by
  sorry

end NUMINAMATH_GPT_sticks_in_100th_stage_l1228_122856


namespace NUMINAMATH_GPT_y_intercept_of_line_b_is_minus_8_l1228_122820

/-- Define a line in slope-intercept form y = mx + c --/
structure Line :=
  (m : ℝ)   -- slope
  (c : ℝ)   -- y-intercept

/-- Define a point in 2D Cartesian coordinate system --/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Define conditions for the problem --/
def line_b_parallel_to (l: Line) (p: Point) : Prop :=
  l.m = 2 ∧ p.x = 3 ∧ p.y = -2

/-- Define the target statement to prove --/
theorem y_intercept_of_line_b_is_minus_8 :
  ∀ (b: Line) (p: Point), line_b_parallel_to b p → b.c = -8 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_b_is_minus_8_l1228_122820


namespace NUMINAMATH_GPT_total_pears_l1228_122801

def jason_pears : Nat := 46
def keith_pears : Nat := 47
def mike_pears : Nat := 12

theorem total_pears : jason_pears + keith_pears + mike_pears = 105 := by
  sorry

end NUMINAMATH_GPT_total_pears_l1228_122801
