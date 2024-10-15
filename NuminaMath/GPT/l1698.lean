import Mathlib

namespace NUMINAMATH_GPT_dog_count_l1698_169875

theorem dog_count 
  (long_furred : ℕ) 
  (brown : ℕ) 
  (neither : ℕ) 
  (long_furred_brown : ℕ) 
  (total : ℕ) 
  (h1 : long_furred = 29) 
  (h2 : brown = 17) 
  (h3 : neither = 8) 
  (h4 : long_furred_brown = 9)
  (h5 : total = long_furred + brown - long_furred_brown + neither) : 
  total = 45 :=
by 
  sorry

end NUMINAMATH_GPT_dog_count_l1698_169875


namespace NUMINAMATH_GPT_sum_of_coordinates_of_A_l1698_169826

noncomputable def point := (ℝ × ℝ)
def B : point := (2, 6)
def C : point := (4, 12)
def AC (A C : point) : ℝ := (A.1 - C.1)^2 + (A.2 - C.2)^2
def AB (A B : point) : ℝ := (A.1 - B.1)^2 + (A.2 - B.2)^2
def BC (B C : point) : ℝ := (B.1 - C.1)^2 + (B.2 - C.2)^2

theorem sum_of_coordinates_of_A :
  ∃ A : point, AC A C / AB A B = (1/3) ∧ BC B C / AB A B = (1/3) ∧ A.1 + A.2 = 24 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_of_A_l1698_169826


namespace NUMINAMATH_GPT_sum_of_coefficients_of_y_terms_l1698_169866

theorem sum_of_coefficients_of_y_terms :
  let p := (5 * x + 3 * y + 2) * (2 * x + 6 * y + 7)
  let expanded_p := 10 * x^2 + 36 * x * y + 39 * x + 18 * y^2 + 33 * y + 14
  (36 + 18 + 33) = 87 := by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_of_y_terms_l1698_169866


namespace NUMINAMATH_GPT_integer_cube_less_than_triple_unique_l1698_169891

theorem integer_cube_less_than_triple_unique (x : ℤ) (h : x^3 < 3*x) : x = 1 :=
sorry

end NUMINAMATH_GPT_integer_cube_less_than_triple_unique_l1698_169891


namespace NUMINAMATH_GPT_compute_abc_l1698_169819

theorem compute_abc (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = 30) 
  (h2 : (1 / a + 1 / b + 1 / c + 420 / (a * b * c) = 1)) : 
  a * b * c = 450 := 
sorry

end NUMINAMATH_GPT_compute_abc_l1698_169819


namespace NUMINAMATH_GPT_chord_line_eq_l1698_169862

theorem chord_line_eq (x y : ℝ) (h : x^2 + 4 * y^2 = 36) (midpoint : x = 4 ∧ y = 2) :
  x + 2 * y - 8 = 0 := 
sorry

end NUMINAMATH_GPT_chord_line_eq_l1698_169862


namespace NUMINAMATH_GPT_triangle_ABC_area_l1698_169889

-- Define the vertices of the triangle
def A := (-4, 0)
def B := (24, 0)
def C := (0, 2)

-- Function to calculate the determinant, used for the area calculation
def det (x1 y1 x2 y2 x3 y3 : ℝ) :=
  x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)

-- Area calculation for triangle given vertices using determinant method
noncomputable def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * |det x1 y1 x2 y2 x3 y3|

-- The goal is to prove that the area of triangle ABC is 14
theorem triangle_ABC_area :
  triangle_area (-4) 0 24 0 0 2 = 14 := sorry

end NUMINAMATH_GPT_triangle_ABC_area_l1698_169889


namespace NUMINAMATH_GPT_jenna_round_trip_pay_l1698_169854

-- Definitions based on conditions
def rate : ℝ := 0.40
def one_way_distance : ℝ := 400
def round_trip_distance : ℝ := 2 * one_way_distance

-- Theorem based on the question and correct answer
theorem jenna_round_trip_pay : round_trip_distance * rate = 320 := by
  sorry

end NUMINAMATH_GPT_jenna_round_trip_pay_l1698_169854


namespace NUMINAMATH_GPT_sum_of_inscribed_angles_l1698_169804

theorem sum_of_inscribed_angles 
  (n : ℕ) 
  (total_degrees : ℝ)
  (arcs : ℕ)
  (x_arcs : ℕ)
  (y_arcs : ℕ) 
  (arc_angle : ℝ)
  (x_central_angle : ℝ)
  (y_central_angle : ℝ)
  (x_inscribed_angle : ℝ)
  (y_inscribed_angle : ℝ)
  (total_inscribed_angles : ℝ) :
  n = 18 →
  total_degrees = 360 →
  x_arcs = 3 →
  y_arcs = 5 →
  arc_angle = total_degrees / n →
  x_central_angle = x_arcs * arc_angle →
  y_central_angle = y_arcs * arc_angle →
  x_inscribed_angle = x_central_angle / 2 →
  y_inscribed_angle = y_central_angle / 2 →
  total_inscribed_angles = x_inscribed_angle + y_inscribed_angle →
  total_inscribed_angles = 80 := sorry

end NUMINAMATH_GPT_sum_of_inscribed_angles_l1698_169804


namespace NUMINAMATH_GPT_petya_vasya_same_result_l1698_169832

theorem petya_vasya_same_result (a b : ℤ) 
  (h1 : b = a + 1)
  (h2 : (a - 1) / (b - 2) = (a + 1) / b) :
  (a / b) = 1 := 
by
  sorry

end NUMINAMATH_GPT_petya_vasya_same_result_l1698_169832


namespace NUMINAMATH_GPT_total_red_papers_l1698_169837

-- Defining the number of red papers in one box and the number of boxes Hoseok has
def red_papers_per_box : ℕ := 2
def number_of_boxes : ℕ := 2

-- Statement to prove
theorem total_red_papers : (red_papers_per_box * number_of_boxes) = 4 := by
  sorry

end NUMINAMATH_GPT_total_red_papers_l1698_169837


namespace NUMINAMATH_GPT_problem_statement_l1698_169851

theorem problem_statement (a b c : ℝ) (h : a^2 + b^2 - a * b = c^2) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a - c) * (b - c) ≤ 0 :=
by sorry

end NUMINAMATH_GPT_problem_statement_l1698_169851


namespace NUMINAMATH_GPT_annual_interest_rate_l1698_169873

/-- Suppose you invested $10000, part at a certain annual interest rate and the rest at 9% annual interest.
After one year, you received $684 in interest. You invested $7200 at this rate and the rest at 9%.
What is the annual interest rate of the first investment? -/
theorem annual_interest_rate (r : ℝ) 
  (h : 7200 * r + 2800 * 0.09 = 684) : r = 0.06 :=
by
  sorry

end NUMINAMATH_GPT_annual_interest_rate_l1698_169873


namespace NUMINAMATH_GPT_complex_pow_simplify_l1698_169833

noncomputable def i : ℂ := Complex.I

theorem complex_pow_simplify :
  (1 + Real.sqrt 3 * Complex.I) ^ 3 * Complex.I = -8 * Complex.I :=
by
  sorry

end NUMINAMATH_GPT_complex_pow_simplify_l1698_169833


namespace NUMINAMATH_GPT_passes_through_point_P_l1698_169881

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 7 + a^(x - 1)

theorem passes_through_point_P
  (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : f a 1 = 8 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_passes_through_point_P_l1698_169881


namespace NUMINAMATH_GPT_Frank_is_14_l1698_169855

variable {d e f : ℕ}

theorem Frank_is_14
  (h1 : d + e + f = 30)
  (h2 : f - 5 = d)
  (h3 : e + 2 = 3 * (d + 2) / 4) :
  f = 14 :=
sorry

end NUMINAMATH_GPT_Frank_is_14_l1698_169855


namespace NUMINAMATH_GPT_triangle_inequality_l1698_169868

-- Define the triangle angles, semiperimeter, and circumcircle radius
variables (α β γ s R : Real)

-- Define the sum of angles in a triangle
axiom angle_sum : α + β + γ = Real.pi

-- The inequality to prove
theorem triangle_inequality (h_sum : α + β + γ = Real.pi) :
  (α + β) * (β + γ) * (γ + α) ≤ 4 * (Real.pi / Real.sqrt 3)^3 * R / s := sorry

end NUMINAMATH_GPT_triangle_inequality_l1698_169868


namespace NUMINAMATH_GPT_kyungsoo_came_second_l1698_169887

theorem kyungsoo_came_second
  (kyungsoo_jump : ℝ) (younghee_jump : ℝ) (jinju_jump : ℝ) (chanho_jump : ℝ)
  (h_kyungsoo : kyungsoo_jump = 2.3)
  (h_younghee : younghee_jump = 0.9)
  (h_jinju : jinju_jump = 1.8)
  (h_chanho : chanho_jump = 2.5) :
  kyungsoo_jump = 2.3 := 
by
  sorry

end NUMINAMATH_GPT_kyungsoo_came_second_l1698_169887


namespace NUMINAMATH_GPT_Xiao_Ming_vertical_height_increase_l1698_169813

noncomputable def vertical_height_increase (slope_ratio_v slope_ratio_h : ℝ) (distance : ℝ) : ℝ :=
  let x := distance / (Real.sqrt (1 + (slope_ratio_h / slope_ratio_v)^2))
  x

theorem Xiao_Ming_vertical_height_increase
  (slope_ratio_v slope_ratio_h distance : ℝ)
  (h_ratio : slope_ratio_v = 1)
  (h_ratio2 : slope_ratio_h = 2.4)
  (h_distance : distance = 130) :
  vertical_height_increase slope_ratio_v slope_ratio_h distance = 50 :=
by
  unfold vertical_height_increase
  rw [h_ratio, h_ratio2, h_distance]
  sorry

end NUMINAMATH_GPT_Xiao_Ming_vertical_height_increase_l1698_169813


namespace NUMINAMATH_GPT_find_b_l1698_169824

noncomputable def circle_center_radius : Prop :=
  let C := (2, 0) -- center
  let r := 2 -- radius
  C.1 = 2 ∧ C.2 = 0 ∧ r = 2

noncomputable def line (b : ℝ) : Prop :=
  ∃ M N : ℝ × ℝ, M ≠ N ∧ 
  (M.2 = M.1 + b) ∧ (N.2 = N.1 + b) -- points on the line are M = (x1, x1 + b) and N = (x2, x2 + b)

noncomputable def perpendicular_condition (M N center: ℝ × ℝ) : Prop :=
  (M.1 - center.1) * (N.1 - center.1) + (M.2 - center.2) * (N.2 - center.2) = 0 -- CM ⟂ CN

theorem find_b (b : ℝ) : 
  circle_center_radius ∧
  (∃ M N, line b ∧ perpendicular_condition M N (2, 0)) →
  b = 0 ∨ b = -4 :=
by {
  -- Proof omitted
  sorry
}

end NUMINAMATH_GPT_find_b_l1698_169824


namespace NUMINAMATH_GPT_serving_calculation_correct_l1698_169823

def prepared_orange_juice_servings (cans_of_concentrate : ℕ) 
                                  (oz_per_concentrate_can : ℕ) 
                                  (water_ratio : ℕ) 
                                  (oz_per_serving : ℕ) : ℕ :=
  let total_concentrate := cans_of_concentrate * oz_per_concentrate_can
  let total_water := cans_of_concentrate * water_ratio * oz_per_concentrate_can
  let total_juice := total_concentrate + total_water
  total_juice / oz_per_serving

theorem serving_calculation_correct :
  prepared_orange_juice_servings 60 5 3 6 = 200 := by
  sorry

end NUMINAMATH_GPT_serving_calculation_correct_l1698_169823


namespace NUMINAMATH_GPT_intersection_M_N_l1698_169801

def M : Set ℝ := {y | ∃ x, x ∈ Set.Icc (-5) 5 ∧ y = 2 * Real.sin x}
def N : Set ℝ := {x | ∃ y, y = Real.log (x - 1) / Real.log 2}

theorem intersection_M_N : {x | 1 < x ∧ x ≤ 2} = {x | x ∈ M ∩ N} :=
by sorry

end NUMINAMATH_GPT_intersection_M_N_l1698_169801


namespace NUMINAMATH_GPT_smallest_integer_gcd_6_l1698_169806

theorem smallest_integer_gcd_6 : ∃ n : ℕ, n > 100 ∧ gcd n 18 = 6 ∧ ∀ m : ℕ, (m > 100 ∧ gcd m 18 = 6) → m ≥ n :=
by
  let n := 114
  have h1 : n > 100 := sorry
  have h2 : gcd n 18 = 6 := sorry
  have h3 : ∀ m : ℕ, (m > 100 ∧ gcd m 18 = 6) → m ≥ n := sorry
  exact ⟨n, h1, h2, h3⟩

end NUMINAMATH_GPT_smallest_integer_gcd_6_l1698_169806


namespace NUMINAMATH_GPT_increasing_interval_of_f_l1698_169827

def f (x : ℝ) : ℝ := (x - 1) ^ 2 - 2

theorem increasing_interval_of_f : ∀ x, 1 < x → f x > f 1 := 
sorry

end NUMINAMATH_GPT_increasing_interval_of_f_l1698_169827


namespace NUMINAMATH_GPT_range_of_a_l1698_169859

variable (x a : ℝ)
def inequality_sys := x < a ∧ x < 3
def solution_set := x < a

theorem range_of_a (h : ∀ x, inequality_sys x a → solution_set x a) : a ≤ 3 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1698_169859


namespace NUMINAMATH_GPT_proposition_2_proposition_4_l1698_169811

-- Definitions from conditions.
def circle_M (x y q : ℝ) : Prop := (x + Real.cos q)^2 + (y - Real.sin q)^2 = 1
def line_l (y k x : ℝ) : Prop := y = k * x

-- Prove that the line l and circle M always intersect for any real k and q.
theorem proposition_2 : ∀ (k q : ℝ), ∃ (x y : ℝ), circle_M x y q ∧ line_l y k x := sorry

-- Prove that for any real k, there exists a real q such that the line l is tangent to the circle M.
theorem proposition_4 : ∀ (k : ℝ), ∃ (q x y : ℝ), circle_M x y q ∧ line_l y k x ∧
  (abs (Real.sin q + k * Real.cos q) = 1 / Real.sqrt (1 + k^2)) := sorry

end NUMINAMATH_GPT_proposition_2_proposition_4_l1698_169811


namespace NUMINAMATH_GPT_negation_of_existence_implies_universal_l1698_169812

theorem negation_of_existence_implies_universal :
  ¬ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ ∀ x : ℝ, x^2 + 2*x + 2 > 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existence_implies_universal_l1698_169812


namespace NUMINAMATH_GPT_y_coordinate_equidistant_l1698_169807

theorem y_coordinate_equidistant :
  ∃ y : ℝ, (∀ ptC ptD : ℝ × ℝ, ptC = (-3, 0) → ptD = (4, 5) → 
    dist (0, y) ptC = dist (0, y) ptD) ∧ y = 16 / 5 :=
by
  sorry

end NUMINAMATH_GPT_y_coordinate_equidistant_l1698_169807


namespace NUMINAMATH_GPT_find_speed_of_man_in_still_water_l1698_169857

noncomputable def speed_of_man_in_still_water (v_m v_s : ℝ) : Prop :=
  (v_m + v_s) * 3 = 42 ∧ (v_m - v_s) * 3 = 18

theorem find_speed_of_man_in_still_water (v_s : ℝ) : ∃ v_m : ℝ, speed_of_man_in_still_water v_m v_s ∧ v_m = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_speed_of_man_in_still_water_l1698_169857


namespace NUMINAMATH_GPT_expand_polynomial_l1698_169840

noncomputable def polynomial_expression (x : ℝ) : ℝ := -2 * (x - 3) * (x + 4) * (2 * x - 1)

theorem expand_polynomial (x : ℝ) :
  polynomial_expression x = -4 * x^3 - 2 * x^2 + 50 * x - 24 :=
sorry

end NUMINAMATH_GPT_expand_polynomial_l1698_169840


namespace NUMINAMATH_GPT_overtaking_time_l1698_169896

variable (a_speed b_speed k_speed : ℕ)
variable (b_delay : ℕ) 
variable (t : ℕ)
variable (t_k : ℕ)

theorem overtaking_time (h1 : a_speed = 30)
                        (h2 : b_speed = 40)
                        (h3 : k_speed = 60)
                        (h4 : b_delay = 5)
                        (h5 : 30 * t = 40 * (t - 5))
                        (h6 : 30 * t = 60 * t_k)
                         : k_speed / 3 = 10 :=
by sorry

end NUMINAMATH_GPT_overtaking_time_l1698_169896


namespace NUMINAMATH_GPT_juan_speed_l1698_169893

theorem juan_speed (J : ℝ) :
  (∀ (time : ℝ) (distance : ℝ) (peter_speed : ℝ),
    time = 1.5 →
    distance = 19.5 →
    peter_speed = 5 →
    distance = J * time + peter_speed * time) →
  J = 8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_juan_speed_l1698_169893


namespace NUMINAMATH_GPT_return_trip_speed_l1698_169861

theorem return_trip_speed (d xy_dist : ℝ) (s xy_speed : ℝ) (avg_speed : ℝ) (r return_speed : ℝ) :
  xy_dist = 150 →
  xy_speed = 75 →
  avg_speed = 50 →
  2 * xy_dist / ((xy_dist / xy_speed) + (xy_dist / return_speed)) = avg_speed →
  return_speed = 37.5 :=
by
  intros hxy_dist hxy_speed h_avg_speed h_avg_speed_eq
  sorry

end NUMINAMATH_GPT_return_trip_speed_l1698_169861


namespace NUMINAMATH_GPT_find_y_l1698_169899

theorem find_y (y : ℝ) (h : (3 * y) / 7 = 12) : y = 28 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_find_y_l1698_169899


namespace NUMINAMATH_GPT_find_x_l1698_169850

theorem find_x
  (x : ℝ)
  (h : 0.20 * x = 0.40 * 140 + 80) :
  x = 680 := 
sorry

end NUMINAMATH_GPT_find_x_l1698_169850


namespace NUMINAMATH_GPT_angle_measure_l1698_169842

theorem angle_measure (y : ℝ) (hyp : 45 + 3 * y + y = 180) : y = 33.75 :=
by
  sorry

end NUMINAMATH_GPT_angle_measure_l1698_169842


namespace NUMINAMATH_GPT_polynomial_factorization_l1698_169852

theorem polynomial_factorization (x y : ℝ) : -(2 * x - y) * (2 * x + y) = -4 * x ^ 2 + y ^ 2 :=
by sorry

end NUMINAMATH_GPT_polynomial_factorization_l1698_169852


namespace NUMINAMATH_GPT_nearest_integer_to_expr_l1698_169883

theorem nearest_integer_to_expr : 
  let a := 3 + Real.sqrt 5
  let b := (a)^6
  abs (b - 2744) < 1
:= sorry

end NUMINAMATH_GPT_nearest_integer_to_expr_l1698_169883


namespace NUMINAMATH_GPT_calculate_expression_l1698_169834

-- Define the expression x + x * (factorial x)^x
def expression (x : ℕ) : ℕ :=
  x + x * (Nat.factorial x) ^ x

-- Set the value of x
def x_value : ℕ := 3

-- State the proposition
theorem calculate_expression : expression x_value = 651 := 
by 
  -- By substitution and calculation, the proof follows.
  sorry

end NUMINAMATH_GPT_calculate_expression_l1698_169834


namespace NUMINAMATH_GPT_triangle_inequality_l1698_169821

variables (a b c : ℝ)

theorem triangle_inequality (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) / (1 + a + b) > c / (1 + c) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l1698_169821


namespace NUMINAMATH_GPT_ab_plus_b_l1698_169802

theorem ab_plus_b (A B : ℤ) (h1 : A * B = 10) (h2 : 3 * A + 7 * B = 51) : A * B + B = 12 :=
by
  sorry

end NUMINAMATH_GPT_ab_plus_b_l1698_169802


namespace NUMINAMATH_GPT_three_digit_cubes_divisible_by_16_l1698_169817

theorem three_digit_cubes_divisible_by_16 (n : ℤ) (x : ℤ) 
  (h_cube : x = n^3)
  (h_div : 16 ∣ x) 
  (h_3digit : 100 ≤ x ∧ x ≤ 999) : 
  x = 512 := 
by {
  sorry
}

end NUMINAMATH_GPT_three_digit_cubes_divisible_by_16_l1698_169817


namespace NUMINAMATH_GPT_initial_population_is_9250_l1698_169869

noncomputable def initial_population : ℝ :=
  let final_population := 6514
  let factor := (1.08 * 0.85 * (1.02)^5 * 0.95 * 0.9)
  final_population / factor

theorem initial_population_is_9250 : initial_population = 9250 := by
  sorry

end NUMINAMATH_GPT_initial_population_is_9250_l1698_169869


namespace NUMINAMATH_GPT_swimming_championship_l1698_169848

theorem swimming_championship (num_swimmers : ℕ) (lanes : ℕ) (advance : ℕ) (eliminated : ℕ) (total_races : ℕ) : 
  num_swimmers = 300 → 
  lanes = 8 → 
  advance = 2 → 
  eliminated = 6 → 
  total_races = 53 :=
by
  intros
  sorry

end NUMINAMATH_GPT_swimming_championship_l1698_169848


namespace NUMINAMATH_GPT_solve_comb_eq_l1698_169874

open Nat

def comb (n k : ℕ) : ℕ := (factorial n) / ((factorial k) * (factorial (n - k)))
def perm (n k : ℕ) : ℕ := (factorial n) / (factorial (n - k))

theorem solve_comb_eq (x : ℕ) :
  comb (x + 5) x = comb (x + 3) (x - 1) + comb (x + 3) (x - 2) + 3/4 * perm (x + 3) 3 ->
  x = 14 := 
by 
  sorry

end NUMINAMATH_GPT_solve_comb_eq_l1698_169874


namespace NUMINAMATH_GPT_number_of_representatives_from_companyA_l1698_169845

-- Define conditions
def companyA_representatives : ℕ := 120
def companyB_representatives : ℕ := 100
def total_selected : ℕ := 11

-- Define the theorem
theorem number_of_representatives_from_companyA : 120 * (11 / (120 + 100)) = 6 := by
  sorry

end NUMINAMATH_GPT_number_of_representatives_from_companyA_l1698_169845


namespace NUMINAMATH_GPT_intersection_is_correct_l1698_169830

def A : Set ℝ := {x | True}
def B : Set ℝ := {y | y ≥ 0}

theorem intersection_is_correct : A ∩ B = { x | x ≥ 0 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_is_correct_l1698_169830


namespace NUMINAMATH_GPT_builder_installed_windows_l1698_169803

-- Conditions
def total_windows : ℕ := 14
def hours_per_window : ℕ := 8
def remaining_hours : ℕ := 48

-- Definition for the problem statement
def installed_windows := total_windows - remaining_hours / hours_per_window

-- The hypothesis we need to prove
theorem builder_installed_windows : installed_windows = 8 := by
  sorry

end NUMINAMATH_GPT_builder_installed_windows_l1698_169803


namespace NUMINAMATH_GPT_M_eq_N_l1698_169876

def M : Set ℝ := {x | ∃ (k : ℤ), x = (2 * k + 1) * Real.pi}
def N : Set ℝ := {x | ∃ (k : ℤ), x = (2 * k - 1) * Real.pi}

theorem M_eq_N : M = N := by
  sorry

end NUMINAMATH_GPT_M_eq_N_l1698_169876


namespace NUMINAMATH_GPT_tax_rate_computation_l1698_169888

-- Define the inputs
def total_value : ℝ := 1720
def non_taxable_amount : ℝ := 600
def tax_paid : ℝ := 134.4

-- Define the derived taxable amount
def taxable_amount : ℝ := total_value - non_taxable_amount

-- Define the expected tax rate
def expected_tax_rate : ℝ := 0.12

-- State the theorem
theorem tax_rate_computation : 
  (tax_paid / taxable_amount * 100) = expected_tax_rate * 100 := 
by
  sorry

end NUMINAMATH_GPT_tax_rate_computation_l1698_169888


namespace NUMINAMATH_GPT_find_original_number_l1698_169846

theorem find_original_number (x : ℝ)
  (h : (((x + 3) * 3 - 3) / 3) = 10) : x = 8 :=
sorry

end NUMINAMATH_GPT_find_original_number_l1698_169846


namespace NUMINAMATH_GPT_work_days_difference_l1698_169853

theorem work_days_difference (d_a d_b : ℕ) (H1 : d_b = 15) (H2 : d_a = d_b / 3) : 15 - d_a = 10 := by
  sorry

end NUMINAMATH_GPT_work_days_difference_l1698_169853


namespace NUMINAMATH_GPT_tutors_all_work_together_after_360_days_l1698_169867

theorem tutors_all_work_together_after_360_days :
  ∀ (n : ℕ), (n > 0) → 
    (∃ k, k > 0 ∧ k = Nat.lcm 5 (Nat.lcm 8 (Nat.lcm 9 10)) ∧ 
     k % 7 = 3) := by
  sorry

end NUMINAMATH_GPT_tutors_all_work_together_after_360_days_l1698_169867


namespace NUMINAMATH_GPT_overall_cost_for_all_projects_l1698_169863

-- Define the daily salaries including 10% taxes and insurance.
def daily_salary_entry_level_worker : ℕ := 100 + 10
def daily_salary_experienced_worker : ℕ := 130 + 13
def daily_salary_electrician : ℕ := 2 * 100 + 20
def daily_salary_plumber : ℕ := 250 + 25
def daily_salary_architect : ℕ := (35/10) * 100 + 35

-- Define the total cost for each project.
def project1_cost : ℕ :=
  daily_salary_entry_level_worker +
  daily_salary_experienced_worker +
  daily_salary_electrician +
  daily_salary_plumber +
  daily_salary_architect

def project2_cost : ℕ :=
  2 * daily_salary_experienced_worker +
  daily_salary_electrician +
  daily_salary_plumber +
  daily_salary_architect

def project3_cost : ℕ :=
  2 * daily_salary_entry_level_worker +
  daily_salary_electrician +
  daily_salary_plumber +
  daily_salary_architect

-- Define the overall cost for all three projects.
def total_cost : ℕ :=
  project1_cost + project2_cost + project3_cost

theorem overall_cost_for_all_projects :
  total_cost = 3399 :=
by
  sorry

end NUMINAMATH_GPT_overall_cost_for_all_projects_l1698_169863


namespace NUMINAMATH_GPT_ticket_savings_percentage_l1698_169809

theorem ticket_savings_percentage:
  ∀ (P : ℝ), 9 * P - 6 * P = (1 / 3) * (9 * P) ∧ (33 + 1/3) = 100 * (3 * P / (9 * P)) := 
by
  intros P
  sorry

end NUMINAMATH_GPT_ticket_savings_percentage_l1698_169809


namespace NUMINAMATH_GPT_z_is_real_iff_m_values_z_in_third_quadrant_iff_m_interval_l1698_169831

section
variable (m : ℝ)
def z : ℂ := (m^2 + 5 * m + 6) + (m^2 - 2 * m - 15) * Complex.I

theorem z_is_real_iff_m_values :
  (z m).im = 0 ↔ m = -3 ∨ m = 5 :=
by sorry

theorem z_in_third_quadrant_iff_m_interval :
  (z m).re < 0 ∧ (z m).im < 0 ↔ m ∈ Set.Ioo (-3) (-2) :=
by sorry
end

end NUMINAMATH_GPT_z_is_real_iff_m_values_z_in_third_quadrant_iff_m_interval_l1698_169831


namespace NUMINAMATH_GPT_vectors_are_coplanar_l1698_169816

-- Definitions of the vectors a, b, and c.
def a (x : ℝ) : ℝ × ℝ × ℝ := (1, x, -2)
def b : ℝ × ℝ × ℝ := (0, 1, 2)
def c : ℝ × ℝ × ℝ := (1, 0, 0)

-- The proof statement 
theorem vectors_are_coplanar (x : ℝ) 
  (h : ∃ m n : ℝ, a x = (n, m, 2 * m)) : 
  x = -1 :=
sorry

end NUMINAMATH_GPT_vectors_are_coplanar_l1698_169816


namespace NUMINAMATH_GPT_modulo_17_residue_l1698_169872

theorem modulo_17_residue : (3^4 + 6 * 49 + 8 * 137 + 7 * 34) % 17 = 5 := 
by
  sorry

end NUMINAMATH_GPT_modulo_17_residue_l1698_169872


namespace NUMINAMATH_GPT_length_of_BD_l1698_169822

theorem length_of_BD (AB AC CB BD : ℝ) (h1 : AB = 10) (h2 : AC = 4 * CB) (h3 : AC = 4 * 2) (h4 : CB = 2) :
  BD = 3 :=
sorry

end NUMINAMATH_GPT_length_of_BD_l1698_169822


namespace NUMINAMATH_GPT_first_term_geometric_series_l1698_169882

variable (a : ℝ)
variable (r : ℝ := 1/4)
variable (S : ℝ := 80)

theorem first_term_geometric_series 
  (h1 : r = 1/4) 
  (h2 : S = 80)
  : a = 60 :=
by 
  sorry

end NUMINAMATH_GPT_first_term_geometric_series_l1698_169882


namespace NUMINAMATH_GPT_max_street_lamps_proof_l1698_169800

noncomputable def max_street_lamps_on_road : ℕ := 1998

theorem max_street_lamps_proof (L : ℕ) (l : ℕ)
    (illuminates : ∀ i, i ≤ max_street_lamps_on_road → 
                  (∃ unique_segment : ℕ, unique_segment ≤ L ∧ unique_segment > L - l )):
  max_street_lamps_on_road = 1998 := by
  sorry

end NUMINAMATH_GPT_max_street_lamps_proof_l1698_169800


namespace NUMINAMATH_GPT_incorrect_statement_D_l1698_169884

noncomputable def f : ℝ → ℝ := sorry

axiom A1 : ∃ x : ℝ, f x ≠ 0
axiom A2 : ∀ x : ℝ, f (x + 1) = -f (2 - x)
axiom A3 : ∀ x : ℝ, f (x + 3) = f (x - 3)

theorem incorrect_statement_D :
  ¬ (∀ x : ℝ, f (3 + x) + f (3 - x) = 0) :=
sorry

end NUMINAMATH_GPT_incorrect_statement_D_l1698_169884


namespace NUMINAMATH_GPT_max_value_bx_plus_a_l1698_169870

variable (a b : ℝ)

theorem max_value_bx_plus_a (h : ∀ x, 0 ≤ x ∧ x ≤ 1 → |a * x + b| ≤ 1) :
  ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ |b * x + a| = 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_max_value_bx_plus_a_l1698_169870


namespace NUMINAMATH_GPT_square_perimeter_l1698_169810

theorem square_perimeter (s : ℝ) (h : s^2 = 625) : 4 * s = 100 := 
by {
  sorry
}

end NUMINAMATH_GPT_square_perimeter_l1698_169810


namespace NUMINAMATH_GPT_find_number_l1698_169865

def initial_condition (x : ℝ) : Prop :=
  ((x + 7) * 3 - 12) / 6 = -8

theorem find_number (x : ℝ) (h : initial_condition x) : x = -19 := by
  sorry

end NUMINAMATH_GPT_find_number_l1698_169865


namespace NUMINAMATH_GPT_packs_needed_l1698_169898

def pouches_per_pack : ℕ := 6
def team_members : ℕ := 13
def coaches : ℕ := 3
def helpers : ℕ := 2
def total_people : ℕ := team_members + coaches + helpers

theorem packs_needed (people : ℕ) (pouches_per_pack : ℕ) : ℕ :=
  (people + pouches_per_pack - 1) / pouches_per_pack

example : packs_needed total_people pouches_per_pack = 3 :=
by
  have h1 : total_people = 18 := rfl
  have h2 : pouches_per_pack = 6 := rfl
  rw [h1, h2]
  norm_num
  sorry

end NUMINAMATH_GPT_packs_needed_l1698_169898


namespace NUMINAMATH_GPT_meaningful_expression_l1698_169849

theorem meaningful_expression (x : ℝ) : (1 / Real.sqrt (x + 2) > 0) → (x > -2) := 
sorry

end NUMINAMATH_GPT_meaningful_expression_l1698_169849


namespace NUMINAMATH_GPT_solve_inequality_l1698_169839

theorem solve_inequality (x : ℝ) : 5 * x - 12 ≤ 2 * (4 * x - 3) → x ≥ -2 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_solve_inequality_l1698_169839


namespace NUMINAMATH_GPT_cost_price_proof_l1698_169805

noncomputable def selling_price : Real := 12000
noncomputable def discount_rate : Real := 0.10
noncomputable def new_selling_price : Real := selling_price * (1 - discount_rate)
noncomputable def profit_rate : Real := 0.08

noncomputable def cost_price : Real := new_selling_price / (1 + profit_rate)

theorem cost_price_proof : cost_price = 10000 := by sorry

end NUMINAMATH_GPT_cost_price_proof_l1698_169805


namespace NUMINAMATH_GPT_quadratic_function_solution_l1698_169864

theorem quadratic_function_solution :
  (∃ g : ℝ → ℝ, ∀ x : ℝ, g (x + 1) - g x = 2 * x + 3 ∧ g 2 - g 6 = -40) :=
sorry

end NUMINAMATH_GPT_quadratic_function_solution_l1698_169864


namespace NUMINAMATH_GPT_solve_linear_system_l1698_169890

theorem solve_linear_system (x y a : ℝ) (h1 : 4 * x + 3 * y = 1) (h2 : a * x + (a - 1) * y = 3) (hxy : x = y) : a = 11 :=
by
  sorry

end NUMINAMATH_GPT_solve_linear_system_l1698_169890


namespace NUMINAMATH_GPT_ways_to_reach_5_5_l1698_169856

def moves_to_destination : ℕ → ℕ → ℕ
| 0, 0     => 1
| 0, j+1   => moves_to_destination 0 j
| i+1, 0   => moves_to_destination i 0
| i+1, j+1 => moves_to_destination i (j+1) + moves_to_destination (i+1) j + moves_to_destination i j

theorem ways_to_reach_5_5 : moves_to_destination 5 5 = 1573 := by
  sorry

end NUMINAMATH_GPT_ways_to_reach_5_5_l1698_169856


namespace NUMINAMATH_GPT_min_value_of_x_plus_y_l1698_169895

theorem min_value_of_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y = x * y) :
  x + y ≥ 18 :=
sorry

end NUMINAMATH_GPT_min_value_of_x_plus_y_l1698_169895


namespace NUMINAMATH_GPT_sum_common_ratios_l1698_169828

variable (k p r : ℝ)
variable (hp : p ≠ r)

theorem sum_common_ratios (h : k * p ^ 2 - k * r ^ 2 = 2 * (k * p - k * r)) : 
  p + r = 2 := by
  have hk : k ≠ 0 := sorry -- From the nonconstancy condition
  sorry

end NUMINAMATH_GPT_sum_common_ratios_l1698_169828


namespace NUMINAMATH_GPT_six_letter_words_count_l1698_169858

def first_letter_possibilities := 26
def second_letter_possibilities := 26
def third_letter_possibilities := 26
def fourth_letter_possibilities := 26

def number_of_six_letter_words : Nat := 
  first_letter_possibilities * 
  second_letter_possibilities * 
  third_letter_possibilities * 
  fourth_letter_possibilities

theorem six_letter_words_count : number_of_six_letter_words = 456976 := by
  sorry

end NUMINAMATH_GPT_six_letter_words_count_l1698_169858


namespace NUMINAMATH_GPT_solve_diff_eq_for_k_ne_zero_solve_diff_eq_for_k_eq_zero_l1698_169836

open Real

theorem solve_diff_eq_for_k_ne_zero (k : ℝ) (h : k ≠ 0) (f g : ℝ → ℝ) 
  (hf : ∀ x, deriv f x = g x * (f x + g x) ^ k)
  (hg : ∀ x, deriv g x = f x * (f x + g x) ^ k)
  (hf0 : f 0 = 1)
  (hg0 : g 0 = 0) :
  (∀ x, f x = (1 / 2) * ((1 / (1 - k * x)) ^ (1 / k) + (1 - k * x) ^ (1 / k)) ∧ g x = (1 / 2) * ((1 / (1 - k * x)) ^ (1 / k) - (1 - k * x) ^ (1 / k))) :=
sorry

theorem solve_diff_eq_for_k_eq_zero (f g : ℝ → ℝ) 
  (hf : ∀ x, deriv f x = g x)
  (hg : ∀ x, deriv g x = f x)
  (hf0 : f 0 = 1)
  (hg0 : g 0 = 0) :
  (∀ x, f x = cosh x ∧ g x = sinh x) :=
sorry

end NUMINAMATH_GPT_solve_diff_eq_for_k_ne_zero_solve_diff_eq_for_k_eq_zero_l1698_169836


namespace NUMINAMATH_GPT_multiplication_mistake_l1698_169894

theorem multiplication_mistake (x : ℕ) (H : 43 * x - 34 * x = 1215) : x = 135 :=
sorry

end NUMINAMATH_GPT_multiplication_mistake_l1698_169894


namespace NUMINAMATH_GPT_linear_function_incorrect_conclusion_C_l1698_169815

theorem linear_function_incorrect_conclusion_C :
  ∀ (x y : ℝ), (y = -2 * x + 4) → ¬(∃ x, y = 0 ∧ (x = 0 ∧ y = 4)) := by
  sorry

end NUMINAMATH_GPT_linear_function_incorrect_conclusion_C_l1698_169815


namespace NUMINAMATH_GPT_golden_section_search_third_point_l1698_169877

noncomputable def golden_ratio : ℝ := 0.618

theorem golden_section_search_third_point :
  let L₀ := 1000
  let U₀ := 2000
  let d₀ := U₀ - L₀
  let x₁ := U₀ - golden_ratio * d₀
  let x₂ := L₀ + golden_ratio * d₀
  let d₁ := U₀ - x₁
  let x₃ := x₁ + golden_ratio * d₁
  x₃ = 1764 :=
by
  sorry

end NUMINAMATH_GPT_golden_section_search_third_point_l1698_169877


namespace NUMINAMATH_GPT_intersection_complement_l1698_169878

open Set Real

noncomputable def U : Set ℝ := univ
noncomputable def A : Set ℝ := { y | y ≥ 0 }
noncomputable def B : Set ℝ := { y | y ≥ 1 }

theorem intersection_complement :
  A ∩ (U \ B) = Ico 0 1 :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_l1698_169878


namespace NUMINAMATH_GPT_t_plus_inv_t_eq_three_l1698_169841

theorem t_plus_inv_t_eq_three {t : ℝ} (h : t^2 - 3 * t + 1 = 0) (hnz : t ≠ 0) : t + 1 / t = 3 :=
sorry

end NUMINAMATH_GPT_t_plus_inv_t_eq_three_l1698_169841


namespace NUMINAMATH_GPT_goldfish_graph_discrete_points_l1698_169814

theorem goldfish_graph_discrete_points : 
  ∀ n : ℤ, 1 ≤ n ∧ n ≤ 10 → ∃ C : ℤ, C = 20 * n + 10 ∧ ∀ m : ℤ, (1 ≤ m ∧ m ≤ 10 ∧ m ≠ n) → C ≠ (20 * m + 10) :=
by
  sorry

end NUMINAMATH_GPT_goldfish_graph_discrete_points_l1698_169814


namespace NUMINAMATH_GPT_alice_sales_surplus_l1698_169835

-- Define the constants
def adidas_cost : ℕ := 45
def nike_cost : ℕ := 60
def reebok_cost : ℕ := 35
def quota : ℕ := 1000

-- Define the quantities sold
def adidas_sold : ℕ := 6
def nike_sold : ℕ := 8
def reebok_sold : ℕ := 9

-- Calculate total sales
def total_sales : ℕ := adidas_sold * adidas_cost + nike_sold * nike_cost + reebok_sold * reebok_cost

-- Prove that Alice's total sales minus her quota is 65
theorem alice_sales_surplus : total_sales - quota = 65 := by
  -- Calculation is omitted here. Here is the mathematical fact to prove:
  sorry

end NUMINAMATH_GPT_alice_sales_surplus_l1698_169835


namespace NUMINAMATH_GPT_find_a9_l1698_169838

variable {a_n : ℕ → ℝ}

-- Definition of arithmetic progression
def is_arithmetic_progression (a : ℕ → ℝ) (a1 d : ℝ) := ∀ n : ℕ, a n = a1 + (n - 1) * d

-- Conditions
variables (a1 d : ℝ)
variable (h1 : a1 + (a1 + d)^2 = -3)
variable (h2 : ((a1 + a1 + 4 * d) * 5 / 2) = 10)

-- Question, needing the final statement
theorem find_a9 (a : ℕ → ℝ) (ha : is_arithmetic_progression a a1 d) : a 9 = 20 :=
by
    -- Since the theorem requires solving the statements, we use sorry to skip the proof.
    sorry

end NUMINAMATH_GPT_find_a9_l1698_169838


namespace NUMINAMATH_GPT_cistern_fill_time_l1698_169879

theorem cistern_fill_time
  (T : ℝ)
  (H1 : 0 < T)
  (rate_first_tap : ℝ := 1 / T)
  (rate_second_tap : ℝ := 1 / 6)
  (net_rate : ℝ := 1 / 12)
  (H2 : rate_first_tap - rate_second_tap = net_rate) :
  T = 4 :=
sorry

end NUMINAMATH_GPT_cistern_fill_time_l1698_169879


namespace NUMINAMATH_GPT_salad_cucumbers_l1698_169818

theorem salad_cucumbers (c t : ℕ) 
  (h1 : c + t = 280)
  (h2 : t = 3 * c) : c = 70 :=
sorry

end NUMINAMATH_GPT_salad_cucumbers_l1698_169818


namespace NUMINAMATH_GPT_jerry_age_l1698_169829

theorem jerry_age (M J : ℕ) (h1 : M = 4 * J - 8) (h2 : M = 24) : J = 8 :=
by
  sorry

end NUMINAMATH_GPT_jerry_age_l1698_169829


namespace NUMINAMATH_GPT_range_of_a_l1698_169886

noncomputable def proposition_p (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - a*x + 1 = 0

noncomputable def proposition_q (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2*x + a > 0

theorem range_of_a (a : ℝ) (hp : proposition_p a) (hq : proposition_q a) : a ≥ 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1698_169886


namespace NUMINAMATH_GPT_fraction_evaporated_l1698_169897

theorem fraction_evaporated (x : ℝ) (h : (1 - x) * (1/4) = 1/6) : x = 1/3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_evaporated_l1698_169897


namespace NUMINAMATH_GPT_polynomial_subtraction_simplify_l1698_169892

open Polynomial

noncomputable def p : Polynomial ℚ := 3 * X^2 + 9 * X - 5
noncomputable def q : Polynomial ℚ := 2 * X^2 + 3 * X - 10
noncomputable def result : Polynomial ℚ := X^2 + 6 * X + 5

theorem polynomial_subtraction_simplify : 
  p - q = result :=
by
  sorry

end NUMINAMATH_GPT_polynomial_subtraction_simplify_l1698_169892


namespace NUMINAMATH_GPT_take_home_pay_correct_l1698_169847

def jonessa_pay : ℝ := 500
def tax_deduction_percent : ℝ := 0.10
def insurance_deduction_percent : ℝ := 0.05
def pension_plan_deduction_percent : ℝ := 0.03
def union_dues_deduction_percent : ℝ := 0.02

def total_deductions : ℝ :=
  jonessa_pay * tax_deduction_percent +
  jonessa_pay * insurance_deduction_percent +
  jonessa_pay * pension_plan_deduction_percent +
  jonessa_pay * union_dues_deduction_percent

def take_home_pay : ℝ := jonessa_pay - total_deductions

theorem take_home_pay_correct : take_home_pay = 400 :=
  by
  sorry

end NUMINAMATH_GPT_take_home_pay_correct_l1698_169847


namespace NUMINAMATH_GPT_smallest_n_for_multiple_of_11_l1698_169871

theorem smallest_n_for_multiple_of_11 
  (x y : ℤ) 
  (hx : x ≡ -2 [ZMOD 11]) 
  (hy : y ≡ 2 [ZMOD 11]) : 
  ∃ n : ℕ, n > 0 ∧ (x^2 + x * y + y^2 + n ≡ 0 [ZMOD 11]) ∧ n = 7 :=
sorry

end NUMINAMATH_GPT_smallest_n_for_multiple_of_11_l1698_169871


namespace NUMINAMATH_GPT_unique_four_digit_number_l1698_169860

theorem unique_four_digit_number (a b c d : ℕ) (ha : 1 ≤ a) (hb : b ≤ 9) (hc : c ≤ 9) (hd : d ≤ 9)
  (h1 : a + b = c + d)
  (h2 : b + d = 2 * (a + c))
  (h3 : a + d = c)
  (h4 : b + c - a = 3 * d) :
  a = 1 ∧ b = 8 ∧ c = 5 ∧ d = 4 :=
by
  sorry

end NUMINAMATH_GPT_unique_four_digit_number_l1698_169860


namespace NUMINAMATH_GPT_sides_of_triangle_l1698_169808

-- Definitions from conditions
variables (a b c : ℕ) (r bk kc : ℕ)
def is_tangent_split : Prop := bk = 8 ∧ kc = 6
def inradius : Prop := r = 4

-- Main theorem statement
theorem sides_of_triangle (h1 : is_tangent_split bk kc) (h2 : inradius r) : a + 6 = 13 ∧ a + 8 = 15 ∧ b = 14 := by
  sorry

end NUMINAMATH_GPT_sides_of_triangle_l1698_169808


namespace NUMINAMATH_GPT_josette_additional_cost_l1698_169844

def small_bottle_cost_eur : ℝ := 1.50
def large_bottle_cost_eur : ℝ := 2.40
def exchange_rate : ℝ := 1.20
def discount_10_percent : ℝ := 0.10
def discount_15_percent : ℝ := 0.15

def initial_small_bottles : ℕ := 3
def initial_large_bottles : ℕ := 2

def initial_total_cost_eur : ℝ :=
  (small_bottle_cost_eur * initial_small_bottles) +
  (large_bottle_cost_eur * initial_large_bottles)

def discounted_cost_eur_10 : ℝ :=
  initial_total_cost_eur * (1 - discount_10_percent)

def additional_bottle_cost_eur : ℝ := small_bottle_cost_eur

def new_total_cost_eur : ℝ :=
  initial_total_cost_eur + additional_bottle_cost_eur

def discounted_cost_eur_15 : ℝ :=
  new_total_cost_eur * (1 - discount_15_percent)

def cost_usd (eur_amount : ℝ) : ℝ :=
  eur_amount * exchange_rate

def discounted_cost_usd_10 : ℝ := cost_usd discounted_cost_eur_10
def discounted_cost_usd_15 : ℝ := cost_usd discounted_cost_eur_15

def additional_cost_usd : ℝ :=
  discounted_cost_usd_15 - discounted_cost_usd_10

theorem josette_additional_cost :
  additional_cost_usd = 0.972 :=
by 
  sorry

end NUMINAMATH_GPT_josette_additional_cost_l1698_169844


namespace NUMINAMATH_GPT_geometric_sequence_div_sum_l1698_169820

noncomputable def a (n : ℕ) : ℝ := sorry

noncomputable def S (n : ℕ) : ℝ := sorry

theorem geometric_sequence_div_sum 
  (h₁ : S 3 = (1 - (2 : ℝ) ^ 3) / (1 - (2 : ℝ) ^ 2) * a 1)
  (h₂ : S 2 = (1 - (2 : ℝ) ^ 2) / (1 - 2) * a 1)
  (h₃ : 8 * a 2 = a 5) : 
  S 3 / S 2 = 7 / 3 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_div_sum_l1698_169820


namespace NUMINAMATH_GPT_total_rainfall_in_january_l1698_169885

theorem total_rainfall_in_january 
  (r1 r2 : ℝ)
  (h1 : r2 = 1.5 * r1)
  (h2 : r2 = 18) : 
  r1 + r2 = 30 := by
  sorry

end NUMINAMATH_GPT_total_rainfall_in_january_l1698_169885


namespace NUMINAMATH_GPT_number_of_ways_to_fill_grid_l1698_169843

noncomputable def totalWaysToFillGrid (S : Finset ℕ) : ℕ :=
  S.card.choose 5

theorem number_of_ways_to_fill_grid : totalWaysToFillGrid ({1, 2, 3, 4, 5, 6} : Finset ℕ) = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_fill_grid_l1698_169843


namespace NUMINAMATH_GPT_quotient_of_fifths_l1698_169825

theorem quotient_of_fifths : (2 / 5) / (1 / 5) = 2 := 
  by 
    sorry

end NUMINAMATH_GPT_quotient_of_fifths_l1698_169825


namespace NUMINAMATH_GPT_roger_initial_money_l1698_169880

theorem roger_initial_money (x : ℤ) 
    (h1 : x + 28 - 25 = 19) : 
    x = 16 := 
by 
    sorry

end NUMINAMATH_GPT_roger_initial_money_l1698_169880
