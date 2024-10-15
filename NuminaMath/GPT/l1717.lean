import Mathlib

namespace NUMINAMATH_GPT_fewer_buses_than_cars_l1717_171750

theorem fewer_buses_than_cars
  (bus_to_car_ratio : ℕ := 1)
  (cars_on_river_road : ℕ := 65)
  (cars_per_bus : ℕ := 13) :
  cars_on_river_road - (cars_on_river_road / cars_per_bus) = 60 :=
by
  sorry

end NUMINAMATH_GPT_fewer_buses_than_cars_l1717_171750


namespace NUMINAMATH_GPT_roots_of_quadratic_l1717_171758

theorem roots_of_quadratic (x : ℝ) : (x * (x - 2) = 2 - x) ↔ (x = 2 ∨ x = -1) :=
by
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_l1717_171758


namespace NUMINAMATH_GPT_neg_p_l1717_171740

variable (x : ℝ)

def p : Prop := ∃ x_0 : ℝ, x_0^2 + x_0 + 2 ≤ 0

theorem neg_p : ¬p ↔ ∀ x : ℝ, x^2 + x + 2 > 0 := by
  sorry

end NUMINAMATH_GPT_neg_p_l1717_171740


namespace NUMINAMATH_GPT_abs_ineq_cond_l1717_171777

theorem abs_ineq_cond (a : ℝ) : 
  (-3 < a ∧ a < 1) ↔ (∃ x : ℝ, |x - a| + |x + 1| < 2) := sorry

end NUMINAMATH_GPT_abs_ineq_cond_l1717_171777


namespace NUMINAMATH_GPT_no_integer_solution_for_equation_l1717_171739

theorem no_integer_solution_for_equation :
  ¬ ∃ (x y : ℤ), x^2 + 3 * x * y - 2 * y^2 = 122 :=
sorry

end NUMINAMATH_GPT_no_integer_solution_for_equation_l1717_171739


namespace NUMINAMATH_GPT_positive_integers_satisfying_inequality_l1717_171727

theorem positive_integers_satisfying_inequality (x : ℕ) (hx : x > 0) : 4 - x > 1 ↔ x = 1 ∨ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_positive_integers_satisfying_inequality_l1717_171727


namespace NUMINAMATH_GPT_find_f_2017_l1717_171736

theorem find_f_2017 {f : ℤ → ℤ}
  (symmetry : ∀ x : ℤ, f (-x) = -f x)
  (periodicity : ∀ x : ℤ, f (x + 4) = f x)
  (f_neg_1 : f (-1) = 2) :
  f 2017 = -2 :=
sorry

end NUMINAMATH_GPT_find_f_2017_l1717_171736


namespace NUMINAMATH_GPT_calculate_x_n_minus_inverse_x_n_l1717_171722

theorem calculate_x_n_minus_inverse_x_n
  (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π) (x : ℝ) (h : x - 1/x = 2 * Real.sin θ) (n : ℕ) (hn : 0 < n) :
  x^n - 1/x^n = 2 * Real.sinh (n * θ) :=
by sorry

end NUMINAMATH_GPT_calculate_x_n_minus_inverse_x_n_l1717_171722


namespace NUMINAMATH_GPT_symmetric_circle_equation_l1717_171752

theorem symmetric_circle_equation (x y : ℝ) :
  (x^2 + y^2 - 4 * x = 0) ↔ (-x ^ 2 + y^2 + 4 * x = 0) :=
sorry

end NUMINAMATH_GPT_symmetric_circle_equation_l1717_171752


namespace NUMINAMATH_GPT_determine_base_solution_l1717_171788

theorem determine_base_solution :
  ∃ (h : ℕ), 
  h > 8 ∧ 
  (8 * h^3 + 6 * h^2 + 7 * h + 4) + (4 * h^3 + 3 * h^2 + 2 * h + 9) = 1 * h^4 + 3 * h^3 + 0 * h^2 + 0 * h + 3 ∧
  (9 + 4) = 13 ∧
  1 * h + 3 = 13 ∧
  (7 + 2 + 1) = 10 ∧
  1 * h + 0 = 10 ∧
  (6 + 3 + 1) = 10 ∧
  1 * h + 0 = 10 ∧
  (8 + 4 + 1) = 13 ∧
  1 * h + 3 = 13 ∧
  h = 10 :=
by
  sorry

end NUMINAMATH_GPT_determine_base_solution_l1717_171788


namespace NUMINAMATH_GPT_domain_of_f_l1717_171709

noncomputable def f (x : ℝ) : ℝ := (3 * x^2) / Real.sqrt (1 - 2 * x) + Real.log (1 + 2 * x)

theorem domain_of_f : {x : ℝ | 1 - 2 * x > 0 ∧ 1 + 2 * x > 0} = {x : ℝ | -1 / 2 < x ∧ x < 1 / 2} :=
by
    sorry

end NUMINAMATH_GPT_domain_of_f_l1717_171709


namespace NUMINAMATH_GPT_solve_cubic_inequality_l1717_171705

theorem solve_cubic_inequality :
  { x : ℝ | x^3 + x^2 - 7 * x + 6 < 0 } = { x : ℝ | -3 < x ∧ x < 1 ∨ 1 < x ∧ x < 2 } :=
by
  sorry

end NUMINAMATH_GPT_solve_cubic_inequality_l1717_171705


namespace NUMINAMATH_GPT_students_making_stars_l1717_171795

theorem students_making_stars (total_stars stars_per_student : ℕ) (h1 : total_stars = 372) (h2 : stars_per_student = 3) : 
  total_stars / stars_per_student = 124 :=
by
  sorry

end NUMINAMATH_GPT_students_making_stars_l1717_171795


namespace NUMINAMATH_GPT_cylinder_radius_in_cone_l1717_171710

theorem cylinder_radius_in_cone (d h r : ℝ) (h_d : d = 20) (h_h : h = 24) (h_cylinder : 2 * r = r):
  r = 60 / 11 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_radius_in_cone_l1717_171710


namespace NUMINAMATH_GPT_travis_discount_percentage_l1717_171775

theorem travis_discount_percentage (P D : ℕ) (hP : P = 2000) (hD : D = 1400) :
  ((P - D) / P * 100) = 30 := by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_travis_discount_percentage_l1717_171775


namespace NUMINAMATH_GPT_rectangle_area_l1717_171716

theorem rectangle_area (p : ℝ) (l : ℝ) (h1 : 2 * (l + 2 * l) = p) :
  l * 2 * l = p^2 / 18 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1717_171716


namespace NUMINAMATH_GPT_success_permutations_correct_l1717_171787

theorem success_permutations_correct :
  let word := "SUCCESS"
  let n := 7
  let s_count := 3
  let c_count := 2
  let u_count := 1
  let e_count := 1
  let total_permutations := (Nat.factorial n) / ((Nat.factorial s_count) * (Nat.factorial c_count) * (Nat.factorial u_count) * (Nat.factorial e_count))
  total_permutations = 420 :=
by
  sorry

end NUMINAMATH_GPT_success_permutations_correct_l1717_171787


namespace NUMINAMATH_GPT_sum_possible_rs_l1717_171779

theorem sum_possible_rs (r s : ℤ) (h1 : r ≠ s) (h2 : r + s = 24) : 
  ∃ sum : ℤ, sum = 1232 := 
sorry

end NUMINAMATH_GPT_sum_possible_rs_l1717_171779


namespace NUMINAMATH_GPT_tickets_problem_l1717_171720

theorem tickets_problem (A C : ℝ) 
  (h1 : A + C = 200) 
  (h2 : 3 * A + 1.5 * C = 510) : C = 60 :=
by
  sorry

end NUMINAMATH_GPT_tickets_problem_l1717_171720


namespace NUMINAMATH_GPT_area_of_side_face_l1717_171702

theorem area_of_side_face (l w h : ℝ)
  (h_front_top : w * h = 0.5 * (l * h))
  (h_top_side : l * h = 1.5 * (w * h))
  (h_volume : l * w * h = 3000) :
  w * h = 200 := 
sorry

end NUMINAMATH_GPT_area_of_side_face_l1717_171702


namespace NUMINAMATH_GPT_point_translation_l1717_171749

variable (P Q : (ℝ × ℝ))
variable (dx : ℝ) (dy : ℝ)

theorem point_translation (hP : P = (-1, 2)) (hdx : dx = 2) (hdy : dy = 3) :
  Q = (P.1 + dx, P.2 - dy) → Q = (1, -1) := by
  sorry

end NUMINAMATH_GPT_point_translation_l1717_171749


namespace NUMINAMATH_GPT_number_of_solutions_l1717_171796

-- Given conditions
def positiveIntSolution (x y : ℤ) : Prop := x > 0 ∧ y > 0 ∧ 4 * x + 7 * y = 2001

-- Theorem statement
theorem number_of_solutions : ∃ (count : ℕ), 
  count = 71 ∧ ∃ f : Fin count → ℤ × ℤ,
    (∀ i, positiveIntSolution (f i).1 (f i).2) :=
by
  sorry

end NUMINAMATH_GPT_number_of_solutions_l1717_171796


namespace NUMINAMATH_GPT_game_is_unfair_swap_to_make_fair_l1717_171783

-- Part 1: Prove the game is unfair
theorem game_is_unfair (y b r : ℕ) (hb : y = 5) (bb : b = 13) (rb : r = 22) :
  ¬((b : ℚ) / (y + b + r) = (y : ℚ) / (y + b + r)) :=
by
  -- The proof is omitted as per the instructions.
  sorry

-- Part 2: Prove that swapping 4 black balls with 4 yellow balls makes the game fair.
theorem swap_to_make_fair (y b r : ℕ) (hb : y = 5) (bb : b = 13) (rb : r = 22) (x: ℕ) :
  x = 4 →
  (b - x : ℚ) / (y + b + r) = (y + x : ℚ) / (y + b + r) :=
by
  -- The proof is omitted as per the instructions.
  sorry

end NUMINAMATH_GPT_game_is_unfair_swap_to_make_fair_l1717_171783


namespace NUMINAMATH_GPT_correct_div_value_l1717_171792

theorem correct_div_value (x : ℝ) (h : 25 * x = 812) : x / 4 = 8.12 :=
by sorry

end NUMINAMATH_GPT_correct_div_value_l1717_171792


namespace NUMINAMATH_GPT_largest_n_divisible_l1717_171733

theorem largest_n_divisible (n : ℕ) : (n^3 + 150) % (n + 15) = 0 → n ≤ 2385 := by
  sorry

end NUMINAMATH_GPT_largest_n_divisible_l1717_171733


namespace NUMINAMATH_GPT_total_figurines_l1717_171724

theorem total_figurines:
  let basswood_blocks := 25
  let butternut_blocks := 30
  let aspen_blocks := 35
  let oak_blocks := 40
  let cherry_blocks := 45
  let basswood_figs_per_block := 3
  let butternut_figs_per_block := 4
  let aspen_figs_per_block := 2 * basswood_figs_per_block
  let oak_figs_per_block := 5
  let cherry_figs_per_block := 7
  let basswood_total := basswood_blocks * basswood_figs_per_block
  let butternut_total := butternut_blocks * butternut_figs_per_block
  let aspen_total := aspen_blocks * aspen_figs_per_block
  let oak_total := oak_blocks * oak_figs_per_block
  let cherry_total := cherry_blocks * cherry_figs_per_block
  let total_figs := basswood_total + butternut_total + aspen_total + oak_total + cherry_total
  total_figs = 920 := by sorry

end NUMINAMATH_GPT_total_figurines_l1717_171724


namespace NUMINAMATH_GPT_largest_n_digit_number_divisible_by_89_l1717_171738

theorem largest_n_digit_number_divisible_by_89 (n : ℕ) (h1 : n % 2 = 1) (h2 : 3 ≤ n ∧ n ≤ 7) :
  ∃ x, x = 9999951 ∧ (x % 89 = 0 ∧ (10 ^ (n-1) ≤ x ∧ x < 10 ^ n)) :=
by
  sorry

end NUMINAMATH_GPT_largest_n_digit_number_divisible_by_89_l1717_171738


namespace NUMINAMATH_GPT_sides_increase_factor_l1717_171735

theorem sides_increase_factor (s k : ℝ) (h : s^2 * 25 = k^2 * s^2) : k = 5 :=
by
  sorry

end NUMINAMATH_GPT_sides_increase_factor_l1717_171735


namespace NUMINAMATH_GPT_functions_identified_l1717_171719

variable (n : ℕ) (hn : n > 1)
variable {f : ℕ → ℝ → ℝ}

-- Define the conditions f1, f2, ..., fn
axiom cond_1 (x y : ℝ) : f 1 x + f 1 y = f 2 x * f 2 y
axiom cond_2 (x y : ℝ) : f 2 (x^2) + f 2 (y^2) = f 3 x * f 3 y
axiom cond_3 (x y : ℝ) : f 3 (x^3) + f 3 (y^3) = f 4 x * f 4 y
-- ... Similarly define conditions up to cond_n
axiom cond_n (x y : ℝ) : f n (x^n) + f n (y^n) = f 1 x * f 1 y

theorem functions_identified (i : ℕ) (hi₁ : 1 ≤ i) (hi₂ : i ≤ n) (x : ℝ) :
  f i x = 0 ∨ f i x = 2 :=
sorry

end NUMINAMATH_GPT_functions_identified_l1717_171719


namespace NUMINAMATH_GPT_compute_expression_l1717_171728

variable (a b : ℝ)

theorem compute_expression : 
  (8 * a^3 * b) * (4 * a * b^2) * (1 / (2 * a * b)^3) = 4 * a := 
by sorry

end NUMINAMATH_GPT_compute_expression_l1717_171728


namespace NUMINAMATH_GPT_first_day_speed_l1717_171773

open Real

-- Define conditions
variables (v : ℝ) (t : ℝ)
axiom distance_home_school : 1.5 = v * (t - 7/60)
axiom second_day_condition : 1.5 = 6 * (t - 8/60)

theorem first_day_speed :
  v = 10 :=
by
  -- The proof will be provided here
  sorry

end NUMINAMATH_GPT_first_day_speed_l1717_171773


namespace NUMINAMATH_GPT_expression_to_diophantine_l1717_171753

theorem expression_to_diophantine (x : ℝ) (y : ℝ) (n : ℕ) :
  (∃ (A B : ℤ), (x - y) ^ (2 * n + 1) = (A * x - B * y) ∧ (1969 : ℤ) * A^2 - (1968 : ℤ) * B^2 = 1) :=
sorry

end NUMINAMATH_GPT_expression_to_diophantine_l1717_171753


namespace NUMINAMATH_GPT_find_dividend_l1717_171746

theorem find_dividend 
  (R : ℤ) 
  (Q : ℤ) 
  (D : ℤ) 
  (h1 : R = 8) 
  (h2 : D = 3 * Q) 
  (h3 : D = 3 * R + 3) : 
  (D * Q + R = 251) :=
by {
  -- The proof would follow, but for now, we'll use sorry.
  sorry
}

end NUMINAMATH_GPT_find_dividend_l1717_171746


namespace NUMINAMATH_GPT_new_person_weight_l1717_171785

theorem new_person_weight (W : ℝ) :
  (∃ (W : ℝ), (390 - W + 70) / 4 = (390 - W) / 4 + 3 ∧ (390 - W + W) = 390) → 
  W = 58 :=
by
  sorry

end NUMINAMATH_GPT_new_person_weight_l1717_171785


namespace NUMINAMATH_GPT_largest_k_dividing_A_l1717_171767

def A : ℤ := 1990^(1991^1992) + 1991^(1990^1992) + 1992^(1991^1990)

theorem largest_k_dividing_A :
  1991^(1991) ∣ A := sorry

end NUMINAMATH_GPT_largest_k_dividing_A_l1717_171767


namespace NUMINAMATH_GPT_option_d_is_deductive_l1717_171741

theorem option_d_is_deductive :
  (∀ (r : ℝ), S_r = Real.pi * r^2) → (S_1 = Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_option_d_is_deductive_l1717_171741


namespace NUMINAMATH_GPT_last_three_digits_of_7_pow_210_l1717_171774

theorem last_three_digits_of_7_pow_210 : (7^210) % 1000 = 599 := by
  sorry

end NUMINAMATH_GPT_last_three_digits_of_7_pow_210_l1717_171774


namespace NUMINAMATH_GPT_add_mul_of_3_l1717_171729

theorem add_mul_of_3 (a b : ℤ) (ha : ∃ m : ℤ, a = 6 * m) (hb : ∃ n : ℤ, b = 9 * n) : ∃ k : ℤ, a + b = 3 * k :=
by
  sorry

end NUMINAMATH_GPT_add_mul_of_3_l1717_171729


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1717_171742

theorem arithmetic_sequence_sum :
  ∀(a_n : ℕ → ℕ) (S : ℕ → ℕ) (a_1 d : ℕ),
    (∀ n, a_n n = a_1 + (n - 1) * d) →
    (∀ n, S n = n * (a_1 + (n - 1) * d) / 2) →
    a_1 = 2 →
    S 4 = 20 →
    S 6 = 42 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1717_171742


namespace NUMINAMATH_GPT_perpendicular_lines_l1717_171762

def line_l1 (m x y : ℝ) : Prop := m * x - y + 1 = 0
def line_l2 (m x y : ℝ) : Prop := 2 * x - (m - 1) * y + 1 = 0

theorem perpendicular_lines (m : ℝ): (∃ x y : ℝ, line_l1 m x y) ∧ (∃ x y : ℝ, line_l2 m x y) ∧ (∀ x y : ℝ, line_l1 m x y → line_l2 m x y → m * (2 / (m - 1)) = -1) → m = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_l1717_171762


namespace NUMINAMATH_GPT_fifteen_horses_fifteen_bags_l1717_171721

-- Definitions based on the problem
def days_for_one_horse_one_bag : ℝ := 1  -- It takes 1 day for 1 horse to eat 1 bag of grain

-- Theorem statement
theorem fifteen_horses_fifteen_bags {d : ℝ} (h : d = days_for_one_horse_one_bag) :
  d = 1 :=
by
  sorry

end NUMINAMATH_GPT_fifteen_horses_fifteen_bags_l1717_171721


namespace NUMINAMATH_GPT_simplify_and_ratio_l1717_171768

theorem simplify_and_ratio (k : ℤ) : 
  let a := 1
  let b := 2
  (∀ (k : ℤ), (6 * k + 12) / 6 = a * k + b) →
  (a / b = 1 / 2) :=
by
  intros
  sorry
  
end NUMINAMATH_GPT_simplify_and_ratio_l1717_171768


namespace NUMINAMATH_GPT_total_turtles_rabbits_l1717_171723

-- Number of turtles and rabbits on Happy Island
def turtles_happy : ℕ := 120
def rabbits_happy : ℕ := 80

-- Number of turtles and rabbits on Lonely Island
def turtles_lonely : ℕ := turtles_happy / 3
def rabbits_lonely : ℕ := turtles_lonely

-- Number of turtles and rabbits on Serene Island
def rabbits_serene : ℕ := 2 * rabbits_lonely
def turtles_serene : ℕ := (3 * rabbits_lonely) / 4

-- Number of turtles and rabbits on Tranquil Island
def turtles_tranquil : ℕ := (turtles_happy - turtles_serene) + 5
def rabbits_tranquil : ℕ := turtles_tranquil

-- Proving the total numbers
theorem total_turtles_rabbits :
    turtles_happy = 120 ∧ rabbits_happy = 80 ∧
    turtles_lonely = 40 ∧ rabbits_lonely = 40 ∧
    turtles_serene = 30 ∧ rabbits_serene = 80 ∧
    turtles_tranquil = 95 ∧ rabbits_tranquil = 95 ∧
    (turtles_happy + turtles_lonely + turtles_serene + turtles_tranquil = 285) ∧
    (rabbits_happy + rabbits_lonely + rabbits_serene + rabbits_tranquil = 295) := 
    by
        -- Here we prove each part step by step using the definitions and conditions provided above
        sorry

end NUMINAMATH_GPT_total_turtles_rabbits_l1717_171723


namespace NUMINAMATH_GPT_polygon_diagonals_integer_l1717_171790

theorem polygon_diagonals_integer (n : ℤ) : ∃ k : ℤ, 2 * k = n * (n - 3) := by
sorry

end NUMINAMATH_GPT_polygon_diagonals_integer_l1717_171790


namespace NUMINAMATH_GPT_parabola_vertex_l1717_171759

theorem parabola_vertex :
  ∃ h k : ℝ, (∀ x y : ℝ, y^2 + 8*y + 4*x + 9 = 0 → x = -1/4 * (y + 4)^2 + 7/4)
  := 
  ⟨7/4, -4, sorry⟩

end NUMINAMATH_GPT_parabola_vertex_l1717_171759


namespace NUMINAMATH_GPT_inequality_maintained_l1717_171772

noncomputable def g (x a : ℝ) := x^2 + Real.log (x + a)

theorem inequality_maintained (x1 x2 a : ℝ) (hx1 : x1 = (-a + Real.sqrt (a^2 - 2))/2)
  (hx2 : x2 = (-a - Real.sqrt (a^2 - 2))/2):
  (a > Real.sqrt 2) → 
  (g x1 a + g x2 a) / 2 > g ((x1 + x2 ) / 2) a :=
by
  sorry

end NUMINAMATH_GPT_inequality_maintained_l1717_171772


namespace NUMINAMATH_GPT_empty_set_subset_zero_set_l1717_171704

-- Define the sets
def zero_set : Set ℕ := {0}
def empty_set : Set ℕ := ∅

-- State the problem
theorem empty_set_subset_zero_set : empty_set ⊂ zero_set :=
sorry

end NUMINAMATH_GPT_empty_set_subset_zero_set_l1717_171704


namespace NUMINAMATH_GPT_tangent_line_slope_at_one_l1717_171708

variable {f : ℝ → ℝ}

theorem tangent_line_slope_at_one (h : ∀ x, f x = e * x - e) : deriv f 1 = e :=
by sorry

end NUMINAMATH_GPT_tangent_line_slope_at_one_l1717_171708


namespace NUMINAMATH_GPT_volume_of_sphere_in_cone_l1717_171725

theorem volume_of_sphere_in_cone :
  let diameter_of_base := 16 * Real.sqrt 2
  let radius_of_base := diameter_of_base / 2
  let side_length := radius_of_base * 2 / Real.sqrt 2
  let inradius := side_length / 2
  let r := inradius
  let V := (4 / 3) * Real.pi * r^3
  V = (2048 / 3) * Real.pi := by
  sorry

end NUMINAMATH_GPT_volume_of_sphere_in_cone_l1717_171725


namespace NUMINAMATH_GPT_problem_solution_l1717_171711

def tens_digit_is_odd (n : ℕ) : Bool :=
  let m := (n * n + n) / 10 % 10
  m % 2 = 1

def count_tens_digit_odd : ℕ :=
  List.range 50 |>.filter tens_digit_is_odd |>.length

theorem problem_solution : count_tens_digit_odd = 25 :=
  sorry

end NUMINAMATH_GPT_problem_solution_l1717_171711


namespace NUMINAMATH_GPT_correct_propositions_l1717_171700

noncomputable def sequence_condition (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n + a (n+1) > 2 * a n

def prop1 (a : ℕ → ℝ) (h : sequence_condition a) : Prop :=
  a 2 > a 1 → ∀ n > 1, a n > a (n-1)

def prop4 (a : ℕ → ℝ) (h : sequence_condition a) : Prop :=
  ∃ d, ∀ n > 1, a n > a 1 + (n-1) * d

theorem correct_propositions {a : ℕ → ℝ}
  (h : sequence_condition a) :
  (prop1 a h) ∧ (prop4 a h) := 
sorry

end NUMINAMATH_GPT_correct_propositions_l1717_171700


namespace NUMINAMATH_GPT_common_ratio_is_0_88_second_term_is_475_2_l1717_171715

-- Define the first term and the sum of the infinite geometric series
def first_term : Real := 540
def sum_infinite_series : Real := 4500

-- Required properties of the common ratio
def common_ratio (r : Real) : Prop :=
  abs r < 1 ∧ sum_infinite_series = first_term / (1 - r)

-- Prove the common ratio is 0.88 given the conditions
theorem common_ratio_is_0_88 : ∃ r : Real, common_ratio r ∧ r = 0.88 :=
by 
  sorry

-- Calculate the second term of the series
def second_term (r : Real) : Real := first_term * r

-- Prove the second term is 475.2 given the common ratio is 0.88
theorem second_term_is_475_2 : second_term 0.88 = 475.2 :=
by 
  sorry

end NUMINAMATH_GPT_common_ratio_is_0_88_second_term_is_475_2_l1717_171715


namespace NUMINAMATH_GPT_find_m_from_parallel_l1717_171726

theorem find_m_from_parallel (m : ℝ) : 
  (∃ (A B : ℝ×ℝ), A = (-2, m) ∧ B = (m, 4) ∧
  (∃ (a b c : ℝ), a = 2 ∧ b = 1 ∧ c = -1 ∧
  (a * (B.1 - A.1) + b * (B.2 - A.2) = 0)) ) 
  → m = -8 :=
by
  sorry

end NUMINAMATH_GPT_find_m_from_parallel_l1717_171726


namespace NUMINAMATH_GPT_problem_solution_l1717_171793

theorem problem_solution (a b : ℝ) (h1 : 2 + 3 = -b) (h2 : 2 * 3 = -2 * a) : a + b = -8 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1717_171793


namespace NUMINAMATH_GPT_proof_2720000_scientific_l1717_171748

def scientific_notation (n : ℕ) : ℝ := 
  2.72 * 10^6 

theorem proof_2720000_scientific :
  scientific_notation 2720000 = 2.72 * 10^6 := by
  sorry

end NUMINAMATH_GPT_proof_2720000_scientific_l1717_171748


namespace NUMINAMATH_GPT_joe_time_to_friends_house_l1717_171786

theorem joe_time_to_friends_house
  (feet_moved : ℕ) (time_taken : ℕ) (remaining_distance : ℕ) (feet_in_yard : ℕ)
  (rate_of_movement : ℕ) (remaining_distance_feet : ℕ) (time_to_cover_remaining_distance : ℕ) :
  feet_moved = 80 →
  time_taken = 40 →
  remaining_distance = 90 →
  feet_in_yard = 3 →
  rate_of_movement = feet_moved / time_taken →
  remaining_distance_feet = remaining_distance * feet_in_yard →
  time_to_cover_remaining_distance = remaining_distance_feet / rate_of_movement →
  time_to_cover_remaining_distance = 135 :=
by
  sorry

end NUMINAMATH_GPT_joe_time_to_friends_house_l1717_171786


namespace NUMINAMATH_GPT_arnold_danny_age_l1717_171776

theorem arnold_danny_age (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 15) : x = 7 :=
sorry

end NUMINAMATH_GPT_arnold_danny_age_l1717_171776


namespace NUMINAMATH_GPT_mike_age_proof_l1717_171794

theorem mike_age_proof (a m : ℝ) (h1 : m = 3 * a - 20) (h2 : m + a = 70) : m = 47.5 := 
by {
  sorry
}

end NUMINAMATH_GPT_mike_age_proof_l1717_171794


namespace NUMINAMATH_GPT_average_of_data_is_six_l1717_171789

def data : List ℕ := [4, 6, 5, 8, 7, 6]

theorem average_of_data_is_six : 
  (data.sum / data.length : ℚ) = 6 := 
by sorry

end NUMINAMATH_GPT_average_of_data_is_six_l1717_171789


namespace NUMINAMATH_GPT_ratio_of_a_b_l1717_171764

theorem ratio_of_a_b (a b : ℝ) (h1 : 2 * a = 3 * b) (h2 : a ≠ 0 ∧ b ≠ 0) : a / b = 3 / 2 :=
by sorry

end NUMINAMATH_GPT_ratio_of_a_b_l1717_171764


namespace NUMINAMATH_GPT_merchant_markup_l1717_171765

theorem merchant_markup (x : ℝ) : 
  let CP := 100
  let MP := CP + (x / 100) * CP
  let SP_discount := MP - 0.1 * MP 
  let SP_profit := CP + 57.5
  SP_discount = SP_profit → x = 75 :=
by
  intros
  let CP := (100 : ℝ)
  let MP := CP + (x / 100) * CP
  let SP_discount := MP - 0.1 * MP 
  let SP_profit := CP + 57.5
  have h : SP_discount = SP_profit := sorry
  sorry

end NUMINAMATH_GPT_merchant_markup_l1717_171765


namespace NUMINAMATH_GPT_part_one_solution_set_part_two_range_of_m_l1717_171730

-- Part I
theorem part_one_solution_set (x : ℝ) : (|x + 1| + |x - 2| - 5 > 0) ↔ (x > 3 ∨ x < -2) :=
sorry

-- Part II
theorem part_two_range_of_m (m : ℝ) : (∀ x : ℝ, |x + 1| + |x - 2| - m ≥ 2) ↔ (m ≤ 1) :=
sorry

end NUMINAMATH_GPT_part_one_solution_set_part_two_range_of_m_l1717_171730


namespace NUMINAMATH_GPT_intersection_A_B_l1717_171712

def setA : Set ℝ := {x : ℝ | x > -1}
def setB : Set ℝ := {x : ℝ | x < 3}
def setIntersection : Set ℝ := {x : ℝ | x > -1 ∧ x < 3}

theorem intersection_A_B :
  setA ∩ setB = setIntersection :=
by sorry

end NUMINAMATH_GPT_intersection_A_B_l1717_171712


namespace NUMINAMATH_GPT_how_long_it_lasts_l1717_171703

-- Define a structure to hold the conditions
structure MoneySpending where
  mowing_income : ℕ
  weeding_income : ℕ
  weekly_expense : ℕ

-- Example conditions given in the problem
def lukesEarnings : MoneySpending :=
{ mowing_income := 9,
  weeding_income := 18,
  weekly_expense := 3 }

-- Main theorem proving the number of weeks he can sustain his spending
theorem how_long_it_lasts (data : MoneySpending) : 
  (data.mowing_income + data.weeding_income) / data.weekly_expense = 9 := by
  sorry

end NUMINAMATH_GPT_how_long_it_lasts_l1717_171703


namespace NUMINAMATH_GPT_exponentiation_problem_l1717_171771

theorem exponentiation_problem : 2^3 * 2^2 * 3^3 * 3^2 = 6^5 :=
by sorry

end NUMINAMATH_GPT_exponentiation_problem_l1717_171771


namespace NUMINAMATH_GPT_new_plants_description_l1717_171706

-- Condition: Anther culture of diploid corn treated with colchicine.
def diploid_corn := Type
def colchicine_treatment (plant : diploid_corn) : Prop := -- assume we have some method to define it
sorry

def anther_culture (plant : diploid_corn) (treated : colchicine_treatment plant) : Type := -- assume we have some method to define it
sorry

-- Describe the properties of new plants
def is_haploid (plant : diploid_corn) : Prop := sorry
def has_no_homologous_chromosomes (plant : diploid_corn) : Prop := sorry
def cannot_form_fertile_gametes (plant : diploid_corn) : Prop := sorry
def has_homologous_chromosomes_in_somatic_cells (plant : diploid_corn) : Prop := sorry
def can_form_fertile_gametes (plant : diploid_corn) : Prop := sorry
def is_homozygous_or_heterozygous (plant : diploid_corn) : Prop := sorry
def is_definitely_homozygous (plant : diploid_corn) : Prop := sorry
def is_diploid (plant : diploid_corn) : Prop := sorry

-- Equivalent math proof problem
theorem new_plants_description (plant : diploid_corn) (treated : colchicine_treatment plant) : 
  is_haploid (anther_culture plant treated) ∧ 
  has_homologous_chromosomes_in_somatic_cells (anther_culture plant treated) ∧ 
  can_form_fertile_gametes (anther_culture plant treated) ∧ 
  is_homozygous_or_heterozygous (anther_culture plant treated) := sorry

end NUMINAMATH_GPT_new_plants_description_l1717_171706


namespace NUMINAMATH_GPT_alchemerion_age_problem_l1717_171714

theorem alchemerion_age_problem
  (A S F : ℕ)  -- Declare the ages as natural numbers
  (h1 : A = 3 * S)  -- Condition 1: Alchemerion is 3 times his son's age
  (h2 : F = 2 * A + 40)  -- Condition 2: His father’s age is 40 years more than twice his age
  (h3 : A + S + F = 1240)  -- Condition 3: Together they are 1240 years old
  (h4 : A = 360)  -- Condition 4: Alchemerion is 360 years old
  : 40 = F - 2 * A :=  -- Conclusion: The number of years more than twice Alchemerion’s age is 40
by
  sorry  -- Proof can be filled in here

end NUMINAMATH_GPT_alchemerion_age_problem_l1717_171714


namespace NUMINAMATH_GPT_remainder_sum_59_l1717_171780

theorem remainder_sum_59 (x y z : ℕ) (h1 : x % 59 = 30) (h2 : y % 59 = 27) (h3 : z % 59 = 4) :
  (x + y + z) % 59 = 2 := 
sorry

end NUMINAMATH_GPT_remainder_sum_59_l1717_171780


namespace NUMINAMATH_GPT_largest_divisor_for_odd_n_l1717_171797

noncomputable def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem largest_divisor_for_odd_n (n : ℤ) (h : is_odd n ∧ n > 0) : 
  15 ∣ (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) := 
by 
  sorry

end NUMINAMATH_GPT_largest_divisor_for_odd_n_l1717_171797


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l1717_171778

noncomputable section

def is_hyperbola_point (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 16 = 1

def foci_distance_condition (P F1 F2 : ℝ × ℝ) : Prop :=
  |(P.1 - F1.1)^2 + (P.2 - F1.2)^2 - (P.1 - F2.1)^2 + (P.2 - F2.2)^2| = 6

theorem sufficient_not_necessary_condition 
  (x y F1_1 F1_2 F2_1 F2_2 : ℝ) (P : ℝ × ℝ)
  (P_hyp: is_hyperbola_point x y)
  (cond : foci_distance_condition P (F1_1, F1_2) (F2_1, F2_2)) :
  ∃ x y, is_hyperbola_point x y ∧ foci_distance_condition P (F1_1, F1_2) (F2_1, F2_2) :=
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l1717_171778


namespace NUMINAMATH_GPT_derivative_of_f_l1717_171791

-- Define the function
def f (x : ℝ) : ℝ := x^2 - x

-- State the theorem to prove
theorem derivative_of_f : ∀ x : ℝ,  (deriv f x = 2 * x - 1) :=
by sorry

end NUMINAMATH_GPT_derivative_of_f_l1717_171791


namespace NUMINAMATH_GPT_gcd_72_120_180_is_12_l1717_171760

theorem gcd_72_120_180_is_12 : Int.gcd (Int.gcd 72 120) 180 = 12 := by
  sorry

end NUMINAMATH_GPT_gcd_72_120_180_is_12_l1717_171760


namespace NUMINAMATH_GPT_student_average_comparison_l1717_171763

theorem student_average_comparison (x y w : ℤ) (hxw : x < w) (hwy : w < y) : 
  (B : ℤ) > (A : ℤ) :=
  let A := (x + y + w) / 3
  let B := ((x + w) / 2 + y) / 2
  sorry

end NUMINAMATH_GPT_student_average_comparison_l1717_171763


namespace NUMINAMATH_GPT_blue_stamp_price_l1717_171756

theorem blue_stamp_price :
  ∀ (red_stamps blue_stamps yellow_stamps : ℕ) (red_price blue_price yellow_price total_earnings : ℝ),
    red_stamps = 20 →
    blue_stamps = 80 →
    yellow_stamps = 7 →
    red_price = 1.1 →
    yellow_price = 2 →
    total_earnings = 100 →
    (red_stamps * red_price + yellow_stamps * yellow_price + blue_stamps * blue_price = total_earnings) →
    blue_price = 0.80 :=
by
  intros red_stamps blue_stamps yellow_stamps red_price blue_price yellow_price total_earnings
  intros h_red_stamps h_blue_stamps h_yellow_stamps h_red_price h_yellow_price h_total_earnings
  intros h_earning_eq
  sorry

end NUMINAMATH_GPT_blue_stamp_price_l1717_171756


namespace NUMINAMATH_GPT_smallest_four_digit_integer_l1717_171701

theorem smallest_four_digit_integer (n : ℕ) (h1 : n ≥ 1000 ∧ n < 10000) 
  (h2 : ∀ d ∈ [1, 5, 6], n % d = 0)
  (h3 : ∀ d1 d2, d1 ≠ d2 → d1 ∈ [1, 5, 6] → d2 ∈ [1, 5, 6] → d1 ≠ d2) :
  n = 1560 :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_integer_l1717_171701


namespace NUMINAMATH_GPT_acute_angle_probability_l1717_171798

noncomputable def prob_acute_angle : ℝ :=
  let m_values := [1, 2, 3, 4, 5, 6]
  let outcomes_count := (36 : ℝ)
  let good_outcomes_count := (15 : ℝ)
  good_outcomes_count / outcomes_count

theorem acute_angle_probability :
  prob_acute_angle = 5 / 12 :=
by
  sorry

end NUMINAMATH_GPT_acute_angle_probability_l1717_171798


namespace NUMINAMATH_GPT_initial_concentration_alcohol_l1717_171707

theorem initial_concentration_alcohol (x : ℝ) 
    (h1 : 0 ≤ x ∧ x ≤ 100)
    (h2 : 0.44 * 10 = (x / 100) * 2 + 3.6) :
    x = 40 :=
sorry

end NUMINAMATH_GPT_initial_concentration_alcohol_l1717_171707


namespace NUMINAMATH_GPT_lines_intersect_lines_perpendicular_lines_parallel_l1717_171755

variables (l1 l2 : ℝ) (m : ℝ)

def intersect (m : ℝ) : Prop :=
  m ≠ -1 ∧ m ≠ 3

def perpendicular (m : ℝ) : Prop :=
  m = 1/2

def parallel (m : ℝ) : Prop :=
  m = -1

theorem lines_intersect (m : ℝ) : intersect m :=
by sorry

theorem lines_perpendicular (m : ℝ) : perpendicular m :=
by sorry

theorem lines_parallel (m : ℝ) : parallel m :=
by sorry

end NUMINAMATH_GPT_lines_intersect_lines_perpendicular_lines_parallel_l1717_171755


namespace NUMINAMATH_GPT_smallest_among_l1717_171734

theorem smallest_among {a b c d : ℝ} (h1 : a = Real.pi) (h2 : b = -2) (h3 : c = 0) (h4 : d = -1) : 
  ∃ (x : ℝ), x = b ∧ x < a ∧ x < c ∧ x < d := 
by {
  sorry
}

end NUMINAMATH_GPT_smallest_among_l1717_171734


namespace NUMINAMATH_GPT_manny_received_fraction_l1717_171745

-- Conditions
def total_marbles : ℕ := 400
def marbles_per_pack : ℕ := 10
def leo_kept_packs : ℕ := 25
def neil_received_fraction : ℚ := 1 / 8

-- Definition of total packs
def total_packs : ℕ := total_marbles / marbles_per_pack

-- Proof problem: What fraction of the total packs did Manny receive?
theorem manny_received_fraction :
  (total_packs - leo_kept_packs - neil_received_fraction * total_packs) / total_packs = 1 / 4 :=
by sorry

end NUMINAMATH_GPT_manny_received_fraction_l1717_171745


namespace NUMINAMATH_GPT_sin_double_angle_value_l1717_171766

theorem sin_double_angle_value 
  (h1 : Real.pi / 2 < α ∧ α < β ∧ β < 3 * Real.pi / 4)
  (h2 : Real.cos (α - β) = 12 / 13)
  (h3 : Real.sin (α + β) = -3 / 5) :
  Real.sin (2 * α) = -16 / 65 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_value_l1717_171766


namespace NUMINAMATH_GPT_find_time_l1717_171731

theorem find_time (s z t : ℝ) (h : ∀ s, 0 ≤ s ∧ s ≤ 7 → z = s^2 + 2 * s) : 
  z = 35 ∧ z = t^2 + 2 * t + 20 → 0 ≤ t ∧ t ≤ 7 → t = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_time_l1717_171731


namespace NUMINAMATH_GPT_gumball_machine_total_l1717_171754

noncomputable def total_gumballs (R B G : ℕ) : ℕ := R + B + G

theorem gumball_machine_total
  (R B G : ℕ)
  (hR : R = 16)
  (hB : B = R / 2)
  (hG : G = 4 * B) :
  total_gumballs R B G = 56 :=
by
  sorry

end NUMINAMATH_GPT_gumball_machine_total_l1717_171754


namespace NUMINAMATH_GPT_age_problem_l1717_171769

theorem age_problem 
  (x y z u : ℕ)
  (h1 : x + 6 = 3 * (y - u))
  (h2 : x = y + z - u)
  (h3: y = x - u) 
  (h4 : x + 19 = 2 * z):
  x = 69 ∧ y = 47 ∧ z = 44 :=
by
  sorry

end NUMINAMATH_GPT_age_problem_l1717_171769


namespace NUMINAMATH_GPT_marble_287_is_blue_l1717_171770

def marble_color (n : ℕ) : String :=
  if n % 15 < 6 then "blue"
  else if n % 15 < 11 then "green"
  else "red"

theorem marble_287_is_blue : marble_color 287 = "blue" :=
by
  sorry

end NUMINAMATH_GPT_marble_287_is_blue_l1717_171770


namespace NUMINAMATH_GPT_gcd_90_450_l1717_171799

theorem gcd_90_450 : Nat.gcd 90 450 = 90 := by
  sorry

end NUMINAMATH_GPT_gcd_90_450_l1717_171799


namespace NUMINAMATH_GPT_problem1_problem2_l1717_171717

noncomputable def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 - a * x + (a - 1) = 0}
noncomputable def C (m : ℝ) : Set ℝ := {x | x^2 - m * x + 2 = 0}

theorem problem1 (a : ℝ) : (∀ x, x ∈ B a → x ∈ A) → a = 2 ∨ a = 3 := sorry

theorem problem2 (m : ℝ) : (∀ x, x ∈ A → x ∈ C m) → m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2) := sorry

end NUMINAMATH_GPT_problem1_problem2_l1717_171717


namespace NUMINAMATH_GPT_prime_roots_sum_product_l1717_171747

theorem prime_roots_sum_product (p q : ℕ) (x1 x2 : ℤ)
  (hp: Nat.Prime p) (hq: Nat.Prime q) 
  (h_sum: x1 + x2 = -↑p)
  (h_prod: x1 * x2 = ↑q) : 
  p = 3 ∧ q = 2 :=
sorry

end NUMINAMATH_GPT_prime_roots_sum_product_l1717_171747


namespace NUMINAMATH_GPT_tracy_total_books_collected_l1717_171744

variable (weekly_books_first_week : ℕ)
variable (multiplier : ℕ)
variable (weeks_next_period : ℕ)

-- Conditions
def first_week_books := 9
def second_period_books_per_week := first_week_books * 10
def books_next_five_weeks := second_period_books_per_week * 5

-- Theorem
theorem tracy_total_books_collected : 
  (first_week_books + books_next_five_weeks) = 459 := 
by 
  sorry

end NUMINAMATH_GPT_tracy_total_books_collected_l1717_171744


namespace NUMINAMATH_GPT_C_days_to_finish_l1717_171761

theorem C_days_to_finish (A B C : ℝ) 
  (h1 : A + B = 1 / 15)
  (h2 : A + B + C = 1 / 11) :
  1 / C = 41.25 :=
by
  -- Given equations
  have h1 : A + B = 1 / 15 := sorry
  have h2 : A + B + C = 1 / 11 := sorry
  -- Calculate C
  let C := 1 / 11 - 1 / 15
  -- Calculate days taken by C
  let days := 1 / C
  -- Prove the days equal to 41.25
  have days_eq : 41.25 = 165 / 4 := sorry
  exact sorry

end NUMINAMATH_GPT_C_days_to_finish_l1717_171761


namespace NUMINAMATH_GPT_correct_statement_is_D_l1717_171743

-- Define each statement as a proposition
def statement_A (a b c : ℕ) : Prop := c ≠ 0 → (a * c = b * c → a = b)
def statement_B : Prop := 30.15 = 30 + 15/60
def statement_C : Prop := ∀ (radius : ℕ), (radius ≠ 0) → (360 * (2 / (2 + 3 + 4)) = 90)
def statement_D : Prop := 9 * 30 + 40/2 = 50

-- Define the theorem to state the correct statement (D)
theorem correct_statement_is_D : statement_D :=
sorry

end NUMINAMATH_GPT_correct_statement_is_D_l1717_171743


namespace NUMINAMATH_GPT_stuffed_animal_total_l1717_171732

/-- McKenna has 34 stuffed animals. -/
def mckenna_stuffed_animals : ℕ := 34

/-- Kenley has twice as many stuffed animals as McKenna. -/
def kenley_stuffed_animals : ℕ := 2 * mckenna_stuffed_animals

/-- Tenly has 5 more stuffed animals than Kenley. -/
def tenly_stuffed_animals : ℕ := kenley_stuffed_animals + 5

/-- The total number of stuffed animals the three girls have. -/
def total_stuffed_animals : ℕ := mckenna_stuffed_animals + kenley_stuffed_animals + tenly_stuffed_animals

/-- Prove that the total number of stuffed animals is 175. -/
theorem stuffed_animal_total : total_stuffed_animals = 175 := by
  sorry

end NUMINAMATH_GPT_stuffed_animal_total_l1717_171732


namespace NUMINAMATH_GPT_base7_arithmetic_l1717_171781

theorem base7_arithmetic : 
  let b1000 := 343  -- corresponding to 1000_7 in decimal
  let b666 := 342   -- corresponding to 666_7 in decimal
  let b1234 := 466  -- corresponding to 1234_7 in decimal
  let s := b1000 + b666  -- sum in decimal
  let s_base7 := 1421    -- sum back in base7 (1421 corresponds to 685 in decimal)
  let r_base7 := 254     -- result from subtraction in base7 (254 corresponds to 172 in decimal)
  (1000 * 7^0 + 0 * 7^1 + 0 * 7^2 + 1 * 7^3) + (6 * 7^0 + 6 * 7^1 + 6 * 7^2) - (4 * 7^0 + 3 * 7^1 + 2 * 7^2 + 1 * 7^3) = (4 * 7^0 + 5 * 7^1 + 2 * 7^2)
  :=
sorry

end NUMINAMATH_GPT_base7_arithmetic_l1717_171781


namespace NUMINAMATH_GPT_wizard_concoction_valid_combinations_l1717_171751

structure WizardConcoction :=
(herbs : Nat)
(crystals : Nat)
(single_incompatible : Nat)
(double_incompatible : Nat)

def valid_combinations (concoction : WizardConcoction) : Nat :=
  concoction.herbs * concoction.crystals - (concoction.single_incompatible + concoction.double_incompatible)

theorem wizard_concoction_valid_combinations (c : WizardConcoction)
  (h_herbs : c.herbs = 4)
  (h_crystals : c.crystals = 6)
  (h_single_incompatible : c.single_incompatible = 1)
  (h_double_incompatible : c.double_incompatible = 2) :
  valid_combinations c = 21 :=
by
  sorry

end NUMINAMATH_GPT_wizard_concoction_valid_combinations_l1717_171751


namespace NUMINAMATH_GPT_solve_inequality_l1717_171757

theorem solve_inequality :
  { x : ℝ // 10 * x^2 - 2 * x - 3 < 0 } =
  { x : ℝ // (1 - Real.sqrt 31) / 10 < x ∧ x < (1 + Real.sqrt 31) / 10 } :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1717_171757


namespace NUMINAMATH_GPT_probability_is_half_l1717_171718

noncomputable def probability_at_least_35_cents : ℚ :=
  let total_outcomes := 32
  let successful_outcomes := 8 + 4 + 4 -- from solution steps (1, 2, 3)
  successful_outcomes / total_outcomes

theorem probability_is_half :
  probability_at_least_35_cents = 1 / 2 := by
  -- proof details are not required as per instructions
  sorry

end NUMINAMATH_GPT_probability_is_half_l1717_171718


namespace NUMINAMATH_GPT_amount_spent_on_giftwrapping_and_expenses_l1717_171782

theorem amount_spent_on_giftwrapping_and_expenses (total_spent : ℝ) (cost_of_gifts : ℝ) (h_total_spent : total_spent = 700) (h_cost_of_gifts : cost_of_gifts = 561) : 
  total_spent - cost_of_gifts = 139 :=
by
  rw [h_total_spent, h_cost_of_gifts]
  norm_num

end NUMINAMATH_GPT_amount_spent_on_giftwrapping_and_expenses_l1717_171782


namespace NUMINAMATH_GPT_raise_percentage_to_original_l1717_171713

-- Let original_salary be a variable representing the original salary.
-- Since the salary was reduced by 50%, the reduced_salary is half of the original_salary.
-- We need to prove that to get the reduced_salary back to the original_salary, 
-- it must be increased by 100%.

noncomputable def original_salary : ℝ := sorry
noncomputable def reduced_salary : ℝ := original_salary * 0.5

theorem raise_percentage_to_original :
  (original_salary - reduced_salary) / reduced_salary * 100 = 100 :=
sorry

end NUMINAMATH_GPT_raise_percentage_to_original_l1717_171713


namespace NUMINAMATH_GPT_product_of_integers_prime_at_most_one_prime_l1717_171784

open Nat

def is_prime (n : ℕ) : Prop :=
  1 < n ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem product_of_integers_prime_at_most_one_prime (a b p : ℤ) (hp : is_prime (Int.natAbs p)) (hprod : a * b = p) :
  (is_prime (Int.natAbs a) ∧ ¬is_prime (Int.natAbs b)) ∨ (¬is_prime (Int.natAbs a) ∧ is_prime (Int.natAbs b)) ∨ ¬is_prime (Int.natAbs a) ∧ ¬is_prime (Int.natAbs b) :=
sorry

end NUMINAMATH_GPT_product_of_integers_prime_at_most_one_prime_l1717_171784


namespace NUMINAMATH_GPT_sequence_term_index_l1717_171737

open Nat

noncomputable def arithmetic_sequence_term (a₁ d n : ℕ) : ℕ :=
a₁ + (n - 1) * d

noncomputable def term_index (a₁ d term : ℕ) : ℕ :=
1 + (term - a₁) / d

theorem sequence_term_index {a₅ a₄₅ term : ℕ}
  (h₁: a₅ = 33)
  (h₂: a₄₅ = 153)
  (h₃: ∀ n, arithmetic_sequence_term 21 3 n = if n = 5 then 33 else if n = 45 then 153 else (21 + (n - 1) * 3))
  : term_index 21 3 201 = 61 :=
sorry

end NUMINAMATH_GPT_sequence_term_index_l1717_171737
