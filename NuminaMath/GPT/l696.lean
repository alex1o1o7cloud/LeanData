import Mathlib

namespace NUMINAMATH_GPT_exponent_equality_l696_69677

theorem exponent_equality (x : ℕ) (hx : (1 / 8 : ℝ) * (2 : ℝ) ^ 40 = (2 : ℝ) ^ x) : x = 37 :=
sorry

end NUMINAMATH_GPT_exponent_equality_l696_69677


namespace NUMINAMATH_GPT_smallest_divisible_by_3_and_4_is_12_l696_69670

theorem smallest_divisible_by_3_and_4_is_12 
  (n : ℕ) 
  (h1 : ∃ k1 : ℕ, n = 3 * k1) 
  (h2 : ∃ k2 : ℕ, n = 4 * k2) 
  : n ≥ 12 := sorry

end NUMINAMATH_GPT_smallest_divisible_by_3_and_4_is_12_l696_69670


namespace NUMINAMATH_GPT_inequality_holds_for_positive_reals_equality_condition_l696_69664

theorem inequality_holds_for_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  4 * (a^3 + b^3 + c^3 + 3) ≥ 3 * (a + 1) * (b + 1) * (c + 1) :=
sorry

theorem equality_condition (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (4 * (a^3 + b^3 + c^3 + 3) = 3 * (a + 1) * (b + 1) * (c + 1)) ↔ (a = 1 ∧ b = 1 ∧ c = 1) :=
sorry

end NUMINAMATH_GPT_inequality_holds_for_positive_reals_equality_condition_l696_69664


namespace NUMINAMATH_GPT_pipe_cut_l696_69671

theorem pipe_cut (x : ℝ) (h1 : x + 2 * x = 177) : 2 * x = 118 :=
by
  sorry

end NUMINAMATH_GPT_pipe_cut_l696_69671


namespace NUMINAMATH_GPT_solve_for_b_l696_69654

/-- 
Given the ellipse \( x^2 + \frac{y^2}{b^2 + 1} = 1 \) where \( b > 0 \),
and the eccentricity of the ellipse is \( \frac{\sqrt{10}}{10} \),
prove that \( b = \frac{1}{3} \).
-/
theorem solve_for_b (b : ℝ) (hb : b > 0) (heccentricity : b / (Real.sqrt (b^2 + 1)) = Real.sqrt 10 / 10) : 
  b = 1 / 3 :=
sorry

end NUMINAMATH_GPT_solve_for_b_l696_69654


namespace NUMINAMATH_GPT_smallest_pos_int_gcd_gt_one_l696_69635

theorem smallest_pos_int_gcd_gt_one : ∃ n: ℕ, n > 0 ∧ (Nat.gcd (8 * n - 3) (5 * n + 4) > 1) ∧ n = 121 :=
by
  sorry

end NUMINAMATH_GPT_smallest_pos_int_gcd_gt_one_l696_69635


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l696_69688

theorem quadratic_inequality_solution (a b c : ℝ) (h1 : a < 0)
  (h2 : (∀ x, ax^2 + bx + c = 0 ↔ x = 1 ∨ x = 3)) : 
  ∀ x, cx^2 + bx + a > 0 ↔ (1/3 < x ∧ x < 1) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l696_69688


namespace NUMINAMATH_GPT_spanish_teams_in_final_probability_l696_69638

noncomputable def probability_of_spanish_teams_in_final : ℚ :=
  let teams := 16
  let spanish_teams := 3
  let non_spanish_teams := teams - spanish_teams
  -- Probability calculation based on given conditions and solution steps
  1 - 7 / 15 * 6 / 14

theorem spanish_teams_in_final_probability :
  probability_of_spanish_teams_in_final = 4 / 5 :=
sorry

end NUMINAMATH_GPT_spanish_teams_in_final_probability_l696_69638


namespace NUMINAMATH_GPT_find_g_x2_minus_2_l696_69648

def g : ℝ → ℝ := sorry -- Define g as some real-valued polynomial function.

theorem find_g_x2_minus_2 (x : ℝ) 
(h1 : g (x^2 + 2) = x^4 + 5 * x^2 + 1) : 
  g (x^2 - 2) = x^4 - 3 * x^2 - 7 := 
by sorry

end NUMINAMATH_GPT_find_g_x2_minus_2_l696_69648


namespace NUMINAMATH_GPT_problem_statement_l696_69661

theorem problem_statement (x y : ℝ) (p : x > 0 ∧ y > 0) : (∃ p, p → xy > 0) ∧ ¬(xy > 0 → x > 0 ∧ y > 0) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l696_69661


namespace NUMINAMATH_GPT_max_value_of_quadratic_l696_69693

theorem max_value_of_quadratic (x : ℝ) (h : 0 < x ∧ x < 6) : (6 - x) * x ≤ 9 := 
by
  sorry

end NUMINAMATH_GPT_max_value_of_quadratic_l696_69693


namespace NUMINAMATH_GPT_largest_gcd_l696_69660

theorem largest_gcd (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h_sum : a + b = 1008) : 
  ∃ d : ℕ, d = Int.gcd a b ∧ d = 504 :=
by
  sorry

end NUMINAMATH_GPT_largest_gcd_l696_69660


namespace NUMINAMATH_GPT_triangle_side_length_sum_l696_69651

theorem triangle_side_length_sum :
  ∃ (a b c : ℕ), (5: ℝ) ^ 2 + (7: ℝ) ^ 2 - 2 * (5: ℝ) * (7: ℝ) * (Real.cos (Real.pi * 80 / 180)) = (a: ℝ) + Real.sqrt b + Real.sqrt c ∧
  b = 62 ∧ c = 0 :=
sorry

end NUMINAMATH_GPT_triangle_side_length_sum_l696_69651


namespace NUMINAMATH_GPT_hyperbola_range_l696_69619

theorem hyperbola_range (m : ℝ) : m * (2 * m - 1) < 0 → 0 < m ∧ m < (1 / 2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_hyperbola_range_l696_69619


namespace NUMINAMATH_GPT_polygon_sides_eq_six_l696_69633

theorem polygon_sides_eq_six (n : ℕ) 
  (h1 : (n - 2) * 180 = (2 * 360)) 
  (h2 : exterior_sum = 360) :
  n = 6 := 
by
  sorry

end NUMINAMATH_GPT_polygon_sides_eq_six_l696_69633


namespace NUMINAMATH_GPT_maximum_value_of_expression_l696_69679

noncomputable def calc_value (x y z : ℝ) : ℝ :=
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2)

theorem maximum_value_of_expression :
  ∃ x y z : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 3 ∧ x ≥ y ∧ y ≥ z ∧
  calc_value x y z = 2916 / 729 :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_of_expression_l696_69679


namespace NUMINAMATH_GPT_solve_x_l696_69689

theorem solve_x (x : ℝ) :
  (5 + 2 * x) / (7 + 3 * x) = (4 + 3 * x) / (9 + 4 * x) ↔
  x = (-5 + Real.sqrt 93) / 2 ∨ x = (-5 - Real.sqrt 93) / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_x_l696_69689


namespace NUMINAMATH_GPT_no_solution_fraction_eq_l696_69640

theorem no_solution_fraction_eq (a : ℝ) :
  (∀ x : ℝ, x ≠ 1 → (a * x / (x - 1) + 3 / (1 - x) = 2) → false) ↔ a = 2 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_fraction_eq_l696_69640


namespace NUMINAMATH_GPT_min_value_f_when_a_eq_one_range_of_a_for_inequality_l696_69684

noncomputable def f (x a : ℝ) : ℝ := |x + 1| + |x - 4| - a

-- Question 1: When a = 1, find the minimum value of the function f(x)
theorem min_value_f_when_a_eq_one : ∃ x : ℝ, ∀ y : ℝ, f y 1 ≥ f x 1 ∧ f x 1 = 4 :=
by
  sorry

-- Question 2: For which values of a does f(x) ≥ 4/a + 1 hold for all real numbers x
theorem range_of_a_for_inequality : (∀ x : ℝ, f x a ≥ 4 / a + 1) ↔ (a < 0 ∨ a = 2) :=
by
  sorry

end NUMINAMATH_GPT_min_value_f_when_a_eq_one_range_of_a_for_inequality_l696_69684


namespace NUMINAMATH_GPT_rail_elevation_correct_angle_l696_69694

noncomputable def rail_elevation_angle (v : ℝ) (R : ℝ) (g : ℝ) : ℝ :=
  Real.arctan (v^2 / (R * g))

theorem rail_elevation_correct_angle :
  rail_elevation_angle (60 * (1000 / 3600)) 200 9.8 = 8.09 := by
  sorry

end NUMINAMATH_GPT_rail_elevation_correct_angle_l696_69694


namespace NUMINAMATH_GPT_rectangle_area_perimeter_max_l696_69681

-- Define the problem conditions
variables {A P : ℝ}

-- Main statement: prove that the maximum value of A / P^2 for a rectangle results in m+n = 17
theorem rectangle_area_perimeter_max (h1 : A = l * w) (h2 : P = 2 * (l + w)) :
  let m := 1
  let n := 16
  m + n = 17 :=
sorry

end NUMINAMATH_GPT_rectangle_area_perimeter_max_l696_69681


namespace NUMINAMATH_GPT_range_c_of_sets_l696_69622

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem range_c_of_sets (c : ℝ) (h₀ : c > 0)
  (A := { x : ℝ | log2 x < 1 })
  (B := { x : ℝ | 0 < x ∧ x < c })
  (hA_union_B_eq_B : A ∪ B = B) :
  c ≥ 2 :=
by
  -- Minimum outline is provided, the proof part is replaced with "sorry" to indicate the point to be proved
  sorry

end NUMINAMATH_GPT_range_c_of_sets_l696_69622


namespace NUMINAMATH_GPT_smallest_percentage_boys_correct_l696_69637

noncomputable def smallest_percentage_boys (B : ℝ) : ℝ :=
  if h : 0 ≤ B ∧ B ≤ 1 then B else 0

theorem smallest_percentage_boys_correct :
  ∃ B : ℝ,
    0 ≤ B ∧ B ≤ 1 ∧
    (67.5 / 100 * B * 200 + 25 / 100 * (1 - B) * 200) ≥ 101 ∧
    B = 0.6 :=
by
  sorry

end NUMINAMATH_GPT_smallest_percentage_boys_correct_l696_69637


namespace NUMINAMATH_GPT_inequality_solution_set_l696_69610

theorem inequality_solution_set (a : ℝ) : (∀ x : ℝ, x > 5 ∧ x > a ↔ x > 5) → a ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l696_69610


namespace NUMINAMATH_GPT_probability_equal_2s_after_4040_rounds_l696_69649

/-- 
Given three players Diana, Nathan, and Olivia each starting with $2, each player (with at least $1) 
simultaneously gives $1 to one of the other two players randomly every 20 seconds. 
Prove that the probability that after the bell has rung 4040 times, 
each player will have $2$ is $\frac{1}{4}$.
-/
theorem probability_equal_2s_after_4040_rounds 
  (n_rounds : ℕ) (start_money : ℕ) (probability_outcome : ℚ) :
  n_rounds = 4040 →
  start_money = 2 →
  probability_outcome = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_equal_2s_after_4040_rounds_l696_69649


namespace NUMINAMATH_GPT_find_values_l696_69650

theorem find_values (a b c : ℝ)
  (h1 : 0.005 * a = 0.8)
  (h2 : 0.0025 * b = 0.6)
  (h3 : c = 0.5 * a - 0.1 * b) :
  a = 160 ∧ b = 240 ∧ c = 56 :=
by sorry

end NUMINAMATH_GPT_find_values_l696_69650


namespace NUMINAMATH_GPT_isosceles_trapezoid_inscribed_circle_ratio_l696_69615

noncomputable def ratio_perimeter_inscribed_circle (x : ℝ) : ℝ := 
  (50 * x) / (10 * Real.pi * x)

theorem isosceles_trapezoid_inscribed_circle_ratio 
  (x : ℝ)
  (h1 : x > 0)
  (r : ℝ) 
  (OK OP : ℝ) 
  (h2 : OK = 3 * x) 
  (h3 : OP = 5 * x) : 
  ratio_perimeter_inscribed_circle x = 5 / Real.pi :=
by
  sorry

end NUMINAMATH_GPT_isosceles_trapezoid_inscribed_circle_ratio_l696_69615


namespace NUMINAMATH_GPT_percent_students_own_only_cats_l696_69631

theorem percent_students_own_only_cats (total_students : ℕ) (students_owning_cats : ℕ) (students_owning_dogs : ℕ) (students_owning_both : ℕ) (h_total : total_students = 500) (h_cats : students_owning_cats = 80) (h_dogs : students_owning_dogs = 150) (h_both : students_owning_both = 40) : 
  (students_owning_cats - students_owning_both) * 100 / total_students = 8 := 
by
  sorry

end NUMINAMATH_GPT_percent_students_own_only_cats_l696_69631


namespace NUMINAMATH_GPT_simon_age_is_10_l696_69627

-- Define the conditions
def alvin_age := 30
def half_alvin_age := alvin_age / 2
def simon_age := half_alvin_age - 5

-- State the theorem
theorem simon_age_is_10 : simon_age = 10 :=
by
  sorry

end NUMINAMATH_GPT_simon_age_is_10_l696_69627


namespace NUMINAMATH_GPT_smallest_x_for_gx_eq_g1458_l696_69617

noncomputable def g : ℝ → ℝ := sorry -- You can define the function later.

theorem smallest_x_for_gx_eq_g1458 :
  (∀ x : ℝ, x > 0 → g (3 * x) = 4 * g x) ∧ (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → g x = 2 - 2 * |x - 2|)
  → ∃ x : ℝ, x ≥ 0 ∧ g x = g 1458 ∧ ∀ y : ℝ, y ≥ 0 ∧ g y = g 1458 → x ≤ y ∧ x = 162 := 
by
  sorry

end NUMINAMATH_GPT_smallest_x_for_gx_eq_g1458_l696_69617


namespace NUMINAMATH_GPT_solve_ineq_l696_69606

theorem solve_ineq (x : ℝ) : (x > 0 ∧ x < 3 ∨ x > 8) → x^3 - 9 * x^2 + 24 * x > 0 :=
by
  sorry

end NUMINAMATH_GPT_solve_ineq_l696_69606


namespace NUMINAMATH_GPT_ap_square_sequel_l696_69628

theorem ap_square_sequel {a b c : ℝ} (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c)
                     (h2 : 2 * (b / (c + a)) = (a / (b + c)) + (c / (a + b))) :
  (a^2 + c^2 = 2 * b^2) :=
by
  sorry

end NUMINAMATH_GPT_ap_square_sequel_l696_69628


namespace NUMINAMATH_GPT_intersection_correct_l696_69603

def setA := {x : ℝ | (x - 2) * (2 * x + 1) ≤ 0}
def setB := {x : ℝ | x < 1}
def expectedIntersection := {x : ℝ | -1 / 2 ≤ x ∧ x < 1}

theorem intersection_correct : (setA ∩ setB) = expectedIntersection := by
  sorry

end NUMINAMATH_GPT_intersection_correct_l696_69603


namespace NUMINAMATH_GPT_find_hyperbola_equation_hyperbola_equation_l696_69636

-- Define the original hyperbola
def original_hyperbola (x y : ℝ) := (x^2 / 2) - y^2 = 1

-- Define the new hyperbola with unknown constant m
def new_hyperbola (x y m : ℝ) := (x^2 / (m * 2)) - (y^2 / m) = 1

variable (m : ℝ)

-- The point (2, 0)
def point_on_hyperbola (x y : ℝ) := x = 2 ∧ y = 0

theorem find_hyperbola_equation (h : ∀ (x y : ℝ), point_on_hyperbola x y → new_hyperbola x y m) :
  m = 2 :=
    sorry

theorem hyperbola_equation :
  ∀ (x y : ℝ), (x = 2 ∧ y = 0) → (x^2 / 4 - y^2 / 2 = 1) :=
    sorry

end NUMINAMATH_GPT_find_hyperbola_equation_hyperbola_equation_l696_69636


namespace NUMINAMATH_GPT_average_tree_height_l696_69607

theorem average_tree_height :
  let tree1 := 8
  let tree2 := if tree3 = 16 then 4 else 16
  let tree3 := 16
  let tree4 := if tree5 = 32 then 8 else 32
  let tree5 := 32
  let tree6 := if tree5 = 32 then 64 else 16
  let total_sum := tree1 + tree2 + tree3 + tree4 + tree5 + tree6
  let average_height := total_sum / 6
  average_height = 14 :=
by
  sorry

end NUMINAMATH_GPT_average_tree_height_l696_69607


namespace NUMINAMATH_GPT_cubes_divisible_by_9_l696_69691

theorem cubes_divisible_by_9 (n: ℕ) (h: n > 0) : 9 ∣ n^3 + (n + 1)^3 + (n + 2)^3 :=
by 
  sorry

end NUMINAMATH_GPT_cubes_divisible_by_9_l696_69691


namespace NUMINAMATH_GPT_intersection_complement_A_l696_69675

def A : Set ℝ := {x | abs (x - 1) < 1}

def B : Set ℝ := {x | x < 1}

def CRB : Set ℝ := {x | x ≥ 1}

theorem intersection_complement_A :
  (CRB ∩ A) = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_A_l696_69675


namespace NUMINAMATH_GPT_largest_5_digit_congruent_l696_69665

theorem largest_5_digit_congruent (n : ℕ) (h1 : 29 * n + 17 < 100000) : 29 * 3447 + 17 = 99982 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_largest_5_digit_congruent_l696_69665


namespace NUMINAMATH_GPT__l696_69639

noncomputable def tan_alpha_theorem (α : ℝ) (h1 : Real.tan (Real.pi / 4 + α) = 2) : Real.tan α = 1 / 3 :=
by
  sorry

noncomputable def evaluate_expression_theorem (α β : ℝ) 
  (h1 : Real.tan (Real.pi / 4 + α) = 2) 
  (h2 : Real.tan β = 1 / 2) 
  (h3 : Real.tan α = 1 / 3) : 
  (Real.sin (α + β) - 2 * Real.sin α * Real.cos β) / (2 * Real.sin α * Real.sin β + Real.cos (α + β)) = 1 / 7 :=
by
  sorry

end NUMINAMATH_GPT__l696_69639


namespace NUMINAMATH_GPT_line_circle_no_intersection_l696_69667

theorem line_circle_no_intersection :
  (∀ (x y : ℝ), 3 * x + 4 * y = 12 ∨ x^2 + y^2 = 4) →
  (∃ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4) →
  false :=
by
  sorry

end NUMINAMATH_GPT_line_circle_no_intersection_l696_69667


namespace NUMINAMATH_GPT_cos_ninety_degrees_l696_69647

theorem cos_ninety_degrees : Real.cos (90 * Real.pi / 180) = 0 := 
by 
  sorry

end NUMINAMATH_GPT_cos_ninety_degrees_l696_69647


namespace NUMINAMATH_GPT_regular_polygon_sides_l696_69699

theorem regular_polygon_sides (n : ℕ) (h : 108 = 180 * (n - 2) / n) : n = 5 := 
sorry

end NUMINAMATH_GPT_regular_polygon_sides_l696_69699


namespace NUMINAMATH_GPT_shoes_per_person_l696_69605

theorem shoes_per_person (friends : ℕ) (pairs_of_shoes : ℕ) 
  (h1 : friends = 35) (h2 : pairs_of_shoes = 36) : 
  (pairs_of_shoes * 2) / (friends + 1) = 2 := by
  sorry

end NUMINAMATH_GPT_shoes_per_person_l696_69605


namespace NUMINAMATH_GPT_sequence_value_at_99_l696_69621

theorem sequence_value_at_99 :
  ∃ a : ℕ → ℚ, (a 1 = 2) ∧ (∀ n : ℕ, a (n + 1) = a n + n / 2) ∧ (a 99 = 2427.5) :=
by
  sorry

end NUMINAMATH_GPT_sequence_value_at_99_l696_69621


namespace NUMINAMATH_GPT_num_divisible_by_10_in_range_correct_l696_69680

noncomputable def num_divisible_by_10_in_range : ℕ :=
  let a1 := 100
  let d := 10
  let an := 500
  (an - a1) / d + 1

theorem num_divisible_by_10_in_range_correct :
  num_divisible_by_10_in_range = 41 := by
  sorry

end NUMINAMATH_GPT_num_divisible_by_10_in_range_correct_l696_69680


namespace NUMINAMATH_GPT_find_x_l696_69652

theorem find_x :
  let a := 5^3
  let b := 6^2
  a - 7 = b + 82 := 
by
  sorry

end NUMINAMATH_GPT_find_x_l696_69652


namespace NUMINAMATH_GPT_quadratic_root_m_value_l696_69685

theorem quadratic_root_m_value (m : ℝ) (x : ℝ) (h : x = 1) (hx : x^2 + m * x + 2 = 0) : m = -3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_m_value_l696_69685


namespace NUMINAMATH_GPT_PRINT_3_3_2_l696_69614

def PRINT (a b : Nat) : Nat × Nat := (a, b)

theorem PRINT_3_3_2 :
  PRINT 3 (3 + 2) = (3, 5) :=
by
  sorry

end NUMINAMATH_GPT_PRINT_3_3_2_l696_69614


namespace NUMINAMATH_GPT_medium_supermarkets_in_sample_l696_69634

-- Define the conditions
def large_supermarkets : ℕ := 200
def medium_supermarkets : ℕ := 400
def small_supermarkets : ℕ := 1400
def total_supermarkets : ℕ := large_supermarkets + medium_supermarkets + small_supermarkets
def sample_size : ℕ := 100
def proportion_medium := (medium_supermarkets : ℚ) / (total_supermarkets : ℚ)

-- The main theorem to prove
theorem medium_supermarkets_in_sample : sample_size * proportion_medium = 20 := by
  sorry

end NUMINAMATH_GPT_medium_supermarkets_in_sample_l696_69634


namespace NUMINAMATH_GPT_sam_puppies_count_l696_69609

variable (initial_puppies : ℝ) (given_away_puppies : ℝ)

theorem sam_puppies_count (h1 : initial_puppies = 6.0) 
                          (h2 : given_away_puppies = 2.0) : 
                          initial_puppies - given_away_puppies = 4.0 :=
by simp [h1, h2]; sorry

end NUMINAMATH_GPT_sam_puppies_count_l696_69609


namespace NUMINAMATH_GPT_range_of_a_iff_condition_l696_69626

theorem range_of_a_iff_condition (a : ℝ) : 
  (∀ x : ℝ, |x + 3| + |x - 7| ≥ a^2 - 3 * a) ↔ (a ≥ -2 ∧ a ≤ 5) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_iff_condition_l696_69626


namespace NUMINAMATH_GPT_inequality_proof_l696_69698

theorem inequality_proof (n : ℕ) (hn : n > 0) : (2 * n + 1) ^ n ≥ (2 * n) ^ n + (2 * n - 1) ^ n :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l696_69698


namespace NUMINAMATH_GPT_real_solutions_count_l696_69643

theorem real_solutions_count :
  ∃ S : Set ℝ, (∀ x : ℝ, x ∈ S ↔ (|x-2| + |x-3| = 1)) ∧ (S = Set.Icc 2 3) :=
sorry

end NUMINAMATH_GPT_real_solutions_count_l696_69643


namespace NUMINAMATH_GPT_find_p_q_l696_69624

noncomputable def roots_of_polynomial (a b c : ℝ) :=
  a^3 - 2018 * a + 2018 = 0 ∧ b^3 - 2018 * b + 2018 = 0 ∧ c^3 - 2018 * c + 2018 = 0

theorem find_p_q (a b c : ℝ) (p q : ℕ) 
  (h1 : roots_of_polynomial a b c)
  (h2 : 0 < p ∧ p ≤ q) 
  (h3 : (a^(p+q) + b^(p+q) + c^(p+q))/(p+q) = (a^p + b^p + c^p)/p * (a^q + b^q + c^q)/q) : 
  p^2 + q^2 = 20 := 
sorry

end NUMINAMATH_GPT_find_p_q_l696_69624


namespace NUMINAMATH_GPT_cost_per_play_l696_69658

-- Conditions
def initial_money : ℝ := 3
def points_per_red_bucket : ℝ := 2
def points_per_green_bucket : ℝ := 3
def rings_per_play : ℕ := 5
def games_played : ℕ := 2
def red_buckets : ℕ := 4
def green_buckets : ℕ := 5
def total_games : ℕ := 3
def total_points : ℝ := 38

-- Point calculations
def points_from_red_buckets : ℝ := red_buckets * points_per_red_bucket
def points_from_green_buckets : ℝ := green_buckets * points_per_green_bucket
def current_points : ℝ := points_from_red_buckets + points_from_green_buckets
def points_needed : ℝ := total_points - current_points

-- Define the theorem statement
theorem cost_per_play :
  (initial_money / (games_played : ℝ)) = 1.50 :=
  sorry

end NUMINAMATH_GPT_cost_per_play_l696_69658


namespace NUMINAMATH_GPT_twelfth_term_of_geometric_sequence_l696_69601

theorem twelfth_term_of_geometric_sequence 
  (a : ℕ → ℕ)
  (h₁ : a 4 = 4)
  (h₂ : a 7 = 32)
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * r) : 
  a 12 = 1024 :=
sorry

end NUMINAMATH_GPT_twelfth_term_of_geometric_sequence_l696_69601


namespace NUMINAMATH_GPT_james_weekly_earnings_l696_69674

def hourly_rate : ℕ := 20
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 4

theorem james_weekly_earnings : hourly_rate * (hours_per_day * days_per_week) = 640 := by
  sorry

end NUMINAMATH_GPT_james_weekly_earnings_l696_69674


namespace NUMINAMATH_GPT_simplify_expression_l696_69620

variable (x y : ℝ)

theorem simplify_expression : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 9 * y = 45 * x + 9 * y := 
by sorry

end NUMINAMATH_GPT_simplify_expression_l696_69620


namespace NUMINAMATH_GPT_part_a_part_b_l696_69687

variable (a b : ℝ)

-- Given conditions
variable (h1 : a^3 - b^3 = 2) (h2 : a^5 - b^5 ≥ 4)

-- Requirement (a): Prove that a > b
theorem part_a : a > b := by 
  sorry

-- Requirement (b): Prove that a^2 + b^2 ≥ 2
theorem part_b : a^2 + b^2 ≥ 2 := by 
  sorry

end NUMINAMATH_GPT_part_a_part_b_l696_69687


namespace NUMINAMATH_GPT_geometric_series_sum_l696_69663

noncomputable def geometric_sum : ℚ :=
  let a := (2^3 : ℚ) / (3^3)
  let r := (2 : ℚ) / 3
  let n := 12 - 3 + 1
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  geometric_sum = 1440600 / 59049 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l696_69663


namespace NUMINAMATH_GPT_students_in_trumpet_or_trombone_l696_69666

theorem students_in_trumpet_or_trombone (h₁ : 0.5 + 0.12 = 0.62) : 
  0.5 + 0.12 = 0.62 :=
by
  exact h₁

end NUMINAMATH_GPT_students_in_trumpet_or_trombone_l696_69666


namespace NUMINAMATH_GPT_pairwise_sums_l696_69623

theorem pairwise_sums (
  a b c d e : ℕ
) : 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧
  (a + b = 21) ∧ (a + c = 26) ∧ (a + d = 35) ∧ (a + e = 40) ∧
  (b + c = 49) ∧ (b + d = 51) ∧ (b + e = 54) ∧ (c + d = 60) ∧
  (c + e = 65) ∧ (d + e = 79)
  ↔ 
  (a = 6) ∧ (b = 15) ∧ (c = 20) ∧ (d = 34) ∧ (e = 45) := 
by 
  sorry

end NUMINAMATH_GPT_pairwise_sums_l696_69623


namespace NUMINAMATH_GPT_find_number_l696_69656

def number_of_faces : ℕ := 6

noncomputable def probability (n : ℕ) : ℚ :=
  (number_of_faces - n : ℕ) / number_of_faces

theorem find_number (n : ℕ) (h: n < number_of_faces) :
  probability n = 1 / 3 → n = 4 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_find_number_l696_69656


namespace NUMINAMATH_GPT_fff1_eq_17_l696_69629

def f (n : ℕ) : ℕ :=
  if n < 3 then n^2 + 1
  else if n < 6 then 3 * n + 2
  else 2 * n - 1

theorem fff1_eq_17 : f (f (f 1)) = 17 :=
  by sorry

end NUMINAMATH_GPT_fff1_eq_17_l696_69629


namespace NUMINAMATH_GPT_zorbs_of_60_deg_l696_69683

-- Define the measurement on Zorblat
def zorbs_in_full_circle := 600
-- Define the Earth angle in degrees
def earth_degrees_full_circle := 360
def angle_in_degrees := 60
-- Calculate the equivalent angle in zorbs
def zorbs_in_angle := zorbs_in_full_circle * angle_in_degrees / earth_degrees_full_circle

theorem zorbs_of_60_deg (h1 : zorbs_in_full_circle = 600)
                        (h2 : earth_degrees_full_circle = 360)
                        (h3 : angle_in_degrees = 60) :
  zorbs_in_angle = 100 :=
by sorry

end NUMINAMATH_GPT_zorbs_of_60_deg_l696_69683


namespace NUMINAMATH_GPT_david_dogs_left_l696_69618

def total_dogs_left (boxes_small: Nat) (dogs_per_small: Nat) (boxes_large: Nat) (dogs_per_large: Nat) (giveaway_small: Nat) (giveaway_large: Nat): Nat :=
  let total_small := boxes_small * dogs_per_small
  let total_large := boxes_large * dogs_per_large
  let remaining_small := total_small - giveaway_small
  let remaining_large := total_large - giveaway_large
  remaining_small + remaining_large

theorem david_dogs_left :
  total_dogs_left 7 4 5 3 2 1 = 40 := by
  sorry

end NUMINAMATH_GPT_david_dogs_left_l696_69618


namespace NUMINAMATH_GPT_compute_expression_l696_69669

noncomputable def quadratic_roots (a b c : ℝ) :
  {x : ℝ × ℝ // a * x.fst^2 + b * x.fst + c = 0 ∧ a * x.snd^2 + b * x.snd + c = 0} :=
  let Δ := b^2 - 4 * a * c
  let root1 := (-b + Real.sqrt Δ) / (2 * a)
  let root2 := (-b - Real.sqrt Δ) / (2 * a)
  ⟨(root1, root2), by sorry⟩

theorem compute_expression :
  let roots := quadratic_roots 5 (-3) (-4)
  let x1 := roots.val.fst
  let x2 := roots.val.snd
  2 * x1^2 + 3 * x2^2 = (178 : ℝ) / 25 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l696_69669


namespace NUMINAMATH_GPT_twelve_year_olds_count_l696_69612

theorem twelve_year_olds_count (x y z w : ℕ) 
  (h1 : x + y + z + w = 23)
  (h2 : 10 * x + 11 * y + 12 * z + 13 * w = 253)
  (h3 : z = 3 * w / 2) : 
  z = 6 :=
by sorry

end NUMINAMATH_GPT_twelve_year_olds_count_l696_69612


namespace NUMINAMATH_GPT_quadratic_no_real_roots_l696_69642

theorem quadratic_no_real_roots (b c : ℝ) (h : ∀ x : ℝ, x^2 + b * x + c > 0) : 
  ¬ ∃ x : ℝ, x^2 + b * x + c = 0 :=
sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_l696_69642


namespace NUMINAMATH_GPT_coupon_value_l696_69668

theorem coupon_value
  (bill : ℝ)
  (milk_cost : ℝ)
  (bread_cost : ℝ)
  (detergent_cost : ℝ)
  (banana_cost_per_pound : ℝ)
  (banana_weight : ℝ)
  (half_off : ℝ)
  (amount_left : ℝ)
  (total_without_coupon : ℝ)
  (total_spent : ℝ)
  (coupon_value : ℝ) :
  bill = 20 →
  milk_cost = 4 →
  bread_cost = 3.5 →
  detergent_cost = 10.25 →
  banana_cost_per_pound = 0.75 →
  banana_weight = 2 →
  half_off = 0.5 →
  amount_left = 4 →
  total_without_coupon = milk_cost * half_off + bread_cost + detergent_cost + banana_cost_per_pound * banana_weight →
  total_spent = bill - amount_left →
  coupon_value = total_without_coupon - total_spent →
  coupon_value = 1.25 :=
by
  sorry

end NUMINAMATH_GPT_coupon_value_l696_69668


namespace NUMINAMATH_GPT_income_expenditure_ratio_l696_69602

noncomputable def I : ℝ := 19000
noncomputable def S : ℝ := 3800
noncomputable def E : ℝ := I - S

theorem income_expenditure_ratio : (I / E) = 5 / 4 := by
  sorry

end NUMINAMATH_GPT_income_expenditure_ratio_l696_69602


namespace NUMINAMATH_GPT_valid_numbers_count_l696_69625

-- Define a predicate that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that counts how many numbers between 100 and 999 are multiples of 13
def count_multiples_of_13 (start finish : ℕ) : ℕ :=
  (finish - start) / 13 + 1

-- Define a function that checks if a permutation of digits of n is a multiple of 13
-- (actual implementation would require digit manipulation, but we assume its existence here)
def is_permutation_of_digits_multiple_of_13 (n : ℕ) : Prop :=
  ∃ (perm : ℕ), is_three_digit perm ∧ perm % 13 = 0

noncomputable def count_valid_permutations (multiples_of_13 : ℕ) : ℕ :=
  multiples_of_13 * 3 -- Assuming on average

-- Problem statement: Prove that there are 207 valid numbers satisfying the condition
theorem valid_numbers_count : (count_valid_permutations (count_multiples_of_13 104 988)) = 207 := 
by {
  -- Place for proof which is omitted here
  sorry
}

end NUMINAMATH_GPT_valid_numbers_count_l696_69625


namespace NUMINAMATH_GPT_value_of_m_div_x_l696_69611

variables (a b : ℝ) (k : ℝ)
-- Condition: The ratio of a to b is 4 to 5
def ratio_a_to_b : Prop := a / b = 4 / 5

-- Condition: x equals a increased by 75 percent of a
def x := a + 0.75 * a

-- Condition: m equals b decreased by 80 percent of b
def m := b - 0.80 * b

-- Prove the given question
theorem value_of_m_div_x (h1 : ratio_a_to_b a b) (ha_pos : 0 < a) (hb_pos : 0 < b) :
  m / x = 1 / 7 := by
sorry

end NUMINAMATH_GPT_value_of_m_div_x_l696_69611


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_l696_69613

theorem sum_of_arithmetic_sequence
  (a : ℕ → ℚ)
  (S : ℕ → ℚ)
  (h1 : a 2 * a 4 * a 6 * a 8 = 120)
  (h2 : 1 / (a 4 * a 6 * a 8) + 1 / (a 2 * a 6 * a 8) + 1 / (a 2 * a 4 * a 8) + 1 / (a 2 * a 4 * a 6) = 7/60) :
  S 9 = 63/2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_l696_69613


namespace NUMINAMATH_GPT_work_problem_l696_69682

theorem work_problem (hA : ∀ n : ℝ, n = 15)
  (h_work_together : ∀ n : ℝ, 3 * (1/15 + 1/n) = 0.35) :  
  1/20 = 1/20 :=
by
  sorry

end NUMINAMATH_GPT_work_problem_l696_69682


namespace NUMINAMATH_GPT_cake_and_milk_tea_cost_l696_69641

noncomputable def slice_cost (milk_tea_cost : ℚ) : ℚ := (3 / 4) * milk_tea_cost

noncomputable def total_cost (milk_tea_cost : ℚ) (slice_cost : ℚ) : ℚ :=
  2 * slice_cost + milk_tea_cost

theorem cake_and_milk_tea_cost 
  (milk_tea_cost : ℚ)
  (h : milk_tea_cost = 2.40) :
  total_cost milk_tea_cost (slice_cost milk_tea_cost) = 6.00 :=
by
  sorry

end NUMINAMATH_GPT_cake_and_milk_tea_cost_l696_69641


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_not_necessary_l696_69690

theorem sufficient_but_not_necessary (a : ℝ) (h : a > 0) : (|a| > 0) := by
  sorry

theorem not_necessary (a : ℝ) : |a| > 0 → ¬(a = 0) ∧ (a ≠ 0 → |a| > 0 ∧ (¬(a > 0) → (|a| > 0))) := by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_not_necessary_l696_69690


namespace NUMINAMATH_GPT_no_positive_ints_cube_l696_69645

theorem no_positive_ints_cube (n : ℕ) : ¬ ∃ y : ℕ, 3 * n^2 + 3 * n + 7 = y^3 := 
sorry

end NUMINAMATH_GPT_no_positive_ints_cube_l696_69645


namespace NUMINAMATH_GPT_height_eight_times_initial_maximum_growth_year_l696_69659

noncomputable def t : ℝ := 2^(-2/3 : ℝ)
noncomputable def f (n : ℕ) (A a b t : ℝ) : ℝ := 9 * A / (a + b * t^n)

theorem height_eight_times_initial (A : ℝ) : 
  ∀ n : ℕ, f n A 1 8 t = 8 * A ↔ n = 9 :=
sorry

theorem maximum_growth_year (A : ℝ) :
  ∃ n : ℕ, (∀ k : ℕ, (f n A 1 8 t - f (n-1) A 1 8 t) ≥ (f k A 1 8 t - f (k-1) A 1 8 t))
  ∧ n = 5 :=
sorry

end NUMINAMATH_GPT_height_eight_times_initial_maximum_growth_year_l696_69659


namespace NUMINAMATH_GPT_technicians_count_l696_69673

-- Define the number of workers
def total_workers : ℕ := 21

-- Define the average salaries
def avg_salary_all : ℕ := 8000
def avg_salary_technicians : ℕ := 12000
def avg_salary_rest : ℕ := 6000

-- Define the number of technicians and rest of workers
variable (T R : ℕ)

-- Define the equations based on given conditions
def equation1 := T + R = total_workers
def equation2 := (T * avg_salary_technicians) + (R * avg_salary_rest) = total_workers * avg_salary_all

-- Prove the number of technicians
theorem technicians_count : T = 7 :=
by
  sorry

end NUMINAMATH_GPT_technicians_count_l696_69673


namespace NUMINAMATH_GPT_solve_for_w_squared_l696_69644

-- Define the original equation
def eqn (w : ℝ) := 2 * (w + 15)^2 = (4 * w + 9) * (3 * w + 6)

-- Define the goal to prove w^2 = 6.7585 based on the given equation
theorem solve_for_w_squared : ∃ w : ℝ, eqn w ∧ w^2 = 6.7585 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_w_squared_l696_69644


namespace NUMINAMATH_GPT_square_area_with_circles_l696_69672

theorem square_area_with_circles 
  (radius : ℝ) 
  (circle_count : ℕ) 
  (side_length : ℝ) 
  (total_area : ℝ)
  (h1 : radius = 7)
  (h2 : circle_count = 4)
  (h3 : side_length = 2 * (2 * radius))
  (h4 : total_area = side_length * side_length)
  : total_area = 784 :=
sorry

end NUMINAMATH_GPT_square_area_with_circles_l696_69672


namespace NUMINAMATH_GPT_carpet_coverage_percentage_l696_69604

variable (l w : ℝ) (floor_area carpet_area : ℝ)

theorem carpet_coverage_percentage 
  (h_carpet_area: carpet_area = l * w) 
  (h_floor_area: floor_area = 180) 
  (hl : l = 4) 
  (hw : w = 9) : 
  carpet_area / floor_area * 100 = 20 := by
  sorry

end NUMINAMATH_GPT_carpet_coverage_percentage_l696_69604


namespace NUMINAMATH_GPT_sum_of_numbers_less_than_2_l696_69692

theorem sum_of_numbers_less_than_2:
  ∀ (a b c : ℝ), a = 0.8 → b = 1/2 → c = 0.9 → a < 2 ∧ b < 2 ∧ c < 2 → a + b + c = 2.2 :=
by
  -- We are stating that if a = 0.8, b = 1/2, and c = 0.9, and all are less than 2, then their sum is 2.2
  sorry

end NUMINAMATH_GPT_sum_of_numbers_less_than_2_l696_69692


namespace NUMINAMATH_GPT_ship_illuminated_by_lighthouse_l696_69630

theorem ship_illuminated_by_lighthouse (d v : ℝ) (hv : v > 0) (ship_speed : ℝ) 
    (hship_speed : ship_speed ≤ v / 8) (rock_distance : ℝ) 
    (hrock_distance : rock_distance = d):
    ∀ t : ℝ, ∃ t' : ℝ, t' ≤ t ∧ t' = (d * t / v) := sorry

end NUMINAMATH_GPT_ship_illuminated_by_lighthouse_l696_69630


namespace NUMINAMATH_GPT_no_triangle_with_perfect_square_sides_l696_69600

theorem no_triangle_with_perfect_square_sides :
  ∃ (a b : ℕ), a > 1000 ∧ b > 1000 ∧
    ∀ (c : ℕ), (∃ d : ℕ, c = d^2) → 
    ¬ (a + b > c ∧ b + c > a ∧ a + c > b) :=
sorry

end NUMINAMATH_GPT_no_triangle_with_perfect_square_sides_l696_69600


namespace NUMINAMATH_GPT_boat_current_ratio_l696_69657

noncomputable def boat_speed_ratio (b c : ℝ) (d : ℝ) : Prop :=
  let time_upstream := 6
  let time_downstream := 10
  d = time_upstream * (b - c) ∧ 
  d = time_downstream * (b + c) → 
  b / c = 4

theorem boat_current_ratio (b c d : ℝ) (h1 : d = 6 * (b - c)) (h2 : d = 10 * (b + c)) : b / c = 4 :=
by sorry

end NUMINAMATH_GPT_boat_current_ratio_l696_69657


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l696_69678

noncomputable def arithmetic_sequence_sum : ℕ → ℕ := sorry  -- Define S_n here

theorem arithmetic_sequence_problem (S : ℕ → ℕ) (a : ℕ → ℕ) (h1 : S 8 - S 3 = 10)
    (h2 : ∀ n, S (n + 1) = S n + a (n + 1)) (h3 : a 6 = 2) : S 11 = 22 :=
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l696_69678


namespace NUMINAMATH_GPT_academic_academy_pass_criteria_l696_69653

theorem academic_academy_pass_criteria :
  ∀ (total_problems : ℕ) (passing_percentage : ℕ)
  (max_missed : ℕ),
  total_problems = 35 →
  passing_percentage = 80 →
  max_missed = total_problems - (passing_percentage * total_problems) / 100 →
  max_missed = 7 :=
by 
  intros total_problems passing_percentage max_missed
  intros h_total_problems h_passing_percentage h_calculation
  rw [h_total_problems, h_passing_percentage] at h_calculation
  sorry

end NUMINAMATH_GPT_academic_academy_pass_criteria_l696_69653


namespace NUMINAMATH_GPT_intersection_A_B_l696_69676

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {-2, -1, 1, 2}

theorem intersection_A_B : A ∩ B = {1, 2} :=
by 
  sorry

end NUMINAMATH_GPT_intersection_A_B_l696_69676


namespace NUMINAMATH_GPT_chocolate_chip_difference_l696_69662

noncomputable def V_v : ℕ := 20 -- Viviana's vanilla chips
noncomputable def S_c : ℕ := 25 -- Susana's chocolate chips
noncomputable def S_v : ℕ := 3 * V_v / 4 -- Susana's vanilla chips

theorem chocolate_chip_difference (V_c : ℕ) (h1 : V_c + V_v + S_c + S_v = 90) :
  V_c - S_c = 5 := by sorry

end NUMINAMATH_GPT_chocolate_chip_difference_l696_69662


namespace NUMINAMATH_GPT_num_marbles_removed_l696_69616

theorem num_marbles_removed (total_marbles red_marbles : ℕ) (prob_neither_red : ℚ) 
  (h₁ : total_marbles = 84) (h₂ : red_marbles = 12) (h₃ : prob_neither_red = 36 / 49) : 
  total_marbles - red_marbles = 2 :=
by
  sorry

end NUMINAMATH_GPT_num_marbles_removed_l696_69616


namespace NUMINAMATH_GPT_johnny_selection_process_l696_69686

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem johnny_selection_process : 
  binomial_coefficient 10 4 * binomial_coefficient 4 2 = 1260 :=
by
  sorry

end NUMINAMATH_GPT_johnny_selection_process_l696_69686


namespace NUMINAMATH_GPT_arithmetic_geometric_progression_inequality_l696_69608

theorem arithmetic_geometric_progression_inequality
  {a b c d e f D g : ℝ}
  (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d)
  (e_pos : 0 < e) (f_pos : 0 < f)
  (h1 : b = a + D)
  (h2 : c = a + 2 * D)
  (h3 : e = a * g)
  (h4 : f = a * g^2)
  (h5 : d = a + 3 * D)
  (h6 : d = a * g^3) : 
  b * c ≥ e * f :=
by sorry

end NUMINAMATH_GPT_arithmetic_geometric_progression_inequality_l696_69608


namespace NUMINAMATH_GPT_distance_between_first_and_last_pots_l696_69655

theorem distance_between_first_and_last_pots (n : ℕ) (d : ℕ) 
  (h₁ : n = 8) 
  (h₂ : d = 100) : 
  ∃ total_distance : ℕ, total_distance = 175 := 
by 
  sorry

end NUMINAMATH_GPT_distance_between_first_and_last_pots_l696_69655


namespace NUMINAMATH_GPT_ratio_sqrt_2_l696_69646

theorem ratio_sqrt_2 {a b : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : a^2 + b^2 = 6 * a * b) :
  (a + b) / (a - b) = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_sqrt_2_l696_69646


namespace NUMINAMATH_GPT_graph_of_f_does_not_pass_through_second_quadrant_l696_69697

def f (x : ℝ) : ℝ := x - 2

theorem graph_of_f_does_not_pass_through_second_quadrant :
  ¬ ∃ x y : ℝ, y = f x ∧ x < 0 ∧ y > 0 :=
sorry

end NUMINAMATH_GPT_graph_of_f_does_not_pass_through_second_quadrant_l696_69697


namespace NUMINAMATH_GPT_sum_of_altitudes_is_less_than_perimeter_l696_69696

theorem sum_of_altitudes_is_less_than_perimeter 
  (a b c h_a h_b h_c : ℝ) 
  (h_a_le_b : h_a ≤ b) 
  (h_b_le_c : h_b ≤ c) 
  (h_c_le_a : h_c ≤ a) 
  (strict_inequality : h_a < b ∨ h_b < c ∨ h_c < a) : h_a + h_b + h_c < a + b + c := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_altitudes_is_less_than_perimeter_l696_69696


namespace NUMINAMATH_GPT_math_proof_problem_l696_69632

noncomputable def discriminant (a : ℝ) : ℝ := a^2 - 4 * a + 2

def is_real_roots (a : ℝ) : Prop := discriminant a ≥ 0

def solution_set_a : Set ℝ := { a | is_real_roots a ∧ (a ≤ 2 - Real.sqrt 2 ∨ a ≥ 2 + Real.sqrt 2) }

def f (a : ℝ) : ℝ := -3 * a^2 + 16 * a - 8

def inequality_m (m t : ℝ) : Prop := m^2 + t * m + 4 * Real.sqrt 2 + 6 ≥ f (2 + Real.sqrt 2)

theorem math_proof_problem :
  (∀ a ∈ solution_set_a, ∃ m : ℝ, ∀ t ∈ Set.Icc (-1 : ℝ) (1 : ℝ), inequality_m m t) ∧
  (∀ m t, inequality_m m t → m ≤ -1 ∨ m = 0 ∨ m ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_math_proof_problem_l696_69632


namespace NUMINAMATH_GPT_total_population_of_cities_l696_69695

theorem total_population_of_cities 
    (number_of_cities : ℕ) 
    (average_population : ℕ) 
    (h1 : number_of_cities = 25) 
    (h2 : average_population = (5200 + 5700) / 2) : 
    number_of_cities * average_population = 136250 := by 
    sorry

end NUMINAMATH_GPT_total_population_of_cities_l696_69695
