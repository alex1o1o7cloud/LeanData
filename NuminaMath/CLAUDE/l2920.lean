import Mathlib

namespace NUMINAMATH_CALUDE_annie_televisions_correct_l2920_292077

/-- The number of televisions Annie bought at a liquidation sale -/
def num_televisions : ℕ := 5

/-- The cost of each television -/
def television_cost : ℕ := 50

/-- The number of figurines Annie bought -/
def num_figurines : ℕ := 10

/-- The cost of each figurine -/
def figurine_cost : ℕ := 1

/-- The total amount Annie spent -/
def total_spent : ℕ := 260

/-- Theorem stating that the number of televisions Annie bought is correct -/
theorem annie_televisions_correct : 
  num_televisions * television_cost + num_figurines * figurine_cost = total_spent :=
by sorry

end NUMINAMATH_CALUDE_annie_televisions_correct_l2920_292077


namespace NUMINAMATH_CALUDE_conical_cylinder_volume_l2920_292098

/-- The volume of a conical cylinder with base radius 3 cm and slant height 5 cm is 12π cm³ -/
theorem conical_cylinder_volume : 
  ∀ (r h s : ℝ), 
  r = 3 → s = 5 → h^2 + r^2 = s^2 →
  (1/3) * π * r^2 * h = 12 * π := by
sorry

end NUMINAMATH_CALUDE_conical_cylinder_volume_l2920_292098


namespace NUMINAMATH_CALUDE_abs_frac_gt_three_iff_x_in_intervals_l2920_292030

theorem abs_frac_gt_three_iff_x_in_intervals (x : ℝ) :
  x ≠ 2 →
  (|(3 * x - 2) / (x - 2)| > 3) ↔ (x > 4/3 ∧ x < 2) ∨ x > 2 :=
by sorry

end NUMINAMATH_CALUDE_abs_frac_gt_three_iff_x_in_intervals_l2920_292030


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_l2920_292035

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x, p x) ↔ ∀ x, ¬ p x :=
by sorry

theorem negation_of_inequality : 
  (¬ ∃ x : ℝ, 2^x ≥ 2*x + 1) ↔ (∀ x : ℝ, 2^x < 2*x + 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_l2920_292035


namespace NUMINAMATH_CALUDE_smallest_degree_for_horizontal_asymptote_l2920_292075

/-- 
Given a rational function f(x) = (5x^7 + 4x^4 - 3x + 2) / q(x),
prove that the smallest degree of q(x) for f(x) to have a horizontal asymptote is 7.
-/
theorem smallest_degree_for_horizontal_asymptote 
  (q : ℝ → ℝ) -- q is a real-valued function of a real variable
  (f : ℝ → ℝ) -- f is the rational function
  (hf : ∀ x, f x = (5*x^7 + 4*x^4 - 3*x + 2) / q x) -- definition of f
  : (∃ L : ℝ, ∀ ε > 0, ∃ M : ℝ, ∀ x, abs x > M → abs (f x - L) < ε) ↔ 
    (∃ n : ℕ, n ≥ 7 ∧ ∀ x, abs (q x) ≤ abs x^n + 1 ∧ abs x^n ≤ abs (q x) + 1) :=
sorry

end NUMINAMATH_CALUDE_smallest_degree_for_horizontal_asymptote_l2920_292075


namespace NUMINAMATH_CALUDE_largest_variable_l2920_292011

theorem largest_variable (a b c d : ℝ) (h : a - 2 = b + 3 ∧ a - 2 = c - 4 ∧ a - 2 = d + 5) :
  c ≥ a ∧ c ≥ b ∧ c ≥ d :=
by sorry

end NUMINAMATH_CALUDE_largest_variable_l2920_292011


namespace NUMINAMATH_CALUDE_unique_perfect_square_in_range_l2920_292089

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y * y

theorem unique_perfect_square_in_range :
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 14 →
    (is_perfect_square (n.factorial * (n + 1).factorial / 3) ↔ n = 11) :=
by sorry

end NUMINAMATH_CALUDE_unique_perfect_square_in_range_l2920_292089


namespace NUMINAMATH_CALUDE_square_of_negative_sum_l2920_292036

theorem square_of_negative_sum (a b : ℝ) : (-a - b)^2 = a^2 + 2*a*b + b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_sum_l2920_292036


namespace NUMINAMATH_CALUDE_appliance_cost_after_discount_l2920_292051

/-- Calculates the total cost of a washing machine and dryer after applying a discount -/
theorem appliance_cost_after_discount
  (washing_machine_cost : ℝ)
  (dryer_cost_difference : ℝ)
  (discount_percentage : ℝ)
  (h1 : washing_machine_cost = 100)
  (h2 : dryer_cost_difference = 30)
  (h3 : discount_percentage = 0.1) :
  let dryer_cost := washing_machine_cost - dryer_cost_difference
  let total_cost := washing_machine_cost + dryer_cost
  let discount_amount := discount_percentage * total_cost
  washing_machine_cost + dryer_cost - discount_amount = 153 := by
sorry

end NUMINAMATH_CALUDE_appliance_cost_after_discount_l2920_292051


namespace NUMINAMATH_CALUDE_parallelogram_area_l2920_292091

def vector_a : Fin 2 → ℝ := ![6, -8]
def vector_b : Fin 2 → ℝ := ![15, 4]

theorem parallelogram_area : 
  |vector_a 0 * vector_b 1 - vector_a 1 * vector_b 0| = 144 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2920_292091


namespace NUMINAMATH_CALUDE_correct_operation_l2920_292067

theorem correct_operation (x y : ℝ) : y * x - 2 * x * y = -x * y := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l2920_292067


namespace NUMINAMATH_CALUDE_greatest_common_piece_length_l2920_292054

theorem greatest_common_piece_length : Nat.gcd 42 (Nat.gcd 63 84) = 21 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_piece_length_l2920_292054


namespace NUMINAMATH_CALUDE_alex_sandwiches_l2920_292062

/-- The number of different sandwiches Alex can make -/
def num_sandwiches (num_meats : ℕ) (num_cheeses : ℕ) (num_breads : ℕ) : ℕ :=
  num_meats * (1 + num_cheeses + (num_cheeses.choose 2)) * num_breads

/-- Theorem stating the number of different sandwiches Alex can make -/
theorem alex_sandwiches :
  num_sandwiches 12 11 3 = 2412 := by sorry

end NUMINAMATH_CALUDE_alex_sandwiches_l2920_292062


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l2920_292090

theorem perfect_square_trinomial (x : ℝ) : 
  (x + 9)^2 = x^2 + 18*x + 81 ∧ 
  ∃ (a b : ℝ), (x + 9)^2 = a^2 + 2*a*b + b^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l2920_292090


namespace NUMINAMATH_CALUDE_scale_division_l2920_292071

/-- Represents the length of a scale in inches -/
def scale_length : ℕ := 10 * 12 + 5

/-- The number of parts the scale is divided into -/
def num_parts : ℕ := 5

/-- Calculates the length of each part when the scale is divided equally -/
def part_length : ℕ := scale_length / num_parts

/-- Theorem stating that each part of the scale is 25 inches long -/
theorem scale_division :
  part_length = 25 := by sorry

end NUMINAMATH_CALUDE_scale_division_l2920_292071


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2920_292097

theorem complex_modulus_problem (z : ℂ) (h : z * (Complex.I + 1) = Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2920_292097


namespace NUMINAMATH_CALUDE_three_numbers_sum_l2920_292082

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b → b ≤ c → 
  b = 7 → 
  (a + b + c) / 3 = a + 8 → 
  (a + b + c) / 3 = c - 20 → 
  a + b + c = 57 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l2920_292082


namespace NUMINAMATH_CALUDE_max_tetrahedron_volume_cube_sphere_l2920_292064

/-- The maximum volume of a tetrahedron formed by a point on the circumscribed sphere
    of a cube and one face of the cube, given the cube's edge length. -/
theorem max_tetrahedron_volume_cube_sphere (edge_length : ℝ) (h : edge_length = 2) :
  let sphere_radius : ℝ := Real.sqrt 3 * edge_length / 2
  let max_height : ℝ := sphere_radius + edge_length / 2
  let base_area : ℝ := edge_length ^ 2
  ∃ (volume : ℝ), volume = base_area * max_height / 3 ∧ 
                  volume = (4 * (1 + Real.sqrt 3)) / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_tetrahedron_volume_cube_sphere_l2920_292064


namespace NUMINAMATH_CALUDE_sum_of_positive_numbers_l2920_292061

theorem sum_of_positive_numbers (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x + y + x * y = 8)
  (eq2 : y + z + y * z = 15)
  (eq3 : z + x + z * x = 35) :
  x + y + z + x * y = 15 := by sorry

end NUMINAMATH_CALUDE_sum_of_positive_numbers_l2920_292061


namespace NUMINAMATH_CALUDE_solution_exists_l2920_292032

theorem solution_exists : ∃ (v : ℝ), 4 * v^2 = 144 ∧ v = 6 := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_l2920_292032


namespace NUMINAMATH_CALUDE_inequality_proof_l2920_292043

theorem inequality_proof (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (hy₁ : x₁ * y₁ > z₁^2) (hy₂ : x₂ * y₂ > z₂^2) : 
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) ≤ 
  1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2920_292043


namespace NUMINAMATH_CALUDE_compute_expression_l2920_292023

theorem compute_expression : 2 * ((3 + 7)^2 + (3^2 + 7^2)) = 316 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l2920_292023


namespace NUMINAMATH_CALUDE_curve_equation_l2920_292000

noncomputable def x (t : ℝ) : ℝ := 3 * Real.cos t - Real.sin t
noncomputable def y (t : ℝ) : ℝ := 5 * Real.sin t

theorem curve_equation :
  ∃ (a b c : ℝ), ∀ (t : ℝ),
    a * (x t)^2 + b * (x t) * (y t) + c * (y t)^2 = 1 ∧
    a = 1/9 ∧ b = 2/45 ∧ c = 4/45 := by
  sorry

end NUMINAMATH_CALUDE_curve_equation_l2920_292000


namespace NUMINAMATH_CALUDE_not_p_or_q_l2920_292018

-- Define proposition p
def p : Prop := ∀ x : ℝ, x^2 + x - 1 > 0

-- Define proposition q
def q : Prop := ∃ x : ℝ, (2 : ℝ)^x > (3 : ℝ)^x

-- Theorem to prove
theorem not_p_or_q : (¬p) ∨ q := by sorry

end NUMINAMATH_CALUDE_not_p_or_q_l2920_292018


namespace NUMINAMATH_CALUDE_parabola_standard_form_l2920_292044

/-- A parabola with axis of symmetry x = 1 -/
structure Parabola where
  axis_of_symmetry : ℝ
  h_axis : axis_of_symmetry = 1

/-- The standard form of a parabola equation y^2 = ax -/
def standard_form (a : ℝ) (x y : ℝ) : Prop :=
  y^2 = a * x

/-- Theorem stating that the standard form of the parabola with axis of symmetry x = 1 is y^2 = -4x -/
theorem parabola_standard_form (p : Parabola) :
  ∃ a : ℝ, (∀ x y : ℝ, standard_form a x y) ∧ a = -4 :=
sorry

end NUMINAMATH_CALUDE_parabola_standard_form_l2920_292044


namespace NUMINAMATH_CALUDE_min_value_expression_l2920_292068

theorem min_value_expression (a b : ℝ) (ha : a > 1) (hb : b > 0) (hab : a + b = 2) :
  (∀ x y, x > 1 ∧ y > 0 ∧ x + y = 2 → 1 / (a - 1) + 1 / (2 * b) ≤ 1 / (x - 1) + 1 / (2 * y)) ∧
  1 / (a - 1) + 1 / (2 * b) = 3 / 2 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2920_292068


namespace NUMINAMATH_CALUDE_range_of_m_for_nonempty_solution_l2920_292055

theorem range_of_m_for_nonempty_solution (m : ℝ) : 
  (∃ x : ℝ, |x - 1| + |x + m| ≤ 4) → m ∈ Set.Icc (-5) 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_for_nonempty_solution_l2920_292055


namespace NUMINAMATH_CALUDE_jason_picked_ten_plums_l2920_292031

def alyssa_plums : ℕ := 17
def total_plums : ℕ := 27

def jason_plums : ℕ := total_plums - alyssa_plums

theorem jason_picked_ten_plums : jason_plums = 10 := by
  sorry

end NUMINAMATH_CALUDE_jason_picked_ten_plums_l2920_292031


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_range_l2920_292072

theorem ellipse_parabola_intersection_range (a : ℝ) :
  (∃ x y : ℝ, x^2 + 4*(y - a)^2 = 4 ∧ x^2 = 2*y) →
  -1 ≤ a ∧ a ≤ 17/8 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_range_l2920_292072


namespace NUMINAMATH_CALUDE_new_student_weight_l2920_292040

/-- Given 5 students, if replacing a 92 kg student with a new student
    causes the average weight to decrease by 4 kg,
    then the new student's weight is 72 kg. -/
theorem new_student_weight
  (n : Nat)
  (old_weight : Nat)
  (weight_decrease : Nat)
  (h1 : n = 5)
  (h2 : old_weight = 92)
  (h3 : weight_decrease = 4)
  : n * weight_decrease = old_weight - (old_weight - n * weight_decrease) :=
by
  sorry

#check new_student_weight

end NUMINAMATH_CALUDE_new_student_weight_l2920_292040


namespace NUMINAMATH_CALUDE_airplane_shot_down_probability_l2920_292052

def probability_airplane_shot_down : ℝ :=
  let p_A : ℝ := 0.4
  let p_B : ℝ := 0.5
  let p_C : ℝ := 0.8
  let p_one_hit : ℝ := 0.4
  let p_two_hit : ℝ := 0.7
  let p_three_hit : ℝ := 1

  let p_A_miss : ℝ := 1 - p_A
  let p_B_miss : ℝ := 1 - p_B
  let p_C_miss : ℝ := 1 - p_C

  let p_one_person_hits : ℝ := 
    (p_A * p_B_miss * p_C_miss + p_A_miss * p_B * p_C_miss + p_A_miss * p_B_miss * p_C) * p_one_hit

  let p_two_people_hit : ℝ := 
    (p_A * p_B * p_C_miss + p_A * p_B_miss * p_C + p_A_miss * p_B * p_C) * p_two_hit

  let p_all_hit : ℝ := p_A * p_B * p_C * p_three_hit

  p_one_person_hits + p_two_people_hit + p_all_hit

theorem airplane_shot_down_probability : 
  probability_airplane_shot_down = 0.604 := by sorry

end NUMINAMATH_CALUDE_airplane_shot_down_probability_l2920_292052


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l2920_292099

-- Define the logarithm functions
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem log_sum_equals_two : lg 0.01 + log2 16 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l2920_292099


namespace NUMINAMATH_CALUDE_max_value_and_min_sum_of_squares_l2920_292010

noncomputable def f (x a b : ℝ) : ℝ := |x - a| - |x + 2*b|

theorem max_value_and_min_sum_of_squares
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∀ x, f x a b ≤ a + 2*b) ∧
  (a + 2*b = 1 → ∃ (a₀ b₀ : ℝ), a₀^2 + 4*b₀^2 = 1/2 ∧ ∀ a' b', a'^2 + 4*b'^2 ≥ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_and_min_sum_of_squares_l2920_292010


namespace NUMINAMATH_CALUDE_m_range_l2920_292066

theorem m_range (x y m : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_eq : 2/x + 1/y = 1) (h_ineq : x + 2*y > m^2 + 2*m) : 
  -4 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l2920_292066


namespace NUMINAMATH_CALUDE_sin_90_plus_alpha_eq_neg_half_l2920_292074

/-- Given that α is an angle in the second quadrant and tan α = -√3, prove that sin(90° + α) = -1/2 -/
theorem sin_90_plus_alpha_eq_neg_half (α : Real) 
  (h1 : π/2 < α ∧ α < π)  -- α is in the second quadrant
  (h2 : Real.tan α = -Real.sqrt 3) : -- tan α = -√3
  Real.sin (π/2 + α) = -1/2 := by sorry

end NUMINAMATH_CALUDE_sin_90_plus_alpha_eq_neg_half_l2920_292074


namespace NUMINAMATH_CALUDE_dot_product_of_parallel_vectors_l2920_292003

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem dot_product_of_parallel_vectors :
  let p : ℝ × ℝ := (1, -2)
  let q : ℝ × ℝ := (x, 4)
  ∀ x : ℝ, parallel p q → p.1 * q.1 + p.2 * q.2 = -10 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_of_parallel_vectors_l2920_292003


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2920_292092

theorem rationalize_denominator : 
  (30 : ℝ) / Real.sqrt 15 = 2 * Real.sqrt 15 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2920_292092


namespace NUMINAMATH_CALUDE_sum_mod_nine_l2920_292070

theorem sum_mod_nine : (3612 + 3613 + 3614 + 3615 + 3616) % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_nine_l2920_292070


namespace NUMINAMATH_CALUDE_travel_distances_l2920_292086

-- Define the given constants
def train_speed : ℚ := 100
def car_speed_ratio : ℚ := 2/3
def bicycle_speed_ratio : ℚ := 1/5
def travel_time : ℚ := 1/2  -- 30 minutes in hours

-- Define the theorem
theorem travel_distances :
  let car_distance := train_speed * car_speed_ratio * travel_time
  let bicycle_distance := train_speed * bicycle_speed_ratio * travel_time
  car_distance = 100/3 ∧ bicycle_distance = 10 := by sorry

end NUMINAMATH_CALUDE_travel_distances_l2920_292086


namespace NUMINAMATH_CALUDE_expression_evaluation_l2920_292033

theorem expression_evaluation (a : ℚ) (h : a = 4/3) :
  (6 * a^2 - 8 * a + 3) * (3 * a - 4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2920_292033


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l2920_292016

/-- An ellipse with foci at (15, 30) and (15, 90) that is tangent to the y-axis has a major axis of length 30√5 -/
theorem ellipse_major_axis_length : 
  ∀ (E : Set (ℝ × ℝ)) (F₁ F₂ Y : ℝ × ℝ),
  F₁ = (15, 30) →
  F₂ = (15, 90) →
  Y.1 = 0 →
  (∀ p ∈ E, dist p F₁ + dist p F₂ = dist Y F₁ + dist Y F₂) →
  (∀ q : ℝ × ℝ, q.1 = 0 → dist q F₁ + dist q F₂ ≥ dist Y F₁ + dist Y F₂) →
  dist Y F₁ + dist Y F₂ = 30 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l2920_292016


namespace NUMINAMATH_CALUDE_tiles_required_to_cover_floor_l2920_292059

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℚ
  width : ℚ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℚ :=
  d.length * d.width

/-- Converts inches to feet -/
def inchesToFeet (inches : ℚ) : ℚ :=
  inches / 12

/-- The dimensions of the floor in feet -/
def floorDimensions : Dimensions :=
  { length := 12, width := 9 }

/-- The dimensions of a tile in inches -/
def tileDimensions : Dimensions :=
  { length := 8, width := 6 }

/-- Theorem stating that 324 tiles are required to cover the floor -/
theorem tiles_required_to_cover_floor :
  (area floorDimensions) / (area { length := inchesToFeet tileDimensions.length,
                                   width := inchesToFeet tileDimensions.width }) = 324 := by
  sorry

end NUMINAMATH_CALUDE_tiles_required_to_cover_floor_l2920_292059


namespace NUMINAMATH_CALUDE_cubic_difference_999_l2920_292015

theorem cubic_difference_999 : 
  ∀ m n : ℕ+, m^3 - n^3 = 999 ↔ (m = 10 ∧ n = 1) ∨ (m = 12 ∧ n = 9) := by
sorry

end NUMINAMATH_CALUDE_cubic_difference_999_l2920_292015


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l2920_292088

theorem perfect_square_trinomial (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - k*x + 9 = (x - a)^2) → (k = 6 ∨ k = -6) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l2920_292088


namespace NUMINAMATH_CALUDE_skill_testing_question_l2920_292021

theorem skill_testing_question : 5 * (10 - 6) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_skill_testing_question_l2920_292021


namespace NUMINAMATH_CALUDE_sixth_face_configuration_l2920_292076

structure Cube where
  size : Nat
  black_cubes : Nat
  white_cubes : Nat

structure Face where
  center_white : Nat
  edge_white : Nat
  corner_white : Nat

def valid_face (f : Face) : Prop :=
  f.center_white = 1 ∧ f.edge_white = 2 ∧ f.corner_white = 1

def cube_configuration (c : Cube) (known_faces : List Face) : Prop :=
  c.size = 3 ∧
  c.black_cubes = 15 ∧
  c.white_cubes = 12 ∧
  known_faces.length = 5

theorem sixth_face_configuration
  (c : Cube)
  (known_faces : List Face)
  (h_config : cube_configuration c known_faces) :
  ∃ (sixth_face : Face), valid_face sixth_face :=
by sorry

end NUMINAMATH_CALUDE_sixth_face_configuration_l2920_292076


namespace NUMINAMATH_CALUDE_framed_rectangle_dimensions_l2920_292046

/-- A rectangle on a grid with a one-cell-wide frame around it. -/
structure FramedRectangle where
  length : ℕ
  width : ℕ

/-- The area of the inner rectangle. -/
def FramedRectangle.inner_area (r : FramedRectangle) : ℕ :=
  r.length * r.width

/-- The area of the frame around the rectangle. -/
def FramedRectangle.frame_area (r : FramedRectangle) : ℕ :=
  (r.length + 2) * (r.width + 2) - r.length * r.width

/-- The property that the inner area equals the frame area. -/
def FramedRectangle.area_equality (r : FramedRectangle) : Prop :=
  r.inner_area = r.frame_area

/-- The theorem stating that if the inner area equals the frame area,
    then the dimensions are either 3 × 10 or 4 × 6. -/
theorem framed_rectangle_dimensions (r : FramedRectangle) :
  r.area_equality →
  ((r.length = 3 ∧ r.width = 10) ∨ (r.length = 4 ∧ r.width = 6) ∨
   (r.length = 10 ∧ r.width = 3) ∨ (r.length = 6 ∧ r.width = 4)) :=
by sorry

end NUMINAMATH_CALUDE_framed_rectangle_dimensions_l2920_292046


namespace NUMINAMATH_CALUDE_hotel_charges_l2920_292078

-- Define the charges for each hotel
variable (P R G S T : ℝ)

-- Define the relationships between the charges
axiom p_r : P = 0.75 * R
axiom p_g : P = 0.90 * G
axiom s_r : S = 1.15 * R
axiom t_g : T = 0.80 * G

-- Theorem to prove
theorem hotel_charges :
  S = 1.5333 * P ∧ 
  T = 0.8888 * P ∧ 
  (R - G) / G = 0.18 := by sorry

end NUMINAMATH_CALUDE_hotel_charges_l2920_292078


namespace NUMINAMATH_CALUDE_triangle_side_equality_l2920_292001

-- Define the triangle ABC
structure Triangle (α : Type) where
  A : α
  B : α
  C : α

-- Define the sides of the triangle
def side_AB (a b : ℤ) : ℤ := b^2 - 1
def side_BC (a b : ℤ) : ℤ := a^2
def side_CA (a b : ℤ) : ℤ := 2*a

-- State the theorem
theorem triangle_side_equality (a b : ℤ) (ABC : Triangle ℤ) :
  a > 1 ∧ b > 1 ∧ 
  side_AB a b = b^2 - 1 ∧
  side_BC a b = a^2 ∧
  side_CA a b = 2*a →
  b - a = 0 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_equality_l2920_292001


namespace NUMINAMATH_CALUDE_m_range_l2920_292024

def f (x : ℝ) := x^3 + x

theorem m_range (m : ℝ) :
  (∀ θ : ℝ, 0 < θ ∧ θ < π/2 → f (m * Real.sin θ) + f (1 - m) > 0) →
  m ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l2920_292024


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l2920_292005

-- System 1
theorem system_one_solution (x z : ℚ) : 
  (3 * x - 5 * z = 6 ∧ x + 4 * z = -15) ↔ (x = -3 ∧ z = -3) := by sorry

-- System 2
theorem system_two_solution (x y : ℚ) :
  ((2 * x - 1) / 5 + (3 * y - 2) / 4 = 2 ∧ 
   (3 * x + 1) / 5 - (3 * y + 2) / 4 = 0) ↔ (x = 3 ∧ y = 2) := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l2920_292005


namespace NUMINAMATH_CALUDE_motel_monthly_charge_l2920_292026

theorem motel_monthly_charge 
  (weeks_per_month : ℕ)
  (num_months : ℕ)
  (weekly_rate : ℕ)
  (total_savings : ℕ)
  (h1 : weeks_per_month = 4)
  (h2 : num_months = 3)
  (h3 : weekly_rate = 280)
  (h4 : total_savings = 360) :
  (num_months * weeks_per_month * weekly_rate - total_savings) / num_months = 1000 := by
  sorry

end NUMINAMATH_CALUDE_motel_monthly_charge_l2920_292026


namespace NUMINAMATH_CALUDE_big_eighteen_game_count_l2920_292039

/-- Calculates the total number of games in a basketball conference. -/
def total_conference_games (num_divisions : ℕ) (teams_per_division : ℕ) 
  (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let total_teams := num_divisions * teams_per_division
  let intra_division_total := num_divisions * (teams_per_division * (teams_per_division - 1) / 2) * intra_division_games
  let inter_division_total := (total_teams * (total_teams - teams_per_division) * inter_division_games) / 2
  intra_division_total + inter_division_total

/-- The Big Eighteen Basketball Conference game count theorem -/
theorem big_eighteen_game_count : 
  total_conference_games 3 6 3 2 = 486 := by
  sorry

end NUMINAMATH_CALUDE_big_eighteen_game_count_l2920_292039


namespace NUMINAMATH_CALUDE_maintenance_team_schedule_l2920_292063

theorem maintenance_team_schedule : Nat.lcm 5 (Nat.lcm 8 (Nat.lcm 9 11)) = 3960 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_team_schedule_l2920_292063


namespace NUMINAMATH_CALUDE_congruence_mod_nine_l2920_292094

theorem congruence_mod_nine : ∃! n : ℤ, 0 ≤ n ∧ n < 9 ∧ -1234 ≡ n [ZMOD 9] ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_congruence_mod_nine_l2920_292094


namespace NUMINAMATH_CALUDE_grid_and_circles_area_sum_l2920_292081

/-- The side length of each small square in the grid -/
def smallSquareSide : ℝ := 3

/-- The number of rows in the grid -/
def gridRows : ℕ := 4

/-- The number of columns in the grid -/
def gridColumns : ℕ := 4

/-- The radius of the large circle -/
def largeCircleRadius : ℝ := 1.5 * smallSquareSide

/-- The radius of each small circle -/
def smallCircleRadius : ℝ := 0.5 * smallSquareSide

/-- The number of small circles -/
def numSmallCircles : ℕ := 3

/-- Theorem: The sum of the total grid area and the total area of the circles is 171 square cm -/
theorem grid_and_circles_area_sum : 
  (gridRows * gridColumns * smallSquareSide^2) + 
  (π * largeCircleRadius^2 + π * numSmallCircles * smallCircleRadius^2) = 171 := by
  sorry

end NUMINAMATH_CALUDE_grid_and_circles_area_sum_l2920_292081


namespace NUMINAMATH_CALUDE_gabrielle_robins_count_l2920_292038

/-- The number of birds Gabrielle saw -/
def gabrielle_total : ℕ := sorry

/-- The number of robins Gabrielle saw -/
def gabrielle_robins : ℕ := sorry

/-- The number of cardinals Gabrielle saw -/
def gabrielle_cardinals : ℕ := 4

/-- The number of blue jays Gabrielle saw -/
def gabrielle_blue_jays : ℕ := 3

/-- The number of birds Chase saw -/
def chase_total : ℕ := 10

/-- The number of robins Chase saw -/
def chase_robins : ℕ := 2

/-- The number of blue jays Chase saw -/
def chase_blue_jays : ℕ := 3

/-- The number of cardinals Chase saw -/
def chase_cardinals : ℕ := 5

theorem gabrielle_robins_count :
  gabrielle_total = chase_total + chase_total / 5 ∧
  gabrielle_total = gabrielle_robins + gabrielle_cardinals + gabrielle_blue_jays ∧
  gabrielle_robins = 5 := by sorry

end NUMINAMATH_CALUDE_gabrielle_robins_count_l2920_292038


namespace NUMINAMATH_CALUDE_spontaneous_low_temp_signs_l2920_292058

/-- Represents the change in enthalpy -/
def ΔH : ℝ := sorry

/-- Represents the change in entropy -/
def ΔS : ℝ := sorry

/-- Represents temperature -/
def T : ℝ := sorry

/-- Represents the change in Gibbs free energy -/
def ΔG (T : ℝ) : ℝ := ΔH - T * ΔS

/-- Represents that the reaction is spontaneous -/
def is_spontaneous (T : ℝ) : Prop := ΔG T < 0

/-- Represents that the reaction is spontaneous only at low temperatures -/
def spontaneous_at_low_temp : Prop :=
  ∃ T₀ > 0, ∀ T, 0 < T → T < T₀ → is_spontaneous T

theorem spontaneous_low_temp_signs :
  spontaneous_at_low_temp → ΔH < 0 ∧ ΔS < 0 := by
  sorry

end NUMINAMATH_CALUDE_spontaneous_low_temp_signs_l2920_292058


namespace NUMINAMATH_CALUDE_tim_has_203_balloons_l2920_292034

/-- The number of violet balloons Dan has -/
def dan_balloons : ℕ := 29

/-- The factor by which Tim's balloons exceed Dan's -/
def tim_factor : ℕ := 7

/-- The number of violet balloons Tim has -/
def tim_balloons : ℕ := dan_balloons * tim_factor

/-- Theorem: Tim has 203 violet balloons -/
theorem tim_has_203_balloons : tim_balloons = 203 := by
  sorry

end NUMINAMATH_CALUDE_tim_has_203_balloons_l2920_292034


namespace NUMINAMATH_CALUDE_hikmet_seventh_l2920_292020

/-- Represents the position of a racer in a 12-person race -/
def Position := Fin 12

/-- The race results -/
structure RaceResult where
  david : Position
  hikmet : Position
  jack : Position
  marta : Position
  rand : Position
  todd : Position

/-- Conditions of the race -/
def race_conditions (result : RaceResult) : Prop :=
  result.marta.val = result.jack.val + 3 ∧
  result.jack.val = result.todd.val + 1 ∧
  result.todd.val = result.rand.val + 3 ∧
  result.rand.val + 5 = result.hikmet.val ∧
  result.hikmet.val + 4 = result.david.val ∧
  result.marta.val = 9

/-- Theorem stating that Hikmet finished in 7th place -/
theorem hikmet_seventh (result : RaceResult) 
  (h : race_conditions result) : result.hikmet.val = 7 := by
  sorry

end NUMINAMATH_CALUDE_hikmet_seventh_l2920_292020


namespace NUMINAMATH_CALUDE_orchard_expansion_l2920_292008

theorem orchard_expansion (n : ℕ) (h1 : n^2 + 146 = 7890) (h2 : (n + 1)^2 = n^2 + 31 + 146) : (n + 1)^2 = 7921 := by
  sorry

end NUMINAMATH_CALUDE_orchard_expansion_l2920_292008


namespace NUMINAMATH_CALUDE_inequality_proof_l2920_292096

theorem inequality_proof (x y : ℝ) (h : x^12 + y^12 ≤ 2) :
  x^2 + y^2 + x^2 * y^2 ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2920_292096


namespace NUMINAMATH_CALUDE_polynomial_evaluation_and_coefficient_sum_l2920_292057

theorem polynomial_evaluation_and_coefficient_sum (d : ℝ) (h : d ≠ 0) :
  (10*d + 16 + 17*d^2 + 3*d^3) + (5*d + 4 + 2*d^2 + 2*d^3) = 5*d^3 + 19*d^2 + 15*d + 20 ∧
  5 + 19 + 15 + 20 = 59 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_and_coefficient_sum_l2920_292057


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2920_292045

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 = 450) →
  (a 2 + a 8 = 180) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2920_292045


namespace NUMINAMATH_CALUDE_probability_second_odd_given_first_odd_l2920_292025

theorem probability_second_odd_given_first_odd (n : ℕ) (odds evens : ℕ) 
  (h1 : n = odds + evens)
  (h2 : n = 9)
  (h3 : odds = 5)
  (h4 : evens = 4) :
  (odds - 1) / (n - 1) = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_probability_second_odd_given_first_odd_l2920_292025


namespace NUMINAMATH_CALUDE_fair_distribution_result_l2920_292037

/-- Represents the fair distribution of talers in the bread-sharing scenario -/
def fair_distribution (loaves1 loaves2 : ℕ) (total_talers : ℕ) : ℕ × ℕ :=
  let total_loaves := loaves1 + loaves2
  let loaves_per_person := total_loaves / 3
  let talers_per_loaf := total_talers / loaves_per_person
  let remaining_loaves1 := loaves1 - loaves_per_person
  let remaining_loaves2 := loaves2 - loaves_per_person
  let talers1 := remaining_loaves1 * talers_per_loaf
  let talers2 := remaining_loaves2 * talers_per_loaf
  (talers1, talers2)

/-- The fair distribution of talers in the given scenario is (1, 7) -/
theorem fair_distribution_result :
  fair_distribution 3 5 8 = (1, 7) := by
  sorry

end NUMINAMATH_CALUDE_fair_distribution_result_l2920_292037


namespace NUMINAMATH_CALUDE_unique_parallel_line_l2920_292006

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relationships between geometric objects
variable (parallel : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (in_plane : Point → Plane → Prop)
variable (passes_through : Line → Point → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- State the theorem
theorem unique_parallel_line 
  (α β : Plane) (a : Line) (B : Point)
  (h1 : parallel α β)
  (h2 : contains α a)
  (h3 : in_plane B β) :
  ∃! l : Line, line_in_plane l β ∧ passes_through l B ∧ line_parallel l a :=
sorry

end NUMINAMATH_CALUDE_unique_parallel_line_l2920_292006


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l2920_292084

theorem complex_expression_simplification :
  (-5 + 3 * Complex.I) - (2 - 7 * Complex.I) * 3 = -11 + 24 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l2920_292084


namespace NUMINAMATH_CALUDE_percentage_b_of_d_l2920_292027

theorem percentage_b_of_d (A B C D : ℝ) 
  (hB : B = 1.71 * A) 
  (hC : C = 1.80 * A) 
  (hD : D = 1.90 * B) : 
  ∃ ε > 0, |100 * B / D - 52.63| < ε :=
sorry

end NUMINAMATH_CALUDE_percentage_b_of_d_l2920_292027


namespace NUMINAMATH_CALUDE_intersection_k_values_eq_four_and_fourteen_l2920_292017

/-- The set of possible k values for which |z - 4| = 3|z + 4| and |z| = k intersect at exactly one point. -/
def intersection_k_values : Set ℝ :=
  {k : ℝ | ∃! (z : ℂ), Complex.abs (z - 4) = 3 * Complex.abs (z + 4) ∧ Complex.abs z = k}

/-- Theorem stating that the intersection_k_values set contains only 4 and 14. -/
theorem intersection_k_values_eq_four_and_fourteen :
  intersection_k_values = {4, 14} := by
  sorry

end NUMINAMATH_CALUDE_intersection_k_values_eq_four_and_fourteen_l2920_292017


namespace NUMINAMATH_CALUDE_gravitational_force_at_distance_l2920_292042

/-- Gravitational force calculation -/
theorem gravitational_force_at_distance 
  (k : ℝ) -- Gravitational constant
  (d₁ d₂ : ℝ) -- Distances from Earth's center
  (f₁ : ℝ) -- Force at distance d₁
  (h₁ : d₁ > 0)
  (h₂ : d₂ > 0)
  (h₃ : f₁ > 0)
  (h₄ : k = f₁ * d₁^2) -- Force-distance relation at d₁
  (h₅ : d₁ = 4000) -- Distance to Earth's surface in miles
  (h₆ : f₁ = 500) -- Force at Earth's surface in Newtons
  (h₇ : d₂ = 40000) -- Distance to space station in miles
  : f₁ * (d₂ / d₁)^2 = 5 := by
  sorry

#check gravitational_force_at_distance

end NUMINAMATH_CALUDE_gravitational_force_at_distance_l2920_292042


namespace NUMINAMATH_CALUDE_hyperbola_foci_coordinates_l2920_292028

/-- Given a hyperbola with equation 5x^2 - 4y^2 + 60 = 0, its foci have coordinates (0, ±3√3) -/
theorem hyperbola_foci_coordinates :
  let hyperbola := fun (x y : ℝ) => 5 * x^2 - 4 * y^2 + 60
  ∃ (c : ℝ), c = 3 * Real.sqrt 3 ∧
    (∀ (x y : ℝ), hyperbola x y = 0 →
      (hyperbola 0 c = 0 ∧ hyperbola 0 (-c) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_coordinates_l2920_292028


namespace NUMINAMATH_CALUDE_rectangle_diagonal_ratio_l2920_292050

theorem rectangle_diagonal_ratio (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≤ b) :
  (a + b - Real.sqrt (a^2 + b^2) = b / 3) → (a / b = 5 / 12) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_ratio_l2920_292050


namespace NUMINAMATH_CALUDE_exists_positive_x_hash_equals_63_l2920_292047

/-- Definition of the # operation -/
def hash (a b : ℝ) : ℝ := a * b - b + b^2

/-- Theorem stating the existence of a positive real number x such that 3 # x = 63 -/
theorem exists_positive_x_hash_equals_63 : ∃ x : ℝ, x > 0 ∧ hash 3 x = 63 := by
  sorry

end NUMINAMATH_CALUDE_exists_positive_x_hash_equals_63_l2920_292047


namespace NUMINAMATH_CALUDE_farmer_remaining_apples_l2920_292022

def initial_apples : ℕ := 127
def apples_given_away : ℕ := 88

theorem farmer_remaining_apples : initial_apples - apples_given_away = 39 := by
  sorry

end NUMINAMATH_CALUDE_farmer_remaining_apples_l2920_292022


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2920_292073

theorem complex_number_quadrant (z : ℂ) (h : (2 + 3*I)*z = 1 + I) : 
  (z.re > 0) ∧ (z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2920_292073


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2920_292049

-- Define the repeating decimals
def repeating_decimal_1 : ℚ := 2/9
def repeating_decimal_2 : ℚ := 2/99

-- Theorem statement
theorem sum_of_repeating_decimals :
  repeating_decimal_1 + repeating_decimal_2 = 8/33 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2920_292049


namespace NUMINAMATH_CALUDE_percentage_with_neither_condition_l2920_292053

/-- Given a survey of teachers, calculate the percentage who have neither high blood pressure nor heart trouble. -/
theorem percentage_with_neither_condition
  (total : ℕ)
  (high_blood_pressure : ℕ)
  (heart_trouble : ℕ)
  (both : ℕ)
  (h1 : total = 150)
  (h2 : high_blood_pressure = 80)
  (h3 : heart_trouble = 60)
  (h4 : both = 30)
  : (total - (high_blood_pressure + heart_trouble - both)) / total * 100 = 800 / 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_with_neither_condition_l2920_292053


namespace NUMINAMATH_CALUDE_triangle_problem_l2920_292002

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : (2 * t.a - t.c) * Real.cos t.B = t.b * Real.cos t.C)
  (h2 : t.a = 3)
  (h3 : (1/2) * t.a * t.c * Real.sin t.B = (3 * Real.sqrt 3) / 2) :
  t.B = π/3 ∧ t.a * t.c * Real.cos (π - t.A) = -1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2920_292002


namespace NUMINAMATH_CALUDE_bike_average_speed_l2920_292093

theorem bike_average_speed (initial_reading final_reading : ℕ) (total_time : ℝ) :
  initial_reading = 2332 →
  final_reading = 2552 →
  total_time = 9 →
  (final_reading - initial_reading : ℝ) / total_time = 220 / 9 := by
  sorry

end NUMINAMATH_CALUDE_bike_average_speed_l2920_292093


namespace NUMINAMATH_CALUDE_triangle_properties_l2920_292014

open Real

theorem triangle_properties (a b c A B C : Real) :
  -- Given conditions
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (2 * Real.sqrt 3 * a * c * Real.sin B = a^2 + b^2 - c^2) →
  -- First part
  (C = π / 6) ∧
  -- Additional conditions for the second part
  (b * Real.sin (π - A) = a * Real.cos B) →
  (b = Real.sqrt 2) →
  -- Second part
  (1/2 * b * c * Real.sin A = (Real.sqrt 3 + 1) / 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2920_292014


namespace NUMINAMATH_CALUDE_quadratic_solution_l2920_292013

theorem quadratic_solution (b : ℝ) : 
  ((-9 : ℝ)^2 + b * (-9) - 36 = 0) → b = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_l2920_292013


namespace NUMINAMATH_CALUDE_cubic_root_sum_squares_l2920_292029

theorem cubic_root_sum_squares (p q r : ℝ) (x : ℝ → ℝ) :
  (∀ t, x t = 0 ↔ t^3 - p*t^2 + q*t - r = 0) →
  ∃ r s t, (x r = 0 ∧ x s = 0 ∧ x t = 0) ∧ 
           (r^2 + s^2 + t^2 = p^2 - 2*q) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_squares_l2920_292029


namespace NUMINAMATH_CALUDE_min_value_of_f_l2920_292041

-- Define the function f
def f (x m : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

-- State the theorem
theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y m ≤ f x m) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ f y m) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = -37) :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2920_292041


namespace NUMINAMATH_CALUDE_power_of_negative_one_product_l2920_292007

theorem power_of_negative_one_product (n : ℕ) : 
  ((-1 : ℤ) ^ n) * ((-1 : ℤ) ^ (2 * n + 1)) * ((-1 : ℤ) ^ (n + 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_one_product_l2920_292007


namespace NUMINAMATH_CALUDE_odell_kershaw_passing_l2920_292087

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ  -- speed in m/min
  radius : ℝ  -- radius of the lane in meters
  direction : ℤ  -- 1 for clockwise, -1 for counterclockwise

/-- Calculates the number of times two runners pass each other on a circular track -/
def passingCount (runner1 runner2 : Runner) (totalTime : ℝ) : ℕ :=
  sorry

theorem odell_kershaw_passing :
  let odell : Runner := { speed := 240, radius := 40, direction := 1 }
  let kershaw : Runner := { speed := 320, radius := 55, direction := -1 }
  let totalTime : ℝ := 40
  passingCount odell kershaw totalTime = 75 := by
  sorry

end NUMINAMATH_CALUDE_odell_kershaw_passing_l2920_292087


namespace NUMINAMATH_CALUDE_wall_width_is_100cm_l2920_292048

/-- Represents the dimensions of a brick in centimeters -/
structure BrickDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a wall in centimeters -/
structure WallDimensions where
  length : ℝ
  width : ℝ
  thickness : ℝ

/-- Calculates the volume of a brick given its dimensions -/
def brickVolume (d : BrickDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Calculates the volume of a wall given its dimensions -/
def wallVolume (d : WallDimensions) : ℝ :=
  d.length * d.width * d.thickness

/-- Theorem stating that the width of the wall is 100 cm -/
theorem wall_width_is_100cm
  (brick : BrickDimensions)
  (wall : WallDimensions)
  (h1 : brick.length = 25)
  (h2 : brick.width = 11)
  (h3 : brick.height = 6)
  (h4 : wall.length = 800)
  (h5 : wall.thickness = 5)
  (h6 : 242.42424242424244 * brickVolume brick = wallVolume wall) :
  wall.width = 100 := by
  sorry

end NUMINAMATH_CALUDE_wall_width_is_100cm_l2920_292048


namespace NUMINAMATH_CALUDE_product_of_smallest_primes_l2920_292079

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def smallest_one_digit_primes : (ℕ × ℕ) :=
  (2, 3)

def smallest_two_digit_prime : ℕ :=
  11

theorem product_of_smallest_primes :
  let (p1, p2) := smallest_one_digit_primes
  p1 * p2 * smallest_two_digit_prime = 66 ∧
  is_prime p1 ∧ is_prime p2 ∧ is_prime smallest_two_digit_prime ∧
  p1 < 10 ∧ p2 < 10 ∧ smallest_two_digit_prime ≥ 10 ∧ smallest_two_digit_prime < 100 :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_smallest_primes_l2920_292079


namespace NUMINAMATH_CALUDE_supplementary_angles_ratio_l2920_292095

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 →  -- The angles are supplementary (sum to 180°)
  a / b = 4 / 5 →  -- The ratio of the angles is 4:5
  b = 100 :=  -- The larger angle is 100°
by
  sorry

end NUMINAMATH_CALUDE_supplementary_angles_ratio_l2920_292095


namespace NUMINAMATH_CALUDE_negative_integer_product_l2920_292083

theorem negative_integer_product (a b : ℤ) : ∃ n : ℤ,
  n < 0 ∧
  n * a < 0 ∧
  -8 * b < 0 ∧
  n * a * (-8 * b) + a * b = 89 ∧
  n = -11 := by sorry

end NUMINAMATH_CALUDE_negative_integer_product_l2920_292083


namespace NUMINAMATH_CALUDE_three_collinear_points_same_color_l2920_292056

-- Define a color type
inductive Color
| Black
| White

-- Define a point as a pair of real number (position) and color
structure Point where
  position : ℝ
  color : Color

-- Define a function to check if three points are collinear with one in the middle
def areCollinearWithMiddle (p1 p2 p3 : Point) : Prop :=
  p2.position = (p1.position + p3.position) / 2

-- State the theorem
theorem three_collinear_points_same_color (points : Set Point) : 
  ∃ (p1 p2 p3 : Point), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ 
  p1.color = p2.color ∧ p2.color = p3.color ∧
  areCollinearWithMiddle p1 p2 p3 := by
  sorry

end NUMINAMATH_CALUDE_three_collinear_points_same_color_l2920_292056


namespace NUMINAMATH_CALUDE_angle_addition_l2920_292012

-- Define a structure for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define addition for Angle
def Angle.add (a b : Angle) : Angle :=
  let totalMinutes := a.minutes + b.minutes
  let extraDegrees := totalMinutes / 60
  let remainingMinutes := totalMinutes % 60
  ⟨a.degrees + b.degrees + extraDegrees, remainingMinutes⟩

-- Theorem statement
theorem angle_addition :
  Angle.add ⟨36, 28⟩ ⟨25, 34⟩ = ⟨62, 2⟩ :=
by sorry

end NUMINAMATH_CALUDE_angle_addition_l2920_292012


namespace NUMINAMATH_CALUDE_inscribed_square_rectangle_l2920_292060

theorem inscribed_square_rectangle (a b : ℝ) : 
  (∃ (s r_short r_long : ℝ),
    s^2 = 9 ∧                     -- Area of square is 9
    r_long = 2 * r_short ∧        -- One side of rectangle is double the other
    r_short * r_long = 18 ∧       -- Area of rectangle is 18
    a + b = r_short ∧             -- a and b divide the shorter side
    a^2 + b^2 = s^2)              -- Pythagorean theorem for the right triangle formed
  → a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_rectangle_l2920_292060


namespace NUMINAMATH_CALUDE_cost_of_socks_socks_cost_proof_l2920_292080

theorem cost_of_socks (initial_amount : ℕ) (shirt_cost : ℕ) (remaining_amount : ℕ) : ℕ :=
  initial_amount - shirt_cost - remaining_amount

theorem socks_cost_proof (initial_amount : ℕ) (shirt_cost : ℕ) (remaining_amount : ℕ) 
    (h1 : initial_amount = 100)
    (h2 : shirt_cost = 24)
    (h3 : remaining_amount = 65) :
  cost_of_socks initial_amount shirt_cost remaining_amount = 11 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_socks_socks_cost_proof_l2920_292080


namespace NUMINAMATH_CALUDE_molly_age_when_stopped_l2920_292065

/-- Calculates the age when a person stops riding their bike daily, given their starting age,
    daily riding distance, total distance covered, and days in a year. -/
def age_when_stopped (starting_age : ℕ) (daily_distance : ℕ) (total_distance : ℕ) (days_per_year : ℕ) : ℕ :=
  starting_age + (total_distance / daily_distance) / days_per_year

/-- Theorem stating that given the specified conditions, Molly's age when she stopped riding
    her bike daily is 16 years old. -/
theorem molly_age_when_stopped :
  let starting_age : ℕ := 13
  let daily_distance : ℕ := 3
  let total_distance : ℕ := 3285
  let days_per_year : ℕ := 365
  age_when_stopped starting_age daily_distance total_distance days_per_year = 16 := by
  sorry


end NUMINAMATH_CALUDE_molly_age_when_stopped_l2920_292065


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l2920_292085

/-- 
Given two infinite geometric series:
1. The first series with first term a₁ = 12 and second term a₂ = 3
2. The second series with first term b₁ = 12 and second term b₂ = 3 + n
If the sum of the second series is three times the sum of the first series,
then n = 6.
-/
theorem geometric_series_ratio (n : ℝ) : 
  let a₁ : ℝ := 12
  let a₂ : ℝ := 3
  let b₁ : ℝ := 12
  let b₂ : ℝ := 3 + n
  let r₁ : ℝ := a₂ / a₁
  let r₂ : ℝ := b₂ / b₁
  let S₁ : ℝ := a₁ / (1 - r₁)
  let S₂ : ℝ := b₁ / (1 - r₂)
  S₂ = 3 * S₁ → n = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l2920_292085


namespace NUMINAMATH_CALUDE_probability_of_red_and_flag_in_three_draws_l2920_292069

/-- Represents a single draw from the bag -/
inductive Ball : Type
| wind : Ball
| exhibition : Ball
| red : Ball
| flag : Ball

/-- Represents a set of three draws -/
def DrawSet := (Ball × Ball × Ball)

/-- The sample data of 20 draw sets -/
def sampleData : List DrawSet := [
  (Ball.wind, Ball.red, Ball.red),
  (Ball.exhibition, Ball.flag, Ball.red),
  (Ball.flag, Ball.exhibition, Ball.wind),
  (Ball.wind, Ball.red, Ball.exhibition),
  (Ball.red, Ball.red, Ball.exhibition),
  (Ball.wind, Ball.wind, Ball.flag),
  (Ball.exhibition, Ball.red, Ball.flag),
  (Ball.red, Ball.wind, Ball.wind),
  (Ball.flag, Ball.flag, Ball.red),
  (Ball.red, Ball.exhibition, Ball.flag),
  (Ball.red, Ball.red, Ball.wind),
  (Ball.red, Ball.wind, Ball.exhibition),
  (Ball.red, Ball.red, Ball.red),
  (Ball.flag, Ball.wind, Ball.wind),
  (Ball.flag, Ball.red, Ball.exhibition),
  (Ball.flag, Ball.flag, Ball.wind),
  (Ball.exhibition, Ball.exhibition, Ball.flag),
  (Ball.red, Ball.exhibition, Ball.exhibition),
  (Ball.red, Ball.red, Ball.flag),
  (Ball.red, Ball.flag, Ball.flag)
]

/-- Checks if a draw set contains both red and flag balls -/
def containsRedAndFlag (s : DrawSet) : Bool :=
  match s with
  | (Ball.red, Ball.flag, _) | (Ball.red, _, Ball.flag) | (Ball.flag, Ball.red, _) 
  | (Ball.flag, _, Ball.red) | (_, Ball.red, Ball.flag) | (_, Ball.flag, Ball.red) => true
  | _ => false

/-- Counts the number of draw sets containing both red and flag balls -/
def countRedAndFlag (data : List DrawSet) : Nat :=
  data.filter containsRedAndFlag |>.length

/-- The theorem to be proved -/
theorem probability_of_red_and_flag_in_three_draws : 
  (countRedAndFlag sampleData : ℚ) / sampleData.length = 3 / 20 := by
  sorry


end NUMINAMATH_CALUDE_probability_of_red_and_flag_in_three_draws_l2920_292069


namespace NUMINAMATH_CALUDE_function_root_implies_a_range_l2920_292004

theorem function_root_implies_a_range (a : ℝ) :
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ a * x + 1 = 0) →
  (a < -1 ∨ a > 1) :=
by sorry

end NUMINAMATH_CALUDE_function_root_implies_a_range_l2920_292004


namespace NUMINAMATH_CALUDE_barbara_shopping_cost_l2920_292019

-- Define the quantities and prices
def tuna_packs : ℕ := 5
def tuna_price : ℚ := 2
def water_bottles : ℕ := 4
def water_price : ℚ := 3/2
def other_goods_cost : ℚ := 40

-- Define the total cost function
def total_cost : ℚ :=
  (tuna_packs * tuna_price) + (water_bottles * water_price) + other_goods_cost

-- Theorem statement
theorem barbara_shopping_cost :
  total_cost = 56 := by
  sorry

end NUMINAMATH_CALUDE_barbara_shopping_cost_l2920_292019


namespace NUMINAMATH_CALUDE_roberto_outfits_l2920_292009

/-- Calculates the number of possible outfits given the number of choices for each item -/
def number_of_outfits (trousers shirts jackets shoes : ℕ) : ℕ :=
  trousers * shirts * jackets * shoes

/-- Theorem stating that Roberto can create 240 different outfits -/
theorem roberto_outfits :
  let trousers : ℕ := 4
  let shirts : ℕ := 5
  let jackets : ℕ := 3
  let shoes : ℕ := 4
  number_of_outfits trousers shirts jackets shoes = 240 := by
  sorry

end NUMINAMATH_CALUDE_roberto_outfits_l2920_292009
