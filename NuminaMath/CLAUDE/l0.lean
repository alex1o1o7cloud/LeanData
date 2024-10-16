import Mathlib

namespace NUMINAMATH_CALUDE_museum_exhibit_count_l0_31

def base5ToBase10 (n : ℕ) : ℕ := sorry

theorem museum_exhibit_count : 
  let clay_tablets := base5ToBase10 1432
  let bronze_sculptures := base5ToBase10 2041
  let stone_carvings := base5ToBase10 232
  clay_tablets + bronze_sculptures + stone_carvings = 580 := by sorry

end NUMINAMATH_CALUDE_museum_exhibit_count_l0_31


namespace NUMINAMATH_CALUDE_max_d_value_l0_20

def is_valid_number (d e : ℕ) : Prop :=
  d < 10 ∧ e < 10 ∧ 
  (5000000 + d * 100000 + 500000 + 2200 + 20 + e) % 44 = 0

theorem max_d_value :
  ∃ (d : ℕ), is_valid_number d 6 ∧
  ∀ (d' e : ℕ), is_valid_number d' e → d' ≤ d :=
by sorry

end NUMINAMATH_CALUDE_max_d_value_l0_20


namespace NUMINAMATH_CALUDE_diagonal_cubes_200_420_480_l0_85

/-- The number of cubes an internal diagonal passes through in a rectangular solid -/
def diagonal_cubes (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

/-- Theorem: The number of cubes an internal diagonal passes through in a 200×420×480 rectangular solid is 1000 -/
theorem diagonal_cubes_200_420_480 :
  diagonal_cubes 200 420 480 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_cubes_200_420_480_l0_85


namespace NUMINAMATH_CALUDE_expand_and_simplify_l0_48

theorem expand_and_simplify (x : ℝ) : 
  (1 + x^3) * (1 - x^4)^2 = 1 + x^3 - 2*x^4 - 2*x^7 + x^8 + x^11 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l0_48


namespace NUMINAMATH_CALUDE_reinforcement_size_l0_91

/-- Calculates the size of the reinforcement given the initial garrison size, 
    initial provision duration, days passed before reinforcement, and 
    remaining provision duration after reinforcement. -/
def calculate_reinforcement (initial_garrison : ℕ) (initial_duration : ℕ) 
                             (days_passed : ℕ) (remaining_duration : ℕ) : ℕ :=
  let provisions := initial_garrison * initial_duration
  let provisions_left := initial_garrison * (initial_duration - days_passed)
  (provisions_left / remaining_duration) - initial_garrison

theorem reinforcement_size :
  calculate_reinforcement 2000 54 18 20 = 1600 := by
  sorry

#eval calculate_reinforcement 2000 54 18 20

end NUMINAMATH_CALUDE_reinforcement_size_l0_91


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_equals_three_l0_58

theorem unique_solution_implies_a_equals_three (a : ℝ) :
  (∃! x : ℝ, x^2 + a * |x| + a^2 - 9 = 0) → a = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_equals_three_l0_58


namespace NUMINAMATH_CALUDE_geometric_sequence_m_value_l0_66

/-- Definition of the sum of the first n terms of the geometric sequence -/
def S (n : ℕ) (m : ℝ) : ℝ := m * 2^(n - 1) - 3

/-- Definition of the nth term of the geometric sequence -/
def a (n : ℕ) (m : ℝ) : ℝ :=
  if n = 1 then S 1 m
  else S n m - S (n - 1) m

/-- Theorem stating that m = 6 for the given geometric sequence -/
theorem geometric_sequence_m_value :
  ∃ (m : ℝ), ∀ (n : ℕ), n ≥ 1 → (a n m) / (a 1 m) = 2^(n - 1) ∧ m = 6 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_m_value_l0_66


namespace NUMINAMATH_CALUDE_triangle_point_inequality_l0_83

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point P
def Point := ℝ × ℝ

-- Function to calculate distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Function to check if a point is inside a triangle
def isInside (t : Triangle) (p : Point) : Prop := sorry

-- Define the perimeter of the triangle
def perimeter (t : Triangle) : ℝ :=
  distance t.A t.B + distance t.B t.C + distance t.C t.A

-- Theorem statement
theorem triangle_point_inequality (t : Triangle) (P : Point) (s : ℝ) :
  perimeter t = 2 * s →
  isInside t P →
  s < distance t.A P + distance t.B P + distance t.C P ∧
  distance t.A P + distance t.B P + distance t.C P < 2 * s :=
by sorry

end NUMINAMATH_CALUDE_triangle_point_inequality_l0_83


namespace NUMINAMATH_CALUDE_intersected_cubes_count_l0_25

/-- Represents a 3D cube composed of unit cubes --/
structure LargeCube where
  side_length : ℕ
  total_cubes : ℕ

/-- Represents a plane intersecting the large cube --/
structure IntersectingPlane where
  perpendicular_to_diagonal : Bool
  bisects_diagonal : Bool

/-- Counts the number of unit cubes intersected by the plane --/
def count_intersected_cubes (cube : LargeCube) (plane : IntersectingPlane) : ℕ :=
  sorry

/-- Theorem stating the number of intersected cubes for a 4x4x4 cube --/
theorem intersected_cubes_count
  (cube : LargeCube)
  (plane : IntersectingPlane)
  (h1 : cube.side_length = 4)
  (h2 : cube.total_cubes = 64)
  (h3 : plane.perpendicular_to_diagonal = true)
  (h4 : plane.bisects_diagonal = true) :
  count_intersected_cubes cube plane = 32 :=
sorry

end NUMINAMATH_CALUDE_intersected_cubes_count_l0_25


namespace NUMINAMATH_CALUDE_equation_solution_l0_94

theorem equation_solution : ∃ x : ℚ, (x - 30) / 3 = (4 - 3*x) / 7 ∧ x = 111/8 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l0_94


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l0_80

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, 4)

theorem perpendicular_vectors (x : ℝ) :
  (vector_a.1 * (vector_b x).1 + vector_a.2 * (vector_b x).2 = 0) → x = -8 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l0_80


namespace NUMINAMATH_CALUDE_area_of_region_l0_22

/-- The area enclosed by the region defined by x^2 + y^2 - 4x + 2y = -2 is 3π -/
theorem area_of_region (x y : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - 4*x + 2*y = -2) → 
  (∃ (center : ℝ × ℝ) (r : ℝ), 
    center = (2, -1) ∧ 
    r = Real.sqrt 3 ∧ 
    (∀ (p : ℝ × ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2) ∧
    π * r^2 = 3 * π) :=
by sorry

end NUMINAMATH_CALUDE_area_of_region_l0_22


namespace NUMINAMATH_CALUDE_alligators_not_hiding_l0_54

theorem alligators_not_hiding (total_alligators hiding_alligators : ℕ) 
  (h1 : total_alligators = 75)
  (h2 : hiding_alligators = 19) : 
  total_alligators - hiding_alligators = 56 := by
  sorry

end NUMINAMATH_CALUDE_alligators_not_hiding_l0_54


namespace NUMINAMATH_CALUDE_quadratic_equation_identification_l0_43

/-- Definition of a quadratic equation in one variable -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equations given in the problem -/
def eq_A : ℝ → ℝ := λ x => 2 * x - 1
def eq_B : ℝ → ℝ := λ x => x^2
def eq_C : ℝ → ℝ → ℝ := λ x y => 5 * x * y - 1
def eq_D : ℝ → ℝ := λ x => 2 * (x + 1)

/-- Theorem stating that eq_B is quadratic while others are not -/
theorem quadratic_equation_identification :
  is_quadratic eq_B ∧ 
  ¬is_quadratic eq_A ∧ 
  ¬is_quadratic (λ x => eq_C x x) ∧ 
  ¬is_quadratic eq_D :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_identification_l0_43


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_reciprocal_l0_0

theorem repeating_decimal_equals_reciprocal (a : ℕ) : 
  (1 ≤ a ∧ a ≤ 9) → 
  ((10 + a - 1) / 90 : ℚ) = 1 / a → 
  a = 6 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_reciprocal_l0_0


namespace NUMINAMATH_CALUDE_harriet_return_speed_harriet_return_speed_approx_145_l0_19

/-- Calculates the return speed given the conditions of Harriet's trip -/
theorem harriet_return_speed (outbound_speed : ℝ) (total_time : ℝ) (outbound_time_minutes : ℝ) : ℝ :=
  let outbound_time : ℝ := outbound_time_minutes / 60
  let distance : ℝ := outbound_speed * outbound_time
  let return_time : ℝ := total_time - outbound_time
  distance / return_time

/-- Proves that Harriet's return speed is approximately 145 km/h -/
theorem harriet_return_speed_approx_145 :
  ∃ ε > 0, abs (harriet_return_speed 105 5 174 - 145) < ε :=
sorry

end NUMINAMATH_CALUDE_harriet_return_speed_harriet_return_speed_approx_145_l0_19


namespace NUMINAMATH_CALUDE_linear_function_proof_l0_79

/-- A linear function passing through two given points -/
structure LinearFunction where
  k : ℝ
  b : ℝ
  point1_x : ℝ
  point1_y : ℝ
  point2_x : ℝ
  point2_y : ℝ
  eq_at_point1 : point1_y = k * point1_x + b
  eq_at_point2 : point2_y = k * point2_x + b

/-- The specific linear function passing through (2,1) and (-3,6) -/
def specificLinearFunction : LinearFunction := {
  k := -1
  b := 3
  point1_x := 2
  point1_y := 1
  point2_x := -3
  point2_y := 6
  eq_at_point1 := by sorry
  eq_at_point2 := by sorry
}

theorem linear_function_proof :
  (specificLinearFunction.k = -1 ∧ specificLinearFunction.b = 3) ∧
  ¬(5 = specificLinearFunction.k * (-1) + specificLinearFunction.b) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_proof_l0_79


namespace NUMINAMATH_CALUDE_log_49_x_equals_half_log_7_x_l0_88

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_49_x_equals_half_log_7_x (x : ℝ) (h : log 7 (x + 6) = 2) :
  log 49 x = (log 7 x) / 2 := by
sorry

end NUMINAMATH_CALUDE_log_49_x_equals_half_log_7_x_l0_88


namespace NUMINAMATH_CALUDE_intersection_point_l0_70

def circle_C (ρ θ : ℝ) : Prop := ρ = Real.cos θ + Real.sin θ

def line_l (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - Real.pi / 4) = Real.sqrt 2 / 2

def valid_polar_coord (ρ θ : ℝ) : Prop := ρ ≥ 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi

theorem intersection_point :
  ∃ (ρ θ : ℝ), 
    circle_C ρ θ ∧ 
    line_l ρ θ ∧ 
    valid_polar_coord ρ θ ∧ 
    ρ = 1 ∧ 
    θ = Real.pi / 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_point_l0_70


namespace NUMINAMATH_CALUDE_train_speed_l0_89

/-- Calculates the speed of a train given its composition and time to cross a bridge -/
theorem train_speed (num_carriages : ℕ) (carriage_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  num_carriages = 24 →
  carriage_length = 60 →
  bridge_length = 1500 →
  crossing_time = 3 →
  (((num_carriages + 1) * carriage_length + bridge_length) / 1000) / (crossing_time / 60) = 60 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l0_89


namespace NUMINAMATH_CALUDE_equation_roots_arithmetic_progression_l0_8

theorem equation_roots_arithmetic_progression (a : ℝ) : 
  (∃ r d : ℝ, (∀ x : ℝ, x^8 + a*x^4 + 1 = 0 ↔ 
    x = (r - 3*d)^(1/4) ∨ x = (r - d)^(1/4) ∨ x = (r + d)^(1/4) ∨ x = (r + 3*d)^(1/4))) 
  → a = -82/9 :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_arithmetic_progression_l0_8


namespace NUMINAMATH_CALUDE_expression_value_l0_78

theorem expression_value (x y : ℝ) (hx : x = 2) (hy : y = 5) :
  (x^4 + 2*y^2) / 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l0_78


namespace NUMINAMATH_CALUDE_solve_for_t_l0_86

theorem solve_for_t (s t u : ℚ) 
  (eq1 : 12 * s + 6 * t + 3 * u = 180)
  (eq2 : t = s + 2)
  (eq3 : t = u + 3) :
  t = 213 / 21 := by
sorry

end NUMINAMATH_CALUDE_solve_for_t_l0_86


namespace NUMINAMATH_CALUDE_y_range_given_inequality_l0_63

/-- Custom multiplication operation on ℝ -/
def star (x y : ℝ) : ℝ := x * (1 - y)

/-- Theorem stating the range of y given the condition -/
theorem y_range_given_inequality :
  (∀ x : ℝ, star (x - y) (x + y) < 1) →
  ∃ a b : ℝ, a = -1/2 ∧ b = 3/2 ∧ y ∈ Set.Ioo a b :=
by sorry

end NUMINAMATH_CALUDE_y_range_given_inequality_l0_63


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l0_44

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_parallel_planes
  (l m : Line) (α β : Plane)
  (different_lines : l ≠ m)
  (non_coincident_planes : α ≠ β)
  (l_perp_α : perpendicular l α)
  (α_parallel_β : parallel α β)
  (m_in_β : contained_in m β) :
  line_perpendicular l m :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l0_44


namespace NUMINAMATH_CALUDE_distance_between_points_l0_10

/-- The distance between points (5, 5) and (0, 0) is 5√2 -/
theorem distance_between_points : 
  let p1 : ℝ × ℝ := (5, 5)
  let p2 : ℝ × ℝ := (0, 0)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l0_10


namespace NUMINAMATH_CALUDE_inequalities_proof_l0_92

theorem inequalities_proof (a b : ℝ) (h : (1 / a) < (1 / b) ∧ (1 / b) < 0) :
  (a + b < a * b) ∧ (b / a + a / b > 2) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l0_92


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l0_3

theorem quadratic_root_relation (b c : ℚ) : 
  (∃ r s : ℚ, 5 * r^2 - 8 * r + 2 = 0 ∧ 5 * s^2 - 8 * s + 2 = 0 ∧
   (r - 3)^2 + b * (r - 3) + c = 0 ∧ (s - 3)^2 + b * (s - 3) + c = 0) →
  c = 23/5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l0_3


namespace NUMINAMATH_CALUDE_converse_and_inverse_false_l0_1

-- Define the properties of quadrilaterals
def is_square (q : Type) : Prop := sorry
def is_rectangle (q : Type) : Prop := sorry

-- Given statement
axiom square_implies_rectangle : ∀ (q : Type), is_square q → is_rectangle q

-- Theorem to prove
theorem converse_and_inverse_false :
  (∃ (q : Type), is_rectangle q ∧ ¬is_square q) ∧
  (∃ (q : Type), ¬is_square q ∧ is_rectangle q) :=
sorry

end NUMINAMATH_CALUDE_converse_and_inverse_false_l0_1


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l0_33

theorem arithmetic_calculation : 28 * 7 * 25 + 12 * 7 * 25 + 7 * 11 * 3 + 44 = 7275 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l0_33


namespace NUMINAMATH_CALUDE_max_radius_circle_x_value_l0_61

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the theorem
theorem max_radius_circle_x_value 
  (C : ℝ × ℝ → ℝ → Set (ℝ × ℝ)) 
  (max_radius : ℝ) 
  (x : ℝ) :
  (∀ r : ℝ, r ≤ max_radius) →
  ((8, 0) ∈ C (0, 0) max_radius) →
  ((x, 0) ∈ C (0, 0) max_radius) →
  (max_radius = 8) →
  (x = -8) :=
by sorry

end NUMINAMATH_CALUDE_max_radius_circle_x_value_l0_61


namespace NUMINAMATH_CALUDE_cuboid_dimensions_l0_38

/-- Represents a cuboid with side areas a, b, and c, and dimensions l, w, h -/
structure Cuboid where
  a : ℝ
  b : ℝ
  c : ℝ
  l : ℝ
  w : ℝ
  h : ℝ

/-- The theorem stating that a cuboid with side areas 5, 8, and 10 has dimensions 4, 2.5, and 2 -/
theorem cuboid_dimensions (cube : Cuboid) 
  (h1 : cube.a = 5) 
  (h2 : cube.b = 8) 
  (h3 : cube.c = 10) 
  (h4 : cube.l * cube.w = cube.a) 
  (h5 : cube.l * cube.h = cube.b) 
  (h6 : cube.w * cube.h = cube.c) :
  cube.l = 4 ∧ cube.w = 2.5 ∧ cube.h = 2 := by
  sorry


end NUMINAMATH_CALUDE_cuboid_dimensions_l0_38


namespace NUMINAMATH_CALUDE_number_difference_l0_15

theorem number_difference (L S : ℕ) (h1 : L = 1637) (h2 : L = 6 * S + 5) : L - S = 1365 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l0_15


namespace NUMINAMATH_CALUDE_rectangle_length_fraction_of_circle_radius_l0_60

theorem rectangle_length_fraction_of_circle_radius 
  (square_area : ℝ) 
  (rectangle_area : ℝ) 
  (rectangle_breadth : ℝ) 
  (h1 : square_area = 1225) 
  (h2 : rectangle_area = 140) 
  (h3 : rectangle_breadth = 10) : 
  (rectangle_area / rectangle_breadth) / Real.sqrt square_area = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_fraction_of_circle_radius_l0_60


namespace NUMINAMATH_CALUDE_parabola_equation_l0_21

/-- A parabola with focus F and a point M satisfying given conditions -/
structure Parabola where
  p : ℝ
  F : ℝ × ℝ
  M : ℝ × ℝ
  h_p_pos : p > 0
  h_F_focus : F = (p/2, 0)
  h_M_on_C : M.2^2 = 2*p*M.1
  h_M_x : M.1 = 4
  h_MF_dist : Real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2) = 5

/-- The theorem stating the equation of the parabola -/
theorem parabola_equation (C : Parabola) : C.M.2^2 = 18 * C.M.1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l0_21


namespace NUMINAMATH_CALUDE_eighth_group_student_number_l0_72

/-- Represents a systematic sampling of students -/
structure SystematicSampling where
  total_students : ℕ
  num_groups : ℕ
  students_per_group : ℕ
  selected_number : ℕ
  selected_group : ℕ

/-- Calculates the number of the student in a given group -/
def student_number (s : SystematicSampling) (group : ℕ) : ℕ :=
  s.selected_number + (group - s.selected_group) * s.students_per_group

/-- Theorem: In the given systematic sampling, the student from the 8th group has number 37 -/
theorem eighth_group_student_number (s : SystematicSampling) 
    (h1 : s.total_students = 50)
    (h2 : s.num_groups = 10)
    (h3 : s.students_per_group = 5)
    (h4 : s.selected_number = 12)
    (h5 : s.selected_group = 3) :
    student_number s 8 = 37 := by
  sorry


end NUMINAMATH_CALUDE_eighth_group_student_number_l0_72


namespace NUMINAMATH_CALUDE_function_periodicity_l0_28

/-- Given a > 0 and f satisfying f(x) + f(x+a) + f(x) f(x+a) = 1 for all x,
    prove that f is periodic with period 2a -/
theorem function_periodicity (a : ℝ) (f : ℝ → ℝ) (ha : a > 0)
  (hf : ∀ x : ℝ, f x + f (x + a) + f x * f (x + a) = 1) :
  ∀ x : ℝ, f (x + 2 * a) = f x := by
  sorry

end NUMINAMATH_CALUDE_function_periodicity_l0_28


namespace NUMINAMATH_CALUDE_matrix_product_AB_l0_82

def A : Matrix (Fin 4) (Fin 3) ℝ := !![0, -1, 2; 2, 1, 1; 3, 0, 1; 3, 7, 1]
def B : Matrix (Fin 3) (Fin 2) ℝ := !![3, 1; 2, 1; 1, 0]

theorem matrix_product_AB :
  A * B = !![0, -1; 9, 3; 10, 3; 24, 10] := by sorry

end NUMINAMATH_CALUDE_matrix_product_AB_l0_82


namespace NUMINAMATH_CALUDE_inequality_proof_l0_37

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l0_37


namespace NUMINAMATH_CALUDE_sequence_median_l0_95

def sequence_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def median_position (n : ℕ) : ℕ := (sequence_sum n + 1) / 2

theorem sequence_median :
  ∃ (m : ℕ), m = 106 ∧
  sequence_sum (m - 1) < median_position 150 ∧
  median_position 150 ≤ sequence_sum m :=
sorry

end NUMINAMATH_CALUDE_sequence_median_l0_95


namespace NUMINAMATH_CALUDE_remainder_385857_div_6_l0_29

theorem remainder_385857_div_6 : 385857 % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_385857_div_6_l0_29


namespace NUMINAMATH_CALUDE_knowledge_competition_probability_l0_55

/-- Represents the probability of correctly answering a single question -/
def p : ℝ := 0.8

/-- Represents the number of preset questions -/
def n : ℕ := 5

/-- Represents the probability of answering exactly 4 questions before advancing -/
def prob_4_questions : ℝ := 2 * p^3 * (1 - p)

theorem knowledge_competition_probability : prob_4_questions = 0.128 := by
  sorry


end NUMINAMATH_CALUDE_knowledge_competition_probability_l0_55


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l0_74

theorem circle_diameter_from_area (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 4 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l0_74


namespace NUMINAMATH_CALUDE_total_nails_and_claws_is_524_l0_4

/-- The total number of nails and claws Cassie needs to cut -/
def total_nails_and_claws : ℕ :=
  -- Dogs
  4 * 4 * 4 +
  -- Parrots
  (7 * 2 * 3 + 1 * 2 * 4 + 1 * 2 * 2) +
  -- Cats
  (1 * 2 * 5 + 1 * 2 * 4 + 1) +
  -- Rabbits
  (5 * 4 * 9 + 3 * 9 + 2) +
  -- Lizards
  (4 * 4 * 5 + 1 * 4 * 4) +
  -- Tortoises
  (2 * 4 * 4 + 3 * 4 + 5 + 3 * 4 + 3)

/-- Theorem stating that the total number of nails and claws is 524 -/
theorem total_nails_and_claws_is_524 : total_nails_and_claws = 524 := by
  sorry

end NUMINAMATH_CALUDE_total_nails_and_claws_is_524_l0_4


namespace NUMINAMATH_CALUDE_unique_a_value_l0_99

theorem unique_a_value (a : ℝ) : 
  (∃! (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    x₁ ∈ Set.Icc (0 : ℝ) (2 * Real.pi) ∧ 
    x₂ ∈ Set.Icc (0 : ℝ) (2 * Real.pi) ∧ 
    x₃ ∈ Set.Icc (0 : ℝ) (2 * Real.pi) ∧
    Real.sin x₁ + Real.sqrt 3 * Real.cos x₁ = a ∧
    Real.sin x₂ + Real.sqrt 3 * Real.cos x₂ = a ∧
    Real.sin x₃ + Real.sqrt 3 * Real.cos x₃ = a) ↔ 
  a = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_unique_a_value_l0_99


namespace NUMINAMATH_CALUDE_isosceles_triangle_areas_l0_30

theorem isosceles_triangle_areas (W X Y : ℝ) : 
  (W = (5 * 5) / 2) →
  (X = (12 * 12) / 2) →
  (Y = (13 * 13) / 2) →
  (X + Y ≠ 2 * W + X) ∧
  (W + X ≠ Y) ∧
  (2 * X ≠ W + Y) ∧
  (X + W ≠ X) ∧
  (W + Y ≠ 2 * X) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_areas_l0_30


namespace NUMINAMATH_CALUDE_min_value_expression_l0_27

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 4*a + 1) * (b^2 + 4*b + 1) * (c^2 + 4*c + 1) / (a * b * c) ≥ 216 ∧
  (∃ a b c, (a^2 + 4*a + 1) * (b^2 + 4*b + 1) * (c^2 + 4*c + 1) / (a * b * c) = 216) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l0_27


namespace NUMINAMATH_CALUDE_sequence_fourth_term_l0_40

theorem sequence_fourth_term (a : ℕ → ℕ) (S : ℕ → ℕ) : 
  (∀ n : ℕ, S n = 3^n + 2*n + 1) →
  (∀ n : ℕ, n ≥ 1 → S n = S (n-1) + a n) →
  a 4 = 56 := by
sorry

end NUMINAMATH_CALUDE_sequence_fourth_term_l0_40


namespace NUMINAMATH_CALUDE_polynomial_not_equal_33_l0_57

theorem polynomial_not_equal_33 (x y : ℤ) :
  x^5 + 3*x^4*y - 5*x^3*y^2 - 15*x^2*y^3 + 4*x*y^4 + 12*y^5 ≠ 33 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_not_equal_33_l0_57


namespace NUMINAMATH_CALUDE_max_valid_n_l0_98

def S : Set ℕ := {n | ∃ x y : ℕ, n = x * y * (x + y)}

def valid (a n : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → (a + 2^k) ∈ S

theorem max_valid_n :
  ∃ a : ℕ, valid a 3 ∧ ∀ n : ℕ, valid a n → n ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_valid_n_l0_98


namespace NUMINAMATH_CALUDE_prism_volume_l0_24

/-- The volume of a right rectangular prism with face areas 10, 15, and 18 square inches -/
theorem prism_volume (l w h : ℝ) (h1 : l * w = 10) (h2 : w * h = 15) (h3 : l * h = 18) :
  l * w * h = 30 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l0_24


namespace NUMINAMATH_CALUDE_min_value_sum_fourth_and_square_l0_16

theorem min_value_sum_fourth_and_square (t : ℝ) :
  let f := fun (a : ℝ) => a^4 + (t - a)^2
  ∃ (min_val : ℝ), (∀ (a : ℝ), f a ≥ min_val) ∧ (min_val = t^4 / 16 + t^2 / 4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_fourth_and_square_l0_16


namespace NUMINAMATH_CALUDE_baker_shopping_cost_l0_56

theorem baker_shopping_cost :
  let flour_boxes : ℕ := 3
  let flour_price : ℕ := 3
  let egg_trays : ℕ := 3
  let egg_price : ℕ := 10
  let milk_liters : ℕ := 7
  let milk_price : ℕ := 5
  let soda_boxes : ℕ := 2
  let soda_price : ℕ := 3
  let total_cost : ℕ := flour_boxes * flour_price + egg_trays * egg_price + 
                        milk_liters * milk_price + soda_boxes * soda_price
  total_cost = 80 := by
sorry


end NUMINAMATH_CALUDE_baker_shopping_cost_l0_56


namespace NUMINAMATH_CALUDE_number_difference_l0_46

theorem number_difference (L S : ℕ) (h1 : L = 1575) (h2 : L = 7 * S + 15) : L - S = 1353 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l0_46


namespace NUMINAMATH_CALUDE_team_composition_proof_l0_84

theorem team_composition_proof (x y : ℕ) (h1 : x > 0) (h2 : y > 0) : 
  (22 * x + 47 * y) / (x + y) = 41 → x / (x + y) = 6 / 25 :=
by
  sorry

end NUMINAMATH_CALUDE_team_composition_proof_l0_84


namespace NUMINAMATH_CALUDE_sum_of_critical_slopes_l0_75

/-- The parabola y = x^2 -/
def parabola (x : ℝ) : ℝ := x^2

/-- The point Q -/
def Q : ℝ × ℝ := (10, 5)

/-- The line through Q with slope m -/
def line (m : ℝ) (x : ℝ) : ℝ := m * (x - Q.1) + Q.2

/-- The quadratic equation representing the intersection of the line and parabola -/
def intersection_quadratic (m : ℝ) (x : ℝ) : ℝ := 
  parabola x - line m x

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ := 
  m^2 - 4 * (10 * m - 5)

/-- The theorem stating that the sum of the critical slopes is 40 -/
theorem sum_of_critical_slopes : 
  ∃ (r s : ℝ), (∀ m, discriminant m < 0 ↔ r < m ∧ m < s) ∧ r + s = 40 :=
sorry

end NUMINAMATH_CALUDE_sum_of_critical_slopes_l0_75


namespace NUMINAMATH_CALUDE_range_of_a_l0_53

-- Define a decreasing function f on the real numbers
def f : ℝ → ℝ := sorry

-- State that f is decreasing
axiom f_decreasing : ∀ x y, x < y → f x > f y

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (f (1 - a) < f (2 * a - 5)) ↔ a < 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l0_53


namespace NUMINAMATH_CALUDE_rhombus_area_l0_49

/-- The area of a rhombus given its vertices in a rectangular coordinate system -/
theorem rhombus_area (A B C D : ℝ × ℝ) : 
  A = (2, 5.5) → 
  B = (8.5, 1) → 
  C = (2, -3.5) → 
  D = (-4.5, 1) → 
  let AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
  let BD : ℝ × ℝ := (D.1 - B.1, D.2 - B.2)
  let cross_product : ℝ := AC.1 * BD.2 - AC.2 * BD.1
  0.5 * |cross_product| = 58.5 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_l0_49


namespace NUMINAMATH_CALUDE_two_solutions_congruence_l0_41

theorem two_solutions_congruence (a : ℕ) (h_a : a < 2007) :
  (∃! u v : ℕ, u < 2007 ∧ v < 2007 ∧ u ≠ v ∧
    (u^2 + a) % 2007 = 0 ∧ (v^2 + a) % 2007 = 0) ↔
  (a % 9 = 0 ∨ a % 9 = 8 ∨ a % 9 = 5 ∨ a % 9 = 2) ∧
  ∃ x : ℕ, x < 223 ∧ (x^2 % 223 = (223 - a % 223) % 223) :=
by sorry

end NUMINAMATH_CALUDE_two_solutions_congruence_l0_41


namespace NUMINAMATH_CALUDE_box_two_three_neg_one_l0_35

-- Define the box operation for integers a, b, and c
def box (a b c : ℤ) : ℚ := (a ^ b : ℚ) - (b ^ c : ℚ) + (c ^ a : ℚ)

-- Theorem statement
theorem box_two_three_neg_one : box 2 3 (-1) = 26 / 3 := by
  sorry

end NUMINAMATH_CALUDE_box_two_three_neg_one_l0_35


namespace NUMINAMATH_CALUDE_properties_of_f_l0_77

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.log (x + 3)

theorem properties_of_f :
  let f' := fun x => Real.exp x - 1 / (x + 3)
  let f'' := fun x => Real.exp x + 1 / ((x + 3)^2)
  (∀ x > -3, f'' x > 0) ∧
  (∃! x₀ : ℝ, -1 < x₀ ∧ x₀ < 0 ∧ f' x₀ = 0) ∧
  (∃ x_min : ℝ, ∀ x > -3, f x ≥ f x_min) ∧
  (∀ x > -3, f x > -1/2) :=
by sorry

end NUMINAMATH_CALUDE_properties_of_f_l0_77


namespace NUMINAMATH_CALUDE_selection_methods_with_female_l0_67

def total_students : ℕ := 8
def male_students : ℕ := 4
def female_students : ℕ := 4
def students_to_select : ℕ := 3

theorem selection_methods_with_female (h1 : total_students = male_students + female_students) 
  (h2 : total_students ≥ students_to_select) :
  (Nat.choose total_students students_to_select) - (Nat.choose male_students students_to_select) = 52 := by
  sorry

end NUMINAMATH_CALUDE_selection_methods_with_female_l0_67


namespace NUMINAMATH_CALUDE_stratified_sampling_third_year_count_l0_5

theorem stratified_sampling_third_year_count 
  (total_students : ℕ) 
  (third_year_students : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 900) 
  (h2 : third_year_students = 400) 
  (h3 : sample_size = 45) :
  (third_year_students * sample_size) / total_students = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_third_year_count_l0_5


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l0_47

-- Define the conditions
def p (x : ℝ) : Prop := abs x > 1
def q (x : ℝ) : Prop := x < -2

-- Theorem statement
theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x) := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l0_47


namespace NUMINAMATH_CALUDE_sin_alpha_value_l0_65

theorem sin_alpha_value (α : Real) : 
  α ∈ Set.Ioo (π) (3*π/2) →  -- α is in the third quadrant
  Real.tan (α + π/4) = 3 → 
  Real.sin α = -Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l0_65


namespace NUMINAMATH_CALUDE_checkerboard_probability_l0_26

/-- Represents a square checkerboard -/
structure Checkerboard where
  size : ℕ

/-- Calculates the number of squares on the perimeter of the board -/
def perimeter_squares (board : Checkerboard) : ℕ :=
  4 * board.size - 4

/-- Calculates the total number of squares on the board -/
def total_squares (board : Checkerboard) : ℕ :=
  board.size * board.size

/-- Calculates the number of squares not on the perimeter -/
def inner_squares (board : Checkerboard) : ℕ :=
  total_squares board - perimeter_squares board

/-- The main theorem to prove -/
theorem checkerboard_probability (board : Checkerboard) (h : board.size = 10) :
  (inner_squares board : ℚ) / (total_squares board) = 16 / 25 := by
  sorry


end NUMINAMATH_CALUDE_checkerboard_probability_l0_26


namespace NUMINAMATH_CALUDE_smallest_integer_l0_32

theorem smallest_integer (a b : ℕ) (x : ℕ) (h1 : b = 18) (h2 : x > 0)
  (h3 : Nat.gcd a b = x + 3) (h4 : Nat.lcm a b = x * (x + 3)) :
  ∃ (a_min : ℕ), a_min = 6 ∧ ∀ (a' : ℕ), (∃ (x' : ℕ), x' > 0 ∧
    Nat.gcd a' b = x' + 3 ∧ Nat.lcm a' b = x' * (x' + 3)) → a' ≥ a_min :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_l0_32


namespace NUMINAMATH_CALUDE_angle_D_measure_l0_97

/-- An isosceles triangle with specific angle relationships -/
structure IsoscelesTriangle where
  /-- Measure of angle E in degrees -/
  angleE : ℝ
  /-- The triangle is isosceles with angle D congruent to angle F -/
  isIsosceles : True
  /-- The measure of angle F is three times the measure of angle E -/
  angleFRelation : True

/-- The measure of angle D in the isosceles triangle -/
def measureAngleD (t : IsoscelesTriangle) : ℝ :=
  3 * t.angleE

/-- Theorem: The measure of angle D is approximately 77 degrees -/
theorem angle_D_measure (t : IsoscelesTriangle) :
  ‖measureAngleD t - 77‖ < 1 := by
  sorry

#check angle_D_measure

end NUMINAMATH_CALUDE_angle_D_measure_l0_97


namespace NUMINAMATH_CALUDE_student_absence_probability_l0_62

theorem student_absence_probability :
  let p_absent : ℚ := 1 / 20
  let p_present : ℚ := 1 - p_absent
  let p_two_absent_one_present : ℚ := 3 * (p_absent * p_absent * p_present)
  p_two_absent_one_present = 57 / 8000 := by
  sorry

end NUMINAMATH_CALUDE_student_absence_probability_l0_62


namespace NUMINAMATH_CALUDE_percentage_problem_l0_13

theorem percentage_problem (x : ℝ) (h : 0.20 * x = 1000) : 1.20 * x = 6000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l0_13


namespace NUMINAMATH_CALUDE_parabola_directrix_l0_36

/-- Given a parabola with equation y² = 2x, its directrix has the equation x = -1/2 -/
theorem parabola_directrix (x y : ℝ) : y^2 = 2*x → (∃ (p : ℝ), p > 0 ∧ y^2 = 4*p*x ∧ x = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l0_36


namespace NUMINAMATH_CALUDE_hexagon_implies_face_fits_l0_11

/-- A rectangular parallelepiped with dimensions a, b, and c. -/
structure RectangularParallelepiped where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : 0 < a
  hb : 0 < b
  hc : 0 < c

/-- A rectangle with dimensions d₁ and d₂. -/
structure Rectangle where
  d₁ : ℝ
  d₂ : ℝ
  hd₁ : 0 < d₁
  hd₂ : 0 < d₂

/-- A hexagonal cross-section of a rectangular parallelepiped. -/
structure HexagonalCrossSection (rp : RectangularParallelepiped) where

/-- The proposition that a hexagonal cross-section fits in a rectangle. -/
def fits_in (h : HexagonalCrossSection rp) (r : Rectangle) : Prop :=
  sorry

/-- The proposition that a face of a rectangular parallelepiped fits in a rectangle. -/
def face_fits_in (rp : RectangularParallelepiped) (r : Rectangle) : Prop :=
  (rp.a ≤ r.d₁ ∧ rp.b ≤ r.d₂) ∨ (rp.a ≤ r.d₂ ∧ rp.b ≤ r.d₁) ∨
  (rp.b ≤ r.d₁ ∧ rp.c ≤ r.d₂) ∨ (rp.b ≤ r.d₂ ∧ rp.c ≤ r.d₁) ∨
  (rp.a ≤ r.d₁ ∧ rp.c ≤ r.d₂) ∨ (rp.a ≤ r.d₂ ∧ rp.c ≤ r.d₁)

/-- The main theorem to be proved. -/
theorem hexagon_implies_face_fits 
  (rp : RectangularParallelepiped) 
  (r : Rectangle) 
  (h : HexagonalCrossSection rp) 
  (h_fits : fits_in h r) : 
  face_fits_in rp r :=
sorry

end NUMINAMATH_CALUDE_hexagon_implies_face_fits_l0_11


namespace NUMINAMATH_CALUDE_bus_seating_problem_l0_14

theorem bus_seating_problem :
  ∀ (bus_seats minibus_seats : ℕ),
    bus_seats = minibus_seats + 20 →
    5 * bus_seats + 5 * minibus_seats = 300 →
    bus_seats = 40 ∧ minibus_seats = 20 :=
by
  sorry

#check bus_seating_problem

end NUMINAMATH_CALUDE_bus_seating_problem_l0_14


namespace NUMINAMATH_CALUDE_fourth_house_number_l0_34

theorem fourth_house_number (x : ℕ) (k : ℕ) : 
  k ≥ 4 → 
  (k + 1) * (x + k) = 78 → 
  x + 6 = 14 :=
by sorry

end NUMINAMATH_CALUDE_fourth_house_number_l0_34


namespace NUMINAMATH_CALUDE_problem_solution_l0_7

noncomputable section

variables (a : ℝ) (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 4 * x + a

def g (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

def t1 (a : ℝ) (x : ℝ) : ℝ := (1/2) * f a x

def t2 (a : ℝ) (x : ℝ) : ℝ := g a x

def t3 (x : ℝ) : ℝ := 2^x

theorem problem_solution (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ m : ℝ, (¬∃ y : ℝ, (∀ x ∈ Set.Icc (-1) (2*m), (f a x ≤ f a y) ∨ (∀ x ∈ Set.Icc (-1) (2*m), f a x ≥ f a y))) ↔ m > 1/2) ∧
  (f a 1 = g a 1 ↔ a = 2) ∧
  (∀ x ∈ Set.Ioo 0 1, t2 a x < t1 a x ∧ t1 a x < t3 x) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l0_7


namespace NUMINAMATH_CALUDE_equal_temperature_proof_l0_73

/-- The temperature at which Fahrenheit and Celsius scales are equal -/
def equal_temperature : ℚ := -40

/-- The relation between Fahrenheit (f) and Celsius (c) temperatures -/
def fahrenheit_celsius_relation (c : ℚ) : ℚ := (9/5) * c + 32

/-- Theorem stating that the equal_temperature is the point where Fahrenheit and Celsius scales meet -/
theorem equal_temperature_proof :
  fahrenheit_celsius_relation equal_temperature = equal_temperature := by
  sorry

end NUMINAMATH_CALUDE_equal_temperature_proof_l0_73


namespace NUMINAMATH_CALUDE_grid_solution_l0_18

/-- Represents a 3x3 grid with numbers from 1 to 9 -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Check if two cells are adjacent in the grid -/
def adjacent (i j k l : Fin 3) : Prop :=
  (i = k ∧ (j.val + 1 = l.val ∨ l.val + 1 = j.val)) ∨
  (j = l ∧ (i.val + 1 = k.val ∨ k.val + 1 = i.val))

/-- The sum of any two numbers in adjacent cells is less than 12 -/
def valid_sum (g : Grid) : Prop :=
  ∀ i j k l, adjacent i j k l → (g i j).val + (g k l).val < 12

/-- The given positions of known numbers in the grid -/
def known_positions (g : Grid) : Prop :=
  g 0 1 = 1 ∧ g 1 0 = 3 ∧ g 1 1 = 5 ∧ g 2 2 = 7 ∧ g 0 2 = 9

/-- The theorem to be proved -/
theorem grid_solution (g : Grid) 
  (h1 : valid_sum g) 
  (h2 : known_positions g) : 
  g 0 0 = 8 ∧ g 2 0 = 6 ∧ g 2 1 = 4 ∧ g 1 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_grid_solution_l0_18


namespace NUMINAMATH_CALUDE_trees_planted_correct_l0_12

/-- The number of maple trees planted in a park --/
def trees_planted (initial final : ℕ) : ℕ := final - initial

/-- Theorem stating that the number of trees planted is the difference between final and initial counts --/
theorem trees_planted_correct (initial final : ℕ) (h : final ≥ initial) :
  trees_planted initial final = final - initial :=
by sorry

end NUMINAMATH_CALUDE_trees_planted_correct_l0_12


namespace NUMINAMATH_CALUDE_only_one_milk_chocolate_affordable_l0_81

-- Define the prices of chocolates
def dark_chocolate_price : ℚ := 5
def milk_chocolate_price : ℚ := 9/2
def white_chocolate_price : ℚ := 6

-- Define the sales tax rate
def sales_tax_rate : ℚ := 7/100

-- Define Leonardo's budget
def leonardo_budget : ℚ := 459/100

-- Function to calculate price with tax
def price_with_tax (price : ℚ) : ℚ := price * (1 + sales_tax_rate)

-- Theorem statement
theorem only_one_milk_chocolate_affordable :
  (price_with_tax dark_chocolate_price > leonardo_budget) ∧
  (price_with_tax white_chocolate_price > leonardo_budget) ∧
  (price_with_tax milk_chocolate_price ≤ leonardo_budget) ∧
  (2 * price_with_tax milk_chocolate_price > leonardo_budget) :=
by sorry

end NUMINAMATH_CALUDE_only_one_milk_chocolate_affordable_l0_81


namespace NUMINAMATH_CALUDE_max_sum_of_distinct_factors_l0_52

theorem max_sum_of_distinct_factors (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b^2 * c^3 = 1350 →
  ∀ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z →
    x * y^2 * z^3 = 1350 →
    a + b + c ≥ x + y + z :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_distinct_factors_l0_52


namespace NUMINAMATH_CALUDE_min_value_product_quotient_l0_50

theorem min_value_product_quotient (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + 3*x + 2) * (y^2 + 3*y + 2) * (z^2 + 3*z + 2) / (x*y*z) ≥ 216 ∧
  (x^2 + 3*x + 2) * (y^2 + 3*y + 2) * (z^2 + 3*z + 2) / (x*y*z) = 216 ↔ x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_quotient_l0_50


namespace NUMINAMATH_CALUDE_grid_sum_theorem_l0_45

/-- Represents a 3x3 grid where each cell contains a number from 1 to 3 --/
def Grid := Fin 3 → Fin 3 → Fin 3

/-- Checks if a row in the grid contains 1, 2, and 3 --/
def valid_row (g : Grid) (r : Fin 3) : Prop :=
  ∀ n : Fin 3, ∃ c : Fin 3, g r c = n.succ

/-- Checks if a column in the grid contains 1, 2, and 3 --/
def valid_column (g : Grid) (c : Fin 3) : Prop :=
  ∀ n : Fin 3, ∃ r : Fin 3, g r c = n.succ

/-- Checks if the entire grid is valid --/
def valid_grid (g : Grid) : Prop :=
  (∀ r : Fin 3, valid_row g r) ∧ (∀ c : Fin 3, valid_column g c)

theorem grid_sum_theorem (g : Grid) :
  valid_grid g →
  g 0 0 = 2 →
  g 1 1 = 3 →
  g 1 2 + g 2 2 + 4 = 8 := by sorry

end NUMINAMATH_CALUDE_grid_sum_theorem_l0_45


namespace NUMINAMATH_CALUDE_nails_per_station_l0_39

theorem nails_per_station (total_nails : ℕ) (num_stations : ℕ) 
  (h1 : total_nails = 140) (h2 : num_stations = 20) :
  total_nails / num_stations = 7 := by
  sorry

end NUMINAMATH_CALUDE_nails_per_station_l0_39


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l0_93

theorem intersection_of_three_lines (k : ℝ) : 
  (∃! p : ℝ × ℝ, 
    (p.1 + k * p.2 = 0) ∧ 
    (2 * p.1 + 3 * p.2 + 8 = 0) ∧ 
    (p.1 - p.2 - 1 = 0)) → 
  k = -1/2 := by
sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l0_93


namespace NUMINAMATH_CALUDE_arithmetic_seq_properties_l0_17

/-- An arithmetic sequence with a_1 = 1 and a_3 - a_2 = 1 -/
def arithmetic_seq (n : ℕ) : ℕ := n

/-- Sum of the first n terms of the arithmetic sequence -/
def arithmetic_seq_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem arithmetic_seq_properties :
  let a := arithmetic_seq
  let S := arithmetic_seq_sum
  (∀ n : ℕ, n ≥ 1 → a n = n) ∧
  (∀ n : ℕ, n ≥ 1 → S n = n * (n + 1) / 2) :=
by
  sorry

#check arithmetic_seq_properties

end NUMINAMATH_CALUDE_arithmetic_seq_properties_l0_17


namespace NUMINAMATH_CALUDE_total_rainfall_five_days_l0_42

/-- Represents the rainfall data for a day -/
structure RainfallData where
  hours : ℝ
  rate : ℝ

/-- Calculates the total rainfall for a given day -/
def totalRainfall (data : RainfallData) : ℝ :=
  data.hours * data.rate

theorem total_rainfall_five_days (monday tuesday wednesday thursday friday : RainfallData)
  (h_monday : monday = { hours := 5, rate := 1 })
  (h_tuesday : tuesday = { hours := 3, rate := 1.5 })
  (h_wednesday : wednesday = { hours := 4, rate := 2 * monday.rate })
  (h_thursday : thursday = { hours := 6, rate := 0.5 * tuesday.rate })
  (h_friday : friday = { hours := 2, rate := 1.5 * wednesday.rate }) :
  totalRainfall monday + totalRainfall tuesday + totalRainfall wednesday +
  totalRainfall thursday + totalRainfall friday = 28 := by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_five_days_l0_42


namespace NUMINAMATH_CALUDE_line_circle_relationship_l0_6

/-- The line equation -/
def line_equation (k x y : ℝ) : Prop :=
  (3*k + 2) * x - k * y - 2 = 0

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y - 2 = 0

/-- The theorem stating the positional relationship between the line and the circle -/
theorem line_circle_relationship :
  ∀ k : ℝ, ∃ x y : ℝ, 
    (line_equation k x y ∧ circle_equation x y) ∨ 
    (∃ x₀ y₀ : ℝ, line_equation k x₀ y₀ ∧ circle_equation x₀ y₀ ∧ 
      ∀ x y : ℝ, line_equation k x y ∧ circle_equation x y → (x, y) = (x₀, y₀)) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_relationship_l0_6


namespace NUMINAMATH_CALUDE_max_remainder_div_by_nine_l0_87

theorem max_remainder_div_by_nine (n : ℕ) (h : n % 9 = 6) : 
  ∀ m : ℕ, m % 9 < 9 ∧ m % 9 ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_max_remainder_div_by_nine_l0_87


namespace NUMINAMATH_CALUDE_quadratic_intersection_l0_51

/-- Given two quadratic functions with two distinct roots each, prove that a third related quadratic function has no real roots -/
theorem quadratic_intersection (a b c : ℝ) :
  ((∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*a*x₁ + b^2 = 0 ∧ x₂^2 + 2*a*x₂ + b^2 = 0) ∧
   (∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ y₁^2 + 2*b*y₁ + c^2 = 0 ∧ y₂^2 + 2*b*y₂ + c^2 = 0)) →
  (∀ z : ℝ, z^2 + 2*c*z + a^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_l0_51


namespace NUMINAMATH_CALUDE_find_M_l0_59

theorem find_M : ∃ M : ℕ+, (18^2 * 45^2 : ℕ) = 30^2 * M^2 ∧ M = 81 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l0_59


namespace NUMINAMATH_CALUDE_largest_prime_divisor_test_l0_69

theorem largest_prime_divisor_test (n : ℕ) : 
  1000 ≤ n → n ≤ 1050 → 
  (∀ p : ℕ, p ≤ 31 → Nat.Prime p → ¬(p ∣ n)) → 
  Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_test_l0_69


namespace NUMINAMATH_CALUDE_coffee_x_ratio_l0_90

/-- Represents the ratio of two quantities -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Represents a coffee mixture -/
structure CoffeeMixture where
  p : ℕ  -- amount of coffee p in lbs
  v : ℕ  -- amount of coffee v in lbs

def total_p : ℕ := 24
def total_v : ℕ := 25

def coffee_x : CoffeeMixture := { p := 20, v := 0 }
def coffee_y : CoffeeMixture := { p := 0, v := 0 }

def ratio_y : Ratio := { numerator := 1, denominator := 5 }

theorem coffee_x_ratio : 
  coffee_x.p * 1 = coffee_x.v * 4 := by sorry

end NUMINAMATH_CALUDE_coffee_x_ratio_l0_90


namespace NUMINAMATH_CALUDE_intersecting_lines_l0_64

theorem intersecting_lines (x y : ℝ) : 
  (2*x - y)^2 - (x + 3*y)^2 = 0 ↔ (x = 4*y ∨ x = -2/3*y) := by
  sorry

end NUMINAMATH_CALUDE_intersecting_lines_l0_64


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_l0_96

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def quadrilateral_properties (ABCD : Quadrilateral) : Prop :=
  let (xa, ya) := ABCD.A
  let (xb, yb) := ABCD.B
  let (xc, yc) := ABCD.C
  let (xd, yd) := ABCD.D
  ∃ (angle_BCD : ℝ),
    angle_BCD = 120 ∧
    (xb - xa)^2 + (yb - ya)^2 = 13^2 ∧
    (xc - xb)^2 + (yc - yb)^2 = 6^2 ∧
    (xd - xc)^2 + (yd - yc)^2 = 5^2 ∧
    (xa - xd)^2 + (ya - yd)^2 = 12^2

-- Define the area of a triangle
def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

-- Define the area of quadrilateral ABCD
def quadrilateral_area (ABCD : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem area_of_quadrilateral (ABCD : Quadrilateral) :
  quadrilateral_properties ABCD →
  quadrilateral_area ABCD = (15 * Real.sqrt 3) / 2 + triangle_area ABCD.B ABCD.D ABCD.A :=
sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_l0_96


namespace NUMINAMATH_CALUDE_f_of_f_of_two_l0_9

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x^2 + 2 * x - 1

-- State the theorem
theorem f_of_f_of_two : f (f 2) = 1481 := by
  sorry

end NUMINAMATH_CALUDE_f_of_f_of_two_l0_9


namespace NUMINAMATH_CALUDE_janessas_initial_cards_l0_23

/-- The number of cards Janessa's father gave her -/
def fathers_cards : ℕ := 13

/-- The number of cards Janessa ordered from eBay -/
def ordered_cards : ℕ := 36

/-- The number of cards Janessa threw away -/
def discarded_cards : ℕ := 4

/-- The number of cards Janessa gave to Dexter -/
def cards_given_to_dexter : ℕ := 29

/-- The number of cards Janessa kept for herself -/
def cards_kept_for_self : ℕ := 20

/-- The initial number of cards Janessa had -/
def initial_cards : ℕ := 4

theorem janessas_initial_cards : 
  initial_cards + fathers_cards + ordered_cards - discarded_cards = 
  cards_given_to_dexter + cards_kept_for_self :=
by sorry

end NUMINAMATH_CALUDE_janessas_initial_cards_l0_23


namespace NUMINAMATH_CALUDE_g_of_5_l0_68

def g (x : ℝ) : ℝ := 3*x^4 - 20*x^3 + 35*x^2 - 40*x + 24

theorem g_of_5 : g 5 = 74 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_l0_68


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l0_2

theorem reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 1 / a > 1 / b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l0_2


namespace NUMINAMATH_CALUDE_exists_decreasing_linear_function_through_origin_l0_76

/-- A linear function that decreases and passes through (0,2) -/
def decreasingLinearFunction (k : ℝ) (x : ℝ) : ℝ := k * x + 2

theorem exists_decreasing_linear_function_through_origin :
  ∃ (k : ℝ), k < 0 ∧
    (∀ (x y : ℝ), x < y → decreasingLinearFunction k x > decreasingLinearFunction k y) ∧
    decreasingLinearFunction k 0 = 2 :=
sorry

end NUMINAMATH_CALUDE_exists_decreasing_linear_function_through_origin_l0_76


namespace NUMINAMATH_CALUDE_roots_sum_cubic_l0_71

theorem roots_sum_cubic (a b c : ℂ) : 
  a^3 + 2*a^2 + 3*a + 4 = 0 →
  b^3 + 2*b^2 + 3*b + 4 = 0 →
  c^3 + 2*c^2 + 3*c + 4 = 0 →
  (a^3 - b^3) / (a - b) + (b^3 - c^3) / (b - c) + (c^3 - a^3) / (c - a) = -1 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_cubic_l0_71
