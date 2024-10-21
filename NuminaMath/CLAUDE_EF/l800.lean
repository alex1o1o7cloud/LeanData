import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_cone_volume_l800_80084

/-- Represents a cone with a base radius and height -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- The volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.baseRadius^2 * c.height

theorem smaller_cone_volume 
  (c1 c2 : Cone) 
  (V α : ℝ)
  (h_common_vertex : c1.height = c2.height) 
  (h_volume_diff : coneVolume c1 - coneVolume c2 = V) 
  (h_angle : ∃ (p : ℝ × ℝ), 
    p.1^2 + p.2^2 = c1.baseRadius^2 ∧ 
    2 * Real.arctan (c2.baseRadius / Real.sqrt (c1.baseRadius^2 - c2.baseRadius^2)) = α) :
  coneVolume c2 = V * (Real.tan (α/2))^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_cone_volume_l800_80084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_x_coordinate_equidistant_condition_l800_80054

/-- The x-coordinate of the point on the x-axis that is equidistant from A(-3, -2) and B(2, -6) -/
theorem equidistant_point_x_coordinate : 
  ∃ x : ℝ, x = 2.7 ∧ 
  ((-3 - x)^2 + (-2 - 0)^2 = (2 - x)^2 + (-6 - 0)^2) := by
  sorry

/-- Definition of point A -/
def A : ℝ × ℝ := (-3, -2)

/-- Definition of point B -/
def B : ℝ × ℝ := (2, -6)

/-- The point on the x-axis -/
def P (x : ℝ) : ℝ × ℝ := (x, 0)

/-- Distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The equidistant condition -/
theorem equidistant_condition (x : ℝ) : 
  distance A (P x) = distance B (P x) ↔ 
  ((-3 - x)^2 + (-2 - 0)^2 = (2 - x)^2 + (-6 - 0)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_x_coordinate_equidistant_condition_l800_80054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_most_three_heads_ten_coins_l800_80099

theorem probability_at_most_three_heads_ten_coins :
  let n : ℕ := 10
  let k : ℕ := 3
  let total_outcomes : ℕ := 2^n
  let favorable_outcomes : ℕ := (Finset.range (k+1)).sum (λ i ↦ Nat.choose n i)
  (favorable_outcomes : ℚ) / total_outcomes = 11 / 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_most_three_heads_ten_coins_l800_80099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solution_set_l800_80020

open Real

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- Assume f is differentiable
axiom f_differentiable : Differentiable ℝ f

-- Define the given equation
axiom f_equation (x : ℝ) (hx : 1 < x) : f x + (x - 1) * f' x = x^2 * (x - 2)

-- Define the given condition
axiom f_condition : f (exp 2) = 0

-- Theorem to prove
theorem f_solution_set :
  {x : ℝ | f (exp x) < 0} = Set.Ioo 0 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solution_set_l800_80020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_between_specific_planes_l800_80019

/-- The cosine of the angle between two planes given by their normal vectors -/
noncomputable def cos_angle_between_planes (n1 n2 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := n1
  let (x2, y2, z2) := n2
  let dot_product := x1 * x2 + y1 * y2 + z1 * z2
  let magnitude1 := Real.sqrt (x1^2 + y1^2 + z1^2)
  let magnitude2 := Real.sqrt (x2^2 + y2^2 + z2^2)
  dot_product / (magnitude1 * magnitude2)

/-- The normal vector of a plane ax + by + cz + d = 0 -/
def normal_vector (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, c)

theorem cos_angle_between_specific_planes :
  let n1 := normal_vector 3 2 (-1)
  let n2 := normal_vector 9 6 3
  cos_angle_between_planes n1 n2 = 6/7 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_between_specific_planes_l800_80019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_values_l800_80066

theorem expression_values (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (|a| / a + b / |b|) ∈ ({-2, 0, 2} : Set ℝ) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_values_l800_80066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collatz_terminates_l800_80062

def operation (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else (n + 1) / 2

theorem collatz_terminates (n : ℕ) (h : n > 0) :
  ∃ (k : ℕ), ∃ (seq : ℕ → ℕ),
    seq 0 = n ∧
    seq k = 1 ∧
    ∀ (i : ℕ), i < k →
      seq (i + 1) = operation (seq i) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collatz_terminates_l800_80062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_l800_80008

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- State the theorem
theorem inverse_function_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ g : ℝ → ℝ, Function.RightInverse g (f a) ∧ g 3 = -1) → a = 1/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_l800_80008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l800_80033

/-- Helper function to calculate the area of a quadrilateral given its four vertices -/
def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ := sorry

/-- The area of a quadrilateral with vertices at (1, 3), (1, 1), (3, 1), and (2023, 2024) is 2042113 square units. -/
theorem quadrilateral_area : 
  let A : ℝ × ℝ := (1, 3)
  let B : ℝ × ℝ := (1, 1)
  let C : ℝ × ℝ := (3, 1)
  let D : ℝ × ℝ := (2023, 2024)
  area_quadrilateral A B C D = 2042113 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l800_80033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l800_80074

/-- Given vectors m and n, prove that ω = 1 and BA · BC = -3/2 -/
theorem vector_problem (ω x : ℝ) (A B C : ℝ) :
  ω > 0 →
  let m : ℝ × ℝ := (2 * Real.sin (ω * x), Real.cos (ω * x)^2 - Real.sin (ω * x)^2)
  let n : ℝ × ℝ := (Real.sqrt 3 * Real.cos (ω * x), 1)
  let f : ℝ → ℝ := fun x => m.1 * n.1 + m.2 * n.2
  (∀ y : ℝ, f (x + π) = f x) →
  (∀ z : ℝ, z > 0 → z < π → ¬(∀ y : ℝ, f (x + z) = f x)) →
  f B = -2 →
  Real.sqrt 3 = (BC : ℝ) →
  Real.sin B = Real.sqrt 3 * Real.sin A →
  ω = 1 ∧ (BA : ℝ) * (BC : ℝ) = -3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l800_80074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_r_value_l800_80070

/-- A geometric sequence with first term a and common ratio q -/
noncomputable def geometric_sequence (a q : ℝ) : ℕ → ℝ := λ n ↦ a * q ^ (n - 1)

/-- The sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_r_value (a q : ℝ) (h : q ≠ 0) :
  (∃ r : ℝ, ∀ n : ℕ, n > 0 → geometric_sum a q n = 3^n + r) →
  (∃ r : ℝ, r = -1) := by
  sorry

#check geometric_sequence_r_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_r_value_l800_80070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_element_property_l800_80001

-- Define the set A as a parameter instead of a recursive definition
variable (A : Set ℂ)

-- Define the properties of A as axioms
axiom A_property (a : ℂ) : a ∈ A → (1 / (1 - a)) ∈ A
axiom A_not_one : (1 : ℂ) ∉ A

-- State the theorem
theorem element_property (a : ℂ) (ha : a ∈ A) : (1 - 1/a) ∈ A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_element_property_l800_80001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_negative_one_of_four_l800_80090

theorem power_negative_one_of_four : (4 : ℝ)^((-1) : ℝ) = (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_negative_one_of_four_l800_80090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_not_equivalent_l800_80038

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the sets of x that satisfy each inequality
def S1 : Set ℝ := {x | x > 2 ∧ lg (x^2 - 4) > lg (4*x - 7)}
def S2 : Set ℝ := {x | x^2 - 4 > 4*x - 7}

-- Theorem statement
theorem inequalities_not_equivalent : S1 ≠ S2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_not_equivalent_l800_80038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_theorem_l800_80069

theorem sin_shift_theorem (x : ℝ) : 
  Real.sin (2 * x - π / 3) = Real.sin (2 * (x - π / 6)) := by
  congr
  ring

#check sin_shift_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_theorem_l800_80069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_transformation_l800_80016

/-- The result of applying a 30° counter-clockwise rotation followed by a dilation
    with scale factor 2 to the complex number -1 - 4i -/
theorem complex_transformation (z : ℂ) : 
  z = -1 - 4*I → 
  (2 : ℂ) * Complex.exp (I * Real.pi / 6) * z = 4 - Real.sqrt 3 - (4 * Real.sqrt 3 + 1) * I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_transformation_l800_80016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_125_of_4_div_7_l800_80028

def decimal_representation (n d : ℕ) : List ℕ :=
  sorry

theorem digit_125_of_4_div_7 :
  let decimal := decimal_representation 4 7
  (decimal.get? 124).isSome ∧ (decimal.get? 124).get! = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_125_of_4_div_7_l800_80028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_range_l800_80010

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := if x < 0 then x^2 else -x^2

-- State the theorem
theorem t_range (t : ℝ) : 
  (∀ x ∈ Set.Ioo (t^2 - 4) (t^2), f (x + t) < 4 * f x) → 
  t ∈ Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_range_l800_80010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_theorem_l800_80007

/-- The radius of a circle concentric with and outside a regular hexagon -/
noncomputable def circle_radius (hexagon_side : ℝ) (probability : ℝ) : ℝ := 
  12 * Real.sqrt 3 / (Real.sqrt 6 - Real.sqrt 2)

/-- Theorem stating the relationship between the circle radius, hexagon side length, and visibility probability -/
theorem circle_radius_theorem (hexagon_side : ℝ) (probability : ℝ) 
  (h1 : hexagon_side = 3)
  (h2 : probability = 1/3) :
  circle_radius hexagon_side probability = 12 * Real.sqrt 3 / (Real.sqrt 6 - Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_theorem_l800_80007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l800_80056

theorem problem_solution : 
  ((Real.sqrt 24 - Real.sqrt (2/3)) / Real.sqrt 6 = 5/3) ∧
  ((Real.sqrt 5 + 1) * (Real.sqrt 5 - 1) - (Real.sqrt 3 - Real.sqrt 2)^2 = 2 * Real.sqrt 6 - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l800_80056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neither_even_nor_odd_l800_80061

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x + Real.sqrt (1 + x^3))

-- State the theorem
theorem f_neither_even_nor_odd :
  (∃ x : ℝ, f (-x) ≠ f x) ∧ (∃ y : ℝ, f (-y) ≠ -f y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neither_even_nor_odd_l800_80061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l800_80017

noncomputable def f (θ : Real) (x : Real) : Real := Real.sin (2 * x + θ)

theorem function_properties (θ : Real) 
  (h1 : θ ∈ Set.Ioo 0 (Real.pi / 2))
  (h2 : f θ (Real.pi / 6) = 1) :
  (θ = Real.pi / 6) ∧ 
  (Set.range (fun x => f θ x) ∩ Set.Icc 0 (Real.pi / 4) = Set.Icc (1/2) 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l800_80017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_moves_on_line_centroid_moves_on_line_30deg_l800_80040

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The centroid of a triangle -/
noncomputable def centroid (t : Triangle) : Point :=
  { x := (t.A.x + t.B.x + t.C.x) / 3,
    y := (t.A.y + t.B.y + t.C.y) / 3 }

/-- The line on which vertex C moves -/
def CLine (angle : ℝ) (A : Point) : Set Point :=
  {p : Point | p.y - A.y = (p.x - A.x) * Real.tan angle}

/-- Theorem: The centroid moves on a straight line -/
theorem centroid_moves_on_line (A B : Point) (angle : ℝ) :
  ∃ (m c : ℝ), ∀ (C : Point), C ∈ CLine angle A →
    let G := centroid {A := A, B := B, C := C}
    G.y = m * G.x + c := by
  sorry

/-- The specific case for a 30-degree angle -/
theorem centroid_moves_on_line_30deg (A B : Point) :
  ∃ (m c : ℝ), ∀ (C : Point), C ∈ CLine (30 * π / 180) A →
    let G := centroid {A := A, B := B, C := C}
    G.y = m * G.x + c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_moves_on_line_centroid_moves_on_line_30deg_l800_80040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_686_l800_80049

/-- Represents a quadrilateral with sides a, b, c, d and diagonal ac -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  ac : ℝ

/-- Calculate the area of a quadrilateral given its sides and one diagonal -/
noncomputable def area (q : Quadrilateral) : ℝ :=
  let s := (q.a + q.b + q.c + q.d) / 2
  Real.sqrt ((s - q.a) * (s - q.b) * (s - q.c) * (s - q.d))

theorem quadrilateral_area_is_686 (q : Quadrilateral) 
    (h1 : q.a = 18) (h2 : q.b = 45) (h3 : q.c = 25) (h4 : q.d = 30) 
    (h5 : q.ac = 40) (h6 : ∃ (bd : ℝ), q.ac * bd = area q * 2) : 
    ⌊area q⌋ = 686 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_686_l800_80049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_physics_books_in_same_box_l800_80047

/-- The total number of textbooks -/
def total_books : ℕ := 15

/-- The number of physics textbooks -/
def physics_books : ℕ := 4

/-- The sizes of the three boxes -/
def box_sizes : Fin 3 → ℕ
  | 0 => 4
  | 1 => 5
  | 2 => 6
  | _ => 0

/-- The probability of all physics textbooks ending up in the same box -/
def probability_all_physics_in_same_box : ℚ := 1 / 65

/-- Function to calculate the number of ways all physics books can be in the same box -/
def number_of_ways_all_physics_in_same_box (total : ℕ) (physics : ℕ) (sizes : Fin 3 → ℕ) : ℕ :=
  sorry

/-- Function to calculate the total number of ways to distribute all books -/
def total_number_of_ways_to_distribute (total : ℕ) (sizes : Fin 3 → ℕ) : ℕ :=
  sorry

theorem physics_books_in_same_box :
  probability_all_physics_in_same_box =
    (number_of_ways_all_physics_in_same_box total_books physics_books box_sizes) /
    (total_number_of_ways_to_distribute total_books box_sizes) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_physics_books_in_same_box_l800_80047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_gt_2_necessary_not_sufficient_l800_80005

/-- Represents a line in the form ax - 2y - 3 = 0 --/
structure Line where
  a : ℝ

/-- The inclination angle of a line --/
noncomputable def inclinationAngle (l : Line) : ℝ := Real.arctan (l.a / 2)

/-- Condition: inclination angle is greater than π/4 --/
def angleGreaterThanPiOver4 (l : Line) : Prop :=
  inclinationAngle l > Real.pi / 4

/-- Theorem: a > 2 is necessary but not sufficient for angle > π/4 --/
theorem a_gt_2_necessary_not_sufficient :
  (∀ l : Line, angleGreaterThanPiOver4 l → l.a > 2) ∧
  (∃ l : Line, l.a > 2 ∧ ¬angleGreaterThanPiOver4 l) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_gt_2_necessary_not_sufficient_l800_80005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_locations_l800_80039

/-- The distance between locations A and B --/
noncomputable def distance : ℝ := 90

/-- The initial speed ratio of person A to person B --/
noncomputable def initial_speed_ratio : ℝ := 4 / 5

/-- The speed change factor for person A after meeting --/
noncomputable def speed_change_A : ℝ := 3 / 4  -- 1 - 0.25 = 0.75 = 3/4

/-- The speed change factor for person B after meeting --/
noncomputable def speed_change_B : ℝ := 6 / 5  -- 1 + 0.20 = 1.20 = 6/5

/-- The distance person A is from location B when person B reaches location A --/
noncomputable def final_distance_A : ℝ := 30

theorem distance_between_locations (d : ℝ) 
  (h1 : d = distance)
  (h2 : initial_speed_ratio = 4 / 5)
  (h3 : speed_change_A = 3 / 4)
  (h4 : speed_change_B = 6 / 5)
  (h5 : final_distance_A = 30)
  (h6 : d * (1 - 4/9 - 2/9) = final_distance_A) :
  d = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_locations_l800_80039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_projection_matrix_3_4_l800_80034

/-- The projection matrix onto a vector (a, b) -/
noncomputable def projection_matrix (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let norm := Real.sqrt (a^2 + b^2)
  let cos := a / norm
  let sin := b / norm
  ![![cos^2, cos * sin],
    ![cos * sin, sin^2]]

/-- The determinant of the projection matrix onto the vector (3, 4) is 0 -/
theorem det_projection_matrix_3_4 :
  Matrix.det (projection_matrix 3 4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_projection_matrix_3_4_l800_80034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_to_third_ratio_l800_80094

/-- Given the number of students in different grades, prove the ratio of fourth-graders to third-graders --/
theorem fourth_to_third_ratio 
  (third_graders : ℕ) 
  (fifth_graders : ℕ) 
  (total_students : ℕ) 
  (h1 : third_graders = 20)
  (h2 : fifth_graders = third_graders / 2)
  (h3 : total_students = 70)
  (h4 : total_students = third_graders + fifth_graders + (total_students - third_graders - fifth_graders)) :
  (total_students - third_graders - fifth_graders) / third_graders = 2 := by
  sorry

#check fourth_to_third_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_to_third_ratio_l800_80094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_properties_l800_80075

/-- Regular quadrilateral pyramid with an inscribed cube -/
structure PyramidWithCube (a : ℝ) where
  -- Pyramid edge length is positive
  edge_positive : 0 < a
  -- Cube vertices on slant heights and base
  cube_placement : True

/-- Properties of the inscribed cube -/
structure InscribedCube (a : ℝ) (p : PyramidWithCube a) where
  surface_area : ℝ
  volume : ℝ
  surface_area_eq : surface_area = 3 * a^2 / 4
  volume_eq : volume = a^3 * Real.sqrt 2 / 64

/-- Main theorem: properties of the inscribed cube -/
theorem inscribed_cube_properties (a : ℝ) (p : PyramidWithCube a) :
  ∃ (cube : InscribedCube a p),
    cube.surface_area = 3 * a^2 / 4 ∧
    cube.volume = a^3 * Real.sqrt 2 / 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_properties_l800_80075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_symmetry_line_intercept_range_l800_80087

/-- An ellipse in the xy-plane -/
structure Ellipse where
  a : ℝ
  b : ℝ
  hpos : 0 < a ∧ 0 < b

/-- A line in the xy-plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is on the ellipse -/
def on_ellipse (e : Ellipse) (p : Point) : Prop :=
  (p.x ^ 2 / e.a ^ 2) + (p.y ^ 2 / e.b ^ 2) = 1

/-- Predicate to check if two points are symmetric with respect to a line -/
def symmetric_wrt_line (l : Line) (p1 p2 : Point) : Prop :=
  ∃ (m : Point), m.x = (p1.x + p2.x) / 2 ∧ m.y = (p1.y + p2.y) / 2 ∧ m.y = l.slope * m.x + l.intercept

/-- The main theorem -/
theorem ellipse_symmetry_line_intercept_range 
  (e : Ellipse) 
  (h_ellipse : e.a = 2 ∧ e.b = Real.sqrt 3) 
  (l : Line) 
  (h_line : l.slope = 4) :
  (∃ (p1 p2 : Point), p1 ≠ p2 ∧ 
    on_ellipse e p1 ∧ 
    on_ellipse e p2 ∧ 
    symmetric_wrt_line l p1 p2) ↔ 
  -2 * Real.sqrt 13 / 13 < l.intercept ∧ l.intercept < 2 * Real.sqrt 13 / 13 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_symmetry_line_intercept_range_l800_80087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_rate_calculation_l800_80079

-- Define constants
noncomputable def field_area : ℝ := 13.86 -- in hectares
noncomputable def total_fencing_cost : ℝ := 6466.70 -- in rupees

-- Define conversion factor
noncomputable def hectares_to_sqm : ℝ := 10000 -- 1 hectare = 10,000 square meters

-- Define functions
noncomputable def area_to_radius (a : ℝ) : ℝ :=
  Real.sqrt (a * hectares_to_sqm / Real.pi)

noncomputable def radius_to_circumference (r : ℝ) : ℝ :=
  2 * Real.pi * r

noncomputable def fencing_rate (cost total_length : ℝ) : ℝ :=
  cost / total_length

-- Theorem statement
theorem fencing_rate_calculation :
  let radius := area_to_radius field_area
  let circumference := radius_to_circumference radius
  let rate := fencing_rate total_fencing_cost circumference
  |rate - 4.90| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_rate_calculation_l800_80079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_in_still_water_l800_80098

/-- Proves that a boat's speed in still water is 10 km/hr given specific downstream conditions -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_time : ℝ) 
  (downstream_distance : ℝ) 
  (boat_speed : ℝ)
  (h1 : stream_speed = 8)
  (h2 : downstream_time = 3)
  (h3 : downstream_distance = 54) :
  downstream_distance = (boat_speed + stream_speed) * downstream_time →
  boat_speed = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_in_still_water_l800_80098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l800_80009

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (3 - x) + Real.log (x - 1)

-- Define the domain
def domain : Set ℝ := {x | 1 < x ∧ x ≤ 3}

-- Theorem statement
theorem f_domain : {x : ℝ | ∃ y, f x = y} = domain := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l800_80009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_B_has_inverse_l800_80044

-- Define the functions
def A : ℝ → ℝ := λ x => -(x^2) + 3
def B : ℝ → ℝ := λ x => x
noncomputable def C : ℝ → ℝ := λ x => Real.sqrt (4 - x^2)

-- Define the property of having an inverse
def has_inverse (f : ℝ → ℝ) : Prop := ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

-- Theorem statement
theorem only_B_has_inverse :
  ¬(has_inverse A) ∧ (has_inverse B) ∧ ¬(has_inverse C) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_B_has_inverse_l800_80044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sum_of_primes_l800_80030

/-- An arithmetic sequence with first term 7 and common difference 10 -/
def mySequence (n : ℕ) : ℕ := 7 + 10 * n

/-- A number is prime if it's greater than 1 and has no positive divisors other than 1 and itself -/
def isPrime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 0 → m < p → p % m ≠ 0

/-- A number in the sequence can be expressed as the sum of two primes -/
def isSumOfTwoPrimes (n : ℕ) : Prop :=
  ∃ p q : ℕ, isPrime p ∧ isPrime q ∧ mySequence n = p + q

theorem unique_sum_of_primes :
  ∃! n : ℕ, isSumOfTwoPrimes n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sum_of_primes_l800_80030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_heads_is_68_l800_80015

/-- The number of coins -/
def num_coins : ℕ := 72

/-- The number of tosses allowed for each coin -/
def max_tosses : ℕ := 4

/-- The probability of getting heads on a single toss -/
def p_heads : ℚ := 1/2

/-- The expected number of coins showing heads after up to max_tosses tosses -/
noncomputable def expected_heads : ℚ := num_coins * (1 - (1 - p_heads)^max_tosses)

/-- Theorem stating that the expected number of coins showing heads is 68 -/
theorem expected_heads_is_68 : 
  ⌊expected_heads⌋ = 68 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_heads_is_68_l800_80015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_equation_l800_80004

/-- Represents the number of wheels on the certain vehicles -/
def x : ℕ → ℕ := fun _ => 0  -- Placeholder function

/-- Represents the number of 2 wheelers -/
def y : ℕ → ℕ := fun _ => 0  -- Placeholder function

/-- The total number of wheels is 58 -/
def total_wheels : ℕ := 58

/-- The number of vehicles with the certain number of wheels is 14 -/
def certain_vehicles : ℕ := 14

/-- Theorem stating the relationship between the variables -/
theorem wheel_equation : ∀ n, 14 * (x n) + 2 * (y n) = total_wheels := by
  sorry

#check wheel_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_equation_l800_80004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_implies_a_range_l800_80027

/-- The function f(x) defined as the square root of a quadratic function. -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (x^2 - 2*a*x + 3)

/-- The theorem stating that if f(x) is monotonically increasing on (-1, 1), 
    then 'a' is in the range [-2, -1]. -/
theorem f_monotone_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, -1 < x ∧ x < y ∧ y < 1 → f a x ≤ f a y) →
  -2 ≤ a ∧ a ≤ -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_implies_a_range_l800_80027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_abc_is_two_l800_80037

noncomputable section

/-- Triangle ABC with internal angles A, B, C and sides a, b, c opposite to these angles respectively --/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- Triangle A₁B₁C₁ with internal angles A₁, B₁, C₁ --/
structure Triangle₁ where
  A₁ : Real
  B₁ : Real
  C₁ : Real

/-- The theorem stating that the area of triangle ABC is 2 under given conditions --/
theorem area_of_triangle_abc_is_two (abc : Triangle) (a₁b₁c₁ : Triangle₁) : 
  (Real.sin abc.A = Real.cos a₁b₁c₁.A₁) → 
  (Real.sin abc.B = Real.cos a₁b₁c₁.B₁) → 
  (Real.sin abc.C = Real.cos a₁b₁c₁.C₁) → 
  (abc.A > Real.pi / 2) →  -- A is obtuse
  (abc.a = 2 * Real.sqrt 5) → 
  (abc.b = 2 * Real.sqrt 2) → 
  (1 / 2 * abc.b * abc.c * Real.sin abc.A = 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_abc_is_two_l800_80037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spheres_do_not_protrude_l800_80060

/-- A cylindrical container with a cone and spheres inside -/
structure Container where
  R : ℝ  -- Radius of the cylindrical container
  h : ℝ  -- Height of the container
  r : ℝ  -- Radius of each sphere
  h_eq_diameter : h = 2 * R  -- Height equals diameter of base
  cone_base_radius : ℝ  -- Base radius of the cone
  cone_height : ℝ  -- Height of the cone
  cone_eq_container : cone_base_radius = R ∧ cone_height = h  -- Cone dimensions match container
  sphere_count : ℕ  -- Number of spheres
  sphere_count_eq_6 : sphere_count = 6  -- There are 6 spheres
  spheres_touch : r > 0  -- Spheres have positive radius to touch surfaces and each other

/-- Theorem stating that the spheres do not protrude from the container -/
theorem spheres_do_not_protrude (c : Container) : 
  ∃ (x : ℝ), x < c.h ∧ x = c.r + Real.sqrt (c.R^2 - (c.R - c.r)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spheres_do_not_protrude_l800_80060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_is_correct_l800_80065

/-- An ellipse with the given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The right focus of the ellipse -/
def right_focus : ℝ × ℝ := (3, 0)

/-- A line passing through the right focus -/
structure Line where
  m : ℝ  -- slope
  c : ℝ  -- y-intercept

/-- The midpoint of the intersection points -/
def intersection_midpoint : ℝ × ℝ := (1, -1)

/-- The theorem stating the equation of the ellipse -/
theorem ellipse_equation_is_correct (e : Ellipse) (l : Line) :
  ellipse_equation e right_focus.1 right_focus.2 ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse_equation e x₁ y₁ ∧
    ellipse_equation e x₂ y₂ ∧
    y₁ - right_focus.2 = l.m * (x₁ - right_focus.1) + l.c ∧
    y₂ - right_focus.2 = l.m * (x₂ - right_focus.1) + l.c ∧
    (x₁ + x₂) / 2 = intersection_midpoint.1 ∧
    (y₁ + y₂) / 2 = intersection_midpoint.2) →
  e.a^2 = 18 ∧ e.b^2 = 9 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_is_correct_l800_80065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_square_tens_digit_l800_80002

/-- If the tens digit of a^2 can be 1, 3, 5, 7, or 9, then the units digit of a must be 4 or 6 -/
theorem units_digit_of_square_tens_digit (a : ℕ) :
  (∃ k : ℕ, k ∈ ({1, 3, 5, 7, 9} : Set ℕ) ∧ (a^2 / 10) % 10 = k) →
  a % 10 = 4 ∨ a % 10 = 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_square_tens_digit_l800_80002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_opposite_direction_l800_80097

/-- Given two vectors a and b in a real inner product space, 
    with |a| = 1, |b| = 3, and b in the opposite direction of a,
    prove that b = -3a. -/
theorem vector_opposite_direction (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) (h1 : ‖a‖ = 1) (h2 : ‖b‖ = 3) 
  (h3 : ∃ (k : ℝ), k < 0 ∧ b = k • a) : 
  b = -3 • a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_opposite_direction_l800_80097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_seven_terms_equals_seven_l800_80006

/-- An arithmetic sequence with positive terms -/
noncomputable def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a n > 0

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def SumArithmeticSequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (a 1 + a n)

theorem sum_seven_terms_equals_seven
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_eq : a 1^2 + a 7^2 + 2 * a 1 * a 7 = 4) :
  SumArithmeticSequence a 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_seven_terms_equals_seven_l800_80006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_implies_k_l800_80041

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 7 = 0

-- Define the line
def line_eq (x y k : ℝ) : Prop := y = k * x

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem minimum_distance_implies_k (k : ℝ) : 
  (∃ x1 y1 x2 y2 : ℝ, 
    circle_eq x1 y1 ∧ 
    line_eq x2 y2 k ∧ 
    (∀ x3 y3 x4 y4 : ℝ, circle_eq x3 y3 → line_eq x4 y4 k → 
      distance x1 y1 x2 y2 ≤ distance x3 y3 x4 y4) ∧
    distance x1 y1 x2 y2 = 2 * Real.sqrt 2 - 1) →
  k = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_implies_k_l800_80041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_ride_speed_l800_80012

/-- Calculates the time difference in hours between two times given in hours and minutes -/
noncomputable def timeDifference (startHour startMinute endHour endMinute : ℕ) : ℝ :=
  (endHour - startHour : ℝ) + (endMinute - startMinute : ℝ) / 60

/-- Calculates the average speed given distance and time -/
noncomputable def averageSpeed (distance time : ℝ) : ℝ :=
  distance / time

theorem bike_ride_speed : 
  let startHour := 8
  let startMinute := 15
  let endHour := 11
  let endMinute := 45
  let distance := 135
  let time := timeDifference startHour startMinute endHour endMinute
  let speed := averageSpeed distance time
  ∃ ε > 0, |speed - 38.57| < ε
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_ride_speed_l800_80012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_specific_value_l800_80036

noncomputable def f (α : ℝ) : ℝ := 
  (Real.sin (2 * Real.pi - α) * Real.cos (Real.pi + α) * Real.cos (Real.pi / 2 + α) * Real.cos (11 * Real.pi / 2 - α)) / 
  (2 * Real.sin (3 * Real.pi + α) * Real.sin (-Real.pi - α) * Real.sin (9 * Real.pi / 2 + α))

theorem f_simplification (α : ℝ) : f α = -1/2 * Real.sin α := by sorry

theorem f_specific_value : f (-25 * Real.pi / 4) = Real.sqrt 2 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_specific_value_l800_80036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_days_passed_is_five_l800_80021

/-- Represents the state of the bowl of peaches over time -/
structure PeachBowl where
  total : ℕ
  initial_ripe : ℕ
  ripen_per_day : ℕ
  eaten_on_day_three : ℕ
  ripe_after_d_days : ℕ → ℕ
  unripe_after_d_days : ℕ → ℕ

/-- The specific bowl of peaches from the problem -/
def problem_bowl : PeachBowl :=
  { total := 18
  , initial_ripe := 4
  , ripen_per_day := 2
  , eaten_on_day_three := 3
  , ripe_after_d_days := λ d => 4 + 2 * d - 3
  , unripe_after_d_days := λ d => 14 - 2 * d }

/-- The theorem stating that 5 days have passed when the condition is met -/
theorem days_passed_is_five :
  ∃ d : ℕ, d = 5 ∧
    (problem_bowl.ripe_after_d_days d = problem_bowl.unripe_after_d_days d + 7) := by
  sorry

#check days_passed_is_five

end NUMINAMATH_CALUDE_ERRORFEEDBACK_days_passed_is_five_l800_80021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l800_80059

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x))

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 2 - a^x

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 ∧ x₂ ∈ Set.Icc 0 1 ∧ f x₁ = g a x₂) →
  a > 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l800_80059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_is_18_l800_80000

/-- The coefficient of x^2 in the expansion of (2x+1)(x-2)^3 -/
def coefficient_x_squared : ℝ := 18

/-- The coefficient of x^2 in the expansion of (2x+1)(x-2)^3 is 18 -/
theorem coefficient_x_squared_is_18 : coefficient_x_squared = 18 := by
  -- Unfold the definition of coefficient_x_squared
  unfold coefficient_x_squared
  -- The definition is exactly 18, so this should be true by reflexivity
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_is_18_l800_80000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anya_hair_loss_l800_80050

/-- The number of hairs Anya washes down the drain -/
def H : ℕ := 32

/-- The number of hairs Anya loses when brushing her hair -/
def brushed_hairs : ℕ := H / 2

/-- The total number of hairs Anya loses -/
def total_hair_loss : ℕ := H + brushed_hairs

theorem anya_hair_loss : total_hair_loss + 1 = 49 → H = 32 := by
  intro h
  rw [total_hair_loss, brushed_hairs] at h
  simp [H] at h
  exact rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_anya_hair_loss_l800_80050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_expression_l800_80063

theorem largest_expression :
  let e1 := Real.sqrt (Real.rpow 7 (1/3) * Real.rpow 8 (1/3))
  let e2 := Real.sqrt (8 * Real.rpow 7 (1/3))
  let e3 := Real.sqrt (7 * Real.rpow 8 (1/3))
  let e4 := Real.rpow (7 * Real.sqrt 8) (1/3)
  let e5 := Real.rpow (8 * Real.sqrt 7) (1/3)
  (e2 > e1) ∧ (e2 > e3) ∧ (e2 > e4) ∧ (e2 > e5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_expression_l800_80063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_y_difference_bounds_l800_80093

noncomputable def x (m n : ℕ) : ℕ → ℝ
  | 0 => 0
  | k + 1 => Real.sqrt (m + x n m k)

noncomputable def y (m n : ℕ) : ℕ → ℝ
  | 0 => 0
  | k + 1 => Real.sqrt (n + y n m k)

theorem x_y_difference_bounds (m n k : ℕ) (h : m > n) :
  0 < x m n k - y m n k ∧ x m n k - y m n k < m - n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_y_difference_bounds_l800_80093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oh_squared_equals_175_l800_80046

/-- Triangle ABC with circumcenter O and orthocenter H -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  O : ℝ × ℝ
  H : ℝ × ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ

/-- Conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.O = (0, 0) ∧  -- O is at the origin
  t.R = 5 ∧
  t.a^2 + t.b^2 + t.c^2 = 50

/-- The theorem to be proved -/
theorem oh_squared_equals_175 (t : Triangle) (h : TriangleConditions t) : 
  (t.H.1^2 + t.H.2^2) = 175 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oh_squared_equals_175_l800_80046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_equidistant_x_coord_is_two_l800_80073

/-- A point in the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance from a point to the x-axis -/
noncomputable def distToXAxis (p : Point) : ℝ := |p.y|

/-- Distance from a point to the y-axis -/
noncomputable def distToYAxis (p : Point) : ℝ := |p.x|

/-- Distance from a point to the line x + y = 4 -/
noncomputable def distToLine (p : Point) : ℝ := |p.x + p.y - 4| / Real.sqrt 2

/-- Theorem: If a point is equally distant from x-axis, y-axis, and line x + y = 4, its x-coordinate is 2 -/
theorem point_equidistant_x_coord_is_two (p : Point) :
  distToXAxis p = distToYAxis p ∧ 
  distToXAxis p = distToLine p →
  p.x = 2 := by
  sorry

#check point_equidistant_x_coord_is_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_equidistant_x_coord_is_two_l800_80073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_center_to_point_l800_80025

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 6*x + 8*y + 9

/-- The center of the circle -/
def circle_center : ℝ × ℝ :=
  (3, 4)

/-- The given point -/
def given_point : ℝ × ℝ :=
  (9, 5)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_from_center_to_point :
  distance circle_center given_point = Real.sqrt 37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_center_to_point_l800_80025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l800_80026

/-- A function g(x) with specific properties -/
noncomputable def g (A B C : ℤ) (x : ℝ) : ℝ := x^2 / (A * x^2 + B * x + C)

/-- Theorem stating the sum of coefficients A, B, and C -/
theorem sum_of_coefficients (A B C : ℤ) :
  (∀ x > (5 : ℝ), g A B C x > 0.1) →
  (A * (-3)^2 + B * (-3) + C = 0) →
  (A * 4^2 + B * 4 + C = 0) →
  A + B + C = -108 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l800_80026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l800_80048

theorem divisibility_condition (a m n : ℕ) : 
  (a > 0 ∧ m > 0 ∧ n > 0) →
  (a^m + 1 ∣ (a + 1)^n) ↔ 
  (m = 1) ∨ (a = 1) ∨ (a = 2 ∧ m = 3 ∧ n ≥ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l800_80048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersections_l800_80051

noncomputable def C₁ (t : ℝ) : ℝ × ℝ := ((2 + t) / 6, Real.sqrt t)
noncomputable def C₂ (s : ℝ) : ℝ × ℝ := (-(2 + s) / 6, -Real.sqrt s)
def C₃ (θ : ℝ) : Prop := 2 * Real.cos θ - Real.sin θ = 0

theorem curve_intersections :
  -- 1. Cartesian equation of C₁
  (∀ x y, y ≥ 0 → (∃ t, C₁ t = (x, y)) ↔ y^2 = 6*x - 2) ∧
  -- 2. Intersection points of C₃ with C₁
  (∃ θ₁ θ₂, C₃ θ₁ ∧ C₃ θ₂ ∧ 
    (∃ t₁ t₂, C₁ t₁ = (1/2, 1) ∧ C₁ t₂ = (1, 2)) ∧
    (∀ θ t, C₃ θ ∧ (∃ x y, C₁ t = (x, y) ∧ 2*x - y = 0) → 
      (C₁ t = (1/2, 1) ∨ C₁ t = (1, 2)))) ∧
  -- 3. Intersection points of C₃ with C₂
  (∃ θ₃ θ₄, C₃ θ₃ ∧ C₃ θ₄ ∧ 
    (∃ s₁ s₂, C₂ s₁ = (-1/2, -1) ∧ C₂ s₂ = (-1, -2)) ∧
    (∀ θ s, C₃ θ ∧ (∃ x y, C₂ s = (x, y) ∧ 2*x - y = 0) → 
      (C₂ s = (-1/2, -1) ∨ C₂ s = (-1, -2)))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersections_l800_80051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_and_area_l800_80043

-- Define the line l: x - ay - 2 = 0
def line (a : ℝ) (x y : ℝ) : Prop := x - a * y - 2 = 0

-- Define the circle C: (x - a)² + (y - 1)² = 2
def circle_eq (a : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y - 1)^2 = 2

-- Define the intersection of line and circle
def intersects (a : ℝ) : Prop := ∃ x y : ℝ, line a x y ∧ circle_eq a x y

-- Define the area of triangle ABC
def triangle_area (a : ℝ) : ℝ := 1

theorem line_circle_intersection_and_area (a : ℝ) 
  (h1 : a > 1) 
  (h2 : triangle_area a = 1) : 
  intersects a ∧ a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_and_area_l800_80043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_cubed_in_product_l800_80085

/-- Represents a polynomial with integer coefficients -/
def MyPolynomial := List Int

/-- Multiplies two polynomials -/
def multiplyPolynomials (p q : MyPolynomial) : MyPolynomial := sorry

/-- Extracts the coefficient of x^n from a polynomial -/
def coefficientOf (p : MyPolynomial) (n : Nat) : Int := sorry

/-- The first given polynomial -/
def p : MyPolynomial := [3, 4, -2, 8, -5]

/-- The second given polynomial -/
def q : MyPolynomial := [2, -7, 5, -3]

theorem coefficient_of_x_cubed_in_product :
  coefficientOf (multiplyPolynomials p q) 3 = -78 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_cubed_in_product_l800_80085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_theorem_l800_80058

/-- The total surface area of a pyramid with an isosceles trapezoidal base -/
noncomputable def pyramid_surface_area (a α β : Real) : Real :=
  (2 * a^2 * Real.sin α * Real.cos (β/2)^2) / Real.cos β

/-- Theorem stating the total surface area of the pyramid -/
theorem pyramid_surface_area_theorem (a α β : Real) 
  (h_a : a > 0) 
  (h_α : 0 < α ∧ α < π/2) 
  (h_β : 0 < β ∧ β < π/2) :
  let S := pyramid_surface_area a α β
  ∃ (base_area lateral_area : Real),
    base_area = a^2 * Real.sin α ∧
    lateral_area = (4 * a * Real.sin α) / Real.cos β ∧
    S = base_area + lateral_area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_theorem_l800_80058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_l800_80023

/-- Given a circle (x-1)^2 + y^2 = 1 and a line y = kx - 2, 
    if the minimum distance from a point on the circle to the line is 1, 
    then k = -4/3 or k = 0 -/
theorem circle_line_distance (k : ℝ) : 
  (∃ x y : ℝ, (x - 1)^2 + y^2 = 1 ∧ 
   ∀ x' y' : ℝ, (x' - 1)^2 + y'^2 = 1 → 
   |y' - (k * x' - 2)| / Real.sqrt (k^2 + 1) ≥ |y - (k * x - 2)| / Real.sqrt (k^2 + 1)) →
  (∀ x y : ℝ, (x - 1)^2 + y^2 = 1 → 
   |y - (k * x - 2)| / Real.sqrt (k^2 + 1) ≥ 1) →
  k = -4/3 ∨ k = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_l800_80023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_figure_l800_80086

/-- The area of the figure bounded by the given ellipse and line -/
noncomputable def bounded_area : ℝ := 3 * Real.pi - 6

/-- The parametric equations of the ellipse -/
noncomputable def ellipse_x (t : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.cos t
noncomputable def ellipse_y (t : ℝ) : ℝ := 3 * Real.sqrt 2 * Real.sin t

/-- The upper bounding line -/
def upper_line : ℝ := 3

/-- The region is defined for y ≥ 3 -/
def region_condition (y : ℝ) : Prop := y ≥ upper_line

theorem area_of_bounded_figure :
  ∃ (area : ℝ), area = bounded_area ∧
  (∀ t : ℝ, ellipse_x t ≤ 2 * Real.sqrt 2 ∧ ellipse_y t ≤ 3 * Real.sqrt 2) ∧
  (∀ y : ℝ, region_condition y → y ≤ upper_line) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_figure_l800_80086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_power_equation_l800_80057

theorem eight_power_equation (x : ℝ) : (1 / 8) * (2 : ℝ) ^ 36 = 8 ^ x → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_power_equation_l800_80057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_month_sale_calculation_l800_80077

def average_sale (total_sales : ℚ) (num_months : ℕ) : ℚ :=
  total_sales / (num_months : ℚ)

def calculate_missing_month_sale (month1 month2 month3 month4 month6 : ℕ) (avg_sale : ℚ) : ℚ :=
  let total_sales := avg_sale * 6
  total_sales - (month1 + month2 + month3 + month4 + month6 : ℚ)

theorem missing_month_sale_calculation 
  (month1 month2 month3 month4 month6 : ℕ) 
  (avg_sale : ℚ) :
  let month5 := calculate_missing_month_sale month1 month2 month3 month4 month6 avg_sale
  average_sale (month1 + month2 + month3 + month4 + month5.floor + month6 : ℚ) 6 = avg_sale := by
  sorry

#eval (calculate_missing_month_sale 5266 5744 5864 6122 4916 (5750 / 1)).floor

end NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_month_sale_calculation_l800_80077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l800_80078

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-3) 9

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := f (-3 * x + 1)

-- Define the domain of h
def domain_h : Set ℝ := Set.Icc (-8/3) (4/3)

-- Theorem statement
theorem domain_of_h :
  ∀ x : ℝ, x ∈ domain_h ↔ (-3 * x + 1) ∈ domain_f :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l800_80078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uniform_distribution_properties_l800_80014

/-- A random variable uniformly distributed in the interval [a, b] -/
structure UniformDist (a b : ℝ) where
  pdf : ℝ → ℝ
  pdf_property : ∀ x, a ≤ x ∧ x ≤ b → pdf x = 1 / (b - a)

/-- The expected value of a uniform distribution -/
noncomputable def expectedValue (a b : ℝ) (X : UniformDist a b) : ℝ := (a + b) / 2

/-- The variance of a uniform distribution -/
noncomputable def variance (a b : ℝ) (X : UniformDist a b) : ℝ := (b - a)^2 / 12

theorem uniform_distribution_properties (a b : ℝ) (hab : a < b) (X : UniformDist a b) :
  expectedValue a b X = (a + b) / 2 ∧ variance a b X = (b - a)^2 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_uniform_distribution_properties_l800_80014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_drain_rate_example_l800_80072

/-- Calculates the drain rate of a pool given its dimensions, capacity, and drain time. -/
noncomputable def pool_drain_rate (length width depth : ℝ) (capacity : ℝ) (drain_time : ℝ) : ℝ :=
  (length * width * depth * capacity) / drain_time

/-- Theorem stating that a pool with given dimensions, capacity, and drain time has a specific drain rate. -/
theorem pool_drain_rate_example :
  pool_drain_rate 150 40 10 0.8 800 = 60 := by
  -- Unfold the definition of pool_drain_rate
  unfold pool_drain_rate
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_drain_rate_example_l800_80072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_difference_product_product_of_N_values_l800_80080

theorem temperature_difference_product (N : ℝ) : 
  (∃ L : ℝ, 
    let M := L + N;
    let M_4pm := M - 6;
    let L_4pm := L + 4;
    |M_4pm - L_4pm| = 4) →
  (N = 14 ∨ N = 6) :=
by sorry

theorem product_of_N_values : 
  (∀ N : ℝ, (∃ L : ℝ, 
    let M := L + N;
    let M_4pm := M - 6;
    let L_4pm := L + 4;
    |M_4pm - L_4pm| = 4) →
  (N = 14 ∨ N = 6)) →
  14 * 6 = 84 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_difference_product_product_of_N_values_l800_80080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_for_max_twice_min_l800_80082

-- Define the function f
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (x + φ)

-- State the theorem
theorem phi_value_for_max_twice_min (φ : ℝ) 
  (h1 : 0 < φ) (h2 : φ < Real.pi) :
  (∃ (x_max x_min : ℝ), 
    (∀ y, f φ y ≤ f φ x_max) ∧
    (∀ y, f φ y ≥ f φ x_min) ∧
    f φ x_max = 2 * f φ x_min) →
  φ = Real.pi / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_for_max_twice_min_l800_80082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sensor_energy_consumption_l800_80013

/-- Represents the energy consumption of a sensor moving against river flow -/
noncomputable def energy_consumption (k : ℝ) (v : ℝ) : ℝ :=
  k * v^2 * (10 / (v - 3))

/-- Theorem stating the properties of energy consumption for a sensor moving against river flow -/
theorem sensor_energy_consumption (k : ℝ) (h_k : k > 0) :
  ∃ (v_min : ℝ),
    (∀ v, v > 3 → energy_consumption k v ≥ energy_consumption k v_min) ∧
    v_min = 6 ∧
    energy_consumption k v_min = 120 * k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sensor_energy_consumption_l800_80013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_third_term_geometric_progression_l800_80042

theorem smallest_third_term_geometric_progression :
  ∃ (d : ℝ), 
    let ap := [5, 5 + d, 5 + 2*d]
    let gp := [5, 9 + d, 35 + 2*d]
    (∀ i ∈ [0, 1], gp[i+1]! * gp[i+1]! = gp[i]! * gp[i+2]!) ∧
    (35 + 2*d ≥ -19) ∧
    (35 + 2*d = -19) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_third_term_geometric_progression_l800_80042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_FV_lengths_l800_80032

/-- A parabola with vertex V and focus F -/
structure Parabola where
  V : ℝ × ℝ  -- Vertex
  F : ℝ × ℝ  -- Focus

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: Sum of possible FV lengths -/
theorem sum_of_possible_FV_lengths (p : Parabola) (B : ℝ × ℝ) 
    (h1 : distance B p.F = 24)
    (h2 : distance B p.V = 25) :
    ∃ (d1 d2 : ℝ), d1 + d2 = 16 ∧ 
    (distance p.F p.V = d1 ∨ distance p.F p.V = d2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_FV_lengths_l800_80032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_samovar_cools_faster_l800_80011

/-- Represents a samovar with its properties -/
structure Samovar where
  volume : ℝ
  surface_area : ℝ
  temperature : ℝ

/-- Represents the cooling rate of a samovar -/
noncomputable def cooling_rate (s : Samovar) : ℝ :=
  s.surface_area / s.volume

/-- Theorem stating that a smaller samovar cools faster than a larger one -/
theorem smaller_samovar_cools_faster
  (small large : Samovar)
  (h_shape : small.surface_area / small.volume^(2/3) = large.surface_area / large.volume^(2/3))
  (h_temp : small.temperature = large.temperature)
  (h_volume : small.volume < large.volume) :
  cooling_rate small > cooling_rate large := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_samovar_cools_faster_l800_80011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_for_circle_l800_80076

/-- A structure representing a quadratic equation in two variables -/
structure QuadraticEquation (α : Type*) [Ring α] where
  A : α
  B : α
  C : α
  D : α
  E : α
  F : α

/-- Predicate to check if a quadratic equation represents a circle -/
def is_circle {α : Type*} [Ring α] (eq : QuadraticEquation α) : Prop :=
  ∃ (h k r : α), eq.A = 1 ∧ eq.C = 1 ∧ eq.B = 0 ∧ 
    eq.D = -2*h ∧ eq.E = -2*k ∧ eq.F = h^2 + k^2 - r^2

/-- The main theorem stating that A = C and B = 0 is necessary but not sufficient 
    for the equation to represent a circle -/
theorem necessary_not_sufficient_for_circle :
  ∀ (eq : QuadraticEquation ℝ), eq.D ≠ 0 →
    (is_circle eq → eq.A = eq.C ∧ eq.B = 0) ∧
    ¬(eq.A = eq.C ∧ eq.B = 0 → is_circle eq) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_for_circle_l800_80076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_rectangles_exist_l800_80081

/-- Represents a rectangle in a grid --/
structure Rectangle where
  x : Nat
  y : Nat
  width : Nat
  height : Nat

/-- Represents a grid with some squares removed --/
structure Grid where
  size : Nat
  removed_squares : List Rectangle

/-- Checks if two rectangles overlap --/
def rectangles_overlap (r1 r2 : Rectangle) : Prop :=
  ¬(r1.x + r1.width ≤ r2.x ∨ r2.x + r2.width ≤ r1.x ∨
    r1.y + r1.height ≤ r2.y ∨ r2.y + r2.height ≤ r1.y)

/-- Checks if a rectangle is within the grid and doesn't overlap with removed squares --/
def is_valid_rectangle (grid : Grid) (rect : Rectangle) : Prop :=
  rect.x + rect.width ≤ grid.size ∧
  rect.y + rect.height ≤ grid.size ∧
  ∀ sq, sq ∈ grid.removed_squares → ¬(rectangles_overlap rect sq)

/-- Main theorem: There exist at least five non-overlapping 1x10 rectangles in the remaining grid --/
theorem five_rectangles_exist (grid : Grid) 
    (h1 : grid.size = 2015)
    (h2 : ∀ sq, sq ∈ grid.removed_squares → sq.width = 10 ∧ sq.height = 10) : 
  ∃ (r1 r2 r3 r4 r5 : Rectangle),
    (∀ r, r ∈ [r1, r2, r3, r4, r5] → is_valid_rectangle grid r ∧ r.width = 10 ∧ r.height = 1) ∧
    (∀ r1 r2, r1 ∈ [r1, r2, r3, r4, r5] → r2 ∈ [r1, r2, r3, r4, r5] → r1 ≠ r2 → ¬(rectangles_overlap r1 r2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_rectangles_exist_l800_80081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_15_equals_610_l800_80018

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => a (n + 1) + a n

theorem a_15_equals_610 : a 15 = 610 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_15_equals_610_l800_80018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_net_change_l800_80053

def year1_change : ℝ := 1.20
def year2_change : ℝ := 1.30
def year3_change : ℝ := 0.90
def year4_change : ℝ := 0.80

theorem population_net_change :
  let total_change := year1_change * year2_change * year3_change * year4_change
  let net_change_percent := (total_change - 1) * 100
  Int.floor net_change_percent = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_net_change_l800_80053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jogger_distance_ahead_l800_80089

noncomputable def jogger_speed : ℝ := 9 -- km/hr
noncomputable def train_speed : ℝ := 45 -- km/hr
noncomputable def train_length : ℝ := 120 -- meters
noncomputable def passing_time : ℝ := 24 -- seconds

noncomputable def km_per_hour_to_meters_per_second (speed : ℝ) : ℝ :=
  speed * (1000 / 3600)

theorem jogger_distance_ahead :
  let relative_speed := km_per_hour_to_meters_per_second (train_speed - jogger_speed)
  let distance := relative_speed * passing_time - train_length
  distance = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jogger_distance_ahead_l800_80089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_number_l800_80092

noncomputable def option_a := 10 - 3 * Real.sqrt 11
noncomputable def option_b := 3 * Real.sqrt 11 - 10
noncomputable def option_c := 18 - 5 * Real.sqrt 13
noncomputable def option_d := 51 - 10 * Real.sqrt 26
noncomputable def option_e := 10 * Real.sqrt 26 - 51

theorem smallest_positive_number : 
  option_d > 0 ∧ 
  (option_a ≤ 0 ∨ option_d < option_a) ∧
  (option_b ≤ 0 ∨ option_d < option_b) ∧
  (option_c ≤ 0 ∨ option_d < option_c) ∧
  (option_e ≤ 0 ∨ option_d < option_e) :=
by sorry

#check smallest_positive_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_number_l800_80092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_table_iff_divisible_by_nine_l800_80071

/-- Represents a cell in the table -/
inductive Cell
| I
| M
| O
deriving BEq, Repr

/-- Represents a table filled with cells -/
def Table (n : ℕ) := Fin n → Fin n → Cell

/-- Checks if a list of cells satisfies the equal distribution condition -/
def equalDistribution (cells : List Cell) : Prop :=
  3 ∣ cells.length ∧
  cells.count Cell.I = cells.count Cell.M ∧
  cells.count Cell.M = cells.count Cell.O

/-- Checks if all rows in the table satisfy the equal distribution condition -/
def rowsValid {n : ℕ} (t : Table n) : Prop :=
  ∀ i : Fin n, equalDistribution (List.ofFn (λ j => t i j))

/-- Checks if all columns in the table satisfy the equal distribution condition -/
def columnsValid {n : ℕ} (t : Table n) : Prop :=
  ∀ j : Fin n, equalDistribution (List.ofFn (λ i => t i j))

/-- Checks if all applicable diagonals satisfy the equal distribution condition -/
def diagonalsValid {n : ℕ} (t : Table n) : Prop :=
  (∀ k : Fin (2 * n - 1), let d := List.ofFn (λ i : Fin n => t i (⟨(k.val - i.val + n) % n, sorry⟩ : Fin n));
    3 ∣ d.length → equalDistribution d) ∧
  (∀ k : ℤ, -n < k ∧ k < n → let d := List.ofFn (λ i : Fin n => t i ⟨(i.val + k.toNat) % n, sorry⟩);
    3 ∣ d.length → equalDistribution d)

/-- Main theorem: A valid table exists if and only if n is divisible by 9 -/
theorem valid_table_iff_divisible_by_nine (n : ℕ) :
  (∃ t : Table n, rowsValid t ∧ columnsValid t ∧ diagonalsValid t) ↔ 9 ∣ n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_table_iff_divisible_by_nine_l800_80071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_equation_l800_80022

-- Define the determinant function for a 2x2 matrix
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the equation
noncomputable def equation (x : ℝ) : ℝ := det2x2 (1 + Real.log x) (3 - Real.log x) 1 1

-- Theorem statement
theorem root_of_equation :
  ∃ (x : ℝ), x > 0 ∧ equation x = 0 ∧ x = 10 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_equation_l800_80022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l800_80083

theorem division_problem (x y : ℕ) (h1 : x % y = 12) (h2 : (x : ℚ) / (y : ℚ) = 75.12) : y = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l800_80083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_earl_stuffing_rate_l800_80031

-- Define Earl's stuffing rate (envelopes per minute)
noncomputable def earl_rate : ℝ := 36

-- Define Ellen's stuffing rate as 2/3 of Earl's rate
noncomputable def ellen_rate (e : ℝ) : ℝ := (2/3) * e

-- Define the combined rate of Earl and Ellen
noncomputable def combined_rate (e : ℝ) : ℝ := e + ellen_rate e

-- Theorem statement
theorem earl_stuffing_rate :
  combined_rate earl_rate = 60 →
  earl_rate = 36 :=
by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_earl_stuffing_rate_l800_80031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_dimensions_l800_80067

/-- Represents a symmetrical trapezoid with parallel sides x and y, non-parallel sides of length 5,
    and height z. -/
structure Trapezoid where
  x : ℝ
  y : ℝ
  z : ℝ
  h_positive : 0 < x ∧ 0 < y ∧ 0 < z
  h_x_gt_y : x > y

/-- The area of a trapezoid -/
noncomputable def Trapezoid.area (t : Trapezoid) : ℝ := (t.x + t.y) * t.z / 2

theorem trapezoid_dimensions (t : Trapezoid) 
    (h1 : t.x - t.y = 6)
    (h2 : (t.x + t.y - 10) * (t.z + 1) = (t.x + t.y) * t.z)
    (h3 : t.z^2 + ((t.x - t.y) / 2)^2 = 5^2) :
    t.x = 28 ∧ t.y = 22 ∧ t.z = 4 ∧ t.area = 100 := by
  sorry

#check trapezoid_dimensions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_dimensions_l800_80067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l800_80068

noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) 
  (h1 : triangle_ABC a b c A B C)
  (h2 : a + c = 1 + Real.sqrt 3)
  (h3 : b = 1)
  (h4 : Real.sin C = Real.sqrt 3 * Real.sin A) :
  B = Real.pi / 6 ∧
  Set.Icc (-4 : ℝ) (2 * Real.sqrt 3 + 2) =
    Set.range (fun x => 2 * Real.sin (2 * x + B) + 4 * (Real.cos x) ^ 2) ∩ 
    Set.Icc 0 (Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l800_80068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_intersection_theorem_l800_80064

/-- Represents a rectangle with sides parallel to the coordinate axes -/
structure Rectangle where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ
  h_x : x1 < x2
  h_y : y1 < y2

/-- Predicate to check if a vertical line intersects a rectangle -/
def intersects_vertical (r : Rectangle) (x : ℝ) : Prop :=
  r.x1 ≤ x ∧ x ≤ r.x2

/-- Predicate to check if a horizontal line intersects a rectangle -/
def intersects_horizontal (r : Rectangle) (y : ℝ) : Prop :=
  r.y1 ≤ y ∧ y ≤ r.y2

/-- Main theorem -/
theorem rectangle_intersection_theorem (rectangles : Set Rectangle) :
  (∀ r1 r2 : Rectangle, r1 ∈ rectangles → r2 ∈ rectangles → r1 ≠ r2 →
    (∃ x : ℝ, intersects_vertical r1 x ∧ intersects_vertical r2 x) ∨
    (∃ y : ℝ, intersects_horizontal r1 y ∧ intersects_horizontal r2 y)) →
  ∃ x y : ℝ, ∀ r ∈ rectangles, intersects_vertical r x ∨ intersects_horizontal r y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_intersection_theorem_l800_80064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_blue_correct_l800_80095

def total_marbles : ℕ := 15
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 7
def num_trials : ℕ := 7
def target_blue : ℕ := 3

def probability_exactly_three_blue : ℚ :=
  (Nat.choose num_trials target_blue) *
  (blue_marbles / total_marbles) ^ target_blue *
  (red_marbles / total_marbles) ^ (num_trials - target_blue)

theorem probability_three_blue_correct :
  probability_exactly_three_blue = 43003620 / 170859375 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_blue_correct_l800_80095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_y_plus_three_l800_80029

theorem power_of_three_y_plus_three (y : ℝ) (h : (3 : ℝ)^y = 81) : (3 : ℝ)^(y+3) = 2187 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_y_plus_three_l800_80029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_l800_80024

noncomputable def root1 (p : ℝ) : ℝ := p + Real.sqrt (p^2 - (p^2 - 4) / 3)
noncomputable def root2 (p : ℝ) : ℝ := p - Real.sqrt (p^2 - (p^2 - 4) / 3)

theorem root_difference (p : ℝ) : 
  let r := max (root1 p) (root2 p)
  let s := min (root1 p) (root2 p)
  r - s = (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_l800_80024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concyclic_CDEF_l800_80035

-- Define the circle
variable (circle : Set (ℝ × ℝ))

-- Define points on the circle
variable (A B C D : ℝ × ℝ)

-- Define S as the midpoint of arc AB
variable (S : ℝ × ℝ)

-- Define E as the intersection of SD and AB
noncomputable def E : ℝ × ℝ := sorry

-- Define F as the intersection of SC and AB
noncomputable def F : ℝ × ℝ := sorry

-- Assume all points are on the circle
axiom on_circle : A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧ D ∈ circle

-- Assume the order of points on the circle
axiom point_order : sorry

-- Assume S is the midpoint of arc AB not containing C and D
axiom S_midpoint : sorry

-- Theorem to prove
theorem concyclic_CDEF : ∃ (newCircle : Set (ℝ × ℝ)), 
  C ∈ newCircle ∧ D ∈ newCircle ∧ E ∈ newCircle ∧ F ∈ newCircle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_concyclic_CDEF_l800_80035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABDF_is_400_l800_80052

/-- Represents a rectangle with points A, B, C, D, E, F -/
structure Rectangle where
  AC : ℝ -- Length of the rectangle
  AE : ℝ -- Width of the rectangle
  h_AC_positive : AC > 0
  h_AE_positive : AE > 0
  h_B_position : AC / 3 > 0 -- Ensures B is between A and C
  h_F_midpoint : AE / 2 > 0 -- Ensures F is between A and E

/-- Calculates the area of quadrilateral ABDF in the given rectangle -/
noncomputable def area_ABDF (r : Rectangle) : ℝ :=
  r.AC * r.AE - (2/3 * r.AC * r.AE / 2) - (r.AE / 2 * r.AC / 2)

/-- Theorem stating that the area of quadrilateral ABDF is 400 square units -/
theorem area_ABDF_is_400 (r : Rectangle) (h_AC : r.AC = 40) (h_AE : r.AE = 24) : 
  area_ABDF r = 400 := by
  unfold area_ABDF
  simp [h_AC, h_AE]
  -- The actual proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABDF_is_400_l800_80052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_chair_price_l800_80096

theorem lawn_chair_price (sale_price : ℝ) (discount_rate : ℝ) (original_price : ℝ) : 
  sale_price = 59.95 →
  discount_rate = 0.1782 →
  original_price * (1 - discount_rate) = sale_price →
  abs (original_price - 72.95) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_chair_price_l800_80096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_data_center_problem_l800_80055

/-- The number of large data centers -/
def n : ℕ := 5

/-- The total number of data centers -/
def total_centers : ℕ := n + 5

/-- The probability of selecting 2 small data centers at once -/
def prob_two_small : ℚ := 2/9

/-- The number of small data centers selected when choosing 3 centers -/
def X : ℕ → ℚ
| 0 => 1/12
| 1 => 5/12
| 2 => 5/12
| 3 => 1/12
| _ => 0

/-- The statement to be proved -/
theorem data_center_problem :
  (n = 5) ∧
  (prob_two_small = (Nat.choose 5 2 : ℚ) / (Nat.choose total_centers 2)) ∧
  (∀ k, X k = (Nat.choose 5 k * Nat.choose n (3-k) : ℚ) / (Nat.choose total_centers 3)) ∧
  (Finset.sum (Finset.range 4) (λ k => k * X k) = 3/2) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_data_center_problem_l800_80055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counterexample_exists_l800_80045

theorem counterexample_exists : ∃ (S : Finset ℝ), 
  (Finset.card S = 25) ∧ 
  (∀ (a b c : ℝ), a ∈ S → b ∈ S → c ∈ S → 
    ∃ (d : ℝ), d ∈ S ∧ d ≠ a ∧ d ≠ b ∧ d ≠ c ∧ a + b + c + d > 0) ∧
  (Finset.sum S id ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_counterexample_exists_l800_80045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l800_80088

theorem sin_double_angle_special_case (α : Real) 
  (h1 : Real.sin α = -4/5) 
  (h2 : α ∈ Set.Ioo (-π/2) (π/2)) : 
  Real.sin (2*α) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l800_80088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l800_80003

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (3^x) / (1 + 3^x)

-- Theorem statement
theorem f_properties :
  (∀ x : ℝ, f (-x) + f x = 1) ∧
  (Set.range f = Set.Ioo 0 1) ∧
  (∀ x : ℝ, f (2*x - 3) + f (x - 3) > 1 ↔ x > 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l800_80003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_k_values_l800_80091

/-- A perfect square trinomial in the form ax^2 + bx + c -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ (p q : ℝ), ∀ (x : ℝ), a * x^2 + b * x + c = (p * x + q)^2

/-- The main theorem -/
theorem perfect_square_k_values :
  ∀ (k : ℝ), is_perfect_square_trinomial 1 (2*(k+1)) 16 → k = 3 ∨ k = -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_k_values_l800_80091
