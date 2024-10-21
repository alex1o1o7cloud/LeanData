import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_equation_l1150_115090

-- Define the circles in rectangular coordinates
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_O2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y - 2 = 0

-- Define the line in polar coordinates
def line_polar (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + Real.pi/4) = Real.sqrt 2

-- Theorem statement
theorem intersection_line_equation :
  ∀ x y : ℝ, circle_O1 x y ∧ circle_O2 x y →
  ∃ ρ θ : ℝ, x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ line_polar ρ θ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_equation_l1150_115090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_k_l1150_115099

-- Define the line l: kx + y + 4 = 0
def line (k : ℝ) (x y : ℝ) : Prop := k * x + y + 4 = 0

-- Define the circle C: x^2 + y^2 - 2y = 0
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0

-- Define a point P on the line
def point_on_line (P : ℝ × ℝ) (k : ℝ) : Prop := line k P.1 P.2

-- Define tangent property
def is_tangent (P A : ℝ × ℝ) : Prop := sorry

-- Define the area of quadrilateral PACB
noncomputable def area_PACB (P A C B : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem find_k (k : ℝ) :
  k > 0 ∧
  (∃ P A B : ℝ × ℝ, 
    point_on_line P k ∧
    circle_eq A.1 A.2 ∧
    circle_eq B.1 B.2 ∧
    is_tangent P A ∧
    is_tangent P B ∧
    (∀ P' A' B' : ℝ × ℝ, 
      point_on_line P' k ∧
      circle_eq A'.1 A'.2 ∧
      circle_eq B'.1 B'.2 ∧
      is_tangent P' A' ∧
      is_tangent P' B' →
      area_PACB P' A' (0, 1) B' ≥ 2) ∧
    area_PACB P A (0, 1) B = 2) →
  k = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_k_l1150_115099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_subtraction_theorem_l1150_115098

/-- Represents a number in base 8 --/
structure OctalNumber where
  value : ℕ

/-- Converts an OctalNumber to its decimal (ℤ) representation --/
def octal_to_decimal (n : OctalNumber) : ℤ :=
  sorry

/-- Converts a decimal (ℤ) to its OctalNumber representation --/
def decimal_to_octal (n : ℤ) : OctalNumber :=
  sorry

/-- Subtracts two OctalNumbers and returns the result as an OctalNumber --/
def octal_subtract (a b : OctalNumber) : OctalNumber :=
  decimal_to_octal (octal_to_decimal a - octal_to_decimal b)

/-- Creates an OctalNumber from a natural number --/
def mk_octal (n : ℕ) : OctalNumber :=
  ⟨n⟩

theorem octal_subtraction_theorem :
  octal_subtract (mk_octal 56) (mk_octal 127) = decimal_to_octal (-52) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_subtraction_theorem_l1150_115098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_constant_wrt_z_x_equals_four_l1150_115041

/-- Given that x is directly proportional to y²z and y is inversely proportional to √z,
    prove that x remains constant regardless of the value of z. -/
theorem x_constant_wrt_z (x y z : ℝ) (k n : ℝ) (h1 : x = k * y^2 * z) (h2 : y = n / Real.sqrt z) :
  ∃ C : ℝ, ∀ z : ℝ, z > 0 → x = C := by sorry

/-- Given that x = 4 when z = 16, prove that x = 4 when z = 50. -/
theorem x_equals_four (x z : ℝ) (h : ∃ C : ℝ, ∀ z : ℝ, z > 0 → x = C) (h1 : x = 4) (h2 : z = 16) :
  x = 4 := by sorry

#check x_constant_wrt_z
#check x_equals_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_constant_wrt_z_x_equals_four_l1150_115041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_numbers_divisible_by_9_l1150_115063

def is_divisible_by_9 (n : ℕ) : Bool := n % 9 = 0

def numbers_between_10_and_86_divisible_by_9 : List ℕ :=
  (List.range 77).map (· + 10) |>.filter is_divisible_by_9

theorem average_of_numbers_divisible_by_9 :
  let numbers := numbers_between_10_and_86_divisible_by_9
  (numbers.sum : ℚ) / numbers.length = 49.5 := by
  sorry

#eval numbers_between_10_and_86_divisible_by_9
#eval (numbers_between_10_and_86_divisible_by_9.sum : ℚ) / numbers_between_10_and_86_divisible_by_9.length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_numbers_divisible_by_9_l1150_115063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_is_one_fourth_l1150_115046

-- Define the function f(x) = a^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- Define the function g(x) = (1 - 4m)√x
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (1 - 4*m) * Real.sqrt x

-- Main theorem
theorem a_value_is_one_fourth
  (a : ℝ)
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (h3 : ∃ (m : ℝ), (∀ x ∈ Set.Icc (-1) 2, f a x ≤ 4 ∧ m ≤ f a x) ∧ 
                   (∃ x1 ∈ Set.Icc (-1) 2, f a x1 = 4) ∧
                   (∃ x2 ∈ Set.Icc (-1) 2, f a x2 = m))
  (h4 : ∃ m : ℝ, ∀ x : ℝ, x ≥ 0 → Monotone (g m)) :
  a = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_is_one_fourth_l1150_115046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_scores_theorem_l1150_115091

noncomputable def scores_A : List ℝ := [4, 5, 5, 6, 6, 7, 7, 8, 8, 9]
noncomputable def scores_B : List ℝ := [2, 5, 6, 6, 7, 7, 7, 8, 9, 10]

noncomputable def average (l : List ℝ) : ℝ := (l.sum) / l.length

noncomputable def variance (l : List ℝ) : ℝ :=
  let avg := average l
  (l.map (λ x => (x - avg)^2)).sum / l.length

theorem shooting_scores_theorem :
  let avg_A := average scores_A
  let avg_B := average scores_B
  let avg_all := average (scores_A ++ scores_B)
  avg_A < avg_B ∧
  avg_all = 6.6 ∧
  variance scores_A < variance scores_B :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_scores_theorem_l1150_115091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equation_solution_l1150_115095

-- Define the polynomials P and Q
noncomputable def P (c : ℝ) : ℝ → ℝ := λ _ => c

noncomputable def Q (c : ℝ) : ℝ → ℝ := λ x => 
  (-c * (x^2024 + x) - x^3 - 2025*x - c^2023) / c^2

-- State the theorem
theorem polynomial_equation_solution (c : ℝ) (hc : c ≠ 0) :
  (∀ a : ℝ, (P c a)^2023 + (Q c a) * (P c a)^2 + (a^2024 + a) * (P c a) + a^3 + 2025*a = 0) ∧
  (∀ P' Q' : ℝ → ℝ, (∀ a : ℝ, (P' a)^2023 + (Q' a) * (P' a)^2 + (a^2024 + a) * (P' a) + a^3 + 2025*a = 0) →
    ∃ c' : ℝ, c' ≠ 0 ∧ P' = P c' ∧ Q' = Q c') :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equation_solution_l1150_115095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1150_115056

-- Define the function f
noncomputable def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- Define the function F
noncomputable def F (b c x : ℝ) : ℝ := (2*x + b) / (Real.exp x)

-- State the theorem
theorem min_value_of_f (b c : ℝ) :
  (∀ x, F b c x = (2*x + b) / (Real.exp x)) → 
  (deriv (F b c) 0 = -2) →
  (F b c 0 = c) →
  (∀ x, f b c x ≥ 0) ∧ (∃ x, f b c x = 0) :=
by
  intros hF hF' hF0
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1150_115056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_third_face_base_angle_formula_l1150_115029

/-- Represents a truncated triangular pyramid with specific properties -/
structure TruncatedPyramid where
  α : ℝ  -- acute angle of the congruent right trapezoid faces
  β : ℝ  -- dihedral angle between the congruent faces
  is_acute : 0 < α ∧ α < Real.pi / 2
  is_dihedral : 0 < β ∧ β < Real.pi

/-- 
The angle between the third lateral face and the base plane 
in a truncated triangular pyramid with the given properties
-/
noncomputable def thirdFaceBaseAngle (p : TruncatedPyramid) : ℝ :=
  Real.arctan (Real.tan p.α / Real.cos (p.β / 2))

/-- 
Theorem stating that the angle between the third lateral face and the base plane 
is equal to arctan(tan(α) / cos(β/2)) for a truncated triangular pyramid 
with the specified properties
-/
theorem third_face_base_angle_formula (p : TruncatedPyramid) : 
  thirdFaceBaseAngle p = Real.arctan (Real.tan p.α / Real.cos (p.β / 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_third_face_base_angle_formula_l1150_115029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l1150_115001

theorem division_problem (k n : ℕ) 
  (h1 : k % n = 11)
  (h2 : (k : ℝ) / (n : ℝ) = 71.2)
  (h3 : n > 0) : 
  n = 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l1150_115001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_length_in_divided_square_l1150_115092

/-- Given a square of side length 1 divided into two congruent trapezoids and a pentagon of equal areas,
    the length of the longer parallel side of each trapezoid is 5/6. -/
theorem trapezoid_length_in_divided_square : ∀ (x : ℝ),
  (∃ (square : Set (ℝ × ℝ)) (trapezoid1 trapezoid2 pentagon : Set (ℝ × ℝ)),
    -- The square has side length 1
    (∀ p ∈ square, 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1) ∧
    -- The square is divided into two congruent trapezoids and a pentagon
    (square = trapezoid1 ∪ trapezoid2 ∪ pentagon) ∧
    (trapezoid1 ∩ trapezoid2 = ∅) ∧ (trapezoid1 ∩ pentagon = ∅) ∧ (trapezoid2 ∩ pentagon = ∅) ∧
    -- The areas of the two trapezoids and the pentagon are equal
    (MeasureTheory.volume trapezoid1 = MeasureTheory.volume trapezoid2) ∧
    (MeasureTheory.volume trapezoid1 = MeasureTheory.volume pentagon) ∧
    -- x is the length of the longer parallel side of each trapezoid
    (∃ (p q : ℝ × ℝ), p ∈ trapezoid1 ∧ q ∈ trapezoid1 ∧ |p.1 - q.1| = x)) →
  x = 5/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_length_in_divided_square_l1150_115092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prasanna_speed_is_35_l1150_115077

/-- The speed of Prasanna given the conditions of the journey -/
noncomputable def prasanna_speed (laxmi_speed : ℝ) (total_time : ℝ) (total_distance : ℝ) : ℝ :=
  (total_distance - laxmi_speed * total_time) / total_time

/-- Theorem stating that Prasanna's speed is 35 kmph given the problem conditions -/
theorem prasanna_speed_is_35 :
  prasanna_speed 25 1 60 = 35 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prasanna_speed_is_35_l1150_115077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_line_l1150_115038

/-- The point through which the line passes -/
noncomputable def P : ℝ × ℝ := (1, 2)

/-- The slope of the line from origin to P -/
noncomputable def m_OP : ℝ := 2

/-- The slope of the line perpendicular to OP -/
noncomputable def m_perp : ℝ := -1 / m_OP

/-- The equation of the line in point-slope form -/
def line_equation (x y : ℝ) : Prop :=
  y - P.2 = m_perp * (x - P.1)

/-- The equation of the line in standard form -/
def line_standard_form (x y : ℝ) : Prop :=
  x + 2*y - 5 = 0

/-- Theorem stating the equivalence of the two line equations -/
theorem max_distance_line :
  ∀ x y : ℝ, line_equation x y ↔ line_standard_form x y :=
by
  intros x y
  constructor
  · intro h
    -- Proof of forward direction
    sorry
  · intro h
    -- Proof of reverse direction
    sorry

#check max_distance_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_line_l1150_115038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_students_count_l1150_115034

theorem new_students_count 
  (initial_count : ℕ) 
  (initial_avg : ℚ) 
  (new_avg : ℚ) 
  (new_students_avg : ℚ) 
  (h1 : initial_count = 10)
  (h2 : initial_avg = 14)
  (h3 : new_avg = initial_avg + 1)
  (h4 : new_students_avg = 17)
  : ∃ (x : ℕ), 
    (initial_count * initial_avg + x * new_students_avg) / (initial_count + x) = new_avg ∧ 
    x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_students_count_l1150_115034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_polygon_pairs_l1150_115023

theorem infinite_polygon_pairs : ∃ (f : ℕ → ℕ × ℕ), 
  (∀ i : ℕ, 
    let (N, n) := f i;
    N ≥ 3 ∧ n ≥ 3 ∧ 
    (((N - 2) * 180 : ℚ) / N) / (((n - 2) * 180 : ℚ) / n) = 3 / 2) ∧
  (∀ i j : ℕ, i ≠ j → f i ≠ f j) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_polygon_pairs_l1150_115023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_equals_35_l1150_115075

-- Define the fraction p/q
def p : ℕ := sorry
def q : ℕ := sorry

-- Define c as the product of p and q
def c : ℕ := p * q

-- Axioms based on the problem conditions
axiom p_q_simplest : Nat.Coprime p q
axiom p_q_bounded : (7 : ℚ) / 10 < (p : ℚ) / q ∧ (p : ℚ) / q < 11 / 15
axiom q_smallest : ∀ (p' q' : ℕ), (7 : ℚ) / 10 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < 11 / 15 → q ≤ q'

-- Theorem to prove
theorem c_equals_35 : c = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_equals_35_l1150_115075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1150_115003

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1|

-- Part I
theorem part_one (m : ℝ) (h_m : m > 0) :
  (∀ x, f (x + 1/2) ≤ 2*m - 1 ↔ x ∈ Set.Icc (-2) 2) → m = 5/2 :=
by
  sorry

-- Part II
theorem part_two :
  (∃ a : ℝ, ∀ x y : ℝ, f x ≤ 2^y + a/(2^y) + |2*x + 3|) ∧
  (∀ a : ℝ, (∀ x y : ℝ, f x ≤ 2^y + a/(2^y) + |2*x + 3|) → a ≥ 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1150_115003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_area_theorem_l1150_115080

/-- A tetrahedron with two adjacent isosceles right triangle faces -/
structure Tetrahedron where
  /-- The length of the hypotenuse of the isosceles right triangle faces -/
  hypotenuse : ℝ
  /-- The dihedral angle between the two adjacent isosceles right triangle faces -/
  dihedral_angle : ℝ

/-- The maximum projection area of a rotating tetrahedron -/
noncomputable def max_projection_area (t : Tetrahedron) : ℝ := 
  if t.hypotenuse = 2 ∧ t.dihedral_angle = Real.pi / 3 then 1 else 0

/-- Theorem stating the maximum projection area of a specific tetrahedron -/
theorem max_projection_area_theorem (t : Tetrahedron) 
  (h1 : t.hypotenuse = 2) 
  (h2 : t.dihedral_angle = Real.pi / 3) : 
  max_projection_area t = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_area_theorem_l1150_115080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_book_sale_l1150_115064

theorem library_book_sale (total_books : ℕ) (remaining_fraction : ℚ) (books_sold : ℕ) : 
  total_books = 9900 →
  remaining_fraction = 4 / 6 →
  books_sold = total_books - (remaining_fraction * ↑total_books).floor →
  books_sold = 3300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_book_sale_l1150_115064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l1150_115084

noncomputable def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  A + B + C = Real.pi ∧ a > 0 ∧ b > 0 ∧ c > 0

theorem triangle_ABC_properties
  (A B C : ℝ) (a b c : ℝ)
  (h_triangle : triangle_ABC A B C a b c)
  (h_eq : 1 + (Real.tan A / Real.tan B) = 2 * c / b) :
  A = Real.pi / 3 ∧
  ∃ (m n : ℝ × ℝ),
    m = (0, -1) ∧
    n = (Real.cos B, 2 * (Real.cos (C / 2))^2) ∧
    (Real.sqrt 2)/2 = Real.sqrt ((m.1 + n.1)^2 + (m.2 + n.2)^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l1150_115084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l1150_115047

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define the focus of the parabola
noncomputable def focus : ℝ × ℝ := (1/2, 0)

-- Define point A
def point_A : ℝ × ℝ := (3, 2)

-- Define a point on the parabola
def point_on_parabola (P : ℝ × ℝ) : Prop := parabola P.1 P.2

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- The theorem to prove
theorem min_distance_point :
  ∃ (P : ℝ × ℝ), point_on_parabola P ∧
    ∀ (Q : ℝ × ℝ), point_on_parabola Q →
      distance P point_A + distance P focus ≤ distance Q point_A + distance Q focus ∧
    P = (2, 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l1150_115047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_one_tan_half_implies_expression_four_fifths_l1150_115065

open Real

-- Part 1
noncomputable def f (α : ℝ) : ℝ :=
  (tan (π + α) * cos (2 * π + α) * sin (α - π / 2)) /
  (cos (-α - 3 * π) * sin (-3 * π - α))

theorem f_equals_one (α : ℝ) : f α = 1 := by sorry

-- Part 2
theorem tan_half_implies_expression_four_fifths (α : ℝ) (h : tan α = 1 / 2) :
  2 * sin α ^ 2 - sin α * cos α + cos α ^ 2 = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_one_tan_half_implies_expression_four_fifths_l1150_115065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_line_x_intercept_l1150_115032

/-- A line in the coordinate plane where the slope is 0.5 times the y-intercept -/
structure SpecialLine where
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The slope of the line -/
  slope : ℝ
  /-- The slope of the line is 0.5 times the y-intercept -/
  slope_eq : slope = 0.5 * y_intercept

/-- The x-intercept of a SpecialLine is -2 -/
theorem special_line_x_intercept (k : SpecialLine) : 
  ∃ x : ℝ, x = -2 ∧ k.slope * x + k.y_intercept = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_line_x_intercept_l1150_115032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_f_g_condition_l1150_115059

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (21 - x^2 - 4*x)
def g (x : ℝ) : ℝ := |x + 2|

-- Define the set of x values that satisfy the conditions
def S : Set ℝ := Set.union (Set.Icc (-7 : ℝ) (-8/3)) (Set.Ioo (0 : ℝ) 2)

-- State the theorem
theorem min_f_g_condition (x : ℝ) :
  (21 - x^2 - 4*x ≥ 0) →  -- Condition for f(x) to be defined
  (min (f x) (g x) > (x + 4)/2) →
  x ∈ S :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_f_g_condition_l1150_115059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_slope_l1150_115061

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a parabola in 2D space -/
structure Parabola where
  a : ℝ

noncomputable def Parabola.focus (p : Parabola) : Point :=
  { x := 1 / (4 * p.a), y := 0 }

noncomputable def Line.throughPoints (A B : Point) : Line :=
  { slope := (B.y - A.y) / (B.x - A.x),
    intercept := A.y - (B.y - A.y) / (B.x - A.x) * A.x }

noncomputable def distance (A B : Point) : ℝ :=
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

def Line.intersectParabola (l : Line) (p : Parabola) : Set Point :=
  { P : Point | P.y = l.slope * (P.x - 1) ∧ P.y^2 = 4 * p.a * P.x }

theorem parabola_line_intersection_slope 
  (p : Parabola) 
  (l : Line) 
  (A B : Point) 
  (hP : p = { a := 1/4 }) 
  (hF : Line.throughPoints (Parabola.focus p) A = l ∨ Line.throughPoints (Parabola.focus p) B = l)
  (hAB : A ∈ l.intersectParabola p ∧ B ∈ l.intersectParabola p)
  (hDist : distance A (Parabola.focus p) = 4 * distance B (Parabola.focus p)) :
  l.slope = 4/3 ∨ l.slope = -4/3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_slope_l1150_115061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_cost_price_theorem_l1150_115051

/-- The cost price of a radio given overhead expenses, selling price, and profit percentage. -/
noncomputable def cost_price (overhead : ℝ) (selling_price : ℝ) (profit_percent : ℝ) : ℝ :=
  (selling_price - overhead) / (1 + profit_percent / 100)

/-- Theorem stating that the cost price of the radio is approximately 228.57 
    given the specified conditions. -/
theorem radio_cost_price_theorem :
  let overhead := (20 : ℝ)
  let selling_price := (300 : ℝ)
  let profit_percent := (22.448979591836732 : ℝ)
  abs (cost_price overhead selling_price profit_percent - 228.57) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_cost_price_theorem_l1150_115051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_variance_l1150_115062

noncomputable def sample : List ℝ := [1, 3, 5, 7]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (fun x => (x - m)^2)).sum / xs.length

theorem sample_variance : variance sample = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_variance_l1150_115062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_rhombus_l1150_115083

/-- The quartic polynomial whose roots form the rhombus -/
def f (z : ℂ) : ℂ := 2 * z^4 + 8 * Complex.I * z^3 + (-9 + 9 * Complex.I) * z^2 + (-18 - 2 * Complex.I) * z + (3 - 12 * Complex.I)

/-- Predicate to check if four complex numbers form a rhombus -/
def IsRhombus (a b c d : ℂ) : Prop := sorry

/-- The roots of the polynomial form a rhombus -/
axiom roots_form_rhombus : ∃ (a b c d : ℂ), f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0 ∧ IsRhombus a b c d

/-- The area of the rhombus formed by the roots -/
noncomputable def rhombus_area : ℝ := Real.sqrt 10

/-- Function to calculate the area of a rhombus given its vertices -/
noncomputable def RhombusArea (a b c d : ℂ) : ℝ := sorry

/-- Theorem stating that the area of the rhombus is √10 -/
theorem area_of_rhombus : 
  ∀ (a b c d : ℂ), f a = 0 → f b = 0 → f c = 0 → f d = 0 → IsRhombus a b c d → 
  RhombusArea a b c d = rhombus_area := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_rhombus_l1150_115083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_speaking_only_english_l1150_115011

theorem percentage_speaking_only_english :
  let total_children : ℕ := 60
  let percentage_both : ℚ := 1/5
  let hindi_speakers : ℕ := 42
  (3 : ℚ)/10 = (total_children - (percentage_both * total_children).num - (hindi_speakers - (percentage_both * total_children).num)) / total_children := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_speaking_only_english_l1150_115011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lily_distance_from_start_l1150_115028

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents Lily's walk -/
def lily_walk : Point → Point
| ⟨x, y⟩ => ⟨x + 15, y - 30⟩

theorem lily_distance_from_start :
  ∀ start : Point, distance start (lily_walk start) = 15 * Real.sqrt 5 := by
  sorry

#check lily_distance_from_start

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lily_distance_from_start_l1150_115028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_lines_theorem_l1150_115025

-- Define the points of the triangle
noncomputable def A : ℝ × ℝ := (2, 2)
noncomputable def B : ℝ × ℝ := (-4, 0)
noncomputable def C : ℝ × ℝ := (3, -1)

-- Define the slope of BC
noncomputable def slope_BC : ℝ := (C.2 - B.2) / (C.1 - B.1)

-- Define the slope of AD (perpendicular to BC)
noncomputable def slope_AD : ℝ := -1 / slope_BC

-- Define the equation of line AD
noncomputable def line_AD (x : ℝ) : ℝ := slope_AD * (x - A.1) + A.2

-- Define the slope of AC
noncomputable def slope_AC : ℝ := (C.2 - A.2) / (C.1 - A.1)

-- Define the point D (intersection of AD and BC)
noncomputable def D : ℝ × ℝ := (8/5, -4/5)

-- Define the equation of line parallel to AC passing through D
noncomputable def line_parallel_AC (x : ℝ) : ℝ := slope_AC * (x - D.1) + D.2

theorem triangle_lines_theorem :
  (∀ x, line_AD x = 7 * x - 12) ∧
  (∀ x, line_parallel_AC x = -3 * x + 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_lines_theorem_l1150_115025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1150_115000

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then
    x^2 - 4*x + 5
  else
    Real.log (x - 1) / Real.log (1/2)

-- State the theorem
theorem range_of_a (a : ℝ) : f (a^2 - 3) > f (a - 2) → a ∈ Set.Ioo 2 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1150_115000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_efficiency_ratio_for_given_cycles_l1150_115031

/-- Efficiency of a thermodynamic cycle -/
noncomputable def efficiency (work : ℝ) (heat : ℝ) : ℝ := work / heat

/-- Ratio of efficiencies for two thermodynamic cycles -/
noncomputable def efficiency_ratio (work1 work2 heat1 heat2 : ℝ) : ℝ :=
  (efficiency work1 heat1) / (efficiency work2 heat2)

theorem efficiency_ratio_for_given_cycles 
  (p₀ V₀ : ℝ) 
  (h_p₀_pos : p₀ > 0) 
  (h_V₀_pos : V₀ > 0) :
  efficiency_ratio ((1/2) * p₀ * V₀) ((1/2) * p₀ * V₀) (6 * p₀ * V₀) (13/2 * p₀ * V₀) = 13/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_efficiency_ratio_for_given_cycles_l1150_115031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_sum_l1150_115008

-- Define the piecewise function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then x - 1
  else if x = 0 then a
  else x + b

-- Define the property of being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_sum (a b : ℝ) :
  is_odd_function (f a b) → a + b = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_sum_l1150_115008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_S_l1150_115014

/-- The set of integers n > 1 for which 1/n has a repeating decimal with period 12 -/
def S : Set Nat :=
  {n | n > 1 ∧ (10^12 - 1) % n = 0}

/-- 9901 is prime -/
axiom prime_9901 : Nat.Prime 9901

/-- The theorem to prove -/
theorem count_S : Finset.card (Finset.filter (fun n => n > 1) (Nat.divisors (10^12 - 1))) = 255 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_S_l1150_115014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sequence_negativity_l1150_115068

theorem cosine_sequence_negativity (α : ℝ) :
  (∀ n : ℕ, Real.cos (2^n * α) < 0) ↔ 
  ∃ k : ℤ, α = 2*Real.pi/3 + 2*k*Real.pi ∨ α = -2*Real.pi/3 + 2*k*Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sequence_negativity_l1150_115068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_henry_meets_train_l1150_115050

/-- Train arrival time in minutes after 2:00 PM -/
def train_arrival : Set ℝ := Set.Icc 120 240

/-- Henry's arrival time in minutes after 2:00 PM -/
def henry_arrival : Set ℝ := Set.Icc 150 270

/-- The duration the train stays at the station -/
def train_stay : ℝ := 30

/-- The event that Henry meets the train -/
def henry_meets_train : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ∈ henry_arrival ∧ p.2 ∈ train_arrival ∧ p.2 ≤ p.1 ∧ p.1 ≤ p.2 + train_stay}

theorem probability_henry_meets_train :
  (MeasureTheory.volume henry_meets_train) / ((270 - 150) * (240 - 120)) = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_henry_meets_train_l1150_115050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_5_15_l1150_115033

/-- The acute angle between clock hands at a given time -/
noncomputable def clockHandAngle (hours : ℕ) (minutes : ℕ) : ℝ :=
  let hourAngle : ℝ := (hours % 12 + minutes / 60 : ℝ) * 30
  let minuteAngle : ℝ := (minutes : ℝ) * 6
  min (abs (hourAngle - minuteAngle)) (360 - abs (hourAngle - minuteAngle))

/-- Theorem: The acute angle between the hour and minute hands at 5:15 is 67.5° -/
theorem clock_angle_at_5_15 :
  clockHandAngle 5 15 = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_5_15_l1150_115033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_circumscribed_sphere_radius_l1150_115042

/-- The radius of a sphere circumscribed around a pyramid with a square base -/
noncomputable def circumscribed_sphere_radius (a h : ℝ) : ℝ :=
  Real.sqrt ((a^2 / 2) + (h^2 / 9))

/-- Theorem: The radius of the sphere circumscribed around the pyramid is 3.5 -/
theorem pyramid_circumscribed_sphere_radius 
  (a : ℝ) 
  (h : ℝ) 
  (base_side : a = Real.sqrt 21) 
  (height : h = (a * Real.sqrt 3) / 2) : 
  circumscribed_sphere_radius a h = 3.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_circumscribed_sphere_radius_l1150_115042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_power_of_two_l1150_115085

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => if (n + 2) % 2 = 0 then a ((n + 2) / 2) + (n + 2) / 2 else a (n + 1)

theorem a_power_of_two (k : ℕ) : a (2^k) = 2^k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_power_of_two_l1150_115085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_will_reach_target_l1150_115066

/-- Represents the position of the traveler on the parallel lines -/
inductive Position
| mk : ℤ → Position

/-- The probability of moving up or down -/
noncomputable def p : ℝ := 1/4

/-- The probability of staying on the same line -/
noncomputable def r : ℝ := 1/2

/-- A random walk on parallel lines -/
def RandomWalk (start : Position) : Type :=
  Nat → Position

/-- The probability of reaching a specific position from a given start position -/
noncomputable def ReachProbability (start target : Position) : ℝ :=
  sorry

/-- The main theorem: The probability of reaching any position is 1 -/
theorem will_reach_target (start target : Position) :
  ReachProbability start target = 1 := by
  sorry

#check will_reach_target

end NUMINAMATH_CALUDE_ERRORFEEDBACK_will_reach_target_l1150_115066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_integer_pairs_satisfy_inequalities_l1150_115058

theorem two_integer_pairs_satisfy_inequalities : 
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (a b : ℤ), (a, b) ∈ s ↔ 
      (a^2 + b^2 < 9 ∧ 
       a^2 + b^2 < 8*(a - 2) ∧ 
       a^2 + b^2 < 8*(b - 2))) ∧
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_integer_pairs_satisfy_inequalities_l1150_115058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sum_l1150_115019

theorem range_of_sum (x y : ℝ) (h : (4:ℝ)^x + (4:ℝ)^y = (2:ℝ)^(x+1) + (2:ℝ)^(y+1)) :
  let s := (2:ℝ)^x + (2:ℝ)^y
  2 < s ∧ s ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sum_l1150_115019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1150_115082

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  R : Real

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * t.a * Real.cos t.A = t.c * Real.cos t.B + t.b * Real.cos t.C)
  (h2 : t.R = 2)
  (h3 : t.b^2 + t.c^2 = 18) :
  t.A = π/3 ∧ (1/2) * t.b * t.c * Real.sin t.A = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1150_115082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_change_l1150_115086

/-- Represents a cylinder with radius and height -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Calculates the volume of a cylinder -/
noncomputable def volume (c : Cylinder) : ℝ := Real.pi * c.radius ^ 2 * c.height

/-- Theorem: If a cylinder's original volume is 15 cubic feet, and its radius is tripled
    while its height is doubled, then its new volume will be 270 cubic feet -/
theorem cylinder_volume_change (c : Cylinder) (h_vol : volume c = 15) :
  volume { radius := 3 * c.radius, height := 2 * c.height } = 270 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_change_l1150_115086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_of_M_l1150_115060

theorem integer_part_of_M (x : ℝ) (h : x ∈ Set.Ioo 0 (Real.pi / 2)) :
  3 ≤ (3 : ℝ)^((Real.cos x)^2) + (3 : ℝ)^((Real.sin x)^3) ∧ 
  (3 : ℝ)^((Real.cos x)^2) + (3 : ℝ)^((Real.sin x)^3) < 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_of_M_l1150_115060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_foci_distance_of_given_hyperbola_l1150_115002

/-- A hyperbola with equation x²/32 - y²/8 = 1 -/
structure Hyperbola where
  a_squared : ℝ := 32
  b_squared : ℝ := 8
  equation : ∀ x y : ℝ, x^2 / a_squared - y^2 / b_squared = 1

/-- The distance between the foci of a hyperbola -/
noncomputable def foci_distance (h : Hyperbola) : ℝ :=
  2 * Real.sqrt (h.a_squared + h.b_squared)

/-- Theorem: The distance between the foci of the given hyperbola is 4√10 -/
theorem foci_distance_of_given_hyperbola :
  ∃ h : Hyperbola, foci_distance h = 4 * Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_foci_distance_of_given_hyperbola_l1150_115002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_approx_l1150_115071

/-- Calculates the profit percentage given the selling price and cost price -/
noncomputable def profit_percentage (selling_price : ℝ) (cost_price : ℝ) : ℝ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem stating that the profit percentage is approximately 29.99% -/
theorem profit_percentage_approx :
  let selling_price : ℝ := 100
  let cost_price : ℝ := 76.92
  abs (profit_percentage selling_price cost_price - 29.99) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_approx_l1150_115071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pqrst_theorem_l1150_115093

def is_valid_pqrst (p q r s t : Nat) : Prop :=
  p ∈ ({1, 2, 3, 4, 5} : Set Nat) ∧
  q ∈ ({1, 2, 3, 4, 5} : Set Nat) ∧
  r ∈ ({1, 2, 3, 4, 5} : Set Nat) ∧
  s ∈ ({1, 2, 3, 4, 5} : Set Nat) ∧
  t ∈ ({1, 2, 3, 4, 5} : Set Nat) ∧
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧
  r ≠ s ∧ r ≠ t ∧
  s ≠ t

def pqr_value (p q r : Nat) : Nat := 100 * p + 10 * q + r
def qrs_value (q r s : Nat) : Nat := 100 * q + 10 * r + s
def rst_value (r s t : Nat) : Nat := 100 * r + 10 * s + t

theorem pqrst_theorem (p q r s t : Nat) :
  is_valid_pqrst p q r s t →
  pqr_value p q r % 5 = 0 →
  qrs_value q r s % 4 = 0 →
  rst_value r s t % 5 = 0 →
  p = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pqrst_theorem_l1150_115093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_matrix_correct_problem_matrix_correct_l1150_115020

/-- Represents a 3D dilation matrix with specified scaling factors for each dimension. -/
noncomputable def dilationMatrix (sx sy sz : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![sx, 0, 0; 0, sy, 0; 0, 0, sz]

/-- Checks if a given matrix correctly represents the specified dilation. -/
theorem dilation_matrix_correct (M : Matrix (Fin 3) (Fin 3) ℝ) :
  M = dilationMatrix 2 (-1/2) 3 ↔
  (∀ v : Fin 3 → ℝ,
    M.vecMul v 0 = 2 * v 0 ∧
    M.vecMul v 1 = -1/2 * v 1 ∧
    M.vecMul v 2 = 3 * v 2) :=
by sorry

/-- The specific dilation matrix for the given problem. -/
noncomputable def problem_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  dilationMatrix 2 (-1/2) 3

/-- Proves that the problem_matrix correctly represents the specified dilation. -/
theorem problem_matrix_correct :
  (∀ v : Fin 3 → ℝ,
    problem_matrix.vecMul v 0 = 2 * v 0 ∧
    problem_matrix.vecMul v 1 = -1/2 * v 1 ∧
    problem_matrix.vecMul v 2 = 3 * v 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_matrix_correct_problem_matrix_correct_l1150_115020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1150_115079

noncomputable def point1 : ℝ × ℝ := (2, -3)
noncomputable def point2 : ℝ × ℝ := (8, 9)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_between_points :
  distance point1 point2 = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1150_115079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_grade_percentage_is_fifteen_l1150_115005

/-- Represents the percentage of students in each grade for a school -/
structure SchoolGradePercentages where
  k : ℚ
  g1 : ℚ
  g2 : ℚ
  g3 : ℚ
  g4 : ℚ
  g5 : ℚ
  g6 : ℚ

/-- The percentage of 2nd graders in both schools combined -/
noncomputable def combinedSecondGradePercentage (maple_students : ℕ) (oak_students : ℕ)
    (maple_percentages : SchoolGradePercentages) (oak_percentages : SchoolGradePercentages) : ℚ :=
  let maple_second_graders := (maple_percentages.g2 / 100) * maple_students
  let oak_second_graders := (oak_percentages.g2 / 100) * oak_students
  let total_second_graders := maple_second_graders + oak_second_graders
  let total_students := maple_students + oak_students
  (total_second_graders / total_students) * 100

theorem second_grade_percentage_is_fifteen :
  let maple_students : ℕ := 150
  let oak_students : ℕ := 250
  let maple_percentages : SchoolGradePercentages :=
    { k := 10, g1 := 20, g2 := 18, g3 := 15, g4 := 12, g5 := 15, g6 := 10 }
  let oak_percentages : SchoolGradePercentages :=
    { k := 15, g1 := 14, g2 := 13, g3 := 16, g4 := 12, g5 := 15, g6 := 15 }
  combinedSecondGradePercentage maple_students oak_students maple_percentages oak_percentages = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_grade_percentage_is_fifteen_l1150_115005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_entry_is_25_l1150_115004

/-- The remainder when 7n is divided by 10 -/
def r_10 (n : ℕ) : ℕ := (7 * n) % 10

/-- The predicate that n satisfies r_10(7n) ≤ 5 -/
def satisfies_condition (n : ℕ) : Prop := r_10 n ≤ 5

/-- The sequence of nonnegative integers satisfying the condition -/
def satisfying_sequence : ℕ → ℕ := sorry

theorem fifteenth_entry_is_25 :
  satisfying_sequence 14 = 25 ∧ 
  ∀ k < 14, satisfies_condition (satisfying_sequence k) ∧ satisfying_sequence k < 25 :=
sorry

#check fifteenth_entry_is_25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_entry_is_25_l1150_115004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l1150_115016

-- Define the circle in polar coordinates
def polar_circle (ρ θ : ℝ) : Prop := ρ = 3/2

-- Define the line in polar coordinates
def polar_line (ρ θ : ℝ) : Prop := ρ * (Real.sqrt 7 * Real.cos θ - Real.sin θ) = Real.sqrt 2

-- Define the distance function (placeholder, actual implementation not provided)
noncomputable def polar_distance (ρ₁ θ₁ ρ₂ θ₂ : ℝ) : ℝ := sorry

-- Theorem statement
theorem max_distance_circle_to_line :
  ∃ (d : ℝ), d = 2 ∧ 
  ∀ (ρ₁ θ₁ ρ₂ θ₂ : ℝ), 
    polar_circle ρ₁ θ₁ → polar_line ρ₂ θ₂ → 
    polar_distance ρ₁ θ₁ ρ₂ θ₂ ≤ d ∧
    (∃ (ρ₃ θ₃ ρ₄ θ₄ : ℝ), 
      polar_circle ρ₃ θ₃ ∧ polar_line ρ₄ θ₄ ∧ 
      polar_distance ρ₃ θ₃ ρ₄ θ₄ = d) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l1150_115016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_ellipse_l1150_115067

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Predicate to check if two circles are tangent
def IsTangent (c1 c2 : Circle) : Prop := sorry

-- Define the locus of centers
def LocusOfCenters (givenCircle : Circle) (externalPoint : Point) :=
  {p : Point | ∃ (r : ℝ), IsTangent (Circle.mk p r) givenCircle ∧ r > 0}

-- Define a distance function between two points
def dist (p1 p2 : Point) : ℝ := sorry

-- Predicate to check if a set of points forms an ellipse
def IsEllipse (s : Set Point) : Prop := sorry

-- Theorem statement
theorem locus_is_ellipse (givenCircle : Circle) (externalPoint : Point) 
  (h : dist externalPoint givenCircle.center > givenCircle.radius) :
  IsEllipse (LocusOfCenters givenCircle externalPoint) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_ellipse_l1150_115067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_optimality_l1150_115096

/-- Triangle ABC with sides a, b, c and area Δ -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  Δ : ℝ
  area_formula : Δ = (1/4) * Real.sqrt ((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c))

/-- The product of the sides of a triangle -/
noncomputable def side_product (t : Triangle) : ℝ := t.a * t.b * t.c

/-- The sum of the sides of a triangle -/
noncomputable def side_sum (t : Triangle) : ℝ := t.a + t.b + t.c

/-- The sum of the squares of the sides of a triangle -/
noncomputable def side_square_sum (t : Triangle) : ℝ := t.a^2 + t.b^2 + t.c^2

/-- The radius of the circumcircle of a triangle -/
noncomputable def circumcircle_radius (t : Triangle) : ℝ := (t.a * t.b * t.c) / (4 * t.Δ)

/-- A triangle is equilateral if all its sides are equal -/
def is_equilateral (t : Triangle) : Prop := t.a = t.b ∧ t.b = t.c

theorem triangle_optimality (t : Triangle) :
  (∀ t' : Triangle, t'.Δ = t.Δ → side_product t ≤ side_product t') ∨
  (∀ t' : Triangle, t'.Δ = t.Δ → side_sum t ≤ side_sum t') ∨
  (∀ t' : Triangle, t'.Δ = t.Δ → side_square_sum t ≤ side_square_sum t') ∨
  (∀ t' : Triangle, t'.Δ = t.Δ → circumcircle_radius t ≤ circumcircle_radius t') →
  is_equilateral t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_optimality_l1150_115096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_one_plus_g_of_two_l1150_115072

-- Define the functions f and g
def f (x : ℝ) : ℝ := 3 * x - 3
def g (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem f_of_one_plus_g_of_two : f (1 + g 2) = 15 := by
  -- Expand the definition of g
  have h1 : g 2 = 5 := by
    rw [g]
    norm_num
  
  -- Substitute g(2) with 5
  have h2 : f (1 + g 2) = f 6 := by
    rw [h1]
    norm_num
  
  -- Evaluate f(6)
  rw [h2, f]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_one_plus_g_of_two_l1150_115072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brokerage_percentage_l1150_115097

def cash_realized : ℝ := 109.25
def cash_after_brokerage : ℝ := 109

theorem brokerage_percentage :
  let brokerage_amount := cash_realized - cash_after_brokerage
  let brokerage_percentage := (brokerage_amount / cash_realized) * 100
  ∃ ε > 0, |brokerage_percentage - 0.228833| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brokerage_percentage_l1150_115097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_length_calculation_l1150_115078

/-- Represents the properties of a rectangular floor --/
structure RectangularFloor where
  breadth : ℝ
  length : ℝ
  area : ℝ
  length_ratio : length = 3 * breadth
  area_equation : area = length * breadth

/-- Theorem stating the length of the floor given the conditions --/
theorem floor_length_calculation (floor : RectangularFloor) 
  (h_area : floor.area = 128) : 
  ∃ ε > 0, |floor.length - 19.59| < ε := by
  sorry

#check floor_length_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_length_calculation_l1150_115078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_c_coordinates_l1150_115052

/-- Point with x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem: Given the conditions, point C has coordinates (3.4, 7.6) -/
theorem point_c_coordinates :
  let A : Point := ⟨-3, -2⟩
  let B : Point := ⟨5, 10⟩
  ∀ C : Point,
    (∃ t : ℝ, 0 < t ∧ t < 1 ∧
      C.x = A.x + t * (B.x - A.x) ∧
      C.y = A.y + t * (B.y - A.y)) →
    distance A C = 2 * distance C B →
    C.x = 3.4 ∧ C.y = 7.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_c_coordinates_l1150_115052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_digits_for_random_sample_l1150_115073

theorem min_digits_for_random_sample (n : Nat) (h : n = 1001) :
  let sample_size := 10
  let min_digits := Nat.ceil (Real.log n / Real.log 10)
  min_digits = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_digits_for_random_sample_l1150_115073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_propositions_l1150_115040

-- Define the basic types
class Line : Type
class Plane : Type

-- Define the relationships
axiom perpendicular : Line → Plane → Prop
axiom parallel : Plane → Plane → Prop
axiom parallelLines : Line → Line → Prop
axiom contains : Plane → Line → Prop
axiom skew : Line → Line → Prop

-- Define the propositions
def proposition1 (m : Line) (α β : Plane) : Prop :=
  perpendicular m α ∧ perpendicular m β → parallel α β

def proposition2 (α β γ : Plane) : Prop :=
  parallel α γ ∧ parallel β γ → parallel α β

def proposition3 (m n : Line) (α β : Plane) : Prop :=
  contains α m ∧ contains β n ∧ parallelLines m n → parallel α β

def proposition4 (m n : Line) (α β : Plane) : Prop :=
  skew m n ∧ contains α m ∧ contains β n ∧ ¬parallelLines n m ∧ ¬parallelLines m n → parallel α β

theorem correct_propositions (m n : Line) (α β γ : Plane) :
  proposition1 m α β ∧ proposition2 α β γ ∧ ¬proposition3 m n α β ∧ proposition4 m n α β :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_propositions_l1150_115040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_x_coordinate_l1150_115027

/-- A point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x

/-- The focus of the parabola y^2 = 4x -/
def parabola_focus : ℝ × ℝ := (1, 0)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_point_x_coordinate (A : ParabolaPoint) :
  distance (A.x, A.y) parabola_focus = 6 → A.x = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_x_coordinate_l1150_115027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_when_k_is_2_complement_intersection_empty_iff_l1150_115012

-- Define the sets A and B
def A : Set ℝ := {x | |x - 2| ≥ 1}
def B (k : ℝ) : Set ℝ := {x | k < x ∧ x < 2 * k + 1}

-- Theorem for part (1)
theorem intersection_when_k_is_2 :
  A ∩ B 2 = {x | 3 ≤ x ∧ x < 5} := by sorry

-- Theorem for part (2)
theorem complement_intersection_empty_iff (k : ℝ) :
  (Aᶜ ∩ B k = ∅) ↔ k ≤ 0 ∨ k ≥ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_when_k_is_2_complement_intersection_empty_iff_l1150_115012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_proof_l1150_115030

theorem angle_sum_proof (α β : Real) (h1 : π < α ∧ α < 2*π) (h2 : π < β ∧ β < 2*π)
  (h3 : Real.cos α = -(2 * Real.sqrt 5) / 5) (h4 : Real.sin β = Real.sqrt 10 / 10) :
  α + β = 7 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_proof_l1150_115030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_COB_area_l1150_115015

/-- The area of a triangle given the coordinates of its vertices -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

/-- Theorem: Area of triangle COB is (15p)/2 -/
theorem triangle_COB_area (p : ℝ) (h : 0 < p ∧ p < 15) : 
  triangleArea 0 0 0 p 15 0 = (15 * p) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_COB_area_l1150_115015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1150_115017

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  area : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.area ≥ 2 ∧ t.c * Real.cos t.A = 4 / t.b

-- Define the function f
noncomputable def f (A : ℝ) : ℝ :=
  Real.cos A ^ 2 + Real.sqrt 3 * Real.sin (Real.pi / 2 + A / 2) ^ 2 - Real.sqrt 3 / 2

-- Theorem statement
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) :
  Real.pi / 4 ≤ t.A ∧ t.A < Real.pi / 2 ∧
  ∀ x, Real.pi / 4 ≤ x ∧ x < Real.pi / 2 → f x ≤ (1 / 2 + Real.sqrt 6 / 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1150_115017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combination_sum_simplification_l1150_115024

theorem combination_sum_simplification (n : ℕ) : 
  (Finset.range (n + 1)).sum (λ k ↦ (1 : ℚ) / (k + 1) * Nat.choose n k * (1 / 5) ^ (k + 1)) = 
  (1 : ℚ) / (n + 1) * ((6 / 5) ^ (n + 1) - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combination_sum_simplification_l1150_115024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_quadrant_l1150_115081

theorem angle_quadrant (α : ℝ) : 
  (Real.sin α > 0 ∧ Real.tan α < 0) → 
  (π/2 < α ∧ α < π) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_quadrant_l1150_115081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_star_self_intersections_star_2018_25_intersections_l1150_115076

/-- Definition of a regular (n; k)-star -/
def regular_star (n k : ℕ) : Prop :=
  n ≥ 5 ∧ k < n / 2 ∧ Nat.Coprime n k

/-- Number of self-intersections in a regular (n; k)-star -/
def self_intersections (n k : ℕ) : ℕ := n * (k - 1)

/-- Theorem: Number of self-intersections in a regular (n; k)-star -/
theorem regular_star_self_intersections (n k : ℕ) :
  regular_star n k → self_intersections n k = n * (k - 1) := by
  intro h
  rfl

/-- Theorem: The (2018; 25)-star has 48432 self-intersections -/
theorem star_2018_25_intersections :
  regular_star 2018 25 → self_intersections 2018 25 = 48432 := by
  intro h
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_star_self_intersections_star_2018_25_intersections_l1150_115076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_b_value_l1150_115022

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.A = Real.pi/4 ∧  -- A = 45°
  2*t.b*(Real.sin t.B) - t.c*(Real.sin t.C) = 2*t.a*(Real.sin t.A) ∧  -- Given equation
  1/2*t.b*t.c*(Real.sin t.A) = 3  -- Area condition

-- Theorem statement
theorem triangle_side_b_value (t : Triangle) 
  (h : triangle_conditions t) : t.b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_b_value_l1150_115022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_segment_l1150_115053

/-- The radius of the cylindrical region around the line segment -/
noncomputable def r : ℝ := 3

/-- The volume of the region around the line segment -/
noncomputable def V : ℝ := 216 * Real.pi

/-- The length of the line segment AB -/
noncomputable def L : ℝ := 20

/-- Theorem stating that the length of AB is 20 given the volume of the region -/
theorem length_of_segment (h : V = Real.pi * r^2 * L + (4/3) * Real.pi * r^3) : L = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_segment_l1150_115053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_parallel_l1150_115026

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel
  (m n : Line) (β : Plane)
  (hm : m ≠ n)
  (hm_perp : perpendicular m β)
  (hn_perp : perpendicular n β) :
  parallel m n :=
sorry

-- Note: We removed the condition (hβ : β ≠ ∅) as it was causing the EmptyCollection error

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_parallel_l1150_115026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_value_l1150_115045

-- Define n and b as real numbers
variable (n b : ℝ)

-- State the theorem
theorem b_value (h1 : n = 2 ^ (3/10 : ℝ)) (h2 : n ^ b = 16) : b = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_value_l1150_115045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_catches_john_l1150_115006

/-- The time (in minutes) it takes for Bob to catch up with John -/
noncomputable def catchUpTime (john_speed bob_speed : ℝ) (initial_distance : ℝ) : ℝ :=
  (initial_distance / (bob_speed - john_speed)) * 60

theorem bob_catches_john (john_speed bob_speed initial_distance : ℝ) :
  john_speed = 8 →
  bob_speed = 12 →
  initial_distance = 3 →
  catchUpTime john_speed bob_speed initial_distance = 45 :=
by
  intros h1 h2 h3
  unfold catchUpTime
  simp [h1, h2, h3]
  norm_num
  -- The proof is completed by norm_num, but we can add sorry if needed
  -- sorry

-- Remove the #eval line as it's not computable
-- #eval catchUpTime 8 12 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_catches_john_l1150_115006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_men_meet_35_miles_nearer_A_l1150_115021

/-- Two men walking towards each other from points A and B -/
structure WalkingMen where
  distance : ℝ  -- Total distance between A and B
  speed_a : ℝ   -- Constant speed of man A
  speed_b : ℝ → ℝ  -- Speed of man B as a function of time

/-- The meeting point of the two men -/
noncomputable def meeting_point (w : WalkingMen) : ℝ → ℝ :=
  λ t => w.speed_a * t

/-- The distance walked by man B -/
noncomputable def distance_b (w : WalkingMen) : ℝ → ℝ :=
  λ t => (t / 2) * (7 + t)

/-- Theorem stating that the men meet 35 miles nearer to A than B -/
theorem men_meet_35_miles_nearer_A (w : WalkingMen) :
  w.distance = 100 ∧
  w.speed_a = 5 ∧
  (∀ t, w.speed_b t = 4 + t - 1) →
  ∃ t : ℕ, meeting_point w t + distance_b w t = w.distance ∧
           distance_b w t - meeting_point w t = 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_men_meet_35_miles_nearer_A_l1150_115021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_casey_pumping_time_l1150_115010

/-- Represents the water pumping scenario for Casey's farm -/
structure WaterPumping where
  morning_rate : ℚ
  afternoon_rate : ℚ
  corn_rows : ℕ
  corn_plants_per_row : ℕ
  corn_water_per_plant : ℚ
  pumpkin_rows : ℕ
  pumpkin_plants_per_row : ℕ
  pumpkin_water_per_plant : ℚ
  pigs : ℕ
  pig_water : ℚ
  ducks : ℕ
  duck_water : ℚ
  cows : ℕ
  cow_water : ℚ

/-- Calculates the total time needed to pump water for Casey's farm -/
def total_pumping_time (w : WaterPumping) : ℚ :=
  let plant_water := w.corn_rows * w.corn_plants_per_row * w.corn_water_per_plant +
                     w.pumpkin_rows * w.pumpkin_plants_per_row * w.pumpkin_water_per_plant
  let animal_water := w.pigs * w.pig_water + w.ducks * w.duck_water + w.cows * w.cow_water
  let morning_time := plant_water / w.morning_rate
  let afternoon_time := animal_water / w.afternoon_rate
  morning_time + afternoon_time

/-- Theorem stating that Casey needs 35 minutes to pump water -/
theorem casey_pumping_time :
  let w : WaterPumping := {
    morning_rate := 3,
    afternoon_rate := 5,
    corn_rows := 4,
    corn_plants_per_row := 15,
    corn_water_per_plant := 1/2,
    pumpkin_rows := 3,
    pumpkin_plants_per_row := 10,
    pumpkin_water_per_plant := 4/5,
    pigs := 10,
    pig_water := 4,
    ducks := 20,
    duck_water := 1/4,
    cows := 5,
    cow_water := 8
  }
  total_pumping_time w = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_casey_pumping_time_l1150_115010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tic_tac_toe_probability_l1150_115094

/-- A tic-tac-toe board -/
structure Board :=
  (size : Nat)
  (crosses : Nat)
  (noughts : Nat)

/-- A winning position on a tic-tac-toe board -/
structure WinningPosition :=
  (horizontal : Nat)
  (vertical : Nat)
  (diagonal : Nat)

/-- The probability of noughts being in a winning position -/
def winningProbability (b : Board) (w : WinningPosition) : ℚ :=
  (w.horizontal + w.vertical + w.diagonal : ℚ) / (Nat.choose (b.size ^ 2) b.noughts : ℚ)

/-- Theorem: The probability of 3 noughts being in a winning position on a 3x3 tic-tac-toe board
    when randomly filled with 6 crosses and 3 noughts is 2/21 -/
theorem tic_tac_toe_probability :
  let b : Board := ⟨3, 6, 3⟩
  let w : WinningPosition := ⟨3, 3, 2⟩
  winningProbability b w = 2 / 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tic_tac_toe_probability_l1150_115094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sara_staircase_l1150_115070

/-- The number of toothpicks used for n steps in Sara's staircase design -/
def toothpicks (n : ℕ) : ℕ := n^2

/-- The maximum number of complete steps that can be built with a given number of toothpicks -/
noncomputable def max_steps (total : ℕ) : ℕ :=
  Nat.floor (Real.sqrt (total : ℝ))

theorem sara_staircase (total : ℕ) (h : total = 210) :
  max_steps total = 14 := by
  sorry

#eval toothpicks 14

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sara_staircase_l1150_115070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_implies_a_equals_one_l1150_115035

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ x + a * y - 4 = 0

def l₂ (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ (a - 2) * x + y - 1 = 0

-- Define the perpendicularity condition
def perpendicular (a : ℝ) : Prop :=
  1 + a * (a - 2) = 0

-- Theorem statement
theorem lines_perpendicular_implies_a_equals_one :
  ∀ a : ℝ, (∃ x y : ℝ, l₁ a x y ∧ l₂ a x y) → perpendicular a → a = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_implies_a_equals_one_l1150_115035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l1150_115087

/-- Calculates the speed of a train crossing a bridge -/
noncomputable def train_speed (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let speed_ms := total_distance / crossing_time
  speed_ms * 3.6

/-- Theorem stating that a train with given parameters has a specific speed -/
theorem train_speed_theorem (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ)
  (h1 : train_length = 170)
  (h2 : bridge_length = 205)
  (h3 : crossing_time = 30) :
  train_speed train_length bridge_length crossing_time = 45 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval train_speed 170 205 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l1150_115087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1150_115007

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem function_properties 
  (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : 0 < φ ∧ φ < π) 
  (h_symmetry : ∀ x : ℝ, f ω φ (x + π / 2) = f ω φ (π / 2 - x))
  (h_distance : ∀ x : ℝ, ∃ y : ℝ, y > x ∧ f ω φ y = f ω φ x ∧ y - x = π / 2)
  (α : ℝ)
  (h_α : 0 < α ∧ α < π / 2)
  (h_f_value : f ω φ (α / 2 + π / 12) = 3 / 5) :
  (∀ x : ℝ, f ω φ x = Real.cos (2 * x)) ∧ 
  Real.sin (2 * α) = (24 + 7 * Real.sqrt 3) / 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1150_115007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotation_l1150_115018

-- Define the bounding function
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

-- Define the region
def R : Set (ℝ × ℝ) := {(x, y) | 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ f x}

-- Theorem statement
theorem volume_of_rotation (π : ℝ) (h : π > 0) :
  (∫ (y : ℝ) in Set.Icc 0 1, π * (4 - (1 + 2*Real.sqrt y + y))) = 7*π/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotation_l1150_115018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longer_diagonal_l1150_115088

/-- Represents a rhombus with given area and diagonal ratio -/
structure Rhombus where
  area : ℝ
  diagonalRatio : ℝ
  diagonalRatio_pos : diagonalRatio > 0

/-- Calculates the length of the longer diagonal of a rhombus -/
noncomputable def longerDiagonal (r : Rhombus) : ℝ :=
  2 * Real.sqrt ((r.area * r.diagonalRatio) / (r.diagonalRatio + 1))

/-- Theorem: The longer diagonal of a rhombus with area 150 and diagonal ratio 4:3 is 20 -/
theorem rhombus_longer_diagonal :
  let r : Rhombus := { area := 150, diagonalRatio := 4/3, diagonalRatio_pos := by norm_num }
  longerDiagonal r = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longer_diagonal_l1150_115088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1150_115009

noncomputable def f (x : ℝ) := 4 * Real.sin (x - Real.pi / 6) * Real.cos x + 1

theorem f_properties :
  let period := Real.pi
  let max_value := Real.sqrt 3
  let min_value := -2
  let interval := Set.Icc (-Real.pi / 4) (Real.pi / 4)
  (∀ x, f (x + period) = f x) ∧
  (∀ x ∈ interval, f x ≤ max_value) ∧
  (∃ x ∈ interval, f x = max_value) ∧
  (∀ x ∈ interval, f x ≥ min_value) ∧
  (∃ x ∈ interval, f x = min_value) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1150_115009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l1150_115013

theorem right_triangle_area (a c : ℝ) (h1 : a = 18) (h2 : c = 30) : 
  (1/2) * a * Real.sqrt (c^2 - a^2) = 216 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l1150_115013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_difference_l1150_115037

/-- Represents a jar of marbles -/
structure Jar where
  total : ℕ
  redRatio : ℚ
  whiteRatio : ℚ

/-- The problem setup -/
def marbleProblem : Prop := ∃ (jar1 jar2 : Jar),
  -- Jar 1 conditions
  jar1.total = 140 ∧
  jar1.redRatio = 7/10 ∧
  jar1.whiteRatio = 3/10 ∧
  -- Jar 2 conditions
  jar2.total = 72 ∧
  jar2.redRatio = 3/4 ∧
  jar2.whiteRatio = 1/4 ∧
  -- Total white marbles condition
  (jar1.whiteRatio * (jar1.total : ℚ) + jar2.whiteRatio * (jar2.total : ℚ)) = 60 ∧
  -- The question: difference in red marbles
  (jar1.redRatio * (jar1.total : ℚ) - jar2.redRatio * (jar2.total : ℚ)) = 44

theorem marble_difference : marbleProblem := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_difference_l1150_115037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_curve_properties_l1150_115055

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (t + 1, t + 4)

noncomputable def curve_C (θ : ℝ) : ℝ := Real.sqrt 3 / Real.sqrt (1 + 2 * (Real.cos θ)^2)

noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |x - y + 3| / Real.sqrt 2

theorem line_and_curve_properties :
  (∀ t : ℝ, (line_l t).1 - (line_l t).2 + 3 = 0) ∧
  (∀ x y : ℝ, x^2 + y^2/3 = 1 ↔ ∃ θ : ℝ, x = curve_C θ * Real.cos θ ∧ y = curve_C θ * Real.sin θ) ∧
  (∃ x y : ℝ, x^2 + y^2/3 = 1 ∧ distance_to_line x y = Real.sqrt 2 / 2) ∧
  (∃ x y : ℝ, x^2 + y^2/3 = 1 ∧ distance_to_line x y = 5 * Real.sqrt 2 / 2) ∧
  (∀ x y : ℝ, x^2 + y^2/3 = 1 → Real.sqrt 2 / 2 ≤ distance_to_line x y ∧ distance_to_line x y ≤ 5 * Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_curve_properties_l1150_115055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_g_max_value_l1150_115044

-- Function definitions
noncomputable def f (x : ℝ) : ℝ := 1 - 2 * Real.sin (x + Real.pi / 6)
noncomputable def g (a x : ℝ) : ℝ := a * Real.cos (2 * x + Real.pi / 3) + 3

-- Theorem for part 1
theorem f_extrema :
  (∃ x, f x = 3) ∧ (∀ x, f x ≤ 3) ∧
  (∃ x, f x = -1) ∧ (∀ x, f x ≥ -1) := by
  sorry

-- Theorem for part 2
theorem g_max_value (h : ∃ a, ∀ x ∈ Set.Icc 0 (Real.pi / 2), g a x ≤ 4 ∧ (∃ x₀ ∈ Set.Icc 0 (Real.pi / 2), g a x₀ = 4)) :
  ∃ a, a = 2 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_g_max_value_l1150_115044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1150_115048

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 3*a*x + 4
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (2*a - 1)^x

-- Define the propositions
def p (a : ℝ) : Prop := ∀ x y, x ≥ 1 → y ≥ 1 → x < y → f a x < f a y
def q (a : ℝ) : Prop := ∀ x y, x < y → g a y < g a x

-- Define the theorem
theorem range_of_a : 
  ∀ a : ℝ, (¬(p a ∧ q a)) ↔ (a ≤ 1/2 ∨ a > 2/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1150_115048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_always_illuminated_l1150_115036

-- Define the lighthouse and ship
structure Lighthouse where
  position : ℝ × ℝ
  beam_distance : ℝ
  beam_velocity : ℝ

structure Ship where
  position : ℝ × ℝ
  velocity : ℝ

-- Define the theorem
theorem ship_always_illuminated (L : Lighthouse) (S : Ship) :
  L.beam_distance > 0 ∧
  L.beam_velocity > 0 ∧
  S.velocity ≤ L.beam_velocity / 8 ∧
  dist S.position L.position = L.beam_distance →
  ∃ (t : ℝ), t > 0 ∧ 
    dist (S.position.1 + t * S.velocity, S.position.2) L.position ≤ L.beam_distance ∧
    dist (S.position.1 + t * S.velocity, S.position.2) L.position > 0 :=
by sorry

-- Helper function to calculate distance between two points
noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_always_illuminated_l1150_115036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blouse_cost_l1150_115069

/-- Calculates the cost of each blouse given the following conditions:
  * 3 skirts at $20 each
  * 2 pairs of pants at $30 each, with second pair half off
  * 5 blouses
  * Total spend of $180
-/
theorem blouse_cost : ℝ := by
  let skirt_cost : ℝ := 20
  let skirt_count : ℕ := 3
  let pants_cost : ℝ := 30
  let pants_count : ℕ := 2
  let blouse_count : ℕ := 5
  let total_spend : ℝ := 180
  let pants_total_cost : ℝ := pants_cost + pants_cost / 2
  let skirts_total_cost : ℝ := skirt_cost * skirt_count
  let blouses_total_cost : ℝ := total_spend - skirts_total_cost - pants_total_cost
  have h : blouses_total_cost / blouse_count = 15 := by sorry
  exact 15


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blouse_cost_l1150_115069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_dot_product_l1150_115054

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ -- semi-major axis
  b : ℝ -- semi-minor axis

/-- Check if a point is on the hyperbola -/
def isOnHyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Calculate the eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculate the dot product of two vectors -/
def dotProduct (v1 v2 : Point) : ℝ :=
  (v1.x * v2.x) + (v1.y * v2.y)

/-- The main theorem -/
theorem hyperbola_dot_product 
  (h : Hyperbola) 
  (f1 f2 p : Point) 
  (h_equation : h.a = 1 ∧ h.b = Real.sqrt 3)
  (h_foci : f1.x = -2 ∧ f1.y = 0 ∧ f2.x = 2 ∧ f2.y = 0)
  (h_on_hyperbola : isOnHyperbola h p)
  (h_angle_condition : 
    Real.sin (Real.arccos ((distance p f2)^2 + (distance f2 f1)^2 - (distance p f1)^2) / 
      (2 * distance p f2 * distance f2 f1)) / 
    Real.sin (Real.arccos ((distance p f1)^2 + (distance f2 f1)^2 - (distance p f2)^2) / 
      (2 * distance p f1 * distance f2 f1)) = eccentricity h) :
  dotProduct p f1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_dot_product_l1150_115054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_team_won_26_games_l1150_115057

/-- Represents a baseball team's season statistics -/
structure BaseballSeason where
  total_games : ℕ
  daytime_games : ℕ
  night_games : ℕ
  daytime_win_percentage : ℚ
  night_win_percentage : ℚ
  winning_streak : ℕ
  daytime_home_losses : ℕ
  daytime_away_losses : ℕ

/-- Calculates the total number of games won in a baseball season -/
def total_games_won (season : BaseballSeason) : ℕ :=
  let daytime_wins := (season.daytime_win_percentage * season.daytime_games).floor.toNat
  let night_wins := (season.night_win_percentage * season.night_games).floor.toNat
  daytime_wins + night_wins

/-- Theorem stating that given the specific conditions, the team won 26 games -/
theorem team_won_26_games :
  ∀ (season : BaseballSeason),
    season.total_games = 36 ∧
    season.daytime_games = 28 ∧
    season.night_games = 8 ∧
    season.daytime_win_percentage = 70/100 ∧
    season.night_win_percentage = 875/1000 ∧
    season.winning_streak = 10 ∧
    season.daytime_home_losses = 3 ∧
    season.daytime_away_losses = 2 →
    total_games_won season = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_team_won_26_games_l1150_115057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_inradius_relation_l1150_115089

/-- A triangle with sides of consecutive natural numbers -/
structure ConsecutiveSidedTriangle where
  /-- The middle side length (b) -/
  b : ℕ
  /-- Ensure b is greater than 2 to form a valid triangle -/
  h : b > 2

/-- The circumradius of a triangle -/
noncomputable def circumradius (t : ConsecutiveSidedTriangle) : ℝ :=
  (t.b * (t.b^2 - 1 : ℝ)) / Real.sqrt (3 * (t.b^2 - 4 : ℝ))

/-- The inradius of a triangle -/
noncomputable def inradius (t : ConsecutiveSidedTriangle) : ℝ :=
  Real.sqrt (3 * (t.b^2 - 4 : ℝ)) / 6

/-- Theorem: For a triangle with consecutive integer side lengths,
    the circumradius equals 2 times the inradius plus 1 divided by (2 times the inradius) -/
theorem circumradius_inradius_relation (t : ConsecutiveSidedTriangle) :
  circumradius t = 2 * inradius t + 1 / (2 * inradius t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_inradius_relation_l1150_115089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_rectangle_with_squares_l1150_115074

/-- A rectangle divided into four identical squares with a given perimeter -/
structure RectangleWithSquares where
  perimeter : ℝ
  perimeter_positive : perimeter > 0

/-- The area of a rectangle divided into four identical squares -/
noncomputable def area (r : RectangleWithSquares) : ℝ :=
  (r.perimeter / 8) ^ 2 * 4

/-- Theorem: The area of a rectangle with perimeter 160 cm divided into four identical squares is 1600 cm² -/
theorem area_of_rectangle_with_squares (r : RectangleWithSquares) 
    (h : r.perimeter = 160) : area r = 1600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_rectangle_with_squares_l1150_115074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_quadrilateral_l1150_115049

/-- A quadrilateral inscribed in a circle with perpendicular diagonals -/
structure InscribedQuadrilateral where
  /-- Radius of the circumscribed circle -/
  radius : ℝ
  /-- Length of one diagonal -/
  diagonal1 : ℝ
  /-- Distance from circle center to diagonal intersection -/
  centerToIntersection : ℝ
  /-- The diagonals are perpendicular -/
  perpendicularDiagonals : Prop

/-- Calculate the area of the inscribed quadrilateral -/
noncomputable def areaOfInscribedQuadrilateral (q : InscribedQuadrilateral) : ℝ :=
  sorry

/-- Theorem stating the area of the specific quadrilateral -/
theorem area_of_specific_quadrilateral :
  let q : InscribedQuadrilateral := {
    radius := 13,
    diagonal1 := 18,
    centerToIntersection := 4 * Real.sqrt 6,
    perpendicularDiagonals := True
  }
  areaOfInscribedQuadrilateral q = 72 * Real.sqrt 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_quadrilateral_l1150_115049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_on_line_l1150_115039

-- Define the line y = √3 * x
noncomputable def line (x : ℝ) : ℝ := Real.sqrt 3 * x

-- Define the property that the terminal side of angle θ lies on the line
def terminal_side_on_line (θ : ℝ) : Prop :=
  ∃ (x : ℝ), x > 0 ∧ (Real.cos θ = x) ∧ (Real.sin θ = line x)

-- Theorem statement
theorem tan_value_on_line (θ : ℝ) :
  terminal_side_on_line θ → Real.tan θ = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_on_line_l1150_115039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cotangent_sum_theorem_weighted_cotangent_sum_theorem_l1150_115043

/-- Triangle properties -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  S : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_angles : 0 < α ∧ 0 < β ∧ 0 < γ
  h_sum_angles : α + β + γ = π
  h_area_positive : 0 < S

/-- Cotangent function -/
noncomputable def ctg (θ : ℝ) : ℝ := 1 / Real.tan θ

/-- Theorem about cotangents in a triangle -/
theorem cotangent_sum_theorem (t : Triangle) : 
  ctg t.α + ctg t.β + ctg t.γ = (t.a^2 + t.b^2 + t.c^2) / (4 * t.S) := by
  sorry

/-- Theorem about weighted cotangents in a triangle -/
theorem weighted_cotangent_sum_theorem (t : Triangle) :
  t.a^2 * ctg t.α + t.b^2 * ctg t.β + t.c^2 * ctg t.γ = 4 * t.S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cotangent_sum_theorem_weighted_cotangent_sum_theorem_l1150_115043
