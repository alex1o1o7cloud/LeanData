import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_function_result_l784_78456

noncomputable def f (x : ℝ) : ℝ := x + 3
noncomputable def g (x : ℝ) : ℝ := x / 4
noncomputable def h (x : ℝ) : ℝ := x + 1

noncomputable def f_inv (x : ℝ) : ℝ := x - 3
noncomputable def g_inv (x : ℝ) : ℝ := 4 * x
noncomputable def h_inv (x : ℝ) : ℝ := x - 1

theorem composite_function_result :
  f (h (g_inv (h_inv (f_inv (h (g (f 20))))))) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_function_result_l784_78456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_divisibility_l784_78444

theorem power_of_two_divisibility (n a b : ℕ) : 
  (2:ℕ)^n = 10*a + b → b < 10 → n > 3 → 6 ∣ (a*b) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_divisibility_l784_78444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_coordinates_l784_78482

/-- The complex number z defined as (3-2i) / i^2015 -/
noncomputable def z : ℂ := (3 - 2 * Complex.I) / (Complex.I ^ 2015)

/-- Theorem stating that the real part of z is 2 and the imaginary part is 3 -/
theorem z_coordinates : Complex.re z = 2 ∧ Complex.im z = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_coordinates_l784_78482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_13_equals_26_l784_78410

/-- An arithmetic sequence with the given property -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  property : a 3 + a 7 + a 11 = 6

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The theorem to be proved -/
theorem sum_13_equals_26 (seq : ArithmeticSequence) : sum_n seq 13 = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_13_equals_26_l784_78410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l784_78400

def b (m : ℕ+) : ℕ := 2 * (Nat.sqrt (2 * m - 1) + 1)

theorem sequence_formula :
  ∃ (e f g : ℤ), (∀ (m : ℕ+), (b m : ℤ) = e * Int.floor ((m + f : ℝ) ^ (1/3 : ℝ)) + g) ∧ e + f + g = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l784_78400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_option_is_90_dollars_l784_78424

/-- Represents the car rental problem with given conditions -/
structure CarRentalProblem where
  trip_distance : ℕ  -- Distance in km (one way)
  first_option_cost : ℕ  -- Cost in dollars (excluding gasoline)
  gasoline_efficiency : ℕ  -- km per liter
  gasoline_cost : ℚ  -- Cost per liter in dollars
  savings : ℕ  -- Amount saved by choosing first option over second

/-- Calculates the cost of the second car rental option -/
def second_option_cost (p : CarRentalProblem) : ℚ :=
  let total_distance := 2 * p.trip_distance
  let gasoline_needed := total_distance / p.gasoline_efficiency
  let gasoline_total_cost := gasoline_needed * p.gasoline_cost
  (p.first_option_cost : ℚ) + gasoline_total_cost + (p.savings : ℚ)

/-- Theorem stating that the second option cost is $90 for the given problem -/
theorem second_option_is_90_dollars (p : CarRentalProblem) 
    (h1 : p.trip_distance = 150)
    (h2 : p.first_option_cost = 50)
    (h3 : p.gasoline_efficiency = 15)
    (h4 : p.gasoline_cost = 9/10)
    (h5 : p.savings = 22) :
  second_option_cost p = 90 := by
  sorry

#eval second_option_cost {
  trip_distance := 150,
  first_option_cost := 50,
  gasoline_efficiency := 15,
  gasoline_cost := 9/10,
  savings := 22
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_option_is_90_dollars_l784_78424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_direction_vector_l784_78416

/-- The line equation y = (3x - 4) / 4 -/
def line_equation (x y : ℝ) : Prop :=
  y = (3 * x - 4) / 4

/-- The parameterization of the line -/
def parameterization (v d : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
  (v.1 + t * d.1, v.2 + t * d.2)

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem line_direction_vector :
  ∃ (d : ℝ × ℝ),
    (∀ (x y : ℝ), line_equation x y →
      ∃ (t : ℝ), parameterization (3, 1) d t = (x, y)) ∧
    (∀ (t : ℝ), let (x, y) := parameterization (3, 1) d t
      x ≥ 3 → distance (x, y) (3, 1) = t) →
    d = (7/2, 5/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_direction_vector_l784_78416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_cosine_existence_l784_78438

theorem rational_cosine_existence (θ : ℝ) (k : ℕ+) (h : ∃ q : ℚ, Real.cos (k * θ) = q) :
  ∃ (n : ℕ+), n > k ∧ (∃ q1 : ℚ, Real.cos ((n - 1) * θ) = q1) ∧ (∃ q2 : ℚ, Real.cos (n * θ) = q2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_cosine_existence_l784_78438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_num_cycles_characterization_l784_78485

/-- The number of cycles in the permutation x ↦ 3x (mod m) on {1, 2, ..., m-1} -/
def num_cycles (m : ℕ) : ℕ := sorry

theorem odd_num_cycles_characterization (m : ℕ) (h1 : m > 1) (h2 : ¬(3 ∣ m)) :
  Odd (num_cycles m) ↔ m % 12 ∈ ({2, 5, 7, 10} : Finset ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_num_cycles_characterization_l784_78485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_given_parallel_chords_l784_78411

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the two lines
def Line1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | Real.sqrt 3 * p.1 - p.2 + 2 = 0}
def Line2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | Real.sqrt 3 * p.1 - p.2 - 10 = 0}

-- Define the chord length
def ChordLength : ℝ := 8

theorem circle_area_given_parallel_chords 
  (C : Set (ℝ × ℝ))
  (h1 : ∃ center radius, C = Circle center radius)
  (h2 : ∃ p1 p2, p1 ∈ C ∧ p2 ∈ C ∧ p1 ∈ Line1 ∧ p2 ∈ Line1 ∧ (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 = ChordLength^2)
  (h3 : ∃ q1 q2, q1 ∈ C ∧ q2 ∈ C ∧ q1 ∈ Line2 ∧ q2 ∈ Line2 ∧ (q1.1 - q2.1)^2 + (q1.2 - q2.2)^2 = ChordLength^2) :
  ∃ center radius, C = Circle center radius ∧ π * radius^2 = 25 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_given_parallel_chords_l784_78411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_with_adjacent_digits_divisible_by_23_l784_78407

/-- A function that checks if a number is divisible by 23 -/
def isDivisibleBy23 (n : ℕ) : Prop := n % 23 = 0

/-- A function that extracts two adjacent digits from a natural number -/
def adjacentDigits (n : ℕ) (i : ℕ) : ℕ :=
  (n / 10^i % 100)

/-- A predicate that checks if all adjacent digit pairs in a number are divisible by 23 -/
def allAdjacentDigitsDivisibleBy23 (n : ℕ) : Prop :=
  ∀ i, i < Nat.log 10 n → isDivisibleBy23 (adjacentDigits n i)

/-- The main theorem stating that 46923 is the largest number satisfying the condition -/
theorem largest_number_with_adjacent_digits_divisible_by_23 :
  (allAdjacentDigitsDivisibleBy23 46923) ∧
  (∀ m : ℕ, m > 46923 → ¬(allAdjacentDigitsDivisibleBy23 m)) := by
  sorry

#check largest_number_with_adjacent_digits_divisible_by_23

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_with_adjacent_digits_divisible_by_23_l784_78407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_focus_of_specific_parabola_l784_78436

/-- Definition of the focus of a parabola -/
noncomputable def focus_of_parabola (f : ℝ → ℝ) : ℝ × ℝ := sorry

/-- The focus of a parabola y = ax^2 + k is at (0, 1/(4a) + k) -/
theorem parabola_focus (a k : ℝ) (h : a ≠ 0) :
  let f : ℝ × ℝ := (0, 1 / (4 * a) + k)
  f = focus_of_parabola (fun x => a * x^2 + k) :=
by sorry

/-- The focus of the parabola y = 9x^2 + 5 is at (0, 181/36) -/
theorem focus_of_specific_parabola :
  focus_of_parabola (fun x => 9 * x^2 + 5) = (0, 181/36) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_focus_of_specific_parabola_l784_78436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_point_coordinates_l784_78419

noncomputable section

/-- The ellipse with parametric equations x = 4cosα and y = 2√3sinα -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ α : ℝ, p.1 = 4 * Real.cos α ∧ p.2 = 2 * Real.sqrt 3 * Real.sin α}

/-- The first quadrant -/
def FirstQuadrant : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0}

/-- The slope of a line through the origin and a point -/
def SlopeFromOrigin (p : ℝ × ℝ) : ℝ := p.2 / p.1

theorem ellipse_point_coordinates :
  ∀ p : ℝ × ℝ,
  p ∈ Ellipse →
  p ∈ FirstQuadrant →
  SlopeFromOrigin p = Real.tan (π / 3) →
  p = (4 * Real.sqrt 5 / 5, 4 * Real.sqrt 15 / 5) := by
    sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_point_coordinates_l784_78419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_max_perimeter_l784_78441

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (2 + (1/2) * t, Real.sqrt 3 + (Real.sqrt 3 / 2) * t)

-- Define curve C in polar coordinates
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 2 * Real.sin θ)

-- Define point M
noncomputable def point_M : ℝ × ℝ := (2, Real.sqrt 3)

-- Define the transformation for curve C'
noncomputable def transform (p : ℝ × ℝ) : ℝ × ℝ := (Real.sqrt 3 * p.1, p.2)

-- Statement for part (I)
theorem distance_sum :
  ∃ A B : ℝ × ℝ, (∃ t : ℝ, A = line_l t) ∧ (∃ t : ℝ, B = line_l t) ∧
  (∃ θ : ℝ, A = curve_C θ) ∧ (∃ θ : ℝ, B = curve_C θ) ∧
  Real.sqrt ((A.1 - point_M.1)^2 + (A.2 - point_M.2)^2) +
  Real.sqrt ((B.1 - point_M.1)^2 + (B.2 - point_M.2)^2) = Real.sqrt 13 := by
sorry

-- Statement for part (II)
theorem max_perimeter :
  ∃ max_p : ℝ, max_p = 16 ∧
  ∀ x y : ℝ, 0 < x → x < 2 * Real.sqrt 3 → 0 < y → y < 2 →
  (x^2 / 12 + y^2 / 3 = 1) →
  4 * x + 4 * y ≤ max_p := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_max_perimeter_l784_78441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_max_k_value_l784_78406

open Real

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * log x

-- Theorem for the minimum value of f(x)
theorem f_min_value :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≥ f x ∧ f x = -1/e :=
sorry

-- Theorem for the maximum value of k
theorem max_k_value :
  (∃ (k : ℤ), ∀ (x : ℝ), x > 2 → f x ≥ k * x - 2 * (k + 1)) ∧
  (∀ (k : ℤ), (∀ (x : ℝ), x > 2 → f x ≥ k * x - 2 * (k + 1)) → k ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_max_k_value_l784_78406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_complement_union_M_subset_A_iff_l784_78414

-- Define the sets A, B, and M
def A : Set ℝ := {x | x < -4 ∨ x > 2}
def B : Set ℝ := {x | -1 ≤ Real.exp (Real.log 2 * x) - 1 ∧ Real.exp (Real.log 2 * x) - 1 ≤ 6}
def M (k : ℝ) : Set ℝ := {x | 2*k - 1 ≤ x ∧ x ≤ 2*k + 1}

-- State the theorems to be proven
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 4} := by sorry

theorem complement_union :
  (Set.univ \ A) ∪ (Set.univ \ B) = {x : ℝ | x ≤ 2 ∨ x > 4} := by sorry

theorem M_subset_A_iff (k : ℝ) : M k ⊆ A ↔ k < -5/2 ∨ k > 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_complement_union_M_subset_A_iff_l784_78414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l784_78418

/-- A polynomial of degree 4 with real coefficients -/
def myPolynomial (a b c d : ℝ) : ℂ → ℂ := fun x ↦ x^4 + a*x^3 + b*x^2 + c*x + d

theorem sum_of_coefficients (a b c d : ℝ) :
  let g := myPolynomial a b c d
  (g (3*I) = 0) → (g (1 + 2*I) = 0) →
  a + b + c + d = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l784_78418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l784_78498

noncomputable def f (x : ℝ) : ℝ := 1 / Real.log x + Real.sqrt (2 - x)

theorem domain_of_f : 
  {x : ℝ | (0 < x ∧ x < 1) ∨ (1 < x ∧ x ≤ 2)} = Set.Ioo 0 1 ∪ Set.Ico 1 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l784_78498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin6_plus_cos6_l784_78486

theorem sin6_plus_cos6 (θ : ℝ) (h : Real.cos (2 * θ) = 1/5) : 
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 7/25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin6_plus_cos6_l784_78486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_colorings_l784_78472

/-- The number of faces in a regular dodecahedron -/
def num_faces : ℕ := 12

/-- The order of rotational symmetry around a face of a regular dodecahedron -/
def face_symmetry : ℕ := 5

/-- The number of distinguishable colorings of a regular dodecahedron -/
def distinguishable_colorings : ℕ := (num_faces - 1).factorial / face_symmetry

theorem dodecahedron_colorings :
  distinguishable_colorings = 7983360 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_colorings_l784_78472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l784_78466

-- Define the lines
def line1 (x y : ℝ) (a : ℝ) : Prop := 2 * x - a * y - 1 = 0
def line2 (x y : ℝ) : Prop := x + 2 * y = 0

-- Define the slope of a line
def lineSlope (m : ℝ) (x y : ℝ) : Prop := y = m * x

-- Define perpendicular lines
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, line1 x y a → lineSlope (-1/2) x y) →
  (∀ x y : ℝ, line2 x y → lineSlope (2/a) x y) →
  perpendicular (-1/2) (2/a) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l784_78466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cos_R_l784_78489

theorem triangle_cos_R (P Q R : ℝ) (h_triangle : P + Q + R = Real.pi) 
  (h_sin_P : Real.sin P = 4/5) (h_cos_Q : Real.cos Q = 12/13) : 
  Real.cos R = -16/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cos_R_l784_78489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_babylonian_formula_accuracy_l784_78479

noncomputable def babylonianArea (a b c d : ℝ) : ℝ := ((a + b) / 2) * ((c + d) / 2)

noncomputable def rectangleArea (length width : ℝ) : ℝ := length * width

def isRectangle (a b c d : ℝ) : Prop := a = b ∧ c = d

theorem babylonian_formula_accuracy (a b c d : ℝ) :
  isRectangle a b c d ↔ babylonianArea a b c d = rectangleArea a c := by
  sorry

#check babylonian_formula_accuracy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_babylonian_formula_accuracy_l784_78479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_longest_side_l784_78475

noncomputable section

def triangle_area (t : Set (ℝ × ℝ)) : ℝ := sorry
def triangle_perimeter (t : Set (ℝ × ℝ)) : ℝ := sorry
def angle_bisector_to_incenter (t : Set (ℝ × ℝ)) : ℝ × ℝ := sorry
def distance (p q : ℝ × ℝ) : ℝ := sorry
def max_side_length (t : Set (ℝ × ℝ)) : ℝ := sorry

theorem triangle_longest_side (A B C : ℝ × ℝ) (O : ℝ × ℝ) :
  let triangle := {A, B, C}
  let area := 4 * Real.sqrt 21
  let perimeter := 24
  let bisector_length := Real.sqrt 30 / 3
  area = triangle_area triangle ∧
  perimeter = triangle_perimeter triangle ∧
  bisector_length = distance O (angle_bisector_to_incenter triangle) →
  11 = max_side_length triangle :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_longest_side_l784_78475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_problem_l784_78487

-- Define the quadratic function
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the condition for the graph lying above y = -2x
def above_line (a b c : ℝ) : Prop :=
  ∀ x, 1 < x → x < 3 → quadratic_function a b c x > -2 * x

-- Define the condition for two equal real roots
def equal_roots (a b c : ℝ) : Prop :=
  ∃ x, a * x^2 + b * x + (6 * a + c) = 0 ∧
       ∀ y, a * y^2 + b * y + (6 * a + c) = 0 → y = x

-- Define the theorem
theorem quadratic_problem (a b c : ℝ) 
  (h1 : above_line a b c) 
  (h2 : equal_roots a b c) :
  (a = -(1/5) ∧ b = -(6/5) ∧ c = -(3/5)) ∧
  (∃ m, (∀ x, quadratic_function a b c m ≥ quadratic_function a b c x) → 
    -2 + Real.sqrt 3 < a ∧ a < 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_problem_l784_78487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_has_inverse_l784_78405

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 1 then Real.log (1 - x)
  else if -1 < x ∧ x < 0 then Real.log (1 / (1 + x))
  else 0  -- Default value for x outside the domain

-- Theorem statement
theorem f_odd_and_has_inverse :
  (∀ x, -1 < x ∧ x < 1 → f (-x) = -f x) ∧
  (∃ g : ℝ → ℝ, ∀ x, -1 < x ∧ x < 1 → g (f x) = x ∧ f (g x) = x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_has_inverse_l784_78405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_in_interval_l784_78413

noncomputable def g (n : ℕ) (x : ℝ) : ℝ := Real.sin x ^ n + Real.cos x ^ n

theorem no_solutions_in_interval :
  ∀ x ∈ Set.Icc 0 (2 * Real.pi),
    10 * g 8 x - 6 * g 10 x ≠ 2 * g 2 x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_in_interval_l784_78413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l784_78465

/-- Predicate to check if a given point is an axis of symmetry for a function -/
def IsSymmetryAxis (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ h, f (x - h) = f (x + h)

/-- Predicate to check if a given value is the maximum value of a function -/
def IsMaxValue (f : ℝ → ℝ) (max : ℝ) : Prop :=
  ∀ x, f x ≤ max

/-- Given a function y = sin x + a cos x with an axis of symmetry at x = 5π/3,
    the maximum value of g(x) = a sin x + cos x is 2√3/3 -/
theorem max_value_of_g (a : ℝ) : 
  (∃ y : ℝ → ℝ, y = λ x ↦ Real.sin x + a * Real.cos x) →
  (∃ x : ℝ, x = 5 * Real.pi / 3 ∧ IsSymmetryAxis y x) →
  (∃ g : ℝ → ℝ, g = λ x ↦ a * Real.sin x + Real.cos x) →
  (∃ max_g : ℝ, IsMaxValue g max_g ∧ max_g = 2 * Real.sqrt 3 / 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l784_78465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_equivalence_l784_78435

def solution_set (a b : ℝ) : Prop :=
  (a = 0) ∨ (b = 0) ∨ (a = b) ∨ (∃ (m n : ℤ), a = ↑m ∧ b = ↑n)

theorem floor_equation_equivalence (a b : ℝ) :
  (∀ n : ℕ+, a * ⌊b * ↑n⌋ = b * ⌊a * ↑n⌋) ↔ solution_set a b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_equivalence_l784_78435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_map_distance_conversion_l784_78497

/-- Represents the scale of a map --/
structure MapScale where
  map_distance : ℚ
  actual_distance : ℚ

/-- Calculates the actual distance given a map distance and a map scale --/
def calculate_actual_distance (map_distance : ℚ) (scale : MapScale) : ℚ :=
  map_distance * (scale.actual_distance / scale.map_distance)

theorem map_distance_conversion :
  let scale : MapScale := { map_distance := 312, actual_distance := 136 }
  let map_distance : ℚ := 42
  let actual_distance : ℚ := calculate_actual_distance map_distance scale
  abs (actual_distance - 18.31) < 0.01 := by
    sorry

#eval calculate_actual_distance 42 { map_distance := 312, actual_distance := 136 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_map_distance_conversion_l784_78497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_fifteenths_pi_zero_l784_78483

theorem cos_sum_fifteenths_pi_zero :
  Real.cos (π / 15) + Real.cos (4 * π / 15) + Real.cos (7 * π / 15) + Real.cos (10 * π / 15) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_fifteenths_pi_zero_l784_78483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l784_78432

theorem equation_solution (b x : ℝ) (hb : b > 1) (hx : x > 0) 
  (heq : (3*x)^(Real.log 3 / Real.log b) - (5*x)^(Real.log 5 / Real.log b) = 0) : 
  x = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l784_78432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mutual_fund_share_price_increase_l784_78448

/-- Calculates the percent increase between two prices -/
noncomputable def percentIncrease (initialPrice finalPrice : ℝ) : ℝ :=
  (finalPrice - initialPrice) / initialPrice * 100

/-- Theorem about mutual fund share price increase -/
theorem mutual_fund_share_price_increase (P : ℝ) (hP : P > 0) :
  let firstQuarterPrice := P * 1.3
  let secondQuarterPrice := P * 1.75
  abs (percentIncrease firstQuarterPrice secondQuarterPrice - 34.62) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mutual_fund_share_price_increase_l784_78448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scientist_probability_l784_78476

-- Define the events as axioms
axiom R : Prop
axiom A : Prop
axiom B : Prop

-- Define the probabilities
noncomputable def P_R : ℝ := 1 / 2
noncomputable def P_not_R : ℝ := 1 / 2
noncomputable def P_A_given_R : ℝ := 0.8
noncomputable def P_B_given_R : ℝ := 0.05
noncomputable def P_A_given_not_R : ℝ := 0.9
noncomputable def P_B_given_not_R : ℝ := 0.02

-- Define the conditional probability we want to prove
noncomputable def P_R_given_A_and_B : ℝ := 
  (P_R * P_A_given_R * P_B_given_R) / 
  ((P_R * P_A_given_R * P_B_given_R) + (P_not_R * P_A_given_not_R * P_B_given_not_R))

theorem scientist_probability : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |P_R_given_A_and_B - 0.69| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scientist_probability_l784_78476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_theorem_l784_78468

-- Define the variables and constants
variable (a b : ℝ)

-- Define the condition that the solution set of ax > b is (-∞, 1/5)
def solution_set_condition (a b : ℝ) : Prop :=
  ∀ x : ℝ, a * x > b ↔ x < (1/5 : ℝ)

-- Define the inequality we want to solve
def target_inequality (a b x : ℝ) : Prop :=
  a * x^2 + b * x - (4/5 : ℝ) * a > 0

-- State the theorem
theorem solution_set_theorem (a b : ℝ) (h : solution_set_condition a b) :
  ∀ x : ℝ, target_inequality a b x ↔ -1 < x ∧ x < (4/5 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_theorem_l784_78468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_time_period_l784_78429

/-- The time period (in years) for which a sum is invested at simple interest -/
noncomputable def timePeriod (principal : ℝ) (rate1 : ℝ) (rate2 : ℝ) (interestDiff : ℝ) : ℝ :=
  interestDiff / (principal * (rate1 - rate2))

/-- Theorem stating that the time period is 2 years for the given conditions -/
theorem investment_time_period :
  timePeriod 15000 0.15 0.12 900 = 2 := by
  -- Unfold the definition of timePeriod
  unfold timePeriod
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_time_period_l784_78429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bookseller_loss_percentage_l784_78408

noncomputable def cost_price : ℝ → ℝ := λ n => n  -- Cost price per book
noncomputable def selling_price : ℝ → ℝ := λ n => (4/5) * n  -- Selling price per book

theorem bookseller_loss_percentage :
  ∀ (c : ℝ), c > 0 →
  20 * (cost_price c) = 25 * (selling_price c) →
  (cost_price c - selling_price c) / (cost_price c) * 100 = 20 := by
  sorry

#check bookseller_loss_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bookseller_loss_percentage_l784_78408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_equals_one_l784_78430

/-- Defines the sequence term at position n -/
def sequenceTerm (n : ℕ) : ℤ :=
  match n % 5 with
  | 0 => n
  | 1 => -n
  | 2 => -n
  | 3 => n
  | 4 => -n
  | _ => 0  -- This case should never occur, but it's needed for completeness

/-- The sum of the sequence from 1 to 1996 -/
def sequenceSum : ℤ :=
  (List.range 1996).map (fun i => sequenceTerm (i + 1)) |>.sum

theorem sequence_sum_equals_one : sequenceSum = 1 := by
  sorry

#eval sequenceSum  -- This line is optional, but it can help verify the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_equals_one_l784_78430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_property_l784_78454

def sum_of_divisors (n : ℕ) : ℕ := (Nat.divisors n).sum id

def num_of_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem divisor_sum_property (n : ℕ) :
  (sum_of_divisors n - n + num_of_divisors n - 1 = n) →
  ∃ m : ℕ, n = 2 * m^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_property_l784_78454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_value_l784_78402

/-- An ellipse with foci on the x-axis -/
structure Ellipse where
  m : ℝ
  equation : ∀ (x y : ℝ), x^2 / m + y^2 / 6 = 1
  foci_on_x_axis : m > 6

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := 
  Real.sqrt (1 - 6 / e.m)

/-- Theorem: For an ellipse with equation x^2/m + y^2/6 = 1, 
    foci on the x-axis, and eccentricity 1/2, m = 8 -/
theorem ellipse_m_value (e : Ellipse) 
    (h : eccentricity e = 1/2) : e.m = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_value_l784_78402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_average_l784_78433

theorem correct_average (n : ℕ) (incorrect_avg : ℚ) (incorrect_readings : List (ℚ × ℚ)) :
  n = 20 ∧
  incorrect_avg = 287/10 ∧
  incorrect_readings = [(753/10, 553/10), (622/10, 422/10), (891/10, 691/10)] →
  (n * incorrect_avg + (incorrect_readings.map (λ (correct, incorrect) => correct - incorrect)).sum) / n = 317/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_average_l784_78433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_f_evaluate_trig_sum_l784_78458

-- Define the function f as noncomputable
noncomputable def f (α : ℝ) : ℝ := 
  (Real.sin (α + 3/2 * Real.pi) * Real.sin (-α + Real.pi) * Real.cos (α + Real.pi/2)) / 
  (Real.cos (-α - Real.pi) * Real.cos (α - Real.pi/2) * Real.tan (α + Real.pi))

-- Theorem 1
theorem simplify_f : ∀ α : ℝ, f α = -Real.cos α := by sorry

-- Theorem 2
theorem evaluate_trig_sum : 
  Real.tan (675 * Real.pi / 180) + Real.sin (-330 * Real.pi / 180) + Real.cos (960 * Real.pi / 180) = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_f_evaluate_trig_sum_l784_78458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_identity_l784_78451

-- Define the piecewise function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then 2*a*x + b else 5 - 2*x

-- State the theorem
theorem function_composition_identity (a b : ℝ) :
  (∀ x, f a b (f a b x) = x) → a + b = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_identity_l784_78451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_with_circles_sides_l784_78461

/-- An isosceles trapezoid with two inscribed circles -/
structure IsoscelesTrapezoidWithCircles where
  R : ℝ
  -- Assume R > 0
  R_pos : R > 0

/-- The sides of the isosceles trapezoid with two inscribed circles -/
noncomputable def trapezoidSides (t : IsoscelesTrapezoidWithCircles) : Finset ℝ :=
  {2 * t.R * Real.sqrt 2, 2 * t.R * Real.sqrt 2, 2 * t.R * Real.sqrt 2, 2 * t.R * (2 + Real.sqrt 2)}

/-- Theorem: The sides of the isosceles trapezoid with two inscribed circles are as calculated -/
theorem isosceles_trapezoid_with_circles_sides (t : IsoscelesTrapezoidWithCircles) :
  trapezoidSides t = {2 * t.R * Real.sqrt 2, 2 * t.R * Real.sqrt 2, 2 * t.R * Real.sqrt 2, 2 * t.R * (2 + Real.sqrt 2)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_with_circles_sides_l784_78461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l784_78480

noncomputable section

/-- Given vectors in R² -/
def m : Fin 2 → ℝ := ![1, 1]
def a : Fin 2 → ℝ := ![1, 0]

/-- The angle between m and n -/
def angle : ℝ := 3 * Real.pi / 4

/-- The dot product of m and n -/
def dot_product_m_n : ℝ := -1

/-- Vector b as a function of x -/
noncomputable def b (x : ℝ) : Fin 2 → ℝ := ![Real.cos x, Real.sin x]

/-- The dot product of n and a is zero -/
def dot_product_n_a_eq_zero (n : Fin 2 → ℝ) : Prop := n 0 * a 0 + n 1 * a 1 = 0

/-- The theorem to be proved -/
theorem vector_problem (n : Fin 2 → ℝ) (h1 : dot_product_m_n = m 0 * n 0 + m 1 * n 1)
  (h2 : Real.cos angle * Real.sqrt ((n 0)^2 + (n 1)^2) * Real.sqrt ((m 0)^2 + (m 1)^2) = dot_product_m_n)
  (h3 : dot_product_n_a_eq_zero n) :
  (n = ![- 1, 0] ∨ n = ![0, -1]) ∧
  ∀ x, 0 ≤ Real.sqrt ((n 0 + b x 0)^2 + (n 1 + b x 1)^2) ∧
       Real.sqrt ((n 0 + b x 0)^2 + (n 1 + b x 1)^2) ≤ 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l784_78480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carpet_area_approx_29_4_l784_78478

/-- Represents a rectangular carpet with given properties -/
structure RectangularCarpet where
  length : ℝ
  width : ℝ
  diagonal : ℝ
  diagonal_plus_length_eq_five_width : diagonal + length = 5 * width
  pythagorean_theorem : diagonal^2 = length^2 + width^2
  length_eq_twelve : length = 12

/-- Calculates the area of a rectangular carpet -/
def carpet_area (c : RectangularCarpet) : ℝ :=
  c.length * c.width

/-- Theorem stating that a carpet with the given properties has an area of approximately 29.4 square meters -/
theorem carpet_area_approx_29_4 (c : RectangularCarpet) :
  ∃ ε > 0, |carpet_area c - 29.4| < ε := by
  sorry

#check carpet_area_approx_29_4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carpet_area_approx_29_4_l784_78478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_cos_100_eq_3_l784_78445

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a * (Real.sin x)^3 + b * (x^(1/3)) * (Real.cos x)^3 + 4

-- State the theorem
theorem f_cos_100_eq_3 (a b : ℝ) :
  f a b (Real.sin (10 * π / 180)) = 5 →
  f a b (Real.cos (100 * π / 180)) = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_cos_100_eq_3_l784_78445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l784_78427

/-- The continuous compound interest formula -/
noncomputable def continuous_compound_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * Real.exp (r * t)

/-- The principal amount given the final amount, interest rate, and time -/
noncomputable def calculate_principal (A : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  A / Real.exp (r * t)

theorem principal_calculation :
  let A : ℝ := 3087
  let r : ℝ := 0.05
  let t : ℝ := 2
  let P : ℝ := calculate_principal A r t
  ∃ ε > 0, abs (P - 2793.57) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l784_78427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_count_l784_78426

/-- The number of zeros immediately following the decimal point in 1 / (15^15 * 3) -/
def zeros_after_decimal (n : ℕ) : Prop :=
  ∃ (x : ℚ), 
    x = 1 / (15^15 * 3) ∧
    ∃ (y : ℚ), y > 0 ∧ y < 1 ∧ x = 10^n * y

theorem zeros_count : zeros_after_decimal 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_count_l784_78426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l784_78484

open Set
open Function
open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the domain of f
def domain : Set ℝ := {x : ℝ | x > 0}

-- State the conditions
axiom f_differentiable : Differentiable ℝ f
axiom f_condition : ∀ x ∈ domain, f x > x * (deriv^[2] f) x

-- Define the inequality
def inequality (x : ℝ) : Prop := x^2 * f (1/x) - f x < 0

-- State the theorem
theorem solution_set : 
  {x ∈ domain | inequality f x} = Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l784_78484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_in_parallelepiped_l784_78499

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular parallelepiped -/
structure Parallelepiped where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D

/-- Calculate the distance between two points -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Calculate the area of a triangle given three points -/
noncomputable def triangleArea (p q r : Point3D) : ℝ :=
  sorry

/-- Theorem: Minimum area of triangle PA₁C in a rectangular parallelepiped -/
theorem min_area_triangle_in_parallelepiped (para : Parallelepiped) 
  (h1 : distance para.A para.B = 1)
  (h2 : distance para.A para.D = 2)
  (h3 : distance para.A para.A₁ = 1) :
  ∃ (P : Point3D), P.y = para.A.y ∧ P.z = para.A.z + (P.x - para.A.x) ∧
    ∀ (Q : Point3D), Q.y = para.A.y ∧ Q.z = para.A.z + (Q.x - para.A.x) →
      triangleArea P para.A₁ para.C ≤ triangleArea Q para.A₁ para.C ∧
      triangleArea P para.A₁ para.C = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_in_parallelepiped_l784_78499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_boat_downstream_time_l784_78496

/-- The time taken for a power boat to travel downstream from dock A to dock B -/
noncomputable def time_downstream (p r : ℝ) : ℝ :=
  12 * (p + r) / (p + 2 * r + 2)

/-- Theorem stating the time taken for the power boat to travel downstream -/
theorem power_boat_downstream_time (p r : ℝ) (hp : p > 0) (hr : r > 0) :
  let t := time_downstream p r
  let downstream_speed := p + r + 2
  let upstream_speed := p - r
  t * downstream_speed + (12 - t) * upstream_speed = 12 * r :=
by
  -- Unfold the definitions
  unfold time_downstream
  -- Introduce the local definitions
  let t := 12 * (p + r) / (p + 2 * r + 2)
  let downstream_speed := p + r + 2
  let upstream_speed := p - r
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_boat_downstream_time_l784_78496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_simplification_l784_78462

/-- Given polynomials p, q, and r, prove their sum is equal to the simplified form -4x^2 + 12x - 12 -/
theorem polynomial_sum_simplification (x : ℝ) :
  let p := fun (x : ℝ) => -4 * x^2 + 2 * x - 5
  let q := fun (x : ℝ) => -6 * x^2 + 4 * x - 9
  let r := fun (x : ℝ) => 6 * x^2 + 6 * x + 2
  p x + q x + r x = -4 * x^2 + 12 * x - 12 := by
  sorry

#check polynomial_sum_simplification

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_simplification_l784_78462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_share_is_2400_l784_78492

/-- Represents the investment and profit sharing scenario -/
structure InvestmentScenario where
  /-- Amount invested by B -/
  b_investment : ℚ
  /-- Total profit -/
  total_profit : ℚ
  /-- Assumption that total profit is positive -/
  profit_positive : total_profit > 0

/-- Calculates B's share of the profit -/
def b_share (scenario : InvestmentScenario) : ℚ :=
  (6 * scenario.b_investment) / ((135 / 2) * scenario.b_investment) * scenario.total_profit

/-- Theorem stating that B's share of the profit is 2400 -/
theorem b_share_is_2400 (scenario : InvestmentScenario) 
  (h : scenario.total_profit = 27000) : 
  b_share scenario = 2400 := by
  sorry

#check b_share_is_2400

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_share_is_2400_l784_78492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_route_distance_l784_78452

/-- Represents a route with a distance and average speed -/
structure Route where
  distance : ℝ
  speed : ℝ

/-- Calculates the time taken for a route -/
noncomputable def time (r : Route) : ℝ := r.distance / r.speed

theorem second_route_distance 
  (route1 : Route) 
  (route2 : Route) 
  (h1 : route1.distance = 1500)
  (h2 : route1.speed = 75)
  (h3 : route2.speed = 25)
  (h4 : time route1 ≤ time route2)
  (h5 : time route1 = 20) :
  route2.distance = 500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_route_distance_l784_78452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_sum_lt_e_l784_78421

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x / (a * x^2 + 1)

theorem extreme_points_sum_lt_e (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > 0)
  (hf : ∀ x : ℝ, deriv (f a) x = 0 ↔ x = x₁ ∨ x = x₂) :
  f a x₁ + f a x₂ < Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_sum_lt_e_l784_78421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_numbers_mean_l784_78469

def numbers : List ℕ := [1212, 1702, 1834, 1956, 2048, 2219, 2300]

theorem remaining_numbers_mean (subset : List ℕ) (h1 : subset.length = 5) 
  (h2 : subset.all (· ∈ numbers)) 
  (h3 : (subset.sum : ℚ) / 5 = 2000) : 
  ((numbers.filter (λ x => x ∉ subset)).sum : ℚ) / 2 = 1635.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_numbers_mean_l784_78469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l784_78495

noncomputable def floor (x : ℝ) := ⌊x⌋
noncomputable def frac (x : ℝ) := x - floor x

theorem inequality_solution (x : ℝ) : 
  (frac x * (floor x - 1) < x - 2) → x ≥ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l784_78495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_X_in_grid_l784_78440

/-- Represents a 5x5 grid where X's can be placed -/
def Grid := Fin 5 → Fin 5 → Bool

/-- Checks if a given position has an X -/
def hasX (g : Grid) (i j : Fin 5) : Prop := g i j = true

/-- Checks if a grid is valid (no more than two X's in a row) -/
def isValidGrid (g : Grid) : Prop :=
  ∀ i j : Fin 5,
    -- Horizontal check
    (hasX g i j ∧ hasX g i (j+1) → ¬hasX g i (j+2)) ∧
    -- Vertical check
    (hasX g i j ∧ hasX g (i+1) j → ¬hasX g (i+2) j) ∧
    -- Diagonal check (top-left to bottom-right)
    (hasX g i j ∧ hasX g (i+1) (j+1) → ¬hasX g (i+2) (j+2)) ∧
    -- Diagonal check (top-right to bottom-left)
    (hasX g i j ∧ hasX g (i+1) (j-1) → ¬hasX g (i+2) (j-2))

/-- Counts the number of X's in a grid -/
def countX (g : Grid) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 5)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 5)) fun j =>
      if g i j then 1 else 0)

/-- The main theorem to prove -/
theorem max_X_in_grid :
  ∃ (g : Grid), isValidGrid g ∧ countX g = 14 ∧
  ∀ (h : Grid), isValidGrid h → countX h ≤ 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_X_in_grid_l784_78440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_l784_78437

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  2 * (Real.sin t.A)^2 = 3 * Real.cos t.A ∧
  t.a^2 - t.c^2 = t.b^2 - t.b * t.c ∧
  t.a = Real.sqrt 3

-- Define the area of the triangle
noncomputable def area (t : Triangle) : ℝ :=
  (1/2) * t.b * t.c * Real.sin t.A

-- Theorem statement
theorem max_area_triangle (t : Triangle) (h : satisfies_conditions t) :
  area t ≤ (3 * Real.sqrt 3) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_l784_78437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_and_perpendicular_existence_l784_78420

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A vector in a plane -/
structure Vec where
  dx : ℝ
  dy : ℝ

/-- A circle in a plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- A line in a plane -/
structure Line where
  point : Point
  direction : Vec

/-- Represents a plane with a circle, a line, and a point -/
structure Plane where
  C : Circle
  l : Line
  M : Point

/-- Check if two vectors are parallel -/
def Vec.isParallel (v1 v2 : Vec) : Prop :=
  v1.dx * v2.dy = v1.dy * v2.dx

/-- Check if two vectors are perpendicular -/
def Vec.isPerpendicular (v1 v2 : Vec) : Prop :=
  v1.dx * v2.dx + v1.dy * v2.dy = 0

theorem parallel_and_perpendicular_existence (π : Plane) : 
  ∃ (l_parallel l_perpendicular : Line),
    (l_parallel.point = π.M) ∧ 
    (l_perpendicular.point = π.M) ∧
    (Vec.isParallel l_parallel.direction π.l.direction) ∧
    (Vec.isPerpendicular l_perpendicular.direction π.l.direction) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_and_perpendicular_existence_l784_78420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pens_bought_l784_78477

/-- The cost of one pen in rubles -/
def pen_cost : ℕ := sorry

/-- The number of pens Masha bought -/
def masha_pens : ℕ := sorry

/-- The number of pens Olya bought -/
def olya_pens : ℕ := sorry

/-- Theorem stating the total number of pens bought by Masha and Olya -/
theorem total_pens_bought :
  pen_cost > 10 ∧
  pen_cost * masha_pens = 357 ∧
  pen_cost * olya_pens = 441 →
  masha_pens + olya_pens = 38 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pens_bought_l784_78477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_problem_l784_78401

theorem log_base_problem (b : ℝ) (h : b > 0) :
  (Real.logb b 256 = -(4 / 3)) → b = 1 / 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_problem_l784_78401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_divisors_count_l784_78449

/-- The star operation -/
def star (a b : ℤ) : ℚ := (a ^ 2 : ℚ) / b

/-- The number of positive integer divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem star_divisors_count :
  num_divisors 144 = 15 ∧ 
  ∀ x : ℤ, (x > 0 ∧ (star 12 x).den = 1) ↔ x ∣ 144 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_divisors_count_l784_78449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gremlin_imp_handshakes_eq_835_l784_78493

/-- The number of handshakes at the gremlin and imp convention -/
def gremlin_imp_handshakes : ℕ := by
  -- Define the number of gremlins and imps
  let num_gremlins : ℕ := 30
  let num_imps : ℕ := 20
  
  -- Define the number of gremlins who shake hands with imps
  let num_gremlins_shaking_imps : ℕ := 20

  -- Calculate handshakes among gremlins
  let gremlin_handshakes : ℕ := num_gremlins.choose 2

  -- Calculate handshakes between imps and gremlins
  let imp_gremlin_handshakes : ℕ := num_imps * num_gremlins_shaking_imps

  -- Total handshakes
  let total_handshakes : ℕ := gremlin_handshakes + imp_gremlin_handshakes

  -- Return the total number of handshakes
  exact total_handshakes

theorem gremlin_imp_handshakes_eq_835 : gremlin_imp_handshakes = 835 := by
  -- Unfold the definition and simplify
  unfold gremlin_imp_handshakes
  simp
  -- The proof is complete
  rfl

#eval gremlin_imp_handshakes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gremlin_imp_handshakes_eq_835_l784_78493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coordinate_equivalence_l784_78471

/-- 
Theorem: The point (-3, π/6) in polar coordinates is equivalent to the point (3, 7π/6) in standard polar coordinate representation.
-/
theorem polar_coordinate_equivalence :
  ∀ (r₁ r₂ : ℝ) (θ₁ θ₂ : ℝ),
    r₁ = -3 →
    θ₁ = π / 6 →
    r₂ = 3 →
    θ₂ = 7 * π / 6 →
    r₂ > 0 →
    0 ≤ θ₂ ∧ θ₂ < 2 * π →
    (r₁ * Real.cos θ₁ = r₂ * Real.cos θ₂ ∧ r₁ * Real.sin θ₁ = r₂ * Real.sin θ₂) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coordinate_equivalence_l784_78471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_count_l784_78467

noncomputable def initial_height : ℝ := 20
noncomputable def bounce_ratio : ℝ := 2/3
noncomputable def target_height : ℝ := 2

noncomputable def height_after_bounces (k : ℕ) : ℝ := initial_height * (bounce_ratio ^ k)

theorem ball_bounce_count :
  ∃ k : ℕ, k > 0 ∧ height_after_bounces k < target_height ∧
  ∀ j : ℕ, 0 < j → j < k → height_after_bounces j ≥ target_height :=
by sorry

#check ball_bounce_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_count_l784_78467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_divisible_by_nine_l784_78491

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def count_divisible_by_nine : ℕ → ℕ
  | 0 => 0
  | n + 1 => count_divisible_by_nine n + 
    (if (1000 ≤ n + 1 ∧ n + 1 ≤ 9999) && (n + 1) % 9 = 0 then 1 else 0)

theorem four_digit_divisible_by_nine :
  count_divisible_by_nine 9999 = 1000 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_divisible_by_nine_l784_78491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_product_inequality_digit_product_equality_condition_l784_78490

def digit_product (a : Fin 9 → ℕ) : ℕ :=
  Finset.prod (Finset.univ : Finset (Fin 9)) (λ i => (i.val + 2) ^ a i)

theorem digit_product_inequality (n : ℕ) (a : Fin 9 → ℕ) 
  (h : ∀ i : Fin 9, a i = (n.digits 10).count (i.val + 1)) :
  digit_product a ≤ n + 1 := by
  sorry

theorem digit_product_equality_condition (n : ℕ) (a : Fin 9 → ℕ) 
  (h : ∀ i : Fin 9, a i = (n.digits 10).count (i.val + 1)) :
  digit_product a = n + 1 ↔ 
    ∃ (r k : ℕ), 1 ≤ r ∧ r ≤ 9 ∧ n = r * 10^k + 10^k - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_product_inequality_digit_product_equality_condition_l784_78490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l784_78464

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  h1 : a > 0
  h2 : b > 0
  h3 : a > b

/-- The foci of an ellipse -/
def Foci (e : Ellipse a b) : ℝ × ℝ × ℝ × ℝ := sorry

/-- The top vertex of an ellipse -/
def TopVertex (e : Ellipse a b) : ℝ × ℝ := sorry

/-- Midpoint of a line segment -/
def Midpoint (P Q : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Perpendicularity of two line segments -/
def Perpendicular (P Q R S : ℝ × ℝ) : Prop := sorry

/-- Eccentricity of an ellipse -/
def Eccentricity (e : Ellipse a b) : ℝ := sorry

theorem ellipse_eccentricity (a b : ℝ) (e : Ellipse a b) 
  (F₁ F₂ A M : ℝ × ℝ) :
  Foci e = (F₁.1, F₁.2, F₂.1, F₂.2) →
  TopVertex e = A →
  Midpoint A F₂ = M →
  Perpendicular M F₁ A F₂ →
  Eccentricity e = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l784_78464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l784_78434

theorem problem_solution (x : ℝ) : (2:ℝ)^x + (2:ℝ)^x + (2:ℝ)^x + (2:ℝ)^x = 256 → x^2 - 1 = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l784_78434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_a_fill_time_l784_78417

/-- The time (in minutes) it takes for pipe a to fill the tank -/
noncomputable def Ta : ℝ := 60 / 17

/-- The time (in minutes) it takes for pipe b to fill the tank -/
def Tb : ℝ := 30

/-- The time (in minutes) it takes for pipe c to empty the tank -/
def Tc : ℝ := 15

/-- The time (in minutes) each pipe is open -/
def openTime : ℝ := 4

/-- The total time (in minutes) it takes to fill the tank when all pipes are used -/
def totalTime : ℝ := 12

theorem pipe_a_fill_time :
  (openTime / Ta + openTime / Tb - openTime / Tc = 1) ∧
  (totalTime = 3 * openTime) → Ta = 60 / 17 := by
  sorry

#check pipe_a_fill_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_a_fill_time_l784_78417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_B_union_A_C_subset_A_iff_l784_78415

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 ≤ (2 : ℝ)^x ∧ (2 : ℝ)^x ≤ 8}
def B : Set ℝ := {x | x > 2}
def C (a : ℝ) : Set ℝ := {x | 1 < x ∧ x < a}

-- Theorem for part (1)
theorem complement_B_union_A : (Set.univ \ B) ∪ A = {x : ℝ | x ≤ 3} := by sorry

-- Theorem for part (2)
theorem C_subset_A_iff (a : ℝ) : C a ⊆ A ↔ a ∈ Set.Iic 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_B_union_A_C_subset_A_iff_l784_78415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_internal_touching_circle_radius_l784_78470

/-- Predicate stating that three circles with given radii touch each other externally -/
def ExternallyTouchingCircles (r₁ r₂ r₃ : ℝ) : Prop :=
  (r₁ + r₂)^2 + (r₂ + r₃)^2 = (r₁ + r₃)^2

/-- Predicate stating that a circle with radius r touches three other circles
    with radii r₁, r₂, r₃ internally -/
def InternallyTouchingCircle (r r₁ r₂ r₃ : ℝ) : Prop :=
  (r - r₁)^2 + (r - r₂)^2 = (r₁ + r₂)^2 ∧
  (r - r₂)^2 + (r - r₃)^2 = (r₂ + r₃)^2 ∧
  (r - r₁)^2 + (r - r₃)^2 = (r₁ + r₃)^2

/-- Given three circles with radii 1, 2, and 3 units that touch each other externally,
    the radius of the circle that touches all three circles internally is 6 units. -/
theorem internal_touching_circle_radius (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 1) (h₂ : r₂ = 2) (h₃ : r₃ = 3)
  (h_external_touch : ExternallyTouchingCircles r₁ r₂ r₃) :
  ∃ r : ℝ, r = 6 ∧ InternallyTouchingCircle r r₁ r₂ r₃ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_internal_touching_circle_radius_l784_78470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_32_seconds_l784_78431

/-- Calculates the time (in seconds) it takes for a train to cross a bridge -/
noncomputable def trainCrossingTime (trainLength : ℝ) (bridgeLength : ℝ) (trainSpeed : ℝ) : ℝ :=
  (trainLength + bridgeLength) / (trainSpeed * 1000 / 3600)

/-- Theorem stating that a train with given specifications takes 32 seconds to cross the bridge -/
theorem train_crossing_time_32_seconds :
  trainCrossingTime 120 200 36 = 32 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_32_seconds_l784_78431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_of_power_of_three_l784_78412

/-- The number of positive integer divisors of 3^2010 -/
def num_divisors : ℕ := 2011

/-- 3^2010 -/
def large_power : ℕ := 3^2010

theorem count_divisors_of_power_of_three :
  (Finset.filter (λ x : ℕ => large_power % x = 0) (Finset.range (large_power + 1))).card = num_divisors := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_of_power_of_three_l784_78412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_sum_l784_78404

theorem complex_power_sum (z : ℂ) (h : z + z⁻¹ = Real.sqrt 2) : z^12 + z⁻¹^12 = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_sum_l784_78404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_in_group_2_l784_78457

structure ExerciseGroup where
  id : Nat
  lower_bound : Real
  upper_bound : Real
  frequency : Nat

def dataset : List ExerciseGroup := [
  ⟨1, 0, 0.5, 12⟩,
  ⟨2, 0.5, 1, 24⟩,
  ⟨3, 1, 1.5, 18⟩,
  ⟨4, 1.5, 2, 10⟩,
  ⟨5, 2, 2.5, 6⟩
]

def total_frequency (data : List ExerciseGroup) : Nat :=
  data.foldl (fun acc g => acc + g.frequency) 0

def cumulative_frequency (data : List ExerciseGroup) (n : Nat) : Nat :=
  (data.take n).foldl (fun acc g => acc + g.frequency) 0

def median_group (data : List ExerciseGroup) : Nat :=
  let total := total_frequency data
  let median_position := (total + 1) / 2
  data.find? (fun g => cumulative_frequency data g.id ≥ median_position)
    |>.map (·.id)
    |>.getD 0

theorem median_in_group_2 (data : List ExerciseGroup := dataset) :
  median_group data = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_in_group_2_l784_78457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l784_78425

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 0}
def B : Set ℝ := {x | x^2 > 1}

theorem union_of_A_and_B : A ∪ B = Set.Iic 0 ∪ Set.Ioi 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l784_78425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_tangent_and_expression_l784_78442

noncomputable def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 3

def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

def point_on_circle (m : ℝ) : Prop := circle_eq m (Real.sqrt 2) ∧ first_quadrant m (Real.sqrt 2)

noncomputable def angle_α (m : ℝ) : ℝ := Real.arctan ((Real.sqrt 2) / m)

theorem circle_intersection_tangent_and_expression (m : ℝ) 
  (h : point_on_circle m) : 
  Real.tan (angle_α m) = Real.sqrt 2 ∧ 
  (2 * (Real.cos (angle_α m / 2))^2 - Real.sin (angle_α m) - 1) / 
  (Real.sqrt 2 * Real.sin (π / 4 + angle_α m)) = 2 * Real.sqrt 2 - 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_tangent_and_expression_l784_78442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_number_is_three_l784_78403

def mySequence : List ℕ := [2, 3, 6, 15, 33, 123]

theorem second_number_is_three : mySequence.get! 1 = 3 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_number_is_three_l784_78403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_product_result_l784_78453

/-- The cross product of two 3D vectors -/
def cross (a b : Fin 3 → ℝ) : Fin 3 → ℝ :=
  λ i => match i with
  | 0 => a 1 * b 2 - a 2 * b 1
  | 1 => a 2 * b 0 - a 0 * b 2
  | 2 => a 0 * b 1 - a 1 * b 0

theorem cross_product_result (a b c : Fin 3 → ℝ)
  (h1 : cross a b = λ i => [3, -2, 5].get i)
  (h2 : c = λ i => [1, 0, -1].get i) :
  cross a (λ i => 5 * (b i) + 2 * (c i)) = λ i => [19, -8, 19].get i := by
  sorry

#check cross_product_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_product_result_l784_78453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_storm_damage_conversion_l784_78488

/-- Converts Euros to British pounds given the exchange rate -/
noncomputable def eurosToPounds (euros : ℝ) (exchangeRate : ℝ) : ℝ :=
  euros / exchangeRate

/-- Theorem: Given a storm damage of €45 million and an exchange rate of 1.2 Euros to 1 British pound,
    the equivalent damage in British pounds is 37,500,000. -/
theorem storm_damage_conversion :
  let damage_euros : ℝ := 45000000
  let exchange_rate : ℝ := 1.2
  eurosToPounds damage_euros exchange_rate = 37500000 := by
  -- Unfold the definition of eurosToPounds
  unfold eurosToPounds
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_storm_damage_conversion_l784_78488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l784_78428

/-- Heron's formula for the area of a triangle -/
noncomputable def herons_formula (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- The area of a triangle with sides 30, 21, and 10 is approximately 54.52 -/
theorem triangle_area_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |herons_formula 30 21 10 - 54.52| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l784_78428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pf_length_l784_78473

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  right_angle : (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0
  pq_length : Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) = 4
  pr_length : Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2) = 4 * Real.sqrt 3

-- Define the point M on PQ
noncomputable def M (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((2 * P.1 + Q.1) / 3, (2 * P.2 + Q.2) / 3)

-- Define the altitude PL
def altitude (P Q R : ℝ × ℝ) (L : ℝ × ℝ) : Prop :=
  (L.1 - P.1) * (Q.1 - R.1) + (L.2 - P.2) * (Q.2 - R.2) = 0

-- Define the median RM
def median (P Q R : ℝ × ℝ) (F : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, F = (R.1 + t * (M P Q).1 - t * R.1, R.2 + t * (M P Q).2 - t * R.2)

-- Theorem statement
theorem pf_length (P Q R : ℝ × ℝ) (h : Triangle P Q R) :
  ∃ F : ℝ × ℝ, altitude P Q R F ∧ median P Q R F ∧
  Real.sqrt ((F.1 - P.1)^2 + (F.2 - P.2)^2) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pf_length_l784_78473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emma_coins_l784_78463

theorem emma_coins (y : ℚ) (hy : y > 0) : 
  let initial_coins := y
  let lost_coins := (1 / 3 : ℚ) * initial_coins
  let found_coins := (3 / 4 : ℚ) * lost_coins
  let remaining_coins := initial_coins - lost_coins + found_coins
  (initial_coins - remaining_coins) / initial_coins = (1 / 12 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_emma_coins_l784_78463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_count_is_correct_l784_78423

/-- Represents the number of paths between two points -/
structure PathCount where
  count : Nat

/-- The number of paths from A to B -/
def paths_A_to_B : PathCount := ⟨2⟩

/-- The number of paths from B to D -/
def paths_B_to_D : PathCount := ⟨2⟩

/-- The number of direct paths from A to D -/
def direct_paths_A_to_D : PathCount := ⟨1⟩

/-- The number of paths from D to C -/
def paths_D_to_C : PathCount := ⟨2⟩

/-- Multiplication for PathCount -/
instance : HMul PathCount PathCount PathCount where
  hMul a b := ⟨a.count * b.count⟩

/-- Addition for PathCount -/
instance : Add PathCount where
  add a b := ⟨a.count + b.count⟩

/-- The total number of paths from A to C -/
def total_paths_A_to_C : PathCount := 
  paths_A_to_B * paths_B_to_D * paths_D_to_C + direct_paths_A_to_D * paths_D_to_C

theorem path_count_is_correct : total_paths_A_to_C.count = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_count_is_correct_l784_78423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_denomination_l784_78443

/-- The denomination of the unknown bills -/
def x : ℕ := sorry

/-- The total value of all bills -/
def total_value : ℕ := 10 * x + 8 * 10 + 4 * 5

/-- The number of $100 bills after exchange -/
def num_hundred_bills : ℕ := 3

theorem bill_denomination :
  (total_value = num_hundred_bills * 100) → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_denomination_l784_78443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersection_l784_78409

/-- Circle in polar coordinates -/
structure PolarCircle where
  equation : ℝ → ℝ

/-- Line in 2D space -/
structure Line where
  point : ℝ × ℝ
  angle : ℝ

/-- Cartesian coordinates -/
def CartesianCoords := ℝ × ℝ

/-- Convert polar to Cartesian coordinates -/
noncomputable def polar_to_cartesian (ρ θ : ℝ) : CartesianCoords :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : CartesianCoords) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Check if a point is on a line -/
def on_line (p : CartesianCoords) (l : Line) : Prop :=
  let (x, y) := p
  let (x0, y0) := l.point
  y - y0 = Real.tan l.angle * (x - x0)

/-- Check if a point is on a circle -/
def on_circle (p : CartesianCoords) (C : PolarCircle) : Prop :=
  let (x, y) := p
  x^2 + (y - 2)^2 = 4

/-- Theorem statement -/
theorem circle_and_line_intersection
  (C : PolarCircle)
  (l : Line)
  (h_C : C.equation = fun θ ↦ 4 * Real.sin θ)
  (h_l : l.point = (1, 1) ∧ l.angle = Real.pi / 4) :
  (∃ (x y : ℝ), x^2 + (y - 2)^2 = 4) ∧
  (∃ (A B : CartesianCoords), 
    on_line A l ∧ on_line B l ∧ 
    on_circle A C ∧ on_circle B C ∧
    distance (1, 1) A * distance (1, 1) B = 2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersection_l784_78409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_dot_product_l784_78460

/-- Theorem: For a hyperbola with equation x²/2 - y²/b² = 1 (b > 0) and asymptote y = x,
    if P(√3, y₀) is on the hyperbola, then the dot product of vectors PF₁ and PF₂ is zero,
    where F₁ and F₂ are the foci of the hyperbola. -/
theorem hyperbola_dot_product (b : ℝ) (y₀ : ℝ) (h₁ : b > 0) :
  let f (x y : ℝ) := x^2 / 2 - y^2 / b^2
  let asym (x : ℝ) := x
  let P := (Real.sqrt 3, y₀)
  let F₁ := (-2, 0)
  let F₂ := (2, 0)
  f (Real.sqrt 3) y₀ = 1 ∧ 
  (∀ x, asym x = x) →
  (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_dot_product_l784_78460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_composition_l784_78422

noncomputable def G (x : ℝ) : ℝ := (x + 1)^2 / 2 - 4

theorem G_composition : G (G (G 0)) = -3.9921875 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_composition_l784_78422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_C₁_and_C₂_l784_78474

-- Define the curves C₁ and C₂
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := (4 + 5 * Real.cos t, 5 + 5 * Real.sin t)

noncomputable def C₂ (θ : ℝ) : ℝ := 2 * Real.sin θ

-- Define the intersection points in polar coordinates
noncomputable def intersection_points : Set (ℝ × ℝ) := {(Real.sqrt 2, Real.pi / 4), (2, Real.pi / 2)}

-- Theorem statement
theorem intersection_of_C₁_and_C₂ :
  ∀ (ρ θ : ℝ), 0 ≤ ρ ∧ 0 ≤ θ ∧ θ < 2 * Real.pi →
  (∃ (t : ℝ), C₁ t = (ρ * Real.cos θ, ρ * Real.sin θ)) ∧
  (ρ = C₂ θ) →
  (ρ, θ) ∈ intersection_points := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_C₁_and_C₂_l784_78474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l784_78494

theorem problem_statement : 
  (¬(∀ x : ℝ, x > 0 → x + 1/2 > 2)) ∧ (∃ x₀ : ℝ, (2 : ℝ)^x₀ < 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l784_78494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_example_l784_78447

/-- The volume of a square-based pyramid --/
noncomputable def pyramidVolume (s h : ℝ) : ℝ := (1/3) * s^2 * h

/-- Theorem: The volume of a square-based pyramid with base side length 8 and height 6 is 128 --/
theorem pyramid_volume_example : pyramidVolume 8 6 = 128 := by
  -- Unfold the definition of pyramidVolume
  unfold pyramidVolume
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_example_l784_78447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_marking_size_l784_78450

/-- A marking of cells on an n x n grid. -/
def Marking (n : ℕ) := Fin n → Fin n → Bool

/-- Whether a rectangle on the grid contains a marked cell. -/
def containsMarkedCell (n : ℕ) (m : Marking n) (x y w h : Fin n) : Prop :=
  ∃ i j : Fin n, (i.val < w.val ∧ j.val < h.val) ∧ m (x + i) (y + j) = true

/-- The property that every rectangle with area at least n contains a marked cell. -/
def validMarking (n : ℕ) (m : Marking n) : Prop :=
  ∀ x y w h : Fin n, (w.val * h.val : ℕ) ≥ n → containsMarkedCell n m x y w h

/-- The number of marked cells in a marking. -/
def numMarkedCells (n : ℕ) (m : Marking n) : ℕ :=
  (Finset.sum (Finset.univ : Finset (Fin n)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin n)) fun j =>
      if m i j then 1 else 0)

theorem max_valid_marking_size :
  (∃ m : Marking 7, validMarking 7 m ∧ numMarkedCells 7 m = 7) ∧
  (∀ n > 7, ¬∃ m : Marking n, validMarking n m ∧ numMarkedCells n m = n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_marking_size_l784_78450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_implication_l784_78459

-- Define the point type
variable {Point : Type*}

-- Define a parallelogram structure
structure Parallelogram (Point : Type*) :=
  (A B C D : Point)

-- Define the property of being a parallelogram
def is_parallelogram (ABCD : Parallelogram Point) : Prop :=
  ∃ (v₁ v₂ : Point → Point), 
    v₁ ABCD.A = ABCD.B ∧
    v₁ ABCD.D = ABCD.C ∧
    v₂ ABCD.A = ABCD.D ∧
    v₂ ABCD.B = ABCD.C

-- State the theorem
theorem parallelogram_implication 
  (A B C E H K M P T X : Point)
  (ACPH : Parallelogram Point)
  (AMBE : Parallelogram Point)
  (AHBT : Parallelogram Point)
  (BKXM : Parallelogram Point)
  (CKXP : Parallelogram Point)
  (h1 : is_parallelogram ACPH)
  (h2 : is_parallelogram AMBE)
  (h3 : is_parallelogram AHBT)
  (h4 : is_parallelogram BKXM)
  (h5 : is_parallelogram CKXP) :
  is_parallelogram (Parallelogram.mk A B T E) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_implication_l784_78459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_MNP_l784_78446

-- Define the ellipse C₁
def C₁ (x y b : ℝ) : Prop := x^2/4 + y^2/b^2 = 1

-- Define the parabola C₂
def C₂ (x y : ℝ) : Prop := x^2 = 4*y

-- Define the eccentricity of C₁
noncomputable def eccentricity (b : ℝ) : ℝ := Real.sqrt 3 / 2

-- Define the condition for M and N
def MN_condition (y₁ y₂ : ℝ) : Prop := y₁ ≠ y₂ ∧ y₁ + y₂ = 4

-- Define the area of triangle MNP
noncomputable def area_MNP (x₀ : ℝ) : ℝ := 
  (1/2) * Real.sqrt ((x₀^2 + 4) * (x₀^2 + 4) * (16 - 2*x₀^2))

-- Main theorem
theorem max_area_MNP :
  ∀ b : ℝ, 0 < b → b < 2 →
  ∀ x₁ y₁ x₂ y₂ : ℝ,
  C₂ x₁ y₁ → C₂ x₂ y₂ →
  MN_condition y₁ y₂ →
  (∀ x₀ : ℝ, area_MNP x₀ ≤ 8) ∧
  (∃ x₀ : ℝ, area_MNP x₀ = 8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_MNP_l784_78446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_l784_78455

noncomputable def f (x : ℝ) (α : ℝ) : ℝ :=
  if x > 0 then x^2 + Real.sin (x + Real.pi/3)
  else -x^2 + Real.cos (x + α)

theorem alpha_value (α : ℝ) :
  (α ∈ Set.Icc 0 (2*Real.pi)) →  -- α is in [0, 2π)
  (∀ x, f x α = -f (-x) α) →  -- f is an odd function
  α = 7*Real.pi/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_l784_78455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_cos_aob_zero_l784_78439

/-- A rectangle with sides AB and AD, and diagonals intersecting at O -/
structure Rectangle where
  AB : ℝ
  AD : ℝ
  O : Point

/-- Angle AOB in a rectangle -/
def angle_AOB (rect : Rectangle) : ℝ := sorry

/-- The theorem stating that in a rectangle with AB = 15 and AD = 20, cos ∠AOB = 0 -/
theorem rectangle_cos_aob_zero (rect : Rectangle) 
  (h1 : rect.AB = 15) 
  (h2 : rect.AD = 20) : 
  Real.cos (angle_AOB rect) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_cos_aob_zero_l784_78439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AD_l784_78481

-- Define the points A, B, C, and D
variable (A B C D : ℝ × ℝ)

-- Define the conditions
axiom east_of : B.1 > A.1 ∧ B.2 = A.2
axiom north_of_B : C.1 = B.1 ∧ C.2 > B.2
axiom distance_AC : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 15
axiom angle_BAC : Real.arccos ((B.1 - A.1) / Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)) = Real.pi / 6
axiom north_of_C : D.1 = C.1 ∧ D.2 = C.2 + 10

-- Theorem statement
theorem distance_AD : 
  Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = Real.sqrt (562.5 + 300 * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AD_l784_78481
