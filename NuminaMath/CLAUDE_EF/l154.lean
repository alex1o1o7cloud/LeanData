import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_group_size_l154_15407

/-- Represents the total man-hours required to complete the work -/
def totalManHours : ℕ := sorry

/-- The number of men in the first group -/
def firstGroupMen : ℕ := sorry

/-- The number of hours worked per day (constant for both groups) -/
def hoursPerDay : ℕ := 9

/-- The number of days the first group takes to complete the work -/
def daysForFirstGroup : ℕ := 24

/-- The number of men in the second group -/
def secondGroupMen : ℕ := 12

/-- The number of days the second group takes to complete the work -/
def daysForSecondGroup : ℕ := 16

/-- Theorem stating that the number of men in the first group is 8 -/
theorem first_group_size :
  (firstGroupMen * hoursPerDay * daysForFirstGroup = secondGroupMen * hoursPerDay * daysForSecondGroup) →
  firstGroupMen = 8 := by
  sorry

#check first_group_size

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_group_size_l154_15407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_class_size_l154_15423

theorem smallest_class_size (n : ℕ) (scores : Fin n → ℕ) : 
  (∀ i, scores i ≥ 50) →  -- Every student scored at least 50 points
  (∀ i, scores i ≤ 80) →  -- Maximum score is 80 points
  (∃ a b c d, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    scores a = 80 ∧ scores b = 80 ∧ scores c = 80 ∧ scores d = 80) →  -- Four distinct students scored 80 points
  (Finset.sum Finset.univ (λ i => scores i) = 65 * n) →  -- Mean score is 65
  n ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_class_size_l154_15423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_painting_possibilities_l154_15478

theorem rectangle_painting_possibilities :
  let count := Finset.card (Finset.filter
    (fun p : ℕ × ℕ => 
      let (a, b) := p
      b > a ∧ 
      (a - 4) * (b - 4) = 2 * a * b / 3 ∧
      a > 0 ∧ b > 0)
    (Finset.range 100 ×ˢ Finset.range 100))
  count = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_painting_possibilities_l154_15478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l154_15475

-- Define the function f
noncomputable def f (t : ℝ) (x : ℝ) : ℝ := x^2 - t*x - 1

-- Define the condition for symmetry
def symmetry_condition (t : ℝ) : Prop :=
  ∀ x : ℝ, f t (2 - x) = f t (2 + x)

-- Define the non-monotonicity condition
def non_monotonic (t : ℝ) : Prop :=
  ¬(∀ x y : ℝ, -1 < x ∧ x < y ∧ y < 2 → f t x < f t y) ∧
  ¬(∀ x y : ℝ, -1 < x ∧ x < y ∧ y < 2 → f t x > f t y)

-- Define the minimum value function
noncomputable def g (t : ℝ) : ℝ :=
  if t ≤ -2 then t
  else if t < 4 then -t^2/4 - 1
  else 3 - 2*t

-- State the theorem
theorem f_properties (t : ℝ) :
  symmetry_condition t →
  (∃ x : ℝ, f t x = x^2 - 4*x - 1) ∧
  (non_monotonic t ↔ -2 < t ∧ t < 4) ∧
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → f t x ≥ g t) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l154_15475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l154_15444

-- Define the hyperbola
def hyperbola_equation (x y : ℝ) : Prop := 16 * x^2 - 9 * y^2 = 144

-- Define the transverse axis length
noncomputable def transverse_axis_length : ℝ := 6

-- Define the conjugate axis length
noncomputable def conjugate_axis_length : ℝ := 8

-- Define the eccentricity
noncomputable def eccentricity : ℝ := 5/3

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop := y^2 = -12 * x

-- Theorem statement
theorem hyperbola_properties :
  (∀ x y, hyperbola_equation x y →
    transverse_axis_length = 6 ∧
    conjugate_axis_length = 8 ∧
    eccentricity = 5/3) ∧
  (∀ x y, parabola_equation x y → True) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l154_15444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_account_growth_l154_15456

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (frequency : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / frequency) ^ (frequency * time)

theorem savings_account_growth (principal rate frequency time final : ℝ) 
  (h1 : principal = 800)
  (h2 : rate = 0.1)
  (h3 : frequency = 2)
  (h4 : time = 1)
  (h5 : final = 882) :
  compound_interest principal rate frequency time = final := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_account_growth_l154_15456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_ratio_l154_15498

/-- Define an arithmetic sequence of four terms -/
def is_arithmetic_sequence (a b c d : ℝ) : Prop :=
  ∃ r : ℝ, b - a = r ∧ c - b = r ∧ d - c = r

/-- Define a geometric sequence of five terms -/
def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  ∃ r : ℝ, b / a = r ∧ c / b = r ∧ d / c = r ∧ e / d = r

theorem arithmetic_geometric_sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  (is_arithmetic_sequence 1 a₁ a₂ 4) →
  (is_geometric_sequence 1 b₁ b₂ b₃ 4) →
  (a₁ + a₂) / b₂ = 5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_ratio_l154_15498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l154_15443

theorem sin_alpha_value (α : Real) 
  (h1 : Real.tan (α + π/4) = 1/2) 
  (h2 : -π/2 < α) 
  (h3 : α < 0) : 
  Real.sin α = -Real.sqrt 10/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l154_15443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_in_specific_tank_l154_15426

/-- Represents a horizontal cylindrical water tank -/
structure WaterTank where
  length : ℝ
  diameter : ℝ
  surfaceArea : ℝ

/-- Calculates the depth of water in the tank -/
noncomputable def waterDepth (tank : WaterTank) : ℝ :=
  4 - 2 * Real.sqrt 3

/-- Theorem stating the depth of water in the given tank -/
theorem water_depth_in_specific_tank :
  let tank : WaterTank := ⟨15, 8, 60⟩
  waterDepth tank = 4 - 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_in_specific_tank_l154_15426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_product_l154_15459

/-- Given positive real numbers a and b, and a triangle in the first quadrant
    bounded by the coordinate axes and the line ax + by = 4 with area 4,
    prove that ab = 2. -/
theorem triangle_area_product (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1/2 * (4/a) * (4/b) = 4) → a * b = 2 := by
  intro h
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_product_l154_15459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_test_result_l154_15442

-- Define the K^2 statistic
def K_squared : ℝ := 5

-- Define the critical values and their corresponding p-values
def critical_value_95 : ℝ := 3.841
def p_value_95 : ℝ := 0.05

def critical_value_99 : ℝ := 6.635
def p_value_99 : ℝ := 0.01

-- Define a proposition for the relationship between X and Y
def X_and_Y_are_related : Prop := True

-- Define the theorem
theorem independence_test_result :
  K_squared > critical_value_95 ∧ K_squared < critical_value_99 →
  (1 - p_value_95) * 100 = 95 →
  ∃ (confidence : ℝ), confidence = 95 ∧ X_and_Y_are_related := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_test_result_l154_15442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_product_l154_15412

/-- Given two lines with slopes p and q, prove that their product is -7/10 under specific conditions -/
theorem line_slope_product (p q : ℝ) : 
  (∃ θ₁ θ₂ : ℝ, θ₁ = 3 * θ₂ ∧ p = -Real.tan θ₁ ∧ q = Real.tan θ₂) →  -- L₁ makes 3 times the angle of L₂
  (p = -q / 2) →                                          -- L₁'s slope is negative half of L₂'s
  (q > 0) →                                               -- q is positive
  (p ≠ 0 ∧ q ≠ 0) →                                       -- Neither line is vertical
  p * q = -7/10 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_product_l154_15412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_probability_l154_15468

def spinner : List ℕ := [3, 6, 1, 4, 5, 2, 8]

def is_prime (n : ℕ) : Bool :=
  n > 1 && (List.range (n - 1)).all (λ m => m <= 1 || n % (m + 1) ≠ 0)

def is_multiple_of_four (n : ℕ) : Bool := n % 4 = 0

theorem spinner_probability :
  let favorable_outcomes := (spinner.filter (λ n => is_prime n || is_multiple_of_four n)).length
  (favorable_outcomes : ℚ) / spinner.length = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_probability_l154_15468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sequences_count_l154_15411

def word : String := "EQUALS"

def valid_sequence (s : List Char) : Bool :=
  s.length = 5 &&
  s.head? = some 'L' &&
  s.get? 2 = some 'E' &&
  s.getLast? = some 'Q' &&
  s.toFinset.card = 5 &&
  s.all (· ∈ word.toList)

theorem distinct_sequences_count :
  (List.filter valid_sequence (List.permutations word.toList)).length = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sequences_count_l154_15411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l154_15455

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * Real.cos x + Real.sin x

-- State the theorem
theorem derivative_of_f (x : ℝ) :
  deriv f x = 2 * Real.cos x - x * Real.sin x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l154_15455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_function_property_f_equals_g_inequality_property_l154_15418

-- Proposition B
theorem monotonic_function_property {f : ℝ → ℝ} {I : Set ℝ} (hf : Monotone f) 
  (hI : Interval I) {x₁ x₂ : ℝ} (hx₁ : x₁ ∈ I) (hx₂ : x₂ ∈ I) :
  f x₁ = f x₂ → x₁ = x₂ :=
by sorry

-- Proposition C
noncomputable def f (x : ℝ) : ℝ := (x^4 - 1) / (x^2 + 1)
def g (x : ℝ) : ℝ := x^2 - 1

theorem f_equals_g : f = g :=
by sorry

-- Proposition D
theorem inequality_property (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (a * c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_function_property_f_equals_g_inequality_property_l154_15418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l154_15450

/-- Representation of a point in 2D space. -/
structure Point where
  x : ℝ
  y : ℝ

/-- A point is inside the disk. -/
def is_inside_disk (p : Point) : Prop :=
  sorry

/-- Count of segments intersecting at a given point for n points on a circle. -/
def count_intersecting_segments (p : Point) (n : ℕ) : ℕ :=
  sorry

/-- The number of intersection points inside the disk for n points on a circle. -/
def number_of_intersection_points (n : ℕ) : ℕ :=
  sorry

/-- No three segments intersect at a single point inside the disk. -/
axiom no_triple_intersections (n : ℕ) :
  ∀ (p : Point), is_inside_disk p → (count_intersecting_segments p n ≤ 2)

/-- Given n points on a circle, with all possible segments drawn between them,
    and no three segments intersecting at a single point inside the disk,
    the number of intersection points inside the disk is equal to (n choose 4). -/
theorem intersection_points_count (n : ℕ) (h : n ≥ 4) :
  (number_of_intersection_points n) = Nat.choose n 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l154_15450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_W_is_zero_l154_15487

-- Define a convex quadrilateral
structure ConvexQuadrilateral (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
  (A B C D : V)
  (convex : Convex ℝ {A, B, C, D})

-- Define the power of a point with respect to a circle
noncomputable def powerOfPoint {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (P A B C : V) : ℝ := sorry

-- Define the area of a triangle
noncomputable def triangleArea {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (A B C : V) : ℝ := sorry

-- Define W for a vertex
noncomputable def W {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (P A B C : V) : ℝ := 
  powerOfPoint P A B C * triangleArea A B C

-- Theorem statement
theorem sum_of_W_is_zero {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (quad : ConvexQuadrilateral V) : 
  W quad.A quad.B quad.C quad.D + 
  W quad.B quad.C quad.D quad.A + 
  W quad.C quad.D quad.A quad.B + 
  W quad.D quad.A quad.B quad.C = 0 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_W_is_zero_l154_15487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_complex_quadrant_l154_15476

/-- Given an infinite geometric sequence with first term 1 and common ratio 1/2,
    prove that z = 1/(a+i) lies in the fourth quadrant, where a is the sum of the infinite series. -/
theorem geometric_sequence_complex_quadrant :
  let a_n : ℕ → ℝ := λ n => (1/2)^(n-1)
  let S_n : ℕ → ℝ := λ n => (1 - (1/2)^n) / (1 - 1/2)
  let a : ℝ := 2
  let z : ℂ := 1 / (a + Complex.I)
  (∀ n : ℕ, n > 0 → a_n n = (1/2)^(n-1)) →
  (∀ n : ℕ, n > 0 → S_n n = (1 - (1/2)^n) / (1 - 1/2)) →
  (Filter.Tendsto S_n Filter.atTop (nhds a)) →
  (z.re > 0 ∧ z.im < 0) :=
by
  intros a_n S_n a z h1 h2 h3
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_complex_quadrant_l154_15476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_altitude_length_l154_15466

-- Define the parallelogram PQRS
structure Parallelogram :=
  (P Q R S : ℝ × ℝ)
  (T : ℝ × ℝ)  -- Add T point explicitly

-- Define the properties of the parallelogram
def is_parallelogram (PQRS : Parallelogram) : Prop :=
  -- Add conditions for parallelogram here
  True

-- Define the length of a line segment
noncomputable def length (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Define the area of a parallelogram
noncomputable def area (PQRS : Parallelogram) : ℝ :=
  abs ((PQRS.Q.1 - PQRS.P.1) * (PQRS.R.2 - PQRS.P.2) - (PQRS.R.1 - PQRS.P.1) * (PQRS.Q.2 - PQRS.P.2))

-- Define an altitude of a parallelogram
def is_altitude (H A B : ℝ × ℝ) (PQRS : Parallelogram) : Prop :=
  -- Add conditions for altitude here
  True

-- Theorem statement
theorem parallelogram_altitude_length 
  (PQRS : Parallelogram) 
  (h_parallelogram : is_parallelogram PQRS)
  (h_PS_altitude : is_altitude PQRS.P PQRS.Q PQRS.R PQRS)
  (h_QT_altitude : is_altitude PQRS.Q PQRS.R PQRS.S PQRS)
  (h_PQ_length : length PQRS.P PQRS.Q = 15)
  (h_SR_length : length PQRS.S PQRS.R = 3)
  (h_PS_length : length PQRS.P PQRS.S = 5) :
  length PQRS.Q PQRS.T = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_altitude_length_l154_15466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l154_15413

/-- Given constants a and b, with a ≠ 0, and a function f(x) = ax² + bx
    such that f(2) = 0 and f(x) = x has two equal real roots,
    prove that f(x) = -½x² + x and F(x) = f(x) - f(-x) is an odd function. -/
theorem quadratic_function_properties :
  ∃ (a b : ℝ) (f F : ℝ → ℝ),
    a ≠ 0 ∧
    (∀ x, f x = a * x^2 + b * x) ∧
    f 2 = 0 ∧
    (∃! x, f x = x) ∧
    (∀ x, f x = -1/2 * x^2 + x) ∧
    (∀ x, F x = f x - f (-x)) ∧
    (∀ x, F (-x) = -F x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l154_15413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_3000_even_integers_digits_l154_15452

/-- The number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ := 
  if n < 10 then 1
  else if n < 100 then 2
  else if n < 1000 then 3
  else 4

/-- The sum of digits for even numbers from 2 to n -/
def sum_digits_even (n : ℕ) : ℕ :=
  (Finset.range ((n + 1) / 2)).sum (λ i => num_digits ((i + 1) * 2))

theorem first_3000_even_integers_digits :
  sum_digits_even 6000 = 11444 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_3000_even_integers_digits_l154_15452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_inequality_l154_15485

theorem negation_of_cosine_inequality :
  (¬ ∀ x : ℝ, Real.cos x ≥ 1) ↔ (∃ x₀ : ℝ, Real.cos x₀ < 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_inequality_l154_15485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_triangle_square_l154_15482

/-- A square with vertices O, P, Q, R where O is the origin, Q is at (3, 3), and P is on the positive x-axis -/
structure Square where
  O : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  is_origin : O = (0, 0)
  q_coord : Q = (3, 3)
  p_on_x_axis : P.2 = 0
  is_square : (P.1 - O.1)^2 + (P.2 - O.2)^2 = (Q.1 - P.1)^2 + (Q.2 - P.2)^2

/-- The area of a triangle given its vertices -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

/-- The area of a square given its side length -/
def square_area (side : ℝ) : ℝ := side^2

/-- The theorem stating that T(3, 3√2) creates a triangle PQT with area equal to square OPQR -/
theorem equal_area_triangle_square (s : Square) :
  let T : ℝ × ℝ := (3, 3 * Real.sqrt 2)
  triangle_area s.P s.Q T = square_area (s.Q.1 - s.O.1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_triangle_square_l154_15482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_specified_coins_probability_l154_15441

/-- The probability of exactly three specified coins out of five coming up heads -/
theorem three_specified_coins_probability :
  let n : ℕ := 5  -- Total number of coins
  let k : ℕ := 3  -- Number of specified coins
  let total_outcomes : ℕ := 2^n
  let successful_outcomes : ℕ := 2^(n-k)
  let probability : ℚ := successful_outcomes / total_outcomes
  probability = 1/8 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_specified_coins_probability_l154_15441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_circle_line_intersection_l154_15460

/-- Given a line and a circle in a 2D plane, prove that the area of the triangle
formed by the circle's center and the intersection points of the line and circle is 2√5. -/
theorem area_triangle_circle_line_intersection :
  let line : ℝ → ℝ → Prop := λ x y => x - 2 * y - 3 = 0
  let circle : ℝ → ℝ → Prop := λ x y => (x - 2)^2 + (y + 3)^2 = 9
  let C : ℝ × ℝ := (2, -3)
  ∃ (E F : ℝ × ℝ),
    line E.1 E.2 ∧ line F.1 F.2 ∧
    circle E.1 E.2 ∧ circle F.1 F.2 ∧
    E ≠ F ∧
    (Real.sqrt ((E.1 - F.1)^2 + (E.2 - F.2)^2) * Real.sqrt ((C.1 - E.1)^2 + (C.2 - E.2)^2)) / 2 = 2 * Real.sqrt 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_circle_line_intersection_l154_15460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_sum_negative_l154_15430

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - x + m

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f m (x + m)

theorem roots_sum_negative (m : ℝ) (x₁ x₂ : ℝ) 
  (hm : m > 1) 
  (hx₁ : g m x₁ = 0) 
  (hx₂ : g m x₂ = 0) 
  (hx₁₂ : x₁ ≠ x₂) : 
  x₁ + x₂ < 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_sum_negative_l154_15430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l154_15403

theorem cos_double_angle (α : ℝ) (h : Real.sin α = 3/5) : Real.cos (2 * α) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l154_15403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l154_15421

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := cos (2 * x) + 6 * cos (π / 2 - x)

-- State the theorem
theorem f_max_value :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l154_15421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l154_15464

/-- An inverse proportion function with parameter k -/
noncomputable def inverse_proportion (k : ℝ) : ℝ → ℝ := fun x ↦ (2 - k) / x

/-- The condition for a function to be in the first and third quadrants -/
def in_first_and_third_quadrants (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≠ 0 → (x > 0 → f x > 0) ∧ (x < 0 → f x < 0)

/-- Theorem stating that for the inverse proportion function to be in the first and third quadrants, k must be less than 2 -/
theorem inverse_proportion_quadrants (k : ℝ) :
  in_first_and_third_quadrants (inverse_proportion k) ↔ k < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l154_15464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mom_eats_three_times_a_day_l154_15457

/-- Represents the dog fostering scenario -/
structure DogFostering where
  momMealSize : ℚ  -- Size of mom's meal in cups
  puppyCount : ℕ  -- Number of puppies
  puppyMealSize : ℚ  -- Size of puppy's meal in cups
  puppyMealsPerDay : ℕ  -- Number of meals per day for puppies
  totalFoodNeeded : ℚ  -- Total food needed for all dogs in cups
  daysConsidered : ℕ  -- Number of days considered

/-- Calculates the number of times the mom foster dog eats per day -/
def momMealsPerDay (df : DogFostering) : ℚ :=
  let puppyFoodPerDay := df.puppyCount * df.puppyMealSize * df.puppyMealsPerDay
  let totalPuppyFood := puppyFoodPerDay * df.daysConsidered
  let momTotalFood := df.totalFoodNeeded - totalPuppyFood
  let momMealsTotal := momTotalFood / df.momMealSize
  momMealsTotal / df.daysConsidered

/-- Theorem stating that the mom foster dog eats 3 times a day -/
theorem mom_eats_three_times_a_day (df : DogFostering) 
    (h1 : df.momMealSize = 3/2)
    (h2 : df.puppyCount = 5)
    (h3 : df.puppyMealSize = 1/2)
    (h4 : df.puppyMealsPerDay = 2)
    (h5 : df.totalFoodNeeded = 57)
    (h6 : df.daysConsidered = 6) : 
  momMealsPerDay df = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mom_eats_three_times_a_day_l154_15457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_bound_l154_15447

/-- The eccentricity of an ellipse with the given conditions is bounded. -/
theorem ellipse_eccentricity_bound (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : b < a) (hc : 0 < c) :
  ∃ (x y : ℝ), 
    (x^2 / a^2 + y^2 / b^2 = 1) ∧ 
    ((x + 2*c)^2 + y^2 = 2*((x + c)^2 + y^2)) →
    (Real.sqrt 3 / 3 : ℝ) ≤ c / a ∧ c / a ≤ (Real.sqrt 2 / 2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_bound_l154_15447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_by_ten_grid_division_l154_15453

/-- Represents a square grid --/
structure Grid :=
  (size : Nat)

/-- Represents a division of a grid into smaller squares --/
structure GridDivision :=
  (grid : Grid)
  (square_sizes : List Nat)
  (square_counts : List Nat)

/-- Checks if a grid division is valid --/
def is_valid_division (d : GridDivision) : Prop :=
  d.square_sizes.length = d.square_counts.length ∧
  d.square_sizes.length > 1 ∧
  d.square_counts.all (λ c => c > 0) ∧
  d.square_counts.all (λ c => c = d.square_counts.head!) ∧
  (d.square_sizes.zip d.square_counts).foldl (λ acc (s, c) => acc + s * s * c) 0 = d.grid.size * d.grid.size

/-- Theorem stating that a 10x10 grid can be divided into squares of two different sizes with equal counts --/
theorem ten_by_ten_grid_division :
  ∃ (d : GridDivision), d.grid.size = 10 ∧ is_valid_division d := by
  sorry

#check ten_by_ten_grid_division

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_by_ten_grid_division_l154_15453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_four_positive_l154_15489

theorem tan_four_positive : Real.tan 4 > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_four_positive_l154_15489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_greater_than_two_range_of_k_l154_15437

def f (x : ℝ) : ℝ := |x - 1| - |x + 3|

theorem solution_set_f_greater_than_two :
  {x : ℝ | f x > 2} = {x : ℝ | x < -2} := by sorry

theorem range_of_k (k : ℝ) :
  (∀ x ∈ Set.Icc (-3) (-1), f x ≤ k * x + 1) ↔ k ≤ -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_greater_than_two_range_of_k_l154_15437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_black_probability_l154_15445

/-- The probability of a square being initially black -/
noncomputable def p_black : ℝ := 3/4

/-- The number of squares in the grid -/
def grid_size : ℕ := 16

/-- The number of pairs of squares that swap positions during rotation -/
def num_pairs : ℕ := grid_size / 2

/-- The probability that a pair of squares ends up black after the operation -/
noncomputable def p_pair_black : ℝ := 1 - (1 - p_black)^2

/-- The probability that the entire grid becomes black after the operation -/
noncomputable def p_all_black : ℝ := p_pair_black ^ num_pairs

theorem grid_black_probability :
  p_all_black = 2562890625 / 4294967296 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_black_probability_l154_15445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_divisible_by_81_l154_15480

/-- Returns the number formed by reversing the digits of a natural number. -/
def reverse_digits (n : ℕ) : ℕ :=
  sorry

/-- Returns the sum of the digits of a natural number. -/
def sum_of_digits (n : ℕ) : ℕ :=
  sorry

/-- Given a positive integer N that is divisible by 81, and the number formed by reversing
    its digits is also divisible by 81, prove that the sum of the digits of N is divisible by 81. -/
theorem sum_of_digits_divisible_by_81 (N : ℕ+) 
  (h1 : 81 ∣ N.val)
  (h2 : 81 ∣ (reverse_digits N.val)) : 
  81 ∣ (sum_of_digits N.val) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_divisible_by_81_l154_15480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_visited_both_countries_l154_15440

theorem visited_both_countries (total iceland norway neither : ℕ) :
  total = 50 →
  iceland = 25 →
  norway = 23 →
  neither = 23 →
  total - neither = iceland + norway - (Finset.card (Finset.filter (λ x => x ∈ Finset.range iceland ∧ x ∈ Finset.range norway) (Finset.range total))) →
  Finset.card (Finset.filter (λ x => x ∈ Finset.range iceland ∧ x ∈ Finset.range norway) (Finset.range total)) = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_visited_both_countries_l154_15440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersecting_parallel_coplanar_three_points_determine_plane_four_intersecting_lines_not_necessarily_coplanar_perpendicular_lines_not_necessarily_coplanar_l154_15479

-- Define basic geometric objects
variable (Point Line Plane : Type)

-- Define geometric relations
variable (intersects : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (coplanar : Line → Line → Line → Prop)
variable (determines_plane : Point → Point → Point → Plane)
variable (in_plane : Line → Plane → Prop)

-- Theorem 1: A line intersecting two parallel lines is coplanar with those lines
theorem line_intersecting_parallel_coplanar 
  (l1 l2 l3 : Line) 
  (h1 : parallel l1 l2) 
  (h2 : intersects l3 l1) 
  (h3 : intersects l3 l2) : 
  coplanar l1 l2 l3 :=
sorry

-- Theorem 2: Three non-collinear points determine a unique plane
theorem three_points_determine_plane 
  (p1 p2 p3 : Point) 
  (h : ¬ ∃ (l : Line), in_plane l (determines_plane p1 p2 p3)) : 
  ∃! (pl : Plane), pl = determines_plane p1 p2 p3 :=
sorry

-- Theorem 3: Four intersecting lines are not necessarily coplanar
theorem four_intersecting_lines_not_necessarily_coplanar :
  ¬ ∀ (l1 l2 l3 l4 : Line), 
    (intersects l1 l2 ∧ intersects l1 l3 ∧ intersects l1 l4 ∧
     intersects l2 l3 ∧ intersects l2 l4 ∧ intersects l3 l4) →
    ∃ (pl : Plane), in_plane l1 pl ∧ in_plane l2 pl ∧ in_plane l3 pl ∧ in_plane l4 pl :=
sorry

-- Theorem 4: Perpendicular lines are not necessarily coplanar
theorem perpendicular_lines_not_necessarily_coplanar :
  ¬ ∀ (l1 l2 : Line), perpendicular l1 l2 → ∃ (pl : Plane), in_plane l1 pl ∧ in_plane l2 pl :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersecting_parallel_coplanar_three_points_determine_plane_four_intersecting_lines_not_necessarily_coplanar_perpendicular_lines_not_necessarily_coplanar_l154_15479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l154_15436

-- Define the parabola
noncomputable def parabola (y : ℝ) : ℝ := -1/4 * y^2

-- Define the focus
noncomputable def focus : ℝ × ℝ := (-1/2, 0)

-- Define the directrix
def directrix : ℝ := 1

-- Theorem statement
theorem parabola_directrix : 
  ∀ y : ℝ, 
  let p := (parabola y, y)
  (p.1 - focus.1)^2 + (p.2 - focus.2)^2 = (p.1 - directrix)^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l154_15436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_peak_implies_t_bound_l154_15419

-- Define the sequence a_n
noncomputable def a (t : ℝ) (n : ℕ) : ℝ := t * Real.log (n : ℝ) - n

-- Define what it means for a sequence to have no peak
def has_no_peak (t : ℝ) : Prop :=
  ∀ k : ℕ, k ≥ 2 → ¬(a t k ≥ a t (k-1) ∧ a t k ≥ a t (k+1))

-- The theorem to prove
theorem no_peak_implies_t_bound (t : ℝ) :
  has_no_peak t → t < 1 / Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_peak_implies_t_bound_l154_15419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_always_greater_than_x_l154_15434

/-- The function f(x) defined in the problem -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 2 * Real.exp 1 * x^2 + m * x - Real.log x

/-- The theorem stating the condition for f(x) > x to always hold -/
theorem f_always_greater_than_x (m : ℝ) :
  (∀ x > 0, f m x > x) ↔ m > Real.exp 2 + 1 / Real.exp 1 + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_always_greater_than_x_l154_15434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l154_15486

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that A = π/3 and the area of the triangle is 3√3/2 under certain conditions. -/
theorem triangle_proof (a b c A B C : Real) : 
  -- Vector m = (a, √3b) is parallel to vector n = (cos A, sin B)
  a * Real.sin B - Real.sqrt 3 * b * Real.cos A = 0 →
  -- a = √7 and b = 2
  a = Real.sqrt 7 →
  b = 2 →
  -- Sine theorem
  Real.sin A / a = Real.sin B / b →
  -- Cosine theorem
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  -- Conclusion
  A = π/3 ∧ (1/2 * b * c * Real.sin A = 3 * Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l154_15486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_range_on_interval_l154_15495

/-- The range of a quadratic function on a closed interval -/
theorem quadratic_range_on_interval
  (a b c : ℝ) (h_a : a > 0) :
  let g : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c
  let range := Set.image g (Set.Icc (-1 : ℝ) 2)
  range = Set.Icc (min (-b^2 / (4 * a) + c) (a - b + c)) (max (a - b + c) (4 * a + 2 * b + c)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_range_on_interval_l154_15495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_givenCurve_is_hyperbola_l154_15491

/-- A curve defined by parameterized equations -/
structure ParametricCurve where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Definition of the given curve -/
noncomputable def givenCurve : ParametricCurve where
  x := fun t => 2 * (t + 1/t)
  y := fun t => 2 * (t - 1/t)

/-- Definition of a hyperbola -/
def isHyperbola (curve : ParametricCurve) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  ∀ (x y : ℝ), (∃ t, t ≠ 0 ∧ x = curve.x t ∧ y = curve.y t) →
  x^2 / (a^2) - y^2 / (b^2) = 1

/-- Theorem stating that the given curve is a hyperbola -/
theorem givenCurve_is_hyperbola : isHyperbola givenCurve := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_givenCurve_is_hyperbola_l154_15491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l154_15449

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 1

-- Define points A and B
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 0)

-- Define the vector equation
def vector_equation (P : ℝ × ℝ) (lambda mu : ℝ) : Prop :=
  P = (lambda * A.1 + mu * B.1, lambda * A.2 + mu * B.2)

-- Main theorem
theorem min_value_theorem :
  ∃ (min : ℝ), min = 12 ∧
  ∀ (P : ℝ × ℝ) (lambda mu : ℝ),
    circle_equation P.1 P.2 →
    vector_equation P lambda mu →
    11 * lambda + 9 * mu ≥ min :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l154_15449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_A_fill_time_l154_15477

/-- Represents the time it takes for Pipe A to fill the tank alone -/
def time_pipe_A : ℝ := sorry

/-- Represents the time it takes for Pipe B to fill the tank alone -/
def time_pipe_B : ℝ := sorry

/-- The rate at which Pipe B fills the tank compared to Pipe A -/
def rate_B_to_A : ℝ := 5

/-- The time it takes for both pipes to fill the tank together -/
def time_both_pipes : ℝ := 5

/-- Theorem stating that Pipe A takes 30 minutes to fill the tank alone -/
theorem pipe_A_fill_time : 
  time_pipe_B = time_pipe_A / rate_B_to_A →
  1 / time_pipe_A + 1 / time_pipe_B = 1 / time_both_pipes →
  time_pipe_A = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_A_fill_time_l154_15477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l154_15490

-- Define the function f(x) = -1/x
noncomputable def f (x : ℝ) : ℝ := -1/x

-- State the theorem
theorem f_monotone_increasing :
  MonotoneOn f (Set.Ioi 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l154_15490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l154_15400

open Real

theorem inequality_proof (a b c : ℝ) (ha : a = log π) (hb : b = log 2 / log 3) (hc : c = 5 ^ (-(1/2 : ℝ))) :
  c < b ∧ b < a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l154_15400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_monotonic_implies_a_range_l154_15465

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 - 2*x + 5

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x - 2

-- State the theorem
theorem not_monotonic_implies_a_range (a : ℝ) :
  (∃ x y, x ∈ Set.Ioo (1/3 : ℝ) (1/2 : ℝ) ∧ y ∈ Set.Ioo (1/3 : ℝ) (1/2 : ℝ) ∧ f a x < f a y ∧
   ∃ u v, u ∈ Set.Ioo (1/3 : ℝ) (1/2 : ℝ) ∧ v ∈ Set.Ioo (1/3 : ℝ) (1/2 : ℝ) ∧ f a u > f a v) →
  a ∈ Set.Ioo (5/4 : ℝ) (5/2 : ℝ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_monotonic_implies_a_range_l154_15465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_tower_heights_l154_15428

/-- Represents the possible height contributions of a single brick in inches -/
inductive BrickHeight : Type
  | small : BrickHeight  -- 3 inches
  | medium : BrickHeight -- 11 inches
  | large : BrickHeight  -- 20 inches

/-- Calculates the total height of a tower given a list of brick orientations -/
def towerHeight (bricks : List BrickHeight) : ℕ :=
  bricks.foldl (fun acc b => acc + match b with
    | BrickHeight.small => 3
    | BrickHeight.medium => 11
    | BrickHeight.large => 20) 0

/-- Theorem stating the number of distinct tower heights -/
theorem distinct_tower_heights :
  (∃ (heights : Finset ℕ), 
    (∀ h ∈ heights, ∃ (bricks : List BrickHeight), bricks.length = 100 ∧ towerHeight bricks = h) ∧
    (∀ h, (∃ (bricks : List BrickHeight), bricks.length = 100 ∧ towerHeight bricks = h) → h ∈ heights) ∧
    heights.card = 1701) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_tower_heights_l154_15428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_sum_l154_15438

/-- Given complex numbers a, b, c forming an equilateral triangle with side length 18
    and |a + b + c| = 36, prove that |bc + ca + ab| = 432 -/
theorem equilateral_triangle_sum (a b c : ℂ) : 
  (∀ (x y : ℂ), x ∈ ({a, b, c} : Set ℂ) ∧ y ∈ ({a, b, c} : Set ℂ) ∧ x ≠ y → Complex.abs (x - y) = 18) →
  Complex.abs (a + b + c) = 36 →
  Complex.abs (b * c + c * a + a * b) = 432 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_sum_l154_15438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_f_to_line_f_leq_g_iff_a_geq_one_l154_15493

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x
def g (a : ℝ) (x : ℝ) : ℝ := a^2 * x^2

-- Define the distance function from a point to a line
noncomputable def distanceToLine (x y : ℝ) : ℝ := 
  |x - y + 3| / Real.sqrt 2

-- Theorem for the minimum distance
theorem min_distance_f_to_line : 
  ∃ (x : ℝ), x > 0 ∧ 
  ∀ (t : ℝ), t > 0 → distanceToLine x (f (-1) x) ≤ distanceToLine t (f (-1) t) ∧
  distanceToLine x (f (-1) x) = (4 + Real.log 2) * Real.sqrt 2 / 2 := by sorry

-- Theorem for the inequality between f and g
theorem f_leq_g_iff_a_geq_one :
  ∀ (a : ℝ), (∀ (x : ℝ), x > 0 → f a x ≤ g a x) ↔ a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_f_to_line_f_leq_g_iff_a_geq_one_l154_15493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_empty_l154_15435

-- Define the sets A and B
def A : Set String := {"line"}
def B : Set String := {"ellipse"}

-- Theorem statement
theorem intersection_empty : A ∩ B = ∅ := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_empty_l154_15435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_equals_sqrt_5_l154_15408

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define z
noncomputable def z : ℂ := (3 - i) / (1 + i)

-- Theorem statement
theorem abs_z_equals_sqrt_5 : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_equals_sqrt_5_l154_15408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_12_54_l154_15496

/-- The area of a circular sector with given radius and central angle -/
noncomputable def sectorArea (radius : ℝ) (centralAngle : ℝ) : ℝ :=
  (centralAngle / 360) * Real.pi * radius^2

/-- Theorem: The area of a circular sector with radius 12 meters and central angle 54 degrees -/
theorem sector_area_12_54 :
  sectorArea 12 54 = (54 / 360) * Real.pi * 12^2 := by
  rfl

/-- Approximate numerical value of the sector area -/
def approxSectorArea : ℚ :=
  (54 / 360) * 355/113 * 12^2

#eval approxSectorArea

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_12_54_l154_15496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_of_solution_l154_15484

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

noncomputable def frac (x : ℝ) : ℝ := x - ↑(floor x)

theorem absolute_difference_of_solution (x y : ℝ) 
  (eq1 : (floor x : ℝ) + frac y = 3.7)
  (eq2 : frac x + (floor y : ℝ) = 4.3) : 
  |x - y| = 1.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_of_solution_l154_15484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_count_l154_15402

-- Define the function f
def f (x : ℝ) : ℝ := |x| - 1

-- Define the n-fold composition of f
def f_n (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => id
  | n + 1 => f ∘ (f_n n)

-- State the theorem
theorem equation_roots_count :
  ∃ (S : Set ℝ), (∀ x ∈ S, f_n 10 x + 1/2 = 0) ∧ S.Finite ∧ S.ncard = 512 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_count_l154_15402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_ellipse_l154_15414

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  4 * x^2 + 16 * x + 9 * y^2 + 36 * y + 64 = 0

/-- The area of the ellipse -/
noncomputable def ellipse_area : ℝ := 2 * Real.pi

/-- Theorem: The area of the ellipse defined by the given equation is 2π -/
theorem area_of_ellipse :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), ellipse_equation x y ↔ (x + 2)^2 / (3 * a^2) + (y + 2)^2 / (3 * b^2) = 1) ∧
  ellipse_area = Real.pi * a * b := by
  sorry

#check area_of_ellipse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_ellipse_l154_15414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_MQ_is_3_AB_passes_through_fixed_point_l154_15451

-- Define the circle ⊙M
noncomputable def circle_M (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

-- Define a point Q on the x-axis
def point_on_x_axis (Q : ℝ × ℝ) : Prop := Q.2 = 0

-- Define the length of AB
noncomputable def length_AB : ℝ := 4 * Real.sqrt 2 / 3

-- Define the center of the circle
def M : ℝ × ℝ := (0, 2)

-- Define IsTangentLine (placeholder)
def IsTangentLine (Q A : ℝ × ℝ) (circle : ℝ → ℝ → Prop) : Prop := sorry

-- Define EuclideanDistance (placeholder)
noncomputable def EuclideanDistance (A B : ℝ × ℝ) : ℝ := sorry

-- Define IsLine (placeholder)
def IsLine (line : Set (ℝ × ℝ)) : Prop := sorry

-- Theorem 1: If |AB| = 4√2/3, then |MQ| = 3
theorem distance_MQ_is_3 (Q : ℝ × ℝ) (A B : ℝ × ℝ) 
  (h_Q : point_on_x_axis Q)
  (h_A : circle_M A.1 A.2)
  (h_B : circle_M B.1 B.2)
  (h_QA_tangent : IsTangentLine Q A circle_M)
  (h_QB_tangent : IsTangentLine Q B circle_M)
  (h_AB_length : EuclideanDistance A B = length_AB) :
  EuclideanDistance M Q = 3 := by sorry

-- Theorem 2: Line AB always passes through the fixed point (0, 3/2)
theorem AB_passes_through_fixed_point (Q : ℝ × ℝ) (A B : ℝ × ℝ) 
  (h_Q : point_on_x_axis Q)
  (h_A : circle_M A.1 A.2)
  (h_B : circle_M B.1 B.2)
  (h_QA_tangent : IsTangentLine Q A circle_M)
  (h_QB_tangent : IsTangentLine Q B circle_M) :
  ∃ (line : Set (ℝ × ℝ)), IsLine line ∧ A ∈ line ∧ B ∈ line ∧ (0, 3/2) ∈ line := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_MQ_is_3_AB_passes_through_fixed_point_l154_15451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_assembly_line_output_l154_15458

/-- Represents the production rate and order quantity for a phase of production -/
structure ProductionPhase where
  rate : ℚ  -- Production rate in cogs per hour
  order : ℚ  -- Number of cogs to produce

/-- Calculates the overall average output given a list of production phases -/
def overallAverageOutput (phases : List ProductionPhase) : ℚ :=
  let totalCogs := phases.foldl (fun acc phase => acc + phase.order) 0
  let totalTime := phases.foldl (fun acc phase => acc + phase.order / phase.rate) 0
  totalCogs / totalTime

/-- Theorem stating the overall average output for the given production scenario -/
theorem assembly_line_output : 
  let phases := [
    ⟨15, 60⟩,  -- Initial phase
    ⟨60, 60⟩,  -- Second phase
    ⟨90, 180⟩, -- Third phase
    ⟨45, 90⟩   -- Final phase
  ]
  overallAverageOutput phases = 390 / 9 := by
  sorry

#eval overallAverageOutput [
  ⟨15, 60⟩,
  ⟨60, 60⟩,
  ⟨90, 180⟩,
  ⟨45, 90⟩
]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_assembly_line_output_l154_15458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l154_15432

theorem problem_statement : 
  ∀ (m n : ℕ), 
  let P := m^2003 * n^2017 - m^2017 * n^2003
  (P % 24 = 0) ∧ (∃ (m' n' : ℕ), (m'^2003 * n'^2017 - m'^2017 * n'^2003) % 7 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l154_15432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_max_bound_l154_15427

open Real

/-- The function F(x) defined on (0, +∞) -/
noncomputable def F (x : ℝ) : ℝ := (1 / x^2) * (x + 2 * log x)

/-- The maximum value of F(x) on the interval [1, 2] -/
noncomputable def M : ℝ := sSup (Set.image F (Set.Icc 1 2))

/-- Theorem stating that the maximum value M is less than 3/2 -/
theorem F_max_bound : M < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_max_bound_l154_15427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_is_three_l154_15463

/-- The radius of the inscribed circle of a triangle given its side lengths -/
noncomputable def inscribed_circle_radius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s

/-- Theorem: The radius of the inscribed circle in triangle PQR is 3 -/
theorem inscribed_circle_radius_is_three :
  inscribed_circle_radius 8 17 15 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_is_three_l154_15463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_amount_after_two_years_l154_15472

/-- The monthly growth rate for an amount that increases by 1/12th of itself each month -/
noncomputable def monthlyGrowthRate : ℝ := 1 + 1 / 12

/-- The initial amount in rupees -/
def initialAmount : ℝ := 64000

/-- The number of months in two years -/
def monthsInTwoYears : ℕ := 24

/-- The final amount after two years of monthly growth -/
noncomputable def finalAmount : ℝ := initialAmount * monthlyGrowthRate ^ monthsInTwoYears

theorem final_amount_after_two_years :
  ‖finalAmount - 75139.84‖ < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_amount_after_two_years_l154_15472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_upper_bound_l154_15405

theorem log_sum_upper_bound (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 1) :
  Real.log (a / (b * c)) / Real.log a + Real.log (b / (c * a)) / Real.log b + Real.log (c / (a * b)) / Real.log c ≤ -3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_upper_bound_l154_15405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_intersecting_segments_l154_15473

noncomputable section

/-- A point in the plane -/
structure Point := (x : ℝ) (y : ℝ)

/-- Distance between two points -/
def dist (p q : Point) : ℝ := Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Angle between three points -/
def angle (p q r : Point) : ℝ := sorry

/-- Check if two line segments intersect -/
def intersect (p1 q1 p2 q2 : Point) : Prop := sorry

/-- Check if angles have the same orientation -/
def same_orientation (α β : ℝ) : Prop := sorry

theorem non_intersecting_segments
  (n : ℕ)
  (A : ℕ → Point)
  (h1 : ∀ i < n, dist (A i) (A (i+1)) ≤ 1 / (2*i + 1) * dist (A (i+1)) (A (i+2)))
  (h2 : ∀ i < n-2, 0 < angle (A i) (A (i+1)) (A (i+2)))
  (h3 : ∀ i j, i < j → j < n-2 → angle (A i) (A (i+1)) (A (i+2)) < angle (A j) (A (j+1)) (A (j+2)))
  (h4 : ∀ i < n-2, angle (A i) (A (i+1)) (A (i+2)) < Real.pi)
  (h5 : ∀ i < n-2, same_orientation (angle (A i) (A (i+1)) (A (i+2))) (angle (A (i+1)) (A (i+2)) (A (i+3))))
  : ∀ k m, 0 ≤ k → k ≤ m-2 → m-2 < n-2 → 
    ¬(intersect (A k) (A (k+1)) (A m) (A (m+1))) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_intersecting_segments_l154_15473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_l154_15420

noncomputable section

def f (x : ℝ) : ℝ → ℝ := fun f' => Real.log x - f' * x^2 + 3 * x + 2

theorem f_derivative_at_one :
  ∃ f', (deriv (f f')) 1 = 4/3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_l154_15420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oplus_negative_sequence_equals_negative_one_l154_15424

noncomputable def oplus (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

def negativeSequence (n : ℕ) : ℝ := -(n : ℝ)

noncomputable def foldrOplus (n : ℕ) : ℝ :=
  List.foldr oplus (-1000 : ℝ) (List.map negativeSequence (List.range 998))

theorem oplus_negative_sequence_equals_negative_one :
  oplus (-1 : ℝ) (foldrOplus 999) = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oplus_negative_sequence_equals_negative_one_l154_15424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l154_15431

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The main theorem -/
theorem triangle_properties (t : Triangle) : 
  (t.a * Real.sin (t.A + t.B) = t.c * Real.sin ((t.B + t.C) / 2)) →
  (∃ (D : Real), 
    (t.a = 4 ∧ t.b * t.c = 3 → t.a + t.b + t.c = 9) ∧
    (t.b = 1 ∧ t.c = 3 ∧ D * t.c = 3 * (t.c - D) → 
      Real.sqrt ((3 * Real.sqrt 3) / 4) ^ 2 = D)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l154_15431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_value_l154_15488

noncomputable def X (x a : ℝ) : ℝ := 2^x / (2^x - 1) + a

theorem odd_function_implies_a_value :
  (∀ x, X x a = -X (-x) a) → a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_value_l154_15488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_2_equals_217_div_16_l154_15497

-- Define the functions t and s
def t (x : ℝ) : ℝ := 4 * x - 9

noncomputable def s (y : ℝ) : ℝ := 
  let x := (y + 9) / 4  -- This is t⁻¹(y)
  x^2 + 4 * x - 5

-- Theorem to prove
theorem s_of_2_equals_217_div_16 : s 2 = 217 / 16 := by
  -- Expand the definition of s
  unfold s
  -- Simplify the expression
  simp [add_div, mul_div, pow_two]
  -- The proof is completed
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_2_equals_217_div_16_l154_15497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_votes_l154_15439

/-- Given two candidates A and B receiving p% and q% of votes respectively,
    with a difference of D votes between them, prove that the total number
    of votes V is equal to (D * 100) / (p - q). -/
theorem total_votes (p q : ℝ) (D : ℝ) :
  let V := (D * 100) / (p - q)
  (p / 100) * V - (q / 100) * V = D := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_votes_l154_15439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l154_15409

theorem system_solution (x : Fin 8 → ℤ) (h : x = ![1, 2, 3, 4, -4, -3, -2, -1]) : 
  (x 0 + x 1 + x 2 = 6) ∧
  (x 1 + x 2 + x 3 = 9) ∧
  (x 2 + x 3 + x 4 = 3) ∧
  (x 3 + x 4 + x 5 = -3) ∧
  (x 4 + x 5 + x 6 = -9) ∧
  (x 5 + x 6 + x 7 = -6) ∧
  (x 6 + x 7 + x 0 = -2) ∧
  (x 7 + x 0 + x 1 = 2) := by
  rw [h]
  simp
  native_decide

#check system_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l154_15409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l154_15410

/-- Fibonacci sequence -/
def fib : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- The sequence a_n -/
noncomputable def a (x y : ℝ) : ℕ → ℝ
  | 0 => x
  | 1 => y
  | (n + 2) => (a x y n * a x y (n + 1) + 1) / (a x y n + a x y (n + 1))

/-- The general term of the sequence -/
noncomputable def a_general_term (x y : ℝ) (n : ℕ) : ℝ :=
  ((x + 1)^(fib (n - 2)) * (y + 1)^(fib (n - 1)) + (x - 1)^(fib (n - 2)) * (y - 1)^(fib (n - 1))) /
  ((x + 1)^(fib (n - 2)) * (y + 1)^(fib (n - 1)) - (x - 1)^(fib (n - 2)) * (y - 1)^(fib (n - 1)))

theorem sequence_properties (x y : ℝ) :
  (∃ (n₀ : ℕ), ∀ (n : ℕ), n ≥ n₀ → a x y n = a x y n₀) ↔ 
  ((abs x = 1 ∧ y + x ≠ 0) ∨ (abs y = 1 ∧ y + x ≠ 0)) ∧
  (∀ (n : ℕ), a x y n = a_general_term x y n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l154_15410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_g_intersection_result_l154_15425

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + 1 / Real.sqrt (2 - x)
def g (x : ℝ) : ℝ := x^2 + 1

-- Define the sets A, B, and U
def A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def B : Set ℝ := {y | 1 ≤ y}
def U : Set ℝ := Set.univ

-- State the theorems to be proved
theorem domain_of_f : Set.Icc (-1 : ℝ) 2 = A := by sorry

theorem range_of_g : Set.Ici (1 : ℝ) = B := by sorry

theorem intersection_result : A ∩ (U \ B) = Set.Ioc (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_g_intersection_result_l154_15425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l154_15483

theorem tan_alpha_value (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π/4)) 
  (h2 : Real.sin (α + π/4) = 4/5) : 
  Real.tan α = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l154_15483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_correct_l154_15454

-- Define the functions f, g, and h
noncomputable def f (x : ℝ) : ℝ := 4 * x + 5
noncomputable def g (x : ℝ) : ℝ := 3 * x - 4
noncomputable def h (x : ℝ) : ℝ := f (g x)

-- Define the inverse function of h
noncomputable def h_inv (x : ℝ) : ℝ := (x + 11) / 12

-- Theorem statement
theorem h_inverse_correct : Function.LeftInverse h_inv h ∧ Function.RightInverse h_inv h := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_correct_l154_15454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l154_15422

open Real

-- Define the function f
noncomputable def f (x a : ℝ) : ℝ := cos (2 * x - π / 3) - 2 * sin x ^ 2 + a

-- Theorem for part (I)
theorem part_one (a : ℝ) : f (π / 3) a = 0 → a = 1 := by sorry

-- Define the simplified function g after solving for a
noncomputable def g (x : ℝ) : ℝ := Real.sqrt 3 * sin (2 * x + π / 3)

-- Theorem for part (II)
theorem part_two (m : ℝ) :
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ m → (g x < g y ∨ g x > g y)) →
  m ≤ π / 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l154_15422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_y_equals_27_l154_15429

theorem sum_x_y_equals_27 (x y : ℝ) 
  (h1 : (2 : ℝ)^x = (8 : ℝ)^(y+1)) 
  (h2 : (9 : ℝ)^y = (3 : ℝ)^(x-9)) : 
  x + y = 27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_y_equals_27_l154_15429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l154_15406

/-- Calculates the length of a bridge given ship parameters and time to pass --/
noncomputable def bridge_length (ship_length : ℝ) (ship_speed_kmh : ℝ) (time_to_pass : ℝ) : ℝ :=
  let ship_speed_ms := ship_speed_kmh * (1000 / 3600)
  let total_distance := ship_speed_ms * time_to_pass
  total_distance - ship_length

/-- Theorem stating that the bridge length is approximately 900.54 m given the specific conditions --/
theorem bridge_length_calculation :
  let ship_length : ℝ := 450
  let ship_speed_kmh : ℝ := 24
  let time_to_pass : ℝ := 202.48
  abs (bridge_length ship_length ship_speed_kmh time_to_pass - 900.54) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l154_15406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trevor_annual_spending_l154_15433

/-- Trevor's annual toy spending -/
noncomputable def trevor_spending : ℚ := 80

/-- Reed's annual toy spending -/
noncomputable def reed_spending : ℚ := trevor_spending - 20

/-- Quinn's annual toy spending -/
noncomputable def quinn_spending : ℚ := reed_spending / 2

/-- Total spending by all three friends over 4 years -/
def total_spending : ℚ := 680

theorem trevor_annual_spending :
  trevor_spending = 80 ∧
  trevor_spending = reed_spending + 20 ∧
  reed_spending = 2 * quinn_spending ∧
  4 * (trevor_spending + reed_spending + quinn_spending) = total_spending :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trevor_annual_spending_l154_15433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fourth_and_sixth_terms_l154_15401

noncomputable def customSequence : ℕ → ℚ
  | 0 => 1
  | n + 1 => ((n + 2 : ℚ)^3) / ((n + 1 : ℚ)^3)

theorem sum_of_fourth_and_sixth_terms :
  customSequence 3 + customSequence 5 = 6121 / 1728 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fourth_and_sixth_terms_l154_15401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_invariant_circles_existence_l154_15467

/-- A convex polygon type -/
structure ConvexPolygon where
  -- Add necessary fields for a convex polygon
  vertices : List (ℝ × ℝ)

/-- Rotation invariance property -/
def is_90_degree_rotation_invariant (M : ConvexPolygon) : Prop :=
  -- Define the property of being invariant under 90° rotation
  sorry

/-- Circle type -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Containment relation between a circle and a polygon -/
def contains (C : Circle) (M : ConvexPolygon) : Prop :=
  -- Define when a circle contains a polygon
  sorry

/-- Containment relation between a polygon and a circle -/
def contains_circle (M : ConvexPolygon) (C : Circle) : Prop :=
  -- Define when a polygon contains a circle
  sorry

/-- Main theorem -/
theorem rotation_invariant_circles_existence 
  (M : ConvexPolygon) 
  (h : is_90_degree_rotation_invariant M) : 
  ∃ (C1 C2 : Circle), 
    contains C1 M ∧ 
    contains_circle M C2 ∧ 
    C1.radius / C2.radius = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_invariant_circles_existence_l154_15467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l154_15469

noncomputable section

def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  -- Definition of triangle ABC with sides a, b, c opposite to angles A, B, C
  true

def vector_m (C : ℝ) : ℝ × ℝ := (Real.cos (C / 2), Real.sin (C / 2))

def vector_n (C : ℝ) : ℝ × ℝ := (Real.cos (C / 2), -Real.sin (C / 2))

def angle_between (v w : ℝ × ℝ) : ℝ :=
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

def area_triangle (a b C : ℝ) : ℝ := (1/2) * a * b * Real.sin C

theorem triangle_properties (a b c A B C : ℝ) :
  triangle_ABC a b c A B C →
  angle_between (vector_m C) (vector_n C) = π/3 →
  c = 7/2 →
  area_triangle a b C = 3/2 * Real.sqrt 3 →
  C = π/3 ∧ a + b = 11/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l154_15469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_increasing_l154_15494

def a_set : Finset ℕ := {0, 1, 2}
def b_set : Finset ℤ := {-1, 1, 3, 5}

def f (a : ℕ) (b : ℤ) (x : ℝ) : ℝ := a * x^2 - 2 * b * x

def is_increasing (a : ℕ) (b : ℤ) : Prop :=
  a ∈ a_set ∧ b ∈ b_set ∧
  (∀ x y : ℝ, x > 1 → y > x → f a b y > f a b x)

def count_increasing : ℕ := 5

theorem probability_increasing :
  (count_increasing : ℚ) / (a_set.card * b_set.card) = 5 / 12 := by
  -- Proof goes here
  sorry

#eval a_set.card * b_set.card  -- Should output 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_increasing_l154_15494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_to_exponential_form_l154_15499

-- Define the complex number
noncomputable def z : ℂ := 1 + Complex.I * Real.sqrt 3

-- Theorem statement
theorem complex_to_exponential_form : 
  Complex.arg z = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_to_exponential_form_l154_15499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ore_alloy_percentage_l154_15471

theorem ore_alloy_percentage (total_ore : ℝ) (pure_iron : ℝ) (alloy_iron_percentage : ℝ) :
  total_ore = 266.6666666666667 →
  pure_iron = 60 →
  alloy_iron_percentage = 0.9 →
  (25 / 100) * total_ore * alloy_iron_percentage = pure_iron := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ore_alloy_percentage_l154_15471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_similar_lattice_triangle_circumcenter_not_lattice_l154_15417

/-- A lattice point in a 2D grid -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A lattice triangle in a 2D grid -/
structure LatticeTriangle where
  a : LatticePoint
  b : LatticePoint
  c : LatticePoint

/-- The circumcenter of a triangle -/
noncomputable def circumcenter (t : LatticeTriangle) : ℝ × ℝ := sorry

/-- Predicate to check if a point is a lattice point -/
def isLatticePoint (p : ℝ × ℝ) : Prop := ∃ (x y : ℤ), p = (↑x, ↑y)

/-- Predicate to check if two triangles are similar -/
def areSimilar (t1 t2 : LatticeTriangle) : Prop := sorry

/-- The area of a lattice triangle -/
noncomputable def area (t : LatticeTriangle) : ℝ := sorry

/-- Predicate to check if a triangle is the smallest area lattice triangle similar to another -/
def isSmallestSimilar (t : LatticeTriangle) (original : LatticeTriangle) : Prop :=
  areSimilar t original ∧
  ∀ (t' : LatticeTriangle), areSimilar t' original → area t ≤ area t'

theorem smallest_similar_lattice_triangle_circumcenter_not_lattice
  (t : LatticeTriangle) (original : LatticeTriangle) :
  isSmallestSimilar t original → ¬isLatticePoint (circumcenter t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_similar_lattice_triangle_circumcenter_not_lattice_l154_15417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_three_times_faster_than_b_l154_15470

/-- The time (in days) it takes for A and B together to complete the work -/
noncomputable def combined_time : ℝ := 15

/-- The time (in days) it takes for A alone to complete the work -/
noncomputable def a_time : ℝ := 20

/-- The rate at which A works (portion of work completed per day) -/
noncomputable def a_rate : ℝ := 1 / a_time

/-- The rate at which B works (portion of work completed per day) -/
noncomputable def b_rate : ℝ := 1 / combined_time - a_rate

/-- The factor by which A is faster than B -/
noncomputable def speed_factor : ℝ := a_rate / b_rate

theorem a_three_times_faster_than_b : speed_factor = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_three_times_faster_than_b_l154_15470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_minimum_l154_15461

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 / x + 9 / (1 - 2 * x)

-- State the theorem
theorem inequality_and_minimum :
  ∀ (a b x y : ℝ),
    a > 0 → b > 0 → a ≠ b → x > 0 → y > 0 →
    (a^2 / x + b^2 / y ≥ (a + b)^2 / (x + y)) ∧
    (a^2 / x + b^2 / y = (a + b)^2 / (x + y) ↔ a / x = b / y) ∧
    (∀ z : ℝ, 0 < z → z < 1/2 → f z ≥ 25) ∧
    (f (1/5) = 25) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_minimum_l154_15461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_floor_equation_l154_15481

theorem min_floor_equation (n : ℕ) : 
  (∀ k : ℕ+, (k : ℝ)^2 + ⌊(n : ℝ) / (k : ℝ)^2⌋ ≥ 1991) ∧ 
  (∃ k : ℕ+, (k : ℝ)^2 + ⌊(n : ℝ) / (k : ℝ)^2⌋ < 1992) ↔ 
  1024 * 967 ≤ n ∧ n ≤ 1024 * 967 + 1023 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_floor_equation_l154_15481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_triangle_area_ratio_l154_15462

theorem rectangle_triangle_area_ratio :
  ∀ (L W : ℝ), L > 0 → W > 0 →
  (L * W) / ((1 / 2) * L * W) = 2 := by
  intros L W hL hW
  field_simp
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_triangle_area_ratio_l154_15462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_couponA_best_at_209_95_l154_15474

/-- Represents the discount amount for a given price and discount percentage -/
noncomputable def discount (price : ℝ) (percentage : ℝ) : ℝ :=
  price * percentage / 100

/-- Represents the discount amount for coupon A -/
noncomputable def couponA (price : ℝ) : ℝ :=
  if price ≥ 75 then discount price 15 else 0

/-- Represents the discount amount for coupon B -/
noncomputable def couponB (price : ℝ) : ℝ :=
  if price ≥ 120 then 30 else 0

/-- Represents the discount amount for coupon C -/
noncomputable def couponC (price : ℝ) : ℝ :=
  if price > 120 then discount (price - 120) 25 else 0

/-- Theorem stating that for a price of $209.95, coupon A offers a greater discount than both B and C -/
theorem couponA_best_at_209_95 :
  let price : ℝ := 209.95
  couponA price > couponB price ∧ couponA price > couponC price := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_couponA_best_at_209_95_l154_15474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_infinite_families_l154_15492

/-- A number is expressible as the sum of three positive integer cubes -/
def is_sum_of_three_cubes (n : ℕ) : Prop :=
  ∃ a b c : ℕ, n = a^3 + b^3 + c^3 ∧ a > 0 ∧ b > 0 ∧ c > 0

/-- For each i in {1, 2, 3}, there exists a function generating infinitely many
    numbers satisfying the required condition -/
theorem existence_of_infinite_families :
  ∀ i : Fin 3, ∃ f : ℕ → ℕ,
    ∀ m : ℕ, (i.val + 1 : ℕ) =
      (if ∃ x, x = is_sum_of_three_cubes (f m) then 1 else 0) +
      (if ∃ x, x = is_sum_of_three_cubes (f m + 2) then 1 else 0) +
      (if ∃ x, x = is_sum_of_three_cubes (f m + 28) then 1 else 0) :=
by sorry

#check existence_of_infinite_families

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_infinite_families_l154_15492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_matrix_equation_l154_15404

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 2, 4],
    ![2, 0, 2],
    ![4, 2, 0]]

theorem cubic_matrix_equation :
  ∃ (s t u : ℤ), 
    B^3 + s • B^2 + t • B + u • (1 : Matrix (Fin 3) (Fin 3) ℤ) = 0 ∧ 
    s = 0 ∧ t = -36 ∧ u = -48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_matrix_equation_l154_15404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l154_15448

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + 1 / (x + 1)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > -1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l154_15448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_area_l154_15416

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculates the area of a triangle given three vertices -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- The vertices of the kite -/
def v1 : Point := ⟨2, 3⟩
def v2 : Point := ⟨6, 7⟩
def v3 : Point := ⟨10, 3⟩
def v4 : Point := ⟨6, 0⟩

/-- The theorem stating that the area of the kite is 56 square units -/
theorem kite_area : triangleArea v1 v2 v3 + triangleArea v1 v3 v4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_area_l154_15416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_property_l154_15415

noncomputable def projection (v : ℝ × ℝ) (u : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * u.1 + v.2 * u.2
  let magnitude_squared := u.1 * u.1 + u.2 * u.2
  (dot_product / magnitude_squared * u.1, dot_product / magnitude_squared * u.2)

theorem projection_property :
  let v₁ : ℝ × ℝ := (3, 3)
  let p₁ : ℝ × ℝ := (45/10, 9/10)
  let v₂ : ℝ × ℝ := (-3, 3)
  let p₂ : ℝ × ℝ := (-30/13, -6/13)
  let u : ℝ × ℝ := (5, 1)
  projection v₁ u = p₁ → projection v₂ u = p₂ := by
  sorry

#check projection_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_property_l154_15415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_point_l154_15446

/-- Given a curve C and a line l, prove the equation of the tangent line and the coordinates of the tangent point -/
theorem tangent_line_and_point (x₀ : ℝ) (h : x₀ ≠ 0) :
  let C := λ x : ℝ ↦ x^3 - 3*x^2 + 2*x
  let l := λ (k : ℝ) (x : ℝ) ↦ k*x
  let y₀ := C x₀
  let k := y₀ / x₀
  let tangent_slope := 3*x₀^2 - 6*x₀ + 2
  (k = tangent_slope) →
  (k = -1/4 ∧ x₀ = 3/2 ∧ y₀ = -3/8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_point_l154_15446
