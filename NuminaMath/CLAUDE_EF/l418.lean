import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_implies_a_value_l418_41826

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < a then 2*a - x - 4/x - 3 else x - 4/x - 3

theorem three_zeros_implies_a_value (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧
    (∀ x : ℝ, f a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
    2*x₂ = x₁ + x₃) →
  a = -11/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_implies_a_value_l418_41826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_y_axis_l418_41844

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The y-intercept of a line -/
noncomputable def y_intercept (l : Line) : ℝ × ℝ :=
  (0, (l.y₂ - l.y₁) / (l.x₂ - l.x₁) * (0 - l.x₁) + l.y₁)

/-- Theorem: A line passing through (10, 0) and (6, -4) intersects the y-axis at (0, -10) -/
theorem line_intersection_y_axis :
  let l : Line := { x₁ := 10, y₁ := 0, x₂ := 6, y₂ := -4 }
  y_intercept l = (0, -10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_y_axis_l418_41844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l418_41853

theorem problem_solution : 
  let a := 935
  let b := 1383
  let d := 32
  let r := 7
  (a % d = r ∧ b % d = r) ∧ 
  Nat.gcd (a - r) (b - r) = d :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l418_41853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mba_committee_size_l418_41802

/-- Represents the number of members in each committee. -/
def committee_size : ℕ := sorry

/-- The total number of second-year MBAs. -/
def total_mbas : ℕ := 6

/-- The number of committees formed. -/
def num_committees : ℕ := 2

/-- The probability of Jane and Albert being on the same committee. -/
def same_committee_prob : ℚ := 2/5

theorem mba_committee_size :
  (committee_size * num_committees = total_mbas) ∧
  ((committee_size - 1 : ℚ) / (total_mbas - 1) = same_committee_prob) →
  committee_size = 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mba_committee_size_l418_41802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_special_parallelogram_l418_41873

-- Define a parallelogram on a coordinate plane
structure Parallelogram where
  A : ℤ × ℤ
  B : ℤ × ℤ
  C : ℤ × ℤ
  D : ℤ × ℤ

-- Define the length of a line segment between two integer points
noncomputable def length (p1 p2 : ℤ × ℤ) : ℝ :=
  Real.sqrt (((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2) : ℝ)

-- Define the angle between two line segments
noncomputable def angle (p1 p2 p3 : ℤ × ℤ) : ℝ := sorry

-- Theorem stating the impossibility of such a parallelogram
theorem no_special_parallelogram :
  ¬ ∃ (p : Parallelogram),
    (length p.A p.C = 2 * length p.B p.D) ∧
    (angle p.A p.B p.C = π / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_special_parallelogram_l418_41873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_equality_l418_41894

theorem trig_expression_equality : 
  (Real.tan (30 * π / 180) ^ 2 - 1 / (Real.cos (30 * π / 180) ^ 2)) / 
  (Real.tan (30 * π / 180) ^ 2 * (1 / (Real.cos (30 * π / 180) ^ 2))) = -9/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_equality_l418_41894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_perpendicular_straight_lines_l418_41866

/-- Represents a time on a clock -/
structure ClockTime where
  hours : ℚ
  minutes : ℚ

/-- The angle between the hour and minute hands at a given time -/
def handAngle (t : ClockTime) : ℚ :=
  (30 * t.hours - 5.5 * t.minutes) % 360

/-- Predicate for when the hour and minute hands form a straight line -/
def straightLine (t : ClockTime) : Prop :=
  handAngle t = 180 ∨ handAngle t = 0

/-- The theorem to be proved -/
theorem no_perpendicular_straight_lines : 
  ¬ ∃ (t1 t2 : ClockTime), t1 ≠ t2 ∧ straightLine t1 ∧ straightLine t2 ∧ 
    (handAngle t1 - handAngle t2) % 180 = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_perpendicular_straight_lines_l418_41866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_0_736_l418_41846

/-- Represents a repeating decimal with a non-repeating part and a repeating part -/
structure RepeatingDecimal where
  nonRepeating : ℚ
  repeating : ℚ
  repeatingLength : ℕ
  nonRepeatingNonneg : 0 ≤ nonRepeating
  repeatingNonneg : 0 ≤ repeating
  repeatingLtOne : repeating < 1

/-- Converts a RepeatingDecimal to its rational representation -/
def RepeatingDecimal.toRational (d : RepeatingDecimal) : ℚ :=
  d.nonRepeating + d.repeating / (10^d.repeatingLength - 1)

theorem repeating_decimal_0_736 :
  let d : RepeatingDecimal := {
    nonRepeating := 7/10,
    repeating := 36/100,
    repeatingLength := 2,
    nonRepeatingNonneg := by norm_num,
    repeatingNonneg := by norm_num,
    repeatingLtOne := by norm_num
  }
  d.toRational = 2452091 / 3330000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_0_736_l418_41846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_implications_l418_41818

-- Define the type of positive rationals
def PositiveRational := {q : ℚ // q > 0}

-- Define multiplication for PositiveRational
instance : Mul PositiveRational where
  mul a b := ⟨a.val * b.val, mul_pos a.property b.property⟩

-- Define the property of the function
def HasPropertyF (f : PositiveRational → ℚ) :=
  ∀ x y : PositiveRational, f (x * y) = f x + f y

-- Theorem statement
theorem function_property_implications :
  ∀ f : PositiveRational → ℚ,
  HasPropertyF f →
  (¬ Function.Injective f) ∧
  (∃ g : PositiveRational → ℚ, HasPropertyF g ∧ Function.Surjective g) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_implications_l418_41818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_not_even_f_2018_odd_not_even_l418_41849

/-- Definition of the function sequence -/
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => fun _ => 0  -- Define for 0 to avoid missing case
  | 1 => fun x => 1 / x
  | n + 1 => fun x => 1 / (x + f n x)

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The main theorem: for all n, f_n is odd but not even -/
theorem f_odd_not_even : ∀ n : ℕ, n ≥ 1 → IsOdd (f n) ∧ ¬IsEven (f n) := by
  sorry

/-- Corollary: f_2018 is odd but not even -/
theorem f_2018_odd_not_even : IsOdd (f 2018) ∧ ¬IsEven (f 2018) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_not_even_f_2018_odd_not_even_l418_41849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_three_average_l418_41891

def max_difference (a b c d e : ℕ) : ℕ := 
  max (max (max (e - a) (e - b)) (max (e - c) (e - d)))
      (max (max (d - a) (d - b)) (max (d - c) (c - a)))

theorem middle_three_average (a b c d e : ℕ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e →  -- five different positive integers
  (a + b + c + d + e) / 5 = 5 →    -- average is 5
  e - a = max_difference a b c d e →  -- difference between largest and smallest is maximized
  (b + c + d) / 3 = 3 :=           -- average of middle three is 3
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_three_average_l418_41891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_time_problem_l418_41807

/-- The number of days it takes for two workers to complete a job together,
    given their individual completion times. -/
noncomputable def combined_work_time (a_time b_time : ℝ) : ℝ :=
  1 / (1 / a_time + 1 / b_time)

/-- Theorem: If A can do a work in 9 days and B can do the same work in 18 days,
    then A and B working together will finish the work in 6 days. -/
theorem combined_work_time_problem :
  combined_work_time 9 18 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_time_problem_l418_41807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_successful_shots_in_last_five_l418_41836

def initial_shots : ℕ := 20
def final_shots : ℕ := 25
def initial_success_rate : ℚ := 55 / 100
def final_success_rate : ℚ := 56 / 100

theorem successful_shots_in_last_five :
  let initial_successful := (initial_success_rate * initial_shots).floor
  let final_successful := (final_success_rate * final_shots).floor
  final_successful - initial_successful = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_successful_shots_in_last_five_l418_41836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l418_41864

theorem tan_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (Real.pi / 2))
  (h2 : Real.cos (2 * α) + Real.sin (2 * α) = 1 / 5) : 
  Real.tan α = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l418_41864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l418_41898

-- Define a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  angleSum : A + B + C = Real.pi
  areaFormula : (b * c * Real.sin A) / 2 = (b^2 + c^2 - a^2) / 4

-- Define the theorem
theorem triangle_properties (t : Triangle) :
  (t.C = 2 * t.B → Real.cos t.A = 3 * Real.cos t.B - 4 * (Real.cos t.B)^3) ∧
  ((t.b * Real.sin t.B - t.c * Real.sin t.C = t.a) →
   ((t.b^2 + t.c^2 - t.a^2) / 4 = (t.b * t.c * Real.sin t.A) / 2) →
   t.B = Real.pi * 77.5 / 180) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l418_41898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l418_41804

noncomputable def floor (x : ℝ) := ⌊x⌋

noncomputable def frac (x : ℝ) := x - floor x

theorem equation_solutions :
  ∃ (S : Set ℝ), S = {x : ℝ | (floor x) * (frac x) = 1991 * x} ∧
  S = {0, -1/1992} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l418_41804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_first_five_primes_from_five_l418_41827

theorem least_multiple_of_first_five_primes_from_five : 
  ∃ (p : ℕ → Prop),
  (∀ n, p n ↔ n ∈ ({5, 7, 11, 13, 17} : Set ℕ)) ∧
  (∀ n ∈ ({5, 7, 11, 13, 17} : Set ℕ), Nat.Prime n) ∧
  (∀ k < 5, k ≠ 2 → k ≠ 3 → ¬ Nat.Prime k) ∧
  (∀ m : ℕ, m > 0 → (∀ n ∈ ({5, 7, 11, 13, 17} : Set ℕ), m % n = 0) → m ≥ 85085) ∧
  (∀ n ∈ ({5, 7, 11, 13, 17} : Set ℕ), 85085 % n = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_first_five_primes_from_five_l418_41827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_perpendicular_points_l418_41834

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the foci of the ellipse
def foci (F₁ F₂ : ℝ × ℝ) : Prop := 
  F₁.1^2 + F₁.2^2 = F₂.1^2 + F₂.2^2 ∧ 
  F₁.1^2 + F₁.2^2 = 4

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

-- Define the perpendicularity condition
def perpendicular (P F₁ F₂ : ℝ × ℝ) : Prop :=
  (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0

-- Theorem statement
theorem ellipse_perpendicular_points 
  (F₁ F₂ : ℝ × ℝ) 
  (h_foci : foci F₁ F₂) :
  ∃! (s : Set (ℝ × ℝ)), 
    (∀ P ∈ s, point_on_ellipse P ∧ perpendicular P F₁ F₂) ∧ 
    (∃ (l : List (ℝ × ℝ)), s = l.toFinset ∧ l.length = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_perpendicular_points_l418_41834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_inequality_l418_41881

theorem triangle_tangent_inequality (A B C : ℝ) : 
  (0 < A) → (0 < B) → (0 < C) →  -- Angles are positive
  (A + B + C = π) →  -- Sum of angles in a triangle
  (π / 2 ≠ A) →  -- A is not a right angle
  (A ≥ B) →  -- A is the largest angle
  (B ≥ C) →  -- B is not smaller than C
  (|Real.tan A| ≥ Real.tan B ∧ Real.tan B ≥ Real.tan C) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_inequality_l418_41881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stick_breaking_minimal_length_correct_l418_41882

/-- Given a number of sticks and their total length, we can always break them into
    consecutive integer lengths from 1 to the number of sticks. -/
theorem stick_breaking (n : ℕ) :
  ∀ (sticks : List ℕ),
    sticks.length = n →
    sticks.sum = n^2 - n + 1 →
    ∃ (broken_sticks : List ℕ),
      broken_sticks.toFinset = Finset.range n.succ ∧
      broken_sticks.sum = n^2 - n + 1 := by
  sorry

/-- The minimal total length for 2012 sticks that can be broken into lengths 1 to 2012 -/
def minimal_length : ℕ := 2012^2 - 2012 + 1

/-- Proof that the minimal length for 2012 sticks is correct -/
theorem minimal_length_correct :
  ∀ (sticks : List ℕ),
    sticks.length = 2012 →
    sticks.sum = minimal_length →
    ∃ (broken_sticks : List ℕ),
      broken_sticks.toFinset = Finset.range 2013 ∧
      broken_sticks.sum = minimal_length := by
  exact stick_breaking 2012

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stick_breaking_minimal_length_correct_l418_41882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_irreducibility_l418_41888

theorem polynomial_irreducibility (n : ℕ) :
  ¬∃ (g h : Polynomial ℤ) (hg : g ≠ 1) (hh : h ≠ 1),
    (X^2 + X)^(2^n) + 1 = g * h :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_irreducibility_l418_41888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_region_area_l418_41825

/-- The area of the region bounded by arcs of circles and a smaller central circle -/
noncomputable def region_area (R : ℝ) (r : ℝ) (θ : ℝ) : ℝ :=
  6 * (θ / (2 * Real.pi) * Real.pi * R^2 - 1/2 * R^2 * Real.sin θ) - Real.pi * r^2

/-- Theorem stating the area of the specific region described in the problem -/
theorem specific_region_area :
  region_area 6 3 (Real.pi / 6) = 9 * Real.pi - 54 := by
  sorry

-- Remove #eval as it's not necessary for building and might cause issues
-- #eval specific_region_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_region_area_l418_41825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_solution_condition_l418_41887

theorem positive_integer_solution_condition (a : ℕ) (A B : ℝ) :
  (∃ x y z : ℕ+, 
    x^2 + y^2 + z^2 = (13*a)^2 ∧ 
    x^2*(A*x^2 + B*y^2) + y^2*(A*y^2 + B*z^2) + z^2*(A*z^2 + B*x^2) = (1/4)*(2*A + B)*(13*a)^4) 
  ↔ 
  A = (1/2)*B :=
by
  sorry

#check positive_integer_solution_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_solution_condition_l418_41887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l418_41884

noncomputable def determinant (a₁ a₂ a₃ a₄ : ℝ) : ℝ := a₁ * a₄ - a₂ * a₃

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := determinant (Real.sqrt 3) (Real.sin (ω * x)) 1 (Real.cos (ω * x))

noncomputable def shifted_f (ω : ℝ) (x : ℝ) : ℝ := f ω (x + 2 * Real.pi / 3)

def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

theorem omega_value (ω : ℝ) : 
  is_even (shifted_f ω) → ω = -7/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l418_41884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_A_l418_41880

def A : Finset ℕ := {1, 2}

theorem proper_subsets_of_A :
  (Finset.powerset A).card - 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_A_l418_41880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_polynomial_with_roots_l418_41808

-- Define the polynomial
noncomputable def p : Polynomial ℝ := 
  Polynomial.monomial 4 1 - Polynomial.monomial 3 4 + Polynomial.monomial 2 15 - Polynomial.monomial 1 20 + Polynomial.monomial 0 2

-- State the theorem
theorem minimal_polynomial_with_roots :
  -- The polynomial has rational coefficients
  (∀ n, ∃ q : ℚ, p.coeff n = ↑q) ∧
  -- The leading coefficient is 1
  p.leadingCoeff = 1 ∧
  -- 2+√2 and 2+√5 are roots of the polynomial
  p.eval (2 + Real.sqrt 2) = 0 ∧ p.eval (2 + Real.sqrt 5) = 0 ∧
  -- The polynomial has minimal degree
  ∀ q : Polynomial ℝ, q ≠ 0 →
    (∀ n, ∃ r : ℚ, q.coeff n = ↑r) →
    q.leadingCoeff = 1 →
    q.eval (2 + Real.sqrt 2) = 0 →
    q.eval (2 + Real.sqrt 5) = 0 →
    q.natDegree ≥ p.natDegree :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_polynomial_with_roots_l418_41808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_difference_soohyun_hwajun_l418_41839

/-- The height difference between two people in millimeters -/
def HeightDifference (person1 : String) (person2 : String) (difference : ℕ) : Prop :=
  sorry

/-- Soohyun's height in millimeters -/
def SoohyunHeight : ℕ := sorry

theorem height_difference_soohyun_hwajun :
  HeightDifference "Kiyoon" "Soohyun" 207 →
  HeightDifference "Taehun" "Soohyun" 252 →
  HeightDifference "Hwajun" "Kiyoon" 839 →
  HeightDifference "Hwajun" "Soohyun" 1046 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_difference_soohyun_hwajun_l418_41839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l418_41811

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a^2 / x

noncomputable def g (x : ℝ) : ℝ := x + Real.log x

theorem problem_statement :
  (∀ x ∈ Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioo 0 2, 
    (deriv (f 2)) x < 0) ∧
  (∀ a : ℝ, a ≥ (Real.exp 1 + 1) / 2 ↔ 
    ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 (Real.exp 1) → x₂ ∈ Set.Icc 1 (Real.exp 1) → f a x₁ ≥ g x₂) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l418_41811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_reduction_theorem_l418_41896

/-- Calculates the percentage reduction in oil price given the conditions --/
noncomputable def calculate_price_reduction (additional_kg : ℝ) (total_cost : ℝ) (reduced_price : ℝ) : ℝ :=
  let original_price := total_cost / (total_cost / reduced_price - additional_kg)
  let price_difference := original_price - reduced_price
  (price_difference / original_price) * 100

/-- Theorem stating that the percentage reduction in oil price is approximately 20.09% --/
theorem oil_price_reduction_theorem :
  let additional_kg : ℝ := 4
  let total_cost : ℝ := 684
  let reduced_price : ℝ := 34.2
  let result := calculate_price_reduction additional_kg total_cost reduced_price
  abs (result - 20.09) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_reduction_theorem_l418_41896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l418_41840

/-- The function f(x) = x^2 - 2x + 2 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

/-- Theorem stating the relationship among f(1), f(√3), and f(-1) -/
theorem f_inequality : f 1 < f (Real.sqrt 3) ∧ f (Real.sqrt 3) < f (-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l418_41840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_one_sixth_l418_41857

noncomputable def g (x y : ℝ) : ℝ :=
  if x + y ≤ 4 then
    (x*y - x + 3) / (3*x)
  else
    (x*y - y - 3) / (-3*y)

theorem g_sum_equals_one_sixth :
  g 3 1 + g 3 2 = 1/6 := by
  -- Unfold the definition of g
  unfold g
  -- Simplify the if-then-else expressions
  simp [if_pos (show 3 + 1 ≤ 4 by norm_num), if_neg (show ¬(3 + 2 ≤ 4) by norm_num)]
  -- Perform arithmetic calculations
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_one_sixth_l418_41857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_incircle_radii_ratio_l418_41810

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point on a line segment
def pointOnSegment (A B : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
  (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Define the incircle radius of a triangle
noncomputable def incircleRadius (A B C : ℝ × ℝ) : ℝ :=
  let a := distance B C
  let b := distance A C
  let c := distance A B
  let s := (a + b + c) / 2
  (s - a) * (s - b) * (s - c) / s

-- Theorem statement
theorem equal_incircle_radii_ratio (ABC : Triangle) (N : ℝ × ℝ) (t : ℝ) :
  distance ABC.A ABC.B = 6 →
  distance ABC.B ABC.C = 8 →
  distance ABC.A ABC.C = 10 →
  N = pointOnSegment ABC.A ABC.C t →
  incircleRadius ABC.A ABC.B N = incircleRadius ABC.C ABC.B N →
  ∃ (p q : ℕ), Nat.Coprime p q ∧ 
    distance ABC.A N / distance N ABC.C = p / q :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_incircle_radii_ratio_l418_41810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_distance_relation_l418_41805

/-- Given a center of inversion O, power of inversion R², and points A, B with their inverses A*, B* -/
theorem inversion_distance_relation 
  (O : EuclideanSpace ℝ (Fin 2))  -- Center of inversion
  (R : ℝ)  -- R² is the power of inversion
  (A B A_star B_star : EuclideanSpace ℝ (Fin 2))  -- Points and their inverses
  (h1 : dist O A * dist O A_star = R^2)  -- Inversion property for A
  (h2 : dist O B * dist O B_star = R^2)  -- Inversion property for B
  : dist A_star B_star = (dist A B * R^2) / (dist O A * dist O B) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_distance_relation_l418_41805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_distance_calculation_l418_41831

theorem walking_distance_calculation (slow_speed fast_speed additional_distance : ℝ) 
  (h1 : slow_speed = 8)
  (h2 : fast_speed = 12)
  (h3 : additional_distance = 20) : 
  (fast_speed * additional_distance) / (fast_speed - slow_speed) = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_distance_calculation_l418_41831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_l418_41856

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define a line passing through two points and the focus
def line_through_focus (A B : PointOnParabola) : Prop :=
  ∃ t : ℝ, (1 - t) * A.x + t * B.x = focus.1 ∧ (1 - t) * A.y + t * B.y = focus.2

-- Define the distance between two points
noncomputable def distance (A B : PointOnParabola) : ℝ :=
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

-- The main theorem
theorem parabola_intersection_distance (A B : PointOnParabola) :
  line_through_focus A B →
  A.x + B.x = 6 →
  distance A B = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_l418_41856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_five_l418_41862

theorem cube_root_sum_equals_five :
  let x := (7 + 2 * Real.sqrt 19) ^ (1/3) + (7 - 2 * Real.sqrt 19) ^ (1/3)
  x + 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_five_l418_41862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l418_41806

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x + a else x + 4/x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∃ (m : ℝ), ∀ (x : ℝ), f a x ≥ m) →
  a ∈ Set.Ici 4 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l418_41806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_quadratic_unique_monotonic_range_characterization_inequality_solution_characterization_l418_41859

/-- A quadratic function satisfying specific conditions -/
structure SpecialQuadratic where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  symmetry : ∀ x, f (-2 - x) = f (-2 + x)
  y_intercept : f 0 = 1
  x_intercept_length : ∃ x₁ x₂, f x₁ = 0 ∧ f x₂ = 0 ∧ |x₂ - x₁| = 2 * Real.sqrt 2

/-- The special quadratic function is uniquely determined -/
theorem special_quadratic_unique (sq : SpecialQuadratic) :
  ∀ x, sq.f x = 1/2 * x^2 + 2 * x + 1 := by
  sorry

/-- The range of 'a' for which f(x) - ax is monotonic on [2,3] -/
def monotonic_range (sq : SpecialQuadratic) : Set ℝ :=
  { a | ∀ x₁ x₂, 2 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 3 → (sq.f x₁ - a * x₁ < sq.f x₂ - a * x₂) ∨
                                              (sq.f x₁ - a * x₁ > sq.f x₂ - a * x₂) }

/-- The monotonic range is (-∞, 4] ∪ [5, +∞) -/
theorem monotonic_range_characterization (sq : SpecialQuadratic) :
  monotonic_range sq = Set.Iic 4 ∪ Set.Ici 5 := by
  sorry

/-- The solution set of the inequality 2f(x) - (a+5)x - 2 + a > 0 -/
def inequality_solution_set (sq : SpecialQuadratic) (a : ℝ) : Set ℝ :=
  { x | 2 * sq.f x - (a + 5) * x - 2 + a > 0 }

/-- Characterization of the inequality solution set based on the value of a -/
theorem inequality_solution_characterization (sq : SpecialQuadratic) :
  ∀ a, inequality_solution_set sq a =
    if a < 1 then { x | x < a ∨ x > 1 }
    else if a = 1 then { x | x ≠ 1 }
    else { x | x < 1 ∨ x > a } := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_quadratic_unique_monotonic_range_characterization_inequality_solution_characterization_l418_41859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l418_41815

noncomputable def f (x y : ℝ) : ℝ := (x * y) / (x^2 + y^2)

theorem max_value_of_f :
  ∃ (x y : ℝ),
    1/3 ≤ x ∧ x ≤ 2/5 ∧
    1/4 ≤ y ∧ y ≤ 5/12 ∧
    f x y = 20/41 ∧
    ∀ (x' y' : ℝ),
      1/3 ≤ x' ∧ x' ≤ 2/5 →
      1/4 ≤ y' ∧ y' ≤ 5/12 →
      f x' y' ≤ 20/41 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l418_41815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_geometric_sum_l418_41850

/-- Sum of a geometric series with n terms, first term a, and common ratio r -/
noncomputable def geometricSum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (r^n - 1) / (r - 1)

/-- The sum of the specific geometric series is 510 -/
theorem specific_geometric_sum :
  geometricSum 2 2 8 = 510 := by
  -- Expand the definition of geometricSum
  unfold geometricSum
  -- Simplify the expression
  simp [pow_succ]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_geometric_sum_l418_41850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_m_value_l418_41895

theorem linear_equation_m_value (m : ℤ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (m - 3 : ℝ) * x^(2 * abs m - 5) - 4 * m = a * x + b) → 
  (m - 3 ≠ 0) → 
  m = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_m_value_l418_41895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_roll_ellipse_eccentricity_probability_dice_roll_ellipse_eccentricity_probability_proof_l418_41828

/-- The probability of rolling two dice and obtaining scores that result in an ellipse
    with eccentricity greater than or equal to √3/2 is 1/4. -/
theorem dice_roll_ellipse_eccentricity_probability : Real → Prop :=
  λ p => ∀ (a b : ℕ) (e : Real),
    (1 ≤ a ∧ a ≤ 6) →
    (1 ≤ b ∧ b ≤ 6) →
    (∀ (x y : Real), y ^ 2 / (a ^ 2 : Real) + x ^ 2 / (b ^ 2 : Real) = 1 →
      e ^ 2 = 1 - (b ^ 2 : Real) / (a ^ 2 : Real)) →
    (e ≥ Real.sqrt 3 / 2) →
    p = 1 / 4

/-- The proof of the theorem. -/
theorem dice_roll_ellipse_eccentricity_probability_proof :
  dice_roll_ellipse_eccentricity_probability (1 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_roll_ellipse_eccentricity_probability_dice_roll_ellipse_eccentricity_probability_proof_l418_41828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l418_41875

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 - m*x + 1 > 0

def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 < 9 - m^2

-- Define the set of m values
def m_set : Set ℝ := Set.Icc (-3) (-2) ∪ Set.Ico 2 3

-- State the theorem
theorem m_range (m : ℝ) : (¬p m ∧ q m) → m ∈ m_set := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l418_41875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l418_41889

noncomputable def Hyperbola (a b : ℝ) := {p : ℝ × ℝ | (p.1 / a)^2 - (p.2 / b)^2 = 1}

noncomputable def Circle (a : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = a^2}

noncomputable def LeftFocus (a b : ℝ) : ℝ × ℝ := (-Real.sqrt (a^2 + b^2), 0)

def Origin : ℝ × ℝ := (0, 0)

noncomputable def Eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + (b / a)^2)

theorem hyperbola_eccentricity 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (F : ℝ × ℝ) 
  (hF : F = LeftFocus a b) 
  (E : ℝ × ℝ) 
  (hE : E ∈ Circle a) 
  (P : ℝ × ℝ) 
  (hP : P ∈ Hyperbola a b) 
  (hTangent : ∃ (t : ℝ), E = F + t • (E - F) ∧ P = F + (1 + t) • (E - F)) 
  (hMidpoint : E = (F + P) / 2) :
  Eccentricity a b = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l418_41889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_solution_replacement_l418_41892

/-- Calculates the percentage of sugar in a solution after partial replacement -/
noncomputable def sugar_percentage_after_replacement (initial_percentage : ℝ) 
                                       (replacement_percentage : ℝ) 
                                       (replaced_fraction : ℝ) : ℝ :=
  (1 - replaced_fraction) * initial_percentage + 
  replaced_fraction * replacement_percentage

/-- Theorem stating that replacing one fourth of a 10% sugar solution 
    with a 34% sugar solution results in a 16% sugar solution -/
theorem sugar_solution_replacement :
  sugar_percentage_after_replacement 10 34 (1/4) = 16 := by
  unfold sugar_percentage_after_replacement
  -- The proof steps would go here, but we'll use sorry for now
  sorry

-- Using norm_num to evaluate the expression
example : sugar_percentage_after_replacement 10 34 (1/4) = 16 := by
  unfold sugar_percentage_after_replacement
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_solution_replacement_l418_41892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_noah_average_speed_l418_41869

/-- A number is a palindrome if it reads the same forwards and backwards -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- The next highest palindrome after a given number -/
def nextPalindrome (n : ℕ) : ℕ := sorry

/-- Calculate average speed given distance and time -/
def averageSpeed (distance : ℕ) (time : ℕ) : ℚ :=
  (distance : ℚ) / (time : ℚ)

/-- Approximate equality for rational numbers -/
def approx_eq (x y : ℚ) (ε : ℚ) : Prop :=
  abs (x - y) < ε

theorem noah_average_speed :
  let initial_reading := 12321
  let final_reading := nextPalindrome initial_reading
  let time := 3
  isPalindrome initial_reading ∧
  isPalindrome final_reading ∧
  final_reading > initial_reading ∧
  approx_eq (averageSpeed (final_reading - initial_reading) time) (33333 / 1000) (1 / 1000) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_noah_average_speed_l418_41869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_of_equation_l418_41867

theorem two_roots_of_equation (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  ∃ (x₁ x₂ : ℝ), x₁ ∈ Set.Ioo c b ∧ x₂ ∈ Set.Ioo b a ∧
  (∀ x : ℝ, (1 / (x - a) + 1 / (x - b) + 1 / (x - c) = 0) ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_of_equation_l418_41867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_value_l418_41833

theorem sin_plus_cos_value (α : ℝ) (h1 : Real.sin (2 * α) = 24 / 25) (h2 : α ∈ Set.Ioo π (3 * π / 2)) :
  Real.sin α + Real.cos α = - 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_value_l418_41833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_implies_tan_g_symmetric_implies_phi_l418_41858

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
noncomputable def vector_b : ℝ × ℝ := (1, Real.sqrt 3)

def parallel (u v : ℝ × ℝ) : Prop := ∃ (k : ℝ), u.1 * v.2 = k * u.2 * v.1

noncomputable def f (x : ℝ) : ℝ := (vector_a x).1 * vector_b.1 + (vector_a x).2 * vector_b.2

noncomputable def g (x φ : ℝ) : ℝ := f (x / 2 + φ)

def symmetric_about_y_axis (h : ℝ → ℝ) : Prop := ∀ x, h x = h (-x)

theorem vector_parallel_implies_tan (x : ℝ) :
  parallel (vector_a x) vector_b → Real.tan x = Real.sqrt 3 / 3 := by sorry

theorem g_symmetric_implies_phi (φ : ℝ) :
  0 < φ → φ < π → symmetric_about_y_axis (g · φ) → φ = π / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_implies_tan_g_symmetric_implies_phi_l418_41858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_y_l418_41865

noncomputable def y : ℕ → ℝ
  | 0 => 0  -- Add a case for 0 to cover all natural numbers
  | 1 => Real.rpow 5 (1/3)
  | 2 => Real.rpow (Real.rpow 5 (1/3)) (Real.rpow 5 (1/3))
  | n + 3 => Real.rpow (y (n + 2)) (Real.rpow 5 (1/3))

theorem smallest_integer_y : (∀ n < 4, ¬ Int.floor (y n) = y n) ∧ Int.floor (y 4) = y 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_y_l418_41865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_price_increase_l418_41838

theorem double_price_increase (P : ℝ) (h : P > 0) : 
  P * (1 + 0.06)^2 = P * (1 + 0.1236) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_price_increase_l418_41838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flipping_game_properties_l418_41822

/-- Represents the coin-flipping game between Grisha and Vanya -/
structure CoinFlippingGame where
  grishaWinCondition : ℕ → Bool → Bool
  vanyaWinCondition : ℕ → Bool → Bool

/-- The specific game rules as described in the problem -/
def gameRules : CoinFlippingGame :=
  { grishaWinCondition := fun n result => n % 2 = 0 ∧ result = true
  , vanyaWinCondition := fun n result => n % 2 = 1 ∧ result = false
  }

/-- The probability of Grisha winning -/
def grishaWinProbability : ℚ := 1/3

/-- The expected number of coin flips until the outcome is decided -/
def expectedFlips : ℚ := 2

/-- Theorem stating the probability of Grisha winning and the expected number of flips -/
theorem coin_flipping_game_properties :
  grishaWinProbability = 1/3 ∧ expectedFlips = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flipping_game_properties_l418_41822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_july_milk_powder_cost_theorem_l418_41851

/-- The cost per pound of milk powder, coffee, and sugar in June -/
def june_cost : ℝ := sorry

/-- The cost of 4 lbs of mixture in July -/
def july_mixture_cost : ℝ := 11.70

/-- The percentage of milk powder in the mixture -/
def milk_powder_percent : ℝ := 0.30

/-- The percentage of coffee in the mixture -/
def coffee_percent : ℝ := 0.35

/-- The percentage of sugar in the mixture -/
def sugar_percent : ℝ := 0.35

/-- The price change factor for milk powder in July -/
def milk_powder_change : ℝ := 0.20

/-- The price change factor for coffee in July -/
def coffee_change : ℝ := 4.00

/-- The price change factor for sugar in July -/
def sugar_change : ℝ := 1.45

/-- The cost of a pound of milk powder in July -/
def july_milk_powder_cost : ℝ := june_cost * milk_powder_change

theorem july_milk_powder_cost_theorem : 
  ∃ (ε : ℝ), abs (july_milk_powder_cost - 0.2974) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_july_milk_powder_cost_theorem_l418_41851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_planes_l418_41820

/-- The distance between two parallel planes -/
noncomputable def plane_distance (a b c d₁ d₂ : ℝ) : ℝ :=
  |d₂ - d₁| / Real.sqrt (a^2 + b^2 + c^2)

/-- Theorem: The distance between the planes 2x - 4y + 4z = 10 and 4x - 8y + 8z = 18 is 1/6 -/
theorem distance_between_planes :
  plane_distance 2 (-4) 4 10 9 = 1/6 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_planes_l418_41820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_time_percentage_is_80_l418_41800

/-- The number of trips Laura took to the park -/
def num_trips : ℕ := 6

/-- The time spent in the park during each trip (in hours) -/
noncomputable def park_time : ℝ := 2

/-- The time spent walking to and from the park during each trip (in hours) -/
noncomputable def walk_time : ℝ := 0.5

/-- The total time spent on all trips (in hours) -/
noncomputable def total_time : ℝ := num_trips * (park_time + walk_time)

/-- The total time spent in the park for all trips (in hours) -/
noncomputable def total_park_time : ℝ := num_trips * park_time

/-- The percentage of time spent in the park out of the total time -/
noncomputable def park_time_percentage : ℝ := (total_park_time / total_time) * 100

theorem park_time_percentage_is_80 : park_time_percentage = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_time_percentage_is_80_l418_41800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_value_l418_41854

theorem sin_2x_value (x y : ℝ) 
  (h1 : Real.sin y = (3/2) * Real.sin x + (2/3) * Real.cos x)
  (h2 : Real.cos y = (2/3) * Real.sin x + (3/2) * Real.cos x) :
  Real.sin (2 * x) = -61/72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_value_l418_41854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l418_41899

open Set
open Real

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := x / (x^2 - 2*x + 2)

-- State the theorem about the range of g
theorem range_of_g :
  range g = Icc 0 (1/2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l418_41899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_for_odd_g_l418_41830

noncomputable def f (x : ℝ) := Real.cos (2 * x + Real.pi / 3)

noncomputable def g (φ : ℝ) (x : ℝ) := Real.cos (2 * (x + φ) + Real.pi / 3)

theorem min_phi_for_odd_g :
  ∀ φ : ℝ, φ > 0 →
  (∀ x : ℝ, g φ (-x) = -(g φ x)) →
  ∀ ψ : ℝ, ψ > 0 → φ ≤ ψ →
  φ = Real.pi / 12 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_for_odd_g_l418_41830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sides_exist_l418_41814

/-- A convex quadrilateral with positive integer side lengths -/
structure ConvexQuadrilateral where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  convex : True  -- We assume convexity without explicitly defining it

/-- The property that the sum of any three sides is divisible by the fourth -/
def divisible_sum_property (q : ConvexQuadrilateral) : Prop :=
  (q.b.val + q.c.val + q.d.val) % q.a.val = 0 ∧
  (q.a.val + q.c.val + q.d.val) % q.b.val = 0 ∧
  (q.a.val + q.b.val + q.d.val) % q.c.val = 0 ∧
  (q.a.val + q.b.val + q.c.val) % q.d.val = 0

/-- The theorem stating that if a convex quadrilateral with positive integer side lengths
    satisfies the divisible sum property, then at least two sides have the same length -/
theorem equal_sides_exist (q : ConvexQuadrilateral) 
  (h : divisible_sum_property q) : 
  q.a = q.b ∨ q.a = q.c ∨ q.a = q.d ∨ q.b = q.c ∨ q.b = q.d ∨ q.c = q.d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sides_exist_l418_41814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_range_l418_41890

theorem inequality_solution_range (a : ℝ) : 
  (∃! (s : Finset ℕ), s.card = 3 ∧ ∀ x ∈ s, 0 < x ∧ 4 * x + a ≤ 5) →
  -11 < a ∧ a ≤ -7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_range_l418_41890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_properties_l418_41897

/-- Triangle with weighted vertices -/
structure WeightedTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  mass_A : ℝ
  mass_B : ℝ
  mass_C : ℝ

/-- Point that divides a line segment in a given ratio -/
noncomputable def divideSegment (P Q : ℝ × ℝ) (r : ℝ) : ℝ × ℝ :=
  ((r * P.1 + Q.1) / (r + 1), (r * P.2 + Q.2) / (r + 1))

/-- Centroid of a weighted triangle -/
noncomputable def centroid (t : WeightedTriangle) : ℝ × ℝ :=
  let S1 := divideSegment t.B t.C (t.mass_C / t.mass_B)
  let S3 := divideSegment t.A t.B (t.mass_B / t.mass_A)
  divideSegment t.A S1 ((t.mass_B + t.mass_C) / t.mass_A)

/-- Theorem stating the properties of the centroid -/
theorem centroid_properties (t : WeightedTriangle) 
  (h1 : t.mass_A = 1) (h2 : t.mass_B = 2) (h3 : t.mass_C = 6) :
  let S := centroid t
  let S1 := divideSegment t.B t.C 3
  let S3 := divideSegment t.A t.B 2
  (S1 = divideSegment t.B t.C 3) ∧
  (S3 = divideSegment t.A t.B 2) ∧
  (S = divideSegment t.A S1 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_properties_l418_41897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_on_a_plus_b_l418_41861

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
variable (a b : V)

theorem projection_a_on_a_plus_b
  (ha : ‖a‖ = 1)
  (hb : ‖b‖ = 1)
  (hab : ‖a + b‖ = Real.sqrt 2 * ‖a - b‖) :
  (inner a (a + b) / ‖a + b‖ : ℝ) = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_on_a_plus_b_l418_41861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peter_avg_time_is_40_l418_41813

/-- Represents a chess match between two players -/
structure ChessMatch where
  duration_minutes : ℕ
  total_moves : ℕ
  polly_avg_time : ℕ

/-- Calculates Peter's average move time given a chess match -/
def peter_avg_move_time (m : ChessMatch) : ℕ :=
  let total_seconds := m.duration_minutes * 60
  let polly_total_time := (m.total_moves / 2) * m.polly_avg_time
  let peter_total_time := total_seconds - polly_total_time
  peter_total_time / (m.total_moves / 2)

/-- Theorem stating that Peter's average move time is 40 seconds -/
theorem peter_avg_time_is_40 (m : ChessMatch) 
  (h1 : m.duration_minutes = 17)
  (h2 : m.total_moves = 30)
  (h3 : m.polly_avg_time = 28) :
  peter_avg_move_time m = 40 := by
  sorry

#eval peter_avg_move_time { duration_minutes := 17, total_moves := 30, polly_avg_time := 28 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_peter_avg_time_is_40_l418_41813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_theorem_l418_41821

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-5, 0)
def F₂ : ℝ × ℝ := (5, 0)

-- Define the angle between PF₁ and PF₂
noncomputable def angle_PF₁F₂ (P : ℝ × ℝ) : ℝ := 60 * Real.pi / 180

-- Define the area of triangle F₁PF₂
noncomputable def area_F₁PF₂ (P : ℝ × ℝ) : ℝ := 
  let d₁ := Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2)
  let d₂ := Real.sqrt ((P.2 - F₂.1)^2 + (P.2 - F₂.2)^2)
  1/2 * d₁ * d₂ * Real.sin (angle_PF₁F₂ P)

-- Theorem statement
theorem area_theorem (P : ℝ × ℝ) : 
  hyperbola P.1 P.2 → area_F₁PF₂ P = 9 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_theorem_l418_41821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_mul_properties_l418_41871

/-- Custom multiplication operation -/
noncomputable def custom_mul (k : ℝ) (x y : ℝ) : ℝ := (k * x * y) / (x + y)

theorem custom_mul_properties (k : ℝ) :
  (∀ (x y : ℝ), x > 0 → y > 0 → custom_mul k x y = custom_mul k y x) ∧
  (∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ custom_mul k (custom_mul k x y) z ≠ custom_mul k x (custom_mul k y z)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_mul_properties_l418_41871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_in_range_l418_41852

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x + 2) * Real.exp x

-- State the theorem
theorem inequality_holds_iff_a_in_range :
  ∀ a : ℝ, (∀ x : ℝ, x ≥ 0 → (f x - Real.exp x) / (a * x + 1) ≥ 1) ↔ (0 ≤ a ∧ a ≤ 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_in_range_l418_41852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_g_subset_implies_a_in_closed_interval_l418_41878

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a x - |x - 2|

-- Define the range of g
def range_g (a : ℝ) : Set ℝ := {y | ∃ x, g a x = y}

-- Theorem statement
theorem range_g_subset_implies_a_in_closed_interval :
  ∀ a : ℝ, range_g a ⊆ Set.Icc (-1) 3 → a ∈ Set.Icc 1 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_g_subset_implies_a_in_closed_interval_l418_41878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_funnel_height_l418_41863

/-- The volume of a right circular cone -/
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

/-- Theorem stating that the height of the funnel rounded to the nearest integer is 3 -/
theorem funnel_height :
  let r : ℝ := 4
  let v : ℝ := 48
  let h : ℝ := v / ((1/3) * Real.pi * r^2)
  round_to_nearest h = 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_funnel_height_l418_41863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l418_41809

open Real

-- Define the functions f and g
noncomputable def f (x : ℝ) := log x
noncomputable def g (x : ℝ) := (1/2) * x^2 - 2*x

-- Define the inequality function
def ineq (k : ℝ) (x : ℝ) := k * (x - 2) < x * f x + 2 * (x - 2) + 3

-- State the theorem
theorem max_k_value : 
  (∃ k : ℤ, ∀ x > 2, ineq (↑k) x) ∧ 
  (∀ k : ℤ, (∀ x > 2, ineq (↑k) x) → k ≤ 5) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l418_41809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_increase_l418_41803

theorem circle_area_increase (a : ℝ) : 
  π * (3 + a)^2 - π * 3^2 = π * (3 + a)^2 - 9 * π := by
  ring

#check circle_area_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_increase_l418_41803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l418_41837

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * Real.cos (Real.pi / 2 * x - Real.pi / 8)

-- State the theorem
theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l418_41837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l418_41868

theorem platform_length
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (time_to_pass : ℝ)
  (h1 : train_length = 360)
  (h2 : train_speed_kmh = 45)
  (h3 : time_to_pass = 40.8) :
  let train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
  let total_distance : ℝ := train_speed_ms * time_to_pass
  let platform_length : ℝ := total_distance - train_length
  platform_length = 150 :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l418_41868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_l418_41877

-- Define the tetrahedron PQRS
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ
  altitude : ℝ

-- Helper function to calculate triangle area
noncomputable def area_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Helper function to calculate tetrahedron volume
noncomputable def volume (t : Tetrahedron) : ℝ :=
  (1 / 3) * area_triangle t.QR t.QS t.RS * t.altitude

-- Define the theorem
theorem tetrahedron_volume 
  (t : Tetrahedron)
  (h_PQ : t.PQ = 6)
  (h_PR : t.PR = 3 * Real.sqrt 2)
  (h_PS : t.PS = Real.sqrt 18)
  (h_QR : t.QR = 3)
  (h_QS : t.QS = 3 * Real.sqrt 3)
  (h_RS : t.RS = 2 * Real.sqrt 6)
  (h_altitude : t.altitude = Real.sqrt 3) :
  ∃ (area_QRS : ℝ), 
    volume t = (Real.sqrt 3 * area_QRS) / 3 ∧ 
    area_QRS = area_triangle t.QR t.QS t.RS :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_l418_41877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_derivative_at_one_l418_41824

theorem function_derivative_at_one (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2*x*(deriv f 1)) :
  deriv f 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_derivative_at_one_l418_41824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_factorial_9_multiple_of_3_l418_41874

-- Define 9!
def factorial_9 : ℕ := 9*8*7*6*5*4*3*2*1

-- Define a function to check if a number is a divisor of 9! and a multiple of 3
def is_valid_divisor (n : ℕ) : Bool :=
  factorial_9 % n = 0 && n % 3 = 0

-- Theorem statement
theorem divisors_of_factorial_9_multiple_of_3 :
  (Finset.filter (fun n => is_valid_divisor n) (Finset.range (factorial_9 + 1))).card = 128 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_factorial_9_multiple_of_3_l418_41874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l418_41835

/-- Represents a geometric sequence -/
structure GeometricSequence where
  firstTerm : ℝ
  commonRatio : ℝ

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def sumOfFirstNTerms (g : GeometricSequence) (n : ℕ) : ℝ :=
  g.firstTerm * (1 - g.commonRatio^n) / (1 - g.commonRatio)

/-- Theorem statement for the geometric sequence problem -/
theorem geometric_sequence_sum (g : GeometricSequence) :
  sumOfFirstNTerms g 1500 = 300 →
  sumOfFirstNTerms g 3000 = 570 →
  sumOfFirstNTerms g 4500 = 813 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l418_41835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_square_dimension_l418_41876

-- Define the rectangle dimensions
def rectangle_width : ℝ := 9
def rectangle_height : ℝ := 16

-- Define the square formed by repositioning the hexagons
noncomputable def square_from_hexagons (w h : ℝ) : ℝ := Real.sqrt (w * h)

-- Define y as one-third of the square's side length
noncomputable def y (s : ℝ) : ℝ := s / 3

-- Theorem statement
theorem hexagon_square_dimension :
  y (square_from_hexagons rectangle_width rectangle_height) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_square_dimension_l418_41876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l418_41801

theorem expression_value (a b c k : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hk : k ≠ 0) (hsum : a + b + c = 0) :
  (k = 1 ∧ (k*a^2*b^2 + a^2*c^2 + b^2*c^2) / ((a^2 - b*c)*(b^2 - a*c) + (a^2 - b*c)*(c^2 - a*b) + (b^2 - a*c)*(c^2 - a*b)) = 1) ∨
  (k ≠ 1 ∧ (k*a^2*b^2 + a^2*c^2 + b^2*c^2) / ((a^2 - b*c)*(b^2 - a*c) + (a^2 - b*c)*(c^2 - a*b) + (b^2 - a*c)*(c^2 - a*b)) = 0/0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l418_41801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l418_41879

theorem problem_statement (x y : ℝ) 
  (h1 : Real.tan x = 1/7)
  (h2 : Real.sin y = Real.sqrt 10 / 10)
  (h3 : 0 < x ∧ x < π/2)
  (h4 : 0 < y ∧ y < π/2) :
  x + 2*y = π/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l418_41879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_neg_necessary_not_sufficient_for_hyperbola_l418_41817

/-- A conic section represented by the equation ax² + by² = c -/
structure ConicSection (a b c : ℝ) where
  equation : ∀ x y : ℝ, a * x^2 + b * y^2 = c

/-- Predicate to check if a conic section is a hyperbola -/
def IsHyperbola (a b c : ℝ) (conic : ConicSection a b c) : Prop :=
  ∃ k m : ℝ, k ≠ 0 ∧ m ≠ 0 ∧ k * m < 0 ∧ 
    ∀ x y : ℝ, (x^2 / k) - (y^2 / m) = 1 ↔ a * x^2 + b * y^2 = c

/-- The main theorem stating that ab < 0 is necessary but not sufficient for a hyperbola -/
theorem ab_neg_necessary_not_sufficient_for_hyperbola :
  (∀ a b c : ℝ, ∀ conic : ConicSection a b c, IsHyperbola a b c conic → a * b < 0) ∧
  ¬(∀ a b : ℝ, a * b < 0 → ∀ c : ℝ, ∀ conic : ConicSection a b c, IsHyperbola a b c conic) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_neg_necessary_not_sufficient_for_hyperbola_l418_41817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_range_l418_41886

noncomputable section

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1/4, 0)

-- Define the line passing through the focus
def line_through_focus (θ : ℝ) (A : ℝ × ℝ) : Prop :=
  A.2 - focus.2 = Real.tan θ * (A.1 - focus.1)

-- Define the intersection of the line and parabola
def intersection (θ : ℝ) (A : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ line_through_focus θ A

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_intersection_distance_range :
  ∀ θ A,
  θ ≥ π/4 →
  A.2 > 0 →
  intersection θ A →
  distance focus A ∈ Set.Ioo (1/4 : ℝ) (1 + Real.sqrt 2 / 2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_range_l418_41886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l418_41823

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def is_right_angled (t : Triangle) : Prop :=
  (distance t.A t.B)^2 + (distance t.A t.C)^2 = (distance t.B t.C)^2

def is_square (A B C D : ℝ × ℝ) : Prop :=
  distance A B = distance B C ∧
  distance B C = distance C D ∧
  distance C D = distance D A ∧
  (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0

noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

theorem triangle_area_theorem (t : Triangle) (D E : ℝ × ℝ) :
  is_right_angled t →
  distance t.A t.B = 3 →
  distance t.A t.C = 4 →
  distance t.B t.C = 5 →
  is_square t.B t.C D E →
  area_triangle t.A t.B E = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l418_41823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l418_41860

-- Define the points
noncomputable def A : ℝ × ℝ := (0, 8)
noncomputable def B : ℝ × ℝ := (0, 0)
noncomputable def C : ℝ × ℝ := (10, 0)

-- Define midpoints
noncomputable def G : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
noncomputable def H : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)

-- Define the intersection point I
noncomputable def I : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem intersection_point_sum : I.1 + I.2 = 0 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l418_41860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_c_l418_41883

theorem triangle_cosine_c (A B C : ℝ) (h1 : A + B + C = Real.pi) 
    (h2 : 0 < A ∧ A < Real.pi) (h3 : 0 < B ∧ B < Real.pi) (h4 : 0 < C ∧ C < Real.pi) 
    (h5 : Real.sin A = 4/5) (h6 : Real.cos B = 12/13) :
  Real.cos C = -16/65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_c_l418_41883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_triangle_property_l418_41847

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + Real.sin (2 * x - Real.pi / 6)

def monotonic_increasing_intervals (k : ℤ) : Set ℝ :=
  Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6)

def max_value : ℝ := 2

def max_value_set : Set ℝ :=
  {x | ∃ k : ℤ, x = k * Real.pi + Real.pi / 6}

theorem f_properties :
  (∀ k : ℤ, StrictMono (fun x => f x)) ∧
  (∀ x : ℝ, f x ≤ max_value) ∧
  (∀ x : ℝ, x ∈ max_value_set ↔ f x = max_value) := by
  sorry

theorem triangle_property (A B C : ℝ) (a b c : ℝ) :
  f A = 3/2 →
  b + c = 2 →
  a ≥ 1 ∧ a < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_triangle_property_l418_41847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_l418_41885

theorem village_population (population_percentage : ℝ) (partial_population : ℕ) (total_population : ℕ) :
  population_percentage = 0.80 →
  partial_population = 32000 →
  (population_percentage * (total_population : ℝ) = (partial_population : ℝ)) →
  total_population = 40000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_l418_41885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l418_41819

-- Define the function f on the interval [-4, 4]
variable (f : ℝ → ℝ)
variable (h : ∀ x ∈ Set.Icc (-4) 4, DifferentiableAt ℝ f x)

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := deriv f x

-- State the conditions
axiom cond1 : ∀ x ∈ Set.Icc (-4) 4, f' f x > f x
axiom cond2 : ∀ x ∈ Set.Icc (-4) 4, Real.exp (x - 1) * f (1 + x) - f (2 * x) < 0

-- Define the set of x that satisfies the conditions
def S (f : ℝ → ℝ) : Set ℝ := {x | x ∈ Set.Icc (-4) 4 ∧ Real.exp (x - 1) * f (1 + x) - f (2 * x) < 0}

-- State the theorem
theorem range_of_x (f : ℝ → ℝ) (h : ∀ x ∈ Set.Icc (-4) 4, DifferentiableAt ℝ f x) :
  S f = Set.Ioc 1 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l418_41819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l418_41816

/-- The distance from a point to a line in 2D space -/
noncomputable def distance_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  (|A * x0 + B * y0 + C|) / Real.sqrt (A^2 + B^2)

/-- The standard equation of a circle -/
def circle_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

theorem circle_tangent_to_line :
  let center_x : ℝ := 0
  let center_y : ℝ := 1
  let line_A : ℝ := 1
  let line_B : ℝ := -1
  let line_C : ℝ := -1
  let radius : ℝ := distance_point_to_line center_x center_y line_A line_B line_C
  ∀ x y : ℝ,
    circle_equation x y center_x center_y radius ↔ x^2 + (y - 1)^2 = 2 :=
by
  intro x y
  simp [circle_equation, distance_point_to_line]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l418_41816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l418_41870

/-- Given a triangle with internal angles of 30° and 60°, if the side opposite to the 30° angle
    measures 2, then the length of the side opposite to the 60° angle is 2√3. -/
theorem triangle_side_length (a b c : ℝ) (α β γ : ℝ) : 
  α = 30 * Real.pi / 180 →  -- 30° in radians
  β = 60 * Real.pi / 180 →  -- 60° in radians
  γ = 90 * Real.pi / 180 →  -- The third angle (90°) in radians
  a = 2 →             -- Side opposite to 30° angle
  b = a * (Real.sin β / Real.sin α) →  -- Law of Sines
  b = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l418_41870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_theorem_l418_41872

/-- Given a curve y = x³ + ax + b and its tangent line y = kx + 1 at the point (1, 3),
    prove that 2a + b = 1 -/
theorem tangent_line_theorem (a b k : ℝ) : 
  (∀ x, x^3 + a*x + b = k*x + 1 → x = 1) →  -- Tangent condition
  3 = 1^3 + a + b →                         -- Curve passes through (1, 3)
  3 = k + 1 →                              -- Tangent line passes through (1, 3)
  2*a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_theorem_l418_41872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_result_l418_41829

/-- The polynomial f(x) = 3x^4 + 9x^3 - 7x^2 + 2x + 6 -/
noncomputable def f : Polynomial ℝ := 3 * Polynomial.X^4 + 9 * Polynomial.X^3 - 7 * Polynomial.X^2 + 2 * Polynomial.X + 6

/-- The polynomial d(x) = x^2 + 2x - 3 -/
noncomputable def d : Polynomial ℝ := Polynomial.X^2 + 2 * Polynomial.X - 3

theorem polynomial_division_result :
  ∃ (q r : Polynomial ℝ), f = q * d + r ∧ r.degree < d.degree → q.eval 1 + r.eval (-1) = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_result_l418_41829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_integers_count_l418_41841

theorem congruent_integers_count : 
  (Finset.filter (fun n : ℕ => n > 0 ∧ n < 2000 ∧ n % 13 = 7) (Finset.range 2000)).card = 154 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_integers_count_l418_41841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_eq_2Q_sqrt2_l418_41845

/-- A right parallelepiped with a rhombus base -/
structure RightParallelepiped where
  /-- Side length of the rhombus base -/
  base_side : ℝ
  /-- Height of the parallelepiped -/
  height : ℝ

/-- The area of the cross-section formed by a plane passing through one side of the lower base
    and the opposite side of the upper base, forming a 45° angle with the base plane -/
noncomputable def cross_section_area (p : RightParallelepiped) : ℝ :=
  p.base_side * p.height * Real.sqrt 2

/-- The lateral surface area of the parallelepiped -/
def lateral_surface_area (p : RightParallelepiped) : ℝ :=
  4 * p.base_side * p.height

theorem lateral_surface_area_eq_2Q_sqrt2 (p : RightParallelepiped) (Q : ℝ) 
    (h : cross_section_area p = Q) : 
    lateral_surface_area p = 2 * Q * Real.sqrt 2 := by
  sorry

#check lateral_surface_area_eq_2Q_sqrt2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_eq_2Q_sqrt2_l418_41845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rowing_round_trip_time_l418_41842

/-- Proves that the total time to row to a destination and back is 1 hour, given the specified conditions. -/
theorem rowing_round_trip_time
  (rowing_speed : ℝ)
  (current_speed : ℝ)
  (distance : ℝ)
  (h1 : rowing_speed = 5)
  (h2 : current_speed = 1)
  (h3 : distance = 2.4) :
  (distance / (rowing_speed + current_speed)) + (distance / (rowing_speed - current_speed)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rowing_round_trip_time_l418_41842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l418_41843

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if one of its asymptotes is parallel to the line 3x - y + 1 = 0,
    then its eccentricity is √10. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : ∃ (k : ℝ), b / a = 3 ∨ b / a = -3) : 
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l418_41843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_W_equals_three_l418_41893

-- Define the digits as natural numbers
def T : ℕ := 7
def W : ℕ := sorry
def O : ℕ := sorry
def F : ℕ := sorry
def U : ℕ := sorry
def R : ℕ := sorry

-- Define the conditions
axiom different_digits : T ≠ W ∧ T ≠ O ∧ T ≠ F ∧ T ≠ U ∧ T ≠ R ∧
                         W ≠ O ∧ W ≠ F ∧ W ≠ U ∧ W ≠ R ∧
                         O ≠ F ∧ O ≠ U ∧ O ≠ R ∧
                         F ≠ U ∧ F ≠ R ∧
                         U ≠ R

axiom O_is_even : ∃ k : ℕ, O = 2 * k

axiom addition_equation : 100 * T + 10 * W + O + 100 * T + 10 * W + O = 1000 * F + 100 * O + 10 * U + R

-- The theorem to prove
theorem W_equals_three : W = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_W_equals_three_l418_41893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_l418_41848

open Real

-- Define the triangle ABC
variable (A B C M N : EuclideanSpace ℝ (Fin 2))

-- Define the properties of the triangle
def is_triangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define median BM
def is_median (B M C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define angle bisector AN
def is_angle_bisector (A N B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define BM is half the length of AN
def median_half_bisector (B M A N : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define angle CBM is three times angle CAN
def angle_relation (C B M A N : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the angle function
noncomputable def angle (P Q R : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- Define the theorem
theorem triangle_angles 
  (h1 : is_triangle A B C)
  (h2 : is_median B M C)
  (h3 : is_angle_bisector A N B C)
  (h4 : median_half_bisector B M A N)
  (h5 : angle_relation C B M A N) :
  angle A B C = 108 * π / 180 ∧ angle B C A = 36 * π / 180 ∧ angle C A B = 36 * π / 180 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_l418_41848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_condition_l418_41855

noncomputable def reflection_matrix (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![a, b],
    ![-1/2, Real.sqrt 3 / 2]]

theorem reflection_condition (a b : ℝ) :
  (reflection_matrix a b) ^ 2 = 1 ↔ a = -(Real.sqrt 3 / 2) ∧ b = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_condition_l418_41855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_sum_of_acute_angles_l418_41832

theorem tan_half_sum_of_acute_angles (α β : Real) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : Real.tan α = 2) 
  (h4 : Real.tan β = 3) : 
  Real.tan ((α + β) / 2) = 1 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_sum_of_acute_angles_l418_41832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_range_and_max_a_l418_41812

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.cos x - (x - Real.pi / 2) * Real.sin x

noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := -(1 + a) * Real.sin x - (x - Real.pi / 2) * Real.cos x

theorem f_derivative_range_and_max_a :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f_prime (-1) x ∈ Set.Icc 0 (Real.pi / 2)) ∧
  (∀ a > -1, ∃ x ∈ Set.Icc 0 (Real.pi / 2), f a x > 0) ∧
  (∀ a ≤ -1, ∀ x ∈ Set.Icc 0 (Real.pi / 2), f a x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_range_and_max_a_l418_41812
