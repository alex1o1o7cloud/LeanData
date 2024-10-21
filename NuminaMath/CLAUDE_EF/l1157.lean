import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_equality_l1157_115764

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a * x * Real.log x + b

-- Define the function g
noncomputable def g (a b x : ℝ) : ℝ := (f a b x + 1) / x

-- State the theorem
theorem tangent_line_and_equality (a b : ℝ) :
  (∀ x, (deriv (f a b)) 1 * (x - 1) + f a b 1 = x - 1) →
  (a = 1 ∧ b = 0) ∧
  (∀ x₁ x₂, 0 < x₁ → 0 < x₂ → x₁ < x₂ → g a b x₁ = g a b x₂ → x₁ + x₂ > 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_equality_l1157_115764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l1157_115756

open Real

theorem sine_function_properties (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c ≠ 0)
  (h4 : ∀ x, a * Real.sin (b * x - c) ≤ 3)
  (h5 : ∀ x, a * Real.sin (b * x - c) ≥ -3)
  (h6 : ∀ x, a * Real.sin (b * x - c) = a * Real.sin (b * (x + 4 * π) - c))
  (h7 : a * Real.sin (b * (π / 2) - c) = 0) :
  a = 3 ∧ b = 1 / 2 ∧ c = π / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l1157_115756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l1157_115735

/-- The circle with center (5, 3) and radius 2 -/
def myCircle (x y : ℝ) : Prop := (x - 5)^2 + (y - 3)^2 = 4

/-- The line 4x + 3y - 4 = 0 -/
def myLine (x y : ℝ) : Prop := 4*x + 3*y - 4 = 0

/-- The minimum distance from a point on the circle to the line is 3 -/
theorem min_distance_circle_to_line :
  ∀ (M : ℝ × ℝ), myCircle M.1 M.2 →
  (∃ (P : ℝ × ℝ), myLine P.1 P.2 ∧
    ∀ (Q : ℝ × ℝ), myLine Q.1 Q.2 →
      Real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2) ≤ Real.sqrt ((M.1 - Q.1)^2 + (M.2 - Q.2)^2)) →
  ∃ (P : ℝ × ℝ), myLine P.1 P.2 ∧
    Real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2) = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l1157_115735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_l1157_115705

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r * t)

def angela_deposit : ℝ := 8000
def bob_deposit : ℝ := 12000
def angela_rate : ℝ := 0.06
def bob_rate : ℝ := 0.08
def time_period : ℝ := 15
def compounding_frequency : ℝ := 2

theorem interest_difference :
  ‖simple_interest bob_deposit bob_rate time_period - 
   compound_interest angela_deposit angela_rate compounding_frequency time_period‖ = 6982 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_l1157_115705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_m_values_l1157_115742

theorem parallel_vectors_m_values (m : ℝ) :
  let a : ℝ × ℝ := (2 * m + 1, 3)
  let b : ℝ × ℝ := (2, m)
  (∃ (k : ℝ), a = k • b) → (m = 3/2 ∨ m = -2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_m_values_l1157_115742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_exponential_l1157_115778

noncomputable def g (x : ℝ) : ℝ := 2^x

theorem inverse_exponential (f : ℝ → ℝ) (h : Function.RightInverse f g) : f 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_exponential_l1157_115778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_range_ABCD_l1157_115785

/-- The line l: y = √3x + 4 -/
def line_l (x y : ℝ) : Prop := y = Real.sqrt 3 * x + 4

/-- The circle O: x² + y² = r², where 1 < r < 2 -/
def circle_O (x y r : ℝ) : Prop := x^2 + y^2 = r^2 ∧ 1 < r ∧ r < 2

/-- Rhombus ABCD with one interior angle of 60° -/
def rhombus_ABCD (A B C D : ℝ × ℝ) : Prop :=
  ∃ (angle : ℝ), angle = 60 * Real.pi / 180 ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - D.1)^2 + (C.2 - D.2)^2 ∧
  (C.1 - D.1)^2 + (C.2 - D.2)^2 = (D.1 - A.1)^2 + (D.2 - A.2)^2 ∧
  Real.cos angle = ((A.1 - B.1) * (B.1 - C.1) + (A.2 - B.2) * (B.2 - C.2)) /
    (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) * Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2))

/-- Vertices A and B are on line l -/
def vertices_on_line_l (A B : ℝ × ℝ) : Prop :=
  line_l A.1 A.2 ∧ line_l B.1 B.2

/-- Vertices C and D are on circle O -/
def vertices_on_circle_O (C D : ℝ × ℝ) (r : ℝ) : Prop :=
  circle_O C.1 C.2 r ∧ circle_O D.1 D.2 r

/-- The area of rhombus ABCD -/
noncomputable def area_ABCD (A B C D : ℝ × ℝ) : ℝ :=
  let d1 := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let d2 := Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2)
  d1 * d2 / 2

/-- The theorem stating the range of possible areas for rhombus ABCD -/
theorem area_range_ABCD (A B C D : ℝ × ℝ) (r : ℝ) :
  rhombus_ABCD A B C D →
  vertices_on_line_l A B →
  vertices_on_circle_O C D r →
  (0 < area_ABCD A B C D ∧ area_ABCD A B C D < (3 * Real.sqrt 3) / 2) ∨
  ((3 * Real.sqrt 3) / 2 < area_ABCD A B C D ∧ area_ABCD A B C D < 6 * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_range_ABCD_l1157_115785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triples_eq_12_l1157_115781

/-- The number of ordered triples of nonnegative integers (a, b, c) that satisfy the equation (a * b + 1) * (b * c + 1) * (c * a + 1) = 84 -/
def count_triples : Nat :=
  Finset.card (Finset.filter (fun t : Nat × Nat × Nat =>
    let (a, b, c) := t
    (a * b + 1) * (b * c + 1) * (c * a + 1) = 84)
    (Finset.product (Finset.range 84) (Finset.product (Finset.range 84) (Finset.range 84))))

theorem count_triples_eq_12 : count_triples = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triples_eq_12_l1157_115781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_group_men_count_l1157_115750

/-- The amount of work that can be completed in one day -/
noncomputable def total_work : ℝ := 1

/-- The amount of work a boy can do in one day -/
noncomputable def boy_work : ℝ := total_work / (5 * (12 * 2 + 16))

/-- The amount of work a man can do in one day -/
noncomputable def man_work : ℝ := 2 * boy_work

/-- The number of men in the second group -/
def second_group_men : ℕ := 13

/-- Theorem stating that the second group can complete the work in 4 days -/
theorem second_group_men_count :
  4 * (second_group_men * man_work + 24 * boy_work) = total_work :=
by sorry

#check second_group_men_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_group_men_count_l1157_115750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_integer_triangle_l1157_115771

/-- A triangle with integer side lengths and perimeter 7 -/
structure IntegerTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  perimeter_eq : a + b + c = 7
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

/-- The area of a triangle given its side lengths -/
noncomputable def triangle_area (t : IntegerTriangle) : ℝ :=
  let s := (t.a + t.b + t.c : ℝ) / 2
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

/-- The maximum area of an integer triangle with perimeter 7 -/
theorem max_area_integer_triangle :
  ∃ (max_area : ℝ), max_area = 3 * Real.sqrt 7 / 4 ∧
  ∀ (t : IntegerTriangle), triangle_area t ≤ max_area := by
  sorry

#check max_area_integer_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_integer_triangle_l1157_115771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annie_solo_time_l1157_115784

/-- The time it takes Annie to complete the job alone -/
noncomputable def annie_time : ℝ := 3.0000000000000004

/-- Dan's rate of completing the job (fraction per hour) -/
noncomputable def dan_rate : ℝ := 1 / 12

/-- The time Dan works on the job -/
noncomputable def dan_work_time : ℝ := 8

/-- The time it takes Annie to complete the remaining work after Dan stops -/
noncomputable def annie_completion_time : ℝ := 3.0000000000000004

/-- Theorem stating that Annie's time to complete the job alone is 3.0000000000000004 hours -/
theorem annie_solo_time :
  annie_time = (1 - dan_rate * dan_work_time) / (1 / annie_completion_time) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_annie_solo_time_l1157_115784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_square_theorem_l1157_115738

/-- A function that varies inversely as the square of its input -/
noncomputable def inverse_square (k : ℝ) (y : ℝ) : ℝ := k / (y ^ 2)

/-- Theorem stating that if an inverse square function equals 2.25 when y = 2, 
    then it equals 1 when y = 3 -/
theorem inverse_square_theorem (f : ℝ → ℝ) (k : ℝ) 
    (h1 : ∀ y, f y = inverse_square k y)
    (h2 : f 2 = 2.25) : 
    f 3 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_square_theorem_l1157_115738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_block_size_of_three_elevenths_l1157_115744

theorem repeating_block_size_of_three_elevenths :
  ∃ (n : ℕ) (s : String),
    n = 2 ∧
    s.length = n ∧
    (∃ (q : ℚ), (3 : ℚ) / 11 = q ∧ 
      ∃ (k : ℕ), k > 0 ∧ (q * 10^k - q).num = 0) ∧
    (∀ (m : ℕ) (t : String),
      m < n →
      (∃ (r : ℚ), (3 : ℚ) / 11 = r ∧ 
        ∀ (k : ℕ), k > 0 → (r * 10^k - r).num ≠ 0) →
      t.length ≠ m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_block_size_of_three_elevenths_l1157_115744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wz_length_l1157_115789

/-- Right triangle XYZ with Y as the right angle -/
structure RightTriangle where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  is_right_angle : (Y.1 - X.1) * (Z.1 - X.1) + (Y.2 - X.2) * (Z.2 - X.2) = 0

/-- Circle with diameter YZ intersecting XZ at W -/
def circle_intersect (t : RightTriangle) (W : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), 0 < k ∧ k < 1 ∧ W = (k * t.X.1 + (1 - k) * t.Z.1, k * t.X.2 + (1 - k) * t.Z.2)

/-- Theorem: If XW = 3, YW = 9, then WZ = 27 -/
theorem wz_length (t : RightTriangle) (W : ℝ × ℝ) 
    (h_circle : circle_intersect t W)
    (h_xw : Real.sqrt ((W.1 - t.X.1)^2 + (W.2 - t.X.2)^2) = 3)
    (h_yw : Real.sqrt ((W.1 - t.Y.1)^2 + (W.2 - t.Y.2)^2) = 9) :
    Real.sqrt ((W.1 - t.Z.1)^2 + (W.2 - t.Z.2)^2) = 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wz_length_l1157_115789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_for_triangle_l1157_115743

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to check if a line passes through a point -/
def passesThrough (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Function to calculate the area of a triangle formed by a line and the coordinate axes -/
noncomputable def triangleArea (l : Line) : ℝ :=
  abs (l.c * l.c / (2 * l.a * l.b))

/-- Function to check if a line forms a triangle in the second quadrant -/
def formsTriangleInSecondQuadrant (l : Line) : Prop :=
  l.a > 0 ∧ l.b < 0 ∧ l.c < 0

theorem unique_line_for_triangle :
  ∃! l : Line,
    passesThrough l ⟨-2, 2⟩ ∧
    formsTriangleInSecondQuadrant l ∧
    triangleArea l = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_for_triangle_l1157_115743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_l1157_115700

/-- The area of an ellipse defined by the equation x^2 - 2x + 9y^2 + 18y + 16 = 0 is 4π/3 -/
theorem ellipse_area (x y : ℝ) : 
  (x^2 - 2*x + 9*y^2 + 18*y + 16 = 0) → 
  (∃ A : ℝ, A = (4 * Real.pi) / 3 ∧ A = Real.pi * 2 * (2/3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_l1157_115700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_statements_true_l1157_115726

-- Sequence 1
def seq1 (n : ℕ+) : ℚ := 1 / (n * (n + 2))

-- Sequence 2
noncomputable def seq2 (n : ℕ) : ℝ := Real.sqrt (3 * n - 1)

-- Sequence 3
def seq3 (k : ℚ) (n : ℕ) : ℚ := k * n - 5

-- Sequence 4
def seq4 (a₁ : ℝ) (n : ℕ) : ℝ := a₁ + 3 * (n - 1)

theorem all_statements_true :
  (seq1 10 = 1 / 120) ∧
  (∀ n : ℕ+, n > 1 → seq1 1 > seq1 n) ∧
  (seq2 1 = Real.sqrt 2 ∧ seq2 2 = Real.sqrt 5 ∧ seq2 3 = 2 * Real.sqrt 2 ∧ seq2 4 = Real.sqrt 11) ∧
  (∃ k : ℚ, seq3 k 8 = 11 ∧ seq3 k 17 = 29) ∧
  (∀ a₁ : ℝ, ∀ n : ℕ, n > 0 → seq4 a₁ (n + 1) > seq4 a₁ n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_statements_true_l1157_115726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_is_four_l1157_115702

/-- A function that satisfies the given condition -/
def SatisfyingFunction (f : ℤ → ℕ) : Prop :=
  ∀ x y : ℤ, |x - y| ∈ ({5, 7, 12} : Set ℤ) → f x ≠ f y

/-- The existence of a function with range {1, ..., k} that satisfies the condition -/
def ExistsFunction (k : ℕ) : Prop :=
  ∃ f : ℤ → Fin k, SatisfyingFunction (fun x => (f x).val + 1)

/-- The main theorem: the minimum k is 4 -/
theorem min_k_is_four :
  (∃ k : ℕ, ExistsFunction k ∧ ∀ m : ℕ, m < k → ¬ExistsFunction m) ∧
  (∀ k : ℕ, ExistsFunction k → k ≥ 4) := by
  sorry

#check min_k_is_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_is_four_l1157_115702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorems_l1157_115762

/-- Triangle properties and theorems --/
theorem triangle_theorems 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π)
  (h_opposite : a = Real.sin A / Real.sin C ∧ b = Real.sin B / Real.sin C ∧ c = Real.sin C / Real.sin C)
  (h_relation : 2 * b = c + 2 * a * Real.cos C)
  (h_cosB : Real.cos B = Real.sqrt 3 / 3)
  (h_area : 1/2 * b * c * Real.sin A = 10 * Real.sqrt 3 / 3)
  (h_side_a : a = 3) : 
  A = π/3 ∧ 
  Real.sin (2*B - A) = (2*Real.sqrt 2 + Real.sqrt 3)/6 ∧
  a + b + c = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorems_l1157_115762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_period_l1157_115752

/-- Given a function f(x) = sin(ωx + π/6) where ω > 0, 
    if the distance between two adjacent zero points of f(x) is π/6, 
    then ω = 6 -/
theorem sine_function_period (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∀ x : ℝ, ∃ y : ℝ, y > x ∧ 
    Real.sin (ω * x + Real.pi / 6) = 0 ∧ 
    Real.sin (ω * y + Real.pi / 6) = 0 ∧ 
    y - x = Real.pi / 6) : 
  ω = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_period_l1157_115752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l1157_115711

theorem sum_remainder (a b c d : ℕ) 
  (ha : a % 53 = 17)
  (hb : b % 53 = 34)
  (hc : c % 53 = 6)
  (hd : d % 53 = 3) :
  (a + b + c + d) % 53 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l1157_115711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1157_115745

/-- The set of digits to be used -/
def digits : Finset Nat := {0, 1, 2, 3}

/-- A function that checks if a number is a valid three-digit number according to our conditions -/
def isValidNumber (n : Nat) : Bool :=
  100 ≤ n ∧ n < 1000 ∧ 
  (n / 100) ∈ digits ∧
  ((n / 10) % 10) ∈ digits ∧
  (n % 10) ∈ digits ∧
  (n / 100) ≠ ((n / 10) % 10) ∧
  (n / 100) ≠ (n % 10) ∧
  ((n / 10) % 10) ≠ (n % 10)

/-- The set of all valid three-digit numbers according to our conditions -/
def validNumbers : Finset Nat :=
  Finset.filter (fun n => isValidNumber n) (Finset.range 1000)

/-- The main theorem: there are exactly 18 valid three-digit numbers -/
theorem count_valid_numbers : Finset.card validNumbers = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1157_115745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_circumscribed_circle_radius_l1157_115739

theorem sector_circumscribed_circle_radius
  (θ : Real) (h_obtuse : π / 2 < θ ∧ θ < π) :
  let R := 8 / Real.cos (θ / 2)
  R = 8 / Real.cos (θ / 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_circumscribed_circle_radius_l1157_115739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_kite_sum_l1157_115791

/-- Given two parabolas that intersect the coordinate axes in four points forming a kite -/
structure ParabolaKite where
  a : ℝ
  b : ℝ
  parabola1 : ℝ → ℝ := fun x ↦ a * x^2 + 3
  parabola2 : ℝ → ℝ := fun x ↦ 5 - b * x^2
  intersect_axes : ∃ (x1 x2 y1 y2 : ℝ), 
    x1 ≠ x2 ∧ y1 ≠ y2 ∧
    parabola1 x1 = 0 ∧ parabola1 x2 = 0 ∧
    parabola2 0 = y1 ∧ parabola1 0 = y2
  form_kite : ∃ (x1 x2 y1 y2 : ℝ),
    x1 ≠ x2 ∧ y1 ≠ y2 ∧
    parabola1 x1 = 0 ∧ parabola1 x2 = 0 ∧
    parabola2 0 = y1 ∧ parabola1 0 = y2 ∧
    -- Define isKite as a property instead of a function
    (abs (x1 - x2) * abs (y1 - y2) / 2 = 16)
  kite_area : ∃ (x1 x2 y1 y2 : ℝ),
    x1 ≠ x2 ∧ y1 ≠ y2 ∧
    parabola1 x1 = 0 ∧ parabola1 x2 = 0 ∧
    parabola2 0 = y1 ∧ parabola1 0 = y2 ∧
    -- Define area calculation directly
    (abs (x1 - x2) * abs (y1 - y2) / 2 = 16)

/-- The sum of a and b is 1/8 -/
theorem parabola_kite_sum (pk : ParabolaKite) : pk.a + pk.b = 1/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_kite_sum_l1157_115791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_range_l1157_115737

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * (x - 1)

theorem tangent_line_and_range :
  (∃ (y : ℝ), x - y + Real.log 2 - 3/2 = 0 ↔ 
    (∃ (y : ℝ), y = f (1/2) 2 ∧ 
      (y - f (1/2) 2) / (x - 2) = deriv (f (1/2)) 2)) ∧
  (∀ a : ℝ, (∀ x > 1, f a x > 0) ↔ a ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_range_l1157_115737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_polar_form_l1157_115767

noncomputable def cis (θ : ℝ) : ℂ := Complex.exp (θ * Complex.I)

theorem complex_product_polar_form :
  let z1 : ℂ := 4 * cis (π/4)
  let z2 : ℂ := -3 * cis (-π/6)
  let product : ℂ := z1 * z2
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2*π ∧ product = r * cis θ ∧ r = 12 ∧ θ = 13*π/12 :=
by
  sorry

#check complex_product_polar_form

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_polar_form_l1157_115767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segments_form_triangle_l1157_115732

-- Define the space and points
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
variable (A B C K L M N F G : E)

-- Define the parallelograms on the sides of the triangle
def is_parallelogram (P Q R S : E) : Prop :=
  (R - P) = (S - Q) ∧ (Q - P) = (S - R)

-- Define the segments
def segment_length (P Q : E) : ℝ := ‖P - Q‖

-- Theorem statement
theorem segments_form_triangle
  (h1 : is_parallelogram A B K L)
  (h2 : is_parallelogram B C M N)
  (h3 : is_parallelogram A C F G) :
  segment_length K N + segment_length M F > segment_length G L ∧
  segment_length M F + segment_length G L > segment_length K N ∧
  segment_length G L + segment_length K N > segment_length M F :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segments_form_triangle_l1157_115732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_score_is_94_l1157_115782

/-- Represents an operation that can be applied to a number -/
inductive Operation
  | Add : Operation
  | Square : Operation

/-- Applies an operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.Add => n + 1
  | Operation.Square => n * n

/-- Calculates the minimum distance from a number to any perfect square -/
noncomputable def minDistanceToPerfectSquare (n : ℕ) : ℕ :=
  sorry

/-- Represents a sequence of 100 operations -/
def OperationSequence := Fin 100 → Operation

/-- Applies a sequence of operations to the starting number 0 -/
def applySequence (seq : OperationSequence) : ℕ :=
  (List.range 100).foldl (fun acc _ => applyOperation acc (seq ⟨acc, sorry⟩)) 0

/-- The maximum score attainable is 94 -/
theorem max_score_is_94 :
  ∃ (seq : OperationSequence),
    ∀ (other_seq : OperationSequence),
      minDistanceToPerfectSquare (applySequence seq) ≥
      minDistanceToPerfectSquare (applySequence other_seq) ∧
      minDistanceToPerfectSquare (applySequence seq) = 94 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_score_is_94_l1157_115782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_range_l1157_115761

-- Define a regular triangular pyramid
structure RegularTriangularPyramid where
  -- We don't need to specify the properties of the pyramid,
  -- as we're only concerned with the dihedral angle

-- Define the dihedral angle between adjacent lateral faces
def dihedralAngle (p : RegularTriangularPyramid) : ℝ := sorry

-- Theorem statement
theorem dihedral_angle_range (p : RegularTriangularPyramid) :
  60 * Real.pi / 180 < dihedralAngle p ∧ dihedralAngle p < Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_range_l1157_115761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l1157_115776

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l1157_115776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_exists_l1157_115760

theorem no_such_function_exists : 
  ¬∃ g : ℝ → ℝ, ∀ x y : ℝ, g (g x + y) = g (x + y) + (x^2 + x) * g y - x * y - 2 * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_exists_l1157_115760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coordinates_of_neg3_plus_3i_l1157_115792

noncomputable def complex_to_polar (z : ℂ) : ℝ × ℝ :=
  (Complex.abs z, Complex.arg z)

theorem polar_coordinates_of_neg3_plus_3i :
  let z : ℂ := -3 + 3*Complex.I
  let (r, θ) := complex_to_polar z
  r = 3 * Real.sqrt 2 ∧ θ = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coordinates_of_neg3_plus_3i_l1157_115792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_pi_over_four_l1157_115736

/-- The inclination angle of a line with equation ax + by + c = 0 -/
noncomputable def inclinationAngle (a b : ℝ) : ℝ := Real.arctan (- a / b)

/-- Proof that the inclination angle of the line x - y + 1 = 0 is π/4 -/
theorem line_inclination_angle_pi_over_four :
  inclinationAngle 1 (-1) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_pi_over_four_l1157_115736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_pyramid_base_side_l1157_115709

/-- Represents a right pyramid with a square base -/
structure RightPyramid where
  base_side : ℝ
  slant_height : ℝ

/-- Calculates the area of one lateral face of the pyramid -/
noncomputable def lateral_face_area (p : RightPyramid) : ℝ :=
  (1 / 2) * p.base_side * p.slant_height

theorem right_pyramid_base_side 
  (p : RightPyramid) 
  (h1 : lateral_face_area p = 200)
  (h2 : p.slant_height = 40) : 
  p.base_side = 10 := by
  sorry

#check right_pyramid_base_side

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_pyramid_base_side_l1157_115709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1157_115758

-- Define the triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) (m : Real) : Prop :=
  Real.sin t.A + Real.sin t.C = m * Real.sin t.B ∧ t.a * t.c = 1/4 * t.b^2

-- Part I
theorem part_one (t : Triangle) :
  triangle_conditions t (5/4) → t.b = 1 →
  ((t.a = 1 ∧ t.c = 1/4) ∨ (t.a = 1/4 ∧ t.c = 1)) := by
  sorry

-- Part II
theorem part_two (t : Triangle) (m : Real) :
  triangle_conditions t m → 0 < t.B ∧ t.B < Real.pi/2 →
  Real.sqrt 6 / 2 < m ∧ m < Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1157_115758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l1157_115797

noncomputable section

def f : ℝ → ℝ := sorry

axiom f_domain (x : ℝ) : x ≠ 0 → f x ≠ 0

axiom f_equation (x : ℝ) (hx : x ≠ 0) : f x - 2 * f (1/x) = 3 * x

theorem f_is_odd : ∀ x : ℝ, x ≠ 0 → f (-x) = -f x := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l1157_115797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l1157_115731

/-- Regular hexagon with side length 2 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 2)

/-- Circle in the context of the problem -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- Configuration of two circles in the hexagon -/
structure CircleConfiguration (hex : RegularHexagon) :=
  (circle1 : Circle)
  (circle2 : Circle)
  (tangent_to_AB : circle1.center.2 = hex.side_length / 2)
  (tangent_to_DE : circle2.center.2 = hex.side_length * 3 / 2)
  (tangent_to_BC_FA : circle1.center.1 = circle2.center.1)
  (externally_tangent : (circle1.center.1 - circle2.center.1)^2 + (circle1.center.2 - circle2.center.2)^2 = (circle1.radius + circle2.radius)^2)

/-- The theorem to be proved -/
theorem circle_area_ratio (hex : RegularHexagon) (config : CircleConfiguration hex) : 
  (config.circle2.radius^2) / (config.circle1.radius^2) = 3 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l1157_115731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_sqrt_5_l1157_115713

/-- Two parallel lines in 2D space -/
structure ParallelLines where
  a : ℝ
  l1 : ℝ → ℝ → Prop
  l2 : ℝ → ℝ → Prop
  l1_eq : ∀ x y, l1 x y ↔ 2*x + y + 3 = 0
  l2_eq : ∀ x y, l2 x y ↔ x - a*y - 1 = 0
  parallel : ∀ (x1 y1 x2 y2 : ℝ), l1 x1 y1 → l2 x2 y2 → (y1 - y2) = -2 * (x1 - x2)

/-- The distance between two parallel lines -/
noncomputable def distance_between_lines (pl : ParallelLines) : ℝ :=
  Real.sqrt 5

/-- Theorem: The distance between the given parallel lines is √5 -/
theorem distance_is_sqrt_5 (pl : ParallelLines) : 
  distance_between_lines pl = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_sqrt_5_l1157_115713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1157_115749

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 6)

theorem f_properties : 
  (f (5 * Real.pi / 12) = 0) ∧ 
  (∀ t : ℝ, f t = Real.sin (2 * t + Real.pi / 6)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1157_115749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transportation_problem_solvable_l1157_115766

/-- Represents a transportation plan for getting 8 people to the train station -/
structure TransportPlan where
  time : ℝ  -- Time taken in hours

/-- The problem setup and constraints -/
def problem_setup : Prop :=
  ∃ (plan1 plan2 : TransportPlan),
    -- Both plans take less than 42 minutes
    plan1.time < 42/60 ∧ plan2.time < 42/60 ∧
    -- Plan 1: Car takes first group, returns for second group
    plan1.time = 35/52 ∧
    -- Plan 2: Car takes both groups partially, everyone walks
    plan2.time = 37/60 ∧
    -- Constants
    let distance_to_station : ℝ := 15  -- km
    let car_speed : ℝ := 60            -- km/h
    let walking_speed : ℝ := 5         -- km/h
    -- Additional constraints can be added here if needed
    True

/-- Theorem stating that a solution exists for the transportation problem -/
theorem transportation_problem_solvable : problem_setup := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transportation_problem_solvable_l1157_115766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1157_115759

-- Problem 1
theorem problem_1 : Real.sqrt 9 + Real.sqrt (5^2) + ((-27) ^ (1/3 : ℝ)) = 5 := by sorry

-- Problem 2
theorem problem_2 : (-3)^2 - |(-1/2)| - Real.sqrt 9 = 11/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1157_115759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_product_l1157_115710

/-- Represents a repeating decimal with a single-digit repeat -/
def SingleDigitRepeat (whole : ℕ) (rep : ℕ) : ℚ :=
  whole + rep / (10 - 1)

/-- Represents a repeating decimal with a two-digit repeat -/
def TwoDigitRepeat (whole : ℕ) (rep : ℕ) : ℚ :=
  whole + rep / (100 - 1)

theorem repeating_decimal_product :
  (TwoDigitRepeat 0 12) * (SingleDigitRepeat 0 3) = 4 / 99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_product_l1157_115710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2alpha_plus_pi_12_l1157_115763

theorem tan_2alpha_plus_pi_12 (α : ℝ) (h1 : 0 < α ∧ α < Real.pi / 2) 
  (h2 : Real.cos (α + Real.pi / 6) = Real.sqrt 5 / 5) : 
  Real.tan (2 * α + Real.pi / 12) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2alpha_plus_pi_12_l1157_115763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_diminished_percentage_l1157_115779

-- Define the original tax, consumption, and revenue
variable (T : ℝ) -- Original tax
variable (C : ℝ) -- Original consumption

-- Define the percentage decrease in tax
variable (X : ℝ)

-- Define the conditions
noncomputable def new_tax (T X : ℝ) : ℝ := T * (1 - X / 100)
noncomputable def new_consumption (C : ℝ) : ℝ := C * 1.1
noncomputable def new_revenue (T C X : ℝ) : ℝ := new_tax T X * new_consumption C
noncomputable def original_revenue (T C : ℝ) : ℝ := T * C

-- State the theorem
theorem tax_diminished_percentage (T C X : ℝ) 
  (h : new_revenue T C X = 0.77 * original_revenue T C) : 
  X = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_diminished_percentage_l1157_115779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_plane_intersection_l1157_115730

/-- Represents a cube with side length a -/
structure Cube (a : ℝ) where
  vertices : Fin 8 → ℝ × ℝ × ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- Returns the midpoint of an edge in the cube -/
def cube_midpoint (c : Cube a) (v1 v2 : Fin 8) : ℝ × ℝ × ℝ :=
  sorry

/-- Checks if a plane is parallel to two lines -/
def is_parallel_to_lines (p : Plane) (l1 l2 : (ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ)) : Prop :=
  sorry

/-- Calculates the ratio in which a plane divides a line segment -/
def division_ratio (p : Plane) (l : (ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ)) : ℝ × ℝ :=
  sorry

/-- Calculates the area of the cross-section created by a plane intersecting a cube -/
noncomputable def cross_section_area (c : Cube a) (p : Plane) : ℝ :=
  sorry

theorem cube_plane_intersection (a : ℝ) (c : Cube a) (p : Plane) 
  (h1 : p.point = cube_midpoint c 0 1)  -- plane passes through midpoint of AB
  (h2 : is_parallel_to_lines p ((c.vertices 1), (c.vertices 7)) ((c.vertices 4), (c.vertices 6))) : 
  (division_ratio p ((c.vertices 3), (c.vertices 5)) = (3, 5)) ∧ 
  (cross_section_area c p = (7 * a^2 * Real.sqrt 6) / 16) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_plane_intersection_l1157_115730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_seventeen_pi_sixths_l1157_115733

theorem sin_seventeen_pi_sixths : Real.sin (17 * π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_seventeen_pi_sixths_l1157_115733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_a_equals_one_range_of_a_l1157_115753

def f (a x : ℝ) : ℝ := |x - a| + |x + 3|

theorem solution_set_for_a_equals_one :
  {x : ℝ | f 1 x ≥ 6} = Set.Iic (-4) ∪ Set.Ici 2 := by sorry

theorem range_of_a :
  {a : ℝ | ∀ x, f a x > -a} = Set.Ioi (-3/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_a_equals_one_range_of_a_l1157_115753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_integer_and_n_not_even_l1157_115769

theorem fraction_sum_integer_and_n_not_even :
  ∃ (n m : ℕ+), 
    (1 / 3 + 1 / 4 + 1 / 9 + 1 / n.val + 1 / m.val : ℚ).num % (1 : ℤ) = 0 ∧ 
    ¬(2 ∣ n.val) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_integer_and_n_not_even_l1157_115769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_set_1_condition_set_2_l1157_115747

-- Define the lines l1 and l2
def l1 (a b x y : ℝ) : Prop := a * x - b * y + 4 = 0
def l2 (a b x y : ℝ) : Prop := (a - 1) * x + y + b = 0

-- Define perpendicularity of lines
def perpendicular (a b : ℝ) : Prop := a * (a - 1) + (-b) * 1 = 0

-- Define parallelism of lines
def parallel (a b : ℝ) : Prop := a + b * (a - 1) = 0

-- Define distance from origin to line ax + by + c = 0
noncomputable def distance_to_origin (a b c : ℝ) : ℝ := |c| / Real.sqrt (a^2 + b^2)

-- Theorem for the first set of conditions
theorem condition_set_1 (a b : ℝ) :
  l1 a b (-3) (-1) ∧ perpendicular a b → a = 2 ∧ b = 2 := by sorry

-- Theorem for the second set of conditions
theorem condition_set_2 (a b : ℝ) :
  parallel a b ∧ distance_to_origin a (-b) 4 = distance_to_origin (a-1) 1 b →
  (a = 2 ∧ b = -2) ∨ (a = -2 ∧ b = 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_set_1_condition_set_2_l1157_115747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_number_in_pascal_row_15_l1157_115774

def pascal_row (n : ℕ) : List ℕ :=
  List.range (n + 1) |>.map (fun k => Nat.choose n k)

theorem fifth_number_in_pascal_row_15 :
  (pascal_row 15).get? 4 = some 1365 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_number_in_pascal_row_15_l1157_115774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l1157_115708

def E : Fin 3 → ℝ := ![2, -5, 1]
def F : Fin 3 → ℝ := ![4, -9, 4]
def G : Fin 3 → ℝ := ![3, -4, -1]
def H : Fin 3 → ℝ := ![5, -8, 2]

def vector_sub (v w : Fin 3 → ℝ) : Fin 3 → ℝ :=
  fun i => v i - w i

def cross_product (v w : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![v 1 * w 2 - v 2 * w 1, v 2 * w 0 - v 0 * w 2, v 0 * w 1 - v 1 * w 0]

noncomputable def magnitude (v : Fin 3 → ℝ) : ℝ :=
  Real.sqrt (v 0^2 + v 1^2 + v 2^2)

theorem parallelogram_area : 
  vector_sub F E = vector_sub H G ∧ 
  magnitude (cross_product (vector_sub F E) (vector_sub G E)) = Real.sqrt 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l1157_115708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thursday_sales_l1157_115716

/-- The number of books sold on Thursday in John's bookshop --/
def books_sold_thursday (initial_stock : ℕ) (mon_sales tue_sales wed_sales fri_sales : ℕ) (unsold_percentage : ℚ) : ℕ :=
  let total_sold : ℚ := (1 - unsold_percentage) * initial_stock
  let other_days_sales : ℕ := mon_sales + tue_sales + wed_sales + fri_sales
  (total_sold - other_days_sales).floor.toNat

/-- Theorem stating the number of books sold on Thursday --/
theorem thursday_sales : books_sold_thursday 1300 75 50 64 135 (69077 / 100000) = 78 := by
  sorry

#eval books_sold_thursday 1300 75 50 64 135 (69077 / 100000)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thursday_sales_l1157_115716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moment_of_inertia_parabolic_arc_l1157_115724

/-- The moment of inertia of a parabolic arc with variable density -/
theorem moment_of_inertia_parabolic_arc (ρ₀ : ℝ) :
  let ρ : ℝ → ℝ := λ x => ρ₀ * Real.sqrt (1 + 4 * x)
  let y : ℝ → ℝ := λ x => x^2
  let I₀ : ℝ := ∫ x in Set.Icc 0 1, ρ x * (x^2 + y x^2) * Real.sqrt (1 + (2 * x)^2) 
  I₀ = 2.2 * ρ₀ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_moment_of_inertia_parabolic_arc_l1157_115724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1157_115768

/-- The original function g(x) -/
noncomputable def g (x : ℝ) : ℝ := Real.sin x + Real.cos x

/-- The function f(x) derived from g(x) by shortening horizontal coordinates -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (2 * x + Real.pi / 4)

/-- The statement that f(x) is derived from g(x) by shortening horizontal coordinates -/
axiom f_derived_from_g : ∀ x, f x = g (2 * x)

/-- The theorem stating that the smallest positive period of f(x) is π -/
theorem smallest_positive_period_of_f : 
  ∃ T : ℝ, T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ S, S > 0 → (∀ x, f (x + S) = f x) → T ≤ S) ∧ T = Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1157_115768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l1157_115757

-- Define an isosceles triangle with side lengths 3 and 5
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  isIsosceles : a = b ∨ a = 5 ∨ b = 5
  hasLengths : (a = 3 ∧ b = 5) ∨ (a = 5 ∧ b = 3)

-- Define the perimeter of the triangle
noncomputable def perimeter (t : IsoscelesTriangle) : ℝ := 
  if t.a = t.b then t.a + t.b + 5 else t.a + t.b + 3

-- Theorem statement
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) : 
  perimeter t = 11 ∨ perimeter t = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l1157_115757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l1157_115780

theorem inequality_theorem (a b : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) (hn : n ≥ 2) :
  let sum := (Finset.range (n+1)).sum (λ i ↦ a^(n-i) * b^i)
  (sum / (n + 1 : ℝ)) ≥ ((a + b) / 2) ^ n ∧
  (sum / (n + 1 : ℝ) = ((a + b) / 2) ^ n ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l1157_115780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_divides_one_plus_a_n_l1157_115772

def sequence_a (p : ℕ) : ℕ → ℤ
  | 0 => 2
  | 1 => 1
  | (n + 2) => sequence_a p (n + 1) + (p^2 - 1) / 4 * sequence_a p n

theorem not_divides_one_plus_a_n (p : ℕ) (h_prime : Nat.Prime p) 
  (h_divides : p ∣ (2^2019 - 1)) :
  ∀ n : ℕ, ¬(p ∣ (Int.toNat (1 + sequence_a p n))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_divides_one_plus_a_n_l1157_115772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l1157_115725

/-- The hyperbola equation -/
def hyperbola (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / m^2 - y^2 / (2*m + 6) = 1

/-- The constraint on m -/
def m_constraint (m : ℝ) : Prop :=
  -2 ≤ m ∧ m < 0

/-- The focal length of the hyperbola -/
noncomputable def focal_length (m : ℝ) : ℝ :=
  2 * Real.sqrt (m^2 + 2*m + 6)

/-- The asymptote equation -/
def asymptote (x y : ℝ) : Prop :=
  y = (1/2) * x ∨ y = -(1/2) * x

/-- The main theorem -/
theorem hyperbola_asymptote :
  ∃ m : ℝ, m_constraint m ∧
  (∀ m' : ℝ, m_constraint m' → focal_length m ≤ focal_length m') ∧
  ∀ x y : ℝ, hyperbola m x y → asymptote x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l1157_115725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scenic_area_max_profit_l1157_115717

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (101/50) * x - b * Real.log (x/10)

-- Define the profit function T
noncomputable def T (a b : ℝ) (x : ℝ) : ℝ := f a b x - x

-- State the theorem
theorem scenic_area_max_profit 
  (a b : ℝ) 
  (h1 : f a b 10 = 19.2) 
  (h2 : f a b 20 = 35.7) 
  (h3 : ∀ x, x ≥ 10 → f a b x = -(x^2/100) + (101/50)*x - Real.log (x/10)) :
  (∃ (x : ℝ), x ≥ 10 ∧ T a b x = 24.4 ∧ ∀ y, y ≥ 10 → T a b y ≤ T a b x) := by
  sorry

#check scenic_area_max_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scenic_area_max_profit_l1157_115717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1157_115734

noncomputable def f (x : ℝ) : ℝ := -3/4 * x^2 + 9/2 * x - 15/4

theorem quadratic_function_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 3 := by
  constructor
  · -- Prove f 1 = 0
    sorry
  constructor
  · -- Prove f 5 = 0
    sorry
  · -- Prove f 3 = 3
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1157_115734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_intervals_a_value_for_max_neg_two_inequality_when_a_neg_one_l1157_115755

noncomputable section

-- Define the function f(x) = ax + ln x
def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x

-- Define the natural logarithm base e
noncomputable def e : ℝ := Real.exp 1

-- Theorem for monotonic intervals
theorem monotonic_intervals (a : ℝ) :
  (a ≥ 0 → StrictMono (f a)) ∧
  (a < 0 → 
    (∀ x y, 0 < x ∧ 0 < y ∧ x < y ∧ y < -1/a → f a x < f a y) ∧
    (∀ x y, -1/a < x ∧ x < y → f a y < f a x)) :=
sorry

-- Theorem for the value of a when maximum is -2
theorem a_value_for_max_neg_two (a : ℝ) :
  a < 0 → (∀ x, 0 < x → x ≤ e → f a x ≤ -2) →
  (∃ x, 0 < x ∧ x ≤ e ∧ f a x = -2) → a = -e :=
sorry

-- Theorem for the inequality when a = -1
theorem inequality_when_a_neg_one (x : ℝ) :
  x > 0 → x * |f (-1) x| > Real.log x + (1/2) * x :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_intervals_a_value_for_max_neg_two_inequality_when_a_neg_one_l1157_115755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_three_heads_one_tail_l1157_115719

/-- The probability of getting exactly three heads and one tail when tossing four fair coins simultaneously -/
theorem prob_three_heads_one_tail : ℝ := by
  let n : ℕ := 4  -- number of coins
  let p : ℝ := 1 / 2  -- probability of getting heads (or tails) for a fair coin
  let k : ℕ := 3  -- number of heads we want
  
  have h : (n.choose k : ℝ) * p^k * (1 - p)^(n - k) = 1 / 4 := by sorry
  
  exact (n.choose k : ℝ) * p^k * (1 - p)^(n - k)


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_three_heads_one_tail_l1157_115719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_to_triangular_ratio_l1157_115727

/-- Represents a square divided into 12 equal angular sections like a clock face -/
structure ClockSquare where
  side : ℝ
  side_positive : 0 < side

/-- The area of one triangular section adjacent to the square's side -/
noncomputable def triangular_area (cs : ClockSquare) : ℝ :=
  (cs.side^2 * Real.sqrt 3) / 8

/-- The area of one corner quadrilateral -/
noncomputable def quadrilateral_area (cs : ClockSquare) : ℝ :=
  (cs.side^2 * (4 - Real.sqrt 3)) / 16

/-- The theorem stating the ratio of quadrilateral area to triangular area -/
theorem quadrilateral_to_triangular_ratio (cs : ClockSquare) :
  quadrilateral_area cs / triangular_area cs = 2 * Real.sqrt 3 - 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_to_triangular_ratio_l1157_115727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_allele_location_determines_f2_ratio_f2_ratio_is_either_1_2_or_1_1_l1157_115748

/-- Represents the genotype of a rice plant -/
inductive Genotype
| BB
| Bb
| bb
| BBb
| Bbb

/-- Represents the phenotype of a rice plant -/
inductive Phenotype
| Fragrant
| NonFragrant

/-- Defines a rice plant with its genotype -/
structure RicePlant where
  genotype : Genotype

/-- Defines a trisomic rice plant -/
structure TrisomicRice extends RicePlant

/-- Function to determine the phenotype based on genotype -/
def phenotype (plant : RicePlant) : Phenotype :=
  match plant.genotype with
  | Genotype.bb => Phenotype.Fragrant
  | _ => Phenotype.NonFragrant

/-- Represents the location of the fragrance allele -/
inductive AlleleLocation
| OnTrisomicChromosome
| NotOnTrisomicChromosome

/-- Function to perform the cross and determine F2 phenotype ratio -/
def f2PhenotypeRatio (alleleLocation : AlleleLocation) : Nat × Nat :=
  match alleleLocation with
  | AlleleLocation.OnTrisomicChromosome => (1, 2)
  | AlleleLocation.NotOnTrisomicChromosome => (1, 1)

/-- Theorem stating the relationship between allele location and F2 phenotype ratio -/
theorem allele_location_determines_f2_ratio 
  (alleleLocation : AlleleLocation) :
  f2PhenotypeRatio alleleLocation = 
    match alleleLocation with
    | AlleleLocation.OnTrisomicChromosome => (1, 2)
    | AlleleLocation.NotOnTrisomicChromosome => (1, 1) := by
  cases alleleLocation
  . rfl
  . rfl

/-- Theorem proving that the F2 phenotype ratio is either 1:2 or 1:1 -/
theorem f2_ratio_is_either_1_2_or_1_1 
  (alleleLocation : AlleleLocation) :
  (f2PhenotypeRatio alleleLocation = (1, 2) ∨ 
   f2PhenotypeRatio alleleLocation = (1, 1)) := by
  cases alleleLocation
  . left; rfl
  . right; rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_allele_location_determines_f2_ratio_f2_ratio_is_either_1_2_or_1_1_l1157_115748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l1157_115770

/-- A parabola with equation y^2 = 2px, where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- A point on the parabola -/
structure ParabolaPoint (parabola : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * parabola.p * x

/-- The focus of the parabola -/
noncomputable def focus (parabola : Parabola) : ℝ × ℝ := (parabola.p / 2, 0)

/-- A line passing through two points -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Check if a line passes through a point -/
def Line.passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  p.2 = l.m * p.1 + l.b

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The theorem to be proved -/
theorem parabola_intersection_theorem (parabola : Parabola) 
  (l : Line) (A B : ParabolaPoint parabola) : 
  l.m = 1 →  -- 45° angle with horizontal axis
  l.passes_through (focus parabola) →
  l.passes_through (A.x, A.y) →
  l.passes_through (B.x, B.y) →
  distance (A.x, A.y) (B.x, B.y) = 8 →
  parabola.p = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l1157_115770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l1157_115721

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₂ - c₁| / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance between the given parallel lines is 1/2 -/
theorem distance_between_given_lines :
  distance_between_parallel_lines 3 4 (-5) (-15/2) = 1/2 := by
  -- Proof steps would go here
  sorry

#check distance_between_given_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l1157_115721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_diagonal_l1157_115707

theorem quadrilateral_diagonal (EF FG GH HE FH : ℕ) :
  EF = 7 →
  FG = 19 →
  GH = 7 →
  HE = 11 →
  FH ∈ ({13, 14, 15, 16, 17} : Set ℕ) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_diagonal_l1157_115707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_sine_value_l1157_115703

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x) * Real.cos φ + Real.cos (2 * x) * Real.sin φ

theorem function_and_sine_value (φ a : ℝ) 
  (h1 : 0 < φ ∧ φ < Real.pi)
  (h2 : f (Real.pi / 4) φ = Real.sqrt 3 / 2)
  (h3 : f (a / 2 - Real.pi / 3) φ = 5 / 13)
  (h4 : Real.pi / 2 < a ∧ a < Real.pi) :
  (∀ x, f x φ = Real.sin (2 * x + Real.pi / 6)) ∧ 
  Real.sin a = 12 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_sine_value_l1157_115703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1157_115777

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * Real.cos x + Real.sqrt 3 * Real.sin (2 * x)

theorem f_properties :
  -- Smallest positive period is π
  (∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧
    (∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S)) ∧
  -- Monotonically decreasing on [π/6 + kπ, 2π/3 + kπ]
  (∀ k : ℤ, ∀ x y : ℝ,
    π/6 + k*π ≤ x ∧ x < y ∧ y ≤ 2*π/3 + k*π → f y < f x) ∧
  -- Maximum value on [-π/6, π/4] is 3 at x = π/6
  (∀ x : ℝ, -π/6 ≤ x ∧ x ≤ π/4 → f x ≤ f (π/6)) ∧
  f (π/6) = 3 ∧
  -- Minimum value on [-π/6, π/4] is 0 at x = -π/6
  (∀ x : ℝ, -π/6 ≤ x ∧ x ≤ π/4 → f (-π/6) ≤ f x) ∧
  f (-π/6) = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1157_115777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l1157_115728

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x + 2 * Real.cos x

-- Define the domain
def domain : Set ℝ := Set.Ioo 0 (2 * Real.pi)

-- Define the decreasing interval
def decreasingInterval : Set ℝ := Set.Ioo (Real.pi / 6) (5 * Real.pi / 6)

-- Theorem statement
theorem f_decreasing_interval :
  ∀ x ∈ domain, (∀ y ∈ domain, x < y → f x > f y) ↔ x ∈ decreasingInterval :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l1157_115728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chebyshev_properties_l1157_115796

-- Define Chebyshev polynomials
noncomputable def T : ℕ → ℝ → ℝ
| 0 => λ _ => 1
| 1 => λ x => x
| (n+2) => λ x => 2 * x * T (n+1) x - T n x

-- State the theorem
theorem chebyshev_properties :
  -- Part (a): Differences between consecutive Chebyshev polynomials
  (∀ x : ℝ, T 2 x - T 1 x = 2*x^2 - x - 1) ∧
  (∀ x : ℝ, T 3 x - T 2 x = 4*x^3 - 2*x^2 - 3*x + 1) ∧
  (∀ x : ℝ, T 4 x - T 3 x = 8*x^4 - 4*x^3 - 8*x^2 + 3*x + 1) ∧
  (∀ x : ℝ, T 5 x - T 4 x = 16*x^5 - 8*x^4 - 20*x^3 + 8*x^2 + 5*x - 1) ∧
  -- Part (b): Roots of 4x² + 2x - 1
  (4 * (Real.cos (2 * Real.pi / 5))^2 + 2 * Real.cos (2 * Real.pi / 5) - 1 = 0) ∧
  (4 * (Real.cos (4 * Real.pi / 5))^2 + 2 * Real.cos (4 * Real.pi / 5) - 1 = 0) ∧
  -- Part (c): Roots of 8x³ + 4x² - 4x - 1
  (8 * (Real.cos (2 * Real.pi / 7))^3 + 4 * (Real.cos (2 * Real.pi / 7))^2 - 4 * Real.cos (2 * Real.pi / 7) - 1 = 0) ∧
  (8 * (Real.cos (4 * Real.pi / 7))^3 + 4 * (Real.cos (4 * Real.pi / 7))^2 - 4 * Real.cos (4 * Real.pi / 7) - 1 = 0) ∧
  (8 * (Real.cos (6 * Real.pi / 7))^3 + 4 * (Real.cos (6 * Real.pi / 7))^2 - 4 * Real.cos (6 * Real.pi / 7) - 1 = 0) ∧
  -- Part (d): Roots of 8x³ - 6x + 1
  (8 * (Real.cos (2 * Real.pi / 9))^3 - 6 * Real.cos (2 * Real.pi / 9) + 1 = 0) ∧
  (8 * (Real.cos (4 * Real.pi / 9))^3 - 6 * Real.cos (4 * Real.pi / 9) + 1 = 0) ∧
  (8 * (Real.cos (8 * Real.pi / 9))^3 - 6 * Real.cos (8 * Real.pi / 9) + 1 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chebyshev_properties_l1157_115796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_condition_positive_condition_l1157_115701

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a * (x + b / x) + b * Real.log x

-- Part I
theorem monotonic_condition (a : ℝ) :
  (∀ x y, 0 < x ∧ 0 < y ∧ x < y → (f a (-4) x < f a (-4) y ∨ f a (-4) x > f a (-4) y)) ↔
  (a ≤ 0 ∨ a ≥ 1) := by
  sorry

-- Part II
theorem positive_condition (b : ℝ) :
  (∀ x, Real.exp 1 ≤ x ∧ x ≤ Real.exp 2 → f (-1) b x > 0) ↔
  (b > Real.exp 2 / (Real.exp 1 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_condition_positive_condition_l1157_115701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1157_115740

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (2, 0)

-- Define the right focus of the hyperbola
noncomputable def hyperbola_right_focus (a : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 + 1), 0)

-- Define the asymptote equation
def asymptote (k : ℝ) (x y : ℝ) : Prop := y = k * x

-- State the theorem
theorem hyperbola_asymptotes (a : ℝ) (h1 : a > 0) 
  (h2 : hyperbola_right_focus a = parabola_focus) :
  ∃ (k : ℝ), k = Real.sqrt 3 / 3 ∧ 
  (∀ (x y : ℝ), hyperbola a x y → (asymptote k x y ∨ asymptote (-k) x y)) :=
by
  sorry

-- Additional lemma to show the value of 'a'
lemma a_value (a : ℝ) (h1 : a > 0) 
  (h2 : hyperbola_right_focus a = parabola_focus) : a = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1157_115740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l1157_115783

/-- The circle with equation x^2 + y^2 - 6x = 0 -/
def myCircle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 6*p.1 = 0}

/-- The point P -/
def P : ℝ × ℝ := (4, 2)

/-- A chord is a line segment whose endpoints lie on the circle -/
def is_chord (l : Set (ℝ × ℝ)) : Prop :=
  ∃ (m n : ℝ × ℝ), m ∈ myCircle ∧ n ∈ myCircle ∧ m ≠ n ∧ l = {p | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (1 - t) • m + t • n}

/-- The line on which chord MN lies -/
def chord_line : Set (ℝ × ℝ) := {p | p.1 + 2*p.2 - 8 = 0}

/-- The theorem statement -/
theorem chord_equation : ∀ (MN : Set (ℝ × ℝ)), 
  is_chord MN → P ∈ MN → (∀ p ∈ MN, p ∈ chord_line) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l1157_115783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_magnitude_l1157_115773

/-- Given two vectors a and b in ℝ², prove that |a - 2b| = 2 -/
theorem vector_subtraction_magnitude (a b : ℝ × ℝ) :
  (a.1 = 2 ∧ a.2 = 0) →  -- a = (2,0)
  (b.1^2 + b.2^2 = 1) →  -- |b| = 1
  (a.1 * b.1 + a.2 * b.2 = 1) →  -- cos(π/3) = 1/2, so a⋅b = |a||b|cos(π/3) = 2*1*1/2 = 1
  (a.1 - 2*b.1)^2 + (a.2 - 2*b.2)^2 = 4 := by  -- |a - 2b|^2 = 4
  intro h1 h2 h3
  sorry

#check vector_subtraction_magnitude

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_magnitude_l1157_115773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_l1157_115788

/-- The function f defined as f(n) = (1/4) * n * (n+1) * (n+2) * (n+3) -/
noncomputable def f (n : ℝ) : ℝ := (1/4) * n * (n+1) * (n+2) * (n+3)

/-- Theorem stating that f(r) - f(r-1) = r * (r+1) * (r+2) for any real number r -/
theorem f_difference (r : ℝ) : f r - f (r-1) = r * (r+1) * (r+2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_l1157_115788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_discard_theorem_l1157_115704

/-- Represents the card discard and move process -/
def card_process (total_sets : ℕ) (cards_per_set : ℕ) : ℕ → ℕ → Prop :=
  sorry

/-- The number of cards with 7 discarded when a certain number of cards remain -/
def sevens_discarded (total_sets : ℕ) (cards_per_set : ℕ) (cards_remaining : ℕ) : ℕ :=
  sorry

/-- The position (set number and card number) of the last remaining card -/
def last_card_position (total_sets : ℕ) (cards_per_set : ℕ) : ℕ × ℕ :=
  sorry

theorem card_discard_theorem :
  let total_sets : ℕ := 288
  let cards_per_set : ℕ := 7
  let total_cards : ℕ := total_sets * cards_per_set
  let cards_remaining : ℕ := 301
  (∀ n : ℕ, card_process total_sets cards_per_set n (n + 1)) →
  sevens_discarded total_sets cards_per_set cards_remaining = 244 ∧
  last_card_position total_sets cards_per_set = (130, 3) := by
  sorry

#check card_discard_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_discard_theorem_l1157_115704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_2100_l1157_115723

-- Define the sequence G_n
def G : ℕ → ℕ
  | 0 => 1  -- Add this case to cover Nat.zero
  | 1 => 1
  | 2 => 1
  | n + 3 => G (n + 2) + G (n + 1) + 1

-- Define the property of being an increasing geometric sequence
def is_increasing_geometric (a b c : ℕ) : Prop :=
  G b * G b = G a * G c ∧ G a < G b ∧ G b < G c

-- State the theorem
theorem geometric_sequence_sum_2100 :
  ∀ a b c : ℕ, 
  is_increasing_geometric a b c → 
  a + b + c = 2100 → 
  a = 698 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_2100_l1157_115723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_l1157_115787

-- Define the circle
def is_in_circle (x y : ℝ) : Prop := x^2 + y^2 ≤ 4

-- Define the inscribed equilateral triangle
def is_in_inscribed_triangle (x y : ℝ) : Prop :=
  ∃ (a b c : ℝ × ℝ),
    (a.1^2 + a.2^2 = 4) ∧
    (b.1^2 + b.2^2 = 4) ∧
    (c.1^2 + c.2^2 = 4) ∧
    ((a.1 - b.1)^2 + (a.2 - b.2)^2 = (b.1 - c.1)^2 + (b.2 - c.2)^2) ∧
    ((a.1 - b.1)^2 + (a.2 - b.2)^2 = (c.1 - a.1)^2 + (c.2 - a.2)^2) ∧
    (x^2 + y^2 ≤ 4)

-- Define the circumscribed square
def is_in_circumscribed_square (x y : ℝ) : Prop := 
  (|x| ≤ 2*Real.sqrt 2) ∧ (|y| ≤ 2*Real.sqrt 2)

-- Theorem statement
theorem inequality_holds (x y : ℝ) 
  (h_circle : is_in_circle x y) 
  (h_triangle : is_in_inscribed_triangle x y) 
  (h_square : is_in_circumscribed_square x y) : 
  |x| + |y| ≤ 4 ∧ 4 ≤ 2*(x^2 + y^2) ∧ 2*(x^2 + y^2) ≤ 8*max (|x|) (|y|) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_l1157_115787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_concyclic_l1157_115795

-- Define the points
variable (A B C D E F P I₁ I₂ : EuclideanPlane)

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : EuclideanPlane) : Prop :=
  sorry

-- Define the incenter of a triangle
def is_incenter (I X Y Z : EuclideanPlane) : Prop :=
  sorry

-- Define the intersection of lines
def line_intersection (P Q R S T : EuclideanPlane) : Prop :=
  sorry

-- Define equality of lengths
def length_eq (P Q R S : EuclideanPlane) : Prop :=
  sorry

-- Define concyclicity
def are_concyclic (A B C D : EuclideanPlane) : Prop :=
  sorry

-- Theorem statement
theorem quadrilateral_concyclic 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : is_incenter I₁ A B C)
  (h3 : is_incenter I₂ D B C)
  (h4 : line_intersection E A B I₁ I₂)
  (h5 : line_intersection F D C I₁ I₂)
  (h6 : line_intersection P A B D C)
  (h7 : length_eq P E P F) :
  are_concyclic A B C D :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_concyclic_l1157_115795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_is_190_l1157_115720

/-- Represents the pricing and quantity information for a batch of clothing --/
structure Batch where
  quantity : ℕ
  price : ℚ
  pieces_per_unit : ℕ

/-- Calculates the selling price for 3 pieces of clothing given two batches and a profit percentage --/
noncomputable def calculate_selling_price (batch1 batch2 : Batch) (profit_percentage : ℚ) : ℚ :=
  let total_cost := batch1.price * (batch1.quantity / batch1.pieces_per_unit) +
                    batch2.price * (batch2.quantity / batch2.pieces_per_unit)
  let total_pieces := batch1.quantity + batch2.quantity
  let cost_per_3_pieces := 3 * total_cost / total_pieces
  cost_per_3_pieces * (1 + profit_percentage)

/-- Theorem stating that the selling price for 3 pieces should be 190 yuan --/
theorem selling_price_is_190 (batch1 batch2 : Batch) (profit_percentage : ℚ) :
  batch1.quantity = 3 ∧ 
  batch1.price = 160 ∧ 
  batch1.pieces_per_unit = 3 ∧
  batch2.quantity = 2 * batch1.quantity ∧ 
  batch2.price = 210 ∧ 
  batch2.pieces_per_unit = 4 ∧
  profit_percentage = 1/5 →
  calculate_selling_price batch1 batch2 profit_percentage = 190 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_is_190_l1157_115720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_exponential_function_l1157_115786

-- Define the function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 1) + 1

-- State the theorem
theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 1 = 2 ∧ ∀ x : ℝ, f a x = x → (x = 1 ∧ f a x = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_exponential_function_l1157_115786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l1157_115754

/-- The equation of the directrix of the parabola x^2 = -8y is y = 2 -/
theorem parabola_directrix : 
  ∃ k : ℝ, k = 2 ∧ ∀ x y : ℝ, x^2 = -8*y → y = k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l1157_115754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_covered_squares_count_l1157_115799

/-- Represents a square on the checkerboard -/
structure Square where
  row : Nat
  col : Nat

/-- Represents the checkerboard -/
def Checkerboard : Type := List (List Square)

/-- Represents a circular disc -/
structure Disc where
  diameter : ℝ
  center : ℝ × ℝ

/-- Checks if a square is completely covered by the disc -/
def is_covered (s : Square) (d : Disc) : Bool := 
  sorry

/-- Counts the number of squares completely covered by the disc -/
def count_covered_squares (board : Checkerboard) (d : Disc) : Nat :=
  sorry

/-- The main theorem to be proved -/
theorem covered_squares_count 
  (board : Checkerboard)
  (d : Disc)
  (h1 : board.length = 6)
  (h2 : (board.head!).length = 8)
  (h3 : d.diameter = 10)
  (h4 : d.center = (3.5, 4.5)) :
  count_covered_squares board d = 20 := by
  sorry

#check covered_squares_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_covered_squares_count_l1157_115799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sliding_time_approximation_l1157_115793

/-- The acceleration due to gravity in m/s² -/
def g : ℝ := 9.8

/-- The mass of the block on the table in kg -/
def m : ℝ := 1.0

/-- The mass of the hanging block in kg -/
def M : ℝ := 2.0

/-- The coefficient of friction between the block and the table -/
def μ : ℝ := 0.50

/-- The angle of the rope with the horizontal in radians -/
noncomputable def θ : ℝ := 10.0 * Real.pi / 180

/-- The distance the block slides in meters -/
def s : ℝ := 1.0

/-- The acceleration of the system -/
noncomputable def a : ℝ :=
  (M * g * (Real.cos θ + μ * Real.sin θ) - μ * m * g) /
  (M * Real.cos θ + M * μ * Real.sin θ + m * Real.cos θ)

/-- The time taken for the block to slide the given distance -/
noncomputable def t : ℝ := Real.sqrt (2 * s / a)

/-- Theorem stating that 100 times the sliding time is approximately 64 -/
theorem sliding_time_approximation : ⌊100 * t⌋ = 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sliding_time_approximation_l1157_115793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l1157_115746

/-- The minimum distance from a circle to a line --/
theorem min_distance_circle_to_line :
  let line_l : ℝ → ℝ → Prop := λ x y ↦ x - y - 8 = 0
  let circle_m_center : ℝ × ℝ := (1, -1)
  let circle_m_radius : ℝ := Real.sqrt 2
  let min_distance : ℝ := 2 * Real.sqrt 2
  (∀ p : ℝ × ℝ, (p.1 - circle_m_center.1)^2 + (p.2 - circle_m_center.2)^2 = circle_m_radius^2 →
    ∃ d : ℝ, d ≥ min_distance ∧ line_l p.1 (p.2 + d)) ∧
  (∃ p : ℝ × ℝ, (p.1 - circle_m_center.1)^2 + (p.2 - circle_m_center.2)^2 = circle_m_radius^2 ∧
    line_l p.1 (p.2 + min_distance)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l1157_115746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salamander_population_decline_l1157_115765

noncomputable def salamander_population (initial_population : ℝ) (years : ℕ) : ℝ :=
  initial_population * (0.8 ^ years)

noncomputable def population_ratio (initial_population : ℝ) (years : ℕ) : ℝ :=
  salamander_population initial_population years / initial_population

theorem salamander_population_decline (initial_population : ℝ) 
  (h : initial_population > 0) :
  ∃ (year : ℕ), 
    (∀ (y : ℕ), y < year → population_ratio initial_population y > 0.05) ∧
    population_ratio initial_population year ≤ 0.05 ∧
    year = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_salamander_population_decline_l1157_115765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1157_115794

noncomputable def S (n : ℕ+) (p : ℝ) : ℝ := n.val^2 * p / 2

noncomputable def a (n : ℕ+) (p : ℝ) : ℝ := (n.val - 1) * p

noncomputable def b (n : ℕ+) (p : ℝ) : ℝ := a n p / n.val + a (n + 1) p / (n + 1).val

noncomputable def T (n : ℕ+) (p : ℝ) : ℝ := 2 * n.val + 3 - 2 * (1 / n.val + 1 / (n + 1).val)

theorem sequence_properties (p : ℝ) (hp : p ≠ 0) :
  (∀ n : ℕ+, S n p = n.val^2 * p / 2) →
  (∀ n : ℕ+, a (n + 1) p - a n p = p) ∧
  (∀ n : ℕ+, T n p = 2 * n.val + 3 - 2 * (1 / n.val + 1 / (n + 1).val)) ∧
  (∃ N : ℕ+, ∀ n : ℕ+, n > N → 2 < T n p - 2 * n.val ∧ T n p - 2 * n.val < 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1157_115794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_completes_journey_l1157_115706

/-- Represents the spiral path of an ant -/
structure SpiralPath where
  k : ℝ
  t : ℝ
  h_k_pos : 0 < k
  h_k_lt_one : k < 1
  h_t_pos : t > 0

/-- The time taken for the ant to complete the entire spiral path -/
noncomputable def total_time (path : SpiralPath) : ℝ := path.t / (1 - path.k)

/-- Theorem stating that the ant can complete its journey in finite time -/
theorem ant_completes_journey (path : SpiralPath) : 
  ∃ (T : ℝ), T > 0 ∧ T = total_time path := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_completes_journey_l1157_115706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_both_groups_stop_prob_l1157_115729

/-- The probability of getting heads for a fair coin -/
noncomputable def fair_coin_prob : ℝ := 1 / 2

/-- The probability of getting heads for a biased coin -/
noncomputable def biased_coin_prob : ℝ := 1 / 3

/-- The number of people in each group -/
def group_size : ℕ := 3

/-- The probability that both groups stop flipping on the same round -/
noncomputable def both_groups_stop_same_round : ℝ :=
  (∑' n : ℕ, (fair_coin_prob ^ (group_size * n)) * 
    ((1 - biased_coin_prob) ^ (group_size * (n - 1)) * biased_coin_prob ^ group_size))

/-- Theorem: The probability that both groups stop flipping on the same round is 1/702 -/
theorem both_groups_stop_prob :
  both_groups_stop_same_round = 1 / 702 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_both_groups_stop_prob_l1157_115729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_condition_l1157_115798

theorem polynomial_roots_condition (a : ℝ) : 
  (∃ x₁ x₂ x₃ : ℝ, 
    (x₁^3 - 6*x₁^2 + a*x₁ + a = 0) ∧
    (x₂^3 - 6*x₂^2 + a*x₂ + a = 0) ∧
    (x₃^3 - 6*x₃^2 + a*x₃ + a = 0) ∧
    (x₁ - 3)^3 + (x₂ - 3)^3 + (x₃ - 3)^3 = 0) →
  a = 27/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_condition_l1157_115798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fundamental_trigonometric_identity_l1157_115790

theorem fundamental_trigonometric_identity (α : ℝ) : Real.sin α ^ 2 + Real.cos α ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fundamental_trigonometric_identity_l1157_115790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_powers_sum_20_factorial_l1157_115714

theorem highest_powers_sum_20_factorial : ∃ (a b : ℕ), 
  (10^a ∣ Nat.factorial 20) ∧ 
  (∀ k > a, ¬(10^k ∣ Nat.factorial 20)) ∧
  (6^b ∣ Nat.factorial 20) ∧ 
  (∀ m > b, ¬(6^m ∣ Nat.factorial 20)) ∧
  a + b = 12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_powers_sum_20_factorial_l1157_115714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_false_q_true_l1157_115722

theorem proposition_p_false_q_true :
  (∃ x : ℝ, x < 0 ∧ Real.log (x + 1) ≥ 0) ∧
  (∀ x : ℝ, Real.log (x + 1) < 0 → x < 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_false_q_true_l1157_115722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_family_average_age_l1157_115715

def family_ages (youngest_age : ℕ) : Prop :=
  let children_ages := [youngest_age, youngest_age + 3, youngest_age + 6, youngest_age + 9, youngest_age + 12]
  let sum_children_ages := children_ages.sum
  let oldest_age := youngest_age * 2
  let father_age := youngest_age * 3 + oldest_age
  let mother_age := father_age - 6
  sum_children_ages = 50 ∧
  children_ages.getLast? = some oldest_age ∧
  (children_ages.sum + father_age + mother_age) / 7 = 100 / 7

theorem family_average_age :
  ∃ (youngest_age : ℕ), family_ages youngest_age := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_family_average_age_l1157_115715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_integer_divisibility_property_l1157_115712

theorem odd_integer_divisibility_property (n : ℕ) : 
  n > 1 ∧ 
  Odd n ∧ 
  (∀ (a b : ℕ), a ∣ n → b ∣ n → Nat.Coprime a b → (a + b - 1) ∣ n) → 
  ∃ (p m : ℕ), n = p^m ∧ Nat.Prime p ∧ Odd p ∧ m > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_integer_divisibility_property_l1157_115712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_through_three_points_l1157_115775

theorem circle_equation_through_three_points :
  let A : ℝ × ℝ := (-1, -1)
  let B : ℝ × ℝ := (2, 2)
  let C : ℝ × ℝ := (-1, 1)
  let circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 5
  (circle_eq A.1 A.2) ∧ (circle_eq B.1 B.2) ∧ (circle_eq C.1 C.2) :=
by
  sorry

#check circle_equation_through_three_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_through_three_points_l1157_115775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_trig_identity_l1157_115751

theorem parallel_vectors_trig_identity (α : ℝ) :
  let a : Fin 2 → ℝ := ![Real.cos α, Real.sin α]
  let b : Fin 2 → ℝ := ![2, 3]
  (∃ (k : ℝ), a = k • b) →
  Real.sin α ^ 2 - Real.sin (2 * α) = -3/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_trig_identity_l1157_115751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1157_115718

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 * a * Real.cos x * Real.sin (x - Real.pi/6)

theorem function_properties :
  ∃ (a : ℝ), 
    (f a (Real.pi/3) = 1) ∧ 
    (a = 1) ∧
    (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f a (x + p) = f a x ∧ 
      ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f a (x + q) = f a x) → p ≤ q) ∧
    (∃ (p : ℝ), p = Real.pi ∧ p > 0 ∧ ∀ (x : ℝ), f a (x + p) = f a x ∧ 
      ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f a (x + q) = f a x) → p ≤ q) ∧
    (∀ (m : ℝ), (∀ (x y : ℝ), 0 ≤ x ∧ x < y ∧ y ≤ m → f a x < f a y) → m ≤ Real.pi/3) ∧
    (∃ (m : ℝ), m = Real.pi/3 ∧ ∀ (x y : ℝ), 0 ≤ x ∧ x < y ∧ y ≤ m → f a x < f a y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1157_115718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_theorem_l1157_115741

/-- The area of a sector in polar coordinates -/
noncomputable def sectorArea (r : ℝ) (θ : ℝ) : ℝ := (1/2) * r^2 * θ

/-- Theorem: Area of the sector bounded by θ = π/3, θ = 2π/3, and ρ = 4 is 8π/3 -/
theorem sector_area_theorem :
  let r : ℝ := 4
  let θ₁ : ℝ := π/3
  let θ₂ : ℝ := 2*π/3
  sectorArea r (θ₂ - θ₁) = 8*π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_theorem_l1157_115741
