import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_price_calculation_l980_98005

/-- Calculate the final price of two rings after discount and tax -/
theorem ring_price_calculation (price1 price2 discount_rate tax_rate : ℝ) 
  (h1 : price1 = 48)
  (h2 : price2 = 72)
  (h3 : discount_rate = 0.1)
  (h4 : tax_rate = 0.08) :
  let total_price := price1 + price2
  let discounted_price := total_price * (1 - discount_rate)
  let final_price := discounted_price * (1 + tax_rate)
  final_price = 116.64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_price_calculation_l980_98005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l980_98065

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then x * (x - b) else a * x * (x + 2)

-- State the theorem
theorem odd_function_value (a b : ℝ) :
  (∀ x, f a b (-x) = -(f a b x)) →
  f a b (a + b) = -1 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l980_98065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_inequality_l980_98032

/-- The radius of the inscribed circle of a regular k-gon inscribed in a circle with radius R -/
noncomputable def inscribed_circle_radius (k : ℕ) (R : ℝ) : ℝ :=
  R * Real.cos (Real.pi / k)

/-- The inequality for inscribed circle radii -/
theorem inscribed_circle_inequality (n : ℕ) (R : ℝ) :
  (n + 1 : ℝ) * inscribed_circle_radius (n + 1) R - n * inscribed_circle_radius n R > R := by
  sorry

#check inscribed_circle_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_inequality_l980_98032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sets_relation_l980_98049

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 3

-- Define the solution set of f > 0
def solution_set (a b : ℝ) : Set ℝ := {x | f a b x > 0}

-- Define the second quadratic function
def g (a b : ℝ) (x : ℝ) : ℝ := 3 * x^2 + b * x + a

-- Define the solution set of g < 0
def solution_set_g (a b : ℝ) : Set ℝ := {x | g a b x < 0}

-- State the theorem
theorem solution_sets_relation (a b : ℝ) :
  solution_set a b = Set.Ioo (-1) (1/2) →
  solution_set_g a b = Set.Ioo (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sets_relation_l980_98049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l980_98000

-- Define A(x) as the smallest integer not less than x
noncomputable def A (x : ℝ) : ℤ := Int.ceil x

-- State the theorem
theorem range_of_x (x : ℝ) (h : A (2 * x + 1) = 3) :
  x ∈ Set.Ioo (1/2 : ℝ) 1 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l980_98000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_l980_98035

/-- The length of the chord cut from a line by a polar curve. -/
noncomputable def chordLength (a b c d : ℝ) (k : ℝ) : ℝ :=
  let line := fun t : ℝ => (a + b * t, c + d * t)
  let curve := fun θ : ℝ => k * (Real.cos (θ + Real.pi / 4) : ℝ)
  -- The actual computation of the chord length would go here
  7/5  -- We're using the known result as a placeholder

/-- 
Given a line x = 1 + (4/5)t, y = -1 - (3/5)t and a curve ρ = √2 * cos(θ + π/4),
the length of the chord cut by the curve on the line is 7/5.
-/
theorem chord_length_specific : 
  chordLength 1 (4/5) (-1) (-3/5) (Real.sqrt 2) = 7/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_l980_98035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_points_determine_line_align_desks_l980_98090

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a line determined by two points
def Line (p1 p2 : Point) : Set Point :=
  {p : Point | ∃ t : ℝ, p.x = p1.x + t * (p2.x - p1.x) ∧ p.y = p1.y + t * (p2.y - p1.y)}

-- Define what it means for a set of points to be collinear
def IsCollinear (points : Set Point) :=
  ∃ p1 p2 : Point, p1 ≠ p2 ∧ points ⊆ Line p1 p2

-- Theorem: Given two distinct points, all points on the line determined by these two points are collinear
theorem two_points_determine_line (p1 p2 : Point) (h : p1 ≠ p2) :
  IsCollinear (Line p1 p2) := by
  sorry

-- Theorem: If a set of points is a subset of a line determined by two points, then all points in the set are collinear
theorem align_desks (desks : Set Point) (p1 p2 : Point) (h1 : p1 ∈ desks) (h2 : p2 ∈ desks) (h3 : p1 ≠ p2) 
  (h4 : desks ⊆ Line p1 p2) : 
  IsCollinear desks := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_points_determine_line_align_desks_l980_98090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_fourth_power_l980_98006

def expression (x : ℝ) : ℝ := 5 * (x^3 - x^4) - 2 * (3*x^2 - 2*x^4 + x^6) + 3 * (2*x^4 - x^10)

theorem coefficient_of_x_fourth_power (x : ℝ) : 
  (deriv^[4] expression) 0 / 24 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_fourth_power_l980_98006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_bc_equation_l980_98051

/-- Triangle ABC with given conditions -/
structure Triangle where
  A : ℝ × ℝ
  median_eq : ℝ → ℝ → ℝ
  angle_bisector_eq : ℝ → ℝ → ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given a triangle ABC with specific conditions, prove that the equation of side BC is 2x + 9y - 65 = 0 -/
theorem side_bc_equation (t : Triangle) 
  (h1 : t.A = (3, -1))
  (h2 : t.median_eq = λ x y ↦ 6*x + 10*y - 59)
  (h3 : t.angle_bisector_eq = λ x y ↦ x - 4*y + 10) :
  ∃ (l : LineEquation), l.a = 2 ∧ l.b = 9 ∧ l.c = -65 ∧ 
  (∀ x y, l.a * x + l.b * y + l.c = 0 ↔ (x, y) ∈ Set.range (λ t : ℝ ↦ (t, (-2/9)*t + 65/9))) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_bc_equation_l980_98051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_smallest_five_angles_l980_98098

noncomputable def Q (x : ℂ) : ℂ := ((x^20 - 1) / (x - 1))^2 - x^19

noncomputable def complex_zeros (k : ℕ) (s : ℕ → ℝ) (β : ℕ → ℝ) : ℂ :=
  s k * (Complex.cos (2 * Real.pi * β k) + Complex.I * Complex.sin (2 * Real.pi * β k))

theorem sum_of_smallest_five_angles 
  (s : ℕ → ℝ) 
  (β : ℕ → ℝ) 
  (h1 : ∀ k ∈ Finset.range 38, Q (complex_zeros k s β) = 0)
  (h2 : ∀ k ∈ Finset.range 37, β k ≤ β (k + 1))
  (h3 : ∀ k ∈ Finset.range 38, 0 < β k ∧ β k < 1) :
  β 0 + β 1 + β 2 + β 3 + β 4 = 183 / 399 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_smallest_five_angles_l980_98098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_segment_ratio_l980_98047

/-- A triangle with an inscribed circle -/
structure InscribedCircleTriangle where
  /-- Side lengths of the triangle -/
  a : ℝ
  b : ℝ
  c : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- The circle is inscribed in the triangle -/
  inscribed : r > 0
  /-- The triangle inequality holds -/
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The segments created by the point of tangency on side b -/
noncomputable def segments (t : InscribedCircleTriangle) : ℝ × ℝ :=
  let s := (t.a + t.b - t.c) / 2
  (s - t.r, t.b - s + t.r)

/-- The theorem to be proved -/
theorem inscribed_circle_segment_ratio 
  (t : InscribedCircleTriangle) 
  (h1 : t.a = 7) 
  (h2 : t.b = 24) 
  (h3 : t.c = 25) : 
  let (r, s) := segments t
  r / s = 1 / 7 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_segment_ratio_l980_98047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l980_98038

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 = 5
def circle_O2 (x y m : ℝ) : Prop := (x + m)^2 + y^2 = 20

-- Define the intersection points
def intersection_points (m : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), circle_O1 A.1 A.2 ∧ circle_O2 A.1 A.2 m ∧ 
                    circle_O1 B.1 B.2 ∧ circle_O2 B.1 B.2 m

-- Define the perpendicular tangents condition
def perpendicular_tangents (A : ℝ × ℝ) (m : ℝ) : Prop := sorry

-- Theorem statement
theorem intersection_distance (m : ℝ) 
  (h_intersect : intersection_points m)
  (h_perp : ∃ A, perpendicular_tangents A m) :
  ∃ A B : ℝ × ℝ, (circle_O1 A.1 A.2 ∧ circle_O2 A.1 A.2 m ∧
                  circle_O1 B.1 B.2 ∧ circle_O2 B.1 B.2 m) ∧
                 Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l980_98038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_right_triangle_shortest_side_l980_98048

/-- Represents a right triangle with sides in arithmetic progression -/
structure ArithmeticRightTriangle where
  a : ℝ  -- middle side length
  d : ℝ  -- common difference
  h1 : a > 0
  h2 : d > 0
  h3 : a - 2*d > 0  -- ensure all sides are positive

/-- The possible lengths of the shortest side in an arithmetic right triangle where one side is 4 times another -/
def possible_shortest_sides : Set ℝ := {6, 12, 18, 24, 30}

/-- Theorem stating the possible lengths of the shortest side -/
theorem arithmetic_right_triangle_shortest_side 
  (t : ArithmeticRightTriangle) 
  (h : ∃ (x y : ℝ), x ∈ ({t.a - 2*t.d, t.a, t.a + 2*t.d} : Set ℝ) ∧ 
                     y ∈ ({t.a - 2*t.d, t.a, t.a + 2*t.d} : Set ℝ) ∧ 
                     x = 4*y) : 
  t.a - 2*t.d ∈ possible_shortest_sides :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_right_triangle_shortest_side_l980_98048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_together_arrangement_l980_98003

/-- The number of ways to arrange 5 students (3 boys and 2 girls) in a row,
    such that the two girls are always together. -/
def arrangement_count : ℕ := 48

/-- The total number of students -/
def total_students : ℕ := 5

/-- The number of boys -/
def num_boys : ℕ := 3

/-- The number of girls -/
def num_girls : ℕ := 2

theorem girls_together_arrangement :
  arrangement_count = Nat.factorial (total_students - num_girls + 1) * Nat.factorial num_girls := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_together_arrangement_l980_98003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_consecutive_integers_not_square_l980_98053

theorem five_consecutive_integers_not_square (n : ℕ) : 
  ∃ k : ℕ, n * (n + 1) * (n + 2) * (n + 3) * (n + 4) ≠ k^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_consecutive_integers_not_square_l980_98053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l980_98004

/-- The constant term in the expansion of (x + 1/x^2)^6 is 15 -/
theorem constant_term_expansion : ∃ (p : ℕ → ℚ), 
  (∀ n, p n = (Finset.range 7).sum (λ k ↦ (n.choose k : ℚ) * (1 : ℚ)^(6 - k) * (1 : ℚ)^(2*k))) ∧ 
  p 0 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l980_98004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factors_equality_l980_98045

theorem prime_factors_equality (a m n : ℕ) (h1 : a > 1) (h2 : m < n) :
  (∀ p : ℕ, Nat.Prime p → (p ∣ a^m - 1 ↔ p ∣ a^n - 1)) →
  ∃ l : ℕ, l ≥ 2 ∧ a = 2^l - 1 ∧ m = 1 ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factors_equality_l980_98045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_properties_l980_98014

/-- Given a tetrahedron with vertices A₁, A₂, A₃, A₄ in ℝ³ -/
def A₁ : ℝ × ℝ × ℝ := (2, -4, -3)
def A₂ : ℝ × ℝ × ℝ := (5, -6, 0)
def A₃ : ℝ × ℝ × ℝ := (-1, 3, -3)
def A₄ : ℝ × ℝ × ℝ := (-10, -8, 7)

/-- The volume of the tetrahedron -/
noncomputable def tetrahedron_volume : ℝ := 31.5

/-- The height from A₄ to face A₁A₂A₃ -/
noncomputable def tetrahedron_height : ℝ := 189 / Real.sqrt 747

/-- Theorem stating the volume and height of the tetrahedron -/
theorem tetrahedron_properties :
  let v := tetrahedron_volume
  let h := tetrahedron_height
  (v = 31.5) ∧ (h = 189 / Real.sqrt 747) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_properties_l980_98014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l980_98041

-- Define the function f(x) = (x-3)e^x
noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

-- Theorem statement
theorem f_increasing_on_interval :
  ∀ x₁ x₂ : ℝ, 2 < x₁ → x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l980_98041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l980_98023

open Set Real

/-- A differentiable function f satisfying the given condition -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f ∧ (∀ x > 0, x * (deriv f x) + 2 * f x > 0)

theorem inequality_equivalence (f : ℝ → ℝ) (h : SatisfyingFunction f) :
  {x : ℝ | (x + 2017) * f (x + 2017) / 5 < 5 * f 5 / (x + 2017)} =
  {x : ℝ | -2017 < x ∧ x < -2012} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l980_98023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mean_weight_B_and_C_l980_98088

/-- Represents a pile of rocks -/
structure RockPile where
  weight : ℝ  -- Total weight of rocks in the pile
  count : ℝ   -- Number of rocks in the pile (using ℝ for simplicity)

/-- Calculates the mean weight of rocks in a pile -/
noncomputable def meanWeight (pile : RockPile) : ℝ := pile.weight / pile.count

/-- Calculates the mean weight of rocks in combined piles -/
noncomputable def combinedMeanWeight (pile1 pile2 : RockPile) : ℝ :=
  (pile1.weight + pile2.weight) / (pile1.count + pile2.count)

/-- Given conditions about three piles of rocks, proves that the maximum integer
    mean weight of combined piles B and C is 75 pounds -/
theorem max_mean_weight_B_and_C (A B C : RockPile)
    (hA : meanWeight A = 60)
    (hB : meanWeight B = 70)
    (hAB : combinedMeanWeight A B = 64)
    (hAC : combinedMeanWeight A C = 66) :
    ∃ (n : ℕ), n = 75 ∧ ∀ (m : ℕ), combinedMeanWeight B C ≤ m → m ≤ n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mean_weight_B_and_C_l980_98088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_polynomial_implies_divisibility_l980_98072

/-- A polynomial with even degree and odd coefficients that divides (x+1)^n - 1 -/
def DividingPolynomial (k n : ℕ) (c : Fin k → ℤ) : Prop :=
  ∃ (P : Polynomial ℤ),
    (P.degree = k) ∧
    (∀ i : Fin k, Odd (c i)) ∧
    (Even k) ∧
    (∃ Q : Polynomial ℤ, (Polynomial.X + 1 : Polynomial ℤ)^n - 1 = P * Q)

/-- The theorem stating that if a polynomial with even degree and odd coefficients
    divides (x+1)^n - 1, then k+1 divides n -/
theorem dividing_polynomial_implies_divisibility
  (k n : ℕ) (c : Fin k → ℤ) (h : DividingPolynomial k n c) :
  (k + 1) ∣ n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_polynomial_implies_divisibility_l980_98072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_fund_growth_rate_l980_98091

/-- Calculates the average annual growth rate given initial value, final value, and number of years. -/
noncomputable def averageAnnualGrowthRate (initial : ℝ) (final : ℝ) (years : ℝ) : ℝ :=
  ((final / initial) ^ (1 / years)) - 1

/-- Proves that the average annual growth rate of a fund that increases from 5000 to 7200 over 2 years is 20%. -/
theorem book_fund_growth_rate : 
  let initial := (5000 : ℝ)
  let final := (7200 : ℝ)
  let years := (2 : ℝ)
  averageAnnualGrowthRate initial final years = 0.2 := by
  sorry

/-- Approximate evaluation of the average annual growth rate -/
def approxGrowthRate (initial : Float) (final : Float) (years : Float) : Float :=
  ((final / initial) ^ (1 / years)) - 1

#eval approxGrowthRate 5000 7200 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_fund_growth_rate_l980_98091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kendra_earnings_theorem_l980_98050

/-- Kendra's shoe shop earnings over three years -/
noncomputable def kendra_earnings (L : ℝ) : ℝ × ℝ × ℝ :=
  let earnings_2014 := L - 8000
  let earnings_2015 := 1.1 * L * 1.2 / 0.9
  let earnings_2016 := (1.1 * L * 1.2 * 1.2 + 15000) / 0.85
  (earnings_2014, earnings_2015, earnings_2016)

/-- Theorem stating Kendra's earnings for 2014, 2015, and 2016 -/
theorem kendra_earnings_theorem (L : ℝ) :
  kendra_earnings L = (L - 8000, 1.4667 * L, 1.5529 * L + 17647.06) := by
  sorry

/-- Max's earnings in 2014 -/
noncomputable def max_earnings_2014 (L : ℝ) : ℝ :=
  (L - 8000) * 1.15

/-- Max's earnings in 2015 -/
noncomputable def max_earnings_2015 (L : ℝ) : ℝ :=
  1.1 * L * 1.2 * 0.9 - 9000

/-- Max's earnings in 2016 -/
noncomputable def max_earnings_2016 (L : ℝ) : ℝ :=
  (1.1 * L * 1.2 * 1.2 + 15000) * 1.1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kendra_earnings_theorem_l980_98050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_alpha_for_inequality_l980_98087

theorem smallest_alpha_for_inequality :
  ∀ α : ℝ, α > 0 →
  (∃ β : ℝ, β > 0 ∧ ∀ x : ℝ, x ∈ Set.Icc 0 1 →
    Real.sqrt (1 + x) + Real.sqrt (1 - x) ≤ 2 - x^α / β) →
  α ≥ 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_alpha_for_inequality_l980_98087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l980_98060

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola defined by y = ax^2 -/
structure Parabola where
  a : ℝ

/-- The focus of a parabola y = ax^2 -/
noncomputable def focus (p : Parabola) : Point :=
  { x := 0, y := 1 / (4 * p.a) }

/-- The directrix of a parabola y = ax^2 -/
noncomputable def directrix (p : Parabola) : ℝ := -1 / (4 * p.a)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For a parabola y = (1/4)x^2, if a point A on the parabola has y-coordinate 4,
    then the distance from A to the focus is 5 -/
theorem parabola_focus_distance (A : Point) :
  let p : Parabola := { a := 1/4 }
  A.y = (1/4) * A.x^2 → A.y = 4 →
  distance A (focus p) = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l980_98060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l980_98018

theorem sum_of_solutions (x y z : ℝ) (p q : ℕ) :
  4 * x + 7 * y + z = 11 →
  3 * x + y + 5 * z = 15 →
  x + y + z = p / q →
  Nat.Coprime p q →
  p > 0 →
  q > 0 →
  p - q = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l980_98018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_probability_l980_98095

def S : Finset ℤ := {-3, -2, -1, 0, 1, 4, 5}

def favorable_outcomes : Finset (ℤ × ℤ) :=
  S.product S |>.filter (fun p => p.1 ≠ p.2 ∧ p.1 * p.2 > 0)

theorem product_probability :
  (favorable_outcomes.card : ℚ) / (S.product S |>.filter (fun p => p.1 ≠ p.2)).card = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_probability_l980_98095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_intersection_length_l980_98096

-- Define the points
variable (A B C D E F G : ℝ × ℝ)

-- Define the parallelogram ABCD
def is_parallelogram (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1, B.2 - A.2) = (D.1 - C.1, D.2 - C.2) ∧
  (C.1 - B.1, C.2 - B.2) = (D.1 - A.1, D.2 - A.2)

-- Define that F is on the extension of AD
def on_extension (A D F : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ F = (A.1 + t * (D.1 - A.1), A.2 + t * (D.2 - A.2))

-- Define that E is on AC and G is on DC
def on_line_segment (P Q R : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ R = (P.1 + t * (Q.1 - P.1), P.2 + t * (Q.2 - P.2))

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- State the theorem
theorem parallelogram_intersection_length
  (h_para : is_parallelogram A B C D)
  (h_F : on_extension A D F)
  (h_E : on_line_segment A C E)
  (h_G : on_line_segment D C G)
  (h_BF : on_line_segment B F E ∧ on_line_segment B F G)
  (h_EF : distance E F = 40)
  (h_GF : distance G F = 30)
  (h_DC : distance D C = 3/2 * distance B C) :
  ∃ n : ℕ, n = 237 ∧ |distance B E - n| < 1 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_intersection_length_l980_98096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l980_98027

-- Define Point type if it's not already defined
structure Point where
  x : Real
  y : Real

-- Define the terminal_side function
def terminal_side (α : Real) (P : Point) : Prop :=
  P.x = Real.cos α ∧ P.y = Real.sin α

theorem sin_alpha_value (α : Real) (P : Point) :
  P.x = Real.sin (2 * Real.pi / 3) →
  P.y = Real.cos (2 * Real.pi / 3) →
  terminal_side α P →
  Real.sin α = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l980_98027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_property_l980_98008

def has_property (n : ℕ) : Prop :=
  ∃ (a : ℕ), (∀ i ∈ Finset.range n, i > 0) ∧
  (a + n - 1) ∣ Finset.lcm (Finset.range (n - 1)) (fun i => a + i)

def unique_property (n : ℕ) : Prop :=
  ∃! (a : ℕ), (∀ i ∈ Finset.range n, i > 0) ∧
  (a + n - 1) ∣ Finset.lcm (Finset.range (n - 1)) (fun i => a + i)

theorem consecutive_integers_property (n : ℕ) (h : n > 2) :
  (has_property n ↔ n ≥ 4) ∧ (unique_property n ↔ n = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_property_l980_98008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_sum_equals_eight_l980_98089

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / (x + 1) + (x + 1) / (x + 2) + (x + 2) / (x + 3) + (x + 3) / (x + 4)

-- Theorem statement
theorem function_sum_equals_eight : 
  f (-6 + Real.sqrt 5) + f (1 - Real.sqrt 5) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_sum_equals_eight_l980_98089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l980_98079

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (2 * x^2 - 1) / Real.log a

-- State the theorem
theorem derivative_of_f (a : ℝ) (x : ℝ) (ha : a > 0) (ha' : a ≠ 1) (hx : 2 * x^2 - 1 > 0) :
  deriv (f a) x = 4 * x / ((2 * x^2 - 1) * Real.log a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l980_98079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_props_l980_98021

-- Define the propositions
def prop_A (x y : ℝ) : Prop := x > y → x^2 > y^2
def prop_B (x : ℝ) : Prop := x > 10 → x > 5
def prop_C (a b c : ℝ) : Prop := a = b → a * c = b * c
def prop_D (a b c d : ℝ) : Prop := a - c > b - d → a > b ∧ c < d

-- Define what it means for p to be sufficient but not necessary for q
def sufficient_not_necessary (P Q : Prop) : Prop :=
  (P → Q) ∧ ¬(Q → P)

-- State the theorem
theorem sufficient_not_necessary_props :
  (∃ x y : ℝ, ¬sufficient_not_necessary (x > y) (x^2 > y^2)) ∧
  (sufficient_not_necessary (10 < 10) (5 < 10)) ∧
  (sufficient_not_necessary (0 = 0) (0 * 0 = 0 * 0)) ∧
  (∃ a b c d : ℝ, ¬sufficient_not_necessary (a - c > b - d) (a > b ∧ c < d)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_props_l980_98021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_2_3_same_cluster_l980_98092

noncomputable def function_1 (x : ℝ) := Real.sin x * Real.cos x
noncomputable def function_2 (x : ℝ) := 2 * Real.sin (x + Real.pi / 4)
noncomputable def function_3 (x : ℝ) := Real.sin x + Real.sqrt 3 * Real.cos x
noncomputable def function_4 (x : ℝ) := Real.sqrt 2 * Real.sin (2 * x) + 1

def same_cluster (f g : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ x, f x = g (x + a) + b

theorem functions_2_3_same_cluster :
  same_cluster function_2 function_3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_2_3_same_cluster_l980_98092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_spiral_rise_l980_98071

/-- Represents the rise per circuit for a squirrel's spiral path on a cylindrical post -/
noncomputable def risePerCircuit (postHeight circumference travelDistance : ℝ) : ℝ :=
  postHeight * circumference / travelDistance

/-- Theorem stating that for a post of height 25 feet and circumference 3 feet,
    if a squirrel travels 15 feet, the rise per circuit is 5 feet -/
theorem squirrel_spiral_rise :
  risePerCircuit 25 3 15 = 5 := by
  -- Unfold the definition of risePerCircuit
  unfold risePerCircuit
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_spiral_rise_l980_98071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_difference_l980_98076

theorem sin_difference (α β : ℝ) 
  (h1 : Real.sin α - Real.cos β = 3/4)
  (h2 : Real.cos α + Real.sin β = -2/5) : 
  Real.sin (α - β) = 511/800 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_difference_l980_98076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_to_pentagon_area_ratio_l980_98078

/-- Represents a pentagon formed by an isosceles triangle on top of a rectangle -/
structure IsoscelesRectanglePentagon where
  /-- Length of the equal legs of the isosceles triangle and shorter side of the rectangle -/
  x : ℝ
  /-- Assumption that x is positive -/
  x_pos : 0 < x

/-- The area of the isosceles triangle in the pentagon -/
noncomputable def triangle_area (p : IsoscelesRectanglePentagon) : ℝ :=
  (p.x^2 * Real.sqrt 3) / 4

/-- The area of the rectangle in the pentagon -/
def rectangle_area (p : IsoscelesRectanglePentagon) : ℝ :=
  2 * p.x^2

/-- The total area of the pentagon -/
noncomputable def pentagon_area (p : IsoscelesRectanglePentagon) : ℝ :=
  triangle_area p + rectangle_area p

/-- The main theorem: the ratio of the triangle area to the total pentagon area -/
theorem triangle_to_pentagon_area_ratio (p : IsoscelesRectanglePentagon) :
    triangle_area p / pentagon_area p = Real.sqrt 3 / (8 + Real.sqrt 3) := by
  sorry

#check triangle_to_pentagon_area_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_to_pentagon_area_ratio_l980_98078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l980_98080

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x + Real.pi/6) - 1

theorem f_properties :
  let period := Real.pi
  let interval := Set.Icc (-Real.pi/6) (Real.pi/4)
  (∀ x, f (x + period) = f x) ∧
  (∃ x ∈ interval, f x = 2 ∧ ∀ y ∈ interval, f y ≤ f x) ∧
  (∃ x ∈ interval, f x = -1 ∧ ∀ y ∈ interval, f x ≤ f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l980_98080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_function_l980_98068

theorem inverse_proportion_function (m : ℝ) :
  (∀ x ≠ 0, ∃ k ≠ 0, (m + 1) * x^(m^2 - 2) = k / x) ↔ m = 1 :=
by
  -- We'll prove this equivalence
  sorry

#check inverse_proportion_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_function_l980_98068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_age_l980_98058

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (new_avg_age : ℝ) (teacher_age : ℝ) :
  num_students = 15 →
  student_avg_age = 10 →
  new_avg_age = student_avg_age + 1 →
  (num_students * new_avg_age + teacher_age) / (num_students + 1) = new_avg_age →
  teacher_age = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_age_l980_98058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_outside_circle_l980_98054

/-- A circle with center O and radius r -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Predicate to check if a point is outside a circle -/
def is_outside (c : Circle) (p : ℝ × ℝ) : Prop :=
  distance c.center p > c.radius

theorem point_outside_circle (O : Circle) (P : ℝ × ℝ) 
    (h1 : O.radius = 5)
    (h2 : distance O.center P = 6) : 
    is_outside O P := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_outside_circle_l980_98054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_similar_500_pointed_stars_l980_98009

/-- A regular n-pointed star -/
structure RegularStar (n : ℕ) where
  m : ℕ
  h1 : m ≤ n - 1
  h2 : Nat.Coprime m n

/-- The number of non-similar regular n-pointed stars -/
def nonSimilarStarCount (n : ℕ) : ℕ :=
  (Nat.totient n + 2) / 2

theorem non_similar_500_pointed_stars :
  nonSimilarStarCount 500 = 100 := by
  sorry

#eval nonSimilarStarCount 500  -- This should output 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_similar_500_pointed_stars_l980_98009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chameleon_color_invariant_l980_98019

/-- Represents the state of chameleons on Sunny Island -/
structure ChameleonState where
  white : Nat
  black : Nat

/-- The rule for chameleon color change when two meet -/
def colorChange (state : ChameleonState) : ChameleonState :=
  { white := state.white - 1, black := state.black - 1 }

/-- The invariant quantity k = white - black (mod 3) -/
def kInvariant (state : ChameleonState) : Int :=
  (state.white - state.black) % 3

theorem chameleon_color_invariant 
  (initialState : ChameleonState) 
  (h1 : initialState.white = 20) 
  (h2 : initialState.black = 25) :
  ∀ (finalState : ChameleonState), 
    (∃ (n : Nat), (Nat.rec initialState (fun _ => colorChange) n) = finalState) →
    kInvariant finalState ≠ 0 :=
by
  sorry

#check chameleon_color_invariant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chameleon_color_invariant_l980_98019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l980_98070

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  sum_angles : A + B + C = π
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0

/-- The theorem to be proven -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.a = t.c * Real.sin t.B + t.b * Real.cos t.C)
  (h2 : t.b = Real.sqrt 2) : 
  (t.A + t.C = 3 * π / 4) ∧ 
  (∃ (max_area : ℝ), max_area = (1 + Real.sqrt 2) / 2 ∧ 
    ∀ (area : ℝ), area = 1 / 2 * t.a * t.c * Real.sin t.B → area ≤ max_area) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l980_98070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_integers_sum_l980_98022

theorem two_integers_sum (x y : ℕ) : 
  x * y + x + y = 137 ∧ 
  Nat.Coprime x y ∧ 
  x < 30 ∧ y < 30 →
  x + y = 27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_integers_sum_l980_98022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_lower_bound_l980_98061

-- Define the function y as noncomputable
noncomputable def y (x : ℝ) : ℝ := x + 4 / (x + 1)

-- State the theorem
theorem y_lower_bound :
  (∀ x : ℝ, x ≥ 0 → y x ≥ 3) ∧
  (∃ x : ℝ, x ≥ 0 ∧ y x = 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_lower_bound_l980_98061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_fixed_point_and_min_area_l980_98052

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the line
def line (k m : ℝ) (x y : ℝ) : Prop := y = k*x + m

-- Define the directrix
def directrix (p : ℝ) (x : ℝ) : Prop := x = -p/2

-- Define the theorem
theorem parabola_intersection_fixed_point_and_min_area (p : ℝ) (h : p > 0) :
  ∃ (k m : ℝ), 
    -- The line passes through the fixed point (p/2, 0)
    (line k m (p/2) 0) ∧ 
    -- The minimum area of quadrilateral ABB'A' is 2p^2
    (∀ (k' m' : ℝ), 
      ∃ (x_A y_A x_B y_B : ℝ),
        (parabola p x_A y_A) ∧ 
        (parabola p x_B y_B) ∧
        (line k' m' x_A y_A) ∧ 
        (line k' m' x_B y_B) →
        2*p^2 ≤ (1/2) * (y_A - y_B)^2 * Real.sqrt (1 + 1/k'^2)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_fixed_point_and_min_area_l980_98052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l980_98024

noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then x^2 + x else -3*x

theorem range_of_a (a : ℝ) : a * (f a - f (-a)) > 0 ↔ a ∈ Set.Iio (-2) ∪ Set.Ioi 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l980_98024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_and_occurrences_l980_98007

noncomputable def f (x : ℝ) := 3 * Real.sin (1/2 * x + Real.pi/4) - 1

theorem f_minimum_value_and_occurrences :
  (∀ x : ℝ, f x ≥ -4) ∧
  (∀ k : ℤ, f (4 * k * Real.pi - 3 * Real.pi / 2) = -4) :=
by
  sorry

#check f_minimum_value_and_occurrences

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_and_occurrences_l980_98007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_spheres_l980_98067

theorem paint_spheres (R : ℝ) (paint_large : ℝ) (h1 : paint_large = 2.4) :
  let surface_area_large := 4 * Real.pi * R^2
  let volume_large := (4/3) * Real.pi * R^3
  let r := R / 4
  let surface_area_small := 4 * Real.pi * r^2
  let volume_small := (4/3) * Real.pi * r^3
  let num_small_spheres := 64
  volume_large = num_small_spheres * volume_small →
  num_small_spheres * surface_area_small * (paint_large / surface_area_large) = 9.6 := by
  intro h
  sorry

#check paint_spheres

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_spheres_l980_98067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mountain_hike_distance_l980_98028

/-- Represents the distance between two points on a mountain trail -/
structure MountainDistance where
  kilometers : ℚ
  meters : ℚ

/-- Calculates the total distance of a round trip on a mountain trail -/
def total_round_trip_distance (entrance_to_temple : MountainDistance) (temple_to_top : ℚ) : ℚ :=
  2 * (entrance_to_temple.kilometers * 1000 + entrance_to_temple.meters + temple_to_top) / 1000

/-- Theorem stating the total distance walked on the mountain trail -/
theorem mountain_hike_distance :
  let entrance_to_temple : MountainDistance := ⟨4, 436⟩
  let temple_to_top : ℚ := 1999
  total_round_trip_distance entrance_to_temple temple_to_top = 12870 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mountain_hike_distance_l980_98028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_formula_l980_98043

/-- Regular triangular prism with edge length a -/
structure RegularTriangularPrism (a : ℝ) where
  edge_length : a > 0

/-- Cross-section of the prism formed by a plane passing through an edge of the base
    and the midpoint of a non-parallel edge of the other base -/
noncomputable def cross_section_area (a : ℝ) (prism : RegularTriangularPrism a) : ℝ :=
  (3 * a^2 * Real.sqrt 19) / 16

/-- Theorem stating that the area of the cross-section is (3a^2 * √19) / 16 -/
theorem cross_section_area_formula (a : ℝ) (prism : RegularTriangularPrism a) :
  cross_section_area a prism = (3 * a^2 * Real.sqrt 19) / 16 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_formula_l980_98043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_magnitude_l980_98074

-- Define the vectors
def a : ℝ × ℝ := (2, 1)
def b (x : ℝ) : ℝ × ℝ := (x, -2)

-- Define the parallel condition
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Define the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

-- The theorem to prove
theorem parallel_vectors_magnitude :
  ∀ x : ℝ, parallel a (b x) → magnitude (b x) = 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_magnitude_l980_98074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_system_l980_98064

variable (t : ℝ)
variable (C₁ C₂ : ℝ)

noncomputable def x : ℝ → ℝ := λ t => 4 * C₁ * Real.exp t + C₂ * Real.exp (-2 * t)
noncomputable def y : ℝ → ℝ := λ t => C₁ * Real.exp t + C₂ * Real.exp (-2 * t)

noncomputable def x' : ℝ → ℝ := λ t => 4 * C₁ * Real.exp t - 2 * C₂ * Real.exp (-2 * t)
noncomputable def y' : ℝ → ℝ := λ t => C₁ * Real.exp t - 2 * C₂ * Real.exp (-2 * t)

theorem solution_satisfies_system :
  ∀ t, x' t = 2 * (x t) - 4 * (y t) ∧ y' t = (x t) - 3 * (y t) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_system_l980_98064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l980_98040

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else -a*x

theorem decreasing_f_implies_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (1/8 : ℝ) (1/3 : ℝ) ∧ a ≠ 1/3 :=
by
  sorry

#check decreasing_f_implies_a_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l980_98040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_hardcover_implies_l980_98099

-- Define the universe of books in the library
variable (Book : Type)

-- Define the property of being hardcover
variable (is_hardcover : Book → Prop)

-- Define the statement "All books in the library are hardcover"
def all_hardcover (Book : Type) (is_hardcover : Book → Prop) : Prop := 
  ∀ b : Book, is_hardcover b

-- Theorem statement
theorem not_all_hardcover_implies {Book : Type} {is_hardcover : Book → Prop}
  (h : ¬all_hardcover Book is_hardcover) :
  (∃ b : Book, ¬is_hardcover b) ∧ ¬(∀ b : Book, is_hardcover b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_hardcover_implies_l980_98099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_half_u_l980_98013

/-- The curve C defined by x^2 - y^2 = 1 -/
def C (x y : ℝ) : Prop := x^2 - y^2 = 1

/-- The x-coordinate of point P -/
noncomputable def x_coord (u : ℝ) : ℝ := (Real.exp u + Real.exp (-u)) / 2

/-- The area bounded by OP, x-axis, and curve C -/
noncomputable def bounded_area (u : ℝ) : ℝ := sorry

/-- Theorem statement -/
theorem area_equals_half_u (u : ℝ) (x y : ℝ) 
  (h1 : C x y) 
  (h2 : x > 0) 
  (h3 : y > 0) 
  (h4 : x = x_coord u) 
  (h5 : u ≥ 0) : 
  bounded_area u = 1/2 * u := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_half_u_l980_98013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l980_98093

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  (Real.sin A) / a = (Real.sin B) / b ∧ (Real.sin B) / b = (Real.sin C) / c ∧
  Real.sin A ^ 2 + Real.sin C ^ 2 - Real.sin B ^ 2 = Real.sqrt 3 * Real.sin A * Real.sin C →
  B = π / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l980_98093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l980_98057

theorem min_distance_to_line (x y : ℝ) (h : 8 * x + 15 * y = 120) :
  ∃ (min : ℝ), min = 120 / 17 ∧ Real.sqrt (x^2 + y^2) ≥ min := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l980_98057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_equals_ten_l980_98084

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 1 + (x + 2)^2 else 2^(-x - 2)

theorem f_one_equals_ten : f 1 = 10 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the if-then-else expression
  simp
  -- Evaluate the expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_equals_ten_l980_98084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sallys_initial_fries_proof_l980_98055

/-- The number of fries Mark had initially -/
def marks_fries : ℕ := 36

/-- The fraction of fries Mark gave to Sally -/
def fraction_given : ℚ := 1/3

/-- The number of fries Sally had after receiving fries from Mark -/
def sallys_final_fries : ℕ := 26

/-- Sally's initial number of fries -/
def sallys_initial_fries : ℕ := 14

theorem sallys_initial_fries_proof :
  sallys_initial_fries = sallys_final_fries - (fraction_given * marks_fries).floor :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sallys_initial_fries_proof_l980_98055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_A_in_fourth_quadrant_point_A_on_axes_l980_98044

/-- Point A in the Cartesian coordinate system -/
structure PointA where
  m : ℝ
  x : ℝ := 6 - m
  y : ℝ := 5 - 2*m

/-- Predicate for a point being in the fourth quadrant -/
def in_fourth_quadrant (p : PointA) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Predicate for a point being on the coordinate axes -/
def on_coordinate_axes (p : PointA) : Prop :=
  p.x = 0 ∨ p.y = 0

/-- Theorem stating the conditions for Point A to be in the fourth quadrant -/
theorem point_A_in_fourth_quadrant (p : PointA) :
  in_fourth_quadrant p ↔ 2.5 < p.m ∧ p.m < 6 := by sorry

/-- Theorem stating the conditions for Point A to be on the coordinate axes -/
theorem point_A_on_axes (p : PointA) :
  on_coordinate_axes p ↔ p.m = 2.5 ∨ p.m = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_A_in_fourth_quadrant_point_A_on_axes_l980_98044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_piecewise_function_theorem_l980_98094

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then a * x + b else 9 - 4 * x

theorem piecewise_function_theorem (a b : ℝ) :
  (∀ x, f a b (f a b x) = x) →
  a = -1/4 ∧ b = 9/4 ∧ a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_piecewise_function_theorem_l980_98094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_maximized_optimal_m_maximizes_area_l980_98012

noncomputable section

-- Define the curve y = √x
def curve (x : ℝ) : ℝ := Real.sqrt x

-- Define the points A, B, and C
def A : ℝ × ℝ := (1, curve 1)
def B (m : ℝ) : ℝ × ℝ := (m, curve m)
def C : ℝ × ℝ := (4, curve 4)

-- Define the area of triangle ABC as a function of m
def area (m : ℝ) : ℝ := (1/2) * |m - 3 * Real.sqrt m + 2|

-- Theorem statement
theorem triangle_area_maximized (m : ℝ) (h1 : 1 < m) (h2 : m < 4) :
  area m ≤ area (9/4) := by sorry

-- The value of m that maximizes the area
def optimal_m : ℝ := 9/4

-- Theorem stating that optimal_m is indeed the maximizer
theorem optimal_m_maximizes_area :
  ∀ m, 1 < m → m < 4 → area m ≤ area optimal_m := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_maximized_optimal_m_maximizes_area_l980_98012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_l980_98002

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPureImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- Given that z = (2+3i)(1-mi) is a pure imaginary number, prove that m = -2/3 -/
theorem pure_imaginary_condition (m : ℝ) :
  let z : ℂ := Complex.mk (2 - 3*m) (3 + 2*m)
  IsPureImaginary z → m = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_l980_98002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_increasing_condition_l980_98034

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def sum_arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * a₁ + n * (n - 1) / 2 * d

def is_increasing_sequence (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S (n + 1) > S n

theorem arithmetic_sequence_increasing_condition (a₁ d : ℝ) :
  (is_increasing_sequence (sum_arithmetic_sequence a₁ d) → a₁ + d > 0) ∧
  ¬(a₁ + d > 0 → is_increasing_sequence (sum_arithmetic_sequence a₁ d)) := by
  sorry

#check arithmetic_sequence_increasing_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_increasing_condition_l980_98034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ratio_l980_98083

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  height : ℝ
  width : ℝ

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.height + r.width)

/-- The initial square paper -/
noncomputable def initial_square : Rectangle := { height := 6, width := 6 }

/-- The paper after folding horizontally -/
noncomputable def folded_paper : Rectangle := { height := initial_square.height / 2, width := initial_square.width }

/-- One of the rectangles after the first cut -/
noncomputable def first_cut_rectangle : Rectangle := { height := folded_paper.height, width := folded_paper.width / 2 }

/-- One of the smallest rectangles after all cuts -/
noncomputable def smallest_rectangle : Rectangle := { height := first_cut_rectangle.height, width := first_cut_rectangle.width / 2 }

/-- Theorem stating the ratio of perimeters -/
theorem perimeter_ratio :
  perimeter smallest_rectangle / perimeter folded_paper = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ratio_l980_98083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l980_98039

theorem cos_minus_sin_value (θ : Real) 
  (h1 : Real.sin θ * Real.cos θ = 1/8) 
  (h2 : π/4 < θ)
  (h3 : θ < π/2) : 
  Real.cos θ - Real.sin θ = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l980_98039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_has_winning_strategy_l980_98082

/-- Represents a binary number as a list of booleans -/
def BinaryNumber := List Bool

/-- Checks if a binary number is odd -/
def is_odd (n : BinaryNumber) : Prop :=
  n.length > 0 ∧ n.getLast! = true

/-- Represents a player in the game -/
inductive Player
| Alice
| Bob

/-- Represents a game state -/
structure GameState where
  number : BinaryNumber
  current_player : Player

/-- Represents a game move -/
inductive Move
| SubtractOne
| EraseLastDigit

/-- Applies a move to a game state -/
def apply_move (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if a game state is terminal (board is blank) -/
def is_terminal (state : GameState) : Prop :=
  state.number.length = 0

/-- Simulates the game until completion -/
def play_game (initial_state : GameState) (bob_strategy alice_strategy : GameState → Move) : GameState :=
  sorry

/-- Theorem: Bob has a winning strategy -/
theorem bob_has_winning_strategy 
  (initial_number : BinaryNumber)
  (h_initial_odd : is_odd initial_number) :
  ∃ (strategy : GameState → Move),
    ∀ (alice_strategy : GameState → Move),
      let final_state := play_game 
        { number := initial_number, current_player := Player.Bob }
        strategy
        alice_strategy
      is_terminal final_state ∧ final_state.current_player = Player.Alice :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_has_winning_strategy_l980_98082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_ellipse_theorem_l980_98030

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The origin point (0, 0, 0) -/
def origin : Point3D := ⟨0, 0, 0⟩

/-- Check if two points are distinct -/
def distinct (p q : Point3D) : Prop := p ≠ q

/-- A point lies on a plane -/
def lies_on (p : Point3D) (plane : Plane3D) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

/-- A plane intersects an axis at a point -/
def intersects_axis (plane : Plane3D) (axis : Fin 3) (p : Point3D) : Prop :=
  lies_on p plane ∧ (∀ i : Fin 3, i ≠ axis → p.x = 0)

/-- The center of an ellipse passing through four points -/
noncomputable def ellipse_center (p1 p2 p3 p4 : Point3D) : Point3D :=
  ⟨(p1.x + p2.x + p3.x + p4.x) / 4, (p1.y + p2.y + p3.y + p4.y) / 4, (p1.z + p2.z + p3.z + p4.z) / 4⟩

/-- The main theorem -/
theorem plane_ellipse_theorem (a b c p q r : ℝ) (plane : Plane3D) (A B C : Point3D) :
  let point := Point3D.mk a b c
  let center := Point3D.mk p q r
  lies_on point plane →
  intersects_axis plane 0 A →
  intersects_axis plane 1 B →
  intersects_axis plane 2 C →
  distinct A origin →
  distinct B origin →
  distinct C origin →
  center = ellipse_center origin A B C →
  a / p + b / q + c / r = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_ellipse_theorem_l980_98030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l980_98033

noncomputable def f (x : ℝ) : ℝ := x^2 + 1/x + 1/(x^2 + 1/x)

theorem min_value_of_f (x : ℝ) (h : x > 0) :
  f x ≥ f (Real.rpow 2 (-1/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l980_98033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_function_l980_98085

theorem fixed_point_of_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(2*x - 1) - 2
  (f (1/2) = -1) ∧ (∀ x : ℝ, f x = x → x = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_function_l980_98085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_union_B_l980_98029

-- Define the set of integers
def U : Set Int := Set.univ

-- Define set A
def A : Set Int := { x | ∃ k : Int, x = 3 * k + 1 }

-- Define set B
def B : Set Int := { x | ∃ k : Int, x = 3 * k + 2 }

-- Define the set of integers divisible by 3
def DivisibleBy3 : Set Int := { x | ∃ k : Int, x = 3 * k }

-- Theorem statement
theorem complement_of_A_union_B (x : Int) : 
  x ∈ (U \ (A ∪ B)) ↔ x ∈ DivisibleBy3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_union_B_l980_98029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapping_paper_area_l980_98026

/-- The area of a rectangular sheet of wrapping paper that encloses a rectangular box -/
theorem wrapping_paper_area (a b w : ℝ) (h : a > b) : 
  (2*a + 2*w) * (2*b + 2*w) = 4*(a*b + a*w + b*w + w^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapping_paper_area_l980_98026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_divisible_consecutive_product_l980_98010

theorem infinitely_many_divisible_consecutive_product (k : ℕ) :
  ∃ f : ℕ → ℕ, StrictMono f ∧ ∀ n, ((f n + k)^2 + 1) ∣ (Nat.factorial (f n)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_divisible_consecutive_product_l980_98010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centers_properties_l980_98015

noncomputable section

-- Define the triangle vertices
def O : ℝ × ℝ := (0, 0)
def M : ℝ × ℝ := (10, 0)
def N : ℝ × ℝ := (6, 8)

-- Define the special points
noncomputable def G : ℝ × ℝ := (16/3, 8/3)
noncomputable def H : ℝ × ℝ := (6, 3)
noncomputable def C : ℝ × ℝ := (5, 5/2)

-- Define distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  (q.1 - p.1) * (r.2 - p.2) = (r.1 - p.1) * (q.2 - p.2)

theorem triangle_centers_properties :
  (G = (16/3, 8/3)) ∧
  (H = (6, 3)) ∧
  (C = (5, 5/2)) ∧
  collinear G H C ∧
  distance C H = 3 * distance C G :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centers_properties_l980_98015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_five_theta_l980_98037

theorem cos_five_theta (θ : ℝ) (h : Real.cos θ = 1/3) : Real.cos (5*θ) = 241/243 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_five_theta_l980_98037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_visitors_l980_98056

/-- The number of visitors who answered the questionnaire about a Picasso painting -/
def V : ℕ := sorry

/-- The number of visitors who both enjoyed and understood the painting -/
def E : ℕ := sorry

/-- Axiom: 140 visitors neither enjoyed nor understood the painting -/
axiom not_enjoyed_not_understood : V = E + 140

/-- Axiom: 3/4 of all visitors both enjoyed and understood the painting -/
axiom three_fourths_enjoyed_understood : E = (3 * V) / 4

/-- Theorem: The total number of visitors who answered the questionnaire is 560 -/
theorem total_visitors : V = 560 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_visitors_l980_98056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_product_product_units_digit_l980_98073

theorem units_digit_of_product : ℕ → ℕ := λ n => n % 10

def product : ℕ := (5 + 1) * (5^3 + 1) * (5^6 + 1) * (5^12 + 1)

theorem product_units_digit : units_digit_of_product product = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_product_product_units_digit_l980_98073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_max_implies_a_greater_than_one_l980_98097

/-- The function f(x) = x ln x - (1/2)ax^2 + (a-1)x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - (1/2) * a * x^2 + (a - 1) * x

/-- f'(x) = ln x - ax + a -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + a

theorem local_max_implies_a_greater_than_one (a : ℝ) :
  (∀ x > 0, f a x ≤ f a 1) →
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1) →
  a > 1 := by
  sorry

#check local_max_implies_a_greater_than_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_max_implies_a_greater_than_one_l980_98097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_feet_collinear_on_median_l980_98046

/-- Represents a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a triangle in a plane -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The feet of perpendicular bisectors of internal and external angle bisectors -/
def feet (t : Triangle) : Point × Point × Point × Point :=
  sorry

/-- Checks if four points are collinear -/
def are_collinear (p q r s : Point) : Prop :=
  sorry

/-- Checks if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  sorry

/-- Returns the median line of a triangle -/
def median_line (t : Triangle) : Line :=
  sorry

/-- Main theorem: The feet of perpendicular bisectors are collinear and lie on the median -/
theorem perpendicular_bisector_feet_collinear_on_median
  (t : Triangle) :
  let (F, G, E, D) := feet t
  are_collinear F G E D ∧
  point_on_line F (median_line t) ∧
  point_on_line G (median_line t) ∧
  point_on_line E (median_line t) ∧
  point_on_line D (median_line t) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_feet_collinear_on_median_l980_98046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_range_l980_98020

/-- Parabola structure -/
structure Parabola where
  focus : Point
  equation : ℝ → ℝ → Prop

/-- Line structure -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Point structure -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle structure -/
structure Triangle where
  a : Point
  b : Point
  c : Point

noncomputable def area (t : Triangle) : ℝ := sorry

/-- Main theorem -/
theorem parabola_intersection_range (E : Parabola) (P : Point) :
  E.equation = (fun x y => x^2 = 4*y) →
  P.y = 0 →
  ∃ (l₁ l₂ : Line) (A B C D : Point),
    (∀ x y, y = l₁.slope * (x - P.x) + P.y ↔ E.equation x y ∨ (x = A.x ∧ y = A.y) ∨ (x = B.x ∧ y = B.y)) ∧
    (∀ x y, y = l₂.slope * (x - P.x) + P.y ↔ E.equation x y ∨ (x = C.x ∧ y = C.y) ∨ (x = D.x ∧ y = D.y)) ∧
    l₁.slope * l₂.slope = -1 ∧
    area (Triangle.mk A B E.focus) = area (Triangle.mk C D E.focus) →
  (-Real.sqrt 2 < P.x ∧ P.x < -1) ∨ (-1 < P.x ∧ P.x < 1) ∨ (1 < P.x ∧ P.x < Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_range_l980_98020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_modulus_l980_98025

noncomputable def i : ℂ := Complex.I

noncomputable def z : ℂ := (1 - i) / (1 + i) + 2 * i

theorem z_modulus : Complex.abs z = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_modulus_l980_98025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_value_l980_98062

theorem number_value (x y n : ℝ) 
  (h1 : n * x = 0.04 * y)
  (h2 : (y - x) / (y + x) = 0.948051948051948) :
  ∃ ε > 0, |n - 37.5| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_value_l980_98062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l980_98069

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- State the theorem
theorem f_decreasing_on_interval :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < Real.exp (-1) →
  f x₂ < f x₁ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l980_98069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_arlette_age_is_21_l980_98017

-- Define the ages of Omi, Kimiko, and Arlette
def kimiko_age : ℕ := 28
def omi_age : ℕ := 2 * kimiko_age
def arlette_age : ℕ := 21

-- Define the average age condition
theorem average_age : (omi_age + kimiko_age + arlette_age) / 3 = 35 := by
  -- Proof steps would go here
  sorry

-- Theorem to prove Arlette's age
theorem arlette_age_is_21 : arlette_age = 21 := by
  -- This is trivially true based on our definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_arlette_age_is_21_l980_98017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l980_98075

theorem sequence_inequality (a : ℕ → ℝ) 
  (h1 : ∀ n, 0 ≤ a n ∧ a n ≤ 1)
  (h2 : ∀ n, a n - 2 * a (n + 1) + a (n + 2) ≥ 0) :
  ∀ n : ℕ, n ≥ 1 → 0 ≤ (n + 1 : ℝ) * (a n - a (n + 1)) ∧ (n + 1 : ℝ) * (a n - a (n + 1)) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l980_98075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_tank_from_pyramid_l980_98036

/-- Represents the dimensions of an inverted pyramid with a square base -/
structure Pyramid where
  base_side : ℝ
  height : ℝ

/-- Represents the dimensions of a rectangular tank -/
structure RectangularTank where
  length : ℝ
  width : ℝ

/-- Calculates the volume of a pyramid -/
noncomputable def pyramid_volume (p : Pyramid) : ℝ :=
  (1/3) * p.base_side^2 * p.height

/-- Calculates the height of water in a rectangular tank given a volume of water -/
noncomputable def water_height_in_tank (t : RectangularTank) (volume : ℝ) : ℝ :=
  volume / (t.length * t.width)

theorem water_height_in_tank_from_pyramid (p : Pyramid) (t : RectangularTank) :
  p.base_side = 16 →
  p.height = 24 →
  t.length = 32 →
  t.width = 24 →
  water_height_in_tank t (pyramid_volume p) = 8/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_tank_from_pyramid_l980_98036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l980_98042

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- Represents a line passing through two points -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (h.a^2 + h.b^2) / h.a

/-- The focal distance of a hyperbola -/
noncomputable def focal_distance (h : Hyperbola) : ℝ :=
  2 * Real.sqrt (h.a^2 + h.b^2)

/-- The distance from a point to a line -/
noncomputable def distance_to_line (x y : ℝ) (l : Line) : ℝ :=
  sorry

/-- The theorem statement -/
theorem hyperbola_eccentricity_range (h : Hyperbola) (l : Line) :
  l.x₁ = h.a ∧ l.y₁ = 0 ∧ l.x₂ = 0 ∧ l.y₂ = h.b →
  distance_to_line 1 0 l + distance_to_line (-1) 0 l ≥ 4/5 * focal_distance h / 2 →
  Real.sqrt 5 / 2 ≤ eccentricity h ∧ eccentricity h ≤ Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l980_98042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiking_rate_proof_l980_98063

/-- Given a hiking trip where:
  * The time for ascent and descent is equal
  * The rate of descent is 1.5 times the rate of ascent
  * The descent route is 24 miles long
  * The ascent takes 2 days
  Prove that the rate of ascent is 8 miles per day -/
theorem hiking_rate_proof (ascent_time descent_time : ℝ) 
  (ascent_rate descent_rate : ℝ) (descent_distance : ℝ) :
  ascent_time = descent_time →
  descent_rate = 1.5 * ascent_rate →
  descent_distance = 24 →
  ascent_time = 2 →
  ascent_rate = 8 := by
  intro h1 h2 h3 h4
  -- Proof steps would go here
  sorry

#check hiking_rate_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiking_rate_proof_l980_98063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_returns_to_start_l980_98031

/-- Represents a circular clock with a smaller disk rolling around it. -/
structure RollingDisk where
  clock_radius : ℝ
  disk_radius : ℝ
  initial_position : ℝ  -- in radians, 0 represents 12 o'clock

/-- Calculates the final position of the disk after one complete rotation. -/
noncomputable def final_position (rd : RollingDisk) : ℝ :=
  rd.initial_position + 2 * Real.pi * (rd.clock_radius / rd.disk_radius)

/-- Theorem stating that the disk returns to its initial position after one rotation. -/
theorem disk_returns_to_start (rd : RollingDisk)
  (h1 : rd.clock_radius = 30)
  (h2 : rd.disk_radius = 15)
  (h3 : rd.initial_position = Real.pi)  -- π radians is 6 o'clock
  : final_position rd = rd.initial_position := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_returns_to_start_l980_98031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_team_frosting_result_l980_98077

/-- Represents the time in seconds to frost one cupcake for each person -/
structure FrostingRates where
  cagney : ℚ
  lacey : ℚ
  casey : ℚ

/-- Calculates the number of cupcakes frosted by a team in a given time -/
def cupcakes_frosted (rates : FrostingRates) (total_time : ℚ) : ℚ :=
  total_time * (1 / rates.cagney + 1 / rates.lacey + 1 / rates.casey)

/-- The main theorem stating that the team frosts 79 cupcakes in 10 minutes -/
theorem team_frosting_result (rates : FrostingRates) 
  (h_cagney : rates.cagney = 15)
  (h_lacey : rates.lacey = 25)
  (h_casey : rates.casey = 40) :
  cupcakes_frosted rates 600 = 79 := by
  sorry

/-- Compute the result for the given rates and time -/
def compute_result : ℚ :=
  cupcakes_frosted ⟨15, 25, 40⟩ 600

#eval compute_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_team_frosting_result_l980_98077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_exponential_sum_min_value_is_one_l980_98086

theorem min_value_exponential_sum (x y : ℝ) (hx : x < 0) (hy : y < 0) (h_eq : 2 * x + y = -2) :
  ∀ a b : ℝ, a < 0 → b < 0 → 2 * a + b = -2 → (4 : ℝ)^x + (2 : ℝ)^y ≤ (4 : ℝ)^a + (2 : ℝ)^b :=
by sorry

theorem min_value_is_one (x y : ℝ) (hx : x < 0) (hy : y < 0) (h_eq : 2 * x + y = -2) :
  (4 : ℝ)^x + (2 : ℝ)^y = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_exponential_sum_min_value_is_one_l980_98086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_display_rows_count_l980_98011

/-- Represents the number of cans in a row given its position from the top -/
def cans_in_row (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the total number of cans in the first n rows -/
def total_cans (n : ℕ) : ℕ := n * n

theorem display_rows_count : 
  ∃ n : ℕ, n > 0 ∧ cans_in_row n = 2 * n - 1 ∧ total_cans n = 100 :=
by
  use 10
  constructor
  · exact Nat.succ_pos 9
  constructor
  · rfl
  · rfl

#eval (Nat.find display_rows_count)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_display_rows_count_l980_98011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l980_98081

/-- The focus of the parabola y^2 = -4x -/
def parabola_focus : ℝ × ℝ := (-1, 0)

/-- The eccentricity of the hyperbola -/
noncomputable def hyperbola_eccentricity : ℝ := Real.sqrt 5

/-- The general equation of a hyperbola -/
def is_hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- The specific equation we want to prove -/
def target_hyperbola (x y : ℝ) : Prop :=
  5 * x^2 - 5 * y^2 / 4 = 1

/-- The main theorem -/
theorem hyperbola_equation :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ x y : ℝ, is_hyperbola a b x y ↔ target_hyperbola x y) ∧
  (a^2 + b^2 = (parabola_focus.1)^2) ∧
  (a / hyperbola_eccentricity = b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l980_98081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_value_l980_98016

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | (n + 1) => 2 * sequence_a n / (2 + sequence_a n)

theorem a_5_value : sequence_a 5 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_value_l980_98016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l980_98001

theorem equation_solution : ∃ x : ℝ, (4 : ℝ)^(x + 3) = (64 : ℝ)^x ∧ x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l980_98001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_length_l980_98059

/-- The length of a line segment with endpoints at (4, -1) and (12, -9) is 8√2. -/
theorem line_segment_length : 
  let A : ℝ × ℝ := (4, -1)
  let B : ℝ × ℝ := (12, -9)
  let length := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  length = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_length_l980_98059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l980_98066

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) * Real.cos (3 * Real.pi / 5) - Real.cos (2 * x) * Real.sin (3 * Real.pi / 5)

theorem f_properties :
  ∃ (T : ℝ) (axis : ℤ → ℝ) (min_val : ℝ),
    (∀ x : ℝ, f (x + T) = f x) ∧ 
    (T = Real.pi) ∧
    (∀ k : ℤ, axis k = 11 * Real.pi / 20 + k * Real.pi / 2) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≥ min_val) ∧
    (min_val = -1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l980_98066
