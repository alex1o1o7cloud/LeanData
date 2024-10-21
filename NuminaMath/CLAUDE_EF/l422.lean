import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_sum_l422_42236

open Complex

def complex_f (α γ : ℂ) : ℂ → ℂ := 
  λ z => (2 + I) * z^3 + (4 + I) * z^2 + α * z + γ

theorem min_abs_sum (α γ : ℂ) :
  (complex_f α γ 1).im = 0 →
  (complex_f α γ I).im = 0 →
  ∃ (α₀ γ₀ : ℂ), 
    abs α₀ + abs γ₀ = Real.sqrt 13 ∧
    ∀ (α' γ' : ℂ), 
      (complex_f α' γ' 1).im = 0 →
      (complex_f α' γ' I).im = 0 →
      abs α₀ + abs γ₀ ≤ abs α' + abs γ' :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_sum_l422_42236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l422_42291

/-- The constant term in the expansion of (2x³ - 1/x)^8 is 112 -/
theorem constant_term_expansion : ∃ (c : ℤ), c = 112 ∧ 
  ∃ (p : ℝ → ℝ), ∀ (x : ℝ), x ≠ 0 → (2 * x^3 - 1/x)^8 = c + x * (p x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l422_42291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_increasing_a_l422_42273

-- Define the piecewise function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 1 else a/x

-- State the theorem
theorem no_increasing_a : ¬ ∃ a : ℝ, ∀ x y : ℝ, x < y → f a x < f a y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_increasing_a_l422_42273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_obtuse_if_sin_cos_negative_l422_42255

-- Define a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = Real.pi
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Define what it means for a triangle to be obtuse-angled
def IsObtuseAngled (t : Triangle) : Prop :=
  t.A > Real.pi/2 ∨ t.B > Real.pi/2 ∨ t.C > Real.pi/2

-- State the theorem
theorem triangle_obtuse_if_sin_cos_negative (t : Triangle) :
  Real.sin t.A * Real.cos t.B < 0 → IsObtuseAngled t :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_obtuse_if_sin_cos_negative_l422_42255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l422_42280

theorem trigonometric_identity (α : ℝ) 
  (h1 : Real.tan α + (1 / Real.tan α) = 10 / 3) 
  (h2 : π / 4 < α ∧ α < π / 2) : 
  Real.sin (2 * α + π / 4) + 2 * Real.cos (π / 4) * (Real.cos α)^2 = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l422_42280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l422_42202

/-- The sum of the infinite series $\sum_{k = 1}^\infty \frac{k^2}{3^k}$ is equal to 2. -/
theorem infinite_series_sum : ∑' k : ℕ, (k : ℝ)^2 / 3^k = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l422_42202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l422_42275

noncomputable def f (x : ℝ) : ℝ := x^3 + (1/2) * x^2 - 4*x

theorem f_properties :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≥ f 1) ∧
  (∃ a b : ℝ, a ≠ b ∧ (∀ x : ℝ, f x ≥ f a ∨ f x ≥ f b) ∧
    (∀ c : ℝ, c ≠ a → c ≠ b → ¬(∀ x : ℝ, f x ≥ f c))) ∧
  (∀ x ∈ Set.Icc 0 2, f x ≤ f 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l422_42275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_functions_l422_42298

-- Define the functions
noncomputable def f (x : ℝ) := Real.tan (x / 2)
noncomputable def g (x : ℝ) := Real.sin (x / 2)
noncomputable def h (x : ℝ) := Real.sin (abs x)
noncomputable def k (x : ℝ) := Real.cos (abs x)

-- Define the property of having a period of 2π
def hasPeriod2Pi (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2 * Real.pi) = f x

-- Theorem statement
theorem periodic_functions :
  (hasPeriod2Pi f ∧ hasPeriod2Pi k) ∧
  (¬ hasPeriod2Pi g ∧ ¬ hasPeriod2Pi h) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_functions_l422_42298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_tree_time_l422_42219

/-- The time it takes for a train to pass a tree -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  train_length / train_speed_ms

/-- Theorem stating that a 560-meter long train traveling at 63 km/hr takes approximately 32 seconds to pass a tree -/
theorem train_passing_tree_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |train_passing_time 560 63 - 32| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_tree_time_l422_42219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_weight_on_bar_l422_42248

/-- The weight bench's maximum capacity in pounds -/
noncomputable def max_capacity : ℚ := 1000

/-- The safety margin as a percentage -/
noncomputable def safety_margin : ℚ := 20

/-- John's weight in pounds -/
noncomputable def john_weight : ℚ := 250

/-- The weight John can put on the bar in pounds -/
noncomputable def weight_on_bar : ℚ := max_capacity * (1 - safety_margin / 100) - john_weight

theorem john_weight_on_bar :
  weight_on_bar = 550 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_weight_on_bar_l422_42248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_square_area_l422_42274

/-- The equation of the circle -/
def circle_equation (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  3 * x^2 + 3 * y^2 - 15 * x + 9 * y + 27 = 0

/-- The circle is inscribed in a square with sides parallel to the coordinate axes -/
def inscribed_in_square (c : (ℝ × ℝ) → Prop) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ p, c p ↔ (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2) ∧
    (∀ p, c p → |p.1 - center.1| ≤ radius ∧ |p.2 - center.2| ≤ radius)

theorem circle_square_area :
  inscribed_in_square circle_equation →
  (∃ side : ℝ, side^2 = 25 ∧
    ∀ p, circle_equation p →
      |p.1 - (side / 2)| ≤ side / 2 ∧ |p.2 - (side / 2)| ≤ side / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_square_area_l422_42274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_seating_theorem_l422_42271

/-- The number of valid seating arrangements for N-1 guests around a circular table with 2N chairs. -/
def circularSeating (N : ℕ) : ℕ := N^2

/-- Represents the number of valid seating arrangements where no two guests are seated next to each other. -/
def number_of_valid_seating_arrangements (N : ℕ) : ℕ :=
  let total_seats := 2 * N
  let guests := N - 1
  -- Count valid arrangements where no two chosen seats are adjacent
  sorry

/-- Theorem stating that the circularSeating function correctly calculates the number of valid seating arrangements. -/
theorem circular_seating_theorem (N : ℕ) (h : N > 0) :
  circularSeating N = number_of_valid_seating_arrangements N :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_seating_theorem_l422_42271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_child_ticket_cost_l422_42201

/-- Represents the cost of tickets and zoo visits over two days -/
structure ZooVisit where
  child_cost : ℚ
  adult_cost : ℚ
  mon_children : ℕ
  mon_adults : ℕ
  tue_children : ℕ
  tue_adults : ℕ
  total_revenue : ℚ

/-- Theorem stating that given the specific conditions, the child ticket cost is $3 -/
theorem child_ticket_cost
  (z : ZooVisit)
  (h1 : z.mon_children = 7)
  (h2 : z.mon_adults = 5)
  (h3 : z.tue_children = 4)
  (h4 : z.tue_adults = 2)
  (h5 : z.adult_cost = 4)
  (h6 : z.total_revenue = 61)
  (h7 : z.child_cost * (z.mon_children + z.tue_children : ℚ) +
        z.adult_cost * (z.mon_adults + z.tue_adults : ℚ) = z.total_revenue) :
  z.child_cost = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_child_ticket_cost_l422_42201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_through_A_perpendicular_to_AB_l422_42299

/-- Given points A and B in 3D space, this theorem states that
    y - 3z + 6 = 0 is the equation of the plane passing through A
    and perpendicular to the line AB. -/
theorem plane_equation_through_A_perpendicular_to_AB :
  let A : ℝ × ℝ × ℝ := (1, 0, 2)
  let B : ℝ × ℝ × ℝ := (1, 1, -1)
  let plane_equation (x y z : ℝ) := y - 3*z + 6 = 0
  ∀ x y z : ℝ,
    (plane_equation x y z ↔
      ((x, y, z) - A) • (B - A) = 0 ∧
      (x, y, z) ≠ A)
:= by
  sorry

#check plane_equation_through_A_perpendicular_to_AB

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_through_A_perpendicular_to_AB_l422_42299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l422_42207

open Real

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (1 - sqrt 2 / 2 * t, 2 + sqrt 2 / 2 * t)

-- Define the curve C in polar form
noncomputable def curve_C (θ : ℝ) : ℝ :=
  4 * cos θ / (sin θ)^2

-- Theorem statement
theorem intersection_segment_length :
  ∃ (A B : ℝ × ℝ) (t₁ t₂ θ₁ θ₂ : ℝ),
    A = line_l t₁ ∧
    B = line_l t₂ ∧
    A.1^2 + A.2^2 = (curve_C θ₁)^2 ∧
    B.1^2 + B.2^2 = (curve_C θ₂)^2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 128 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l422_42207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l422_42261

def σ (n : ℕ) : ℕ := (Finset.range (n + 1)).sum (fun d => if n % d = 0 then d else 0)

def g (n : ℕ) : ℚ := (σ n - n) / n

theorem g_difference : g 320 - g 160 = 3 / 160 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l422_42261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_49_5_l422_42276

/-- A line in the xy-plane defined by y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- A trapezoid defined by four lines -/
structure Trapezoid where
  line1 : Line
  line2 : Line
  line3 : Line
  line4 : Line

/-- Calculate the area of a trapezoid -/
noncomputable def trapezoidArea (t : Trapezoid) : ℝ := sorry

/-- The specific trapezoid from the problem -/
def problemTrapezoid : Trapezoid :=
  { line1 := { m := 1, b := 2 }  -- y = x + 2
    line2 := { m := 0, b := 12 } -- y = 12
    line3 := { m := 0, b := 3 }  -- y = 3
    line4 := { m := 0, b := 0 } } -- x = 0 (y-axis)

theorem trapezoid_area_is_49_5 :
  trapezoidArea problemTrapezoid = 49.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_49_5_l422_42276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l422_42213

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapezium_area (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides of lengths 20 cm and 18 cm, 
    and a distance of 10 cm between them, is equal to 190 square centimeters. -/
theorem trapezium_area_example : trapezium_area 20 18 10 = 190 := by
  -- Unfold the definition of trapezium_area
  unfold trapezium_area
  -- Simplify the arithmetic
  simp [add_mul, mul_div_assoc]
  -- Check that the result is equal to 190
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l422_42213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_triple_equality_l422_42282

-- Define the piecewise function h
noncomputable def h (x : ℝ) : ℝ :=
  if x ≤ 0 then -x else 3 * x - 50

-- State the theorem
theorem h_triple_equality (a : ℝ) :
  a < 0 ∧ h (h (h 15)) = h (h (h a)) ↔ a = -55/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_triple_equality_l422_42282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l422_42286

/-- The distance from a point (a, b) to the line Ax + By + C = 0 -/
noncomputable def distancePointToLine (a b A B C : ℝ) : ℝ :=
  (abs (A * a + B * b + C)) / (Real.sqrt (A^2 + B^2))

/-- Theorem: If the line x + y + m = 0 is tangent to the circle x^2 + y^2 = m, then m = 2 -/
theorem line_tangent_to_circle (m : ℝ) : 
  (∀ x y : ℝ, x + y + m = 0 → x^2 + y^2 ≤ m) ∧ 
  (∃ x y : ℝ, x + y + m = 0 ∧ x^2 + y^2 = m) → 
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l422_42286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_box_diagonal_l422_42204

/-- The distance between midpoints of opposite sides of a square face in a box 
    formed by four regular pentagons with side length 1 and a square --/
theorem pentagon_box_diagonal (
  pentagon_side_length : ℝ) 
  (h_pentagon_side : pentagon_side_length = 1) 
  (A B : ℝ × ℝ) 
  (h_square_side : ∀ (side : ℝ × ℝ → ℝ × ℝ), 
    (side (0, 0)).1 = 2 ∧ (side (0, 0)).2 = 0 ∨ 
    (side (0, 0)).1 = 0 ∧ (side (0, 0)).2 = 2) 
  (h_A_midpoint : A = (1, 0)) 
  (h_B_midpoint : B = (0, 1)) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_box_diagonal_l422_42204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_proof_l422_42243

def binomial_sum (n : ℕ) : ℕ → ℕ
  | 0 => 1
  | k + 1 => binomial_sum n k + n.choose (k + 1) * 2^k

def a : ℕ := 2 + binomial_sum 20 19

theorem congruence_proof (b : ℤ) (h : b ≡ a [ZMOD 10]) : b = 2012 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_proof_l422_42243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_theorem_l422_42239

-- Define the numbers
noncomputable def a : ℝ := 2^(0.3 : ℝ)
def b : ℝ := 1
noncomputable def c : ℝ := (0.3 : ℝ)^2

-- State the theorem
theorem ordering_theorem : a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_theorem_l422_42239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l422_42217

noncomputable def f (x a : ℝ) : ℝ := x * Real.log x - 2 * x + a^2 - a

theorem a_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 (Real.exp 2), f x a ≤ 0) → a ∈ Set.Icc 0 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l422_42217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_chord_length_circle_C₂_equation_l₁_always_intersects_C₁_shortest_chord_l₁_equation_l422_42250

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 4 = 0

-- Define the line l
def l (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Define points E and F
def E : ℝ × ℝ := (1, -3)
def F : ℝ × ℝ := (0, 4)

-- Define the line parallel to the common chord of C₁ and C₂
def common_chord_parallel (x y : ℝ) : Prop := 2*x + y + 1 = 0

-- Define the line l₁
def l₁ (lambda x y : ℝ) : Prop := 2*lambda*x - 2*y + 3 - lambda = 0

theorem circle_intersection_chord_length :
  ∃ (A B : ℝ × ℝ), C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧ l A.1 A.2 ∧ l B.1 B.2 ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 5 / 5 :=
sorry

theorem circle_C₂_equation :
  ∃ (C₂ : ℝ → ℝ → Prop),
  (C₂ E.1 E.2 ∧ C₂ F.1 F.2) ∧
  (∀ x y : ℝ, C₂ x y ↔ x^2 + y^2 + 6*x - 16 = 0) ∧
  (∃ a b : ℝ, ∀ x y : ℝ, (C₁ x y ∧ C₂ x y) → common_chord_parallel x y) :=
sorry

theorem l₁_always_intersects_C₁ :
  ∀ lambda : ℝ, ∃ (P Q : ℝ × ℝ), P ≠ Q ∧ C₁ P.1 P.2 ∧ C₁ Q.1 Q.2 ∧ l₁ lambda P.1 P.2 ∧ l₁ lambda Q.1 Q.2 :=
sorry

theorem shortest_chord_l₁_equation :
  ∃ lambda : ℝ, ∀ x y : ℝ, l₁ lambda x y ↔ x + y - 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_chord_length_circle_C₂_equation_l₁_always_intersects_C₁_shortest_chord_l₁_equation_l422_42250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_to_circle_l422_42289

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 3*x - 4*y + 6 = 0

-- Define the tangent line
def tangent_line (x : ℝ) : Prop := x = -2

-- Theorem statement
theorem tangent_to_circle :
  ∃ (x₀ y₀ : ℝ), circle_eq x₀ y₀ ∧
  (∀ (x y : ℝ), circle_eq x y → (x - x₀)^2 + (y - y₀)^2 ≥ (x₀ + 2)^2) ∧
  (∃ (y : ℝ), circle_eq (-2) y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_to_circle_l422_42289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_exterior_pentagon_l422_42279

/-- A regular pentagon with side length s -/
structure RegularPentagon where
  s : ℝ
  s_pos : 0 < s

/-- An equilateral triangle -/
structure EquilateralTriangle where
  side : ℝ
  side_pos : 0 < side

/-- The center of an equilateral triangle -/
noncomputable def center (t : EquilateralTriangle) : ℝ × ℝ := sorry

/-- The area of a pentagon -/
noncomputable def area_pentagon (p : RegularPentagon) : ℝ := sorry

/-- Construct an equilateral triangle on each side of the pentagon -/
noncomputable def exterior_triangles (p : RegularPentagon) : Fin 5 → EquilateralTriangle := sorry

/-- The points P, Q, R, S, T as centers of the exterior triangles -/
noncomputable def exterior_points (p : RegularPentagon) : Fin 5 → ℝ × ℝ := 
  λ i ↦ center (exterior_triangles p i)

/-- The pentagon PQRST formed by connecting the exterior points -/
noncomputable def exterior_pentagon (p : RegularPentagon) : RegularPentagon := sorry

/-- The main theorem -/
theorem area_ratio_exterior_pentagon (p : RegularPentagon) : 
  area_pentagon (exterior_pentagon p) / area_pentagon p = (3 + 2 * Real.sqrt 3)^2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_exterior_pentagon_l422_42279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_a_and_b_range_of_m_l422_42237

-- Define the condition for a and b
def satisfies_condition (a b : ℝ) : Prop :=
  ∀ x : ℝ, |x^2 + a*x + b| ≤ |2*x^2 - 4*x - 16|

-- Theorem 1: Characterization of a and b that satisfy the condition
theorem characterization_of_a_and_b :
  ∀ a b : ℝ, satisfies_condition a b ↔ (a = -2 ∧ b = -8) :=
sorry

-- Theorem 2: Range of m given the condition on a and b
theorem range_of_m (a b : ℝ) (h : satisfies_condition a b) :
  ∀ m : ℝ, (∀ x > 2, x^2 + a*x + b ≥ (m + 2)*x - m - 15) ↔ m ≤ -11 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_a_and_b_range_of_m_l422_42237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_more_heads_half_l422_42232

/-- Represents the number of coin tosses -/
def n : ℕ := sorry

/-- Represents the outcome of a coin toss -/
inductive CoinToss
| Heads
| Tails

/-- Represents a sequence of coin tosses -/
def TossSequence := List CoinToss

/-- Returns the number of heads in a toss sequence -/
def countHeads : TossSequence → ℕ
| [] => 0
| (CoinToss.Heads :: rest) => 1 + countHeads rest
| (CoinToss.Tails :: rest) => countHeads rest

/-- Represents the probability space of all possible outcomes -/
def ProbabilitySpace := TossSequence × TossSequence

/-- The probability of an event in the probability space -/
noncomputable def probability (event : ProbabilitySpace → Prop) : ℝ := sorry

/-- The event where the first sequence has more heads than the second -/
def moreHeadsEvent (p : ProbabilitySpace) : Prop :=
  countHeads p.1 > countHeads p.2

/-- The main theorem: probability of getting more heads in n+1 tosses compared to n tosses is 1/2 -/
theorem prob_more_heads_half :
  probability moreHeadsEvent = 1 / 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_more_heads_half_l422_42232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_painting_cost_l422_42278

-- Define the cube and paint properties
def cube_edge_length : ℝ := 10
def color_A_cost_per_quart : ℝ := 3.20
def color_B_cost_per_quart : ℝ := 4.30
def color_A_coverage_per_quart : ℝ := 10
def color_B_coverage_per_quart : ℝ := 8

-- Define the painting pattern
def faces_with_color_A : ℝ := 4
def faces_with_color_B : ℝ := 2

-- Theorem statement
theorem cube_painting_cost : 
  let face_area := cube_edge_length * cube_edge_length
  let total_area_A := face_area * faces_with_color_A
  let total_area_B := face_area * faces_with_color_B
  let quarts_A := total_area_A / color_A_coverage_per_quart
  let quarts_B := total_area_B / color_B_coverage_per_quart
  let cost_A := quarts_A * color_A_cost_per_quart
  let cost_B := quarts_B * color_B_cost_per_quart
  cost_A + cost_B = 235.50 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_painting_cost_l422_42278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_equation_l422_42210

/-- Represents a hyperbola with parameters a and b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := 
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The equation of the asymptotes of a hyperbola -/
def asymptote_equation (h : Hyperbola) : ℝ → ℝ → Prop := 
  fun x y => y^2 = (h.b^2 / h.a^2) * x^2

theorem hyperbola_asymptotes_equation (h : Hyperbola) 
  (h_ecc : eccentricity h = 2) :
  asymptote_equation h = fun x y => y^2 = 3 * x^2 := by
  sorry

#check hyperbola_asymptotes_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_equation_l422_42210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_theorem_l422_42238

-- Define the variables
variable (a b c d : ℝ)

-- Define the conditions
def condition1 (a b c d : ℝ) : Prop := c = 0.25 * a
def condition2 (a b c d : ℝ) : Prop := c = 0.1 * b
def condition3 (a b c d : ℝ) : Prop := d = 0.5 * b

-- State the theorem
theorem percentage_theorem
  (h1 : condition1 a b c d)
  (h2 : condition2 a b c d)
  (h3 : condition3 a b c d)
  (h4 : a ≠ 0) -- To avoid division by zero
  (h5 : c ≠ 0) -- To avoid division by zero
  : (d * b) / (a * c) * 100 = 1250 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_theorem_l422_42238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_acquainted_pairs_is_minimum_acquainted_pairs_l422_42220

/-- Represents a village with residents and their acquaintances. -/
structure Village where
  residents : ℕ
  acquainted : Fin residents → Fin residents → Prop

/-- Checks if a given arrangement of 5 residents satisfies the round table condition. -/
def satisfiesRoundTableCondition (v : Village) (arrangement : Fin 5 → Fin v.residents) : Prop :=
  ∀ i : Fin 5, v.acquainted (arrangement i) (arrangement ((i + 1) % 5)) ∧
               v.acquainted (arrangement i) (arrangement ((i + 4) % 5))

/-- The round table property for the entire village. -/
def hasRoundTableProperty (v : Village) : Prop :=
  ∀ subset : Fin 5 → Fin v.residents, ∃ arrangement : Fin 5 → Fin v.residents,
    (∀ i : Fin 5, arrangement i ∈ Set.range subset) ∧
    satisfiesRoundTableCondition v arrangement

/-- Counts the number of acquainted pairs in the village. -/
def countAcquaintedPairs (v : Village) : ℕ :=
  (v.residents * (v.residents - 3)) / 2

/-- The main theorem stating the minimum number of acquainted pairs. -/
theorem min_acquainted_pairs (v : Village) (h : v.residents = 240) (hp : hasRoundTableProperty v) :
  countAcquaintedPairs v = 28440 := by
  sorry

/-- Proof that this is indeed the minimum number of acquainted pairs. -/
theorem is_minimum_acquainted_pairs (v : Village) (h : v.residents = 240) (hp : hasRoundTableProperty v) :
  ∀ v' : Village, v'.residents = 240 → hasRoundTableProperty v' →
    countAcquaintedPairs v' ≥ countAcquaintedPairs v := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_acquainted_pairs_is_minimum_acquainted_pairs_l422_42220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_of_zeros_l422_42287

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := -1/3 + Real.sin x

-- State the theorem
theorem cos_difference_of_zeros (x₁ x₂ : ℝ) :
  x₁ ∈ Set.Icc 0 Real.pi →
  x₂ ∈ Set.Icc 0 Real.pi →
  x₁ ≠ x₂ →
  f x₁ = 0 →
  f x₂ = 0 →
  Real.cos (x₁ - x₂) = -7/9 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_of_zeros_l422_42287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_line_m_value_l422_42265

/-- A line intersecting a circle -/
structure IntersectingLine where
  m : ℝ
  intersects_circle : ∃ (x y : ℝ), x - m * y + 9 = 0 ∧ x^2 + y^2 + 2*x - 24 = 0
  segment_length : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁ - m * y₁ + 9 = 0 ∧ x₁^2 + y₁^2 + 2*x₁ - 24 = 0) ∧
    (x₂ - m * y₂ + 9 = 0 ∧ x₂^2 + y₂^2 + 2*x₂ - 24 = 0) ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 36

theorem intersecting_line_m_value (l : IntersectingLine) : l.m = Real.sqrt 3 ∨ l.m = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_line_m_value_l422_42265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_interval_l422_42292

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - 16 * Real.log x

theorem monotone_decreasing_interval (a : ℝ) : 
  (∀ x ∈ Set.Icc (a - 1) (a + 2), StrictMonoOn f (Set.Icc (a - 1) (a + 2))) → 
  a ∈ Set.Ioo 1 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_interval_l422_42292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_removed_numbers_proof_l422_42225

theorem removed_numbers_proof (n : ℕ) (h1 : n = 1993) :
  ∃ x : ℕ,
    x > 0 ∧
    x < n - 1 ∧
    (((n * (n + 1)) / 2 - (3 * x + 3)) / (n - 3) : ℚ).floor = ((n * (n + 1)) / 2 - (3 * x + 3)) / (n - 3) →
    x = 996 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_removed_numbers_proof_l422_42225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_max_a_for_increasing_f_l422_42203

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - x^2 - a*x

-- Part I
theorem tangent_line_at_zero (a b : ℝ) :
  (∀ x, (deriv (f a)) 0 * x + (f a 0) = 2*x + b) → a = -1 ∧ b = 1 := by sorry

-- Part II
theorem max_a_for_increasing_f :
  (∃ a, Monotone (f a)) ∧ 
  (∀ a, Monotone (f a) → a ≤ 2 - 2 * Real.log 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_max_a_for_increasing_f_l422_42203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_always_umbrella_best_l422_42267

/-- Represents the expected unpleasantness of a umbrella-carrying strategy -/
def ExpectedUnpleasantness (x : ℝ) : Prop := x ≥ 0

/-- Probability of rain on any given day -/
noncomputable def P_rain : ℝ := 1/3

/-- Probability of no rain on any given day -/
noncomputable def P_no_rain : ℝ := 1 - P_rain

/-- Probability that the forecast is rain given that it actually rains -/
noncomputable def P_forecast_rain_given_rain : ℝ := 0.8

/-- Probability that the forecast is rain given that it does not rain -/
noncomputable def P_forecast_rain_given_no_rain : ℝ := 0.5

/-- Unpleasantness of getting wet -/
noncomputable def unpleasantness_wet : ℝ := 2

/-- Unpleasantness of carrying umbrella unnecessarily -/
noncomputable def unpleasantness_carry : ℝ := 1

/-- Expected unpleasantness when always taking the umbrella -/
noncomputable def EU_always_umbrella : ℝ := P_no_rain * unpleasantness_carry

/-- Expected unpleasantness when never taking the umbrella -/
noncomputable def EU_never_umbrella : ℝ := P_rain * unpleasantness_wet

/-- Expected unpleasantness when following the forecast -/
noncomputable def EU_follow_forecast : ℝ := 
  let P_forecast_rain := P_forecast_rain_given_rain * P_rain + P_forecast_rain_given_no_rain * P_no_rain
  let P_forecast_no_rain := 1 - P_forecast_rain
  let P_rain_given_forecast_rain := (P_forecast_rain_given_rain * P_rain) / P_forecast_rain
  let P_no_rain_given_forecast_rain := 1 - P_rain_given_forecast_rain
  let P_rain_given_forecast_no_rain := (P_forecast_rain_given_rain * P_rain) / P_forecast_no_rain
  P_forecast_rain * (P_rain_given_forecast_rain * unpleasantness_carry + P_no_rain_given_forecast_rain * unpleasantness_carry) +
  P_forecast_no_rain * (P_rain_given_forecast_no_rain * unpleasantness_wet)

/-- Theorem stating that always taking the umbrella is the best strategy -/
theorem always_umbrella_best : 
  ExpectedUnpleasantness EU_always_umbrella ∧ 
  EU_always_umbrella ≤ EU_never_umbrella ∧ 
  EU_always_umbrella ≤ EU_follow_forecast := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_always_umbrella_best_l422_42267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sara_movie_expenses_l422_42252

def movie_theater_ticket_price : ℚ := 10.62
def number_of_tickets : ℕ := 2
def rented_movie_price : ℚ := 1.59
def bought_movie_price : ℚ := 13.95

theorem sara_movie_expenses : 
  (movie_theater_ticket_price * number_of_tickets + rented_movie_price + bought_movie_price) = 36.78 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sara_movie_expenses_l422_42252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_great_circle_sum_equality_l422_42246

/-- Represents a point on a sphere where great circles intersect -/
structure IntersectionPoint where
  value : ℕ

/-- Represents a great circle on a sphere -/
structure GreatCircle where
  points : List IntersectionPoint

/-- Represents the arrangement of numbers on a sphere -/
structure Arrangement where
  n : ℕ
  circles : List GreatCircle

/-- The sum of values on a great circle -/
def circleSum (circle : GreatCircle) : ℕ :=
  (circle.points.map (λ p => p.value)).sum

/-- Theorem: There exists an arrangement of numbers on n great circles
    such that the sum on each circle is equal to [n(n-1)+1](n-1) -/
theorem great_circle_sum_equality (n : ℕ) :
  ∃ (arr : Arrangement),
    arr.n = n ∧
    arr.circles.length = n ∧
    (∀ c ∈ arr.circles, c.points.length = n - 1) ∧
    (∀ c ∈ arr.circles, circleSum c = (n * (n - 1) + 1) * (n - 1)) ∧
    (∀ i : Fin (n * (n - 1)), ∃ c ∈ arr.circles, ∃ p ∈ c.points, p.value = i.val + 1) :=
  sorry

#check great_circle_sum_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_great_circle_sum_equality_l422_42246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repunit_divisibility_l422_42297

theorem repunit_divisibility (m : ℕ) (h : Nat.Coprime m 10) :
  ∃ n : ℕ, m ∣ (10^n - 1) / 9 ∧ ∃ f : ℕ → ℕ, Monotone f ∧ ∀ k, m ∣ (10^(f k) - 1) / 9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repunit_divisibility_l422_42297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_truth_values_l422_42288

-- Define the propositions
def prop1 : Prop := ¬(∀ x : ℝ, x > 2 → x > 0)
def prop2 : Prop := ¬(∀ a : ℝ, a > 0 → StrictMono (fun x => a^x))
def prop3 : Prop := (∀ x : ℝ, Real.sin x = Real.sin (x + Real.pi)) ∨ 
                    (∀ x : ℝ, Real.sin (2*x) = Real.sin (2*(x + 2*Real.pi)))
def prop4 : Prop := ¬(∀ x y : ℝ, x*y = 0 → x^2 + y^2 = 0)

-- Theorem stating the truth values of the propositions
theorem proposition_truth_values : 
  ¬prop1 ∧ prop2 ∧ prop3 ∧ prop4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_truth_values_l422_42288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elaine_rent_percentage_l422_42258

/-- Represents Elaine's financial situation over two years -/
structure ElaineFinances where
  last_year_earnings : ℚ
  last_year_rent_percentage : ℚ
  earnings_increase_percentage : ℚ
  rent_increase_percentage : ℚ

/-- Calculates the percentage of earnings spent on rent this year -/
def rent_percentage_this_year (e : ElaineFinances) : ℚ :=
  (e.rent_increase_percentage * e.last_year_rent_percentage) /
  (100 + e.earnings_increase_percentage)

/-- Theorem stating that given the conditions in the problem, 
    Elaine spent 30% of her earnings on rent this year -/
theorem elaine_rent_percentage 
  (e : ElaineFinances)
  (h1 : e.last_year_rent_percentage = 10)
  (h2 : e.earnings_increase_percentage = 15)
  (h3 : e.rent_increase_percentage = 345) :
  rent_percentage_this_year e = 30 := by
  sorry

#eval rent_percentage_this_year {
  last_year_earnings := 100,
  last_year_rent_percentage := 10,
  earnings_increase_percentage := 15,
  rent_increase_percentage := 345
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elaine_rent_percentage_l422_42258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_through_origin_g_non_negative_iff_k_eq_e_l422_42229

-- Define the function f(x) = e^x
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Define the function g(x) = e^x - kx + k - e
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := Real.exp x - k * x + k - Real.exp 1

-- Statement for the tangent line problem
theorem tangent_line_through_origin :
  ∃ t : ℝ, (f t - 0) / (t - 0) = Real.exp 1 ∧
           ∀ x : ℝ, f t + (Real.exp 1) * (x - t) = Real.exp 1 * x :=
by
  sorry

-- Statement for the non-negative g(x) problem
theorem g_non_negative_iff_k_eq_e :
  ∃! k : ℝ, (∀ x : ℝ, g k x ≥ 0) ∧ k = Real.exp 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_through_origin_g_non_negative_iff_k_eq_e_l422_42229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_price_change_l422_42200

theorem tv_price_change (P : ℝ) (h : P > 0) : 
  P * (1 - 0.1) * (1 + 0.1) = P * 0.99 :=
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_price_change_l422_42200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_children_in_families_with_children_l422_42284

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (average_children_all : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 15)
  (h2 : average_children_all = 3)
  (h3 : childless_families = 3) :
  let total_children : ℚ := (total_families : ℚ) * average_children_all
  let families_with_children : ℕ := total_families - childless_families
  total_children / (families_with_children : ℚ) = 3.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_children_in_families_with_children_l422_42284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_bucket_sum_l422_42223

theorem fruit_bucket_sum 
  (bucket_a bucket_b bucket_c : ℕ)
  (h1 : bucket_a = bucket_b + 4)
  (h2 : bucket_b = bucket_c + 3)
  (h3 : bucket_c = 9) :
  bucket_a + bucket_b + bucket_c = 37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_bucket_sum_l422_42223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l422_42263

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x + 5

noncomputable def tangent_slope (x : ℝ) : ℝ := 3*x^2 - 6*x + 2

noncomputable def slope_angle (x : ℝ) : ℝ := Real.arctan (tangent_slope x)

theorem slope_angle_range :
  Set.range slope_angle = {α | 0 ≤ α ∧ α < π/2} ∪ {α | 3*π/4 ≤ α ∧ α < π} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l422_42263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_second_tap_empties_in_four_hours_l422_42253

/-- Represents the time (in hours) it takes for a tap to fill or empty the cistern -/
def Time := ℝ

/-- The time it takes for the first tap to fill the cistern -/
def fill_time : Time := (2 : ℝ)

/-- The time it takes to fill the cistern when both taps are open -/
def both_taps_time : Time := (4 : ℝ)

/-- The time it takes for the second tap to empty the cistern -/
def empty_time : Time := (4 : ℝ)

/-- 
Theorem stating that given the fill time of the first tap and the time it takes to fill
the cistern with both taps open, the time it takes for the second tap to empty the cistern is 4 hours.
-/
theorem second_tap_empties_in_four_hours 
  (h1 : fill_time = (2 : ℝ))
  (h2 : both_taps_time = (4 : ℝ)) :
  empty_time = (4 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_second_tap_empties_in_four_hours_l422_42253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_l422_42230

/-- An ellipse with given properties and its intersecting line -/
theorem ellipse_and_line (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  ∃ (C : Set (ℝ × ℝ)) (F1 F2 A B : ℝ × ℝ),
    -- Ellipse equation
    C = {(x, y) | x^2 / a^2 + y^2 / b^2 = 1} ∧
    -- Distance between foci
    ‖F1 - F2‖ = 2 ∧
    -- Ellipse passes through (-1, -√2/2)
    (-1, -Real.sqrt 2 / 2) ∈ C ∧
    -- A and B are on the ellipse
    A ∈ C ∧ B ∈ C ∧
    -- H is on the line AB
    ∃ (t : ℝ), (1 - t) • A + t • B = (-2, 0) ∧
    -- AF1 ⟂ BF1
    (A - F1) • (B - F1) = 0 →
    -- Standard equation of the ellipse
    C = {(x, y) | x^2 / 2 + y^2 = 1} ∧
    -- Equation of line AB
    ∃ (s : ℝ), s = 1 ∨ s = -1 ∧ A.1 + s * 2 * A.2 + 2 = 0 ∧ B.1 + s * 2 * B.2 + 2 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_l422_42230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_d_l422_42235

/-- The smallest positive value of d that satisfies the distance equation --/
noncomputable def smallest_d : ℝ :=
  (8 + Real.sqrt 5824) / 30

/-- The point's x-coordinate --/
noncomputable def x : ℝ := 4 * Real.sqrt 5

/-- The point's y-coordinate in terms of d --/
def y (d : ℝ) : ℝ := d + 4

/-- The distance equation --/
def distance_equation (d : ℝ) : Prop :=
  Real.sqrt (x^2 + y d^2) = 4 * d

theorem smallest_positive_d :
  ∀ d : ℝ, d > 0 → distance_equation d → d ≥ smallest_d := by
  sorry

#check smallest_positive_d

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_d_l422_42235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_a_and_b_worked_together_for_two_days_l422_42221

/-- The number of days A needs to complete the work alone -/
noncomputable def a_days : ℝ := 5

/-- The number of days B needs to complete the work alone -/
noncomputable def b_days : ℝ := 16

/-- The number of days B works alone after A leaves -/
noncomputable def b_alone_days : ℝ := 6

/-- The fraction of work completed by A in one day -/
noncomputable def a_rate : ℝ := 1 / a_days

/-- The fraction of work completed by B in one day -/
noncomputable def b_rate : ℝ := 1 / b_days

/-- The combined rate of work when A and B work together -/
noncomputable def combined_rate : ℝ := a_rate + b_rate

/-- The number of days A and B worked together -/
noncomputable def days_worked_together : ℝ := 2

theorem work_completion :
  combined_rate * days_worked_together + b_rate * b_alone_days = 1 := by
  sorry

/-- The main theorem proving that A and B worked together for 2 days -/
theorem a_and_b_worked_together_for_two_days :
  ∃ (x : ℝ), x = days_worked_together ∧ 
  combined_rate * x + b_rate * b_alone_days = 1 ∧
  x ≥ 0 ∧ x < 3 ∧ x = ⌊x⌋ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_a_and_b_worked_together_for_two_days_l422_42221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_2n_divisible_by_3_l422_42260

/-- S(n) represents the number of distinct paths from (0,0) to (n,n) 
    that only move right, up, or diagonally up-right, 
    and never go above the line y = x -/
def S : ℕ → ℕ := sorry

/-- Theorem: For all positive integers n, S(2n) is divisible by 3 -/
theorem S_2n_divisible_by_3 (n : ℕ) (h : n > 0) : 
  3 ∣ S (2 * n) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_2n_divisible_by_3_l422_42260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_usage_related_to_gender_l422_42211

/-- Represents the contingency table data --/
structure ContingencyTable :=
  (male_using : ℕ)
  (male_not_using : ℕ)
  (female_using : ℕ)
  (female_not_using : ℕ)

/-- Calculates K^2 value --/
noncomputable def calculate_k_squared (ct : ContingencyTable) : ℝ :=
  let n := ct.male_using + ct.male_not_using + ct.female_using + ct.female_not_using
  let a := ct.male_using
  let b := ct.male_not_using
  let c := ct.female_using
  let d := ct.female_not_using
  (n * (a * d - b * c)^2 : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The theorem to be proved --/
theorem phone_usage_related_to_gender (ct : ContingencyTable) 
  (h1 : ct.male_using = 40)
  (h2 : ct.male_not_using = 15)
  (h3 : ct.female_using = 20)
  (h4 : ct.female_not_using = 25)
  (critical_value : ℝ)
  (h5 : critical_value = 7.879) : 
  calculate_k_squared ct > critical_value := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_usage_related_to_gender_l422_42211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_exponential_sum_l422_42262

theorem min_value_exponential_sum (x y : ℝ) (h : x + 2*y = 2) :
  ∃ (m : ℝ), m = 6 ∧ ∀ (a b : ℝ), a + 2*b = 2 → (3 : ℝ)^x + (9 : ℝ)^y ≤ (3 : ℝ)^a + (9 : ℝ)^b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_exponential_sum_l422_42262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_array_multiples_of_31_l422_42272

/-- Represents the triangular array of numbers -/
def TriangularArray : Type := Nat → Nat → Nat

/-- The first row consists of odd integers from 1 to 199 -/
def firstRowOddIntegers (a : TriangularArray) : Prop :=
  ∀ k, a 1 k = 2 * k - 1 ∧ 1 ≤ a 1 k ∧ a 1 k ≤ 199

/-- Each row below the first has one fewer entry than the row above it -/
def decreasingSizeRows (a : TriangularArray) : Prop :=
  ∀ n k, n > 1 → a n k ≠ 0 → a (n - 1) k ≠ 0 ∧ a (n - 1) (k + 1) ≠ 0

/-- The bottom row has a single entry -/
def bottomRowSingleEntry (a : TriangularArray) : Prop :=
  ∃ n, a n 1 ≠ 0 ∧ a n 2 = 0

/-- Each entry equals the sum of the two entries diagonally above it -/
def entrySum (a : TriangularArray) : Prop :=
  ∀ n k, n > 1 → a n k = a (n - 1) k + a (n - 1) (k + 1)

/-- The main theorem stating that there are exactly 32 entries that are multiples of 31 -/
theorem triangular_array_multiples_of_31
  (a : TriangularArray)
  (h1 : firstRowOddIntegers a)
  (h2 : decreasingSizeRows a)
  (h3 : bottomRowSingleEntry a)
  (h4 : entrySum a) :
  (Finset.sum (Finset.range 100) (λ n ↦ Finset.sum (Finset.range (101 - n)) (λ k ↦ if 31 ∣ a n k then 1 else 0))) = 32 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_array_multiples_of_31_l422_42272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_minimum_value_zero_range_of_a_for_nonnegative_exponential_logarithm_inequality_l422_42295

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - 1 - x - a * x^2

theorem tangent_line_at_one : 
  ∀ x, f 0 x = Real.exp x - x - 2 := by sorry

theorem minimum_value_zero :
  (∀ x, f 0 x ≥ 0) ∧ ∃ x₀, f 0 x₀ = 0 := by sorry

theorem range_of_a_for_nonnegative :
  ∀ a, (∀ x ≥ 0, f a x ≥ 0) ↔ a ≤ (1/2) := by sorry

theorem exponential_logarithm_inequality :
  ∀ x > 0, (Real.exp x - 1) * Real.log (x + 1) > x^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_minimum_value_zero_range_of_a_for_nonnegative_exponential_logarithm_inequality_l422_42295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_in_equilateral_triangle_l422_42266

/-- The area of the hexagon formed by the intersection of three triangles in an equilateral triangle -/
theorem hexagon_area_in_equilateral_triangle (x : ℝ) (h1 : 0 < x) (h2 : x < 0.5) :
  (8 * x^2 - 8 * x + 2) / ((2 - x) * (x + 1)) * (Real.sqrt 3 / 4) =
  (8 * x^2 - 8 * x + 2) / ((2 - x) * (x + 1)) * (Real.sqrt 3 / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_in_equilateral_triangle_l422_42266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_sum_gt_100_l422_42231

/-- The number of pairs (a, b) where a and b are integers from 1 to 100 (inclusive) 
    and a + b > 100 is equal to 5050. -/
theorem count_pairs_sum_gt_100 : 
  (Finset.sum (Finset.range 100) (λ a => 
    (Finset.filter (λ b => a + 1 + b + 1 > 100) (Finset.range 100)).card
  )) = 5050 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_sum_gt_100_l422_42231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_arithmetic_sequence_length_l422_42214

/-- The number of terms in an arithmetic sequence -/
def arithmetic_sequence_length (a₁ : ℤ) (aₙ : ℤ) (d : ℕ) : ℕ :=
  (aₙ - a₁).natAbs / d + 1

/-- Theorem: The arithmetic sequence with first term -48, last term 72, and common difference 6 has 21 terms -/
theorem specific_arithmetic_sequence_length :
  arithmetic_sequence_length (-48) 72 6 = 21 := by
  rfl

#eval arithmetic_sequence_length (-48) 72 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_arithmetic_sequence_length_l422_42214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_study_days_theorem_l422_42242

/-- Represents the study habits of a group of students -/
structure StudyGroup where
  students : ℕ
  days : ℕ

/-- Calculates the total number of study days for a group -/
def totalDays (group : StudyGroup) : ℕ := group.students * group.days

/-- Represents Mrs. Thompson's class study data -/
def classData : List StudyGroup := [
  ⟨2, 3⟩,
  ⟨4, 5⟩,
  ⟨9, 8⟩,
  ⟨5, 10⟩,
  ⟨3, 15⟩,
  ⟨2, 20⟩
]

/-- Calculates the total number of students in the class -/
def totalStudents : ℕ := (classData.map (·.students)).sum

/-- Calculates the total number of study days for the entire class -/
def totalClassDays : ℕ := (classData.map totalDays).sum

/-- The average number of study days -/
def averageStudyDays : ℚ := totalClassDays / totalStudents

theorem average_study_days_theorem : 
  (averageStudyDays * 100).floor / 100 = 932 / 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_study_days_theorem_l422_42242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_marks_theorem_l422_42293

/-- Represents the marks scored in each subject -/
structure Marks where
  physics : ℝ
  chemistry : ℝ
  mathematics : ℝ

/-- The average of two numbers -/
noncomputable def average (a b : ℝ) : ℝ := (a + b) / 2

/-- The average of three numbers -/
noncomputable def average3 (a b c : ℝ) : ℝ := (a + b + c) / 3

/-- Theorem stating that given the conditions, the average marks in all 3 subjects is 85 -/
theorem average_marks_theorem (m : Marks) 
    (h1 : average m.physics m.mathematics = 90)
    (h2 : average m.physics m.chemistry = 70)
    (h3 : m.physics = 65) :
  average3 m.physics m.chemistry m.mathematics = 85 := by
  sorry

#check average_marks_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_marks_theorem_l422_42293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friday_running_speed_l422_42233

/-- Represents Jonathan's exercise routine for a week -/
structure ExerciseRoutine where
  monday_speed : ℝ
  wednesday_speed : ℝ
  friday_speed : ℝ
  distance_per_day : ℝ
  total_time : ℝ

/-- Calculates the time spent exercising on a given day -/
noncomputable def time_spent (speed : ℝ) (distance : ℝ) : ℝ :=
  distance / speed

/-- Theorem: Jonathan's running speed on Fridays is 6 miles per hour -/
theorem friday_running_speed (routine : ExerciseRoutine)
  (h1 : routine.monday_speed = 2)
  (h2 : routine.wednesday_speed = 3)
  (h3 : routine.distance_per_day = 6)
  (h4 : routine.total_time = 6)
  : routine.friday_speed = 6 := by
  sorry

#check friday_running_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friday_running_speed_l422_42233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_student_age_l422_42226

theorem fifteenth_student_age (n : ℕ) (total_age : ℕ) (group1_age : ℕ) (group2_age : ℕ) 
  (h1 : n = 15)
  (h2 : total_age = n * 15)
  (h3 : group1_age = 7 * 14)
  (h4 : group2_age = 7 * 16) :
  total_age - (group1_age + group2_age) = 15 := by
  sorry

#check fifteenth_student_age

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_student_age_l422_42226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_four_l422_42245

def mySequence (n : ℕ) : ℚ :=
  if n = 0 then 3
  else if n = 1 then 4
  else 12 / mySequence (n - 1)

theorem tenth_term_is_four :
  mySequence 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_four_l422_42245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_hits_lower_bound_l422_42205

variable (n : ℕ+)

-- Define the probability of a target being hit
noncomputable def prob_hit (n : ℕ+) : ℝ := 1 - (1 - 1 / (n : ℝ)) ^ (n : ℕ)

-- Define the expected number of hit targets
noncomputable def expected_hits (n : ℕ+) : ℝ := (n : ℝ) * prob_hit n

-- Theorem statement
theorem expected_hits_lower_bound (n : ℕ+) :
  expected_hits n ≥ (n : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_hits_lower_bound_l422_42205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_lower_bound_h_inequality_l422_42247

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := Real.log (x + 1) - x

-- Theorem for part (2)(i)
theorem h_lower_bound (k : ℝ) :
  (∀ x : ℝ, x ≥ 0 → h x ≥ k * x^2) ↔ k ≤ -1/2 :=
sorry

-- Theorem for part (2)(ii)
theorem h_inequality (x₁ x₂ : ℝ) (h₁ : x₁ > x₂) (h₂ : x₂ > -1) :
  (x₁ - x₂) / (h x₁ - h x₂ + x₁ - x₂) < (x₁ + x₂ + 2) / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_lower_bound_h_inequality_l422_42247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_sum_l422_42281

/-- Given a triangle ABC with vectors a and b representing AB and BC respectively,
    point D on AC such that AD = 4DC, and E the midpoint of AB,
    if DE = λ₁ * a + λ₂ * b, then λ₁ + λ₂ = -3/10 -/
theorem triangle_vector_sum (A B C D E : EuclideanSpace ℝ (Fin 2)) 
  (a b : EuclideanSpace ℝ (Fin 2)) (lambda1 lambda2 : ℝ) :
  a = B - A →
  b = C - B →
  D - A = (4 : ℝ) • (C - D) →
  E - A = (1 / 2 : ℝ) • (B - A) →
  E - D = lambda1 • a + lambda2 • b →
  lambda1 + lambda2 = -3/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_sum_l422_42281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l422_42209

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

noncomputable def f₁ (x : ℝ) : ℤ := floor (4 * x)
noncomputable def g (x : ℝ) : ℝ := 4 * x - (f₁ x : ℝ)
noncomputable def f₂ (x : ℝ) : ℤ := f₁ (g x)

theorem problem_solution :
  (∀ x : ℝ, x = 5/4 → f₁ x = 1 ∧ f₂ x = 3) ∧
  (∀ x : ℝ, f₁ x = 1 ∧ f₂ x = 3 → 1/4 ≤ x ∧ x < 5/16) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l422_42209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_function_min_value_of_function_range_of_sum_l422_42290

theorem max_value_of_function (x : ℝ) (h : x < -2) :
  ∃ m : ℝ, m = -2 * Real.sqrt 2 - 4 ∧ 
    ∀ y, 2 * y + 1 / (y + 2) ≤ m := by
  sorry

theorem min_value_of_function :
  ∃ m : ℝ, m = 5/2 ∧ 
    ∀ x : ℝ, (x^2 + 5) / Real.sqrt (x^2 + 4) ≥ m := by
  sorry

theorem range_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + b + 3) :
  a + b ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_function_min_value_of_function_range_of_sum_l422_42290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isaacs_bicycle_journey_l422_42294

/-- Calculates the distance covered in the last part of Mr. Isaac's bicycle journey -/
theorem isaacs_bicycle_journey (speed : ℝ) (initial_time : ℝ) (middle_distance : ℝ) (rest_time : ℝ) (total_time : ℝ) : 
  speed = 10 →
  initial_time = 0.5 →
  middle_distance = 15 →
  rest_time = 0.5 →
  total_time = 4.5 →
  speed * (total_time - (initial_time + middle_distance / speed + rest_time)) = 20 := by
  intro h_speed h_initial_time h_middle_distance h_rest_time h_total_time
  -- The proof steps would go here
  sorry

#check isaacs_bicycle_journey

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isaacs_bicycle_journey_l422_42294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2007_value_l422_42215

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Add this case for 0
  | 1 => 2
  | n + 2 => (1 + sequence_a (n + 1)) / (1 - sequence_a (n + 1))

theorem sequence_a_2007_value :
  -2008 * sequence_a 2007 = 1004 := by
  -- Proof steps would go here
  sorry

#eval sequence_a 5  -- This line is optional, for testing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2007_value_l422_42215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_example_l422_42218

/-- Calculates the selling price of an article given its cost price and profit percentage. -/
noncomputable def selling_price (cost_price : ℝ) (profit_percentage : ℝ) : ℝ :=
  cost_price * (1 + profit_percentage / 100)

/-- Theorem stating that an article with a cost price of 240 and a profit percentage of 20% has a selling price of 288. -/
theorem selling_price_example : selling_price 240 20 = 288 := by
  -- Unfold the definition of selling_price
  unfold selling_price
  -- Simplify the arithmetic
  simp [mul_add, mul_div_right_comm]
  -- Perform the final calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_example_l422_42218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_theorem_l422_42222

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x < y → f x < f y

theorem even_function_theorem (f : ℝ → ℝ) (m n : ℝ) :
  is_even f →
  increasing_on f {x | x < 0} →
  (∀ a, f (2*a^2 - 3*a + 2) < f (a^2 - 5*a + 9)) →
  (∀ a, f (2*a^2 - 3*a + 2) < f (a^2 - 5*a + 9) ↔ 2*a^2 + (m-4)*a + (n-m+3) > 0) →
  m = 8 ∧ n = -9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_theorem_l422_42222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_from_hyperbola_l422_42224

/-- Given a hyperbola and a circle satisfying certain conditions, prove the equation of the circle -/
theorem circle_equation_from_hyperbola (m : ℝ) (h_m_pos : m > 0) :
  let hyperbola := fun x y ↦ x^2 / 16 - y^2 / 9 = 1
  let imaginary_semi_axis := 3
  let asymptotes := fun x y ↦ (3 * x = 4 * y) ∨ (3 * x = -4 * y)
  let circle := fun x y ↦ (x - m)^2 + y^2 = imaginary_semi_axis^2
  (∀ x y, asymptotes x y → (x - m)^2 + y^2 ≥ imaginary_semi_axis^2) ∧
  (∃ x y, asymptotes x y ∧ (x - m)^2 + y^2 = imaginary_semi_axis^2) →
  m = 5 ∧ (∀ x y, circle x y ↔ (x - 5)^2 + y^2 = 9) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_from_hyperbola_l422_42224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_inscribed_octagon_l422_42240

/-- The area of a regular octagon inscribed in a circle with radius r -/
noncomputable def octagon_area (r : ℝ) : ℝ := 2 * r^2 * Real.sqrt 2

/-- Theorem: The area of a regular octagon inscribed in a circle with radius r is 2r²√2 -/
theorem area_of_inscribed_octagon (r : ℝ) (h : r > 0) :
  octagon_area r = 2 * r^2 * Real.sqrt 2 := by
  -- Unfold the definition of octagon_area
  unfold octagon_area
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_inscribed_octagon_l422_42240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l422_42206

-- Part 1
theorem part_one (α : ℝ) (h : Real.tan α = 1/3) :
  1 / (2 * Real.sin α * Real.cos α + Real.cos (2 * α)) = 5/7 := by sorry

-- Part 2
theorem part_two (α : ℝ) :
  (Real.tan (π - α) * Real.cos (2*π - α) * Real.sin (-α + 3/2*π)) / (Real.cos (-α - π) * Real.sin (-π - α)) = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l422_42206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l422_42277

-- Define the line equation
def line_equation (x y m : ℝ) : Prop := x - m * y - 1 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

-- Define the length of chord AB
noncomputable def chord_length : ℝ := 2 * Real.sqrt 3

-- Theorem statement
theorem intersection_chord_length (m : ℝ) :
  (∃ A B : ℝ × ℝ, 
    line_equation A.1 A.2 m ∧
    line_equation B.1 B.2 m ∧
    circle_equation A.1 A.2 ∧
    circle_equation B.1 B.2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = chord_length^2) →
  m = Real.sqrt 3 / 3 ∨ m = -Real.sqrt 3 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l422_42277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_proof_l422_42296

noncomputable section

theorem problem_proof : 
  Real.sqrt 4 - (2023 : ℝ) ^ 0 + 2 * Real.cos (60 * Real.pi / 180) = 2 := by
  sorry

-- Additional definitions based on the conditions
axiom sqrt_4 : Real.sqrt 4 = 2
axiom power_2023_0 : (2023 : ℝ) ^ 0 = 1
axiom cos_60 : Real.cos (60 * Real.pi / 180) = 1/2

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_proof_l422_42296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_l422_42241

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x - a

-- Define the interval [0, 3]
def interval : Set ℝ := Set.Icc 0 3

-- State the theorem
theorem max_min_difference (a : ℝ) :
  let M := sSup (f a '' interval)
  let N := sInf (f a '' interval)
  M - N = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_l422_42241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l422_42254

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

-- Define the function g
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := f x - k * x

-- Theorem for part 1
theorem part1 (k : ℝ) :
  (∀ x > 0, f x < k * x) ↔ k > 1 / (2 * Real.exp 1) :=
sorry

-- Theorem for part 2
theorem part2 (k : ℝ) :
  (∃ x y, 1 / Real.exp 1 ≤ x ∧ x < y ∧ y ≤ Real.exp 2 ∧ g k x = 0 ∧ g k y = 0 ∧
  ∀ z, x < z ∧ z < y → g k z ≠ 0) ↔
  2 / (Real.exp 4) ≤ k ∧ k < 1 / (2 * Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l422_42254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radical_combination_l422_42212

theorem radical_combination (a b : ℝ) :
  (∃ (k : ℝ), k > 0 ∧ Real.sqrt 3 = k * Real.sqrt (1/3)) ∧ 
  (∀ (k : ℝ), k > 0 → Real.sqrt 3 ≠ k * Real.sqrt 18) ∧
  (∀ (k : ℝ), k > 0 → Real.sqrt (a + 1) ≠ k * Real.sqrt (a - 1)) ∧
  (∀ (k : ℝ), k > 0 → Real.sqrt (a^2 * b) ≠ k * Real.sqrt (a * b^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radical_combination_l422_42212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_R_evaluation_l422_42256

def R (x : ℝ) : ℝ := 1*x^2 + 4*x + 2

theorem R_evaluation :
  (∀ i : ℕ, 0 ≤ R i ∧ R i < 5) →
  R (Real.sqrt 5) = 30 + 22 * Real.sqrt 5 →
  R 3 = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_R_evaluation_l422_42256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_theorem_l422_42270

noncomputable section

-- Define the ellipse C₁
def C₁ (x y : ℝ) : Prop := y^2/4 + x^2/3 = 1

-- Define the parabola C₂
def C₂ (x y : ℝ) : Prop := x^2 = 4*y

-- Define point M
def M : ℝ × ℝ := (2*Real.sqrt 6/3, 2/3)

-- Define the area of triangle QAB
noncomputable def area_QAB (Q A B : ℝ × ℝ) : ℝ := sorry

theorem ellipse_parabola_theorem :
  -- Given conditions
  C₁ M.1 M.2 ∧
  (∃ (a : ℝ), a > 0 ∧ ∀ (x y : ℝ), C₁ x y ↔ y^2/a^2 + x^2/3 = 1) ∧
  (∃ (p : ℝ), p > 0 ∧ ∀ (x y : ℝ), C₂ x y ↔ x^2 = 2*p*y) ∧
  -- Upper focus of C₁ coincides with focus of C₂
  (∃ (f : ℝ × ℝ), f.2 > 0 ∧ 
    (∀ (x y : ℝ), C₁ x y → (x - f.1)^2 + (y - f.2)^2 = (x - f.1)^2 + (y + f.2)^2) ∧
    (∀ (x y : ℝ), C₂ x y → (x - f.1)^2 + (y - f.2)^2 = (x + f.1)^2 + (y - f.2)^2)) →
  -- Conclusions
  (∀ (x y : ℝ), C₁ x y ↔ y^2/4 + x^2/3 = 1) ∧
  (∀ (x y : ℝ), C₂ x y ↔ x^2 = 4*y) ∧
  (∃ (max_area : ℝ), 
    max_area = 8 * Real.sqrt 2 ∧
    ∀ (Q A B : ℝ × ℝ), 
      C₁ Q.1 Q.2 ∧ Q.2 < 0 ∧ C₂ A.1 A.2 ∧ C₂ B.1 B.2 →
      area_QAB Q A B ≤ max_area) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_theorem_l422_42270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_average_marks_specific_classes_average_l422_42259

/-- Given two classes with different numbers of students and average marks,
    calculate the average marks of all students combined. -/
theorem combined_average_marks (n1 n2 : ℕ) (avg1 avg2 : ℚ) 
    (h1 : n1 > 0) (h2 : n2 > 0) :
  let total_marks : ℚ := n1 * avg1 + n2 * avg2
  let total_students : ℕ := n1 + n2
  (total_marks / total_students : ℚ) = (n1 * avg1 + n2 * avg2) / (n1 + n2) :=
by
  sorry

/-- The average marks of all students in two specific classes. -/
theorem specific_classes_average :
  let n1 : ℕ := 28
  let n2 : ℕ := 50
  let avg1 : ℚ := 40
  let avg2 : ℚ := 60
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 4120 / 78 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_average_marks_specific_classes_average_l422_42259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_in_cube_l422_42227

/-- The length of the portion of the line segment from (0,0,0) to (6,6,18) contained within a cube of edge length 5 -/
theorem line_segment_in_cube : ∃ (length : Real),
  let cube_edge_length : Real := 5
  let entry_point : Fin 3 → Real := ![0, 0, 7]
  let exit_point : Fin 3 → Real := ![5, 5, 12]
  let line_start : Fin 3 → Real := ![0, 0, 0]
  let line_end : Fin 3 → Real := ![6, 6, 18]
  length = 5 * Real.sqrt 3 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_in_cube_l422_42227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_sum_cone_cylinder_sphere_l422_42285

theorem volume_sum_cone_cylinder_sphere (r : ℝ) (hr : r > 0) : 
  (1/3) * Real.pi * r^3 + Real.pi * r^3 + (4/3) * Real.pi * r^3 = (8/3) * Real.pi * r^3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_sum_cone_cylinder_sphere_l422_42285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l422_42268

theorem equation_solutions :
  (∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x < y ∧ y < z ∧ y = x + 1 ∧ z = y + 1 ∧ x + y + z = 36) ∧
  (∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x < y ∧ y < z ∧ y = x + 2 ∧ z = y + 2 ∧ x % 2 = 0 ∧ y % 2 = 0 ∧ z % 2 = 0 ∧ x + y + z = 36) ∧
  (¬ ∃ x y z w : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ x < y ∧ y < z ∧ z < w ∧ y = x + 1 ∧ z = y + 1 ∧ w = z + 1 ∧ x + y + z + w = 36) ∧
  (¬ ∃ x y z w : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ x < y ∧ y < z ∧ z < w ∧ y = x + 2 ∧ z = y + 2 ∧ w = z + 2 ∧ x % 2 = 0 ∧ y % 2 = 0 ∧ z % 2 = 0 ∧ w % 2 = 0 ∧ x + y + z + w = 36) ∧
  (¬ ∃ x y z w : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ x < y ∧ y < z ∧ z < w ∧ y = x + 2 ∧ z = y + 2 ∧ w = z + 2 ∧ x % 2 = 1 ∧ y % 2 = 1 ∧ z % 2 = 1 ∧ w % 2 = 1 ∧ x + y + z + w = 36) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l422_42268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colonization_combinations_l422_42251

def earth_like_planets : ℕ := 7
def mars_like_planets : ℕ := 8
def earth_like_effort : ℕ := 3
def mars_like_effort : ℕ := 1
def total_effort : ℕ := 18

def valid_combination (e m : ℕ) : Bool :=
  e ≤ earth_like_planets ∧ 
  m ≤ mars_like_planets ∧ 
  e * earth_like_effort + m * mars_like_effort = total_effort

def count_combinations : ℕ := 
  (Finset.range (earth_like_planets + 1)).sum (λ e =>
    (Finset.range (mars_like_planets + 1)).sum (λ m =>
      if valid_combination e m then 
        Nat.choose earth_like_planets e * Nat.choose mars_like_planets m
      else 0))

theorem colonization_combinations : count_combinations = 2163 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_colonization_combinations_l422_42251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_formula_l422_42244

/-- The volume of a pyramid with lateral edges b, and a right triangular base with legs in ratio m:n and hypotenuse c -/
noncomputable def pyramidVolume (b m n c : ℝ) : ℝ :=
  (m * n * c^2 * Real.sqrt (4 * b^2 - c^2)) / (12 * (m^2 + n^2))

/-- Theorem stating the volume of the pyramid with given parameters -/
theorem pyramid_volume_formula (b m n c : ℝ) 
  (hb : b > 0) 
  (hm : m > 0) 
  (hn : n > 0) 
  (hc : c > 0) 
  (hbase : c^2 = (m * c / Real.sqrt (m^2 + n^2))^2 + (n * c / Real.sqrt (m^2 + n^2))^2) 
  (hlateral : b^2 ≥ c^2 / 4) :
  ∃ V : ℝ, V = pyramidVolume b m n c ∧ 
  V = (1/3) * ((m * n * c^2) / (2 * (m^2 + n^2))) * (Real.sqrt (4 * b^2 - c^2) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_formula_l422_42244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l422_42264

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  A = π / 3 →
  0 < b ∧ 0 < c →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) →
  (∃ (S : ℝ), S = (1/2) * b * c * (Real.sin A) ∧ S ≤ Real.sqrt 3) ∧
  (∃ (S : ℝ), S = (1/2) * b * c * (Real.sin A) ∧ S = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l422_42264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_semicircles_area_l422_42257

-- Define the function for calculating the area outside hexagon but inside semicircles
noncomputable def areaOutsideHexagonInsideSemicircles (hexagonSideLength : ℝ) : ℝ :=
  6 * Real.pi * (hexagonSideLength / 2)^2 - (3 * Real.sqrt 3 * hexagonSideLength^2 / 2)

-- The main theorem
theorem hexagon_semicircles_area (hexagonSideLength : ℝ) :
  hexagonSideLength = 2 →
  areaOutsideHexagonInsideSemicircles hexagonSideLength = 6 * Real.pi - 6 * Real.sqrt 3 :=
by
  intro h
  simp [areaOutsideHexagonInsideSemicircles]
  rw [h]
  -- The rest of the proof would go here
  sorry

-- You can add more helper lemmas or definitions if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_semicircles_area_l422_42257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_discount_is_five_percent_l422_42228

/-- Given an original price, first discount rate, and final price after two discounts,
    calculate the second discount rate. -/
noncomputable def second_discount_rate (original_price first_discount_rate final_price : ℝ) : ℝ :=
  let price_after_first_discount := original_price * (1 - first_discount_rate / 100)
  100 * (1 - final_price / price_after_first_discount)

/-- Theorem stating that for the given conditions, the second discount rate is 5% -/
theorem second_discount_is_five_percent :
  second_discount_rate 400 15 323 = 5 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval second_discount_rate 400 15 323

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_discount_is_five_percent_l422_42228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_uniqueness_l422_42269

theorem stock_price_uniqueness :
  ¬ ∃ (n : ℕ), n < 100 ∧ ∃ (k l : ℕ), (100 + n)^k * (100 - n)^l = 100^(k + l) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_uniqueness_l422_42269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_without_A_l422_42234

/-- Represents the class of students --/
structure StudentClass where
  total : Nat
  history_A : Nat
  math_A : Nat
  both_A : Nat

/-- Theorem stating the number of students who didn't receive an A in either subject --/
theorem students_without_A (c : StudentClass)
  (h_total : c.total = 40)
  (h_history : c.history_A = 10)
  (h_math : c.math_A = 18)
  (h_both : c.both_A = 6) :
  c.total - (c.history_A + c.math_A - c.both_A) = 18 := by
  sorry

#check students_without_A

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_without_A_l422_42234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_ribbon_length_l422_42216

def ribbon_length : ℕ := 51 * 100  -- 51 meters in centimeters
def piece_length : ℕ := 15         -- 15 centimeters
def num_pieces : ℕ := 100          -- 100 pieces

theorem remaining_ribbon_length :
  ribbon_length - (piece_length * num_pieces) = 3600 := by
  -- Convert the problem to natural numbers
  unfold ribbon_length piece_length num_pieces
  -- Perform the calculation
  norm_num

#eval ribbon_length - (piece_length * num_pieces)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_ribbon_length_l422_42216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_base_side_length_l422_42208

/-- The side length of a square base of a right pyramid, given the area of one lateral face and the slant height. -/
noncomputable def square_base_side_length (lateral_face_area : ℝ) (slant_height : ℝ) : ℝ :=
  lateral_face_area / (0.5 * slant_height)

/-- Theorem stating that the side length of the square base is 5 meters under given conditions. -/
theorem pyramid_base_side_length :
  square_base_side_length 50 20 = 5 := by
  -- Unfold the definition of square_base_side_length
  unfold square_base_side_length
  -- Simplify the expression
  simp [div_eq_mul_inv]
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_base_side_length_l422_42208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_length_l422_42249

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the length of a side
noncomputable def sideLength (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the area of a triangle
noncomputable def triangleArea (t : Triangle) : ℝ :=
  let a := sideLength t.B t.C
  let b := sideLength t.A t.C
  let c := sideLength t.A t.B
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- State the theorem
theorem min_side_length (t : Triangle) :
  triangleArea t = 1 →
  sideLength t.A t.B = 2 * sideLength t.A t.C →
  ∃ (minBC : ℝ), minBC = Real.sqrt 3 ∧ 
    ∀ (t' : Triangle), 
      triangleArea t' = 1 → 
      sideLength t'.A t'.B = 2 * sideLength t'.A t'.C → 
      sideLength t'.B t'.C ≥ minBC :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_length_l422_42249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_l422_42283

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 2 * Real.sqrt 2)

-- Define the circle (renamed to avoid conflict)
def circleEq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the area function
noncomputable def S (k : ℝ) : ℝ := (4 * Real.sqrt 2 * Real.sqrt (k^2 * (1 - k^2))) / (1 + k^2)

-- Theorem statement
theorem max_area_triangle :
  ∃ (k : ℝ), k ∈ Set.Ioo (-1 : ℝ) 1 ∧
  (∀ (k' : ℝ), k' ∈ Set.Ioo (-1 : ℝ) 1 → S k' ≤ S k) ∧
  S k = 2 ∧
  (k = Real.sqrt 3 / 3 ∨ k = -Real.sqrt 3 / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_l422_42283
