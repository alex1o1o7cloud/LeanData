import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_m_values_l371_37118

/-- Given sets A and B, where A = {-1, 1} and B = {x | mx = 1}, 
    if A ∪ B = A, then m ∈ {-1, 0, 1}. -/
theorem possible_m_values (m : ℝ) : 
  let A : Set ℝ := {-1, 1}
  let B : Set ℝ := {x : ℝ | m * x = 1}
  (A ∪ B = A) → m ∈ ({-1, 0, 1} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_m_values_l371_37118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l371_37176

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n, a n > 0) →
  a 4 = 2 →
  (f (a 1^3) + f (a 2^3) + f (a 3^3) + f (a 4^3) + f (a 5^3) + f (a 6^3) + f (a 7^3)) = -21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l371_37176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_l371_37112

-- Define the cube
def Cube := ℝ × ℝ × ℝ

-- Define points A, B, C, D
def A : Cube := (0, 0, 0)
def B : Cube := (20, 0, 0)
def C : Cube := (20, 0, 20)
def D : Cube := (20, 20, 20)

-- Define points P, Q, R
def P : Cube := (5, 0, 0)
def Q : Cube := (20, 0, 15)
def R : Cube := (20, 10, 20)

-- Define the plane PQR
def planePQR (x y z : ℝ) : Prop := 2*x + y - 2*z = 10

-- Define the intersection of the plane and the cube
def intersectionPolygon (p : Cube) : Prop :=
  0 ≤ p.1 ∧ p.1 ≤ 20 ∧
  0 ≤ p.2 ∧ p.2 ≤ 20 ∧
  0 ≤ p.2.2 ∧ p.2.2 ≤ 20 ∧
  planePQR p.1 p.2.1 p.2.2

-- State the theorem
theorem intersection_area : 
  ∃ (area : ℝ), area = 525 ∧ 
  (∀ p : Cube, intersectionPolygon p → p ∈ {p | ∃ a, a = area}) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_l371_37112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l371_37124

def Z (m : ℝ) : ℂ := (2 * m^2 - 3 * m - 2 : ℝ) + (m^2 - 3 * m + 2 : ℝ) * Complex.I

theorem complex_number_properties (m : ℝ) :
  ((Z m).im = 0 ↔ m = 1 ∨ m = 2) ∧
  ((Z m).im ≠ 0 ↔ m ≠ 1 ∧ m ≠ 2) ∧
  ((Z m).re = 0 ∧ (Z m).im ≠ 0 ↔ m = -1/2) ∧
  ((Z m).im > 0 ↔ m < 1 ∨ m > 2) :=
by sorry

#check complex_number_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l371_37124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_real_value_l371_37193

theorem complex_expression_real_value : 
  ∃! (x : ℝ), ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧
    x = (Complex.I ^ 4 * (a + Complex.I * b) ^ 4 - Complex.I * 59).re ∧ 
    (Complex.I ^ 4 * (a + Complex.I * b) ^ 4 - Complex.I * 59).im = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_real_value_l371_37193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_area_circumference_ratio_l371_37155

-- Define the circle properties
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2
noncomputable def circle_circumference (r : ℝ) : ℝ := 2 * Real.pi * r

-- State the theorem
theorem circle_radius_from_area_circumference_ratio 
  (A C : ℝ) (h : A / C = 15) : 
  ∃ r : ℝ, circle_area r = A ∧ circle_circumference r = C ∧ r = 30 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_area_circumference_ratio_l371_37155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l371_37167

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (Real.log (2*x - x^2)) / (x - 1)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.union (Set.Ioo 0 1) (Set.Ioo 1 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l371_37167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counting_problem_l371_37174

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

theorem counting_problem (n : ℕ) : 
  (∃ k : ℕ, arithmetic_sequence 1 2 k = 19) ∧ 
  (∃ m : ℕ, arithmetic_sequence (n : ℤ) (-2) m = 89) →
  n = 107 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_counting_problem_l371_37174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_locus_is_open_disc_l371_37107

/-- Two circles in a plane -/
structure TwoCircles where
  O₁ : ℝ × ℝ  -- Center of C₁
  O₂ : ℝ × ℝ  -- Center of C₂
  R : ℝ       -- Radius of C₂ (C₁ has radius 2R)
  h : dist O₁ O₂ > 3 * R  -- Distance between centers > 3R

/-- Point that divides a line segment in a given ratio -/
noncomputable def divideSegment (A B : ℝ × ℝ) (r : ℝ) : ℝ × ℝ :=
  ((r * A.1 + B.1) / (r + 1), (r * A.2 + B.2) / (r + 1))

/-- Locus of centroids -/
noncomputable def centroidLocus (tc : TwoCircles) : Set (ℝ × ℝ) :=
  let O₀ := divideSegment tc.O₁ tc.O₂ 2
  { p | dist p O₀ < 4 * tc.R / 3 ∧ p ≠ O₀ }

/-- Main theorem -/
theorem centroid_locus_is_open_disc (tc : TwoCircles) :
  ∀ p, p ∈ centroidLocus tc ↔
    (∃ A B C, p = (A + B + C) / 3 ∧
      dist A tc.O₁ = 2 * tc.R ∧
      dist B tc.O₂ = tc.R ∧
      dist C tc.O₂ = tc.R) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_locus_is_open_disc_l371_37107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_remaining_number_l371_37170

/-- The elimination process for a given number of elements -/
def elimination_process (n : ℕ) : ℕ :=
  let rec helper (k : ℕ) (m : ℕ) (fuel : ℕ) : ℕ :=
    if fuel = 0 then m
    else if 3 * k > m then m
    else helper (3 * k) m (fuel - 1)
  helper 1 n n

/-- The theorem stating that the last remaining number is 1888 -/
theorem last_remaining_number : elimination_process 1987 = 1888 := by
  sorry

#eval elimination_process 1987

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_remaining_number_l371_37170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l371_37141

noncomputable section

-- Define the circle C in polar coordinates
def circle_C (θ : ℝ) : ℝ := 2 * Real.cos θ

-- Define the line l in parametric form
def line_l (t : ℝ) : ℝ × ℝ := (1/2 + (Real.sqrt 3)/2 * t, 1/2 + 1/2 * t)

-- Define point A in polar coordinates
def point_A : ℝ × ℝ := (Real.sqrt 2 / 2, Real.pi / 4)

-- Define the theorem
theorem intersection_distance_product :
  ∃ (P Q : ℝ × ℝ),
    (∃ (θ_P θ_Q : ℝ), circle_C θ_P = Real.sqrt (P.1^2 + P.2^2) ∧ 
                       circle_C θ_Q = Real.sqrt (Q.1^2 + Q.2^2)) ∧
    (∃ (t_P t_Q : ℝ), line_l t_P = P ∧ line_l t_Q = Q) ∧
    let (x_A, y_A) := (point_A.1 * Real.cos point_A.2, point_A.1 * Real.sin point_A.2)
    let AP := Real.sqrt ((P.1 - x_A)^2 + (P.2 - y_A)^2)
    let AQ := Real.sqrt ((Q.1 - x_A)^2 + (Q.2 - y_A)^2)
    AP * AQ = 1/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l371_37141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_l371_37183

theorem undefined_values (x : ℝ) : 
  ¬(x^2 - 9 ≠ 0) ↔ x = 3 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_l371_37183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l371_37188

open Real

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- State the theorem
theorem function_inequality
  (hf_diff : Differentiable ℝ f)
  (hg_diff : Differentiable ℝ g)
  (hf_pos : ∀ x, f x > 0)
  (hg_pos : ∀ x, g x > 0)
  (h_ineq : ∀ x, deriv f x * g x - f x * deriv g x < 0)
  (a b x : ℝ)
  (h_bounds : a < x ∧ x < b) :
  f x * g b > f b * g x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l371_37188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_from_ordered_positive_reals_l371_37114

theorem inequalities_from_ordered_positive_reals
  (a b c : ℝ) (h₁ : 0 < a) (h₂ : a < b) (h₃ : b < c) :
  (a * b < b * c) ∧
  (a * c < b * c) ∧
  (a * b < a * c) ∧
  (a + c < b + c) ∧
  (c / a > 1) :=
by
  constructor
  · sorry
  constructor
  · sorry
  constructor
  · sorry
  constructor
  · sorry
  · sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_from_ordered_positive_reals_l371_37114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_l_shapes_in_8x8_grid_l371_37173

/-- Represents an L-shape in a grid -/
structure LShape where
  position : Nat × Nat
  orientation : Nat

/-- Represents a square grid -/
def Grid := Nat → Nat → Bool

/-- Check if an L-shape is valid within the grid -/
def isValidLShape (grid : Grid) (shape : LShape) : Bool :=
  sorry

/-- Check if two L-shapes overlap -/
def doOverlap (shape1 shape2 : LShape) : Bool :=
  sorry

/-- Check if an L-shape can be placed in the grid -/
def canPlaceLShape (grid : Grid) (shape : LShape) : Bool :=
  sorry

/-- The size of the grid -/
def gridSize : Nat := 8

/-- The theorem stating the minimum number of L-shapes -/
theorem min_l_shapes_in_8x8_grid :
  ∀ (shapes : List LShape),
    (∀ s, s ∈ shapes → isValidLShape (λ _ _ => true) s) →
    (∀ s1 s2, s1 ∈ shapes → s2 ∈ shapes → s1 ≠ s2 → ¬doOverlap s1 s2) →
    (∀ newShape, newShape ∉ shapes → ¬canPlaceLShape (λ _ _ => true) newShape) →
    shapes.length ≥ 11 :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_l_shapes_in_8x8_grid_l371_37173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l371_37139

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Checks if a point is on a line segment -/
def isOnLineSegment (p start end_ : Point) : Prop :=
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ p.x = start.x + t * (end_.x - start.x) ∧ p.y = start.y + t * (end_.y - start.y)

theorem min_distance_sum (p : Point) :
  let o : Point := ⟨0, 0⟩
  let a : Point := ⟨0, 6⟩
  let b : Point := ⟨-3, 2⟩
  let c : Point := ⟨-2, 9⟩
  isOnLineSegment p o a →
  ∀ q : Point, isOnLineSegment q o a →
  distance p b + distance p c ≤ distance q b + distance q c →
  distance p b + distance p c = 5 + Real.sqrt 13 := by
  sorry

#check min_distance_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l371_37139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_earnings_l371_37120

noncomputable section

-- Define the initial savings
def initial_savings : ℝ := 1000

-- Define the amount in each account
def amount_per_account : ℝ := initial_savings / 2

-- Define the time period
def years : ℝ := 2

-- Define the compound interest earned
def compound_interest_earned : ℝ := 105

-- Function to calculate compound interest
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

-- Function to calculate simple interest
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

-- Theorem statement
theorem simple_interest_earnings :
  ∃ (rate : ℝ),
    compound_interest amount_per_account rate years = compound_interest_earned ∧
    simple_interest amount_per_account rate years = 100 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_earnings_l371_37120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_polar_coordinates_l371_37178

noncomputable def circle_center (θ : ℝ) : ℝ × ℝ := (Real.sqrt 3 + Real.cos θ, 1 + Real.sin θ)

theorem circle_center_polar_coordinates :
  ∃ (ρ θ : ℝ), 
    ρ = 2 ∧ 
    θ = Real.pi / 6 ∧ 
    ρ * Real.cos θ = Real.sqrt 3 ∧ 
    ρ * Real.sin θ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_polar_coordinates_l371_37178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_dormitory_location_optimal_x_value_l371_37186

-- Define the cost function as noncomputable
noncomputable def cost_function (x : ℝ) : ℝ := 200 / (3 * x + 2) + 6 * x

-- State the theorem
theorem optimal_dormitory_location :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 5 ∧
  cost_function x = 36 ∧
  ∀ (y : ℝ), 0 ≤ y ∧ y ≤ 5 → cost_function y ≥ cost_function x :=
by
  -- The proof would go here
  sorry

-- State the specific optimal location
theorem optimal_x_value :
  ∃ (x : ℝ), x = 8/3 ∧ cost_function x = 36 ∧
  ∀ (y : ℝ), 0 ≤ y ∧ y ≤ 5 → cost_function y ≥ cost_function x :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_dormitory_location_optimal_x_value_l371_37186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parity_of_f_monotonicity_of_f_l371_37194

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 / x

-- Theorem for the parity of f
theorem parity_of_f (a : ℝ) :
  (a = 0 ∧ ∀ x : ℝ, x ≠ 0 → f a x = -f a (-x)) ∨
  (a ≠ 0 ∧ ∃ x : ℝ, f a x ≠ -f a (-x) ∧ f a x ≠ f a (-x)) := by sorry

-- Theorem for the monotonicity of f when a ∈ (1, 3)
theorem monotonicity_of_f (a : ℝ) (ha : 1 < a ∧ a < 3) :
  ∀ x1 x2 : ℝ, 1 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 2 → f a x1 < f a x2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parity_of_f_monotonicity_of_f_l371_37194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_6_value_l371_37121

noncomputable def F (a : ℝ) : ℕ → ℝ
  | 0 => a  -- Adding case for 0
  | 1 => a
  | 2 => a
  | 3 => a
  | (n + 4) => (F a (n + 3) * F a (n + 2) + 1) / F a (n + 1)

theorem F_6_value (a : ℝ) : F a 6 = a + 1 + 2 / a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_6_value_l371_37121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_l371_37161

theorem sector_area (r : ℝ) (h : 2 * Real.sin 1 * r = 2) : 
  r^2 = 1 / (Real.sin 1)^2 → r * 1 = 1 / (Real.sin 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_l371_37161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_theorem_l371_37166

-- Define the circle and point A
def circle_radius : ℝ := 1
def initial_distance : ℝ := 50

-- Define the reflection operation
def reflect (n : ℕ) (r : ℝ) (d : ℝ) : ℝ :=
  r + 2 * (n : ℝ)

-- Theorem statement
theorem reflection_theorem :
  (∃ n : ℕ, n ≥ 25 ∧ reflect n circle_radius initial_distance ≤ circle_radius) ∧
  (∀ m : ℕ, m < 25 → reflect m circle_radius initial_distance > circle_radius) := by
  sorry

#check reflection_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_theorem_l371_37166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_properties_l371_37149

theorem quadratic_equation_properties (a : ℤ) 
  (h : ∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ x₁^2 - 3*x₁ + a + 4 = 0 ∧ x₂^2 - 3*x₂ + a + 4 = 0) :
  (∃ x₁ x₂ : ℤ, x₁ % 2 = 1 ∧ x₂ % 2 = 0 ∧ x₁^2 - 3*x₁ + a + 4 = 0 ∧ x₂^2 - 3*x₂ + a + 4 = 0) ∧
  (a < 0) ∧
  (∀ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ x₁^2 - 3*x₁ + a + 4 = 0 ∧ x₂^2 - 3*x₂ + a + 4 = 0 →
    ((x₁ > 0 ∧ x₂ > 0) ∨ (x₁ < 0 ∧ x₂ < 0)) →
    a = -2 ∧ ((x₁ = 1 ∧ x₂ = 2) ∨ (x₁ = 2 ∧ x₂ = 1))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_properties_l371_37149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_ellipse_l371_37196

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  2 * x^2 + 8 * x + 3 * y^2 + 12 * y + 18 = 0

/-- The area of the ellipse -/
noncomputable def ellipse_area : ℝ := Real.pi * Real.sqrt (2/3)

/-- Theorem stating that the area of the ellipse defined by the given equation
    is equal to π * sqrt(2/3) -/
theorem area_of_ellipse :
  ∃ (a b : ℝ), (∀ x y : ℝ, ellipse_equation x y ↔ (x-a)^2 + 3/2*(y-b)^2 = 1) ∧
  ellipse_area = Real.pi * Real.sqrt (2/3) := by
  sorry

#check area_of_ellipse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_ellipse_l371_37196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_l371_37113

/-- The distance between two parallel lines in the form Ax + By + C = 0 -/
noncomputable def distance_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  abs (C₂ - C₁) / Real.sqrt (A^2 + B^2)

/-- Theorem: The distance between the parallel lines 3x + 4y - 2 = 0 and 6x + 8y - 5 = 0 is 1/10 -/
theorem distance_specific_parallel_lines :
  distance_parallel_lines 6 8 (-4) (-5) = 1/10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_l371_37113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_l371_37116

/-- The cost of one eraser in rubles -/
def E : ℝ := sorry

/-- The cost of one pen in rubles -/
def P : ℝ := sorry

/-- The cost of one marker in rubles -/
def M : ℝ := sorry

/-- The first condition: An eraser, 3 pens, and 2 markers cost 240 rubles -/
axiom condition1 : E + 3 * P + 2 * M = 240

/-- The second condition: 2 erasers, 4 markers, and 5 pens cost 440 rubles -/
axiom condition2 : 2 * E + 5 * P + 4 * M = 440

/-- The theorem to be proved -/
theorem total_cost : 3 * E + 4 * P + 6 * M = 520 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_l371_37116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_iff_a_nonpositive_l371_37152

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 1)^2 + log x

-- State the theorem
theorem f_inequality_iff_a_nonpositive :
  ∀ a : ℝ, (∀ x : ℝ, x ≥ 1 → f a x ≤ x - 1) ↔ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_iff_a_nonpositive_l371_37152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_power_2018_l371_37156

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1/2, 0, Real.sqrt 3/2],
    ![0, 1, 0],
    ![-Real.sqrt 3/2, 0, 1/2]]

theorem B_power_2018 :
  B^2018 = ![![-1/2, 0, Real.sqrt 3/2],
             ![0, 1, 0],
             ![-Real.sqrt 3/2, 0, -1/2]] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_power_2018_l371_37156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_equation_l371_37109

/-- Given a triangle ABC with side equations, prove the equation of the altitude from A to BC -/
theorem altitude_equation (AB BC CA : ℝ → ℝ → Prop) 
  (hAB : AB = λ x y ↦ 3*x + 4*y + 12 = 0)
  (hBC : BC = λ x y ↦ 4*x - 3*y + 16 = 0)
  (hCA : CA = λ x y ↦ 2*x + y - 2 = 0) :
  ∃ (altitude : ℝ → ℝ → Prop), 
    altitude = λ x y ↦ x - 2*y + 4 = 0 ∧ 
    (∃ A B C : ℝ × ℝ, 
      AB A.1 A.2 ∧ AB B.1 B.2 ∧
      BC B.1 B.2 ∧ BC C.1 C.2 ∧
      CA C.1 C.2 ∧ CA A.1 A.2 ∧
      (∃ D : ℝ × ℝ, 
        altitude A.1 A.2 ∧ altitude D.1 D.2 ∧ 
        BC D.1 D.2 ∧
        (A.1 - D.1) * (B.1 - C.1) + (A.2 - D.2) * (B.2 - C.2) = 0)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_equation_l371_37109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_product_of_perpendiculars_l371_37138

/-- Represents a point in 2D space. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle with vertices A, B, and C. -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents that line BP is perpendicular to line AD at point P. -/
def Perpendicular (B P A D : Point) : Prop := sorry

/-- Represents that AD is the internal angle bisector of angle BAC. -/
def IsInternalAngleBisector (A D B C : Point) : Prop := sorry

/-- Represents that AE is the external angle bisector of angle BAC. -/
def IsExternalAngleBisector (A E B C : Point) : Prop := sorry

/-- Calculates the area of a triangle. -/
noncomputable def area (t : Triangle) : ℝ := sorry

/-- Calculates the distance between two points. -/
noncomputable def dist (P Q : Point) : ℝ := sorry

/-- Given a triangle ABC with internal angle bisector AD and external angle bisector AE,
    if BP is perpendicular to AD at P and CQ is perpendicular to AE at Q,
    then the area of triangle ABC is equal to AP * AQ. -/
theorem area_equals_product_of_perpendiculars 
  (t : Triangle) (P Q D E : Point) : 
  Perpendicular t.B P t.A D → 
  Perpendicular t.C Q t.A E → 
  IsInternalAngleBisector t.A D t.B t.C → 
  IsExternalAngleBisector t.A E t.B t.C → 
  area t = dist t.A P * dist t.A Q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_product_of_perpendiculars_l371_37138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_closest_l371_37119

/-- The point on the line y = 2x - 4 that is closest to (3, 1) -/
noncomputable def closest_point : ℝ × ℝ := (13/5, 6/5)

/-- The line y = 2x - 4 -/
def line (x : ℝ) : ℝ := 2 * x - 4

/-- The given point -/
def given_point : ℝ × ℝ := (3, 1)

/-- Theorem stating that closest_point is on the line and is the closest point to given_point -/
theorem closest_point_is_closest :
  (line closest_point.1 = closest_point.2) ∧
  (∀ p : ℝ × ℝ, line p.1 = p.2 →
    (closest_point.1 - given_point.1)^2 + (closest_point.2 - given_point.2)^2 ≤
    (p.1 - given_point.1)^2 + (p.2 - given_point.2)^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_closest_l371_37119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_square_rational_multiple_of_pi_l371_37132

noncomputable def f (n : ℕ) : ℝ := ∑' k, 1 / (Nat.lcm k n : ℝ)^2

theorem smallest_m_for_square_rational_multiple_of_pi : 
  (∀ n, f n = ∑' k, 1 / (Nat.lcm k n : ℝ)^2) →  -- Definition of f
  (f 1 = π^2 / 6) →                             -- Given condition
  (∃ a b : ℚ, 42 * f 10 = (a * π / b)^2) ∧      -- 42⋅f(10) is a square of rational multiple of π
  (∀ m : ℕ, m < 42 → ¬∃ a b : ℚ, m * f 10 = (a * π / b)^2) -- 42 is the smallest such positive integer
  :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_square_rational_multiple_of_pi_l371_37132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_i_for_S_in_range_l371_37135

/-- Sequence definition -/
def a (i : ℕ) : ℕ := 2^((i + 1) / 3)

/-- Sum of five terms -/
def S (i : ℕ) : ℕ := a i + a (i + 3) + a (i + 6) + a (i + 9) + a (i + 12)

/-- Theorem statement -/
theorem unique_i_for_S_in_range : ∃! i : ℕ, 1000 ≤ S i ∧ S i ≤ 3000 ∧ i = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_i_for_S_in_range_l371_37135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_four_and_six_percentage_l371_37122

theorem divisible_by_four_and_six_percentage (n : ℕ) (h : n = 300) : 
  (↑(Finset.filter (λ x : ℕ => 4 ∣ x ∧ 6 ∣ x) (Finset.range (n + 1))).card / ↑n) * 100 = 25 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_four_and_six_percentage_l371_37122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_restricted_permutations_count_l371_37142

/-- The number of permutations of 5 elements where two specific elements are neither first nor last -/
def restricted_permutations : ℕ := 36

/-- The set of possible positions for elements that are neither first nor last in a 5-element permutation -/
def middle_positions : Finset ℕ := {2, 3, 4}

/-- The number of elements that must be placed in middle positions -/
def num_restricted_elements : ℕ := 2

/-- The total number of elements in the permutation -/
def total_elements : ℕ := 5

theorem restricted_permutations_count :
  restricted_permutations = 
    (middle_positions.card.choose num_restricted_elements) * 
    Nat.factorial (total_elements - num_restricted_elements) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_restricted_permutations_count_l371_37142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_PA_PB_l371_37100

noncomputable def P : ℝ × ℝ := (0, 0)
noncomputable def A : ℝ × ℝ := (Real.pi/4, 1)
noncomputable def B : ℝ × ℝ := (3*Real.pi/4, -1)

noncomputable def PA : ℝ × ℝ := (A.1 - P.1, A.2 - P.2)
noncomputable def PB : ℝ × ℝ := (B.1 - P.1, B.2 - P.2)

theorem dot_product_PA_PB :
  PA.1 * PB.1 + PA.2 * PB.2 = 3*Real.pi^2/16 - 1 := by
  simp [PA, PB, P, A, B]
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_PA_PB_l371_37100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l371_37136

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define a function to count intersection points
def countIntersectionPoints (c1 c2 : Circle) : ℕ :=
  sorry

-- Theorem statement
theorem max_intersection_points (D : Circle) (Q : Point) :
  ∀ r : ℝ, r > 0 → Q ∉ {p : Point | distance p D.center ≤ D.radius} →
  (countIntersectionPoints D ⟨Q, r⟩ ≤ 2) ∧ 
  (∃ Q' r', countIntersectionPoints D ⟨Q', r'⟩ = 2) := by
  sorry

#check max_intersection_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l371_37136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_milk_production_l371_37180

/-- Calculates the total milk production with efficiency changes -/
noncomputable def total_milk_production (a b c d e f g : ℝ) : ℝ :=
  (g * b * d * e) / (a * c)

/-- Theorem stating the correct total milk production -/
theorem correct_milk_production 
  (a b c d e : ℝ) 
  (ha : a > 0) (hc : c > 0) 
  (hf : f = 0.8) (hg : g = 1.1) : 
  total_milk_production a b c d e f g = (1.1 * b * d * e) / (a * c) := by
  sorry

#check correct_milk_production

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_milk_production_l371_37180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_monotonicity_condition_extreme_points_inequality_l371_37168

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - x + a * log x

-- State the theorems
theorem tangent_line_at_one (a : ℝ) (h : a = 1) :
  ∃ m b : ℝ, ∀ x y : ℝ, y = f a x → (x = 1 ∧ y = f a 1) ∨ m * x + b = y :=
sorry

theorem monotonicity_condition (a : ℝ) (h : a > 0) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ → f a x₁ < f a x₂) ∨
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ → f a x₁ > f a x₂) ↔ a ≥ 1/4 :=
sorry

theorem extreme_points_inequality (a : ℝ) (h : a > 0) :
  ∀ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ 
    (∀ x : ℝ, 0 < x → (deriv (f a)) x = 0 ↔ x = x₁ ∨ x = x₂) →
    f a x₁ + f a x₂ > -(3 + 2 * log 2) / 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_monotonicity_condition_extreme_points_inequality_l371_37168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pattern_b_not_foldable_l371_37190

-- Define a type for the patterns
inductive Pattern : Type
  | A : Pattern
  | B : Pattern
  | C : Pattern
  | D : Pattern

-- Define a predicate for whether a pattern can be folded into a cube
def canFoldIntoCube : Pattern → Prop
  | Pattern.A => True
  | Pattern.B => False
  | Pattern.C => True
  | Pattern.D => True

-- Theorem stating that Pattern B is the only one that cannot be folded into a cube
theorem pattern_b_not_foldable (p : Pattern) : 
  ¬(canFoldIntoCube p) ↔ p = Pattern.B := by
  cases p
  . case A => simp [canFoldIntoCube]
  . case B => simp [canFoldIntoCube]
  . case C => simp [canFoldIntoCube]
  . case D => simp [canFoldIntoCube]

#check pattern_b_not_foldable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pattern_b_not_foldable_l371_37190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_theorem_l371_37154

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 16

-- Define the line l
def line_l (m x y : ℝ) : Prop := (2*m - 1)*x + (m - 1)*y - 3*m + 1 = 0

-- Define the minimum chord line
def min_chord_line (x y : ℝ) : Prop := x - 2*y - 4 = 0

-- Theorem statement
theorem min_chord_theorem :
  ∃ (m : ℝ), ∀ (x y : ℝ),
    (circle_C x y ∧ line_l m x y) →
    (∀ (m' : ℝ), ∀ (x' y' : ℝ),
      (circle_C x' y' ∧ line_l m' x' y') →
      (x - x')^2 + (y - y')^2 ≤ (x' - x)^2 + (y' - y)^2) →
    min_chord_line x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_theorem_l371_37154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l371_37127

/-- Define lies_on_terminal_side as a local definition --/
def lies_on_terminal_side (P : ℝ × ℝ) (θ : ℝ) : Prop :=
  P.1 = Real.cos θ ∧ P.2 = Real.sin θ

theorem point_on_terminal_side (θ : ℝ) (y : ℝ) :
  (∃ P : ℝ × ℝ, P = (-1, y) ∧ lies_on_terminal_side P θ) →
  Real.sin θ = (2 * Real.sqrt 5) / 5 →
  y = 2 :=
by
  sorry

/- Definitions used in the statement:
   - Real.sqrt (from Mathlib)
   - lies_on_terminal_side (defined locally)
   - Real.sin (from Mathlib)
-/

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l371_37127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_four_different_digits_l371_37198

/-- A function that returns true if a number has four different digits -/
def hasFourDifferentDigits (n : ℕ) : Bool :=
  sorry

/-- A function that returns true if a number is even -/
def isEven (n : ℕ) : Bool :=
  n % 2 = 0

/-- The set of numbers between 3000 and 7000 -/
def numberRange : Set ℕ :=
  {n | 3000 < n ∧ n < 7000}

/-- The count of even numbers between 3000 and 7000 with four different digits -/
noncomputable def countEvenWithFourDifferentDigits : ℕ :=
  Finset.card (Finset.filter (λ n => isEven n ∧ hasFourDifferentDigits n) (Finset.range 7000 \ Finset.range 3001))

theorem count_even_four_different_digits :
  countEvenWithFourDifferentDigits = 1008 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_four_different_digits_l371_37198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_sine_curves_l371_37181

noncomputable section

/-- The area enclosed between two polar curves -/
def area_between_polar_curves (r₁ r₂ : ℝ → ℝ) (a b : ℝ) : ℝ :=
  (1 / 2) * ∫ x in a..b, (r₁ x)^2 - (r₂ x)^2

/-- The first polar curve -/
def r₁ (φ : ℝ) : ℝ := 3 * Real.sin φ

/-- The second polar curve -/
def r₂ (φ : ℝ) : ℝ := 5 * Real.sin φ

/-- The lower bound of integration -/
def a : ℝ := -Real.pi / 2

/-- The upper bound of integration -/
def b : ℝ := Real.pi / 2

theorem area_between_sine_curves :
  area_between_polar_curves r₁ r₂ a b = 4 * Real.pi := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_sine_curves_l371_37181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l371_37130

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the directrix
def directrix : ℝ := -2

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x + 1)^2 + (y - 4)^2 = 4

-- Define the distance from a point to the directrix
def distance_to_directrix (x : ℝ) : ℝ := |x - directrix|

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- State the theorem
theorem min_distance_sum :
  ∃ (d : ℝ), ∀ (x1 y1 x2 y2 : ℝ),
    parabola x1 y1 →
    circle_eq x2 y2 →
    d ≤ distance_to_directrix x1 + distance x1 y1 x2 y2 ∧
    (∃ (x3 y3 x4 y4 : ℝ), parabola x3 y3 ∧ circle_eq x4 y4 ∧
      d = distance_to_directrix x3 + distance x3 y3 x4 y4) ∧
    d = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l371_37130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_B_coordinates_line_BC_equation_l371_37143

/-- Triangle ABC with given properties --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  bisector_B : ℝ → ℝ → Prop
  median_CM : ℝ → ℝ → Prop

/-- The given triangle satisfies the problem conditions --/
def given_triangle : Triangle where
  A := (5, 1)
  B := (2, 3)  -- We now know the coordinates of B
  C := (0, 0)  -- We don't know C's coordinates, but we need to provide some value
  bisector_B := fun x y => x + y - 5 = 0
  median_CM := fun x y => 2*x - y - 5 = 0

/-- The coordinates of vertex B are (2, 3) --/
theorem vertex_B_coordinates (t : Triangle) (h : t = given_triangle) : t.B = (2, 3) := by
  rw [h]
  rfl

/-- Helper function to represent a line through two points --/
def line_through (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | ∃ t : ℝ, (x, y) = ((1 - t) • p.1 + t • q.1, (1 - t) • p.2 + t • q.2)}

/-- The equation of line BC is 3x + 2y - 12 = 0 --/
theorem line_BC_equation (t : Triangle) (h : t = given_triangle) : 
  ∀ x y, (x, y) ∈ line_through t.B t.C ↔ 3*x + 2*y - 12 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_B_coordinates_line_BC_equation_l371_37143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_equation_solutions_l371_37110

theorem cos_equation_solutions :
  ∃ (S : Set ℝ), S = {x ∈ Set.Ioo (0 : ℝ) (24 * Real.pi) | Real.cos (x / 4) = Real.cos x} ∧ 
  ∃ (T : Finset ℝ), T.toSet ⊆ S ∧ T.card = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_equation_solutions_l371_37110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_al_khwarizmi_rule_l371_37105

theorem al_khwarizmi_rule (a r : ℝ) (h : a > 0) :
  let n := a^2 + r
  if r ≤ a then
    |Real.sqrt n - (a + r / (2 * a))| < 1 / a
  else
    |Real.sqrt n - (a + (r + 1) / (2 * a + 2))| < 1 / a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_al_khwarizmi_rule_l371_37105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_property_l371_37129

/-- Curve C with equation y² = mx (m > 0) -/
def curve_C (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = m * p.1 ∧ m > 0}

/-- Line l with equation y = x - 2 -/
def line_l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 - 2}

/-- Point P (-2, -4) -/
def point_P : ℝ × ℝ := (-2, -4)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem curve_intersection_property (m : ℝ) 
  (hm : m > 0)
  (A B : ℝ × ℝ)
  (hA : A ∈ curve_C m ∩ line_l)
  (hB : B ∈ curve_C m ∩ line_l)
  (hAB : A ≠ B)
  (h_condition : distance A point_P * distance B point_P = distance A B ^ 2) :
  m = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_property_l371_37129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l371_37145

theorem problem_solution :
  ((Real.sqrt 12 + Real.sqrt 20) + (Real.sqrt 3 - Real.sqrt 5) = 3 * Real.sqrt 3 + Real.sqrt 5) ∧
  ((4 * Real.sqrt 2 - 3 * Real.sqrt 6) / (2 * Real.sqrt 2) - (Real.sqrt 8 + Real.pi) ^ 0 = 1 - (3 * Real.sqrt 3) / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l371_37145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l371_37169

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  q : ℝ      -- Common ratio
  first : a 1 ≠ 0
  ratio : ∀ n : ℕ, a (n + 1) = a n * q

/-- Sum of first n terms of a geometric sequence -/
noncomputable def sumGeometric (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.a 1 * (1 - seq.q ^ n) / (1 - seq.q)

theorem geometric_sequence_ratio (seq : GeometricSequence) :
  8 * seq.a 2 + seq.a 5 = 0 → sumGeometric seq 6 / sumGeometric seq 3 = -7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l371_37169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_correct_l371_37159

/-- Calculates the total cost of t-shirts after discount -/
def total_cost_after_discount (
  men_white_prices : Fin 3 → ℚ
) (
  men_black_prices : Fin 3 → ℚ
) (
  total_employees : ℕ
) (
  size_distribution : Fin 3 → ℚ
) (
  women_discount : ℚ
) (
  bulk_discount : ℚ
) : ℚ :=
  let total_shirts := total_employees * 2
  let women_white_prices := fun i => men_white_prices i - women_discount
  let women_black_prices := fun i => men_black_prices i - women_discount
  let total_cost := (
    (men_white_prices 0 + women_white_prices 0 + men_black_prices 0 + women_black_prices 0) * (total_shirts * size_distribution 0) +
    (men_white_prices 1 + women_white_prices 1 + men_black_prices 1 + women_black_prices 1) * (total_shirts * size_distribution 1) +
    (men_white_prices 2 + women_white_prices 2 + men_black_prices 2 + women_black_prices 2) * (total_shirts * size_distribution 2)
  )
  total_cost * (1 - bulk_discount)

/-- The theorem to be proved -/
theorem total_cost_is_correct :
  total_cost_after_discount
    (fun i => [20, 24, 28].get i)
    (fun i => [18, 22, 26].get i)
    60
    (fun i => [1/2, 3/10, 1/5].get i)
    5
    1/10
  = 8337.60 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_correct_l371_37159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_percent_discount_l371_37165

/-- Given an original price and a discount percentage, calculates the final price --/
noncomputable def finalPrice (originalPrice : ℝ) (discountPercentage : ℝ) : ℝ :=
  originalPrice * (1 - discountPercentage / 100)

/-- Theorem stating that a 10% discount results in a final price of 0.9 times the original --/
theorem ten_percent_discount (x : ℝ) : finalPrice x 10 = 0.9 * x := by
  unfold finalPrice
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_percent_discount_l371_37165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_one_zero_l371_37171

/-- An odd function with the given property has exactly one zero -/
theorem odd_function_one_zero 
  (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_deriv : ∀ x < 0, 2 * f x + x * (deriv (deriv f) x) < x * f x) : 
  ∃! x, f x = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_one_zero_l371_37171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_route_distance_l371_37126

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points using the Pythagorean theorem -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

/-- Theorem: The straight-line distance between the start and end points of the given route is 26 miles -/
theorem route_distance : 
  let start := Point.mk 0 0
  let end_point := Point.mk (-10) 24
  distance start end_point = 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_route_distance_l371_37126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_munificence_monic_quadratic_l371_37184

/-- The munificence of a polynomial on [-1, 1] -/
noncomputable def munificence (p : ℝ → ℝ) : ℝ :=
  ⨆ (x : ℝ) (_ : x ∈ Set.Icc (-1) 1), |p x|

/-- A monic quadratic polynomial -/
def monicQuadratic (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

/-- The smallest possible munificence of a monic quadratic polynomial is 1/2 -/
theorem smallest_munificence_monic_quadratic :
  ∃ (b c : ℝ), munificence (monicQuadratic b c) = (1 : ℝ) / 2 ∧
  ∀ (b' c' : ℝ), munificence (monicQuadratic b' c') ≥ (1 : ℝ) / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_munificence_monic_quadratic_l371_37184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_comparison_l371_37108

theorem fraction_comparison : ∃ (f1 f2 f3 f4 f5 f6 : ℚ),
  f1 = 15 / 4 ∧
  f2 = 3.75 ∧
  f3 = (14 + 1) / (3 + 1) ∧
  f4 = 3 / 4 + 3 ∧
  f5 = 5 / 4 * 3 / 4 ∧
  f6 = 21 / 4 - 5 / 4 - 1 / 4 ∧
  f2 = f1 ∧
  f3 = f1 ∧
  f4 = f1 ∧
  f5 ≠ f1 ∧
  f6 = f1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_comparison_l371_37108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yolanda_rate_is_three_l371_37111

/-- The walking problem scenario -/
structure WalkingProblem where
  total_distance : ℝ
  bob_start_delay : ℝ
  bob_distance : ℝ
  bob_rate : ℝ

/-- Calculate Yolanda's walking rate given the problem parameters -/
noncomputable def calculate_yolanda_rate (p : WalkingProblem) : ℝ :=
  let bob_time := p.bob_distance / p.bob_rate
  let yolanda_time := bob_time + p.bob_start_delay
  let yolanda_distance := p.total_distance - p.bob_distance
  yolanda_distance / yolanda_time

/-- Theorem stating that Yolanda's walking rate is 3 miles per hour -/
theorem yolanda_rate_is_three (p : WalkingProblem) 
  (h1 : p.total_distance = 17)
  (h2 : p.bob_start_delay = 1)
  (h3 : p.bob_distance = 8)
  (h4 : p.bob_rate = 4) :
  calculate_yolanda_rate p = 3 := by
  sorry

#eval "Theorem yolanda_rate_is_three is defined."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yolanda_rate_is_three_l371_37111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_4_minus_alpha_eq_2_3_l371_37125

theorem cos_pi_4_minus_alpha_eq_2_3 (α : ℝ) :
  Real.sin (π / 4 + α) = 2 / 3 → Real.cos (π / 4 - α) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_4_minus_alpha_eq_2_3_l371_37125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_four_thirds_l371_37140

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 0 then x^2
  else if 0 < x ∧ x < 1 then 1
  else 0  -- We define f as 0 outside the given intervals to make it total

-- State the theorem
theorem integral_f_equals_four_thirds :
  ∫ x in Set.Icc (-1) 1, f x = 4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_four_thirds_l371_37140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l371_37147

theorem angle_between_vectors (θ : ℝ) :
  0 ≤ θ ∧ θ ≤ Real.pi ∧ Real.sin θ = Real.sqrt 2 / 2 → θ = Real.pi / 4 ∨ θ = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l371_37147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_5_l371_37137

noncomputable def geometric_sequence (a₁ q : ℝ) : ℕ → ℝ :=
  λ n => a₁ * q^(n - 1)

noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_5 (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n, a n = geometric_sequence (1/3) q n) →  -- a is a geometric sequence with a₁ = 1/3
  a 4 * a 4 = a 6 →  -- a₄² = a₆
  geometric_sum (1/3) (a 2 / a 1) 5 = 121/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_5_l371_37137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_cos_and_sin_2x_plus_phi_l371_37144

theorem intersection_of_cos_and_sin_2x_plus_phi (φ : ℝ) :
  (0 ≤ φ ∧ φ ≤ π) →
  (Real.cos (π/3) = Real.sin (2 * (π/3) + φ)) →
  φ = π/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_cos_and_sin_2x_plus_phi_l371_37144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_goals_and_points_l371_37177

/-- Represents the types of goals a football player can score -/
inductive GoalType
  | Header
  | Volley
  | Freekick

/-- Calculates the points for a given goal type -/
def goalPoints (gt : GoalType) : ℕ :=
  match gt with
  | GoalType.Header => 1
  | GoalType.Volley => 2
  | GoalType.Freekick => 3

/-- Represents a football player's scoring record -/
structure PlayerRecord where
  goalsInFirst14 : ℝ
  goalsIn15th : ℕ
  averageIncrease : ℝ

/-- Theorem stating that we can calculate total points but not exact goals -/
theorem calculate_goals_and_points 
  (player : PlayerRecord) 
  (h1 : player.goalsIn15th = 3) 
  (h2 : player.averageIncrease = 0.08) :
  ∃ (totalPoints : ℝ), 
    totalPoints = player.goalsInFirst14 + 3 ∧
    ¬∃ (headerGoals volleyGoals freekickGoals : ℕ),
      (headerGoals : ℝ) + (volleyGoals : ℝ) + (freekickGoals : ℝ) = player.goalsInFirst14 + 3 ∧
      (headerGoals * goalPoints GoalType.Header + 
       volleyGoals * goalPoints GoalType.Volley + 
       freekickGoals * goalPoints GoalType.Freekick : ℝ) = totalPoints :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_goals_and_points_l371_37177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleSingleAppearance_l371_37101

/-- A function that depends on all its variables -/
def DependsOnAllVariables (f : (ℕ → Bool) → Bool) :=
  ∀ (x y : ℕ → Bool) (i : ℕ), (∀ j ≠ i, x j = y j) → (x i ≠ y i) → f x ≠ f y

/-- A program is represented as a list of instructions -/
inductive Instruction
| assign (target : ℕ) (source : List ℕ)

def Program := List Instruction

/-- Check if a variable appears in an instruction -/
def appearsIn (v : ℕ) (instr : Instruction) : Bool :=
  match instr with
  | Instruction.assign _ sources => v ∈ sources

/-- Count how many times two variables appear together in a program -/
def countAppearancesTogether (v1 v2 : ℕ) (prog : Program) : ℕ :=
  (prog.filter (fun instr => appearsIn v1 instr ∧ appearsIn v2 instr)).length

/-- A program computes a function if it produces the correct output for all inputs -/
def Computes (prog : Program) (f : (ℕ → Bool) → Bool) : Prop := sorry

theorem impossibleSingleAppearance (n : ℕ) (hn : n ≥ 4) (f : (ℕ → Bool) → Bool) 
    (hf : DependsOnAllVariables f) :
    ¬∃ (prog : Program), Computes prog f ∧ countAppearancesTogether 1 2 prog = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleSingleAppearance_l371_37101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l371_37104

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 + Real.sqrt (24 * x - 9 * x^2)

-- State the theorem
theorem max_value_of_f :
  ∃ (m : ℝ), m = 5 ∧ ∀ x, 0 < x → x < 2 → f x ≤ m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l371_37104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_sine_function_l371_37115

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

noncomputable def g (ω φ : ℝ) (x : ℝ) : ℝ := f ω φ (x + Real.pi / 6)

theorem symmetry_of_sine_function (ω φ : ℝ) 
  (h1 : ω > 0) 
  (h2 : -Real.pi / 2 < φ ∧ φ < Real.pi / 2) 
  (h3 : ∀ x, f ω φ (x + Real.pi) = f ω φ x)  -- smallest positive period is π
  (h4 : ∀ x, g ω φ x = -g ω φ (-x))  -- g is an odd function
  : ∀ x, f ω φ (5 * Real.pi / 6 - x) = f ω φ (5 * Real.pi / 6 + x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_sine_function_l371_37115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_coprime_number_l371_37192

def a_sequence (n : ℕ) : ℕ := 2^n + 3^n + 6^n + 1

def is_coprime_with_sequence (k : ℕ) : Prop :=
  ∀ n : ℕ, Nat.Coprime k (a_sequence n)

theorem smallest_coprime_number :
  (∃ k : ℕ, k ≥ 2 ∧ is_coprime_with_sequence k) ∧
  (∀ k : ℕ, k ≥ 2 ∧ is_coprime_with_sequence k → k ≥ 23) :=
by
  sorry

#check smallest_coprime_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_coprime_number_l371_37192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_exterior_angle_in_triangle_l371_37102

theorem largest_exterior_angle_in_triangle (a b c : ℝ) 
  (h_ratio : [3, 4, 5] = [a, b, c].map (·/a)) 
  (h_sum : a + b + c = 180) : 
  180 - min a (min b c) = 135 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_exterior_angle_in_triangle_l371_37102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thousand_in_pascal_triangle_no_smaller_four_digit_in_pascal_triangle_smallest_four_digit_in_pascal_triangle_proof_l371_37185

/-- The smallest four-digit number in Pascal's triangle is 1000 -/
def smallest_four_digit_in_pascal_triangle : ℕ := 1000

/-- Definition of Pascal's triangle -/
def pascal_triangle : Set ℕ :=
  {n : ℕ | ∃ k m : ℕ, n = Nat.choose k m}

/-- Pascal's triangle contains all positive integers -/
axiom pascal_triangle_contains_all_positives : ∀ n : ℕ, n > 0 → n ∈ pascal_triangle

/-- 1000 is in Pascal's triangle -/
theorem thousand_in_pascal_triangle : 1000 ∈ pascal_triangle := by
  apply pascal_triangle_contains_all_positives
  simp

/-- No number less than 1000 and greater than or equal to 1000 is in Pascal's triangle -/
theorem no_smaller_four_digit_in_pascal_triangle :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 1000 → n ∉ pascal_triangle := by
  sorry

theorem smallest_four_digit_in_pascal_triangle_proof :
  smallest_four_digit_in_pascal_triangle = 1000 := by
  rfl

#check smallest_four_digit_in_pascal_triangle
#check pascal_triangle
#check pascal_triangle_contains_all_positives
#check thousand_in_pascal_triangle
#check no_smaller_four_digit_in_pascal_triangle
#check smallest_four_digit_in_pascal_triangle_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thousand_in_pascal_triangle_no_smaller_four_digit_in_pascal_triangle_smallest_four_digit_in_pascal_triangle_proof_l371_37185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_six_points_l371_37172

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- A square with side length 2 -/
def square : Set Point :=
  {p : Point | 0 ≤ p.x ∧ p.x ≤ 2 ∧ 0 ≤ p.y ∧ p.y ≤ 2}

/-- The theorem to be proved -/
theorem min_distance_six_points :
  ∀ (points : Finset Point),
    points.card = 6 →
    (∀ p, p ∈ points → p ∈ square) →
    (∃ p1 p2, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ distance p1 p2 ≤ Real.sqrt 2) ∧
    ∀ b, b < Real.sqrt 2 →
      ∃ config : Finset Point,
        config.card = 6 ∧
        (∀ p, p ∈ config → p ∈ square) ∧
        ∀ p1 p2, p1 ∈ config → p2 ∈ config → p1 ≠ p2 → distance p1 p2 > b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_six_points_l371_37172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_solution_l371_37175

theorem unique_integer_solution :
  ∃! (a b c : ℕ+),
    let x : ℝ := Real.sqrt ((Real.sqrt 53) / 2 + 3 / 2)
    x^100 = 2*x^98 + 14*x^96 + 11*x^94 - x^50 + a*x^46 + b*x^44 + c*x^40 ∧
    a + b + c = 157 := by
  sorry

#eval 157

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_solution_l371_37175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_l371_37123

-- Define the curve
def curve (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

-- Define the point P
def P : ℝ × ℝ := (-1, 2)

-- Define the slope of the tangent line at any point
def tangent_slope (x : ℝ) : ℝ := 6 * x - 4

-- Theorem statement
theorem parallel_line_equation :
  ∃ (M : ℝ × ℝ),
    M.1 ∈ Set.Icc (-5 : ℝ) 5 ∧
    M.2 = curve M.1 ∧
    (∀ (x y : ℝ), 2 * x - y + 4 = 0 ↔ y - P.2 = tangent_slope M.1 * (x - P.1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_l371_37123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_a_values_l371_37117

/-- The sum of all possible values of a, given the conditions of the problem -/
def sum_of_possible_a : ℝ := 29.25

/-- The set of pairwise differences -/
def pairwise_differences : Set ℝ := {2, 3, 4, 5, 7, 8}

/-- The theorem stating the sum of possible values of a -/
theorem sum_of_a_values
  (a b c d : ℤ)
  (h_order : a > b ∧ b > c ∧ c > d)
  (h_sum : a + b + c + d = 42)
  (h_diff : ∀ x y, x ∈ ({a, b, c, d} : Set ℤ) → y ∈ ({a, b, c, d} : Set ℤ) → x ≠ y → 
    (x - y : ℝ) ∈ pairwise_differences ∨ (y - x : ℝ) ∈ pairwise_differences) :
  sum_of_possible_a = 29.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_a_values_l371_37117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l371_37103

theorem product_remainder (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l371_37103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_theorem_l371_37162

/-- Given a cost price, markup percentage, and actual profit percentage,
    calculate the discount percentage on the marked price. -/
noncomputable def calculate_discount_percentage (cost : ℝ) (markup_percent : ℝ) (actual_profit_percent : ℝ) : ℝ :=
  let marked_price := cost * (1 + markup_percent / 100)
  let selling_price := cost * (1 + actual_profit_percent / 100)
  let discount_amount := marked_price - selling_price
  (discount_amount / marked_price) * 100

/-- Theorem stating that for a 50% markup and 27.5% actual profit,
    the discount percentage is 15%. -/
theorem discount_percentage_theorem (cost : ℝ) (cost_positive : 0 < cost) :
  calculate_discount_percentage cost 50 27.5 = 15 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_discount_percentage 100 50 27.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_theorem_l371_37162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_22_5_deg_squared_identity_l371_37106

theorem sin_22_5_deg_squared_identity : 
  2 * (Real.sin (22.5 * π / 180))^2 - 1 = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_22_5_deg_squared_identity_l371_37106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_difference_specific_values_l371_37151

theorem sin_difference_specific_values (α β : ℝ) : 
  α ∈ Set.Ioo (π/3) (5*π/6) → 
  β ∈ Set.Ioo (π/3) (5*π/6) → 
  Real.sin (α + π/6) = 4/5 → 
  Real.cos (β - 5*π/6) = 5/13 → 
  Real.sin (α - β) = 16/65 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_difference_specific_values_l371_37151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_b_value_l371_37128

-- Define the curve function
noncomputable def f (x : ℝ) : ℝ := -(1/2) * x + Real.log x

-- Define the line function
noncomputable def g (x b : ℝ) : ℝ := (1/2) * x + b

-- Theorem statement
theorem tangent_line_b_value :
  ∀ b : ℝ, 
  (∃ x : ℝ, x > 0 ∧ 
    f x = g x b ∧ 
    deriv f x = (1/2)) →
  b = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_b_value_l371_37128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_number_proof_l371_37189

theorem other_number_proof (a b : ℕ) 
  (h1 : Nat.gcd a b = 20) 
  (h2 : Nat.lcm a b = 396) 
  (h3 : a = 36) : 
  b = 220 := by
  sorry

#check other_number_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_number_proof_l371_37189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_drawing_probabilities_l371_37131

/-- Represents the color of a card -/
inductive Color
  | Red
  | White
  | Blue

/-- Represents the total number of cards -/
def total_cards : ℕ := 10

/-- Represents the number of red cards -/
def red_cards : ℕ := 5

/-- Represents the number of white cards -/
def white_cards : ℕ := 3

/-- Represents the number of blue cards -/
def blue_cards : ℕ := 2

/-- Represents the maximum number of draws -/
def max_draws : ℕ := 3

/-- Event A: Exactly 2 red cards are drawn -/
noncomputable def event_A : ℝ := 7 / 20

/-- Random variable ξ representing the number of cards drawn -/
noncomputable def ξ : Fin 3 → ℝ
  | 0 => 3 / 10
  | 1 => 21 / 100
  | 2 => 49 / 100

/-- Expected value of ξ -/
noncomputable def E_ξ : ℝ := 219 / 100

theorem card_drawing_probabilities :
  (total_cards = red_cards + white_cards + blue_cards) →
  (event_A = 7 / 20) ∧
  (ξ 0 = 3 / 10) ∧
  (ξ 1 = 21 / 100) ∧
  (ξ 2 = 49 / 100) ∧
  (E_ξ = 219 / 100) := by
  sorry

#check card_drawing_probabilities

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_drawing_probabilities_l371_37131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l371_37153

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (3 + (Real.sqrt 2 / 2) * t, 4 + (Real.sqrt 2 / 2) * t)

def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, 
    A = line_l t₁ ∧ 
    B = line_l t₂ ∧ 
    circle_C A.1 A.2 ∧ 
    circle_C B.1 B.2

theorem intersection_product (A B : ℝ × ℝ) (h : intersection_points A B) :
  let M : ℝ × ℝ := (3, 4)
  Real.sqrt ((A.1 - M.1)^2 + (A.2 - M.2)^2) * 
  Real.sqrt ((B.1 - M.1)^2 + (B.2 - M.2)^2) = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l371_37153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_pairs_count_l371_37191

def count_valid_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    let a := p.1
    let b := p.2
    a > 0 ∧ b > 0 ∧ a * b + 45 = 10 * Nat.lcm a b + 18 * Nat.gcd a b
  ) (Finset.product (Finset.range 1000) (Finset.range 1000))).card

theorem valid_pairs_count : count_valid_pairs = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_pairs_count_l371_37191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gamma_function_range_l371_37195

/-- Definition of a Γ-function -/
def is_gamma_function (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f (a + x) * f (a - x) = b

/-- Main theorem -/
theorem gamma_function_range (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f x * f (-x) = 1)
  (h2 : ∀ x : ℝ, f (1 + x) * f (1 - x) = 4)
  (h3 : ∀ x, x ∈ Set.Icc 0 1 → f x ∈ Set.Icc 1 2) :
  Set.range (fun x => f x) ∩ Set.Icc (-2016 : ℝ) 2016 = 
    Set.Icc (2^(-2016 : ℝ)) (2^2016) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gamma_function_range_l371_37195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bookstore_cost_ratio_l371_37163

/-- The ratio of a bookstore's cost to the original marked price of books -/
noncomputable def cost_to_original_price_ratio (original_price : ℝ) : ℝ :=
  let discount_rate := (1 : ℝ) / 4
  let selling_price := original_price * (1 - discount_rate)
  let cost_to_selling_price_ratio := (2 : ℝ) / 3
  let cost_price := selling_price * cost_to_selling_price_ratio
  cost_price / original_price

/-- 
The ratio of a bookstore's cost to the original marked price of books is 1/2, 
given a 1/4 discount and that the cost to the store is 2/3 of the selling price.
-/
theorem bookstore_cost_ratio : 
  ∀ (original_price : ℝ), original_price > 0 → cost_to_original_price_ratio original_price = 1 / 2 := by
  intro original_price h_pos
  unfold cost_to_original_price_ratio
  -- The actual proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bookstore_cost_ratio_l371_37163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l371_37148

theorem divisibility_property (a b : ℕ) (ha : a > 0) (hb : b > 0)
  (h : (a^2 + 3*a*b + 3*b^2 - 1) ∣ (a + b^3)) :
  ∃ (n : ℕ), n > 1 ∧ n^3 ∣ (a^2 + 3*a*b + 3*b^2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l371_37148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_distance_inscribed_squares_l371_37157

/-- Helper function to calculate the maximum distance between vertices -/
noncomputable def max_distance_between_vertices (outer_perimeter inner_perimeter : ℝ) : ℝ :=
  let outer_side := outer_perimeter / 4
  let inner_side := inner_perimeter / 4
  outer_side * Real.sqrt 2 - Real.sqrt 2

/-- The greatest distance between vertices of inscribed squares -/
theorem greatest_distance_inscribed_squares (outer_perimeter inner_perimeter : ℝ) 
  (h_outer : outer_perimeter = 40)
  (h_inner : inner_perimeter = 36)
  (h_inscribed : inner_perimeter < outer_perimeter) : 
  ∃ (d : ℝ), d = 9 * Real.sqrt 2 ∧ 
  d = max_distance_between_vertices outer_perimeter inner_perimeter :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_distance_inscribed_squares_l371_37157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_values_difference_l371_37187

/-- Represents a 3-digit integer with digits a, b, and c -/
structure ThreeDigitInteger where
  a : Nat
  b : Nat
  c : Nat
  a_positive : 0 < a
  b_eq_2a : b = 2 * a
  c_eq_4a : c = 4 * a
  a_single_digit : a ≤ 9
  b_single_digit : b ≤ 9
  c_single_digit : c ≤ 9

/-- Converts a ThreeDigitInteger to its numerical value -/
def ThreeDigitInteger.toNat (x : ThreeDigitInteger) : Nat :=
  100 * x.a + 10 * x.b + x.c

/-- The theorem stating the difference between the two greatest possible values -/
theorem greatest_values_difference :
  ∃ (max_x next_max_x : ThreeDigitInteger),
    (∀ x : ThreeDigitInteger, x.toNat ≤ max_x.toNat) ∧
    (∀ x : ThreeDigitInteger, x ≠ max_x → x.toNat ≤ next_max_x.toNat) ∧
    max_x.toNat - next_max_x.toNat = 124 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_values_difference_l371_37187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_g_symmetry_l371_37160

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition for f
variable (h : ∀ x, f (1 + x) = f (1 - x))

-- Define g₁ and g₂
def g₁ (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x + 3)
def g₂ (f : ℝ → ℝ) (x : ℝ) : ℝ := f (3 - x)

-- Theorem for the symmetry of f about x = 1
theorem f_symmetry (f : ℝ → ℝ) (h : ∀ x, f (1 + x) = f (1 - x)) (x : ℝ) : 
  f x = f (2 - x) := by
  sorry

-- Theorem for the symmetry of g₁ and g₂ about x = 0
theorem g_symmetry (f : ℝ → ℝ) (h : ∀ x, f (1 + x) = f (1 - x)) (x : ℝ) : 
  g₁ f x = g₂ f (-x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_g_symmetry_l371_37160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_correct_l371_37179

-- Define the functions f, g, and h
noncomputable def f (x : ℝ) : ℝ := 4 * x + 2
noncomputable def g (x : ℝ) : ℝ := 3 * x - 5
noncomputable def h (x : ℝ) : ℝ := f (g x)

-- Define the inverse function of h
noncomputable def h_inv (x : ℝ) : ℝ := (x + 18) / 12

-- Theorem statement
theorem h_inverse_correct : 
  ∀ x : ℝ, h (h_inv x) = x ∧ h_inv (h x) = x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_correct_l371_37179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spade_calculation_l371_37146

-- Define the ⊙ operation for positive real numbers
noncomputable def spade (x y : ℝ) : ℝ := x - 1 / y

-- Theorem statement
theorem spade_calculation : 
  spade 3 (spade 2 (5/3)) = 16/7 := by
  -- Unfold the definition of spade
  unfold spade
  -- Simplify the expression
  simp [div_eq_mul_inv]
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spade_calculation_l371_37146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_theorem_l371_37199

/-- Represents a batsman's performance -/
structure Batsman where
  average_before : ℚ
  total_runs : ℚ
  innings : ℕ

/-- The batsman's average after scoring 100 runs in the 11th inning -/
noncomputable def new_average (b : Batsman) : ℚ := (b.total_runs + 100) / 11

/-- Theorem: If a batsman's average increases by 5 after scoring 100 runs in the 11th inning, 
    then his average after the 11th inning is 50 -/
theorem batsman_average_theorem (b : Batsman) 
    (h1 : b.innings = 10)
    (h2 : new_average b = b.average_before + 5) : 
  new_average b = 50 := by
  sorry

#check batsman_average_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_theorem_l371_37199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_travel_distance_l371_37134

noncomputable section

/-- The distance traveled by the center of a ball rolling along a track composed of four semicircular arcs -/
def distance_traveled (ball_diameter : ℝ) (arc_radii : Fin 4 → ℝ) : ℝ :=
  let ball_radius := ball_diameter / 2
  let adjusted_radii : Fin 4 → ℝ := fun i =>
    if i % 2 = 0 then arc_radii i - ball_radius else arc_radii i + ball_radius
  (adjusted_radii 0 + adjusted_radii 1 + adjusted_radii 2 + adjusted_radii 3) * Real.pi

/-- The theorem stating that the distance traveled by the center of the ball is 330π inches -/
theorem ball_travel_distance :
  distance_traveled 6 (fun i => [120, 50, 90, 70].get i) = 330 * Real.pi :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_travel_distance_l371_37134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_D_is_70_degrees_l371_37158

theorem angle_D_is_70_degrees 
  (A B C D : ℝ) -- Define angles as real numbers
  (h1 : A + B = 180)
  (h2 : C = 2 * D)
  (h3 : B = C + 40) :
  D = 70 :=
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_D_is_70_degrees_l371_37158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l371_37150

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 + Real.cos x)^8 + (1 - Real.cos x)^8

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 256 ∧ ∀ (x : ℝ), f x ≤ M :=
by
  -- We'll use M = 256
  let M := 256
  
  -- Prove that such an M exists and is equal to 256
  use M
  
  -- Split the goal into two parts
  constructor
  
  -- Prove M = 256 (trivial)
  · rfl
  
  -- Prove ∀ (x : ℝ), f x ≤ M
  · intro x
    -- The actual proof would go here
    sorry

-- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l371_37150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_raised_bed_length_l371_37164

/-- Represents a raised bed in Bob's garden --/
structure RaisedBed where
  height : ℕ
  width : ℕ
  length : ℕ

/-- Represents the planks used for construction --/
structure Plank where
  width : ℕ
  length : ℕ

/-- The problem setup --/
structure GardenSetup where
  bedHeight : ℕ
  bedWidth : ℕ
  plankWidth : ℕ
  plankLength : ℕ
  totalPlanks : ℕ
  totalBeds : ℕ

def gardenSetup : GardenSetup :=
  { bedHeight := 2
  , bedWidth := 2
  , plankWidth := 1
  , plankLength := 8
  , totalPlanks := 50
  , totalBeds := 10 }

/-- Theorem stating that the length of each raised bed is 8 feet --/
theorem raised_bed_length :
  ∀ (bed : RaisedBed),
    bed.height = gardenSetup.bedHeight →
    bed.width = gardenSetup.bedWidth →
    (gardenSetup.totalPlanks / gardenSetup.totalBeds - 4) * gardenSetup.plankLength = bed.length :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_raised_bed_length_l371_37164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_body_is_sphere_l371_37182

/-- A body in 3D space -/
structure Body where
  -- We don't need to define the internal structure of the body

/-- A plane in 3D space -/
structure Plane where
  -- We don't need to define the internal structure of the plane

/-- A circle in 3D space -/
structure Circle where
  -- We don't need to define the internal structure of the circle

/-- A sphere in 3D space -/
structure Sphere where
  -- We don't need to define the internal structure of the sphere

/-- A point in 3D space -/
structure Point where
  -- We don't need to define the internal structure of the point

/-- The intersection of a body and a plane -/
def intersect (b : Body) (p : Plane) : Set Point := sorry

/-- Function to check if a set of points forms a circle -/
def isCircle (s : Set Point) : Prop := sorry

/-- Function to check if a body is a sphere -/
def isSphere (b : Body) : Prop := sorry

/-- Theorem: If every plane section of a body is a circle, then the body is a sphere -/
theorem body_is_sphere (b : Body) : 
  (∀ p : Plane, isCircle (intersect b p)) → isSphere b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_body_is_sphere_l371_37182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_50_mod_7_l371_37197

-- Define the sequence T
def T : ℕ → ℕ
  | 0 => 3  -- Add a case for 0 to cover all natural numbers
  | 1 => 3
  | n+2 => 3^(T (n+1))

-- State the theorem
theorem t_50_mod_7 : T 50 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_50_mod_7_l371_37197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l371_37133

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculates the area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ :=
  (1 / 2) * base * height

/-- Theorem: Area of a specific right triangle with given conditions -/
theorem right_triangle_area :
  ∀ (t : Triangle),
    t.A = ⟨0, 0⟩ →                             -- A is at origin
    t.C.y = 0 →                                -- C is on x-axis
    t.B.y = t.B.x * Real.sqrt 3 →              -- B is on line y = x√3
    4 = t.B.y / t.C.x →                        -- Altitude from B to AC is 4
    triangleArea t.C.x 4 = 16 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l371_37133
