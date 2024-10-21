import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_cos_2x_l657_65715

noncomputable def f (x : ℝ) := Real.cos (2 * x)

theorem smallest_positive_period_of_cos_2x :
  ∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧
  (∀ T' : ℝ, T' > 0 → (∀ x : ℝ, f (x + T') = f x) → T ≤ T') ∧
  T = π := by
  sorry

#check smallest_positive_period_of_cos_2x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_cos_2x_l657_65715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l657_65749

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_b_pos : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- A point on the hyperbola -/
structure Point (h : Hyperbola) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The right focus of the hyperbola -/
noncomputable def right_focus (h : Hyperbola) : ℝ × ℝ :=
  (Real.sqrt (h.a^2 + h.b^2), 0)

/-- The left focus of the hyperbola -/
noncomputable def left_focus (h : Hyperbola) : ℝ × ℝ :=
  (-Real.sqrt (h.a^2 + h.b^2), 0)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: If the maximum value of |PF₂|²/|PF₁| is 8a for any point P on the left branch,
    then the eccentricity e satisfies 1 < e ≤ √3 -/
theorem hyperbola_eccentricity_range (h : Hyperbola) :
  (∀ (p : Point h), p.x < 0 →
    (distance (p.x, p.y) (right_focus h))^2 / distance (p.x, p.y) (left_focus h) ≤ 8 * h.a) →
  1 < eccentricity h ∧ eccentricity h ≤ Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l657_65749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_combination_l657_65767

/-- Given two parallel vectors a and b in R², prove that their linear combination 
    2a + 3b equals (-4, -8) -/
theorem parallel_vectors_combination (a b : ℝ × ℝ) (m : ℝ) 
    (h1 : a = (1, 2)) 
    (h2 : b = (-2, m)) 
    (h3 : ∃ (k : ℝ), k ≠ 0 ∧ a = k • b) : 
  (2 : ℝ) • a + (3 : ℝ) • b = (-4, -8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_combination_l657_65767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_vector_sum_l657_65771

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.cos x
noncomputable def g (x : ℝ) : ℝ := Real.tan x

-- Define the domain of f
def domain : Set ℝ := Set.Icc 0 (2 * Real.pi)

-- Define the intersection points M and N
def M : ℝ × ℝ := sorry
def N : ℝ × ℝ := sorry

-- State that M and N are intersection points of f and g
axiom M_intersection : M.1 ∈ domain ∧ f M.1 = g M.1 ∧ M.2 = f M.1
axiom N_intersection : N.1 ∈ domain ∧ f N.1 = g N.1 ∧ N.2 = f N.1

-- Define the vector sum
def vector_sum : ℝ × ℝ := (M.1 + N.1, M.2 + N.2)

-- State the theorem
theorem intersection_vector_sum : 
  Real.sqrt (vector_sum.1^2 + vector_sum.2^2) = Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_vector_sum_l657_65771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_area_AEFC_l657_65770

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let d := 4 -- side length in cm
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = d^2 ∧
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = d^2 ∧
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = d^2

-- Define point D
def PointD (A B D : ℝ × ℝ) : Prop :=
  (D.1 - A.1)^2 + (D.2 - A.2)^2 = 9 * ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- Define point E
def PointE (A C E : ℝ × ℝ) : Prop :=
  (E.1 - A.1)^2 + (E.2 - A.2)^2 = 1/16 * ((C.1 - A.1)^2 + (C.2 - A.2)^2)

-- Define point F
def PointF (B C D E F : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, F = (1 - t) • B + t • C ∧
             ∃ s : ℝ, F = (1 - s) • D + s • E

-- Define area of quadrilateral
noncomputable def area_quadrilateral (A E F C : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem exists_unique_area_AEFC 
  (A B C D E F : ℝ × ℝ) 
  (h1 : Triangle A B C) 
  (h2 : PointD A B D) 
  (h3 : PointE A C E) 
  (h4 : PointF B C D E F) :
  ∃! area : ℝ, area = area_quadrilateral A E F C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_area_AEFC_l657_65770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l657_65700

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) ^ Real.sqrt (x^2 - 4*a*x + 8)

-- Define the monotonicity of f on the interval [2, 6]
def is_monotonic_on_interval (a : ℝ) : Prop :=
  Monotone (fun x => f a x) ∨ StrictMono (fun x => f a x) ∨ 
  StrictAnti (fun x => f a x) ∨ Antitone (fun x => f a x)

-- State the theorem
theorem range_of_a (a : ℝ) :
  is_monotonic_on_interval a → a ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l657_65700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_mixture_percentage_l657_65739

/-- Represents a mixture of acid and water -/
structure Mixture where
  acid : ℚ
  water : ℚ

/-- Calculates the percentage of acid in a mixture -/
def acid_percentage (m : Mixture) : ℚ :=
  m.acid / (m.acid + m.water) * 100

theorem original_mixture_percentage 
  (original : Mixture)
  (h1 : acid_percentage { acid := original.acid, water := original.water + 1 } = 25)
  (h2 : acid_percentage { acid := original.acid + 1, water := original.water + 1 } = 40) :
  acid_percentage original = 100/3 := by
  sorry

#eval acid_percentage { acid := 1, water := 2 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_mixture_percentage_l657_65739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_of_triangle_l657_65713

/-- Given a triangle with vertices at (3,3), (7,3), and (5,8), 
    the length of its longest side is √29 units. -/
theorem longest_side_of_triangle : 
  let A : ℝ × ℝ := (3, 3)
  let B : ℝ × ℝ := (7, 3)
  let C : ℝ × ℝ := (5, 8)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let sides := [dist A B, dist B C, dist C A]
  Real.sqrt 29 = List.maximum sides :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_of_triangle_l657_65713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_isosceles_l657_65706

noncomputable section

-- Define the lines
def line1 (x : ℝ) : ℝ := 4 * x + 3
def line2 (x : ℝ) : ℝ := -4 * x + 5
def line3 : ℝ := -3

-- Define the intersection points
def point1 : ℝ × ℝ := (1/4, 4)
def point2 : ℝ × ℝ := (-3/2, -3)
def point3 : ℝ × ℝ := (2, -3)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem triangle_is_isosceles :
  let d12 := distance point1 point2
  let d13 := distance point1 point3
  let d23 := distance point2 point3
  (d12 = d13 ∧ d12 ≠ d23) ∨ (d12 = d23 ∧ d12 ≠ d13) ∨ (d13 = d23 ∧ d13 ≠ d12) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_isosceles_l657_65706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_like_inequality_l657_65796

-- Define the Fibonacci-like sequence
def F : ℕ → ℕ
  | 0 => 1  -- Add this case to cover Nat.zero
  | 1 => 1
  | 2 => 2
  | (n + 3) => F (n + 2) + F (n + 1)

-- State the theorem
theorem fibonacci_like_inequality (n : ℕ) (h : n > 0) :
  (F (n + 1) : ℝ) ^ (1 / n : ℝ) ≥ 1 + (F n : ℝ) ^ (-1 / n : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_like_inequality_l657_65796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_property_l657_65731

/-- The function f(x) defined as x + 1/(x+a) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + 1 / (x + a)

theorem function_extrema_property :
  ∃ (a : ℝ), ∃ (y : ℝ),
    (∀ x, f a x ≥ f a y) ∧
    (∀ x, f a x ≤ f a (2*y)) ∧
    (a = 1) := by
  -- We claim that a = 1 and y = -3 satisfy the conditions
  use 1, -3
  constructor
  · sorry  -- Proof that f 1 x ≥ f 1 (-3) for all x
  constructor
  · sorry  -- Proof that f 1 x ≤ f 1 (-6) for all x
  · rfl    -- Proof that 1 = 1


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_property_l657_65731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_trajectory_and_dot_product_l657_65793

/-- The ellipse with right focus F -/
structure Ellipse (a : ℝ) where
  equation : ∀ x y : ℝ, x^2 / (1 + a^2) + y^2 = 1

/-- The right focus of the ellipse -/
def rightFocus (a : ℝ) : ℝ × ℝ := (a, 0)

/-- Moving point M on x-axis -/
def M (m : ℝ) : ℝ × ℝ := (m, 0)

/-- Moving point N on y-axis -/
def N (n : ℝ) : ℝ × ℝ := (0, n)

/-- Vector MN -/
def vecMN (m n : ℝ) : ℝ × ℝ := (-m, n)

/-- Vector NF -/
def vecNF (a n : ℝ) : ℝ × ℝ := (a, -n)

/-- Dot product of two 2D vectors -/
def dotProduct (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Vector OM -/
def vecOM (m : ℝ) : ℝ × ℝ := (m, 0)

/-- Vector ON -/
def vecON (n : ℝ) : ℝ × ℝ := (0, n)

/-- Vector PO -/
def vecPO (x y : ℝ) : ℝ × ℝ := (-x, -y)

/-- Point S on line x = -a -/
def S (a : ℝ) : ℝ × ℝ := (-a, a^2)

/-- Point T on line x = -a -/
def T (a : ℝ) : ℝ × ℝ := (-a, a^2)

/-- Vector FS -/
def vecFS (a : ℝ) : ℝ × ℝ := (-2*a, a^2)

/-- Vector FT -/
def vecFT (a : ℝ) : ℝ × ℝ := (-2*a, a^2)

theorem ellipse_trajectory_and_dot_product (a : ℝ) (h : a > 0) :
  (∀ m n : ℝ, dotProduct (vecMN m n) (vecNF a n) = 0) →
  (∀ m n x y : ℝ, vecOM m = vecON n + vecON n + vecPO x y) →
  (∃ C : ℝ → ℝ, C = λ x ↦ -a*x) ∧
  (dotProduct (vecFS a) (vecFT a) = 4*a^2 + a^4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_trajectory_and_dot_product_l657_65793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_l657_65721

-- Define the curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, 2 + 2 * Real.sin α)

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (Real.sqrt 3 - (Real.sqrt 3 / 2) * t, 3 + (1 / 2) * t)

-- Define point A
noncomputable def point_A : ℝ × ℝ := (-Real.sqrt 3, 3)

-- Define the theorem
theorem curve_line_intersection :
  ∃ (α θ : ℝ),
    -- Point A is on curve C
    curve_C α = point_A ∧
    -- θ is the polar angle of point A
    θ = 2 * Real.pi / 3 ∧
    θ > Real.pi / 2 ∧ θ < Real.pi ∧
    -- The distance between A and B is 2√3
    ∃ (t : ℝ),
      let B := line_l t
      (B.1 + Real.sqrt 3 * B.2 - 4 * Real.sqrt 3 = 0) ∧  -- B is on line l
      (B.2 = -Real.sqrt 3 * B.1) ∧  -- B is on ray OA
      ((B.1 - point_A.1)^2 + (B.2 - point_A.2)^2 = 12)  -- |AB|^2 = (2√3)^2 = 12
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_l657_65721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_in_ellipse_side_length_l657_65760

/-- An ellipse with equation x^2 + 4y^2 = 4 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + 4 * p.2^2 = 4}

/-- An equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ
  is_equilateral : ∀ i j : Fin 3, i ≠ j → 
    Real.sqrt ((vertices i).1 - (vertices j).1)^2 + ((vertices i).2 - (vertices j).2)^2 = 
    Real.sqrt ((vertices 0).1 - (vertices 1).1)^2 + ((vertices 0).2 - (vertices 1).2)^2

/-- The triangle is inscribed in the ellipse -/
def inscribed (t : EquilateralTriangle) : Prop :=
  ∀ i : Fin 3, t.vertices i ∈ Ellipse

/-- One vertex is at (0, 1) -/
def has_vertex_at_origin (t : EquilateralTriangle) : Prop :=
  ∃ i : Fin 3, t.vertices i = (0, 1)

/-- One altitude is on the y-axis -/
def has_altitude_on_y_axis (t : EquilateralTriangle) : Prop :=
  ∃ i j : Fin 3, i ≠ j ∧ (t.vertices i).1 = 0 ∧ (t.vertices j).1 = 0

/-- The side length is √(m/n) where m and n are coprime positive integers -/
def side_length_is_sqrt_m_over_n (t : EquilateralTriangle) (m n : ℕ) : Prop :=
  ∃ i j : Fin 3, i ≠ j ∧
  Real.sqrt ((t.vertices i).1 - (t.vertices j).1)^2 + ((t.vertices i).2 - (t.vertices j).2)^2 = 
  Real.sqrt (m / n : ℚ) ∧
  Nat.Coprime m n

theorem equilateral_triangle_in_ellipse_side_length 
  (t : EquilateralTriangle) (m n : ℕ) 
  (h1 : inscribed t) 
  (h2 : has_vertex_at_origin t) 
  (h3 : has_altitude_on_y_axis t) 
  (h4 : side_length_is_sqrt_m_over_n t m n) : 
  m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_in_ellipse_side_length_l657_65760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_canoe_kayak_difference_is_seven_l657_65764

/-- Represents the rental business scenario --/
structure RentalBusiness where
  canoe_price : ℕ
  kayak_price : ℕ
  canoe_kayak_ratio : ℚ
  total_revenue : ℕ

/-- Calculates the difference between canoes and kayaks rented --/
def canoe_kayak_difference (rb : RentalBusiness) : ℤ :=
  let canoes := (rb.total_revenue * rb.canoe_kayak_ratio.den) / 
    (rb.canoe_price * rb.canoe_kayak_ratio.den + rb.kayak_price * rb.canoe_kayak_ratio.num)
  let kayaks := canoes * rb.canoe_kayak_ratio.num / rb.canoe_kayak_ratio.den
  ↑canoes - kayaks

/-- Theorem stating the difference between canoes and kayaks rented --/
theorem canoe_kayak_difference_is_seven : 
  canoe_kayak_difference { canoe_price := 12
                         , kayak_price := 18
                         , canoe_kayak_ratio := 3 / 2
                         , total_revenue := 504 } = 7 := by
  sorry

#eval canoe_kayak_difference { canoe_price := 12
                             , kayak_price := 18
                             , canoe_kayak_ratio := 3 / 2
                             , total_revenue := 504 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_canoe_kayak_difference_is_seven_l657_65764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_runners_meet_at_start_l657_65782

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ

/-- Represents the circular track -/
def track_length : ℝ := 400

/-- The time at which all runners meet again at the starting point -/
def meeting_time : ℝ := 1200

/-- The list of runners with their respective speeds -/
def runners : List Runner := [
  ⟨4.0⟩,
  ⟨5.0⟩,
  ⟨6.0⟩,
  ⟨7.0⟩
]

/-- 
  Theorem stating that all runners meet at the starting point 
  after the specified meeting time
-/
theorem runners_meet_at_start : 
  ∀ (r : Runner), r ∈ runners → 
    ∃ (n : ℕ), r.speed * meeting_time = n * track_length :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_runners_meet_at_start_l657_65782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_approximation_l657_65798

/-- Represents the problem of Xavier's journey from P to Q -/
structure JourneyProblem where
  distance : ℝ  -- Distance between P and Q in km
  initialSpeed : ℝ  -- Initial speed in km/h
  speedIncrease : ℝ  -- Speed increase in km/h
  increaseInterval : ℝ  -- Time interval for speed increase in minutes

/-- Calculates the time taken for Xavier's journey -/
noncomputable def calculateJourneyTime (problem : JourneyProblem) : ℝ :=
  sorry  -- Proof to be implemented

/-- Theorem stating that the journey time is approximately 53.14 minutes -/
theorem journey_time_approximation (problem : JourneyProblem) 
  (h1 : problem.distance = 60)
  (h2 : problem.initialSpeed = 60)
  (h3 : problem.speedIncrease = 10)
  (h4 : problem.increaseInterval = 12) :
  abs (calculateJourneyTime problem - 53.14) < 0.01 := by sorry

#check journey_time_approximation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_approximation_l657_65798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flowers_per_pot_l657_65735

theorem flowers_per_pot (num_pots sticks_per_pot total_items flowers_per_pot : ℕ) 
  (h1 : num_pots = 466)
  (h2 : sticks_per_pot = 181)
  (h3 : total_items = 109044)
  (h4 : total_items = num_pots * sticks_per_pot + num_pots * flowers_per_pot) :
  flowers_per_pot = 53 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flowers_per_pot_l657_65735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_ratio_l657_65724

noncomputable def area_triangle (a b c : ℝ) : ℝ := sorry
noncomputable def area_hexagon (a b c : ℝ) : ℝ := sorry

theorem hexagon_area_ratio (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let S_triangle := area_triangle a b c
  let S_hexagon := area_hexagon a b c
  S_hexagon / S_triangle ≥ 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_ratio_l657_65724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_ratio_is_one_third_l657_65707

-- Define the shares as rational numbers instead of real numbers
def bills_share : ℚ := 300
def bobs_share : ℚ := 900

-- Define the ratio of their shares
noncomputable def share_ratio : ℚ := bills_share / bobs_share

-- Theorem to prove
theorem profit_ratio_is_one_third : share_ratio = 1/3 := by
  -- Unfold the definition of share_ratio
  unfold share_ratio
  -- Simplify the fraction
  simp [bills_share, bobs_share]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_ratio_is_one_third_l657_65707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_change_l657_65751

theorem profit_change (initial_profit : ℝ) : 
  (initial_profit * 1.1 * 0.8 * 1.5 - initial_profit) / initial_profit = 0.32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_change_l657_65751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l657_65733

/-- Represents the sum of the first n terms of an arithmetic sequence -/
noncomputable def S (n : ℕ) (a₁ d : ℝ) : ℝ := n * a₁ + (n * (n - 1) / 2) * d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a₁ d : ℝ) 
  (h : S 8 a₁ d - S 3 a₁ d = 10) : 
  S 11 a₁ d = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l657_65733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_of_parallelepiped_l657_65769

/-- The lateral surface area of a rectangular parallelepiped with base sides a and b,
    and whose diagonal is inclined at an angle of 60° to the base plane. -/
noncomputable def lateralSurfaceArea (a b : ℝ) : ℝ :=
  2 * (a + b) * Real.sqrt (3 * (a^2 + b^2))

/-- Theorem stating that the lateral surface area of a rectangular parallelepiped
    with the given conditions is equal to the formula derived. -/
theorem lateral_surface_area_of_parallelepiped (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let diagonal_angle := Real.pi / 3  -- 60° in radians
  lateralSurfaceArea a b = 2 * (a + b) * Real.sqrt (3 * (a^2 + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_of_parallelepiped_l657_65769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_donation_proof_l657_65725

noncomputable def restaurant_donation (average_donation : ℝ) (num_customers : ℕ) : ℝ :=
  let total_donation := average_donation * (num_customers : ℝ)
  let num_increments := total_donation / 10
  2 * num_increments

theorem restaurant_donation_proof (average_donation : ℝ) (num_customers : ℕ) 
  (h1 : average_donation = 3)
  (h2 : num_customers = 40) :
  restaurant_donation average_donation num_customers = 24 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval restaurant_donation 3 40

end NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_donation_proof_l657_65725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_filling_station_discount_l657_65792

/-- Represents the percent discount offered by the filling station -/
def discount_percent : ℝ := sorry

/-- Represents the number of gallons purchased by Kim -/
def kim_gallons : ℝ := 20

/-- Represents the number of gallons purchased by Isabella -/
def isabella_gallons : ℝ := 25

/-- Represents the number of gallons not eligible for discount -/
def non_discounted_gallons : ℝ := 6

/-- The ratio of Isabella's total per-gallon discount to Kim's -/
def discount_ratio : ℝ := 1.0857142857142861

theorem filling_station_discount :
  (isabella_gallons - non_discounted_gallons) * discount_percent = 
  discount_ratio * ((kim_gallons - non_discounted_gallons) * discount_percent) ∧
  discount_percent = 0.8 := by
  sorry

#check filling_station_discount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_filling_station_discount_l657_65792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gross_profit_calculation_l657_65758

/-- Given a sales price and gross profit percentage, calculate the gross profit -/
noncomputable def calculate_gross_profit (sales_price : ℝ) (gross_profit_percentage : ℝ) : ℝ :=
  let cost := sales_price / (1 + gross_profit_percentage)
  cost * gross_profit_percentage

/-- Theorem: Given a sales price of $54 and gross profit being 125% of cost, the gross profit is $30 -/
theorem gross_profit_calculation :
  calculate_gross_profit 54 1.25 = 30 := by
  -- Unfold the definition of calculate_gross_profit
  unfold calculate_gross_profit
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gross_profit_calculation_l657_65758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_on_segment_l657_65745

-- Define the triangle ABC
variable (A B C : EuclideanPlane) 

-- Define the segment EF
variable (E F : EuclideanPlane)

-- Define the incenter of triangle ABC
noncomputable def incenter (A B C : EuclideanPlane) : EuclideanPlane := sorry

-- Define the condition that EF divides ABC into two figures with equal perimeters
def equal_perimeters (A B C E F : EuclideanPlane) : Prop := sorry

-- Define the condition that EF divides ABC into two figures with equal areas
def equal_areas (A B C E F : EuclideanPlane) : Prop := sorry

-- Define the condition that a point lies on a segment
def lies_on_segment (P E F : EuclideanPlane) : Prop := sorry

-- Theorem statement
theorem incenter_on_segment 
  (h_perimeter : equal_perimeters A B C E F)
  (h_area : equal_areas A B C E F) :
  lies_on_segment (incenter A B C) E F := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_on_segment_l657_65745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_eq_cos_implies_cos_2alpha_zero_but_not_conversely_l657_65746

-- Define the theorem
theorem sin_eq_cos_implies_cos_2alpha_zero_but_not_conversely :
  (∃ α : ℝ, Real.sin α = Real.cos α ∧ Real.cos (2 * α) = 0) ∧
  (∃ β : ℝ, Real.cos (2 * β) = 0 ∧ Real.sin β ≠ Real.cos β) := by
  sorry

-- Note: The proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_eq_cos_implies_cos_2alpha_zero_but_not_conversely_l657_65746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_slope_intercept_l657_65768

/-- Given a line equation, prove its slope and y-intercept -/
theorem line_equation_slope_intercept :
  ∃ (m b : ℝ), m = 2 ∧ b = -10 ∧
  ∀ x y : ℝ, (2 * (x - 3) + (-1) * (y - (-4)) = 0) ↔ y = m * x + b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_slope_intercept_l657_65768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_medians_intersect_at_centroid_l657_65795

/-- A median of a triangle is a line segment from a vertex to the midpoint of the opposite side. -/
def Median {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (triangle : Set V) : Set (Set V) :=
  sorry

/-- The centroid of a triangle is the point where its three medians intersect. -/
def Centroid {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (triangle : Set V) : V :=
  sorry

/-- Theorem: The three medians of a triangle intersect at a single point (the centroid). -/
theorem medians_intersect_at_centroid {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (triangle : Set V) :
  ∃! c : V, ∀ m : Set V, m ∈ Median triangle → c ∈ m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_medians_intersect_at_centroid_l657_65795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_xy_l657_65711

theorem existence_of_xy (f g : ℝ → ℝ) : ∃ x y, x ∈ Set.Icc 0 1 ∧ y ∈ Set.Icc 0 1 ∧ |x * y - f x - g y| ≥ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_xy_l657_65711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_15_deg_l657_65723

open Real

-- Define the half-angle formula for cosine
axiom half_angle_formula (x : ℝ) : cos (x / 2) = sqrt ((1 + cos x) / 2)

-- Define the value of cos 30°
axiom cos_30 : cos (30 * π / 180) = sqrt 3 / 2

-- Theorem to prove
theorem cos_15_deg : cos (15 * π / 180) = (sqrt 6 + sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_15_deg_l657_65723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_and_tangent_circle_l657_65717

/-- Represents a parabola in the form y^2 = a - bx --/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Represents a circle in the form (x - h)^2 + (y - k)^2 = r^2 --/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Calculates the x-coordinate of the directrix for a parabola --/
def directrix_x (p : Parabola) : ℝ := sorry

/-- Constructs a circle with center at the vertex of the parabola and tangent to its directrix --/
def tangent_circle (p : Parabola) : Circle := sorry

theorem parabola_directrix_and_tangent_circle 
  (p : Parabola) 
  (h : p.a = 8 ∧ p.b = 4) : 
  directrix_x p = 3 ∧ 
  (let c := tangent_circle p
   c.h = 2 ∧ c.k = 0 ∧ c.r = 1) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_and_tangent_circle_l657_65717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_candy_consumption_l657_65741

/-- The number of weeks it takes for Lisa to eat all her candies -/
def weeks_to_eat_candies (initial_candies : ℕ) (candies_per_week : ℕ) : ℕ :=
  (initial_candies + candies_per_week - 1) / candies_per_week

/-- Theorem stating that it takes 4 weeks for Lisa to eat all her candies -/
theorem lisa_candy_consumption : 
  (let initial_candies : ℕ := 72
   let monday_wednesday : ℕ := 3 * 2
   let tuesday_thursday : ℕ := 2 * 2
   let friday_saturday : ℕ := 4 * 2
   let sunday : ℕ := 1
   let candies_per_week : ℕ := monday_wednesday + tuesday_thursday + friday_saturday + sunday
   weeks_to_eat_candies initial_candies candies_per_week) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_candy_consumption_l657_65741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_ratio_l657_65737

/-- Represents a frustum with upper and lower base areas and volumes of two parts divided by midsection -/
structure Frustum where
  upper_area : ℚ
  lower_area : ℚ
  upper_volume : ℚ
  lower_volume : ℚ

/-- Creates a frustum with given area ratio and calculates the volume ratio -/
def create_frustum (area_ratio : ℚ) : Frustum :=
  { upper_area := 1
  , lower_area := 1 / area_ratio
  , upper_volume := 7
  , lower_volume := 19 }

/-- Theorem stating that for a frustum with area ratio 1:9, the volume ratio is 7:19 -/
theorem frustum_volume_ratio :
  let f := create_frustum (1/9)
  f.upper_volume / f.lower_volume = 7/19 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_ratio_l657_65737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_in_interval_f_f_strictly_increasing_f_10_negative_f_11_positive_l657_65729

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^x + (x - 1)^3 - 2014

-- State the theorem
theorem unique_zero_in_interval :
  ∃! x, x ∈ Set.Ioo 10 11 ∧ f x = 0 := by
  sorry

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 2^x * Real.log 2 + 3 * (x - 1)^2

-- Theorem stating that f' is positive in the interval (10, 11)
theorem f'_positive_in_interval :
  ∀ x ∈ Set.Ioo 10 11, f' x > 0 := by
  sorry

-- Theorem stating that f is strictly increasing in the interval (10, 11)
theorem f_strictly_increasing :
  StrictMonoOn f (Set.Ioo 10 11) := by
  sorry

-- Theorem stating that f(10) < 0
theorem f_10_negative : f 10 < 0 := by
  sorry

-- Theorem stating that f(11) > 0
theorem f_11_positive : f 11 > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_in_interval_f_f_strictly_increasing_f_10_negative_f_11_positive_l657_65729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_in_interval_tangent_line_at_minus_one_correct_statements_l657_65720

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := -1/x + Real.log x / Real.log 10

-- State the theorem
theorem f_has_zero_in_interval : ∃ x : ℝ, x ∈ Set.Ioo 2 3 ∧ f x = 0 := by
  sorry

-- Define the function for the second statement
def g (x : ℝ) : ℝ := 4*x - x^3

-- State the theorem for the second statement
theorem tangent_line_at_minus_one : 
  (deriv g (-1)) = -1 ∧ g (-1) = -3 ∧ 
  (fun x ↦ x - 2) = fun x ↦ g (-1) + (deriv g (-1)) * (x - (-1)) := by
  sorry

-- The main theorem combining both correct statements
theorem correct_statements : 
  (∃ x : ℝ, x ∈ Set.Ioo 2 3 ∧ f x = 0) ∧
  ((deriv g (-1)) = -1 ∧ g (-1) = -3 ∧ 
   (fun x ↦ x - 2) = fun x ↦ g (-1) + (deriv g (-1)) * (x - (-1))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_in_interval_tangent_line_at_minus_one_correct_statements_l657_65720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l657_65755

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.sin x - Real.cos x

theorem min_translation_for_symmetry :
  ∃ (a : ℝ), a > 0 ∧ 
  (∀ (x : ℝ), f (x - a) = f (a - x)) ∧
  (∀ (b : ℝ), b > 0 → (∀ (x : ℝ), f (x - b) = f (b - x)) → b ≥ a) ∧
  a = Real.pi / 3 :=
by
  sorry

#check min_translation_for_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l657_65755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_expense_is_19572_l657_65783

-- Define the structure for soda orders
structure SodaOrder where
  cases_a : ℕ
  cases_b : ℕ
  bottles_per_case_a : ℕ
  bottles_per_case_b : ℕ
  price_per_bottle_a : ℚ
  price_per_bottle_b : ℚ

-- Define the monthly orders
def april_order : SodaOrder := {
  cases_a := 100,
  cases_b := 50,
  bottles_per_case_a := 24,
  bottles_per_case_b := 30,
  price_per_bottle_a := 3/2,
  price_per_bottle_b := 2
}

def may_order : SodaOrder := {
  cases_a := 80,
  cases_b := 40,
  bottles_per_case_a := 24,
  bottles_per_case_b := 30,
  price_per_bottle_a := 3/2,
  price_per_bottle_b := 2
}

def june_order : SodaOrder := {
  cases_a := 120,
  cases_b := 60,
  bottles_per_case_a := 24,
  bottles_per_case_b := 30,
  price_per_bottle_a := 3/2,
  price_per_bottle_b := 2
}

-- Define discount and tax rates
def april_discount_rate : ℚ := 15/100
def may_tax_rate : ℚ := 10/100
def june_discount_per_case : ℚ := 36/10

-- Define helper functions for calculations
def calculate_april_total (order : SodaOrder) : ℚ :=
  (1 - april_discount_rate) * (order.cases_a * order.bottles_per_case_a * order.price_per_bottle_a) +
  (order.cases_b * order.bottles_per_case_b * order.price_per_bottle_b)

def calculate_may_total (order : SodaOrder) : ℚ :=
  let subtotal := (order.cases_a * order.bottles_per_case_a * order.price_per_bottle_a) +
                  (order.cases_b * order.bottles_per_case_b * order.price_per_bottle_b)
  subtotal * (1 + may_tax_rate)

def calculate_june_total (order : SodaOrder) : ℚ :=
  let discount := order.cases_b * june_discount_per_case
  (order.cases_a * order.bottles_per_case_a * order.price_per_bottle_a - discount) +
  (order.cases_b * order.bottles_per_case_b * order.price_per_bottle_b)

-- Theorem statement
theorem total_expense_is_19572 :
  let april_total := calculate_april_total april_order
  let may_total := calculate_may_total may_order
  let june_total := calculate_june_total june_order
  let net_total := april_total + may_total + june_total
  net_total = 19572 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_expense_is_19572_l657_65783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_in_circle_l657_65718

/-- The area of a regular octagon inscribed in a circle with radius 3 units -/
noncomputable def regularOctagonArea : ℝ := 18 * Real.sqrt 2

/-- The radius of the circle -/
def circleRadius : ℝ := 3

/-- Theorem: The area of a regular octagon inscribed in a circle with radius 3 units is 18√2 square units -/
theorem octagon_area_in_circle (r : ℝ) (h : r = circleRadius) : 
  regularOctagonArea = 18 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_in_circle_l657_65718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_proof_l657_65777

/-- Calculate simple interest -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem interest_difference_proof (principal rate time1 time2 : ℝ) :
  principal = 640 ∧ rate = 15 ∧ time1 = 3.5 ∧ time2 = 5 →
  simpleInterest principal rate time2 - simpleInterest principal rate time1 = 144 := by
  sorry

#check interest_difference_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_proof_l657_65777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_min_area_l657_65738

/-- Triangle ABC with vertices B(-1, 0), C(1, 0), and A(x, y) where y ≠ 0 -/
structure Triangle where
  x : ℝ
  y : ℝ
  h : y ≠ 0

/-- Orthocenter of triangle ABC -/
noncomputable def orthocenter (t : Triangle) : ℝ × ℝ := (t.x, (1 - t.x^2) / t.y)

/-- Centroid of triangle ABC -/
noncomputable def centroid (t : Triangle) : ℝ × ℝ := (t.x / 3, t.y / 3)

/-- Midpoint of line segment HG -/
noncomputable def midpoint_HG (t : Triangle) : ℝ × ℝ :=
  let h := orthocenter t
  let g := centroid t
  ((h.1 + g.1) / 2, (h.2 + g.2) / 2)

/-- Predicate for the midpoint K lying on line BC -/
def K_on_BC (t : Triangle) : Prop :=
  (midpoint_HG t).2 = 0

/-- Equation of locus E -/
def locus_equation (t : Triangle) : Prop :=
  t.x^2 - t.y^2 / 3 = 1

/-- Theorem statement -/
theorem locus_and_min_area (t : Triangle) :
  K_on_BC t →
  (locus_equation t ∧
   ∃ (min_area : ℝ), min_area = 9 * Real.pi / 2 ∧
     ∀ (area : ℝ),
       (∃ (p q d : ℝ × ℝ),
         locus_equation ⟨p.1, p.2, by sorry⟩ ∧
         locus_equation ⟨q.1, q.2, by sorry⟩ ∧
         locus_equation ⟨d.1, d.2, by sorry⟩ ∧
         d.2 - 0 = d.1 - 1 ∧  -- slope of CD is 1
         area = Real.pi * (p.1^2 + p.2^2 + q.1^2 + q.2^2 + d.1^2 + d.2^2 + 1) / 4) →
       area ≥ min_area) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_min_area_l657_65738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_amount_after_five_minutes_l657_65776

/-- Represents the amount of salt in the vessel at time t -/
noncomputable def salt_amount (t : ℝ) : ℝ :=
  3 - 3 * Real.exp (-0.2 * t)

/-- The initial volume of the vessel in liters -/
noncomputable def initial_volume : ℝ := 10 * Real.pi

/-- The flow rate in liters per minute -/
noncomputable def flow_rate : ℝ := 2 * Real.pi

/-- The salt concentration in the inflow in kg/liter -/
def salt_concentration : ℝ := 0.3

/-- The time period in minutes -/
def time_period : ℝ := 5

/-- Theorem stating that the amount of salt after 5 minutes is approximately 1.9π -/
theorem salt_amount_after_five_minutes :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  |salt_amount time_period - 1.9 * Real.pi| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_amount_after_five_minutes_l657_65776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l657_65747

theorem equation_solution (s : ℝ) : 3 = (5 : ℝ)^(4*s + 2) ↔ s = (Real.log 3 / Real.log 5 - 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l657_65747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l657_65716

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem function_properties (ω φ : ℝ) 
  (h_ω_pos : ω > 0)
  (h_φ_bound : |φ| < π/2)
  (h_period : ∀ x, f ω φ (x + 4*π) = f ω φ x)
  (h_symmetry : ∀ x, f ω φ (4*π/3 - x) = f ω φ (4*π/3 + x)) :
  ω = 1/2 ∧ 
  φ = π/6 ∧
  (∀ x ∈ Set.Icc 0 (2*π/3), ∀ y ∈ Set.Icc 0 (2*π/3), x < y → f ω φ x < f ω φ y) ∧
  (∀ x ∈ Set.Icc (2*π/3) (4*π/3), ∀ y ∈ Set.Icc (2*π/3) (4*π/3), x < y → f ω φ x > f ω φ y) ∧
  (∀ x, f ω φ (-π/3 + x) = f ω φ (-π/3 - x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l657_65716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_properties_l657_65789

/-- A circle in the 2D plane --/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- A line in the 2D plane --/
structure Line where
  equation : ℝ → ℝ → Prop

/-- The circle C with equation x^2 + y^2 + x - 2y + 1 = 0 --/
def C : Circle :=
  { equation := fun x y ↦ x^2 + y^2 + x - 2*y + 1 = 0 }

/-- The line with equation x + 2y + 3 = 0 --/
def line1 : Line :=
  { equation := fun x y ↦ x + 2*y + 3 = 0 }

/-- The line l with equation 2x - y + 2 = 0 --/
def l : Line :=
  { equation := fun x y ↦ 2*x - y + 2 = 0 }

/-- Predicate to check if a line bisects a circle --/
def bisects (line : Line) (circle : Circle) : Prop :=
  sorry

/-- Predicate to check if two lines are perpendicular --/
def perpendicular (l1 l2 : Line) : Prop :=
  sorry

/-- Theorem stating that line l bisects circle C and is perpendicular to line1 --/
theorem line_l_properties : bisects l C ∧ perpendicular l line1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_properties_l657_65789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_perfect_square_to_315_l657_65775

theorem closest_perfect_square_to_315 : 
  ∀ n : ℤ, n ≠ 18 → n^2 ≠ 324 → |315 - 324| ≤ |315 - n^2| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_perfect_square_to_315_l657_65775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_production_quantity_l657_65705

-- Define the revenue function
noncomputable def revenue (t : ℝ) : ℝ := 5 * t - (1 / 200) * t^2

-- Define the profit function
noncomputable def profit (x : ℝ) : ℝ :=
  if x ≤ 500 then
    revenue x - 0.5 * x - 25
  else
    revenue 500 - 0.5 * x - 25

-- Theorem stating the production quantity that maximizes profit
theorem max_profit_production_quantity :
  ∃ (x : ℝ), x = 450 ∧ ∀ (y : ℝ), y > 0 → profit x ≥ profit y := by
  sorry

-- Additional lemma to show that 450 is indeed the maximizing quantity
lemma profit_max_at_450 :
  ∀ (y : ℝ), y > 0 → profit 450 ≥ profit y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_production_quantity_l657_65705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_network_exists_l657_65762

/-- A city network with n cities and n-1 roads. -/
structure CityNetwork (n : ℕ) where
  roads : Fin (n - 1) → Fin n × Fin n
  connected : ∀ (i j : Fin n), ∃ (path : List (Fin n)), path.head? = some i ∧ path.getLast? = some j

/-- The shortest distances between pairs of cities are 1, 2, 3, ..., n(n-1)/2 km. -/
def valid_distances (n : ℕ) (distances : Fin n → Fin n → ℕ) : Prop :=
  ∀ (k : Fin (n * (n - 1) / 2)), ∃ (i j : Fin n), distances i j = k.val + 1

/-- The theorem stating the condition for the existence of a valid city network. -/
theorem city_network_exists (n : ℕ) :
  (∃ (net : CityNetwork n) (distances : Fin n → Fin n → ℕ), valid_distances n distances) ↔
  (∃ (k : ℕ), n = k^2 ∨ n = k^2 + 1) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_network_exists_l657_65762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l657_65702

theorem trig_inequality : Real.tan (-1) < Real.sin (-1) ∧ Real.sin (-1) < Real.cos (-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l657_65702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_arrangements_l657_65714

/-- Represents a parking space unit, which can be either a single space or a double space -/
inductive ParkingUnit
| Single : ParkingUnit
| Double : ParkingUnit

/-- The number of parking units to be arranged -/
def numUnits : Nat := 4

/-- The number of distinct ways to arrange the parking units -/
def numArrangements : Nat := 24

/-- Theorem stating that the number of ways to arrange the parking units is equal to 24 -/
theorem parking_arrangements :
  (List.permutations (List.replicate 2 ParkingUnit.Single ++ List.replicate 2 ParkingUnit.Double)).length = numArrangements := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_arrangements_l657_65714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l657_65799

theorem inequality_solution_set : 
  {x : ℝ | (x + 1) / (2 - x) ≤ 0} = Set.Iic (-1) ∪ Set.Ioi 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l657_65799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_equality_l657_65766

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := ((x + 5) / 5) ^ (1/4)

-- State the theorem
theorem solution_equality (x : ℝ) : 
  f (3 * x) = 3 * f x ↔ x = -200/39 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_equality_l657_65766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l657_65743

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := 1 / (2 * x + b)

-- Define the inverse function f_inv
noncomputable def f_inv (x : ℝ) : ℝ := (1 - 2 * x) / (2 * x)

-- Theorem statement
theorem inverse_function_condition (b : ℝ) :
  (∀ x, f b (f_inv x) = x) → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l657_65743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oldest_child_age_l657_65753

theorem oldest_child_age (ages : Fin 4 → ℕ) 
  (avg_age : (ages 0 + ages 1 + ages 2 + ages 3) / 4 = 9)
  (younger_ages : Multiset.ofList [ages 0, ages 1, ages 2] = Multiset.ofList [6, 8, 10]) :
  ages 3 = 12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oldest_child_age_l657_65753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_ratio_l657_65740

theorem sphere_surface_area_ratio (a : ℝ) (h : a > 0) :
  (4 * Real.pi * (a * Real.sqrt 3 / 2)^2) / (4 * Real.pi * (a / 2)^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_ratio_l657_65740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pigpen_cost_optimal_pigpen_dimensions_l657_65732

/-- Represents the cost of building a rectangular pigpen -/
noncomputable def pigpen_cost (x : ℝ) : ℝ :=
  360 * x + 5760 / x + 1120

/-- Theorem stating the minimum cost of the pigpen -/
theorem min_pigpen_cost :
  ∃ (x : ℝ), x > 0 ∧ x * (12 / x) = 12 ∧
  ∀ (y : ℝ), y > 0 → y * (12 / y) = 12 → pigpen_cost x ≤ pigpen_cost y ∧
  pigpen_cost x = 4000 := by
  sorry

/-- Theorem stating the optimal dimensions of the pigpen -/
theorem optimal_pigpen_dimensions :
  ∃ (x : ℝ), x > 0 ∧ x * (12 / x) = 12 ∧
  ∀ (y : ℝ), y > 0 → y * (12 / y) = 12 → pigpen_cost x ≤ pigpen_cost y ∧
  x = 4 ∧ 12 / x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pigpen_cost_optimal_pigpen_dimensions_l657_65732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_telephone_number_theorem_l657_65744

/-- Represents a telephone number in the format ABC-DEFG-HIJK --/
structure TelephoneNumber where
  A : Nat
  B : Nat
  C : Nat
  D : Nat
  E : Nat
  F : Nat
  G : Nat
  H : Nat
  I : Nat
  J : Nat
  K : Nat

/-- Checks if all digits in the telephone number are distinct --/
def all_distinct (t : TelephoneNumber) : Prop :=
  t.A ≠ t.B ∧ t.A ≠ t.C ∧ t.A ≠ t.D ∧ t.A ≠ t.E ∧ t.A ≠ t.F ∧ t.A ≠ t.G ∧ t.A ≠ t.H ∧ t.A ≠ t.I ∧ t.A ≠ t.J ∧ t.A ≠ t.K ∧
  t.B ≠ t.C ∧ t.B ≠ t.D ∧ t.B ≠ t.E ∧ t.B ≠ t.F ∧ t.B ≠ t.G ∧ t.B ≠ t.H ∧ t.B ≠ t.I ∧ t.B ≠ t.J ∧ t.B ≠ t.K ∧
  t.C ≠ t.D ∧ t.C ≠ t.E ∧ t.C ≠ t.F ∧ t.C ≠ t.G ∧ t.C ≠ t.H ∧ t.C ≠ t.I ∧ t.C ≠ t.J ∧ t.C ≠ t.K ∧
  t.D ≠ t.E ∧ t.D ≠ t.F ∧ t.D ≠ t.G ∧ t.D ≠ t.H ∧ t.D ≠ t.I ∧ t.D ≠ t.J ∧ t.D ≠ t.K ∧
  t.E ≠ t.F ∧ t.E ≠ t.G ∧ t.E ≠ t.H ∧ t.E ≠ t.I ∧ t.E ≠ t.J ∧ t.E ≠ t.K ∧
  t.F ≠ t.G ∧ t.F ≠ t.H ∧ t.F ≠ t.I ∧ t.F ≠ t.J ∧ t.F ≠ t.K ∧
  t.G ≠ t.H ∧ t.G ≠ t.I ∧ t.G ≠ t.J ∧ t.G ≠ t.K ∧
  t.H ≠ t.I ∧ t.H ≠ t.J ∧ t.H ≠ t.K ∧
  t.I ≠ t.J ∧ t.I ≠ t.K ∧
  t.J ≠ t.K

/-- Checks if digits in each part are in decreasing order --/
def decreasing_order (t : TelephoneNumber) : Prop :=
  t.A > t.B ∧ t.B > t.C ∧
  t.D > t.E ∧ t.E > t.F ∧ t.F > t.G ∧
  t.H > t.I ∧ t.I > t.J ∧ t.J > t.K

/-- Checks if D, E, F, G are consecutive digits --/
def consecutive_DEFG (t : TelephoneNumber) : Prop :=
  t.E = t.D - 1 ∧ t.F = t.E - 1 ∧ t.G = t.F - 1

/-- Checks if H, I, J, K are consecutive digits starting from an odd number --/
def consecutive_HIJK_odd (t : TelephoneNumber) : Prop :=
  t.H % 2 = 1 ∧
  t.I = t.H - 2 ∧ t.J = t.I - 2 ∧ t.K = t.J - 2

/-- The main theorem: Given the conditions, A must be 9 --/
theorem telephone_number_theorem (t : TelephoneNumber) 
  (h_distinct : all_distinct t)
  (h_decreasing : decreasing_order t)
  (h_consecutive_DEFG : consecutive_DEFG t)
  (h_consecutive_HIJK : consecutive_HIJK_odd t)
  (h_sum : t.A + t.B + t.C = 16) :
  t.A = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_telephone_number_theorem_l657_65744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ratio_l657_65748

/-- A point with integer coordinates on the circle x^2 + y^2 = 25 -/
structure IntegerCirclePoint where
  x : ℤ
  y : ℤ
  h : x^2 + y^2 = 25

/-- Distance between two points -/
noncomputable def distance (p q : IntegerCirclePoint) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem: The maximum ratio of distances between two pairs of distinct points
    on the circle x^2 + y^2 = 25 with integer coordinates, where both distances
    are irrational, is 7 -/
theorem max_distance_ratio :
  ∃ (p q r s : IntegerCirclePoint),
    p ≠ q ∧ r ≠ s ∧
    ¬ (∃ m : ℚ, distance p q = ↑m) ∧
    ¬ (∃ n : ℚ, distance r s = ↑n) ∧
    ∀ (a b c d : IntegerCirclePoint),
      a ≠ b → c ≠ d →
      ¬ (∃ m : ℚ, distance a b = ↑m) →
      ¬ (∃ n : ℚ, distance c d = ↑n) →
      (distance a b) / (distance c d) ≤ 7 ∧
      (distance p q) / (distance r s) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ratio_l657_65748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_of_5_l657_65779

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sqrt x + 15 / Real.sqrt x
def g (x : ℝ) : ℝ := 3 * x^2 - 3 * x - 4

-- State the theorem
theorem f_of_g_of_5 : f (g 5) = (183/28) * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_of_5_l657_65779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a2_value_l657_65774

noncomputable def z : ℂ := (1/2 : ℝ) + (Complex.I * (Real.sqrt 3 / 2))

theorem a2_value (a₀ a₁ a₂ a₃ a₄ : ℂ) :
  (∀ x : ℂ, (x - z)^4 = a₀*x^4 + a₁*x^3 + a₂*x^2 + a₃*x + a₄) →
  a₂ = -3 + Complex.I * (3 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a2_value_l657_65774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kureishi_readers_fraction_kureishi_readers_fraction_proof_l657_65773

theorem kureishi_readers_fraction (total_workers : ℕ) 
  (saramago_fraction : ℚ) (both_readers : ℕ) 
  (kureishi_fraction : ℚ) : Prop :=
  -- Total number of workers
  (total_workers = 40)
  ∧ -- Fraction of workers who have read Saramago's book
  (saramago_fraction = 1/4)
  ∧ -- Number of workers who have read both books
  (both_readers = 2)
  ∧ -- Condition about workers who have read neither book
  ((total_workers : ℚ) * (1 - saramago_fraction) - 
    ((total_workers : ℚ) * saramago_fraction - both_readers) = 1)
  ∧ -- Theorem statement
  ((total_workers : ℚ) * kureishi_fraction + 
    (total_workers : ℚ) * saramago_fraction - both_readers = 
    total_workers - ((total_workers : ℚ) * (1 - saramago_fraction) - 
    ((total_workers : ℚ) * saramago_fraction - both_readers) - 1))
  ∧ (kureishi_fraction = 5/8)

theorem kureishi_readers_fraction_proof : 
  ∃ (total_workers : ℕ) (saramago_fraction kureishi_fraction : ℚ) (both_readers : ℕ),
  kureishi_readers_fraction total_workers saramago_fraction both_readers kureishi_fraction := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kureishi_readers_fraction_kureishi_readers_fraction_proof_l657_65773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_91_84_35_l657_65709

/-- Heron's formula for the area of a triangle -/
noncomputable def heronFormula (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  (s * (s - a) * (s - b) * (s - c)) ^ (1/2 : ℝ)

/-- The area of a triangle with sides 91, 84, and 35 is approximately 453.19 -/
theorem triangle_area_91_84_35 :
  ∃ (area : ℝ), abs (area - heronFormula 91 84 35) < 0.01 ∧ abs (area - 453.19) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_91_84_35_l657_65709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_inequality_l657_65750

-- State the theorem
theorem abs_inequality (x : ℝ) : |3 - x| < 4 ↔ -1 < x ∧ x < 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_inequality_l657_65750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_half_correct_l657_65786

def points_per_question : ℕ := 8
def first_half_correct : ℕ := 8
def final_score : ℕ := 80

theorem second_half_correct : 
  (final_score - first_half_correct * points_per_question) / points_per_question = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_half_correct_l657_65786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_payment_500_actual_payment_formula_actual_payment_two_purchases_l657_65712

/-- Calculates the discount for a given purchase amount -/
noncomputable def discount (x : ℝ) : ℝ :=
  if x < 100 then 0
  else if x < 300 then 0.1 * x
  else 0.1 * 300 + 0.2 * (x - 300)

/-- Calculates the actual payment for a given purchase amount -/
noncomputable def actualPayment (x : ℝ) : ℝ := x - discount x

/-- Theorem: The actual payment for a 500 yuan purchase is 430 yuan -/
theorem actual_payment_500 : actualPayment 500 = 430 := by sorry

/-- Theorem: The actual payment formula for x ≥ 300 -/
theorem actual_payment_formula (x : ℝ) (h : x ≥ 300) : 
  actualPayment x = 0.8 * x + 30 := by sorry

/-- Theorem: The actual payment for two purchases totaling 620 yuan -/
theorem actual_payment_two_purchases (a : ℝ) (h1 : 100 < a) (h2 : a < 300) :
  actualPayment a + actualPayment (620 - a) = 0.1 * a + 526 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_payment_500_actual_payment_formula_actual_payment_two_purchases_l657_65712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bounds_l657_65736

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 + 1

-- Define the line
def line (s : ℝ) : ℝ := s

-- Define the vertices of the triangle
def vertex1 : ℝ × ℝ := (0, 1)

noncomputable def vertex2 (s : ℝ) : ℝ × ℝ := (-Real.sqrt (s - 1), s)
noncomputable def vertex3 (s : ℝ) : ℝ × ℝ := (Real.sqrt (s - 1), s)

-- Define the area of the triangle
noncomputable def triangleArea (s : ℝ) : ℝ := (s - 1) * Real.sqrt (s - 1)

-- State the theorem
theorem triangle_area_bounds (s : ℝ) :
  (10 ≤ triangleArea s ∧ triangleArea s ≤ 50) →
  (5.64 ≤ s ∧ s ≤ 18.32) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bounds_l657_65736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_infinite_square_free_sum_set_l657_65781

/-- A set of positive integers satisfying the square-free sum condition -/
def SquareFreeSumSet (M : Set ℕ) : Prop :=
  (∀ a b, a ∈ M → b ∈ M → a < b → ∀ k > 1, ¬(k^2 ∣ (a + b))) ∧ Set.Infinite M

/-- The existence of an infinite set of positive integers with square-free sums -/
theorem exists_infinite_square_free_sum_set :
  ∃ M : Set ℕ, SquareFreeSumSet M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_infinite_square_free_sum_set_l657_65781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_length_MQ_l657_65756

-- Define the circle O
def circleO (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define point A
def point_A (a : ℝ) : ℝ × ℝ := (a, 0)

-- Define that A is outside the circle
def A_outside_circle (a : ℝ) : Prop := a > 1

-- Define points P and Q on the circle
def P_on_circle (P : ℝ × ℝ) : Prop := circleO P.1 P.2
def Q_on_circle (Q : ℝ × ℝ) : Prop := circleO Q.1 Q.2

-- Define OP perpendicular to OQ
def OP_perp_OQ (O P Q : ℝ × ℝ) : Prop := 
  (P.1 - O.1) * (Q.1 - O.1) + (P.2 - O.2) * (Q.2 - O.2) = 0

-- Define point M as symmetrical to P with respect to A
def point_M (P : ℝ × ℝ) (a : ℝ) : ℝ × ℝ := (2*a - P.1, -P.2)

-- Define the length of segment MQ
noncomputable def length_MQ (M Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((M.1 - Q.1)^2 + (M.2 - Q.2)^2)

theorem max_length_MQ (a : ℝ) (P Q : ℝ × ℝ) (O : ℝ × ℝ := (0, 0)) :
  A_outside_circle a →
  P_on_circle P →
  Q_on_circle Q →
  OP_perp_OQ O P Q →
  a > 0 →
  ∃ (max_length : ℝ), ∀ (P' Q' : ℝ × ℝ),
    P_on_circle P' →
    Q_on_circle Q' →
    OP_perp_OQ O P' Q' →
    length_MQ (point_M P' a) Q' ≤ max_length ∧
    max_length = Real.sqrt (4*a^2 + 4*Real.sqrt 2*a + 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_length_MQ_l657_65756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_theorem_l657_65790

-- Define the complex number z
variable (z : ℂ)

-- Define the conditions
def condition1 (z : ℂ) : Prop := (z + 3 + 4*Complex.I).im = 0
def condition2 (z : ℂ) : Prop := (z / (1 - 2*Complex.I)).im = 0

-- Define the second quadrant condition
def second_quadrant (w : ℂ) : Prop := w.re < 0 ∧ w.im > 0

-- State the theorem
theorem complex_number_theorem (h1 : condition1 z) (h2 : condition2 z) :
  z = 2 - 4*Complex.I ∧ 
  ∀ m : ℝ, second_quadrant ((2 - 4*Complex.I - m*Complex.I)^2) ↔ m < -6 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_theorem_l657_65790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_cube_volume_ratio_is_sqrt_2_over_162_l657_65788

/-- The ratio of the volume of a regular tetrahedron formed by joining the centroids of adjoining faces of a cube to the volume of the cube -/
noncomputable def tetrahedron_cube_volume_ratio : ℝ := Real.sqrt 2 / 162

/-- Theorem stating that the ratio of the volume of a regular tetrahedron formed by joining the centroids of adjoining faces of a cube to the volume of the cube is √2/162 -/
theorem tetrahedron_cube_volume_ratio_is_sqrt_2_over_162 :
  tetrahedron_cube_volume_ratio = Real.sqrt 2 / 162 := by
  rfl

-- We can't use #eval for noncomputable definitions
-- Instead, we can use the following to check the definition
#check tetrahedron_cube_volume_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_cube_volume_ratio_is_sqrt_2_over_162_l657_65788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_and_function_range_l657_65772

noncomputable def a (x : ℝ) : Fin 2 → ℝ := ![Real.cos x, Real.sin x]
noncomputable def b : Fin 2 → ℝ := ![3, -Real.sqrt 3]

theorem vector_parallel_and_function_range 
  (x : ℝ) 
  (h_x : x ∈ Set.Icc 0 Real.pi) :
  (∃ k : ℝ, (fun i => a x i + b i) = fun i => k * b i) → 
  x = 5 * Real.pi / 6 ∧ 
  Set.range (fun x => -2 * Real.sqrt 3 * Real.sin (x / 2 - 2 * Real.pi / 3)) = Set.Icc (-3) (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_and_function_range_l657_65772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barrel_leak_problem_l657_65761

/-- Calculates the remaining volume in a barrel after a percentage loss. -/
noncomputable def remainingVolume (initialVolume : ℝ) (percentageLost : ℝ) : ℝ :=
  initialVolume * (1 - percentageLost / 100)

/-- Theorem: A 220-liter barrel that loses 10% of its contents has 198 liters remaining. -/
theorem barrel_leak_problem :
  remainingVolume 220 10 = 198 := by
  -- Unfold the definition of remainingVolume
  unfold remainingVolume
  -- Simplify the arithmetic expression
  simp [mul_sub, mul_div_cancel']
  -- Check that the resulting expression equals 198
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_barrel_leak_problem_l657_65761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_sqrt_difference_smallest_n_is_626_l657_65703

theorem smallest_n_for_sqrt_difference (n : ℕ) : n ≥ 626 ↔ Real.sqrt (n : ℝ) - Real.sqrt ((n - 1) : ℝ) < 0.02 := by
  sorry

theorem smallest_n_is_626 : ∃ (n : ℕ), n = 626 ∧ (∀ m : ℕ, m < n → Real.sqrt (m : ℝ) - Real.sqrt ((m - 1) : ℝ) ≥ 0.02) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_sqrt_difference_smallest_n_is_626_l657_65703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_translation_l657_65730

open Set

variable {α : Type*} [LinearOrder α]

def HasRange (f : ℝ → α) (S : Set α) :=
  ∀ y ∈ S, ∃ x, f x = y

theorem range_translation (f : ℝ → α) (a : ℝ) (b c : α) (h : HasRange f (Icc b c)) :
  HasRange (fun x ↦ f (x + a)) (Icc b c) := by
  intro y hy
  obtain ⟨x, hx⟩ := h y hy
  exists x - a
  simp [hx]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_translation_l657_65730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l657_65757

noncomputable def f (x : ℝ) := 3^x - 1 / (Real.sqrt x + 1) - 6

theorem root_exists_in_interval :
  ∃ c ∈ Set.Ioo 1 2, f c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l657_65757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_diff_eq_l657_65797

open Real

-- Define the differential equation
def diff_eq (y : ℝ → ℝ) (x : ℝ) : Prop :=
  (deriv^[2] y) x + 4 * (deriv y) x + 3 * y x = (8 * x^2 + 84 * x) * exp x

-- Define the general solution
noncomputable def general_solution (C₁ C₂ : ℝ) (x : ℝ) : ℝ :=
  C₁ * exp (-3 * x) + C₂ * exp (-x) + (x^2 + 9 * x - 7) * exp x

-- Theorem statement
theorem general_solution_satisfies_diff_eq (C₁ C₂ : ℝ) :
  ∀ x, diff_eq (general_solution C₁ C₂) x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_diff_eq_l657_65797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_leq_neg_one_iff_in_interval_l657_65791

theorem tan_leq_neg_one_iff_in_interval (x : ℝ) :
  Real.tan x ≤ -1 ↔ ∃ k : ℤ, k * Real.pi - Real.pi / 2 < x ∧ x ≤ k * Real.pi - Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_leq_neg_one_iff_in_interval_l657_65791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_digits_count_l657_65759

theorem product_digits_count : 
  let a : Nat := 5987456123789012345
  let b : Nat := 67823456789
  let Q : Nat := a * b
  (Nat.digits 10 Q).length = 33 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_digits_count_l657_65759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_number_is_14_l657_65785

def mySequence : List ℕ := [12, 14, 15, 17, 111, 113, 117, 119, 123, 129, 131]

theorem second_number_is_14 (seq : List ℕ) (h1 : seq.head? = some 12) 
  (h2 : seq.drop 2 = [15, 17, 111, 113, 117, 119, 123, 129, 131]) : 
  seq.get? 1 = some 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_number_is_14_l657_65785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_f_to_g_l657_65784

/-- The original function -/
noncomputable def f (x : ℝ) : ℝ := -1 / x

/-- The transformed function -/
noncomputable def g (x : ℝ) : ℝ := (2 * x + 1) / (x + 1)

/-- Theorem stating that g is the result of translating f one unit left and two units up -/
theorem transform_f_to_g : 
  ∀ x : ℝ, g x = f (x + 1) + 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_f_to_g_l657_65784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l657_65742

/-- The line l in rectangular coordinates -/
def line_l (x y : ℝ) : Prop := x - y + 6 = 0

/-- The ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

/-- The distance from a point (x, y) to the line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (x - y + 6) / Real.sqrt 2

/-- Theorem: The minimum distance from any point on ellipse C to line l is √2/2 -/
theorem min_distance_ellipse_to_line :
  ∃ (d : ℝ), d = Real.sqrt 2 / 2 ∧
  (∀ x y : ℝ, ellipse_C x y → distance_to_line x y ≥ d) ∧
  (∃ x y : ℝ, ellipse_C x y ∧ distance_to_line x y = d) := by
  sorry

#check min_distance_ellipse_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l657_65742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_journey_speeds_l657_65734

/-- Represents a journey with distance in km and time in minutes -/
structure Journey where
  distance : ℚ
  time : ℚ

/-- Calculates the average speed in km/hr given a journey -/
def averageSpeed (j : Journey) : ℚ :=
  (j.distance / j.time) * 60

theorem mary_journey_speeds :
  let uphill : Journey := { distance := 3/2, time := 45 }
  let downhill : Journey := { distance := 3/2, time := 5 }
  let roundTrip : Journey := { distance := 3, time := 50 }
  averageSpeed uphill = 2 ∧
  averageSpeed downhill = 18 ∧
  averageSpeed roundTrip = 36/10 := by
  sorry

#check mary_journey_speeds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_journey_speeds_l657_65734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_problem_l657_65752

/-- The chord length cut by a line on a circle -/
noncomputable def chordLength (a b c : ℝ) (x₀ y₀ r : ℝ) : ℝ :=
  2 * Real.sqrt (r^2 - ((a * x₀ + b * y₀ + c)^2 / (a^2 + b^2)))

/-- The problem statement -/
theorem chord_length_problem :
  let circle : ℝ × ℝ → Prop := fun p => (p.1 - 2)^2 + (p.2 - 2)^2 = 25
  let line : ℝ → ℝ := fun x => 2 * x - 2
  chordLength 2 (-1) 2 2 2 5 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_problem_l657_65752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_implies_range_of_a_l657_65708

-- Define the functions f and g
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + a * x^2 + b * x + 1
noncomputable def g (x : ℝ) : ℝ := Real.exp x

-- State the theorem
theorem common_tangent_implies_range_of_a (a b : ℝ) :
  (∀ x : ℝ, x < 0 → g x > f a b x) ↔ 
  (b = 1 ∧ a ≤ 1/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_implies_range_of_a_l657_65708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_x_in_interval_l657_65727

-- Define the expression as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x - 8*x^2 + 16*x^3) / (9 - x^3)

-- State the theorem
theorem f_nonnegative_iff_x_in_interval :
  ∀ x : ℝ, f x ≥ 0 ↔ x ∈ Set.Ici 0 ∩ Set.Iio 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_x_in_interval_l657_65727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_e_zeros_of_f_l657_65710

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log (2 * x) - Real.exp (2 * x / Real.exp 1)

theorem tangent_line_at_e (x y : ℝ) :
  let a := Real.exp 2
  let f_e := f a (Real.exp 1)
  (Real.exp 1) * x + y - (Real.exp 2) * Real.log (2 * Real.exp 1) = 0 ↔
    y - f_e = (x - Real.exp 1) * (deriv (f a)) (Real.exp 1) := by sorry

theorem zeros_of_f (a : ℝ) :
  (0 ≤ a ∧ a < Real.exp 1 → ∀ x, x > 0 → f a x ≠ 0) ∧
  (a < 0 ∨ a = Real.exp 1 → ∃! x, x > 0 ∧ f a x = 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_e_zeros_of_f_l657_65710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_length_l657_65719

/-- The length of the diagonal of a rectangle with sides 30√2 cm and 30(√2 + 2) cm -/
noncomputable def diagonal_length : ℝ := Real.sqrt (7200 + 3600 * Real.sqrt 2)

/-- The shorter side of the rectangle -/
noncomputable def side_a : ℝ := 30 * Real.sqrt 2

/-- The longer side of the rectangle -/
noncomputable def side_b : ℝ := 30 * (Real.sqrt 2 + 2)

theorem rectangle_diagonal_length : 
  Real.sqrt (side_a ^ 2 + side_b ^ 2) = diagonal_length := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_length_l657_65719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_demographics_l657_65794

/-- Represents the number of employees in a company -/
def total_employees : ℕ := sorry

/-- Represents the number of male employees in the company -/
def male_employees : ℕ := sorry

/-- Represents the number of female employees in the company -/
def female_employees : ℕ := sorry

/-- Represents the number of male employees without a college degree -/
def male_no_degree : ℕ := sorry

/-- The percentage of employees who are women -/
def women_percentage : ℚ := sorry

/-- The percentage of men who have a college degree -/
def men_degree_percentage : ℚ := sorry

theorem company_demographics (h1 : women_percentage = 60 / 100)
    (h2 : men_degree_percentage = 75 / 100)
    (h3 : male_no_degree = 8)
    (h4 : female_employees + male_employees = total_employees)
    (h5 : female_employees = (women_percentage * ↑total_employees).floor)
    (h6 : male_no_degree = ((1 - men_degree_percentage) * ↑male_employees).floor) :
    female_employees = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_demographics_l657_65794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_angle_MF1F2_l657_65780

/-- Hyperbola with eccentricity √5 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1
  eccentricity : Real.sqrt 5 = Real.sqrt (1 + b^2 / a^2)

/-- Foci of the hyperbola -/
noncomputable def foci (h : Hyperbola) : ℝ × ℝ × ℝ × ℝ := 
  let c := Real.sqrt (h.a^2 + h.b^2)
  (-c, 0, c, 0)

/-- Point M on the hyperbola -/
noncomputable def point_M (h : Hyperbola) : ℝ × ℝ := 
  let c := Real.sqrt (h.a^2 + h.b^2)
  (c, h.b^2 / h.a)

/-- Angle between three points -/
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Theorem: tan∠MF1F2 = 2√5/5 -/
theorem tan_angle_MF1F2 (h : Hyperbola) : 
  let (f1x, f1y, f2x, f2y) := foci h
  let m := point_M h
  let f1 := (f1x, f1y)
  let f2 := (f2x, f2y)
  Real.tan (angle m f1 f2) = 2 * Real.sqrt 5 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_angle_MF1F2_l657_65780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_mean_variation_functions_l657_65778

/-- A function is a constant mean variation function if for any x₁, x₂ (x₁ ≠ x₂) in the domain,
    the equation (f(x₁) - f(x₂)) / (x₁ - x₂) = f'((x₁ + x₂) / 2) always holds. -/
def IsConstantMeanVariation (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) = deriv f ((x₁ + x₂) / 2)

/-- Function 1: f(x) = 2x + 3 -/
def f₁ (x : ℝ) : ℝ := 2 * x + 3

/-- Function 2: f(x) = x² - 2x + 3 -/
def f₂ (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- Function 3: f(x) = 1/x -/
noncomputable def f₃ (x : ℝ) : ℝ := 1 / x

/-- Function 4: f(x) = e^x -/
noncomputable def f₄ (x : ℝ) : ℝ := Real.exp x

/-- Function 5: f(x) = ln(x) -/
noncomputable def f₅ (x : ℝ) : ℝ := Real.log x

theorem constant_mean_variation_functions :
  IsConstantMeanVariation f₁ ∧ 
  IsConstantMeanVariation f₂ ∧ 
  ¬IsConstantMeanVariation f₃ ∧ 
  ¬IsConstantMeanVariation f₄ ∧ 
  ¬IsConstantMeanVariation f₅ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_mean_variation_functions_l657_65778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_of_centroids_l657_65726

-- Define a structure for a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a function to calculate the centroid of three points
noncomputable def centroid (p1 p2 p3 : Point2D) : Point2D :=
  { x := (p1.x + p2.x + p3.x) / 3,
    y := (p1.y + p2.y + p3.y) / 3 }

theorem centroid_of_centroids 
  (A B C : Point2D) 
  (P : Point2D) 
  (h_P : P = centroid A B C) 
  (A₁ : Point2D) 
  (h_A₁ : A₁ = centroid B C P) 
  (B₁ : Point2D) 
  (h_B₁ : B₁ = centroid C A P) 
  (C₁ : Point2D) 
  (h_C₁ : C₁ = centroid A B P) : 
  centroid A₁ B₁ C₁ = P := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_of_centroids_l657_65726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin1_cos2_in_fourth_quadrant_l657_65763

def point_in_fourth_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem sin1_cos2_in_fourth_quadrant :
  point_in_fourth_quadrant (Real.cos 2) (Real.sin 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin1_cos2_in_fourth_quadrant_l657_65763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_side_length_approx_l657_65754

/-- Represents a trapezoidal roof with given dimensions -/
structure TrapezoidalRoof where
  area : ℝ
  height : ℝ
  upper_side_difference : ℝ

/-- Calculates the length of the lower side of a trapezoidal roof -/
noncomputable def lower_side_length (roof : TrapezoidalRoof) : ℝ :=
  (roof.area / roof.height - roof.upper_side_difference / 2) / 2

/-- Theorem stating that for a trapezoidal roof with given dimensions, 
    the lower side length is approximately 17.65 cm -/
theorem lower_side_length_approx 
  (roof : TrapezoidalRoof) 
  (h1 : roof.area = 100.62)
  (h2 : roof.height = 5.2)
  (h3 : roof.upper_side_difference = 3.4) :
  ∃ ε > 0, |lower_side_length roof - 17.65| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_side_length_approx_l657_65754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_beta_value_l657_65701

theorem tan_beta_value (α β : ℝ) (h1 : π/2 < α ∧ α < π) 
  (h2 : Real.sin α = 3 / Real.sqrt 10) (h3 : Real.tan (α + β) = -2) : 
  Real.tan β = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_beta_value_l657_65701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_for_given_point_l657_65787

def angle_terminal_point (α : ℝ) : ℝ × ℝ := (1, -1)

theorem sin_value_for_given_point (α : ℝ) :
  angle_terminal_point α = (1, -1) → Real.sin α = -Real.sqrt 2 / 2 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_for_given_point_l657_65787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_concurrent_l657_65728

-- Define the basic types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the given points and conditions
variable (A B C D X Y Z P M N : Point)
variable (line_ABCD : Line)
variable (circle_AC circle_BD : Circle)
variable (line_XY line_CP line_BP : Line)

-- Define a membership relation for Point and Line
def Point.mem (p : Point) (l : Line) : Prop := sorry

-- Define a membership relation for Point and Circle
def Point.memCircle (p : Point) (c : Circle) : Prop := sorry

-- Introduce notation for membership
instance : Membership Point Line where
  mem := Point.mem

instance : Membership Point Circle where
  mem := Point.memCircle

-- Define the conditions
axiom distinct_points : A ≠ B ∧ B ≠ C ∧ C ≠ D
axiom points_on_line : 
  A ∈ line_ABCD ∧ B ∈ line_ABCD ∧ C ∈ line_ABCD ∧ D ∈ line_ABCD
axiom order_of_points : 
  (B.x - A.x) * (C.x - B.x) > 0 ∧ (C.x - B.x) * (D.x - C.x) > 0

axiom circles_intersect : X ∈ circle_AC ∧ X ∈ circle_BD ∧ Y ∈ circle_AC ∧ Y ∈ circle_BD
axiom XY_intersects_BC : Z ∈ line_XY ∧ Z ∈ line_ABCD
axiom P_on_XY : P ∈ line_XY ∧ P ≠ Z

axiom CP_intersects_circle_AC : C ∈ line_CP ∧ M ∈ line_CP ∧ M ∈ circle_AC
axiom BP_intersects_circle_BD : B ∈ line_BP ∧ N ∈ line_BP ∧ N ∈ circle_BD

-- Define the lines AM and DN
def line_AM : Line := sorry
def line_DN : Line := sorry

-- State the theorem to be proved
theorem lines_concurrent : 
  ∃ Q : Point, Q ∈ line_AM ∧ Q ∈ line_DN ∧ Q ∈ line_XY := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_concurrent_l657_65728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_next_l657_65765

open BigOperators

def S (k : ℕ) : ℚ :=
  ∑ i in Finset.range (k + 1), 1 / (k + i)

theorem S_next (k : ℕ) : S (k + 1) = S k + 1 / (2 * k + 1) - 1 / (2 * k + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_next_l657_65765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C2_and_m_values_l657_65704

-- Define the curve C1
noncomputable def C1 (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 2 * Real.sin θ)

-- Define the curve C2 derived from C1
noncomputable def C2 (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sin θ)

-- Define the line l in Cartesian form
def l (m : ℝ) (x y : ℝ) : Prop := Real.sqrt 3 * x + 2 * y + m = 0

-- Define the distance function from a point to line l
noncomputable def distance_to_l (m : ℝ) (x y : ℝ) : ℝ :=
  abs (Real.sqrt 3 * x + 2 * y + m) / Real.sqrt 7

-- Main theorem
theorem C2_and_m_values :
  (∀ θ, C2 θ = (2 * Real.cos θ, Real.sin θ)) ∧
  (∃ m : ℝ, m = 10 ∨ m = -10) ∧
  (∀ m : ℝ, (m = 10 ∨ m = -10) →
    (∀ θ, distance_to_l m (C2 θ).1 (C2 θ).2 ≤ 2 * Real.sqrt 7) ∧
    (∃ θ, distance_to_l m (C2 θ).1 (C2 θ).2 = 2 * Real.sqrt 7)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C2_and_m_values_l657_65704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_for_hash_equation_l657_65722

-- Define the # operation
noncomputable def hash (x y : ℝ) : ℝ := (x - y) / (x * y)

-- Theorem statement
theorem no_solutions_for_hash_equation :
  ¬ ∃ a : ℝ, hash a (hash a 2) = 1 := by
  -- The proof will be skipped using 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_for_hash_equation_l657_65722
