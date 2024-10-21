import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_two_equals_three_l1074_107416

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a^x else Real.log (x^2 + a^2) / Real.log a

theorem f_neg_two_equals_three (a : ℝ) (h : f a 2 = 4) : f a (-2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_two_equals_three_l1074_107416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_length_l1074_107404

-- Define the ellipse parameters
noncomputable def a : ℝ := 4
noncomputable def c : ℝ := 2 * Real.sqrt 3

-- Define the ellipse equation
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 4 = 1

-- Define the line equation
def is_on_line (x y : ℝ) : Prop :=
  y = x + 2

-- Theorem statement
theorem ellipse_chord_length :
  (c^2 = 12) →
  (a = 4) →
  (∀ x y, is_on_ellipse x y ↔ x^2 / 16 + y^2 / 4 = 1) →
  (∃ x₁ y₁ x₂ y₂, 
    is_on_ellipse x₁ y₁ ∧ is_on_line x₁ y₁ ∧
    is_on_ellipse x₂ y₂ ∧ is_on_line x₂ y₂ ∧
    x₁ ≠ x₂ ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 16 * Real.sqrt 2 / 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_length_l1074_107404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_sum_l1074_107417

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2 + x - 3

-- State the theorem
theorem zero_in_interval_sum (a b : ℤ) : 
  (∃ x : ℝ, a < x ∧ x < b ∧ f x = 0) →  -- f has a zero in (a, b)
  b = a + 1 →                           -- a and b are consecutive integers
  a + b = 5 :=                          -- conclusion: a + b = 5
by
  sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_sum_l1074_107417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triple_sum_of_squares_l1074_107407

noncomputable def r : ℝ := Real.sqrt ((Real.sqrt 53 / 2) + (3 / 2))

theorem unique_triple_sum_of_squares : 
  ∃! (a b c : ℕ+), 
    (r^100 = 2*r^98 + 14*r^96 + 11*r^94 - r^50 + (a:ℝ)*r^46 + (b:ℝ)*r^44 + (c:ℝ)*r^40) ∧
    (a^2 + b^2 + c^2 = 15339) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triple_sum_of_squares_l1074_107407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_y_over_x_l1074_107444

theorem max_y_over_x (x y : ℝ) (h1 : y ≥ 0) (h2 : x^2 + y^2 - 4*x + 1 = 0) : 
  ∃ (k : ℝ), k = Real.sqrt 3 ∧ ∀ (z : ℝ), z ≥ 0 ∧ x^2 + z^2 - 4*x + 1 = 0 → z/x ≤ k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_y_over_x_l1074_107444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_in_U_l1074_107483

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - x)}

-- State the theorem
theorem complement_of_M_in_U :
  (Set.compl M) = Set.Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_in_U_l1074_107483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l1074_107467

/-- The number of non-negative integer solutions to x + 2y = n -/
def r (n : ℕ) : ℚ :=
  (1/2 : ℚ) * (n + 1 : ℚ) + (1/4 : ℚ) * (1 + (-1 : ℤ)^n : ℚ)

/-- The set of non-negative integer solutions to x + 2y = n -/
def solutions (n : ℕ) : Set (ℕ × ℕ) :=
  {p : ℕ × ℕ | p.1 + 2 * p.2 = n}

theorem solution_count (n : ℕ) : Finset.card (Finset.filter (fun p => p.1 + 2 * p.2 = n) (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1)))) = Int.floor (r n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l1074_107467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_slope_l1074_107418

noncomputable section

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (m : ℝ) (x y : ℝ) : Prop :=
  y - focus.2 = m * (x - focus.1)

-- Define the intersection points of the line and the parabola
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | parabola p.1 p.2 ∧ line_through_focus m p.1 p.2}

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the area of a triangle given three points
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  abs ((p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) / 2)

theorem parabola_line_intersection_slope :
  ∀ (m : ℝ), 
  (∃ (A B : ℝ × ℝ), A ∈ intersection_points m ∧ B ∈ intersection_points m ∧ 
    triangle_area origin A B = (3 * Real.sqrt 2) / 2) →
  m = 2 * Real.sqrt 2 ∨ m = -2 * Real.sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_slope_l1074_107418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_of_five_events_l1074_107480

def number_of_arrangements (n : ℕ) (no_first : ℕ) (must_last : ℕ) : ℕ :=
  (n - 2) * Nat.factorial (n - 2)

theorem arrangements_of_five_events :
  number_of_arrangements 5 1 1 = 18 := by
  rfl

#eval number_of_arrangements 5 1 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_of_five_events_l1074_107480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_median_length_l1074_107413

/-- Triangle with perpendicular medians -/
structure TriangleWithPerpendicularMedians where
  -- Vertices of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Midpoints of the sides
  D : ℝ × ℝ  -- Midpoint of BC
  E : ℝ × ℝ  -- Midpoint of AC
  F : ℝ × ℝ  -- Midpoint of AB
  -- D is midpoint of BC
  hD : D = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  -- E is midpoint of AC
  hE : E = ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
  -- F is midpoint of AB
  hF : F = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  -- AD and BE are perpendicular
  hPerp : (A.1 - D.1) * (B.1 - E.1) + (A.2 - D.2) * (B.2 - E.2) = 0
  -- Length of AD is 18
  hAD : Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) = 18
  -- Length of BE is 13.5
  hBE : Real.sqrt ((B.1 - E.1)^2 + (B.2 - E.2)^2) = 13.5

/-- The length of the third median CF is 22.5 -/
theorem third_median_length (t : TriangleWithPerpendicularMedians) : 
  Real.sqrt ((t.C.1 - t.F.1)^2 + (t.C.2 - t.F.2)^2) = 22.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_median_length_l1074_107413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l1074_107493

/-- The series function defined for n ≥ 3 -/
noncomputable def seriesFunc (n : ℕ) : ℝ :=
  (n^4 + 4*n^3 + 10*n^2 + 10*n + 10) / (2^n * (n^4 + 9))

/-- The infinite series starting from n = 3 -/
noncomputable def infiniteSeries : ℝ := ∑' n, seriesFunc (n + 3)

/-- Theorem stating that the infinite series converges to 7/36 -/
theorem series_sum : infiniteSeries = 7/36 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l1074_107493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_tan_alpha_l1074_107427

theorem sin_tan_alpha (α : ℝ) (P : ℝ × ℝ) :
  P.1 = 3/5 ∧ P.2 = -4/5 →
  Real.sin α * Real.tan α = 16/15 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_tan_alpha_l1074_107427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1074_107438

-- Define the function f with domain [-2, 4]
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-2) 4

-- Define the function g(x) = f(x) - f(-x)
def g (x : ℝ) : ℝ := f x - f (-x)

-- Theorem statement
theorem domain_of_g :
  {x : ℝ | x ∈ domain_f ∧ -x ∈ domain_f} = Set.Icc (-2) 2 := by
  sorry

#check domain_of_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1074_107438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_base_region_area_l1074_107499

/-- Represents a triangular pyramid with vertex P and base ABC. -/
structure TriangularPyramid where
  -- Base triangle side length
  baseSide : ℝ
  -- Height of the pyramid
  height : ℝ
  -- Radius of the inscribed sphere
  sphereRadius : ℝ

/-- The area of the region in the base of a triangular pyramid where points 
    are at a distance not greater than twice the inscribed sphere's radius 
    from the sphere's center. -/
noncomputable def baseRegionArea (p : TriangularPyramid) : ℝ := 
  1/4 + Real.pi/24

/-- Theorem stating the area of the specified region in the base of a 
    triangular pyramid with given dimensions. -/
theorem triangular_pyramid_base_region_area 
  (p : TriangularPyramid) 
  (h1 : p.baseSide = 1) 
  (h2 : p.height = Real.sqrt 2) : 
  baseRegionArea p = 1/4 + Real.pi/24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_base_region_area_l1074_107499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_ordering_l1074_107488

-- Define the quadratic function
noncomputable def f (x : ℝ) : ℝ := -(x - 1)^2 + 4

-- Define the points A, B, and C
noncomputable def A : ℝ × ℝ := (-2, f (-2))
noncomputable def B : ℝ × ℝ := (-1, f (-1))
noncomputable def C : ℝ × ℝ := (1/2, f (1/2))

-- Theorem statement
theorem point_ordering :
  A.2 < B.2 ∧ B.2 < C.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_ordering_l1074_107488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alper_cannot_guarantee_win_l1074_107476

/-- Represents the state of the game -/
inductive GameState
  | Winning
  | Losing
deriving Repr, DecidableEq

/-- Defines the game rules and determines the game state for a given number of stones -/
def gameState : ℕ → GameState
  | 0 => GameState.Losing
  | 1 => GameState.Losing
  | 2 => GameState.Winning
  | n + 3 => 
    if gameState (n + 2) = GameState.Losing ∨ gameState (n + 1) = GameState.Losing
    then GameState.Winning
    else GameState.Losing

/-- Theorem: Alper cannot guarantee a win for any k in {5, 6, 7, 8, 9} -/
theorem alper_cannot_guarantee_win :
  ∀ k : ℕ, k ∈ ({5, 6, 7, 8, 9} : Set ℕ) → gameState k = GameState.Losing :=
by
  sorry

#eval gameState 5
#eval gameState 6
#eval gameState 7
#eval gameState 8
#eval gameState 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alper_cannot_guarantee_win_l1074_107476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wedge_volume_cylindrical_log_l1074_107495

/-- The volume of a wedge cut from a cylindrical log --/
noncomputable def wedge_volume (d : ℝ) (angle : ℝ) : ℝ :=
  (d / 2)^2 * d * Real.pi / 2 * Real.cos angle

theorem wedge_volume_cylindrical_log :
  ∃ (m : ℕ), wedge_volume 20 (Real.pi / 6) = m * Real.pi ∧ m = 1732 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wedge_volume_cylindrical_log_l1074_107495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pairs_2023_l1074_107410

/-- Number of ones in the binary representation of n -/
def s₂ (n : ℕ) : ℕ := 
  n.digits 2 |>.sum

/-- Sum of s₂(k) for k from 0 to n-1 -/
def S (n : ℕ) : ℕ := 
  Finset.sum (Finset.range n) s₂

/-- Maximum number of pairs (x, y) in a set of size n where x - y is a power of e -/
def maxPairs (n : ℕ) : ℕ := S n

theorem max_pairs_2023 :
  maxPairs 2023 = 11043 := by sorry

#eval S 2023

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pairs_2023_l1074_107410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x2_expansion_l1074_107454

/-- The coefficient of x^2 in the expansion of (3x^2 - 1/(2∛x))^8 -/
noncomputable def coefficientX2 : ℝ :=
  (Nat.choose 8 6) * 3^2 * (1/2)^6

theorem coefficient_x2_expansion :
  coefficientX2 = 63/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x2_expansion_l1074_107454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l1074_107414

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y - 3 = Real.sqrt 3 * (x - 4)

-- Define the angle of inclination
noncomputable def angle_of_inclination (m : ℝ) : ℝ := Real.arctan m

-- Theorem statement
theorem line_properties :
  -- The point (4, 3) lies on the line
  line_equation 4 3 ∧
  -- The angle of inclination is π/3
  angle_of_inclination (Real.sqrt 3) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l1074_107414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mushroom_distribution_l1074_107422

/-- Represents a basket of mushrooms -/
structure MushroomBasket where
  total : ℕ
  orange : ℕ
  milk : ℕ
  total_sum : total = orange + milk

/-- Predicate to check if the basket satisfies the given conditions -/
def satisfies_conditions (basket : MushroomBasket) : Prop :=
  (∀ (subset : Finset ℕ), subset.card = 12 → (∃ m ∈ subset, m ≤ basket.orange)) ∧
  (∀ (subset : Finset ℕ), subset.card = 20 → (∃ m ∈ subset, basket.orange < m ∧ m ≤ basket.total))

theorem mushroom_distribution :
  ∀ (basket : MushroomBasket),
    basket.total = 30 →
    satisfies_conditions basket →
    basket.orange = 19 ∧ basket.milk = 11 := by
  sorry

#check mushroom_distribution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mushroom_distribution_l1074_107422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sales_amount_l1074_107466

-- Define the variables and constants
noncomputable def jemma_price : ℝ := 5
def jemma_frames_sold : ℕ := 400

-- Define the relationships
noncomputable def dorothy_price : ℝ := jemma_price / 2
def dorothy_frames_sold : ℕ := jemma_frames_sold / 2

-- Define the total amount made
noncomputable def total_amount : ℝ := jemma_price * (jemma_frames_sold : ℝ) + dorothy_price * (dorothy_frames_sold : ℝ)

-- Theorem to prove
theorem total_sales_amount : total_amount = 2500 := by
  -- Unfold definitions
  unfold total_amount
  unfold dorothy_price
  unfold dorothy_frames_sold
  -- Simplify the expression
  simp [jemma_price, jemma_frames_sold]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sales_amount_l1074_107466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_third_side_range_l1074_107429

-- Define Triangle as a structure
structure Triangle where
  sides : Finset ℝ
  valid : sides.card = 3
  inequality : ∀ a b c, sides = {a, b, c} → a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) ↔ 
  (∃ (t : Triangle), t.sides = {a, b, c}) :=
sorry

theorem third_side_range (x : ℝ) :
  (∃ (t : Triangle), t.sides = {5, 8, x}) →
  3 < x ∧ x < 13 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_third_side_range_l1074_107429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_robin_matches_l1074_107441

/-- Represents the number of teams in the tournament -/
def x : ℕ := sorry

/-- Represents the number of planned matches -/
def planned_matches : ℕ := 15

/-- The total number of matches in a single round-robin tournament -/
def total_matches (x : ℕ) : ℕ := x * (x - 1) / 2

/-- Theorem stating that the number of matches in a round-robin tournament with x teams is equal to the planned number of matches -/
theorem round_robin_matches : total_matches x = planned_matches := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_robin_matches_l1074_107441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_abs_times_g_is_even_l1074_107492

-- Define functions f and g
variable (f g : ℝ → ℝ)

-- Define the property of being an odd function
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the property of being an even function
def IsEven (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- State the theorem
theorem f_abs_times_g_is_even
  (h_f_odd : IsOdd f)
  (h_g_even : IsEven g) :
  IsEven (λ x => f (|x|) * g x) := by
  intro x
  have h1 : |-x| = |x| := by simp
  calc
    f (|-x|) * g (-x) = f (|x|) * g (-x) := by rw [h1]
    _ = f (|x|) * g x := by rw [h_g_even]

-- The proof is complete

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_abs_times_g_is_even_l1074_107492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_projection_preserves_parallel_and_equal_l1074_107405

/-- Represents a line segment in a 2D plane --/
structure LineSegment where
  start : ℝ × ℝ
  end_ : ℝ × ℝ

/-- Represents an oblique projection transformation --/
def ObliqueProjection := (ℝ × ℝ) → (ℝ × ℝ)

/-- Two line segments are parallel --/
def parallel (l1 l2 : LineSegment) : Prop := sorry

/-- Two line segments are equal in length --/
def equal_length (l1 l2 : LineSegment) : Prop := sorry

/-- Apply oblique projection to a line segment --/
def apply_projection (proj : ObliqueProjection) (l : LineSegment) : LineSegment := sorry

theorem oblique_projection_preserves_parallel_and_equal 
  (l1 l2 : LineSegment) (proj : ObliqueProjection) :
  parallel l1 l2 → equal_length l1 l2 → 
  parallel (apply_projection proj l1) (apply_projection proj l2) ∧ 
  equal_length (apply_projection proj l1) (apply_projection proj l2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_projection_preserves_parallel_and_equal_l1074_107405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_decreasing_f_l1074_107415

/-- A piecewise function f(x) defined as follows:
    f(x) = x^2 - a*x + 5 for x < 1
    f(x) = a/x for x ≥ 1
    where a is a real number -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then x^2 - a*x + 5 else a/x

/-- Theorem stating that if f(x) is monotonically decreasing on ℝ,
    then a is in the range [2, 3] -/
theorem a_range_for_decreasing_f (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) → 2 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_decreasing_f_l1074_107415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excircle_radii_bounds_l1074_107437

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def circumscribed (t : Triangle) (r : ℝ) : Prop :=
  sorry

def inscribed (t : Triangle) (r : ℝ) : Prop :=
  sorry

noncomputable def excircle_radius (t : Triangle) (vertex : ℝ × ℝ) : ℝ :=
  sorry

theorem excircle_radii_bounds (t : Triangle) 
  (h_circum : circumscribed t 1)
  (h_inscr : inscribed t 1)
  (ra rb rc : ℝ)
  (h_ra : ra = excircle_radius t t.A)
  (h_rb : rb = excircle_radius t t.B)
  (h_rc : rc = excircle_radius t t.C)
  (h_order : ra ≤ rb ∧ rb ≤ rc) :
  1 < ra ∧ ra ≤ 3 ∧ 2 < rb ∧ 3 ≤ rc := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_excircle_radii_bounds_l1074_107437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_range_of_a_l1074_107497

-- Define the function f(x) = x^2e^x
noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

-- Define the property of having an extremum in an interval
def has_extremum_in (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x ∈ Set.Ioo a b, ∀ y ∈ Set.Ioo a b, f y ≤ f x ∨ f y ≥ f x

-- Theorem statement
theorem extremum_range_of_a :
  ∀ a : ℝ, has_extremum_in f a (a + 1) ↔ a ∈ Set.Ioo (-3) (-2) ∪ Set.Ioo (-1) 0 := by
  sorry

#check extremum_range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_range_of_a_l1074_107497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_not_in_biology_count_l1074_107443

/-- The number of students not enrolled in a biology class -/
def students_not_in_biology (total : ℕ) (percent_in_biology : ℚ) : ℕ :=
  total - Int.toNat ((total : ℚ) * (percent_in_biology / 100)).floor

/-- Theorem stating the number of students not in biology classes -/
theorem students_not_in_biology_count :
  students_not_in_biology 880 (47.5 : ℚ) = 462 := by
  sorry

#eval students_not_in_biology 880 (47.5 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_not_in_biology_count_l1074_107443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_product_difference_trig_expression_value_l1074_107457

-- Part 1
theorem cos_product_difference (α β γ δ : Real) 
  (h1 : α = 25 * Real.pi / 180) 
  (h2 : β = 35 * Real.pi / 180) 
  (h3 : γ = 65 * Real.pi / 180) 
  (h4 : δ = 55 * Real.pi / 180) : 
  Real.cos α * Real.cos β - Real.cos γ * Real.cos δ = 1/2 := by
  sorry

-- Part 2
theorem trig_expression_value (θ : Real) 
  (h : Real.sin θ + 2 * Real.cos θ = 0) : 
  (Real.cos (2*θ) - Real.sin (2*θ)) / (1 + Real.cos θ^2) = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_product_difference_trig_expression_value_l1074_107457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_exp_plus_2x_l1074_107471

theorem definite_integral_exp_plus_2x : ∫ x in (0:ℝ)..(1:ℝ), (Real.exp x + 2 * x) = Real.exp 1 - 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_exp_plus_2x_l1074_107471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_sine_curve_l1074_107462

-- Define the function for the curve
noncomputable def f (x : ℝ) : ℝ := Real.sin x

-- Define the left and right boundaries
noncomputable def a : ℝ := -Real.pi/3
noncomputable def b : ℝ := Real.pi/2

-- State the theorem
theorem area_enclosed_by_sine_curve : 
  ∫ x in a..b, |f x| = 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_sine_curve_l1074_107462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_lower_bound_l1074_107432

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x * log x

-- State the theorem
theorem root_product_lower_bound 
  (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ < x₂) 
  (h2 : x₂ ≥ 3 * x₁) 
  (h3 : f a x₁ = x₁) 
  (h4 : f a x₂ = x₂) : 
  x₁ * x₂ ≥ 9 / exp 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_lower_bound_l1074_107432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_extrema_l1074_107465

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := (x - 2) / (x - 1)

-- Define the domain
def domain : Set ℝ := Set.Icc 2 6

-- State the theorem
theorem f_properties :
  (∀ x ∈ domain, ∀ y ∈ domain, x < y → f x < f y) ∧
  (∀ x ∈ domain, f x ≥ 0) ∧
  (∀ x ∈ domain, f x ≤ 4/5) ∧
  (∃ x ∈ domain, f x = 0) ∧
  (∃ x ∈ domain, f x = 4/5) :=
by sorry

-- Define the minimum and maximum values
def min_value : ℝ := 0
def max_value : ℝ := 4/5

-- State that these are indeed the minimum and maximum values
theorem f_extrema :
  (∀ x ∈ domain, f x ≥ min_value) ∧
  (∀ x ∈ domain, f x ≤ max_value) ∧
  (∃ x ∈ domain, f x = min_value) ∧
  (∃ x ∈ domain, f x = max_value) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_extrema_l1074_107465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l1074_107498

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Determines if two circles are intersecting -/
def are_intersecting (c1 c2 : Circle) : Prop :=
  let d := distance c1.center c2.center
  abs (c1.radius - c2.radius) < d ∧ d < c1.radius + c2.radius

/-- The first circle: x^2 + y^2 + 4x = 0 -/
def circle1 : Circle :=
  { center := (-2, 0), radius := 2 }

/-- The second circle: (x-2)^2 + (y-1)^2 = 9 -/
def circle2 : Circle :=
  { center := (2, 1), radius := 3 }

/-- Theorem stating that the two given circles are intersecting -/
theorem circles_intersect : are_intersecting circle1 circle2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l1074_107498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_B_in_triangle_l1074_107478

theorem max_tan_B_in_triangle (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi ∧ 
  (Real.sin B) / (Real.sin A) = Real.cos (A + B) →
  ∀ B' A' C', 0 < B' ∧ 0 < A' ∧ 0 < C' ∧ A' + B' + C' = Real.pi ∧ 
        (Real.sin B') / (Real.sin A') = Real.cos (A' + B') →
  Real.tan B ≤ Real.tan B' →
  Real.tan B ≤ Real.sqrt 2 / 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_B_in_triangle_l1074_107478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_mean_value_theorem_l1074_107452

theorem polynomial_mean_value_theorem (n : ℕ) (a : Fin (n + 1) → ℝ) :
  ∃ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧
    (λ x => (Finset.range (n + 1)).sum (λ i => a i * x^i)) (Real.sin θ) =
    (Finset.range (n + 1)).sum (λ i => a i / (↑i + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_mean_value_theorem_l1074_107452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1074_107435

def A : Set ℝ := {x : ℝ | |x - 1| < 2}
def B : Set ℝ := {x : ℝ | (x - 2) / (x + 4) < 0}

theorem intersection_of_A_and_B : A ∩ B = Set.Ioo (-1 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1074_107435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_speed_calculation_l1074_107487

-- Define the harmonic mean function
noncomputable def harmonic_mean (a b : ℝ) : ℝ := 2 * a * b / (a + b)

-- State the theorem
theorem upstream_speed_calculation 
  (v : ℝ) -- upstream speed
  (downstream_speed : ℝ) -- downstream speed
  (average_speed : ℝ) -- average speed for round-trip
  (h1 : downstream_speed = 7) -- given downstream speed
  (h2 : average_speed = 4.2) -- given average speed
  (h3 : harmonic_mean v downstream_speed = average_speed) -- definition of average speed
  : v = 3 := by
  sorry

#check upstream_speed_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_speed_calculation_l1074_107487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_i_inv_plus_real_part_one_plus_i_squared_l1074_107428

open Complex

theorem imaginary_part_i_inv_plus_real_part_one_plus_i_squared (a b : ℝ) : 
  (a = (I⁻¹).im) → (b = ((1 + I)^2).re) → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_i_inv_plus_real_part_one_plus_i_squared_l1074_107428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_odd_and_increasing_l1074_107442

-- Define the functions
noncomputable def f₁ (x : ℝ) : ℝ := x⁻¹
noncomputable def f₂ (x : ℝ) : ℝ := Real.sqrt x
noncomputable def f₃ (x : ℝ) : ℝ := Real.exp x
def f₄ (x : ℝ) : ℝ := x^3

-- Define what it means for a function to be odd
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define what it means for a function to be monotonically increasing on (0, +∞)
def IsMonoIncreasing (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x → x < y → f x < f y

theorem unique_odd_and_increasing :
  (IsOdd f₄ ∧ IsMonoIncreasing f₄) ∧
  (¬(IsOdd f₁ ∧ IsMonoIncreasing f₁)) ∧
  (¬(IsOdd f₂ ∧ IsMonoIncreasing f₂)) ∧
  (¬(IsOdd f₃ ∧ IsMonoIncreasing f₃)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_odd_and_increasing_l1074_107442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eliminate_y_in_system_l1074_107450

/-- Given a system of two linear equations in two variables,
    prove that multiplying the first equation by 2 and adding it to the second
    eliminates one variable (y in this case). -/
theorem eliminate_y_in_system (x y : ℝ) :
  (2 * x + 3 * y = 1) →
  (3 * x - 6 * y = 7) →
  ∃ k : ℝ, (2 * (2 * x + 3 * y) + (3 * x - 6 * y) = k) ∧ y ∉ {t | 2 * (2 * x + 3 * t) + (3 * x - 6 * t) = k} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eliminate_y_in_system_l1074_107450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_probability_is_two_thirds_l1074_107423

/-- The set of ball numbers in the box -/
def BallNumbers : Finset ℕ := {0, 1, 2, 3}

/-- The set of all possible pairs of balls that can be drawn -/
def AllPairs : Finset (ℕ × ℕ) := 
  BallNumbers.product BallNumbers |> Finset.filter (fun (a, b) => a < b)

/-- The set of winning pairs (pairs whose sum is 3, 4, or 5) -/
def WinningPairs : Finset (ℕ × ℕ) := 
  AllPairs.filter (fun (a, b) => a + b = 3 ∨ a + b = 4 ∨ a + b = 5)

/-- The probability of winning a prize -/
def WinningProbability : ℚ := (WinningPairs.card : ℚ) / (AllPairs.card : ℚ)

theorem winning_probability_is_two_thirds : WinningProbability = 2/3 := by
  -- Expand the definition of WinningProbability
  unfold WinningProbability
  -- Calculate the cardinalities
  have h1 : WinningPairs.card = 4 := by rfl
  have h2 : AllPairs.card = 6 := by rfl
  -- Simplify the fraction
  simp [h1, h2]
  -- The result follows
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_probability_is_two_thirds_l1074_107423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedral_probabilities_l1074_107484

-- Define the sample space
def Ω : Finset (Fin 4 × Fin 4) := Finset.univ

-- Define events A, B, and C
def A : Finset (Fin 4 × Fin 4) := Ω.filter (λ x => (x.1 + x.2) % 2 = 0)
def B : Finset (Fin 4 × Fin 4) := Ω.filter (λ x => x.1 % 2 = 0)
def C : Finset (Fin 4 × Fin 4) := Ω.filter (λ x => x.2 % 2 = 0)

-- Define probability measure
def P (S : Finset (Fin 4 × Fin 4)) : ℚ := (S.card : ℚ) / (Ω.card : ℚ)

-- Theorem statement
theorem tetrahedral_probabilities :
  P A = 1/2 ∧ P B = 1/2 ∧ P C = 1/2 ∧
  P (A ∩ B) = 1/4 ∧ P (B ∩ C) = 1/4 ∧ P (A ∩ C) = 1/4 ∧
  P (A ∩ B ∩ C) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedral_probabilities_l1074_107484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l1074_107472

/-- Given points P and C in a 3D coordinate system, prove that M is equidistant from P and C -/
theorem equidistant_point (P C M : ℝ × ℝ × ℝ) : 
  P = (0, 0, Real.sqrt 3) →
  C = (-1, 2, 0) →
  M = (0, 1/2, 0) →
  (M.2.1 = 0 ∧ M.2.2 = 0) →  -- M is on the y-axis
  (M.1 - P.1)^2 + (M.2.1 - P.2.1)^2 + (M.2.2 - P.2.2)^2 = 
  (M.1 - C.1)^2 + (M.2.1 - C.2.1)^2 + (M.2.2 - C.2.2)^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l1074_107472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_function_satisfies_equation_l1074_107426

theorem zero_function_satisfies_equation :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f (f x + x + y) = f (x + y) + y * f y) →
  ∀ x : ℝ, f x = 0 :=
by
  intro f h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_function_satisfies_equation_l1074_107426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_124_in_base7_has_three_consecutive_digits_l1074_107456

/-- Converts a decimal number to its base 7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else (n % 7) :: toBase7 (n / 7)

/-- Checks if a list of digits are consecutive -/
def areConsecutive (digits : List ℕ) : Bool :=
  match digits with
  | [] => true
  | [_] => true
  | d1 :: d2 :: rest => (d2 = d1 + 1) && areConsecutive (d2 :: rest)

theorem decimal_124_in_base7_has_three_consecutive_digits :
  let base7Digits := toBase7 124
  base7Digits.length = 3 ∧ areConsecutive base7Digits.reverse := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_124_in_base7_has_three_consecutive_digits_l1074_107456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_exponential_function_l1074_107460

theorem max_value_exponential_function :
  ∃ (x : ℝ), ∀ (y : ℝ), (3 : ℝ)^x - 2 * (9 : ℝ)^x ≥ (3 : ℝ)^y - 2 * (9 : ℝ)^y ∧ (3 : ℝ)^x - 2 * (9 : ℝ)^x = 1/8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_exponential_function_l1074_107460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1074_107446

/-- The line equation ax - y + 3 = 0 -/
def line_equation (a x y : ℝ) : Prop := a * x - y + 3 = 0

/-- The circle equation (x - 1)² + (y - 2)² = 4 -/
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

/-- The distance between two points on a plane -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- The theorem stating the value of a -/
theorem line_circle_intersection (a : ℝ) : 
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    line_equation a x₁ y₁ ∧ 
    line_equation a x₂ y₂ ∧
    circle_equation x₁ y₁ ∧ 
    circle_equation x₂ y₂ ∧
    distance x₁ y₁ x₂ y₂ = 2 * Real.sqrt 2) →
  a = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1074_107446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_trees_planted_l1074_107453

/-- Represents the number of trees planted by one gardener -/
def trees_per_gardener : ℕ := sorry

/-- The total number of gardeners -/
def total_gardeners : ℕ := 11

/-- The number of gardeners on Street A -/
def gardeners_a : ℕ := 2

/-- The number of gardeners on Street B -/
def gardeners_b : ℕ := 9

/-- The ratio of Street B's length to Street A's length -/
def street_ratio : ℕ := 5

/-- Theorem stating that the total number of trees planted is 44 -/
theorem total_trees_planted : 
  (gardeners_a * trees_per_gardener - 1) * street_ratio = 
  (gardeners_b * trees_per_gardener - 1) → 
  total_gardeners * trees_per_gardener = 44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_trees_planted_l1074_107453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_separating_lines_l1074_107489

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A type representing a line in a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

/-- Predicate to check if a point is on a line -/
def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Predicate to check if two points are on opposite sides of a line -/
def opposite_sides (p q : Point) (l : Line) : Prop :=
  (l.a * p.x + l.b * p.y + l.c) * (l.a * q.x + l.b * q.y + l.c) < 0

/-- The main theorem -/
theorem minimal_separating_lines (n : ℕ) (points : Fin n → Point) 
  (h1 : n ≥ 2)
  (h2 : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → ¬collinear (points i) (points j) (points k)) :
  ∃ (m : ℕ) (lines : Fin m → Line), 
    (∀ i : Fin n, ∀ j : Fin m, ¬on_line (points i) (lines j)) ∧ 
    (∀ i j : Fin n, i ≠ j → ∃ k : Fin m, opposite_sides (points i) (points j) (lines k)) ∧
    m = Nat.ceil (n / 2 : ℚ) ∧
    (∀ m' : ℕ, m' < m → ¬∃ (lines' : Fin m' → Line), 
      (∀ i : Fin n, ∀ j : Fin m', ¬on_line (points i) (lines' j)) ∧
      (∀ i j : Fin n, i ≠ j → ∃ k : Fin m', opposite_sides (points i) (points j) (lines' k))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_separating_lines_l1074_107489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_level_rise_rate_l1074_107412

/-- The rate of water level rise in a truncated pyramid-shaped pond -/
theorem water_level_rise_rate
  (h : ℝ) -- height of the pond
  (s₁ : ℝ) -- side length of the top square
  (s₂ : ℝ) -- side length of the bottom square
  (fill_rate : ℝ) -- water fill rate in cubic meters per hour
  (h_pos : h > 0)
  (s₁_pos : s₁ > 0)
  (s₂_pos : s₂ > 0)
  (s₁_gt_s₂ : s₁ > s₂)
  (fill_rate_pos : fill_rate > 0)
  (h_val : h = 6)
  (s₁_val : s₁ = 8)
  (s₂_val : s₂ = 2)
  (fill_rate_val : fill_rate = 19 / 3) :
  ∃ (rise_rate : ℝ), rise_rate = 19 / (6 * Real.sqrt 3) ∧
    rise_rate = (deriv (λ x => x * (x^2 + 6*x + 8) / 3)) (Real.sqrt 3) := by
  sorry

#check water_level_rise_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_level_rise_rate_l1074_107412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_score_game_12_l1074_107434

def scores_8_to_11 : List ℕ := [18, 22, 9, 29]

theorem min_score_game_12 
  (avg_11_greater_than_avg_7 : ∀ x : ℕ, x < (List.sum scores_8_to_11 + x) / 11) 
  (total_points_7 : ℕ) 
  (score_12 : ℕ) :
  (total_points_7 + List.sum scores_8_to_11 + score_12) / 12 > 20 → score_12 ≥ 30 :=
by
  sorry

#check min_score_game_12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_score_game_12_l1074_107434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_a_l1074_107486

open Real

/-- The function f(x) = ae^x - ax --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - a * x

/-- The function g(x) = (x-1)e^x + x --/
noncomputable def g (x : ℝ) : ℝ := (x - 1) * Real.exp x + x

/-- The derivative of f with respect to x --/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x - 1)

/-- The derivative of g with respect to x --/
noncomputable def g_deriv (x : ℝ) : ℝ := x * Real.exp x + 1

theorem smallest_integer_a :
  ∃ (a : ℝ), (∃ (t : ℝ), t > 0 ∧ f_deriv a t = g_deriv t) ∧
  (∀ (b : ℝ), (∃ (t : ℝ), t > 0 ∧ f_deriv b t = g_deriv t) → b ≥ a) ∧
  a = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_a_l1074_107486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_reachable_area_is_7pi_l1074_107400

/-- The area outside a regular octagonal doghouse that a dog can reach when tethered to a vertex -/
noncomputable def dogReachableArea (sideLength : ℝ) : ℝ :=
  let ropeLength := 3 * sideLength
  7 * Real.pi

/-- Theorem stating that the area the dog can reach is 7π square yards when the octagon's side length is 1 yard -/
theorem dog_reachable_area_is_7pi :
  dogReachableArea 1 = 7 * Real.pi :=
by
  -- Unfold the definition of dogReachableArea
  unfold dogReachableArea
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_reachable_area_is_7pi_l1074_107400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1074_107473

/-- Given points A, B, C, and D, where D is in the first quadrant and satisfies
    the vector equation AD = AB + λAC, prove that the range of λ is (-3, 5). -/
theorem lambda_range (A B C D : ℝ × ℝ) (lambda : ℝ) : 
  A = (4, 2) →
  B = (3, 5) →
  C = (5, 1) →
  D.1 > 0 →
  D.2 > 0 →
  D - A = (B - A) + lambda • (C - A) →
  -3 < lambda ∧ lambda < 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1074_107473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1074_107469

theorem trigonometric_equation_solution (t : ℝ) (p q r : ℕ) :
  (1 + Real.sin t ^ 2) * (1 + Real.cos t ^ 2) = 9 / 4 →
  (1 - Real.sin t ^ 2) * (1 - Real.cos t ^ 2) = p / q - Real.sqrt r →
  Nat.Coprime p q →
  p + q + r = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1074_107469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_magnitude_l1074_107477

def a : ℝ × ℝ × ℝ := (1, -3, 1)
def b : ℝ × ℝ × ℝ := (-1, 1, -3)

theorem vector_subtraction_magnitude : 
  Real.sqrt ((a.fst - b.fst)^2 + (a.snd.fst - b.snd.fst)^2 + (a.snd.snd - b.snd.snd)^2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_magnitude_l1074_107477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_isosceles_l1074_107403

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, if 
    a*cos(B) + b*cos(C) + c*cos(A) = b*cos(A) + c*cos(B) + a*cos(C),
    then the triangle is isosceles. -/
theorem triangle_isosceles (a b c : ℝ) (A B C : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : A > 0 ∧ B > 0 ∧ C > 0)
  (h_sum_angles : A + B + C = Real.pi)
  (h_eq : a * Real.cos B + b * Real.cos C + c * Real.cos A = 
          b * Real.cos A + c * Real.cos B + a * Real.cos C) :
  (a = b) ∨ (b = c) ∨ (a = c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_isosceles_l1074_107403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_approx_998_to_6th_power_l1074_107419

theorem approx_998_to_6th_power : |(0.998 : ℝ)^6 - 0.988| < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_approx_998_to_6th_power_l1074_107419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourteen_points_guarantees_win_l1074_107475

/-- Represents the possible placements in a race -/
inductive Placement
  | First
  | Second
  | Third
  | Other

/-- Calculates the points earned for a given placement -/
def points_for_placement (p : Placement) : ℕ :=
  match p with
  | Placement.First => 4
  | Placement.Second => 2
  | Placement.Third => 1
  | Placement.Other => 0

/-- Represents the results of a student in four races -/
def RaceResults := Fin 4 → Placement

/-- Calculates the total points for a set of race results -/
def total_points (results : RaceResults) : ℕ :=
  (Finset.range 4).sum (λ i => points_for_placement (results i))

/-- States that 14 is the smallest number of points that guarantees winning -/
theorem fourteen_points_guarantees_win :
  ∃ (winning_results : RaceResults),
    total_points winning_results = 14 ∧
    ∀ (other_results : RaceResults),
      other_results ≠ winning_results →
      total_points other_results < 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourteen_points_guarantees_win_l1074_107475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_1035_is_2_l1074_107420

def concatenated_digits (n : ℕ) : List ℕ :=
  List.join ((List.range (n + 1)).map (λ i => i.repr.toList.map (λ c => c.toNat - '0'.toNat)))

def nth_digit (n : ℕ) (digits : List ℕ) : Option ℕ :=
  digits.get? n

theorem digit_1035_is_2 :
  nth_digit 1034 (concatenated_digits 500) = some 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_1035_is_2_l1074_107420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_product_equals_two_l1074_107430

theorem cube_root_product_equals_two :
  let x := (5 * Real.sqrt 3 - 3 * Real.sqrt 7) ^ (1/3 : ℝ)
  let k := ((2/3) * (5 * Real.sqrt 3 + 3 * Real.sqrt 7)) ^ (1/3 : ℝ)
  k * x = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_product_equals_two_l1074_107430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_C₂_C₃_max_distance_AB_l1074_107490

noncomputable section

/-- Curve C₁ in parametric form -/
def C₁ (t : ℝ) (α : ℝ) : ℝ × ℝ :=
  (t * Real.cos α, t * Real.sin α)

/-- Curve C₂ in polar form -/
def C₂ (θ : ℝ) : ℝ :=
  4 * Real.sin θ

/-- Curve C₃ in polar form -/
def C₃ (θ : ℝ) : ℝ :=
  4 * Real.sqrt 3 * Real.cos θ

/-- Convert polar coordinates to Cartesian coordinates -/
def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem intersection_points_C₂_C₃ :
  ∃ θ₁ θ₂, 
    polar_to_cartesian (C₂ θ₁) θ₁ = (0, 0) ∧
    polar_to_cartesian (C₃ θ₂) θ₂ = (Real.sqrt 3, 3) :=
by sorry

theorem max_distance_AB :
  ∃ α : ℝ, 0 ≤ α ∧ α < π ∧
  ∃ t₁ t₂ θ₁ θ₂ : ℝ, t₁ ≠ 0 ∧ t₂ ≠ 0 ∧
    C₁ t₁ α = polar_to_cartesian (C₂ θ₁) θ₁ ∧
    C₁ t₂ α = polar_to_cartesian (C₃ θ₂) θ₂ ∧
    ∀ β : ℝ, 0 ≤ β ∧ β < π →
      ∃ s₁ s₂ φ₁ φ₂ : ℝ, s₁ ≠ 0 ∧ s₂ ≠ 0 →
        C₁ s₁ β = polar_to_cartesian (C₂ φ₁) φ₁ ∧
        C₁ s₂ β = polar_to_cartesian (C₃ φ₂) φ₂ →
          dist (C₁ t₁ α) (C₁ t₂ α) ≥ dist (C₁ s₁ β) (C₁ s₂ β) ∧
          dist (C₁ t₁ α) (C₁ t₂ α) = 8 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_C₂_C₃_max_distance_AB_l1074_107490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_ends_with_zero_l1074_107409

theorem product_ends_with_zero (n : ℕ) (h : n = 20) : 
  ∃ p : ℝ, (p = (1 - (9/10)^n) + (9/10)^n * (1 - (5/9)^n) * (1 - (8/9)^(n-1))) ∧ 
  (abs (p - 0.987) < 0.001) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_ends_with_zero_l1074_107409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_five_difference_l1074_107411

theorem power_five_difference (m n : ℕ) (h : m - n = 2) : (5 : ℝ)^m / (5 : ℝ)^n = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_five_difference_l1074_107411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_slope_range_l1074_107459

/-- Given a line segment PQ and a line l that intersects PQ at an extension, 
    prove the range of values for the slope of l. -/
theorem line_intersection_slope_range :
  let P : ℝ × ℝ := (-1, 1)
  let Q : ℝ × ℝ := (2, 2)
  let l (m : ℝ) := {(x, y) : ℝ × ℝ | x + m * y + m = 0}
  ∀ m : ℝ, (∃ t > 1, l m ∩ {p | ∃ s : ℝ, p = ((1-s)*P.1 + s*Q.1, (1-s)*P.2 + s*Q.2)} ≠ ∅) 
  → -3 < m ∧ m < -2/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_slope_range_l1074_107459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_inhabitants_can_reach_ball_l1074_107496

/-- Represents the side length of the square kingdom in kilometers. -/
def kingdomSideLength : ℝ := 2

/-- Represents the speed of inhabitants in km/h. -/
def inhabitantSpeed : ℝ := 3

/-- Represents the available time in hours. -/
def availableTime : ℝ := 7

/-- Represents the maximum distance an inhabitant can travel in the available time. -/
def maxTravelDistance : ℝ := inhabitantSpeed * availableTime

/-- Represents the diagonal length of the square kingdom. -/
noncomputable def kingdomDiagonal : ℝ := kingdomSideLength * Real.sqrt 2

theorem all_inhabitants_can_reach_ball :
  maxTravelDistance > kingdomDiagonal := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_inhabitants_can_reach_ball_l1074_107496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_m_eq_5_range_of_m_for_inequality_l1074_107458

def f (m x : ℝ) := |x - m| + |x + 6|

theorem solution_set_when_m_eq_5 :
  {x : ℝ | f 5 x ≤ 12} = {x : ℝ | -13/2 ≤ x ∧ x ≤ 11/2} := by sorry

theorem range_of_m_for_inequality :
  {m : ℝ | ∀ x, f m x ≥ 7} = Set.Iic (-13) ∪ Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_m_eq_5_range_of_m_for_inequality_l1074_107458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_ADEC_l1074_107439

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the quadrilateral ADEC
structure Quadrilateral :=
  (A D E C : ℝ × ℝ)

-- Define the properties of the triangle and quadrilateral
def IsRightTriangle (t : Triangle) : Prop :=
  sorry

def IsMidpoint (D : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  sorry

def IsPerpendicular (DE AB : ℝ × ℝ → ℝ × ℝ) : Prop :=
  sorry

def LengthAB (t : Triangle) : ℝ := 24

def LengthAC (t : Triangle) : ℝ := 15

-- Define the area of the quadrilateral
noncomputable def AreaOfQuadrilateral (q : Quadrilateral) : ℝ :=
  sorry

-- State the theorem
theorem area_of_quadrilateral_ADEC (t : Triangle) (q : Quadrilateral) :
  IsRightTriangle t →
  IsMidpoint q.D t.A t.B →
  IsPerpendicular (λ p => q.E - q.D) (λ p => t.B - t.A) →
  LengthAB t = 24 →
  LengthAC t = 15 →
  abs (AreaOfQuadrilateral q - 82.9023) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_ADEC_l1074_107439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_circles_l1074_107440

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Checks if a point is on a given circle -/
def isOnCircle (p : Point) (c : Circle) : Prop :=
  distance p c.center = c.radius

/-- The statement to be proved -/
theorem min_distance_between_circles : 
  let c1 : Circle := { center := { x := 0, y := 0 }, radius := 1 }
  let c2 : Circle := { center := { x := 2, y := 0 }, radius := 3 }
  ∀ p q : Point, isOnCircle p c1 → isOnCircle q c2 → 
    distance p q ≥ 2 ∧ ∃ p' q' : Point, isOnCircle p' c1 ∧ isOnCircle q' c2 ∧ distance p' q' = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_circles_l1074_107440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_squared_plus_sqrt_one_minus_x_squared_l1074_107494

open Real MeasureTheory

theorem integral_x_squared_plus_sqrt_one_minus_x_squared :
  ∫ x in Set.Icc 0 1, (x^2 + Real.sqrt (1 - x^2)) = π/4 + 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_squared_plus_sqrt_one_minus_x_squared_l1074_107494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_l1074_107445

/-- Represents the number of large buses -/
def large_buses : ℕ := sorry

/-- Represents the number of medium-sized buses -/
def medium_buses : ℕ := sorry

/-- The total number of buses is 20 -/
axiom total_buses : large_buses + medium_buses = 20

/-- The number of medium-sized buses is less than the number of large buses -/
axiom bus_inequality : medium_buses < large_buses

/-- Cost function: y = 22x + 800, where y is the total cost in million yuan and x is the number of large buses -/
def cost_function (x : ℕ) : ℕ := 22 * x + 800

/-- The minimum cost is 1042 million yuan -/
theorem min_cost : ∃ (x : ℕ), x = large_buses ∧ cost_function x = 1042 := by
  sorry

#check min_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_l1074_107445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_speed_calculation_l1074_107479

/-- The speed of person A in kilometers per hour -/
noncomputable def speed_A : ℝ := 4

/-- The time B starts walking after A, in hours -/
noncomputable def time_delay : ℝ := 0.5

/-- The time it takes B to overtake A, in hours -/
noncomputable def time_to_overtake : ℝ := 1.8

/-- The speed of person B in kilometers per hour -/
noncomputable def speed_B : ℝ := 9.2 / time_to_overtake

theorem b_speed_calculation :
  ∃ ε > 0, |speed_B - 5.11| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_speed_calculation_l1074_107479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fiona_reach_food_probability_l1074_107464

/-- Represents a lily pad with its number -/
structure LilyPad where
  number : Nat
deriving Repr

/-- Represents the frog's position -/
structure FrogPosition where
  pad : LilyPad
deriving Repr

/-- Represents the game setup -/
structure GameSetup where
  pads : List LilyPad
  predators : List LilyPad
  food : LilyPad
  start : LilyPad
deriving Repr

/-- Represents a single hop or jump probability -/
def hop_probability : Rat := 1/2

/-- The probability of Fiona reaching the food pad without landing on predator pads -/
noncomputable def reach_food_probability (setup : GameSetup) : Rat :=
  sorry

/-- The main theorem to prove -/
theorem fiona_reach_food_probability :
  let setup : GameSetup := {
    pads := (List.range 16).map (λ n => ⟨n⟩),
    predators := [⟨4⟩, ⟨9⟩],
    food := ⟨13⟩,
    start := ⟨0⟩
  }
  reach_food_probability setup = 27/1024 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fiona_reach_food_probability_l1074_107464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_assignment_3x4x5_l1074_107421

/-- Represents a rectangular parallelepiped with dimensions l, w, h -/
structure Parallelepiped :=
  (length : ℕ)
  (width : ℕ)
  (height : ℕ)

/-- Represents an assignment of numbers to unit squares on the surface of a parallelepiped -/
def Assignment (p : Parallelepiped) := 
  (Fin p.length × Fin p.width → ℕ) × 
  (Fin p.width × Fin p.height → ℕ) × 
  (Fin p.length × Fin p.height → ℕ)

/-- Checks if the sum of numbers in each 1-unit wide grid ring equals the target sum -/
def ValidAssignment (p : Parallelepiped) (a : Assignment p) (target : ℕ) : Prop :=
  ∀ (i : Fin p.length) (j : Fin p.width) (k : Fin p.height),
    (a.1 (i, j) + a.2.1 (j, k) + a.2.2 (i, k)) = target

/-- The main theorem stating that there exists a valid assignment for a 3x4x5 parallelepiped -/
theorem exists_valid_assignment_3x4x5 :
  ∃ (a : Assignment (Parallelepiped.mk 3 4 5)), ValidAssignment (Parallelepiped.mk 3 4 5) a 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_assignment_3x4x5_l1074_107421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_legs_on_street_l1074_107481

def total_animals : ℕ := 300
def cat_percentage : ℚ := 45/100
def dog_percentage : ℚ := 25/100
def bird_percentage : ℚ := 10/100
def insect_percentage : ℚ := 15/100
def three_legged_dog_percentage : ℚ := 5/100
def cat_legs : ℕ := 4
def dog_legs : ℕ := 4
def bird_legs : ℕ := 2
def insect_legs : ℕ := 6
def three_legged_dog_legs : ℕ := 3

theorem total_legs_on_street : 
  (↑total_animals * cat_percentage).floor * cat_legs +
  (↑total_animals * (dog_percentage - three_legged_dog_percentage)).floor * dog_legs +
  (↑total_animals * three_legged_dog_percentage).floor * three_legged_dog_legs +
  (↑total_animals * bird_percentage).floor * bird_legs +
  (↑total_animals * insect_percentage).floor * insect_legs = 1155 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_legs_on_street_l1074_107481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_of_negative_three_l1074_107468

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 4 - Real.sqrt (x^2)
def g (x : ℝ) : ℝ := 7*x + 3*x^3

-- Theorem statement
theorem f_of_g_of_negative_three : f (g (-3)) = -56 := by
  -- We'll use sorry to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_of_negative_three_l1074_107468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1074_107425

/-- Given a triangle ABC with the following properties:
    - A, B, and C are internal angles
    - tan A and tan B are real roots of x^2 + √3mx - m + 1 = 0
    - AB = √6
    - AC = 2
    Prove that:
    1. C = π/3
    2. The area of triangle ABC is (√6 + 3√2) / 4 -/
theorem triangle_properties (A B C : ℝ) (m : ℝ) :
  -- Conditions
  (∃ (tanA tanB : ℝ), 
    tanA^2 + Real.sqrt 3 * m * tanA - m + 1 = 0 ∧
    tanB^2 + Real.sqrt 3 * m * tanB - m + 1 = 0 ∧
    tanA = Real.tan A ∧
    tanB = Real.tan B) →
  A + B + C = π →
  Real.sqrt 6 = 2 * Real.sin C / Real.sin B →
  -- Conclusions
  C = π / 3 ∧
  1/2 * Real.sqrt 6 * 2 * Real.sin A = (Real.sqrt 6 + 3 * Real.sqrt 2) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1074_107425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_fraction_l1074_107406

theorem factorial_fraction (N : ℕ) (hN : N > 0) : 
  (Nat.factorial N.pred * N) / Nat.factorial (N + 1) = 1 / (N + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_fraction_l1074_107406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1074_107449

/-- An arithmetic sequence with a non-zero common difference, where a₁ = 4 and a₁, a₃, a₆ form a geometric sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h1 : d ≠ 0
  h2 : ∀ n, a (n + 1) = a n + d
  h3 : a 1 = 4
  h4 : (a 3) ^ 2 = a 1 * a 6

/-- The sum of the first n terms of the arithmetic sequence -/
def S_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * seq.a 1 + (1 / 2) * (n : ℚ) * ((n : ℚ) - 1) * seq.d

/-- The main theorem stating that S_n equals (n^2 + 7n) / 2 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) (n : ℕ) :
  S_n seq n = ((n : ℚ) ^ 2 + 7 * (n : ℚ)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1074_107449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_odd_is_four_fifths_l1074_107408

def digits : Finset Nat := {2, 3, 5, 7, 9}

def is_odd (n : Nat) : Bool := n % 2 = 1

def probability_odd : ℚ :=
  (digits.filter (fun n => n % 2 = 1)).card / digits.card

theorem probability_odd_is_four_fifths : probability_odd = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_odd_is_four_fifths_l1074_107408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_proof_l1074_107451

noncomputable def line_l (t : ℝ) : ℝ × ℝ := 
  ((Real.sqrt 2 / 2) * t, (Real.sqrt 2 / 2) * t + 4 * Real.sqrt 2)

noncomputable def circle_C (θ : ℝ) : ℝ := 
  2 * Real.cos (θ + Real.pi / 4)

noncomputable def min_tangent_length : ℝ := 
  2 * Real.sqrt 6

theorem min_tangent_length_proof :
  ∀ (t θ : ℝ),
  let (x, y) := line_l t
  let ρ := circle_C θ
  ∃ (tangent_length : ℝ),
  tangent_length ≥ min_tangent_length ∧
  (∃ (point_on_line : ℝ × ℝ) (point_on_circle : ℝ × ℝ),
   point_on_line.1 = (Real.sqrt 2 / 2) * point_on_line.2 - 4 * Real.sqrt 2 ∧
   point_on_circle.1^2 + point_on_circle.2^2 = ρ^2 ∧
   tangent_length^2 = (point_on_line.1 - point_on_circle.1)^2 + (point_on_line.2 - point_on_circle.2)^2) :=
by sorry

#check min_tangent_length_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_proof_l1074_107451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_e_equals_two_l1074_107401

theorem floor_e_equals_two : ⌊Real.exp 1⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_e_equals_two_l1074_107401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_distance_is_one_l1074_107447

noncomputable def vertex_x (a b : ℝ) : ℝ := -b / (2 * a)

noncomputable def vertex_y (a b c : ℝ) : ℝ := c - b^2 / (4 * a)

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem vertex_distance_is_one :
  let x1 := vertex_x 2 (-8)
  let y1 := vertex_y 2 (-8) 18
  let x2 := vertex_x (-3) 6
  let y2 := vertex_y (-3) 6 7
  distance x1 y1 x2 y2 = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_distance_is_one_l1074_107447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_chord_length_l1074_107424

noncomputable section

/-- A circle with center at the origin and radius 1 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

/-- The point P outside the circle -/
def P : ℝ × ℝ := (1, Real.sqrt 3)

/-- A point on the circle -/
def PointOnCircle (p : ℝ × ℝ) : Prop := p ∈ Circle

/-- A line is tangent to the circle at a point -/
def IsTangent (p : ℝ × ℝ) (q : ℝ × ℝ) : Prop :=
  PointOnCircle q ∧ (p.1 - q.1) * q.1 + (p.2 - q.2) * q.2 = 0

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem tangent_chord_length :
  ∀ A B : ℝ × ℝ,
  PointOnCircle A →
  PointOnCircle B →
  IsTangent P A →
  IsTangent P B →
  distance A B = Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_chord_length_l1074_107424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parity_of_expression_l1074_107463

theorem parity_of_expression (a b c : ℕ) (ha : Odd a) (hb : Odd b) :
  Odd (3^a + (b+1)^2*c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parity_of_expression_l1074_107463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonals_perpendicular_and_bisect_l1074_107485

/-- A rhombus is a quadrilateral with four equal sides -/
structure Rhombus where
  sides : Fin 4 → ℝ
  all_sides_equal : ∀ i j : Fin 4, sides i = sides j

/-- A diagonal of a quadrilateral -/
structure Diagonal (R : Rhombus) where
  endpoints : Fin 2 → Fin 4
  distinct : endpoints 0 ≠ endpoints 1

/-- The midpoint of a diagonal -/
def diagonalMidpoint (R : Rhombus) (D : Diagonal R) : ℝ × ℝ := sorry

/-- Two diagonals are perpendicular if their dot product is zero -/
def perpendicular (R : Rhombus) (D1 D2 : Diagonal R) : Prop := sorry

/-- A point bisects a diagonal if it's the midpoint of that diagonal -/
def bisects (R : Rhombus) (P : ℝ × ℝ) (D : Diagonal R) : Prop := sorry

theorem rhombus_diagonals_perpendicular_and_bisect (R : Rhombus) :
  ∃ D1 D2 : Diagonal R, perpendicular R D1 D2 ∧
    ∃ P : ℝ × ℝ, bisects R P D1 ∧ bisects R P D2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonals_perpendicular_and_bisect_l1074_107485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1074_107436

/-- A line in the form ax + y + b = 0 -/
structure Line where
  a : ℝ
  b : ℝ

/-- A circle with center (0, 0) and radius 2 -/
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

/-- The point M -/
noncomputable def M : ℝ × ℝ := (Real.sqrt 3, -1)

/-- Theorem statement -/
theorem line_circle_intersection (l : Line) (A B : Circle) :
  (∃ (O : ℝ × ℝ), O = (0, 0)) →
  (∃ (vecOA vecOB vecOM : ℝ × ℝ),
    vecOA + vecOB = (2/3 : ℝ) • vecOM ∧
    vecOM = M) →
  Real.sqrt 3 * l.a * l.b = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1074_107436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_theorem_l1074_107431

-- Define the sequences
def a : ℕ → ℝ := sorry

def b : ℕ → ℝ := sorry

-- Define the sum function
def S : ℕ → ℝ := sorry

-- State the theorem
theorem geometric_sequence_theorem :
  (∀ n : ℕ, b (n + 1) / b n = b 2 / b 1) →  -- b_n is geometric
  (∀ n : ℕ, b n = 2^(a n - 1)) →           -- Relation between b_n and a_n
  a 1 = 2 →                                -- First term of a_n
  a 3 = 4 →                                -- Third term of a_n
  (∀ n : ℕ, a n = n + 1) ∧                 -- General formula for a_n
  (∀ n : ℕ, S n = 3 - (n + 3) / 2^n) :=    -- Sum formula for a_n / b_n
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_theorem_l1074_107431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equalities_l1074_107402

theorem complex_expression_equalities : 
  (1/2 : ℝ) = (2 + 4/5)^0 + 2^(-2 : ℝ) * (2 + 1/4)^((-1/2) : ℝ) - (8/27)^(1/3) ∧
  (23/12 : ℝ) = (25/16)^(1/2) + (27/8)^((-1/3) : ℝ) - 2*Real.pi^0 + 4^(Real.log 5 / Real.log 4) - Real.log (Real.exp 5) + Real.log 200 / Real.log 10 - Real.log 2 / Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equalities_l1074_107402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_motion_theorem_l1074_107482

/-- A particle moving in a plane with specific angular velocity properties -/
structure Particle where
  x : ℝ → ℝ
  y : ℝ → ℝ
  hAngularVelocity : ∀ t, ((x t - 1) * (deriv y t) - (y t) * (deriv x t)) / ((x t - 1)^2 + (y t)^2) = 
                          -(((x t + 1) * (deriv y t) - (y t) * (deriv x t)) / ((x t + 1)^2 + (y t)^2))

/-- The differential equation derived from the particle's motion -/
def differentialEquation (x y : ℝ → ℝ) : Prop :=
  ∀ t, (deriv y t) * (x t) * ((x t)^2 + (y t)^2 - 1) = (y t) * ((x t)^2 + (y t)^2 + 1)

/-- A rectangular hyperbola passing through (±1, 0) -/
def rectangularHyperbola (x y : ℝ → ℝ) : Prop :=
  ∀ t, (x t)^2 - (y t)^2 = 1

theorem particle_motion_theorem (p : Particle) :
  differentialEquation p.x p.y ∧ rectangularHyperbola p.x p.y := by
  sorry

#check particle_motion_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_motion_theorem_l1074_107482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1074_107470

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 - a * x^2) / (1 + x^2)

theorem problem_solution (a : ℝ) :
  (∀ x : ℝ, x ≠ 0 → f a x + f a (1/x) = -1) →
  (a = 2 ∧
   (∀ x : ℝ, f a (-x) = f a x) ∧
   (∀ x y : ℝ, 0 ≤ x ∧ x < y → f a y < f a x) ∧
   (∀ x y : ℝ, x < y ∧ y ≤ 0 → f a y > f a x) ∧
   (∀ x : ℝ, f a x + f a (1/(2*x-1)) + 1 < 0 ↔ 
     (1/3 < x ∧ x < 1/2) ∨ (1/2 < x ∧ x < 1))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1074_107470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt2_fraction_pairs_count_l1074_107433

theorem sqrt2_fraction_pairs_count : 
  let count := Finset.filter (fun p : ℕ × ℕ => 
    let (m, n) := p
    m ≤ 1000 ∧ n ≤ 1000 ∧ 
    (m : ℝ) / ((n : ℝ) + 1) < Real.sqrt 2 ∧ 
    Real.sqrt 2 < ((m : ℝ) + 1) / (n : ℝ)) (Finset.product (Finset.range 1001) (Finset.range 1001))
  count.card = 1706 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt2_fraction_pairs_count_l1074_107433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_probability_l1074_107474

-- Define the card groups
def group_A : Finset ℕ := {2, 4, 6}
def group_B : Finset ℕ := {3, 5}

-- Define a function to check if a product is a multiple of 3
def is_multiple_of_three (n : ℕ) : Bool :=
  n % 3 = 0

-- Define the winning condition for Player A
def player_A_wins (a : ℕ) (b : ℕ) : Bool :=
  is_multiple_of_three (a * b)

-- The main theorem
theorem game_probability : 
  (Finset.card (Finset.filter (fun p => player_A_wins p.1 p.2) (group_A.product group_B)) : ℚ) / 
  ((Finset.card group_A * Finset.card group_B) : ℚ) = 2/3 := by
  sorry

#eval Finset.card (Finset.filter (fun p => player_A_wins p.1 p.2) (group_A.product group_B))
#eval Finset.card group_A * Finset.card group_B

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_probability_l1074_107474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_l1074_107491

-- Define the point F
def F : ℝ × ℝ := (1, 0)

-- Define the trajectory of point P
def trajectory (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  Real.sqrt ((x - F.1)^2 + y^2) - abs x = 1

-- Define perpendicular lines through F
def perpendicular_lines (l₁ l₂ : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ l₁ = (λ x ↦ k * (x - F.1)) ∧ l₂ = (λ x ↦ -1/k * (x - F.1))

-- Define the intersection points
def intersection_points (l₁ l₂ : ℝ → ℝ) (A B D E : ℝ × ℝ) : Prop :=
  trajectory A ∧ trajectory B ∧ trajectory D ∧ trajectory E ∧
  A.2 = l₁ A.1 ∧ B.2 = l₁ B.1 ∧ D.2 = l₂ D.1 ∧ E.2 = l₂ E.1

-- Define the dot product of vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- The main theorem
theorem min_dot_product :
  ∀ (l₁ l₂ : ℝ → ℝ) (A B D E : ℝ × ℝ),
    perpendicular_lines l₁ l₂ →
    intersection_points l₁ l₂ A B D E →
    ∀ (P : ℝ × ℝ), trajectory P →
      dot_product (A.1 - D.1, A.2 - D.2) (E.1 - B.1, E.2 - B.2) ≥ 16 :=
by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_l1074_107491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_arrangements_l1074_107461

theorem banana_arrangements : 
  (let total_letters : ℕ := 6
   let a_count : ℕ := 3
   let n_count : ℕ := 2
   let b_count : ℕ := 1
   (Nat.factorial total_letters) / ((Nat.factorial a_count) * (Nat.factorial n_count) * (Nat.factorial b_count))) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_arrangements_l1074_107461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intercept_l1074_107455

/-- The circle equation -/
def circle_eq (a x y : ℝ) : Prop := (x - a)^2 + y^2 = 4

/-- The line equation -/
def line_eq (x y : ℝ) : Prop := x - y = 2

/-- The length of the chord -/
noncomputable def chordLength : ℝ := 2 * Real.sqrt 2

/-- Theorem stating the possible values of a -/
theorem chord_intercept (a : ℝ) : 
  (∃ x y : ℝ, circle_eq a x y ∧ line_eq x y) → 
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_eq a x₁ y₁ ∧ line_eq x₁ y₁ ∧
    circle_eq a x₂ y₂ ∧ line_eq x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = chordLength^2) →
  a = 0 ∨ a = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intercept_l1074_107455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_digits_20_sided_die_l1074_107448

/-- The expected number of digits when rolling a fair 20-sided die -/
theorem expected_digits_20_sided_die : ℝ := by
  -- Define the die
  let die_faces : Finset ℕ := Finset.range 20

  -- Define the probability of rolling each face
  let prob_each_face : ℝ := 1 / 20

  -- Define the function that returns the number of digits for a given face
  let num_digits (n : ℕ) : ℕ := if n < 10 then 1 else 2

  -- Define the expected value calculation
  let expected_value := (die_faces.sum (λ face => prob_each_face * (num_digits face : ℝ)))

  -- State and prove the theorem
  have : expected_value = 1.55 := by
    sorry -- Proof omitted

  -- Return the result
  exact expected_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_digits_20_sided_die_l1074_107448
