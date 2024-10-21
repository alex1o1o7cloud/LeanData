import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_satisfying_polynomial_l913_91346

/-- A polynomial of degree 7 with coefficients in {0,1} -/
def MyPolynomial := Fin 7 → Fin 2

/-- The number of distinct integer roots of a polynomial -/
def distinct_integer_roots (p : MyPolynomial) : ℕ := sorry

/-- Whether zero is a root of the polynomial -/
def has_zero_root (p : MyPolynomial) : Prop := p 0 = 0

/-- The set of polynomials satisfying the conditions -/
def satisfying_polynomials : Set MyPolynomial :=
  {p | distinct_integer_roots p = 3 ∧ has_zero_root p}

theorem unique_satisfying_polynomial : 
  ∃! p : MyPolynomial, p ∈ satisfying_polynomials := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_satisfying_polynomial_l913_91346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_count_in_set_l913_91312

theorem difference_count_in_set : ∃ (S : Finset ℕ), 
  (∀ n, n ∈ S → 1 ≤ n ∧ n ≤ 20) ∧ 
  (∀ m, 1 ≤ m ∧ m ≤ 19 → ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ m = a - b) ∧
  (Finset.card S = 20) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_count_in_set_l913_91312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l913_91350

noncomputable section

-- Define the vectors a and b
def a (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, -1)
def b (x m : ℝ) : ℝ × ℝ := (Real.cos x, m)

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, v = (k * w.1, k * w.2)

-- Define the function f
def f (x m : ℝ) : ℝ := 2 * dot_product (a x + b x m) (b x m) - 2 * m^2 - 1

-- Part 1
theorem part_one (x : ℝ) :
  parallel (a x) (b x (Real.sqrt 3)) →
  (3 * Real.sin x - Real.cos x) / (Real.sin x + Real.cos x) = -3 := by sorry

-- Part 2
theorem part_two (x m : ℝ) :
  (∃ y ∈ Set.Icc 0 (Real.pi / 2), f y m = 0) →
  m ∈ Set.Icc (-1/2) 1 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l913_91350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l913_91391

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y = 2

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 2

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-1, 1)

-- Define the radius of the circle
noncomputable def circle_radius : ℝ := Real.sqrt 2

-- Theorem stating that the line is tangent to the circle
theorem line_tangent_to_circle :
  ∃ (p : ℝ × ℝ), line_l p.1 p.2 ∧ circle_C p.1 p.2 ∧
  ∀ (q : ℝ × ℝ), line_l q.1 q.2 → circle_C q.1 q.2 → q = p :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l913_91391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_properties_l913_91335

/-- Represents a point on a parabola --/
structure ParabolaPoint (p : ℝ) where
  y : ℝ
  x : ℝ
  eq : y^2 = 2*p*x

/-- Theorem about the minimum area and maximum angle of a triangle on a parabola --/
theorem parabola_triangle_properties (p : ℝ) (hp : p > 0) 
  (A B : ParabolaPoint p) (hA : A ≠ B) 
  (hO : A ≠ ⟨0, 0, by simp [ParabolaPoint.eq]⟩ ∧ B ≠ ⟨0, 0, by simp [ParabolaPoint.eq]⟩)
  (θ : ℝ) (hθ : θ ≠ π/2) (m : ℝ) 
  (h_area : abs (A.x * B.y - A.y * B.x) / 2 = m * Real.tan θ) :
  (∃ (m_min : ℝ), m ≥ m_min ∧ m_min = -p^2/2) ∧
  (∃ (θ_max : ℝ), abs (Real.tan θ) ≤ θ_max ∧ θ_max = 2*Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_properties_l913_91335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l913_91375

-- Define the function as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) := Real.sqrt (x^2 - 4*x)

-- State the theorem
theorem monotonic_increasing_interval :
  ∃ (a : ℝ), a = 4 ∧ 
  ∀ (x y : ℝ), x ≥ a → y > x → f y > f x :=
by
  -- Proof sketch
  -- We'll use a = 4 as our lower bound
  use 4
  constructor
  -- First part: a = 4
  · rfl
  -- Second part: ∀ (x y : ℝ), x ≥ 4 → y > x → f y > f x
  · intro x y hx hy
    -- The actual proof would go here
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l913_91375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_distance_from_line_l913_91336

/-- The distance from a point (x, y) to the line ax + by + c = 0 -/
noncomputable def distancePointToLine (x y a b c : ℝ) : ℝ :=
  |a * x + b * y + c| / Real.sqrt (a^2 + b^2)

/-- The theorem stating that for a point A(a, 6) at a distance of 4 from the line 3x - 4y - 4 = 0, 
    the value of a is either 16 or 8/3 -/
theorem point_distance_from_line (a : ℝ) :
  distancePointToLine a 6 3 (-4) (-4) = 4 → a = 16 ∨ a = 8/3 := by
  sorry

#check point_distance_from_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_distance_from_line_l913_91336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaolong_average_speed_l913_91380

/-- Xiaolong's journey from home to school --/
structure Journey where
  home_to_store_dist : ℚ
  home_to_store_time : ℚ
  store_to_playground_speed : ℚ
  store_to_playground_time : ℚ
  playground_to_school_dist : ℚ
  playground_to_school_speed : ℚ

/-- Calculate the average speed of Xiaolong's journey --/
def average_speed (j : Journey) : ℚ :=
  let store_to_playground_dist := j.store_to_playground_speed * j.store_to_playground_time
  let playground_to_school_time := j.playground_to_school_dist / j.playground_to_school_speed
  let total_distance := j.home_to_store_dist + store_to_playground_dist + j.playground_to_school_dist
  let total_time := j.home_to_store_time + j.store_to_playground_time + playground_to_school_time
  total_distance / total_time

/-- Theorem stating that Xiaolong's average speed is 72 meters per minute --/
theorem xiaolong_average_speed :
  let j : Journey := {
    home_to_store_dist := 500,
    home_to_store_time := 7,
    store_to_playground_speed := 80,
    store_to_playground_time := 8,
    playground_to_school_dist := 300,
    playground_to_school_speed := 60
  }
  average_speed j = 72 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaolong_average_speed_l913_91380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_100_mod_7_l913_91365

def T : ℕ → ℕ
  | 0 => 8  -- Add this case to handle Nat.zero
  | 1 => 8
  | n + 2 => 5^(T (n + 1))

theorem t_100_mod_7 : T 100 % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_100_mod_7_l913_91365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_roots_l913_91377

/-- An arithmetic sequence of non-zero integers -/
def ArithmeticSequence (a : ℤ → ℤ) : Prop :=
  ∃ d : ℤ, d ≠ 0 ∧ ∀ n : ℤ, a n ≠ 0 ∧ a (n + 1) = a n + d

/-- Quadratic equation E_i : a_i x^2 + a_{i+1} x + a_{i+2} = 0 -/
def QuadraticEquation (a : ℤ → ℤ) (i : ℤ) (x : ℝ) : Prop :=
  (a i : ℝ) * x^2 + (a (i + 1) : ℝ) * x + (a (i + 2) : ℝ) = 0

/-- Set of all roots of the quadratic equations -/
def RootSet (a : ℤ → ℤ) : Set ℝ :=
  {x : ℝ | ∃ i : ℤ, 1 ≤ i ∧ i ≤ 2020 ∧ QuadraticEquation a i x}

/-- Theorem: The maximum number of distinct real roots is 6 -/
theorem max_distinct_roots (a : ℤ → ℤ) (h : ArithmeticSequence a) :
  ∃ n : ℕ, n ≤ 6 ∧ Set.ncard (RootSet a) = n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_roots_l913_91377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_960_hours_l913_91332

/-- Calculates the total time for a round trip boat journey given the boat's speed in standing water,
    the stream's speed, and the distance to travel one way. -/
noncomputable def totalRoundTripTime (boatSpeed streamSpeed distance : ℝ) : ℝ :=
  let downstreamSpeed := boatSpeed + streamSpeed
  let upstreamSpeed := boatSpeed - streamSpeed
  (distance / downstreamSpeed) + (distance / upstreamSpeed)

/-- Theorem stating that for a boat with speed 16 kmph in standing water, a stream with speed 2 kmph,
    and a distance of 7560 km to travel (one way), the total round trip time is 960 hours. -/
theorem round_trip_time_960_hours :
  totalRoundTripTime 16 2 7560 = 960 := by
  -- Unfold the definition of totalRoundTripTime
  unfold totalRoundTripTime
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_960_hours_l913_91332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_fourth_times_B_l913_91351

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; -1, 4]
def B : Matrix (Fin 2) (Fin 1) ℝ := !![5; 3]

theorem A_fourth_times_B :
  A ^ 4 * B = !![145; 113] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_fourth_times_B_l913_91351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l913_91397

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2/8 - y^2/4 = 1

-- Define the axis of the parabola
def parabola_axis : ℝ → Prop := λ x ↦ x = -2

-- Define the asymptotes of the hyperbola
def hyperbola_asymptotes (x y : ℝ) : Prop := y = Real.sqrt 2/2 * x ∨ y = -Real.sqrt 2/2 * x

-- The theorem to prove
theorem triangle_area : 
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    parabola_axis x₁ ∧ parabola_axis x₂ ∧ parabola_axis x₃ ∧
    hyperbola_asymptotes x₁ y₁ ∧ hyperbola_asymptotes x₂ y₂ ∧
    hyperbola_asymptotes x₃ y₃ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ (x₂ ≠ x₃ ∨ y₂ ≠ y₃) ∧ (x₃ ≠ x₁ ∨ y₃ ≠ y₁) ∧
    (1/2 * abs (x₁*(y₂ - y₃) + x₂*(y₃ - y₁) + x₃*(y₁ - y₂)) = 2*Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l913_91397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_intersection_points_l913_91349

/-- The set A of integer points (x, y) satisfying ax + y = 1 -/
def A (a : ℤ) : Set (ℤ × ℤ) :=
  {p | a * p.1 + p.2 = 1}

/-- The set B of integer points (x, y) satisfying x + ay = 1 -/
def B (a : ℤ) : Set (ℤ × ℤ) :=
  {p | p.1 + a * p.2 = 1}

/-- The set C of integer points (x, y) on the unit circle -/
def C : Set (ℤ × ℤ) :=
  {p | p.1^2 + p.2^2 = 1}

/-- The theorem stating that when a = -1, (A ∪ B) ∩ C has exactly four elements -/
theorem four_intersection_points :
  ∃ (S : Finset (ℤ × ℤ)), (↑S : Set (ℤ × ℤ)) = (A (-1) ∪ B (-1)) ∩ C ∧ Finset.card S = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_intersection_points_l913_91349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drive_time_after_break_l913_91340

/-- Calculates the number of hours driven after a break given the total distance, 
    initial driving time, and constant speed. -/
noncomputable def hours_after_break (total_distance : ℝ) (initial_hours : ℝ) (speed : ℝ) : ℝ :=
  (total_distance - initial_hours * speed) / speed

/-- Theorem stating that given the specified conditions, 
    the time driven after the break is 9 hours. -/
theorem drive_time_after_break :
  let speed := (60 : ℝ)
  let initial_hours := (4 : ℝ)
  let total_distance := (780 : ℝ)
  hours_after_break total_distance initial_hours speed = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_drive_time_after_break_l913_91340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_approximation_l913_91393

/-- The principal amount given specific interest conditions -/
noncomputable def principal_amount (rate : ℝ) (time : ℝ) (difference : ℝ) : ℝ :=
  difference / (((1 + rate / 100) ^ time - 1) - (rate * time / 100))

/-- Theorem stating the principal amount under given conditions -/
theorem principal_amount_approximation :
  let rate : ℝ := 10
  let time : ℝ := 3
  let difference : ℝ := 31.000000000000455
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
    |principal_amount rate time difference - 1000| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_approximation_l913_91393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_interest_rate_l913_91330

theorem other_interest_rate (total_investment : ℝ) (investment_11_percent : ℝ) 
  (h1 : total_investment = 4725)
  (h2 : investment_11_percent = 1925)
  (h3 : investment_11_percent < total_investment) :
  let other_investment := total_investment - investment_11_percent
  let interest_11_percent := investment_11_percent * 0.11
  let other_rate := interest_11_percent / (2 * other_investment)
  ∃ ε > 0, |other_rate - 0.03781| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_interest_rate_l913_91330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_calculation_l913_91374

-- Define the custom operation as noncomputable
noncomputable def custom_op (a b : ℝ) : ℝ := (a^2 - b) / (a - b)

-- State the theorem
theorem custom_op_calculation :
  custom_op (custom_op 7 5) 2 = 24 := by
  -- Proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_calculation_l913_91374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_enclosing_sphere_radius_l913_91344

/-- The radius of the small spheres -/
def small_sphere_radius : ℝ := 2

/-- The number of small spheres -/
def num_spheres : ℕ := 10

/-- The side length of the cube -/
def cube_side_length : ℝ := 2 * small_sphere_radius

/-- The space diagonal of the cube -/
noncomputable def cube_space_diagonal : ℝ := cube_side_length * Real.sqrt 3

/-- The radius of the enclosing sphere -/
noncomputable def enclosing_sphere_radius : ℝ := (cube_space_diagonal + 2 * small_sphere_radius) / 2

/-- Theorem stating the radius of the smallest enclosing sphere -/
theorem smallest_enclosing_sphere_radius :
  enclosing_sphere_radius = 2 * (Real.sqrt 3 + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_enclosing_sphere_radius_l913_91344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_clears_obstacle_l913_91385

noncomputable section

-- Define the trajectory equations
def x (v t θ : ℝ) : ℝ := v * t * Real.cos θ

noncomputable def y (v t θ g : ℝ) : ℝ := v * t * Real.sin θ - (1/2) * g * t^2

-- Define the obstacle location
noncomputable def obstacle_x (v g : ℝ) : ℝ := v^2 / (4 * g)

noncomputable def obstacle_y (v g : ℝ) : ℝ := 3 * v^2 / (8 * g)

-- Theorem statement
theorem projectile_clears_obstacle (v g : ℝ) (hv : v > 0) (hg : g > 0) :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ Real.pi/2 ∧
  (∃ t : ℝ, x v t θ = obstacle_x v g ∧ y v t θ g = obstacle_y v g) ∧
  θ = Real.pi/12 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_clears_obstacle_l913_91385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_duchess_stole_cookbook_l913_91339

-- Define the suspects
inductive Suspect
| Duchess
| CheshireCat
| Cook

-- Define the thief
variable (thief : Suspect)

-- Define the statements made by each suspect
def statement (s : Suspect) : Prop :=
  match s with
  | Suspect.Duchess => Suspect.CheshireCat = thief
  | Suspect.CheshireCat => Suspect.CheshireCat = thief
  | Suspect.Cook => Suspect.Cook ≠ thief

-- Theorem to prove
theorem duchess_stole_cookbook :
  -- The thief lied about stealing the cookbook
  (∀ s : Suspect, s = thief → ¬statement thief s) →
  -- At least one of the non-thieves told the truth
  (∃ s : Suspect, s ≠ thief ∧ statement thief s) →
  -- The Duchess claimed the Cheshire Cat stole the cookbook
  statement thief Suspect.Duchess →
  -- The Cheshire Cat admitted to stealing the cookbook
  statement thief Suspect.CheshireCat →
  -- The cook asserted she didn't steal the cookbook
  statement thief Suspect.Cook →
  -- Conclusion: The Duchess stole the cookbook
  thief = Suspect.Duchess :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_duchess_stole_cookbook_l913_91339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l913_91383

/-- A line in the form kx - y + 1 + 2k = 0 where k is a real number -/
noncomputable def Line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | k * p.1 - p.2 + 1 + 2*k = 0}

/-- The area of the triangle formed by a line and the coordinate axes -/
noncomputable def TriangleArea (k : ℝ) : ℝ :=
  (1/2) * |2*k + 1| * |-(1/k) - 2|

theorem line_properties :
  (∀ k : ℝ, (-2, 1) ∈ Line k) ∧
  (∃ k : ℝ, TriangleArea k = 4 ∧ Line k = {p : ℝ × ℝ | p.1 - 2*p.2 + 4 = 0}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l913_91383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_decrease_g_l913_91328

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x)

theorem interval_of_decrease_g 
  (φ : ℝ) 
  (h1 : 0 < φ) 
  (h2 : φ < Real.pi / 2) 
  (g : ℝ → ℝ) 
  (h3 : ∀ x, g x = f (x + φ)) 
  (h4 : ∀ x, g x ≤ |g (Real.pi / 6)|) :
  ∀ k : ℤ, 
    let a := k * Real.pi + Real.pi / 12
    let b := k * Real.pi + 7 * Real.pi / 12
    ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → g y < g x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_decrease_g_l913_91328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_in_rectangular_prism_l913_91390

/-- A rectangular prism IJKLMNOP with base rectangle IJKLMN and vertex O above the base -/
structure RectangularPrism where
  /-- The length of each edge of the base rectangle -/
  base_edge : ℝ
  /-- The height from the base to the top vertex -/
  height : ℝ

/-- The volume of a pyramid with a triangular base -/
noncomputable def pyramid_volume (base_area : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * base_area * height

/-- Given a rectangular prism with base edge length 2 and height 2,
    the volume of the pyramid IJKOP is 4/3 -/
theorem pyramid_volume_in_rectangular_prism :
  let prism : RectangularPrism := { base_edge := 2, height := 2 }
  let base_area : ℝ := (1 / 2) * prism.base_edge * prism.base_edge
  pyramid_volume base_area prism.height = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_in_rectangular_prism_l913_91390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_functions_and_increasing_h_l913_91305

-- Define the functions
def f (x : ℝ) := x^2 + x
def g (x : ℝ) := -x^2 + x
def h (lambda : ℝ) (x : ℝ) := g x - lambda * f x + 1

-- State the theorem
theorem symmetric_functions_and_increasing_h :
  (∀ x, g (-x) = -g x) →  -- Symmetry condition for g
  (∀ x, f (-x) = f x) →   -- Symmetry condition for f
  (∀ x, g x = -x^2 + x) ∧ -- Correct form of g
  {lambda : ℝ | ∀ x ∈ (Set.Icc (-1) 1), 
    ∀ y ∈ (Set.Icc (-1) 1), x < y → h lambda x < h lambda y} = 
  Set.Icc (-3) (-1/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_functions_and_increasing_h_l913_91305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l913_91302

-- Define the necessary structures and functions
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

def isOnCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

def isOnAngleBisector (p : Point) (a b c : Point) : Prop :=
  sorry -- Define this properly later

def isOnSegment (p : Point) (a b : Point) : Prop :=
  sorry -- Define this properly later

def area (pts : List Point) : ℝ :=
  sorry -- Define this properly later

def BC_ratio_to_BD (b c d : Point) : ℝ :=
  sorry -- Define this properly later

structure CircleIntersection where
  O : Point
  Q : Point
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  circle_O : Circle
  circle_Q : Circle
  h1 : isOnCircle A circle_O
  h2 : isOnCircle B circle_O
  h3 : isOnCircle A circle_Q
  h4 : isOnCircle B circle_Q
  h5 : isOnAngleBisector C O A Q
  h6 : isOnAngleBisector D O A Q
  h7 : isOnSegment E A D
  h8 : isOnSegment E O Q
  h9 : area [O, A, E] = 18
  h10 : area [Q, A, E] = 42

theorem circle_intersection_theorem (ci : CircleIntersection) :
  area [ci.O, ci.A, ci.Q, ci.D] = 200 ∧ 
  BC_ratio_to_BD ci.B ci.C ci.D = 3 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l913_91302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l913_91315

-- Define the complex numbers
def z1 : ℂ := 1 + Complex.I
def z2 (m : ℝ) : ℂ := m - 2 * Complex.I

-- Theorem for part 1
theorem part1 (m : ℝ) : z1 + z2 m = 2 - Complex.I → m = 1 := by sorry

-- Theorem for part 2
theorem part2 (m : ℝ) : (z1 * z2 m).re > 0 ↔ m > -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l913_91315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_m_value_l913_91354

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line with slope 1
def line (x y m : ℝ) : Prop := y = x - m

-- Define the intersection points A and B
def intersectionPoints (xA yA xB yB m : ℝ) : Prop :=
  parabola xA yA ∧ parabola xB yB ∧ line xA yA m ∧ line xB yB m

-- Define the midpoint P
def midpointP (xP yP xA yA xB yB : ℝ) : Prop :=
  xP = (xA + xB) / 2 ∧ yP = (yA + yB) / 2

-- Theorem 1: Circle equation when m = 2
theorem circle_equation (xA yA xB yB : ℝ) :
  intersectionPoints xA yA xB yB 2 →
  ∃ x y, (x - 4)^2 + (y - 2)^2 = 24 :=
by
  sorry

-- Theorem 2: Value of m when the reciprocal condition is satisfied
theorem m_value (xA yA xB yB xP yP m : ℝ) :
  intersectionPoints xA yA xB yB m →
  midpointP xP yP xA yA xB yB →
  (1 / abs yA + 1 / abs yB = 1 / abs yP) →
  m = 2 + 2 * Real.sqrt 2 ∨ m = 2 - 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_m_value_l913_91354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_and_transformation_l913_91321

-- Define the function f
noncomputable def f (x : ℝ) (φ : ℝ) := Real.sin (2 * x + φ)

-- State the theorem
theorem function_symmetry_and_transformation 
  (φ : ℝ) 
  (h1 : -π < φ) 
  (h2 : φ < 0) 
  (h3 : ∀ x, f x φ = f (π/4 - x) φ) : -- Line of symmetry at x = π/8
  (φ = π/4) ∧ 
  (∀ x, 2 * Real.sin (2*x - π/6) = f (x - 5*π/24) φ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_and_transformation_l913_91321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_floor_sqrt5_and_neg_pi_l913_91303

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Theorem statement
theorem distance_between_floor_sqrt5_and_neg_pi : 
  (floor (Real.sqrt 5) : ℤ) - (floor (-Real.pi) : ℤ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_floor_sqrt5_and_neg_pi_l913_91303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_l913_91324

open Real

theorem tangent_sum (x y : ℝ) 
  (h1 : sin x / cos y + sin y / cos x = 1)
  (h2 : cos x / sin y + cos y / sin x = 5) :
  tan x / tan y + tan y / tan x = 310 / 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_l913_91324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l913_91327

noncomputable def Trajectory (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 2*p*x

noncomputable def FixedPoint (p : ℝ) : ℝ × ℝ :=
  (p/2, 0)

def TangentLine (p : ℝ) (x : ℝ) : Prop :=
  x = -p/2

theorem trajectory_equation (p : ℝ) (hp : p > 0) :
  ∀ x y : ℝ, 
  (∃ r : ℝ, r > 0 ∧ 
    ((x - (FixedPoint p).1)^2 + (y - (FixedPoint p).2)^2 = r^2) ∧
    (∃ xt : ℝ, TangentLine p xt ∧ (x - xt)^2 + y^2 = r^2)) →
  Trajectory p x y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l913_91327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l913_91360

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + 6 * Real.cos (Real.pi / 2 - x)

theorem f_max_value : ∀ x : ℝ, f x ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l913_91360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_product_l913_91304

theorem complex_fraction_product (a b : ℝ) :
  (Complex.I + 1) / (1 - Complex.I) = Complex.ofReal a + Complex.I * Complex.ofReal b → a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_product_l913_91304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_embedable_theorem_l913_91367

-- Define the concept of an embedable sequence
def embedable (a : List ℝ) (b c : ℝ) : Prop :=
  ∃ x : List ℝ, x.length = a.length + 1 ∧
    (∀ i, i < x.length → x[i]! ∈ Set.Icc b c) ∧
    (∀ i, i < a.length → |x[i+1]! - x[i]!| = a[i]!)

-- Define a normalized sequence
def normalized (a : List ℝ) : Prop :=
  ∀ x, x ∈ a → 0 ≤ x ∧ x ≤ 1

-- Main theorem
theorem embedable_theorem (n : ℕ) :
  (∀ a : List ℝ, a.length = 2*n+1 → normalized a → embedable a 0 (2 - 1/2^n)) ∧
  (∃ a : List ℝ, a.length = 4*n+3 ∧ normalized a ∧ ¬embedable a 0 (2 - 1/2^n)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_embedable_theorem_l913_91367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_distance_l913_91310

-- Define the points
def A : ℝ × ℝ := (-3, -3)
def B : ℝ × ℝ := (6, 3)
def C (n : ℝ) : ℝ × ℝ := (3, n)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the total distance function
noncomputable def total_distance (n : ℝ) : ℝ :=
  distance A (C n) + distance (C n) B

-- Theorem statement
theorem minimal_distance :
  ∃ (n : ℝ), ∀ (m : ℝ), total_distance n ≤ total_distance m ∧ n = -1/3 := by
  sorry

#check minimal_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_distance_l913_91310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_append_digits_l913_91311

/-- The least common multiple of numbers 1 to 9 -/
def lcm_1_to_9 : ℕ := 2520

/-- The original number to which digits are appended -/
def base_number : ℕ := 2014

/-- Function to check if a number is divisible by all natural numbers less than 10 -/
def divisible_by_all_less_than_10 (n : ℕ) : Prop := n % lcm_1_to_9 = 0

/-- Function to append digits to the base number -/
def append_digits (digits : ℕ) : ℕ := base_number * 10^(Nat.log 10 digits + 1) + digits

/-- Theorem stating that 4 is the smallest number of digits to append -/
theorem smallest_append_digits :
  (∃ (d : ℕ), divisible_by_all_less_than_10 (append_digits d) ∧ Nat.log 10 d + 1 = 4) ∧
  (∀ (d : ℕ), divisible_by_all_less_than_10 (append_digits d) → Nat.log 10 d + 1 ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_append_digits_l913_91311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_theorem_l913_91389

/-- Three points in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if three points are collinear -/
def collinear (A B C : Point3D) : Prop :=
  ∃ t : ℝ, (B.x - A.x, B.y - A.y, B.z - A.z) = t • (C.x - A.x, C.y - A.y, C.z - A.z)

/-- The main theorem -/
theorem collinear_points_theorem :
  let A : Point3D := ⟨1, -2, 11⟩
  let B : Point3D := ⟨4, 2, 3⟩
  let C : Point3D := ⟨x, y, 15⟩
  collinear A B C → x = -1/2 ∧ y = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_theorem_l913_91389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_implies_bc_negative_l913_91384

/-- A function f(x) with parameters a, b, and c. -/
noncomputable def f (a b c x : ℝ) : ℝ := a * Real.log x + b / x + c / (x^2)

/-- Theorem stating that if f(x) has both a maximum and a minimum, then bc < 0. -/
theorem f_max_min_implies_bc_negative (a b c : ℝ) (ha : a ≠ 0) 
  (h_max_min : ∃ (x_max x_min : ℝ), x_max ≠ x_min ∧ 
    (∀ x > 0, f a b c x ≤ f a b c x_max) ∧ 
    (∀ x > 0, f a b c x ≥ f a b c x_min)) :
  b * c < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_implies_bc_negative_l913_91384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_null_hypothesis_for_sports_gender_test_l913_91313

/-- Represents a categorical variable in a chi-square test -/
structure CategoricalVariable where
  name : String

/-- Represents a chi-square test of independence -/
structure ChiSquareTest where
  variable1 : CategoricalVariable
  variable2 : CategoricalVariable
  statistic : ℝ

/-- The null hypothesis for a chi-square test of independence -/
def NullHypothesis (test : ChiSquareTest) : Prop :=
  ¬ (∃ (relation : Prop), relation ∧ relation ↔ test.variable1.name ≠ test.variable2.name)

/-- Theorem stating the correct null hypothesis for the given problem -/
theorem correct_null_hypothesis_for_sports_gender_test 
  (sports_preference : CategoricalVariable) 
  (gender : CategoricalVariable) 
  (test : ChiSquareTest) 
  (h1 : sports_preference.name = "Liking to participate in sports activities")
  (h2 : gender.name = "gender")
  (h3 : test.variable1 = sports_preference)
  (h4 : test.variable2 = gender) :
  NullHypothesis test ↔ 
    ¬ (∃ (relation : Prop), relation ∧ relation ↔ "Liking to participate in sports activities" ≠ "gender") :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_null_hypothesis_for_sports_gender_test_l913_91313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_half_sector_l913_91300

noncomputable section

-- Define the radius of the original circle
def circle_radius : ℝ := 6

-- Define the cone's base radius (derived from the solution, but could be calculated)
def cone_base_radius : ℝ := 3

-- Define the cone's height (derived from the solution, but could be calculated)
noncomputable def cone_height : ℝ := 3 * Real.sqrt 3

-- State the theorem
theorem cone_volume_from_half_sector :
  let volume := (1/3) * Real.pi * cone_base_radius^2 * cone_height
  volume = 9 * Real.pi * Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_half_sector_l913_91300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l913_91378

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
variable (c₁ c₂ c₃ : Circle)
axiom r₁_gt_r₂ : c₁.radius > c₂.radius
axiom r₁_gt_r₃ : c₁.radius > c₃.radius

-- Define the external tangent points
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Define the conditions for A and B
axiom A_outside_c₃ : ∀ p : ℝ × ℝ, ‖p - c₃.center‖ ≤ c₃.radius → ‖A - p‖ > 0
axiom B_outside_c₂ : ∀ p : ℝ × ℝ, ‖p - c₂.center‖ ≤ c₂.radius → ‖B - p‖ > 0

-- Define the tangent lines
noncomputable def tangent_A_c₃ : Set (ℝ × ℝ) := sorry
noncomputable def tangent_B_c₂ : Set (ℝ × ℝ) := sorry

-- Define the quadrilateral formed by the tangents
noncomputable def quadrilateral : Set (ℝ × ℝ) := sorry

-- Define the inscribed circle
noncomputable def inscribed_circle : Circle := sorry

-- Theorem to prove
theorem inscribed_circle_radius (c₁ c₂ c₃ : Circle) :
  inscribed_circle.radius = (c₁.radius * c₂.radius * c₃.radius) /
    (c₁.radius * c₂.radius - c₂.radius * c₃.radius + c₁.radius * c₃.radius) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l913_91378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_l913_91352

-- Define the curves
noncomputable def C₁ (φ : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos φ, 2 * Real.sin φ)
noncomputable def C₂ (θ : ℝ) : ℝ := 4 * Real.sin θ

-- Define the intersection points
noncomputable def A (α : ℝ) : ℝ × ℝ := C₁ α
noncomputable def B (α : ℝ) : ℝ × ℝ := (C₂ α * Real.cos α, C₂ α * Real.sin α)

-- State the theorem
theorem intersection_angle (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi) :
  (‖A α - B α‖ = 4 * Real.sqrt 2) → α = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_l913_91352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_solution_l913_91363

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem floor_equation_solution (x : ℝ) : 
  floor (x * floor x) = 144 ↔ 12 ≤ x ∧ x < 12.08333 := by
  sorry

#check floor_equation_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_solution_l913_91363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_R_l913_91331

/-- Sum of arithmetic progression -/
noncomputable def sum_ap (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a₁ + (n - 1) * d)

/-- Arithmetic progression with first term b+2 and common difference d+1 -/
noncomputable def ap (b d : ℝ) (n : ℕ) : ℝ :=
  sum_ap (b + 2) (d + 1) n

/-- Definition of t₁, t₂, t₃ -/
noncomputable def t₁ (b d : ℝ) (m : ℕ) : ℝ := ap b d m
noncomputable def t₂ (b d : ℝ) (m : ℕ) : ℝ := ap b d (3 * m)
noncomputable def t₃ (b d : ℝ) (m : ℕ) : ℝ := ap b d (5 * m)

/-- Definition of R' -/
noncomputable def R' (b d : ℝ) (m : ℕ) : ℝ :=
  t₃ b d m - t₂ b d m - t₁ b d m

/-- Theorem: R' depends only on d and m -/
theorem R'_depends_on_d_m (b₁ b₂ d : ℝ) (m : ℕ) :
  R' b₁ d m = R' b₂ d m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_R_l913_91331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_observations_l913_91364

theorem number_of_observations (original_mean wrong_value correct_value new_mean : ℝ)
  (h_original_mean : original_mean = 36)
  (h_wrong_value : wrong_value = 21)
  (h_correct_value : correct_value = 48)
  (h_new_mean : new_mean = 36.54)
  : ∃ n : ℕ, (n : ℝ) * original_mean + (correct_value - wrong_value) = n * new_mean ∧ n = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_observations_l913_91364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wholesale_prices_solution_l913_91307

-- Define the stores
inductive Store : Type
| bakerPlus : Store
| star : Store

-- Define the structure for a day's prices
structure DayPrices where
  wholesale : ℝ
  bakerPlus : ℝ
  star : ℝ

-- Define the problem parameters
structure Problem where
  d : ℝ  -- Constant markup for Baker Plus
  k : ℝ  -- Constant factor for Star
  day1 : DayPrices
  day2 : DayPrices

-- Define the conditions of the problem
def validProblem (p : Problem) : Prop :=
  -- Baker Plus markup is positive
  p.d > 0 ∧
  -- Star factor is greater than 1
  p.k > 1 ∧
  -- Baker Plus price is wholesale plus markup
  p.day1.bakerPlus = p.day1.wholesale + p.d ∧
  p.day2.bakerPlus = p.day2.wholesale + p.d ∧
  -- Star price is wholesale times factor
  p.day1.star = p.k * p.day1.wholesale ∧
  p.day2.star = p.k * p.day2.wholesale ∧
  -- All prices are in the set {64, 64, 70, 72}
  ({p.day1.bakerPlus, p.day1.star, p.day2.bakerPlus, p.day2.star} : Set ℝ) = {64, 70, 72}

-- The theorem to prove
theorem wholesale_prices_solution (p : Problem) 
  (h : validProblem p) : 
  (p.day1.wholesale = 60 ∧ p.day2.wholesale = 40) ∨ 
  (p.day1.wholesale = 40 ∧ p.day2.wholesale = 60) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wholesale_prices_solution_l913_91307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_symmetry_center_main_theorem_l913_91394

noncomputable section

/-- The function f(x) defined as 2sin(ωx + π/6) -/
def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 6)

/-- The period of the function f(x) -/
def period (ω : ℝ) : ℝ := 2 * Real.pi / |ω|

/-- The theorem stating that if the distance between two adjacent common points
    with y = 2 is 2, then the minimum distance from the symmetry center of f(x)
    to the axis of symmetry is 1/2 -/
theorem min_distance_symmetry_center (ω : ℝ) :
  (period ω = 2) → (1/4 * period ω = 1/2) := by
  sorry

/-- The main theorem combining the previous results -/
theorem main_theorem (ω : ℝ) :
  (∃ x₁ x₂, x₁ < x₂ ∧ x₂ - x₁ = 2 ∧ f ω x₁ = 2 ∧ f ω x₂ = 2) →
  (∃ x, |x - (period ω / 4)| = 1/2 ∧
    ∀ y, |y - (period ω / 4)| ≥ 1/2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_symmetry_center_main_theorem_l913_91394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_correct_l913_91348

/-- Conversion from cylindrical coordinates to rectangular coordinates -/
noncomputable def cylindrical_to_rectangular (r θ z : Real) : Real × Real × Real :=
  (r * Real.cos θ, r * Real.sin θ, z)

/-- The given point in cylindrical coordinates -/
noncomputable def cylindrical_point : Real × Real × Real := (3, Real.pi / 4, -2)

/-- The expected point in rectangular coordinates -/
noncomputable def rectangular_point : Real × Real × Real := (3 * Real.sqrt 2 / 2, 3 * Real.sqrt 2 / 2, -2)

/-- Theorem stating that the conversion from cylindrical to rectangular coordinates is correct -/
theorem cylindrical_to_rectangular_correct :
  cylindrical_to_rectangular cylindrical_point.1 cylindrical_point.2.1 cylindrical_point.2.2 = rectangular_point := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_correct_l913_91348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olympic_javelin_throw_l913_91338

/-- Olympic javelin thrower problem -/
theorem olympic_javelin_throw (throw1 throw2 throw3 : ℝ) 
  (h1 : throw1 = 2 * throw2)
  (h2 : throw1 = (1/2) * throw3)
  (h3 : throw1 * 0.95 + throw2 * 0.92 + throw3 = 1050) :
  ∃ ε > 0, |throw1 - 305.66| < ε := by
  sorry

-- Remove the #eval line as it's not necessary and causing an error

end NUMINAMATH_CALUDE_ERRORFEEDBACK_olympic_javelin_throw_l913_91338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l913_91386

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define a point on the ellipse
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse x y

-- Define the distance function
noncomputable def distance (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

-- Define the dot product
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

-- Theorem statement
theorem ellipse_properties 
  (A B : PointOnEllipse) 
  (perpendicular : dot_product A.x A.y B.x B.y = 0) :
  -- 1. Distance from O to AB is constant and equal to 2√5/5
  (∃ (d : ℝ), d = 2 * Real.sqrt 5 / 5 ∧ 
    ∀ (x y : ℝ), x * A.y - y * A.x + (B.x * A.y - B.y * A.x) = 0 → 
      distance x y = d) ∧
  -- 2. Minimum value of |OA| * |OB| is 8/5
  (∀ (C D : PointOnEllipse), 
    dot_product C.x C.y D.x D.y = 0 → 
      distance C.x C.y * distance D.x D.y ≥ 8/5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l913_91386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nuts_per_box_l913_91309

/-- Given the conditions of the builder's project, prove that each box of nuts contained 15 nuts. -/
theorem nuts_per_box 
  (total_bolt_boxes : ℕ) 
  (bolts_per_box : ℕ) 
  (total_nut_boxes : ℕ) 
  (leftover_bolts : ℕ) 
  (leftover_nuts : ℕ) 
  (total_used : ℕ) 
  (h1 : total_bolt_boxes = 7)
  (h2 : bolts_per_box = 11)
  (h3 : total_nut_boxes = 3)
  (h4 : leftover_bolts = 3)
  (h5 : leftover_nuts = 6)
  (h6 : total_used = 113) :
  ∃ (nuts_per_box : ℕ),
    (total_bolt_boxes * bolts_per_box - leftover_bolts) +
    (total_nut_boxes * nuts_per_box - leftover_nuts) = total_used ∧
    nuts_per_box = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nuts_per_box_l913_91309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l913_91396

theorem trigonometric_identities (x y : Real) : 
  (Real.tan (x + y) = (Real.tan x + Real.tan y) / (1 - Real.tan x * Real.tan y)) ∧ 
  (Real.cos (x + y) = Real.cos x * Real.cos y - Real.sin x * Real.sin y) ∧ 
  (Real.cos (2 * x) = 2 * (Real.cos x)^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l913_91396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l913_91325

/-- Sequence definition -/
def x (a b : ℕ) : ℕ → ℕ 
  | 0 => 2
  | 1 => a
  | (n + 2) => a * x a b (n + 1) + b * x a b n

/-- Main theorem -/
theorem sequence_properties (a b : ℕ) (h_coprime : Nat.Coprime a b) (h_b_odd : Odd b) (h_a_gt_2 : a > 2) :
  (Even a → ∀ m n p : ℕ, m > 0 → n > 0 → p > 0 → ¬(∃ k : ℕ, x a b m = k * x a b n * x a b p)) ∧ 
  (Odd a → ∀ m n p : ℕ, m > 0 → n > 0 → p > 0 → Even (m * n * p) → ¬(∃ k : ℕ, x a b m = k^2 * x a b n * x a b p)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l913_91325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l913_91343

noncomputable def f (x : ℝ) : ℝ := 
  -Real.sqrt 2 * Real.sin (2*x + Real.pi/4) + 6 * Real.sin x * Real.cos x - 2 * (Real.cos x)^2 + 1

theorem f_properties :
  ∃ (T : ℝ),
    (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), 0 < T' → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
    T = Real.pi ∧
    (∃ (x_max : ℝ), x_max ∈ Set.Icc 0 (Real.pi/2) ∧ f x_max = 2 * Real.sqrt 2 ∧ 
      ∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi/2) → f x ≤ f x_max) ∧
    (∃ (x_min : ℝ), x_min ∈ Set.Icc 0 (Real.pi/2) ∧ f x_min = -2 ∧ 
      ∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi/2) → f x_min ≤ f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l913_91343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l913_91337

/-- The line ax + by = 0 is tangent to the circle x^2 + y^2 + ax + by = 0 for any real numbers a and b, where a^2 + b^2 ≠ 0. -/
theorem line_tangent_to_circle (a b : ℝ) (h : a^2 + b^2 ≠ 0) :
  let line := {p : ℝ × ℝ | a * p.1 + b * p.2 = 0}
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 + a * p.1 + b * p.2 = 0}
  let center := (-a/2, -b/2)
  let radius := Real.sqrt (a^2 + b^2) / 2
  let distance := |a * (-a/2) + b * (-b/2)| / Real.sqrt (a^2 + b^2)
  distance = radius :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l913_91337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_unique_l913_91362

/-- A right triangle with given leg lengths -/
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  leg1_pos : 0 < leg1
  leg2_pos : 0 < leg2

/-- The hypotenuse of a right triangle -/
noncomputable def hypotenuse (t : RightTriangle) : ℝ := Real.sqrt (t.leg1^2 + t.leg2^2)

/-- Theorem: The hypotenuse is uniquely determined by the legs of a right triangle -/
theorem hypotenuse_unique (t1 t2 : RightTriangle) (h : t1.leg1 = t2.leg1 ∧ t1.leg2 = t2.leg2) :
  hypotenuse t1 = hypotenuse t2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_unique_l913_91362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l913_91319

def is_pure_imaginary (z : ℂ) : Prop := ∃ b : ℝ, z = Complex.I * b

theorem equation_solution :
  ∀ (x : ℝ) (y : ℂ),
  is_pure_imaginary y →
  ((2 * x - 1 : ℂ) + Complex.I = y + (y - 3 * Complex.I)) ↔ (x = 2 ∧ y = Complex.I) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l913_91319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bulldozer_cannot_hit_both_sides_of_all_walls_l913_91316

-- Define a wall as a line segment in the coordinate plane
structure Wall where
  start : ℝ × ℝ
  finish : ℝ × ℝ

-- Define the coordinate plane with walls
structure CoordinatePlane where
  walls : List Wall

-- Define the bulldozer's initial position and direction
structure Bulldozer where
  position : ℝ × ℝ
  direction : ℝ × ℝ

-- Define a function to check if two walls are disjoint
def are_walls_disjoint (w1 w2 : Wall) : Prop := sorry

-- Define a function to check if a wall is not parallel to either axis
def is_wall_not_parallel_to_axes (w : Wall) : Prop := sorry

-- Define a function to simulate the bulldozer's movement
noncomputable def simulate_bulldozer_movement (plane : CoordinatePlane) (dozer : Bulldozer) : Bulldozer := sorry

-- Define a function to check if the bulldozer has hit both sides of a wall
def has_hit_both_sides (wall : Wall) (dozer : Bulldozer) : Prop := sorry

-- Theorem statement
theorem bulldozer_cannot_hit_both_sides_of_all_walls
  (plane : CoordinatePlane)
  (dozer : Bulldozer)
  (h1 : ∀ w1 w2, w1 ∈ plane.walls → w2 ∈ plane.walls → w1 ≠ w2 → are_walls_disjoint w1 w2)
  (h2 : ∀ w, w ∈ plane.walls → is_wall_not_parallel_to_axes w)
  (h3 : dozer.direction = (1, 0)) :
  ¬ (∀ w, w ∈ plane.walls → has_hit_both_sides w (simulate_bulldozer_movement plane dozer)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bulldozer_cannot_hit_both_sides_of_all_walls_l913_91316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_intersects_lateral_side_l913_91372

/-- Represents an isosceles trapezoid ABCD -/
structure IsoscelesTrapezoid where
  /-- Length of the shorter base BC -/
  bc : ℝ
  /-- Length of the longer base AD -/
  ad : ℝ
  /-- Area of the trapezoid -/
  area : ℝ
  /-- Condition that AD > BC -/
  base_inequality : ad > bc
  /-- Condition that the trapezoid is isosceles -/
  isosceles : True

/-- 
Theorem: In an isosceles trapezoid ABCD with bases BC = 4 cm and AD = 8 cm, 
and area 21 cm², the angle bisector of angle A intersects the lateral side CD.
-/
theorem angle_bisector_intersects_lateral_side 
  (trapezoid : IsoscelesTrapezoid) 
  (h1 : trapezoid.bc = 4) 
  (h2 : trapezoid.ad = 8) 
  (h3 : trapezoid.area = 21) : 
  ∃ (point : ℝ × ℝ), point.1 ≥ 0 ∧ point.1 ≤ trapezoid.bc :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_intersects_lateral_side_l913_91372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_overlap_shift_l913_91382

/-- The given function f(x) = sin(2x + π/3) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

/-- The property that the graph overlaps when shifted by φ units left and right -/
def overlaps (φ : ℝ) : Prop := ∀ x, f (x + φ) = f (x - φ)

/-- The theorem stating that π/2 is the smallest positive φ that satisfies the overlap condition -/
theorem smallest_overlap_shift :
  (∃ φ > 0, overlaps φ) ∧ (∀ φ > 0, overlaps φ → φ ≥ Real.pi / 2) ∧ overlaps (Real.pi / 2) := by
  sorry

#check smallest_overlap_shift

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_overlap_shift_l913_91382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_pi_irrational_l913_91317

-- Define the set of numbers we're considering
def numbers : Set ℝ := {0, 1/2, Real.pi, Real.sqrt 4}

-- Define a predicate for irrationality
def IsIrrational (x : ℝ) : Prop := ∀ p q : ℤ, q ≠ 0 → x ≠ (p : ℝ) / (q : ℝ)

theorem only_pi_irrational :
  ∃! x, x ∈ numbers ∧ IsIrrational x ∧ x = Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_pi_irrational_l913_91317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_for_radius_6_l913_91358

/-- The area of the shaded region formed by four intersecting circles of radius 6 units -/
noncomputable def shaded_area (r : ℝ) : ℝ :=
  8 * (Real.pi * r^2 / 4 - r^2 / 2)

/-- Theorem stating that the shaded area for circles of radius 6 is 72π - 144 -/
theorem shaded_area_for_radius_6 :
  shaded_area 6 = 72 * Real.pi - 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_for_radius_6_l913_91358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_cutting_l913_91381

theorem rectangle_cutting (m n k : ℕ) :
  (∃ (partition : List (ℕ × ℕ)), 
    (∀ (rect : ℕ × ℕ), rect ∈ partition → rect.1 = 1 ∧ rect.2 = k) ∧
    (List.sum (List.map (λ rect => rect.1 * rect.2) partition) = m * n)) ↔ 
  (k ∣ m ∨ k ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_cutting_l913_91381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_workers_is_ten_l913_91334

/-- Represents the job completion scenario -/
structure JobCompletion where
  initial_days : ℕ
  work_done_fraction : ℚ
  days_passed : ℕ
  people_fired : ℕ
  total_days : ℕ

/-- Calculates the number of people initially hired -/
def calculate_initial_workers (job : JobCompletion) : ℚ :=
  let remaining_work := 1 - job.work_done_fraction
  let remaining_days := job.initial_days - job.days_passed
  let actual_remaining_days := job.total_days - job.days_passed
  (100 * remaining_work * remaining_days) / (actual_remaining_days * remaining_days - 100 * remaining_work * job.people_fired)

/-- Theorem stating that the number of people initially hired is 10 -/
theorem initial_workers_is_ten (job : JobCompletion) :
  job.initial_days = 100 ∧
  job.work_done_fraction = 1/4 ∧
  job.days_passed = 20 ∧
  job.people_fired = 2 ∧
  job.total_days = 95 →
  calculate_initial_workers job = 10 := by
  sorry

#eval calculate_initial_workers {
  initial_days := 100,
  work_done_fraction := 1/4,
  days_passed := 20,
  people_fired := 2,
  total_days := 95
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_workers_is_ten_l913_91334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_problem_l913_91368

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 3)^2 + (y - 6)^2 = 36

-- Define point M
def point_M : ℝ × ℝ := (0, 3)

-- Define the line l passing through point M
def line_l (k b : ℝ) (x y : ℝ) : Prop := y = k * x + b ∧ point_M.2 = k * point_M.1 + b

-- Define the function to calculate the area of a triangle
noncomputable def triangle_area (base height : ℝ) : ℝ := (1 / 2) * base * height

theorem circle_line_intersection_problem :
  -- Part 1: Equation of line l
  ∃ (k b : ℝ), (∀ x y, line_l k b x y ↔ x - y + 3 = 0) ∧
  -- Part 2: Maximum area of triangle PAB
  ∃ (A B P : ℝ × ℝ),
    circle_C A.1 A.2 ∧
    circle_C B.1 B.2 ∧
    circle_C P.1 P.2 ∧
    P ≠ A ∧ P ≠ B ∧
    line_l k b A.1 A.2 ∧
    line_l k b B.1 B.2 ∧
    triangle_area (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) 
                  (Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)) ≤ 18 + 18 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_problem_l913_91368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_representation_nonzero_digits_l913_91322

-- Define a function to count non-zero digits in base b representation
def number_of_nonzero_digits_base (b : ℕ) (a : ℕ) : ℕ :=
  sorry -- Implementation details omitted for brevity

theorem base_representation_nonzero_digits
  (a b n : ℕ) 
  (h_b : b > 1)
  (h_n : n > 0)
  (h_div : (b ^ n - 1) ∣ a) :
  (number_of_nonzero_digits_base b a) ≥ n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_representation_nonzero_digits_l913_91322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_lines_l913_91356

-- Define the points
def P : ℝ × ℝ := (1, 2)
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (0, -5)

-- Define the distance function
noncomputable def distance (p : ℝ × ℝ) (line : ℝ → ℝ) : ℝ :=
  abs (line p.1 - p.2) / Real.sqrt (1 + ((line (p.1 + 1) - line p.1) ^ 2))

-- Define the two lines
def line1 (_ : ℝ) : ℝ := 1
def line2 (x : ℝ) : ℝ := 4 * x - 2

-- Theorem statement
theorem equidistant_lines :
  (line1 P.1 = P.2 ∧ distance A line1 = distance B line1) ∨
  (line2 P.1 = P.2 ∧ distance A line2 = distance B line2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_lines_l913_91356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_through_point_function_through_point_2_function_through_point_3_function_through_point_4_l913_91345

theorem function_through_point (x y b : ℝ) (f : ℝ → ℝ) : 
  (x = 3 ∧ y = 10 ∧ f x = y ∧ f = λ t ↦ t + b) → b = 7 :=
by sorry

theorem function_through_point_2 (x y b : ℝ) (f : ℝ → ℝ) : 
  (x = 3 ∧ y = 10 ∧ f x = y ∧ f = λ t ↦ 3*t + b) → b = 1 :=
by sorry

theorem function_through_point_3 (x y b : ℝ) (f : ℝ → ℝ) : 
  (x = 3 ∧ y = 10 ∧ f x = y ∧ f = λ t ↦ -1/3*t + b) → b = 11 :=
by sorry

theorem function_through_point_4 (x y b : ℝ) (f : ℝ → ℝ) : 
  (x = 3 ∧ y = 10 ∧ f x = y ∧ f = λ t ↦ -1/2*t + b) → b = 11.5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_through_point_function_through_point_2_function_through_point_3_function_through_point_4_l913_91345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l913_91329

-- Define necessary concepts
def is_focus (p : ℝ × ℝ) : Prop := sorry
def is_asymptote (l : Set (ℝ × ℝ)) : Prop := sorry
def distance (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ := sorry
def eccentricity (a b : ℝ) : ℝ := sorry

theorem hyperbola_eccentricity 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_eq : ∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1) 
  (h_dist : ∃ f : ℝ × ℝ, ∃ l : Set (ℝ × ℝ), 
    is_focus f ∧ 
    is_asymptote l ∧ 
    distance f l = 2 * Real.sqrt 5 / 5 * a) :
  eccentricity a b = 3 * Real.sqrt 5 / 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l913_91329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_cubes_between_powers_of_two_l913_91308

theorem perfect_cubes_between_powers_of_two : 
  (Finset.filter (fun n => 2^7 + 1 ≤ n^3 ∧ n^3 ≤ 2^12 + 1) (Finset.range 17)).card = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_cubes_between_powers_of_two_l913_91308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_to_hexagon_ratio_l913_91366

/-- Shape type to represent different geometric shapes -/
inductive Shape
  | Rectangle : ℝ → ℝ → Shape
  | Diamond : ℝ → Shape
  | Hexagon : ℝ → Shape

/-- Function representing the folding and cutting process -/
def diagonal_fold_then_median_fold (a b : ℝ) : Shape :=
  sorry

/-- Representation of a regular hexagon -/
def regular_hexagon : Shape :=
  sorry

/-- 
Given a rectangle with sides a and b, if folding along the diagonal
and then along the median of the resulting diamond shape produces
a regular hexagon, then a = √3 * b.
-/
theorem rectangle_to_hexagon_ratio (a b : ℝ) 
  (h_positive : a > 0 ∧ b > 0)
  (h_hexagon : diagonal_fold_then_median_fold a b = regular_hexagon) :
  a = Real.sqrt 3 * b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_to_hexagon_ratio_l913_91366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odot_ten_four_l913_91314

noncomputable def odot (a b : ℝ) : ℝ := a + (4 * a) / (3 * b)

theorem odot_ten_four : odot 10 4 = 40/3 := by
  -- Unfold the definition of odot
  unfold odot
  -- Simplify the expression
  simp [div_eq_mul_inv]
  -- Perform arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odot_ten_four_l913_91314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_identity_l913_91318

/-- The function f(x) = ax / (x + 2) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x) / (x + 2)

/-- Theorem stating that f(f(x)) = x for all x ≠ -2 if and only if a = -1 -/
theorem f_composition_identity (a : ℝ) :
  (∀ x : ℝ, x ≠ -2 → f a (f a x) = x) ↔ a = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_identity_l913_91318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_is_one_half_l913_91326

/-- Given that the terminal side of angle α passes through the point (1/2, √3/2), prove that cos(α) = 1/2 -/
theorem cos_alpha_is_one_half (α : ℝ) (h : ∃ (r : ℝ), r * Real.cos α = 1/2 ∧ r * Real.sin α = Real.sqrt 3/2) : 
  Real.cos α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_is_one_half_l913_91326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_intersection_l913_91357

/-- An ellipse with eccentricity e₁ -/
structure Ellipse :=
  (e₁ : ℝ)
  (h₁ : 0 < e₁ ∧ e₁ < 1)

/-- A hyperbola with eccentricity e₂ -/
structure Hyperbola :=
  (e₂ : ℝ)
  (h₂ : e₂ > 1)

/-- A point in the first quadrant -/
structure Point :=
  (x : ℝ)
  (y : ℝ)
  (hx : x > 0)
  (hy : y > 0)

/-- The foci of the ellipse and hyperbola -/
structure Foci :=
  (F₁ : Point)
  (F₂ : Point)

/-- Membership of a point in an ellipse -/
def Point.memberOf (P : Point) (C : Ellipse) : Prop := sorry

/-- Membership of a point in a hyperbola -/
def Point.memberOfHyperbola (P : Point) (C : Hyperbola) : Prop := sorry

/-- Distance between two points -/
noncomputable def dist (P Q : Point) : ℝ := sorry

/-- The theorem statement -/
theorem ellipse_hyperbola_intersection (C₁ : Ellipse) (C₂ : Hyperbola) (F : Foci) (P : Point) 
  (h_common : P.memberOf C₁ ∧ P.memberOfHyperbola C₂)
  (h_dist : ∃ k : ℝ, dist P F.F₁ = k * dist P F.F₂)
  (h_ecc : C₁.e₁ / C₂.e₂ = 1 / (k - 1)) :
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_intersection_l913_91357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_operation_preserves_sum_of_squares_invariant_sum_of_squares_impossibility_of_transformation_main_theorem_l913_91388

def initial_triple : Fin 3 → ℝ := ![3, 4, 12]

def target_triple : Fin 3 → ℝ := ![2, 8, 10]

def sum_of_squares (v : Fin 3 → ℝ) : ℝ :=
  (v 0) ^ 2 + (v 1) ^ 2 + (v 2) ^ 2

def operation (a b : ℝ) : Fin 2 → ℝ :=
  ![0.6 * a - 0.8 * b, 0.8 * a + 0.6 * b]

theorem operation_preserves_sum_of_squares (a b : ℝ) :
  (a ^ 2 + b ^ 2) = ((operation a b 0) ^ 2 + (operation a b 1) ^ 2) := by sorry

theorem invariant_sum_of_squares (v : Fin 3 → ℝ) (i j : Fin 3) (h : i ≠ j) :
  sum_of_squares v = sum_of_squares (Function.update (Function.update v i (operation (v i) (v j) 0)) j (operation (v i) (v j) 1)) := by sorry

theorem impossibility_of_transformation :
  sum_of_squares initial_triple ≠ sum_of_squares target_triple := by sorry

theorem main_theorem : ¬∃ (n : ℕ), ∃ (sequence : Fin (n + 1) → (Fin 3 → ℝ)),
  sequence 0 = initial_triple ∧
  sequence n = target_triple ∧
  ∀ i : Fin n, ∃ j k : Fin 3, j ≠ k ∧
    sequence (i + 1) = Function.update (Function.update (sequence i) j (operation (sequence i j) (sequence i k) 0))
      k (operation (sequence i j) (sequence i k) 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_operation_preserves_sum_of_squares_invariant_sum_of_squares_impossibility_of_transformation_main_theorem_l913_91388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_perpendicular_lines_l913_91399

-- Define the slope of a line given its equation y = mx + b
def line_slope (m : ℝ) : ℝ := m

-- Define when two lines are parallel
def parallel (m₁ m₂ : ℝ) : Prop := m₁ = m₂

-- Define when two lines are perpendicular
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

theorem parallel_lines (a : ℝ) :
  parallel (line_slope (-1)) (line_slope (a^2 - 2)) ↔ a = -1 := by sorry

theorem perpendicular_lines (a : ℝ) :
  perpendicular (line_slope (2*a - 1)) (line_slope 4) ↔ a = 3/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_perpendicular_lines_l913_91399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_subtraction_correction_fraction_subtraction_correction_proof_l913_91353

theorem fraction_subtraction_correction (incorrect_result : ℚ) 
  (mistaken_subtrahend : ℚ) (correct_subtrahend : ℚ) : ℚ :=
  by
    -- Define the conditions
    have h1 : incorrect_result = 9/8 := by sorry
    have h2 : mistaken_subtrahend = 1/8 := by sorry
    have h3 : correct_subtrahend = 5/8 := by sorry

    -- Define the minuend
    let minuend : ℚ := incorrect_result + mistaken_subtrahend

    -- Calculate the correct result
    let correct_result : ℚ := minuend - correct_subtrahend

    -- Return the correct result
    exact correct_result

-- Proof that the correct result is indeed 5/8
theorem fraction_subtraction_correction_proof :
  fraction_subtraction_correction (9/8) (1/8) (5/8) = 5/8 :=
  by
    -- Unfold the definition of fraction_subtraction_correction
    unfold fraction_subtraction_correction
    -- Simplify the arithmetic
    simp
    -- The proof is complete
    rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_subtraction_correction_fraction_subtraction_correction_proof_l913_91353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l913_91323

/-- Represents a pipe that can fill a tank. -/
structure Pipe where
  fillTime : ℚ
  fillRate : ℚ := 1 / fillTime

/-- Represents a tank that can be filled by pipes. -/
structure Tank where
  pipes : List Pipe
  fillSequence : List (List Pipe)
  totalTime : ℚ

instance : Inhabited Pipe where
  default := ⟨1, 1⟩

def Tank.fillFraction (t : Tank) (pipes : List Pipe) (time : ℚ) : ℚ :=
  (pipes.map Pipe.fillRate).sum * time

theorem tank_fill_time (t : Tank) (hPipes : t.pipes.length = 4)
  (hFillTimes : t.pipes.map Pipe.fillTime = [60, 40, 30, 24])
  (hSequence : t.fillSequence = [
    [t.pipes[0]!, t.pipes[1]!],
    [t.pipes[1]!, t.pipes[2]!, t.pipes[3]!],
    [t.pipes[0]!, t.pipes[2]!, t.pipes[3]!]
  ])
  (hTotalFill : t.fillFraction [t.pipes[0]!, t.pipes[1]!] (t.totalTime / 3) +
                t.fillFraction [t.pipes[1]!, t.pipes[2]!, t.pipes[3]!] (t.totalTime / 3) +
                t.fillFraction [t.pipes[0]!, t.pipes[2]!, t.pipes[3]!] (t.totalTime / 3) = 1) :
  ⌈t.totalTime⌉ = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l913_91323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2008_times_30_l913_91341

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Add this case for n = 0
  | n + 1 => (1 + sequence_a n) / (1 - sequence_a n)

theorem a_2008_times_30 : 30 * sequence_a 2008 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2008_times_30_l913_91341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_inverse_cancellation_l913_91395

theorem power_inverse_cancellation (x : ℝ) (hx : x ≠ 0) :
  x^4 * x^(-4 : ℤ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_inverse_cancellation_l913_91395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l913_91370

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Check if two vectors are parallel -/
def areParallel (v1 v2 : Vector2D) : Prop :=
  v1.x * v2.y = v1.y * v2.x

/-- Calculate the area of a triangle given two sides and the included angle -/
noncomputable def triangleArea (a c B : ℝ) : ℝ :=
  1/2 * a * c * Real.sin B

theorem triangle_proof (t : Triangle) 
  (hm : Vector2D := { x := t.a - t.c, y := t.a - t.b })
  (hn : Vector2D := { x := t.a + t.b, y := t.c })
  (h_parallel : areParallel hm hn)
  (ha : t.a = 1)
  (hb : t.b = Real.sqrt 7) :
  t.B = π/3 ∧ triangleArea t.a t.c t.B = 3 * Real.sqrt 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l913_91370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l913_91392

theorem max_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2 - 3*x*y + 4*y^2 - z = 0) :
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧
    x₀^2 - 3*x₀*y₀ + 4*y₀^2 - z₀ = 0 ∧
    (∀ (x' y' z' : ℝ), x' > 0 → y' > 0 → z' > 0 →
      x'^2 - 3*x'*y' + 4*y'^2 - z' = 0 →
      x'*y'/z' ≤ x₀*y₀/z₀) ∧
    (2/x₀ + 1/y₀ - 2/z₀ = 1) ∧
    (∀ (x' y' z' : ℝ), x' > 0 → y' > 0 → z' > 0 →
      x'^2 - 3*x'*y' + 4*y'^2 - z' = 0 →
      2/x' + 1/y' - 2/z' ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l913_91392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_always_nonnegative_iff_m_in_range_l913_91333

theorem quadratic_always_nonnegative_iff_m_in_range (m : ℝ) : 
  (∀ x : ℝ, x^2 - m*x + 1 ≥ 0) ↔ m ∈ Set.Icc (-2) 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_always_nonnegative_iff_m_in_range_l913_91333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_gains_five_minutes_l913_91376

/-- A watch that gains time -/
structure GainingWatch where
  /-- The number of degrees the second hand moves per minute -/
  degrees_per_minute : ℝ
  /-- Assumption that the watch moves faster than normal -/
  moves_fast : degrees_per_minute > 360

/-- The number of minutes gained by the watch in one hour -/
noncomputable def minutes_gained (w : GainingWatch) : ℝ :=
  ((w.degrees_per_minute - 360) / 6) / 60

/-- Theorem stating that a watch with a second hand moving 390 degrees per minute gains 5 minutes per hour -/
theorem watch_gains_five_minutes 
  (w : GainingWatch) 
  (h : w.degrees_per_minute = 390) : 
  minutes_gained w = 5 := by
  sorry

#check watch_gains_five_minutes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_gains_five_minutes_l913_91376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_who_just_passed_l913_91355

theorem students_who_just_passed 
  (total_students : ℕ) 
  (first_division_percent : ℚ)
  (second_division_percent : ℚ)
  (h1 : total_students = 300)
  (h2 : first_division_percent = 28 / 100)
  (h3 : second_division_percent = 54 / 100)
  (h4 : first_division_percent + second_division_percent < 1)
  : (total_students : ℚ) * (1 - first_division_percent - second_division_percent) = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_who_just_passed_l913_91355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_divisibility_l913_91359

theorem unique_prime_divisibility :
  ∃! n : ℕ, 8001 < n ∧ n < 8200 ∧
  (∀ k : ℕ, k > n → (2^n - 1 : ℕ) ∣ (2^(k * Nat.factorial (n-1) + k^n) - 1 : ℕ)) ∧
  n = 8111 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_divisibility_l913_91359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_price_difference_l913_91387

/-- Proves that the difference between the original price and the final price
    of a dress after a series of discounts and increases is $3.15 -/
theorem dress_price_difference (original_price : ℝ) (price_after_discount : ℝ) : 
  original_price = 72 →
  price_after_discount = original_price * 0.85 →
  price_after_discount = 61.2 →
  original_price - (price_after_discount * 1.25 * 0.9) = 3.15 := by
  sorry

#check dress_price_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_price_difference_l913_91387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_sum_divisible_by_100_l913_91398

theorem card_sum_divisible_by_100 (S : Finset ℕ) : 
  S.card = 5000 ∧ 
  (∀ n, n ∈ S → 1 ≤ n ∧ n ≤ 5000) ∧ 
  (∀ m n, m ∈ S → n ∈ S → m ≠ n → m ≠ n) →
  (S.filter (λ x => ∃ y, y ∈ S ∧ x ≠ y ∧ (x + y) % 100 = 0)).card = 124950 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_sum_divisible_by_100_l913_91398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_inequalities_l913_91369

/-- An acute triangle with angle A greater than angle B -/
structure AcuteTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi
  A_gt_B : A > B

theorem acute_triangle_inequalities (t : AcuteTriangle) :
  (Real.sin t.A > Real.sin t.B) ∧
  (Real.cos t.A < Real.cos t.B) ∧
  (Real.cos (2 * t.A) < Real.cos (2 * t.B)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_inequalities_l913_91369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l913_91373

noncomputable def f (x : ℝ) : ℝ := (x - 4) / (x - 3)

def P (m : ℝ) : Prop := ∀ x y, m ≤ x ∧ x < y → f x < f y

def Q (m : ℝ) : Prop := ∀ x, Real.pi/2 ≤ x ∧ x ≤ 3*Real.pi/4 → 4 * Real.sin (2*x + Real.pi/4) ≤ m

theorem range_of_m (m : ℝ) : 
  (P m ∨ Q m) ∧ ¬(P m ∧ Q m) → -2 * Real.sqrt 2 ≤ m ∧ m ≤ 3 := by
  sorry

#check range_of_m

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l913_91373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_bulb_not_census_suitable_l913_91306

/-- Represents a type of investigation --/
inductive Investigation
| SecurityCheck
| TeacherRecruitment
| ReadingTime
| LightBulbLifespan

/-- Defines whether an investigation is suitable for a census --/
def isCensusSuitable (i : Investigation) : Prop :=
  match i with
  | Investigation.SecurityCheck => True
  | Investigation.TeacherRecruitment => True
  | Investigation.ReadingTime => True
  | Investigation.LightBulbLifespan => False

/-- Represents the ability to examine a unit without alteration --/
def can_examine_without_alteration (i : Investigation) (unit : Unit) : Prop := sorry

/-- Defines that a census requires examining every unit without alteration --/
axiom census_requirement : 
  ∀ (i : Investigation), isCensusSuitable i ↔ 
    (∀ (unit : Unit), can_examine_without_alteration i unit)

/-- Axiom stating that light bulb lifespan testing alters the units --/
axiom light_bulb_alters : 
  ∀ (unit : Unit), ¬(can_examine_without_alteration Investigation.LightBulbLifespan unit)

/-- Theorem: Light bulb lifespan investigation is not suitable for a census --/
theorem light_bulb_not_census_suitable : 
  ¬(isCensusSuitable Investigation.LightBulbLifespan) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_bulb_not_census_suitable_l913_91306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_exponent_sum_for_1983_l913_91342

/-- Represents a sum of distinct powers of 2 -/
def PowerSum : Type := List Nat

/-- Checks if a PowerSum is valid for a given number -/
def isValidPowerSum (n : Nat) (ps : PowerSum) : Prop :=
  (ps.map (fun x => 2^x)).sum = n ∧
  ps.length ≥ 5 ∧
  ps.Nodup

/-- The target number we're working with -/
def targetNumber : Nat := 1983

/-- The theorem to prove -/
theorem least_exponent_sum_for_1983 :
  ∀ (ps : PowerSum),
    isValidPowerSum targetNumber ps →
    ps.sum ≥ 55 := by
  sorry

#check least_exponent_sum_for_1983

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_exponent_sum_for_1983_l913_91342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_composite_function_l913_91361

-- Define the function f with domain [0,2]
def f : Set ℝ := Set.Icc 0 2

-- Define the composite function g(x) = f(2^x - 2)
noncomputable def g (x : ℝ) : ℝ := Real.exp (Real.log 2 * x) - 2

-- Theorem statement
theorem domain_of_composite_function :
  {x : ℝ | g x ∈ f} = Set.Icc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_composite_function_l913_91361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_max_min_in_interval_l913_91301

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x + 2) / (2*x - 4)

-- Define the open interval
def I : Set ℝ := Set.Ioo (-1) 5

-- Theorem statement
theorem no_max_min_in_interval :
  ¬∃ (y : ℝ), (∀ (x : ℝ), x ∈ I → f x ≥ y) ∨ (∀ (x : ℝ), x ∈ I → f x ≤ y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_max_min_in_interval_l913_91301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_area_l913_91320

/-- Represents the dimensions of the floor -/
noncomputable def floor_length : ℝ := 12

/-- Represents the dimensions of the floor -/
noncomputable def floor_width : ℝ := 15

/-- Represents the dimensions of a single tile -/
noncomputable def tile_size : ℝ := 2

/-- Represents the radius of each quarter circle on a tile -/
noncomputable def quarter_circle_radius : ℝ := 1 / 2

/-- Calculates the total number of tiles on the floor -/
noncomputable def total_tiles : ℝ := (floor_length / tile_size) * (floor_width / tile_size)

/-- Represents the shaded area of a single tile -/
noncomputable def shaded_area_per_tile : ℝ := tile_size^2 - Real.pi * quarter_circle_radius^2

/-- Theorem stating the total shaded area of the floor -/
theorem total_shaded_area :
  total_tiles * shaded_area_per_tile = 180 - 45 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_area_l913_91320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_conic_l913_91371

/-- The geometric mean of two positive real numbers -/
noncomputable def geometric_mean (a b : ℝ) : ℝ := Real.sqrt (a * b)

/-- The eccentricity of a conic section given by x^2 + y^2/m = 1 -/
noncomputable def eccentricity (m : ℝ) : ℝ :=
  if m > 0 then
    Real.sqrt 3 / 2
  else if m < 0 then
    Real.sqrt 5
  else
    0  -- undefined for m = 0

theorem eccentricity_of_conic (m : ℝ) :
  m = geometric_mean 2 8 →
  eccentricity m = Real.sqrt 3 / 2 ∨ eccentricity m = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_conic_l913_91371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_theorem_l913_91379

/-- Represents a rhombus with given side length and one diagonal --/
structure Rhombus where
  side : ℝ
  diagonal1 : ℝ

/-- Calculates the area of a rhombus given its two diagonals --/
noncomputable def rhombusArea (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

/-- Theorem: A rhombus with side 26 and one diagonal 20 has an area of 480 --/
theorem rhombus_area_theorem (r : Rhombus) (h1 : r.side = 26) (h2 : r.diagonal1 = 20) :
  ∃ d2 : ℝ, rhombusArea r.diagonal1 d2 = 480 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_theorem_l913_91379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_containing_sphere_radius_is_correct_l913_91347

/-- The radius of a small sphere -/
def small_sphere_radius : ℝ := 2

/-- The coordinates of the centers of the four spheres -/
def sphere_centers : List (ℝ × ℝ) := [(2, 2), (-2, 2), (-2, -2), (2, -2)]

/-- The radius of the smallest sphere that contains all four spheres -/
noncomputable def containing_sphere_radius : ℝ := 2 * Real.sqrt 2 + 2

/-- Theorem stating that the containing sphere radius is correct -/
theorem containing_sphere_radius_is_correct :
  ∀ center ∈ sphere_centers,
    ‖(containing_sphere_radius, 0, 0) - (center.1, center.2, small_sphere_radius)‖ = containing_sphere_radius :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_containing_sphere_radius_is_correct_l913_91347
