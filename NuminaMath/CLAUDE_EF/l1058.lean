import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1058_105893

noncomputable def sequence_a : ℕ → ℝ := sorry

noncomputable def S : ℕ → ℝ := sorry

axiom a_1 : sequence_a 1 = 1
axiom S_arithmetic : ∃ d : ℝ, ∀ n : ℕ, n ≥ 1 → S (n+1) / (n+1) - S n / n = d
axiom S_sum : S 2 / 2 + S 3 / 3 + S 4 / 4 = 6

noncomputable def sequence_b (n : ℕ) : ℝ := 
  sequence_a (n+1) / sequence_a (n+2) + sequence_a (n+2) / sequence_a (n+1) - 2

noncomputable def T : ℕ → ℝ := sorry

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → sequence_a n = n) ∧
  (∀ n : ℕ, n ≥ 1 → T n = 1/2 - 1/(n+2)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1058_105893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sliding_right_triangle_locus_l1058_105817

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a right triangle -/
structure RightTriangle where
  a : Point  -- Right angle vertex
  b : Point  -- One end of hypotenuse
  c : Point  -- Other end of hypotenuse

/-- The length of the hypotenuse of a right triangle -/
noncomputable def hypotenuseLength (t : RightTriangle) : ℝ :=
  Real.sqrt ((t.b.x - t.c.x)^2 + (t.b.y - t.c.y)^2)

/-- Predicate to check if a point is on a quarter-circle arc -/
def isOnQuarterCircle (center : Point) (radius : ℝ) (p : Point) : Prop :=
  (p.x - center.x)^2 + (p.y - center.y)^2 = radius^2 ∧
  p.x ≥ center.x ∧ p.y ≥ center.y

/-- Theorem: The right angle vertex of a sliding right triangle forms a quarter-circle arc -/
theorem sliding_right_triangle_locus (t : RightTriangle) 
  (h1 : t.b.x = 0 ∨ t.b.y = 0)  -- One end of hypotenuse on x or y axis
  (h2 : t.c.x = 0 ∨ t.c.y = 0)  -- Other end of hypotenuse on x or y axis
  (h3 : t.b.x ≠ t.c.x ∨ t.b.y ≠ t.c.y)  -- Hypotenuse not degenerate
  : isOnQuarterCircle ⟨0, 0⟩ (hypotenuseLength t) t.a := by
  sorry

-- Part (b) is not formalized here as it would require additional structures and definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sliding_right_triangle_locus_l1058_105817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_from_tan_l1058_105822

theorem sin_value_from_tan (α : ℝ) (h1 : Real.tan α = -1/2) (h2 : π/2 < α) (h3 : α < π) :
  Real.sin α = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_from_tan_l1058_105822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_every_nat_appears_periodic_on_lines_placement_possible_l1058_105840

/-- A function that assigns natural numbers to integer coordinate points on a plane. -/
def f : ℤ × ℤ → ℕ := sorry

/-- Theorem stating that every natural number appears at some point. -/
theorem every_nat_appears : ∀ n : ℕ, ∃ p : ℤ × ℤ, f p = n := by
  sorry

/-- Theorem stating that the arrangement of numbers is periodic on lines not passing through the origin. -/
theorem periodic_on_lines : 
  ∀ a b c : ℤ, (a ≠ 0 ∨ b ≠ 0) → 
  ∃ u v : ℤ, (u ≠ 0 ∨ v ≠ 0) ∧ 
  (∀ x y : ℤ, a * x + b * y = c → f (x + u, y + v) = f (x, y)) := by
  sorry

/-- Main theorem stating that it's possible to place natural numbers on the plane with the given properties. -/
theorem placement_possible : ∃ f : ℤ × ℤ → ℕ, 
  (∀ n : ℕ, ∃ p : ℤ × ℤ, f p = n) ∧ 
  (∀ a b c : ℤ, (a ≠ 0 ∨ b ≠ 0) → 
    ∃ u v : ℤ, (u ≠ 0 ∨ v ≠ 0) ∧ 
    (∀ x y : ℤ, a * x + b * y = c → f (x + u, y + v) = f (x, y))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_every_nat_appears_periodic_on_lines_placement_possible_l1058_105840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_range_of_a_l1058_105895

-- Define the function f(x) = 2|x-1| - |x-4|
def f (x : ℝ) : ℝ := 2 * |x - 1| - |x - 4|

-- Define the range of f
def range_f : Set ℝ := {y | ∃ x, f x = y}

-- Theorem 1: The range of f is [-3, +∞)
theorem range_of_f : range_f = Set.Ici (-3) := by sorry

-- Define the function g(x, a) = 2|x-1| - |x-a|
def g (x a : ℝ) : ℝ := 2 * |x - 1| - |x - a|

-- Theorem 2: g(x, a) ≥ -1 for all x ∈ ℝ if and only if a ∈ [0, 2]
theorem range_of_a : (∀ x, g x a ≥ -1) ↔ a ∈ Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_range_of_a_l1058_105895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l1058_105899

theorem trigonometric_inequality : Real.tan 1 > Real.sin 1 ∧ Real.sin 1 > Real.cos 1 ∧ Real.cos 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l1058_105899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_on_parabola_l1058_105860

-- Define the set of points x = y^2
def set_points (x y : ℝ) : Prop := x = y^2

-- Define the circle (x - 11)^2 + (y - 1)^2 = 25
def circle_equation (x y : ℝ) : Prop := (x - 11)^2 + (y - 1)^2 = 25

-- Define the parabola y = (1/2)x^2 - (21/2)x + 97/2
def parabola (x y : ℝ) : Prop := y = (1/2) * x^2 - (21/2) * x + 97/2

theorem intersection_points_on_parabola :
  ∀ x y : ℝ, set_points x y ∧ circle_equation x y → parabola x y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_on_parabola_l1058_105860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_antiderivative_of_inverse_cosh_l1058_105888

noncomputable def f (x : ℝ) : ℝ := 1 / Real.cosh x

noncomputable def F (x : ℝ) : ℝ := 2 * Real.arctan (Real.exp x)

theorem antiderivative_of_inverse_cosh :
  ∀ x : ℝ, HasDerivAt F (f x) x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_antiderivative_of_inverse_cosh_l1058_105888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1058_105833

/-- The speed of the train the man is sitting in, in km/h -/
noncomputable def V : ℝ := 60

/-- The speed of the goods train, in km/h -/
noncomputable def goods_train_speed : ℝ := 52

/-- The length of the goods train, in meters -/
noncomputable def goods_train_length : ℝ := 280

/-- The time it takes for the goods train to pass the man, in seconds -/
noncomputable def passing_time : ℝ := 9

/-- Conversion factor from km/h to m/s -/
noncomputable def kmh_to_ms : ℝ := 5 / 18

theorem train_speed_calculation :
  V = (goods_train_length / passing_time / kmh_to_ms) - goods_train_speed := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1058_105833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1058_105885

/-- The function g(t) = (t^2 + t) / (t^2 + 1) -/
noncomputable def g (t : ℝ) : ℝ := (t^2 + t) / (t^2 + 1)

/-- The range of g is the singleton set {1/2} -/
theorem range_of_g :
  Set.range g = {1/2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1058_105885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_prime_squares_average_l1058_105803

def is_prime (n : ℕ) : Bool :=
  if n ≤ 1 then false
  else
    (List.range (n - 1)).all (λ d => d + 2 = n ∨ n % (d + 2) ≠ 0)

def non_primes : List ℕ :=
  (List.range 49).map (λ x => x + 51) |>.filter (λ x => ¬(is_prime x))

def sum_of_squares (lst : List ℕ) : ℕ :=
  lst.map (λ x => x * x) |>.sum

theorem non_prime_squares_average : 
  (sum_of_squares non_primes) / (non_primes.length) = 250289 / 39 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_prime_squares_average_l1058_105803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_ratio_l1058_105850

-- Define the parabola
noncomputable def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the focus of the parabola
noncomputable def focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define the line passing through the focus with inclination angle π/3
noncomputable def line (p : ℝ) (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * (x - p/2)

-- Define the intersection points
noncomputable def intersection_points (p : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((p, Real.sqrt 3 * p), (p/3, -Real.sqrt 3 * p/3))

-- Theorem statement
theorem parabola_line_intersection_ratio (p : ℝ) :
  let (A, B) := intersection_points p
  let F := focus p
  parabola p A.1 A.2 ∧ parabola p B.1 B.2 ∧ 
  line p A.1 A.2 ∧ line p B.1 B.2 →
  (A.2 - F.2) / (F.2 - B.2) = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_ratio_l1058_105850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_five_ninths_l1058_105806

/-- The set of ball numbers -/
def BallNumbers : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- Function to check if a product is divisible by 3 -/
def isDivisibleByThree (a b : ℕ) : Bool :=
  (a * b) % 3 = 0

/-- The probability of drawing two balls with a product divisible by 3 -/
noncomputable def probabilityOfProductDivisibleByThree : ℚ :=
  let totalOutcomes := (BallNumbers.card * BallNumbers.card : ℕ)
  let favorableOutcomes := Finset.sum BallNumbers (fun a => 
    Finset.sum BallNumbers (fun b => if isDivisibleByThree a b then 1 else 0))
  ↑favorableOutcomes / ↑totalOutcomes

theorem probability_is_five_ninths :
  probabilityOfProductDivisibleByThree = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_five_ninths_l1058_105806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l1058_105877

/-- The maximum distance from any point on the circle x² + y² = 3/2 to the line √7x - y = √2 is (√6/2) + (1/2). -/
theorem max_distance_circle_to_line :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 3/2}
  let line := {(x, y) : ℝ × ℝ | Real.sqrt 7 * x - y = Real.sqrt 2}
  ∃ d_max : ℝ, d_max = Real.sqrt 6 / 2 + 1 / 2 ∧
    ∀ p ∈ circle, ∀ q ∈ line, dist p q ≤ d_max ∧
    ∃ p' ∈ circle, ∃ q' ∈ line, dist p' q' = d_max :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l1058_105877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_from_shadows_l1058_105801

/-- Parallel sun rays assumption -/
axiom ParallelSunRays : Prop

/-- Meter stick as line segment assumption -/
axiom StickAsLineSegment : Prop

/-- Predicate to check if a given radius satisfies the shadow conditions -/
def IsSphereRadius (r sphere_shadow stick_height stick_shadow : ℝ) : Prop :=
  (r / sphere_shadow) = (stick_height / stick_shadow)

/-- The radius of a sphere given its shadow length and the shadow of a meter stick. -/
theorem sphere_radius_from_shadows
  (sphere_shadow : ℝ)  -- Length of sphere's shadow
  (stick_height : ℝ)   -- Height of the meter stick
  (stick_shadow : ℝ)   -- Length of meter stick's shadow
  (h_sphere : sphere_shadow = 12)
  (h_stick_height : stick_height = 1.5)
  (h_stick_shadow : stick_shadow = 3)
  (h_parallel : ParallelSunRays)  -- Assumption of parallel sun rays
  (h_stick : StickAsLineSegment)  -- Assumption that the meter stick behaves as a line segment
  : ∃ (r : ℝ), r = 6 ∧ IsSphereRadius r sphere_shadow stick_height stick_shadow := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_from_shadows_l1058_105801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_length_of_T_l1058_105825

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (abs (abs (p.1 - 3) - 2) + abs (abs (p.2 - 3) - 2)) = 2}

-- Define the total length of lines in T
noncomputable def totalLength (S : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem total_length_of_T : totalLength T = 16 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_length_of_T_l1058_105825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSum_eq_quarter_l1058_105800

/-- The sum of the series (3^n) / (1 + 3^n + 3^(n+1) + 3^(2n+1)) from n=1 to infinity -/
noncomputable def infiniteSum : ℝ := ∑' n, (3^n : ℝ) / (1 + 3^n + 3^(n+1) + 3^(2*n+1))

/-- Theorem stating that the infinite sum equals 1/4 -/
theorem infiniteSum_eq_quarter : infiniteSum = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSum_eq_quarter_l1058_105800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_radius_for_area_l1058_105839

-- Define the minimum area required
def min_area : ℝ := 500

-- Define the function to calculate the area of a circle
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

-- Define the theorem
theorem min_radius_for_area (r : ℝ) : 
  (∀ r' : ℝ, circle_area r' ≥ min_area → r ≤ r') → 
  circle_area r ≥ min_area → 
  r = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_radius_for_area_l1058_105839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_functions_l1058_105841

noncomputable def f (x : ℝ) := 2 * Real.sin x ^ 2
noncomputable def g (x : ℝ) := Real.sqrt 3 * Real.sin (2 * x)

theorem max_distance_between_functions : 
  ∃ (C : ℝ), C = 3 ∧ ∀ (a : ℝ), |f a - g a| ≤ C ∧ ∃ (b : ℝ), |f b - g b| = C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_functions_l1058_105841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_three_l1058_105835

/-- The sum of the series ∑(n=1 to ∞) (4n+2)/3^n is equal to 3 -/
theorem series_sum_equals_three :
  let series := fun n : ℕ => (4 * n.succ + 2) / (3 : ℝ) ^ n.succ
  ∑' n, series n = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_three_l1058_105835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_symmetry_max_ab_l1058_105871

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y + 4 = 0

-- Define the line
def line_equation (a b x y : ℝ) : Prop := 2*a*x - b*y - 2 = 0

-- State the theorem
theorem circle_line_symmetry_max_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, circle_equation x y → ∃ x' y', line_equation a b x' y' ∧ 
    ((x - x')^2 + (y - y')^2 = (x' - 1)^2 + (y' + 2)^2)) →
  (∃ max : ℝ, max = 1/4 ∧ ∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    (∀ x y : ℝ, circle_equation x y → ∃ x' y', line_equation a' b' x' y' ∧ 
      ((x - x')^2 + (y - y')^2 = (x' - 1)^2 + (y' + 2)^2)) →
    a' * b' ≤ max) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_symmetry_max_ab_l1058_105871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_ink_cost_l1058_105863

theorem yellow_ink_cost (initial_money : ℕ) (black_ink_cost : ℕ) (red_ink_cost : ℕ) 
  (black_ink_count : ℕ) (red_ink_count : ℕ) (yellow_ink_count : ℕ) (additional_money_needed : ℕ) : 
  initial_money = 50 →
  black_ink_cost = 11 →
  red_ink_cost = 15 →
  black_ink_count = 2 →
  red_ink_count = 3 →
  yellow_ink_count = 2 →
  additional_money_needed = 43 →
  (initial_money + additional_money_needed - (black_ink_cost * black_ink_count + red_ink_cost * red_ink_count)) / yellow_ink_count = 13 := by
  intro h1 h2 h3 h4 h5 h6 h7
  -- Proof steps would go here
  sorry

#check yellow_ink_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_ink_cost_l1058_105863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_determinant_l1058_105805

def v : Fin 3 → ℝ := ![3, 2, -4]
def w : Fin 3 → ℝ := ![-1, 5, 2]

def u_direction : Fin 3 → ℝ := ![1, 1, 1]

theorem largest_determinant :
  ∃ (u : Fin 3 → ℝ),
    (∀ i, u i * u i = (1 : ℝ) / 3) ∧
    (∀ i, u i * u_direction i > 0) ∧
    Matrix.det (Matrix.of (λ i j => ![u, v, w] j i)) = 13 * Real.sqrt 3 ∧
    ∀ (u' : Fin 3 → ℝ),
      (∀ i, u' i * u' i = (1 : ℝ) / 3) →
      (∀ i, u' i * u_direction i > 0) →
      Matrix.det (Matrix.of (λ i j => ![u', v, w] j i)) ≤ 13 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_determinant_l1058_105805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_locations_distance_between_locations_proof_l1058_105830

-- Define the given information
noncomputable def fast_train_speed : ℝ := 33
noncomputable def meeting_point_ratio : ℝ := 4/7
noncomputable def slow_train_total_time : ℝ := 8

-- Define the theorem
theorem distance_between_locations : ℝ := by
  let remaining_distance_ratio := 1 - meeting_point_ratio
  let slow_train_remaining_time := remaining_distance_ratio * slow_train_total_time
  let fast_train_distance := slow_train_remaining_time * fast_train_speed
  let total_distance := fast_train_distance / meeting_point_ratio
  exact 198

-- Proof of the theorem
theorem distance_between_locations_proof :
  distance_between_locations = 198 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_locations_distance_between_locations_proof_l1058_105830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_for_room_l1058_105887

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℚ
  width : ℚ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℚ := d.length * d.width

/-- Converts inches to feet -/
def inchesToFeet (inches : ℚ) : ℚ := inches / 12

/-- Calculates the number of tiles needed to cover a room -/
def tilesNeeded (room : Dimensions) (tile : Dimensions) : ℕ :=
  Nat.ceil (area room / area tile)

theorem tiles_for_room : 
  let room : Dimensions := ⟨15, 18⟩
  let tile : Dimensions := ⟨inchesToFeet 3, inchesToFeet 9⟩
  tilesNeeded room tile = 1440 := by
  sorry

#eval tilesNeeded ⟨15, 18⟩ ⟨inchesToFeet 3, inchesToFeet 9⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_for_room_l1058_105887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l1058_105859

-- Define the power function f
noncomputable def f (m : ℕ+) (x : ℝ) : ℝ := x^(1 / (m^2 + m : ℝ))

-- Theorem statement
theorem power_function_properties (m : ℕ+) :
  (f m 2 = Real.sqrt 2) →
  (m = 1) ∧
  (∀ a : ℝ, f m (2 - a) > f m (a - 1) ↔ 1 ≤ a ∧ a < 3/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l1058_105859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_satisfies_conditions_l1058_105815

theorem matrix_N_satisfies_conditions : ∃ (N : Matrix (Fin 2) (Fin 2) ℚ),
  N = !![47/9, -16/9; -10/3, 14/3] ∧
  N.mulVec ![4, 1] = ![12, 10] ∧
  N.mulVec ![1, -2] = ![7, -8] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_satisfies_conditions_l1058_105815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_students_count_l1058_105872

/-- Represents the number of students in a class -/
def ClassSize := Nat

/-- Represents the number of students in an overlap between classes -/
def OverlapSize := Nat

/-- Calculate the total number of unique students taking the Math Olympiad -/
def totalUniqueStudents (riemannSize lovelaceSize eulerSize lovelaceEulerOverlap : Nat) : Nat :=
  riemannSize + lovelaceSize + eulerSize - lovelaceEulerOverlap

/-- Theorem stating that the total number of unique students taking the Math Olympiad is 27 -/
theorem unique_students_count :
  totalUniqueStudents 12 10 8 3 = 27 := by
  rfl

#eval totalUniqueStudents 12 10 8 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_students_count_l1058_105872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_specific_l1058_105824

/-- Calculate the distance between two points in a 2D plane -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Calculate the perimeter of a triangle given the coordinates of its vertices -/
noncomputable def triangle_perimeter (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  distance x1 y1 x2 y2 + distance x2 y2 x3 y3 + distance x3 y3 x1 y1

/-- Theorem: The perimeter of the triangle formed by (3,7), (5,2), and (0,0) is 2√29 + √58 -/
theorem triangle_perimeter_specific : 
  triangle_perimeter 3 7 5 2 0 0 = 2 * Real.sqrt 29 + Real.sqrt 58 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_specific_l1058_105824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_MN_angle_l1058_105827

/-- An isosceles right triangle with points M and N on its hypotenuse -/
structure IsoscelesRightTriangleMN where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  M : ℝ × ℝ
  N : ℝ × ℝ
  isIsoscelesRight : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2 ∧
                     (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  MNOnHypotenuse : ∃ t s : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 0 ≤ s ∧ s ≤ 1 ∧
    M = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2) ∧
    N = (s * B.1 + (1 - s) * C.1, s * B.2 + (1 - s) * C.2)
  hypotenuseProp : (B.1 - M.1)^2 + (B.2 - M.2)^2 - ((M.1 - N.1)^2 + (M.2 - N.2)^2) + (N.1 - C.1)^2 + (N.2 - C.2)^2 = 0

/-- The angle MAN is 45° -/
def angleMAN45 (triangle : IsoscelesRightTriangleMN) : Prop :=
  let AM := (triangle.M.1 - triangle.A.1, triangle.M.2 - triangle.A.2)
  let AN := (triangle.N.1 - triangle.A.1, triangle.N.2 - triangle.A.2)
  (AM.1 * AN.1 + AM.2 * AN.2)^2 = (AM.1^2 + AM.2^2) * (AN.1^2 + AN.2^2) / 2

/-- Main theorem: If M and N on the hypotenuse of an isosceles right triangle satisfy the given condition, then angle MAN is 45° -/
theorem isosceles_right_triangle_MN_angle (triangle : IsoscelesRightTriangleMN) :
  angleMAN45 triangle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_MN_angle_l1058_105827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_n_with_gcd_m_l1058_105818

theorem infinitely_many_n_with_gcd_m (D m : ℕ+) (h : ¬ ∃ k : ℕ, D = k^2) :
  ∃ f : ℕ → ℕ+, Set.Infinite {n : ℕ+ | Nat.gcd n.val (Int.floor (Real.sqrt ↑D.val * ↑n.val)).toNat = m} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_n_with_gcd_m_l1058_105818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_through_M_l1058_105810

/-- The circle's equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x - 2*y + 10 = 0

/-- The point M -/
def point_M : ℝ × ℝ := (3, 0)

/-- The line equation -/
def line_equation (x y : ℝ) : Prop :=
  x + y - 3 = 0

/-- Theorem stating that the line contains the shortest chord through M -/
theorem shortest_chord_through_M :
  ∃ (chord_length : ℝ),
    ∀ (x y other_x other_y : ℝ),
      circle_equation x y →
      line_equation x y →
      circle_equation other_x other_y →
      (other_x, other_y) ≠ point_M →
      let other_chord_length := Real.sqrt ((other_x - point_M.1)^2 + (other_y - point_M.2)^2)
      other_chord_length ≥ chord_length :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_through_M_l1058_105810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l1058_105868

def is_odd_multiple_of_7877 (a : ℤ) : Prop := ∃ k : ℤ, a = (2*k + 1) * 7877

theorem gcd_problem (a : ℤ) (h : is_odd_multiple_of_7877 a) :
  Int.gcd (7 * a^2 + 54 * a + 117) (3 * a + 10) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l1058_105868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_c_value_l1058_105879

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The slope of the first line y = -3x + 4 -/
noncomputable def m₁ : ℝ := -3

/-- The slope of the second line 9y - cx = 18 in terms of c -/
noncomputable def m₂ (c : ℝ) : ℝ := c / 9

/-- The value of c that makes the lines perpendicular -/
noncomputable def c_perpendicular : ℝ := 3

theorem perpendicular_lines_c_value :
  ∃! c : ℝ, perpendicular m₁ (m₂ c) ∧ c = c_perpendicular := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_c_value_l1058_105879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_point_x_coordinate_l1058_105813

/-- Given a line passing through points (1,3,2) and (6,0,2), 
    prove that the x-coordinate of the point on this line with y-coordinate 1 is 13/3 -/
theorem line_point_x_coordinate : 
  let p₁ : Fin 3 → ℝ := ![1, 3, 2]
  let p₂ : Fin 3 → ℝ := ![6, 0, 2]
  let line := {p : Fin 3 → ℝ | ∃ t : ℝ, ∀ i, p i = p₁ i + t * (p₂ i - p₁ i)}
  let point_on_line := {p ∈ line | p 1 = 1}
  ∀ p ∈ point_on_line, p 0 = 13/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_point_x_coordinate_l1058_105813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_paint_time_l1058_105848

/-- Represents the time (in hours) it takes for a person to paint a house. -/
structure PaintTime where
  hours : ℝ
  hours_pos : hours > 0

/-- Represents the rate at which a person paints a house (portion of house per hour). -/
noncomputable def paintRate (t : PaintTime) : ℝ := 1 / t.hours

theorem john_paint_time 
  (sally : PaintTime)
  (together : PaintTime)
  (h_sally : sally.hours = 4)
  (h_together : together.hours = 2.4) :
  ∃ (john : PaintTime), john.hours = 6 ∧ 
    paintRate sally + paintRate john = paintRate together := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_paint_time_l1058_105848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_shape_divisible_l1058_105858

/-- A shape with a vertical line of symmetry -/
structure SymmetricShape where
  /-- The shape has a vertical line of symmetry through its center -/
  has_vertical_symmetry : Prop

/-- A cut dividing a shape into two parts -/
structure Cut (α : Type) where
  /-- The first part of the cut -/
  part1 : α
  /-- The second part of the cut -/
  part2 : α

/-- Two shapes are identical if they have the same shape and area -/
def identical_shapes (α : Type) (s1 s2 : α) : Prop := sorry

/-- Theorem stating that a symmetric shape can be divided into two identical parts -/
theorem symmetric_shape_divisible (shape : SymmetricShape) :
  ∃ (cut : Cut SymmetricShape), identical_shapes SymmetricShape cut.part1 cut.part2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_shape_divisible_l1058_105858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_activities_equally_popular_l1058_105876

/-- Represents the popularity of an extracurricular activity --/
structure ActivityPopularity where
  numerator : ℚ
  denominator : ℚ

/-- The four extracurricular activities and their popularity fractions --/
def drama : ActivityPopularity := ⟨8, 24⟩
def sports : ActivityPopularity := ⟨9, 27⟩
def art : ActivityPopularity := ⟨10, 30⟩
def music : ActivityPopularity := ⟨7, 21⟩

/-- Function to convert ActivityPopularity to a rational number --/
def toRational (a : ActivityPopularity) : ℚ :=
  a.numerator / a.denominator

/-- Theorem stating that all activities are equally popular --/
theorem all_activities_equally_popular :
  toRational drama = toRational sports ∧
  toRational sports = toRational art ∧
  toRational art = toRational music := by
  sorry

#eval toRational drama
#eval toRational sports
#eval toRational art
#eval toRational music

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_activities_equally_popular_l1058_105876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1058_105857

theorem lambda_range (a b c lambda : ℝ) (h1 : a > b) (h2 : b > c)
  (h3 : ∀ (a b c : ℝ), a > b → b > c → (1 / (a - b) + 1 / (b - c) + lambda / (c - a) > 0)) :
  lambda < 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1058_105857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_tangent_lines_through_M_l1058_105843

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

-- Define the line
def my_line (a x y : ℝ) : Prop := a*x - y + 4 = 0

-- Define the point M
def point_M : ℝ × ℝ := (3, 1)

-- Theorem 1: Line tangency condition
theorem line_tangent_to_circle :
  ∀ a : ℝ, (∃ x y : ℝ, my_line a x y ∧ my_circle x y ∧
    ∀ x' y' : ℝ, my_line a x' y' → my_circle x' y' → (x', y') = (x, y)) ↔
  (a = 0 ∨ a = 4/3) :=
sorry

-- Theorem 2: Tangent lines through point M
theorem tangent_lines_through_M :
  ∀ x y : ℝ,
    (my_circle x y ∧ (∃ t : ℝ, x = point_M.1 + t ∧ y = point_M.2 + t) ∧
     (∀ x' y' : ℝ, my_circle x' y' → (∃ t' : ℝ, x' = point_M.1 + t' ∧ y' = point_M.2 + t') →
       (x', y') = (x, y))) ↔
    (x = 3 ∨ 3*x - 4*y - 5 = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_tangent_lines_through_M_l1058_105843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_Q_l1058_105812

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 24

-- Define the line l
def line_l (x y : ℝ) : Prop := x/12 + y/8 = 1

-- Define a point P on the line l
structure Point_P where
  x : ℝ
  y : ℝ
  on_line : line_l x y

-- Define point R as the intersection of OP and the circle
structure Point_R (P : Point_P) where
  x : ℝ
  y : ℝ
  on_circle : circle_C x y
  on_line_OP : ∃ t : ℝ, x = t * P.x ∧ y = t * P.y

-- Define point Q on OP satisfying the given condition
structure Point_Q (P : Point_P) (R : Point_R P) where
  x : ℝ
  y : ℝ
  on_line_OP : ∃ t : ℝ, x = t * P.x ∧ y = t * P.y
  condition : (x^2 + y^2) * (P.x^2 + P.y^2) = (R.x^2 + R.y^2)^2

-- Theorem statement
theorem locus_of_Q (P : Point_P) (R : Point_R P) (Q : Point_Q P R) : 
  (Q.x - 1)^2 + (Q.y - 3/2)^2 = 13/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_Q_l1058_105812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_imply_m_l1058_105867

/-- Given vectors OA, OB, and OC in ℝ², if AB is parallel to AC, then m = -1 --/
theorem parallel_vectors_imply_m (OA OB OC : ℝ × ℝ) (m : ℝ) :
  OA = (0, 1) →
  OB = (1, 3) →
  OC = (m, m) →
  ∃ (k : ℝ), (OB.1 - OA.1, OB.2 - OA.2) = k • (OC.1 - OA.1, OC.2 - OA.2) →
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_imply_m_l1058_105867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_T_l1058_105891

/-- A list of 25 distinct prime numbers, each not exceeding 2004 -/
def primes : List Nat := sorry

/-- Assertion that the list contains exactly 25 elements -/
axiom primes_length : primes.length = 25

/-- Assertion that all elements in the list are prime -/
axiom all_prime : ∀ p ∈ primes, Nat.Prime p

/-- Assertion that all elements in the list are distinct -/
axiom all_distinct : ∀ i j, i ≠ j → primes.get? i ≠ primes.get? j

/-- Assertion that all elements in the list do not exceed 2004 -/
axiom all_le_2004 : ∀ p ∈ primes, p ≤ 2004

/-- Definition of T as the product of (p^2005 - 1) / (p - 1) for all p in primes -/
noncomputable def T : Nat :=
  primes.foldl (fun acc p => acc * ((p^2005 - 1) / (p - 1))) 1

/-- Theorem stating that T is the largest integer such that any integer whose digits sum to T
    can be represented as the sum of distinct positive divisors of (p₁ * p₂ * ... * p₂₅)² -/
theorem largest_T : True := by sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_T_l1058_105891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1058_105826

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The domain of a function g -/
def Domain (g : ℝ → ℝ) : Set ℝ :=
  {x | ∃ y, g x = y}

/-- Given f(x) = x^2 + (2a-1)x + b is an even function,
    the domain of g(x) = sqrt(log_a(x) - 1) is (0, 1/2] -/
theorem domain_of_g (a b : ℝ) :
  let f := fun x => x^2 + (2*a - 1)*x + b
  let g := fun x => Real.sqrt (Real.log x / Real.log a - 1)
  IsEven f →
  Domain g = Set.Ioo 0 (1/2) ∪ {1/2} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1058_105826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_union_B_range_of_m_l1058_105816

-- Define the sets A, B, and C
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x^2 - 5*x - 14)}
def B : Set ℝ := {x | ∃ y, y = Real.log (-x^2 - 7*x - 12)}
def C (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Theorem for the complement of A ∪ B
theorem complement_A_union_B : 
  (A ∪ B)ᶜ = Set.Ioc (-2) 7 := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) : 
  A ∪ C m = A → m < 2 ∨ m ≥ 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_union_B_range_of_m_l1058_105816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_divisibility_properties_l1058_105884

/-- Given a function f(x) = (x + b)^2 + c, where b and c are integers,
    and prime numbers p and q satisfying certain conditions,
    prove two divisibility properties of f. -/
theorem function_divisibility_properties
  (b c : ℤ)
  (p q : ℕ)
  (f : ℤ → ℤ)
  (hp : Nat.Prime p)
  (hq : Nat.Prime q)
  (hqodd : q % 2 = 1)
  (hpc : (p : ℤ) ∣ c)
  (hp2c : ¬((p^2 : ℤ) ∣ c))
  (hqc : ¬((q : ℤ) ∣ c))
  (hf : ∀ x, f x = (x + b)^2 + c)
  (hqfn : ∃ n, (q : ℤ) ∣ f n) :
  (∀ n, ¬((p^2 : ℤ) ∣ f n)) ∧
  (∀ r, ∃ N, (q^r : ℤ) ∣ f N) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_divisibility_properties_l1058_105884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_difference_hundreds_digit_l1058_105845

-- Define a function to get the hundreds digit of a natural number
def hundreds_digit (n : ℕ) : ℕ :=
  (n / 100) % 10

-- State the theorem
theorem factorial_difference_hundreds_digit :
  hundreds_digit (Nat.factorial 17 - Nat.factorial 13) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_difference_hundreds_digit_l1058_105845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_angle_l1058_105873

/-- Given a circle with an inscribed triangle with sides 5√3, 10√3, and 15,
    the angle subtended at the centre of the circle by the shortest side is 60°. -/
theorem inscribed_triangle_angle (circle : Real → Real → Prop) 
                                 (triangle : Real → Real → Real → Prop) 
                                 (inscribedIn : (Real → Real → Real → Prop) → (Real → Real → Prop) → Prop)
                                 (sides : (Real → Real → Real → Prop) → List Real)
                                 (angleSubtendedByShortestSide : (Real → Real → Prop) → (Real → Real → Real → Prop) → Real) :
  inscribedIn triangle circle →
  sides triangle = [5 * Real.sqrt 3, 10 * Real.sqrt 3, 15] →
  angleSubtendedByShortestSide circle triangle = 60 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_angle_l1058_105873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_range_of_a_l1058_105821

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (2 - a/2)*x + 2

theorem increasing_f_range_of_a :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) → a ∈ Set.Icc (8/3) 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_range_of_a_l1058_105821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1058_105811

/-- An ellipse in a 2D plane -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Vector between two points -/
def vector (p q : Point) : Point :=
  ⟨q.x - p.x, q.y - p.y⟩

/-- Dot product of two vectors -/
def dot_product (v w : Point) : ℝ :=
  v.x * w.x + v.y * w.y

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((q.x - p.x)^2 + (q.y - p.y)^2)

/-- Eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- Main theorem -/
theorem ellipse_theorem (e : Ellipse) 
  (A B F P Q : Point)
  (h_A : A.x = e.a ∧ A.y = 0)
  (h_B : B.x = 0 ∧ B.y = e.b)
  (h_P : (P.x / e.a)^2 + (P.y / e.b)^2 = 1)
  (h_Q : Q.x = -P.x ∧ Q.y = -P.y)
  (h_condition : dot_product (vector F P) (vector F Q) + 
                 dot_product (vector F A) (vector F B) = 
                 distance A B ^ 2) :
  (F.x = e.a * eccentricity e ∧ F.y = 0) ∧
  (3/4 ≤ eccentricity e ∧ eccentricity e ≤ (Real.sqrt 37 - 1) / 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1058_105811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_l1058_105856

-- Define the circle C
noncomputable def circle_C (φ : Real) : Real × Real := (1 + Real.cos φ, Real.sin φ)

-- Define the polar equation of line l
def line_l (ρ θ : Real) : Prop := 2 * ρ * Real.sin (θ + Real.pi/3) = 6 * Real.sqrt 3

-- Define the ray OM
def ray_OM (θ : Real) : Prop := θ = Real.pi/6

-- Define the intersection point P on circle C
noncomputable def point_P : Real × Real := circle_C (Real.pi/6)

-- Define the intersection point Q on line l
noncomputable def point_Q : Real × Real := 
  (3 * Real.sqrt 3 * Real.cos (Real.pi/6), 3 * Real.sqrt 3 * Real.sin (Real.pi/6))

-- State the theorem
theorem length_PQ : Real.sqrt ((point_P.1 - point_Q.1)^2 + (point_P.2 - point_Q.2)^2) = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_l1058_105856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_graph_f2_f3_different_f1_l1058_105834

noncomputable def f₁ (x : ℝ) : ℝ := x^2 - 2
noncomputable def f₂ (x : ℝ) : ℝ := (x^3 - 8) / (x - 2)
noncomputable def f₃ (x : ℝ) : ℝ := (x^3 - 8) / (x - 2)

theorem same_graph_f2_f3_different_f1 :
  (∀ x : ℝ, x ≠ 2 → f₂ x = f₃ x) ∧
  (∃ x : ℝ, f₁ x ≠ f₂ x) ∧
  (∃ x : ℝ, f₁ x ≠ f₃ x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_graph_f2_f3_different_f1_l1058_105834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_area_is_13pi_over_8_l1058_105838

/-- A semicircle circumscribes a 1 × 3 rectangle with the longer side on the diameter. -/
def SemicircleWithRectangle :=
  {r : ℝ // ∃ (d : ℝ), d^2 = 13 ∧ r = d/2}

/-- The area of the semicircle described above. -/
noncomputable def semicircleArea (s : SemicircleWithRectangle) : ℝ :=
  Real.pi * s.val^2 / 2

/-- Theorem: The area of the semicircle is 13π/8. -/
theorem semicircle_area_is_13pi_over_8 (s : SemicircleWithRectangle) :
  semicircleArea s = 13 * Real.pi / 8 := by
  sorry

#check semicircle_area_is_13pi_over_8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_area_is_13pi_over_8_l1058_105838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_fourth_quadrant_l1058_105837

-- Define what it means for an angle to be in the second quadrant
def in_second_quadrant (α : Real) : Prop :=
  Real.pi / 2 < α ∧ α < Real.pi

-- Define what it means for a point to be in the fourth quadrant
def in_fourth_quadrant (x y : Real) : Prop :=
  x > 0 ∧ y < 0

-- Theorem statement
theorem point_in_fourth_quadrant (α : Real) :
  in_second_quadrant α → in_fourth_quadrant (Real.sin α) (Real.cos α) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_fourth_quadrant_l1058_105837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quad_problem_l1058_105855

-- Define the quadrilateral PQRS
structure Quadrilateral :=
  (P Q R S : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_convex (quad : Quadrilateral) : Prop := sorry

def is_perpendicular (A B C D : ℝ × ℝ) : Prop := sorry

def distance (A B : ℝ × ℝ) : ℝ := sorry

-- Define the theorem
theorem quad_problem (PQRS : Quadrilateral) :
  is_convex PQRS →
  is_perpendicular PQRS.R PQRS.S PQRS.P PQRS.Q →
  is_perpendicular PQRS.P PQRS.Q PQRS.S PQRS.R →
  distance PQRS.R PQRS.S = 52 →
  distance PQRS.P PQRS.Q = 34 →
  ∃ T : ℝ × ℝ,
    is_perpendicular PQRS.Q T PQRS.P PQRS.S ∧
    (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ T = (1 - t) • PQRS.P + t • PQRS.Q) ∧
    distance PQRS.P T = 9 →
  distance PQRS.Q T = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quad_problem_l1058_105855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_force_length_l1058_105875

/-- Represents a lever with a fulcrum at one end -/
structure Lever where
  length : ℝ
  weightPerMeter : ℝ
  objectWeight : ℝ
  objectDistance : ℝ

/-- Calculates the moment caused by the lever's own weight -/
noncomputable def leverMoment (l : Lever) : ℝ := l.length^2 * l.weightPerMeter / 2

/-- Calculates the moment caused by the object -/
noncomputable def objectMoment (l : Lever) : ℝ := l.objectWeight * l.objectDistance

/-- The condition for minimum force is when the lever's moment equals the object's moment -/
def isMinimumForceLength (l : Lever) : Prop := leverMoment l = objectMoment l

/-- The theorem stating that 7 meters is the length that minimizes the required force -/
theorem minimum_force_length :
  ∃ (l : Lever), l.length = 7 ∧ l.weightPerMeter = 2 ∧ l.objectWeight = 49 ∧ l.objectDistance = 1 ∧
  isMinimumForceLength l := by
  sorry

#check minimum_force_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_force_length_l1058_105875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_min_value_l1058_105832

noncomputable def power_function (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

theorem power_function_min_value (α : ℝ) (h : power_function α (-2) = 4) :
  ∃ x₀, ∀ x, power_function α x ≥ power_function α x₀ := by
  -- We know that α = 2 from the given condition
  have α_eq_two : α = 2 := by
    -- Proof that α = 2
    sorry
  
  -- Now we can show that x₀ = 0 is the minimum point
  use 0
  intro x
  -- Proof that for all x, x^2 ≥ 0
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_min_value_l1058_105832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_c_value_l1058_105878

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 : ℝ} : 
  (∃ b1 b2 : ℝ, ∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) → m1 = m2

/-- The slope-intercept form of a line -/
def line_equation (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

theorem parallel_lines_c_value :
  (∃ c : ℝ, 
    (∀ x y : ℝ, line_equation 3 c x y ↔ 3 * y - 3 * c = 9 * x) ∧ 
    (∀ x y : ℝ, line_equation (c - 3) 2 x y ↔ y - 2 = (c - 3) * x) ∧
    (∃ b1 b2 : ℝ, ∀ x y : ℝ, y = 3 * x + b1 ↔ y = (c - 3) * x + b2)) →
  ∃ c : ℝ, c = 6 := by
  intro h
  rcases h with ⟨c, h1, h2, h3⟩
  use c
  have : 3 = c - 3 := parallel_lines_equal_slopes h3
  linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_c_value_l1058_105878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_has_eight_children_l1058_105862

/-- Represents the number of children Max has -/
def max_children : ℕ := sorry

/-- Represents the total number of Max's grandchildren -/
def total_grandchildren : ℕ := sorry

/-- The total number of Max's grandchildren is 58 -/
axiom grandchildren_count : total_grandchildren = 58

/-- The number of Max's grandchildren is equal to the number of children
    Max has multiplied by (Max's children - 2), plus 10 (for the two children with 5 kids each) -/
axiom grandchildren_equation : total_grandchildren = max_children * (max_children - 2) + 10

/-- Theorem: Max has 8 children -/
theorem max_has_eight_children : max_children = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_has_eight_children_l1058_105862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_same_foci_hyperbola_ellipse_l1058_105894

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Define the ellipse
def is_ellipse (A B : ℝ × ℝ) (k : ℝ) (P : ℝ × ℝ) : Prop :=
  distance A P + distance B P = k

-- Define the hyperbola
def is_hyperbola (a b : ℝ) (P : ℝ × ℝ) : Prop :=
  P.1^2 / a^2 - P.2^2 / b^2 = 1

-- Define the ellipse equation
def is_ellipse_eq (a b : ℝ) (P : ℝ × ℝ) : Prop :=
  P.1^2 / a^2 + P.2^2 / b^2 = 1

-- Theorem for proposition 2
theorem max_distance_ellipse (A B : ℝ × ℝ) :
  (∀ P : ℝ × ℝ, distance A P = 10 - distance B P) →
  distance A B = 8 →
  (∃ P : ℝ × ℝ, distance A P = 9) ∧ 
  (∀ P : ℝ × ℝ, distance A P ≤ 9) :=
by sorry

-- Theorem for proposition 4
theorem same_foci_hyperbola_ellipse :
  ∃ c : ℝ, 
    (∀ P : ℝ × ℝ, is_hyperbola 4 (Real.sqrt 10) P ↔ distance (c, 0) P - distance (-c, 0) P = 8) ∧
    (∀ P : ℝ × ℝ, is_ellipse_eq (Real.sqrt 30) 2 P ↔ distance (c, 0) P + distance (-c, 0) P = 2 * Real.sqrt 30) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_same_foci_hyperbola_ellipse_l1058_105894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heron_formula_example_l1058_105851

/-- Heron's formula for the area of a triangle -/
noncomputable def heronFormula (a b c : ℝ) : ℝ :=
  let p := (a + b + c) / 2
  Real.sqrt (p * (p - a) * (p - b) * (p - c))

/-- The semi-perimeter of a triangle -/
noncomputable def semiPerimeter (a b c : ℝ) : ℝ :=
  (a + b + c) / 2

theorem heron_formula_example :
  let a : ℝ := 2
  let b : ℝ := 3
  let c : ℝ := 4
  semiPerimeter a b c = 9 / 2 ∧
  heronFormula a b c = 3 * Real.sqrt 15 / 4 := by
  sorry

#check heron_formula_example

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heron_formula_example_l1058_105851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_volume_l1058_105814

/-- A parallelepiped with square bases and rhombi lateral faces -/
structure Parallelepiped where
  /-- The side length of the square bases -/
  b : ℝ
  /-- The bases are squares -/
  square_bases : b > 0
  /-- All lateral faces are rhombi -/
  rhombi_lateral_faces : True
  /-- One vertex of the upper base is equidistant from all vertices of the lower base -/
  equidistant_vertex : True

/-- The volume of the parallelepiped -/
noncomputable def volume (p : Parallelepiped) : ℝ := p.b^3 / Real.sqrt 2

/-- Theorem: The volume of the parallelepiped is b^3 / √2 -/
theorem parallelepiped_volume (p : Parallelepiped) : volume p = p.b^3 / Real.sqrt 2 := by
  -- Proof goes here
  sorry

#check parallelepiped_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_volume_l1058_105814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_cube_with_specific_digits_l1058_105864

theorem unique_cube_with_specific_digits : ∃! n : ℕ, 
  (40000 ≤ n^3 ∧ n^3 < 50000) ∧ 
  (n^3 % 10 = 6) ∧ 
  (n = 36) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_cube_with_specific_digits_l1058_105864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_formula_l1058_105820

def mySequence (n : ℕ) : ℤ :=
  match n with
  | 0 => 1
  | 1 => 5
  | n + 2 => 2 * mySequence (n + 1) - mySequence n

theorem mySequence_formula (n : ℕ) : mySequence n = 4 * n.succ - 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_formula_l1058_105820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ratio_smallest_to_largest_l1058_105847

noncomputable def original_width : ℝ := 8
noncomputable def original_height : ℝ := 6
noncomputable def folded_height : ℝ := original_height / 2
noncomputable def cut_1 : ℝ := 3
noncomputable def cut_2 : ℝ := 5

noncomputable def small_rect_width : ℝ := cut_1
noncomputable def small_rect_height : ℝ := folded_height
noncomputable def medium_rect_width : ℝ := cut_2 - cut_1
noncomputable def medium_rect_height : ℝ := folded_height
noncomputable def large_rect_width : ℝ := original_width - cut_2
noncomputable def large_rect_height : ℝ := folded_height

def perimeter (width : ℝ) (height : ℝ) : ℝ := 2 * (width + height)

theorem perimeter_ratio_smallest_to_largest :
  let small_perimeter := perimeter small_rect_width small_rect_height
  let medium_perimeter := perimeter medium_rect_width medium_rect_height
  let large_perimeter := perimeter large_rect_width large_rect_height
  let smallest_perimeter := min small_perimeter (min medium_perimeter large_perimeter)
  let largest_perimeter := max small_perimeter (max medium_perimeter large_perimeter)
  smallest_perimeter / largest_perimeter = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ratio_smallest_to_largest_l1058_105847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_hourly_wage_l1058_105880

/-- Calculates the hourly wage of employees in a t-shirt factory --/
theorem employee_hourly_wage (num_employees : ℕ) (shirts_per_employee : ℕ) (shift_hours : ℕ)
  (shirt_bonus : ℚ) (shirt_price : ℚ) (nonemployee_expenses : ℚ) (daily_profit : ℚ) :
  num_employees = 20 →
  shirts_per_employee = 20 →
  shift_hours = 8 →
  shirt_bonus = 5 →
  shirt_price = 35 →
  nonemployee_expenses = 1000 →
  daily_profit = 9080 →
  (num_employees * shirts_per_employee * shirt_price - 
   (num_employees * shirts_per_employee * shirt_bonus + nonemployee_expenses) - 
   daily_profit) / (num_employees * shift_hours) = 12 := by
  intro h1 h2 h3 h4 h5 h6 h7
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_hourly_wage_l1058_105880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_points_in_rectangle_l1058_105808

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Checks if a point is inside a rectangle -/
def isInside (p : Point) (r : Rectangle) : Prop :=
  0 ≤ p.x ∧ p.x ≤ r.width ∧ 0 ≤ p.y ∧ p.y ≤ r.height

theorem seven_points_in_rectangle (r : Rectangle) (points : Finset Point) :
  r.width = 3 ∧ r.height = 4 →
  points.card = 7 →
  (∀ p ∈ points, isInside p r) →
  ∃ p1 p2, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ distance p1 p2 ≤ Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_points_in_rectangle_l1058_105808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_intercept_sum_l1058_105882

/-- Given points A, B, C, and D where D is one-third of the way from A to B,
    prove that the sum of the slope and y-intercept of line CD is 12/5 -/
theorem slope_intercept_sum (A B C D : ℝ × ℝ) : 
  A = (0, 8) →
  B = (0, 0) →
  C = (10, 0) →
  D.1 = (1/3 * A.1 + 2/3 * B.1) →
  D.2 = (1/3 * A.2 + 2/3 * B.2) →
  (D.2 - C.2) / (D.1 - C.1) + D.2 = 12/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_intercept_sum_l1058_105882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1058_105870

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * b + b * c + c * a = 1) :
  (((1 / a) + 6 * b) ^ (1/3)) + (((1 / b) + 6 * c) ^ (1/3)) + (((1 / c) + 6 * a) ^ (1/3)) ≤ 1 / (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1058_105870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_l1058_105869

noncomputable def A : ℝ × ℝ := (2, 5)
noncomputable def B : ℝ × ℝ := (4, 3)

noncomputable def angle_of_inclination (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.arctan ((p2.2 - p1.2) / (p2.1 - p1.1))

theorem line_inclination :
  angle_of_inclination A B = 3 * π / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_l1058_105869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_reverse_digit_difference_l1058_105874

/-- Represents a three-digit positive integer -/
def ThreeDigitInt := {n : ℕ | 100 ≤ n ∧ n < 1000}

/-- Returns true if two numbers have the same digits in reverse order -/
def sameDigitsReversed (a b : ThreeDigitInt) : Prop :=
  ∃ (x y z : ℕ), x < 10 ∧ y < 10 ∧ z < 10 ∧ 
    (↑a : ℕ) = 100 * x + 10 * y + z ∧
    (↑b : ℕ) = 100 * z + 10 * y + x

theorem greatest_reverse_digit_difference (q r : ThreeDigitInt) :
  sameDigitsReversed q r →
  ∃ (p : ℕ), Nat.Prime p ∧ (↑q - ↑r : ℤ).natAbs % p = 0 →
  (↑q - ↑r : ℤ).natAbs < 300 →
  ∀ (q' r' : ThreeDigitInt), sameDigitsReversed q' r' →
    ∃ (p' : ℕ), Nat.Prime p' ∧ (↑q' - ↑r' : ℤ).natAbs % p' = 0 →
    (↑q' - ↑r' : ℤ).natAbs < 300 →
    (↑q - ↑r : ℤ).natAbs ≥ (↑q' - ↑r' : ℤ).natAbs →
  (↑q - ↑r : ℤ).natAbs = 297 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_reverse_digit_difference_l1058_105874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_events_independent_prob_one_passing_l1058_105807

-- Define the sample space for two dice
def Ω : Type := Fin 6 × Fin 6

-- Define event A: first die shows an odd number
def A : Set Ω := {ω | Odd ω.fst.val}

-- Define event B: sum of two dice is a multiple of 3
def B : Set Ω := {ω | (ω.fst.val + ω.snd.val) % 3 = 0}

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- Define the probability of an athlete hitting the target
def p_hit_A : ℝ := 0.7
def p_hit_B : ℝ := 0.6

-- Define the event of an athlete passing the assessment
def athlete_pass (p : ℝ) : ℝ := 1 - (1 - p)^2

-- Theorem 1: Independence of events A and B
theorem events_independent : P (A ∩ B) = P A * P B := by sorry

-- Theorem 2: Probability of exactly one athlete passing
theorem prob_one_passing : 
  athlete_pass p_hit_A * (1 - athlete_pass p_hit_B) + (1 - athlete_pass p_hit_A) * athlete_pass p_hit_B = 0.2212 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_events_independent_prob_one_passing_l1058_105807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_2_3_5_7_less_than_300_l1058_105802

theorem divisible_by_2_3_5_7_less_than_300 : 
  Finset.card (Finset.filter (λ n : ℕ => n > 0 ∧ n < 300 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n) (Finset.range 300)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_2_3_5_7_less_than_300_l1058_105802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_unique_pair_sums_l1058_105846

/-- A function representing the card values -/
def card_value (n : ℕ) : ℕ := 
  if n ≤ 10 then n else n + 1

/-- The theorem stating the maximum subset size with unique pair sums -/
theorem max_subset_unique_pair_sums :
  ∃ (S : Finset ℕ), 
    (∀ n, n ∈ S → n ∈ Finset.range 13) ∧ 
    (∀ a b c d, a ∈ S → b ∈ S → c ∈ S → d ∈ S → a < b → c < d → (a ≠ c ∨ b ≠ d) → 
      card_value a + card_value b ≠ card_value c + card_value d) ∧
    S.card = 6 ∧
    (∀ T : Finset ℕ, (∀ n, n ∈ T → n ∈ Finset.range 13) → 
      (∀ a b c d, a ∈ T → b ∈ T → c ∈ T → d ∈ T → a < b → c < d → (a ≠ c ∨ b ≠ d) → 
        card_value a + card_value b ≠ card_value c + card_value d) →
      T.card ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_unique_pair_sums_l1058_105846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_nonnegative_iff_l1058_105892

noncomputable def f (x : ℝ) : ℝ := (x - 12*x^2 + 36*x^3) / (9 - x^3)

theorem fraction_nonnegative_iff (x : ℝ) : f x ≥ 0 ↔ x ∈ Set.Ici 0 ∩ Set.Iio 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_nonnegative_iff_l1058_105892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_implies_sin_cos_value_l1058_105854

theorem root_implies_sin_cos_value (α : ℝ) (h : (2 + Real.sqrt 3)^2 - (Real.tan α + (1 / Real.tan α)) * (2 + Real.sqrt 3) + 1 = 0) : 
  Real.sin α * Real.cos α = 1/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_implies_sin_cos_value_l1058_105854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_even_indexed_binomial_coefficients_l1058_105890

open Nat BigOperators Finset

theorem sum_even_indexed_binomial_coefficients (n : ℕ) :
  ∑ k in range (n + 1), choose (2 * n) (2 * k) = 2^(2 * n - 1) - 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_even_indexed_binomial_coefficients_l1058_105890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ryegrass_in_Y_mixture_check_l1058_105853

/-- Represents a seed mixture -/
structure SeedMixture where
  ryegrass : ℚ  -- Percentage of ryegrass
  bluegrass : ℚ  -- Percentage of bluegrass
  fescue : ℚ  -- Percentage of fescue

/-- The properties of seed mixture X -/
def X : SeedMixture := { ryegrass := 40, bluegrass := 60, fescue := 0 }

/-- The properties of seed mixture Y -/
def Y : SeedMixture := { ryegrass := 25, bluegrass := 0, fescue := 75 }

/-- The proportion of X in the final mixture -/
def X_proportion : ℚ := 100 / 3

/-- The proportion of Y in the final mixture -/
def Y_proportion : ℚ := 200 / 3

/-- The percentage of ryegrass in the final mixture -/
def final_ryegrass : ℚ := 30

theorem ryegrass_in_Y : Y.ryegrass = 25 := by sorry

theorem mixture_check : 
  X_proportion * X.ryegrass / 100 + Y_proportion * Y.ryegrass / 100 = final_ryegrass := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ryegrass_in_Y_mixture_check_l1058_105853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_self_counting_lists_l1058_105861

def is_valid_list (list : List Nat) : Prop :=
  list.length = 4 ∧ list.all (λ x => x ≤ 3)

def counts_match_list (list : List Nat) : Prop :=
  list.length = 4 ∧
  list[0]! = list.count 0 ∧
  list[1]! = list.count 1 ∧
  list[2]! = list.count 2 ∧
  list[3]! = list.count 3

theorem unique_self_counting_lists :
  ∀ list : List Nat,
    is_valid_list list →
    counts_match_list list →
    (list = [1, 2, 1, 0] ∨ list = [2, 0, 2, 0]) :=
by sorry

#check unique_self_counting_lists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_self_counting_lists_l1058_105861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reach_food_probability_l1058_105809

/-- Represents a lily pad with its number -/
structure LilyPad where
  number : Nat

/-- Represents the frog's possible moves -/
inductive Move
  | Hop
  | Jump

/-- Represents the state of the game -/
structure GameState where
  currentPad : LilyPad
  predatorPads : List LilyPad
  foodPad : LilyPad
  totalPads : Nat

/-- Defines the probability of each move -/
def moveProbability : Move → Rat
  | Move.Hop => 1/2
  | Move.Jump => 1/2

/-- Defines the game setup -/
def initialState : GameState :=
  { currentPad := ⟨0⟩
  , predatorPads := [⟨3⟩, ⟨6⟩, ⟨9⟩]
  , foodPad := ⟨12⟩
  , totalPads := 14 }

/-- Defines a safe move that avoids predator pads -/
def isSafeMove (state : GameState) (move : Move) : Prop :=
  match move with
  | Move.Hop => state.currentPad.number + 1 ∉ state.predatorPads.map LilyPad.number
  | Move.Jump => state.currentPad.number + 2 ∉ state.predatorPads.map LilyPad.number

/-- Theorem stating the probability of reaching the food pad safely -/
theorem reach_food_probability :
  ∃ (path : List Move),
    (∀ (move : Move), move ∈ path → isSafeMove initialState move) ∧
    (path.foldl (λ acc _ => acc * (1/2)) 1 = 9/512) ∧
    (path.foldl (λ state move => 
      match move with
      | Move.Hop => { state with currentPad := ⟨state.currentPad.number + 1⟩ }
      | Move.Jump => { state with currentPad := ⟨state.currentPad.number + 2⟩ }
    ) initialState).currentPad = initialState.foodPad := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reach_food_probability_l1058_105809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_area_specific_l1058_105886

/-- The total area of four shaded regions formed by two circles -/
noncomputable def shaded_area (r₁ r₂ : ℝ) : ℝ :=
  let small_rectangle : ℝ := 2 * r₁ * r₁
  let small_semicircle : ℝ := (1/2) * Real.pi * r₁^2
  let large_rectangle : ℝ := 2 * r₂ * r₂
  let large_quarter_circle : ℝ := (1/4) * Real.pi * r₂^2
  (small_rectangle - small_semicircle) + (large_rectangle - large_quarter_circle)

/-- Theorem stating the total shaded area for specific circle radii -/
theorem total_shaded_area_specific : 
  shaded_area 3 5 = 68 - (14.25 : ℝ) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_area_specific_l1058_105886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l1058_105896

theorem trigonometric_problem (α β : Real) 
  (h1 : α ∈ Set.Ioo 0 (π/2))
  (h2 : β ∈ Set.Ioo 0 (π/2))
  (h3 : Real.cos α = 4/5)
  (h4 : Real.cos (α + β) = 3/5) :
  Real.sin β = 7/25 ∧ 2*α + β = π/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l1058_105896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_dot_product_l1058_105898

noncomputable def f (x : ℝ) : ℝ := (2*x + 3) / (2*x - 4)

def O : ℝ × ℝ := (0, 0)
def P : ℝ × ℝ := (2, 1)

theorem intersection_dot_product 
  (A B : ℝ × ℝ) -- A and B are the intersection points
  (h1 : ∃ (m b : ℝ), ∀ x, f x = m * x + b) -- f is a linear function
  (h2 : A.1 ≠ B.1) -- A and B are distinct points
  (h3 : f A.1 = A.2 ∧ f B.1 = B.2) -- A and B lie on the graph of f
  (h4 : ∃ (m b : ℝ), A.2 = m * A.1 + b ∧ B.2 = m * B.1 + b ∧ P.2 = m * P.1 + b) -- A, B, and P are collinear
  : (A.1 - O.1 + B.1 - O.1) * (P.1 - O.1) + (A.2 - O.2 + B.2 - O.2) * (P.2 - O.2) = 17/2 := by
  sorry

#check intersection_dot_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_dot_product_l1058_105898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circuit_max_voltage_l1058_105852

/-- Represents the circuit parameters and calculates the maximum voltage --/
noncomputable def MaxVoltage (C L₁ L₂ I_max : ℝ) : ℝ :=
  let I := L₂ * I_max / (L₁ + L₂)
  Real.sqrt ((L₂ * I_max^2 / C) - ((L₁ + L₂) * I^2 / C))

/-- Theorem stating that given the circuit conditions, the maximum voltage is approximately 2.89 V --/
theorem circuit_max_voltage :
  let C := 2e-6  -- 2 μF
  let L₁ := 2e-3 -- 2 mH
  let L₂ := 1e-3 -- 1 mH
  let I_max := 5e-3 -- 5 mA
  ∃ ε > 0, |MaxVoltage C L₁ L₂ I_max - 2.89| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circuit_max_voltage_l1058_105852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pretzel_puzzle_l1058_105889

theorem pretzel_puzzle : ∃ x : ℕ, 
  x > 0 ∧
  (let remaining1 := x / 2 - 1;
   let remaining2 := remaining1 / 2 - 1;
   let remaining3 := remaining2 / 2 - 1;
   remaining3 / 2 - 1 = 0) ∧
  x = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pretzel_puzzle_l1058_105889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_roots_of_quadratic_l1058_105897

theorem complex_roots_of_quadratic (k : ℂ) (h_k : k.re = 0) : 
  ∀ z, (2 * z^2 + 5 * Complex.I * z - k = 0) → z.re ≠ 0 ∧ z.im ≠ 0 :=
by
  sorry

#check complex_roots_of_quadratic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_roots_of_quadratic_l1058_105897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_equidistant_l1058_105829

def A : ℝ × ℝ × ℝ := (10, 0, 0)
def B : ℝ × ℝ × ℝ := (0, -6, 0)
def C : ℝ × ℝ × ℝ := (0, 0, 8)
def D : ℝ × ℝ × ℝ := (0, 0, 0)
def P : ℝ × ℝ × ℝ := (5, -3, 4)

noncomputable def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

theorem point_equidistant : 
  distance A P = distance B P ∧ 
  distance A P = distance C P ∧ 
  distance A P = distance D P := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_equidistant_l1058_105829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_composition_square_l1058_105849

/-- A polynomial is not a square if it cannot be expressed as p^2 for some polynomial p. -/
def IsNotSquare (p : Polynomial ℝ) : Prop :=
  ∀ q : Polynomial ℝ, p ≠ q^2

/-- A polynomial is non-constant if its degree is greater than 0. -/
def IsNonConstant (p : Polynomial ℝ) : Prop :=
  p.degree > 0

/-- Composition of two polynomials -/
noncomputable def Compose (p q : Polynomial ℝ) : Polynomial ℝ :=
  p.comp q

theorem polynomial_composition_square (f g : Polynomial ℝ) :
  IsNonConstant f → IsNonConstant g →
  IsNotSquare f → IsNotSquare g →
  (∃ h : Polynomial ℝ, Compose f g = h^2) →
  ¬∃ k : Polynomial ℝ, Compose g f = k^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_composition_square_l1058_105849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_f_inequality_l1058_105881

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := Real.exp x / x

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  ∃ m b : ℝ, 
    let P : ℝ × ℝ := (2, Real.exp 2 / 2)
    ∀ x y : ℝ,
      y - f P.1 = m * (x - P.1) →
      Real.exp 2 * x - 4 * y = 0 :=
sorry

-- Theorem for the inequality
theorem f_inequality :
  ∀ x : ℝ, x > 0 → f x > 2 * (x - Real.log x) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_f_inequality_l1058_105881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_eight_l1058_105865

theorem sum_of_solutions_is_eight :
  ∃ (S : Finset ℝ), (∀ x ∈ S, |x^2 - 16*x + 68| = 4) ∧ (S.sum id = 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_eight_l1058_105865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l1058_105836

noncomputable def num1 : ℝ := -8
noncomputable def num2 : ℝ := -Real.pi/3
noncomputable def num3 : ℝ := 22/7
noncomputable def num4 : ℝ := 0
noncomputable def num5 : ℝ := -19/9  -- equivalent to -2.11111...
noncomputable def num6 : ℝ := 10     -- simplified from -(-10)
noncomputable def num7 : ℝ := -8     -- simplified from -2^3
noncomputable def num8 : ℝ := 5/100  -- equivalent to 5%
noncomputable def num9 : ℝ := 2.323323332  -- approximation of the given number

def positive_set : Set ℝ := {num3, num6, num8, num9}
def fraction_set : Set ℝ := {num3, num5, num8}
def integer_set : Set ℝ := {num1, num4, num6, num7}
def irrational_set : Set ℝ := {num2, num9}

theorem number_categorization :
  (∀ x ∈ positive_set, x > 0) ∧
  (∀ x ∈ fraction_set, ∃ a b : ℤ, b ≠ 0 ∧ x = a / b) ∧
  (∀ x ∈ integer_set, ∃ n : ℤ, x = n) ∧
  (∀ x ∈ irrational_set, ¬∃ a b : ℤ, b ≠ 0 ∧ x = a / b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l1058_105836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1058_105831

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 6*x + 11) / Real.log (1/2)

theorem f_monotone_increasing :
  StrictMonoOn f (Set.Iio 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1058_105831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_count_is_fifty_l1058_105866

/-- Proves that the initial number of elements in a set is 50, given the specified conditions --/
theorem initial_count_is_fifty (S : Finset ℝ) (n : ℕ) : 
  (S.sum id) / n = 56 →
  (S.sum id - 45 - 55) / (n - 2) = 56.25 →
  n = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_count_is_fifty_l1058_105866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_property_dot_product_property_l1058_105823

noncomputable section

/-- Parabola definition -/
def Parabola (p : ℝ) := {P : ℝ × ℝ | P.2^2 = 2 * p * P.1 ∧ p > 0}

/-- Focus of the parabola -/
def Focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

/-- Distance from a point to the y-axis -/
def DistToYAxis (P : ℝ × ℝ) : ℝ := |P.1|

/-- Distance between two points -/
noncomputable def Distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

/-- Dot product of two vectors -/
def DotProduct (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem parabola_property (p : ℝ) :
  (∀ P ∈ Parabola p, DistToYAxis P = Distance P (Focus p) - 1) → p = 2 := by sorry

theorem dot_product_property :
  ∃ m : ℝ, m > 0 ∧
  (∀ A B : ℝ × ℝ, A ∈ Parabola 2 → B ∈ Parabola 2 →
    (∃ t : ℝ, A.1 = t * A.2 + m ∧ B.1 = t * B.2 + m) →
      DotProduct (A.1 - 1, A.2) (B.1 - 1, B.2) < 0) ∧
  (m > 3 - 2 * Real.sqrt 2 ∧ m < 3 + 2 * Real.sqrt 2) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_property_dot_product_property_l1058_105823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_brush_ratio_l1058_105828

/-- Proves that for a square with area 49 cm², if a brush of unknown width is swept along both diagonals
    such that one-third of the square's area is painted, then the ratio of the side length of the square
    to the brush width is 3√2. -/
theorem square_brush_ratio (s w : ℝ) (h1 : s^2 = 49) (h2 : s * (Real.sqrt 2 * w) = 49 / 3) : 
  s / w = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_brush_ratio_l1058_105828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1058_105842

open Int

def satisfies_condition (n : ℕ) : Prop :=
  n ≥ 2 ∧ ∀ a b : ℤ, (Nat.gcd a.natAbs n = 1) → (Nat.gcd b.natAbs n = 1) →
    (a ≡ b [ZMOD n] ↔ a * b ≡ 1 [ZMOD n])

theorem solution_set : {n : ℕ | satisfies_condition n} = {3, 4, 6, 8, 12, 24} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1058_105842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_rectangle_area_relation_l1058_105804

theorem square_rectangle_area_relation : 
  ∃ (x₁ x₂ : ℝ), 
    (∀ x : ℝ, x = x₁ ∨ x = x₂ → 
      (x - 2) * (x + 5) = 3 * (x - 3)^2) ∧ 
    x₁ + x₂ = 21/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_rectangle_area_relation_l1058_105804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1058_105844

-- Define the function f
def f (x : ℝ) : ℝ := 2 * abs (x + 1) - x

-- State the theorem
theorem inequality_proof (a : ℝ) (b : ℝ) (h1 : ∀ x, f x ≥ b) (h2 : a ≥ b) (h3 : b = 1) :
  Real.sqrt (2 * a - b) + Real.sqrt (a^2 - b) ≥ a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1058_105844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_M_x_coordinate_l1058_105819

-- Define the line equation
def line (x y : ℝ) : Prop := 2 * x - y + 3 = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 5

-- Define the condition for tangents forming a 90° angle
def tangents_90_degrees (x y : ℝ) : Prop :=
  line x y ∧ 
  ∃ (p q : ℝ × ℝ), 
    circle_eq p.1 p.2 ∧ 
    circle_eq q.1 q.2 ∧ 
    (x - p.1)^2 + (y - p.2)^2 = (x - q.1)^2 + (y - q.2)^2 ∧
    (x - p.1) * (x - q.1) + (y - p.2) * (y - q.2) = 0

theorem point_M_x_coordinate :
  ∀ (x y : ℝ), tangents_90_degrees x y → x = -1 ∨ x = -3/5 := by
  sorry

#check point_M_x_coordinate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_M_x_coordinate_l1058_105819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_sum_lower_bound_l1058_105883

/-- The parabola y² = -4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = -4 * p.1}

/-- Point A -/
def A : ℝ × ℝ := (0, 1)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Distance from a point to the y-axis (axis of symmetry) -/
def distToYAxis (p : ℝ × ℝ) : ℝ := |p.1|

theorem parabola_distance_sum_lower_bound :
  ∀ P ∈ Parabola, distToYAxis P + distance P A ≥ Real.sqrt 2 := by
  sorry

#check parabola_distance_sum_lower_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_sum_lower_bound_l1058_105883
