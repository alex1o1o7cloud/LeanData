import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tim_workdays_biked_l500_50075

/-- Calculates the number of workdays Tim rides his bike to work -/
noncomputable def workdaysBiked (distanceToWork : ℝ) (weekendRideDistance : ℝ) (bikingSpeed : ℝ) (totalBikingHours : ℝ) : ℝ :=
  let totalDistance := bikingSpeed * totalBikingHours
  let workdayDistance := totalDistance - weekendRideDistance
  let roundTripDistance := 2 * distanceToWork
  workdayDistance / roundTripDistance

theorem tim_workdays_biked :
  workdaysBiked 20 200 25 16 = 5 := by
  -- Unfold the definition of workdaysBiked
  unfold workdaysBiked
  -- Simplify the expression
  simp
  -- The proof is completed using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tim_workdays_biked_l500_50075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_with_diameter_AB_l500_50052

/-- The line that encloses the diameter of the circle -/
def line (x y : ℝ) : Prop := 3 * x - 4 * y + 12 = 0

/-- Point A where the line intersects the x-axis -/
noncomputable def A : ℝ × ℝ := (-4, 0)

/-- Point B where the line intersects the y-axis -/
noncomputable def B : ℝ × ℝ := (0, 3)

/-- The center of the circle -/
noncomputable def C : ℝ × ℝ := (-2, 3/2)

/-- The radius of the circle -/
noncomputable def r : ℝ := 5/2

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := (x + 2)^2 + (y - 3/2)^2 = 25/4

theorem circle_with_diameter_AB :
  ∀ x y : ℝ, line x y → (x = A.1 ∨ x = B.1) → (y = A.2 ∨ y = B.2) →
  ∀ p : ℝ × ℝ, p ∈ Set.Icc A B → circle_equation p.1 p.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_with_diameter_AB_l500_50052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_max_l500_50067

theorem triangle_angle_max (a b c : ℝ) (A B C : ℝ) :
  (a > 0) →
  (b > 0) →
  (c > 0) →
  (A > 0) →
  (B > 0) →
  (C > 0) →
  (A + B + C = π) →
  (a = b * Real.sin A) →
  (b = c * Real.sin B) →
  (c = a * Real.sin C) →
  (∀ x : ℝ, ∃ y : ℝ, ((1/3) * x^3 + b * x^2 + (a^2 + c^2 + Real.sqrt 3 * a * c) * x < y)) →
  (B ≤ 5 * π / 6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_max_l500_50067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_non_overlapping_sets_l500_50034

/-- Definition of an original set in an n × n grid -/
def OriginalSet (n : ℕ) := { s : Finset (ℕ × ℕ) // s.card = n - 1 ∧ 
  ∀ (x y : ℕ × ℕ), x ∈ s → y ∈ s → x ≠ y → x.1 ≠ y.1 ∧ x.2 ≠ y.2 }

/-- Theorem stating that n+1 non-overlapping original sets can be chosen from an n × n grid -/
theorem exist_non_overlapping_sets (n : ℕ) (h : n > 0) :
  ∃ (sets : Finset (OriginalSet n)), sets.card = n + 1 ∧
  ∀ (s t : OriginalSet n), s ∈ sets → t ∈ sets → s ≠ t → s.1 ∩ t.1 = ∅ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_non_overlapping_sets_l500_50034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l500_50057

-- Define the star operation
noncomputable def star (a b : ℝ) : ℝ := Real.sqrt (a + b) / Real.sqrt (a - b)

-- Theorem statement
theorem solution_exists (x : ℝ) (h : star x 10 = 6) : x = 74 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l500_50057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_surface_area_base_edge_l500_50001

/-- The volume of a right triangular prism with an equilateral triangular base -/
noncomputable def volume (a h : ℝ) : ℝ := (Real.sqrt 3 / 4) * a^2 * h

/-- The surface area of a right triangular prism with an equilateral triangular base -/
noncomputable def surface_area (a h : ℝ) : ℝ := Real.sqrt 3 * a^2 + 3 * a * h

/-- Theorem: For a right triangular prism with an equilateral triangular base and volume 16,
    the length of the base edge that minimizes the surface area is 4. -/
theorem min_surface_area_base_edge :
  ∃ (a : ℝ), a > 0 ∧ 
  (∃ (h : ℝ), h > 0 ∧ volume a h = 16) ∧
  (∀ (b : ℝ), b > 0 → 
    (∃ (k : ℝ), k > 0 ∧ volume b k = 16) →
    surface_area a (16 / ((Real.sqrt 3 / 4) * a^2)) ≤ surface_area b (16 / ((Real.sqrt 3 / 4) * b^2))) ∧
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_surface_area_base_edge_l500_50001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_different_points_l500_50015

def A : Finset ℕ := {5}
def B : Finset ℕ := {1, 2}
def C : Finset ℕ := {1, 3, 4}

def cartesian_product (A B C : Finset ℕ) : Finset (ℕ × ℕ × ℕ) :=
  A.product (B.product C)

theorem num_different_points : (cartesian_product A B C).card = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_different_points_l500_50015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinearity_iff_real_simple_ratio_concyclicity_or_collinearity_iff_real_cross_ratio_l500_50054

/-- Given complex numbers a, b, c, the function checks if they are collinear -/
def areCollinear (a b c : ℂ) : Prop :=
  ∃ (k : ℝ), b - a = k • (c - a)

/-- Given complex numbers a, b, c, d, the function checks if they are concyclic or collinear -/
noncomputable def areConcyclicOrCollinear (a b c d : ℂ) : Prop :=
  ∃ (z : ℂ), Complex.abs ((a - z) / (b - z)) = Complex.abs ((c - z) / (d - z))

/-- The simple ratio of three complex numbers -/
noncomputable def simpleRatio (a b c : ℂ) : ℂ :=
  (a - b) / (a - c)

/-- The cross ratio of four complex numbers -/
noncomputable def crossRatio (a b c d : ℂ) : ℂ :=
  ((a - c) / (a - d)) / ((b - c) / (b - d))

theorem collinearity_iff_real_simple_ratio (a b c : ℂ) :
  areCollinear a b c ↔ ∃ (r : ℝ), simpleRatio a b c = r := by
  sorry

theorem concyclicity_or_collinearity_iff_real_cross_ratio (a b c d : ℂ) :
  areConcyclicOrCollinear a b c d ↔ ∃ (r : ℝ), crossRatio a b c d = r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinearity_iff_real_simple_ratio_concyclicity_or_collinearity_iff_real_cross_ratio_l500_50054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_range_l500_50036

noncomputable def A : Set ℝ := {x | |x + 1/2| < 3/2}
noncomputable def B (a : ℝ) : Set ℝ := {x | (1/Real.pi)^(2*x) > Real.pi^(-a-x)}
def U : Set ℝ := Set.univ

theorem solution_range (a : ℝ) : (Uᶜ ∩ B a = B a) ↔ a ∈ Set.Iic (-2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_range_l500_50036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_inverse_cancellation_l500_50031

theorem power_inverse_cancellation (a : ℝ) (h : a ≠ 0) : a^5 * a^(-5 : ℤ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_inverse_cancellation_l500_50031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_b_quotient_l500_50030

def factorial_b (m b : ℕ) : ℕ :=
  let k := (m - 1) / b
  (List.range (k + 1)).foldl (λ acc i => acc * (m - i * b)) 1

theorem factorial_b_quotient :
  factorial_b 40 6 / factorial_b 10 3 = 2293760 := by
  -- Proof goes here
  sorry

#eval factorial_b 40 6 / factorial_b 10 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_b_quotient_l500_50030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lucas_fibonacci_product_l500_50091

noncomputable def fibonacci : ℕ → ℝ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

noncomputable def lucas : ℕ → ℝ
  | 0 => 2
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

noncomputable def α : ℝ := (1 + Real.sqrt 5) / 2
noncomputable def β : ℝ := (1 - Real.sqrt 5) / 2

axiom fibonacci_formula (n : ℕ) : 
  fibonacci n = (α ^ n - β ^ n) / Real.sqrt 5

axiom lucas_formula (n : ℕ) : 
  lucas n = α ^ n + β ^ n

theorem lucas_fibonacci_product (n : ℕ) (h : n ≥ 1) :
  ∃ k : ℕ, lucas (2 * n + 1) + (-1) ^ (n + 1) = 
    fibonacci (2 * k) * fibonacci (2 * k + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lucas_fibonacci_product_l500_50091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_remaining_in_tank_l500_50025

/-- Calculates the number of fish remaining in Danny's tank after selling specified fractions of each type of fish. -/
theorem fish_remaining_in_tank : 
  let initial_guppies : ℕ := 225
  let initial_angelfish : ℕ := 175
  let initial_tiger_sharks : ℕ := 200
  let initial_oscar_fish : ℕ := 140
  let initial_discus_fish : ℕ := 120
  let sold_guppies : ℕ := (3 * initial_guppies) / 5
  let sold_angelfish : ℕ := (3 * initial_angelfish) / 7
  let sold_tiger_sharks : ℕ := initial_tiger_sharks / 4
  let sold_oscar_fish : ℕ := initial_oscar_fish / 2
  let sold_discus_fish : ℕ := (2 * initial_discus_fish) / 3
  let remaining_guppies : ℕ := initial_guppies - sold_guppies
  let remaining_angelfish : ℕ := initial_angelfish - sold_angelfish
  let remaining_tiger_sharks : ℕ := initial_tiger_sharks - sold_tiger_sharks
  let remaining_oscar_fish : ℕ := initial_oscar_fish - sold_oscar_fish
  let remaining_discus_fish : ℕ := initial_discus_fish - sold_discus_fish
  remaining_guppies + remaining_angelfish + remaining_tiger_sharks + 
  remaining_oscar_fish + remaining_discus_fish = 450
:= by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_remaining_in_tank_l500_50025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_triangle_bound_l500_50029

/-- Pascal's triangle (probability triangle) -/
def pascal_triangle : ℕ → List ℚ := sorry

/-- The n-th row of Pascal's triangle -/
def nth_row (n : ℕ) : List ℚ := pascal_triangle n

/-- Theorem: All elements in the n-th row of Pascal's triangle are bounded above by 1/√n -/
theorem pascal_triangle_bound (n : ℕ) (x : ℚ) (h : x ∈ nth_row n) :
  x ≤ 1 / Real.sqrt (n : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_triangle_bound_l500_50029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_representation_l500_50020

def is_valid_sequence (a : ℕ → ℕ) : Prop :=
  (∀ n, Even (a n)) ∧
  (∀ n, a n ≤ a (n + 1)) ∧
  (∀ k, Even k → (k / 2) = (Finset.filter (fun n ↦ a n = k) (Finset.range (k * k / 2 + 1))).card)

theorem sequence_representation :
  ∃ (a : ℕ → ℕ) (b c d : ℤ),
    is_valid_sequence a ∧
    (∀ n : ℕ, a n = b * ⌊Real.sqrt (n + c : ℝ)⌋ + d) ∧
    b + c + d = 2 := by
  sorry

#check sequence_representation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_representation_l500_50020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l500_50059

/-- An ellipse with axes parallel to the coordinate axes, tangent to the x-axis at (6, 0) and to the y-axis at (0, 3) -/
structure TangentEllipse where
  center : ℝ × ℝ
  major_axis : ℝ
  minor_axis : ℝ
  x_tangent : center.1 - major_axis / 2 = 6
  y_tangent : center.2 - minor_axis / 2 = 3
  axes_parallel : True

/-- The distance between the foci of the ellipse -/
noncomputable def foci_distance (e : TangentEllipse) : ℝ :=
  Real.sqrt (e.major_axis^2 / 4 - e.minor_axis^2 / 4)

theorem ellipse_foci_distance (e : TangentEllipse) :
  foci_distance e = 3 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l500_50059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_of_z_l500_50083

-- Define the complex number z
noncomputable def z : ℂ := (Complex.I * Real.sqrt 3 + 1) / (1 + Complex.I)

-- State the theorem
theorem absolute_value_of_z : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_of_z_l500_50083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_y_axis_reflection_distance_p_to_p_prime_l500_50095

/-- The distance between a point and its reflection over the y-axis -/
theorem distance_to_y_axis_reflection (x y : ℝ) : 
  let p : ℝ × ℝ := (x, y)
  let p_reflected : ℝ × ℝ := (-x, y)
  Real.sqrt ((p_reflected.1 - p.1)^2 + (p_reflected.2 - p.2)^2) = 2 * abs x :=
by sorry

/-- The specific case for P(2, -4) -/
theorem distance_p_to_p_prime : 
  let p : ℝ × ℝ := (2, -4)
  let p_reflected : ℝ × ℝ := (-2, -4)
  Real.sqrt ((p_reflected.1 - p.1)^2 + (p_reflected.2 - p.2)^2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_y_axis_reflection_distance_p_to_p_prime_l500_50095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_empty_time_l500_50094

noncomputable def tank_capacity : ℝ := 5760
noncomputable def leak_empty_time : ℝ := 6
noncomputable def inlet_rate : ℝ := 4

noncomputable def leak_rate : ℝ := tank_capacity / leak_empty_time
noncomputable def inlet_rate_per_hour : ℝ := inlet_rate * 60
noncomputable def net_rate : ℝ := leak_rate - inlet_rate_per_hour

theorem tank_empty_time : tank_capacity / net_rate = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_empty_time_l500_50094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_exists_and_perpendicular_l500_50008

noncomputable def h (x : ℝ) : ℝ := 2 * Real.cos (x / 2)

def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (2, 6)
def P : ℝ × ℝ := (0, 2)

theorem point_P_exists_and_perpendicular :
  P.1 = 0 ∧ P.2 = h P.1 ∧
  ((P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_exists_and_perpendicular_l500_50008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_factorial_base_eight_zeroes_l500_50004

/-- The number of trailing zeroes in n! when written in base b -/
def trailingZeroes (n : ℕ) (b : ℕ) : ℕ := sorry

/-- 15 factorial -/
def factorial15 : ℕ := Nat.factorial 15

theorem fifteen_factorial_base_eight_zeroes :
  trailingZeroes factorial15 8 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_factorial_base_eight_zeroes_l500_50004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_triangular_region_l500_50018

/-- A line in a plane --/
structure Line where
  -- We'll use a simple representation for now
  slope : ℝ
  intercept : ℝ

/-- A region in a plane --/
structure Region where
  -- We'll use a simple representation for now
  vertices : List (ℝ × ℝ)

/-- A configuration of lines on a plane --/
structure LineConfiguration where
  n : ℕ
  lines : Fin n → Line
  h_n : n ≥ 3

/-- Predicate to check if two lines are parallel --/
def isParallelTo (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Predicate to check if three lines are concurrent --/
def isConcurrentWith (l1 l2 l3 : Line) : Prop :=
  ∃ x y : ℝ, (y = l1.slope * x + l1.intercept) ∧
              (y = l2.slope * x + l2.intercept) ∧
              (y = l3.slope * x + l3.intercept)

/-- Predicate to check if a region is triangular --/
def isTriangular (r : Region) : Prop :=
  r.vertices.length = 3

/-- Predicate to check if a region is adjacent to a line --/
def isAdjacentTo (r : Region) (l : Line) : Prop :=
  ∃ v1 v2 : ℝ × ℝ, v1 ∈ r.vertices ∧ v2 ∈ r.vertices ∧
    v1.1 ≠ v2.1 ∧ v1.2 = l.slope * v1.1 + l.intercept ∧
    v2.2 = l.slope * v2.1 + l.intercept

/-- Main theorem --/
theorem adjacent_triangular_region (config : LineConfiguration) :
  ∀ i, ∃ r : Region, isTriangular r ∧ isAdjacentTo r (config.lines i) :=
by
  intro i
  sorry  -- The actual proof would go here

/-- Additional axioms to represent the problem conditions --/
axiom not_parallel (config : LineConfiguration) :
  ∀ i j, i ≠ j → ¬ isParallelTo (config.lines i) (config.lines j)

axiom not_concurrent (config : LineConfiguration) :
  ∀ i j k, i ≠ j → j ≠ k → i ≠ k →
    ¬ isConcurrentWith (config.lines i) (config.lines j) (config.lines k)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_triangular_region_l500_50018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l500_50043

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 1 - a / (2^x + 1)

-- Part 1
theorem part_one (a : ℝ) :
  (∀ x : ℝ, f a x = -f a (-x)) →
  (a = 2 ∧ ∀ x y : ℝ, x < y → f a x < f a y) := by
  sorry

-- Part 2
theorem part_two :
  (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧
    f 2 (k * 4^x) + f 2 (1 - 2^(x+1)) ≥ 0) →
  k ≥ 0 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l500_50043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l500_50042

noncomputable def f (x : ℝ) : ℝ := (x^3 - 3*x^2 + 6*x + 4) / (x^2 - 5*x + 6)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 2 ∨ (2 < x ∧ x < 3) ∨ 3 < x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l500_50042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_lower_bound_l500_50006

theorem lambda_lower_bound (l : ℝ) : 
  (∃ S : Set (ℕ × ℕ), Set.Infinite S ∧ 
    ∀ (a b : ℕ), (a, b) ∈ S → 
      0 < Real.sqrt 2002 - (a : ℝ) / b ∧ 
      Real.sqrt 2002 - (a : ℝ) / b < l / ((a : ℝ) * b)) → 
  l ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_lower_bound_l500_50006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_angle_exists_l500_50090

-- Define the angle in radians
noncomputable def θ : ℝ := 22.5 * Real.pi / 180

-- Define the condition for tan(2^n θ)
def tan_condition (n : ℕ) : Prop :=
  if n % 3 = 0 then Real.tan (2^n * θ) > 0 else Real.tan (2^n * θ) < 0

theorem unique_angle_exists :
  -- θ is between 0 and π/2 radians (0° and 90°)
  0 < θ ∧ θ < Real.pi / 2 ∧
  -- The tan condition holds for all nonnegative integers
  (∀ n : ℕ, tan_condition n) ∧
  -- θ is equal to 22.5°
  θ = 22.5 * Real.pi / 180 ∧
  -- When expressed as p/q degrees where p and q are relatively prime,
  -- p + q = 235
  ∃ p q : ℕ, Nat.Coprime p q ∧ θ * 180 / Real.pi = p / q ∧ p + q = 235 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_angle_exists_l500_50090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_of_three_to_negative_two_l500_50093

theorem sqrt_of_three_to_negative_two (x : ℝ) :
  x^2 = (1/3)^2 → x = 1/3 ∨ x = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_of_three_to_negative_two_l500_50093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_B_sin_C_perimeter_special_case_l500_50038

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  area : ℝ

-- Define the conditions
axiom triangle_area (t : Triangle) : t.area = t.a^2 / (3 * Real.sin t.A)

-- Theorem 1
theorem sin_B_sin_C (t : Triangle) : Real.sin t.B * Real.sin t.C = 2/3 := by
  sorry

-- Theorem 2
theorem perimeter_special_case (t : Triangle) 
  (h1 : 6 * Real.cos t.B * Real.cos t.C = 1) 
  (h2 : t.a = 3) : 
  t.a + t.b + t.c = 3 + Real.sqrt 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_B_sin_C_perimeter_special_case_l500_50038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_volume_at_target_temperature_l500_50058

/-- Represents the volume of gas at a given temperature -/
def gas_volume (temperature : ℝ) : ℝ := sorry

/-- The rate of volume change per 5 degree temperature change -/
def volume_change_rate : ℝ := 5

/-- The reference temperature -/
def reference_temperature : ℝ := 30

/-- The reference volume at the reference temperature -/
def reference_volume : ℝ := 40

/-- The target temperature for which we want to find the volume -/
def target_temperature : ℝ := 10

theorem gas_volume_at_target_temperature :
  gas_volume target_temperature = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_volume_at_target_temperature_l500_50058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_base_angle_l500_50080

structure IsoscelesTriangle where
  apexAngle : ℝ
  baseAngle : ℝ
  isIsosceles : baseAngle + baseAngle + apexAngle = 180

theorem isosceles_triangle_base_angle (triangle : IsoscelesTriangle) 
  (h : triangle.apexAngle = 40) : 
  triangle.baseAngle = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_base_angle_l500_50080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2017_l500_50028

def sequence_a : ℕ → ℚ
  | 0 => 1
  | n + 1 => (2016 * sequence_a n) / (2014 * sequence_a n + 2016)

theorem sequence_a_2017 : sequence_a 2016 = 1008 / (1007 * 2017 + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2017_l500_50028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_points_and_monotonicity_l500_50016

noncomputable def f (x : ℝ) := (1/3) * x^3 + (1/2) * x^2 - 2 * x

theorem extremum_points_and_monotonicity :
  (∃ (a b : ℝ), a ≠ 0 ∧ f = λ x ↦ a * x^3 + b * x^2 - 2 * x) ∧
  (∀ x, deriv f x = 0 ↔ x = 1 ∨ x = -2) ∧
  (∀ x, x < -2 → deriv f x > 0) ∧
  (∀ x, x > 1 → deriv f x > 0) ∧
  (∀ x, -2 < x ∧ x < 1 → deriv f x < 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_points_and_monotonicity_l500_50016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simson_lines_intersection_l500_50040

-- Define a circle
def Circle : Type := ℂ → Prop

-- Define points on a circle
def PointsOnCircle (c : Circle) (A B C D : ℂ) : Prop :=
  c A ∧ c B ∧ c C ∧ c D

-- Define the concept of points being in sequence on a circle
def PointsInSequence (c : Circle) (A B C D : ℂ) : Prop :=
  PointsOnCircle c A B C D ∧ ∃ (θ₁ θ₂ θ₃ θ₄ : ℝ), 
    0 ≤ θ₁ ∧ θ₁ < θ₂ ∧ θ₂ < θ₃ ∧ θ₃ < θ₄ ∧ θ₄ < 2 * Real.pi ∧
    A = Complex.exp (Complex.I * θ₁) ∧
    B = Complex.exp (Complex.I * θ₂) ∧
    C = Complex.exp (Complex.I * θ₃) ∧
    D = Complex.exp (Complex.I * θ₄)

-- Define Simson line
def SimsonLine (P X Y Z : ℂ) : Set ℂ := sorry

-- Define the intersection point of Simson lines
noncomputable def SimsonIntersection (A B C D : ℂ) : ℂ := (A + B + C + D) / 2

-- State the theorem
theorem simson_lines_intersection (c : Circle) (A B C D : ℂ) :
  PointsInSequence c A B C D →
  ∃ (P : ℂ), P ∈ SimsonLine A B C D ∩ SimsonLine B C D A ∩ 
             SimsonLine C D A B ∩ SimsonLine D A B C :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simson_lines_intersection_l500_50040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_at_most_one_obtuse_l500_50089

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_angles : angles 0 + angles 1 + angles 2 = Real.pi
  positive_angles : ∀ i, 0 < angles i

-- Define an obtuse angle
def is_obtuse (angle : ℝ) : Prop := Real.pi / 2 < angle

-- Theorem statement
theorem triangle_at_most_one_obtuse (t : Triangle) :
  ¬ (∃ i j : Fin 3, i ≠ j ∧ is_obtuse (t.angles i) ∧ is_obtuse (t.angles j)) :=
by sorry

#check triangle_at_most_one_obtuse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_at_most_one_obtuse_l500_50089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_arrangement_sum_zero_l500_50061

def integers : List Int := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

def is_valid_arrangement (arr : Matrix (Fin 3) (Fin 3) Int) : Prop :=
  ∀ i j, arr i j ∈ integers

def common_sum (arr : Matrix (Fin 3) (Fin 3) Int) : Int :=
  arr 0 0 + arr 0 1 + arr 0 2

def has_common_sum (arr : Matrix (Fin 3) (Fin 3) Int) : Prop :=
  ∀ i : Fin 3,
    (arr i 0 + arr i 1 + arr i 2 = common_sum arr) ∧
    (arr 0 i + arr 1 i + arr 2 i = common_sum arr)

def diagonals_have_common_sum (arr : Matrix (Fin 3) (Fin 3) Int) : Prop :=
  (arr 0 0 + arr 1 1 + arr 2 2 = common_sum arr) ∧
  (arr 0 2 + arr 1 1 + arr 2 0 = common_sum arr)

theorem square_arrangement_sum_zero
  (arr : Matrix (Fin 3) (Fin 3) Int)
  (h1 : is_valid_arrangement arr)
  (h2 : has_common_sum arr)
  (h3 : diagonals_have_common_sum arr) :
  common_sum arr = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_arrangement_sum_zero_l500_50061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_difference_l500_50011

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Define the x-coordinates of the intercepts
def x₁ : ℝ := sorry
def x₂ : ℝ := sorry
def x₃ : ℝ := sorry
def x₄ : ℝ := sorry

-- State the theorem
theorem intercept_difference :
  (∀ x, g x = -f (120 - x)) →  -- g(x) = -f(120-x)
  (∃ v, g v = f v ∧ ∀ x, f x ≤ f v) →  -- g contains vertex of f
  (x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄) →  -- x₁, x₂, x₃, x₄ in increasing order
  (x₃ - x₂ = 180) →  -- given condition
  (x₄ - x₁ = 720 + 540 * Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_difference_l500_50011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_mersenne_prime_under_500_l500_50078

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- Check if a number is a Fibonacci number -/
def isFibonacci (n : ℕ) : Prop :=
  ∃ k, fib k = n

/-- Mersenne number -/
def mersenne_number (p : ℕ) : ℕ := 2^p - 1

theorem largest_mersenne_prime_under_500 :
  ∃ (n : ℕ), 
    isFibonacci n ∧ 
    Nat.Prime n ∧
    Nat.Prime (mersenne_number n) ∧
    mersenne_number n < 500 ∧
    ∀ (m : ℕ), 
      isFibonacci m → 
      Nat.Prime m → 
      Nat.Prime (mersenne_number m) → 
      mersenne_number m < 500 → 
      mersenne_number m ≤ mersenne_number n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_mersenne_prime_under_500_l500_50078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_when_t_zero_t_range_when_f_geq_4x_l500_50017

-- Define the function f
noncomputable def f (t : ℝ) (x : ℝ) : ℝ := (t + 1) * Real.log x + t * x^2 + 3 * t

-- Statement 1
theorem f_inequality_when_t_zero (x : ℝ) (h : x ≥ 0) :
  f 0 (x + 1) ≥ x - (1/2) * x^2 := by
  sorry

-- Statement 2
theorem t_range_when_f_geq_4x (t : ℝ) :
  (∀ x ≥ 1, f t x ≥ 4 * x) → t ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_when_t_zero_t_range_when_f_geq_4x_l500_50017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perimeter_obtuse_double_angle_l500_50087

/-- A triangle with integral sides --/
structure IntegralTriangle where
  a : ℕ
  b : ℕ
  c : ℕ

/-- An obtuse-angled triangle --/
def IsObtuseAngled (t : IntegralTriangle) : Prop :=
  t.a * t.a > t.b * t.b + t.c * t.c

/-- One acute angle is twice the other --/
def HasDoubleAcuteAngle (t : IntegralTriangle) : Prop :=
  ∃ θ : Real, 0 < θ ∧ θ < Real.pi/2 ∧ 
    Real.sin θ / t.c = Real.sin (2*θ) / t.b

/-- The perimeter of a triangle --/
def Perimeter (t : IntegralTriangle) : ℕ :=
  t.a + t.b + t.c

/-- The theorem statement --/
theorem smallest_perimeter_obtuse_double_angle :
  ∀ t : IntegralTriangle,
    IsObtuseAngled t → HasDoubleAcuteAngle t →
    Perimeter t ≥ 77 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perimeter_obtuse_double_angle_l500_50087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_and_g_l500_50071

noncomputable def f (x : ℝ) : ℝ := x * (x - 1 / x^2)

noncomputable def g (x : ℝ) : ℝ := (Real.cos x - x) / x^2

theorem derivative_f_and_g :
  (∀ x, x ≠ 0 → deriv f x = 2 * x + 1 / x^2) ∧
  (∀ x, x ≠ 0 → deriv g x = (x - x * Real.sin x - 2 * Real.cos x) / x^3) := by
  sorry

#check derivative_f_and_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_and_g_l500_50071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_shift_sine_graph_l500_50097

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)

theorem horizontal_shift_sine_graph :
  ∀ x : ℝ, g x = f (x - Real.pi / 4) :=
by
  intro x
  simp [f, g]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_shift_sine_graph_l500_50097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estimated_height_is_166_l500_50079

/-- Represents the data and conditions of the linear regression problem -/
structure RegressionData where
  n : ℕ             -- sample size
  sum_x : ℝ         -- sum of foot lengths
  sum_y : ℝ         -- sum of heights
  slope : ℝ         -- slope of regression line
  foot_length : ℝ   -- foot length for estimation

/-- Calculates the estimated height based on the regression data -/
noncomputable def estimate_height (data : RegressionData) : ℝ :=
  let mean_x := data.sum_x / data.n
  let mean_y := data.sum_y / data.n
  let intercept := mean_y - data.slope * mean_x
  data.slope * data.foot_length + intercept

/-- Theorem stating that the estimated height for the given data is 166 cm -/
theorem estimated_height_is_166 (data : RegressionData) 
    (h1 : data.n = 10)
    (h2 : data.sum_x = 225)
    (h3 : data.sum_y = 1600)
    (h4 : data.slope = 4)
    (h5 : data.foot_length = 24) :
    estimate_height data = 166 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_estimated_height_is_166_l500_50079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l500_50077

/-- The directrix of the parabola y = (1/4)x^2 is y = -1 -/
theorem parabola_directrix (x y : ℝ) : 
  (y = (1/4) * x^2) → (∃ p : ℝ, p > 0 ∧ y = (1/(4*p)) * x^2 ∧ directrix = -p) :=
by
  sorry

def directrix : ℝ := -1


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l500_50077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_group_l500_50047

open Real Matrix

-- Define the rigid body rotation transformation
noncomputable def T (α : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos α, Real.sin α],
    ![-Real.sin α, Real.cos α]]

-- Define the set of all rigid body rotation transformations
def RotationSet : Set (Matrix (Fin 2) (Fin 2) ℝ) :=
  {M | ∃ α : ℝ, M = T α}

-- Theorem statement
theorem rotation_group : Group RotationSet := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_group_l500_50047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_from_sin_sum_diff_l500_50092

theorem tan_ratio_from_sin_sum_diff (α β : ℝ) 
  (h1 : Real.sin (α + β) = 2/3) 
  (h2 : Real.sin (α - β) = 1/3) : 
  Real.tan α / Real.tan β = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_from_sin_sum_diff_l500_50092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_l500_50035

-- Define the complex number z as a function of a
noncomputable def z (a : ℝ) : ℂ := a - 10 / (3 - Complex.I)

-- Theorem statement
theorem pure_imaginary_condition (a : ℝ) : 
  (z a).re = 0 → a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_l500_50035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_well_diameter_approx_l500_50022

/-- The diameter of a cylindrical well -/
noncomputable def well_diameter (depth : ℝ) (cost_per_cubic_meter : ℝ) (total_cost : ℝ) : ℝ :=
  2 * Real.sqrt ((total_cost / cost_per_cubic_meter) / (Real.pi * depth))

/-- Theorem stating the diameter of the well given the conditions -/
theorem well_diameter_approx :
  let depth : ℝ := 14
  let cost_per_cubic_meter : ℝ := 17
  let total_cost : ℝ := 1682.32
  ∃ ε > 0, |well_diameter depth cost_per_cubic_meter total_cost - 2.996| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_well_diameter_approx_l500_50022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_microphotonics_percentage_l500_50019

noncomputable def total_degrees : ℝ := 360
noncomputable def total_percentage : ℝ := 100

noncomputable def home_electronics : ℝ := 24
noncomputable def food_additives : ℝ := 15
noncomputable def genetically_modified_microorganisms : ℝ := 29
noncomputable def industrial_lubricants : ℝ := 8
noncomputable def basic_astrophysics_degrees : ℝ := 50.4

noncomputable def basic_astrophysics_percentage : ℝ := (basic_astrophysics_degrees / total_degrees) * total_percentage

noncomputable def total_known_percentages : ℝ := home_electronics + food_additives + genetically_modified_microorganisms + industrial_lubricants + basic_astrophysics_percentage

theorem microphotonics_percentage : 
  total_percentage - total_known_percentages = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_microphotonics_percentage_l500_50019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l500_50064

/-- Two parallel lines in a 2D plane -/
structure ParallelLines where
  l₁ : ℝ → ℝ → Prop
  l₂ : ℝ → ℝ → Prop

/-- Distance between two parallel lines -/
noncomputable def distance (lines : ParallelLines) : ℝ := sorry

/-- Theorem: If the distance between two parallel lines l₁: x - y = 0 and l₂: x - y + b = 0 is √2, then b = 2 or b = -2 -/
theorem parallel_lines_distance (b : ℝ) :
  let lines : ParallelLines := {
    l₁ := λ x y ↦ x - y = 0,
    l₂ := λ x y ↦ x - y + b = 0
  }
  distance lines = Real.sqrt 2 → b = 2 ∨ b = -2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l500_50064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_product_l500_50021

theorem sine_cosine_product (θ : ℝ) 
  (h : (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 2) : 
  Real.sin θ * Real.cos θ = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_product_l500_50021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_range_l500_50050

-- Define the curve
noncomputable def curve (x : ℝ) : ℝ := x^3 - Real.sqrt 3 * x + 2/3

-- Define the angle of inclination
noncomputable def angle_of_inclination (x : ℝ) : ℝ :=
  Real.arctan (3 * x^2 - Real.sqrt 3)

-- Theorem statement
theorem tangent_angle_range :
  ∀ x : ℝ, 
    (0 ≤ angle_of_inclination x ∧ angle_of_inclination x ≤ Real.pi/2) ∨
    (2*Real.pi/3 ≤ angle_of_inclination x ∧ angle_of_inclination x < Real.pi) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_range_l500_50050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_is_25_l500_50082

def smallest_two_digit_number_with_special_product : ℕ → Prop :=
  fun n =>
    (n ≥ 10 ∧ n < 100) ∧  -- n is a two-digit number
    (let reverse := (n % 10) * 10 + n / 10;
      let product := n * reverse;
      product ≥ 1000 ∧ product < 10000 ∧  -- product is a four-digit number
      product % 100 = 0) ∧  -- ones and tens digits of product are 0
    (∀ m, m ≥ 10 ∧ m < n →
      let reverse := (m % 10) * 10 + m / 10;
      let product := m * reverse;
      ¬(product ≥ 1000 ∧ product < 10000 ∧ product % 100 = 0))

theorem smallest_number_is_25 : smallest_two_digit_number_with_special_product 25 := by
  sorry

#check smallest_number_is_25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_is_25_l500_50082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_problem_l500_50062

noncomputable section

variable (z a : ℂ)

theorem complex_problem :
  (z - 3) * (2 - Complex.I) = 5 →
  (∃ (b : ℝ), z * (a + Complex.I) = b • Complex.I) →
  (z = 5 + Complex.I ∧ Complex.abs (z - 2 + 3 * Complex.I) = 5 ∧ a = 1/5) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_problem_l500_50062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_one_implies_a_ge_one_l500_50099

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x

theorem f_greater_than_one_implies_a_ge_one (a : ℝ) :
  (∀ x > 1, f a x > 1) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_one_implies_a_ge_one_l500_50099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_with_area_8_l500_50000

/-- A line in the xy-plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

/-- The area of the triangle formed by a line and the coordinate axes -/
noncomputable def triangle_area (l : Line) : ℝ :=
  abs (l.c * l.c) / (2 * abs (l.a * l.b))

/-- The main theorem -/
theorem parallel_line_with_area_8 (l : Line) (l1 : Line) (h1 : l1.a = 1 ∧ l1.b = -3 ∧ l1.c = 6) 
    (h2 : parallel l l1) (h3 : triangle_area l = 8) :
    (l.a = 1 ∧ l.b = -3 ∧ l.c = 4 * Real.sqrt 3) ∨ 
    (l.a = 1 ∧ l.b = -3 ∧ l.c = -4 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_with_area_8_l500_50000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l500_50081

/-- A triangle with integral sides and perimeter 11 has area 5√11/4 -/
theorem triangle_area (a b c : ℕ) : 
  a + b + c = 11 → 
  a > 0 → b > 0 → c > 0 →
  a + b > c → b + c > a → c + a > b →
  (Real.sqrt (((a + b + c) / 2 : ℝ) * 
    (((a + b + c) / 2 : ℝ) - a) * 
    (((a + b + c) / 2 : ℝ) - b) * 
    (((a + b + c) / 2 : ℝ) - c))) = (5 * Real.sqrt 11) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l500_50081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l500_50088

/-- The family of curves parameterized by θ --/
def curve (θ : ℝ) (x y : ℝ) : Prop :=
  2 * (2 * Real.sin θ - Real.cos θ + 3) * x^2 - (8 * Real.sin θ + Real.cos θ + 1) * y = 0

/-- The line y = 2x --/
def line (x y : ℝ) : Prop := y = 2 * x

/-- The chord length function --/
noncomputable def chord_length (x : ℝ) : ℝ := 2 * Real.sqrt 2 * |x|

/-- Theorem stating the maximum chord length --/
theorem max_chord_length :
  ∃ (max_length : ℝ), max_length = 8 * Real.sqrt 5 ∧
  ∀ (θ x y : ℝ), curve θ x y → line x y →
  chord_length x ≤ max_length := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l500_50088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_existence_l500_50041

-- Define the basic types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Rectangle where
  p : Point
  q : Point
  r : Point
  s : Point

-- Define the given points and length
variable (A B C D : Point) (l : ℝ)

-- Define the conditions for the rectangle
def passes_through (rect : Rectangle) (p : Point) : Prop :=
  (rect.p.x = p.x ∨ rect.p.y = p.y) ∨
  (rect.q.x = p.x ∨ rect.q.y = p.y) ∨
  (rect.r.x = p.x ∨ rect.r.y = p.y) ∨
  (rect.s.x = p.x ∨ rect.s.y = p.y)

noncomputable def diagonal_length (rect : Rectangle) : ℝ :=
  Real.sqrt ((rect.p.x - rect.r.x)^2 + (rect.p.y - rect.r.y)^2)

-- State the theorem
theorem rectangle_existence :
  ∃ (rect : Rectangle),
    passes_through rect A ∧
    passes_through rect B ∧
    passes_through rect C ∧
    passes_through rect D ∧
    diagonal_length rect = l :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_existence_l500_50041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_A_given_B_l500_50060

/-- Represents the number of first prize works -/
noncomputable def x : ℝ := sorry

/-- Represents the number of second (and third) prize works -/
noncomputable def y : ℝ := sorry

/-- The probability of selecting a first prize work -/
noncomputable def P_A : ℝ := x / (x + 2*y)

/-- The probability of selecting a work from a senior year student -/
noncomputable def P_B : ℝ := (0.4*x + 0.4*y + 0.6*y) / (x + 2*y)

/-- The probability of selecting a first prize work from a senior year student -/
def P_AB : ℝ := 0.16

/-- The number of second and third prize works is the same -/
axiom second_third_equal : y = y

/-- The distribution of senior year students' works for first prize is 40% -/
axiom first_prize_senior : 0.4 * x / (x + 2*y) = P_AB

/-- The main theorem: given the conditions, prove that P(A|B) = 8/23 -/
theorem prob_A_given_B : P_AB / P_B = 8/23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_A_given_B_l500_50060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_surface_area_l500_50009

def cube_volumes : List ℕ := [1, 8, 27, 64, 125, 216, 343, 512]

def cube_side_lengths : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

def cube_surface_areas : List ℕ := cube_side_lengths.map (fun s => 6 * s^2)

def overlapped_areas : List ℕ := cube_side_lengths.tail.map (fun s => s^2)

def adjusted_surface_areas : List ℕ := 
  match cube_surface_areas, overlapped_areas with
  | [], _ => []
  | h::t, o => h :: (List.zipWith (fun s o => s - o) t o)

theorem tower_surface_area : 
  adjusted_surface_areas.sum = 1021 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_surface_area_l500_50009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_ratio_theorem_l500_50039

-- Define the angle α
variable (α : Real)

-- Define the radii of the three circles
variable (r₁ r₂ r₃ : Real)

-- Assume r₁ is the radius of the smaller circle
axiom h1 : r₁ ≤ r₂

-- Assume the circles are inscribed in the angle and touch each other
axiom h2 : Real.sin (α / 2) = (r₂ - r₁) / (r₁ + r₂)

-- Define the ratio of r₁ to r₃
noncomputable def ratio (α r₁ r₂ r₃ : Real) : Real := r₁ / r₃

-- State the theorem
theorem circle_ratio_theorem (α r₁ r₂ r₃ : Real) :
  ratio α r₁ r₂ r₃ = (Real.sqrt ((1 - Real.sin (α / 2)) / (1 + Real.sin (α / 2))) + 1)^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_ratio_theorem_l500_50039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_proof_l500_50056

/-- A linear function passing through points (0,3) and (-4,0) -/
noncomputable def linear_function (x : ℝ) : ℝ := -3/4 * x + 3

theorem linear_function_proof :
  (∀ x, linear_function x = -3/4 * x + 3) ∧
  (linear_function 0 = 3) ∧
  (linear_function (-4) = 0) ∧
  (∃ a, linear_function a = 6 ∧ a = -4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_proof_l500_50056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_show_segment_ratio_l500_50084

/-- Represents a radio show with three interview segments -/
structure RadioShow where
  total_length : ℕ
  first_segment : ℕ
  second_segment : ℕ
  third_segment : ℕ

/-- The properties of the radio show as described in the problem -/
def valid_radio_show (s : RadioShow) : Prop :=
  s.total_length = 90 ∧
  s.third_segment = 10 ∧
  s.third_segment * 2 = s.second_segment ∧
  s.first_segment + s.second_segment + s.third_segment = s.total_length

/-- The theorem stating that the ratio of the first segment to the other two is 2:1 -/
theorem radio_show_segment_ratio (s : RadioShow) 
  (h : valid_radio_show s) : 
  s.first_segment * 1 = (s.second_segment + s.third_segment) * 2 := by
  sorry

#check radio_show_segment_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_show_segment_ratio_l500_50084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l500_50068

theorem problem_solution : ∃ x : ℝ, (24 : ℝ)^3 = (16^2 / 4) * 2^(8*x) → x = 3/8 := by
  use (3/8 : ℝ)
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l500_50068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_odd_coefficients_l500_50027

def f (a : ℝ) (x : ℝ) : ℝ := (a + x) * (1 + x)^4

theorem sum_of_odd_coefficients (a : ℝ) : 
  (∃ c₁ c₃ c₅ : ℝ, ∀ x, f a x = f a 0 + c₁ * x + (f a 0 - a - c₁) * x^2 + c₃ * x^3 + (16 - f a 0 - c₃) * x^4 + c₅ * x^5 ∧ c₁ + c₃ + c₅ = 32) → 
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_odd_coefficients_l500_50027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_l500_50098

/-- CoeffX2 f returns the coefficient of x^2 in the polynomial f -/
noncomputable def CoeffX2 (f : ℝ → ℝ) : ℝ := sorry

/-- The coefficient of x^2 in the expansion of (3/x + x)(2 - √x)^6 is 243 -/
theorem coefficient_x_squared (x : ℝ) : 
  (CoeffX2 ((fun x => 3/x + x) * (fun x => (2 - Real.sqrt x)^6))) = 243 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_l500_50098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passing_time_l500_50069

/-- The time (in seconds) it takes for a train to pass a bridge -/
noncomputable def train_pass_time (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem stating that a 410-meter long train traveling at 45 km/hour 
    takes approximately 44 seconds to pass a 140-meter long bridge -/
theorem train_bridge_passing_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ 
  |train_pass_time 410 140 45 - 44| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passing_time_l500_50069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a_exists_l500_50026

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

-- State the theorem
theorem unique_a_exists : ∃! a : ℝ, 
  (A a ∩ B = A a ∪ B) ∧ 
  (Set.Nonempty (A a ∩ B) ∧ A a ∩ C = ∅) ∧ 
  (A a ∩ B = A a ∩ C ∧ A a ∩ B ≠ ∅) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a_exists_l500_50026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_surface_area_l500_50045

/-- A right pyramid with a square base and shifted peak -/
structure ShiftedPyramid where
  base_side : ℝ
  height : ℝ

/-- Calculate the total surface area of a shifted pyramid -/
noncomputable def total_surface_area (p : ShiftedPyramid) : ℝ :=
  p.base_side ^ 2 + 10 * Real.sqrt 164 + 8 * Real.sqrt 228

/-- Theorem: The total surface area of a specific shifted pyramid -/
theorem specific_pyramid_surface_area :
  let p : ShiftedPyramid := { base_side := 8, height := 10 }
  total_surface_area p = 64 + 10 * Real.sqrt 164 + 8 * Real.sqrt 228 := by
  sorry

#check specific_pyramid_surface_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_surface_area_l500_50045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l500_50048

theorem right_triangle_hypotenuse (S : ℝ) : 
  Real.cos S = (1 : ℝ) / 2 → 
  ∃ (P T : ℝ), 
    P = 10 ∧ 
    S * S = P * P + T * T ∧ 
    S = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l500_50048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_deletion_divisibility_l500_50024

theorem decimal_deletion_divisibility (n : ℕ) (a b : ℕ) :
  n > 0 →
  ¬(2 ∣ n) →
  ¬(5 ∣ n) →
  ∃ (k : ℕ), k > 0 ∧ (∃ (m : ℕ), a * 10^k = b * m + b / n) →
  n ∣ b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_deletion_divisibility_l500_50024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_share_difference_l500_50066

/-- Represents the share of profit for each person -/
structure Share where
  amount : ℝ

/-- Represents the distribution of profit -/
structure ProfitDistribution where
  total_profit : ℝ
  proportions : Fin 4 → ℝ
  shares : Fin 4 → Share

/-- Calculate the difference between two shares -/
def shareDifference (s1 s2 : Share) : ℝ :=
  s1.amount - s2.amount

/-- Main theorem: The difference between C's and B's shares is approximately 2295.09 -/
theorem profit_share_difference (d : ProfitDistribution) 
  (h1 : d.total_profit = 20000)
  (h2 : d.proportions 0 = 2)
  (h3 : d.proportions 1 = 3.5)
  (h4 : d.proportions 2 = 5.25)
  (h5 : d.proportions 3 = 4.5)
  (h6 : ∀ i, d.shares i = ⟨d.total_profit * d.proportions i / (d.proportions 0 + d.proportions 1 + d.proportions 2 + d.proportions 3)⟩) :
  abs (shareDifference (d.shares 2) (d.shares 1) - 2295.09) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_share_difference_l500_50066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l500_50044

-- Define the quadratic function f
def f (x : ℝ) : ℝ := -x^2 + 2*x + 15

-- Define function g
def g (m x : ℝ) : ℝ := (2 - 2*m)*x - f x

-- Define the minimum value function
noncomputable def min_value (m : ℝ) : ℝ :=
  if m ≤ 0 then -15
  else if m < 2 then -m^2 - 15
  else -4*m - 11

theorem quadratic_function_properties :
  (f (-3) = 0 ∧ f 5 = 0 ∧ f 2 = 15) ∧
  (∀ m : ℝ, ∀ x ∈ Set.Icc 0 2, g m x ≥ min_value m) := by
  sorry

#check quadratic_function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l500_50044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_of_numbers_l500_50005

noncomputable def numbers : List ℝ := [1, 2, 3, 4, 10]

noncomputable def mean (xs : List ℝ) : ℝ := xs.sum / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (fun x => (x - m) ^ 2)).sum / xs.length

noncomputable def standardDeviation (xs : List ℝ) : ℝ :=
  Real.sqrt (variance xs)

theorem standard_deviation_of_numbers :
  mean numbers = 4 ∧ standardDeviation numbers = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_of_numbers_l500_50005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_equation_l500_50049

-- Define the points of the triangle
noncomputable def A : ℝ × ℝ := (2, 1)
noncomputable def B : ℝ × ℝ := (-2, 3)
noncomputable def C : ℝ × ℝ := (0, 1)

-- Define the midpoint of BC
noncomputable def M : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the slope of AM
noncomputable def slope_AM : ℝ := (M.2 - A.2) / (M.1 - A.1)

-- Theorem statement
theorem median_equation :
  ∀ (x y : ℝ), (x + y - 3 = 0) ↔ (y - A.2 = slope_AM * (x - A.1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_equation_l500_50049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_projections_is_circle_l500_50051

structure RightCircularCone where
  height : ℝ
  base_radius : ℝ

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def distance_to_base_plane (p : Point3D) : ℝ :=
  p.z

def is_on_cone_surface (cone : RightCircularCone) (p : Point3D) : Prop :=
  p.x^2 + p.y^2 = (cone.base_radius * p.z / cone.height)^2

def is_parallel_to_base_plane (v : Point3D) : Prop :=
  v.z = 0

noncomputable def reflect_ray (incident : Point3D) (surface_normal : Point3D) : Point3D :=
  sorry -- Definition of ray reflection

def projection_to_base_plane (p : Point3D) : Point3D :=
  ⟨p.x, p.y, 0⟩

def vector_sub (a b : Point3D) : Point3D :=
  ⟨a.x - b.x, a.y - b.y, a.z - b.z⟩

theorem locus_of_projections_is_circle 
  (cone : RightCircularCone) 
  (A : Point3D) 
  (h1 : distance_to_base_plane A = cone.height) :
  ∃ (center : Point3D) (radius : ℝ),
    ∀ (M : Point3D), 
      is_on_cone_surface cone M → 
      is_parallel_to_base_plane (reflect_ray (vector_sub A M) (sorry : Point3D)) →
      ∃ (P : Point3D), 
        P = projection_to_base_plane M ∧ 
        (P.x - center.x)^2 + (P.y - center.y)^2 = radius^2 :=
by
  sorry -- Proof goes here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_projections_is_circle_l500_50051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_sequence_l500_50055

/-- Given positive integers a and b, with b > a > 1, and a not dividing b,
    and a sequence {b_n} satisfying b_{n+1} ≥ 2b_n,
    there exists a sequence {a_n} satisfying specific conditions. -/
theorem existence_of_special_sequence
  (a b : ℕ+) 
  (h_order : b > a)
  (h_a_gt_one : a > 1)
  (h_not_divide : ¬ (a ∣ b))
  (b_seq : ℕ → ℕ+)
  (h_b_seq : ∀ n : ℕ, b_seq (n + 1) ≥ 2 * b_seq n) :
  ∃ a_seq : ℕ → ℕ+,
    (∀ n : ℕ, (a_seq (n + 1) - a_seq n : ℤ) ∈ ({↑a, ↑b} : Set ℤ)) ∧
    (∀ m l : ℕ, (a_seq m + a_seq l : ℕ+) ∉ Set.range b_seq) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_sequence_l500_50055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_players_l500_50053

/-- The number of students who like to play basketball -/
def B : ℕ := sorry

/-- The number of students who like to play cricket -/
def C : ℕ := 8

/-- The number of students who like to play both basketball and cricket -/
def B_and_C : ℕ := 3

/-- The number of students who like to play basketball or cricket or both -/
def B_or_C : ℕ := 17

theorem basketball_players : B = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_players_l500_50053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_satisfies_conditions_l500_50063

/-- Calculates the compound interest amount after n years -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate / 100) ^ years - principal

/-- Calculates the simple interest amount -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * rate * (years : ℝ) / 100

/-- The principal amount that satisfies the given conditions -/
def principalAmount : ℝ := 1833.33

theorem principal_satisfies_conditions : 
  simpleInterest principalAmount 16 6 = 
    (1/2) * compoundInterest 8000 20 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_satisfies_conditions_l500_50063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_real_solutions_count_l500_50013

theorem distinct_real_solutions_count : ∃! (S : Finset ℝ), 
  (∀ a ∈ S, ∃ x₀ : ℤ, 
    (x₀ % 2 = 0) ∧ 
    (|x₀| < 1000) ∧ 
    (x₀^3 : ℝ) = a * x₀ + a + 1) ∧
  Finset.card S = 999 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_real_solutions_count_l500_50013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_reflection_point_l500_50002

def plane_equation (p : Fin 3 → ℝ) : ℝ := p 0 + p 1 + p 2

noncomputable def reflect_point (p : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![(-25/3), 19/3, 19/3]

def is_on_line (p q r : Fin 3 → ℝ) : Prop :=
  ∃ t : ℝ, p = λ i => q i + t * (r i - q i)

noncomputable def angle_of_incidence (p q : Fin 3 → ℝ) : ℝ := sorry
noncomputable def angle_of_reflection (p q : Fin 3 → ℝ) : ℝ := sorry

theorem light_reflection_point :
  let A : Fin 3 → ℝ := ![(-4), 10, 10]
  let C : Fin 3 → ℝ := ![4, 6, 8]
  let B : Fin 3 → ℝ := ![118/43, 244/43, 202/43]
  (plane_equation B = 10) ∧
  (angle_of_incidence A B = angle_of_reflection B C) ∧
  (is_on_line B (reflect_point A) C) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_reflection_point_l500_50002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_condition_l500_50046

noncomputable def f (a x : ℝ) : ℝ := Real.log x + (1/2) * x^2 + a * x

theorem extreme_points_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, deriv (f a) x = 0 → x = x₁ ∨ x = x₂) ∧
    f a x₁ + f a x₂ ≤ -5) →
  a ≤ -2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_condition_l500_50046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_milk_water_ratio_simplified_final_milk_water_ratio_l500_50033

/-- A can containing a mixture of milk and water -/
structure Can where
  capacity : ℕ
  initialMilkRatio : ℕ
  initialWaterRatio : ℕ
  additionalMilk : ℕ

/-- The final ratio of milk to water in the can -/
def finalRatio (can : Can) : ℕ × ℕ :=
  let initialTotal := can.capacity - can.additionalMilk
  let initialMilk := initialTotal * can.initialMilkRatio / (can.initialMilkRatio + can.initialWaterRatio)
  let initialWater := initialTotal * can.initialWaterRatio / (can.initialMilkRatio + can.initialWaterRatio)
  let finalMilk := initialMilk + can.additionalMilk
  let finalWater := initialWater
  (finalMilk, finalWater)

/-- Theorem stating the final ratio of milk to water in the can -/
theorem final_milk_water_ratio (can : Can) 
  (h1 : can.capacity = 72)
  (h2 : can.initialMilkRatio = 5)
  (h3 : can.initialWaterRatio = 3)
  (h4 : can.additionalMilk = 8) :
  finalRatio can = (48, 24) := by
  sorry

/-- Simplify the final ratio -/
def simplifyRatio (a b : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd a b
  (a / gcd, b / gcd)

/-- Theorem stating the simplified final ratio of milk to water in the can -/
theorem simplified_final_milk_water_ratio (can : Can) 
  (h1 : can.capacity = 72)
  (h2 : can.initialMilkRatio = 5)
  (h3 : can.initialWaterRatio = 3)
  (h4 : can.additionalMilk = 8) :
  simplifyRatio (finalRatio can).1 (finalRatio can).2 = (2, 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_milk_water_ratio_simplified_final_milk_water_ratio_l500_50033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compounded_ratio_simplification_l500_50085

def ratio1 : ℚ := 2 / 3
def ratio2 : ℚ := 6 / 11
def ratio3 : ℚ := 11 / 2

def compounded_ratio : ℚ := (ratio1.num * ratio2.num * ratio3.num) / (ratio1.den * ratio2.den * ratio3.den)

theorem compounded_ratio_simplification : compounded_ratio = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compounded_ratio_simplification_l500_50085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l500_50037

/-- Definition of the ellipse C -/
def C (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

/-- The length of the major axis of ellipse C -/
def majorAxisLength : ℝ := 8

/-- The focal length of ellipse C -/
noncomputable def focalLength : ℝ := 4 * Real.sqrt 3

/-- The eccentricity of ellipse C -/
noncomputable def eccentricity : ℝ := Real.sqrt 3 / 2

/-- Theorem stating the properties of ellipse C -/
theorem ellipse_properties :
  (∀ x y, C x y → 
    majorAxisLength = 8 ∧
    focalLength = 4 * Real.sqrt 3 ∧
    eccentricity = Real.sqrt 3 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l500_50037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_values_third_quadrant_trig_values_from_point_l500_50074

-- Part 1
theorem trig_values_third_quadrant (α : Real) :
  α ∈ Set.Icc π (3*π/2) →  -- α is in the third quadrant
  Real.tan α = 1/3 →
  Real.sin α = -Real.sqrt 10 / 10 ∧ Real.cos α = -3 * Real.sqrt 10 / 10 := by sorry

-- Part 2
theorem trig_values_from_point (α a : Real) :
  a ≠ 0 →
  ∃ (r : Real), r > 0 ∧ 3*a = r * Real.cos α ∧ 4*a = r * Real.sin α →
  (a > 0 → Real.sin α = 4/5 ∧ Real.cos α = 3/5 ∧ Real.tan α = 4/3) ∧
  (a < 0 → Real.sin α = -4/5 ∧ Real.cos α = -3/5 ∧ Real.tan α = 4/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_values_third_quadrant_trig_values_from_point_l500_50074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_equivalence_l500_50023

/-- Proves that 0.000000023 is equal to 2.3 × 10^(-8) in scientific notation -/
theorem scientific_notation_equivalence : 
  (0.000000023 : Real) = 2.3 * (10 : Real)^(-8 : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_equivalence_l500_50023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l500_50096

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (Real.sin x))

-- State the theorem
theorem f_range :
  Set.range f = Set.Ioo (-Real.pi / 2 : ℝ) 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l500_50096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_four_consecutive_integers_divisible_by_two_l500_50012

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) : 
  ∃ k : ℤ, (n - 1) + n + (n + 1) + (n + 2) = 2 * k := by
  use 2 * n + 1
  ring

#check sum_of_four_consecutive_integers_divisible_by_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_four_consecutive_integers_divisible_by_two_l500_50012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_evaluation_closest_whole_number_l500_50010

open BigOperators

theorem ratio_evaluation : (10^3000 + 10^3003) / (10^3001 + 10^3004) = (1 : ℚ) / 10 := by
  sorry

theorem closest_whole_number : 
  ∀ n : ℤ, |(1 : ℚ) / 10 - 0| < |(1 : ℚ) / 10 - n| ∨ (|(1 : ℚ) / 10 - 0| = |(1 : ℚ) / 10 - n| ∧ 0 ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_evaluation_closest_whole_number_l500_50010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_3000_pencils_l500_50014

/-- The cost of buying a given number of pencils with a potential discount -/
noncomputable def cost_of_pencils (box_size : ℕ) (box_cost : ℝ) (discount_rate : ℝ) (discount_threshold : ℕ) (num_pencils : ℕ) : ℝ :=
  let cost_per_pencil := box_cost / (box_size : ℝ)
  let total_cost := cost_per_pencil * (num_pencils : ℝ)
  if num_pencils > discount_threshold then
    total_cost * (1 - discount_rate)
  else
    total_cost

/-- The cost of buying 3000 pencils is $540 -/
theorem cost_of_3000_pencils :
  cost_of_pencils 200 40 0.1 1000 3000 = 540 := by
  -- Unfold the definition of cost_of_pencils
  unfold cost_of_pencils
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_3000_pencils_l500_50014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_l500_50073

structure Square where
  side : ℝ
  has_stars_and_crosses : Bool

structure SquarePart where
  area : ℝ
  has_star : Bool
  has_cross : Bool

def cut_square (s : Square) : List SquarePart :=
  sorry

theorem square_division (s : Square) 
  (h : s.has_stars_and_crosses) : 
  ∃ (parts : List SquarePart), 
    (parts.length = 4) ∧ 
    (∀ p ∈ parts, p.area = s.side^2 / 4) ∧
    (∀ p ∈ parts, p.has_star ∧ p.has_cross) ∧
    (cut_square s = parts) := by
  sorry

#check square_division

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_l500_50073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_in_special_triangle_l500_50070

/-- Given a triangle ABC where angles A, B, C form a geometric progression and b^2 - a^2 = ac, 
    prove that the measure of angle B is 2π/7 radians. -/
theorem angle_measure_in_special_triangle (A B C : ℝ) (a b c : ℝ) : 
  -- Triangle conditions
  A > 0 ∧ B > 0 ∧ C > 0 ∧ 
  A + B + C = Real.pi ∧
  -- Sides are positive
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  -- Law of sines
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C ∧
  -- Angles form a geometric progression
  ∃ (q : ℝ), q > 0 ∧ A = B / q ∧ C = q * B ∧
  -- Given condition
  b^2 - a^2 = a * c →
  -- Conclusion
  B = 2 * Real.pi / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_in_special_triangle_l500_50070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_correct_l500_50007

def sequenceAlt (n : ℕ+) : ℤ :=
  (-1)^(n.val + 1) * (2 * n.val - 1)

theorem sequence_formula_correct :
  ∀ (n : ℕ+), sequenceAlt n = (-1)^(n.val + 1) * (2 * n.val - 1) :=
by
  intro n
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_correct_l500_50007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l500_50086

-- Define the function f(x) = -1/x + log₂(x)
noncomputable def f (x : ℝ) : ℝ := -1/x + Real.log x / Real.log 2

-- Theorem statement
theorem root_in_interval :
  ∃ c ∈ Set.Ioo 1 2, f c = 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l500_50086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_subset_l500_50032

open Set Finset

theorem existence_of_special_subset (n : ℕ) : 
  ∃ (A : Finset ℕ), A ⊆ Finset.range (2^n + 1) ∧ 
  A.card = n ∧
  ∀ (S T : Finset ℕ), S ⊆ A → T ⊆ A → S ≠ ∅ → T ≠ ∅ → S ≠ T → 
    ¬(∃ k : ℕ, k * (S.sum id) = T.sum id) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_subset_l500_50032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_squared_sin_plus_cos_l500_50065

theorem sin_plus_cos_squared (θ : ℝ) (b : ℝ) 
  (h1 : 0 < θ ∧ θ < Real.pi / 2) -- θ is acute
  (h2 : Real.cos (2 * θ) = b) : 
  (Real.sin θ + Real.cos θ) ^ 2 = 2 - b := by
  sorry

theorem sin_plus_cos (θ : ℝ) (b : ℝ) 
  (h1 : 0 < θ ∧ θ < Real.pi / 2) -- θ is acute
  (h2 : Real.cos (2 * θ) = b) : 
  Real.sin θ + Real.cos θ = Real.sqrt (2 - b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_squared_sin_plus_cos_l500_50065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_transformation_l500_50076

/-- Given a quadratic equation ax^2 + bx + c = 0 with roots r and s,
    prove that cr+b and cs+b are roots of ax^2 + (bc - 2ab)x + c^3 - b^2c + ab^2 = 0 -/
theorem roots_transformation (a b c r s : ℝ) 
  (h1 : a ≠ 0)
  (h2 : a * r^2 + b * r + c = 0)
  (h3 : a * s^2 + b * s + c = 0) :
  let new_eq := λ x => a * x^2 + (b * c - 2 * a * b) * x + c^3 - b^2 * c + a * b^2
  new_eq (c * r + b) = 0 ∧ new_eq (c * s + b) = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_transformation_l500_50076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_female_fraction_after_growth_l500_50003

/-- The fraction of female participants in a basketball league after a year of growth -/
theorem female_fraction_after_growth (males_last_year : ℕ) 
  (total_growth_rate male_growth_rate female_growth_rate : ℚ) :
  males_last_year = 30 →
  total_growth_rate = 15/100 →
  male_growth_rate = 1/10 →
  female_growth_rate = 1/4 →
  (∃ females_last_year : ℕ,
    (males_last_year + females_last_year) * (1 + total_growth_rate) = 
    (males_last_year * (1 + male_growth_rate)).floor + 
    (females_last_year * (1 + female_growth_rate)).ceil) →
  (((females_last_year * (1 + female_growth_rate)).ceil : ℚ) / 
   ((males_last_year * (1 + male_growth_rate)).floor + 
    (females_last_year * (1 + female_growth_rate)).ceil : ℚ)) = 19/52 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_female_fraction_after_growth_l500_50003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_limit_l500_50072

noncomputable def my_sequence (n : ℝ) : ℝ := (5 * n^2 - 2) / ((n - 3) * (n + 1))

theorem my_sequence_limit : 
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |my_sequence n - 5| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_limit_l500_50072
