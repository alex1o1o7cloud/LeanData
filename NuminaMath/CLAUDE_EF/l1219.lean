import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1219_121999

/-- A real number is an angle bisector length if it satisfies the properties
    of an angle bisector in a triangle with the given semi-perimeter. -/
def Real.is_angle_bisector_length (l p : ℝ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  p = (a + b + c) / 2 ∧
  l = Real.sqrt (p * (p - c)) / (Real.sqrt (p * (p - b)) + Real.sqrt (p * (p - a)))

/-- A real number is a median length if it satisfies the properties
    of a median in a triangle with the given semi-perimeter. -/
def Real.is_median_length (m p : ℝ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  p = (a + b + c) / 2 ∧
  m ^ 2 = (2 * a ^ 2 + 2 * b ^ 2 - c ^ 2) / 4

/-- For any triangle ABC with semi-perimeter p, angle bisector lengths l_a and l_b,
    and median length m_c, the sum of these three lengths is less than or equal to
    √3 times the semi-perimeter. -/
theorem triangle_inequality (p l_a l_b m_c : ℝ) 
  (hp : p > 0)
  (hl_a : l_a > 0)
  (hl_b : l_b > 0)
  (hm_c : m_c > 0)
  (h_bisector_a : l_a.is_angle_bisector_length p)
  (h_bisector_b : l_b.is_angle_bisector_length p)
  (h_median_c : m_c.is_median_length p) :
  l_a + l_b + m_c ≤ Real.sqrt 3 * p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1219_121999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_travel_time_l1219_121966

/-- Represents the time taken to travel from home to school under different walking and running conditions -/
noncomputable def travel_time (total_distance : ℝ) (walk_speed : ℝ) (walk_distance : ℝ) (run_distance : ℝ) : ℝ :=
  walk_distance / walk_speed + run_distance / (2 * walk_speed)

theorem school_travel_time 
  (total_distance : ℝ) 
  (walk_speed : ℝ) 
  (h1 : walk_speed > 0) 
  (h2 : travel_time total_distance walk_speed (2 * total_distance / 3) (total_distance / 3) = 30) :
  travel_time total_distance walk_speed (total_distance / 3) (2 * total_distance / 3) = 24 :=
by
  sorry

#check school_travel_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_travel_time_l1219_121966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S₁_subset_S₂_iff_in_primeSet_l1219_121983

/-- S₁(p) is the set of integer triples (a,b,c) such that a²b² + b²c² + c²a² + 1 ≡ 0 (mod p) -/
def S₁ (p : ℕ) : Set (ℤ × ℤ × ℤ) :=
  {abc : ℤ × ℤ × ℤ | (abc.1^2 * abc.2.1^2 + abc.2.1^2 * abc.2.2^2 + abc.2.2^2 * abc.1^2 + 1) % p = 0}

/-- S₂(p) is the set of integer triples (a,b,c) such that a²b²c²(a²b²c² + a² + b² + c²) ≡ 0 (mod p) -/
def S₂ (p : ℕ) : Set (ℤ × ℤ × ℤ) :=
  {abc : ℤ × ℤ × ℤ | (abc.1^2 * abc.2.1^2 * abc.2.2^2 * (abc.1^2 * abc.2.1^2 * abc.2.2^2 + abc.1^2 + abc.2.1^2 + abc.2.2^2)) % p = 0}

/-- The set of primes for which S₁(p) is a subset of S₂(p) -/
def primeSet : Set ℕ := {2, 3, 5, 13, 17}

/-- For a prime number p, S₁(p) is a subset of S₂(p) if and only if p is in the set {2, 3, 5, 13, 17} -/
theorem S₁_subset_S₂_iff_in_primeSet (p : ℕ) (hp : Nat.Prime p) :
  S₁ p ⊆ S₂ p ↔ p ∈ primeSet := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S₁_subset_S₂_iff_in_primeSet_l1219_121983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1219_121912

noncomputable def f (x : ℝ) := Real.sin (x * Real.cos x)

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x, 0 < x ∧ x < Real.pi / 2 → f x > 0) ∧
  (∃ g : ℕ → ℝ, Filter.Tendsto g Filter.atTop Filter.atTop ∧ ∀ n, f (g n) = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1219_121912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_equals_minus_one_plus_three_i_l1219_121949

/-- The complex number z is defined as (2+4i)/(1-i) -/
noncomputable def z : ℂ := (2 + 4 * Complex.I) / (1 - Complex.I)

/-- Theorem stating that z is equal to -1 + 3i -/
theorem z_equals_minus_one_plus_three_i : z = -1 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_equals_minus_one_plus_three_i_l1219_121949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_C_l1219_121958

/-- Definition of the triangle ABC -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- Definition of the distance function -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem stating the trajectory of vertex C -/
theorem trajectory_of_C (ABC : Triangle) (h1 : ABC.A = (-3, 0)) (h2 : ABC.B = (3, 0))
    (h3 : distance ABC.C ABC.B - distance ABC.C ABC.A = 4) :
    ∃ (x y : ℝ), (x^2 / 4 - y^2 / 5 = 1) ∧ (x ≥ 2) ∧ ABC.C = (x, y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_C_l1219_121958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l1219_121977

-- Define the points and lines
def line_A : ℝ → ℝ → Prop := λ x y ↦ x - y = 0
def line_B : ℝ → ℝ → Prop := λ x y ↦ x + y = 0
def point_mid : ℝ × ℝ := (-1, 0)
def ray_midpoint : ℝ → ℝ → Prop := λ x y ↦ x - 2*y = 0 ∧ x ≤ 0

-- Define the conditions
structure Conditions where
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_A : line_A A.1 A.2
  h_B : line_B B.1 B.2
  h_mid : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ 
    (1 - t) * A.1 + t * B.1 = point_mid.1 ∧
    (1 - t) * A.2 + t * B.2 = point_mid.2
  h_ray : ray_midpoint ((A.1 + B.1) / 2) ((A.2 + B.2) / 2)

-- State the theorem
theorem length_of_AB (c : Conditions) :
  Real.sqrt ((c.A.1 - c.B.1)^2 + (c.A.2 - c.B.2)^2) = (4/3) * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l1219_121977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_half_l1219_121982

theorem tan_alpha_half (α : Real) (h : Real.tan α = 1/2) : 
  (Real.sin α + Real.cos α) / (2 * Real.sin α - 3 * Real.cos α) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_half_l1219_121982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_exists_l1219_121964

noncomputable section

-- Define the curves
def curve1 (x : ℝ) : ℝ := x^3
def curve2 (a x : ℝ) : ℝ := a*x^2 + (15/4)*x - 9

-- Define the tangent line
def tangent_line (k : ℝ) (x : ℝ) : ℝ := k*(x - 1)

theorem tangent_line_exists (a : ℝ) : 
  (∃ k : ℝ, 
    (∃ x₀ : ℝ, curve1 x₀ = tangent_line k x₀ ∧ 
                curve2 a x₀ = tangent_line k x₀) ∧
    tangent_line k 1 = 0) → 
  (a = -25/64 ∨ a = -1) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_exists_l1219_121964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bijection_exists_l1219_121929

/-- A bijection from ℕ to ℕ satisfying the given functional equation -/
noncomputable def f : ℕ → ℕ := sorry

/-- The functional equation that f must satisfy -/
axiom f_eq (m n : ℕ) : f (3 * m * n + m + n) = 4 * f m * f n + f m + f n

/-- f is injective -/
axiom f_injective : Function.Injective f

/-- f is surjective -/
axiom f_surjective : Function.Surjective f

/-- Theorem: There exists a bijection f: ℕ → ℕ satisfying the given functional equation -/
theorem bijection_exists : ∃ (f : ℕ → ℕ), 
  Function.Bijective f ∧ 
  (∀ m n : ℕ, f (3 * m * n + m + n) = 4 * f m * f n + f m + f n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bijection_exists_l1219_121929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_perpendiculars_equals_altitude_l1219_121902

-- Define an equilateral triangle
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

-- Define a point inside the triangle
structure PointInTriangle (t : EquilateralTriangle) where
  x : ℝ
  y : ℝ
  in_triangle : x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ t.side

-- Define the perpendicular distance from a point to a side
noncomputable def perpendicular_distance (t : EquilateralTriangle) (p : PointInTriangle t) (side : Fin 3) : ℝ :=
  sorry

-- Define the altitude of the triangle
noncomputable def altitude (t : EquilateralTriangle) : ℝ :=
  t.side * Real.sqrt 3 / 2

-- Theorem statement
theorem sum_of_perpendiculars_equals_altitude (t : EquilateralTriangle) (p : PointInTriangle t) :
  (perpendicular_distance t p 0) + (perpendicular_distance t p 1) + (perpendicular_distance t p 2) = altitude t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_perpendiculars_equals_altitude_l1219_121902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_number_proof_l1219_121984

theorem smaller_number_proof (a b : ℕ) 
  (h_ratio : a * 5 = b * 2)
  (h_lcm : Nat.lcm a b = 160) :
  a = 64 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_number_proof_l1219_121984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_vertical_asymptote_l1219_121930

noncomputable def f (c : ℝ) (x : ℝ) : ℝ := (x^2 - x + c) / (x^2 + x - 6)

theorem one_vertical_asymptote (c : ℝ) :
  (∃! x, ∀ ε > 0, ∃ δ > 0, ∀ y, |y - x| < δ ∧ y ≠ x → |f c y| > 1/ε) ↔ c = -2 ∨ c = -12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_vertical_asymptote_l1219_121930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_minus_d_squared_eq_zero_l1219_121987

/-- The number of positive multiples of 9 that are less than 60 -/
def c : ℕ := (Finset.filter (fun x => x % 9 = 0 ∧ x > 0) (Finset.range 60)).card

/-- The number of positive integers less than 60 that are multiples of both 9 and 3 -/
def d : ℕ := (Finset.filter (fun x => x % 9 = 0 ∧ x % 3 = 0 ∧ x > 0) (Finset.range 60)).card

theorem c_minus_d_squared_eq_zero : (c - d)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_minus_d_squared_eq_zero_l1219_121987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersection_theorem_l1219_121952

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a plane -/
def Point := ℝ × ℝ

/-- Define membership for a point in a circle -/
def pointInCircle (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 ≤ c.radius^2

instance : Membership Point Circle where
  mem := pointInCircle

/-- Given a set of circles, returns whether any three of them have a common point -/
def anyThreeIntersect (circles : Set Circle) : Prop :=
  ∀ c₁ c₂ c₃, c₁ ∈ circles → c₂ ∈ circles → c₃ ∈ circles →
    ∃ p : Point, p ∈ c₁ ∧ p ∈ c₂ ∧ p ∈ c₃

/-- Given a set of circles, returns whether all of them have a common point -/
def allIntersect (circles : Set Circle) : Prop :=
  ∃ p : Point, ∀ c ∈ circles, p ∈ c

/-- The main theorem -/
theorem circles_intersection_theorem (circles : Set Circle) (h : Fintype circles) :
  (Fintype.card circles ≥ 5) →
  anyThreeIntersect circles →
  allIntersect circles := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersection_theorem_l1219_121952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_integer_product_l1219_121997

theorem smallest_odd_integer_product : ∃ (m : ℕ), 
  Odd m ∧ 
  (∀ k < m, Odd k → (3 : ℝ) ^ ((k + 1)^2 / 9) ≤ 5000) ∧
  (3 : ℝ) ^ ((m + 1)^2 / 9) > 5000 ∧
  m = 8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_integer_product_l1219_121997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airline_route_within_republic_l1219_121967

/-- A city in the country -/
structure City where
  id : Nat
  republic : Nat
  routes : Nat
  deriving Repr, DecidableEq

/-- The country setup -/
structure Country where
  cities : Finset City
  num_republics : Nat

/-- Predicate to check if a city is a "millionaire city" -/
def is_millionaire (c : City) : Bool := c.routes ≥ 70

/-- The main theorem -/
theorem airline_route_within_republic (country : Country)
  (h1 : country.cities.card = 100)
  (h2 : country.num_republics = 3)
  (h3 : (country.cities.filter (fun c => is_millionaire c)).card ≥ 70) :
  ∃ (c1 c2 : City), c1 ∈ country.cities ∧ c2 ∈ country.cities ∧ 
    c1.republic = c2.republic ∧ c1 ≠ c2 := by
  sorry

#check airline_route_within_republic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_airline_route_within_republic_l1219_121967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_germination_probability_approximation_l1219_121926

/-- The probability of a specific number of seeds germinating given a total number of seeds and germination rate -/
noncomputable def germination_probability (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  let q := 1 - p
  let npq := n * p * q
  let x := (k - n * p) / Real.sqrt npq
  Real.exp (-(x^2) / 2) / (Real.sqrt (2 * Real.pi * npq))

theorem germination_probability_approximation :
  let n := 400
  let p := 0.9
  let k := 350
  abs (germination_probability n p k - 0.0167) < 0.0001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_germination_probability_approximation_l1219_121926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longer_leg_of_smallest_triangle_l1219_121910

/-- Represents a 30-60-90 triangle with its hypotenuse length -/
structure Triangle30_60_90 where
  hypotenuse : ℝ

/-- Calculates the length of the longer leg of a 30-60-90 triangle -/
noncomputable def longerLeg (t : Triangle30_60_90) : ℝ := t.hypotenuse * (Real.sqrt 3) / 2

/-- Represents a sequence of four 30-60-90 triangles where each triangle's hypotenuse
    is the longer leg of the adjacent smaller triangle -/
structure TriangleSequence where
  t1 : Triangle30_60_90
  t2 : Triangle30_60_90
  t3 : Triangle30_60_90
  t4 : Triangle30_60_90
  h2 : t2.hypotenuse = longerLeg t1
  h3 : t3.hypotenuse = longerLeg t2
  h4 : t4.hypotenuse = longerLeg t3

/-- Theorem stating that the longer leg of the smallest triangle is 9/2 when the
    hypotenuse of the largest triangle is 8 -/
theorem longer_leg_of_smallest_triangle (seq : TriangleSequence) 
    (h : seq.t1.hypotenuse = 8) : 
    longerLeg seq.t4 = 9 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longer_leg_of_smallest_triangle_l1219_121910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_absolute_value_equation_l1219_121974

/-- The equation |2x - 1| + |x - 2| = |x + 1| has infinitely many solutions -/
theorem infinite_solutions_absolute_value_equation :
  ∃ S : Set ℝ, (∀ x ∈ S, |2*x - 1| + |x - 2| = |x + 1|) ∧ Set.Infinite S :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_absolute_value_equation_l1219_121974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_intersection_range_inequality_proof_l1219_121934

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x - m * (x + 1) * Real.log (x + 1)

-- Theorem for monotonicity intervals
theorem monotonicity_intervals (m : ℝ) (h : m ≥ 0) :
  (m = 0 → ∀ x > -1, Monotone (f m)) ∧
  (m > 0 → ∃ y > -1, StrictMonoOn (f m) (Set.Icc (-1) y) ∧ StrictAntiOn (f m) (Set.Ici y)) :=
sorry

-- Theorem for the range of t when m = 1
theorem intersection_range :
  ∃ t_min t_max : ℝ, t_min < t_max ∧
  ∀ t : ℝ, (∃ x y : ℝ, -1/2 ≤ x ∧ x < y ∧ y ≤ 1 ∧ f 1 x = t ∧ f 1 y = t) ↔ t_min ≤ t ∧ t < t_max :=
sorry

-- Theorem for the inequality (1+a)^b < (1+b)^a
theorem inequality_proof (a b : ℝ) (h : a > b ∧ b > 0) :
  (1 + a) ^ b < (1 + b) ^ a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_intersection_range_inequality_proof_l1219_121934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_and_inequality_l1219_121939

-- Define the function f(x) = b * a^x
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := b * (a ^ x)

-- State the theorem
theorem exponential_function_and_inequality 
  (a b : ℝ) 
  (ha : a > 0) 
  (ha_neq : a ≠ 1) 
  (hf1 : f a b 1 = 6) 
  (hf3 : f a b 3 = 24) :
  (∃ c : ℝ, c = 3 ∧ f 2 c = f a b) ∧ 
  (∀ m : ℝ, m ≤ 5/6 → ∀ x : ℝ, x ≤ 1 → (1/a)^x + (1/b)^x - m ≥ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_and_inequality_l1219_121939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_equality_periodic_l1219_121985

theorem tan_equality_periodic (n : ℤ) : 
  -90 < n ∧ n < 90 → Real.tan (n * π / 180) = Real.tan (1234 * π / 180) → n = -26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_equality_periodic_l1219_121985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1219_121913

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) / (x - 2)

-- State the theorem about the domain of f
theorem f_domain :
  ∀ x : ℝ, (x ≥ -1 ∧ x ≠ 2) ↔ (∃ y : ℝ, f x = y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1219_121913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l1219_121979

/-- Predicate for an isosceles triangle -/
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = b ∧ a ≠ c) ∨ (b = c ∧ b ≠ a) ∨ (a = c ∧ a ≠ b)

/-- An isosceles triangle with side lengths 4 and 8 has a perimeter of 20. -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 4 →
  b = 8 →
  c = 8 →
  IsoscelesTriangle a b c →
  a + b + c = 20 :=
by
  intros a b c h1 h2 h3 h4
  rw [h1, h2, h3]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l1219_121979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_meaningful_range_l1219_121968

-- Define the expression as noncomputable
noncomputable def f (x : ℝ) := (Real.sqrt (x - 3)) / 2

-- Theorem stating the range of x for which f is meaningful
theorem f_meaningful_range :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_meaningful_range_l1219_121968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_m_equals_two_l1219_121917

theorem subset_implies_m_equals_two (m : ℤ) : 
  let A : Set ℤ := {1, 3, m + 2}
  let B : Set ℤ := {3, m^2}
  B ⊆ A → m = 2 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_m_equals_two_l1219_121917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_roll_path_length_for_2cm_cube_l1219_121954

/-- The length of the path traced by the center of a face of a cube when it rolls over three edges -/
noncomputable def cube_roll_path_length (edge_length : ℝ) : ℝ :=
  (1 + 4 * Real.pi/4) * edge_length

/-- Theorem: The path length for a cube with edge length 2 cm is 5π cm -/
theorem cube_roll_path_length_for_2cm_cube :
  cube_roll_path_length 2 = 5 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_roll_path_length_for_2cm_cube_l1219_121954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_fraction_proof_l1219_121903

theorem missing_fraction_proof (total_sum : ℚ) 
  (h1 : total_sum = 5/6) : ℚ := by
  let given_fractions : List ℚ := [1/3, 1/2, 1/5, 1/4, -9/20, -5/6]
  let missing_fraction : ℚ := total_sum - given_fractions.sum
  have h2 : missing_fraction = 5/6 := by
    sorry
  exact missing_fraction

#eval (5 : ℚ) / 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_fraction_proof_l1219_121903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_tangent_length_l1219_121986

-- Define the circles and points
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

def O : ℝ × ℝ := (0, 0)
def P : ℝ × ℝ := (13, 0)
def Q : ℝ × ℝ := (8, 0)

-- Define the radii of the circles
def r_O : ℝ := 8
def r_P : ℝ := 5

-- Define a line through two points
def Line (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {x : ℝ × ℝ | ∃ t : ℝ, x = (1 - t) • p + t • q}

-- State the theorem
theorem external_tangent_length (T S : ℝ × ℝ) :
  -- Conditions
  T ∈ Circle O r_O →
  S ∈ Circle P r_P →
  Q ∈ Circle O r_O →
  Q ∈ Circle P r_P →
  (∀ x : ℝ × ℝ, x ∈ Line T S → x ∉ Circle O r_O ∧ x ∉ Circle P r_P) →
  -- Conclusion
  dist O T = r_O := by
  sorry

#check external_tangent_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_tangent_length_l1219_121986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_distance_from_ceiling_l1219_121994

/-- The distance of a point (x, y, z) from the origin (0, 0, 0) in 3D space --/
noncomputable def distance_from_origin (x y z : ℝ) : ℝ := Real.sqrt (x^2 + y^2 + z^2)

/-- Theorem: Given a point (2, 7, z) that is 10 units away from the origin,
    prove that z = √47 --/
theorem fly_distance_from_ceiling :
  ∃ z : ℝ, distance_from_origin 2 7 z = 10 ∧ z = Real.sqrt 47 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_distance_from_ceiling_l1219_121994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_angle_l1219_121905

/-- The angle between the asymptotes of the hyperbola 3y^2 - x^2 = 1 is π/3 -/
theorem hyperbola_asymptote_angle :
  ∃ (θ : ℝ), θ = π/3 ∧ 
  θ = Real.arctan (Real.sqrt 3 / 3) - Real.arctan (-Real.sqrt 3 / 3) :=
by
  -- We'll use the fact that the asymptotes are y = ±(√3/3)x
  -- The angle between these lines is the difference of their arctangents
  use π/3
  constructor
  · rfl  -- Proves θ = π/3
  · sorry  -- The actual proof would go here, but we'll use sorry for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_angle_l1219_121905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_consumption_reduction_l1219_121901

theorem gas_consumption_reduction (initial_price : ℝ) (initial_consumption : ℝ) 
  (h1 : initial_price > 0) (h2 : initial_consumption > 0) : 
  (1 - initial_price * initial_consumption / (initial_price * 1.25 * 1.1 * initial_consumption)) * 100 = (1 - 1 / 1.375) * 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_consumption_reduction_l1219_121901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l1219_121960

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides of lengths 20 cm and 18 cm, 
    and a distance of 12 cm between them, is equal to 228 cm². -/
theorem trapezium_area_example : trapeziumArea 20 18 12 = 228 := by
  -- Unfold the definition of trapeziumArea
  unfold trapeziumArea
  -- Simplify the arithmetic
  simp [add_mul, mul_div_assoc]
  -- Check that the result is equal to 228
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l1219_121960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_supermarket_distance_is_five_l1219_121993

/-- Represents Bobby's trip details -/
structure TripDetails where
  initial_gas : ℚ
  final_gas : ℚ
  consumption_rate : ℚ
  farm_distance : ℚ
  partial_farm_trip : ℚ

/-- Calculates the distance to the supermarket given Bobby's trip details -/
def distance_to_supermarket (trip : TripDetails) : ℚ :=
  let total_distance := (trip.initial_gas - trip.final_gas) * trip.consumption_rate
  let farm_trips_distance := 2 * trip.partial_farm_trip + trip.farm_distance
  (total_distance - farm_trips_distance) / 2

/-- Theorem stating that the distance to the supermarket is 5 miles -/
theorem supermarket_distance_is_five (trip : TripDetails) 
  (h1 : trip.initial_gas = 12)
  (h2 : trip.final_gas = 2)
  (h3 : trip.consumption_rate = 2)
  (h4 : trip.farm_distance = 6)
  (h5 : trip.partial_farm_trip = 2) : 
  distance_to_supermarket trip = 5 := by
  sorry

def main : IO Unit := do
  let result := distance_to_supermarket {
    initial_gas := 12, 
    final_gas := 2, 
    consumption_rate := 2, 
    farm_distance := 6, 
    partial_farm_trip := 2
  }
  IO.println s!"The distance to the supermarket is {result} miles"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_supermarket_distance_is_five_l1219_121993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_quadrilateral_classification_l1219_121945

/-- A quadrilateral is defined by its four angles -/
structure Quadrilateral where
  α : Real
  β : Real
  γ : Real
  δ : Real

/-- Properties of a special quadrilateral -/
def IsSpecialQuadrilateral (q : Quadrilateral) : Prop :=
  Real.cos q.α + Real.cos q.β + Real.cos q.γ + Real.cos q.δ = 0 ∧
  q.α + q.β + q.γ + q.δ = 2 * Real.pi

/-- Definition of a parallelogram or trapezoid -/
def IsParallelogramOrTrapezoid (q : Quadrilateral) : Prop :=
  q.α + q.β = Real.pi ∨ q.β + q.γ = Real.pi ∨ q.γ + q.δ = Real.pi ∨ q.δ + q.α = Real.pi

/-- Definition of an inscribed quadrilateral -/
def IsInscribed (q : Quadrilateral) : Prop :=
  q.α + q.γ = Real.pi ∨ q.β + q.δ = Real.pi

theorem special_quadrilateral_classification (q : Quadrilateral) :
  IsSpecialQuadrilateral q → IsParallelogramOrTrapezoid q ∨ IsInscribed q := by
  sorry

#check special_quadrilateral_classification

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_quadrilateral_classification_l1219_121945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_example_l1219_121919

/-- The sum of an infinite geometric series with first term a and common ratio r -/
noncomputable def geometricSeriesSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Theorem: The sum of the infinite geometric series with first term 1 and common ratio 2/3 is 3 -/
theorem geometric_series_sum_example : geometricSeriesSum 1 (2/3) = 3 := by
  -- Unfold the definition of geometricSeriesSum
  unfold geometricSeriesSum
  -- Simplify the expression
  simp
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_example_l1219_121919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_of_f_l1219_121944

noncomputable def f (x a : ℝ) : ℝ := 4 * x + a / x

theorem minimum_value_of_f (a : ℝ) (ha : a > 0) :
  (∀ x > 0, f x a ≥ f 3 a) →
  a = 36 ∧ f 3 a = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_of_f_l1219_121944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_existence_l1219_121969

-- Define the function f(x) = 3/x - log₂(x)
noncomputable def f (x : ℝ) : ℝ := 3 / x - Real.log x / Real.log 2

-- State the theorem
theorem zero_point_existence :
  ∃ c : ℝ, 2 < c ∧ c < 3 ∧ f c = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_existence_l1219_121969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_max_value_l1219_121990

/-- The minimum of three real numbers -/
noncomputable def M (x y z : ℝ) : ℝ := min x (min y z)

/-- A quadratic function with positive coefficients -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_max_value 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hroots : ∃ x, f a b c x = 0) : 
  (∀ x y z : ℝ, M x y z ≤ 1) ∧ 
  (∃ x y z : ℝ, M x y z = 1 ∧ 
    x = (b + c) / a ∧ 
    y = (c + a) / b ∧ 
    z = (a + b) / c) := by
  sorry

#check quadratic_max_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_max_value_l1219_121990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1219_121971

-- Define the ⊙ operation
noncomputable def odot (a b : ℝ) : ℝ := if a ≥ b then b else a

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := odot x (2 - x)

-- Theorem statement
theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1219_121971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_phase_shift_l1219_121941

noncomputable def phase_shift (B : ℝ) (C : ℝ) : ℝ := C / B

theorem sine_phase_shift (x : ℝ) :
  let f : ℝ → ℝ := fun x => Real.sin (3 * x - Real.pi)
  let shift := phase_shift 3 Real.pi
  (∀ x, f (x + shift) = Real.sin (3 * x)) ∨
  (∀ x, f (x - shift) = Real.sin (3 * x)) := by
  sorry

#check sine_phase_shift

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_phase_shift_l1219_121941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_7_l1219_121957

noncomputable def a (n : ℕ) : ℝ :=
  13 - 2 * (n - 1)

noncomputable def S (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem max_sum_at_7 :
  ∀ n : ℕ, n ≠ 0 → S 7 ≥ S n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_7_l1219_121957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sequences_count_l1219_121922

def letters : List Char := ['T', 'R', 'I', 'A', 'N', 'G', 'L', 'E']

def validSequence (s : List Char) : Bool :=
  s.length = 6 && 
  s.head? = some 'T' && 
  s.getLast? = some 'E' && 
  s.toFinset.card = 6 &&
  s.toFinset ⊆ letters.toFinset

theorem distinct_sequences_count : 
  (List.filter validSequence (List.permutations letters)).length = 360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sequences_count_l1219_121922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_points_count_l1219_121900

noncomputable def regular_polygon_vertices (n : ℕ+) : Fin n → ℂ :=
  λ k => Complex.exp (2 * Real.pi * Complex.I * (k.val : ℂ) / (n : ℂ))

noncomputable def exponentiated_vertices (n : ℕ+) : Fin n → ℂ :=
  λ k => if k.val = 0 then (regular_polygon_vertices n k) ^ (2010.5 : ℂ)
         else (regular_polygon_vertices n k) ^ (2005 : ℂ)

theorem distinct_points_count :
  Finset.card (Finset.image (exponentiated_vertices 20) (Finset.univ : Finset (Fin 20))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_points_count_l1219_121900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_40_cents_l1219_121937

/-- Represents the value of a coin in cents -/
def CoinValue : Type := Nat

/-- The set of coins being flipped -/
def Coins : Finset Nat := {1, 5, 10, 25, 50}

/-- The minimum value in cents required to be considered a success -/
def MinimumValue : Nat := 40

/-- The number of coins being flipped -/
def NumCoins : Nat := Finset.card Coins

/-- The total number of possible outcomes when flipping the coins -/
def TotalOutcomes : Nat := 2^NumCoins

/-- A function that calculates the number of successful outcomes -/
def SuccessfulOutcomes : Nat := 18

/-- The probability of getting at least 40 cents worth of coins as heads -/
theorem probability_at_least_40_cents : 
  (SuccessfulOutcomes : ℚ) / TotalOutcomes = 9 / 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_40_cents_l1219_121937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_run_around_square_field_l1219_121963

-- Define the side length of the square field in meters
noncomputable def side_length : ℝ := 55

-- Define the boy's speed in km/hr
noncomputable def speed_km_hr : ℝ := 9

-- Convert km/hr to m/s
noncomputable def speed_m_s : ℝ := speed_km_hr * 1000 / 3600

-- Calculate the perimeter of the square field
noncomputable def perimeter : ℝ := 4 * side_length

-- Define the time taken to run around the field
noncomputable def time_taken : ℝ := perimeter / speed_m_s

-- Theorem to prove
theorem run_around_square_field : 
  time_taken = 88 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_run_around_square_field_l1219_121963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_and_perpendicular_bisector_l1219_121904

/-- Given two points A and B in the plane, this theorem proves:
    1. The coordinates of the midpoint D
    2. The equation of the perpendicular bisector of AB -/
theorem midpoint_and_perpendicular_bisector 
  (A B : ℝ × ℝ) 
  (hA : A = (0, -6)) 
  (hB : B = (1, -5)) :
  let D := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (D = (1/2, -11/2) ∧
   ∀ (x y : ℝ), (x + y + 5 = 0) ↔ 
     ((x - D.1) * (B.1 - A.1) = (D.2 - y) * (B.2 - A.2))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_and_perpendicular_bisector_l1219_121904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cylinder_volume_ratio_l1219_121906

/-- The ratio of the volume of a cone to the volume of a cylinder with the same base radius,
    where the cone's height is one-third the height of the cylinder, is 1/9. -/
theorem cone_cylinder_volume_ratio :
  ∀ (r h_cylinder : ℝ),
  r > 0 → h_cylinder > 0 →
  let h_cone := (1 / 3 : ℝ) * h_cylinder
  let v_cone := (1 / 3 : ℝ) * π * r^2 * h_cone
  let v_cylinder := π * r^2 * h_cylinder
  v_cone / v_cylinder = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cylinder_volume_ratio_l1219_121906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1219_121916

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x - Real.pi / 6)

theorem f_properties :
  (∃ p : ℝ, p > 0 ∧ (∀ x : ℝ, f (x + p) = f x) ∧
    (∀ q : ℝ, q > 0 ∧ (∀ x : ℝ, f (x + q) = f x) → p ≤ q)) ∧
  f (Real.pi / 5) = f (-3 * Real.pi / 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1219_121916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_estimate_to_58_times_41_l1219_121950

theorem closest_estimate_to_58_times_41 : 
  let actual : ℤ := 58 * 41
  let estimate : ℤ := 60 * 40
  ∀ x ∈ ({50 * 40, 60 * 50} : Set ℤ), |actual - estimate| < |actual - x| :=
by
  intro actual estimate x hx
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_estimate_to_58_times_41_l1219_121950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1219_121927

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ) + Real.sqrt 3 * Real.cos (x + φ)

theorem function_properties (φ : ℝ) 
  (h1 : 0 < |φ| ∧ |φ| < π)
  (h2 : ∀ x ∈ Set.Icc 0 (π/4), ∀ y ∈ Set.Icc 0 (π/4), x < y → f x φ > f y φ)
  (h3 : ∀ x : ℝ, f x φ = f (π/2 - x) φ) :
  φ = 2*π/3 ∧ 
  (∀ x : ℝ, f (x + π/3) φ = 2 * Real.sin (2*x - π/3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1219_121927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_of_a_l1219_121976

theorem divisor_of_a (a b c d : ℕ+) 
  (h1 : Nat.gcd a.val b.val = 30)
  (h2 : Nat.gcd b.val c.val = 42)
  (h3 : Nat.gcd c.val d.val = 66)
  (h4 : Nat.lcm c.val d.val = 2772)
  (h5 : 100 < Nat.gcd d.val a.val ∧ Nat.gcd d.val a.val < 150) :
  13 ∣ a.val := by
  sorry

#check divisor_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_of_a_l1219_121976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_4026_l1219_121918

theorem sum_of_divisors_4026 : 
  (Finset.filter (λ n : ℕ ↦ 4026 % n = 0) (Finset.range (4026 + 1))).sum id = 2976 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_4026_l1219_121918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1219_121991

noncomputable def triangle (A B C : ℝ × ℝ) : Prop := True

noncomputable def length (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

noncomputable def angle (p q r : ℝ × ℝ) : ℝ := sorry

theorem triangle_side_length (A B C : ℝ × ℝ) :
  triangle A B C →
  length A C = 2 →
  length B C = Real.sqrt 7 →
  angle B A C = π / 3 →
  length A B = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1219_121991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_R_l1219_121938

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the region R
def R : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1 + p.2 + (floor p.1 : ℝ) + (floor p.2 : ℝ) ≤ 7}

-- State the theorem
theorem area_of_R : MeasureTheory.volume R = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_R_l1219_121938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_in_expansion_l1219_121998

/-- Given that the sum of coefficients in the expansion of (3x-1)(x+1)^n is 64,
    prove that the coefficient of x^4 in this expansion is 25 -/
theorem coefficient_x4_in_expansion (n : ℕ) : 
  (∀ x : ℝ, (3*x - 1) * (x + 1)^n = 64) →
  (∃ a b c d e f : ℝ, (3*x - 1) * (x + 1)^n = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f ∧ b = 25) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_in_expansion_l1219_121998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1219_121915

-- Define f as a noncomputable function
noncomputable section
  variable (f : ℝ → ℝ)

  -- Axioms
  axiom add_hom : ∀ x y : ℝ, f (x + y) = f x + f y
  axiom pos_val : ∀ x : ℝ, x > 0 → f x > 0
  axiom f_one : f 1 = (1 : ℝ) / 2

  -- Theorem
  theorem f_properties :
    (∀ x : ℝ, f (-x) = -f x) ∧
    (f (-2) = -1) ∧
    (f 6 = 3) := by
    sorry
end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1219_121915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_bounded_above_l1219_121907

open Set
open Real

-- Define the function f on the open interval (0,1)
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_bounded_above
  (h1 : ∀ x ∈ Set.Ioo 0 1, f x > 0)
  (h2 : ∀ x ∈ Set.Ioo 0 1, f ((2*x)/(1+x^2)) = 2 * f x) :
  ∃ M > 0, ∀ x ∈ Set.Ioo 0 1, f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_bounded_above_l1219_121907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1219_121914

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (a + 1)^(1 - x)

-- Define the condition for a decreasing function on an interval
def is_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- State the theorem
theorem a_range (a : ℝ) :
  a > -1 ∧ a ≠ 0 ∧
  is_decreasing (f a) 1 2 ∧
  is_decreasing (g a) 1 2 →
  0 < a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1219_121914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l1219_121992

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.arcsin (Real.sqrt (1 - Real.exp x))

-- State the theorem
theorem f_derivative (x : ℝ) (h : x ≤ 0) :
  deriv f x = -Real.exp (x/2) / (2 * Real.sqrt (1 - Real.exp x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l1219_121992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_b_to_a_range_l1219_121911

/-- Given a triangle ABC where angle B is twice angle A, 
    prove that the ratio of side b to side a is between 1 and 2 exclusively -/
theorem ratio_b_to_a_range (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : A + B + C = Real.pi)
  (h_positive : A > 0 ∧ B > 0 ∧ C > 0)
  (h_B_twice_A : B = 2 * A)
  (h_sine_law : b / Real.sin B = a / Real.sin A) : 
  1 < b / a ∧ b / a < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_b_to_a_range_l1219_121911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_logarithmic_equation_l1219_121962

noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem product_of_logarithmic_equation (x y : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0)
  (h_eq : Real.sqrt (log x) + Real.sqrt (log y) + 2 * log (Real.sqrt x) + 2 * log (Real.sqrt y) = 84)
  (h_int_1 : ∃ m : ℤ, Real.sqrt (log x) = m)
  (h_int_2 : ∃ n : ℤ, Real.sqrt (log y) = n)
  (h_int_3 : ∃ p : ℤ, log (Real.sqrt x) = p)
  (h_int_4 : ∃ q : ℤ, log (Real.sqrt y) = q) :
  x * y = (10 : ℝ) ^ 72 := by
  sorry

#eval (10 : ℝ) ^ 72

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_logarithmic_equation_l1219_121962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daily_average_rainfall_is_four_l1219_121955

/-- Represents the rainfall data for a week --/
structure WeeklyRainfall where
  monday : ℚ
  tuesday : ℚ
  wednesday : ℚ
  thursday : ℚ
  friday : ℚ

/-- Calculates the daily average rainfall for the week --/
def dailyAverageRainfall (w : WeeklyRainfall) : ℚ :=
  (w.monday + w.tuesday + w.wednesday + w.thursday + w.friday) / 5

/-- The given rainfall data for the week --/
def givenWeek : WeeklyRainfall where
  monday := 3
  tuesday := 6
  wednesday := 0
  thursday := 1
  friday := 10

/-- Theorem stating that the daily average rainfall for the given week is 4 inches --/
theorem daily_average_rainfall_is_four :
  dailyAverageRainfall givenWeek = 4 := by
  -- Unfold the definition of dailyAverageRainfall
  unfold dailyAverageRainfall
  -- Unfold the definition of givenWeek
  unfold givenWeek
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_daily_average_rainfall_is_four_l1219_121955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_miss_tree_class_size_l1219_121935

/-- The number of children between Anna and Zara going anti-clockwise -/
def n : ℕ := sorry

/-- The total number of children in Miss Tree's class -/
def total_children : ℕ := 7 * n + 2

theorem miss_tree_class_size :
  (20 < total_children) ∧
  (total_children < 30) ∧
  (n > 0) →
  total_children = 23 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_miss_tree_class_size_l1219_121935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tour_routes_count_l1219_121975

def number_of_cities : ℕ := 7
def cities_to_choose : ℕ := 5
def mandatory_cities : ℕ := 2

theorem tour_routes_count :
  (number_of_cities.choose (cities_to_choose - mandatory_cities)) *
  Nat.factorial (cities_to_choose - mandatory_cities) *
  (cities_to_choose - 1) = 600 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tour_routes_count_l1219_121975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_explicit_formula_l1219_121908

-- Define the custom operation ⊙
noncomputable def odot (a b : ℝ) : ℝ :=
  if a ≥ b then b else a

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := odot x (2 - x)

-- Theorem statement
theorem f_explicit_formula :
  ∀ x : ℝ, f x = if x ≥ 1 then 2 - x else x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_explicit_formula_l1219_121908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_always_simplest_l1219_121980

theorem fraction_always_simplest (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_always_simplest_l1219_121980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_nice_number_is_correct_l1219_121943

/-- A digit is either 1 or 2 -/
inductive NiceDigit
  | one : NiceDigit
  | two : NiceDigit

/-- A nice number is a list of NiceDigits -/
def NiceNumber := List NiceDigit

/-- Convert a NiceNumber to a natural number -/
def niceNumberToNat (n : NiceNumber) : ℕ := sorry

/-- Get all 3-digit subsequences of a NiceNumber -/
def threeDigitSubsequences (n : NiceNumber) : List (List NiceDigit) := sorry

/-- Check if all 3-digit subsequences are distinct -/
def hasDistinctSubsequences (n : NiceNumber) : Prop := sorry

/-- Definition of a nice number -/
def isNice (n : NiceNumber) : Prop :=
  hasDistinctSubsequences n

/-- The greatest nice number -/
def greatestNiceNumber : NiceNumber := sorry

theorem greatest_nice_number_is_correct :
  isNice greatestNiceNumber ∧
  niceNumberToNat greatestNiceNumber = 2221211122 ∧
  (threeDigitSubsequences greatestNiceNumber).length = 8 ∧
  ∀ n : NiceNumber, isNice n → niceNumberToNat n ≤ niceNumberToNat greatestNiceNumber := by
  sorry

#eval "The proof is complete."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_nice_number_is_correct_l1219_121943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_magnitude_l1219_121946

theorem projection_magnitude (a b : ℝ × ℝ) : 
  a = (1, 3) → b = (-2, 1) → 
  ‖((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) • b‖ = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_magnitude_l1219_121946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_between_one_sixth_and_one_fourth_l1219_121936

theorem fraction_between_one_sixth_and_one_fourth :
  ∃! x : ℚ, x ∈ [5/12, 5/36, 5/24, 5/60, 5/48] ∧ 1/6 < x ∧ x < 1/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_between_one_sixth_and_one_fourth_l1219_121936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagram_arrangement_count_l1219_121973

/-- A hexagram is a regular six-pointed star -/
structure Hexagram where
  points : Fin 12 → Point

/-- An arrangement is a mapping from the points of a hexagram to objects -/
def Arrangement (n : ℕ) := Fin n → Fin n

/-- The group of symmetries of a hexagram -/
def HexagramSymmetries : Type := sorry

/-- The number of elements in the symmetry group of a hexagram -/
def num_hexagram_symmetries : ℕ := 12

/-- The number of distinct arrangements of n different objects on a hexagram,
    considering rotational and reflectional symmetries -/
def num_distinct_arrangements (n : ℕ) : ℕ :=
  Nat.factorial n / num_hexagram_symmetries

theorem hexagram_arrangement_count :
  num_distinct_arrangements 12 = 39916800 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagram_arrangement_count_l1219_121973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_eq_four_plus_four_sqrt_five_l1219_121972

/-- A square with side length 4 units -/
structure Square :=
  (side_length : ℝ)
  (is_four : side_length = 4)

/-- The midpoint of a side of the square -/
structure Midpoint :=
  (x : ℝ)
  (y : ℝ)

/-- Calculate the distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The sum of distances from the bottom-left vertex to all midpoints -/
noncomputable def sum_distances (s : Square) (m1 m2 m3 m4 : Midpoint) : ℝ :=
  distance 0 0 m1.x m1.y +
  distance 0 0 m2.x m2.y +
  distance 0 0 m3.x m3.y +
  distance 0 0 m4.x m4.y

/-- Theorem: The sum of distances from the bottom-left vertex to the midpoints is 4 + 4√5 -/
theorem sum_distances_eq_four_plus_four_sqrt_five (s : Square) 
  (m1 m2 m3 m4 : Midpoint) : sum_distances s m1 m2 m3 m4 = 4 + 4 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_eq_four_plus_four_sqrt_five_l1219_121972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_DEF_l1219_121909

-- Define the points
def D : ℚ × ℚ := (2, 5)

-- Define reflection over y-axis
def reflect_y (p : ℚ × ℚ) : ℚ × ℚ := (-p.1, p.2)

-- Define reflection over y = -x
def reflect_neg_x (p : ℚ × ℚ) : ℚ × ℚ := (-p.2, -p.1)

-- Define E as reflection of D over y-axis
def E : ℚ × ℚ := reflect_y D

-- Define F as reflection of E over y = -x
def F : ℚ × ℚ := reflect_neg_x E

-- Define the area of a triangle given three points
noncomputable def triangle_area (p1 p2 p3 : ℚ × ℚ) : ℚ :=
  (1/2) * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

-- Theorem statement
theorem area_of_DEF : triangle_area D E F = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_DEF_l1219_121909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_password_factorization_l1219_121961

/-- Given a polynomial x^3 + (m-n)x^2 + nx that can be factored as x(x+2)(x+3) when x = 10,
    prove that m = 11 and n = 6 -/
theorem password_factorization (m n : ℤ) :
  (fun x : ℤ ↦ x^3 + (m-n)*x^2 + n*x) = (fun x : ℤ ↦ x*(x+2)*(x+3)) →
  m = 11 ∧ n = 6 := by
  sorry

#check password_factorization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_password_factorization_l1219_121961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l1219_121965

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_magnitude (a b : V) 
  (h1 : ‖a - b‖ = Real.sqrt 3)
  (h2 : ‖a + b‖ = ‖(2 : ℝ) • a - b‖) : 
  ‖b‖ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l1219_121965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_modification_theorem_l1219_121953

/-- Represents the characteristics and performance of a car before and after modification -/
structure Car where
  fuelEfficiency : ℚ  -- Miles per gallon
  tankCapacity : ℚ    -- Gallons
  fuelUsageReduction : ℚ  -- Percentage of original fuel usage after modification

/-- Calculates the increase in travel distance after modification -/
def increasedTravelDistance (car : Car) : ℚ :=
  let originalDistance := car.fuelEfficiency * car.tankCapacity
  let newEfficiency := car.fuelEfficiency / car.fuelUsageReduction
  let newDistance := newEfficiency * car.tankCapacity
  newDistance - originalDistance

/-- Theorem stating the increased travel distance for the given car specifications -/
theorem car_modification_theorem (car : Car) 
    (h1 : car.fuelEfficiency = 27)
    (h2 : car.tankCapacity = 14)
    (h3 : car.fuelUsageReduction = 3/4) :
    increasedTravelDistance car = 189/2 := by
  sorry

#eval increasedTravelDistance { fuelEfficiency := 27, tankCapacity := 14, fuelUsageReduction := 3/4 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_modification_theorem_l1219_121953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_number_in_scientific_notation_l1219_121923

-- Define scientific notation
noncomputable def scientific_notation (a : ℝ) (n : ℤ) : ℝ := a * (10 : ℝ) ^ n

-- Define the original number
def original_number : ℕ := 105000

-- Theorem statement
theorem original_number_in_scientific_notation :
  (original_number : ℝ) = scientific_notation 1.05 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_number_in_scientific_notation_l1219_121923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_bounded_slope_l1219_121933

open Real

-- Define the function F
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := log x + a / x

-- State the theorem
theorem min_a_for_bounded_slope (a : ℝ) :
  (a > 0) →
  (∀ x ∈ Set.Ioo 0 3, (deriv (F a) x) ≤ (1 / 2)) →
  a ≥ (1 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_bounded_slope_l1219_121933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_red_beads_l1219_121932

/-- A necklace with blue and red beads -/
structure Necklace where
  blue : ℕ
  red : ℕ

/-- Predicate to check if a necklace satisfies the condition -/
def satisfies_condition (n : Necklace) : Prop :=
  ∀ (segment : List Bool), 
    segment.length ≤ n.blue + n.red →
    (segment.count Bool.true = 8 → segment.length - segment.count Bool.true ≥ 4)

/-- The main theorem -/
theorem minimum_red_beads : 
  ∀ (n : Necklace), 
    n.blue = 50 → 
    satisfies_condition n → 
    n.red ≥ 29 := by
  sorry

#check minimum_red_beads

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_red_beads_l1219_121932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_range_l1219_121947

open Real

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := exp x + x^2 + (3*a + 2)*x

-- State the theorem
theorem min_value_implies_a_range (a : ℝ) :
  (∃ x₀ ∈ Set.Ioo (-1) 0, ∀ x ∈ Set.Ioo (-1) 0, f a x₀ ≤ f a x) →
  a ∈ Set.Ioo (-1) (-1/(3*exp 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_range_l1219_121947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_minus_95_deg_l1219_121981

theorem sin_alpha_minus_95_deg (α : Real) : 
  (α ∈ Set.Icc π (3*π/2)) → 
  (Real.cos (85 * π/180 + α) = 4/5) → 
  (Real.sin (α - 95 * π/180) = 3/5) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_minus_95_deg_l1219_121981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_genuine_l1219_121959

/-- The probability of selecting at least one genuine product from a set of products. -/
def atLeastOneGenuine (total : ℕ) (genuine : ℕ) (selected : ℕ) : ℚ :=
  1 - (Nat.choose (total - genuine) selected / Nat.choose total selected)

/-- The probability of selecting at least one genuine product when randomly choosing 3 products out of 16, where 14 are genuine and 2 are defective, is equal to 1. -/
theorem prob_at_least_one_genuine (total : ℕ) (genuine : ℕ) (defective : ℕ) (selected : ℕ) : 
  total = 16 → genuine = 14 → defective = 2 → selected = 3 →
  atLeastOneGenuine total genuine selected = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_genuine_l1219_121959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_property_l1219_121942

-- Define the parabola C
def C : Set (ℝ × ℝ) := {p | p.2^2 = p.1}

-- Define the circle M
def M : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + p.2^2 = 1}

-- Define what it means for a line to be tangent to the circle M
def is_tangent_to_M (l : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ l ∩ M ∧ ∀ q : ℝ × ℝ, q ∈ l \ {p} → q ∉ M

-- Define a function to create a line through two points
def line_through (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p = (1 - t) • A + t • B}

-- State the theorem
theorem tangent_property (A₁ A₂ A₃ : ℝ × ℝ) 
  (h₁ : A₁ ∈ C) (h₂ : A₂ ∈ C) (h₃ : A₃ ∈ C)
  (h₄ : is_tangent_to_M (line_through A₁ A₂))
  (h₅ : is_tangent_to_M (line_through A₁ A₃)) :
  is_tangent_to_M (line_through A₂ A₃) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_property_l1219_121942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_output_relationship_problem_solution_l1219_121924

/-- Represents the production rate of a machine in boxes per minute -/
structure MachineRate where
  boxes_per_minute : ℝ
  boxes_per_minute_pos : 0 < boxes_per_minute

/-- The number of boxes produced by Machine A in 10 minutes -/
noncomputable def machine_a_output : ℝ := 
  Real.exp 1 -- Using an arbitrary positive real number

/-- Machine A's production rate -/
noncomputable def machine_a_rate : MachineRate :=
  { boxes_per_minute := machine_a_output / 10
    boxes_per_minute_pos := by
      apply div_pos
      . exact Real.exp_pos 1
      . norm_num }

/-- Machine B's production rate -/
noncomputable def machine_b_rate : MachineRate :=
  { boxes_per_minute := machine_a_output / 5
    boxes_per_minute_pos := by
      apply div_pos
      . exact Real.exp_pos 1
      . norm_num }

theorem machine_output_relationship :
  (machine_a_rate.boxes_per_minute + machine_b_rate.boxes_per_minute) * 20 = 10 * machine_a_output := by
  sorry

/-- The answer to the problem is the number of boxes Machine A produces in 10 minutes -/
theorem problem_solution : ∃ (x : ℝ), x = machine_a_output ∧ x > 0 := by
  use machine_a_output
  constructor
  . rfl
  . exact Real.exp_pos 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_output_relationship_problem_solution_l1219_121924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_formula_l1219_121951

/-- The radius of a sphere touching all edges of a regular tetrahedron with edge length a -/
noncomputable def sphere_radius (a : ℝ) : ℝ := (a * Real.sqrt 2) / 4

/-- A regular tetrahedron with edge length a -/
structure RegularTetrahedron (a : ℝ) where
  edge_length : a > 0

/-- A sphere touching all edges of a regular tetrahedron -/
structure TouchingSphere (a : ℝ) where
  tetrahedron : RegularTetrahedron a
  radius : ℝ
  touches_all_edges : radius = sphere_radius a

/-- Theorem stating that for any regular tetrahedron, there exists a sphere touching all its edges -/
theorem sphere_radius_formula (a : ℝ) (tetra : RegularTetrahedron a) :
  ∃ (sphere : TouchingSphere a), sphere.tetrahedron = tetra := by
  sorry

#check sphere_radius_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_formula_l1219_121951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_three_max_points_l1219_121928

/-- A function f(x) = sin(ωx + π/3) with ω > 0 has exactly 3 maximum points 
    on the interval [0,1] if and only if 25π/6 ≤ ω < 37π/6 -/
theorem sin_three_max_points (ω : ℝ) (h_ω_pos : ω > 0) :
  (∃ (x₁ x₂ x₃ : ℝ), 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ ≤ 1 ∧
    (∀ x ∈ Set.Icc 0 1, Real.sin (ω * x + π / 3) ≤ Real.sin (ω * x₁ + π / 3)) ∧
    (∀ x ∈ Set.Icc 0 1, Real.sin (ω * x + π / 3) ≤ Real.sin (ω * x₂ + π / 3)) ∧
    (∀ x ∈ Set.Icc 0 1, Real.sin (ω * x + π / 3) ≤ Real.sin (ω * x₃ + π / 3)) ∧
    (∀ x ∈ Set.Icc 0 1, x ≠ x₁ ∧ x ≠ x₂ ∧ x ≠ x₃ → 
      Real.sin (ω * x + π / 3) < max (Real.sin (ω * x₁ + π / 3)) 
                               (max (Real.sin (ω * x₂ + π / 3)) (Real.sin (ω * x₃ + π / 3)))))
  ↔ 25 * π / 6 ≤ ω ∧ ω < 37 * π / 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_three_max_points_l1219_121928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_height_squared_l1219_121948

/-- A right prism with regular hexagonal bases and a specific triangular pyramid. -/
structure HexagonalPrism where
  side_length : ℝ
  height : ℝ
  dihedral_angle : ℝ

/-- The conditions of our specific prism. -/
def our_prism (h : ℝ) : HexagonalPrism where
  side_length := 12
  height := h
  dihedral_angle := 60

theorem prism_height_squared (h : ℝ) : 
  (our_prism h).height^2 = 108 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_height_squared_l1219_121948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_nonconstant_prime_poly_l1219_121921

/-- A polynomial with integral coefficients -/
def IntPolynomial := ℕ → ℤ

/-- Evaluates a polynomial at a given point -/
def evalPoly (f : IntPolynomial) (x : ℕ) : ℤ :=
  f x

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  Nat.Prime n

/-- A polynomial is constant if it returns the same value for all inputs -/
def isConstant (f : IntPolynomial) : Prop :=
  ∀ x y : ℕ, evalPoly f x = evalPoly f y

theorem no_nonconstant_prime_poly :
  ∀ f : IntPolynomial,
    (∀ n : ℕ, isPrime (Int.natAbs (evalPoly f n))) →
    isConstant f :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_nonconstant_prime_poly_l1219_121921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_origin_to_line_distance_point_to_line_with_b_distance_between_parallel_lines_l1219_121989

noncomputable def distance_point_to_line (x₀ y₀ k b : ℝ) : ℝ :=
  (|k * x₀ - y₀ + b|) / Real.sqrt (1 + k^2)

theorem distance_origin_to_line :
  distance_point_to_line 0 0 1 1 = Real.sqrt 2 / 2 := by sorry

theorem distance_point_to_line_with_b (b : ℝ) :
  distance_point_to_line 1 1 1 b = 1 → b = Real.sqrt 2 ∨ b = -Real.sqrt 2 := by sorry

theorem distance_between_parallel_lines (b d : ℝ) :
  (∀ x, Real.sqrt 2 ≤ x → x ≤ 2 * Real.sqrt 2 → 
    distance_point_to_line 0 b (-1) 1 = d) →
  (3 ≤ b ∧ b ≤ 5) ∨ (-3 ≤ b ∧ b ≤ -1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_origin_to_line_distance_point_to_line_with_b_distance_between_parallel_lines_l1219_121989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_plus_pi_half_l1219_121988

theorem sin_theta_plus_pi_half (θ : ℝ) (h : Real.cos θ = -3/5) :
  Real.sin (θ + π/2) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_plus_pi_half_l1219_121988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_l1219_121925

noncomputable section

/-- The ellipse with equation x²/8 + y²/4 = 1 and directrix x = -4 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 8 + p.2^2 / 4 = 1}

/-- The directrix of the ellipse -/
def Directrix : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = -4}

/-- The point P at the intersection of the directrix and x-axis -/
def P : ℝ × ℝ := (-4, 0)

/-- The foci of the ellipse -/
def Foci : Set (ℝ × ℝ) :=
  {(-2, 0), (2, 0)}

/-- The vertices of the minor axis of the ellipse -/
def MinorAxisVertices : Set (ℝ × ℝ) :=
  {(0, -2), (0, 2)}

/-- The quadrilateral formed by the foci and minor axis vertices -/
def Quadrilateral : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (a b c d : ℝ × ℝ), a ∈ Foci ∧ b ∈ Foci ∧ c ∈ MinorAxisVertices ∧ d ∈ MinorAxisVertices ∧
    p.1 ≥ min a.1 b.1 ∧ p.1 ≤ max a.1 b.1 ∧ p.2 ≥ min c.2 d.2 ∧ p.2 ≤ max c.2 d.2}

/-- A line passing through P with slope k -/
def Line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * (p.1 + 4)}

/-- The intersection points of the line and the ellipse -/
def Intersection (k : ℝ) : Set (ℝ × ℝ) :=
  Ellipse ∩ Line k

/-- The midpoint of the two intersection points -/
def Midpoint (k : ℝ) : ℝ × ℝ :=
  let x₀ := -8 * k^2 / (1 + 2 * k^2)
  let y₀ := 4 * k / (1 + 2 * k^2)
  (x₀, y₀)

theorem slope_range :
  ∀ k : ℝ, (Midpoint k ∈ Quadrilateral) → (-((Real.sqrt 3 - 1) / 2) ≤ k ∧ k ≤ (Real.sqrt 3 - 1) / 2) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_l1219_121925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_property_P_implies_arithmetic_sequence_l1219_121978

/-- Property P for a set of real numbers -/
def property_P (A : Set ℝ) : Prop :=
  ∀ (x y : ℝ), x ∈ A → y ∈ A → x ≤ y → (x + y ∈ A ∨ y - x ∈ A)

/-- A set of 8 strictly increasing non-negative real numbers -/
structure IncreasingSet8 :=
  (a : Fin 8 → ℝ)
  (non_neg : ∀ i, 0 ≤ a i)
  (increasing : ∀ i j, i < j → a i < a j)

/-- Main theorem: If a set of 8 strictly increasing non-negative real numbers
    satisfies property P, then it forms an arithmetic sequence -/
theorem property_P_implies_arithmetic_sequence (S : IncreasingSet8) 
    (h_P : property_P (Finset.image S.a Finset.univ).toSet) :
    ∃ d, ∀ i : Fin 7, S.a i.succ = S.a i + d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_property_P_implies_arithmetic_sequence_l1219_121978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_homecoming_dress_discount_l1219_121920

theorem homecoming_dress_discount (original_price store_discount_rate member_discount_rate : ℝ) : 
  original_price = 300 →
  store_discount_rate = 0.2 →
  member_discount_rate = 0.1 →
  let price_after_store_discount := original_price * (1 - store_discount_rate)
  let final_price := price_after_store_discount * (1 - member_discount_rate)
  let total_discount := original_price - final_price
  let final_discount_percentage := total_discount / original_price
  final_discount_percentage = 0.28 := by
  sorry

#check homecoming_dress_discount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_homecoming_dress_discount_l1219_121920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_possible_values_l1219_121996

/-- The function f(n) as defined in the problem -/
def f (n : ℕ+) : ℚ :=
  (Finset.sum (Finset.range n) (fun k => (-1)^(k+1))) / n

/-- The set of possible values of f(n) -/
def possible_values : Set ℚ :=
  {q | ∃ n : ℕ+, q = 0 ∨ q = 1 / (↑n : ℚ)}

/-- Theorem stating that the set of possible values of f(n) is {0, 1/n} -/
theorem f_possible_values :
  (∀ n : ℕ+, f n ∈ possible_values) ∧
  (∀ q ∈ possible_values, ∃ m : ℕ+, f m = q) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_possible_values_l1219_121996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_sampling_theorem_l1219_121970

/-- Represents a class with male and female students. -/
structure MyClass where
  total_students : ℕ
  male_students : ℕ
  female_students : ℕ

/-- Represents an interest group formed from the class. -/
structure InterestGroup where
  size : ℕ
  male_members : ℕ
  female_members : ℕ

/-- Calculates the variance of a list of numbers. -/
noncomputable def variance (data : List ℝ) : ℝ :=
  let mean := data.sum / data.length
  (data.map (fun x => (x - mean) ^ 2)).sum / data.length

theorem class_sampling_theorem
  (c : MyClass)
  (g : InterestGroup)
  (h1 : c.total_students = 60)
  (h2 : c.male_students = 45)
  (h3 : c.female_students = 15)
  (h4 : g.size = 4)
  (h5 : g.male_members + g.female_members = g.size)
  (data1 : List ℝ := [68, 70, 71, 72, 74])
  (data2 : List ℝ := [69, 70, 70, 72, 74]) :
  (g.size : ℝ) / c.total_students = 1 / 15 ∧
  g.male_members = 3 ∧
  g.female_members = 1 ∧
  (3 : ℝ) / 6 = 1 / 2 ∧
  variance data1 = 4 ∧
  variance data2 = 3.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_sampling_theorem_l1219_121970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_samantha_overall_percentage_l1219_121940

/-- Represents a test with multiple-choice questions -/
structure Test where
  questions : ℕ
  correct_percentage : ℚ

/-- Calculates the overall correct percentage for a list of tests -/
def overall_correct_percentage (tests : List Test) : ℚ :=
  let total_questions := tests.map (·.questions) |>.sum
  let total_correct := tests.map (λ t => (t.questions : ℚ) * t.correct_percentage) |>.sum
  total_correct / total_questions

/-- The main theorem stating that given the specific test results, 
    the overall correct percentage is 81.5% -/
theorem samantha_overall_percentage : 
  let tests := [
    { questions := 30, correct_percentage := 9/10 },
    { questions := 50, correct_percentage := 17/20 },
    { questions := 20, correct_percentage := 3/5 }
  ]
  overall_correct_percentage tests = 163/200 := by sorry

#eval overall_correct_percentage [
  { questions := 30, correct_percentage := 9/10 },
  { questions := 50, correct_percentage := 17/20 },
  { questions := 20, correct_percentage := 3/5 }
]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_samantha_overall_percentage_l1219_121940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_days_l1219_121931

/-- Proves that the original plan was to complete the work in 17 days given the conditions of the problem. -/
theorem work_completion_days (total_men : ℕ) (absent_men : ℕ) (actual_days : ℕ) 
  (h1 : total_men = 42)
  (h2 : absent_men = 8)
  (h3 : actual_days = 21) :
  ∃ (original_days : ℕ), (total_men * original_days = (total_men - absent_men) * actual_days) ∧ 
  original_days = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_days_l1219_121931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_ratio_l1219_121956

/-- An arithmetic sequence with common difference d ≠ 0 -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- Three terms of a sequence form a geometric sequence -/
def geometric_subsequence (a : ℕ → ℝ) (i j k : ℕ) : Prop :=
  (a j)^2 = a i * a k

/-- The common ratio of a geometric sequence -/
noncomputable def common_ratio (a : ℕ → ℝ) (i j : ℕ) : ℝ :=
  a j / a i

theorem arithmetic_geometric_ratio
  (a : ℕ → ℝ) (d : ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_geom : geometric_subsequence a 1 3 9) :
  common_ratio a 1 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_ratio_l1219_121956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_bisection_l1219_121995

/-- The circle's equation -/
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 2*y + 8 = 0

/-- The line's equation -/
def line_eq (x y b : ℝ) : Prop :=
  y = x + b

/-- The center of the circle -/
def center : ℝ × ℝ :=
  (4, -1)

/-- A line bisects a circle's circumference if it passes through the center -/
def bisects (b : ℝ) : Prop :=
  line_eq center.1 center.2 b

theorem circle_bisection (b : ℝ) :
  (∀ x y, circle_eq x y → line_eq x y b → bisects b) →
  b = -5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_bisection_l1219_121995
