import Mathlib

namespace NUMINAMATH_CALUDE_expected_value_is_point_seven_l4020_402021

/-- A two-point distribution random variable -/
structure TwoPointDistribution where
  p : ℝ  -- Probability of X=1
  h1 : 0 ≤ p ∧ p ≤ 1  -- Probability is between 0 and 1
  h2 : p - (1 - p) = 0.4  -- Given condition

/-- Expected value of a two-point distribution -/
def expectedValue (X : TwoPointDistribution) : ℝ := X.p

theorem expected_value_is_point_seven (X : TwoPointDistribution) :
  expectedValue X = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_is_point_seven_l4020_402021


namespace NUMINAMATH_CALUDE_vasya_no_purchase_days_l4020_402062

/-- Represents Vasya's purchases over 15 school days -/
structure VasyaPurchases where
  marshmallow_days : ℕ -- Days buying 9 marshmallows
  meatpie_days : ℕ -- Days buying 2 meat pies
  combo_days : ℕ -- Days buying 4 marshmallows and 1 meat pie
  nothing_days : ℕ -- Days buying nothing

/-- Theorem stating the number of days Vasya didn't buy anything -/
theorem vasya_no_purchase_days (p : VasyaPurchases) : 
  p.marshmallow_days + p.meatpie_days + p.combo_days + p.nothing_days = 15 → 
  9 * p.marshmallow_days + 4 * p.combo_days = 30 →
  2 * p.meatpie_days + p.combo_days = 9 →
  p.nothing_days = 7 := by
  sorry

#check vasya_no_purchase_days

end NUMINAMATH_CALUDE_vasya_no_purchase_days_l4020_402062


namespace NUMINAMATH_CALUDE_inequality_proofs_l4020_402057

theorem inequality_proofs :
  (∀ a b : ℝ, a > 0 → b > 0 → (a + b) * (1 / a + 1 / b) ≥ 4) ∧
  (Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proofs_l4020_402057


namespace NUMINAMATH_CALUDE_coin_array_digit_sum_l4020_402086

/-- The sum of the first n natural numbers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

/-- Theorem: For a triangular array of 3003 coins, where the n-th row has n coins,
    the sum of the digits of the total number of rows is 14 -/
theorem coin_array_digit_sum :
  ∃ (n : ℕ), triangular_sum n = 3003 ∧ digit_sum n = 14 := by
  sorry

end NUMINAMATH_CALUDE_coin_array_digit_sum_l4020_402086


namespace NUMINAMATH_CALUDE_monotonicity_condition_inequality_solution_correct_l4020_402075

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - (3*m - 1) * x + m - 2

-- Part 1: Monotonicity condition
theorem monotonicity_condition (m : ℝ) :
  (∀ x ∈ Set.Icc 2 3, Monotone (f m)) ↔ m ≥ -1/3 :=
sorry

-- Part 2: Inequality solution
def inequality_solution (m : ℝ) : Set ℝ :=
  if m = 0 then Set.Ioi 2
  else if m > 0 then Set.Iio ((m-1)/m) ∪ Set.Ioi 2
  else if -1 < m ∧ m < 0 then Set.Ioo 2 ((m-1)/m)
  else if m = -1 then ∅
  else Set.Ioo ((m-1)/m) 2

theorem inequality_solution_correct (m : ℝ) (x : ℝ) :
  x ∈ inequality_solution m ↔ f m x + m > 0 :=
sorry

end NUMINAMATH_CALUDE_monotonicity_condition_inequality_solution_correct_l4020_402075


namespace NUMINAMATH_CALUDE_total_rainfall_2004_l4020_402083

/-- The average monthly rainfall in Mathborough in 2003 (in mm) -/
def avg_rainfall_2003 : ℝ := 41.5

/-- The increase in average monthly rainfall from 2003 to 2004 (in mm) -/
def rainfall_increase : ℝ := 2

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- Theorem: The total amount of rain that fell in Mathborough in 2004 was 522 mm -/
theorem total_rainfall_2004 : 
  (avg_rainfall_2003 + rainfall_increase) * months_in_year = 522 := by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_2004_l4020_402083


namespace NUMINAMATH_CALUDE_ship_speed_and_distance_l4020_402078

theorem ship_speed_and_distance 
  (downstream_time : ℝ) 
  (upstream_time : ℝ) 
  (current_speed : ℝ) 
  (h1 : downstream_time = 3)
  (h2 : upstream_time = 4)
  (h3 : current_speed = 3) :
  ∃ (still_water_speed : ℝ) (distance : ℝ),
    still_water_speed = 21 ∧
    distance = 72 ∧
    downstream_time * (still_water_speed + current_speed) = distance ∧
    upstream_time * (still_water_speed - current_speed) = distance :=
by sorry

end NUMINAMATH_CALUDE_ship_speed_and_distance_l4020_402078


namespace NUMINAMATH_CALUDE_binomial_expansion_max_term_max_term_for_sqrt11_expansion_l4020_402072

theorem binomial_expansion_max_term (n : ℕ) (x : ℝ) (h : x > 0) :
  ∃ k : ℕ, k ≤ n ∧
    ∀ j : ℕ, j ≤ n →
      (n.choose k : ℝ) * x^k ≥ (n.choose j : ℝ) * x^j :=
by sorry

theorem max_term_for_sqrt11_expansion :
  let n : ℕ := 208
  let x : ℝ := Real.sqrt 11
  ∃ k : ℕ, k = 160 ∧
    ∀ j : ℕ, j ≤ n →
      (n.choose k : ℝ) * x^k ≥ (n.choose j : ℝ) * x^j :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_max_term_max_term_for_sqrt11_expansion_l4020_402072


namespace NUMINAMATH_CALUDE_complement_of_67_is_23_l4020_402069

-- Define the complement of an angle
def complement (angle : ℝ) : ℝ := 90 - angle

-- State the theorem
theorem complement_of_67_is_23 : complement 67 = 23 := by sorry

end NUMINAMATH_CALUDE_complement_of_67_is_23_l4020_402069


namespace NUMINAMATH_CALUDE_tan_alpha_neg_half_implies_expression_neg_third_l4020_402084

theorem tan_alpha_neg_half_implies_expression_neg_third (α : Real) 
  (h : Real.tan α = -1/2) : 
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_neg_half_implies_expression_neg_third_l4020_402084


namespace NUMINAMATH_CALUDE_sqrt_mantissa_equality_l4020_402063

theorem sqrt_mantissa_equality (m n : ℕ) (h1 : m ≠ n) (h2 : m > 0) (h3 : n > 0) :
  (∃ (k : ℤ), Real.sqrt m - Real.sqrt n = k) → (∃ (a b : ℕ), m = a^2 ∧ n = b^2) :=
sorry

end NUMINAMATH_CALUDE_sqrt_mantissa_equality_l4020_402063


namespace NUMINAMATH_CALUDE_willy_crayon_count_l4020_402064

/-- The number of crayons Lucy has -/
def lucy_crayons : ℕ := 290

/-- The number of additional crayons Willy has compared to Lucy -/
def additional_crayons : ℕ := 1110

/-- The total number of crayons Willy has -/
def willy_crayons : ℕ := lucy_crayons + additional_crayons

theorem willy_crayon_count : willy_crayons = 1400 := by
  sorry

end NUMINAMATH_CALUDE_willy_crayon_count_l4020_402064


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_solution_l4020_402085

-- Define the set M as the domain of y = 1 / √(1-2x)
def M : Set ℝ := {x : ℝ | x < 1/2}

-- Define the set N as the range of y = x^2 - 4
def N : Set ℝ := {y : ℝ | y ≥ -4}

-- Theorem stating that the intersection of M and N is {x | -4 ≤ x < 1/2}
theorem M_intersect_N_eq_solution : M ∩ N = {x : ℝ | -4 ≤ x ∧ x < 1/2} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_solution_l4020_402085


namespace NUMINAMATH_CALUDE_triangle_law_of_sines_l4020_402040

theorem triangle_law_of_sines (A B : ℝ) (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : 0 ≤ A) (h4 : A < π) (h5 : 0 ≤ B) (h6 : B < π) :
  a = 3 → b = 5 → Real.sin A = 1/3 → Real.sin B = 5/9 := by sorry

end NUMINAMATH_CALUDE_triangle_law_of_sines_l4020_402040


namespace NUMINAMATH_CALUDE_min_values_xy_l4020_402038

theorem min_values_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 9*x + y - x*y = 0) :
  (∃ (min_sum : ℝ), min_sum = 16 ∧ 
    (∀ (a b : ℝ), a > 0 → b > 0 → 9*a + b - a*b = 0 → a + b ≥ min_sum) ∧
    (x + y = min_sum ↔ x = 4 ∧ y = 12)) ∧
  (∃ (min_prod : ℝ), min_prod = 36 ∧
    (∀ (a b : ℝ), a > 0 → b > 0 → 9*a + b - a*b = 0 → a*b ≥ min_prod) ∧
    (x*y = min_prod ↔ x = 2 ∧ y = 18)) :=
by sorry

end NUMINAMATH_CALUDE_min_values_xy_l4020_402038


namespace NUMINAMATH_CALUDE_max_product_digits_l4020_402049

theorem max_product_digits : ∀ a b : ℕ, 
  10000 ≤ a ∧ a < 100000 → 1000 ≤ b ∧ b < 10000 → 
  a * b < 1000000000 := by
  sorry

end NUMINAMATH_CALUDE_max_product_digits_l4020_402049


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_min_value_achievable_l4020_402047

theorem min_value_sum_reciprocals (x y z : ℕ+) (h : x + y + z = 12) :
  (x + y + z : ℚ) * (1 / (x + y : ℚ) + 1 / (x + z : ℚ) + 1 / (y + z : ℚ)) ≥ 9 / 2 :=
by sorry

theorem min_value_achievable :
  ∃ (x y z : ℕ+), x + y + z = 12 ∧
    (x + y + z : ℚ) * (1 / (x + y : ℚ) + 1 / (x + z : ℚ) + 1 / (y + z : ℚ)) = 9 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_min_value_achievable_l4020_402047


namespace NUMINAMATH_CALUDE_quadratic_function_max_value_l4020_402074

def f (a x : ℝ) : ℝ := a * x^2 + (2*a - 1) * x - 3

theorem quadratic_function_max_value (a : ℝ) (h1 : a ≠ 0) :
  (∀ x ∈ Set.Icc (-3/2 : ℝ) 2, f a x ≤ 1) ∧
  (∃ x ∈ Set.Icc (-3/2 : ℝ) 2, f a x = 1) →
  a = 3/4 ∨ a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_max_value_l4020_402074


namespace NUMINAMATH_CALUDE_convex_quadrilaterals_from_circle_points_l4020_402099

theorem convex_quadrilaterals_from_circle_points (n : ℕ) (k : ℕ) :
  n = 12 → k = 4 → Nat.choose n k = 495 := by
  sorry

end NUMINAMATH_CALUDE_convex_quadrilaterals_from_circle_points_l4020_402099


namespace NUMINAMATH_CALUDE_tutor_schedule_lcm_l4020_402060

theorem tutor_schedule_lcm : Nat.lcm (Nat.lcm (Nat.lcm 3 4) 6) 7 = 84 := by
  sorry

end NUMINAMATH_CALUDE_tutor_schedule_lcm_l4020_402060


namespace NUMINAMATH_CALUDE_chord_diameter_relationship_l4020_402096

/-- Represents a sphere with a chord and diameter -/
structure SphereWithChord where
  /-- The radius of the sphere -/
  radius : ℝ
  /-- The length of the chord AB -/
  chord_length : ℝ
  /-- The angle between the chord AB and the diameter CD -/
  angle : ℝ
  /-- The distance from C to A -/
  distance_CA : ℝ

/-- Theorem stating the relationship between the given conditions and BD -/
theorem chord_diameter_relationship (s : SphereWithChord) 
  (h1 : s.radius = 1)
  (h2 : s.chord_length = 1)
  (h3 : s.angle = Real.pi / 3)  -- 60 degrees in radians
  (h4 : s.distance_CA = Real.sqrt 2) :
  ∃ (BD : ℝ), BD = 1 := by
  sorry


end NUMINAMATH_CALUDE_chord_diameter_relationship_l4020_402096


namespace NUMINAMATH_CALUDE_urn_problem_l4020_402009

theorem urn_problem (N : ℕ) : 
  (6 : ℝ) / 10 * 10 / (10 + N) + (4 : ℝ) / 10 * N / (10 + N) = 1 / 2 → N = 10 := by
  sorry

end NUMINAMATH_CALUDE_urn_problem_l4020_402009


namespace NUMINAMATH_CALUDE_triangle_rotation_reflection_l4020_402076

/-- Rotation of 90 degrees clockwise about the origin -/
def rotate90Clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

/-- Reflection over the y-axis -/
def reflectOverYAxis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Given triangle ABC with vertices A(-3, 2), B(0, 5), and C(0, 2),
    prove that after rotating 90 degrees clockwise about the origin
    and then reflecting over the y-axis, point A ends up at (-2, 3) -/
theorem triangle_rotation_reflection :
  let A : ℝ × ℝ := (-3, 2)
  let B : ℝ × ℝ := (0, 5)
  let C : ℝ × ℝ := (0, 2)
  reflectOverYAxis (rotate90Clockwise A) = (-2, 3) := by
sorry


end NUMINAMATH_CALUDE_triangle_rotation_reflection_l4020_402076


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l4020_402002

theorem cube_root_equation_solution :
  ∃ y : ℝ, (5 - 2 / y)^(1/3 : ℝ) = -3 ∧ y = 1/16 := by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l4020_402002


namespace NUMINAMATH_CALUDE_cat_mouse_meet_iff_both_odd_cat_mouse_not_meet_when_sum_odd_cat_mouse_not_meet_when_both_even_l4020_402010

/-- Represents the movement of a cat and a mouse on a grid -/
def CatMouseMeet (m n : ℕ) : Prop :=
  m > 1 ∧ n > 1 ∧ Odd m ∧ Odd n

/-- Theorem stating the conditions for the cat and mouse to meet -/
theorem cat_mouse_meet_iff_both_odd (m n : ℕ) :
  CatMouseMeet m n ↔ (m > 1 ∧ n > 1 ∧ Odd m ∧ Odd n) :=
by sorry

/-- Theorem stating the impossibility of meeting when m + n is odd -/
theorem cat_mouse_not_meet_when_sum_odd (m n : ℕ) :
  m > 1 → n > 1 → Odd (m + n) → ¬(CatMouseMeet m n) :=
by sorry

/-- Theorem stating the impossibility of meeting when both m and n are even -/
theorem cat_mouse_not_meet_when_both_even (m n : ℕ) :
  m > 1 → n > 1 → Even m → Even n → ¬(CatMouseMeet m n) :=
by sorry

end NUMINAMATH_CALUDE_cat_mouse_meet_iff_both_odd_cat_mouse_not_meet_when_sum_odd_cat_mouse_not_meet_when_both_even_l4020_402010


namespace NUMINAMATH_CALUDE_specific_pyramid_lateral_area_l4020_402052

/-- Represents a pyramid with a parallelogram base -/
structure Pyramid :=
  (base_side1 : ℝ)
  (base_side2 : ℝ)
  (base_area : ℝ)
  (height : ℝ)

/-- Calculates the lateral surface area of a pyramid -/
def lateral_surface_area (p : Pyramid) : ℝ :=
  sorry

/-- Theorem stating the lateral surface area of the specific pyramid -/
theorem specific_pyramid_lateral_area :
  let p : Pyramid := { 
    base_side1 := 10,
    base_side2 := 18,
    base_area := 90,
    height := 6
  }
  lateral_surface_area p = 192 := by sorry

end NUMINAMATH_CALUDE_specific_pyramid_lateral_area_l4020_402052


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l4020_402022

noncomputable def f (x : ℝ) : ℝ := Real.exp (2*x + 1) - 3*x

theorem f_derivative_at_zero : 
  deriv f 0 = 2 * Real.exp 1 - 3 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l4020_402022


namespace NUMINAMATH_CALUDE_running_track_dimensions_l4020_402019

/-- Given two concentric circles forming a running track, prove the width and radii -/
theorem running_track_dimensions (r₁ r₂ : ℝ) : 
  (2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 20 * Real.pi) →  -- Difference in circumferences
  (2 * Real.pi * r₂ = 40 * Real.pi) →                     -- Circumference of inner circle
  (r₁ - r₂ = 10) ∧                                        -- Width of the track
  (r₂ = 20) ∧                                             -- Radius of inner circle
  (r₁ = 30) :=                                            -- Radius of outer circle
by
  sorry

end NUMINAMATH_CALUDE_running_track_dimensions_l4020_402019


namespace NUMINAMATH_CALUDE_ellipse_fixed_points_l4020_402020

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  center : Point

/-- Checks if a point is on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop :=
  (p.x - e.center.x)^2 / e.a^2 + (p.y - e.center.y)^2 / e.b^2 = 1

/-- Calculates the dot product of two vectors -/
def dotProduct (v1 v2 : Point) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

/-- Theorem about the existence of fixed points Q on the ellipse -/
theorem ellipse_fixed_points (e : Ellipse) (f a : Point) :
  e.a = 2 ∧ e.b = Real.sqrt 3 ∧ e.center = Point.mk 0 0 ∧
  f = Point.mk 1 0 ∧ a = Point.mk (-2) 0 →
  ∃ q1 q2 : Point,
    q1 = Point.mk 1 0 ∧ q2 = Point.mk 7 0 ∧
    ∀ b c m n : Point,
      isOnEllipse e b ∧ isOnEllipse e c ∧
      b ≠ c ∧
      (∃ t : ℝ, b.x = t * b.y + 1 ∧ c.x = t * c.y + 1) ∧
      m.x = 4 ∧ n.x = 4 ∧
      (m.y - a.y) / (m.x - a.x) = (b.y - a.y) / (b.x - a.x) ∧
      (n.y - a.y) / (n.x - a.x) = (c.y - a.y) / (c.x - a.x) →
      dotProduct (Point.mk (q1.x - m.x) (q1.y - m.y)) (Point.mk (q1.x - n.x) (q1.y - n.y)) = 0 ∧
      dotProduct (Point.mk (q2.x - m.x) (q2.y - m.y)) (Point.mk (q2.x - n.x) (q2.y - n.y)) = 0 :=
sorry

end NUMINAMATH_CALUDE_ellipse_fixed_points_l4020_402020


namespace NUMINAMATH_CALUDE_jogger_train_distance_jogger_train_problem_l4020_402054

theorem jogger_train_distance (jogger_speed : Real) (train_speed : Real) 
  (train_length : Real) (passing_time : Real) : Real :=
  let jogger_speed_ms := jogger_speed * 1000 / 3600
  let train_speed_ms := train_speed * 1000 / 3600
  let relative_speed := train_speed_ms - jogger_speed_ms
  let distance_covered := relative_speed * passing_time
  distance_covered - train_length

theorem jogger_train_problem :
  jogger_train_distance 9 45 120 40 = 280 := by
  sorry

end NUMINAMATH_CALUDE_jogger_train_distance_jogger_train_problem_l4020_402054


namespace NUMINAMATH_CALUDE_volume_tetrahedron_C₁LMN_l4020_402007

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cuboid in 3D space -/
structure Cuboid where
  a : Point3D
  b : Point3D
  c : Point3D
  d : Point3D
  a₁ : Point3D
  b₁ : Point3D
  c₁ : Point3D
  d₁ : Point3D

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the volume of a tetrahedron given its four vertices -/
def tetrahedronVolume (p₁ p₂ p₃ p₄ : Point3D) : ℝ := sorry

/-- Finds the intersection of a line and a plane -/
def lineIntersectPlane (p₁ p₂ : Point3D) (plane : Plane) : Point3D := sorry

/-- Theorem: Volume of tetrahedron C₁LMN in the given cuboid -/
theorem volume_tetrahedron_C₁LMN (cuboid : Cuboid) 
  (h₁ : cuboid.a₁.z - cuboid.a.z = 2)
  (h₂ : cuboid.d.y - cuboid.a.y = 3)
  (h₃ : cuboid.b.x - cuboid.a.x = 251) :
  ∃ (volume : ℝ),
    let plane_A₁BD : Plane := sorry
    let L : Point3D := lineIntersectPlane cuboid.c cuboid.c₁ plane_A₁BD
    let M : Point3D := lineIntersectPlane cuboid.c₁ cuboid.b₁ plane_A₁BD
    let N : Point3D := lineIntersectPlane cuboid.c₁ cuboid.d₁ plane_A₁BD
    volume = tetrahedronVolume cuboid.c₁ L M N := by sorry

end NUMINAMATH_CALUDE_volume_tetrahedron_C₁LMN_l4020_402007


namespace NUMINAMATH_CALUDE_squarefree_primes_property_l4020_402061

theorem squarefree_primes_property : 
  {p : ℕ | Nat.Prime p ∧ p ≥ 3 ∧ 
    ∀ q : ℕ, Nat.Prime q → q < p → 
      Squarefree (p - p / q * q)} = {5, 7, 13} := by sorry

end NUMINAMATH_CALUDE_squarefree_primes_property_l4020_402061


namespace NUMINAMATH_CALUDE_marble_jar_problem_l4020_402097

theorem marble_jar_problem (r g b : ℕ) :
  r + g = 5 →
  r + b = 7 →
  g + b = 9 →
  r + g + b = 12 := by
  sorry

end NUMINAMATH_CALUDE_marble_jar_problem_l4020_402097


namespace NUMINAMATH_CALUDE_square_difference_formula_l4020_402041

theorem square_difference_formula (a b : ℚ) 
  (sum_eq : a + b = 11/17) 
  (diff_eq : a - b = 1/143) : 
  a^2 - b^2 = 11/2431 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_formula_l4020_402041


namespace NUMINAMATH_CALUDE_phone_number_probability_l4020_402067

/-- The probability of randomly dialing the correct seven-digit number -/
theorem phone_number_probability :
  let first_three_options : ℕ := 2  -- 298 or 299
  let last_four_digits : ℕ := 4  -- 0, 2, 6, 7
  let total_combinations := first_three_options * (Nat.factorial last_four_digits)
  (1 : ℚ) / total_combinations = 1 / 48 :=
by sorry

end NUMINAMATH_CALUDE_phone_number_probability_l4020_402067


namespace NUMINAMATH_CALUDE_sunzi_problem_l4020_402032

theorem sunzi_problem : ∃! n : ℕ, 
  100 < n ∧ n < 200 ∧ 
  n % 3 = 2 ∧ 
  n % 5 = 3 ∧ 
  n % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sunzi_problem_l4020_402032


namespace NUMINAMATH_CALUDE_unique_triple_solution_l4020_402080

theorem unique_triple_solution : 
  ∃! (p x y : ℕ), 
    Prime p ∧ 
    p ^ x = y ^ 4 + 4 ∧ 
    p = 5 ∧ x = 1 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l4020_402080


namespace NUMINAMATH_CALUDE_connecting_line_is_correct_l4020_402077

/-- The equation of a circle in the form (x-h)^2 + (y-k)^2 = r^2 -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given two circles, returns the line connecting their centers -/
def line_connecting_centers (c1 c2 : Circle) : Line :=
  sorry

/-- The first circle: x^2+y^2-4x+6y=0 -/
def circle1 : Circle :=
  { h := 2, k := -3, r := 5 }

/-- The second circle: x^2+y^2-6x=0 -/
def circle2 : Circle :=
  { h := 3, k := 0, r := 3 }

/-- The expected line: 3x-y-9=0 -/
def expected_line : Line :=
  { a := 3, b := -1, c := -9 }

theorem connecting_line_is_correct :
  line_connecting_centers circle1 circle2 = expected_line :=
sorry

end NUMINAMATH_CALUDE_connecting_line_is_correct_l4020_402077


namespace NUMINAMATH_CALUDE_place_value_comparison_l4020_402033

def number : ℚ := 43597.2468

theorem place_value_comparison : 
  (100 : ℚ) * (number % 1000 - number % 100) / 100 = (number % 0.1 - number % 0.01) / 0.01 := by
  sorry

end NUMINAMATH_CALUDE_place_value_comparison_l4020_402033


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l4020_402044

theorem quadratic_form_ratio (k : ℝ) : 
  ∃ (d r s : ℝ), 8 * k^2 - 6 * k + 16 = d * (k + r)^2 + s ∧ s / r = -118 / 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l4020_402044


namespace NUMINAMATH_CALUDE_eighty_factorial_zeroes_l4020_402070

/-- Count the number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ := sorry

theorem eighty_factorial_zeroes :
  trailingZeroes 73 = 16 → trailingZeroes 80 = 19 := by sorry

end NUMINAMATH_CALUDE_eighty_factorial_zeroes_l4020_402070


namespace NUMINAMATH_CALUDE_min_A_over_C_is_zero_l4020_402028

theorem min_A_over_C_is_zero (x : ℝ) (A C : ℝ) (h1 : x ≠ 0) (h2 : A > 0) (h3 : C > 0)
  (h4 : x^2 + 1/x^2 = A) (h5 : x + 1/x = C) :
  ∃ ε > 0, ∀ δ > 0, ∃ A' C', A' > 0 ∧ C' > 0 ∧ A'/C' < δ ∧
  ∃ x' : ℝ, x' ≠ 0 ∧ x'^2 + 1/x'^2 = A' ∧ x' + 1/x' = C' :=
sorry

end NUMINAMATH_CALUDE_min_A_over_C_is_zero_l4020_402028


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l4020_402024

/-- The distance between the vertices of the hyperbola x²/64 - y²/49 = 1 is 16 -/
theorem hyperbola_vertex_distance : 
  let h : ℝ → ℝ → Prop := fun x y => x^2/64 - y^2/49 = 1
  ∃ (x₁ x₂ : ℝ), h x₁ 0 ∧ h x₂ 0 ∧ |x₁ - x₂| = 16 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l4020_402024


namespace NUMINAMATH_CALUDE_sandbox_perimeter_l4020_402071

/-- The perimeter of a rectangular sandbox with width 5 feet and length twice the width is 30 feet. -/
theorem sandbox_perimeter : 
  ∀ (width length perimeter : ℝ), 
  width = 5 → 
  length = 2 * width → 
  perimeter = 2 * (length + width) → 
  perimeter = 30 := by sorry

end NUMINAMATH_CALUDE_sandbox_perimeter_l4020_402071


namespace NUMINAMATH_CALUDE_initial_speed_satisfies_conditions_initial_speed_is_unique_l4020_402059

/-- The child's initial walking speed in meters per minute -/
def initial_speed : ℝ := 5

/-- The time it takes for the child to walk to school at the initial speed -/
def initial_time : ℝ := 126

/-- The distance from home to school in meters -/
def distance : ℝ := 630

/-- Theorem stating that the initial speed satisfies the given conditions -/
theorem initial_speed_satisfies_conditions :
  distance = initial_speed * initial_time ∧
  distance = 7 * (initial_time - 36) :=
sorry

/-- Theorem proving that the initial speed is unique -/
theorem initial_speed_is_unique (v : ℝ) :
  (∃ t : ℝ, distance = v * t ∧ distance = 7 * (t - 36)) →
  v = initial_speed :=
sorry

end NUMINAMATH_CALUDE_initial_speed_satisfies_conditions_initial_speed_is_unique_l4020_402059


namespace NUMINAMATH_CALUDE_option_B_is_inductive_reasoning_l4020_402065

-- Define a sequence
def a : ℕ → ℕ
| 1 => 1
| n => 3 * n - 1

-- Define the sum of the first n terms
def S (n : ℕ) : ℕ := (List.range n).map a |>.sum

-- Define inductive reasoning
def is_inductive_reasoning (process : Prop) : Prop :=
  ∃ (n : ℕ), n > 0 ∧ (∀ k ≤ n, ∃ (result : Prop), process → result)

-- Theorem statement
theorem option_B_is_inductive_reasoning :
  is_inductive_reasoning (∃ (formula : ℕ → ℕ), ∀ n, S n = formula n) :=
sorry

end NUMINAMATH_CALUDE_option_B_is_inductive_reasoning_l4020_402065


namespace NUMINAMATH_CALUDE_inequality_range_l4020_402027

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x + 2| + |x - 1| > a^2 - 2*a) → 
  -1 < a ∧ a < 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l4020_402027


namespace NUMINAMATH_CALUDE_rectangle_y_value_l4020_402029

/-- A rectangle with vertices (-2, y), (6, y), (-2, 2), and (6, 2) -/
structure Rectangle where
  y : ℝ
  h_positive : y > 0

/-- The area of the rectangle -/
def area (r : Rectangle) : ℝ := 8 * (r.y - 2)

/-- The perimeter of the rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (8 + (r.y - 2))

/-- Theorem stating that if the area is 64 and the perimeter is 32, then y = 10 -/
theorem rectangle_y_value (r : Rectangle) 
  (h_area : area r = 64) 
  (h_perimeter : perimeter r = 32) : 
  r.y = 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l4020_402029


namespace NUMINAMATH_CALUDE_scooter_gain_percent_l4020_402039

/-- Calculate the gain percent on a scooter sale -/
theorem scooter_gain_percent
  (purchase_price : ℝ)
  (repair_cost : ℝ)
  (selling_price : ℝ)
  (h1 : purchase_price = 900)
  (h2 : repair_cost = 300)
  (h3 : selling_price = 1500) :
  (selling_price - (purchase_price + repair_cost)) / (purchase_price + repair_cost) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_scooter_gain_percent_l4020_402039


namespace NUMINAMATH_CALUDE_nine_more_likely_than_ten_l4020_402048

def roll_combinations (sum : ℕ) : Finset (ℕ × ℕ) :=
  (Finset.range 6 ×ˢ Finset.range 6).filter (fun (a, b) => a + b + 2 = sum)

theorem nine_more_likely_than_ten :
  (roll_combinations 9).card > (roll_combinations 10).card := by
  sorry

end NUMINAMATH_CALUDE_nine_more_likely_than_ten_l4020_402048


namespace NUMINAMATH_CALUDE_min_value_quadratic_l4020_402073

theorem min_value_quadratic (x y : ℝ) :
  3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 ≥ 9 ∧
  (3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 = 9 ↔ x = 0 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l4020_402073


namespace NUMINAMATH_CALUDE_initial_marbles_l4020_402018

/-- Proves that if a person has 7 marbles left after sharing 3 marbles, 
    then they initially had 10 marbles. -/
theorem initial_marbles (shared : ℕ) (left : ℕ) (initial : ℕ) : 
  shared = 3 → left = 7 → initial = shared + left → initial = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_marbles_l4020_402018


namespace NUMINAMATH_CALUDE_cherry_pie_count_l4020_402095

theorem cherry_pie_count (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) 
  (h1 : total_pies = 24)
  (h2 : apple_ratio = 1)
  (h3 : blueberry_ratio = 4)
  (h4 : cherry_ratio = 3) : 
  (total_pies * cherry_ratio) / (apple_ratio + blueberry_ratio + cherry_ratio) = 9 := by
sorry

end NUMINAMATH_CALUDE_cherry_pie_count_l4020_402095


namespace NUMINAMATH_CALUDE_lisa_caffeine_consumption_l4020_402026

/-- Represents the number of beverages Lisa drinks -/
structure BeverageCount where
  coffee : ℕ
  soda : ℕ
  tea : ℕ

/-- Represents the caffeine content of each beverage in milligrams -/
structure CaffeineContent where
  coffee : ℕ
  soda : ℕ
  tea : ℕ

/-- Calculates the total caffeine consumed based on beverage count and caffeine content -/
def totalCaffeine (count : BeverageCount) (content : CaffeineContent) : ℕ :=
  count.coffee * content.coffee + count.soda * content.soda + count.tea * content.tea

/-- Lisa's daily caffeine goal in milligrams -/
def caffeineGoal : ℕ := 200

/-- Theorem stating Lisa's caffeine consumption and excess -/
theorem lisa_caffeine_consumption 
  (lisas_beverages : BeverageCount)
  (caffeine_per_beverage : CaffeineContent)
  (h1 : lisas_beverages.coffee = 3)
  (h2 : lisas_beverages.soda = 1)
  (h3 : lisas_beverages.tea = 2)
  (h4 : caffeine_per_beverage.coffee = 80)
  (h5 : caffeine_per_beverage.soda = 40)
  (h6 : caffeine_per_beverage.tea = 50) :
  totalCaffeine lisas_beverages caffeine_per_beverage = 380 ∧
  totalCaffeine lisas_beverages caffeine_per_beverage - caffeineGoal = 180 := by
  sorry

end NUMINAMATH_CALUDE_lisa_caffeine_consumption_l4020_402026


namespace NUMINAMATH_CALUDE_oliver_old_cards_l4020_402082

/-- Calculates the number of old baseball cards Oliver had. -/
def old_cards (cards_per_page : ℕ) (new_cards : ℕ) (pages_used : ℕ) : ℕ :=
  cards_per_page * pages_used - new_cards

/-- Theorem stating that Oliver had 10 old cards. -/
theorem oliver_old_cards : old_cards 3 2 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_oliver_old_cards_l4020_402082


namespace NUMINAMATH_CALUDE_gcd_91_49_l4020_402016

theorem gcd_91_49 : Nat.gcd 91 49 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_91_49_l4020_402016


namespace NUMINAMATH_CALUDE_distinct_numbers_squared_differences_l4020_402081

theorem distinct_numbers_squared_differences (n : ℕ) (a : Fin n → ℝ) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) (h_n : n = 10) : 
  {x | ∃ i j, i < j ∧ x = (a j - a i)^2} ≠ 
  {y | ∃ i j, i < j ∧ y = |a j^2 - a i^2|} :=
sorry

end NUMINAMATH_CALUDE_distinct_numbers_squared_differences_l4020_402081


namespace NUMINAMATH_CALUDE_painting_cost_is_474_l4020_402053

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a wall -/
def wallArea (d : Dimensions) : ℝ := d.length * d.height

/-- Calculates the area of a rectangular opening -/
def openingArea (d : Dimensions) : ℝ := d.length * d.width

/-- Represents a room with its dimensions and openings -/
structure Room where
  dimensions : Dimensions
  doorDimensions : Dimensions
  largeDoorCount : ℕ
  largeWindowDimensions : Dimensions
  largeWindowCount : ℕ
  smallWindowDimensions : Dimensions
  smallWindowCount : ℕ

/-- Calculates the total wall area of a room -/
def totalWallArea (r : Room) : ℝ :=
  2 * (wallArea r.dimensions + wallArea { length := r.dimensions.width, width := r.dimensions.width, height := r.dimensions.height })

/-- Calculates the total area of openings in a room -/
def totalOpeningArea (r : Room) : ℝ :=
  (r.largeDoorCount : ℝ) * openingArea r.doorDimensions +
  (r.largeWindowCount : ℝ) * openingArea r.largeWindowDimensions +
  (r.smallWindowCount : ℝ) * openingArea r.smallWindowDimensions

/-- Calculates the paintable area of a room -/
def paintableArea (r : Room) : ℝ :=
  totalWallArea r - totalOpeningArea r

/-- Theorem: The cost of painting the given room is Rs. 474 -/
theorem painting_cost_is_474 (r : Room)
  (h1 : r.dimensions = { length := 10, width := 7, height := 5 })
  (h2 : r.doorDimensions = { length := 1, width := 3, height := 3 })
  (h3 : r.largeDoorCount = 2)
  (h4 : r.largeWindowDimensions = { length := 2, width := 1.5, height := 1.5 })
  (h5 : r.largeWindowCount = 1)
  (h6 : r.smallWindowDimensions = { length := 1, width := 1.5, height := 1.5 })
  (h7 : r.smallWindowCount = 2)
  : paintableArea r * 3 = 474 := by
  sorry

end NUMINAMATH_CALUDE_painting_cost_is_474_l4020_402053


namespace NUMINAMATH_CALUDE_cos_two_alpha_plus_pi_third_l4020_402088

theorem cos_two_alpha_plus_pi_third (α : ℝ) 
  (h : Real.sin (π / 6 - α) - Real.cos α = 1 / 3) : 
  Real.cos (2 * α + π / 3) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_alpha_plus_pi_third_l4020_402088


namespace NUMINAMATH_CALUDE_triangle_side_length_l4020_402092

theorem triangle_side_length (b c : ℝ) (A : ℝ) (S : ℝ) : 
  b = 2 → 
  A = 2 * π / 3 → 
  S = 2 * Real.sqrt 3 → 
  S = 1/2 * b * c * Real.sin A →
  b^2 + c^2 - 2*b*c*Real.cos A = (2 * Real.sqrt 7)^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4020_402092


namespace NUMINAMATH_CALUDE_families_increase_l4020_402098

theorem families_increase (F : ℝ) (h1 : F > 0) : 
  let families_with_computers_1992 := 0.3 * F
  let families_with_computers_1999 := 1.5 * families_with_computers_1992
  let total_families_1999 := families_with_computers_1999 / (3/7)
  total_families_1999 = 1.05 * F :=
by sorry

end NUMINAMATH_CALUDE_families_increase_l4020_402098


namespace NUMINAMATH_CALUDE_no_such_function_l4020_402037

theorem no_such_function : ¬∃ f : ℕ → ℕ, ∀ n : ℕ, n > 1 → f n = f (f (n - 1)) + f (f (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_no_such_function_l4020_402037


namespace NUMINAMATH_CALUDE_hexagon_wire_remainder_l4020_402014

/-- Calculates the remaining wire length after creating a regular hexagon -/
def remaining_wire_length (total_wire : ℝ) (hexagon_side : ℝ) : ℝ :=
  total_wire - 6 * hexagon_side

/-- Theorem: Given a 50 cm wire and a regular hexagon with 8 cm sides, 2 cm of wire remains -/
theorem hexagon_wire_remainder :
  remaining_wire_length 50 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_wire_remainder_l4020_402014


namespace NUMINAMATH_CALUDE_mityas_age_l4020_402094

theorem mityas_age (shura_age mitya_age : ℚ) : 
  (mitya_age = shura_age + 11) →
  (mitya_age - shura_age = 2 * (shura_age - (mitya_age - shura_age))) →
  mitya_age = 27.5 := by sorry

end NUMINAMATH_CALUDE_mityas_age_l4020_402094


namespace NUMINAMATH_CALUDE_coefficient_x4_equals_240_l4020_402030

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the coefficient of x^4 in (1+2x)^6
def coefficient_x4 : ℕ := binomial 6 4 * 2^4

-- Theorem statement
theorem coefficient_x4_equals_240 : coefficient_x4 = 240 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_equals_240_l4020_402030


namespace NUMINAMATH_CALUDE_wanda_blocks_calculation_l4020_402093

theorem wanda_blocks_calculation (initial_blocks : ℕ) (theresa_percentage : ℚ) (give_away_fraction : ℚ) : 
  initial_blocks = 2450 →
  theresa_percentage = 35 / 100 →
  give_away_fraction = 1 / 8 →
  (initial_blocks + Int.floor (theresa_percentage * initial_blocks) - 
   Int.floor (give_away_fraction * (initial_blocks + Int.floor (theresa_percentage * initial_blocks)))) = 2894 := by
  sorry

end NUMINAMATH_CALUDE_wanda_blocks_calculation_l4020_402093


namespace NUMINAMATH_CALUDE_tangent_circles_a_value_l4020_402005

/-- Circle C₁ with equation x² + y² = 16 -/
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 16}

/-- Circle C₂ with equation (x - a)² + y² = 1, parameterized by a -/
def C₂ (a : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - a)^2 + p.2^2 = 1}

/-- Two circles are tangent if they intersect at exactly one point -/
def are_tangent (S T : Set (ℝ × ℝ)) : Prop :=
  ∃! p, p ∈ S ∧ p ∈ T

/-- The main theorem: if C₁ and C₂(a) are tangent, then a = ±5 or a = ±3 -/
theorem tangent_circles_a_value :
  ∀ a : ℝ, are_tangent C₁ (C₂ a) → a = 5 ∨ a = -5 ∨ a = 3 ∨ a = -3 :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_a_value_l4020_402005


namespace NUMINAMATH_CALUDE_cube_root_of_negative_one_eighth_l4020_402090

theorem cube_root_of_negative_one_eighth :
  ∃ y : ℝ, y^3 = (-1/8 : ℝ) ∧ y = (-1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_one_eighth_l4020_402090


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l4020_402079

theorem sum_of_coefficients : 
  let p (x : ℝ) := 3*(x^8 - x^5 + 2*x^3 - 6) - 5*(x^4 + 3*x^2) + 2*(x^6 - 5)
  (p 1) = -40 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l4020_402079


namespace NUMINAMATH_CALUDE_profit_maximizing_volume_l4020_402034

/-- Annual fixed cost in ten thousand dollars -/
def fixed_cost : ℝ := 10

/-- Variable cost per thousand items in ten thousand dollars -/
def variable_cost : ℝ := 2.7

/-- Revenue function in ten thousand dollars -/
noncomputable def R (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then
    10.8 - x^2 / 30
  else if x > 10 then
    108 / x - 1000 / (3 * x^2)
  else
    0

/-- Profit function in ten thousand dollars -/
noncomputable def W (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then
    x * R x - (fixed_cost + variable_cost * x)
  else if x > 10 then
    x * R x - (fixed_cost + variable_cost * x)
  else
    0

/-- Theorem stating that the profit-maximizing production volume is 9 thousand items -/
theorem profit_maximizing_volume :
  ∃ (max_profit : ℝ), W 9 = max_profit ∧ ∀ x, W x ≤ max_profit :=
by sorry

end NUMINAMATH_CALUDE_profit_maximizing_volume_l4020_402034


namespace NUMINAMATH_CALUDE_geometric_series_relation_l4020_402025

/-- Given two infinite geometric series with specified terms, prove that if the sum of the second series
    is three times the sum of the first series, then n = 4. -/
theorem geometric_series_relation (n : ℝ) : 
  let first_series_term1 : ℝ := 15
  let first_series_term2 : ℝ := 9
  let second_series_term1 : ℝ := 15
  let second_series_term2 : ℝ := 9 + n
  let first_series_sum : ℝ := first_series_term1 / (1 - (first_series_term2 / first_series_term1))
  let second_series_sum : ℝ := second_series_term1 / (1 - (second_series_term2 / second_series_term1))
  second_series_sum = 3 * first_series_sum → n = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_geometric_series_relation_l4020_402025


namespace NUMINAMATH_CALUDE_not_divisible_by_361_l4020_402011

theorem not_divisible_by_361 (k : ℕ) : ¬(361 ∣ k^2 + 11*k - 22) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_361_l4020_402011


namespace NUMINAMATH_CALUDE_rectangle_area_l4020_402050

/-- Calculates the area of a rectangle given its perimeter and width. -/
theorem rectangle_area (perimeter : ℝ) (width : ℝ) (h1 : perimeter = 42) (h2 : width = 8) :
  width * (perimeter / 2 - width) = 104 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l4020_402050


namespace NUMINAMATH_CALUDE_cube_root_inequality_l4020_402045

theorem cube_root_inequality (x : ℤ) : 
  (2 : ℝ) < (2 * (x : ℝ)^2)^(1/3) ∧ (2 * (x : ℝ)^2)^(1/3) < 3 ↔ x = 3 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_inequality_l4020_402045


namespace NUMINAMATH_CALUDE_no_valid_k_exists_l4020_402031

/-- The nth odd prime number -/
def nthOddPrime (n : ℕ) : ℕ := sorry

/-- The product of the first n odd prime numbers -/
def productFirstNOddPrimes (n : ℕ) : ℕ := sorry

/-- Statement: There does not exist a natural number k such that the product 
    of the first k odd prime numbers, decreased by 1, is an exact power 
    of a natural number greater than 1 -/
theorem no_valid_k_exists : 
  ¬ ∃ (k : ℕ), ∃ (a n : ℕ), n > 1 ∧ productFirstNOddPrimes k - 1 = a^n := by
  sorry


end NUMINAMATH_CALUDE_no_valid_k_exists_l4020_402031


namespace NUMINAMATH_CALUDE_g_of_5_l4020_402015

def g (x : ℝ) : ℝ := x^2 - 2*x

theorem g_of_5 : g 5 = 15 := by sorry

end NUMINAMATH_CALUDE_g_of_5_l4020_402015


namespace NUMINAMATH_CALUDE_average_trees_planted_l4020_402056

theorem average_trees_planted (total_students : ℕ) (trees_3 trees_4 trees_5 trees_6 : ℕ) 
  (h1 : total_students = 50)
  (h2 : trees_3 = 20)
  (h3 : trees_4 = 15)
  (h4 : trees_5 = 10)
  (h5 : trees_6 = 5)
  (h6 : trees_3 + trees_4 + trees_5 + trees_6 = total_students) :
  (3 * trees_3 + 4 * trees_4 + 5 * trees_5 + 6 * trees_6) / total_students = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_trees_planted_l4020_402056


namespace NUMINAMATH_CALUDE_count_rational_roots_l4020_402001

/-- The number of different possible rational roots for a polynomial of the form
    12x^4 + b₃x³ + b₂x² + b₁x + 18 = 0 with integer coefficients -/
def num_rational_roots (b₃ b₂ b₁ : ℤ) : ℕ := 28

/-- Theorem stating that the number of different possible rational roots for the given polynomial is 28 -/
theorem count_rational_roots (b₃ b₂ b₁ : ℤ) : 
  num_rational_roots b₃ b₂ b₁ = 28 := by sorry

end NUMINAMATH_CALUDE_count_rational_roots_l4020_402001


namespace NUMINAMATH_CALUDE_jason_coin_difference_l4020_402012

/-- Given that Jayden received 300 coins, the total coins given to both boys is 660,
    and Jason received more coins than Jayden, prove that Jason received 60 more coins than Jayden. -/
theorem jason_coin_difference (jayden_coins : ℕ) (total_coins : ℕ) (jason_coins : ℕ)
  (h1 : jayden_coins = 300)
  (h2 : total_coins = 660)
  (h3 : jason_coins + jayden_coins = total_coins)
  (h4 : jason_coins > jayden_coins) :
  jason_coins - jayden_coins = 60 := by
  sorry

end NUMINAMATH_CALUDE_jason_coin_difference_l4020_402012


namespace NUMINAMATH_CALUDE_inverse_implies_negation_angle_60_iff_arithmetic_sequence_not_necessary_condition_xy_not_necessary_condition_ab_l4020_402043

-- Proposition 1
theorem inverse_implies_negation (P : Prop) : 
  (¬P → P) → (P → ¬P) := by sorry

-- Proposition 2
theorem angle_60_iff_arithmetic_sequence (A B C : ℝ) : 
  (B = 60 ∧ A + B + C = 180) ↔ (∃ d : ℝ, A = B - d ∧ C = B + d) := by sorry

-- Proposition 3 (counterexample)
theorem not_necessary_condition_xy : 
  ∃ x y : ℝ, x + y > 3 ∧ x * y > 2 ∧ ¬(x > 1 ∧ y > 2) := by sorry

-- Proposition 4 (counterexample)
theorem not_necessary_condition_ab : 
  ∃ a b m : ℝ, a < b ∧ ¬(a * m^2 < b * m^2) := by sorry

end NUMINAMATH_CALUDE_inverse_implies_negation_angle_60_iff_arithmetic_sequence_not_necessary_condition_xy_not_necessary_condition_ab_l4020_402043


namespace NUMINAMATH_CALUDE_male_outnumber_female_l4020_402004

theorem male_outnumber_female (total : ℕ) (male : ℕ) 
  (h1 : total = 928) 
  (h2 : male = 713) : 
  male - (total - male) = 498 := by
  sorry

end NUMINAMATH_CALUDE_male_outnumber_female_l4020_402004


namespace NUMINAMATH_CALUDE_park_area_l4020_402051

/-- Represents a rectangular park with sides in ratio 3:2 -/
structure RectangularPark where
  x : ℝ
  length : ℝ := 3 * x
  width : ℝ := 2 * x

/-- The cost of fencing in pence per meter -/
def fencing_cost_per_meter : ℝ := 40

/-- The total cost of fencing in dollars -/
def total_fencing_cost : ℝ := 100

theorem park_area (park : RectangularPark) : 
  (2 * (park.length + park.width) * fencing_cost_per_meter / 100 = total_fencing_cost) →
  (park.length * park.width = 3750) := by
  sorry

end NUMINAMATH_CALUDE_park_area_l4020_402051


namespace NUMINAMATH_CALUDE_smallest_coprime_to_210_l4020_402023

def is_relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem smallest_coprime_to_210 :
  ∀ x : ℕ, x > 1 → x < 11 → ¬(is_relatively_prime x 210) ∧ is_relatively_prime 11 210 :=
by sorry

end NUMINAMATH_CALUDE_smallest_coprime_to_210_l4020_402023


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l4020_402008

theorem systematic_sampling_interval 
  (total_numbers : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_numbers = 2014) 
  (h2 : sample_size = 100) :
  (total_numbers - total_numbers % sample_size) / sample_size = 20 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l4020_402008


namespace NUMINAMATH_CALUDE_property_implies_increasing_l4020_402066

-- Define the property that (f(a) - f(b)) / (a - b) > 0 for all distinct a and b
def satisfies_property (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, a ≠ b → (f a - f b) / (a - b) > 0

-- Define what it means for a function to be increasing
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- State the theorem
theorem property_implies_increasing (f : ℝ → ℝ) :
  satisfies_property f → is_increasing f :=
by
  sorry

end NUMINAMATH_CALUDE_property_implies_increasing_l4020_402066


namespace NUMINAMATH_CALUDE_couples_satisfy_handshake_equation_l4020_402068

/-- The number of couples at a gathering where each person shakes hands with everyone
    except themselves and their partner, resulting in a total of 31,000 handshakes. -/
def num_couples : ℕ := 125

/-- The total number of handshakes at the gathering. -/
def total_handshakes : ℕ := 31000

/-- Theorem stating that the number of couples satisfies the equation derived from
    the handshake conditions. -/
theorem couples_satisfy_handshake_equation :
  2 * (num_couples * num_couples) - 2 * num_couples = total_handshakes :=
by sorry

end NUMINAMATH_CALUDE_couples_satisfy_handshake_equation_l4020_402068


namespace NUMINAMATH_CALUDE_area_of_right_triangle_with_inscribed_circle_l4020_402087

/-- A right triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- Length of the leg divided by the point of tangency -/
  a : ℝ
  /-- Length of the other leg -/
  b : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- The leg a is divided into segments of 6 and 10 by the point of tangency -/
  h_a : a = 16
  /-- The radius of the inscribed circle is 6 -/
  h_r : r = 6
  /-- The semi-perimeter of the triangle -/
  p : ℝ
  /-- Relation between semi-perimeter and leg b -/
  h_p : p = b + 10

/-- The area of the right triangle with an inscribed circle is 240 -/
theorem area_of_right_triangle_with_inscribed_circle 
  (t : RightTriangleWithInscribedCircle) : t.a * t.b / 2 = 240 := by
  sorry

end NUMINAMATH_CALUDE_area_of_right_triangle_with_inscribed_circle_l4020_402087


namespace NUMINAMATH_CALUDE_pool_depth_l4020_402091

/-- Represents the dimensions and properties of a rectangular pool -/
structure Pool :=
  (length : ℝ)
  (width : ℝ)
  (depth : ℝ)
  (chlorine_coverage : ℝ)
  (chlorine_cost : ℝ)
  (money_spent : ℝ)

/-- Theorem stating the depth of the pool given the conditions -/
theorem pool_depth (p : Pool) 
  (h1 : p.length = 10)
  (h2 : p.width = 8)
  (h3 : p.chlorine_coverage = 120)
  (h4 : p.chlorine_cost = 3)
  (h5 : p.money_spent = 12) :
  p.depth = 6 := by
  sorry

#check pool_depth

end NUMINAMATH_CALUDE_pool_depth_l4020_402091


namespace NUMINAMATH_CALUDE_negation_equivalence_l4020_402013

theorem negation_equivalence (a : ℝ) :
  (¬ ∀ x > 1, 2^x - a > 0) ↔ (∃ x > 1, 2^x - a ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l4020_402013


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l4020_402017

theorem necessary_not_sufficient_condition (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) :
  (∃ y, 0 < y ∧ y < Real.pi / 2 ∧ (Real.sqrt y - 1 / Real.sin y < 0) ∧ ¬(1 / Real.sin y - y > 0)) ∧
  (∀ z, 0 < z ∧ z < Real.pi / 2 ∧ (1 / Real.sin z - z > 0) → (Real.sqrt z - 1 / Real.sin z < 0)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l4020_402017


namespace NUMINAMATH_CALUDE_total_distance_apart_l4020_402036

/-- Represents the speeds of a skater for three hours -/
structure SkaterSpeeds where
  hour1 : ℝ
  hour2 : ℝ
  hour3 : ℝ

/-- Calculates the total distance traveled by a skater given their speeds -/
def totalDistance (speeds : SkaterSpeeds) : ℝ :=
  speeds.hour1 + speeds.hour2 + speeds.hour3

/-- Ann's skating speeds for each hour -/
def annSpeeds : SkaterSpeeds :=
  { hour1 := 6, hour2 := 8, hour3 := 4 }

/-- Glenda's skating speeds for each hour -/
def glendaSpeeds : SkaterSpeeds :=
  { hour1 := 8, hour2 := 5, hour3 := 9 }

/-- Theorem stating the total distance between Ann and Glenda after three hours -/
theorem total_distance_apart : totalDistance annSpeeds + totalDistance glendaSpeeds = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_apart_l4020_402036


namespace NUMINAMATH_CALUDE_derivative_exp_cos_l4020_402035

theorem derivative_exp_cos (x : ℝ) : 
  deriv (λ x => Real.exp x * Real.cos x) x = Real.exp x * (Real.cos x - Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_exp_cos_l4020_402035


namespace NUMINAMATH_CALUDE_tangent_y_intercept_l4020_402006

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + 11

-- Define the point of tangency
def P : ℝ × ℝ := (1, 12)

-- Theorem statement
theorem tangent_y_intercept :
  let m := (2 : ℝ) -- Slope of the tangent line
  let b := P.2 - m * P.1 -- y-intercept of the tangent line
  b = 10 := by sorry

end NUMINAMATH_CALUDE_tangent_y_intercept_l4020_402006


namespace NUMINAMATH_CALUDE_aquarium_fish_count_l4020_402000

/-- The number of stingrays in the aquarium -/
def num_stingrays : ℕ := 28

/-- The number of sharks in the aquarium -/
def num_sharks : ℕ := 2 * num_stingrays

/-- The total number of fish (sharks and stingrays) in the aquarium -/
def total_fish : ℕ := num_sharks + num_stingrays

/-- Theorem stating that the total number of fish in the aquarium is 84 -/
theorem aquarium_fish_count : total_fish = 84 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_fish_count_l4020_402000


namespace NUMINAMATH_CALUDE_pickle_theorem_l4020_402042

def pickle_problem (sammy_slices tammy_slices ron_slices : ℕ) : Prop :=
  tammy_slices = 2 * sammy_slices →
  sammy_slices = 15 →
  ron_slices = 24 →
  (tammy_slices - ron_slices : ℚ) / tammy_slices * 100 = 20

theorem pickle_theorem : pickle_problem 15 30 24 := by sorry

end NUMINAMATH_CALUDE_pickle_theorem_l4020_402042


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l4020_402058

theorem min_value_geometric_sequence (b₁ b₂ b₃ : ℝ) (s : ℝ) :
  b₁ = 2 →
  b₂ = b₁ * s →
  b₃ = b₂ * s →
  ∃ (min : ℝ), min = -9/8 ∧ ∀ (s : ℝ), 3*b₂ + 4*b₃ ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l4020_402058


namespace NUMINAMATH_CALUDE_perfect_squares_between_100_and_400_l4020_402003

theorem perfect_squares_between_100_and_400 : 
  (Finset.filter (fun n => 100 < n^2 ∧ n^2 < 400) (Finset.range 20)).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_between_100_and_400_l4020_402003


namespace NUMINAMATH_CALUDE_hotel_halls_first_wing_l4020_402046

/-- Represents the number of halls on each floor of the first wing -/
def halls_first_wing : ℕ := sorry

/-- Represents the total number of rooms in the hotel -/
def total_rooms : ℕ := 4248

/-- Represents the number of floors in the first wing -/
def floors_first_wing : ℕ := 9

/-- Represents the number of rooms in each hall of the first wing -/
def rooms_per_hall_first_wing : ℕ := 32

/-- Represents the number of floors in the second wing -/
def floors_second_wing : ℕ := 7

/-- Represents the number of halls on each floor of the second wing -/
def halls_second_wing : ℕ := 9

/-- Represents the number of rooms in each hall of the second wing -/
def rooms_per_hall_second_wing : ℕ := 40

theorem hotel_halls_first_wing : 
  halls_first_wing * floors_first_wing * rooms_per_hall_first_wing + 
  floors_second_wing * halls_second_wing * rooms_per_hall_second_wing = total_rooms ∧ 
  halls_first_wing = 6 := by sorry

end NUMINAMATH_CALUDE_hotel_halls_first_wing_l4020_402046


namespace NUMINAMATH_CALUDE_function_value_at_50_l4020_402055

theorem function_value_at_50 (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (x^2 + x) + 2 * f (x^2 - 3*x + 2) = 9*x^2 - 15*x) :
  f 50 = 146 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_50_l4020_402055


namespace NUMINAMATH_CALUDE_mara_janet_ratio_l4020_402089

/-- Represents the number of cards each person has -/
structure Cards where
  brenda : ℕ
  janet : ℕ
  mara : ℕ

/-- The conditions of the card problem -/
def card_problem (c : Cards) : Prop :=
  c.janet = c.brenda + 9 ∧
  ∃ k : ℕ, c.mara = k * c.janet ∧
  c.brenda + c.janet + c.mara = 211 ∧
  c.mara = 150 - 40

/-- The theorem stating the ratio of Mara's cards to Janet's cards -/
theorem mara_janet_ratio (c : Cards) :
  card_problem c → c.mara = 2 * c.janet :=
by
  sorry

#check mara_janet_ratio

end NUMINAMATH_CALUDE_mara_janet_ratio_l4020_402089
