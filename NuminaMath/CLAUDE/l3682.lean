import Mathlib

namespace NUMINAMATH_CALUDE_simplify_fraction_l3682_368222

theorem simplify_fraction : 5 * (21 / 6) * (18 / -63) = -5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3682_368222


namespace NUMINAMATH_CALUDE_rectangular_prism_width_zero_l3682_368209

/-- A rectangular prism with given dimensions --/
structure RectangularPrism where
  l : ℝ  -- length
  h : ℝ  -- height
  d : ℝ  -- diagonal
  w : ℝ  -- width

/-- The theorem stating that a rectangular prism with length 6, height 8, and diagonal 10 has width 0 --/
theorem rectangular_prism_width_zero (p : RectangularPrism) 
  (hl : p.l = 6) 
  (hh : p.h = 8) 
  (hd : p.d = 10) : 
  p.w = 0 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_width_zero_l3682_368209


namespace NUMINAMATH_CALUDE_number_problem_l3682_368217

theorem number_problem (x : ℝ) : (x / 6) * 12 = 15 → x = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3682_368217


namespace NUMINAMATH_CALUDE_sasha_plucked_leaves_l3682_368212

/-- The number of leaves Sasha plucked -/
def leaves_plucked (apple_trees poplar_trees masha_last_apple sasha_start_apple unphotographed : ℕ) : ℕ :=
  (apple_trees + poplar_trees) - (sasha_start_apple - 1) - unphotographed

/-- Theorem stating the number of leaves Sasha plucked -/
theorem sasha_plucked_leaves :
  leaves_plucked 17 18 10 8 13 = 22 := by
  sorry

#eval leaves_plucked 17 18 10 8 13

end NUMINAMATH_CALUDE_sasha_plucked_leaves_l3682_368212


namespace NUMINAMATH_CALUDE_distance_to_yz_plane_l3682_368215

/-- The distance from a point to the yz-plane -/
def distToYZPlane (x y z : ℝ) : ℝ := |x|

/-- The distance from a point to the x-axis -/
def distToXAxis (x y z : ℝ) : ℝ := |y|

/-- Point P satisfies the given conditions -/
def satisfiesConditions (x y z : ℝ) : Prop :=
  y = -6 ∧ x^2 + z^2 = 36 ∧ distToXAxis x y z = (1/2) * distToYZPlane x y z

theorem distance_to_yz_plane (x y z : ℝ) 
  (h : satisfiesConditions x y z) : distToYZPlane x y z = 12 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_yz_plane_l3682_368215


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3682_368281

/-- The hyperbola struct represents a hyperbola with semi-major axis a and semi-minor axis b. -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- The Point struct represents a point in 2D space. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The Focus struct represents a focus point of the hyperbola. -/
structure Focus where
  x : ℝ
  y : ℝ

/-- Represents the condition that P is on the right branch of the hyperbola. -/
def is_on_right_branch (h : Hyperbola) (p : Point) : Prop :=
  (p.x^2 / h.a^2) - (p.y^2 / h.b^2) = 1 ∧ p.x > 0

/-- Represents the condition that the distance from O to PF₁ equals the real semi-axis. -/
def distance_condition (h : Hyperbola) (p : Point) (f₁ : Focus) : Prop :=
  ∃ (d : ℝ), d = h.a ∧ d = abs (f₁.y * p.x - f₁.x * p.y) / Real.sqrt ((p.x - f₁.x)^2 + (p.y - f₁.y)^2)

/-- The main theorem stating the eccentricity of the hyperbola under given conditions. -/
theorem hyperbola_eccentricity (h : Hyperbola) (p : Point) (f₁ f₂ : Focus) :
  is_on_right_branch h p →
  (p.x - f₂.x)^2 + (p.y - f₂.y)^2 = (f₁.x - f₂.x)^2 + (f₁.y - f₂.y)^2 →
  distance_condition h p f₁ →
  let e := Real.sqrt (1 + h.b^2 / h.a^2)
  e = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3682_368281


namespace NUMINAMATH_CALUDE_shopping_lottery_largest_number_l3682_368243

/-- Represents the largest number in a systematic sample -/
def largest_sample_number (total : ℕ) (start : ℕ) (interval : ℕ) : ℕ :=
  start + interval * ((total - start) / interval)

/-- The problem statement as a theorem -/
theorem shopping_lottery_largest_number :
  let total := 160
  let start := 7
  let second := 23
  let interval := second - start
  largest_sample_number total start interval = 151 := by
  sorry

#eval largest_sample_number 160 7 16

end NUMINAMATH_CALUDE_shopping_lottery_largest_number_l3682_368243


namespace NUMINAMATH_CALUDE_circle_polygons_l3682_368219

/-- The number of points marked on the circle -/
def n : ℕ := 12

/-- The number of distinct convex polygons with 3 or more sides -/
def num_polygons : ℕ := 2^n - (n.choose 0 + n.choose 1 + n.choose 2)

theorem circle_polygons :
  num_polygons = 4017 :=
sorry

end NUMINAMATH_CALUDE_circle_polygons_l3682_368219


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l3682_368204

theorem greatest_two_digit_multiple_of_17 : 
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 17 ∣ n → n ≤ 85 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l3682_368204


namespace NUMINAMATH_CALUDE_elise_remaining_money_l3682_368273

/-- Calculates the remaining money for Elise --/
def remaining_money (initial saved comic_book puzzle : ℕ) : ℕ :=
  initial + saved - (comic_book + puzzle)

/-- Theorem: Elise's remaining money is $1 --/
theorem elise_remaining_money :
  remaining_money 8 13 2 18 = 1 := by
  sorry

end NUMINAMATH_CALUDE_elise_remaining_money_l3682_368273


namespace NUMINAMATH_CALUDE_initial_time_theorem_l3682_368221

/-- Given a distance of 720 km, if increasing the initial time by 3/2 results in a speed of 80 kmph,
    then the initial time taken to cover the distance was 6 hours. -/
theorem initial_time_theorem (t : ℝ) (h1 : t > 0) : 
  (720 : ℝ) / ((3/2) * t) = 80 → t = 6 := by
  sorry

end NUMINAMATH_CALUDE_initial_time_theorem_l3682_368221


namespace NUMINAMATH_CALUDE_curve_is_parabola_l3682_368210

/-- The curve defined by √X + √Y = 1 is a parabola -/
theorem curve_is_parabola :
  ∃ (a b c : ℝ) (h : a ≠ 0),
    ∀ (x y : ℝ),
      (Real.sqrt x + Real.sqrt y = 1) ↔ (y = a * x^2 + b * x + c) :=
sorry

end NUMINAMATH_CALUDE_curve_is_parabola_l3682_368210


namespace NUMINAMATH_CALUDE_circle_with_n_integer_points_l3682_368207

/-- A point on the coordinate plane with rational x-coordinate and irrational y-coordinate -/
structure SpecialPoint where
  x : ℚ
  y : ℝ
  y_irrational : Irrational y

/-- The number of integer points inside a circle -/
def IntegerPointsInside (center : ℝ × ℝ) (radius : ℝ) : ℕ :=
  sorry

/-- Theorem: For any non-negative integer n, there exists a circle on the coordinate plane
    that contains exactly n integer points in its interior -/
theorem circle_with_n_integer_points (n : ℕ) :
  ∃ (center : ℝ × ℝ) (radius : ℝ), IntegerPointsInside center radius = n :=
sorry

end NUMINAMATH_CALUDE_circle_with_n_integer_points_l3682_368207


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l3682_368289

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), a.1 = t * b.1 ∧ a.2 = t * b.2

theorem parallel_vectors_k_value :
  ∀ (k : ℝ),
  let a : ℝ × ℝ := (2*k - 3, -6)
  let c : ℝ × ℝ := (2, 1)
  are_parallel a c → k = -9/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l3682_368289


namespace NUMINAMATH_CALUDE_point_set_is_hyperbola_l3682_368246

-- Define the set of points (x, y) based on the given parametric equations
def point_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, t ≠ 0 ∧ p.1 = (2 * t + 1) / t ∧ p.2 = (t - 2) / t}

-- Theorem stating that the point_set forms a hyperbola
theorem point_set_is_hyperbola : 
  ∃ a b c d e f : ℝ, a ≠ 0 ∧ 
    (∀ p : ℝ × ℝ, p ∈ point_set ↔ 
      a * p.1 * p.1 + b * p.1 * p.2 + c * p.2 * p.2 + d * p.1 + e * p.2 + f = 0) ∧
    b * b - 4 * a * c > 0 := by
  sorry

end NUMINAMATH_CALUDE_point_set_is_hyperbola_l3682_368246


namespace NUMINAMATH_CALUDE_expression_evaluation_l3682_368254

theorem expression_evaluation (a b : ℤ) (ha : a = -1) (hb : b = 1) :
  (a^2 * b - 4 * a * b^2 - 1) - 3 * (a * b^2 - 2 * a^2 * b + 1) = 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3682_368254


namespace NUMINAMATH_CALUDE_perpendicular_lines_l3682_368242

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- Definition of line l1 -/
def l1 (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y => x + a * y - 1 = 0

/-- Definition of line l2 -/
def l2 (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y => (a^2 - 2) * x + y + 2 = 0

/-- Theorem: If l1 and l2 are perpendicular, then a = -2 or a = 1 -/
theorem perpendicular_lines (a : ℝ) :
  perpendicular (-a) (1 / (a^2 - 2)) → a = -2 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l3682_368242


namespace NUMINAMATH_CALUDE_share_distribution_l3682_368232

theorem share_distribution (total : ℕ) (a b c : ℕ) : 
  total = 120 →
  a = b + 20 →
  c = a + 20 →
  total = a + b + c →
  b = 20 := by
sorry

end NUMINAMATH_CALUDE_share_distribution_l3682_368232


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3682_368298

theorem inequality_system_solution :
  ∀ x : ℝ, (3 * x + 1 ≥ 7 ∧ 4 * x - 3 < 9) ↔ (2 ≤ x ∧ x < 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3682_368298


namespace NUMINAMATH_CALUDE_floor_abs_sum_abs_floor_l3682_368225

theorem floor_abs_sum_abs_floor : ⌊|(-5.7:ℝ)|⌋ + |⌊(-5.7:ℝ)⌋| = 11 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_sum_abs_floor_l3682_368225


namespace NUMINAMATH_CALUDE_winnie_lollipops_left_l3682_368250

/-- The number of lollipops Winnie has left after distributing them equally among her friends -/
def lollipops_left (cherry wintergreen grape shrimp friends : ℕ) : ℕ :=
  (cherry + wintergreen + grape + shrimp) % friends

theorem winnie_lollipops_left :
  lollipops_left 32 150 7 280 14 = 7 := by
  sorry

end NUMINAMATH_CALUDE_winnie_lollipops_left_l3682_368250


namespace NUMINAMATH_CALUDE_modified_cube_surface_area_l3682_368239

/-- Represents a 9x9x9 cube composed of 3x3x3 subcubes -/
structure LargeCube where
  subcubes : Fin 3 → Fin 3 → Fin 3 → Unit

/-- Represents the modified structure after removing center cubes and facial units -/
structure ModifiedCube where
  remaining_subcubes : Fin 20 → Unit
  removed_centers : Unit
  removed_facial_units : Unit

/-- Calculates the surface area of the modified cube structure -/
def surface_area (cube : ModifiedCube) : ℕ :=
  sorry

/-- Theorem stating that the surface area of the modified cube is 1056 square units -/
theorem modified_cube_surface_area :
  ∀ (cube : LargeCube),
  ∃ (modified : ModifiedCube),
  surface_area modified = 1056 :=
sorry

end NUMINAMATH_CALUDE_modified_cube_surface_area_l3682_368239


namespace NUMINAMATH_CALUDE_cylindrical_fortress_pi_l3682_368278

/-- Given a cylindrical fortress with circumference 38 feet and height 11 feet,
    if its volume is calculated as V = (1/12) * (circumference^2 * height),
    then the implied value of π is 3. -/
theorem cylindrical_fortress_pi (circumference height : ℝ) (π : ℝ) : 
  circumference = 38 →
  height = 11 →
  (1/12) * (circumference^2 * height) = π * (circumference / (2 * π))^2 * height →
  π = 3 := by
  sorry

end NUMINAMATH_CALUDE_cylindrical_fortress_pi_l3682_368278


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3682_368284

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 10 * x + 6
  ∃ x₁ x₂ : ℝ, x₁ = 5/3 + Real.sqrt 7/3 ∧ 
             x₂ = 5/3 - Real.sqrt 7/3 ∧
             f x₁ = 0 ∧ f x₂ = 0 ∧
             ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3682_368284


namespace NUMINAMATH_CALUDE_athena_spent_14_l3682_368200

/-- The total amount Athena spent on snacks for her friends -/
def total_spent (sandwich_price : ℝ) (sandwich_count : ℕ) (drink_price : ℝ) (drink_count : ℕ) : ℝ :=
  sandwich_price * sandwich_count + drink_price * drink_count

/-- Theorem stating that Athena spent $14 on snacks -/
theorem athena_spent_14 :
  total_spent 3 3 2.5 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_athena_spent_14_l3682_368200


namespace NUMINAMATH_CALUDE_alice_prob_after_three_turns_l3682_368288

/-- Represents the possessor of the ball -/
inductive Possessor : Type
| Alice : Possessor
| Bob : Possessor

/-- The game state after a turn -/
structure GameState :=
  (possessor : Possessor)

/-- The probability of Alice having the ball after one turn, given the current possessor -/
def aliceProbAfterOneTurn (current : Possessor) : ℚ :=
  match current with
  | Possessor.Alice => 2/3
  | Possessor.Bob => 3/5

/-- The probability of Alice having the ball after three turns, given Alice starts -/
def aliceProbAfterThreeTurns : ℚ := 7/45

theorem alice_prob_after_three_turns :
  aliceProbAfterThreeTurns = 7/45 :=
sorry

end NUMINAMATH_CALUDE_alice_prob_after_three_turns_l3682_368288


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l3682_368226

theorem sum_of_fractions_equals_one (x y z : ℝ) (h : x * y * z = 1) :
  (1 / (1 + x + x * y)) + (1 / (1 + y + y * z)) + (1 / (1 + z + z * x)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l3682_368226


namespace NUMINAMATH_CALUDE_church_cookies_total_l3682_368202

/-- Calculates the total number of cookies baked by church members -/
theorem church_cookies_total (members : ℕ) (sheets_per_member : ℕ) (cookies_per_sheet : ℕ) : 
  members = 100 → sheets_per_member = 10 → cookies_per_sheet = 16 → 
  members * sheets_per_member * cookies_per_sheet = 16000 := by
  sorry

end NUMINAMATH_CALUDE_church_cookies_total_l3682_368202


namespace NUMINAMATH_CALUDE_sum_of_squared_distances_l3682_368228

/-- Triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Circumcenter of a triangle -/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- Orthocenter of a triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- Circumradius of a triangle -/
def circumradius (t : Triangle) : ℝ := sorry

/-- A point on the circumcircle of a triangle -/
def point_on_circumcircle (t : Triangle) : ℝ × ℝ := sorry

/-- Squared distance between two points -/
def squared_distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem sum_of_squared_distances (t : Triangle) :
  let O := circumcenter t
  let H := orthocenter t
  let R := circumradius t
  let P := point_on_circumcircle t
  squared_distance P t.A + squared_distance P t.B + squared_distance P t.C + squared_distance P H = 8 * R^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_distances_l3682_368228


namespace NUMINAMATH_CALUDE_factorial_quotient_equals_56_l3682_368206

theorem factorial_quotient_equals_56 :
  ∃! (n : ℕ), n > 0 ∧ n.factorial / (n - 2).factorial = 56 := by
  sorry

end NUMINAMATH_CALUDE_factorial_quotient_equals_56_l3682_368206


namespace NUMINAMATH_CALUDE_sequence_bounded_above_l3682_368280

/-- The sequence {aₙ} defined by the given recurrence relation is bounded above. -/
theorem sequence_bounded_above (α : ℝ) (h_α : α > 1) :
  ∃ (M : ℝ), ∀ (a : ℕ → ℝ), 
    (a 1 ∈ Set.Ioo 0 1) → 
    (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + (a n / n)^α) → 
    (∀ n : ℕ, n ≥ 1 → a n ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_sequence_bounded_above_l3682_368280


namespace NUMINAMATH_CALUDE_bat_wings_area_l3682_368267

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  topLeft : Point
  bottomRight : Point

/-- Calculates the area of a triangle given three points -/
def triangleArea (a b c : Point) : ℝ :=
  0.5 * abs ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y))

/-- Theorem: The area of the "bat wings" in the given rectangle configuration is 7.5 -/
theorem bat_wings_area (rect : Rectangle)
  (h_width : rect.bottomRight.x - rect.topLeft.x = 4)
  (h_height : rect.bottomRight.y - rect.topLeft.y = 5)
  (j : Point) (k : Point) (l : Point) (m : Point)
  (h_j : j = rect.topLeft)
  (h_k : k.x - j.x = 2 ∧ k.y = rect.bottomRight.y)
  (h_l : l.x = rect.bottomRight.x ∧ l.y - k.y = 2)
  (h_m : m.x = rect.topLeft.x ∧ m.y = rect.bottomRight.y)
  (h_mj : m.y - j.y = 2)
  (h_jk : k.x - j.x = 2)
  (h_kl : l.x - k.x = 2) :
  triangleArea j m k + triangleArea j k l = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_bat_wings_area_l3682_368267


namespace NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_l3682_368247

theorem quadratic_equation_distinct_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    (k - 1) * x₁^2 - 2 * x₁ + 3 = 0 ∧
    (k - 1) * x₂^2 - 2 * x₂ + 3 = 0) ↔
  (k < 4/3 ∧ k ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_l3682_368247


namespace NUMINAMATH_CALUDE_angle_a_is_30_angle_b_is_150_polygon_sides_is_12_march_day_is_24_l3682_368272

-- Define the angle a from the geometric figure
def a : ℝ := 30

-- Define the angle b
def b : ℝ := 150

-- Define the number of sides n in the regular polygon
def n : ℕ := 12

-- Define k as the day of March
def k : ℕ := 24

-- Theorem 1: The angle a in the given geometric figure is 30°
theorem angle_a_is_30 : a = 30 := by sorry

-- Theorem 2: If sin(30° + 210°) = cos b° and 90° < b < 180°, then b = 150°
theorem angle_b_is_150 (h1 : Real.sin (30 + 210) = Real.cos b) (h2 : 90 < b ∧ b < 180) : b = 150 := by sorry

-- Theorem 3: If each interior angle of an n-sided regular polygon is 150°, then n = 12
theorem polygon_sides_is_12 (h : (n - 2) * 180 / n = 150) : n = 12 := by sorry

-- Theorem 4: If the nth day of March is Friday, the kth day is Wednesday, and 20 < k < 25, then k = 24
theorem march_day_is_24 (h1 : k % 7 = (n + 3) % 7) (h2 : 20 < k ∧ k < 25) : k = 24 := by sorry

end NUMINAMATH_CALUDE_angle_a_is_30_angle_b_is_150_polygon_sides_is_12_march_day_is_24_l3682_368272


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l3682_368270

/-- The lateral surface area of a cylinder with base diameter and height both equal to 2 cm is 4π cm² -/
theorem cylinder_lateral_surface_area :
  ∀ (d h r : ℝ),
  d = 2 →  -- base diameter is 2 cm
  h = 2 →  -- height is 2 cm
  r = d / 2 →  -- radius is half the diameter
  2 * Real.pi * r * h = 4 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l3682_368270


namespace NUMINAMATH_CALUDE_expand_and_simplify_l3682_368283

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l3682_368283


namespace NUMINAMATH_CALUDE_ferris_wheel_seats_l3682_368249

/-- The number of people each seat can hold -/
def seat_capacity : ℕ := 6

/-- The total number of people the Ferris wheel can hold -/
def total_capacity : ℕ := 84

/-- The number of seats on the Ferris wheel -/
def num_seats : ℕ := total_capacity / seat_capacity

theorem ferris_wheel_seats : num_seats = 14 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_seats_l3682_368249


namespace NUMINAMATH_CALUDE_calculate_rates_l3682_368248

/-- Represents the rates and quantities in the problem -/
structure Rates where
  b : ℕ  -- number of bananas Charles cooked
  d : ℕ  -- number of dishes Sandrine washed
  r1 : ℚ  -- rate at which Charles picks pears (pears per hour)
  r2 : ℚ  -- rate at which Charles cooks bananas (bananas per hour)
  r3 : ℚ  -- rate at which Sandrine washes dishes (dishes per hour)

/-- The main theorem representing the problem -/
theorem calculate_rates (rates : Rates) : 
  rates.d = rates.b + 10 ∧ 
  rates.b = 3 * 50 ∧ 
  rates.r1 = 50 / 4 ∧ 
  rates.r2 = rates.b / 2 ∧ 
  rates.r3 = rates.d / 5 → 
  rates.r1 = 12.5 ∧ rates.r2 = 75 ∧ rates.r3 = 32 := by
  sorry

end NUMINAMATH_CALUDE_calculate_rates_l3682_368248


namespace NUMINAMATH_CALUDE_sine_function_translation_l3682_368237

theorem sine_function_translation (ω : ℝ) (h_ω_pos : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x)
  let g : ℝ → ℝ := λ x ↦ f (x + π / (4 * ω))
  (∀ x : ℝ, g (2 * ω - x) = g x) →
  (∀ x y : ℝ, -ω < x ∧ x < y ∧ y < ω → g x < g y) →
  ω = Real.sqrt (π / 2) := by
sorry

end NUMINAMATH_CALUDE_sine_function_translation_l3682_368237


namespace NUMINAMATH_CALUDE_unique_solution_for_m_l3682_368257

theorem unique_solution_for_m :
  ∀ (x y m : ℚ),
  (2 * x + y = 3 * m) →
  (x - 4 * y = -2 * m) →
  (y + 2 * m = 1 + x) →
  m = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_for_m_l3682_368257


namespace NUMINAMATH_CALUDE_consecutive_cs_majors_probability_l3682_368211

/-- The number of people sitting at the round table -/
def total_people : ℕ := 12

/-- The number of computer science majors -/
def cs_majors : ℕ := 5

/-- The number of engineering majors -/
def eng_majors : ℕ := 4

/-- The number of art majors -/
def art_majors : ℕ := 3

/-- The probability of all computer science majors sitting consecutively -/
def consecutive_cs_prob : ℚ := 1 / 66

theorem consecutive_cs_majors_probability :
  (total_people = cs_majors + eng_majors + art_majors) →
  (consecutive_cs_prob = (total_people : ℚ) / (total_people.choose cs_majors)) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_cs_majors_probability_l3682_368211


namespace NUMINAMATH_CALUDE_isosceles_triangle_exists_l3682_368240

/-- A regular polygon with 101 vertices -/
structure RegularPolygon101 where
  vertices : Fin 101 → ℝ × ℝ

/-- A selection of 51 vertices from a 101-regular polygon -/
structure Selection51 (polygon : RegularPolygon101) where
  selected : Fin 51 → Fin 101
  distinct : ∀ i j, i ≠ j → selected i ≠ selected j

/-- Three points form an isosceles triangle -/
def IsIsoscelesTriangle (a b c : ℝ × ℝ) : Prop :=
  (a.1 - b.1)^2 + (a.2 - b.2)^2 = (a.1 - c.1)^2 + (a.2 - c.2)^2 ∨
  (b.1 - a.1)^2 + (b.2 - a.2)^2 = (b.1 - c.1)^2 + (b.2 - c.2)^2 ∨
  (c.1 - a.1)^2 + (c.2 - a.2)^2 = (c.1 - b.1)^2 + (c.2 - b.2)^2

/-- Main theorem: Among any 51 vertices of the 101-regular polygon, 
    there are three that form an isosceles triangle -/
theorem isosceles_triangle_exists (polygon : RegularPolygon101) 
  (selection : Selection51 polygon) : 
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    IsIsoscelesTriangle 
      (polygon.vertices (selection.selected i))
      (polygon.vertices (selection.selected j))
      (polygon.vertices (selection.selected k)) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_exists_l3682_368240


namespace NUMINAMATH_CALUDE_cookie_problem_l3682_368290

theorem cookie_problem :
  let n : ℕ := 1817
  (∀ m : ℕ, m > 0 ∧ m < n →
    ¬(m % 6 = 5 ∧ m % 7 = 3 ∧ m % 9 = 7 ∧ m % 11 = 10)) ∧
  (n % 6 = 5 ∧ n % 7 = 3 ∧ n % 9 = 7 ∧ n % 11 = 10) := by
  sorry

end NUMINAMATH_CALUDE_cookie_problem_l3682_368290


namespace NUMINAMATH_CALUDE_floor_sum_equals_140_l3682_368255

theorem floor_sum_equals_140 
  (p q r s : ℝ) 
  (pos_p : 0 < p) (pos_q : 0 < q) (pos_r : 0 < r) (pos_s : 0 < s)
  (sum_squares : p^2 + q^2 = 2512 ∧ r^2 + s^2 = 2512)
  (products : p * r = 1225 ∧ q * s = 1225) : 
  ⌊p + q + r + s⌋ = 140 := by
sorry

end NUMINAMATH_CALUDE_floor_sum_equals_140_l3682_368255


namespace NUMINAMATH_CALUDE_expression_evaluation_l3682_368230

theorem expression_evaluation :
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * 30^1001 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3682_368230


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3682_368264

theorem quadratic_inequality_solution (a : ℝ) (h : a < 0) :
  let solution := {x : ℝ | a * x^2 - (a - 1) * x - 1 < 0}
  ((-1 < a ∧ a < 0) → solution = {x | x < 1 ∨ x > -1/a}) ∧
  (a = -1 → solution = {x | x ≠ 1}) ∧
  (a < -1 → solution = {x | x < -1/a ∨ x > 1}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3682_368264


namespace NUMINAMATH_CALUDE_bella_prob_reach_edge_l3682_368271

/-- Represents a position on the 4x4 grid -/
inductive Position
| Central : Position
| NearEdge : Position
| Edge : Position

/-- Represents the possible directions of movement -/
inductive Direction
| Up | Down | Left | Right

/-- Represents the state of Bella's movement -/
structure BellaState where
  position : Position
  hops : Nat

/-- Transition function for Bella's movement -/
def transition (state : BellaState) (dir : Direction) : BellaState :=
  match state.position with
  | Position.Central => ⟨Position.NearEdge, state.hops + 1⟩
  | Position.NearEdge => 
      if state.hops < 5 then ⟨Position.Edge, state.hops + 1⟩
      else state
  | Position.Edge => state

/-- Probability of reaching an edge square within 5 hops -/
def prob_reach_edge (state : BellaState) : ℚ :=
  sorry

/-- Main theorem: Probability of reaching an edge square within 5 hops is 7/8 -/
theorem bella_prob_reach_edge :
  prob_reach_edge ⟨Position.Central, 0⟩ = 7/8 :=
sorry

end NUMINAMATH_CALUDE_bella_prob_reach_edge_l3682_368271


namespace NUMINAMATH_CALUDE_perimeter_plus_area_of_specific_parallelogram_l3682_368266

/-- A parallelogram in a 2D coordinate plane -/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Calculate the perimeter of a parallelogram -/
def perimeter (p : Parallelogram) : ℝ := sorry

/-- Calculate the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := sorry

/-- The sum of perimeter and area of a specific parallelogram -/
theorem perimeter_plus_area_of_specific_parallelogram :
  let p := Parallelogram.mk (2, 1) (7, 1) (5, 6) (10, 6)
  perimeter p + area p = 35 + 2 * Real.sqrt 34 := by sorry

end NUMINAMATH_CALUDE_perimeter_plus_area_of_specific_parallelogram_l3682_368266


namespace NUMINAMATH_CALUDE_problem_solution_l3682_368251

open Real

def p : Prop := ∀ x > 0, log x + 4 * x ≥ 3

def q : Prop := ∃ x₀ > 0, 8 * x₀ + 1 / (2 * x₀) ≤ 4

theorem problem_solution : (¬p ∧ q) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3682_368251


namespace NUMINAMATH_CALUDE_less_than_reciprocal_check_l3682_368216

def is_less_than_reciprocal (x : ℚ) : Prop :=
  x ≠ 0 ∧ x < 1 / x

theorem less_than_reciprocal_check :
  is_less_than_reciprocal (-3) ∧
  is_less_than_reciprocal (3/4) ∧
  ¬is_less_than_reciprocal (-1/2) ∧
  ¬is_less_than_reciprocal 3 ∧
  ¬is_less_than_reciprocal 0 :=
by sorry

end NUMINAMATH_CALUDE_less_than_reciprocal_check_l3682_368216


namespace NUMINAMATH_CALUDE_triangle_area_proof_l3682_368227

/-- The slope of the first line -/
def m₁ : ℚ := 3/4

/-- The slope of the second line -/
def m₂ : ℚ := -3/2

/-- The x-coordinate of the intersection point of the first two lines -/
def x₀ : ℚ := 4

/-- The y-coordinate of the intersection point of the first two lines -/
def y₀ : ℚ := 3

/-- The equation of the third line: ax + by = c -/
def a : ℚ := 1
def b : ℚ := 2
def c : ℚ := 12

/-- The area of the triangle -/
def triangle_area : ℚ := 15/4

theorem triangle_area_proof :
  let line1 := fun x => m₁ * (x - x₀) + y₀
  let line2 := fun x => m₂ * (x - x₀) + y₀
  let line3 := fun x y => a * x + b * y = c
  ∃ x₁ y₁ x₂ y₂,
    line1 x₁ = y₁ ∧ line3 x₁ y₁ ∧
    line2 x₂ = y₂ ∧ line3 x₂ y₂ ∧
    triangle_area = (1/2) * abs ((x₀ * (y₁ - y₂) + x₁ * (y₂ - y₀) + x₂ * (y₀ - y₁))) :=
sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l3682_368227


namespace NUMINAMATH_CALUDE_perfect_pairs_iff_even_l3682_368294

/-- A pair of integers (a, b) is perfect if ab + 1 is a perfect square. -/
def IsPerfectPair (a b : ℤ) : Prop :=
  ∃ k : ℤ, a * b + 1 = k ^ 2

/-- The set {1, ..., 2n} can be divided into n perfect pairs. -/
def CanDivideIntoPerfectPairs (n : ℕ) : Prop :=
  ∃ f : Fin n → Fin (2 * n) × Fin (2 * n),
    (∀ i : Fin n, IsPerfectPair (f i).1.val.succ (f i).2.val.succ) ∧
    (∀ i j : Fin n, i ≠ j → (f i).1 ≠ (f j).1 ∧ (f i).1 ≠ (f j).2 ∧ 
                            (f i).2 ≠ (f j).1 ∧ (f i).2 ≠ (f j).2)

/-- The main theorem: The set {1, ..., 2n} can be divided into n perfect pairs 
    if and only if n is even. -/
theorem perfect_pairs_iff_even (n : ℕ) :
  CanDivideIntoPerfectPairs n ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_perfect_pairs_iff_even_l3682_368294


namespace NUMINAMATH_CALUDE_first_digit_base_5_of_627_l3682_368259

theorem first_digit_base_5_of_627 :
  ∃ (d : ℕ) (r : ℕ), 627 = d * 5^4 + r ∧ d = 1 ∧ r < 5^4 := by
  sorry

end NUMINAMATH_CALUDE_first_digit_base_5_of_627_l3682_368259


namespace NUMINAMATH_CALUDE_similar_triangle_shortest_side_l3682_368223

theorem similar_triangle_shortest_side 
  (side1 : ℝ) 
  (hyp1 : ℝ) 
  (hyp2 : ℝ) 
  (is_right_triangle : side1^2 + (hyp1^2 - side1^2) = hyp1^2)
  (hyp1_positive : hyp1 > 0)
  (hyp2_positive : hyp2 > 0)
  (h_similar : hyp2 / hyp1 * side1 = 72) :
  ∃ (side2 : ℝ), side2 = 72 ∧ side2 ≤ hyp2 := by sorry

end NUMINAMATH_CALUDE_similar_triangle_shortest_side_l3682_368223


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3682_368268

/-- The equation of asymptotes for a hyperbola with given parameters -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) (he : Real.sqrt ((a^2 + b^2) / a^2) = 2) :
  ∃ (k : ℝ), k = Real.sqrt 3 ∧ 
  ∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1) → (y = k * x ∨ y = -k * x) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3682_368268


namespace NUMINAMATH_CALUDE_xy9z_divisible_by_132_l3682_368224

def is_form_xy9z (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), x < 10 ∧ y < 10 ∧ z < 10 ∧ n = x * 1000 + y * 100 + 90 + z

def valid_numbers : Set ℕ := {3696, 4092, 6996, 7392}

theorem xy9z_divisible_by_132 :
  ∀ n : ℕ, is_form_xy9z n ∧ 132 ∣ n ↔ n ∈ valid_numbers := by sorry

end NUMINAMATH_CALUDE_xy9z_divisible_by_132_l3682_368224


namespace NUMINAMATH_CALUDE_function_properties_l3682_368295

def f (x : ℝ) := -5 * x + 1

theorem function_properties :
  (∃ (x₁ x₂ x₃ : ℝ), f x₁ > 0 ∧ x₁ > 0 ∧ f x₂ < 0 ∧ x₂ > 0 ∧ f x₃ < 0 ∧ x₃ < 0) ∧
  (∀ x : ℝ, x > 1 → f x < 0) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l3682_368295


namespace NUMINAMATH_CALUDE_specific_parallelogram_area_l3682_368213

/-- A parallelogram in 2D space -/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Calculate the area of a parallelogram -/
def parallelogramArea (p : Parallelogram) : ℝ := sorry

/-- The specific parallelogram in the problem -/
def specificParallelogram : Parallelogram := {
  v1 := (0, 0)
  v2 := (4, 0)
  v3 := (1, 5)
  v4 := (5, 5)
}

/-- Theorem: The area of the specific parallelogram is 20 square units -/
theorem specific_parallelogram_area :
  parallelogramArea specificParallelogram = 20 := by sorry

end NUMINAMATH_CALUDE_specific_parallelogram_area_l3682_368213


namespace NUMINAMATH_CALUDE_cosine_range_in_triangle_l3682_368252

theorem cosine_range_in_triangle (A B C : EuclideanSpace ℝ (Fin 2)) 
  (h_AB : dist A B = 3)
  (h_AC : dist A C = 2)
  (h_BC : dist B C > Real.sqrt 2) :
  ∃ (cosA : ℝ), cosA = (dist A B)^2 + (dist A C)^2 - (dist B C)^2 / (2 * dist A B * dist A C) ∧ 
  -1 < cosA ∧ cosA < 11/12 :=
sorry

end NUMINAMATH_CALUDE_cosine_range_in_triangle_l3682_368252


namespace NUMINAMATH_CALUDE_expand_expression_l3682_368203

theorem expand_expression (y : ℝ) : 12 * (3 * y + 7) = 36 * y + 84 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3682_368203


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_two_l3682_368231

theorem sum_of_x_and_y_is_two (x y : ℝ) 
  (hx : (x - 1)^3 + 1997*(x - 1) = -1)
  (hy : (y - 1)^3 + 1997*(y - 1) = 1) : 
  x + y = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_two_l3682_368231


namespace NUMINAMATH_CALUDE_sequence_sum_l3682_368244

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence with positive terms -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, b (n + 1) = b n * q

theorem sequence_sum (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence b →
  b 1 = 1 →
  b 3 = b 2 + 2 →
  b 4 = a 3 + a 5 →
  b 5 = a 4 + 2 * a 6 →
  a 2018 + b 9 = 2274 := by
  sorry


end NUMINAMATH_CALUDE_sequence_sum_l3682_368244


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l3682_368233

-- Define an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Define the transformed function
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (2*x + 1)

-- Theorem statement
theorem axis_of_symmetry (f : ℝ → ℝ) (h : even_function f) :
  ∀ x : ℝ, g f ((-1) + x) = g f ((-1) - x) :=
sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_l3682_368233


namespace NUMINAMATH_CALUDE_simplify_polynomial_expression_l3682_368277

/-- Given two polynomials A and B in x and y, prove that 2A - B simplifies to a specific form. -/
theorem simplify_polynomial_expression (x y : ℝ) :
  let A := 2 * x^2 + x * y - 3
  let B := -x^2 + 2 * x * y - 1
  2 * A - B = 5 * x^2 - 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_expression_l3682_368277


namespace NUMINAMATH_CALUDE_num_lineups_eq_2277_l3682_368261

def team_size : ℕ := 15
def lineup_size : ℕ := 5
def special_players : ℕ := 3

/-- The number of possible lineups given the constraints -/
def num_lineups : ℕ :=
  3 * (Nat.choose (team_size - special_players) (lineup_size - 1)) +
  Nat.choose (team_size - special_players) lineup_size

/-- Theorem stating that the number of possible lineups is 2277 -/
theorem num_lineups_eq_2277 : num_lineups = 2277 := by
  sorry

end NUMINAMATH_CALUDE_num_lineups_eq_2277_l3682_368261


namespace NUMINAMATH_CALUDE_paper_cup_cost_theorem_l3682_368297

/-- The total number of pallets -/
def total_pallets : ℕ := 20

/-- The number of paper towel pallets -/
def paper_towel_pallets : ℕ := total_pallets / 2

/-- The number of tissue pallets -/
def tissue_pallets : ℕ := total_pallets / 4

/-- The number of paper plate pallets -/
def paper_plate_pallets : ℕ := total_pallets / 5

/-- The number of paper cup pallets -/
def paper_cup_pallets : ℕ := total_pallets - (paper_towel_pallets + tissue_pallets + paper_plate_pallets)

/-- The cost of a single paper cup pallet -/
def paper_cup_pallet_cost : ℕ := 50

/-- The total cost spent on paper cup pallets -/
def total_paper_cup_cost : ℕ := paper_cup_pallets * paper_cup_pallet_cost

theorem paper_cup_cost_theorem : total_paper_cup_cost = 50 := by
  sorry

end NUMINAMATH_CALUDE_paper_cup_cost_theorem_l3682_368297


namespace NUMINAMATH_CALUDE_power_relation_l3682_368287

theorem power_relation (a x y : ℝ) (ha : a > 0) (hx : a^x = 2) (hy : a^y = 3) :
  a^(x - 2*y) = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_power_relation_l3682_368287


namespace NUMINAMATH_CALUDE_trays_from_second_table_l3682_368262

theorem trays_from_second_table 
  (trays_per_trip : ℕ) 
  (total_trips : ℕ) 
  (trays_first_table : ℕ) 
  (h1 : trays_per_trip = 4) 
  (h2 : total_trips = 9) 
  (h3 : trays_first_table = 20) : 
  trays_per_trip * total_trips - trays_first_table = 16 := by
  sorry

end NUMINAMATH_CALUDE_trays_from_second_table_l3682_368262


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_roots_l3682_368265

theorem smallest_n_for_integer_roots : ∃ (x y : ℤ),
  x^2 - 91*x + 2014 = 0 ∧ y^2 - 91*y + 2014 = 0 ∧
  (∀ (n : ℕ) (a b : ℤ), n < 91 → (a^2 - n*a + 2014 = 0 ∧ b^2 - n*b + 2014 = 0) → False) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_roots_l3682_368265


namespace NUMINAMATH_CALUDE_triangle_toothpick_count_l3682_368208

/-- The number of small equilateral triangles in the base row -/
def base_triangles : ℕ := 10

/-- The number of additional toothpicks in the isosceles row compared to the last equilateral row -/
def extra_isosceles_toothpicks : ℕ := 9

/-- The total number of small equilateral triangles in the main part of the large triangle -/
def total_equilateral_triangles : ℕ := (base_triangles * (base_triangles + 1)) / 2

/-- The number of toothpicks needed for the described triangle construction -/
def total_toothpicks : ℕ := 
  let equilateral_toothpicks := (3 * total_equilateral_triangles + 1) / 2
  let boundary_toothpicks := 2 * base_triangles + extra_isosceles_toothpicks
  equilateral_toothpicks + extra_isosceles_toothpicks + boundary_toothpicks - base_triangles

theorem triangle_toothpick_count : total_toothpicks = 110 := by sorry

end NUMINAMATH_CALUDE_triangle_toothpick_count_l3682_368208


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_eighty_l3682_368241

theorem thirty_percent_less_than_eighty (x : ℚ) : x + x/2 = 80 - 80*3/10 → x = 112/3 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_eighty_l3682_368241


namespace NUMINAMATH_CALUDE_area_of_PQRSUV_l3682_368282

-- Define the polygon and its components
structure Polygon where
  PQ : ℝ
  QR : ℝ
  UV : ℝ
  SU : ℝ
  TU : ℝ
  RS : ℝ

-- Define the conditions
def polygon_conditions (p : Polygon) : Prop :=
  p.PQ = 8 ∧
  p.QR = 10 ∧
  p.UV = 6 ∧
  p.SU = 3 ∧
  p.PQ = p.TU + p.UV ∧
  p.QR = p.RS + p.SU

-- Define the area calculation
def area_PQRSUV (p : Polygon) : ℝ :=
  p.PQ * p.QR - p.SU * p.UV

-- Theorem statement
theorem area_of_PQRSUV (p : Polygon) (h : polygon_conditions p) :
  area_PQRSUV p = 62 := by
  sorry

end NUMINAMATH_CALUDE_area_of_PQRSUV_l3682_368282


namespace NUMINAMATH_CALUDE_remaining_quarters_count_l3682_368279

def initial_amount : ℚ := 40
def pizza_cost : ℚ := 2.75
def soda_cost : ℚ := 1.5
def jeans_cost : ℚ := 11.5

def remaining_money : ℚ := initial_amount - (pizza_cost + soda_cost + jeans_cost)

def quarters_in_dollar : ℕ := 4

theorem remaining_quarters_count : 
  (remaining_money * quarters_in_dollar).floor = 97 := by sorry

end NUMINAMATH_CALUDE_remaining_quarters_count_l3682_368279


namespace NUMINAMATH_CALUDE_y_intercept_approx_20_l3682_368293

/-- A straight line in the xy-plane with given slope and point -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line -/
def y_intercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- Theorem: The y-intercept of the given line is approximately 20 -/
theorem y_intercept_approx_20 (l : Line) 
  (h1 : l.slope = 3.8666666666666667)
  (h2 : l.point = (150, 600)) :
  ∃ ε > 0, |y_intercept l - 20| < ε :=
sorry

end NUMINAMATH_CALUDE_y_intercept_approx_20_l3682_368293


namespace NUMINAMATH_CALUDE_problem_statement_l3682_368275

-- Define propositions p and q as functions of m
def p (m : ℝ) : Prop := m > 2
def q (m : ℝ) : Prop := m > 1

-- State the theorem
theorem problem_statement (m : ℝ) :
  ((p m ∨ q m) ∧ ¬(p m ∧ q m)) →
  (p m ∧ ¬(q m)) ∧ (1 < m ∧ m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3682_368275


namespace NUMINAMATH_CALUDE_college_students_count_l3682_368296

theorem college_students_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 175) : boys + girls = 455 := by
  sorry

end NUMINAMATH_CALUDE_college_students_count_l3682_368296


namespace NUMINAMATH_CALUDE_simplify_expression_l3682_368201

theorem simplify_expression (x : ℝ) : 3*x + 4*x - 2*x + 6*x - 3*x = 8*x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3682_368201


namespace NUMINAMATH_CALUDE_square_fraction_count_l3682_368291

theorem square_fraction_count : 
  ∃ (S : Finset ℤ), 
    (∀ n ∈ S, 0 ≤ n ∧ n ≤ 29 ∧ ∃ k : ℤ, n / (30 - n) = k^2) ∧ 
    (∀ n : ℤ, 0 ≤ n ∧ n ≤ 29 ∧ (∃ k : ℤ, n / (30 - n) = k^2) → n ∈ S) ∧
    Finset.card S = 3 :=
by sorry

end NUMINAMATH_CALUDE_square_fraction_count_l3682_368291


namespace NUMINAMATH_CALUDE_no_primes_in_list_l3682_368285

/-- Represents a number formed by repeating 57 a certain number of times -/
def repeatedNumber (repetitions : ℕ) : ℕ :=
  57 * ((10^(2*repetitions) - 1) / 99)

/-- The list of numbers formed by repeating 57 from 1 to n times -/
def numberList (n : ℕ) : List ℕ :=
  List.map repeatedNumber (List.range n)

/-- Counts the number of prime numbers in the list -/
def countPrimes (list : List ℕ) : ℕ :=
  (list.filter Nat.Prime).length

theorem no_primes_in_list (n : ℕ) : countPrimes (numberList n) = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_primes_in_list_l3682_368285


namespace NUMINAMATH_CALUDE_dog_food_consumption_l3682_368263

/-- The amount of dog food two dogs eat together daily -/
def total_food : ℝ := 0.25

/-- Given two dogs that eat the same amount of food daily, 
    prove that each dog eats half of the total food -/
theorem dog_food_consumption (dog1_food dog2_food : ℝ) 
  (h1 : dog1_food = dog2_food) 
  (h2 : dog1_food + dog2_food = total_food) : 
  dog1_food = 0.125 := by sorry

end NUMINAMATH_CALUDE_dog_food_consumption_l3682_368263


namespace NUMINAMATH_CALUDE_message_sending_methods_l3682_368299

/-- The number of friends the student has -/
def num_friends : ℕ := 4

/-- The number of suitable messages in the draft box -/
def num_messages : ℕ := 3

/-- The number of different methods to send messages -/
def num_methods : ℕ := num_messages ^ num_friends

/-- Theorem stating that the number of different methods to send messages is 81 -/
theorem message_sending_methods : num_methods = 81 := by
  sorry

end NUMINAMATH_CALUDE_message_sending_methods_l3682_368299


namespace NUMINAMATH_CALUDE_dice_sum_theorem_l3682_368256

def Die := Fin 6

def roll_sum (d1 d2 : Die) : ℕ := d1.val + d2.val + 2

def possible_sums : Set ℕ := {n | ∃ (d1 d2 : Die), roll_sum d1 d2 = n}

theorem dice_sum_theorem : possible_sums = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} := by
  sorry

end NUMINAMATH_CALUDE_dice_sum_theorem_l3682_368256


namespace NUMINAMATH_CALUDE_light_ray_distance_l3682_368234

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/9 + y^2/5 = 1

-- Define the foci of the ellipse
def left_focus : ℝ × ℝ := sorry
def right_focus : ℝ × ℝ := sorry

-- Define the total distance traveled by the light ray
def total_distance (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem light_ray_distance :
  ∀ p q : ℝ × ℝ,
  ellipse p.1 p.2 →
  ellipse q.1 q.2 →
  total_distance left_focus p + total_distance p right_focus +
  total_distance right_focus q + total_distance q left_focus = 12 :=
sorry

end NUMINAMATH_CALUDE_light_ray_distance_l3682_368234


namespace NUMINAMATH_CALUDE_quadratic_shift_l3682_368218

/-- The original quadratic function -/
def f (b : ℝ) (x : ℝ) : ℝ := 2 * x^2 - b * x + 3

/-- The mistakenly drawn function -/
def g (b : ℝ) (x : ℝ) : ℝ := 2 * x^2 + b * x + 3

/-- The shifted original function -/
def f_shifted (b : ℝ) (x : ℝ) : ℝ := f b (x + 6)

theorem quadratic_shift (b : ℝ) : 
  (∀ x, g b x = f_shifted b x) → b = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_shift_l3682_368218


namespace NUMINAMATH_CALUDE_parking_lot_theorem_l3682_368269

/-- Represents a parking lot configuration --/
structure ParkingLot where
  grid : Fin 7 → Fin 7 → Bool
  gate : Fin 7 × Fin 7

/-- Checks if a car can exit from its position --/
def canExit (lot : ParkingLot) (pos : Fin 7 × Fin 7) : Bool :=
  sorry

/-- Counts the number of cars in the parking lot --/
def carCount (lot : ParkingLot) : Nat :=
  sorry

/-- Checks if all cars in the lot can exit --/
def allCarsCanExit (lot : ParkingLot) : Bool :=
  sorry

/-- The maximum number of cars that can be parked --/
def maxCars : Nat := 28

theorem parking_lot_theorem (lot : ParkingLot) :
  (allCarsCanExit lot) → (carCount lot ≤ maxCars) :=
  sorry

end NUMINAMATH_CALUDE_parking_lot_theorem_l3682_368269


namespace NUMINAMATH_CALUDE_fabric_cost_difference_l3682_368245

/-- The amount of fabric Kenneth bought in ounces -/
def kenneth_fabric : ℝ := 700

/-- The price per ounce of fabric in dollars -/
def price_per_oz : ℝ := 40

/-- The amount of fabric Nicholas bought in ounces -/
def nicholas_fabric : ℝ := 6 * kenneth_fabric

/-- The total cost of Kenneth's fabric in dollars -/
def kenneth_cost : ℝ := kenneth_fabric * price_per_oz

/-- The total cost of Nicholas's fabric in dollars -/
def nicholas_cost : ℝ := nicholas_fabric * price_per_oz

/-- The difference in cost between Nicholas's and Kenneth's fabric purchases -/
theorem fabric_cost_difference : nicholas_cost - kenneth_cost = 140000 := by
  sorry

end NUMINAMATH_CALUDE_fabric_cost_difference_l3682_368245


namespace NUMINAMATH_CALUDE_cube_distance_to_plane_l3682_368258

theorem cube_distance_to_plane (cube_side : ℝ) (h1 h2 h3 : ℝ) :
  cube_side = 10 →
  h1 = 10 ∧ h2 = 11 ∧ h3 = 12 →
  ∃ (d : ℝ), d = (33 - Real.sqrt 294) / 3 ∧
    d = min h1 (min h2 h3) - (cube_side - Real.sqrt (cube_side^2 + (h2 - h1)^2 + (h3 - h1)^2)) :=
by sorry

end NUMINAMATH_CALUDE_cube_distance_to_plane_l3682_368258


namespace NUMINAMATH_CALUDE_no_real_solutions_l3682_368220

theorem no_real_solutions :
  ∀ y : ℝ, (8 * y^2 + 155 * y + 3) / (4 * y + 45) ≠ 4 * y + 3 :=
by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3682_368220


namespace NUMINAMATH_CALUDE_correct_scientific_statement_only_mathematical_models_correct_l3682_368292

-- Define the type for scientific statements
inductive ScientificStatement
  | PopulationDensityEstimation
  | PreliminaryExperiment
  | MathematicalModels
  | SpeciesRichness

-- Define a function to check if a statement is correct
def isCorrectStatement (s : ScientificStatement) : Prop :=
  match s with
  | .MathematicalModels => True
  | _ => False

-- Theorem to prove
theorem correct_scientific_statement :
  ∃ (s : ScientificStatement), isCorrectStatement s :=
  sorry

-- Additional theorem to show that only MathematicalModels is correct
theorem only_mathematical_models_correct (s : ScientificStatement) :
  isCorrectStatement s ↔ s = ScientificStatement.MathematicalModels :=
  sorry

end NUMINAMATH_CALUDE_correct_scientific_statement_only_mathematical_models_correct_l3682_368292


namespace NUMINAMATH_CALUDE_tee_price_calculation_l3682_368235

/-- The price of a single tee shirt in Linda's store -/
def tee_price : ℝ := 8

/-- The price of a single pair of jeans in Linda's store -/
def jeans_price : ℝ := 11

/-- The number of tee shirts sold in a day -/
def tees_sold : ℕ := 7

/-- The number of jeans sold in a day -/
def jeans_sold : ℕ := 4

/-- The total revenue for the day -/
def total_revenue : ℝ := 100

theorem tee_price_calculation :
  tee_price * tees_sold + jeans_price * jeans_sold = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_tee_price_calculation_l3682_368235


namespace NUMINAMATH_CALUDE_solution_values_l3682_368260

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def is_valid_p (a b p : E) : Prop :=
  ‖p - b‖ = 3 * ‖p - a‖

def fixed_distance (a b p : E) (t u : ℝ) : Prop :=
  ∃ (k : ℝ), ‖p - (t • a + u • b)‖ = k

theorem solution_values (a b : E) :
  ∃ (p : E), is_valid_p a b p ∧
  fixed_distance a b p (9/8) (-1/8) :=
sorry

end NUMINAMATH_CALUDE_solution_values_l3682_368260


namespace NUMINAMATH_CALUDE_absolute_opposite_reciprocal_of_negative_three_halves_l3682_368274

theorem absolute_opposite_reciprocal_of_negative_three_halves :
  let x : ℚ := -3/2
  (abs x = 3/2) ∧ (-x = 3/2) ∧ (x⁻¹ = -2/3) := by
  sorry

end NUMINAMATH_CALUDE_absolute_opposite_reciprocal_of_negative_three_halves_l3682_368274


namespace NUMINAMATH_CALUDE_transportation_time_savings_l3682_368236

def walking_time : ℕ := 98
def bicycle_saved_time : ℕ := 64
def car_saved_time : ℕ := 85
def bus_saved_time : ℕ := 55

theorem transportation_time_savings :
  (walking_time - (walking_time - bicycle_saved_time) = bicycle_saved_time) ∧
  (walking_time - (walking_time - car_saved_time) = car_saved_time) ∧
  (walking_time - (walking_time - bus_saved_time) = bus_saved_time) := by
  sorry

end NUMINAMATH_CALUDE_transportation_time_savings_l3682_368236


namespace NUMINAMATH_CALUDE_local_minimum_of_f_l3682_368214

/-- The function f(x) defined as (x^2 + ax - 1)e^(x-1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + a*x - 1) * Real.exp (x - 1)

/-- The derivative of f(x) with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := (x^2 + x - 2) * Real.exp (x - 1)

theorem local_minimum_of_f (a : ℝ) :
  (f_deriv a (-2) = 0) →  -- x = -2 is a point of extremum
  (∃ (x : ℝ), x > -2 ∧ x < 1 ∧ ∀ y, y > -2 ∧ y < 1 → f a x ≤ f a y) ∧ -- local minimum exists
  (f a 1 = -1) -- the local minimum value is -1
  := by sorry

end NUMINAMATH_CALUDE_local_minimum_of_f_l3682_368214


namespace NUMINAMATH_CALUDE_exist_polynomials_with_degree_properties_l3682_368276

open Polynomial

/-- Prove the existence of polynomials with specific degree properties -/
theorem exist_polynomials_with_degree_properties :
  ∃ (P Q R : Polynomial ℝ),
    (P ≠ 0) ∧ (Q ≠ 0) ∧ (R ≠ 0) ∧
    (degree (P * Q) = degree (Q * R)) ∧
    (degree (Q * R) = degree (P * R)) ∧
    (degree (P + Q) ≠ degree (P + R)) ∧
    (degree (P + R) ≠ degree (Q + R)) ∧
    (degree (Q + R) ≠ degree (P + Q)) :=
by sorry

end NUMINAMATH_CALUDE_exist_polynomials_with_degree_properties_l3682_368276


namespace NUMINAMATH_CALUDE_greatest_m_for_ratio_bound_l3682_368205

/-- Definition of a(m,n) -/
def a (m n : ℕ) : ℕ := (2^m - 1)^(2*n)

/-- Definition of b(m,n) -/
def b (m n : ℕ) : ℕ := (3^m - 2^(m+1) + 1)^n

/-- The main theorem -/
theorem greatest_m_for_ratio_bound :
  (∃ n : ℕ+, (a 26 n : ℚ) / (b 26 n) ≤ 2021) ∧
  (∀ m > 26, ∀ n : ℕ+, (a m n : ℚ) / (b m n) > 2021) :=
sorry

end NUMINAMATH_CALUDE_greatest_m_for_ratio_bound_l3682_368205


namespace NUMINAMATH_CALUDE_oblique_triangular_prism_volume_l3682_368229

/-- The volume of an oblique triangular prism with specific properties -/
theorem oblique_triangular_prism_volume (a : ℝ) (ha : a > 0) :
  let base_area := (a^2 * Real.sqrt 3) / 4
  let height := a * Real.sqrt 3 / 2
  base_area * height = (3 * a^3) / 8 := by sorry

end NUMINAMATH_CALUDE_oblique_triangular_prism_volume_l3682_368229


namespace NUMINAMATH_CALUDE_arithmetic_sequence_log_implies_square_product_square_product_not_sufficient_for_arithmetic_sequence_log_l3682_368253

def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  2 * b = a + c

theorem arithmetic_sequence_log_implies_square_product
  (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : is_arithmetic_sequence (Real.log x) (Real.log y) (Real.log z)) :
  y^2 = x * z :=
sorry

theorem square_product_not_sufficient_for_arithmetic_sequence_log :
  ∃ x y z : ℝ, y^2 = x * z ∧ ¬is_arithmetic_sequence (Real.log x) (Real.log y) (Real.log z) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_log_implies_square_product_square_product_not_sufficient_for_arithmetic_sequence_log_l3682_368253


namespace NUMINAMATH_CALUDE_trig_identity_l3682_368238

theorem trig_identity (α : ℝ) : 
  1 - Real.cos (2 * α - π) + Real.cos (4 * α - 2 * π) = 
  4 * Real.cos (2 * α) * Real.cos (π / 6 + α) * Real.cos (π / 6 - α) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3682_368238


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l3682_368286

theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 40)
  (area2 : w * h = 15)
  (area3 : l * h = 12) :
  l * w * h = 60 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l3682_368286
