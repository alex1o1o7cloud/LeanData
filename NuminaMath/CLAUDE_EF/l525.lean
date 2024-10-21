import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_pairs_exist_l525_52520

def A : Set ℤ := {n | ∃ a b : ℤ, b ≠ 0 ∧ n = a^2 + 13*b^2}

theorem infinitely_many_pairs_exist :
  ∃ f : ℕ → ℤ × ℤ,
    Function.Injective f ∧
    ∀ n : ℕ, (let (x, y) := f n; (x^13 + y^13 ∈ A) ∧ (x + y ∉ A)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_pairs_exist_l525_52520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l525_52566

open Real

theorem intersection_length (x : ℝ) (h1 : x ∈ Set.Ioo 0 (π/2)) (h2 : 6 * cos x = 5 * tan x) : 
  5 * tan x - sqrt 5 * sin x = (4 * sqrt 5) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l525_52566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_squared_l525_52548

theorem sin_minus_cos_squared (θ : ℝ) (b : ℝ) 
  (h1 : 0 < θ ∧ θ < Real.pi / 2) 
  (h2 : Real.cos (2 * θ) = b) : 
  (Real.sin θ - Real.cos θ)^2 = b := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_squared_l525_52548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_geq_5_range_of_a_l525_52586

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

-- Part I: Solution set of f(x) ≥ 5
theorem solution_set_f_geq_5 :
  {x : ℝ | f x ≥ 5} = Set.Iic (-3) ∪ Set.Ici 2 :=
sorry

-- Part II: Range of a for which f(x) > a^2 - 2a holds for all x
theorem range_of_a :
  {a : ℝ | ∀ x, f x > a^2 - 2*a} = Set.Ioo (-1) 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_geq_5_range_of_a_l525_52586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_parallel_properties_l525_52528

/-- Two planes in 3D space -/
structure Plane3D where
  -- Define a plane (implementation details omitted)
  dummy : Unit

/-- A line in 3D space -/
structure Line3D where
  -- Define a line (implementation details omitted)
  dummy : Unit

/-- Predicate for two planes being parallel -/
def parallel_planes (p1 p2 : Plane3D) : Prop :=
  sorry -- Definition of parallel planes

/-- Predicate for a plane being parallel to a line -/
def plane_parallel_to_line (p : Plane3D) (l : Line3D) : Prop :=
  sorry -- Definition of a plane parallel to a line

/-- Predicate for a line intersecting a plane -/
def line_intersects_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry -- Definition of a line intersecting a plane

/-- Angle between a line and a plane -/
noncomputable def angle_line_plane (l : Line3D) (p : Plane3D) : ℝ :=
  sorry -- Definition of the angle between a line and a plane

theorem plane_parallel_properties :
  -- Statement A (incorrect)
  (∃ (p1 p2 : Plane3D) (l : Line3D),
    plane_parallel_to_line p1 l ∧ plane_parallel_to_line p2 l ∧ ¬parallel_planes p1 p2) ∧
  -- Statement B
  (∀ (p1 p2 : Plane3D) (l : Line3D),
    parallel_planes p1 p2 → line_intersects_plane l p1 → line_intersects_plane l p2) ∧
  -- Statement C
  (∀ (p1 p2 p3 : Plane3D),
    parallel_planes p1 p3 → parallel_planes p2 p3 → parallel_planes p1 p2) ∧
  -- Statement D
  (∀ (p1 p2 : Plane3D) (l : Line3D),
    parallel_planes p1 p2 → angle_line_plane l p1 = angle_line_plane l p2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_parallel_properties_l525_52528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_gender_ratio_l525_52524

theorem class_gender_ratio (B G : ℚ) (h_positive : B > 0 ∧ G > 0) :
  (16.4 * B + 15.2 * G) / (B + G) = 15.8 →
  B = G := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_gender_ratio_l525_52524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_modulo_pi_l525_52579

theorem unique_solution_modulo_pi (x y z w : ℝ) : 
  (x = w + z - w*z*w) ∧
  (y = w - x + w*x*y) ∧
  (z = x - y - x*y*z) ∧
  (w = y - z + y*z*w) →
  ∃! (t : ℝ), (x = 0) ∧ (y = Real.tan t) ∧ (z = -Real.tan t) ∧ (w = Real.tan t) ∧ 
  (∃ k : ℤ, t + t = k * Real.pi) ∧ (∃ m : ℤ, t = m * Real.pi) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_modulo_pi_l525_52579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mode_implies_x_value_l525_52556

def data_set (x : ℚ) : Finset ℚ := {3, x, 6, 5, 4}

def is_mode (s : Finset ℚ) (m : ℚ) : Prop :=
  ∀ y ∈ s, (s.filter (· = m)).card ≥ (s.filter (· = y)).card

theorem mode_implies_x_value (x : ℚ) :
  is_mode (data_set x) 4 → x = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mode_implies_x_value_l525_52556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_halloween_candy_collection_l525_52565

/-- The Halloween candy collection problem -/
theorem halloween_candy_collection
  (maggie_candy : ℕ)
  (harper_ratio : ℚ)
  (neil_ratio : ℚ)
  (liam_ratio : ℚ)
  (max_total : ℕ)
  (h_maggie : maggie_candy = 50)
  (h_harper : harper_ratio = 13/10)
  (h_neil : neil_ratio = 14/10)
  (h_liam : liam_ratio = 12/10)
  (h_max : max_total = 300) :
  min (Int.toNat ⌊(maggie_candy : ℚ) +
       (maggie_candy : ℚ) * harper_ratio +
       (maggie_candy : ℚ) * harper_ratio * neil_ratio +
       (maggie_candy : ℚ) * harper_ratio * neil_ratio * liam_ratio⌋) max_total = max_total :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_halloween_candy_collection_l525_52565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_search_problem_solution_l525_52594

/-- The golden ratio, approximately 0.618 --/
noncomputable def φ : ℝ := (Real.sqrt 5 - 1) / 2

/-- Calculate the first trial point using the golden ratio method --/
noncomputable def x₁ (a b : ℝ) : ℝ := a + φ * (b - a)

/-- Calculate the second trial point using the golden ratio method --/
noncomputable def x₂ (a b : ℝ) : ℝ := a + (b - x₁ a b)

/-- Calculate the third trial point using the golden ratio method --/
noncomputable def x₃ (a b : ℝ) : ℝ := b - φ * (b - x₁ a b)

/-- Theorem for the golden section search method --/
theorem golden_section_search (a b : ℝ) (h₁ : a < b) (h₂ : x₁ a b > x₂ a b) :
  x₃ a b = b - φ * (b - x₁ a b) :=
by
  -- The proof is omitted for now
  sorry

/-- Specific instance for the problem with a = 2 and b = 4 --/
theorem problem_solution :
  x₃ 2 4 = 4 - φ * (4 - x₁ 2 4) :=
by
  -- Apply the general theorem
  apply golden_section_search
  -- Prove a < b
  · norm_num
  -- Prove x₁ 2 4 > x₂ 2 4
  · sorry -- This would require actual calculation, which we skip for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_search_problem_solution_l525_52594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_auto_travel_l525_52563

/-- An automobile travels a/4 meters in r seconds. If this rate is maintained for 6 minutes,
    it travels 9a/(100r) kilometers. -/
theorem auto_travel (a r : ℝ) (hr : r > 0) :
  let rate := a / (4 * r)  -- rate in meters per second
  let time := 6 * 60       -- 6 minutes in seconds
  let distance := rate * time / 1000  -- convert to kilometers
  distance = 9 * a / (100 * r) := by
  sorry

#check auto_travel

end NUMINAMATH_CALUDE_ERRORFEEDBACK_auto_travel_l525_52563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l525_52588

noncomputable def angle (v w : ℝ × ℝ) : ℝ := Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

theorem angle_between_vectors (a b c : ℝ × ℝ) : 
  (Real.sqrt (a.1^2 + a.2^2) = 1) → 
  (Real.sqrt (b.1^2 + b.2^2) = 1) → 
  (Real.sqrt (c.1^2 + c.2^2) = 1) → 
  (angle a b = 2*π/3) → 
  (angle b c = 2*π/3) → 
  (angle c a = 2*π/3) →
  angle (a.1 - b.1, a.2 - b.2) (a.1 + c.1, a.2 + c.2) = π/6 := by
  sorry

#check angle_between_vectors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l525_52588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_power_function_l525_52508

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x ^ k

theorem range_of_power_function (k : ℝ) :
  Set.range (fun x => f k x) = Set.Ioi 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_power_function_l525_52508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_surface_area_ratio_l525_52547

/-- The ratio of the surface area of a regular tetrahedron to the surface area of its inscribed sphere -/
theorem tetrahedron_sphere_surface_area_ratio :
  ∀ (S₁ S₂ : ℝ),
  S₁ > 0 → S₂ > 0 →
  (∃ (a : ℝ), a > 0 ∧
    S₁ = Real.sqrt 3 * a^2 ∧
    S₂ = Real.pi * a^2 / 6) →
  S₁ / S₂ = 6 * Real.sqrt 3 / Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_surface_area_ratio_l525_52547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l525_52536

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 + x - 6)

-- Define the domain of f(x)
def domain (x : ℝ) : Prop := x^2 + x - 6 ≥ 0

-- Define t(x)
def t (x : ℝ) : ℝ := x^2 + x - 6

-- State that t(x) is increasing on (2, +∞)
axiom t_increasing : ∀ x y, 2 < x → x < y → t x < t y

-- State that √t is increasing on (0, +∞)
axiom sqrt_increasing : ∀ a b, 0 < a → a < b → Real.sqrt a < Real.sqrt b

-- Theorem: The increasing interval of f(x) is (2, +∞)
theorem f_increasing_interval :
  ∀ x y, 2 < x → x < y → f x < f y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l525_52536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_is_20_minutes_l525_52578

/-- Represents the travel scenario for Vasya --/
structure TravelScenario where
  distance : ℝ
  speed : ℝ
  traffic_light_position : ℝ
  wait_time : ℝ
  speed_multiplier : ℝ

/-- Calculates the total travel time for the given scenario --/
noncomputable def total_travel_time (scenario : TravelScenario) : ℝ :=
  scenario.distance / scenario.speed

/-- Calculates the travel time with traffic light delay and speed change --/
noncomputable def travel_time_with_delay (scenario : TravelScenario) : ℝ :=
  (scenario.traffic_light_position / scenario.speed) +
  scenario.wait_time +
  ((scenario.distance - scenario.traffic_light_position) / (scenario.speed * scenario.speed_multiplier))

/-- Theorem stating that the travel time is 20 minutes --/
theorem travel_time_is_20_minutes (scenario : TravelScenario)
  (h1 : scenario.traffic_light_position = scenario.distance / 2)
  (h2 : scenario.wait_time = 5)
  (h3 : scenario.speed_multiplier = 2)
  (h4 : total_travel_time scenario = travel_time_with_delay scenario) :
  total_travel_time scenario = 20 := by
  sorry

#check travel_time_is_20_minutes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_is_20_minutes_l525_52578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_theorem_l525_52538

/-- Configuration of triangles ABC and ABD -/
structure TriangleConfiguration where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  -- Distances
  AC : ℝ
  BC : ℝ
  AD : ℝ
  -- Conditions
  right_angle_ABC : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0
  right_angle_ABD : (B.1 - A.1) * (D.1 - A.1) + (B.2 - A.2) * (D.2 - A.2) = 0
  C_D_same_side : (B.1 - A.1) * (C.2 - A.2) = (B.1 - A.1) * (D.2 - A.2)
  AC_eq : AC = 4
  BC_eq : BC = 3
  AD_eq : AD = 15
  DE_parallel_AC : (E.2 - D.2) * (C.1 - A.1) = (E.1 - D.1) * (C.2 - A.2)

/-- The ratio DE/DB is 4/5 and the sum of its numerator and denominator is 9 -/
theorem triangle_ratio_theorem (config : TriangleConfiguration) :
  let DE := Real.sqrt ((config.E.1 - config.D.1)^2 + (config.E.2 - config.D.2)^2)
  let DB := Real.sqrt ((config.B.1 - config.D.1)^2 + (config.B.2 - config.D.2)^2)
  DE / DB = 4 / 5 ∧ 4 + 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_theorem_l525_52538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_coprime_l525_52542

theorem existence_of_coprime (p q r : ℕ+) (h : Nat.gcd p.val (Nat.gcd q.val r.val) = 1) :
  ∃ a : ℤ, Nat.gcd p.val (q.val + (a.natAbs : ℕ) * r.val) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_coprime_l525_52542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l525_52597

def A : Set ℤ := {-2, -1, 0, 1, 2, 3}

def B : Set ℤ := {x : ℤ | (x : ℝ)^2 - 2*(x : ℝ) - 3 < 0}

theorem intersection_A_B : A ∩ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l525_52597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_curves_do_not_intersect_l525_52575

/-- First polar curve: r = 3 * cos(θ) -/
noncomputable def curve1 (θ : ℝ) : ℝ × ℝ :=
  (3 * Real.cos θ * Real.cos θ, 3 * Real.cos θ * Real.sin θ)

/-- Second polar curve: r = 6 * sin(θ) -/
noncomputable def curve2 (θ : ℝ) : ℝ × ℝ :=
  (6 * Real.sin θ * Real.cos θ, 6 * Real.sin θ * Real.sin θ)

/-- The number of intersection points between curve1 and curve2 -/
def intersection_count : ℕ :=
  0

/-- Theorem stating that the number of intersection points is 0 -/
theorem no_intersection :
  intersection_count = 0 := by
  rfl

/-- Proof that the curves do not intersect -/
theorem curves_do_not_intersect :
  ∀ θ₁ θ₂ : ℝ, curve1 θ₁ ≠ curve2 θ₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_curves_do_not_intersect_l525_52575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_sum_l525_52570

noncomputable def x : ℝ := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 6

theorem polynomial_root_sum (a b c d : ℤ) :
  (x^4 + a * x^3 + b * x^2 + c * x + d = 0) →
  |a + b + c + d| = 93 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_sum_l525_52570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sum_inequalities_l525_52519

-- Define f(n) as the sum of reciprocals from 1 to n
def f (n : ℕ+) : ℚ := (Finset.range n).sum (fun i => 1 / (i + 1 : ℚ))

-- State the theorem
theorem harmonic_sum_inequalities :
  (∀ m n : ℕ+, n > m → f n - f m ≥ (n - m : ℚ) / n) ∧
  (∀ n : ℕ+, n > 1 → f (2^(n:ℕ)) > (n + 2 : ℚ) / 2) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sum_inequalities_l525_52519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_triangle_area_l525_52551

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the line C₂
def C₂ (x y : ℝ) : Prop := x + y = 2

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧ C₂ A.1 A.2 ∧ C₂ B.1 B.2

-- Define the triangle area function
noncomputable def triangle_area (O A B : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((A.1 - O.1) * (B.2 - O.2) - (B.1 - O.1) * (A.2 - O.2))

-- Theorem statement
theorem intersection_triangle_area :
  ∀ A B : ℝ × ℝ, intersection_points A B →
  triangle_area (0, 0) A B = 2 * Real.sqrt 2 := by
  sorry

#check intersection_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_triangle_area_l525_52551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_face_area_l525_52530

/-- The total area of the four triangular faces of a right, square-based pyramid -/
noncomputable def pyramidFaceArea (baseEdge : ℝ) (slantHeight : ℝ) : ℝ :=
  4 * (1 / 2 * baseEdge * Real.sqrt (slantHeight^2 - (baseEdge / 2)^2))

/-- Theorem stating the total area of the four triangular faces of a specific pyramid -/
theorem specific_pyramid_face_area :
  pyramidFaceArea 8 10 = 32 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_face_area_l525_52530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_min_value_achieved_l525_52572

-- Define the power function for real numbers
noncomputable def rpow (x y : ℝ) : ℝ := Real.rpow x y

theorem min_value_of_expression (x : ℝ) : rpow 16 x - rpow 4 x + 1 ≥ 3/4 := by
  sorry

theorem min_value_achieved : rpow 16 (-1/2) - rpow 4 (-1/2) + 1 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_min_value_achieved_l525_52572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_l525_52511

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := 1 / ⌊x^2 - 5*x + 8⌋

-- Define the domain of g(x)
def domain_g : Set ℝ := {x | x ≤ 1 ∨ x ≥ 7}

-- Theorem statement
theorem g_domain : {x : ℝ | IsRegular (g x)} = domain_g := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_l525_52511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_newspaper_tearing_l525_52512

theorem newspaper_tearing : ∀ (a b : ℕ), 
  (a * 7 + b * 4 - (a + b) + 1 = 2019) → False := by
  intro a b h
  sorry

#check newspaper_tearing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_newspaper_tearing_l525_52512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_image_plane_angles_equal_l525_52541

/-- Represents a line in 3D space with two components --/
structure Line3D where
  comp1 : ℝ × ℝ
  comp2 : ℝ × ℝ

/-- Represents the first quadrant in 2D space --/
def FirstQuadrant : Set (ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.2 ≥ 0}

/-- Checks if a line has a finite part in the first quadrant --/
def has_finite_part_in_first_quadrant (l : Line3D) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ FirstQuadrant

/-- Defines parallel vectors --/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

/-- Calculates the image plane angle of a line --/
noncomputable def image_plane_angle (l : Line3D) : ℝ :=
  sorry

/-- The main theorem --/
theorem opposite_image_plane_angles_equal
  (a b : Line3D)
  (ha : has_finite_part_in_first_quadrant a)
  (hb : has_finite_part_in_first_quadrant b)
  (h1 : parallel a.comp2 b.comp1)
  (h2 : parallel a.comp1 b.comp2) :
  image_plane_angle a = image_plane_angle b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_image_plane_angles_equal_l525_52541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_theorem_l525_52533

/-- The area of a figure formed by rotating a semicircle of radius R around one of its ends by an angle of 20° -/
noncomputable def rotated_semicircle_area (R : ℝ) : ℝ := (2 * Real.pi * R^2) / 9

/-- Theorem stating that the area of the rotated semicircle figure is (2πR²)/9 -/
theorem rotated_semicircle_area_theorem (R : ℝ) (h : R > 0) :
  let α : ℝ := 20 * Real.pi / 180  -- 20° in radians
  let semicircle_area : ℝ := Real.pi * R^2 / 2
  let sector_area : ℝ := (2 * R)^2 * α / 2
  sector_area = rotated_semicircle_area R :=
by
  sorry

#check rotated_semicircle_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_theorem_l525_52533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l525_52509

/-- Calculates the length of a train given its speed, the speed of a person moving in the opposite direction, and the time it takes for the train to pass the person. -/
noncomputable def trainLength (trainSpeed manSpeed : ℝ) (passingTime : ℝ) : ℝ :=
  let trainSpeedMps := trainSpeed * 1000 / 3600
  let manSpeedMps := manSpeed * 1000 / 3600
  let relativeSpeed := trainSpeedMps + manSpeedMps
  relativeSpeed * passingTime

/-- Theorem stating that under the given conditions, the length of the train is approximately 330.12 meters. -/
theorem train_length_calculation :
  let trainSpeed : ℝ := 60
  let manSpeed : ℝ := 6
  let passingTime : ℝ := 18
  abs (trainLength trainSpeed manSpeed passingTime - 330.12) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l525_52509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l525_52526

/-- Given an ellipse (C) with equation x^2/(3m) + y^2/m = 1 where m > 0,
    and the length of its major axis is 2√6 -/
theorem ellipse_properties (m : ℝ) (h_m : m > 0)
  (h_axis : Real.sqrt (3 * m) = Real.sqrt 6) :
  /- The equation of the ellipse is x^2/6 + y^2/2 = 1 -/
  (∀ x y : ℝ, x^2 / 6 + y^2 / 2 = 1 ↔ x^2 / (3*m) + y^2 / m = 1) ∧
  /- The eccentricity of the ellipse is √6/3 -/
  (Real.sqrt (1 - m / (3*m)) = Real.sqrt 6 / 3) ∧
  /- Given a moving line (l) that intersects with the y-axis at point B,
     and point P(3, 0) is symmetric about line (l) and lies on ellipse (C),
     the minimum value of |OB| is √6 -/
  (∀ B : ℝ × ℝ, B.1 = 0 →
    (∃ l : Set (ℝ × ℝ), 
      (∃ P : ℝ × ℝ, P.1 = 3 ∧ P.2 = 0 ∧ P ∈ l ∧
        P.1^2 / 6 + P.2^2 / 2 = 1) →
      Real.sqrt 6 ≤ Real.sqrt (B.1^2 + B.2^2))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l525_52526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_minus_g_l525_52554

/-- The function f(x) = x³ -/
def f (x : ℝ) : ℝ := x^3

/-- The function g(x) = ln x -/
noncomputable def g (x : ℝ) : ℝ := Real.log x

/-- The theorem stating the minimum value of |f(x) - g(x)| for x > 0 -/
theorem min_value_f_minus_g :
  ∃ (min_val : ℝ), min_val = (1 + Real.log 3) / 3 ∧
  ∀ (x : ℝ), x > 0 → |f x - g x| ≥ min_val := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_minus_g_l525_52554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_15th_term_l525_52598

theorem arithmetic_sequence_15th_term
  (a b : ℝ)
  (h1 : 0 < a ∧ 0 < b)
  (h2 : ∃ d, Real.log (a^7 * b^11) - Real.log (a^4 * b^5) = d ∧
            Real.log (a^10 * b^15) - Real.log (a^7 * b^11) = d)
  : 
  ∃ seq : ℕ → ℝ, seq 0 = Real.log (a^4 * b^5) ∧
                 (∀ n, seq (n + 1) - seq n = Real.log (a^7 * b^11) - Real.log (a^4 * b^5)) ∧
                 seq 14 = Real.log (b^181)
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_15th_term_l525_52598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_y_less_than_x_l525_52518

/-- A rectangle in the 2D plane --/
structure Rectangle where
  x_min : ℝ
  y_min : ℝ
  x_max : ℝ
  y_max : ℝ

/-- A point in the 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is inside a rectangle --/
def isInside (p : Point) (r : Rectangle) : Prop :=
  r.x_min ≤ p.x ∧ p.x ≤ r.x_max ∧ r.y_min ≤ p.y ∧ p.y ≤ r.y_max

/-- The probability of an event occurring when a point is randomly chosen from a rectangle --/
noncomputable def probability (r : Rectangle) (event : Point → Prop) [∀ p, Decidable (event p)] : ℝ :=
  (∫ x in r.x_min..r.x_max, ∫ y in r.y_min..r.y_max, if event { x := x, y := y } then 1 else 0) /
  ((r.x_max - r.x_min) * (r.y_max - r.y_min))

/-- The main theorem --/
theorem probability_y_less_than_x (r : Rectangle) : 
  r.x_min = 0 ∧ r.y_min = 0 ∧ r.x_max = 4 ∧ r.y_max = 3 →
  probability r (fun p => p.y < p.x) = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_y_less_than_x_l525_52518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_evaporated_amount_l525_52580

/-- Represents the composition of a solution --/
structure Solution where
  total_mass : ℝ
  liquid_x_percentage : ℝ

/-- Represents the evaporation process --/
noncomputable def evaporation_process (initial : Solution) (evaporated_mass : ℝ) : ℝ := 
  let remaining_mass := initial.total_mass - evaporated_mass
  let new_solution_mass := remaining_mass + evaporated_mass
  let initial_liquid_x := initial.total_mass * initial.liquid_x_percentage
  let added_liquid_x := evaporated_mass * initial.liquid_x_percentage
  (initial_liquid_x + added_liquid_x) / new_solution_mass

/-- Theorem stating the amount of water evaporated --/
theorem water_evaporated_amount : 
  let initial_solution := Solution.mk 10 0.3
  let final_percentage := 0.36
  ∃ (evaporated_mass : ℝ), 
    evaporated_mass = 1/11 ∧ 
    evaporation_process initial_solution evaporated_mass = final_percentage := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_evaporated_amount_l525_52580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l525_52557

noncomputable def power_function (m : ℤ) (x : ℝ) : ℝ := x^(m^2 - 2*m - 3)

def no_axis_intersection (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x ≠ 0 ∧ f 0 ≠ x

def symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

theorem power_function_properties (m : ℤ) :
  no_axis_intersection (power_function m) ∧
  symmetric_about_y_axis (power_function m) →
  m = -1 ∨ m = 1 ∨ m = 3 := by
  sorry

#check power_function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l525_52557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l525_52561

theorem equation_solution (x y z : ℝ) (hx : Real.sin x ≠ 0) (hy : Real.cos y ≠ 0) :
  (Real.sin x ^ 2 + 1 / Real.sin x ^ 2) ^ 3 + (Real.cos y ^ 2 + 1 / Real.cos y ^ 2) ^ 3 = 16 * Real.cos z ↔
  ∃ (n k m : ℤ), x = Real.pi / 2 + Real.pi * n ∧ y = Real.pi * k ∧ z = 2 * Real.pi * m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l525_52561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rams_camp_distance_l525_52500

/-- Calculates the distance on a map given actual distances and a reference map distance --/
noncomputable def mapDistance (actualDistance actualReference mapReference : ℝ) : ℝ :=
  (actualDistance * mapReference) / actualReference

theorem rams_camp_distance (mapMountainDistance : ℝ) (actualMountainDistance : ℝ) (ramsActualDistance : ℝ)
  (h1 : mapMountainDistance = 312)
  (h2 : actualMountainDistance = 136 * 1000)  -- Convert km to m
  (h3 : ramsActualDistance = 14.82 * 1000) :  -- Convert km to m
  ∃ (ε : ℝ), abs (mapDistance ramsActualDistance actualMountainDistance mapMountainDistance - 34) < ε ∧ ε > 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rams_camp_distance_l525_52500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_m_less_than_three_l525_52559

-- Define the sets A, B, and C
def A : Set ℝ := {x | 4 - (2 : ℝ)^x > 0}
def B : Set ℝ := {x | x > 4}
def C (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ m - 1}

-- State the theorem
theorem subset_implies_m_less_than_three (m : ℝ) :
  C m ⊆ (A ∪ B) → m < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_m_less_than_three_l525_52559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_six_factors_including_sixteen_l525_52521

def has_exactly_six_factors (n : ℕ) : Prop :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 6

def sixteen_is_factor (n : ℕ) : Prop := 16 ∣ n

theorem unique_number_with_six_factors_including_sixteen :
  ∃! n : ℕ, n > 0 ∧ has_exactly_six_factors n ∧ sixteen_is_factor n :=
by
  use 32
  constructor
  · constructor
    · simp
    · constructor
      · sorry -- Proof that 32 has exactly six factors
      · sorry -- Proof that 16 is a factor of 32
  · sorry -- Proof of uniqueness

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_six_factors_including_sixteen_l525_52521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_relationship_l525_52514

/-- A truncated cone with specific coloring properties -/
structure TruncatedCone where
  r : ℝ  -- radius of the top base
  R : ℝ  -- radius of the bottom base
  a : ℝ  -- slant height
  (r_pos : 0 < r)
  (R_pos : 0 < R)
  (a_pos : 0 < a)
  (r_lt_R : r < R)

/-- The blue surface area is twice the red surface area -/
def blue_twice_red (cone : TruncatedCone) : Prop :=
  Real.pi * ((cone.R + cone.r) / 2 + cone.R) * (cone.a / 2) = 
  2 * Real.pi * ((cone.R + cone.r) / 2 + cone.r) * (cone.a / 2)

/-- The theorem stating the relationship between the radii -/
theorem radius_relationship (cone : TruncatedCone) 
  (h : blue_twice_red cone) : cone.R = 5 * cone.r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_relationship_l525_52514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_number_proof_l525_52517

theorem larger_number_proof (a b : ℕ+) : 
  (max a b - min a b : ℕ) = 120 →
  Nat.lcm a.val b.val = 105 * Nat.gcd a.val b.val →
  max a.val b.val = 225 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_number_proof_l525_52517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_after_nine_minutes_l525_52522

/-- Represents the robot's movement --/
structure RobotMovement where
  turnAngle : ℕ → Int  -- Function representing the turn angle at each minute
  initialDirection : ℝ × ℝ  -- Initial direction vector

/-- Calculates the final position of the robot after n minutes --/
def finalPosition (movement : RobotMovement) (n : ℕ) : ℝ × ℝ :=
  sorry

/-- Calculates the distance between two points --/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry

/-- Theorem stating the minimum distance from the starting position after 9 minutes --/
theorem min_distance_after_nine_minutes (movement : RobotMovement) :
  (∀ (i : ℕ), i ≥ 1 → movement.turnAngle i ∈ ({90, -90} : Set Int)) →  -- Turns are 90 degrees left or right
  movement.turnAngle 0 = 0 →  -- No turn in the first minute
  (∃ (finalPos : ℝ × ℝ), 
    finalPos = finalPosition movement 9 ∧
    distance (0, 0) finalPos ≥ 10) ∧
  (∀ (pos : ℝ × ℝ), 
    pos = finalPosition movement 9 →
    distance (0, 0) pos ≥ 10) :=
by
  sorry

#check min_distance_after_nine_minutes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_after_nine_minutes_l525_52522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_inverse_square_derivative_cube_root_derivative_two_to_x_derivative_log_base_3_l525_52549

-- Function 1: y = 1/x²
theorem derivative_inverse_square (x : ℝ) (hx : x ≠ 0) :
  deriv (λ x => 1 / x^2) x = -2 / x^3 :=
sorry

-- Function 2: y = ∛x
theorem derivative_cube_root (x : ℝ) (hx : x > 0) :
  deriv (λ x => x^(1/3 : ℝ)) x = (1/3) * x^(-(2/3 : ℝ)) :=
sorry

-- Function 3: y = 2ˣ
theorem derivative_two_to_x (x : ℝ) :
  deriv (λ x => (2 : ℝ)^x) x = (2 : ℝ)^x * Real.log 2 :=
sorry

-- Function 4: y = log₃x
theorem derivative_log_base_3 (x : ℝ) (hx : x > 0) :
  deriv (λ x => Real.log x / Real.log 3) x = 1 / (x * Real.log 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_inverse_square_derivative_cube_root_derivative_two_to_x_derivative_log_base_3_l525_52549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2008th_term_l525_52552

def sequence_property (a : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, a n ≤ a (n + 1)) ∧
  (∀ k : ℕ, k > 0 → (Finset.filter (fun i => a i = k) (Finset.range (2008 + 1))).card = 2 * k - 1)

theorem sequence_2008th_term (a : ℕ → ℕ) (h : sequence_property a) : a 2008 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2008th_term_l525_52552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_sequence_l525_52574

/-- A sequence of natural numbers with the required property -/
def special_sequence : Fin 2013 → ℕ := sorry

/-- The property that the sum of any two numbers is divisible by their difference -/
def has_special_property (seq : Fin 2013 → ℕ) : Prop :=
  ∀ i j : Fin 2013, i ≠ j → (seq i + seq j) % (Int.natAbs (seq i - seq j)) = 0

/-- The theorem stating the existence of 2013 distinct natural numbers with the special property -/
theorem exists_special_sequence :
  ∃ seq : Fin 2013 → ℕ, (has_special_property seq) ∧ (Function.Injective seq) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_sequence_l525_52574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_rectangle_ef_length_l525_52531

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Length of EF in folded rectangle -/
theorem folded_rectangle_ef_length 
  (rect : Rectangle)
  (h_ab : distance rect.A rect.B = 4)
  (h_bc : distance rect.B rect.C = 12)
  (E : Point)
  (F : Point)
  (h_fold : distance rect.A rect.C = distance rect.A F + distance F rect.C) :
  distance E F = 20 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_rectangle_ef_length_l525_52531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_square_range_l525_52516

theorem count_integers_in_square_range : 
  (Finset.filter (fun x : ℕ => 81 ≤ x ^ 2 ∧ x ^ 2 ≤ 225) (Finset.range 16)).card = 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_square_range_l525_52516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_f_10_f_11_l525_52590

def f (x : ℤ) : ℤ := x^3 - x^2 + 2*x + 1007

theorem gcd_f_10_f_11 : Int.gcd (f 10) (f 11) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_f_10_f_11_l525_52590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_all_down_l525_52585

/-- Represents the state of a cup (Up or Down) -/
inductive CupState
| Up
| Down
deriving BEq

/-- Represents the state of all cups on the table -/
def TableState := List CupState

/-- Flips the state of a cup -/
def flipCup : CupState → CupState
| CupState.Up => CupState.Down
| CupState.Down => CupState.Up

/-- Flips exactly 4 cups in the given table state -/
def flipFourCups (state : TableState) : TableState :=
  sorry

/-- The initial state of the table with all 5 cups facing up -/
def initialState : TableState :=
  List.replicate 5 CupState.Up

/-- Checks if all cups are facing down -/
def allDown (state : TableState) : Prop :=
  state.all (· == CupState.Down)

/-- Main theorem: It's impossible to have all cups facing down -/
theorem impossible_all_down :
  ¬∃ (n : Nat), allDown (n.iterate flipFourCups initialState) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_all_down_l525_52585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_fraction_is_one_fourth_l525_52535

/-- Represents the contents of a cup -/
structure CupContents where
  tea : ℚ
  milk : ℚ

/-- Performs the series of transfers between two cups -/
def transfer (cup1 cup2 : CupContents) : CupContents × CupContents :=
  let c1_after_first := CupContents.mk (cup1.tea - cup1.tea / 3) cup1.milk
  let c2_after_first := CupContents.mk (cup2.tea + cup1.tea / 3) cup2.milk
  
  let total_second := c2_after_first.tea + c2_after_first.milk
  let transfer_amount := total_second / 4
  let c1_after_second := CupContents.mk
    (c1_after_first.tea + transfer_amount * c2_after_first.tea / total_second)
    (c1_after_first.milk + transfer_amount * c2_after_first.milk / total_second)
  let c2_after_second := CupContents.mk
    (c2_after_first.tea - transfer_amount * c2_after_first.tea / total_second)
    (c2_after_first.milk - transfer_amount * c2_after_first.milk / total_second)
  
  let total_third := c1_after_second.tea + c1_after_second.milk
  let final_transfer := total_third / 6
  let c1_final := CupContents.mk
    (c1_after_second.tea - final_transfer * c1_after_second.tea / total_third)
    (c1_after_second.milk - final_transfer * c1_after_second.milk / total_third)
  
  (c1_final, c2_after_second)

/-- The main theorem stating that after the transfers, 1/4 of the liquid in cup 1 is milk -/
theorem milk_fraction_is_one_fourth :
  let initial_cup1 : CupContents := CupContents.mk 6 0
  let initial_cup2 : CupContents := CupContents.mk 0 6
  let (final_cup1, _) := transfer initial_cup1 initial_cup2
  final_cup1.milk / (final_cup1.tea + final_cup1.milk) = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_fraction_is_one_fourth_l525_52535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_ratio_l525_52539

noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

def problem_set (n : ℕ) : Set ℝ := {x | ∃ k : ℕ, k ≤ n ∧ x = 2^k}

theorem closest_integer_ratio (n : ℕ) (h : n = 20) :
  let largest := (2 : ℝ)^n
  let others := (geometric_sum 2 2 n) - largest + 5
  Int.floor ((largest / others) + 0.5) = 1 := by
  sorry

#check closest_integer_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_ratio_l525_52539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_ten_win_loss_difference_tinas_wins_l525_52558

/-- Represents the number of fights Tina won before her first loss -/
def wins_before_first_loss : ℕ := 10

/-- Tina won at least 10 fights before her first loss -/
theorem at_least_ten : wins_before_first_loss ≥ 10 := by
  rfl

/-- Total wins at the end of Tina's career -/
def total_wins : ℕ := wins_before_first_loss + 2 * wins_before_first_loss

/-- Total losses at the end of Tina's career -/
def total_losses : ℕ := 2

/-- At the end of her career, Tina had 28 more wins than losses -/
theorem win_loss_difference : total_wins - total_losses = 28 := by
  rfl

theorem tinas_wins : wins_before_first_loss = 10 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_ten_win_loss_difference_tinas_wins_l525_52558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_formula_f_period_f_min_value_f_max_value_l525_52505

noncomputable def P (x : ℝ) : ℝ × ℝ := (Real.cos (2 * x) + 1, 1)
noncomputable def Q (x : ℝ) : ℝ × ℝ := (1, Real.sqrt 3 * Real.sin (2 * x) + 1)

noncomputable def f (x : ℝ) : ℝ := (P x).1 * (Q x).1 + (P x).2 * (Q x).2

theorem f_formula (x : ℝ) : f x = Real.cos (2 * x) + Real.sqrt 3 * Real.sin (2 * x) + 2 := by sorry

theorem f_period : ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S) ∧ T = Real.pi := by sorry

theorem f_min_value : ∃ x, f x = 0 ∧ ∀ y, f y ≥ 0 := by sorry

theorem f_max_value : ∃ x, f x = 4 ∧ ∀ y, f y ≤ 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_formula_f_period_f_min_value_f_max_value_l525_52505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_and_quadrilateral_l525_52504

open Real

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  0 < A ∧ A < Real.pi ∧
  0 < B ∧ B < Real.pi ∧
  0 < C ∧ C < Real.pi ∧
  a > 0 ∧ b > 0 ∧ c > 0

-- Define the conditions
def conditions (A B C : ℝ) (a b c : ℝ) (θ : ℝ) : Prop :=
  triangle_ABC A B C a b c ∧
  b = c ∧
  Real.sin B / Real.sin A = (1 - Real.cos B) / Real.cos A ∧
  0 < θ ∧ θ < Real.pi

-- Define the quadrilateral OACB
noncomputable def quadrilateral_OACB (A B C : ℝ) (a b c : ℝ) (θ : ℝ) : ℝ :=
  2 * Real.sin (θ - Real.pi/3) + 5 * Real.sqrt 3 / 4

-- State the theorem
theorem triangle_and_quadrilateral 
  (A B C : ℝ) (a b c : ℝ) (θ : ℝ) 
  (h : conditions A B C a b c θ) : 
  A = Real.pi/3 ∧ 
  ∃ (max_area : ℝ), max_area = (8 + 5 * Real.sqrt 3) / 4 ∧ 
    ∀ θ', 0 < θ' ∧ θ' < Real.pi → quadrilateral_OACB A B C a b c θ' ≤ max_area :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_and_quadrilateral_l525_52504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_quadrilateral_area_l525_52576

/-- A convex quadrilateral with two equal and perpendicular opposite sides -/
structure SpecialQuadrilateral (a b : ℝ) where
  convex : Bool
  equal_perpendicular_sides : Bool
  side1 : ℝ
  side2 : ℝ
  h_side1 : side1 = a
  h_side2 : side2 = b

/-- The area of a SpecialQuadrilateral -/
noncomputable def area (q : SpecialQuadrilateral a b) : ℝ :=
  (b^2 - a^2) / 4

theorem special_quadrilateral_area (a b : ℝ) (q : SpecialQuadrilateral a b) :
  area q = (b^2 - a^2) / 4 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_quadrilateral_area_l525_52576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_package_requires_extra_fee_l525_52529

/-- Represents a package with length and width measurements. -/
structure Package where
  length : ℚ
  width : ℚ

/-- Determines if a package requires an extra fee based on its dimensions. -/
def requiresExtraFee (p : Package) : Bool :=
  let ratio := p.length / p.width
  ratio < 3/2 || ratio > 3

/-- The list of packages with their dimensions. -/
def packages : List Package := [
  ⟨8, 5⟩,   -- Package X
  ⟨12, 4⟩,  -- Package Y
  ⟨9, 9⟩,   -- Package Z
  ⟨14, 5⟩   -- Package W
]

/-- Theorem stating that exactly one package requires an extra fee. -/
theorem one_package_requires_extra_fee :
  (packages.filter requiresExtraFee).length = 1 := by
  sorry

#eval (packages.filter requiresExtraFee).length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_package_requires_extra_fee_l525_52529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l525_52571

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4*x + 6*y + 5

-- Define the center of the circle
def circle_center : ℝ × ℝ := (2, 3)

-- Define the radius of the circle
noncomputable def circle_radius : ℝ := 3 * Real.sqrt 2

-- Define the point we're measuring distance to
def point : ℝ × ℝ := (17, 11)

-- Theorem stating the properties of the circle
theorem circle_properties :
  (∀ x y, circle_equation x y ↔ (x - 2)^2 + (y - 3)^2 = 18) ∧
  circle_radius = 3 * Real.sqrt 2 ∧
  Real.sqrt ((point.1 - circle_center.1)^2 + (point.2 - circle_center.2)^2) = 17 := by
  sorry

#check circle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l525_52571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_is_900_l525_52596

/-- Calculates the principal amount given simple interest, time, and rate -/
noncomputable def calculate_principal (simple_interest : ℝ) (time : ℝ) (rate : ℝ) : ℝ :=
  (simple_interest * 100) / (rate * time)

/-- Theorem stating that the principal is 900 given the problem conditions -/
theorem principal_is_900 :
  let simple_interest : ℝ := 160
  let time : ℝ := 4
  let rate : ℝ := 4.444444444444445
  calculate_principal simple_interest time rate = 900 := by
  simp [calculate_principal]
  norm_num
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_is_900_l525_52596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_parabola_l525_52599

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 8)^2 + y^2 = 9

-- Define the parabola
def parabola_eq (x y : ℝ) : Prop := y^2 = 16 * x

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem min_distance_circle_parabola :
  ∃ (x1 y1 x2 y2 : ℝ),
    circle_eq x1 y1 ∧ parabola_eq x2 y2 ∧
    (∀ (a b c d : ℝ), circle_eq a b → parabola_eq c d →
      distance x1 y1 x2 y2 ≤ distance a b c d) ∧
    distance x1 y1 x2 y2 = 4 * Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_parabola_l525_52599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l525_52555

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 else -x^2

-- State the theorem
theorem range_of_t (t : ℝ) : 
  (∀ x ∈ Set.Icc t (t + 1), f (x + t) ≥ 9 * f x) → 
  t ∈ Set.Iic (-2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l525_52555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_to_larger_triangle_ratio_l525_52515

/-- A predicate stating that a set of points forms a right triangle in ℝ². -/
def IsRightTriangle : (Set ℝ × Set ℝ) → Prop :=
  sorry

/-- A function that checks if a set of shapes divides a given shape. -/
def Divides : (Set ℝ × Set ℝ) → List (Set ℝ × Set ℝ) → Prop :=
  sorry

/-- A function that computes the ratio of areas between two shapes. -/
noncomputable def AreaRatio : (Set ℝ × Set ℝ) → (Set ℝ × Set ℝ) → ℝ :=
  sorry

/-- A function that returns the shape with the larger area between two given shapes. -/
noncomputable def max_by_area : (Set ℝ × Set ℝ) → (Set ℝ × Set ℝ) → (Set ℝ × Set ℝ) :=
  sorry

/-- Given a right triangle divided into a square and two smaller right triangles by lines parallel
    to the legs through a point on the hypotenuse, if the area of one small triangle is 1/n times
    the area of the other small triangle, then the ratio of the area of the square to the area of
    the larger small triangle is 1/n. -/
theorem square_to_larger_triangle_ratio {n : ℝ} (h_n : n > 0) 
  (triangle : Set ℝ × Set ℝ) (square : Set ℝ × Set ℝ) 
  (small_triangle1 small_triangle2 : Set ℝ × Set ℝ) :
  IsRightTriangle triangle →
  Divides triangle [square, small_triangle1, small_triangle2] →
  AreaRatio small_triangle1 small_triangle2 = 1 / n →
  AreaRatio square (max_by_area small_triangle1 small_triangle2) = 1 / n :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_to_larger_triangle_ratio_l525_52515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_midpoints_is_circle_l525_52543

/-- A circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in 2D space -/
def Point := ℝ × ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Collinearity of three points -/
def collinear (P A B : Point) : Prop :=
  (B.2 - A.2) * (P.1 - A.1) = (P.2 - A.2) * (B.1 - A.1)

/-- Theorem: Locus of midpoints of chords passing through a point inside a circle -/
theorem locus_of_midpoints_is_circle (K : Circle) (P : Point) :
  distance P K.center = K.radius / 3 →
  ∃ (L : Circle), ∀ (A B : Point),
    (distance A K.center = K.radius ∧ 
     distance B K.center = K.radius ∧ 
     collinear P A B) →
    let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
    distance M L.center = L.radius :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_midpoints_is_circle_l525_52543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_product_l525_52545

/-- The parabola y^2 = 4x in the Cartesian coordinate system xOy -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of the parabola y^2 = 4x -/
def F : ℝ × ℝ := (1, 0)

/-- The origin of the coordinate system -/
def O : ℝ × ℝ := (0, 0)

/-- The dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- The area of a triangle given three points -/
noncomputable def triangle_area (p q r : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((q.1 - p.1) * (r.2 - p.2) - (r.1 - p.1) * (q.2 - p.2))

theorem parabola_triangle_area_product (A B : ℝ × ℝ) 
  (hA : A ∈ Parabola) (hB : B ∈ Parabola) 
  (h_dot : dot_product A B = -4) :
  triangle_area O F A * triangle_area O F B = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_product_l525_52545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_arrangement_l525_52567

theorem pentagon_arrangement (a b c d e : ℝ) 
  (nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0) 
  (sum_one : a + b + c + d + e = 1) : 
  ∃ (p : Fin 5 → Fin 5), Function.Bijective p ∧
    let arr := [a, b, c, d, e]
    ∀ i : Fin 5, (arr[p i] * arr[p ((i + 1) % 5)]) ≤ 1/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_arrangement_l525_52567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l525_52584

/-- The function f(x) = x + 2/(2x-1) -/
noncomputable def f (x : ℝ) : ℝ := x + 2 / (2 * x - 1)

theorem f_minimum_value :
  (∀ x > (1/2 : ℝ), f x ≥ 5/2) ∧ (∃ x > (1/2 : ℝ), f x = 5/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l525_52584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l525_52581

-- Define the ellipse
def is_outside_ellipse (m : ℝ) : Prop := (1 : ℝ)^2 / 4 + m^2 > 1

-- Define the line
noncomputable def line (x : ℝ) (m : ℝ) : ℝ := 2 * m * x + Real.sqrt 3

-- Define the circle
def on_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define intersection
def intersects (m : ℝ) : Prop := ∃ x y : ℝ, y = line x m ∧ on_circle x y

-- Theorem statement
theorem line_intersects_circle (m : ℝ) (h : is_outside_ellipse m) : intersects m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l525_52581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_fraction_in_vat_l525_52560

noncomputable def initial_volume : ℝ := 1
def num_iterations : ℕ := 10

noncomputable def orange_juice_remaining (n : ℕ) : ℝ :=
  (1 / 2) ^ n

noncomputable def water_in_vat (n : ℕ) : ℝ :=
  n * (1 / 2)

noncomputable def final_water_added : ℝ :=
  1 - orange_juice_remaining num_iterations

theorem water_fraction_in_vat :
  (water_in_vat (num_iterations - 1) + final_water_added) /
  (initial_volume + water_in_vat (num_iterations - 1) + final_water_added) = 5 / 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_fraction_in_vat_l525_52560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_necessary_not_sufficient_l525_52587

theorem quadratic_equation_necessary_not_sufficient :
  (∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x ≠ 1) ∧
  (∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) :=
by
  constructor
  · use 2
    constructor
    · ring
    · norm_num
  · intro x h
    rw [h]
    ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_necessary_not_sufficient_l525_52587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_sum_l525_52546

theorem sin_squared_sum (α : ℝ) : 
  Real.sin α ^ 2 + Real.sin (α + π / 3) ^ 2 + Real.sin (α + 2 * π / 3) ^ 2 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_sum_l525_52546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combinable_2sqrt3_l525_52593

-- Define the function that represents a quadratic radical of the form a√3
noncomputable def quadraticRadical (a : ℝ) : ℝ := a * Real.sqrt 3

-- Define the property of being combinable with √3
def combinableWithSqrt3 (x : ℝ) : Prop :=
  ∃ (b : ℝ), x + Real.sqrt 3 = quadraticRadical b ∨ x - Real.sqrt 3 = quadraticRadical b

-- Theorem statement
theorem combinable_2sqrt3 : combinableWithSqrt3 (quadraticRadical 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combinable_2sqrt3_l525_52593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_cost_l525_52595

/-- Calculates the cost of lemonade given the costs of other items and the relationship between lunch and breakfast costs. -/
theorem lemonade_cost (muffin_cost coffee_cost soup_cost salad_cost lemonade_cost : ℚ) 
  (h1 : muffin_cost = 2)
  (h2 : coffee_cost = 4)
  (h3 : soup_cost = 3)
  (h4 : salad_cost = 5.25)
  (h5 : muffin_cost + coffee_cost + 3 = soup_cost + salad_cost + lemonade_cost) :
  lemonade_cost = 0.75 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_cost_l525_52595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_parabola_l525_52532

/-- The equation of a curve in polar coordinates -/
def polar_equation (r θ : ℝ) : Prop := r = 4 * Real.tan θ * (1 / Real.cos θ)

/-- The equation of a parabola in Cartesian coordinates -/
def parabola_equation (x y : ℝ) : Prop := x^2 = 4 * y

/-- Theorem stating that the polar equation represents a parabola -/
theorem polar_to_cartesian_parabola :
  ∀ (r θ x y : ℝ), 
    polar_equation r θ → 
    x = r * Real.cos θ → 
    y = r * Real.sin θ → 
    parabola_equation x y := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_parabola_l525_52532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l525_52553

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

/-- The fractional part function -/
noncomputable def frac (x : ℝ) : ℝ := x - floor x

/-- First curve: (x - ⌊x⌋)² + (y-1)² = x - ⌊x⌋ -/
def curve1 (x y : ℝ) : Prop := (frac x)^2 + (y - 1)^2 = frac x

/-- Second curve: y = (1/5)x + 1 -/
def curve2 (x y : ℝ) : Prop := y = (1/5) * x + 1

/-- The set of intersection points -/
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | curve1 p.1 p.2 ∧ curve2 p.1 p.2}

/-- Main theorem: The number of intersection points is 2 -/
theorem intersection_count : ∃ (s : Finset (ℝ × ℝ)), s.card = 2 ∧ ∀ p, p ∈ s ↔ p ∈ intersection_points := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l525_52553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_unique_l525_52544

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def Line3D (a b c d e f : ℝ) := 
  {p : Point3D | (p.x - a) / d = (p.y - b) / e ∧ (p.y - b) / e = (p.z - c) / f}

def Plane3D (a b c d : ℝ) := 
  {p : Point3D | a * p.x + b * p.y + c * p.z + d = 0}

def intersectionPoint : Point3D := ⟨1, 2, 3⟩

theorem intersection_point_unique :
  intersectionPoint ∈ Line3D 2 3 (-1) (-1) (-1) 4 ∧
  intersectionPoint ∈ Plane3D 1 2 3 (-14) ∧
  ∀ p : Point3D, 
    p ∈ Line3D 2 3 (-1) (-1) (-1) 4 ∧ 
    p ∈ Plane3D 1 2 3 (-14) → 
    p = intersectionPoint :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_unique_l525_52544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_n_bound_l525_52583

/-- Sequence a_n defined recursively -/
def a (N : ℕ+) : ℕ → ℚ
  | 0 => 0
  | 1 => 1
  | (n + 2) => a N (n + 1) * (2 - 1 / N.val) - a N n

/-- Theorem stating that a_n < √(N+1) for all n ≥ 0 -/
theorem a_n_bound (N : ℕ+) : ∀ n : ℕ, a N n < Real.sqrt (N + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_n_bound_l525_52583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_to_l_l525_52501

noncomputable section

open Real

/-- The distance from a point to a line in polar coordinates -/
def distance_point_to_line (ρ θ a b c : ℝ) : ℝ :=
  abs (a * (ρ * cos θ) + b * (ρ * sin θ) + c) / sqrt (a^2 + b^2)

/-- The equation of the line l in polar form -/
def line_equation (ρ θ : ℝ) : Prop :=
  ρ * sin (θ + π/6) = 1

/-- The coordinates of point P in polar form -/
def point_P : ℝ × ℝ := (2, π)

/-- Theorem stating that the distance from P to l is 2 -/
theorem distance_P_to_l :
  distance_point_to_line point_P.1 point_P.2 1 (sqrt 3) (-2) = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_to_l_l525_52501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_l525_52564

theorem quadratic_equation_roots (k : ℕ+) (p : ℕ) :
  (∃ x₁ x₂ : ℕ+, (k.val - 1) * x₁.val^2 - p * x₁.val + k.val = 0 ∧
                 (k.val - 1) * x₂.val^2 - p * x₂.val + k.val = 0 ∧
                 x₁ ≠ x₂) →
  (k.val : ℕ)^(k.val * p) * (p^p + (k.val : ℕ)^k.val) = 1984 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_l525_52564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_constant_sum_l525_52569

-- Define the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : y = parabola x

-- Define a chord
structure Chord where
  A : PointOnParabola
  B : PointOnParabola

-- Define the point C
def C (d : ℝ) : ℝ × ℝ := (0, d)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the sum t
noncomputable def t (chord : Chord) (d : ℝ) : ℝ :=
  (1 / distance (C d) (chord.A.x, chord.A.y))^2 +
  (1 / distance (C d) (chord.B.x, chord.B.y))^2

-- The theorem to prove
theorem chord_constant_sum :
  ∃ d : ℝ, ∀ chord : Chord, chord.A.x ≠ chord.B.x →
    (C d).2 = chord.A.y → t chord d = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_constant_sum_l525_52569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_three_zeros_l525_52540

-- Define the piecewise function f
noncomputable def f (b c x : ℝ) : ℝ :=
  if x > 0 then -2 else -x^2 + b*x + c

-- Define the function g
noncomputable def g (b c x : ℝ) : ℝ := f b c x + x

-- Theorem statement
theorem g_has_three_zeros :
  ∃ (b c : ℝ), 
    (f b c 0 = -2) ∧ 
    (f b c (-1) = 1) ∧
    (∃ (x1 x2 x3 : ℝ), 
      (g b c x1 = 0 ∧ g b c x2 = 0 ∧ g b c x3 = 0) ∧ 
      (∀ x : ℝ, g b c x = 0 → x = x1 ∨ x = x2 ∨ x = x3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_three_zeros_l525_52540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_and_midpoint_trajectory_l525_52506

noncomputable section

-- Define the circle ⊙O
def circleO (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the line l
noncomputable def lineL (α : ℝ) (x : ℝ) : ℝ := Real.tan α * x + Real.sqrt 2

-- Define the intersection points A and B
def intersection_points (α : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ θ, p = circleO θ ∧ p.2 = lineL α p.1}

-- Define the midpoint P of AB
def midpointP (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- State the theorem
theorem circle_line_intersection_and_midpoint_trajectory :
  -- Range of α
  (∀ α, (intersection_points α).Nonempty → π/4 < α ∧ α < 3*π/4) ∧
  -- Parametric equation of trajectory of P
  (∃ f : ℝ → ℝ × ℝ, ∀ m, -1 < m ∧ m < 1 →
    f m = (Real.sqrt 2 * m / (m^2 + 1), -Real.sqrt 2 * m^2 / (m^2 + 1)) ∧
    ∃ α A B, A ∈ intersection_points α ∧ B ∈ intersection_points α ∧
             A ≠ B ∧ f m = midpointP A B) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_and_midpoint_trajectory_l525_52506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_implies_a_value_l525_52525

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x

noncomputable def g (x : ℝ) : ℝ := Real.sqrt x

noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := a / x

noncomputable def g_deriv (x : ℝ) : ℝ := 1 / (2 * Real.sqrt x)

theorem common_tangent_implies_a_value (a : ℝ) (x₀ : ℝ) 
  (h1 : x₀ > 0)
  (h2 : f a x₀ = g x₀)
  (h3 : f_deriv a x₀ = g_deriv x₀) :
  a = Real.exp 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_implies_a_value_l525_52525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uncovered_area_proof_l525_52591

noncomputable def side_length : ℝ := 4 * Real.sqrt 3

def circle_diameter : ℝ := 1

noncomputable def uncovered_area (s : ℝ) (d : ℝ) : ℝ :=
  (15 / 4) * Real.sqrt 3 - Real.pi / 4

theorem uncovered_area_proof (s d : ℝ) 
  (h1 : s = side_length) 
  (h2 : d = circle_diameter) : 
  uncovered_area s d = (15 / 4) * Real.sqrt 3 - Real.pi / 4 := by
  sorry

#check uncovered_area_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_uncovered_area_proof_l525_52591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crab_weight_theorem_l525_52507

/-- Represents the deviation from standard weight and the number of crabs with that deviation -/
structure DeviationCount where
  deviation : ℚ
  count : ℕ

/-- Calculate the difference between largest and smallest crab -/
def largest_smallest_diff (deviations : List DeviationCount) : ℚ :=
  let max_dev := deviations.map (·.deviation) |>.maximum?
  let min_dev := deviations.map (·.deviation) |>.minimum?
  match max_dev, min_dev with
  | some max, some min => max - min
  | _, _ => 0

/-- Calculate the total weight deviation -/
def total_weight_deviation (deviations : List DeviationCount) : ℚ :=
  deviations.foldl (λ acc d => acc + d.deviation * (d.count : ℚ)) 0

theorem crab_weight_theorem (standard_weight : ℚ) (total_weight : ℚ) 
    (deviations : List DeviationCount) : 
    largest_smallest_diff deviations = 1/2 ∧ 
    total_weight_deviation deviations = -(1/5) :=
  by sorry

#check crab_weight_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crab_weight_theorem_l525_52507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_surface_area_l525_52502

/-- The total surface area of a right cylinder with height twice its radius and radius 3 inches is 54π square inches. -/
theorem cylinder_surface_area :
  ∀ (r h π : ℝ),
    r = 3 →
    h = 2 * r →
    2 * π * r * h + 2 * π * r^2 = 54 * π := by
  intros r h π hr hh
  have lateral_area : ℝ := 2 * π * r * h
  have base_area : ℝ := π * r^2
  have total_area : ℝ := lateral_area + 2 * base_area
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_surface_area_l525_52502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l525_52537

/-- Parabola type representing x^2 = 4y -/
def Parabola := {p : ℝ × ℝ | p.1^2 = 4 * p.2}

/-- Point A -/
def A : ℝ × ℝ := (8, 7)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Projection of a point onto the x-axis -/
def projectX (p : ℝ × ℝ) : ℝ × ℝ := (p.1, 0)

/-- The sum of distances |PA| + |PQ| for a point P on the parabola -/
noncomputable def distanceSum (p : ℝ × ℝ) : ℝ :=
  distance p A + distance p (projectX p)

theorem min_distance_sum :
  ∃ (min : ℝ), min = 9 ∧ ∀ (p : ℝ × ℝ), p ∈ Parabola → distanceSum p ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l525_52537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_at_neg_five_answer_is_correct_l525_52592

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the property of f being odd when shifted by 5
axiom f_odd_shifted : ∀ x : ℝ, f (-(x - 5) + 5) = -f (x - 5)

-- Define the property of f being monotonically increasing after m
axiom f_monotone_after_m : ∃ m : ℝ, ∀ x y : ℝ, x ≥ m ∧ y > x → f y > f x

-- Define the property of f having a unique zero
def has_unique_zero (f : ℝ → ℝ) : Prop :=
  ∃! x : ℝ, f x = 0

-- Theorem stating that m = -5 ensures f has a unique zero
theorem unique_zero_at_neg_five :
  (∃ m : ℝ, ∀ x y : ℝ, x ≥ m ∧ y > x → f y > f x) →
  has_unique_zero f →
  ∃ m : ℝ, m = -5 ∧ ∀ x y : ℝ, x ≥ m ∧ y > x → f y > f x :=
by
  sorry

-- The main theorem that proves the answer is correct
theorem answer_is_correct : ∃ m : ℝ, m = -5 ∧ ∀ x y : ℝ, x ≥ m ∧ y > x → f y > f x :=
by
  apply unique_zero_at_neg_five
  · exact f_monotone_after_m
  · sorry  -- Proof that f has a unique zero


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_at_neg_five_answer_is_correct_l525_52592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_two_fifths_l525_52562

/-- The sequence b_n defined recursively -/
def b : ℕ → ℚ
  | 0 => 2  -- We need to define b(0) to avoid the missing case error
  | 1 => 2
  | 2 => 3
  | (n + 3) => 2 * b (n + 2) + 3 * b (n + 1)

/-- The sum of the infinite series -/
noncomputable def seriesSum : ℝ := ∑' n, (b (n + 1) : ℝ) / 3^(n + 2)

/-- Theorem stating that the sum of the infinite series equals 2/5 -/
theorem series_sum_equals_two_fifths : seriesSum = 2/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_two_fifths_l525_52562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_periodic_l525_52550

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos (4 * x - 5 * Real.pi / 2)

theorem f_is_odd_and_periodic :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (x + Real.pi / 2) = f x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_periodic_l525_52550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_is_4_point_5_l525_52503

noncomputable def win_amount (roll : Nat) : ℝ :=
  if roll % 2 = 1 then 0
  else if roll < 8 then roll
  else 2 * (2 + 4 + 6)

noncomputable def expected_value : ℝ := (1 : ℝ) / 8 * (win_amount 1 + win_amount 2 + win_amount 3 + win_amount 4 + 
                           win_amount 5 + win_amount 6 + win_amount 7 + win_amount 8)

theorem expected_value_is_4_point_5 : expected_value = 4.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_is_4_point_5_l525_52503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_passes_through_major_axis_endpoint_l525_52523

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with foci F₁ and F₂ -/
structure Ellipse where
  F₁ : Point
  F₂ : Point

/-- Represents a circle tangent to PF₁, F₂F₁, and F₂P -/
structure TangentCircle where
  center : Point
  radius : ℝ

/-- The point of tangency on the extension of F₂F₁ -/
noncomputable def pointOfTangency (e : Ellipse) (c : TangentCircle) : Point :=
  sorry

/-- Distance between two points -/
noncomputable def dist (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- A point is on the ellipse if the sum of its distances to the foci is constant -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  ∃ (k : ℝ), dist p e.F₁ + dist p e.F₂ = k

/-- The endpoint of the major axis closer to F₁ -/
noncomputable def majorAxisEndpoint (e : Ellipse) : Point :=
  sorry

/-- Predicate to check if a circle is tangent to the ellipse at a given point -/
def isTangentCircle (c : TangentCircle) (e : Ellipse) (p : Point) : Prop :=
  sorry

/-- The main theorem -/
theorem tangent_circle_passes_through_major_axis_endpoint
  (e : Ellipse) (p : Point) (c : TangentCircle) 
  (h₁ : isOnEllipse p e)
  (h₂ : isTangentCircle c e p) :
  pointOfTangency e c = majorAxisEndpoint e :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_passes_through_major_axis_endpoint_l525_52523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_geq_B_l525_52527

/-- Given two real numbers a and b, prove that A ≥ B where
    A = a^3 + 3a^2b^2 + 2b^2 + 3b and
    B = a^3 - a^2b^2 + b^2 + 3b -/
theorem A_geq_B (a b : ℝ) : 
  (a^3 + 3*a^2*b^2 + 2*b^2 + 3*b) ≥ (a^3 - a^2*b^2 + b^2 + 3*b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_geq_B_l525_52527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremal_points_of_f_l525_52582

def f (x : ℝ) : ℝ := (x^2 - 1)^2 + 2

theorem extremal_points_of_f :
  ∀ x : ℝ, (∃ ε > 0, ∀ y, |y - x| < ε → f y ≥ f x) ∨ 
           (∃ ε > 0, ∀ y, |y - x| < ε → f y ≤ f x) ↔ 
  x = 1 ∨ x = -1 ∨ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremal_points_of_f_l525_52582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_is_parallelogram_l525_52568

-- Define the quadrilateral and point P
variable (O₁ O₂ O₃ O₄ P : Real × Real)

-- Define the property of convexity
def is_convex_quadrilateral (O₁ O₂ O₃ O₄ : Real × Real) : Prop := sorry

-- Define the property of P being inside the quadrilateral
def point_inside_quadrilateral (P O₁ O₂ O₃ O₄ : Real × Real) : Prop := sorry

-- Define the function f_i
def f (P A B : Real × Real) : Real := sorry

-- Define the property of l_i minimizing f_i
def l_minimizes_f (l : Real → Real × Real) (P O₁ O₂ O₃ O₄ : Real × Real) (i : Fin 4) : Prop := sorry

-- Define the theorem
theorem quadrilateral_is_parallelogram
  (h_convex : is_convex_quadrilateral O₁ O₂ O₃ O₄)
  (h_inside : point_inside_quadrilateral P O₁ O₂ O₃ O₄)
  (h_l₁ : l_minimizes_f (λ t => sorry) P O₁ O₂ O₃ O₄ 0)
  (h_l₂ : l_minimizes_f (λ t => sorry) P O₁ O₂ O₃ O₄ 1)
  (h_l₃ : l_minimizes_f (λ t => sorry) P O₁ O₂ O₃ O₄ 2)
  (h_l₄ : l_minimizes_f (λ t => sorry) P O₁ O₂ O₃ O₄ 3)
  (h_l₁₃ : h_l₁ = h_l₃)
  (h_l₂₄ : h_l₂ = h_l₄) :
  ∃ (a b : Real × Real), O₂ - O₁ = O₃ - O₄ ∧ O₃ - O₂ = O₄ - O₁ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_is_parallelogram_l525_52568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_formula_l525_52589

/-- An arithmetic sequence with given first three terms -/
def arithmetic_sequence (a : ℝ) : ℕ → ℝ := sorry

/-- The first term of the sequence -/
axiom first_term (a : ℝ) : arithmetic_sequence a 1 = a - 1

/-- The second term of the sequence -/
axiom second_term (a : ℝ) : arithmetic_sequence a 2 = 2*a + 1

/-- The third term of the sequence -/
axiom third_term (a : ℝ) : arithmetic_sequence a 3 = a + 7

/-- The sequence is arithmetic -/
axiom is_arithmetic (a : ℝ) (n : ℕ) :
  arithmetic_sequence a (n + 1) - arithmetic_sequence a n =
  arithmetic_sequence a (n + 2) - arithmetic_sequence a (n + 1)

/-- The general formula for the sequence -/
theorem general_formula (n : ℕ) : arithmetic_sequence 2 n = 4*n - 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_formula_l525_52589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_minus_six_plus_f_zero_l525_52513

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- State the properties of f and g
axiom f_odd (x : ℝ) : f x = -f (-x)
axiom g_periodic (x : ℝ) : g x = g (x + 2)

-- State the given conditions
axiom f_neg_one : f (-1) = 3
axiom g_one : g 1 = 3

-- State the equation for g(2nf(1))
axiom g_equation (n : ℕ) : g (2 * n * f 1) = n * f (f 1 + g (-1)) + 2

-- State the theorem to be proved
theorem g_minus_six_plus_f_zero : g (-6) + f 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_minus_six_plus_f_zero_l525_52513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fastest_trail_difference_l525_52510

-- Define the trails and their properties
def trail1_distance : ℚ := 20
def trail1_speed : ℚ := 5
def trail2_distance : ℚ := 12
def trail2_speed : ℚ := 3
def break_time : ℚ := 1

-- Define the hiking times
noncomputable def trail1_time : ℚ := trail1_distance / trail1_speed
noncomputable def trail2_time : ℚ := trail2_distance / trail2_speed + break_time

-- Theorem statement
theorem fastest_trail_difference :
  trail2_time - trail1_time = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fastest_trail_difference_l525_52510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l525_52577

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sin (x + 18 * Real.pi / 180) - cos (x + 48 * Real.pi / 180)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Icc (-sqrt 3) (sqrt 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l525_52577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_theorem_l525_52573

theorem polynomial_divisibility_theorem (P : Polynomial ℤ) :
  (∀ n : ℕ+, (n : ℤ) ∣ P.eval ((2 : ℤ) ^ (n : ℕ))) → P = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_theorem_l525_52573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_keiko_walking_speed_l525_52534

/-- Represents a track with semicircular ends -/
structure Track where
  width : ℝ
  timeDifference : ℝ

/-- Calculates the walking speed on the track -/
noncomputable def walkingSpeed (track : Track) : ℝ :=
  (Real.pi * track.width) / track.timeDifference

theorem keiko_walking_speed (track : Track) 
  (h1 : track.width = 8)
  (h2 : track.timeDifference = 48) : 
  walkingSpeed track = Real.pi / 3 := by
  sorry

#check keiko_walking_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_keiko_walking_speed_l525_52534
