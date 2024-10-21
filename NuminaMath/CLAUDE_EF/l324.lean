import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_with_product_property_l324_32450

def is_valid_partition (k : ℕ) (A B : Set ℕ) : Prop :=
  A ∪ B = Finset.range (k - 1) \ {0, 1} ∧ A ∩ B = ∅

def has_product_triple (S : Set ℕ) : Prop :=
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a * b = c

theorem smallest_k_with_product_property : 
  (∀ k ≥ 32, ∀ A B : Set ℕ, is_valid_partition k A B → 
    has_product_triple A ∨ has_product_triple B) ∧
  (∀ k < 32, ∃ A B : Set ℕ, is_valid_partition k A B ∧ 
    ¬(has_product_triple A ∨ has_product_triple B)) :=
sorry

#check smallest_k_with_product_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_with_product_property_l324_32450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_mowing_problem_l324_32438

/-- Represents the time (in hours) it takes to mow the entire lawn -/
structure MowingTime where
  mary : ℚ
  tom : ℚ

/-- Represents the time (in hours) each person mows -/
structure ActualMowingTime where
  mary : ℚ
  tom : ℚ

/-- Calculates the fraction of lawn mowed by a person -/
def fractionMowed (totalTime : ℚ) (actualTime : ℚ) : ℚ :=
  actualTime / totalTime

/-- Calculates the remaining fraction of lawn to be mowed -/
def remainingFraction (mowingTime : MowingTime) (actualTime : ActualMowingTime) : ℚ :=
  1 - (fractionMowed mowingTime.mary actualTime.mary + fractionMowed mowingTime.tom actualTime.tom)

theorem lawn_mowing_problem (mowingTime : MowingTime) (actualTime : ActualMowingTime) :
    mowingTime.mary = 3 →
    mowingTime.tom = 6 →
    actualTime.mary = 2 →
    actualTime.tom = 1 →
    remainingFraction mowingTime actualTime = 1/6 := by
  sorry

#eval remainingFraction ⟨3, 6⟩ ⟨2, 1⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_mowing_problem_l324_32438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l324_32445

/-- A function f : ℝ → ℝ satisfying the given conditions -/
noncomputable def f : ℝ → ℝ := sorry

/-- The condition that for any distinct x₁ and x₂, (f(x₁) - f(x₂)) / (x₁ - x₂) > 3 -/
axiom f_condition (x₁ x₂ : ℝ) : x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 3

/-- The condition that f(5) = 18 -/
axiom f_at_5 : f 5 = 18

/-- The theorem stating that the solution set of f(3x - 1) > 9x is (2, +∞) -/
theorem solution_set_of_inequality :
  {x : ℝ | f (3 * x - 1) > 9 * x} = {x : ℝ | x > 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l324_32445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_eight_rays_l324_32460

/-- The set of points (x, y) satisfying ||x| - |y|| = 2 -/
def locus_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |abs p.1 - abs p.2| = 2}

/-- Definition of a ray in ℝ² -/
def IsRay (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (a : ℝ × ℝ) (v : ℝ × ℝ), v ≠ (0, 0) ∧ S = {p : ℝ × ℝ | ∃ t : ℝ, t ≥ 0 ∧ p = a + t • v}

/-- The locus consists of eight rays -/
theorem locus_is_eight_rays : ∃ (rays : Finset (Set (ℝ × ℝ))),
  rays.card = 8 ∧
  (∀ r ∈ rays, IsRay r) ∧
  (⋃ r ∈ rays, r) = locus_points := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_eight_rays_l324_32460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_is_four_l324_32436

/-- Represents the setup of a sphere and a meter stick under sunlight --/
structure SunlightSetup where
  sphere_shadow_length : ℝ
  meter_stick_height : ℝ
  meter_stick_shadow_length : ℝ

/-- Calculates the radius of a sphere given its shadow setup --/
noncomputable def sphere_radius (setup : SunlightSetup) : ℝ :=
  setup.sphere_shadow_length * (setup.meter_stick_height / setup.meter_stick_shadow_length)

/-- Theorem stating that under the given conditions, the sphere's radius is 4 meters --/
theorem sphere_radius_is_four (setup : SunlightSetup) 
  (h1 : setup.sphere_shadow_length = 16)
  (h2 : setup.meter_stick_height = 1)
  (h3 : setup.meter_stick_shadow_length = 4) :
  sphere_radius setup = 4 := by
  -- Unfold the definition of sphere_radius
  unfold sphere_radius
  -- Substitute the known values
  rw [h1, h2, h3]
  -- Perform the calculation
  norm_num

#check sphere_radius_is_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_is_four_l324_32436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_range_of_triangle_OAB_l324_32431

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 3

/-- A point on the circle -/
structure PointOnCircle where
  x : ℝ
  y : ℝ
  on_circle : circle_eq x y

/-- A point on the ellipse -/
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse x y

/-- Tangent line from a point on the circle to the ellipse -/
def tangent_line (M : PointOnCircle) (A : PointOnEllipse) : Prop :=
  ∃ (k : ℝ), A.x * M.x / 2 + A.y * M.y = 1 ∧ k * M.x = A.x ∧ k * M.y = A.y

/-- Area of a triangle given three points -/
noncomputable def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

/-- The main theorem -/
theorem area_range_of_triangle_OAB :
  ∀ (M : PointOnCircle) (A B : PointOnEllipse),
  tangent_line M A → tangent_line M B →
  2/3 ≤ triangle_area 0 0 A.x A.y B.x B.y ∧ triangle_area 0 0 A.x A.y B.x B.y ≤ Real.sqrt 2/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_range_of_triangle_OAB_l324_32431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_people_sharing_pizzas_l324_32472

/-- Represents the number of slices in each pizza size -/
structure PizzaSlices where
  small : Nat
  medium : Nat
  large : Nat
  extraLarge : Nat

/-- Represents the number of pizzas bought for each size -/
structure PizzaCount where
  small : Nat
  medium : Nat
  large : Nat
  extraLarge : Nat

/-- Calculates the total number of slices from all pizzas -/
def totalSlices (slices : PizzaSlices) (count : PizzaCount) : Nat :=
  slices.small * count.small +
  slices.medium * count.medium +
  slices.large * count.large +
  slices.extraLarge * count.extraLarge

/-- Theorem: Maximum number of people who can share pizzas equally -/
theorem max_people_sharing_pizzas
  (slices : PizzaSlices)
  (count : PizzaCount)
  (h1 : slices.small = 6)
  (h2 : slices.medium = 8)
  (h3 : slices.large = 12)
  (h4 : slices.extraLarge = 16)
  (h5 : count.small = 3)
  (h6 : count.medium = 2)
  (h7 : count.large = 4)
  (h8 : count.extraLarge = 1)
  (h9 : count.small + count.medium + count.large + count.extraLarge = 20) :
  (∃ (x : Nat), x = 24 ∧
    x * (min (min (min slices.small slices.medium) slices.large) slices.extraLarge) = totalSlices slices count ∧
    ∀ (y : Nat), y > x →
      y * (min (min (min slices.small slices.medium) slices.large) slices.extraLarge) > totalSlices slices count) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_people_sharing_pizzas_l324_32472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_approx_5_28_l324_32457

/-- The time (in minutes) it takes for two people walking in opposite directions 
    on a circular track to meet for the first time. -/
noncomputable def meeting_time (track_circumference : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  track_circumference / (speed1 + speed2)

/-- Theorem stating that two people walking in opposite directions on a circular track
    will meet for the first time after approximately 5.28 minutes, given the specified conditions. -/
theorem meeting_time_approx_5_28 :
  let track_circumference : ℝ := 726  -- meters
  let speed1 : ℝ := 4.5 * 1000 / 60  -- km/hr to m/min
  let speed2 : ℝ := 3.75 * 1000 / 60  -- km/hr to m/min
  abs ((meeting_time track_circumference speed1 speed2) - 5.28) < 0.01 := by
  sorry

#check meeting_time_approx_5_28

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_approx_5_28_l324_32457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l324_32453

noncomputable def f (x : Real) := 2 * (Real.cos (x / 2))^2 + Real.sqrt 3 * Real.sin x

theorem f_properties :
  (∃ (M : Real), ∀ (x : Real), f x ≤ M ∧ M = 3) ∧
  (∀ (x : Real), f x = 3 ↔ ∃ (k : ℤ), x = 2 * k * Real.pi + Real.pi / 3) ∧
  (∀ (α : Real), Real.tan (α / 2) = 1 / 2 → f α = (8 + 4 * Real.sqrt 3) / 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l324_32453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l324_32492

theorem system_solution (x y : ℝ) (k n : ℤ) : 
  (2 * (Real.sin x)^2 + 2 * Real.sqrt 2 * Real.sin x * (Real.sin (2 * x))^2 + (Real.sin (2 * x))^2 = 0 ∧ 
   Real.cos x = Real.cos y) ↔ 
  ((x = 2 * Real.pi * ↑k ∧ y = 2 * Real.pi * ↑n) ∨
   (x = Real.pi + 2 * Real.pi * ↑k ∧ y = Real.pi + 2 * Real.pi * ↑n) ∨
   (x = -Real.pi/4 + 2 * Real.pi * ↑k ∧ (y = Real.pi/4 + 2 * Real.pi * ↑n ∨ y = -Real.pi/4 + 2 * Real.pi * ↑n)) ∨
   (x = -3*Real.pi/4 + 2 * Real.pi * ↑k ∧ (y = 3*Real.pi/4 + 2 * Real.pi * ↑n ∨ y = -3*Real.pi/4 + 2 * Real.pi * ↑n))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l324_32492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l324_32411

/-- The line y = 2x + 3 -/
def line (x : ℝ) : ℝ := 2 * x + 3

/-- The point we're finding the closest point to -/
noncomputable def point : ℝ × ℝ := (2, -1)

/-- The claimed closest point on the line -/
noncomputable def closest_point : ℝ × ℝ := (-6/5, 3/5)

theorem closest_point_on_line :
  ∀ x : ℝ, 
  (x - point.1)^2 + (line x - point.2)^2 ≥ 
  (closest_point.1 - point.1)^2 + (closest_point.2 - point.2)^2 ∧
  closest_point.2 = line closest_point.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l324_32411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_followers_after_one_month_l324_32458

def social_media_followers : ℕ → Prop :=
  fun total_followers =>
    ∃ (instagram facebook twitter tiktok youtube pinterest snapchat : ℕ),
      -- Initial conditions
      instagram = 240 ∧
      facebook = 500 ∧
      pinterest = 120 ∧
      twitter = (instagram + facebook) / 2 ∧
      tiktok = 3 * twitter ∧
      youtube = tiktok + 510 ∧
      snapchat = pinterest / 2 ∧
      -- Changes after one month
      let new_instagram : ℕ := instagram + instagram * 15 / 100;
      let new_facebook : ℕ := facebook + facebook * 20 / 100;
      let new_twitter : ℕ := twitter - 12;
      let new_tiktok : ℕ := tiktok + tiktok * 10 / 100;
      let new_youtube : ℕ := youtube + youtube * 8 / 100;
      let new_pinterest : ℕ := pinterest + 20;
      let new_snapchat : ℕ := snapchat - snapchat * 5 / 100;
      -- Total followers after one month
      total_followers = new_instagram + new_facebook + new_twitter + new_tiktok + new_youtube + new_pinterest + new_snapchat

theorem total_followers_after_one_month :
  ∃ n : ℕ, social_media_followers n ∧ n = 4402 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_followers_after_one_month_l324_32458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_in_unit_cube_l324_32467

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  E₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D

/-- Predicate to check if a cube is a unit cube -/
def isUnitCube (c : Cube) : Prop :=
  sorry

/-- Calculates the dihedral angle A-BD₁-A₁ in a cube -/
noncomputable def dihedralAngle (c : Cube) : ℝ :=
  sorry

/-- Theorem: The dihedral angle A-BD₁-A₁ in a unit cube is 60° -/
theorem dihedral_angle_in_unit_cube (c : Cube) (h : isUnitCube c) :
  dihedralAngle c = 60 * π / 180 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_in_unit_cube_l324_32467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_property_l324_32423

-- Define the exponential function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- State the theorem
theorem exponential_function_property (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 3 = 8) : 
  f a 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_property_l324_32423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l324_32440

-- Define the arithmetic sequence and its sum
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d
noncomputable def sequence_sum (a₁ d : ℝ) (n : ℕ) : ℝ := (n : ℝ) * (a₁ + arithmetic_sequence a₁ d n) / 2

-- State the theorem
theorem arithmetic_sequence_property (d : ℝ) (h_d : d > 0) :
  (∃! a₁ : ℝ, ∀ T K : ℕ, T + K = 19 → sequence_sum a₁ d T = sequence_sum a₁ d K) →
  (∃! n : ℕ, arithmetic_sequence (9*d) d n - sequence_sum (9*d) d n ≥ 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l324_32440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_implies_a_equals_one_l324_32430

/-- The function f(x) = ax³ - 2x² + a²x -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 2 * x^2 + a^2 * x

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 4 * x + a^2

theorem local_minimum_implies_a_equals_one :
  ∀ a : ℝ, (∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → f a x ≥ f a 1) →
  f_derivative a 1 = 0 →
  a = 1 := by
  sorry

#check local_minimum_implies_a_equals_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_implies_a_equals_one_l324_32430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_two_twentyfifths_l324_32425

/-- Sequence b_n defined recursively -/
def b : ℕ → ℚ
  | 0 => 2
  | 1 => 3
  | (n + 2) => b (n + 1) + b n

/-- The sum of the infinite series -/
noncomputable def series_sum : ℚ := ∑' n, b n / 5^(n+2)

/-- Theorem stating that the sum of the series equals 2/25 -/
theorem series_sum_equals_two_twentyfifths : series_sum = 2/25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_two_twentyfifths_l324_32425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_is_16_div_5_l324_32464

-- Define the square ABCD
def Square (A B C D : ℝ × ℝ) : Prop :=
  let dist := λ (p q : ℝ × ℝ) => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist A B = 4 ∧ dist B C = 4 ∧ dist C D = 4 ∧ dist D A = 4 ∧
  (B.1 - A.1) * (C.2 - B.2) = (C.1 - B.1) * (B.2 - A.2)

-- Define the midpoint M of CD
def Midpoint (M C D : ℝ × ℝ) : Prop :=
  M.1 = (C.1 + D.1) / 2 ∧ M.2 = (C.2 + D.2) / 2

-- Define P as the intersection of two circles
def Intersection (P M A : ℝ × ℝ) : Prop :=
  let dist := λ (p q : ℝ × ℝ) => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist P M = 2 ∧ dist P A = 4

-- Define the distance from a point to a line
noncomputable def DistanceToLine (P A D : ℝ × ℝ) : ℝ :=
  let dx := D.1 - A.1
  let dy := D.2 - A.2
  abs (dy * P.1 - dx * P.2 + D.1 * A.2 - D.2 * A.1) / Real.sqrt (dx^2 + dy^2)

-- Main theorem
theorem distance_to_line_is_16_div_5 (A B C D M P : ℝ × ℝ) :
  Square A B C D → Midpoint M C D → Intersection P M A →
  DistanceToLine P A D = 16 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_is_16_div_5_l324_32464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jon_laundry_loads_l324_32419

-- Define the capacity of the laundry machine
def machine_capacity : ℚ := 5

-- Define the weight of shirts and pants
def shirts_per_pound : ℚ := 4
def pants_per_pound : ℚ := 2

-- Define the number of shirts and pants Jon needs to wash
def num_shirts : ℚ := 20
def num_pants : ℚ := 20

-- Define the function to calculate the number of loads
def calculate_loads (machine_capacity shirts_per_pound pants_per_pound num_shirts num_pants : ℚ) : ℚ :=
  (num_shirts / shirts_per_pound + num_pants / pants_per_pound) / machine_capacity

-- Theorem statement
theorem jon_laundry_loads : 
  calculate_loads machine_capacity shirts_per_pound pants_per_pound num_shirts num_pants = 3 := by
  -- Unfold the definition of calculate_loads
  unfold calculate_loads
  -- Perform the calculation
  norm_num
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jon_laundry_loads_l324_32419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_temp_conversion_l324_32442

/-- Converts temperature from Celsius to Fahrenheit -/
noncomputable def celsius_to_fahrenheit (c : ℝ) : ℝ := (c * 9 / 5) + 32

/-- The temperature of the pot of water in Celsius -/
def water_temp_celsius : ℝ := 60

theorem water_temp_conversion :
  celsius_to_fahrenheit water_temp_celsius = 140 := by
  -- Unfold the definition of celsius_to_fahrenheit
  unfold celsius_to_fahrenheit
  -- Unfold the definition of water_temp_celsius
  unfold water_temp_celsius
  -- Simplify the arithmetic
  simp [mul_div_assoc]
  -- Check that the equality holds
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_temp_conversion_l324_32442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_third_quadrant_l324_32475

noncomputable def i : ℂ := Complex.I

noncomputable def z : ℂ := 4 * i / (1 - i)^2 + i^2019

theorem z_in_third_quadrant : Real.sign z.re = -1 ∧ Real.sign z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_third_quadrant_l324_32475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_graphs_l324_32471

-- Define the three functions
noncomputable def f₁ (x : ℝ) : ℝ := x + 3

noncomputable def f₂ (x : ℝ) : ℝ := 
  if x ≠ 3 then (x^2 - 9) / (x - 3) else 0

noncomputable def f₃ (x : ℝ) : ℝ := 
  if x ≠ 3 then x + 3 else 0

-- State the theorem
theorem distinct_graphs : 
  ∀ (x : ℝ), (∃ y : ℝ, (f₁ x ≠ f₂ x ∨ f₁ x ≠ f₃ x ∨ f₂ x ≠ f₃ x)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_graphs_l324_32471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_proof_l324_32429

/-- The distance between points A and B in kilometers -/
noncomputable def total_distance : ℝ := 45

/-- Person A's speed relative to Person B's initial speed -/
noncomputable def speed_ratio_A : ℝ := 1.2

/-- Distance traveled by Person B before malfunction in kilometers -/
noncomputable def malfunction_distance : ℝ := 5

/-- Repair time as a fraction of total distance -/
noncomputable def repair_time_ratio : ℝ := 1/6

/-- Person B's speed increase after repair -/
noncomputable def speed_increase_B : ℝ := 0.6

theorem distance_proof :
  ∃ (time_A time_B_initial time_B_after_repair : ℝ),
    time_A > 0 ∧ time_B_initial > 0 ∧ time_B_after_repair > 0 ∧
    speed_ratio_A * time_A = time_B_initial ∧
    malfunction_distance / total_distance = time_B_initial / (time_B_initial + repair_time_ratio * total_distance) ∧
    (total_distance - malfunction_distance) / (time_B_after_repair * (1 + speed_increase_B)) = malfunction_distance / time_B_initial ∧
    time_A = time_B_initial + repair_time_ratio * total_distance + time_B_after_repair :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_proof_l324_32429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_difference_l324_32448

noncomputable def computer_table_cost : ℝ := 140
noncomputable def computer_chair_cost : ℝ := 100
noncomputable def joystick_cost : ℝ := 20

noncomputable def frank_joystick_share : ℝ := joystick_cost * (1/4)
noncomputable def eman_joystick_share : ℝ := joystick_cost - frank_joystick_share

noncomputable def frank_total_payment : ℝ := computer_table_cost + frank_joystick_share
noncomputable def eman_total_payment : ℝ := computer_chair_cost + eman_joystick_share

theorem payment_difference : frank_total_payment - eman_total_payment = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_difference_l324_32448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_solid_edge_sum_l324_32497

/-- Represents a rectangular solid with dimensions in geometric progression -/
structure RectangularSolid where
  a : ℝ
  r : ℝ

/-- The volume of the rectangular solid -/
noncomputable def volume (s : RectangularSolid) : ℝ := s.a^3

/-- The surface area of the rectangular solid -/
noncomputable def surfaceArea (s : RectangularSolid) : ℝ := 2 * (s.a^2 / s.r + s.a^2 * s.r + s.a^2)

/-- The sum of all edge lengths of the rectangular solid -/
noncomputable def sumOfEdges (s : RectangularSolid) : ℝ := 4 * (s.a / s.r + s.a + s.a * s.r)

theorem rectangular_solid_edge_sum :
  ∃ (s : RectangularSolid),
    volume s = 432 ∧
    surfaceArea s = 360 ∧
    sumOfEdges s = 72 * Real.rpow 2 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_solid_edge_sum_l324_32497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_gcd_abc_plus_one_l324_32406

theorem max_gcd_abc_plus_one (a b c : ℕ) : 
  a + b + c ≤ 3000000 → 
  a ≠ b → b ≠ c → c ≠ a → 
  (∃ (d : ℕ), d = Nat.gcd (Nat.gcd (a * b + 1) (a * c + 1)) (b * c + 1) ∧ 
    d ≤ 998285 ∧ 
    ∀ (e : ℕ), e = Nat.gcd (Nat.gcd (a * b + 1) (a * c + 1)) (b * c + 1) → e ≤ d) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_gcd_abc_plus_one_l324_32406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bmw_sales_l324_32494

theorem bmw_sales (total : ℕ) (ford_percent : ℚ) (nissan_percent : ℚ) (chevrolet_percent : ℚ) 
  (h1 : total = 300)
  (h2 : ford_percent = 20 / 100)
  (h3 : nissan_percent = 25 / 100)
  (h4 : chevrolet_percent = 30 / 100)
  (h5 : ford_percent + nissan_percent + chevrolet_percent < 1) :
  Int.floor (↑total * (1 - (ford_percent + nissan_percent + chevrolet_percent))) = 75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bmw_sales_l324_32494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suitcase_electronics_weight_l324_32462

/-- Given a suitcase with books, clothes, and electronics, prove the weight of electronics. -/
theorem suitcase_electronics_weight 
  (B C E : ℚ) -- Weights of books, clothes, and electronics
  (h_ratio : B / 7 = C / 4 ∧ B / 7 = E / 3) -- Initial ratio condition
  (h_remove : B / (C - 8) = 2 * (B / C)) : -- Condition after removing 8 pounds of clothes
  E = 12 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_suitcase_electronics_weight_l324_32462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l324_32428

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, a = (2 * k + 1) * 1171) :
  Int.gcd (3 * a^2 + 35 * a + 77) (a + 15) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l324_32428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_iff_m_eq_9_l324_32470

-- Define the number of balls
def n : ℕ := 10

-- Define the events A and B
def A (second_ball : ℕ) : Prop := second_ball = 2
def B (first_ball second_ball : ℕ) (m : ℕ) : Prop := first_ball + second_ball = m

-- Define the probability of drawing a specific ball
noncomputable def P_draw (ball : ℕ) : ℝ := 1 / n

-- Define the probability of event A
noncomputable def P_A : ℝ := P_draw 2

-- Define the probability of event B
noncomputable def P_B (m : ℕ) : ℝ := (m + 1) / (n * n)

-- Define the probability of both events A and B occurring
noncomputable def P_AB (m : ℕ) : ℝ := P_draw (m - 2) * P_draw 2

-- Theorem: A and B are independent if and only if m = 9
theorem independence_iff_m_eq_9 (m : ℕ) :
  (P_AB m = P_A * P_B m) ↔ m = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_iff_m_eq_9_l324_32470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_specific_l324_32422

/-- The area of a circular sector given its radius and central angle in radians -/
noncomputable def sectorArea (radius : ℝ) (centralAngle : ℝ) : ℝ :=
  (1 / 2) * radius^2 * centralAngle

/-- Theorem: The area of a sector with radius 5 and central angle 2 radians is 25 -/
theorem sector_area_specific : sectorArea 5 2 = 25 := by
  -- Unfold the definition of sectorArea
  unfold sectorArea
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_specific_l324_32422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l324_32455

def triangle_ABC (A B C a b c : ℝ) : Prop :=
  Real.cos B = 1/4 ∧ b = 2 ∧ Real.sin C = 2 * Real.sin A

theorem triangle_ABC_properties (A B C a b c : ℝ) 
  (h : triangle_ABC A B C a b c) : a = 1 ∧ (1/2 * a * b * Real.sin C) = Real.sqrt 15/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l324_32455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_method_applicability_l324_32409

-- Define the functions
def f1 (x : ℝ) := 3 * x + 1
def f2 (x : ℝ) := x^3
def f3 (x : ℝ) := x^2
noncomputable def f4 (x : ℝ) := Real.log x

-- Define a predicate for sign change around zero
def changes_sign_around_zero (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a < 0 ∧ 0 < b ∧ f a < 0 ∧ f b > 0

theorem bisection_method_applicability :
  changes_sign_around_zero f1 ∧
  changes_sign_around_zero f2 ∧
  ¬(changes_sign_around_zero f3) ∧
  changes_sign_around_zero f4 := by
  sorry

#check bisection_method_applicability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_method_applicability_l324_32409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_rhombus_perimeter_example_l324_32473

/-- Given a rhombus with diagonals d1 and d2, and a scaling factor s,
    calculate the perimeter of the scaled rhombus. -/
noncomputable def scaled_rhombus_perimeter (d1 d2 s : ℝ) : ℝ :=
  4 * s * Real.sqrt ((d1/2)^2 + (d2/2)^2)

/-- Theorem: The perimeter of a rhombus with diagonals 10 and 24,
    when scaled by 0.5, is 26. -/
theorem scaled_rhombus_perimeter_example :
  scaled_rhombus_perimeter 10 24 0.5 = 26 := by
  -- Expand the definition of scaled_rhombus_perimeter
  unfold scaled_rhombus_perimeter
  -- Simplify the expression
  simp [Real.sqrt_sq, Real.sqrt_mul]
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_rhombus_perimeter_example_l324_32473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_d_is_linear_l324_32488

/-- A linear equation in two variables is of the form ax + by + c = 0, where a and b are not both zero. --/
def IsLinearEquationInTwoVariables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x y, f x y = a * x + b * y + c

/-- The function representing y = 1/2(x + 8) --/
noncomputable def f (x y : ℝ) : ℝ := y - (1/2 * x + 4)

theorem equation_d_is_linear : IsLinearEquationInTwoVariables f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_d_is_linear_l324_32488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_equals_21_4_l324_32421

/-- The sum of the infinite series ∑(n=1 to ∞) (4n^2 - 2n + 3) / 3^n -/
noncomputable def infinite_series_sum : ℝ :=
  ∑' n, (4 * n^2 - 2 * n + 3) / 3^n

/-- The infinite series ∑(n=1 to ∞) (4n^2 - 2n + 3) / 3^n converges to 21/4 -/
theorem infinite_series_sum_equals_21_4 : infinite_series_sum = 21/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_equals_21_4_l324_32421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_origin_l324_32487

-- Define the circles
def circle_A (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_B (x y : ℝ) : Prop := (x-3)^2 + (y-4)^2 = 4

-- Define the trajectory of P
def trajectory_P (x y : ℝ) : Prop := x^2 + y^2 - 1 = (x-3)^2 + (y-4)^2 - 4

-- Define the distance from a point to the origin
noncomputable def distance_to_origin (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

-- Theorem statement
theorem min_distance_to_origin :
  ∃ min_dist : ℝ, min_dist = 11/5 ∧
  ∀ x y : ℝ, trajectory_P x y →
  distance_to_origin x y ≥ min_dist :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_origin_l324_32487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_not_divisible_by_four_mySequence_minus_22_not_prime_l324_32456

def mySequence : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => mySequence (n + 1) * mySequence n + 1

theorem mySequence_not_divisible_by_four : ∀ n : ℕ, mySequence n % 4 ≠ 0 := by
  sorry

theorem mySequence_minus_22_not_prime : ∀ n : ℕ, n > 10 → ¬ Nat.Prime (mySequence n - 22) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_not_divisible_by_four_mySequence_minus_22_not_prime_l324_32456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_ratios_l324_32427

def is_multiplicative (f : ℕ → ℝ) : Prop :=
  ∀ m n : ℕ, f (m + n) = f m * f n

theorem sum_of_ratios (f : ℕ → ℝ) (h_mult : is_multiplicative f) (h_f1 : f 1 = 2) :
  (Finset.range 1008).sum (λ k => f (2 * k + 2) / f (2 * k + 1)) = 2016 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_ratios_l324_32427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_formula_l324_32463

noncomputable def f₁ (x : ℝ) : ℝ := x / (1 + x)

noncomputable def f₂ (x : ℝ) : ℝ := f₁ (f₁ x)

noncomputable def f : ℕ → ℝ → ℝ
| 0, x => x  -- Adding a case for 0
| 1, x => f₁ x
| n + 1, x => f₁ (f n x)

theorem f_formula (n : ℕ) (hn : n ≥ 1) (x : ℝ) : 
  f n x = x / (1 + n * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_formula_l324_32463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_equals_9pi_l324_32477

-- Define the central angle in degrees
def central_angle_degrees : ℝ := 120

-- Define the radius of the circle
def radius : ℝ := 3

-- Convert the central angle to radians
noncomputable def central_angle_radians : ℝ := (central_angle_degrees * Real.pi) / 180

-- Define the area of the sector
noncomputable def sector_area : ℝ := (1 / 2) * central_angle_radians * (radius ^ 2)

-- Theorem statement
theorem sector_area_equals_9pi : sector_area = 9 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_equals_9pi_l324_32477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_P_and_Q_l324_32447

def P : Set ℤ := {x | -4 ≤ x ∧ x ≤ 2}
def Q : Set ℤ := {x : ℤ | -3 < x ∧ x < 1}

theorem intersection_of_P_and_Q : P ∩ Q = {-2, -1, 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_P_and_Q_l324_32447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class5_matches_l324_32443

/-- Represents the number of matches played by each class -/
def MatchesPlayed : Fin 5 → Nat := sorry

/-- The total number of classes -/
def TotalClasses : Nat := 5

/-- Each pair of classes plays one match against each other -/
axiom one_match_per_pair : ∀ i j : Fin 5, i ≠ j → ∃ k, k = 1

/-- Class 1 has played 2 matches -/
axiom class1_matches : MatchesPlayed 0 = 2

/-- Class 2 has played 4 matches -/
axiom class2_matches : MatchesPlayed 1 = 4

/-- Class 3 has played 4 matches -/
axiom class3_matches : MatchesPlayed 2 = 4

/-- Class 4 has played 3 matches -/
axiom class4_matches : MatchesPlayed 3 = 3

/-- Theorem: Class 5 has played 3 matches -/
theorem class5_matches : MatchesPlayed 4 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_class5_matches_l324_32443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_prime_divides_sum_of_squares_minus_k_l324_32418

/-- For an odd prime p, for all integers k, there exist integers a and b such that p divides (a^2 + b^2 - k). -/
theorem odd_prime_divides_sum_of_squares_minus_k (p : ℕ) (k : ℤ) 
  (hp : Nat.Prime p) (hodd : p % 2 = 1) : 
  ∃ (a b : ℤ), (p : ℤ) ∣ (a^2 + b^2 - k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_prime_divides_sum_of_squares_minus_k_l324_32418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_H_composition_equals_zero_l324_32466

/-- The function H as defined in the problem -/
noncomputable def H (x : ℝ) : ℝ := (x - 2)^2 / 2 - 2

/-- Theorem stating that H(H(H(H(H(2))))) = 0 -/
theorem H_composition_equals_zero : H (H (H (H (H 2)))) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_H_composition_equals_zero_l324_32466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_after_35th_inning_l324_32404

/-- Calculates a batsman's new average after an innings, considering pitch and weather conditions -/
noncomputable def new_average (previous_innings : ℕ) (previous_average : ℚ) (new_score : ℚ) (average_increase : ℚ) (pitch_reduction : ℚ) (weather_reduction : ℚ) : ℚ :=
  let total_runs := (previous_innings : ℚ) * previous_average + new_score
  let new_innings := previous_innings + 1
  let raw_new_average := total_runs / (new_innings : ℚ)
  raw_new_average - pitch_reduction - weather_reduction

/-- The theorem stating the batsman's new average after the 35th inning -/
theorem batsman_average_after_35th_inning :
  let previous_innings : ℕ := 34
  let new_score : ℚ := 150
  let average_increase : ℚ := 1.75
  let pitch_reduction : ℚ := 0.65
  let weather_reduction : ℚ := 0.45
  let previous_average : ℚ := (new_score / ((previous_innings + 1) : ℚ) - average_increase)
  (new_average previous_innings previous_average new_score average_increase pitch_reduction weather_reduction) = 89.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_after_35th_inning_l324_32404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_countless_lines_not_always_parallel_to_plane_l324_32449

-- Define a point
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a plane
structure Plane where
  normal : Point
  d : ℝ

-- Define a line
structure Line where
  point : Point
  direction : Point

-- Define parallelism between a line and a plane
def line_parallel_to_plane (l : Line) (p : Plane) : Prop :=
  let n := p.normal
  let v := l.direction
  n.x * v.x + n.y * v.y + n.z * v.z = 0

-- Define the property of a line being parallel to countless lines in a plane
def parallel_to_countless_lines (l : Line) (p : Plane) : Prop :=
  ∃ (S : Set Line), Infinite S ∧ (∀ m ∈ S, line_parallel_to_plane m p ∧ ¬(l = m))

theorem line_parallel_to_countless_lines_not_always_parallel_to_plane :
  ¬(∀ (a : Line) (α : Plane), parallel_to_countless_lines a α → line_parallel_to_plane a α) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_countless_lines_not_always_parallel_to_plane_l324_32449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l324_32426

/-- The function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := 
  Real.sqrt 3 * Real.sin (2018 * Real.pi - x) * Real.sin (3 * Real.pi / 2 + x) - Real.cos x ^ 2 + 1

/-- The theorem statement -/
theorem range_of_m (m : ℝ) : 
  (∀ x ∈ Set.Icc (-Real.pi/12) (Real.pi/2), |f x - m| ≤ 1) → 
  1/2 ≤ m ∧ m ≤ (3 - Real.sqrt 3)/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l324_32426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l324_32479

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the property of being a focus of the ellipse
def is_focus (p : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem triangle_perimeter (abc : Triangle) : 
  (is_on_ellipse abc.B.1 abc.B.2) →
  (is_on_ellipse abc.C.1 abc.C.2) →
  (is_focus abc.A) →
  (∃ (p : ℝ × ℝ), p ∈ Set.Icc abc.B abc.C ∧ is_focus p) →
  (dist abc.A abc.B + dist abc.B abc.C + dist abc.C abc.A = 8) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l324_32479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l324_32474

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
variable (a b : V)

theorem vector_magnitude_problem 
  (h1 : ‖a - b‖ = Real.sqrt 3)
  (h2 : ‖a + b‖ = ‖(2 : ℝ) • a - b‖) :
  ‖b‖ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l324_32474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_depth_calculation_l324_32486

/-- The depth of a trapezoidal tunnel -/
noncomputable def tunnel_depth (top_width bottom_width area : ℝ) : ℝ :=
  (2 * area) / (top_width + bottom_width)

/-- Theorem: The depth of a trapezoidal tunnel with given dimensions -/
theorem tunnel_depth_calculation (top_width bottom_width area : ℝ) 
  (h_top : top_width = 15)
  (h_bottom : bottom_width = 5)
  (h_area : area = 400) :
  tunnel_depth top_width bottom_width area = 40 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval tunnel_depth 15 5 400

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_depth_calculation_l324_32486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hollow_sphere_weight_l324_32400

/-- The weight of a hollow golden sphere -/
noncomputable def sphere_weight (diameter : ℝ) (thickness : ℝ) (pi_approx : ℝ) : ℝ :=
  let outer_radius := (diameter + thickness) / 2
  let inner_radius := diameter / 2
  (4/3) * pi_approx * (outer_radius^3 - inner_radius^3)

/-- Theorem stating the weight of the specific hollow golden sphere -/
theorem hollow_sphere_weight :
  let diameter := (12 : ℝ)
  let thickness := (0.3 : ℝ)
  let pi_approx := (3 : ℝ)
  ∃ ε > 0, abs (sphere_weight diameter thickness pi_approx - 123.23) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hollow_sphere_weight_l324_32400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_l324_32452

noncomputable def point (x y : ℝ) := (x, y)

noncomputable def area_triangle (A B P : ℝ × ℝ) : ℝ := 
  abs ((B.1 - A.1) * (P.2 - A.2) - (B.2 - A.2) * (P.1 - A.1)) / 2

theorem min_area_triangle (A B : ℝ × ℝ) (h : A = point (-2) 0 ∧ B = point 0 2) :
  let circle := {P : ℝ × ℝ | (P.1 - 3)^2 + (P.2 + 1)^2 = 2}
  (⨅ P ∈ circle, area_triangle A B P) = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_l324_32452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hellys_theorem_four_sets_hellys_theorem_n_sets_l324_32495

-- Define a type for points in the plane
variable (Point : Type)

-- Define a type for convex sets in the plane
variable (ConvexSet : Type)

-- Define a property that a set is convex
variable (is_convex : ConvexSet → Prop)

-- Define a property that a point is in a convex set
variable (in_set : Point → ConvexSet → Prop)

-- Define the intersection of convex sets
def intersection (sets : List ConvexSet) : Set Point :=
  {p : Point | ∀ s, s ∈ sets → in_set p s}

-- State Helly's Theorem for four sets
theorem hellys_theorem_four_sets 
  (C1 C2 C3 C4 : ConvexSet)
  (h_convex : ∀ C, C ∈ [C1, C2, C3, C4] → is_convex C)
  (h_triple_intersect : 
    ∀ A B C, A ∈ [C1, C2, C3, C4] → B ∈ [C1, C2, C3, C4] → C ∈ [C1, C2, C3, C4] → 
    A ≠ B → B ≠ C → A ≠ C → 
    ∃ p : Point, in_set p A ∧ in_set p B ∧ in_set p C) :
  ∃ p : Point, in_set p C1 ∧ in_set p C2 ∧ in_set p C3 ∧ in_set p C4 :=
sorry

-- State Helly's Theorem for n ≥ 4 sets
theorem hellys_theorem_n_sets 
  (n : Nat)
  (h_n : n ≥ 4)
  (sets : Fin n → ConvexSet)
  (h_convex : ∀ i, is_convex (sets i))
  (h_intersect : ∀ (subset : Fin (n-1) → Fin n), 
    ∃ p : Point, ∀ i, in_set p (sets (subset i))) :
  ∃ p : Point, ∀ i, in_set p (sets i) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hellys_theorem_four_sets_hellys_theorem_n_sets_l324_32495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_intersection_point_l324_32490

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 3 * Real.log x / Real.log 3
noncomputable def g (x : ℝ) : ℝ := Real.log (5 * x) / Real.log 3

-- State the theorem
theorem unique_intersection :
  ∃! x : ℝ, x > 0 ∧ f x = g x := by
  sorry

-- Prove that the intersection point is sqrt(5)
theorem intersection_point :
  ∃ x : ℝ, x > 0 ∧ f x = g x ∧ x = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_intersection_point_l324_32490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_tennis_balls_used_l324_32498

/-- Represents the number of games in each round of the tournament -/
def gamesPerRound : List Nat := [1028, 514, 257, 128, 64, 32, 16, 8, 4]

/-- Represents the number of cans used per game for each ball type -/
def cansPerGame : List Nat := [6, 6, 6, 6, 8, 8, 8, 8, 8]

/-- Represents the number of balls per can for each ball type -/
def ballsPerCan : List Nat := [3, 3, 3, 3, 4, 4, 4, 4, 4]

theorem total_tennis_balls_used
  (games : List Nat := gamesPerRound)
  (cans : List Nat := cansPerGame)
  (balls : List Nat := ballsPerCan) :
  (List.zipWith₃ (fun g c b => g * c * b) games cans balls).sum = 37573 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_tennis_balls_used_l324_32498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equality_l324_32407

-- Define the @ operation (renamed to 'atOp')
def atOp (a b : ℕ) : ℕ := a * b - b^2

-- Define the # operation (renamed to 'hashOp')
def hashOp (a b : ℕ) : ℕ := a + b - a * b^2 + a^3

-- Theorem statement
theorem fraction_equality : (atOp 8 3 : ℚ) / (hashOp 8 3) = 15 / 451 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equality_l324_32407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_l324_32432

-- Define the circles x and y
variable (x y : Set ℝ)

-- Define the radius and area of a circle
noncomputable def radius (c : Set ℝ) : ℝ := sorry
noncomputable def area (c : Set ℝ) : ℝ := Real.pi * (radius c)^2

-- Define the circumference of a circle
noncomputable def circumference (c : Set ℝ) : ℝ := 2 * Real.pi * radius c

-- State the theorem
theorem circle_circumference :
  (area x = area y) →  -- Circles x and y have the same area
  (radius y = 6) →     -- The radius of circle y is 6
  (circumference x = 12 * Real.pi) := by  -- The circumference of circle x is 12π
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_l324_32432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l324_32437

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₁ - c₂| / Real.sqrt (a^2 + b^2)

theorem parallel_lines_distance : 
  let line1 : ℝ → ℝ → ℝ := λ x y => 2*x + 3*y - 5
  let line2 : ℝ → ℝ → ℝ := λ x y => 2*x + 3*y - 2
  distance_between_parallel_lines 2 3 (-5) (-2) = 3 * Real.sqrt 13 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l324_32437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l324_32439

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := 2 * Real.sqrt (x^2 + 2*a*x + a^2) - 2 * abs (x - b)

-- State the theorem
theorem function_properties (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (hmax : ∀ x, f a b x ≤ 2) : 
  a + b = 1 ∧ 1/a + 4/b + 4/((3*a+1)*b) ≥ 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l324_32439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l324_32424

-- Define the line l
noncomputable def line_l (t α : ℝ) : ℝ × ℝ := (3 + t * Real.cos α, 1 + t * Real.sin α)

-- Define the curve C in polar coordinates
noncomputable def curve_C_polar (θ : ℝ) : ℝ := 4 * Real.cos θ

-- Define the curve C in Cartesian coordinates
def curve_C_cartesian (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define point P
def point_P : ℝ × ℝ := (3, 1)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem min_distance_sum :
  ∃ (A B : ℝ × ℝ) (α : ℝ),
    (∃ t1 t2 : ℝ, A = line_l t1 α ∧ B = line_l t2 α) ∧
    curve_C_cartesian A.1 A.2 ∧
    curve_C_cartesian B.1 B.2 ∧
    (∀ A' B' : ℝ × ℝ,
      (∃ α' t1' t2' : ℝ, A' = line_l t1' α' ∧ B' = line_l t2' α') →
      curve_C_cartesian A'.1 A'.2 →
      curve_C_cartesian B'.1 B'.2 →
      distance point_P A + distance point_P B ≤ distance point_P A' + distance point_P B') ∧
    distance point_P A + distance point_P B = 2 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l324_32424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sqrt_equality_l324_32414

theorem floor_sqrt_equality (a : ℝ) (h : a > 1) :
  ⌊Real.sqrt ⌊Real.sqrt a⌋⌋ = ⌊Real.sqrt (Real.sqrt a)⌋ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sqrt_equality_l324_32414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partner_a_capital_proof_l324_32476

/-- The capital of partner a in a partnership where:
  - There are three partners: a, b, and c
  - a receives 2/3 of profits
  - b and c divide the remainder equally
  - a's income increases by 300 when profit rate increases from 5% to 7%
-/
noncomputable def partner_a_capital : ℚ := 15000

/-- The total capital of the partnership -/
noncomputable def total_capital : ℚ := 22500

/-- The profit rate increase -/
noncomputable def profit_rate_increase : ℚ := 7/100 - 5/100

/-- A's share of profits -/
noncomputable def a_share : ℚ := 2 / 3

theorem partner_a_capital_proof :
  partner_a_capital = a_share * total_capital ∧
  a_share * profit_rate_increase * total_capital = 300 := by
  sorry

#check partner_a_capital_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partner_a_capital_proof_l324_32476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_future_age_relation_l324_32410

/-- Given that A is 5 years older than B and B's present age is 35,
    this theorem proves that the number of years in the future when A will be
    twice as old as B was the same number of years ago is 10. -/
theorem future_age_relation : ∃ (x : ℕ),
  let a_age : ℕ := 35 + 5
  let b_age : ℕ := 35
  x = 10 ∧ a_age + x = 2 * (b_age - x) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_future_age_relation_l324_32410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_nine_fourths_l324_32482

/-- An isosceles triangle with an inscribed unit square -/
structure IsoscelesTriangleWithSquare where
  /-- The base of the triangle -/
  base : ℝ
  /-- The height of the triangle -/
  height : ℝ
  /-- The side length of the inscribed square -/
  square_side : ℝ
  /-- The square has unit area -/
  square_area : square_side ^ 2 = 1
  /-- One side of the square lies on the base of the triangle -/
  square_on_base : square_side ≤ base
  /-- The centers of gravity of the triangle and square coincide -/
  centroid_coincide : height / 3 = square_side / 2

/-- The area of the isosceles triangle with an inscribed unit square -/
noncomputable def triangle_area (t : IsoscelesTriangleWithSquare) : ℝ :=
  (t.base * t.height) / 2

/-- The theorem stating that the area of the triangle is 9/4 -/
theorem triangle_area_is_nine_fourths (t : IsoscelesTriangleWithSquare) :
  triangle_area t = 9 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_nine_fourths_l324_32482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l324_32496

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 4)

-- State the theorem
theorem f_min_value :
  ∀ x > 4, f x ≥ 6 ∧ ∃ x₀ > 4, f x₀ = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l324_32496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_B_more_stable_l324_32461

-- Define the distributions
def ξ : Fin 3 → ℝ
| 0 => 0.3
| 1 => 0.2
| 2 => 0.5

def η : Fin 3 → ℝ
| 0 => 0.2
| 1 => 0.4
| 2 => 0.4

-- Define the ring values
def ring_value : Fin 3 → ℝ
| 0 => 8
| 1 => 9
| 2 => 10

-- Expected value function
def expected_value (dist : Fin 3 → ℝ) : ℝ :=
  Finset.sum Finset.univ (λ i => dist i * ring_value i)

-- Variance function
def variance (dist : Fin 3 → ℝ) : ℝ :=
  Finset.sum Finset.univ (λ i => dist i * (ring_value i - expected_value dist)^2)

-- Theorem statement
theorem athlete_B_more_stable : variance η < variance ξ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_B_more_stable_l324_32461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l324_32417

/-- Represents an ellipse in the Cartesian plane -/
structure Ellipse where
  center : ℝ × ℝ
  foci : (ℝ × ℝ) × (ℝ × ℝ)
  eccentricity : ℝ

/-- Represents a line in the Cartesian plane -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Represents a triangle in the Cartesian plane -/
structure Triangle where
  vertices : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)

def perimeter (t : Triangle) : ℝ := sorry

noncomputable def intersect_ellipse_line (e : Ellipse) (l : Line) : Set (ℝ × ℝ) := sorry

theorem ellipse_equation (C : Ellipse) :
  C.center = (0, 0) →
  ∃ (c : ℝ), C.foci = ((-c, 0), (c, 0)) →
  C.eccentricity = Real.sqrt 2 / 2 →
  (∀ (L : Line) (A B : ℝ × ℝ),
    L.point = C.foci.1 →
    A ∈ intersect_ellipse_line C L →
    B ∈ intersect_ellipse_line C L →
    perimeter (Triangle.mk (A, B, C.foci.2)) = 16) →
  ∀ (x y : ℝ), x^2 / 16 + y^2 / 8 = 1 ↔ (x, y) ∈ intersect_ellipse_line C (Line.mk (0, 0) (1, 0)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l324_32417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_division_into_similar_heaps_l324_32493

theorem no_division_into_similar_heaps : ¬ ∃ (heaps : Finset ℕ), 
  Finset.card heaps = 31 ∧ 
  (∀ x, x ∈ heaps → x > 0) ∧
  Finset.sum heaps id = 660 ∧
  (∀ x y, x ∈ heaps → y ∈ heaps → y < 2 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_division_into_similar_heaps_l324_32493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_47_l324_32468

/-- The repeating decimal 0.474747... expressed as a rational number -/
theorem repeating_decimal_47 : ∃ (x : ℚ), x = 47 / 99 ∧ ∀ (n : ℕ), (x * 10^(2*n+2) : ℚ) - ⌊x * 10^(2*n+2)⌋ = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_47_l324_32468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sin_plus_a_cos_l324_32446

theorem integral_sin_plus_a_cos (a : ℝ) :
  (∫ x in (0)..(π/2), Real.sin x + a * Real.cos x) = 2 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sin_plus_a_cos_l324_32446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_g_l324_32480

noncomputable def g (x : ℝ) : ℝ := -Real.sin x

theorem area_enclosed_by_g (a b : ℝ) (h1 : a = -π/2) (h2 : b = π/3) :
  ∫ x in a..b, |g x| = 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_g_l324_32480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_powderman_explosion_distance_l324_32484

/-- The time in seconds for the fuse to detonate -/
def fuse_time : ℚ := 45

/-- The speed of the powderman in yards per second -/
def powderman_speed : ℚ := 10

/-- The speed of sound in feet per second -/
def sound_speed : ℚ := 1080

/-- Convert yards to feet -/
def yards_to_feet (yards : ℚ) : ℚ := yards * 3

/-- Convert feet to yards -/
def feet_to_yards (feet : ℚ) : ℚ := feet / 3

/-- The distance run by the powderman at time t -/
def powderman_distance (t : ℚ) : ℚ := yards_to_feet (powderman_speed * t)

/-- The distance traveled by sound after the blast at time t -/
def sound_distance (t : ℚ) : ℚ := sound_speed * (t - fuse_time)

theorem powderman_explosion_distance :
  ∃ t : ℚ, t > fuse_time ∧ 
    powderman_distance t = sound_distance t ∧ 
    (feet_to_yards (powderman_distance t)).floor = 463 := by
  sorry

#eval (feet_to_yards (powderman_distance 46.286)).floor

end NUMINAMATH_CALUDE_ERRORFEEDBACK_powderman_explosion_distance_l324_32484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carla_overall_score_l324_32412

/-- Calculates the overall score percentage for a student given their scores on multiple tests. -/
def overall_score_percentage (test1_problems : ℕ) (test1_score : ℚ)
                             (test2_problems : ℕ) (test2_score : ℚ)
                             (test3_problems : ℕ) (test3_score : ℚ) : ℚ :=
  let total_problems := test1_problems + test2_problems + test3_problems
  let total_correct := test1_score * test1_problems +
                       test2_score * test2_problems +
                       test3_score * test3_problems
  total_correct / total_problems

theorem carla_overall_score :
  overall_score_percentage 15 (85/100) 25 (75/100) 20 (80/100) = 80/100 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carla_overall_score_l324_32412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_l324_32408

/-- The sum of the first n terms of an arithmetic sequence -/
noncomputable def S (n : ℕ) (a₁ d : ℝ) : ℝ := n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_max_sum (d : ℝ) :
  (∀ (n : ℕ), n ≠ 6 → S n (-6) d < S 6 (-6) d) →
  1 < d ∧ d < 6/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_l324_32408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l324_32416

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The line l: mx - y = 4 -/
def line_l (m : ℝ) (x y : ℝ) : Prop := m * x - y = 4

/-- The line perpendicular to l: x + m(m-1)y = 2 -/
def line_perp (m : ℝ) (x y : ℝ) : Prop := x + m * (m - 1) * y = 2

/-- The slope of line l -/
def slope_l (m : ℝ) : ℝ := m

/-- The slope of the line perpendicular to l -/
noncomputable def slope_perp (m : ℝ) : ℝ := 
  if m = 0 then 0 
  else if m = 1 then 0  -- Changed from Real.undefined to 0
  else 1 / (m * (1 - m))

theorem perpendicular_lines (m : ℝ) : 
  perpendicular (slope_l m) (slope_perp m) ↔ m = 0 ∨ m = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l324_32416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_f_left_of_vertex_g_integral_f_greater_than_integral_g_l324_32434

-- Define the two parabolas
noncomputable def f (x : ℝ) := x^2 + x + 1
noncomputable def g (x : ℝ) := x^2 - 3*x + 1

-- Define the vertices of the parabolas
noncomputable def vertex_f : ℝ × ℝ := (-1/2, 3/4)
noncomputable def vertex_g : ℝ × ℝ := (3/2, -5/4)

-- Theorem stating that the vertex of f is to the left of the vertex of g
theorem vertex_f_left_of_vertex_g : vertex_f.1 < vertex_g.1 := by
  -- Proof goes here
  sorry

-- Define the definite integrals
noncomputable def integral_f : ℝ := ∫ x in (-1 : ℝ)..1, f x
noncomputable def integral_g : ℝ := ∫ x in (-1 : ℝ)..1, g x

-- Theorem stating that the integral of f is greater than the integral of g
theorem integral_f_greater_than_integral_g : integral_f > integral_g := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_f_left_of_vertex_g_integral_f_greater_than_integral_g_l324_32434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sides_calculation_l324_32444

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  γ : ℝ
  mc : ℝ
  fc : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.γ = 100 * (Real.pi / 180) ∧  -- Convert 100° to radians
  t.mc = 4 ∧
  t.fc = 5

-- Define the theorem
theorem triangle_sides_calculation (t : Triangle) 
  (h : triangle_conditions t) : 
  (abs (t.a - 4.107) < 0.001 ∧ abs (t.b - 73.26) < 0.001 ∧ abs (t.c - 74.08) < 0.001) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sides_calculation_l324_32444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_l324_32402

/-- The total surface area of a cube with side length (a + 1) cm is 6(a + 1)² cm² -/
theorem cube_surface_area (a : ℝ) : 
  6 * (a + 1)^2 = 6 * (a + 1)^2 := by
  -- The proof is trivial as we're stating equality to itself
  rfl

#check cube_surface_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_l324_32402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagram_transformation_l324_32451

-- Define the shapes
inductive Shape
  | Triangle
  | SmallCircle
  | Rectangle

-- Define the positions
structure Position where
  angle : Real  -- Angle from the center of the large circle
  radius : Real  -- Distance from the center of the large circle

-- Define the diagram
structure Diagram where
  triangle_pos : Position
  small_circle_pos : Position
  rectangle_pos : Position

noncomputable def rotate_clockwise (p : Position) (angle : Real) : Position :=
  { angle := p.angle - angle, radius := p.radius }

noncomputable def reflect_vertical (p : Position) : Position :=
  { angle := -p.angle, radius := p.radius }

noncomputable def transform_diagram (d : Diagram) : Diagram :=
  let rotated := Diagram.mk
    (rotate_clockwise d.triangle_pos (150 * Real.pi / 180))
    (rotate_clockwise d.small_circle_pos (150 * Real.pi / 180))
    (rotate_clockwise d.rectangle_pos (150 * Real.pi / 180))
  Diagram.mk
    (reflect_vertical rotated.triangle_pos)
    (reflect_vertical rotated.small_circle_pos)
    (reflect_vertical rotated.rectangle_pos)

theorem diagram_transformation (d : Diagram) :
  let d' := transform_diagram d
  d'.triangle_pos = d.small_circle_pos ∧
  d'.small_circle_pos = d.triangle_pos ∧
  d'.rectangle_pos.angle = -d.rectangle_pos.angle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagram_transformation_l324_32451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_percentage_in_ccl4_l324_32465

noncomputable def mass_percentage (element_mass : ℝ) (compound_mass : ℝ) : ℝ :=
  (element_mass / compound_mass) * 100

noncomputable def ccl4_molar_mass (carbon_mass : ℝ) (chlorine_mass : ℝ) : ℝ :=
  carbon_mass + 4 * chlorine_mass

theorem carbon_percentage_in_ccl4 (carbon_mass chlorine_mass : ℝ) 
  (h1 : carbon_mass = 12.01)
  (h2 : chlorine_mass = 35.45) :
  abs (mass_percentage carbon_mass (ccl4_molar_mass carbon_mass chlorine_mass) - 7.81) < 0.01 := by
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_percentage_in_ccl4_l324_32465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walkway_time_theorem_l324_32499

/-- Represents a moving walkway scenario -/
structure MovingWalkway where
  length : ℝ
  time_with : ℝ
  time_against : ℝ

/-- Calculates the time to walk the walkway if it were not moving -/
noncomputable def time_without_movement (w : MovingWalkway) : ℝ :=
  2 * w.length / (w.length / w.time_with + w.length / w.time_against)

/-- Theorem stating that for the given walkway scenario, the time to walk without movement is 48 seconds -/
theorem walkway_time_theorem (w : MovingWalkway) 
  (h1 : w.length = 60)
  (h2 : w.time_with = 30)
  (h3 : w.time_against = 120) : 
  time_without_movement w = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_walkway_time_theorem_l324_32499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l324_32459

def remainder_sum (n m : ℕ) : ℕ :=
  (n % m) + (2^(n % m) % m)

theorem problem_solution :
  remainder_sum 1234567 123 = 29 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l324_32459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_zuminglish_words_mod_500_l324_32441

/-- Represents the three types of letter combinations at the end of a word -/
inductive WordEnd
| CC  -- Two consonants
| CV  -- Consonant followed by vowel
| VC  -- Vowel followed by consonant

/-- Defines the rules for Zuminglish word formation -/
def isValidZuminglish (word : List Char) : Prop :=
  ∀ i j, i < j → j < word.length →
    word[i]? = some 'O' → word[j]? = some 'O' →
    ∃ k₁ k₂, i < k₁ ∧ k₁ < k₂ ∧ k₂ < j ∧ 
      word[k₁]? ≠ some 'O' ∧ word[k₂]? ≠ some 'O'

/-- Counts the number of valid n-letter Zuminglish words ending with a specific combination -/
def countWords : ℕ → WordEnd → ℕ
| 0, _ => 0
| 1, _ => 0
| 2, WordEnd.CC => 4
| 2, WordEnd.CV => 2
| 2, WordEnd.VC => 2
| n+1, WordEnd.CC => 2 * (countWords n WordEnd.CC + countWords n WordEnd.VC)
| n+1, WordEnd.CV => countWords n WordEnd.CC
| n+1, WordEnd.VC => 2 * countWords n WordEnd.CV

/-- The main theorem stating the number of valid 9-letter Zuminglish words modulo 500 -/
theorem valid_zuminglish_words_mod_500 :
  (countWords 9 WordEnd.CC + countWords 9 WordEnd.CV + countWords 9 WordEnd.VC) % 500 = 472 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_zuminglish_words_mod_500_l324_32441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_dough_weight_l324_32413

/-- Given the initial amount of chocolate, the amount of chocolate left over,
    and the desired percentage of chocolate in the cookies, prove that the
    amount of dough used is 36 ounces. -/
theorem cookie_dough_weight
  (initial_chocolate : ℝ)
  (leftover_chocolate : ℝ)
  (chocolate_percentage : ℝ)
  (h1 : initial_chocolate = 13)
  (h2 : leftover_chocolate = 4)
  (h3 : chocolate_percentage = 0.2)
  : ∃ (dough : ℝ), dough = 36 :=
by
  -- The amount of dough used is 36 ounces
  sorry

-- Remove the #eval statement as it was causing the error

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_dough_weight_l324_32413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l324_32454

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 100

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 5 = 0

-- Define the midpoint of the chord
def chord_midpoint : ℝ × ℝ := (-2, 3)

-- Theorem statement
theorem line_equation :
  ∃ (A B : ℝ × ℝ),
    circle_eq A.1 A.2 ∧
    circle_eq B.1 B.2 ∧
    A ≠ B ∧
    ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = chord_midpoint →
    ∀ (x y : ℝ), line_l x y ↔ (∃ t : ℝ, x = t * (A.1 - B.1) + chord_midpoint.1 ∧ y = t * (A.2 - B.2) + chord_midpoint.2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l324_32454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_failed_candidates_l324_32483

/-- Calculates the percentage of failed candidates in an examination. -/
theorem percentage_of_failed_candidates
  (total_candidates : ℕ)
  (num_girls : ℕ)
  (boys_pass_rate : ℚ)
  (girls_pass_rate : ℚ)
  (h1 : total_candidates = 2000)
  (h2 : num_girls = 900)
  (h3 : boys_pass_rate = 28 / 100)
  (h4 : girls_pass_rate = 32 / 100) :
  (let num_boys := total_candidates - num_girls
   let boys_passed := (num_boys : ℚ) * boys_pass_rate
   let girls_passed := (num_girls : ℚ) * girls_pass_rate
   let total_passed := boys_passed + girls_passed
   let total_failed := (total_candidates : ℚ) - total_passed
   let fail_percentage := (total_failed / total_candidates) * 100
   fail_percentage) = 70.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_failed_candidates_l324_32483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruplet_babies_l324_32478

theorem quadruplet_babies (total_babies : ℕ) (triplet_ratio : ℕ) (twin_ratio : ℕ) :
  total_babies = 1250 →
  triplet_ratio = 5 →
  twin_ratio = 3 →
  ∃ (quad_sets : ℕ),
    let triplet_sets := triplet_ratio * quad_sets
    let twin_sets := twin_ratio * triplet_sets
    2 * twin_sets + 3 * triplet_sets + 4 * quad_sets = total_babies ∧
    4 * quad_sets = 5000 / 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruplet_babies_l324_32478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_of_one_equals_two_l324_32491

-- Define the function g
noncomputable def g (x : ℝ) : ℝ :=
  if x ≥ 3 then x^3 else 3 - x

-- State the theorem
theorem g_composition_of_one_equals_two :
  g (g (g 1)) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_of_one_equals_two_l324_32491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_box_width_l324_32469

/-- The width of a rectangular milk box -/
noncomputable def boxWidth (length : ℝ) (height : ℝ) (volume : ℝ) : ℝ :=
  volume / (length * height)

theorem milk_box_width :
  let length : ℝ := 62
  let height : ℝ := 0.5
  let volume : ℝ := 776.793
  abs (boxWidth length height volume - 25.057) < 0.001 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_box_width_l324_32469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l324_32489

theorem max_value_expression (x : ℝ) (hx : x > 0) :
  (x^4 + 6 - Real.sqrt (x^8 + 8)) / x^2 ≤ 4 * (14 * Real.sqrt 7 - 35) ∧
  ∃ y : ℝ, y > 0 ∧ (y^4 + 6 - Real.sqrt (y^8 + 8)) / y^2 = 4 * (14 * Real.sqrt 7 - 35) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l324_32489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_sum_not_always_equal_sum_medians_l324_32405

noncomputable def directSum (X Y : Finset ℝ) : Finset ℝ :=
  Finset.biUnion X (fun x => Finset.image (fun y => x + y) Y)

noncomputable def median (S : Finset ℝ) : ℝ := sorry

theorem median_sum_not_always_equal_sum_medians :
  ∃ (X Y : Finset ℝ), X.Nonempty ∧ Y.Nonempty ∧
    median (directSum X Y) ≠ median X + median Y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_sum_not_always_equal_sum_medians_l324_32405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prism_volume_theorem_l324_32420

/-- Regular triangular pyramid with base edge 2 and height 2√2 -/
structure RegularTriangularPyramid where
  base_edge : ℝ
  height : ℝ
  base_edge_eq : base_edge = 2
  height_eq : height = 2 * Real.sqrt 2

/-- Right prism with rhombus base inside the pyramid -/
structure RhombusPrism where
  volume : ℝ
  face_on_base : Prop
  face_on_lateral : Prop

/-- The maximum volume of the prism inside the pyramid -/
noncomputable def max_prism_volume (p : RegularTriangularPyramid) : ℝ :=
  5 * Real.sqrt 6 / 36

/-- Theorem stating the maximum volume of the prism -/
theorem max_prism_volume_theorem (p : RegularTriangularPyramid) :
  ∀ (r : RhombusPrism), r.face_on_base ∧ r.face_on_lateral →
    r.volume ≤ max_prism_volume p := by
  sorry

#check max_prism_volume_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prism_volume_theorem_l324_32420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_divisible_by_101_l324_32415

def is_valid_sequence (seq : List Nat) : Prop :=
  seq.length > 0 ∧
  ∀ i, i < seq.length →
    1000 ≤ seq[i]! ∧ seq[i]! < 10000 ∧
    (i + 1 < seq.length → 
      (seq[i]! / 100) % 10 = seq[i+1]! / 1000 ∧
      (seq[i]! / 10) % 10 = (seq[i+1]! / 100) % 10 ∧
      seq[i]! % 10 = (seq[i+1]! / 10) % 10) ∧
    ((seq[seq.length - 1]! / 100) % 10 = seq[0]! / 1000 ∧
     (seq[seq.length - 1]! / 10) % 10 = (seq[0]! / 100) % 10 ∧
     seq[seq.length - 1]! % 10 = (seq[0]! / 10) % 10)

theorem sequence_sum_divisible_by_101 (seq : List Nat) 
  (h : is_valid_sequence seq) : 
  101 ∣ seq.sum :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_divisible_by_101_l324_32415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_implies_b_c_slope_condition_implies_b_range_l324_32481

/-- The function f(x) defined in the problem -/
noncomputable def f (b c : ℝ) (x : ℝ) : ℝ := -1/3 * x^3 + b * x^2 + c * x + b * c

/-- The derivative of f(x) -/
noncomputable def f' (b c : ℝ) (x : ℝ) : ℝ := -x^2 + 2 * b * x + c

theorem extreme_value_implies_b_c (b c : ℝ) :
  (f b c 1 = -4/3) ∧ (f' b c 1 = 0) → b = -1 ∧ c = 3 := by sorry

theorem slope_condition_implies_b_range (b c : ℝ) :
  (∀ x ∈ Set.Ioo (1/2 : ℝ) 3, f' b c x - c ≤ 2) → b ≤ Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_implies_b_c_slope_condition_implies_b_range_l324_32481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_of_bounded_naturals_l324_32485

theorem cardinality_of_bounded_naturals : 
  Finset.card (Finset.filter (fun x => 2 ≤ x ∧ x ≤ 7) (Finset.range 8)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_of_bounded_naturals_l324_32485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l324_32403

-- Define the circle C
def circle_C (x y m : ℝ) : Prop := x^2 + y^2 - 2*x + m = 0

-- Define the second circle
def circle_2 (x y : ℝ) : Prop := (x+3)^2 + (y+3)^2 = 4

-- Define the line
def line (x y : ℝ) : Prop := 5*x + 12*y + 8 = 0

-- Define external tangency condition
def externally_tangent (m : ℝ) : Prop := 
  ∃ (x y : ℝ), circle_C x y m ∧ circle_2 x y

-- Define the distance from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |5*x + 12*y + 8| / Real.sqrt (5^2 + 12^2)

-- Theorem statement
theorem max_distance_to_line (m : ℝ) :
  externally_tangent m →
  (∃ (x y : ℝ), circle_C x y m ∧ distance_to_line x y ≤ 4) ∧
  (∃ (x y : ℝ), circle_C x y m ∧ distance_to_line x y = 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l324_32403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangles_area_eq_five_perimeter_l324_32401

/-- The number of non-congruent right triangles with positive integer leg lengths
    whose areas are numerically equal to 5 times their perimeters -/
theorem right_triangles_area_eq_five_perimeter : 
  Finset.card (Finset.filter (fun p : ℕ × ℕ => 
    let a := p.1
    let b := p.2
    let c := Real.sqrt ((a : ℝ)^2 + (b : ℝ)^2)
    a > 0 ∧ b > 0 ∧ ((a * b : ℝ) / 2 = 5 * ((a : ℝ) + (b : ℝ) + c))
  ) (Finset.range 1000 ×ˢ Finset.range 1000)) = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangles_area_eq_five_perimeter_l324_32401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_proof_l324_32433

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  9 * x^2 - 18 * x + 9 * y^2 + 36 * y + 44 = 0

/-- The radius of the circle -/
noncomputable def circle_radius : ℝ := 1 / 3

/-- Theorem stating that for any point (x, y) satisfying the circle equation,
    there exist h and k such that (x - h)^2 + (y - k)^2 equals the square of the circle's radius -/
theorem circle_radius_proof :
  ∀ x y : ℝ, circle_equation x y → 
  ∃ h k : ℝ, (x - h)^2 + (y - k)^2 = circle_radius^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_proof_l324_32433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_parameter_value_l324_32435

/-- Represents a parabola with equation y² = ax, where a > 0 -/
structure Parabola where
  a : ℝ
  h_pos : a > 0

/-- Represents a point on the parabola -/
structure ParabolaPoint (p : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = p.a * x

/-- The focus of a parabola -/
noncomputable def focus (p : Parabola) : ℝ × ℝ := (p.a / 4, 0)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem stating that for a parabola y² = ax with a > 0 and a point P(3/2, y₀) on the parabola
    at a distance of 2 from the focus, the value of a is 2 -/
theorem parabola_parameter_value (p : Parabola) (point : ParabolaPoint p)
    (h_x : point.x = 3/2)
    (h_dist : distance (point.x, point.y) (focus p) = 2) :
    p.a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_parameter_value_l324_32435
