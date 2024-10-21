import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2009_equals_4_l222_22206

-- Define the function f
noncomputable def f (x α β : ℝ) : ℝ := Real.sin (Real.pi * x + α) + Real.cos (Real.pi * x + β) + 3

-- State the theorem
theorem f_2009_equals_4 (α β : ℝ) (h : f 2008 α β = 2) : f 2009 α β = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2009_equals_4_l222_22206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lucius_earnings_l222_22285

/-- Represents the daily sales distribution for Lucius's business -/
inductive SalesDay
| MWF -- Monday, Wednesday, Friday
| TT  -- Tuesday, Thursday

/-- Represents the product types sold by Lucius -/
inductive Product
| FrenchFries
| Poutine
| OnionRings

/-- Lucius's small business model -/
structure Business where
  frenchFriesPrice : ℚ
  poutinePrice : ℚ
  onionRingsPrice : ℚ
  minIngredientCost : ℚ
  maxIngredientCost : ℚ
  weeklyFrenchFries : ℕ
  weeklyPoutine : ℕ
  weeklyOnionRings : ℕ
  taxRate : ℚ

/-- Calculate the total sales for a given product -/
def totalSales (b : Business) (p : Product) : ℚ :=
  match p with
  | Product.FrenchFries => b.frenchFriesPrice * b.weeklyFrenchFries
  | Product.Poutine => b.poutinePrice * b.weeklyPoutine
  | Product.OnionRings => b.onionRingsPrice * b.weeklyOnionRings

/-- Calculate the total weekly sales -/
def weeklySales (b : Business) : ℚ :=
  totalSales b Product.FrenchFries + totalSales b Product.Poutine + totalSales b Product.OnionRings

/-- Calculate the average daily ingredient cost -/
def avgIngredientCost (b : Business) : ℚ :=
  (b.minIngredientCost + b.maxIngredientCost) / 2

/-- Calculate the total weekly ingredient cost -/
def weeklyIngredientCost (b : Business) : ℚ :=
  avgIngredientCost b * 7

/-- Calculate the weekly tax -/
def weeklyTax (b : Business) : ℚ :=
  b.taxRate * weeklySales b

/-- Calculate the total earnings after taxes and costs -/
def totalEarnings (b : Business) : ℚ :=
  weeklySales b - weeklyTax b - weeklyIngredientCost b

/-- Theorem stating Lucius's total earnings -/
theorem lucius_earnings (b : Business) 
  (h1 : b.frenchFriesPrice = 12)
  (h2 : b.poutinePrice = 8)
  (h3 : b.onionRingsPrice = 6)
  (h4 : b.minIngredientCost = 8)
  (h5 : b.maxIngredientCost = 15)
  (h6 : b.weeklyFrenchFries = 75)
  (h7 : b.weeklyPoutine = 50)
  (h8 : b.weeklyOnionRings = 60)
  (h9 : b.taxRate = 1/10)
  : totalEarnings b = 2827/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lucius_earnings_l222_22285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_subset_l222_22210

/-- A function that checks if a subset of natural numbers contains an arithmetic progression of a given length -/
def containsArithmeticProgression (s : Finset ℕ) (length : ℕ) : Prop :=
  ∃ a d : ℕ, ∀ k : ℕ, k < length → (a + k * d) ∈ s

theorem arithmetic_progression_subset (n : ℕ) :
  (∀ s : Finset ℕ, s ⊆ Finset.range 1989 → s.card = n →
    containsArithmeticProgression s 29) →
  n > 1850 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_subset_l222_22210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_six_digit_common_remainder_l222_22258

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def same_remainder (n r : ℕ) : Prop :=
  n % 4 = r ∧ n % 610 = r ∧ n % 15 = r

def sum_of_digits (n : ℕ) : ℕ :=
  (n.repr.toList.map (fun c => c.toString.toNat!)).sum

theorem least_six_digit_common_remainder :
  ∃ (n : ℕ), is_six_digit n ∧
             (∀ m, is_six_digit m → m < n → ¬(∃ r, same_remainder m r)) ∧
             (∃ r, same_remainder n r) ∧
             sum_of_digits n = 5 →
  ∃ r, same_remainder n r ∧ r = 0 := by
  sorry

#check least_six_digit_common_remainder

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_six_digit_common_remainder_l222_22258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_equation_solution_l222_22236

def base_representation (n : ℕ) (b : ℕ) : ℕ → ℕ
  | 0 => 0
  | (k+1) => (n % b) + b * base_representation (n / b) b k

theorem base_equation_solution :
  ∃! b : ℕ, b > 1 ∧ 
    base_representation 142 b 3 + base_representation 243 b 3 = base_representation 405 b 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_equation_solution_l222_22236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_tiling_divisibility_equivalence_l222_22220

theorem box_tiling_divisibility_equivalence
  (n : ℕ) (b : Fin n → ℕ) (h_pos : ∀ i, 0 < b i) (h_sorted : ∀ i j, i ≤ j → b i ≤ b j) :
  ∀ k : Fin n,
    (∀ i j : Fin n, i ≤ j → j ≤ k → b i ∣ b j) ↔
    (∀ (B : Fin n → ℕ), (∃ (t : Fin n → ℕ), ∀ i : Fin n, i ≤ k → B i = t i * b i) →
      ∀ i : Fin n, i ≤ k → b i ∣ B i) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_tiling_divisibility_equivalence_l222_22220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_l222_22262

-- Define g₀(x)
def g₀ (x : ℝ) : ℝ := x + |x - 150| - |x + 150|

-- Define gₙ(x) for n ≥ 1
def g (n : ℕ) (x : ℝ) : ℝ := 
  match n with
  | 0 => g₀ x
  | n + 1 => |g n x| - 1

-- Define the set of solutions for g₁₀₀(x) = 0
def solutions : Set ℝ := {x | g 100 x = 0}

-- Theorem statement
theorem count_solutions : ∃ (S : Finset ℝ), S.card = 603 ∧ ∀ x, x ∈ S ↔ g 100 x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_l222_22262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_modulus_for_all_residues_l222_22201

/-- Represents the sum of the first x positive integers -/
def triangular_sum (x : ℕ) : ℕ := x * (x + 1) / 2

/-- 
Theorem: The smallest positive integer n such that for any integer a, 
there exists an x where triangular_sum x ≡ a (mod n) is 2.
-/
theorem smallest_modulus_for_all_residues : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (a : ℤ), ∃ (x : ℕ), (triangular_sum x : ℤ) % n = a % n) ∧
  (∀ (m : ℕ), m > 0 → m < n → 
    ∃ (a : ℤ), ∀ (x : ℕ), (triangular_sum x : ℤ) % m ≠ a % m) ∧
  n = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_modulus_for_all_residues_l222_22201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_distance_speed_l222_22268

-- Define the given parameters
noncomputable def total_distance : ℝ := 250
noncomputable def partial_distance : ℝ := 160
noncomputable def partial_speed : ℝ := 40
noncomputable def total_time : ℝ := 5.5

-- Define the function to calculate average speed
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

-- Theorem statement
theorem remaining_distance_speed : 
  let remaining_distance := total_distance - partial_distance
  let partial_time := partial_distance / partial_speed
  let remaining_time := total_time - partial_time
  average_speed remaining_distance remaining_time = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_distance_speed_l222_22268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_puzzle_solutions_l222_22232

theorem digit_puzzle_solutions :
  {(A, B) : ℕ × ℕ | 
    A ≠ B ∧
    A ≥ 1 ∧ A ≤ 9 ∧
    B ≥ 1 ∧ B ≤ 9 ∧
    A^B = 10*B + A ∧
    10*B + A ≠ B*A} =
  {(2, 5), (6, 2), (4, 3)} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_puzzle_solutions_l222_22232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_value_l222_22277

theorem sin_2x_value (x : ℝ) 
  (h : 2 * Real.sin x + Real.cos x + Real.tan x + (Real.cos x / Real.sin x) + (1 / Real.cos x) + (1 / Real.sin x) = 9) : 
  Real.sin (2 * x) = 38 - 2 * Real.sqrt 361 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_value_l222_22277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_OPAB_l222_22202

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  m : ℝ
  h : m > 0

/-- The equation of the ellipse -/
def Ellipse.equation (e : Ellipse) (p : Point) : Prop :=
  e.m * p.x^2 + 3 * e.m * p.y^2 = 1

/-- The major axis length of the ellipse -/
noncomputable def Ellipse.majorAxisLength (e : Ellipse) : ℝ := 2 * Real.sqrt 6

/-- The origin point -/
def O : Point := ⟨0, 0⟩

/-- Point A -/
def A : Point := ⟨3, 0⟩

/-- Predicate to check if a point is on the y-axis -/
def isOnYAxis (p : Point) : Prop := p.x = 0

/-- Predicate to check if a point is on the right side of the y-axis -/
def isOnRightSideOfYAxis (p : Point) : Prop := p.x > 0

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Area of a quadrilateral given its four vertices -/
noncomputable def quadrilateralArea (p1 p2 p3 p4 : Point) : ℝ :=
  sorry  -- The actual calculation would go here

/-- The main theorem -/
theorem min_area_OPAB (e : Ellipse) (B P : Point) 
    (h1 : e.majorAxisLength = 2 * Real.sqrt 6)
    (h2 : isOnYAxis B)
    (h3 : e.equation P)
    (h4 : isOnRightSideOfYAxis P)
    (h5 : distance B A = distance B P) :
    ∃ (minArea : ℝ), minArea = 3 * Real.sqrt 3 ∧ 
    ∀ (Q : Point), e.equation Q → isOnRightSideOfYAxis Q → 
    quadrilateralArea O P A B ≥ minArea := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_OPAB_l222_22202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_person_a_earth_weight_approx_l222_22280

/-- The weight ratio between earth and moon -/
noncomputable def moon_earth_ratio : ℝ := 27.2 / 136

/-- The weight of Person A on the moon in pounds -/
def person_a_moon_weight : ℝ := 26.6

/-- The weight of Person B on earth in pounds -/
def person_b_earth_weight : ℝ := 136

/-- The weight of Person B on the moon in pounds -/
def person_b_moon_weight : ℝ := 27.2

/-- Calculates the earth weight given the moon weight -/
noncomputable def earth_weight (moon_weight : ℝ) : ℝ := moon_weight / moon_earth_ratio

theorem person_a_earth_weight_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |earth_weight person_a_moon_weight - 132.43| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_person_a_earth_weight_approx_l222_22280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l222_22234

/-- Represents a right circular cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

/-- Calculates the volume of a sphere -/
noncomputable def sphereVolume (s : Sphere) : ℝ := (4/3) * Real.pi * s.radius^3

/-- Calculates the rise in liquid level when a sphere is submerged in a cone -/
noncomputable def liquidRise (c : Cone) (s : Sphere) : ℝ :=
  sphereVolume s / (Real.pi * c.radius^2)

theorem liquid_rise_ratio :
  let c1 : Cone := { radius := 4, height := 1 }
  let c2 : Cone := { radius := 8, height := 1 }
  let s : Sphere := { radius := 2 }
  liquidRise c1 s / liquidRise c2 s = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l222_22234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_regular_quadrilateral_pyramid_l222_22253

/-- The cosine of the angle between adjacent lateral faces of a regular quadrilateral pyramid -/
noncomputable def cos_angle_between_faces (a : ℝ) : ℝ :=
  let ao := a * Real.sqrt 3 / 2
  let ac := a * Real.sqrt 2
  (2 * ao^2 - ac^2) / (2 * ao^2)

/-- Theorem: The cosine of the angle between adjacent lateral faces of a regular quadrilateral pyramid,
    where the lateral edge is equal to the side of the base, is equal to -1/3 -/
theorem cos_angle_regular_quadrilateral_pyramid :
  ∀ a : ℝ, a > 0 → cos_angle_between_faces a = -1/3 :=
by
  intro a ha
  unfold cos_angle_between_faces
  -- The proof steps would go here, but for now we'll use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_regular_quadrilateral_pyramid_l222_22253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_spheres_from_large_sphere_l222_22273

/-- The volume of a sphere given its radius -/
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The number of small spheres that can be made from a large sphere -/
noncomputable def numSmallSpheres (largeDiameter smallDiameter : ℝ) : ℝ :=
  sphereVolume (largeDiameter / 2) / sphereVolume (smallDiameter / 2)

theorem small_spheres_from_large_sphere :
  numSmallSpheres 8 2 = 64 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_spheres_from_large_sphere_l222_22273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_sphere_volume_ratio_l222_22292

theorem cylinder_sphere_volume_ratio : 
  ∀ (R : ℝ), R > 0 →
  (2 * Real.pi * R^3) / ((4 / 3) * Real.pi * R^3) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_sphere_volume_ratio_l222_22292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_period_is_pi_l222_22294

noncomputable def f (x : ℝ) : ℝ := -Real.cos (2 * x)

noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi/4)

theorem g_period_is_pi : ∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), g (x + p) = g x ∧ ∀ (q : ℝ), 0 < q ∧ q < p → ∃ (y : ℝ), g (y + q) ≠ g y :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_period_is_pi_l222_22294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mothers_with_full_time_jobs_l222_22219

theorem mothers_with_full_time_jobs 
  (percent_women : ℝ) 
  (percent_fathers_full_time : ℝ) 
  (percent_without_full_time : ℝ) 
  (h1 : percent_women = 0.6)
  (h2 : percent_fathers_full_time = 3/4)
  (h3 : percent_without_full_time = 0.2)
  : (1 - percent_without_full_time - (1 - percent_women) * percent_fathers_full_time) / percent_women = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mothers_with_full_time_jobs_l222_22219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l222_22281

noncomputable def hyperbola (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + (y + 3)^2) - Real.sqrt ((x - 8)^2 + (y + 3)^2) = 4

noncomputable def focal_distance : ℝ := 6

noncomputable def semi_major_axis : ℝ := 2

noncomputable def semi_minor_axis : ℝ := Real.sqrt 5

theorem hyperbola_asymptote_slope :
  ∃ (slope : ℝ), slope > 0 ∧
    (∀ x y : ℝ, hyperbola x y → slope = semi_minor_axis / semi_major_axis) ∧
    slope = Real.sqrt 5 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l222_22281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_whitewashing_cost_is_six_l222_22298

/-- The cost per square foot of whitewashing a room -/
noncomputable def whitewashing_cost_per_sq_ft (room_length room_width room_height : ℝ)
  (door_width door_height : ℝ) (window_width window_height : ℝ)
  (num_doors num_windows : ℕ) (total_cost : ℝ) : ℝ :=
  let wall_area := 2 * (room_length * room_height + room_width * room_height)
  let door_area := door_width * door_height
  let window_area := window_width * window_height
  let total_openings_area := (num_doors : ℝ) * door_area + (num_windows : ℝ) * window_area
  let whitewash_area := wall_area - total_openings_area
  total_cost / whitewash_area

/-- Theorem: The cost per square foot of whitewashing the room is 6 Rs. -/
theorem whitewashing_cost_is_six :
  whitewashing_cost_per_sq_ft 25 15 12 6 3 4 3 1 3 5436 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_whitewashing_cost_is_six_l222_22298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_angle_l222_22249

-- Define the vectors a and b
noncomputable def a (θ : Real) : Fin 2 → Real := fun i => 
  if i = 0 then 1 - Real.sin θ else 1

noncomputable def b (θ : Real) : Fin 2 → Real := fun i => 
  if i = 0 then 1/2 else 1 + Real.sin θ

-- Define the parallel condition
def parallel (u v : Fin 2 → Real) : Prop :=
  ∃ (k : Real), k ≠ 0 ∧ ∀ i, u i = k * v i

-- State the theorem
theorem parallel_vectors_angle (θ : Real) 
  (h_acute : 0 < θ ∧ θ < Real.pi/2) 
  (h_parallel : parallel (a θ) (b θ)) : 
  θ = Real.pi/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_angle_l222_22249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_l222_22282

theorem parallel_vectors (x : ℝ) :
  let a : ℝ × ℝ := (x, 2)
  let b : ℝ × ℝ := (3, x - 1)
  (∃ (k : ℝ), a = k • b) ↔ x = 3 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_l222_22282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_pass_count_l222_22299

/-- The number of times two runners pass each other on a circular track -/
def pass_count (
  duration : ℝ
) (inner_radius : ℝ
) (outer_radius : ℝ
) (inner_speed : ℝ
) (outer_speed : ℝ
) : ℕ :=
  -- The actual implementation is not provided here
  sorry

/-- Theorem: Under the given conditions, the runners pass each other 72 times -/
theorem runners_pass_count : pass_count 45 40 55 200 280 = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_pass_count_l222_22299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_specific_gravity_l222_22254

/-- Represents a homogeneous sphere. -/
structure HomogeneousSphere where
  weight : ℝ
  volume : ℝ

/-- The apparent weight of a sphere when fully submerged in water. -/
noncomputable def fullySubmergedWeight (s : HomogeneousSphere) : ℝ :=
  s.weight - s.volume

/-- The apparent weight of a sphere when half submerged in water. -/
noncomputable def halfSubmergedWeight (s : HomogeneousSphere) : ℝ :=
  s.weight - s.volume / 2

/-- The specific gravity of a sphere. -/
noncomputable def specificGravity (s : HomogeneousSphere) : ℝ :=
  s.weight / s.volume

/-- 
Theorem: If a homogeneous sphere's weight when half submerged in water 
is twice its weight when fully submerged, then its specific gravity is 5/2.
-/
theorem sphere_specific_gravity 
  (s : HomogeneousSphere) 
  (h : halfSubmergedWeight s = 2 * fullySubmergedWeight s) : 
  specificGravity s = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_specific_gravity_l222_22254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_family_probability_l222_22276

theorem family_probability (p_boy : ℝ) (h_prob : p_boy = 1 / 2) :
  1 - (p_boy^4 + (1 - p_boy)^4) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_family_probability_l222_22276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_price_l222_22274

/-- The original price of a tub of ice cream -/
noncomputable def original_price : ℚ := 12

/-- The sale price of a tub of ice cream -/
noncomputable def sale_price (x : ℚ) : ℚ := x - 2

/-- The price of one can of juice -/
noncomputable def juice_price : ℚ := 2 / 5

theorem ice_cream_price :
  ∀ x : ℚ,
  sale_price x = x - 2 →
  juice_price = 2 / 5 →
  2 * sale_price x + 10 * juice_price = 24 →
  x = original_price :=
by
  intro x h1 h2 h3
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_price_l222_22274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_number_l222_22207

-- Helper function to swap digits (declared but not implemented)
noncomputable def swap_digits (n d1 d2 pos1 pos2 : ℕ) : ℕ := sorry

theorem existence_of_special_number :
  ∃ (n : ℕ), 
    n > 10^1000 ∧ 
    ¬(10 ∣ n) ∧
    ∃ (d1 d2 : ℕ) (pos1 pos2 : ℕ), 
      d1 ≠ d2 ∧ 
      d1 ≠ 0 ∧ 
      d2 ≠ 0 ∧
      pos1 ≠ pos2 ∧
      (∀ (p : ℕ), Nat.Prime p → (p ∣ n ↔ p ∣ (swap_digits n d1 d2 pos1 pos2))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_number_l222_22207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangular_pyramid_volume_l222_22286

-- Define the right triangular pyramid ABCD
structure RightTriangularPyramid where
  x : ℝ
  h : ℝ
  x_pos : x > 0
  h_pos : h > 0

-- Define the volume function for the pyramid
noncomputable def volume (p : RightTriangularPyramid) : ℝ :=
  (1 / 6) * p.x^2 * p.h

-- Theorem statement
theorem right_triangular_pyramid_volume (p : RightTriangularPyramid) :
  volume p = (1 / 6) * p.x^2 * p.h :=
by
  -- Unfold the definition of volume
  unfold volume
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangular_pyramid_volume_l222_22286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_AD_length_l222_22261

/-- Represents a trapezoid ABCD with a point M on CD and perpendicular AH from A to BM -/
structure Trapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  M : ℝ × ℝ
  H : ℝ × ℝ
  AD_parallel_BC : (A.2 - D.2) / (A.1 - D.1) = (B.2 - C.2) / (B.1 - C.1)
  M_on_CD : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • C + t • D
  AH_perpendicular_BM : (A.1 - H.1) * (B.1 - M.1) + (A.2 - H.2) * (B.2 - M.2) = 0
  AD_eq_HD : dist A D = dist H D
  BC_length : dist B C = 16
  CM_length : dist C M = 8
  MD_length : dist M D = 9

/-- The length of AD in the trapezoid is 18 -/
theorem trapezoid_AD_length (t : Trapezoid) : dist t.A t.D = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_AD_length_l222_22261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l222_22209

/-- Given a triangle with sides in arithmetic progression with common difference 2
    and the sine of its largest angle being √3/2, its area is 15√3/4 -/
theorem triangle_area (a b c : ℝ) (h_arithmetic : b - a = c - b ∧ b - a = 2) 
  (h_sine : Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))) = Real.sqrt 3 / 2) : 
  (1/2) * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))) = 15 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l222_22209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l222_22214

-- Part I
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (x/4) * Real.cos (x/4) + Real.cos (x/4)^2

theorem part_one (x : ℝ) (h : f x = 1) : Real.cos (x + π/3) = 1/2 := by
  sorry

-- Part II
theorem part_two (A B C a b c : ℝ) 
  (h1 : 0 < A ∧ A < π/2) 
  (h2 : 0 < B ∧ B < π/2) 
  (h3 : 0 < C ∧ C < π/2)
  (h4 : A + B + C = π)
  (h5 : (2*a - c) * Real.cos B = b * Real.cos C) :
  (Real.sqrt 3 + 1)/2 < Real.sin (A + π/6) + 1/2 ∧ Real.sin (A + π/6) + 1/2 ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l222_22214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_max_sum_of_distances_l222_22248

noncomputable section

/-- Line l in polar coordinates -/
def line_l (θ : ℝ) : ℝ := 2 / (Real.sin θ + Real.cos θ)

/-- Curve C in polar coordinates -/
def curve_C (m θ : ℝ) : ℝ := 2 * m * Real.cos θ

/-- Condition for m to be positive -/
def m_positive (m : ℝ) : Prop := m > 0

/-- Condition for l and C to have a common point -/
def common_point (m : ℝ) : Prop :=
  ∃ θ : ℝ, line_l θ = curve_C m θ

/-- Theorem for the range of m when l and C have a common point -/
theorem range_of_m (m : ℝ) (h : m_positive m) :
  common_point m ↔ m ≥ 2 * Real.sqrt 2 - 2 :=
sorry

/-- Function to calculate |OA| + |OB| -/
def sum_of_distances (m θ : ℝ) : ℝ :=
  curve_C m θ + curve_C m (θ + Real.pi / 4)

/-- Theorem for the maximum value of |OA| + |OB| -/
theorem max_sum_of_distances (m : ℝ) (h : m_positive m) :
  (∀ θ : ℝ, sum_of_distances m θ ≤ 2 * Real.sqrt (2 + Real.sqrt 2) * m) ∧
  (∃ θ : ℝ, sum_of_distances m θ = 2 * Real.sqrt (2 + Real.sqrt 2) * m) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_max_sum_of_distances_l222_22248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_root_exists_and_floor_is_three_l222_22238

noncomputable def g (x : ℝ) := Real.sin x - Real.cos x + 4 * Real.tan x

theorem g_root_exists_and_floor_is_three :
  ∃ s : ℝ, π < s ∧ s < 3 * π / 2 ∧ g s = 0 ∧ ⌊s⌋ = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_root_exists_and_floor_is_three_l222_22238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l222_22255

/-- The length of a platform given train speed, crossing time, and train length -/
theorem platform_length
  (train_speed : ℝ)
  (crossing_time : ℝ)
  (train_length : ℝ)
  (h1 : train_speed = 72)  -- km/hr
  (h2 : crossing_time = 22)  -- seconds
  (h3 : train_length = 190)  -- meters
  : ℝ :=
  by
  -- Convert train speed from km/hr to m/s
  let train_speed_ms := train_speed * (5/18)
  
  -- Calculate total distance covered
  let total_distance := train_speed_ms * crossing_time
  
  -- Calculate platform length
  let platform_length := total_distance - train_length
  
  -- Return the platform length
  exact platform_length

-- Example usage (commented out to avoid evaluation issues)
/- 
#eval platform_length 72 22 190
-/

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l222_22255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_x_satisfying_distance_l222_22265

/-- The distance between a real number and the nearest integer -/
noncomputable def dist_to_nearest_int (x : ℝ) : ℝ := min (x - ⌊x⌋) (⌈x⌉ - x)

/-- The existence of a real number satisfying the distance condition for all given positive integers -/
theorem existence_of_x_satisfying_distance (n : ℕ+) (a : Fin n → ℕ+) :
  ∃ x : ℝ, ∀ i : Fin n, dist_to_nearest_int ((a i : ℝ) * x) ≥ 1 / (2 * (n : ℝ)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_x_satisfying_distance_l222_22265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_plane_l222_22250

/-- The distance from a point (x₀, y₀) to a line Ax + By + C = 0 in 2D space -/
noncomputable def distance_2d (A B C x₀ y₀ : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

/-- The distance from a point (x₀, y₀, z₀) to a plane Ax + By + Cz + D = 0 in 3D space -/
noncomputable def distance_3d (A B C D x₀ y₀ z₀ : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C * z₀ + D| / Real.sqrt (A^2 + B^2 + C^2)

theorem distance_to_plane :
  distance_3d 1 2 3 3 2 4 1 = 8 * Real.sqrt 14 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_plane_l222_22250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_coefficients_l222_22223

/-- A polynomial of degree 4 with a known factor -/
structure MyPolynomial where
  p : ℝ
  q : ℝ
  has_factor : ∃ (f g : ℝ), ∀ x, p * x^4 + q * x^3 + 40 * x^2 - 24 * x + 12 = (3 * x^2 - 2 * x + 1) * (f * x^2 + g * x + 12)

/-- The theorem stating the values of p and q -/
theorem polynomial_coefficients (poly : MyPolynomial) : poly.p = 12 ∧ poly.q = -8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_coefficients_l222_22223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l222_22297

theorem circle_area_ratio (O Y P : EuclideanSpace ℝ (Fin 2)) (r₁ r₂ : ℝ) :
  ‖Y - O‖ / ‖P - O‖ = 2 / 3 →
  (π * r₁^2) / (π * r₂^2) = 4 / 9 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l222_22297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_equals_1000_l222_22228

/-- Compound interest calculation function -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

/-- Total compound interest calculation over 10 years with changing principal -/
noncomputable def total_compound_interest (initial_principal : ℝ) (rate : ℝ) : ℝ :=
  compound_interest initial_principal rate 3 +
  compound_interest (2 * initial_principal) rate 3 +
  compound_interest (4 * initial_principal) rate 1 +
  compound_interest (16 * initial_principal) rate 3

/-- Theorem stating that there exists an initial principal and interest rate
    that results in a total compound interest of 1000 after 10 years -/
theorem compound_interest_equals_1000 :
  ∃ (initial_principal : ℝ) (rate : ℝ),
    total_compound_interest initial_principal rate = 1000 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_equals_1000_l222_22228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_12_possible_partition_22_impossible_l222_22290

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def valid_partition (s : Set ℕ) (n : ℕ) : Prop :=
  ∃ (p : List (ℕ × ℕ)),
    p.length = n ∧
    (∀ pair : ℕ × ℕ, pair ∈ p → pair.1 ∈ s ∧ pair.2 ∈ s ∧ pair.1 ≠ pair.2) ∧
    (∀ i j, i ≠ j → p.get! i ≠ p.get! j) ∧
    (∀ pair : ℕ × ℕ, pair ∈ p → is_prime (pair.1 + pair.2)) ∧
    (∀ i j, i ≠ j → (p.get! i).1 + (p.get! i).2 ≠ (p.get! j).1 + (p.get! j).2)

theorem partition_12_possible :
  valid_partition (Finset.range 12).toSet 6 := by
  sorry

theorem partition_22_impossible :
  ¬ valid_partition (Finset.range 22).toSet 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_12_possible_partition_22_impossible_l222_22290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_angle_ratio_sum_l222_22229

theorem double_angle_ratio_sum (a b : ℝ) 
  (h1 : Real.sin a / Real.sin b = 4) 
  (h2 : Real.cos a / Real.cos b = 1/3) : 
  Real.sin (2*a) / Real.sin (2*b) + Real.cos (2*a) / Real.cos (2*b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_angle_ratio_sum_l222_22229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_interval_l222_22252

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 - (1/2) * Real.log x + 1

-- Define the derivative of f
noncomputable def f_deriv (x : ℝ) : ℝ := 2*x - 1/(2*x)

-- Theorem statement
theorem non_monotonic_interval (a : ℝ) :
  (∀ x, x ∈ Set.Ioo (a - 2) (a + 2) → x > 0) →
  (∃ x y, x ∈ Set.Ioo (a - 2) (a + 2) ∧ y ∈ Set.Ioo (a - 2) (a + 2) ∧ x < y ∧ f x > f y) →
  (∃ x y, x ∈ Set.Ioo (a - 2) (a + 2) ∧ y ∈ Set.Ioo (a - 2) (a + 2) ∧ x < y ∧ f x < f y) →
  a ∈ Set.Icc 2 (5/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_interval_l222_22252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_blocks_with_two_differences_l222_22204

-- Define the characteristics of a block
inductive Material : Type where
  | plastic | wood | metal
  deriving BEq, Repr

inductive Size : Type where
  | small | medium | large
  deriving BEq, Repr

inductive Color : Type where
  | blue | green | red | yellow
  deriving BEq, Repr

inductive Shape : Type where
  | circle | hexagon | square | triangle
  deriving BEq, Repr

inductive Finish : Type where
  | matte | glossy
  deriving BEq, Repr

-- Define a block
structure Block where
  material : Material
  size : Size
  color : Color
  shape : Shape
  finish : Finish
  deriving BEq, Repr

def countDifferences (b1 b2 : Block) : Nat :=
  (if b1.material != b2.material then 1 else 0) +
  (if b1.size != b2.size then 1 else 0) +
  (if b1.color != b2.color then 1 else 0) +
  (if b1.shape != b2.shape then 1 else 0) +
  (if b1.finish != b2.finish then 1 else 0)

def referenceBlock : Block := {
  material := Material.plastic
  size := Size.medium
  color := Color.red
  shape := Shape.circle
  finish := Finish.matte
}

def allBlocks : List Block := sorry

theorem count_blocks_with_two_differences :
  (allBlocks.filter (fun b => countDifferences b referenceBlock = 2)).length = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_blocks_with_two_differences_l222_22204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_l222_22243

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- Predicate for integers satisfying the equation -/
def satisfies_equation (n : ℕ) : Prop :=
  (n + 500) / 50 = floor (Real.sqrt (n : ℝ))

/-- Theorem stating there are exactly two positive integers satisfying the equation -/
theorem two_solutions : ∃! (s : Finset ℕ), 
  (∀ n ∈ s, satisfies_equation n) ∧ s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_l222_22243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_ends_in_36_l222_22218

theorem product_ends_in_36 (a b : ℕ) (ha : a < 10) (hb : b < 10) :
  (10 * a + 6) * (10 * b + 6) % 100 = 36 ↔ a + b ∈ ({0, 5, 10, 15} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_ends_in_36_l222_22218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_half_floor_inequality_l222_22225

-- Define the greatest integer function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define the logarithm with base 0.5
noncomputable def log_half (x : ℝ) : ℝ :=
  Real.log x / Real.log (1/2)

-- State the theorem
theorem log_half_floor_inequality (x : ℝ) :
  log_half (floor x) ≥ -1 ↔ floor x ∈ ({0, 1, 2} : Set ℤ) :=
by
  sorry

#check log_half_floor_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_half_floor_inequality_l222_22225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_centroids_triangle_l222_22246

/-- Triangle type with vertices A, B, C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Centroid of a triangle -/
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

/-- Area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := sorry

/-- Main theorem -/
theorem area_of_centroids_triangle (ABC : Triangle) (P G1 G2 G3 : ℝ × ℝ) : 
  P = centroid ABC →
  G1 = centroid { A := P, B := ABC.B, C := ABC.C } →
  G2 = centroid { A := P, B := ABC.C, C := ABC.A } →
  G3 = centroid { A := P, B := ABC.A, C := ABC.B } →
  area ABC = 24 →
  area { A := G1, B := G2, C := G3 } = 8/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_centroids_triangle_l222_22246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_color_draw_probability_l222_22200

def tan_chips : ℕ := 4
def pink_chips : ℕ := 3
def violet_chips : ℕ := 5
def blue_chips : ℕ := 2

def total_chips : ℕ := tan_chips + pink_chips + violet_chips + blue_chips

theorem consecutive_color_draw_probability :
  (tan_chips.factorial * pink_chips.factorial * violet_chips.factorial * blue_chips.factorial : ℚ) / total_chips.factorial = 1 / 2522520 := by
  sorry

#eval total_chips

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_color_draw_probability_l222_22200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_dist_probability_l222_22239

-- Define a random variable following a normal distribution
structure NormalDist (μ σ : ℝ) where
  value : ℝ
  h : σ > 0

-- Define the probability measure
noncomputable def P {μ σ : ℝ} (h : σ > 0) : Set (NormalDist μ σ) → ℝ := sorry

-- Theorem statement
theorem normal_dist_probability 
  {σ : ℝ} (h : σ > 0) 
  (h1 : P h {x : NormalDist 2 σ | 0 < x.value ∧ x.value < 2} = 0.4) : 
  P h {x : NormalDist 2 σ | x.value < 4} = 0.9 := by sorry

#check normal_dist_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_dist_probability_l222_22239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eggs_per_hen_approx_l222_22296

/-- The average number of eggs laid per hen -/
noncomputable def average_eggs_per_hen (total_eggs : ℝ) (num_hens : ℝ) : ℝ :=
  total_eggs / num_hens

/-- Theorem stating that the average number of eggs per hen is approximately 10.82 -/
theorem eggs_per_hen_approx :
  let total_eggs : ℝ := 303.0
  let num_hens : ℝ := 28.0
  let result := average_eggs_per_hen total_eggs num_hens
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ |result - 10.82| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eggs_per_hen_approx_l222_22296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_on_unit_square_l222_22217

noncomputable section

-- Define a unit square
def UnitSquare : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the perimeter of the unit square
def Perimeter : Set (ℝ × ℝ) :=
  {p | (p.1 = 0 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1) ∨
       (p.1 = 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1) ∨
       (p.2 = 0 ∧ 0 ≤ p.1 ∧ p.1 ≤ 1) ∨
       (p.2 = 1 ∧ 0 ≤ p.1 ∧ p.1 ≤ 1)}

-- Define a triangle on the perimeter
def TriangleOnPerimeter (a b c : ℝ × ℝ) : Prop :=
  a ∈ Perimeter ∧ b ∈ Perimeter ∧ c ∈ Perimeter

-- Define the area of a triangle given its vertices
noncomputable def TriangleArea (a b c : ℝ × ℝ) : ℝ :=
  (1/2) * abs (a.1 * (b.2 - c.2) + b.1 * (c.2 - a.2) + c.1 * (a.2 - b.2))

-- Theorem statement
theorem max_triangle_area_on_unit_square :
  ∀ a b c : ℝ × ℝ, TriangleOnPerimeter a b c →
  TriangleArea a b c ≤ 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_on_unit_square_l222_22217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_first_four_l222_22264

/-- The sum of the first n terms of a geometric sequence with first term a and common ratio r -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

/-- Theorem: The sum of the first four terms of a geometric sequence {a_n} is 40, given a_1 = 1 and a_4 = 27 -/
theorem geometric_sequence_sum_first_four (a : ℕ → ℝ) :
  a 1 = 1 → a 4 = 27 → geometric_sum (a 1) ((a 4 / a 1)^(1/3)) 4 = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_first_four_l222_22264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2011_l222_22256

def sequenceA : ℕ → ℕ
  | 0 => 8
  | 1 => 3
  | n + 2 => (sequenceA n + sequenceA (n + 1)) % 10

theorem sequence_2011 : sequenceA 2010 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2011_l222_22256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_difference_two_l222_22241

def numbers : Finset ℤ := {5, 6, 7, 8}

def valid_pairs (s : Finset ℤ) : Finset (ℤ × ℤ) :=
  s.product s |> Finset.filter (fun p => p.1 ≠ p.2 ∧ (p.1 - p.2).natAbs = 2)

theorem probability_of_difference_two :
  (valid_pairs numbers).card / Nat.choose numbers.card 2 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_difference_two_l222_22241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_book_purchase_l222_22212

/-- Represents the price and quantity of books purchased by a school. -/
structure BookPurchase where
  lit_price : ℝ  -- Price of each literature book
  sci_price : ℝ  -- Price of each popular science book
  lit_count : ℕ  -- Number of literature books
  sci_count : ℕ  -- Number of popular science books

/-- The conditions of the book purchase problem. -/
def purchase_conditions (bp : BookPurchase) : Prop :=
  bp.sci_price = 1.2 * bp.lit_price ∧
  bp.lit_count + 10 = Int.floor (1200 / bp.lit_price) ∧
  bp.sci_count = Int.floor (1200 / bp.sci_price) ∧
  bp.lit_count + bp.sci_count = 1000 ∧
  bp.sci_count ≥ Int.floor ((2/3 : ℝ) * bp.lit_count)

/-- The total cost of the book purchase. -/
def total_cost (bp : BookPurchase) : ℝ :=
  bp.lit_price * bp.lit_count + bp.sci_price * bp.sci_count

/-- The theorem stating the minimum cost of the book purchase. -/
theorem min_cost_book_purchase :
  ∃ (bp : BookPurchase), purchase_conditions bp ∧
    total_cost bp = 21600 ∧
    bp.lit_count = 600 ∧
    bp.sci_count = 400 ∧
    ∀ (bp' : BookPurchase), purchase_conditions bp' → total_cost bp' ≥ total_cost bp :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_book_purchase_l222_22212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_section_distances_l222_22269

/-- 
Given a cone of height m divided into three sections by two planes parallel to the base,
such that the areas of the surfaces of the sections are equal,
prove that the distances of the intersecting planes from the apex of the cone are m/√3 and (m/3)√6.
-/
theorem cone_section_distances (m : ℝ) (h : m > 0) :
  ∃ (m₁ m₂ : ℝ), 
    0 < m₁ ∧ m₁ < m₂ ∧ m₂ < m ∧
    (∀ (r : ℝ), r > 0 → 
      let r₁ := r * m₁ / m
      let r₂ := r * m₂ / m
      let l := Real.sqrt (m^2 + r^2)
      let l₁ := Real.sqrt (m₁^2 + r₁^2)
      let l₂ := Real.sqrt (m₂^2 + r₂^2)
      r₁ * l₁ = (1/3) * r * l ∧
      (r₂ * l₂ - r₁ * l₁) = (1/3) * r * l ∧
      (r * l - r₂ * l₂) = (1/3) * r * l) →
    m₁ = m / Real.sqrt 3 ∧ m₂ = (m / 3) * Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_section_distances_l222_22269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l222_22233

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log (abs (x - 2)) + x^2
def g (x : ℝ) : ℝ := 4 * x

-- State the theorem
theorem intersection_sum : 
  ∃ (x₁ x₂ : ℝ), 
    (f x₁ = g x₁) ∧ 
    (f x₂ = g x₂) ∧ 
    (x₁ ≠ x₂) ∧
    (∀ (x : ℝ), f x = g x → x = x₁ ∨ x = x₂) ∧
    (∀ (x : ℝ), f (4 - x) = f x) ∧
    (∀ (x : ℝ), g (4 - x) = g x) →
    x₁ + x₂ = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l222_22233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solution_l222_22245

theorem inequality_system_solution : 
  {(x, y) : ℤ × ℤ | (y : ℝ) + (1/2 : ℝ) > |(x : ℝ)^2 - 2*(x : ℝ)| ∧ (y : ℝ) ≤ 2 - |(x : ℝ) - 1|} = 
  {(0, 0), (0, 1), (1, 1), (1, 2), (2, 0), (2, 1)} := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solution_l222_22245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_range_l222_22221

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then (1/2)^x - 1 else (a-2)*x + 1

theorem monotone_decreasing_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) → a ∈ Set.Icc (1/2) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_range_l222_22221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_riccati_general_solution_l222_22205

-- Define the Riccati equation
def riccati_equation (y : ℝ → ℝ) : Prop :=
  ∀ x, (deriv y x) + 2 * (Real.exp x) * (y x) - (y x)^2 = (Real.exp x) + (Real.exp (2*x))

-- Define the particular solution
noncomputable def particular_solution (x : ℝ) : ℝ := Real.exp x

-- Define the general solution
noncomputable def general_solution (C : ℝ) (x : ℝ) : ℝ := Real.exp x + 1 / (C - x)

-- Theorem statement
theorem riccati_general_solution :
  (riccati_equation particular_solution) →
  (∀ C, riccati_equation (general_solution C)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_riccati_general_solution_l222_22205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radii_sum_l222_22208

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define an inscribed circle
structure InscribedCircle where
  center : Point
  radius : ℝ

-- Helper definitions (not proved)
def is_inscribed (circle : InscribedCircle) (T : Triangle) : Prop :=
  sorry

def are_tangent_parallel_inscribed (circle_A : InscribedCircle) (circle_B : InscribedCircle) (circle_C : InscribedCircle) (T : Triangle) (main_circle : InscribedCircle) : Prop :=
  sorry

-- Define the main theorem
theorem inscribed_circle_radii_sum (T : Triangle) 
  (main_circle : InscribedCircle) 
  (small_circle_A : InscribedCircle) 
  (small_circle_B : InscribedCircle) 
  (small_circle_C : InscribedCircle) :
  is_inscribed main_circle T → 
  are_tangent_parallel_inscribed small_circle_A small_circle_B small_circle_C T main_circle →
  small_circle_A.radius + small_circle_B.radius + small_circle_C.radius = main_circle.radius :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radii_sum_l222_22208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_C_l222_22213

open Real

variable (a b c : ℝ)
variable (A B C : ℝ)

-- Define a triangle ABC
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the relationship between sides and angles in a triangle
def triangle_angles (a b c A B C : ℝ) : Prop :=
  is_triangle a b c ∧
  0 < A ∧ A < Real.pi ∧
  0 < B ∧ B < Real.pi ∧
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi ∧
  a / (sin A) = b / (sin B) ∧
  b / (sin B) = c / (sin C)

-- State the theorem
theorem angle_measure_C (a b c A B C : ℝ) :
  triangle_angles a b c A B C →
  (a + b + c) * (sin A + sin B - sin C) = 3 * a * sin B →
  C = Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_C_l222_22213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_cross_time_l222_22222

/-- Represents the time it takes for a train to cross a telegraph post -/
noncomputable def crossTime (trainLength : ℝ) (trainSpeed : ℝ) : ℝ :=
  trainLength / trainSpeed

/-- Represents the relative speed of two trains moving in opposite directions -/
noncomputable def relativeSpeed (train1Speed : ℝ) (train2Speed : ℝ) : ℝ :=
  train1Speed + train2Speed

theorem second_train_cross_time 
  (trainLength : ℝ)
  (train1CrossTime : ℝ)
  (crossingTime : ℝ)
  (h1 : trainLength = 120)
  (h2 : train1CrossTime = 10)
  (h3 : crossingTime = 12)
  : crossTime trainLength ((relativeSpeed (trainLength / train1CrossTime) 
    ((2 * trainLength) / crossingTime - trainLength / train1CrossTime))) = 15 := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_cross_time_l222_22222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_270_of_4_minus_2i_l222_22289

def rotate_270 (z : ℂ) : ℂ := -z * Complex.I

theorem rotation_270_of_4_minus_2i :
  rotate_270 (4 - 2 * Complex.I) = 2 - 4 * Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_270_of_4_minus_2i_l222_22289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_2_4_km_l222_22251

/-- Calculates the distance to a place given rowing speed, current velocity, and round trip time --/
noncomputable def distance_to_place (rowing_speed : ℝ) (current_velocity : ℝ) (round_trip_time : ℝ) : ℝ :=
  (round_trip_time * (rowing_speed^2 - current_velocity^2)) / (2 * rowing_speed)

/-- Theorem: The distance to the place is 2.4 km given the specified conditions --/
theorem distance_is_2_4_km 
  (rowing_speed : ℝ) 
  (current_velocity : ℝ) 
  (round_trip_time : ℝ) 
  (h1 : rowing_speed = 5)
  (h2 : current_velocity = 1)
  (h3 : round_trip_time = 1) :
  distance_to_place rowing_speed current_velocity round_trip_time = 2.4 := by
  sorry

-- Remove the #eval statement as it's not compatible with noncomputable definitions
-- #eval distance_to_place 5 1 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_2_4_km_l222_22251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_combined_time_l222_22216

/-- Given three workers with different work rates, calculates the time taken when they work together -/
theorem workers_combined_time (x : ℝ) (hx : x > 0) : 
  (1 / ((1 / x) + (1 / (2 * x)) + (1 / (4 * x)))) = 4 * x / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_combined_time_l222_22216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fredholm_integral_equation_solution_l222_22211

open Real

-- Define the kernel K(x, t)
noncomputable def K (x t : ℝ) : ℝ :=
  if t ≤ x then (1 - x) * t else x * (1 - t)

-- Define the right-hand side of the equation
noncomputable def f (x : ℝ) : ℝ := (sin (Real.pi * x)) ^ 3

-- Define the proposed solution φ(x)
noncomputable def φ (x : ℝ) : ℝ := (3 * Real.pi^2 / 4) * (sin (Real.pi * x) - 3 * sin (3 * Real.pi * x))

-- State the theorem
theorem fredholm_integral_equation_solution (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) :
  ∫ t in Set.Icc 0 1, K x t * φ t = f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fredholm_integral_equation_solution_l222_22211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_choose_three_from_eight_l222_22271

theorem choose_three_from_eight : Finset.card (Finset.powersetCard 3 (Finset.range 8)) = 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_choose_three_from_eight_l222_22271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_polyhedron_volume_theorem_l222_22279

/-- The volume of the solid formed by planes passing through every set of three vertices 
    at the ends of edges meeting at each vertex of a cube with edge length a -/
noncomputable def innerPolyhedronVolume (a : ℝ) : ℝ := a^3 / 6

/-- Theorem stating that the volume of the inner polyhedron in a cube of edge length a 
    is equal to a^3 / 6 -/
theorem inner_polyhedron_volume_theorem (a : ℝ) (h : a > 0) :
  innerPolyhedronVolume a = a^3 / 6 := by
  -- Unfold the definition of innerPolyhedronVolume
  unfold innerPolyhedronVolume
  -- The equality is trivial by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_polyhedron_volume_theorem_l222_22279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_comparison_l222_22291

theorem exponent_comparison (a b c d : ℝ) (ha : a > 1) (hb : 0 < b ∧ b < 1) (hc : c > 0) (hd : d > 0) :
  a^c > b^d := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_comparison_l222_22291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_plus_cos_pi_ninth_l222_22293

theorem sec_plus_cos_pi_ninth :
  1 / Real.cos (π / 9) + 3 * Real.cos (π / 9) = 4 - 3 * Real.sin (π / 9)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_plus_cos_pi_ninth_l222_22293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_positive_real_solutions_l222_22270

def f (x : ℝ) : ℝ := x^10 - 4*x^9 + 6*x^8 + 878*x^7 - 3791*x^6

theorem two_positive_real_solutions :
  ∃! (s : Finset ℝ), (∀ x ∈ s, x > 0 ∧ f x = 0) ∧ s.card = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_positive_real_solutions_l222_22270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_value_theorem_l222_22288

noncomputable def point_P : ℝ × ℝ := (Real.sqrt 3 / 2, -1 / 2)

def is_on_terminal_side (θ : ℝ) (p : ℝ × ℝ) : Prop :=
  p.1 = Real.cos θ * Real.sqrt (p.1^2 + p.2^2) ∧
  p.2 = Real.sin θ * Real.sqrt (p.1^2 + p.2^2)

theorem angle_value_theorem (θ : ℝ) 
  (h1 : 0 ≤ θ ∧ θ < 2 * Real.pi)
  (h2 : is_on_terminal_side θ point_P) :
  θ = 11 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_value_theorem_l222_22288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l222_22257

-- Define the job completion rates
noncomputable def rate_A : ℝ := 1 / 15
noncomputable def rate_D : ℝ := 1 / 30
noncomputable def rate_combined : ℝ := 1 / 10

-- Theorem statement
theorem job_completion_time :
  rate_A = 1 / 15 →
  rate_A + rate_D = rate_combined →
  rate_combined = 1 / 10 →
  rate_D = 1 / 30 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l222_22257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l222_22231

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2 * x^2 + a * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 3 * x + 2 * a = 0}

-- Define the universal set U
def U (a : ℝ) : Set ℝ := A a ∪ B a

-- State the theorem
theorem problem_solution :
  ∃ a : ℝ,
    (A a ∩ B a = {2}) ∧
    (a = -5) ∧
    (A a = {2, 1/2}) ∧
    (B a = {2, -5}) ∧
    (Set.powerset ((U a \ A a) ∪ (U a \ B a)) = {∅, {-5}, {1/2}, {-5, 1/2}}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l222_22231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decrease_interval_max_k_value_l222_22275

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 - 3*x

-- Define the derivative of f(x)
noncomputable def f' (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Theorem for the interval of monotonic decrease
theorem monotonic_decrease_interval :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 3, f' x < 0 := by
  sorry

-- Theorem for the maximum value of k
theorem max_k_value :
  ∀ k : ℕ, k ≤ 6 ↔ 
    ∀ x : ℝ, x > 0 → f' x > k * (x * Real.log x - 1) - 6*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decrease_interval_max_k_value_l222_22275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_when_intersection_angle_30deg_l222_22230

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := 
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The angle between the line connecting the left vertex to the point of intersection 
    of the asymptote and the circle through the foci -/
noncomputable def intersectionAngle (h : Hyperbola) : ℝ := 
  Real.arctan (h.b / h.a)

theorem hyperbola_eccentricity_when_intersection_angle_30deg (h : Hyperbola) :
  intersectionAngle h = π / 6 → eccentricity h = Real.sqrt 21 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_when_intersection_angle_30deg_l222_22230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equality_and_integer_impossibility_l222_22237

theorem fraction_equality_and_integer_impossibility :
  -- Part 1: Addition equality
  (169 : ℚ) / 30 + 13 / 15 = 13 / 2 ∧
  -- Part 2: Division equality
  (169 : ℚ) / 30 / (13 / 15) = 13 / 2 ∧
  -- Part 3: No positive integer solution
  ∀ a b : ℕ, a > 0 ∧ b > 0 → (a : ℚ) + b ≠ (a : ℚ) / (b : ℚ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equality_and_integer_impossibility_l222_22237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l222_22272

theorem remainder_theorem (x : ℤ) (h : (7 * x) % 31 = 1) : 
  (13 + x) % 31 = 22 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l222_22272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_system_l222_22244

/-- The number of people -/
def x : ℕ := sorry

/-- The number of cars -/
def y : ℕ := sorry

/-- Condition 1: If each car carries 3 people, there are 2 empty cars -/
axiom condition1 : x / 3 = y - 2

/-- Condition 2: If each car carries 2 people, there are 9 people walking -/
axiom condition2 : (x - 9) / 2 = y

/-- The correct system of equations for the problem -/
theorem correct_system :
  (x / 3 = y - 2) ∧ ((x - 9) / 2 = y) := by
  constructor
  . exact condition1
  . exact condition2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_system_l222_22244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_statements_imply_negation_l222_22283

theorem three_statements_imply_negation (p q : Prop) :
  let statement1 := p ∨ q
  let statement2 := p ∨ ¬q
  let statement3 := ¬p ∨ q
  let statement4 := ¬p ∨ ¬q
  let negation := ¬(p ∧ q)
  3 = (([statement1, statement2, statement3, statement4].filter (λ s => (s → negation) = True)).length : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_statements_imply_negation_l222_22283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_irreducibility_l222_22247

theorem polynomial_irreducibility (n : ℕ) (h : n > 1) :
  ¬∃ (g h : Polynomial ℤ) (hg : Polynomial.degree g ≥ 1) (hh : Polynomial.degree h ≥ 1),
    (X : Polynomial ℤ)^n + 5*(X : Polynomial ℤ)^(n-1) + 3 = g * h := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_irreducibility_l222_22247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_is_positive_integer_l222_22259

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1  -- Add case for 0
  | 1 => 1
  | n + 2 => sequence_a (n + 1) / 2 + 1 / (4 * sequence_a (n + 1))

theorem sqrt_is_positive_integer (n : ℕ) (h : n > 1) :
  ∃ k : ℕ+, Real.sqrt (2 / (2 * sequence_a n ^ 2 - 1)) = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_is_positive_integer_l222_22259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l222_22224

/-- The distance between the vertices of a hyperbola with equation x²/121 - y²/49 = 1 is 22 -/
theorem hyperbola_vertex_distance :
  ∃ (v₁ v₂ : ℝ × ℝ), 
    (v₁.1^2 / 121 - v₁.2^2 / 49 = 1) ∧ 
    (v₂.1^2 / 121 - v₂.2^2 / 49 = 1) ∧ 
    ‖v₁ - v₂‖ = 22 ∧ 
    ∀ (v : ℝ × ℝ), (v.1^2 / 121 - v.2^2 / 49 = 1) → ‖v - v₁‖ ≤ 22 ∧ ‖v - v₂‖ ≤ 22 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l222_22224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_proof_l222_22260

theorem triangle_angle_proof (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → A < π →
  (Real.sqrt 3 * (b^2 + c^2 - a^2)) / 4 = (1/2) * b * c * Real.sin A →
  A = π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_proof_l222_22260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_tank_design_l222_22240

/-- Represents the specifications and cost of a rectangular prism water tank -/
structure TankSpecs where
  volume : ℝ
  depth : ℝ
  bottom_cost : ℝ
  wall_cost : ℝ

/-- Calculates the total cost of the tank given its length -/
noncomputable def total_cost (specs : TankSpecs) (length : ℝ) : ℝ :=
  let width := specs.volume / (specs.depth * length)
  let bottom_area := length * width
  let wall_area := 2 * specs.depth * (length + width)
  specs.bottom_cost * bottom_area + specs.wall_cost * wall_area

/-- Theorem stating the minimum cost and optimal length for the water tank -/
theorem optimal_tank_design (specs : TankSpecs) 
    (h_volume : specs.volume = 4800)
    (h_depth : specs.depth = 3)
    (h_bottom_cost : specs.bottom_cost = 150)
    (h_wall_cost : specs.wall_cost = 120) :
    ∃ (optimal_length : ℝ),
      optimal_length = 40 ∧
      (∀ (length : ℝ), length > 0 → total_cost specs length ≥ 297600) ∧
      total_cost specs optimal_length = 297600 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_tank_design_l222_22240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_two_thirty_between_consecutive_integers_l222_22295

theorem log_two_thirty_between_consecutive_integers :
  ∃ (a b : ℤ), a + 1 = b ∧ (a : ℝ) < Real.log 30 / Real.log 2 ∧ Real.log 30 / Real.log 2 < b ∧ a + b = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_two_thirty_between_consecutive_integers_l222_22295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_theorem_l222_22242

/-- The x-coordinate of the point on the x-axis that is equidistant from A(-4, 0) and B(0, 10) -/
noncomputable def equidistant_point : ℝ := 21 / 2

/-- Point A -/
def A : ℝ × ℝ := (-4, 0)

/-- Point B -/
def B : ℝ × ℝ := (0, 10)

/-- Distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem equidistant_point_theorem :
  distance A (equidistant_point, 0) = distance B (equidistant_point, 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_theorem_l222_22242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_robot_encoding_l222_22263

/-- Represents the encoding of a letter in the cipher. -/
inductive Encoding
| one_digit : Fin 3 → Encoding
| two_digit : Fin 3 → Fin 3 → Encoding

/-- The cipher used by the robot. -/
def Cipher := Char → Encoding

def is_valid_cipher (c : Cipher) : Prop :=
  ∀ x y : Char, x ≠ y → c x ≠ c y

def encode (c : Cipher) (s : String) : List (Fin 3) :=
  s.toList.foldl (fun acc ch =>
    match c ch with
    | Encoding.one_digit d => acc ++ [d]
    | Encoding.two_digit d1 d2 => acc ++ [d1, d2]
  ) []

theorem robot_encoding (c : Cipher) :
  is_valid_cipher c →
  encode c "ROBOT" = [3, 1, 1, 2, 1, 3, 1, 2, 3, 3] →
  encode c "CROCODILE" = encode c "HIPPOPOTAMUS" →
  encode c "MATHEMATICS" = [2, 2, 3, 2, 3, 3, 1, 1, 2, 2, 3, 2, 3, 3, 2, 3, 1, 3, 2] :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_robot_encoding_l222_22263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_equals_two_l222_22266

/-- Represents a circular sector -/
structure CircularSector where
  circumference : ℝ
  centralAngle : ℝ

/-- Calculates the area of a circular sector -/
noncomputable def sectorArea (s : CircularSector) : ℝ :=
  let r := s.circumference / (2 + s.centralAngle)
  (1 / 2) * r * r * s.centralAngle

theorem sector_area_equals_two 
  (s : CircularSector) 
  (h1 : s.circumference = 6) 
  (h2 : s.centralAngle = 1) : 
  sectorArea s = 2 := by
  sorry

#check sector_area_equals_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_equals_two_l222_22266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dna_sequence_count_l222_22203

theorem dna_sequence_count (chain_length : ℝ) (base_count : ℕ) : 
  chain_length = 2.1 * (10 : ℝ)^10 → 
  base_count = 4 → 
  ∃ N : ℝ, N > (10 : ℝ)^(1.26 * (10 : ℝ)^10) ∧ N = (base_count : ℝ)^chain_length :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dna_sequence_count_l222_22203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_f_monotonic_decrease_g_min_value_l222_22278

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + 2 * Real.pi / 3) + 2 * (Real.cos x) ^ 2

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi / 3)

-- Theorem for the smallest positive period of f
theorem f_period : ∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S → S < T → ∃ y, f (y + S) ≠ f y :=
sorry

-- Theorem for the intervals of monotonic decrease of f
theorem f_monotonic_decrease (k : ℤ) :
  StrictMonoOn f (Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3)) :=
sorry

-- Theorem for the minimum value of g in [0, π/2]
theorem g_min_value : 
  ∃ x₀ ∈ Set.Icc 0 (Real.pi / 2), ∀ x ∈ Set.Icc 0 (Real.pi / 2), g x₀ ≤ g x ∧ g x₀ = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_f_monotonic_decrease_g_min_value_l222_22278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_operation_l222_22215

-- Define the operations
def operation_a (x : ℝ) : Prop := Real.sqrt x = x ∨ Real.sqrt x = -x
def operation_b (x : ℝ) : Prop := Real.sqrt x = 3.5
def operation_c (x : ℝ) : Prop := x^(1/3 : ℝ) = -1
def operation_d (x : ℝ) : Prop := Real.sqrt (x^2) = -1

-- Theorem statement
theorem correct_operation :
  ¬(operation_a 16) ∧
  ¬(operation_b (37/4)) ∧
  operation_c (-1) ∧
  ¬(operation_d (-1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_operation_l222_22215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_from_sine_half_angle_l222_22227

theorem tangent_from_sine_half_angle (θ x : ℝ) (h1 : 0 < θ) (h2 : θ < π / 2) 
  (h3 : Real.sin (θ / 2) = Real.sqrt ((x - 1) / (2 * x))) : 
  Real.tan θ = Real.sqrt (x^2 - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_from_sine_half_angle_l222_22227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_increasing_implies_a_geq_neg_2_sqrt_2_exists_a_between_neg_2_sqrt_2_and_neg_2_l222_22284

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x + x^2 + a*x + 1

theorem monotonically_increasing_implies_a_geq_neg_2_sqrt_2 (a : ℝ) :
  (∀ x > 0, StrictMono (f a)) → a ≥ -2 * sqrt 2 :=
by sorry

theorem exists_a_between_neg_2_sqrt_2_and_neg_2 :
  ∃ a : ℝ, -2 * sqrt 2 ≤ a ∧ a < -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_increasing_implies_a_geq_neg_2_sqrt_2_exists_a_between_neg_2_sqrt_2_and_neg_2_l222_22284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_3_proposition_4_l222_22287

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Notation for parallel and perpendicular
local notation:50 l " ∥ " α => parallel l α
local notation:50 l " ⊥ " α => perpendicular l α
local notation:50 α " ∥ " β => plane_parallel α β
local notation:50 α " ⊥ " β => plane_perpendicular α β

-- Theorem statements
theorem proposition_3 {l : Line} {α β : Plane} :
  (l ∥ α) → (l ⊥ β) → (α ⊥ β) :=
sorry

theorem proposition_4 {l : Line} {α β : Plane} :
  (α ⊥ β) → (l ∥ α) → (l ⊥ β) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_3_proposition_4_l222_22287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_sum_l222_22226

/-- Reflects a point (x, y) across the line y = mx + b -/
noncomputable def reflect (x y m b : ℝ) : ℝ × ℝ :=
  let x' := ((1 - m^2) * x + 2*m*(y - b)) / (1 + m^2)
  let y' := (2*m*x - (1 - m^2)*(y - b)) / (1 + m^2) + b
  (x', y')

/-- The main theorem stating that if (1,1) reflects to (9,5) across y = mx + b, then m + b = 11 -/
theorem reflection_sum (m b : ℝ) : reflect 1 1 m b = (9, 5) → m + b = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_sum_l222_22226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l222_22235

-- Define the line equation
def line_equation (k x : ℝ) : ℝ := k * x + 2

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := x^2 - y^2 = 6

-- Define the condition for two distinct intersection points
def has_two_distinct_intersections (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  hyperbola_equation x₁ (line_equation k x₁) ∧
  hyperbola_equation x₂ (line_equation k x₂) ∧
  x₁ > 0 ∧ x₂ > 0  -- Ensure right branch of hyperbola

-- State the theorem
theorem intersection_range :
  ∀ k : ℝ, has_two_distinct_intersections k ↔ -Real.sqrt 15 / 3 < k ∧ k < Real.sqrt 15 / 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l222_22235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daffodil_fraction_l222_22267

/-- Represents the total number of flowers in the garden -/
def total : ℕ → ℕ := id

/-- Represents the number of yellow flowers -/
def yellow (t : ℕ) : ℕ := (4 * t) / 7

/-- Represents the number of red flowers -/
def red (t : ℕ) : ℕ := t - yellow t

/-- Represents the number of yellow tulips -/
def yellow_tulips (t : ℕ) : ℕ := yellow t / 2

/-- Represents the number of yellow daffodils -/
def yellow_daffodils (t : ℕ) : ℕ := yellow t - yellow_tulips t

/-- Represents the number of red daffodils -/
def red_daffodils (t : ℕ) : ℕ := (2 * red t) / 3

/-- Represents the total number of daffodils -/
def total_daffodils (t : ℕ) : ℕ := yellow_daffodils t + red_daffodils t

theorem daffodil_fraction (t : ℕ) (h : t > 0) :
  (total_daffodils t : ℚ) / t = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_daffodil_fraction_l222_22267
