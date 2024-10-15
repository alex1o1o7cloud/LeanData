import Mathlib

namespace NUMINAMATH_CALUDE_polygon_contains_circle_l3582_358236

/-- A convex polygon with width 1 -/
structure ConvexPolygon where
  width : ℝ
  width_eq_one : width = 1
  is_convex : Bool  -- This is a simplification, as convexity is more complex to define

/-- A circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Predicate to check if a circle is contained within a polygon -/
def containsCircle (p : ConvexPolygon) (c : Circle) : Prop :=
  sorry  -- The actual implementation would depend on how we represent polygons and circles

theorem polygon_contains_circle (M : ConvexPolygon) : 
  ∃ (c : Circle), c.radius ≥ 1/3 ∧ containsCircle M c := by
  sorry

#check polygon_contains_circle

end NUMINAMATH_CALUDE_polygon_contains_circle_l3582_358236


namespace NUMINAMATH_CALUDE_edmund_earnings_is_64_l3582_358215

/-- Calculates Edmund's earnings for extra chores over two weeks -/
def edmund_earnings (normal_chores_per_week : ℕ) (chores_per_day : ℕ) (days : ℕ) (pay_per_extra_chore : ℕ) : ℕ :=
  let total_chores := chores_per_day * days
  let normal_chores := normal_chores_per_week * 2
  let extra_chores := total_chores - normal_chores
  extra_chores * pay_per_extra_chore

/-- Proves that Edmund's earnings for extra chores over two weeks is $64 -/
theorem edmund_earnings_is_64 :
  edmund_earnings 12 4 14 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_edmund_earnings_is_64_l3582_358215


namespace NUMINAMATH_CALUDE_percentage_fraction_difference_l3582_358218

theorem percentage_fraction_difference : 
  (85 / 100 * 40) - (4 / 5 * 25) = 14 := by sorry

end NUMINAMATH_CALUDE_percentage_fraction_difference_l3582_358218


namespace NUMINAMATH_CALUDE_star_calculation_l3582_358262

-- Define the binary operation *
def star (a b : ℝ) : ℝ := (a - b)^2

-- State the theorem
theorem star_calculation (x y z : ℝ) : 
  star ((x - z)^2) ((z - y)^2) = (x^2 - 2*x*z + 2*z*y - y^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l3582_358262


namespace NUMINAMATH_CALUDE_water_depth_calculation_l3582_358234

def water_depth (ron_height dean_height_difference : ℝ) : ℝ :=
  let dean_height := ron_height - dean_height_difference
  2.5 * dean_height + 3

theorem water_depth_calculation (ron_height dean_height_difference : ℝ) 
  (h1 : ron_height = 14.2)
  (h2 : dean_height_difference = 8.3) :
  water_depth ron_height dean_height_difference = 17.75 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_calculation_l3582_358234


namespace NUMINAMATH_CALUDE_chord_length_is_four_l3582_358288

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a line in polar form ρ(sin θ - cos θ) = k -/
structure PolarLine where
  k : ℝ

/-- Represents a circle in polar form ρ = a sin θ -/
structure PolarCircle where
  a : ℝ

/-- The length of the chord cut by a polar line from a polar circle -/
noncomputable def chordLength (l : PolarLine) (c : PolarCircle) : ℝ := sorry

/-- Theorem: The chord length is 4 for the given line and circle -/
theorem chord_length_is_four :
  let l : PolarLine := { k := 2 }
  let c : PolarCircle := { a := 4 }
  chordLength l c = 4 := by sorry

end NUMINAMATH_CALUDE_chord_length_is_four_l3582_358288


namespace NUMINAMATH_CALUDE_original_decimal_value_l3582_358282

theorem original_decimal_value : 
  ∃ x : ℝ, (x / 100 = x - 1.782) ∧ (x = 1.8) := by
  sorry

end NUMINAMATH_CALUDE_original_decimal_value_l3582_358282


namespace NUMINAMATH_CALUDE_opposite_pairs_l3582_358210

theorem opposite_pairs : 
  (3^2 = -(-(3^2))) ∧ 
  (-4 ≠ -(-4)) ∧ 
  (-3 ≠ -(-|-3|)) ∧ 
  (-2^3 ≠ -((-2)^3)) := by
  sorry

end NUMINAMATH_CALUDE_opposite_pairs_l3582_358210


namespace NUMINAMATH_CALUDE_extra_planks_count_l3582_358294

/-- The number of planks Charlie got -/
def charlie_planks : ℕ := 10

/-- The number of planks Charlie's father got -/
def father_planks : ℕ := 10

/-- The total number of wood pieces they have -/
def total_wood : ℕ := 35

/-- The number of extra planks initially in the house -/
def extra_planks : ℕ := total_wood - (charlie_planks + father_planks)

theorem extra_planks_count : extra_planks = 15 := by
  sorry

end NUMINAMATH_CALUDE_extra_planks_count_l3582_358294


namespace NUMINAMATH_CALUDE_base_conversion_count_l3582_358299

theorem base_conversion_count : 
  ∃! n : ℕ, n = (Finset.filter (fun c : ℕ => c ≥ 2 ∧ c^2 ≤ 256 ∧ 256 < c^3) (Finset.range 257)).card ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_count_l3582_358299


namespace NUMINAMATH_CALUDE_greatest_t_value_l3582_358211

theorem greatest_t_value : 
  let f : ℝ → ℝ := λ t => (t^2 - t - 90) / (t - 8)
  let g : ℝ → ℝ := λ t => 6 / (t + 7)
  ∃ t_max : ℝ, t_max = -1 ∧ 
    (∀ t : ℝ, t ≠ 8 ∧ t ≠ -7 → f t = g t → t ≤ t_max) :=
by sorry

end NUMINAMATH_CALUDE_greatest_t_value_l3582_358211


namespace NUMINAMATH_CALUDE_james_total_cost_l3582_358254

/-- The total cost of buying dirt bikes, off-road vehicles, and registering them all -/
def total_cost (dirt_bike_count : ℕ) (dirt_bike_price : ℕ) 
                (offroad_count : ℕ) (offroad_price : ℕ) 
                (registration_fee : ℕ) : ℕ :=
  dirt_bike_count * dirt_bike_price + 
  offroad_count * offroad_price + 
  (dirt_bike_count + offroad_count) * registration_fee

/-- Theorem stating the total cost for James' purchase -/
theorem james_total_cost : 
  total_cost 3 150 4 300 25 = 1825 := by
  sorry

end NUMINAMATH_CALUDE_james_total_cost_l3582_358254


namespace NUMINAMATH_CALUDE_grid_coloring_4x2011_l3582_358206

/-- Represents the number of ways to color a 4 × n grid with the given constraints -/
def coloringWays (n : ℕ) : ℕ :=
  64 * 3^(2*n)

/-- The problem statement -/
theorem grid_coloring_4x2011 :
  coloringWays 2011 = 64 * 3^4020 :=
by sorry

end NUMINAMATH_CALUDE_grid_coloring_4x2011_l3582_358206


namespace NUMINAMATH_CALUDE_solution_set_l3582_358283

theorem solution_set (x : ℝ) :
  (|x^2 - x - 2| + |1/x| = |x^2 - x - 2 + 1/x|) →
  ((-1 ≤ x ∧ x < 0) ∨ x ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_l3582_358283


namespace NUMINAMATH_CALUDE_abcd_efgh_ratio_l3582_358285

theorem abcd_efgh_ratio 
  (a b c d e f g h : ℝ) 
  (hab : a / b = 1 / 3)
  (hbc : b / c = 2)
  (hcd : c / d = 1 / 2)
  (hde : d / e = 3)
  (hef : e / f = 1 / 2)
  (hfg : f / g = 5 / 3)
  (hgh : g / h = 4 / 9)
  (h_nonzero : b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0) :
  a * b * c * d / (e * f * g * h) = 1 / 97 := by
  sorry

end NUMINAMATH_CALUDE_abcd_efgh_ratio_l3582_358285


namespace NUMINAMATH_CALUDE_digitSum5_125th_l3582_358278

/-- The sum of digits of a natural number --/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The sequence of natural numbers with digit sum 5 in ascending order --/
def digitSum5Seq : ℕ → ℕ := sorry

/-- The 125th number in the sequence of natural numbers with digit sum 5 --/
theorem digitSum5_125th :
  digitSum5Seq 125 = 41000 ∧ sumOfDigits (digitSum5Seq 125) = 5 := by sorry

end NUMINAMATH_CALUDE_digitSum5_125th_l3582_358278


namespace NUMINAMATH_CALUDE_girls_total_distance_l3582_358286

/-- The number of laps run by the boys -/
def boys_laps : ℕ := 27

/-- The number of additional laps run by the girls compared to the boys -/
def extra_girls_laps : ℕ := 9

/-- The length of each boy's lap in miles -/
def boys_lap_length : ℚ := 3/4

/-- The length of the first type of girl's lap in miles -/
def girls_lap_length1 : ℚ := 3/4

/-- The length of the second type of girl's lap in miles -/
def girls_lap_length2 : ℚ := 7/8

/-- The total number of laps run by the girls -/
def girls_laps : ℕ := boys_laps + extra_girls_laps

/-- The number of laps of each type run by the girls -/
def girls_laps_each_type : ℕ := girls_laps / 2

theorem girls_total_distance :
  girls_laps_each_type * girls_lap_length1 + girls_laps_each_type * girls_lap_length2 = 29.25 := by
  sorry

end NUMINAMATH_CALUDE_girls_total_distance_l3582_358286


namespace NUMINAMATH_CALUDE_symmetry_implies_values_l3582_358267

-- Define the two linear functions
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 2
def g (b : ℝ) (x : ℝ) : ℝ := 3 * x - b

-- Define the symmetry condition about y = x
def symmetric (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- State the theorem
theorem symmetry_implies_values :
  ∃ a b : ℝ, symmetric (f a) (g b) → a = 1/3 ∧ b = 6 :=
sorry

end NUMINAMATH_CALUDE_symmetry_implies_values_l3582_358267


namespace NUMINAMATH_CALUDE_smallest_multiple_1_to_10_l3582_358219

theorem smallest_multiple_1_to_10 : ∀ n : ℕ, n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) → n ≥ 2520 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_1_to_10_l3582_358219


namespace NUMINAMATH_CALUDE_true_discount_calculation_l3582_358238

/-- Given a present worth and banker's discount, calculate the true discount -/
theorem true_discount_calculation (present_worth banker_discount : ℚ) :
  present_worth = 400 →
  banker_discount = 21 →
  ∃ true_discount : ℚ,
    banker_discount = true_discount + (true_discount * banker_discount / present_worth) ∧
    true_discount = 8400 / 421 :=
by sorry

end NUMINAMATH_CALUDE_true_discount_calculation_l3582_358238


namespace NUMINAMATH_CALUDE_binomial_8_choose_5_l3582_358296

theorem binomial_8_choose_5 : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_choose_5_l3582_358296


namespace NUMINAMATH_CALUDE_donation_to_second_home_l3582_358276

theorem donation_to_second_home 
  (total_donation : ℝ)
  (first_home : ℝ)
  (third_home : ℝ)
  (h1 : total_donation = 700)
  (h2 : first_home = 245)
  (h3 : third_home = 230) :
  total_donation - first_home - third_home = 225 :=
by sorry

end NUMINAMATH_CALUDE_donation_to_second_home_l3582_358276


namespace NUMINAMATH_CALUDE_tan_plus_cot_l3582_358263

theorem tan_plus_cot (α : Real) : 
  sinα - cosα = -Real.sqrt 5 / 2 → tanα + 1 / tanα = -8 := by
  sorry

end NUMINAMATH_CALUDE_tan_plus_cot_l3582_358263


namespace NUMINAMATH_CALUDE_nail_decoration_time_l3582_358274

def base_coat_time : ℕ := 20
def paint_coat_time : ℕ := 20
def glitter_coat_time : ℕ := 20
def drying_time : ℕ := 20
def pattern_time : ℕ := 40

def total_decoration_time : ℕ :=
  base_coat_time + drying_time +
  paint_coat_time + drying_time +
  glitter_coat_time + drying_time +
  pattern_time

theorem nail_decoration_time :
  total_decoration_time = 160 :=
by sorry

end NUMINAMATH_CALUDE_nail_decoration_time_l3582_358274


namespace NUMINAMATH_CALUDE_quadratic_function_ordering_l3582_358241

/-- A quadratic function with the given properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0
  symmetry : ∀ x : ℝ, a * (2 + x)^2 + b * (2 + x) + c = a * (2 - x)^2 + b * (2 - x) + c

/-- The theorem stating the ordering of function values -/
theorem quadratic_function_ordering (f : QuadraticFunction) :
  f.a * 2^2 + f.b * 2 + f.c < f.a * 1^2 + f.b * 1 + f.c ∧
  f.a * 1^2 + f.b * 1 + f.c < f.a * 4^2 + f.b * 4 + f.c :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_ordering_l3582_358241


namespace NUMINAMATH_CALUDE_only_cone_no_rectangular_cross_section_l3582_358251

-- Define the geometric solids
inductive GeometricSolid
  | Cylinder
  | Cone
  | RectangularPrism
  | Cube

-- Define a function that checks if a solid can have a rectangular cross-section
def canHaveRectangularCrossSection (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.Cylinder => true
  | GeometricSolid.Cone => false
  | GeometricSolid.RectangularPrism => true
  | GeometricSolid.Cube => true

-- Theorem stating that only a cone cannot have a rectangular cross-section
theorem only_cone_no_rectangular_cross_section :
  ∀ (solid : GeometricSolid),
    ¬(canHaveRectangularCrossSection solid) ↔ solid = GeometricSolid.Cone :=
by sorry

end NUMINAMATH_CALUDE_only_cone_no_rectangular_cross_section_l3582_358251


namespace NUMINAMATH_CALUDE_urn_has_eleven_marbles_l3582_358223

/-- Represents the number of marbles in an urn -/
structure Urn where
  green : ℕ
  yellow : ℕ

/-- The conditions of the marble problem -/
def satisfies_conditions (u : Urn) : Prop :=
  (4 * (u.green - 3) = u.green + u.yellow - 3) ∧
  (3 * u.green = u.green + u.yellow - 4)

/-- The theorem stating that an urn satisfying the conditions has 11 marbles -/
theorem urn_has_eleven_marbles (u : Urn) 
  (h : satisfies_conditions u) : u.green + u.yellow = 11 := by
  sorry

#check urn_has_eleven_marbles

end NUMINAMATH_CALUDE_urn_has_eleven_marbles_l3582_358223


namespace NUMINAMATH_CALUDE_subset_range_l3582_358259

theorem subset_range (a : ℝ) : 
  let A := {x : ℝ | 1 ≤ x ∧ x ≤ a}
  let B := {x : ℝ | 0 < x ∧ x < 5}
  A ⊆ B → (1 ≤ a ∧ a < 5) :=
by
  sorry

end NUMINAMATH_CALUDE_subset_range_l3582_358259


namespace NUMINAMATH_CALUDE_product_equals_sum_solution_l3582_358293

theorem product_equals_sum_solution (x y : ℝ) (h1 : x * y = x + y) (h2 : y ≠ 1) :
  x = y / (y - 1) := by
  sorry

end NUMINAMATH_CALUDE_product_equals_sum_solution_l3582_358293


namespace NUMINAMATH_CALUDE_train_speed_l3582_358248

theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 300) (h2 : time = 30) :
  length / time = 10 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3582_358248


namespace NUMINAMATH_CALUDE_sum_of_max_min_a_l3582_358235

theorem sum_of_max_min_a (a : ℝ) : 
  (∀ x y : ℝ, x^2 - a*x - 20*a^2 < 0 ∧ y^2 - a*y - 20*a^2 < 0 → |x - y| ≤ 9) →
  ∃ a_min a_max : ℝ, 
    (∀ a' : ℝ, (∃ x : ℝ, x^2 - a'*x - 20*a'^2 < 0) → a_min ≤ a' ∧ a' ≤ a_max) ∧
    a_min + a_max = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_max_min_a_l3582_358235


namespace NUMINAMATH_CALUDE_candy_sharing_theorem_l3582_358242

/-- Represents the amount of candy each person has initially -/
structure CandyDistribution where
  hugh : ℕ
  tommy : ℕ
  melany : ℕ

/-- Calculates the amount of candy each person gets when shared equally -/
def equalShare (dist : CandyDistribution) : ℕ :=
  (dist.hugh + dist.tommy + dist.melany) / 3

/-- Theorem: When Hugh has 8 pounds, Tommy has 6 pounds, and Melany has 7 pounds of candy,
    sharing equally results in each person having 7 pounds of candy -/
theorem candy_sharing_theorem (dist : CandyDistribution) 
  (h1 : dist.hugh = 8) 
  (h2 : dist.tommy = 6) 
  (h3 : dist.melany = 7) : 
  equalShare dist = 7 := by
  sorry

end NUMINAMATH_CALUDE_candy_sharing_theorem_l3582_358242


namespace NUMINAMATH_CALUDE_blue_marble_probability_l3582_358226

theorem blue_marble_probability (total : ℕ) (yellow : ℕ) :
  total = 60 →
  yellow = 20 →
  let green := yellow / 2
  let remaining := total - yellow - green
  let blue := remaining / 2
  (blue : ℚ) / total * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_blue_marble_probability_l3582_358226


namespace NUMINAMATH_CALUDE_popsicle_sticks_problem_l3582_358281

theorem popsicle_sticks_problem (steve sid sam : ℕ) : 
  sid = 2 * steve →
  sam = 3 * sid →
  steve + sid + sam = 108 →
  steve = 12 := by
sorry

end NUMINAMATH_CALUDE_popsicle_sticks_problem_l3582_358281


namespace NUMINAMATH_CALUDE_special_triangle_perimeter_l3582_358204

/-- A triangle with sides satisfying x^2 - 6x + 8 = 0 --/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a^2 - 6*a + 8 = 0
  hb : b^2 - 6*b + 8 = 0
  hc : c^2 - 6*c + 8 = 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The set of possible perimeters for the special triangle --/
def possible_perimeters : Set ℝ := {6, 10, 12}

/-- Theorem stating that the perimeter of a SpecialTriangle is in the set of possible perimeters --/
theorem special_triangle_perimeter (t : SpecialTriangle) : 
  (t.a + t.b + t.c) ∈ possible_perimeters := by
  sorry

#check special_triangle_perimeter

end NUMINAMATH_CALUDE_special_triangle_perimeter_l3582_358204


namespace NUMINAMATH_CALUDE_uniform_motion_final_position_l3582_358233

/-- A point moving with uniform velocity in a 2D plane. -/
structure MovingPoint where
  initial_position : ℝ × ℝ
  velocity : ℝ × ℝ

/-- Calculate the final position of a moving point after a given time. -/
def final_position (p : MovingPoint) (t : ℝ) : ℝ × ℝ :=
  (p.initial_position.1 + t * p.velocity.1, p.initial_position.2 + t * p.velocity.2)

theorem uniform_motion_final_position :
  let p : MovingPoint := { initial_position := (-10, 10), velocity := (4, -3) }
  final_position p 5 = (10, -5) := by
  sorry

end NUMINAMATH_CALUDE_uniform_motion_final_position_l3582_358233


namespace NUMINAMATH_CALUDE_inequality_system_no_solution_l3582_358270

theorem inequality_system_no_solution : 
  ∀ x : ℝ, ¬(2 * x^2 - 5 * x + 3 < 0 ∧ (x - 1) / (2 - x) ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_no_solution_l3582_358270


namespace NUMINAMATH_CALUDE_coconut_grove_problem_l3582_358268

/-- Coconut grove problem -/
theorem coconut_grove_problem (x : ℝ) : 
  (60 * (x + 4) + 120 * x + 180 * (x - 4)) / (3 * x) = 100 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_coconut_grove_problem_l3582_358268


namespace NUMINAMATH_CALUDE_x_axis_reflection_l3582_358255

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

theorem x_axis_reflection :
  let p : ℝ × ℝ := (-3, 5)
  reflect_x p = (-3, -5) := by sorry

end NUMINAMATH_CALUDE_x_axis_reflection_l3582_358255


namespace NUMINAMATH_CALUDE_lcm_of_21_and_12_l3582_358272

theorem lcm_of_21_and_12 (h : Nat.gcd 21 12 = 6) : Nat.lcm 21 12 = 42 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_21_and_12_l3582_358272


namespace NUMINAMATH_CALUDE_arithmetic_operations_l3582_358213

theorem arithmetic_operations : 
  (-3 + 5 - (-2) = 4) ∧ 
  (-6 / (1/4) * (-4) = 96) ∧ 
  ((5/6 - 3/4 + 1/3) * (-24) = -10) ∧ 
  ((-1)^2023 - (4 - (-3)^2) / (2/7 - 1) = -8) := by
  sorry

#check arithmetic_operations

end NUMINAMATH_CALUDE_arithmetic_operations_l3582_358213


namespace NUMINAMATH_CALUDE_line_circle_intersection_l3582_358275

theorem line_circle_intersection (k : ℝ) : 
  ∃ (x y : ℝ), y = k * x + 1 ∧ x^2 + y^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l3582_358275


namespace NUMINAMATH_CALUDE_strawberry_jelly_amount_l3582_358212

/-- The total amount of jelly in grams -/
def total_jelly : ℕ := 6310

/-- The amount of blueberry jelly in grams -/
def blueberry_jelly : ℕ := 4518

/-- The amount of strawberry jelly in grams -/
def strawberry_jelly : ℕ := total_jelly - blueberry_jelly

theorem strawberry_jelly_amount : strawberry_jelly = 1792 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_jelly_amount_l3582_358212


namespace NUMINAMATH_CALUDE_sum_of_fractions_in_base_10_l3582_358265

/-- Convert a number from base b to base 10 -/
def toBase10 (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Express a fraction in base 10 given numerator and denominator in different bases -/
def fractionToBase10 (num : ℕ) (num_base : ℕ) (den : ℕ) (den_base : ℕ) : ℚ := sorry

/-- Main theorem: The integer part of the sum of the given fractions in base 10 is 29 -/
theorem sum_of_fractions_in_base_10 : 
  ⌊(fractionToBase10 254 8 13 4 + fractionToBase10 132 5 22 3)⌋ = 29 := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_in_base_10_l3582_358265


namespace NUMINAMATH_CALUDE_hockey_season_games_l3582_358239

/-- Calculate the number of games in a hockey season -/
theorem hockey_season_games (n : ℕ) (m : ℕ) (h1 : n = 16) (h2 : m = 10) :
  (n * (n - 1) / 2) * m = 2400 := by
  sorry

end NUMINAMATH_CALUDE_hockey_season_games_l3582_358239


namespace NUMINAMATH_CALUDE_exam_marks_category_c_l3582_358266

theorem exam_marks_category_c (total_candidates : ℕ) 
                               (category_a_count : ℕ) 
                               (category_b_count : ℕ) 
                               (category_c_count : ℕ) 
                               (category_a_avg : ℕ) 
                               (category_b_avg : ℕ) 
                               (category_c_avg : ℕ) : 
  total_candidates = 80 →
  category_a_count = 30 →
  category_b_count = 25 →
  category_c_count = 25 →
  category_a_avg = 35 →
  category_b_avg = 42 →
  category_c_avg = 46 →
  category_c_count * category_c_avg = 1150 :=
by sorry

end NUMINAMATH_CALUDE_exam_marks_category_c_l3582_358266


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3582_358280

/-- Proves the equality of two polynomial expressions -/
theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + x^5 + 3 * x^4 + x^3 + 8) - (x^6 + 2 * x^5 - 2 * x^4 + x^2 + 5) =
  x^6 - x^5 + 5 * x^4 + x^3 - x^2 + 3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3582_358280


namespace NUMINAMATH_CALUDE_not_integer_fraction_l3582_358244

theorem not_integer_fraction (a b : ℤ) : ¬ (∃ (k : ℤ), (a^2 + b^2) = k * (a^2 - b^2)) :=
by
  sorry

end NUMINAMATH_CALUDE_not_integer_fraction_l3582_358244


namespace NUMINAMATH_CALUDE_integer_roots_quadratic_l3582_358231

theorem integer_roots_quadratic (n : ℤ) : 
  (∃ x y : ℤ, x^2 + (n+1)*x + 2*n - 1 = 0 ∧ y^2 + (n+1)*y + 2*n - 1 = 0) → 
  (n = 1 ∨ n = 5) := by
sorry

end NUMINAMATH_CALUDE_integer_roots_quadratic_l3582_358231


namespace NUMINAMATH_CALUDE_consecutive_odd_power_sum_divisible_l3582_358205

-- Define consecutive odd numbers
def ConsecutiveOddNumbers (a b : ℕ) : Prop :=
  ∃ k : ℕ, a = 2*k + 1 ∧ b = 2*k + 3

-- Define divisibility
def Divides (d n : ℕ) : Prop := ∃ k : ℕ, n = d * k

-- Theorem statement
theorem consecutive_odd_power_sum_divisible (a b : ℕ) :
  ConsecutiveOddNumbers a b → Divides (a + b) (a^b + b^a) :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_power_sum_divisible_l3582_358205


namespace NUMINAMATH_CALUDE_gmat_test_percentage_l3582_358222

theorem gmat_test_percentage (S B N : ℝ) : 
  S = 70 → B = 60 → N = 5 → 100 - S + B - N = 85 :=
sorry

end NUMINAMATH_CALUDE_gmat_test_percentage_l3582_358222


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l3582_358227

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (planePerp : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m n : Line) (α β : Plane) :
  perpendicular m α → 
  parallel m n → 
  contains β n → 
  planePerp α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l3582_358227


namespace NUMINAMATH_CALUDE_probability_four_of_each_color_l3582_358221

-- Define the number of balls
def n : ℕ := 8

-- Define the probability of painting a ball black or white
def p : ℚ := 1/2

-- Define the number of ways to choose 4 balls out of 8
def ways_to_choose : ℕ := Nat.choose n (n/2)

-- Define the probability of one specific arrangement
def prob_one_arrangement : ℚ := p^n

-- Statement to prove
theorem probability_four_of_each_color :
  ways_to_choose * prob_one_arrangement = 35/128 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_of_each_color_l3582_358221


namespace NUMINAMATH_CALUDE_vector_perpendicular_l3582_358237

/-- Given vectors m and n in ℝ², prove that if m + n is perpendicular to m - n, then t = -3 -/
theorem vector_perpendicular (t : ℝ) : 
  let m : Fin 2 → ℝ := ![t + 1, 1]
  let n : Fin 2 → ℝ := ![t + 2, 2]
  (m + n) • (m - n) = 0 → t = -3 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_l3582_358237


namespace NUMINAMATH_CALUDE_expression_value_at_negative_one_l3582_358240

theorem expression_value_at_negative_one :
  let x : ℤ := -1
  (x^2 + 5*x - 6) = -10 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_negative_one_l3582_358240


namespace NUMINAMATH_CALUDE_sum_18_29_base4_l3582_358287

/-- Converts a number from base 10 to base 4 -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 4 to a natural number -/
def fromBase4 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_18_29_base4 :
  toBase4 (18 + 29) = [2, 3, 3] :=
sorry

end NUMINAMATH_CALUDE_sum_18_29_base4_l3582_358287


namespace NUMINAMATH_CALUDE_acid_dilution_l3582_358209

theorem acid_dilution (m : ℝ) (x : ℝ) (h : m > 25) :
  (m * m / 100 = (m - 5) / 100 * (m + x)) → x = 5 * m / (m - 5) := by
  sorry

end NUMINAMATH_CALUDE_acid_dilution_l3582_358209


namespace NUMINAMATH_CALUDE_factorial_difference_l3582_358261

theorem factorial_difference : (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) - 
                               (8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) - 
                               (7 * 6 * 5 * 4 * 3 * 2 * 1) + 
                               (6 * 5 * 4 * 3 * 2 * 1) = 318240 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l3582_358261


namespace NUMINAMATH_CALUDE_expected_winnings_l3582_358284

-- Define the spinner outcomes
inductive Outcome
| Green
| Red
| Blue

-- Define the probability function
def probability (o : Outcome) : ℚ :=
  match o with
  | Outcome.Green => 1/4
  | Outcome.Red => 1/2
  | Outcome.Blue => 1/4

-- Define the winnings function
def winnings (o : Outcome) : ℤ :=
  match o with
  | Outcome.Green => 2
  | Outcome.Red => 4
  | Outcome.Blue => -6

-- Define the expected value function
def expectedValue : ℚ :=
  (probability Outcome.Green * winnings Outcome.Green) +
  (probability Outcome.Red * winnings Outcome.Red) +
  (probability Outcome.Blue * winnings Outcome.Blue)

-- Theorem stating the expected winnings
theorem expected_winnings : expectedValue = 1 := by
  sorry

end NUMINAMATH_CALUDE_expected_winnings_l3582_358284


namespace NUMINAMATH_CALUDE_exam_score_l3582_358245

theorem exam_score (total_questions : ℕ) (correct_answers : ℕ) 
  (marks_per_correct : ℕ) (marks_lost_per_wrong : ℕ) : 
  total_questions = 60 →
  correct_answers = 38 →
  marks_per_correct = 4 →
  marks_lost_per_wrong = 1 →
  (correct_answers * marks_per_correct) - 
  ((total_questions - correct_answers) * marks_lost_per_wrong) = 130 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_score_l3582_358245


namespace NUMINAMATH_CALUDE_problem_solution_l3582_358250

theorem problem_solution : (120 / (6 / 3)) - 15 = 45 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3582_358250


namespace NUMINAMATH_CALUDE_negation_of_existential_l3582_358229

theorem negation_of_existential (p : Prop) :
  (¬∃ (x : ℝ), x = Real.sin x) ↔ (∀ (x : ℝ), x ≠ Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_l3582_358229


namespace NUMINAMATH_CALUDE_x_geq_y_l3582_358224

theorem x_geq_y (a : ℝ) : 2 * a * (a + 3) ≥ (a - 3) * (a + 3) := by
  sorry

end NUMINAMATH_CALUDE_x_geq_y_l3582_358224


namespace NUMINAMATH_CALUDE_factorization_of_5_power_1985_minus_1_l3582_358258

theorem factorization_of_5_power_1985_minus_1 :
  ∃ (a b c : ℤ),
    (5^1985 - 1 : ℤ) = a * b * c ∧
    a > 5^100 ∧
    b > 5^100 ∧
    c > 5^100 ∧
    a = 5^397 - 1 ∧
    b = 5^794 - 5^596 + 3*5^397 - 5^199 + 1 ∧
    c = 5^794 + 5^596 + 3*5^397 + 5^199 + 1 :=
by sorry

end NUMINAMATH_CALUDE_factorization_of_5_power_1985_minus_1_l3582_358258


namespace NUMINAMATH_CALUDE_heather_biking_days_l3582_358247

/-- Given that Heather bicycled 40.0 kilometers per day for some days and 320 kilometers in total,
    prove that the number of days she biked is 8. -/
theorem heather_biking_days (daily_distance : ℝ) (total_distance : ℝ) 
    (h1 : daily_distance = 40.0)
    (h2 : total_distance = 320) :
    total_distance / daily_distance = 8 := by
  sorry

end NUMINAMATH_CALUDE_heather_biking_days_l3582_358247


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l3582_358207

theorem quadratic_equation_solutions :
  {x : ℝ | x^2 - Real.sqrt 2 * x = 0} = {0, Real.sqrt 2} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l3582_358207


namespace NUMINAMATH_CALUDE_distinct_roots_condition_root_condition_l3582_358220

-- Define the quadratic equation
def quadratic (x k : ℝ) : ℝ := x^2 + 2*x + k

-- Theorem for part 1
theorem distinct_roots_condition (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic x k = 0 ∧ quadratic y k = 0) ↔ k < 1 :=
sorry

-- Theorem for part 2
theorem root_condition (m k : ℝ) :
  quadratic m k = 0 ∧ m^2 + 2*m = 2 → k = -2 :=
sorry

end NUMINAMATH_CALUDE_distinct_roots_condition_root_condition_l3582_358220


namespace NUMINAMATH_CALUDE_expression_evaluation_l3582_358290

theorem expression_evaluation : 
  ((-1) ^ 2022) + |1 - Real.sqrt 2| + ((-27) ^ (1/3 : ℝ)) - Real.sqrt (((-2) ^ 2)) = Real.sqrt 2 - 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3582_358290


namespace NUMINAMATH_CALUDE_fraction_evaluation_l3582_358208

theorem fraction_evaluation :
  let x : ℚ := 4/3
  let y : ℚ := 5/7
  (3*x + 7*y) / (21*x*y) = 9/140 := by
sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3582_358208


namespace NUMINAMATH_CALUDE_number_division_problem_l3582_358269

theorem number_division_problem :
  ∃ x : ℚ, (x / 5 = 80 + x / 6) → x = 2400 := by
sorry

end NUMINAMATH_CALUDE_number_division_problem_l3582_358269


namespace NUMINAMATH_CALUDE_march_greatest_drop_l3582_358256

-- Define the months
inductive Month
| January
| February
| March
| April
| May
| June

-- Define the price change for each month
def price_change (m : Month) : ℝ :=
  match m with
  | Month.January => -0.50
  | Month.February => 2.00
  | Month.March => -2.50
  | Month.April => 3.00
  | Month.May => -0.50
  | Month.June => -2.00

-- Define a function to check if a month has a price drop
def has_price_drop (m : Month) : Prop :=
  price_change m < 0

-- Define a function to compare price drops between two months
def greater_price_drop (m1 m2 : Month) : Prop :=
  has_price_drop m1 ∧ has_price_drop m2 ∧ price_change m1 < price_change m2

-- Theorem statement
theorem march_greatest_drop :
  ∀ m : Month, m ≠ Month.March → ¬(greater_price_drop m Month.March) :=
by sorry

end NUMINAMATH_CALUDE_march_greatest_drop_l3582_358256


namespace NUMINAMATH_CALUDE_inequality_condition_l3582_358252

theorem inequality_condition (m : ℝ) : 
  (∀ x : ℝ, (m - 1) * x^2 + (m - 1) * x - 1 < 0) → 
  (-3 < m ∧ m < 1) ∧ 
  ¬(∀ m : ℝ, (∀ x : ℝ, (m - 1) * x^2 + (m - 1) * x - 1 < 0) → (-3 < m ∧ m < 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l3582_358252


namespace NUMINAMATH_CALUDE_tv_show_episodes_l3582_358217

/-- Proves that a TV show with given conditions has 20 episodes per season in its first half -/
theorem tv_show_episodes (total_seasons : ℕ) (second_half_episodes : ℕ) (total_episodes : ℕ) :
  total_seasons = 10 →
  second_half_episodes = 25 →
  total_episodes = 225 →
  (total_seasons / 2 : ℕ) * second_half_episodes + (total_seasons / 2 : ℕ) * (total_episodes - (total_seasons / 2 : ℕ) * second_half_episodes) / (total_seasons / 2 : ℕ) = total_episodes →
  (total_episodes - (total_seasons / 2 : ℕ) * second_half_episodes) / (total_seasons / 2 : ℕ) = 20 :=
by sorry

end NUMINAMATH_CALUDE_tv_show_episodes_l3582_358217


namespace NUMINAMATH_CALUDE_prime_square_plus_twelve_mod_twelve_l3582_358289

theorem prime_square_plus_twelve_mod_twelve (p : ℕ) (h_prime : Nat.Prime p) (h_gt_three : p > 3) :
  (p^2 + 12) % 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_plus_twelve_mod_twelve_l3582_358289


namespace NUMINAMATH_CALUDE_union_equals_interval_l3582_358292

def A : Set ℝ := {1, 2, 3, 4}

def B (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}

theorem union_equals_interval (a : ℝ) :
  A ∪ B a = Set.Iic 5 → a = 5 := by sorry

end NUMINAMATH_CALUDE_union_equals_interval_l3582_358292


namespace NUMINAMATH_CALUDE_venus_meal_cost_is_35_l3582_358230

/-- The cost per meal at Venus Hall -/
def venus_meal_cost : ℚ := 35

/-- The room rental cost at Caesar's -/
def caesars_rental : ℚ := 800

/-- The cost per meal at Caesar's -/
def caesars_meal_cost : ℚ := 30

/-- The room rental cost at Venus Hall -/
def venus_rental : ℚ := 500

/-- The number of guests at which the total costs are equal -/
def num_guests : ℚ := 60

theorem venus_meal_cost_is_35 :
  caesars_rental + caesars_meal_cost * num_guests =
  venus_rental + venus_meal_cost * num_guests := by
  sorry

end NUMINAMATH_CALUDE_venus_meal_cost_is_35_l3582_358230


namespace NUMINAMATH_CALUDE_tensor_A_equals_result_l3582_358225

def A : Set ℕ := {0, 2, 3}

def tensor_op (S : Set ℕ) : Set ℕ := {x | ∃ a b, a ∈ S ∧ b ∈ S ∧ x = a + b}

theorem tensor_A_equals_result : tensor_op A = {0, 2, 3, 4, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_tensor_A_equals_result_l3582_358225


namespace NUMINAMATH_CALUDE_arrival_time_difference_l3582_358232

-- Define the distance to the park
def distance_to_park : ℝ := 3

-- Define Jack's speed
def jack_speed : ℝ := 3

-- Define Jill's speed
def jill_speed : ℝ := 12

-- Define the conversion factor from hours to minutes
def hours_to_minutes : ℝ := 60

-- Theorem statement
theorem arrival_time_difference : 
  (distance_to_park / jack_speed - distance_to_park / jill_speed) * hours_to_minutes = 45 := by
  sorry

end NUMINAMATH_CALUDE_arrival_time_difference_l3582_358232


namespace NUMINAMATH_CALUDE_second_derivative_y_l3582_358297

noncomputable def x (t : ℝ) : ℝ := Real.log t
noncomputable def y (t : ℝ) : ℝ := Real.sin (2 * t)

theorem second_derivative_y (t : ℝ) (h : t > 0) :
  (deriv^[2] (y ∘ (x⁻¹))) (x t) = -4 * t^2 * Real.sin (2 * t) + 2 * t * Real.cos (2 * t) :=
by sorry

end NUMINAMATH_CALUDE_second_derivative_y_l3582_358297


namespace NUMINAMATH_CALUDE_a_6_equals_one_half_l3582_358200

def a (n : ℕ+) : ℚ := (3 * n.val - 2) / (2 ^ (n.val - 1))

theorem a_6_equals_one_half : a 6 = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_a_6_equals_one_half_l3582_358200


namespace NUMINAMATH_CALUDE_economics_problem_l3582_358253

def R (x : ℕ) : ℚ := x^2 + 16/x^2 + 40

def C (x : ℕ) : ℚ := 10*x + 40/x

def MC (x : ℕ) : ℚ := C (x+1) - C x

def z (x : ℕ) : ℚ := R x - C x

theorem economics_problem (x : ℕ) (h : 1 ≤ x ∧ x ≤ 10) :
  (∀ y : ℕ, 1 ≤ y ∧ y ≤ 10 → R y ≥ 72) ∧
  (∀ y : ℕ, 1 ≤ y ∧ y ≤ 9 → MC y ≤ 86/9) ∧
  (∀ y : ℕ, 1 ≤ y ∧ y ≤ 10 → z y ≥ 7) :=
by sorry

end NUMINAMATH_CALUDE_economics_problem_l3582_358253


namespace NUMINAMATH_CALUDE_batsman_average_l3582_358246

/-- Given a batsman whose average increases by 3 after scoring 66 runs in the 17th inning,
    his new average after the 17th inning is 18. -/
theorem batsman_average (prev_average : ℝ) : 
  (16 * prev_average + 66) / 17 = prev_average + 3 → prev_average + 3 = 18 := by
sorry

end NUMINAMATH_CALUDE_batsman_average_l3582_358246


namespace NUMINAMATH_CALUDE_smallest_reducible_fraction_l3582_358203

theorem smallest_reducible_fraction : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → ¬(∃ (k : ℕ), k > 1 ∧ k ∣ (m - 17) ∧ k ∣ (7 * m + 8))) ∧
  (∃ (k : ℕ), k > 1 ∧ k ∣ (n - 17) ∧ k ∣ (7 * n + 8)) ∧
  n = 144 := by
  sorry

end NUMINAMATH_CALUDE_smallest_reducible_fraction_l3582_358203


namespace NUMINAMATH_CALUDE_P_outside_triangle_l3582_358249

/-- The point P with coordinates (15.2, 12.4) -/
def P : ℝ × ℝ := (15.2, 12.4)

/-- The first line bounding the triangle: 8x - 15y - 35 = 0 -/
def line1 (x y : ℝ) : Prop := 8 * x - 15 * y - 35 = 0

/-- The second line bounding the triangle: x - 2y - 2 = 0 -/
def line2 (x y : ℝ) : Prop := x - 2 * y - 2 = 0

/-- The third line bounding the triangle: y = 0 -/
def line3 (y : ℝ) : Prop := y = 0

/-- The triangle bounded by the three lines -/
def triangle (x y : ℝ) : Prop := 
  (line1 x y ∨ line2 x y ∨ line3 y) ∧ 
  x ≥ 0 ∧ y ≥ 0 ∧ 8 * x - 15 * y - 35 ≤ 0 ∧ x - 2 * y - 2 ≤ 0

/-- Theorem stating that P is outside the triangle -/
theorem P_outside_triangle : ¬ triangle P.1 P.2 := by
  sorry

end NUMINAMATH_CALUDE_P_outside_triangle_l3582_358249


namespace NUMINAMATH_CALUDE_parallelogram_product_l3582_358298

-- Define the parallelogram EFGH
def EFGH (EF FG GH HE : ℝ) : Prop :=
  EF = GH ∧ FG = HE

-- Theorem statement
theorem parallelogram_product (x y : ℝ) :
  EFGH 47 (6 * y^2) (3 * x + 7) 27 →
  x * y = 20 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_product_l3582_358298


namespace NUMINAMATH_CALUDE_evaluate_expression_l3582_358291

theorem evaluate_expression : 
  (125 : ℝ) ^ (1/3) / (64 : ℝ) ^ (1/2) * (81 : ℝ) ^ (1/4) = 15/8 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3582_358291


namespace NUMINAMATH_CALUDE_increments_theorem_l3582_358273

/-- The function z(x, y) = xy -/
def z (x y : ℝ) : ℝ := x * y

/-- The initial point M₀ -/
def M₀ : ℝ × ℝ := (1, 2)

/-- The point M₁ -/
def M₁ : ℝ × ℝ := (1.1, 2)

/-- The point M₂ -/
def M₂ : ℝ × ℝ := (1, 1.9)

/-- The point M₃ -/
def M₃ : ℝ × ℝ := (1.1, 2.2)

/-- The increment of z from M₀ to another point -/
def increment (M : ℝ × ℝ) : ℝ := z M.1 M.2 - z M₀.1 M₀.2

theorem increments_theorem :
  increment M₁ = 0.2 ∧ increment M₂ = -0.1 ∧ increment M₃ = 0.42 := by
  sorry

end NUMINAMATH_CALUDE_increments_theorem_l3582_358273


namespace NUMINAMATH_CALUDE_total_books_l3582_358201

theorem total_books (sam_books joan_books : ℕ) 
  (h1 : sam_books = 110) 
  (h2 : joan_books = 102) : 
  sam_books + joan_books = 212 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l3582_358201


namespace NUMINAMATH_CALUDE_wrench_hammer_weight_ratio_l3582_358228

/-- Given that hammers and wrenches have uniform weights, prove that if the total weight of 2 hammers
    and 2 wrenches is 1/3 of the weight of 8 hammers and 5 wrenches, then the weight of one wrench
    is 2 times the weight of one hammer. -/
theorem wrench_hammer_weight_ratio 
  (h : ℝ) -- weight of one hammer
  (w : ℝ) -- weight of one wrench
  (h_pos : h > 0) -- hammer weight is positive
  (w_pos : w > 0) -- wrench weight is positive
  (weight_ratio : 2 * h + 2 * w = (1 / 3) * (8 * h + 5 * w)) -- given condition
  : w = 2 * h := by
  sorry

end NUMINAMATH_CALUDE_wrench_hammer_weight_ratio_l3582_358228


namespace NUMINAMATH_CALUDE_feet_to_inches_conversion_l3582_358279

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℕ := 12

/-- Initial height of the tree in feet -/
def initial_height : ℕ := 52

/-- Annual growth of the tree in feet -/
def annual_growth : ℕ := 5

/-- Time period in years -/
def time_period : ℕ := 8

/-- Theorem stating that the conversion factor from feet to inches is 12 -/
theorem feet_to_inches_conversion :
  feet_to_inches = 12 :=
sorry

end NUMINAMATH_CALUDE_feet_to_inches_conversion_l3582_358279


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_real_l3582_358216

theorem sqrt_x_minus_2_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_real_l3582_358216


namespace NUMINAMATH_CALUDE_bus_driver_worked_69_hours_l3582_358264

/-- Represents the payment structure and total compensation for a bus driver --/
structure BusDriverPayment where
  regular_rate : ℝ
  overtime_rate : ℝ
  double_overtime_rate : ℝ
  total_compensation : ℝ

/-- Calculates the total hours worked by a bus driver given their payment structure and total compensation --/
def calculate_total_hours (payment : BusDriverPayment) : ℕ :=
  sorry

/-- Theorem stating that given the specific payment structure and total compensation, the bus driver worked 69 hours --/
theorem bus_driver_worked_69_hours : 
  let payment := BusDriverPayment.mk 14 18.90 24.50 1230
  calculate_total_hours payment = 69 := by
  sorry

end NUMINAMATH_CALUDE_bus_driver_worked_69_hours_l3582_358264


namespace NUMINAMATH_CALUDE_harmonic_sum_increase_l3582_358277

theorem harmonic_sum_increase (k : ℕ) :
  (Finset.range (2^(k+1) - 1)).card - (Finset.range (2^k - 1)).card = 2^k := by
  sorry

end NUMINAMATH_CALUDE_harmonic_sum_increase_l3582_358277


namespace NUMINAMATH_CALUDE_smallest_m_satisfying_conditions_l3582_358214

theorem smallest_m_satisfying_conditions : ∃ m : ℕ,
  (100 ≤ m ∧ m ≤ 999) ∧
  (m + 6) % 9 = 0 ∧
  (m - 9) % 6 = 0 ∧
  (∀ n : ℕ, (100 ≤ n ∧ n < m ∧ (n + 6) % 9 = 0 ∧ (n - 9) % 6 = 0) → False) ∧
  m = 111 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_satisfying_conditions_l3582_358214


namespace NUMINAMATH_CALUDE_goods_train_speed_l3582_358295

theorem goods_train_speed
  (express_speed : ℝ)
  (head_start : ℝ)
  (catch_up_time : ℝ)
  (h1 : express_speed = 90)
  (h2 : head_start = 6)
  (h3 : catch_up_time = 4) :
  ∃ (goods_speed : ℝ),
    goods_speed * (head_start + catch_up_time) = express_speed * catch_up_time ∧
    goods_speed = 36 := by
  sorry

end NUMINAMATH_CALUDE_goods_train_speed_l3582_358295


namespace NUMINAMATH_CALUDE_equation_proof_l3582_358243

theorem equation_proof : 42 / (7 - 4/3) = 126/17 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l3582_358243


namespace NUMINAMATH_CALUDE_equation_transformation_l3582_358257

theorem equation_transformation (x : ℝ) : 3*(x+1) - 5*(1-x) = 3*x + 3 - 5 + 5*x := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l3582_358257


namespace NUMINAMATH_CALUDE_nine_point_four_minutes_in_seconds_l3582_358271

/-- Converts minutes to seconds -/
def minutes_to_seconds (minutes : ℝ) : ℝ :=
  minutes * 60

/-- Theorem stating that 9.4 minutes is equal to 564 seconds -/
theorem nine_point_four_minutes_in_seconds : 
  minutes_to_seconds 9.4 = 564 := by
  sorry

end NUMINAMATH_CALUDE_nine_point_four_minutes_in_seconds_l3582_358271


namespace NUMINAMATH_CALUDE_twentynine_is_perfect_number_pairing_x_squared_minus_6x_plus_13_perfect_number_condition_for_S_l3582_358260

-- Definition of a perfect number
def isPerfectNumber (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^2 ∧ a > 0 ∧ b > 0

-- Theorem 1
theorem twentynine_is_perfect_number : isPerfectNumber 29 :=
sorry

-- Theorem 2
theorem pairing_x_squared_minus_6x_plus_13 :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧
  (∀ x : ℝ, x^2 - 6*x + 13 = (x - m)^2 + n^2) ∧
  m * n = 6 :=
sorry

-- Theorem 3
theorem perfect_number_condition_for_S (k : ℝ) :
  (∀ x y : ℤ, ∃ a b : ℤ, x^2 + 4*y^2 + 4*x - 12*y + k = a^2 + b^2) ↔ k = 13 :=
sorry

end NUMINAMATH_CALUDE_twentynine_is_perfect_number_pairing_x_squared_minus_6x_plus_13_perfect_number_condition_for_S_l3582_358260


namespace NUMINAMATH_CALUDE_rainwater_solution_l3582_358202

/-- A tank collecting rainwater over three days -/
structure RainwaterTank where
  capacity : ℝ
  initialFill : ℝ
  day1Collection : ℝ
  day2Collection : ℝ
  day3Excess : ℝ

/-- The conditions of the rainwater collection problem -/
def rainProblem (tank : RainwaterTank) : Prop :=
  tank.capacity = 100 ∧
  tank.initialFill = 2/5 * tank.capacity ∧
  tank.day2Collection = tank.day1Collection + 5 ∧
  tank.initialFill + tank.day1Collection + tank.day2Collection = tank.capacity ∧
  tank.day3Excess = 25

/-- The theorem stating the solution to the rainwater problem -/
theorem rainwater_solution (tank : RainwaterTank) 
  (h : rainProblem tank) : tank.day1Collection = 27.5 := by
  sorry


end NUMINAMATH_CALUDE_rainwater_solution_l3582_358202
