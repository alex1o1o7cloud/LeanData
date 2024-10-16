import Mathlib

namespace NUMINAMATH_CALUDE_reassemble_squares_l511_51172

/-- Represents a square with a given side length -/
structure Square where
  side : ℝ

/-- Represents the original figure composed of two squares -/
structure OriginalFigure where
  square1 : Square
  square2 : Square

/-- Represents the final square after reassembly -/
structure FinalSquare where
  side : ℝ

/-- Theorem stating that the original figure can be reassembled into a square with side length 10 -/
theorem reassemble_squares (fig : OriginalFigure) (final : FinalSquare) :
  fig.square1.side = 8 ∧ 
  fig.square2.side = 6 ∧ 
  final.side = 10 →
  fig.square1.side ^ 2 + fig.square2.side ^ 2 = final.side ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_reassemble_squares_l511_51172


namespace NUMINAMATH_CALUDE_factor_sum_problem_l511_51147

theorem factor_sum_problem (N : ℕ) 
  (h1 : N > 0)
  (h2 : ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a ∣ N ∧ b ∣ N ∧ a + b = 4 ∧ ∀ (x : ℕ), x > 0 → x ∣ N → x ≥ a ∧ x ≥ b)
  (h3 : ∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ c ∣ N ∧ d ∣ N ∧ c + d = 204 ∧ ∀ (x : ℕ), x > 0 → x ∣ N → x ≤ c ∧ x ≤ d) :
  N = 153 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_problem_l511_51147


namespace NUMINAMATH_CALUDE_tank_capacity_l511_51122

/-- Represents the tank filling problem --/
def TankFilling (fill_rate_A fill_rate_B drain_rate_C : ℝ) 
                (time_A time_B time_C : ℝ) 
                (total_time : ℝ) : Prop :=
  let cycle_time := time_A + time_B + time_C
  let net_fill_per_cycle := fill_rate_A * time_A + fill_rate_B * time_B - drain_rate_C * time_C
  let num_cycles := total_time / cycle_time
  num_cycles * net_fill_per_cycle = 1000

/-- The tank capacity is 1000 L given the specified rates and times --/
theorem tank_capacity :
  TankFilling 200 50 25 1 2 2 20 :=
sorry

end NUMINAMATH_CALUDE_tank_capacity_l511_51122


namespace NUMINAMATH_CALUDE_factory_production_excess_l511_51154

theorem factory_production_excess (monthly_plan : ℝ) :
  let january_production := 1.05 * monthly_plan
  let february_production := 1.04 * january_production
  let two_month_plan := 2 * monthly_plan
  let total_production := january_production + february_production
  (total_production - two_month_plan) / two_month_plan = 0.071 := by
sorry

end NUMINAMATH_CALUDE_factory_production_excess_l511_51154


namespace NUMINAMATH_CALUDE_equation_transformation_l511_51114

theorem equation_transformation (a b : ℝ) : 
  (∀ x, x^2 - 6*x - 5 = 0 ↔ (x + a)^2 = b) → a + b = 11 := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l511_51114


namespace NUMINAMATH_CALUDE_happy_valley_kennel_arrangement_l511_51153

def num_chickens : Nat := 5
def num_dogs : Nat := 2
def num_cats : Nat := 5
def num_rabbits : Nat := 3
def total_animals : Nat := num_chickens + num_dogs + num_cats + num_rabbits

def animal_types : Nat := 4

theorem happy_valley_kennel_arrangement :
  (animal_types.factorial * num_chickens.factorial * num_dogs.factorial * 
   num_cats.factorial * num_rabbits.factorial) = 4147200 := by
  sorry

end NUMINAMATH_CALUDE_happy_valley_kennel_arrangement_l511_51153


namespace NUMINAMATH_CALUDE_dispersion_measures_l511_51102

-- Define a sample as a list of real numbers
def Sample := List Real

-- Define statistics
def standardDeviation (s : Sample) : Real :=
  sorry

def median (s : Sample) : Real :=
  sorry

def range (s : Sample) : Real :=
  sorry

def mean (s : Sample) : Real :=
  sorry

-- Define a predicate for measures of dispersion
def measuresDispersion (f : Sample → Real) : Prop :=
  sorry

-- Theorem statement
theorem dispersion_measures (s : Sample) :
  measuresDispersion (standardDeviation) ∧
  measuresDispersion (range) ∧
  ¬measuresDispersion (median) ∧
  ¬measuresDispersion (mean) :=
sorry

end NUMINAMATH_CALUDE_dispersion_measures_l511_51102


namespace NUMINAMATH_CALUDE_rectangle_sides_from_ratio_and_area_l511_51117

theorem rectangle_sides_from_ratio_and_area 
  (m n S : ℝ) (hm : m > 0) (hn : n > 0) (hS : S > 0) :
  ∃ (x y : ℝ), 
    x / y = m / n ∧ 
    x * y = S ∧ 
    x = Real.sqrt ((m * S) / n) ∧ 
    y = Real.sqrt ((n * S) / m) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_sides_from_ratio_and_area_l511_51117


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l511_51164

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 1) - 1
  f (-1) = 0 ∧ ∀ x : ℝ, f x = 0 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l511_51164


namespace NUMINAMATH_CALUDE_stratified_sample_size_l511_51129

/-- Represents the composition of a population --/
structure Population where
  elderly : Nat
  middleAged : Nat
  young : Nat

/-- Represents a stratified sample --/
structure StratifiedSample where
  population : Population
  sampleSize : Nat
  youngInSample : Nat

/-- Theorem stating the relationship between the sample size and the number of young people in the sample --/
theorem stratified_sample_size 
  (sample : StratifiedSample) 
  (h1 : sample.population = { elderly := 20, middleAged := 120, young := 100 })
  (h2 : sample.youngInSample = 10) : 
  sample.sampleSize = 24 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l511_51129


namespace NUMINAMATH_CALUDE_completing_square_l511_51115

theorem completing_square (x : ℝ) : x^2 - 4*x = 6 ↔ (x - 2)^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_l511_51115


namespace NUMINAMATH_CALUDE_ken_released_three_fish_l511_51124

/-- The number of fish Ken released -/
def fish_released (ken_caught : ℕ) (kendra_caught : ℕ) (brought_home : ℕ) : ℕ :=
  ken_caught + kendra_caught - brought_home

theorem ken_released_three_fish :
  ∀ (ken_caught kendra_caught brought_home : ℕ),
  ken_caught = 2 * kendra_caught →
  kendra_caught = 30 →
  brought_home = 87 →
  fish_released ken_caught kendra_caught brought_home = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ken_released_three_fish_l511_51124


namespace NUMINAMATH_CALUDE_max_value_theorem_l511_51101

theorem max_value_theorem (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) 
  (h_sum : a^2 + b^2 + c^2 = 1) : 
  2*a*b + 2*b*c*Real.sqrt 2 + 2*a*c ≤ 2*(1 + Real.sqrt 2)/3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l511_51101


namespace NUMINAMATH_CALUDE_race_cars_l511_51186

theorem race_cars (p_x p_y p_z p_total : ℚ) : 
  p_x = 1/8 → p_y = 1/12 → p_z = 1/6 → p_total = 375/1000 → 
  p_x + p_y + p_z = p_total → 
  ∀ p_other : ℚ, p_other ≥ 0 → p_x + p_y + p_z + p_other = p_total → p_other = 0 := by
  sorry

end NUMINAMATH_CALUDE_race_cars_l511_51186


namespace NUMINAMATH_CALUDE_no_roots_of_equation_l511_51132

theorem no_roots_of_equation (x : ℝ) (h : x ≠ 4) :
  ¬∃x, x - 9 / (x - 4) = 4 - 9 / (x - 4) :=
sorry

end NUMINAMATH_CALUDE_no_roots_of_equation_l511_51132


namespace NUMINAMATH_CALUDE_mei_fruit_baskets_l511_51107

theorem mei_fruit_baskets : Nat.gcd 15 (Nat.gcd 9 18) = 3 := by
  sorry

end NUMINAMATH_CALUDE_mei_fruit_baskets_l511_51107


namespace NUMINAMATH_CALUDE_not_all_cells_tetraploid_l511_51127

/-- Represents a watermelon plant --/
structure WatermelonPlant where
  /-- The number of chromosome sets in somatic cells --/
  somaticChromosomeSets : ℕ
  /-- The number of chromosome sets in root cells --/
  rootChromosomeSets : ℕ

/-- Represents the process of culturing and treating watermelon plants --/
def cultureAndTreat (original : WatermelonPlant) : WatermelonPlant :=
  { somaticChromosomeSets := 2 * original.somaticChromosomeSets,
    rootChromosomeSets := original.rootChromosomeSets }

/-- Theorem: Not all cells in a watermelon plant obtained from treating diploid seedlings
    with colchicine contain four sets of chromosomes --/
theorem not_all_cells_tetraploid (original : WatermelonPlant)
    (h_diploid : original.somaticChromosomeSets = 2)
    (h_root_untreated : (cultureAndTreat original).rootChromosomeSets = original.rootChromosomeSets) :
    ∃ (cell_type : WatermelonPlant → ℕ),
      cell_type (cultureAndTreat original) ≠ 4 :=
  sorry


end NUMINAMATH_CALUDE_not_all_cells_tetraploid_l511_51127


namespace NUMINAMATH_CALUDE_campsite_distance_l511_51121

-- Define d as the distance to the nearest campsite
variable (d : ℝ)

-- Define the conditions based on the false statements
def paula_false : Prop := ¬(d ≥ 10)
def daniel_false : Prop := ¬(d ≤ 9)
def emily_false : Prop := ¬(d = 11)

-- Theorem to prove
theorem campsite_distance (h1 : paula_false d) (h2 : daniel_false d) (h3 : emily_false d) :
  d ∈ Set.Ioi 9 := by
  sorry

end NUMINAMATH_CALUDE_campsite_distance_l511_51121


namespace NUMINAMATH_CALUDE_no_m_exists_for_equality_m_range_for_subset_l511_51136

-- Define the sets P and S
def P : Set ℝ := {x | x^2 - 8*x - 20 ≤ 0}
def S (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

-- Theorem for question 1
theorem no_m_exists_for_equality :
  ¬ ∃ m : ℝ, P = S m :=
sorry

-- Theorem for question 2
theorem m_range_for_subset :
  {m : ℝ | P ⊆ S m} = {m : ℝ | m ≤ 3} :=
sorry

end NUMINAMATH_CALUDE_no_m_exists_for_equality_m_range_for_subset_l511_51136


namespace NUMINAMATH_CALUDE_unique_zero_point_condition_l511_51182

/-- The function f(x) = ax³ - 3x² + 2 has only one zero point if and only if a ∈ (-∞, -√2) ∪ (√2, +∞) -/
theorem unique_zero_point_condition (a : ℝ) :
  (∃! x, a * x^3 - 3 * x^2 + 2 = 0) ↔ a < -Real.sqrt 2 ∨ a > Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_zero_point_condition_l511_51182


namespace NUMINAMATH_CALUDE_no_solution_exists_l511_51138

/-- P(n) denotes the greatest prime factor of n -/
def greatest_prime_factor (n : ℕ) : ℕ := sorry

/-- Theorem: There are no positive integers n > 1 such that both 
    P(n) = √n and P(n+36) = √(n+36) -/
theorem no_solution_exists : ¬ ∃ (n : ℕ), n > 1 ∧ 
  (greatest_prime_factor n = Nat.sqrt n) ∧ 
  (greatest_prime_factor (n + 36) = Nat.sqrt (n + 36)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l511_51138


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l511_51181

/-- Given a sector with a central angle of 60° and an arc length of 2π,
    its inscribed circle has a radius of 2. -/
theorem inscribed_circle_radius (θ : ℝ) (arc_length : ℝ) (R : ℝ) (r : ℝ) :
  θ = π / 3 →
  arc_length = 2 * π →
  arc_length = θ * R →
  3 * r = R →
  r = 2 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l511_51181


namespace NUMINAMATH_CALUDE_sum_of_positive_numbers_l511_51119

theorem sum_of_positive_numbers (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  x * y = 30 → x * z = 60 → y * z = 90 → 
  x + y + z = 11 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_positive_numbers_l511_51119


namespace NUMINAMATH_CALUDE_g_3_equals_9_l511_51108

-- Define the function g
def g (x : ℝ) : ℝ := 3*x^6 - 2*x^4 + 5*x^2 - 7

-- Theorem statement
theorem g_3_equals_9 (h : g (-3) = 9) : g 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_g_3_equals_9_l511_51108


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l511_51190

theorem unique_solution_for_equation (m n : ℕ+) : 
  (m : ℤ)^(n : ℕ) - (n : ℤ)^(m : ℕ) = 3 ↔ m = 4 ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l511_51190


namespace NUMINAMATH_CALUDE_cube_collinear_points_l511_51177

/-- Represents a point in a cube -/
inductive CubePoint
  | Vertex
  | EdgeMidpoint
  | FaceCenter
  | CubeCenter

/-- Represents a line in a cube -/
structure CubeLine where
  points : Finset CubePoint
  collinear : points.card = 3

/-- The set of all points in the cube -/
def cubePoints : Finset CubePoint := sorry

/-- The set of all lines in the cube -/
def cubeLines : Finset CubeLine := sorry

/-- The number of vertices in a cube -/
def numVertices : Nat := 8

/-- The number of edge midpoints in a cube -/
def numEdgeMidpoints : Nat := 12

/-- The number of face centers in a cube -/
def numFaceCenters : Nat := 6

/-- The number of cube centers in a cube -/
def numCubeCenters : Nat := 1

theorem cube_collinear_points :
  cubePoints.card = numVertices + numEdgeMidpoints + numFaceCenters + numCubeCenters ∧
  cubeLines.card = 49 := by sorry

end NUMINAMATH_CALUDE_cube_collinear_points_l511_51177


namespace NUMINAMATH_CALUDE_sam_read_100_pages_l511_51106

def minimum_assigned : ℕ := 25

def harrison_extra : ℕ := 10

def pam_extra : ℕ := 15

def sam_multiplier : ℕ := 2

def harrison_pages : ℕ := minimum_assigned + harrison_extra

def pam_pages : ℕ := harrison_pages + pam_extra

def sam_pages : ℕ := sam_multiplier * pam_pages

theorem sam_read_100_pages : sam_pages = 100 := by
  sorry

end NUMINAMATH_CALUDE_sam_read_100_pages_l511_51106


namespace NUMINAMATH_CALUDE_quadratic_complete_square_l511_51137

theorem quadratic_complete_square (x : ℝ) : ∃ (p q : ℝ), 
  (4 * x^2 + 8 * x - 448 = 0) ↔ ((x + p)^2 = q) ∧ q = 113 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l511_51137


namespace NUMINAMATH_CALUDE_guard_circles_l511_51111

/-- Calculates the number of times a guard should circle a rectangular warehouse --/
def warehouseCircles (length width walked skipped : ℕ) : ℕ :=
  let perimeter := 2 * (length + width)
  let actualCircles := walked / perimeter
  actualCircles + skipped

/-- Theorem stating that for the given warehouse and guard's walk, the number of circles is 10 --/
theorem guard_circles : 
  warehouseCircles 600 400 16000 2 = 10 := by sorry

end NUMINAMATH_CALUDE_guard_circles_l511_51111


namespace NUMINAMATH_CALUDE_nuts_distribution_l511_51188

/-- The number of ways to distribute n identical objects into k distinct groups -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of nuts to be distributed -/
def num_nuts : ℕ := 9

/-- The number of pockets -/
def num_pockets : ℕ := 3

theorem nuts_distribution :
  distribute num_nuts num_pockets = 55 := by sorry

end NUMINAMATH_CALUDE_nuts_distribution_l511_51188


namespace NUMINAMATH_CALUDE_camper_difference_is_nine_l511_51178

/-- The number of campers who went rowing in the morning -/
def morning_campers : ℕ := 52

/-- The number of campers who went rowing in the afternoon -/
def afternoon_campers : ℕ := 61

/-- The difference in the number of campers rowing in the afternoon compared to the morning -/
def camper_difference : ℕ := afternoon_campers - morning_campers

/-- Theorem stating that the difference in campers is 9 -/
theorem camper_difference_is_nine : camper_difference = 9 := by
  sorry

end NUMINAMATH_CALUDE_camper_difference_is_nine_l511_51178


namespace NUMINAMATH_CALUDE_jolene_total_earnings_l511_51157

/-- The amount of money Jolene raised through babysitting and car washing -/
def jolene_earnings (num_families : ℕ) (babysitting_rate : ℕ) (num_cars : ℕ) (car_wash_rate : ℕ) : ℕ :=
  num_families * babysitting_rate + num_cars * car_wash_rate

/-- Theorem stating that Jolene raised $180 given the specified conditions -/
theorem jolene_total_earnings :
  jolene_earnings 4 30 5 12 = 180 := by
  sorry

end NUMINAMATH_CALUDE_jolene_total_earnings_l511_51157


namespace NUMINAMATH_CALUDE_hexagonal_pyramid_vertices_l511_51103

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- Add necessary fields

/-- A pyramid with a regular polygon base -/
structure Pyramid (n : ℕ) where
  base : RegularPolygon n

/-- The number of vertices in a pyramid -/
def Pyramid.numVertices (p : Pyramid n) : ℕ := sorry

/-- Theorem: A pyramid with a base that is a regular polygon with six equal angles has 7 vertices -/
theorem hexagonal_pyramid_vertices :
  ∀ (p : Pyramid 6), p.numVertices = 7 := by sorry

end NUMINAMATH_CALUDE_hexagonal_pyramid_vertices_l511_51103


namespace NUMINAMATH_CALUDE_sum_of_specific_coefficients_l511_51152

/-- The coefficient of x^m * y^n in the expansion of (1+x)^4 * (1+y)^6 -/
def P (m n : ℕ) : ℕ := Nat.choose 4 m * Nat.choose 6 n

/-- The sum of coefficients of x^2*y^1 and x^1*y^2 in the expansion of (1+x)^4 * (1+y)^6 is 96 -/
theorem sum_of_specific_coefficients : P 2 1 + P 1 2 = 96 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_coefficients_l511_51152


namespace NUMINAMATH_CALUDE_total_area_after_expansion_l511_51105

/-- Theorem: Total area of two houses after expansion -/
theorem total_area_after_expansion (small_house large_house expansion : ℕ) 
  (h1 : small_house = 5200)
  (h2 : large_house = 7300)
  (h3 : expansion = 3500) :
  small_house + large_house + expansion = 16000 := by
  sorry

#check total_area_after_expansion

end NUMINAMATH_CALUDE_total_area_after_expansion_l511_51105


namespace NUMINAMATH_CALUDE_ln_exp_equals_id_l511_51179

theorem ln_exp_equals_id : ∀ x : ℝ, Real.log (Real.exp x) = x := by sorry

end NUMINAMATH_CALUDE_ln_exp_equals_id_l511_51179


namespace NUMINAMATH_CALUDE_ratio_of_divisors_sums_l511_51159

def N : ℕ := 68 * 68 * 125 * 135

def sum_of_odd_divisors (n : ℕ) : ℕ := sorry
def sum_of_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisors_sums :
  (sum_of_odd_divisors N) * 30 = sum_of_even_divisors N :=
sorry

end NUMINAMATH_CALUDE_ratio_of_divisors_sums_l511_51159


namespace NUMINAMATH_CALUDE_cat_arrangement_count_l511_51183

/-- Represents the number of cat cages -/
def num_cages : ℕ := 5

/-- Represents the number of golden tabby cats -/
def num_golden : ℕ := 3

/-- Represents the number of silver tabby cats -/
def num_silver : ℕ := 4

/-- Represents the number of ragdoll cats -/
def num_ragdoll : ℕ := 1

/-- Represents the number of ways to arrange silver tabby cats in pairs -/
def silver_arrangements : ℕ := 3

/-- Represents the total number of units to arrange (golden group, 2 silver pairs, ragdoll) -/
def total_units : ℕ := 4

/-- Theorem stating the number of possible arrangements -/
theorem cat_arrangement_count :
  (Nat.choose num_cages total_units) * Nat.factorial total_units * silver_arrangements = 360 := by
  sorry

end NUMINAMATH_CALUDE_cat_arrangement_count_l511_51183


namespace NUMINAMATH_CALUDE_unique_solution_for_system_l511_51148

theorem unique_solution_for_system (x y : ℝ) :
  (x + y = (7 - x) + (7 - y)) ∧ (x - y = (x - 2) + (y - 2)) →
  x = 5 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_system_l511_51148


namespace NUMINAMATH_CALUDE_sphere_volume_from_box_diagonal_l511_51168

theorem sphere_volume_from_box_diagonal (a b c : ℝ) (ha : a = 3 * Real.sqrt 2) (hb : b = 4 * Real.sqrt 2) (hc : c = 5 * Real.sqrt 2) :
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  (4 / 3) * Real.pi * (diagonal / 2)^3 = 500 * Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_sphere_volume_from_box_diagonal_l511_51168


namespace NUMINAMATH_CALUDE_tire_quality_probability_l511_51165

def tire_widths : List ℕ := [195, 196, 190, 194, 200]

def is_within_range (w : ℕ) : Bool :=
  192 ≤ w ∧ w ≤ 198

def count_within_range (l : List ℕ) : ℕ :=
  (l.filter is_within_range).length

def combinations_count (n k : ℕ) : ℕ :=
  Nat.choose n k

def favorable_outcomes : ℕ :=
  combinations_count 3 2 * combinations_count 2 1 + combinations_count 3 3

def total_outcomes : ℕ :=
  combinations_count 5 3

theorem tire_quality_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 7 / 10 :=
sorry

end NUMINAMATH_CALUDE_tire_quality_probability_l511_51165


namespace NUMINAMATH_CALUDE_february_first_is_sunday_l511_51194

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in February of a leap year -/
structure FebruaryDate where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to check if a given day is Monday -/
def isMonday (d : DayOfWeek) : Bool :=
  match d with
  | DayOfWeek.Monday => true
  | _ => false

/-- Theorem: In a leap year, if February has exactly four Mondays, then February 1st must be a Sunday -/
theorem february_first_is_sunday (february : List FebruaryDate) 
  (leap_year : february.length = 29)
  (four_mondays : (february.filter (fun d => isMonday d.dayOfWeek)).length = 4) :
  (february.head?.map (fun d => d.dayOfWeek) = some DayOfWeek.Sunday) :=
by
  sorry


end NUMINAMATH_CALUDE_february_first_is_sunday_l511_51194


namespace NUMINAMATH_CALUDE_dream_car_gas_consumption_l511_51143

/-- Calculates the total gas consumption for a car over two days -/
def total_gas_consumption (consumption_rate : ℝ) (miles_day1 : ℝ) (miles_day2 : ℝ) : ℝ :=
  consumption_rate * (miles_day1 + miles_day2)

/-- Proves that given the specified conditions, the total gas consumption is 4000 gallons -/
theorem dream_car_gas_consumption :
  let consumption_rate : ℝ := 4
  let miles_day1 : ℝ := 400
  let miles_day2 : ℝ := miles_day1 + 200
  total_gas_consumption consumption_rate miles_day1 miles_day2 = 4000 :=
by
  sorry

#eval total_gas_consumption 4 400 600

end NUMINAMATH_CALUDE_dream_car_gas_consumption_l511_51143


namespace NUMINAMATH_CALUDE_intersection_implies_a_values_l511_51185

-- Define sets A and B
def A : Set ℝ := {1, 2}
def B (a : ℝ) : Set ℝ := {a, a^2 - 1}

-- State the theorem
theorem intersection_implies_a_values (a : ℝ) :
  (A ∩ B a = {1}) → (a = 1 ∨ a = Real.sqrt 2 ∨ a = -Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_values_l511_51185


namespace NUMINAMATH_CALUDE_sum_of_b_and_c_is_eleven_l511_51134

theorem sum_of_b_and_c_is_eleven
  (a b c : ℕ+)
  (ha : a ≠ 1)
  (hb : b ≤ 9)
  (hc : c ≤ 9)
  (hbc : b ≠ c)
  (heq : (10 * a + b) * (10 * a + c) = 100 * a^2 + 110 * a + b * c) :
  b + c = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_b_and_c_is_eleven_l511_51134


namespace NUMINAMATH_CALUDE_age_difference_proof_l511_51100

theorem age_difference_proof : ∃ (a b : ℕ), 
  (a ≥ 10 ∧ a < 100) ∧ 
  (b ≥ 10 ∧ b < 100) ∧ 
  (a / 10 = b % 10) ∧ 
  (a % 10 = b / 10) ∧ 
  (a + 7 = 3 * (b + 7)) ∧ 
  (a - b = 36) := by
sorry

end NUMINAMATH_CALUDE_age_difference_proof_l511_51100


namespace NUMINAMATH_CALUDE_triangle_side_sum_unbounded_l511_51113

theorem triangle_side_sum_unbounded (b c : ℝ) :
  ∀ ε > 0, ∃ b' c' : ℝ,
    b' > 0 ∧ c' > 0 ∧
    b'^2 + c'^2 + b' * c' = 25 ∧
    b' + c' > ε :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_sum_unbounded_l511_51113


namespace NUMINAMATH_CALUDE_grid_product_theorem_l511_51166

/-- Represents a 3x3 grid of natural numbers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- The condition that all numbers in the grid are distinct -/
def all_distinct (g : Grid) : Prop :=
  ∀ i j i' j', g i j = g i' j' → (i = i' ∧ j = j')

/-- The product of numbers in a row -/
def row_product (g : Grid) (i : Fin 3) : ℕ :=
  (g i 0) * (g i 1) * (g i 2)

/-- The product of numbers in a column -/
def col_product (g : Grid) (j : Fin 3) : ℕ :=
  (g 0 j) * (g 1 j) * (g 2 j)

/-- The condition that all row and column products are equal -/
def all_products_equal (g : Grid) (P : ℕ) : Prop :=
  (∀ i : Fin 3, row_product g i = P) ∧
  (∀ j : Fin 3, col_product g j = P)

/-- The set of possible values for P -/
def P_values : Set ℕ := {1996, 1997, 1998, 1999, 2000, 2001}

/-- The main theorem -/
theorem grid_product_theorem :
  ∃ (g : Grid) (P : ℕ),
    (∀ i j, g i j ∈ Finset.range 10) ∧
    all_distinct g ∧
    all_products_equal g P ∧
    P ∈ P_values ↔
    P = 1998 ∨ P = 2000 :=
  sorry

end NUMINAMATH_CALUDE_grid_product_theorem_l511_51166


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_2sqrt7_l511_51104

theorem sqrt_sum_equals_2sqrt7 :
  Real.sqrt (10 - 2 * Real.sqrt 21) + Real.sqrt (10 + 2 * Real.sqrt 21) = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_2sqrt7_l511_51104


namespace NUMINAMATH_CALUDE_remaining_calories_l511_51139

-- Define the given conditions
def calories_per_serving : ℕ := 110
def servings_per_block : ℕ := 16
def servings_eaten : ℕ := 5

-- Define the theorem
theorem remaining_calories :
  (servings_per_block - servings_eaten) * calories_per_serving = 1210 := by
  sorry

end NUMINAMATH_CALUDE_remaining_calories_l511_51139


namespace NUMINAMATH_CALUDE_expression_equals_one_tenth_l511_51128

-- Define the ceiling function
def ceiling (x : ℚ) : ℤ := ⌈x⌉

-- Define the expression
def expression : ℚ := 
  (ceiling ((25 : ℚ) / 11 - ceiling ((35 : ℚ) / 19))) / 
  (ceiling ((35 : ℚ) / 11 + ceiling ((11 * 19 : ℚ) / 35)))

-- Theorem statement
theorem expression_equals_one_tenth : expression = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_tenth_l511_51128


namespace NUMINAMATH_CALUDE_lola_poptarts_count_l511_51170

/-- The number of mini cupcakes Lola baked -/
def lola_cupcakes : ℕ := 13

/-- The number of blueberry pies Lola baked -/
def lola_pies : ℕ := 8

/-- The number of mini cupcakes Lulu made -/
def lulu_cupcakes : ℕ := 16

/-- The number of pop tarts Lulu made -/
def lulu_poptarts : ℕ := 12

/-- The number of blueberry pies Lulu made -/
def lulu_pies : ℕ := 14

/-- The total number of pastries made by Lola and Lulu -/
def total_pastries : ℕ := 73

/-- The number of pop tarts Lola baked -/
def lola_poptarts : ℕ := 10

theorem lola_poptarts_count :
  lola_cupcakes + lola_poptarts + lola_pies + lulu_cupcakes + lulu_poptarts + lulu_pies = total_pastries :=
by sorry

end NUMINAMATH_CALUDE_lola_poptarts_count_l511_51170


namespace NUMINAMATH_CALUDE_unique_single_digit_polynomial_exists_l511_51198

/-- A polynomial with single-digit coefficients -/
def SingleDigitPolynomial (p : Polynomial ℤ) : Prop :=
  ∀ i, (p.coeff i) ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℤ)

/-- The theorem statement -/
theorem unique_single_digit_polynomial_exists (n : ℤ) :
  ∃! p : Polynomial ℤ, SingleDigitPolynomial p ∧ p.eval (-2) = n ∧ p.eval (-5) = n := by
  sorry

end NUMINAMATH_CALUDE_unique_single_digit_polynomial_exists_l511_51198


namespace NUMINAMATH_CALUDE_A_ends_with_14_zeros_l511_51174

theorem A_ends_with_14_zeros :
  let A := 2^7 * (7^14 + 1) + 2^6 * 7^11 * 10^2 + 2^6 * 7^7 * 10^4 + 2^4 * 7^3 * 10^6
  A = 10^14 := by sorry

end NUMINAMATH_CALUDE_A_ends_with_14_zeros_l511_51174


namespace NUMINAMATH_CALUDE_quadratic_comparison_l511_51155

/-- A quadratic function f(x) = x^2 - 2x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + m

theorem quadratic_comparison (m : ℝ) (y₁ y₂ : ℝ) 
  (h1 : f m (-1) = y₁)
  (h2 : f m 2 = y₂) :
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_comparison_l511_51155


namespace NUMINAMATH_CALUDE_pink_tie_probability_l511_51175

-- Define the number of ties of each color
def black_ties : ℕ := 5
def gold_ties : ℕ := 7
def pink_ties : ℕ := 8

-- Define the total number of ties
def total_ties : ℕ := black_ties + gold_ties + pink_ties

-- Define the probability of choosing a pink tie
def prob_pink_tie : ℚ := pink_ties / total_ties

-- Theorem statement
theorem pink_tie_probability :
  prob_pink_tie = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_pink_tie_probability_l511_51175


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_quadratic_roots_l511_51141

theorem perpendicular_lines_from_quadratic_roots (b : ℝ) :
  ∀ k₁ k₂ : ℝ, (k₁^2 + b*k₁ - 1 = 0) → (k₂^2 + b*k₂ - 1 = 0) → k₁ * k₂ = -1 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_quadratic_roots_l511_51141


namespace NUMINAMATH_CALUDE_franks_allowance_l511_51110

/-- The amount Frank had saved up -/
def savings : ℕ := 3

/-- The number of toys Frank could buy -/
def num_toys : ℕ := 5

/-- The price of each toy -/
def toy_price : ℕ := 8

/-- The amount Frank received for his allowance -/
def allowance : ℕ := 37

theorem franks_allowance :
  savings + allowance = num_toys * toy_price :=
by sorry

end NUMINAMATH_CALUDE_franks_allowance_l511_51110


namespace NUMINAMATH_CALUDE_peters_fish_catch_l511_51187

theorem peters_fish_catch (n : ℕ) : (3 * n = n + 24) → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_peters_fish_catch_l511_51187


namespace NUMINAMATH_CALUDE_initial_bench_weight_l511_51160

/-- Represents the weightlifting scenario for John --/
structure WeightliftingScenario where
  initialSquat : ℝ
  initialDeadlift : ℝ
  squatLossPercentage : ℝ
  deadliftLoss : ℝ
  newTotal : ℝ

/-- Calculates the initial bench weight given the weightlifting scenario --/
def calculateInitialBench (scenario : WeightliftingScenario) : ℝ :=
  scenario.newTotal - 
  (scenario.initialSquat * (1 - scenario.squatLossPercentage)) - 
  (scenario.initialDeadlift - scenario.deadliftLoss)

/-- Theorem stating that the initial bench weight is 400 pounds --/
theorem initial_bench_weight (scenario : WeightliftingScenario) 
  (h1 : scenario.initialSquat = 700)
  (h2 : scenario.initialDeadlift = 800)
  (h3 : scenario.squatLossPercentage = 0.3)
  (h4 : scenario.deadliftLoss = 200)
  (h5 : scenario.newTotal = 1490) :
  calculateInitialBench scenario = 400 := by
  sorry


end NUMINAMATH_CALUDE_initial_bench_weight_l511_51160


namespace NUMINAMATH_CALUDE_calculate_example_not_commutative_l511_51145

-- Define the new operation
def otimes (a b : ℤ) : ℤ := a * b + a - b

-- Theorem 1: Calculate ((-2) ⊗ 5) ⊗ 6
theorem calculate_example : otimes (otimes (-2) 5) 6 = -125 := by
  sorry

-- Theorem 2: The operation is not commutative
theorem not_commutative : ∃ a b : ℤ, otimes a b ≠ otimes b a := by
  sorry

end NUMINAMATH_CALUDE_calculate_example_not_commutative_l511_51145


namespace NUMINAMATH_CALUDE_empty_set_proof_l511_51162

theorem empty_set_proof : {x : ℝ | x > 6 ∧ x < 1} = ∅ := by
  sorry

end NUMINAMATH_CALUDE_empty_set_proof_l511_51162


namespace NUMINAMATH_CALUDE_plane_through_point_and_line_l511_51146

def line_equation (x y z : ℝ) : Prop :=
  (x - 2) / 4 = (y + 1) / (-5) ∧ (y + 1) / (-5) = (z - 3) / 2

def plane_equation (x y z : ℝ) : Prop :=
  3 * x + 4 * y + 4 * z - 14 = 0

def point_on_plane (x y z : ℝ) : Prop :=
  x = 2 ∧ y = -3 ∧ z = 5

def coefficients_conditions (A B C D : ℤ) : Prop :=
  A > 0 ∧ Nat.gcd (Nat.gcd (Nat.gcd (A.natAbs) (B.natAbs)) (C.natAbs)) (D.natAbs) = 1

theorem plane_through_point_and_line :
  ∀ (x y z : ℝ),
    (∃ (t : ℝ), line_equation (x + t) (y + t) (z + t)) →
    point_on_plane 2 (-3) 5 →
    coefficients_conditions 3 4 4 (-14) →
    plane_equation x y z :=
sorry

end NUMINAMATH_CALUDE_plane_through_point_and_line_l511_51146


namespace NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l511_51118

theorem gcd_from_lcm_and_ratio (A B : ℕ) (h1 : lcm A B = 180) (h2 : ∃ k : ℕ, A = 2 * k ∧ B = 3 * k) : 
  gcd A B = 30 := by
sorry

end NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l511_51118


namespace NUMINAMATH_CALUDE_sector_area_l511_51195

/-- Given a circular sector with perimeter 6 cm and central angle 1 radian, its area is 3 cm² -/
theorem sector_area (r : ℝ) (h1 : r + r + r = 6) (h2 : 1 = 1) : r * r / 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l511_51195


namespace NUMINAMATH_CALUDE_line_intersects_parabola_vertex_once_l511_51120

/-- The number of values of a for which the line y = x + a passes through
    the vertex of the parabola y = x^3 - 3ax + a^2 is exactly one. -/
theorem line_intersects_parabola_vertex_once :
  ∃! a : ℝ, ∃ x y : ℝ,
    (y = x + a) ∧                   -- Line equation
    (y = x^3 - 3*a*x + a^2) ∧       -- Parabola equation
    (∀ x' : ℝ, x'^3 - 3*a*x' + a^2 ≤ x^3 - 3*a*x + a^2) -- Vertex condition
    := by sorry

end NUMINAMATH_CALUDE_line_intersects_parabola_vertex_once_l511_51120


namespace NUMINAMATH_CALUDE_feet_in_garden_l511_51169

theorem feet_in_garden (num_dogs num_ducks : ℕ) (dog_feet duck_feet : ℕ) :
  num_dogs = 6 → num_ducks = 2 → dog_feet = 4 → duck_feet = 2 →
  num_dogs * dog_feet + num_ducks * duck_feet = 28 := by
sorry

end NUMINAMATH_CALUDE_feet_in_garden_l511_51169


namespace NUMINAMATH_CALUDE_piggy_bank_theorem_specific_piggy_bank_case_l511_51142

/-- Represents a configuration of piggy banks and their keys -/
structure PiggyBankConfig (n : ℕ) where
  keys : Fin n → Fin n
  injective : Function.Injective keys

/-- The probability of opening all remaining piggy banks given n total and k broken -/
def openProbability (n k : ℕ) : ℚ :=
  if k ≤ n then k / n else 0

theorem piggy_bank_theorem (n k : ℕ) (h : k ≤ n) :
  openProbability n k = k / n := by sorry

theorem specific_piggy_bank_case :
  openProbability 30 2 = 1 / 15 := by sorry

end NUMINAMATH_CALUDE_piggy_bank_theorem_specific_piggy_bank_case_l511_51142


namespace NUMINAMATH_CALUDE_breakfast_cost_l511_51125

def toast_price : ℕ := 1
def egg_price : ℕ := 3

def dale_toast : ℕ := 2
def dale_eggs : ℕ := 2

def andrew_toast : ℕ := 1
def andrew_eggs : ℕ := 2

theorem breakfast_cost : 
  (dale_toast * toast_price + dale_eggs * egg_price) +
  (andrew_toast * toast_price + andrew_eggs * egg_price) = 15 := by
sorry

end NUMINAMATH_CALUDE_breakfast_cost_l511_51125


namespace NUMINAMATH_CALUDE_range_of_t_l511_51144

theorem range_of_t (a b : ℝ) (h : a^2 + a*b + b^2 = 1) :
  let t := a^2 - a*b + b^2
  ∃ (x : ℝ), t = x ∧ 1/3 ≤ x ∧ x ≤ 3 ∧
  ∀ (y : ℝ), (∃ (a' b' : ℝ), a'^2 + a'*b' + b'^2 = 1 ∧ a'^2 - a'*b' + b'^2 = y) → 1/3 ≤ y ∧ y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_t_l511_51144


namespace NUMINAMATH_CALUDE_derivative_product_at_4_and_neg1_l511_51189

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -1 / Real.sqrt x else 1 + x^2

theorem derivative_product_at_4_and_neg1 :
  (deriv f 4) * (deriv f (-1)) = -1/8 := by sorry

end NUMINAMATH_CALUDE_derivative_product_at_4_and_neg1_l511_51189


namespace NUMINAMATH_CALUDE_money_left_over_l511_51180

-- Define the given conditions
def video_game_cost : ℝ := 60
def discount_rate : ℝ := 0.15
def candy_cost : ℝ := 5
def sales_tax_rate : ℝ := 0.10
def shipping_fee : ℝ := 3
def babysitting_rate : ℝ := 8
def hours_worked : ℝ := 9

-- Define the theorem
theorem money_left_over :
  let discounted_price := video_game_cost * (1 - discount_rate)
  let video_game_total := discounted_price + shipping_fee
  let video_game_with_tax := video_game_total * (1 + sales_tax_rate)
  let candy_with_tax := candy_cost * (1 + sales_tax_rate)
  let total_cost := video_game_with_tax + candy_with_tax
  let earnings := babysitting_rate * hours_worked
  earnings - total_cost = 7.10 := by
  sorry

end NUMINAMATH_CALUDE_money_left_over_l511_51180


namespace NUMINAMATH_CALUDE_quadratic_root_power_sums_l511_51192

/-- Given a quadratic equation ax^2 + bx + c = 0 with roots x₁ and x₂,
    s_n denotes the sum of the n-th powers of the roots. -/
def s (n : ℕ) (x₁ x₂ : ℝ) : ℝ := x₁^n + x₂^n

/-- Theorem stating the relations between sums of powers of roots of a quadratic equation -/
theorem quadratic_root_power_sums 
  (a b c : ℝ) (x₁ x₂ : ℝ) 
  (h : a ≠ 0)
  (hroot : a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) :
  (∀ n : ℕ, n ≥ 2 → a * s n x₁ x₂ + b * s (n-1) x₁ x₂ + c * s (n-2) x₁ x₂ = 0) ∧
  (a * s 2 x₁ x₂ + b * s 1 x₁ x₂ + 2 * c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_power_sums_l511_51192


namespace NUMINAMATH_CALUDE_solve_equation_l511_51109

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((3 / x) + 3) = 5 / 3 → x = -27 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l511_51109


namespace NUMINAMATH_CALUDE_puppies_per_cage_l511_51133

/-- Given a pet store scenario with puppies and cages, calculate puppies per cage -/
theorem puppies_per_cage 
  (total_puppies : ℕ) 
  (sold_puppies : ℕ) 
  (num_cages : ℕ) 
  (h1 : total_puppies = 45)
  (h2 : sold_puppies = 39)
  (h3 : num_cages = 3)
  (h4 : sold_puppies < total_puppies) :
  (total_puppies - sold_puppies) / num_cages = 2 := by
sorry

end NUMINAMATH_CALUDE_puppies_per_cage_l511_51133


namespace NUMINAMATH_CALUDE_intersection_constraint_l511_51196

theorem intersection_constraint (a : ℝ) : 
  let A : Set ℝ := {a^2, a+1, -3}
  let B : Set ℝ := {a-3, 2*a-1, a^2+1}
  A ∩ B = {-3} → a = -1 ∨ a = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_constraint_l511_51196


namespace NUMINAMATH_CALUDE_quadratic_function_unique_coefficients_l511_51167

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 - 2 * a * x + 1 + b

theorem quadratic_function_unique_coefficients 
  (a b : ℝ) 
  (h_a_pos : a > 0) 
  (h_max : ∀ x ∈ Set.Icc 2 3, f a b x ≤ 4) 
  (h_min : ∀ x ∈ Set.Icc 2 3, f a b x ≥ 1) 
  (h_max_achieved : ∃ x ∈ Set.Icc 2 3, f a b x = 4) 
  (h_min_achieved : ∃ x ∈ Set.Icc 2 3, f a b x = 1) : 
  a = 1 ∧ b = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_coefficients_l511_51167


namespace NUMINAMATH_CALUDE_solution_to_linear_equation_l511_51191

theorem solution_to_linear_equation :
  ∃ (x y : ℝ), 2 * x + y = 6 ∧ x = 2 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_linear_equation_l511_51191


namespace NUMINAMATH_CALUDE_work_completion_time_l511_51123

/-- Given that two workers A and B can complete a work together in a certain number of days,
    and worker A can complete the work alone in a certain number of days,
    this function calculates the number of days worker B needs to complete the work alone. -/
def days_for_b_alone (days_together : ℚ) (days_a_alone : ℚ) : ℚ :=
  (days_together * days_a_alone) / (days_a_alone - days_together)

/-- Theorem stating that if A and B together can complete a work in 4 days,
    and A alone can complete the same work in 12 days,
    then B alone can complete the work in 6 days. -/
theorem work_completion_time :
  days_for_b_alone 4 12 = 6 := by
  sorry


end NUMINAMATH_CALUDE_work_completion_time_l511_51123


namespace NUMINAMATH_CALUDE_extension_point_coordinates_l511_51161

/-- Given two points P₁ and P₂ in ℝ², and a point P on the extension line of P₁P₂
    such that the distance from P₁ to P is twice the distance from P to P₂,
    prove that P has the specified coordinates. -/
theorem extension_point_coordinates (P₁ P₂ P : ℝ × ℝ) : 
  P₁ = (2, -1) →
  P₂ = (0, 5) →
  (∃ t : ℝ, t ∉ [0, 1] ∧ P = P₁ + t • (P₂ - P₁)) →
  ‖P - P₁‖ = 2 * ‖P - P₂‖ →
  P = (-2, 11) := by sorry

end NUMINAMATH_CALUDE_extension_point_coordinates_l511_51161


namespace NUMINAMATH_CALUDE_locus_of_circumcenter_l511_51151

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcenter
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the product OA · OB · OC
def product (c : Circle) (t : Triangle) : ℝ := sorry

theorem locus_of_circumcenter 
  (c : Circle) 
  (t : Triangle) 
  (p : ℝ) 
  (h : product c t = p^3) :
  ∃ (P : ℝ × ℝ), 
    P = circumcenter t ∧ 
    distance c.center P = (p / (4 * c.radius^2)) * Real.sqrt (p * (p^3 - 8 * c.radius^3)) :=
sorry

end NUMINAMATH_CALUDE_locus_of_circumcenter_l511_51151


namespace NUMINAMATH_CALUDE_problem1_l511_51131

theorem problem1 (x y : ℝ) (h : y ≠ 0) :
  ((x + 3 * y) * (x - 3 * y) - x^2) / (9 * y) = -y := by sorry

end NUMINAMATH_CALUDE_problem1_l511_51131


namespace NUMINAMATH_CALUDE_output_is_76_l511_51130

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 ≤ 30 then
    (step1 + 10)
  else
    ((step1 - 7) * 2)

theorem output_is_76 : function_machine 15 = 76 := by
  sorry

end NUMINAMATH_CALUDE_output_is_76_l511_51130


namespace NUMINAMATH_CALUDE_investment_rate_proof_l511_51140

/-- Proves that given an investment scenario, the unknown rate is 1% -/
theorem investment_rate_proof (total_investment : ℝ) (amount_at_10_percent : ℝ) (total_interest : ℝ)
  (h1 : total_investment = 31000)
  (h2 : amount_at_10_percent = 12000)
  (h3 : total_interest = 1390)
  : (total_interest - 0.1 * amount_at_10_percent) / (total_investment - amount_at_10_percent) = 0.01 := by
  sorry

end NUMINAMATH_CALUDE_investment_rate_proof_l511_51140


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_products_min_value_is_achievable_l511_51149

def is_permutation_of_1_to_9 (a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ : ℕ) : Prop :=
  ({a₁, a₂, a₃, b₁, b₂, b₃, c₁, c₂, c₃} : Finset ℕ) = Finset.range 9

theorem min_value_of_sum_of_products 
  (a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ : ℕ) 
  (h : is_permutation_of_1_to_9 a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃) : 
  a₁ * a₂ * a₃ + b₁ * b₂ * b₃ + c₁ * c₂ * c₃ ≥ 214 :=
sorry

theorem min_value_is_achievable : 
  ∃ a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ : ℕ, 
    is_permutation_of_1_to_9 a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ ∧ 
    a₁ * a₂ * a₃ + b₁ * b₂ * b₃ + c₁ * c₂ * c₃ = 214 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_products_min_value_is_achievable_l511_51149


namespace NUMINAMATH_CALUDE_sin_squared_value_l511_51116

theorem sin_squared_value (α : Real) (h : Real.tan (α + π/4) = 3/4) :
  Real.sin (π/4 - α) ^ 2 = 16/25 := by sorry

end NUMINAMATH_CALUDE_sin_squared_value_l511_51116


namespace NUMINAMATH_CALUDE_first_pair_price_is_22_l511_51150

/-- The price of the first pair of shoes -/
def first_pair_price : ℝ := 22

/-- The price of the second pair of shoes -/
def second_pair_price : ℝ := 1.5 * first_pair_price

/-- The total price of both pairs of shoes -/
def total_price : ℝ := 55

/-- Theorem stating that the price of the first pair of shoes is $22 -/
theorem first_pair_price_is_22 :
  first_pair_price = 22 ∧
  second_pair_price = 1.5 * first_pair_price ∧
  total_price = first_pair_price + second_pair_price :=
by sorry

end NUMINAMATH_CALUDE_first_pair_price_is_22_l511_51150


namespace NUMINAMATH_CALUDE_max_abs_sum_of_coeffs_l511_51171

/-- A quadratic polynomial p(x) = ax^2 + bx + c -/
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The condition that |p(x)| ≤ 1 for all x in [0,1] -/
def BoundedOnInterval (p : ℝ → ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc 0 1 → |p x| ≤ 1

/-- The theorem stating that the maximum value of |a|+|b|+|c| is 4 -/
theorem max_abs_sum_of_coeffs (a b c : ℝ) :
  BoundedOnInterval (QuadraticPolynomial a b c) →
  |a| + |b| + |c| ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_abs_sum_of_coeffs_l511_51171


namespace NUMINAMATH_CALUDE_kvass_theorem_l511_51173

/-- The volume of kvass remaining after n people have drunk from it -/
def remainingVolume (n : ℕ) : ℚ :=
  1.5 * (Nat.factorial n) / (Nat.factorial (n + 1))

/-- The statement to be proved -/
theorem kvass_theorem : remainingVolume 14 = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_kvass_theorem_l511_51173


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l511_51135

theorem smallest_n_congruence : 
  ∃ n : ℕ+, (∀ k < n, (7 : ℤ)^(k : ℕ) % 5 ≠ (k : ℤ)^7 % 5) ∧ (7 : ℤ)^(n : ℕ) % 5 = (n : ℤ)^7 % 5 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l511_51135


namespace NUMINAMATH_CALUDE_total_potatoes_sold_l511_51176

/-- The number of bags of potatoes sold in the morning -/
def morning_bags : ℕ := 29

/-- The number of bags of potatoes sold in the afternoon -/
def afternoon_bags : ℕ := 17

/-- The weight of each bag of potatoes in kilograms -/
def bag_weight : ℕ := 7

/-- The total kilograms of potatoes sold for the whole day -/
def total_kg : ℕ := (morning_bags + afternoon_bags) * bag_weight

theorem total_potatoes_sold : total_kg = 322 := by
  sorry

end NUMINAMATH_CALUDE_total_potatoes_sold_l511_51176


namespace NUMINAMATH_CALUDE_shift_left_3_units_l511_51163

-- Define the original function
def f (x : ℝ) : ℝ := (x - 1)^2

-- Define the shifted function
def g (x : ℝ) : ℝ := (x + 2)^2

-- Define the shift operation
def shift (h : ℝ → ℝ) (s : ℝ) : ℝ → ℝ := fun x ↦ h (x + s)

-- Theorem statement
theorem shift_left_3_units :
  shift f 3 = g := by sorry

end NUMINAMATH_CALUDE_shift_left_3_units_l511_51163


namespace NUMINAMATH_CALUDE_vector_parallelism_transitivity_l511_51199

/-- Given three non-zero vectors, if the first is parallel to the second and the second is parallel to the third, then the first is parallel to the third. -/
theorem vector_parallelism_transitivity 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b c : V) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : ∃ (k : ℝ), a = k • b) (hbc : ∃ (m : ℝ), b = m • c) : 
  ∃ (n : ℝ), a = n • c :=
sorry

end NUMINAMATH_CALUDE_vector_parallelism_transitivity_l511_51199


namespace NUMINAMATH_CALUDE_b_41_mod_49_l511_51158

/-- The sequence b_n defined as 6^n + 8^n -/
def b (n : ℕ) : ℕ := 6^n + 8^n

/-- The theorem stating that b_41 is congruent to 35 modulo 49 -/
theorem b_41_mod_49 : b 41 ≡ 35 [ZMOD 49] := by sorry

end NUMINAMATH_CALUDE_b_41_mod_49_l511_51158


namespace NUMINAMATH_CALUDE_no_combination_for_3_4_meters_l511_51193

theorem no_combination_for_3_4_meters :
  ¬ ∃ (a b : ℕ), 0.7 * (a : ℝ) + 0.8 * (b : ℝ) = 3.4 := by
  sorry

end NUMINAMATH_CALUDE_no_combination_for_3_4_meters_l511_51193


namespace NUMINAMATH_CALUDE_jordan_income_proof_l511_51126

-- Define the daily incomes and work days
def terry_daily_income : ℝ := 24
def work_days : ℕ := 7
def weekly_income_difference : ℝ := 42

-- Define Jordan's daily income as a variable
def jordan_daily_income : ℝ := 30

-- Theorem to prove
theorem jordan_income_proof :
  jordan_daily_income * work_days - terry_daily_income * work_days = weekly_income_difference :=
by sorry

end NUMINAMATH_CALUDE_jordan_income_proof_l511_51126


namespace NUMINAMATH_CALUDE_circle_under_translation_l511_51197

/-- A parallel translation in a 2D plane. -/
structure ParallelTranslation where
  shift : ℝ × ℝ

/-- A circle in a 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The result of applying a parallel translation to a circle. -/
def translateCircle (c : Circle) (t : ParallelTranslation) : Circle :=
  { center := (c.center.1 + t.shift.1, c.center.2 + t.shift.2),
    radius := c.radius }

/-- Theorem: A circle remains a circle under parallel translation. -/
theorem circle_under_translation (c : Circle) (t : ParallelTranslation) :
  ∃ (c' : Circle), c' = translateCircle c t ∧ c'.radius = c.radius :=
by sorry

end NUMINAMATH_CALUDE_circle_under_translation_l511_51197


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l511_51184

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (3 + Real.sqrt x) = 4 → x = 169 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l511_51184


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_l511_51112

theorem sum_of_roots_cubic (x : ℝ) : 
  (∃ s : ℝ, (∀ x, x^3 - x^2 - 13*x + 13 = 0 → (∃ y z : ℝ, y ≠ x ∧ z ≠ x ∧ z ≠ y ∧ 
    x + y + z = s))) → 
  (∃ s : ℝ, (∀ x, x^3 - x^2 - 13*x + 13 = 0 → (∃ y z : ℝ, y ≠ x ∧ z ≠ x ∧ z ≠ y ∧ 
    x + y + z = s)) ∧ s = 1) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_l511_51112


namespace NUMINAMATH_CALUDE_polynomial_simplification_l511_51156

theorem polynomial_simplification (x : ℝ) : 
  (3*x - 4)*(x + 6) - (x + 3)*(3*x + 2) = 3*x - 30 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l511_51156
