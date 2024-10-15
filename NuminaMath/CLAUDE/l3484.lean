import Mathlib

namespace NUMINAMATH_CALUDE_matrix_inverse_and_solution_l3484_348499

theorem matrix_inverse_and_solution (A B M : Matrix (Fin 2) (Fin 2) ℝ) : 
  A = ![![2, 0], ![-1, 1]] →
  B = ![![2, 4], ![3, 5]] →
  A * M = B →
  A⁻¹ = ![![1/2, 0], ![1/2, 1]] ∧
  M = ![![1, 2], ![4, 7]] := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_and_solution_l3484_348499


namespace NUMINAMATH_CALUDE_integer_sum_problem_l3484_348423

theorem integer_sum_problem (x y : ℤ) : 
  x > 0 → y > 0 → x - y = 8 → x * y = 120 → x + y = 4 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_problem_l3484_348423


namespace NUMINAMATH_CALUDE_joans_books_l3484_348471

/-- Given that Sam has 110 books and the total number of books Sam and Joan have together is 212,
    prove that Joan has 102 books. -/
theorem joans_books (sam_books : ℕ) (total_books : ℕ) (h1 : sam_books = 110) (h2 : total_books = 212) :
  total_books - sam_books = 102 := by
  sorry

end NUMINAMATH_CALUDE_joans_books_l3484_348471


namespace NUMINAMATH_CALUDE_nathan_tokens_used_l3484_348466

/-- The number of tokens Nathan used at the arcade -/
def tokens_used (air_hockey_games basketball_games tokens_per_game : ℕ) : ℕ :=
  (air_hockey_games + basketball_games) * tokens_per_game

/-- Theorem: Nathan used 18 tokens at the arcade -/
theorem nathan_tokens_used :
  tokens_used 2 4 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_nathan_tokens_used_l3484_348466


namespace NUMINAMATH_CALUDE_maries_age_l3484_348434

theorem maries_age (marie_age marco_age : ℕ) : 
  marco_age = 2 * marie_age + 1 →
  marie_age + marco_age = 37 →
  marie_age = 12 := by
sorry

end NUMINAMATH_CALUDE_maries_age_l3484_348434


namespace NUMINAMATH_CALUDE_count_no_adjacent_same_digits_eq_597880_l3484_348469

/-- Counts the number of integers from 0 to 999999 with no two adjacent digits being the same. -/
def count_no_adjacent_same_digits : ℕ :=
  10 + (9^2 + 9^3 + 9^4 + 9^5 + 9^6)

/-- Theorem stating that the count of integers from 0 to 999999 with no two adjacent digits 
    being the same is equal to 597880. -/
theorem count_no_adjacent_same_digits_eq_597880 : 
  count_no_adjacent_same_digits = 597880 := by
  sorry

end NUMINAMATH_CALUDE_count_no_adjacent_same_digits_eq_597880_l3484_348469


namespace NUMINAMATH_CALUDE_max_distance_sin_cosin_l3484_348412

/-- The maximum distance between sin x and sin(π/2 - x) for any real x is √2 -/
theorem max_distance_sin_cosin (x : ℝ) : 
  ∃ (m : ℝ), ∀ (x : ℝ), |Real.sin x - Real.sin (π/2 - x)| ≤ m ∧ 
  ∃ (y : ℝ), |Real.sin y - Real.sin (π/2 - y)| = m ∧ 
  m = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_max_distance_sin_cosin_l3484_348412


namespace NUMINAMATH_CALUDE_initial_water_is_11_l3484_348447

/-- Represents the hiking scenario with given conditions -/
structure HikingScenario where
  totalDistance : ℝ
  totalTime : ℝ
  waterRemaining : ℝ
  leakRate : ℝ
  lastMileConsumption : ℝ
  firstSixMilesRate : ℝ

/-- Calculates the initial amount of water in the canteen -/
def initialWater (scenario : HikingScenario) : ℝ :=
  scenario.waterRemaining +
  scenario.leakRate * scenario.totalTime +
  scenario.lastMileConsumption +
  scenario.firstSixMilesRate * (scenario.totalDistance - 1)

/-- Theorem stating that the initial amount of water is 11 cups -/
theorem initial_water_is_11 (scenario : HikingScenario)
  (h1 : scenario.totalDistance = 7)
  (h2 : scenario.totalTime = 2)
  (h3 : scenario.waterRemaining = 3)
  (h4 : scenario.leakRate = 1)
  (h5 : scenario.lastMileConsumption = 2)
  (h6 : scenario.firstSixMilesRate = 0.6666666666666666) :
  initialWater scenario = 11 := by
  sorry

end NUMINAMATH_CALUDE_initial_water_is_11_l3484_348447


namespace NUMINAMATH_CALUDE_median_unchanged_after_removing_extremes_l3484_348494

theorem median_unchanged_after_removing_extremes 
  (x : Fin 10 → ℝ) 
  (h_ordered : ∀ i j : Fin 10, i ≤ j → x i ≤ x j) :
  (x 4 + x 5) / 2 = (x 5 + x 6) / 2 := by
  sorry

end NUMINAMATH_CALUDE_median_unchanged_after_removing_extremes_l3484_348494


namespace NUMINAMATH_CALUDE_compare_logarithms_and_sqrt_l3484_348490

theorem compare_logarithms_and_sqrt : 
  let a := 2 * Real.log (21/20)
  let b := Real.log (11/10)
  let c := Real.sqrt 1.2 - 1
  c < a ∧ a < b :=
by sorry

end NUMINAMATH_CALUDE_compare_logarithms_and_sqrt_l3484_348490


namespace NUMINAMATH_CALUDE_expression_evaluation_l3484_348467

theorem expression_evaluation (x y z : ℤ) (hx : x = 25) (hy : y = 33) (hz : z = 7) :
  (x - (y - z)) - ((x - y) - z) = 14 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3484_348467


namespace NUMINAMATH_CALUDE_vectors_parallel_iff_l3484_348403

def a (m : ℝ) : Fin 2 → ℝ := ![1, m + 1]
def b (m : ℝ) : Fin 2 → ℝ := ![m, 2]

theorem vectors_parallel_iff (m : ℝ) :
  (∃ (k : ℝ), a m = k • b m) ↔ m = -2 ∨ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_vectors_parallel_iff_l3484_348403


namespace NUMINAMATH_CALUDE_square_perimeter_l3484_348475

theorem square_perimeter (rectangle_width : ℝ) (rectangle_length : ℝ) 
  (h1 : rectangle_length = 4 * rectangle_width) 
  (h2 : 28 * rectangle_width = 56) : 
  4 * (rectangle_width + rectangle_length) = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3484_348475


namespace NUMINAMATH_CALUDE_complex_division_simplification_l3484_348421

theorem complex_division_simplification : 
  let i : ℂ := Complex.I
  (2 - 3 * i) / (1 + i) = -1/2 - 5/2 * i := by sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l3484_348421


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3484_348463

theorem inequality_solution_set : 
  {x : ℝ | x * (x - 1) * (x - 2) > 0} = {x : ℝ | (0 < x ∧ x < 1) ∨ x > 2} := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3484_348463


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3484_348406

/-- A geometric sequence with positive terms. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 6 + 2 * a 4 * a 5 + a 5 ^ 2 = 25 →
  a 4 + a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3484_348406


namespace NUMINAMATH_CALUDE_two_x_equals_y_l3484_348441

theorem two_x_equals_y (x y : ℝ) 
  (h1 : (x + y) / 3 = 1) 
  (h2 : x + 2*y = 5) : 
  2*x = y := by sorry

end NUMINAMATH_CALUDE_two_x_equals_y_l3484_348441


namespace NUMINAMATH_CALUDE_prob_same_length_is_17_35_l3484_348443

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The set of all sides and diagonals of a regular hexagon -/
def S : Finset ℕ := Finset.range (num_sides + num_diagonals)

/-- The probability of selecting two segments of the same length from S -/
def prob_same_length : ℚ :=
  (Nat.choose num_sides 2 + Nat.choose num_diagonals 2) / Nat.choose S.card 2

theorem prob_same_length_is_17_35 : prob_same_length = 17 / 35 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_length_is_17_35_l3484_348443


namespace NUMINAMATH_CALUDE_parabola_directrix_l3484_348487

/-- Given a parabola with equation y = 4x^2 - 6, its directrix has equation y = -97/16 -/
theorem parabola_directrix (x y : ℝ) :
  y = 4 * x^2 - 6 → ∃ (k : ℝ), k = -97/16 ∧ (∀ (x₀ y₀ : ℝ), y₀ = 4 * x₀^2 - 6 → y₀ - k = (x₀ - 0)^2 + (y₀ - (k + 1/4))^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3484_348487


namespace NUMINAMATH_CALUDE_karcsi_travels_further_l3484_348431

def karcsi_speed : ℝ := 6
def joska_speed : ℝ := 4
def bus_speed : ℝ := 60

theorem karcsi_travels_further (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) 
  (h : x / karcsi_speed + (x + y) / bus_speed = y / joska_speed) : x > y := by
  sorry

end NUMINAMATH_CALUDE_karcsi_travels_further_l3484_348431


namespace NUMINAMATH_CALUDE_elevator_floors_l3484_348426

/-- The number of floors the elevator needs to move down. -/
def total_floors : ℕ := sorry

/-- The time taken for the first half of the floors (in minutes). -/
def first_half_time : ℕ := 15

/-- The time taken per floor for the next 5 floors (in minutes). -/
def middle_time_per_floor : ℕ := 5

/-- The number of floors in the middle section. -/
def middle_floors : ℕ := 5

/-- The time taken per floor for the final 5 floors (in minutes). -/
def final_time_per_floor : ℕ := 16

/-- The number of floors in the final section. -/
def final_floors : ℕ := 5

/-- The total time taken to reach the bottom (in minutes). -/
def total_time : ℕ := 120

theorem elevator_floors :
  first_half_time + 
  (middle_time_per_floor * middle_floors) + 
  (final_time_per_floor * final_floors) = total_time ∧
  total_floors = (total_floors / 2) + middle_floors + final_floors ∧
  total_floors = 20 := by sorry

end NUMINAMATH_CALUDE_elevator_floors_l3484_348426


namespace NUMINAMATH_CALUDE_parallelogram_area_parallelogram_area_proof_l3484_348472

/-- The area of a parallelogram with base 28 cm and height 32 cm is 896 square centimeters. -/
theorem parallelogram_area : ℝ → ℝ → ℝ → Prop :=
  fun base height area =>
    base = 28 ∧ height = 32 → area = base * height ∧ area = 896

/-- Proof of the theorem -/
theorem parallelogram_area_proof : parallelogram_area 28 32 896 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_parallelogram_area_proof_l3484_348472


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3484_348489

/-- Given a geometric sequence {a_n} with specific properties, prove a_7 = -2 -/
theorem geometric_sequence_property (a : ℕ → ℝ) :
  (∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q) →  -- geometric sequence condition
  a 2 * a 4 * a 5 = a 3 * a 6 →                   -- given condition
  a 9 * a 10 = -8 →                               -- given condition
  a 7 = -2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3484_348489


namespace NUMINAMATH_CALUDE_ufo_convention_attendees_l3484_348465

theorem ufo_convention_attendees (total : ℕ) (male : ℕ) 
  (h1 : total = 120) 
  (h2 : male = 62) 
  (h3 : male > total - male) : 
  male - (total - male) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ufo_convention_attendees_l3484_348465


namespace NUMINAMATH_CALUDE_timeDifference_div_by_40_l3484_348425

/-- Represents time in days, hours, minutes, and seconds -/
structure Time where
  days : ℕ
  hours : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Converts Time to its numerical representation (ignoring punctuation) -/
def Time.toNumerical (t : Time) : ℕ :=
  10^6 * t.days + 10^4 * t.hours + 100 * t.minutes + t.seconds

/-- Converts Time to total seconds -/
def Time.toSeconds (t : Time) : ℕ :=
  86400 * t.days + 3600 * t.hours + 60 * t.minutes + t.seconds

/-- The difference between numerical representation and total seconds -/
def timeDifference (t : Time) : ℤ :=
  (t.toNumerical : ℤ) - (t.toSeconds : ℤ)

/-- Theorem: 40 always divides the time difference -/
theorem timeDifference_div_by_40 (t : Time) : 
  (40 : ℤ) ∣ timeDifference t := by
  sorry

end NUMINAMATH_CALUDE_timeDifference_div_by_40_l3484_348425


namespace NUMINAMATH_CALUDE_simplify_fraction_l3484_348435

theorem simplify_fraction (a : ℝ) (h : a = 2) : 15 * a^5 / (75 * a^3) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3484_348435


namespace NUMINAMATH_CALUDE_submerged_sphere_segment_height_l3484_348450

/-- 
Theorem: For a homogeneous spherical segment of radius r floating in water, 
the height of the submerged portion m is equal to r/2 * (3 - √5) when it 
submerges up to the edge of its base spherical cap.
-/
theorem submerged_sphere_segment_height 
  (r : ℝ) -- radius of the sphere
  (h_pos : r > 0) -- assumption that radius is positive
  : ∃ m : ℝ, 
    -- m is the height of the submerged portion
    -- Volume of spherical sector
    (2 * π * m^3 / 3 = 
    -- Volume of submerged spherical segment
    π * m^2 * (3*r - m) / 3) ∧ 
    -- m is less than r (physical constraint)
    m < r ∧ 
    -- m equals the derived formula
    m = r/2 * (3 - Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_submerged_sphere_segment_height_l3484_348450


namespace NUMINAMATH_CALUDE_linear_combination_of_reals_with_rational_products_l3484_348439

theorem linear_combination_of_reals_with_rational_products 
  (a b c : ℝ) 
  (hab : ∃ (q : ℚ), a * b = q) 
  (hbc : ∃ (q : ℚ), b * c = q) 
  (hca : ∃ (q : ℚ), c * a = q) 
  (hnz : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) : 
  ∃ (x y z : ℤ), a * (x : ℝ) + b * (y : ℝ) + c * (z : ℝ) = 0 := by
sorry

end NUMINAMATH_CALUDE_linear_combination_of_reals_with_rational_products_l3484_348439


namespace NUMINAMATH_CALUDE_survivor_quitters_probability_l3484_348495

def total_people : ℕ := 18
def num_tribes : ℕ := 3
def people_per_tribe : ℕ := 6
def num_quitters : ℕ := 3

theorem survivor_quitters_probability :
  let total_ways := Nat.choose total_people num_quitters
  let same_tribe_ways := num_tribes * Nat.choose people_per_tribe num_quitters
  (same_tribe_ways : ℚ) / total_ways = 5 / 68 := by
    sorry

end NUMINAMATH_CALUDE_survivor_quitters_probability_l3484_348495


namespace NUMINAMATH_CALUDE_highway_mileage_l3484_348429

/-- Proves that the highway mileage is 37 mpg given the problem conditions -/
theorem highway_mileage (city_mpg : ℝ) (total_miles : ℝ) (total_gallons : ℝ) (highway_city_diff : ℝ) :
  city_mpg = 30 →
  total_miles = 365 →
  total_gallons = 11 →
  highway_city_diff = 5 →
  ∃ (city_miles highway_miles : ℝ),
    city_miles + highway_miles = total_miles ∧
    highway_miles = city_miles + highway_city_diff ∧
    city_miles / city_mpg + highway_miles / 37 = total_gallons :=
by sorry

end NUMINAMATH_CALUDE_highway_mileage_l3484_348429


namespace NUMINAMATH_CALUDE_alice_plate_stacking_l3484_348462

theorem alice_plate_stacking (initial_plates : ℕ) (first_addition : ℕ) (total_plates : ℕ) : 
  initial_plates = 27 → 
  first_addition = 37 → 
  total_plates = 83 → 
  total_plates - (initial_plates + first_addition) = 19 := by
sorry

end NUMINAMATH_CALUDE_alice_plate_stacking_l3484_348462


namespace NUMINAMATH_CALUDE_least_x_value_l3484_348407

theorem least_x_value (x p : ℕ) (h1 : x > 0) (h2 : Prime p) 
  (h3 : Prime (x / (9 * p))) (h4 : Odd (x / (9 * p))) :
  x ≥ 90 ∧ ∃ (x₀ : ℕ), x₀ = 90 ∧ 
    Prime (x₀ / (9 * p)) ∧ Odd (x₀ / (9 * p)) :=
sorry

end NUMINAMATH_CALUDE_least_x_value_l3484_348407


namespace NUMINAMATH_CALUDE_intersection_nonempty_iff_m_leq_neg_one_l3484_348478

/-- Sets A and B are defined as follows:
    A = {(x, y) | y = x^2 + mx + 2}
    B = {(x, y) | x - y + 1 = 0 and 0 ≤ x ≤ 2}
    This theorem states that A ∩ B is non-empty if and only if m ≤ -1 -/
theorem intersection_nonempty_iff_m_leq_neg_one (m : ℝ) :
  (∃ x y : ℝ, y = x^2 + m*x + 2 ∧ x - y + 1 = 0 ∧ 0 ≤ x ∧ x ≤ 2) ↔ m ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_iff_m_leq_neg_one_l3484_348478


namespace NUMINAMATH_CALUDE_farm_area_is_1200_l3484_348473

/-- Represents a rectangular farm with fencing on one long side, one short side, and the diagonal -/
structure RectangularFarm where
  short_side : ℝ
  long_side : ℝ
  diagonal : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ

/-- Calculates the area of a rectangular farm -/
def farm_area (farm : RectangularFarm) : ℝ :=
  farm.short_side * farm.long_side

/-- Calculates the total length of fencing required -/
def total_fencing_length (farm : RectangularFarm) : ℝ :=
  farm.short_side + farm.long_side + farm.diagonal

/-- The main theorem: If a rectangular farm satisfies the given conditions, its area is 1200 square meters -/
theorem farm_area_is_1200 (farm : RectangularFarm) 
    (h1 : farm.short_side = 30)
    (h2 : farm.fencing_cost_per_meter = 13)
    (h3 : farm.total_fencing_cost = 1560)
    (h4 : farm.total_fencing_cost = total_fencing_length farm * farm.fencing_cost_per_meter)
    (h5 : farm.diagonal^2 = farm.long_side^2 + farm.short_side^2) :
    farm_area farm = 1200 := by
  sorry


end NUMINAMATH_CALUDE_farm_area_is_1200_l3484_348473


namespace NUMINAMATH_CALUDE_special_triangle_area_special_triangle_area_is_48_l3484_348437

/-- A triangle with two sides of length 10 and 12, and a median to the third side of length 5 -/
structure SpecialTriangle where
  side1 : ℝ
  side2 : ℝ
  median : ℝ
  h_side1 : side1 = 10
  h_side2 : side2 = 12
  h_median : median = 5

/-- The area of a SpecialTriangle is 48 -/
theorem special_triangle_area (t : SpecialTriangle) : ℝ :=
  48

/-- The area of a SpecialTriangle is indeed 48 -/
theorem special_triangle_area_is_48 (t : SpecialTriangle) :
  special_triangle_area t = 48 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_area_special_triangle_area_is_48_l3484_348437


namespace NUMINAMATH_CALUDE_avery_wall_time_l3484_348445

/-- The time it takes Avery to build the wall alone -/
def avery_time : ℝ := 4

/-- The time it takes Tom to build the wall alone -/
def tom_time : ℝ := 2

/-- The additional time Tom needs to finish the wall after working with Avery for 1 hour -/
def tom_additional_time : ℝ := 0.5

theorem avery_wall_time : 
  (1 / avery_time + 1 / tom_time) + tom_additional_time / tom_time = 1 := by sorry

end NUMINAMATH_CALUDE_avery_wall_time_l3484_348445


namespace NUMINAMATH_CALUDE_calculate_expression_l3484_348440

theorem calculate_expression : -1^2023 - (-2)^3 - (-2) * (-3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3484_348440


namespace NUMINAMATH_CALUDE_adjacent_sum_theorem_l3484_348455

/-- Represents a 3x3 table with numbers from 1 to 9 -/
def Table := Fin 3 → Fin 3 → Fin 9

/-- Checks if a table contains each number from 1 to 9 exactly once -/
def isValidTable (t : Table) : Prop :=
  ∀ n : Fin 9, ∃! (i j : Fin 3), t i j = n

/-- Checks if the table has 1, 2, 3, and 4 in the correct positions -/
def hasCorrectCorners (t : Table) : Prop :=
  t 0 0 = 0 ∧ t 2 0 = 1 ∧ t 0 2 = 2 ∧ t 2 2 = 3

/-- Returns the sum of adjacent numbers to the given position -/
def adjacentSum (t : Table) (i j : Fin 3) : Nat :=
  (if i > 0 then (t (i-1) j).val + 1 else 0) +
  (if i < 2 then (t (i+1) j).val + 1 else 0) +
  (if j > 0 then (t i (j-1)).val + 1 else 0) +
  (if j < 2 then (t i (j+1)).val + 1 else 0)

/-- The main theorem to prove -/
theorem adjacent_sum_theorem (t : Table) 
  (valid : isValidTable t) 
  (corners : hasCorrectCorners t) 
  (sum_5 : ∃ i j : Fin 3, t i j = 4 ∧ adjacentSum t i j = 9) :
  ∃ i j : Fin 3, t i j = 5 ∧ adjacentSum t i j = 29 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_sum_theorem_l3484_348455


namespace NUMINAMATH_CALUDE_linear_function_common_quadrants_l3484_348432

/-- A linear function is represented by its slope and y-intercept -/
structure LinearFunction where
  k : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Quadrants of the Cartesian plane -/
inductive Quadrant
  | first
  | second
  | third
  | fourth

/-- The set of quadrants that a linear function passes through -/
def quadrants_passed (f : LinearFunction) : Set Quadrant :=
  sorry

/-- The common quadrants for all linear functions satisfying kb < 0 -/
def common_quadrants : Set Quadrant :=
  sorry

theorem linear_function_common_quadrants (f : LinearFunction) 
  (h : f.k * f.b < 0) : 
  quadrants_passed f ∩ common_quadrants = {Quadrant.first, Quadrant.fourth} :=
sorry

end NUMINAMATH_CALUDE_linear_function_common_quadrants_l3484_348432


namespace NUMINAMATH_CALUDE_oak_willow_difference_l3484_348418

theorem oak_willow_difference (total_trees : ℕ) (willows : ℕ) 
  (h1 : total_trees = 83) (h2 : willows = 36) : total_trees - willows - willows = 11 := by
  sorry

end NUMINAMATH_CALUDE_oak_willow_difference_l3484_348418


namespace NUMINAMATH_CALUDE_unique_solution_inequality_l3484_348427

theorem unique_solution_inequality (x : ℝ) :
  x > 0 →
  16 - x ≥ 0 →
  16 * x - x^3 ≥ 0 →
  x * Real.sqrt (16 - x) + Real.sqrt (16 * x - x^3) ≥ 16 →
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_inequality_l3484_348427


namespace NUMINAMATH_CALUDE_thirtieth_term_is_59_l3484_348452

def arithmetic_sequence (n : ℕ) : ℕ := 2 * n - 1

theorem thirtieth_term_is_59 : arithmetic_sequence 30 = 59 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_is_59_l3484_348452


namespace NUMINAMATH_CALUDE_parabola_coefficients_l3484_348479

/-- A parabola passing through (1, 1) with a tangent line of slope 1 at (2, -1) has coefficients a = 3, b = -11, and c = 9. -/
theorem parabola_coefficients : 
  ∀ (a b c : ℝ), 
  (a * 1^2 + b * 1 + c = 1) →  -- Passes through (1, 1)
  (a * 2^2 + b * 2 + c = -1) →  -- Passes through (2, -1)
  (2 * a * 2 + b = 1) →  -- Slope of tangent line at (2, -1) is 1
  (a = 3 ∧ b = -11 ∧ c = 9) := by
sorry

end NUMINAMATH_CALUDE_parabola_coefficients_l3484_348479


namespace NUMINAMATH_CALUDE_max_handshakes_l3484_348483

theorem max_handshakes (n : ℕ) (h : n = 25) : 
  (n * (n - 1)) / 2 = 300 :=
by
  sorry

end NUMINAMATH_CALUDE_max_handshakes_l3484_348483


namespace NUMINAMATH_CALUDE_board_symbols_l3484_348460

theorem board_symbols (total : Nat) (plus minus : Nat → Prop) : 
  total = 23 →
  (∀ n : Nat, n ≤ total → plus n ∨ minus n) →
  (∀ s : Finset Nat, s.card = 10 → ∃ i ∈ s, plus i) →
  (∀ s : Finset Nat, s.card = 15 → ∃ i ∈ s, minus i) →
  (∃! p m : Nat, p + m = total ∧ plus = λ i => i < p ∧ minus = λ i => p ≤ i ∧ i < total ∧ p = 14 ∧ m = 9) :=
by sorry

end NUMINAMATH_CALUDE_board_symbols_l3484_348460


namespace NUMINAMATH_CALUDE_points_collinear_l3484_348476

/-- Three points are collinear if the slope between any two pairs of points is equal. -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x2) = (y3 - y2) * (x2 - x1)

/-- The three given points are collinear. -/
theorem points_collinear : collinear (2, 5) (-6, -3) (-4, -1) := by
  sorry


end NUMINAMATH_CALUDE_points_collinear_l3484_348476


namespace NUMINAMATH_CALUDE_inscribed_square_distances_l3484_348497

/-- A circle with radius 5 containing an inscribed square -/
structure InscribedSquareCircle where
  radius : ℝ
  radius_eq : radius = 5

/-- A point on the circumference of the circle -/
structure CircumferencePoint (c : InscribedSquareCircle) where
  point : ℝ × ℝ
  on_circle : (point.1 - c.radius)^2 + point.2^2 = c.radius^2

/-- Vertices of the inscribed square -/
def square_vertices (c : InscribedSquareCircle) : Fin 4 → ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating the distances from a point on the circumference to the square vertices -/
theorem inscribed_square_distances
  (c : InscribedSquareCircle)
  (m : CircumferencePoint c)
  (h : distance m.point (square_vertices c 0) = 6) :
  ∃ (perm : Fin 3 → Fin 3),
    distance m.point (square_vertices c 1) = 8 ∧
    distance m.point (square_vertices c 2) = Real.sqrt 2 ∧
    distance m.point (square_vertices c 3) = 7 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_distances_l3484_348497


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l3484_348448

-- Problem 1
theorem problem_one : -2^2 - |2 - 5| + (-1) * 2 = -1 := by sorry

-- Problem 2
theorem problem_two : ∃! x : ℝ, 5 * x - 2 = 3 * x + 18 ∧ x = 10 := by sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l3484_348448


namespace NUMINAMATH_CALUDE_trivia_team_groups_l3484_348492

theorem trivia_team_groups (total_students : ℕ) (not_picked : ℕ) (students_per_group : ℕ) :
  total_students = 64 →
  not_picked = 36 →
  students_per_group = 7 →
  (total_students - not_picked) / students_per_group = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_trivia_team_groups_l3484_348492


namespace NUMINAMATH_CALUDE_class_notification_problem_l3484_348438

theorem class_notification_problem (n : ℕ) : 
  (1 + n + n^2 = 43) ↔ (n = 6) :=
by sorry

end NUMINAMATH_CALUDE_class_notification_problem_l3484_348438


namespace NUMINAMATH_CALUDE_product_ab_equals_six_l3484_348488

def A : Set ℝ := {-1, 3}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b = 0}

theorem product_ab_equals_six (a b : ℝ) (h : A = B a b) : a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_ab_equals_six_l3484_348488


namespace NUMINAMATH_CALUDE_fruit_salad_ratio_l3484_348457

def total_salads : ℕ := 600
def alaya_salads : ℕ := 200

theorem fruit_salad_ratio :
  let angel_salads := total_salads - alaya_salads
  (angel_salads : ℚ) / alaya_salads = 2 := by
  sorry

end NUMINAMATH_CALUDE_fruit_salad_ratio_l3484_348457


namespace NUMINAMATH_CALUDE_combination_number_identity_l3484_348485

theorem combination_number_identity (n r : ℕ) (h1 : n > r) (h2 : r ≥ 1) :
  Nat.choose n r = (n / r) * Nat.choose (n - 1) (r - 1) := by
  sorry

end NUMINAMATH_CALUDE_combination_number_identity_l3484_348485


namespace NUMINAMATH_CALUDE_modulus_of_z_l3484_348428

theorem modulus_of_z (z : ℂ) (h : (z - Complex.I) * Complex.I = 1 + Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3484_348428


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3484_348484

theorem fourth_root_equation_solutions :
  {x : ℝ | x > 0 ∧ (x^(1/4) = 20 / (9 - x^(1/4)))} = {256, 625} := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3484_348484


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3484_348419

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4 < 0}
def B : Set ℝ := {x | x < 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 < x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3484_348419


namespace NUMINAMATH_CALUDE_log_sum_equality_l3484_348430

theorem log_sum_equality : Real.log 4 / Real.log 10 + 2 * (Real.log 5 / Real.log 10) = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l3484_348430


namespace NUMINAMATH_CALUDE_starters_with_twin_l3484_348413

def total_players : ℕ := 16
def starters : ℕ := 6
def twins : ℕ := 2

theorem starters_with_twin (total_players starters twins : ℕ) :
  total_players = 16 →
  starters = 6 →
  twins = 2 →
  (Nat.choose total_players starters) - (Nat.choose (total_players - twins) starters) = 5005 := by
  sorry

end NUMINAMATH_CALUDE_starters_with_twin_l3484_348413


namespace NUMINAMATH_CALUDE_workshop_workers_correct_l3484_348422

/-- The number of workers in a workshop with given salary conditions -/
def workshop_workers : ℕ :=
  let average_salary : ℚ := 750
  let technician_count : ℕ := 5
  let technician_salary : ℚ := 900
  let non_technician_salary : ℚ := 700
  20

/-- Proof that the number of workers in the workshop is correct -/
theorem workshop_workers_correct :
  let average_salary : ℚ := 750
  let technician_count : ℕ := 5
  let technician_salary : ℚ := 900
  let non_technician_salary : ℚ := 700
  let total_workers := workshop_workers
  (average_salary * total_workers : ℚ) =
    technician_salary * technician_count +
    non_technician_salary * (total_workers - technician_count) :=
by
  sorry

#eval workshop_workers

end NUMINAMATH_CALUDE_workshop_workers_correct_l3484_348422


namespace NUMINAMATH_CALUDE_correct_average_weight_l3484_348417

/-- Given a class of 20 boys with an initial average weight and a misread weight,
    calculate the correct average weight. -/
theorem correct_average_weight
  (num_boys : ℕ)
  (initial_avg : ℝ)
  (misread_weight : ℝ)
  (correct_weight : ℝ)
  (h1 : num_boys = 20)
  (h2 : initial_avg = 58.4)
  (h3 : misread_weight = 56)
  (h4 : correct_weight = 62) :
  (num_boys : ℝ) * initial_avg + (correct_weight - misread_weight) = num_boys * 58.7 :=
by sorry

#check correct_average_weight

end NUMINAMATH_CALUDE_correct_average_weight_l3484_348417


namespace NUMINAMATH_CALUDE_amanda_coffee_blend_typeA_quantity_l3484_348496

/-- Represents the cost and quantity of coffee in Amanda's Coffee Shop blend --/
structure CoffeeBlend where
  typeA_cost : ℝ
  typeB_cost : ℝ
  typeA_quantity : ℝ
  typeB_quantity : ℝ
  total_cost : ℝ

/-- Theorem stating the quantity of type A coffee in the blend --/
theorem amanda_coffee_blend_typeA_quantity (blend : CoffeeBlend) 
  (h1 : blend.typeA_cost = 4.60)
  (h2 : blend.typeB_cost = 5.95)
  (h3 : blend.typeB_quantity = 2 * blend.typeA_quantity)
  (h4 : blend.total_cost = 511.50)
  (h5 : blend.total_cost = blend.typeA_cost * blend.typeA_quantity + blend.typeB_cost * blend.typeB_quantity) :
  blend.typeA_quantity = 31 := by
  sorry


end NUMINAMATH_CALUDE_amanda_coffee_blend_typeA_quantity_l3484_348496


namespace NUMINAMATH_CALUDE_decreasing_f_implies_a_geq_2_l3484_348486

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x - 3

-- State the theorem
theorem decreasing_f_implies_a_geq_2 :
  ∀ a : ℝ, (∀ x y : ℝ, -8 < x ∧ x < y ∧ y < 2 → f a x > f a y) → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_f_implies_a_geq_2_l3484_348486


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_two_l3484_348404

theorem gcd_of_powers_of_two : Nat.gcd (2^1015 - 1) (2^1024 - 1) = 2^9 - 1 := by sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_two_l3484_348404


namespace NUMINAMATH_CALUDE_online_employees_probability_l3484_348402

/-- Probability of exactly k successes in n independent trials with probability p each -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ := sorry

/-- Probability of at least k successes in n independent trials with probability p each -/
def at_least_probability (n k : ℕ) (p : ℝ) : ℝ := sorry

theorem online_employees_probability (n : ℕ) (p : ℝ) 
  (h_n : n = 6) (h_p : p = 0.5) : 
  at_least_probability n 3 p = 21/32 ∧ 
  (∀ k : ℕ, at_least_probability n k p < 0.3 ↔ k ≥ 4) := by sorry

end NUMINAMATH_CALUDE_online_employees_probability_l3484_348402


namespace NUMINAMATH_CALUDE_book_pages_book_pages_proof_l3484_348408

theorem book_pages : ℝ → Prop :=
  fun x => 
    let day1_read := x / 4 + 17
    let day1_remain := x - day1_read
    let day2_read := day1_remain / 3 + 20
    let day2_remain := day1_remain - day2_read
    let day3_read := day2_remain / 2 + 23
    let day3_remain := day2_remain - day3_read
    day3_remain = 70 → x = 394

-- The proof goes here
theorem book_pages_proof : ∃ x : ℝ, book_pages x := by
  sorry

end NUMINAMATH_CALUDE_book_pages_book_pages_proof_l3484_348408


namespace NUMINAMATH_CALUDE_existence_of_polynomials_l3484_348424

theorem existence_of_polynomials : ∃ (p q : Polynomial ℤ),
  (∃ (i j : ℕ), (abs (p.coeff i) > 2015) ∧ (abs (q.coeff j) > 2015)) ∧
  (∀ k : ℕ, abs ((p * q).coeff k) ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_polynomials_l3484_348424


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3484_348409

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_a2 : a 2 = 2)
  (h_a5 : a 5 = 1/4) :
  ∃ q : ℝ, q = 1/2 ∧ ∀ n : ℕ, a (n + 1) = a n * q := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3484_348409


namespace NUMINAMATH_CALUDE_square_diagonal_perimeter_l3484_348477

theorem square_diagonal_perimeter (d : ℝ) (h : d = 20) :
  let side := d / Real.sqrt 2
  4 * side = 40 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_perimeter_l3484_348477


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3484_348458

def a : ℝ × ℝ := (2, 1)
def b (x : ℝ) : ℝ × ℝ := (3, x)

theorem perpendicular_vectors (x : ℝ) : 
  (a.1 * (b x).1 + a.2 * (b x).2 = 0) → x = -6 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3484_348458


namespace NUMINAMATH_CALUDE_sum_of_solutions_l3484_348482

theorem sum_of_solutions (N : ℝ) : (N * (N + 4) = 8) → (∃ x y : ℝ, x + y = -4 ∧ x * (x + 4) = 8 ∧ y * (y + 4) = 8) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l3484_348482


namespace NUMINAMATH_CALUDE_calculation_proof_l3484_348470

theorem calculation_proof : 101 * 102^2 - 101 * 98^2 = 80800 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3484_348470


namespace NUMINAMATH_CALUDE_jasons_books_l3484_348453

/-- Given that Keith has 20 books and together with Jason they have 41 books,
    prove that Jason has 21 books. -/
theorem jasons_books (keith_books : ℕ) (total_books : ℕ) (h1 : keith_books = 20) (h2 : total_books = 41) :
  total_books - keith_books = 21 := by
  sorry

end NUMINAMATH_CALUDE_jasons_books_l3484_348453


namespace NUMINAMATH_CALUDE_solution_set_equality_l3484_348415

def equation (x : ℝ) : Prop :=
  (1 / (x^2 + 12*x - 9)) + (1 / (x^2 + 3*x - 9)) + (1 / (x^2 - 12*x - 9)) = 0

theorem solution_set_equality :
  {x : ℝ | equation x} = {1, -9, 3, -3} := by sorry

end NUMINAMATH_CALUDE_solution_set_equality_l3484_348415


namespace NUMINAMATH_CALUDE_exists_same_color_transformation_l3484_348459

/-- Represents the color of a square on the chessboard -/
inductive Color
| Black
| White

/-- Represents a 16x16 chessboard -/
def Chessboard := Fin 16 → Fin 16 → Color

/-- Initial chessboard with alternating colors -/
def initialChessboard : Chessboard :=
  fun i j => if (i.val + j.val) % 2 = 0 then Color.Black else Color.White

/-- Apply operation A to the chessboard at position (i, j) -/
def applyOperationA (board : Chessboard) (i j : Fin 16) : Chessboard :=
  fun x y =>
    if x = i || y = j then
      match board x y with
      | Color.Black => Color.White
      | Color.White => Color.Black
    else
      board x y

/-- Check if all squares on the chessboard have the same color -/
def allSameColor (board : Chessboard) : Prop :=
  ∀ i j : Fin 16, board i j = board 0 0

/-- Theorem: There exists a sequence of operations A that transforms all squares to the same color -/
theorem exists_same_color_transformation :
  ∃ (operations : List (Fin 16 × Fin 16)),
    allSameColor (operations.foldl (fun b (i, j) => applyOperationA b i j) initialChessboard) :=
  sorry

end NUMINAMATH_CALUDE_exists_same_color_transformation_l3484_348459


namespace NUMINAMATH_CALUDE_polygon_sides_l3484_348449

theorem polygon_sides (sum_angles : ℕ) (h1 : sum_angles = 1980) : ∃ n : ℕ, n = 13 ∧ sum_angles = 180 * (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l3484_348449


namespace NUMINAMATH_CALUDE_problem_statement_l3484_348433

theorem problem_statement (a : ℝ) : 
  let A : Set ℝ := {1, 2, a + 3}
  let B : Set ℝ := {a, 5}
  A ∪ B = A → a = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3484_348433


namespace NUMINAMATH_CALUDE_cos_alpha_value_l3484_348456

/-- Given an angle α in the second quadrant, if the slope of the line 2x + (tan α)y + 1 = 0 is 8/3, 
    then cos α = -4/5 -/
theorem cos_alpha_value (α : Real) : 
  (π/2 < α ∧ α < π) →  -- α is in the second quadrant
  (-(2 : Real) / Real.tan α = 8/3) →  -- slope of the line
  Real.cos α = -4/5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l3484_348456


namespace NUMINAMATH_CALUDE_triangle_ABC_angle_proof_l3484_348491

def triangle_ABC_angle (A B C : ℝ × ℝ) : Prop :=
  let BA : ℝ × ℝ := (Real.sqrt 3, 1)
  let BC : ℝ × ℝ := (0, 1)
  let AB : ℝ × ℝ := (-BA.1, -BA.2)
  let angle := Real.arccos (AB.1 * BC.1 + AB.2 * BC.2) / 
               (Real.sqrt (AB.1^2 + AB.2^2) * Real.sqrt (BC.1^2 + BC.2^2))
  angle = 2 * Real.pi / 3

theorem triangle_ABC_angle_proof (A B C : ℝ × ℝ) : 
  triangle_ABC_angle A B C := by sorry

end NUMINAMATH_CALUDE_triangle_ABC_angle_proof_l3484_348491


namespace NUMINAMATH_CALUDE_episode_length_l3484_348454

/-- Given a TV mini series with 6 episodes and a total watching time of 5 hours,
    prove that the length of each episode is 50 minutes. -/
theorem episode_length (num_episodes : ℕ) (total_time : ℕ) : 
  num_episodes = 6 → total_time = 5 * 60 → total_time / num_episodes = 50 := by
  sorry

end NUMINAMATH_CALUDE_episode_length_l3484_348454


namespace NUMINAMATH_CALUDE_log_inequality_equiv_interval_l3484_348416

-- Define the logarithm function with base 2
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem log_inequality_equiv_interval (x : ℝ) :
  (log2 (4 - x) > log2 (3 * x)) ↔ (0 < x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_equiv_interval_l3484_348416


namespace NUMINAMATH_CALUDE_complex_magnitude_theorem_l3484_348410

theorem complex_magnitude_theorem (z₁ z₂ z₃ : ℂ) (a b c : ℝ) 
  (h₁ : (z₁ / z₂ + z₂ / z₃ + z₃ / z₁).im = 0)
  (h₂ : Complex.abs z₁ = 1)
  (h₃ : Complex.abs z₂ = 1)
  (h₄ : Complex.abs z₃ = 1) :
  ∃ (x : ℝ), x = Complex.abs (a * z₁ + b * z₂ + c * z₃) ∧
    (x = Real.sqrt ((a + b)^2 + c^2) ∨
     x = Real.sqrt ((a + c)^2 + b^2) ∨
     x = Real.sqrt ((b + c)^2 + a^2)) :=
by sorry

end NUMINAMATH_CALUDE_complex_magnitude_theorem_l3484_348410


namespace NUMINAMATH_CALUDE_only_η_is_hypergeometric_l3484_348405

-- Define the types for balls and random variables
inductive BallColor
| Black
| White

structure Ball :=
  (color : BallColor)
  (number : Nat)

def TotalBalls : Nat := 10
def BlackBalls : Nat := 6
def WhiteBalls : Nat := 4
def DrawnBalls : Nat := 4

-- Define the random variables
def X (draw : Finset Ball) : Nat := sorry
def Y (draw : Finset Ball) : Nat := sorry
def ξ (draw : Finset Ball) : Nat := sorry
def η (draw : Finset Ball) : Nat := sorry

-- Define the hypergeometric distribution
def IsHypergeometric (f : (Finset Ball) → Nat) : Prop := sorry

-- State the theorem
theorem only_η_is_hypergeometric :
  IsHypergeometric η ∧
  ¬IsHypergeometric X ∧
  ¬IsHypergeometric Y ∧
  ¬IsHypergeometric ξ :=
sorry

end NUMINAMATH_CALUDE_only_η_is_hypergeometric_l3484_348405


namespace NUMINAMATH_CALUDE_max_value_of_product_sum_l3484_348444

theorem max_value_of_product_sum (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a + b + c + d = 200 → 
  a * b + b * c + c * d ≤ 2500 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_product_sum_l3484_348444


namespace NUMINAMATH_CALUDE_fine_calculation_l3484_348461

/-- Calculates the fine for inappropriate items in the recycling bin -/
def calculate_fine (weeks : ℕ) (trash_bin_cost : ℚ) (recycling_bin_cost : ℚ) 
  (trash_bins : ℕ) (recycling_bins : ℕ) (discount_percent : ℚ) (total_bill : ℚ) : ℚ := 
  let weekly_cost := trash_bin_cost * trash_bins + recycling_bin_cost * recycling_bins
  let monthly_cost := weekly_cost * weeks
  let discount := discount_percent * monthly_cost
  let discounted_cost := monthly_cost - discount
  total_bill - discounted_cost

theorem fine_calculation :
  calculate_fine 4 10 5 2 1 (18/100) 102 = 20 := by
  sorry

end NUMINAMATH_CALUDE_fine_calculation_l3484_348461


namespace NUMINAMATH_CALUDE_tan_ratio_max_tan_A_l3484_348474

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  t.b^2 + 3*t.a^2 = t.c^2

-- Theorem 1: tan(C) / tan(B) = -2
theorem tan_ratio (t : Triangle) (h : triangle_condition t) :
  Real.tan t.C / Real.tan t.B = -2 := by sorry

-- Theorem 2: Maximum value of tan(A) is √2/4
theorem max_tan_A (t : Triangle) (h : triangle_condition t) :
  ∃ (max_tan_A : ℝ), (∀ (t' : Triangle), triangle_condition t' → Real.tan t'.A ≤ max_tan_A) ∧ max_tan_A = Real.sqrt 2 / 4 := by sorry

end NUMINAMATH_CALUDE_tan_ratio_max_tan_A_l3484_348474


namespace NUMINAMATH_CALUDE_problem_l_shape_surface_area_l3484_348464

/-- Represents a 3D L-shaped structure made of unit cubes -/
structure LShape where
  verticalHeight : ℕ
  verticalWidth : ℕ
  horizontalLength : ℕ
  totalCubes : ℕ

/-- Calculates the surface area of an L-shaped structure -/
def surfaceArea (l : LShape) : ℕ :=
  sorry

/-- The specific L-shape described in the problem -/
def problemLShape : LShape :=
  { verticalHeight := 3
  , verticalWidth := 2
  , horizontalLength := 3
  , totalCubes := 15 }

/-- Theorem stating that the surface area of the problem's L-shape is 34 square units -/
theorem problem_l_shape_surface_area :
  surfaceArea problemLShape = 34 :=
sorry

end NUMINAMATH_CALUDE_problem_l_shape_surface_area_l3484_348464


namespace NUMINAMATH_CALUDE_inequality_preservation_l3484_348451

theorem inequality_preservation (a b : ℝ) (h : a > b) : a - 5 > b - 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3484_348451


namespace NUMINAMATH_CALUDE_incorrect_transformation_l3484_348498

theorem incorrect_transformation (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a / 2 = b / 3) :
  ¬(2 * a = 3 * b) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_transformation_l3484_348498


namespace NUMINAMATH_CALUDE_dot_product_equality_l3484_348411

/-- Given points in 2D space, prove that OA · OP₃ = OP₁ · OP₂ -/
theorem dot_product_equality (α β : ℝ) :
  let O : ℝ × ℝ := (0, 0)
  let P₁ : ℝ × ℝ := (Real.cos α, Real.sin α)
  let P₂ : ℝ × ℝ := (Real.cos β, -Real.sin β)
  let P₃ : ℝ × ℝ := (Real.cos (α + β), Real.sin (α + β))
  let A : ℝ × ℝ := (1, 0)
  (A.1 - O.1) * (P₃.1 - O.1) + (A.2 - O.2) * (P₃.2 - O.2) =
  (P₁.1 - O.1) * (P₂.1 - O.1) + (P₁.2 - O.2) * (P₂.2 - O.2) :=
by
  sorry

#check dot_product_equality

end NUMINAMATH_CALUDE_dot_product_equality_l3484_348411


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_ratio_l3484_348400

theorem hyperbola_asymptote_ratio (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) → 
  (Real.arctan (2 * (b / a) / (1 - (b / a)^2)) = π / 4) →
  a / b = 1 / (-1 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_ratio_l3484_348400


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l3484_348493

theorem line_segment_endpoint (x : ℝ) : 
  x > 0 → 
  ((4 - 1)^2 + (x - 3)^2 : ℝ) = 5^2 → 
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l3484_348493


namespace NUMINAMATH_CALUDE_cat_mouse_problem_l3484_348481

theorem cat_mouse_problem (n : ℕ+) (h1 : n * (n + 18) = 999919) : n = 991 := by
  sorry

end NUMINAMATH_CALUDE_cat_mouse_problem_l3484_348481


namespace NUMINAMATH_CALUDE_zach_stadium_goal_l3484_348480

/-- The number of stadiums Zach wants to visit --/
def num_stadiums : ℕ := 30

/-- The cost per stadium in dollars --/
def cost_per_stadium : ℕ := 900

/-- Zach's yearly savings in dollars --/
def yearly_savings : ℕ := 1500

/-- The number of years to accomplish the goal --/
def years_to_goal : ℕ := 18

/-- Theorem stating that the number of stadiums Zach wants to visit is 30 --/
theorem zach_stadium_goal :
  num_stadiums = (yearly_savings * years_to_goal) / cost_per_stadium :=
by sorry

end NUMINAMATH_CALUDE_zach_stadium_goal_l3484_348480


namespace NUMINAMATH_CALUDE_mother_three_times_daughter_age_l3484_348414

/-- Proves that in 9 years, a mother who is currently 42 years old will be three times as old as her daughter who is currently 8 years old. -/
theorem mother_three_times_daughter_age (mother_age : ℕ) (daughter_age : ℕ) (years : ℕ) : 
  mother_age = 42 → daughter_age = 8 → years = 9 → 
  mother_age + years = 3 * (daughter_age + years) :=
by sorry

end NUMINAMATH_CALUDE_mother_three_times_daughter_age_l3484_348414


namespace NUMINAMATH_CALUDE_other_endpoint_coordinates_l3484_348446

/-- Given a line segment with midpoint (3, 0) and one endpoint at (7, -4), 
    prove that the other endpoint is at (-1, 4) -/
theorem other_endpoint_coordinates :
  ∀ (A B : ℝ × ℝ),
    (A.1 + B.1) / 2 = 3 ∧
    (A.2 + B.2) / 2 = 0 ∧
    A = (7, -4) →
    B = (-1, 4) := by
  sorry

end NUMINAMATH_CALUDE_other_endpoint_coordinates_l3484_348446


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3484_348442

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) :
  surface_area = 150 →
  (6 : ℝ) * (volume ^ (1/3 : ℝ))^2 = surface_area →
  volume = 125 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3484_348442


namespace NUMINAMATH_CALUDE_chess_game_probability_l3484_348420

theorem chess_game_probability (prob_draw prob_B_win : ℚ) 
  (h1 : prob_draw = 1/2) 
  (h2 : prob_B_win = 1/3) : 
  1 - prob_draw - prob_B_win = 1/6 := by
sorry

end NUMINAMATH_CALUDE_chess_game_probability_l3484_348420


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l3484_348436

theorem smallest_part_of_proportional_division (total : ℕ) (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 90 ∧ a = 3 ∧ b = 5 ∧ c = 7 →
  ∃ x : ℚ, x > 0 ∧ total = a * x + b * x + c * x ∧ min (a * x) (min (b * x) (c * x)) = 18 :=
by sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l3484_348436


namespace NUMINAMATH_CALUDE_fraction_simplification_l3484_348401

theorem fraction_simplification :
  (5 : ℝ) / (2 * Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 18) = (5 * Real.sqrt 2) / 38 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3484_348401


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l3484_348468

theorem quadratic_equation_m_value :
  ∀ m : ℝ,
  (∀ x : ℝ, (m + 1) * x^(m * (m - 2) - 1) + 2 * m * x - 1 = 0 → ∃ a b c : ℝ, a ≠ 0 ∧ (m + 1) * x^(m * (m - 2) - 1) + 2 * m * x - 1 = a * x^2 + b * x + c) →
  m = 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l3484_348468
