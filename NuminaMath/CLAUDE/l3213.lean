import Mathlib

namespace NUMINAMATH_CALUDE_sector_area_l3213_321335

/-- Given a circular sector with arc length 3π and central angle 3/4π, its area is 6π. -/
theorem sector_area (r : ℝ) (h1 : (3/4) * π * r = 3 * π) : (1/2) * (3/4 * π) * r^2 = 6 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3213_321335


namespace NUMINAMATH_CALUDE_max_value_3xy_plus_yz_l3213_321333

theorem max_value_3xy_plus_yz (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  3*x*y + y*z ≤ Real.sqrt 10 / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_3xy_plus_yz_l3213_321333


namespace NUMINAMATH_CALUDE_multiple_identification_l3213_321327

/-- Given two integers a and b that are multiples of n, and q is the set of consecutive integers
    between a and b (inclusive), prove that if q contains 11 multiples of n and 21 multiples of 7,
    then n = 14. -/
theorem multiple_identification (a b n : ℕ) (q : Finset ℕ) (h1 : a ∣ n) (h2 : b ∣ n)
    (h3 : q = Finset.Icc a b) (h4 : (q.filter (· ∣ n)).card = 11)
    (h5 : (q.filter (· ∣ 7)).card = 21) : n = 14 := by
  sorry

end NUMINAMATH_CALUDE_multiple_identification_l3213_321327


namespace NUMINAMATH_CALUDE_unique_solution_l3213_321359

/-- Given a real number a, returns the sum of coefficients of odd powers of x 
    in the expansion of (1+ax)^2(1-x)^5 -/
def oddPowerSum (a : ℝ) : ℝ := sorry

theorem unique_solution : 
  ∃! (a : ℝ), a > 0 ∧ oddPowerSum a = -64 :=
by
  use 3
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3213_321359


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_equilateral_l3213_321399

/-- An ellipse with a vertex and foci forming an equilateral triangle has eccentricity 1/2 -/
theorem ellipse_eccentricity_equilateral (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive semi-axes and focal distance
  a^2 = b^2 + c^2 →        -- Relationship between semi-axes and focal distance
  b = Real.sqrt 3 * c →    -- Condition for equilateral triangle
  c / a = 1 / 2 :=         -- Eccentricity definition and target value
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_equilateral_l3213_321399


namespace NUMINAMATH_CALUDE_jack_hunting_frequency_l3213_321369

/-- Represents the hunting scenario for Jack --/
structure HuntingScenario where
  seasonLength : ℚ  -- Length of hunting season in quarters of a year
  deersPerTrip : ℕ  -- Number of deers caught per hunting trip
  deerWeight : ℕ    -- Weight of each deer in pounds
  keepRatio : ℚ     -- Ratio of deer weight kept per year
  keptWeight : ℕ    -- Total weight of deer kept in pounds

/-- Calculates the number of hunting trips per month --/
def tripsPerMonth (scenario : HuntingScenario) : ℚ :=
  let totalWeight := scenario.keptWeight / scenario.keepRatio
  let weightPerTrip := scenario.deersPerTrip * scenario.deerWeight
  let tripsPerYear := totalWeight / weightPerTrip
  let monthsInSeason := scenario.seasonLength * 12
  tripsPerYear / monthsInSeason

/-- Theorem stating that Jack goes hunting 6 times per month --/
theorem jack_hunting_frequency :
  let scenario : HuntingScenario := {
    seasonLength := 1/4,
    deersPerTrip := 2,
    deerWeight := 600,
    keepRatio := 1/2,
    keptWeight := 10800
  }
  tripsPerMonth scenario = 6 := by sorry

end NUMINAMATH_CALUDE_jack_hunting_frequency_l3213_321369


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_binomial_expansion_l3213_321393

theorem coefficient_x_squared_in_binomial_expansion :
  let binomial := (x + 2/x)^4
  ∃ (a b c d e : ℝ), binomial = a*x^4 + b*x^3 + c*x^2 + d*x + e ∧ c = 8 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_binomial_expansion_l3213_321393


namespace NUMINAMATH_CALUDE_exists_zero_implies_a_range_l3213_321379

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * abs x - 3 * a - 1

-- State the theorem
theorem exists_zero_implies_a_range (a : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc (-1) 1 ∧ f a x₀ = 0) →
  a ∈ Set.Icc (-1/2) (-1/3) :=
by sorry

end NUMINAMATH_CALUDE_exists_zero_implies_a_range_l3213_321379


namespace NUMINAMATH_CALUDE_seven_balls_three_boxes_l3213_321316

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 36 ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
theorem seven_balls_three_boxes : distribute_balls 7 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_three_boxes_l3213_321316


namespace NUMINAMATH_CALUDE_decimal_point_problem_l3213_321319

theorem decimal_point_problem (x : ℝ) (h1 : x > 0) (h2 : 100 * x = 9 / x) : x = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l3213_321319


namespace NUMINAMATH_CALUDE_sum_of_decimals_l3213_321337

/-- The sum of 5.47 and 2.359 is equal to 7.829 -/
theorem sum_of_decimals : (5.47 : ℚ) + (2.359 : ℚ) = (7.829 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l3213_321337


namespace NUMINAMATH_CALUDE_sale_ratio_l3213_321371

def floral_shop_sales (monday_sales : ℕ) (total_sales : ℕ) : Prop :=
  let tuesday_sales := 3 * monday_sales
  let wednesday_sales := total_sales - (monday_sales + tuesday_sales)
  (wednesday_sales : ℚ) / tuesday_sales = 1 / 3

theorem sale_ratio : floral_shop_sales 12 60 := by
  sorry

end NUMINAMATH_CALUDE_sale_ratio_l3213_321371


namespace NUMINAMATH_CALUDE_shaded_area_fraction_l3213_321346

theorem shaded_area_fraction (s : ℝ) (h : s > 0) : 
  let square_area := s^2
  let triangle_area := (1/2) * (s/2) * (s/2)
  let shaded_area := 2 * triangle_area
  shaded_area / square_area = (1 : ℝ) / 4 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_fraction_l3213_321346


namespace NUMINAMATH_CALUDE_max_guaranteed_pastries_l3213_321395

/-- Represents a game with circular arrangement of plates and pastries. -/
structure PastryGame where
  num_plates : Nat
  max_move : Nat

/-- Represents the result of the game. -/
inductive GameResult
  | CanGuarantee
  | CannotGuarantee

/-- Determines if a certain number of pastries can be guaranteed on a single plate. -/
def can_guarantee (game : PastryGame) (k : Nat) : GameResult :=
  sorry

/-- The main theorem stating the maximum number of pastries that can be guaranteed. -/
theorem max_guaranteed_pastries (game : PastryGame) : 
  game.num_plates = 2019 → game.max_move = 16 → can_guarantee game 32 = GameResult.CanGuarantee ∧ 
  can_guarantee game 33 = GameResult.CannotGuarantee :=
  sorry

end NUMINAMATH_CALUDE_max_guaranteed_pastries_l3213_321395


namespace NUMINAMATH_CALUDE_arrangement_counts_l3213_321300

/-- Represents the number of teachers -/
def num_teachers : Nat := 2

/-- Represents the number of students -/
def num_students : Nat := 4

/-- Represents the total number of people -/
def total_people : Nat := num_teachers + num_students

/-- Calculates the number of arrangements with teachers at the ends -/
def arrangements_teachers_at_ends : Nat :=
  Nat.factorial num_students * Nat.factorial num_teachers

/-- Calculates the number of arrangements with teachers next to each other -/
def arrangements_teachers_together : Nat :=
  Nat.factorial (total_people - 1) * Nat.factorial num_teachers

/-- Calculates the number of arrangements with teachers not next to each other -/
def arrangements_teachers_apart : Nat :=
  Nat.factorial num_students * (num_students + 1) * (num_students + 1)

/-- Calculates the number of arrangements with two students between teachers -/
def arrangements_two_students_between : Nat :=
  (Nat.factorial num_students / (Nat.factorial 2 * Nat.factorial (num_students - 2))) *
  Nat.factorial num_teachers * Nat.factorial 3

theorem arrangement_counts :
  arrangements_teachers_at_ends = 48 ∧
  arrangements_teachers_together = 240 ∧
  arrangements_teachers_apart = 480 ∧
  arrangements_two_students_between = 144 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_counts_l3213_321300


namespace NUMINAMATH_CALUDE_carousel_horse_ratio_l3213_321320

theorem carousel_horse_ratio : 
  ∀ (purple green gold : ℕ),
  purple > 0 →
  green = 2 * purple →
  gold = green / 6 →
  3 + purple + green + gold = 33 →
  (purple : ℚ) / 3 = 3 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_carousel_horse_ratio_l3213_321320


namespace NUMINAMATH_CALUDE_light_travel_distance_l3213_321343

/-- The distance light travels in one year in kilometers -/
def light_year_distance : ℝ := 9460800000000

/-- The number of years we're calculating for -/
def years : ℝ := 70

/-- The expected distance light travels in the given number of years -/
def expected_distance : ℝ := 6.62256 * (10 ^ 14)

/-- Theorem stating that the distance light travels in the given number of years
    is equal to the expected distance -/
theorem light_travel_distance : light_year_distance * years = expected_distance := by
  sorry

end NUMINAMATH_CALUDE_light_travel_distance_l3213_321343


namespace NUMINAMATH_CALUDE_sqrt_calculation_problems_l3213_321332

theorem sqrt_calculation_problems :
  (∃ (x : ℝ), x = Real.sqrt 18 - Real.sqrt 8 - Real.sqrt 2 ∧ x = 0) ∧
  (∃ (y : ℝ), y = 6 * Real.sqrt 2 * Real.sqrt 3 + 3 * Real.sqrt 30 / Real.sqrt 5 ∧ y = 9 * Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculation_problems_l3213_321332


namespace NUMINAMATH_CALUDE_max_distinct_pairs_l3213_321322

theorem max_distinct_pairs (n : ℕ) (h : n = 3010) :
  let S := Finset.range n
  ∃ (k : ℕ) (pairs : Finset (ℕ × ℕ)),
    k = 1201 ∧
    pairs.card = k ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ 3005) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) ∧
    (∀ (m : ℕ), m > k →
      ¬∃ (pairs' : Finset (ℕ × ℕ)),
        pairs'.card = m ∧
        (∀ (p : ℕ × ℕ), p ∈ pairs' → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) ∧
        (∀ (p q : ℕ × ℕ), p ∈ pairs' → q ∈ pairs' → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
        (∀ (p : ℕ × ℕ), p ∈ pairs' → p.1 + p.2 ≤ 3005) ∧
        (∀ (p q : ℕ × ℕ), p ∈ pairs' → q ∈ pairs' → p ≠ q → p.1 + p.2 ≠ q.1 + q.2)) :=
by
  sorry

end NUMINAMATH_CALUDE_max_distinct_pairs_l3213_321322


namespace NUMINAMATH_CALUDE_cable_car_travel_time_l3213_321375

/-- Represents the time in minutes to travel half the circular route -/
def travel_time : ℝ := 22.5

/-- Represents the number of cable cars on the circular route -/
def num_cars : ℕ := 80

/-- Represents the time interval in seconds between encounters with opposing cars -/
def encounter_interval : ℝ := 15

/-- Theorem stating that given the conditions, the travel time from A to B is 22.5 minutes -/
theorem cable_car_travel_time :
  ∀ (cars : ℕ) (interval : ℝ),
  cars = num_cars →
  interval = encounter_interval →
  travel_time = (cars : ℝ) * interval / (2 * 60) :=
by sorry

end NUMINAMATH_CALUDE_cable_car_travel_time_l3213_321375


namespace NUMINAMATH_CALUDE_pythagorean_triple_3_4_5_l3213_321362

theorem pythagorean_triple_3_4_5 :
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2 ∧ a = 3 ∧ b = 4 ∧ c = 5 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_3_4_5_l3213_321362


namespace NUMINAMATH_CALUDE_activities_alignment_period_l3213_321361

def activity_frequencies : List Nat := [6, 4, 16, 12, 8, 13, 17]

theorem activities_alignment_period :
  Nat.lcm (List.foldl Nat.lcm 1 activity_frequencies) = 10608 := by
  sorry

end NUMINAMATH_CALUDE_activities_alignment_period_l3213_321361


namespace NUMINAMATH_CALUDE_vector_magnitude_condition_l3213_321342

theorem vector_magnitude_condition (n : Type*) [NormedAddCommGroup n] :
  ∃ (a b : n),
    (‖a‖ = ‖b‖ ∧ ‖a + b‖ ≠ ‖a - b‖) ∧
    (‖a‖ ≠ ‖b‖ ∧ ‖a + b‖ = ‖a - b‖) :=
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_condition_l3213_321342


namespace NUMINAMATH_CALUDE_root_product_equals_sixteen_l3213_321376

theorem root_product_equals_sixteen :
  (16 : ℝ)^(1/4) * (64 : ℝ)^(1/3) * (4 : ℝ)^(1/2) = 16 := by sorry

end NUMINAMATH_CALUDE_root_product_equals_sixteen_l3213_321376


namespace NUMINAMATH_CALUDE_fraction_representation_l3213_321384

theorem fraction_representation (n : ℕ) : ∃ x y : ℕ, n = x^2 / y^3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_representation_l3213_321384


namespace NUMINAMATH_CALUDE_eight_chairs_subsets_l3213_321313

/-- The number of subsets of n chairs arranged in a circle that contain at least k adjacent chairs. -/
def subsets_with_adjacent_chairs (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The main theorem: For 8 chairs arranged in a circle, there are 33 subsets containing at least 4 adjacent chairs. -/
theorem eight_chairs_subsets : subsets_with_adjacent_chairs 8 4 = 33 := by sorry

end NUMINAMATH_CALUDE_eight_chairs_subsets_l3213_321313


namespace NUMINAMATH_CALUDE_extended_quad_ratio_gt_one_ratio_always_gt_one_l3213_321391

/-- Represents a convex quadrilateral ABCD with an extended construction --/
structure ExtendedQuadrilateral where
  /-- The sum of all internal angles of the quadrilateral --/
  internal_sum : ℝ
  /-- The sum of angles BAD and ABC --/
  partial_sum : ℝ
  /-- Assumption that the quadrilateral is convex --/
  convex : 0 < partial_sum ∧ partial_sum < internal_sum

/-- The ratio of external angle sum to partial internal angle sum is greater than 1 --/
theorem extended_quad_ratio_gt_one (q : ExtendedQuadrilateral) : 
  q.internal_sum / q.partial_sum > 1 := by
  sorry

/-- Main theorem: For any convex quadrilateral with the given construction, 
    the ratio r is always greater than 1 --/
theorem ratio_always_gt_one : 
  ∀ q : ExtendedQuadrilateral, q.internal_sum / q.partial_sum > 1 := by
  sorry

end NUMINAMATH_CALUDE_extended_quad_ratio_gt_one_ratio_always_gt_one_l3213_321391


namespace NUMINAMATH_CALUDE_objective_function_minimum_range_l3213_321340

-- Define the objective function
def objective_function (k x y : ℝ) : ℝ := k * x + 2 * y

-- Define the constraints
def constraint1 (x y : ℝ) : Prop := 2 * x - y ≤ 1
def constraint2 (x y : ℝ) : Prop := x + y ≥ 2
def constraint3 (x y : ℝ) : Prop := y - x ≤ 2

-- Define the feasible region
def feasible_region (x y : ℝ) : Prop :=
  constraint1 x y ∧ constraint2 x y ∧ constraint3 x y

-- Define the minimum point
def is_minimum_point (k : ℝ) (x y : ℝ) : Prop :=
  feasible_region x y ∧
  ∀ x' y', feasible_region x' y' →
    objective_function k x y ≤ objective_function k x' y'

-- Theorem statement
theorem objective_function_minimum_range :
  ∀ k : ℝ, (is_minimum_point k 1 1 ∧
    ∀ x y, x ≠ 1 ∨ y ≠ 1 → ¬(is_minimum_point k x y)) →
  -4 < k ∧ k < 2 := by
  sorry

end NUMINAMATH_CALUDE_objective_function_minimum_range_l3213_321340


namespace NUMINAMATH_CALUDE_sin_b_in_arithmetic_sequence_triangle_l3213_321305

/-- In a triangle ABC where the interior angles form an arithmetic sequence, sin B = √3/2 -/
theorem sin_b_in_arithmetic_sequence_triangle (A B C : Real) : 
  A + B + C = Real.pi →  -- Sum of angles in radians
  A + C = 2 * B →        -- Arithmetic sequence property
  Real.sin B = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_b_in_arithmetic_sequence_triangle_l3213_321305


namespace NUMINAMATH_CALUDE_perfume_dilution_l3213_321381

/-- Proves that adding 7.2 ounces of water to 12 ounces of a 40% alcohol solution
    results in a 25% alcohol solution -/
theorem perfume_dilution (initial_volume : ℝ) (initial_concentration : ℝ)
                         (target_concentration : ℝ) (water_added : ℝ) :
  initial_volume = 12 →
  initial_concentration = 0.4 →
  target_concentration = 0.25 →
  water_added = 7.2 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = target_concentration :=
by
  sorry

#check perfume_dilution

end NUMINAMATH_CALUDE_perfume_dilution_l3213_321381


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3213_321326

/-- The repeating decimal 0.565656... -/
def repeating_decimal : ℚ := 0.56565656

/-- The fraction 56/99 -/
def target_fraction : ℚ := 56 / 99

/-- Theorem stating that the repeating decimal 0.565656... is equal to 56/99 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = target_fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3213_321326


namespace NUMINAMATH_CALUDE_intersection_product_sum_l3213_321382

/-- Given a line and a circle in R², prove that the sum of the products of the x-coordinate of one
    intersection point and the y-coordinate of the other equals 16. -/
theorem intersection_product_sum (x₁ y₁ x₂ y₂ : ℝ) : 
  (x₁ + y₁ = 5) →
  (x₂ + y₂ = 5) →
  (x₁^2 + y₁^2 = 16) →
  (x₂^2 + y₂^2 = 16) →
  x₁ * y₂ + x₂ * y₁ = 16 := by
  sorry

end NUMINAMATH_CALUDE_intersection_product_sum_l3213_321382


namespace NUMINAMATH_CALUDE_sequence_correct_l3213_321378

def sequence_term (n : ℕ) : ℤ := (-1)^n * (2^n - 1)

theorem sequence_correct : 
  sequence_term 1 = -1 ∧ 
  sequence_term 2 = 3 ∧ 
  sequence_term 3 = -7 ∧ 
  sequence_term 4 = 15 := by
sorry

end NUMINAMATH_CALUDE_sequence_correct_l3213_321378


namespace NUMINAMATH_CALUDE_exists_phi_sin_2x_plus_phi_even_l3213_321351

theorem exists_phi_sin_2x_plus_phi_even : ∃ φ : ℝ, ∀ x : ℝ, 
  Real.sin (2 * x + φ) = Real.sin (2 * (-x) + φ) := by
  sorry

end NUMINAMATH_CALUDE_exists_phi_sin_2x_plus_phi_even_l3213_321351


namespace NUMINAMATH_CALUDE_expression_evaluation_l3213_321347

theorem expression_evaluation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hsum : 3 * x + y / 3 ≠ 0) :
  (3 * x + y / 3)⁻¹ * ((3 * x)⁻¹ + (y / 3)⁻¹) = (x * y)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3213_321347


namespace NUMINAMATH_CALUDE_election_win_percentage_l3213_321306

theorem election_win_percentage 
  (total_votes : ℕ) 
  (geoff_percentage : ℚ) 
  (additional_votes_needed : ℕ) 
  (h1 : total_votes = 6000)
  (h2 : geoff_percentage = 1/200)  -- 0.5% as a rational number
  (h3 : additional_votes_needed = 3000) :
  (((geoff_percentage * total_votes + additional_votes_needed) / total_votes) : ℚ) = 101/200 := by
sorry

end NUMINAMATH_CALUDE_election_win_percentage_l3213_321306


namespace NUMINAMATH_CALUDE_f_is_even_and_increasing_on_negative_l3213_321331

def f (x : ℝ) := -x^2

theorem f_is_even_and_increasing_on_negative : 
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x y : ℝ, x < y ∧ y ≤ 0 → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_and_increasing_on_negative_l3213_321331


namespace NUMINAMATH_CALUDE_towels_used_theorem_l3213_321314

/-- Calculates the number of towels used in a gym based on guest distribution and staff usage. -/
def towels_used (first_hour_guests : ℕ) : ℕ :=
  let second_hour_guests := first_hour_guests + first_hour_guests / 5
  let third_hour_guests := second_hour_guests + second_hour_guests / 4
  let fourth_hour_guests := third_hour_guests + third_hour_guests / 3
  let fifth_hour_guests := fourth_hour_guests - fourth_hour_guests * 3 / 20
  let sixth_hour_guests := fifth_hour_guests
  let seventh_hour_guests := sixth_hour_guests - sixth_hour_guests * 3 / 10
  let eighth_hour_guests := seventh_hour_guests / 2
  let total_guests := first_hour_guests + second_hour_guests + third_hour_guests + 
                      fourth_hour_guests + fifth_hour_guests + sixth_hour_guests + 
                      seventh_hour_guests + eighth_hour_guests
  let three_towel_guests := total_guests / 10
  let two_towel_guests := total_guests * 6 / 10
  let one_towel_guests := total_guests * 3 / 10
  let guest_towels := three_towel_guests * 3 + two_towel_guests * 2 + one_towel_guests
  guest_towels + 20

/-- The theorem stating that given 40 guests in the first hour, the total towels used is 807. -/
theorem towels_used_theorem : towels_used 40 = 807 := by
  sorry

#eval towels_used 40

end NUMINAMATH_CALUDE_towels_used_theorem_l3213_321314


namespace NUMINAMATH_CALUDE_sqrt_three_irrational_l3213_321388

theorem sqrt_three_irrational :
  (∃ (q : ℚ), (1 : ℝ) / 3 = ↑q) ∧ 
  (∃ (q : ℚ), (3.14 : ℝ) = ↑q) ∧ 
  (∃ (q : ℚ), Real.sqrt 9 = ↑q) →
  ¬ ∃ (q : ℚ), Real.sqrt 3 = ↑q :=
by sorry

end NUMINAMATH_CALUDE_sqrt_three_irrational_l3213_321388


namespace NUMINAMATH_CALUDE_sum_of_roots_squared_equation_l3213_321334

theorem sum_of_roots_squared_equation (x : ℝ) :
  (∃ a b : ℝ, (a - 3)^2 = 16 ∧ (b - 3)^2 = 16 ∧ a + b = 6) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_squared_equation_l3213_321334


namespace NUMINAMATH_CALUDE_cube_opposite_face_l3213_321367

/-- Represents a face of the cube -/
inductive Face : Type
| A | B | C | D | E | F

/-- Represents the adjacency relation between faces -/
def adjacent : Face → Face → Prop := sorry

/-- Represents the opposite relation between faces -/
def opposite : Face → Face → Prop := sorry

/-- The theorem stating that F is opposite to A in the given cube configuration -/
theorem cube_opposite_face :
  (adjacent Face.A Face.B) →
  (adjacent Face.A Face.C) →
  (adjacent Face.B Face.D) →
  (opposite Face.A Face.F) := by sorry

end NUMINAMATH_CALUDE_cube_opposite_face_l3213_321367


namespace NUMINAMATH_CALUDE_optimal_sampling_methods_l3213_321368

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a box of ping-pong balls -/
structure Box where
  color : String
  count : Nat

/-- Represents the ping-pong ball problem -/
structure PingPongProblem where
  totalBalls : Nat
  boxes : List Box
  sampleSize : Nat

/-- Represents the student selection problem -/
structure StudentProblem where
  totalStudents : Nat
  selectCount : Nat

/-- Determines the optimal sampling method for a given problem -/
def optimalSamplingMethod (p : PingPongProblem ⊕ StudentProblem) : SamplingMethod :=
  match p with
  | .inl _ => SamplingMethod.Stratified
  | .inr _ => SamplingMethod.SimpleRandom

/-- The main theorem stating the optimal sampling methods for both problems -/
theorem optimal_sampling_methods 
  (pingPong : PingPongProblem)
  (student : StudentProblem)
  (h1 : pingPong.totalBalls = 1000)
  (h2 : pingPong.boxes = [
    { color := "red", count := 500 },
    { color := "blue", count := 200 },
    { color := "yellow", count := 300 }
  ])
  (h3 : pingPong.sampleSize = 100)
  (h4 : student.totalStudents = 20)
  (h5 : student.selectCount = 3) :
  (optimalSamplingMethod (.inl pingPong) = SamplingMethod.Stratified) ∧
  (optimalSamplingMethod (.inr student) = SamplingMethod.SimpleRandom) :=
sorry


end NUMINAMATH_CALUDE_optimal_sampling_methods_l3213_321368


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l3213_321385

theorem min_value_of_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 9/y ≥ 1/a + 9/b) →
  1/a + 9/b = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l3213_321385


namespace NUMINAMATH_CALUDE_max_cube_volume_in_pyramid_l3213_321324

/-- The maximum volume of a cube inscribed in a pyramid --/
theorem max_cube_volume_in_pyramid (base_side : ℝ) (pyramid_height : ℝ) : 
  base_side = 2 →
  pyramid_height = 3 →
  ∃ (cube_volume : ℝ), 
    cube_volume = (81 * Real.sqrt 6) / 32 ∧ 
    ∀ (other_volume : ℝ), 
      (∃ (cube_side : ℝ), 
        cube_side > 0 ∧
        other_volume = cube_side ^ 3 ∧
        cube_side * Real.sqrt 2 ≤ 3 * Real.sqrt 3 / 2) →
      other_volume ≤ cube_volume :=
by sorry

end NUMINAMATH_CALUDE_max_cube_volume_in_pyramid_l3213_321324


namespace NUMINAMATH_CALUDE_kids_difference_l3213_321355

/-- The number of kids Julia played with on each day of the week. -/
structure WeeklyKids where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ
  sunday : ℕ

/-- Theorem stating the difference in the number of kids played with on specific days. -/
theorem kids_difference (w : WeeklyKids)
    (h1 : w.monday = 6)
    (h2 : w.tuesday = 17)
    (h3 : w.wednesday = 4)
    (h4 : w.thursday = 12)
    (h5 : w.friday = 10)
    (h6 : w.saturday = 15)
    (h7 : w.sunday = 9) :
    (w.tuesday + w.thursday) - (w.monday + w.wednesday + w.sunday) = 10 := by
  sorry


end NUMINAMATH_CALUDE_kids_difference_l3213_321355


namespace NUMINAMATH_CALUDE_triangle_area_l3213_321386

theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * Real.cos (B - C) + a * Real.cos A = 2 * Real.sqrt 3 * c * Real.sin B * Real.cos A →
  b^2 + c^2 - a^2 = 2 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l3213_321386


namespace NUMINAMATH_CALUDE_coyote_speed_calculation_l3213_321321

/-- The speed of the coyote in miles per hour -/
def coyote_speed : ℝ := 15

/-- The time elapsed since the coyote left its prints, in hours -/
def time_elapsed : ℝ := 1

/-- Darrel's speed on his motorbike in miles per hour -/
def darrel_speed : ℝ := 30

/-- The time it takes Darrel to catch up to the coyote, in hours -/
def catch_up_time : ℝ := 1

theorem coyote_speed_calculation :
  coyote_speed * time_elapsed + coyote_speed * catch_up_time = darrel_speed * catch_up_time := by
  sorry

#check coyote_speed_calculation

end NUMINAMATH_CALUDE_coyote_speed_calculation_l3213_321321


namespace NUMINAMATH_CALUDE_infinite_solutions_imply_a_equals_five_l3213_321370

/-- If the equation 3(5 + ay) = 15y + 15 has infinitely many solutions for y, then a = 5 -/
theorem infinite_solutions_imply_a_equals_five (a : ℝ) : 
  (∀ y : ℝ, 3 * (5 + a * y) = 15 * y + 15) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_imply_a_equals_five_l3213_321370


namespace NUMINAMATH_CALUDE_star_equality_implies_y_value_l3213_321345

/-- Binary operation ★ on ordered pairs of integers -/
def star : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) := fun (a, b) (c, d) ↦ (a - c, b + d)

/-- Theorem stating that if (5, 0) ★ (2, -2) = (x, y) ★ (0, 3), then y = -5 -/
theorem star_equality_implies_y_value (x y : ℤ) :
  star (5, 0) (2, -2) = star (x, y) (0, 3) → y = -5 := by
  sorry

end NUMINAMATH_CALUDE_star_equality_implies_y_value_l3213_321345


namespace NUMINAMATH_CALUDE_puppy_count_l3213_321397

theorem puppy_count (total_ears : ℕ) (ears_per_puppy : ℕ) (h1 : total_ears = 210) (h2 : ears_per_puppy = 2) :
  total_ears / ears_per_puppy = 105 :=
by sorry

end NUMINAMATH_CALUDE_puppy_count_l3213_321397


namespace NUMINAMATH_CALUDE_geometric_sequence_a3_l3213_321338

/-- A geometric sequence with specific properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a3 (a : ℕ → ℝ) :
  GeometricSequence a →
  a 1 + a 5 = 82 →
  a 2 * a 4 = 81 →
  a 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a3_l3213_321338


namespace NUMINAMATH_CALUDE_reciprocal_of_two_l3213_321304

theorem reciprocal_of_two :
  ∃ x : ℚ, x * 2 = 1 ∧ x = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_two_l3213_321304


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3213_321329

/-- Given a geometric sequence with positive terms and common ratio q where q^2 = 4,
    prove that (a_3 + a_4) / (a_4 + a_5) = 1/2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- Common ratio is q
  q^2 = 4 →  -- Given condition
  (a 3 + a 4) / (a 4 + a 5) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3213_321329


namespace NUMINAMATH_CALUDE_complement_of_union_l3213_321310

theorem complement_of_union (U A B : Set ℕ) : 
  U = {x : ℕ | x > 0 ∧ x < 6} →
  A = {1, 3} →
  B = {3, 5} →
  (U \ (A ∪ B)) = {2, 4} := by
sorry

end NUMINAMATH_CALUDE_complement_of_union_l3213_321310


namespace NUMINAMATH_CALUDE_golden_ratio_solution_l3213_321358

theorem golden_ratio_solution (x : ℝ) :
  x > 0 ∧ x = Real.sqrt (x - 1 / x) + Real.sqrt (1 - 1 / x) ↔ x = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_solution_l3213_321358


namespace NUMINAMATH_CALUDE_keiko_text_messages_l3213_321390

/-- The number of text messages Keiko sent in the first week -/
def first_week : ℕ := 111

/-- The number of text messages Keiko sent in the second week -/
def second_week : ℕ := 2 * first_week - 50

/-- The number of text messages Keiko sent in the third week -/
def third_week : ℕ := second_week + (second_week / 4)

/-- The total number of text messages Keiko sent over three weeks -/
def total_messages : ℕ := first_week + second_week + third_week

theorem keiko_text_messages : total_messages = 498 := by
  sorry

end NUMINAMATH_CALUDE_keiko_text_messages_l3213_321390


namespace NUMINAMATH_CALUDE_calculation_proof_l3213_321311

theorem calculation_proof :
  let expr1 := -1^4 - (1/6) * (2 - (-3)^2) / (-7)
  let expr2 := (1 + 1/2 - 5/8 + 7/12) / (-1/24) - 8 * (-1/2)^3
  expr1 = -7/6 ∧ expr2 = -34 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3213_321311


namespace NUMINAMATH_CALUDE_average_carnations_value_l3213_321317

/-- The average number of carnations in Trevor's bouquets -/
def average_carnations : ℚ :=
  let bouquets : List ℕ := [9, 23, 13, 36, 28, 45]
  (bouquets.sum : ℚ) / bouquets.length

/-- Proof that the average number of carnations is 25.67 -/
theorem average_carnations_value :
  average_carnations = 25.67 := by
  sorry

end NUMINAMATH_CALUDE_average_carnations_value_l3213_321317


namespace NUMINAMATH_CALUDE_commodity_consumption_increase_l3213_321380

theorem commodity_consumption_increase
  (original_tax : ℝ)
  (original_consumption : ℝ)
  (h_tax_positive : original_tax > 0)
  (h_consumption_positive : original_consumption > 0)
  (h_tax_reduction : ℝ)
  (h_revenue_decrease : ℝ)
  (h_consumption_increase : ℝ)
  (h_tax_reduction_eq : h_tax_reduction = 0.20)
  (h_revenue_decrease_eq : h_revenue_decrease = 0.16)
  (h_new_tax : ℝ := original_tax * (1 - h_tax_reduction))
  (h_new_consumption : ℝ := original_consumption * (1 + h_consumption_increase))
  (h_new_revenue : ℝ := h_new_tax * h_new_consumption)
  (h_original_revenue : ℝ := original_tax * original_consumption)
  (h_revenue_equation : h_new_revenue = h_original_revenue * (1 - h_revenue_decrease)) :
  h_consumption_increase = 0.05 := by sorry

end NUMINAMATH_CALUDE_commodity_consumption_increase_l3213_321380


namespace NUMINAMATH_CALUDE_urn_problem_l3213_321377

theorem urn_problem (M : ℕ) : M = 111 ↔ 
  (5 : ℝ) / 12 * 20 / (20 + M) + 7 / 12 * M / (20 + M) = 0.62 := by
  sorry

end NUMINAMATH_CALUDE_urn_problem_l3213_321377


namespace NUMINAMATH_CALUDE_expand_equals_difference_of_squares_l3213_321302

theorem expand_equals_difference_of_squares (x y : ℝ) :
  (-x + 2*y) * (-x - 2*y) = x^2 - 4*y^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_equals_difference_of_squares_l3213_321302


namespace NUMINAMATH_CALUDE_first_digit_base_16_l3213_321325

def base_4_representation : List ℕ := [2, 0, 3, 1, 3, 3, 2, 0, 1, 3, 2, 2, 2, 0, 3, 1, 2, 0, 3, 1]

def y : ℕ := (List.foldl (λ acc d => acc * 4 + d) 0 base_4_representation)

theorem first_digit_base_16 : ∃ (rest : ℕ), y = 5 * 16^rest + (y % 16^rest) ∧ y < 6 * 16^rest :=
sorry

end NUMINAMATH_CALUDE_first_digit_base_16_l3213_321325


namespace NUMINAMATH_CALUDE_probability_different_topics_l3213_321318

/-- The number of topics in the essay competition -/
def num_topics : ℕ := 6

/-- The probability that two students select different topics -/
def prob_different_topics : ℚ := 5/6

/-- Theorem stating the probability of two students selecting different topics -/
theorem probability_different_topics :
  (num_topics : ℚ) * (num_topics - 1) / (num_topics * num_topics) = prob_different_topics :=
sorry

end NUMINAMATH_CALUDE_probability_different_topics_l3213_321318


namespace NUMINAMATH_CALUDE_quadratic_monotonic_condition_l3213_321307

-- Define the quadratic function
def f (t : ℝ) (x : ℝ) : ℝ := x^2 - 2*t*x + 1

-- Define monotonicity in an interval
def monotonic_in_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → (f x < f y ∨ f y < f x)

-- Theorem statement
theorem quadratic_monotonic_condition (t : ℝ) :
  monotonic_in_interval (f t) 1 3 → t ≤ 1 ∨ t ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_monotonic_condition_l3213_321307


namespace NUMINAMATH_CALUDE_oatmeal_raisin_cookies_l3213_321387

/-- Given a class of students and cookie preferences, calculate the number of oatmeal raisin cookies to be made. -/
theorem oatmeal_raisin_cookies (total_students : ℕ) (cookies_per_student : ℕ) (oatmeal_raisin_percentage : ℚ) : 
  total_students = 40 → 
  cookies_per_student = 2 → 
  oatmeal_raisin_percentage = 1/10 →
  (total_students : ℚ) * oatmeal_raisin_percentage * cookies_per_student = 8 := by
  sorry

#check oatmeal_raisin_cookies

end NUMINAMATH_CALUDE_oatmeal_raisin_cookies_l3213_321387


namespace NUMINAMATH_CALUDE_mrs_taylor_purchase_cost_l3213_321336

/-- Calculates the total cost of smart televisions and soundbars with discounts -/
def total_cost (tv_count : ℕ) (tv_price : ℚ) (tv_discount : ℚ)
                (soundbar_count : ℕ) (soundbar_price : ℚ) (soundbar_discount : ℚ) : ℚ :=
  let tv_total := tv_count * tv_price * (1 - tv_discount)
  let soundbar_total := soundbar_count * soundbar_price * (1 - soundbar_discount)
  tv_total + soundbar_total

/-- Theorem stating that Mrs. Taylor's purchase totals $2085 -/
theorem mrs_taylor_purchase_cost :
  total_cost 2 750 0.15 3 300 0.10 = 2085 := by
  sorry

end NUMINAMATH_CALUDE_mrs_taylor_purchase_cost_l3213_321336


namespace NUMINAMATH_CALUDE_egg_groups_l3213_321396

/-- Given 16 eggs split into groups of 2, prove that the number of groups is 8 -/
theorem egg_groups (total_eggs : ℕ) (eggs_per_group : ℕ) (num_groups : ℕ) : 
  total_eggs = 16 → eggs_per_group = 2 → num_groups = total_eggs / eggs_per_group → num_groups = 8 := by
  sorry

end NUMINAMATH_CALUDE_egg_groups_l3213_321396


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_factorization_3_l3213_321394

-- Define variables
variable (a b m x y : ℝ)

-- Theorem 1
theorem factorization_1 : 3*m - 3*y + a*m - a*y = (m - y) * (3 + a) := by sorry

-- Theorem 2
theorem factorization_2 : a^2*x + a^2*y + b^2*x + b^2*y = (x + y) * (a^2 + b^2) := by sorry

-- Theorem 3
theorem factorization_3 : a^2 + 2*a*b + b^2 - 1 = (a + b + 1) * (a + b - 1) := by sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_factorization_3_l3213_321394


namespace NUMINAMATH_CALUDE_k_value_l3213_321344

theorem k_value (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - 2*k)*(x + 3*k) = x^3 + 3*k*(x^2 - x - 4)) →
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_k_value_l3213_321344


namespace NUMINAMATH_CALUDE_min_yellow_surface_fraction_l3213_321366

/-- Represents a 4x4x4 cube constructed from smaller 1-inch cubes -/
structure LargeCube where
  small_cubes : Fin 64 → Color
  blue_count : Nat
  yellow_count : Nat
  h_blue_count : blue_count = 32
  h_yellow_count : yellow_count = 32
  h_total_count : blue_count + yellow_count = 64

inductive Color
  | Blue
  | Yellow

/-- Calculates the surface area of the large cube -/
def surface_area : Nat := 6 * 4 * 4

/-- Calculates the minimum yellow surface area possible -/
def min_yellow_surface_area (cube : LargeCube) : Nat :=
  sorry

/-- Theorem stating that the minimum fraction of yellow surface area is 1/4 -/
theorem min_yellow_surface_fraction (cube : LargeCube) :
  (min_yellow_surface_area cube : ℚ) / surface_area = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_yellow_surface_fraction_l3213_321366


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l3213_321365

theorem fraction_product_simplification :
  (18 : ℚ) / 17 * 13 / 24 * 68 / 39 = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l3213_321365


namespace NUMINAMATH_CALUDE_fourth_intersection_point_l3213_321352

def curve (x y : ℝ) : Prop := x * y = 2

theorem fourth_intersection_point (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) 
  (h₁ : curve x₁ y₁) (h₂ : curve x₂ y₂) (h₃ : curve x₃ y₃) (h₄ : curve x₄ y₄)
  (p₁ : x₁ = 4 ∧ y₁ = 1/2) (p₂ : x₂ = -2 ∧ y₂ = -1) (p₃ : x₃ = 1/4 ∧ y₃ = 8)
  (distinct : x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) :
  x₄ = 1 ∧ y₄ = 2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_intersection_point_l3213_321352


namespace NUMINAMATH_CALUDE_line_l_equation_l3213_321341

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Returns true if the point (x, y) lies on the given line -/
def Line.containsPoint (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.yIntercept

/-- The reference line y = x + 1 -/
def referenceLine : Line :=
  { slope := 1, yIntercept := 1 }

/-- The line we're trying to prove -/
def lineL : Line :=
  { slope := 2, yIntercept := -3 }

theorem line_l_equation :
  (lineL.slope = 2 * referenceLine.slope) ∧
  (lineL.containsPoint 3 3) →
  ∀ x y : ℝ, lineL.containsPoint x y ↔ y = 2*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_line_l_equation_l3213_321341


namespace NUMINAMATH_CALUDE_golden_ratio_from_logarithms_l3213_321357

theorem golden_ratio_from_logarithms (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.log a / Real.log 4 = Real.log b / Real.log 18) ∧ 
  (Real.log b / Real.log 18 = Real.log (a + b) / Real.log 32) →
  b / a = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_from_logarithms_l3213_321357


namespace NUMINAMATH_CALUDE_work_completion_time_l3213_321303

/-- Given that person B can complete 2/3 of a job in 12 days, 
    prove that B can complete the entire job in 18 days. -/
theorem work_completion_time (B_partial_time : ℕ) (B_partial_work : ℚ) 
  (h1 : B_partial_time = 12) 
  (h2 : B_partial_work = 2/3) : 
  ∃ (B_full_time : ℕ), B_full_time = 18 ∧ 
  B_partial_work / B_partial_time = 1 / B_full_time :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3213_321303


namespace NUMINAMATH_CALUDE_range_of_g_l3213_321349

theorem range_of_g (x : ℝ) : -1 ≤ Real.sin x ^ 3 + Real.cos x ^ 2 ∧ Real.sin x ^ 3 + Real.cos x ^ 2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_g_l3213_321349


namespace NUMINAMATH_CALUDE_factorial_of_factorial_divided_by_factorial_l3213_321339

theorem factorial_of_factorial_divided_by_factorial :
  (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end NUMINAMATH_CALUDE_factorial_of_factorial_divided_by_factorial_l3213_321339


namespace NUMINAMATH_CALUDE_circle_ratio_after_increase_l3213_321323

theorem circle_ratio_after_increase (r : ℝ) : 
  let new_radius := r + 2
  let new_diameter := 2 * new_radius
  let new_circumference := 2 * Real.pi * new_radius
  new_circumference / new_diameter = Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_ratio_after_increase_l3213_321323


namespace NUMINAMATH_CALUDE_special_polygon_area_l3213_321308

/-- A polygon with 24 congruent sides, where each side is perpendicular to its adjacent sides -/
structure SpecialPolygon where
  sides : ℕ
  side_length : ℝ
  perimeter : ℝ
  sides_eq : sides = 24
  perimeter_eq : perimeter = 48
  perimeter_formula : perimeter = sides * side_length

/-- The area of the special polygon is 64 -/
theorem special_polygon_area (p : SpecialPolygon) : 16 * p.side_length ^ 2 = 64 := by
  sorry

#check special_polygon_area

end NUMINAMATH_CALUDE_special_polygon_area_l3213_321308


namespace NUMINAMATH_CALUDE_equation_solution_range_l3213_321348

-- Define the equation
def equation (x a : ℝ) : Prop := Real.sqrt (x^2 - 1) = a*x - 2

-- Define the condition of having exactly one solution
def has_unique_solution (a : ℝ) : Prop :=
  ∃! x, equation x a

-- Define the range of a
def a_range (a : ℝ) : Prop :=
  (a ∈ Set.Icc (-Real.sqrt 5) (-1) ∪ Set.Ioc 1 (Real.sqrt 5))

-- Theorem statement
theorem equation_solution_range :
  ∀ a : ℝ, has_unique_solution a ↔ a_range a :=
sorry

end NUMINAMATH_CALUDE_equation_solution_range_l3213_321348


namespace NUMINAMATH_CALUDE_product_and_sum_inequality_l3213_321389

theorem product_and_sum_inequality (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h : 2/x + 8/y = 1) : 
  x * y ≥ 64 ∧ x + y ≥ 18 := by
sorry

end NUMINAMATH_CALUDE_product_and_sum_inequality_l3213_321389


namespace NUMINAMATH_CALUDE_harolds_car_payment_l3213_321312

def monthly_income : ℚ := 2500
def rent : ℚ := 700
def groceries : ℚ := 50
def remaining_money : ℚ := 650

def car_payment (x : ℚ) : Prop :=
  let utilities := x / 2
  let total_expenses := rent + x + utilities + groceries
  let retirement_contribution := (monthly_income - total_expenses) / 2
  monthly_income - total_expenses - retirement_contribution = remaining_money

theorem harolds_car_payment :
  ∃ (x : ℚ), car_payment x ∧ x = 300 :=
sorry

end NUMINAMATH_CALUDE_harolds_car_payment_l3213_321312


namespace NUMINAMATH_CALUDE_x_coordinate_at_y_3_l3213_321364

-- Define the line
def line (x y : ℝ) : Prop :=
  y + 3 = (1/2) * (x + 2)

-- Define the point (-2, -3) on the line
axiom point_on_line : line (-2) (-3)

-- Define the x-intercept
axiom x_intercept : line 4 0

-- Theorem to prove
theorem x_coordinate_at_y_3 :
  ∃ (x : ℝ), line x 3 ∧ x = 10 :=
sorry

end NUMINAMATH_CALUDE_x_coordinate_at_y_3_l3213_321364


namespace NUMINAMATH_CALUDE_exponent_division_l3213_321363

theorem exponent_division (x : ℝ) (h : x ≠ 0) : x^2 / x^5 = 1 / x^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l3213_321363


namespace NUMINAMATH_CALUDE_positive_integer_solutions_independent_of_m_compare_M_N_l3213_321330

def oplus (a b : ℝ) : ℝ := a * (a - b)

theorem positive_integer_solutions :
  ∀ a b : ℕ+, (oplus 3 a = b) ↔ ((a = 2 ∧ b = 3) ∨ (a = 1 ∧ b = 6)) :=
sorry

theorem independent_of_m (a b m : ℝ) :
  (oplus 2 a = 5*b - 2*m ∧ oplus 3 b = 5*a + m) → 12*a + 11*b = 22 :=
sorry

theorem compare_M_N (a b : ℝ) (h : a > 1) :
  oplus (a*b) b ≥ oplus b (a*b) :=
sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_independent_of_m_compare_M_N_l3213_321330


namespace NUMINAMATH_CALUDE_division_remainder_l3213_321372

theorem division_remainder : ∃ (A : ℕ), 17 = 6 * 2 + A ∧ A < 6 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l3213_321372


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3213_321328

/-- The set of points satisfying the inequality -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let x := p.1; let y := p.2;
    (abs x ≤ 1 ∧ abs y ≤ 1 ∧ x * y ≤ 0) ∨
    (x^2 + y^2 ≤ 1 ∧ x * y > 0)}

/-- The main theorem -/
theorem inequality_equivalence (x y : ℝ) :
  Real.sqrt (1 - x^2) * Real.sqrt (1 - y^2) ≥ x * y ↔ (x, y) ∈ S :=
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3213_321328


namespace NUMINAMATH_CALUDE_count_flippy_divisible_by_25_is_24_l3213_321360

/-- A flippy number alternates between two distinct digits. -/
def is_flippy (n : ℕ) : Prop := sorry

/-- Checks if a number is six digits long. -/
def is_six_digit (n : ℕ) : Prop := sorry

/-- Counts the number of six-digit flippy numbers divisible by 25. -/
def count_flippy_divisible_by_25 : ℕ := sorry

/-- Theorem stating that the count of six-digit flippy numbers divisible by 25 is 24. -/
theorem count_flippy_divisible_by_25_is_24 : count_flippy_divisible_by_25 = 24 := by sorry

end NUMINAMATH_CALUDE_count_flippy_divisible_by_25_is_24_l3213_321360


namespace NUMINAMATH_CALUDE_exists_function_with_properties_l3213_321354

theorem exists_function_with_properties : ∃ f : ℝ → ℝ, 
  (∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1) ∧ 
  (∀ x : ℝ, f x = f (-x)) := by
  sorry

end NUMINAMATH_CALUDE_exists_function_with_properties_l3213_321354


namespace NUMINAMATH_CALUDE_handshake_arrangement_count_l3213_321315

/-- Represents a handshake arrangement for a group of people --/
def HandshakeArrangement := Fin 12 → Finset (Fin 12)

/-- A valid handshake arrangement satisfies the problem conditions --/
def is_valid_arrangement (h : HandshakeArrangement) : Prop :=
  ∀ i : Fin 12, (h i).card = 3 ∧ ∀ j ∈ h i, i ∈ h j

/-- The number of distinct valid handshake arrangements --/
def num_arrangements : ℕ := sorry

theorem handshake_arrangement_count :
  num_arrangements = 13296960 ∧ num_arrangements % 1000 = 960 := by sorry

end NUMINAMATH_CALUDE_handshake_arrangement_count_l3213_321315


namespace NUMINAMATH_CALUDE_adams_lawn_mowing_l3213_321373

/-- Given that Adam earns 9 dollars per lawn, forgot to mow 8 lawns, and actually earned 36 dollars,
    prove that the total number of lawns he had to mow is 12. -/
theorem adams_lawn_mowing (dollars_per_lawn : ℕ) (forgotten_lawns : ℕ) (actual_earnings : ℕ) :
  dollars_per_lawn = 9 →
  forgotten_lawns = 8 →
  actual_earnings = 36 →
  (actual_earnings / dollars_per_lawn) + forgotten_lawns = 12 :=
by sorry

end NUMINAMATH_CALUDE_adams_lawn_mowing_l3213_321373


namespace NUMINAMATH_CALUDE_intersection_slope_range_l3213_321398

/-- Given two points P and Q in the Cartesian plane, and a linear function y = kx - 1
    that intersects the extension of line segment PQ (excluding Q),
    prove that the range of k is between 1/3 and 3/2 (exclusive). -/
theorem intersection_slope_range (P Q : ℝ × ℝ) (k : ℝ) : 
  P = (-1, 1) →
  Q = (2, 2) →
  (∃ x y : ℝ, y = k * x - 1 ∧ 
              (y - 1) / (x + 1) = (2 - 1) / (2 + 1) ∧
              (x, y) ≠ Q) →
  1/3 < k ∧ k < 3/2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_slope_range_l3213_321398


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l3213_321301

theorem perpendicular_lines_b_value (a : ℝ) :
  (∀ x y : ℝ, ax + 2*y + 1 = 0 ∧ 3*x + b*y + 5 = 0 → 
   ((-a/2) * (-3/b) = -1)) →
  b = -3 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l3213_321301


namespace NUMINAMATH_CALUDE_perfect_cube_values_l3213_321374

theorem perfect_cube_values (Z K : ℤ) (h1 : 600 < Z) (h2 : Z < 2000) (h3 : K > 1) (h4 : Z = K^3) :
  K = 9 ∨ K = 10 ∨ K = 11 ∨ K = 12 := by
  sorry

end NUMINAMATH_CALUDE_perfect_cube_values_l3213_321374


namespace NUMINAMATH_CALUDE_coin_probability_l3213_321309

/-- The probability of a specific sequence of coin flips -/
def sequence_probability (p : ℝ) : ℝ := p^2 * (1 - p)^3

/-- Theorem: If the probability of getting heads on the first 2 flips
    and tails on the last 3 flips is 1/32, then the probability of
    getting heads on a single flip is 1/2 -/
theorem coin_probability (p : ℝ) 
  (h1 : 0 ≤ p ∧ p ≤ 1) 
  (h2 : sequence_probability p = 1/32) : 
  p = 1/2 := by
  sorry

#check coin_probability

end NUMINAMATH_CALUDE_coin_probability_l3213_321309


namespace NUMINAMATH_CALUDE_sqrt_ratio_eq_three_implies_x_eq_eighteen_sevenths_l3213_321392

theorem sqrt_ratio_eq_three_implies_x_eq_eighteen_sevenths :
  ∀ x : ℚ, (Real.sqrt (8 * x) / Real.sqrt (4 * (x - 2)) = 3) → x = 18 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_ratio_eq_three_implies_x_eq_eighteen_sevenths_l3213_321392


namespace NUMINAMATH_CALUDE_vector_operation_proof_l3213_321350

theorem vector_operation_proof (a b : ℝ × ℝ) :
  a = (2, 1) → b = (2, -2) → 2 • a - b = (2, 4) := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_proof_l3213_321350


namespace NUMINAMATH_CALUDE_problem_statement_l3213_321383

theorem problem_statement (x y : ℝ) (hx : x = 12) (hy : y = 18) :
  (x - y) * ((x + y)^2) = -5400 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3213_321383


namespace NUMINAMATH_CALUDE_circle_area_ratio_l3213_321353

theorem circle_area_ratio (C D : Real) (hC : C > 0) (hD : D > 0)
  (h_arc : (60 / 360) * (2 * Real.pi * C) = (40 / 360) * (2 * Real.pi * D)) :
  (Real.pi * C^2) / (Real.pi * D^2) = 9/4 := by
sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l3213_321353


namespace NUMINAMATH_CALUDE_altitude_segment_theorem_l3213_321356

-- Define the triangle and its properties
structure AcuteTriangle where
  -- We don't need to explicitly define the vertices, just the properties we need
  altitude1_segment1 : ℝ
  altitude1_segment2 : ℝ
  altitude2_segment1 : ℝ
  altitude2_segment2 : ℝ
  acute : Bool
  h_acute : acute = true
  h_altitude1 : altitude1_segment1 = 6 ∧ altitude1_segment2 = 4
  h_altitude2 : altitude2_segment1 = 3

-- State the theorem
theorem altitude_segment_theorem (t : AcuteTriangle) : t.altitude2_segment2 = 31/3 := by
  sorry

end NUMINAMATH_CALUDE_altitude_segment_theorem_l3213_321356
