import Mathlib

namespace NUMINAMATH_CALUDE_volume_not_occupied_by_cones_l2642_264208

/-- The volume of a cylinder not occupied by two identical cones -/
theorem volume_not_occupied_by_cones (r h_cyl h_cone : ℝ) 
  (hr : r = 10)
  (h_cyl_height : h_cyl = 30)
  (h_cone_height : h_cone = 15) :
  π * r^2 * h_cyl - 2 * (1/3 * π * r^2 * h_cone) = 2000 * π := by
  sorry

end NUMINAMATH_CALUDE_volume_not_occupied_by_cones_l2642_264208


namespace NUMINAMATH_CALUDE_smallest_square_area_for_rectangles_l2642_264205

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the area of a square -/
def square_area (side : ℕ) : ℕ := side * side

/-- Checks if a square can contain two rectangles -/
def can_contain_rectangles (side : ℕ) (rect1 rect2 : Rectangle) : Prop :=
  (max rect1.width rect2.width ≤ side) ∧ (rect1.height + rect2.height ≤ side)

theorem smallest_square_area_for_rectangles :
  ∃ (side : ℕ),
    let rect1 : Rectangle := ⟨3, 4⟩
    let rect2 : Rectangle := ⟨4, 5⟩
    can_contain_rectangles side rect1 rect2 ∧
    square_area side = 49 ∧
    ∀ (smaller_side : ℕ), smaller_side < side →
      ¬ can_contain_rectangles smaller_side rect1 rect2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_area_for_rectangles_l2642_264205


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2642_264271

theorem perfect_square_condition (n : ℤ) : 
  (∃ k : ℤ, 9 + 8 * n = k^2) ↔ (∃ m : ℤ, n = (m - 1) * (m + 2) / 2) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2642_264271


namespace NUMINAMATH_CALUDE_restaurant_glasses_count_l2642_264247

/-- Represents the number of glasses in a restaurant with two box sizes --/
def total_glasses (small_box_count : ℕ) (large_box_count : ℕ) : ℕ :=
  12 * small_box_count + 16 * large_box_count

/-- Represents the average number of glasses per box --/
def average_glasses_per_box (small_box_count : ℕ) (large_box_count : ℕ) : ℚ :=
  (total_glasses small_box_count large_box_count : ℚ) / (small_box_count + large_box_count : ℚ)

theorem restaurant_glasses_count :
  ∃ (small_box_count large_box_count : ℕ),
    large_box_count = small_box_count + 16 ∧
    average_glasses_per_box small_box_count large_box_count = 15 ∧
    total_glasses small_box_count large_box_count = 480 :=
sorry

end NUMINAMATH_CALUDE_restaurant_glasses_count_l2642_264247


namespace NUMINAMATH_CALUDE_richard_twice_scott_age_l2642_264257

/-- The number of years until Richard is twice as old as Scott -/
def years_until_double : ℕ := 8

/-- David's current age -/
def david_age : ℕ := 14

/-- Richard's current age -/
def richard_age : ℕ := david_age + 6

/-- Scott's current age -/
def scott_age : ℕ := david_age - 8

theorem richard_twice_scott_age : 
  richard_age + years_until_double = 2 * (scott_age + years_until_double) :=
by sorry

end NUMINAMATH_CALUDE_richard_twice_scott_age_l2642_264257


namespace NUMINAMATH_CALUDE_specialArrangements_eq_480_l2642_264292

/-- The number of ways to arrange six distinct objects in a row,
    where two specific objects must be on the same side of a third specific object -/
def specialArrangements : ℕ :=
  let totalPositions := 6
  let fixedObjects := 3  -- A, B, and C
  let remainingObjects := 3  -- D, E, and F
  let positionsForC := totalPositions
  let waysToArrangeAB := 2  -- A and B can be swapped
  let waysToChooseSide := 2  -- A and B can be on either side of C
  let remainingArrangements := Nat.factorial remainingObjects

  positionsForC * waysToArrangeAB * waysToChooseSide * remainingArrangements

theorem specialArrangements_eq_480 : specialArrangements = 480 := by
  sorry

end NUMINAMATH_CALUDE_specialArrangements_eq_480_l2642_264292


namespace NUMINAMATH_CALUDE_max_handshakes_l2642_264218

/-- In a group of N people (N > 5), if at least two people have not shaken hands with everyone else,
    then the maximum number of people who could have shaken hands with every other person is N-2. -/
theorem max_handshakes (N : ℕ) (h1 : N > 5) :
  ∃ (max : ℕ), max = N - 2 ∧
  ∀ (shaken : Fin N → Fin N → Bool),
    (∃ (p1 p2 : Fin N), p1 ≠ p2 ∧
      (∃ (q : Fin N), shaken p1 q = false ∧ shaken p2 q = false)) →
    (∀ (p : Fin N), (∀ (q : Fin N), p ≠ q → shaken p q = true) → p.val < max) :=
sorry

end NUMINAMATH_CALUDE_max_handshakes_l2642_264218


namespace NUMINAMATH_CALUDE_largest_inscribed_triangle_area_l2642_264297

theorem largest_inscribed_triangle_area (r : ℝ) (h : r = 8) :
  let circle_area := π * r^2
  let diameter := 2 * r
  let max_triangle_area := (1/2) * diameter * r
  max_triangle_area = 64 := by sorry

end NUMINAMATH_CALUDE_largest_inscribed_triangle_area_l2642_264297


namespace NUMINAMATH_CALUDE_mans_speed_in_still_water_l2642_264258

/-- The speed of a man rowing a boat in still water, given downstream conditions. -/
theorem mans_speed_in_still_water (current_speed : ℝ) (distance : ℝ) (time : ℝ) :
  current_speed = 8 →
  distance = 40 →
  time = 4.499640028797696 →
  ∃ (speed_still_water : ℝ), 
    abs (speed_still_water - ((distance / time) - (current_speed * 1000 / 3600))) < 0.001 :=
by sorry

end NUMINAMATH_CALUDE_mans_speed_in_still_water_l2642_264258


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l2642_264274

/-- A line with equation y = kx + 2 and a parabola with equation y² = 8x have exactly one point in common if and only if k = 1 or k = 0 -/
theorem line_parabola_intersection (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = k * p.1 + 2 ∧ p.2^2 = 8 * p.1) ↔ (k = 1 ∨ k = 0) :=
sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l2642_264274


namespace NUMINAMATH_CALUDE_headcount_analysis_l2642_264279

/-- Student headcount data for spring terms -/
structure HeadcountData where
  y02_03 : ℕ
  y03_04 : ℕ
  y04_05 : ℕ
  y05_06 : ℕ

/-- Calculate average headcount -/
def average_headcount (data : HeadcountData) : ℚ :=
  (data.y02_03 + data.y03_04 + data.y04_05 + data.y05_06) / 4

/-- Calculate percentage change -/
def percentage_change (initial : ℕ) (final : ℕ) : ℚ :=
  (final - initial : ℚ) / initial * 100

/-- Theorem stating the average headcount and percentage change -/
theorem headcount_analysis (data : HeadcountData)
  (h1 : data.y02_03 = 10000)
  (h2 : data.y03_04 = 11000)
  (h3 : data.y04_05 = 9500)
  (h4 : data.y05_06 = 10500) :
  average_headcount data = 10125 ∧ percentage_change data.y02_03 data.y05_06 = 5 := by
  sorry

#eval average_headcount ⟨10000, 11000, 9500, 10500⟩
#eval percentage_change 10000 10500

end NUMINAMATH_CALUDE_headcount_analysis_l2642_264279


namespace NUMINAMATH_CALUDE_longest_segment_in_cylinder_l2642_264242

/-- The longest segment in a cylinder --/
theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 4) (hh : h = 10) :
  Real.sqrt ((2 * r) ^ 2 + h ^ 2) = 2 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_in_cylinder_l2642_264242


namespace NUMINAMATH_CALUDE_remainder_2345_times_1976_mod_300_l2642_264248

theorem remainder_2345_times_1976_mod_300 : (2345 * 1976) % 300 = 220 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2345_times_1976_mod_300_l2642_264248


namespace NUMINAMATH_CALUDE_ferry_tourist_count_l2642_264233

/-- The number of trips the ferry makes -/
def num_trips : ℕ := 7

/-- The number of tourists on the first trip -/
def initial_tourists : ℕ := 100

/-- The decrease in tourists per trip -/
def tourist_decrease : ℕ := 2

/-- The total number of tourists transported -/
def total_tourists : ℕ := 
  (num_trips * (2 * initial_tourists - (num_trips - 1) * tourist_decrease)) / 2

theorem ferry_tourist_count : total_tourists = 658 := by
  sorry

end NUMINAMATH_CALUDE_ferry_tourist_count_l2642_264233


namespace NUMINAMATH_CALUDE_system_two_solutions_l2642_264237

-- Define the system of equations
def system (a x y : ℝ) : Prop :=
  (x - a)^2 + y^2 = 64 ∧ (|x| - 8)^2 + (|y| - 15)^2 = 289

-- Define the set of values for parameter a
def valid_a_set : Set ℝ :=
  {-28} ∪ Set.Ioc (-24) (-8) ∪ Set.Ico 8 24 ∪ {28}

-- Theorem statement
theorem system_two_solutions (a : ℝ) :
  (∃! x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∨ y₁ ≠ y₂ ∧ system a x₁ y₁ ∧ system a x₂ y₂) ↔
  a ∈ valid_a_set :=
sorry

end NUMINAMATH_CALUDE_system_two_solutions_l2642_264237


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l2642_264238

theorem greatest_divisor_four_consecutive_integers :
  ∃ (d : ℕ), d > 0 ∧
  (∀ (n : ℕ), n > 0 → (n * (n + 1) * (n + 2) * (n + 3)) % d = 0) ∧
  (∀ (k : ℕ), k > d → ∃ (m : ℕ), m > 0 ∧ (m * (m + 1) * (m + 2) * (m + 3)) % k ≠ 0) ∧
  d = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l2642_264238


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2642_264241

theorem trigonometric_identities (α : Real) (h : Real.tan (π + α) = -1/2) :
  (2 * Real.cos (π - α) - 3 * Real.sin (π + α)) / (4 * Real.cos (α - 2*π) + Real.sin (4*π - α)) = -7/9 ∧
  Real.sin (α - 7*π) * Real.cos (α + 5*π) = -2/5 := by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2642_264241


namespace NUMINAMATH_CALUDE_diamond_two_five_l2642_264255

-- Define the diamond operation
def diamond (a b : ℤ) : ℤ := a + 3 * b ^ 2 + b

-- Theorem statement
theorem diamond_two_five : diamond 2 5 = 82 := by
  sorry

end NUMINAMATH_CALUDE_diamond_two_five_l2642_264255


namespace NUMINAMATH_CALUDE_complex_solutions_count_l2642_264268

theorem complex_solutions_count : ∃ (S : Finset ℂ), 
  (∀ z ∈ S, (z^3 - 8) / (z^2 - 3*z + 2) = 0) ∧ 
  (∀ z : ℂ, (z^3 - 8) / (z^2 - 3*z + 2) = 0 → z ∈ S) ∧ 
  Finset.card S = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_solutions_count_l2642_264268


namespace NUMINAMATH_CALUDE_G_equals_negative_three_F_l2642_264251

noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

noncomputable def G (x : ℝ) : ℝ := F ((5 * x - x^3) / (1 - 5 * x^2))

theorem G_equals_negative_three_F (x : ℝ) : G x = -3 * F x :=
by sorry

end NUMINAMATH_CALUDE_G_equals_negative_three_F_l2642_264251


namespace NUMINAMATH_CALUDE_main_theorem_l2642_264294

-- Define the propositions p and q
def p (m a : ℝ) : Prop := (m - a) * (m - 3 * a) ≤ 0
def q (m : ℝ) : Prop := (m + 2) * (m + 1) < 0

-- Define the condition for m to represent a hyperbola
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (2 + m) + y^2 / (m + 1) = 1

-- Main theorem
theorem main_theorem (m a : ℝ) 
  (h1 : m^2 - 4*a*m + 3*a^2 ≤ 0)
  (h2 : is_hyperbola m) :
  (a = -1 ∧ (p m a ∨ q m) → -3 ≤ m ∧ m ≤ -1) ∧
  (∀ m, p m a → ¬q m) ∧ (∃ m, p m a ∧ q m) →
  (-1/3 ≤ a ∧ a < 0) ∨ a ≤ -2 :=
sorry

end NUMINAMATH_CALUDE_main_theorem_l2642_264294


namespace NUMINAMATH_CALUDE_plot_size_in_acres_l2642_264219

-- Define the scale of the map
def map_scale : ℝ := 1

-- Define the dimensions of the plot on the map
def map_length : ℝ := 20
def map_width : ℝ := 25

-- Define the conversion from square miles to acres
def acres_per_square_mile : ℝ := 640

-- State the theorem
theorem plot_size_in_acres :
  let real_area : ℝ := map_length * map_width * map_scale^2
  real_area * acres_per_square_mile = 320000 := by
  sorry

end NUMINAMATH_CALUDE_plot_size_in_acres_l2642_264219


namespace NUMINAMATH_CALUDE_binomial_12_10_l2642_264226

theorem binomial_12_10 : Nat.choose 12 10 = 66 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_10_l2642_264226


namespace NUMINAMATH_CALUDE_pascal_triangle_complete_residue_l2642_264262

theorem pascal_triangle_complete_residue (p : ℕ) (hp : Prime p) :
  ∃ n : ℕ, n ≤ p^2 ∧
    ∀ k : ℕ, k < p → ∃ j : ℕ, j ≤ n ∧ (Nat.choose n j) % p = k := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_complete_residue_l2642_264262


namespace NUMINAMATH_CALUDE_power_calculation_l2642_264293

theorem power_calculation (a : ℝ) : (-a)^2 * (-a^5)^4 / a^12 * (-2 * a^4) = -2 * a^14 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l2642_264293


namespace NUMINAMATH_CALUDE_sum_of_distances_constant_l2642_264275

/-- An equilateral triangle with side length a -/
structure EquilateralTriangle where
  a : ℝ
  a_pos : a > 0

/-- A point on one side of the equilateral triangle -/
structure PointOnSide (triangle : EquilateralTriangle) where
  x : ℝ
  y : ℝ

/-- The sum of perpendicular distances from a point on one side to the other two sides -/
def sumOfDistances (triangle : EquilateralTriangle) (point : PointOnSide triangle) : ℝ := sorry

/-- Theorem: The sum of distances from any point on one side of an equilateral triangle
    to the other two sides is constant and equal to (a√3)/2 -/
theorem sum_of_distances_constant (triangle : EquilateralTriangle) 
  (point : PointOnSide triangle) : 
  sumOfDistances triangle point = (triangle.a * Real.sqrt 3) / 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_distances_constant_l2642_264275


namespace NUMINAMATH_CALUDE_upgrade_ways_count_l2642_264273

/-- Represents the number of levels in the game -/
def totalLevels : ℕ := 16

/-- Represents the level at which the special ability can first be upgraded -/
def firstSpecialLevel : ℕ := 6

/-- Represents the level at which the special ability can be upgraded for the second time -/
def secondSpecialLevel : ℕ := 11

/-- Represents the number of times the special ability must be upgraded -/
def specialUpgrades : ℕ := 2

/-- Represents the number of choices for upgrading regular abilities at each level -/
def regularChoices : ℕ := 3

/-- The function that calculates the number of ways to upgrade abilities -/
def upgradeWays : ℕ := 5 * (regularChoices ^ totalLevels)

/-- Theorem stating that the number of ways to upgrade abilities is 5 · 3^16 -/
theorem upgrade_ways_count : upgradeWays = 5 * (3 ^ 16) := by
  sorry

end NUMINAMATH_CALUDE_upgrade_ways_count_l2642_264273


namespace NUMINAMATH_CALUDE_probability_of_trio_l2642_264298

-- Define the original deck
def original_deck : ℕ := 52

-- Define the number of cards for each number
def cards_per_number : ℕ := 4

-- Define the number of different numbers in the deck
def different_numbers : ℕ := 13

-- Define the number of cards removed
def cards_removed : ℕ := 3

-- Define the remaining deck size
def remaining_deck : ℕ := original_deck - cards_removed

-- Define the number of ways to choose 3 cards from the remaining deck
def total_ways : ℕ := Nat.choose remaining_deck 3

-- Define the number of ways to choose a trio of the same number
def trio_ways : ℕ := (different_numbers - 2) * Nat.choose cards_per_number 3 + 1

-- Theorem statement
theorem probability_of_trio : 
  (trio_ways : ℚ) / total_ways = 45 / 18424 := by sorry

end NUMINAMATH_CALUDE_probability_of_trio_l2642_264298


namespace NUMINAMATH_CALUDE_erin_savings_days_l2642_264261

/-- The daily amount Erin receives in dollars -/
def daily_amount : ℕ := 3

/-- The total amount Erin needs to receive in dollars -/
def total_amount : ℕ := 30

/-- The number of days it takes Erin to receive the total amount -/
def days_to_total : ℕ := total_amount / daily_amount

theorem erin_savings_days : days_to_total = 10 := by
  sorry

end NUMINAMATH_CALUDE_erin_savings_days_l2642_264261


namespace NUMINAMATH_CALUDE_fencing_cost_calculation_l2642_264240

-- Define the ratio of the sides
def side_ratio : ℚ := 3 / 4

-- Define the area of the field in square meters
def field_area : ℝ := 7500

-- Define the cost of fencing in paise per meter
def fencing_cost_paise : ℝ := 25

-- Theorem statement
theorem fencing_cost_calculation :
  let length : ℝ := Real.sqrt (field_area * side_ratio / (side_ratio + 1))
  let width : ℝ := length / side_ratio
  let perimeter : ℝ := 2 * (length + width)
  let total_cost : ℝ := perimeter * fencing_cost_paise / 100
  total_cost = 87.5 := by sorry

end NUMINAMATH_CALUDE_fencing_cost_calculation_l2642_264240


namespace NUMINAMATH_CALUDE_least_number_with_remainder_forty_is_least_forty_has_remainder_four_least_number_is_forty_l2642_264283

theorem least_number_with_remainder (n : ℕ) : n ≥ 40 → n % 6 = 4 → ∃ k : ℕ, n = 6 * k + 4 :=
sorry

theorem forty_is_least : ∀ n : ℕ, n < 40 → n % 6 ≠ 4 :=
sorry

theorem forty_has_remainder_four : 40 % 6 = 4 :=
sorry

theorem least_number_is_forty : 
  (∃ n : ℕ, n % 6 = 4) ∧ 
  (∀ n : ℕ, n % 6 = 4 → n ≥ 40) ∧
  (40 % 6 = 4) :=
sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_forty_is_least_forty_has_remainder_four_least_number_is_forty_l2642_264283


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l2642_264207

theorem smallest_n_for_inequality : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (a : Fin n → ℝ), (∀ i, 1 < a i ∧ a i < 1000) → (∀ i j, i ≠ j → a i ≠ a j) → 
    ∃ i j, i ≠ j ∧ 0 < a i - a j ∧ a i - a j < 1 + 3 * Real.rpow (a i * a j) (1/3)) ∧
  (∀ m : ℕ, m < n → 
    ∃ (a : Fin m → ℝ), (∀ i, 1 < a i ∧ a i < 1000) ∧ (∀ i j, i ≠ j → a i ≠ a j) ∧
      ∀ i j, i ≠ j → ¬(0 < a i - a j ∧ a i - a j < 1 + 3 * Real.rpow (a i * a j) (1/3))) ∧
  n = 11 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l2642_264207


namespace NUMINAMATH_CALUDE_absent_workers_l2642_264264

/-- Given a group of workers and their work schedule, calculate the number of absent workers. -/
theorem absent_workers 
  (total_workers : ℕ) 
  (original_days : ℕ) 
  (actual_days : ℕ) 
  (h1 : total_workers = 15)
  (h2 : original_days = 40)
  (h3 : actual_days = 60) :
  ∃ (absent : ℕ), 
    absent = 5 ∧ 
    (total_workers - absent) * actual_days = total_workers * original_days :=
by sorry


end NUMINAMATH_CALUDE_absent_workers_l2642_264264


namespace NUMINAMATH_CALUDE_A_intersect_B_l2642_264281

def A : Set ℝ := {-1, 0, 1}

def B : Set ℝ := {y | ∃ x ∈ A, y = Real.cos (Real.pi * x)}

theorem A_intersect_B : A ∩ B = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l2642_264281


namespace NUMINAMATH_CALUDE_triangle_area_l2642_264235

/-- Given a triangle with one side of length 2, a median to this side of length 1,
    and the sum of the other two sides equal to 1 + √3,
    prove that the area of the triangle is √3/2. -/
theorem triangle_area (a b c : ℝ) (h1 : c = 2) (h2 : a + b = 1 + Real.sqrt 3)
  (h3 : ∃ (m : ℝ), m = 1 ∧ m^2 = (a^2 + b^2) / 4 + c^2 / 16) :
  (a * b) / 2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2642_264235


namespace NUMINAMATH_CALUDE_star_properties_l2642_264269

-- Define the "※" operation
def star (m n : ℚ) : ℚ := 3 * m - n

-- Theorem statement
theorem star_properties :
  (star 2 10 = -4) ∧
  (∃ a b c : ℚ, star a (b + c) ≠ star a b + star a c) :=
by sorry

end NUMINAMATH_CALUDE_star_properties_l2642_264269


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l2642_264244

theorem abs_sum_inequality (x : ℝ) : 
  |x - 3| + |x + 4| < 10 ↔ x ∈ Set.Ioo (-5.5) 4.5 := by sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l2642_264244


namespace NUMINAMATH_CALUDE_largest_n_with_odd_residues_l2642_264259

theorem largest_n_with_odd_residues : ∃ (n : ℕ), n = 505 ∧ n > 10 ∧
  (∀ (k : ℕ), 2 ≤ k^2 ∧ k^2 ≤ n / 2 → n % (k^2) % 2 = 1) ∧
  (∀ (m : ℕ), m > n → ∃ (j : ℕ), 2 ≤ j^2 ∧ j^2 ≤ m / 2 ∧ m % (j^2) % 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_largest_n_with_odd_residues_l2642_264259


namespace NUMINAMATH_CALUDE_log_exists_iff_power_of_base_no_log_for_numbers_between_zero_and_one_l2642_264289

-- Define the base for logarithms
variable (a : ℝ)

-- Define the conditions for the base
variable (ha : a > 0 ∧ a ≠ 1)

-- Theorem 1: With only integer exponents, logarithms exist only for powers of the base
theorem log_exists_iff_power_of_base (b : ℝ) :
  (∃ n : ℤ, b = a^n) ↔ ∃ x : ℝ, a^x = b :=
sorry

-- Theorem 2: With only positive exponents, logarithms don't exist for numbers between 0 and 1
theorem no_log_for_numbers_between_zero_and_one (x : ℝ) (hx : 0 < x ∧ x < 1) :
  ¬∃ y : ℝ, y > 0 ∧ a^y = x :=
sorry

end NUMINAMATH_CALUDE_log_exists_iff_power_of_base_no_log_for_numbers_between_zero_and_one_l2642_264289


namespace NUMINAMATH_CALUDE_circle_symmetry_implies_slope_l2642_264245

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 6*y + 14 = 0

-- Define the line equation
def line_equation (a x y : ℝ) : Prop :=
  a*x + 4*y - 6 = 0

-- Define symmetry of circle about line
def circle_symmetrical_about_line (a : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), circle_equation x₀ y₀ ∧ line_equation a x₀ y₀

-- Theorem statement
theorem circle_symmetry_implies_slope :
  ∀ a : ℝ, circle_symmetrical_about_line a → (a = 6) :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_implies_slope_l2642_264245


namespace NUMINAMATH_CALUDE_neds_weekly_revenue_l2642_264299

/-- Calculates the weekly revenue for Ned's left-handed mouse store -/
def calculate_weekly_revenue (normal_mouse_price : ℝ) (price_increase_percentage : ℝ) 
  (daily_sales : ℕ) (open_days_per_week : ℕ) : ℝ :=
  let left_handed_mouse_price := normal_mouse_price * (1 + price_increase_percentage)
  let daily_revenue := left_handed_mouse_price * daily_sales
  daily_revenue * open_days_per_week

/-- Theorem stating that Ned's weekly revenue is $15600 -/
theorem neds_weekly_revenue : 
  calculate_weekly_revenue 120 0.3 25 4 = 15600 := by
  sorry

#eval calculate_weekly_revenue 120 0.3 25 4

end NUMINAMATH_CALUDE_neds_weekly_revenue_l2642_264299


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l2642_264243

def B : Set ℕ := {x | ∃ n : ℕ, x = 4*n + 6 ∧ n > 0}

theorem gcd_of_B_is_two : 
  ∃ d : ℕ, d > 1 ∧ (∀ x ∈ B, d ∣ x) ∧ 
  (∀ k : ℕ, k > 1 → (∀ x ∈ B, k ∣ x) → k ≤ d) ∧ d = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l2642_264243


namespace NUMINAMATH_CALUDE_negation_of_no_vegetarian_students_eat_at_cafeteria_l2642_264224

-- Define the universe of discourse
variable (Student : Type)

-- Define predicates
variable (isVegetarian : Student → Prop)
variable (eatsAtCafeteria : Student → Prop)

-- State the theorem
theorem negation_of_no_vegetarian_students_eat_at_cafeteria :
  ¬(∀ s : Student, isVegetarian s → ¬(eatsAtCafeteria s)) ↔
  ∃ s : Student, isVegetarian s ∧ eatsAtCafeteria s :=
by sorry


end NUMINAMATH_CALUDE_negation_of_no_vegetarian_students_eat_at_cafeteria_l2642_264224


namespace NUMINAMATH_CALUDE_no_integer_root_2016_l2642_264214

/-- A cubic polynomial with integer coefficients -/
def cubic_poly (a b c d : ℤ) : ℤ → ℤ := fun x ↦ a * x^3 + b * x^2 + c * x + d

theorem no_integer_root_2016 (a b c d : ℤ) :
  cubic_poly a b c d 1 = 2015 →
  cubic_poly a b c d 2 = 2017 →
  ∀ k : ℤ, cubic_poly a b c d k ≠ 2016 := by
sorry

end NUMINAMATH_CALUDE_no_integer_root_2016_l2642_264214


namespace NUMINAMATH_CALUDE_unique_root_of_equation_l2642_264263

open Real

theorem unique_root_of_equation :
  ∃! x : ℝ, x > 0 ∧ 1 - x - x * log x = 0 :=
by
  -- Define the function
  let f : ℝ → ℝ := λ x ↦ 1 - x - x * log x

  -- Assume f is decreasing on (0, +∞)
  have h_decreasing : ∀ x y, 0 < x → 0 < y → x < y → f y < f x := sorry

  -- Prove there exists exactly one root
  sorry

end NUMINAMATH_CALUDE_unique_root_of_equation_l2642_264263


namespace NUMINAMATH_CALUDE_unique_modular_congruence_l2642_264276

theorem unique_modular_congruence :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 6 ∧ n ≡ -2023 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_congruence_l2642_264276


namespace NUMINAMATH_CALUDE_unique_perfect_square_solution_l2642_264230

/-- A number is a perfect square if it's equal to some integer squared. -/
def IsPerfectSquare (x : ℤ) : Prop := ∃ k : ℤ, x = k^2

/-- The theorem states that 125 is the only positive integer n such that
    both 20n and 5n + 275 are perfect squares. -/
theorem unique_perfect_square_solution :
  ∀ n : ℕ+, (IsPerfectSquare (20 * n.val)) ∧ (IsPerfectSquare (5 * n.val + 275)) ↔ n = 125 := by
  sorry

end NUMINAMATH_CALUDE_unique_perfect_square_solution_l2642_264230


namespace NUMINAMATH_CALUDE_pirate_loot_sum_l2642_264249

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List ℕ) (b : ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The value from S.S. AOPS in base 5 -/
def aops_value : List ℕ := [4, 2, 1, 3]

/-- The value from S.S. BOPS in base 7 -/
def bops_value : List ℕ := [2, 1, 0, 1]

/-- The value from S.S. COPS in base 8 -/
def cops_value : List ℕ := [3, 2, 1]

/-- The theorem to be proved -/
theorem pirate_loot_sum :
  to_base_10 aops_value 5 + to_base_10 bops_value 7 + to_base_10 cops_value 8 = 849 := by
  sorry

end NUMINAMATH_CALUDE_pirate_loot_sum_l2642_264249


namespace NUMINAMATH_CALUDE_division_problem_l2642_264204

theorem division_problem (n t : ℝ) (hn : n > 0) (ht : t > 0) 
  (h : n / t = (n + 2) / (t + 7)) : 
  ∃ z, n / t = (n + 3) / (t + z) ∧ z = 21 / 2 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2642_264204


namespace NUMINAMATH_CALUDE_tigger_climbing_speed_ratio_l2642_264201

theorem tigger_climbing_speed_ratio :
  ∀ (T t : ℝ),
  T > 0 ∧ t > 0 →
  2 * T = t / 3 →
  T + t = 2 * T + t / 3 →
  T / t = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_tigger_climbing_speed_ratio_l2642_264201


namespace NUMINAMATH_CALUDE_coes_speed_l2642_264280

theorem coes_speed (teena_speed : ℝ) (initial_distance : ℝ) (time : ℝ) (final_distance : ℝ) :
  teena_speed = 55 →
  initial_distance = 7.5 →
  time = 1.5 →
  final_distance = 15 →
  ∃ coe_speed : ℝ,
    coe_speed = 50 ∧
    teena_speed * time - coe_speed * time = final_distance + initial_distance :=
by
  sorry

end NUMINAMATH_CALUDE_coes_speed_l2642_264280


namespace NUMINAMATH_CALUDE_zeroth_power_of_nonzero_rational_l2642_264231

theorem zeroth_power_of_nonzero_rational (r : ℚ) (h : r ≠ 0) : r^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zeroth_power_of_nonzero_rational_l2642_264231


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2642_264282

theorem unique_positive_solution : ∃! y : ℝ, y > 0 ∧ (y - 6) / 16 = 6 / (y - 16) := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2642_264282


namespace NUMINAMATH_CALUDE_shipping_cost_calculation_l2642_264229

def fish_weight : ℕ := 540
def crate_capacity : ℕ := 30
def crate_cost : ℚ := 3/2

theorem shipping_cost_calculation :
  (fish_weight / crate_capacity) * crate_cost = 27 := by
  sorry

end NUMINAMATH_CALUDE_shipping_cost_calculation_l2642_264229


namespace NUMINAMATH_CALUDE_sum_of_xyz_l2642_264252

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 30) (h2 : x * z = 60) (h3 : y * z = 90) :
  x + y + z = 11 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l2642_264252


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l2642_264256

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The main theorem -/
theorem line_through_point_parallel_to_line 
  (M : Point) 
  (A B : Point) 
  (h_M : M.x = 2 ∧ M.y = -3)
  (h_A : A.x = 1 ∧ A.y = 2)
  (h_B : B.x = -1 ∧ B.y = -5) :
  ∃ (l : Line), 
    l.a = 7 ∧ l.b = -2 ∧ l.c = -20 ∧ 
    M.liesOn l ∧
    l.isParallelTo (Line.mk (B.y - A.y) (A.x - B.x) (B.x * A.y - A.x * B.y)) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l2642_264256


namespace NUMINAMATH_CALUDE_array_sum_remainder_l2642_264287

/-- Represents the sum of all terms in a 1/1004-array -/
def array_sum : ℚ := (2 * 1004^2) / ((2 * 1004 - 1) * (1004 - 1))

/-- Numerator of the array sum when expressed in lowest terms -/
def m : ℕ := 2 * 1004^2

/-- Denominator of the array sum when expressed in lowest terms -/
def n : ℕ := (2 * 1004 - 1) * (1004 - 1)

/-- The main theorem stating that (m + n) ≡ 0 (mod 1004) -/
theorem array_sum_remainder :
  (m + n) % 1004 = 0 := by sorry

end NUMINAMATH_CALUDE_array_sum_remainder_l2642_264287


namespace NUMINAMATH_CALUDE_christines_stickers_l2642_264277

theorem christines_stickers (total_needed : ℕ) (more_needed : ℕ) (h1 : total_needed = 30) (h2 : more_needed = 19) :
  total_needed - more_needed = 11 := by
  sorry

end NUMINAMATH_CALUDE_christines_stickers_l2642_264277


namespace NUMINAMATH_CALUDE_symmetry_about_x_equals_one_l2642_264286

/-- Given a function f(x) = x, if its graph is symmetric about the line x = 1,
    then the corresponding function g(x) is equal to 3x - 2. -/
theorem symmetry_about_x_equals_one (f g : ℝ → ℝ) :
  (∀ x, f x = x) →
  (∀ x, f (2 - x) = g x) →
  (∀ x, g x = 3*x - 2) := by
sorry

end NUMINAMATH_CALUDE_symmetry_about_x_equals_one_l2642_264286


namespace NUMINAMATH_CALUDE_triangle_rectangle_equal_area_l2642_264290

theorem triangle_rectangle_equal_area (s h : ℝ) (s_pos : 0 < s) :
  (1 / 2) * s * h = 2 * s^2 → h = 4 * s :=
by sorry

end NUMINAMATH_CALUDE_triangle_rectangle_equal_area_l2642_264290


namespace NUMINAMATH_CALUDE_arithmetic_sequence_remainder_l2642_264296

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- The problem statement -/
theorem arithmetic_sequence_remainder (a₁ aₙ d : ℕ) 
  (h₁ : a₁ = 3)
  (h₂ : aₙ = 273)
  (h₃ : d = 6)
  (h₄ : ∀ k, 1 ≤ k ∧ k ≤ (aₙ - a₁) / d + 1 → a₁ + (k - 1) * d = 6 * k - 3) :
  arithmetic_sum a₁ aₙ d % 8 = 2 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_remainder_l2642_264296


namespace NUMINAMATH_CALUDE_expected_worth_is_one_third_l2642_264202

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the probability of a coin flip outcome -/
def probability : CoinFlip → ℚ
| CoinFlip.Heads => 2/3
| CoinFlip.Tails => 1/3

/-- Represents the monetary outcome of a coin flip -/
def monetaryOutcome : CoinFlip → ℤ
| CoinFlip.Heads => 5
| CoinFlip.Tails => -9

/-- The expected worth of a coin flip -/
def expectedWorth : ℚ :=
  (probability CoinFlip.Heads * monetaryOutcome CoinFlip.Heads) +
  (probability CoinFlip.Tails * monetaryOutcome CoinFlip.Tails)

theorem expected_worth_is_one_third :
  expectedWorth = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_expected_worth_is_one_third_l2642_264202


namespace NUMINAMATH_CALUDE_even_function_implies_m_equals_two_l2642_264265

def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + (m - 2) * x + (m^2 - 7 * m + 12)

theorem even_function_implies_m_equals_two (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_m_equals_two_l2642_264265


namespace NUMINAMATH_CALUDE_composite_divides_factorial_l2642_264221

theorem composite_divides_factorial (n : ℕ) (h1 : n > 4) (h2 : ¬ Nat.Prime n) :
  n ∣ Nat.factorial (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_composite_divides_factorial_l2642_264221


namespace NUMINAMATH_CALUDE_numbered_cube_sum_l2642_264236

/-- Represents a cube with numbered faces -/
structure NumberedCube where
  numbers : Fin 6 → ℕ
  consecutive_even : ∀ i : Fin 5, numbers i.succ = numbers i + 2
  smallest_is_12 : numbers 0 = 12
  opposite_faces_sum_equal : 
    numbers 0 + numbers 5 = numbers 1 + numbers 4 ∧ 
    numbers 1 + numbers 4 = numbers 2 + numbers 3

/-- The sum of all numbers on the cube is 102 -/
theorem numbered_cube_sum (cube : NumberedCube) : 
  (Finset.univ.sum cube.numbers) = 102 := by
  sorry

end NUMINAMATH_CALUDE_numbered_cube_sum_l2642_264236


namespace NUMINAMATH_CALUDE_point_location_l2642_264223

theorem point_location (m : ℝ) :
  (m < 0 ∧ 1 > 0) →  -- P (m, 1) is in the second quadrant
  (-m > 0 ∧ 0 = 0)   -- Q (-m, 0) is on the positive half of the x-axis
  := by sorry

end NUMINAMATH_CALUDE_point_location_l2642_264223


namespace NUMINAMATH_CALUDE_hyperbola_minimum_value_l2642_264246

theorem hyperbola_minimum_value (a b : ℝ) (ha : a ≥ 1) (hb : b ≥ 1) :
  let eccentricity := (a^2 + b^2).sqrt / a
  eccentricity = 2 →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (b^2 + 1) / (Real.sqrt 3 * a) ≥ 4 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_minimum_value_l2642_264246


namespace NUMINAMATH_CALUDE_meaningful_expression_l2642_264209

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x - 1)) ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_l2642_264209


namespace NUMINAMATH_CALUDE_general_quadratic_is_quadratic_specific_quadratic_is_quadratic_l2642_264228

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation ax² + bx + c = 0 is quadratic -/
theorem general_quadratic_is_quadratic (a b c : ℝ) (ha : a ≠ 0) :
  is_quadratic_equation (λ x => a * x^2 + b * x + c) :=
sorry

/-- The equation x² - 4 = 0 is quadratic -/
theorem specific_quadratic_is_quadratic :
  is_quadratic_equation (λ x => x^2 - 4) :=
sorry

end NUMINAMATH_CALUDE_general_quadratic_is_quadratic_specific_quadratic_is_quadratic_l2642_264228


namespace NUMINAMATH_CALUDE_min_m_is_one_l2642_264203

/-- A function is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem min_m_is_one (f g h : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = Real.exp x) →
  (∀ x, f x = g x - h x) →
  IsEven g →
  IsOdd h →
  (∀ x ∈ Set.Icc (-1) 1, m * g x + h x ≥ 0) →
  (∀ m' : ℝ, (∀ x ∈ Set.Icc (-1) 1, m' * g x + h x ≥ 0) → m' ≥ m) →
  m = 1 := by sorry

end NUMINAMATH_CALUDE_min_m_is_one_l2642_264203


namespace NUMINAMATH_CALUDE_exercise_book_count_l2642_264288

theorem exercise_book_count (pencil_count : ℕ) (pencil_ratio : ℕ) (book_ratio : ℕ) :
  pencil_count = 120 →
  pencil_ratio = 10 →
  book_ratio = 3 →
  (pencil_count * book_ratio) / pencil_ratio = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_exercise_book_count_l2642_264288


namespace NUMINAMATH_CALUDE_two_roots_k_range_l2642_264225

theorem two_roots_k_range (f : ℝ → ℝ) (k : ℝ) :
  (∀ x, f x = x * Real.exp (-2 * x) + k) →
  (∃! x₁ x₂, x₁ ∈ Set.Ioo (-2 : ℝ) 2 ∧ x₂ ∈ Set.Ioo (-2 : ℝ) 2 ∧ x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) →
  k ∈ Set.Ioo (-(1 / (2 * Real.exp 1))) (-(2 / Real.exp 4)) :=
by sorry

end NUMINAMATH_CALUDE_two_roots_k_range_l2642_264225


namespace NUMINAMATH_CALUDE_factor_expression_l2642_264220

theorem factor_expression (x : ℝ) : 84 * x^7 - 270 * x^13 = 6 * x^7 * (14 - 45 * x^6) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2642_264220


namespace NUMINAMATH_CALUDE_solution_and_parabola_equivalence_l2642_264232

-- Define the set of solutions for x - 3 > 0
def solution_set : Set ℝ := {x | x - 3 > 0}

-- Define the set of points on the parabola y = x^2 - 1
def parabola_points : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2 - 1}

theorem solution_and_parabola_equivalence :
  (solution_set = {x : ℝ | x > 3}) ∧
  (parabola_points = {p : ℝ × ℝ | p.2 = p.1^2 - 1}) := by
  sorry

end NUMINAMATH_CALUDE_solution_and_parabola_equivalence_l2642_264232


namespace NUMINAMATH_CALUDE_line_intersects_segment_slope_range_l2642_264260

-- Define points A and B
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (-2, -1)

-- Define the line l
def l (k : ℝ) (x : ℝ) : ℝ := k * (x - 2) + 1

-- Define the segment AB
def segmentAB (t : ℝ) : ℝ × ℝ := (
  (1 - t) * A.1 + t * B.1,
  (1 - t) * A.2 + t * B.2
)

-- Theorem statement
theorem line_intersects_segment_slope_range :
  ∀ k : ℝ, (∃ t ∈ (Set.Icc 0 1), l k (segmentAB t).1 = (segmentAB t).2) →
  -2 ≤ k ∧ k ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_line_intersects_segment_slope_range_l2642_264260


namespace NUMINAMATH_CALUDE_lilibeth_baskets_l2642_264212

/-- The number of strawberries each basket holds -/
def strawberries_per_basket : ℕ := 50

/-- The total number of strawberries picked by Lilibeth and her friends -/
def total_strawberries : ℕ := 1200

/-- The number of people picking strawberries (Lilibeth and her three friends) -/
def number_of_pickers : ℕ := 4

/-- The number of baskets Lilibeth filled -/
def baskets_filled : ℕ := 6

theorem lilibeth_baskets :
  strawberries_per_basket * number_of_pickers * baskets_filled = total_strawberries :=
by sorry

end NUMINAMATH_CALUDE_lilibeth_baskets_l2642_264212


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_l2642_264253

/-- Given five consecutive odd numbers, prove that if the sum of the first and third is 146, then the fifth number is 79 -/
theorem consecutive_odd_numbers (a b c d e : ℤ) : 
  (∃ k : ℤ, a = 2*k + 1) →  -- a is odd
  b = a + 2 →               -- b is the next odd number after a
  c = a + 4 →               -- c is the next odd number after b
  d = a + 6 →               -- d is the next odd number after c
  e = a + 8 →               -- e is the next odd number after d
  a + c = 146 →             -- sum of a and c is 146
  e = 79 := by              -- prove that e equals 79
sorry


end NUMINAMATH_CALUDE_consecutive_odd_numbers_l2642_264253


namespace NUMINAMATH_CALUDE_traveler_distance_l2642_264210

theorem traveler_distance (north south west east : ℝ) : 
  north = 25 → 
  south = 10 → 
  west = 15 → 
  east = 7 → 
  let net_north := north - south
  let net_west := west - east
  Real.sqrt (net_north ^ 2 + net_west ^ 2) = 17 :=
by sorry

end NUMINAMATH_CALUDE_traveler_distance_l2642_264210


namespace NUMINAMATH_CALUDE_constant_term_expansion_l2642_264295

theorem constant_term_expansion (x : ℝ) : 
  (x^4 + 3*x^2 + 6) * (2*x^3 + x^2 + 10) = x^7 + 2*x^6 + 3*x^5 + 10*x^4 + 6*x^3 + 30*x^2 + 60 := by
  sorry

#check constant_term_expansion

end NUMINAMATH_CALUDE_constant_term_expansion_l2642_264295


namespace NUMINAMATH_CALUDE_total_angles_count_l2642_264234

/-- The number of 90° angles in a rectangle -/
def rectangleAngles : ℕ := 4

/-- The number of 90° angles in a square -/
def squareAngles : ℕ := 4

/-- The number of rectangular flower beds in the park -/
def flowerBeds : ℕ := 3

/-- The number of square goal areas in the football field -/
def goalAreas : ℕ := 4

/-- The total number of 90° angles in the park and football field -/
def totalAngles : ℕ := 
  rectangleAngles + flowerBeds * rectangleAngles + 
  squareAngles + goalAreas * squareAngles

theorem total_angles_count : totalAngles = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_angles_count_l2642_264234


namespace NUMINAMATH_CALUDE_commercial_viewers_l2642_264285

/-- Calculates the number of commercial viewers given revenue data -/
theorem commercial_viewers (revenue_per_view : ℚ) (revenue_per_sub : ℚ) 
  (num_subs : ℕ) (total_revenue : ℚ) : 
  revenue_per_view > 0 → 
  (total_revenue - revenue_per_sub * num_subs) / revenue_per_view = 100 → 
  ∃ (num_viewers : ℕ), num_viewers = 100 :=
by
  sorry

#check commercial_viewers (1/2) 1 27 77

end NUMINAMATH_CALUDE_commercial_viewers_l2642_264285


namespace NUMINAMATH_CALUDE_borya_segments_imply_isosceles_l2642_264213

/-- A triangle with vertices A, B, and C. -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- A segment represented by its length. -/
structure Segment where
  length : ℝ

/-- The set of nine segments drawn by Borya. -/
def BoryaSegments : Set Segment := sorry

/-- The three altitudes of the triangle. -/
def altitudes (t : Triangle) : Set Segment := sorry

/-- The three angle bisectors of the triangle. -/
def angleBisectors (t : Triangle) : Set Segment := sorry

/-- The three medians of the triangle. -/
def medians (t : Triangle) : Set Segment := sorry

/-- Predicate to check if a triangle is isosceles. -/
def isIsosceles (t : Triangle) : Prop := sorry

theorem borya_segments_imply_isosceles (t : Triangle) 
  (h1 : ∃ s1 s2 s3 : Segment, s1 ∈ BoryaSegments ∧ s2 ∈ BoryaSegments ∧ s3 ∈ BoryaSegments ∧ 
                              {s1, s2, s3} = altitudes t)
  (h2 : ∃ s1 s2 s3 : Segment, s1 ∈ BoryaSegments ∧ s2 ∈ BoryaSegments ∧ s3 ∈ BoryaSegments ∧ 
                              {s1, s2, s3} = angleBisectors t)
  (h3 : ∃ s1 s2 s3 : Segment, s1 ∈ BoryaSegments ∧ s2 ∈ BoryaSegments ∧ s3 ∈ BoryaSegments ∧ 
                              {s1, s2, s3} = medians t)
  (h4 : ∀ s ∈ BoryaSegments, ∃ s' ∈ BoryaSegments, s ≠ s' ∧ s.length = s'.length) :
  isIsosceles t :=
sorry

end NUMINAMATH_CALUDE_borya_segments_imply_isosceles_l2642_264213


namespace NUMINAMATH_CALUDE_problem_solution_l2642_264222

theorem problem_solution : ∃ x : ℝ, 70 + 5 * 12 / (x / 3) = 71 ∧ x = 180 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2642_264222


namespace NUMINAMATH_CALUDE_a_less_than_abs_a_implies_negative_l2642_264217

theorem a_less_than_abs_a_implies_negative (a : ℝ) : a < |a| → a < 0 := by
  sorry

end NUMINAMATH_CALUDE_a_less_than_abs_a_implies_negative_l2642_264217


namespace NUMINAMATH_CALUDE_unique_sum_of_squares_and_product_l2642_264284

theorem unique_sum_of_squares_and_product (a b : ℕ+) : 
  a ≤ b → 
  a.val^2 + b.val^2 + 8 * a.val * b.val = 2010 → 
  a.val + b.val = 42 :=
by sorry

end NUMINAMATH_CALUDE_unique_sum_of_squares_and_product_l2642_264284


namespace NUMINAMATH_CALUDE_min_value_theorem_l2642_264215

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 3*y = 5*x*y) :
  3*x + 2*y ≥ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + 3*y₀ = 5*x₀*y₀ ∧ 3*x₀ + 2*y₀ = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2642_264215


namespace NUMINAMATH_CALUDE_arthur_walk_distance_l2642_264267

/-- The distance Arthur walked in miles -/
def distance_walked (blocks_west : ℕ) (blocks_south : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_west + blocks_south : ℚ) * miles_per_block

/-- Theorem: Arthur walks 4.5 miles -/
theorem arthur_walk_distance :
  distance_walked 8 10 (1/4) = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_arthur_walk_distance_l2642_264267


namespace NUMINAMATH_CALUDE_b_4_lt_b_7_l2642_264291

def b (n : ℕ) (α : ℕ → ℕ) : ℚ :=
  match n with
  | 0 => 0
  | 1 => 1 + 1 / α 1
  | n+1 => 1 + 1 / (b n α + 1 / α (n+1))

theorem b_4_lt_b_7 (α : ℕ → ℕ) : b 4 α < b 7 α := by
  sorry

end NUMINAMATH_CALUDE_b_4_lt_b_7_l2642_264291


namespace NUMINAMATH_CALUDE_original_fraction_is_two_thirds_l2642_264211

theorem original_fraction_is_two_thirds 
  (x y : ℚ) 
  (h1 : x / (y + 1) = 1/2) 
  (h2 : (x + 1) / y = 1) : 
  x / y = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_original_fraction_is_two_thirds_l2642_264211


namespace NUMINAMATH_CALUDE_sphere_volume_circumscribing_rectangular_solid_l2642_264254

theorem sphere_volume_circumscribing_rectangular_solid :
  let length : ℝ := 2
  let width : ℝ := 1
  let height : ℝ := 2
  let space_diagonal : ℝ := Real.sqrt (length^2 + width^2 + height^2)
  let sphere_radius : ℝ := space_diagonal / 2
  let sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius^3
  sphere_volume = (9 / 2) * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_circumscribing_rectangular_solid_l2642_264254


namespace NUMINAMATH_CALUDE_problem_solution_l2642_264216

theorem problem_solution (x y : ℝ) (h1 : x + y = 6) (h2 : x * y = 5) : 
  (2 / x + 2 / y = 12 / 5) ∧ ((x - y)^2 = 16) ∧ (x^2 + y^2 = 26) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2642_264216


namespace NUMINAMATH_CALUDE_ashok_pyarelal_capital_ratio_l2642_264278

/-- Given a total loss and Pyarelal's loss, prove the ratio of Ashok's capital to Pyarelal's capital -/
theorem ashok_pyarelal_capital_ratio 
  (total_loss : ℕ) 
  (pyarelal_loss : ℕ) 
  (h1 : total_loss = 1200) 
  (h2 : pyarelal_loss = 1080) : 
  ∃ (a p : ℕ), a ≠ 0 ∧ p ≠ 0 ∧ a * 9 = p * 1 := by
  sorry

end NUMINAMATH_CALUDE_ashok_pyarelal_capital_ratio_l2642_264278


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2642_264250

theorem simplify_and_evaluate (a b : ℝ) (h1 : a = 1) (h2 : b = 2) :
  (2*a - b)^2 - (2*a + b)*(b - 2*a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2642_264250


namespace NUMINAMATH_CALUDE_number_of_grandchildren_excluding_shelby_l2642_264239

/-- Proves the number of grandchildren excluding Shelby, given the inheritance details --/
theorem number_of_grandchildren_excluding_shelby
  (total_inheritance : ℕ)
  (shelby_share : ℕ)
  (remaining_share : ℕ)
  (one_grandchild_share : ℕ)
  (h1 : total_inheritance = 124600)
  (h2 : shelby_share = total_inheritance / 2)
  (h3 : remaining_share = total_inheritance - shelby_share)
  (h4 : one_grandchild_share = 6230)
  (h5 : remaining_share % one_grandchild_share = 0) :
  remaining_share / one_grandchild_share = 10 := by
  sorry

#check number_of_grandchildren_excluding_shelby

end NUMINAMATH_CALUDE_number_of_grandchildren_excluding_shelby_l2642_264239


namespace NUMINAMATH_CALUDE_aquarium_fish_count_l2642_264200

/-- Given an initial number of fish and a number of fish added to an aquarium,
    the total number of fish is equal to the sum of the initial number and the number added. -/
theorem aquarium_fish_count (initial : ℕ) (added : ℕ) :
  initial + added = initial + added :=
by sorry

end NUMINAMATH_CALUDE_aquarium_fish_count_l2642_264200


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l2642_264270

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: For an arithmetic sequence, a₆ < a₇ if and only if a₆ < a₈ -/
theorem arithmetic_sequence_inequality (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  a 6 < a 7 ↔ a 6 < a 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l2642_264270


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2642_264272

theorem arithmetic_sequence_sum (a₁ d : ℝ) (h₁ : d ≠ 0) :
  let a : ℕ → ℝ := λ n => a₁ + (n - 1) * d
  let S : ℕ → ℝ := λ n => n * a₁ + n * (n - 1) / 2 * d
  (a 4)^2 = (a 3) * (a 7) ∧ S 8 = 32 → S 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2642_264272


namespace NUMINAMATH_CALUDE_alyssa_book_count_l2642_264266

/-- The number of books Alyssa has -/
def alyssas_books : ℕ := 36

/-- The number of books Nancy has -/
def nancys_books : ℕ := 252

theorem alyssa_book_count :
  (nancys_books = 7 * alyssas_books) → alyssas_books = 36 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_book_count_l2642_264266


namespace NUMINAMATH_CALUDE_mixed_number_calculation_l2642_264206

theorem mixed_number_calculation : 7 * (12 + 2/5) - 3 = 83.8 := by
  sorry

end NUMINAMATH_CALUDE_mixed_number_calculation_l2642_264206


namespace NUMINAMATH_CALUDE_sector_perimeter_l2642_264227

theorem sector_perimeter (θ : Real) (r : Real) (h1 : θ = 54) (h2 : r = 20) : 
  let α := θ * (π / 180)
  let arc_length := α * r
  let perimeter := arc_length + 2 * r
  perimeter = 6 * π + 40 := by sorry

end NUMINAMATH_CALUDE_sector_perimeter_l2642_264227
