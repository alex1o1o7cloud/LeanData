import Mathlib

namespace find_p_value_l852_85289

theorem find_p_value (x y z p : ℝ) 
  (h1 : 8 / (x + y) = p / (x + z)) 
  (h2 : p / (x + z) = 12 / (z - y)) : p = 20 := by
  sorry

end find_p_value_l852_85289


namespace exists_center_nail_pierces_one_cardboard_l852_85201

/-- A cardboard figure -/
structure Cardboard where
  shape : Set (ℝ × ℝ)

/-- A rectangular box bottom -/
structure Box where
  width : ℝ
  height : ℝ

/-- A configuration of two cardboard pieces on a box bottom -/
structure Configuration where
  box : Box
  piece1 : Cardboard
  piece2 : Cardboard
  position1 : ℝ × ℝ
  position2 : ℝ × ℝ

/-- Predicate to check if a point is covered by a cardboard piece at a given position -/
def covers (c : Cardboard) (pos : ℝ × ℝ) (point : ℝ × ℝ) : Prop :=
  (point.1 - pos.1, point.2 - pos.2) ∈ c.shape

/-- Predicate to check if a configuration completely covers the box bottom -/
def completelyCovers (config : Configuration) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ config.box.width → 0 ≤ y ∧ y ≤ config.box.height →
    covers config.piece1 config.position1 (x, y) ∨ covers config.piece2 config.position2 (x, y)

/-- Theorem stating that there exists a configuration where the center nail pierces only one cardboard -/
theorem exists_center_nail_pierces_one_cardboard :
  ∃ (config : Configuration), completelyCovers config ∧
    (covers config.piece1 config.position1 (config.box.width / 2, config.box.height / 2) ≠
     covers config.piece2 config.position2 (config.box.width / 2, config.box.height / 2)) :=
by sorry

end exists_center_nail_pierces_one_cardboard_l852_85201


namespace x_days_to_complete_work_l852_85235

/-- The number of days required for x and y to complete the work together -/
def days_xy : ℚ := 12

/-- The number of days required for y to complete the work alone -/
def days_y : ℚ := 24

/-- The fraction of work completed by a worker in one day -/
def work_per_day (days : ℚ) : ℚ := 1 / days

theorem x_days_to_complete_work : 
  1 / (work_per_day days_xy - work_per_day days_y) = 24 := by sorry

end x_days_to_complete_work_l852_85235


namespace complex_number_in_first_quadrant_l852_85228

theorem complex_number_in_first_quadrant : 
  let z : ℂ := 1 - (1 / Complex.I)
  (z.re > 0) ∧ (z.im > 0) := by sorry

end complex_number_in_first_quadrant_l852_85228


namespace dissimilar_terms_eq_distribution_ways_l852_85267

/-- The number of dissimilar terms in the expansion of (a + b + c + d)^7 -/
def dissimilar_terms : ℕ := sorry

/-- The number of ways to distribute 7 indistinguishable objects into 4 distinguishable boxes -/
def distribution_ways : ℕ := sorry

/-- Theorem stating that the number of dissimilar terms in (a + b + c + d)^7 
    is equal to the number of ways to distribute 7 objects into 4 boxes -/
theorem dissimilar_terms_eq_distribution_ways : 
  dissimilar_terms = distribution_ways := by sorry

end dissimilar_terms_eq_distribution_ways_l852_85267


namespace condition_satisfied_pairs_l852_85231

/-- Checks if a pair of positive integers (m, n) satisfies the given condition -/
def satisfies_condition (m n : ℕ+) : Prop :=
  ∀ x y : ℝ, m ≤ x ∧ x ≤ n ∧ m ≤ y ∧ y ≤ n → m ≤ (5/x + 7/y) ∧ (5/x + 7/y) ≤ n

/-- The only positive integer pairs (m, n) satisfying the condition are (1,12), (2,6), and (3,4) -/
theorem condition_satisfied_pairs :
  ∀ m n : ℕ+, satisfies_condition m n ↔ (m = 1 ∧ n = 12) ∨ (m = 2 ∧ n = 6) ∨ (m = 3 ∧ n = 4) :=
sorry

end condition_satisfied_pairs_l852_85231


namespace paula_cans_used_l852_85295

/-- Represents the painting scenario with Paula the painter --/
structure PaintingScenario where
  initial_rooms : ℕ
  lost_cans : ℕ
  final_rooms : ℕ

/-- Calculates the number of cans used for painting given a scenario --/
def cans_used (scenario : PaintingScenario) : ℕ :=
  sorry

/-- Theorem stating the number of cans used in Paula's specific scenario --/
theorem paula_cans_used :
  let scenario : PaintingScenario := {
    initial_rooms := 45,
    lost_cans := 5,
    final_rooms := 35
  }
  cans_used scenario = 18 :=
by sorry

end paula_cans_used_l852_85295


namespace pascal_triangle_100th_row_10th_number_l852_85204

theorem pascal_triangle_100th_row_10th_number :
  let n : ℕ := 99  -- row number (100 numbers in the row, so n + 1 = 100)
  let k : ℕ := 9   -- 10th number (0-indexed)
  (n.choose k) = (Nat.choose 99 9) := by
  sorry

end pascal_triangle_100th_row_10th_number_l852_85204


namespace julie_income_calculation_l852_85277

/-- Calculates Julie's net monthly income based on given conditions --/
def julies_net_monthly_income (
  starting_pay : ℝ)
  (experience_bonus : ℝ)
  (years_experience : ℕ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (biweekly_bonus : ℝ)
  (tax_rate : ℝ)
  (insurance_premium : ℝ)
  (missed_days : ℕ) : ℝ :=
  sorry

/-- Theorem stating that Julie's net monthly income is $963.20 --/
theorem julie_income_calculation :
  julies_net_monthly_income 5 0.5 3 8 6 50 0.12 40 1 = 963.20 :=
by sorry

end julie_income_calculation_l852_85277


namespace micah_typing_speed_l852_85216

/-- The number of words Isaiah can type per minute. -/
def isaiah_words_per_minute : ℕ := 40

/-- The number of minutes in an hour. -/
def minutes_per_hour : ℕ := 60

/-- The difference in words typed per hour between Isaiah and Micah. -/
def word_difference_per_hour : ℕ := 1200

/-- The number of words Micah can type per minute. -/
def micah_words_per_minute : ℕ := 20

/-- Theorem stating that Micah can type 20 words per minute given the conditions. -/
theorem micah_typing_speed : micah_words_per_minute = 20 := by sorry

end micah_typing_speed_l852_85216


namespace power_of_two_equality_l852_85240

theorem power_of_two_equality (x : ℕ) : (1 / 8 : ℝ) * 2^33 = 2^x → x = 30 := by
  sorry

end power_of_two_equality_l852_85240


namespace fixed_point_of_linear_function_l852_85258

theorem fixed_point_of_linear_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a * x + 2
  f (-1) = 2 := by
  sorry

end fixed_point_of_linear_function_l852_85258


namespace complex_simplification_l852_85208

theorem complex_simplification :
  (4 - 3 * Complex.I) * 2 - (6 - 3 * Complex.I) = 2 - 3 * Complex.I :=
by
  sorry

end complex_simplification_l852_85208


namespace eighteen_percent_of_500_is_90_l852_85219

theorem eighteen_percent_of_500_is_90 : 
  (18 : ℚ) / 100 * 500 = 90 := by
  sorry

end eighteen_percent_of_500_is_90_l852_85219


namespace scale_E_accurate_l852_85249

/-- Represents the weight measured by a scale -/
structure Scale where
  weight : ℝ

/-- Represents a set of five scales used in a health check center -/
structure HealthCheckScales where
  A : Scale
  B : Scale
  C : Scale
  D : Scale
  E : Scale

/-- The conditions of the health check scales problem -/
def ScaleConditions (s : HealthCheckScales) : Prop :=
  s.C.weight = s.B.weight - 0.3 ∧
  s.D.weight = s.C.weight - 0.1 ∧
  s.E.weight = s.A.weight - 0.1 ∧
  s.C.weight = s.E.weight - 0.1

/-- The average weight of all scales is accurate -/
def AverageWeightAccurate (s : HealthCheckScales) (actualWeight : ℝ) : Prop :=
  (s.A.weight + s.B.weight + s.C.weight + s.D.weight + s.E.weight) / 5 = actualWeight

/-- Theorem stating that scale E is accurate given the conditions -/
theorem scale_E_accurate (s : HealthCheckScales) (actualWeight : ℝ)
  (h1 : ScaleConditions s)
  (h2 : AverageWeightAccurate s actualWeight) :
  s.E.weight = actualWeight :=
sorry

end scale_E_accurate_l852_85249


namespace lap_time_improvement_l852_85297

def initial_laps : ℕ := 10
def initial_time : ℕ := 25
def current_laps : ℕ := 12
def current_time : ℕ := 24

def improvement : ℚ := 1/2

theorem lap_time_improvement : 
  (initial_time : ℚ) / initial_laps - (current_time : ℚ) / current_laps = improvement :=
sorry

end lap_time_improvement_l852_85297


namespace circle_max_area_center_l852_85281

/-- A circle with equation x^2 + y^2 + kx + 2y + k^2 = 0 -/
def Circle (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + k * p.1 + 2 * p.2 + k^2 = 0}

/-- The center of a circle -/
def center (c : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- The area of a circle -/
def area (c : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: The center of the circle is (0, -1) when its area is maximized -/
theorem circle_max_area_center (k : ℝ) :
  (∀ k' : ℝ, area (Circle k') ≤ area (Circle k)) →
  center (Circle k) = (0, -1) := by sorry

end circle_max_area_center_l852_85281


namespace pool_perimeter_is_20_l852_85293

/-- Represents the dimensions and properties of a garden with a rectangular pool -/
structure GardenPool where
  garden_length : ℝ
  garden_width : ℝ
  pool_area : ℝ
  walkway_width : ℝ

/-- Calculates the perimeter of the pool given the garden dimensions and pool properties -/
def pool_perimeter (g : GardenPool) : ℝ :=
  2 * ((g.garden_length - 2 * g.walkway_width) + (g.garden_width - 2 * g.walkway_width))

/-- Theorem stating that the perimeter of the pool is 20 meters under the given conditions -/
theorem pool_perimeter_is_20 (g : GardenPool) 
    (h1 : g.garden_length = 8)
    (h2 : g.garden_width = 6)
    (h3 : g.pool_area = 24)
    (h4 : (g.garden_length - 2 * g.walkway_width) * (g.garden_width - 2 * g.walkway_width) = g.pool_area) :
  pool_perimeter g = 20 := by
  sorry

#check pool_perimeter_is_20

end pool_perimeter_is_20_l852_85293


namespace red_paint_amount_l852_85200

/-- Given a paint mixture with a ratio of red:green:white as 4:3:5,
    and using 15 quarts of white paint, prove that the amount of
    red paint required is 12 quarts. -/
theorem red_paint_amount (red green white : ℚ) : 
  red / white = 4 / 5 →
  green / white = 3 / 5 →
  white = 15 →
  red = 12 := by
  sorry

end red_paint_amount_l852_85200


namespace calculation_proof_l852_85265

theorem calculation_proof : 
  (3 + 1 / 117) * (4 + 1 / 119) - (1 + 116 / 117) * (5 + 118 / 119) - 5 / 119 = 10 / 117 := by
  sorry

end calculation_proof_l852_85265


namespace find_divisor_l852_85242

theorem find_divisor : ∃ (D : ℕ), 
  (23 = 5 * D + 3) ∧ 
  (∃ (N : ℕ), N = 7 * D + 5) ∧ 
  D = 4 := by
  sorry

end find_divisor_l852_85242


namespace senate_committee_arrangements_l852_85217

/-- The number of ways to arrange n distinguishable people around a circular table -/
def circularArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of politicians on the Senate committee -/
def numPoliticians : ℕ := 12

theorem senate_committee_arrangements :
  circularArrangements numPoliticians = 39916800 := by
  sorry

end senate_committee_arrangements_l852_85217


namespace equation_solution_l852_85218

theorem equation_solution : 
  ∃! y : ℚ, (5 * y - 2) / (6 * y - 6) = 3 / 4 ∧ y = -5 := by
  sorry

end equation_solution_l852_85218


namespace sum_of_roots_quadratic_l852_85207

theorem sum_of_roots_quadratic (x : ℝ) : 
  (x^2 - 5*x + 6 = -4) → (∃ y : ℝ, y^2 - 5*y + 6 = -4 ∧ x + y = 5) :=
by sorry

end sum_of_roots_quadratic_l852_85207


namespace same_color_probability_l852_85253

/-- The probability of drawing two balls of the same color from a box containing
    4 white balls and 2 black balls when drawing two balls at once. -/
theorem same_color_probability (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ)
    (h_total : total_balls = white_balls + black_balls)
    (h_white : white_balls = 4)
    (h_black : black_balls = 2) :
    (Nat.choose white_balls 2 + Nat.choose black_balls 2) / Nat.choose total_balls 2 = 7 / 15 :=
  sorry

end same_color_probability_l852_85253


namespace water_for_chickens_l852_85236

/-- Calculates the amount of water needed for chickens given the total water needed and the water needed for pigs and horses. -/
theorem water_for_chickens 
  (num_pigs : ℕ) 
  (num_horses : ℕ) 
  (water_per_pig : ℕ) 
  (total_water : ℕ) 
  (h1 : num_pigs = 8)
  (h2 : num_horses = 10)
  (h3 : water_per_pig = 3)
  (h4 : total_water = 114) :
  total_water - (num_pigs * water_per_pig + num_horses * (2 * water_per_pig)) = 30 := by
  sorry

#check water_for_chickens

end water_for_chickens_l852_85236


namespace square_eq_nine_solutions_l852_85239

theorem square_eq_nine_solutions (x : ℝ) : (x + 1)^2 = 9 ↔ x = 2 ∨ x = -4 := by
  sorry

end square_eq_nine_solutions_l852_85239


namespace problem_1_problem_2_l852_85280

-- Problem 1
theorem problem_1 (a b : ℝ) :
  a^2 * (2*a*b - 1) + (a - 3*b) * (a + b) = 2*a^3*b - 2*a*b - 3*b^2 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) :
  (2*x - 3)^2 - (x + 2)^2 = 3*x^2 - 16*x + 5 := by
  sorry

end problem_1_problem_2_l852_85280


namespace probability_rain_all_three_days_l852_85211

theorem probability_rain_all_three_days 
  (prob_friday : ℝ) 
  (prob_saturday : ℝ) 
  (prob_sunday : ℝ) 
  (h1 : prob_friday = 0.4)
  (h2 : prob_saturday = 0.5)
  (h3 : prob_sunday = 0.3)
  (h4 : 0 ≤ prob_friday ∧ prob_friday ≤ 1)
  (h5 : 0 ≤ prob_saturday ∧ prob_saturday ≤ 1)
  (h6 : 0 ≤ prob_sunday ∧ prob_sunday ≤ 1) :
  prob_friday * prob_saturday * prob_sunday = 0.06 := by
  sorry

end probability_rain_all_three_days_l852_85211


namespace subtraction_of_negative_l852_85282

theorem subtraction_of_negative : 3 - (-3) = 6 := by
  sorry

end subtraction_of_negative_l852_85282


namespace smallest_positive_translation_l852_85292

theorem smallest_positive_translation (f : ℝ → ℝ) (φ : ℝ) : 
  (f = λ x => Real.sin (2 * x) + Real.cos (2 * x)) →
  (∀ x, f (x - φ) = f (φ - x)) →
  (∀ ψ, 0 < ψ ∧ ψ < φ → ¬(∀ x, f (x - ψ) = f (ψ - x))) →
  φ = 3 * Real.pi / 8 := by
  sorry

end smallest_positive_translation_l852_85292


namespace license_plate_count_l852_85246

/-- Represents the number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- Represents the number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- Represents the number of digits in the license plate -/
def digit_count : ℕ := 5

/-- Represents the number of letters in the license plate -/
def letter_count : ℕ := 3

/-- Represents the total number of characters in the license plate -/
def total_chars : ℕ := digit_count + letter_count

/-- Calculates the number of ways to arrange the letter block within the license plate -/
def letter_block_positions : ℕ := total_chars - letter_count + 1

/-- Calculates the number of valid letter combinations (at least one 'A') -/
def valid_letter_combinations : ℕ := 3 * num_letters^2

/-- The main theorem stating the total number of distinct license plates -/
theorem license_plate_count : 
  letter_block_positions * num_digits^digit_count * valid_letter_combinations = 1216800000 := by
  sorry


end license_plate_count_l852_85246


namespace cone_max_cross_section_area_l852_85221

/-- Given a cone with lateral surface formed by a sector of radius 1 and central angle 3/2 π,
    the maximum area of a cross-section passing through the vertex is 1/2. -/
theorem cone_max_cross_section_area (r : ℝ) (θ : ℝ) (h : ℝ) : 
  r = 1 → θ = 3/2 * Real.pi → h = Real.sqrt (r^2 - (r * θ / (2 * Real.pi))^2) → 
  (1/2 : ℝ) * (r * θ / (2 * Real.pi)) * h ≤ 1/2 := by
  sorry

#check cone_max_cross_section_area

end cone_max_cross_section_area_l852_85221


namespace fraction_value_l852_85287

theorem fraction_value (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : y = x / (x + 1)) :
  (x - y + 4 * x * y) / (x * y) = 5 := by
  sorry

end fraction_value_l852_85287


namespace power_equation_solution_l852_85229

theorem power_equation_solution : ∃ n : ℤ, (5 : ℝ) ^ (4 * n) = (1 / 5 : ℝ) ^ (n - 30) ∧ n = 6 := by
  sorry

end power_equation_solution_l852_85229


namespace diameter_is_chord_l852_85237

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a chord
def isChord (c : Circle) (p q : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
  (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 = c.radius^2

-- Define a diameter
def isDiameter (c : Circle) (p q : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
  (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 = c.radius^2 ∧
  (p.1 - q.1)^2 + (p.2 - q.2)^2 = 4 * c.radius^2

-- Theorem: A diameter is a chord
theorem diameter_is_chord (c : Circle) (p q : ℝ × ℝ) :
  isDiameter c p q → isChord c p q :=
by
  sorry


end diameter_is_chord_l852_85237


namespace equal_negative_exponents_l852_85278

theorem equal_negative_exponents : -2^3 = (-2)^3 ∧ 
  -3^2 ≠ -2^3 ∧ 
  (-3 * 2)^2 ≠ -3 * 2^2 ∧ 
  -3^2 ≠ (-3)^2 :=
by sorry

end equal_negative_exponents_l852_85278


namespace exponent_multiplication_l852_85296

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end exponent_multiplication_l852_85296


namespace range_of_a_l852_85252

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -1 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}

-- State the theorem
theorem range_of_a (a : ℝ) : A ∪ B a = B a → -1 < a ∧ a ≤ 1 := by
  sorry

end range_of_a_l852_85252


namespace star_value_l852_85212

-- Define the sequence type
def Sequence := Fin 12 → ℕ

-- Define the property that the sum of any four adjacent numbers is 11
def SumProperty (s : Sequence) : Prop :=
  ∀ i : Fin 9, s i + s (i + 1) + s (i + 2) + s (i + 3) = 11

-- Define the repeating pattern property
def PatternProperty (s : Sequence) : Prop :=
  ∀ i : Fin 3, 
    s (4 * i) = 2 ∧ 
    s (4 * i + 1) = 0 ∧ 
    s (4 * i + 2) = 1

-- Main theorem
theorem star_value (s : Sequence) 
  (h1 : SumProperty s) 
  (h2 : PatternProperty s) : 
  ∀ i : Fin 3, s (4 * i + 3) = 8 := by
  sorry

end star_value_l852_85212


namespace chord_length_l852_85247

/-- The circle passing through the intersection points of y = x, y = 2x, and y = 15 - 0.5x -/
def special_circle : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ (x - 5)^2 + (y - 5)^2 = 50}

/-- The line x + y = 16 -/
def line : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ x + y = 16}

/-- The chord formed by the intersection of the special circle and the line -/
def chord : Set (ℝ × ℝ) :=
  special_circle ∩ line

theorem chord_length : 
  ∃ (p q : ℝ × ℝ), p ∈ chord ∧ q ∈ chord ∧ p ≠ q ∧ 
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 8 * Real.sqrt 2 :=
sorry

end chord_length_l852_85247


namespace january_first_is_monday_l852_85279

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month with its properties -/
structure Month where
  days : Nat
  firstDay : DayOfWeek
  mondayCount : Nat
  thursdayCount : Nat

/-- Theorem stating that a month with 31 days, 5 Mondays, and 4 Thursdays must start on a Monday -/
theorem january_first_is_monday (m : Month) :
  m.days = 31 ∧ m.mondayCount = 5 ∧ m.thursdayCount = 4 →
  m.firstDay = DayOfWeek.Monday := by
  sorry


end january_first_is_monday_l852_85279


namespace geometric_sequence_q_value_l852_85234

/-- A geometric sequence {a_n} -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_q_value
  (a : ℕ → ℝ)
  (h_monotone : ∀ n : ℕ, a n ≤ a (n + 1))
  (h_geometric : geometric_sequence a)
  (h_sum : a 3 + a 7 = 5)
  (h_product : a 6 * a 4 = 6) :
  ∃ q : ℝ, q > 1 ∧ q^4 = 3/2 :=
sorry

end geometric_sequence_q_value_l852_85234


namespace certain_number_value_l852_85266

theorem certain_number_value : ∃ x : ℝ, 
  (x + 40 + 60) / 3 = (10 + 70 + 13) / 3 + 9 → x = 20 := by
  sorry

end certain_number_value_l852_85266


namespace root_product_theorem_l852_85269

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^5 + 3*x^2 + 1

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 5

-- State the theorem
theorem root_product_theorem (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (hroots : f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0 ∧ f x₅ = 0) :
  g x₁ * g x₂ * g x₃ * g x₄ * g x₅ = 131 := by
  sorry

end root_product_theorem_l852_85269


namespace real_part_of_z_l852_85255

theorem real_part_of_z (z : ℂ) (h : (z + 1).re = 0) : z.re = -1 := by
  sorry

end real_part_of_z_l852_85255


namespace smoothie_combinations_l852_85230

theorem smoothie_combinations (n_flavors : ℕ) (n_supplements : ℕ) : 
  n_flavors = 5 → n_supplements = 8 → n_flavors * (n_supplements.choose 3) = 280 := by
  sorry

end smoothie_combinations_l852_85230


namespace power_equation_solution_l852_85276

theorem power_equation_solution : ∃ m : ℤ, 2^4 - 3 = 5^2 + m ∧ m = -12 := by
  sorry

end power_equation_solution_l852_85276


namespace cross_section_area_unit_cube_l852_85290

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  edge_length : ℝ

/-- Theorem: Area of cross-section in a unit cube -/
theorem cross_section_area_unit_cube (c : Cube) (X Y Z : Point3D) :
  c.edge_length = 1 →
  X = ⟨1/2, 1/2, 0⟩ →
  Y = ⟨1, 1/2, 1/2⟩ →
  Z = ⟨3/4, 3/4, 3/4⟩ →
  let sphere_radius := Real.sqrt 3 / 2
  let plane_distance := Real.sqrt ((1/4)^2 + (1/4)^2 + (3/4)^2)
  let cross_section_radius := Real.sqrt (sphere_radius^2 - plane_distance^2)
  let cross_section_area := π * cross_section_radius^2
  cross_section_area = 5 * π / 8 := by
  sorry

end cross_section_area_unit_cube_l852_85290


namespace cubic_roots_relation_l852_85274

theorem cubic_roots_relation (a b c : ℂ) : 
  (a^3 - 3*a^2 + 5*a - 8 = 0) → 
  (b^3 - 3*b^2 + 5*b - 8 = 0) → 
  (c^3 - 3*c^2 + 5*c - 8 = 0) → 
  (∃ r s : ℂ, (a-b)^3 + r*(a-b)^2 + s*(a-b) + 243 = 0 ∧ 
               (b-c)^3 + r*(b-c)^2 + s*(b-c) + 243 = 0 ∧ 
               (c-a)^3 + r*(c-a)^2 + s*(c-a) + 243 = 0) :=
by sorry

end cubic_roots_relation_l852_85274


namespace sum_of_coefficients_is_negative_27_l852_85273

def polynomial (x : ℝ) : ℝ :=
  -3 * (x^7 - 2*x^6 + x^4 - 3*x^2 + 6) + 6 * (x^3 - 4*x + 1) - 2 * (x^5 - 5*x + 7)

theorem sum_of_coefficients_is_negative_27 : 
  (polynomial 1) = -27 := by sorry

end sum_of_coefficients_is_negative_27_l852_85273


namespace solution_equation1_solution_equation2_l852_85285

-- Define the equations
def equation1 (x : ℝ) : Prop := 3 * x + 3 = 7 - x
def equation2 (x : ℝ) : Prop := (1/2) * x - 6 = (3/4) * x

-- Theorem for equation 1
theorem solution_equation1 : ∃! x : ℝ, equation1 x ∧ x = 1 := by sorry

-- Theorem for equation 2
theorem solution_equation2 : ∃! x : ℝ, equation2 x ∧ x = -24 := by sorry

end solution_equation1_solution_equation2_l852_85285


namespace f_2_eq_0_l852_85215

def f (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

theorem f_2_eq_0 :
  f 2 = horner_eval [1, -12, 60, -160, 240, -192, 64] 2 ∧
  horner_eval [1, -12, 60, -160, 240, -192, 64] 2 = 0 :=
by sorry

end f_2_eq_0_l852_85215


namespace g_four_equals_thirteen_l852_85238

-- Define the function g
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^6 + b * x^4 + c * x^2 + 7

-- State the theorem
theorem g_four_equals_thirteen 
  (a b c : ℝ) 
  (h : g a b c (-4) = 13) : 
  g a b c 4 = 13 := by
sorry

end g_four_equals_thirteen_l852_85238


namespace smallest_base_for_100_in_three_digits_l852_85270

theorem smallest_base_for_100_in_three_digits : ∃ (b : ℕ), b = 5 ∧ 
  (∀ (n : ℕ), n^2 ≤ 100 ∧ 100 < n^3 → b ≤ n) := by
  sorry

end smallest_base_for_100_in_three_digits_l852_85270


namespace min_value_on_transformed_curve_l852_85251

-- Define the original curve C
def curve_C (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line l
def line_l (x y t : ℝ) : Prop := x = 1 + t/2 ∧ y = 2 + (Real.sqrt 3)/2 * t

-- Define the transformation
def transform (x y x' y' : ℝ) : Prop := x' = 3*x ∧ y' = y

-- Define the transformed curve C'
def curve_C' (x y : ℝ) : Prop := x^2/9 + y^2 = 1

-- State the theorem
theorem min_value_on_transformed_curve :
  ∀ (x y : ℝ), curve_C' x y → (x + 2 * Real.sqrt 3 * y ≥ -Real.sqrt 21) :=
by sorry

end min_value_on_transformed_curve_l852_85251


namespace linear_equation_solution_l852_85245

theorem linear_equation_solution (m : ℝ) : 
  (∃ x : ℝ, 2 * x + m = 5 ∧ x = 1) → m = 3 := by
  sorry

end linear_equation_solution_l852_85245


namespace shekar_science_score_l852_85272

def average_marks : ℝ := 77
def num_subjects : ℕ := 5
def math_score : ℝ := 76
def social_studies_score : ℝ := 82
def english_score : ℝ := 67
def biology_score : ℝ := 95

theorem shekar_science_score :
  ∃ (science_score : ℝ),
    (math_score + social_studies_score + english_score + biology_score + science_score) / num_subjects = average_marks ∧
    science_score = 65 := by
  sorry

end shekar_science_score_l852_85272


namespace distinct_values_theorem_l852_85223

/-- The number of distinct values expressible as ip + jq -/
def distinct_values (n p q : ℕ) : ℕ :=
  if p = q ∧ p = 1 then
    n + 1
  else if p > q ∧ n < p then
    (n + 1) * (n + 2) / 2
  else if p > q ∧ n ≥ p then
    p * (2 * n - p + 3) / 2
  else
    0  -- This case is not specified in the problem, but needed for completeness

/-- Theorem stating the number of distinct values expressible as ip + jq -/
theorem distinct_values_theorem (n p q : ℕ) (h_coprime : Nat.Coprime p q) :
  distinct_values n p q =
    if p = q ∧ p = 1 then
      n + 1
    else if p > q ∧ n < p then
      (n + 1) * (n + 2) / 2
    else if p > q ∧ n ≥ p then
      p * (2 * n - p + 3) / 2
    else
      0 := by sorry

end distinct_values_theorem_l852_85223


namespace arithmetic_sequence_sum_l852_85202

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 4 = 12 → a 1 + a 7 = 24 := by
  sorry

end arithmetic_sequence_sum_l852_85202


namespace arithmetic_mean_problem_l852_85264

theorem arithmetic_mean_problem (a b c : ℝ) 
  (h1 : (a + b) / 2 = 80) 
  (h2 : (b + c) / 2 = 180) : 
  a - c = -200 := by
sorry

end arithmetic_mean_problem_l852_85264


namespace rectangular_prism_face_fits_in_rectangle_l852_85250

/-- Represents a rectangular prism with dimensions a ≤ b ≤ c -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : 0 < a
  h2 : a ≤ b
  h3 : b ≤ c

/-- Represents a rectangle with dimensions d₁ ≤ d₂ -/
structure Rectangle where
  d1 : ℝ
  d2 : ℝ
  h : d1 ≤ d2

/-- Theorem: Given a rectangular prism and a rectangle that can contain
    the prism's hexagonal cross-section, prove that the rectangle can
    contain one face of the prism -/
theorem rectangular_prism_face_fits_in_rectangle
  (prism : RectangularPrism) (rect : Rectangle)
  (hex_fits : ∃ (h : ℝ), h > 0 ∧ h^2 + rect.d1^2 ≥ prism.b^2 + prism.c^2) :
  min rect.d1 rect.d2 ≥ prism.a ∧ max rect.d1 rect.d2 ≥ prism.b :=
by sorry

end rectangular_prism_face_fits_in_rectangle_l852_85250


namespace initial_salty_cookies_count_l852_85283

/-- The number of salty cookies Paco had initially -/
def initial_salty_cookies : ℕ := sorry

/-- The number of salty cookies Paco ate -/
def eaten_salty_cookies : ℕ := 3

/-- The number of salty cookies Paco had left after eating -/
def remaining_salty_cookies : ℕ := 3

/-- Theorem stating that the initial number of salty cookies is 6 -/
theorem initial_salty_cookies_count : initial_salty_cookies = 6 := by sorry

end initial_salty_cookies_count_l852_85283


namespace f_equals_F_l852_85210

/-- The function f(x) = 3x^4 - x^3 -/
def f (x : ℝ) : ℝ := 3 * x^4 - x^3

/-- The function F(x) = x(3x^3 - 1) -/
def F (x : ℝ) : ℝ := x * (3 * x^3 - 1)

/-- Theorem stating that f and F are the same function -/
theorem f_equals_F : f = F := by sorry

end f_equals_F_l852_85210


namespace right_triangle_third_side_l852_85214

theorem right_triangle_third_side 
  (a b c : ℝ) 
  (ha : a = 10) 
  (hb : b = 24) 
  (hright : a^2 + c^2 = b^2) : 
  c = 2 * Real.sqrt 119 := by
sorry

end right_triangle_third_side_l852_85214


namespace average_ducks_is_35_l852_85262

/-- The average number of ducks bought by three students -/
def averageDucks (adelaide ephraim kolton : ℕ) : ℚ :=
  (adelaide + ephraim + kolton : ℚ) / 3

/-- Theorem: The average number of ducks bought is 35 -/
theorem average_ducks_is_35 :
  let adelaide := 30
  let ephraim := adelaide / 2
  let kolton := ephraim + 45
  averageDucks adelaide ephraim kolton = 35 := by
sorry

end average_ducks_is_35_l852_85262


namespace sum_interior_angles_polygon_l852_85263

theorem sum_interior_angles_polygon (n : ℕ) (h : n ≥ 3) :
  (360 / 30 : ℕ) = n → (n - 2) * 180 = 1800 := by
  sorry

end sum_interior_angles_polygon_l852_85263


namespace ramsey_three_three_three_l852_85271

/-- A coloring of edges in a complete graph with three colors -/
def Coloring (n : ℕ) := Fin n → Fin n → Fin 3

/-- A monochromatic triangle in a coloring -/
def HasMonochromaticTriangle (n : ℕ) (c : Coloring n) : Prop :=
  ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    c i j = c j k ∧ c j k = c i k

/-- The Ramsey theorem R(3,3,3) ≤ 17 -/
theorem ramsey_three_three_three :
  ∀ (c : Coloring 17), HasMonochromaticTriangle 17 c :=
sorry

end ramsey_three_three_three_l852_85271


namespace simplify_expression_l852_85294

theorem simplify_expression (x : ℝ) : x + 3 - 4*x - 5 + 6*x + 7 - 8*x - 9 = -5*x - 4 := by
  sorry

end simplify_expression_l852_85294


namespace polynomial_property_l852_85206

def P (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem polynomial_property (a b c : ℝ) :
  P a b c 0 = 5 →
  (-a/3 : ℝ) = -c →
  (-a/3 : ℝ) = 1 + a + b + c →
  b = -26 := by sorry

end polynomial_property_l852_85206


namespace sum_of_fractions_l852_85257

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end sum_of_fractions_l852_85257


namespace smallest_winning_number_sum_of_digits_56_l852_85226

def bernardo_win (N : ℕ) : Prop :=
  N ≤ 999 ∧
  3 * N < 1000 ∧
  3 * N + 100 < 1000 ∧
  9 * N + 300 < 1000 ∧
  9 * N + 400 < 1000 ∧
  27 * N + 1200 ≥ 1000

theorem smallest_winning_number :
  ∀ n : ℕ, n < 56 → ¬(bernardo_win n) ∧ bernardo_win 56 :=
sorry

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_56 :
  sum_of_digits 56 = 11 :=
sorry

end smallest_winning_number_sum_of_digits_56_l852_85226


namespace frontal_view_correct_l852_85205

/-- Represents a column of stacked cubes -/
def Column := List Nat

/-- Calculates the maximum height of a column -/
def maxHeight (col : Column) : Nat :=
  col.foldl max 0

/-- Represents the arrangement of cube stacks -/
structure CubeArrangement where
  col1 : Column
  col2 : Column
  col3 : Column

/-- Calculates the frontal view heights of a cube arrangement -/
def frontalView (arr : CubeArrangement) : List Nat :=
  [maxHeight arr.col1, maxHeight arr.col2, maxHeight arr.col3]

/-- The specific cube arrangement described in the problem -/
def problemArrangement : CubeArrangement :=
  { col1 := [4, 2]
    col2 := [3, 0, 3]
    col3 := [1, 5] }

theorem frontal_view_correct :
  frontalView problemArrangement = [4, 3, 5] := by sorry

end frontal_view_correct_l852_85205


namespace fraction_division_l852_85224

theorem fraction_division (x y z : ℚ) (hx : x = 3) (hy : y = 4) (hz : z = 5) :
  (z / y) / (z / x) = 3 / 4 := by
  sorry

end fraction_division_l852_85224


namespace increase_amount_l852_85244

theorem increase_amount (x : ℝ) (amount : ℝ) (h : 15 * x + amount = 14) :
  amount = 14 - 14/15 := by
  sorry

end increase_amount_l852_85244


namespace mode_is_80_l852_85298

/-- Represents the frequency of each score in the test results -/
def score_frequency : List (Nat × Nat) := [
  (61, 1), (61, 1), (62, 1),
  (75, 1), (77, 1),
  (80, 3), (81, 1), (83, 2),
  (92, 2), (94, 1), (96, 1), (97, 2),
  (105, 2), (109, 1),
  (110, 2)
]

/-- The maximum score possible on the test -/
def max_score : Nat := 120

/-- Definition of the mode: the score that appears most frequently -/
def is_mode (score : Nat) (frequencies : List (Nat × Nat)) : Prop :=
  ∀ other_score, other_score ≠ score →
    (frequencies.filter (λ pair => pair.1 = score)).length ≥
    (frequencies.filter (λ pair => pair.1 = other_score)).length

/-- Theorem stating that 80 is the mode of the scores -/
theorem mode_is_80 : is_mode 80 score_frequency := by
  sorry

end mode_is_80_l852_85298


namespace complement_A_inter_B_wrt_U_l852_85254

def U : Set ℤ := {x | -1 ≤ x ∧ x ≤ 2}
def A : Set ℤ := {x | x^2 - x = 0}
def B : Set ℤ := {x | -1 < x ∧ x < 2}

theorem complement_A_inter_B_wrt_U : (U \ (A ∩ B)) = {-1, 2} := by sorry

end complement_A_inter_B_wrt_U_l852_85254


namespace square_gt_iff_abs_gt_l852_85222

theorem square_gt_iff_abs_gt (a b : ℝ) : a^2 > b^2 ↔ |a| > |b| := by sorry

end square_gt_iff_abs_gt_l852_85222


namespace happy_valley_kennel_arrangements_l852_85248

/-- The number of ways to arrange animals in cages -/
def arrange_animals (chickens dogs cats rabbits : ℕ) : ℕ :=
  (Nat.factorial 4) * 
  (Nat.factorial chickens) * 
  (Nat.factorial dogs) * 
  (Nat.factorial cats) * 
  (Nat.factorial rabbits)

/-- Theorem stating the number of arrangements for the given problem -/
theorem happy_valley_kennel_arrangements :
  arrange_animals 3 3 5 2 = 207360 := by
  sorry

end happy_valley_kennel_arrangements_l852_85248


namespace smallest_digit_divisible_by_3_and_9_l852_85225

def is_divisible_by (n m : ℕ) : Prop := n % m = 0

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem smallest_digit_divisible_by_3_and_9 : 
  ∃ (d : ℕ), d < 10 ∧ 
    is_divisible_by (528000 + d * 100 + 74) 3 ∧ 
    is_divisible_by (528000 + d * 100 + 74) 9 ∧
    ∀ (d' : ℕ), d' < d → 
      ¬(is_divisible_by (528000 + d' * 100 + 74) 3 ∧ 
        is_divisible_by (528000 + d' * 100 + 74) 9) :=
by
  sorry

end smallest_digit_divisible_by_3_and_9_l852_85225


namespace m_range_l852_85220

-- Define propositions p and q
def p (x m : ℝ) : Prop := x^2 - 4*m*x + 3*m^2 < 0
def q (x : ℝ) : Prop := |x - 3| ≤ 1

-- Define the condition that q is sufficient but not necessary for p
def q_sufficient_not_necessary (m : ℝ) : Prop :=
  (∀ x, q x → p x m) ∧ (∃ x, p x m ∧ ¬q x)

-- Main theorem
theorem m_range (m : ℝ) (h1 : m > 0) (h2 : q_sufficient_not_necessary m) :
  m > 4/3 ∧ m < 2 :=
sorry

end m_range_l852_85220


namespace cost_of_300_candies_l852_85209

/-- The cost of a single candy in cents -/
def candy_cost : ℕ := 5

/-- The number of candies -/
def num_candies : ℕ := 300

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Theorem: The cost of 300 candies is 15 dollars -/
theorem cost_of_300_candies :
  (num_candies * candy_cost) / cents_per_dollar = 15 := by
  sorry

end cost_of_300_candies_l852_85209


namespace triangle_angle_measure_l852_85260

/-- Given a triangle ABC with side lengths a, b, and c satisfying (a+b+c)(b+c-a) = bc,
    prove that the measure of angle A is 120 degrees. -/
theorem triangle_angle_measure (a b c : ℝ) (h : (a + b + c) * (b + c - a) = b * c) :
  let A := Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))
  A = 2 * π / 3 := by
  sorry

end triangle_angle_measure_l852_85260


namespace expression_factorization_l852_85232

theorem expression_factorization (x : ℝ) : 
  (12 * x^5 + 33 * x^3 + 10) - (3 * x^5 - 4 * x^3 - 1) = x^3 * (9 * x^2 + 37) + 11 := by
  sorry

end expression_factorization_l852_85232


namespace trapezoid_area_l852_85259

-- Define the rectangle ABCD
structure Rectangle where
  width : ℝ
  height : ℝ
  area : ℝ
  area_eq : area = width * height

-- Define the trapezoid DEFG
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  height : ℝ

-- Define the problem setup
def problem_setup (rect : Rectangle) (trap : Trapezoid) : Prop :=
  rect.area = 108 ∧
  trap.base1 = rect.height / 2 ∧
  trap.base2 = rect.width / 2 ∧
  trap.height = rect.height / 2

-- Theorem to prove
theorem trapezoid_area 
  (rect : Rectangle) 
  (trap : Trapezoid) 
  (h : problem_setup rect trap) : 
  (trap.base1 + trap.base2) / 2 * trap.height = 27 := by
  sorry

end trapezoid_area_l852_85259


namespace segment_construction_l852_85233

theorem segment_construction (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let x := a + b
  4 * (c * (a * c).sqrt).sqrt * x = 4 * (c * (a * c).sqrt).sqrt * a + 4 * (c * (a * c).sqrt).sqrt * b :=
by sorry

end segment_construction_l852_85233


namespace rectangular_plot_area_l852_85241

/-- The area of a rectangular plot with length thrice its breadth and breadth of 11 meters is 363 square meters. -/
theorem rectangular_plot_area (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  breadth = 11 →
  length = 3 * breadth →
  area = length * breadth →
  area = 363 := by
  sorry

end rectangular_plot_area_l852_85241


namespace harold_catch_up_distance_l852_85213

/-- The distance from X to Y in miles -/
def total_distance : ℝ := 60

/-- Adrienne's walking speed in miles per hour -/
def adrienne_speed : ℝ := 3

/-- Harold's walking speed in miles per hour -/
def harold_speed : ℝ := adrienne_speed + 1

/-- Time difference between Adrienne's and Harold's start in hours -/
def time_difference : ℝ := 1

/-- The distance Harold will have traveled when he catches up to Adrienne -/
def catch_up_distance : ℝ := 12

theorem harold_catch_up_distance :
  ∃ (t : ℝ), t > 0 ∧ 
  adrienne_speed * (t + time_difference) = harold_speed * t ∧
  catch_up_distance = harold_speed * t :=
by sorry

end harold_catch_up_distance_l852_85213


namespace inequality_proof_l852_85227

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y + y * z + z * x = 1) : 
  (27 / 4) * (x + y) * (y + z) * (z + x) ≥ (Real.sqrt (x + y) + Real.sqrt (y + z) + Real.sqrt (z + x))^2 ∧ 
  (Real.sqrt (x + y) + Real.sqrt (y + z) + Real.sqrt (z + x))^2 ≥ 6 * Real.sqrt 3 :=
by sorry

end inequality_proof_l852_85227


namespace smallest_integer_solution_inequality_l852_85243

theorem smallest_integer_solution_inequality :
  ∀ (x : ℤ), (9 * x + 8) / 6 - x / 3 ≥ -1 → x ≥ -2 ∧
  ∃ (y : ℤ), y < -2 ∧ (9 * y + 8) / 6 - y / 3 < -1 :=
by sorry

end smallest_integer_solution_inequality_l852_85243


namespace suit_price_calculation_suit_price_proof_l852_85256

theorem suit_price_calculation (original_price : ℝ) 
  (increase_percentage : ℝ) (discount_percentage : ℝ) : ℝ :=
  let increased_price := original_price * (1 + increase_percentage)
  let final_price := increased_price * (1 - discount_percentage)
  final_price

theorem suit_price_proof :
  suit_price_calculation 200 0.3 0.3 = 182 := by
  sorry

end suit_price_calculation_suit_price_proof_l852_85256


namespace isosceles_triangle_removal_l852_85284

/-- Given a square with side length x, from which isosceles right triangles
    with leg length s are removed from each corner to form a rectangle
    with longer side 15 units, prove that the combined area of the four
    removed triangles is 225 square units. -/
theorem isosceles_triangle_removal (x s : ℝ) : 
  x > 0 →
  s > 0 →
  x - 2*s = 15 →
  (x - s)^2 + (x - s)^2 = x^2 →
  4 * (1/2 * s^2) = 225 := by
  sorry

end isosceles_triangle_removal_l852_85284


namespace simplify_expression_l852_85288

theorem simplify_expression (p : ℝ) : 
  ((7 * p - 3) - 3 * p * 2) * 2 + (5 - 2 / 2) * (8 * p - 12) = 34 * p - 54 := by
  sorry

end simplify_expression_l852_85288


namespace population_growth_rate_l852_85286

/-- Given a population increase of 160 persons in 40 minutes, 
    proves that the time taken for one person to be added is 15 seconds. -/
theorem population_growth_rate (persons : ℕ) (minutes : ℕ) (seconds_per_person : ℕ) : 
  persons = 160 ∧ minutes = 40 → seconds_per_person = 15 := by
  sorry

end population_growth_rate_l852_85286


namespace ribbon_calculation_l852_85261

/-- Represents the types of ribbons available --/
inductive RibbonType
  | A
  | B

/-- Represents the wrapping pattern for a gift --/
structure WrappingPattern where
  typeA : Nat
  typeB : Nat

/-- Calculates the number of ribbons needed for a given number of gifts and wrapping pattern --/
def ribbonsNeeded (numGifts : Nat) (pattern : WrappingPattern) : Nat × Nat :=
  (numGifts * pattern.typeA, numGifts * pattern.typeB)

theorem ribbon_calculation (tomSupplyA tomSupplyB : Nat) :
  let oddPattern : WrappingPattern := { typeA := 1, typeB := 2 }
  let evenPattern : WrappingPattern := { typeA := 2, typeB := 1 }
  let (oddA, oddB) := ribbonsNeeded 4 oddPattern
  let (evenA, evenB) := ribbonsNeeded 4 evenPattern
  let totalA := oddA + evenA
  let totalB := oddB + evenB
  tomSupplyA = 10 ∧ tomSupplyB = 12 →
  totalA - tomSupplyA = 2 ∧ totalB - tomSupplyB = 0 := by
  sorry


end ribbon_calculation_l852_85261


namespace sum_of_digits_of_gcd_l852_85203

-- Define the numbers given in the problem
def a : ℕ := 1305
def b : ℕ := 4665
def c : ℕ := 6905

-- Define the function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

-- State the theorem
theorem sum_of_digits_of_gcd : sum_of_digits (Nat.gcd (b - a) (Nat.gcd (c - b) (c - a))) = 4 := by
  sorry

end sum_of_digits_of_gcd_l852_85203


namespace max_product_constrained_max_product_achieved_l852_85299

theorem max_product_constrained (x y : ℕ+) (h : 7 * x + 4 * y = 140) :
  x * y ≤ 168 := by
  sorry

theorem max_product_achieved : ∃ (x y : ℕ+), 7 * x + 4 * y = 140 ∧ x * y = 168 := by
  sorry

end max_product_constrained_max_product_achieved_l852_85299


namespace combined_age_theorem_l852_85275

/-- The combined age of Jane and John after 12 years -/
def combined_age_after_12_years (justin_age : ℕ) (jessica_age_diff : ℕ) (james_age_diff : ℕ) (julia_age_diff : ℕ) (jane_age_diff : ℕ) (john_age_diff : ℕ) : ℕ :=
  let jessica_age := justin_age + jessica_age_diff
  let james_age := jessica_age + james_age_diff
  let jane_age := james_age + jane_age_diff
  let john_age := jane_age + john_age_diff
  (jane_age + 12) + (john_age + 12)

/-- Theorem stating the combined age of Jane and John after 12 years -/
theorem combined_age_theorem :
  combined_age_after_12_years 26 6 7 8 25 3 = 155 := by
  sorry

end combined_age_theorem_l852_85275


namespace existence_of_representation_l852_85291

theorem existence_of_representation (m : ℤ) :
  ∃ (a b k : ℤ), Odd a ∧ Odd b ∧ k ≥ 0 ∧ 2 * m = a^19 + b^99 + k * 2^1999 := by
  sorry

end existence_of_representation_l852_85291


namespace servings_count_l852_85268

/-- Represents the number of cups of cereal in a box -/
def total_cups : ℕ := 18

/-- Represents the number of cups per serving -/
def cups_per_serving : ℕ := 2

/-- Calculates the number of servings in a cereal box -/
def servings_in_box : ℕ := total_cups / cups_per_serving

/-- Proves that the number of servings in the cereal box is 9 -/
theorem servings_count : servings_in_box = 9 := by
  sorry

end servings_count_l852_85268
