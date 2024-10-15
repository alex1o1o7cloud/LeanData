import Mathlib

namespace NUMINAMATH_CALUDE_perfect_square_sum_l795_79587

theorem perfect_square_sum : 529 + 2 * 23 * 7 + 49 = 900 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l795_79587


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l795_79597

theorem rectangular_to_polar_conversion :
  ∀ (x y : ℝ),
  x = -3 ∧ y = 1 →
  ∃ (r θ : ℝ),
  r > 0 ∧
  0 ≤ θ ∧ θ < 2 * Real.pi ∧
  r = Real.sqrt 10 ∧
  θ = Real.pi - Real.arctan (1 / 3) ∧
  x = r * Real.cos θ ∧
  y = r * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l795_79597


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l795_79500

theorem quadratic_inequality_solution_range (d : ℝ) :
  d > 0 →
  (∃ x : ℝ, x^2 - 8*x + d < 0) ↔ d < 16 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l795_79500


namespace NUMINAMATH_CALUDE_daily_expense_reduction_l795_79528

theorem daily_expense_reduction (total_expense : ℕ) (original_days : ℕ) (extended_days : ℕ) :
  total_expense = 360 →
  original_days = 20 →
  extended_days = 24 →
  (total_expense / original_days) - (total_expense / extended_days) = 3 := by
  sorry

end NUMINAMATH_CALUDE_daily_expense_reduction_l795_79528


namespace NUMINAMATH_CALUDE_problem_solution_l795_79513

theorem problem_solution (y : ℝ) (h : y + Real.sqrt (y^2 - 4) + 1 / (y - Real.sqrt (y^2 - 4)) = 12) :
  y^2 + Real.sqrt (y^4 - 4) + 1 / (y^2 - Real.sqrt (y^4 - 4)) = 200 / 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l795_79513


namespace NUMINAMATH_CALUDE_min_distance_circle_to_line_l795_79557

/-- The minimum distance from any point on a circle to a line --/
theorem min_distance_circle_to_line :
  let circle := {(x, y) : ℝ × ℝ | (x - 1)^2 + (y - 1)^2 = 4}
  let line := {(x, y) : ℝ × ℝ | x - y + 4 = 0}
  (∃ (d : ℝ), d = 2 * Real.sqrt 2 - 2 ∧
    ∀ (p : ℝ × ℝ), p ∈ circle →
      ∀ (q : ℝ × ℝ), q ∈ line →
        d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_circle_to_line_l795_79557


namespace NUMINAMATH_CALUDE_expression_value_l795_79504

theorem expression_value : 4^3 - 2 * 4^2 + 2 * 4 - 1 = 39 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l795_79504


namespace NUMINAMATH_CALUDE_cube_surface_area_l795_79529

/-- Given three vertices of a cube, prove that its surface area is 150 -/
theorem cube_surface_area (A B C : ℝ × ℝ × ℝ) : 
  A = (5, 9, 6) → B = (5, 14, 6) → C = (5, 14, 11) → 
  (let surface_area := 6 * (B.2 - A.2)^2
   surface_area = 150) :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l795_79529


namespace NUMINAMATH_CALUDE_apple_bags_l795_79549

theorem apple_bags (A B C : ℕ) 
  (h1 : A + B + C = 24) 
  (h2 : A + B = 11) 
  (h3 : B + C = 18) : 
  A + C = 19 := by
  sorry

end NUMINAMATH_CALUDE_apple_bags_l795_79549


namespace NUMINAMATH_CALUDE_count_hens_and_cows_l795_79554

theorem count_hens_and_cows (total_animals : ℕ) (total_feet : ℕ) (hen_feet : ℕ) (cow_feet : ℕ) : 
  total_animals = 44 → 
  total_feet = 128 → 
  hen_feet = 2 → 
  cow_feet = 4 → 
  ∃ (hens cows : ℕ), 
    hens + cows = total_animals ∧ 
    hen_feet * hens + cow_feet * cows = total_feet ∧ 
    hens = 24 := by
  sorry

end NUMINAMATH_CALUDE_count_hens_and_cows_l795_79554


namespace NUMINAMATH_CALUDE_marcus_car_mpg_l795_79510

/-- Represents a car with its mileage and fuel efficiency characteristics -/
structure Car where
  initial_mileage : ℕ
  final_mileage : ℕ
  tank_capacity : ℕ
  num_fills : ℕ

/-- Calculates the miles per gallon for a given car -/
def miles_per_gallon (c : Car) : ℚ :=
  (c.final_mileage - c.initial_mileage : ℚ) / (c.tank_capacity * c.num_fills : ℚ)

/-- Theorem stating that Marcus's car gets 30 miles per gallon -/
theorem marcus_car_mpg :
  let marcus_car : Car := {
    initial_mileage := 1728,
    final_mileage := 2928,
    tank_capacity := 20,
    num_fills := 2
  }
  miles_per_gallon marcus_car = 30 := by
  sorry

end NUMINAMATH_CALUDE_marcus_car_mpg_l795_79510


namespace NUMINAMATH_CALUDE_problem_solution_l795_79586

theorem problem_solution (a b c : ℝ) 
  (h1 : a + 2*b + 3*c = 12) 
  (h2 : a^2 + b^2 + c^2 = a*b + a*c + b*c) : 
  a + b^2 + c^3 = 14 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l795_79586


namespace NUMINAMATH_CALUDE_cubic_function_extrema_condition_l795_79511

/-- Given a cubic function f(x) = x³ - 3x² + ax - b that has both a maximum and a minimum value,
    prove that the parameter a must be less than 3. -/
theorem cubic_function_extrema_condition (a b : ℝ) : 
  (∃ (x_min x_max : ℝ), ∀ x : ℝ, 
    x^3 - 3*x^2 + a*x - b ≤ x_max^3 - 3*x_max^2 + a*x_max - b ∧ 
    x^3 - 3*x^2 + a*x - b ≥ x_min^3 - 3*x_min^2 + a*x_min - b) →
  a < 3 :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_condition_l795_79511


namespace NUMINAMATH_CALUDE_chocolate_bars_per_small_box_l795_79526

theorem chocolate_bars_per_small_box 
  (total_bars : ℕ) 
  (small_boxes : ℕ) 
  (h1 : total_bars = 525) 
  (h2 : small_boxes = 21) : 
  total_bars / small_boxes = 25 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_per_small_box_l795_79526


namespace NUMINAMATH_CALUDE_polynomial_factorization_l795_79546

theorem polynomial_factorization (x : ℝ) : 
  x^8 - 16 = (x^2 - 2) * (x^2 + 2) * (x^2 - 2*x + 2) * (x^2 + 2*x + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l795_79546


namespace NUMINAMATH_CALUDE_spade_calculation_l795_79568

-- Define the spade operation
def spade (x y : ℝ) : ℝ := (x + y + 1) * (x - y)

-- Theorem statement
theorem spade_calculation : spade 2 (spade 3 6) = -864 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l795_79568


namespace NUMINAMATH_CALUDE_constant_term_is_165_l795_79533

-- Define the derivative function
def derivative (q : ℝ → ℝ) : ℝ → ℝ := sorry

-- Define the equation q' = 3q + c
def equation (c : ℝ) (q : ℝ → ℝ) : Prop :=
  ∀ x, derivative q x = 3 * q x + c

-- State the theorem
theorem constant_term_is_165 :
  ∃ (q : ℝ → ℝ) (c : ℝ),
    equation c q ∧
    derivative (derivative q) 6 = 210 ∧
    c = 165 :=
sorry

end NUMINAMATH_CALUDE_constant_term_is_165_l795_79533


namespace NUMINAMATH_CALUDE_even_function_condition_l795_79575

/-- A function f is even if f(-x) = f(x) for all x in ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = (x+a)(x-4) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * (x - 4)

theorem even_function_condition (a : ℝ) : IsEven (f a) ↔ a = 4 := by
  sorry

end NUMINAMATH_CALUDE_even_function_condition_l795_79575


namespace NUMINAMATH_CALUDE_not_square_of_two_pow_minus_one_l795_79592

theorem not_square_of_two_pow_minus_one (n : ℕ) (h : n > 1) :
  ¬ ∃ k : ℕ, 2^n - 1 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_square_of_two_pow_minus_one_l795_79592


namespace NUMINAMATH_CALUDE_bobs_work_hours_l795_79589

/-- Given Bob's wage increase, benefit reduction, and net weekly gain, 
    prove that he works 40 hours per week. -/
theorem bobs_work_hours : 
  ∀ (h : ℝ), 
    (0.50 * h - 15 = 5) → 
    h = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_bobs_work_hours_l795_79589


namespace NUMINAMATH_CALUDE_two_digit_number_puzzle_l795_79591

theorem two_digit_number_puzzle :
  ∀ x y : ℕ,
  x < 10 ∧ y < 10 ∧  -- Ensuring x and y are single digits
  x + y = 7 ∧  -- Sum of digits is 7
  (x + 2) + 10 * (y + 2) = 2 * (10 * y + x) - 3  -- Condition after adding 2 to each digit
  → 10 * y + x = 25 :=  -- The original number is 25
by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_puzzle_l795_79591


namespace NUMINAMATH_CALUDE_city_male_population_l795_79573

theorem city_male_population (total_population : ℕ) (num_parts : ℕ) (male_parts : ℕ) :
  total_population = 800 →
  num_parts = 4 →
  male_parts = 2 →
  (total_population / num_parts) * male_parts = 400 :=
by sorry

end NUMINAMATH_CALUDE_city_male_population_l795_79573


namespace NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l795_79520

/-- Theorem: Area of a triangle with given perimeter and inradius -/
theorem triangle_area_from_perimeter_and_inradius 
  (perimeter : ℝ) 
  (inradius : ℝ) 
  (h_perimeter : perimeter = 40) 
  (h_inradius : inradius = 2.5) : 
  inradius * (perimeter / 2) = 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l795_79520


namespace NUMINAMATH_CALUDE_cafeteria_line_swaps_l795_79502

/-- Represents a student in the line -/
inductive Student
| Boy : Student
| Girl : Student

/-- The initial line of students -/
def initial_line : List Student :=
  (List.range 8).bind (fun _ => [Student.Boy, Student.Girl])

/-- The final line of students -/
def final_line : List Student :=
  (List.replicate 8 Student.Girl) ++ (List.replicate 8 Student.Boy)

/-- The number of swaps required -/
def num_swaps : Nat := (List.range 8).sum

theorem cafeteria_line_swaps :
  num_swaps = 36 ∧
  num_swaps = (initial_line.length / 2) * ((initial_line.length / 2) + 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_cafeteria_line_swaps_l795_79502


namespace NUMINAMATH_CALUDE_max_actors_in_tournament_l795_79536

/-- Represents the result of a chess match -/
inductive MatchResult
  | Win
  | Draw
  | Loss

/-- Calculates the score for a given match result -/
def scoreForResult (result : MatchResult) : Rat :=
  match result with
  | MatchResult.Win => 1
  | MatchResult.Draw => 1/2
  | MatchResult.Loss => 0

/-- Represents a chess tournament -/
structure ChessTournament (n : ℕ) where
  /-- The results of all matches in the tournament -/
  results : Fin n → Fin n → MatchResult
  /-- Each player plays exactly one match against each other player -/
  no_self_play : ∀ i, results i i = MatchResult.Draw
  /-- Matches are symmetric: if A wins against B, B loses against A -/
  symmetry : ∀ i j, results i j = MatchResult.Win ↔ results j i = MatchResult.Loss

/-- Calculates the score of player i against player j -/
def score (tournament : ChessTournament n) (i j : Fin n) : Rat :=
  scoreForResult (tournament.results i j)

/-- The tournament satisfies the "1.5 solido" condition -/
def satisfies_condition (tournament : ChessTournament n) : Prop :=
  ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    (score tournament i j + score tournament i k = 3/2) ∨
    (score tournament j i + score tournament j k = 3/2) ∨
    (score tournament k i + score tournament k j = 3/2)

/-- The main theorem: the maximum number of actors in a valid tournament is 5 -/
theorem max_actors_in_tournament :
  (∃ (tournament : ChessTournament 5), satisfies_condition tournament) ∧
  (∀ n > 5, ¬∃ (tournament : ChessTournament n), satisfies_condition tournament) :=
sorry

end NUMINAMATH_CALUDE_max_actors_in_tournament_l795_79536


namespace NUMINAMATH_CALUDE_smallest_a1_l795_79544

def is_valid_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n > 1, a n = 11 * a (n - 1) - n)

theorem smallest_a1 (a : ℕ → ℝ) (h : is_valid_sequence a) :
  ∀ ε > 0, a 1 ≥ 21 / 100 - ε :=
sorry

end NUMINAMATH_CALUDE_smallest_a1_l795_79544


namespace NUMINAMATH_CALUDE_no_equal_sum_partition_l795_79553

/-- A group of four consecutive natural numbers -/
structure NumberGroup :=
  (start : ℕ)
  (h : start > 0 ∧ start ≤ 69)

/-- The product of four consecutive natural numbers starting from n -/
def groupProduct (g : NumberGroup) : ℕ :=
  g.start * (g.start + 1) * (g.start + 2) * (g.start + 3)

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- A partition of 72 consecutive natural numbers into 18 groups -/
def Partition := Fin 18 → NumberGroup

/-- The theorem stating that no partition exists where all groups have the same sum of digits of their product -/
theorem no_equal_sum_partition :
  ¬ ∃ (p : Partition), ∃ (s : ℕ), ∀ i : Fin 18, sumOfDigits (groupProduct (p i)) = s :=
sorry

end NUMINAMATH_CALUDE_no_equal_sum_partition_l795_79553


namespace NUMINAMATH_CALUDE_inequality_implies_x_equals_one_l795_79561

theorem inequality_implies_x_equals_one (x : ℝ) : 
  (∀ m : ℝ, m > 0 → (m * x - 1) * (3 * m^2 - (x + 1) * m - 1) ≥ 0) → 
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_x_equals_one_l795_79561


namespace NUMINAMATH_CALUDE_inequality_proof_l795_79558

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a^2 + 1) * (b^3 + 2) * (c^5 + 4) ≥ 30 ∧
  ((a^2 + 1) * (b^3 + 2) * (c^5 + 4) = 30 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l795_79558


namespace NUMINAMATH_CALUDE_ABABCDCD_square_theorem_l795_79579

/-- Represents an 8-digit number in the form ABABCDCD -/
def ABABCDCD (A B C D : Nat) : Nat :=
  A * 10000000 + B * 1000000 + A * 100000 + B * 10000 + C * 1000 + D * 100 + C * 10 + D

/-- Checks if four numbers are distinct digits -/
def areDistinctDigits (A B C D : Nat) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10

/-- The main theorem stating that only two sets of digits satisfy the conditions -/
theorem ABABCDCD_square_theorem :
  ∀ A B C D : Nat,
    (ABABCDCD A B C D)^2 = ABABCDCD A B C D ∧ areDistinctDigits A B C D →
    ((A = 9 ∧ B = 7 ∧ C = 0 ∧ D = 4) ∨ (A = 8 ∧ B = 0 ∧ C = 2 ∧ D = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ABABCDCD_square_theorem_l795_79579


namespace NUMINAMATH_CALUDE_dodecahedron_triangle_probability_l795_79506

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  vertices : Nat
  connections_per_vertex : Nat

/-- The probability of forming a triangle with three randomly chosen vertices -/
def triangle_probability (d : RegularDodecahedron) : ℚ :=
  sorry

/-- Theorem stating the probability of forming a triangle in a regular dodecahedron -/
theorem dodecahedron_triangle_probability :
  let d : RegularDodecahedron := ⟨20, 3⟩
  triangle_probability d = 1 / 57 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_triangle_probability_l795_79506


namespace NUMINAMATH_CALUDE_race_speed_factor_l795_79543

/-- Represents the race scenario described in the problem -/
structure RaceScenario where
  k : ℝ  -- Factor by which A is faster than B
  startAdvantage : ℝ  -- Head start given to B in meters
  totalDistance : ℝ  -- Total race distance in meters

/-- Theorem stating that under the given conditions, A must be 4 times faster than B -/
theorem race_speed_factor (race : RaceScenario) 
  (h1 : race.startAdvantage = 72)
  (h2 : race.totalDistance = 96)
  (h3 : race.totalDistance / race.k = (race.totalDistance - race.startAdvantage)) :
  race.k = 4 := by
  sorry


end NUMINAMATH_CALUDE_race_speed_factor_l795_79543


namespace NUMINAMATH_CALUDE_find_heaviest_and_lightest_in_13_weighings_l795_79559

/-- Represents a coin with a unique weight -/
structure Coin where
  weight : ℕ

/-- Represents the result of weighing two coins -/
inductive WeighResult
  | Left  : WeighResult  -- left coin is heavier
  | Right : WeighResult  -- right coin is heavier
  | Equal : WeighResult  -- coins have equal weight

/-- A function that simulates weighing two coins -/
def weigh (a b : Coin) : WeighResult :=
  if a.weight > b.weight then WeighResult.Left
  else if a.weight < b.weight then WeighResult.Right
  else WeighResult.Equal

/-- Theorem stating that it's possible to find the heaviest and lightest coins in 13 weighings -/
theorem find_heaviest_and_lightest_in_13_weighings
  (coins : List Coin)
  (h_distinct : ∀ i j, i ≠ j → (coins.get i).weight ≠ (coins.get j).weight)
  (h_count : coins.length = 10) :
  ∃ (heaviest lightest : Coin) (steps : List (Coin × Coin)),
    heaviest ∈ coins ∧
    lightest ∈ coins ∧
    (∀ c ∈ coins, c.weight ≤ heaviest.weight) ∧
    (∀ c ∈ coins, c.weight ≥ lightest.weight) ∧
    steps.length ≤ 13 ∧
    (∀ step ∈ steps, ∃ a b, step = (a, b) ∧ a ∈ coins ∧ b ∈ coins) :=
by sorry

end NUMINAMATH_CALUDE_find_heaviest_and_lightest_in_13_weighings_l795_79559


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l795_79508

/-- Given that x² varies inversely with √w, prove that w = 1 when x = 6,
    given that x = 3 when w = 16. -/
theorem inverse_variation_problem (x w : ℝ) (k : ℝ) (h1 : x^2 * Real.sqrt w = k)
    (h2 : 3^2 * Real.sqrt 16 = k) (h3 : x = 6) : w = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l795_79508


namespace NUMINAMATH_CALUDE_mikes_weekly_exercises_l795_79562

/-- Represents the number of repetitions for each exercise -/
structure ExerciseReps where
  pullUps : ℕ
  pushUps : ℕ
  squats : ℕ

/-- Represents the number of daily visits to each room -/
structure RoomVisits where
  office : ℕ
  kitchen : ℕ
  livingRoom : ℕ

/-- Calculates the total number of exercises performed in a week -/
def weeklyExercises (reps : ExerciseReps) (visits : RoomVisits) : ExerciseReps :=
  { pullUps := reps.pullUps * visits.office * 7,
    pushUps := reps.pushUps * visits.kitchen * 7,
    squats := reps.squats * visits.livingRoom * 7 }

/-- Mike's exercise routine -/
def mikesRoutine : ExerciseReps :=
  { pullUps := 2, pushUps := 5, squats := 10 }

/-- Mike's daily room visits -/
def mikesVisits : RoomVisits :=
  { office := 5, kitchen := 8, livingRoom := 7 }

theorem mikes_weekly_exercises :
  weeklyExercises mikesRoutine mikesVisits = { pullUps := 70, pushUps := 280, squats := 490 } := by
  sorry

end NUMINAMATH_CALUDE_mikes_weekly_exercises_l795_79562


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l795_79576

/-- Two lines are parallel if their slopes are equal and they are not coincident -/
def parallel (a b c d e f : ℝ) : Prop :=
  a / d = b / e ∧ a / d ≠ c / f

/-- First line equation: ax + 2y + 3a = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop :=
  a * x + 2 * y + 3 * a = 0

/-- Second line equation: 3x + (a-1)y + a^2 - a + 3 = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop :=
  3 * x + (a - 1) * y + a^2 - a + 3 = 0

/-- Theorem stating that a=3 is neither sufficient nor necessary for the lines to be parallel -/
theorem not_sufficient_nor_necessary :
  ¬(∀ a : ℝ, a = 3 → parallel a 2 (3*a) 3 (a-1) (a^2 - a + 3)) ∧
  ¬(∀ a : ℝ, parallel a 2 (3*a) 3 (a-1) (a^2 - a + 3) → a = 3) :=
sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l795_79576


namespace NUMINAMATH_CALUDE_arthur_walked_four_point_five_miles_l795_79577

/-- The distance Arthur walked in miles -/
def arthurs_distance (blocks_west : ℕ) (blocks_south : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_west + blocks_south : ℚ) * miles_per_block

/-- Theorem stating that Arthur walked 4.5 miles -/
theorem arthur_walked_four_point_five_miles :
  arthurs_distance 8 10 (1/4) = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_arthur_walked_four_point_five_miles_l795_79577


namespace NUMINAMATH_CALUDE_book_cost_price_l795_79551

theorem book_cost_price (cost_price : ℝ) : cost_price = 2200 :=
  let selling_price_10_percent := 1.10 * cost_price
  let selling_price_15_percent := 1.15 * cost_price
  have h1 : selling_price_15_percent - selling_price_10_percent = 110 := by sorry
  sorry

end NUMINAMATH_CALUDE_book_cost_price_l795_79551


namespace NUMINAMATH_CALUDE_diane_has_27_cents_l795_79514

/-- The amount of money Diane has, given the cost of cookies and the additional amount she needs. -/
def dianes_money (cookie_cost additional_needed : ℕ) : ℕ :=
  cookie_cost - additional_needed

/-- Theorem stating that Diane has 27 cents given the problem conditions. -/
theorem diane_has_27_cents :
  dianes_money 65 38 = 27 := by
  sorry

end NUMINAMATH_CALUDE_diane_has_27_cents_l795_79514


namespace NUMINAMATH_CALUDE_log_reciprocal_l795_79547

theorem log_reciprocal (M : ℝ) (a : ℤ) (b : ℝ) 
  (h_pos : M > 0) 
  (h_log : Real.log M / Real.log 10 = a + b) 
  (h_b : 0 < b ∧ b < 1) : 
  Real.log (1 / M) / Real.log 10 = (-a - 1) + (1 - b) := by
  sorry

end NUMINAMATH_CALUDE_log_reciprocal_l795_79547


namespace NUMINAMATH_CALUDE_julia_remaining_money_l795_79540

def initial_amount : ℚ := 40
def game_fraction : ℚ := 1/2
def in_game_purchase_fraction : ℚ := 1/4

theorem julia_remaining_money :
  let amount_after_game := initial_amount * (1 - game_fraction)
  let final_amount := amount_after_game * (1 - in_game_purchase_fraction)
  final_amount = 15 := by sorry

end NUMINAMATH_CALUDE_julia_remaining_money_l795_79540


namespace NUMINAMATH_CALUDE_no_base_for_square_l795_79584

theorem no_base_for_square (b : ℤ) : b > 4 → ¬∃ (k : ℤ), b^2 + 4*b + 3 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_no_base_for_square_l795_79584


namespace NUMINAMATH_CALUDE_inequality_squared_not_always_true_l795_79512

theorem inequality_squared_not_always_true : ¬ ∀ x y : ℝ, x < y → x^2 < y^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_squared_not_always_true_l795_79512


namespace NUMINAMATH_CALUDE_cousins_distribution_l795_79538

/-- The number of ways to distribute n indistinguishable objects into k distinct boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 5 cousins -/
def num_cousins : ℕ := 5

/-- There are 4 rooms -/
def num_rooms : ℕ := 4

/-- The number of ways to distribute the cousins into the rooms -/
def num_distributions : ℕ := distribute num_cousins num_rooms

theorem cousins_distribution :
  num_distributions = 66 := by sorry

end NUMINAMATH_CALUDE_cousins_distribution_l795_79538


namespace NUMINAMATH_CALUDE_henry_twice_jill_age_l795_79594

/-- Represents the number of years ago when Henry was twice Jill's age -/
def years_ago (henry_age : ℕ) (jill_age : ℕ) : ℕ :=
  henry_age - jill_age

/-- Theorem stating that Henry was twice Jill's age 7 years ago -/
theorem henry_twice_jill_age (henry_age jill_age : ℕ) : 
  henry_age = 25 → 
  jill_age = 16 → 
  henry_age + jill_age = 41 → 
  years_ago henry_age jill_age = 7 := by
sorry

end NUMINAMATH_CALUDE_henry_twice_jill_age_l795_79594


namespace NUMINAMATH_CALUDE_harry_needs_five_spellbooks_l795_79599

/-- Represents the cost and quantity of items Harry needs to buy --/
structure HarrysPurchase where
  spellbookCost : ℕ
  potionKitCost : ℕ
  owlCost : ℕ
  silverToGoldRatio : ℕ
  totalSilver : ℕ
  potionKitQuantity : ℕ

/-- Calculates the number of spellbooks Harry needs to buy --/
def calculateSpellbooks (purchase : HarrysPurchase) : ℕ :=
  let remainingSilver := purchase.totalSilver -
    (purchase.owlCost * purchase.silverToGoldRatio + 
     purchase.potionKitCost * purchase.potionKitQuantity)
  remainingSilver / (purchase.spellbookCost * purchase.silverToGoldRatio)

/-- Theorem stating that Harry needs to buy 5 spellbooks --/
theorem harry_needs_five_spellbooks (purchase : HarrysPurchase) 
  (h1 : purchase.spellbookCost = 5)
  (h2 : purchase.potionKitCost = 20)
  (h3 : purchase.owlCost = 28)
  (h4 : purchase.silverToGoldRatio = 9)
  (h5 : purchase.totalSilver = 537)
  (h6 : purchase.potionKitQuantity = 3) :
  calculateSpellbooks purchase = 5 := by
  sorry


end NUMINAMATH_CALUDE_harry_needs_five_spellbooks_l795_79599


namespace NUMINAMATH_CALUDE_john_ray_difference_l795_79563

/-- The number of chickens each person took -/
structure ChickenCount where
  john : ℕ
  mary : ℕ
  ray : ℕ

/-- The conditions of the chicken problem -/
def chicken_problem (c : ChickenCount) : Prop :=
  c.john = c.mary + 5 ∧
  c.mary = c.ray + 6 ∧
  c.ray = 10

/-- The theorem stating the difference between John's and Ray's chickens -/
theorem john_ray_difference (c : ChickenCount) 
  (h : chicken_problem c) : c.john - c.ray = 11 := by
  sorry

end NUMINAMATH_CALUDE_john_ray_difference_l795_79563


namespace NUMINAMATH_CALUDE_a_range_l795_79503

theorem a_range (a : ℝ) (h1 : a < 9 * a^3 - 11 * a) (h2 : 9 * a^3 - 11 * a < |a|) (h3 : a < 0) :
  -2 * Real.sqrt 3 / 3 < a ∧ a < -Real.sqrt 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l795_79503


namespace NUMINAMATH_CALUDE_find_n_l795_79590

theorem find_n : ∃ n : ℤ, (5 : ℝ) ^ (2 * n) = (1 / 5 : ℝ) ^ (n - 12) ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l795_79590


namespace NUMINAMATH_CALUDE_brenda_mice_fraction_l795_79532

/-- The fraction of baby mice Brenda gave to Robbie -/
def f : ℚ := sorry

/-- The total number of baby mice -/
def total_mice : ℕ := 3 * 8

theorem brenda_mice_fraction :
  (f * total_mice : ℚ) +                        -- Mice given to Robbie
  (3 * f * total_mice : ℚ) +                    -- Mice sold to pet store
  ((1 - 4 * f) * total_mice / 2 : ℚ) +          -- Mice sold to snake owners
  4 = total_mice ∧                              -- Remaining mice
  f = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_brenda_mice_fraction_l795_79532


namespace NUMINAMATH_CALUDE_sum_of_fractions_bounds_l795_79531

theorem sum_of_fractions_bounds (v w x y z : ℝ) (hv : v > 0) (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  1 < (v / (v + w)) + (w / (w + x)) + (x / (x + y)) + (y / (y + z)) + (z / (z + v)) ∧
  (v / (v + w)) + (w / (w + x)) + (x / (x + y)) + (y / (y + z)) + (z / (z + v)) < 4 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_bounds_l795_79531


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l795_79539

/-- A line in the 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) is on the line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if a line has equal intercepts on both axes -/
def Line.hasEqualIntercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c ≠ 0 ∧ l.c / l.a = - l.c / l.b

theorem line_through_point_with_equal_intercepts :
  ∃ (l : Line), l.contains (-1) 2 ∧ l.hasEqualIntercepts ∧
  ((l.a = 2 ∧ l.b = 1 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = 1 ∧ l.c = -1)) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l795_79539


namespace NUMINAMATH_CALUDE_proposition_relationship_l795_79516

theorem proposition_relationship :
  (∀ a : ℝ, 0 < a ∧ a < 1 → ∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ∧
  (∃ a : ℝ, (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ∧ ¬(0 < a ∧ a < 1)) := by
  sorry

end NUMINAMATH_CALUDE_proposition_relationship_l795_79516


namespace NUMINAMATH_CALUDE_frank_remaining_money_l795_79522

/-- Calculates the remaining money after buying the most expensive lamp -/
def remaining_money (cheapest_lamp_cost most_expensive_factor current_money : ℕ) : ℕ :=
  current_money - (cheapest_lamp_cost * most_expensive_factor)

/-- Proves that Frank will have $30 remaining after buying the most expensive lamp -/
theorem frank_remaining_money :
  remaining_money 20 3 90 = 30 := by sorry

end NUMINAMATH_CALUDE_frank_remaining_money_l795_79522


namespace NUMINAMATH_CALUDE_solution_set_is_x_gt_one_l795_79569

/-- A linear function y = kx + b with a table of x and y values -/
structure LinearFunction where
  k : ℝ
  b : ℝ
  k_nonzero : k ≠ 0
  x_values : List ℝ := [-2, -1, 0, 1, 2, 3]
  y_values : List ℝ := [3, 2, 1, 0, -1, -2]
  table_valid : x_values.length = y_values.length

/-- The solution set of kx + b < 0 for the given linear function -/
def solutionSet (f : LinearFunction) : Set ℝ :=
  {x | f.k * x + f.b < 0}

/-- Theorem stating that the solution set is x > 1 -/
theorem solution_set_is_x_gt_one (f : LinearFunction) : 
  solutionSet f = {x | x > 1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_is_x_gt_one_l795_79569


namespace NUMINAMATH_CALUDE_selling_multiple_satisfies_profit_equation_l795_79521

/-- The multiple of the value of components that John sells computers for -/
def selling_multiple : ℝ := 1.4

/-- Cost of parts for one computer -/
def parts_cost : ℝ := 800

/-- Number of computers built per month -/
def computers_per_month : ℕ := 60

/-- Monthly rent -/
def monthly_rent : ℝ := 5000

/-- Monthly non-rent extra expenses -/
def extra_expenses : ℝ := 3000

/-- Monthly profit -/
def monthly_profit : ℝ := 11200

/-- Theorem stating that the selling multiple satisfies the profit equation -/
theorem selling_multiple_satisfies_profit_equation :
  computers_per_month * parts_cost * selling_multiple -
  (computers_per_month * parts_cost + monthly_rent + extra_expenses) = monthly_profit := by
  sorry

end NUMINAMATH_CALUDE_selling_multiple_satisfies_profit_equation_l795_79521


namespace NUMINAMATH_CALUDE_shaded_area_ratio_l795_79583

/-- Given a rectangle divided into a 4x5 grid of 1cm x 1cm smaller rectangles,
    with a shaded area consisting of 3 full small rectangles and 4 half small rectangles,
    prove that the ratio of the shaded area to the total area is 1/4. -/
theorem shaded_area_ratio (total_width : ℝ) (total_height : ℝ) 
  (full_rectangles : ℕ) (half_rectangles : ℕ) :
  total_width = 4 →
  total_height = 5 →
  full_rectangles = 3 →
  half_rectangles = 4 →
  (full_rectangles + half_rectangles / 2) / (total_width * total_height) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_ratio_l795_79583


namespace NUMINAMATH_CALUDE_ratio_problem_l795_79581

theorem ratio_problem (A B C : ℝ) (h1 : A + B + C = 98) (h2 : B / C = 5 / 8) (h3 : B = 30) :
  A / B = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l795_79581


namespace NUMINAMATH_CALUDE_concert_drive_distance_l795_79595

/-- Calculates the remaining distance to drive given the total distance and the distance already driven. -/
def remaining_distance (total : ℕ) (driven : ℕ) : ℕ :=
  total - driven

/-- Theorem stating that given a total distance of 78 miles and a distance already driven of 32 miles, 
    the remaining distance to drive is 46 miles. -/
theorem concert_drive_distance : remaining_distance 78 32 = 46 := by
  sorry

end NUMINAMATH_CALUDE_concert_drive_distance_l795_79595


namespace NUMINAMATH_CALUDE_last_two_digits_sum_of_squares_l795_79527

theorem last_two_digits_sum_of_squares :
  ∀ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℕ),
  a₁ % 100 = 11 →
  a₂ % 100 = 12 →
  a₃ % 100 = 13 →
  a₄ % 100 = 14 →
  a₅ % 100 = 15 →
  a₆ % 100 = 16 →
  a₇ % 100 = 17 →
  a₈ % 100 = 18 →
  a₉ % 100 = 19 →
  (a₁^2 + a₂^2 + a₃^2 + a₄^2 + a₅^2 + a₆^2 + a₇^2 + a₈^2 + a₉^2) % 100 = 85 :=
by sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_of_squares_l795_79527


namespace NUMINAMATH_CALUDE_car_speed_comparison_l795_79550

theorem car_speed_comparison (u v w : ℝ) (hu : u > 0) (hv : v > 0) (hw : w > 0) :
  3 / (1/u + 1/v + 1/w) ≤ (u + v) / 2 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_comparison_l795_79550


namespace NUMINAMATH_CALUDE_perpendicular_tangents_and_inequality_l795_79593

noncomputable def f (x : ℝ) := x^2 + 4*x + 2

noncomputable def g (t x : ℝ) := t * Real.exp x * ((2*x + 4) - 2)

theorem perpendicular_tangents_and_inequality (t k : ℝ) : 
  (((2 * (-17/8) + 4) * (2 * t * Real.exp 0 * (0 + 2)) = -1) ∧
   (∀ x : ℝ, x ≥ 2 → k * g 1 x ≥ 2 * f x)) ↔ 
  (t = 1 ∧ 2 ≤ k ∧ k ≤ 2 * Real.exp 2) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_and_inequality_l795_79593


namespace NUMINAMATH_CALUDE_lo_length_l795_79564

/-- Represents a parallelogram LMNO with given properties -/
structure Parallelogram where
  -- Length of side MN
  mn_length : ℝ
  -- Altitude from O to MN
  altitude_o_to_mn : ℝ
  -- Altitude from N to LO
  altitude_n_to_lo : ℝ
  -- Condition that LMNO is a parallelogram
  is_parallelogram : True

/-- Theorem stating the length of LO in the parallelogram LMNO -/
theorem lo_length (p : Parallelogram)
  (h1 : p.mn_length = 15)
  (h2 : p.altitude_o_to_mn = 9)
  (h3 : p.altitude_n_to_lo = 7) :
  ∃ (lo_length : ℝ), lo_length = 19 + 2 / 7 ∧ 
  p.mn_length * p.altitude_o_to_mn = lo_length * p.altitude_n_to_lo :=
sorry

end NUMINAMATH_CALUDE_lo_length_l795_79564


namespace NUMINAMATH_CALUDE_min_both_beethoven_chopin_l795_79501

theorem min_both_beethoven_chopin 
  (total : ℕ) 
  (beethoven_fans : ℕ) 
  (chopin_fans : ℕ) 
  (h1 : total = 150) 
  (h2 : beethoven_fans = 120) 
  (h3 : chopin_fans = 95) :
  (beethoven_fans + chopin_fans - total : ℤ).natAbs ≥ 65 :=
by sorry

end NUMINAMATH_CALUDE_min_both_beethoven_chopin_l795_79501


namespace NUMINAMATH_CALUDE_milk_water_ratio_l795_79571

theorem milk_water_ratio (initial_volume : ℝ) (water_added : ℝ) 
  (milk : ℝ) (water : ℝ) : 
  initial_volume = 115 →
  water_added = 46 →
  milk + water = initial_volume →
  milk / (water + water_added) = 3 / 4 →
  milk / water = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_milk_water_ratio_l795_79571


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l795_79580

theorem contrapositive_equivalence (x : ℝ) :
  (x = 1 → x^2 - 3*x + 2 = 0) ↔ (x^2 - 3*x + 2 ≠ 0 → x ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l795_79580


namespace NUMINAMATH_CALUDE_quadratic_inequality_coefficient_sum_l795_79598

/-- Given a quadratic inequality ax^2 + bx + 2 > 0 with solution set (-1/2, 1/3),
    prove that a + b = -14 -/
theorem quadratic_inequality_coefficient_sum (a b : ℝ) : 
  (∀ x, a * x^2 + b * x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) →
  a + b = -14 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_coefficient_sum_l795_79598


namespace NUMINAMATH_CALUDE_parabola_focal_chord_inclination_l795_79560

theorem parabola_focal_chord_inclination (x y : ℝ) (α : ℝ) : 
  y^2 = 6*x →  -- parabola equation
  12 = 6 / (Real.sin α)^2 →  -- focal chord length condition
  α = π/4 ∨ α = 3*π/4 :=  -- conclusion
by sorry

end NUMINAMATH_CALUDE_parabola_focal_chord_inclination_l795_79560


namespace NUMINAMATH_CALUDE_parallel_transitive_l795_79515

-- Define the parallel relation
def parallel (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem parallel_transitive (a b c : Line) :
  parallel a b → parallel b c → parallel a c := by sorry

end NUMINAMATH_CALUDE_parallel_transitive_l795_79515


namespace NUMINAMATH_CALUDE_probability_sum_10_l795_79566

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The set of possible outcomes when throwing two dice -/
def outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range numFaces) (Finset.range numFaces)

/-- The total number of possible outcomes -/
def totalOutcomes : ℕ := numFaces * numFaces

/-- The sum of two numbers -/
def sum (pair : ℕ × ℕ) : ℕ := pair.1 + pair.2

/-- The set of favorable outcomes (sum equals 10) -/
def favorableOutcomes : Finset (ℕ × ℕ) :=
  outcomes.filter (fun pair => sum pair = 10)

/-- The number of favorable outcomes -/
def numFavorableOutcomes : ℕ := favorableOutcomes.card

theorem probability_sum_10 :
  (numFavorableOutcomes : ℚ) / totalOutcomes = 5 / 36 := by
  sorry

#eval numFavorableOutcomes -- Should output 5
#eval totalOutcomes -- Should output 36

end NUMINAMATH_CALUDE_probability_sum_10_l795_79566


namespace NUMINAMATH_CALUDE_teacher_student_grouping_probability_l795_79534

/-- The number of teachers -/
def num_teachers : ℕ := 2

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of groups -/
def num_groups : ℕ := 2

/-- The number of teachers per group -/
def teachers_per_group : ℕ := 1

/-- The number of students per group -/
def students_per_group : ℕ := 2

/-- The probability that teacher A and student B are in the same group -/
def prob_same_group : ℚ := 1/2

theorem teacher_student_grouping_probability :
  (num_teachers = 2) →
  (num_students = 4) →
  (num_groups = 2) →
  (teachers_per_group = 1) →
  (students_per_group = 2) →
  prob_same_group = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_teacher_student_grouping_probability_l795_79534


namespace NUMINAMATH_CALUDE_sum_of_periodic_functions_periodicity_l795_79507

/-- A periodic function with period T -/
def IsPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

/-- A function with smallest positive period T -/
def HasSmallestPeriod (f : ℝ → ℝ) (T : ℝ) : Prop :=
  IsPeriodic f T ∧ ∀ S, 0 < S → S < T → ¬IsPeriodic f S

/-- The main theorem about the sum of two periodic functions -/
theorem sum_of_periodic_functions_periodicity
  (f₁ f₂ : ℝ → ℝ) (T : ℝ) (hT : T > 0)
  (h₁ : HasSmallestPeriod f₁ T) (h₂ : HasSmallestPeriod f₂ T) :
  ∃ y : ℝ → ℝ, (y = f₁ + f₂) ∧ 
  IsPeriodic y T ∧ 
  ¬(∃ S : ℝ, HasSmallestPeriod y S) :=
sorry

end NUMINAMATH_CALUDE_sum_of_periodic_functions_periodicity_l795_79507


namespace NUMINAMATH_CALUDE_sum_of_edges_l795_79518

/-- A rectangular solid with given properties -/
structure RectangularSolid where
  a : ℝ  -- length
  b : ℝ  -- width
  c : ℝ  -- height
  volume_eq : a * b * c = 8
  surface_area_eq : 2 * (a * b + b * c + c * a) = 32
  width_sq_eq : b ^ 2 = a * c

/-- The sum of all edges of the rectangular solid is 32 -/
theorem sum_of_edges (solid : RectangularSolid) :
  4 * (solid.a + solid.b + solid.c) = 32 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_edges_l795_79518


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_of_cubes_l795_79578

/-- Given an arithmetic sequence with first term x and common difference 2,
    this function returns the sum of cubes of the first n+1 terms. -/
def sumOfCubes (x : ℤ) (n : ℕ) : ℤ :=
  (Finset.range (n+1)).sum (fun i => (x + 2 * i)^3)

/-- Theorem stating that for an arithmetic sequence with integer first term,
    if the sum of cubes of its terms is -6859 and the number of terms is greater than 6,
    then the number of terms is exactly 7 (i.e., n = 6). -/
theorem arithmetic_sequence_sum_of_cubes (x : ℤ) (n : ℕ) 
    (h1 : sumOfCubes x n = -6859)
    (h2 : n > 5) : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_of_cubes_l795_79578


namespace NUMINAMATH_CALUDE_time_2nd_to_7th_floor_l795_79556

/-- Time needed to go from floor a to floor b, given the time to go from floor c to floor d -/
def time_between_floors (a b c d : ℕ) (time_cd : ℕ) : ℕ :=
  ((b - a) * time_cd) / (d - c)

/-- The theorem stating that it takes 50 seconds to go from the 2nd to the 7th floor -/
theorem time_2nd_to_7th_floor : 
  time_between_floors 2 7 1 5 40 = 50 := by sorry

end NUMINAMATH_CALUDE_time_2nd_to_7th_floor_l795_79556


namespace NUMINAMATH_CALUDE_six_minutes_to_hours_l795_79545

-- Define the conversion factor from minutes to hours
def minutes_to_hours (minutes : ℚ) : ℚ := minutes / 60

-- State the theorem
theorem six_minutes_to_hours : 
  minutes_to_hours 6 = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_six_minutes_to_hours_l795_79545


namespace NUMINAMATH_CALUDE_guaranteed_scores_l795_79541

/-- Represents a player in the card game -/
inductive Player : Type
| One
| Two

/-- The deck of cards for each player -/
def player_deck (p : Player) : List Nat :=
  match p with
  | Player.One => List.range 1000 |>.map (fun n => 2 * n + 2)
  | Player.Two => List.range 1001 |>.map (fun n => 2 * n + 1)

/-- The number of turns in the game -/
def num_turns : Nat := 1000

/-- The result of the game -/
structure GameResult where
  player1_score : Nat
  player2_score : Nat

/-- A strategy for playing the game -/
def Strategy := List Nat → Nat

/-- Play the game with given strategies -/
def play_game (s1 s2 : Strategy) : GameResult :=
  sorry

/-- The theorem stating the guaranteed minimum scores for both players -/
theorem guaranteed_scores :
  ∃ (s1 : Strategy), ∀ (s2 : Strategy), (play_game s1 s2).player1_score ≥ 499 ∧
  ∃ (s2 : Strategy), ∀ (s1 : Strategy), (play_game s1 s2).player2_score ≥ 501 :=
  sorry

end NUMINAMATH_CALUDE_guaranteed_scores_l795_79541


namespace NUMINAMATH_CALUDE_units_digit_of_7_to_2023_l795_79565

theorem units_digit_of_7_to_2023 : 7^2023 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_to_2023_l795_79565


namespace NUMINAMATH_CALUDE_exists_point_with_no_interior_lattice_points_l795_79530

-- Define a point with integer coordinates
structure IntPoint where
  x : Int
  y : Int

-- Define a function to check if a point is on a line
def onLine (p : IntPoint) (a b c : Int) : Prop :=
  a * p.x + b * p.y = c

-- Define a function to check if a point is in the interior of a segment
def inInterior (p q r : IntPoint) : Prop :=
  ∃ t : Rat, 0 < t ∧ t < 1 ∧
  p.x = q.x + t * (r.x - q.x) ∧
  p.y = q.y + t * (r.y - q.y)

-- Main theorem
theorem exists_point_with_no_interior_lattice_points
  (A B C : IntPoint) (hABC : A ≠ B ∧ B ≠ C ∧ A ≠ C) :
  ∃ P : IntPoint,
    P ≠ A ∧ P ≠ B ∧ P ≠ C ∧
    (∀ Q : IntPoint, ¬(inInterior Q P A ∨ inInterior Q P B ∨ inInterior Q P C)) :=
  sorry

end NUMINAMATH_CALUDE_exists_point_with_no_interior_lattice_points_l795_79530


namespace NUMINAMATH_CALUDE_union_of_A_and_B_complement_of_A_l795_79588

-- Define the sets
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x^2 < 4}

-- State the theorems
theorem union_of_A_and_B : A ∪ B = {x | -2 < x ∧ x ≤ 3} := by sorry

theorem complement_of_A : (Set.univ \ A) = {x | x < -1 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_complement_of_A_l795_79588


namespace NUMINAMATH_CALUDE_mary_has_more_euros_l795_79572

-- Define initial amounts
def michelle_initial : ℚ := 30
def alice_initial : ℚ := 18
def marco_initial : ℚ := 24
def mary_initial : ℚ := 15

-- Define conversion rate
def usd_to_eur : ℚ := 0.85

-- Define transactions
def marco_to_mary : ℚ := marco_initial / 2
def michelle_to_alice : ℚ := michelle_initial * (40 / 100)
def mary_spend : ℚ := 5
def alice_convert : ℚ := 10

-- Calculate final amounts
def marco_final : ℚ := marco_initial - marco_to_mary
def mary_final : ℚ := mary_initial + marco_to_mary - mary_spend
def alice_final_usd : ℚ := alice_initial + michelle_to_alice - alice_convert
def alice_final_eur : ℚ := alice_convert * usd_to_eur

-- Theorem statement
theorem mary_has_more_euros :
  mary_final = marco_final + alice_final_eur + (3/2) := by sorry

end NUMINAMATH_CALUDE_mary_has_more_euros_l795_79572


namespace NUMINAMATH_CALUDE_circle_equation_tangent_to_line_l795_79567

/-- The equation of a circle with center (-1, 1) that is tangent to the line x - y = 0 -/
theorem circle_equation_tangent_to_line (x y : ℝ) : 
  (∃ (r : ℝ), (x + 1)^2 + (y - 1)^2 = r^2 ∧ 
  r = |(-1 - 1 + 0)| / Real.sqrt (1^2 + (-1)^2) ∧
  r > 0) ↔ 
  (x + 1)^2 + (y - 1)^2 = 2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_tangent_to_line_l795_79567


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l795_79596

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_shift_theorem :
  let original := Parabola.mk 1 0 0  -- y = x^2
  let shifted := shift_parabola original 1 2
  shifted = Parabola.mk 1 (-2) 3  -- y = (x-1)^2 + 2
  := by sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l795_79596


namespace NUMINAMATH_CALUDE_parabola_properties_l795_79517

/-- Parabola represented by its parameter p -/
structure Parabola where
  p : ℝ
  p_pos : p > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given a parabola and a point on it, prove the standard form of the parabola
    and the ratio of distances from two points to the focus -/
theorem parabola_properties (E : Parabola) (A : Point)
    (h_on_parabola : A.y^2 = 2 * E.p * A.x)
    (h_y_pos : A.y > 0)
    (h_A_coords : A.x = 9 ∧ A.y = 6)
    (h_AF_length : 5 = |A.x - E.p| + |A.y|) : 
  (∀ (x y : ℝ), y^2 = 4*x ↔ y^2 = 2*E.p*x) ∧ 
  ∃ (B : Point), B ≠ A ∧ 
    (∃ (t : ℝ), 0 < t ∧ t < 1 ∧ 
      B.x = t * A.x + (1 - t) * E.p ∧
      B.y = t * A.y) ∧
    5 / (|B.x - E.p| + |B.y|) = 4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l795_79517


namespace NUMINAMATH_CALUDE_stating_optimal_swap_distance_maximizes_total_distance_l795_79582

/-- Front tire lifespan in kilometers -/
def front_lifespan : ℝ := 11000

/-- Rear tire lifespan in kilometers -/
def rear_lifespan : ℝ := 9000

/-- The optimal swap distance in kilometers -/
def optimal_swap_distance : ℝ := 4950

/-- 
Theorem stating that the optimal swap distance maximizes total distance traveled
while ensuring both tires wear out simultaneously.
-/
theorem optimal_swap_distance_maximizes_total_distance :
  let total_distance := front_lifespan + rear_lifespan
  let front_remaining := 1 - (optimal_swap_distance / front_lifespan)
  let rear_remaining := 1 - (optimal_swap_distance / rear_lifespan)
  let distance_after_swap := front_remaining * rear_lifespan
  (front_remaining * rear_lifespan = rear_remaining * front_lifespan) ∧
  (optimal_swap_distance + distance_after_swap = total_distance) ∧
  (∀ x : ℝ, x ≠ optimal_swap_distance →
    let front_remaining' := 1 - (x / front_lifespan)
    let rear_remaining' := 1 - (x / rear_lifespan)
    let distance_after_swap' := min (front_remaining' * rear_lifespan) (rear_remaining' * front_lifespan)
    x + distance_after_swap' ≤ total_distance) :=
by
  sorry

end NUMINAMATH_CALUDE_stating_optimal_swap_distance_maximizes_total_distance_l795_79582


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l795_79524

theorem polynomial_remainder_theorem (a b : ℚ) : 
  let f : ℚ → ℚ := λ x ↦ a * x^4 + 3 * x^3 - 5 * x^2 + b * x - 7
  (f 2 = 9 ∧ f (-1) = -4) → (a = 7/9 ∧ b = -2/9) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l795_79524


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l795_79523

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_general_term 
  (a : ℕ → ℚ) 
  (h_geo : geometric_sequence a) 
  (h_a1 : a 1 = 3/2) 
  (h_S3 : (a 1) + (a 2) + (a 3) = 9/2) :
  (∃ n : ℕ, a n = 3/2 * (-2)^(n-1)) ∨ (∀ n : ℕ, a n = 3/2) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l795_79523


namespace NUMINAMATH_CALUDE_sequence_problem_l795_79542

theorem sequence_problem (a : ℕ → ℝ) :
  a 1 = 1 ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1) - a n ≤ 3^n) ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 2) - a n ≥ 4 * 3^n) →
  a 2017 = (3^2017 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l795_79542


namespace NUMINAMATH_CALUDE_multiply_preserves_inequality_l795_79519

theorem multiply_preserves_inequality (a b c : ℝ) : a > b → c > 0 → a * c > b * c := by
  sorry

end NUMINAMATH_CALUDE_multiply_preserves_inequality_l795_79519


namespace NUMINAMATH_CALUDE_function_from_derivative_and_point_l795_79525

/-- Given a function f: ℝ → ℝ, if its derivative is 4x³ for all x
and f(1) = -1, then f(x) = x⁴ - 2 for all x. -/
theorem function_from_derivative_and_point (f : ℝ → ℝ) 
    (h1 : ∀ x, deriv f x = 4 * x^3)
    (h2 : f 1 = -1) :
    ∀ x, f x = x^4 - 2 := by
  sorry

end NUMINAMATH_CALUDE_function_from_derivative_and_point_l795_79525


namespace NUMINAMATH_CALUDE_temperature_conversion_l795_79509

theorem temperature_conversion (C F : ℝ) : 
  C = 4/7 * (F - 40) → C = 25 → F = 83.75 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l795_79509


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l795_79555

-- Define the circles
def circle_O₁ (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 4
def circle_O₂ (x y r : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = r^2

-- Define the theorem
theorem circle_intersection_theorem :
  -- Part 1: Tangent case
  (∀ x y : ℝ, circle_O₁ x y → ¬(circle_O₂ x y (12 - 8 * Real.sqrt 2))) →
  -- Part 2: Intersection case
  (∃ A B : ℝ × ℝ, 
    circle_O₁ A.1 A.2 ∧ circle_O₁ B.1 B.2 ∧
    circle_O₂ A.1 A.2 2 ∧ circle_O₂ B.1 B.2 2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8) →
  (∀ x y : ℝ, circle_O₂ x y 2 ∨ circle_O₂ x y (Real.sqrt 20)) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l795_79555


namespace NUMINAMATH_CALUDE_worker_a_time_l795_79535

/-- Proves that Worker A takes 10 hours to do a job alone, given the conditions of the problem -/
theorem worker_a_time (time_b time_together : ℝ) : 
  time_b = 15 → 
  time_together = 6 → 
  (1 / 10 : ℝ) + (1 / time_b) = (1 / time_together) := by
  sorry

end NUMINAMATH_CALUDE_worker_a_time_l795_79535


namespace NUMINAMATH_CALUDE_equality_implies_equal_expressions_l795_79505

theorem equality_implies_equal_expressions (a b : ℝ) : a = b → 2 * (a - 1) = 2 * (b - 1) := by
  sorry

end NUMINAMATH_CALUDE_equality_implies_equal_expressions_l795_79505


namespace NUMINAMATH_CALUDE_pole_wire_distance_l795_79585

/-- Given a vertical pole and three equally spaced wires, calculate the distance between anchor points -/
theorem pole_wire_distance (pole_height : ℝ) (wire_length : ℝ) (anchor_distance : ℝ) : 
  pole_height = 70 →
  wire_length = 490 →
  (pole_height ^ 2 + (anchor_distance / (3 ^ (1/2))) ^ 2 = wire_length ^ 2) →
  anchor_distance = 840 := by
  sorry

end NUMINAMATH_CALUDE_pole_wire_distance_l795_79585


namespace NUMINAMATH_CALUDE_specific_pyramid_surface_area_l795_79552

/-- Represents a pyramid with a parallelogram base -/
structure Pyramid where
  base_side1 : ℝ
  base_side2 : ℝ
  base_diagonal : ℝ
  height : ℝ

/-- Calculates the total surface area of the pyramid -/
def totalSurfaceArea (p : Pyramid) : ℝ :=
  sorry

/-- Theorem stating the total surface area of the specific pyramid -/
theorem specific_pyramid_surface_area :
  let p : Pyramid := { base_side1 := 10, base_side2 := 8, base_diagonal := 6, height := 4 }
  totalSurfaceArea p = 8 * (11 + Real.sqrt 34) := by
  sorry

end NUMINAMATH_CALUDE_specific_pyramid_surface_area_l795_79552


namespace NUMINAMATH_CALUDE_purple_shoes_count_l795_79574

/-- Prove the number of purple shoes in a warehouse --/
theorem purple_shoes_count (total : ℕ) (blue : ℕ) (green : ℕ) (purple : ℕ) : 
  total = 1250 →
  blue = 540 →
  green + purple = total - blue →
  green = purple →
  purple = 355 := by
sorry

end NUMINAMATH_CALUDE_purple_shoes_count_l795_79574


namespace NUMINAMATH_CALUDE_batsman_total_score_l795_79570

/-- Represents a batsman's score in cricket -/
structure BatsmanScore where
  boundaries : ℕ
  sixes : ℕ
  runningPercentage : ℚ

/-- Calculates the total score of a batsman -/
def totalScore (score : BatsmanScore) : ℕ :=
  sorry

theorem batsman_total_score (score : BatsmanScore) 
  (h1 : score.boundaries = 6)
  (h2 : score.sixes = 4)
  (h3 : score.runningPercentage = 60/100) :
  totalScore score = 120 := by
  sorry

end NUMINAMATH_CALUDE_batsman_total_score_l795_79570


namespace NUMINAMATH_CALUDE_problem_statement_l795_79537

theorem problem_statement (x₁ x₂ : ℝ) 
  (h₁ : |x₁ - 2| < 1) 
  (h₂ : |x₂ - 2| < 1) : 
  (2 < x₁ + x₂ ∧ x₁ + x₂ < 6 ∧ |x₁ - x₂| < 2) ∧ 
  (let f := fun x => x^2 - x + 1
   |x₁ - x₂| < |f x₁ - f x₂| ∧ |f x₁ - f x₂| < 5 * |x₁ - x₂|) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l795_79537


namespace NUMINAMATH_CALUDE_floor_equation_solution_l795_79548

theorem floor_equation_solution (a b : ℝ) : 
  (∀ x y : ℝ, ⌊a*x + b*y⌋ + ⌊b*x + a*y⌋ = (a + b)*⌊x + y⌋) ↔ 
  ((a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 1)) :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l795_79548
