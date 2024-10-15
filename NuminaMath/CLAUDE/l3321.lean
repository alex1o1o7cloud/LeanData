import Mathlib

namespace NUMINAMATH_CALUDE_two_digit_number_problem_l3321_332135

/-- Represents a two-digit number as a pair of natural numbers -/
def TwoDigitNumber := Nat × Nat

/-- Converts a two-digit number to its decimal representation -/
def toDecimal (n : TwoDigitNumber) : ℚ :=
  (n.1 * 10 + n.2 : ℚ) / 100

/-- Converts a two-digit number to its repeating decimal representation -/
def toRepeatingDecimal (n : TwoDigitNumber) : ℚ :=
  1 + (n.1 * 10 + n.2 : ℚ) / 99

/-- The problem statement -/
theorem two_digit_number_problem (cd : TwoDigitNumber) :
  55 * (toRepeatingDecimal cd - toDecimal cd) = 1 → cd = (1, 8) := by
  sorry


end NUMINAMATH_CALUDE_two_digit_number_problem_l3321_332135


namespace NUMINAMATH_CALUDE_medal_distribution_theorem_l3321_332128

/-- The number of ways to distribute medals to students -/
def distribute_medals (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of students -/
def num_students : ℕ := 12

/-- The number of medal types -/
def num_medal_types : ℕ := 3

/-- The number of ways to distribute medals -/
def num_distributions : ℕ := distribute_medals (num_students - num_medal_types) num_medal_types

theorem medal_distribution_theorem : num_distributions = 55 := by
  sorry

end NUMINAMATH_CALUDE_medal_distribution_theorem_l3321_332128


namespace NUMINAMATH_CALUDE_system_solutions_l3321_332163

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x * y^2 - 2 * y + 3 * x^2 = 0
def equation2 (x y : ℝ) : Prop := y^2 + x^2 * y + 2 * x = 0

-- Define the set of solutions
def solutions : Set (ℝ × ℝ) := {(-1, 1), (-2 / Real.rpow 3 (1/3), -2 * Real.rpow 3 (1/3)), (0, 0)}

-- Theorem statement
theorem system_solutions :
  ∀ (x y : ℝ), (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solutions :=
sorry

end NUMINAMATH_CALUDE_system_solutions_l3321_332163


namespace NUMINAMATH_CALUDE_product_sum_of_three_numbers_l3321_332159

theorem product_sum_of_three_numbers (a b c : ℝ) 
  (sum_of_squares : a^2 + b^2 + c^2 = 179)
  (sum_of_numbers : a + b + c = 21) :
  a*b + b*c + a*c = 131 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_of_three_numbers_l3321_332159


namespace NUMINAMATH_CALUDE_binomial_8_3_l3321_332120

theorem binomial_8_3 : Nat.choose 8 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_3_l3321_332120


namespace NUMINAMATH_CALUDE_residue_mod_17_l3321_332198

theorem residue_mod_17 : (245 * 15 - 18 * 8 + 5) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_residue_mod_17_l3321_332198


namespace NUMINAMATH_CALUDE_farm_chickens_l3321_332152

/-- Represents the number of roosters initially on the farm. -/
def initial_roosters : ℕ := sorry

/-- Represents the number of hens initially on the farm. -/
def initial_hens : ℕ := 6 * initial_roosters

/-- Represents the number of roosters added to the farm. -/
def added_roosters : ℕ := 60

/-- Represents the number of hens added to the farm. -/
def added_hens : ℕ := 60

/-- Represents the total number of roosters after additions. -/
def final_roosters : ℕ := initial_roosters + added_roosters

/-- Represents the total number of hens after additions. -/
def final_hens : ℕ := initial_hens + added_hens

/-- States that after additions, the number of hens is 4 times the number of roosters. -/
axiom final_ratio : final_hens = 4 * final_roosters

/-- Represents the total number of chickens initially on the farm. -/
def total_chickens : ℕ := initial_roosters + initial_hens

/-- Proves that the total number of chickens initially on the farm was 630. -/
theorem farm_chickens : total_chickens = 630 := by sorry

end NUMINAMATH_CALUDE_farm_chickens_l3321_332152


namespace NUMINAMATH_CALUDE_sixtieth_pair_is_five_seven_l3321_332161

/-- Represents a pair of integers -/
structure IntPair :=
  (first : ℤ)
  (second : ℤ)

/-- Generates the nth pair in the sequence -/
def generateNthPair (n : ℕ) : IntPair :=
  sorry

/-- The sequence of integer pairs as described in the problem -/
def sequencePairs : ℕ → IntPair :=
  generateNthPair

theorem sixtieth_pair_is_five_seven :
  sequencePairs 60 = IntPair.mk 5 7 := by
  sorry

end NUMINAMATH_CALUDE_sixtieth_pair_is_five_seven_l3321_332161


namespace NUMINAMATH_CALUDE_total_plums_picked_l3321_332188

-- Define the number of plums picked by each person
def melanie_plums : ℕ := 4
def dan_plums : ℕ := 9
def sally_plums : ℕ := 3

-- Define Ben's plums as twice the sum of Melanie's and Dan's
def ben_plums : ℕ := 2 * (melanie_plums + dan_plums)

-- Define the number of plums Sally ate
def sally_ate : ℕ := 2

-- Theorem: The total number of plums picked is 40
theorem total_plums_picked : 
  melanie_plums + dan_plums + sally_plums + ben_plums - sally_ate = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_plums_picked_l3321_332188


namespace NUMINAMATH_CALUDE_triangle_area_l3321_332180

theorem triangle_area (A B C : ℝ) (r R : ℝ) (h1 : r = 7) (h2 : R = 25) (h3 : Real.cos B = Real.sin A) :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a + b + c) / 2 * r = 525 * Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l3321_332180


namespace NUMINAMATH_CALUDE_finite_sequence_k_value_l3321_332107

/-- A finite sequence with k terms satisfying the given conditions -/
def FiniteSequence (k : ℕ) (a : ℕ → ℝ) : Prop :=
  (∀ n ∈ Finset.range (k - 2), a (n + 2) = a n - (n + 1) / a (n + 1)) ∧
  a 1 = 24 ∧
  a 2 = 51 ∧
  a k = 0

/-- The theorem stating that k must be 50 for the given conditions -/
theorem finite_sequence_k_value :
  ∀ k : ℕ, ∀ a : ℕ → ℝ, FiniteSequence k a → k = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_finite_sequence_k_value_l3321_332107


namespace NUMINAMATH_CALUDE_new_students_calculation_l3321_332103

/-- Proves that the number of new students is equal to the final number minus
    the difference between the initial number and the number who left. -/
theorem new_students_calculation
  (initial_students : ℕ)
  (students_left : ℕ)
  (final_students : ℕ)
  (h1 : initial_students = 8)
  (h2 : students_left = 5)
  (h3 : final_students = 11) :
  final_students - (initial_students - students_left) = 8 :=
by sorry

end NUMINAMATH_CALUDE_new_students_calculation_l3321_332103


namespace NUMINAMATH_CALUDE_expand_expression_l3321_332151

theorem expand_expression (x y : ℝ) : 24 * (3 * x - 4 * y + 6) = 72 * x - 96 * y + 144 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3321_332151


namespace NUMINAMATH_CALUDE_maria_receives_no_funds_main_result_l3321_332105

/-- Represents the deposit insurance system in rubles -/
def deposit_insurance_threshold : ℕ := 1600000

/-- Represents Maria's deposit amount in rubles -/
def maria_deposit : ℕ := 0  -- We don't know the exact amount, so we use 0 as a placeholder

/-- Theorem stating that Maria will not receive any funds -/
theorem maria_receives_no_funds (h : maria_deposit < deposit_insurance_threshold) :
  maria_deposit = 0 := by
  sorry

/-- Main theorem combining the conditions and the result -/
theorem main_result : 
  maria_deposit < deposit_insurance_threshold → maria_deposit = 0 := by
  sorry

end NUMINAMATH_CALUDE_maria_receives_no_funds_main_result_l3321_332105


namespace NUMINAMATH_CALUDE_bench_and_student_count_l3321_332100

theorem bench_and_student_count :
  ∃ (a b s : ℕ), 
    (s = a * b + 5 ∧ s = 8 * b - 4) →
    ((b = 9 ∧ s = 68) ∨ (b = 3 ∧ s = 20)) := by
  sorry

end NUMINAMATH_CALUDE_bench_and_student_count_l3321_332100


namespace NUMINAMATH_CALUDE_line_passes_through_quadrants_l3321_332156

theorem line_passes_through_quadrants (a b c p : ℝ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : (a + b) / c = p) (h5 : (b + c) / a = p) (h6 : (c + a) / b = p) :
  ∃ (x y : ℝ), x < 0 ∧ y = p * x + p ∧ y < 0 :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_quadrants_l3321_332156


namespace NUMINAMATH_CALUDE_range_of_a_l3321_332173

/-- A function f(x) = -x^2 + 2ax, where a is a real number -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x

/-- The theorem stating the range of a given the conditions on f -/
theorem range_of_a (a : ℝ) :
  (∀ x y, x ∈ Set.Icc 0 1 → y ∈ Set.Icc 0 1 → x < y → f a x < f a y) →
  (∀ x y, x ∈ Set.Icc 2 3 → y ∈ Set.Icc 2 3 → x < y → f a x > f a y) →
  1 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3321_332173


namespace NUMINAMATH_CALUDE_counterexample_exists_l3321_332171

theorem counterexample_exists : ∃ n : ℕ, ¬(Nat.Prime n) ∧ ¬(Nat.Prime (n - 4)) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3321_332171


namespace NUMINAMATH_CALUDE_provisions_last_20_days_l3321_332136

/-- Calculates the number of days provisions will last after reinforcement arrives -/
def daysAfterReinforcement (initialGarrison : ℕ) (initialDays : ℕ) (daysPassed : ℕ) (reinforcement : ℕ) : ℕ :=
  let remainingDays := initialDays - daysPassed
  let totalMen := initialGarrison + reinforcement
  (initialGarrison * remainingDays) / totalMen

/-- Theorem stating that given the initial conditions, the provisions will last 20 more days after reinforcement -/
theorem provisions_last_20_days :
  daysAfterReinforcement 2000 54 15 1900 = 20 := by
  sorry

#eval daysAfterReinforcement 2000 54 15 1900

end NUMINAMATH_CALUDE_provisions_last_20_days_l3321_332136


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l3321_332140

theorem cubic_inequality_solution (x : ℝ) : 
  x^3 - 12*x^2 + 35*x + 48 < 0 ↔ x ∈ Set.Ioo (-1 : ℝ) 3 ∪ Set.Ioi 16 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l3321_332140


namespace NUMINAMATH_CALUDE_volume_ratio_in_partitioned_cube_l3321_332118

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  center : Point3D
  edgeLength : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the volume of a cube -/
def cubeVolume (c : Cube) : ℝ := c.edgeLength ^ 3

/-- Calculates the volume of the part of the cube on one side of a plane -/
noncomputable def volumePartition (c : Cube) (p : Plane) : ℝ := sorry

/-- Theorem: The ratio of volumes in a cube partitioned by a specific plane -/
theorem volume_ratio_in_partitioned_cube (c : Cube) (e f : Point3D) : 
  let p := Plane.mk 1 1 1 0  -- Placeholder plane, actual coefficients would depend on B, E, F
  volumePartition c p / cubeVolume c = 25 / 72 := by sorry

end NUMINAMATH_CALUDE_volume_ratio_in_partitioned_cube_l3321_332118


namespace NUMINAMATH_CALUDE_max_perfect_squares_l3321_332192

theorem max_perfect_squares (a b : ℕ) (h : a ≠ b) : 
  let products := [a * (a + 2), a * b, a * (b + 2), (a + 2) * b, (a + 2) * (b + 2), b * (b + 2)]
  (∃ (n : ℕ), n * n ∈ products) ∧ 
  ¬(∃ (m n : ℕ) (hm : m * m ∈ products) (hn : n * n ∈ products), m ≠ n) :=
by sorry

end NUMINAMATH_CALUDE_max_perfect_squares_l3321_332192


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3321_332142

theorem absolute_value_inequality (x : ℝ) :
  |x - 2| + |x + 3| > 7 ↔ x < -4 ∨ x > 3 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3321_332142


namespace NUMINAMATH_CALUDE_problem_solution_l3321_332193

theorem problem_solution (m : ℚ) : 
  let f (x : ℚ) := 3 * x^3 - 1/x + 2
  let g (x : ℚ) := 2 * x^3 - 3*x + m
  let h (x : ℚ) := x^2
  (f 3 - g 3 + h 3 = 5) → m = 122/3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3321_332193


namespace NUMINAMATH_CALUDE_perfect_square_theorem_l3321_332195

/-- A function that checks if a number is a 3-digit number -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- A function that represents the 6-digit number formed by n and its successor -/
def sixDigitNumber (n : ℕ) : ℕ := 1001 * n + 1

/-- The set of valid 3-digit numbers satisfying the condition -/
def validNumbers : Set ℕ := {183, 328, 528, 715}

/-- Theorem stating that the set of 3-digit numbers n such that 1001n + 1 
    is a perfect square is exactly the set {183, 328, 528, 715} -/
theorem perfect_square_theorem :
  ∀ n : ℕ, isThreeDigit n ∧ (∃ m : ℕ, sixDigitNumber n = m ^ 2) ↔ n ∈ validNumbers := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_theorem_l3321_332195


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l3321_332186

/-- Given two points M and N in a 2D plane, where N is symmetric to M about the y-axis,
    this theorem proves that the coordinates of M with respect to N are (2, 1) when M has coordinates (-2, 1). -/
theorem symmetric_point_coordinates (M N : ℝ × ℝ) :
  M = (-2, 1) →
  N.1 = -M.1 ∧ N.2 = M.2 →
  (M.1 - N.1, M.2 - N.2) = (2, 1) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l3321_332186


namespace NUMINAMATH_CALUDE_temperature_conversion_l3321_332158

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → t = 105 → k = 221 := by sorry

end NUMINAMATH_CALUDE_temperature_conversion_l3321_332158


namespace NUMINAMATH_CALUDE_exterior_angle_regular_octagon_l3321_332126

theorem exterior_angle_regular_octagon :
  ∀ (n : ℕ) (interior_angle exterior_angle : ℝ),
    n = 8 →
    interior_angle = (180 * (n - 2 : ℝ)) / n →
    exterior_angle = 180 - interior_angle →
    exterior_angle = 45 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_regular_octagon_l3321_332126


namespace NUMINAMATH_CALUDE_quadratic_roots_distance_l3321_332115

theorem quadratic_roots_distance (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, x^2 + p*x + q = 0 ↔ (x = x₁ ∨ x = x₂)) ∧
    (x₁ - x₂)^2 = 1) →
  p = 3 ∧ q = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_distance_l3321_332115


namespace NUMINAMATH_CALUDE_certain_number_proof_l3321_332101

theorem certain_number_proof (p q : ℚ) 
  (h1 : 3 / p = 8)
  (h2 : 3 / q = 18)
  (h3 : p - q = 0.20833333333333334) : 
  q = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3321_332101


namespace NUMINAMATH_CALUDE_unique_divisible_by_18_l3321_332104

/-- The function that constructs the four-digit number x47x from a single digit x -/
def construct_number (x : ℕ) : ℕ := 1000 * x + 470 + x

/-- Predicate that checks if a number is a single digit -/
def is_single_digit (x : ℕ) : Prop := x ≥ 0 ∧ x ≤ 9

theorem unique_divisible_by_18 :
  ∃! x : ℕ, is_single_digit x ∧ (construct_number x) % 18 = 0 ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisible_by_18_l3321_332104


namespace NUMINAMATH_CALUDE_harkamal_fruit_purchase_l3321_332168

/-- The total amount paid by Harkamal for his fruit purchase --/
def total_amount : ℕ := by sorry

theorem harkamal_fruit_purchase :
  let grapes_quantity : ℕ := 8
  let grapes_price : ℕ := 80
  let mangoes_quantity : ℕ := 9
  let mangoes_price : ℕ := 55
  let apples_quantity : ℕ := 6
  let apples_price : ℕ := 120
  let oranges_quantity : ℕ := 4
  let oranges_price : ℕ := 75
  total_amount = grapes_quantity * grapes_price +
                 mangoes_quantity * mangoes_price +
                 apples_quantity * apples_price +
                 oranges_quantity * oranges_price :=
by sorry

end NUMINAMATH_CALUDE_harkamal_fruit_purchase_l3321_332168


namespace NUMINAMATH_CALUDE_barbara_candies_l3321_332191

/-- The number of candies Barbara used -/
def candies_used (initial : ℝ) (remaining : ℕ) : ℝ :=
  initial - remaining

theorem barbara_candies : 
  let initial : ℝ := 18.0
  let remaining : ℕ := 9
  candies_used initial remaining = 9 := by
  sorry

end NUMINAMATH_CALUDE_barbara_candies_l3321_332191


namespace NUMINAMATH_CALUDE_right_triangle_with_given_sides_l3321_332166

theorem right_triangle_with_given_sides :
  ∃ (a b c : ℝ), a = 8 ∧ b = 15 ∧ c = Real.sqrt 161 ∧ a^2 + b^2 = c^2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_with_given_sides_l3321_332166


namespace NUMINAMATH_CALUDE_distinct_arrangements_of_basic_l3321_332174

theorem distinct_arrangements_of_basic (n : ℕ) (h : n = 5) : 
  Nat.factorial n = 120 := by
  sorry

end NUMINAMATH_CALUDE_distinct_arrangements_of_basic_l3321_332174


namespace NUMINAMATH_CALUDE_arc_length_from_central_angle_l3321_332122

/-- Given a circle with circumference 80 feet and an arc subtended by a central angle of 120°,
    the length of this arc is 80/3 feet. -/
theorem arc_length_from_central_angle (circle : Real) (arc : Real) :
  (circle = 80) →  -- circumference of the circle is 80 feet
  (arc = 120 / 360 * circle) →  -- arc is subtended by a 120° angle
  (arc = 80 / 3) :=  -- length of the arc is 80/3 feet
by sorry

end NUMINAMATH_CALUDE_arc_length_from_central_angle_l3321_332122


namespace NUMINAMATH_CALUDE_intersection_empty_implies_m_range_l3321_332160

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | x^2 + x + m + 2 = 0}
def B : Set ℝ := {x | x > 0}

-- Theorem statement
theorem intersection_empty_implies_m_range (m : ℝ) :
  A m ∩ B = ∅ → m ≤ -2 := by
  sorry


end NUMINAMATH_CALUDE_intersection_empty_implies_m_range_l3321_332160


namespace NUMINAMATH_CALUDE_beach_trip_seashells_l3321_332179

/-- Calculates the total number of seashells found during a beach trip -/
def total_seashells (days : ℕ) (shells_per_day : ℕ) : ℕ :=
  days * shells_per_day

theorem beach_trip_seashells :
  let days : ℕ := 5
  let shells_per_day : ℕ := 7
  total_seashells days shells_per_day = 35 := by
  sorry

end NUMINAMATH_CALUDE_beach_trip_seashells_l3321_332179


namespace NUMINAMATH_CALUDE_wash_time_is_three_hours_l3321_332102

/-- The number of hours required to wash all clothes given the number of items and washing machine capacity -/
def wash_time (shirts pants sweaters jeans : ℕ) (max_items_per_cycle : ℕ) (minutes_per_cycle : ℕ) : ℚ :=
  let total_items := shirts + pants + sweaters + jeans
  let num_cycles := (total_items + max_items_per_cycle - 1) / max_items_per_cycle
  (num_cycles * minutes_per_cycle : ℚ) / 60

/-- Theorem stating that it takes 3 hours to wash all the clothes under given conditions -/
theorem wash_time_is_three_hours :
  wash_time 18 12 17 13 15 45 = 3 := by
  sorry

end NUMINAMATH_CALUDE_wash_time_is_three_hours_l3321_332102


namespace NUMINAMATH_CALUDE_soccer_league_games_times_each_team_plays_l3321_332181

/-- 
Proves that in a soccer league with 12 teams, where a total of 66 games are played, 
each team plays every other team exactly 2 times.
-/
theorem soccer_league_games (n : ℕ) (total_games : ℕ) (h1 : n = 12) (h2 : total_games = 66) :
  (n * (n - 1) * 2) / 2 = total_games :=
by sorry

/-- 
Proves that the number of times each team plays others is 2.
-/
theorem times_each_team_plays (n : ℕ) (total_games : ℕ) (h1 : n = 12) (h2 : total_games = 66) :
  ∃ x : ℕ, (n * (n - 1) * x) / 2 = total_games ∧ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_soccer_league_games_times_each_team_plays_l3321_332181


namespace NUMINAMATH_CALUDE_tic_tac_toe_winning_probability_l3321_332143

/-- Represents a 3x3 tic-tac-toe board --/
def TicTacToeBoard := Fin 3 → Fin 3 → Bool

/-- The total number of cells on the board --/
def totalCells : Nat := 9

/-- The number of noughts on the board --/
def numNoughts : Nat := 3

/-- The number of crosses on the board --/
def numCrosses : Nat := 6

/-- The number of ways to arrange noughts on the board --/
def totalArrangements : Nat := Nat.choose totalCells numNoughts

/-- The number of winning positions for noughts --/
def winningPositions : Nat := 8

/-- The probability of noughts being in a winning position --/
def winningProbability : ℚ := winningPositions / totalArrangements

theorem tic_tac_toe_winning_probability :
  winningProbability = 2 / 21 := by sorry

end NUMINAMATH_CALUDE_tic_tac_toe_winning_probability_l3321_332143


namespace NUMINAMATH_CALUDE_greatest_x_with_lcm_l3321_332162

theorem greatest_x_with_lcm (x : ℕ+) : 
  (Nat.lcm x (Nat.lcm 15 21) = 105) → x ≤ 105 ∧ ∃ y : ℕ+, y = 105 ∧ Nat.lcm y (Nat.lcm 15 21) = 105 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_with_lcm_l3321_332162


namespace NUMINAMATH_CALUDE_dice_collinearity_probability_l3321_332170

def dice_roll := Finset.range 6

def vector_p (m n : ℕ) := (m, n)
def vector_q := (3, 6)

def is_collinear (p q : ℕ × ℕ) : Prop :=
  p.1 * q.2 = p.2 * q.1

def collinear_outcomes : Finset (ℕ × ℕ) :=
  {(1, 2), (2, 4), (3, 6)}

theorem dice_collinearity_probability :
  (collinear_outcomes.card : ℚ) / (dice_roll.card * dice_roll.card : ℚ) = 1 / 12 :=
sorry

end NUMINAMATH_CALUDE_dice_collinearity_probability_l3321_332170


namespace NUMINAMATH_CALUDE_other_rectangle_perimeter_l3321_332106

/-- Represents the perimeter of a rectangle --/
def rectangle_perimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Represents the side length of the original square --/
def square_side : ℝ := 5

/-- Represents the perimeter of the first rectangle --/
def first_rectangle_perimeter : ℝ := 16

theorem other_rectangle_perimeter :
  ∀ (l w : ℝ),
  l + w = square_side →
  rectangle_perimeter l w = first_rectangle_perimeter →
  rectangle_perimeter square_side (square_side - w) = 14 :=
by sorry

end NUMINAMATH_CALUDE_other_rectangle_perimeter_l3321_332106


namespace NUMINAMATH_CALUDE_garden_perimeter_l3321_332197

/-- The perimeter of a rectangular garden with width 8 meters and the same area as a rectangular playground of length 16 meters and width 12 meters is equal to 64 meters. -/
theorem garden_perimeter : 
  ∀ (garden_length : ℝ),
  garden_length > 0 →
  8 * garden_length = 16 * 12 →
  2 * (garden_length + 8) = 64 := by
sorry

end NUMINAMATH_CALUDE_garden_perimeter_l3321_332197


namespace NUMINAMATH_CALUDE_jack_bought_55_apples_l3321_332146

def apples_for_father : Nat := 10
def number_of_friends : Nat := 4
def apples_per_person : Nat := 9

theorem jack_bought_55_apples : 
  (apples_for_father + (number_of_friends + 1) * apples_per_person) = 55 := by
  sorry

end NUMINAMATH_CALUDE_jack_bought_55_apples_l3321_332146


namespace NUMINAMATH_CALUDE_movie_ticket_change_change_is_nine_l3321_332154

/-- The change received by two sisters after buying movie tickets -/
theorem movie_ticket_change (ticket_cost : ℕ) (money_brought : ℕ) : ℕ :=
  let num_sisters : ℕ := 2
  let total_cost : ℕ := num_sisters * ticket_cost
  money_brought - total_cost

/-- Proof that the change received is $9 -/
theorem change_is_nine :
  movie_ticket_change 8 25 = 9 := by
  sorry

end NUMINAMATH_CALUDE_movie_ticket_change_change_is_nine_l3321_332154


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3321_332131

/-- For a geometric sequence with first term 1, if the first term, the sum of first two terms,
    and 5 form an arithmetic sequence, then the common ratio is 2. -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- a_n is a geometric sequence with common ratio q
  (a 1 = 1) →  -- First term is 1
  (S 2 = a 1 + a 2) →  -- S_2 is the sum of first two terms
  (S 2 - a 1 = 5 - S 2) →  -- a_1, S_2, and 5 form an arithmetic sequence
  q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3321_332131


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3321_332110

theorem geometric_sequence_problem :
  ∀ (a b c d : ℝ),
    (∃ (r : ℝ), r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r) →  -- geometric sequence condition
    a + d = 20 →  -- sum of extreme terms
    b + c = 34 →  -- sum of middle terms
    a^2 + b^2 + c^2 + d^2 = 1300 →  -- sum of squares
    ((a = 16 ∧ b = 8 ∧ c = 4 ∧ d = 2) ∨ (a = 4 ∧ b = 8 ∧ c = 16 ∧ d = 32)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3321_332110


namespace NUMINAMATH_CALUDE_special_ten_digit_count_l3321_332157

/-- A natural number is special if all its digits are different or if changing one digit results in all digits being different. -/
def IsSpecial (n : ℕ) : Prop := sorry

/-- The count of 10-digit numbers. -/
def TenDigitCount : ℕ := 9000000000

/-- The count of special 10-digit numbers. -/
def SpecialTenDigitCount : ℕ := sorry

theorem special_ten_digit_count :
  SpecialTenDigitCount = 414 * Nat.factorial 9 := by sorry

end NUMINAMATH_CALUDE_special_ten_digit_count_l3321_332157


namespace NUMINAMATH_CALUDE_rational_numbers_include_integers_and_fractions_l3321_332113

theorem rational_numbers_include_integers_and_fractions : 
  (∀ n : ℤ, ∃ q : ℚ, (n : ℚ) = q) ∧ 
  (∀ a b : ℤ, b ≠ 0 → ∃ q : ℚ, (a : ℚ) / (b : ℚ) = q) :=
sorry

end NUMINAMATH_CALUDE_rational_numbers_include_integers_and_fractions_l3321_332113


namespace NUMINAMATH_CALUDE_top_three_average_score_l3321_332150

theorem top_three_average_score (total_students : ℕ) (top_students : ℕ) 
  (class_average : ℝ) (score_difference : ℝ) : 
  total_students = 12 →
  top_students = 3 →
  class_average = 85 →
  score_difference = 8 →
  let other_students := total_students - top_students
  let top_average := (total_students * class_average - other_students * (class_average - score_difference)) / top_students
  top_average = 91 := by sorry

end NUMINAMATH_CALUDE_top_three_average_score_l3321_332150


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_squared_l3321_332117

theorem quadratic_roots_difference_squared :
  ∀ a b : ℝ, (2 * a^2 - 8 * a + 6 = 0) → (2 * b^2 - 8 * b + 6 = 0) → (a - b)^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_squared_l3321_332117


namespace NUMINAMATH_CALUDE_quadratic_form_and_sum_l3321_332196

theorem quadratic_form_and_sum (x : ℝ) : ∃! (a b c : ℝ),
  (6 * x^2 + 48 * x + 300 = a * (x + b)^2 + c) ∧
  (a + b + c = 214) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_and_sum_l3321_332196


namespace NUMINAMATH_CALUDE_discriminant_of_quadratic_discriminant_of_specific_quadratic_l3321_332187

theorem discriminant_of_quadratic (a b c : ℝ) : 
  (a ≠ 0) → (b^2 - 4*a*c = (b^2 - 4*a*c)) := by sorry

theorem discriminant_of_specific_quadratic : 
  let a : ℝ := 4
  let b : ℝ := -9
  let c : ℝ := -15
  b^2 - 4*a*c = 321 := by sorry

end NUMINAMATH_CALUDE_discriminant_of_quadratic_discriminant_of_specific_quadratic_l3321_332187


namespace NUMINAMATH_CALUDE_probability_product_216_l3321_332167

def standard_die := Finset.range 6

def roll_product (x y z : ℕ) : ℕ := x * y * z

theorem probability_product_216 :
  (Finset.filter (λ (t : ℕ × ℕ × ℕ) => roll_product t.1 t.2.1 t.2.2 = 216) 
    (standard_die.product (standard_die.product standard_die))).card / 
  (standard_die.card ^ 3 : ℚ) = 1 / 216 :=
sorry

end NUMINAMATH_CALUDE_probability_product_216_l3321_332167


namespace NUMINAMATH_CALUDE_game_theorists_board_size_l3321_332116

/-- Represents the voting process for the game theorists' leadership board. -/
def BoardVotingProcess (initial_members : ℕ) : Prop :=
  ∃ (final_members : ℕ),
    -- The final number of members is less than or equal to the initial number
    final_members ≤ initial_members ∧
    -- The final number of members is of the form 2^n - 1
    ∃ (n : ℕ), final_members = 2^n - 1 ∧
    -- There is no larger number of the form 2^m - 1 that's less than or equal to the initial number
    ∀ (m : ℕ), 2^m - 1 ≤ initial_members → m ≤ n

/-- The theorem stating the result of the voting process for 2020 initial members. -/
theorem game_theorists_board_size :
  BoardVotingProcess 2020 → ∃ (final_members : ℕ), final_members = 1023 :=
by
  sorry


end NUMINAMATH_CALUDE_game_theorists_board_size_l3321_332116


namespace NUMINAMATH_CALUDE_product_of_integers_with_lcm_and_gcd_l3321_332121

theorem product_of_integers_with_lcm_and_gcd (a b : ℕ+) : 
  Nat.lcm a b = 72 → 
  Nat.gcd a b = 8 → 
  (a = 4 * Nat.gcd a b ∨ b = 4 * Nat.gcd a b) → 
  a * b = 576 := by
sorry

end NUMINAMATH_CALUDE_product_of_integers_with_lcm_and_gcd_l3321_332121


namespace NUMINAMATH_CALUDE_tan_theta_minus_pi_over_four_l3321_332109

theorem tan_theta_minus_pi_over_four (θ : ℝ) :
  (Complex.mk (Real.sin θ - 3/5) (Real.cos θ - 4/5)).re = 0 →
  Real.tan (θ - π/4) = -7 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_minus_pi_over_four_l3321_332109


namespace NUMINAMATH_CALUDE_real_part_of_inverse_one_minus_z_squared_l3321_332176

/-- For a complex number z = re^(iθ) where |z| = r ≠ 1 and r > 0, 
    the real part of 1 / (1 - z^2) is (1 - r^2 cos(2θ)) / (1 - 2r^2 cos(2θ) + r^4) -/
theorem real_part_of_inverse_one_minus_z_squared 
  (z : ℂ) (r θ : ℝ) (h1 : z = r * Complex.exp (θ * Complex.I)) 
  (h2 : Complex.abs z = r) (h3 : r ≠ 1) (h4 : r > 0) : 
  (1 / (1 - z^2)).re = (1 - r^2 * Real.cos (2 * θ)) / (1 - 2 * r^2 * Real.cos (2 * θ) + r^4) := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_inverse_one_minus_z_squared_l3321_332176


namespace NUMINAMATH_CALUDE_committee_formation_count_l3321_332134

/-- The number of ways to form a committee with specific requirements -/
theorem committee_formation_count : ∀ (n m k : ℕ),
  n ≥ m ∧ m ≥ k ∧ k ≥ 2 →
  (Nat.choose (n - 2) (k - 2) : ℕ) = Nat.choose n m →
  n = 12 ∧ m = 5 ∧ k = 3 →
  Nat.choose (n - 2) (k - 2) = 120 := by
  sorry

#check committee_formation_count

end NUMINAMATH_CALUDE_committee_formation_count_l3321_332134


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l3321_332177

theorem complex_magnitude_product : 
  Complex.abs ((3 * Real.sqrt 2 - 3 * Complex.I) * (2 * Real.sqrt 5 + 5 * Complex.I)) = 9 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l3321_332177


namespace NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l3321_332175

/-- Given a parabola y² = 2px (p > 0) with focus F, if its directrix intersects 
    the hyperbola y²/3 - x² = 1 at points M and N, and MF is perpendicular to NF, 
    then p = 2√3. -/
theorem parabola_hyperbola_intersection (p : ℝ) (F M N : ℝ × ℝ) : 
  p > 0 →  -- p is positive
  (∀ x y, y^2 = 2*p*x) →  -- equation of parabola
  (∀ x y, y^2/3 - x^2 = 1) →  -- equation of hyperbola
  (M.1 = -p/2 ∧ N.1 = -p/2) →  -- M and N are on the directrix
  (M.2^2/3 - M.1^2 = 1 ∧ N.2^2/3 - N.1^2 = 1) →  -- M and N are on the hyperbola
  ((M.1 - F.1) * (N.1 - F.1) + (M.2 - F.2) * (N.2 - F.2) = 0) →  -- MF ⊥ NF
  p = 2 * Real.sqrt 3 := by
    sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l3321_332175


namespace NUMINAMATH_CALUDE_longest_segment_in_cylinder_l3321_332119

def cylinder_radius : ℝ := 5
def cylinder_height : ℝ := 12

theorem longest_segment_in_cylinder :
  let diameter := 2 * cylinder_radius
  let longest_segment := Real.sqrt (cylinder_height ^ 2 + diameter ^ 2)
  longest_segment = Real.sqrt 244 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_in_cylinder_l3321_332119


namespace NUMINAMATH_CALUDE_open_box_volume_l3321_332164

/-- The volume of an open box constructed from a rectangular sheet --/
def boxVolume (sheetLength sheetWidth x : ℝ) : ℝ :=
  (sheetLength - 2*x) * (sheetWidth - 2*x) * x

theorem open_box_volume :
  ∀ x : ℝ, 1 ≤ x → x ≤ 3 →
  boxVolume 16 12 x = 4*x^3 - 56*x^2 + 192*x :=
by sorry

end NUMINAMATH_CALUDE_open_box_volume_l3321_332164


namespace NUMINAMATH_CALUDE_train_length_specific_train_length_l3321_332149

/-- The length of a train given its speed and time to cross a fixed point -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ := by
  sorry

/-- Proof that a train with speed 144 km/hr crossing a point in 9.99920006399488 seconds has length approximately 399.97 meters -/
theorem specific_train_length : 
  ∃ (length : ℝ), abs (length - train_length 144 9.99920006399488) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_specific_train_length_l3321_332149


namespace NUMINAMATH_CALUDE_ellipse_properties_l3321_332114

/-- Represents an ellipse with semi-major axis a and semi-minor axis 2 -/
structure Ellipse where
  a : ℝ
  h : a > 2

/-- Represents a point on the ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h : x^2 / e.a^2 + y^2 / 4 = 1

/-- The eccentricity of the ellipse -/
def eccentricity (e : Ellipse) : ℝ := sorry

/-- The x-coordinate of the right focus -/
def rightFocusX (e : Ellipse) : ℝ := sorry

/-- Theorem stating the properties of the ellipse and point P -/
theorem ellipse_properties (e : Ellipse) (P : PointOnEllipse e) 
  (h_dist : ∃ (F₁ F₂ : ℝ × ℝ), Real.sqrt ((P.x - F₁.1)^2 + (P.y - F₁.2)^2) + 
                                Real.sqrt ((P.x - F₂.1)^2 + (P.y - F₂.2)^2) = 6)
  (h_perp : ∃ (F₂_x : ℝ), P.x = F₂_x) :
  eccentricity e = Real.sqrt 5 / 3 ∧ 
  rightFocusX e = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3321_332114


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l3321_332125

theorem largest_angle_in_triangle (x : ℝ) : 
  x + 50 + 55 = 180 → 
  max x (max 50 55) = 75 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l3321_332125


namespace NUMINAMATH_CALUDE_fraction_problem_l3321_332124

theorem fraction_problem (x y : ℚ) (h1 : x + y = 3/4) (h2 : x * y = 1/8) : 
  min x y = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3321_332124


namespace NUMINAMATH_CALUDE_travel_distance_theorem_l3321_332153

/-- The total distance Amoli and Anayet need to travel -/
def total_distance (amoli_speed : ℝ) (amoli_time : ℝ) (anayet_speed : ℝ) (anayet_time : ℝ) (remaining_distance : ℝ) : ℝ :=
  amoli_speed * amoli_time + anayet_speed * anayet_time + remaining_distance

/-- Theorem stating the total distance Amoli and Anayet need to travel -/
theorem travel_distance_theorem :
  let amoli_speed : ℝ := 42
  let amoli_time : ℝ := 3
  let anayet_speed : ℝ := 61
  let anayet_time : ℝ := 2
  let remaining_distance : ℝ := 121
  total_distance amoli_speed amoli_time anayet_speed anayet_time remaining_distance = 369 := by
sorry

end NUMINAMATH_CALUDE_travel_distance_theorem_l3321_332153


namespace NUMINAMATH_CALUDE_min_value_triangle_ratio_l3321_332144

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if c - a = 2a * cos B, then the minimum possible value of (3a + c) / b is 2√2. -/
theorem min_value_triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  c - a = 2 * a * Real.cos B →
  ∃ (m : ℝ), m = 2 * Real.sqrt 2 ∧ ∀ (x : ℝ), (3 * a + c) / b ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_triangle_ratio_l3321_332144


namespace NUMINAMATH_CALUDE_cindys_envelopes_l3321_332145

/-- Cindy's envelope problem -/
theorem cindys_envelopes (initial_envelopes : ℕ) (friends : ℕ) (envelopes_per_friend : ℕ) :
  initial_envelopes = 37 →
  friends = 5 →
  envelopes_per_friend = 3 →
  initial_envelopes - friends * envelopes_per_friend = 22 := by
  sorry

#check cindys_envelopes

end NUMINAMATH_CALUDE_cindys_envelopes_l3321_332145


namespace NUMINAMATH_CALUDE_share_of_a_l3321_332138

theorem share_of_a (total : ℝ) (a b c : ℝ) : 
  total = 600 →
  a = (2/3) * (b + c) →
  b = (6/9) * (a + c) →
  a + b + c = total →
  a = 240 := by sorry

end NUMINAMATH_CALUDE_share_of_a_l3321_332138


namespace NUMINAMATH_CALUDE_richmond_victoria_difference_l3321_332199

def richmond_population : ℕ := 3000
def beacon_population : ℕ := 500
def victoria_population : ℕ := 4 * beacon_population

theorem richmond_victoria_difference : 
  richmond_population - victoria_population = 1000 ∧ richmond_population > victoria_population := by
  sorry

end NUMINAMATH_CALUDE_richmond_victoria_difference_l3321_332199


namespace NUMINAMATH_CALUDE_num_valid_schedules_is_336_l3321_332189

/-- Represents the days of the week excluding Saturday -/
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday

/-- Represents the teachers -/
inductive Teacher
| Math
| English
| Other1
| Other2
| Other3
| Other4

/-- A schedule is a function from Teacher to Day -/
def Schedule := Teacher → Day

/-- Predicate to check if a schedule is valid -/
def validSchedule (s : Schedule) : Prop :=
  s Teacher.Math ≠ Day.Monday ∧
  s Teacher.Math ≠ Day.Wednesday ∧
  s Teacher.English ≠ Day.Tuesday ∧
  s Teacher.English ≠ Day.Thursday ∧
  (∀ t1 t2 : Teacher, t1 ≠ t2 → s t1 ≠ s t2)

/-- The number of valid schedules -/
def numValidSchedules : ℕ := sorry

theorem num_valid_schedules_is_336 : numValidSchedules = 336 := by sorry

end NUMINAMATH_CALUDE_num_valid_schedules_is_336_l3321_332189


namespace NUMINAMATH_CALUDE_paul_weekly_spending_l3321_332194

-- Define the given conditions
def lawn_money : ℕ := 3
def weed_eating_money : ℕ := 3
def weeks : ℕ := 2

-- Define the total money earned
def total_money : ℕ := lawn_money + weed_eating_money

-- Define the theorem to prove
theorem paul_weekly_spending :
  total_money / weeks = 3 := by
  sorry

end NUMINAMATH_CALUDE_paul_weekly_spending_l3321_332194


namespace NUMINAMATH_CALUDE_roden_fish_count_l3321_332190

/-- The number of gold fish Roden bought -/
def gold_fish : ℕ := 15

/-- The number of blue fish Roden bought -/
def blue_fish : ℕ := 7

/-- The total number of fish Roden bought -/
def total_fish : ℕ := gold_fish + blue_fish

theorem roden_fish_count : total_fish = 22 := by
  sorry

end NUMINAMATH_CALUDE_roden_fish_count_l3321_332190


namespace NUMINAMATH_CALUDE_apples_in_good_condition_l3321_332147

-- Define the total number of apples
def total_apples : ℕ := 75

-- Define the percentage of rotten apples
def rotten_percentage : ℚ := 12 / 100

-- Define the number of apples in good condition
def good_apples : ℕ := 66

-- Theorem statement
theorem apples_in_good_condition :
  (total_apples : ℚ) * (1 - rotten_percentage) = good_apples := by
  sorry

end NUMINAMATH_CALUDE_apples_in_good_condition_l3321_332147


namespace NUMINAMATH_CALUDE_cube_root_of_product_l3321_332111

theorem cube_root_of_product (a : ℕ) : a^3 = 21 * 25 * 45 * 49 → a = 105 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_product_l3321_332111


namespace NUMINAMATH_CALUDE_radical_product_equality_l3321_332132

theorem radical_product_equality : Real.sqrt 81 * Real.sqrt 16 * (64 ^ (1/4 : ℝ)) = 72 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_radical_product_equality_l3321_332132


namespace NUMINAMATH_CALUDE_ramesh_profit_share_l3321_332133

/-- Calculates the share of profit for a partner in a business partnership -/
def calculateProfitShare (investment1 : ℕ) (investment2 : ℕ) (totalProfit : ℕ) : ℕ :=
  (investment2 * totalProfit) / (investment1 + investment2)

/-- Theorem stating that Ramesh's share of the profit is 11,875 -/
theorem ramesh_profit_share :
  calculateProfitShare 24000 40000 19000 = 11875 := by
  sorry

end NUMINAMATH_CALUDE_ramesh_profit_share_l3321_332133


namespace NUMINAMATH_CALUDE_point_inside_circle_range_l3321_332178

theorem point_inside_circle_range (a : ℝ) : 
  ((1 - a)^2 + (1 + a)^2 < 4) → (-1 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_point_inside_circle_range_l3321_332178


namespace NUMINAMATH_CALUDE_variance_scaling_l3321_332155

-- Define a function to calculate variance
def variance (data : List ℝ) : ℝ := sorry

-- Define a function to scale a list of real numbers
def scaleList (k : ℝ) (data : List ℝ) : List ℝ := sorry

theorem variance_scaling (data : List ℝ) (h : variance data = 0.01) :
  variance (scaleList 10 data) = 1 := by sorry

end NUMINAMATH_CALUDE_variance_scaling_l3321_332155


namespace NUMINAMATH_CALUDE_pear_percentage_difference_l3321_332184

/-- Proves that the percentage difference between canned and poached pears is 20% -/
theorem pear_percentage_difference (total pears_sold pears_canned pears_poached : ℕ) :
  total = 42 →
  pears_sold = 20 →
  pears_poached = pears_sold / 2 →
  total = pears_sold + pears_canned + pears_poached →
  (pears_canned - pears_poached : ℚ) / pears_poached * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_pear_percentage_difference_l3321_332184


namespace NUMINAMATH_CALUDE_picture_books_count_l3321_332172

theorem picture_books_count (total : ℕ) (fiction : ℕ) 
  (h1 : total = 35)
  (h2 : fiction = 5)
  (h3 : total = fiction + (fiction + 4) + (2 * fiction) + picture_books) :
  picture_books = 11 := by
  sorry

end NUMINAMATH_CALUDE_picture_books_count_l3321_332172


namespace NUMINAMATH_CALUDE_perpendicular_vector_scalar_l3321_332165

/-- Given plane vectors a and b, if m * a + b is perpendicular to a, then m = 1 -/
theorem perpendicular_vector_scalar (a b : ℝ × ℝ) (m : ℝ) 
  (h1 : a = (-1, 3)) 
  (h2 : b = (4, -2)) 
  (h3 : (m * a.1 + b.1) * a.1 + (m * a.2 + b.2) * a.2 = 0) : 
  m = 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vector_scalar_l3321_332165


namespace NUMINAMATH_CALUDE_absolute_value_of_z_l3321_332112

theorem absolute_value_of_z (z : ℂ) (h : z^2 = 16 - 30*I) : Complex.abs z = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_z_l3321_332112


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_l3321_332137

/-- A hyperbola with eccentricity √3 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0
  h_ecc : b^2 = 2 * a^2

/-- A line with slope 1 -/
structure Line where
  k : ℝ

/-- Points P and Q on the hyperbola, and R on the y-axis -/
structure Points (h : Hyperbola) (l : Line) where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  h_on_hyperbola : 
    (P.1^2 / h.a^2 - P.2^2 / h.b^2 = 1) ∧
    (Q.1^2 / h.a^2 - Q.2^2 / h.b^2 = 1)
  h_on_line : 
    (P.2 = P.1 + l.k) ∧
    (Q.2 = Q.1 + l.k)
  h_R : R = (0, l.k)
  h_dot_product : P.1 * Q.1 + P.2 * Q.2 = -3
  h_vector_ratio : (Q.1 - P.1, Q.2 - P.2) = (4 * (Q.1 - R.1), 4 * (Q.2 - R.2))

theorem hyperbola_line_intersection 
  (h : Hyperbola) (l : Line) (pts : Points h l) :
  (l.k = 1 ∨ l.k = -1) ∧ h.a = 1 := by sorry

#check hyperbola_line_intersection

end NUMINAMATH_CALUDE_hyperbola_line_intersection_l3321_332137


namespace NUMINAMATH_CALUDE_total_jump_distance_l3321_332185

/-- The total distance jumped by a grasshopper and a frog -/
def total_jump (grasshopper_jump frog_jump : ℕ) : ℕ :=
  grasshopper_jump + frog_jump

/-- Theorem: The total jump distance is 66 inches -/
theorem total_jump_distance :
  total_jump 31 35 = 66 := by
  sorry

end NUMINAMATH_CALUDE_total_jump_distance_l3321_332185


namespace NUMINAMATH_CALUDE_positive_intervals_l3321_332139

-- Define the expression
def f (x : ℝ) : ℝ := (x + 1) * (x - 1) * (x - 2)

-- State the theorem
theorem positive_intervals (x : ℝ) : f x > 0 ↔ x ∈ Set.Ioo (-1) 1 ∪ Set.Ioi 2 :=
sorry

end NUMINAMATH_CALUDE_positive_intervals_l3321_332139


namespace NUMINAMATH_CALUDE_initial_marbles_l3321_332182

-- Define the variables
def marbles_given : ℕ := 14
def marbles_left : ℕ := 50

-- State the theorem
theorem initial_marbles : marbles_given + marbles_left = 64 := by
  sorry

end NUMINAMATH_CALUDE_initial_marbles_l3321_332182


namespace NUMINAMATH_CALUDE_textbook_cost_l3321_332123

/-- Given a textbook sold by a bookstore, prove that the cost to the bookstore
    is $44 when the selling price is $55 and the profit is $11. -/
theorem textbook_cost (selling_price profit : ℕ) (h1 : selling_price = 55) (h2 : profit = 11) :
  selling_price - profit = 44 := by
  sorry

end NUMINAMATH_CALUDE_textbook_cost_l3321_332123


namespace NUMINAMATH_CALUDE_apple_banana_equivalence_l3321_332129

theorem apple_banana_equivalence (apple_value banana_value : ℚ) : 
  (3 / 4 * 12 : ℚ) * apple_value = 10 * banana_value →
  (2 / 3 * 9 : ℚ) * apple_value = (20 / 3 : ℚ) * banana_value := by
  sorry

end NUMINAMATH_CALUDE_apple_banana_equivalence_l3321_332129


namespace NUMINAMATH_CALUDE_sum_and_divide_theorem_l3321_332108

theorem sum_and_divide_theorem (n a : ℕ) (ha : a > 1) :
  let sum := (n * (n + 1)) / 2 - (n / a) * ((n / a) * a + a) / 2
  sum / (a * (a - 1) / 2) = (n / a)^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_divide_theorem_l3321_332108


namespace NUMINAMATH_CALUDE_store_price_difference_l3321_332141

/-- Calculates the final price after applying a discount percentage to a full price -/
def final_price (full_price : ℚ) (discount_percent : ℚ) : ℚ :=
  full_price * (1 - discount_percent / 100)

/-- Proves that Store A's smartphone is $2 cheaper than Store B's after discounts -/
theorem store_price_difference (store_a_full_price store_b_full_price : ℚ)
  (store_a_discount store_b_discount : ℚ)
  (h1 : store_a_full_price = 125)
  (h2 : store_b_full_price = 130)
  (h3 : store_a_discount = 8)
  (h4 : store_b_discount = 10) :
  final_price store_b_full_price store_b_discount -
  final_price store_a_full_price store_a_discount = 2 := by
sorry

end NUMINAMATH_CALUDE_store_price_difference_l3321_332141


namespace NUMINAMATH_CALUDE_ball_hitting_ground_time_l3321_332183

theorem ball_hitting_ground_time : 
  let f (t : ℝ) := -4.9 * t^2 + 4.5 * t + 6
  ∃ t : ℝ, t > 0 ∧ f t = 0 ∧ t = 8121 / 4900 := by
  sorry

end NUMINAMATH_CALUDE_ball_hitting_ground_time_l3321_332183


namespace NUMINAMATH_CALUDE_bowTie_equation_solution_l3321_332127

-- Define the bow tie operation
noncomputable def bowTie (a b : ℝ) : ℝ := a + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- Theorem statement
theorem bowTie_equation_solution (y : ℝ) : bowTie 4 y = 10 → y = 30 := by
  sorry

end NUMINAMATH_CALUDE_bowTie_equation_solution_l3321_332127


namespace NUMINAMATH_CALUDE_ice_cream_cost_theorem_l3321_332169

/-- Ice cream shop prices and orders -/
structure IceCreamShop where
  chocolate_price : ℝ
  vanilla_price : ℝ
  strawberry_price : ℝ
  mint_price : ℝ
  waffle_cone_price : ℝ
  chocolate_chips_price : ℝ
  fudge_price : ℝ
  whipped_cream_price : ℝ

/-- Calculate the cost of Pierre's order -/
def pierre_order_cost (shop : IceCreamShop) : ℝ :=
  2 * shop.chocolate_price + shop.mint_price + shop.waffle_cone_price + shop.chocolate_chips_price

/-- Calculate the cost of Pierre's mother's order -/
def mother_order_cost (shop : IceCreamShop) : ℝ :=
  2 * shop.vanilla_price + shop.strawberry_price + shop.mint_price + 
  shop.waffle_cone_price + shop.fudge_price + shop.whipped_cream_price

/-- The total cost of both orders -/
def total_cost (shop : IceCreamShop) : ℝ :=
  pierre_order_cost shop + mother_order_cost shop

/-- Theorem stating that the total cost is $21.65 -/
theorem ice_cream_cost_theorem (shop : IceCreamShop) 
  (h1 : shop.chocolate_price = 2.50)
  (h2 : shop.vanilla_price = 2.00)
  (h3 : shop.strawberry_price = 2.25)
  (h4 : shop.mint_price = 2.20)
  (h5 : shop.waffle_cone_price = 1.50)
  (h6 : shop.chocolate_chips_price = 1.00)
  (h7 : shop.fudge_price = 1.25)
  (h8 : shop.whipped_cream_price = 0.75) :
  total_cost shop = 21.65 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_cost_theorem_l3321_332169


namespace NUMINAMATH_CALUDE_train_length_equals_distance_traveled_l3321_332148

/-- Calculates the length of a train based on its speed and the time it takes to pass through a tunnel. -/
def train_length (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Theorem stating that the length of a train is equal to the distance it travels while passing through a tunnel. -/
theorem train_length_equals_distance_traveled (speed : ℝ) (time : ℝ) :
  train_length speed time = speed * time :=
by
  sorry

#check train_length_equals_distance_traveled

end NUMINAMATH_CALUDE_train_length_equals_distance_traveled_l3321_332148


namespace NUMINAMATH_CALUDE_possible_k_values_l3321_332130

theorem possible_k_values (a b k : ℕ) (h1 : a > 0) (h2 : b > 0) :
  (b + 1 : ℚ) / a + (a + 1 : ℚ) / b = k →
  k = 3 ∨ k = 4 := by
sorry

end NUMINAMATH_CALUDE_possible_k_values_l3321_332130
