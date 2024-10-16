import Mathlib

namespace NUMINAMATH_CALUDE_unique_years_arithmetic_sequence_l981_98107

/-- A year in the 19th century -/
structure Year19thCentury where
  x : Nat
  y : Nat
  x_range : x ≤ 9
  y_range : y ≤ 9

/-- Check if the differences between adjacent digits form an arithmetic sequence -/
def isArithmeticSequence (year : Year19thCentury) : Prop :=
  ∃ d : Int, (year.x - 8 : Int) - 7 = d ∧ (year.y - year.x : Int) - (year.x - 8) = d

/-- The theorem stating that 1881 and 1894 are the only years satisfying the condition -/
theorem unique_years_arithmetic_sequence :
  ∀ year : Year19thCentury, isArithmeticSequence year ↔ (year.x = 8 ∧ year.y = 1) ∨ (year.x = 9 ∧ year.y = 4) :=
by sorry

end NUMINAMATH_CALUDE_unique_years_arithmetic_sequence_l981_98107


namespace NUMINAMATH_CALUDE_cos_sin_power_eight_identity_l981_98108

theorem cos_sin_power_eight_identity (α : ℝ) : 
  Real.cos α ^ 8 - Real.sin α ^ 8 = Real.cos (2 * α) * ((3 + Real.cos (4 * α)) / 4) := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_power_eight_identity_l981_98108


namespace NUMINAMATH_CALUDE_total_flight_distance_l981_98102

/-- Given the distances between Spain, Russia, and a stopover country, 
    calculate the total distance to fly from the stopover to Russia and back to Spain. -/
theorem total_flight_distance 
  (spain_russia : ℕ) 
  (spain_stopover : ℕ) 
  (h1 : spain_russia = 7019)
  (h2 : spain_stopover = 1615) : 
  spain_stopover + (spain_russia - spain_stopover) + spain_russia = 12423 :=
by sorry

#check total_flight_distance

end NUMINAMATH_CALUDE_total_flight_distance_l981_98102


namespace NUMINAMATH_CALUDE_lcm_24_36_42_l981_98105

theorem lcm_24_36_42 : Nat.lcm (Nat.lcm 24 36) 42 = 504 := by sorry

end NUMINAMATH_CALUDE_lcm_24_36_42_l981_98105


namespace NUMINAMATH_CALUDE_inequality_solution_equivalence_l981_98146

theorem inequality_solution_equivalence (f : ℝ → ℝ) :
  (∃ x : ℝ, f x > 0) ↔ (∃ x₁ : ℝ, f x₁ > 0) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_equivalence_l981_98146


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l981_98145

theorem sum_of_reciprocals (x y : ℚ) 
  (h1 : x⁻¹ + y⁻¹ = 4)
  (h2 : x⁻¹ - y⁻¹ = 8) : 
  x + y = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l981_98145


namespace NUMINAMATH_CALUDE_derek_initial_money_l981_98150

theorem derek_initial_money (initial_money : ℚ) : 
  (initial_money / 2 - (initial_money / 2) / 4 = 360) → initial_money = 960 := by
  sorry

end NUMINAMATH_CALUDE_derek_initial_money_l981_98150


namespace NUMINAMATH_CALUDE_milo_cash_reward_l981_98178

def grades : List ℕ := [2, 2, 2, 3, 3, 3, 3, 4, 5]

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

def cashReward (avg : ℚ) : ℚ := 5 * avg

theorem milo_cash_reward :
  cashReward (average grades) = 15 := by
  sorry

end NUMINAMATH_CALUDE_milo_cash_reward_l981_98178


namespace NUMINAMATH_CALUDE_base_sequences_count_l981_98131

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of different base sequences that can be formed from one base A, two bases C, and three bases G --/
def base_sequences : ℕ :=
  choose 6 1 * choose 5 2 * choose 3 3

theorem base_sequences_count : base_sequences = 60 := by sorry

end NUMINAMATH_CALUDE_base_sequences_count_l981_98131


namespace NUMINAMATH_CALUDE_probability_two_girls_five_tickets_l981_98135

/-- The probability of selecting 2 girls when drawing 5 tickets from a group of 25 students, of which 10 are girls, is 195/506. -/
theorem probability_two_girls_five_tickets (total_students : Nat) (girls : Nat) (tickets : Nat) :
  total_students = 25 →
  girls = 10 →
  tickets = 5 →
  (Nat.choose girls 2 * Nat.choose (total_students - girls) (tickets - 2)) / Nat.choose total_students tickets = 195 / 506 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_girls_five_tickets_l981_98135


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l981_98112

/-- Given a rhombus with diagonals of 10 inches and 24 inches, its perimeter is 52 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l981_98112


namespace NUMINAMATH_CALUDE_parallel_vectors_x_coordinate_l981_98123

/-- Given two parallel vectors a and b in R², prove that the x-coordinate of b is -1/3 -/
theorem parallel_vectors_x_coordinate (a b : ℝ × ℝ) :
  a = (-1, 3) →
  b.snd = 1 →
  ∃ (k : ℝ), k ≠ 0 ∧ a = k • b →
  b.fst = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_coordinate_l981_98123


namespace NUMINAMATH_CALUDE_chewbacca_gum_pack_size_l981_98153

theorem chewbacca_gum_pack_size :
  ∀ x : ℕ,
  (20 : ℚ) / 30 = (20 - x) / 30 →
  (20 : ℚ) / 30 = 20 / (30 + 5 * x) →
  x ≠ 0 →
  x = 14 := by
sorry

end NUMINAMATH_CALUDE_chewbacca_gum_pack_size_l981_98153


namespace NUMINAMATH_CALUDE_egg_problem_solution_l981_98126

/-- Calculates the difference between perfect and cracked eggs given the initial conditions --/
def egg_difference (total_dozens : ℕ) (broken : ℕ) : ℕ :=
  let total := total_dozens * 12
  let cracked := 2 * broken
  let perfect := total - broken - cracked
  perfect - cracked

theorem egg_problem_solution :
  egg_difference 2 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_egg_problem_solution_l981_98126


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l981_98193

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  Complex.im (i^3 / (2*i - 1)) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l981_98193


namespace NUMINAMATH_CALUDE_candies_eaten_l981_98125

theorem candies_eaten (initial : ℕ) (remaining : ℕ) (eaten : ℕ) : 
  initial = 23 → remaining = 7 → eaten = initial - remaining → eaten = 16 := by sorry

end NUMINAMATH_CALUDE_candies_eaten_l981_98125


namespace NUMINAMATH_CALUDE_sample_size_calculation_l981_98195

theorem sample_size_calculation (num_classes : ℕ) (papers_per_class : ℕ) : 
  num_classes = 8 → papers_per_class = 12 → num_classes * papers_per_class = 96 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_calculation_l981_98195


namespace NUMINAMATH_CALUDE_sum_of_squares_l981_98183

theorem sum_of_squares (a b c : ℝ) 
  (sum_zero : a + b + c = 0)
  (sum_products : a * b + a * c + b * c = -3)
  (product : a * b * c = 2) :
  a^2 + b^2 + c^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l981_98183


namespace NUMINAMATH_CALUDE_angle_on_straight_line_l981_98174

/-- Given a straight line ABC with two angles, one measuring 40° and the other measuring x°, 
    prove that the value of x is 140°. -/
theorem angle_on_straight_line (x : ℝ) : 
  x + 40 = 180 → x = 140 := by
  sorry

end NUMINAMATH_CALUDE_angle_on_straight_line_l981_98174


namespace NUMINAMATH_CALUDE_function_satisfies_equation_l981_98197

/-- Prove that for y = e^(x + x^2) + 2e^x, the equation y' - y = 2x e^(x + x^2) holds. -/
theorem function_satisfies_equation (x : ℝ) : 
  let y := Real.exp (x + x^2) + 2 * Real.exp x
  let y' := Real.exp (x + x^2) * (1 + 2*x) + 2 * Real.exp x
  y' - y = 2 * x * Real.exp (x + x^2) := by
sorry


end NUMINAMATH_CALUDE_function_satisfies_equation_l981_98197


namespace NUMINAMATH_CALUDE_initial_speed_is_three_l981_98182

/-- Represents the scenario of two pedestrians walking towards each other --/
structure PedestrianScenario where
  totalDistance : ℝ
  delayDistance : ℝ
  delayTime : ℝ
  meetingDistanceAfterDelay : ℝ
  speedIncrease : ℝ

/-- Calculates the initial speed of the pedestrians --/
def initialSpeed (scenario : PedestrianScenario) : ℝ :=
  sorry

/-- Theorem stating that the initial speed is 3 km/h for the given scenario --/
theorem initial_speed_is_three 
  (scenario : PedestrianScenario) 
  (h1 : scenario.totalDistance = 28)
  (h2 : scenario.delayDistance = 9)
  (h3 : scenario.delayTime = 1)
  (h4 : scenario.meetingDistanceAfterDelay = 4)
  (h5 : scenario.speedIncrease = 1) :
  initialSpeed scenario = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_speed_is_three_l981_98182


namespace NUMINAMATH_CALUDE_star_difference_l981_98190

-- Define the ⋆ operation
def star (x y : ℤ) : ℤ := x * y - 3 * x + 1

-- Theorem statement
theorem star_difference : star 5 3 - star 3 5 = -6 := by sorry

end NUMINAMATH_CALUDE_star_difference_l981_98190


namespace NUMINAMATH_CALUDE_money_distribution_l981_98142

/-- Given three people A, B, and C with money, prove that A and C together have 200 Rs. -/
theorem money_distribution (A B C : ℕ) : 
  A + B + C = 450 →
  B + C = 350 →
  C = 100 →
  A + C = 200 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l981_98142


namespace NUMINAMATH_CALUDE_largest_number_l981_98157

theorem largest_number : ∀ (a b c : ℝ), 
  a = 5 ∧ b = 0 ∧ c = -2 → 
  a > b ∧ a > c ∧ a > -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l981_98157


namespace NUMINAMATH_CALUDE_no_nontrivial_integer_solution_l981_98114

theorem no_nontrivial_integer_solution :
  ∀ (a b c d : ℤ), a^2 - b = c^2 ∧ b^2 - a = d^2 → a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_nontrivial_integer_solution_l981_98114


namespace NUMINAMATH_CALUDE_thabo_books_l981_98121

/-- The number of books Thabo owns -/
def total_books : ℕ := 220

/-- The number of hardcover nonfiction books Thabo owns -/
def hardcover_nonfiction : ℕ := sorry

/-- The number of paperback nonfiction books Thabo owns -/
def paperback_nonfiction : ℕ := sorry

/-- The number of paperback fiction books Thabo owns -/
def paperback_fiction : ℕ := sorry

/-- Theorem stating the conditions and the result to be proved -/
theorem thabo_books :
  (paperback_nonfiction = hardcover_nonfiction + 20) ∧
  (paperback_fiction = 2 * paperback_nonfiction) ∧
  (hardcover_nonfiction + paperback_nonfiction + paperback_fiction = total_books) →
  hardcover_nonfiction = 40 := by
  sorry

end NUMINAMATH_CALUDE_thabo_books_l981_98121


namespace NUMINAMATH_CALUDE_find_d_l981_98143

theorem find_d : ∃ d : ℝ, 
  (∃ x : ℤ, x^2 + 5*x - 36 = 0 ∧ x = ⌊d⌋) ∧ 
  (∃ y : ℝ, 3*y^2 - 11*y + 2 = 0 ∧ y = d - ⌊d⌋) ∧
  d = 13/3 := by
sorry

end NUMINAMATH_CALUDE_find_d_l981_98143


namespace NUMINAMATH_CALUDE_faye_remaining_money_l981_98141

/-- Calculates the remaining money for Faye after her purchases -/
def remaining_money (initial_money : ℚ) (cupcake_price : ℚ) (cupcake_quantity : ℕ) 
  (cookie_box_price : ℚ) (cookie_box_quantity : ℕ) : ℚ :=
  let mother_gift := 2 * initial_money
  let total_money := initial_money + mother_gift
  let cupcake_cost := cupcake_price * cupcake_quantity
  let cookie_cost := cookie_box_price * cookie_box_quantity
  let total_spent := cupcake_cost + cookie_cost
  total_money - total_spent

/-- Theorem stating that Faye's remaining money is $30 -/
theorem faye_remaining_money :
  remaining_money 20 1.5 10 3 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_faye_remaining_money_l981_98141


namespace NUMINAMATH_CALUDE_chess_game_draw_probability_l981_98154

theorem chess_game_draw_probability (p_win p_not_lose : ℝ) 
  (h_win : p_win = 2/5)
  (h_not_lose : p_not_lose = 9/10) :
  p_not_lose - p_win = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_draw_probability_l981_98154


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_3_5_7_l981_98177

theorem smallest_five_digit_divisible_by_3_5_7 :
  ∀ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n → n ≥ 10080 :=
by
  sorry

#check smallest_five_digit_divisible_by_3_5_7

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_3_5_7_l981_98177


namespace NUMINAMATH_CALUDE_impossible_to_fill_board_l981_98129

/-- Represents a piece on the board -/
inductive Piece
  | Regular
  | Special

/-- Represents the color of a square -/
inductive Color
  | White
  | Grey

/-- Represents the board configuration -/
structure Board :=
  (rows : Nat)
  (cols : Nat)
  (total_squares : Nat)
  (white_squares : Nat)
  (grey_squares : Nat)

/-- Represents the coverage of a piece -/
structure PieceCoverage :=
  (white : Nat)
  (grey : Nat)

/-- The board configuration -/
def puzzle_board : Board :=
  { rows := 5
  , cols := 8
  , total_squares := 40
  , white_squares := 20
  , grey_squares := 20 }

/-- The coverage of a regular piece -/
def regular_coverage : PieceCoverage :=
  { white := 2, grey := 2 }

/-- The coverage of the special piece -/
def special_coverage : PieceCoverage :=
  { white := 3, grey := 1 }

/-- The theorem to be proved -/
theorem impossible_to_fill_board : 
  ∀ (special_piece_count : Nat) (regular_piece_count : Nat),
    special_piece_count = 1 →
    regular_piece_count = 9 →
    ¬ (special_piece_count * special_coverage.white + regular_piece_count * regular_coverage.white = puzzle_board.white_squares ∧
       special_piece_count * special_coverage.grey + regular_piece_count * regular_coverage.grey = puzzle_board.grey_squares) :=
by sorry

end NUMINAMATH_CALUDE_impossible_to_fill_board_l981_98129


namespace NUMINAMATH_CALUDE_line_point_a_value_l981_98170

theorem line_point_a_value (k : ℝ) (a : ℝ) :
  k = 0.75 →
  5 = k * a + 1 →
  a = 16/3 := by sorry

end NUMINAMATH_CALUDE_line_point_a_value_l981_98170


namespace NUMINAMATH_CALUDE_worker_count_l981_98134

theorem worker_count (total : ℕ) (extra_total : ℕ) (extra_contribution : ℕ) :
  total = 300000 →
  extra_total = 375000 →
  extra_contribution = 50 →
  ∃ n : ℕ, 
    n * (total / n) = total ∧
    n * (total / n + extra_contribution) = extra_total ∧
    n = 1500 :=
by
  sorry

end NUMINAMATH_CALUDE_worker_count_l981_98134


namespace NUMINAMATH_CALUDE_water_needed_for_lemonade_l981_98140

/-- Given a ratio of water to lemon juice and a total amount of lemonade to make,
    calculate the amount of water needed in quarts. -/
theorem water_needed_for_lemonade 
  (water_ratio : ℚ)
  (lemon_juice_ratio : ℚ)
  (total_gallons : ℚ)
  (quarts_per_gallon : ℚ) :
  water_ratio = 8 →
  lemon_juice_ratio = 1 →
  total_gallons = 3/2 →
  quarts_per_gallon = 4 →
  (water_ratio * total_gallons * quarts_per_gallon) / (water_ratio + lemon_juice_ratio) = 16/3 :=
by
  sorry

end NUMINAMATH_CALUDE_water_needed_for_lemonade_l981_98140


namespace NUMINAMATH_CALUDE_sqrt_relation_l981_98147

theorem sqrt_relation (h : Real.sqrt 262.44 = 16.2) : Real.sqrt 2.6244 = 1.62 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_relation_l981_98147


namespace NUMINAMATH_CALUDE_all_numbers_in_first_hundred_l981_98173

/-- Represents the color of a number in the sequence -/
inductive Color
| Blue
| Red

/-- Represents the sequence of 200 numbers with their colors -/
def Sequence := Fin 200 → (ℕ × Color)

/-- The blue numbers form a sequence from 1 to 100 in ascending order -/
def blue_ascending (s : Sequence) : Prop :=
  ∀ i j : Fin 200, i < j →
    (s i).2 = Color.Blue → (s j).2 = Color.Blue →
      (s i).1 < (s j).1

/-- The red numbers form a sequence from 100 to 1 in descending order -/
def red_descending (s : Sequence) : Prop :=
  ∀ i j : Fin 200, i < j →
    (s i).2 = Color.Red → (s j).2 = Color.Red →
      (s i).1 > (s j).1

/-- The blue numbers are all natural numbers from 1 to 100 -/
def blue_range (s : Sequence) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 →
    ∃ i : Fin 200, (s i).1 = n ∧ (s i).2 = Color.Blue

/-- The red numbers are all natural numbers from 1 to 100 -/
def red_range (s : Sequence) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 →
    ∃ i : Fin 200, (s i).1 = n ∧ (s i).2 = Color.Red

theorem all_numbers_in_first_hundred (s : Sequence)
  (h1 : blue_ascending s)
  (h2 : red_descending s)
  (h3 : blue_range s)
  (h4 : red_range s) :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 →
    ∃ i : Fin 100, (s i).1 = n :=
by sorry

end NUMINAMATH_CALUDE_all_numbers_in_first_hundred_l981_98173


namespace NUMINAMATH_CALUDE_no_solutions_equation_l981_98186

theorem no_solutions_equation (x y : ℕ+) : x * (x + 1) ≠ 4 * y * (y + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_equation_l981_98186


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l981_98120

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ
  is_arithmetic : ∀ n : ℕ+, a (n + 1) - a n = a 2 - a 1
  sum_condition : a 1 + a 3 + a 8 = 99
  fifth_term : a 5 = 31

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The theorem to be proven -/
theorem arithmetic_sequence_max_sum (seq : ArithmeticSequence) :
  ∃ k : ℕ+, ∀ n : ℕ+, S seq n ≤ S seq k ∧ k = 20 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l981_98120


namespace NUMINAMATH_CALUDE_greatest_integer_jo_l981_98136

theorem greatest_integer_jo (n : ℕ) : n < 150 → 
  (∃ k : ℤ, n = 9 * k - 1) → 
  (∃ l : ℤ, n = 6 * l - 5) → 
  n ≤ 125 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_jo_l981_98136


namespace NUMINAMATH_CALUDE_mcq_options_count_l981_98180

theorem mcq_options_count (p_all_correct : ℚ) (p_tf_correct : ℚ) (n : ℕ) : 
  p_all_correct = 1 / 12 →
  p_tf_correct = 1 / 2 →
  (1 / n : ℚ) * p_tf_correct * p_tf_correct = p_all_correct →
  n = 3 := by
  sorry

end NUMINAMATH_CALUDE_mcq_options_count_l981_98180


namespace NUMINAMATH_CALUDE_negative_128_squared_div_64_l981_98163

theorem negative_128_squared_div_64 : ((-128)^2) / 64 = 256 := by sorry

end NUMINAMATH_CALUDE_negative_128_squared_div_64_l981_98163


namespace NUMINAMATH_CALUDE_mock_exam_girls_count_l981_98128

theorem mock_exam_girls_count :
  ∀ (total_students : ℕ) (boys girls : ℕ) (boys_cleared girls_cleared : ℕ),
    total_students = 400 →
    boys + girls = total_students →
    boys_cleared = (60 * boys) / 100 →
    girls_cleared = (80 * girls) / 100 →
    boys_cleared + girls_cleared = (65 * total_students) / 100 →
    girls = 100 := by
  sorry

end NUMINAMATH_CALUDE_mock_exam_girls_count_l981_98128


namespace NUMINAMATH_CALUDE_union_equals_A_l981_98124

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Define set B
def B : Set ℝ := {-1, 0, 1, 2, 3}

-- Theorem statement
theorem union_equals_A : A ∪ B = A := by
  sorry

end NUMINAMATH_CALUDE_union_equals_A_l981_98124


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_of_36_l981_98191

theorem smallest_non_factor_product_of_36 (x y : ℕ+) : 
  x ≠ y →
  x ∣ 36 →
  y ∣ 36 →
  ¬(x * y ∣ 36) →
  (∀ a b : ℕ+, a ≠ b → a ∣ 36 → b ∣ 36 → ¬(a * b ∣ 36) → x * y ≤ a * b) →
  x * y = 8 := by
sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_of_36_l981_98191


namespace NUMINAMATH_CALUDE_b_2017_equals_1_l981_98194

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def b (n : ℕ) : ℕ := fibonacci n % 3

theorem b_2017_equals_1 : b 2017 = 1 := by sorry

end NUMINAMATH_CALUDE_b_2017_equals_1_l981_98194


namespace NUMINAMATH_CALUDE_expression_equals_minus_seven_l981_98151

theorem expression_equals_minus_seven :
  Real.sqrt 8 + (1/2)⁻¹ - 4 * Real.cos (π/4) - 2 / (1/2) * 2 - (2009 - Real.sqrt 3)^0 = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_minus_seven_l981_98151


namespace NUMINAMATH_CALUDE_tinas_fourth_hour_coins_verify_final_coins_l981_98130

/-- Represents the number of coins in Tina's jar at different stages -/
structure CoinJar where
  initial : ℕ := 0
  first_hour : ℕ
  second_third_hours : ℕ
  fourth_hour : ℕ
  fifth_hour : ℕ

/-- The coin jar problem setup -/
def tinas_jar : CoinJar :=
  { first_hour := 20
  , second_third_hours := 60
  , fourth_hour := 40  -- This is what we want to prove
  , fifth_hour := 100 }

/-- Theorem stating that the number of coins Tina put in during the fourth hour is 40 -/
theorem tinas_fourth_hour_coins :
  tinas_jar.fourth_hour = 40 :=
by
  -- The actual proof would go here
  sorry

/-- Verify that the final number of coins matches the problem statement -/
theorem verify_final_coins :
  tinas_jar.first_hour + tinas_jar.second_third_hours + tinas_jar.fourth_hour - 20 = tinas_jar.fifth_hour :=
by
  -- The actual proof would go here
  sorry

end NUMINAMATH_CALUDE_tinas_fourth_hour_coins_verify_final_coins_l981_98130


namespace NUMINAMATH_CALUDE_larger_integer_problem_l981_98199

theorem larger_integer_problem (a b : ℕ+) : 
  (a : ℚ) / (b : ℚ) = 7 / 3 → 
  a * b = 189 → 
  max a b = 21 := by
sorry

end NUMINAMATH_CALUDE_larger_integer_problem_l981_98199


namespace NUMINAMATH_CALUDE_min_value_of_u_l981_98106

theorem min_value_of_u (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h : 2 * x + y = 6) :
  ∃ (min_u : ℝ), min_u = 27 / 2 ∧ ∀ (u : ℝ), u = 4 * x^2 + 3 * x * y + y^2 - 6 * x - 3 * y → u ≥ min_u :=
sorry

end NUMINAMATH_CALUDE_min_value_of_u_l981_98106


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l981_98138

theorem least_subtraction_for_divisibility : ∃! x : ℕ, 
  (∀ y : ℕ, y < x → ¬((50248 - y) % 20 = 0 ∧ (50248 - y) % 37 = 0)) ∧ 
  (50248 - x) % 20 = 0 ∧ 
  (50248 - x) % 37 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l981_98138


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l981_98176

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n, a n > 0) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sum_l981_98176


namespace NUMINAMATH_CALUDE_tournament_games_theorem_l981_98155

/-- A single-elimination tournament with no ties -/
structure Tournament :=
  (num_teams : ℕ)
  (no_ties : Bool)

/-- The number of games required to declare a winner in a tournament -/
def games_to_winner (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- Theorem: In a single-elimination tournament with 24 teams and no ties,
    the number of games required to declare a winner is 23 -/
theorem tournament_games_theorem (t : Tournament) 
  (h1 : t.num_teams = 24) (h2 : t.no_ties = true) : 
  games_to_winner t = 23 := by
  sorry

end NUMINAMATH_CALUDE_tournament_games_theorem_l981_98155


namespace NUMINAMATH_CALUDE_bean_in_circle_probability_l981_98117

/-- The probability of a randomly thrown bean landing inside the inscribed circle of an equilateral triangle with side length 2 -/
theorem bean_in_circle_probability : 
  let triangle_side : ℝ := 2
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side^2
  let circle_radius : ℝ := (Real.sqrt 3 / 3) * triangle_side
  let circle_area : ℝ := Real.pi * circle_radius^2
  let probability : ℝ := circle_area / triangle_area
  probability = (Real.sqrt 3 * Real.pi) / 9 := by
sorry

end NUMINAMATH_CALUDE_bean_in_circle_probability_l981_98117


namespace NUMINAMATH_CALUDE_not_true_from_false_premises_l981_98101

theorem not_true_from_false_premises (p q : Prop) : 
  ¬ (∀ (p q : Prop), (p → q) → (¬p → q)) :=
sorry

end NUMINAMATH_CALUDE_not_true_from_false_premises_l981_98101


namespace NUMINAMATH_CALUDE_f_properties_l981_98165

noncomputable def f (x : ℝ) : ℝ := 1/2 - 1/(2^x + 1)

theorem f_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → f x ∈ Set.Icc (1/6) (3/10)) ∧
  f 1 = 1/6 ∧ f 2 = 3/10 :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l981_98165


namespace NUMINAMATH_CALUDE_parabola_intersection_value_l981_98172

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - x - 1

-- Define the theorem
theorem parabola_intersection_value :
  ∀ m : ℝ, parabola m = 0 → -2 * m^2 + 2 * m + 2023 = 2021 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_value_l981_98172


namespace NUMINAMATH_CALUDE_complex_magnitude_l981_98103

theorem complex_magnitude (z : ℂ) (h : z - 2 + Complex.I = 1) : Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l981_98103


namespace NUMINAMATH_CALUDE_rectangular_plot_dimensions_l981_98144

theorem rectangular_plot_dimensions (area : ℝ) (fence_length : ℝ) :
  area = 800 ∧ fence_length = 100 →
  ∃ (length width : ℝ),
    (length * width = area ∧
     2 * length + width = fence_length) ∧
    ((length = 40 ∧ width = 20) ∨ (length = 10 ∧ width = 80)) := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_dimensions_l981_98144


namespace NUMINAMATH_CALUDE_rick_has_two_sisters_l981_98181

/-- Calculates the number of Rick's sisters based on the given card distribution. -/
def number_of_sisters (total_cards : ℕ) (kept_cards : ℕ) (miguel_cards : ℕ) 
  (friends : ℕ) (cards_per_friend : ℕ) (cards_per_sister : ℕ) : ℕ :=
  let remaining_cards := total_cards - kept_cards - miguel_cards - (friends * cards_per_friend)
  remaining_cards / cards_per_sister

/-- Theorem stating that Rick has 2 sisters given the card distribution. -/
theorem rick_has_two_sisters :
  number_of_sisters 130 15 13 8 12 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_rick_has_two_sisters_l981_98181


namespace NUMINAMATH_CALUDE_proposition_truth_l981_98104

theorem proposition_truth (p q : Prop) (hp : p) (hq : ¬q) : (¬p) ∨ (¬q) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_l981_98104


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l981_98118

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_12th_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 7 + a 9 = 16) 
  (h_4th : a 4 = 1) : 
  a 12 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l981_98118


namespace NUMINAMATH_CALUDE_frog_max_hop_sum_l981_98179

/-- The maximum sum of hop lengths for a frog hopping on integers -/
theorem frog_max_hop_sum (n : ℕ+) : 
  ∃ (S : ℕ), S = (4^n.val - 1) / 3 ∧ 
  ∀ (hop_lengths : List ℕ), 
    (∀ l ∈ hop_lengths, ∃ k : ℕ, l = 2^k) →
    (∀ p ∈ List.range (2^n.val), List.count p (List.scanl (λ acc x => (acc + x) % (2^n.val)) 0 hop_lengths) ≤ 1) →
    List.sum hop_lengths ≤ S :=
sorry

end NUMINAMATH_CALUDE_frog_max_hop_sum_l981_98179


namespace NUMINAMATH_CALUDE_farm_animals_difference_l981_98109

theorem farm_animals_difference (initial_horses : ℕ) (initial_cows : ℕ) : 
  initial_horses = 4 * initial_cows →  -- Initial ratio condition
  (initial_horses - 15) / (initial_cows + 15) = 13 / 7 →  -- New ratio condition
  initial_horses - 15 - (initial_cows + 15) = 30 :=  -- Difference after transaction
by
  sorry

end NUMINAMATH_CALUDE_farm_animals_difference_l981_98109


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_l981_98127

theorem quadratic_real_solutions (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 4*x - 1 = 0) ↔ a ≥ -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_l981_98127


namespace NUMINAMATH_CALUDE_inscribed_polygon_existence_l981_98184

/-- Represents a line in a plane --/
structure Line where
  -- Add necessary fields for a line

/-- Represents a circle in a plane --/
structure Circle where
  -- Add necessary fields for a circle

/-- Represents a polygon --/
structure Polygon where
  -- Add necessary fields for a polygon

/-- Function to check if two lines are parallel --/
def are_parallel (l1 l2 : Line) : Prop :=
  sorry

/-- Function to check if a polygon is inscribed in a circle --/
def is_inscribed (p : Polygon) (c : Circle) : Prop :=
  sorry

/-- Function to check if a polygon side is parallel to a given line --/
def side_parallel_to_line (p : Polygon) (l : Line) : Prop :=
  sorry

/-- Main theorem statement --/
theorem inscribed_polygon_existence 
  (c : Circle) (lines : List Line) (n : Nat) 
  (h1 : lines.length = n)
  (h2 : ∀ (l1 l2 : Line), l1 ∈ lines → l2 ∈ lines → l1 ≠ l2 → ¬(are_parallel l1 l2)) :
  (n % 2 = 0 → 
    (∃ (p : Polygon), is_inscribed p c ∧ 
      (∀ (side : Line), side_parallel_to_line p side → side ∈ lines)) ∨
    (∀ (p : Polygon), ¬(is_inscribed p c ∧ 
      (∀ (side : Line), side_parallel_to_line p side → side ∈ lines)))) ∧
  (n % 2 = 1 → 
    ∃! (p1 p2 : Polygon), p1 ≠ p2 ∧ 
      is_inscribed p1 c ∧ is_inscribed p2 c ∧
      (∀ (side : Line), side_parallel_to_line p1 side → side ∈ lines) ∧
      (∀ (side : Line), side_parallel_to_line p2 side → side ∈ lines)) :=
sorry

end NUMINAMATH_CALUDE_inscribed_polygon_existence_l981_98184


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_2beta_l981_98122

theorem cos_2alpha_plus_2beta (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1/3) 
  (h2 : Real.cos α * Real.sin β = 1/6) : 
  Real.cos (2*α + 2*β) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_2beta_l981_98122


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l981_98119

/-- Given that x is inversely proportional to y, prove that if x = 5 when y = 10, then x = -25/2 when y = -4 -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : 5 * 10 = k) :
  -4 * x = k → x = -25/2 := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l981_98119


namespace NUMINAMATH_CALUDE_cookies_left_l981_98198

theorem cookies_left (whole_cookies : ℕ) (greg_ate : ℕ) (brad_ate : ℕ) : 
  whole_cookies = 14 → greg_ate = 4 → brad_ate = 6 → 
  whole_cookies * 2 - (greg_ate + brad_ate) = 18 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_l981_98198


namespace NUMINAMATH_CALUDE_cylinder_height_l981_98196

theorem cylinder_height (r h : ℝ) : 
  r = 4 →
  2 * Real.pi * r^2 + 2 * Real.pi * r * h = 40 * Real.pi →
  h = 1 := by
sorry

end NUMINAMATH_CALUDE_cylinder_height_l981_98196


namespace NUMINAMATH_CALUDE_smith_family_laundry_l981_98185

/-- The number of bath towels that can fit in one load of laundry for the Smith family. -/
def towels_per_load (kylie_towels : ℕ) (daughters_towels : ℕ) (husband_towels : ℕ) (total_loads : ℕ) : ℕ :=
  (kylie_towels + daughters_towels + husband_towels) / total_loads

/-- Theorem stating that the washing machine can fit 4 bath towels in one load of laundry. -/
theorem smith_family_laundry :
  towels_per_load 3 6 3 3 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_smith_family_laundry_l981_98185


namespace NUMINAMATH_CALUDE_max_investment_at_lower_rate_l981_98159

theorem max_investment_at_lower_rate
  (total_investment : ℝ)
  (lower_rate : ℝ)
  (higher_rate : ℝ)
  (min_interest : ℝ)
  (h1 : total_investment = 25000)
  (h2 : lower_rate = 0.07)
  (h3 : higher_rate = 0.12)
  (h4 : min_interest = 2450)
  : ∃ (x : ℝ), x ≤ 11000 ∧
    x + (total_investment - x) = total_investment ∧
    lower_rate * x + higher_rate * (total_investment - x) ≥ min_interest ∧
    ∀ (y : ℝ), y > x →
      lower_rate * y + higher_rate * (total_investment - y) < min_interest :=
by sorry


end NUMINAMATH_CALUDE_max_investment_at_lower_rate_l981_98159


namespace NUMINAMATH_CALUDE_fuel_cost_calculation_l981_98148

theorem fuel_cost_calculation (original_cost : ℝ) (capacity_increase : ℝ) (price_increase : ℝ) : 
  original_cost = 200 → 
  capacity_increase = 2 → 
  price_increase = 1.2 → 
  original_cost * capacity_increase * price_increase = 480 := by
sorry

end NUMINAMATH_CALUDE_fuel_cost_calculation_l981_98148


namespace NUMINAMATH_CALUDE_rose_mother_age_ratio_l981_98161

/-- Represents the ratio of two ages -/
structure AgeRatio where
  numerator : ℕ
  denominator : ℕ

/-- Rose's age in years -/
def rose_age : ℕ := 25

/-- Rose's mother's age in years -/
def mother_age : ℕ := 75

/-- The ratio of Rose's age to her mother's age -/
def rose_to_mother_ratio : AgeRatio := ⟨1, 3⟩

/-- Theorem stating that the ratio of Rose's age to her mother's age is 1:3 -/
theorem rose_mother_age_ratio : 
  (rose_age : ℚ) / (mother_age : ℚ) = (rose_to_mother_ratio.numerator : ℚ) / (rose_to_mother_ratio.denominator : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_rose_mother_age_ratio_l981_98161


namespace NUMINAMATH_CALUDE_tetrahedron_side_sum_squares_l981_98164

/-- A tetrahedron with side lengths a, b, c and circumradius 1 -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  circumradius : ℝ
  circumradius_eq_one : circumradius = 1

/-- The sum of squares of the side lengths of the tetrahedron is 8 -/
theorem tetrahedron_side_sum_squares (t : Tetrahedron) : t.a^2 + t.b^2 + t.c^2 = 8 := by
  sorry


end NUMINAMATH_CALUDE_tetrahedron_side_sum_squares_l981_98164


namespace NUMINAMATH_CALUDE_sports_activity_division_l981_98133

theorem sports_activity_division :
  ∀ (a b c : ℕ),
    a + b + c = 48 →
    a ≠ b ∧ b ≠ c ∧ a ≠ c →
    (∃ (x : ℕ), a = 10 * x + 6) →
    (∃ (y : ℕ), b = 10 * y + 6) →
    (∃ (z : ℕ), c = 10 * z + 6) →
    (a = 6 ∧ b = 16 ∧ c = 26) ∨ (a = 6 ∧ b = 26 ∧ c = 16) ∨
    (a = 16 ∧ b = 6 ∧ c = 26) ∨ (a = 16 ∧ b = 26 ∧ c = 6) ∨
    (a = 26 ∧ b = 6 ∧ c = 16) ∨ (a = 26 ∧ b = 16 ∧ c = 6) :=
by sorry


end NUMINAMATH_CALUDE_sports_activity_division_l981_98133


namespace NUMINAMATH_CALUDE_angle_C_is_60_degrees_angle_B_and_area_l981_98100

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem -/
def given_condition (t : Triangle) : Prop :=
  ((t.a + t.b)^2 - t.c^2) / (3 * t.a * t.b) = 1

/-- Part 1 of the theorem -/
theorem angle_C_is_60_degrees (t : Triangle) (h : given_condition t) :
  t.C = Real.pi / 3 := by sorry

/-- Part 2 of the theorem -/
theorem angle_B_and_area (t : Triangle) 
  (h1 : t.c = Real.sqrt 3) 
  (h2 : t.b = Real.sqrt 2) 
  (h3 : t.C = Real.pi / 3) :
  t.B = Real.pi / 4 ∧ 
  (1/2 * t.b * t.c * Real.sin t.A) = (3 + Real.sqrt 3) / 4 := by sorry

end NUMINAMATH_CALUDE_angle_C_is_60_degrees_angle_B_and_area_l981_98100


namespace NUMINAMATH_CALUDE_hyperbola_triangle_l981_98115

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x * y = 1

-- Define the branches of the hyperbola
def C₁ (x y : ℝ) : Prop := hyperbola x y ∧ x > 0
def C₂ (x y : ℝ) : Prop := hyperbola x y ∧ x < 0

-- Define a regular triangle
def regular_triangle (P Q R : ℝ × ℝ) : Prop :=
  let (px, py) := P
  let (qx, qy) := Q
  let (rx, ry) := R
  (px - qx)^2 + (py - qy)^2 = (qx - rx)^2 + (qy - ry)^2 ∧
  (qx - rx)^2 + (qy - ry)^2 = (rx - px)^2 + (ry - py)^2

-- Theorem statement
theorem hyperbola_triangle :
  ∀ (Q R : ℝ × ℝ),
  let P := (-1, -1)
  regular_triangle P Q R ∧
  C₂ P.1 P.2 ∧
  C₁ Q.1 Q.2 ∧
  C₁ R.1 R.2 →
  (¬(C₁ P.1 P.2 ∧ C₁ Q.1 Q.2 ∧ C₁ R.1 R.2) ∧
   ¬(C₂ P.1 P.2 ∧ C₂ Q.1 Q.2 ∧ C₂ R.1 R.2)) ∧
  Q = (2 - Real.sqrt 3, 2 + Real.sqrt 3) ∧
  R = (2 + Real.sqrt 3, 2 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_triangle_l981_98115


namespace NUMINAMATH_CALUDE_range_of_a_l981_98188

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ 2 * x > a - x^2) → a < 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l981_98188


namespace NUMINAMATH_CALUDE_total_spent_is_122_80_l981_98166

-- Define the cost per deck
def cost_per_deck : ℚ := 8

-- Define the number of decks bought by each person
def victor_decks : ℕ := 6
def friend_a_decks : ℕ := 4
def friend_b_decks : ℕ := 5
def friend_c_decks : ℕ := 3

-- Define the discount rates
def discount_rate (n : ℕ) : ℚ :=
  if n ≥ 6 then 0.20
  else if n = 5 then 0.15
  else if n ≥ 3 then 0.10
  else 0

-- Define the function to calculate the total cost for a person
def total_cost (decks : ℕ) : ℚ :=
  let base_cost := cost_per_deck * decks
  base_cost - (base_cost * discount_rate decks)

-- Theorem statement
theorem total_spent_is_122_80 :
  total_cost victor_decks +
  total_cost friend_a_decks +
  total_cost friend_b_decks +
  total_cost friend_c_decks = 122.80 :=
by sorry

end NUMINAMATH_CALUDE_total_spent_is_122_80_l981_98166


namespace NUMINAMATH_CALUDE_stamp_cost_correct_l981_98139

/-- The cost of one stamp, given that three stamps cost $1.02 and the cost is constant -/
def stamp_cost : ℚ := 0.34

/-- The cost of three stamps -/
def three_stamps_cost : ℚ := 1.02

/-- Theorem stating that the cost of one stamp is correct -/
theorem stamp_cost_correct : 3 * stamp_cost = three_stamps_cost := by sorry

end NUMINAMATH_CALUDE_stamp_cost_correct_l981_98139


namespace NUMINAMATH_CALUDE_tylenol_consumption_l981_98116

/-- Calculates the total grams of Tylenol taken given the dosage and duration -/
def totalTylenolGrams (tabletsPer4Hours : ℕ) (mgPerTablet : ℕ) (totalHours : ℕ) : ℚ :=
  let dosesCount := totalHours / 4
  let totalTablets := dosesCount * tabletsPer4Hours
  let totalMg := totalTablets * mgPerTablet
  (totalMg : ℚ) / 1000

/-- Theorem stating that under the given conditions, 3 grams of Tylenol are taken -/
theorem tylenol_consumption : totalTylenolGrams 2 500 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tylenol_consumption_l981_98116


namespace NUMINAMATH_CALUDE_lunch_break_duration_l981_98110

/-- Represents the painting scenario with Paula and her helpers --/
structure PaintingScenario where
  paula_rate : ℝ
  helpers_rate : ℝ
  lunch_break : ℝ

/-- Conditions of the painting scenario --/
def painting_conditions (s : PaintingScenario) : Prop :=
  -- Monday's work
  (9 - s.lunch_break) * (s.paula_rate + s.helpers_rate) = 0.4 ∧
  -- Tuesday's work
  (8 - s.lunch_break) * s.helpers_rate = 0.33 ∧
  -- Wednesday's work
  (12 - s.lunch_break) * s.paula_rate = 0.27

/-- The main theorem: lunch break duration is 420 minutes --/
theorem lunch_break_duration (s : PaintingScenario) :
  painting_conditions s → s.lunch_break * 60 = 420 := by
  sorry

end NUMINAMATH_CALUDE_lunch_break_duration_l981_98110


namespace NUMINAMATH_CALUDE_investor_share_calculation_l981_98192

/-- Calculates the share of profit for an investor given the investments and time periods. -/
theorem investor_share_calculation
  (investment_a investment_b total_profit : ℚ)
  (time_a time_b total_time : ℕ)
  (h1 : investment_a = 150)
  (h2 : investment_b = 200)
  (h3 : total_profit = 100)
  (h4 : time_a = 12)
  (h5 : time_b = 6)
  (h6 : total_time = 12)
  : (investment_a * time_a) / ((investment_a * time_a) + (investment_b * time_b)) * total_profit = 60 := by
  sorry

#check investor_share_calculation

end NUMINAMATH_CALUDE_investor_share_calculation_l981_98192


namespace NUMINAMATH_CALUDE_sum_always_six_digits_l981_98160

def first_number : Nat := 98765

def second_number (C : Nat) : Nat := C * 1000 + 433

def third_number (D : Nat) : Nat := D * 100 + 22

def is_nonzero_digit (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 9

theorem sum_always_six_digits (C D : Nat) 
  (hC : is_nonzero_digit C) (hD : is_nonzero_digit D) : 
  ∃ (n : Nat), 100000 ≤ first_number + second_number C + third_number D ∧ 
               first_number + second_number C + third_number D < 1000000 :=
sorry

end NUMINAMATH_CALUDE_sum_always_six_digits_l981_98160


namespace NUMINAMATH_CALUDE_polynomial_root_sum_l981_98167

theorem polynomial_root_sum (a b c d e : ℝ) : 
  a ≠ 0 → 
  (∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ x = 5 ∨ x = -3 ∨ x = 2 ∨ x = (-(b + c + d) / a - 4)) →
  (b + c + d) / a = -6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_l981_98167


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l981_98132

/-- Two vectors in R² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- Given vectors a = (-2, 3) and b = (3, m) are perpendicular, prove that m = 2 -/
theorem perpendicular_vectors_m_value :
  let a : ℝ × ℝ := (-2, 3)
  let b : ℝ × ℝ := (3, m)
  perpendicular a b → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l981_98132


namespace NUMINAMATH_CALUDE_detergent_calculation_l981_98168

/-- The amount of detergent used per pound of clothes -/
def detergent_per_pound : ℝ := 2

/-- The number of pounds of clothes washed -/
def pounds_of_clothes : ℝ := 9

/-- The total amount of detergent used -/
def total_detergent : ℝ := detergent_per_pound * pounds_of_clothes

theorem detergent_calculation : total_detergent = 18 := by
  sorry

end NUMINAMATH_CALUDE_detergent_calculation_l981_98168


namespace NUMINAMATH_CALUDE_sam_read_100_pages_l981_98171

def minimum_assigned : ℕ := 25

def harrison_pages (minimum : ℕ) : ℕ := minimum + 10

def pam_pages (harrison : ℕ) : ℕ := harrison + 15

def sam_pages (pam : ℕ) : ℕ := 2 * pam

theorem sam_read_100_pages :
  sam_pages (pam_pages (harrison_pages minimum_assigned)) = 100 := by
  sorry

end NUMINAMATH_CALUDE_sam_read_100_pages_l981_98171


namespace NUMINAMATH_CALUDE_intersection_property_l981_98137

noncomputable section

-- Define the line l
def line_l (α t : ℝ) : ℝ × ℝ := (2 + t * Real.cos α, 1 + t * Real.sin α)

-- Define the curve C in polar coordinates
def curve_C_polar (θ : ℝ) : ℝ := 4 * Real.cos θ

-- Define the curve C in Cartesian coordinates
def curve_C_cartesian (x y : ℝ) : Prop := x^2 + y^2 = 4*x

-- Define the point P
def point_P : ℝ × ℝ := (2, 1)

-- Theorem statement
theorem intersection_property (α : ℝ) (h_α : 0 ≤ α ∧ α < Real.pi) :
  ∃ t₁ t₂ : ℝ, 
    let A := line_l α t₁
    let B := line_l α t₂
    let P := point_P
    curve_C_cartesian A.1 A.2 ∧ 
    curve_C_cartesian B.1 B.2 ∧
    (A.1 - P.1, A.2 - P.2) = 2 • (P.1 - B.1, P.2 - B.2) →
    Real.tan α = Real.sqrt (3/5) ∨ Real.tan α = -Real.sqrt (3/5) := by
  sorry

end

end NUMINAMATH_CALUDE_intersection_property_l981_98137


namespace NUMINAMATH_CALUDE_min_value_sum_l981_98158

theorem min_value_sum (a b : ℝ) (h : a^2 + 2*b^2 = 6) : 
  ∃ (m : ℝ), m = -3 ∧ ∀ (x y : ℝ), x^2 + 2*y^2 = 6 → m ≤ x + y :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_l981_98158


namespace NUMINAMATH_CALUDE_time_spent_calculation_susan_time_allocation_l981_98175

/-- Given a ratio of activities and time spent on one activity, calculate the time spent on another activity -/
theorem time_spent_calculation (reading_ratio : ℕ) (hangout_ratio : ℕ) (reading_hours : ℕ) 
  (h1 : reading_ratio > 0)
  (h2 : hangout_ratio > 0)
  (h3 : reading_hours > 0) :
  (reading_ratio * (hangout_ratio * reading_hours) / reading_ratio) = hangout_ratio * reading_hours :=
by sorry

/-- Susan's time allocation problem -/
theorem susan_time_allocation :
  let reading_ratio : ℕ := 4
  let hangout_ratio : ℕ := 10
  let reading_hours : ℕ := 8
  (reading_ratio * (hangout_ratio * reading_hours) / reading_ratio) = 20 :=
by sorry

end NUMINAMATH_CALUDE_time_spent_calculation_susan_time_allocation_l981_98175


namespace NUMINAMATH_CALUDE_towel_average_price_l981_98152

/-- Calculates the average price of towels given their quantities and prices -/
theorem towel_average_price 
  (quantity1 quantity2 quantity3 : ℕ)
  (price1 price2 price3 : ℕ)
  (h1 : quantity1 = 3)
  (h2 : quantity2 = 5)
  (h3 : quantity3 = 2)
  (h4 : price1 = 100)
  (h5 : price2 = 150)
  (h6 : price3 = 650) :
  (quantity1 * price1 + quantity2 * price2 + quantity3 * price3) / 
  (quantity1 + quantity2 + quantity3) = 235 := by
  sorry

#check towel_average_price

end NUMINAMATH_CALUDE_towel_average_price_l981_98152


namespace NUMINAMATH_CALUDE_harveys_steaks_l981_98111

theorem harveys_steaks (initial_steaks : ℕ) 
  (h1 : initial_steaks - 17 = 12) 
  (h2 : 17 ≥ 4) : initial_steaks = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_harveys_steaks_l981_98111


namespace NUMINAMATH_CALUDE_intersection_P_Q_l981_98149

def P : Set ℤ := {x | (x - 3) * (x - 6) ≤ 0}
def Q : Set ℤ := {5, 7}

theorem intersection_P_Q : P ∩ Q = {5} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l981_98149


namespace NUMINAMATH_CALUDE_real_root_range_l981_98113

theorem real_root_range (a : ℝ) : 
  (∃ x : ℝ, (2 : ℝ)^(2*x) + (2 : ℝ)^x * a + a + 1 = 0) → 
  a ≤ 2 - 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_real_root_range_l981_98113


namespace NUMINAMATH_CALUDE_zongzi_purchase_l981_98169

/-- Represents the unit price and quantity of zongzi -/
structure Zongzi where
  price : ℝ
  quantity : ℝ

/-- Represents the purchase of zongzi -/
def Purchase (a b : Zongzi) : Prop :=
  a.price * a.quantity = 1500 ∧
  b.price * b.quantity = 1000 ∧
  b.quantity = a.quantity + 50 ∧
  a.price = 2 * b.price

/-- Represents the additional purchase constraint -/
def AdditionalPurchase (a b : Zongzi) (x : ℝ) : Prop :=
  x + (200 - x) = 200 ∧
  a.price * x + b.price * (200 - x) ≤ 1450

/-- Main theorem -/
theorem zongzi_purchase (a b : Zongzi) (x : ℝ) 
  (h1 : Purchase a b) (h2 : AdditionalPurchase a b x) : 
  b.price = 5 ∧ a.price = 10 ∧ x ≤ 90 := by
  sorry

#check zongzi_purchase

end NUMINAMATH_CALUDE_zongzi_purchase_l981_98169


namespace NUMINAMATH_CALUDE_popped_kernel_probability_l981_98162

theorem popped_kernel_probability (white_ratio : ℚ) (yellow_ratio : ℚ) 
  (white_pop_prob : ℚ) (yellow_pop_prob : ℚ) 
  (h1 : white_ratio = 2/3) 
  (h2 : yellow_ratio = 1/3)
  (h3 : white_pop_prob = 1/2)
  (h4 : yellow_pop_prob = 2/3) :
  (white_ratio * white_pop_prob) / (white_ratio * white_pop_prob + yellow_ratio * yellow_pop_prob) = 3/5 := by
  sorry

#check popped_kernel_probability

end NUMINAMATH_CALUDE_popped_kernel_probability_l981_98162


namespace NUMINAMATH_CALUDE_system_solution_l981_98189

theorem system_solution : ∃ (x y : ℚ), 
  (x - 30) / 3 = (2 * y + 7) / 4 ∧ 
  x - y = 10 ∧ 
  x = -81/2 ∧ 
  y = -101/2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l981_98189


namespace NUMINAMATH_CALUDE_parabola_equation_l981_98156

/-- The equation of a parabola with given focus and directrix -/
theorem parabola_equation (x y : ℝ) : 
  let focus : ℝ × ℝ := (4, 4)
  let directrix : ℝ → ℝ → ℝ := λ x y => 4*x + 8*y - 32
  let parabola : ℝ → ℝ → ℝ := λ x y => 64*x^2 - 128*x*y + 64*y^2 - 512*x - 512*y + 1024
  (∀ (p : ℝ × ℝ), p ∈ {p | parabola p.1 p.2 = 0} ↔ 
    (p.1 - focus.1)^2 + (p.2 - focus.2)^2 = (directrix p.1 p.2 / (4 * Real.sqrt 5))^2) :=
by sorry

#check parabola_equation

end NUMINAMATH_CALUDE_parabola_equation_l981_98156


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l981_98187

theorem degree_to_radian_conversion (π : Real) (h : π * 1 = 180) : 
  60 * (π / 180) = π / 3 := by sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l981_98187
