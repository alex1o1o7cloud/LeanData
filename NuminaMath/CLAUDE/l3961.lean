import Mathlib

namespace sqrt_two_is_quadratic_radical_l3961_396159

-- Define what a quadratic radical is
def is_quadratic_radical (x : ℝ) : Prop :=
  ∃ (y : ℝ), x = Real.sqrt y ∧ ¬ (∃ (n : ℤ), x = n)

-- Theorem statement
theorem sqrt_two_is_quadratic_radical : 
  is_quadratic_radical (Real.sqrt 2) :=
sorry

end sqrt_two_is_quadratic_radical_l3961_396159


namespace ninth_minus_eighth_square_tiles_l3961_396192

/-- The number of tiles in a square with side length n -/
def tilesInSquare (n : ℕ) : ℕ := n * n

/-- The difference in tiles between two consecutive squares in the sequence -/
def tileDifference (n : ℕ) : ℕ :=
  tilesInSquare (n + 1) - tilesInSquare n

theorem ninth_minus_eighth_square_tiles : tileDifference 8 = 17 := by
  sorry

end ninth_minus_eighth_square_tiles_l3961_396192


namespace rectangular_prism_diagonal_intersections_l3961_396168

/-- The number of unit cubes a space diagonal passes through in a rectangular prism -/
def spaceDiagonalIntersections (a b c : ℕ) : ℕ :=
  a + b + c - Nat.gcd a b - Nat.gcd b c - Nat.gcd c a + Nat.gcd a (Nat.gcd b c)

/-- Theorem: For a 150 × 324 × 375 rectangular prism, the space diagonal passes through 768 unit cubes -/
theorem rectangular_prism_diagonal_intersections :
  spaceDiagonalIntersections 150 324 375 = 768 := by
  sorry

end rectangular_prism_diagonal_intersections_l3961_396168


namespace grunters_win_probabilities_l3961_396117

/-- The number of games played -/
def num_games : ℕ := 6

/-- The probability of winning a single game -/
def win_prob : ℚ := 7/10

/-- The probability of winning all games -/
def prob_win_all : ℚ := 117649/1000000

/-- The probability of winning exactly 5 out of 6 games -/
def prob_win_five : ℚ := 302526/1000000

/-- Theorem stating the probabilities for winning all games and winning exactly 5 out of 6 games -/
theorem grunters_win_probabilities :
  (win_prob ^ num_games = prob_win_all) ∧
  (Nat.choose num_games 5 * win_prob ^ 5 * (1 - win_prob) ^ 1 = prob_win_five) := by
  sorry

end grunters_win_probabilities_l3961_396117


namespace last_digits_divisible_by_three_l3961_396120

theorem last_digits_divisible_by_three :
  ∃ (S : Finset Nat), (∀ n ∈ S, n < 10) ∧ (Finset.card S = 10) ∧
  (∀ d ∈ S, ∃ (m : Nat), m % 3 = 0 ∧ m % 10 = d) :=
sorry

end last_digits_divisible_by_three_l3961_396120


namespace m_range_for_z_in_third_quadrant_l3961_396103

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := m * (3 + Complex.I) - (2 + Complex.I)

-- Define the condition for a point to be in the third quadrant
def in_third_quadrant (w : ℂ) : Prop := w.re < 0 ∧ w.im < 0

-- State the theorem
theorem m_range_for_z_in_third_quadrant :
  ∀ m : ℝ, in_third_quadrant (z m) ↔ m < 2/3 := by
  sorry

end m_range_for_z_in_third_quadrant_l3961_396103


namespace expression_evaluation_l3961_396179

theorem expression_evaluation :
  3 + 2 * Real.sqrt 3 + (3 + 2 * Real.sqrt 3)⁻¹ + (2 * Real.sqrt 3 - 3)⁻¹ = 3 + (16 * Real.sqrt 3) / 3 := by
  sorry

end expression_evaluation_l3961_396179


namespace cosine_product_equality_l3961_396152

theorem cosine_product_equality : Real.cos (2 * Real.pi / 31) * Real.cos (4 * Real.pi / 31) * Real.cos (8 * Real.pi / 31) * Real.cos (16 * Real.pi / 31) * Real.cos (32 * Real.pi / 31) * 3.418 = 1 / 32 := by
  sorry

end cosine_product_equality_l3961_396152


namespace money_distribution_l3961_396143

theorem money_distribution (A B C : ℕ) 
  (h1 : A + B + C = 500) 
  (h2 : B + C = 330) 
  (h3 : C = 30) : 
  A + C = 200 := by
sorry

end money_distribution_l3961_396143


namespace toms_fruit_bowl_l3961_396170

/-- The number of fruits remaining in Tom's fruit bowl after eating some fruits -/
def remaining_fruits (initial_oranges initial_lemons eaten : ℕ) : ℕ :=
  initial_oranges + initial_lemons - eaten

/-- Theorem: Given Tom's fruit bowl with 3 oranges and 6 lemons, after eating 3 fruits, 6 fruits remain -/
theorem toms_fruit_bowl : remaining_fruits 3 6 3 = 6 := by
  sorry

end toms_fruit_bowl_l3961_396170


namespace integral_abs_x_minus_one_l3961_396115

-- Define the function to be integrated
def f (x : ℝ) : ℝ := |x - 1|

-- State the theorem
theorem integral_abs_x_minus_one : ∫ x in (-2)..2, f x = 5 := by
  sorry

end integral_abs_x_minus_one_l3961_396115


namespace greatest_x_given_lcm_l3961_396106

def is_lcm (a b c m : ℕ) : Prop := 
  (∀ n : ℕ, n % a = 0 ∧ n % b = 0 ∧ n % c = 0 → m ∣ n) ∧
  (m % a = 0 ∧ m % b = 0 ∧ m % c = 0)

theorem greatest_x_given_lcm : 
  ∀ x : ℕ, is_lcm x 15 21 105 → x ≤ 105 :=
by sorry

end greatest_x_given_lcm_l3961_396106


namespace dans_cards_correct_l3961_396114

/-- The number of Pokemon cards Sally initially had -/
def initial_cards : ℕ := 27

/-- The number of Pokemon cards Sally lost -/
def lost_cards : ℕ := 20

/-- The number of Pokemon cards Sally has now -/
def final_cards : ℕ := 48

/-- The number of Pokemon cards Dan gave Sally -/
def dans_cards : ℕ := 41

theorem dans_cards_correct : 
  initial_cards + dans_cards - lost_cards = final_cards :=
by sorry

end dans_cards_correct_l3961_396114


namespace julie_school_year_hours_l3961_396105

/-- Julie's summer work and earnings information -/
structure SummerWork where
  hoursPerWeek : ℕ
  weeks : ℕ
  earnings : ℕ

/-- Julie's school year work information -/
structure SchoolYearWork where
  weeks : ℕ
  targetEarnings : ℕ

/-- Calculate the required hours per week during school year -/
def calculateSchoolYearHours (summer : SummerWork) (schoolYear : SchoolYearWork) : ℕ :=
  let hourlyRate := summer.earnings / (summer.hoursPerWeek * summer.weeks)
  let weeklyEarningsNeeded := schoolYear.targetEarnings / schoolYear.weeks
  weeklyEarningsNeeded / hourlyRate

/-- Theorem stating that Julie needs to work 10 hours per week during the school year -/
theorem julie_school_year_hours 
    (summer : SummerWork) 
    (schoolYear : SchoolYearWork) 
    (h1 : summer.hoursPerWeek = 40)
    (h2 : summer.weeks = 10)
    (h3 : summer.earnings = 4000)
    (h4 : schoolYear.weeks = 40)
    (h5 : schoolYear.targetEarnings = 4000) :
  calculateSchoolYearHours summer schoolYear = 10 := by
  sorry

#eval calculateSchoolYearHours 
  { hoursPerWeek := 40, weeks := 10, earnings := 4000 } 
  { weeks := 40, targetEarnings := 4000 }

end julie_school_year_hours_l3961_396105


namespace a_7_equals_two_l3961_396165

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- Arithmetic sequence property -/
def is_arithmetic (b : Sequence) : Prop :=
  ∀ n m : ℕ, b (n + 1) - b n = b (m + 1) - b m

theorem a_7_equals_two (a b : Sequence) 
  (h1 : ∀ n, a n ≠ 0)
  (h2 : a 4 - 2 * a 7 + a 8 = 0)
  (h3 : is_arithmetic b)
  (h4 : b 7 = a 7)
  (h5 : b 2 < b 8)
  (h6 : b 8 < b 11) :
  a 7 = 2 :=
sorry

end a_7_equals_two_l3961_396165


namespace no_tiling_with_all_tetrominoes_l3961_396173

/-- A tetromino is a shape consisting of 4 squares that can be rotated but not reflected. -/
structure Tetromino :=
  (squares : Fin 4 → (Fin 2 × Fin 2))

/-- There are exactly 7 different tetrominoes. -/
axiom num_tetrominoes : {n : ℕ // n = 7}

/-- A 4 × n rectangle. -/
def Rectangle (n : ℕ) := Fin 4 × Fin n

/-- A tiling of a rectangle with tetrominoes. -/
def Tiling (n : ℕ) := Rectangle n → Tetromino

/-- Theorem: It is impossible to tile a 4 × n rectangle with one copy of each of the 7 different tetrominoes. -/
theorem no_tiling_with_all_tetrominoes (n : ℕ) :
  ¬∃ (t : Tiling n), (∀ tetromino : Tetromino, ∃! (x : Rectangle n), t x = tetromino) :=
sorry

end no_tiling_with_all_tetrominoes_l3961_396173


namespace sufficient_not_necessary_l3961_396123

theorem sufficient_not_necessary (x y : ℝ) :
  (x ≥ 1 ∧ y ≥ 2 → x + y ≥ 3) ∧
  ∃ x y, x + y ≥ 3 ∧ ¬(x ≥ 1 ∧ y ≥ 2) :=
by sorry

end sufficient_not_necessary_l3961_396123


namespace painter_rooms_problem_l3961_396141

theorem painter_rooms_problem (hours_per_room : ℕ) (rooms_painted : ℕ) (remaining_hours : ℕ) :
  hours_per_room = 7 →
  rooms_painted = 5 →
  remaining_hours = 49 →
  rooms_painted + remaining_hours / hours_per_room = 12 :=
by sorry

end painter_rooms_problem_l3961_396141


namespace binomial_coefficient_equality_l3961_396169

theorem binomial_coefficient_equality (m : ℕ) : 
  (Nat.choose 13 (m + 1) = Nat.choose 13 (2 * m - 3)) ↔ (m = 4 ∨ m = 5) := by
  sorry

end binomial_coefficient_equality_l3961_396169


namespace max_sphere_cone_volume_ratio_l3961_396182

/-- The maximum volume ratio of a sphere inscribed in a cone to the cone itself -/
theorem max_sphere_cone_volume_ratio :
  ∃ (r m R : ℝ) (α : ℝ),
    r > 0 ∧ m > 0 ∧ R > 0 ∧ 0 < α ∧ α < π / 2 ∧
    r = m * Real.tan α ∧
    R = (m - R) * Real.sin α ∧
    ∀ (r' m' R' : ℝ) (α' : ℝ),
      r' > 0 → m' > 0 → R' > 0 → 0 < α' → α' < π / 2 →
      r' = m' * Real.tan α' →
      R' = (m' - R') * Real.sin α' →
      (4 / 3 * π * R' ^ 3) / ((1 / 3) * π * r' ^ 2 * m') ≤ 1 / 2 :=
by sorry

end max_sphere_cone_volume_ratio_l3961_396182


namespace student_calculation_l3961_396137

theorem student_calculation (chosen_number : ℕ) (h : chosen_number = 121) : 
  2 * chosen_number - 138 = 104 := by
  sorry

#check student_calculation

end student_calculation_l3961_396137


namespace donkeys_and_boys_l3961_396104

theorem donkeys_and_boys (b d : ℕ) : 
  (d = b - 1) →  -- Condition 1: When each boy sits on a donkey, one boy is left
  (b / 2 = d - 1) →  -- Condition 2: When two boys sit on each donkey, one donkey is left
  (b = 4 ∧ d = 3) :=  -- Conclusion: There are 4 boys and 3 donkeys
by sorry

end donkeys_and_boys_l3961_396104


namespace complement_of_M_in_U_l3961_396193

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}

theorem complement_of_M_in_U :
  (U \ M) = {2, 3, 5} := by sorry

end complement_of_M_in_U_l3961_396193


namespace xyz_sum_l3961_396142

theorem xyz_sum (x y z : ℕ+) 
  (h1 : x * y + z = 53)
  (h2 : y * z + x = 53)
  (h3 : z * x + y = 53) : 
  x + y + z = 54 := by
  sorry

end xyz_sum_l3961_396142


namespace train_crossing_time_l3961_396195

/-- Given a train crossing a platform, calculate the time it takes to cross a signal pole -/
theorem train_crossing_time (train_length platform_length platform_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_length = 333.33)
  (h3 : platform_crossing_time = 38)
  : ∃ (signal_pole_crossing_time : ℝ),
    signal_pole_crossing_time = train_length / ((train_length + platform_length) / platform_crossing_time) ∧
    (signal_pole_crossing_time ≥ 17.9 ∧ signal_pole_crossing_time ≤ 18.1) :=
by sorry

end train_crossing_time_l3961_396195


namespace folded_paper_length_l3961_396190

/-- Given a rectangle with sides of lengths 1 and √2, where one vertex is folded to touch the opposite side, the length d of the folded edge is √2 - 1. -/
theorem folded_paper_length (a b d : ℝ) : 
  a = 1 → b = Real.sqrt 2 → 
  d = Real.sqrt ((b - d)^2 + a^2) → 
  d = Real.sqrt 2 - 1 := by
  sorry

end folded_paper_length_l3961_396190


namespace vertical_translation_by_two_l3961_396189

/-- For any real-valued function f and any real number x,
    f(x) + 2 is equal to a vertical translation of f(x) by 2 units upward -/
theorem vertical_translation_by_two (f : ℝ → ℝ) (x : ℝ) :
  f x + 2 = (fun y ↦ f y + 2) x :=
by sorry

end vertical_translation_by_two_l3961_396189


namespace sixth_term_of_geometric_sequence_l3961_396130

-- Define a geometric sequence
def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

-- Theorem statement
theorem sixth_term_of_geometric_sequence :
  ∀ (r : ℝ),
  (geometric_sequence 16 r 8 = 11664) →
  (geometric_sequence 16 r 6 = 3888) :=
by
  sorry

end sixth_term_of_geometric_sequence_l3961_396130


namespace fraction_equality_l3961_396166

theorem fraction_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x / y + y / x = 4) : 
  x * y / (x^2 - y^2) = Real.sqrt 3 / 3 := by
sorry

end fraction_equality_l3961_396166


namespace stating_not_always_triangle_from_parallelogram_l3961_396145

/-- A stick represents a line segment with a positive length. -/
structure Stick :=
  (length : ℝ)
  (positive : length > 0)

/-- A parallelogram composed of four equal sticks. -/
structure Parallelogram :=
  (stick : Stick)

/-- Represents a potential triangle formed from the parallelogram's sticks. -/
structure PotentialTriangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)

/-- Checks if a triangle can be formed given three side lengths. -/
def isValidTriangle (t : PotentialTriangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side2 + t.side3 > t.side1 ∧
  t.side1 + t.side3 > t.side2

/-- 
Theorem stating that it's not always possible to form a triangle 
from a parallelogram's sticks.
-/
theorem not_always_triangle_from_parallelogram :
  ∃ p : Parallelogram, ¬∃ t : PotentialTriangle, 
    (t.side1 = p.stick.length ∧ t.side2 = p.stick.length ∧ t.side3 = 2 * p.stick.length) ∧
    isValidTriangle t :=
sorry

end stating_not_always_triangle_from_parallelogram_l3961_396145


namespace bryan_total_books_l3961_396111

/-- The number of bookshelves Bryan has -/
def num_bookshelves : ℕ := 15

/-- The number of books in each bookshelf -/
def books_per_shelf : ℕ := 78

/-- The total number of books Bryan has -/
def total_books : ℕ := num_bookshelves * books_per_shelf

/-- Theorem stating that Bryan has 1170 books in total -/
theorem bryan_total_books : total_books = 1170 := by
  sorry

end bryan_total_books_l3961_396111


namespace cross_section_ratio_cube_l3961_396185

theorem cross_section_ratio_cube (a : ℝ) (ha : a > 0) :
  let cube_diagonal := a * Real.sqrt 3
  let min_area := (a / Real.sqrt 2) * cube_diagonal
  let max_area := Real.sqrt 2 * a^2
  max_area / min_area = 2 * Real.sqrt 3 / 3 := by sorry

end cross_section_ratio_cube_l3961_396185


namespace expansion_coefficient_l3961_396183

/-- Represents the coefficient of x^n in the expansion of (x^2 + x + 1)^k -/
def generalized_pascal (k n : ℕ) : ℕ := sorry

/-- The coefficient of x^8 in the expansion of (1+ax)(x^2+x+1)^5 -/
def coeff_x8 (a : ℝ) : ℝ := generalized_pascal 5 2 + a * generalized_pascal 5 1

theorem expansion_coefficient (a : ℝ) : coeff_x8 a = 75 → a = 2 := by sorry

end expansion_coefficient_l3961_396183


namespace line_inclination_l3961_396110

def line_equation (x y : ℝ) : Prop := y = x + 1

def angle_of_inclination (θ : ℝ) : Prop := θ = Real.arctan 1

theorem line_inclination :
  ∀ x y θ : ℝ, line_equation x y → angle_of_inclination θ → θ * (180 / Real.pi) = 45 :=
by sorry

end line_inclination_l3961_396110


namespace upsilon_value_l3961_396198

theorem upsilon_value (Υ : ℤ) : 5 * (-3) = Υ - 3 → Υ = -12 := by
  sorry

end upsilon_value_l3961_396198


namespace unique_exponent_solution_l3961_396176

theorem unique_exponent_solution :
  ∃! w : ℤ, (3 : ℝ) ^ 6 * (3 : ℝ) ^ w = (3 : ℝ) ^ 4 :=
by sorry

end unique_exponent_solution_l3961_396176


namespace divisibility_condition_l3961_396172

theorem divisibility_condition (N : ℤ) : 
  (7 * N + 55) ∣ (N^2 - 71) ↔ N = 57 ∨ N = -8 := by
  sorry

end divisibility_condition_l3961_396172


namespace line_passes_through_point_l3961_396171

theorem line_passes_through_point (m : ℝ) : m * 1 - 1 + 1 - m = 0 := by
  sorry

end line_passes_through_point_l3961_396171


namespace smaller_root_of_equation_l3961_396112

theorem smaller_root_of_equation :
  let f (x : ℚ) := (x - 7/8)^2 + (x - 1/4) * (x - 7/8)
  ∃ (r : ℚ), f r = 0 ∧ r < 9/16 ∧ f (9/16) = 0 :=
by sorry

end smaller_root_of_equation_l3961_396112


namespace lottery_probability_calculation_l3961_396178

def megaBallCount : ℕ := 30
def winnerBallCount : ℕ := 50
def winnerBallsDrawn : ℕ := 5
def specialBallCount : ℕ := 45

def lotteryProbability : ℚ :=
  1 / (megaBallCount * (winnerBallCount.choose winnerBallsDrawn) * specialBallCount)

theorem lottery_probability_calculation :
  lotteryProbability = 1 / 2861184000 := by
  sorry

end lottery_probability_calculation_l3961_396178


namespace march_first_is_thursday_l3961_396197

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in March -/
structure MarchDate where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Given that March 15th is a Thursday, prove that March 1st is also a Thursday -/
theorem march_first_is_thursday (march15 : MarchDate) 
    (h : march15.day = 15 ∧ march15.dayOfWeek = DayOfWeek.Thursday) :
    ∃ (march1 : MarchDate), march1.day = 1 ∧ march1.dayOfWeek = DayOfWeek.Thursday :=
  sorry

end march_first_is_thursday_l3961_396197


namespace first_car_years_earlier_l3961_396102

-- Define the manufacture years of the cars
def first_car_year : ℕ := 1970
def third_car_year : ℕ := 2000

-- Define the time difference between the second and third cars
def years_between_second_and_third : ℕ := 20

-- Define the manufacture year of the second car
def second_car_year : ℕ := third_car_year - years_between_second_and_third

-- Theorem to prove
theorem first_car_years_earlier (h : second_car_year = third_car_year - years_between_second_and_third) :
  second_car_year - first_car_year = 10 := by
  sorry

end first_car_years_earlier_l3961_396102


namespace wrong_to_right_ratio_l3961_396177

theorem wrong_to_right_ratio (total : ℕ) (correct : ℕ) (h1 : total = 48) (h2 : correct = 16) :
  (total - correct) / correct = 2 := by
  sorry

end wrong_to_right_ratio_l3961_396177


namespace remainder_of_n_mod_500_l3961_396199

/-- The set S containing elements from 1 to 12 -/
def S : Finset ℕ := Finset.range 12

/-- The number of sets of two non-empty disjoint subsets of S -/
def n : ℕ := ((3^12 - 2 * 2^12 + 1) / 2 : ℕ)

/-- Theorem stating that the remainder of n divided by 500 is 125 -/
theorem remainder_of_n_mod_500 : n % 500 = 125 := by
  sorry

end remainder_of_n_mod_500_l3961_396199


namespace find_y_l3961_396151

def v (y : ℝ) : Fin 2 → ℝ := ![1, y]
def w : Fin 2 → ℝ := ![9, 3]
def proj_w_v : Fin 2 → ℝ := ![-6, -2]

theorem find_y : ∃ y : ℝ, v y = v y ∧ w = w ∧ proj_w_v = proj_w_v → y = -23 := by
  sorry

end find_y_l3961_396151


namespace water_duration_village_water_duration_l3961_396161

/-- Calculates how long water will last in a village given specific conditions. -/
theorem water_duration (water_per_person : ℝ) (small_households : ℕ) (large_households : ℕ) 
  (small_household_size : ℕ) (large_household_size : ℕ) (total_water : ℝ) : ℝ :=
  let water_usage_per_month := 
    (small_households * small_household_size * water_per_person) + 
    (large_households * large_household_size * water_per_person)
  total_water / water_usage_per_month

/-- Proves that the water lasts approximately 4.31 months under given conditions. -/
theorem village_water_duration : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |water_duration 20 7 3 2 5 2500 - 4.31| < ε :=
sorry

end water_duration_village_water_duration_l3961_396161


namespace no_integer_solution_for_dog_nails_l3961_396109

theorem no_integer_solution_for_dog_nails :
  ¬ ∃ (x : ℕ), 16 * x + 64 = 113 := by
  sorry

end no_integer_solution_for_dog_nails_l3961_396109


namespace cyclic_sum_inequality_l3961_396158

theorem cyclic_sum_inequality (a b c : ℝ) : 
  1 < (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ∧ 
  (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ≤ (3 * Real.sqrt 2) / 2 := by
  sorry

end cyclic_sum_inequality_l3961_396158


namespace all_normal_all_false_l3961_396121

-- Define the possible types of people
inductive PersonType
  | Knight
  | Liar
  | Normal

-- Define the four people
structure Person :=
  (name : String)
  (type : PersonType)

-- Define the statements made
def statement1 (mr_b : Person) : Prop := mr_b.type = PersonType.Knight
def statement2 (mr_a : Person) (mr_b : Person) : Prop := 
  mr_a.type = PersonType.Knight ∧ mr_b.type = PersonType.Knight
def statement3 (mr_b : Person) : Prop := mr_b.type = PersonType.Knight

-- Define the problem setup
def problem_setup (mr_a mrs_a mr_b mrs_b : Person) : Prop :=
  mr_a.name = "Mr. A" ∧
  mrs_a.name = "Mrs. A" ∧
  mr_b.name = "Mr. B" ∧
  mrs_b.name = "Mrs. B"

-- Theorem statement
theorem all_normal_all_false 
  (mr_a mrs_a mr_b mrs_b : Person) 
  (h_setup : problem_setup mr_a mrs_a mr_b mrs_b) :
  (mr_a.type = PersonType.Normal ∧
   mrs_a.type = PersonType.Normal ∧
   mr_b.type = PersonType.Normal ∧
   mrs_b.type = PersonType.Normal) ∧
  (¬statement1 mr_b ∧
   ¬statement2 mr_a mr_b ∧
   ¬statement3 mr_b) :=
by sorry


end all_normal_all_false_l3961_396121


namespace correct_calculation_l3961_396144

theorem correct_calculation (y : ℝ) : 3 * y^2 - 2 * y^2 = y^2 := by
  sorry

end correct_calculation_l3961_396144


namespace weight_of_barium_fluoride_l3961_396132

/-- The atomic weight of barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of fluorine in g/mol -/
def atomic_weight_F : ℝ := 19.00

/-- The number of moles of Barium fluoride -/
def moles_BaF2 : ℝ := 3

/-- The molecular weight of Barium fluoride (BaF2) in g/mol -/
def molecular_weight_BaF2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_F

/-- The weight of Barium fluoride in grams -/
def weight_BaF2 : ℝ := moles_BaF2 * molecular_weight_BaF2

theorem weight_of_barium_fluoride : weight_BaF2 = 525.99 := by
  sorry

end weight_of_barium_fluoride_l3961_396132


namespace escalator_problem_l3961_396107

/-- Represents the escalator system in the shopping mall -/
structure EscalatorSystem where
  boyStepRate : ℕ
  girlStepRate : ℕ
  boyStepsToTop : ℕ
  girlStepsToTop : ℕ
  escalatorSpeed : ℝ
  exposedSteps : ℕ

/-- The conditions of the problem -/
def problemConditions (sys : EscalatorSystem) : Prop :=
  sys.boyStepRate = 2 * sys.girlStepRate ∧
  sys.boyStepsToTop = 27 ∧
  sys.girlStepsToTop = 18 ∧
  sys.escalatorSpeed > 0

/-- The theorem to prove -/
theorem escalator_problem (sys : EscalatorSystem) 
  (h : problemConditions sys) : 
  sys.exposedSteps = 54 ∧ 
  ∃ (boySteps : ℕ), boySteps = 198 ∧ 
    (boySteps = 3 * sys.boyStepsToTop + 2 * sys.exposedSteps) :=
sorry

end escalator_problem_l3961_396107


namespace watch_sale_gain_percentage_l3961_396154

/-- Proves that for a watch with a given cost price, sold at a loss, 
    if the selling price is increased by a certain amount, 
    the resulting gain percentage is as expected. -/
theorem watch_sale_gain_percentage 
  (cost_price : ℝ) 
  (loss_percentage : ℝ) 
  (price_increase : ℝ) : 
  cost_price = 1200 →
  loss_percentage = 10 →
  price_increase = 168 →
  let loss_amount := (loss_percentage / 100) * cost_price
  let initial_selling_price := cost_price - loss_amount
  let new_selling_price := initial_selling_price + price_increase
  let gain_amount := new_selling_price - cost_price
  let gain_percentage := (gain_amount / cost_price) * 100
  gain_percentage = 4 := by
sorry


end watch_sale_gain_percentage_l3961_396154


namespace expected_sufferers_l3961_396119

theorem expected_sufferers (sample_size : ℕ) (probability : ℚ) (h1 : sample_size = 400) (h2 : probability = 1/4) :
  ↑sample_size * probability = 100 := by
  sorry

end expected_sufferers_l3961_396119


namespace ducks_joining_l3961_396148

theorem ducks_joining (original : ℕ) (total : ℕ) (joined : ℕ) : 
  original = 13 → total = 33 → joined = total - original → joined = 20 := by
sorry

end ducks_joining_l3961_396148


namespace weight_estimate_error_l3961_396126

/-- The weight of a disk with precise 1-meter diameter in kg -/
def precise_disk_weight : ℝ := 100

/-- The radius of a disk in meters -/
def disk_radius : ℝ := 0.5

/-- The standard deviation of the disk radius in meters -/
def radius_std_dev : ℝ := 0.01

/-- The number of disks in the stack -/
def num_disks : ℕ := 100

/-- The expected weight of a single disk with variable radius -/
noncomputable def expected_disk_weight : ℝ := sorry

/-- The error in the weight estimate -/
theorem weight_estimate_error :
  num_disks * expected_disk_weight - (num_disks : ℝ) * precise_disk_weight = 4 := by sorry

end weight_estimate_error_l3961_396126


namespace ellipse_foci_distance_l3961_396167

/-- The distance between the foci of the ellipse 9x^2 + y^2 = 900 is 40√2 -/
theorem ellipse_foci_distance :
  let a : ℝ := Real.sqrt 100
  let b : ℝ := Real.sqrt 900
  let c : ℝ := Real.sqrt (b^2 - a^2)
  2 * c = 40 * Real.sqrt 2 :=
by sorry

end ellipse_foci_distance_l3961_396167


namespace sector_angle_in_unit_circle_l3961_396156

theorem sector_angle_in_unit_circle (sector_area : ℝ) (central_angle : ℝ) : 
  sector_area = 1 → central_angle = 2 := by
  sorry

end sector_angle_in_unit_circle_l3961_396156


namespace trigonometric_equation_solution_l3961_396133

theorem trigonometric_equation_solution (α : Real) 
  (h : 3 * Real.sin α + Real.cos α = 0) : 
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10 / 3 := by
  sorry

end trigonometric_equation_solution_l3961_396133


namespace repeating_decimal_fraction_sum_l3961_396116

theorem repeating_decimal_fraction_sum (a b : ℕ+) : 
  (a.val : ℚ) / b.val = 35 / 99 → 
  Nat.gcd a.val b.val = 1 → 
  a.val + b.val = 134 := by
sorry

end repeating_decimal_fraction_sum_l3961_396116


namespace quadratic_polynomials_intersection_l3961_396162

-- Define the type for quadratic polynomials
def QuadraticPolynomial := ℝ → ℝ

-- Define a function to check if three polynomials have pairwise distinct leading coefficients
def pairwiseDistinctLeadingCoeff (f g h : QuadraticPolynomial) : Prop :=
  ∃ (a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ : ℝ),
    (∀ x, f x = a₁ * x^2 + b₁ * x + c₁) ∧
    (∀ x, g x = a₂ * x^2 + b₂ * x + c₂) ∧
    (∀ x, h x = a₃ * x^2 + b₃ * x + c₃) ∧
    a₁ ≠ a₂ ∧ a₂ ≠ a₃ ∧ a₁ ≠ a₃

-- Define a function to check if two polynomials intersect at exactly one point
def intersectAtOnePoint (f g : QuadraticPolynomial) : Prop :=
  ∃! x, f x = g x

-- Main theorem
theorem quadratic_polynomials_intersection
  (f g h : QuadraticPolynomial)
  (h₁ : pairwiseDistinctLeadingCoeff f g h)
  (h₂ : intersectAtOnePoint f g)
  (h₃ : intersectAtOnePoint g h)
  (h₄ : intersectAtOnePoint f h) :
  ∃! x, f x = g x ∧ g x = h x :=
sorry

end quadratic_polynomials_intersection_l3961_396162


namespace robert_reading_capacity_l3961_396139

/-- Calculates the maximum number of complete books that can be read given the reading speed, book length, and available time. -/
def max_complete_books_read (reading_speed : ℕ) (book_length : ℕ) (available_time : ℕ) : ℕ :=
  (available_time * reading_speed) / book_length

/-- Proves that Robert can read at most 2 complete 360-page books in 8 hours at a speed of 120 pages per hour. -/
theorem robert_reading_capacity :
  max_complete_books_read 120 360 8 = 2 := by
  sorry

end robert_reading_capacity_l3961_396139


namespace kit_prices_correct_l3961_396129

/-- The price of kit B in yuan -/
def price_B : ℝ := 150

/-- The price of kit A in yuan -/
def price_A : ℝ := 180

/-- The relationship between the prices of kit A and kit B -/
def price_relation : Prop := price_A = 1.2 * price_B

/-- The equation representing the difference in quantities purchased -/
def quantity_difference : Prop :=
  (9900 / price_A) - (7500 / price_B) = 5

theorem kit_prices_correct :
  price_relation ∧ quantity_difference → price_A = 180 ∧ price_B = 150 := by
  sorry

end kit_prices_correct_l3961_396129


namespace arithmetic_sequence_problem_l3961_396122

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (n : ℕ) :
  a 1 = 1 →
  a 2 = 4 →
  (∀ k ≥ 2, 2 * a k = a (k - 1) + a (k + 1)) →
  a n = 301 →
  n = 101 := by
sorry

end arithmetic_sequence_problem_l3961_396122


namespace high_school_relationships_l3961_396164

/-- The number of people in the group -/
def n : ℕ := 12

/-- The number of categories for each pair -/
def categories : ℕ := 3

/-- The number of pairs in a group of n people -/
def pairs (n : ℕ) : ℕ := n.choose 2

/-- The total number of pair categorizations -/
def totalCategorizations (n : ℕ) (categories : ℕ) : ℕ :=
  pairs n * categories

theorem high_school_relationships :
  totalCategorizations n categories = 198 := by sorry

end high_school_relationships_l3961_396164


namespace repeating_decimal_sum_l3961_396101

/-- Expresses the sum of repeating decimals 0.3̅, 0.07̅, and 0.008̅ as a common fraction -/
theorem repeating_decimal_sum : 
  (1 : ℚ) / 3 + 7 / 99 + 8 / 999 = 418 / 999 := by sorry

end repeating_decimal_sum_l3961_396101


namespace individual_egg_price_is_50_l3961_396149

/-- The price per individual egg in cents -/
def individual_egg_price : ℕ := sorry

/-- The number of eggs in a tray -/
def eggs_per_tray : ℕ := 30

/-- The price of a tray of eggs in cents -/
def tray_price : ℕ := 1200

/-- The savings per egg when buying a tray, in cents -/
def savings_per_egg : ℕ := 10

theorem individual_egg_price_is_50 : 
  individual_egg_price = 50 :=
by
  sorry

end individual_egg_price_is_50_l3961_396149


namespace temple_storage_cost_l3961_396175

/-- Calculates the total cost for storing items for a group of people -/
def totalCost (numPeople : ℕ) (numPeopleWithGloves : ℕ) (costPerObject : ℕ) : ℕ :=
  let numObjectsPerPerson := 2 + 2 + 1 + 1  -- 2 shoes, 2 socks, 1 mobile, 1 umbrella
  let totalObjects := numPeople * numObjectsPerPerson + numPeopleWithGloves * 2
  totalObjects * costPerObject

/-- Proves that the total cost for the given scenario is 374 dollars -/
theorem temple_storage_cost : totalCost 5 2 11 = 374 := by
  sorry

end temple_storage_cost_l3961_396175


namespace ring_arrangement_count_l3961_396100

def ring_arrangements (total_rings : ℕ) (rings_to_use : ℕ) (fingers : ℕ) : ℕ :=
  Nat.choose total_rings rings_to_use * 
  Nat.factorial rings_to_use * 
  Nat.choose (rings_to_use + fingers - 1) (fingers - 1)

theorem ring_arrangement_count :
  ring_arrangements 8 5 4 = 376320 :=
by sorry

end ring_arrangement_count_l3961_396100


namespace parallel_vectors_x_value_l3961_396138

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ × ℝ := (x, 4)
  are_parallel a b → x = 6 := by
  sorry

end parallel_vectors_x_value_l3961_396138


namespace science_club_election_l3961_396155

theorem science_club_election (total_candidates : Nat) (past_officers : Nat) (positions : Nat) :
  total_candidates = 20 →
  past_officers = 10 →
  positions = 4 →
  (Nat.choose total_candidates positions -
   (Nat.choose (total_candidates - past_officers) positions +
    Nat.choose past_officers 1 * Nat.choose (total_candidates - past_officers) (positions - 1))) = 3435 :=
by sorry

end science_club_election_l3961_396155


namespace selina_leaves_with_30_l3961_396160

/-- The amount of money Selina leaves the store with after selling and buying clothes -/
def selina_final_money (pants_price shorts_price shirts_price : ℕ) 
  (pants_sold shorts_sold shirts_sold : ℕ) 
  (shirts_bought new_shirt_price : ℕ) : ℕ :=
  pants_price * pants_sold + shorts_price * shorts_sold + shirts_price * shirts_sold - 
  shirts_bought * new_shirt_price

/-- Theorem stating that Selina leaves the store with $30 -/
theorem selina_leaves_with_30 : 
  selina_final_money 5 3 4 3 5 5 2 10 = 30 := by
  sorry

end selina_leaves_with_30_l3961_396160


namespace tshirt_jersey_cost_difference_l3961_396124

/-- The amount the Razorback shop makes off each t-shirt -/
def tshirt_profit : ℕ := 192

/-- The amount the Razorback shop makes off each jersey -/
def jersey_profit : ℕ := 34

/-- The difference in cost between a t-shirt and a jersey -/
def cost_difference : ℕ := tshirt_profit - jersey_profit

theorem tshirt_jersey_cost_difference :
  cost_difference = 158 :=
sorry

end tshirt_jersey_cost_difference_l3961_396124


namespace jan_roses_cost_l3961_396140

theorem jan_roses_cost : 
  let dozen : ℕ := 12
  let roses_bought : ℕ := 5 * dozen
  let cost_per_rose : ℕ := 6
  let discount_rate : ℚ := 4/5
  (roses_bought * cost_per_rose : ℚ) * discount_rate = 288 := by
  sorry

end jan_roses_cost_l3961_396140


namespace sequence_sum_l3961_396125

theorem sequence_sum (a b c d : ℕ) 
  (h1 : b - a = d - c) 
  (h2 : d - a = 24) 
  (h3 : b - a = (d - c) + 2) 
  (h4 : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) : 
  a + b + c + d = 54 := by
  sorry

end sequence_sum_l3961_396125


namespace equation_solution_l3961_396113

theorem equation_solution (x : ℚ) : 9 / (5 + 3 / x) = 1 → x = 3/4 := by
  sorry

end equation_solution_l3961_396113


namespace accurate_estimation_l3961_396186

/-- Represents a scale with a lower and upper bound -/
structure Scale where
  lower : ℝ
  upper : ℝ
  h : lower < upper

/-- Represents the position of an arrow on the scale -/
def ArrowPosition (s : Scale) := {x : ℝ // s.lower ≤ x ∧ x ≤ s.upper}

/-- The set of possible readings -/
def PossibleReadings : Set ℝ := {10.1, 10.2, 10.3, 10.4, 10.5}

/-- Function to determine the most accurate estimation -/
noncomputable def mostAccurateEstimation (s : Scale) (arrow : ArrowPosition s) : ℝ :=
  sorry

/-- Theorem stating that 10.3 is the most accurate estimation -/
theorem accurate_estimation (s : Scale) (arrow : ArrowPosition s) 
    (h1 : s.lower = 10.15) (h2 : s.upper = 10.4) : 
    mostAccurateEstimation s arrow = 10.3 := by
  sorry

end accurate_estimation_l3961_396186


namespace point_on_terminal_side_l3961_396150

/-- Proves that for a point P(-√3, y) on the terminal side of angle β, 
    where sin β = √13/13, the value of y is 1/2. -/
theorem point_on_terminal_side (β : ℝ) (y : ℝ) : 
  (∃ P : ℝ × ℝ, P.1 = -Real.sqrt 3 ∧ P.2 = y ∧ 
    Real.sin β = Real.sqrt 13 / 13 ∧ 
    (P.1 ≥ 0 ∨ (P.1 < 0 ∧ P.2 > 0))) → 
  y = 1/2 := by
  sorry

end point_on_terminal_side_l3961_396150


namespace garden_length_l3961_396157

/-- A rectangular garden with length twice the width and 180 yards of fencing. -/
structure Garden where
  width : ℝ
  length : ℝ
  fencing : ℝ
  twice_width : length = 2 * width
  total_fencing : 2 * length + 2 * width = fencing

/-- The length of a garden with 180 yards of fencing is 60 yards. -/
theorem garden_length (g : Garden) (h : g.fencing = 180) : g.length = 60 := by
  sorry

end garden_length_l3961_396157


namespace rosy_fish_count_l3961_396174

/-- The number of fish Lilly has -/
def lilly_fish : ℕ := 10

/-- The total number of fish Lilly and Rosy have together -/
def total_fish : ℕ := 21

/-- The number of fish Rosy has -/
def rosy_fish : ℕ := total_fish - lilly_fish

theorem rosy_fish_count : rosy_fish = 11 := by
  sorry

end rosy_fish_count_l3961_396174


namespace equation_solution_l3961_396134

theorem equation_solution : 
  ∃! x : ℝ, 4 * (4 ^ x) + Real.sqrt (16 * (16 ^ x)) = 64 ∧ x = (3 : ℝ) / 2 := by
  sorry

end equation_solution_l3961_396134


namespace cards_distribution_l3961_396127

theorem cards_distribution (total_cards : Nat) (num_people : Nat) 
  (h1 : total_cards = 60) 
  (h2 : num_people = 8) : 
  (num_people - (total_cards % num_people)) = 4 := by
  sorry

end cards_distribution_l3961_396127


namespace book_length_l3961_396188

theorem book_length (P : ℕ) 
  (h1 : 2 * P = 3 * ((2 * P) / 3 - P / 3 + 100)) : P = 300 := by
  sorry

end book_length_l3961_396188


namespace fruit_display_ratio_l3961_396187

theorem fruit_display_ratio (apples oranges bananas : ℕ) : 
  apples = 2 * oranges →
  apples + oranges + bananas = 35 →
  bananas = 5 →
  oranges = 2 * bananas :=
by
  sorry

end fruit_display_ratio_l3961_396187


namespace slope_angle_of_line_PQ_l3961_396128

theorem slope_angle_of_line_PQ 
  (a b c : ℝ) 
  (hab : a ≠ b) 
  (hbc : b ≠ c) 
  (hac : a ≠ c) 
  (P : ℝ × ℝ) 
  (hP : P = (b, b + c)) 
  (Q : ℝ × ℝ) 
  (hQ : Q = (a, c + a)) : 
  Real.arctan ((Q.2 - P.2) / (Q.1 - P.1)) = π / 4 := by
  sorry

#check slope_angle_of_line_PQ

end slope_angle_of_line_PQ_l3961_396128


namespace f_increasing_f_sum_zero_l3961_396181

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem f_increasing (a : ℝ) : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂ := by sorry

theorem f_sum_zero : 
  f 1 (-5) + f 1 (-3) + f 1 (-1) + f 1 1 + f 1 3 + f 1 5 = 0 := by sorry

end f_increasing_f_sum_zero_l3961_396181


namespace solution_set_theorem_min_value_g_min_value_fraction_l3961_396180

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1|

-- Define the function g
def g (x : ℝ) : ℝ := f x + f (x - 1)

-- Theorem 1: Solution set of f(x) + |x+1| < 2
theorem solution_set_theorem :
  {x : ℝ | f x + |x + 1| < 2} = {x : ℝ | 0 < x ∧ x < 2/3} :=
sorry

-- Theorem 2: Minimum value of g(x)
theorem min_value_g :
  ∀ x : ℝ, g x ≥ 2 :=
sorry

-- Theorem 3: Minimum value of 4/m + 1/n
theorem min_value_fraction (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 2) :
  4/m + 1/n ≥ 9/2 :=
sorry

end solution_set_theorem_min_value_g_min_value_fraction_l3961_396180


namespace expression_simplification_l3961_396184

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 + 3) :
  (x^2 - 1) / (x^2 - 6*x + 9) * (1 - x / (x - 1)) / ((x + 1) / (x - 3)) = -Real.sqrt 2 / 2 := by
  sorry

end expression_simplification_l3961_396184


namespace line_translation_down_5_l3961_396147

/-- Represents a line in 2D space -/
structure Line where
  slope : ℚ
  intercept : ℚ

/-- Translates a line vertically by a given amount -/
def translateLine (l : Line) (dy : ℚ) : Line :=
  { slope := l.slope, intercept := l.intercept + dy }

theorem line_translation_down_5 :
  let original_line := { slope := -1/2, intercept := 2 : Line }
  let translated_line := translateLine original_line (-5)
  translated_line = { slope := -1/2, intercept := -3 : Line } := by
  sorry

end line_translation_down_5_l3961_396147


namespace complex_number_quadrant_l3961_396191

theorem complex_number_quadrant : ∃ (z : ℂ), z = Complex.I * (1 - Complex.I) ∧ z.re > 0 ∧ z.im > 0 := by
  sorry

end complex_number_quadrant_l3961_396191


namespace vector_addition_l3961_396135

theorem vector_addition (a b : ℝ × ℝ) : 
  a = (-1, 6) → b = (3, -2) → a + b = (2, 4) := by
  sorry

end vector_addition_l3961_396135


namespace kangaroo_equality_days_l3961_396163

/-- The number of days required for Bert to have the same number of kangaroos as Kameron -/
def days_to_equal_kangaroos (kameron_kangaroos bert_kangaroos bert_daily_purchase : ℕ) : ℕ :=
  (kameron_kangaroos - bert_kangaroos) / bert_daily_purchase

/-- Theorem stating that it takes 40 days for Bert to have the same number of kangaroos as Kameron -/
theorem kangaroo_equality_days :
  days_to_equal_kangaroos 100 20 2 = 40 := by
  sorry

#eval days_to_equal_kangaroos 100 20 2

end kangaroo_equality_days_l3961_396163


namespace simplify_complex_expression_l3961_396194

-- Define the complex number i
def i : ℂ := Complex.I

-- Theorem statement
theorem simplify_complex_expression : i * (1 - i)^2 = 2 := by
  sorry

end simplify_complex_expression_l3961_396194


namespace quadratic_min_iff_m_gt_neg_one_l3961_396153

/-- A quadratic function with coefficient (m + 1) has a minimum value if and only if m > -1 -/
theorem quadratic_min_iff_m_gt_neg_one (m : ℝ) :
  (∃ (min : ℝ), ∀ (x : ℝ), (m + 1) * x^2 ≥ min) ↔ m > -1 :=
sorry

end quadratic_min_iff_m_gt_neg_one_l3961_396153


namespace hamburgers_left_over_l3961_396108

theorem hamburgers_left_over (made served : ℕ) (h1 : made = 9) (h2 : served = 3) :
  made - served = 6 := by
  sorry

end hamburgers_left_over_l3961_396108


namespace curve_inequality_l3961_396146

/-- Given real numbers a, b, c satisfying certain conditions, 
    prove an inequality for points on a specific curve. -/
theorem curve_inequality (a b c : ℝ) 
  (h1 : b^2 - a*c < 0) 
  (h2 : ∀ x y : ℝ, x > 0 → y > 0 → 
    a * (Real.log x)^2 + 2*b*(Real.log x * Real.log y) + c * (Real.log y)^2 = 1 → 
    (x = 10 ∧ y = 1/10) ∨ 
    (-1 / Real.sqrt (a*c - b^2) ≤ Real.log (x*y) ∧ 
     Real.log (x*y) ≤ 1 / Real.sqrt (a*c - b^2))) : 
  ∀ x y : ℝ, x > 0 → y > 0 → 
    a * (Real.log x)^2 + 2*b*(Real.log x * Real.log y) + c * (Real.log y)^2 = 1 → 
    -1 / Real.sqrt (a*c - b^2) ≤ Real.log (x*y) ∧ 
    Real.log (x*y) ≤ 1 / Real.sqrt (a*c - b^2) := by
  sorry

end curve_inequality_l3961_396146


namespace descent_route_length_l3961_396131

/- Define the hiking trip parameters -/
def forest_speed : ℝ := 8
def rocky_speed : ℝ := 5
def snowy_speed : ℝ := 3
def forest_time : ℝ := 1
def rocky_time : ℝ := 1
def snowy_time : ℝ := 0.5
def speed_multiplier : ℝ := 1.5
def total_days : ℝ := 2

/- Define the theorem -/
theorem descent_route_length :
  let grassland_speed := forest_speed * speed_multiplier
  let sandy_speed := rocky_speed * speed_multiplier
  let descent_distance := grassland_speed * forest_time + sandy_speed * rocky_time
  descent_distance = 19.5 := by sorry

end descent_route_length_l3961_396131


namespace max_distance_difference_l3961_396118

/-- The hyperbola E with equation x²/m - y²/3 = 1 where m > 0 -/
structure Hyperbola where
  m : ℝ
  h_m_pos : m > 0

/-- The eccentricity of the hyperbola -/
def eccentricity (E : Hyperbola) : ℝ := 2

/-- The right focus F of the hyperbola -/
def right_focus (E : Hyperbola) : ℝ × ℝ := sorry

/-- Point A -/
def point_A : ℝ × ℝ := (0, 1)

/-- A point P on the right branch of the hyperbola -/
def point_P (E : Hyperbola) : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The theorem stating the maximum value of |PF| - |PA| -/
theorem max_distance_difference (E : Hyperbola) :
  ∃ (max : ℝ), ∀ (P : ℝ × ℝ), P = point_P E →
    distance P (right_focus E) - distance P point_A ≤ max ∧
    max = Real.sqrt 5 - 2 :=
sorry

end max_distance_difference_l3961_396118


namespace distance_from_p_to_ad_l3961_396136

/-- Square with side length 6 -/
structure Square :=
  (side : ℝ)
  (is_six : side = 6)

/-- Point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Circle in 2D space -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Given a square ABCD, find the distance from point P to side AD, where P is an intersection
    point of two circles: one centered at M (midpoint of CD) with radius 3, and another centered
    at A with radius 5. -/
def distance_to_side (s : Square) : ℝ :=
  let a := Point.mk 0 s.side
  let d := Point.mk 0 0
  let m := Point.mk (s.side / 2) 0
  let circle_m := Circle.mk m 3
  let circle_a := Circle.mk a 5
  -- The actual calculation of the distance would go here
  sorry

/-- The theorem stating that the distance from P to AD is equal to some specific value -/
theorem distance_from_p_to_ad (s : Square) : ∃ x : ℝ, distance_to_side s = x :=
  sorry

end distance_from_p_to_ad_l3961_396136


namespace partial_fraction_decomposition_product_l3961_396196

theorem partial_fraction_decomposition_product : 
  ∀ (A B C : ℚ),
  (∀ x : ℚ, x ≠ 1 ∧ x ≠ -3 ∧ x ≠ 4 →
    (x^2 - 23) / (x^3 - 3*x^2 - 4*x + 12) = 
    A / (x - 1) + B / (x + 3) + C / (x - 4)) →
  A * B * C = 11/36 := by
sorry

end partial_fraction_decomposition_product_l3961_396196
