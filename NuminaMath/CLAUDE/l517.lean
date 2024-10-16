import Mathlib

namespace NUMINAMATH_CALUDE_probability_to_reach_target_l517_51797

-- Define the robot's position as a pair of integers
def Position := ℤ × ℤ

-- Define the possible directions
inductive Direction
| Left
| Right
| Up
| Down

-- Define a step as a movement in a direction
def step (pos : Position) (dir : Direction) : Position :=
  match dir with
  | Direction.Left  => (pos.1 - 1, pos.2)
  | Direction.Right => (pos.1 + 1, pos.2)
  | Direction.Up    => (pos.1, pos.2 + 1)
  | Direction.Down  => (pos.1, pos.2 - 1)

-- Define the probability of each direction
def directionProbability : ℚ := 1 / 4

-- Define the maximum number of steps
def maxSteps : ℕ := 6

-- Define the target position
def target : Position := (3, 1)

-- Define the function to calculate the probability of reaching the target
noncomputable def probabilityToReachTarget : ℚ := sorry

-- State the theorem
theorem probability_to_reach_target :
  probabilityToReachTarget = 37 / 512 := by sorry

end NUMINAMATH_CALUDE_probability_to_reach_target_l517_51797


namespace NUMINAMATH_CALUDE_infinite_sum_equality_l517_51782

/-- Given positive real numbers a and b with a > b, the infinite sum
    1/(b*a^2) + 1/(a^2*(2a^2 - b^2)) + 1/((2a^2 - b^2)*(3a^2 - 2b^2)) + 1/((3a^2 - 2b^2)*(4a^2 - 3b^2)) + ...
    is equal to 1 / ((a^2 - b^2) * b^2) -/
theorem infinite_sum_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let series := fun n : ℕ => 1 / ((n * a^2 - (n-1) * b^2) * ((n+1) * a^2 - n * b^2))
  ∑' n, series n = 1 / ((a^2 - b^2) * b^2) := by
  sorry

end NUMINAMATH_CALUDE_infinite_sum_equality_l517_51782


namespace NUMINAMATH_CALUDE_exponent_multiplication_l517_51763

theorem exponent_multiplication (a : ℝ) : a^4 * a^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l517_51763


namespace NUMINAMATH_CALUDE_feb_first_is_monday_l517_51716

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
structure FebruaryDate :=
  (day : Nat)
  (dayOfWeek : DayOfWeek)

/-- Given that February 29th is a Monday in a leap year, 
    prove that February 1st is also a Monday -/
theorem feb_first_is_monday 
  (feb29 : FebruaryDate)
  (h1 : feb29.day = 29)
  (h2 : feb29.dayOfWeek = DayOfWeek.Monday) :
  ∃ (feb1 : FebruaryDate), 
    feb1.day = 1 ∧ feb1.dayOfWeek = DayOfWeek.Monday :=
by sorry

end NUMINAMATH_CALUDE_feb_first_is_monday_l517_51716


namespace NUMINAMATH_CALUDE_greatest_integer_x_squared_less_than_25_l517_51718

theorem greatest_integer_x_squared_less_than_25 :
  ∀ x : ℕ+, (x : ℝ) ^ 2 < 25 → x ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_x_squared_less_than_25_l517_51718


namespace NUMINAMATH_CALUDE_g_zero_eq_one_l517_51794

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x + y) = g x + g y - 1

/-- Theorem stating that g(0) = 1 for any function satisfying the functional equation -/
theorem g_zero_eq_one (g : ℝ → ℝ) (h : FunctionalEquation g) : g 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_g_zero_eq_one_l517_51794


namespace NUMINAMATH_CALUDE_f_min_value_l517_51748

-- Define the function
def f (x : ℝ) : ℝ := |x - 2| + |3 - x|

-- State the theorem
theorem f_min_value :
  (∀ x : ℝ, f x ≥ 1) ∧ (∃ x : ℝ, f x = 1) :=
sorry

end NUMINAMATH_CALUDE_f_min_value_l517_51748


namespace NUMINAMATH_CALUDE_multiply_monomials_l517_51703

theorem multiply_monomials (x : ℝ) : 2*x * 5*x^2 = 10*x^3 := by
  sorry

end NUMINAMATH_CALUDE_multiply_monomials_l517_51703


namespace NUMINAMATH_CALUDE_max_profit_price_l517_51773

/-- The cost of one item in yuan -/
def cost : ℝ := 30

/-- The number of items sold as a function of price -/
def itemsSold (x : ℝ) : ℝ := 200 - x

/-- The profit function -/
def profit (x : ℝ) : ℝ := (x - cost) * (itemsSold x)

/-- Theorem: The price that maximizes profit is 115 yuan -/
theorem max_profit_price : 
  ∃ (x : ℝ), x = 115 ∧ ∀ (y : ℝ), profit y ≤ profit x :=
sorry

end NUMINAMATH_CALUDE_max_profit_price_l517_51773


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l517_51712

theorem degree_to_radian_conversion :
  ((-300 : ℝ) * (π / 180)) = -(5 / 3 : ℝ) * π :=
by sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l517_51712


namespace NUMINAMATH_CALUDE_range_of_a_l517_51779

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - x - 2 ≤ 0 → x^2 - a*x - a - 2 ≤ 0) ↔ a ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l517_51779


namespace NUMINAMATH_CALUDE_average_score_is_1_9_l517_51746

/-- Represents the score distribution of a class -/
structure ScoreDistribution where
  total_students : ℕ
  score_3_percent : ℚ
  score_2_percent : ℚ
  score_1_percent : ℚ
  score_0_percent : ℚ

/-- Calculates the average score of a class given its score distribution -/
def average_score (sd : ScoreDistribution) : ℚ :=
  (3 * sd.score_3_percent + 2 * sd.score_2_percent + sd.score_1_percent) * sd.total_students / 100

/-- The theorem stating that the average score for the given distribution is 1.9 -/
theorem average_score_is_1_9 (sd : ScoreDistribution)
  (h1 : sd.total_students = 30)
  (h2 : sd.score_3_percent = 30)
  (h3 : sd.score_2_percent = 40)
  (h4 : sd.score_1_percent = 20)
  (h5 : sd.score_0_percent = 10) :
  average_score sd = 19/10 := by sorry

end NUMINAMATH_CALUDE_average_score_is_1_9_l517_51746


namespace NUMINAMATH_CALUDE_cos_10_cos_20_minus_sin_10_sin_20_l517_51791

theorem cos_10_cos_20_minus_sin_10_sin_20 :
  Real.cos (10 * π / 180) * Real.cos (20 * π / 180) -
  Real.sin (10 * π / 180) * Real.sin (20 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_10_cos_20_minus_sin_10_sin_20_l517_51791


namespace NUMINAMATH_CALUDE_number_of_tests_l517_51721

theorem number_of_tests (n : ℕ) (S : ℝ) : 
  (S + 97) / n = 90 → 
  (S + 73) / n = 87 → 
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_number_of_tests_l517_51721


namespace NUMINAMATH_CALUDE_complex_magnitude_example_l517_51729

theorem complex_magnitude_example : Complex.abs (-3 - (5/4)*Complex.I) = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_example_l517_51729


namespace NUMINAMATH_CALUDE_c_is_largest_l517_51742

-- Define the five numbers
def a : ℚ := 7.4683
def b : ℚ := 7 + 468/1000 + 3/9990  -- 7.468̅3
def c : ℚ := 7 + 46/100 + 83/9900   -- 7.46̅83
def d : ℚ := 7 + 4/10 + 683/999     -- 7.4̅683
def e : ℚ := 7 + 4683/9999          -- 7.̅4683

-- Theorem stating that c is the largest
theorem c_is_largest : c > a ∧ c > b ∧ c > d ∧ c > e := by sorry

end NUMINAMATH_CALUDE_c_is_largest_l517_51742


namespace NUMINAMATH_CALUDE_expression_factorization_l517_51711

theorem expression_factorization (x : ℝ) : 
  (15 * x^3 + 80 * x - 5) - (-4 * x^3 + 4 * x - 5) = 19 * x * (x^2 + 4) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l517_51711


namespace NUMINAMATH_CALUDE_range_of_a_l517_51750

theorem range_of_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x : ℝ, x > 0 → 9*x + a^2/x ≥ a^2 + 8) : 
  2 ≤ a ∧ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l517_51750


namespace NUMINAMATH_CALUDE_shelter_cat_dog_difference_l517_51720

/-- Given an animal shelter with a total of 60 animals and 40 cats,
    prove that the number of cats exceeds the number of dogs by 20. -/
theorem shelter_cat_dog_difference :
  let total_animals : ℕ := 60
  let num_cats : ℕ := 40
  let num_dogs : ℕ := total_animals - num_cats
  num_cats - num_dogs = 20 := by
  sorry

end NUMINAMATH_CALUDE_shelter_cat_dog_difference_l517_51720


namespace NUMINAMATH_CALUDE_smallest_base_for_fourth_power_l517_51767

theorem smallest_base_for_fourth_power : ∃ (N : ℤ) (x : ℕ),
  (∀ b : ℕ, b > 0 → (7 * b^2 + 7 * b + 7 = N ↔ N.toNat.digits b = [7, 7, 7])) ∧
  N = x^4 ∧
  (∀ b : ℕ, b > 0 ∧ b < 18 → ¬∃ y : ℕ, 7 * b^2 + 7 * b + 7 = y^4) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_fourth_power_l517_51767


namespace NUMINAMATH_CALUDE_max_value_on_interval_l517_51783

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

-- State the theorem
theorem max_value_on_interval :
  ∃ (M : ℝ), M = 5 ∧ 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → f x ≤ M) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ f x = M) :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_interval_l517_51783


namespace NUMINAMATH_CALUDE_smallest_k_for_single_root_l517_51788

-- Define the functions f and g
def f (x : ℝ) : ℝ := 41 * x^2 - 4 * x + 4
def g (x : ℝ) : ℝ := -2 * x^2 + x

-- Define the combined function h
def h (k : ℝ) (x : ℝ) : ℝ := f x + k * g x

-- Define the discriminant of h
def discriminant (k : ℝ) : ℝ := (k - 4)^2 - 4 * (41 - 2*k) * 4

-- Theorem statement
theorem smallest_k_for_single_root :
  ∃ d : ℝ, d = -40 ∧ 
  (∀ k : ℝ, (∃ x : ℝ, h k x = 0 ∧ (∀ y : ℝ, h k y = 0 → y = x)) → k ≥ d) ∧
  (∃ x : ℝ, h d x = 0 ∧ (∀ y : ℝ, h d y = 0 → y = x)) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_single_root_l517_51788


namespace NUMINAMATH_CALUDE_percentage_difference_l517_51759

theorem percentage_difference (x y : ℝ) (P : ℝ) (h1 : x = y - (P / 100) * y) (h2 : y = 2 * x) :
  P = 50 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l517_51759


namespace NUMINAMATH_CALUDE_complex_equation_solution_l517_51724

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l517_51724


namespace NUMINAMATH_CALUDE_speed_ratio_l517_51714

/-- The speed of object A -/
def v_A : ℝ := sorry

/-- The speed of object B -/
def v_B : ℝ := sorry

/-- The distance B is initially short of O -/
def initial_distance : ℝ := 600

/-- The time when A and B are first equidistant from O -/
def t1 : ℝ := 3

/-- The time when A and B are again equidistant from O -/
def t2 : ℝ := 12

/-- The theorem stating the ratio of speeds -/
theorem speed_ratio : v_A / v_B = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_l517_51714


namespace NUMINAMATH_CALUDE_max_color_difference_is_six_l517_51786

/-- Represents a basket of marbles -/
structure Basket where
  color1 : String
  count1 : Nat
  color2 : String
  count2 : Nat

/-- Calculates the difference between the counts of two colors in a basket -/
def colorDifference (basket : Basket) : Nat :=
  max basket.count1 basket.count2 - min basket.count1 basket.count2

/-- The three baskets given in the problem -/
def basketA : Basket := { color1 := "red", count1 := 4, color2 := "yellow", count2 := 2 }
def basketB : Basket := { color1 := "green", count1 := 6, color2 := "yellow", count2 := 1 }
def basketC : Basket := { color1 := "white", count1 := 3, color2 := "yellow", count2 := 9 }

/-- Theorem stating that the maximum difference between marble counts is 6 -/
theorem max_color_difference_is_six :
  max (colorDifference basketA) (max (colorDifference basketB) (colorDifference basketC)) = 6 := by
  sorry


end NUMINAMATH_CALUDE_max_color_difference_is_six_l517_51786


namespace NUMINAMATH_CALUDE_students_in_class_l517_51789

/-- Proves that the number of students in Ms. Leech's class is 30 -/
theorem students_in_class (num_boys : ℕ) (num_girls : ℕ) (cups_per_boy : ℕ) (total_cups : ℕ) :
  num_boys = 10 →
  num_girls = 2 * num_boys →
  cups_per_boy = 5 →
  total_cups = 90 →
  num_boys * cups_per_boy + num_girls * ((total_cups - num_boys * cups_per_boy) / num_girls) = total_cups →
  num_boys + num_girls = 30 := by
  sorry

end NUMINAMATH_CALUDE_students_in_class_l517_51789


namespace NUMINAMATH_CALUDE_grapes_cost_proof_l517_51793

/-- The amount Alyssa paid for grapes -/
def grapes_cost : ℝ := 12.08

/-- The amount Alyssa paid for cherries -/
def cherries_cost : ℝ := 9.85

/-- The total amount Alyssa spent -/
def total_cost : ℝ := 21.93

/-- Theorem: Given the total cost and the cost of cherries, prove that the cost of grapes is correct -/
theorem grapes_cost_proof : grapes_cost = total_cost - cherries_cost := by
  sorry

end NUMINAMATH_CALUDE_grapes_cost_proof_l517_51793


namespace NUMINAMATH_CALUDE_leftover_fraction_l517_51787

def fractions : List ℚ := [5/4, 17/6, -5/4, 10/7, 2/3, 14/8, -1/3, 5/3, -3/2]

def satisfies_property (a b : ℚ) : Bool :=
  a + b = 2/5 ∨ a - b = 2/5 ∨ a * b = 2/5 ∨ (b ≠ 0 ∧ a / b = 2/5)

theorem leftover_fraction :
  ∀ (pairs : List (ℚ × ℚ)),
    (∀ (pair : ℚ × ℚ), pair ∈ pairs → pair.1 ∈ fractions ∧ pair.2 ∈ fractions) →
    (∀ (pair : ℚ × ℚ), pair ∈ pairs → satisfies_property pair.1 pair.2) →
    (∀ (f : ℚ), f ∈ fractions → f ≠ -3/2 → ∃ (pair : ℚ × ℚ), pair ∈ pairs ∧ (f = pair.1 ∨ f = pair.2)) →
    List.length pairs = 4 →
    ¬∃ (pair : ℚ × ℚ), pair ∈ pairs ∧ (-3/2 = pair.1 ∨ -3/2 = pair.2) :=
by sorry

end NUMINAMATH_CALUDE_leftover_fraction_l517_51787


namespace NUMINAMATH_CALUDE_girls_in_class_l517_51704

theorem girls_in_class (num_boys : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ) :
  num_boys = 16 →
  ratio_boys = 4 →
  ratio_girls = 5 →
  (num_boys * ratio_girls) / ratio_boys = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_girls_in_class_l517_51704


namespace NUMINAMATH_CALUDE_log_base_six_two_point_five_l517_51796

theorem log_base_six_two_point_five (x : ℝ) :
  (Real.log x) / (Real.log 6) = 2.5 → x = 36 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_log_base_six_two_point_five_l517_51796


namespace NUMINAMATH_CALUDE_fibonacci_sum_odd_equals_next_l517_51752

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def sum_odd_fibonacci (n : ℕ) : ℕ :=
  1 + (List.range n).foldl (λ acc i => acc + fibonacci (2 * i + 3)) 0

theorem fibonacci_sum_odd_equals_next (n : ℕ) :
  sum_odd_fibonacci n = fibonacci (2 * n + 2) := by
  sorry

#eval fibonacci 2018
#eval sum_odd_fibonacci 1008

end NUMINAMATH_CALUDE_fibonacci_sum_odd_equals_next_l517_51752


namespace NUMINAMATH_CALUDE_pineapples_theorem_l517_51736

/-- Calculates the number of fresh pineapples left in a store. -/
def fresh_pineapples_left (initial : ℕ) (sold : ℕ) (rotten : ℕ) : ℕ :=
  initial - sold - rotten

/-- Proves that the number of fresh pineapples left is 29. -/
theorem pineapples_theorem :
  fresh_pineapples_left 86 48 9 = 29 := by
  sorry

end NUMINAMATH_CALUDE_pineapples_theorem_l517_51736


namespace NUMINAMATH_CALUDE_cube_edge_length_in_pyramid_l517_51715

/-- Represents a pyramid with an equilateral triangular base -/
structure EquilateralPyramid where
  base_side_length : ℝ
  apex_height : ℝ

/-- Represents a cube -/
structure Cube where
  edge_length : ℝ

/-- Theorem stating the edge length of the cube in the given pyramid configuration -/
theorem cube_edge_length_in_pyramid (p : EquilateralPyramid) (c : Cube) 
  (h1 : p.base_side_length = 3)
  (h2 : p.apex_height = 9)
  (h3 : c.edge_length * Real.sqrt 3 = p.apex_height) : 
  c.edge_length = 3 := by
  sorry


end NUMINAMATH_CALUDE_cube_edge_length_in_pyramid_l517_51715


namespace NUMINAMATH_CALUDE_proposition_equivalences_l517_51772

theorem proposition_equivalences (x y : ℝ) : 
  (((Real.sqrt (x - 2) + (y + 1)^2 = 0) → (x = 2 ∧ y = -1)) ↔
   ((x = 2 ∧ y = -1) → (Real.sqrt (x - 2) + (y + 1)^2 = 0))) ∧
  (((Real.sqrt (x - 2) + (y + 1)^2 ≠ 0) → (x ≠ 2 ∨ y ≠ -1)) ↔
   ((x ≠ 2 ∨ y ≠ -1) → (Real.sqrt (x - 2) + (y + 1)^2 ≠ 0))) :=
by sorry

#check proposition_equivalences

end NUMINAMATH_CALUDE_proposition_equivalences_l517_51772


namespace NUMINAMATH_CALUDE_remaining_nails_l517_51725

def initial_nails : ℕ := 400

def kitchen_repair (n : ℕ) : ℕ := n - (n * 30 / 100)

def fence_repair (n : ℕ) : ℕ := n - (n * 70 / 100)

def table_repair (n : ℕ) : ℕ := n - (n * 50 / 100)

def floorboard_repair (n : ℕ) : ℕ := n - (n * 25 / 100)

theorem remaining_nails :
  floorboard_repair (table_repair (fence_repair (kitchen_repair initial_nails))) = 32 := by
  sorry

end NUMINAMATH_CALUDE_remaining_nails_l517_51725


namespace NUMINAMATH_CALUDE_prob_A_more_points_theorem_l517_51708

/-- Represents a soccer tournament with given conditions -/
structure SoccerTournament where
  num_teams : Nat
  num_games_per_team : Nat
  prob_A_wins_B : ℝ
  prob_win_other_games : ℝ

/-- Calculates the probability that Team A ends up with more points than Team B -/
def prob_A_more_points_than_B (tournament : SoccerTournament) : ℝ :=
  sorry

/-- The main theorem stating the probability for Team A to end up with more points -/
theorem prob_A_more_points_theorem (tournament : SoccerTournament) :
  tournament.num_teams = 7 ∧
  tournament.num_games_per_team = 6 ∧
  tournament.prob_A_wins_B = 0.6 ∧
  tournament.prob_win_other_games = 0.5 →
  prob_A_more_points_than_B tournament = 779 / 1024 :=
  sorry

end NUMINAMATH_CALUDE_prob_A_more_points_theorem_l517_51708


namespace NUMINAMATH_CALUDE_license_plate_count_l517_51774

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def digits_count : ℕ := 5

/-- The number of letters in a license plate -/
def letters_count : ℕ := 3

/-- The number of possible positions for the letter block -/
def letter_block_positions : ℕ := digits_count + 1

/-- The total number of distinct license plates -/
def total_license_plates : ℕ := 
  letter_block_positions * (num_digits ^ digits_count) * (num_letters ^ letters_count)

theorem license_plate_count : total_license_plates = 105456000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l517_51774


namespace NUMINAMATH_CALUDE_range_of_a_l517_51738

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≥ 0}
def B (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (A a ∪ B a = Set.univ) → a ∈ Set.Iic 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l517_51738


namespace NUMINAMATH_CALUDE_power_sum_of_i_l517_51749

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^23 + i^58 = -1 - i := by sorry

end NUMINAMATH_CALUDE_power_sum_of_i_l517_51749


namespace NUMINAMATH_CALUDE_special_function_property_l517_51732

/-- A function that is even, has period 2, and is monotonically decreasing on [-3, -2] -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (∀ x, f (x + 2) = f x) ∧
  (∀ x y, -3 ≤ x ∧ x < y ∧ y ≤ -2 → f y < f x)

/-- Acute angle in a triangle -/
def is_acute_angle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < Real.pi / 2

theorem special_function_property 
  (f : ℝ → ℝ) 
  (h_f : special_function f) 
  (α β : ℝ) 
  (h_α : is_acute_angle α) 
  (h_β : is_acute_angle β) : 
  f (Real.sin α) > f (Real.cos β) := by
    sorry

end NUMINAMATH_CALUDE_special_function_property_l517_51732


namespace NUMINAMATH_CALUDE_two_percent_of_one_l517_51722

theorem two_percent_of_one : (2 : ℚ) / 100 = (2 : ℚ) / 100 * 1 := by sorry

end NUMINAMATH_CALUDE_two_percent_of_one_l517_51722


namespace NUMINAMATH_CALUDE_constant_remainder_iff_b_eq_l517_51781

/-- The dividend polynomial -/
def dividend (b x : ℚ) : ℚ := 8 * x^3 + 5 * x^2 + b * x - 8

/-- The divisor polynomial -/
def divisor (x : ℚ) : ℚ := 3 * x^2 - 2 * x + 4

/-- The remainder when dividend is divided by divisor -/
def remainder (b x : ℚ) : ℚ := dividend b x - divisor x * ((8/3) * x + 2/3)

theorem constant_remainder_iff_b_eq (b : ℚ) : 
  (∃ (c : ℚ), ∀ (x : ℚ), remainder b x = c) ↔ b = -98/9 := by sorry

end NUMINAMATH_CALUDE_constant_remainder_iff_b_eq_l517_51781


namespace NUMINAMATH_CALUDE_f_range_l517_51709

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos (3*x/2), Real.sin (3*x/2))
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos (x/2), -Real.sin (x/2))

noncomputable def f (x : ℝ) : ℝ :=
  (a x).1 * (b x).1 + (a x).2 * (b x).2 -
  Real.sqrt (((a x).1 - (b x).1)^2 + ((a x).2 - (b x).2)^2)

theorem f_range :
  ∀ y ∈ Set.Icc (-3 : ℝ) (-1/2),
    ∃ x ∈ Set.Ico (π/6 : ℝ) (2*π/3),
      f x = y :=
sorry

end NUMINAMATH_CALUDE_f_range_l517_51709


namespace NUMINAMATH_CALUDE_first_three_terms_b_is_geometric_T_sum_l517_51762

def sequence_a (n : ℕ) : ℝ := sorry

def S (n : ℕ) : ℝ := sorry

axiom S_def (n : ℕ) : S n = 2 * sequence_a n - 2 * n

theorem first_three_terms :
  sequence_a 1 = 2 ∧ sequence_a 2 = 6 ∧ sequence_a 3 = 14 := by sorry

def sequence_b (n : ℕ) : ℝ := sequence_a n + 2

theorem b_is_geometric :
  ∃ (r : ℝ), ∀ (n : ℕ), n ≥ 2 → sequence_b n = r * sequence_b (n-1) := by sorry

def T (n : ℕ) : ℝ := sorry

theorem T_sum (n : ℕ) :
  T n = (n + 1) * 2^(n + 2) + 4 - n * (n + 1) := by sorry

end NUMINAMATH_CALUDE_first_three_terms_b_is_geometric_T_sum_l517_51762


namespace NUMINAMATH_CALUDE_proposition_relationship_l517_51770

theorem proposition_relationship (p q : Prop) : 
  (¬p ∨ ¬q → ¬p ∧ ¬q) ∧ 
  ∃ (p q : Prop), (¬p ∧ ¬q) ∧ ¬(¬p ∨ ¬q) :=
by sorry

end NUMINAMATH_CALUDE_proposition_relationship_l517_51770


namespace NUMINAMATH_CALUDE_system_solution_iff_a_in_interval_l517_51754

/-- The system of equations has a solution for some b if and only if a is in the interval (-8, 7] -/
theorem system_solution_iff_a_in_interval (a : ℝ) : 
  (∃ (b x y : ℝ), x^2 + y^2 + 2*a*(a - x - y) = 64 ∧ y = 7 / ((x + b)^2 + 1)) ↔ 
  -8 < a ∧ a ≤ 7 := by sorry

end NUMINAMATH_CALUDE_system_solution_iff_a_in_interval_l517_51754


namespace NUMINAMATH_CALUDE_extracurricular_teams_problem_l517_51758

theorem extracurricular_teams_problem (total_activities : ℕ) 
  (initial_ratio_tt : ℕ) (initial_ratio_bb : ℕ) 
  (new_ratio_tt : ℕ) (new_ratio_bb : ℕ) 
  (transfer : ℕ) :
  total_activities = 38 →
  initial_ratio_tt = 7 →
  initial_ratio_bb = 3 →
  new_ratio_tt = 3 →
  new_ratio_bb = 2 →
  transfer = 8 →
  ∃ (tt_original bb_original : ℕ),
    tt_original * initial_ratio_bb = bb_original * initial_ratio_tt ∧
    (tt_original - transfer) * new_ratio_bb = (bb_original + transfer) * new_ratio_tt ∧
    tt_original = 35 ∧
    bb_original = 15 := by
  sorry

end NUMINAMATH_CALUDE_extracurricular_teams_problem_l517_51758


namespace NUMINAMATH_CALUDE_sufficient_condition_absolute_value_necessary_condition_inequality_l517_51771

-- Statement ③
theorem sufficient_condition_absolute_value (a b : ℝ) :
  a^2 ≠ b^2 → |a| = |b| :=
sorry

-- Statement ④
theorem necessary_condition_inequality (a b c : ℝ) :
  a * c^2 < b * c^2 → a < b :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_absolute_value_necessary_condition_inequality_l517_51771


namespace NUMINAMATH_CALUDE_expected_sum_of_marbles_l517_51705

/-- The set of marble numbers -/
def marbleNumbers : Finset ℕ := {2, 3, 4, 5, 6, 7}

/-- The sum of two different elements from the set -/
def pairSum (a b : ℕ) : ℕ := a + b

/-- The set of all possible pairs of different marbles -/
def marblePairs : Finset (ℕ × ℕ) :=
  (marbleNumbers.product marbleNumbers).filter (fun p => p.1 < p.2)

/-- The expected value of the sum of two randomly drawn marbles -/
def expectedSum : ℚ :=
  (marblePairs.sum (fun p => pairSum p.1 p.2)) / marblePairs.card

theorem expected_sum_of_marbles :
  expectedSum = 145 / 15 := by sorry

end NUMINAMATH_CALUDE_expected_sum_of_marbles_l517_51705


namespace NUMINAMATH_CALUDE_work_completion_time_l517_51792

/-- Work rates and completion times for a team project -/
theorem work_completion_time 
  (man_rate : ℚ) 
  (woman_rate : ℚ) 
  (girl_rate : ℚ) 
  (team_rate : ℚ) 
  (h1 : man_rate = 1/6) 
  (h2 : woman_rate = 1/18) 
  (h3 : girl_rate = 1/12) 
  (h4 : team_rate = 1/3) 
  (h5 : man_rate + woman_rate + girl_rate + (team_rate - man_rate - woman_rate - girl_rate) = team_rate) : 
  (1 / ((team_rate - man_rate - woman_rate - girl_rate) + 2 * girl_rate)) = 36/7 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l517_51792


namespace NUMINAMATH_CALUDE_perfect_squares_l517_51778

theorem perfect_squares (k : ℕ) (h1 : k > 0) (h2 : ∃ a : ℕ, k * (k + 1) = 3 * a^2) : 
  (∃ m : ℕ, k = 3 * m^2) ∧ (∃ n : ℕ, k + 1 = n^2) := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_l517_51778


namespace NUMINAMATH_CALUDE_alex_has_sixty_shells_l517_51701

/-- The number of seashells in a dozen -/
def dozen : ℕ := 12

/-- The number of seashells Mimi picked up -/
def mimi_shells : ℕ := 2 * dozen

/-- The number of seashells Kyle found -/
def kyle_shells : ℕ := 2 * mimi_shells

/-- The number of seashells Leigh grabbed -/
def leigh_shells : ℕ := kyle_shells / 3

/-- The number of seashells Alex unearthed -/
def alex_shells : ℕ := 3 * leigh_shells + mimi_shells / 2

/-- Theorem stating that Alex had 60 seashells -/
theorem alex_has_sixty_shells : alex_shells = 60 := by
  sorry

end NUMINAMATH_CALUDE_alex_has_sixty_shells_l517_51701


namespace NUMINAMATH_CALUDE_max_absolute_value_constrained_complex_l517_51739

theorem max_absolute_value_constrained_complex (z : ℂ) (h : Complex.abs (z - 2 * Complex.I) ≤ 1) :
  Complex.abs z ≤ 3 ∧ ∃ w : ℂ, Complex.abs (w - 2 * Complex.I) ≤ 1 ∧ Complex.abs w = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_absolute_value_constrained_complex_l517_51739


namespace NUMINAMATH_CALUDE_john_twice_james_age_john_twice_james_age_proof_l517_51730

/-- Proves that John will be twice as old as James in 15 years -/
theorem john_twice_james_age : ℕ → Prop :=
  fun years_until_twice_age : ℕ =>
    let john_current_age : ℕ := 39
    let james_brother_age : ℕ := 16
    let age_difference_james_brother : ℕ := 4
    let james_current_age : ℕ := james_brother_age - age_difference_james_brother
    let john_age_3_years_ago : ℕ := john_current_age - 3
    let james_age_in_future : ℕ → ℕ := fun x => james_current_age + x
    ∃ x : ℕ, john_age_3_years_ago = 2 * (james_age_in_future x) →
    (john_current_age + years_until_twice_age = 2 * (james_current_age + years_until_twice_age)) →
    years_until_twice_age = 15

/-- Proof of the theorem -/
theorem john_twice_james_age_proof : john_twice_james_age 15 := by
  sorry

end NUMINAMATH_CALUDE_john_twice_james_age_john_twice_james_age_proof_l517_51730


namespace NUMINAMATH_CALUDE_num_paths_through_F_and_H_l517_51776

/-- A point on the grid --/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Calculate the number of paths between two points on a grid --/
def numPaths (start finish : GridPoint) : ℕ :=
  Nat.choose (finish.x - start.x + finish.y - start.y) (finish.x - start.x)

/-- The grid layout --/
def E : GridPoint := ⟨0, 0⟩
def F : GridPoint := ⟨3, 2⟩
def H : GridPoint := ⟨5, 4⟩
def G : GridPoint := ⟨8, 4⟩

/-- The theorem to prove --/
theorem num_paths_through_F_and_H : 
  numPaths E F * numPaths F H * numPaths H G = 60 := by
  sorry

end NUMINAMATH_CALUDE_num_paths_through_F_and_H_l517_51776


namespace NUMINAMATH_CALUDE_not_mysterious_consecutive_odd_squares_diff_l517_51765

/-- A positive integer that can be expressed as the difference of squares of two consecutive even numbers. -/
def MysteriousNumber (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (2*k + 2)^2 - (2*k)^2 ∧ n > 0

/-- The difference of squares of two consecutive odd numbers. -/
def ConsecutiveOddSquaresDiff (k : ℤ) : ℤ :=
  (2*k + 1)^2 - (2*k - 1)^2

theorem not_mysterious_consecutive_odd_squares_diff :
  ∀ k : ℤ, ¬(MysteriousNumber (ConsecutiveOddSquaresDiff k).natAbs) :=
by sorry

end NUMINAMATH_CALUDE_not_mysterious_consecutive_odd_squares_diff_l517_51765


namespace NUMINAMATH_CALUDE_amelia_wins_probability_l517_51747

/-- Probability of Amelia's coin landing heads -/
def p_amelia : ℚ := 1/4

/-- Probability of Blaine's coin landing heads -/
def p_blaine : ℚ := 1/6

/-- Probability of Amelia winning -/
noncomputable def p_amelia_wins : ℚ := 2/3

/-- Theorem stating that the probability of Amelia winning is 2/3 -/
theorem amelia_wins_probability :
  p_amelia_wins = p_amelia * (1 - p_blaine) + p_amelia * p_blaine + 
  (1 - p_amelia) * (1 - p_blaine) * p_amelia_wins :=
by sorry

end NUMINAMATH_CALUDE_amelia_wins_probability_l517_51747


namespace NUMINAMATH_CALUDE_sector_area_l517_51734

/-- Given a sector with central angle 7/(2π) and arc length 7, its area is 7π. -/
theorem sector_area (central_angle : Real) (arc_length : Real) (area : Real) :
  central_angle = 7 / (2 * Real.pi) →
  arc_length = 7 →
  area = 7 * Real.pi :=
by
  sorry

#check sector_area

end NUMINAMATH_CALUDE_sector_area_l517_51734


namespace NUMINAMATH_CALUDE_convex_polygon_diagonal_triangles_l517_51717

theorem convex_polygon_diagonal_triangles (n : ℕ) (h : n = 2002) : 
  ¬ ∃ (num_all_diagonal_triangles : ℕ),
    num_all_diagonal_triangles = (n - 2) / 2 ∧
    num_all_diagonal_triangles * 2 = n - 2 :=
by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_diagonal_triangles_l517_51717


namespace NUMINAMATH_CALUDE_chairs_built_in_ten_days_l517_51723

/-- Calculates the number of chairs a worker can build in a given number of days -/
def chairs_built (hours_per_shift : ℕ) (hours_per_chair : ℕ) (days : ℕ) : ℕ :=
  (hours_per_shift * days) / hours_per_chair

/-- Proves that a worker working 8-hour shifts, taking 5 hours per chair, can build 16 chairs in 10 days -/
theorem chairs_built_in_ten_days :
  chairs_built 8 5 10 = 16 := by
  sorry

end NUMINAMATH_CALUDE_chairs_built_in_ten_days_l517_51723


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l517_51798

theorem quadratic_roots_range (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   (a - 1) * x^2 - 2 * x + 1 = 0 ∧ 
   (a - 1) * y^2 - 2 * y + 1 = 0) → 
  (a < 2 ∧ a ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l517_51798


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l517_51733

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l517_51733


namespace NUMINAMATH_CALUDE_intersection_A_complementB_l517_51744

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x < 0}
def B : Set ℝ := {x | x > 1}

-- Define the complement of B
def complementB : Set ℝ := {x | ¬ (x ∈ B)}

-- State the theorem
theorem intersection_A_complementB : A ∩ complementB = {x | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complementB_l517_51744


namespace NUMINAMATH_CALUDE_point_not_in_third_quadrant_l517_51728

-- Define the linear function
def f (x : ℝ) : ℝ := -x + 8

-- Define what it means for a point to be in the third quadrant
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

-- Theorem statement
theorem point_not_in_third_quadrant (x y : ℝ) (h : y = f x) : ¬ in_third_quadrant x y := by
  sorry

end NUMINAMATH_CALUDE_point_not_in_third_quadrant_l517_51728


namespace NUMINAMATH_CALUDE_quadratic_inequality_l517_51760

theorem quadratic_inequality (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) 
  (ha₁ : 0 < a₁) (hb₁ : 0 < b₁) (hc₁ : 0 < c₁)
  (ha₂ : 0 < a₂) (hb₂ : 0 < b₂) (hc₂ : 0 < c₂)
  (h₁ : b₁^2 ≤ a₁*c₁) (h₂ : b₂^2 ≤ a₂*c₂) :
  (a₁ + a₂ + 5) * (c₁ + c₂ + 2) > (b₁ + b₂ + 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l517_51760


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l517_51784

theorem not_sufficient_nor_necessary (a b : ℝ) : 
  ¬(∀ a b : ℝ, a^2 > b^2 → a > b) ∧ ¬(∀ a b : ℝ, a > b → a^2 > b^2) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l517_51784


namespace NUMINAMATH_CALUDE_halfway_fraction_l517_51713

theorem halfway_fraction : 
  let a := (3 : ℚ) / 4
  let b := (5 : ℚ) / 7
  let midpoint := (a + b) / 2
  midpoint = (41 : ℚ) / 56 := by sorry

end NUMINAMATH_CALUDE_halfway_fraction_l517_51713


namespace NUMINAMATH_CALUDE_components_upper_bound_l517_51755

/-- Represents a square grid with diagonals --/
structure DiagonalGrid (n : ℕ) where
  size : n > 8
  cells : Fin n → Fin n → Bool
  -- True represents one diagonal, False represents the other

/-- Counts the number of connected components in the grid --/
def countComponents (g : DiagonalGrid n) : ℕ := sorry

/-- Theorem stating that the number of components is not greater than n²/4 --/
theorem components_upper_bound (n : ℕ) (g : DiagonalGrid n) :
  countComponents g ≤ n^2 / 4 := by sorry

end NUMINAMATH_CALUDE_components_upper_bound_l517_51755


namespace NUMINAMATH_CALUDE_sum_of_edges_for_specific_solid_l517_51768

/-- A rectangular solid with dimensions in arithmetic progression -/
structure ArithmeticProgressionSolid where
  a : ℝ
  d : ℝ

/-- Volume of the solid -/
def volume (s : ArithmeticProgressionSolid) : ℝ :=
  (s.a - s.d) * s.a * (s.a + s.d)

/-- Surface area of the solid -/
def surfaceArea (s : ArithmeticProgressionSolid) : ℝ :=
  2 * ((s.a - s.d) * s.a + s.a * (s.a + s.d) + (s.a - s.d) * (s.a + s.d))

/-- Sum of lengths of all edges -/
def sumOfEdges (s : ArithmeticProgressionSolid) : ℝ :=
  4 * ((s.a - s.d) + s.a + (s.a + s.d))

/-- Theorem stating the relationship between volume, surface area, and sum of edges -/
theorem sum_of_edges_for_specific_solid :
  ∃ (s : ArithmeticProgressionSolid),
    volume s = 512 ∧
    surfaceArea s = 352 ∧
    sumOfEdges s = 12 * Real.sqrt 59 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_edges_for_specific_solid_l517_51768


namespace NUMINAMATH_CALUDE_total_pins_used_l517_51761

/-- The number of sides of a rectangle -/
def rectangle_sides : ℕ := 4

/-- The number of pins used on each side of the cardboard -/
def pins_per_side : ℕ := 35

/-- Theorem: The total number of pins used to attach a rectangular cardboard to a box -/
theorem total_pins_used : rectangle_sides * pins_per_side = 140 := by
  sorry

end NUMINAMATH_CALUDE_total_pins_used_l517_51761


namespace NUMINAMATH_CALUDE_accounting_majors_l517_51735

theorem accounting_majors (p q r s t u : ℕ) : 
  p * q * r * s * t * u = 51030 → 
  1 < p → p < q → q < r → r < s → s < t → t < u → 
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_accounting_majors_l517_51735


namespace NUMINAMATH_CALUDE_remainder_problem_l517_51766

theorem remainder_problem (x : ℤ) : (x + 11) % 31 = 18 → x % 62 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l517_51766


namespace NUMINAMATH_CALUDE_sheila_hourly_rate_l517_51775

/-- Sheila's work schedule and earnings -/
structure WorkSchedule where
  monday_hours : ℕ
  tuesday_hours : ℕ
  wednesday_hours : ℕ
  thursday_hours : ℕ
  friday_hours : ℕ
  weekly_earnings : ℕ

/-- Calculate total weekly hours -/
def total_weekly_hours (schedule : WorkSchedule) : ℕ :=
  schedule.monday_hours + schedule.tuesday_hours + schedule.wednesday_hours +
  schedule.thursday_hours + schedule.friday_hours

/-- Calculate hourly rate -/
def hourly_rate (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_weekly_hours schedule)

/-- Sheila's work schedule -/
def sheila_schedule : WorkSchedule :=
  { monday_hours := 8
    tuesday_hours := 6
    wednesday_hours := 8
    thursday_hours := 6
    friday_hours := 8
    weekly_earnings := 360 }

/-- Theorem: Sheila's hourly rate is $10 -/
theorem sheila_hourly_rate :
  hourly_rate sheila_schedule = 10 := by
  sorry


end NUMINAMATH_CALUDE_sheila_hourly_rate_l517_51775


namespace NUMINAMATH_CALUDE_fill_time_three_pipes_l517_51741

-- Define the tank's volume
variable (T : ℝ)

-- Define the rates at which pipes X, Y, and Z fill the tank
variable (X Y Z : ℝ)

-- Define the conditions
def condition1 : Prop := X + Y = T / 3
def condition2 : Prop := X + Z = T / 4
def condition3 : Prop := Y + Z = T / 2

-- State the theorem
theorem fill_time_three_pipes 
  (h1 : condition1 T X Y) 
  (h2 : condition2 T X Z) 
  (h3 : condition3 T Y Z) :
  1 / (X + Y + Z) = 24 / 13 := by
  sorry

end NUMINAMATH_CALUDE_fill_time_three_pipes_l517_51741


namespace NUMINAMATH_CALUDE_tan_315_degrees_l517_51777

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_315_degrees_l517_51777


namespace NUMINAMATH_CALUDE_sqrt_four_fourth_powers_sum_l517_51700

theorem sqrt_four_fourth_powers_sum : Real.sqrt (4^4 + 4^4 + 4^4 + 4^4) = 32 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_fourth_powers_sum_l517_51700


namespace NUMINAMATH_CALUDE_P_in_fourth_quadrant_iff_m_in_range_l517_51757

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point being in the fourth quadrant -/
def in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The point P with coordinates dependent on m -/
def P (m : ℝ) : Point :=
  { x := m + 3, y := m - 2 }

/-- Theorem stating the range of m for P to be in the fourth quadrant -/
theorem P_in_fourth_quadrant_iff_m_in_range (m : ℝ) :
  in_fourth_quadrant (P m) ↔ -3 < m ∧ m < 2 := by sorry

end NUMINAMATH_CALUDE_P_in_fourth_quadrant_iff_m_in_range_l517_51757


namespace NUMINAMATH_CALUDE_multiple_of_six_l517_51727

theorem multiple_of_six (n : ℤ) 
  (h : ∃ k : ℤ, (n^5 / 120) + (n^3 / 24) + (n / 30) = k) : 
  ∃ m : ℤ, n = 6 * m :=
by
  sorry

end NUMINAMATH_CALUDE_multiple_of_six_l517_51727


namespace NUMINAMATH_CALUDE_square_equals_cube_of_digit_sum_l517_51706

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem square_equals_cube_of_digit_sum (n : ℕ) :
  n ∈ Finset.range 1000 →
  (n^2 = (sum_of_digits n)^3) ↔ (n = 1 ∨ n = 27) := by sorry

end NUMINAMATH_CALUDE_square_equals_cube_of_digit_sum_l517_51706


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l517_51753

theorem geometric_sequence_problem (b : ℝ) : 
  b > 0 ∧ 
  (∃ r : ℝ, 160 * r = b ∧ b * r = 1) → 
  b = 4 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l517_51753


namespace NUMINAMATH_CALUDE_robert_reading_capacity_l517_51769

/-- The number of complete books Robert can read in a given time -/
def books_read (pages_per_hour : ℕ) (pages_per_book : ℕ) (available_hours : ℕ) : ℕ :=
  (pages_per_hour * available_hours) / pages_per_book

/-- Theorem: Robert can read 2 complete 360-page books in 8 hours at a rate of 120 pages per hour -/
theorem robert_reading_capacity :
  books_read 120 360 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_robert_reading_capacity_l517_51769


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l517_51707

theorem cyclic_sum_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_one : a + b + c + d = 1) :
  (a / (a^2 + 1)) + (b / (b^2 + 1)) + (c / (c^2 + 1)) + (d / (d^2 + 1)) ≤ 16/17 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l517_51707


namespace NUMINAMATH_CALUDE_expression_evaluation_l517_51731

theorem expression_evaluation : 
  let expr := 125 - 25 * 4
  expr = 25 := by sorry

#check expression_evaluation

end NUMINAMATH_CALUDE_expression_evaluation_l517_51731


namespace NUMINAMATH_CALUDE_segment_length_in_dihedral_angle_l517_51740

/-- Given a segment AB with ends on the faces of a dihedral angle φ, where the distances from A and B
    to the edge of the angle are a and b respectively, and the distance between the projections of A
    and B on the edge is c, the length of AB is equal to √(a² + b² + c² - 2ab cos φ). -/
theorem segment_length_in_dihedral_angle (φ a b c : ℝ) (h_φ : 0 < φ ∧ φ < π) 
    (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) :
  ∃ (AB : ℝ), AB = Real.sqrt (a^2 + b^2 + c^2 - 2 * a * b * Real.cos φ) := by
  sorry

end NUMINAMATH_CALUDE_segment_length_in_dihedral_angle_l517_51740


namespace NUMINAMATH_CALUDE_special_square_divisions_l517_51702

/-- Represents a 5x5 square with a 3x3 center and 1x3 rectangles on each side -/
structure SpecialSquare :=
  (size : Nat)
  (center_size : Nat)
  (side_rectangle_size : Nat)
  (h_size : size = 5)
  (h_center : center_size = 3)
  (h_side : side_rectangle_size = 3)

/-- Counts the number of ways to divide the SpecialSquare into 1x3 rectangles -/
def count_divisions (square : SpecialSquare) : Nat :=
  2

/-- Theorem stating that the number of ways to divide the SpecialSquare into 1x3 rectangles is 2 -/
theorem special_square_divisions (square : SpecialSquare) :
  count_divisions square = 2 := by
  sorry

end NUMINAMATH_CALUDE_special_square_divisions_l517_51702


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l517_51737

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a →
  (a 5 + a 6 + a 7 = 1) →
  (a 3 + a 9 = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l517_51737


namespace NUMINAMATH_CALUDE_max_hot_dogs_is_3250_l517_51743

/-- Represents the available pack sizes and their prices --/
structure PackInfo where
  size : Nat
  price : Rat

/-- The maximum number of hot dogs that can be purchased with the given budget --/
def maxHotDogs (packs : List PackInfo) (budget : Rat) : Nat :=
  sorry

/-- The available pack sizes and prices --/
def availablePacks : List PackInfo := [
  ⟨8, 155/100⟩,
  ⟨20, 305/100⟩,
  ⟨250, 2295/100⟩
]

/-- The budget in dollars --/
def totalBudget : Rat := 300

/-- Theorem stating that the maximum number of hot dogs that can be purchased is 3250 --/
theorem max_hot_dogs_is_3250 :
  maxHotDogs availablePacks totalBudget = 3250 := by sorry

end NUMINAMATH_CALUDE_max_hot_dogs_is_3250_l517_51743


namespace NUMINAMATH_CALUDE_max_d_value_l517_51719

def is_multiple_of_66 (n : ℕ) : Prop := n % 66 = 0

def has_form_4d645e (n : ℕ) (d e : ℕ) : Prop :=
  n = 400000 + 10000 * d + 6000 + 400 + 50 + e ∧ d < 10 ∧ e < 10

theorem max_d_value (n : ℕ) (d e : ℕ) :
  is_multiple_of_66 n → has_form_4d645e n d e → d ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_d_value_l517_51719


namespace NUMINAMATH_CALUDE_smallest_sum_mn_smallest_sum_value_unique_smallest_sum_l517_51785

theorem smallest_sum_mn (m n : ℕ) (h : 3 * n^3 = 5 * m^2) : 
  ∀ (a b : ℕ), (3 * a^3 = 5 * b^2) → m + n ≤ a + b :=
by
  sorry

theorem smallest_sum_value : 
  ∃ (m n : ℕ), (3 * n^3 = 5 * m^2) ∧ (m + n = 60) :=
by
  sorry

theorem unique_smallest_sum : 
  ∀ (m n : ℕ), (3 * n^3 = 5 * m^2) → m + n ≥ 60 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_mn_smallest_sum_value_unique_smallest_sum_l517_51785


namespace NUMINAMATH_CALUDE_amy_haircut_l517_51780

/-- Given an initial hair length and the amount cut off, calculates the final hair length -/
def final_hair_length (initial_length cut_off : ℕ) : ℕ :=
  initial_length - cut_off

/-- Proves that given an initial hair length of 11 inches and cutting off 4 inches, 
    the resulting hair length is 7 inches -/
theorem amy_haircut : final_hair_length 11 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_amy_haircut_l517_51780


namespace NUMINAMATH_CALUDE_students_taking_art_l517_51745

/-- Given a school with 400 students, where 120 take dance and 20% take music,
    prove that 200 students take art. -/
theorem students_taking_art (total : ℕ) (dance : ℕ) (music_percent : ℚ) :
  total = 400 →
  dance = 120 →
  music_percent = 1/5 →
  total - (dance + (music_percent * total).floor) = 200 := by
sorry

end NUMINAMATH_CALUDE_students_taking_art_l517_51745


namespace NUMINAMATH_CALUDE_b6_b8_value_l517_51710

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem b6_b8_value (a b : ℕ → ℝ) 
    (ha : arithmetic_sequence a)
    (hb : geometric_sequence b)
    (h1 : a 3 + a 11 = 8)
    (h2 : b 7 = a 7) :
  b 6 * b 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_b6_b8_value_l517_51710


namespace NUMINAMATH_CALUDE_rectangle_area_l517_51790

theorem rectangle_area (width : ℝ) (length : ℝ) (h1 : width = 4) (h2 : length = 3 * width) :
  width * length = 48 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l517_51790


namespace NUMINAMATH_CALUDE_kaleb_cherries_left_l517_51756

/-- Calculates the number of cherries Kaleb has left after eating some. -/
def cherries_left (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

/-- Theorem: Given Kaleb had 67 cherries initially and ate 25 cherries,
    the number of cherries he had left is equal to 42. -/
theorem kaleb_cherries_left : cherries_left 67 25 = 42 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_cherries_left_l517_51756


namespace NUMINAMATH_CALUDE_inequality_proof_l517_51751

theorem inequality_proof (x y : ℝ) (h1 : x > y) (h2 : x * y = 1) :
  (x^2 + y^2) / (x - y) ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l517_51751


namespace NUMINAMATH_CALUDE_set_equality_l517_51799

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | x < 1}

-- Define set N
def N : Set ℝ := {x | -1 < x ∧ x < 2}

-- State the theorem
theorem set_equality : {x : ℝ | x ≥ 2} = (M ∪ N)ᶜ := by sorry

end NUMINAMATH_CALUDE_set_equality_l517_51799


namespace NUMINAMATH_CALUDE_box_cost_is_111_kopecks_l517_51795

/-- The cost of a box of matches in kopecks -/
def box_cost : ℕ := sorry

/-- Nine boxes cost more than 9 rubles but less than 10 rubles -/
axiom nine_boxes_cost : 900 < 9 * box_cost ∧ 9 * box_cost < 1000

/-- Ten boxes cost more than 11 rubles but less than 12 rubles -/
axiom ten_boxes_cost : 1100 < 10 * box_cost ∧ 10 * box_cost < 1200

/-- The cost of one box of matches is 1 ruble 11 kopecks -/
theorem box_cost_is_111_kopecks : box_cost = 111 := by sorry

end NUMINAMATH_CALUDE_box_cost_is_111_kopecks_l517_51795


namespace NUMINAMATH_CALUDE_f_monotonically_decreasing_l517_51726

noncomputable def f (x : ℝ) : ℝ := 3 * (Real.sin x)^2 + 2 * (Real.sin x) * (Real.cos x) + (Real.cos x)^2 - 2

def monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem f_monotonically_decreasing (k : ℤ) :
  monotonically_decreasing f (π/3 + k*π) (5*π/3 + k*π) := by sorry

end NUMINAMATH_CALUDE_f_monotonically_decreasing_l517_51726


namespace NUMINAMATH_CALUDE_jerry_throws_before_office_l517_51764

def penalty_system (interrupt : ℕ) (insult : ℕ) (throw : ℕ) : ℕ :=
  5 * interrupt + 10 * insult + 25 * throw

def jerry_current_points : ℕ :=
  penalty_system 2 4 0

theorem jerry_throws_before_office : 
  ∃ (n : ℕ), 
    n = 2 ∧ 
    jerry_current_points + 25 * n < 100 ∧
    jerry_current_points + 25 * (n + 1) ≥ 100 :=
by sorry

end NUMINAMATH_CALUDE_jerry_throws_before_office_l517_51764
