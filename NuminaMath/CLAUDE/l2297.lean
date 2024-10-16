import Mathlib

namespace NUMINAMATH_CALUDE_unique_solution_l2297_229790

/-- Represents a pair of digits in base r -/
structure DigitPair (r : ℕ) where
  first : ℕ
  second : ℕ
  h_first : first < r
  h_second : second < r

/-- Constructs a number from repeating a digit pair n times in base r -/
def construct_number (r : ℕ) (pair : DigitPair r) (n : ℕ) : ℕ :=
  pair.first * r + pair.second

/-- Checks if a number consists of only ones in base r -/
def all_ones (r : ℕ) (x : ℕ) : Prop :=
  ∀ k, (x / r^k) % r = 1 ∨ (x / r^k) = 0

theorem unique_solution :
  ∀ (r : ℕ) (x : ℕ) (n : ℕ) (pair : DigitPair r),
    2 ≤ r →
    r ≤ 70 →
    x = construct_number r pair n →
    all_ones r (x^2) →
    (r = 7 ∧ x = 26) := by sorry

end NUMINAMATH_CALUDE_unique_solution_l2297_229790


namespace NUMINAMATH_CALUDE_probability_quarter_or_dime_l2297_229765

/-- Represents the types of coins in the jar -/
inductive Coin
  | Quarter
  | Dime
  | Nickel

/-- The value of each coin type in cents -/
def coinValue : Coin → ℕ
  | Coin.Quarter => 25
  | Coin.Dime => 10
  | Coin.Nickel => 5

/-- The total value of each coin type in the jar in cents -/
def totalValue : Coin → ℕ
  | Coin.Quarter => 500
  | Coin.Dime => 600
  | Coin.Nickel => 200

/-- The number of coins of each type in the jar -/
def coinCount (c : Coin) : ℕ := totalValue c / coinValue c

/-- The total number of coins in the jar -/
def totalCoins : ℕ := coinCount Coin.Quarter + coinCount Coin.Dime + coinCount Coin.Nickel

/-- The probability of selecting either a quarter or a dime from the jar -/
def probQuarterOrDime : ℚ :=
  (coinCount Coin.Quarter + coinCount Coin.Dime : ℚ) / totalCoins

theorem probability_quarter_or_dime :
  probQuarterOrDime = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_probability_quarter_or_dime_l2297_229765


namespace NUMINAMATH_CALUDE_value_of_a_l2297_229763

def U (a : ℝ) : Set ℝ := {2, 3, a^2 + 2*a - 3}
def A (a : ℝ) : Set ℝ := {2, |a + 1|}

theorem value_of_a (a : ℝ) : U a = A a ∪ {5} → a = -4 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2297_229763


namespace NUMINAMATH_CALUDE_race_time_patrick_l2297_229785

theorem race_time_patrick (patrick_time manu_time amy_time : ℕ) : 
  manu_time = patrick_time + 12 →
  amy_time * 2 = manu_time →
  amy_time = 36 →
  patrick_time = 60 := by
sorry

end NUMINAMATH_CALUDE_race_time_patrick_l2297_229785


namespace NUMINAMATH_CALUDE_triangle_problem_l2297_229774

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < C ∧ C < π / 2 →
  a * Real.sin A = b * Real.sin B * Real.sin C →
  b = Real.sqrt 2 * a →
  C = π / 6 ∧ c^2 / a^2 = 3 - Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2297_229774


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2297_229700

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def repeatingDecimalToRational (d : RepeatingDecimal) : ℚ := sorry

/-- The repeating decimal 7.316316316... -/
def ourDecimal : RepeatingDecimal := { integerPart := 7, repeatingPart := 316 }

theorem repeating_decimal_equals_fraction :
  repeatingDecimalToRational ourDecimal = 7309 / 999 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2297_229700


namespace NUMINAMATH_CALUDE_probability_two_rainy_days_l2297_229724

/-- Represents the weather condition for a day -/
inductive Weather
| Rainy
| NotRainy

/-- Represents the weather for three consecutive days -/
def ThreeDayWeather := (Weather × Weather × Weather)

/-- Checks if a ThreeDayWeather has exactly two rainy days -/
def hasTwoRainyDays (w : ThreeDayWeather) : Bool :=
  match w with
  | (Weather.Rainy, Weather.Rainy, Weather.NotRainy) => true
  | (Weather.Rainy, Weather.NotRainy, Weather.Rainy) => true
  | (Weather.NotRainy, Weather.Rainy, Weather.Rainy) => true
  | _ => false

/-- The total number of weather groups in the sample -/
def totalGroups : Nat := 20

/-- The number of groups with exactly two rainy days -/
def groupsWithTwoRainyDays : Nat := 5

/-- Theorem: The probability of exactly two rainy days out of three is 0.25 -/
theorem probability_two_rainy_days :
  (groupsWithTwoRainyDays : ℚ) / totalGroups = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_probability_two_rainy_days_l2297_229724


namespace NUMINAMATH_CALUDE_acid_mixture_problem_l2297_229757

/-- Represents the contents of a jar --/
structure Jar where
  volume : ℚ
  acid_concentration : ℚ

/-- Represents the problem setup --/
structure ProblemSetup where
  jar_a : Jar
  jar_b : Jar
  jar_c : Jar
  m : ℕ
  n : ℕ

/-- The initial setup of the problem --/
def initial_setup (k : ℚ) : ProblemSetup where
  jar_a := { volume := 4, acid_concentration := 45/100 }
  jar_b := { volume := 5, acid_concentration := 48/100 }
  jar_c := { volume := 1, acid_concentration := k/100 }
  m := 2
  n := 3

/-- The final state after mixing --/
def final_state (setup : ProblemSetup) : Prop :=
  let new_jar_a_volume := setup.jar_a.volume + setup.m / setup.n
  let new_jar_b_volume := setup.jar_b.volume + (1 - setup.m / setup.n)
  let new_jar_a_acid := setup.jar_a.volume * setup.jar_a.acid_concentration + 
                        setup.jar_c.volume * setup.jar_c.acid_concentration * (setup.m / setup.n)
  let new_jar_b_acid := setup.jar_b.volume * setup.jar_b.acid_concentration + 
                        setup.jar_c.volume * setup.jar_c.acid_concentration * (1 - setup.m / setup.n)
  (new_jar_a_acid / new_jar_a_volume = 1/2) ∧ (new_jar_b_acid / new_jar_b_volume = 1/2)

/-- The main theorem --/
theorem acid_mixture_problem (k : ℚ) :
  final_state (initial_setup k) → k + 2 + 3 = 85 := by
  sorry


end NUMINAMATH_CALUDE_acid_mixture_problem_l2297_229757


namespace NUMINAMATH_CALUDE_average_weight_calculation_l2297_229705

/-- Given the average weights of pairs of individuals and the weight of one individual,
    calculate the average weight of all three individuals. -/
theorem average_weight_calculation
  (avg_ab avg_bc b_weight : ℝ)
  (h_avg_ab : (a + b_weight) / 2 = avg_ab)
  (h_avg_bc : (b_weight + c) / 2 = avg_bc)
  (h_b : b_weight = 37)
  (a c : ℝ) :
  (a + b_weight + c) / 3 = 45 :=
by sorry


end NUMINAMATH_CALUDE_average_weight_calculation_l2297_229705


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l2297_229735

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + 3 * b = 1) :
  (2 / a + 3 / b) ≥ 26 + 12 * Real.sqrt 6 :=
by sorry

theorem min_value_achieved (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + 3 * b = 1) :
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + 3 * b₀ = 1 ∧ (2 / a₀ + 3 / b₀) = 26 + 12 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l2297_229735


namespace NUMINAMATH_CALUDE_manufacturer_measures_l2297_229784

def samples_A : List ℝ := [3, 4, 5, 6, 8, 8, 8, 10]
def samples_B : List ℝ := [4, 6, 6, 6, 8, 9, 12, 13]
def samples_C : List ℝ := [3, 3, 4, 7, 9, 10, 11, 12]

def claimed_lifespan : ℝ := 8

def mode (l : List ℝ) : ℝ := sorry
def mean (l : List ℝ) : ℝ := sorry
def median (l : List ℝ) : ℝ := sorry

theorem manufacturer_measures :
  mode samples_A = claimed_lifespan ∧
  mean samples_B = claimed_lifespan ∧
  median samples_C = claimed_lifespan :=
sorry

end NUMINAMATH_CALUDE_manufacturer_measures_l2297_229784


namespace NUMINAMATH_CALUDE_square_areas_tiles_l2297_229789

theorem square_areas_tiles (x : ℝ) : 
  x > 0 ∧ 
  x^2 + (x + 12)^2 = 2120 → 
  x = 26 ∧ x + 12 = 38 := by
sorry

end NUMINAMATH_CALUDE_square_areas_tiles_l2297_229789


namespace NUMINAMATH_CALUDE_identity_implies_a_minus_b_zero_l2297_229742

theorem identity_implies_a_minus_b_zero 
  (x : ℚ) 
  (h_pos : x > 0) 
  (h_identity : ∀ x > 0, a / (2^x - 1) + b / (2^x + 2) = (2 * 2^x + 1) / ((2^x - 1) * (2^x + 2))) : 
  a - b = 0 :=
by sorry

end NUMINAMATH_CALUDE_identity_implies_a_minus_b_zero_l2297_229742


namespace NUMINAMATH_CALUDE_multiplicative_inverse_600_mod_4901_l2297_229764

theorem multiplicative_inverse_600_mod_4901 :
  ∃ n : ℕ, n < 4901 ∧ (600 * n) % 4901 = 1 ∧ n = 3196 := by sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_600_mod_4901_l2297_229764


namespace NUMINAMATH_CALUDE_sqrt_D_irrational_l2297_229776

/-- Given a real number x, D is defined as a² + b² + c², where a = x, b = x + 2, and c = a + b -/
def D (x : ℝ) : ℝ :=
  let a := x
  let b := x + 2
  let c := a + b
  a^2 + b^2 + c^2

/-- Theorem stating that the square root of D is always irrational for any real input x -/
theorem sqrt_D_irrational (x : ℝ) : Irrational (Real.sqrt (D x)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_D_irrational_l2297_229776


namespace NUMINAMATH_CALUDE_card_selection_ways_l2297_229780

theorem card_selection_ways (left_cards right_cards : ℕ) 
  (h1 : left_cards = 15) 
  (h2 : right_cards = 20) : 
  left_cards + right_cards = 35 := by
  sorry

end NUMINAMATH_CALUDE_card_selection_ways_l2297_229780


namespace NUMINAMATH_CALUDE_problem_solution_l2297_229753

theorem problem_solution (n m q q' r r' : ℕ) : 
  n > m ∧ m > 1 ∧
  n = q * m + r ∧ r < m ∧
  n - 1 = q' * m + r' ∧ r' < m ∧
  q + q' = 99 ∧ r + r' = 99 →
  n = 5000 ∧ ∃ k : ℕ, 2 * n = k * k :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2297_229753


namespace NUMINAMATH_CALUDE_even_increasing_function_solution_set_l2297_229736

-- Define the function f
def f (a b x : ℝ) : ℝ := (x - 2) * (a * x + b)

-- State the theorem
theorem even_increasing_function_solution_set
  (a b : ℝ)
  (h_even : ∀ x, f a b x = f a b (-x))
  (h_increasing : ∀ x y, 0 < x → x < y → f a b x < f a b y)
  : {x : ℝ | f a b (2 - x) > 0} = {x : ℝ | x < 0 ∨ x > 4} :=
by sorry

end NUMINAMATH_CALUDE_even_increasing_function_solution_set_l2297_229736


namespace NUMINAMATH_CALUDE_first_valve_fills_in_two_hours_l2297_229758

/-- Represents the time in hours taken by the first valve to fill the pool -/
def first_valve_fill_time (pool_capacity : ℝ) (both_valves_fill_time : ℝ) (valve_difference : ℝ) : ℝ :=
  2

/-- Theorem stating that under given conditions, the first valve takes 2 hours to fill the pool -/
theorem first_valve_fills_in_two_hours 
  (pool_capacity : ℝ) 
  (both_valves_fill_time : ℝ) 
  (valve_difference : ℝ) 
  (h1 : pool_capacity = 12000)
  (h2 : both_valves_fill_time = 48 / 60) -- Convert 48 minutes to hours
  (h3 : valve_difference = 50) :
  first_valve_fill_time pool_capacity both_valves_fill_time valve_difference = 2 := by
  sorry

#check first_valve_fills_in_two_hours

end NUMINAMATH_CALUDE_first_valve_fills_in_two_hours_l2297_229758


namespace NUMINAMATH_CALUDE_f_2x_l2297_229738

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem f_2x (x : ℝ) : f (2 * x) = 4 * x^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_f_2x_l2297_229738


namespace NUMINAMATH_CALUDE_pikes_caught_l2297_229745

theorem pikes_caught (total_fishes sturgeons herrings : ℕ) 
  (h1 : total_fishes = 145)
  (h2 : sturgeons = 40)
  (h3 : herrings = 75) :
  total_fishes - (sturgeons + herrings) = 30 := by
  sorry

end NUMINAMATH_CALUDE_pikes_caught_l2297_229745


namespace NUMINAMATH_CALUDE_area_inside_EFG_outside_AFD_l2297_229770

/-- Square ABCD with side length 36 -/
def square_side_length : ℝ := 36

/-- Point E is on side AB, 12 units from B -/
def distance_E_from_B : ℝ := 12

/-- Point F is the midpoint of side BC -/
def F_is_midpoint : Prop := True

/-- Point G is on side CD, 12 units from C -/
def distance_G_from_C : ℝ := 12

/-- The area of the region inside triangle EFG and outside triangle AFD -/
def area_difference : ℝ := 0

theorem area_inside_EFG_outside_AFD :
  square_side_length = 36 →
  distance_E_from_B = 12 →
  F_is_midpoint →
  distance_G_from_C = 12 →
  area_difference = 0 := by
  sorry

end NUMINAMATH_CALUDE_area_inside_EFG_outside_AFD_l2297_229770


namespace NUMINAMATH_CALUDE_average_water_added_l2297_229797

def water_day1 : ℝ := 318
def water_day2 : ℝ := 312
def water_day3_morning : ℝ := 180
def water_day3_afternoon : ℝ := 162
def num_days : ℝ := 3

theorem average_water_added (water_day1 water_day2 water_day3_morning water_day3_afternoon num_days : ℝ) :
  (water_day1 + water_day2 + water_day3_morning + water_day3_afternoon) / num_days = 324 := by
  sorry

end NUMINAMATH_CALUDE_average_water_added_l2297_229797


namespace NUMINAMATH_CALUDE_total_cost_is_1027_2_l2297_229791

/-- The cost relationship between mangos, rice, and flour -/
structure CostRelationship where
  mango_cost : ℝ  -- Cost per kg of mangos
  rice_cost : ℝ   -- Cost per kg of rice
  flour_cost : ℝ  -- Cost per kg of flour
  mango_rice_relation : 10 * mango_cost = 24 * rice_cost
  flour_rice_relation : 6 * flour_cost = 2 * rice_cost
  flour_cost_value : flour_cost = 24

/-- The total cost of 4 kg of mangos, 3 kg of rice, and 5 kg of flour -/
def total_cost (cr : CostRelationship) : ℝ :=
  4 * cr.mango_cost + 3 * cr.rice_cost + 5 * cr.flour_cost

/-- Theorem stating that the total cost is $1027.2 -/
theorem total_cost_is_1027_2 (cr : CostRelationship) :
  total_cost cr = 1027.2 := by
  sorry

#check total_cost_is_1027_2

end NUMINAMATH_CALUDE_total_cost_is_1027_2_l2297_229791


namespace NUMINAMATH_CALUDE_number_problem_l2297_229766

theorem number_problem (x : ℝ) : (0.95 * x - 12 = 178) → x = 200 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2297_229766


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2297_229732

/-- An arithmetic sequence with the given property has a common difference of 3. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h : a 2015 = a 2013 + 6)  -- The given condition
  : ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2297_229732


namespace NUMINAMATH_CALUDE_parallel_lines_and_planes_l2297_229771

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields here
  
/-- Represents a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields here

/-- Returns whether two lines are parallel -/
def are_parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Returns whether a line is parallel to a plane -/
def line_parallel_to_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Returns whether a line is a subset of a plane -/
def line_subset_of_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

theorem parallel_lines_and_planes 
  (a b : Line3D) (α : Plane3D) 
  (h : line_subset_of_plane b α) :
  ¬(∀ (a b : Line3D) (α : Plane3D), are_parallel a b → line_parallel_to_plane a α) ∧
  ¬(∀ (a b : Line3D) (α : Plane3D), line_parallel_to_plane a α → are_parallel a b) :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_and_planes_l2297_229771


namespace NUMINAMATH_CALUDE_intercept_sum_mod_50_l2297_229708

theorem intercept_sum_mod_50 : ∃! (x₀ y₀ : ℕ), 
  x₀ < 50 ∧ y₀ < 50 ∧ 
  (7 * x₀ ≡ 2 [MOD 50]) ∧
  (3 * y₀ ≡ 48 [MOD 50]) ∧
  ((x₀ + y₀) ≡ 2 [MOD 50]) := by
sorry

end NUMINAMATH_CALUDE_intercept_sum_mod_50_l2297_229708


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2297_229729

theorem polynomial_simplification (x : ℝ) :
  (x^5 + 3*x^4 + x^2 + 13) + (x^5 - 4*x^4 + x^3 - x^2 + 15) = 2*x^5 - x^4 + x^3 + 28 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2297_229729


namespace NUMINAMATH_CALUDE_no_natural_square_diff_2014_l2297_229761

theorem no_natural_square_diff_2014 : ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_square_diff_2014_l2297_229761


namespace NUMINAMATH_CALUDE_single_elimination_tournament_games_l2297_229752

/-- Calculates the number of games needed in a single-elimination tournament. -/
def gamesNeeded (n : ℕ) : ℕ := n - 1

/-- Theorem: In a single-elimination tournament with 512 players, 511 games are needed to crown the champion. -/
theorem single_elimination_tournament_games :
  gamesNeeded 512 = 511 := by
  sorry

end NUMINAMATH_CALUDE_single_elimination_tournament_games_l2297_229752


namespace NUMINAMATH_CALUDE_one_point_six_million_scientific_notation_l2297_229718

theorem one_point_six_million_scientific_notation :
  (1.6 : ℝ) * (1000000 : ℝ) = (1.6 : ℝ) * (10 : ℝ) ^ 6 := by
  sorry

end NUMINAMATH_CALUDE_one_point_six_million_scientific_notation_l2297_229718


namespace NUMINAMATH_CALUDE_square_greater_than_self_for_x_greater_than_one_l2297_229781

theorem square_greater_than_self_for_x_greater_than_one (x : ℝ) : x > 1 → x^2 > x := by
  sorry

end NUMINAMATH_CALUDE_square_greater_than_self_for_x_greater_than_one_l2297_229781


namespace NUMINAMATH_CALUDE_expression_nonnegative_iff_l2297_229707

theorem expression_nonnegative_iff (x : ℝ) :
  (x - 15 * x^2 + 56 * x^3) / (10 - x^3) ≥ 0 ↔ x ∈ Set.Icc 0 (1/8) :=
by sorry

end NUMINAMATH_CALUDE_expression_nonnegative_iff_l2297_229707


namespace NUMINAMATH_CALUDE_triangle_side_length_l2297_229739

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if 2asinB = √3 * b, b + c = 5, and bc = 6, then a = √7 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  0 < A ∧ A < π/2 →  -- ABC is acute
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  a > 0 ∧ b > 0 ∧ c > 0 →  -- sides are positive
  2 * a * Real.sin B = Real.sqrt 3 * b →
  b + c = 5 →
  b * c = 6 →
  a = Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2297_229739


namespace NUMINAMATH_CALUDE_cameron_total_questions_l2297_229717

def usual_questions_per_tourist : ℕ := 2

def group_size_1 : ℕ := 6
def group_size_2 : ℕ := 11
def group_size_3 : ℕ := 8
def group_size_4 : ℕ := 7

def inquisitive_tourist_multiplier : ℕ := 3

theorem cameron_total_questions :
  let group_1_questions := group_size_1 * usual_questions_per_tourist
  let group_2_questions := group_size_2 * usual_questions_per_tourist
  let group_3_questions := (group_size_3 - 1) * usual_questions_per_tourist +
                           usual_questions_per_tourist * inquisitive_tourist_multiplier
  let group_4_questions := group_size_4 * usual_questions_per_tourist
  group_1_questions + group_2_questions + group_3_questions + group_4_questions = 68 := by
  sorry

end NUMINAMATH_CALUDE_cameron_total_questions_l2297_229717


namespace NUMINAMATH_CALUDE_only_B_is_equation_l2297_229798

-- Define what an equation is
def is_equation (e : String) : Prop :=
  ∃ (lhs rhs : String), e = lhs ++ "=" ++ rhs

-- Define the given expressions
def expr_A : String := "x-6"
def expr_B : String := "3r+y=5"
def expr_C : String := "-3+x>-2"
def expr_D : String := "4/6=2/3"

-- Theorem statement
theorem only_B_is_equation :
  is_equation expr_B ∧
  ¬is_equation expr_A ∧
  ¬is_equation expr_C ∧
  ¬is_equation expr_D :=
by sorry

end NUMINAMATH_CALUDE_only_B_is_equation_l2297_229798


namespace NUMINAMATH_CALUDE_min_phase_shift_l2297_229713

/-- Given a sinusoidal function with a phase shift, prove that under certain symmetry conditions, 
    the smallest possible absolute value of the phase shift is π/4. -/
theorem min_phase_shift (φ : ℝ) : 
  (∀ x, 3 * Real.sin (3 * (x - π/4) + φ) = 3 * Real.sin (3 * (2*π/3 - x) + φ)) →
  (∃ k : ℤ, φ = k * π - π/4) →
  ∃ ψ : ℝ, abs ψ = π/4 ∧ (∀ θ : ℝ, (∃ k : ℤ, θ = k * π - π/4) → abs θ ≥ abs ψ) :=
by sorry

end NUMINAMATH_CALUDE_min_phase_shift_l2297_229713


namespace NUMINAMATH_CALUDE_bike_trip_distance_l2297_229743

/-- Calculates the total distance traveled given outbound and return times and average speed -/
def total_distance (outbound_time return_time : ℚ) (average_speed : ℚ) : ℚ :=
  let total_time := (outbound_time + return_time) / 60
  total_time * average_speed

/-- Proves that the total distance traveled is 4 miles given the specified conditions -/
theorem bike_trip_distance :
  let outbound_time : ℚ := 15
  let return_time : ℚ := 25
  let average_speed : ℚ := 6
  total_distance outbound_time return_time average_speed = 4 := by
  sorry

#eval total_distance 15 25 6

end NUMINAMATH_CALUDE_bike_trip_distance_l2297_229743


namespace NUMINAMATH_CALUDE_money_distribution_l2297_229768

theorem money_distribution (a b c d : ℚ) : 
  a = (1 : ℚ) / 3 * (b + c + d) →
  b = (2 : ℚ) / 7 * (a + c + d) →
  c = (3 : ℚ) / 11 * (a + b + d) →
  a = b + 20 →
  b = c + 15 →
  a + b + c + d = 720 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l2297_229768


namespace NUMINAMATH_CALUDE_beaver_dam_theorem_l2297_229777

/-- The number of hours it takes the first group of beavers to build the dam -/
def first_group_time : ℝ := 8

/-- The number of beavers in the second group -/
def second_group_size : ℝ := 36

/-- The number of hours it takes the second group of beavers to build the dam -/
def second_group_time : ℝ := 4

/-- The number of beavers in the first group -/
def first_group_size : ℝ := 18

theorem beaver_dam_theorem :
  first_group_size * first_group_time = second_group_size * second_group_time :=
by sorry

#check beaver_dam_theorem

end NUMINAMATH_CALUDE_beaver_dam_theorem_l2297_229777


namespace NUMINAMATH_CALUDE_unique_quadratic_family_l2297_229706

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The property that the product of roots equals the sum of coefficients -/
def roots_product_equals_coeff_sum (p : QuadraticPolynomial) : Prop :=
  ∃ r s : ℝ, r * s = p.a + p.b + p.c ∧ p.a * r^2 + p.b * r + p.c = 0 ∧ p.a * s^2 + p.b * s + p.c = 0

/-- The theorem stating that there's exactly one family of quadratic polynomials satisfying the condition -/
theorem unique_quadratic_family :
  ∃! f : ℝ → QuadraticPolynomial,
    (∀ c : ℝ, (f c).a = 1 ∧ (f c).b = -1 ∧ (f c).c = c) ∧
    (∀ p : QuadraticPolynomial, roots_product_equals_coeff_sum p ↔ ∃ c : ℝ, p = f c) :=
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_family_l2297_229706


namespace NUMINAMATH_CALUDE_line_equation_l2297_229747

theorem line_equation (slope_angle : Real) (y_intercept : Real) :
  slope_angle = Real.pi / 4 → y_intercept = 2 →
  ∃ f : Real → Real, f = λ x => x + 2 :=
by
  sorry

end NUMINAMATH_CALUDE_line_equation_l2297_229747


namespace NUMINAMATH_CALUDE_distinct_collections_count_l2297_229788

def word : String := "COMPUTATIONS"

def vowels : Finset Char := {'O', 'U', 'A', 'I'}
def consonants : Multiset Char := {'C', 'M', 'P', 'T', 'T', 'S', 'N'}

def vowel_count : Nat := 4
def consonant_count : Nat := 11

def selected_vowels : Nat := 3
def selected_consonants : Nat := 4

theorem distinct_collections_count :
  (Nat.choose vowel_count selected_vowels) *
  (Nat.choose (consonant_count - 1) selected_consonants +
   Nat.choose (consonant_count - 1) (selected_consonants - 1) +
   Nat.choose (consonant_count - 1) (selected_consonants - 2)) = 200 := by
  sorry

end NUMINAMATH_CALUDE_distinct_collections_count_l2297_229788


namespace NUMINAMATH_CALUDE_ellipse_m_range_l2297_229716

/-- The equation of an ellipse with foci on the x-axis -/
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (10 - m) + y^2 / (m - 2) = 1

/-- Conditions for the ellipse -/
def ellipse_conditions (m : ℝ) : Prop :=
  10 - m > 0 ∧ m - 2 > 0 ∧ 10 - m > m - 2

/-- The range of m for which the ellipse exists -/
theorem ellipse_m_range :
  ∀ m : ℝ, ellipse_conditions m ↔ 2 < m ∧ m < 6 :=
sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l2297_229716


namespace NUMINAMATH_CALUDE_rectangular_yard_area_l2297_229714

theorem rectangular_yard_area (w : ℝ) (l : ℝ) : 
  l = 2 * w + 30 →
  2 * w + 2 * l = 700 →
  w * l = 233600 / 9 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_yard_area_l2297_229714


namespace NUMINAMATH_CALUDE_cubic_roots_sum_min_l2297_229786

theorem cubic_roots_sum_min (a : ℝ) (x₁ x₂ x₃ : ℝ) (h_pos : a > 0) 
  (h_roots : x₁^3 - a*x₁^2 + a*x₁ - a = 0 ∧ 
             x₂^3 - a*x₂^2 + a*x₂ - a = 0 ∧ 
             x₃^3 - a*x₃^2 + a*x₃ - a = 0) : 
  ∃ (m : ℝ), m = -4 ∧ ∀ (y : ℝ), y ≥ m → x₁^3 + x₂^3 + x₃^3 - 3*x₁*x₂*x₃ ≥ y :=
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_min_l2297_229786


namespace NUMINAMATH_CALUDE_square_sum_given_product_and_sum_of_squares_l2297_229733

theorem square_sum_given_product_and_sum_of_squares (a b : ℝ) 
  (h1 : a * b = 3) 
  (h2 : a^2 * b + a * b^2 = 15) : 
  a^2 + b^2 = 19 := by sorry

end NUMINAMATH_CALUDE_square_sum_given_product_and_sum_of_squares_l2297_229733


namespace NUMINAMATH_CALUDE_sweets_per_person_l2297_229726

/-- Represents the number of sweets Jennifer has of each color -/
structure Sweets :=
  (green : ℕ)
  (blue : ℕ)
  (yellow : ℕ)

/-- Calculates the total number of sweets -/
def total_sweets (s : Sweets) : ℕ := s.green + s.blue + s.yellow

/-- Represents the number of people sharing the sweets -/
def num_people : ℕ := 4

/-- Jennifer's sweets -/
def jennifer_sweets : Sweets := ⟨212, 310, 502⟩

/-- Theorem: Each person gets 256 sweets when Jennifer's sweets are shared equally -/
theorem sweets_per_person :
  (total_sweets jennifer_sweets) / num_people = 256 := by sorry

end NUMINAMATH_CALUDE_sweets_per_person_l2297_229726


namespace NUMINAMATH_CALUDE_bryans_offer_l2297_229775

/-- Represents the problem of determining Bryan's offer for half of Peggy's record collection. -/
theorem bryans_offer (total_records : ℕ) (sammys_price : ℚ) (bryans_uninterested_price : ℚ) 
  (profit_difference : ℚ) (h1 : total_records = 200) (h2 : sammys_price = 4) 
  (h3 : bryans_uninterested_price = 1) (h4 : profit_difference = 100) : 
  ∃ (bryans_interested_price : ℚ),
    sammys_price * total_records - 
    (bryans_interested_price * (total_records / 2) + 
     bryans_uninterested_price * (total_records / 2)) = profit_difference ∧
    bryans_interested_price = 6 := by
  sorry

end NUMINAMATH_CALUDE_bryans_offer_l2297_229775


namespace NUMINAMATH_CALUDE_reptiles_in_swamps_l2297_229711

theorem reptiles_in_swamps (num_swamps : ℕ) (reptiles_per_swamp : ℕ) :
  num_swamps = 4 →
  reptiles_per_swamp = 356 →
  num_swamps * reptiles_per_swamp = 1424 := by
  sorry

end NUMINAMATH_CALUDE_reptiles_in_swamps_l2297_229711


namespace NUMINAMATH_CALUDE_xy_inequality_and_equality_l2297_229712

theorem xy_inequality_and_equality (x y : ℝ) (h : (x + 1) * (y + 2) = 8) :
  ((x * y - 10)^2 ≥ 64) ∧
  ((x * y - 10)^2 = 64 ↔ (x = 1 ∧ y = 2) ∨ (x = -3 ∧ y = -6)) := by
  sorry

end NUMINAMATH_CALUDE_xy_inequality_and_equality_l2297_229712


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2297_229769

-- Define the line l
def line_l (x y : ℝ) : Prop := y = (1/2) * x

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 1

-- Define the line ST
def line_ST (x y : ℝ) : Prop := y = -2 * x + 11/2

-- Theorem statement
theorem tangent_line_equation 
  (h1 : line_l 2 1)
  (h2 : line_l 6 3)
  (h3 : ∃ (x₀ y₀ : ℝ), line_l x₀ y₀ ∧ circle_C x₀ y₀)
  (h4 : circle_C 2 0) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_ST x₁ y₁ ∧ line_ST x₂ y₂ ∧
    line_ST 6 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2297_229769


namespace NUMINAMATH_CALUDE_theta_range_l2297_229723

theorem theta_range (θ : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, x^2 * Real.cos θ - x*(1-x) + (1-x)^2 * Real.sin θ > 0) →
  ∃ k : ℤ, 2*k*Real.pi + Real.pi/12 < θ ∧ θ < 2*k*Real.pi + 5*Real.pi/12 :=
by sorry

end NUMINAMATH_CALUDE_theta_range_l2297_229723


namespace NUMINAMATH_CALUDE_money_sum_existence_l2297_229730

theorem money_sum_existence : ∃ (k n : ℕ), 
  1 ≤ k ∧ k ≤ 9 ∧ n ≥ 1 ∧
  (k * (100 * n + 10 + 1) = 10666612) ∧
  (k * (n + 2) = (1 + 0 + 6 + 6 + 6 + 6 + 1 + 2)) :=
sorry

end NUMINAMATH_CALUDE_money_sum_existence_l2297_229730


namespace NUMINAMATH_CALUDE_inequality_proof_l2297_229748

open BigOperators

theorem inequality_proof (n : ℕ) (δ : ℝ) (a b : ℕ → ℝ) 
  (h_pos_a : ∀ i ∈ Finset.range (n + 1), a i > 0)
  (h_pos_b : ∀ i ∈ Finset.range (n + 1), b i > 0)
  (h_delta : ∀ i ∈ Finset.range n, b (i + 1) - b i ≥ δ)
  (h_delta_pos : δ > 0)
  (h_sum_a : ∑ i in Finset.range n, a i = 1) :
  ∑ i in Finset.range n, (i + 1 : ℝ) * (∏ j in Finset.range (i + 1), (a j * b j)) ^ (1 / (i + 1 : ℝ)) / (b (i + 1) * b i) < 1 / δ :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2297_229748


namespace NUMINAMATH_CALUDE_pencil_pen_cost_l2297_229795

/-- Given the cost of pencils and pens, calculate the cost of a different combination -/
theorem pencil_pen_cost (p q : ℝ) 
  (h1 : 3 * p + 2 * q = 4.30)
  (h2 : 2 * p + 3 * q = 4.05) :
  4 * p + 3 * q = 5.97 := by
  sorry

end NUMINAMATH_CALUDE_pencil_pen_cost_l2297_229795


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2297_229751

/-- The quadratic equation (k-2)x^2 + 3x + k^2 - 4 = 0 has one solution as x = 0 -/
def has_zero_solution (k : ℝ) : Prop :=
  k^2 - 4 = 0

/-- The coefficient of x^2 is not zero -/
def is_quadratic (k : ℝ) : Prop :=
  k - 2 ≠ 0

theorem quadratic_equation_solution :
  ∀ k : ℝ, has_zero_solution k → is_quadratic k → k = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2297_229751


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2297_229746

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 / x + 9 / y) ≥ 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2297_229746


namespace NUMINAMATH_CALUDE_sum_of_xyz_equals_695_l2297_229755

theorem sum_of_xyz_equals_695 (a b : ℝ) (x y z : ℕ+) :
  a^2 = 9/25 →
  b^2 = (3 + Real.sqrt 2)^2 / 14 →
  a < 0 →
  b > 0 →
  (a + b)^3 = (x.val : ℝ) * Real.sqrt y.val / z.val →
  x.val + y.val + z.val = 695 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_equals_695_l2297_229755


namespace NUMINAMATH_CALUDE_polynomial_root_sum_l2297_229727

theorem polynomial_root_sum (p q : ℝ) : 
  (Complex.I * Real.sqrt 2 + 2 : ℂ) ^ 3 + p * (Complex.I * Real.sqrt 2 + 2) + q = 0 → 
  p + q = 14 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_l2297_229727


namespace NUMINAMATH_CALUDE_project_work_time_l2297_229721

/-- Calculates the time spent working on a project given the project duration and nap information -/
def time_spent_working (project_days : ℕ) (num_naps : ℕ) (nap_duration : ℕ) : ℕ :=
  let total_hours := project_days * 24
  let total_nap_hours := num_naps * nap_duration
  total_hours - total_nap_hours

/-- Proves that given a 4-day project and 6 seven-hour naps, the time spent working is 54 hours -/
theorem project_work_time : time_spent_working 4 6 7 = 54 := by
  sorry

end NUMINAMATH_CALUDE_project_work_time_l2297_229721


namespace NUMINAMATH_CALUDE_sqrt_product_equation_l2297_229720

theorem sqrt_product_equation (x : ℝ) (h_pos : x > 0) 
  (h_eq : Real.sqrt (8 * x) * Real.sqrt (10 * x) * Real.sqrt (3 * x) * Real.sqrt (15 * x) = 15) : 
  x = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_product_equation_l2297_229720


namespace NUMINAMATH_CALUDE_candy_distribution_l2297_229762

theorem candy_distribution (total_candy : ℕ) (num_bags : ℕ) (candy_per_bag : ℕ) : 
  total_candy = 22 → num_bags = 2 → candy_per_bag = total_candy / num_bags → candy_per_bag = 11 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l2297_229762


namespace NUMINAMATH_CALUDE_variance_scaling_l2297_229767

/-- Given a set of data points, this function returns the variance of the data set. -/
noncomputable def variance (data : Finset ℝ) : ℝ := sorry

/-- Given a set of data points, this function multiplies each point by a scalar. -/
def scaleData (data : Finset ℝ) (scalar : ℝ) : Finset ℝ := sorry

theorem variance_scaling (data : Finset ℝ) (s : ℝ) :
  variance data = s^2 → variance (scaleData data 2) = 4 * s^2 := by sorry

end NUMINAMATH_CALUDE_variance_scaling_l2297_229767


namespace NUMINAMATH_CALUDE_law_of_sines_l2297_229719

/-- The Law of Sines for a triangle ABC -/
theorem law_of_sines (a b c : ℝ) (A B C : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (0 < A) ∧ (A < π) →
  (0 < B) ∧ (B < π) →
  (0 < C) ∧ (C < π) →
  (A + B + C = π) →
  (a / Real.sin A = b / Real.sin B) ∧
  (b / Real.sin B = c / Real.sin C) :=
sorry

end NUMINAMATH_CALUDE_law_of_sines_l2297_229719


namespace NUMINAMATH_CALUDE_sum_of_five_consecutive_even_integers_l2297_229744

theorem sum_of_five_consecutive_even_integers (a : ℤ) : 
  (a + (a + 4) = 150) → (a + (a + 2) + (a + 4) + (a + 6) + (a + 8) = 385) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_consecutive_even_integers_l2297_229744


namespace NUMINAMATH_CALUDE_trig_identity_proof_l2297_229701

theorem trig_identity_proof : 
  Real.cos (28 * π / 180) * Real.cos (17 * π / 180) - 
  Real.sin (28 * π / 180) * Real.cos (73 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l2297_229701


namespace NUMINAMATH_CALUDE_least_possible_difference_l2297_229793

theorem least_possible_difference (x y z : ℤ) : 
  x < y → y < z → y - x > 5 → Even x → Odd y → Odd z → 
  ∀ a : ℤ, (∃ x' y' z' : ℤ, x' < y' ∧ y' < z' ∧ y' - x' > 5 ∧ Even x' ∧ Odd y' ∧ Odd z' ∧ z' - x' = a) → 
  a ≥ 9 :=
by
  sorry

end NUMINAMATH_CALUDE_least_possible_difference_l2297_229793


namespace NUMINAMATH_CALUDE_first_train_speed_calculation_l2297_229731

/-- The speed of the first train in kmph -/
def first_train_speed : ℝ := 72

/-- The speed of the second train in kmph -/
def second_train_speed : ℝ := 36

/-- The length of the first train in meters -/
def first_train_length : ℝ := 200

/-- The length of the second train in meters -/
def second_train_length : ℝ := 300

/-- The time taken for the first train to cross the second train in seconds -/
def crossing_time : ℝ := 49.9960003199744

theorem first_train_speed_calculation :
  first_train_speed = 
    (first_train_length + second_train_length) / crossing_time * 3600 / 1000 + second_train_speed :=
by sorry

end NUMINAMATH_CALUDE_first_train_speed_calculation_l2297_229731


namespace NUMINAMATH_CALUDE_students_just_passed_l2297_229754

theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ) 
  (h_total : total = 300)
  (h_first : first_div_percent = 29/100)
  (h_second : second_div_percent = 54/100)
  (h_no_fail : first_div_percent + second_div_percent < 1) :
  total - (total * first_div_percent).floor - (total * second_div_percent).floor = 51 := by
  sorry

end NUMINAMATH_CALUDE_students_just_passed_l2297_229754


namespace NUMINAMATH_CALUDE_probability_equals_three_fourteenths_l2297_229792

-- Define the number of red and blue marbles
def red_marbles : ℕ := 15
def blue_marbles : ℕ := 10

-- Define the total number of marbles
def total_marbles : ℕ := red_marbles + blue_marbles

-- Define the number of marbles to be selected
def selected_marbles : ℕ := 4

-- Define the function to calculate combinations
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the probability of selecting two red and two blue marbles
def probability_two_red_two_blue : ℚ :=
  (6 * combination red_marbles 2 * combination blue_marbles 2) / (combination total_marbles selected_marbles)

-- Theorem statement
theorem probability_equals_three_fourteenths : 
  probability_two_red_two_blue = 3 / 14 := by sorry

end NUMINAMATH_CALUDE_probability_equals_three_fourteenths_l2297_229792


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_and_eccentricity_l2297_229703

/-- The focal length of a hyperbola with equation x^2 - y^2/3 = 1 is 4 and its eccentricity is 2 -/
theorem hyperbola_focal_length_and_eccentricity :
  let a : ℝ := 1
  let b : ℝ := Real.sqrt 3
  let c : ℝ := Real.sqrt (a^2 + b^2)
  let focal_length : ℝ := 2 * c
  let eccentricity : ℝ := c / a
  focal_length = 4 ∧ eccentricity = 2 := by
sorry


end NUMINAMATH_CALUDE_hyperbola_focal_length_and_eccentricity_l2297_229703


namespace NUMINAMATH_CALUDE_shekars_mathematics_marks_l2297_229794

/-- Represents the marks scored by Shekar in different subjects -/
structure Marks where
  mathematics : ℕ
  science : ℕ
  social_studies : ℕ
  english : ℕ
  biology : ℕ

/-- Calculates the average of marks -/
def average (m : Marks) : ℚ :=
  (m.mathematics + m.science + m.social_studies + m.english + m.biology) / 5

/-- Theorem stating that Shekar's marks in mathematics are 76 -/
theorem shekars_mathematics_marks :
  ∃ m : Marks,
    m.science = 65 ∧
    m.social_studies = 82 ∧
    m.english = 67 ∧
    m.biology = 75 ∧
    average m = 73 ∧
    m.mathematics = 76 := by
  sorry


end NUMINAMATH_CALUDE_shekars_mathematics_marks_l2297_229794


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2297_229734

/-- Proves that the speed of a boat in still water is 30 km/hr given specific conditions -/
theorem boat_speed_in_still_water 
  (current_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : current_speed = 7)
  (h2 : downstream_distance = 22.2)
  (h3 : downstream_time = 0.6) :
  ∃ (boat_speed : ℝ), 
    boat_speed = 30 ∧ 
    downstream_distance = (boat_speed + current_speed) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2297_229734


namespace NUMINAMATH_CALUDE_acute_triangle_in_right_triangle_l2297_229782

/-- A triangle represented by its vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Predicate to check if a triangle is acute-angled -/
def IsAcuteAngled (t : Triangle) : Prop := sorry

/-- Function to calculate the area of a triangle -/
def TriangleArea (t : Triangle) : ℝ := sorry

/-- Predicate to check if a triangle is right-angled -/
def IsRightAngled (t : Triangle) : Prop := sorry

/-- Predicate to check if one triangle contains another -/
def Contains (t1 t2 : Triangle) : Prop := sorry

theorem acute_triangle_in_right_triangle :
  ∀ (t : Triangle), IsAcuteAngled t → TriangleArea t = 1 →
  ∃ (r : Triangle), IsRightAngled r ∧ TriangleArea r = Real.sqrt 3 ∧ Contains r t := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_in_right_triangle_l2297_229782


namespace NUMINAMATH_CALUDE_jungkook_paper_count_l2297_229799

/-- Calculates the total number of pieces of colored paper given the number of bundles,
    pieces per bundle, and individual pieces. -/
def total_pieces (bundles : ℕ) (pieces_per_bundle : ℕ) (individual_pieces : ℕ) : ℕ :=
  bundles * pieces_per_bundle + individual_pieces

/-- Proves that given 3 bundles of 10 pieces each and 8 individual pieces,
    the total number of pieces is 38. -/
theorem jungkook_paper_count :
  total_pieces 3 10 8 = 38 := by
  sorry

end NUMINAMATH_CALUDE_jungkook_paper_count_l2297_229799


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2297_229772

theorem complex_fraction_simplification :
  (2 - Complex.I) / (1 + 2 * Complex.I) = -Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2297_229772


namespace NUMINAMATH_CALUDE_football_team_handedness_l2297_229704

theorem football_team_handedness (total_players : ℕ) (throwers : ℕ) (right_handed : ℕ)
  (h1 : total_players = 70)
  (h2 : throwers = 28)
  (h3 : right_handed = 56)
  (h4 : throwers ≤ right_handed) :
  (total_players - throwers - (right_handed - throwers)) / (total_players - throwers) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_football_team_handedness_l2297_229704


namespace NUMINAMATH_CALUDE_function_not_in_first_quadrant_l2297_229709

theorem function_not_in_first_quadrant (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : b < -1) :
  ∀ x : ℝ, x > 0 → a^x + b < 0 := by sorry

end NUMINAMATH_CALUDE_function_not_in_first_quadrant_l2297_229709


namespace NUMINAMATH_CALUDE_exponent_simplification_l2297_229728

theorem exponent_simplification (x : ℝ) : (x^5 * x^2) * x^4 = x^11 := by
  sorry

end NUMINAMATH_CALUDE_exponent_simplification_l2297_229728


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l2297_229702

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 12 + y^2 / 4 = 1

-- Define the line l
def line_l (x : ℝ) : Prop := x = -2 * Real.sqrt 2

-- Define a point on the ellipse
def point_on_ellipse (x y : ℝ) : Prop := ellipse x y

-- Define a point on line l
def point_on_line_l (x y : ℝ) : Prop := line_l x

-- Define the perpendicular line l' through P
def line_l_prime (x y x_p y_p : ℝ) : Prop :=
  y - y_p = -(3 * y_p) / (2 * Real.sqrt 2) * (x - x_p)

theorem ellipse_intersection_theorem :
  ∀ (x_p y_p : ℝ),
    point_on_line_l x_p y_p →
    ∃ (x_m y_m x_n y_n : ℝ),
      point_on_ellipse x_m y_m ∧
      point_on_ellipse x_n y_n ∧
      (x_p - x_m)^2 + (y_p - y_m)^2 = (x_p - x_n)^2 + (y_p - y_n)^2 →
      line_l_prime (-4 * Real.sqrt 2 / 3) 0 x_p y_p :=
by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l2297_229702


namespace NUMINAMATH_CALUDE_pizzeria_sales_l2297_229760

theorem pizzeria_sales (small_price large_price total_revenue small_count : ℕ) 
  (h1 : small_price = 2)
  (h2 : large_price = 8)
  (h3 : total_revenue = 40)
  (h4 : small_count = 8) :
  ∃ (large_count : ℕ), 
    small_price * small_count + large_price * large_count = total_revenue ∧ 
    large_count = 3 := by
  sorry

end NUMINAMATH_CALUDE_pizzeria_sales_l2297_229760


namespace NUMINAMATH_CALUDE_charlies_subtraction_l2297_229722

theorem charlies_subtraction (charlie_add : 41^2 = 40^2 + 81) : 39^2 = 40^2 - 79 := by
  sorry

end NUMINAMATH_CALUDE_charlies_subtraction_l2297_229722


namespace NUMINAMATH_CALUDE_painting_gift_options_l2297_229725

theorem painting_gift_options (n : ℕ) (h : n = 10) : 
  (Finset.powerset (Finset.range n)).card - 1 = 2^n - 1 := by
  sorry

end NUMINAMATH_CALUDE_painting_gift_options_l2297_229725


namespace NUMINAMATH_CALUDE_prob_at_least_two_different_fruits_l2297_229783

def num_fruit_types : ℕ := 4
def num_meals : ℕ := 4

def prob_same_fruit : ℚ := (1 / num_fruit_types) ^ num_meals * num_fruit_types

theorem prob_at_least_two_different_fruits :
  1 - prob_same_fruit = 63 / 64 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_two_different_fruits_l2297_229783


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2297_229740

theorem perfect_square_condition (n : ℕ+) :
  (∃ m : ℕ, n^4 + 2*n^3 + 5*n^2 + 12*n + 5 = m^2) ↔ (n = 1 ∨ n = 2) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2297_229740


namespace NUMINAMATH_CALUDE_line_through_point_perpendicular_to_line_l2297_229750

/-- Given a point A and two lines l₁ and l₂, this theorem states that
    l₂ passes through A and is perpendicular to l₁. -/
theorem line_through_point_perpendicular_to_line
  (A : ℝ × ℝ)  -- Point A
  (l₁ : ℝ → ℝ → Prop)  -- Line l₁
  (l₂ : ℝ → ℝ → Prop)  -- Line l₂
  (h₁ : l₁ = fun x y ↦ 2 * x + 3 * y + 4 = 0)  -- Equation of l₁
  (h₂ : l₂ = fun x y ↦ 3 * x - 2 * y + 7 = 0)  -- Equation of l₂
  (h₃ : A = (-1, 2))  -- Coordinates of point A
  : (l₂ (A.1) (A.2)) ∧  -- l₂ passes through A
    (∀ (x y : ℝ), l₁ x y → l₂ x y → (2 * 3 + 3 * (-2) = 0)) :=  -- l₁ ⊥ l₂
by sorry

end NUMINAMATH_CALUDE_line_through_point_perpendicular_to_line_l2297_229750


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_integers_l2297_229773

theorem sum_of_three_consecutive_integers (a b c : ℕ) : 
  (a + 1 = b ∧ b + 1 = c) → c = 7 → a + b + c = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_integers_l2297_229773


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_third_l2297_229749

theorem opposite_of_negative_one_third :
  -((-1 : ℚ) / 3) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_third_l2297_229749


namespace NUMINAMATH_CALUDE_vanessa_album_pictures_l2297_229710

/-- The number of albums created by Vanessa -/
def num_albums : ℕ := 10

/-- The number of pictures from the phone in each album -/
def phone_pics_per_album : ℕ := 8

/-- The number of pictures from the camera in each album -/
def camera_pics_per_album : ℕ := 4

/-- The total number of pictures in each album -/
def pics_per_album : ℕ := phone_pics_per_album + camera_pics_per_album

theorem vanessa_album_pictures :
  pics_per_album = 12 :=
sorry

end NUMINAMATH_CALUDE_vanessa_album_pictures_l2297_229710


namespace NUMINAMATH_CALUDE_equation_solutions_l2297_229787

/-- A parabola that intersects the x-axis at (-1, 0) and (3, 0) -/
structure Parabola where
  m : ℝ
  n : ℝ
  intersect_neg_one : (-1 - m)^2 + n = 0
  intersect_three : (3 - m)^2 + n = 0

/-- The equation to solve -/
def equation (p : Parabola) (x : ℝ) : Prop :=
  (x - 1)^2 + p.m^2 = 2 * p.m * (x - 1) - p.n

/-- The theorem stating the solutions to the equation -/
theorem equation_solutions (p : Parabola) :
  ∃ (x₁ x₂ : ℝ), x₁ = 0 ∧ x₂ = 4 ∧ equation p x₁ ∧ equation p x₂ :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2297_229787


namespace NUMINAMATH_CALUDE_line_symmetry_l2297_229779

/-- The equation of a line in the Cartesian plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ∀ x y : ℝ, a * x + b * y + c = 0

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry of a point about another point -/
def symmetric_point (p q : Point) : Point :=
  ⟨2 * q.x - p.x, 2 * q.y - p.y⟩

/-- Two lines are symmetric about a point if for any point on one line,
    its symmetric point about the given point lies on the other line -/
def symmetric_lines (l₁ l₂ : Line) (p : Point) : Prop :=
  ∀ x y : ℝ, l₁.a * x + l₁.b * y + l₁.c = 0 →
    let sym := symmetric_point ⟨x, y⟩ p
    l₂.a * sym.x + l₂.b * sym.y + l₂.c = 0

theorem line_symmetry :
  let l₁ : Line := ⟨3, -1, 2, sorry⟩
  let l₂ : Line := ⟨3, -1, -6, sorry⟩
  let p : Point := ⟨1, 1⟩
  symmetric_lines l₁ l₂ p := by sorry

end NUMINAMATH_CALUDE_line_symmetry_l2297_229779


namespace NUMINAMATH_CALUDE_derivative_x_ln_x_l2297_229741

noncomputable section

open Real

theorem derivative_x_ln_x (x : ℝ) (h : x > 0) :
  deriv (λ x => x * log x) x = 1 + log x :=
sorry

end NUMINAMATH_CALUDE_derivative_x_ln_x_l2297_229741


namespace NUMINAMATH_CALUDE_linear_function_properties_l2297_229715

def f (x : ℝ) : ℝ := x + 2

theorem linear_function_properties :
  (f 1 = 3) ∧
  (f (-2) = 0) ∧
  (∀ x y, f x = y → x ≥ 0 ∧ y ≤ 0 → x = 0 ∧ y = 2) ∧
  (∃ x, x > 2 ∧ f x ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_properties_l2297_229715


namespace NUMINAMATH_CALUDE_tablecloth_black_percentage_l2297_229759

/-- Represents a square tablecloth -/
structure Tablecloth :=
  (size : ℕ)
  (black_outer_ratio : ℚ)

/-- Calculates the percentage of black area on the tablecloth -/
def black_percentage (t : Tablecloth) : ℚ :=
  let total_squares := t.size * t.size
  let outer_squares := 4 * (t.size - 1)
  let black_squares := (outer_squares : ℚ) * t.black_outer_ratio
  (black_squares / total_squares) * 100

/-- Theorem stating that a 5x5 tablecloth with half of each outer square black is 32% black -/
theorem tablecloth_black_percentage :
  let t : Tablecloth := ⟨5, 1/2⟩
  black_percentage t = 32 := by
  sorry

end NUMINAMATH_CALUDE_tablecloth_black_percentage_l2297_229759


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2297_229737

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (9 - 2 * x) = 5 → x = -8 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2297_229737


namespace NUMINAMATH_CALUDE_carrot_theorem_l2297_229756

def carrot_problem (initial_carrots additional_carrots final_total : ℕ) : Prop :=
  initial_carrots + additional_carrots - final_total = 4

theorem carrot_theorem : carrot_problem 19 46 61 := by
  sorry

end NUMINAMATH_CALUDE_carrot_theorem_l2297_229756


namespace NUMINAMATH_CALUDE_cost_of_500_pencils_l2297_229796

/-- The cost of 500 pencils in dollars, given that 1 pencil costs 3 cents -/
theorem cost_of_500_pencils :
  let cost_of_one_pencil_cents : ℕ := 3
  let number_of_pencils : ℕ := 500
  let cents_per_dollar : ℕ := 100
  (cost_of_one_pencil_cents * number_of_pencils) / cents_per_dollar = 15 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_500_pencils_l2297_229796


namespace NUMINAMATH_CALUDE_no_common_points_necessary_not_sufficient_for_parallel_l2297_229778

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields to define a line in 3D space
  -- This is a placeholder and should be properly defined

/-- Two lines have no common points -/
def no_common_points (l1 l2 : Line3D) : Prop :=
  -- Define the condition for two lines having no common points
  sorry

/-- Two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop :=
  -- Define the condition for two lines being parallel
  sorry

/-- Skew lines: lines that are not parallel and do not intersect -/
def skew (l1 l2 : Line3D) : Prop :=
  no_common_points l1 l2 ∧ ¬parallel l1 l2

theorem no_common_points_necessary_not_sufficient_for_parallel :
  (∀ l1 l2 : Line3D, parallel l1 l2 → no_common_points l1 l2) ∧
  (∃ l1 l2 : Line3D, no_common_points l1 l2 ∧ ¬parallel l1 l2) :=
by
  sorry

#check no_common_points_necessary_not_sufficient_for_parallel

end NUMINAMATH_CALUDE_no_common_points_necessary_not_sufficient_for_parallel_l2297_229778
