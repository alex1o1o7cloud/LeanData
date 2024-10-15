import Mathlib

namespace NUMINAMATH_CALUDE_lemonade_ratio_l2175_217547

/-- Given that 36 lemons make 48 gallons of lemonade, this theorem proves that 4.5 lemons are needed for 6 gallons of lemonade. -/
theorem lemonade_ratio (lemons : ℝ) (gallons : ℝ) 
  (h1 : lemons / gallons = 36 / 48) 
  (h2 : gallons = 6) : 
  lemons = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_ratio_l2175_217547


namespace NUMINAMATH_CALUDE_fishing_trip_cost_l2175_217534

/-- The fishing trip cost problem -/
theorem fishing_trip_cost (alice_paid bob_paid chris_paid : ℝ) 
  (h1 : alice_paid = 135)
  (h2 : bob_paid = 165)
  (h3 : chris_paid = 225)
  (x y : ℝ) 
  (h4 : x = (alice_paid + bob_paid + chris_paid) / 3 - alice_paid)
  (h5 : y = (alice_paid + bob_paid + chris_paid) / 3 - bob_paid) :
  x - y = 30 := by
sorry

end NUMINAMATH_CALUDE_fishing_trip_cost_l2175_217534


namespace NUMINAMATH_CALUDE_parallel_line_slope_l2175_217549

theorem parallel_line_slope (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  let given_line := {(x, y) : ℝ × ℝ | a * x - b * y = c}
  let slope := a / b
  ∀ m : ℝ, (∃ k : ℝ, ∀ x y : ℝ, y = m * x + k ↔ (x, y) ∈ given_line) → m = slope :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l2175_217549


namespace NUMINAMATH_CALUDE_correct_prices_l2175_217570

/-- Represents the purchase and sale of golden passion fruit -/
structure GoldenPassionFruit where
  first_batch_cost : ℝ
  second_batch_cost : ℝ
  weight_ratio : ℝ
  price_difference : ℝ
  profit_margin : ℝ

/-- Calculates the unit prices and minimum selling price for golden passion fruit -/
def calculate_prices (gpf : GoldenPassionFruit) : 
  (ℝ × ℝ × ℝ) :=
  let first_batch_price := 20
  let second_batch_price := first_batch_price - gpf.price_difference
  let min_selling_price := 25
  (first_batch_price, second_batch_price, min_selling_price)

/-- Theorem stating the correctness of the calculated prices -/
theorem correct_prices (gpf : GoldenPassionFruit) 
  (h1 : gpf.first_batch_cost = 3600)
  (h2 : gpf.second_batch_cost = 5400)
  (h3 : gpf.weight_ratio = 2)
  (h4 : gpf.price_difference = 5)
  (h5 : gpf.profit_margin = 0.5) :
  let (first_price, second_price, min_price) := calculate_prices gpf
  first_price = 20 ∧ 
  second_price = 15 ∧ 
  min_price ≥ 25 ∧
  min_price * (gpf.first_batch_cost / first_price + gpf.second_batch_cost / second_price) ≥ 
    (gpf.first_batch_cost + gpf.second_batch_cost) * (1 + gpf.profit_margin) :=
by sorry


end NUMINAMATH_CALUDE_correct_prices_l2175_217570


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l2175_217511

theorem ferris_wheel_capacity (total_people : ℕ) (total_seats : ℕ) 
  (h1 : total_people = 16) (h2 : total_seats = 4) : 
  total_people / total_seats = 4 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l2175_217511


namespace NUMINAMATH_CALUDE_negation_equivalence_l2175_217590

theorem negation_equivalence :
  (¬ ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 - 1 ≤ 0) ↔ (∀ (x y : ℝ), x^2 + y^2 - 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2175_217590


namespace NUMINAMATH_CALUDE_perfect_square_divisibility_l2175_217539

theorem perfect_square_divisibility (x y : ℕ+) (h : (2 * x * y) ∣ (x^2 + y^2 - x)) : 
  ∃ (n : ℕ+), x = n^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_divisibility_l2175_217539


namespace NUMINAMATH_CALUDE_set_relations_l2175_217566

variable {α : Type*}
variable (I A B : Set α)

theorem set_relations (h : A ∪ B = I) :
  (Aᶜ ∩ Bᶜ = ∅) ∧ (B ⊇ Aᶜ) := by
  sorry

end NUMINAMATH_CALUDE_set_relations_l2175_217566


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_a_solution_set_eq_interval_l2175_217574

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |2*x + 1|

-- Theorem for part (I)
theorem solution_set_f (x : ℝ) : 
  f x ≥ 4*x + 3 ↔ x ∈ Set.Iic (-3/7) := by sorry

-- Theorem for part (II)
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 2 * f x ≥ 3*a^2 - a - 1) → 
  a ∈ Set.Icc (-1) (4/3) := by sorry

-- Define the set of solutions for part (I)
def solution_set : Set ℝ := {x : ℝ | f x ≥ 4*x + 3}

-- Theorem stating that the solution set is equal to (-∞, -3/7]
theorem solution_set_eq_interval : 
  solution_set = Set.Iic (-3/7) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_a_solution_set_eq_interval_l2175_217574


namespace NUMINAMATH_CALUDE_solve_equation_l2175_217526

theorem solve_equation (x : ℝ) : 3 * x = (62 - x) + 26 → x = 22 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2175_217526


namespace NUMINAMATH_CALUDE_people_who_left_train_l2175_217518

/-- The number of people who left a train given initial, boarding, and final passenger counts. -/
theorem people_who_left_train (initial : ℕ) (boarded : ℕ) (final : ℕ) : 
  initial = 82 → boarded = 17 → final = 73 → initial + boarded - final = 26 := by
  sorry

end NUMINAMATH_CALUDE_people_who_left_train_l2175_217518


namespace NUMINAMATH_CALUDE_power_difference_divisibility_l2175_217531

theorem power_difference_divisibility (N : ℕ+) :
  ∃ (r s : ℕ), r ≠ s ∧ ∀ (A : ℤ), (N : ℤ) ∣ (A^r - A^s) := by
  sorry

end NUMINAMATH_CALUDE_power_difference_divisibility_l2175_217531


namespace NUMINAMATH_CALUDE_fresh_grapes_water_content_l2175_217516

/-- Percentage of water in dried grapes -/
def dried_water_percentage : ℝ := 20

/-- Weight of fresh grapes in kg -/
def fresh_weight : ℝ := 40

/-- Weight of dried grapes in kg -/
def dried_weight : ℝ := 10

/-- Percentage of water in fresh grapes -/
def fresh_water_percentage : ℝ := 80

theorem fresh_grapes_water_content :
  (1 - fresh_water_percentage / 100) * fresh_weight = (1 - dried_water_percentage / 100) * dried_weight :=
sorry

end NUMINAMATH_CALUDE_fresh_grapes_water_content_l2175_217516


namespace NUMINAMATH_CALUDE_probability_same_result_is_seven_twentyfourths_l2175_217527

/-- Represents a 12-sided die with specific colored sides -/
structure TwelveSidedDie :=
  (purple : Nat)
  (green : Nat)
  (orange : Nat)
  (glittery : Nat)
  (total_sides : Nat)
  (is_valid : purple + green + orange + glittery = total_sides)

/-- Calculates the probability of two dice showing the same result -/
def probability_same_result (d : TwelveSidedDie) : Rat :=
  let p_purple := (d.purple : Rat) / d.total_sides
  let p_green := (d.green : Rat) / d.total_sides
  let p_orange := (d.orange : Rat) / d.total_sides
  let p_glittery := (d.glittery : Rat) / d.total_sides
  p_purple * p_purple + p_green * p_green + p_orange * p_orange + p_glittery * p_glittery

/-- Theorem: The probability of two 12-sided dice with specific colored sides showing the same result is 7/24 -/
theorem probability_same_result_is_seven_twentyfourths :
  let d : TwelveSidedDie := {
    purple := 3,
    green := 4,
    orange := 4,
    glittery := 1,
    total_sides := 12,
    is_valid := by simp
  }
  probability_same_result d = 7 / 24 := by
  sorry


end NUMINAMATH_CALUDE_probability_same_result_is_seven_twentyfourths_l2175_217527


namespace NUMINAMATH_CALUDE_year_spans_weeks_l2175_217509

/-- Represents the number of days in a week -/
def daysInWeek : ℕ := 7

/-- Represents the number of days in a common year -/
def daysInCommonYear : ℕ := 365

/-- Represents the number of days in a leap year -/
def daysInLeapYear : ℕ := 366

/-- Represents the minimum number of days a week must have in a year to be counted -/
def minDaysInWeekForYear : ℕ := 6

/-- Definition of how many weeks a year can span -/
def weeksInYear : Set ℕ := {53, 54}

/-- Theorem stating the number of weeks a year can span -/
theorem year_spans_weeks : 
  ∀ (year : ℕ), 
    (year = daysInCommonYear ∨ year = daysInLeapYear) → 
    ∃ (weeks : ℕ), weeks ∈ weeksInYear ∧ 
      (weeks - 1) * daysInWeek + minDaysInWeekForYear ≤ year ∧
      year < (weeks + 1) * daysInWeek :=
sorry

end NUMINAMATH_CALUDE_year_spans_weeks_l2175_217509


namespace NUMINAMATH_CALUDE_dunkers_lineup_count_l2175_217528

/-- The number of players in the team -/
def team_size : ℕ := 15

/-- The number of players who refuse to play together -/
def excluded_players : ℕ := 3

/-- The size of a starting lineup -/
def lineup_size : ℕ := 5

/-- The number of ways to select a lineup from the team, excluding the combinations
    where the excluded players play together -/
def valid_lineups : ℕ := 2277

theorem dunkers_lineup_count :
  (excluded_players * (Nat.choose (team_size - excluded_players) (lineup_size - 1))) +
  (Nat.choose (team_size - excluded_players) lineup_size) = valid_lineups :=
sorry

end NUMINAMATH_CALUDE_dunkers_lineup_count_l2175_217528


namespace NUMINAMATH_CALUDE_min_value_theorem_l2175_217551

theorem min_value_theorem (a : ℝ) (ha : a > 0) :
  a + (a + 4) / a ≥ 5 ∧ ∃ a₀ > 0, a₀ + (a₀ + 4) / a₀ = 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2175_217551


namespace NUMINAMATH_CALUDE_jerry_book_pages_l2175_217589

/-- Calculates the total number of pages in a book given the number of pages read on Saturday, Sunday, and the number of pages remaining. -/
def total_pages (pages_saturday : ℕ) (pages_sunday : ℕ) (pages_remaining : ℕ) : ℕ :=
  pages_saturday + pages_sunday + pages_remaining

/-- Theorem stating that the total number of pages in Jerry's book is 93. -/
theorem jerry_book_pages : total_pages 30 20 43 = 93 := by
  sorry

end NUMINAMATH_CALUDE_jerry_book_pages_l2175_217589


namespace NUMINAMATH_CALUDE_intersection_of_lines_l2175_217561

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (-1/4, -1/4)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := y = 5 * x + 1

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 2 * y - 3 = -6 * x

theorem intersection_of_lines :
  let (x, y) := intersection_point
  (line1 x y ∧ line2 x y) ∧
  ∀ x' y', (line1 x' y' ∧ line2 x' y') → (x' = x ∧ y' = y) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l2175_217561


namespace NUMINAMATH_CALUDE_min_correct_answers_to_pass_l2175_217537

/-- Represents a test with given parameters -/
structure Test where
  total_questions : ℕ
  points_correct : ℕ
  points_wrong : ℕ
  passing_score : ℕ

/-- Calculates the score based on the number of correct answers -/
def calculate_score (test : Test) (correct_answers : ℕ) : ℤ :=
  (test.points_correct * correct_answers : ℤ) - 
  (test.points_wrong * (test.total_questions - correct_answers) : ℤ)

/-- Theorem stating the minimum number of correct answers needed to pass the test -/
theorem min_correct_answers_to_pass (test : Test) 
  (h1 : test.total_questions = 20)
  (h2 : test.points_correct = 5)
  (h3 : test.points_wrong = 3)
  (h4 : test.passing_score = 60) :
  ∀ n : ℕ, n ≥ 15 ↔ calculate_score test n ≥ test.passing_score := by
  sorry

#check min_correct_answers_to_pass

end NUMINAMATH_CALUDE_min_correct_answers_to_pass_l2175_217537


namespace NUMINAMATH_CALUDE_sqrt_difference_sum_abs_l2175_217599

theorem sqrt_difference_sum_abs : 1 + (Real.sqrt 2 - Real.sqrt 3) + |Real.sqrt 2 - Real.sqrt 3| = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_sum_abs_l2175_217599


namespace NUMINAMATH_CALUDE_sum_of_powers_inequality_l2175_217515

theorem sum_of_powers_inequality (x : ℝ) (hx : x > 0) :
  1 + x + x^2 + x^3 + x^4 ≥ 5 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_inequality_l2175_217515


namespace NUMINAMATH_CALUDE_second_smallest_packs_l2175_217562

/-- The number of hot dogs in each pack -/
def hot_dogs_per_pack : ℕ := 10

/-- The number of buns in each pack -/
def buns_per_pack : ℕ := 8

/-- The number of hot dogs left over -/
def leftover_hot_dogs : ℕ := 4

/-- A function that checks if a given number of packs satisfies the condition -/
def satisfies_condition (n : ℕ) : Prop :=
  (hot_dogs_per_pack * n) % buns_per_pack = leftover_hot_dogs

/-- The theorem stating that 6 is the second smallest number of packs satisfying the condition -/
theorem second_smallest_packs : 
  ∃ (m : ℕ), m < 6 ∧ satisfies_condition m ∧ 
  (∀ (k : ℕ), k < m → ¬satisfies_condition k) ∧
  (∀ (k : ℕ), m < k → k < 6 → ¬satisfies_condition k) ∧
  satisfies_condition 6 := by
  sorry

end NUMINAMATH_CALUDE_second_smallest_packs_l2175_217562


namespace NUMINAMATH_CALUDE_quarters_count_l2175_217536

/-- Proves that given 21 coins consisting of nickels and quarters with a total value of $3.65, the number of quarters is 13. -/
theorem quarters_count (total_coins : ℕ) (total_value : ℚ) (nickels : ℕ) (quarters : ℕ) : 
  total_coins = 21 →
  total_value = 365/100 →
  total_coins = nickels + quarters →
  total_value = (5 * nickels + 25 * quarters) / 100 →
  quarters = 13 := by
  sorry

end NUMINAMATH_CALUDE_quarters_count_l2175_217536


namespace NUMINAMATH_CALUDE_remainder_mod_88_l2175_217514

theorem remainder_mod_88 : (1 - 90) ^ 10 ≡ 1 [MOD 88] := by sorry

end NUMINAMATH_CALUDE_remainder_mod_88_l2175_217514


namespace NUMINAMATH_CALUDE_jason_seashells_l2175_217510

theorem jason_seashells (initial_seashells : ℕ) (given_seashells : ℕ) 
  (h1 : initial_seashells = 49) (h2 : given_seashells = 13) :
  initial_seashells - given_seashells = 36 := by
  sorry

#check jason_seashells

end NUMINAMATH_CALUDE_jason_seashells_l2175_217510


namespace NUMINAMATH_CALUDE_book_distribution_l2175_217542

theorem book_distribution (m : ℕ) (x y : ℕ) : 
  (y - 14 = m * x) →  -- If each child gets m books, 14 books are left
  (y + 3 = 9 * x) →   -- If each child gets 9 books, the last child gets 6 books
  (x = 17 ∧ y = 150) := by
  sorry

end NUMINAMATH_CALUDE_book_distribution_l2175_217542


namespace NUMINAMATH_CALUDE_geometric_sequence_proof_minimum_years_proof_l2175_217512

def initial_capital : ℝ := 50
def growth_rate : ℝ := 0.5
def payment (t : ℝ) : ℝ := t

def remaining_capital (n : ℕ) (t : ℝ) : ℝ :=
  if n = 0 then initial_capital
  else (1 + growth_rate) * remaining_capital (n - 1) t - payment t

theorem geometric_sequence_proof (t : ℝ) (h : 0 < t ∧ t < 2500) :
  ∀ n : ℕ, (remaining_capital (n + 1) t - 2 * t) / (remaining_capital n t - 2 * t) = 3 / 2 :=
sorry

theorem minimum_years_proof :
  let t := 1500
  (∃ m : ℕ, remaining_capital m t > 21000) ∧
  (∀ k : ℕ, k < 6 → remaining_capital k t ≤ 21000) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_proof_minimum_years_proof_l2175_217512


namespace NUMINAMATH_CALUDE_guitar_strings_problem_l2175_217554

theorem guitar_strings_problem (total_strings : ℕ) 
  (num_basses : ℕ) (strings_per_bass : ℕ) 
  (strings_per_normal_guitar : ℕ) (strings_difference : ℕ) :
  let num_normal_guitars := 2 * num_basses
  let strings_for_basses := num_basses * strings_per_bass
  let strings_for_normal_guitars := num_normal_guitars * strings_per_normal_guitar
  let remaining_strings := total_strings - strings_for_basses - strings_for_normal_guitars
  let strings_per_fewer_guitar := strings_per_normal_guitar - strings_difference
  total_strings = 72 ∧ 
  num_basses = 3 ∧ 
  strings_per_bass = 4 ∧ 
  strings_per_normal_guitar = 6 ∧ 
  strings_difference = 3 →
  strings_per_fewer_guitar = 3 :=
by sorry

end NUMINAMATH_CALUDE_guitar_strings_problem_l2175_217554


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l2175_217503

theorem polar_to_rectangular_conversion :
  let r : ℝ := 6
  let θ : ℝ := 5 * Real.pi / 3
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  x = 3 ∧ y = -3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l2175_217503


namespace NUMINAMATH_CALUDE_f_decreasing_after_one_l2175_217545

def f (x : ℝ) : ℝ := -(x - 1)^2 + 2

theorem f_decreasing_after_one :
  ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f x₂ < f x₁ := by
  sorry

end NUMINAMATH_CALUDE_f_decreasing_after_one_l2175_217545


namespace NUMINAMATH_CALUDE_original_ratio_proof_l2175_217583

/-- Represents the number of students in each category -/
structure StudentCount where
  boarders : ℕ
  dayStudents : ℕ

/-- The ratio of boarders to day students -/
def ratio (sc : StudentCount) : ℚ :=
  sc.boarders / sc.dayStudents

theorem original_ratio_proof (initial : StudentCount) (final : StudentCount) :
  initial.boarders = 330 →
  final.boarders = initial.boarders + 66 →
  ratio final = 1 / 2 →
  ratio initial = 5 / 12 := by
  sorry

#check original_ratio_proof

end NUMINAMATH_CALUDE_original_ratio_proof_l2175_217583


namespace NUMINAMATH_CALUDE_expression_evaluation_l2175_217555

theorem expression_evaluation (x y : ℚ) (hx : x = 3) (hy : y = 4) :
  (x^2 / y^3) / (y / x) = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2175_217555


namespace NUMINAMATH_CALUDE_age_problem_l2175_217541

/-- The age problem -/
theorem age_problem (a b : ℝ) : 
  b = 3 * a - 20 →  -- Bob's age is 20 years less than three times Alice's age
  a + b = 70 →      -- The sum of their ages is 70
  b = 47.5 :=       -- Bob's age is 47.5
by sorry

end NUMINAMATH_CALUDE_age_problem_l2175_217541


namespace NUMINAMATH_CALUDE_power_less_than_threshold_l2175_217557

theorem power_less_than_threshold : ∃ (n1 n2 n3 : ℕ+),
  (0.99 : ℝ) ^ (n1 : ℝ) < 0.000001 ∧
  (0.999 : ℝ) ^ (n2 : ℝ) < 0.000001 ∧
  (0.999999 : ℝ) ^ (n3 : ℝ) < 0.000001 := by
  sorry

end NUMINAMATH_CALUDE_power_less_than_threshold_l2175_217557


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l2175_217593

theorem right_triangle_third_side (x y : ℝ) :
  (x > 0 ∧ y > 0) →
  (|x^2 - 4| + Real.sqrt (y^2 - 5*y + 6) = 0) →
  ∃ z : ℝ, (z = 2 * Real.sqrt 2 ∨ z = Real.sqrt 13 ∨ z = Real.sqrt 5) ∧
           (x^2 + y^2 = z^2 ∨ x^2 + z^2 = y^2 ∨ y^2 + z^2 = x^2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l2175_217593


namespace NUMINAMATH_CALUDE_work_hours_per_day_l2175_217565

theorem work_hours_per_day (days : ℕ) (total_hours : ℕ) (h1 : days = 5) (h2 : total_hours = 40) :
  total_hours / days = 8 :=
by sorry

end NUMINAMATH_CALUDE_work_hours_per_day_l2175_217565


namespace NUMINAMATH_CALUDE_vectors_not_collinear_l2175_217586

/-- Given two vectors a and b in ℝ³, we define c₁ and c₂ as linear combinations of a and b.
    This theorem states that c₁ and c₂ are not collinear. -/
theorem vectors_not_collinear :
  let a : Fin 3 → ℝ := ![1, 4, -2]
  let b : Fin 3 → ℝ := ![1, 1, -1]
  let c₁ : Fin 3 → ℝ := a + b
  let c₂ : Fin 3 → ℝ := 4 • a + 2 • b
  ¬ (∃ (k : ℝ), c₁ = k • c₂) :=
by sorry

end NUMINAMATH_CALUDE_vectors_not_collinear_l2175_217586


namespace NUMINAMATH_CALUDE_average_income_of_M_and_N_l2175_217598

/-- Given the monthly incomes of three individuals M, N, and O, prove that the average income of M and N is 5050. -/
theorem average_income_of_M_and_N (M N O : ℕ) : 
  (N + O) / 2 = 6250 →
  (M + O) / 2 = 5200 →
  M = 4000 →
  (M + N) / 2 = 5050 := by
sorry

end NUMINAMATH_CALUDE_average_income_of_M_and_N_l2175_217598


namespace NUMINAMATH_CALUDE_fifth_day_distance_l2175_217575

/-- Represents the daily walking distance of a man -/
def walkingSequence (firstDay : ℕ) (dailyIncrease : ℕ) : ℕ → ℕ :=
  fun n => firstDay + (n - 1) * dailyIncrease

theorem fifth_day_distance
  (firstDay : ℕ)
  (dailyIncrease : ℕ)
  (h1 : firstDay = 100)
  (h2 : (Finset.range 9).sum (walkingSequence firstDay dailyIncrease) = 1260) :
  walkingSequence firstDay dailyIncrease 5 = 140 := by
  sorry

#check fifth_day_distance

end NUMINAMATH_CALUDE_fifth_day_distance_l2175_217575


namespace NUMINAMATH_CALUDE_seven_eighths_of_64_l2175_217567

theorem seven_eighths_of_64 : (7 / 8 : ℚ) * 64 = 56 := by sorry

end NUMINAMATH_CALUDE_seven_eighths_of_64_l2175_217567


namespace NUMINAMATH_CALUDE_union_subset_iff_m_nonpositive_no_m_exists_for_equality_l2175_217546

-- Define the sets P and S
def P : Set ℝ := {x | x^2 - 8*x - 20 ≤ 0}
def S (m : ℝ) : Set ℝ := {x | |x - 1| ≤ m}

-- Question 1
theorem union_subset_iff_m_nonpositive (m : ℝ) :
  (P ∪ S m) ⊆ P ↔ m ≤ 0 :=
sorry

-- Question 2
theorem no_m_exists_for_equality :
  ¬∃ m : ℝ, P = S m :=
sorry

end NUMINAMATH_CALUDE_union_subset_iff_m_nonpositive_no_m_exists_for_equality_l2175_217546


namespace NUMINAMATH_CALUDE_max_n_for_factorable_quadratic_l2175_217573

/-- 
Given a quadratic expression 6x^2 + nx + 108 that can be factored as the product 
of two linear factors with integer coefficients, the maximum possible value of n is 649.
-/
theorem max_n_for_factorable_quadratic : 
  ∀ n : ℤ, 
  (∃ A B : ℤ, 6 * A * B = 108 ∧ 6 * B + A = n) → 
  n ≤ 649 :=
by sorry

end NUMINAMATH_CALUDE_max_n_for_factorable_quadratic_l2175_217573


namespace NUMINAMATH_CALUDE_student_scores_average_l2175_217538

theorem student_scores_average (math physics chem : ℕ) : 
  math + physics = 30 → 
  chem = physics + 20 → 
  (math + chem) / 2 = 25 := by
sorry

end NUMINAMATH_CALUDE_student_scores_average_l2175_217538


namespace NUMINAMATH_CALUDE_complex_quadrant_l2175_217577

theorem complex_quadrant (a b : ℝ) (z : ℂ) :
  z = a + b * Complex.I →
  (a / (1 - Complex.I) + b / (1 - 2 * Complex.I) = 5 / (3 + Complex.I)) →
  0 < a ∧ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_quadrant_l2175_217577


namespace NUMINAMATH_CALUDE_max_min_values_of_f_l2175_217560

def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

theorem max_min_values_of_f :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-2 : ℝ) 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 3, f x = max) ∧
    (∀ x ∈ Set.Icc (-2 : ℝ) 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 3, f x = min) ∧
    max = 4 ∧ min = -5 :=
by sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_l2175_217560


namespace NUMINAMATH_CALUDE_two_mixers_one_tv_cost_is_7000_l2175_217500

/-- The cost of a mixer in Rupees -/
def mixer_cost : ℕ := sorry

/-- The cost of a TV in Rupees -/
def tv_cost : ℕ := 4200

/-- The total cost of two mixers and one TV in Rupees -/
def two_mixers_one_tv_cost : ℕ := 2 * mixer_cost + tv_cost

theorem two_mixers_one_tv_cost_is_7000 :
  (two_mixers_one_tv_cost = 7000) ∧ (2 * tv_cost + mixer_cost = 9800) :=
sorry

end NUMINAMATH_CALUDE_two_mixers_one_tv_cost_is_7000_l2175_217500


namespace NUMINAMATH_CALUDE_choose_and_assign_officers_l2175_217517

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The factorial of a natural number -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to choose 3 people from 5 and assign them to 3 distinct roles -/
def waysToChooseAndAssign : ℕ := choose 5 3 * factorial 3

theorem choose_and_assign_officers :
  waysToChooseAndAssign = 60 := by
  sorry

end NUMINAMATH_CALUDE_choose_and_assign_officers_l2175_217517


namespace NUMINAMATH_CALUDE_tangent_line_at_one_monotonicity_intervals_negative_condition_l2175_217529

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - a * x

theorem tangent_line_at_one (h : ℝ → ℝ) :
  (∀ x > 0, h x = f (-2) x) →
  (∀ x, h x = -x + 1) →
  ∃ c, h c = f (-2) c ∧ (∀ x, h x - (f (-2) c) = -(x - c)) :=
sorry

theorem monotonicity_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x y, 0 < x → 0 < y → x < y → f a x < f a y) ∧
  (a > 0 → ∀ x y, 0 < x → x < 1/a → x < y → f a x < f a y) ∧
  (a > 0 → ∀ x y, 1/a < x → x < y → f a y < f a x) :=
sorry

theorem negative_condition (a : ℝ) :
  (∀ x > 0, f a x < 0) ↔ a > 1/exp 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_monotonicity_intervals_negative_condition_l2175_217529


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l2175_217504

theorem sqrt_product_equality : Real.sqrt 50 * Real.sqrt 18 * Real.sqrt 8 = 60 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l2175_217504


namespace NUMINAMATH_CALUDE_pond_volume_pond_volume_proof_l2175_217520

/-- The volume of a rectangular prism with dimensions 20 m, 10 m, and 5 m is 1000 cubic meters. -/
theorem pond_volume : ℝ → Prop :=
  fun volume =>
    let length : ℝ := 20
    let width : ℝ := 10
    let depth : ℝ := 5
    volume = length * width * depth ∧ volume = 1000

/-- Proof of the pond volume theorem -/
theorem pond_volume_proof : ∃ volume : ℝ, pond_volume volume := by
  sorry

end NUMINAMATH_CALUDE_pond_volume_pond_volume_proof_l2175_217520


namespace NUMINAMATH_CALUDE_alpine_ridge_length_l2175_217592

/-- Represents the Alpine Ridge Trail hike --/
structure AlpineRidgeTrail where
  /-- Distance hiked on each of the five days --/
  day : Fin 5 → ℝ
  /-- First three days total 30 miles --/
  first_three_days : day 0 + day 1 + day 2 = 30
  /-- Second and fourth days average 15 miles --/
  second_fourth_avg : (day 1 + day 3) / 2 = 15
  /-- Last two days total 28 miles --/
  last_two_days : day 3 + day 4 = 28
  /-- First and fourth days total 34 miles --/
  first_fourth_days : day 0 + day 3 = 34

/-- The total length of the Alpine Ridge Trail is 58 miles --/
theorem alpine_ridge_length (trail : AlpineRidgeTrail) : 
  trail.day 0 + trail.day 1 + trail.day 2 + trail.day 3 + trail.day 4 = 58 := by
  sorry


end NUMINAMATH_CALUDE_alpine_ridge_length_l2175_217592


namespace NUMINAMATH_CALUDE_intersection_A_B_l2175_217553

def A : Set ℝ := {x | x * (x - 2) < 0}
def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_A_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2175_217553


namespace NUMINAMATH_CALUDE_two_fruits_from_five_l2175_217506

/-- The number of ways to choose k items from a set of n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of fruit types available -/
def num_fruits : ℕ := 5

/-- The number of fruits to be chosen -/
def fruits_to_choose : ℕ := 2

theorem two_fruits_from_five :
  choose num_fruits fruits_to_choose = 10 := by
  sorry

end NUMINAMATH_CALUDE_two_fruits_from_five_l2175_217506


namespace NUMINAMATH_CALUDE_valid_galaxish_words_remainder_l2175_217532

/-- Represents the set of letters in Galaxish --/
inductive GalaxishLetter
| S
| T
| U

/-- Represents a Galaxish word as a list of letters --/
def GalaxishWord := List GalaxishLetter

/-- Checks if a letter is a consonant --/
def is_consonant (l : GalaxishLetter) : Bool :=
  match l with
  | GalaxishLetter.S => true
  | GalaxishLetter.T => true
  | GalaxishLetter.U => false

/-- Checks if a Galaxish word is valid --/
def is_valid_galaxish_word (word : GalaxishWord) : Bool :=
  let rec check (w : GalaxishWord) (consonant_count : Nat) : Bool :=
    match w with
    | [] => true
    | l::ls => 
      if is_consonant l then
        check ls (consonant_count + 1)
      else if consonant_count >= 3 then
        check ls 0
      else
        false
  check word 0

/-- Counts the number of valid 8-letter Galaxish words --/
def count_valid_galaxish_words : Nat :=
  sorry

theorem valid_galaxish_words_remainder :
  count_valid_galaxish_words % 1000 = 56 := by sorry

end NUMINAMATH_CALUDE_valid_galaxish_words_remainder_l2175_217532


namespace NUMINAMATH_CALUDE_bat_wings_area_is_3_25_l2175_217591

/-- Rectangle DEFA with given dimensions and points -/
structure Rectangle where
  width : ℝ
  height : ℝ
  dc : ℝ
  cb : ℝ
  ba : ℝ

/-- Calculate the area of the "bat wings" in the given rectangle -/
def batWingsArea (rect : Rectangle) : ℝ := sorry

/-- Theorem stating that the area of the "bat wings" is 3.25 -/
theorem bat_wings_area_is_3_25 (rect : Rectangle) 
  (h1 : rect.width = 5)
  (h2 : rect.height = 3)
  (h3 : rect.dc = 2)
  (h4 : rect.cb = 1.5)
  (h5 : rect.ba = 1.5) :
  batWingsArea rect = 3.25 := by sorry

end NUMINAMATH_CALUDE_bat_wings_area_is_3_25_l2175_217591


namespace NUMINAMATH_CALUDE_f_extrema_on_interval_l2175_217533

def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 36 * x + 1

theorem f_extrema_on_interval :
  let a : ℝ := 1
  let b : ℝ := 11
  ∀ x ∈ Set.Icc a b, 
    f x ≥ -43 ∧ f x ≤ 2630 ∧ 
    (∃ x₁ ∈ Set.Icc a b, f x₁ = -43) ∧ 
    (∃ x₂ ∈ Set.Icc a b, f x₂ = 2630) := by
  sorry

end NUMINAMATH_CALUDE_f_extrema_on_interval_l2175_217533


namespace NUMINAMATH_CALUDE_right_triangle_345_ratio_l2175_217585

theorem right_triangle_345_ratio (a b c : ℝ) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_ratio : a / b = 3 / 4 ∧ b / c = 4 / 5) : a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_345_ratio_l2175_217585


namespace NUMINAMATH_CALUDE_triplet_sum_not_two_l2175_217501

theorem triplet_sum_not_two : ∃! (x y z : ℝ), 
  ((x = 2.2 ∧ y = -3.2 ∧ z = 2.0) ∨
   (x = 3/4 ∧ y = 1/2 ∧ z = 3/4) ∨
   (x = 4 ∧ y = -6 ∧ z = 4) ∨
   (x = 0.4 ∧ y = 0.5 ∧ z = 1.1) ∨
   (x = 2/3 ∧ y = 1/3 ∧ z = 1)) ∧
  x + y + z ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_triplet_sum_not_two_l2175_217501


namespace NUMINAMATH_CALUDE_equation_is_linear_with_one_var_l2175_217563

/-- A linear equation with one variable is an equation of the form ax + b = c, where a ≠ 0 and x is the variable. --/
def is_linear_equation_one_var (eq : ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, eq x ↔ a * x + b = c)

/-- The equation 4a - 1 = 8 --/
def equation (a : ℝ) : Prop := 4 * a - 1 = 8

theorem equation_is_linear_with_one_var : is_linear_equation_one_var equation := by
  sorry

end NUMINAMATH_CALUDE_equation_is_linear_with_one_var_l2175_217563


namespace NUMINAMATH_CALUDE_modular_inverse_13_mod_1728_l2175_217596

theorem modular_inverse_13_mod_1728 : ∃ x : ℕ, x < 1728 ∧ (13 * x) % 1728 = 1 :=
by
  use 133
  sorry

end NUMINAMATH_CALUDE_modular_inverse_13_mod_1728_l2175_217596


namespace NUMINAMATH_CALUDE_sheep_raising_profit_range_l2175_217584

/-- Represents the profit calculation for sheep raising with and without technical guidance. -/
theorem sheep_raising_profit_range (x : ℝ) : 
  x > 0 →
  (0.15 * (1 + 0.25*x) * (100000 - x) ≥ 0.15 * 100000) ↔ 
  (0 < x ∧ x ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_sheep_raising_profit_range_l2175_217584


namespace NUMINAMATH_CALUDE_intersection_of_sets_l2175_217597

theorem intersection_of_sets : ∀ (A B : Set ℕ),
  A = {1, 2, 3, 4, 5} →
  B = {3, 5} →
  A ∩ B = {3, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l2175_217597


namespace NUMINAMATH_CALUDE_geometric_sequence_101st_term_l2175_217505

def geometric_sequence (a : ℕ → ℝ) := ∃ (r : ℝ), ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_101st_term
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_3rd : a 3 = 3)
  (h_sum : a 2016 + a 2017 = 0) :
  a 101 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_101st_term_l2175_217505


namespace NUMINAMATH_CALUDE_gina_credits_l2175_217578

/-- Proves that given the conditions of Gina's college expenses, she is taking 14 credits -/
theorem gina_credits : 
  let credit_cost : ℕ := 450
  let textbook_count : ℕ := 5
  let textbook_cost : ℕ := 120
  let facilities_fee : ℕ := 200
  let total_cost : ℕ := 7100
  ∃ (credits : ℕ), 
    credits * credit_cost + 
    textbook_count * textbook_cost + 
    facilities_fee = total_cost ∧ 
    credits = 14 :=
by sorry

end NUMINAMATH_CALUDE_gina_credits_l2175_217578


namespace NUMINAMATH_CALUDE_ranked_choice_voting_theorem_l2175_217513

theorem ranked_choice_voting_theorem 
  (initial_votes_A initial_votes_B initial_votes_C initial_votes_D initial_votes_E : ℚ)
  (redistribution_D_to_A redistribution_D_to_B redistribution_D_to_C : ℚ)
  (redistribution_E_to_A redistribution_E_to_B redistribution_E_to_C : ℚ)
  (majority_difference : ℕ)
  (h1 : initial_votes_A = 35/100)
  (h2 : initial_votes_B = 25/100)
  (h3 : initial_votes_C = 20/100)
  (h4 : initial_votes_D = 15/100)
  (h5 : initial_votes_E = 5/100)
  (h6 : redistribution_D_to_A = 60/100)
  (h7 : redistribution_D_to_B = 25/100)
  (h8 : redistribution_D_to_C = 15/100)
  (h9 : redistribution_E_to_A = 50/100)
  (h10 : redistribution_E_to_B = 30/100)
  (h11 : redistribution_E_to_C = 20/100)
  (h12 : majority_difference = 1890) :
  ∃ (total_votes : ℕ),
    total_votes = 11631 ∧
    (initial_votes_A + redistribution_D_to_A * initial_votes_D + redistribution_E_to_A * initial_votes_E) * total_votes -
    (initial_votes_B + redistribution_D_to_B * initial_votes_D + redistribution_E_to_B * initial_votes_E) * total_votes =
    majority_difference := by
  sorry


end NUMINAMATH_CALUDE_ranked_choice_voting_theorem_l2175_217513


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2175_217587

/-- The complex number z = (3+i)/(1-i) corresponds to a point in the first quadrant -/
theorem complex_number_in_first_quadrant : 
  let z : ℂ := (3 + Complex.I) / (1 - Complex.I)
  (0 < z.re) ∧ (0 < z.im) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2175_217587


namespace NUMINAMATH_CALUDE_circumscribed_circle_radius_of_specific_trapezoid_l2175_217558

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  a : ℝ  -- Length of the longer base
  b : ℝ  -- Length of the shorter base
  h : ℝ  -- Height of the trapezoid

/-- The radius of the circumscribed circle of an isosceles trapezoid -/
def circumscribedCircleRadius (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem: The radius of the circumscribed circle of the given isosceles trapezoid is 85/8 -/
theorem circumscribed_circle_radius_of_specific_trapezoid :
  let t : IsoscelesTrapezoid := { a := 21, b := 9, h := 8 }
  circumscribedCircleRadius t = 85 / 8 := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_circle_radius_of_specific_trapezoid_l2175_217558


namespace NUMINAMATH_CALUDE_counterfeit_probability_l2175_217550

def total_bills : ℕ := 20
def counterfeit_bills : ℕ := 5
def selected_bills : ℕ := 2

def prob_both_counterfeit : ℚ := (counterfeit_bills.choose selected_bills : ℚ) / (total_bills.choose selected_bills)
def prob_at_least_one_counterfeit : ℚ := 1 - ((total_bills - counterfeit_bills).choose selected_bills : ℚ) / (total_bills.choose selected_bills)

theorem counterfeit_probability :
  prob_both_counterfeit / prob_at_least_one_counterfeit = 2 / 17 := by
  sorry

end NUMINAMATH_CALUDE_counterfeit_probability_l2175_217550


namespace NUMINAMATH_CALUDE_parallel_line_with_chord_sum_exists_l2175_217568

-- Define a circle in a plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line in a plane
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

-- Theorem statement
theorem parallel_line_with_chord_sum_exists 
  (S₁ S₂ : Circle) (l : Line) (a : ℝ) (h : a > 0) :
  ∃ (l' : Line),
    (∀ (p : ℝ × ℝ), l.point.1 + p.1 * l.direction.1 = l'.point.1 + p.1 * l'.direction.1 ∧
                     l.point.2 + p.2 * l.direction.2 = l'.point.2 + p.2 * l'.direction.2) ∧
    ∃ (A B C D : ℝ × ℝ),
      (A.1 - B.1)^2 + (A.2 - B.2)^2 = ((A.1 - S₁.center.1)^2 + (A.2 - S₁.center.2)^2 - S₁.radius^2) ∧
      (C.1 - D.1)^2 + (C.2 - D.2)^2 = ((C.1 - S₂.center.1)^2 + (C.2 - S₂.center.2)^2 - S₂.radius^2) ∧
      (A.1 - B.1)^2 + (A.2 - B.2)^2 + (C.1 - D.1)^2 + (C.2 - D.2)^2 = a^2 :=
by
  sorry


end NUMINAMATH_CALUDE_parallel_line_with_chord_sum_exists_l2175_217568


namespace NUMINAMATH_CALUDE_solution_properties_l2175_217523

def system_of_equations (t x y : ℝ) : Prop :=
  (4*t^2 + t + 4)*x + (5*t + 1)*y = 4*t^2 - t - 3 ∧
  (t + 2)*x + 2*y = t

theorem solution_properties :
  ∀ t : ℝ,
  (∀ x y : ℝ, system_of_equations t x y →
    (t < -1 → x < 0 ∧ y < 0) ∧
    (-1 < t ∧ t < 1 ∧ t ≠ 0 → x = (t+1)/(t-1) ∧ y = (2*t+1)/(t-1)) ∧
    (t = 1 → ∀ k : ℝ, ∃ x y : ℝ, system_of_equations t x y) ∧
    (t = 2 → ¬∃ x y : ℝ, system_of_equations t x y) ∧
    (t > 2 → x > 0 ∧ y > 0)) :=
by sorry

end NUMINAMATH_CALUDE_solution_properties_l2175_217523


namespace NUMINAMATH_CALUDE_value_of_b_l2175_217556

theorem value_of_b (x y : ℝ) : 
  x = (1 - Real.sqrt 3) / (1 + Real.sqrt 3) →
  y = (1 + Real.sqrt 3) / (1 - Real.sqrt 3) →
  2 * x^2 - 3 * x * y + 2 * y^2 = 25 :=
by sorry

end NUMINAMATH_CALUDE_value_of_b_l2175_217556


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2175_217552

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 5 + a 6 + a 7 + a 8 + a 9 = 450) :
  a 3 + a 11 = 180 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2175_217552


namespace NUMINAMATH_CALUDE_g_composition_theorem_l2175_217507

/-- The function g defined as g(x) = bx^3 - 1 --/
def g (b : ℝ) (x : ℝ) : ℝ := b * x^3 - 1

/-- Theorem stating that if g(g(1)) = -1 and b is positive, then b = 1 --/
theorem g_composition_theorem (b : ℝ) (h1 : b > 0) (h2 : g b (g b 1) = -1) : b = 1 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_theorem_l2175_217507


namespace NUMINAMATH_CALUDE_no_common_terms_except_one_l2175_217588

def x : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => x (n + 1) + 2 * x n

def y : ℕ → ℤ
  | 0 => 1
  | 1 => 7
  | (n + 2) => 2 * y (n + 1) + 3 * y n

theorem no_common_terms_except_one :
  ∀ m n : ℕ, m > 0 ∧ n > 0 → x m ≠ y n :=
by sorry

end NUMINAMATH_CALUDE_no_common_terms_except_one_l2175_217588


namespace NUMINAMATH_CALUDE_treasure_chest_rubies_l2175_217544

/-- Given a treasure chest with gems, calculate the number of rubies. -/
theorem treasure_chest_rubies (total_gems diamonds : ℕ) (h1 : total_gems = 5155) (h2 : diamonds = 45) :
  total_gems - diamonds = 5110 := by
  sorry

end NUMINAMATH_CALUDE_treasure_chest_rubies_l2175_217544


namespace NUMINAMATH_CALUDE_students_liking_both_sea_and_mountains_l2175_217576

theorem students_liking_both_sea_and_mountains 
  (total_students : ℕ)
  (sea_lovers : ℕ)
  (mountain_lovers : ℕ)
  (neither_lovers : ℕ)
  (h1 : total_students = 500)
  (h2 : sea_lovers = 337)
  (h3 : mountain_lovers = 289)
  (h4 : neither_lovers = 56) :
  sea_lovers + mountain_lovers - (total_students - neither_lovers) = 182 := by
  sorry

end NUMINAMATH_CALUDE_students_liking_both_sea_and_mountains_l2175_217576


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l2175_217519

theorem negative_fraction_comparison : -5/6 < -4/5 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l2175_217519


namespace NUMINAMATH_CALUDE_pentagon_centroid_intersection_l2175_217579

/-- Given a convex pentagon ABCDE in a real vector space, prove that the point P defined as
    (1/5)(A + B + C + D + E) is the intersection point of all segments connecting the midpoint
    of each side to the centroid of the triangle formed by the other three vertices. -/
theorem pentagon_centroid_intersection
  (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
  (A B C D E : V) :
  let P := (1/5 : ℝ) • (A + B + C + D + E)
  let midpoint (X Y : V) := (1/2 : ℝ) • (X + Y)
  let centroid (X Y Z : V) := (1/3 : ℝ) • (X + Y + Z)
  ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 →
    (midpoint A B + t • (centroid C D E - midpoint A B) = P) ∧
    (midpoint B C + t • (centroid D E A - midpoint B C) = P) ∧
    (midpoint C D + t • (centroid E A B - midpoint C D) = P) ∧
    (midpoint D E + t • (centroid A B C - midpoint D E) = P) ∧
    (midpoint E A + t • (centroid B C D - midpoint E A) = P) :=
by sorry


end NUMINAMATH_CALUDE_pentagon_centroid_intersection_l2175_217579


namespace NUMINAMATH_CALUDE_parabola_shift_correct_l2175_217564

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = 2x^2 -/
def original_parabola : Parabola := { a := 2, b := 0, c := 0 }

/-- Shift a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c - v }

theorem parabola_shift_correct :
  let shifted := shift_parabola original_parabola 1 5
  shifted.a = 2 ∧ shifted.b = -4 ∧ shifted.c = -5 := by sorry

end NUMINAMATH_CALUDE_parabola_shift_correct_l2175_217564


namespace NUMINAMATH_CALUDE_equation_solution_l2175_217522

theorem equation_solution (x : ℝ) : 
  (1 - |Real.cos x|) / (1 + |Real.cos x|) = Real.sin x → 
  (∃ k : ℤ, x = k * Real.pi ∨ x = 2 * k * Real.pi + Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2175_217522


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2175_217548

theorem quadratic_roots_property (m n : ℝ) : 
  (m^2 + 2*m - 5 = 0) → (n^2 + 2*n - 5 = 0) → (m^2 + m*n + 2*m = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2175_217548


namespace NUMINAMATH_CALUDE_teacher_student_arrangement_l2175_217530

theorem teacher_student_arrangement (n : ℕ) (m : ℕ) :
  n = 1 ∧ m = 6 →
  (n + m - 2) * (m.factorial) = 3600 :=
by sorry

end NUMINAMATH_CALUDE_teacher_student_arrangement_l2175_217530


namespace NUMINAMATH_CALUDE_art_display_side_length_l2175_217571

/-- Represents the dimensions of a glass pane -/
structure GlassPane where
  width : ℝ
  height : ℝ
  ratio : height = 3 * width

/-- Represents the square art display -/
structure ArtDisplay where
  pane : GlassPane
  border_width : ℝ
  horizontal_panes : ℕ
  vertical_panes : ℕ
  is_square : horizontal_panes * pane.width + (horizontal_panes + 1) * border_width = 
              vertical_panes * pane.height + (vertical_panes + 1) * border_width

/-- The side length of the square display is 17.4 inches -/
theorem art_display_side_length (display : ArtDisplay) 
  (h1 : display.horizontal_panes = 4)
  (h2 : display.vertical_panes = 3)
  (h3 : display.border_width = 3) :
  display.horizontal_panes * display.pane.width + (display.horizontal_panes + 1) * display.border_width = 17.4 := by
  sorry

end NUMINAMATH_CALUDE_art_display_side_length_l2175_217571


namespace NUMINAMATH_CALUDE_prob_sum_greater_than_9_l2175_217521

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := numFaces * numFaces

/-- The number of favorable outcomes (sum > 9) -/
def favorableOutcomes : ℕ := 6

/-- The probability of rolling a sum greater than 9 with two dice -/
def probSumGreaterThan9 : ℚ := favorableOutcomes / totalOutcomes

theorem prob_sum_greater_than_9 : probSumGreaterThan9 = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_greater_than_9_l2175_217521


namespace NUMINAMATH_CALUDE_fraction_sum_l2175_217581

theorem fraction_sum : (3 : ℚ) / 8 + (9 : ℚ) / 12 = (9 : ℚ) / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l2175_217581


namespace NUMINAMATH_CALUDE_sum_equals_negative_six_l2175_217569

theorem sum_equals_negative_six (a b c d : ℤ) :
  (∃ x : ℤ, a + 2 = x ∧ b + 3 = x ∧ c + 4 = x ∧ d + 5 = x ∧ a + b + c + d + 8 = x) →
  a + b + c + d = -6 := by
sorry

end NUMINAMATH_CALUDE_sum_equals_negative_six_l2175_217569


namespace NUMINAMATH_CALUDE_min_sum_with_conditions_l2175_217572

theorem min_sum_with_conditions (a b : ℕ+) 
  (h1 : ¬ 5 ∣ a.val)
  (h2 : ¬ 5 ∣ b.val)
  (h3 : (5 : ℕ)^5 ∣ a.val^5 + b.val^5) :
  ∀ (x y : ℕ+), 
    (¬ 5 ∣ x.val) → 
    (¬ 5 ∣ y.val) → 
    ((5 : ℕ)^5 ∣ x.val^5 + y.val^5) → 
    (a.val + b.val ≤ x.val + y.val) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_with_conditions_l2175_217572


namespace NUMINAMATH_CALUDE_misread_division_sum_l2175_217559

theorem misread_division_sum (D : ℕ) (h : D = 56 * 25 + 25) :
  ∃ (q r : ℕ), D = 65 * q + r ∧ r < 65 ∧ q + r = 81 := by
  sorry

end NUMINAMATH_CALUDE_misread_division_sum_l2175_217559


namespace NUMINAMATH_CALUDE_marks_tomatoes_l2175_217502

/-- Given that Mark bought tomatoes at $5 per pound and 5 pounds of apples at $6 per pound,
    spending a total of $40, prove that he bought 2 pounds of tomatoes. -/
theorem marks_tomatoes :
  ∀ (tomato_price apple_price : ℝ) (apple_pounds : ℝ) (total_spent : ℝ),
    tomato_price = 5 →
    apple_price = 6 →
    apple_pounds = 5 →
    total_spent = 40 →
    ∃ (tomato_pounds : ℝ),
      tomato_pounds * tomato_price + apple_pounds * apple_price = total_spent ∧
      tomato_pounds = 2 :=
by sorry

end NUMINAMATH_CALUDE_marks_tomatoes_l2175_217502


namespace NUMINAMATH_CALUDE_socks_theorem_l2175_217540

def socks_problem (week1 week2 week3 week4 total : ℕ) : Prop :=
  week1 = 12 ∧
  week2 = week1 + 4 ∧
  week3 = (week1 + week2) / 2 ∧
  total = 57 ∧
  total = week1 + week2 + week3 + week4 ∧
  week3 - week4 = 1

theorem socks_theorem : ∃ week1 week2 week3 week4 total : ℕ,
  socks_problem week1 week2 week3 week4 total := by
  sorry

end NUMINAMATH_CALUDE_socks_theorem_l2175_217540


namespace NUMINAMATH_CALUDE_exists_30digit_root_l2175_217535

/-- A function that checks if a number is a three-digit natural number -/
def isThreeDigitNatural (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The main theorem -/
theorem exists_30digit_root (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℕ) 
  (h₀ : isThreeDigitNatural a₀)
  (h₁ : isThreeDigitNatural a₁)
  (h₂ : isThreeDigitNatural a₂)
  (h₃ : isThreeDigitNatural a₃)
  (h₄ : isThreeDigitNatural a₄)
  (h₅ : isThreeDigitNatural a₅)
  (h₆ : isThreeDigitNatural a₆)
  (h₇ : isThreeDigitNatural a₇)
  (h₈ : isThreeDigitNatural a₈)
  (h₉ : isThreeDigitNatural a₉) :
  ∃ (N : ℕ) (x : ℤ), 
    (N ≥ 10^29 ∧ N < 10^30) ∧ 
    (a₉ * x^9 + a₈ * x^8 + a₇ * x^7 + a₆ * x^6 + a₅ * x^5 + 
     a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀ = N) := by
  sorry

end NUMINAMATH_CALUDE_exists_30digit_root_l2175_217535


namespace NUMINAMATH_CALUDE_offspring_different_genes_l2175_217508

structure Eukaryote where
  genes : Set String

def sexualReproduction (parent1 parent2 : Eukaryote) : Eukaryote :=
  sorry

theorem offspring_different_genes (parent1 parent2 : Eukaryote) :
  let offspring := sexualReproduction parent1 parent2
  ∃ (gene : String), (gene ∈ offspring.genes ∧ gene ∉ parent1.genes) ∨
                     (gene ∈ offspring.genes ∧ gene ∉ parent2.genes) :=
  sorry

end NUMINAMATH_CALUDE_offspring_different_genes_l2175_217508


namespace NUMINAMATH_CALUDE_largest_base6_5digit_value_l2175_217582

def largest_base6_5digit : ℕ := 5 * 6^4 + 5 * 6^3 + 5 * 6^2 + 5 * 6^1 + 5 * 6^0

theorem largest_base6_5digit_value : largest_base6_5digit = 7775 := by
  sorry

end NUMINAMATH_CALUDE_largest_base6_5digit_value_l2175_217582


namespace NUMINAMATH_CALUDE_infinitely_many_primes_congruent_one_mod_power_of_two_l2175_217543

theorem infinitely_many_primes_congruent_one_mod_power_of_two (r : ℕ) (hr : r ≥ 1) :
  Set.Infinite {p : ℕ | Nat.Prime p ∧ p ≡ 1 [MOD 2^r]} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_congruent_one_mod_power_of_two_l2175_217543


namespace NUMINAMATH_CALUDE_tenth_odd_multiple_of_5_l2175_217525

/-- The nth odd multiple of 5 -/
def nthOddMultipleOf5 (n : ℕ) : ℕ := 10 * n - 5

/-- Predicate for a number being odd and a multiple of 5 -/
def isOddMultipleOf5 (k : ℕ) : Prop := k % 2 = 1 ∧ k % 5 = 0

theorem tenth_odd_multiple_of_5 : 
  (∃ (k : ℕ), k > 0 ∧ isOddMultipleOf5 k ∧ 
    (∃ (count : ℕ), count = 10 ∧ 
      (∀ (j : ℕ), j > 0 ∧ j < k → isOddMultipleOf5 j → 
        (∃ (m : ℕ), m < count ∧ nthOddMultipleOf5 m = j)))) → 
  nthOddMultipleOf5 10 = 95 :=
sorry

end NUMINAMATH_CALUDE_tenth_odd_multiple_of_5_l2175_217525


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_5_pow_5_plus_10_pow_4_l2175_217580

theorem greatest_prime_factor_of_5_pow_5_plus_10_pow_4 :
  (Nat.factors (5^5 + 10^4)).maximum? = some 7 :=
sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_5_pow_5_plus_10_pow_4_l2175_217580


namespace NUMINAMATH_CALUDE_divisibility_theorem_l2175_217524

theorem divisibility_theorem (N : ℕ) (h : N > 1) :
  ∃ k : ℤ, (N^2)^2014 - (N^11)^106 = k * (N^6 + N^3 + 1) := by sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l2175_217524


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2175_217595

theorem quadratic_equation_solution (x : ℝ) : 
  -x^2 - (-18 + 12) * x - 8 = -(x - 2) * (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2175_217595


namespace NUMINAMATH_CALUDE_sum_of_squares_first_15_sum_of_squares_second_15_l2175_217594

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_squares_first_15 :
  sum_of_squares 15 = 1200 :=
by sorry

-- The condition about the second 15 integers is not directly used in the proof,
-- but we include it as a hypothesis to match the original problem
theorem sum_of_squares_second_15 :
  sum_of_squares 30 - sum_of_squares 15 = 8175 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_first_15_sum_of_squares_second_15_l2175_217594
