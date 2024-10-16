import Mathlib

namespace NUMINAMATH_CALUDE_extra_page_number_l2846_284611

/-- Given a book with 77 pages, if one page number is included three times
    instead of once, resulting in a sum of 3028, then the page number
    that was added extra times is 25. -/
theorem extra_page_number :
  let n : ℕ := 77
  let correct_sum := n * (n + 1) / 2
  let incorrect_sum := 3028
  ∃ k : ℕ, k ≤ n ∧ correct_sum + 2 * k = incorrect_sum ∧ k = 25 := by
  sorry

#check extra_page_number

end NUMINAMATH_CALUDE_extra_page_number_l2846_284611


namespace NUMINAMATH_CALUDE_math_club_composition_l2846_284648

theorem math_club_composition (boys girls : ℕ) : 
  boys = girls →
  (girls : ℚ) = 3/4 * (boys + girls - 1 : ℚ) →
  boys = 2 ∧ girls = 3 := by
sorry

end NUMINAMATH_CALUDE_math_club_composition_l2846_284648


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2846_284695

theorem inequality_system_solution (x : ℤ) :
  (-3/2 : ℚ) < x ∧ (x : ℚ) ≤ 2 →
  -2*x + 7 < 10 ∧ (7*x + 1)/5 - 1 ≤ x := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2846_284695


namespace NUMINAMATH_CALUDE_number_of_winning_scores_l2846_284640

/-- Represents a cross country meet between two teams -/
structure CrossCountryMeet where
  /-- The number of runners in each team -/
  runnersPerTeam : Nat
  /-- The total number of runners -/
  totalRunners : Nat
  /-- Assertion that there are two teams -/
  twoTeams : totalRunners = 2 * runnersPerTeam

/-- Calculates the total score of all runners -/
def totalScore (meet : CrossCountryMeet) : Nat :=
  (meet.totalRunners * (meet.totalRunners + 1)) / 2

/-- Calculates the minimum possible team score -/
def minTeamScore (meet : CrossCountryMeet) : Nat :=
  (meet.runnersPerTeam * (meet.runnersPerTeam + 1)) / 2

/-- Calculates the maximum possible winning score -/
def maxWinningScore (meet : CrossCountryMeet) : Nat :=
  (totalScore meet) / 2 - 1

/-- The main theorem stating the number of possible winning scores -/
theorem number_of_winning_scores (meet : CrossCountryMeet) 
  (h : meet.runnersPerTeam = 6) : 
  (maxWinningScore meet) - (minTeamScore meet) + 1 = 18 := by
  sorry


end NUMINAMATH_CALUDE_number_of_winning_scores_l2846_284640


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2846_284649

theorem sqrt_equation_solution : ∃! x : ℝ, Real.sqrt (2 * x + 3) = x := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2846_284649


namespace NUMINAMATH_CALUDE_complex_sum_product_real_l2846_284660

theorem complex_sum_product_real (a b : ℝ) : 
  let z1 : ℂ := -1 + a * I
  let z2 : ℂ := b - I
  (∃ (r1 : ℝ), z1 + z2 = r1) ∧ (∃ (r2 : ℝ), z1 * z2 = r2) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_product_real_l2846_284660


namespace NUMINAMATH_CALUDE_range_of_a_for_max_and_min_l2846_284688

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*(a + 2)*x + 1

/-- The theorem stating the range of a for which f has both a maximum and a minimum -/
theorem range_of_a_for_max_and_min (a : ℝ) :
  (∃ (max min : ℝ), ∀ x, f a x ≤ max ∧ f a x ≥ min) → (a > 2 ∨ a < -1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_max_and_min_l2846_284688


namespace NUMINAMATH_CALUDE_max_value_P_l2846_284617

theorem max_value_P (a b x₁ x₂ x₃ : ℝ) 
  (h1 : a = x₁ + x₂ + x₃)
  (h2 : a = x₁ * x₂ * x₃)
  (h3 : a * b = x₁ * x₂ + x₂ * x₃ + x₃ * x₁)
  (h4 : x₁ > 0)
  (h5 : x₂ > 0)
  (h6 : x₃ > 0) :
  let P := (a^2 + 6*b + 1) / (a^2 + a)
  ∃ (max_P : ℝ), ∀ (P_val : ℝ), P ≤ P_val → max_P ≥ P_val ∧ max_P = (9 + Real.sqrt 3) / 9 := by
  sorry


end NUMINAMATH_CALUDE_max_value_P_l2846_284617


namespace NUMINAMATH_CALUDE_power_seven_mod_nine_l2846_284674

theorem power_seven_mod_nine : 7^145 % 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_mod_nine_l2846_284674


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2846_284622

/-- Given vectors a and b in ℝ³, if a is perpendicular to b, then x = -2 -/
theorem perpendicular_vectors (a b : ℝ × ℝ × ℝ) (h : a.1 = -1 ∧ a.2.1 = 2 ∧ a.2.2 = 1/2) 
  (k : b.1 = -3 ∧ b.2.2 = 2) (perp : a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2 = 0) :
  b.2.1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2846_284622


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2846_284692

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 20) : 
  1 / x + 1 / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2846_284692


namespace NUMINAMATH_CALUDE_intersection_of_lines_l2846_284678

/-- The intersection point of two lines in 2D space -/
def intersection_point (a b c d e f : ℝ) : ℝ × ℝ := sorry

/-- Theorem: The point (-1, -2) is the unique intersection of the given lines -/
theorem intersection_of_lines :
  let line1 : ℝ → ℝ → Prop := λ x y => 2 * x + 3 * y + 8 = 0
  let line2 : ℝ → ℝ → Prop := λ x y => x - y - 1 = 0
  let point := (-1, -2)
  (line1 point.1 point.2 ∧ line2 point.1 point.2) ∧
  (∀ x y, line1 x y ∧ line2 x y → (x, y) = point) := by
sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l2846_284678


namespace NUMINAMATH_CALUDE_shells_calculation_l2846_284635

/-- Given an initial amount of shells and an additional amount added, 
    calculate the total amount of shells -/
def total_shells (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that with 5 pounds initial and 23 pounds added, 
    the total is 28 pounds -/
theorem shells_calculation :
  total_shells 5 23 = 28 := by
  sorry

end NUMINAMATH_CALUDE_shells_calculation_l2846_284635


namespace NUMINAMATH_CALUDE_birthday_gifts_l2846_284698

theorem birthday_gifts (gifts_12th : ℕ) (fewer_gifts : ℕ) : 
  gifts_12th = 20 → fewer_gifts = 8 → 
  gifts_12th + (gifts_12th - fewer_gifts) = 32 := by
  sorry

end NUMINAMATH_CALUDE_birthday_gifts_l2846_284698


namespace NUMINAMATH_CALUDE_special_numbers_count_l2846_284681

/-- A function that checks if a number's digits are consecutive integers -/
def has_consecutive_digits (n : ℕ) : Prop := sorry

/-- A function that returns the number of integers satisfying the given conditions -/
def count_special_numbers : ℕ := sorry

/-- Theorem stating that there are exactly 66 numbers satisfying the given conditions -/
theorem special_numbers_count :
  (∃ (S : Finset ℕ), 
    S.card = 66 ∧ 
    (∀ n ∈ S, 
      1000 ≤ n ∧ n < 10000 ∧
      has_consecutive_digits n ∧
      n % 3 = 0) ∧
    (∀ n : ℕ, 
      1000 ≤ n ∧ n < 10000 ∧
      has_consecutive_digits n ∧
      n % 3 = 0 → n ∈ S)) :=
by sorry

#check special_numbers_count

end NUMINAMATH_CALUDE_special_numbers_count_l2846_284681


namespace NUMINAMATH_CALUDE_choir_arrangement_l2846_284676

theorem choir_arrangement (n : ℕ) : n ≥ 32400 ∧ 
  (∃ k : ℕ, n = k^2) ∧ 
  n % 8 = 0 ∧ n % 9 = 0 ∧ n % 10 = 0 →
  n = 32400 :=
sorry

end NUMINAMATH_CALUDE_choir_arrangement_l2846_284676


namespace NUMINAMATH_CALUDE_eunji_score_l2846_284637

theorem eunji_score (minyoung_score yuna_score eunji_score : ℕ) : 
  minyoung_score = 55 →
  yuna_score = 57 →
  eunji_score > minyoung_score →
  eunji_score < yuna_score →
  eunji_score = 56 := by
sorry

end NUMINAMATH_CALUDE_eunji_score_l2846_284637


namespace NUMINAMATH_CALUDE_find_a_l2846_284619

def U : Set ℕ := {1, 3, 5, 7}

theorem find_a (M : Set ℕ) (a : ℕ) (h1 : M = {1, a}) 
  (h2 : (U \ M) = {5, 7}) : a = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l2846_284619


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l2846_284672

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l2846_284672


namespace NUMINAMATH_CALUDE_tank_capacity_l2846_284644

theorem tank_capacity (x : ℝ) 
  (h1 : x / 4 + 180 = 2 * x / 3) : x = 432 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l2846_284644


namespace NUMINAMATH_CALUDE_cricket_team_right_handed_players_l2846_284669

theorem cricket_team_right_handed_players 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (h1 : total_players = 67)
  (h2 : throwers = 37)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0) :
  throwers + (total_players - throwers) * 2 / 3 = 57 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_right_handed_players_l2846_284669


namespace NUMINAMATH_CALUDE_gcf_of_36_48_72_l2846_284677

theorem gcf_of_36_48_72 : Nat.gcd 36 (Nat.gcd 48 72) = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_36_48_72_l2846_284677


namespace NUMINAMATH_CALUDE_cube_sum_power_of_two_l2846_284655

theorem cube_sum_power_of_two (k : ℕ+) :
  (∃ (a b c : ℕ+), |((a:ℤ) - b)^3 + ((b:ℤ) - c)^3 + ((c:ℤ) - a)^3| = 3 * 2^(k:ℕ)) ↔
  (∃ (n : ℕ), k = 3 * n + 1) :=
sorry

end NUMINAMATH_CALUDE_cube_sum_power_of_two_l2846_284655


namespace NUMINAMATH_CALUDE_scientific_notation_of_small_number_l2846_284606

theorem scientific_notation_of_small_number :
  ∃ (a : ℝ) (n : ℤ), 0.00000000005 = a * 10^n ∧ 1 ≤ a ∧ a < 10 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_small_number_l2846_284606


namespace NUMINAMATH_CALUDE_coeff_x_squared_expansion_l2846_284609

/-- The coefficient of x^2 in the expansion of (x+1)^5(x-2) is -15 -/
theorem coeff_x_squared_expansion : Int := by
  sorry

end NUMINAMATH_CALUDE_coeff_x_squared_expansion_l2846_284609


namespace NUMINAMATH_CALUDE_power_function_through_point_l2846_284651

/-- Given a power function f(x) = x^n that passes through (2, √2), prove f(9) = 3 -/
theorem power_function_through_point (f : ℝ → ℝ) (n : ℝ) :
  (∀ x, f x = x^n) →
  f 2 = Real.sqrt 2 →
  f 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l2846_284651


namespace NUMINAMATH_CALUDE_bouquets_to_buy_is_correct_l2846_284699

/-- Represents the number of roses in a bouquet Bill buys -/
def roses_per_bought_bouquet : ℕ := 7

/-- Represents the number of roses in a bouquet Bill sells -/
def roses_per_sold_bouquet : ℕ := 5

/-- Represents the price of a bouquet (both buying and selling) -/
def price_per_bouquet : ℕ := 20

/-- Represents the target profit -/
def target_profit : ℕ := 1000

/-- Calculates the number of bouquets Bill needs to buy to earn the target profit -/
def bouquets_to_buy : ℕ :=
  let bought_bouquets_per_operation := roses_per_sold_bouquet
  let sold_bouquets_per_operation := roses_per_bought_bouquet
  let profit_per_operation := sold_bouquets_per_operation * price_per_bouquet - bought_bouquets_per_operation * price_per_bouquet
  let operations_needed := target_profit / profit_per_operation
  operations_needed * bought_bouquets_per_operation

theorem bouquets_to_buy_is_correct :
  bouquets_to_buy = 125 := by sorry

end NUMINAMATH_CALUDE_bouquets_to_buy_is_correct_l2846_284699


namespace NUMINAMATH_CALUDE_teacher_weight_l2846_284630

theorem teacher_weight (num_students : ℕ) (avg_weight : ℝ) (weight_increase : ℝ) : 
  num_students = 24 →
  avg_weight = 35 →
  weight_increase = 0.4 →
  (num_students * avg_weight + (avg_weight + weight_increase) * (num_students + 1)) / (num_students + 1) - avg_weight = weight_increase →
  (num_students + 1) * (avg_weight + weight_increase) - num_students * avg_weight = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_teacher_weight_l2846_284630


namespace NUMINAMATH_CALUDE_unique_prime_sum_10003_l2846_284653

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def sum_of_two_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = n

theorem unique_prime_sum_10003 :
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 10003 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_sum_10003_l2846_284653


namespace NUMINAMATH_CALUDE_at_least_one_less_than_one_l2846_284625

theorem at_least_one_less_than_one (a b c : ℝ) (ha : a < 3) (hb : b < 3) (hc : c < 3) :
  min a (min b c) < 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_less_than_one_l2846_284625


namespace NUMINAMATH_CALUDE_total_length_figure_c_is_59_l2846_284614

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.length)

/-- Represents the configuration of figures A, B, and C -/
structure FigureConfiguration where
  figureA : Rectangle
  figureB : Rectangle
  sharedSegmentLength : ℝ

/-- The given configuration from the problem -/
def givenConfiguration : FigureConfiguration := {
  figureA := { width := 4, length := 9 }
  figureB := { width := 9, length := 9 }
  sharedSegmentLength := 3
}

/-- Calculates the total length of segments in Figure C -/
def totalLengthFigureC (config : FigureConfiguration) : ℝ :=
  perimeter config.figureA + perimeter config.figureB - config.sharedSegmentLength

/-- Theorem stating that the total length of segments in Figure C is 59 units -/
theorem total_length_figure_c_is_59 :
  totalLengthFigureC givenConfiguration = 59 := by
  sorry

end NUMINAMATH_CALUDE_total_length_figure_c_is_59_l2846_284614


namespace NUMINAMATH_CALUDE_five_hour_charge_l2846_284673

/-- Represents the pricing structure and total charge calculation for a psychologist's therapy sessions. -/
structure TherapyPricing where
  /-- The charge for the first hour of therapy -/
  first_hour : ℕ
  /-- The charge for each additional hour of therapy -/
  additional_hour : ℕ
  /-- The first hour costs $30 more than each additional hour -/
  first_hour_premium : first_hour = additional_hour + 30
  /-- The total charge for 3 hours of therapy is $252 -/
  three_hour_charge : first_hour + 2 * additional_hour = 252

/-- Theorem stating that given the pricing structure, the total charge for 5 hours of therapy is $400 -/
theorem five_hour_charge (p : TherapyPricing) : p.first_hour + 4 * p.additional_hour = 400 := by
  sorry

end NUMINAMATH_CALUDE_five_hour_charge_l2846_284673


namespace NUMINAMATH_CALUDE_geometric_sum_not_always_geometric_arithmetic_and_geometric_is_constant_sum_power_not_always_arithmetic_or_geometric_arithmetic_sequence_no_equal_terms_l2846_284668

-- Definition of a geometric sequence
def is_geometric_sequence (a b c d : ℝ) : Prop :=
  (b / a = c / b) ∧ (c / b = d / c)

-- Definition of an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Definition of a geometric sequence (for infinite sequences)
def is_geometric_sequence_inf (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Definition of a constant sequence
def is_constant_sequence (a : ℕ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ n : ℕ, a n = c

theorem geometric_sum_not_always_geometric :
  ∃ a b c d : ℝ, is_geometric_sequence a b c d ∧
  ¬ is_geometric_sequence (a + b) (b + c) (c + d) (d + a) :=
sorry

theorem arithmetic_and_geometric_is_constant (a : ℕ → ℝ) :
  is_arithmetic_sequence a → is_geometric_sequence_inf a → is_constant_sequence a :=
sorry

theorem sum_power_not_always_arithmetic_or_geometric :
  ∃ (a : ℝ) (S : ℕ → ℝ), (∀ n : ℕ, S n = a^n - 1) ∧
  ¬ (is_arithmetic_sequence S ∨ is_geometric_sequence_inf S) :=
sorry

theorem arithmetic_sequence_no_equal_terms (a : ℕ → ℝ) (d : ℝ) :
  is_arithmetic_sequence a → d ≠ 0 → ∀ m n : ℕ, m ≠ n → a m ≠ a n :=
sorry

end NUMINAMATH_CALUDE_geometric_sum_not_always_geometric_arithmetic_and_geometric_is_constant_sum_power_not_always_arithmetic_or_geometric_arithmetic_sequence_no_equal_terms_l2846_284668


namespace NUMINAMATH_CALUDE_remainder_problem_l2846_284636

theorem remainder_problem (h1 : Nat.Prime 73) (h2 : ¬(73 ∣ 57)) :
  (57^35 + 47) % 73 = 55 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2846_284636


namespace NUMINAMATH_CALUDE_sum_equals_3000_length_conversion_l2846_284626

-- Problem 1
theorem sum_equals_3000 : 1361 + 972 + 639 + 28 = 3000 := by sorry

-- Problem 2
theorem length_conversion :
  ∀ (meters decimeters centimeters : ℕ),
    meters * 10 + decimeters - (centimeters / 10) = 91 →
    9 * 10 + 9 - (80 / 10) = 91 := by sorry

end NUMINAMATH_CALUDE_sum_equals_3000_length_conversion_l2846_284626


namespace NUMINAMATH_CALUDE_fraction_of_y_l2846_284618

theorem fraction_of_y (y : ℝ) (h : y > 0) : (9 * y / 20 + 3 * y / 10) / y = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_y_l2846_284618


namespace NUMINAMATH_CALUDE_factor_expression_l2846_284662

theorem factor_expression (x : ℝ) : 35 * x^13 + 245 * x^26 = 35 * x^13 * (1 + 7 * x^13) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2846_284662


namespace NUMINAMATH_CALUDE_eagles_score_is_24_l2846_284697

/-- The combined score of both teams -/
def total_score : ℕ := 56

/-- The margin by which the Falcons won -/
def winning_margin : ℕ := 8

/-- The score of the Eagles -/
def eagles_score : ℕ := total_score / 2 - winning_margin / 2

theorem eagles_score_is_24 : eagles_score = 24 := by
  sorry

end NUMINAMATH_CALUDE_eagles_score_is_24_l2846_284697


namespace NUMINAMATH_CALUDE_log_inequality_l2846_284645

/-- Given a = log_3(2), b = log_2(3), and c = log_(1/2)(5), prove that c < a < b -/
theorem log_inequality (a b c : ℝ) 
  (ha : a = Real.log 2 / Real.log 3)
  (hb : b = Real.log 3 / Real.log 2)
  (hc : c = Real.log 5 / Real.log (1/2)) :
  c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l2846_284645


namespace NUMINAMATH_CALUDE_stratified_sample_problem_l2846_284613

/-- Represents the number of items selected from a specific type in stratified sampling -/
def stratified_sample (total : ℕ) (sample_size : ℕ) (type_ratio : ℕ) (total_ratio : ℕ) : ℕ :=
  (sample_size * type_ratio) / total_ratio

/-- Theorem: In a stratified sampling of 120 items from 600 total items with ratio 1:2:3, 
    the number of items selected from the first type (ratio 1) is 20 -/
theorem stratified_sample_problem :
  stratified_sample 600 120 1 6 = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_problem_l2846_284613


namespace NUMINAMATH_CALUDE_angle_rotation_l2846_284690

theorem angle_rotation (initial_angle rotation : ℝ) (h1 : initial_angle = 25) (h2 : rotation = 350) :
  (initial_angle - (rotation - 360)) % 360 = 15 :=
sorry

end NUMINAMATH_CALUDE_angle_rotation_l2846_284690


namespace NUMINAMATH_CALUDE_standard_deviation_of_data_set_l2846_284643

def data_set : List ℝ := [11, 13, 15, 17, 19]

theorem standard_deviation_of_data_set :
  let n : ℕ := data_set.length
  let mean : ℝ := data_set.sum / n
  let variance : ℝ := (data_set.map (λ x => (x - mean)^2)).sum / n
  let std_dev : ℝ := Real.sqrt variance
  (mean = 15) → (std_dev = 2 * Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_standard_deviation_of_data_set_l2846_284643


namespace NUMINAMATH_CALUDE_inequality_solution_l2846_284646

theorem inequality_solution (x : ℝ) :
  x ≥ 0 →
  (2021 * (x^2020)^(1/202) - 1 ≥ 2020 * x) ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2846_284646


namespace NUMINAMATH_CALUDE_tangent_line_theorem_l2846_284661

/-- A line passing through the origin that is tangent to two curves -/
def TangentLine (f g : ℝ → ℝ) : Prop :=
  ∃ (m : ℝ), (∀ x, f x = m * x) ∧ 
    (∃ x₁, f x₁ = g x₁ ∧ (∀ y, y ≠ x₁ → f y ≠ g y)) ∧
    (f 0 = 0) -- Line passes through origin

/-- The first curve -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

/-- The second curve, parameterized by a -/
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a

theorem tangent_line_theorem :
  ∀ a : ℝ, TangentLine f (g a) → (a = 1 ∨ a = 1/64) := by sorry

end NUMINAMATH_CALUDE_tangent_line_theorem_l2846_284661


namespace NUMINAMATH_CALUDE_slope_of_right_triangle_l2846_284612

/-- Given a right triangle ABC in the x-y plane where:
  * ∠B = 90°
  * AC = 225
  * AB = 180
  Prove that the slope of line segment AC is 4/3 -/
theorem slope_of_right_triangle (A B C : ℝ × ℝ) :
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 180^2 →
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 225^2 →
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 - (B.1 - A.1)^2 - (B.2 - A.2)^2 →
  (C.2 - A.2) / (C.1 - A.1) = 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_slope_of_right_triangle_l2846_284612


namespace NUMINAMATH_CALUDE_min_squares_to_exceed_1000_l2846_284670

/-- The function that represents repeated squaring of a number -/
def repeated_square (x : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => (repeated_square x n) ^ 2

/-- The theorem stating that 3 is the smallest number of squaring operations needed for 5 to exceed 1000 -/
theorem min_squares_to_exceed_1000 :
  (∀ k < 3, repeated_square 5 k ≤ 1000) ∧
  (repeated_square 5 3 > 1000) :=
sorry

end NUMINAMATH_CALUDE_min_squares_to_exceed_1000_l2846_284670


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l2846_284610

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem seventh_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_sum1 : a 1 + a 2 = 3)
  (h_sum2 : a 2 + a 3 = 6) :
  a 7 = 64 := by
sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l2846_284610


namespace NUMINAMATH_CALUDE_r_earns_75_l2846_284656

/-- Represents the daily earnings of individuals p, q, r, and s -/
structure DailyEarnings where
  p : ℚ
  q : ℚ
  r : ℚ
  s : ℚ

/-- The conditions of the problem -/
def earnings_conditions (e : DailyEarnings) : Prop :=
  e.p + e.q + e.r + e.s = 2400 / 8 ∧
  e.p + e.r = 600 / 5 ∧
  e.q + e.r = 910 / 7 ∧
  e.s + e.r = 800 / 4 ∧
  e.p + e.s = 700 / 6

/-- Theorem stating that under the given conditions, r earns 75 per day -/
theorem r_earns_75 (e : DailyEarnings) : 
  earnings_conditions e → e.r = 75 := by
  sorry

#check r_earns_75

end NUMINAMATH_CALUDE_r_earns_75_l2846_284656


namespace NUMINAMATH_CALUDE_ians_jogging_laps_l2846_284682

/-- Given information about Ian's jogging routine, calculate the number of laps he does every night -/
theorem ians_jogging_laps 
  (lap_length : ℝ)
  (feet_per_calorie : ℝ)
  (total_calories : ℝ)
  (total_days : ℝ)
  (h1 : lap_length = 100)
  (h2 : feet_per_calorie = 25)
  (h3 : total_calories = 100)
  (h4 : total_days = 5)
  : (total_calories * feet_per_calorie / total_days) / lap_length = 5 := by
  sorry

end NUMINAMATH_CALUDE_ians_jogging_laps_l2846_284682


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2846_284603

theorem solution_set_quadratic_inequality :
  ∀ x : ℝ, x * (x - 2) ≤ 0 ↔ 0 ≤ x ∧ x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2846_284603


namespace NUMINAMATH_CALUDE_division_problem_l2846_284607

theorem division_problem : (501 : ℝ) / (0.5 : ℝ) = 1002 := by sorry

end NUMINAMATH_CALUDE_division_problem_l2846_284607


namespace NUMINAMATH_CALUDE_max_value_x_l2846_284632

theorem max_value_x : 
  ∃ (x_max : ℝ), 
    (∀ x : ℝ, ((5*x - 25)/(4*x - 5))^2 + ((5*x - 25)/(4*x - 5)) = 20 → x ≤ x_max) ∧
    ((5*x_max - 25)/(4*x_max - 5))^2 + ((5*x_max - 25)/(4*x_max - 5)) = 20 ∧
    x_max = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_l2846_284632


namespace NUMINAMATH_CALUDE_product_of_specific_primes_l2846_284608

def largest_one_digit_prime : ℕ := 7

def largest_two_digit_primes : List ℕ := [97, 89]

theorem product_of_specific_primes : 
  (largest_one_digit_prime * (largest_two_digit_primes.prod)) = 60431 := by
  sorry

end NUMINAMATH_CALUDE_product_of_specific_primes_l2846_284608


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2846_284657

/-- Defines an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Defines a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines a line passing through two points -/
def Line (p1 p2 : Point) :=
  {p : Point | (p.y - p1.y) * (p2.x - p1.x) = (p.x - p1.x) * (p2.y - p1.y)}

theorem ellipse_eccentricity (Γ : Ellipse) 
  (F : Point) 
  (A : Point) 
  (B : Point) 
  (N : Point) :
  F.x = 3 ∧ F.y = 0 →
  A.x = 0 ∧ A.y = Γ.b →
  B.x = 0 ∧ B.y = -Γ.b →
  N.x = 12 ∧ N.y = 0 →
  ∃ (M : Point), M ∈ Line A F ∧ M ∈ Line B N ∧ 
    (M.x^2 / Γ.a^2 + M.y^2 / Γ.b^2 = 1) →
  (Γ.a^2 - Γ.b^2) / Γ.a^2 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2846_284657


namespace NUMINAMATH_CALUDE_abcd_product_magnitude_l2846_284604

theorem abcd_product_magnitude (a b c d : ℝ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 →
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  a^2 + 1/b = b^2 + 1/c → b^2 + 1/c = c^2 + 1/d → c^2 + 1/d = d^2 + 1/a →
  |a*b*c*d| = 1 := by
sorry

end NUMINAMATH_CALUDE_abcd_product_magnitude_l2846_284604


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2846_284641

theorem quadratic_two_distinct_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (a - 3) * x^2 - 4 * x - 1 = 0 ∧ (a - 3) * y^2 - 4 * y - 1 = 0) ↔
  (a > -1 ∧ a ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2846_284641


namespace NUMINAMATH_CALUDE_largest_c_for_4_in_range_l2846_284628

/-- The quadratic function f(x) = x^2 + 5x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 5*x + c

/-- Theorem: The largest value of c such that 4 is in the range of f(x) = x^2 + 5x + c is 10.25 -/
theorem largest_c_for_4_in_range : 
  (∃ (x : ℝ), f 10.25 x = 4) ∧ 
  (∀ (c : ℝ), c > 10.25 → ¬∃ (x : ℝ), f c x = 4) := by
  sorry


end NUMINAMATH_CALUDE_largest_c_for_4_in_range_l2846_284628


namespace NUMINAMATH_CALUDE_cube_surface_area_proof_l2846_284642

-- Define the edge length of the cube
def edge_length : ℝ → ℝ := λ a => 7 * a

-- Define the surface area of a cube given its edge length
def cube_surface_area (edge : ℝ) : ℝ := 6 * edge^2

-- Theorem statement
theorem cube_surface_area_proof (a : ℝ) :
  cube_surface_area (edge_length a) = 294 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_proof_l2846_284642


namespace NUMINAMATH_CALUDE_solve_inequality_find_a_range_l2846_284602

-- Define the function f
def f (x : ℝ) := |x + 2|

-- Part 1: Solve the inequality
theorem solve_inequality :
  {x : ℝ | 2 * f x < 4 - |x - 1|} = {x : ℝ | -7/3 < x ∧ x < -1} := by sorry

-- Part 2: Find the range of a
theorem find_a_range (m n : ℝ) (h1 : m + n = 1) (h2 : m > 0) (h3 : n > 0) :
  (∀ x : ℝ, |x - a| - f x ≤ 1/m + 1/n) ↔ -6 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_solve_inequality_find_a_range_l2846_284602


namespace NUMINAMATH_CALUDE_trip_distance_is_3_6_miles_l2846_284600

/-- Calculates the trip distance given the initial fee, charge per segment, and total charge -/
def calculate_trip_distance (initial_fee : ℚ) (charge_per_segment : ℚ) (segment_length : ℚ) (total_charge : ℚ) : ℚ :=
  let distance_charge := total_charge - initial_fee
  let num_segments := distance_charge / charge_per_segment
  num_segments * segment_length

/-- Proves that the trip distance is 3.6 miles given the specified conditions -/
theorem trip_distance_is_3_6_miles :
  let initial_fee : ℚ := 5/2
  let charge_per_segment : ℚ := 7/20
  let segment_length : ℚ := 2/5
  let total_charge : ℚ := 113/20
  calculate_trip_distance initial_fee charge_per_segment segment_length total_charge = 18/5 := by
  sorry

#eval (18 : ℚ) / 5

end NUMINAMATH_CALUDE_trip_distance_is_3_6_miles_l2846_284600


namespace NUMINAMATH_CALUDE_polar_to_circle_l2846_284663

/-- The equation of the curve in polar coordinates -/
def polar_equation (r θ : ℝ) : Prop :=
  r = 1 / (2 * Real.sin θ - Real.cos θ)

/-- The equation of a circle in Cartesian coordinates -/
def circle_equation (x y : ℝ) (h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Theorem stating that the polar equation represents a circle -/
theorem polar_to_circle :
  ∃ h k r, ∀ x y θ,
    polar_equation (Real.sqrt (x^2 + y^2)) θ →
    x = (Real.sqrt (x^2 + y^2)) * Real.cos θ →
    y = (Real.sqrt (x^2 + y^2)) * Real.sin θ →
    circle_equation x y h k r :=
sorry

end NUMINAMATH_CALUDE_polar_to_circle_l2846_284663


namespace NUMINAMATH_CALUDE_payment_difference_equation_l2846_284623

/-- Represents the payment structure for two artists painting murals. -/
structure MuralPayment where
  diego : ℝ  -- Diego's payment
  celina : ℝ  -- Celina's payment
  total : ℝ   -- Total payment
  h1 : celina > 4 * diego  -- Celina's payment is more than 4 times Diego's
  h2 : celina + diego = total  -- Sum of payments equals total

/-- The difference between Celina's payment and 4 times Diego's payment. -/
def payment_difference (p : MuralPayment) : ℝ := p.celina - 4 * p.diego

/-- Theorem stating the relationship between the payment difference and Diego's payment. -/
theorem payment_difference_equation (p : MuralPayment) (h3 : p.total = 50000) :
  payment_difference p = 50000 - 5 * p.diego := by
  sorry


end NUMINAMATH_CALUDE_payment_difference_equation_l2846_284623


namespace NUMINAMATH_CALUDE_oranges_count_l2846_284667

theorem oranges_count (joan_initial : ℕ) (tom_initial : ℕ) (sara_sold : ℕ) (christine_gave : ℕ)
  (h1 : joan_initial = 75)
  (h2 : tom_initial = 42)
  (h3 : sara_sold = 40)
  (h4 : christine_gave = 15) :
  joan_initial + tom_initial - sara_sold + christine_gave = 92 :=
by sorry

end NUMINAMATH_CALUDE_oranges_count_l2846_284667


namespace NUMINAMATH_CALUDE_seashell_difference_l2846_284631

/-- The number of seashells Tom found -/
def tom_seashells : ℕ := 15

/-- The number of seashells Fred found -/
def fred_seashells : ℕ := 43

/-- The number of cracked seashells -/
def cracked_seashells : ℕ := 29

/-- Theorem stating the difference between Fred's and Tom's seashell counts -/
theorem seashell_difference : fred_seashells - tom_seashells = 28 := by
  sorry

end NUMINAMATH_CALUDE_seashell_difference_l2846_284631


namespace NUMINAMATH_CALUDE_bob_raised_beds_l2846_284686

/-- Represents the dimensions of a raised bed -/
structure BedDimensions where
  height : ℕ
  width : ℕ
  length : ℕ

/-- Calculates the number of planks needed for one raised bed -/
def planksPerBed (dims : BedDimensions) (plankWidth : ℕ) : ℕ :=
  2 * dims.height * (dims.length / plankWidth) + 1

/-- Calculates the number of raised beds that can be constructed -/
def numberOfBeds (dims : BedDimensions) (plankWidth : ℕ) (totalPlanks : ℕ) : ℕ :=
  totalPlanks / planksPerBed dims plankWidth

/-- Theorem: Bob can construct 10 raised beds -/
theorem bob_raised_beds :
  let dims : BedDimensions := { height := 2, width := 2, length := 8 }
  let plankWidth := 1
  let totalPlanks := 50
  numberOfBeds dims plankWidth totalPlanks = 10 := by
  sorry

end NUMINAMATH_CALUDE_bob_raised_beds_l2846_284686


namespace NUMINAMATH_CALUDE_expression_evaluation_l2846_284627

theorem expression_evaluation :
  let a : ℤ := -1
  (2 - a)^2 - (1 + a)*(a - 1) - a*(a - 3) = 5 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2846_284627


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2846_284694

theorem algebraic_expression_value (x : ℝ) : -2 * (2 - x) + (1 + x) = 0 → 2 * x^2 - 7 = -5 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2846_284694


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l2846_284664

theorem consecutive_integers_sum (n : ℤ) : 
  (n - 1) * n * (n + 1) = 336 → (n - 1) + n + (n + 1) = 21 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l2846_284664


namespace NUMINAMATH_CALUDE_chicken_nuggets_order_l2846_284650

/-- The number of chicken nuggets ordered by Alyssa, Keely, and Kendall -/
theorem chicken_nuggets_order (alyssa keely kendall : ℕ) 
  (h1 : alyssa = 20)
  (h2 : keely = 2 * alyssa)
  (h3 : kendall = 2 * alyssa) :
  alyssa + keely + kendall = 100 := by
  sorry

end NUMINAMATH_CALUDE_chicken_nuggets_order_l2846_284650


namespace NUMINAMATH_CALUDE_lawrence_county_kids_at_camp_l2846_284693

def lawrence_county_kids_at_home : ℕ := 134867
def outside_county_kids_at_camp : ℕ := 424944
def total_kids_at_camp : ℕ := 458988

theorem lawrence_county_kids_at_camp :
  total_kids_at_camp - outside_county_kids_at_camp = 34044 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_county_kids_at_camp_l2846_284693


namespace NUMINAMATH_CALUDE_perfect_square_sum_l2846_284684

theorem perfect_square_sum (a b : ℤ) : 
  (∃ x : ℤ, a^4 + (a+b)^4 + b^4 = x^2) ↔ a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l2846_284684


namespace NUMINAMATH_CALUDE_relationship_between_p_and_q_l2846_284634

theorem relationship_between_p_and_q (p q : ℝ) (h : p > 0) (h' : q > 0) (h'' : q ≠ 1) 
  (eq : Real.log p + Real.log q = Real.log (p + q + q^2)) : 
  p = (q + q^2) / (q - 1) := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_p_and_q_l2846_284634


namespace NUMINAMATH_CALUDE_exists_unformable_figure_l2846_284647

/-- Represents a geometric shape --/
inductive Shape
  | Square : Shape
  | Rectangle1x3 : Shape
  | Rectangle2x1 : Shape
  | LShape : Shape

/-- Represents a geometric figure --/
structure Figure where
  area : ℕ
  canBeFormed : Bool

/-- The set of available shapes --/
def availableShapes : List Shape :=
  [Shape.Square, Shape.Square, Shape.Rectangle1x3, Shape.Rectangle2x1, Shape.LShape]

/-- The total area of all available shapes --/
def totalArea : ℕ := 13

/-- There are eight different geometric figures --/
def figures : List Figure := sorry

/-- Theorem: There exists a figure that cannot be formed from the available shapes --/
theorem exists_unformable_figure :
  ∃ (f : Figure), f ∈ figures ∧ f.canBeFormed = false :=
sorry

end NUMINAMATH_CALUDE_exists_unformable_figure_l2846_284647


namespace NUMINAMATH_CALUDE_not_p_and_q_l2846_284621

-- Define proposition p
def p : Prop := ∀ (a b c : ℝ), a < b → a * c^2 < b * c^2

-- Define proposition q
def q : Prop := ∃ (x₀ : ℝ), x₀ > 0 ∧ x₀ - 1 - Real.log x₀ = 0

-- Theorem statement
theorem not_p_and_q : (¬p) ∧ q := by sorry

end NUMINAMATH_CALUDE_not_p_and_q_l2846_284621


namespace NUMINAMATH_CALUDE_hyperbola_iff_mn_neg_l2846_284605

/-- Defines whether an equation represents a hyperbola -/
def is_hyperbola (m n : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / m + y^2 / n = 1 ∧ 
  ¬∃ (a b : ℝ), ∀ (x y : ℝ), x^2 / m + y^2 / n = 1 ↔ (x - a)^2 + (y - b)^2 = 1

/-- Proves that mn < 0 is necessary and sufficient for the equation to represent a hyperbola -/
theorem hyperbola_iff_mn_neg (m n : ℝ) :
  is_hyperbola m n ↔ m * n < 0 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_iff_mn_neg_l2846_284605


namespace NUMINAMATH_CALUDE_sequence_is_increasing_l2846_284629

def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

theorem sequence_is_increasing (a : ℕ → ℝ) 
  (h1 : a 1 < 0) 
  (h2 : ∀ n, a (n + 1) / a n = 1 / 3) : 
  is_increasing a :=
sorry

end NUMINAMATH_CALUDE_sequence_is_increasing_l2846_284629


namespace NUMINAMATH_CALUDE_units_digit_of_quotient_units_digit_zero_l2846_284685

theorem units_digit_of_quotient (n : ℕ) : 
  (7^n + 4^n) % 9 = 2 :=
sorry

theorem units_digit_zero : 
  (7^2023 + 4^2023) / 9 % 10 = 0 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_quotient_units_digit_zero_l2846_284685


namespace NUMINAMATH_CALUDE_green_peaches_count_l2846_284624

theorem green_peaches_count (red : ℕ) (yellow : ℕ) (total : ℕ) (green : ℕ) : 
  red = 7 → yellow = 15 → total = 30 → green = total - (red + yellow) → green = 8 := by
sorry

end NUMINAMATH_CALUDE_green_peaches_count_l2846_284624


namespace NUMINAMATH_CALUDE_desk_lamp_profit_l2846_284620

/-- Profit function for desk lamp sales -/
def profit_function (n : ℝ) (x : ℝ) : ℝ := (x - 20) * (-10 * x + n)

/-- Theorem stating the maximum profit and corresponding selling price -/
theorem desk_lamp_profit (n : ℝ) :
  (profit_function n 25 = 120) →
  (n = 370) ∧
  (∀ x : ℝ, x > 32 → profit_function n x ≤ 160) :=
by sorry

end NUMINAMATH_CALUDE_desk_lamp_profit_l2846_284620


namespace NUMINAMATH_CALUDE_average_height_l2846_284683

def heights_problem (h₁ h₂ h₃ h₄ : ℝ) : Prop :=
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ ∧
  h₂ - h₁ = 2 ∧
  h₃ - h₂ = 2 ∧
  h₄ - h₃ = 6 ∧
  h₄ = 83

theorem average_height (h₁ h₂ h₃ h₄ : ℝ) 
  (hproblem : heights_problem h₁ h₂ h₃ h₄) : 
  (h₁ + h₂ + h₃ + h₄) / 4 = 77 := by
  sorry

end NUMINAMATH_CALUDE_average_height_l2846_284683


namespace NUMINAMATH_CALUDE_square_area_l2846_284666

theorem square_area (x : ℝ) : 
  (5 * x - 10 = 3 * (x + 4)) → 
  (5 * x - 10)^2 = 2025 := by
  sorry

end NUMINAMATH_CALUDE_square_area_l2846_284666


namespace NUMINAMATH_CALUDE_repeating_decimal_eq_l2846_284652

/-- The repeating decimal 0.565656... expressed as a rational number -/
def repeating_decimal : ℚ := 56 / 99

/-- The theorem stating that the repeating decimal 0.565656... equals 56/99 -/
theorem repeating_decimal_eq : repeating_decimal = 56 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_eq_l2846_284652


namespace NUMINAMATH_CALUDE_problem_solution_l2846_284689

theorem problem_solution (x : ℝ) : (0.20 * x = 0.15 * 1500 - 15) → x = 1050 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2846_284689


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l2846_284659

/-- A line is tangent to a circle if and only if the discriminant of the resulting quadratic equation is zero -/
theorem line_tangent_to_circle (a b : ℤ) : 
  (∃ x y : ℝ, y - 1 = (4 - a*x - b) / b ∧ 
              b^2*(x-1)^2 + (a*x+b-4)^2 - b^2 = 0 ∧ 
              (a*b - 4*a - b^2)^2 = (a^2 + b^2)*(b - 4)^2) ↔ 
  ((a = 12 ∧ b = 5) ∨ (a = -4 ∧ b = 3) ∨ (a = 8 ∧ b = 6) ∨ 
   (a = 0 ∧ b = 2) ∨ (a = 6 ∧ b = 8) ∨ (a = 2 ∧ b = 0) ∨ 
   (a = 5 ∧ b = 12) ∨ (a = 3 ∧ b = -4)) :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l2846_284659


namespace NUMINAMATH_CALUDE_car_journey_distance_l2846_284633

/-- Represents the car's journey with given speeds and break times -/
structure CarJourney where
  initial_speed : ℝ
  initial_duration : ℝ
  second_speed : ℝ
  second_duration : ℝ
  final_speed : ℝ
  final_duration : ℝ

/-- Calculates the total distance covered by the car -/
def total_distance (journey : CarJourney) : ℝ :=
  journey.initial_speed * journey.initial_duration +
  journey.second_speed * journey.second_duration +
  journey.final_speed * journey.final_duration

/-- Theorem stating that the car's journey covers 390 miles -/
theorem car_journey_distance :
  let journey : CarJourney := {
    initial_speed := 65,
    initial_duration := 2,
    second_speed := 60,
    second_duration := 2.5,
    final_speed := 55,
    final_duration := 2
  }
  total_distance journey = 390 := by sorry

end NUMINAMATH_CALUDE_car_journey_distance_l2846_284633


namespace NUMINAMATH_CALUDE_female_emu_ratio_is_half_l2846_284675

/-- Represents the emu farm setup and egg production --/
structure EmuFarm where
  num_pens : ℕ
  emus_per_pen : ℕ
  eggs_per_week : ℕ

/-- Calculates the ratio of female emus to total emus --/
def female_emu_ratio (farm : EmuFarm) : ℚ :=
  let total_emus := farm.num_pens * farm.emus_per_pen
  let eggs_per_day := farm.eggs_per_week / 7
  eggs_per_day / total_emus

/-- Theorem stating that the ratio of female emus to total emus is 1/2 --/
theorem female_emu_ratio_is_half (farm : EmuFarm) 
    (h1 : farm.num_pens = 4)
    (h2 : farm.emus_per_pen = 6)
    (h3 : farm.eggs_per_week = 84) : 
  female_emu_ratio farm = 1/2 := by
  sorry

#eval female_emu_ratio ⟨4, 6, 84⟩

end NUMINAMATH_CALUDE_female_emu_ratio_is_half_l2846_284675


namespace NUMINAMATH_CALUDE_cat_mouse_positions_after_196_moves_l2846_284665

/-- Represents the four squares in the grid --/
inductive Square
| TopLeft
| TopRight
| BottomLeft
| BottomRight

/-- Represents the eight outer segments of the squares --/
inductive Segment
| TopLeft
| TopMiddle
| TopRight
| RightMiddle
| BottomRight
| BottomMiddle
| BottomLeft
| LeftMiddle

/-- The cat's position after a given number of moves --/
def catPosition (moves : Nat) : Square :=
  match moves % 4 with
  | 0 => Square.TopLeft
  | 1 => Square.BottomLeft
  | 2 => Square.BottomRight
  | 3 => Square.TopRight
  | _ => Square.TopLeft  -- This case is unreachable, but needed for exhaustiveness

/-- The mouse's position after a given number of moves --/
def mousePosition (moves : Nat) : Segment :=
  match moves % 8 with
  | 0 => Segment.TopMiddle
  | 1 => Segment.TopRight
  | 2 => Segment.RightMiddle
  | 3 => Segment.BottomRight
  | 4 => Segment.BottomMiddle
  | 5 => Segment.BottomLeft
  | 6 => Segment.LeftMiddle
  | 7 => Segment.TopLeft
  | _ => Segment.TopMiddle  -- This case is unreachable, but needed for exhaustiveness

theorem cat_mouse_positions_after_196_moves :
  catPosition 196 = Square.TopLeft ∧ mousePosition 196 = Segment.BottomMiddle := by
  sorry


end NUMINAMATH_CALUDE_cat_mouse_positions_after_196_moves_l2846_284665


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l2846_284680

/-- The product of the coordinates of the midpoint of a segment with endpoints (8, -4) and (-2, 10) is 9. -/
theorem midpoint_coordinate_product : 
  let x1 : ℝ := 8
  let y1 : ℝ := -4
  let x2 : ℝ := -2
  let y2 : ℝ := 10
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x * midpoint_y = 9 := by sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l2846_284680


namespace NUMINAMATH_CALUDE_reflect_M_across_y_axis_l2846_284658

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflectAcrossYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- The theorem stating that reflecting M(3,2) across the y-axis results in (-3,2) -/
theorem reflect_M_across_y_axis :
  let M : Point := { x := 3, y := 2 }
  reflectAcrossYAxis M = { x := -3, y := 2 } := by
  sorry

end NUMINAMATH_CALUDE_reflect_M_across_y_axis_l2846_284658


namespace NUMINAMATH_CALUDE_product_of_one_plus_tans_l2846_284691

theorem product_of_one_plus_tans : (1 + Real.tan (23 * π / 180)) * (1 + Real.tan (22 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_one_plus_tans_l2846_284691


namespace NUMINAMATH_CALUDE_sum_product_theorem_l2846_284615

def number_list : List ℕ := [2, 3, 4, 6]

theorem sum_product_theorem :
  ∃! (subset : Finset ℕ),
    subset.card = 3 ∧ 
    (∀ x ∈ subset, x ∈ number_list) ∧
    (subset.sum id = 11) ∧
    (subset.prod id = 36) :=
sorry

end NUMINAMATH_CALUDE_sum_product_theorem_l2846_284615


namespace NUMINAMATH_CALUDE_arithmetic_sequence_constant_ratio_l2846_284671

/-- Sum of first n terms of an arithmetic sequence -/
def S (a : ℚ) (n : ℕ) : ℚ := n * (2 * a + (n - 1) * 5) / 2

/-- Theorem: If the ratio S_{4n}/S_n is constant for all positive n,
    then the first term of the sequence is 5/2 -/
theorem arithmetic_sequence_constant_ratio
  (h : ∃ (c : ℚ), ∀ (n : ℕ), n > 0 → S a (4*n) / S a n = c) :
  a = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_constant_ratio_l2846_284671


namespace NUMINAMATH_CALUDE_equation_solution_l2846_284638

theorem equation_solution : ∃ X : ℝ,
  (15.2 * 0.25 - 48.51 / 14.7) / X =
  ((13/44 - 2/11 - 5/66 / (5/2)) * (6/5)) / (3.2 + 0.8 * (5.5 - 3.25)) ∧
  X = 137.5 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2846_284638


namespace NUMINAMATH_CALUDE_consecutive_squares_sum_l2846_284616

theorem consecutive_squares_sum (n : ℕ) (h : n = 26) :
  (n - 1)^2 + n^2 + (n + 1)^2 = 2030 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_sum_l2846_284616


namespace NUMINAMATH_CALUDE_simplify_fraction_l2846_284654

theorem simplify_fraction :
  (5 : ℝ) / (Real.sqrt 75 + 3 * Real.sqrt 48 + Real.sqrt 147) = 5 * Real.sqrt 3 / 72 := by
sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2846_284654


namespace NUMINAMATH_CALUDE_S_is_infinite_l2846_284639

/-- A point in the xy-plane with rational coordinates -/
structure RationalPoint where
  x : ℚ
  y : ℚ

/-- The set of points satisfying the given conditions -/
def S : Set RationalPoint :=
  {p : RationalPoint | p.x > 0 ∧ p.y > 0 ∧ p.x * p.y ≤ 12}

/-- Theorem stating that the set S is infinite -/
theorem S_is_infinite : Set.Infinite S := by
  sorry

end NUMINAMATH_CALUDE_S_is_infinite_l2846_284639


namespace NUMINAMATH_CALUDE_soccer_challenge_kicks_l2846_284601

/-- The number of penalty kicks needed for a soccer team challenge --/
def penalty_kicks (total_players : ℕ) (goalies : ℕ) : ℕ :=
  (total_players - 1) * goalies

theorem soccer_challenge_kicks :
  penalty_kicks 25 5 = 120 :=
by sorry

end NUMINAMATH_CALUDE_soccer_challenge_kicks_l2846_284601


namespace NUMINAMATH_CALUDE_moe_has_least_money_l2846_284696

-- Define the set of people
inductive Person : Type
  | Bo | Coe | Flo | Jo | Moe | Zoe

-- Define a function to represent the amount of money each person has
variable (money : Person → ℕ)

-- Define the conditions
axiom different_amounts : ∀ (p q : Person), p ≠ q → money p ≠ money q

axiom flo_more_than_jo_bo : money Person.Flo > money Person.Jo ∧ money Person.Flo > money Person.Bo

axiom bo_coe_more_than_moe_less_than_zoe : 
  money Person.Bo > money Person.Moe ∧ 
  money Person.Coe > money Person.Moe ∧
  money Person.Zoe > money Person.Bo ∧
  money Person.Zoe > money Person.Coe

axiom jo_more_than_moe_zoe_less_than_bo : 
  money Person.Jo > money Person.Moe ∧
  money Person.Jo > money Person.Zoe ∧
  money Person.Bo > money Person.Jo

-- Theorem to prove
theorem moe_has_least_money : 
  ∀ (p : Person), p ≠ Person.Moe → money Person.Moe < money p :=
sorry

end NUMINAMATH_CALUDE_moe_has_least_money_l2846_284696


namespace NUMINAMATH_CALUDE_marble_arrangement_count_l2846_284679

/-- The number of ways to arrange 7 red marbles and n blue marbles in a row,
    where n is the maximum number of blue marbles that can be arranged such that
    the number of adjacent same-color pairs equals the number of adjacent different-color pairs -/
def M : ℕ := sorry

/-- The maximum number of blue marbles that can be arranged with 7 red marbles
    such that the number of adjacent same-color pairs equals the number of adjacent different-color pairs -/
def n : ℕ := sorry

/-- The theorem stating that M modulo 1000 equals 716 -/
theorem marble_arrangement_count : M % 1000 = 716 := by sorry

end NUMINAMATH_CALUDE_marble_arrangement_count_l2846_284679


namespace NUMINAMATH_CALUDE_bug_return_probability_l2846_284687

/-- Probability of returning to the starting vertex after n steps -/
def Q (n : ℕ) : ℚ :=
  if n = 0 then 1
  else (1/4 : ℚ) + (1/2 : ℚ) * Q (n-1)

/-- Regular tetrahedron with bug movement rules -/
theorem bug_return_probability :
  Q 6 = 354/729 := by sorry

end NUMINAMATH_CALUDE_bug_return_probability_l2846_284687
