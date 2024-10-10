import Mathlib

namespace common_tangents_possible_values_l1304_130492

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- The number of common tangent lines between two circles -/
def num_common_tangents (c1 c2 : Circle) : ℕ := sorry

/-- Theorem stating the possible values for the number of common tangents -/
theorem common_tangents_possible_values (c1 c2 : Circle) (h : c1 ≠ c2) :
  ∃ n : ℕ, num_common_tangents c1 c2 = n ∧ n ∈ ({0, 1, 2, 3, 4} : Set ℕ) := by sorry

end common_tangents_possible_values_l1304_130492


namespace students_taking_none_in_high_school_l1304_130455

/-- The number of students taking neither music, nor art, nor science in a high school -/
def students_taking_none (total : ℕ) (music art science : ℕ) (music_and_art music_and_science art_and_science : ℕ) (all_three : ℕ) : ℕ :=
  total - (music + art + science - music_and_art - music_and_science - art_and_science + all_three)

/-- Theorem stating the number of students taking neither music, nor art, nor science -/
theorem students_taking_none_in_high_school :
  students_taking_none 800 80 60 50 30 25 20 15 = 670 := by
  sorry

end students_taking_none_in_high_school_l1304_130455


namespace correct_mean_calculation_l1304_130406

theorem correct_mean_calculation (n : ℕ) (incorrect_mean : ℚ) 
  (correct_values wrong_values : List ℚ) :
  n = 30 ∧ 
  incorrect_mean = 170 ∧
  correct_values = [190, 200, 175] ∧
  wrong_values = [150, 195, 160] →
  (n * incorrect_mean - wrong_values.sum + correct_values.sum) / n = 172 :=
by sorry

end correct_mean_calculation_l1304_130406


namespace one_third_of_seven_times_nine_l1304_130471

-- Define the problem
theorem one_third_of_seven_times_nine : (1 / 3 : ℚ) * (7 * 9) = 21 := by
  sorry

end one_third_of_seven_times_nine_l1304_130471


namespace exam_average_l1304_130410

theorem exam_average (students_group1 : ℕ) (average_group1 : ℚ) 
                      (students_group2 : ℕ) (average_group2 : ℚ) : 
  students_group1 = 15 →
  average_group1 = 73 / 100 →
  students_group2 = 10 →
  average_group2 = 88 / 100 →
  let total_students := students_group1 + students_group2
  let total_score := students_group1 * average_group1 + students_group2 * average_group2
  let overall_average := total_score / total_students
  overall_average = 79 / 100 := by
  sorry

end exam_average_l1304_130410


namespace paper_tearing_impossibility_l1304_130409

theorem paper_tearing_impossibility : ¬ ∃ (n : ℕ), 1 + 3 * n = 2007 := by
  sorry

end paper_tearing_impossibility_l1304_130409


namespace number_difference_equation_l1304_130423

theorem number_difference_equation (x : ℝ) : 
  0.62 * x - 0.20 * 250 = 43 → x = 150 := by
  sorry

end number_difference_equation_l1304_130423


namespace pyarelals_loss_is_1800_l1304_130469

/-- Represents the loss incurred by Pyarelal in a business partnership with Ashok -/
def pyarelals_loss (pyarelals_capital : ℚ) (total_loss : ℚ) : ℚ :=
  (pyarelals_capital / (pyarelals_capital + pyarelals_capital / 9)) * total_loss

/-- Theorem stating that Pyarelal's loss is 1800 given the conditions of the problem -/
theorem pyarelals_loss_is_1800 (pyarelals_capital : ℚ) (h : pyarelals_capital > 0) :
  pyarelals_loss pyarelals_capital 2000 = 1800 := by
  sorry

end pyarelals_loss_is_1800_l1304_130469


namespace rectangle_width_l1304_130466

/-- Given a rectangle with perimeter 6a + 4b and length 2a + b, prove its width is a + b -/
theorem rectangle_width (a b : ℝ) : 
  let perimeter := 6*a + 4*b
  let length := 2*a + b
  let width := (perimeter / 2) - length
  width = a + b := by
sorry

end rectangle_width_l1304_130466


namespace point_A_on_line_l_l1304_130495

/-- A line passing through the origin with slope -2 -/
def line_l (x y : ℝ) : Prop := y = -2 * x

/-- The point (1, -2) -/
def point_A : ℝ × ℝ := (1, -2)

/-- Theorem: The point (1, -2) lies on the line l -/
theorem point_A_on_line_l : line_l point_A.1 point_A.2 := by sorry

end point_A_on_line_l_l1304_130495


namespace equal_time_per_style_l1304_130459

-- Define the swimming styles
inductive SwimmingStyle
| FrontCrawl
| Breaststroke
| Backstroke
| Butterfly

-- Define the problem parameters
def totalDistance : ℝ := 600
def totalTime : ℝ := 15
def numStyles : ℕ := 4

-- Define the speed for each style (yards per minute)
def speed (style : SwimmingStyle) : ℝ :=
  match style with
  | SwimmingStyle.FrontCrawl => 45
  | SwimmingStyle.Breaststroke => 35
  | SwimmingStyle.Backstroke => 40
  | SwimmingStyle.Butterfly => 30

-- Theorem to prove
theorem equal_time_per_style :
  ∀ (style : SwimmingStyle),
  (totalTime / numStyles : ℝ) = 3.75 ∧
  (totalDistance / numStyles : ℝ) / speed style ≤ totalTime / numStyles :=
by sorry

end equal_time_per_style_l1304_130459


namespace expression_simplification_l1304_130499

theorem expression_simplification (x y : ℝ) (hx : x = -1) (hy : y = 3) :
  5 * (3 * x^2 * y - 2 * x * y^2) - 2 * (3 * x^2 * y - 5 * x * y^2) = 27 := by
  sorry

end expression_simplification_l1304_130499


namespace jerry_weller_votes_l1304_130405

theorem jerry_weller_votes 
  (total_votes : ℕ) 
  (vote_difference : ℕ) 
  (h1 : total_votes = 196554)
  (h2 : vote_difference = 20196) :
  ∃ (jerry_votes john_votes : ℕ),
    jerry_votes + john_votes = total_votes ∧
    jerry_votes = john_votes + vote_difference ∧
    jerry_votes = 108375 := by
sorry

end jerry_weller_votes_l1304_130405


namespace jerrys_cartridge_cost_l1304_130436

/-- The total cost of printer cartridges for Jerry -/
def total_cost (color_cartridge_cost : ℕ) (bw_cartridge_cost : ℕ) (color_cartridge_count : ℕ) (bw_cartridge_count : ℕ) : ℕ :=
  color_cartridge_cost * color_cartridge_count + bw_cartridge_cost * bw_cartridge_count

/-- Theorem: Jerry's total cost for printer cartridges is $123 -/
theorem jerrys_cartridge_cost :
  total_cost 32 27 3 1 = 123 := by
  sorry

end jerrys_cartridge_cost_l1304_130436


namespace square_root_squared_specific_square_root_squared_l1304_130416

theorem square_root_squared (n : ℝ) (h : 0 ≤ n) : (Real.sqrt n) ^ 2 = n := by sorry

theorem specific_square_root_squared : (Real.sqrt 978121) ^ 2 = 978121 := by
  apply square_root_squared
  norm_num


end square_root_squared_specific_square_root_squared_l1304_130416


namespace huangshan_temperature_difference_l1304_130438

def temperature_difference (lowest highest : ℤ) : ℤ :=
  highest - lowest

theorem huangshan_temperature_difference :
  let lowest : ℤ := -13
  let highest : ℤ := 11
  temperature_difference lowest highest = 24 := by
  sorry

end huangshan_temperature_difference_l1304_130438


namespace unacceptable_weight_l1304_130427

def acceptable_range (x : ℝ) : Prop := 49.7 ≤ x ∧ x ≤ 50.3

theorem unacceptable_weight : ¬(acceptable_range 49.6) := by
  sorry

end unacceptable_weight_l1304_130427


namespace knights_selection_l1304_130425

/-- The number of ways to select k non-adjacent elements from n elements in a circular arrangement -/
def circularNonAdjacentSelection (n k : ℕ) : ℕ :=
  Nat.choose (n - k + 1) k - Nat.choose (n - k - 1) (k - 2)

/-- The problem statement -/
theorem knights_selection :
  circularNonAdjacentSelection 50 15 = 463991880 := by
  sorry

end knights_selection_l1304_130425


namespace rectangle_width_l1304_130443

theorem rectangle_width (area : ℝ) (perimeter : ℝ) (width : ℝ) (length : ℝ) :
  area = 50 →
  perimeter = 30 →
  area = length * width →
  perimeter = 2 * (length + width) →
  width = 5 ∨ width = 10 :=
by
  sorry

end rectangle_width_l1304_130443


namespace periodic_function_value_l1304_130456

def periodic_function (f : ℝ → ℝ) (period : ℝ) : Prop :=
  ∀ x : ℝ, f (x + period) = f x

theorem periodic_function_value 
  (f : ℝ → ℝ) 
  (h_periodic : periodic_function f (π / 2))
  (h_value : f (π / 3) = 1) : 
  f (17 * π / 6) = 1 := by
  sorry

end periodic_function_value_l1304_130456


namespace simplify_expression_l1304_130452

theorem simplify_expression (x : ℝ) : 3*x + 6*x + 9*x + 12 + 15*x + 18 = 33*x + 30 := by
  sorry

end simplify_expression_l1304_130452


namespace exactly_two_approvals_probability_l1304_130444

/-- The probability of success in a single trial -/
def p : ℝ := 0.6

/-- The number of trials -/
def n : ℕ := 5

/-- The number of desired successes -/
def k : ℕ := 2

/-- The probability of exactly k successes in n independent trials with probability p -/
def binomial_probability (p : ℝ) (n k : ℕ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

theorem exactly_two_approvals_probability :
  binomial_probability p n k = 0.3648 := by
  sorry

end exactly_two_approvals_probability_l1304_130444


namespace function_range_function_range_with_condition_l1304_130403

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x - 1/2 * Real.cos (2*x) + a - 3/a + 1/2

theorem function_range (a : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, f a x ≤ 0) → a ∈ Set.Ioo 0 1 :=
by sorry

theorem function_range_with_condition (a : ℝ) (h : a ≠ 0) :
  a ≥ 2 → (∃ x : ℝ, f a x ≤ 0) → a ∈ Set.Icc 2 3 :=
by sorry

end function_range_function_range_with_condition_l1304_130403


namespace sum_of_coefficients_l1304_130441

theorem sum_of_coefficients (x : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (1 - 2*x)^5 = a₀ + a₁*(1+x) + a₂*(1+x)^2 + a₃*(1+x)^3 + a₄*(1+x)^4 + a₅*(1+x)^5 →
  a₀ + a₁ + a₂ + a₃ + a₄ = 33 := by
sorry

end sum_of_coefficients_l1304_130441


namespace numbers_close_to_zero_not_set_l1304_130437

-- Define the property of being "close to 0"
def CloseToZero (x : ℝ) : Prop := sorry

-- Define the criteria for set formation
structure SetCriteria :=
  (definiteness : Prop)
  (distinctness : Prop)
  (unorderedness : Prop)

-- Define a function to check if a collection satisfies set criteria
def SatisfiesSetCriteria (S : Set ℝ) (criteria : SetCriteria) : Prop := sorry

-- Theorem stating that "numbers close to 0" cannot form a set
theorem numbers_close_to_zero_not_set :
  ¬ ∃ (S : Set ℝ) (criteria : SetCriteria), 
    (∀ x ∈ S, CloseToZero x) ∧ 
    SatisfiesSetCriteria S criteria :=
sorry

end numbers_close_to_zero_not_set_l1304_130437


namespace stream_speed_proof_l1304_130467

/-- Proves that the speed of the stream is 21 kmph given the conditions of the rowing problem. -/
theorem stream_speed_proof (boat_speed : ℝ) (stream_speed : ℝ) : 
  boat_speed = 63 →
  (1 / (boat_speed - stream_speed)) = (2 / (boat_speed + stream_speed)) →
  stream_speed = 21 := by
sorry

end stream_speed_proof_l1304_130467


namespace geometric_sequence_2010th_term_l1304_130482

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  p : ℝ
  q : ℝ
  first_term : ℝ
  second_term : ℝ
  third_term : ℝ
  fourth_term : ℝ
  h1 : first_term = p
  h2 : second_term = 9
  h3 : third_term = 3 * p / q
  h4 : fourth_term = 3 * p * q

/-- The 2010th term of the geometric sequence is 9 -/
theorem geometric_sequence_2010th_term (seq : GeometricSequence) :
  let r := seq.second_term / seq.first_term
  seq.first_term * r^(2009 : ℕ) = 9 := by sorry

end geometric_sequence_2010th_term_l1304_130482


namespace max_value_of_x_plus_inverse_l1304_130487

theorem max_value_of_x_plus_inverse (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  (x + 1/x) ≤ Real.sqrt 15 ∧ ∃ y : ℝ, y + 1/y = Real.sqrt 15 ∧ 13 = y^2 + 1/y^2 :=
sorry

end max_value_of_x_plus_inverse_l1304_130487


namespace variance_best_stability_measure_l1304_130468

/-- A measure of stability for a set of test scores -/
class StabilityMeasure where
  measure : List ℝ → ℝ

/-- Average as a stability measure -/
def average : StabilityMeasure := sorry

/-- Median as a stability measure -/
def median : StabilityMeasure := sorry

/-- Variance as a stability measure -/
def variance : StabilityMeasure := sorry

/-- Mode as a stability measure -/
def mode : StabilityMeasure := sorry

/-- A function that determines if a stability measure is the best for test scores -/
def isBestStabilityMeasure (m : StabilityMeasure) : Prop := sorry

theorem variance_best_stability_measure : isBestStabilityMeasure variance := by
  sorry

end variance_best_stability_measure_l1304_130468


namespace onion_weight_problem_l1304_130420

theorem onion_weight_problem (total_weight : Real) (total_count : Nat) (removed_count : Nat) (removed_avg : Real) (remaining_count : Nat) :
  total_weight = 7.68 →
  total_count = 40 →
  removed_count = 5 →
  removed_avg = 0.206 →
  remaining_count = total_count - removed_count →
  let remaining_weight := total_weight - (removed_count * removed_avg)
  let remaining_avg := remaining_weight / remaining_count
  remaining_avg = 0.190 := by
sorry

end onion_weight_problem_l1304_130420


namespace smallest_five_digit_palindrome_div_by_6_l1304_130457

/-- A function that checks if a number is a five-digit palindrome -/
def is_five_digit_palindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ 
  (n / 10000 = n % 10) ∧ 
  ((n / 1000) % 10 = (n / 10) % 10)

/-- The theorem stating that 20002 is the smallest five-digit palindrome divisible by 6 -/
theorem smallest_five_digit_palindrome_div_by_6 :
  ∀ n : ℕ, is_five_digit_palindrome n ∧ n % 6 = 0 → n ≥ 20002 :=
sorry

end smallest_five_digit_palindrome_div_by_6_l1304_130457


namespace work_duration_problem_l1304_130404

/-- The problem of determining how long a worker worked on a task before another worker finished it. -/
theorem work_duration_problem 
  (W : ℝ) -- Total work
  (x_rate : ℝ) -- x's work rate per day
  (y_rate : ℝ) -- y's work rate per day
  (y_finish_time : ℝ) -- Time y took to finish the remaining work
  (hx : x_rate = W / 40) -- x's work rate condition
  (hy : y_rate = W / 20) -- y's work rate condition
  (h_finish : y_finish_time = 16) -- y's finish time condition
  : ∃ (d : ℝ), d * x_rate + y_finish_time * y_rate = W ∧ d = 8 := by
  sorry

end work_duration_problem_l1304_130404


namespace no_primes_in_range_l1304_130489

theorem no_primes_in_range (n : ℕ) (h : n > 2) :
  ∀ k ∈ Set.Icc (n! + 3) (n! + 2*n), ¬ Nat.Prime k := by
  sorry

end no_primes_in_range_l1304_130489


namespace negation_of_exponential_inequality_l1304_130483

theorem negation_of_exponential_inequality :
  (¬ (∀ x : ℝ, Real.exp x ≥ 1)) ↔ (∃ x : ℝ, Real.exp x < 1) := by sorry

end negation_of_exponential_inequality_l1304_130483


namespace product_with_zero_is_zero_l1304_130445

theorem product_with_zero_is_zero :
  (-2.5) * 0.37 * 1.25 * (-4) * (-8) * 0 = 0 := by
  sorry

end product_with_zero_is_zero_l1304_130445


namespace mother_age_twice_alex_l1304_130414

/-- Alex's birth year -/
def alexBirthYear : ℕ := 2000

/-- The year when Alex's mother's age was five times his age -/
def referenceYear : ℕ := 2010

/-- Alex's mother's age is five times Alex's age in the reference year -/
axiom mother_age_five_times (y : ℕ) : y - alexBirthYear = 10 → y = referenceYear → 
  ∃ (motherAge : ℕ), motherAge = 5 * (y - alexBirthYear)

/-- The year when Alex's mother's age will be twice his age -/
def targetYear : ℕ := 2040

theorem mother_age_twice_alex :
  ∃ (motherAge alexAge : ℕ),
    motherAge = 2 * alexAge ∧
    alexAge = targetYear - alexBirthYear ∧
    motherAge = (referenceYear - alexBirthYear) * 5 + (targetYear - referenceYear) :=
sorry

end mother_age_twice_alex_l1304_130414


namespace tip_fraction_is_55_93_l1304_130402

/-- Represents the waiter's salary structure over four weeks -/
structure WaiterSalary where
  base : ℚ  -- Base salary
  tips1 : ℚ := 5/3 * base  -- Tips in week 1
  tips2 : ℚ := 3/2 * base  -- Tips in week 2
  tips3 : ℚ := base        -- Tips in week 3
  tips4 : ℚ := 4/3 * base  -- Tips in week 4
  expenses : ℚ := 2/5 * base  -- Total expenses over 4 weeks (10% per week)

/-- Calculates the fraction of total income after expenses that came from tips -/
def tipFraction (s : WaiterSalary) : ℚ :=
  let totalTips := s.tips1 + s.tips2 + s.tips3 + s.tips4
  let totalIncome := 4 * s.base + totalTips
  let incomeAfterExpenses := totalIncome - s.expenses
  totalTips / incomeAfterExpenses

/-- Theorem stating that the fraction of total income after expenses that came from tips is 55/93 -/
theorem tip_fraction_is_55_93 (s : WaiterSalary) : tipFraction s = 55/93 := by
  sorry


end tip_fraction_is_55_93_l1304_130402


namespace pens_sold_is_226_l1304_130400

/-- Represents the profit and cost structure of a store promotion -/
structure StorePromotion where
  penProfit : ℕ        -- Profit from selling one pen (in yuan)
  bearCost : ℕ         -- Cost of one teddy bear (in yuan)
  pensPerBundle : ℕ    -- Number of pens in a promotion bundle
  totalProfit : ℕ      -- Total profit from the promotion (in yuan)

/-- Calculates the number of pens sold during a store promotion -/
def pensSold (promo : StorePromotion) : ℕ :=
  -- Implementation details are omitted as per instructions
  sorry

/-- Theorem stating that the number of pens sold is 226 for the given promotion -/
theorem pens_sold_is_226 (promo : StorePromotion) 
  (h1 : promo.penProfit = 9)
  (h2 : promo.bearCost = 2)
  (h3 : promo.pensPerBundle = 4)
  (h4 : promo.totalProfit = 1922) : 
  pensSold promo = 226 := by
  sorry

end pens_sold_is_226_l1304_130400


namespace roller_coaster_problem_l1304_130401

/-- The number of times a roller coaster must run to accommodate all people in line -/
def roller_coaster_runs (people_in_line : ℕ) (cars : ℕ) (people_per_car : ℕ) : ℕ :=
  (people_in_line + cars * people_per_car - 1) / (cars * people_per_car)

/-- Theorem stating that for 84 people in line, 7 cars, and 2 people per car, 6 runs are needed -/
theorem roller_coaster_problem : roller_coaster_runs 84 7 2 = 6 := by
  sorry

end roller_coaster_problem_l1304_130401


namespace pizza_theorem_l1304_130442

def pizza_problem (total_slices : ℕ) (slices_per_person : ℕ) : ℕ :=
  (total_slices / slices_per_person) - 1

theorem pizza_theorem :
  pizza_problem 12 4 = 2 := by
  sorry

end pizza_theorem_l1304_130442


namespace average_of_quadratic_solutions_l1304_130430

theorem average_of_quadratic_solutions (x₁ x₂ : ℝ) : 
  (3 * x₁^2 - 4 * x₁ + 1 = 0) → 
  (3 * x₂^2 - 4 * x₂ + 1 = 0) → 
  x₁ ≠ x₂ → 
  (x₁ + x₂) / 2 = 2/3 := by
sorry

end average_of_quadratic_solutions_l1304_130430


namespace parallel_lines_interior_alternate_angles_l1304_130432

/-- Two lines are parallel -/
def parallel (l1 l2 : Line) : Prop := sorry

/-- A line intersects two other lines -/
def intersects (l : Line) (l1 l2 : Line) : Prop := sorry

/-- Interior alternate angles between two lines and a transversal -/
def interior_alternate_angles (l1 l2 l : Line) (α β : Angle) : Prop := sorry

/-- The proposition about parallel lines and interior alternate angles -/
theorem parallel_lines_interior_alternate_angles 
  (l1 l2 l : Line) (α β : Angle) :
  parallel l1 l2 → 
  intersects l l1 l2 → 
  interior_alternate_angles l1 l2 l α β → 
  α = β := 
sorry

end parallel_lines_interior_alternate_angles_l1304_130432


namespace painted_rectangle_ratio_l1304_130486

/-- Given a rectangle with length 2s and width s, and a paint brush of width w,
    if half the area of the rectangle is painted when the brush is swept along both diagonals,
    then the ratio of the length of the rectangle to the brush width is 6. -/
theorem painted_rectangle_ratio (s w : ℝ) (h_pos_s : 0 < s) (h_pos_w : 0 < w) :
  w^2 + 2*(s-w)^2 = s^2 → (2*s) / w = 6 := by sorry

end painted_rectangle_ratio_l1304_130486


namespace independence_test_most_appropriate_l1304_130417

/-- Represents the survey data in a 2x2 contingency table --/
structure SurveyData where
  male_total : ℕ
  male_doping : ℕ
  female_total : ℕ
  female_framed : ℕ

/-- Represents different statistical methods --/
inductive StatMethod
  | MeanVariance
  | RegressionAnalysis
  | IndependenceTest
  | Probability

/-- Checks if a method is most appropriate for analyzing the given survey data --/
def is_most_appropriate (method : StatMethod) (data : SurveyData) : Prop :=
  method = StatMethod.IndependenceTest

/-- The main theorem stating that the Independence Test is the most appropriate method --/
theorem independence_test_most_appropriate (data : SurveyData) :
  is_most_appropriate StatMethod.IndependenceTest data :=
sorry

end independence_test_most_appropriate_l1304_130417


namespace divisibility_property_l1304_130485

theorem divisibility_property (n : ℕ) (hn : n > 1) : 
  ∃ k : ℤ, n^(n-1) - 1 = (n-1)^2 * k := by sorry

end divisibility_property_l1304_130485


namespace no_positive_integer_pairs_l1304_130498

theorem no_positive_integer_pairs : 
  ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x > y ∧ (x^2 : ℝ) + y^2 = x^4 := by
  sorry

end no_positive_integer_pairs_l1304_130498


namespace system_solution_l1304_130422

theorem system_solution (a b c x y z : ℝ) 
  (eq1 : x * (x + y + z) = a^2)
  (eq2 : y * (x + y + z) = b^2)
  (eq3 : z * (x + y + z) = c^2) :
  (x = a^2 / Real.sqrt (a^2 + b^2 + c^2) ∧ 
   y = b^2 / Real.sqrt (a^2 + b^2 + c^2) ∧ 
   z = c^2 / Real.sqrt (a^2 + b^2 + c^2)) ∨
  (x = -a^2 / Real.sqrt (a^2 + b^2 + c^2) ∧ 
   y = -b^2 / Real.sqrt (a^2 + b^2 + c^2) ∧ 
   z = -c^2 / Real.sqrt (a^2 + b^2 + c^2)) := by
  sorry

end system_solution_l1304_130422


namespace partition_S_l1304_130428

def S : Set ℚ := {-5/6, 0, -7/2, 6/5, 6}

theorem partition_S :
  (∃ (A B : Set ℚ), A ∪ B = S ∧ A ∩ B = ∅ ∧
    A = {x ∈ S | x < 0} ∧
    B = {x ∈ S | x ≥ 0} ∧
    A = {-5/6, -7/2} ∧
    B = {0, 6/5, 6}) :=
by sorry

end partition_S_l1304_130428


namespace brigade_plowing_rates_l1304_130431

/-- Represents the daily plowing rate and work duration of a brigade --/
structure Brigade where
  daily_rate : ℝ
  days_worked : ℝ

/-- Proves that given the problem conditions, the brigades' daily rates are 24 and 27 hectares --/
theorem brigade_plowing_rates 
  (first_brigade second_brigade : Brigade)
  (h1 : first_brigade.daily_rate * first_brigade.days_worked = 240)
  (h2 : second_brigade.daily_rate * second_brigade.days_worked = 240 * 1.35)
  (h3 : second_brigade.daily_rate = first_brigade.daily_rate + 3)
  (h4 : second_brigade.days_worked = first_brigade.days_worked + 2)
  (h5 : first_brigade.daily_rate > 20)
  (h6 : second_brigade.daily_rate > 20)
  : first_brigade.daily_rate = 24 ∧ second_brigade.daily_rate = 27 := by
  sorry

#check brigade_plowing_rates

end brigade_plowing_rates_l1304_130431


namespace tangent_when_zero_discriminant_l1304_130419

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the discriminant of a quadratic function -/
def discriminant (f : QuadraticFunction) : ℝ :=
  f.b^2 - 4 * f.a * f.c

/-- Determines if a quadratic function's graph is tangent to the x-axis -/
def is_tangent_to_x_axis (f : QuadraticFunction) : Prop :=
  ∃ x : ℝ, f.a * x^2 + f.b * x + f.c = 0 ∧
    ∀ y : ℝ, y ≠ x → f.a * y^2 + f.b * y + f.c > 0

/-- The main theorem: if the discriminant is zero, the graph is tangent to the x-axis -/
theorem tangent_when_zero_discriminant (k : ℝ) :
  let f : QuadraticFunction := ⟨3, 9, k⟩
  discriminant f = 0 → is_tangent_to_x_axis f :=
by sorry

end tangent_when_zero_discriminant_l1304_130419


namespace halloween_candy_problem_l1304_130472

/-- Given Katie's candy count, her sister's candy count, and the number of pieces eaten,
    calculate the remaining candy pieces. -/
theorem halloween_candy_problem (katie_candy : ℕ) (sister_candy : ℕ) (eaten_candy : ℕ) :
  katie_candy = 10 →
  sister_candy = 6 →
  eaten_candy = 9 →
  katie_candy + sister_candy - eaten_candy = 7 :=
by
  sorry

end halloween_candy_problem_l1304_130472


namespace quadratic_one_solution_sum_l1304_130418

theorem quadratic_one_solution_sum (a : ℝ) : 
  (∃ (a₁ a₂ : ℝ), 
    (∀ x : ℝ, 3 * x^2 + a₁ * x + 6 * x + 7 = 0 ↔ x = -((a₁ + 6) / 6)) ∧
    (∀ x : ℝ, 3 * x^2 + a₂ * x + 6 * x + 7 = 0 ↔ x = -((a₂ + 6) / 6)) ∧
    a₁ ≠ a₂ ∧ 
    (∀ a' : ℝ, a' ≠ a₁ ∧ a' ≠ a₂ → 
      ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 3 * x₁^2 + a' * x₁ + 6 * x₁ + 7 = 0 ∧ 
                     3 * x₂^2 + a' * x₂ + 6 * x₂ + 7 = 0)) →
  a₁ + a₂ = -12 :=
by sorry

end quadratic_one_solution_sum_l1304_130418


namespace problem_statement_l1304_130424

theorem problem_statement (x : ℝ) (Q : ℝ) (h : 5 * (3 * x + 4 * Real.sqrt 2) = Q) :
  10 * (6 * x + 8 * Real.sqrt 2 - Real.sqrt 2) = 4 * Q - 10 * Real.sqrt 2 := by
  sorry

end problem_statement_l1304_130424


namespace digit_sum_s_99_l1304_130411

/-- s(n) is the number formed by concatenating the first n perfect squares -/
def s (n : ℕ) : ℕ := sorry

/-- The digit sum of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Theorem: The digit sum of s(99) is 4 -/
theorem digit_sum_s_99 : digitSum (s 99) = 4 := by sorry

end digit_sum_s_99_l1304_130411


namespace min_value_of_y_l1304_130415

theorem min_value_of_y (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 → 1/x + 4/y ≥ 9/2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 2 ∧ 1/x + 4/y = 9/2) :=
by sorry

end min_value_of_y_l1304_130415


namespace percentage_relationships_l1304_130473

theorem percentage_relationships (a b c d e f g : ℝ) 
  (h1 : d = 0.22 * b) 
  (h2 : d = 0.35 * f) 
  (h3 : e = 0.27 * a) 
  (h4 : e = 0.60 * f) 
  (h5 : c = 0.14 * a) 
  (h6 : c = 0.40 * b) 
  (h7 : d = 2 * c) 
  (h8 : g = 3 * e) : 
  g = 0.81 * a ∧ b = 0.7 * a ∧ f = 0.45 * a := by
  sorry


end percentage_relationships_l1304_130473


namespace intersection_point_l1304_130449

/-- The x-coordinate of the intersection point of y = 2x - 1 and y = x + 1 -/
def x : ℝ := 2

/-- The y-coordinate of the intersection point of y = 2x - 1 and y = x + 1 -/
def y : ℝ := 3

/-- The first linear function -/
def f (x : ℝ) : ℝ := 2 * x - 1

/-- The second linear function -/
def g (x : ℝ) : ℝ := x + 1

theorem intersection_point :
  f x = y ∧ g x = y ∧ f x = g x :=
by sorry

end intersection_point_l1304_130449


namespace max_sum_with_length_constraint_l1304_130439

-- Define the length of an integer as the number of prime factors
def length (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem max_sum_with_length_constraint :
  ∀ x y : ℕ,
    x > 1 →
    y > 1 →
    length x + length y ≤ 16 →
    x + 3 * y ≤ 98306 :=
by sorry

end max_sum_with_length_constraint_l1304_130439


namespace first_four_seeds_l1304_130446

/-- Represents a row in the random number table -/
def RandomTableRow := List Nat

/-- The random number table -/
def randomTable : List RandomTableRow := [
  [78226, 85384, 40527, 48987, 60602, 16085, 29971, 61279],
  [43021, 92980, 27768, 26916, 27783, 84572, 78483, 39820],
  [61459, 39073, 79242, 20372, 21048, 87088, 34600, 74636],
  [63171, 58247, 12907, 50303, 28814, 40422, 97895, 61421],
  [42372, 53183, 51546, 90385, 12120, 64042, 51320, 22983]
]

/-- The starting position in the random number table -/
def startPosition : Nat × Nat := (2, 5)

/-- The total number of seeds -/
def totalSeeds : Nat := 850

/-- Function to get the next valid seed number -/
def getNextValidSeed (table : List RandomTableRow) (pos : Nat × Nat) (maxSeed : Nat) : Option (Nat × (Nat × Nat)) :=
  sorry

/-- Theorem stating that the first 4 valid seed numbers are 390, 737, 220, and 372 -/
theorem first_four_seeds :
  let seedNumbers := [390, 737, 220, 372]
  ∃ (pos1 pos2 pos3 pos4 : Nat × Nat),
    getNextValidSeed randomTable startPosition totalSeeds = some (seedNumbers[0], pos1) ∧
    getNextValidSeed randomTable pos1 totalSeeds = some (seedNumbers[1], pos2) ∧
    getNextValidSeed randomTable pos2 totalSeeds = some (seedNumbers[2], pos3) ∧
    getNextValidSeed randomTable pos3 totalSeeds = some (seedNumbers[3], pos4) :=
  sorry

end first_four_seeds_l1304_130446


namespace binomial_20_9_l1304_130497

theorem binomial_20_9 (h1 : Nat.choose 18 7 = 31824)
                      (h2 : Nat.choose 18 8 = 43758)
                      (h3 : Nat.choose 18 9 = 43758) :
  Nat.choose 20 9 = 163098 := by
  sorry

end binomial_20_9_l1304_130497


namespace geometric_sequence_sixth_term_l1304_130478

theorem geometric_sequence_sixth_term 
  (a : ℕ → ℝ) 
  (n : ℕ) 
  (h1 : n = 9) 
  (h2 : a 1 = 9) 
  (h3 : a n = 26244) 
  (h4 : ∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ n → a j / a i = a (i + 1) / a i) : 
  a 6 = 2187 := by
sorry

end geometric_sequence_sixth_term_l1304_130478


namespace factor_expression_l1304_130407

theorem factor_expression (x : ℝ) : 35 * x^13 + 245 * x^26 = 35 * x^13 * (1 + 7 * x^13) := by
  sorry

end factor_expression_l1304_130407


namespace geometric_sequence_ratio_l1304_130474

/-- Given a geometric sequence {a_n} with positive terms where a₁, (1/2)a₃, and 2a₂ form an arithmetic sequence,
    prove that (a₉ + a₁₀) / (a₇ + a₈) = 3 + 2√2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0)
  (h_geom : ∃ q : ℝ, ∀ n, a (n + 1) = q * a n)
  (h_arith : a 1 + 2 * a 2 = a 3) :
  (a 9 + a 10) / (a 7 + a 8) = 3 + 2 * Real.sqrt 2 := by
  sorry

end geometric_sequence_ratio_l1304_130474


namespace machine_work_time_l1304_130435

theorem machine_work_time (x : ℝ) : 
  (x > 0) →
  (1 / (x + 4) + 1 / (x + 2) + 1 / (2 * x + 2) = 1 / x) →
  x = 2 / 3 := by
  sorry

end machine_work_time_l1304_130435


namespace origami_distribution_l1304_130493

theorem origami_distribution (total_papers : ℝ) (num_cousins : ℝ) (papers_per_cousin : ℝ) : 
  total_papers = 48.0 →
  num_cousins = 6.0 →
  total_papers = num_cousins * papers_per_cousin →
  papers_per_cousin = 8.0 := by
  sorry

end origami_distribution_l1304_130493


namespace pentadecagon_diagonals_l1304_130448

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A pentadecagon is a polygon with 15 sides -/
def pentadecagon_sides : ℕ := 15

theorem pentadecagon_diagonals :
  num_diagonals pentadecagon_sides = 90 := by
  sorry

end pentadecagon_diagonals_l1304_130448


namespace gum_pieces_per_package_l1304_130494

theorem gum_pieces_per_package (total_packages : ℕ) (total_pieces : ℕ) 
  (h1 : total_packages = 9) 
  (h2 : total_pieces = 135) : 
  total_pieces / total_packages = 15 := by
  sorry

end gum_pieces_per_package_l1304_130494


namespace vertex_locus_is_partial_parabola_l1304_130496

/-- The locus of points (x_t, y_t) where x_t = -t / (t^2 + 1) and y_t = c - t^2 / (t^2 + 1),
    as t ranges over all real numbers, forms part, but not all, of a parabola. -/
theorem vertex_locus_is_partial_parabola (c : ℝ) (h : c > 0) :
  ∃ (a b d : ℝ), ∀ (t : ℝ),
    ∃ (x y : ℝ), x = -t / (t^2 + 1) ∧ y = c - t^2 / (t^2 + 1) ∧
    (y = a * x^2 + b * x + d ∨ y < a * x^2 + b * x + d) :=
sorry

end vertex_locus_is_partial_parabola_l1304_130496


namespace parrots_per_cage_l1304_130476

/-- Given a pet store with birds, calculate the number of parrots per cage. -/
theorem parrots_per_cage
  (num_cages : ℕ)
  (parakeets_per_cage : ℕ)
  (total_birds : ℕ)
  (h1 : num_cages = 6)
  (h2 : parakeets_per_cage = 7)
  (h3 : total_birds = 54) :
  (total_birds - num_cages * parakeets_per_cage) / num_cages = 2 :=
by sorry

end parrots_per_cage_l1304_130476


namespace midpoint_sum_equals_vertex_sum_l1304_130453

theorem midpoint_sum_equals_vertex_sum (a b : ℝ) 
  (h : a + b + (a + 5) = 15) : 
  (a + b) / 2 + (2 * a + 5) / 2 + (b + a + 5) / 2 = 15 := by
  sorry

end midpoint_sum_equals_vertex_sum_l1304_130453


namespace birds_in_tree_l1304_130440

theorem birds_in_tree (initial_birds final_birds : ℕ) 
  (h1 : initial_birds = 29) 
  (h2 : final_birds = 42) : 
  final_birds - initial_birds = 13 := by
  sorry

end birds_in_tree_l1304_130440


namespace student_number_calculation_l1304_130462

theorem student_number_calculation (x : ℕ) (h : x = 129) : 2 * x - 148 = 110 := by
  sorry

end student_number_calculation_l1304_130462


namespace union_A_B_intersection_A_complement_B_l1304_130454

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 - x - 6 ≤ 0}

-- Define set B
def B : Set ℝ := {x | x^2 - 3*x - 4 > 0}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {x | x ≤ 3 ∨ x > 4} := by sorry

-- Theorem for A ∩ (U \ B)
theorem intersection_A_complement_B : A ∩ (U \ B) = {x | -1 ≤ x ∧ x ≤ 3} := by sorry

end union_A_B_intersection_A_complement_B_l1304_130454


namespace parabola_coefficient_l1304_130451

/-- A quadratic function of the form y = mx^2 + 2 -/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := m * x^2 + 2

/-- The condition for a downward-opening parabola -/
def is_downward_opening (m : ℝ) : Prop := m < 0

theorem parabola_coefficient :
  ∀ m : ℝ, is_downward_opening m → m = -2 := by sorry

end parabola_coefficient_l1304_130451


namespace limit_at_infinity_limit_at_point_l1304_130433

-- Part 1
theorem limit_at_infinity (ε : ℝ) (hε : ε > 0) :
  ∃ M : ℝ, ∀ x : ℝ, x > M → |(2*x + 3)/(3*x) - 2/3| < ε :=
sorry

-- Part 2
theorem limit_at_point (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, 0 < |x - 3| ∧ |x - 3| < δ → |(2*x + 1) - 7| < ε :=
sorry

end limit_at_infinity_limit_at_point_l1304_130433


namespace polygonal_chain_circle_cover_l1304_130434

/-- A planar closed polygonal chain -/
structure ClosedPolygonalChain where
  vertices : Set (ℝ × ℝ)
  is_closed : True  -- This is a placeholder for the closure property
  perimeter : ℝ

/-- Theorem: For any closed polygonal chain with perimeter 1, 
    there exists a point such that all points on the chain 
    are within distance 1/4 from it -/
theorem polygonal_chain_circle_cover 
  (chain : ClosedPolygonalChain) 
  (h_perimeter : chain.perimeter = 1) :
  ∃ (center : ℝ × ℝ), ∀ (p : ℝ × ℝ), p ∈ chain.vertices → 
    Real.sqrt ((p.1 - center.1)^2 + (p.2 - center.2)^2) ≤ 1/4 := by
  sorry

end polygonal_chain_circle_cover_l1304_130434


namespace equation_solution_l1304_130480

theorem equation_solution (x : ℝ) : 
  (x / 3) / 5 = 5 / (x / 3) → x = 15 ∨ x = -15 := by
sorry

end equation_solution_l1304_130480


namespace square_difference_equality_l1304_130464

theorem square_difference_equality : 1013^2 - 991^2 - 1007^2 + 997^2 = 24048 := by
  sorry

end square_difference_equality_l1304_130464


namespace sqrt_equation_solution_l1304_130488

theorem sqrt_equation_solution (x : ℝ) :
  (x > 2) → (Real.sqrt (7 * x) / Real.sqrt (4 * (x - 2)) = 3) → (x = 72 / 29) := by
  sorry

end sqrt_equation_solution_l1304_130488


namespace line_x_axis_intersection_l1304_130460

/-- The line equation 5y - 6x = 15 intersects the x-axis at the point (-2.5, 0) -/
theorem line_x_axis_intersection :
  ∃! (x : ℝ), 5 * 0 - 6 * x = 15 ∧ x = -2.5 := by
  sorry

end line_x_axis_intersection_l1304_130460


namespace product_difference_square_equals_negative_one_l1304_130450

theorem product_difference_square_equals_negative_one :
  2021 * 2023 - 2022^2 = -1 := by
  sorry

end product_difference_square_equals_negative_one_l1304_130450


namespace square_side_length_average_l1304_130490

theorem square_side_length_average (a₁ a₂ a₃ a₄ : ℝ) 
  (h₁ : a₁ = 25) (h₂ : a₂ = 64) (h₃ : a₃ = 144) (h₄ : a₄ = 225) : 
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃ + Real.sqrt a₄) / 4 = 10 := by
sorry

end square_side_length_average_l1304_130490


namespace janous_inequality_l1304_130479

theorem janous_inequality (α x y z : ℝ) (hα : α > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x * y + y * z + z * x = α) :
  (1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) ≥ 16 * (x / z + z / x + 2) := by
  sorry

end janous_inequality_l1304_130479


namespace equation_equivalence_l1304_130484

theorem equation_equivalence (x y : ℝ) 
  (hx1 : x ≠ 0) (hx2 : x ≠ 4) (hy1 : y ≠ 0) (hy2 : y ≠ 6) :
  (2 / x + 3 / y = 1 / 2) ↔ (4 * y / (y - 6) = x) :=
by sorry

end equation_equivalence_l1304_130484


namespace two_green_marbles_probability_l1304_130463

/-- The probability of drawing two green marbles consecutively without replacement -/
theorem two_green_marbles_probability 
  (red green white blue : ℕ) 
  (h_red : red = 3)
  (h_green : green = 4)
  (h_white : white = 8)
  (h_blue : blue = 5) : 
  (green : ℚ) / (red + green + white + blue) * 
  ((green - 1) : ℚ) / (red + green + white + blue - 1) = 3 / 95 := by
sorry

end two_green_marbles_probability_l1304_130463


namespace odd_perfect_square_theorem_l1304_130426

/-- 
The sum of divisors function σ(n) is the sum of all positive divisors of n, including n itself.
-/
def sum_of_divisors (n : ℕ+) : ℕ := sorry

/-- 
A number is a perfect square if it is the product of an integer with itself.
-/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem odd_perfect_square_theorem (n : ℕ+) : 
  sum_of_divisors n = 2 * n.val + 1 → Odd n.val ∧ is_perfect_square n.val :=
sorry

end odd_perfect_square_theorem_l1304_130426


namespace fraction_five_times_seven_over_ten_l1304_130461

theorem fraction_five_times_seven_over_ten : (5 * 7) / 10 = 3.5 := by
  sorry

end fraction_five_times_seven_over_ten_l1304_130461


namespace original_number_proof_l1304_130470

theorem original_number_proof : ∃ x : ℝ, 16 * x = 3408 ∧ x = 213 := by
  sorry

end original_number_proof_l1304_130470


namespace solution_set_of_equations_l1304_130413

theorem solution_set_of_equations (x y z : ℝ) : 
  (3 * (x^2 + y^2 + z^2) = 1 ∧ 
   x^2*y^2 + y^2*z^2 + z^2*x^2 = x*y*z*(x+y+z)^2) ↔ 
  ((x = 0 ∧ y = 0 ∧ z = Real.sqrt 3 / 3) ∨
   (x = 0 ∧ y = 0 ∧ z = -Real.sqrt 3 / 3) ∨
   (x = 0 ∧ y = Real.sqrt 3 / 3 ∧ z = 0) ∨
   (x = 0 ∧ y = -Real.sqrt 3 / 3 ∧ z = 0) ∨
   (x = Real.sqrt 3 / 3 ∧ y = 0 ∧ z = 0) ∨
   (x = -Real.sqrt 3 / 3 ∧ y = 0 ∧ z = 0) ∨
   (x = 1/3 ∧ y = 1/3 ∧ z = 1/3) ∨
   (x = -1/3 ∧ y = -1/3 ∧ z = -1/3)) := by
sorry

end solution_set_of_equations_l1304_130413


namespace polar_to_circle_l1304_130408

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

end polar_to_circle_l1304_130408


namespace first_train_speed_l1304_130481

/-- Given two trains with a speed ratio of 7:8, where the second train travels 400 km in 4 hours,
    prove that the speed of the first train is 87.5 km/h. -/
theorem first_train_speed
  (speed_ratio : ℚ) -- Ratio of speeds between the two trains
  (distance : ℝ) -- Distance traveled by the second train
  (time : ℝ) -- Time taken by the second train
  (h1 : speed_ratio = 7 / 8) -- The ratio of speeds is 7:8
  (h2 : distance = 400) -- The second train travels 400 km
  (h3 : time = 4) -- The second train takes 4 hours
  : ∃ (speed1 : ℝ), speed1 = 87.5 := by
  sorry

end first_train_speed_l1304_130481


namespace sin_cos_product_zero_l1304_130421

theorem sin_cos_product_zero (θ : Real) (h : Real.sin θ + Real.cos θ = -1) : 
  Real.sin θ * Real.cos θ = 0 := by
  sorry

end sin_cos_product_zero_l1304_130421


namespace isosceles_minimizes_perimeter_l1304_130477

/-- Given a base length and area, the isosceles triangle minimizes the sum of the other two sides -/
theorem isosceles_minimizes_perimeter (a S : ℝ) (ha : a > 0) (hS : S > 0) :
  ∃ (h : ℝ), h > 0 ∧
  ∀ (b c : ℝ), b > 0 → c > 0 →
  (a * h / 2 = S) →
  (a * (b^2 - h^2).sqrt / 2 = S) →
  (a * (c^2 - h^2).sqrt / 2 = S) →
  b + c ≥ 2 * (4 * S^2 / a^2 + a^2 / 4).sqrt :=
sorry

end isosceles_minimizes_perimeter_l1304_130477


namespace min_a_value_l1304_130458

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp x * (x^3 - 3*x + 3) - a * exp x - x

theorem min_a_value :
  ∀ a : ℝ, (∃ x : ℝ, x ≥ -2 ∧ f a x ≤ 0) → a ≥ 1 - 1/exp 1 :=
by sorry

end min_a_value_l1304_130458


namespace find_p_l1304_130412

/-- Given a system of equations with a known solution, prove the value of p. -/
theorem find_p (p q : ℝ) (h1 : p * 2 + q * (-4) = 8) (h2 : 3 * 2 - q * (-4) = 38) : p = 20 := by
  sorry

#check find_p

end find_p_l1304_130412


namespace total_groceries_l1304_130475

def cookies : ℕ := 12
def noodles : ℕ := 16

theorem total_groceries : cookies + noodles = 28 := by
  sorry

end total_groceries_l1304_130475


namespace sum_of_M_subset_products_l1304_130447

def M : Set ℚ := {-2/3, 5/4, 1, 4}

def f (x : ℚ) : ℚ := (x + 2/3) * (x - 5/4) * (x - 1) * (x - 4)

def sum_of_subset_products (S : Set ℚ) : ℚ :=
  (f 1) - 1

theorem sum_of_M_subset_products :
  sum_of_subset_products M = 13/2 := by
  sorry

end sum_of_M_subset_products_l1304_130447


namespace supplementary_angle_measure_l1304_130465

-- Define the angle x
def x : ℝ := 10

-- Define the complementary angle
def complementary_angle (x : ℝ) : ℝ := 90 - x

-- Define the supplementary angle
def supplementary_angle (x : ℝ) : ℝ := 180 - x

-- Theorem statement
theorem supplementary_angle_measure :
  (x / complementary_angle x = 1 / 8) →
  supplementary_angle x = 170 := by
  sorry

end supplementary_angle_measure_l1304_130465


namespace ashley_age_is_8_l1304_130429

-- Define Ashley's and Mary's ages as natural numbers
variable (ashley_age mary_age : ℕ)

-- Define the conditions
def age_ratio : Prop := ashley_age * 7 = mary_age * 4
def age_sum : Prop := ashley_age + mary_age = 22

-- State the theorem
theorem ashley_age_is_8 
  (h1 : age_ratio ashley_age mary_age) 
  (h2 : age_sum ashley_age mary_age) : 
  ashley_age = 8 := by
sorry

end ashley_age_is_8_l1304_130429


namespace right_triangular_pyramid_relation_l1304_130491

/-- Right triangular pyramid with pairwise perpendicular side edges -/
structure RightTriangularPyramid where
  a : ℝ
  b : ℝ
  c : ℝ
  h : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  h_pos : 0 < h

/-- The relationship between side edges and altitude in a right triangular pyramid -/
theorem right_triangular_pyramid_relation (p : RightTriangularPyramid) :
  1 / p.a ^ 2 + 1 / p.b ^ 2 + 1 / p.c ^ 2 = 1 / p.h ^ 2 := by
  sorry

end right_triangular_pyramid_relation_l1304_130491
