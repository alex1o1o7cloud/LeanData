import Mathlib

namespace NUMINAMATH_CALUDE_competition_distance_l3371_337115

/-- Represents the distances cycled on each day of the week -/
structure WeekDistances where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ
  saturday : ℝ
  sunday : ℝ

/-- Calculates the total distance cycled in a week -/
def totalDistance (distances : WeekDistances) : ℝ :=
  distances.monday + distances.tuesday + distances.wednesday + 
  distances.thursday + distances.friday + distances.saturday + distances.sunday

/-- Theorem stating the total distance cycled in the competition week -/
theorem competition_distance : ∃ (distances : WeekDistances),
  distances.monday = 40 ∧
  distances.tuesday = 50 ∧
  distances.wednesday = distances.tuesday * 0.5 ∧
  distances.thursday = distances.monday + distances.wednesday ∧
  distances.friday = distances.thursday * 1.2 ∧
  distances.saturday = distances.friday * 0.75 ∧
  distances.sunday = distances.saturday - distances.wednesday ∧
  totalDistance distances = 350 := by
  sorry


end NUMINAMATH_CALUDE_competition_distance_l3371_337115


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l3371_337191

theorem quadratic_roots_problem (α β : ℝ) (h1 : α^2 - α - 2021 = 0)
                                         (h2 : β^2 - β - 2021 = 0)
                                         (h3 : α > β) : 
  let A := α^2 - 2*β^2 + 2*α*β + 3*β + 7
  ⌊A⌋ = -5893 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l3371_337191


namespace NUMINAMATH_CALUDE_normal_distribution_two_std_dev_below_mean_l3371_337147

theorem normal_distribution_two_std_dev_below_mean 
  (μ : ℝ) (σ : ℝ) (h_μ : μ = 17.5) (h_σ : σ = 2.5) :
  μ - 2 * σ = 12.5 :=
by sorry

end NUMINAMATH_CALUDE_normal_distribution_two_std_dev_below_mean_l3371_337147


namespace NUMINAMATH_CALUDE_no_nonnegative_solutions_quadratic_l3371_337161

theorem no_nonnegative_solutions_quadratic :
  ¬ ∃ x : ℝ, x ≥ 0 ∧ x^2 + 6*x + 9 = 0 := by
sorry

end NUMINAMATH_CALUDE_no_nonnegative_solutions_quadratic_l3371_337161


namespace NUMINAMATH_CALUDE_ambiguous_dates_count_l3371_337188

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The minimum number of days in each month -/
def min_days_per_month : ℕ := 12

/-- The number of days with ambiguous date interpretation -/
def ambiguous_days : ℕ := months_in_year * min_days_per_month - months_in_year

theorem ambiguous_dates_count :
  ambiguous_days = 132 :=
sorry

end NUMINAMATH_CALUDE_ambiguous_dates_count_l3371_337188


namespace NUMINAMATH_CALUDE_digit_456_is_8_l3371_337177

/-- The decimal representation of 17/59 has a repeating cycle of 29 digits -/
def decimal_cycle : List Nat := [2, 8, 8, 1, 3, 5, 5, 9, 3, 2, 2, 0, 3, 3, 8, 9, 8, 3, 0, 5, 0, 8, 4, 7, 4, 5, 7, 6, 2, 7, 1, 1]

/-- The length of the repeating cycle in the decimal representation of 17/59 -/
def cycle_length : Nat := 29

/-- The 456th digit after the decimal point in the representation of 17/59 -/
def digit_456 : Nat := decimal_cycle[(456 % cycle_length) - 1]

theorem digit_456_is_8 : digit_456 = 8 := by sorry

end NUMINAMATH_CALUDE_digit_456_is_8_l3371_337177


namespace NUMINAMATH_CALUDE_black_ball_count_l3371_337130

/-- Given a bag with white and black balls, prove the number of black balls when the probability of drawing a white ball is known -/
theorem black_ball_count 
  (white_balls : ℕ) 
  (black_balls : ℕ) 
  (total_balls : ℕ) 
  (prob_white : ℚ) 
  (h1 : white_balls = 20)
  (h2 : total_balls = white_balls + black_balls)
  (h3 : prob_white = 2/5)
  (h4 : prob_white = white_balls / total_balls) :
  black_balls = 30 := by
sorry

end NUMINAMATH_CALUDE_black_ball_count_l3371_337130


namespace NUMINAMATH_CALUDE_equation_solution_l3371_337144

theorem equation_solution : ∃ x : ℝ, 24 - (4 * 2) = 5 + x ∧ x = 11 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3371_337144


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l3371_337171

/-- Given a square carpet with shaded squares, calculate the total shaded area -/
theorem shaded_area_calculation (carpet_side : ℝ) (S T : ℝ) 
  (h1 : carpet_side = 12)
  (h2 : carpet_side / S = 4)
  (h3 : S / T = 4) : 
  S^2 + 12 * T^2 = 15.75 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l3371_337171


namespace NUMINAMATH_CALUDE_fraction_inequality_l3371_337105

theorem fraction_inequality (x : ℝ) (h : x ≠ -5) :
  x / (x + 5) ≥ 0 ↔ x ∈ Set.Ici 0 ∪ Set.Iic (-5) :=
sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3371_337105


namespace NUMINAMATH_CALUDE_box_dimensions_l3371_337141

theorem box_dimensions (a b c : ℝ) 
  (h1 : a + c = 17)
  (h2 : a + b = 13)
  (h3 : 2 * (b + c) = 40)
  (h4 : a < b)
  (h5 : b < c) :
  a = 5 ∧ b = 8 ∧ c = 12 := by
  sorry

end NUMINAMATH_CALUDE_box_dimensions_l3371_337141


namespace NUMINAMATH_CALUDE_rectangle_area_l3371_337180

theorem rectangle_area (L W : ℝ) (h1 : L + W = 7) (h2 : L^2 + W^2 = 25) : L * W = 12 := by
  sorry

#check rectangle_area

end NUMINAMATH_CALUDE_rectangle_area_l3371_337180


namespace NUMINAMATH_CALUDE_weight_loss_difference_l3371_337197

theorem weight_loss_difference (total_loss weight_first weight_third weight_fourth : ℕ) :
  total_loss = weight_first + weight_third + weight_fourth + (weight_first - 7) →
  weight_third = weight_fourth →
  total_loss = 103 →
  weight_first = 27 →
  weight_third = 28 →
  7 = weight_first - (total_loss - weight_first - weight_third - weight_fourth) :=
by sorry

end NUMINAMATH_CALUDE_weight_loss_difference_l3371_337197


namespace NUMINAMATH_CALUDE_harveys_steak_sales_l3371_337174

/-- Calculates the total number of steaks sold given the initial count, 
    the count after the first sale, and the count of the second sale. -/
def total_steaks_sold (initial : Nat) (after_first_sale : Nat) (second_sale : Nat) : Nat :=
  (initial - after_first_sale) + second_sale

/-- Theorem stating that given Harvey's specific situation, 
    the total number of steaks sold is 17. -/
theorem harveys_steak_sales : 
  total_steaks_sold 25 12 4 = 17 := by
  sorry

end NUMINAMATH_CALUDE_harveys_steak_sales_l3371_337174


namespace NUMINAMATH_CALUDE_diana_grace_age_ratio_l3371_337107

/-- The ratio of Diana's age to Grace's age -/
def age_ratio (diana_age : ℕ) (grace_age : ℕ) : ℚ :=
  diana_age / grace_age

/-- Grace's current age -/
def grace_current_age (grace_last_year : ℕ) : ℕ :=
  grace_last_year + 1

theorem diana_grace_age_ratio :
  let diana_age : ℕ := 8
  let grace_last_year : ℕ := 3
  age_ratio diana_age (grace_current_age grace_last_year) = 2 := by
sorry

end NUMINAMATH_CALUDE_diana_grace_age_ratio_l3371_337107


namespace NUMINAMATH_CALUDE_quadratic_inequality_sum_l3371_337116

theorem quadratic_inequality_sum (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → 
  a + b = -14 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_sum_l3371_337116


namespace NUMINAMATH_CALUDE_sum_in_base5_l3371_337110

/-- Converts a number from base 10 to base 5 -/
def toBase5 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 5 to base 10 -/
def fromBase5 (n : ℕ) : ℕ := sorry

theorem sum_in_base5 : toBase5 (45 + 27) = 242 := by sorry

end NUMINAMATH_CALUDE_sum_in_base5_l3371_337110


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l3371_337160

theorem largest_constant_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x / Real.sqrt (y + z)) + (y / Real.sqrt (z + x)) + (z / Real.sqrt (x + y)) ≤ (Real.sqrt 6 / 2) * Real.sqrt (x + y + z) ∧
  ∀ k > Real.sqrt 6 / 2, ∃ x' y' z' : ℝ, x' > 0 ∧ y' > 0 ∧ z' > 0 ∧
    (x' / Real.sqrt (y' + z')) + (y' / Real.sqrt (z' + x')) + (z' / Real.sqrt (x' + y')) > k * Real.sqrt (x' + y' + z') :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l3371_337160


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3371_337155

theorem quadratic_inequality_range :
  {a : ℝ | ∃ x : ℝ, a * x^2 + 2 * x + a < 0} = {a : ℝ | a < 1} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3371_337155


namespace NUMINAMATH_CALUDE_puzzle_solution_l3371_337154

/-- Represents a single digit (0-9) -/
def Digit := Fin 10

/-- Represents a two-digit number -/
def TwoDigitNumber := Fin 100

theorem puzzle_solution (P Q R S : Digit) 
  (h1 : (P.val * 10 + Q.val) * R.val = S.val * 10 + P.val)
  (h2 : (P.val * 10 + Q.val) + (R.val * 10 + P.val) = S.val * 10 + Q.val) :
  S.val = 1 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_solution_l3371_337154


namespace NUMINAMATH_CALUDE_petya_vasya_meet_at_64_l3371_337111

/-- The number of lampposts along the alley -/
def num_lampposts : ℕ := 100

/-- The lamppost number where Petya is observed -/
def petya_observed : ℕ := 22

/-- The lamppost number where Vasya is observed -/
def vasya_observed : ℕ := 88

/-- The function to calculate the meeting point of Petya and Vasya -/
def meeting_point : ℕ := sorry

/-- Theorem stating that Petya and Vasya meet at lamppost 64 -/
theorem petya_vasya_meet_at_64 : meeting_point = 64 := by sorry

end NUMINAMATH_CALUDE_petya_vasya_meet_at_64_l3371_337111


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l3371_337121

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 10) / (Nat.factorial 4)) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l3371_337121


namespace NUMINAMATH_CALUDE_probability_rain_july_approx_l3371_337149

/-- The probability of rain on at most 1 day in July, given the daily rain probability and number of days. -/
def probability_rain_at_most_one_day (daily_prob : ℝ) (num_days : ℕ) : ℝ :=
  (1 - daily_prob) ^ num_days + num_days * daily_prob * (1 - daily_prob) ^ (num_days - 1)

/-- Theorem stating that the probability of rain on at most 1 day in July is approximately 0.271. -/
theorem probability_rain_july_approx : 
  ∃ ε > 0, ε < 0.001 ∧ 
  |probability_rain_at_most_one_day (1/20) 31 - 0.271| < ε :=
sorry

end NUMINAMATH_CALUDE_probability_rain_july_approx_l3371_337149


namespace NUMINAMATH_CALUDE_competition_scores_l3371_337186

theorem competition_scores (n d : ℕ) : 
  n > 1 → 
  d > 0 → 
  d * (n * (n + 1)) / 2 = 26 * n → 
  ((n = 3 ∧ d = 13) ∨ (n = 12 ∧ d = 4) ∨ (n = 25 ∧ d = 2)) := by
  sorry

end NUMINAMATH_CALUDE_competition_scores_l3371_337186


namespace NUMINAMATH_CALUDE_probability_at_least_10_rubles_l3371_337129

-- Define the total number of tickets
def total_tickets : ℕ := 100

-- Define the number of tickets for each prize category
def tickets_20_rubles : ℕ := 5
def tickets_15_rubles : ℕ := 10
def tickets_10_rubles : ℕ := 15
def tickets_2_rubles : ℕ := 25

-- Define the probability of winning at least 10 rubles
def prob_at_least_10_rubles : ℚ :=
  (tickets_20_rubles + tickets_15_rubles + tickets_10_rubles) / total_tickets

-- Theorem statement
theorem probability_at_least_10_rubles :
  prob_at_least_10_rubles = 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_10_rubles_l3371_337129


namespace NUMINAMATH_CALUDE_correct_substitution_l3371_337165

theorem correct_substitution (x y : ℝ) : 
  y = 1 - x ∧ x - 2*y = 4 → x - 2 + 2*x = 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_substitution_l3371_337165


namespace NUMINAMATH_CALUDE_quadruplet_solution_l3371_337120

theorem quadruplet_solution (x₁ x₂ x₃ x₄ : ℝ) :
  (x₁ + x₂ = x₃^2 + x₄^2 + 6*x₃*x₄) ∧
  (x₁ + x₃ = x₂^2 + x₄^2 + 6*x₂*x₄) ∧
  (x₁ + x₄ = x₂^2 + x₃^2 + 6*x₂*x₃) ∧
  (x₂ + x₃ = x₁^2 + x₄^2 + 6*x₁*x₄) ∧
  (x₂ + x₄ = x₁^2 + x₃^2 + 6*x₁*x₃) ∧
  (x₃ + x₄ = x₁^2 + x₂^2 + 6*x₁*x₂) →
  (∃ c : ℝ, (x₁ = c ∧ x₂ = c ∧ x₃ = c ∧ x₄ = -3*c) ∨
            (x₁ = c ∧ x₂ = c ∧ x₃ = c ∧ x₄ = 1 - 3*c)) :=
by sorry


end NUMINAMATH_CALUDE_quadruplet_solution_l3371_337120


namespace NUMINAMATH_CALUDE_complement_union_theorem_l3371_337109

def I : Set Nat := {0, 1, 2, 3}
def A : Set Nat := {0, 1, 2}
def B : Set Nat := {2, 3}

theorem complement_union_theorem :
  (I \ A) ∪ (I \ B) = {0, 1, 3} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l3371_337109


namespace NUMINAMATH_CALUDE_min_value_expression_l3371_337103

theorem min_value_expression (a b m n : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_m : 0 < m) (h_pos_n : 0 < n)
  (h_sum : a + b = 1) (h_prod : m * n = 2) :
  2 ≤ (a * m + b * n) * (b * m + a * n) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3371_337103


namespace NUMINAMATH_CALUDE_snow_volume_on_blocked_sidewalk_l3371_337127

/-- Calculates the volume of snow to shovel from a partially blocked rectangular sidewalk. -/
theorem snow_volume_on_blocked_sidewalk
  (total_length : ℝ)
  (width : ℝ)
  (blocked_length : ℝ)
  (snow_depth : ℝ)
  (h1 : total_length = 30)
  (h2 : width = 3)
  (h3 : blocked_length = 5)
  (h4 : snow_depth = 2/3)
  : (total_length - blocked_length) * width * snow_depth = 50 := by
  sorry

#check snow_volume_on_blocked_sidewalk

end NUMINAMATH_CALUDE_snow_volume_on_blocked_sidewalk_l3371_337127


namespace NUMINAMATH_CALUDE_min_students_in_class_l3371_337151

theorem min_students_in_class (b g : ℕ) : 
  (2 * b / 3 : ℚ) = (3 * g / 4 : ℚ) →
  b + g ≥ 17 ∧ 
  ∃ (b' g' : ℕ), b' + g' = 17 ∧ (2 * b' / 3 : ℚ) = (3 * g' / 4 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_min_students_in_class_l3371_337151


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l3371_337193

theorem smallest_number_with_remainders (n : ℕ) : 
  (n > 1) →
  (n % 3 = 1) → 
  (n % 4 = 1) → 
  (n % 7 = 1) → 
  (∀ m : ℕ, m > 1 → m % 3 = 1 → m % 4 = 1 → m % 7 = 1 → n ≤ m) →
  (n = 85 ∧ 84 < n ∧ n ≤ 107) := by
sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l3371_337193


namespace NUMINAMATH_CALUDE_holdens_class_results_l3371_337164

/-- Proves the number of students who received an A in Mr. Holden's class exam
    and the number of students who did not receive an A in Mr. Holden's class quiz -/
theorem holdens_class_results (kennedy_total : ℕ) (kennedy_a : ℕ) (holden_total : ℕ)
    (h1 : kennedy_total = 20)
    (h2 : kennedy_a = 8)
    (h3 : holden_total = 30)
    (h4 : (kennedy_a : ℚ) / kennedy_total = (holden_a : ℚ) / holden_total)
    (h5 : (holden_total - holden_a : ℚ) / holden_total = 2 * (holden_not_a_quiz : ℚ) / holden_total) :
    holden_a = 12 ∧ holden_not_a_quiz = 9 := by
  sorry

#check holdens_class_results

end NUMINAMATH_CALUDE_holdens_class_results_l3371_337164


namespace NUMINAMATH_CALUDE_area_of_specific_circumscribed_rectangle_l3371_337192

/-- A rectangle circumscribed around a right triangle -/
structure CircumscribedRectangle where
  /-- Length of one leg of the right triangle -/
  leg1 : ℝ
  /-- Length of the other leg of the right triangle -/
  leg2 : ℝ
  /-- The legs are positive -/
  leg1_pos : 0 < leg1
  leg2_pos : 0 < leg2

/-- The area of a rectangle circumscribed around a right triangle -/
def area (r : CircumscribedRectangle) : ℝ := r.leg1 * r.leg2

/-- Theorem: The area of a rectangle circumscribed around a right triangle
    with legs of length 5 and 6 is equal to 30 square units -/
theorem area_of_specific_circumscribed_rectangle :
  ∃ (r : CircumscribedRectangle), r.leg1 = 5 ∧ r.leg2 = 6 ∧ area r = 30 := by
  sorry


end NUMINAMATH_CALUDE_area_of_specific_circumscribed_rectangle_l3371_337192


namespace NUMINAMATH_CALUDE_phil_remaining_books_pages_l3371_337140

def book_pages : List Nat := [120, 150, 80, 200, 90, 180, 75, 190, 110, 160, 130, 170, 100, 140, 210]

def misplaced_indices : List Nat := [1, 5, 9, 14]  -- 0-based indices

def remaining_pages : Nat := book_pages.sum - (misplaced_indices.map (λ i => book_pages.get! i)).sum

theorem phil_remaining_books_pages :
  remaining_pages = 1305 := by sorry

end NUMINAMATH_CALUDE_phil_remaining_books_pages_l3371_337140


namespace NUMINAMATH_CALUDE_tax_deduction_proof_l3371_337190

/-- Represents the hourly wage in dollars -/
def hourly_wage : ℝ := 25

/-- Represents the local tax rate as a decimal -/
def tax_rate : ℝ := 0.025

/-- Represents the number of cents in a dollar -/
def cents_per_dollar : ℝ := 100

/-- Calculates the tax deduction in cents -/
def tax_deduction_cents : ℝ := hourly_wage * cents_per_dollar * tax_rate

theorem tax_deduction_proof : tax_deduction_cents = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_tax_deduction_proof_l3371_337190


namespace NUMINAMATH_CALUDE_expression_simplification_l3371_337131

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  (x / (x^2 - 2*x + 1)) / ((x + 1) / (x^2 - 1) + 1) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3371_337131


namespace NUMINAMATH_CALUDE_quadratic_j_value_l3371_337125

-- Define the quadratic function
def quadratic (p q r : ℝ) (x : ℝ) : ℝ := p * x^2 + q * x + r

-- State the theorem
theorem quadratic_j_value (p q r : ℝ) :
  (∃ m n : ℝ, ∀ x : ℝ, 4 * (quadratic p q r x) = m * (x - 5)^2 + n) →
  (∃ m n : ℝ, ∀ x : ℝ, quadratic p q r x = 3 * (x - 5)^2 + 15) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_j_value_l3371_337125


namespace NUMINAMATH_CALUDE_osmanthus_price_is_300_l3371_337196

/-- The unit price of osmanthus trees given the following conditions:
  - Total amount raised is 7000 yuan
  - Total number of trees is 30
  - Cost of osmanthus trees is 3000 yuan
  - Unit price of osmanthus trees is 50% higher than cherry trees
-/
def osmanthus_price : ℝ :=
  let total_amount : ℝ := 7000
  let total_trees : ℝ := 30
  let osmanthus_cost : ℝ := 3000
  let price_ratio : ℝ := 1.5
  300

theorem osmanthus_price_is_300 :
  osmanthus_price = 300 := by sorry

end NUMINAMATH_CALUDE_osmanthus_price_is_300_l3371_337196


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l3371_337133

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z ≥ 3) :
  1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) ≤ 1 :=
sorry

theorem equality_condition (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z ≥ 3) :
  (1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) = 1) ↔ 
  (x = 1 ∧ y = 1 ∧ z = 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l3371_337133


namespace NUMINAMATH_CALUDE_smallest_cookie_boxes_l3371_337162

theorem smallest_cookie_boxes : ∃ (n : ℕ), n > 0 ∧ (15 * n - 1) % 11 = 0 ∧ ∀ (m : ℕ), m > 0 → (15 * m - 1) % 11 = 0 → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_cookie_boxes_l3371_337162


namespace NUMINAMATH_CALUDE_distance_to_origin_problem_l3371_337172

theorem distance_to_origin_problem (a : ℝ) : 
  (|a| = 2) → (a - 2 = 0 ∨ a - 2 = -4) := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_problem_l3371_337172


namespace NUMINAMATH_CALUDE_factors_of_72_l3371_337118

theorem factors_of_72 : Nat.card (Nat.divisors 72) = 12 := by sorry

end NUMINAMATH_CALUDE_factors_of_72_l3371_337118


namespace NUMINAMATH_CALUDE_james_change_calculation_l3371_337142

/-- Calculates the change James receives after purchasing items with discounts. -/
theorem james_change_calculation (candy_packs : ℕ) (chocolate_bars : ℕ) (chip_bags : ℕ)
  (candy_price : ℚ) (chocolate_price : ℚ) (chip_price : ℚ)
  (candy_discount : ℚ) (chip_discount : ℚ) (payment : ℚ) :
  candy_packs = 3 →
  chocolate_bars = 2 →
  chip_bags = 4 →
  candy_price = 12 →
  chocolate_price = 3 →
  chip_price = 2 →
  candy_discount = 15 / 100 →
  chip_discount = 10 / 100 →
  payment = 50 →
  let candy_total := candy_packs * candy_price * (1 - candy_discount)
  let chocolate_total := chocolate_price -- Due to buy-one-get-one-free offer
  let chip_total := chip_bags * chip_price * (1 - chip_discount)
  let total_cost := candy_total + chocolate_total + chip_total
  payment - total_cost = 9.2 := by sorry

end NUMINAMATH_CALUDE_james_change_calculation_l3371_337142


namespace NUMINAMATH_CALUDE_conference_attendees_l3371_337157

theorem conference_attendees :
  ∃ n : ℕ,
    n < 50 ∧
    n % 8 = 5 ∧
    n % 6 = 3 ∧
    n = 45 := by
  sorry

end NUMINAMATH_CALUDE_conference_attendees_l3371_337157


namespace NUMINAMATH_CALUDE_complement_of_union_is_four_l3371_337132

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 3}
def N : Set ℕ := {2, 5}

theorem complement_of_union_is_four :
  (M ∪ N)ᶜ = {4} :=
by sorry

end NUMINAMATH_CALUDE_complement_of_union_is_four_l3371_337132


namespace NUMINAMATH_CALUDE_discriminant_of_2x2_minus_5x_plus_6_l3371_337176

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem: The discriminant of 2x^2 - 5x + 6 is -23 -/
theorem discriminant_of_2x2_minus_5x_plus_6 :
  discriminant 2 (-5) 6 = -23 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_of_2x2_minus_5x_plus_6_l3371_337176


namespace NUMINAMATH_CALUDE_train_speed_l3371_337189

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length time_to_cross : ℝ) 
  (h1 : train_length = 110)
  (h2 : bridge_length = 170)
  (h3 : time_to_cross = 13.998880089592832) :
  (train_length + bridge_length) / time_to_cross = 20.0014286607 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3371_337189


namespace NUMINAMATH_CALUDE_initial_files_correct_l3371_337170

/-- The number of files Megan initially had on her computer -/
def initial_files : ℕ := 93

/-- The number of files Megan deleted -/
def deleted_files : ℕ := 21

/-- The number of files in each folder -/
def files_per_folder : ℕ := 8

/-- The number of folders Megan ended up with -/
def num_folders : ℕ := 9

/-- Theorem stating that the initial number of files is correct -/
theorem initial_files_correct : 
  initial_files = deleted_files + num_folders * files_per_folder :=
by sorry

end NUMINAMATH_CALUDE_initial_files_correct_l3371_337170


namespace NUMINAMATH_CALUDE_exists_valid_sequence_l3371_337183

def is_valid_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ k, k ≥ 1 → a (2*k + 1) = (a (2*k) + a (2*k + 2)) / 2) ∧
  (∀ k, k ≥ 1 → a (2*k) = Real.sqrt (a (2*k - 1) * a (2*k + 1)))

theorem exists_valid_sequence : ∃ a : ℕ → ℝ, is_valid_sequence a :=
sorry

end NUMINAMATH_CALUDE_exists_valid_sequence_l3371_337183


namespace NUMINAMATH_CALUDE_reflection_of_A_across_y_axis_l3371_337173

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- The original point A -/
def A : ℝ × ℝ := (-2, 5)

theorem reflection_of_A_across_y_axis :
  reflect_y A = (2, 5) := by sorry

end NUMINAMATH_CALUDE_reflection_of_A_across_y_axis_l3371_337173


namespace NUMINAMATH_CALUDE_brown_eggs_survived_l3371_337145

/-- Given that Linda initially had three times as many white eggs as brown eggs,
    and after dropping her basket she ended up with a dozen eggs in total,
    prove that 3 brown eggs survived the fall. -/
theorem brown_eggs_survived (white_eggs brown_eggs : ℕ) : 
  white_eggs = 3 * brown_eggs →  -- Initial condition
  white_eggs + brown_eggs = 12 →  -- Total eggs after the fall
  brown_eggs > 0 →  -- Some brown eggs survived
  brown_eggs = 3 := by
  sorry

end NUMINAMATH_CALUDE_brown_eggs_survived_l3371_337145


namespace NUMINAMATH_CALUDE_polygon_angles_l3371_337117

theorem polygon_angles (n : ℕ) : 
  (n - 2) * 180 + (180 - 180 / n) = 2007 → n = 13 := by
  sorry

end NUMINAMATH_CALUDE_polygon_angles_l3371_337117


namespace NUMINAMATH_CALUDE_real_part_reciprocal_l3371_337126

theorem real_part_reciprocal (z : ℂ) (h1 : z ≠ (z.re : ℂ)) (h2 : Complex.abs z = 2) :
  ((2 - z)⁻¹).re = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_reciprocal_l3371_337126


namespace NUMINAMATH_CALUDE_baker_cupcake_distribution_l3371_337114

/-- The number of cupcakes left over when distributing cupcakes equally -/
def cupcakes_left_over (total : ℕ) (children : ℕ) : ℕ :=
  total % children

/-- Theorem: When distributing 17 cupcakes among 3 children equally, 2 cupcakes are left over -/
theorem baker_cupcake_distribution :
  cupcakes_left_over 17 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_baker_cupcake_distribution_l3371_337114


namespace NUMINAMATH_CALUDE_min_value_expression_l3371_337158

theorem min_value_expression (x y : ℝ) (h : 4 - 16*x^2 - 8*x*y - y^2 > 0) :
  (13*x^2 + 24*x*y + 13*y^2 - 14*x - 16*y + 61) / (4 - 16*x^2 - 8*x*y - y^2)^(7/2) ≥ 7/16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3371_337158


namespace NUMINAMATH_CALUDE_original_class_size_l3371_337139

/-- Proves that the original number of students in a class is 12, given the conditions of the problem. -/
theorem original_class_size (initial_avg : ℝ) (new_students : ℕ) (new_avg : ℝ) (avg_decrease : ℝ) :
  initial_avg = 40 →
  new_students = 12 →
  new_avg = 32 →
  avg_decrease = 4 →
  ∃ (original_size : ℕ),
    original_size * initial_avg + new_students * new_avg = (original_size + new_students) * (initial_avg - avg_decrease) ∧
    original_size = 12 := by
  sorry

end NUMINAMATH_CALUDE_original_class_size_l3371_337139


namespace NUMINAMATH_CALUDE_fruit_seller_apples_l3371_337113

theorem fruit_seller_apples (initial_apples : ℕ) (remaining_apples : ℕ) : 
  remaining_apples = 420 → 
  (initial_apples : ℚ) * (70 / 100) = remaining_apples → 
  initial_apples = 600 := by
sorry

end NUMINAMATH_CALUDE_fruit_seller_apples_l3371_337113


namespace NUMINAMATH_CALUDE_cousin_age_l3371_337166

/-- Given the ages of Rick and his brothers, prove the age of their cousin -/
theorem cousin_age (rick_age : ℕ) (oldest_brother_age : ℕ) (middle_brother_age : ℕ) 
  (smallest_brother_age : ℕ) (youngest_brother_age : ℕ) (cousin_age : ℕ) 
  (h1 : rick_age = 15)
  (h2 : oldest_brother_age = 2 * rick_age)
  (h3 : middle_brother_age = oldest_brother_age / 3)
  (h4 : smallest_brother_age = middle_brother_age / 2)
  (h5 : youngest_brother_age = smallest_brother_age - 2)
  (h6 : cousin_age = 5 * youngest_brother_age) :
  cousin_age = 15 := by
  sorry

#check cousin_age

end NUMINAMATH_CALUDE_cousin_age_l3371_337166


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l3371_337167

theorem triangle_third_side_length (a b c : ℝ) (θ : ℝ) : 
  a = 10 → b = 12 → θ = π / 3 → c = Real.sqrt 124 := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l3371_337167


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3371_337136

theorem complex_equation_solution (a : ℝ) (i : ℂ) : 
  i * i = -1 →
  (a^2 - a : ℂ) + (3*a - 1 : ℂ) * i = 2 + 5*i →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3371_337136


namespace NUMINAMATH_CALUDE_triple_hash_72_l3371_337178

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.5 * N - 1

-- Theorem statement
theorem triple_hash_72 : hash (hash (hash 72)) = 7.25 := by
  sorry

end NUMINAMATH_CALUDE_triple_hash_72_l3371_337178


namespace NUMINAMATH_CALUDE_smallest_a_for_sum_of_squares_l3371_337156

theorem smallest_a_for_sum_of_squares (a : ℝ) : 
  (∀ x : ℝ, x^2 - 3*a*x + a^2 = 0 → 
   ∃ x1 x2 : ℝ, x1^2 + x2^2 = 0.28 ∧ x1 ≠ x2 ∧ 
   (∀ y : ℝ, y^2 - 3*a*y + a^2 = 0 → y = x1 ∨ y = x2)) →
  a = -0.2 ∧ 
  (∀ b : ℝ, b < -0.2 → 
   ¬(∀ x : ℝ, x^2 - 3*b*x + b^2 = 0 → 
     ∃ x1 x2 : ℝ, x1^2 + x2^2 = 0.28 ∧ x1 ≠ x2 ∧ 
     (∀ y : ℝ, y^2 - 3*b*y + b^2 = 0 → y = x1 ∨ y = x2))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_for_sum_of_squares_l3371_337156


namespace NUMINAMATH_CALUDE_neg_p_sufficient_not_necessary_for_neg_q_l3371_337137

-- Define p and q as predicates on real numbers
def p (x : ℝ) : Prop := x < -1 ∨ x > 1
def q (x : ℝ) : Prop := x < -2 ∨ x > 1

-- Define the relationship between ¬p and ¬q
theorem neg_p_sufficient_not_necessary_for_neg_q :
  (∀ x : ℝ, ¬(p x) → ¬(q x)) ∧ 
  ¬(∀ x : ℝ, ¬(q x) → ¬(p x)) := by
  sorry

end NUMINAMATH_CALUDE_neg_p_sufficient_not_necessary_for_neg_q_l3371_337137


namespace NUMINAMATH_CALUDE_average_temperature_l3371_337182

theorem average_temperature (temperatures : List ℝ) (h1 : temperatures = [18, 21, 19, 22, 20]) :
  temperatures.sum / temperatures.length = 20 := by
sorry

end NUMINAMATH_CALUDE_average_temperature_l3371_337182


namespace NUMINAMATH_CALUDE_no_quadratic_term_implies_k_equals_three_l3371_337108

/-- 
Given an algebraic expression in x and y: (-3kxy+3y)+(9xy-8x+1),
prove that if there is no quadratic term, then k = 3.
-/
theorem no_quadratic_term_implies_k_equals_three (k : ℚ) : 
  (∀ x y : ℚ, (-3*k*x*y + 3*y) + (9*x*y - 8*x + 1) = (-3*k + 9)*x*y + 3*y - 8*x + 1) →
  (∀ x y : ℚ, (-3*k + 9)*x*y + 3*y - 8*x + 1 = 3*y - 8*x + 1) →
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_no_quadratic_term_implies_k_equals_three_l3371_337108


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l3371_337185

theorem floor_ceiling_sum_seven (x : ℝ) :
  (Int.floor x + Int.ceil x = 7) ↔ (3 < x ∧ x < 4) ∨ x = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l3371_337185


namespace NUMINAMATH_CALUDE_building_height_l3371_337163

/-- Given a flagpole and a building casting shadows under similar conditions,
    calculate the height of the building. -/
theorem building_height
  (flagpole_height : ℝ)
  (flagpole_shadow : ℝ)
  (building_shadow : ℝ)
  (h_flagpole : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_shadow : building_shadow = 50)
  : (flagpole_height / flagpole_shadow) * building_shadow = 20 := by
  sorry

end NUMINAMATH_CALUDE_building_height_l3371_337163


namespace NUMINAMATH_CALUDE_decimal_to_base5_l3371_337143

theorem decimal_to_base5 : 
  (89 : ℕ) = 3 * 5^2 + 2 * 5^1 + 4 * 5^0 :=
by sorry

end NUMINAMATH_CALUDE_decimal_to_base5_l3371_337143


namespace NUMINAMATH_CALUDE_food_distribution_l3371_337101

theorem food_distribution (initial_men : ℕ) (initial_days : ℕ) (additional_men : ℕ) (days_before_increase : ℕ) : 
  initial_men = 760 →
  initial_days = 22 →
  additional_men = 40 →
  days_before_increase = 2 →
  (initial_men * initial_days - initial_men * days_before_increase) / (initial_men + additional_men) = 19 :=
by sorry

end NUMINAMATH_CALUDE_food_distribution_l3371_337101


namespace NUMINAMATH_CALUDE_sufficient_condition_B_proper_subset_A_l3371_337198

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B (m : ℝ) : Set ℝ := {x | x * m + 1 = 0}

theorem sufficient_condition_B_proper_subset_A :
  ∃ S : Set ℝ, (S = {0, 1/3}) ∧ 
  (∀ m : ℝ, m ∈ S → B m ⊂ A) ∧
  (∃ m : ℝ, m ∉ S ∧ B m ⊂ A) :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_B_proper_subset_A_l3371_337198


namespace NUMINAMATH_CALUDE_brendas_age_l3371_337199

/-- Given that Addison's age is four times Brenda's age, Janet is seven years older than Brenda,
    and Addison and Janet are twins, prove that Brenda is 7/3 years old. -/
theorem brendas_age (addison janet brenda : ℚ)
  (h1 : addison = 4 * brenda)
  (h2 : janet = brenda + 7)
  (h3 : addison = janet) :
  brenda = 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_brendas_age_l3371_337199


namespace NUMINAMATH_CALUDE_like_terms_imply_n_eq_one_l3371_337168

/-- Two terms are considered like terms if they have the same variables with the same exponents -/
def like_terms (term1 term2 : ℤ → ℤ → ℤ) : Prop :=
  ∃ (a b : ℤ), ∀ (x y : ℤ), term1 x y = a * x^2 * y ∧ term2 x y = b * x^2 * y

/-- If -x^2y^n and 3yx^2 are like terms, then n = 1 -/
theorem like_terms_imply_n_eq_one :
  ∀ n : ℕ, like_terms (λ x y => -x^2 * y^n) (λ x y => 3 * y * x^2) → n = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_n_eq_one_l3371_337168


namespace NUMINAMATH_CALUDE_mary_flour_calculation_l3371_337184

/-- The amount of flour needed for the recipe -/
def total_flour : ℕ := 9

/-- The amount of flour Mary has already added -/
def added_flour : ℕ := 3

/-- The remaining amount of flour Mary needs to add -/
def remaining_flour : ℕ := total_flour - added_flour

theorem mary_flour_calculation :
  remaining_flour = 6 := by
  sorry

end NUMINAMATH_CALUDE_mary_flour_calculation_l3371_337184


namespace NUMINAMATH_CALUDE_fiftieth_ring_squares_l3371_337195

/-- Represents the number of unit squares in the nth ring of the described square array. -/
def ring_squares (n : ℕ) : ℕ := 32 * n - 16

/-- The theorem states that the 50th ring contains 1584 unit squares. -/
theorem fiftieth_ring_squares : ring_squares 50 = 1584 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_ring_squares_l3371_337195


namespace NUMINAMATH_CALUDE_ashleys_notebooks_l3371_337100

theorem ashleys_notebooks :
  ∀ (notebook_price pencil_price : ℕ) (notebooks_in_93 : ℕ),
    notebook_price + pencil_price = 5 →
    21 * pencil_price + notebooks_in_93 * notebook_price = 93 →
    notebooks_in_93 = 15 →
    ∃ (notebooks_in_5 : ℕ),
      notebooks_in_5 * notebook_price + 1 * pencil_price = 5 ∧
      notebooks_in_5 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ashleys_notebooks_l3371_337100


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l3371_337123

def complex_condition (z : ℂ) : Prop := z * (1 + Complex.I) = 2 * Complex.I

theorem z_in_first_quadrant (z : ℂ) (h : complex_condition z) : 
  0 < z.re ∧ 0 < z.im :=
sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l3371_337123


namespace NUMINAMATH_CALUDE_rectangle_to_square_cut_l3371_337124

theorem rectangle_to_square_cut (width : ℕ) (height : ℕ) : 
  width = 4 ∧ height = 9 → 
  ∃ (s : ℕ) (w1 w2 h1 h2 : ℕ),
    s * s = width * height ∧
    w1 + w2 = width ∧
    h1 = height ∧ h2 = height ∧
    (w1 * h1 + w2 * h2 = s * s) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_cut_l3371_337124


namespace NUMINAMATH_CALUDE_terminal_side_quadrant_l3371_337128

-- Define the quadrants
inductive Quadrant
  | I
  | II
  | III
  | IV

-- Define a function to determine the quadrant of an angle
def angle_quadrant (α : Real) : Quadrant := sorry

-- Define the theorem
theorem terminal_side_quadrant (α : Real) 
  (h1 : Real.sin α * Real.cos α < 0) 
  (h2 : Real.sin α * Real.tan α > 0) : 
  (angle_quadrant (α / 2) = Quadrant.II) ∨ (angle_quadrant (α / 2) = Quadrant.IV) := by
  sorry

end NUMINAMATH_CALUDE_terminal_side_quadrant_l3371_337128


namespace NUMINAMATH_CALUDE_opposite_of_three_l3371_337134

theorem opposite_of_three : ∃ x : ℤ, x + 3 = 0 ∧ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_three_l3371_337134


namespace NUMINAMATH_CALUDE_greatest_multiple_3_4_under_500_l3371_337179

theorem greatest_multiple_3_4_under_500 : ∃ n : ℕ, n = 492 ∧ 
  (∀ m : ℕ, m < 500 ∧ 3 ∣ m ∧ 4 ∣ m → m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_3_4_under_500_l3371_337179


namespace NUMINAMATH_CALUDE_five_students_three_locations_l3371_337106

/-- The number of ways to assign students to locations -/
def assignment_count (n : ℕ) (k : ℕ) : ℕ :=
  -- n: number of students
  -- k: number of locations
  sorry

/-- Theorem stating the number of assignment plans for 5 students and 3 locations -/
theorem five_students_three_locations :
  assignment_count 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_five_students_three_locations_l3371_337106


namespace NUMINAMATH_CALUDE_truck_wheels_l3371_337102

/-- Toll calculation function -/
def toll (x : ℕ) : ℚ :=
  0.5 + 0.5 * (x - 2)

/-- Number of wheels on the front axle -/
def frontWheels : ℕ := 2

/-- Number of wheels on each non-front axle -/
def otherWheels : ℕ := 4

/-- Theorem stating the total number of wheels on the truck -/
theorem truck_wheels (x : ℕ) (h1 : toll x = 2) (h2 : x > 0) : 
  frontWheels + (x - 1) * otherWheels = 18 := by
  sorry

#check truck_wheels

end NUMINAMATH_CALUDE_truck_wheels_l3371_337102


namespace NUMINAMATH_CALUDE_pencil_distribution_l3371_337119

/-- Given a classroom with 4 children and 8 pencils to be distributed,
    prove that each child receives 2 pencils. -/
theorem pencil_distribution (num_children : ℕ) (num_pencils : ℕ) 
  (h1 : num_children = 4) 
  (h2 : num_pencils = 8) : 
  num_pencils / num_children = 2 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l3371_337119


namespace NUMINAMATH_CALUDE_no_valid_labeling_l3371_337122

/-- A labeling function that assigns a label to each point in the integer lattice -/
def Labeling := ℤ → ℤ → Fin 4

/-- Predicate that checks if a unit square has all four labels at its vertices -/
def has_all_labels (f : Labeling) (x y : ℤ) : Prop :=
  ∀ i j : Fin 4, ∃ dx dy : Fin 2, f (x + dx) (y + dy) = i

/-- Predicate that checks if a row contains all four labels -/
def row_has_all_labels (f : Labeling) (y : ℤ) : Prop :=
  ∀ i : Fin 4, ∃ x : ℤ, f x y = i

/-- Predicate that checks if a column contains all four labels -/
def col_has_all_labels (f : Labeling) (x : ℤ) : Prop :=
  ∀ i : Fin 4, ∃ y : ℤ, f x y = i

/-- Theorem stating that no valid labeling exists -/
theorem no_valid_labeling : ¬∃ f : Labeling, 
  (∀ x y : ℤ, has_all_labels f x y) ∧ 
  (∀ y : ℤ, row_has_all_labels f y) ∧ 
  (∀ x : ℤ, col_has_all_labels f x) :=
sorry

end NUMINAMATH_CALUDE_no_valid_labeling_l3371_337122


namespace NUMINAMATH_CALUDE_production_quantity_for_36000_min_production_for_profit_8500_l3371_337159

-- Define the production cost function
def C (n : ℕ) : ℝ := 4000 + 50 * n

-- Define the profit function
def P (n : ℕ) : ℝ := 40 * n - 4000

-- Theorem 1: Production quantity when cost is 36,000
theorem production_quantity_for_36000 :
  ∃ n : ℕ, C n = 36000 ∧ n = 640 := by sorry

-- Theorem 2: Minimum production for profit ≥ 8,500
theorem min_production_for_profit_8500 :
  ∃ n : ℕ, (∀ m : ℕ, P m ≥ 8500 → m ≥ n) ∧ P n ≥ 8500 ∧ n = 313 := by sorry

end NUMINAMATH_CALUDE_production_quantity_for_36000_min_production_for_profit_8500_l3371_337159


namespace NUMINAMATH_CALUDE_f_max_min_l3371_337187

noncomputable def f (x : ℝ) : ℝ := 2^(x+2) - 3 * 4^x

theorem f_max_min :
  ∀ x ∈ Set.Icc (-1 : ℝ) 0,
    f x ≤ 4/3 ∧ f x ≥ 1 ∧
    (∃ x₁ ∈ Set.Icc (-1 : ℝ) 0, f x₁ = 4/3) ∧
    (∃ x₂ ∈ Set.Icc (-1 : ℝ) 0, f x₂ = 1) :=
by sorry

end NUMINAMATH_CALUDE_f_max_min_l3371_337187


namespace NUMINAMATH_CALUDE_cosine_symmetry_center_l3371_337148

/-- The symmetry center of the cosine function with a phase shift --/
theorem cosine_symmetry_center (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.cos (2 * x - π / 4)
  ∃ c : ℝ × ℝ, c = (π / 8, 0) ∧ 
    (∀ x : ℝ, f (c.1 + x) = f (c.1 - x)) :=
by sorry

end NUMINAMATH_CALUDE_cosine_symmetry_center_l3371_337148


namespace NUMINAMATH_CALUDE_length_PS_is_sqrt_32_5_l3371_337135

-- Define the quadrilateral PQRS
structure Quadrilateral :=
  (P Q R S T : ℝ × ℝ)

-- Define the conditions
def is_valid_quadrilateral (quad : Quadrilateral) : Prop :=
  let d_PT := Real.sqrt ((quad.P.1 - quad.T.1)^2 + (quad.P.2 - quad.T.2)^2)
  let d_TR := Real.sqrt ((quad.T.1 - quad.R.1)^2 + (quad.T.2 - quad.R.2)^2)
  let d_QT := Real.sqrt ((quad.Q.1 - quad.T.1)^2 + (quad.Q.2 - quad.T.2)^2)
  let d_TS := Real.sqrt ((quad.T.1 - quad.S.1)^2 + (quad.T.2 - quad.S.2)^2)
  let d_PQ := Real.sqrt ((quad.P.1 - quad.Q.1)^2 + (quad.P.2 - quad.Q.2)^2)
  d_PT = 5 ∧ d_TR = 4 ∧ d_QT = 7 ∧ d_TS = 2 ∧ d_PQ = 7

-- Theorem statement
theorem length_PS_is_sqrt_32_5 (quad : Quadrilateral) 
  (h : is_valid_quadrilateral quad) : 
  Real.sqrt ((quad.P.1 - quad.S.1)^2 + (quad.P.2 - quad.S.2)^2) = Real.sqrt 32.5 := by
  sorry

end NUMINAMATH_CALUDE_length_PS_is_sqrt_32_5_l3371_337135


namespace NUMINAMATH_CALUDE_area_of_twelve_sided_figure_l3371_337153

/-- A vertex is represented by its x and y coordinates -/
structure Vertex :=
  (x : ℝ)
  (y : ℝ)

/-- A polygon is represented by a list of vertices -/
def Polygon := List Vertex

/-- The vertices of our 12-sided figure -/
def twelveSidedFigure : Polygon := [
  ⟨1, 3⟩, ⟨2, 4⟩, ⟨2, 5⟩, ⟨3, 6⟩, ⟨4, 6⟩, ⟨5, 5⟩,
  ⟨6, 4⟩, ⟨6, 3⟩, ⟨5, 2⟩, ⟨4, 1⟩, ⟨3, 1⟩, ⟨2, 2⟩
]

/-- Function to calculate the area of a polygon -/
def areaOfPolygon (p : Polygon) : ℝ := sorry

/-- Theorem stating that the area of our 12-sided figure is 16 cm² -/
theorem area_of_twelve_sided_figure :
  areaOfPolygon twelveSidedFigure = 16 := by sorry

end NUMINAMATH_CALUDE_area_of_twelve_sided_figure_l3371_337153


namespace NUMINAMATH_CALUDE_katie_earnings_l3371_337152

/-- The number of bead necklaces Katie sold -/
def bead_necklaces : ℕ := 4

/-- The number of gem stone necklaces Katie sold -/
def gem_necklaces : ℕ := 3

/-- The cost of each necklace in dollars -/
def necklace_cost : ℕ := 3

/-- The total money Katie earned from selling necklaces -/
def total_earned : ℕ := (bead_necklaces + gem_necklaces) * necklace_cost

theorem katie_earnings : total_earned = 21 := by
  sorry

end NUMINAMATH_CALUDE_katie_earnings_l3371_337152


namespace NUMINAMATH_CALUDE_rent_calculation_l3371_337150

def problem (salary : ℕ) (milk groceries education petrol misc rent : ℕ) : Prop :=
  let savings := salary / 10
  let other_expenses := milk + groceries + education + petrol + misc
  salary = savings + rent + other_expenses ∧
  milk = 1500 ∧
  groceries = 4500 ∧
  education = 2500 ∧
  petrol = 2000 ∧
  misc = 2500 ∧
  savings = 2000

theorem rent_calculation :
  ∀ salary milk groceries education petrol misc rent,
    problem salary milk groceries education petrol misc rent →
    rent = 5000 := by
  sorry

end NUMINAMATH_CALUDE_rent_calculation_l3371_337150


namespace NUMINAMATH_CALUDE_prob_more_ones_than_sixes_proof_l3371_337194

/-- The number of possible outcomes when rolling five fair six-sided dice -/
def total_outcomes : ℕ := 6^5

/-- The number of ways to roll an equal number of 1's and 6's when rolling five fair six-sided dice -/
def equal_ones_and_sixes : ℕ := 2334

/-- The probability of rolling more 1's than 6's when rolling five fair six-sided dice -/
def prob_more_ones_than_sixes : ℚ := 2711 / 7776

theorem prob_more_ones_than_sixes_proof :
  prob_more_ones_than_sixes = 1/2 * (1 - equal_ones_and_sixes / total_outcomes) :=
by sorry

end NUMINAMATH_CALUDE_prob_more_ones_than_sixes_proof_l3371_337194


namespace NUMINAMATH_CALUDE_swimmer_speed_l3371_337169

/-- The speed of a swimmer in still water, given his downstream and upstream speeds and distances. -/
theorem swimmer_speed (downstream_distance : ℝ) (downstream_time : ℝ) 
  (upstream_distance : ℝ) (upstream_time : ℝ) :
  downstream_distance = 62 →
  downstream_time = 10 →
  upstream_distance = 84 →
  upstream_time = 14 →
  ∃ (v_m v_s : ℝ),
    v_m + v_s = downstream_distance / downstream_time ∧
    v_m - v_s = upstream_distance / upstream_time ∧
    v_m = 6.1 :=
by sorry

end NUMINAMATH_CALUDE_swimmer_speed_l3371_337169


namespace NUMINAMATH_CALUDE_proportion_equation_proof_l3371_337181

theorem proportion_equation_proof (x y : ℝ) (h1 : y ≠ 0) (h2 : 3 * x = 5 * y) :
  x / 5 = y / 3 := by sorry

end NUMINAMATH_CALUDE_proportion_equation_proof_l3371_337181


namespace NUMINAMATH_CALUDE_min_value_expression_l3371_337138

theorem min_value_expression (n : ℕ+) : 
  (n : ℝ) / 3 + 27 / (n : ℝ) ≥ 6 ∧ 
  ((n : ℝ) / 3 + 27 / (n : ℝ) = 6 ↔ n = 9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3371_337138


namespace NUMINAMATH_CALUDE_power_equality_implies_exponent_l3371_337104

theorem power_equality_implies_exponent (n : ℕ) : 4^8 = 16^n → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_implies_exponent_l3371_337104


namespace NUMINAMATH_CALUDE_bacteria_growth_proof_l3371_337112

/-- The number of 30-second intervals in 5 minutes -/
def intervals : ℕ := 10

/-- The growth factor of bacteria population in one interval -/
def growth_factor : ℕ := 4

/-- The final number of bacteria after 5 minutes -/
def final_population : ℕ := 4194304

/-- The initial number of bacteria -/
def initial_population : ℕ := 4

theorem bacteria_growth_proof :
  initial_population * growth_factor ^ intervals = final_population :=
by sorry

end NUMINAMATH_CALUDE_bacteria_growth_proof_l3371_337112


namespace NUMINAMATH_CALUDE_last_digit_sum_powers_l3371_337175

theorem last_digit_sum_powers : (2^2011 + 3^2011) % 10 = 5 := by sorry

end NUMINAMATH_CALUDE_last_digit_sum_powers_l3371_337175


namespace NUMINAMATH_CALUDE_fraction_calculation_l3371_337146

theorem fraction_calculation : 
  (2 / 7 + 5 / 8 * 1 / 3) / (3 / 4 - 2 / 9) = 15 / 16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l3371_337146
