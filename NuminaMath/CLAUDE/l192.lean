import Mathlib

namespace sum_digits_greatest_prime_divisor_of_16777_l192_19281

def n : ℕ := 16777

-- Define a function to get the greatest prime divisor
def greatest_prime_divisor (m : ℕ) : ℕ := sorry

-- Define a function to sum the digits of a number
def sum_of_digits (m : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_digits_greatest_prime_divisor_of_16777 :
  sum_of_digits (greatest_prime_divisor n) = 2 := by sorry

end sum_digits_greatest_prime_divisor_of_16777_l192_19281


namespace fourth_day_pages_l192_19225

/-- Represents the number of pages read each day -/
structure DailyPages where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day4 : ℕ

/-- Represents the conditions of the book reading problem -/
structure BookReading where
  totalPages : ℕ
  dailyPages : DailyPages
  day1Condition : dailyPages.day1 = 63
  day2Condition : dailyPages.day2 = 2 * dailyPages.day1
  day3Condition : dailyPages.day3 = dailyPages.day2 + 10
  totalCondition : totalPages = dailyPages.day1 + dailyPages.day2 + dailyPages.day3 + dailyPages.day4

/-- Theorem stating that given the conditions, the number of pages read on the fourth day is 29 -/
theorem fourth_day_pages (br : BookReading) (h : br.totalPages = 354) : br.dailyPages.day4 = 29 := by
  sorry

end fourth_day_pages_l192_19225


namespace probability_three_green_apples_l192_19257

theorem probability_three_green_apples (total_apples green_apples selected_apples : ℕ) :
  total_apples = 10 →
  green_apples = 4 →
  selected_apples = 3 →
  (Nat.choose green_apples selected_apples : ℚ) / (Nat.choose total_apples selected_apples) = 1 / 30 := by
  sorry

end probability_three_green_apples_l192_19257


namespace jan_cindy_age_difference_l192_19233

def age_difference (cindy_age jan_age marcia_age greg_age : ℕ) : Prop :=
  (cindy_age = 5) ∧
  (jan_age > cindy_age) ∧
  (marcia_age = 2 * jan_age) ∧
  (greg_age = marcia_age + 2) ∧
  (greg_age = 16) ∧
  (jan_age - cindy_age = 2)

theorem jan_cindy_age_difference :
  ∃ (cindy_age jan_age marcia_age greg_age : ℕ),
    age_difference cindy_age jan_age marcia_age greg_age := by
  sorry

end jan_cindy_age_difference_l192_19233


namespace jamies_mothers_age_twice_l192_19291

/-- 
Given:
- Jamie's age in 2010 is 10 years
- Jamie's mother's age in 2010 is 5 times Jamie's age
Prove that the year when Jamie's mother's age will be twice Jamie's age is 2040
-/
theorem jamies_mothers_age_twice (jamie_age_2010 : ℕ) (mother_age_multiplier : ℕ) : 
  jamie_age_2010 = 10 →
  mother_age_multiplier = 5 →
  ∃ (years_passed : ℕ),
    (jamie_age_2010 + years_passed) * 2 = (jamie_age_2010 * mother_age_multiplier + years_passed) ∧
    2010 + years_passed = 2040 := by
  sorry

#check jamies_mothers_age_twice

end jamies_mothers_age_twice_l192_19291


namespace simplify_expression_l192_19258

theorem simplify_expression (x : ℝ) : (3 * x - 4) * (x + 8) - (x + 6) * (3 * x - 2) = 4 * x - 20 := by
  sorry

end simplify_expression_l192_19258


namespace quadratic_factorization_l192_19272

theorem quadratic_factorization (b c : ℝ) :
  (∀ x, x^2 + b*x + c = 0 ↔ x = -2 ∨ x = 3) →
  ∀ x, x^2 + b*x + c = (x + 2) * (x - 3) :=
by
  sorry

end quadratic_factorization_l192_19272


namespace area_of_ring_l192_19298

/-- The area of a ring formed between two concentric circles -/
theorem area_of_ring (r₁ r₂ : ℝ) (h₁ : r₁ = 12) (h₂ : r₂ = 7) :
  π * r₁^2 - π * r₂^2 = 95 * π := by
  sorry

end area_of_ring_l192_19298


namespace fifth_to_third_grade_ratio_l192_19200

/-- Proves that the ratio of fifth-graders to third-graders is 1:2 given the conditions -/
theorem fifth_to_third_grade_ratio : 
  ∀ (third_graders fourth_graders fifth_graders : ℕ),
  third_graders = 20 →
  fourth_graders = 2 * third_graders →
  third_graders + fourth_graders + fifth_graders = 70 →
  fifth_graders.gcd third_graders * 2 = fifth_graders ∧ 
  fifth_graders.gcd third_graders * 1 = fifth_graders.gcd third_graders :=
by
  sorry

end fifth_to_third_grade_ratio_l192_19200


namespace neg_sufficient_but_not_necessary_l192_19227

-- Define the propositions
variable (p q : Prop)

-- Define the concept of sufficient but not necessary condition
def SufficientButNotNecessary (p q : Prop) : Prop :=
  (p → q) ∧ ¬(q → p)

-- State the theorem
theorem neg_sufficient_but_not_necessary (h : SufficientButNotNecessary p q) :
  SufficientButNotNecessary (¬q) (¬p) :=
sorry

end neg_sufficient_but_not_necessary_l192_19227


namespace gcd_divisibility_l192_19247

theorem gcd_divisibility (p q r s : ℕ+) 
  (h1 : Nat.gcd p.val q.val = 40)
  (h2 : Nat.gcd q.val r.val = 50)
  (h3 : Nat.gcd r.val s.val = 75)
  (h4 : 80 < Nat.gcd s.val p.val)
  (h5 : Nat.gcd s.val p.val < 120) :
  17 ∣ p.val :=
by sorry

end gcd_divisibility_l192_19247


namespace complex_equation_solution_l192_19245

theorem complex_equation_solution (z : ℂ) :
  (3 + 4*I) * z = 25 → z = 3 - 4*I := by sorry

end complex_equation_solution_l192_19245


namespace existence_of_p_and_q_l192_19202

theorem existence_of_p_and_q : ∃ (p q : ℝ), 
  ((p - 1)^2 - 4*q > 0) ∧ 
  ((p + 1)^2 - 4*q > 0) ∧ 
  (p^2 - 4*q ≤ 0) := by
sorry

end existence_of_p_and_q_l192_19202


namespace math_problem_time_calculation_l192_19270

theorem math_problem_time_calculation 
  (num_problems : ℕ) 
  (time_per_problem : ℕ) 
  (checking_time : ℕ) : 
  num_problems = 7 → 
  time_per_problem = 4 → 
  checking_time = 3 → 
  num_problems * time_per_problem + checking_time = 31 :=
by sorry

end math_problem_time_calculation_l192_19270


namespace arithmetic_sequence_general_term_l192_19232

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) ∧
  (a 2 + a 6) / 2 = 5 ∧
  (a 3 + a 7) / 2 = 7

/-- The general term of the arithmetic sequence satisfying the given conditions -/
theorem arithmetic_sequence_general_term (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  ∀ n : ℕ, a n = 2 * n - 3 := by
  sorry

end arithmetic_sequence_general_term_l192_19232


namespace rectangular_field_width_l192_19296

theorem rectangular_field_width (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = (7 / 5) * width →
  perimeter = 240 →
  2 * length + 2 * width = perimeter →
  width = 50 := by
sorry

end rectangular_field_width_l192_19296


namespace function_range_l192_19277

/-- The function f(x) = (x^2 - 2x - 3)(x^2 - 2x - 5) has a range of [-1, +∞) -/
theorem function_range (x : ℝ) : 
  let f : ℝ → ℝ := λ x => (x^2 - 2*x - 3) * (x^2 - 2*x - 5)
  ∃ (y : ℝ), y ≥ -1 ∧ ∃ (x : ℝ), f x = y :=
by sorry

end function_range_l192_19277


namespace circle_properties_l192_19217

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the center of the circle
def circle_center : ℝ × ℝ := (2, 0)

-- Define the radius of the circle
def circle_radius : ℝ := 2

-- Theorem statement
theorem circle_properties :
  ∀ (x y : ℝ), circle_equation x y ↔ 
    ((x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2) :=
sorry

end circle_properties_l192_19217


namespace purely_imaginary_complex_number_l192_19243

theorem purely_imaginary_complex_number (a : ℝ) : 
  (2 : ℂ) + Complex.I * ((1 : ℂ) - a + a * Complex.I) = Complex.I * (Complex.I.im * ((1 : ℂ) - a + a * Complex.I)) → a = 2 := by
  sorry

end purely_imaginary_complex_number_l192_19243


namespace notebook_boxes_l192_19248

theorem notebook_boxes (notebooks_per_box : ℕ) (total_notebooks : ℕ) (h1 : notebooks_per_box = 9) (h2 : total_notebooks = 27) :
  total_notebooks / notebooks_per_box = 3 :=
by
  sorry

end notebook_boxes_l192_19248


namespace work_completion_time_l192_19284

/-- Given two workers x and y who can complete a work in 10 and 15 days respectively,
    prove that they can complete the work together in 6 days. -/
theorem work_completion_time (x y : ℝ) (hx : x = 1 / 10) (hy : y = 1 / 15) :
  1 / (x + y) = 6 := by
  sorry

end work_completion_time_l192_19284


namespace smallest_third_term_of_geometric_progression_l192_19278

/-- Given an arithmetic progression with first term 7, prove that the smallest
    possible value for the third term of the resulting geometric progression is 3.752 -/
theorem smallest_third_term_of_geometric_progression
  (a b c : ℝ)
  (h_arithmetic : ∃ (d : ℝ), a = 7 ∧ b = 7 + d ∧ c = 7 + 2*d)
  (h_geometric : ∃ (r : ℝ), (7 : ℝ) * r = b - 3 ∧ (b - 3) * r = c + 15) :
  ∃ (x : ℝ), (∀ (y : ℝ), (7 : ℝ) * (b - 3) = (b - 3) * (c + 15) → c + 15 ≥ x) ∧ c + 15 ≥ 3.752 :=
sorry

end smallest_third_term_of_geometric_progression_l192_19278


namespace geometric_sequence_condition_l192_19263

-- Define a geometric sequence
def is_geometric_sequence (a b c d : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r

-- Theorem statement
theorem geometric_sequence_condition (a b c d : ℝ) :
  (is_geometric_sequence a b c d → a * d = b * c) ∧
  ∃ a b c d : ℝ, a * d = b * c ∧ ¬(is_geometric_sequence a b c d) :=
sorry

end geometric_sequence_condition_l192_19263


namespace monotonic_difference_increasing_decreasing_l192_19211

-- Define monotonic functions on ℝ
def Monotonic (f : ℝ → ℝ) : Prop := 
  ∀ x y, x ≤ y → f x ≤ f y ∨ f x ≥ f y

-- Define increasing function
def Increasing (f : ℝ → ℝ) : Prop := 
  ∀ x y, x < y → f x < f y

-- Define decreasing function
def Decreasing (f : ℝ → ℝ) : Prop := 
  ∀ x y, x < y → f x > f y

-- Theorem statement
theorem monotonic_difference_increasing_decreasing 
  (f g : ℝ → ℝ) 
  (hf : Monotonic f) (hg : Monotonic g) :
  (Increasing f ∧ Decreasing g → Increasing (fun x ↦ f x - g x)) ∧
  (Decreasing f ∧ Increasing g → Decreasing (fun x ↦ f x - g x)) := by
  sorry


end monotonic_difference_increasing_decreasing_l192_19211


namespace nell_final_baseball_cards_l192_19266

/-- Represents the number of cards Nell has --/
structure Cards where
  initial_baseball : ℕ
  initial_ace : ℕ
  final_ace : ℕ
  difference : ℕ

/-- Calculates the final number of baseball cards Nell has --/
def final_baseball_cards (c : Cards) : ℕ :=
  c.final_ace - c.difference

/-- Theorem stating that Nell's final baseball card count is 111 --/
theorem nell_final_baseball_cards :
  let c : Cards := {
    initial_baseball := 239,
    initial_ace := 38,
    final_ace := 376,
    difference := 265
  }
  final_baseball_cards c = 111 := by
  sorry

end nell_final_baseball_cards_l192_19266


namespace power_of_three_difference_l192_19287

theorem power_of_three_difference : 3^(1+2+3) - (3^1 + 3^2 + 3^3) = 690 := by
  sorry

end power_of_three_difference_l192_19287


namespace fraction_to_decimal_l192_19252

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end fraction_to_decimal_l192_19252


namespace interest_rate_calculation_l192_19283

theorem interest_rate_calculation (principal : ℝ) (interest_paid : ℝ) 
  (h1 : principal = 900) 
  (h2 : interest_paid = 729) : ∃ (rate : ℝ), 
  interest_paid = principal * rate * rate / 100 ∧ rate = 9 := by
  sorry

end interest_rate_calculation_l192_19283


namespace rates_sum_of_squares_l192_19229

/-- Represents the rates for running, bicycling, and roller-skating -/
structure Rates where
  running : ℕ
  bicycling : ℕ
  roller_skating : ℕ

/-- Tom's total distance -/
def tom_distance (r : Rates) : ℕ := 3 * r.running + 4 * r.bicycling + 2 * r.roller_skating

/-- Jerry's total distance -/
def jerry_distance (r : Rates) : ℕ := 3 * r.running + 6 * r.bicycling + 2 * r.roller_skating

/-- Sum of squares of rates -/
def sum_of_squares (r : Rates) : ℕ := r.running^2 + r.bicycling^2 + r.roller_skating^2

theorem rates_sum_of_squares :
  ∃ r : Rates,
    tom_distance r = 104 ∧
    jerry_distance r = 140 ∧
    sum_of_squares r = 440 := by
  sorry

end rates_sum_of_squares_l192_19229


namespace west_movement_l192_19239

-- Define a type for direction
inductive Direction
| East
| West

-- Define a function to represent movement
def movement (distance : ℤ) (direction : Direction) : ℤ :=
  match direction with
  | Direction.East => distance
  | Direction.West => -distance

-- State the theorem
theorem west_movement :
  (∀ (d : ℤ), movement d Direction.East = d) →
  (∀ (d : ℤ), movement d Direction.West = -d) →
  movement 5 Direction.West = -5 := by
  sorry

end west_movement_l192_19239


namespace test_scores_theorem_l192_19274

/-- Represents the test scores for three students -/
structure TestScores where
  alisson : ℕ
  jose : ℕ
  meghan : ℕ

/-- Calculates the total score for the three students -/
def totalScore (scores : TestScores) : ℕ :=
  scores.alisson + scores.jose + scores.meghan

/-- Theorem stating the total score for the three students -/
theorem test_scores_theorem (scores : TestScores) : totalScore scores = 210 :=
  by
  have h1 : scores.jose = scores.alisson + 40 := sorry
  have h2 : scores.meghan = scores.jose - 20 := sorry
  have h3 : scores.jose = 100 - 10 := sorry
  sorry

#check test_scores_theorem

end test_scores_theorem_l192_19274


namespace sequence_term_proof_l192_19212

/-- The sum of the first n terms of the sequence a_n -/
def S (n : ℕ) : ℚ := (2/3) * n^2 - (1/3) * n

/-- The nth term of the sequence a_n -/
def a (n : ℕ) : ℚ := (4/3) * n - 1

theorem sequence_term_proof (n : ℕ) (h : n > 0) : 
  a n = S n - S (n-1) :=
sorry

end sequence_term_proof_l192_19212


namespace smaller_to_larger_base_ratio_l192_19267

/-- An isosceles trapezoid with an inscribed equilateral triangle -/
structure IsoscelesTrapezoidWithTriangle where
  /-- Length of the smaller base (and side of the equilateral triangle) -/
  s : ℝ
  /-- Length of the larger base -/
  b : ℝ
  /-- s is positive -/
  s_pos : 0 < s
  /-- b is positive -/
  b_pos : 0 < b
  /-- The larger base is twice the length of a diagonal of the equilateral triangle -/
  diag_relation : b = 2 * s

/-- The ratio of the smaller base to the larger base is 1/2 -/
theorem smaller_to_larger_base_ratio 
  (t : IsoscelesTrapezoidWithTriangle) : t.s / t.b = 1 / 2 := by
  sorry


end smaller_to_larger_base_ratio_l192_19267


namespace eighth_group_frequency_l192_19250

theorem eighth_group_frequency 
  (total_sample : ℕ) 
  (num_groups : ℕ) 
  (freq_1 freq_2 freq_3 freq_4 : ℕ) 
  (sum_freq_5_to_7 : ℚ) :
  total_sample = 100 →
  num_groups = 8 →
  freq_1 = 15 →
  freq_2 = 17 →
  freq_3 = 11 →
  freq_4 = 13 →
  sum_freq_5_to_7 = 32 / 100 →
  (freq_1 + freq_2 + freq_3 + freq_4 + (sum_freq_5_to_7 * total_sample).num + 
    (total_sample - freq_1 - freq_2 - freq_3 - freq_4 - (sum_freq_5_to_7 * total_sample).num)) / total_sample = 1 →
  (total_sample - freq_1 - freq_2 - freq_3 - freq_4 - (sum_freq_5_to_7 * total_sample).num) / total_sample = 12 / 100 :=
by sorry

end eighth_group_frequency_l192_19250


namespace homeless_donation_calculation_l192_19285

theorem homeless_donation_calculation (total amount_first amount_second : ℝ) 
  (h1 : total = 900)
  (h2 : amount_first = 325)
  (h3 : amount_second = 260) :
  total - amount_first - amount_second = 315 :=
by sorry

end homeless_donation_calculation_l192_19285


namespace polygon_sides_l192_19203

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1260) : n = 9 := by
  sorry

end polygon_sides_l192_19203


namespace confectioner_pastries_l192_19237

theorem confectioner_pastries :
  ∀ (total_pastries : ℕ) 
    (regular_customers : ℕ) 
    (actual_customers : ℕ) 
    (pastry_difference : ℕ),
  regular_customers = 28 →
  actual_customers = 49 →
  pastry_difference = 6 →
  regular_customers * (total_pastries / regular_customers) = 
    actual_customers * (total_pastries / regular_customers - pastry_difference) →
  total_pastries = 1176 :=
by
  sorry

end confectioner_pastries_l192_19237


namespace same_solution_implies_a_plus_b_zero_power_l192_19238

theorem same_solution_implies_a_plus_b_zero_power (a b : ℝ) :
  (∃ (x y : ℝ), 4*x + 3*y = 11 ∧ a*x + b*y = -2 ∧ 3*x - 5*y = 1 ∧ b*x - a*y = 6) →
  (a + b)^2023 = 0 := by
  sorry

end same_solution_implies_a_plus_b_zero_power_l192_19238


namespace batting_average_increase_l192_19209

theorem batting_average_increase (current_average : ℚ) (matches_played : ℕ) (new_average : ℚ) : 
  current_average = 52 →
  matches_played = 12 →
  new_average = 54 →
  (new_average * (matches_played + 1) - current_average * matches_played : ℚ) = 78 := by
sorry

end batting_average_increase_l192_19209


namespace set_union_problem_l192_19236

def A (x : ℝ) : Set ℝ := {x^2, 2*x - 1, -4}
def B (x : ℝ) : Set ℝ := {x - 5, 1 - x, 9}

theorem set_union_problem (x : ℝ) :
  (∃ y, A y ∩ B y = {9}) →
  (∃ z, A z ∪ B z = {-4, -7, -8, 4, 9}) :=
by sorry

end set_union_problem_l192_19236


namespace b_subset_a_iff_a_eq_two_or_three_c_subset_a_iff_m_condition_l192_19271

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + (a-1) = 0}
def C (m : ℝ) : Set ℝ := {x | x^2 - m*x + 2 = 0}

-- Theorem 1: B is a subset of A if and only if a = 2 or a = 3
theorem b_subset_a_iff_a_eq_two_or_three :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ B a → x ∈ A) ↔ (a = 2 ∨ a = 3) :=
sorry

-- Theorem 2: C is a subset of A if and only if m = 3 or -2√2 < m < 2√2
theorem c_subset_a_iff_m_condition (m : ℝ) :
  C m ⊆ A ↔ (m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) :=
sorry

end b_subset_a_iff_a_eq_two_or_three_c_subset_a_iff_m_condition_l192_19271


namespace question_mark_value_l192_19279

theorem question_mark_value : ∃ (x : ℕ), x * 240 = 347 * 480 ∧ x = 694 := by
  sorry

end question_mark_value_l192_19279


namespace movie_length_ratio_l192_19242

/-- The lengths of favorite movies for Joyce, Michael, Nikki, and Ryn. -/
structure MovieLengths where
  michael : ℝ
  joyce : ℝ
  nikki : ℝ
  ryn : ℝ

/-- The conditions of the movie length problem. -/
def movieConditions (m : MovieLengths) : Prop :=
  m.joyce = m.michael + 2 ∧
  m.ryn = (4/5) * m.nikki ∧
  m.nikki = 30 ∧
  m.michael + m.joyce + m.nikki + m.ryn = 76

/-- The theorem stating that under the given conditions, 
    the ratio of Nikki's movie length to Michael's is 3:1. -/
theorem movie_length_ratio (m : MovieLengths) 
  (h : movieConditions m) : m.nikki / m.michael = 3 := by
  sorry

end movie_length_ratio_l192_19242


namespace treehouse_rope_length_l192_19223

theorem treehouse_rope_length : 
  let rope_lengths : List Nat := [24, 20, 14, 12, 18, 22]
  List.sum rope_lengths = 110 := by
  sorry

end treehouse_rope_length_l192_19223


namespace sufficient_not_necessary_l192_19224

theorem sufficient_not_necessary (a : ℝ) (h : a > 0) :
  (∀ a, a > 2 → a^a > a^2) ∧
  (∃ a, 0 < a ∧ a < 2 ∧ a^a > a^2) :=
by sorry

end sufficient_not_necessary_l192_19224


namespace range_of_m_l192_19234

theorem range_of_m (x y z : ℝ) (h1 : 6 * x = 3 * y + 12) (h2 : 6 * x = 2 * z) 
  (h3 : y ≥ 0) (h4 : z ≤ 9) : 
  let m := 2 * x + y - 3 * z
  ∀ m', m = m' → -19 ≤ m' ∧ m' ≤ -14 :=
by sorry

end range_of_m_l192_19234


namespace solution_sum_comparison_l192_19246

theorem solution_sum_comparison
  (a a' b b' c c' : ℝ)
  (ha : a ≠ 0)
  (ha' : a' ≠ 0) :
  (c' - b') / a' < (c - b) / a ↔
  (c - b) / a > (c' - b') / a' :=
by sorry

end solution_sum_comparison_l192_19246


namespace point_on_line_l192_19219

-- Define the points
def A : ℝ × ℝ := (2, -3)
def B : ℝ × ℝ := (4, 3)
def C : ℝ → ℝ × ℝ := λ m ↦ (5, m)

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  (q.2 - p.2) * (r.1 - q.1) = (r.2 - q.2) * (q.1 - p.1)

-- Theorem statement
theorem point_on_line (m : ℝ) :
  collinear A B (C m) → m = 6 := by sorry

end point_on_line_l192_19219


namespace quadratic_inequality_solution_l192_19255

theorem quadratic_inequality_solution (x : ℝ) :
  (-x^2 - 2*x + 3 ≤ 0) ↔ (x ≤ -3 ∨ x ≥ 1) := by
  sorry

end quadratic_inequality_solution_l192_19255


namespace distance_to_directrix_l192_19293

/-- The distance from a point on a parabola to its directrix -/
theorem distance_to_directrix (p : ℝ) (x y : ℝ) (h : y^2 = 2*p*x) :
  x + p/2 = 9/4 :=
sorry

end distance_to_directrix_l192_19293


namespace julie_school_year_earnings_l192_19205

/-- Julie's summer work details and school year work conditions -/
structure WorkDetails where
  summer_weeks : ℕ
  summer_hours_per_week : ℕ
  summer_earnings : ℕ
  school_weeks : ℕ
  school_hours_per_week : ℕ
  rate_increase : ℚ

/-- Calculate Julie's school year earnings based on her work details -/
def calculate_school_year_earnings (w : WorkDetails) : ℚ :=
  let summer_hourly_rate := w.summer_earnings / (w.summer_weeks * w.summer_hours_per_week)
  let school_hourly_rate := summer_hourly_rate * (1 + w.rate_increase)
  school_hourly_rate * w.school_weeks * w.school_hours_per_week

/-- Theorem stating that Julie's school year earnings are $3750 -/
theorem julie_school_year_earnings :
  let w : WorkDetails := {
    summer_weeks := 10,
    summer_hours_per_week := 40,
    summer_earnings := 4000,
    school_weeks := 30,
    school_hours_per_week := 10,
    rate_increase := 1/4
  }
  calculate_school_year_earnings w = 3750 := by sorry

end julie_school_year_earnings_l192_19205


namespace smallest_room_width_l192_19261

theorem smallest_room_width 
  (largest_width : ℝ) 
  (largest_length : ℝ) 
  (smallest_length : ℝ) 
  (area_difference : ℝ) :
  largest_width = 45 →
  largest_length = 30 →
  smallest_length = 8 →
  largest_width * largest_length - smallest_length * (largest_width * largest_length - area_difference) / smallest_length = 1230 →
  (largest_width * largest_length - area_difference) / smallest_length = 15 :=
by sorry

end smallest_room_width_l192_19261


namespace complex_number_quadrant_l192_19241

theorem complex_number_quadrant (i : ℂ) (h : i * i = -1) :
  let z : ℂ := 2 * i / (1 + i)
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end complex_number_quadrant_l192_19241


namespace unique_intersection_point_l192_19216

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if a point satisfies all four linear equations -/
def satisfiesAllEquations (p : Point2D) : Prop :=
  3 * p.x - 2 * p.y = 12 ∧
  2 * p.x + 5 * p.y = -1 ∧
  p.x + 4 * p.y = 8 ∧
  5 * p.x - 3 * p.y = 15

/-- Theorem stating that there exists exactly one point satisfying all equations -/
theorem unique_intersection_point :
  ∃! p : Point2D, satisfiesAllEquations p :=
sorry


end unique_intersection_point_l192_19216


namespace negation_relationship_l192_19228

theorem negation_relationship (x : ℝ) : 
  (¬(0 < x ∧ x < 2) → ¬(1/x ≥ 1)) ∧ ¬(¬(1/x ≥ 1) → ¬(0 < x ∧ x < 2)) := by
  sorry

end negation_relationship_l192_19228


namespace max_roses_theorem_l192_19269

/-- Represents the pricing options for roses -/
structure RosePricing where
  individual : Nat  -- Price in cents for an individual rose
  dozen : Nat       -- Price in cents for a dozen roses
  two_dozen : Nat   -- Price in cents for two dozen roses

/-- Calculates the maximum number of roses that can be purchased with a given budget -/
def max_roses_purchasable (pricing : RosePricing) (budget : Nat) : Nat :=
  sorry

/-- The theorem stating the maximum number of roses purchasable with the given pricing and budget -/
theorem max_roses_theorem (pricing : RosePricing) (budget : Nat) :
  pricing.individual = 630 →
  pricing.dozen = 3600 →
  pricing.two_dozen = 5000 →
  budget = 68000 →
  max_roses_purchasable pricing budget = 316 :=
sorry

end max_roses_theorem_l192_19269


namespace solution_to_equation_l192_19251

theorem solution_to_equation (x y : ℝ) : 
  x^2 + (1 - y)^2 + (x - y)^2 = 1/3 ↔ x = 1/3 ∧ y = 2/3 := by
  sorry

end solution_to_equation_l192_19251


namespace simplify_expression_l192_19299

theorem simplify_expression (n : ℕ) : (2^(n+4) - 3*(2^n)) / (2*(2^(n+3))) = 13/16 := by
  sorry

end simplify_expression_l192_19299


namespace smallest_addition_for_divisibility_l192_19290

theorem smallest_addition_for_divisibility (n : ℕ) (h : n = 8261955) :
  ∃ x : ℕ, x = 2 ∧ 
  (∀ y : ℕ, y < x → ¬(11 ∣ (n + y))) ∧
  (11 ∣ (n + x)) := by
  sorry

end smallest_addition_for_divisibility_l192_19290


namespace smallest_n_divisible_by_2019_l192_19286

theorem smallest_n_divisible_by_2019 : ∃ (n : ℕ), n = 2000 ∧ 
  (∀ (m : ℕ), m < n → ¬(2019 ∣ (m^2 + 20*m + 19))) ∧ 
  (2019 ∣ (n^2 + 20*n + 19)) := by
  sorry

end smallest_n_divisible_by_2019_l192_19286


namespace total_points_is_265_l192_19206

/-- Given information about Paul's point assignment in the first quarter -/
structure PointAssignment where
  homework_points : ℕ
  quiz_points : ℕ
  test_points : ℕ
  hw_quiz_relation : quiz_points = homework_points + 5
  quiz_test_relation : test_points = 4 * quiz_points
  hw_given : homework_points = 40

/-- The total points assigned by Paul in the first quarter -/
def total_points (pa : PointAssignment) : ℕ :=
  pa.homework_points + pa.quiz_points + pa.test_points

/-- Theorem stating that the total points assigned is 265 -/
theorem total_points_is_265 (pa : PointAssignment) : total_points pa = 265 := by
  sorry

end total_points_is_265_l192_19206


namespace system_solution_l192_19214

theorem system_solution :
  ∃ (a b c d : ℝ),
    (a + c = 4 ∧
     a * c + b + d = 6 ∧
     a * d + b * c = 5 ∧
     b * d = 2) ∧
    ((a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 1) ∨
     (a = 1 ∧ b = 1 ∧ c = 3 ∧ d = 2)) :=
by sorry

end system_solution_l192_19214


namespace partnership_investment_time_l192_19264

/-- Represents the investment and profit scenario of two partners -/
structure PartnershipScenario where
  /-- Ratio of partner p's investment to partner q's investment -/
  investment_ratio_p_q : Rat
  /-- Ratio of partner p's profit to partner q's profit -/
  profit_ratio_p_q : Rat
  /-- Number of months partner p invested -/
  p_investment_time : ℕ
  /-- Number of months partner q invested -/
  q_investment_time : ℕ

/-- Theorem stating the relationship between investment ratios, profit ratios, and investment times -/
theorem partnership_investment_time 
  (scenario : PartnershipScenario) 
  (h1 : scenario.investment_ratio_p_q = 7 / 5)
  (h2 : scenario.profit_ratio_p_q = 7 / 10)
  (h3 : scenario.p_investment_time = 7) :
  scenario.q_investment_time = 14 := by
  sorry

#check partnership_investment_time

end partnership_investment_time_l192_19264


namespace min_y_over_x_on_ellipse_l192_19253

theorem min_y_over_x_on_ellipse :
  ∀ x y : ℝ, 4 * (x - 2)^2 + y^2 = 4 →
  ∃ k : ℝ, k = -2/3 * Real.sqrt 3 ∧ ∀ z : ℝ, z = y / x → z ≥ k := by
  sorry

end min_y_over_x_on_ellipse_l192_19253


namespace joe_oranges_count_l192_19220

/-- The number of boxes Joe has for oranges -/
def num_boxes : ℕ := 9

/-- The number of oranges required in each box -/
def oranges_per_box : ℕ := 5

/-- The total number of oranges Joe has -/
def total_oranges : ℕ := num_boxes * oranges_per_box

theorem joe_oranges_count : total_oranges = 45 := by
  sorry

end joe_oranges_count_l192_19220


namespace divisible_by_four_l192_19254

theorem divisible_by_four (x : Nat) : 
  x < 10 → (3280 + x).mod 4 = 0 ↔ x = 0 ∨ x = 2 ∨ x = 4 ∨ x = 6 ∨ x = 8 := by
  sorry

end divisible_by_four_l192_19254


namespace total_food_items_donated_l192_19204

/-- The total number of food items donated by five companies given specific donation rules -/
theorem total_food_items_donated (foster_chickens : ℕ) : foster_chickens = 45 →
  ∃ (american_water hormel_chickens boudin_chickens delmonte_water : ℕ),
    american_water = 2 * foster_chickens ∧
    hormel_chickens = 3 * foster_chickens ∧
    boudin_chickens = hormel_chickens / 3 ∧
    delmonte_water = american_water - 30 ∧
    (boudin_chickens + delmonte_water) % 7 = 0 ∧
    foster_chickens = (hormel_chickens + boudin_chickens) / 2 ∧
    foster_chickens + american_water + hormel_chickens + boudin_chickens + delmonte_water = 375 :=
by sorry

end total_food_items_donated_l192_19204


namespace percentage_increase_proof_l192_19249

def old_cost : ℝ := 150
def new_cost : ℝ := 195

theorem percentage_increase_proof :
  (new_cost - old_cost) / old_cost * 100 = 30 := by sorry

end percentage_increase_proof_l192_19249


namespace anna_apples_total_l192_19210

def apples_eaten (tuesday wednesday thursday : ℕ) : ℕ :=
  tuesday + wednesday + thursday

theorem anna_apples_total :
  ∀ (tuesday : ℕ),
    tuesday = 4 →
    ∀ (wednesday thursday : ℕ),
      wednesday = 2 * tuesday →
      thursday = tuesday / 2 →
      apples_eaten tuesday wednesday thursday = 14 := by
sorry

end anna_apples_total_l192_19210


namespace linear_function_characterization_l192_19273

theorem linear_function_characterization (f : ℕ → ℕ) :
  (∀ x y : ℕ, f (x + y) = f x + f y) →
  ∃ a : ℕ, ∀ x : ℕ, f x = a * x :=
by sorry

end linear_function_characterization_l192_19273


namespace contrapositive_isosceles_equal_angles_l192_19294

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define properties of a triangle
def Triangle.isIsosceles (t : Triangle) : Prop := sorry
def Triangle.hasEqualInteriorAngles (t : Triangle) : Prop := sorry

-- State the theorem
theorem contrapositive_isosceles_equal_angles (t : Triangle) :
  (¬(t.isIsosceles) → ¬(t.hasEqualInteriorAngles)) ↔
  (t.hasEqualInteriorAngles → t.isIsosceles) := by sorry

end contrapositive_isosceles_equal_angles_l192_19294


namespace smallest_n_for_quadruplets_l192_19230

theorem smallest_n_for_quadruplets : ∃ (n : ℕ+), 
  (∃! (quad_count : ℕ), quad_count = 154000 ∧ 
    (∃ (S : Finset (ℕ+ × ℕ+ × ℕ+ × ℕ+)), 
      Finset.card S = quad_count ∧
      ∀ (a b c d : ℕ+), (a, b, c, d) ∈ S ↔ 
        (Nat.gcd a.val (Nat.gcd b.val (Nat.gcd c.val d.val)) = 154 ∧
         Nat.lcm a.val (Nat.lcm b.val (Nat.lcm c.val d.val)) = n.val))) ∧
  (∀ (m : ℕ+), m < n →
    ¬∃ (quad_count : ℕ), quad_count = 154000 ∧
      (∃ (S : Finset (ℕ+ × ℕ+ × ℕ+ × ℕ+)),
        Finset.card S = quad_count ∧
        ∀ (a b c d : ℕ+), (a, b, c, d) ∈ S ↔
          (Nat.gcd a.val (Nat.gcd b.val (Nat.gcd c.val d.val)) = 154 ∧
           Nat.lcm a.val (Nat.lcm b.val (Nat.lcm c.val d.val)) = m.val))) ∧
  n = 25520328 := by
  sorry

end smallest_n_for_quadruplets_l192_19230


namespace largest_divisor_of_n4_minus_n_l192_19235

/-- A positive integer greater than 1 is composite if it has a factor other than 1 and itself. -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ k, 1 < k ∧ k < n ∧ n % k = 0

/-- For all composite integers n, 6 divides n^4 - n and is the largest such divisor. -/
theorem largest_divisor_of_n4_minus_n (n : ℕ) (h : IsComposite n) :
    (6 ∣ n^4 - n) ∧ ∀ m : ℕ, m > 6 → ¬(∀ k : ℕ, IsComposite k → (m ∣ k^4 - k)) :=
by sorry

end largest_divisor_of_n4_minus_n_l192_19235


namespace parallel_vectors_x_value_l192_19259

/-- Given two vectors a and b in ℝ², prove that if a = (3,1) and b = (x,-1) are parallel, then x = -3 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![3, 1]
  let b : Fin 2 → ℝ := ![x, -1]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  x = -3 :=
by sorry

end parallel_vectors_x_value_l192_19259


namespace archer_arrow_recovery_percentage_l192_19280

-- Define the given constants
def shots_per_day : ℕ := 200
def days_per_week : ℕ := 4
def arrow_cost : ℚ := 5.5
def team_payment_percentage : ℚ := 0.7
def archer_weekly_spend : ℚ := 1056

-- Define the theorem
theorem archer_arrow_recovery_percentage :
  let total_shots := shots_per_day * days_per_week
  let total_cost := archer_weekly_spend / (1 - team_payment_percentage)
  let arrows_bought := total_cost / arrow_cost
  let arrows_recovered := total_shots - arrows_bought
  arrows_recovered / total_shots = 1/5 := by
sorry

end archer_arrow_recovery_percentage_l192_19280


namespace triangle_pieces_count_l192_19222

/-- The number of rods in the nth row of the triangle -/
def rods_in_row (n : ℕ) : ℕ := 4 * n

/-- The total number of rods in a triangle with n rows -/
def total_rods (n : ℕ) : ℕ := (n * (n + 1) / 2) * 4

/-- The number of connectors in a triangle with n rows of rods -/
def total_connectors (n : ℕ) : ℕ := ((n + 1) * (n + 2)) / 2

/-- The total number of pieces (rods and connectors) in a triangle with n rows -/
def total_pieces (n : ℕ) : ℕ := total_rods n + total_connectors n

theorem triangle_pieces_count :
  total_pieces 10 = 286 := by sorry

end triangle_pieces_count_l192_19222


namespace leon_payment_l192_19256

/-- The total amount Leon paid for toy organizers, gaming chairs, and delivery fee. -/
def total_paid (toy_organizer_sets : ℕ) (toy_organizer_price : ℚ) 
                (gaming_chairs : ℕ) (gaming_chair_price : ℚ) 
                (delivery_fee_percentage : ℚ) : ℚ :=
  let total_sales := toy_organizer_sets * toy_organizer_price + gaming_chairs * gaming_chair_price
  let delivery_fee := delivery_fee_percentage * total_sales
  total_sales + delivery_fee

/-- Theorem stating that Leon paid $420 in total -/
theorem leon_payment : 
  total_paid 3 78 2 83 (5/100) = 420 := by
  sorry

end leon_payment_l192_19256


namespace total_work_experience_approx_l192_19244

def daysPerYear : ℝ := 365
def daysPerMonth : ℝ := 30.44
def daysPerWeek : ℝ := 7

def bartenderYears : ℝ := 9
def bartenderMonths : ℝ := 8

def managerYears : ℝ := 3
def managerMonths : ℝ := 6

def salesMonths : ℝ := 11

def coordinatorYears : ℝ := 2
def coordinatorMonths : ℝ := 5
def coordinatorWeeks : ℝ := 3

def totalWorkExperience : ℝ :=
  (bartenderYears * daysPerYear + bartenderMonths * daysPerMonth) +
  (managerYears * daysPerYear + managerMonths * daysPerMonth) +
  (salesMonths * daysPerMonth) +
  (coordinatorYears * daysPerYear + coordinatorMonths * daysPerMonth + coordinatorWeeks * daysPerWeek)

theorem total_work_experience_approx :
  ⌊totalWorkExperience⌋ = 6044 := by sorry

end total_work_experience_approx_l192_19244


namespace derivative_f_at_pi_l192_19282

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x * Real.sin x

theorem derivative_f_at_pi :
  deriv f π = -Real.sqrt π := by sorry

end derivative_f_at_pi_l192_19282


namespace line_perpendicular_to_plane_l192_19265

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem line_perpendicular_to_plane 
  (l m : Line) (α : Plane) : 
  perpendicular l α → parallel l m → perpendicular m α := by
  sorry

end line_perpendicular_to_plane_l192_19265


namespace prob_man_satisfied_correct_expected_satisfied_men_correct_l192_19268

/-- Represents the number of men in the seating arrangement -/
def num_men : ℕ := 50

/-- Represents the number of women in the seating arrangement -/
def num_women : ℕ := 50

/-- Represents the total number of people in the seating arrangement -/
def total_people : ℕ := num_men + num_women

/-- Represents the probability of a specific man being satisfied -/
def prob_man_satisfied : ℚ := 25 / 33

/-- Represents the expected number of satisfied men -/
def expected_satisfied_men : ℚ := 1250 / 33

/-- Theorem stating the probability of a specific man being satisfied -/
theorem prob_man_satisfied_correct : 
  prob_man_satisfied = 1 - (num_men - 1) / (total_people - 1) * (num_men - 2) / (total_people - 2) :=
sorry

/-- Theorem stating the expected number of satisfied men -/
theorem expected_satisfied_men_correct : 
  expected_satisfied_men = num_men * prob_man_satisfied :=
sorry

end prob_man_satisfied_correct_expected_satisfied_men_correct_l192_19268


namespace sum_of_special_numbers_l192_19240

-- Define the properties for m and n
def has_two_divisors (x : ℕ) : Prop := (Finset.filter (· ∣ x) (Finset.range (x + 1))).card = 2

def has_four_divisors (x : ℕ) : Prop := (Finset.filter (· ∣ x) (Finset.range (x + 1))).card = 4

def is_smallest_with_two_divisors (m : ℕ) : Prop :=
  has_two_divisors m ∧ ∀ k < m, ¬has_two_divisors k

def is_largest_under_200_with_four_divisors (n : ℕ) : Prop :=
  n < 200 ∧ has_four_divisors n ∧ ∀ k > n, k < 200 → ¬has_four_divisors k

-- State the theorem
theorem sum_of_special_numbers :
  ∃ (m n : ℕ), is_smallest_with_two_divisors m ∧ is_largest_under_200_with_four_divisors n ∧ m + n = 127 := by
  sorry

end sum_of_special_numbers_l192_19240


namespace sally_bread_consumption_l192_19218

theorem sally_bread_consumption :
  let saturday_sandwiches : ℕ := 2
  let sunday_sandwiches : ℕ := 1
  let bread_per_sandwich : ℕ := 2
  let total_sandwiches := saturday_sandwiches + sunday_sandwiches
  let total_bread := total_sandwiches * bread_per_sandwich
  total_bread = 6 := by
  sorry

end sally_bread_consumption_l192_19218


namespace ruby_starting_lineup_combinations_l192_19221

def total_players : ℕ := 15
def all_stars : ℕ := 5
def starting_lineup : ℕ := 7

theorem ruby_starting_lineup_combinations :
  Nat.choose (total_players - all_stars) (starting_lineup - all_stars) = 45 := by
  sorry

end ruby_starting_lineup_combinations_l192_19221


namespace cosine_rule_with_ratio_l192_19208

theorem cosine_rule_with_ratio (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_ratio : ∃ (k : ℝ), a = 2*k ∧ b = 3*k ∧ c = 4*k) : 
  (a^2 + b^2 - c^2) / (2*a*b) = -1/4 := by
  sorry

end cosine_rule_with_ratio_l192_19208


namespace sum_and_subtract_l192_19215

theorem sum_and_subtract : (2345 + 3452 + 4523 + 5234) - 1234 = 14320 := by
  sorry

end sum_and_subtract_l192_19215


namespace equation_solution_set_l192_19201

theorem equation_solution_set : 
  {(x, y) : ℕ × ℕ | 3 * x^2 + 2 * 9^y = x * (4^(y + 1) - 1)} = 
  {(2, 1), (3, 1), (3, 2), (18, 2)} :=
by sorry

end equation_solution_set_l192_19201


namespace weight_loss_days_l192_19207

/-- The number of days it takes to lose a given amount of weight, given daily calorie intake, burn rate, and calories needed to lose one pound. -/
def days_to_lose_weight (calories_eaten : ℕ) (calories_burned : ℕ) (calories_per_pound : ℕ) (pounds_to_lose : ℕ) : ℕ :=
  let daily_deficit := calories_burned - calories_eaten
  let days_per_pound := calories_per_pound / daily_deficit
  days_per_pound * pounds_to_lose

/-- Theorem stating that it takes 80 days to lose 10 pounds under given conditions -/
theorem weight_loss_days : days_to_lose_weight 1800 2300 4000 10 = 80 := by
  sorry

#eval days_to_lose_weight 1800 2300 4000 10

end weight_loss_days_l192_19207


namespace only_cylinder_produces_quadrilateral_section_l192_19275

-- Define the types of geometric solids
inductive GeometricSolid
  | Cone
  | Sphere
  | Cylinder

-- Define a function that checks if a geometric solid can produce a quadrilateral section
def can_produce_quadrilateral_section (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.Cylinder => True
  | _ => False

-- Theorem statement
theorem only_cylinder_produces_quadrilateral_section :
  ∀ (solid : GeometricSolid),
    can_produce_quadrilateral_section solid ↔ solid = GeometricSolid.Cylinder :=
by
  sorry


end only_cylinder_produces_quadrilateral_section_l192_19275


namespace intersection_M_N_l192_19295

def M : Set ℝ := {x | x^2 + 3*x + 2 > 0}
def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := by sorry

end intersection_M_N_l192_19295


namespace walter_coins_percentage_l192_19276

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "nickel" => 5
  | "half-dollar" => 50
  | _ => 0

/-- Calculates the total value of coins in cents -/
def total_value (pennies nickels half_dollars : ℕ) : ℕ :=
  pennies * coin_value "penny" +
  nickels * coin_value "nickel" +
  half_dollars * coin_value "half-dollar"

/-- Converts cents to percentage of a dollar -/
def cents_to_percentage (cents : ℕ) : ℚ :=
  (cents : ℚ) / 100

theorem walter_coins_percentage :
  cents_to_percentage (total_value 3 2 1) = 63 / 100 := by
  sorry

end walter_coins_percentage_l192_19276


namespace car_tank_capacity_l192_19262

def distance_to_home : ℝ := 220
def fuel_efficiency : ℝ := 20
def additional_distance : ℝ := 100

theorem car_tank_capacity :
  let total_distance := distance_to_home + additional_distance
  let tank_capacity := total_distance / fuel_efficiency
  tank_capacity = 16 := by sorry

end car_tank_capacity_l192_19262


namespace integer_solutions_of_quadratic_equation_l192_19213

theorem integer_solutions_of_quadratic_equation :
  ∀ x y : ℤ, x^2 = y^2 * (x + y^4 + 2*y^2) →
  (x = 0 ∧ y = 0) ∨ (x = 12 ∧ y = 2) ∨ (x = -8 ∧ y = 2) :=
by sorry

end integer_solutions_of_quadratic_equation_l192_19213


namespace max_length_sequence_l192_19289

def sequence_term (n : ℕ) (x : ℕ) : ℤ :=
  match n with
  | 0 => 5000
  | 1 => x
  | n + 2 => sequence_term n x - sequence_term (n + 1) x

def is_positive (n : ℤ) : Prop := n > 0

theorem max_length_sequence (x : ℕ) : 
  (∀ n : ℕ, n < 11 → is_positive (sequence_term n x)) ∧ 
  ¬(is_positive (sequence_term 11 x)) ↔ 
  x = 3089 :=
sorry

end max_length_sequence_l192_19289


namespace loan_balance_years_l192_19226

theorem loan_balance_years (c V t n : ℝ) (hc : c > 0) (hV : V > 0) (ht : t > -1) :
  V = c / (1 + t)^(3 * n) → n = (Real.log (c / V)) / (3 * Real.log (1 + t)) := by
  sorry

end loan_balance_years_l192_19226


namespace percentage_increase_l192_19297

theorem percentage_increase (original : ℝ) (new : ℝ) :
  original = 30 →
  new = 40 →
  (new - original) / original = 1 / 3 := by
sorry

end percentage_increase_l192_19297


namespace geometric_sequence_properties_l192_19231

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_properties
  (a : ℕ → ℝ)
  (h : is_geometric_sequence a) :
  (is_geometric_sequence (fun n ↦ (a n)^2)) ∧
  (∀ k : ℝ, k ≠ 0 → is_geometric_sequence (fun n ↦ k * a n)) ∧
  (is_geometric_sequence (fun n ↦ 1 / (a n))) :=
sorry

end geometric_sequence_properties_l192_19231


namespace no_eulerian_path_in_picture_graph_l192_19288

/-- A graph representing the regions in the picture --/
structure PictureGraph where
  vertices : Finset (Fin 6)
  edges : Finset (Fin 6 × Fin 6)
  adjacent : (a b : Fin 6) → (a, b) ∈ edges → a ≠ b

/-- The degree of a vertex in the graph --/
def degree (G : PictureGraph) (v : Fin 6) : Nat :=
  (G.edges.filter (fun e => e.1 = v ∨ e.2 = v)).card

/-- An Eulerian path visits each edge exactly once --/
def hasEulerianPath (G : PictureGraph) : Prop :=
  ∃ path : List (Fin 6), path.length = G.edges.card + 1 ∧
    (∀ e ∈ G.edges, ∃ i, path[i]? = some e.1 ∧ path[i+1]? = some e.2)

/-- The main theorem: No Eulerian path exists in this graph --/
theorem no_eulerian_path_in_picture_graph (G : PictureGraph) 
  (h1 : ∃ v1 v2 v3 : Fin 6, v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧ 
    degree G v1 = 5 ∧ degree G v2 = 5 ∧ degree G v3 = 9) :
  ¬ hasEulerianPath G := by
  sorry


end no_eulerian_path_in_picture_graph_l192_19288


namespace square_ratio_proof_l192_19292

theorem square_ratio_proof : 
  ∀ (s₁ s₂ : ℝ), s₁ > 0 ∧ s₂ > 0 →
  (s₁^2 / s₂^2 = 45 / 64) →
  ∃ (a b c : ℕ), (a > 0 ∧ b > 0 ∧ c > 0) ∧
  (s₁ / s₂ = (a : ℝ) * Real.sqrt b / c) ∧
  (a + b + c = 16) :=
by sorry

end square_ratio_proof_l192_19292


namespace day_care_toddlers_l192_19260

/-- Given the initial ratio of toddlers to infants and the ratio after more infants join,
    prove the number of toddlers -/
theorem day_care_toddlers (t i : ℕ) (h1 : t * 3 = i * 7) (h2 : t * 5 = (i + 12) * 7) : t = 42 := by
  sorry

end day_care_toddlers_l192_19260
