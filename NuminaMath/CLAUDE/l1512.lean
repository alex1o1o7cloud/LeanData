import Mathlib

namespace NUMINAMATH_CALUDE_sequence_ratio_density_l1512_151239

theorem sequence_ratio_density (a : ℕ → ℕ) 
  (h : ∀ n : ℕ, 0 < a (n + 1) - a n ∧ a (n + 1) - a n < Real.sqrt (a n)) :
  ∀ x y : ℝ, 0 < x → x < y → y < 1 → 
  ∃ k m : ℕ, 0 < k ∧ 0 < m ∧ x < (a k : ℝ) / (a m : ℝ) ∧ (a k : ℝ) / (a m : ℝ) < y :=
by sorry

end NUMINAMATH_CALUDE_sequence_ratio_density_l1512_151239


namespace NUMINAMATH_CALUDE_largest_divisor_of_three_consecutive_even_integers_l1512_151267

theorem largest_divisor_of_three_consecutive_even_integers :
  ∃ (d : ℕ), d = 24 ∧ 
  (∀ (n : ℕ), n > 0 → d ∣ (2*n) * (2*n + 2) * (2*n + 4)) ∧
  (∀ (k : ℕ), k > d → ∃ (m : ℕ), m > 0 ∧ ¬(k ∣ (2*m) * (2*m + 2) * (2*m + 4))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_three_consecutive_even_integers_l1512_151267


namespace NUMINAMATH_CALUDE_easter_egg_hunt_friends_l1512_151266

/-- Proves the number of friends at Shonda's Easter egg hunt --/
theorem easter_egg_hunt_friends (baskets : ℕ) (eggs_per_basket : ℕ) (eggs_per_person : ℕ)
  (shonda_kids : ℕ) (shonda : ℕ) (other_adults : ℕ) :
  baskets = 15 →
  eggs_per_basket = 12 →
  eggs_per_person = 9 →
  shonda_kids = 2 →
  shonda = 1 →
  other_adults = 7 →
  baskets * eggs_per_basket / eggs_per_person - (shonda_kids + shonda + other_adults) = 10 :=
by
  sorry


end NUMINAMATH_CALUDE_easter_egg_hunt_friends_l1512_151266


namespace NUMINAMATH_CALUDE_sugar_replacement_theorem_l1512_151299

/-- Calculates the final sugar percentage when replacing part of a solution --/
def finalSugarPercentage (originalPercent : Float) (replacedFraction : Float) (newPercent : Float) : Float :=
  let remainingFraction := 1 - replacedFraction
  let remainingSugar := originalPercent * remainingFraction
  let addedSugar := newPercent * replacedFraction
  remainingSugar + addedSugar

/-- Theorem stating the final sugar percentage after partial replacement --/
theorem sugar_replacement_theorem :
  finalSugarPercentage 10 0.25 26.000000000000007 = 14.000000000000002 := by
  sorry

end NUMINAMATH_CALUDE_sugar_replacement_theorem_l1512_151299


namespace NUMINAMATH_CALUDE_arithmetic_mean_proof_l1512_151279

theorem arithmetic_mean_proof (x a b : ℝ) (hx : x ≠ b ∧ x ≠ -b) :
  (1/2) * ((x + a + b)/(x + b) + (x - a - b)/(x - b)) = 1 - a*b/(x^2 - b^2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_proof_l1512_151279


namespace NUMINAMATH_CALUDE_polynomial_value_symmetry_l1512_151250

theorem polynomial_value_symmetry (a b c : ℝ) :
  ((-3)^5 * a + (-3)^3 * b + (-3) * c - 5 = 7) →
  (3^5 * a + 3^3 * b + 3 * c - 5 = -17) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_symmetry_l1512_151250


namespace NUMINAMATH_CALUDE_real_estate_calendar_problem_l1512_151229

/-- Proves that given the conditions of the real estate problem, the number of calendars ordered is 200 -/
theorem real_estate_calendar_problem :
  ∀ (calendar_cost date_book_cost : ℚ) (total_items : ℕ) (total_spent : ℚ) (calendars date_books : ℕ),
    calendar_cost = 3/4 →
    date_book_cost = 1/2 →
    total_items = 500 →
    total_spent = 300 →
    calendars + date_books = total_items →
    calendar_cost * calendars + date_book_cost * date_books = total_spent →
    calendars = 200 := by
  sorry

end NUMINAMATH_CALUDE_real_estate_calendar_problem_l1512_151229


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l1512_151228

theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ),
  x^2 / 121 - y^2 / 49 = 1 →
  ∃ (v1 v2 : ℝ × ℝ),
    v1 ∈ {p : ℝ × ℝ | p.1^2 / 121 - p.2^2 / 49 = 1} ∧
    v2 ∈ {p : ℝ × ℝ | p.1^2 / 121 - p.2^2 / 49 = 1} ∧
    v1 ≠ v2 ∧
    ∀ (v : ℝ × ℝ),
      v ∈ {p : ℝ × ℝ | p.1^2 / 121 - p.2^2 / 49 = 1} →
      v.2 = 0 →
      v = v1 ∨ v = v2 ∧
    Real.sqrt ((v1.1 - v2.1)^2 + (v1.2 - v2.2)^2) = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l1512_151228


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1512_151211

theorem fraction_evaluation : (1 - 2/5) / (1 - 1/4) = 4/5 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1512_151211


namespace NUMINAMATH_CALUDE_equal_paper_distribution_l1512_151210

theorem equal_paper_distribution (total_sheets : ℕ) (num_friends : ℕ) (sheets_per_friend : ℕ) :
  total_sheets = 15 →
  num_friends = 3 →
  total_sheets = num_friends * sheets_per_friend →
  sheets_per_friend = 5 := by
  sorry

end NUMINAMATH_CALUDE_equal_paper_distribution_l1512_151210


namespace NUMINAMATH_CALUDE_original_salary_proof_l1512_151231

/-- Given a 6% raise resulting in a new salary of $530, prove that the original salary was $500. -/
theorem original_salary_proof (original_salary : ℝ) : 
  original_salary * 1.06 = 530 → original_salary = 500 := by
  sorry

end NUMINAMATH_CALUDE_original_salary_proof_l1512_151231


namespace NUMINAMATH_CALUDE_no_solution_l1512_151269

/-- P(n) denotes the greatest prime factor of n -/
def P (n : ℕ) : ℕ := sorry

/-- The theorem states that there are no positive integers n > 1 satisfying both conditions -/
theorem no_solution :
  ¬ ∃ (n : ℕ), n > 1 ∧ (P n : ℝ) = Real.sqrt n ∧ (P (n + 60) : ℝ) = Real.sqrt (n + 60) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_l1512_151269


namespace NUMINAMATH_CALUDE_less_amount_proof_l1512_151259

theorem less_amount_proof (a b c d x : ℝ) 
  (h1 : a - b = c + d + 9)
  (h2 : a + b = c - d - x)
  (h3 : a - c = 3) :
  x = 3 := by sorry

end NUMINAMATH_CALUDE_less_amount_proof_l1512_151259


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1512_151253

/-- An arithmetic sequence {a_n} with a_1 = 1 and a_3 = a_2^2 - 4 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ 
  a 3 = (a 2)^2 - 4 ∧
  ∀ n m : ℕ, n < m → a n < a m

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℝ) 
  (h : ArithmeticSequence a) :
  ∀ n : ℕ, a n = 2 * n - 1 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1512_151253


namespace NUMINAMATH_CALUDE_find_n_l1512_151258

theorem find_n (n : ℕ) 
  (h1 : Nat.gcd n 180 = 12) 
  (h2 : Nat.lcm n 180 = 720) : 
  n = 48 := by
sorry

end NUMINAMATH_CALUDE_find_n_l1512_151258


namespace NUMINAMATH_CALUDE_solve_complex_equation_l1512_151230

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (z : ℂ) : Prop := z * (1 + i) = 2 + i

-- Theorem statement
theorem solve_complex_equation :
  ∀ z : ℂ, equation z → z = (3/2 : ℝ) - (1/2 : ℝ) * i :=
by
  sorry

end NUMINAMATH_CALUDE_solve_complex_equation_l1512_151230


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1512_151242

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | 0 ≤ x}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 ≤ x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1512_151242


namespace NUMINAMATH_CALUDE_math_homework_percentage_l1512_151243

/-- Proves that the percentage of time spent on math homework is 30%, given the total homework time,
    time spent on science, and time spent on other subjects. -/
theorem math_homework_percentage
  (total_time : ℝ)
  (science_percentage : ℝ)
  (other_subjects_time : ℝ)
  (h1 : total_time = 150)
  (h2 : science_percentage = 0.4)
  (h3 : other_subjects_time = 45)
  : (total_time - science_percentage * total_time - other_subjects_time) / total_time = 0.3 := by
  sorry

#check math_homework_percentage

end NUMINAMATH_CALUDE_math_homework_percentage_l1512_151243


namespace NUMINAMATH_CALUDE_sum_remainder_seven_l1512_151204

theorem sum_remainder_seven (n : ℤ) : (7 - n + (n + 3)) % 7 = 3 := by sorry

end NUMINAMATH_CALUDE_sum_remainder_seven_l1512_151204


namespace NUMINAMATH_CALUDE_min_quotient_is_20_5_l1512_151260

/-- Represents a three-digit number with digits a, b, and c -/
structure ThreeDigitNumber where
  a : ℕ
  b : ℕ
  c : ℕ
  a_nonzero : a > 0
  b_nonzero : b > 0
  c_nonzero : c > 0
  all_different : a ≠ b ∧ b ≠ c ∧ a ≠ c
  b_relation : b = a + 1
  c_relation : c = b + 1

/-- The quotient of the number divided by the sum of its digits -/
def quotient (n : ThreeDigitNumber) : ℚ :=
  (100 * n.a + 10 * n.b + n.c) / (n.a + n.b + n.c)

/-- The theorem stating that the minimum quotient is 20.5 -/
theorem min_quotient_is_20_5 :
  ∀ n : ThreeDigitNumber, quotient n ≥ 20.5 ∧ ∃ n : ThreeDigitNumber, quotient n = 20.5 :=
sorry

end NUMINAMATH_CALUDE_min_quotient_is_20_5_l1512_151260


namespace NUMINAMATH_CALUDE_fractional_equation_positive_root_l1512_151257

theorem fractional_equation_positive_root (x m : ℝ) : 
  (∃ x > 0, (3 / (x - 4) = 1 - (x + m) / (4 - x))) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_positive_root_l1512_151257


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l1512_151237

theorem cube_root_equation_solution :
  ∃ x : ℝ, (2 * x * (x^3)^(1/2))^(1/3) = 6 ∧ x = 108^(2/5) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l1512_151237


namespace NUMINAMATH_CALUDE_horner_v₁_value_l1512_151209

def f (x : ℝ) : ℝ := 3 * x^6 + 4 * x^5 + 5 * x^4 + 6 * x^3 + 7 * x^2 + 8 * x + 1

def horner_v₁ (a₆ a₅ a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  ((((a₆ * x + a₅) * x + a₄) * x + a₃) * x + a₂) * x + a₁

theorem horner_v₁_value :
  horner_v₁ 3 4 5 6 7 8 1 0.4 = 5.2 :=
sorry

end NUMINAMATH_CALUDE_horner_v₁_value_l1512_151209


namespace NUMINAMATH_CALUDE_cost_of_dozen_rolls_l1512_151295

/-- The cost of a dozen rolls given the total spent and number of rolls purchased -/
theorem cost_of_dozen_rolls (total_spent : ℚ) (total_rolls : ℕ) (h1 : total_spent = 15) (h2 : total_rolls = 36) : 
  total_spent / (total_rolls / 12 : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_dozen_rolls_l1512_151295


namespace NUMINAMATH_CALUDE_ap_terms_count_l1512_151285

theorem ap_terms_count (n : ℕ) (a d : ℚ) : 
  Even n → 
  (n / 2 : ℚ) * (2 * a + (n - 2) * d) = 36 →
  (n / 2 : ℚ) * (2 * a + (n - 1) * d) = 44 →
  a + (n - 1) * d - a = 12 →
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_ap_terms_count_l1512_151285


namespace NUMINAMATH_CALUDE_root_implies_constant_value_l1512_151287

theorem root_implies_constant_value (c : ℝ) : 
  ((-5 : ℝ)^2 = c^2) → (c = 5 ∨ c = -5) := by
  sorry

end NUMINAMATH_CALUDE_root_implies_constant_value_l1512_151287


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1512_151298

/-- An arithmetic sequence {a_n} with a_2 = 2 and S_11 = 66 -/
def a (n : ℕ) : ℚ :=
  sorry

/-- The sum of the first n terms of the sequence a -/
def S (n : ℕ) : ℚ :=
  sorry

/-- The sequence b_n defined as 1 / (a_n * a_n+1) -/
def b (n : ℕ) : ℚ :=
  1 / (a n * a (n + 1))

/-- The sum of the first n terms of sequence b -/
def b_sum (n : ℕ) : ℚ :=
  sorry

theorem arithmetic_sequence_property :
  a 2 = 2 ∧ S 11 = 66 ∧ ∀ n : ℕ, n > 0 → b_sum n < 1 :=
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1512_151298


namespace NUMINAMATH_CALUDE_stratified_sampling_distribution_l1512_151238

theorem stratified_sampling_distribution 
  (total : ℕ) (senior : ℕ) (intermediate : ℕ) (junior : ℕ) (sample_size : ℕ)
  (h_total : total = 150)
  (h_senior : senior = 45)
  (h_intermediate : intermediate = 90)
  (h_junior : junior = 15)
  (h_sum : senior + intermediate + junior = total)
  (h_sample : sample_size = 30) :
  ∃ (sample_senior sample_intermediate sample_junior : ℕ),
    sample_senior + sample_intermediate + sample_junior = sample_size ∧
    sample_senior * total = senior * sample_size ∧
    sample_intermediate * total = intermediate * sample_size ∧
    sample_junior * total = junior * sample_size ∧
    sample_senior = 3 ∧
    sample_intermediate = 18 ∧
    sample_junior = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_distribution_l1512_151238


namespace NUMINAMATH_CALUDE_budget_allocation_l1512_151284

theorem budget_allocation (salaries utilities equipment supplies transportation : ℝ) 
  (h1 : salaries = 60)
  (h2 : utilities = 5)
  (h3 : equipment = 4)
  (h4 : supplies = 2)
  (h5 : transportation = 72 / 360 * 100)
  (h6 : salaries + utilities + equipment + supplies + transportation < 100) :
  100 - (salaries + utilities + equipment + supplies + transportation) = 9 := by
sorry

end NUMINAMATH_CALUDE_budget_allocation_l1512_151284


namespace NUMINAMATH_CALUDE_line_intersects_circle_l1512_151274

/-- The line l defined by 2mx - y - 8m - 3 = 0 -/
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  2 * m * x - y - 8 * m - 3 = 0

/-- The circle C defined by (x - 3)² + (y + 6)² = 25 -/
def circle_C (x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 6)^2 = 25

/-- The theorem stating that the line l intersects the circle C for any real m -/
theorem line_intersects_circle :
  ∀ m : ℝ, ∃ x y : ℝ, line_l m x y ∧ circle_C x y :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l1512_151274


namespace NUMINAMATH_CALUDE_positive_solution_x_l1512_151225

theorem positive_solution_x (x y z : ℝ)
  (eq1 : x * y = 8 - 2 * x - 3 * y)
  (eq2 : y * z = 8 - 4 * y - 2 * z)
  (eq3 : x * z = 40 - 4 * x - 3 * z)
  (h_pos : x > 0) :
  x = (7 * Real.sqrt 13 - 6) / 2 := by
sorry

end NUMINAMATH_CALUDE_positive_solution_x_l1512_151225


namespace NUMINAMATH_CALUDE_tangent_line_sum_l1512_151256

/-- Given a function f: ℝ → ℝ with a tangent line at x = 2 
    described by the equation 2x + y - 3 = 0,
    prove that f(2) + f'(2) = -3 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_tangent : ∀ x y, y = f x → 2 * x + y - 3 = 0 → x = 2) :
  f 2 + deriv f 2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l1512_151256


namespace NUMINAMATH_CALUDE_second_discount_percentage_l1512_151218

theorem second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ) : 
  original_price = 400 →
  first_discount = 12 →
  final_price = 334.4 →
  ∃ (second_discount : ℝ),
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) ∧
    second_discount = 5 := by
  sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l1512_151218


namespace NUMINAMATH_CALUDE_penny_pudding_grains_l1512_151244

-- Define the given conditions
def cans_per_tonne : ℕ := 25000
def grains_per_tonne : ℕ := 50000000

-- Define the function to calculate grains per can
def grains_per_can : ℕ := grains_per_tonne / cans_per_tonne

-- Theorem statement
theorem penny_pudding_grains :
  grains_per_can = 2000 :=
sorry

end NUMINAMATH_CALUDE_penny_pudding_grains_l1512_151244


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l1512_151205

theorem negative_fraction_comparison : -3/4 > -4/5 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l1512_151205


namespace NUMINAMATH_CALUDE_betty_order_total_cost_l1512_151220

/-- Calculate the total cost of Betty's order -/
theorem betty_order_total_cost :
  let slipper_quantity : ℕ := 6
  let slipper_price : ℚ := 5/2
  let lipstick_quantity : ℕ := 4
  let lipstick_price : ℚ := 5/4
  let hair_color_quantity : ℕ := 8
  let hair_color_price : ℚ := 3
  let total_items : ℕ := slipper_quantity + lipstick_quantity + hair_color_quantity
  let total_cost : ℚ := slipper_quantity * slipper_price + 
                        lipstick_quantity * lipstick_price + 
                        hair_color_quantity * hair_color_price
  total_items = 18 ∧ total_cost = 44 := by
  sorry

end NUMINAMATH_CALUDE_betty_order_total_cost_l1512_151220


namespace NUMINAMATH_CALUDE_partner_investment_period_l1512_151282

/-- Given two partners P and Q with investment and profit ratios, and Q's investment period,
    calculate P's investment period. -/
theorem partner_investment_period
  (investment_ratio_p investment_ratio_q : ℕ)
  (profit_ratio_p profit_ratio_q : ℕ)
  (q_months : ℕ)
  (h_investment : investment_ratio_p * 5 = investment_ratio_q * 7)
  (h_profit : profit_ratio_p * 9 = profit_ratio_q * 7)
  (h_q_months : q_months = 9) :
  ∃ (p_months : ℕ),
    p_months * profit_ratio_q * investment_ratio_q =
    q_months * profit_ratio_p * investment_ratio_p ∧
    p_months = 5 :=
by sorry

end NUMINAMATH_CALUDE_partner_investment_period_l1512_151282


namespace NUMINAMATH_CALUDE_asian_math_competition_l1512_151221

theorem asian_math_competition (total_countries : ℕ) 
  (solved_1 solved_1_2 solved_1_3 solved_1_4 solved_all : ℕ) :
  total_countries = 846 →
  solved_1 = 235 →
  solved_1_2 = 59 →
  solved_1_3 = 29 →
  solved_1_4 = 15 →
  solved_all = 3 →
  ∃ (country : ℕ), country ≤ total_countries ∧ 
    ∃ (students : ℕ), students ≥ 4 ∧
      students ≤ (solved_1 - solved_1_2 - solved_1_3 - solved_1_4 + solved_all) :=
by sorry

end NUMINAMATH_CALUDE_asian_math_competition_l1512_151221


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1512_151245

theorem cubic_equation_solution (x : ℝ) (h1 : x ≠ 0) (h2 : x^3 - 2*x^2 = 0) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1512_151245


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1512_151265

theorem necessary_but_not_sufficient (A B C : Set α) (h : ∀ a, a ∈ A ↔ (a ∈ B ∧ a ∈ C)) :
  (∀ a, a ∈ A → a ∈ B) ∧ ¬(∀ a, a ∈ B → a ∈ A) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1512_151265


namespace NUMINAMATH_CALUDE_fourth_term_of_solution_sequence_l1512_151261

def is_solution (x : ℤ) : Prop := x^2 - 2*x - 3 < 0

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem fourth_term_of_solution_sequence :
  ∃ a : ℕ → ℤ,
    (∀ n : ℕ, is_solution (a n)) ∧
    arithmetic_sequence a ∧
    (a 4 = 3 ∨ a 4 = -1) := by sorry

end NUMINAMATH_CALUDE_fourth_term_of_solution_sequence_l1512_151261


namespace NUMINAMATH_CALUDE_problem_solution_l1512_151223

theorem problem_solution (m n : ℝ) : 
  (∃ k : ℝ, k^2 = 3*m + 1 ∧ (k = 2 ∨ k = -2)) →
  (∃ l : ℝ, l^3 = 5*n - 2 ∧ l = 2) →
  m = 1 ∧ n = 2 ∧ (∃ r : ℝ, r^2 = 4*m + 5/2*n ∧ (r = 3 ∨ r = -3)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1512_151223


namespace NUMINAMATH_CALUDE_interest_difference_approximately_128_l1512_151227

-- Define the initial deposit
def initial_deposit : ℝ := 14000

-- Define the interest rates
def compound_rate : ℝ := 0.06
def simple_rate : ℝ := 0.08

-- Define the time period
def years : ℕ := 10

-- Define the compound interest function
def compound_interest (p r : ℝ) (n : ℕ) : ℝ := p * (1 + r) ^ n

-- Define the simple interest function
def simple_interest (p r : ℝ) (t : ℕ) : ℝ := p * (1 + r * t)

-- State the theorem
theorem interest_difference_approximately_128 :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧
  abs (simple_interest initial_deposit simple_rate years - 
       compound_interest initial_deposit compound_rate years - 128) < ε :=
sorry

end NUMINAMATH_CALUDE_interest_difference_approximately_128_l1512_151227


namespace NUMINAMATH_CALUDE_coin_problem_l1512_151291

theorem coin_problem (total_coins : ℕ) (total_value : ℚ) : 
  total_coins = 32 ∧ 
  total_value = 47/10 →
  ∃ (quarters dimes : ℕ), 
    quarters + dimes = total_coins ∧ 
    (1/4 : ℚ) * quarters + (1/10 : ℚ) * dimes = total_value ∧
    quarters = 10 := by sorry

end NUMINAMATH_CALUDE_coin_problem_l1512_151291


namespace NUMINAMATH_CALUDE_math_books_count_l1512_151203

theorem math_books_count (total_books : ℕ) (math_price history_price total_price : ℕ) :
  total_books = 80 →
  math_price = 4 →
  history_price = 5 →
  total_price = 373 →
  ∃ (math_books : ℕ), 
    math_books * math_price + (total_books - math_books) * history_price = total_price ∧
    math_books = 27 := by
  sorry

end NUMINAMATH_CALUDE_math_books_count_l1512_151203


namespace NUMINAMATH_CALUDE_karen_group_size_l1512_151296

/-- Proves that if Zack tutors students in groups of 14, and both Zack and Karen tutor
    the same total number of 70 students, then Karen must also tutor students in groups of 14. -/
theorem karen_group_size (zack_group_size : ℕ) (total_students : ℕ) (karen_group_size : ℕ) :
  zack_group_size = 14 →
  total_students = 70 →
  total_students % zack_group_size = 0 →
  total_students % karen_group_size = 0 →
  total_students / zack_group_size = total_students / karen_group_size →
  karen_group_size = 14 := by
sorry

end NUMINAMATH_CALUDE_karen_group_size_l1512_151296


namespace NUMINAMATH_CALUDE_division_error_problem_l1512_151293

theorem division_error_problem (x : ℝ) (y : ℝ) (h : y > 0) :
  (abs (5 * x - x / y) / (5 * x)) * 100 = 98 → y = 10 := by
  sorry

end NUMINAMATH_CALUDE_division_error_problem_l1512_151293


namespace NUMINAMATH_CALUDE_candy_bar_cost_proof_l1512_151276

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The number of quarters John used -/
def quarters_used : ℕ := 4

/-- The number of dimes John used -/
def dimes_used : ℕ := 3

/-- The number of nickels John used -/
def nickels_used : ℕ := 1

/-- The amount of change John received in cents -/
def change_received : ℕ := 4

/-- The cost of the candy bar in cents -/
def candy_bar_cost : ℕ := 131

theorem candy_bar_cost_proof :
  (quarters_used * quarter_value + dimes_used * dime_value + nickels_used * nickel_value) - change_received = candy_bar_cost :=
by sorry

end NUMINAMATH_CALUDE_candy_bar_cost_proof_l1512_151276


namespace NUMINAMATH_CALUDE_semi_annual_compounding_l1512_151283

noncomputable def compound_interest_frequency 
  (initial_investment : ℝ) 
  (annual_rate : ℝ) 
  (final_amount : ℝ) 
  (years : ℝ) : ℝ :=
  let r := annual_rate / 100
  ((final_amount / initial_investment) ^ (1 / (r * years)) - 1) / (r / years)

theorem semi_annual_compounding 
  (initial_investment : ℝ) 
  (annual_rate : ℝ) 
  (final_amount : ℝ) 
  (years : ℝ) 
  (h1 : initial_investment = 10000) 
  (h2 : annual_rate = 3.96) 
  (h3 : final_amount = 10815.83) 
  (h4 : years = 2) :
  ∃ ε > 0, |compound_interest_frequency initial_investment annual_rate final_amount years - 2| < ε :=
sorry

end NUMINAMATH_CALUDE_semi_annual_compounding_l1512_151283


namespace NUMINAMATH_CALUDE_divisibility_condition_l1512_151212

theorem divisibility_condition (n : ℕ+) :
  (∃ (A B : ℕ), 
    A ≠ B ∧ 
    10^(n.val-1) ≤ A ∧ A < 10^n.val ∧
    10^(n.val-1) ≤ B ∧ B < 10^n.val ∧
    (10^n.val * A + B) % (10^n.val * B + A) = 0) ↔ 
  n.val % 6 = 3 :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1512_151212


namespace NUMINAMATH_CALUDE_adult_ticket_price_l1512_151286

theorem adult_ticket_price 
  (child_price : ℕ)
  (total_attendance : ℕ)
  (total_collection : ℕ)
  (children_attendance : ℕ) :
  child_price = 25 →
  total_attendance = 280 →
  total_collection = 140 * 100 →
  children_attendance = 80 →
  (total_attendance - children_attendance) * 60 + children_attendance * child_price = total_collection :=
by sorry

end NUMINAMATH_CALUDE_adult_ticket_price_l1512_151286


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l1512_151255

theorem min_value_of_reciprocal_sum (t q a b : ℝ) : 
  (∀ x, x^2 - t*x + q = 0 ↔ x = a ∨ x = b) →
  a + b = a^2 + b^2 →
  a + b = a^3 + b^3 →
  a + b = a^4 + b^4 →
  ∃ (min : ℝ), min = 128 * Real.sqrt 3 / 45 ∧ 
    ∀ (t' q' a' b' : ℝ), 
      (∀ x, x^2 - t'*x + q' = 0 ↔ x = a' ∨ x = b') →
      a' + b' = a'^2 + b'^2 →
      a' + b' = a'^3 + b'^3 →
      a' + b' = a'^4 + b'^4 →
      1/a'^5 + 1/b'^5 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l1512_151255


namespace NUMINAMATH_CALUDE_vector_addition_l1512_151277

theorem vector_addition (a b : ℝ × ℝ) :
  a = (5, -3) → b = (-6, 4) → a + b = (-1, 1) := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_l1512_151277


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l1512_151270

def angle_between (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_magnitude (a b : ℝ × ℝ) 
  (h1 : angle_between a b = π / 3)
  (h2 : a = (2, 0))
  (h3 : Real.sqrt ((Prod.fst b)^2 + (Prod.snd b)^2) = 1) :
  Real.sqrt ((Prod.fst (a + 2 • b))^2 + (Prod.snd (a + 2 • b))^2) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l1512_151270


namespace NUMINAMATH_CALUDE_shoebox_surface_area_l1512_151232

/-- The surface area of a rectangular prism -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a rectangular prism with dimensions 12 cm × 5 cm × 3 cm is 222 square centimeters -/
theorem shoebox_surface_area :
  surface_area 12 5 3 = 222 := by
  sorry

end NUMINAMATH_CALUDE_shoebox_surface_area_l1512_151232


namespace NUMINAMATH_CALUDE_max_product_of_distances_l1512_151278

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 = 1

-- Define the foci
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem max_product_of_distances (P : ℝ × ℝ) (h : is_on_ellipse P.1 P.2) :
  ∃ (max : ℝ), max = 8 ∧ ∀ Q : ℝ × ℝ, is_on_ellipse Q.1 Q.2 →
    distance Q F1 * distance Q F2 ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_product_of_distances_l1512_151278


namespace NUMINAMATH_CALUDE_correct_proposition_l1512_151208

-- Define propositions p and q
variable (p q : Prop)

-- Define the conditions
axiom p_true : p
axiom q_false : ¬q

-- Theorem to prove
theorem correct_proposition : (¬p) ∨ (¬q) :=
sorry

end NUMINAMATH_CALUDE_correct_proposition_l1512_151208


namespace NUMINAMATH_CALUDE_additional_male_workers_l1512_151292

theorem additional_male_workers (initial_female_percent : ℚ) 
                                (final_female_percent : ℚ) 
                                (final_total : ℕ) : ℕ :=
  let initial_female_percent := 60 / 100
  let final_female_percent := 55 / 100
  let final_total := 312
  26

#check additional_male_workers

end NUMINAMATH_CALUDE_additional_male_workers_l1512_151292


namespace NUMINAMATH_CALUDE_f_inequality_range_l1512_151201

/-- The function f(x) = |x-a| + |x+2| -/
def f (a x : ℝ) : ℝ := |x - a| + |x + 2|

/-- The theorem stating the range of a values for which ∃x₀ ∈ ℝ such that f(x₀) ≤ |2a+1| -/
theorem f_inequality_range (a : ℝ) : 
  (∃ x₀ : ℝ, f a x₀ ≤ |2*a + 1|) ↔ a ≤ -1 ∨ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_range_l1512_151201


namespace NUMINAMATH_CALUDE_decomposition_675_l1512_151288

theorem decomposition_675 (n : Nat) (h : n = 675) :
  ∃ (num_stacks height : Nat),
    num_stacks > 1 ∧
    height > 1 ∧
    n = 3^3 * 5^2 ∧
    num_stacks = 3 ∧
    height = 3^2 * 5^2 ∧
    height^num_stacks = n := by
  sorry

end NUMINAMATH_CALUDE_decomposition_675_l1512_151288


namespace NUMINAMATH_CALUDE_min_books_borrowed_l1512_151262

theorem min_books_borrowed (total_students : Nat) (zero_books : Nat) (one_book : Nat) (two_books : Nat)
  (avg_books : Rat) (max_books : Nat) :
  total_students = 32 →
  zero_books = 2 →
  one_book = 12 →
  two_books = 10 →
  avg_books = 2 →
  max_books = 11 →
  ∃ (min_books : Nat),
    min_books = 4 ∧
    min_books ≤ max_books ∧
    (total_students - zero_books - one_book - two_books) * min_books +
    one_book * 1 + two_books * 2 =
    (total_students : Rat) * avg_books := by
  sorry

end NUMINAMATH_CALUDE_min_books_borrowed_l1512_151262


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1512_151271

def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b

def nth_term (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

theorem arithmetic_sequence_problem (x y : ℝ) (m : ℕ) 
  (h1 : is_arithmetic_sequence (Real.log (x^2 * y^5)) (Real.log (x^4 * y^9)) (Real.log (x^7 * y^12)))
  (h2 : nth_term (Real.log (x^2 * y^5)) 
               ((Real.log (x^4 * y^9)) - (Real.log (x^2 * y^5))) 
               10 = Real.log (y^m)) :
  m = 55 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1512_151271


namespace NUMINAMATH_CALUDE_student_score_problem_l1512_151254

theorem student_score_problem (total_questions : ℕ) (score : ℤ) 
  (h1 : total_questions = 100)
  (h2 : score = 73) :
  ∃ (correct incorrect : ℕ),
    correct + incorrect = total_questions ∧
    (correct : ℤ) - 2 * (incorrect : ℤ) = score ∧
    correct = 91 := by
  sorry

end NUMINAMATH_CALUDE_student_score_problem_l1512_151254


namespace NUMINAMATH_CALUDE_subset_implies_a_geq_two_disjoint_implies_a_leq_one_l1512_151217

-- Define set A
def A : Set ℝ := {x | x^2 - 3*x + 2 < 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for case (1)
theorem subset_implies_a_geq_two (a : ℝ) : A ⊆ B a → a ≥ 2 := by
  sorry

-- Theorem for case (2)
theorem disjoint_implies_a_leq_one (a : ℝ) : A ∩ B a = ∅ → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_geq_two_disjoint_implies_a_leq_one_l1512_151217


namespace NUMINAMATH_CALUDE_intersection_of_sets_l1512_151275

theorem intersection_of_sets : 
  let P : Set ℕ := {3, 5, 6, 8}
  let Q : Set ℕ := {4, 5, 7, 8}
  P ∩ Q = {5, 8} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l1512_151275


namespace NUMINAMATH_CALUDE_emily_earnings_theorem_l1512_151233

/-- The amount of money Emily makes by selling chocolate bars -/
def emily_earnings (total_bars : ℕ) (price_per_bar : ℕ) (unsold_bars : ℕ) : ℕ :=
  (total_bars - unsold_bars) * price_per_bar

/-- Theorem: Emily makes $77 by selling all but 4 bars from a box of 15 bars costing $7 each -/
theorem emily_earnings_theorem : emily_earnings 15 7 4 = 77 := by
  sorry

end NUMINAMATH_CALUDE_emily_earnings_theorem_l1512_151233


namespace NUMINAMATH_CALUDE_circle_C_properties_l1512_151289

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 4*y + 4 = 0

-- Define the line that contains the center of circle C
def center_line (x y : ℝ) : Prop :=
  x + 2*y + 1 = 0

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop :=
  8*x - 15*y - 3 = 0 ∨ x = 6

-- Define the line l
def line_l (x y m : ℝ) : Prop :=
  y = x + m

theorem circle_C_properties :
  -- Circle C passes through M(0, -2) and N(3, 1)
  circle_C 0 (-2) ∧ circle_C 3 1 ∧
  -- The center of circle C lies on the line x + 2y + 1 = 0
  ∃ (cx cy : ℝ), center_line cx cy ∧
    ∀ (x y : ℝ), circle_C x y ↔ (x - cx)^2 + (y - cy)^2 = (cx^2 + cy^2 - 4) →
  -- The tangent line to circle C passing through (6, 3) is correct
  tangent_line 6 3 ∧
  -- The line l has the correct equations
  (line_l x y (-1) ∨ line_l x y (-4)) ∧
  -- Circle C₁ with diameter AB (intersection of l and C) passes through the origin
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    ((line_l x₁ y₁ (-1) ∧ line_l x₂ y₂ (-1)) ∨ (line_l x₁ y₁ (-4) ∧ line_l x₂ y₂ (-4))) ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 2 * ((x₁^2 + y₁^2) + (x₂^2 + y₂^2)) :=
sorry

end NUMINAMATH_CALUDE_circle_C_properties_l1512_151289


namespace NUMINAMATH_CALUDE_square_difference_value_l1512_151235

theorem square_difference_value (m n : ℝ) (h : m^2 + n^2 = 6*m - 4*n - 13) : 
  m^2 - n^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_square_difference_value_l1512_151235


namespace NUMINAMATH_CALUDE_total_retail_price_calculation_l1512_151213

def calculate_retail_price (wholesale_price : ℝ) (profit_margin : ℝ) (discount : ℝ) : ℝ :=
  let retail_before_discount := wholesale_price * (1 + profit_margin)
  retail_before_discount * (1 - discount)

theorem total_retail_price_calculation (P Q R : ℝ) 
  (h1 : P = 90) (h2 : Q = 120) (h3 : R = 150) : 
  calculate_retail_price P 0.2 0.1 + 
  calculate_retail_price Q 0.25 0.15 + 
  calculate_retail_price R 0.3 0.2 = 380.7 := by
  sorry

#eval calculate_retail_price 90 0.2 0.1 + 
      calculate_retail_price 120 0.25 0.15 + 
      calculate_retail_price 150 0.3 0.2

end NUMINAMATH_CALUDE_total_retail_price_calculation_l1512_151213


namespace NUMINAMATH_CALUDE_factory_conditional_probability_l1512_151202

/-- Represents the production data for a factory --/
structure FactoryData where
  total_parts : ℕ
  a_parts : ℕ
  a_qualified : ℕ
  b_parts : ℕ
  b_qualified : ℕ

/-- Calculates the conditional probability of a part being qualified given it was produced by A --/
def conditional_probability (data : FactoryData) : ℚ :=
  data.a_qualified / data.a_parts

/-- Theorem stating the conditional probability for the given problem --/
theorem factory_conditional_probability 
  (data : FactoryData)
  (h1 : data.total_parts = 100)
  (h2 : data.a_parts = 40)
  (h3 : data.a_qualified = 35)
  (h4 : data.b_parts = 60)
  (h5 : data.b_qualified = 50)
  (h6 : data.total_parts = data.a_parts + data.b_parts) :
  conditional_probability data = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_factory_conditional_probability_l1512_151202


namespace NUMINAMATH_CALUDE_smallest_base_for_124_l1512_151249

theorem smallest_base_for_124 (b : ℕ) : b ≥ 5 ↔ b ^ 2 ≤ 124 ∧ 124 < b ^ 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_base_for_124_l1512_151249


namespace NUMINAMATH_CALUDE_problem_solution_l1512_151280

theorem problem_solution (N : ℚ) : 
  (4 / 5 : ℚ) * (3 / 8 : ℚ) * N = 24 → (5 / 2 : ℚ) * N = 200 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1512_151280


namespace NUMINAMATH_CALUDE_purchase_group_equation_l1512_151234

/-- A group of people buying an item -/
structure PurchaseGroup where
  price : ℝ  -- Price of the item in coins
  excess_contribution : ℝ := 8  -- Contribution that exceeds the price
  excess_amount : ℝ := 3  -- Amount by which the excess contribution exceeds the price
  shortfall_contribution : ℝ := 7  -- Contribution that falls short of the price
  shortfall_amount : ℝ := 4  -- Amount by which the shortfall contribution falls short of the price

/-- The equation holds for a purchase group -/
theorem purchase_group_equation (g : PurchaseGroup) :
  (g.price + g.excess_amount) / g.excess_contribution = (g.price - g.shortfall_amount) / g.shortfall_contribution :=
sorry

end NUMINAMATH_CALUDE_purchase_group_equation_l1512_151234


namespace NUMINAMATH_CALUDE_point_on_line_with_equal_distances_l1512_151252

theorem point_on_line_with_equal_distances (P : ℝ × ℝ) :
  P.1 + 3 * P.2 = 0 →
  (P.1^2 + P.2^2).sqrt = |P.1 + 3 * P.2 - 2| / (1^2 + 3^2).sqrt →
  (P = (3/5, -1/5) ∨ P = (-3/5, 1/5)) :=
by sorry

end NUMINAMATH_CALUDE_point_on_line_with_equal_distances_l1512_151252


namespace NUMINAMATH_CALUDE_bottles_drunk_per_day_l1512_151263

theorem bottles_drunk_per_day (initial_bottles : ℕ) (remaining_bottles : ℕ) (days : ℕ) : 
  initial_bottles = 301 → remaining_bottles = 157 → days = 1 →
  initial_bottles - remaining_bottles = 144 := by
sorry

end NUMINAMATH_CALUDE_bottles_drunk_per_day_l1512_151263


namespace NUMINAMATH_CALUDE_simplify_expression_l1512_151248

theorem simplify_expression : 
  (81 ^ (1/4) - Real.sqrt (17/2)) ^ 2 = 17.5 - 3 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1512_151248


namespace NUMINAMATH_CALUDE_senegal_total_points_l1512_151241

-- Define the point values for victory and draw
def victory_points : ℕ := 3
def draw_points : ℕ := 1

-- Define Senegal's match results
def senegal_victories : ℕ := 1
def senegal_draws : ℕ := 2

-- Define the function to calculate total points
def calculate_points (victories draws : ℕ) : ℕ :=
  victories * victory_points + draws * draw_points

-- Theorem to prove
theorem senegal_total_points :
  calculate_points senegal_victories senegal_draws = 5 := by
  sorry

end NUMINAMATH_CALUDE_senegal_total_points_l1512_151241


namespace NUMINAMATH_CALUDE_six_playing_cards_distribution_l1512_151222

/-- Given a deck of cards with playing cards and instruction cards,
    distributed as evenly as possible among a group of people,
    calculate the number of people who end up with exactly 6 playing cards. -/
def people_with_six_playing_cards (total_cards : ℕ) (playing_cards : ℕ) (instruction_cards : ℕ) (num_people : ℕ) : ℕ :=
  let cards_per_person := total_cards / num_people
  let extra_cards := total_cards % num_people
  let playing_cards_distribution := playing_cards / num_people
  let extra_playing_cards := playing_cards % num_people
  min extra_playing_cards (num_people - instruction_cards)

theorem six_playing_cards_distribution :
  people_with_six_playing_cards 60 52 8 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_six_playing_cards_distribution_l1512_151222


namespace NUMINAMATH_CALUDE_height_of_cube_with_corner_cut_l1512_151281

/-- The height of a cube with one corner cut off -/
theorem height_of_cube_with_corner_cut (s : ℝ) (h : s = 2) :
  let diagonal := s * Real.sqrt 3
  let cut_face_side := diagonal / Real.sqrt 2
  let cut_face_area := Real.sqrt 3 / 4 * cut_face_side^2
  let pyramid_volume := 1 / 6 * s^3
  let remaining_height := s - (3 * pyramid_volume) / cut_face_area
  remaining_height = 2 - Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_height_of_cube_with_corner_cut_l1512_151281


namespace NUMINAMATH_CALUDE_checkers_placement_divisibility_l1512_151240

theorem checkers_placement_divisibility (p : Nat) (h_prime : Nat.Prime p) (h_ge_5 : p ≥ 5) :
  (Nat.choose (p^2) p) % (p^5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_checkers_placement_divisibility_l1512_151240


namespace NUMINAMATH_CALUDE_marking_exists_l1512_151224

/-- Represents a 50x50 board with some cells occupied -/
def Board := Fin 50 → Fin 50 → Bool

/-- Represents a marking of free cells on the board -/
def Marking := Fin 50 → Fin 50 → Bool

/-- Check if a marking is valid (at most 99 cells marked) -/
def valid_marking (b : Board) (m : Marking) : Prop :=
  (Finset.sum Finset.univ (fun i => Finset.sum Finset.univ (fun j => if m i j then 1 else 0))) ≤ 99

/-- Check if the total number of marked and originally occupied cells in a row is even -/
def row_even (b : Board) (m : Marking) (i : Fin 50) : Prop :=
  Even (Finset.sum Finset.univ (fun j => if b i j || m i j then 1 else 0))

/-- Check if the total number of marked and originally occupied cells in a column is even -/
def col_even (b : Board) (m : Marking) (j : Fin 50) : Prop :=
  Even (Finset.sum Finset.univ (fun i => if b i j || m i j then 1 else 0))

/-- Main theorem: For any board configuration, there exists a valid marking that makes all rows and columns even -/
theorem marking_exists (b : Board) : ∃ m : Marking, 
  valid_marking b m ∧ 
  (∀ i : Fin 50, row_even b m i) ∧ 
  (∀ j : Fin 50, col_even b m j) := by
  sorry

end NUMINAMATH_CALUDE_marking_exists_l1512_151224


namespace NUMINAMATH_CALUDE_range_of_m_l1512_151294

/-- Given propositions p and q, where ¬q is a sufficient but not necessary condition for ¬p,
    prove that the range of values for m is m ≥ 1. -/
theorem range_of_m (x m : ℝ) : 
  (∀ x, (x^2 + x - 2 > 0 ↔ x > 1 ∨ x < -2)) →
  (∀ x, (x ≤ m → x^2 + x - 2 ≤ 0) ∧ 
        ∃ y, (y^2 + y - 2 ≤ 0 ∧ y > m)) →
  m ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1512_151294


namespace NUMINAMATH_CALUDE_even_integers_in_pascal_triangle_l1512_151251

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : Type := Unit

/-- Counts the number of even integers in the first n rows of Pascal's Triangle -/
def countEvenIntegers (pt : PascalTriangle n) : ℕ := sorry

theorem even_integers_in_pascal_triangle :
  ∀ (pt10 : PascalTriangle 10) (pt15 : PascalTriangle 15),
    countEvenIntegers pt10 = 22 →
    countEvenIntegers pt15 = 53 := by sorry

end NUMINAMATH_CALUDE_even_integers_in_pascal_triangle_l1512_151251


namespace NUMINAMATH_CALUDE_cubic_equation_properties_l1512_151246

/-- The cubic equation (x-1)(x^2-3x+m) = 0 -/
def cubic_equation (x m : ℝ) : Prop := (x - 1) * (x^2 - 3*x + m) = 0

/-- The discriminant of the quadratic part x^2 - 3x + m -/
def discriminant (m : ℝ) : ℝ := 9 - 4*m

theorem cubic_equation_properties :
  /- When m = 4, the equation has only one real root x = 1 -/
  (∀ x : ℝ, cubic_equation x 4 ↔ x = 1) ∧
  /- The equation has exactly two equal roots when m = 2 or m = 9/4 -/
  (∀ x₁ x₂ x₃ : ℝ, (cubic_equation x₁ 2 ∧ cubic_equation x₂ 2 ∧ cubic_equation x₃ 2 ∧
    ((x₁ = x₂ ∧ x₁ ≠ x₃) ∨ (x₁ = x₃ ∧ x₁ ≠ x₂) ∨ (x₂ = x₃ ∧ x₁ ≠ x₂))) ∨
   (cubic_equation x₁ (9/4) ∧ cubic_equation x₂ (9/4) ∧ cubic_equation x₃ (9/4) ∧
    ((x₁ = x₂ ∧ x₁ ≠ x₃) ∨ (x₁ = x₃ ∧ x₁ ≠ x₂) ∨ (x₂ = x₃ ∧ x₁ ≠ x₂)))) ∧
  /- The three real roots form a triangle if and only if 2 < m ≤ 9/4 -/
  (∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, cubic_equation x₁ m ∧ cubic_equation x₂ m ∧ cubic_equation x₃ m ∧
    x₁ + x₂ > x₃ ∧ x₁ + x₃ > x₂ ∧ x₂ + x₃ > x₁) ↔ (2 < m ∧ m ≤ 9/4)) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_properties_l1512_151246


namespace NUMINAMATH_CALUDE_four_card_selection_with_face_l1512_151219

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of suits in a standard deck -/
def NumSuits : ℕ := 4

/-- Number of face cards per suit -/
def FaceCardsPerSuit : ℕ := 3

/-- Number of cards per suit -/
def CardsPerSuit : ℕ := 13

/-- Theorem: Number of ways to choose 4 cards from a standard deck
    such that all four cards are of different suits and at least one is a face card -/
theorem four_card_selection_with_face (deck : ℕ) (suits : ℕ) (face_per_suit : ℕ) (cards_per_suit : ℕ)
    (h1 : deck = StandardDeck)
    (h2 : suits = NumSuits)
    (h3 : face_per_suit = FaceCardsPerSuit)
    (h4 : cards_per_suit = CardsPerSuit) :
  suits * face_per_suit * (cards_per_suit ^ (suits - 1)) = 26364 :=
sorry

end NUMINAMATH_CALUDE_four_card_selection_with_face_l1512_151219


namespace NUMINAMATH_CALUDE_solution_to_system_of_equations_l1512_151226

theorem solution_to_system_of_equations :
  ∃ (x y : ℚ), 3 * x - 18 * y = 5 ∧ 4 * y - x = 6 ∧ x = -64/3 ∧ y = -23/6 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_of_equations_l1512_151226


namespace NUMINAMATH_CALUDE_equation_holds_for_negative_eight_l1512_151200

theorem equation_holds_for_negative_eight :
  let t : ℝ := -8
  let f (x : ℝ) : ℝ := (2 / (x + 3)) + (x / (x + 3)) - (4 / (x + 3))
  f t = 2 := by
sorry

end NUMINAMATH_CALUDE_equation_holds_for_negative_eight_l1512_151200


namespace NUMINAMATH_CALUDE_midpoint_fraction_l1512_151206

theorem midpoint_fraction : 
  let a := (3 : ℚ) / 4
  let b := (5 : ℚ) / 7
  let midpoint := (a + b) / 2
  midpoint = (41 : ℚ) / 56 := by
sorry

end NUMINAMATH_CALUDE_midpoint_fraction_l1512_151206


namespace NUMINAMATH_CALUDE_book_pages_count_l1512_151268

/-- The number of pages read each night -/
def pages_per_night : ℕ := 12

/-- The number of nights needed to finish the book -/
def nights_to_finish : ℕ := 10

/-- The total number of pages in the book -/
def total_pages : ℕ := pages_per_night * nights_to_finish

theorem book_pages_count : total_pages = 120 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_count_l1512_151268


namespace NUMINAMATH_CALUDE_height_for_weight_35_l1512_151236

/-- Linear regression equation relating height to weight -/
def linear_regression (x : ℝ) : ℝ := 0.1 * x + 20

/-- Theorem stating that a person weighing 35 kg has a height of 150 cm
    according to the given linear regression equation -/
theorem height_for_weight_35 :
  ∃ x : ℝ, linear_regression x = 35 ∧ x = 150 := by
  sorry

end NUMINAMATH_CALUDE_height_for_weight_35_l1512_151236


namespace NUMINAMATH_CALUDE_area_difference_is_quarter_l1512_151214

/-- Represents a regular octagon with side length 1 -/
structure RegularOctagon :=
  (side_length : ℝ)
  (is_regular : side_length = 1)

/-- Represents the cutting operation on the octagon -/
def cut (o : RegularOctagon) : ℝ × ℝ := sorry

/-- The difference in area between the larger and smaller parts after cutting -/
def area_difference (o : RegularOctagon) : ℝ :=
  let (larger, smaller) := cut o
  larger - smaller

/-- Theorem stating that the area difference is 1/4 -/
theorem area_difference_is_quarter (o : RegularOctagon) :
  area_difference o = 1/4 := by sorry

end NUMINAMATH_CALUDE_area_difference_is_quarter_l1512_151214


namespace NUMINAMATH_CALUDE_vectors_not_basis_l1512_151273

/-- Two vectors are non-collinear if they are not scalar multiples of each other -/
def NonCollinear (v w : ℝ × ℝ) : Prop :=
  ∀ (c : ℝ), v ≠ c • w

/-- Two vectors are linearly dependent if one is a scalar multiple of the other -/
def LinearlyDependent (v w : ℝ × ℝ) : Prop :=
  ∃ (c : ℝ), v = c • w

theorem vectors_not_basis (e₁ e₂ : ℝ × ℝ) (h : NonCollinear e₁ e₂) :
  LinearlyDependent (e₁ + 3 • e₂) (6 • e₂ + 2 • e₁) :=
sorry

end NUMINAMATH_CALUDE_vectors_not_basis_l1512_151273


namespace NUMINAMATH_CALUDE_range_of_a_for_always_positive_quadratic_l1512_151247

theorem range_of_a_for_always_positive_quadratic :
  ∀ (a : ℝ), (∀ (x : ℝ), a * x^2 - 3 * a * x + 9 > 0) ↔ (0 ≤ a ∧ a < 4) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_always_positive_quadratic_l1512_151247


namespace NUMINAMATH_CALUDE_unique_a_value_l1512_151215

theorem unique_a_value (a : ℕ) : 
  (∃ k : ℕ, 88 * a = 2 * k + 1) →  -- 88a is odd
  (∃ m : ℕ, 88 * a = 3 * m) →      -- 88a is a multiple of 3
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_unique_a_value_l1512_151215


namespace NUMINAMATH_CALUDE_house_sale_revenue_distribution_l1512_151272

theorem house_sale_revenue_distribution (market_value : ℝ) (selling_price_percentage : ℝ) 
  (num_people : ℕ) (tax_rate : ℝ) (individual_share : ℝ) : 
  market_value = 500000 →
  selling_price_percentage = 1.20 →
  num_people = 4 →
  tax_rate = 0.10 →
  individual_share = (market_value * selling_price_percentage * (1 - tax_rate)) / num_people →
  individual_share = 135000 := by
  sorry

end NUMINAMATH_CALUDE_house_sale_revenue_distribution_l1512_151272


namespace NUMINAMATH_CALUDE_equation_solution_exists_l1512_151216

theorem equation_solution_exists : ∃ x : ℝ, 
  (0.76 : ℝ)^3 - (0.1 : ℝ)^3 / (0.76 : ℝ)^2 + x + (0.1 : ℝ)^2 = 0.66 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l1512_151216


namespace NUMINAMATH_CALUDE_matts_bike_ride_l1512_151207

/-- Given Matt's bike ride scenario, prove the remaining distance after the second sign. -/
theorem matts_bike_ride (total_distance : ℕ) (distance_to_first_sign : ℕ) (distance_between_signs : ℕ)
  (h1 : total_distance = 1000)
  (h2 : distance_to_first_sign = 350)
  (h3 : distance_between_signs = 375) :
  total_distance - (distance_to_first_sign + distance_between_signs) = 275 := by
  sorry

end NUMINAMATH_CALUDE_matts_bike_ride_l1512_151207


namespace NUMINAMATH_CALUDE_test_score_properties_l1512_151290

/-- A test with multiple-choice questions. -/
structure Test where
  num_questions : ℕ
  correct_points : ℕ
  incorrect_points : ℕ
  max_score : ℕ
  prob_correct : ℝ

/-- Calculate the expected score for a given test. -/
def expected_score (t : Test) : ℝ :=
  t.num_questions * (t.correct_points * t.prob_correct + t.incorrect_points * (1 - t.prob_correct))

/-- Calculate the variance of scores for a given test. -/
def score_variance (t : Test) : ℝ :=
  t.num_questions * (t.correct_points^2 * t.prob_correct + t.incorrect_points^2 * (1 - t.prob_correct) - 
    (t.correct_points * t.prob_correct + t.incorrect_points * (1 - t.prob_correct))^2)

/-- Theorem stating the expected score and variance for the given test conditions. -/
theorem test_score_properties :
  ∃ (t : Test),
    t.num_questions = 25 ∧
    t.correct_points = 4 ∧
    t.incorrect_points = 0 ∧
    t.max_score = 100 ∧
    t.prob_correct = 0.8 ∧
    expected_score t = 80 ∧
    score_variance t = 64 := by
  sorry

end NUMINAMATH_CALUDE_test_score_properties_l1512_151290


namespace NUMINAMATH_CALUDE_sports_package_channels_l1512_151264

/-- The number of channels in Larry's cable package at different stages --/
structure CablePackage where
  initial : Nat
  after_replacement : Nat
  after_reduction : Nat
  after_sports : Nat
  after_supreme : Nat
  final : Nat

/-- The number of channels in the sports package --/
def sports_package (cp : CablePackage) : Nat :=
  cp.final - cp.after_supreme

theorem sports_package_channels : ∀ cp : CablePackage,
  cp.initial = 150 →
  cp.after_replacement = cp.initial - 20 + 12 →
  cp.after_reduction = cp.after_replacement - 10 →
  cp.after_supreme = cp.after_sports + 7 →
  cp.final = 147 →
  sports_package cp = 8 := by
  sorry

#eval sports_package { 
  initial := 150,
  after_replacement := 142,
  after_reduction := 132,
  after_sports := 140,
  after_supreme := 147,
  final := 147
}

end NUMINAMATH_CALUDE_sports_package_channels_l1512_151264


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_inequality_l1512_151297

theorem negation_of_proposition (P : ℝ → Prop) :
  (∀ x : ℝ, P x) ↔ ¬(∃ x : ℝ, ¬(P x)) :=
by sorry

theorem negation_of_inequality :
  ¬(∀ x : ℝ, x^2 + 2 > 2*x) ↔ (∃ x : ℝ, x^2 + 2 ≤ 2*x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_inequality_l1512_151297
