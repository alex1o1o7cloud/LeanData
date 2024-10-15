import Mathlib

namespace NUMINAMATH_CALUDE_johns_number_is_eight_l3742_374238

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem johns_number_is_eight :
  ∃! x : ℕ, is_two_digit x ∧
    81 ≤ reverse_digits (5 * x + 18) ∧
    reverse_digits (5 * x + 18) ≤ 85 ∧
    x = 8 := by
  sorry

end NUMINAMATH_CALUDE_johns_number_is_eight_l3742_374238


namespace NUMINAMATH_CALUDE_evaluate_f_l3742_374253

def f (x : ℝ) : ℝ := 3 * x^2 + 5 * x - 7

theorem evaluate_f : 3 * f 2 - 2 * f (-2) = 55 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_f_l3742_374253


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l3742_374276

/-- Given an arithmetic sequence {a_n} where a₃ + a₅ = 10, prove that a₄ = 5 -/
theorem arithmetic_sequence_fourth_term 
  (a : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : a 3 + a 5 = 10) : 
  a 4 = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l3742_374276


namespace NUMINAMATH_CALUDE_soup_donation_per_person_l3742_374217

theorem soup_donation_per_person
  (num_shelters : ℕ)
  (people_per_shelter : ℕ)
  (total_cans : ℕ)
  (h1 : num_shelters = 6)
  (h2 : people_per_shelter = 30)
  (h3 : total_cans = 1800) :
  total_cans / (num_shelters * people_per_shelter) = 10 := by
sorry

end NUMINAMATH_CALUDE_soup_donation_per_person_l3742_374217


namespace NUMINAMATH_CALUDE_street_painting_cost_l3742_374200

/-- Calculates the total cost for painting house numbers on a street --/
def total_painting_cost (south_start : ℕ) (north_start : ℕ) (common_diff : ℕ) (houses_per_side : ℕ) : ℚ :=
  let south_end := south_start + common_diff * (houses_per_side - 1)
  let north_end := north_start + common_diff * (houses_per_side - 1)
  let south_two_digit := min houses_per_side (((99 - south_start) / common_diff) + 1)
  let north_two_digit := min houses_per_side (((99 - north_start) / common_diff) + 1)
  let south_three_digit := houses_per_side - south_two_digit
  let north_three_digit := houses_per_side - north_two_digit
  (2 * south_two_digit + 1.5 * south_three_digit + 2 * north_two_digit + 1.5 * north_three_digit : ℚ)

/-- The theorem stating the total cost for the given street configuration --/
theorem street_painting_cost :
  total_painting_cost 5 2 7 25 = 88.5 := by
  sorry

end NUMINAMATH_CALUDE_street_painting_cost_l3742_374200


namespace NUMINAMATH_CALUDE_gcd_612_840_468_l3742_374234

theorem gcd_612_840_468 : Nat.gcd 612 (Nat.gcd 840 468) = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_612_840_468_l3742_374234


namespace NUMINAMATH_CALUDE_mp3_song_count_l3742_374249

/-- Given an initial number of songs, number of deleted songs, and number of added songs,
    calculate the final number of songs on the mp3 player. -/
def final_song_count (initial : ℕ) (deleted : ℕ) (added : ℕ) : ℕ :=
  initial - deleted + added

/-- Theorem stating that given the specific numbers in the problem,
    the final song count is 64. -/
theorem mp3_song_count : final_song_count 34 14 44 = 64 := by
  sorry

end NUMINAMATH_CALUDE_mp3_song_count_l3742_374249


namespace NUMINAMATH_CALUDE_cistern_solution_l3742_374251

/-- Represents the time (in hours) it takes to fill or empty the cistern -/
structure CisternTime where
  fill : ℝ
  empty : ℝ
  both : ℝ

/-- The cistern filling problem -/
def cistern_problem (t : CisternTime) : Prop :=
  t.fill = 10 ∧ 
  t.empty = 12 ∧ 
  t.both = 60 ∧
  t.both = (t.fill * t.empty) / (t.empty - t.fill)

theorem cistern_solution :
  ∃ t : CisternTime, cistern_problem t :=
sorry

end NUMINAMATH_CALUDE_cistern_solution_l3742_374251


namespace NUMINAMATH_CALUDE_moms_balloons_l3742_374213

/-- The number of balloons Tommy's mom gave him -/
def balloons_from_mom (initial_balloons final_balloons : ℕ) : ℕ :=
  final_balloons - initial_balloons

/-- Proof that Tommy's mom gave him 34 balloons -/
theorem moms_balloons : balloons_from_mom 26 60 = 34 := by
  sorry

end NUMINAMATH_CALUDE_moms_balloons_l3742_374213


namespace NUMINAMATH_CALUDE_remainder_difference_l3742_374209

theorem remainder_difference (d r : ℤ) : d > 1 →
  1134 % d = r →
  1583 % d = r →
  2660 % d = r →
  d - r = 213 := by sorry

end NUMINAMATH_CALUDE_remainder_difference_l3742_374209


namespace NUMINAMATH_CALUDE_y_coordinate_difference_l3742_374248

/-- Given two points on a line, prove that the difference between their y-coordinates is 9 -/
theorem y_coordinate_difference (m n : ℝ) : 
  (m = (n / 3) - (2 / 5)) → 
  (m + 3 = ((n + 9) / 3) - (2 / 5)) → 
  ((n + 9) - n = 9) := by
  sorry

end NUMINAMATH_CALUDE_y_coordinate_difference_l3742_374248


namespace NUMINAMATH_CALUDE_specific_mixture_problem_l3742_374255

/-- Represents a mixture of three components -/
structure Mixture where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_to_100 : a + b + c = 100

/-- The problem of finding coefficients for mixing three mixtures to obtain a desired mixture -/
def mixture_problem (m₁ m₂ m₃ : Mixture) (desired : Mixture) :=
  ∃ (k₁ k₂ k₃ : ℝ),
    k₁ ≥ 0 ∧ k₂ ≥ 0 ∧ k₃ ≥ 0 ∧
    k₁ + k₂ + k₃ = 1 ∧
    k₁ * m₁.a + k₂ * m₂.a + k₃ * m₃.a = desired.a ∧
    k₁ * m₁.b + k₂ * m₂.b + k₃ * m₃.b = desired.b ∧
    k₁ * m₁.c + k₂ * m₂.c + k₃ * m₃.c = desired.c

/-- The specific mixture problem instance -/
theorem specific_mixture_problem :
  let m₁ : Mixture := ⟨10, 30, 60, by norm_num⟩
  let m₂ : Mixture := ⟨20, 60, 20, by norm_num⟩
  let m₃ : Mixture := ⟨80, 10, 10, by norm_num⟩
  let desired : Mixture := ⟨50, 30, 20, by norm_num⟩
  mixture_problem m₁ m₂ m₃ desired := by
    sorry

end NUMINAMATH_CALUDE_specific_mixture_problem_l3742_374255


namespace NUMINAMATH_CALUDE_roots_arithmetic_progression_implies_sum_zero_l3742_374289

theorem roots_arithmetic_progression_implies_sum_zero 
  (a b c : ℝ) 
  (p₁ p₂ q₁ q₂ : ℝ) 
  (h₁ : a * p₁^2 + b * p₁ + c = 0)
  (h₂ : a * p₂^2 + b * p₂ + c = 0)
  (h₃ : c * q₁^2 + b * q₁ + a = 0)
  (h₄ : c * q₂^2 + b * q₂ + a = 0)
  (h₅ : ∃ (d : ℝ), d ≠ 0 ∧ q₁ - p₁ = d ∧ p₂ - q₁ = d ∧ q₂ - p₂ = d)
  (h₆ : p₁ ≠ q₁ ∧ q₁ ≠ p₂ ∧ p₂ ≠ q₂) :
  a + c = 0 := by
sorry

end NUMINAMATH_CALUDE_roots_arithmetic_progression_implies_sum_zero_l3742_374289


namespace NUMINAMATH_CALUDE_medal_award_count_l3742_374202

/-- The number of sprinters in the event -/
def total_sprinters : ℕ := 12

/-- The number of American sprinters -/
def american_sprinters : ℕ := 5

/-- The number of medals to be awarded -/
def medals : ℕ := 3

/-- The maximum number of Americans that can receive medals -/
def max_american_medalists : ℕ := 2

/-- The function that calculates the number of ways to award medals -/
def award_medals : ℕ := sorry

theorem medal_award_count : award_medals = 1260 := by sorry

end NUMINAMATH_CALUDE_medal_award_count_l3742_374202


namespace NUMINAMATH_CALUDE_min_games_for_90_percent_win_l3742_374231

theorem min_games_for_90_percent_win (N : ℕ) : 
  (∀ k : ℕ, k < N → (2 + k : ℚ) / (5 + k) ≤ 9/10) ∧
  (2 + N : ℚ) / (5 + N) > 9/10 →
  N = 26 :=
sorry

end NUMINAMATH_CALUDE_min_games_for_90_percent_win_l3742_374231


namespace NUMINAMATH_CALUDE_quadratic_monotone_iff_a_geq_one_l3742_374269

/-- A quadratic function f(x) = x^2 + 2ax + b is monotonically increasing
    on [-1, +∞) if and only if a ≥ 1 -/
theorem quadratic_monotone_iff_a_geq_one (a b : ℝ) :
  (∀ x y : ℝ, -1 ≤ x → x ≤ y → x^2 + 2*a*x + b ≤ y^2 + 2*a*y + b) ↔ a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_monotone_iff_a_geq_one_l3742_374269


namespace NUMINAMATH_CALUDE_abs_m_minus_one_geq_abs_m_minus_one_l3742_374291

theorem abs_m_minus_one_geq_abs_m_minus_one (m : ℝ) : |m - 1| ≥ |m| - 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_m_minus_one_geq_abs_m_minus_one_l3742_374291


namespace NUMINAMATH_CALUDE_max_value_x_sqrt_1_minus_4x_squared_l3742_374215

theorem max_value_x_sqrt_1_minus_4x_squared :
  (∃ (x : ℝ), x > 0 ∧ x * Real.sqrt (1 - 4 * x^2) = 1/4) ∧
  (∀ (x : ℝ), x > 0 → x * Real.sqrt (1 - 4 * x^2) ≤ 1/4) := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_sqrt_1_minus_4x_squared_l3742_374215


namespace NUMINAMATH_CALUDE_smallest_undefined_fraction_value_l3742_374270

theorem smallest_undefined_fraction_value : ∃ x : ℚ, x = 2/9 ∧ 
  (∀ y : ℚ, y < x → 9*y^2 - 74*y + 8 ≠ 0) ∧ 9*x^2 - 74*x + 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_undefined_fraction_value_l3742_374270


namespace NUMINAMATH_CALUDE_b_investment_is_10000_l3742_374299

/-- Represents the capital and profit distribution in a business partnership --/
structure BusinessPartnership where
  capitalA : ℝ
  capitalB : ℝ
  capitalC : ℝ
  profitShareB : ℝ
  profitShareDiffAC : ℝ

/-- Theorem stating that under given conditions, B's investment is 10000 --/
theorem b_investment_is_10000 (bp : BusinessPartnership)
  (h1 : bp.capitalA = 8000)
  (h2 : bp.capitalC = 12000)
  (h3 : bp.profitShareB = 1900)
  (h4 : bp.profitShareDiffAC = 760) :
  bp.capitalB = 10000 := by
  sorry

#check b_investment_is_10000

end NUMINAMATH_CALUDE_b_investment_is_10000_l3742_374299


namespace NUMINAMATH_CALUDE_tv_contest_probabilities_l3742_374240

-- Define the pass rates for each level
def pass_rate_1 : ℝ := 0.6
def pass_rate_2 : ℝ := 0.5
def pass_rate_3 : ℝ := 0.4

-- Define the prize amounts
def first_prize : ℕ := 300
def second_prize : ℕ := 200

-- Define the function to calculate the probability of not winning any prize
def prob_no_prize : ℝ := 1 - pass_rate_1 + pass_rate_1 * (1 - pass_rate_2)

-- Define the function to calculate the probability of total prize money being 700,
-- given both contestants passed the first level
def prob_total_700_given_pass_1 : ℝ :=
  2 * (pass_rate_2 * (1 - pass_rate_3)) * (pass_rate_2 * pass_rate_3)

-- State the theorem
theorem tv_contest_probabilities :
  prob_no_prize = 0.7 ∧
  prob_total_700_given_pass_1 = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_tv_contest_probabilities_l3742_374240


namespace NUMINAMATH_CALUDE_yellow_face_probability_l3742_374245

/-- The probability of rolling a yellow face on a modified 10-sided die -/
theorem yellow_face_probability (total_faces : ℕ) (yellow_faces : ℕ) : 
  total_faces = 10 → yellow_faces = 4 → (yellow_faces : ℚ) / total_faces = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_yellow_face_probability_l3742_374245


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3742_374288

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a = 80 ∧ b = 150 ∧ c^2 = a^2 + b^2 → c = 170 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3742_374288


namespace NUMINAMATH_CALUDE_soda_price_before_increase_l3742_374236

/-- The original price of a can of soda -/
def original_price : ℝ := 6

/-- The percentage increase in the price of a can of soda -/
def price_increase_percentage : ℝ := 50

/-- The new price of a can of soda after the price increase -/
def new_price : ℝ := 9

/-- Theorem stating that the original price of a can of soda was 6 pounds -/
theorem soda_price_before_increase :
  original_price * (1 + price_increase_percentage / 100) = new_price :=
by sorry

end NUMINAMATH_CALUDE_soda_price_before_increase_l3742_374236


namespace NUMINAMATH_CALUDE_rational_expressions_theorem_l3742_374235

theorem rational_expressions_theorem 
  (a b c : ℚ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) :
  (a < 0 → a / |a| = -1) ∧ 
  (∃ m : ℚ, m = -2 ∧ 
    ∀ x y z : ℚ, x ≠ 0 → y ≠ 0 → z ≠ 0 → 
      m ≤ (x*y/|x*y| + |y*z|/(y*z) + z*x/|z*x| + |x*y*z|/(x*y*z))) :=
by sorry

end NUMINAMATH_CALUDE_rational_expressions_theorem_l3742_374235


namespace NUMINAMATH_CALUDE_alyssa_cans_collected_l3742_374227

theorem alyssa_cans_collected (total_cans : ℕ) (abigail_cans : ℕ) (cans_needed : ℕ) 
  (h1 : total_cans = 100)
  (h2 : abigail_cans = 43)
  (h3 : cans_needed = 27) :
  total_cans - (abigail_cans + cans_needed) = 30 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_cans_collected_l3742_374227


namespace NUMINAMATH_CALUDE_debate_team_girls_l3742_374250

/-- The number of boys on the debate team -/
def num_boys : ℕ := 11

/-- The number of groups the team can be split into -/
def num_groups : ℕ := 8

/-- The number of students in each group -/
def students_per_group : ℕ := 7

/-- The total number of students on the debate team -/
def total_students : ℕ := num_groups * students_per_group

/-- The number of girls on the debate team -/
def num_girls : ℕ := total_students - num_boys

theorem debate_team_girls : num_girls = 45 := by
  sorry

end NUMINAMATH_CALUDE_debate_team_girls_l3742_374250


namespace NUMINAMATH_CALUDE_combustible_ice_reserves_scientific_notation_l3742_374216

/-- Expresses a number in scientific notation -/
def scientific_notation (n : ℕ) : ℝ × ℤ :=
  sorry

theorem combustible_ice_reserves_scientific_notation :
  scientific_notation 150000000000 = (1.5, 11) :=
sorry

end NUMINAMATH_CALUDE_combustible_ice_reserves_scientific_notation_l3742_374216


namespace NUMINAMATH_CALUDE_students_answering_one_question_l3742_374265

/-- Represents the number of questions answered by students in each grade -/
structure GradeAnswers :=
  (g1 g2 g3 g4 g5 : Nat)

/-- The problem setup -/
structure ProblemSetup :=
  (total_students : Nat)
  (total_grades : Nat)
  (total_questions : Nat)
  (grade_answers : GradeAnswers)

/-- The conditions of the problem -/
def satisfies_conditions (setup : ProblemSetup) : Prop :=
  setup.total_students = 30 ∧
  setup.total_grades = 5 ∧
  setup.total_questions = 40 ∧
  setup.grade_answers.g1 < setup.grade_answers.g2 ∧
  setup.grade_answers.g2 < setup.grade_answers.g3 ∧
  setup.grade_answers.g3 < setup.grade_answers.g4 ∧
  setup.grade_answers.g4 < setup.grade_answers.g5 ∧
  setup.grade_answers.g1 ≥ 1 ∧
  setup.grade_answers.g2 ≥ 1 ∧
  setup.grade_answers.g3 ≥ 1 ∧
  setup.grade_answers.g4 ≥ 1 ∧
  setup.grade_answers.g5 ≥ 1

/-- The theorem to be proved -/
theorem students_answering_one_question (setup : ProblemSetup) 
  (h : satisfies_conditions setup) : 
  setup.total_students - (setup.total_questions - (setup.grade_answers.g1 + 
  setup.grade_answers.g2 + setup.grade_answers.g3 + setup.grade_answers.g4 + 
  setup.grade_answers.g5)) = 26 :=
by sorry

end NUMINAMATH_CALUDE_students_answering_one_question_l3742_374265


namespace NUMINAMATH_CALUDE_perimeter_is_200_l3742_374218

/-- A rectangle with an inscribed rhombus -/
structure RectangleWithRhombus where
  -- Length of half of side AB
  wa : ℝ
  -- Length of half of side BC
  xb : ℝ
  -- Length of diagonal WY of the rhombus
  wy : ℝ

/-- The perimeter of the rectangle -/
def perimeter (r : RectangleWithRhombus) : ℝ :=
  2 * (2 * r.wa + 2 * r.xb)

/-- Theorem: The perimeter of the rectangle is 200 -/
theorem perimeter_is_200 (r : RectangleWithRhombus)
    (h1 : r.wa = 20)
    (h2 : r.xb = 30)
    (h3 : r.wy = 50) :
    perimeter r = 200 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_is_200_l3742_374218


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3742_374294

theorem fractional_equation_solution (x : ℝ) (hx : x ≠ 0) (hx2 : 2 * x - 1 ≠ 0) :
  (2 / x = 3 / (2 * x - 1)) ↔ x = 2 := by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3742_374294


namespace NUMINAMATH_CALUDE_nicole_clothes_proof_l3742_374204

/-- Calculates the total number of clothing pieces Nicole ends up with --/
def nicole_total_clothes (nicole_initial : ℕ) : ℕ :=
  let sister1 := nicole_initial / 2
  let sister2 := nicole_initial + 2
  let sister3 := (nicole_initial + sister1 + sister2) / 3
  nicole_initial + sister1 + sister2 + sister3

/-- Proves that Nicole ends up with 36 pieces of clothing --/
theorem nicole_clothes_proof :
  nicole_total_clothes 10 = 36 := by
  sorry

#eval nicole_total_clothes 10

end NUMINAMATH_CALUDE_nicole_clothes_proof_l3742_374204


namespace NUMINAMATH_CALUDE_sum_of_extremes_3point5_l3742_374201

/-- A number that rounds to 3.5 when rounded to one decimal place -/
def RoundsTo3Point5 (x : ℝ) : Prop :=
  (x ≥ 3.45) ∧ (x < 3.55)

/-- The theorem stating the sum of the largest and smallest 3-digit decimals
    that round to 3.5 is 6.99 -/
theorem sum_of_extremes_3point5 :
  ∃ (min max : ℝ),
    (∀ x, RoundsTo3Point5 x → x ≥ min) ∧
    (∀ x, RoundsTo3Point5 x → x ≤ max) ∧
    (RoundsTo3Point5 min) ∧
    (RoundsTo3Point5 max) ∧
    (min + max = 6.99) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_extremes_3point5_l3742_374201


namespace NUMINAMATH_CALUDE_substitution_result_l3742_374273

noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

theorem substitution_result (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1 ∧ 1 + 3 * x^2 ≠ 0) :
  F ((3 * x - x^3) / (1 + 3 * x^2)) = 3 * F x :=
by sorry

end NUMINAMATH_CALUDE_substitution_result_l3742_374273


namespace NUMINAMATH_CALUDE_sin_theta_value_l3742_374284

theorem sin_theta_value (θ : Real) 
  (h1 : 10 * Real.tan θ = 4 * Real.cos θ) 
  (h2 : 0 < θ) (h3 : θ < π) : 
  Real.sin θ = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_value_l3742_374284


namespace NUMINAMATH_CALUDE_prime_divisor_equality_l3742_374259

theorem prime_divisor_equality (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hdiv : p ∣ q) : p = q := by
  sorry

end NUMINAMATH_CALUDE_prime_divisor_equality_l3742_374259


namespace NUMINAMATH_CALUDE_min_value_m_plus_n_l3742_374262

theorem min_value_m_plus_n (m n : ℝ) : 
  m * 1 + n * 1 - 3 * m * n = 0 → 
  m * n > 0 → 
  m + n ≥ 4/3 ∧ ∃ (m₀ n₀ : ℝ), m₀ * 1 + n₀ * 1 - 3 * m₀ * n₀ = 0 ∧ m₀ * n₀ > 0 ∧ m₀ + n₀ = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_m_plus_n_l3742_374262


namespace NUMINAMATH_CALUDE_discount_order_difference_l3742_374207

def original_price : ℝ := 50
def flat_discount : ℝ := 5
def percentage_discount : ℝ := 0.1

def price_flat_then_percent : ℝ := (original_price - flat_discount) * (1 - percentage_discount)
def price_percent_then_flat : ℝ := original_price * (1 - percentage_discount) - flat_discount

theorem discount_order_difference :
  price_flat_then_percent - price_percent_then_flat = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_discount_order_difference_l3742_374207


namespace NUMINAMATH_CALUDE_larry_wins_probability_l3742_374212

theorem larry_wins_probability (p : ℝ) (q : ℝ) (hp : p = 1/3) (hq : q = 1/4) :
  let win_prob := p / (1 - (1 - p) * (1 - q))
  win_prob = 2/3 := by sorry

end NUMINAMATH_CALUDE_larry_wins_probability_l3742_374212


namespace NUMINAMATH_CALUDE_absolute_value_of_negative_l3742_374225

theorem absolute_value_of_negative (x : ℝ) : x < 0 → |x| = -x := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_negative_l3742_374225


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l3742_374222

/-- A rectangle inscribed in a triangle -/
structure InscribedRectangle where
  /-- The length of the rectangle's side along the triangle's base -/
  base_length : ℝ
  /-- The width of the rectangle -/
  width : ℝ
  /-- The length of the triangle's base -/
  triangle_base : ℝ
  /-- The height of the triangle -/
  triangle_height : ℝ
  /-- The width is one-third of the base length -/
  width_constraint : width = base_length / 3
  /-- The triangle's base is 15 inches -/
  triangle_base_length : triangle_base = 15
  /-- The triangle's height is 12 inches -/
  triangle_height_value : triangle_height = 12

/-- The area of the inscribed rectangle -/
def area (r : InscribedRectangle) : ℝ := r.base_length * r.width

/-- Theorem: The area of the inscribed rectangle is 10800/289 square inches -/
theorem inscribed_rectangle_area (r : InscribedRectangle) : area r = 10800 / 289 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l3742_374222


namespace NUMINAMATH_CALUDE_exterior_angle_of_regular_polygon_l3742_374287

theorem exterior_angle_of_regular_polygon (n : ℕ) (h : (n - 2) * 180 = 720) :
  360 / n = 60 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_of_regular_polygon_l3742_374287


namespace NUMINAMATH_CALUDE_y_derivative_l3742_374260

noncomputable def y (x : ℝ) : ℝ :=
  (3^x * (Real.log 3 * Real.sin (2*x) - 2 * Real.cos (2*x))) / ((Real.log 3)^2 + 4)

theorem y_derivative (x : ℝ) :
  deriv y x = 3^x * Real.sin (2*x) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l3742_374260


namespace NUMINAMATH_CALUDE_walkers_speed_l3742_374296

/-- Proves that given the conditions of the problem, A's walking speed is 10 kmph -/
theorem walkers_speed (v : ℝ) : 
  (∃ t : ℝ, v * (t + 7) = 20 * t) →  -- B catches up with A
  (∃ t : ℝ, v * (t + 7) = 140) →     -- Distance traveled is 140 km
  v = 10 := by sorry

end NUMINAMATH_CALUDE_walkers_speed_l3742_374296


namespace NUMINAMATH_CALUDE_cos_2alpha_value_l3742_374286

theorem cos_2alpha_value (α : ℝ) (h : Real.sin (α - 3 * Real.pi / 2) = 3 / 5) : 
  Real.cos (2 * α) = -7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_value_l3742_374286


namespace NUMINAMATH_CALUDE_a_10_equals_505_l3742_374223

def sequence_a (n : ℕ) : ℕ :=
  if n = 1 then 1
  else
    let start := (n * (n - 1)) / 2 + 1
    (start + start + n - 1) * n / 2

theorem a_10_equals_505 : sequence_a 10 = 505 := by
  sorry

end NUMINAMATH_CALUDE_a_10_equals_505_l3742_374223


namespace NUMINAMATH_CALUDE_b_over_a_range_l3742_374242

-- Define an acute triangle
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π
  sides_positive : a > 0 ∧ b > 0 ∧ c > 0
  sine_law : a / Real.sin A = b / Real.sin B
  B_eq_2A : B = 2 * A

-- Theorem statement
theorem b_over_a_range (t : AcuteTriangle) : Real.sqrt 2 < t.b / t.a ∧ t.b / t.a < Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_b_over_a_range_l3742_374242


namespace NUMINAMATH_CALUDE_carter_reading_rate_l3742_374257

/-- The number of pages Oliver can read in 1 hour -/
def oliver_pages : ℕ := 40

/-- The number of pages Lucy can read in 1 hour -/
def lucy_pages : ℕ := oliver_pages + 20

/-- The number of pages Carter can read in 1 hour -/
def carter_pages : ℕ := lucy_pages / 2

/-- Proves that Carter can read 30 pages in 1 hour given the conditions -/
theorem carter_reading_rate : carter_pages = 30 := by
  sorry

end NUMINAMATH_CALUDE_carter_reading_rate_l3742_374257


namespace NUMINAMATH_CALUDE_expansion_coefficient_sum_l3742_374264

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The coefficient of x^k in the expansion of (1-2x)^n -/
def coeff (n k : ℕ) : ℤ :=
  (-2)^k * binomial n k

theorem expansion_coefficient_sum (n : ℕ) 
  (h : coeff n 1 + coeff n 4 = 70) : 
  coeff n 5 = -32 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_sum_l3742_374264


namespace NUMINAMATH_CALUDE_edward_binders_l3742_374226

/-- The number of baseball cards Edward had -/
def total_cards : ℕ := 763

/-- The number of cards in each binder -/
def cards_per_binder : ℕ := 109

/-- The number of binders Edward had -/
def number_of_binders : ℕ := total_cards / cards_per_binder

theorem edward_binders : number_of_binders = 7 := by
  sorry

end NUMINAMATH_CALUDE_edward_binders_l3742_374226


namespace NUMINAMATH_CALUDE_f_of_4_equals_22_l3742_374237

/-- Given a function f(x) = 5x + 2, prove that f(4) = 22 -/
theorem f_of_4_equals_22 :
  let f : ℝ → ℝ := λ x ↦ 5 * x + 2
  f 4 = 22 := by sorry

end NUMINAMATH_CALUDE_f_of_4_equals_22_l3742_374237


namespace NUMINAMATH_CALUDE_function_sum_zero_at_five_sevenths_l3742_374203

-- Define the functions f and g
def f (x : ℝ) : ℝ := 5 * x - 6
def g (x : ℝ) : ℝ := 2 * x + 1

-- Theorem statement
theorem function_sum_zero_at_five_sevenths :
  ∃! a : ℝ, f a + g a = 0 ∧ a = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_function_sum_zero_at_five_sevenths_l3742_374203


namespace NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l3742_374254

/-- The coefficient of x^2 in the expansion of (1/√x + x)^8 -/
def coefficient_x_squared : ℕ := 70

/-- The binomial coefficient (8 choose 4) -/
def binomial_8_4 : ℕ := 70

theorem coefficient_x_squared_expansion :
  coefficient_x_squared = binomial_8_4 := by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l3742_374254


namespace NUMINAMATH_CALUDE_cheese_slices_total_l3742_374293

/-- The number of cheese slices used for ham sandwiches -/
def ham_cheese_slices (num_ham_sandwiches : ℕ) (cheese_per_ham : ℕ) : ℕ :=
  num_ham_sandwiches * cheese_per_ham

/-- The number of cheese slices used for grilled cheese sandwiches -/
def grilled_cheese_slices (num_grilled_cheese : ℕ) (cheese_per_grilled : ℕ) : ℕ :=
  num_grilled_cheese * cheese_per_grilled

/-- The total number of cheese slices used for both types of sandwiches -/
def total_cheese_slices (ham_slices : ℕ) (grilled_slices : ℕ) : ℕ :=
  ham_slices + grilled_slices

/-- Theorem: The total number of cheese slices used for 10 ham sandwiches
    (each requiring 2 slices) and 10 grilled cheese sandwiches
    (each requiring 3 slices) is equal to 50. -/
theorem cheese_slices_total :
  total_cheese_slices
    (ham_cheese_slices 10 2)
    (grilled_cheese_slices 10 3) = 50 := by
  sorry

end NUMINAMATH_CALUDE_cheese_slices_total_l3742_374293


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3742_374274

theorem simplify_and_evaluate (a b : ℝ) :
  -(-a^2 + 2*a*b + b^2) + (-a^2 - a*b + b^2) = -3*a*b ∧
  (a*b = 1 → -(-a^2 + 2*a*b + b^2) + (-a^2 - a*b + b^2) = -3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3742_374274


namespace NUMINAMATH_CALUDE_root_equation_sum_l3742_374267

theorem root_equation_sum (a b c : ℝ) : 
  (∀ x : ℝ, x^2 + 3*x - 1 = 0 → x^4 + a*x^2 + b*x + c = 0) →
  a + b + 4*c = -7 := by
sorry

end NUMINAMATH_CALUDE_root_equation_sum_l3742_374267


namespace NUMINAMATH_CALUDE_decimal_digit_17_99_l3742_374298

/-- The fraction we're examining -/
def f : ℚ := 17 / 99

/-- The position of the digit we're looking for -/
def n : ℕ := 150

/-- Function to get the nth digit after the decimal point in the decimal representation of a rational number -/
noncomputable def nth_decimal_digit (q : ℚ) (n : ℕ) : ℕ := sorry

/-- Theorem stating that the 150th digit after the decimal point in 17/99 is 7 -/
theorem decimal_digit_17_99 : nth_decimal_digit f n = 7 := by sorry

end NUMINAMATH_CALUDE_decimal_digit_17_99_l3742_374298


namespace NUMINAMATH_CALUDE_magazines_per_bookshelf_l3742_374279

theorem magazines_per_bookshelf
  (num_books : ℕ)
  (num_bookshelves : ℕ)
  (total_items : ℕ)
  (h1 : num_books = 23)
  (h2 : num_bookshelves = 29)
  (h3 : total_items = 2436) :
  (total_items - num_books) / num_bookshelves = 83 := by
sorry

end NUMINAMATH_CALUDE_magazines_per_bookshelf_l3742_374279


namespace NUMINAMATH_CALUDE_max_min_difference_c_l3742_374271

theorem max_min_difference_c (a b c : ℝ) 
  (sum_eq : a + b + c = 3) 
  (sum_squares_eq : a^2 + b^2 + c^2 = 18) : 
  ∃ (c_max c_min : ℝ), 
    (∀ c', a + b + c' = 3 ∧ a^2 + b^2 + c'^2 = 18 → c' ≤ c_max ∧ c' ≥ c_min) ∧
    c_max - c_min = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_min_difference_c_l3742_374271


namespace NUMINAMATH_CALUDE_prop_evaluation_l3742_374228

-- Define the propositions p and q
def p (x y : ℝ) : Prop := (x > y) → (-x < -y)
def q (x y : ℝ) : Prop := (x < y) → (x^2 < y^2)

-- State the theorem
theorem prop_evaluation : ∃ (x y : ℝ), (p x y ∨ q x y) ∧ (p x y ∧ ¬(q x y)) := by
  sorry

end NUMINAMATH_CALUDE_prop_evaluation_l3742_374228


namespace NUMINAMATH_CALUDE_min_value_of_3a_plus_2_l3742_374258

theorem min_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^2 + 7 * a + 6 = 5) :
  ∃ (m : ℝ), (3 * a + 2 ≥ m) ∧ (∀ x, 8 * x^2 + 7 * x + 6 = 5 → 3 * x + 2 ≥ m) ∧ m = -1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_3a_plus_2_l3742_374258


namespace NUMINAMATH_CALUDE_aaron_sweaters_count_l3742_374263

/-- The number of sweaters Aaron made -/
def aaron_sweaters : ℕ := 5

/-- The number of scarves Aaron made -/
def aaron_scarves : ℕ := 10

/-- The number of sweaters Enid made -/
def enid_sweaters : ℕ := 8

/-- The number of balls of wool used for one scarf -/
def wool_per_scarf : ℕ := 3

/-- The number of balls of wool used for one sweater -/
def wool_per_sweater : ℕ := 4

/-- The total number of balls of wool used -/
def total_wool : ℕ := 82

theorem aaron_sweaters_count : 
  aaron_sweaters * wool_per_sweater + 
  aaron_scarves * wool_per_scarf + 
  enid_sweaters * wool_per_sweater = total_wool :=
sorry

end NUMINAMATH_CALUDE_aaron_sweaters_count_l3742_374263


namespace NUMINAMATH_CALUDE_sandy_marbles_count_l3742_374221

/-- The number of red marbles Jessica has -/
def jessica_marbles : ℕ := 3 * 12

/-- The factor by which Sandy has more marbles than Jessica -/
def sandy_factor : ℕ := 4

/-- The number of red marbles Sandy has -/
def sandy_marbles : ℕ := sandy_factor * jessica_marbles

theorem sandy_marbles_count : sandy_marbles = 144 := by
  sorry

end NUMINAMATH_CALUDE_sandy_marbles_count_l3742_374221


namespace NUMINAMATH_CALUDE_line_slope_l3742_374295

/-- Given a line with equation 3y + 2x = 6x - 9, its slope is -4/3 -/
theorem line_slope (x y : ℝ) : 3*y + 2*x = 6*x - 9 → (y - 3 = (-4/3) * (x - 0)) := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l3742_374295


namespace NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l3742_374297

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17

theorem sum_first_six_primes_mod_seventh_prime :
  (first_six_primes.sum % seventh_prime) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l3742_374297


namespace NUMINAMATH_CALUDE_dealer_profit_is_25_percent_l3742_374278

/-- Represents a dishonest dealer's selling strategy -/
structure DishonestDealer where
  weight_reduction : ℝ  -- Percentage reduction in weight
  impurity_addition : ℝ  -- Percentage of impurities added
  
/-- Calculates the net profit percentage for a dishonest dealer -/
def net_profit_percentage (dealer : DishonestDealer) : ℝ :=
  sorry

/-- Theorem stating that the net profit percentage is 25% for the given conditions -/
theorem dealer_profit_is_25_percent :
  let dealer : DishonestDealer := { weight_reduction := 0.20, impurity_addition := 0.25 }
  net_profit_percentage dealer = 0.25 := by sorry

end NUMINAMATH_CALUDE_dealer_profit_is_25_percent_l3742_374278


namespace NUMINAMATH_CALUDE_set_operations_and_range_l3742_374277

def A : Set ℝ := {x | 4 ≤ x ∧ x < 8}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

theorem set_operations_and_range :
  (A ∪ B = {x | 2 < x ∧ x < 10}) ∧
  ((Set.univ \ A) ∩ B = {x | (8 ≤ x ∧ x < 10) ∨ (2 < x ∧ x < 4)}) ∧
  (∀ a : ℝ, (A ∩ C a).Nonempty → 4 ≤ a) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_and_range_l3742_374277


namespace NUMINAMATH_CALUDE_january_oil_bill_l3742_374211

theorem january_oil_bill (february_bill january_bill : ℚ) : 
  (february_bill / january_bill = 5 / 4) →
  ((february_bill + 45) / january_bill = 3 / 2) →
  january_bill = 180 := by
sorry

end NUMINAMATH_CALUDE_january_oil_bill_l3742_374211


namespace NUMINAMATH_CALUDE_jenny_rommel_age_difference_l3742_374268

/-- Given the ages and relationships of Tim, Rommel, and Jenny, prove that Jenny is 2 years older than Rommel -/
theorem jenny_rommel_age_difference :
  ∀ (tim_age rommel_age jenny_age : ℕ),
  tim_age = 5 →
  rommel_age = 3 * tim_age →
  jenny_age = tim_age + 12 →
  jenny_age - rommel_age = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_jenny_rommel_age_difference_l3742_374268


namespace NUMINAMATH_CALUDE_multiplication_and_addition_l3742_374280

theorem multiplication_and_addition : 2 * (-2) + (-3) = -7 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_and_addition_l3742_374280


namespace NUMINAMATH_CALUDE_sum_of_arguments_l3742_374229

def complex_equation (z : ℂ) : Prop := z^6 = 64 * Complex.I

theorem sum_of_arguments (z₁ z₂ z₃ z₄ z₅ z₆ : ℂ) 
  (h₁ : complex_equation z₁)
  (h₂ : complex_equation z₂)
  (h₃ : complex_equation z₃)
  (h₄ : complex_equation z₄)
  (h₅ : complex_equation z₅)
  (h₆ : complex_equation z₆)
  (distinct : z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₁ ≠ z₅ ∧ z₁ ≠ z₆ ∧
              z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₂ ≠ z₅ ∧ z₂ ≠ z₆ ∧
              z₃ ≠ z₄ ∧ z₃ ≠ z₅ ∧ z₃ ≠ z₆ ∧
              z₄ ≠ z₅ ∧ z₄ ≠ z₆ ∧
              z₅ ≠ z₆) :
  (Complex.arg z₁ + Complex.arg z₂ + Complex.arg z₃ + 
   Complex.arg z₄ + Complex.arg z₅ + Complex.arg z₆) * (180 / Real.pi) = 990 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_arguments_l3742_374229


namespace NUMINAMATH_CALUDE_f_negative_a_l3742_374239

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log (-x + Real.sqrt (x^2 + 1)) + 1

theorem f_negative_a (a : ℝ) (h : f a = 11) : f (-a) = -9 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_a_l3742_374239


namespace NUMINAMATH_CALUDE_fraction_of_product_l3742_374244

theorem fraction_of_product (x : ℚ) : x * ((3 / 4 : ℚ) * (2 / 5 : ℚ) * 5040) = 756.0000000000001 → x = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_product_l3742_374244


namespace NUMINAMATH_CALUDE_prob_log_inequality_l3742_374247

open Real MeasureTheory ProbabilityTheory

/-- The probability of selecting a number x from [0,3] such that -1 ≤ log_(1/2)(x + 1/2) ≤ 1 is 1/2 -/
theorem prob_log_inequality (μ : Measure ℝ) [IsProbabilityMeasure μ] : 
  μ {x ∈ Set.Icc 0 3 | -1 ≤ log (x + 1/2) / log (1/2) ∧ log (x + 1/2) / log (1/2) ≤ 1} = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_prob_log_inequality_l3742_374247


namespace NUMINAMATH_CALUDE_m_range_l3742_374232

def A (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

def B : Set ℝ := {x | x^2 - 2*x - 15 ≤ 0}

theorem m_range (m : ℝ) :
  (∃ x, x ∈ A m) ∧ (A m ⊆ B) → 2 ≤ m ∧ m ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l3742_374232


namespace NUMINAMATH_CALUDE_cristinas_croissants_l3742_374285

theorem cristinas_croissants (total_croissants : ℕ) (num_guests : ℕ) 
  (h1 : total_croissants = 17) 
  (h2 : num_guests = 7) : 
  total_croissants % num_guests = 3 := by
  sorry

end NUMINAMATH_CALUDE_cristinas_croissants_l3742_374285


namespace NUMINAMATH_CALUDE_fraction_product_l3742_374283

theorem fraction_product : (2 : ℚ) / 3 * (4 : ℚ) / 9 = (8 : ℚ) / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l3742_374283


namespace NUMINAMATH_CALUDE_parabola_sum_l3742_374272

/-- A parabola with equation y = px^2 + qx + r, vertex (3, 10), vertical axis of symmetry, and containing the point (0, 7) -/
structure Parabola where
  p : ℝ
  q : ℝ
  r : ℝ
  vertex_x : ℝ := 3
  vertex_y : ℝ := 10
  point_x : ℝ := 0
  point_y : ℝ := 7
  eq_at_vertex : 10 = p * 3^2 + q * 3 + r
  eq_at_point : 7 = p * 0^2 + q * 0 + r
  vertical_symmetry : ∀ (x : ℝ), p * (vertex_x - x)^2 + vertex_y = p * (vertex_x + x)^2 + vertex_y

theorem parabola_sum (par : Parabola) : par.p + par.q + par.r = 26/3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_sum_l3742_374272


namespace NUMINAMATH_CALUDE_tommy_balloons_l3742_374220

theorem tommy_balloons (x : ℕ) : x + 34 = 60 → x = 26 := by
  sorry

end NUMINAMATH_CALUDE_tommy_balloons_l3742_374220


namespace NUMINAMATH_CALUDE_bike_cost_calculation_l3742_374230

/-- The cost of Trey's new bike -/
def bike_cost : ℕ := 112

/-- The number of days in two weeks -/
def days_in_two_weeks : ℕ := 14

/-- The number of bracelets Trey needs to sell each day -/
def bracelets_per_day : ℕ := 8

/-- The price of each bracelet in dollars -/
def price_per_bracelet : ℕ := 1

/-- Theorem stating that the bike cost is equal to the product of days, bracelets per day, and price per bracelet -/
theorem bike_cost_calculation : 
  bike_cost = days_in_two_weeks * bracelets_per_day * price_per_bracelet := by
  sorry

end NUMINAMATH_CALUDE_bike_cost_calculation_l3742_374230


namespace NUMINAMATH_CALUDE_area_equality_l3742_374205

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  E : Point
  F : Point
  G : Point
  H : Point

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := sorry

/-- Calculates the area of the part of the parallelogram below the x-axis -/
def areaBelow (p : Parallelogram) : ℝ := sorry

/-- Calculates the area of the part of the parallelogram above the x-axis -/
def areaAbove (p : Parallelogram) : ℝ := sorry

/-- Theorem: For the given parallelogram, the area below the x-axis equals the area above -/
theorem area_equality (p : Parallelogram) 
  (h1 : p.E = ⟨-1, 2⟩) 
  (h2 : p.F = ⟨5, 2⟩) 
  (h3 : p.G = ⟨1, -2⟩) 
  (h4 : p.H = ⟨-5, -2⟩) : 
  areaBelow p = areaAbove p := by sorry

end NUMINAMATH_CALUDE_area_equality_l3742_374205


namespace NUMINAMATH_CALUDE_xiao_li_score_l3742_374281

/-- Calculates the comprehensive score based on content and culture scores -/
def comprehensive_score (content_score culture_score : ℝ) : ℝ :=
  0.4 * content_score + 0.6 * culture_score

/-- Theorem stating that Xiao Li's comprehensive score is 86 points -/
theorem xiao_li_score : comprehensive_score 80 90 = 86 := by
  sorry

end NUMINAMATH_CALUDE_xiao_li_score_l3742_374281


namespace NUMINAMATH_CALUDE_purple_to_seafoam_ratio_is_one_fourth_l3742_374290

/-- The number of skirts in Azure Valley -/
def azure_skirts : ℕ := 60

/-- The number of skirts in Purple Valley -/
def purple_skirts : ℕ := 10

/-- The number of skirts in Seafoam Valley -/
def seafoam_skirts : ℕ := (2 * azure_skirts) / 3

/-- The ratio of skirts in Purple Valley to Seafoam Valley -/
def purple_to_seafoam_ratio : ℚ := purple_skirts / seafoam_skirts

theorem purple_to_seafoam_ratio_is_one_fourth :
  purple_to_seafoam_ratio = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_purple_to_seafoam_ratio_is_one_fourth_l3742_374290


namespace NUMINAMATH_CALUDE_inverse_113_mod_114_l3742_374241

theorem inverse_113_mod_114 : ∃ x : ℕ, x ≡ 113 [ZMOD 114] ∧ 113 * x ≡ 1 [ZMOD 114] :=
by sorry

end NUMINAMATH_CALUDE_inverse_113_mod_114_l3742_374241


namespace NUMINAMATH_CALUDE_problem_solution_l3742_374282

theorem problem_solution : 
  (∃ x : ℝ, 1/x < x + 1) ∧ 
  (¬(∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3742_374282


namespace NUMINAMATH_CALUDE_chessboard_coloring_l3742_374252

/-- A color used to paint the chessboard squares -/
inductive Color
| Red
| Green
| Blue

/-- A chessboard configuration is a function from (row, column) to Color -/
def ChessboardConfig := Fin 4 → Fin 19 → Color

/-- The theorem statement -/
theorem chessboard_coloring (config : ChessboardConfig) :
  ∃ (r₁ r₂ : Fin 4) (c₁ c₂ : Fin 19),
    r₁ ≠ r₂ ∧ c₁ ≠ c₂ ∧
    config r₁ c₁ = config r₁ c₂ ∧
    config r₁ c₁ = config r₂ c₁ ∧
    config r₁ c₁ = config r₂ c₂ :=
sorry

end NUMINAMATH_CALUDE_chessboard_coloring_l3742_374252


namespace NUMINAMATH_CALUDE_zeros_of_f_l3742_374210

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 16*x

-- State the theorem
theorem zeros_of_f :
  {x : ℝ | f x = 0} = {-4, 0, 4} := by sorry

end NUMINAMATH_CALUDE_zeros_of_f_l3742_374210


namespace NUMINAMATH_CALUDE_rock_skipping_total_l3742_374214

/-- The number of times Bob can skip a rock -/
def bob_skips : ℕ := 12

/-- The number of times Jim can skip a rock -/
def jim_skips : ℕ := 15

/-- The number of rocks each person skipped -/
def rocks_skipped : ℕ := 10

/-- The total number of skips achieved by Bob and Jim -/
def total_skips : ℕ := bob_skips * rocks_skipped + jim_skips * rocks_skipped

theorem rock_skipping_total : total_skips = 270 := by
  sorry

end NUMINAMATH_CALUDE_rock_skipping_total_l3742_374214


namespace NUMINAMATH_CALUDE_average_daily_low_temperature_l3742_374266

def daily_low_temperatures : List ℝ := [40, 47, 45, 41, 39, 43]

theorem average_daily_low_temperature :
  (daily_low_temperatures.sum / daily_low_temperatures.length : ℝ) = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_average_daily_low_temperature_l3742_374266


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l3742_374206

theorem fraction_sum_equals_decimal : (2 / 5 : ℚ) + (2 / 50 : ℚ) + (2 / 500 : ℚ) = 0.444 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l3742_374206


namespace NUMINAMATH_CALUDE_max_value_reciprocal_sum_l3742_374256

theorem max_value_reciprocal_sum (x y a b : ℝ) 
  (ha : a > 1) 
  (hb : b > 1) 
  (hax : a^x = 3) 
  (hby : b^y = 3) 
  (hab : a + b = 2 * Real.sqrt 3) : 
  (∃ (z : ℝ), ∀ (w : ℝ), 1/x + 1/y ≤ w → w ≤ z) ∧ 
  (∃ (x0 y0 : ℝ), 1/x0 + 1/y0 = 1 ∧ 
    a^x0 = 3 ∧ b^y0 = 3 ∧ a + b = 2 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_max_value_reciprocal_sum_l3742_374256


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_of_f_l3742_374224

/-- The function f(x) = 2 - x^2 -/
def f (x : ℝ) : ℝ := 2 - x^2

/-- The monotonic decreasing interval of f(x) = 2 - x^2 is (0, +∞) -/
theorem monotonic_decreasing_interval_of_f :
  ∀ x y, 0 < x → x < y → f y < f x :=
sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_of_f_l3742_374224


namespace NUMINAMATH_CALUDE_test_questions_missed_l3742_374261

theorem test_questions_missed (friend_missed : ℕ) (your_missed : ℕ) : 
  your_missed = 5 * friend_missed →
  your_missed + friend_missed = 216 →
  your_missed = 180 := by
sorry

end NUMINAMATH_CALUDE_test_questions_missed_l3742_374261


namespace NUMINAMATH_CALUDE_prime_product_range_l3742_374233

theorem prime_product_range (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  15 < p * q → p * q ≤ 36 → 8 < q → q < 24 → p * q = 33 → p = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_product_range_l3742_374233


namespace NUMINAMATH_CALUDE_sum_of_cubes_l3742_374219

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 3) (h2 : x * y = 2) (h3 : x = 2 * y) : x^3 + y^3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l3742_374219


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3742_374292

theorem sufficient_not_necessary (x : ℝ) : 
  (∀ x, Real.sqrt (x + 2) - Real.sqrt (1 - 2*x) > 0 → (x + 1) / (x - 1) ≤ 0) ∧
  (∃ x, (x + 1) / (x - 1) ≤ 0 ∧ Real.sqrt (x + 2) - Real.sqrt (1 - 2*x) ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3742_374292


namespace NUMINAMATH_CALUDE_antonov_remaining_packs_l3742_374246

/-- Calculates the number of candy packs remaining after giving away one pack -/
def remaining_packs (total_candies : ℕ) (candies_per_pack : ℕ) : ℕ :=
  (total_candies - candies_per_pack) / candies_per_pack

/-- Proves that Antonov has 2 packs of candy remaining -/
theorem antonov_remaining_packs :
  let total_candies : ℕ := 60
  let candies_per_pack : ℕ := 20
  remaining_packs total_candies candies_per_pack = 2 := by
  sorry

end NUMINAMATH_CALUDE_antonov_remaining_packs_l3742_374246


namespace NUMINAMATH_CALUDE_min_tiles_to_cover_region_l3742_374243

/-- The number of tiles needed to cover a rectangular region -/
def tiles_needed (tile_width : ℕ) (tile_height : ℕ) (region_width : ℕ) (region_height : ℕ) : ℕ :=
  let region_area := region_width * region_height
  let tile_area := tile_width * tile_height
  (region_area + tile_area - 1) / tile_area

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℕ := 12

theorem min_tiles_to_cover_region :
  tiles_needed 5 7 (3 * feet_to_inches) (7 * feet_to_inches) = 87 := by
  sorry

#eval tiles_needed 5 7 (3 * feet_to_inches) (7 * feet_to_inches)

end NUMINAMATH_CALUDE_min_tiles_to_cover_region_l3742_374243


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l3742_374275

theorem negative_fraction_comparison : -3/4 > -4/5 := by sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l3742_374275


namespace NUMINAMATH_CALUDE_geometric_sequence_tenth_term_l3742_374208

theorem geometric_sequence_tenth_term
  (a₁ : ℚ)
  (a₂ : ℚ)
  (h₁ : a₁ = 4)
  (h₂ : a₂ = -2) :
  let r := a₂ / a₁
  let a_k (k : ℕ) := a₁ * r^(k - 1)
  a_k 10 = -1/128 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_tenth_term_l3742_374208
