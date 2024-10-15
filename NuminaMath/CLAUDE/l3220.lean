import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_ac_l3220_322085

theorem sum_of_ac (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 40) 
  (h2 : b + d = 8) : 
  a + c = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ac_l3220_322085


namespace NUMINAMATH_CALUDE_perfect_square_pairs_l3220_322030

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

def valid_pair (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ a < 100 ∧ b < 100 ∧ 
  is_perfect_square (a + b) ∧ is_perfect_square (a * b)

def solution_set : List (ℕ × ℕ) :=
  [(2, 2), (5, 20), (8, 8), (10, 90), (18, 18), (20, 80), (9, 16), 
   (32, 32), (50, 50), (72, 72), (2, 98), (98, 98), (36, 64)]

theorem perfect_square_pairs :
  ∀ a b : ℕ, valid_pair a b ↔ (a, b) ∈ solution_set ∨ (b, a) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_perfect_square_pairs_l3220_322030


namespace NUMINAMATH_CALUDE_half_power_inequality_l3220_322053

theorem half_power_inequality (x y : ℝ) (h : x > y) : (1/2: ℝ)^x < (1/2 : ℝ)^y := by
  sorry

end NUMINAMATH_CALUDE_half_power_inequality_l3220_322053


namespace NUMINAMATH_CALUDE_problem_solution_l3220_322017

theorem problem_solution (x : ℝ) : x = 22.142857142857142 →
  2 * ((((x + 5) * 7) / 5) - 5) = 66 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3220_322017


namespace NUMINAMATH_CALUDE_marks_fish_count_l3220_322012

/-- Calculates the total number of young fish given the number of tanks, 
    pregnant fish per tank, and young per fish. -/
def total_young_fish (num_tanks : ℕ) (fish_per_tank : ℕ) (young_per_fish : ℕ) : ℕ :=
  num_tanks * fish_per_tank * young_per_fish

/-- Proves that given 3 tanks, 4 pregnant fish per tank, and 20 young per fish, 
    the total number of young fish is equal to 240. -/
theorem marks_fish_count : total_young_fish 3 4 20 = 240 := by
  sorry

end NUMINAMATH_CALUDE_marks_fish_count_l3220_322012


namespace NUMINAMATH_CALUDE_fraction_simplification_l3220_322086

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3220_322086


namespace NUMINAMATH_CALUDE_quadratic_root_implies_coefficient_l3220_322065

theorem quadratic_root_implies_coefficient (c : ℝ) : 
  ((-9 : ℝ)^2 + c*(-9) + 36 = 0) → c = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_coefficient_l3220_322065


namespace NUMINAMATH_CALUDE_sqrt_ab_plus_one_l3220_322071

theorem sqrt_ab_plus_one (a b : ℝ) (h : b = Real.sqrt (3 - a) + Real.sqrt (a - 3) + 8) : 
  (Real.sqrt (a * b + 1) = 5) ∨ (Real.sqrt (a * b + 1) = -5) := by
sorry

end NUMINAMATH_CALUDE_sqrt_ab_plus_one_l3220_322071


namespace NUMINAMATH_CALUDE_special_arithmetic_sequence_sum_l3220_322009

/-- An arithmetic sequence with special properties -/
structure ArithmeticSequence (m n : ℕ) :=
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h1 : S n = m)
  (h2 : S m = n)
  (h3 : m ≠ n)

/-- The sum of the first (m+n) terms of the special arithmetic sequence -/
def sumMPlusN (seq : ArithmeticSequence m n) : ℝ :=
  seq.S (m + n)

theorem special_arithmetic_sequence_sum (m n : ℕ) (seq : ArithmeticSequence m n) :
  sumMPlusN seq = -(m + n) := by
  sorry

#check special_arithmetic_sequence_sum

end NUMINAMATH_CALUDE_special_arithmetic_sequence_sum_l3220_322009


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3220_322058

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, a n > 0) 
  (h_geom : ∀ n, a (n + 1) = q * a n) (h_cond : 4 * a 2 = a 4) :
  let S_4 := (a 1) * (1 - q^4) / (1 - q)
  (S_4) / (a 2 + a 5) = 5/6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3220_322058


namespace NUMINAMATH_CALUDE_max_phoenix_number_l3220_322015

/-- Represents a four-digit number --/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  h1 : 2 ≤ a
  h2 : a ≤ b
  h3 : b < c
  h4 : c ≤ d
  h5 : d ≤ 9

/-- Defines a Phoenix number --/
def isPhoenixNumber (n : FourDigitNumber) : Prop :=
  n.b - n.a = 2 * (n.d - n.c)

/-- Defines the G function for a four-digit number --/
def G (n : FourDigitNumber) : Rat :=
  (49 * n.a * n.c - 2 * n.a + 2 * n.d + 23 * n.b - 6) / 24

/-- Theorem stating the maximum Phoenix number --/
theorem max_phoenix_number :
    ∃ (M : FourDigitNumber),
      isPhoenixNumber M ∧
      (G M).isInt ∧
      (∀ (N : FourDigitNumber),
        isPhoenixNumber N →
        (G N).isInt →
        1000 * N.a + 100 * N.b + 10 * N.c + N.d ≤ 1000 * M.a + 100 * M.b + 10 * M.c + M.d) ∧
      1000 * M.a + 100 * M.b + 10 * M.c + M.d = 6699 := by
  sorry

end NUMINAMATH_CALUDE_max_phoenix_number_l3220_322015


namespace NUMINAMATH_CALUDE_sequence_sum_l3220_322004

theorem sequence_sum (a : ℕ → ℚ) (x y : ℚ) :
  (∀ n, a (n + 1) = a n * (1 / 4)) →
  a 0 = 256 ∧ a 1 = x ∧ a 2 = y ∧ a 3 = 4 →
  x + y = 80 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l3220_322004


namespace NUMINAMATH_CALUDE_largest_square_size_for_rectangle_l3220_322098

theorem largest_square_size_for_rectangle (width height : ℕ) 
  (h_width : width = 63) (h_height : height = 42) :
  Nat.gcd width height = 21 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_size_for_rectangle_l3220_322098


namespace NUMINAMATH_CALUDE_roberto_outfits_l3220_322001

/-- The number of different outfits that can be created from a given number of trousers, shirts, and jackets. -/
def number_of_outfits (trousers shirts jackets : ℕ) : ℕ :=
  trousers * shirts * jackets

/-- Theorem stating that with 5 trousers, 6 shirts, and 4 jackets, the number of possible outfits is 120. -/
theorem roberto_outfits :
  number_of_outfits 5 6 4 = 120 := by
  sorry

end NUMINAMATH_CALUDE_roberto_outfits_l3220_322001


namespace NUMINAMATH_CALUDE_union_equals_interval_l3220_322038

-- Define sets A and B
def A : Set ℝ := {x | 0 < x ∧ x < 16}
def B : Set ℝ := {y | -4 < 4 * y ∧ 4 * y < 16}

-- Define the open interval (-1, 16)
def openInterval : Set ℝ := {z | -1 < z ∧ z < 16}

-- Theorem statement
theorem union_equals_interval : A ∪ B = openInterval := by
  sorry

end NUMINAMATH_CALUDE_union_equals_interval_l3220_322038


namespace NUMINAMATH_CALUDE_log_216_equals_3_log_6_l3220_322096

theorem log_216_equals_3_log_6 : Real.log 216 = 3 * Real.log 6 := by sorry

end NUMINAMATH_CALUDE_log_216_equals_3_log_6_l3220_322096


namespace NUMINAMATH_CALUDE_root_difference_implies_k_value_l3220_322091

theorem root_difference_implies_k_value (k : ℝ) : 
  (∃ r s : ℝ, 
    (r^2 + k*r + 12 = 0 ∧ s^2 + k*s + 12 = 0) ∧ 
    ((r+3)^2 - k*(r+3) + 12 = 0 ∧ (s+3)^2 - k*(s+3) + 12 = 0)) 
  → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_difference_implies_k_value_l3220_322091


namespace NUMINAMATH_CALUDE_bird_cage_problem_l3220_322010

theorem bird_cage_problem (initial_birds : ℕ) (final_birds : ℕ) : 
  initial_birds = 60 → final_birds = 8 → 
  ∃ F : ℚ, 
    (1/3 : ℚ) * (2/3 : ℚ) * initial_birds * (1 - F) = final_birds ∧ 
    F = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_bird_cage_problem_l3220_322010


namespace NUMINAMATH_CALUDE_production_days_calculation_l3220_322062

theorem production_days_calculation (n : ℕ) : 
  (∀ (P : ℝ), P / n = 60 → 
    (P + 90) / (n + 1) = 62) → 
  n = 14 := by
sorry

end NUMINAMATH_CALUDE_production_days_calculation_l3220_322062


namespace NUMINAMATH_CALUDE_difficult_math_problems_not_set_l3220_322090

-- Define the criteria for set elements
structure SetCriteria where
  definiteness : Bool
  distinctness : Bool
  unorderedness : Bool

-- Define a function to check if something can form a set
def canFormSet (criteria : SetCriteria) : Bool :=
  criteria.definiteness ∧ criteria.distinctness ∧ criteria.unorderedness

-- Define the characteristics of "All difficult math problems"
def difficultMathProblems : SetCriteria :=
  { definiteness := false,  -- Not definite
    distinctness := true,   -- Assumed distinct
    unorderedness := true } -- Assumed unordered

-- Theorem to prove
theorem difficult_math_problems_not_set : ¬(canFormSet difficultMathProblems) := by
  sorry

end NUMINAMATH_CALUDE_difficult_math_problems_not_set_l3220_322090


namespace NUMINAMATH_CALUDE_robert_nickel_vs_ashley_l3220_322037

/-- The number of chocolates eaten by Robert -/
def robert_chocolates : ℕ := 10

/-- The number of chocolates eaten by Nickel -/
def nickel_chocolates : ℕ := 5

/-- The number of chocolates eaten by Ashley -/
def ashley_chocolates : ℕ := 15

/-- Theorem stating that Robert and Nickel together ate the same number of chocolates as Ashley -/
theorem robert_nickel_vs_ashley : 
  robert_chocolates + nickel_chocolates - ashley_chocolates = 0 :=
by sorry

end NUMINAMATH_CALUDE_robert_nickel_vs_ashley_l3220_322037


namespace NUMINAMATH_CALUDE_phd_total_time_l3220_322088

def phd_timeline (acclimation_time : ℝ) (basics_time : ℝ) (research_factor : ℝ) (dissertation_factor : ℝ) : ℝ :=
  let research_time := basics_time * (1 + research_factor)
  let dissertation_time := acclimation_time * dissertation_factor
  acclimation_time + basics_time + research_time + dissertation_time

theorem phd_total_time :
  phd_timeline 1 2 0.75 0.5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_phd_total_time_l3220_322088


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_4_range_of_a_for_inequality_l3220_322076

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 + 2
def g (a x : ℝ) : ℝ := |x - a| - |x - 1|

-- Theorem for the first part of the problem
theorem solution_set_when_a_is_4 :
  {x : ℝ | f x > g 4 x} = {x : ℝ | x < -1 ∨ x > 1} := by sorry

-- Theorem for the second part of the problem
theorem range_of_a_for_inequality :
  {a : ℝ | ∀ x₁ x₂, f x₁ ≥ g a x₂} = Set.Icc (-1) 3 := by sorry

-- Note: Set.Icc represents a closed interval [a, b]

end NUMINAMATH_CALUDE_solution_set_when_a_is_4_range_of_a_for_inequality_l3220_322076


namespace NUMINAMATH_CALUDE_optimal_pill_count_l3220_322097

/-- Represents the vitamin content of a single pill -/
structure PillContent where
  vitaminA : ℕ
  vitaminB : ℕ
  vitaminC : ℕ

/-- Represents the recommended weekly servings for vitamins -/
structure WeeklyRecommendation where
  vitaminA : ℕ
  vitaminB : ℕ
  vitaminC : ℕ

/-- Checks if the given number of pills meets or exceeds all vitamin requirements -/
def meetsRequirements (pillContent : PillContent) (weeklyRecommendation : WeeklyRecommendation) (numPills : ℕ) : Prop :=
  numPills * pillContent.vitaminA ≥ weeklyRecommendation.vitaminA ∧
  numPills * pillContent.vitaminB ≥ weeklyRecommendation.vitaminB ∧
  numPills * pillContent.vitaminC ≥ weeklyRecommendation.vitaminC

theorem optimal_pill_count 
  (pillContent : PillContent)
  (weeklyRecommendation : WeeklyRecommendation)
  (h1 : pillContent.vitaminA = 50)
  (h2 : pillContent.vitaminB = 20)
  (h3 : pillContent.vitaminC = 10)
  (h4 : weeklyRecommendation.vitaminA = 1400)
  (h5 : weeklyRecommendation.vitaminB = 700)
  (h6 : weeklyRecommendation.vitaminC = 280) :
  meetsRequirements pillContent weeklyRecommendation 35 :=
by sorry

end NUMINAMATH_CALUDE_optimal_pill_count_l3220_322097


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l3220_322094

/-- Given a line segment from (2, 5) to (x, y) with length 10, prove that (x, y) = (8, 13) -/
theorem line_segment_endpoint (x y : ℝ) (h1 : x > 2) (h2 : y > 5) 
  (h3 : Real.sqrt ((x - 2)^2 + (y - 5)^2) = 10) : x = 8 ∧ y = 13 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l3220_322094


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3220_322095

theorem sum_of_fractions : (9 : ℚ) / 10 + (5 : ℚ) / 6 = (26 : ℚ) / 15 := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3220_322095


namespace NUMINAMATH_CALUDE_quiz_correct_percentage_l3220_322056

theorem quiz_correct_percentage (y : ℝ) (h : y > 0) :
  let total_questions := 7 * y
  let incorrect_questions := y / 3
  let correct_questions := total_questions - incorrect_questions
  (correct_questions / total_questions) = 20 / 21 := by
sorry

end NUMINAMATH_CALUDE_quiz_correct_percentage_l3220_322056


namespace NUMINAMATH_CALUDE_meaningful_fraction_l3220_322089

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l3220_322089


namespace NUMINAMATH_CALUDE_adult_ticket_price_l3220_322035

def student_price : ℚ := 5/2

theorem adult_ticket_price 
  (total_tickets : ℕ) 
  (total_revenue : ℚ) 
  (student_tickets : ℕ) 
  (h1 : total_tickets = 59) 
  (h2 : total_revenue = 445/2) 
  (h3 : student_tickets = 9) : 
  (total_revenue - student_price * student_tickets) / (total_tickets - student_tickets) = 4 := by
sorry

end NUMINAMATH_CALUDE_adult_ticket_price_l3220_322035


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l3220_322064

theorem solve_system_of_equations (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 20)
  (eq2 : 6 * p + 5 * q = 27) :
  p = 62 / 11 := by
sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l3220_322064


namespace NUMINAMATH_CALUDE_units_digit_of_n_squared_plus_two_to_n_l3220_322060

theorem units_digit_of_n_squared_plus_two_to_n (n : ℕ) : n = 3123^2 + 2^3123 → (n^2 + 2^n) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_n_squared_plus_two_to_n_l3220_322060


namespace NUMINAMATH_CALUDE_polygon_sides_count_l3220_322092

/-- A convex polygon with the sum of all angles except one equal to 2970° has 19 sides. -/
theorem polygon_sides_count (n : ℕ) (sum_angles : ℝ) (h1 : sum_angles = 2970) : n = 19 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l3220_322092


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l3220_322026

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 47 ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : distribute_balls 6 3 = 47 := by sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l3220_322026


namespace NUMINAMATH_CALUDE_discount_reduction_l3220_322087

/-- Proves that applying a 30% discount followed by a 20% discount
    results in a total reduction of 44% from the original price. -/
theorem discount_reduction (P : ℝ) (P_pos : P > 0) :
  let first_discount := 0.3
  let second_discount := 0.2
  let price_after_first := P * (1 - first_discount)
  let price_after_second := price_after_first * (1 - second_discount)
  let total_reduction := (P - price_after_second) / P
  total_reduction = 0.44 := by
  sorry

end NUMINAMATH_CALUDE_discount_reduction_l3220_322087


namespace NUMINAMATH_CALUDE_consecutive_numbers_problem_l3220_322084

theorem consecutive_numbers_problem (x y z w : ℤ) : 
  x > y → y > z →  -- x, y, and z are consecutive with x > y > z
  w > x →  -- w is greater than x
  5 * x = 3 * w →  -- ratio of x to w is 3:5
  2 * x + 3 * y + 3 * z = 5 * y + 11 →  -- given equation
  x - y = y - z →  -- consecutive numbers condition
  z = 3 := by
sorry

end NUMINAMATH_CALUDE_consecutive_numbers_problem_l3220_322084


namespace NUMINAMATH_CALUDE_expression_is_linear_binomial_when_k_is_3_l3220_322052

-- Define the algebraic expression
def algebraic_expression (k x : ℝ) : ℝ :=
  (-3*k*x^2 + x - 1) + (9*x^2 - 4*k*x + 3*k)

-- Define what it means for an expression to be a linear binomial
def is_linear_binomial (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b

-- Theorem statement
theorem expression_is_linear_binomial_when_k_is_3 :
  is_linear_binomial (algebraic_expression 3) :=
sorry

end NUMINAMATH_CALUDE_expression_is_linear_binomial_when_k_is_3_l3220_322052


namespace NUMINAMATH_CALUDE_product_even_even_is_even_product_odd_odd_is_odd_product_even_odd_is_even_product_odd_even_is_even_l3220_322023

-- Define even and odd integers
def IsEven (n : Int) : Prop := ∃ k : Int, n = 2 * k
def IsOdd (n : Int) : Prop := ∃ k : Int, n = 2 * k + 1

-- Theorem statements
theorem product_even_even_is_even (a b : Int) (ha : IsEven a) (hb : IsEven b) :
  IsEven (a * b) := by sorry

theorem product_odd_odd_is_odd (a b : Int) (ha : IsOdd a) (hb : IsOdd b) :
  IsOdd (a * b) := by sorry

theorem product_even_odd_is_even (a b : Int) (ha : IsEven a) (hb : IsOdd b) :
  IsEven (a * b) := by sorry

theorem product_odd_even_is_even (a b : Int) (ha : IsOdd a) (hb : IsEven b) :
  IsEven (a * b) := by sorry

end NUMINAMATH_CALUDE_product_even_even_is_even_product_odd_odd_is_odd_product_even_odd_is_even_product_odd_even_is_even_l3220_322023


namespace NUMINAMATH_CALUDE_c_share_calculation_l3220_322043

theorem c_share_calculation (total : ℝ) (a b c d : ℝ) : 
  total = 392 →
  a = b / 2 →
  b = c / 2 →
  d = total / 4 →
  a + b + c + d = total →
  c = 168 := by
sorry

end NUMINAMATH_CALUDE_c_share_calculation_l3220_322043


namespace NUMINAMATH_CALUDE_negation_of_neither_odd_l3220_322093

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem negation_of_neither_odd (a b : ℤ) : 
  ¬(¬(is_odd a) ∧ ¬(is_odd b)) ↔ (is_odd a ∨ is_odd b) :=
sorry

end NUMINAMATH_CALUDE_negation_of_neither_odd_l3220_322093


namespace NUMINAMATH_CALUDE_julie_school_year_hours_l3220_322046

/-- Calculates the number of hours Julie needs to work per week during the school year -/
def school_year_hours_per_week (summer_hours_per_week : ℕ) (summer_weeks : ℕ) (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) (school_year_earnings : ℕ) : ℕ :=
  let summer_total_hours := summer_hours_per_week * summer_weeks
  let hourly_wage := summer_earnings / summer_total_hours
  let school_year_total_hours := school_year_earnings / hourly_wage
  school_year_total_hours / school_year_weeks

/-- Theorem stating that Julie needs to work 20 hours per week during the school year -/
theorem julie_school_year_hours : 
  school_year_hours_per_week 60 8 6000 40 10000 = 20 := by
  sorry

end NUMINAMATH_CALUDE_julie_school_year_hours_l3220_322046


namespace NUMINAMATH_CALUDE_conference_games_count_l3220_322014

/-- The number of teams in the conference -/
def total_teams : ℕ := 16

/-- The number of divisions in the conference -/
def num_divisions : ℕ := 2

/-- The number of teams in each division -/
def teams_per_division : ℕ := 8

/-- The number of times each team plays others in its division -/
def intra_division_games : ℕ := 3

/-- The number of times each team plays teams in the other division -/
def inter_division_games : ℕ := 2

/-- Calculates the total number of games in a complete season for the conference -/
def total_games : ℕ :=
  let intra_division_total := num_divisions * (teams_per_division * (teams_per_division - 1) / 2) * intra_division_games
  let inter_division_total := (total_teams * teams_per_division / 2) * inter_division_games
  intra_division_total + inter_division_total

theorem conference_games_count : total_games = 296 := by
  sorry

end NUMINAMATH_CALUDE_conference_games_count_l3220_322014


namespace NUMINAMATH_CALUDE_frog_hop_ratio_l3220_322073

/-- The number of hops taken by each frog -/
structure FrogHops where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The conditions of the frog hopping problem -/
def frog_hop_conditions (h : FrogHops) : Prop :=
  ∃ (m : ℕ), 
    h.first = m * h.second ∧
    h.second = 2 * h.third ∧
    h.first + h.second + h.third = 99 ∧
    h.second = 18

/-- The theorem stating the ratio of hops between the first and second frog -/
theorem frog_hop_ratio (h : FrogHops) (hc : frog_hop_conditions h) :
  h.first / h.second = 4 := by
  sorry

#check frog_hop_ratio

end NUMINAMATH_CALUDE_frog_hop_ratio_l3220_322073


namespace NUMINAMATH_CALUDE_two_distinct_roots_characterization_l3220_322049

noncomputable def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ 
  (x^2 + x * |x| = 2 * (3 + a * x - 2 * a)) ∧
  (y^2 + y * |y| = 2 * (3 + a * y - 2 * a))

theorem two_distinct_roots_characterization (a : ℝ) :
  has_two_distinct_roots a ↔ 
  ((3/4 ≤ a ∧ a < 1) ∨ (a > 3)) ∨ (0 < a ∧ a < 3/4) :=
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_characterization_l3220_322049


namespace NUMINAMATH_CALUDE_paul_weekly_spending_l3220_322051

theorem paul_weekly_spending 
  (lawn_money : ℕ) 
  (weed_money : ℕ) 
  (weeks : ℕ) 
  (h1 : lawn_money = 68) 
  (h2 : weed_money = 13) 
  (h3 : weeks = 9) : 
  (lawn_money + weed_money) / weeks = 9 := by
sorry

end NUMINAMATH_CALUDE_paul_weekly_spending_l3220_322051


namespace NUMINAMATH_CALUDE_get_ready_time_l3220_322070

/-- The time it takes for Jack and his three toddlers to get ready -/
def total_time (jack_socks jack_shoes jack_jacket toddler_socks toddler_shoes toddler_laces num_toddlers : ℕ) : ℕ :=
  let jack_time := jack_socks + jack_shoes + jack_jacket
  let toddler_time := toddler_socks + toddler_shoes + toddler_laces
  jack_time + num_toddlers * toddler_time

/-- Theorem stating that it takes 33 minutes for Jack and his three toddlers to get ready -/
theorem get_ready_time : total_time 2 4 3 2 5 1 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_get_ready_time_l3220_322070


namespace NUMINAMATH_CALUDE_smallest_t_value_l3220_322048

theorem smallest_t_value : 
  let f (t : ℝ) := (16*t^2 - 36*t + 15)/(4*t - 3) + 4*t
  ∃ t_min : ℝ, t_min = (51 - Real.sqrt 2073) / 8 ∧
  (∀ t : ℝ, f t = 7*t + 6 → t ≥ t_min) ∧
  (f t_min = 7*t_min + 6) := by
sorry

end NUMINAMATH_CALUDE_smallest_t_value_l3220_322048


namespace NUMINAMATH_CALUDE_valera_coin_count_l3220_322008

/-- Represents the number of coins of each denomination -/
structure CoinCount where
  fifteenKopecks : Nat
  twentyKopecks : Nat

/-- Calculates the total value in kopecks given a CoinCount -/
def totalValue (coins : CoinCount) : Nat :=
  15 * coins.fifteenKopecks + 20 * coins.twentyKopecks

/-- Represents the conditions of the problem -/
structure ProblemConditions where
  initialCoins : CoinCount
  movieTicketCoins : Nat
  movieTicketValue : Nat
  lunchCoins : Nat

/-- The main theorem to prove -/
theorem valera_coin_count (conditions : ProblemConditions) : 
  (conditions.initialCoins.twentyKopecks > conditions.initialCoins.fifteenKopecks) →
  (conditions.movieTicketCoins = 2) →
  (conditions.lunchCoins = 3) →
  (totalValue conditions.initialCoins / 5 = conditions.movieTicketValue) →
  (((totalValue conditions.initialCoins - conditions.movieTicketValue) / 2) % (totalValue conditions.initialCoins - conditions.movieTicketValue) = 0) →
  (conditions.initialCoins = CoinCount.mk 2 6) :=
by sorry

#check valera_coin_count

end NUMINAMATH_CALUDE_valera_coin_count_l3220_322008


namespace NUMINAMATH_CALUDE_complex_sum_inverse_real_iff_unit_magnitude_l3220_322018

theorem complex_sum_inverse_real_iff_unit_magnitude 
  (a b : ℝ) (hb : b ≠ 0) : 
  let z : ℂ := Complex.mk a b
  (z + z⁻¹).im = 0 ↔ Complex.abs z = 1 := by sorry

end NUMINAMATH_CALUDE_complex_sum_inverse_real_iff_unit_magnitude_l3220_322018


namespace NUMINAMATH_CALUDE_f_is_even_l3220_322045

-- Define g as an even function
def g_even (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = g x

-- Define f in terms of g
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := |g (x^2)|

-- Theorem statement
theorem f_is_even (g : ℝ → ℝ) (h : g_even g) : ∀ x : ℝ, f g (-x) = f g x := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l3220_322045


namespace NUMINAMATH_CALUDE_smallest_multiple_l3220_322079

theorem smallest_multiple (x : ℕ+) : (∀ y : ℕ+, 720 * y.val % 1250 = 0 → x ≤ y) ∧ 720 * x.val % 1250 = 0 ↔ x = 125 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_l3220_322079


namespace NUMINAMATH_CALUDE_walking_distance_problem_l3220_322036

theorem walking_distance_problem (D : ℝ) : 
  D / 10 = (D + 20) / 20 → D = 20 := by
  sorry

end NUMINAMATH_CALUDE_walking_distance_problem_l3220_322036


namespace NUMINAMATH_CALUDE_sum_properties_l3220_322024

theorem sum_properties (x y : ℤ) (hx : ∃ m : ℤ, x = 5 * m) (hy : ∃ n : ℤ, y = 10 * n) :
  (∃ k : ℤ, x + y = 5 * k) ∧ x + y ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_properties_l3220_322024


namespace NUMINAMATH_CALUDE_subset_condition_A_eq_zero_four_six_A_proper_subsets_l3220_322047

def M : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def N (a : ℝ) : Set ℝ := {x | a*x = 12}

def A : Set ℝ := {a | N a ⊆ M}

theorem subset_condition (a : ℝ) : a ∈ A ↔ a = 0 ∨ a = 4 ∨ a = 6 := by sorry

theorem A_eq_zero_four_six : A = {0, 4, 6} := by sorry

def proper_subsets (S : Set ℝ) : Set (Set ℝ) :=
  {T | T ⊆ S ∧ T ≠ ∅ ∧ T ≠ S}

theorem A_proper_subsets :
  proper_subsets A = {{0}, {4}, {6}, {0, 4}, {0, 6}, {4, 6}} := by sorry

end NUMINAMATH_CALUDE_subset_condition_A_eq_zero_four_six_A_proper_subsets_l3220_322047


namespace NUMINAMATH_CALUDE_percentage_of_amount_twenty_five_percent_of_400_l3220_322066

theorem percentage_of_amount (amount : ℝ) (percentage : ℝ) :
  (percentage / 100) * amount = (percentage * amount) / 100 := by sorry

theorem twenty_five_percent_of_400 :
  (25 : ℝ) / 100 * 400 = 100 := by sorry

end NUMINAMATH_CALUDE_percentage_of_amount_twenty_five_percent_of_400_l3220_322066


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3220_322013

/-- Rationalize the denominator of (2 + √5) / (3 - √5) -/
theorem rationalize_denominator :
  ∃ (A B : ℚ) (C : ℕ), 
    (2 + Real.sqrt 5) / (3 - Real.sqrt 5) = A + B * Real.sqrt C ∧
    A = 11 / 4 ∧
    B = 5 / 4 ∧
    C = 5 ∧
    A * B * C = 275 / 16 := by
  sorry

#check rationalize_denominator

end NUMINAMATH_CALUDE_rationalize_denominator_l3220_322013


namespace NUMINAMATH_CALUDE_sets_A_and_B_solutions_l3220_322054

theorem sets_A_and_B_solutions (A B : Set ℕ) :
  (A ∩ B = {1, 2, 3}) ∧ (A ∪ B = {1, 2, 3, 4, 5}) →
  ((A = {1, 2, 3} ∧ B = {1, 2, 3, 4, 5}) ∨
   (A = {1, 2, 3, 4, 5} ∧ B = {1, 2, 3}) ∨
   (A = {1, 2, 3, 4} ∧ B = {1, 2, 3, 5}) ∨
   (A = {1, 2, 3, 5} ∧ B = {1, 2, 3, 4})) :=
by sorry

end NUMINAMATH_CALUDE_sets_A_and_B_solutions_l3220_322054


namespace NUMINAMATH_CALUDE_quadratic_inequality_all_reals_l3220_322068

theorem quadratic_inequality_all_reals (a b c : ℝ) :
  (∀ x : ℝ, -a/3 * x^2 + 2*b*x - c < 0) ↔ (a > 0 ∧ 4*b^2 - 4/3*a*c < 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_all_reals_l3220_322068


namespace NUMINAMATH_CALUDE_length_AB_l3220_322021

/-- Given an angle α with vertex A and a point B at distances a and b from the sides of the angle,
    the length of AB is either (√(a² + b² + 2ab*cos(α))) / sin(α) or (√(a² + b² - 2ab*cos(α))) / sin(α) -/
theorem length_AB (α a b : ℝ) (hα : 0 < α ∧ α < π) (ha : a > 0) (hb : b > 0) :
  ∃ (AB : ℝ), AB > 0 ∧
  (AB = (Real.sqrt (a^2 + b^2 + 2*a*b*Real.cos α)) / Real.sin α ∨
   AB = (Real.sqrt (a^2 + b^2 - 2*a*b*Real.cos α)) / Real.sin α) :=
by sorry

end NUMINAMATH_CALUDE_length_AB_l3220_322021


namespace NUMINAMATH_CALUDE_ellipse_minor_axis_length_l3220_322039

/-- Given an ellipse with equation x^2 + ky^2 = 2, foci on the x-axis, and focal distance √3, 
    its minor axis length is √5 -/
theorem ellipse_minor_axis_length (k : ℝ) : 
  (∀ x y : ℝ, x^2 + k*y^2 = 2) →  -- Equation of the ellipse
  (∃ c : ℝ, c^2 = 3 ∧ 
    ∀ x y : ℝ, x^2 + k*y^2 = 2 → 
      (x - c)^2 + y^2 = (x + c)^2 + y^2) →  -- Foci on x-axis with distance √3
  ∃ b : ℝ, b^2 = 5 ∧ 
    ∀ x y : ℝ, x^2 + k*y^2 = 2 → 
      y^2 ≤ b^2/4 :=  -- Minor axis length is √5
by sorry

end NUMINAMATH_CALUDE_ellipse_minor_axis_length_l3220_322039


namespace NUMINAMATH_CALUDE_log_x_inequality_l3220_322006

theorem log_x_inequality (x : ℝ) (h : 1 < x ∧ x < 2) :
  ((Real.log x) / x)^2 < (Real.log x) / x ∧ (Real.log x) / x < (Real.log (x^2)) / (x^2) := by
  sorry

end NUMINAMATH_CALUDE_log_x_inequality_l3220_322006


namespace NUMINAMATH_CALUDE_complex_in_second_quadrant_l3220_322042

/-- The complex number z = (2+3i)/(1-i) lies in the second quadrant of the complex plane. -/
theorem complex_in_second_quadrant : 
  let z : ℂ := (2 + 3*I) / (1 - I)
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_in_second_quadrant_l3220_322042


namespace NUMINAMATH_CALUDE_fake_to_total_purse_ratio_l3220_322029

/-- Given a collection of purses and handbags, prove that the ratio of fake purses to total purses is 1:2 -/
theorem fake_to_total_purse_ratio (total_purses total_handbags : ℕ) 
  (authentic_items : ℕ) (h1 : total_purses = 26) (h2 : total_handbags = 24) 
  (h3 : authentic_items = 31) : 
  (total_purses - authentic_items + total_handbags - total_handbags / 4) / total_purses = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fake_to_total_purse_ratio_l3220_322029


namespace NUMINAMATH_CALUDE_rectangle_area_15_20_l3220_322007

/-- The area of a rectangular field with given length and width -/
def rectangle_area (length width : ℝ) : ℝ := length * width

/-- Theorem: The area of a rectangular field with length 15 meters and width 20 meters is 300 square meters -/
theorem rectangle_area_15_20 :
  rectangle_area 15 20 = 300 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_15_20_l3220_322007


namespace NUMINAMATH_CALUDE_salary_difference_l3220_322032

theorem salary_difference (ram_salary raja_salary : ℝ) 
  (h : raja_salary = 0.8 * ram_salary) : 
  (ram_salary - raja_salary) / raja_salary = 0.25 := by
sorry

end NUMINAMATH_CALUDE_salary_difference_l3220_322032


namespace NUMINAMATH_CALUDE_prime_sum_problem_l3220_322044

theorem prime_sum_problem (p q r : ℕ) : 
  Prime p → Prime q → Prime r →
  p * q + q * r + r * p = 191 →
  p + q = r - 1 →
  p + q + r = 25 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_problem_l3220_322044


namespace NUMINAMATH_CALUDE_sara_remaining_pears_l3220_322050

def remaining_pears (initial : ℕ) (given_to_dan : ℕ) (given_to_monica : ℕ) (given_to_jenny : ℕ) : ℕ :=
  initial - given_to_dan - given_to_monica - given_to_jenny

theorem sara_remaining_pears :
  remaining_pears 35 28 4 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sara_remaining_pears_l3220_322050


namespace NUMINAMATH_CALUDE_product_evaluation_l3220_322019

theorem product_evaluation :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) = 5^32 - 4^32 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l3220_322019


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3220_322005

/-- The number of games played in a chess tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 9 players, where each player plays every
    other player exactly once, the total number of games played is 36. -/
theorem chess_tournament_games :
  num_games 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3220_322005


namespace NUMINAMATH_CALUDE_total_distance_flown_l3220_322057

/-- The speed of an eagle in miles per hour -/
def eagle_speed : ℝ := 15

/-- The speed of a falcon in miles per hour -/
def falcon_speed : ℝ := 46

/-- The speed of a pelican in miles per hour -/
def pelican_speed : ℝ := 33

/-- The speed of a hummingbird in miles per hour -/
def hummingbird_speed : ℝ := 30

/-- The time the birds flew in hours -/
def flight_time : ℝ := 2

/-- Theorem stating that the total distance flown by all birds in 2 hours is 248 miles -/
theorem total_distance_flown : 
  eagle_speed * flight_time + 
  falcon_speed * flight_time + 
  pelican_speed * flight_time + 
  hummingbird_speed * flight_time = 248 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_flown_l3220_322057


namespace NUMINAMATH_CALUDE_square_sum_theorem_l3220_322077

theorem square_sum_theorem (x y z : ℝ) 
  (eq1 : x^2 + 3*y = 8)
  (eq2 : y^2 + 5*z = -9)
  (eq3 : z^2 + 7*x = -16) :
  x^2 + y^2 + z^2 = 20.75 := by
sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l3220_322077


namespace NUMINAMATH_CALUDE_vector_collinearity_l3220_322099

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def are_collinear (v w : V) : Prop := ∃ (c : ℝ), v = c • w ∨ w = c • v

theorem vector_collinearity 
  (e₁ e₂ a b : V) 
  (h_nonzero_e₁ : e₁ ≠ 0)
  (h_nonzero_e₂ : e₂ ≠ 0)
  (h_a : a = 2 • e₁ - e₂)
  (h_b : ∃ k : ℝ, b = k • e₁ + e₂) :
  (¬ are_collinear e₁ e₂ ∧ are_collinear a b → ∃ k : ℝ, b = k • e₁ + e₂ ∧ k = -2) ∧
  (∀ k : ℝ, are_collinear e₁ e₂ → are_collinear a b) := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l3220_322099


namespace NUMINAMATH_CALUDE_rachel_budget_value_l3220_322063

/-- The cost of Sara's shoes -/
def sara_shoes : ℕ := 50

/-- The cost of Sara's dress -/
def sara_dress : ℕ := 200

/-- The cost of Tina's shoes -/
def tina_shoes : ℕ := 70

/-- The cost of Tina's dress -/
def tina_dress : ℕ := 150

/-- Rachel's budget is twice the sum of Sara's and Tina's expenses -/
def rachel_budget : ℕ := 2 * (sara_shoes + sara_dress + tina_shoes + tina_dress)

theorem rachel_budget_value : rachel_budget = 940 := by
  sorry

end NUMINAMATH_CALUDE_rachel_budget_value_l3220_322063


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3220_322020

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ 4) :
  6 * x / ((x - 4) * (x - 3)^2) = 
    24 / (x - 4) + (-162/7) / (x - 3) + (-18) / (x - 3)^2 :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3220_322020


namespace NUMINAMATH_CALUDE_tank_filling_time_l3220_322055

theorem tank_filling_time (pipe1_time pipe2_time : ℝ) (leak_fraction : ℝ) : 
  pipe1_time = 20 ∧ 
  pipe2_time = 30 ∧ 
  leak_fraction = 1/3 → 
  (1 / (1/pipe1_time + 1/pipe2_time)) * (1 / (1 - leak_fraction)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_tank_filling_time_l3220_322055


namespace NUMINAMATH_CALUDE_carols_spending_contradiction_l3220_322078

theorem carols_spending_contradiction (savings : ℝ) (tv_fraction : ℝ) 
  (h1 : savings > 0)
  (h2 : 0 < tv_fraction)
  (h3 : tv_fraction < 1/4)
  (h4 : 1/4 * savings + tv_fraction * savings = 0.25 * savings) : False :=
sorry

end NUMINAMATH_CALUDE_carols_spending_contradiction_l3220_322078


namespace NUMINAMATH_CALUDE_symmetric_points_implies_power_of_negative_two_l3220_322027

/-- If points M(3a+b, 8) and N(9, 2a+3b) are symmetric about the x-axis, then (-2)^(2a+b) = 16 -/
theorem symmetric_points_implies_power_of_negative_two (a b : ℝ) : 
  (3 * a + b = 9 ∧ 2 * a + 3 * b = -8) → (-2 : ℝ) ^ (2 * a + b) = 16 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_implies_power_of_negative_two_l3220_322027


namespace NUMINAMATH_CALUDE_roots_product_minus_one_l3220_322011

theorem roots_product_minus_one (d e : ℝ) : 
  (3 * d^2 + 4 * d - 7 = 0) → 
  (3 * e^2 + 4 * e - 7 = 0) → 
  (d - 1) * (e - 1) = 1 := by
sorry

end NUMINAMATH_CALUDE_roots_product_minus_one_l3220_322011


namespace NUMINAMATH_CALUDE_negation_of_forall_exp_gt_x_l3220_322067

theorem negation_of_forall_exp_gt_x :
  ¬(∀ x : ℝ, Real.exp x > x) ↔ ∃ x : ℝ, Real.exp x ≤ x := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_exp_gt_x_l3220_322067


namespace NUMINAMATH_CALUDE_fahrenheit_to_celsius_l3220_322041

theorem fahrenheit_to_celsius (C F : ℝ) : C = (5/9) * (F - 32) → C = 40 → F = 104 := by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_to_celsius_l3220_322041


namespace NUMINAMATH_CALUDE_boys_in_biology_class_l3220_322075

/-- Represents the number of students in a class -/
structure ClassCount where
  boys : ℕ
  girls : ℕ

/-- Represents the counts for all three classes -/
structure SchoolCounts where
  biology : ClassCount
  physics : ClassCount
  chemistry : ClassCount

/-- The conditions of the problem -/
def school_conditions (counts : SchoolCounts) : Prop :=
  -- Biology class condition
  counts.biology.girls = 3 * counts.biology.boys ∧
  -- Physics class condition
  2 * counts.physics.boys = 3 * counts.physics.girls ∧
  -- Chemistry class condition
  counts.chemistry.boys = counts.chemistry.girls ∧
  counts.chemistry.boys + counts.chemistry.girls = 270 ∧
  -- Relation between Biology and Physics classes
  counts.biology.boys + counts.biology.girls = 
    (counts.physics.boys + counts.physics.girls) / 2 ∧
  -- Total number of students
  counts.biology.boys + counts.biology.girls +
  counts.physics.boys + counts.physics.girls +
  counts.chemistry.boys + counts.chemistry.girls = 1000

/-- The theorem to be proved -/
theorem boys_in_biology_class (counts : SchoolCounts) :
  school_conditions counts → counts.biology.boys = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_boys_in_biology_class_l3220_322075


namespace NUMINAMATH_CALUDE_expansion_coefficients_properties_l3220_322083

theorem expansion_coefficients_properties :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ),
  (∀ x : ℚ, (2*x + 1)^6 = a₀*x^6 + a₁*x^5 + a₂*x^4 + a₃*x^3 + a₄*x^2 + a₅*x + a₆) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 729) ∧
  (a₁ + a₃ + a₅ = 364) ∧
  (a₂ + a₄ = 300) := by
sorry

end NUMINAMATH_CALUDE_expansion_coefficients_properties_l3220_322083


namespace NUMINAMATH_CALUDE_money_distribution_l3220_322002

theorem money_distribution (a b c total : ℕ) : 
  (a + b + c = total) →  -- total is the sum of all shares
  (2 * b = 3 * a) →      -- ratio between a and b is 2:3
  (3 * c = 4 * b) →      -- ratio between b and c is 3:4
  (b = 1800) →           -- b's share is $1800
  (total = 5400) :=      -- prove that total is $5400
by
  sorry

#check money_distribution

end NUMINAMATH_CALUDE_money_distribution_l3220_322002


namespace NUMINAMATH_CALUDE_anns_age_l3220_322061

theorem anns_age (a b : ℕ) : 
  a + b = 72 → 
  b = (a / 3 : ℚ) + 2 * (a - b) → 
  a = 46 :=
by sorry

end NUMINAMATH_CALUDE_anns_age_l3220_322061


namespace NUMINAMATH_CALUDE_total_cats_l3220_322074

/-- The number of cats in a pet store -/
def num_cats (white black gray : ℕ) : ℕ := white + black + gray

/-- Theorem stating that the total number of cats is 15 -/
theorem total_cats : num_cats 2 10 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_cats_l3220_322074


namespace NUMINAMATH_CALUDE_special_set_average_l3220_322022

/-- A finite set of positive integers satisfying specific conditions -/
def SpecialSet (T : Finset ℕ) : Prop :=
  ∃ (m : ℕ), m > 1 ∧ 
  (∃ (b₁ bₘ : ℕ), b₁ ∈ T ∧ bₘ ∈ T ∧
    (∀ x ∈ T, b₁ ≤ x ∧ x ≤ bₘ) ∧
    bₘ = b₁ + 50 ∧
    (T.sum id - bₘ) / (m - 1) = 45 ∧
    (T.sum id - b₁ - bₘ) / (m - 2) = 50 ∧
    (T.sum id - b₁) / (m - 1) = 55)

/-- The average of all integers in a SpecialSet is 50 -/
theorem special_set_average (T : Finset ℕ) (h : SpecialSet T) :
  (T.sum id) / T.card = 50 := by
  sorry

end NUMINAMATH_CALUDE_special_set_average_l3220_322022


namespace NUMINAMATH_CALUDE_f_properties_l3220_322025

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 2 * (Real.sin x)^2

theorem f_properties :
  let T := Real.pi
  let monotonic_interval (k : ℤ) := Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6)
  ∀ x : ℝ,
  (∀ y : ℝ, f (x + T) = f x) ∧ 
  (∀ k : ℤ, StrictMono (f ∘ (fun t => t + k * Real.pi - Real.pi / 3) : monotonic_interval k → ℝ)) ∧
  (x ∈ Set.Icc 0 (Real.pi / 4) → 
    f x ∈ Set.Icc 0 1 ∧
    (f x = 0 ↔ x = 0) ∧
    (f x = 1 ↔ x = Real.pi / 6)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3220_322025


namespace NUMINAMATH_CALUDE_find_number_l3220_322003

theorem find_number : ∃ x : ℚ, x = 15 ∧ (4/5) * x + 20 = (80/100) * 40 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3220_322003


namespace NUMINAMATH_CALUDE_shorter_leg_is_15_l3220_322000

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  a : ℕ  -- shorter leg
  b : ℕ  -- longer leg
  c : ℕ  -- hypotenuse
  right_angle : a ^ 2 + b ^ 2 = c ^ 2
  a_shorter : a ≤ b

/-- The length of the shorter leg in a right triangle with hypotenuse 25 -/
def shorter_leg_length : ℕ := 15

/-- Theorem stating that in a right triangle with integer side lengths and hypotenuse 25, 
    the shorter leg has length 15 -/
theorem shorter_leg_is_15 (t : RightTriangle) (hyp_25 : t.c = 25) : 
  t.a = shorter_leg_length := by
  sorry


end NUMINAMATH_CALUDE_shorter_leg_is_15_l3220_322000


namespace NUMINAMATH_CALUDE_cubic_polynomial_roots_l3220_322016

theorem cubic_polynomial_roots :
  let p (x : ℝ) := x^3 - 2*x^2 - 5*x + 6
  ∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = -2 ∨ x = 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_roots_l3220_322016


namespace NUMINAMATH_CALUDE_otimes_sqrt_two_otimes_sum_zero_l3220_322080

-- Define the ⊗ operation
def otimes (a b : ℝ) : ℝ := a * (1 - b)

-- Theorem 1
theorem otimes_sqrt_two : otimes (1 + Real.sqrt 2) (Real.sqrt 2) = -1 := by sorry

-- Theorem 2
theorem otimes_sum_zero (a b : ℝ) : a + b = 0 → otimes a a + otimes b b = 2 * a * b := by sorry

end NUMINAMATH_CALUDE_otimes_sqrt_two_otimes_sum_zero_l3220_322080


namespace NUMINAMATH_CALUDE_teresas_siblings_teresa_has_three_siblings_l3220_322031

/-- Given Teresa's pencil collection and distribution rules, calculate the number of her siblings --/
theorem teresas_siblings (colored_pencils : ℕ) (black_pencils : ℕ) (kept_pencils : ℕ) (pencils_per_sibling : ℕ) : ℕ :=
  let total_pencils := colored_pencils + black_pencils
  let shared_pencils := total_pencils - kept_pencils
  shared_pencils / pencils_per_sibling

/-- Prove that Teresa has 3 siblings given the problem conditions --/
theorem teresa_has_three_siblings :
  teresas_siblings 14 35 10 13 = 3 := by
  sorry

end NUMINAMATH_CALUDE_teresas_siblings_teresa_has_three_siblings_l3220_322031


namespace NUMINAMATH_CALUDE_michael_rubber_bands_l3220_322059

/-- The number of rubber bands in Michael's pack -/
def total_rubber_bands (small_ball_bands small_balls large_ball_bands large_balls : ℕ) : ℕ :=
  small_ball_bands * small_balls + large_ball_bands * large_balls

/-- Proof that Michael's pack contained 5000 rubber bands -/
theorem michael_rubber_bands :
  let small_ball_bands := 50
  let large_ball_bands := 300
  let small_balls := 22
  let large_balls := 13
  total_rubber_bands small_ball_bands small_balls large_ball_bands large_balls = 5000 := by
sorry

end NUMINAMATH_CALUDE_michael_rubber_bands_l3220_322059


namespace NUMINAMATH_CALUDE_bus_journey_distance_l3220_322081

/-- Proves that a bus journey with given conditions results in a total distance of 250 km -/
theorem bus_journey_distance (speed1 speed2 distance1 total_time : ℝ) 
  (h1 : speed1 = 40)
  (h2 : speed2 = 60)
  (h3 : distance1 = 100)
  (h4 : total_time = 5)
  (h5 : distance1 / speed1 + (total_distance - distance1) / speed2 = total_time) :
  total_distance = 250 := by
  sorry

#check bus_journey_distance

end NUMINAMATH_CALUDE_bus_journey_distance_l3220_322081


namespace NUMINAMATH_CALUDE_added_number_forms_geometric_sequence_l3220_322072

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  first_term : a 1 = 2
  third_term : a 3 = 6
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- The property that adding x to certain terms forms a geometric sequence -/
def FormsGeometricSequence (seq : ArithmeticSequence) (x : ℝ) : Prop :=
  (seq.a 4 + x)^2 = (seq.a 1 + x) * (seq.a 5 + x)

/-- The main theorem -/
theorem added_number_forms_geometric_sequence (seq : ArithmeticSequence) :
  ∃ x : ℝ, FormsGeometricSequence seq x ∧ x = -11 := by
  sorry

end NUMINAMATH_CALUDE_added_number_forms_geometric_sequence_l3220_322072


namespace NUMINAMATH_CALUDE_valid_representation_count_l3220_322040

/-- Represents a natural number in binary form -/
def BinaryRepresentation := List Bool

/-- Checks if a binary representation has three consecutive identical digits -/
def hasThreeConsecutiveIdenticalDigits (bin : BinaryRepresentation) : Bool :=
  sorry

/-- Counts the number of valid binary representations between 4 and 1023 -/
def countValidRepresentations : Nat :=
  sorry

/-- The main theorem stating the count of valid representations is 228 -/
theorem valid_representation_count :
  countValidRepresentations = 228 := by
  sorry

end NUMINAMATH_CALUDE_valid_representation_count_l3220_322040


namespace NUMINAMATH_CALUDE_rectangle_area_change_l3220_322034

theorem rectangle_area_change (L B : ℝ) (h1 : L > 0) (h2 : B > 0) : 
  let original_area := L * B
  let new_length := L / 2
  let new_breadth := 3 * B
  let new_area := new_length * new_breadth
  ((new_area - original_area) / original_area) * 100 = 50 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l3220_322034


namespace NUMINAMATH_CALUDE_exam_average_problem_l3220_322028

theorem exam_average_problem (n₁ n₂ : ℕ) (avg₁ avg_total : ℚ) (avg₂ : ℚ) :
  n₁ = 15 →
  n₂ = 10 →
  avg₁ = 70 / 100 →
  avg_total = 78 / 100 →
  (n₁ + n₂ : ℚ) * avg_total = n₁ * avg₁ + n₂ * avg₂ →
  avg₂ = 90 / 100 := by
sorry

end NUMINAMATH_CALUDE_exam_average_problem_l3220_322028


namespace NUMINAMATH_CALUDE_yoojeong_line_length_l3220_322069

/-- Conversion factor from centimeters to millimeters -/
def cm_to_mm : ℕ := 10

/-- The length of the reference line in centimeters -/
def reference_length_cm : ℕ := 31

/-- The difference in millimeters between the reference line and Yoojeong's line -/
def difference_mm : ℕ := 3

/-- The length of Yoojeong's line in millimeters -/
def yoojeong_line_mm : ℕ := reference_length_cm * cm_to_mm - difference_mm

theorem yoojeong_line_length : yoojeong_line_mm = 307 :=
by sorry

end NUMINAMATH_CALUDE_yoojeong_line_length_l3220_322069


namespace NUMINAMATH_CALUDE_garrison_provisions_l3220_322082

theorem garrison_provisions (initial_men : ℕ) (reinforcement : ℕ) (days_before_reinforcement : ℕ) (days_after_reinforcement : ℕ) 
  (h1 : initial_men = 2000)
  (h2 : reinforcement = 2000)
  (h3 : days_before_reinforcement = 20)
  (h4 : days_after_reinforcement = 10) :
  ∃ (initial_duration : ℕ), 
    initial_men * (initial_duration - days_before_reinforcement) = 
    (initial_men + reinforcement) * days_after_reinforcement ∧
    initial_duration = 40 := by
sorry

end NUMINAMATH_CALUDE_garrison_provisions_l3220_322082


namespace NUMINAMATH_CALUDE_selection_and_assignment_problem_l3220_322033

def number_of_ways (male_students female_students total_selected num_tasks : ℕ) : ℕ :=
  sorry

theorem selection_and_assignment_problem :
  let male_students := 4
  let female_students := 3
  let total_selected := 4
  let num_tasks := 3
  number_of_ways male_students female_students total_selected num_tasks = 792 := by
  sorry

end NUMINAMATH_CALUDE_selection_and_assignment_problem_l3220_322033
