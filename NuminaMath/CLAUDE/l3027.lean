import Mathlib

namespace NUMINAMATH_CALUDE_prob_same_color_specific_l3027_302775

/-- The probability of drawing 4 marbles of the same color from an urn -/
def prob_same_color (red white blue : ℕ) : ℚ :=
  let total := red + white + blue
  let prob_red := (red.descFactorial 4 : ℚ) / (total.descFactorial 4 : ℚ)
  let prob_white := (white.descFactorial 4 : ℚ) / (total.descFactorial 4 : ℚ)
  let prob_blue := (blue.descFactorial 4 : ℚ) / (total.descFactorial 4 : ℚ)
  prob_red + prob_white + prob_blue

/-- Theorem: The probability of drawing 4 marbles of the same color from an urn
    containing 5 red, 6 white, and 7 blue marbles is 55/3060 -/
theorem prob_same_color_specific : prob_same_color 5 6 7 = 55 / 3060 := by
  sorry


end NUMINAMATH_CALUDE_prob_same_color_specific_l3027_302775


namespace NUMINAMATH_CALUDE_coyote_prints_time_l3027_302782

/-- The time elapsed since the coyote left the prints -/
def time_elapsed : ℝ := 2

/-- The speed of the coyote in miles per hour -/
def coyote_speed : ℝ := 15

/-- The speed of Darrel in miles per hour -/
def darrel_speed : ℝ := 30

/-- The time it takes Darrel to catch up to the coyote in hours -/
def catch_up_time : ℝ := 1

theorem coyote_prints_time :
  time_elapsed * coyote_speed = darrel_speed * catch_up_time :=
sorry

end NUMINAMATH_CALUDE_coyote_prints_time_l3027_302782


namespace NUMINAMATH_CALUDE_iris_count_after_addition_l3027_302752

/-- Calculates the number of irises needed to maintain a ratio of 3:7 with roses -/
def calculate_irises (initial_roses : ℕ) (added_roses : ℕ) : ℕ :=
  let total_roses := initial_roses + added_roses
  let irises := (3 * total_roses) / 7
  irises

theorem iris_count_after_addition 
  (initial_roses : ℕ) 
  (added_roses : ℕ) 
  (h1 : initial_roses = 35) 
  (h2 : added_roses = 25) : 
  calculate_irises initial_roses added_roses = 25 := by
sorry

#eval calculate_irises 35 25

end NUMINAMATH_CALUDE_iris_count_after_addition_l3027_302752


namespace NUMINAMATH_CALUDE_no_half_rectangle_exists_l3027_302768

theorem no_half_rectangle_exists (a b : ℝ) (h : 0 < a ∧ a < b) :
  ¬ ∃ (x y : ℝ), 
    x < a / 2 ∧ 
    y < a / 2 ∧ 
    2 * (x + y) = a + b ∧ 
    x * y = a * b / 2 :=
by sorry

end NUMINAMATH_CALUDE_no_half_rectangle_exists_l3027_302768


namespace NUMINAMATH_CALUDE_product_of_square_roots_l3027_302759

theorem product_of_square_roots (q : ℝ) (hq : q ≥ 0) :
  Real.sqrt (15 * q) * Real.sqrt (10 * q^3) * Real.sqrt (8 * q^5) = 20 * q^4 * Real.sqrt (3 * q) :=
by sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l3027_302759


namespace NUMINAMATH_CALUDE_skew_lines_angle_equals_dihedral_angle_l3027_302749

-- Define the dihedral angle
def dihedral_angle (α l β : Line3) : ℝ := sorry

-- Define perpendicularity between a line and a plane
def perpendicular (m : Line3) (α : Plane3) : Prop := sorry

-- Define the angle between two skew lines
def skew_line_angle (m n : Line3) : ℝ := sorry

-- Theorem statement
theorem skew_lines_angle_equals_dihedral_angle 
  (α l β : Line3) (m n : Line3) :
  dihedral_angle α l β = 60 →
  perpendicular m α →
  perpendicular n β →
  skew_line_angle m n = 60 := by sorry

end NUMINAMATH_CALUDE_skew_lines_angle_equals_dihedral_angle_l3027_302749


namespace NUMINAMATH_CALUDE_exists_n_good_not_n_plus_one_good_l3027_302785

/-- Sum of digits of a natural number -/
def S (k : ℕ) : ℕ := sorry

/-- Function f(n) = n - S(n) -/
def f (n : ℕ) : ℕ := n - S n

/-- Iterated application of f, k times -/
def f_iter (k : ℕ) (n : ℕ) : ℕ := 
  match k with
  | 0 => n
  | k + 1 => f (f_iter k n)

/-- A number a is n-good if there exists a sequence a₀, ..., aₙ where aₙ = a and aᵢ₊₁ = f(aᵢ) -/
def is_n_good (n : ℕ) (a : ℕ) : Prop :=
  ∃ (a₀ : ℕ), f_iter n a₀ = a

/-- Main theorem: For all n, there exists an a that is n-good but not (n+1)-good -/
theorem exists_n_good_not_n_plus_one_good :
  ∀ n : ℕ, ∃ a : ℕ, is_n_good n a ∧ ¬is_n_good (n + 1) a := by sorry

end NUMINAMATH_CALUDE_exists_n_good_not_n_plus_one_good_l3027_302785


namespace NUMINAMATH_CALUDE_science_problems_count_l3027_302799

theorem science_problems_count (math_problems finished_problems left_problems : ℕ) 
  (h1 : math_problems = 18)
  (h2 : finished_problems = 24)
  (h3 : left_problems = 5) :
  let total_problems := finished_problems + left_problems
  let science_problems := total_problems - math_problems
  science_problems = 11 := by
sorry

end NUMINAMATH_CALUDE_science_problems_count_l3027_302799


namespace NUMINAMATH_CALUDE_min_distance_point_to_y_axis_l3027_302745

/-- Given point A (-3, -2) and point B on the y-axis, the distance between A and B is minimized when B has coordinates (0, -2) -/
theorem min_distance_point_to_y_axis (A B : ℝ × ℝ) :
  A = (-3, -2) →
  B.1 = 0 →
  (∀ C : ℝ × ℝ, C.1 = 0 → dist A B ≤ dist A C) →
  B = (0, -2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_point_to_y_axis_l3027_302745


namespace NUMINAMATH_CALUDE_a_share_is_one_third_l3027_302758

/-- Represents the investment and profit distribution scenario -/
structure InvestmentScenario where
  initial_investment : ℝ
  annual_gain : ℝ
  months_in_year : ℕ

/-- Calculates the effective investment value for a partner -/
def effective_investment (scenario : InvestmentScenario) 
  (investment_multiplier : ℝ) (investment_duration : ℕ) : ℝ :=
  scenario.initial_investment * investment_multiplier * investment_duration

/-- Theorem stating that A's share of the gain is one-third of the total gain -/
theorem a_share_is_one_third (scenario : InvestmentScenario) 
  (h1 : scenario.months_in_year = 12)
  (h2 : scenario.annual_gain > 0) : 
  let a_investment := effective_investment scenario 1 scenario.months_in_year
  let b_investment := effective_investment scenario 2 6
  let c_investment := effective_investment scenario 3 4
  let total_effective_investment := a_investment + b_investment + c_investment
  scenario.annual_gain / 3 = (a_investment / total_effective_investment) * scenario.annual_gain := by
  sorry

#check a_share_is_one_third

end NUMINAMATH_CALUDE_a_share_is_one_third_l3027_302758


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3027_302704

-- Define the inequality function
def f (x : ℝ) : Prop := (3 * x + 5) / (x - 1) > x

-- Define the solution set
def solution_set (x : ℝ) : Prop := x < -1 ∨ (1 < x ∧ x < 5)

-- Theorem statement
theorem inequality_solution_set : 
  ∀ x : ℝ, x ≠ 1 → (f x ↔ solution_set x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3027_302704


namespace NUMINAMATH_CALUDE_virginia_may_rainfall_l3027_302798

/-- Calculates the rainfall in May given the rainfall amounts for other months and the average -/
def rainfall_in_may (march april june july average : ℝ) : ℝ :=
  5 * average - (march + april + june + july)

/-- Theorem stating that the rainfall in May is 3.95 inches given the specified conditions -/
theorem virginia_may_rainfall :
  let march : ℝ := 3.79
  let april : ℝ := 4.5
  let june : ℝ := 3.09
  let july : ℝ := 4.67
  let average : ℝ := 4
  rainfall_in_may march april june july average = 3.95 := by
  sorry


end NUMINAMATH_CALUDE_virginia_may_rainfall_l3027_302798


namespace NUMINAMATH_CALUDE_division_theorem_l3027_302742

theorem division_theorem (A B : ℕ) : 23 = 6 * A + B ∧ B < 6 → A = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_theorem_l3027_302742


namespace NUMINAMATH_CALUDE_cube_volume_problem_l3027_302711

theorem cube_volume_problem (a : ℝ) : 
  (a > 0) →  -- Ensure a is positive for a valid cube
  (a^3 - ((a + 1)^2 * (a - 2)) = 10) → 
  (a^3 = 216) := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l3027_302711


namespace NUMINAMATH_CALUDE_m_n_properties_l3027_302769

theorem m_n_properties (m n : ℤ) (hm : |m| = 1) (hn : |n| = 4) :
  (∃ k : ℤ, mn < 0 → m + n = k ∧ (k = 3 ∨ k = -3)) ∧
  (∀ x y : ℤ, |x| = 1 → |y| = 4 → x - y ≤ 5) ∧
  (∃ a b : ℤ, |a| = 1 ∧ |b| = 4 ∧ a - b = 5) :=
by sorry

end NUMINAMATH_CALUDE_m_n_properties_l3027_302769


namespace NUMINAMATH_CALUDE_grassy_width_is_60_l3027_302701

/-- Represents a rectangular plot with a gravel path around it. -/
structure RectangularPlot where
  length : ℝ
  totalWidth : ℝ
  pathWidth : ℝ

/-- Calculates the width of the grassy area in a rectangular plot. -/
def grassyWidth (plot : RectangularPlot) : ℝ :=
  plot.totalWidth - 2 * plot.pathWidth

/-- Theorem stating that for a given rectangular plot with specified dimensions,
    the width of the grassy area is 60 meters. -/
theorem grassy_width_is_60 (plot : RectangularPlot)
    (h1 : plot.length = 110)
    (h2 : plot.totalWidth = 65)
    (h3 : plot.pathWidth = 2.5) :
  grassyWidth plot = 60 := by
  sorry

end NUMINAMATH_CALUDE_grassy_width_is_60_l3027_302701


namespace NUMINAMATH_CALUDE_percentage_calculation_l3027_302743

theorem percentage_calculation (total : ℝ) (part : ℝ) (h1 : total = 600) (h2 : part = 150) :
  (part / total) * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3027_302743


namespace NUMINAMATH_CALUDE_relay_team_selection_l3027_302756

/-- The number of sprinters available -/
def total_sprinters : ℕ := 6

/-- The number of sprinters needed for the relay race -/
def relay_team_size : ℕ := 4

/-- The number of sprinters who cannot run the first leg -/
def first_leg_restricted : ℕ := 2

/-- The number of possible team compositions for the relay race -/
def team_compositions : ℕ := 240

theorem relay_team_selection :
  (total_sprinters - first_leg_restricted).choose 1 * 
  (total_sprinters - 1).descFactorial (relay_team_size - 1) = 
  team_compositions := by
  sorry

end NUMINAMATH_CALUDE_relay_team_selection_l3027_302756


namespace NUMINAMATH_CALUDE_shortest_side_right_triangle_l3027_302773

theorem shortest_side_right_triangle (a b c : ℝ) (ha : a = 9) (hb : b = 12) (hc : c^2 = a^2 + b^2) :
  min a (min b c) = 9 := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_right_triangle_l3027_302773


namespace NUMINAMATH_CALUDE_age_difference_l3027_302747

/-- Represents the ages of three brothers -/
structure BrothersAges where
  richard : ℕ
  david : ℕ
  scott : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : BrothersAges) : Prop :=
  ages.richard = ages.david + 6 ∧
  ages.david > ages.scott ∧
  ∃ (y : ℕ), ages.richard + y = 2 * (ages.scott + y) ∧
  ages.david = 14

/-- The theorem to prove -/
theorem age_difference (ages : BrothersAges) :
  problem_conditions ages →
  ∃ (s : ℕ), s < 14 ∧ ages.david - ages.scott = 14 - s :=
by sorry

end NUMINAMATH_CALUDE_age_difference_l3027_302747


namespace NUMINAMATH_CALUDE_circles_intersect_l3027_302714

/-- Two circles in a 2D plane --/
structure TwoCircles where
  /-- First circle: (x-1)^2 + y^2 = 1 --/
  c1 : (ℝ × ℝ) → Prop := fun p => (p.1 - 1)^2 + p.2^2 = 1
  /-- Second circle: x^2 + y^2 + 2x + 4y - 4 = 0 --/
  c2 : (ℝ × ℝ) → Prop := fun p => p.1^2 + p.2^2 + 2*p.1 + 4*p.2 - 4 = 0

/-- The two circles intersect --/
theorem circles_intersect (tc : TwoCircles) : ∃ p : ℝ × ℝ, tc.c1 p ∧ tc.c2 p := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l3027_302714


namespace NUMINAMATH_CALUDE_missing_digit_divisible_by_three_l3027_302712

theorem missing_digit_divisible_by_three :
  ∃ d : ℕ, d < 10 ∧ (43500 + d * 10 + 1) % 3 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_missing_digit_divisible_by_three_l3027_302712


namespace NUMINAMATH_CALUDE_abc_equation_solution_l3027_302710

theorem abc_equation_solution (a b c : ℕ+) (h1 : b ≤ c) 
  (h2 : (a * b - 1) * (a * c - 1) = 2023 * b * c) : 
  c = 82 ∨ c = 167 ∨ c = 1034 := by
sorry

end NUMINAMATH_CALUDE_abc_equation_solution_l3027_302710


namespace NUMINAMATH_CALUDE_sqrt_25_equals_5_l3027_302740

theorem sqrt_25_equals_5 : Real.sqrt 25 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_25_equals_5_l3027_302740


namespace NUMINAMATH_CALUDE_sin_135_degrees_l3027_302786

theorem sin_135_degrees : Real.sin (135 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_135_degrees_l3027_302786


namespace NUMINAMATH_CALUDE_middle_school_running_average_middle_school_running_average_proof_l3027_302770

/-- The average number of minutes run per day by middle school students -/
theorem middle_school_running_average : ℝ :=
  let sixth_grade_minutes : ℝ := 14
  let seventh_grade_minutes : ℝ := 18
  let eighth_grade_minutes : ℝ := 12
  let sixth_to_seventh_ratio : ℝ := 3
  let seventh_to_eighth_ratio : ℝ := 4
  let sports_day_additional_minutes : ℝ := 4
  let days_per_week : ℝ := 7

  let sixth_grade_students : ℝ := seventh_to_eighth_ratio * sixth_to_seventh_ratio
  let seventh_grade_students : ℝ := seventh_to_eighth_ratio
  let eighth_grade_students : ℝ := 1

  let total_students : ℝ := sixth_grade_students + seventh_grade_students + eighth_grade_students

  let average_minutes_with_sports_day : ℝ :=
    (sixth_grade_students * (sixth_grade_minutes * days_per_week + sports_day_additional_minutes) +
     seventh_grade_students * (seventh_grade_minutes * days_per_week + sports_day_additional_minutes) +
     eighth_grade_students * (eighth_grade_minutes * days_per_week + sports_day_additional_minutes)) /
    (total_students * days_per_week)

  15.6

theorem middle_school_running_average_proof : 
  (middle_school_running_average : ℝ) = 15.6 := by sorry

end NUMINAMATH_CALUDE_middle_school_running_average_middle_school_running_average_proof_l3027_302770


namespace NUMINAMATH_CALUDE_linear_function_increasing_l3027_302789

/-- A linear function f(x) = mx + b where m > 0 is increasing -/
theorem linear_function_increasing (m b : ℝ) (h : m > 0) :
  Monotone (fun x => m * x + b) := by sorry

end NUMINAMATH_CALUDE_linear_function_increasing_l3027_302789


namespace NUMINAMATH_CALUDE_count_valid_numbers_is_1800_l3027_302788

/-- Define a 5-digit number -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- Define the quotient and remainder when n is divided by 50 -/
def quotient_remainder (n q r : ℕ) : Prop :=
  n = 50 * q + r ∧ r < 50

/-- Count of 5-digit numbers n where q + r is divisible by 9 -/
def count_valid_numbers : ℕ := sorry

/-- Theorem stating the count of valid numbers is 1800 -/
theorem count_valid_numbers_is_1800 :
  count_valid_numbers = 1800 := by sorry

end NUMINAMATH_CALUDE_count_valid_numbers_is_1800_l3027_302788


namespace NUMINAMATH_CALUDE_mistaken_division_correct_multiplication_l3027_302793

theorem mistaken_division_correct_multiplication : 
  ∀ n : ℕ, 
  (n / 96 = 5) → 
  (n % 96 = 17) → 
  (n * 69 = 34293) := by
sorry

end NUMINAMATH_CALUDE_mistaken_division_correct_multiplication_l3027_302793


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l3027_302733

theorem divisibility_equivalence (a m x n : ℕ) :
  m ∣ n ↔ (x^m - a^m) ∣ (x^n - a^n) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l3027_302733


namespace NUMINAMATH_CALUDE_min_a_bound_l3027_302737

theorem min_a_bound (a : ℝ) : (∀ x : ℝ, x > 0 → x / (x^2 + 3*x + 1) ≤ a) ↔ a ≥ 1/5 := by
  sorry

end NUMINAMATH_CALUDE_min_a_bound_l3027_302737


namespace NUMINAMATH_CALUDE_only_one_correct_statement_l3027_302739

/-- Represents the confidence level in the study conclusion -/
def confidence_level : ℝ := 0.99

/-- Represents the four statements about smoking and lung cancer -/
inductive Statement
  | all_smokers_have_cancer
  | high_probability_of_cancer
  | some_smokers_have_cancer
  | possibly_no_smokers_have_cancer

/-- Determines if a statement is correct given the confidence level -/
def is_correct (s : Statement) (conf : ℝ) : Prop :=
  match s with
  | Statement.possibly_no_smokers_have_cancer => conf < 1
  | _ => False

/-- The main theorem stating that only one statement is correct -/
theorem only_one_correct_statement : 
  (∃! s : Statement, is_correct s confidence_level) ∧ 
  (is_correct Statement.possibly_no_smokers_have_cancer confidence_level) :=
sorry

end NUMINAMATH_CALUDE_only_one_correct_statement_l3027_302739


namespace NUMINAMATH_CALUDE_root_equation_implies_expression_value_l3027_302732

theorem root_equation_implies_expression_value (m : ℝ) : 
  m^2 - 4*m - 2 = 0 → 2*m^2 - 8*m = 4 := by
sorry

end NUMINAMATH_CALUDE_root_equation_implies_expression_value_l3027_302732


namespace NUMINAMATH_CALUDE_division_addition_equality_l3027_302705

theorem division_addition_equality : 0.2 / 0.005 + 0.1 = 40.1 := by
  sorry

end NUMINAMATH_CALUDE_division_addition_equality_l3027_302705


namespace NUMINAMATH_CALUDE_unique_stamp_denomination_l3027_302784

/-- Given stamps of denominations 6, n, and n+2 cents, 
    this function returns the greatest postage that cannot be formed. -/
def greatest_unattainable_postage (n : ℕ) : ℕ :=
  6 * n * (n + 2) - (6 + n + (n + 2))

/-- This theorem states that there exists a unique positive integer n 
    such that the greatest unattainable postage is 120 cents, 
    and this n is equal to 8. -/
theorem unique_stamp_denomination :
  ∃! n : ℕ, n > 0 ∧ greatest_unattainable_postage n = 120 ∧ n = 8 :=
sorry

end NUMINAMATH_CALUDE_unique_stamp_denomination_l3027_302784


namespace NUMINAMATH_CALUDE_video_game_lives_l3027_302791

theorem video_game_lives (initial_lives gained_lives final_lives : ℕ) 
  (h1 : initial_lives = 47)
  (h2 : gained_lives = 46)
  (h3 : final_lives = 70) :
  initial_lives - (final_lives - gained_lives) = 23 := by
  sorry

end NUMINAMATH_CALUDE_video_game_lives_l3027_302791


namespace NUMINAMATH_CALUDE_vegetarians_count_l3027_302716

/-- Represents the number of people in different dietary categories in a family -/
structure FamilyDiet where
  onlyVeg : ℕ
  onlyNonVeg : ℕ
  bothVegAndNonVeg : ℕ

/-- Calculates the total number of people who eat vegetarian food in the family -/
def totalVegetarians (fd : FamilyDiet) : ℕ :=
  fd.onlyVeg + fd.bothVegAndNonVeg

/-- Theorem stating that the number of vegetarians in the family is 28 -/
theorem vegetarians_count (fd : FamilyDiet) 
  (h1 : fd.onlyVeg = 16)
  (h2 : fd.onlyNonVeg = 9)
  (h3 : fd.bothVegAndNonVeg = 12) :
  totalVegetarians fd = 28 := by
  sorry

end NUMINAMATH_CALUDE_vegetarians_count_l3027_302716


namespace NUMINAMATH_CALUDE_discount_and_total_amount_l3027_302776

/-- Given two positive real numbers P and Q where P > Q, 
    this theorem proves the correct calculation of the percentage discount
    and the total amount paid for 10 items. -/
theorem discount_and_total_amount (P Q : ℝ) (h1 : P > Q) (h2 : Q > 0) :
  let d := 100 * (P - Q) / P
  let total := 10 * Q
  (d = 100 * (P - Q) / P) ∧ (total = 10 * Q) := by
  sorry

#check discount_and_total_amount

end NUMINAMATH_CALUDE_discount_and_total_amount_l3027_302776


namespace NUMINAMATH_CALUDE_newspaper_subscription_cost_l3027_302741

theorem newspaper_subscription_cost (discount_rate : ℝ) (discounted_price : ℝ) (normal_price : ℝ) : 
  discount_rate = 0.45 →
  discounted_price = 44 →
  normal_price * (1 - discount_rate) = discounted_price →
  normal_price = 80 := by
sorry

end NUMINAMATH_CALUDE_newspaper_subscription_cost_l3027_302741


namespace NUMINAMATH_CALUDE_prime_pythagorean_inequality_l3027_302777

theorem prime_pythagorean_inequality (p m n : ℕ) 
  (prime_p : Nat.Prime p) 
  (pos_m : m > 0) 
  (pos_n : n > 0) 
  (pyth_eq : p^2 + m^2 = n^2) : 
  m > p := by
sorry

end NUMINAMATH_CALUDE_prime_pythagorean_inequality_l3027_302777


namespace NUMINAMATH_CALUDE_square_with_hole_l3027_302702

theorem square_with_hole (n m : ℕ) (h1 : n^2 - m^2 = 209) (h2 : n > m) : n^2 = 225 := by
  sorry

end NUMINAMATH_CALUDE_square_with_hole_l3027_302702


namespace NUMINAMATH_CALUDE_stock_decrease_duration_l3027_302796

/-- Represents the monthly decrease in bicycle stock -/
def monthly_decrease : ℕ := 4

/-- Represents the total decrease in bicycle stock from January 1 to October 1 -/
def total_decrease : ℕ := 36

/-- Represents the number of months from January 1 to October 1 -/
def total_months : ℕ := 9

/-- Represents the number of months the stock has been decreasing -/
def months_decreasing : ℕ := 5

theorem stock_decrease_duration :
  months_decreasing * monthly_decrease = total_decrease - (total_months - months_decreasing) * monthly_decrease :=
by sorry

end NUMINAMATH_CALUDE_stock_decrease_duration_l3027_302796


namespace NUMINAMATH_CALUDE_no_integer_solution_l3027_302735

theorem no_integer_solution : ¬∃ (m n : ℤ), m^2 = n^2 + 1954 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3027_302735


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l3027_302703

/-- Given an arithmetic sequence {a_n} where a_5/a_3 = 5/9, prove that S_9/S_5 = 1 -/
theorem arithmetic_sequence_sum_ratio (a : ℕ → ℝ) (h : a 5 / a 3 = 5 / 9) :
  let S : ℕ → ℝ := λ n => (n / 2) * (a 1 + a n)
  S 9 / S 5 = 1 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l3027_302703


namespace NUMINAMATH_CALUDE_team_selection_count_l3027_302725

/-- The number of ways to select a team of 5 members from a group of 7 boys and 9 girls, 
    with at least 2 boys in the team -/
def select_team (num_boys num_girls : ℕ) : ℕ :=
  (num_boys.choose 2 * num_girls.choose 3) +
  (num_boys.choose 3 * num_girls.choose 2) +
  (num_boys.choose 4 * num_girls.choose 1) +
  (num_boys.choose 5 * num_girls.choose 0)

/-- Theorem stating that the number of ways to select the team is 3360 -/
theorem team_selection_count :
  select_team 7 9 = 3360 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_count_l3027_302725


namespace NUMINAMATH_CALUDE_expression_value_l3027_302764

theorem expression_value (x y z : ℝ) (hx : x = 1 + Real.sqrt 2) (hy : y = x + 1) (hz : z = x - 1) :
  y^2 * z^4 - 4 * y^3 * z^3 + 6 * y^2 * z^2 + 4 * y = -120 - 92 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3027_302764


namespace NUMINAMATH_CALUDE_square_intersection_perimeter_ratio_l3027_302736

/-- Given a square with vertices at (-2b, -2b), (2b, -2b), (-2b, 2b), and (2b, 2b),
    intersected by the line y = bx, the ratio of the perimeter of one of the
    resulting quadrilaterals to b is equal to 12 + 4√2. -/
theorem square_intersection_perimeter_ratio (b : ℝ) (b_pos : b > 0) :
  let square_vertices := [(-2*b, -2*b), (2*b, -2*b), (-2*b, 2*b), (2*b, 2*b)]
  let intersecting_line := fun x => b * x
  let quadrilateral_perimeter := 12 * b + 4 * b * Real.sqrt 2
  quadrilateral_perimeter / b = 12 + 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_square_intersection_perimeter_ratio_l3027_302736


namespace NUMINAMATH_CALUDE_cost_price_is_100_l3027_302761

/-- Given a toy's cost price, calculate the final selling price after markup and discount --/
def final_price (cost : ℝ) : ℝ := cost * 1.5 * 0.8

/-- The profit made on the toy --/
def profit (cost : ℝ) : ℝ := final_price cost - cost

/-- Theorem stating that if the profit is 20 yuan, the cost price must be 100 yuan --/
theorem cost_price_is_100 : 
  ∀ x : ℝ, profit x = 20 → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_is_100_l3027_302761


namespace NUMINAMATH_CALUDE_polygon_problem_l3027_302783

/-- Represents a polygon with a given number of sides -/
structure Polygon where
  sides : ℕ

/-- The sum of interior angles of a polygon -/
def interiorAngleSum (p : Polygon) : ℕ := 180 * (p.sides - 2)

/-- The number of diagonals in a polygon -/
def diagonalCount (p : Polygon) : ℕ := p.sides * (p.sides - 3) / 2

theorem polygon_problem (x y : Polygon) 
  (h1 : interiorAngleSum x + interiorAngleSum y = 1440)
  (h2 : x.sides * 3 = y.sides) :
  720 = 360 + 360 ∧ 
  x.sides = 3 ∧ 
  y.sides = 9 ∧ 
  diagonalCount y = 27 := by
  sorry

end NUMINAMATH_CALUDE_polygon_problem_l3027_302783


namespace NUMINAMATH_CALUDE_container_capacity_l3027_302760

/-- Given a container where 8 liters represents 20% of its capacity,
    this theorem proves that the total capacity of 40 such containers is 1600 liters. -/
theorem container_capacity (container_capacity : ℝ) 
  (h1 : 8 = 0.2 * container_capacity) 
  (num_containers : ℕ := 40) : 
  (num_containers : ℝ) * container_capacity = 1600 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_l3027_302760


namespace NUMINAMATH_CALUDE_sum_inequality_l3027_302706

theorem sum_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≤ c) : a + b ≤ 3 * c := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l3027_302706


namespace NUMINAMATH_CALUDE_certain_number_value_certain_number_value_proof_l3027_302774

theorem certain_number_value : ℝ → Prop :=
  fun y =>
    let x : ℝ := (390 - (48 + 62 + 98 + 124)) -- x from the second set
    let first_set : List ℝ := [28, x, 42, 78, y]
    let second_set : List ℝ := [48, 62, 98, 124, x]
    (List.sum first_set / first_set.length = 62) ∧
    (List.sum second_set / second_set.length = 78) →
    y = 104

-- The proof goes here
theorem certain_number_value_proof : certain_number_value 104 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_certain_number_value_proof_l3027_302774


namespace NUMINAMATH_CALUDE_ellas_food_consumption_l3027_302754

/-- 
Given that:
1. Ella's dog eats 4 times as much food as Ella each day
2. Ella eats 20 pounds of food per day
3. The total food consumption for Ella and her dog over some number of days is 1000 pounds

This theorem proves that the number of days is 10.
-/
theorem ellas_food_consumption (dog_ratio : ℕ) (ella_daily : ℕ) (total_food : ℕ) :
  dog_ratio = 4 →
  ella_daily = 20 →
  total_food = 1000 →
  ∃ (days : ℕ), days = 10 ∧ total_food = (ella_daily + dog_ratio * ella_daily) * days :=
by sorry

end NUMINAMATH_CALUDE_ellas_food_consumption_l3027_302754


namespace NUMINAMATH_CALUDE_domain_of_z_l3027_302718

def z (x : ℝ) : ℝ := (x - 5) ^ (1/4) + (x + 1) ^ (1/2)

theorem domain_of_z : 
  {x : ℝ | ∃ y, z x = y} = {x : ℝ | x ≥ 5} :=
sorry

end NUMINAMATH_CALUDE_domain_of_z_l3027_302718


namespace NUMINAMATH_CALUDE_multiplication_subtraction_difference_l3027_302779

theorem multiplication_subtraction_difference : ∃ (x : ℝ), x = 10 ∧ (3 * x) - (26 - x) = 14 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_subtraction_difference_l3027_302779


namespace NUMINAMATH_CALUDE_time_to_park_l3027_302762

/-- Represents the jogging scenario with constant pace -/
structure JoggingScenario where
  pace : ℝ  -- Jogging pace in minutes per mile
  cafe_distance : ℝ  -- Distance to café in miles
  cafe_time : ℝ  -- Time to jog to café in minutes
  park_distance : ℝ  -- Distance to park in miles

/-- Given a jogging scenario with constant pace, proves that the time to jog to the park is 36 minutes -/
theorem time_to_park (scenario : JoggingScenario)
  (h1 : scenario.cafe_distance = 3)
  (h2 : scenario.cafe_time = 24)
  (h3 : scenario.park_distance = 4.5)
  (h4 : scenario.pace > 0) :
  scenario.pace * scenario.park_distance = 36 := by
  sorry

#check time_to_park

end NUMINAMATH_CALUDE_time_to_park_l3027_302762


namespace NUMINAMATH_CALUDE_expression_simplification_l3027_302797

theorem expression_simplification 
  (b c d x y : ℝ) (h : cx + dy ≠ 0) :
  (c * x * (c^2 * x^2 + 3 * b^2 * y^2 + c^2 * y^2) + 
   d * y * (b^2 * x^2 + 3 * c^2 * x^2 + b^2 * y^2)) / 
  (c * x + d * y) = 
  c^2 * x^2 + d * b^2 * y^2 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l3027_302797


namespace NUMINAMATH_CALUDE_copresidents_count_l3027_302757

/-- Represents a club with members distributed across departments. -/
structure Club where
  total_members : ℕ
  num_departments : ℕ
  members_per_department : ℕ
  h_total : total_members = num_departments * members_per_department

/-- The number of ways to choose co-presidents from different departments. -/
def choose_copresidents (c : Club) : ℕ :=
  (c.num_departments * c.members_per_department * (c.num_departments - 1) * c.members_per_department) / 2

/-- Theorem stating the number of ways to choose co-presidents for the given club configuration. -/
theorem copresidents_count (c : Club) 
  (h_total : c.total_members = 24)
  (h_departments : c.num_departments = 4)
  (h_distribution : c.members_per_department = 6) : 
  choose_copresidents c = 54 := by
  sorry

#eval choose_copresidents ⟨24, 4, 6, rfl⟩

end NUMINAMATH_CALUDE_copresidents_count_l3027_302757


namespace NUMINAMATH_CALUDE_chess_tournament_matches_l3027_302766

/-- Represents a single elimination tournament --/
structure Tournament :=
  (total_players : ℕ)
  (bye_players : ℕ)
  (h_bye : bye_players < total_players)

/-- Calculates the number of matches in a tournament --/
def matches_played (t : Tournament) : ℕ := t.total_players - 1

/-- Main theorem about the chess tournament --/
theorem chess_tournament_matches :
  ∃ (t : Tournament),
    t.total_players = 120 ∧
    t.bye_players = 40 ∧
    matches_played t = 119 ∧
    119 % 7 = 0 := by sorry

end NUMINAMATH_CALUDE_chess_tournament_matches_l3027_302766


namespace NUMINAMATH_CALUDE_total_investment_total_investment_is_6647_l3027_302709

/-- The problem of calculating total investments --/
theorem total_investment (raghu_investment : ℕ) : ℕ :=
  let trishul_investment := raghu_investment - raghu_investment / 10
  let vishal_investment := trishul_investment + trishul_investment / 10
  raghu_investment + trishul_investment + vishal_investment

/-- The theorem stating that the total investment is 6647 when Raghu invests 2300 --/
theorem total_investment_is_6647 : total_investment 2300 = 6647 := by
  sorry

end NUMINAMATH_CALUDE_total_investment_total_investment_is_6647_l3027_302709


namespace NUMINAMATH_CALUDE_motorcyclist_travel_l3027_302744

theorem motorcyclist_travel (total_distance : ℕ) (first_two_days : ℕ) (second_day_extra : ℕ)
  (h1 : total_distance = 980)
  (h2 : first_two_days = 725)
  (h3 : second_day_extra = 123) :
  ∃ (day1 day2 day3 : ℕ),
    day1 + day2 + day3 = total_distance ∧
    day1 + day2 = first_two_days ∧
    day2 = day3 + second_day_extra ∧
    day1 = 347 ∧
    day2 = 378 ∧
    day3 = 255 := by
  sorry

end NUMINAMATH_CALUDE_motorcyclist_travel_l3027_302744


namespace NUMINAMATH_CALUDE_average_study_time_difference_l3027_302778

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of weekdays -/
def weekdays : ℕ := 5

/-- The number of weekend days -/
def weekend_days : ℕ := 2

/-- The differences in study time on weekdays -/
def weekday_differences : List ℤ := [5, -5, 15, 25, -15]

/-- The additional time Sasha studied on weekends compared to usual -/
def weekend_additional_time : ℤ := 15

/-- The average difference in study time per day -/
def average_difference : ℚ := 12

theorem average_study_time_difference :
  (weekday_differences.sum + 2 * (weekend_additional_time + 15)) / days_in_week = average_difference := by
  sorry

end NUMINAMATH_CALUDE_average_study_time_difference_l3027_302778


namespace NUMINAMATH_CALUDE_father_son_age_relation_l3027_302731

/-- Proves the number of years it takes for a father to be twice as old as his son -/
theorem father_son_age_relation (father_age : ℕ) (son_age : ℕ) (years : ℕ) : 
  father_age = 45 →
  father_age = 3 * son_age →
  father_age + years = 2 * (son_age + years) →
  years = 15 := by
sorry

end NUMINAMATH_CALUDE_father_son_age_relation_l3027_302731


namespace NUMINAMATH_CALUDE_movie_night_kernels_calculation_l3027_302734

/-- Represents a person's popcorn preference --/
structure PopcornPreference where
  name : String
  cups_wanted : ℚ
  kernel_tablespoons : ℚ
  popcorn_cups : ℚ

/-- Calculates the tablespoons of kernels needed for a given preference --/
def kernels_needed (pref : PopcornPreference) : ℚ :=
  pref.kernel_tablespoons * (pref.cups_wanted / pref.popcorn_cups)

/-- The list of popcorn preferences for the movie night --/
def movie_night_preferences : List PopcornPreference := [
  { name := "Joanie", cups_wanted := 3, kernel_tablespoons := 3, popcorn_cups := 6 },
  { name := "Mitchell", cups_wanted := 4, kernel_tablespoons := 2, popcorn_cups := 4 },
  { name := "Miles and Davis", cups_wanted := 6, kernel_tablespoons := 4, popcorn_cups := 8 },
  { name := "Cliff", cups_wanted := 3, kernel_tablespoons := 1, popcorn_cups := 3 }
]

/-- The total tablespoons of kernels needed for the movie night --/
def total_kernels_needed : ℚ :=
  movie_night_preferences.map kernels_needed |>.sum

theorem movie_night_kernels_calculation :
  total_kernels_needed = 15/2 := by
  sorry

#eval total_kernels_needed

end NUMINAMATH_CALUDE_movie_night_kernels_calculation_l3027_302734


namespace NUMINAMATH_CALUDE_remainder_when_consecutive_primes_l3027_302780

theorem remainder_when_consecutive_primes (n : ℕ) :
  Nat.Prime (n + 3) ∧ Nat.Prime (n + 7) → n % 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_when_consecutive_primes_l3027_302780


namespace NUMINAMATH_CALUDE_distance_between_cities_l3027_302708

/-- The distance between City A and City B in miles -/
def distance : ℝ := 427.5

/-- The travel time from City A to City B in hours -/
def time_A_to_B : ℝ := 6

/-- The travel time from City B to City A in hours -/
def time_B_to_A : ℝ := 4.5

/-- The time saved on each trip in hours -/
def time_saved : ℝ := 0.5

/-- The average speed for the round trip if time were saved, in miles per hour -/
def average_speed : ℝ := 90

theorem distance_between_cities :
  distance = 427.5 ∧
  (2 * distance) / (time_A_to_B + time_B_to_A - 2 * time_saved) = average_speed :=
sorry

end NUMINAMATH_CALUDE_distance_between_cities_l3027_302708


namespace NUMINAMATH_CALUDE_units_digit_2019_power_2019_l3027_302781

theorem units_digit_2019_power_2019 : (2019^2019) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_2019_power_2019_l3027_302781


namespace NUMINAMATH_CALUDE_round_310242_to_nearest_thousand_l3027_302719

def round_to_nearest_thousand (n : ℕ) : ℕ :=
  (n + 500) / 1000 * 1000

theorem round_310242_to_nearest_thousand :
  round_to_nearest_thousand 310242 = 310000 := by
  sorry

end NUMINAMATH_CALUDE_round_310242_to_nearest_thousand_l3027_302719


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l3027_302750

def vector_a : ℝ × ℝ := (2, 3)
def vector_b (y : ℝ) : ℝ × ℝ := (4, -1 + y)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_vectors_y_value :
  parallel vector_a (vector_b y) → y = 7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l3027_302750


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_inequality_l3027_302700

theorem smallest_n_for_sqrt_inequality : 
  ∀ n : ℕ, n > 0 → (Real.sqrt n - Real.sqrt (n - 1) < 0.01 ↔ n ≥ 2501) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_inequality_l3027_302700


namespace NUMINAMATH_CALUDE_painted_unit_cubes_in_3x3x3_l3027_302795

/-- Represents a 3D cube -/
structure Cube :=
  (size : ℕ)

/-- Represents a painted cube -/
def PaintedCube := Cube

/-- Represents a unit cube (1x1x1) -/
def UnitCube := Cube

/-- The number of unit cubes with at least one painted surface in a painted cube -/
def num_painted_unit_cubes (c : PaintedCube) : ℕ :=
  sorry

/-- The main theorem: In a 3x3x3 painted cube, 26 unit cubes have at least one painted surface -/
theorem painted_unit_cubes_in_3x3x3 (c : PaintedCube) (h : c.size = 3) :
  num_painted_unit_cubes c = 26 :=
sorry

end NUMINAMATH_CALUDE_painted_unit_cubes_in_3x3x3_l3027_302795


namespace NUMINAMATH_CALUDE_total_points_theorem_l3027_302728

/-- The total points scored by Zach and Ben in a football game -/
def total_points (zach_points ben_points : Float) : Float :=
  zach_points + ben_points

/-- Theorem stating that the total points scored by Zach and Ben is 63.0 -/
theorem total_points_theorem (zach_points ben_points : Float)
  (h1 : zach_points = 42.0)
  (h2 : ben_points = 21.0) :
  total_points zach_points ben_points = 63.0 := by
  sorry

end NUMINAMATH_CALUDE_total_points_theorem_l3027_302728


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_factorial_sum_l3027_302751

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def largest_prime_factor (n : ℕ) : ℕ :=
  (Nat.factors n).foldl max 0

theorem largest_prime_factor_of_factorial_sum :
  largest_prime_factor (factorial 6 + factorial 7) = 5 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_factorial_sum_l3027_302751


namespace NUMINAMATH_CALUDE_fully_filled_boxes_l3027_302721

theorem fully_filled_boxes (total_cards : ℕ) (max_per_box : ℕ) (h1 : total_cards = 94) (h2 : max_per_box = 8) :
  (total_cards / max_per_box : ℕ) = 11 :=
by sorry

end NUMINAMATH_CALUDE_fully_filled_boxes_l3027_302721


namespace NUMINAMATH_CALUDE_frog_jump_probability_l3027_302790

-- Define the square
def Square := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 6}

-- Define a valid jump
def ValidJump (p q : ℝ × ℝ) : Prop :=
  (p.1 = q.1 ∧ |p.2 - q.2| = 1) ∨ (p.2 = q.2 ∧ |p.1 - q.1| = 1)

-- Define the boundary of the square
def Boundary (p : ℝ × ℝ) : Prop :=
  p.1 = 0 ∨ p.1 = 6 ∨ p.2 = 0 ∨ p.2 = 6

-- Define vertical sides
def VerticalSide (p : ℝ × ℝ) : Prop :=
  (p.1 = 0 ∨ p.1 = 6) ∧ 0 ≤ p.2 ∧ p.2 ≤ 6

-- Define the probability function
noncomputable def P (p : ℝ × ℝ) : ℝ :=
  sorry -- The actual implementation would go here

-- State the theorem
theorem frog_jump_probability :
  P (2, 3) = 3/5 :=
sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l3027_302790


namespace NUMINAMATH_CALUDE_max_xy_given_constraint_l3027_302765

theorem max_xy_given_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 7 * x + 8 * y = 112) :
  x * y ≤ 56 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 7 * x₀ + 8 * y₀ = 112 ∧ x₀ * y₀ = 56 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_given_constraint_l3027_302765


namespace NUMINAMATH_CALUDE_subset_families_inequality_l3027_302729

/-- Given an n-element set X and two families of subsets 𝓐 and 𝓑 of X, 
    where each subset in 𝓐 cannot be compared with every subset in 𝓑, 
    prove that √|𝓐| + √|𝓑| ≤ 2^(7/2). -/
theorem subset_families_inequality (n : ℕ) (X : Finset (Finset ℕ)) 
  (𝓐 𝓑 : Finset (Finset ℕ)) : 
  (∀ A ∈ 𝓐, ∀ B ∈ 𝓑, ¬(A ⊆ B ∨ B ⊆ A)) →
  X.card = n →
  (∀ A ∈ 𝓐, A ∈ X) →
  (∀ B ∈ 𝓑, B ∈ X) →
  Real.sqrt (𝓐.card : ℝ) + Real.sqrt (𝓑.card : ℝ) ≤ 2^(7/2) :=
by sorry

end NUMINAMATH_CALUDE_subset_families_inequality_l3027_302729


namespace NUMINAMATH_CALUDE_sum_of_seventh_powers_squared_l3027_302771

theorem sum_of_seventh_powers_squared (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (sum_zero : a + b + c = 0) :
  (a^7 + b^7 + c^7)^2 / ((a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) * (a^4 + b^4 + c^4) * (a^5 + b^5 + c^5)) = 49/60 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seventh_powers_squared_l3027_302771


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3027_302746

/-- Given two vectors AB and CD in R², where AB is perpendicular to CD,
    prove that the y-coordinate of AB is 1. -/
theorem perpendicular_vectors (x : ℝ) : 
  let AB : ℝ × ℝ := (3, x)
  let CD : ℝ × ℝ := (-2, 6)
  (AB.1 * CD.1 + AB.2 * CD.2 = 0) → x = 1 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3027_302746


namespace NUMINAMATH_CALUDE_cube_root_over_fifth_root_of_five_l3027_302715

theorem cube_root_over_fifth_root_of_five (x : ℝ) (hx : x > 0) :
  (x^(1/3)) / (x^(1/5)) = x^(2/15) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_over_fifth_root_of_five_l3027_302715


namespace NUMINAMATH_CALUDE_max_sundays_in_53_days_l3027_302723

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of days we're considering -/
def total_days : ℕ := 53

/-- A function that returns the number of Sundays in a given number of days -/
def sundays_in_days (days : ℕ) : ℕ := days / days_in_week

theorem max_sundays_in_53_days : 
  sundays_in_days total_days = 7 := by sorry

end NUMINAMATH_CALUDE_max_sundays_in_53_days_l3027_302723


namespace NUMINAMATH_CALUDE_conic_parabola_focus_coincidence_l3027_302722

/-- Given a conic section and a parabola, prove that the parameter m of the conic section is 9 when their foci coincide. -/
theorem conic_parabola_focus_coincidence (m : ℝ) : 
  m ≠ 0 → m ≠ 5 → 
  (∃ (x y : ℝ), x^2 / m + y^2 / 5 = 1) →
  (∃ (x y : ℝ), y^2 = 8*x) →
  (∃ (x₀ y₀ : ℝ), x₀^2 / m + y₀^2 / 5 = 1 ∧ y₀^2 = 8*x₀ ∧ x₀ = 2 ∧ y₀ = 0) →
  m = 9 :=
by sorry

end NUMINAMATH_CALUDE_conic_parabola_focus_coincidence_l3027_302722


namespace NUMINAMATH_CALUDE_cubic_transformation_1993_l3027_302707

/-- Cubic transformation: sum of cubes of digits --/
def cubicTransform (n : ℕ) : ℕ := sorry

/-- Sequence of cubic transformations starting from n --/
def cubicSequence (n : ℕ) : ℕ → ℕ
  | 0 => n
  | i + 1 => cubicTransform (cubicSequence n i)

/-- Predicate for sequence alternating between two values --/
def alternatesBetween (seq : ℕ → ℕ) (a b : ℕ) : Prop :=
  ∃ k : ℕ, ∀ i ≥ k, seq i = a ∧ seq (i + 1) = b ∨ seq i = b ∧ seq (i + 1) = a

theorem cubic_transformation_1993 :
  alternatesBetween (cubicSequence 1993) 1459 919 := by sorry

end NUMINAMATH_CALUDE_cubic_transformation_1993_l3027_302707


namespace NUMINAMATH_CALUDE_rhombus60_min_rotation_l3027_302726

/-- A rhombus with a 60° angle -/
structure Rhombus60 where
  /-- The rhombus has a 60° angle -/
  angle : ℝ
  angle_eq : angle = 60

/-- The minimum rotation angle for a Rhombus60 to coincide with its original position -/
def min_rotation_angle (r : Rhombus60) : ℝ := 180

/-- Theorem stating that the minimum rotation angle for a Rhombus60 is 180° -/
theorem rhombus60_min_rotation (r : Rhombus60) :
  min_rotation_angle r = 180 := by sorry

end NUMINAMATH_CALUDE_rhombus60_min_rotation_l3027_302726


namespace NUMINAMATH_CALUDE_mahesh_estimate_less_than_true_value_l3027_302767

theorem mahesh_estimate_less_than_true_value 
  (a b d : ℕ) 
  (h1 : a > b) 
  (h2 : d > 0) : 
  (a - d)^2 - (b + d)^2 < a^2 - b^2 := by
  sorry

end NUMINAMATH_CALUDE_mahesh_estimate_less_than_true_value_l3027_302767


namespace NUMINAMATH_CALUDE_smallest_consecutive_triangle_perimeter_l3027_302738

/-- A triangle with consecutive integer side lengths. -/
structure ConsecutiveTriangle where
  a : ℕ
  valid : a > 0

/-- The three side lengths of a ConsecutiveTriangle. -/
def ConsecutiveTriangle.sides (t : ConsecutiveTriangle) : Fin 3 → ℕ
  | 0 => t.a
  | 1 => t.a + 1
  | 2 => t.a + 2

/-- The perimeter of a ConsecutiveTriangle. -/
def ConsecutiveTriangle.perimeter (t : ConsecutiveTriangle) : ℕ :=
  3 * t.a + 3

/-- Predicate for whether a ConsecutiveTriangle satisfies the Triangle Inequality. -/
def ConsecutiveTriangle.satisfiesTriangleInequality (t : ConsecutiveTriangle) : Prop :=
  t.sides 0 + t.sides 1 > t.sides 2 ∧
  t.sides 0 + t.sides 2 > t.sides 1 ∧
  t.sides 1 + t.sides 2 > t.sides 0

/-- The smallest ConsecutiveTriangle that satisfies the Triangle Inequality. -/
def smallestValidConsecutiveTriangle : ConsecutiveTriangle :=
  { a := 2
    valid := by simp }

/-- Theorem: The smallest possible perimeter of a triangle with consecutive integer side lengths is 9. -/
theorem smallest_consecutive_triangle_perimeter :
  (∀ t : ConsecutiveTriangle, t.satisfiesTriangleInequality → t.perimeter ≥ 9) ∧
  smallestValidConsecutiveTriangle.satisfiesTriangleInequality ∧
  smallestValidConsecutiveTriangle.perimeter = 9 :=
sorry

end NUMINAMATH_CALUDE_smallest_consecutive_triangle_perimeter_l3027_302738


namespace NUMINAMATH_CALUDE_sides_formula_l3027_302794

/-- The number of sides in the nth figure of a sequence starting with a hexagon,
    where each subsequent figure has 5 more sides than the previous one. -/
def sides (n : ℕ) : ℕ := 5 * n + 1

/-- Theorem stating that the number of sides in the nth figure is 5n + 1 -/
theorem sides_formula (n : ℕ) : sides n = 5 * n + 1 := by
  sorry

end NUMINAMATH_CALUDE_sides_formula_l3027_302794


namespace NUMINAMATH_CALUDE_courier_travel_times_l3027_302792

/-- Represents a courier with their travel times -/
structure Courier where
  meetingTime : ℝ
  remainingTime : ℝ

/-- Proves that given the conditions, the couriers' total travel times are 28 and 21 hours -/
theorem courier_travel_times (c1 c2 : Courier) 
  (h1 : c1.remainingTime = 16)
  (h2 : c2.remainingTime = 9)
  (h3 : c1.meetingTime = c2.meetingTime)
  (h4 : c1.meetingTime * (1 / c1.meetingTime + 1 / c2.meetingTime) = 1) :
  (c1.meetingTime + c1.remainingTime = 28) ∧ 
  (c2.meetingTime + c2.remainingTime = 21) := by
  sorry

#check courier_travel_times

end NUMINAMATH_CALUDE_courier_travel_times_l3027_302792


namespace NUMINAMATH_CALUDE_coloring_books_shelves_l3027_302720

/-- Calculates the number of shelves needed to display remaining coloring books --/
def shelves_needed (initial_stock : ℕ) (sold : ℕ) (donated : ℕ) (books_per_shelf : ℕ) : ℕ :=
  ((initial_stock - sold - donated) + books_per_shelf - 1) / books_per_shelf

/-- Theorem stating that given the problem conditions, 6 shelves are needed --/
theorem coloring_books_shelves :
  shelves_needed 150 55 30 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_coloring_books_shelves_l3027_302720


namespace NUMINAMATH_CALUDE_solve_grocery_problem_l3027_302772

def grocery_problem (total_brought chicken veggies eggs dog_food left_after meat : ℕ) : Prop :=
  total_brought = 167 ∧
  chicken = 22 ∧
  veggies = 43 ∧
  eggs = 5 ∧
  dog_food = 45 ∧
  left_after = 35 ∧
  meat = total_brought - (chicken + veggies + eggs + dog_food + left_after)

theorem solve_grocery_problem :
  ∃ meat, grocery_problem 167 22 43 5 45 35 meat ∧ meat = 17 := by sorry

end NUMINAMATH_CALUDE_solve_grocery_problem_l3027_302772


namespace NUMINAMATH_CALUDE_quadratic_monotonicity_l3027_302748

/-- A quadratic function f(x) = ax² + bx + 3 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 3

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 2 * a * x + b

theorem quadratic_monotonicity (a b : ℝ) :
  (∀ x ≤ -1, 0 ≤ f' a b x) →
  (∀ x ≥ -1, f' a b x ≤ 0) →
  b = 2 * a ∧ a < 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_monotonicity_l3027_302748


namespace NUMINAMATH_CALUDE_julia_balls_count_l3027_302717

/-- The number of balls Julia bought -/
def total_balls (red_packs yellow_packs green_packs balls_per_pack : ℕ) : ℕ :=
  (red_packs + yellow_packs + green_packs) * balls_per_pack

/-- Proof that Julia bought 399 balls -/
theorem julia_balls_count :
  total_balls 3 10 8 19 = 399 := by
  sorry

end NUMINAMATH_CALUDE_julia_balls_count_l3027_302717


namespace NUMINAMATH_CALUDE_lapis_share_is_correct_l3027_302724

/-- Represents the share of treasure for a person -/
structure TreasureShare where
  amount : ℚ
  deriving Repr

/-- Calculates the share of treasure based on contribution -/
def calculateShare (contribution : ℚ) (totalContribution : ℚ) (treasureValue : ℚ) : TreasureShare :=
  { amount := (contribution / totalContribution) * treasureValue }

theorem lapis_share_is_correct (fonzie_contribution : ℚ) (aunt_bee_contribution : ℚ) (lapis_contribution : ℚ) (treasure_value : ℚ)
    (h1 : fonzie_contribution = 7000)
    (h2 : aunt_bee_contribution = 8000)
    (h3 : lapis_contribution = 9000)
    (h4 : treasure_value = 900000) :
  (calculateShare lapis_contribution (fonzie_contribution + aunt_bee_contribution + lapis_contribution) treasure_value).amount = 337500 := by
  sorry

#eval calculateShare 9000 24000 900000

end NUMINAMATH_CALUDE_lapis_share_is_correct_l3027_302724


namespace NUMINAMATH_CALUDE_apple_cost_price_l3027_302727

/-- Proves that the cost price of an apple is 24 rupees, given the selling price and loss ratio. -/
theorem apple_cost_price (selling_price : ℚ) (loss_ratio : ℚ) : 
  selling_price = 20 → loss_ratio = 1/6 → 
  ∃ cost_price : ℚ, cost_price = 24 ∧ selling_price = cost_price - loss_ratio * cost_price :=
by sorry

end NUMINAMATH_CALUDE_apple_cost_price_l3027_302727


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3027_302753

/-- A geometric sequence with positive terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 1 →
  a 1 + a 3 + a 5 = 21 →
  a 2 + a 4 + a 6 = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3027_302753


namespace NUMINAMATH_CALUDE_existence_of_n_l3027_302763

theorem existence_of_n (p : ℕ) (a k : ℕ+) (h_prime : Nat.Prime p) 
  (h_bound : p ^ a.val < k.val ∧ k.val < 2 * p ^ a.val) :
  ∃ n : ℕ, n < p ^ (2 * a.val) ∧ 
    (Nat.choose n k.val) % (p ^ a.val) = n % (p ^ a.val) ∧ 
    n % (p ^ a.val) = k.val % (p ^ a.val) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_n_l3027_302763


namespace NUMINAMATH_CALUDE_point_line_plane_relationship_l3027_302713

-- Define the types for point, line, and plane
variable (Point Line Plane : Type)

-- Define the relationships
variable (lies_on : Point → Line → Prop)
variable (lies_in : Line → Plane → Prop)

-- Define the set membership and subset relations
variable (elem : Point → Line → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem point_line_plane_relationship 
  (A : Point) (a : Line) (α : Plane) :
  lies_on A a → lies_in a α → 
  (elem A a ∧ subset a α) :=
sorry

end NUMINAMATH_CALUDE_point_line_plane_relationship_l3027_302713


namespace NUMINAMATH_CALUDE_planes_parallel_to_same_plane_are_parallel_lines_perpendicular_to_same_plane_are_parallel_l3027_302730

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relationships between geometric objects
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_plane_to : Plane → Plane → Prop)
variable (perpendicular_line_to_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- Theorem 1: Two planes parallel to the same plane are parallel
theorem planes_parallel_to_same_plane_are_parallel
  (P Q R : Plane)
  (h1 : parallel_plane_to P R)
  (h2 : parallel_plane_to Q R) :
  parallel_planes P Q :=
sorry

-- Theorem 2: Two lines perpendicular to the same plane are parallel
theorem lines_perpendicular_to_same_plane_are_parallel
  (l1 l2 : Line) (P : Plane)
  (h1 : perpendicular_line_to_plane l1 P)
  (h2 : perpendicular_line_to_plane l2 P) :
  parallel_lines l1 l2 :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_to_same_plane_are_parallel_lines_perpendicular_to_same_plane_are_parallel_l3027_302730


namespace NUMINAMATH_CALUDE_sin_cos_sixty_degrees_l3027_302787

theorem sin_cos_sixty_degrees :
  Real.sin (π / 3) = Real.sqrt 3 / 2 ∧ Real.cos (π / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sixty_degrees_l3027_302787


namespace NUMINAMATH_CALUDE_expression_result_l3027_302755

theorem expression_result : (3.242 * 10) / 100 = 0.3242 := by
  sorry

end NUMINAMATH_CALUDE_expression_result_l3027_302755
