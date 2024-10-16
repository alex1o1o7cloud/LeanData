import Mathlib

namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l1467_146752

theorem y_in_terms_of_x (x y : ℝ) (h : y - 2*x = 5) : y = 2*x + 5 := by
  sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l1467_146752


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1467_146727

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := 4 * n + 1

-- Define the sum of the first n terms of a_n
def S (n : ℕ) : ℚ := n * (a 1 + a n) / 2

-- Define T_n
def T (n : ℕ) : ℚ := n / (2 * n + 2)

theorem arithmetic_sequence_properties :
  (a 2 = 9) ∧ (S 5 = 65) →
  (∀ n : ℕ, n ≥ 1 → a n = 4 * n + 1) ∧
  (∀ n : ℕ, n ≥ 1 → T n = n / (2 * n + 2)) := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1467_146727


namespace NUMINAMATH_CALUDE_proposition_is_true_l1467_146739

theorem proposition_is_true : ∀ x : ℝ, x > 2 → Real.log (x - 1) + x^2 + 4 > 4*x := by
  sorry

end NUMINAMATH_CALUDE_proposition_is_true_l1467_146739


namespace NUMINAMATH_CALUDE_parallel_lines_iff_a_eq_two_l1467_146783

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel (m1 m2 : ℝ) : Prop := m1 = m2

theorem parallel_lines_iff_a_eq_two (a : ℝ) :
  parallel (2 / a) ((a - 1) / 1) ↔ a = 2 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_iff_a_eq_two_l1467_146783


namespace NUMINAMATH_CALUDE_smallest_integer_l1467_146732

theorem smallest_integer (a b : ℕ) (ha : a = 80) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 40) :
  ∃ (m : ℕ), m ≥ b ∧ m = 50 ∧ Nat.lcm a m / Nat.gcd a m = 40 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_l1467_146732


namespace NUMINAMATH_CALUDE_books_read_l1467_146795

theorem books_read (total : ℕ) (unread : ℕ) (h1 : total = 21) (h2 : unread = 8) :
  total - unread = 13 := by
  sorry

end NUMINAMATH_CALUDE_books_read_l1467_146795


namespace NUMINAMATH_CALUDE_class_size_l1467_146763

theorem class_size (n : ℕ) 
  (h1 : 30 * 160 + (n - 30) * 156 = n * 159) : n = 40 := by
  sorry

#check class_size

end NUMINAMATH_CALUDE_class_size_l1467_146763


namespace NUMINAMATH_CALUDE_uninsured_part_time_percentage_l1467_146751

/-- Represents the survey data and calculates the percentage of uninsured part-time employees -/
def survey_data (total : ℕ) (uninsured : ℕ) (part_time : ℕ) (neither_prob : ℚ) : ℚ :=
  let uninsured_part_time := total - (neither_prob * total).num - uninsured - part_time
  (uninsured_part_time / uninsured) * 100

/-- Theorem stating that given the survey conditions, the percentage of uninsured employees
    who work part-time is approximately 12.5% -/
theorem uninsured_part_time_percentage :
  let result := survey_data 330 104 54 (559606060606060606 / 1000000000000000000)
  ∃ (ε : ℚ), abs (result - 125/10) < ε ∧ ε < 1/10 := by
  sorry

end NUMINAMATH_CALUDE_uninsured_part_time_percentage_l1467_146751


namespace NUMINAMATH_CALUDE_max_area_rectangle_perimeter_100_l1467_146782

/-- The maximum area of a rectangle with perimeter 100 and integer side lengths --/
theorem max_area_rectangle_perimeter_100 :
  ∃ (w h : ℕ), w + h = 50 ∧ w * h = 625 ∧ 
  ∀ (x y : ℕ), x + y = 50 → x * y ≤ 625 := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_perimeter_100_l1467_146782


namespace NUMINAMATH_CALUDE_factor_calculation_l1467_146786

theorem factor_calculation (original : ℝ) (factor : ℝ) : 
  original = 5 → 
  (2 * original + 9) * factor = 57 → 
  factor = 3 := by
sorry

end NUMINAMATH_CALUDE_factor_calculation_l1467_146786


namespace NUMINAMATH_CALUDE_smallest_block_with_297_hidden_cubes_l1467_146776

/-- Given a rectangular block where 297 cubes are not visible when viewed from a corner,
    prove that the smallest possible total number of cubes is 192. -/
theorem smallest_block_with_297_hidden_cubes :
  ∀ l m n : ℕ,
  (l - 1) * (m - 1) * (n - 1) = 297 →
  l * m * n ≥ 192 ∧
  (∃ l' m' n' : ℕ, (l' - 1) * (m' - 1) * (n' - 1) = 297 ∧ l' * m' * n' = 192) :=
by sorry

end NUMINAMATH_CALUDE_smallest_block_with_297_hidden_cubes_l1467_146776


namespace NUMINAMATH_CALUDE_caps_first_week_l1467_146780

/-- The number of caps made in the first week -/
def first_week : ℕ := sorry

/-- The number of caps made in the second week -/
def second_week : ℕ := 400

/-- The number of caps made in the third week -/
def third_week : ℕ := 300

/-- The total number of caps made in four weeks -/
def total_caps : ℕ := 1360

theorem caps_first_week : 
  first_week = 320 ∧
  second_week = 400 ∧
  third_week = 300 ∧
  first_week + second_week + third_week + (first_week + second_week + third_week) / 3 = total_caps :=
by sorry

end NUMINAMATH_CALUDE_caps_first_week_l1467_146780


namespace NUMINAMATH_CALUDE_choose_three_cooks_from_ten_l1467_146799

theorem choose_three_cooks_from_ten (n : ℕ) (k : ℕ) : n = 10 → k = 3 → Nat.choose n k = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_cooks_from_ten_l1467_146799


namespace NUMINAMATH_CALUDE_fifth_number_in_list_l1467_146754

theorem fifth_number_in_list (numbers : List ℕ) : 
  numbers.length = 9 ∧ 
  numbers.sum = 207 * 9 ∧
  201 ∈ numbers ∧ 
  202 ∈ numbers ∧ 
  204 ∈ numbers ∧ 
  205 ∈ numbers ∧ 
  209 ∈ numbers ∧ 
  209 ∈ numbers ∧ 
  210 ∈ numbers ∧ 
  212 ∈ numbers ∧ 
  212 ∈ numbers →
  ∃ (fifth : ℕ), fifth ∈ numbers ∧ fifth = 211 := by
sorry

end NUMINAMATH_CALUDE_fifth_number_in_list_l1467_146754


namespace NUMINAMATH_CALUDE_equipment_cost_proof_l1467_146714

/-- The number of players on the team -/
def num_players : ℕ := 16

/-- The cost of a jersey in dollars -/
def jersey_cost : ℚ := 25

/-- The cost of shorts in dollars -/
def shorts_cost : ℚ := 152/10

/-- The cost of socks in dollars -/
def socks_cost : ℚ := 68/10

/-- The total cost of equipment for all players on the team -/
def total_cost : ℚ := num_players * (jersey_cost + shorts_cost + socks_cost)

theorem equipment_cost_proof : total_cost = 752 := by
  sorry

end NUMINAMATH_CALUDE_equipment_cost_proof_l1467_146714


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1467_146790

theorem triangle_abc_properties (A B C : Real) (R : Real) (BC AC : Real) :
  0 < R →
  0 < BC →
  0 < AC →
  C = 3 * Real.pi / 4 →
  Real.sin (A + C) = (BC / R) * Real.cos (A + B) →
  (1 / 2) * BC * AC * Real.sin C = 1 →
  (BC * AC = AC * (2 * BC)) ∧ 
  (AC * BC = A + B) ∧
  AC ^ 2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1467_146790


namespace NUMINAMATH_CALUDE_investment_condition_l1467_146768

/-- Represents the investment scenario with three banks -/
structure InvestmentScenario where
  national_investment : ℝ
  national_rate : ℝ
  a_rate : ℝ
  b_rate : ℝ
  total_rate : ℝ

/-- The given investment scenario -/
def given_scenario : InvestmentScenario :=
  { national_investment := 7500
  , national_rate := 0.09
  , a_rate := 0.12
  , b_rate := 0.14
  , total_rate := 0.11 }

/-- The total annual income from all three banks -/
def total_income (s : InvestmentScenario) (a b : ℝ) : ℝ :=
  s.national_rate * s.national_investment + s.a_rate * a + s.b_rate * b

/-- The total investment across all three banks -/
def total_investment (s : InvestmentScenario) (a b : ℝ) : ℝ :=
  s.national_investment + a + b

/-- The theorem stating the condition for the desired total annual income -/
theorem investment_condition (s : InvestmentScenario) (a b : ℝ) :
  total_income s a b = s.total_rate * total_investment s a b ↔ 0.01 * a + 0.03 * b = 150 :=
by sorry

end NUMINAMATH_CALUDE_investment_condition_l1467_146768


namespace NUMINAMATH_CALUDE_division_equivalence_l1467_146778

theorem division_equivalence (h : 43 * 47 = 2021) : (-43) / (1 / 47) = -2021 := by
  sorry

end NUMINAMATH_CALUDE_division_equivalence_l1467_146778


namespace NUMINAMATH_CALUDE_degree_of_monomial_l1467_146744

/-- The degree of a monomial is the sum of the exponents of its variables -/
def monomialDegree (coefficient : ℤ) (xExponent yExponent : ℕ) : ℕ :=
  xExponent + yExponent

/-- The monomial -3x^5y^2 has degree 7 -/
theorem degree_of_monomial :
  monomialDegree (-3) 5 2 = 7 := by sorry

end NUMINAMATH_CALUDE_degree_of_monomial_l1467_146744


namespace NUMINAMATH_CALUDE_tv_show_watch_time_l1467_146717

/-- Calculates the total watch time for a TV show with regular seasons and a final season -/
def total_watch_time (regular_seasons : ℕ) (episodes_per_regular_season : ℕ) 
  (extra_episodes_final_season : ℕ) (hours_per_episode : ℚ) : ℚ :=
  let total_episodes := regular_seasons * episodes_per_regular_season + 
    (episodes_per_regular_season + extra_episodes_final_season)
  total_episodes * hours_per_episode

/-- Theorem stating that the total watch time for the given TV show is 112 hours -/
theorem tv_show_watch_time : 
  total_watch_time 9 22 4 (1/2) = 112 := by sorry

end NUMINAMATH_CALUDE_tv_show_watch_time_l1467_146717


namespace NUMINAMATH_CALUDE_robins_haircut_l1467_146761

theorem robins_haircut (initial_length current_length : ℕ) 
  (h1 : initial_length = 17)
  (h2 : current_length = 13) :
  initial_length - current_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_robins_haircut_l1467_146761


namespace NUMINAMATH_CALUDE_smallest_consecutive_multiples_l1467_146726

theorem smallest_consecutive_multiples : 
  let a := 1735
  ∀ n : ℕ, n < a → ¬(
    (n.succ % 5 = 0) ∧ 
    ((n + 2) % 7 = 0) ∧ 
    ((n + 3) % 9 = 0) ∧ 
    ((n + 4) % 11 = 0)
  ) ∧
  (a % 5 = 0) ∧ 
  ((a + 1) % 7 = 0) ∧ 
  ((a + 2) % 9 = 0) ∧ 
  ((a + 3) % 11 = 0) := by
sorry

end NUMINAMATH_CALUDE_smallest_consecutive_multiples_l1467_146726


namespace NUMINAMATH_CALUDE_smallest_prime_factors_difference_l1467_146756

def number : Nat := 96043

theorem smallest_prime_factors_difference (p q : Nat) : 
  Prime p ∧ Prime q ∧ p ∣ number ∧ q ∣ number ∧
  (∀ r, Prime r → r ∣ number → r ≥ p) ∧
  (∀ r, Prime r → r ∣ number → r = p ∨ r ≥ q) →
  q - p = 4 := by sorry

end NUMINAMATH_CALUDE_smallest_prime_factors_difference_l1467_146756


namespace NUMINAMATH_CALUDE_nesbitts_inequality_l1467_146716

theorem nesbitts_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2 ∧
  (a / (b + c) + b / (c + a) + c / (a + b) = 3 / 2 ↔ a = b ∧ b = c) := by
  sorry

end NUMINAMATH_CALUDE_nesbitts_inequality_l1467_146716


namespace NUMINAMATH_CALUDE_modular_inverse_72_l1467_146749

theorem modular_inverse_72 (h : (17⁻¹ : ZMod 89) = 53) : (72⁻¹ : ZMod 89) = 36 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_72_l1467_146749


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_l1467_146721

theorem sqrt_sum_fractions : Real.sqrt (25 / 36 + 16 / 9) = Real.sqrt 89 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_l1467_146721


namespace NUMINAMATH_CALUDE_expression_evaluation_l1467_146700

theorem expression_evaluation : 200 * (200 + 5) - (200 * 200 + 5) = 995 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1467_146700


namespace NUMINAMATH_CALUDE_max_bishops_on_mountain_board_l1467_146706

/-- A chessboard with two mountains --/
structure MountainChessboard :=
  (black_regions : ℕ)
  (white_regions : ℕ)

/-- The maximum number of non-attacking bishops on a mountain chessboard --/
def max_bishops (board : MountainChessboard) : ℕ :=
  board.black_regions + board.white_regions

/-- Theorem: The maximum number of non-attacking bishops on the given mountain chessboard is 19 --/
theorem max_bishops_on_mountain_board :
  ∃ (board : MountainChessboard), 
    board.black_regions = 11 ∧ 
    board.white_regions = 8 ∧ 
    max_bishops board = 19 := by
  sorry

#eval max_bishops ⟨11, 8⟩

end NUMINAMATH_CALUDE_max_bishops_on_mountain_board_l1467_146706


namespace NUMINAMATH_CALUDE_kimberly_skittles_l1467_146769

/-- The number of Skittles Kimberly bought -/
def skittles_bought (initial : ℕ) (final : ℕ) : ℕ := final - initial

/-- Proof that Kimberly bought 7 Skittles -/
theorem kimberly_skittles : skittles_bought 5 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_kimberly_skittles_l1467_146769


namespace NUMINAMATH_CALUDE_cube_gt_of_gt_l1467_146745

theorem cube_gt_of_gt (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_gt_of_gt_l1467_146745


namespace NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l1467_146710

theorem smallest_m_for_integral_solutions : 
  ∃ (m : ℕ), m > 0 ∧ 
  (∃ (x y : ℤ), 10 * x^2 - m * x + 180 = 0 ∧ 10 * y^2 - m * y + 180 = 0 ∧ x ≠ y) ∧
  (∀ (k : ℕ), k > 0 ∧ k < m → 
    ¬∃ (x y : ℤ), 10 * x^2 - k * x + 180 = 0 ∧ 10 * y^2 - k * y + 180 = 0 ∧ x ≠ y) ∧
  m = 90 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l1467_146710


namespace NUMINAMATH_CALUDE_candidate_a_vote_percentage_l1467_146758

theorem candidate_a_vote_percentage
  (total_voters : ℕ)
  (democrat_percentage : ℚ)
  (republican_percentage : ℚ)
  (democrat_for_a_percentage : ℚ)
  (republican_for_a_percentage : ℚ)
  (h1 : democrat_percentage = 60 / 100)
  (h2 : republican_percentage = 1 - democrat_percentage)
  (h3 : democrat_for_a_percentage = 70 / 100)
  (h4 : republican_for_a_percentage = 20 / 100)
  : (democrat_percentage * democrat_for_a_percentage +
     republican_percentage * republican_for_a_percentage) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_candidate_a_vote_percentage_l1467_146758


namespace NUMINAMATH_CALUDE_probability_even_sum_is_one_third_l1467_146784

def digits : Finset ℕ := {2, 3, 5}

def is_valid_arrangement (a b c d : ℕ) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  (a = 2 ∧ b = 2) ∨ (a = 2 ∧ c = 2) ∨ (a = 2 ∧ d = 2) ∨
  (b = 2 ∧ c = 2) ∨ (b = 2 ∧ d = 2) ∨ (c = 2 ∧ d = 2)

def sum_first_last_even (a d : ℕ) : Prop :=
  (a + d) % 2 = 0

def count_valid_arrangements : ℕ := 12

def count_even_sum_arrangements : ℕ := 4

theorem probability_even_sum_is_one_third :
  (count_even_sum_arrangements : ℚ) / count_valid_arrangements = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_sum_is_one_third_l1467_146784


namespace NUMINAMATH_CALUDE_students_with_d_grade_l1467_146788

/-- Proves that in a course with approximately 600 students, where 1/5 of grades are A's,
    1/4 are B's, 1/2 are C's, and the remaining are D's, the number of students who
    received a D is 30. -/
theorem students_with_d_grade (total_students : ℕ) (a_fraction b_fraction c_fraction : ℚ)
  (h_total : total_students = 600)
  (h_a : a_fraction = 1 / 5)
  (h_b : b_fraction = 1 / 4)
  (h_c : c_fraction = 1 / 2)
  (h_sum : a_fraction + b_fraction + c_fraction < 1) :
  total_students - (a_fraction + b_fraction + c_fraction) * total_students = 30 :=
sorry

end NUMINAMATH_CALUDE_students_with_d_grade_l1467_146788


namespace NUMINAMATH_CALUDE_increasing_function_implies_a_leq_neg_two_l1467_146701

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

-- State the theorem
theorem increasing_function_implies_a_leq_neg_two :
  ∀ a : ℝ, (∀ x y : ℝ, -2 < x ∧ x < y ∧ y < 2 → f a x < f a y) →
  a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_implies_a_leq_neg_two_l1467_146701


namespace NUMINAMATH_CALUDE_jennifer_remaining_money_l1467_146723

def initial_amount : ℚ := 360
def sandwich_proportion : ℚ := 3/10
def museum_proportion : ℚ := 1/4
def book_proportion : ℚ := 35/100
def charity_proportion : ℚ := 1/8

theorem jennifer_remaining_money :
  let sandwich_cost := initial_amount * sandwich_proportion
  let museum_cost := initial_amount * museum_proportion
  let book_cost := initial_amount * book_proportion
  let total_spent := sandwich_cost + museum_cost + book_cost
  let remaining_before_charity := initial_amount - total_spent
  let charity_donation := remaining_before_charity * charity_proportion
  let final_remaining := remaining_before_charity - charity_donation
  final_remaining = 63/2 := by
sorry

end NUMINAMATH_CALUDE_jennifer_remaining_money_l1467_146723


namespace NUMINAMATH_CALUDE_wang_loss_is_97_l1467_146787

-- Define the relevant quantities
def gift_cost : ℕ := 18
def gift_price : ℕ := 21
def payment : ℕ := 100
def change_given : ℕ := 79
def counterfeit_bill : ℕ := 100
def neighbor_repayment : ℕ := 100

-- Define Mr. Wang's loss
def wang_loss : ℕ := change_given + gift_cost + neighbor_repayment - payment

-- Theorem statement
theorem wang_loss_is_97 : wang_loss = 97 := by
  sorry

end NUMINAMATH_CALUDE_wang_loss_is_97_l1467_146787


namespace NUMINAMATH_CALUDE_unique_solution_for_specific_k_and_a_l1467_146725

/-- The equation (x + 2) / (kx - ax - 1) = x has exactly one solution when k = 0 and a = 1/2 -/
theorem unique_solution_for_specific_k_and_a :
  ∃! x : ℝ, (x + 2) / (0 * x - (1/2) * x - 1) = x :=
sorry

end NUMINAMATH_CALUDE_unique_solution_for_specific_k_and_a_l1467_146725


namespace NUMINAMATH_CALUDE_not_right_triangle_l1467_146772

theorem not_right_triangle (a b c : ℚ) (ha : a = 2/3) (hb : b = 2) (hc : c = 5/4) :
  ¬(a^2 + b^2 = c^2) := by sorry

end NUMINAMATH_CALUDE_not_right_triangle_l1467_146772


namespace NUMINAMATH_CALUDE_cookies_left_to_take_home_l1467_146753

-- Define the initial number of cookies
def initial_cookies : ℕ := 120

-- Define the number of cookies in a dozen
def cookies_per_dozen : ℕ := 12

-- Define the number of dozens sold in the morning
def morning_dozens_sold : ℕ := 3

-- Define the number of cookies sold during lunch
def lunch_cookies_sold : ℕ := 57

-- Define the number of cookies sold in the afternoon
def afternoon_cookies_sold : ℕ := 16

-- Theorem statement
theorem cookies_left_to_take_home :
  initial_cookies - (morning_dozens_sold * cookies_per_dozen + lunch_cookies_sold + afternoon_cookies_sold) = 11 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_to_take_home_l1467_146753


namespace NUMINAMATH_CALUDE_range_of_g_l1467_146785

def f (x : ℝ) : ℝ := 4 * x + 1

def g (x : ℝ) : ℝ := f (f (f x))

theorem range_of_g :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 →
  ∃ y : ℝ, g y = x ∧ -43 ≤ x ∧ x ≤ 213 :=
by sorry

end NUMINAMATH_CALUDE_range_of_g_l1467_146785


namespace NUMINAMATH_CALUDE_hare_leaps_per_dog_leap_is_two_l1467_146734

/-- The number of hare leaps equal to one dog leap -/
def hare_leaps_per_dog_leap : ℕ := 2

/-- The number of dog leaps for a given number of hare leaps -/
def dog_leaps (hare_leaps : ℕ) : ℕ := (5 * hare_leaps : ℕ)

/-- The ratio of dog speed to hare speed -/
def speed_ratio : ℕ := 10

theorem hare_leaps_per_dog_leap_is_two :
  hare_leaps_per_dog_leap = 2 ∧
  (∀ h : ℕ, dog_leaps h = 5 * h) ∧
  speed_ratio = 10 := by
  sorry

end NUMINAMATH_CALUDE_hare_leaps_per_dog_leap_is_two_l1467_146734


namespace NUMINAMATH_CALUDE_segment_combination_uniqueness_l1467_146735

theorem segment_combination_uniqueness :
  ∃! (x y : ℕ), 7 * x + 12 * y = 100 :=
by sorry

end NUMINAMATH_CALUDE_segment_combination_uniqueness_l1467_146735


namespace NUMINAMATH_CALUDE_library_shelves_l1467_146738

/-- The number of type C shelves in a library with given conditions -/
theorem library_shelves (total_books : ℕ) (books_per_a : ℕ) (books_per_b : ℕ) (books_per_c : ℕ)
  (percent_a : ℚ) (percent_b : ℚ) (percent_c : ℚ) :
  total_books = 200000 →
  books_per_a = 12 →
  books_per_b = 15 →
  books_per_c = 20 →
  percent_a = 2/5 →
  percent_b = 7/20 →
  percent_c = 1/4 →
  percent_a + percent_b + percent_c = 1 →
  ∃ (shelves_a shelves_b : ℕ),
    ↑shelves_a * books_per_a ≥ ↑total_books * percent_a ∧
    ↑shelves_b * books_per_b ≥ ↑total_books * percent_b ∧
    2500 * books_per_c = ↑total_books * percent_c :=
by sorry


end NUMINAMATH_CALUDE_library_shelves_l1467_146738


namespace NUMINAMATH_CALUDE_tony_curl_weight_l1467_146742

/-- Represents the weight Tony can lift in different exercises --/
structure TonyLifts where
  curl : ℝ
  military_press : ℝ
  squat : ℝ

/-- Tony's lifting capabilities satisfy the given conditions --/
def valid_lifts (t : TonyLifts) : Prop :=
  t.military_press = 2 * t.curl ∧
  t.squat = 5 * t.military_press ∧
  t.squat = 900

/-- Theorem: Tony can lift 90 pounds in the curl exercise --/
theorem tony_curl_weight (t : TonyLifts) (h : valid_lifts t) : t.curl = 90 := by
  sorry

end NUMINAMATH_CALUDE_tony_curl_weight_l1467_146742


namespace NUMINAMATH_CALUDE_local_odd_function_part1_local_odd_function_part2_l1467_146728

-- Definition of local odd function
def is_local_odd_function (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∃ x ∈ domain, f (-x) = -f x

-- Part 1
theorem local_odd_function_part1 (m : ℝ) :
  is_local_odd_function (fun x => 2^x + m) (Set.Icc (-2) 2) →
  m ∈ Set.Icc (-17/8) (-1) :=
sorry

-- Part 2
theorem local_odd_function_part2 (m : ℝ) :
  is_local_odd_function (fun x => 4^x + m*2^(x+1) + m^2 - 4) Set.univ →
  m ∈ Set.Icc (-1) (Real.sqrt 10) :=
sorry

end NUMINAMATH_CALUDE_local_odd_function_part1_local_odd_function_part2_l1467_146728


namespace NUMINAMATH_CALUDE_min_value_of_linear_combination_l1467_146713

theorem min_value_of_linear_combination (x y : ℝ) : 
  3 * x^2 + 3 * y^2 = 20 * x + 10 * y + 10 → 
  5 * x + 6 * y ≥ 122 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_linear_combination_l1467_146713


namespace NUMINAMATH_CALUDE_april_rose_price_l1467_146767

/-- Calculates the price per rose given the initial number of roses, remaining roses, and total earnings. -/
def price_per_rose (initial_roses : ℕ) (remaining_roses : ℕ) (total_earnings : ℕ) : ℚ :=
  total_earnings / (initial_roses - remaining_roses)

/-- Proves that the price per rose is $4 given the problem conditions. -/
theorem april_rose_price : price_per_rose 13 4 36 = 4 := by
  sorry

end NUMINAMATH_CALUDE_april_rose_price_l1467_146767


namespace NUMINAMATH_CALUDE_hotel_assignment_count_l1467_146733

/-- Represents a hotel with a specific number of rooms and guests -/
structure Hotel :=
  (num_rooms : ℕ)
  (num_guests : ℕ)

/-- Represents the constraints for room assignments -/
structure RoomConstraints :=
  (max_guests_regular : ℕ)
  (min_guests_deluxe : ℕ)
  (max_guests_deluxe : ℕ)

/-- Calculates the number of valid room assignments -/
def count_valid_assignments (h : Hotel) (c : RoomConstraints) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem hotel_assignment_count :
  let h : Hotel := ⟨7, 7⟩
  let c : RoomConstraints := ⟨3, 2, 3⟩
  count_valid_assignments h c = 27720 :=
sorry

end NUMINAMATH_CALUDE_hotel_assignment_count_l1467_146733


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l1467_146718

theorem unique_solution_quadratic (n : ℚ) : 
  (∃! x : ℝ, (x + 5) * (x + 3) = n + 3 * x) ↔ n = 35 / 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l1467_146718


namespace NUMINAMATH_CALUDE_complex_square_l1467_146798

theorem complex_square (z : ℂ) (h : z = 5 + 6 * Complex.I) : z^2 = -11 + 60 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_l1467_146798


namespace NUMINAMATH_CALUDE_sqrt_200_equals_10_l1467_146741

theorem sqrt_200_equals_10 : Real.sqrt 200 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_200_equals_10_l1467_146741


namespace NUMINAMATH_CALUDE_binomial_expansion_property_l1467_146707

/-- Given a positive integer n, if the sum of the binomial coefficients of the first three terms
    in the expansion of (1/2 + 2x)^n equals 79, then n = 12 and the 11th term has the largest coefficient -/
theorem binomial_expansion_property (n : ℕ) (hn : n > 0) 
  (h_sum : Nat.choose n 0 + Nat.choose n 1 + Nat.choose n 2 = 79) : 
  n = 12 ∧ ∀ k, 0 ≤ k ∧ k ≤ 12 → 
    Nat.choose 12 10 * 4^10 ≥ Nat.choose 12 k * 4^k := by
  sorry


end NUMINAMATH_CALUDE_binomial_expansion_property_l1467_146707


namespace NUMINAMATH_CALUDE_renovation_project_materials_l1467_146781

theorem renovation_project_materials (sand dirt cement : ℝ) 
  (h_sand : sand = 0.17)
  (h_dirt : dirt = 0.33)
  (h_cement : cement = 0.17) :
  sand + dirt + cement = 0.67 := by
  sorry

end NUMINAMATH_CALUDE_renovation_project_materials_l1467_146781


namespace NUMINAMATH_CALUDE_school_attendance_l1467_146708

/-- Calculates the number of years a student attends school given the cost per semester,
    number of semesters per year, and total cost. -/
def years_of_school (cost_per_semester : ℕ) (semesters_per_year : ℕ) (total_cost : ℕ) : ℕ :=
  total_cost / (cost_per_semester * semesters_per_year)

/-- Theorem stating that given the specific costs and duration, the student attends 13 years of school. -/
theorem school_attendance : years_of_school 20000 2 520000 = 13 := by
  sorry

end NUMINAMATH_CALUDE_school_attendance_l1467_146708


namespace NUMINAMATH_CALUDE_system_solution_ratio_l1467_146794

theorem system_solution_ratio (x y a b : ℝ) (h1 : 4*x - 3*y = a) (h2 : 6*y - 8*x = b) (h3 : b ≠ 0) :
  a / b = -1 / 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l1467_146794


namespace NUMINAMATH_CALUDE_cooler_capacity_increase_l1467_146791

/-- Given three coolers with specific capacity relationships, prove the percentage increase from the first to the second cooler --/
theorem cooler_capacity_increase (a b c : ℝ) : 
  a = 100 → 
  b > a → 
  c = b / 2 → 
  a + b + c = 325 → 
  (b - a) / a * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_cooler_capacity_increase_l1467_146791


namespace NUMINAMATH_CALUDE_square_plus_inverse_square_equals_six_l1467_146771

theorem square_plus_inverse_square_equals_six (m : ℝ) (h : m^2 - 2*m - 1 = 0) : 
  m^2 + 1/m^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_inverse_square_equals_six_l1467_146771


namespace NUMINAMATH_CALUDE_inequality_holds_l1467_146722

theorem inequality_holds (a : ℝ) : (∀ x > 1, x^2 + a*x - 6 > 0) ↔ a ≥ 5 := by sorry

end NUMINAMATH_CALUDE_inequality_holds_l1467_146722


namespace NUMINAMATH_CALUDE_scale_division_l1467_146709

/-- Represents a length in feet and inches -/
structure Length where
  feet : ℕ
  inches : ℕ

/-- Converts a Length to total inches -/
def Length.to_inches (l : Length) : ℕ := l.feet * 12 + l.inches

/-- Converts total inches to a Length -/
def inches_to_length (total_inches : ℕ) : Length :=
  { feet := total_inches / 12, inches := total_inches % 12 }

theorem scale_division (scale : Length) (parts : ℕ) 
    (h1 : scale.feet = 6 ∧ scale.inches = 8) 
    (h2 : parts = 4) : 
  inches_to_length (scale.to_inches / parts) = { feet := 1, inches := 8 } := by
sorry

end NUMINAMATH_CALUDE_scale_division_l1467_146709


namespace NUMINAMATH_CALUDE_normal_distribution_std_dev_l1467_146703

/-- Represents a normal distribution --/
structure NormalDistribution where
  mean : ℝ
  stdDev : ℝ

/-- The value that is exactly k standard deviations from the mean --/
def valueAtStdDev (d : NormalDistribution) (k : ℝ) : ℝ :=
  d.mean + k * d.stdDev

theorem normal_distribution_std_dev (d : NormalDistribution) :
  d.mean = 15 ∧ valueAtStdDev d (-2) = 12 → d.stdDev = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_std_dev_l1467_146703


namespace NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l1467_146779

/-- Prove that the number of boys who are neither happy nor sad is 5 -/
theorem boys_neither_happy_nor_sad (total_children : ℕ) (happy_children : ℕ) (sad_children : ℕ) 
  (neither_children : ℕ) (total_boys : ℕ) (total_girls : ℕ) (happy_boys : ℕ) (sad_girls : ℕ) :
  total_children = 60 →
  happy_children = 30 →
  sad_children = 10 →
  neither_children = 20 →
  total_boys = 17 →
  total_girls = 43 →
  happy_boys = 6 →
  sad_girls = 4 →
  total_children = happy_children + sad_children + neither_children →
  total_children = total_boys + total_girls →
  (total_boys - happy_boys - (sad_children - sad_girls) : ℤ) = 5 := by
sorry


end NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l1467_146779


namespace NUMINAMATH_CALUDE_inverse_function_property_l1467_146796

theorem inverse_function_property (f : ℝ → ℝ) (hf : Function.Bijective f) 
  (h : ∀ x : ℝ, f x + f (1 - x) = 2) :
  ∀ x : ℝ, Function.invFun f (x - 2) + Function.invFun f (4 - x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_property_l1467_146796


namespace NUMINAMATH_CALUDE_power_fraction_equality_l1467_146777

theorem power_fraction_equality : (2^2015 + 2^2013 + 2^2011) / (2^2015 - 2^2013 + 2^2011) = 21/13 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l1467_146777


namespace NUMINAMATH_CALUDE_green_pieces_count_l1467_146724

/-- The number of green pieces of candy in a jar, given the total number of pieces and the number of red and blue pieces. -/
def green_pieces (total red blue : ℚ) : ℚ :=
  total - red - blue

/-- Theorem: The number of green pieces is 9468 given the specified conditions. -/
theorem green_pieces_count :
  let total : ℚ := 12509.72
  let red : ℚ := 568.29
  let blue : ℚ := 2473.43
  green_pieces total red blue = 9468 := by
  sorry

end NUMINAMATH_CALUDE_green_pieces_count_l1467_146724


namespace NUMINAMATH_CALUDE_find_f_one_l1467_146730

/-- A function with the property f(x + y) = f(x) + f(y) + 7xy + 4 -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x + f y + 7 * x * y + 4

theorem find_f_one (f : ℝ → ℝ) (h1 : special_function f) (h2 : f 2 + f 5 = 125) :
  f 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_f_one_l1467_146730


namespace NUMINAMATH_CALUDE_trainees_seating_theorem_l1467_146736

/-- Represents the number of trainees and plates -/
def n : ℕ := 67

/-- Represents the number of correct seatings after rotating i positions -/
def correct_seatings (i : ℕ) : ℕ := sorry

theorem trainees_seating_theorem :
  ∃ i : ℕ, i > 0 ∧ i < n ∧ correct_seatings i ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_trainees_seating_theorem_l1467_146736


namespace NUMINAMATH_CALUDE_problem_solution_l1467_146766

def sequence_property (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n ≤ 98 → a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0) ∧
  (a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0) ∧
  (a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)

theorem problem_solution (a : ℕ → ℝ) (h : sequence_property a) (h10 : a 10 = 10) : 
  a 22 = 10 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1467_146766


namespace NUMINAMATH_CALUDE_linear_function_properties_l1467_146789

def f (x : ℝ) := -2 * x + 2

theorem linear_function_properties :
  (∃ (x y : ℝ), f x = y ∧ x > 0 ∧ y > 0) ∧  -- First quadrant
  (∃ (x y : ℝ), f x = y ∧ x < 0 ∧ y > 0) ∧  -- Second quadrant
  (∃ (x y : ℝ), f x = y ∧ x > 0 ∧ y < 0) ∧  -- Fourth quadrant
  (f 2 ≠ 0) ∧                               -- x-intercept is not at (2, 0)
  (∀ x > 0, f x < 2) ∧                      -- When x > 0, y < 2
  (∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂)          -- Function is decreasing
  := by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l1467_146789


namespace NUMINAMATH_CALUDE_sqrt_pattern_l1467_146774

-- Define the square root function
noncomputable def sqrt (x : ℝ) := Real.sqrt x

-- Define the approximation relation
def approximately_equal (x y : ℝ) := ∃ (ε : ℝ), ε > 0 ∧ |x - y| < ε

-- State the theorem
theorem sqrt_pattern :
  (sqrt 0.0625 = 0.25) →
  (approximately_equal (sqrt 0.625) 0.791) →
  (sqrt 625 = 25) →
  (sqrt 6250 = 79.1) →
  (sqrt 62500 = 250) →
  (sqrt 625000 = 791) →
  (sqrt 6.25 = 2.5) ∧ (approximately_equal (sqrt 62.5) 7.91) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_pattern_l1467_146774


namespace NUMINAMATH_CALUDE_sum_y_invariant_under_rotation_l1467_146731

/-- A rectangle in 2D space -/
structure Rectangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  is_opposite : v1 ≠ v2

/-- The sum of y-coordinates of two points -/
def sum_y (p1 p2 : ℝ × ℝ) : ℝ := p1.2 + p2.2

/-- Theorem: The sum of y-coordinates of the other two vertices of a rectangle
    remains unchanged after a 90-degree rotation around its center -/
theorem sum_y_invariant_under_rotation (r : Rectangle) 
    (h1 : r.v1 = (5, 20))
    (h2 : r.v2 = (11, -8)) :
    ∃ (v3 v4 : ℝ × ℝ), sum_y v3 v4 = 12 ∧ 
    (∀ (v3' v4' : ℝ × ℝ), sum_y v3' v4' = 12) :=
  sorry

#check sum_y_invariant_under_rotation

end NUMINAMATH_CALUDE_sum_y_invariant_under_rotation_l1467_146731


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l1467_146737

-- Define the equations
def equation1 (x : ℝ) : Prop := 6 * x - 7 = 4 * x - 5
def equation2 (x : ℝ) : Prop := (x + 1) / 2 - 1 = 2 + (2 - x) / 4

-- Theorem for equation 1
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = 1 := by
  sorry

-- Theorem for equation 2
theorem solution_equation2 : ∃ x : ℝ, equation2 x ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l1467_146737


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l1467_146729

theorem quadratic_roots_condition (p q : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    x₁^2 + p*x₁ + q = 0 ∧ 
    x₂^2 + p*x₂ + q = 0 ∧
    x₁ = 2*p ∧ 
    x₂ = p + q) →
  p = 2/3 ∧ q = -8/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l1467_146729


namespace NUMINAMATH_CALUDE_quadratic_unique_root_l1467_146743

-- Define the arithmetic sequence property
def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  ∃ d : ℝ, c = a - d ∧ b = a - 2*d

-- Define the quadratic function
def quadratic (a c b : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + c * x + b

-- Statement of the theorem
theorem quadratic_unique_root (a b c : ℝ) :
  is_arithmetic_sequence a b c →
  a ≥ c →
  c ≥ b →
  b ≥ 0 →
  (∃! x : ℝ, quadratic a c b x = 0) →
  (∃ x : ℝ, quadratic a c b x = 0 ∧ x = -2 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_unique_root_l1467_146743


namespace NUMINAMATH_CALUDE_giant_kite_area_l1467_146760

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a kite given its four vertices -/
def kiteArea (p1 p2 p3 p4 : Point) : ℝ :=
  let base := p3.x - p1.x
  let height := p2.y - p1.y
  base * height

/-- Theorem: The area of the specified kite is 72 square inches -/
theorem giant_kite_area :
  let p1 : Point := ⟨2, 12⟩
  let p2 : Point := ⟨8, 18⟩
  let p3 : Point := ⟨14, 12⟩
  let p4 : Point := ⟨8, 2⟩
  kiteArea p1 p2 p3 p4 = 72 := by
  sorry

end NUMINAMATH_CALUDE_giant_kite_area_l1467_146760


namespace NUMINAMATH_CALUDE_range_of_a_l1467_146755

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, StrictMono (fun x => (3 - 2*a)^x)
def q (a : ℝ) : Prop := ∀ x : ℝ, 0 < x^2 + 2*a*x + 4

-- Define the theorem
theorem range_of_a (a : ℝ) 
  (h1 : p a ∨ q a) 
  (h2 : ¬(p a ∧ q a)) : 
  a ≤ -2 ∨ (1 ≤ a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1467_146755


namespace NUMINAMATH_CALUDE_total_spent_candy_and_chocolate_l1467_146715

/-- The total amount spent on a candy bar and chocolate -/
def total_spent (candy_bar_cost chocolate_cost : ℕ) : ℕ :=
  candy_bar_cost + chocolate_cost

/-- Theorem: The total amount spent on a candy bar costing $7 and chocolate costing $6 is $13 -/
theorem total_spent_candy_and_chocolate :
  total_spent 7 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_candy_and_chocolate_l1467_146715


namespace NUMINAMATH_CALUDE_rudolph_travel_distance_l1467_146764

/-- Represents the number of stop signs Rudolph encountered -/
def total_stop_signs : ℕ := 17 - 3

/-- Represents the number of stop signs per mile -/
def stop_signs_per_mile : ℕ := 2

/-- Calculates the number of miles Rudolph traveled -/
def miles_traveled : ℚ := total_stop_signs / stop_signs_per_mile

theorem rudolph_travel_distance :
  miles_traveled = 7 := by sorry

end NUMINAMATH_CALUDE_rudolph_travel_distance_l1467_146764


namespace NUMINAMATH_CALUDE_sequence_arrangements_l1467_146720

-- Define a type for our sequence
def Sequence := Fin 5 → Fin 5

-- Define a predicate for valid permutations
def is_valid_permutation (s : Sequence) : Prop :=
  Function.Injective s ∧ Function.Surjective s

-- Define a predicate for non-adjacent odd and even numbers
def non_adjacent_odd_even (s : Sequence) : Prop :=
  ∀ i : Fin 4, (s i).val % 2 ≠ (s (i + 1)).val % 2

-- Define a predicate for decreasing then increasing sequence
def decreasing_then_increasing (s : Sequence) : Prop :=
  ∃ j : Fin 4, (∀ i : Fin 5, i < j → s i > s (i + 1)) ∧
               (∀ i : Fin 5, i ≥ j → s i < s (i + 1))

-- Define a predicate for the specific inequality condition
def specific_inequality (s : Sequence) : Prop :=
  s 0 < s 1 ∧ s 1 > s 2 ∧ s 2 > s 3 ∧ s 3 < s 4

-- State the theorem
theorem sequence_arrangements (s : Sequence) 
  (h : is_valid_permutation s) : 
  (∃ l : List Sequence, (∀ s' ∈ l, is_valid_permutation s' ∧ non_adjacent_odd_even s') ∧ l.length = 12) ∧
  (∃ l : List Sequence, (∀ s' ∈ l, is_valid_permutation s' ∧ decreasing_then_increasing s') ∧ l.length = 14) ∧
  (∃ l : List Sequence, (∀ s' ∈ l, is_valid_permutation s' ∧ specific_inequality s') ∧ l.length = 11) :=
sorry

end NUMINAMATH_CALUDE_sequence_arrangements_l1467_146720


namespace NUMINAMATH_CALUDE_factorization_of_polynomial_l1467_146719

theorem factorization_of_polynomial (x : ℝ) : 
  x^4 - 3*x^3 - 28*x^2 = x^2 * (x - 7) * (x + 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_polynomial_l1467_146719


namespace NUMINAMATH_CALUDE_four_digit_multiples_of_seven_l1467_146759

theorem four_digit_multiples_of_seven : 
  (Finset.filter (fun n => n % 7 = 0) (Finset.range 9000)).card = 1286 :=
by
  sorry


end NUMINAMATH_CALUDE_four_digit_multiples_of_seven_l1467_146759


namespace NUMINAMATH_CALUDE_number_relationship_l1467_146740

theorem number_relationship (s l : ℕ) : 
  s + l = 124 → s = 31 → l = s + 62 := by
  sorry

end NUMINAMATH_CALUDE_number_relationship_l1467_146740


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l1467_146748

/-- The eccentricity of a hyperbola given its equation and a point in the "up" region -/
theorem hyperbola_eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_up : b / a < 2) : ∃ e : ℝ, 1 < e ∧ e < Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l1467_146748


namespace NUMINAMATH_CALUDE_quadratic_root_m_value_l1467_146750

theorem quadratic_root_m_value :
  ∀ m : ℝ, (1 : ℝ)^2 + m * 1 + 2 = 0 → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_m_value_l1467_146750


namespace NUMINAMATH_CALUDE_class_size_l1467_146770

theorem class_size (female_students : ℕ) (male_students : ℕ) : 
  female_students = 13 → 
  male_students = 3 * female_students → 
  female_students + male_students = 52 := by
sorry

end NUMINAMATH_CALUDE_class_size_l1467_146770


namespace NUMINAMATH_CALUDE_kathleen_savings_problem_l1467_146705

/-- Kathleen's savings and expenses problem -/
theorem kathleen_savings_problem (june july august september : ℚ)
  (school_supplies clothes gift book donation : ℚ) :
  june = 21 →
  july = 46 →
  august = 45 →
  september = 32 →
  school_supplies = 12 →
  clothes = 54 →
  gift = 37 →
  book = 25 →
  donation = 10 →
  let october : ℚ := august / 2
  let november : ℚ := 2 * september - 20
  let total_savings : ℚ := june + july + august + september + october + november
  let total_expenses : ℚ := school_supplies + clothes + gift + book + donation
  let aunt_bonus : ℚ := if total_savings > 200 ∧ donation = 10 then 25 else 0
  total_savings - total_expenses + aunt_bonus = 97.5 := by
  sorry

end NUMINAMATH_CALUDE_kathleen_savings_problem_l1467_146705


namespace NUMINAMATH_CALUDE_range_of_a_for_sqrt_function_l1467_146757

theorem range_of_a_for_sqrt_function (a : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt (2^x - a)) → a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_sqrt_function_l1467_146757


namespace NUMINAMATH_CALUDE_parabola_equation_l1467_146792

/-- Represents a parabola with vertex at the origin and focus on the x-axis -/
structure Parabola where
  a : ℝ
  eq : ∀ x y : ℝ, y^2 = a * x

/-- Represents a line in the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ
  eq : ∀ x y : ℝ, y = m * x + b

/-- The chord length of a parabola intercepted by a line -/
def chordLength (p : Parabola) (l : Line) : ℝ := sorry

theorem parabola_equation (p : Parabola) (l : Line) :
  l.m = 2 ∧ l.b = -4 ∧ chordLength p l = 3 * Real.sqrt 5 →
  p.a = 4 ∨ p.a = -36 := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1467_146792


namespace NUMINAMATH_CALUDE_cousin_distribution_l1467_146711

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers --/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 5 cousins and 5 rooms --/
def cousins : ℕ := 5
def rooms : ℕ := 5

/-- The main theorem: there are 52 ways to distribute the cousins into the rooms --/
theorem cousin_distribution : distribute cousins rooms = 52 := by sorry

end NUMINAMATH_CALUDE_cousin_distribution_l1467_146711


namespace NUMINAMATH_CALUDE_sum_of_digits_inequality_l1467_146793

/-- Sum of digits function -/
def sum_of_digits (n : ℕ+) : ℕ :=
  sorry

/-- Theorem: For any positive integer n, s(n) ≤ 8 * s(8n) -/
theorem sum_of_digits_inequality (n : ℕ+) : sum_of_digits n ≤ 8 * sum_of_digits (8 * n) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_inequality_l1467_146793


namespace NUMINAMATH_CALUDE_tan_thirteen_pi_fourths_l1467_146797

theorem tan_thirteen_pi_fourths : Real.tan (13 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_thirteen_pi_fourths_l1467_146797


namespace NUMINAMATH_CALUDE_fraction_equality_l1467_146765

theorem fraction_equality (a b : ℝ) (h : a / (a + b) = 3 / 4) : a / b = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1467_146765


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_l1467_146762

theorem quadratic_two_real_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 6 * x - 1 = 0 ∧ k * y^2 - 6 * y - 1 = 0) ↔
  (k ≥ -9 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_l1467_146762


namespace NUMINAMATH_CALUDE_exists_six_digit_number_with_digit_sum_43_l1467_146702

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem exists_six_digit_number_with_digit_sum_43 :
  ∃ n : ℕ, n < 500000 ∧ n ≥ 100000 ∧ sum_of_digits n = 43 :=
by
  sorry

end NUMINAMATH_CALUDE_exists_six_digit_number_with_digit_sum_43_l1467_146702


namespace NUMINAMATH_CALUDE_equal_vector_sums_implies_equilateral_or_equal_l1467_146704

-- Define the circle and points
def Circle := {p : ℂ | ∃ r : ℝ, r > 0 ∧ Complex.abs p = r}

-- Define the property of equal vector sums
def EqualVectorSums (A B C : ℂ) : Prop :=
  Complex.abs (A + B) = Complex.abs (B + C) ∧ 
  Complex.abs (B + C) = Complex.abs (C + A)

-- Define an equilateral triangle
def IsEquilateralTriangle (A B C : ℂ) : Prop :=
  Complex.abs (A - B) = Complex.abs (B - C) ∧
  Complex.abs (B - C) = Complex.abs (C - A)

-- State the theorem
theorem equal_vector_sums_implies_equilateral_or_equal 
  (A B C : ℂ) (hA : A ∈ Circle) (hB : B ∈ Circle) (hC : C ∈ Circle) 
  (hEqual : EqualVectorSums A B C) :
  A = B ∧ B = C ∨ IsEquilateralTriangle A B C := by
  sorry

end NUMINAMATH_CALUDE_equal_vector_sums_implies_equilateral_or_equal_l1467_146704


namespace NUMINAMATH_CALUDE_remaining_area_formula_l1467_146746

/-- The area of a rectangular field with dimensions (x + 8) and (x + 6), 
    excluding a rectangular patch with dimensions (2x - 4) and (x - 3) -/
def remaining_area (x : ℝ) : ℝ :=
  (x + 8) * (x + 6) - (2*x - 4) * (x - 3)

/-- Theorem stating that the remaining area is equal to -x^2 + 24x + 36 -/
theorem remaining_area_formula (x : ℝ) : 
  remaining_area x = -x^2 + 24*x + 36 := by
  sorry

end NUMINAMATH_CALUDE_remaining_area_formula_l1467_146746


namespace NUMINAMATH_CALUDE_inequality_proof_l1467_146712

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a / (a^2 + 2)) + (b / (b^2 + 2)) + (c / (c^2 + 2)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1467_146712


namespace NUMINAMATH_CALUDE_gcf_of_75_and_90_l1467_146773

theorem gcf_of_75_and_90 : Nat.gcd 75 90 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_75_and_90_l1467_146773


namespace NUMINAMATH_CALUDE_rectangle_area_l1467_146775

/-- Given a rectangle with diagonal length y and length three times its width, 
    prove that its area is 3y²/10 -/
theorem rectangle_area (y : ℝ) (h : y > 0) : 
  ∃ w l : ℝ, w > 0 ∧ l > 0 ∧ l = 3 * w ∧ y^2 = l^2 + w^2 ∧ w * l = (3 * y^2) / 10 := by
  sorry

#check rectangle_area

end NUMINAMATH_CALUDE_rectangle_area_l1467_146775


namespace NUMINAMATH_CALUDE_system_solution_l1467_146747

theorem system_solution : ∃ (x y z : ℝ), 
  (x^2 + y - 2*z = -3) ∧ 
  (3*x + y + z^2 = 14) ∧ 
  (7*x - y^2 + 4*z = 25) ∧
  (x = 2 ∧ y = -1 ∧ z = 3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1467_146747
