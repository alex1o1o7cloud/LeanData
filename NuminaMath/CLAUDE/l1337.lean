import Mathlib

namespace NUMINAMATH_CALUDE_square_of_85_l1337_133767

theorem square_of_85 : (85 : ℕ)^2 = 7225 := by
  sorry

end NUMINAMATH_CALUDE_square_of_85_l1337_133767


namespace NUMINAMATH_CALUDE_f_periodicity_and_smallest_a_l1337_133714

def is_valid_f (f : ℕ+ → ℝ) (a : ℕ+) : Prop :=
  f a = f 1995 ∧
  f (a + 1) = f 1996 ∧
  f (a + 2) = f 1997 ∧
  ∀ n : ℕ+, f (n + a) = (f n - 1) / (f n + 1)

theorem f_periodicity_and_smallest_a :
  ∃ (f : ℕ+ → ℝ) (a : ℕ+),
    is_valid_f f a ∧
    (∀ n : ℕ+, f (n + 4 * a) = f n) ∧
    (∀ a' : ℕ+, a' < a → ¬ is_valid_f f a') :=
  sorry

end NUMINAMATH_CALUDE_f_periodicity_and_smallest_a_l1337_133714


namespace NUMINAMATH_CALUDE_intersection_point_m_value_l1337_133712

theorem intersection_point_m_value : 
  ∀ (x y m : ℝ),
  (3 * x + y = m) →
  (-0.5 * x + y = 20) →
  (x = -6.7) →
  (m = -3.45) := by
sorry

end NUMINAMATH_CALUDE_intersection_point_m_value_l1337_133712


namespace NUMINAMATH_CALUDE_prob_product_div_by_3_l1337_133716

/-- The number of sides on a standard die -/
def num_sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The probability of rolling a number not divisible by 3 on one die -/
def prob_not_div_by_3 : ℚ := 2/3

/-- The probability that the product of the numbers rolled on 5 dice is divisible by 3 -/
theorem prob_product_div_by_3 : 
  (1 - prob_not_div_by_3 ^ num_dice) = 211/243 := by sorry

end NUMINAMATH_CALUDE_prob_product_div_by_3_l1337_133716


namespace NUMINAMATH_CALUDE_system_solution_set_system_solutions_l1337_133768

def system_has_solution (a : ℝ) : Prop :=
  ∃ x y : ℝ, (x - 4 = a * (y^3 - 2)) ∧ (2 * x / (|y^3| + y^3) = Real.sqrt x)

theorem system_solution_set :
  {a : ℝ | system_has_solution a} = Set.Ioi 2 ∪ Set.Iic 0 :=
sorry

theorem system_solutions (a : ℝ) (h : system_has_solution a) :
  (∃ x y : ℝ, x = 4 ∧ y^3 = 2) ∨
  (∃ x y : ℝ, x = 0 ∧ y^3 = 2*a - 4) :=
sorry

end NUMINAMATH_CALUDE_system_solution_set_system_solutions_l1337_133768


namespace NUMINAMATH_CALUDE_least_n_satisfying_inequality_l1337_133703

theorem least_n_satisfying_inequality : 
  ∃ (n : ℕ), n > 0 ∧ (∀ (k : ℕ), k > 0 → (1 : ℚ) / k - (1 : ℚ) / (k + 1) < (1 : ℚ) / 20 → k ≥ n) ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_least_n_satisfying_inequality_l1337_133703


namespace NUMINAMATH_CALUDE_attendance_difference_l1337_133737

/-- Calculates the total attendance for a week of football games given the conditions. -/
def totalAttendance (saturdayAttendance : ℕ) (expectedTotal : ℕ) : ℕ :=
  let mondayAttendance := saturdayAttendance - 20
  let wednesdayAttendance := mondayAttendance + 50
  let fridayAttendance := saturdayAttendance + mondayAttendance
  saturdayAttendance + mondayAttendance + wednesdayAttendance + fridayAttendance

/-- Theorem stating that the actual attendance exceeds the expected attendance by 40 people. -/
theorem attendance_difference (saturdayAttendance : ℕ) (expectedTotal : ℕ) 
  (h1 : saturdayAttendance = 80) 
  (h2 : expectedTotal = 350) : 
  totalAttendance saturdayAttendance expectedTotal - expectedTotal = 40 := by
  sorry

#eval totalAttendance 80 350 - 350

end NUMINAMATH_CALUDE_attendance_difference_l1337_133737


namespace NUMINAMATH_CALUDE_select_four_shoes_with_one_match_l1337_133774

/-- The number of ways to select four shoes from four different pairs, such that exactly one pair matches. -/
def selectFourShoesWithOneMatch : ℕ := 48

/-- The number of different pairs of shoes. -/
def numPairs : ℕ := 4

/-- The number of shoes to be selected. -/
def shoesToSelect : ℕ := 4

theorem select_four_shoes_with_one_match :
  selectFourShoesWithOneMatch = 
    numPairs * (Nat.choose (numPairs - 1) 2) * 2^2 := by
  sorry

end NUMINAMATH_CALUDE_select_four_shoes_with_one_match_l1337_133774


namespace NUMINAMATH_CALUDE_systematic_sampling_probabilities_l1337_133743

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  population : ℕ
  sample_size : ℕ
  removed : ℕ
  (population_positive : population > 0)
  (sample_size_le_population : sample_size ≤ population)
  (removed_le_population : removed ≤ population)

/-- The probability of an individual being removed in a systematic sampling scenario -/
def prob_removed (s : SystematicSampling) : ℚ :=
  s.removed / s.population

/-- The probability of an individual being sampled in a systematic sampling scenario -/
def prob_sampled (s : SystematicSampling) : ℚ :=
  s.sample_size / s.population

/-- Theorem stating the probabilities for the given systematic sampling scenario -/
theorem systematic_sampling_probabilities :
  let s : SystematicSampling :=
    { population := 1003
    , sample_size := 50
    , removed := 3
    , population_positive := by norm_num
    , sample_size_le_population := by norm_num
    , removed_le_population := by norm_num }
  prob_removed s = 3 / 1003 ∧ prob_sampled s = 50 / 1003 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_probabilities_l1337_133743


namespace NUMINAMATH_CALUDE_triangle_5_7_14_not_exists_l1337_133758

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that determines if a triangle with given side lengths can exist. -/
def triangle_exists (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem stating that a triangle with side lengths 5, 7, and 14 cannot exist. -/
theorem triangle_5_7_14_not_exists : ¬ triangle_exists 5 7 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_5_7_14_not_exists_l1337_133758


namespace NUMINAMATH_CALUDE_jeff_running_schedule_l1337_133706

/-- Jeff's running schedule problem -/
theorem jeff_running_schedule (x : ℕ) : 
  (3 * x + (x - 20) + (x + 10) = 290) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_jeff_running_schedule_l1337_133706


namespace NUMINAMATH_CALUDE_saline_solution_concentration_l1337_133702

theorem saline_solution_concentration (x : ℝ) : 
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (a * x / 100 = (a + b) * 20 / 100) ∧ 
    ((a + b) * (1 - 20 / 100) = (a + 2*b) * (1 - 30 / 100))) →
  x = 70 / 3 := by
  sorry

end NUMINAMATH_CALUDE_saline_solution_concentration_l1337_133702


namespace NUMINAMATH_CALUDE_rhombus_side_length_l1337_133783

/-- A rhombus with area K and one diagonal three times the length of the other has side length √(5K/3). -/
theorem rhombus_side_length (K : ℝ) (d₁ d₂ s : ℝ) (h₁ : K > 0) (h₂ : d₁ > 0) (h₃ : d₂ > 0) (h₄ : s > 0) :
  d₂ = 3 * d₁ →
  K = (1/2) * d₁ * d₂ →
  s^2 = (d₁/2)^2 + (d₂/2)^2 →
  s = Real.sqrt ((5 * K) / 3) :=
by sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l1337_133783


namespace NUMINAMATH_CALUDE_spurs_basketballs_l1337_133779

/-- The total number of basketballs for a team -/
def total_basketballs (num_players : ℕ) (balls_per_player : ℕ) : ℕ :=
  num_players * balls_per_player

/-- Theorem: A team of 22 players, each with 11 basketballs, has 242 basketballs in total -/
theorem spurs_basketballs : total_basketballs 22 11 = 242 := by
  sorry

end NUMINAMATH_CALUDE_spurs_basketballs_l1337_133779


namespace NUMINAMATH_CALUDE_root_sum_absolute_value_l1337_133739

theorem root_sum_absolute_value (n : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 2027*x + n = 0 ↔ x = p ∨ x = q ∨ x = r) →
  |p| + |q| + |r| = 98 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_absolute_value_l1337_133739


namespace NUMINAMATH_CALUDE_tan_is_periodic_l1337_133790

-- Define the tangent function
noncomputable def tan (x : ℝ) : ℝ := Real.tan x

-- Define the property of being periodic
def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

-- State the theorem
theorem tan_is_periodic : is_periodic tan π := by
  sorry

end NUMINAMATH_CALUDE_tan_is_periodic_l1337_133790


namespace NUMINAMATH_CALUDE_cake_frosting_time_difference_l1337_133725

/-- The time difference for frosting 10 cakes between normal and sprained wrist conditions -/
theorem cake_frosting_time_difference 
  (normal_time : ℕ) 
  (sprained_time : ℕ) 
  (num_cakes : ℕ) 
  (h1 : normal_time = 5)
  (h2 : sprained_time = 8)
  (h3 : num_cakes = 10) : 
  (sprained_time * num_cakes) - (normal_time * num_cakes) = 30 := by
  sorry

#check cake_frosting_time_difference

end NUMINAMATH_CALUDE_cake_frosting_time_difference_l1337_133725


namespace NUMINAMATH_CALUDE_fishing_problem_l1337_133718

theorem fishing_problem (jason ryan jeffery : ℕ) : 
  ryan = 3 * jason →
  jeffery = 2 * ryan →
  jeffery = 60 →
  jason + ryan + jeffery = 100 := by
sorry

end NUMINAMATH_CALUDE_fishing_problem_l1337_133718


namespace NUMINAMATH_CALUDE_simplify_complex_root_expression_l1337_133760

theorem simplify_complex_root_expression (x : ℝ) (h : x ≥ 0) :
  (4 * x * (11 + 4 * Real.sqrt 6)) ^ (1/6) *
  (4 * Real.sqrt (2 * x) - 2 * Real.sqrt (3 * x)) ^ (1/3) =
  (20 * x) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_root_expression_l1337_133760


namespace NUMINAMATH_CALUDE_cone_lateral_area_l1337_133778

/-- The lateral area of a cone with base radius 3 and height 4 is 15π -/
theorem cone_lateral_area : 
  let r : ℝ := 3
  let h : ℝ := 4
  let l : ℝ := Real.sqrt (r^2 + h^2)
  let S : ℝ := π * r * l
  S = 15 * π := by sorry

end NUMINAMATH_CALUDE_cone_lateral_area_l1337_133778


namespace NUMINAMATH_CALUDE_peach_cost_per_pound_l1337_133763

def initial_amount : ℚ := 20
def final_amount : ℚ := 14
def pounds_of_peaches : ℚ := 3

theorem peach_cost_per_pound :
  (initial_amount - final_amount) / pounds_of_peaches = 2 := by
  sorry

end NUMINAMATH_CALUDE_peach_cost_per_pound_l1337_133763


namespace NUMINAMATH_CALUDE_complex_point_on_real_axis_l1337_133764

theorem complex_point_on_real_axis (a : ℝ) : 
  (Complex.I + 1) * (Complex.I + a) ∈ Set.range Complex.ofReal → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_point_on_real_axis_l1337_133764


namespace NUMINAMATH_CALUDE_two_digit_numbers_equal_three_times_product_of_digits_l1337_133731

theorem two_digit_numbers_equal_three_times_product_of_digits :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n = 3 * (n / 10) * (n % 10)} = {15, 24} := by
  sorry

end NUMINAMATH_CALUDE_two_digit_numbers_equal_three_times_product_of_digits_l1337_133731


namespace NUMINAMATH_CALUDE_remainder_problem_l1337_133748

theorem remainder_problem (x y : ℕ) 
  (h1 : 1059 % x = y)
  (h2 : 1417 % x = y)
  (h3 : 2312 % x = y) :
  x - y = 15 := by sorry

end NUMINAMATH_CALUDE_remainder_problem_l1337_133748


namespace NUMINAMATH_CALUDE_rectangle_area_error_percentage_l1337_133794

/-- Given a rectangle with actual length L and width W, if the measured length is 1.06L
    and the measured width is 0.95W, then the error percentage in the calculated area is 0.7%. -/
theorem rectangle_area_error_percentage (L W : ℝ) (L_pos : L > 0) (W_pos : W > 0) :
  let measured_length := 1.06 * L
  let measured_width := 0.95 * W
  let actual_area := L * W
  let calculated_area := measured_length * measured_width
  let error_percentage := (calculated_area - actual_area) / actual_area * 100
  error_percentage = 0.7 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_error_percentage_l1337_133794


namespace NUMINAMATH_CALUDE_remaining_money_is_24_l1337_133799

/-- Given an initial amount of money, calculates the remaining amount after a series of transactions. -/
def remainingMoney (initialAmount : ℚ) : ℚ :=
  let afterIceCream := initialAmount - 5
  let afterTShirt := afterIceCream / 2
  let afterDeposit := afterTShirt * (4/5)
  afterDeposit

/-- Proves that given an initial amount of $65, the remaining money after transactions is $24. -/
theorem remaining_money_is_24 :
  remainingMoney 65 = 24 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_is_24_l1337_133799


namespace NUMINAMATH_CALUDE_monotonicity_and_extrema_of_f_l1337_133730

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x - 9

theorem monotonicity_and_extrema_of_f :
  (∀ x y, x < -1 → y < -1 → f x < f y) ∧ 
  (∀ x y, 3 < x → 3 < y → f x < f y) ∧
  (∀ x y, -1 < x → x < y → y < 3 → f x > f y) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - (-1)| → |x - (-1)| < δ → f x < f (-1)) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 3| → |x - 3| < δ → f x > f 3) ∧
  f (-1) = 6 ∧
  f 3 = -26 :=
by sorry

end NUMINAMATH_CALUDE_monotonicity_and_extrema_of_f_l1337_133730


namespace NUMINAMATH_CALUDE_inverse_sum_theorem_l1337_133761

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the inverse of f
variable (f_inv : ℝ → ℝ)

-- State that f_inv is the inverse of f
axiom inverse_relation : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- State the given condition
axiom condition : ∀ x, f (x + 1) + f (-x - 3) = 2

-- State the theorem to be proved
theorem inverse_sum_theorem : 
  ∀ x, f_inv (2009 - x) + f_inv (x - 2007) = -2 := by sorry

end NUMINAMATH_CALUDE_inverse_sum_theorem_l1337_133761


namespace NUMINAMATH_CALUDE_unique_N_value_l1337_133715

theorem unique_N_value (a b N : ℕ) (h1 : N = (a^2 + b^2) / (a*b - 1)) : N = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_N_value_l1337_133715


namespace NUMINAMATH_CALUDE_sum_of_seventh_powers_l1337_133704

/-- Given two real numbers a and b satisfying certain conditions, prove that a^7 + b^7 = 29 -/
theorem sum_of_seventh_powers (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11)
  (h_rec : ∀ n ≥ 3, a^n + b^n = (a^(n-1) + b^(n-1)) + (a^(n-2) + b^(n-2))) :
  a^7 + b^7 = 29 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seventh_powers_l1337_133704


namespace NUMINAMATH_CALUDE_coefficient_x5y3_in_expansion_l1337_133740

def binomial_expansion (a b : ℤ) (n : ℕ) : Polynomial ℤ := sorry

def coefficient_of_term (p : Polynomial ℤ) (x_power y_power : ℕ) : ℤ := sorry

theorem coefficient_x5y3_in_expansion :
  let p := binomial_expansion 2 (-3) 6
  coefficient_of_term (p - Polynomial.C (-1) * Polynomial.X ^ 6) 5 3 = 720 := by sorry

end NUMINAMATH_CALUDE_coefficient_x5y3_in_expansion_l1337_133740


namespace NUMINAMATH_CALUDE_parallelogram_area_l1337_133729

/-- The area of a parallelogram with given dimensions -/
theorem parallelogram_area (h : ℝ) (angle : ℝ) (s : ℝ) : 
  h = 30 → angle = 60 * π / 180 → s = 15 → h * s * Real.cos angle = 225 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1337_133729


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l1337_133775

theorem reciprocal_of_negative_fraction :
  ((-1 : ℚ) / 2023)⁻¹ = -2023 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l1337_133775


namespace NUMINAMATH_CALUDE_weighted_average_calculation_l1337_133735

theorem weighted_average_calculation (math_score math_weight history_score history_weight third_weight target_average : ℚ)
  (h1 : math_score = 72 / 100)
  (h2 : math_weight = 50 / 100)
  (h3 : history_score = 84 / 100)
  (h4 : history_weight = 30 / 100)
  (h5 : third_weight = 20 / 100)
  (h6 : target_average = 75 / 100)
  (h7 : math_weight + history_weight + third_weight ≤ 1) :
  ∃ (third_score fourth_weight : ℚ),
    third_score = 69 / 100 ∧
    fourth_weight = 0 ∧
    math_weight + history_weight + third_weight + fourth_weight = 1 ∧
    math_score * math_weight + history_score * history_weight + third_score * third_weight = target_average :=
by sorry

end NUMINAMATH_CALUDE_weighted_average_calculation_l1337_133735


namespace NUMINAMATH_CALUDE_negation_of_implication_intersection_l1337_133757

theorem negation_of_implication_intersection (A B : Set α) :
  ¬(∀ x, x ∈ A ∩ B → x ∈ A ∨ x ∈ B) ↔ ∃ x, x ∉ A ∩ B ∧ x ∉ A ∧ x ∉ B :=
sorry

end NUMINAMATH_CALUDE_negation_of_implication_intersection_l1337_133757


namespace NUMINAMATH_CALUDE_Q_equals_G_l1337_133749

-- Define the sets
def P : Set ℝ := {y | ∃ x, y = x^2 + 1}
def Q : Set ℝ := {y | ∃ x, y = x^2 + 1}
def E : Set ℝ := {x | ∃ y, y = x^2 + 1}
def F : Set (ℝ × ℝ) := {(x, y) | y = x^2 + 1}
def G : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem Q_equals_G : Q = G := by sorry

end NUMINAMATH_CALUDE_Q_equals_G_l1337_133749


namespace NUMINAMATH_CALUDE_a_fourth_plus_inverse_a_fourth_l1337_133728

theorem a_fourth_plus_inverse_a_fourth (a : ℝ) (h : (a + 1/a)^3 = 7) :
  a^4 + 1/a^4 = 1519/81 := by
  sorry

end NUMINAMATH_CALUDE_a_fourth_plus_inverse_a_fourth_l1337_133728


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l1337_133785

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^2023 + i^2024 + i^2025 + i^2026 = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l1337_133785


namespace NUMINAMATH_CALUDE_ones_digit_of_first_prime_in_sequence_l1337_133756

-- Define the property of being a prime number
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Define the property of being an increasing arithmetic sequence
def isIncreasingArithmeticSequence (a b c d : ℕ) : Prop :=
  b - a = c - b ∧ c - b = d - c ∧ a < b ∧ b < c ∧ c < d

-- Define the ones digit of a natural number
def onesDigit (n : ℕ) : ℕ := n % 10

theorem ones_digit_of_first_prime_in_sequence (p q r s : ℕ) :
  isPrime p → isPrime q → isPrime r → isPrime s →
  isIncreasingArithmeticSequence p q r s →
  q - p = 4 →
  p > 5 →
  onesDigit p = 9 :=
sorry

end NUMINAMATH_CALUDE_ones_digit_of_first_prime_in_sequence_l1337_133756


namespace NUMINAMATH_CALUDE_intersection_circles_power_l1337_133747

/-- Given two circles centered on the x-axis that intersect at points M(3a-b, 5) and N(9, 2a+3b), prove that a^b = 1/8 -/
theorem intersection_circles_power (a b : ℝ) : 
  (3 * a - b = 9) → (2 * a + 3 * b = -5) → a^b = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_intersection_circles_power_l1337_133747


namespace NUMINAMATH_CALUDE_min_sum_given_reciprocal_sum_l1337_133723

theorem min_sum_given_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1/a + 4/b = 2) : a + b ≥ 9/2 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_given_reciprocal_sum_l1337_133723


namespace NUMINAMATH_CALUDE_units_digit_of_2015_powers_l1337_133752

/-- The units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

/-- The property that a number ends with 5 -/
def ends_with_5 (n : ℕ) : Prop := units_digit n = 5

/-- The property that powers of numbers ending in 5 always end in 5 for exponents ≥ 1 -/
def power_ends_with_5 (n : ℕ) : Prop := 
  ends_with_5 n → ∀ k : ℕ, k ≥ 1 → ends_with_5 (n^k)

theorem units_digit_of_2015_powers : 
  ends_with_5 2015 → 
  power_ends_with_5 2015 → 
  units_digit (2015^2 + 2015^0 + 2015^1 + 2015^5) = 6 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_2015_powers_l1337_133752


namespace NUMINAMATH_CALUDE_min_value_of_objective_function_l1337_133787

-- Define the constraint region
def ConstraintRegion (x y : ℝ) : Prop :=
  3 * x + y - 6 ≥ 0 ∧ x - y - 2 ≤ 0 ∧ y - 3 ≤ 0

-- Define the objective function
def ObjectiveFunction (x y : ℝ) : ℝ := 4 * x + y

-- Theorem statement
theorem min_value_of_objective_function :
  ∀ x y : ℝ, ConstraintRegion x y →
  ∃ x₀ y₀ : ℝ, ConstraintRegion x₀ y₀ ∧
  ObjectiveFunction x₀ y₀ = 7 ∧
  ∀ x' y' : ℝ, ConstraintRegion x' y' →
  ObjectiveFunction x' y' ≥ ObjectiveFunction x₀ y₀ :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_objective_function_l1337_133787


namespace NUMINAMATH_CALUDE_smallest_n_value_l1337_133765

/-- Counts the number of factors of 5 in k! -/
def count_factors_of_5 (k : ℕ) : ℕ := sorry

theorem smallest_n_value (a b c m n : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 3000 →
  a.factorial * b.factorial * c.factorial = m * (10 ^ n) →
  ¬(10 ∣ m) →
  (∀ n' : ℕ, n' < n → ∃ m' : ℕ, a.factorial * b.factorial * c.factorial ≠ m' * (10 ^ n')) →
  n = 747 := by sorry

end NUMINAMATH_CALUDE_smallest_n_value_l1337_133765


namespace NUMINAMATH_CALUDE_five_million_squared_l1337_133766

theorem five_million_squared (five_million : ℕ) (h : five_million = 5 * 10^6) :
  five_million^2 = 25 * 10^12 := by
  sorry

end NUMINAMATH_CALUDE_five_million_squared_l1337_133766


namespace NUMINAMATH_CALUDE_power_function_property_l1337_133708

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- State the theorem
theorem power_function_property (f : ℝ → ℝ) 
  (h_power : isPowerFunction f) 
  (h_condition : f 4 = 2 * f 2) : 
  f 3 = 3 := by
sorry

end NUMINAMATH_CALUDE_power_function_property_l1337_133708


namespace NUMINAMATH_CALUDE_tower_heights_count_l1337_133745

/-- Represents the dimensions of a brick in inches -/
structure BrickDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of distinct tower heights achievable -/
def distinctTowerHeights (num_bricks : ℕ) (brick_dims : BrickDimensions) : ℕ :=
  sorry

/-- Theorem stating the number of distinct tower heights for the given problem -/
theorem tower_heights_count :
  let brick_dims : BrickDimensions := ⟨20, 10, 6⟩
  distinctTowerHeights 100 brick_dims = 701 := by
  sorry

end NUMINAMATH_CALUDE_tower_heights_count_l1337_133745


namespace NUMINAMATH_CALUDE_downstream_distance_l1337_133751

/-- Calculates the distance traveled downstream given boat speed, stream rate, and time. -/
theorem downstream_distance
  (boat_speed : ℝ)
  (stream_rate : ℝ)
  (time : ℝ)
  (h1 : boat_speed = 16)
  (h2 : stream_rate = 5)
  (h3 : time = 6) :
  boat_speed + stream_rate * time = 126 :=
by sorry

end NUMINAMATH_CALUDE_downstream_distance_l1337_133751


namespace NUMINAMATH_CALUDE_surface_area_circumscribed_sphere_l1337_133741

/-- The surface area of a sphere circumscribed about a rectangular solid -/
theorem surface_area_circumscribed_sphere
  (length width height : ℝ)
  (h_length : length = 2)
  (h_width : width = 1)
  (h_height : height = 2) :
  4 * Real.pi * ((length^2 + width^2 + height^2) / 4) = 9 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_surface_area_circumscribed_sphere_l1337_133741


namespace NUMINAMATH_CALUDE_box_tape_relation_l1337_133700

def tape_needed (long_side short_side : ℝ) (num_boxes : ℕ) : ℝ :=
  num_boxes * (long_side + 2 * short_side)

theorem box_tape_relation (L S : ℝ) :
  tape_needed L S 5 + tape_needed 40 40 2 = 540 →
  L = 60 - 2 * S :=
by
  sorry

end NUMINAMATH_CALUDE_box_tape_relation_l1337_133700


namespace NUMINAMATH_CALUDE_percentage_equality_l1337_133736

theorem percentage_equality : ∃ x : ℝ, (x / 100) * 75 = (2.5 / 100) * 450 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l1337_133736


namespace NUMINAMATH_CALUDE_number_equal_to_square_plus_opposite_l1337_133797

theorem number_equal_to_square_plus_opposite :
  ∀ x : ℝ, x = x^2 + (-x) → x = 0 ∨ x = 2 := by
sorry

end NUMINAMATH_CALUDE_number_equal_to_square_plus_opposite_l1337_133797


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l1337_133789

theorem sum_of_coefficients_equals_one 
  (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 + 2*x)^7 = a + a₁*(1-x) + a₂*(1-x)^2 + a₃*(1-x)^3 + 
                      a₄*(1-x)^4 + a₅*(1-x)^5 + a₆*(1-x)^6 + a₇*(1-x)^7) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l1337_133789


namespace NUMINAMATH_CALUDE_rectangle_circumference_sum_l1337_133798

/-- Calculates the sum of coins around the circumference of a rectangle formed by coins -/
def circumference_sum (horizontal : Nat) (vertical : Nat) (coin_value : Nat) : Nat :=
  let horizontal_edge := 2 * (horizontal - 2)
  let vertical_edge := 2 * (vertical - 2)
  let corners := 4
  (horizontal_edge + vertical_edge + corners) * coin_value

/-- Theorem stating that the sum of coins around the circumference of a 6x4 rectangle of 100-won coins is 1600 won -/
theorem rectangle_circumference_sum :
  circumference_sum 6 4 100 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circumference_sum_l1337_133798


namespace NUMINAMATH_CALUDE_four_digit_sum_l1337_133707

theorem four_digit_sum (A B C D : Nat) : 
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  A < 10 → B < 10 → C < 10 → D < 10 →
  (A + B + C) % 9 = 0 →
  (B + C + D) % 9 = 0 →
  A + B + C + D = 18 := by
sorry

end NUMINAMATH_CALUDE_four_digit_sum_l1337_133707


namespace NUMINAMATH_CALUDE_round_trip_distance_l1337_133722

/-- Calculates the total distance of a round trip journey given speeds and times -/
theorem round_trip_distance 
  (speed_to : ℝ) 
  (speed_from : ℝ) 
  (time_to : ℝ) 
  (time_from : ℝ) 
  (h1 : speed_to = 4)
  (h2 : speed_from = 3)
  (h3 : time_to = 30 / 60)
  (h4 : time_from = 40 / 60) :
  speed_to * time_to + speed_from * time_from = 4 := by
  sorry

#check round_trip_distance

end NUMINAMATH_CALUDE_round_trip_distance_l1337_133722


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l1337_133791

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∀ n : ℕ, n > 0 → n.sqrt ^ 2 = n → n % 2 = 0 → n % 3 = 0 → n % 5 = 0 → n ≥ 900 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l1337_133791


namespace NUMINAMATH_CALUDE_cubic_polynomial_property_l1337_133750

theorem cubic_polynomial_property (x : ℂ) (h : x^3 + x^2 + x + 1 = 0) :
  x^4 + 2*x^3 + 2*x^2 + 2*x + 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_property_l1337_133750


namespace NUMINAMATH_CALUDE_stating_first_player_strategy_l1337_133744

/-- 
Represents a game where two players fill coefficients of quadratic equations.
n is the number of equations.
-/
def QuadraticGame (n : ℕ) :=
  { rootless : ℕ // rootless ≤ n }

/-- 
The maximum number of rootless equations the first player can guarantee.
-/
def maxRootlessEquations (n : ℕ) : ℕ :=
  (n + 1) / 2

/-- 
Theorem stating that the first player can always ensure at least (n+1)/2 
equations have no roots, regardless of the second player's actions.
-/
theorem first_player_strategy (n : ℕ) :
  ∃ (strategy : QuadraticGame n), 
    (strategy.val ≥ maxRootlessEquations n) :=
sorry

end NUMINAMATH_CALUDE_stating_first_player_strategy_l1337_133744


namespace NUMINAMATH_CALUDE_isabellas_hair_length_l1337_133710

/-- Calculates the final length of Isabella's hair after a haircut -/
def hair_length_after_cut (initial_length cut_length : ℕ) : ℕ :=
  initial_length - cut_length

/-- Theorem stating that Isabella's hair length after the cut is 9 inches -/
theorem isabellas_hair_length :
  hair_length_after_cut 18 9 = 9 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_hair_length_l1337_133710


namespace NUMINAMATH_CALUDE_state_quarter_fraction_l1337_133733

theorem state_quarter_fraction :
  ∀ (total_quarters state_quarters pennsylvania_quarters : ℕ),
    total_quarters = 35 →
    pennsylvania_quarters = 7 →
    2 * pennsylvania_quarters = state_quarters →
    (state_quarters : ℚ) / total_quarters = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_state_quarter_fraction_l1337_133733


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1337_133771

theorem regular_polygon_sides (exterior_angle : ℝ) :
  exterior_angle = 40 →
  (∃ n : ℕ, n * exterior_angle = 360 ∧ n = 9) :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1337_133771


namespace NUMINAMATH_CALUDE_mary_number_is_14_l1337_133769

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def switch_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem mary_number_is_14 :
  ∃! x : ℕ, is_two_digit x ∧
    91 ≤ switch_digits (4 * x - 7) ∧
    switch_digits (4 * x - 7) ≤ 95 :=
by
  sorry

end NUMINAMATH_CALUDE_mary_number_is_14_l1337_133769


namespace NUMINAMATH_CALUDE_compare_negatives_l1337_133713

theorem compare_negatives : -8 > -abs (-9) := by
  sorry

end NUMINAMATH_CALUDE_compare_negatives_l1337_133713


namespace NUMINAMATH_CALUDE_largest_hexagon_angle_l1337_133784

-- Define the hexagon's properties
def is_valid_hexagon (angles : List ℕ) : Prop :=
  angles.length = 6 ∧
  angles.sum = 720 ∧
  ∃ (a d : ℕ), angles = [a, a + d, a + 2*d, a + 3*d, a + 4*d, a + 5*d] ∧
  ∀ x ∈ angles, 0 < x ∧ x < 180

-- Theorem statement
theorem largest_hexagon_angle (angles : List ℕ) :
  is_valid_hexagon angles →
  (∀ x ∈ angles, x ≤ 175) ∧
  (∃ x ∈ angles, x = 175) :=
by sorry

end NUMINAMATH_CALUDE_largest_hexagon_angle_l1337_133784


namespace NUMINAMATH_CALUDE_f_properties_l1337_133795

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2

-- State the theorem
theorem f_properties :
  (∀ x y, x < y ∧ ((x < 0 ∧ y ≤ 0) ∨ (x ≥ 2 ∧ y > 2)) → f x < f y) ∧
  (∃ δ₁ > 0, ∀ x, 0 < |x| ∧ |x| < δ₁ → f x < f 0) ∧
  (∃ δ₂ > 0, ∀ x, 0 < |x - 2| ∧ |x - 2| < δ₂ → f x > f 2) :=
by sorry


end NUMINAMATH_CALUDE_f_properties_l1337_133795


namespace NUMINAMATH_CALUDE_exists_solution_with_y_seven_l1337_133772

theorem exists_solution_with_y_seven :
  ∃ (x y z t : ℕ), 
    x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 ∧
    x + y + z + t = 10 ∧
    y = 7 := by
  sorry

end NUMINAMATH_CALUDE_exists_solution_with_y_seven_l1337_133772


namespace NUMINAMATH_CALUDE_cost_calculation_l1337_133746

-- Define the number of caramel apples and ice cream cones
def caramel_apples : ℕ := 3
def ice_cream_cones : ℕ := 4

-- Define the price difference between a caramel apple and an ice cream cone
def price_difference : ℚ := 25 / 100

-- Define the total amount spent
def total_spent : ℚ := 2

-- Define the cost of an ice cream cone
def ice_cream_cost : ℚ := 125 / 700

-- Define the cost of a caramel apple
def caramel_apple_cost : ℚ := ice_cream_cost + price_difference

-- Theorem statement
theorem cost_calculation :
  (caramel_apples : ℚ) * caramel_apple_cost + (ice_cream_cones : ℚ) * ice_cream_cost = total_spent :=
sorry

end NUMINAMATH_CALUDE_cost_calculation_l1337_133746


namespace NUMINAMATH_CALUDE_equation_solutions_l1337_133711

theorem equation_solutions : ∃ (x₁ x₂ : ℝ), 
  (x₁ = -Real.sqrt 2 ∧ x₁^2 + Real.sqrt 2 * x₁ - Real.sqrt 6 = Real.sqrt 3 * x₁) ∧
  (x₂ = Real.sqrt 3 ∧ x₂^2 + Real.sqrt 2 * x₂ - Real.sqrt 6 = Real.sqrt 3 * x₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1337_133711


namespace NUMINAMATH_CALUDE_square_area_increase_when_side_tripled_l1337_133721

theorem square_area_increase_when_side_tripled :
  ∀ (s : ℝ), s > 0 →
  (3 * s)^2 = 9 * s^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_increase_when_side_tripled_l1337_133721


namespace NUMINAMATH_CALUDE_solution_is_3x_l1337_133776

-- Define the interval
def I : Set ℝ := Set.Icc (-1 : ℝ) 1

-- Define the integral equation
def integral_equation (φ : ℝ → ℝ) : Prop :=
  ∀ x ∈ I, φ x = x + ∫ t in I, x * t * φ t

-- State the theorem
theorem solution_is_3x :
  ∃ φ : ℝ → ℝ, integral_equation φ ∧ (∀ x ∈ I, φ x = 3 * x) :=
sorry

end NUMINAMATH_CALUDE_solution_is_3x_l1337_133776


namespace NUMINAMATH_CALUDE_aspirin_percentage_of_max_dosage_l1337_133792

-- Define the medication schedule and dosages
def aspirin_dosage : ℕ := 325
def aspirin_frequency : ℕ := 12
def aspirin_max_dosage : ℕ := 4000
def hours_per_day : ℕ := 24

-- Define the function to calculate total daily dosage
def total_daily_dosage (dosage frequency : ℕ) : ℕ :=
  dosage * (hours_per_day / frequency)

-- Define the function to calculate percentage of max dosage
def percentage_of_max_dosage (daily_dosage max_dosage : ℕ) : ℚ :=
  (daily_dosage : ℚ) / (max_dosage : ℚ) * 100

-- Theorem statement
theorem aspirin_percentage_of_max_dosage :
  percentage_of_max_dosage 
    (total_daily_dosage aspirin_dosage aspirin_frequency) 
    aspirin_max_dosage = 16.25 := by
  sorry

end NUMINAMATH_CALUDE_aspirin_percentage_of_max_dosage_l1337_133792


namespace NUMINAMATH_CALUDE_two_numbers_sum_product_l1337_133709

theorem two_numbers_sum_product : ∃ x y : ℝ, x + y = 20 ∧ x * y = 96 ∧ ((x = 12 ∧ y = 8) ∨ (x = 8 ∧ y = 12)) := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_sum_product_l1337_133709


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l1337_133759

def off_rack_suit_price : ℝ := 300
def tailored_suit_price (off_rack_price : ℝ) : ℝ := 3 * off_rack_price + 200
def dress_shirt_price : ℝ := 80
def shoes_price : ℝ := 120
def tie_price : ℝ := 40
def discount_rate : ℝ := 0.1
def sales_tax_rate : ℝ := 0.08
def shipping_fee : ℝ := 25

def total_cost : ℝ :=
  let discounted_suit_price := off_rack_suit_price * (1 - discount_rate)
  let suits_cost := off_rack_suit_price + discounted_suit_price
  let tailored_suit_cost := tailored_suit_price off_rack_suit_price
  let accessories_cost := dress_shirt_price + shoes_price + tie_price
  let subtotal := suits_cost + tailored_suit_cost + accessories_cost
  let tax := subtotal * sales_tax_rate
  subtotal + tax + shipping_fee

theorem total_cost_is_correct : total_cost = 2087.80 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l1337_133759


namespace NUMINAMATH_CALUDE_triangle_inequality_l1337_133732

theorem triangle_inequality (A B C : Real) (h : A + B + C = π) :
  (Real.sqrt (Real.sin A * Real.sin B)) / (Real.sin (C / 2)) ≥ 3 * Real.sqrt 3 * Real.tan (A / 2) * Real.tan (B / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1337_133732


namespace NUMINAMATH_CALUDE_green_beads_count_l1337_133786

/-- The number of white beads in each necklace -/
def white_beads : ℕ := 6

/-- The number of orange beads in each necklace -/
def orange_beads : ℕ := 3

/-- The maximum number of necklaces that can be made -/
def max_necklaces : ℕ := 5

/-- The total number of beads available for each color -/
def total_beads : ℕ := 45

/-- The number of green beads in each necklace -/
def green_beads : ℕ := 9

theorem green_beads_count : 
  white_beads * max_necklaces ≤ total_beads ∧ 
  orange_beads * max_necklaces ≤ total_beads ∧ 
  green_beads * max_necklaces = total_beads := by
  sorry

end NUMINAMATH_CALUDE_green_beads_count_l1337_133786


namespace NUMINAMATH_CALUDE_probability_not_ab_l1337_133720

def num_courses : ℕ := 4
def num_selected : ℕ := 2

def probability_not_selected_together : ℚ :=
  1 - (1 / (num_courses.choose num_selected))

theorem probability_not_ab : probability_not_selected_together = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_ab_l1337_133720


namespace NUMINAMATH_CALUDE_jogging_distance_l1337_133717

/-- Alice's jogging speed in miles per minute -/
def alice_speed : ℚ := 1 / 12

/-- Bob's jogging speed in miles per minute -/
def bob_speed : ℚ := 3 / 40

/-- Total jogging time in minutes -/
def total_time : ℕ := 120

/-- The distance between Alice and Bob after jogging for the total time -/
def distance_apart : ℚ := alice_speed * total_time + bob_speed * total_time

theorem jogging_distance : distance_apart = 19 := by sorry

end NUMINAMATH_CALUDE_jogging_distance_l1337_133717


namespace NUMINAMATH_CALUDE_root_transformation_l1337_133777

theorem root_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 3*r₁^2 + 8 = 0) ∧ 
  (r₂^3 - 3*r₂^2 + 8 = 0) ∧ 
  (r₃^3 - 3*r₃^2 + 8 = 0) →
  ((3*r₁)^3 - 9*(3*r₁)^2 + 216 = 0) ∧
  ((3*r₂)^3 - 9*(3*r₂)^2 + 216 = 0) ∧
  ((3*r₃)^3 - 9*(3*r₃)^2 + 216 = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_transformation_l1337_133777


namespace NUMINAMATH_CALUDE_total_points_earned_l1337_133770

/-- The number of pounds required to earn one point -/
def pounds_per_point : ℕ := 4

/-- The number of pounds Paige recycled -/
def paige_pounds : ℕ := 14

/-- The number of pounds Paige's friends recycled -/
def friends_pounds : ℕ := 2

/-- The total number of pounds recycled -/
def total_pounds : ℕ := paige_pounds + friends_pounds

/-- The theorem stating that the total points earned is 4 -/
theorem total_points_earned : (total_pounds / pounds_per_point : ℕ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_total_points_earned_l1337_133770


namespace NUMINAMATH_CALUDE_platform_length_l1337_133705

/-- Given a train of length 300 meters that crosses a platform in 39 seconds
    and passes a signal pole in 12 seconds, prove that the platform length is 675 meters. -/
theorem platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_time = 39)
  (h3 : pole_time = 12) :
  train_length * (platform_time / pole_time - 1) = 675 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l1337_133705


namespace NUMINAMATH_CALUDE_reflection_properties_l1337_133780

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a figure as a set of points
def Figure := Set Point2D

-- Define the reflection of a point about a line
def reflect (p : Point2D) (l : Line2D) : Point2D :=
  sorry

-- Define the reflection of a figure about a line
def reflectFigure (f : Figure) (l : Line2D) : Figure :=
  sorry

-- Define a predicate to check if a point is on a specific side of a line
def onSide (p : Point2D) (l : Line2D) (side : Bool) : Prop :=
  sorry

-- Define a predicate to check if a figure is on a specific side of a line
def figureOnSide (f : Figure) (l : Line2D) (side : Bool) : Prop :=
  sorry

-- Define a predicate to check if two figures have the same shape
def sameShape (f1 f2 : Figure) : Prop :=
  sorry

-- Define a predicate to check if a figure touches a line at given points
def touchesAt (f : Figure) (l : Line2D) (p q : Point2D) : Prop :=
  sorry

theorem reflection_properties 
  (f : Figure) (l : Line2D) (p q : Point2D) (side : Bool) :
  figureOnSide f l side →
  touchesAt f l p q →
  let f' := reflectFigure f l
  figureOnSide f' l (!side) ∧
  sameShape f f' ∧
  touchesAt f' l p q :=
by
  sorry

end NUMINAMATH_CALUDE_reflection_properties_l1337_133780


namespace NUMINAMATH_CALUDE_min_value_expression_l1337_133726

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a * b + 3 = b) :
  (1 / a + 2 * b) ≥ 8 + 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1337_133726


namespace NUMINAMATH_CALUDE_lcm_of_20_45_75_l1337_133762

theorem lcm_of_20_45_75 : Nat.lcm (Nat.lcm 20 45) 75 = 900 := by sorry

end NUMINAMATH_CALUDE_lcm_of_20_45_75_l1337_133762


namespace NUMINAMATH_CALUDE_sum_due_proof_l1337_133753

/-- Represents the relationship between banker's discount, true discount, and face value -/
def discount_relation (bd td fv : ℚ) : Prop :=
  bd = td + (td * bd / fv)

/-- Proves that given a banker's discount of 36 and a true discount of 30,
    the face value (sum due) is 180 -/
theorem sum_due_proof :
  ∃ (fv : ℚ), discount_relation 36 30 fv ∧ fv = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_due_proof_l1337_133753


namespace NUMINAMATH_CALUDE_solve_candy_problem_l1337_133755

def candy_problem (total : ℕ) (snickers : ℕ) (mars : ℕ) : Prop :=
  ∃ butterfingers : ℕ, 
    total = snickers + mars + butterfingers ∧ 
    butterfingers = 7

theorem solve_candy_problem : candy_problem 12 3 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_candy_problem_l1337_133755


namespace NUMINAMATH_CALUDE_circle_area_equals_circumference_squared_l1337_133734

theorem circle_area_equals_circumference_squared : 
  ∀ (r : ℝ), r > 0 → 2 * (π * r^2 / 2) = (2 * π * r)^2 / (4 * π) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_equals_circumference_squared_l1337_133734


namespace NUMINAMATH_CALUDE_triangle_side_product_range_l1337_133793

theorem triangle_side_product_range (x y : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ y) 
  (h3 : y < x + 1) : 
  let t := max (1/x) (max (x/y) y) * min (1/x) (min (x/y) y)
  1 ≤ t ∧ t < (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_product_range_l1337_133793


namespace NUMINAMATH_CALUDE_factory_max_profit_l1337_133738

/-- The annual profit function for the factory -/
noncomputable def L (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 80 then
    -(1/3) * x^2 + 40 * x - 250
  else if x ≥ 80 then
    50 * x - 10000 / x + 1200
  else
    0

/-- The maximum profit and corresponding production level -/
theorem factory_max_profit :
  (∃ (x : ℝ), L x = 1000 ∧ x = 100) ∧
  (∀ (y : ℝ), L y ≤ 1000) := by
  sorry

end NUMINAMATH_CALUDE_factory_max_profit_l1337_133738


namespace NUMINAMATH_CALUDE_odd_function_value_at_negative_two_l1337_133773

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_value_at_negative_two
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_nonneg : ∀ x ≥ 0, f x = x * (x - 1)) :
  f (-2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_value_at_negative_two_l1337_133773


namespace NUMINAMATH_CALUDE_envelope_addressing_problem_l1337_133788

/-- A manufacturer's envelope addressing problem -/
theorem envelope_addressing_problem 
  (initial_machine : ℝ) 
  (first_added_machine : ℝ) 
  (combined_initial_and_first : ℝ) 
  (all_three_machines : ℝ) 
  (h1 : initial_machine = 600 / 10)
  (h2 : first_added_machine = 600 / 5)
  (h3 : combined_initial_and_first = 600 / 3)
  (h4 : all_three_machines = 600 / 1) :
  (600 / (all_three_machines - initial_machine - first_added_machine)) = 10 / 7 := by
  sorry

#check envelope_addressing_problem

end NUMINAMATH_CALUDE_envelope_addressing_problem_l1337_133788


namespace NUMINAMATH_CALUDE_reasoning_method_is_inductive_l1337_133701

-- Define the set of animals
inductive Animal : Type
| Ape : Animal
| Cat : Animal
| Elephant : Animal
| OtherMammal : Animal

-- Define the breathing method
inductive BreathingMethod : Type
| Lungs : BreathingMethod

-- Define the reasoning method
inductive ReasoningMethod : Type
| Inductive : ReasoningMethod
| Deductive : ReasoningMethod
| Analogical : ReasoningMethod
| CompleteInductive : ReasoningMethod

-- Define a function that represents breathing for specific animals
def breathes : Animal → BreathingMethod
| Animal.Ape => BreathingMethod.Lungs
| Animal.Cat => BreathingMethod.Lungs
| Animal.Elephant => BreathingMethod.Lungs
| Animal.OtherMammal => BreathingMethod.Lungs

-- Define a predicate for reasoning from specific to general
def reasonsFromSpecificToGeneral (method : ReasoningMethod) : Prop :=
  method = ReasoningMethod.Inductive

-- Theorem statement
theorem reasoning_method_is_inductive :
  (∀ a : Animal, breathes a = BreathingMethod.Lungs) →
  (reasonsFromSpecificToGeneral ReasoningMethod.Inductive) :=
by sorry

end NUMINAMATH_CALUDE_reasoning_method_is_inductive_l1337_133701


namespace NUMINAMATH_CALUDE_power_sum_six_l1337_133754

theorem power_sum_six (x : ℝ) (h : x + 1/x = 5) : x^6 + 1/x^6 = 12098 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_six_l1337_133754


namespace NUMINAMATH_CALUDE_sandy_molly_age_ratio_l1337_133719

/-- The ratio of ages between two people -/
def age_ratio (age1 : ℕ) (age2 : ℕ) : ℚ := age1 / age2

/-- Sandy's age -/
def sandy_age : ℕ := 49

/-- Age difference between Molly and Sandy -/
def age_difference : ℕ := 14

/-- Molly's age -/
def molly_age : ℕ := sandy_age + age_difference

theorem sandy_molly_age_ratio :
  age_ratio sandy_age molly_age = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sandy_molly_age_ratio_l1337_133719


namespace NUMINAMATH_CALUDE_deaf_to_blind_ratio_l1337_133796

theorem deaf_to_blind_ratio (total : ℕ) (deaf : ℕ) (h1 : total = 240) (h2 : deaf = 180) :
  (deaf : ℚ) / (total - deaf) = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_deaf_to_blind_ratio_l1337_133796


namespace NUMINAMATH_CALUDE_consecutive_integers_sqrt_seven_l1337_133742

theorem consecutive_integers_sqrt_seven (a b : ℤ) : 
  (b = a + 1) →  -- a and b are consecutive integers
  (a < Real.sqrt 7) →  -- a < √7
  (Real.sqrt 7 < b) →  -- √7 < b
  a + b = 5 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sqrt_seven_l1337_133742


namespace NUMINAMATH_CALUDE_sequence_formula_l1337_133727

theorem sequence_formula (n : ℕ+) (S : ℕ+ → ℝ) (a : ℕ+ → ℝ) 
  (h : ∀ k : ℕ+, S k = a k - 3) : 
  a n = 2 * 3^(n : ℝ) := by
sorry

end NUMINAMATH_CALUDE_sequence_formula_l1337_133727


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l1337_133724

theorem ceiling_floor_difference (x : ℝ) 
  (h : ⌈x⌉ - ⌊x⌋ = 2) : 
  3 * (⌈x⌉ - x) = 6 - 3 * (x - ⌊x⌋) := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l1337_133724


namespace NUMINAMATH_CALUDE_bus_distribution_solution_l1337_133782

/-- Represents the problem of distributing passengers among buses --/
structure BusDistribution where
  k : ℕ  -- Original number of buses
  n : ℕ  -- Number of passengers per bus after redistribution
  max_capacity : ℕ  -- Maximum capacity of each bus

/-- The conditions of the bus distribution problem --/
def valid_distribution (bd : BusDistribution) : Prop :=
  bd.k ≥ 2 ∧
  bd.n ≤ bd.max_capacity ∧
  22 * bd.k + 1 = bd.n * (bd.k - 1)

/-- The theorem stating the solution to the bus distribution problem --/
theorem bus_distribution_solution :
  ∃ (bd : BusDistribution),
    bd.max_capacity = 32 ∧
    valid_distribution bd ∧
    bd.k = 24 ∧
    bd.n * (bd.k - 1) = 529 :=
sorry


end NUMINAMATH_CALUDE_bus_distribution_solution_l1337_133782


namespace NUMINAMATH_CALUDE_early_arrival_time_l1337_133781

/-- Proves that a boy walking at 5/4 of his usual rate arrives 4 minutes early when his usual time is 20 minutes. -/
theorem early_arrival_time (usual_time : ℝ) (usual_rate : ℝ) (faster_rate : ℝ) :
  usual_time = 20 →
  faster_rate = (5 / 4) * usual_rate →
  usual_time - (usual_time * usual_rate / faster_rate) = 4 := by
sorry

end NUMINAMATH_CALUDE_early_arrival_time_l1337_133781
