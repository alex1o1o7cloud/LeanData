import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_perimeter_bound_l3230_323078

theorem rectangle_perimeter_bound (a b : ℝ) (h : a > 0 ∧ b > 0) 
  (area_gt_perimeter : a * b > 2 * (a + b)) : 2 * (a + b) > 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_bound_l3230_323078


namespace NUMINAMATH_CALUDE_arithmetic_sequence_50th_term_l3230_323092

/-- An arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) : ℕ → ℤ := fun n => a + (n - 1) * d

/-- The 50th term of the specific arithmetic sequence -/
def term50 : ℤ := arithmeticSequence 3 2 50

/-- Theorem: The 50th term of the arithmetic sequence with first term 3 and common difference 2 is 101 -/
theorem arithmetic_sequence_50th_term : term50 = 101 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_50th_term_l3230_323092


namespace NUMINAMATH_CALUDE_kristy_cookies_theorem_l3230_323023

/-- The number of cookies Kristy baked -/
def total_cookies : ℕ := 22

/-- The number of cookies Kristy ate -/
def cookies_eaten : ℕ := 2

/-- The number of cookies Kristy gave to her brother -/
def cookies_given_to_brother : ℕ := 1

/-- The number of cookies taken by the first friend -/
def cookies_taken_by_first_friend : ℕ := 3

/-- The number of cookies taken by the second friend -/
def cookies_taken_by_second_friend : ℕ := 5

/-- The number of cookies taken by the third friend -/
def cookies_taken_by_third_friend : ℕ := 5

/-- The number of cookies left -/
def cookies_left : ℕ := 6

/-- Theorem stating that the total number of cookies equals the sum of all distributed cookies and those left -/
theorem kristy_cookies_theorem : 
  total_cookies = 
    cookies_eaten + 
    cookies_given_to_brother + 
    cookies_taken_by_first_friend + 
    cookies_taken_by_second_friend + 
    cookies_taken_by_third_friend + 
    cookies_left :=
by
  sorry

end NUMINAMATH_CALUDE_kristy_cookies_theorem_l3230_323023


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l3230_323047

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_number_divisibility : 
  (∀ n : ℕ, n < 6303 → ¬(is_divisible (n + 3) 18 ∧ is_divisible (n + 3) 1051 ∧ is_divisible (n + 3) 100 ∧ is_divisible (n + 3) 21)) ∧
  (is_divisible (6303 + 3) 18 ∧ is_divisible (6303 + 3) 1051 ∧ is_divisible (6303 + 3) 100 ∧ is_divisible (6303 + 3) 21) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l3230_323047


namespace NUMINAMATH_CALUDE_only_prop2_and_prop3_true_l3230_323001

-- Define the propositions
def proposition1 : Prop :=
  (∀ x : ℝ, (x ≠ 1 ∨ x ≠ 2) → (x^2 - 3*x + 2 ≠ 0)) →
  (∀ x : ℝ, (x^2 - 3*x + 2 = 0) → (x = 1 ∨ x = 2))

def proposition2 : Prop :=
  ¬(∃ x : ℝ, x^2 - 2*x + 1 < 0)

def proposition3 (m : ℝ) : Prop :=
  (m = 1/2) →
  ((m + 2) * (m - 2) + 3 * m * (m + 2) = 0)

def proposition4 (m n : ℝ) : Prop :=
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ x₁^2 - m*x₁ + n = 0 ∧ x₂^2 - m*x₂ + n = 0) →
  (m > 0 ∧ n > 0)

-- Theorem stating that only propositions 2 and 3 are true
theorem only_prop2_and_prop3_true :
  ¬proposition1 ∧ proposition2 ∧ (∃ m : ℝ, proposition3 m) ∧ ¬(∀ m n : ℝ, proposition4 m n) :=
sorry

end NUMINAMATH_CALUDE_only_prop2_and_prop3_true_l3230_323001


namespace NUMINAMATH_CALUDE_only_5_12_13_is_right_triangle_l3230_323011

/-- Checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- The given sets of numbers --/
def number_sets : List (ℕ × ℕ × ℕ) :=
  [(2, 3, 4), (4, 5, 6), (5, 12, 13), (5, 6, 7)]

/-- Theorem stating that only (5, 12, 13) forms a right triangle --/
theorem only_5_12_13_is_right_triangle :
  ∃! (a b c : ℕ), (a, b, c) ∈ number_sets ∧ is_right_triangle a b c :=
by sorry

end NUMINAMATH_CALUDE_only_5_12_13_is_right_triangle_l3230_323011


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3230_323081

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(1 + b²/a²) -/
theorem hyperbola_eccentricity (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  let e := Real.sqrt (1 + b^2 / a^2)
  ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → e = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3230_323081


namespace NUMINAMATH_CALUDE_roses_in_vase_after_actions_l3230_323058

/-- Represents the number of flowers in a vase -/
structure FlowerVase where
  roses : ℕ
  orchids : ℕ

/-- Represents the actions taken by Jessica -/
structure JessicaActions where
  addedRoses : ℕ
  addedOrchids : ℕ
  cutRoses : ℕ

def initial : FlowerVase := { roses := 15, orchids := 62 }

def actions : JessicaActions := { addedRoses := 0, addedOrchids := 34, cutRoses := 2 }

def final : FlowerVase := { roses := 96, orchids := initial.orchids + actions.addedOrchids }

theorem roses_in_vase_after_actions (R : ℕ) : 
  final.roses = 13 + R ↔ actions.addedRoses = R := by sorry

end NUMINAMATH_CALUDE_roses_in_vase_after_actions_l3230_323058


namespace NUMINAMATH_CALUDE_kitchen_tile_comparison_l3230_323026

theorem kitchen_tile_comparison : 
  let area_figure1 : ℝ := π / 3 - Real.sqrt 3 / 4
  let area_figure2 : ℝ := Real.sqrt 3 / 2 - π / 6
  area_figure1 > area_figure2 := by
sorry

end NUMINAMATH_CALUDE_kitchen_tile_comparison_l3230_323026


namespace NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l3230_323064

theorem six_digit_numbers_with_zero (total_six_digit : ℕ) (no_zero_six_digit : ℕ) :
  total_six_digit = 900000 →
  no_zero_six_digit = 531441 →
  total_six_digit - no_zero_six_digit = 368559 :=
by sorry

end NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l3230_323064


namespace NUMINAMATH_CALUDE_candidate_votes_l3230_323043

theorem candidate_votes (total_votes : ℕ) (invalid_percent : ℚ) (candidate_percent : ℚ) : 
  total_votes = 560000 →
  invalid_percent = 15/100 →
  candidate_percent = 75/100 →
  ↑⌊(total_votes : ℚ) * (1 - invalid_percent) * candidate_percent⌋ = 357000 := by
sorry

end NUMINAMATH_CALUDE_candidate_votes_l3230_323043


namespace NUMINAMATH_CALUDE_fifth_individual_is_one_l3230_323090

def random_numbers : List ℕ := [65, 72, 08, 02, 63, 14, 07, 02, 43, 69, 97, 08, 01]

def is_valid (n : ℕ) : Bool := n ≥ 1 ∧ n ≤ 20

def select_individuals (numbers : List ℕ) : List ℕ :=
  numbers.filter is_valid |>.eraseDups

theorem fifth_individual_is_one :
  (select_individuals random_numbers).nthLe 4 sorry = 1 := by
  sorry

end NUMINAMATH_CALUDE_fifth_individual_is_one_l3230_323090


namespace NUMINAMATH_CALUDE_remainder_x5_plus_3_div_x_minus_3_squared_l3230_323029

open Polynomial

theorem remainder_x5_plus_3_div_x_minus_3_squared (x : ℝ) :
  ∃ q : Polynomial ℝ, X^5 + C 3 = (X - C 3)^2 * q + (C 405 * X - C 969) := by
  sorry

end NUMINAMATH_CALUDE_remainder_x5_plus_3_div_x_minus_3_squared_l3230_323029


namespace NUMINAMATH_CALUDE_smaller_special_integer_l3230_323077

/-- Two positive three-digit integers satisfying the given condition -/
def SpecialIntegers (m n : ℕ) : Prop :=
  100 ≤ m ∧ m < 1000 ∧
  100 ≤ n ∧ n < 1000 ∧
  (m + n) / 2 = m + n / 200

/-- The smaller of two integers satisfying the condition is 891 -/
theorem smaller_special_integer (m n : ℕ) (h : SpecialIntegers m n) : 
  min m n = 891 := by
  sorry

#check smaller_special_integer

end NUMINAMATH_CALUDE_smaller_special_integer_l3230_323077


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3230_323089

def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {-1, 0, 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3230_323089


namespace NUMINAMATH_CALUDE_weekly_allowance_calculation_l3230_323062

/-- Proves that if a person spends 3/5 of their allowance, then 1/3 of the remainder, 
    and finally $0.96, their original allowance was $3.60. -/
theorem weekly_allowance_calculation (A : ℝ) : 
  (A > 0) →
  ((2/5) * A - (1/3) * ((2/5) * A) = 0.96) →
  A = 3.60 := by
sorry

end NUMINAMATH_CALUDE_weekly_allowance_calculation_l3230_323062


namespace NUMINAMATH_CALUDE_value_of_M_l3230_323010

theorem value_of_M (m n p M : ℝ) 
  (h1 : M = m / (n + p))
  (h2 : M = n / (p + m))
  (h3 : M = p / (m + n)) :
  M = 1/2 ∨ M = -1 := by
sorry

end NUMINAMATH_CALUDE_value_of_M_l3230_323010


namespace NUMINAMATH_CALUDE_volume_of_region_l3230_323068

-- Define the region
def Region := {p : ℝ × ℝ × ℝ | 
  let (x, y, z) := p
  (|x - y + z| + |x - y - z| ≤ 12) ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0}

-- State the theorem
theorem volume_of_region : MeasureTheory.volume Region = 108 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_region_l3230_323068


namespace NUMINAMATH_CALUDE_comparison_and_estimation_l3230_323042

theorem comparison_and_estimation : 
  (2 * Real.sqrt 3 < 4) ∧ 
  (4 < Real.sqrt 17) ∧ 
  (Real.sqrt 17 < 5) := by sorry

end NUMINAMATH_CALUDE_comparison_and_estimation_l3230_323042


namespace NUMINAMATH_CALUDE_largest_six_digit_divisible_by_88_l3230_323049

theorem largest_six_digit_divisible_by_88 : ∃ n : ℕ, 
  n ≤ 999999 ∧ 
  n ≥ 100000 ∧
  n % 88 = 0 ∧
  ∀ m : ℕ, m ≤ 999999 ∧ m ≥ 100000 ∧ m % 88 = 0 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_six_digit_divisible_by_88_l3230_323049


namespace NUMINAMATH_CALUDE_identity_is_unique_strictly_increasing_double_application_less_than_successor_l3230_323012

-- Define a strictly increasing function from ℕ to ℕ
def StrictlyIncreasing (f : ℕ → ℕ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem identity_is_unique_strictly_increasing_double_application_less_than_successor
  (f : ℕ → ℕ)
  (h_increasing : StrictlyIncreasing f)
  (h_condition : ∀ n, f (f n) < n + 1) :
  ∀ n, f n = n :=
by sorry

end NUMINAMATH_CALUDE_identity_is_unique_strictly_increasing_double_application_less_than_successor_l3230_323012


namespace NUMINAMATH_CALUDE_lives_gained_l3230_323039

theorem lives_gained (initial_lives lost_lives final_lives : ℕ) :
  initial_lives = 14 →
  lost_lives = 4 →
  final_lives = 46 →
  final_lives - (initial_lives - lost_lives) = 36 := by
sorry

end NUMINAMATH_CALUDE_lives_gained_l3230_323039


namespace NUMINAMATH_CALUDE_integral_gt_one_minus_one_over_n_l3230_323096

theorem integral_gt_one_minus_one_over_n (n : ℕ+) :
  ∫ x in (0:ℝ)..1, (1 / (1 + x ^ (n:ℝ))) > 1 - 1 / (n:ℝ) := by sorry

end NUMINAMATH_CALUDE_integral_gt_one_minus_one_over_n_l3230_323096


namespace NUMINAMATH_CALUDE_crayons_lost_or_given_away_l3230_323008

theorem crayons_lost_or_given_away (initial_crayons remaining_crayons : ℕ) 
  (h1 : initial_crayons = 606)
  (h2 : remaining_crayons = 291) :
  initial_crayons - remaining_crayons = 315 := by
  sorry

end NUMINAMATH_CALUDE_crayons_lost_or_given_away_l3230_323008


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3230_323060

theorem quadratic_inequality_range (x : ℝ) : x^2 + 3*x - 10 < 0 ↔ -5 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3230_323060


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l3230_323045

/-- The system of linear equations -/
def system (x₁ x₂ x₃ : ℝ) : Prop :=
  x₁ + 2*x₂ + 4*x₃ = 5 ∧
  2*x₁ + x₂ + 5*x₃ = 7 ∧
  3*x₁ + 2*x₂ + 6*x₃ = 9

/-- The solution satisfies the system of equations -/
theorem solution_satisfies_system :
  system 1 0 1 := by sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l3230_323045


namespace NUMINAMATH_CALUDE_car_speed_l3230_323093

/-- Theorem: Given a car traveling for 5 hours and covering a distance of 800 km, its speed is 160 km/hour. -/
theorem car_speed (time : ℝ) (distance : ℝ) (speed : ℝ) : 
  time = 5 → distance = 800 → speed = distance / time → speed = 160 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_l3230_323093


namespace NUMINAMATH_CALUDE_florist_bouquet_problem_l3230_323052

theorem florist_bouquet_problem (narcissus : ℕ) (chrysanthemums : ℕ) (total_bouquets : ℕ) :
  narcissus = 75 →
  chrysanthemums = 90 →
  total_bouquets = 33 →
  (narcissus + chrysanthemums) % total_bouquets = 0 →
  (narcissus + chrysanthemums) / total_bouquets = 5 :=
by sorry

end NUMINAMATH_CALUDE_florist_bouquet_problem_l3230_323052


namespace NUMINAMATH_CALUDE_f_equals_g_l3230_323024

-- Define the functions
def f (x : ℝ) : ℝ := x^2 - 1
def g (x : ℝ) : ℝ := (x^2 - 1)^(1/3)

-- State the theorem
theorem f_equals_g : f = g := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l3230_323024


namespace NUMINAMATH_CALUDE_minimum_value_implies_ratio_l3230_323055

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sin x * Real.cos x

theorem minimum_value_implies_ratio (θ : ℝ) 
  (h : ∀ x, f x ≥ f θ) : 
  (Real.sin (2 * θ) + 2 * Real.cos θ) / (Real.sin (2 * θ) - 2 * Real.cos (2 * θ)) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_implies_ratio_l3230_323055


namespace NUMINAMATH_CALUDE_max_factors_bound_l3230_323091

/-- The number of positive factors of n -/
def num_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The maximum number of factors for a^m where 1 ≤ a ≤ 20 and 1 ≤ m ≤ 10 is 231 -/
theorem max_factors_bound :
  ∀ a m : ℕ, 1 ≤ a → a ≤ 20 → 1 ≤ m → m ≤ 10 → num_factors (a^m) ≤ 231 := by
  sorry

end NUMINAMATH_CALUDE_max_factors_bound_l3230_323091


namespace NUMINAMATH_CALUDE_value_range_sqrt_sum_bounds_are_tight_l3230_323082

theorem value_range_sqrt_sum (x : ℝ) : 
  ∃ (y : ℝ), y = Real.sqrt (1 + 2*x) + Real.sqrt (1 - 2*x) ∧ 
  Real.sqrt 2 ≤ y ∧ y ≤ 2 :=
sorry

theorem bounds_are_tight : 
  (∃ (x : ℝ), Real.sqrt (1 + 2*x) + Real.sqrt (1 - 2*x) = Real.sqrt 2) ∧
  (∃ (x : ℝ), Real.sqrt (1 + 2*x) + Real.sqrt (1 - 2*x) = 2) :=
sorry

end NUMINAMATH_CALUDE_value_range_sqrt_sum_bounds_are_tight_l3230_323082


namespace NUMINAMATH_CALUDE_egypt_promotion_theorem_l3230_323022

/-- The number of tourists who went to Egypt for free -/
def free_tourists : ℕ := 29

/-- The number of tourists who came on their own -/
def solo_tourists : ℕ := 13

/-- The number of tourists who did not bring anyone -/
def no_referral_tourists : ℕ := 100

theorem egypt_promotion_theorem :
  ∃ (total_tourists : ℕ),
    total_tourists = solo_tourists + 4 * free_tourists ∧
    total_tourists = free_tourists + no_referral_tourists ∧
    free_tourists = 29 := by
  sorry

end NUMINAMATH_CALUDE_egypt_promotion_theorem_l3230_323022


namespace NUMINAMATH_CALUDE_probability_sum_less_than_12_l3230_323051

def roll_dice : ℕ := 6

def total_outcomes : ℕ := roll_dice * roll_dice

def favorable_outcomes : ℕ := total_outcomes - 1

theorem probability_sum_less_than_12 : 
  (favorable_outcomes : ℚ) / total_outcomes = 35 / 36 := by sorry

end NUMINAMATH_CALUDE_probability_sum_less_than_12_l3230_323051


namespace NUMINAMATH_CALUDE_probability_two_red_cards_value_l3230_323099

/-- A standard deck of cards -/
structure Deck :=
  (total_cards : ℕ := 54)
  (red_cards : ℕ := 27)
  (jokers : ℕ := 2)

/-- The probability of drawing two red cards from a standard deck -/
def probability_two_red_cards (d : Deck) : ℚ :=
  (d.red_cards : ℚ) / d.total_cards * (d.red_cards - 1) / (d.total_cards - 1)

/-- Theorem stating the probability of drawing two red cards from a standard deck -/
theorem probability_two_red_cards_value (d : Deck) :
  probability_two_red_cards d = 13 / 53 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_cards_value_l3230_323099


namespace NUMINAMATH_CALUDE_raspberry_harvest_calculation_l3230_323050

/-- Calculates the expected raspberry harvest given garden dimensions and planting parameters. -/
theorem raspberry_harvest_calculation 
  (length width : ℕ) 
  (plants_per_sqft : ℕ) 
  (raspberries_per_plant : ℕ) : 
  length = 10 → 
  width = 7 → 
  plants_per_sqft = 5 → 
  raspberries_per_plant = 12 → 
  length * width * plants_per_sqft * raspberries_per_plant = 4200 :=
by sorry

end NUMINAMATH_CALUDE_raspberry_harvest_calculation_l3230_323050


namespace NUMINAMATH_CALUDE_committee_choice_count_l3230_323007

/-- The number of members in the club -/
def total_members : ℕ := 18

/-- The minimum tenure required for eligibility -/
def min_tenure : ℕ := 10

/-- The number of members to be chosen for the committee -/
def committee_size : ℕ := 3

/-- The number of eligible members (those with tenure ≥ 10 years) -/
def eligible_members : ℕ := total_members - min_tenure + 1

/-- The number of ways to choose the committee -/
def committee_choices : ℕ := Nat.choose eligible_members committee_size

theorem committee_choice_count :
  committee_choices = 84 := by sorry

end NUMINAMATH_CALUDE_committee_choice_count_l3230_323007


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_minus_product_l3230_323044

theorem quadratic_roots_sum_minus_product (a b : ℝ) : 
  a^2 - 3*a + 1 = 0 → b^2 - 3*b + 1 = 0 → a + b - a*b = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_minus_product_l3230_323044


namespace NUMINAMATH_CALUDE_arctan_sum_equals_pi_over_two_l3230_323083

theorem arctan_sum_equals_pi_over_two (y : ℝ) :
  2 * Real.arctan (1/3) + Real.arctan (1/10) + Real.arctan (1/30) + Real.arctan (1/y) = π/2 →
  y = 547/620 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_pi_over_two_l3230_323083


namespace NUMINAMATH_CALUDE_circle_symmetry_line_l3230_323004

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

-- Define the line equation
def line_eq (a x y : ℝ) : Prop := a*x + y + 1 = 0

-- Define symmetry condition
def is_symmetric (a : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_eq x y ∧ line_eq a (-1) 2

-- Theorem statement
theorem circle_symmetry_line (a : ℝ) :
  is_symmetric a → a = 3 := by sorry

end NUMINAMATH_CALUDE_circle_symmetry_line_l3230_323004


namespace NUMINAMATH_CALUDE_zero_sum_points_for_m_3_unique_zero_sum_point_condition_l3230_323009

/-- Definition of a "zero-sum point" in the Cartesian coordinate system -/
def is_zero_sum_point (x y : ℝ) : Prop := x + y = 0

/-- The quadratic function y = x^2 + 3x + m -/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := x^2 + 3*x + m

theorem zero_sum_points_for_m_3 :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_zero_sum_point x₁ y₁ ∧
    is_zero_sum_point x₂ y₂ ∧
    quadratic_function 3 x₁ = y₁ ∧
    quadratic_function 3 x₂ = y₂ ∧
    x₁ = -1 ∧ y₁ = 1 ∧
    x₂ = -3 ∧ y₂ = 3 :=
sorry

theorem unique_zero_sum_point_condition (m : ℝ) :
  (∃! (x y : ℝ), is_zero_sum_point x y ∧ quadratic_function m x = y) ↔ m = 4 :=
sorry

end NUMINAMATH_CALUDE_zero_sum_points_for_m_3_unique_zero_sum_point_condition_l3230_323009


namespace NUMINAMATH_CALUDE_N_subset_M_l3230_323046

def M : Set Nat := {1, 2, 3}
def N : Set Nat := {1}

theorem N_subset_M : N ⊆ M := by sorry

end NUMINAMATH_CALUDE_N_subset_M_l3230_323046


namespace NUMINAMATH_CALUDE_intersection_not_in_first_quadrant_l3230_323074

theorem intersection_not_in_first_quadrant (m : ℝ) : 
  let x := -(m + 4) / 2
  let y := m / 2 - 2
  ¬(x > 0 ∧ y > 0) := by
sorry

end NUMINAMATH_CALUDE_intersection_not_in_first_quadrant_l3230_323074


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_quadrant_l3230_323041

def arithmetic_sequence (n : ℕ) : ℚ := 1 - (n - 1) * (1 / 2)

def intersection_x (a_n : ℚ) : ℚ := (a_n + 1) / 3

def intersection_y (a_n : ℚ) : ℚ := (8 * a_n - 1) / 3

theorem arithmetic_sequence_fourth_quadrant :
  ∀ n : ℕ, n > 0 →
  (intersection_x (arithmetic_sequence n) > 0 ∧ 
   intersection_y (arithmetic_sequence n) < 0) →
  (n = 3 ∨ n = 4) ∧ 
  arithmetic_sequence n = -1/2 * n + 3/2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_quadrant_l3230_323041


namespace NUMINAMATH_CALUDE_three_not_in_range_of_g_l3230_323048

/-- The function g(x) defined as x^2 + 2x + c -/
def g (c : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + c

/-- Theorem stating that 3 is not in the range of g(x) if and only if c > 4 -/
theorem three_not_in_range_of_g (c : ℝ) :
  (∀ x, g c x ≠ 3) ↔ c > 4 := by
  sorry

end NUMINAMATH_CALUDE_three_not_in_range_of_g_l3230_323048


namespace NUMINAMATH_CALUDE_running_to_basketball_ratio_l3230_323021

def trumpet_time : ℕ := 40

theorem running_to_basketball_ratio :
  let running_time := trumpet_time / 2
  let basketball_time := running_time + trumpet_time
  (running_time : ℚ) / basketball_time = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_running_to_basketball_ratio_l3230_323021


namespace NUMINAMATH_CALUDE_num_divisors_not_div_by_3_eq_8_l3230_323036

/-- The number of positive divisors of 210 that are not divisible by 3 -/
def num_divisors_not_div_by_3 : ℕ :=
  (Finset.filter (fun d => d ∣ 210 ∧ ¬(3 ∣ d)) (Finset.range 211)).card

/-- Theorem: The number of positive divisors of 210 that are not divisible by 3 is 8 -/
theorem num_divisors_not_div_by_3_eq_8 : num_divisors_not_div_by_3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_num_divisors_not_div_by_3_eq_8_l3230_323036


namespace NUMINAMATH_CALUDE_lisa_quiz_goal_l3230_323025

theorem lisa_quiz_goal (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (completed_quizzes : ℕ) (current_as : ℕ) : 
  total_quizzes = 60 →
  goal_percentage = 3/4 →
  completed_quizzes = 40 →
  current_as = 26 →
  ∃ (max_non_as : ℕ), 
    max_non_as = 1 ∧
    (current_as + (total_quizzes - completed_quizzes - max_non_as) : ℚ) / total_quizzes ≥ goal_percentage ∧
    ∀ (n : ℕ), n > max_non_as →
      (current_as + (total_quizzes - completed_quizzes - n) : ℚ) / total_quizzes < goal_percentage :=
by sorry

end NUMINAMATH_CALUDE_lisa_quiz_goal_l3230_323025


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l3230_323084

theorem complex_number_in_second_quadrant : ∃ (z : ℂ), 
  z = (1 + 2*I) - (3 - 4*I) ∧ 
  (z.re < 0 ∧ z.im > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l3230_323084


namespace NUMINAMATH_CALUDE_min_value_problem_l3230_323070

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1 / (a + 3) + 1 / (b + 3) = 1 / 4 → 
  x + 3 * y ≤ a + 3 * b ∧ x + 3 * y = 19 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l3230_323070


namespace NUMINAMATH_CALUDE_square_root_of_625_l3230_323067

theorem square_root_of_625 (x : ℝ) (h1 : x > 0) (h2 : x^2 = 625) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_625_l3230_323067


namespace NUMINAMATH_CALUDE_blueberry_pies_count_l3230_323020

/-- Proves that the number of blueberry pies is 10, given 30 total pies equally divided among three types -/
theorem blueberry_pies_count (total_pies : ℕ) (num_types : ℕ) (h1 : total_pies = 30) (h2 : num_types = 3) :
  total_pies / num_types = 10 := by
  sorry

#check blueberry_pies_count

end NUMINAMATH_CALUDE_blueberry_pies_count_l3230_323020


namespace NUMINAMATH_CALUDE_system_equation_ratio_l3230_323035

theorem system_equation_ratio (x y z : ℝ) 
  (eq1 : 3 * x - 4 * y - 2 * z = 0)
  (eq2 : x + 4 * y - 10 * z = 0)
  (z_nonzero : z ≠ 0) :
  (x^2 + 4*x*y) / (y^2 + z^2) = 96/13 := by sorry

end NUMINAMATH_CALUDE_system_equation_ratio_l3230_323035


namespace NUMINAMATH_CALUDE_sqrt_16_div_2_l3230_323053

theorem sqrt_16_div_2 : Real.sqrt 16 / 2 = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_16_div_2_l3230_323053


namespace NUMINAMATH_CALUDE_sum_interior_angles_num_diagonals_l3230_323056

/-- A regular polygon with exterior angles measuring 20° -/
structure RegularPolygon20 where
  n : ℕ
  exterior_angle : ℝ
  h_exterior : exterior_angle = 20

/-- The sum of interior angles of a regular polygon with 20° exterior angles is 2880° -/
theorem sum_interior_angles (p : RegularPolygon20) : 
  (p.n - 2) * 180 = 2880 := by sorry

/-- The number of diagonals in a regular polygon with 20° exterior angles is 135 -/
theorem num_diagonals (p : RegularPolygon20) : 
  p.n * (p.n - 3) / 2 = 135 := by sorry

end NUMINAMATH_CALUDE_sum_interior_angles_num_diagonals_l3230_323056


namespace NUMINAMATH_CALUDE_ticket_cost_difference_l3230_323015

theorem ticket_cost_difference : 
  let num_adults : ℕ := 9
  let num_children : ℕ := 7
  let adult_ticket_price : ℕ := 11
  let child_ticket_price : ℕ := 7
  let adult_total_cost := num_adults * adult_ticket_price
  let child_total_cost := num_children * child_ticket_price
  adult_total_cost - child_total_cost = 50 := by
sorry

end NUMINAMATH_CALUDE_ticket_cost_difference_l3230_323015


namespace NUMINAMATH_CALUDE_five_player_tournament_games_l3230_323057

/-- The number of games in a tournament where each player plays every other player once -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a tournament with 5 players, where each player plays against every other player
    exactly once, the total number of games is 10 -/
theorem five_player_tournament_games :
  tournament_games 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_five_player_tournament_games_l3230_323057


namespace NUMINAMATH_CALUDE_greatest_five_digit_divisible_by_12_15_18_l3230_323016

theorem greatest_five_digit_divisible_by_12_15_18 : 
  ∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ 12 ∣ n ∧ 15 ∣ n ∧ 18 ∣ n → n ≤ 99900 := by
  sorry

#check greatest_five_digit_divisible_by_12_15_18

end NUMINAMATH_CALUDE_greatest_five_digit_divisible_by_12_15_18_l3230_323016


namespace NUMINAMATH_CALUDE_negation_of_forall_exp_minus_x_minus_one_geq_zero_l3230_323054

theorem negation_of_forall_exp_minus_x_minus_one_geq_zero :
  (¬ ∀ x : ℝ, Real.exp x - x - 1 ≥ 0) ↔ (∃ x : ℝ, Real.exp x - x - 1 < 0) :=
sorry

end NUMINAMATH_CALUDE_negation_of_forall_exp_minus_x_minus_one_geq_zero_l3230_323054


namespace NUMINAMATH_CALUDE_jerry_money_left_l3230_323088

-- Define the quantities and prices
def mustard_oil_quantity : ℝ := 2
def mustard_oil_price : ℝ := 13
def pasta_quantity : ℝ := 3
def pasta_price : ℝ := 4
def sauce_quantity : ℝ := 1
def sauce_price : ℝ := 5
def initial_money : ℝ := 50

-- Define the total cost of groceries
def total_cost : ℝ :=
  mustard_oil_quantity * mustard_oil_price +
  pasta_quantity * pasta_price +
  sauce_quantity * sauce_price

-- Define the money left after shopping
def money_left : ℝ := initial_money - total_cost

-- Theorem statement
theorem jerry_money_left : money_left = 7 := by
  sorry

end NUMINAMATH_CALUDE_jerry_money_left_l3230_323088


namespace NUMINAMATH_CALUDE_consecutive_odd_squares_sum_l3230_323005

theorem consecutive_odd_squares_sum (k : ℤ) (n : ℕ) :
  (2 * k - 1)^2 + (2 * k + 1)^2 = n * (n + 1) / 2 ↔ k = 1 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_squares_sum_l3230_323005


namespace NUMINAMATH_CALUDE_expression_evaluation_l3230_323076

theorem expression_evaluation :
  ∀ (a b c d : ℝ),
    d = c + 1 →
    c = b - 8 →
    b = a + 4 →
    a = 7 →
    a + 3 ≠ 0 →
    b - 3 ≠ 0 →
    c + 10 ≠ 0 →
    d + 1 ≠ 0 →
    ((a + 5) / (a + 3)) * ((b - 2) / (b - 3)) * ((c + 7) / (c + 10)) * ((d - 4) / (d + 1)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3230_323076


namespace NUMINAMATH_CALUDE_area_of_four_presentable_set_l3230_323073

/-- A complex number is four-presentable if there exists a complex number w with |w| = 5 such that z = (w - 1/w) / 2 -/
def FourPresentable (z : ℂ) : Prop :=
  ∃ w : ℂ, Complex.abs w = 5 ∧ z = (w - 1 / w) / 2

/-- The set of all four-presentable complex numbers -/
def S : Set ℂ :=
  {z : ℂ | FourPresentable z}

/-- The area of the closed curve formed by S -/
noncomputable def area_S : ℝ := sorry

theorem area_of_four_presentable_set :
  area_S = 18.025 * Real.pi := by sorry

end NUMINAMATH_CALUDE_area_of_four_presentable_set_l3230_323073


namespace NUMINAMATH_CALUDE_men_to_women_ratio_l3230_323002

/-- Proves that the ratio of men to women is 2:1 given the average heights -/
theorem men_to_women_ratio (M W : ℕ) (h_total : M * 185 + W * 170 = (M + W) * 180) :
  M / W = 2 / 1 := by
  sorry

#check men_to_women_ratio

end NUMINAMATH_CALUDE_men_to_women_ratio_l3230_323002


namespace NUMINAMATH_CALUDE_edward_baseball_cards_l3230_323065

/-- The number of binders Edward has -/
def num_binders : ℕ := 7

/-- The number of cards in each binder -/
def cards_per_binder : ℕ := 109

/-- The total number of baseball cards Edward has -/
def total_cards : ℕ := num_binders * cards_per_binder

theorem edward_baseball_cards : total_cards = 763 := by
  sorry

end NUMINAMATH_CALUDE_edward_baseball_cards_l3230_323065


namespace NUMINAMATH_CALUDE_percent_of_percent_l3230_323069

theorem percent_of_percent (y : ℝ) (hy : y ≠ 0) :
  (18 / 100) * y = (30 / 100) * ((60 / 100) * y) :=
by sorry

end NUMINAMATH_CALUDE_percent_of_percent_l3230_323069


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l3230_323003

theorem function_inequality_implies_a_bound (a : ℝ) : 
  (∀ x₁ ∈ Set.Icc 0 1, ∃ x₂ ∈ Set.Icc 1 2, 
    (x₁ - 1 / (x₁ + 1)) ≥ (x₂^2 - 2*a*x₂ + 4)) → 
  a ≥ 9/4 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l3230_323003


namespace NUMINAMATH_CALUDE_perpendicular_line_proof_l3230_323033

-- Define the given line
def given_line (x y : ℝ) : Prop := 3 * x - 6 * y = 9

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := y = -1/2 * x - 2

-- Define the point that the perpendicular line passes through
def point : ℝ × ℝ := (2, -3)

-- Theorem statement
theorem perpendicular_line_proof :
  -- The perpendicular line passes through the given point
  perp_line point.1 point.2 ∧
  -- The two lines are perpendicular
  (∀ x₁ y₁ x₂ y₂ : ℝ, given_line x₁ y₁ → perp_line x₂ y₂ →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((y₂ - y₁) / (x₂ - x₁)) * ((y₁ - y₂) / (x₁ - x₂)) = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_proof_l3230_323033


namespace NUMINAMATH_CALUDE_equation_solution_l3230_323017

theorem equation_solution : ∃ t : ℝ, t = 1.5 ∧ 4 * (4 : ℝ)^t + Real.sqrt (16 * 16^t) = 40 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3230_323017


namespace NUMINAMATH_CALUDE_linear_equation_condition_l3230_323034

/-- The equation (m-1)x^|m|+4=0 is linear if and only if m = -1 -/
theorem linear_equation_condition (m : ℤ) : 
  (∃ a b : ℝ, a ≠ 0 ∧ ∀ x : ℝ, (m - 1 : ℝ) * |x|^|m| + 4 = a * x + b) ↔ m = -1 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l3230_323034


namespace NUMINAMATH_CALUDE_largest_multiple_of_9_less_than_110_l3230_323072

theorem largest_multiple_of_9_less_than_110 : 
  ∀ n : ℕ, n % 9 = 0 → n < 110 → n ≤ 108 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_9_less_than_110_l3230_323072


namespace NUMINAMATH_CALUDE_problem_statement_l3230_323013

theorem problem_statement (a b : ℝ) (h1 : a = 2 + Real.sqrt 3) (h2 : b = 2 - Real.sqrt 3) :
  a^2 + 2*a*b - b*(3*a - b) = 13 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3230_323013


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l3230_323086

def valid_pairs : List (Int × Int) := [
  (12, 6), (-2, 6), (12, 4), (-2, 4), (10, 10), (0, 10), (10, 0)
]

theorem fraction_equation_solution (x y : Int) :
  x + y ≠ 0 →
  (x^2 + y^2) / (x + y) = 10 ↔ (x, y) ∈ valid_pairs :=
by sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l3230_323086


namespace NUMINAMATH_CALUDE_simplify_expression_l3230_323063

theorem simplify_expression (x : ℝ) (h : x ≠ 0) :
  (25 * x^3) * (12 * x^2) * (1 / (5 * x)^3) = 12/5 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3230_323063


namespace NUMINAMATH_CALUDE_stating_time_for_one_click_approx_10_seconds_l3230_323038

/-- Represents the length of a rail in feet -/
def rail_length : ℝ := 15

/-- Represents the number of feet in a mile -/
def feet_per_mile : ℝ := 5280

/-- Represents the number of minutes in an hour -/
def minutes_per_hour : ℝ := 60

/-- Represents the number of seconds in a minute -/
def seconds_per_minute : ℝ := 60

/-- 
Theorem stating that the time taken to hear one click (passing over one rail joint) 
is approximately 10 seconds for a train traveling at any speed.
-/
theorem time_for_one_click_approx_10_seconds (train_speed : ℝ) : 
  train_speed > 0 → 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
    |((rail_length * minutes_per_hour) / (train_speed * feet_per_mile)) * seconds_per_minute - 10| < ε :=
sorry

end NUMINAMATH_CALUDE_stating_time_for_one_click_approx_10_seconds_l3230_323038


namespace NUMINAMATH_CALUDE_andrew_flooring_planks_l3230_323006

/-- The number of planks Andrew bought for his flooring project -/
def total_planks : ℕ := 65

/-- The number of planks used in Andrew's bedroom -/
def bedroom_planks : ℕ := 8

/-- The number of planks used in the living room -/
def living_room_planks : ℕ := 20

/-- The number of planks used in the kitchen -/
def kitchen_planks : ℕ := 11

/-- The number of planks used in the guest bedroom -/
def guest_bedroom_planks : ℕ := bedroom_planks - 2

/-- The number of planks used in each hallway -/
def hallway_planks : ℕ := 4

/-- The number of planks ruined and replaced in each bedroom -/
def ruined_planks_per_bedroom : ℕ := 3

/-- The number of leftover planks -/
def leftover_planks : ℕ := 6

/-- The number of hallways -/
def num_hallways : ℕ := 2

/-- The number of bedrooms -/
def num_bedrooms : ℕ := 2

theorem andrew_flooring_planks :
  total_planks = 
    bedroom_planks + living_room_planks + kitchen_planks + guest_bedroom_planks + 
    (num_hallways * hallway_planks) + (num_bedrooms * ruined_planks_per_bedroom) + 
    leftover_planks :=
by sorry

end NUMINAMATH_CALUDE_andrew_flooring_planks_l3230_323006


namespace NUMINAMATH_CALUDE_inequality_proof_l3230_323059

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h1 : a ≤ 2 * b) (h2 : 2 * b ≤ 4 * a) :
  4 * a * b ≤ 2 * (a^2 + b^2) ∧ 2 * (a^2 + b^2) ≤ 5 * a * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3230_323059


namespace NUMINAMATH_CALUDE_donnelly_class_size_l3230_323075

/-- The number of cupcakes Quinton brought to school -/
def total_cupcakes : ℕ := 40

/-- The number of students in Ms. Delmont's class -/
def delmont_students : ℕ := 18

/-- The number of staff members who received cupcakes -/
def staff_members : ℕ := 4

/-- The number of cupcakes left over -/
def leftover_cupcakes : ℕ := 2

/-- The number of students in Mrs. Donnelly's class -/
def donnelly_students : ℕ := total_cupcakes - delmont_students - staff_members - leftover_cupcakes

theorem donnelly_class_size : donnelly_students = 16 := by
  sorry

end NUMINAMATH_CALUDE_donnelly_class_size_l3230_323075


namespace NUMINAMATH_CALUDE_division_remainder_and_divisibility_l3230_323097

theorem division_remainder_and_divisibility : 
  let dividend : ℕ := 1234567
  let divisor : ℕ := 256
  let remainder : ℕ := dividend % divisor
  remainder = 933 ∧ ¬(∃ k : ℕ, remainder = 7 * k) := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_and_divisibility_l3230_323097


namespace NUMINAMATH_CALUDE_fudge_price_per_pound_l3230_323018

-- Define the given quantities
def total_revenue : ℚ := 212
def fudge_pounds : ℚ := 20
def truffle_dozens : ℚ := 5
def truffle_price : ℚ := 3/2  -- $1.50 as a rational number
def pretzel_dozens : ℚ := 3
def pretzel_price : ℚ := 2

-- Define the theorem
theorem fudge_price_per_pound :
  (total_revenue - (truffle_dozens * 12 * truffle_price + pretzel_dozens * 12 * pretzel_price)) / fudge_pounds = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_fudge_price_per_pound_l3230_323018


namespace NUMINAMATH_CALUDE_arrangement_theorem_l3230_323095

def number_of_people : ℕ := 6
def people_per_row : ℕ := 3

def arrangement_count : ℕ := 216

theorem arrangement_theorem :
  let total_arrangements := number_of_people.factorial
  let front_row_without_A := (people_per_row - 1).choose 1
  let back_row_without_B := (people_per_row - 1).choose 1
  let remaining_arrangements := (number_of_people - 2).factorial
  front_row_without_A * back_row_without_B * remaining_arrangements = arrangement_count :=
by sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l3230_323095


namespace NUMINAMATH_CALUDE_symmetric_line_wrt_x_axis_l3230_323037

/-- Given a line with equation 3x-4y+5=0, this theorem states that its symmetric line
    with respect to the x-axis has the equation 3x+4y+5=0 -/
theorem symmetric_line_wrt_x_axis : 
  ∀ (x y : ℝ), (3 * x - 4 * y + 5 = 0) → 
  ∃ (x' y' : ℝ), (x' = x ∧ y' = -y) ∧ (3 * x' + 4 * y' + 5 = 0) :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_wrt_x_axis_l3230_323037


namespace NUMINAMATH_CALUDE_same_color_sock_pairs_l3230_323061

def num_white_socks : ℕ := 5
def num_black_socks : ℕ := 6
def num_red_socks : ℕ := 3
def num_green_socks : ℕ := 2

def choose_2 (n : ℕ) : ℕ := n * (n - 1) / 2

theorem same_color_sock_pairs :
  choose_2 num_white_socks +
  choose_2 num_black_socks +
  choose_2 num_red_socks +
  choose_2 num_green_socks = 29 := by sorry

end NUMINAMATH_CALUDE_same_color_sock_pairs_l3230_323061


namespace NUMINAMATH_CALUDE_special_quadratic_relation_l3230_323030

theorem special_quadratic_relation (q a b : ℕ) (h : a^2 - q*a*b + b^2 = q) :
  ∃ (c : ℤ), c ≠ a ∧ c^2 - q*b*c + b^2 = q ∧ ∃ (k : ℕ), q = k^2 := by
  sorry

end NUMINAMATH_CALUDE_special_quadratic_relation_l3230_323030


namespace NUMINAMATH_CALUDE_inequality_proof_l3230_323000

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  (1/x) + (2/y) ≥ 25 / (1 + 48*x*y^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3230_323000


namespace NUMINAMATH_CALUDE_z_real_z_pure_imaginary_z_second_quadrant_l3230_323027

/-- Definition of the complex number z in terms of real number m -/
def z (m : ℝ) : ℂ := (m^2 - 2*m - 3 : ℝ) + (m^2 + 3*m + 2 : ℝ) * Complex.I

/-- z is a real number if and only if m = -1 or m = -2 -/
theorem z_real (m : ℝ) : (z m).im = 0 ↔ m = -1 ∨ m = -2 := by sorry

/-- z is a pure imaginary number if and only if m = 3 -/
theorem z_pure_imaginary (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 3 := by sorry

/-- z is in the second quadrant of the complex plane if and only if -1 < m < 3 -/
theorem z_second_quadrant (m : ℝ) : (z m).re < 0 ∧ (z m).im > 0 ↔ -1 < m ∧ m < 3 := by sorry

end NUMINAMATH_CALUDE_z_real_z_pure_imaginary_z_second_quadrant_l3230_323027


namespace NUMINAMATH_CALUDE_smallest_s_value_l3230_323028

theorem smallest_s_value : ∃ s : ℚ, s = 4/7 ∧ 
  (∀ t : ℚ, (15*t^2 - 40*t + 18) / (4*t - 3) + 7*t = 9*t - 2 → s ≤ t) ∧ 
  (15*s^2 - 40*s + 18) / (4*s - 3) + 7*s = 9*s - 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_s_value_l3230_323028


namespace NUMINAMATH_CALUDE_pirate_coin_problem_l3230_323098

def coin_distribution (x : ℕ) : Prop :=
  let paul_coins := x
  let pete_coins := x * (x + 1) / 2
  pete_coins = 5 * paul_coins ∧ 
  paul_coins + pete_coins = 54

theorem pirate_coin_problem :
  ∃ x : ℕ, coin_distribution x :=
sorry

end NUMINAMATH_CALUDE_pirate_coin_problem_l3230_323098


namespace NUMINAMATH_CALUDE_cases_purchased_is_13_l3230_323087

/-- The number of cases of water purchased initially for a children's camp --/
def cases_purchased (group1 group2 group3 : ℕ) 
  (bottles_per_case bottles_per_child_per_day camp_days additional_bottles : ℕ) : ℕ :=
  let group4 := (group1 + group2 + group3) / 2
  let total_children := group1 + group2 + group3 + group4
  let total_bottles_needed := total_children * bottles_per_child_per_day * camp_days
  let bottles_already_have := total_bottles_needed - additional_bottles
  bottles_already_have / bottles_per_case

/-- Theorem stating that the number of cases purchased initially is 13 --/
theorem cases_purchased_is_13 :
  cases_purchased 14 16 12 24 3 3 255 = 13 := by
  sorry

end NUMINAMATH_CALUDE_cases_purchased_is_13_l3230_323087


namespace NUMINAMATH_CALUDE_father_son_age_ratio_l3230_323031

def father_son_ages (son_age : ℕ) (age_difference : ℕ) : Prop :=
  let father_age : ℕ := son_age + age_difference
  let son_age_in_two_years : ℕ := son_age + 2
  let father_age_in_two_years : ℕ := father_age + 2
  (father_age_in_two_years : ℚ) / (son_age_in_two_years : ℚ) = 2

theorem father_son_age_ratio :
  father_son_ages 33 35 := by sorry

end NUMINAMATH_CALUDE_father_son_age_ratio_l3230_323031


namespace NUMINAMATH_CALUDE_distance_before_break_l3230_323071

/-- Proves the distance walked before the break given initial, final, and total distances -/
theorem distance_before_break 
  (initial_distance : ℕ) 
  (final_distance : ℕ) 
  (total_distance : ℕ) 
  (h1 : initial_distance = 3007)
  (h2 : final_distance = 840)
  (h3 : total_distance = 6030) :
  total_distance - (initial_distance + final_distance) = 2183 := by
  sorry

#check distance_before_break

end NUMINAMATH_CALUDE_distance_before_break_l3230_323071


namespace NUMINAMATH_CALUDE_emmas_room_length_l3230_323032

/-- The length of Emma's room, given the width, tiled area, and fraction of room tiled. -/
theorem emmas_room_length (width : ℝ) (tiled_area : ℝ) (tiled_fraction : ℝ) :
  width = 12 →
  tiled_area = 40 →
  tiled_fraction = 1/6 →
  ∃ length : ℝ, length = 20 ∧ tiled_area = tiled_fraction * (width * length) := by
  sorry

end NUMINAMATH_CALUDE_emmas_room_length_l3230_323032


namespace NUMINAMATH_CALUDE_spade_calculation_l3230_323066

-- Define the ◆ operation
def spade (a b : ℝ) : ℝ := (a + b) * (a - b)

-- State the theorem
theorem spade_calculation : spade 4 (spade 5 (-2)) = -425 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l3230_323066


namespace NUMINAMATH_CALUDE_unique_solution_complex_magnitude_one_l3230_323094

/-- There exists exactly one real value of x that satisfies |1 - (x/2)i| = 1 -/
theorem unique_solution_complex_magnitude_one :
  ∃! x : ℝ, Complex.abs (1 - (x / 2) * Complex.I) = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_complex_magnitude_one_l3230_323094


namespace NUMINAMATH_CALUDE_M_lower_bound_l3230_323079

theorem M_lower_bound (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_one : a + b + c = 1) : 
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 := by sorry

end NUMINAMATH_CALUDE_M_lower_bound_l3230_323079


namespace NUMINAMATH_CALUDE_white_marbles_count_l3230_323085

theorem white_marbles_count (total : ℕ) (blue : ℕ) (red : ℕ) (prob_red_or_white : ℚ) :
  total = 20 →
  blue = 6 →
  red = 9 →
  prob_red_or_white = 7/10 →
  total - blue - red = 5 := by
  sorry

end NUMINAMATH_CALUDE_white_marbles_count_l3230_323085


namespace NUMINAMATH_CALUDE_expand_binomials_l3230_323040

theorem expand_binomials (x : ℝ) : (2*x - 3) * (4*x + 5) = 8*x^2 - 2*x - 15 := by
  sorry

end NUMINAMATH_CALUDE_expand_binomials_l3230_323040


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3230_323014

def A : Set ℤ := {x | x^2 - 3*x - 4 < 0}
def B : Set ℤ := {-2, -1, 0, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {0, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3230_323014


namespace NUMINAMATH_CALUDE_no_profit_after_ten_requests_l3230_323080

def genie_operation (x : ℕ) : ℕ := (x + 1000) / 2

def iterate_genie (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | m + 1 => genie_operation (iterate_genie m x)

theorem no_profit_after_ten_requests (x : ℕ) : iterate_genie 10 x ≤ x := by
  sorry


end NUMINAMATH_CALUDE_no_profit_after_ten_requests_l3230_323080


namespace NUMINAMATH_CALUDE_forgotten_lawns_l3230_323019

/-- Proves the number of forgotten lawns given Henry's lawn mowing situation -/
theorem forgotten_lawns (dollars_per_lawn : ℕ) (total_lawns : ℕ) (actual_earnings : ℕ) : 
  dollars_per_lawn = 5 → 
  total_lawns = 12 → 
  actual_earnings = 25 → 
  total_lawns - (actual_earnings / dollars_per_lawn) = 7 := by
  sorry

end NUMINAMATH_CALUDE_forgotten_lawns_l3230_323019
