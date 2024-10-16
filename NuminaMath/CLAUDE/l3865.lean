import Mathlib

namespace NUMINAMATH_CALUDE_asterisk_replacement_l3865_386565

theorem asterisk_replacement : ∃! (x : ℝ), x > 0 ∧ (x / 20) * (x / 80) = 1 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l3865_386565


namespace NUMINAMATH_CALUDE_average_speed_calculation_l3865_386555

def initial_reading : ℕ := 2332
def final_reading : ℕ := 2552
def driving_time_day1 : ℕ := 5
def driving_time_day2 : ℕ := 3

theorem average_speed_calculation :
  let total_distance : ℕ := final_reading - initial_reading
  let total_time : ℕ := driving_time_day1 + driving_time_day2
  (total_distance : ℚ) / (total_time : ℚ) = 27.5 := by sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l3865_386555


namespace NUMINAMATH_CALUDE_prop_1_prop_2_prop_3_prop_4_all_props_correct_l3865_386561

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Proposition 1
theorem prop_1 (m : ℝ) (a b : V) : m • (a - b) = m • a - m • b := by sorry

-- Proposition 2
theorem prop_2 (m n : ℝ) (a : V) : (m - n) • a = m • a - n • a := by sorry

-- Proposition 3
theorem prop_3 (m : ℝ) (a b : V) (h : m ≠ 0) : m • a = m • b → a = b := by sorry

-- Proposition 4
theorem prop_4 (m n : ℝ) (a : V) (h : a ≠ 0) : m • a = n • a → m = n := by sorry

-- All propositions are correct
theorem all_props_correct : 
  (∀ m : ℝ, ∀ a b : V, m • (a - b) = m • a - m • b) ∧
  (∀ m n : ℝ, ∀ a : V, (m - n) • a = m • a - n • a) ∧
  (∀ m : ℝ, ∀ a b : V, m ≠ 0 → (m • a = m • b → a = b)) ∧
  (∀ m n : ℝ, ∀ a : V, a ≠ 0 → (m • a = n • a → m = n)) := by sorry

end NUMINAMATH_CALUDE_prop_1_prop_2_prop_3_prop_4_all_props_correct_l3865_386561


namespace NUMINAMATH_CALUDE_unique_x_with_three_prime_divisors_including_31_l3865_386568

theorem unique_x_with_three_prime_divisors_including_31 :
  ∀ (x n : ℕ),
    x = 8^n - 1 →
    (∃ (p q : ℕ), Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 31 ∧ q ≠ 31 ∧ x = 31 * p * q) →
    (∀ (r : ℕ), Prime r ∧ r ∣ x → r = 31 ∨ r = p ∨ r = q) →
    x = 32767 :=
by sorry

end NUMINAMATH_CALUDE_unique_x_with_three_prime_divisors_including_31_l3865_386568


namespace NUMINAMATH_CALUDE_simplify_expression_l3865_386586

theorem simplify_expression : 5000 * (5000^9) * 2^1000 = 5000^10 * 2^1000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3865_386586


namespace NUMINAMATH_CALUDE_sum_of_special_numbers_l3865_386525

/-- A natural number that ends with 7 zeros and has exactly 72 divisors -/
def SeventyTwoDivisorNumber (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 10^7 * k ∧ (Nat.divisors n).card = 72

theorem sum_of_special_numbers :
  ∃ (a b : ℕ), a ≠ b ∧
    SeventyTwoDivisorNumber a ∧
    SeventyTwoDivisorNumber b ∧
    a + b = 70000000 :=
sorry

end NUMINAMATH_CALUDE_sum_of_special_numbers_l3865_386525


namespace NUMINAMATH_CALUDE_range_of_m_l3865_386573

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, -1 < x ∧ x < 4 → x > 2 * m^2 - 3) ∧
  (∃ x : ℝ, x > 2 * m^2 - 3 ∧ (x ≤ -1 ∨ x ≥ 4)) →
  -1 ≤ m ∧ m ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3865_386573


namespace NUMINAMATH_CALUDE_expression_evaluation_l3865_386581

theorem expression_evaluation :
  let x : ℝ := -2
  let y : ℝ := 1
  ((2*x + y)^2 - y*(y + 4*x) - 8*x) / (-2*x) = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3865_386581


namespace NUMINAMATH_CALUDE_common_remainder_exists_l3865_386524

theorem common_remainder_exists : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 312837 % n = 310650 % n ∧ 312837 % n = 96 := by
  sorry

end NUMINAMATH_CALUDE_common_remainder_exists_l3865_386524


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3865_386549

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  first : ℝ  -- First term of the sequence
  d : ℝ       -- Common difference
  seq_def : ∀ n, a n = first + (n - 1) * d
  sum_def : ∀ n, S n = n * (2 * first + (n - 1) * d) / 2

/-- The main theorem -/
theorem arithmetic_sequence_difference 
  (seq : ArithmeticSequence)
  (h : seq.S 2016 / 2016 = seq.S 2015 / 2015 + 2) :
  seq.d = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3865_386549


namespace NUMINAMATH_CALUDE_thirteen_pow_seven_mod_eight_l3865_386514

theorem thirteen_pow_seven_mod_eight : 13^7 % 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_pow_seven_mod_eight_l3865_386514


namespace NUMINAMATH_CALUDE_book_price_is_two_l3865_386575

/-- The price of a book in rubles -/
def book_price : ℝ := 2

/-- The amount paid for the book in rubles -/
def amount_paid : ℝ := 1

/-- The remaining amount to be paid for the book -/
def remaining_amount : ℝ := book_price - amount_paid

theorem book_price_is_two :
  book_price = 2 ∧
  amount_paid = 1 ∧
  remaining_amount = book_price - amount_paid ∧
  remaining_amount = amount_paid + (book_price - (book_price - amount_paid)) :=
by sorry

end NUMINAMATH_CALUDE_book_price_is_two_l3865_386575


namespace NUMINAMATH_CALUDE_solution_set_when_a_neg_three_a_range_when_subset_condition_l3865_386577

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := |x + a| + |x - 2|

-- Theorem 1
theorem solution_set_when_a_neg_three :
  {x : ℝ | f x (-3) ≥ 3} = {x : ℝ | x ≤ 1 ∨ x ≥ 4} := by sorry

-- Theorem 2
theorem a_range_when_subset_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f x a ≤ |x - 4|) → a ∈ Set.Icc (-3) 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_neg_three_a_range_when_subset_condition_l3865_386577


namespace NUMINAMATH_CALUDE_min_sum_squares_l3865_386562

theorem min_sum_squares (x y : ℝ) (h : (x - 2)^2 + (y - 3)^2 = 1) : 
  ∃ (z : ℝ), z = x^2 + y^2 ∧ (∀ (a b : ℝ), (a - 2)^2 + (b - 3)^2 = 1 → a^2 + b^2 ≥ z) ∧ z = 14 - 2 * Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3865_386562


namespace NUMINAMATH_CALUDE_solution_equation1_no_solution_equation2_l3865_386521

-- Define the equations
def equation1 (x : ℝ) : Prop := (1 / (x - 1) = 5 / (2 * x + 1))
def equation2 (x : ℝ) : Prop := ((x + 1) / (x - 1) - 4 / (x^2 - 1) = 1)

-- Theorem for equation 1
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = 2 := by sorry

-- Theorem for equation 2
theorem no_solution_equation2 : ¬ ∃ x : ℝ, equation2 x := by sorry

end NUMINAMATH_CALUDE_solution_equation1_no_solution_equation2_l3865_386521


namespace NUMINAMATH_CALUDE_apple_pear_difference_l3865_386529

theorem apple_pear_difference :
  let num_apples : ℕ := 17
  let num_pears : ℕ := 9
  num_apples - num_pears = 8 := by sorry

end NUMINAMATH_CALUDE_apple_pear_difference_l3865_386529


namespace NUMINAMATH_CALUDE_speed_conversion_l3865_386588

/-- Conversion factor from m/s to km/h -/
def mps_to_kmh : ℝ := 3.6

/-- The given speed in km/h -/
def given_speed_kmh : ℝ := 1.1076923076923078

/-- The speed in m/s to be proven -/
def speed_mps : ℝ := 0.3076923076923077

theorem speed_conversion :
  speed_mps * mps_to_kmh = given_speed_kmh := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l3865_386588


namespace NUMINAMATH_CALUDE_david_more_than_zachary_pushup_difference_is_thirty_l3865_386587

/-- The number of push-ups David did -/
def david_pushups : ℕ := 37

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 7

/-- David did more push-ups than Zachary -/
theorem david_more_than_zachary : david_pushups > zachary_pushups := by sorry

/-- The difference in push-ups between David and Zachary -/
def pushup_difference : ℕ := david_pushups - zachary_pushups

theorem pushup_difference_is_thirty : pushup_difference = 30 := by sorry

end NUMINAMATH_CALUDE_david_more_than_zachary_pushup_difference_is_thirty_l3865_386587


namespace NUMINAMATH_CALUDE_divisibility_count_l3865_386505

def count_numbers (n : ℕ) : ℕ :=
  (n / 6) - ((n / 12) + (n / 18) - (n / 36))

theorem divisibility_count : count_numbers 2018 = 112 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_count_l3865_386505


namespace NUMINAMATH_CALUDE_first_term_greater_than_2017_l3865_386533

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem first_term_greater_than_2017 :
  ∃ n : ℕ, 
    arithmetic_sequence 5 7 n > 2017 ∧
    arithmetic_sequence 5 7 n = 2021 ∧
    ∀ m : ℕ, m < n → arithmetic_sequence 5 7 m ≤ 2017 :=
by sorry

end NUMINAMATH_CALUDE_first_term_greater_than_2017_l3865_386533


namespace NUMINAMATH_CALUDE_probability_of_y_selection_l3865_386585

theorem probability_of_y_selection (p_x p_both : ℝ) 
  (h_x : p_x = 1/5)
  (h_both : p_both = 0.13333333333333333)
  (h_independent : p_both = p_x * (p_both / p_x)) :
  p_both / p_x = 0.6666666666666667 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_y_selection_l3865_386585


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l3865_386527

theorem largest_angle_in_special_triangle : ∀ (a b c : ℝ),
  -- Two angles sum to 6/5 of a right angle
  a + b = 6/5 * 90 →
  -- One angle is 30° larger than the other
  b = a + 30 →
  -- All angles are non-negative
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c →
  -- Sum of all angles in a triangle is 180°
  a + b + c = 180 →
  -- The largest angle is 72°
  max a (max b c) = 72 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l3865_386527


namespace NUMINAMATH_CALUDE_linear_function_solution_set_l3865_386510

/-- A linear function f(x) = ax + b where a > 0 and f(-2) = 0 -/
def f (a b : ℝ) (h : a > 0) (h2 : a * (-2) + b = 0) (x : ℝ) : ℝ := a * x + b

theorem linear_function_solution_set
  (a b : ℝ) (h : a > 0) (h2 : a * (-2) + b = 0) :
  ∀ x, a * x > b ↔ x > 2 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_solution_set_l3865_386510


namespace NUMINAMATH_CALUDE_angle_with_same_terminal_side_l3865_386584

theorem angle_with_same_terminal_side : ∃ k : ℤ, 2019 + k * 360 = -141 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_same_terminal_side_l3865_386584


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_values_l3865_386591

theorem quadratic_root_implies_a_values (a : ℝ) : 
  ((-2)^2 + (3/2) * a * (-2) - a^2 = 0) → (a = 1 ∨ a = -4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_values_l3865_386591


namespace NUMINAMATH_CALUDE_sum_of_first_six_primes_mod_seventh_prime_l3865_386554

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17

theorem sum_of_first_six_primes_mod_seventh_prime : 
  (first_six_primes.sum) % seventh_prime = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_six_primes_mod_seventh_prime_l3865_386554


namespace NUMINAMATH_CALUDE_divisibility_of_expression_l3865_386509

theorem divisibility_of_expression (a b c : ℤ) (h : 4 * b = 10 - 3 * a + c) :
  ∃ k : ℤ, 3 * b + 15 - c = 1 * k :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_expression_l3865_386509


namespace NUMINAMATH_CALUDE_third_flip_expected_value_l3865_386520

/-- The expected value of a biased coin flip -/
def expected_value (p_heads : ℚ) (win_amount : ℚ) (loss_amount : ℚ) : ℚ :=
  p_heads * win_amount + (1 - p_heads) * (-loss_amount)

theorem third_flip_expected_value :
  let p_heads : ℚ := 2/5
  let win_amount : ℚ := 4
  let loss_amount : ℚ := 6  -- doubled loss amount due to previous two tails
  expected_value p_heads win_amount loss_amount = -2 := by
sorry

end NUMINAMATH_CALUDE_third_flip_expected_value_l3865_386520


namespace NUMINAMATH_CALUDE_mystery_book_shelves_l3865_386547

theorem mystery_book_shelves (books_per_shelf : ℕ) (picture_book_shelves : ℕ) (total_books : ℕ) :
  books_per_shelf = 9 →
  picture_book_shelves = 2 →
  total_books = 72 →
  (total_books - picture_book_shelves * books_per_shelf) / books_per_shelf = 6 :=
by sorry

end NUMINAMATH_CALUDE_mystery_book_shelves_l3865_386547


namespace NUMINAMATH_CALUDE_solution_symmetry_l3865_386574

theorem solution_symmetry (x y : ℝ) : 
  ((x - y) * (x^2 - y^2) = 160 ∧ (x + y) * (x^2 + y^2) = 580) →
  ((3 - 7) * (3^2 - 7^2) = 160 ∧ (3 + 7) * (3^2 + 7^2) = 580) →
  ((7 - 3) * (7^2 - 3^2) = 160 ∧ (7 + 3) * (7^2 + 3^2) = 580) := by
sorry

end NUMINAMATH_CALUDE_solution_symmetry_l3865_386574


namespace NUMINAMATH_CALUDE_expression_evaluation_l3865_386560

theorem expression_evaluation : -3 * 5 - (-4 * -2) + (-15 * -3) / 3 = -8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3865_386560


namespace NUMINAMATH_CALUDE_lowest_score_proof_l3865_386540

def scores_sum (n : ℕ) (mean : ℝ) : ℝ := n * mean

theorem lowest_score_proof 
  (total_scores : ℕ) 
  (original_mean : ℝ) 
  (new_mean : ℝ) 
  (highest_score : ℝ) 
  (h1 : total_scores = 15)
  (h2 : original_mean = 85)
  (h3 : new_mean = 90)
  (h4 : highest_score = 105)
  (h5 : scores_sum total_scores original_mean = 
        scores_sum (total_scores - 2) new_mean + highest_score + 
        (scores_sum total_scores original_mean - scores_sum (total_scores - 2) new_mean - highest_score)) :
  scores_sum total_scores original_mean - scores_sum (total_scores - 2) new_mean - highest_score = 0 := by
sorry

end NUMINAMATH_CALUDE_lowest_score_proof_l3865_386540


namespace NUMINAMATH_CALUDE_complex_quotient_real_l3865_386542

/-- Given complex numbers Z₁ and Z₂, where Z₁ = a + 2i and Z₂ = 3 - 4i,
    if Z₁/Z₂ is a real number, then a = -3/2 -/
theorem complex_quotient_real (a : ℝ) :
  let Z₁ : ℂ := a + 2*I
  let Z₂ : ℂ := 3 - 4*I
  (∃ (r : ℝ), Z₁ / Z₂ = r) → a = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_quotient_real_l3865_386542


namespace NUMINAMATH_CALUDE_paving_stone_length_l3865_386543

/-- Given a rectangular courtyard and paving stones with specific properties,
    prove that the length of each paving stone is 4 meters. -/
theorem paving_stone_length
  (courtyard_length : ℝ)
  (courtyard_width : ℝ)
  (num_stones : ℕ)
  (stone_width : ℝ)
  (h1 : courtyard_length = 40)
  (h2 : courtyard_width = 20)
  (h3 : num_stones = 100)
  (h4 : stone_width = 2)
  : ∃ (stone_length : ℝ), stone_length = 4 ∧
    courtyard_length * courtyard_width = ↑num_stones * stone_length * stone_width :=
by
  sorry


end NUMINAMATH_CALUDE_paving_stone_length_l3865_386543


namespace NUMINAMATH_CALUDE_lego_volume_proof_l3865_386572

/-- The number of rows of Legos -/
def num_rows : ℕ := 7

/-- The number of columns of Legos -/
def num_columns : ℕ := 5

/-- The number of layers of Legos -/
def num_layers : ℕ := 3

/-- The length of a single Lego in centimeters -/
def lego_length : ℝ := 1

/-- The width of a single Lego in centimeters -/
def lego_width : ℝ := 1

/-- The height of a single Lego in centimeters -/
def lego_height : ℝ := 1

/-- The total volume of stacked Legos in cubic centimeters -/
def total_volume : ℝ := num_rows * num_columns * num_layers * lego_length * lego_width * lego_height

theorem lego_volume_proof : total_volume = 105 := by
  sorry

end NUMINAMATH_CALUDE_lego_volume_proof_l3865_386572


namespace NUMINAMATH_CALUDE_square_root_problem_l3865_386531

theorem square_root_problem (n : ℝ) (h : Real.sqrt (9 + n) = 8) : n + 2 = 57 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l3865_386531


namespace NUMINAMATH_CALUDE_probability_not_pulling_prize_l3865_386513

/-- Given odds of 5:6 for pulling a prize, prove that the probability of not pulling the prize is 6/11 -/
theorem probability_not_pulling_prize (favorable_outcomes unfavorable_outcomes : ℕ) 
  (h : favorable_outcomes = 5 ∧ unfavorable_outcomes = 6) : 
  (unfavorable_outcomes : ℚ) / (favorable_outcomes + unfavorable_outcomes) = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_pulling_prize_l3865_386513


namespace NUMINAMATH_CALUDE_green_tractor_price_l3865_386502

-- Define the variables and conditions
def red_tractor_price : ℕ := 20000
def red_tractor_commission : ℚ := 1/10
def green_tractor_commission : ℚ := 1/5
def red_tractors_sold : ℕ := 2
def green_tractors_sold : ℕ := 3
def total_salary : ℕ := 7000

-- Define the theorem
theorem green_tractor_price :
  ∃ (green_tractor_price : ℕ),
    green_tractor_price * green_tractor_commission * green_tractors_sold +
    red_tractor_price * red_tractor_commission * red_tractors_sold =
    total_salary ∧
    green_tractor_price = 5000 :=
by
  sorry

end NUMINAMATH_CALUDE_green_tractor_price_l3865_386502


namespace NUMINAMATH_CALUDE_frog_jump_distance_l3865_386515

/-- The jumping contest between a grasshopper and a frog -/
theorem frog_jump_distance (grasshopper_jump : ℕ) (extra_distance : ℕ) : 
  grasshopper_jump = 17 → extra_distance = 22 → grasshopper_jump + extra_distance = 39 :=
by
  sorry

#check frog_jump_distance

end NUMINAMATH_CALUDE_frog_jump_distance_l3865_386515


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3865_386526

theorem absolute_value_inequality (x : ℝ) : 
  3 ≤ |x + 2| ∧ |x + 2| ≤ 6 ↔ (1 ≤ x ∧ x ≤ 4) ∨ (-8 ≤ x ∧ x ≤ -5) := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3865_386526


namespace NUMINAMATH_CALUDE_expression_evaluation_l3865_386598

theorem expression_evaluation :
  let a : ℚ := 2
  let b : ℚ := -1/2
  let c : ℚ := -1
  a * b * c - (2 * a * b - (3 * a * b * c - b * c) + 4 * a * b * c) = 3/2 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3865_386598


namespace NUMINAMATH_CALUDE_paradise_park_ferris_wheel_large_seat_capacity_l3865_386558

/-- Represents a Ferris wheel with small and large seats -/
structure FerrisWheel where
  small_seats : Nat
  large_seats : Nat
  small_seat_capacity : Nat
  large_seat_capacity : Nat

/-- Calculates the total number of people who can ride on large seats -/
def large_seat_capacity (fw : FerrisWheel) : Nat :=
  fw.large_seats * fw.large_seat_capacity

/-- Theorem stating that the capacity of large seats in the given Ferris wheel is 84 -/
theorem paradise_park_ferris_wheel_large_seat_capacity :
  let fw := FerrisWheel.mk 3 7 16 12
  large_seat_capacity fw = 84 := by
  sorry

end NUMINAMATH_CALUDE_paradise_park_ferris_wheel_large_seat_capacity_l3865_386558


namespace NUMINAMATH_CALUDE_sam_watermelons_l3865_386569

theorem sam_watermelons (grown : ℕ) (eaten : ℕ) (h1 : grown = 4) (h2 : eaten = 3) :
  grown - eaten = 1 := by
  sorry

end NUMINAMATH_CALUDE_sam_watermelons_l3865_386569


namespace NUMINAMATH_CALUDE_jane_ticket_purchase_l3865_386544

/-- Calculates the maximum number of football tickets that can be bought with a given budget, 
    considering a discount for buying more than 5 tickets. -/
def max_tickets_bought (regular_price : ℕ) (discounted_price : ℕ) (budget : ℕ) : ℕ :=
  let full_price_tickets := min (budget / regular_price) 5
  let remaining_budget := budget - full_price_tickets * regular_price
  let discounted_tickets := remaining_budget / discounted_price
  full_price_tickets + discounted_tickets

/-- Proves that given the specific prices and budget, Jane can buy exactly 7 tickets. -/
theorem jane_ticket_purchase : max_tickets_bought 15 12 100 = 7 := by
  sorry

end NUMINAMATH_CALUDE_jane_ticket_purchase_l3865_386544


namespace NUMINAMATH_CALUDE_addition_equality_l3865_386535

theorem addition_equality : 731 + 672 = 1403 := by
  sorry

end NUMINAMATH_CALUDE_addition_equality_l3865_386535


namespace NUMINAMATH_CALUDE_goldfish_death_rate_l3865_386552

/-- The number of goldfish that die each week -/
def goldfish_deaths_per_week : ℕ := 5

/-- The initial number of goldfish -/
def initial_goldfish : ℕ := 18

/-- The number of goldfish purchased each week -/
def goldfish_purchased_per_week : ℕ := 3

/-- The number of weeks -/
def weeks : ℕ := 7

/-- The final number of goldfish -/
def final_goldfish : ℕ := 4

theorem goldfish_death_rate : 
  initial_goldfish + (goldfish_purchased_per_week * weeks) - (goldfish_deaths_per_week * weeks) = final_goldfish :=
by sorry

end NUMINAMATH_CALUDE_goldfish_death_rate_l3865_386552


namespace NUMINAMATH_CALUDE_area_of_valid_fold_points_l3865_386550

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Check if a point is a valid fold point -/
def isValidFoldPoint (t : Triangle) (p : Point) : Prop := sorry

/-- Calculate the area of a set of points -/
def areaOfSet (s : Set Point) : ℝ := sorry

/-- The set of all valid fold points in the triangle -/
def validFoldPoints (t : Triangle) : Set Point := sorry

/-- The main theorem -/
theorem area_of_valid_fold_points (t : Triangle) 
  (h1 : distance t.A t.B = 24)
  (h2 : distance t.B t.C = 48)
  (h3 : distance t.A t.C = 24 * Real.sqrt 2) :
  areaOfSet (validFoldPoints t) = 240 * Real.pi - 360 * Real.sqrt 3 := sorry

end NUMINAMATH_CALUDE_area_of_valid_fold_points_l3865_386550


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l3865_386592

theorem sum_of_reciprocals (a b c d : ℝ) (ω : ℂ) : 
  a ≠ -1 → b ≠ -1 → c ≠ -1 → d ≠ -1 →
  ω^4 = 1 →
  ω ≠ 1 →
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) : ℂ) = 4 / ω^2 →
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) : ℝ) = -1 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l3865_386592


namespace NUMINAMATH_CALUDE_corrected_mean_l3865_386538

theorem corrected_mean (n : ℕ) (original_mean : ℝ) (incorrect_value correct_value : ℝ) :
  n = 50 →
  original_mean = 36 →
  incorrect_value = 23 →
  correct_value = 44 →
  let original_sum := n * original_mean
  let difference := correct_value - incorrect_value
  let corrected_sum := original_sum + difference
  let corrected_mean := corrected_sum / n
  corrected_mean = 36.42 := by
sorry

end NUMINAMATH_CALUDE_corrected_mean_l3865_386538


namespace NUMINAMATH_CALUDE_sam_distance_l3865_386507

/-- Given Marguerite's travel information and Sam's time details, calculate Sam's total distance traveled. -/
theorem sam_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_total_time : ℝ) (sam_rest_time : ℝ)
  (h1 : marguerite_distance = 100)
  (h2 : marguerite_time = 2.4)
  (h3 : sam_total_time = 4)
  (h4 : sam_rest_time = 1) :
  let marguerite_speed := marguerite_distance / marguerite_time
  let sam_drive_time := sam_total_time - sam_rest_time
  sam_drive_time * marguerite_speed = 125 := by
  sorry

end NUMINAMATH_CALUDE_sam_distance_l3865_386507


namespace NUMINAMATH_CALUDE_potato_cooking_time_l3865_386517

def cooking_problem (total_potatoes cooked_potatoes time_per_potato : ℕ) : Prop :=
  let remaining_potatoes := total_potatoes - cooked_potatoes
  remaining_potatoes * time_per_potato = 45

theorem potato_cooking_time :
  cooking_problem 16 7 5 := by
  sorry

end NUMINAMATH_CALUDE_potato_cooking_time_l3865_386517


namespace NUMINAMATH_CALUDE_inner_prism_volume_l3865_386595

theorem inner_prism_volume (w l h : ℕ+) : 
  w * l * h = 128 ↔ (w : ℕ) * (l : ℕ) * (h : ℕ) = 128 := by sorry

end NUMINAMATH_CALUDE_inner_prism_volume_l3865_386595


namespace NUMINAMATH_CALUDE_geometric_sequence_a3_l3865_386570

/-- A geometric sequence with a_1 = 1 and a_5 = 5 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 5 = 5 ∧ ∀ n m : ℕ, a (n + m) = a n * a m

theorem geometric_sequence_a3 (a : ℕ → ℝ) (h : geometric_sequence a) :
  a 3 = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a3_l3865_386570


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l3865_386566

theorem not_p_sufficient_not_necessary_for_q :
  (∃ a : ℝ, (a < -1 → ∀ x > 0, a ≤ (x^2 + 1) / x) ∧
   (∀ x > 0, a ≤ (x^2 + 1) / x → ¬(a < -1))) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l3865_386566


namespace NUMINAMATH_CALUDE_point_quadrant_relation_l3865_386500

/-- If point M(1+a, 2b-1) is in the third quadrant, then point N(a-1, 1-2b) is in the second quadrant. -/
theorem point_quadrant_relation (a b : ℝ) : 
  (1 + a < 0 ∧ 2*b - 1 < 0) → (a - 1 < 0 ∧ 1 - 2*b > 0) :=
by sorry

end NUMINAMATH_CALUDE_point_quadrant_relation_l3865_386500


namespace NUMINAMATH_CALUDE_unwashed_shirts_l3865_386534

theorem unwashed_shirts (short_sleeve : ℕ) (long_sleeve : ℕ) (washed : ℕ) : 
  short_sleeve = 40 → long_sleeve = 23 → washed = 29 → 
  short_sleeve + long_sleeve - washed = 34 := by
  sorry

end NUMINAMATH_CALUDE_unwashed_shirts_l3865_386534


namespace NUMINAMATH_CALUDE_exponential_function_value_l3865_386537

/-- Given an exponential function f(x) = a^x where a > 0, a ≠ 1, and f(3) = 8, prove that f(1) = 2 -/
theorem exponential_function_value (a : ℝ) (f : ℝ → ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : ∀ x, f x = a^x) 
  (h4 : f 3 = 8) : 
  f 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_exponential_function_value_l3865_386537


namespace NUMINAMATH_CALUDE_opposite_sign_implications_l3865_386541

theorem opposite_sign_implications (a b : ℝ) 
  (h : (|a - 2| ≥ 0 ∧ (b + 1)^2 ≤ 0) ∨ (|a - 2| ≤ 0 ∧ (b + 1)^2 ≥ 0)) : 
  b^a = 1 ∧ a^3 + b^15 = 7 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sign_implications_l3865_386541


namespace NUMINAMATH_CALUDE_smallest_n_for_euler_totient_equation_l3865_386559

def euler_totient (n : ℕ) : ℕ := sorry

theorem smallest_n_for_euler_totient_equation : 
  ∀ n : ℕ, n > 0 → euler_totient n = (2^5 * n) / 47 → n ≥ 59895 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_euler_totient_equation_l3865_386559


namespace NUMINAMATH_CALUDE_no_valid_covering_l3865_386551

/-- Represents a 3x3 grid with alternating colors -/
def AlternateColoredGrid : Type := Fin 3 → Fin 3 → Bool

/-- An L-shaped piece covers exactly 3 squares -/
def LPiece : Type := Fin 3 → Fin 2 → Bool

/-- A covering of the grid is a list of L-pieces and their positions -/
def Covering : Type := List (LPiece × Fin 3 × Fin 3)

/-- Checks if a given covering is valid for the 3x3 grid -/
def is_valid_covering (c : Covering) : Prop := sorry

/-- The coloring of the 3x3 grid where corners and center are one color (true) 
    and the rest are another color (false) -/
def standard_coloring : AlternateColoredGrid := 
  fun i j => (i = j) || (i + j = 2)

/-- Any L-piece covers an uneven number of squares of each color -/
axiom l_piece_uneven_coverage (l : LPiece) (i j : Fin 3) :
  (∃ (x : Fin 3) (y : Fin 3), 
    (x ≠ i ∨ y ≠ j) ∧ 
    standard_coloring x y ≠ standard_coloring i j ∧
    l (x - i) (y - j))

/-- Theorem: It's impossible to cover a 3x3 grid with L-shaped pieces -/
theorem no_valid_covering : ¬∃ (c : Covering), is_valid_covering c := by
  sorry

end NUMINAMATH_CALUDE_no_valid_covering_l3865_386551


namespace NUMINAMATH_CALUDE_movie_theater_tickets_l3865_386580

theorem movie_theater_tickets (adult_price child_price total_revenue adult_tickets : ℕ) 
  (h1 : adult_price = 7)
  (h2 : child_price = 4)
  (h3 : total_revenue = 5100)
  (h4 : adult_tickets = 500) :
  ∃ child_tickets : ℕ, 
    adult_price * adult_tickets + child_price * child_tickets = total_revenue ∧
    adult_tickets + child_tickets = 900 :=
by sorry

end NUMINAMATH_CALUDE_movie_theater_tickets_l3865_386580


namespace NUMINAMATH_CALUDE_double_average_marks_l3865_386553

theorem double_average_marks (n : ℕ) (initial_avg : ℝ) (h1 : n = 30) (h2 : initial_avg = 45) :
  let total_marks := n * initial_avg
  let new_total_marks := 2 * total_marks
  let new_avg := new_total_marks / n
  new_avg = 90 := by sorry

end NUMINAMATH_CALUDE_double_average_marks_l3865_386553


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l3865_386583

/-- Given a circle with equation (x+1)^2 + (y-1)^2 = 4, prove that its center is (-1,1) and its radius is 2 -/
theorem circle_center_and_radius :
  ∃ (C : ℝ × ℝ) (r : ℝ),
    (∀ (x y : ℝ), (x + 1)^2 + (y - 1)^2 = 4 ↔ (x - C.1)^2 + (y - C.2)^2 = r^2) ∧
    C = (-1, 1) ∧
    r = 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l3865_386583


namespace NUMINAMATH_CALUDE_turnover_growth_equation_l3865_386582

/-- Represents the turnover growth of a supermarket from January to March -/
structure SupermarketGrowth where
  january_turnover : ℝ
  march_turnover : ℝ
  monthly_growth_rate : ℝ

/-- The equation correctly represents the relationship between turnovers and growth rate -/
theorem turnover_growth_equation (sg : SupermarketGrowth) 
  (h1 : sg.january_turnover = 36)
  (h2 : sg.march_turnover = 48) :
  sg.january_turnover * (1 + sg.monthly_growth_rate)^2 = sg.march_turnover :=
sorry

end NUMINAMATH_CALUDE_turnover_growth_equation_l3865_386582


namespace NUMINAMATH_CALUDE_unique_three_digit_sum_l3865_386536

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is a three-digit number -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Theorem stating that 198 is the only three-digit number equal to 11 times the sum of its digits -/
theorem unique_three_digit_sum : ∃! n : ℕ, isThreeDigit n ∧ n = 11 * sumOfDigits n := by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_sum_l3865_386536


namespace NUMINAMATH_CALUDE_fifteen_percent_of_a_minus_70_l3865_386599

theorem fifteen_percent_of_a_minus_70 (a : ℝ) : (0.15 * a) - 70 = 0.15 * a - 70 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_percent_of_a_minus_70_l3865_386599


namespace NUMINAMATH_CALUDE_filling_pipe_time_calculation_l3865_386518

/-- The time it takes to fill the tank when both pipes are open -/
def both_pipes_time : ℝ := 180

/-- The time it takes for the emptying pipe to empty the tank -/
def emptying_pipe_time : ℝ := 45

/-- The time it takes for the filling pipe to fill the tank -/
def filling_pipe_time : ℝ := 36

theorem filling_pipe_time_calculation :
  (1 / filling_pipe_time) - (1 / emptying_pipe_time) = (1 / both_pipes_time) :=
sorry

end NUMINAMATH_CALUDE_filling_pipe_time_calculation_l3865_386518


namespace NUMINAMATH_CALUDE_sum_of_interior_angles_formula_l3865_386519

-- Define a non-crossed polygon
def NonCrossedPolygon (n : ℕ) : Type := sorry

-- Define the sum of interior angles of a polygon
def SumOfInteriorAngles (p : NonCrossedPolygon n) : ℝ := sorry

-- Theorem statement
theorem sum_of_interior_angles_formula {n : ℕ} (h : n ≥ 3) (p : NonCrossedPolygon n) :
  SumOfInteriorAngles p = (n - 2) * 180 := by sorry

end NUMINAMATH_CALUDE_sum_of_interior_angles_formula_l3865_386519


namespace NUMINAMATH_CALUDE_cows_that_ran_away_l3865_386593

/-- Represents the cow feeding scenario --/
structure CowFeeding where
  initial_cows : ℕ
  feeding_period : ℕ
  days_passed : ℕ
  cows_ran_away : ℕ

/-- Calculates the total food in cow-days --/
def total_food (cf : CowFeeding) : ℕ :=
  cf.initial_cows * cf.feeding_period

/-- Calculates the food consumed before cows ran away --/
def food_consumed (cf : CowFeeding) : ℕ :=
  cf.initial_cows * cf.days_passed

/-- Calculates the remaining food after some cows ran away --/
def remaining_food (cf : CowFeeding) : ℕ :=
  total_food cf - food_consumed cf

/-- Theorem stating the number of cows that ran away --/
theorem cows_that_ran_away (cf : CowFeeding) :
  cf.initial_cows = 1000 →
  cf.feeding_period = 50 →
  cf.days_passed = 10 →
  remaining_food cf = (cf.initial_cows - cf.cows_ran_away) * cf.feeding_period →
  cf.cows_ran_away = 200 := by
  sorry

#check cows_that_ran_away

end NUMINAMATH_CALUDE_cows_that_ran_away_l3865_386593


namespace NUMINAMATH_CALUDE_unique_integer_solution_l3865_386597

theorem unique_integer_solution : 
  ∃! x : ℤ, (12*x - 1) * (6*x - 1) * (4*x - 1) * (3*x - 1) = 330 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l3865_386597


namespace NUMINAMATH_CALUDE_point_coordinates_l3865_386579

def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem point_coordinates :
  ∀ (x y : ℝ),
  fourth_quadrant x y →
  |x| = 3 →
  |y| = 5 →
  (x = 3 ∧ y = -5) :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_l3865_386579


namespace NUMINAMATH_CALUDE_infinite_triangles_with_side_ten_l3865_386530

/-- A function that checks if three positive integers can form a triangle -/
def can_form_triangle (a b c : ℕ+) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- The theorem stating that there are infinitely many triangles with sides x, y, and 10 -/
theorem infinite_triangles_with_side_ten :
  ∀ n : ℕ, ∃ x y : ℕ+, x > n ∧ y > n ∧ can_form_triangle x y 10 :=
sorry

end NUMINAMATH_CALUDE_infinite_triangles_with_side_ten_l3865_386530


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l3865_386556

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 + 8*x + 9 = 0 ↔ (x + 4)^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l3865_386556


namespace NUMINAMATH_CALUDE_equation_solution_l3865_386501

theorem equation_solution (n k l m : ℕ) :
  l > 1 →
  (1 + n^k)^l = 1 + n^m →
  n = 2 ∧ k = 1 ∧ l = 2 ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3865_386501


namespace NUMINAMATH_CALUDE_total_sleep_time_in_week_l3865_386548

/-- The number of hours a cougar sleeps per night -/
def cougar_sleep : ℕ := 4

/-- The additional hours a zebra sleeps compared to a cougar -/
def zebra_extra_sleep : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: The total sleep time for both a cougar and a zebra in one week is 70 hours -/
theorem total_sleep_time_in_week : 
  (cougar_sleep * days_in_week) + ((cougar_sleep + zebra_extra_sleep) * days_in_week) = 70 := by
  sorry


end NUMINAMATH_CALUDE_total_sleep_time_in_week_l3865_386548


namespace NUMINAMATH_CALUDE_cat_relocation_proportion_l3865_386504

/-- Calculates the proportion of cats relocated in the second mission -/
def proportion_relocated (initial_cats : ℕ) (first_mission_relocated : ℕ) (final_remaining : ℕ) : ℚ :=
  let remaining_after_first := initial_cats - first_mission_relocated
  let relocated_second := remaining_after_first - final_remaining
  relocated_second / remaining_after_first

theorem cat_relocation_proportion :
  proportion_relocated 1800 600 600 = 1/2 := by
  sorry

#eval proportion_relocated 1800 600 600

end NUMINAMATH_CALUDE_cat_relocation_proportion_l3865_386504


namespace NUMINAMATH_CALUDE_infinitely_many_increasing_prime_divisors_l3865_386512

-- Define w(n) as the number of different prime divisors of n
def w (n : Nat) : Nat :=
  (Nat.factors n).toFinset.card

-- Theorem statement
theorem infinitely_many_increasing_prime_divisors :
  ∃ (S : Set Nat), Set.Infinite S ∧ ∀ n ∈ S, w n < w (n + 1) ∧ w (n + 1) < w (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_increasing_prime_divisors_l3865_386512


namespace NUMINAMATH_CALUDE_common_chord_of_circles_l3865_386557

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 10*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 2*y - 40 = 0

-- Define the line
def common_chord (x y : ℝ) : Prop := 2*x + y - 5 = 0

-- Theorem statement
theorem common_chord_of_circles :
  ∀ (x y : ℝ), circle1 x y ∧ circle2 x y → common_chord x y :=
by sorry

end NUMINAMATH_CALUDE_common_chord_of_circles_l3865_386557


namespace NUMINAMATH_CALUDE_max_type_a_mascots_l3865_386590

/-- Represents the mascot types -/
inductive MascotType
| A
| B

/-- Represents the mascot purchase scenario -/
structure MascotPurchase where
  totalMascots : ℕ
  totalCost : ℕ
  unitPriceA : ℕ
  unitPriceB : ℕ
  newBudget : ℕ
  newTotalMascots : ℕ

/-- Conditions for the mascot purchase -/
def validMascotPurchase (mp : MascotPurchase) : Prop :=
  mp.totalMascots = 110 ∧
  mp.totalCost = 6000 ∧
  mp.unitPriceA = (6 * mp.unitPriceB) / 5 ∧
  mp.totalCost = mp.totalMascots / 2 * (mp.unitPriceA + mp.unitPriceB) ∧
  mp.newBudget = 16800 ∧
  mp.newTotalMascots = 300

/-- Theorem: The maximum number of type A mascots that can be purchased in the second round is 180 -/
theorem max_type_a_mascots (mp : MascotPurchase) (h : validMascotPurchase mp) :
  ∀ n : ℕ, n ≤ mp.newTotalMascots → n * mp.unitPriceA + (mp.newTotalMascots - n) * mp.unitPriceB ≤ mp.newBudget →
  n ≤ 180 :=
sorry

end NUMINAMATH_CALUDE_max_type_a_mascots_l3865_386590


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_congruence_l3865_386508

def arithmetic_sequence_sum (a : ℕ) (l : ℕ) (d : ℕ) : ℕ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem arithmetic_sequence_sum_congruence :
  let a := 2
  let l := 137
  let d := 5
  let S := arithmetic_sequence_sum a l d
  S % 20 = 6 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_congruence_l3865_386508


namespace NUMINAMATH_CALUDE_father_twice_son_age_l3865_386522

/-- Proves that the number of years after which a father aged 42 will be twice as old as his son aged 14 is 14 years. -/
theorem father_twice_son_age (father_age son_age : ℕ) (h1 : father_age = 42) (h2 : son_age = 14) :
  ∃ x : ℕ, father_age + x = 2 * (son_age + x) ∧ x = 14 :=
by sorry

end NUMINAMATH_CALUDE_father_twice_son_age_l3865_386522


namespace NUMINAMATH_CALUDE_john_total_spend_l3865_386503

-- Define the given quantities
def silver_amount : Real := 1.5
def silver_price_per_ounce : Real := 20
def gold_amount : Real := 2 * silver_amount
def gold_price_per_ounce : Real := 50 * silver_price_per_ounce

-- Define the total cost function
def total_cost : Real :=
  silver_amount * silver_price_per_ounce + gold_amount * gold_price_per_ounce

-- Theorem statement
theorem john_total_spend :
  total_cost = 3030 := by
  sorry

end NUMINAMATH_CALUDE_john_total_spend_l3865_386503


namespace NUMINAMATH_CALUDE_abs_gt_one_iff_square_minus_one_gt_zero_l3865_386563

theorem abs_gt_one_iff_square_minus_one_gt_zero :
  ∀ x : ℝ, |x| > 1 ↔ x^2 - 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_gt_one_iff_square_minus_one_gt_zero_l3865_386563


namespace NUMINAMATH_CALUDE_triangle_area_l3865_386539

-- Define the lines
def line1 (x : ℝ) : ℝ := x
def line2 (x : ℝ) : ℝ := -x
def line3 : ℝ := 8

-- Define the theorem
theorem triangle_area : 
  let A : ℝ × ℝ := (8, 8)
  let B : ℝ × ℝ := (-8, 8)
  let O : ℝ × ℝ := (0, 0)
  let base := |A.1 - B.1|
  let height := |O.2 - line3|
  (1 / 2 : ℝ) * base * height = 64 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3865_386539


namespace NUMINAMATH_CALUDE_jimmys_income_l3865_386594

/-- Given Rebecca's current and increased income, and the fact that her increased income
    is 50% of the combined income, prove Jimmy's current income. -/
theorem jimmys_income
  (rebecca_current : ℕ)
  (rebecca_increase : ℕ)
  (h1 : rebecca_current = 15000)
  (h2 : rebecca_increase = 3000)
  (h3 : rebecca_current + rebecca_increase = (rebecca_current + rebecca_increase + jimmy_current) / 2) :
  jimmy_current = 18000 :=
by sorry

end NUMINAMATH_CALUDE_jimmys_income_l3865_386594


namespace NUMINAMATH_CALUDE_starters_selection_count_l3865_386546

def total_players : ℕ := 18
def quadruplets : ℕ := 4
def starters : ℕ := 5

def select_starters : ℕ := Nat.choose total_players starters - 
  (Nat.choose quadruplets 3 * Nat.choose (total_players - quadruplets) (starters - 3) + 
   Nat.choose quadruplets 4 * Nat.choose (total_players - quadruplets) (starters - 4))

theorem starters_selection_count : select_starters = 8190 := by
  sorry

end NUMINAMATH_CALUDE_starters_selection_count_l3865_386546


namespace NUMINAMATH_CALUDE_sasha_can_paint_all_l3865_386567

/-- A cell in the rectangle grid -/
structure Cell where
  x : Nat
  y : Nat

/-- The state of the rectangle grid -/
def Grid := Cell → Bool

/-- Check if a cell has an odd number of painted neighbors -/
def hasOddPaintedNeighbors (grid : Grid) (cell : Cell) : Bool :=
  sorry

/-- The painting rule: a cell can be painted if it has an odd number of painted neighbors -/
def canPaint (grid : Grid) (cell : Cell) : Bool :=
  !grid cell ∧ hasOddPaintedNeighbors grid cell

/-- Check if all cells in the grid are painted -/
def allPainted (m n : Nat) (grid : Grid) : Bool :=
  sorry

/-- The main theorem: characterizes when Sasha can paint all cells -/
theorem sasha_can_paint_all (m n : Nat) :
  (m % 2 = 0 ∧ n % 2 = 1 → ∀ (initialCell : Cell), ∃ (finalGrid : Grid), allPainted m n finalGrid) ∧
  (m % 2 = 0 ∧ n % 2 = 0 → ∀ (initialCell : Cell), ¬∃ (finalGrid : Grid), allPainted m n finalGrid) :=
sorry

end NUMINAMATH_CALUDE_sasha_can_paint_all_l3865_386567


namespace NUMINAMATH_CALUDE_product_difference_implies_sum_l3865_386576

theorem product_difference_implies_sum (p q : ℤ) 
  (h1 : p * q = 1764) 
  (h2 : p - q = 20) : 
  p + q = 86 := by
sorry

end NUMINAMATH_CALUDE_product_difference_implies_sum_l3865_386576


namespace NUMINAMATH_CALUDE_inscribed_circle_square_side_length_l3865_386532

theorem inscribed_circle_square_side_length 
  (circle_area : ℝ) 
  (h_area : circle_area = 36 * Real.pi) : 
  ∃ (square_side : ℝ), 
    square_side = 12 ∧ 
    circle_area = Real.pi * (square_side / 2) ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_square_side_length_l3865_386532


namespace NUMINAMATH_CALUDE_seven_digit_divisible_by_13_l3865_386589

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ n = 3000000 + 100000 * a + 100 * b + 3

def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 13 * k

theorem seven_digit_divisible_by_13 :
  {n : ℕ | is_valid_number n ∧ is_divisible_by_13 n} =
  {3000803, 3020303, 3030703, 3050203, 3060603, 3080103, 3090503} :=
sorry

end NUMINAMATH_CALUDE_seven_digit_divisible_by_13_l3865_386589


namespace NUMINAMATH_CALUDE_lisa_walks_2100_meters_l3865_386511

/-- Calculates the total distance Lisa walks over two days given her usual pace and terrain conditions -/
def lisa_total_distance (usual_pace : ℝ) : ℝ :=
  let day1_morning := usual_pace * 60
  let day1_evening := (usual_pace * 0.7) * 60
  let day2_morning := (usual_pace * 1.2) * 60
  let day2_evening := (usual_pace * 0.6) * 60
  day1_morning + day1_evening + day2_morning + day2_evening

/-- Theorem stating that Lisa's total distance over two days is 2100 meters -/
theorem lisa_walks_2100_meters :
  lisa_total_distance 10 = 2100 := by
  sorry

#eval lisa_total_distance 10

end NUMINAMATH_CALUDE_lisa_walks_2100_meters_l3865_386511


namespace NUMINAMATH_CALUDE_equal_differences_l3865_386528

theorem equal_differences (x : Fin 102 → ℕ) 
  (h_increasing : ∀ i j : Fin 102, i < j → x i < x j)
  (h_upper_bound : ∀ i : Fin 102, x i < 255) :
  ∃ (S : Finset (Fin 101)) (d : ℕ), 
    S.card ≥ 26 ∧ ∀ i ∈ S, x (i + 1) - x i = d := by
  sorry

end NUMINAMATH_CALUDE_equal_differences_l3865_386528


namespace NUMINAMATH_CALUDE_circle_tangent_line_equation_non_intersecting_line_condition_intersecting_line_orthogonal_condition_l3865_386564

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 9

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x + y + 3 * Real.sqrt 2 + 1 = 0

-- Define the non-intersecting line
def non_intersecting_line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

-- Define the intersecting line
def intersecting_line (m : ℝ) (x y : ℝ) : Prop := y = x + m

theorem circle_tangent_line_equation :
  ∀ x y : ℝ, circle_C x y ↔ (x - 1)^2 + (y + 2)^2 = 9 := by sorry

theorem non_intersecting_line_condition :
  ∀ k : ℝ, (∀ x y : ℝ, ¬(circle_C x y ∧ non_intersecting_line k x y)) ↔ (0 < k ∧ k < 3/4) := by sorry

theorem intersecting_line_orthogonal_condition :
  ∀ m : ℝ, (∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ 
    intersecting_line m x₁ y₁ ∧ intersecting_line m x₂ y₂ ∧
    x₁ * x₂ + y₁ * y₂ = 0) ↔ (m = 1 ∨ m = -4) := by sorry

end NUMINAMATH_CALUDE_circle_tangent_line_equation_non_intersecting_line_condition_intersecting_line_orthogonal_condition_l3865_386564


namespace NUMINAMATH_CALUDE_probability_james_and_david_chosen_l3865_386516

def total_workers : ℕ := 22
def workers_to_choose : ℕ := 4

theorem probability_james_and_david_chosen :
  (Nat.choose (total_workers - 2) (workers_to_choose - 2)) / 
  (Nat.choose total_workers workers_to_choose) = 2 / 231 := by
  sorry

end NUMINAMATH_CALUDE_probability_james_and_david_chosen_l3865_386516


namespace NUMINAMATH_CALUDE_triangle_two_solutions_l3865_386578

theorem triangle_two_solutions (a b : ℝ) (A : ℝ) (ha : a = Real.sqrt 3) (hb : b = 3) (hA : A = π / 6) :
  (b * Real.sin A < a) ∧ (a < b) → ∃ (B C : ℝ), 0 < B ∧ 0 < C ∧ A + B + C = π ∧
  a = b * Real.sin C / Real.sin A ∧ 
  b = a * Real.sin B / Real.sin A :=
sorry

end NUMINAMATH_CALUDE_triangle_two_solutions_l3865_386578


namespace NUMINAMATH_CALUDE_smaller_circle_circumference_l3865_386596

theorem smaller_circle_circumference 
  (R : ℝ) 
  (h1 : R > 0) 
  (h2 : π * (3*R)^2 - π * R^2 = 32 / π) : 
  2 * π * R = 4 := by
sorry

end NUMINAMATH_CALUDE_smaller_circle_circumference_l3865_386596


namespace NUMINAMATH_CALUDE_min_area_rectangle_l3865_386545

theorem min_area_rectangle (l w : ℕ) : 
  l > 0 → w > 0 → 2 * (l + w) = 80 → l * w ≥ 39 := by
  sorry

end NUMINAMATH_CALUDE_min_area_rectangle_l3865_386545


namespace NUMINAMATH_CALUDE_yuan_david_age_difference_l3865_386523

theorem yuan_david_age_difference : 
  ∀ (yuan_age david_age : ℕ),
    david_age = 7 →
    yuan_age = 2 * david_age →
    yuan_age - david_age = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_yuan_david_age_difference_l3865_386523


namespace NUMINAMATH_CALUDE_greyEyedBlackHairedCount_l3865_386506

/-- Represents the characteristics of students in a class. -/
structure ClassCharacteristics where
  total : ℕ
  greenEyedRedHaired : ℕ
  blackHaired : ℕ
  greyEyed : ℕ

/-- Calculates the number of grey-eyed black-haired students given the class characteristics. -/
def greyEyedBlackHaired (c : ClassCharacteristics) : ℕ :=
  c.greyEyed - (c.total - c.blackHaired - c.greenEyedRedHaired)

/-- Theorem stating that for the given class characteristics, 
    the number of grey-eyed black-haired students is 20. -/
theorem greyEyedBlackHairedCount : 
  let c : ClassCharacteristics := {
    total := 60,
    greenEyedRedHaired := 20,
    blackHaired := 35,
    greyEyed := 25
  }
  greyEyedBlackHaired c = 20 := by
  sorry


end NUMINAMATH_CALUDE_greyEyedBlackHairedCount_l3865_386506


namespace NUMINAMATH_CALUDE_outfit_choices_l3865_386571

/-- The number of shirts -/
def num_shirts : ℕ := 5

/-- The number of pants -/
def num_pants : ℕ := 5

/-- The number of hats -/
def num_hats : ℕ := 7

/-- The number of colors shared by shirts and pants -/
def num_shared_colors : ℕ := 5

/-- The total number of possible outfit combinations -/
def total_combinations : ℕ := num_shirts * num_pants * num_hats

/-- The number of undesired outfit combinations (shirt and pants same color) -/
def undesired_combinations : ℕ := num_shared_colors * num_hats

/-- The number of valid outfit choices -/
def valid_outfit_choices : ℕ := total_combinations - undesired_combinations

theorem outfit_choices :
  valid_outfit_choices = 140 := by sorry

end NUMINAMATH_CALUDE_outfit_choices_l3865_386571
