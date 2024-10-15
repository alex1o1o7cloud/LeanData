import Mathlib

namespace NUMINAMATH_CALUDE_katies_games_l1384_138483

theorem katies_games (friends_games : ℕ) (katies_extra_games : ℕ) 
  (h1 : friends_games = 59)
  (h2 : katies_extra_games = 22) : 
  friends_games + katies_extra_games = 81 :=
by sorry

end NUMINAMATH_CALUDE_katies_games_l1384_138483


namespace NUMINAMATH_CALUDE_equation_solution_l1384_138480

theorem equation_solution :
  ∃ x : ℝ, 3 * x - 2 * x = 7 ∧ x = 7 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1384_138480


namespace NUMINAMATH_CALUDE_train_length_calculation_l1384_138490

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length_calculation (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 45 ∧ 
  crossing_time = 30 ∧ 
  bridge_length = 255 →
  (train_speed * 1000 / 3600) * crossing_time - bridge_length = 120 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l1384_138490


namespace NUMINAMATH_CALUDE_union_covers_reals_l1384_138475

def A : Set ℝ := {x | x ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x > a}

theorem union_covers_reals (a : ℝ) : A ∪ B a = Set.univ → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_union_covers_reals_l1384_138475


namespace NUMINAMATH_CALUDE_bee_closest_point_to_flower_l1384_138408

/-- The point where the bee starts moving away from the flower -/
def closest_point : ℝ × ℝ := (4.6, 13.8)

/-- The location of the flower -/
def flower_location : ℝ × ℝ := (10, 12)

/-- The path of the bee -/
def bee_path (x : ℝ) : ℝ := 3 * x

theorem bee_closest_point_to_flower :
  let (c, d) := closest_point
  -- The point is on the bee's path
  (d = bee_path c) ∧
  -- This point is the closest to the flower
  (∀ x y, y = bee_path x → (x - 10)^2 + (y - 12)^2 ≥ (c - 10)^2 + (d - 12)^2) ∧
  -- The sum of coordinates is 18.4
  (c + d = 18.4) := by sorry

end NUMINAMATH_CALUDE_bee_closest_point_to_flower_l1384_138408


namespace NUMINAMATH_CALUDE_newton_sports_club_membership_ratio_l1384_138458

/-- Proves that given the average ages of female and male members, and the overall average age,
    the ratio of female to male members is 8:17 --/
theorem newton_sports_club_membership_ratio
  (avg_age_female : ℝ)
  (avg_age_male : ℝ)
  (avg_age_all : ℝ)
  (h_female : avg_age_female = 45)
  (h_male : avg_age_male = 20)
  (h_all : avg_age_all = 28)
  : ∃ (f m : ℝ), f > 0 ∧ m > 0 ∧ f / m = 8 / 17 := by
  sorry

end NUMINAMATH_CALUDE_newton_sports_club_membership_ratio_l1384_138458


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1384_138487

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  a 3 + 3 * a 8 + a 13 = 120 → a 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1384_138487


namespace NUMINAMATH_CALUDE_complex_square_on_positive_imaginary_axis_l1384_138423

theorem complex_square_on_positive_imaginary_axis (a : ℝ) :
  let z : ℂ := a + 2 * Complex.I
  (∃ (y : ℝ), y > 0 ∧ z^2 = Complex.I * y) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_on_positive_imaginary_axis_l1384_138423


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_smallest_odd_primes_l1384_138434

theorem smallest_four_digit_divisible_by_smallest_odd_primes : 
  ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 → (3 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧ (11 ∣ n) → n ≥ 1155 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_smallest_odd_primes_l1384_138434


namespace NUMINAMATH_CALUDE_profit_percent_approximation_l1384_138460

def selling_price : ℝ := 2524.36
def cost_price : ℝ := 2400

theorem profit_percent_approximation :
  let profit := selling_price - cost_price
  let profit_percent := (profit / cost_price) * 100
  ∃ ε > 0, abs (profit_percent - 5.18) < ε :=
by sorry

end NUMINAMATH_CALUDE_profit_percent_approximation_l1384_138460


namespace NUMINAMATH_CALUDE_min_squares_to_exceed_10000_l1384_138447

/-- Represents the squaring operation on a calculator --/
def square (x : ℕ) : ℕ := x * x

/-- Represents n iterations of squaring, starting from x --/
def iterate_square (x : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => square (iterate_square x n)

/-- The theorem to be proved --/
theorem min_squares_to_exceed_10000 :
  (∃ n : ℕ, iterate_square 5 n > 10000) ∧
  (∀ n : ℕ, iterate_square 5 n > 10000 → n ≥ 3) ∧
  (iterate_square 5 3 > 10000) :=
sorry

end NUMINAMATH_CALUDE_min_squares_to_exceed_10000_l1384_138447


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_nine_l1384_138422

theorem arithmetic_square_root_of_nine : ∃ x : ℝ, x ≥ 0 ∧ x^2 = 9 ∧ ∀ y : ℝ, y ≥ 0 ∧ y^2 = 9 → y = x :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_nine_l1384_138422


namespace NUMINAMATH_CALUDE_no_error_in_calculation_l1384_138493

theorem no_error_in_calculation : 
  (7 * 4) / (5/3) = (7 * 4) * (3/5) := by
  sorry

#eval (7 * 4) / (5/3) -- To verify the result
#eval (7 * 4) * (3/5) -- To verify the result

end NUMINAMATH_CALUDE_no_error_in_calculation_l1384_138493


namespace NUMINAMATH_CALUDE_reciprocal_of_fraction_difference_l1384_138437

theorem reciprocal_of_fraction_difference : 
  (((2 : ℚ) / 5 - (3 : ℚ) / 4)⁻¹ : ℚ) = -(20 : ℚ) / 7 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_fraction_difference_l1384_138437


namespace NUMINAMATH_CALUDE_movie_session_duration_l1384_138488

/-- Represents the start time of a movie session -/
structure SessionTime where
  hour : ℕ
  minute : ℕ
  h_valid : hour < 24
  m_valid : minute < 60

/-- Represents the duration of a movie session -/
structure SessionDuration where
  hours : ℕ
  minutes : ℕ
  m_valid : minutes < 60

/-- Checks if a given SessionTime is consistent with the known session times -/
def is_consistent (st : SessionTime) (duration : SessionDuration) : Prop :=
  let next_session := SessionTime.mk 
    ((st.hour + duration.hours + (st.minute + duration.minutes) / 60) % 24)
    ((st.minute + duration.minutes) % 60)
    sorry
    sorry
  (st.hour = 12 ∧ next_session.hour = 13) ∨
  (st.hour = 13 ∧ next_session.hour = 14) ∨
  (st.hour = 23 ∧ next_session.hour = 24) ∨
  (st.hour = 24 ∧ next_session.hour = 1)

theorem movie_session_duration : 
  ∃ (start : SessionTime) (duration : SessionDuration),
    duration.hours = 1 ∧ 
    duration.minutes = 50 ∧
    is_consistent start duration ∧
    (∀ (other_duration : SessionDuration),
      is_consistent start other_duration → 
      other_duration = duration) := by
  sorry

end NUMINAMATH_CALUDE_movie_session_duration_l1384_138488


namespace NUMINAMATH_CALUDE_triangle_side_length_l1384_138465

theorem triangle_side_length 
  (x y z : ℝ) 
  (X Y Z : ℝ) 
  (h1 : y = 7)
  (h2 : z = 3)
  (h3 : Real.cos (Y - Z) = 7/8)
  (h4 : x > 0 ∧ y > 0 ∧ z > 0)
  (h5 : X + Y + Z = Real.pi)
  (h6 : x / Real.sin X = y / Real.sin Y)
  (h7 : y / Real.sin Y = z / Real.sin Z) :
  x = Real.sqrt 18.625 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1384_138465


namespace NUMINAMATH_CALUDE_correct_setup_is_valid_l1384_138452

-- Define the structure for an experimental setup
structure ExperimentalSetup :=
  (num_plates : Nat)
  (bacteria_counts : List Nat)
  (average_count : Nat)

-- Define the conditions for a valid experimental setup
def is_valid_setup (setup : ExperimentalSetup) : Prop :=
  setup.num_plates ≥ 3 ∧
  setup.bacteria_counts.length = setup.num_plates ∧
  setup.average_count = setup.bacteria_counts.sum / setup.num_plates ∧
  setup.bacteria_counts.all (λ count => 
    setup.bacteria_counts.all (λ other_count => 
      (count : Int) - other_count ≤ 50 ∧ other_count - count ≤ 50))

-- Define the correct setup (option D)
def correct_setup : ExperimentalSetup :=
  { num_plates := 3,
    bacteria_counts := [210, 240, 250],
    average_count := 233 }

-- Theorem to prove
theorem correct_setup_is_valid :
  is_valid_setup correct_setup :=
sorry

end NUMINAMATH_CALUDE_correct_setup_is_valid_l1384_138452


namespace NUMINAMATH_CALUDE_sachins_age_l1384_138459

theorem sachins_age (sachin rahul : ℝ) 
  (h1 : rahul = sachin + 7)
  (h2 : sachin / rahul = 7 / 9) :
  sachin = 24.5 := by
sorry

end NUMINAMATH_CALUDE_sachins_age_l1384_138459


namespace NUMINAMATH_CALUDE_inverse_and_determinant_properties_l1384_138474

/-- Given a 2x2 matrix A with its inverse, prove properties about A^2 and (A^(-1))^2 -/
theorem inverse_and_determinant_properties (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A⁻¹ = ![![3, 4], ![-2, -2]]) : 
  (A^2)⁻¹ = ![![1, 4], ![-2, 0]] ∧ 
  Matrix.det ((A⁻¹)^2) = 8 := by
  sorry


end NUMINAMATH_CALUDE_inverse_and_determinant_properties_l1384_138474


namespace NUMINAMATH_CALUDE_orchestra_members_count_l1384_138463

theorem orchestra_members_count :
  ∃! n : ℕ, 100 ≤ n ∧ n ≤ 300 ∧ 
  n % 4 = 3 ∧ 
  n % 5 = 1 ∧ 
  n % 7 = 5 ∧
  n = 651 := by sorry

end NUMINAMATH_CALUDE_orchestra_members_count_l1384_138463


namespace NUMINAMATH_CALUDE_find_missing_number_l1384_138401

/-- Given two sets of numbers with known means, find the missing number in the second set. -/
theorem find_missing_number (x : ℝ) (missing : ℝ) : 
  (12 + x + 42 + 78 + 104) / 5 = 62 →
  (128 + 255 + 511 + missing + x) / 5 = 398.2 →
  missing = 1023 := by
sorry

end NUMINAMATH_CALUDE_find_missing_number_l1384_138401


namespace NUMINAMATH_CALUDE_frank_reading_total_l1384_138441

theorem frank_reading_total (book1_pages_per_day book1_days book2_pages_per_day book2_days book3_pages_per_day book3_days : ℕ) :
  book1_pages_per_day = 22 →
  book1_days = 569 →
  book2_pages_per_day = 35 →
  book2_days = 315 →
  book3_pages_per_day = 18 →
  book3_days = 450 →
  book1_pages_per_day * book1_days + book2_pages_per_day * book2_days + book3_pages_per_day * book3_days = 31643 :=
by
  sorry

end NUMINAMATH_CALUDE_frank_reading_total_l1384_138441


namespace NUMINAMATH_CALUDE_sum_remainder_mod_17_l1384_138413

theorem sum_remainder_mod_17 : ∃ k : ℕ, (78 + 79 + 80 + 81 + 82 + 83 + 84 + 85) = 17 * k + 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_17_l1384_138413


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1384_138430

theorem tan_alpha_plus_pi_fourth (α : Real) (h : Real.tan α = Real.sqrt 3) :
  Real.tan (α + π/4) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1384_138430


namespace NUMINAMATH_CALUDE_opposite_of_2023_l1384_138477

theorem opposite_of_2023 : 
  (∀ x : ℤ, x + 2023 = 0 → x = -2023) ∧ (-2023 + 2023 = 0) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l1384_138477


namespace NUMINAMATH_CALUDE_x_gt_1_sufficient_not_necessary_for_x_sq_gt_x_l1384_138482

theorem x_gt_1_sufficient_not_necessary_for_x_sq_gt_x :
  (∀ x : ℝ, x > 1 → x^2 > x) ∧ 
  (∃ x : ℝ, x^2 > x ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_1_sufficient_not_necessary_for_x_sq_gt_x_l1384_138482


namespace NUMINAMATH_CALUDE_remainder_when_c_divided_by_b_l1384_138476

theorem remainder_when_c_divided_by_b (a b c : ℕ) 
  (h1 : b = 3 * a + 3) 
  (h2 : c = 9 * a + 11) : 
  c % b = 2 := by
sorry

end NUMINAMATH_CALUDE_remainder_when_c_divided_by_b_l1384_138476


namespace NUMINAMATH_CALUDE_equal_slopes_iff_parallel_l1384_138419

-- Define the concept of a line in 2D space
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define what it means for two lines to be parallel
def are_parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

-- Define what it means for two lines to be distinct
def are_distinct (l1 l2 : Line) : Prop :=
  l1 ≠ l2

-- Theorem statement
theorem equal_slopes_iff_parallel (l1 l2 : Line) :
  are_distinct l1 l2 → (l1.slope = l2.slope ↔ are_parallel l1 l2) :=
sorry

end NUMINAMATH_CALUDE_equal_slopes_iff_parallel_l1384_138419


namespace NUMINAMATH_CALUDE_seed_testing_methods_eq_18_l1384_138457

/-- The number of ways to select and arrange seeds for testing -/
def seed_testing_methods : ℕ :=
  (Nat.choose 3 2) * (Nat.factorial 3)

/-- Theorem stating that the number of seed testing methods is 18 -/
theorem seed_testing_methods_eq_18 : seed_testing_methods = 18 := by
  sorry

end NUMINAMATH_CALUDE_seed_testing_methods_eq_18_l1384_138457


namespace NUMINAMATH_CALUDE_exactly_one_machine_maintenance_probability_l1384_138416

/-- The probability that exactly one of three independent machines needs maintenance,
    given their individual maintenance probabilities. -/
theorem exactly_one_machine_maintenance_probability
  (p_A p_B p_C : ℝ)
  (h_A : 0 ≤ p_A ∧ p_A ≤ 1)
  (h_B : 0 ≤ p_B ∧ p_B ≤ 1)
  (h_C : 0 ≤ p_C ∧ p_C ≤ 1)
  (h_p_A : p_A = 0.1)
  (h_p_B : p_B = 0.2)
  (h_p_C : p_C = 0.4) :
  p_A * (1 - p_B) * (1 - p_C) +
  (1 - p_A) * p_B * (1 - p_C) +
  (1 - p_A) * (1 - p_B) * p_C = 0.444 :=
sorry

end NUMINAMATH_CALUDE_exactly_one_machine_maintenance_probability_l1384_138416


namespace NUMINAMATH_CALUDE_amys_tomato_soup_cans_l1384_138442

/-- Amy's soup purchase problem -/
theorem amys_tomato_soup_cans (total_soups chicken_soups tomato_soups : ℕ) : 
  total_soups = 9 →
  chicken_soups = 6 →
  total_soups = chicken_soups + tomato_soups →
  tomato_soups = 3 := by
sorry

end NUMINAMATH_CALUDE_amys_tomato_soup_cans_l1384_138442


namespace NUMINAMATH_CALUDE_not_all_five_digit_extendable_all_one_digit_extendable_minimal_extension_l1384_138481

/-- Represents a natural number with a specified number of digits -/
def NDigitNumber (n : ℕ) := { x : ℕ // x ≥ 10^(n-1) ∧ x < 10^n }

/-- Checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- Appends k digits to an n-digit number -/
def append_digits (x : NDigitNumber n) (y : ℕ) (k : ℕ) : ℕ :=
  x.val * 10^k + y

theorem not_all_five_digit_extendable : ∃ x : NDigitNumber 6, 
  x.val ≥ 5 * 10^5 ∧ x.val < 6 * 10^5 ∧ 
  ¬∃ y : ℕ, y < 10^6 ∧ is_perfect_square (append_digits x y 6) :=
sorry

theorem all_one_digit_extendable : ∀ x : NDigitNumber 6, 
  x.val ≥ 10^5 ∧ x.val < 2 * 10^5 → 
  ∃ y : ℕ, y < 10^6 ∧ is_perfect_square (append_digits x y 6) :=
sorry

theorem minimal_extension (n : ℕ) : 
  (∀ x : NDigitNumber n, ∃ y : ℕ, y < 10^(n+1) ∧ is_perfect_square (append_digits x y (n+1))) ∧
  (∃ x : NDigitNumber n, ∀ y : ℕ, y < 10^n → ¬is_perfect_square (append_digits x y n)) :=
sorry

end NUMINAMATH_CALUDE_not_all_five_digit_extendable_all_one_digit_extendable_minimal_extension_l1384_138481


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_achieved_l1384_138455

theorem min_value_of_function (x : ℝ) : 
  (x^2 + 3) / Real.sqrt (x^2 + 2) ≥ 3 * Real.sqrt 2 / 2 :=
by sorry

theorem min_value_achieved : 
  ∃ x : ℝ, (x^2 + 3) / Real.sqrt (x^2 + 2) = 3 * Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_achieved_l1384_138455


namespace NUMINAMATH_CALUDE_expression_one_expression_two_expression_three_expression_four_l1384_138464

-- 1. 75 + 7 × 5
theorem expression_one : 75 + 7 * 5 = 110 := by sorry

-- 2. 148 - 48 ÷ 2
theorem expression_two : 148 - 48 / 2 = 124 := by sorry

-- 3. (400 - 160) ÷ 8
theorem expression_three : (400 - 160) / 8 = 30 := by sorry

-- 4. 4 × 25 × 7
theorem expression_four : 4 * 25 * 7 = 700 := by sorry

end NUMINAMATH_CALUDE_expression_one_expression_two_expression_three_expression_four_l1384_138464


namespace NUMINAMATH_CALUDE_total_tile_cost_l1384_138427

def courtyard_length : ℝ := 10
def courtyard_width : ℝ := 25
def tiles_per_sqft : ℝ := 4
def green_tile_percentage : ℝ := 0.4
def green_tile_cost : ℝ := 3
def red_tile_cost : ℝ := 1.5

theorem total_tile_cost : 
  let area := courtyard_length * courtyard_width
  let total_tiles := area * tiles_per_sqft
  let green_tiles := total_tiles * green_tile_percentage
  let red_tiles := total_tiles * (1 - green_tile_percentage)
  green_tiles * green_tile_cost + red_tiles * red_tile_cost = 2100 := by
sorry

end NUMINAMATH_CALUDE_total_tile_cost_l1384_138427


namespace NUMINAMATH_CALUDE_smallest_unsubmitted_integer_l1384_138454

/-- Represents the HMMT tournament --/
structure HMMTTournament where
  num_questions : ℕ
  max_expected_answer : ℕ

/-- Defines the property of an integer being submitted as an answer --/
def is_submitted (t : HMMTTournament) (n : ℕ) : Prop := sorry

/-- Theorem stating the smallest unsubmitted positive integer in the HMMT tournament --/
theorem smallest_unsubmitted_integer (t : HMMTTournament) 
  (h1 : t.num_questions = 66)
  (h2 : t.max_expected_answer = 150) :
  ∃ (N : ℕ), N = 139 ∧ 
  (∀ m : ℕ, m < N → is_submitted t m) ∧
  ¬(is_submitted t N) := by
  sorry

end NUMINAMATH_CALUDE_smallest_unsubmitted_integer_l1384_138454


namespace NUMINAMATH_CALUDE_monday_tuesday_widget_difference_l1384_138433

/-- The number of widgets David produces on Monday minus the number of widgets he produces on Tuesday -/
def widget_difference (w t : ℕ) : ℕ :=
  w * t - (w + 5) * (t - 3)

theorem monday_tuesday_widget_difference (t : ℕ) (h : t ≥ 3) :
  widget_difference (2 * t) t = t + 15 := by
  sorry

end NUMINAMATH_CALUDE_monday_tuesday_widget_difference_l1384_138433


namespace NUMINAMATH_CALUDE_total_sleep_is_53_l1384_138428

/-- Represents Janna's sleep schedule and calculates total weekly sleep hours -/
def weekly_sleep_hours : ℝ :=
  let weekday_sleep := 7
  let weekend_sleep := 8
  let nap_hours := 0.5
  let friday_extra := 1
  
  -- Monday, Wednesday
  2 * weekday_sleep +
  -- Tuesday, Thursday (with naps)
  2 * (weekday_sleep + nap_hours) +
  -- Friday (with extra hour)
  (weekday_sleep + friday_extra) +
  -- Saturday, Sunday
  2 * weekend_sleep

/-- Theorem stating that Janna's total sleep hours in a week is 53 -/
theorem total_sleep_is_53 : weekly_sleep_hours = 53 := by
  sorry

end NUMINAMATH_CALUDE_total_sleep_is_53_l1384_138428


namespace NUMINAMATH_CALUDE_arrangements_part1_arrangements_part2_arrangements_part3_arrangements_part4_arrangements_part5_arrangements_part6_l1384_138498

/- Given: 3 male students and 4 female students -/
def num_male : ℕ := 3
def num_female : ℕ := 4
def total_students : ℕ := num_male + num_female

/- Part 1: Select 5 people and arrange them in a row -/
theorem arrangements_part1 : (Nat.choose total_students 5) * (Nat.factorial 5) = 2520 := by sorry

/- Part 2: Arrange them in two rows, with 3 in the front row and 4 in the back row -/
theorem arrangements_part2 : (Nat.factorial 7) * (Nat.factorial 6) * (Nat.factorial 5) = 5040 := by sorry

/- Part 3: Arrange all of them in a row, with a specific person not standing at the head or tail of the row -/
theorem arrangements_part3 : (Nat.factorial 6) * 5 = 3600 := by sorry

/- Part 4: Arrange all of them in a row, with all female students standing together -/
theorem arrangements_part4 : (Nat.factorial 4) * (Nat.factorial 4) = 576 := by sorry

/- Part 5: Arrange all of them in a row, with male students not standing next to each other -/
theorem arrangements_part5 : (Nat.factorial 4) * (Nat.factorial 5) * (Nat.factorial 3) = 1440 := by sorry

/- Part 6: Arrange all of them in a row, with exactly 3 people between person A and person B -/
theorem arrangements_part6 : (Nat.factorial 5) * 2 * (Nat.factorial 2) = 720 := by sorry

end NUMINAMATH_CALUDE_arrangements_part1_arrangements_part2_arrangements_part3_arrangements_part4_arrangements_part5_arrangements_part6_l1384_138498


namespace NUMINAMATH_CALUDE_distance_on_line_l1384_138436

/-- Given two points on a line, prove that their distance is |x₁ - x₂|√(1 + k²) -/
theorem distance_on_line (k b x₁ x₂ : ℝ) :
  let y₁ := k * x₁ + b
  let y₂ := k * x₂ + b
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = |x₁ - x₂| * Real.sqrt (1 + k^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_on_line_l1384_138436


namespace NUMINAMATH_CALUDE_greatest_missed_problems_to_pass_l1384_138429

/-- The number of problems on the test -/
def total_problems : ℕ := 50

/-- The minimum percentage required to pass the test -/
def passing_percentage : ℚ := 85 / 100

/-- The greatest number of problems that can be missed while still passing -/
def max_missed_problems : ℕ := 7

theorem greatest_missed_problems_to_pass :
  max_missed_problems = 
    (total_problems - Int.floor (passing_percentage * total_problems : ℚ)) := by
  sorry

end NUMINAMATH_CALUDE_greatest_missed_problems_to_pass_l1384_138429


namespace NUMINAMATH_CALUDE_bamboo_pole_sections_l1384_138445

/-- Represents the properties of a bamboo pole with n sections -/
structure BambooPole (n : ℕ) where
  -- The common difference of the arithmetic sequence
  d : ℝ
  -- The number of sections is at least 6
  h_n_ge_6 : n ≥ 6
  -- The length of the top section is 10 cm
  h_top_length : 10 = 10
  -- The total length of the last three sections is 114 cm
  h_last_three : (10 + (n - 3) * d) + (10 + (n - 2) * d) + (10 + (n - 1) * d) = 114
  -- The length of the 6th section is the geometric mean of the lengths of the first and last sections
  h_geometric_mean : (10 + 5 * d)^2 = 10 * (10 + (n - 1) * d)

/-- The number of sections in the bamboo pole is 16 -/
theorem bamboo_pole_sections : ∃ (n : ℕ), ∃ (p : BambooPole n), n = 16 :=
sorry

end NUMINAMATH_CALUDE_bamboo_pole_sections_l1384_138445


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1384_138499

theorem rationalize_denominator (x : ℝ) (h : x^4 = 81) :
  1 / (x + x^(1/4)) = 1 / 27 :=
sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1384_138499


namespace NUMINAMATH_CALUDE_negation_distribution_l1384_138415

theorem negation_distribution (x y : ℝ) : -(x + y) = -x + -y := by sorry

end NUMINAMATH_CALUDE_negation_distribution_l1384_138415


namespace NUMINAMATH_CALUDE_triangle_property_l1384_138418

theorem triangle_property (a b c A B C : Real) (h1 : b = a * (Real.cos C - Real.sin C))
  (h2 : a = Real.sqrt 10) (h3 : Real.sin B = Real.sqrt 2 * Real.sin C) :
  A = 3 * Real.pi / 4 ∧ 1/2 * b * c * Real.sin A = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l1384_138418


namespace NUMINAMATH_CALUDE_empty_bucket_weight_l1384_138425

/-- Given a bucket with known weights at different fill levels, 
    calculate the weight of the empty bucket -/
theorem empty_bucket_weight (M N P : ℝ) 
  (h_full : ∃ x y : ℝ, x + y = M ∧ x + 3/4 * y = N ∧ x + 1/3 * y = P) : 
  ∃ x : ℝ, x = 4 * N - 3 * M := by
  sorry

end NUMINAMATH_CALUDE_empty_bucket_weight_l1384_138425


namespace NUMINAMATH_CALUDE_ben_win_probability_l1384_138494

theorem ben_win_probability (p_loss p_tie : ℚ) 
  (h_loss : p_loss = 5 / 12)
  (h_tie : p_tie = 1 / 6) :
  1 - p_loss - p_tie = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ben_win_probability_l1384_138494


namespace NUMINAMATH_CALUDE_max_silver_tokens_l1384_138446

/-- Represents the number of tokens of each color --/
structure TokenCount where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents an exchange booth --/
structure Booth where
  inputRed : ℕ
  inputBlue : ℕ
  outputSilver : ℕ
  outputRed : ℕ
  outputBlue : ℕ

/-- The initial token count --/
def initialTokens : TokenCount :=
  { red := 100, blue := 50, silver := 0 }

/-- The first exchange booth --/
def booth1 : Booth :=
  { inputRed := 3, inputBlue := 0, outputSilver := 1, outputRed := 0, outputBlue := 2 }

/-- The second exchange booth --/
def booth2 : Booth :=
  { inputRed := 0, inputBlue := 4, outputSilver := 1, outputRed := 2, outputBlue := 0 }

/-- Predicate to check if an exchange is possible --/
def canExchange (tokens : TokenCount) (booth : Booth) : Prop :=
  tokens.red ≥ booth.inputRed ∧ tokens.blue ≥ booth.inputBlue

/-- The final token count after all possible exchanges --/
noncomputable def finalTokens : TokenCount :=
  sorry

/-- Theorem stating that the maximum number of silver tokens is 103 --/
theorem max_silver_tokens : finalTokens.silver = 103 := by
  sorry

end NUMINAMATH_CALUDE_max_silver_tokens_l1384_138446


namespace NUMINAMATH_CALUDE_boric_acid_mixture_volume_l1384_138485

theorem boric_acid_mixture_volume 
  (volume_1_percent : ℝ) 
  (volume_5_percent : ℝ) 
  (h1 : volume_1_percent = 15) 
  (h2 : volume_5_percent = 15) : 
  volume_1_percent + volume_5_percent = 30 := by
  sorry

end NUMINAMATH_CALUDE_boric_acid_mixture_volume_l1384_138485


namespace NUMINAMATH_CALUDE_total_accepted_cartons_is_990_l1384_138456

/-- Represents the number of cartons delivered to a customer -/
def delivered_cartons (customer : Fin 5) : ℕ :=
  if customer.val < 2 then 300 else 200

/-- Represents the number of damaged cartons for a customer -/
def damaged_cartons (customer : Fin 5) : ℕ :=
  match customer.val with
  | 0 => 70
  | 1 => 50
  | 2 => 40
  | 3 => 30
  | 4 => 20
  | _ => 0  -- This case should never occur due to Fin 5

/-- Calculates the number of accepted cartons for a customer -/
def accepted_cartons (customer : Fin 5) : ℕ :=
  delivered_cartons customer - damaged_cartons customer

/-- The main theorem stating that the total number of accepted cartons is 990 -/
theorem total_accepted_cartons_is_990 :
  (Finset.sum Finset.univ accepted_cartons) = 990 := by
  sorry


end NUMINAMATH_CALUDE_total_accepted_cartons_is_990_l1384_138456


namespace NUMINAMATH_CALUDE_inequality_proof_l1384_138424

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1384_138424


namespace NUMINAMATH_CALUDE_no_perfect_square_sum_l1384_138468

theorem no_perfect_square_sum (n : ℕ) : n ≥ 1 → ¬∃ (m : ℕ), 2^n + 12^n + 2014^n = m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_sum_l1384_138468


namespace NUMINAMATH_CALUDE_jia_jia_clovers_l1384_138426

theorem jia_jia_clovers : 
  ∀ (total_leaves : ℕ) (four_leaf_clovers : ℕ) (three_leaf_clovers : ℕ),
  total_leaves = 100 →
  four_leaf_clovers = 1 →
  total_leaves = 4 * four_leaf_clovers + 3 * three_leaf_clovers →
  three_leaf_clovers = 32 := by
sorry

end NUMINAMATH_CALUDE_jia_jia_clovers_l1384_138426


namespace NUMINAMATH_CALUDE_min_sum_of_bases_l1384_138486

theorem min_sum_of_bases (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  (3 * a + 6 = 6 * b + 3) → (∀ x y : ℕ, x > 0 ∧ y > 0 ∧ 3 * x + 6 = 6 * y + 3 → a + b ≤ x + y) → 
  a + b = 20 := by sorry

end NUMINAMATH_CALUDE_min_sum_of_bases_l1384_138486


namespace NUMINAMATH_CALUDE_function_passes_through_point_l1384_138497

theorem function_passes_through_point (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 3
  f 1 = 4 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l1384_138497


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l1384_138438

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 7*x^2 + 7*x - 1

-- Define the roots
noncomputable def a : ℝ := 3 + Real.sqrt 8
noncomputable def b : ℝ := 3 - Real.sqrt 8

-- Theorem statement
theorem root_sum_reciprocal : 
  f a = 0 ∧ f b = 0 ∧ (∀ x, f x = 0 → b ≤ x ∧ x ≤ a) → a / b + b / a = 34 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l1384_138438


namespace NUMINAMATH_CALUDE_science_competition_accuracy_l1384_138406

theorem science_competition_accuracy (correct : ℕ) (wrong : ℕ) (target_accuracy : ℚ) (additional : ℕ) : 
  correct = 30 →
  wrong = 6 →
  target_accuracy = 85/100 →
  (correct + additional) / (correct + wrong + additional) = target_accuracy →
  additional = 4 := by
sorry

end NUMINAMATH_CALUDE_science_competition_accuracy_l1384_138406


namespace NUMINAMATH_CALUDE_function_property_l1384_138417

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x + 2

theorem function_property (a : ℝ) :
  (∀ x₁ ∈ Set.Icc 1 (Real.exp 1), ∃ x₂ ∈ Set.Icc 1 (Real.exp 1), f a x₁ + f a x₂ = 4) ↔
  a = Real.exp 1 + 1 :=
by sorry

end NUMINAMATH_CALUDE_function_property_l1384_138417


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l1384_138450

-- Define the types for lines and planes
def Line : Type := Unit
def Plane : Type := Unit

-- Define the parallel and perpendicular relations
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry

theorem line_plane_perpendicularity 
  (a b : Line) (α : Plane) 
  (h1 : parallel a α) 
  (h2 : perpendicular b α) : 
  perpendicular_lines a b :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l1384_138450


namespace NUMINAMATH_CALUDE_f_odd_a_range_l1384_138491

variable (f : ℝ → ℝ)

/-- f is an increasing function -/
axiom f_increasing : ∀ x y, x < y → f x < f y

/-- f satisfies the functional equation f(x+y) = f(x) + f(y) for all x, y ∈ ℝ -/
axiom f_add : ∀ x y, f (x + y) = f x + f y

/-- f is an odd function -/
theorem f_odd : ∀ x, f (-x) = -f x := by sorry

/-- The range of a for which f(x²) - 2f(x) < f(ax) - 2f(a) has exactly 3 positive integer solutions -/
theorem a_range : 
  {a : ℝ | 5 < a ∧ a ≤ 6} = 
  {a : ℝ | ∃! (x₁ x₂ x₃ : ℕ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    (∀ x : ℝ, x > 0 → (f (x^2) - 2*f x < f (a*x) - 2*f a ↔ x = x₁ ∨ x = x₂ ∨ x = x₃))} := by sorry

end NUMINAMATH_CALUDE_f_odd_a_range_l1384_138491


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l1384_138451

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 4) 
  (h2 : a^2 + b^2 = 80) : 
  a * b = 32 := by sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l1384_138451


namespace NUMINAMATH_CALUDE_necessary_sufficient_condition_for_ax0_eq_b_l1384_138407

theorem necessary_sufficient_condition_for_ax0_eq_b 
  (a b x₀ : ℝ) (h : a < 0) :
  (a * x₀ = b) ↔ 
  (∀ x : ℝ, (1/2) * a * x^2 - b * x ≤ (1/2) * a * x₀^2 - b * x₀) :=
by sorry

end NUMINAMATH_CALUDE_necessary_sufficient_condition_for_ax0_eq_b_l1384_138407


namespace NUMINAMATH_CALUDE_f_10_equals_756_l1384_138421

def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 5*x + 6

theorem f_10_equals_756 : f 10 = 756 := by sorry

end NUMINAMATH_CALUDE_f_10_equals_756_l1384_138421


namespace NUMINAMATH_CALUDE_computer_cost_l1384_138411

theorem computer_cost (C : ℝ) : 
  C + (1/5) * C + 300 = 2100 → C = 1500 := by
  sorry

end NUMINAMATH_CALUDE_computer_cost_l1384_138411


namespace NUMINAMATH_CALUDE_initial_value_exists_and_unique_l1384_138462

theorem initial_value_exists_and_unique : 
  ∃! x : ℤ, ∃ k : ℤ, x + 7 = k * 456 := by sorry

end NUMINAMATH_CALUDE_initial_value_exists_and_unique_l1384_138462


namespace NUMINAMATH_CALUDE_quadratic_one_solution_sum_l1384_138440

theorem quadratic_one_solution_sum (b : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + b * x + 5 * x + 12 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 + b * y + 5 * y + 12 = 0 → y = x) →
  (∃ c : ℝ, 3 * c^2 + (-b) * c + 5 * c + 12 = 0 ∧ 
   ∀ z : ℝ, 3 * z^2 + (-b) * z + 5 * z + 12 = 0 → z = c) →
  b + (-b) = -10 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_sum_l1384_138440


namespace NUMINAMATH_CALUDE_skill_player_water_consumption_l1384_138404

/-- Proves that skill position players drink 6 ounces each given the conditions of the football team's water consumption problem. -/
theorem skill_player_water_consumption
  (total_water : ℕ)
  (num_linemen : ℕ)
  (num_skill_players : ℕ)
  (lineman_consumption : ℕ)
  (num_skill_players_before_refill : ℕ)
  (h1 : total_water = 126)
  (h2 : num_linemen = 12)
  (h3 : num_skill_players = 10)
  (h4 : lineman_consumption = 8)
  (h5 : num_skill_players_before_refill = 5)
  : (total_water - num_linemen * lineman_consumption) / num_skill_players_before_refill = 6 := by
  sorry

#check skill_player_water_consumption

end NUMINAMATH_CALUDE_skill_player_water_consumption_l1384_138404


namespace NUMINAMATH_CALUDE_greatest_power_of_three_in_factorial_l1384_138402

theorem greatest_power_of_three_in_factorial :
  (∃ n : ℕ, n = 6 ∧ 
   ∀ k : ℕ, 3^k ∣ Nat.factorial 16 → k ≤ n) ∧
   3^6 ∣ Nat.factorial 16 :=
by sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_in_factorial_l1384_138402


namespace NUMINAMATH_CALUDE_same_color_probability_l1384_138431

def red_plates : ℕ := 6
def blue_plates : ℕ := 5
def total_plates : ℕ := red_plates + blue_plates
def plates_selected : ℕ := 3

theorem same_color_probability :
  (Nat.choose red_plates plates_selected + Nat.choose blue_plates plates_selected) / 
  Nat.choose total_plates plates_selected = 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l1384_138431


namespace NUMINAMATH_CALUDE_product_of_functions_l1384_138467

theorem product_of_functions (x : ℝ) (h : x > 0) :
  Real.sqrt (x * (x + 1)) * (1 / Real.sqrt x) = Real.sqrt (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_product_of_functions_l1384_138467


namespace NUMINAMATH_CALUDE_stream_speed_l1384_138410

/-- Proves that the speed of the stream is 6 kmph given the conditions --/
theorem stream_speed (boat_speed : ℝ) (stream_speed : ℝ) : 
  boat_speed = 18 →
  (1 / (boat_speed - stream_speed)) = (2 * (1 / (boat_speed + stream_speed))) →
  stream_speed = 6 := by
sorry

end NUMINAMATH_CALUDE_stream_speed_l1384_138410


namespace NUMINAMATH_CALUDE_pyramid_volume_l1384_138405

/-- Volume of a pyramid with a right triangular base -/
theorem pyramid_volume (c α β : ℝ) (hc : c > 0) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) :
  let volume := c^3 * Real.sin (2*α) * Real.tan β / 24
  ∃ (V : ℝ), V = volume ∧ V > 0 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_l1384_138405


namespace NUMINAMATH_CALUDE_problem_statement_l1384_138471

theorem problem_statement :
  (¬ (∃ x : ℝ, Real.tan x = 1 ∧ ∃ x : ℝ, x^2 - x + 1 ≤ 0)) ∧
  (∀ x : ℝ, x ≠ 1 → x^2 - 3*x + 2 ≠ 0) ↔ (∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1384_138471


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l1384_138472

def set_A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

def set_B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + (a^2-5) = 0}

theorem intersection_implies_a_value :
  ∀ a : ℝ, set_A ∩ set_B a = {2} → a = -1 ∨ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l1384_138472


namespace NUMINAMATH_CALUDE_baker_remaining_cakes_l1384_138420

/-- The number of cakes remaining after a sale -/
def cakes_remaining (initial : ℕ) (sold : ℕ) : ℕ := initial - sold

/-- Proof that the baker has 32 cakes remaining -/
theorem baker_remaining_cakes :
  let initial_cakes : ℕ := 169
  let sold_cakes : ℕ := 137
  cakes_remaining initial_cakes sold_cakes = 32 := by
sorry

end NUMINAMATH_CALUDE_baker_remaining_cakes_l1384_138420


namespace NUMINAMATH_CALUDE_candy_difference_l1384_138412

/-- Theorem about the difference in candy counts in a box of rainbow nerds -/
theorem candy_difference (purple : ℕ) (yellow : ℕ) (green : ℕ) (total : ℕ) : 
  purple = 10 →
  yellow = purple + 4 →
  total = 36 →
  purple + yellow + green = total →
  yellow - green = 2 := by
sorry

end NUMINAMATH_CALUDE_candy_difference_l1384_138412


namespace NUMINAMATH_CALUDE_poverty_education_relationship_l1384_138400

/-- Regression line for poverty and education data -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Define a point on the regression line -/
structure Point where
  x : ℝ
  y : ℝ

/-- The regression line equation -/
def on_regression_line (line : RegressionLine) (p : Point) : Prop :=
  p.y = line.slope * p.x + line.intercept

theorem poverty_education_relationship (line : RegressionLine) 
    (h_slope : line.slope = 0.8) (h_intercept : line.intercept = 4.6)
    (p1 p2 : Point) (h_on_line1 : on_regression_line line p1) 
    (h_on_line2 : on_regression_line line p2) (h_x_diff : p2.x - p1.x = 1) :
    p2.y - p1.y = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_poverty_education_relationship_l1384_138400


namespace NUMINAMATH_CALUDE_b_is_eighteen_l1384_138432

/-- Represents the ages of three people a, b, and c. -/
structure Ages where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.a = ages.b + 2 ∧
  ages.b = 2 * ages.c ∧
  ages.a + ages.b + ages.c = 47

/-- The theorem statement -/
theorem b_is_eighteen (ages : Ages) (h : satisfiesConditions ages) : ages.b = 18 := by
  sorry

end NUMINAMATH_CALUDE_b_is_eighteen_l1384_138432


namespace NUMINAMATH_CALUDE_medication_expiration_time_l1384_138478

-- Define the number of seconds in 8!
def medication_duration : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1

-- Define the number of seconds in a minute in this system
def seconds_per_minute : ℕ := 50

-- Define the release time
def release_time : String := "3 PM on February 14"

-- Define a function to calculate the expiration time
def calculate_expiration_time (duration : ℕ) (seconds_per_min : ℕ) (start_time : String) : String :=
  sorry

-- Theorem statement
theorem medication_expiration_time :
  calculate_expiration_time medication_duration seconds_per_minute release_time = "February 15, around 4 AM" :=
sorry

end NUMINAMATH_CALUDE_medication_expiration_time_l1384_138478


namespace NUMINAMATH_CALUDE_congruence_problem_l1384_138473

theorem congruence_problem (x : ℤ) : 
  (5 * x + 9) % 18 = 3 → (3 * x + 14) % 18 = 14 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l1384_138473


namespace NUMINAMATH_CALUDE_ancient_chinese_journey_l1384_138484

/-- Represents the distance walked on each day of a 6-day journey -/
structure JourneyDistances where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ
  day6 : ℝ

/-- The theorem statement for the ancient Chinese mathematical problem -/
theorem ancient_chinese_journey 
  (j : JourneyDistances) 
  (total_distance : j.day1 + j.day2 + j.day3 + j.day4 + j.day5 + j.day6 = 378)
  (day2_half : j.day2 = j.day1 / 2)
  (day3_half : j.day3 = j.day2 / 2)
  (day4_half : j.day4 = j.day3 / 2)
  (day5_half : j.day5 = j.day4 / 2)
  (day6_half : j.day6 = j.day5 / 2) :
  j.day3 = 48 := by
  sorry


end NUMINAMATH_CALUDE_ancient_chinese_journey_l1384_138484


namespace NUMINAMATH_CALUDE_supremum_of_expression_l1384_138403

theorem supremum_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ s : ℝ, s = -9/2 ∧ (- 1/(2*a) - 2/b ≤ s) ∧ 
  ∀ t : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → - 1/(2*x) - 2/y ≤ t) → s ≤ t :=
by sorry

end NUMINAMATH_CALUDE_supremum_of_expression_l1384_138403


namespace NUMINAMATH_CALUDE_highest_price_scheme_l1384_138495

theorem highest_price_scheme (n m : ℝ) (hn : 0 < n) (hnm : n < m) (hm : m < 100) :
  let price_A := 100 * (1 + m / 100) * (1 - n / 100)
  let price_B := 100 * (1 + n / 100) * (1 - m / 100)
  let price_C := 100 * (1 + (m + n) / 200) * (1 - (m + n) / 200)
  let price_D := 100 * (1 + m * n / 10000) * (1 - m * n / 10000)
  price_A ≥ price_B ∧ price_A ≥ price_C ∧ price_A ≥ price_D :=
by sorry

end NUMINAMATH_CALUDE_highest_price_scheme_l1384_138495


namespace NUMINAMATH_CALUDE_polynomial_property_l1384_138453

/-- Given a polynomial P(x) = x^4 + ax^3 + bx^2 + cx + d where a, b, c, d are constants,
    if P(1) = 1993, P(2) = 3986, and P(3) = 5979, then 1/4[P(11) + P(-7)] = 4693. -/
theorem polynomial_property (a b c d : ℝ) (P : ℝ → ℝ) 
    (h1 : ∀ x, P x = x^4 + a*x^3 + b*x^2 + c*x + d)
    (h2 : P 1 = 1993)
    (h3 : P 2 = 3986)
    (h4 : P 3 = 5979) :
    (1/4) * (P 11 + P (-7)) = 4693 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_property_l1384_138453


namespace NUMINAMATH_CALUDE_orphanage_donation_percentage_l1384_138469

def total_income : ℝ := 200000
def children_percentage : ℝ := 0.15
def num_children : ℕ := 3
def wife_percentage : ℝ := 0.30
def final_amount : ℝ := 40000

theorem orphanage_donation_percentage :
  let children_total_percentage := children_percentage * num_children
  let total_given_percentage := children_total_percentage + wife_percentage
  let remaining_amount := total_income * (1 - total_given_percentage)
  let donated_amount := remaining_amount - final_amount
  donated_amount / remaining_amount = 0.20 := by
sorry

end NUMINAMATH_CALUDE_orphanage_donation_percentage_l1384_138469


namespace NUMINAMATH_CALUDE_petyas_calculation_error_l1384_138409

theorem petyas_calculation_error (a : ℕ) (h1 : a > 2) : 
  ¬ (∃ (n : ℕ), 
    (a - 2) * (a + 3) - a = n ∧ 
    (∃ (k : ℕ), n.digits 10 = List.replicate 2023 8 ++ List.replicate 2023 3 ++ List.replicate k 0)) :=
by sorry

end NUMINAMATH_CALUDE_petyas_calculation_error_l1384_138409


namespace NUMINAMATH_CALUDE_calculate_at_20at_l1384_138466

-- Define the @ operation (postfix)
def at_post (x : ℤ) : ℤ := 9 - x

-- Define the @ operation (prefix)
def at_pre (x : ℤ) : ℤ := x - 9

-- Theorem statement
theorem calculate_at_20at : at_pre (at_post 20) = -20 := by
  sorry

end NUMINAMATH_CALUDE_calculate_at_20at_l1384_138466


namespace NUMINAMATH_CALUDE_reflection_line_equation_l1384_138489

/-- The equation of the reflection line for a triangle. -/
def reflection_line (D E F D' E' F' : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 0}

/-- Theorem stating the equation of the reflection line for the given triangle. -/
theorem reflection_line_equation
  (D : ℝ × ℝ) (E : ℝ × ℝ) (F : ℝ × ℝ)
  (D' : ℝ × ℝ) (E' : ℝ × ℝ) (F' : ℝ × ℝ)
  (hD : D = (1, 2)) (hE : E = (6, 3)) (hF : F = (-3, 4))
  (hD' : D' = (1, -2)) (hE' : E' = (6, -3)) (hF' : F' = (-3, -4)) :
  reflection_line D E F D' E' F' = {p : ℝ × ℝ | p.2 = 0} :=
sorry

end NUMINAMATH_CALUDE_reflection_line_equation_l1384_138489


namespace NUMINAMATH_CALUDE_problem_statement_l1384_138479

theorem problem_statement (x y z : ℝ) 
  (hx : x ≠ 1) (hy : y ≠ 1) (hxy : x ≠ y)
  (h : (y * z - x^2) / (1 - x) = (x * z - y^2) / (1 - y)) :
  (y * z - x^2) / (1 - x) = x + y + z := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1384_138479


namespace NUMINAMATH_CALUDE_simplify_expression_l1384_138470

theorem simplify_expression (x : ℝ) : 4*x - 3*x^2 + 6 + (8 - 5*x + 2*x^2) = -x^2 - x + 14 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1384_138470


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1384_138449

theorem max_value_of_expression (x y : ℝ) 
  (hx : |x - 1| ≤ 2) (hy : |y - 1| ≤ 2) : 
  ∃ (a b : ℝ), |a - 2*b + 1| = 6 ∧ |x - 2*y + 1| ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1384_138449


namespace NUMINAMATH_CALUDE_problem_statement_l1384_138435

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) : 
  (-1 < x - y ∧ x - y < 1) ∧ 
  ((1 / x + x / y) ≥ 3 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 1 ∧ 1 / a + a / b = 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1384_138435


namespace NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l1384_138448

def is_prime (n : ℕ) : Prop := sorry

def is_square (n : ℕ) : Prop := sorry

def has_prime_factor_less_than (n m : ℕ) : Prop := sorry

theorem smallest_non_prime_non_square_no_small_factors : 
  (∀ k < 4087, k > 0 → is_prime k ∨ is_square k ∨ has_prime_factor_less_than k 60) ∧
  ¬ is_prime 4087 ∧
  ¬ is_square 4087 ∧
  ¬ has_prime_factor_less_than 4087 60 := by sorry

end NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l1384_138448


namespace NUMINAMATH_CALUDE_lcm_sum_triplet_l1384_138492

theorem lcm_sum_triplet (a b c : ℕ+) :
  a + b + c = Nat.lcm (Nat.lcm a.val b.val) c.val ↔ b = 2 * a ∧ c = 3 * a := by
  sorry

end NUMINAMATH_CALUDE_lcm_sum_triplet_l1384_138492


namespace NUMINAMATH_CALUDE_equal_integers_from_cyclic_equation_l1384_138461

theorem equal_integers_from_cyclic_equation (n : ℕ+) (p : ℕ) (a b c : ℤ) 
  (hp : Nat.Prime p) 
  (h : a^n.val + p * b = b^n.val + p * c ∧ b^n.val + p * c = c^n.val + p * a) : 
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_equal_integers_from_cyclic_equation_l1384_138461


namespace NUMINAMATH_CALUDE_cos_two_pi_seventh_inequality_l1384_138439

theorem cos_two_pi_seventh_inequality (a : ℝ) (h : a = Real.cos (2 * Real.pi / 7)) : 
  2^(a - 1/2) < 2 * a := by sorry

end NUMINAMATH_CALUDE_cos_two_pi_seventh_inequality_l1384_138439


namespace NUMINAMATH_CALUDE_quadratic_value_l1384_138444

/-- A quadratic function with vertex (2,7) passing through (0,-7) -/
def f (x : ℝ) : ℝ :=
  let a : ℝ := -3.5
  a * (x - 2)^2 + 7

theorem quadratic_value : f 5 = -24.5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_value_l1384_138444


namespace NUMINAMATH_CALUDE_stability_promotion_criterion_l1384_138443

/-- Represents a rice variety with its yield statistics -/
structure RiceVariety where
  name : String
  average_yield : ℝ
  variance : ℝ

/-- Determines if a rice variety is more stable than another -/
def is_more_stable (a b : RiceVariety) : Prop :=
  a.variance < b.variance

/-- Determines if a rice variety is suitable for promotion based on stability -/
def suitable_for_promotion (a b : RiceVariety) : Prop :=
  is_more_stable a b

theorem stability_promotion_criterion 
  (a b : RiceVariety) 
  (h1 : a.average_yield = b.average_yield) 
  (h2 : a.variance < b.variance) : 
  suitable_for_promotion a b :=
sorry

end NUMINAMATH_CALUDE_stability_promotion_criterion_l1384_138443


namespace NUMINAMATH_CALUDE_distance_between_5th_and_25th_red_light_l1384_138414

/-- Represents the color of a light -/
inductive LightColor
| Red
| Green

/-- Represents the pattern of lights -/
def lightPattern : List LightColor :=
  [LightColor.Red, LightColor.Red, LightColor.Red, LightColor.Green, LightColor.Green]

/-- The distance between adjacent lights in inches -/
def lightDistance : ℕ := 8

/-- The number of inches in a foot -/
def inchesPerFoot : ℕ := 12

/-- The position of a red light given its index -/
def redLightPosition (n : ℕ) : ℕ :=
  (n - 1) / 3 * lightPattern.length + (n - 1) % 3 + 1

/-- The distance between two red lights given their indices -/
def distanceBetweenRedLights (n m : ℕ) : ℕ :=
  (redLightPosition m - redLightPosition n) * lightDistance

/-- The theorem to be proved -/
theorem distance_between_5th_and_25th_red_light :
  distanceBetweenRedLights 5 25 / inchesPerFoot = 56 := by
  sorry


end NUMINAMATH_CALUDE_distance_between_5th_and_25th_red_light_l1384_138414


namespace NUMINAMATH_CALUDE_average_age_problem_l1384_138496

theorem average_age_problem (total_students : Nat) (average_age : ℝ) 
  (group1_students : Nat) (group1_average : ℝ) (student12_age : ℝ) :
  total_students = 16 →
  average_age = 16 →
  group1_students = 5 →
  group1_average = 14 →
  student12_age = 42 →
  let remaining_students := total_students - group1_students - 1
  let total_age := average_age * total_students
  let group1_total_age := group1_average * group1_students
  let remaining_total_age := total_age - group1_total_age - student12_age
  remaining_total_age / remaining_students = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_age_problem_l1384_138496
