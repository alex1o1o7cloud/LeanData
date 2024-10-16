import Mathlib

namespace NUMINAMATH_CALUDE_greatest_common_multiple_under_120_l744_74479

theorem greatest_common_multiple_under_120 : 
  ∀ n : ℕ, n < 120 → n % 10 = 0 → n % 15 = 0 → n ≤ 90 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_under_120_l744_74479


namespace NUMINAMATH_CALUDE_parabola_intersection_l744_74416

/-- Prove that for a parabola y² = 2px (p > 0) with focus F on the x-axis, 
    if a line with slope angle π/4 passes through F and intersects the parabola at points A and B, 
    and the perpendicular bisector of AB passes through (0, 2), then p = 4/5. -/
theorem parabola_intersection (p : ℝ) (A B : ℝ × ℝ) :
  p > 0 →
  (∃ F : ℝ × ℝ, F.2 = 0 ∧ F.1 = p / 2) →
  (∀ x y : ℝ, y ^ 2 = 2 * p * x) →
  (∃ m b : ℝ, m = 1 ∧ A.2 = m * A.1 + b ∧ B.2 = m * B.1 + b) →
  (A.2 ^ 2 = 2 * p * A.1 ∧ B.2 ^ 2 = 2 * p * B.1) →
  ((A.1 + B.1) / 2 = 3 * p / 2 ∧ (A.2 + B.2) / 2 = p) →
  (∃ m' b' : ℝ, m' = -1 ∧ 2 = m' * 0 + b') →
  p = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_parabola_intersection_l744_74416


namespace NUMINAMATH_CALUDE_functional_equation_solution_l744_74424

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y z t : ℝ, (f x + f z) * (f y + f t) = f (x * y - z * t) + f (x * t + y * z)) →
  (∀ x : ℝ, f x = 0 ∨ f x = (1 : ℝ) / 2 ∨ f x = x ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l744_74424


namespace NUMINAMATH_CALUDE_square_sum_bound_l744_74435

theorem square_sum_bound (a b c : ℝ) :
  (|a^2 + b + c| + |a + b^2 - c| ≤ 1) → (a^2 + b^2 + c^2 < 100) := by
  sorry

end NUMINAMATH_CALUDE_square_sum_bound_l744_74435


namespace NUMINAMATH_CALUDE_circle_properties_l744_74457

/-- The equation of a circle in the form x^2 + y^2 + ax + by + c = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The center and radius of a circle -/
structure CircleProperties where
  center : ℝ × ℝ
  radius : ℝ

/-- Theorem: Given the circle equation x^2 + y^2 + 4x - 6y - 3 = 0,
    prove that its center is (-2, 3) and its radius is 4 -/
theorem circle_properties :
  let eq : CircleEquation := ⟨4, -6, -3⟩
  let props : CircleProperties := ⟨(-2, 3), 4⟩
  (∀ x y : ℝ, x^2 + y^2 + eq.a * x + eq.b * y + eq.c = 0 ↔ 
    (x - props.center.1)^2 + (y - props.center.2)^2 = props.radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l744_74457


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l744_74499

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 + a 5 + a 8 + a 11 = 48 →
  a 6 + a 7 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l744_74499


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l744_74491

def p (x : ℝ) : ℝ := 3*x^4 + 16*x^3 - 36*x^2 + 8*x

theorem roots_of_polynomial :
  ∃ (a b : ℝ), a^2 = 17 ∧
  (p 0 = 0) ∧
  (p (1/3) = 0) ∧
  (p (-3 + 2*a) = 0) ∧
  (p (-3 - 2*a) = 0) ∧
  (∀ x : ℝ, p x = 0 → x = 0 ∨ x = 1/3 ∨ x = -3 + 2*a ∨ x = -3 - 2*a) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l744_74491


namespace NUMINAMATH_CALUDE_magician_earned_four_dollars_l744_74439

/-- The amount earned by a magician selling magic card decks -/
def magician_earnings (initial_decks : ℕ) (remaining_decks : ℕ) (price_per_deck : ℕ) : ℕ :=
  (initial_decks - remaining_decks) * price_per_deck

/-- Theorem: The magician earned 4 dollars -/
theorem magician_earned_four_dollars :
  magician_earnings 5 3 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_magician_earned_four_dollars_l744_74439


namespace NUMINAMATH_CALUDE_sum_of_specific_pair_l744_74483

theorem sum_of_specific_pair : 
  ∃ (pairs : List (ℕ × ℕ)), 
    (pairs.length = 300) ∧ 
    (∀ (p : ℕ × ℕ), p ∈ pairs → 
      p.1 < 1500 ∧ p.2 < 1500 ∧ 
      p.2 = p.1 + 1 ∧ 
      (p.1 + p.2) % 5 = 0) ∧
    (57, 58) ∈ pairs →
    57 + 58 = 115 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_pair_l744_74483


namespace NUMINAMATH_CALUDE_library_books_count_l744_74467

theorem library_books_count :
  ∀ (total_books : ℕ),
    (total_books : ℝ) * 0.8 * 0.4 = 736 →
    total_books = 2300 :=
by
  sorry

end NUMINAMATH_CALUDE_library_books_count_l744_74467


namespace NUMINAMATH_CALUDE_distance_traveled_l744_74489

/-- Given a person traveling at a constant speed for a certain time,
    prove that the distance traveled is equal to the product of speed and time. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) (h1 : speed = 25) (h2 : time = 5) :
  speed * time = 125 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l744_74489


namespace NUMINAMATH_CALUDE_f_inequality_solution_f_minimum_value_l744_74454

def f (x : ℝ) := |x - 2| + |x + 1|

theorem f_inequality_solution (x : ℝ) :
  f x > 4 ↔ x < -1.5 ∨ x > 2.5 := by sorry

theorem f_minimum_value :
  ∃ (a : ℝ), (∀ x, f x ≥ a) ∧ (∀ b, (∀ x, f x ≥ b) → b ≤ a) ∧ a = 3 := by sorry

end NUMINAMATH_CALUDE_f_inequality_solution_f_minimum_value_l744_74454


namespace NUMINAMATH_CALUDE_probability_sum_5_is_one_ninth_l744_74406

/-- The number of possible outcomes when rolling two fair dice -/
def total_outcomes : ℕ := 36

/-- The number of favorable outcomes (sum of 5) when rolling two fair dice -/
def favorable_outcomes : ℕ := 4

/-- The probability of rolling a sum of 5 with two fair dice -/
def probability_sum_5 : ℚ := favorable_outcomes / total_outcomes

/-- Theorem stating that the probability of rolling a sum of 5 with two fair dice is 1/9 -/
theorem probability_sum_5_is_one_ninth : probability_sum_5 = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_5_is_one_ninth_l744_74406


namespace NUMINAMATH_CALUDE_projection_vector_a_on_b_l744_74476

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-1, 2)

theorem projection_vector_a_on_b :
  let proj_b_a := ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) • b
  proj_b_a = (-3/5, 6/5) := by sorry

end NUMINAMATH_CALUDE_projection_vector_a_on_b_l744_74476


namespace NUMINAMATH_CALUDE_smallest_prime_digit_sum_20_l744_74449

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

/-- Check if a natural number is prime -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- The smallest prime whose digits sum to 20 -/
theorem smallest_prime_digit_sum_20 :
  ∃ p : ℕ, is_prime p ∧ digit_sum p = 20 ∧
    ∀ q : ℕ, is_prime q → digit_sum q = 20 → p ≤ q :=
by
  use 299
  sorry

end NUMINAMATH_CALUDE_smallest_prime_digit_sum_20_l744_74449


namespace NUMINAMATH_CALUDE_smallest_abs_value_rational_l744_74408

theorem smallest_abs_value_rational : ∀ q : ℚ, |0| ≤ |q| := by
  sorry

end NUMINAMATH_CALUDE_smallest_abs_value_rational_l744_74408


namespace NUMINAMATH_CALUDE_divisible_by_five_count_is_correct_l744_74497

/-- The number of different positive, seven-digit integers divisible by 5,
    formed using the digits 2 (three times), 5 (two times), and 9 (two times) -/
def divisible_by_five_count : ℕ :=
  let total_digits : ℕ := 7
  let two_count : ℕ := 3
  let five_count : ℕ := 2
  let nine_count : ℕ := 2
  60

theorem divisible_by_five_count_is_correct :
  divisible_by_five_count = 60 := by sorry

end NUMINAMATH_CALUDE_divisible_by_five_count_is_correct_l744_74497


namespace NUMINAMATH_CALUDE_all_sheep_with_one_peasant_l744_74460

/-- Represents the state of sheep distribution among peasants -/
structure SheepDistribution where
  peasants : List Nat
  deriving Repr

/-- Represents a single expropriation event -/
def expropriation (dist : SheepDistribution) : SheepDistribution :=
  sorry

/-- The total number of sheep -/
def totalSheep : Nat := 128

/-- Theorem: After 7 expropriations, all sheep end up with one peasant -/
theorem all_sheep_with_one_peasant 
  (initial : SheepDistribution) 
  (h_total : initial.peasants.sum = totalSheep) 
  (h_expropriations : ∃ (d : SheepDistribution), 
    d = (expropriation^[7]) initial ∧ 
    d.peasants.length > 0) : 
  ∃ (final : SheepDistribution), 
    final = (expropriation^[7]) initial ∧ 
    final.peasants = [totalSheep] :=
sorry

end NUMINAMATH_CALUDE_all_sheep_with_one_peasant_l744_74460


namespace NUMINAMATH_CALUDE_expression_evaluation_l744_74490

theorem expression_evaluation (y : ℝ) (h : y ≠ 1/2) :
  (2*y - 1)^0 / (6⁻¹ + 2⁻¹) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l744_74490


namespace NUMINAMATH_CALUDE_sheet_width_l744_74417

/-- Given a rectangular sheet of paper with length 10 inches, a 1.5-inch margin all around,
    and a picture covering 38.5 square inches, prove that the width of the sheet is 8.5 inches. -/
theorem sheet_width (W : ℝ) (margin : ℝ) (picture_area : ℝ) : 
  margin = 1.5 →
  picture_area = 38.5 →
  (W - 2 * margin) * (10 - 2 * margin) = picture_area →
  W = 8.5 := by
sorry

end NUMINAMATH_CALUDE_sheet_width_l744_74417


namespace NUMINAMATH_CALUDE_division_scaling_l744_74431

theorem division_scaling (a b c : ℝ) (h : a / b = c) : (a / 100) / (b / 100) = c := by
  sorry

end NUMINAMATH_CALUDE_division_scaling_l744_74431


namespace NUMINAMATH_CALUDE_commodity_price_equality_l744_74409

/-- The year when commodity X costs 40 cents more than commodity Y -/
def target_year : ℕ := 2007

/-- The base year for price comparison -/
def base_year : ℕ := 2001

/-- The initial price of commodity X in dollars -/
def initial_price_X : ℚ := 4.20

/-- The initial price of commodity Y in dollars -/
def initial_price_Y : ℚ := 4.40

/-- The yearly price increase of commodity X in dollars -/
def price_increase_X : ℚ := 0.30

/-- The yearly price increase of commodity Y in dollars -/
def price_increase_Y : ℚ := 0.20

/-- The price difference between X and Y in the target year, in dollars -/
def price_difference : ℚ := 0.40

theorem commodity_price_equality :
  initial_price_X + price_increase_X * (target_year - base_year : ℚ) =
  initial_price_Y + price_increase_Y * (target_year - base_year : ℚ) + price_difference :=
by sorry

end NUMINAMATH_CALUDE_commodity_price_equality_l744_74409


namespace NUMINAMATH_CALUDE_polynomial_equality_l744_74468

theorem polynomial_equality (g : ℝ → ℝ) : 
  (∀ x, 5 * x^5 + 3 * x^3 - 4 * x + 2 + g x = 7 * x^3 - 9 * x^2 + x + 5) →
  (∀ x, g x = -5 * x^5 + 4 * x^3 - 9 * x^2 + 5 * x + 3) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_l744_74468


namespace NUMINAMATH_CALUDE_min_difference_f_g_l744_74418

noncomputable def f (x : ℝ) : ℝ := x^2

noncomputable def g (x : ℝ) : ℝ := Real.log x

theorem min_difference_f_g :
  ∃ (min_val : ℝ), min_val = 1/2 + 1/2 * Real.log 2 ∧
  ∀ (x : ℝ), x > 0 → |f x - g x| ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_difference_f_g_l744_74418


namespace NUMINAMATH_CALUDE_playlist_song_length_l744_74442

theorem playlist_song_length 
  (total_songs : Nat) 
  (song1_length : Nat) 
  (song2_length : Nat) 
  (total_playtime : Nat) 
  (playlist_repeats : Nat) 
  (h1 : total_songs = 3) 
  (h2 : song1_length = 3) 
  (h3 : song2_length = 3) 
  (h4 : total_playtime = 40) 
  (h5 : playlist_repeats = 5) :
  ∃ (song3_length : Nat), 
    song1_length + song2_length + song3_length = total_playtime / playlist_repeats ∧ 
    song3_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_playlist_song_length_l744_74442


namespace NUMINAMATH_CALUDE_sqrt_12_between_3_and_4_l744_74436

theorem sqrt_12_between_3_and_4 : 3 < Real.sqrt 12 ∧ Real.sqrt 12 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_between_3_and_4_l744_74436


namespace NUMINAMATH_CALUDE_poly_expansion_nonzero_terms_l744_74425

/-- The polynomial expression -/
def poly (x : ℝ) : ℝ := (2*x+5)*(3*x^2 - x + 4) - 4*(2*x^3 - 3*x^2 + x - 1)

/-- The expanded form of the polynomial -/
def expanded_poly (x : ℝ) : ℝ := 14*x^3 + x^2 + 7*x + 16

/-- The number of nonzero terms in the expanded polynomial -/
def num_nonzero_terms : ℕ := 4

theorem poly_expansion_nonzero_terms :
  (∀ x : ℝ, poly x = expanded_poly x) →
  num_nonzero_terms = 4 :=
by sorry

end NUMINAMATH_CALUDE_poly_expansion_nonzero_terms_l744_74425


namespace NUMINAMATH_CALUDE_johns_remaining_money_l744_74420

def remaining_money (initial amount_spent_on_sweets amount_given_to_each_friend : ℚ) : ℚ :=
  initial - (amount_spent_on_sweets + 2 * amount_given_to_each_friend)

theorem johns_remaining_money :
  remaining_money 10.50 2.25 2.20 = 3.85 := by
  sorry

end NUMINAMATH_CALUDE_johns_remaining_money_l744_74420


namespace NUMINAMATH_CALUDE_fish_pond_population_l744_74461

theorem fish_pond_population (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) 
  (h1 : initial_tagged = 70)
  (h2 : second_catch = 50)
  (h3 : tagged_in_second = 2)
  (h4 : (tagged_in_second : ℚ) / second_catch = initial_tagged / total_fish) :
  total_fish = 1750 := by
  sorry

#check fish_pond_population

end NUMINAMATH_CALUDE_fish_pond_population_l744_74461


namespace NUMINAMATH_CALUDE_geometric_sequence_a7_l744_74448

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a7 (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 2 * a 4 * a 5 = a 3 * a 6 →
  a 9 * a 10 = -8 →
  a 7 = -2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a7_l744_74448


namespace NUMINAMATH_CALUDE_correct_distribution_l744_74422

structure Participant where
  name : String
  initialDeposit : ℕ
  depositTime : ℕ

def totalValue : ℕ := 108000
def secondCarValue : ℕ := 48000
def secondCarSoldValue : ℕ := 42000

def calculateShare (p : Participant) (totalDays : ℕ) : ℚ :=
  (p.initialDeposit * p.depositTime : ℚ) / (totalDays * totalValue : ℚ)

def adjustedShare (share : ℚ) : ℚ :=
  share * (secondCarSoldValue : ℚ) / (secondCarValue : ℚ)

theorem correct_distribution 
  (istvan kalman laszlo miklos : Participant)
  (h1 : istvan.initialDeposit = 5000 + 4000 - 2500)
  (h2 : istvan.depositTime = 90)
  (h3 : kalman.initialDeposit = 4000)
  (h4 : kalman.depositTime = 70)
  (h5 : laszlo.initialDeposit = 2500)
  (h6 : laszlo.depositTime = 40)
  (h7 : miklos.initialDeposit = 2000)
  (h8 : miklos.depositTime = 90)
  : adjustedShare (calculateShare istvan 90) * secondCarValue = 54600 ∧
    adjustedShare (calculateShare kalman 90) * secondCarValue - 
    adjustedShare (calculateShare miklos 90) * secondCarValue = 7800 ∧
    adjustedShare (calculateShare laszlo 90) * secondCarValue = 10500 ∧
    adjustedShare (calculateShare miklos 90) * secondCarValue = 18900 := by
  sorry

#eval totalValue
#eval secondCarValue
#eval secondCarSoldValue

end NUMINAMATH_CALUDE_correct_distribution_l744_74422


namespace NUMINAMATH_CALUDE_inequalities_proof_l744_74423

theorem inequalities_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a^2 > b^2) ∧ (a^3 > b^3) ∧ (Real.sqrt (a - b) > Real.sqrt a - Real.sqrt b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l744_74423


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l744_74414

/-- The coefficient of x^3 in the expansion of (1-2x^2)(1+x)^4 is -4 -/
theorem coefficient_x_cubed_in_expansion : ∃ (p : Polynomial ℤ), 
  p = (1 - 2 * X^2) * (1 + X)^4 ∧ p.coeff 3 = -4 := by sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l744_74414


namespace NUMINAMATH_CALUDE_factorial_99_trailing_zeros_l744_74407

/-- The number of trailing zeros in n factorial -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 99! has 22 trailing zeros -/
theorem factorial_99_trailing_zeros :
  trailingZeros 99 = 22 := by
  sorry

end NUMINAMATH_CALUDE_factorial_99_trailing_zeros_l744_74407


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l744_74413

theorem cubic_equation_solution :
  ∀ (x y : ℤ), y^2 = x^3 - 3*x + 2 ↔ ∃ (k : ℕ), x = k^2 - 2 ∧ (y = k*(k^2 - 3) ∨ y = -k*(k^2 - 3)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l744_74413


namespace NUMINAMATH_CALUDE_student_divisor_error_l744_74478

theorem student_divisor_error (D : ℚ) (x : ℚ) : 
  D / 36 = 48 → D / x = 24 → x = 72 := by
  sorry

end NUMINAMATH_CALUDE_student_divisor_error_l744_74478


namespace NUMINAMATH_CALUDE_problem_solution_l744_74404

theorem problem_solution (a b : ℝ) (h : |a + 1| + (b - 2)^2 = 0) :
  (a + b)^9 + a^6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l744_74404


namespace NUMINAMATH_CALUDE_miles_travel_time_l744_74459

/-- Proves that if the distance between two cities is 57 miles and Miles takes 40 hours
    to complete 4 round trips, then Miles takes 10 hours for one round trip. -/
theorem miles_travel_time 
  (distance : ℝ) 
  (total_time : ℝ) 
  (num_round_trips : ℕ) 
  (h1 : distance = 57) 
  (h2 : total_time = 40) 
  (h3 : num_round_trips = 4) : 
  (total_time / num_round_trips) = 10 := by
  sorry


end NUMINAMATH_CALUDE_miles_travel_time_l744_74459


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l744_74419

-- Problem 1
theorem problem_1 : (1 * -5) + 9 = 4 := by sorry

-- Problem 2
theorem problem_2 : 12 - (-16) + (-2) - 1 = 25 := by sorry

-- Problem 3
theorem problem_3 : 6 / (-2) * (-1/3) = 1 := by sorry

-- Problem 4
theorem problem_4 : (-15) * (1/3 + 1/5) = -8 := by sorry

-- Problem 5
theorem problem_5 : (-2)^3 - (-8) / |-(4/3)| = -2 := by sorry

-- Problem 6
theorem problem_6 : -(1^2022) - (1/2 - 1/3) * 3 = -3/2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l744_74419


namespace NUMINAMATH_CALUDE_usd_share_change_l744_74444

/-- The change in the share of the US dollar in the NWF from 01.02.2021 to 01.04.2021 -/
theorem usd_share_change (total_nwf : ℝ) (other_currencies : ℝ) (initial_usd_share : ℝ) :
  total_nwf = 794.26 →
  other_currencies = 34.72 + 8.55 + 600.3 + 110.54 + 0.31 →
  initial_usd_share = 49.17 →
  ∃ (ε : ℝ), abs ε < 0.5 ∧
    (((total_nwf - other_currencies) / total_nwf * 100 - initial_usd_share) + ε = -44) := by
  sorry

end NUMINAMATH_CALUDE_usd_share_change_l744_74444


namespace NUMINAMATH_CALUDE_unique_prime_six_digit_number_l744_74415

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def six_digit_number (B : ℕ) : ℕ :=
  303100 + B

theorem unique_prime_six_digit_number :
  ∃! B : ℕ, B < 10 ∧ is_prime (six_digit_number B) ∧ six_digit_number B = 303101 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_six_digit_number_l744_74415


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_l744_74411

theorem sqrt_product_sqrt : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_l744_74411


namespace NUMINAMATH_CALUDE_sidney_monday_jj_l744_74458

/-- The number of jumping jacks Sidney did on Monday -/
def monday_jj : ℕ := sorry

/-- The number of jumping jacks Sidney did on Tuesday -/
def tuesday_jj : ℕ := 36

/-- The number of jumping jacks Sidney did on Wednesday -/
def wednesday_jj : ℕ := 40

/-- The number of jumping jacks Sidney did on Thursday -/
def thursday_jj : ℕ := 50

/-- The total number of jumping jacks Brooke did -/
def brooke_total : ℕ := 438

/-- Theorem stating that Sidney did 20 jumping jacks on Monday -/
theorem sidney_monday_jj : monday_jj = 20 := by
  sorry

end NUMINAMATH_CALUDE_sidney_monday_jj_l744_74458


namespace NUMINAMATH_CALUDE_exactly_one_true_l744_74472

-- Define the three propositions
def prop1 : Prop := ∀ x : ℝ, x^4 > x^2

def prop2 : Prop := (∀ p q : Prop, ¬(p ∧ q) → (¬p ∧ ¬q))

def prop3 : Prop := (¬(∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0))

-- Theorem statement
theorem exactly_one_true : (prop1 ∨ prop2 ∨ prop3) ∧ ¬(prop1 ∧ prop2) ∧ ¬(prop1 ∧ prop3) ∧ ¬(prop2 ∧ prop3) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_true_l744_74472


namespace NUMINAMATH_CALUDE_fraction_equality_l744_74405

theorem fraction_equality (q r s t : ℚ) 
  (h1 : q / r = 12)
  (h2 : s / r = 8)
  (h3 : s / t = 1 / 3) :
  t / q = 2 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l744_74405


namespace NUMINAMATH_CALUDE_coeff_x_cubed_in_expansion_coeff_x_cubed_proof_l744_74421

/-- The coefficient of x^3 in the expansion of (1-x)^10 is -120 -/
theorem coeff_x_cubed_in_expansion : Int :=
  -120

/-- The binomial coefficient (10 choose 3) -/
def binomial_coeff : Int :=
  120

theorem coeff_x_cubed_proof : coeff_x_cubed_in_expansion = -binomial_coeff := by
  sorry

end NUMINAMATH_CALUDE_coeff_x_cubed_in_expansion_coeff_x_cubed_proof_l744_74421


namespace NUMINAMATH_CALUDE_sin_alpha_value_l744_74450

theorem sin_alpha_value (α : Real) (h : Real.sin (Real.pi + α) = 0.2) : 
  Real.sin α = -0.2 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l744_74450


namespace NUMINAMATH_CALUDE_simplify_fraction_l744_74462

theorem simplify_fraction : 
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l744_74462


namespace NUMINAMATH_CALUDE_quadratic_root_property_l744_74427

theorem quadratic_root_property (a : ℝ) : 
  a^2 - a - 50 = 0 → a^4 - 101*a = 2550 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l744_74427


namespace NUMINAMATH_CALUDE_shara_shell_collection_l744_74498

/-- Calculates the total number of shells Shara has after her vacations -/
def total_shells (initial : ℕ) (vacation1 : ℕ) (vacation2 : ℕ) (vacation3 : ℕ) : ℕ :=
  initial + vacation1 + vacation2 + vacation3

/-- The number of shells Shara collected during her first vacation -/
def first_vacation : ℕ := 5 * 3 + 6

/-- The number of shells Shara collected during her second vacation -/
def second_vacation : ℕ := 4 * 2 + 7

/-- The number of shells Shara collected during her third vacation -/
def third_vacation : ℕ := 8 + 4 + 3 * 2

theorem shara_shell_collection :
  total_shells 20 first_vacation second_vacation third_vacation = 74 := by
  sorry

end NUMINAMATH_CALUDE_shara_shell_collection_l744_74498


namespace NUMINAMATH_CALUDE_tracy_art_fair_sales_l744_74402

theorem tracy_art_fair_sales : 
  let total_customers : ℕ := 20
  let first_group_size : ℕ := 4
  let second_group_size : ℕ := 12
  let third_group_size : ℕ := 4
  let first_group_paintings_per_customer : ℕ := 2
  let second_group_paintings_per_customer : ℕ := 1
  let third_group_paintings_per_customer : ℕ := 4
  first_group_size + second_group_size + third_group_size = total_customers →
  first_group_size * first_group_paintings_per_customer + 
  second_group_size * second_group_paintings_per_customer + 
  third_group_size * third_group_paintings_per_customer = 36 := by
sorry


end NUMINAMATH_CALUDE_tracy_art_fair_sales_l744_74402


namespace NUMINAMATH_CALUDE_house_transaction_result_l744_74494

def house_transaction (initial_value : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) : ℝ :=
  let first_sale := initial_value * (1 - loss_percent)
  let second_sale := first_sale * (1 + gain_percent)
  second_sale - initial_value

theorem house_transaction_result :
  house_transaction 12000 0.15 0.20 = -240 := by
  sorry

end NUMINAMATH_CALUDE_house_transaction_result_l744_74494


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l744_74493

theorem system_of_equations_solution (a b x y : ℝ) : 
  (2 * a - 3 * b = 13 ∧ 3 * a + 5 * b = 30.9) →
  (a = 8.3 ∧ b = 1.2) →
  (2 * (x + 2) - 3 * (y - 1) = 13 ∧ 3 * (x + 2) + 5 * (y - 1) = 30.9) →
  (x = 6.3 ∧ y = 2.2) :=
by sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l744_74493


namespace NUMINAMATH_CALUDE_parabola_unique_values_l744_74441

/-- A parabola passing through three given points -/
structure Parabola where
  b : ℝ
  c : ℝ
  eq : ℝ → ℝ := fun x ↦ x^2 + b*x + c
  point1 : eq (-2) = -8
  point2 : eq 4 = 28
  point3 : eq 1 = 4

/-- The unique values of b and c for the parabola -/
theorem parabola_unique_values (p : Parabola) : p.b = 4 ∧ p.c = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_unique_values_l744_74441


namespace NUMINAMATH_CALUDE_solution_set_when_a_neg_one_range_of_a_when_always_ge_three_l744_74455

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1: Solution set when a = -1
theorem solution_set_when_a_neg_one :
  {x : ℝ | f (-1) x ≥ 3} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} := by sorry

-- Part 2: Range of a when f(x) ≥ 3 for all x
theorem range_of_a_when_always_ge_three :
  {a : ℝ | ∀ x, f a x ≥ 3} = {a : ℝ | a ≤ -2 ∨ a ≥ 4} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_neg_one_range_of_a_when_always_ge_three_l744_74455


namespace NUMINAMATH_CALUDE_count_differing_blocks_l744_74485

/-- Represents the properties of a block -/
structure BlockProperties where
  material : Fin 3
  size : Fin 3
  color : Fin 4
  shape : Fin 5

/-- The reference block: metal medium blue hexagon -/
def referenceBlock : BlockProperties := {
  material := 2, -- Assuming 2 represents metal
  size := 1,     -- Assuming 1 represents medium
  color := 0,    -- Assuming 0 represents blue
  shape := 1     -- Assuming 1 represents hexagon
}

/-- Count of blocks differing in exactly two properties -/
def countDifferingBlocks : Nat :=
  let materialOptions := 2  -- Excluding the reference material
  let sizeOptions := 2      -- Excluding the reference size
  let colorOptions := 3     -- Excluding the reference color
  let shapeOptions := 4     -- Excluding the reference shape
  
  -- Sum of all combinations of choosing 2 properties to vary
  materialOptions * sizeOptions +
  materialOptions * colorOptions +
  materialOptions * shapeOptions +
  sizeOptions * colorOptions +
  sizeOptions * shapeOptions +
  colorOptions * shapeOptions

/-- Theorem stating the count of blocks differing in exactly two properties -/
theorem count_differing_blocks :
  countDifferingBlocks = 44 := by
  sorry


end NUMINAMATH_CALUDE_count_differing_blocks_l744_74485


namespace NUMINAMATH_CALUDE_sequence_well_defined_and_nonzero_l744_74470

def x : ℕ → ℚ
  | 0 => 2
  | n + 1 => 2 * x n / ((x n)^2 - 1)

def y : ℕ → ℚ
  | 0 => 4
  | n + 1 => 2 * y n / ((y n)^2 - 1)

def z : ℕ → ℚ
  | 0 => 6/7
  | n + 1 => 2 * z n / ((z n)^2 - 1)

theorem sequence_well_defined_and_nonzero (n : ℕ) :
  (x n ≠ 1 ∧ x n ≠ -1) ∧
  (y n ≠ 1 ∧ y n ≠ -1) ∧
  (z n ≠ 1 ∧ z n ≠ -1) ∧
  x n + y n + z n ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_sequence_well_defined_and_nonzero_l744_74470


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l744_74432

theorem least_number_for_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬((2496 + y) % 7 = 0 ∧ (2496 + y) % 11 = 0)) ∧ 
  (2496 + x) % 7 = 0 ∧ (2496 + x) % 11 = 0 → 
  x = 37 := by sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l744_74432


namespace NUMINAMATH_CALUDE_sum_of_fractions_l744_74443

theorem sum_of_fractions : 
  (2 / 10 : ℚ) + (3 / 10 : ℚ) + (5 / 10 : ℚ) + (6 / 10 : ℚ) + (7 / 10 : ℚ) + 
  (9 / 10 : ℚ) + (14 / 10 : ℚ) + (15 / 10 : ℚ) + (20 / 10 : ℚ) + (41 / 10 : ℚ) = 
  (122 / 10 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l744_74443


namespace NUMINAMATH_CALUDE_james_shoe_purchase_l744_74482

theorem james_shoe_purchase (price1 price2 : ℝ) : 
  price1 = 40 →
  price2 = 60 →
  let cheaper_price := min price1 price2
  let discounted_price2 := price2 - cheaper_price / 2
  let total_before_extra_discount := price1 + discounted_price2
  let extra_discount := total_before_extra_discount / 4
  let final_price := total_before_extra_discount - extra_discount
  final_price = 45 := by sorry

end NUMINAMATH_CALUDE_james_shoe_purchase_l744_74482


namespace NUMINAMATH_CALUDE_multiple_birth_statistics_l744_74471

theorem multiple_birth_statistics (total_babies : ℕ) 
  (twins triplets quadruplets quintuplets : ℕ) : 
  total_babies = 1200 →
  quintuplets = 2 * quadruplets →
  quadruplets = 3 * triplets →
  triplets = 2 * twins →
  2 * twins + 3 * triplets + 4 * quadruplets + 5 * quintuplets = total_babies →
  5 * quintuplets = 18000 / 23 := by
sorry

end NUMINAMATH_CALUDE_multiple_birth_statistics_l744_74471


namespace NUMINAMATH_CALUDE_depreciation_is_one_fourth_l744_74463

/-- Represents the depreciation of a scooter over one year -/
def scooter_depreciation (initial_value : ℚ) (value_after_one_year : ℚ) : ℚ :=
  (initial_value - value_after_one_year) / initial_value

/-- Theorem stating that for the given initial value and value after one year,
    the depreciation fraction is 1/4 -/
theorem depreciation_is_one_fourth :
  scooter_depreciation 40000 30000 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_depreciation_is_one_fourth_l744_74463


namespace NUMINAMATH_CALUDE_lucy_fish_purchase_l744_74400

/-- The number of fish Lucy needs to buy to reach her desired total -/
def fish_to_buy (initial : ℕ) (desired : ℕ) : ℕ := desired - initial

/-- Theorem: Lucy needs to buy 68 fish -/
theorem lucy_fish_purchase : fish_to_buy 212 280 = 68 := by
  sorry

end NUMINAMATH_CALUDE_lucy_fish_purchase_l744_74400


namespace NUMINAMATH_CALUDE_scaling_transformation_for_given_points_l744_74487

/-- A scaling transformation in 2D space -/
structure ScalingTransformation where
  sx : ℝ  -- scaling factor for x
  sy : ℝ  -- scaling factor for y

/-- Apply a scaling transformation to a point -/
def apply_scaling (t : ScalingTransformation) (p : ℝ × ℝ) : ℝ × ℝ :=
  (t.sx * p.1, t.sy * p.2)

theorem scaling_transformation_for_given_points :
  ∃ (t : ScalingTransformation),
    apply_scaling t (-2, 2) = (-6, 1) ∧
    t.sx = 3 ∧
    t.sy = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_scaling_transformation_for_given_points_l744_74487


namespace NUMINAMATH_CALUDE_sister_bars_count_l744_74401

/-- Represents the number of granola bars in a pack --/
def pack_size : ℕ := 20

/-- Represents the number of days in a week --/
def days_in_week : ℕ := 7

/-- Represents the number of bars traded to Pete --/
def bars_traded : ℕ := 3

/-- Represents the number of sisters --/
def num_sisters : ℕ := 2

/-- Calculates the number of granola bars each sister receives --/
def bars_per_sister : ℕ := (pack_size - days_in_week - bars_traded) / num_sisters

/-- Proves that each sister receives 5 granola bars --/
theorem sister_bars_count : bars_per_sister = 5 := by
  sorry

#eval bars_per_sister  -- This will evaluate to 5

end NUMINAMATH_CALUDE_sister_bars_count_l744_74401


namespace NUMINAMATH_CALUDE_train_station_wheels_l744_74480

theorem train_station_wheels :
  let num_trains : ℕ := 6
  let carriages_per_train : ℕ := 5
  let wheel_rows_per_carriage : ℕ := 4
  let wheels_per_row : ℕ := 6
  
  num_trains * carriages_per_train * wheel_rows_per_carriage * wheels_per_row = 720 :=
by
  sorry

end NUMINAMATH_CALUDE_train_station_wheels_l744_74480


namespace NUMINAMATH_CALUDE_lemonade_parts_l744_74475

/-- Punch mixture properties -/
structure PunchMixture where
  lemonade : ℕ
  cranberry : ℕ
  total_volume : ℕ
  ratio : ℚ

/-- Theorem: The number of parts of lemonade in the punch mixture is 12 -/
theorem lemonade_parts (p : PunchMixture) : p.lemonade = 12 :=
  by
  have h1 : p.cranberry = p.lemonade + 18 := sorry
  have h2 : p.ratio = p.lemonade / 5 := sorry
  have h3 : p.total_volume = 72 := sorry
  have h4 : p.lemonade + p.cranberry = p.total_volume := sorry
  sorry

#check lemonade_parts

end NUMINAMATH_CALUDE_lemonade_parts_l744_74475


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l744_74428

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) := {x : ℝ | f a b c x > 0}

-- Define the theorem
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : solution_set a b c = Set.Ioo (-1) 2) :
  a < 0 ∧
  a + b + c > 0 ∧
  solution_set b c (3*a) = Set.Iic (-3) ∪ Set.Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l744_74428


namespace NUMINAMATH_CALUDE_combined_average_marks_l744_74434

/-- Given two classes with the specified number of students and average marks,
    calculate the combined average mark of all students in both classes. -/
theorem combined_average_marks
  (n1 : ℕ) (n2 : ℕ) (avg1 : ℝ) (avg2 : ℝ)
  (h1 : n1 = 55) (h2 : n2 = 48) (h3 : avg1 = 60) (h4 : avg2 = 58) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℝ) = (55 * 60 + 48 * 58) / (55 + 48 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_combined_average_marks_l744_74434


namespace NUMINAMATH_CALUDE_exists_valid_nail_sequence_l744_74473

/-- Represents a nail operation -/
inductive NailOp
| Blue1 | Blue2 | Blue3
| Red1 | Red2 | Red3

/-- Represents a sequence of nail operations -/
def NailSeq := List NailOp

/-- Checks if a nail sequence becomes trivial when a specific operation is removed -/
def becomes_trivial_without (seq : NailSeq) (op : NailOp) : Prop := sorry

/-- Checks if a nail sequence becomes trivial when two specific operations are removed -/
def becomes_trivial_without_two (seq : NailSeq) (op1 op2 : NailOp) : Prop := sorry

/-- The main theorem stating the existence of a valid nail sequence -/
theorem exists_valid_nail_sequence :
  ∃ (W : NailSeq),
    (∀ blue : NailOp, blue ∈ [NailOp.Blue1, NailOp.Blue2, NailOp.Blue3] →
      becomes_trivial_without W blue) ∧
    (∀ red1 red2 : NailOp, red1 ≠ red2 →
      red1 ∈ [NailOp.Red1, NailOp.Red2, NailOp.Red3] →
      red2 ∈ [NailOp.Red1, NailOp.Red2, NailOp.Red3] →
      becomes_trivial_without_two W red1 red2) ∧
    (∀ red : NailOp, red ∈ [NailOp.Red1, NailOp.Red2, NailOp.Red3] →
      ¬becomes_trivial_without W red) :=
sorry

end NUMINAMATH_CALUDE_exists_valid_nail_sequence_l744_74473


namespace NUMINAMATH_CALUDE_max_notebooks_nine_notebooks_possible_l744_74477

/-- Represents the cost and quantity of an item --/
structure Item where
  cost : ℕ
  quantity : ℕ

/-- Represents a purchase of pens, pencils, and notebooks --/
structure Purchase where
  pens : Item
  pencils : Item
  notebooks : Item

/-- The total cost of a purchase --/
def totalCost (p : Purchase) : ℕ :=
  p.pens.cost * p.pens.quantity +
  p.pencils.cost * p.pencils.quantity +
  p.notebooks.cost * p.notebooks.quantity

/-- A purchase is valid if it meets the given conditions --/
def isValidPurchase (p : Purchase) : Prop :=
  p.pens.cost = 3 ∧
  p.pencils.cost = 4 ∧
  p.notebooks.cost = 10 ∧
  p.pens.quantity ≥ 1 ∧
  p.pencils.quantity ≥ 1 ∧
  p.notebooks.quantity ≥ 1 ∧
  totalCost p ≤ 100

theorem max_notebooks (p : Purchase) (h : isValidPurchase p) :
  p.notebooks.quantity ≤ 9 := by
  sorry

theorem nine_notebooks_possible :
  ∃ p : Purchase, isValidPurchase p ∧ p.notebooks.quantity = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_notebooks_nine_notebooks_possible_l744_74477


namespace NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_abs_x_eq_x_l744_74403

theorem x_positive_sufficient_not_necessary_for_abs_x_eq_x :
  (∀ x : ℝ, x > 0 → |x| = x) ∧
  (∃ x : ℝ, |x| = x ∧ ¬(x > 0)) := by
  sorry

end NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_abs_x_eq_x_l744_74403


namespace NUMINAMATH_CALUDE_specific_arithmetic_sequence_sum_l744_74492

/-- The sum of an arithmetic sequence with given parameters -/
def arithmeticSequenceSum (a1 : ℤ) (an : ℤ) (d : ℤ) : ℤ :=
  let n : ℤ := (an - a1) / d + 1
  n * (a1 + an) / 2

/-- Theorem: The sum of the specific arithmetic sequence is -440 -/
theorem specific_arithmetic_sequence_sum :
  arithmeticSequenceSum (-41) 1 2 = -440 := by
  sorry

end NUMINAMATH_CALUDE_specific_arithmetic_sequence_sum_l744_74492


namespace NUMINAMATH_CALUDE_max_y_coordinate_l744_74445

theorem max_y_coordinate (x y : ℝ) : 
  x^2 / 49 + (y - 3)^2 / 25 = 0 → y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_y_coordinate_l744_74445


namespace NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l744_74453

theorem pyramid_height_equals_cube_volume (cube_edge : ℝ) (pyramid_base : ℝ) (pyramid_height : ℝ) :
  cube_edge = 5 →
  pyramid_base = 10 →
  cube_edge ^ 3 = (1 / 3) * pyramid_base ^ 2 * pyramid_height →
  pyramid_height = 3.75 := by
sorry

end NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l744_74453


namespace NUMINAMATH_CALUDE_range_of_fraction_l744_74484

theorem range_of_fraction (a b : ℝ) (ha : 1 < a ∧ a < 3) (hb : 2 < b ∧ b < 4) :
  1/4 < a/b ∧ a/b < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_fraction_l744_74484


namespace NUMINAMATH_CALUDE_dress_sewing_time_l744_74496

/-- The time Allison and Al worked together on sewing dresses -/
def timeWorkedTogether (allisonRate alRate : ℚ) (allisonAloneTime : ℚ) : ℚ :=
  (1 - allisonRate * allisonAloneTime) / (allisonRate + alRate)

theorem dress_sewing_time : 
  let allisonRate : ℚ := 1/9
  let alRate : ℚ := 1/12
  let allisonAloneTime : ℚ := 15/4
  timeWorkedTogether allisonRate alRate allisonAloneTime = 3 := by
sorry

end NUMINAMATH_CALUDE_dress_sewing_time_l744_74496


namespace NUMINAMATH_CALUDE_base_r_transaction_l744_74466

/-- Represents a number in base r --/
def BaseR (digits : List Nat) (r : Nat) : Nat :=
  digits.foldr (fun d acc => d + r * acc) 0

/-- The problem statement --/
theorem base_r_transaction (r : Nat) : 
  (BaseR [5, 2, 1] r) + (BaseR [1, 1, 0] r) - (BaseR [3, 7, 1] r) = (BaseR [1, 0, 0, 2] r) →
  r^3 - 3*r^2 + 4*r - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_base_r_transaction_l744_74466


namespace NUMINAMATH_CALUDE_locus_of_right_angle_vertex_l744_74447

-- Define the right triangle
def RightTriangle (A B C : ℝ × ℝ) : Prop :=
  (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0

-- Define the perpendicular lines (x-axis and y-axis)
def OnXAxis (P : ℝ × ℝ) : Prop := P.2 = 0
def OnYAxis (P : ℝ × ℝ) : Prop := P.1 = 0

-- Define the locus of point C
def LocusC (C : ℝ × ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), RightTriangle A B C ∧ OnXAxis A ∧ OnYAxis B

-- State the theorem
theorem locus_of_right_angle_vertex :
  ∃ (S₁ S₂ : Set (ℝ × ℝ)), 
    (∀ C, LocusC C ↔ C ∈ S₁ ∨ C ∈ S₂) ∧ 
    (∃ (a b c d : ℝ × ℝ), S₁ = {x | ∃ t, 0 ≤ t ∧ t ≤ 1 ∧ x = (1-t) • a + t • b}) ∧
    (∃ (e f g h : ℝ × ℝ), S₂ = {x | ∃ t, 0 ≤ t ∧ t ≤ 1 ∧ x = (1-t) • e + t • f}) :=
sorry

end NUMINAMATH_CALUDE_locus_of_right_angle_vertex_l744_74447


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l744_74464

theorem quadratic_roots_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - Real.sqrt m * x₁ + 1 = 0 ∧ x₂^2 - Real.sqrt m * x₂ + 1 = 0) →
  m > 2 ∧
  ∃ m₀ : ℝ, m₀ > 2 ∧ ¬(∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - Real.sqrt m₀ * x₁ + 1 = 0 ∧ x₂^2 - Real.sqrt m₀ * x₂ + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l744_74464


namespace NUMINAMATH_CALUDE_final_value_properties_l744_74426

/-- Represents the transformation process on the blackboard -/
def blackboard_transform (n : ℕ) : Set ℕ → Set ℕ := sorry

/-- The final state of the blackboard after transformations -/
def final_state (n : ℕ) : Set ℕ := sorry

/-- Theorem stating the properties of the final value k -/
theorem final_value_properties (n : ℕ) (h : n ≥ 3) :
  ∃ (k t : ℕ), final_state n = {k} ∧ k = 2^t ∧ k ≥ n :=
sorry

end NUMINAMATH_CALUDE_final_value_properties_l744_74426


namespace NUMINAMATH_CALUDE_complex_parts_l744_74488

theorem complex_parts (z : ℂ) (h : z = 2 - 3*I) : 
  z.re = 2 ∧ z.im = -3 := by sorry

end NUMINAMATH_CALUDE_complex_parts_l744_74488


namespace NUMINAMATH_CALUDE_milk_students_l744_74440

theorem milk_students (total_students : ℕ) (soda_students : ℕ) (milk_percent : ℚ) (soda_percent : ℚ) :
  soda_percent = 1/2 →
  milk_percent = 3/10 →
  soda_students = 90 →
  (milk_percent / soda_percent) * soda_students = 54 :=
by sorry

end NUMINAMATH_CALUDE_milk_students_l744_74440


namespace NUMINAMATH_CALUDE_jordan_running_time_l744_74456

/-- Given Steve's running time and distance, and Jordan's relative speed,
    calculate Jordan's time to run a specified distance. -/
theorem jordan_running_time
  (steve_distance : ℝ)
  (steve_time : ℝ)
  (jordan_relative_speed : ℝ)
  (jordan_distance : ℝ)
  (h1 : steve_distance = 6)
  (h2 : steve_time = 36)
  (h3 : jordan_relative_speed = 3)
  (h4 : jordan_distance = 8) :
  (jordan_distance * steve_time) / (steve_distance * jordan_relative_speed) = 24 :=
by sorry

end NUMINAMATH_CALUDE_jordan_running_time_l744_74456


namespace NUMINAMATH_CALUDE_hypotenuse_area_change_l744_74486

theorem hypotenuse_area_change
  (a b c : ℝ)
  (h_right_triangle : a^2 + b^2 = c^2)
  (h_area_increase : (a - 5) * (b + 5) / 2 = a * b / 2 + 5)
  : c^2 - ((a - 5)^2 + (b + 5)^2) = 20 :=
by sorry

end NUMINAMATH_CALUDE_hypotenuse_area_change_l744_74486


namespace NUMINAMATH_CALUDE_fraction_reciprocal_product_l744_74412

theorem fraction_reciprocal_product : (1 / (5 / 3)) * (5 / 3) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_reciprocal_product_l744_74412


namespace NUMINAMATH_CALUDE_units_digit_of_fraction_l744_74469

def numerator : ℕ := 15 * 16 * 17 * 18 * 19 * 20
def denominator : ℕ := 500

theorem units_digit_of_fraction : 
  (numerator / denominator) % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_fraction_l744_74469


namespace NUMINAMATH_CALUDE_policeman_speed_l744_74446

/-- Calculates the speed of a policeman chasing a criminal given the initial conditions --/
theorem policeman_speed
  (initial_distance : ℝ)
  (criminal_speed : ℝ)
  (time_elapsed : ℝ)
  (final_distance : ℝ)
  (h1 : initial_distance = 180)
  (h2 : criminal_speed = 8)
  (h3 : time_elapsed = 5 / 60)
  (h4 : final_distance = 96.66666666666667)
  : ∃ (policeman_speed : ℝ), policeman_speed = 1000 := by
  sorry

#check policeman_speed

end NUMINAMATH_CALUDE_policeman_speed_l744_74446


namespace NUMINAMATH_CALUDE_regional_math_competition_l744_74438

theorem regional_math_competition (initial_contestants : ℕ) : 
  (initial_contestants : ℚ) * (2/5) * (1/2) = 30 → initial_contestants = 150 := by
  sorry

end NUMINAMATH_CALUDE_regional_math_competition_l744_74438


namespace NUMINAMATH_CALUDE_harry_potion_kits_l744_74410

/-- The number of spellbooks Harry needs to buy -/
def num_spellbooks : ℕ := 5

/-- The cost of one spellbook in gold -/
def cost_spellbook : ℕ := 5

/-- The cost of one potion kit in silver -/
def cost_potion_kit : ℕ := 20

/-- The cost of one owl in gold -/
def cost_owl : ℕ := 28

/-- The number of silver in one gold -/
def silver_per_gold : ℕ := 9

/-- The total amount Harry will pay in silver -/
def total_cost : ℕ := 537

/-- The number of potion kits Harry needs to buy -/
def num_potion_kits : ℕ := (total_cost - (num_spellbooks * cost_spellbook * silver_per_gold + cost_owl * silver_per_gold)) / cost_potion_kit

theorem harry_potion_kits : num_potion_kits = 3 := by
  sorry

end NUMINAMATH_CALUDE_harry_potion_kits_l744_74410


namespace NUMINAMATH_CALUDE_quadratic_complete_square_l744_74451

/-- Given a quadratic equation 16x^2 + 32x - 1280 = 0, prove that when rewritten
    in the form (x + r)^2 = s by completing the square, the value of s is 81. -/
theorem quadratic_complete_square (x : ℝ) :
  (16 * x^2 + 32 * x - 1280 = 0) →
  ∃ (r s : ℝ), ((x + r)^2 = s ∧ s = 81) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l744_74451


namespace NUMINAMATH_CALUDE_savings_calculation_l744_74429

/-- Calculates a person's savings given their income and income-to-expenditure ratio --/
def calculate_savings (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Theorem: Given an income of 18000 and an income-to-expenditure ratio of 5:4, the savings are 3600 --/
theorem savings_calculation :
  calculate_savings 18000 5 4 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l744_74429


namespace NUMINAMATH_CALUDE_additional_emails_per_day_l744_74437

theorem additional_emails_per_day 
  (initial_emails_per_day : ℕ)
  (total_days : ℕ)
  (subscription_day : ℕ)
  (total_emails : ℕ)
  (h1 : initial_emails_per_day = 20)
  (h2 : total_days = 30)
  (h3 : subscription_day = 15)
  (h4 : total_emails = 675) :
  ∃ (additional_emails : ℕ),
    additional_emails = 5 ∧
    total_emails = initial_emails_per_day * subscription_day + 
      (initial_emails_per_day + additional_emails) * (total_days - subscription_day) :=
by sorry

end NUMINAMATH_CALUDE_additional_emails_per_day_l744_74437


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l744_74433

/-- Theorem: For a parallelogram with area 216 cm² and height 18 cm, the base length is 12 cm. -/
theorem parallelogram_base_length
  (area : ℝ) (height : ℝ) (base : ℝ)
  (h_area : area = 216)
  (h_height : height = 18)
  (h_parallelogram : area = base * height) :
  base = 12 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l744_74433


namespace NUMINAMATH_CALUDE_two_digit_numbers_problem_l744_74474

theorem two_digit_numbers_problem : ∃ (a b : ℕ), 
  10 ≤ a ∧ a < 100 ∧  -- a is a two-digit number
  10 ≤ b ∧ b < 100 ∧  -- b is a two-digit number
  a = 2 * b ∧         -- a is double b
  (a / 10 ≠ b / 10) ∧ (a / 10 ≠ b % 10) ∧ (a % 10 ≠ b / 10) ∧ (a % 10 ≠ b % 10) ∧  -- no common digits
  (a / 10 + a % 10 = b / 10) ∧  -- sum of digits of a equals tens digit of b
  (a % 10 - a / 10 = b % 10) ∧  -- difference of digits of a equals ones digit of b
  a = 34 ∧ b = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_two_digit_numbers_problem_l744_74474


namespace NUMINAMATH_CALUDE_trader_gain_percentage_l744_74452

/-- The gain percentage of a trader selling pens -/
def gain_percentage (num_sold : ℕ) (num_gain : ℕ) : ℚ :=
  (num_gain : ℚ) / (num_sold : ℚ) * 100

/-- Theorem: The trader's gain percentage is 33.33% -/
theorem trader_gain_percentage : 
  ∃ (ε : ℚ), abs (gain_percentage 90 30 - 100/3) < ε ∧ ε < 1/100 := by
  sorry

end NUMINAMATH_CALUDE_trader_gain_percentage_l744_74452


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l744_74430

theorem complex_fraction_simplification :
  (2 - Complex.I) / (1 + 2 * Complex.I) = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l744_74430


namespace NUMINAMATH_CALUDE_fewer_twos_to_hundred_l744_74481

theorem fewer_twos_to_hundred : (222 / 2 - 22 / 2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_fewer_twos_to_hundred_l744_74481


namespace NUMINAMATH_CALUDE_zlatoust_miass_distance_l744_74465

theorem zlatoust_miass_distance :
  ∀ (g m k : ℝ),  -- speeds of GAZ, MAZ, and KamAZ
  (∀ x : ℝ, 
    (x + 18) / k = (x - 18) / m ∧
    (x + 25) / k = (x - 25) / g ∧
    (x + 8) / m = (x - 8) / g) →
  ∃ x : ℝ, x = 60 ∧ 
    (x + 18) / k = (x - 18) / m ∧
    (x + 25) / k = (x - 25) / g ∧
    (x + 8) / m = (x - 8) / g :=
by sorry

end NUMINAMATH_CALUDE_zlatoust_miass_distance_l744_74465


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l744_74495

theorem quadratic_equation_solution (x c d : ℕ) (h1 : x^2 + 14*x = 72) 
  (h2 : x = Int.sqrt c - d) (h3 : 0 < c) (h4 : 0 < d) : c + d = 128 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l744_74495
