import Mathlib

namespace NUMINAMATH_CALUDE_union_of_A_and_B_l596_59639

def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B : Set ℝ := {x | x > -2 ∧ x < 0}

theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | x > -2 ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l596_59639


namespace NUMINAMATH_CALUDE_second_tap_empty_time_l596_59666

/-- The time it takes for the second tap to empty the cistern -/
def T : ℝ := 8

/-- The time it takes for the first tap to fill the cistern -/
def fill_time : ℝ := 3

/-- The time it takes to fill the cistern when both taps are open -/
def both_open_time : ℝ := 4.8

/-- Theorem stating that T is the correct time for the second tap to empty the cistern -/
theorem second_tap_empty_time : 
  (1 / fill_time) - (1 / T) = (1 / both_open_time) :=
sorry

end NUMINAMATH_CALUDE_second_tap_empty_time_l596_59666


namespace NUMINAMATH_CALUDE_fraction_ordering_l596_59661

theorem fraction_ordering : 6 / 22 < 5 / 17 ∧ 5 / 17 < 8 / 24 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l596_59661


namespace NUMINAMATH_CALUDE_quartic_roots_l596_59663

/-- Given a quadratic equation x² + px + q = 0 with roots x₁ and x₂,
    prove that x⁴ - (p² - 2q)x² + q² = 0 has roots x₁, x₂, -x₁, -x₂ -/
theorem quartic_roots (p q x₁ x₂ : ℝ) : 
  (x₁^2 + p*x₁ + q = 0) → 
  (x₂^2 + p*x₂ + q = 0) → 
  (∀ x : ℝ, x^4 - (p^2 - 2*q)*x^2 + q^2 = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = -x₁ ∨ x = -x₂) :=
by sorry

end NUMINAMATH_CALUDE_quartic_roots_l596_59663


namespace NUMINAMATH_CALUDE_set_of_positive_rationals_l596_59682

theorem set_of_positive_rationals (S : Set ℚ) 
  (closure : ∀ a b : ℚ, a ∈ S → b ∈ S → (a + b) ∈ S ∧ (a * b) ∈ S)
  (trichotomy : ∀ r : ℚ, (r ∈ S ∧ -r ∉ S ∧ r ≠ 0) ∨ (-r ∈ S ∧ r ∉ S ∧ r ≠ 0) ∨ (r = 0 ∧ r ∉ S ∧ -r ∉ S)) :
  S = {r : ℚ | 0 < r} := by
sorry

end NUMINAMATH_CALUDE_set_of_positive_rationals_l596_59682


namespace NUMINAMATH_CALUDE_equation_holds_l596_59698

/-- Prove that for positive integers a, b, c with given conditions, 
    the equation (12a + b)(12a + c) = 144a(a + 1) + b + c holds. -/
theorem equation_holds (a b c : ℕ) 
  (ha : a < 12) (hb : b < 12) (hc : c < 12) (hbc : b + c = 12) : 
  (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b + c := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_l596_59698


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l596_59653

/-- Given a geometric sequence {a_n}, if a_1 + a_2 = 20 and a_3 + a_4 = 80, then a_5 + a_6 = 320 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 + a 2 = 20 →                           -- first condition
  a 3 + a 4 = 80 →                           -- second condition
  a 5 + a 6 = 320 := by                      -- conclusion
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l596_59653


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l596_59664

theorem quadratic_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 36 * x + c = 0) →
  (a + c = 37) →
  (a < c) →
  (a = (37 - Real.sqrt 73) / 2 ∧ c = (37 + Real.sqrt 73) / 2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l596_59664


namespace NUMINAMATH_CALUDE_stripe_area_on_cylindrical_silo_l596_59689

/-- The area of a stripe on a cylindrical silo -/
theorem stripe_area_on_cylindrical_silo 
  (d h w : ℝ) -- d: diameter, h: height, w: stripe width
  (n : ℕ) -- number of revolutions
  (h_d : d = 40)
  (h_h : h = 120)
  (h_w : w = 4)
  (h_n : n = 3) :
  w * n * (h / n) * Real.sqrt ((π * d)^2 + (h / n)^2) = 480 * Real.sqrt (π^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_stripe_area_on_cylindrical_silo_l596_59689


namespace NUMINAMATH_CALUDE_train_crossing_time_l596_59613

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_time (train_length platform_length : ℝ) (time_cross_pole : ℝ) :
  train_length = 900 →
  platform_length = 1050 →
  time_cross_pole = 18 →
  (train_length + platform_length) / (train_length / time_cross_pole) = 39 := by
  sorry


end NUMINAMATH_CALUDE_train_crossing_time_l596_59613


namespace NUMINAMATH_CALUDE_greatest_distance_between_sets_l596_59690

def set_A : Set ℂ := {z : ℂ | z^4 - 16 = 0}
def set_B : Set ℂ := {z : ℂ | z^4 - 16*z^3 + 64*z^2 - 16*z + 16 = 0}

theorem greatest_distance_between_sets : 
  ∃ (a : ℂ) (b : ℂ), a ∈ set_A ∧ b ∈ set_B ∧ 
    (∀ (x : ℂ) (y : ℂ), x ∈ set_A → y ∈ set_B → Complex.abs (x - y) ≤ Complex.abs (a - b)) ∧
    Complex.abs (a - b) = 3 :=
sorry

end NUMINAMATH_CALUDE_greatest_distance_between_sets_l596_59690


namespace NUMINAMATH_CALUDE_expression_evaluation_l596_59629

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -3
  let z : ℝ := 1
  2 * x^2 + 3 * y^2 - 4 * z^2 + 5 * x * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l596_59629


namespace NUMINAMATH_CALUDE_star_expression_equals_six_l596_59645

-- Define the new * operation
def star (a b : ℝ) : ℝ := (a + 1) * (b - 1)

-- Define the *2 operation
def star_squared (a : ℝ) : ℝ := star a a

-- Theorem statement
theorem star_expression_equals_six :
  let x : ℝ := 2
  star 3 (star_squared x) - star 2 x + 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_star_expression_equals_six_l596_59645


namespace NUMINAMATH_CALUDE_blue_candy_probability_l596_59672

/-- The probability of selecting a blue candy from a bag with green, blue, and red candies. -/
theorem blue_candy_probability
  (green : ℕ) (blue : ℕ) (red : ℕ)
  (h_green : green = 5)
  (h_blue : blue = 3)
  (h_red : red = 4) :
  (blue : ℚ) / (green + blue + red) = 1 / 4 :=
sorry

end NUMINAMATH_CALUDE_blue_candy_probability_l596_59672


namespace NUMINAMATH_CALUDE_count_five_digit_numbers_with_eight_l596_59667

/-- The number of five-digit numbers in decimal notation containing at least one digit 8 -/
def fiveDigitNumbersWithEight : ℕ := 37512

/-- The total number of five-digit numbers in decimal notation -/
def totalFiveDigitNumbers : ℕ := 90000

/-- The number of five-digit numbers in decimal notation not containing the digit 8 -/
def fiveDigitNumbersWithoutEight : ℕ := 52488

/-- Theorem stating that the number of five-digit numbers containing at least one digit 8
    is equal to the total number of five-digit numbers minus the number of five-digit numbers
    not containing the digit 8 -/
theorem count_five_digit_numbers_with_eight :
  fiveDigitNumbersWithEight = totalFiveDigitNumbers - fiveDigitNumbersWithoutEight :=
by sorry

end NUMINAMATH_CALUDE_count_five_digit_numbers_with_eight_l596_59667


namespace NUMINAMATH_CALUDE_inequality_proof_l596_59676

theorem inequality_proof (x y a b : ℝ) 
  (h1 : x^2 + y^2 = 1) 
  (h2 : a > 0) 
  (h3 : b > 0) : 
  Real.sqrt (a^2 * x^2 + b^2 * y^2) + Real.sqrt (a^2 * y^2 + b^2 * x^2) ≥ a + b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l596_59676


namespace NUMINAMATH_CALUDE_ball_hitting_ground_time_l596_59600

theorem ball_hitting_ground_time : 
  let f (t : ℝ) := -20 * t^2 + 30 * t + 60
  ∃ t : ℝ, t > 0 ∧ f t = 0 ∧ t = (3 + Real.sqrt 57) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ball_hitting_ground_time_l596_59600


namespace NUMINAMATH_CALUDE_segments_form_triangle_l596_59651

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle 
    must be greater than the length of the remaining side -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A set of line segments can form a triangle if they satisfy the triangle inequality -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- The set of line segments 5cm, 8cm, and 12cm can form a triangle -/
theorem segments_form_triangle : can_form_triangle 5 8 12 := by
  sorry

end NUMINAMATH_CALUDE_segments_form_triangle_l596_59651


namespace NUMINAMATH_CALUDE_angle_addition_theorem_l596_59671

/-- Represents an angle in degrees and minutes -/
structure Angle where
  degrees : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Addition of two angles -/
def add_angles (a b : Angle) : Angle :=
  let total_minutes := a.degrees * 60 + a.minutes + b.degrees * 60 + b.minutes
  { degrees := total_minutes / 60,
    minutes := total_minutes % 60,
    valid := by sorry }

/-- The theorem to prove -/
theorem angle_addition_theorem :
  let a := { degrees := 48, minutes := 39, valid := by sorry }
  let b := { degrees := 67, minutes := 31, valid := by sorry }
  let result := add_angles a b
  result.degrees = 116 ∧ result.minutes = 10 := by sorry

end NUMINAMATH_CALUDE_angle_addition_theorem_l596_59671


namespace NUMINAMATH_CALUDE_complex_sum_of_powers_l596_59641

theorem complex_sum_of_powers (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + x*y + y^2 = 0) : 
  (x / (x + y))^1990 + (y / (x + y))^1990 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_powers_l596_59641


namespace NUMINAMATH_CALUDE_extreme_points_inequality_l596_59680

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + a * log x

theorem extreme_points_inequality (a : ℝ) (m : ℝ) :
  a > 0 →
  ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧
    (∀ x, x > 0 → (deriv (f a)) x = 0 ↔ x = x₁ ∨ x = x₂) →
    (∀ x, x > 0 → f a x₁ ≥ m * x₂) →
    m ≤ -3/2 - log 2 := by
  sorry

end NUMINAMATH_CALUDE_extreme_points_inequality_l596_59680


namespace NUMINAMATH_CALUDE_smallest_divisible_addition_l596_59635

theorem smallest_divisible_addition (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬((2013 + m) % 11 = 0 ∧ (2013 + m) % 13 = 0)) ∧
  (2013 + n) % 11 = 0 ∧ (2013 + n) % 13 = 0 →
  n = 132 := by
sorry

end NUMINAMATH_CALUDE_smallest_divisible_addition_l596_59635


namespace NUMINAMATH_CALUDE_diamond_crack_l596_59611

theorem diamond_crack (p p₁ p₂ : ℝ) (h₁ : p > 0) (h₂ : p₁ > 0) (h₃ : p₂ > 0) 
  (h₄ : p₁ + p₂ = p) (h₅ : p₁^2 + p₂^2 = 0.68 * p^2) : 
  min p₁ p₂ / p = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_diamond_crack_l596_59611


namespace NUMINAMATH_CALUDE_lottery_probability_theorem_l596_59681

/-- The number of StarBalls in the lottery -/
def num_starballs : ℕ := 30

/-- The number of MagicBalls in the lottery -/
def num_magicballs : ℕ := 49

/-- The number of MagicBalls that need to be picked -/
def num_picked_magicballs : ℕ := 6

/-- The probability of winning the lottery -/
def lottery_win_probability : ℚ := 1 / 419514480

theorem lottery_probability_theorem :
  (1 : ℚ) / num_starballs * (1 : ℚ) / (num_magicballs.choose num_picked_magicballs) = lottery_win_probability := by
  sorry

end NUMINAMATH_CALUDE_lottery_probability_theorem_l596_59681


namespace NUMINAMATH_CALUDE_fraction_sum_equals_two_l596_59636

theorem fraction_sum_equals_two (p q : ℚ) (h : p / q = 4 / 5) :
  11 / 7 + (2 * q - p) / (2 * q + p) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_two_l596_59636


namespace NUMINAMATH_CALUDE_first_digit_base_nine_is_three_l596_59669

def base_three_representation : List Nat := [2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 1]

def y : Nat := base_three_representation.enum.foldl (fun acc (i, digit) => acc + digit * (3 ^ (base_three_representation.length - 1 - i))) 0

theorem first_digit_base_nine_is_three :
  ∃ (rest : Nat), y = 3 * (9 ^ (Nat.log 9 y)) + rest ∧ rest < 9 ^ (Nat.log 9 y) :=
by sorry

end NUMINAMATH_CALUDE_first_digit_base_nine_is_three_l596_59669


namespace NUMINAMATH_CALUDE_teacher_distribution_l596_59614

theorem teacher_distribution (n : ℕ) (m : ℕ) :
  n = 6 →
  m = 3 →
  (Nat.choose n 3) * (Nat.choose (n - 3) 2) * (Nat.factorial m) = 360 := by
  sorry

end NUMINAMATH_CALUDE_teacher_distribution_l596_59614


namespace NUMINAMATH_CALUDE_ceiling_sqrt_sum_times_two_l596_59643

theorem ceiling_sqrt_sum_times_two : 
  2 * (⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉) = 54 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_sum_times_two_l596_59643


namespace NUMINAMATH_CALUDE_p_true_q_false_range_l596_59604

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 2*x₀ - m - 1 < 0

def q (m : ℝ) : Prop := ∀ x ∈ Set.Icc 1 4, x + 4/x > m

-- Theorem statement
theorem p_true_q_false_range (m : ℝ) :
  p m ∧ ¬(q m) ↔ m ≥ 4 ∨ m ≤ -2 := by sorry

end NUMINAMATH_CALUDE_p_true_q_false_range_l596_59604


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l596_59695

theorem smallest_n_congruence (n : ℕ) : 
  (23 * n ≡ 310 [ZMOD 9]) ∧ (∀ m : ℕ, m < n → ¬(23 * m ≡ 310 [ZMOD 9])) ↔ n = 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l596_59695


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_18_24_30_l596_59638

def gcd3 (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)

def lcm3 (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem gcd_lcm_sum_18_24_30 : 
  gcd3 18 24 30 + lcm3 18 24 30 = 366 := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_18_24_30_l596_59638


namespace NUMINAMATH_CALUDE_inequality_proof_l596_59618

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2*b + b^2*c + c^2*a) * (a*b^2 + b*c^2 + c*a^2) ≥ 9*a^2*b^2*c^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l596_59618


namespace NUMINAMATH_CALUDE_four_digit_sum_l596_59633

/-- Given four distinct non-zero digits, the sum of all possible four-digit numbers formed using these digits without repetition is 73,326 if and only if the digits are 1, 2, 3, and 5. -/
theorem four_digit_sum (a b c d : ℕ) : 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
  6 * (a + b + c + d) * 1111 = 73326 →
  ({a, b, c, d} : Set ℕ) = {1, 2, 3, 5} :=
by sorry

end NUMINAMATH_CALUDE_four_digit_sum_l596_59633


namespace NUMINAMATH_CALUDE_base5_polynomial_representation_l596_59699

/-- Represents a polynomial in base 5 with coefficients less than 5 -/
def Base5Polynomial (coeffs : List Nat) : Prop :=
  coeffs.all (· < 5) ∧ coeffs.length > 0

/-- Converts a list of coefficients to a natural number in base 5 -/
def toBase5Number (coeffs : List Nat) : Nat :=
  coeffs.foldl (fun acc d => 5 * acc + d) 0

/-- Theorem: A polynomial with coefficients less than 5 can be uniquely represented as a base-5 number -/
theorem base5_polynomial_representation 
  (coeffs : List Nat) 
  (h : Base5Polynomial coeffs) :
  ∃! n : Nat, n = toBase5Number coeffs :=
sorry

end NUMINAMATH_CALUDE_base5_polynomial_representation_l596_59699


namespace NUMINAMATH_CALUDE_divisor_problem_l596_59658

theorem divisor_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 997 →
  quotient = 43 →
  remainder = 8 →
  dividend = divisor * quotient + remainder →
  Nat.Prime divisor →
  divisor = 23 := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l596_59658


namespace NUMINAMATH_CALUDE_three_same_one_different_probability_l596_59623

def number_of_dice : ℕ := 4
def faces_per_die : ℕ := 6

def total_outcomes : ℕ := faces_per_die ^ number_of_dice

def successful_outcomes : ℕ := faces_per_die * (number_of_dice.choose 3) * (faces_per_die - 1)

def probability_three_same_one_different : ℚ := successful_outcomes / total_outcomes

theorem three_same_one_different_probability :
  probability_three_same_one_different = 5 / 54 := by
  sorry

end NUMINAMATH_CALUDE_three_same_one_different_probability_l596_59623


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l596_59649

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 138) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 20 := by sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l596_59649


namespace NUMINAMATH_CALUDE_tan_150_degrees_l596_59628

theorem tan_150_degrees : Real.tan (150 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_150_degrees_l596_59628


namespace NUMINAMATH_CALUDE_factorial_equation_solutions_l596_59624

theorem factorial_equation_solutions :
  ∀ a b c : ℕ+,
    a^2 + b^2 + 1 = c! →
    ((a = 2 ∧ b = 1 ∧ c = 3) ∨ (a = 1 ∧ b = 2 ∧ c = 3)) :=
by sorry

end NUMINAMATH_CALUDE_factorial_equation_solutions_l596_59624


namespace NUMINAMATH_CALUDE_product_w_k_value_l596_59692

theorem product_w_k_value : 
  let a : ℕ := 105
  let b : ℕ := 60
  let c : ℕ := 42
  let w : ℕ := (a^3 * b^4) / (21 * 25 * 45 * 50)
  let k : ℕ := c^5 / (35 * 28 * 56)
  w * k = 18458529600 := by sorry

end NUMINAMATH_CALUDE_product_w_k_value_l596_59692


namespace NUMINAMATH_CALUDE_theo_selling_price_l596_59601

/-- Represents the selling price and profit for a camera seller --/
structure CameraSeller where
  buyPrice : ℕ
  sellPrice : ℕ
  quantity : ℕ

/-- Calculates the total profit for a camera seller --/
def totalProfit (seller : CameraSeller) : ℕ :=
  (seller.sellPrice - seller.buyPrice) * seller.quantity

theorem theo_selling_price 
  (maddox theo : CameraSeller)
  (h1 : maddox.buyPrice = 20)
  (h2 : theo.buyPrice = 20)
  (h3 : maddox.quantity = 3)
  (h4 : theo.quantity = 3)
  (h5 : maddox.sellPrice = 28)
  (h6 : totalProfit maddox = totalProfit theo + 15) :
  theo.sellPrice = 23 := by
sorry

end NUMINAMATH_CALUDE_theo_selling_price_l596_59601


namespace NUMINAMATH_CALUDE_no_real_solutions_l596_59691

theorem no_real_solutions : ¬∃ (a b : ℝ), 
  (a - 8 = b - a) ∧ (ab - b = b - a) ∧ (a * (1 + b) = 3 * b) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l596_59691


namespace NUMINAMATH_CALUDE_trekking_meals_theorem_l596_59657

/-- Represents the number of meals available for children given the trekking group scenario -/
def meals_for_children (total_adults : ℕ) (total_children : ℕ) (max_adult_meals : ℕ) 
  (adults_eaten : ℕ) (children_fed_with_remainder : ℕ) : ℕ :=
  2 * children_fed_with_remainder

/-- Theorem stating that the number of meals initially available for children is 90 -/
theorem trekking_meals_theorem (total_adults : ℕ) (total_children : ℕ) (max_adult_meals : ℕ) 
  (adults_eaten : ℕ) (children_fed_with_remainder : ℕ) :
  total_adults = 55 →
  total_children = 70 →
  max_adult_meals = 70 →
  adults_eaten = 35 →
  children_fed_with_remainder = 45 →
  meals_for_children total_adults total_children max_adult_meals adults_eaten children_fed_with_remainder = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_trekking_meals_theorem_l596_59657


namespace NUMINAMATH_CALUDE_min_distance_line_curve_l596_59621

/-- The minimum distance between a point on the line y = 1 - x and a point on the curve y = -e^x is √2 -/
theorem min_distance_line_curve : 
  ∃ (d : ℝ), d = Real.sqrt 2 ∧ 
  ∀ (P Q : ℝ × ℝ), 
    (P.2 = 1 - P.1) → 
    (Q.2 = -Real.exp Q.1) → 
    d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
sorry

end NUMINAMATH_CALUDE_min_distance_line_curve_l596_59621


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l596_59697

/-- Given a rectangle divided into three congruent squares, where each square has a perimeter of 24 inches, prove that the perimeter of the original rectangle is 48 inches. -/
theorem rectangle_perimeter (square_perimeter : ℝ) (h1 : square_perimeter = 24) : 
  let square_side := square_perimeter / 4
  let rectangle_length := 3 * square_side
  let rectangle_width := square_side
  2 * (rectangle_length + rectangle_width) = 48 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l596_59697


namespace NUMINAMATH_CALUDE_field_trip_totals_budget_exceeded_l596_59642

/-- Represents the total number of people and cost for the field trip -/
structure FieldTrip where
  students : ℕ
  teachers : ℕ
  parents : ℕ
  cost : ℕ

/-- Calculates the total number of people and cost for a given vehicle type -/
def vehicleTotal (vehicles : ℕ) (studentsPerVehicle : ℕ) (teachersPerVehicle : ℕ) (parentsPerVehicle : ℕ) (costPerVehicle : ℕ) : FieldTrip :=
  { students := vehicles * studentsPerVehicle,
    teachers := vehicles * teachersPerVehicle,
    parents := vehicles * parentsPerVehicle,
    cost := vehicles * costPerVehicle }

/-- Combines two FieldTrip structures -/
def combineFieldTrips (a b : FieldTrip) : FieldTrip :=
  { students := a.students + b.students,
    teachers := a.teachers + b.teachers,
    parents := a.parents + b.parents,
    cost := a.cost + b.cost }

theorem field_trip_totals : 
  let vans := vehicleTotal 6 10 2 1 100
  let minibusses := vehicleTotal 4 24 3 2 200
  let coachBuses := vehicleTotal 2 48 4 4 350
  let schoolBus := vehicleTotal 1 35 5 3 250
  let total := combineFieldTrips (combineFieldTrips (combineFieldTrips vans minibusses) coachBuses) schoolBus
  total.students = 287 ∧ 
  total.teachers = 37 ∧ 
  total.parents = 25 ∧ 
  total.cost = 2350 :=
by sorry

theorem budget_exceeded (budget : ℕ) : 
  let vans := vehicleTotal 6 10 2 1 100
  let minibusses := vehicleTotal 4 24 3 2 200
  let coachBuses := vehicleTotal 2 48 4 4 350
  let schoolBus := vehicleTotal 1 35 5 3 250
  let total := combineFieldTrips (combineFieldTrips (combineFieldTrips vans minibusses) coachBuses) schoolBus
  budget = 2000 →
  total.cost - budget = 350 :=
by sorry

end NUMINAMATH_CALUDE_field_trip_totals_budget_exceeded_l596_59642


namespace NUMINAMATH_CALUDE_fraction_product_equals_one_l596_59685

theorem fraction_product_equals_one : 
  (2 * 7) / (14 * 6) * (6 * 14) / (2 * 7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equals_one_l596_59685


namespace NUMINAMATH_CALUDE_sqrt_eight_and_three_ninths_simplification_l596_59648

theorem sqrt_eight_and_three_ninths_simplification :
  Real.sqrt (8 + 3 / 9) = 5 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_and_three_ninths_simplification_l596_59648


namespace NUMINAMATH_CALUDE_evaluate_expression_l596_59644

theorem evaluate_expression (x : ℝ) (h : x = 3) : x - x * (x ^ x) = -78 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l596_59644


namespace NUMINAMATH_CALUDE_distinct_collections_eq_sixteen_l596_59612

/-- The number of vowels in MATHCOUNTS -/
def num_vowels : ℕ := 3

/-- The number of consonants in MATHCOUNTS excluding T -/
def num_consonants_without_t : ℕ := 5

/-- The number of T's in MATHCOUNTS -/
def num_t : ℕ := 2

/-- The number of vowels to be selected -/
def vowels_to_select : ℕ := 3

/-- The number of consonants to be selected -/
def consonants_to_select : ℕ := 2

/-- The function to calculate the number of distinct collections -/
def distinct_collections : ℕ :=
  (Nat.choose num_vowels vowels_to_select) * (Nat.choose num_consonants_without_t consonants_to_select) +
  (Nat.choose num_vowels vowels_to_select) * (Nat.choose num_consonants_without_t (consonants_to_select - 1)) +
  (Nat.choose num_vowels vowels_to_select) * (Nat.choose num_consonants_without_t (consonants_to_select - 2))

theorem distinct_collections_eq_sixteen :
  distinct_collections = 16 := by sorry

end NUMINAMATH_CALUDE_distinct_collections_eq_sixteen_l596_59612


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l596_59622

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : 0 < a
  b_pos : 0 < b

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The focal length of a hyperbola -/
def focal_length (h : Hyperbola a b) : ℝ := sorry

/-- A point on a hyperbola -/
structure PointOnHyperbola (h : Hyperbola a b) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / a^2 - y^2 / b^2 = 1

/-- The distance between two points -/
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := sorry

/-- The origin (0, 0) -/
def origin : ℝ × ℝ := (0, 0)

/-- Left focus of the hyperbola -/
def left_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- Right focus of the hyperbola -/
def right_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) 
  (P : PointOnHyperbola h) :
  distance P.x P.y (origin.1) (origin.2) = focal_length h / 2 →
  distance P.x P.y (left_focus h).1 (left_focus h).2 + 
    distance P.x P.y (right_focus h).1 (right_focus h).2 = 4 * a →
  eccentricity h = Real.sqrt 10 / 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l596_59622


namespace NUMINAMATH_CALUDE_system_solution_l596_59616

theorem system_solution (x y m : ℤ) :
  x + 2*y - 6 = 0 ∧ 
  x - 2*y + m*x + 5 = 0 →
  m = -1 ∨ m = -3 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l596_59616


namespace NUMINAMATH_CALUDE_division_theorem_l596_59631

theorem division_theorem (a b q r : ℕ) (h1 : a = 1270) (h2 : b = 17) (h3 : q = 74) (h4 : r = 12) :
  a = b * q + r ∧ r < b := by
  sorry

end NUMINAMATH_CALUDE_division_theorem_l596_59631


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l596_59637

theorem equilateral_triangle_area (perimeter : ℝ) (area : ℝ) :
  perimeter = 120 →
  area = (400 : ℝ) * Real.sqrt 3 →
  ∃ (side : ℝ), 
    side > 0 ∧
    perimeter = 3 * side ∧
    area = (Real.sqrt 3 / 4) * side^2 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l596_59637


namespace NUMINAMATH_CALUDE_unique_digit_arrangement_l596_59632

theorem unique_digit_arrangement :
  ∃! (A B C D E : ℕ),
    (A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0 ∧ E ≠ 0) ∧
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
     B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
     C ≠ D ∧ C ≠ E ∧
     D ≠ E) ∧
    (A + B : ℚ) = (C + D + E : ℚ) / 7 ∧
    (A + C : ℚ) = (B + D + E : ℚ) / 5 :=
by sorry

end NUMINAMATH_CALUDE_unique_digit_arrangement_l596_59632


namespace NUMINAMATH_CALUDE_boat_license_count_l596_59606

def boat_license_options : ℕ :=
  let letter_options := 3  -- A, M, or S
  let digit_options := 10  -- 0 to 9
  let digit_positions := 5
  letter_options * digit_options ^ digit_positions

theorem boat_license_count : boat_license_options = 300000 := by
  sorry

end NUMINAMATH_CALUDE_boat_license_count_l596_59606


namespace NUMINAMATH_CALUDE_trapezoid_segment_equality_l596_59646

-- Define the points
variable (A B C D M N : Point)

-- Define the trapezoid
def is_trapezoid (A B C D : Point) : Prop := sorry

-- Define that M is on CD and N is on AB
def point_on_segment (P Q R : Point) : Prop := sorry

-- Define angle equality
def angle_eq (P Q R S T U : Point) : Prop := sorry

-- Define segment equality
def segment_eq (P Q R S : Point) : Prop := sorry

-- Theorem statement
theorem trapezoid_segment_equality 
  (h1 : is_trapezoid A B C D)
  (h2 : point_on_segment C D M)
  (h3 : point_on_segment A B N)
  (h4 : segment_eq A N B N)
  (h5 : angle_eq A B N C D M) :
  segment_eq C M M D :=
sorry

end NUMINAMATH_CALUDE_trapezoid_segment_equality_l596_59646


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l596_59670

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 1 ∧ x₂ = 3 ∧ 
  (∀ x : ℝ, x^2 - 4*x + 3 = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l596_59670


namespace NUMINAMATH_CALUDE_test_pumping_result_l596_59659

/-- Calculates the total gallons pumped during a test with two pumps -/
def total_gallons_pumped (pump1_rate : ℝ) (pump2_rate : ℝ) (total_time : ℝ) (pump2_time : ℝ) : ℝ :=
  let pump1_time := total_time - pump2_time
  pump1_rate * pump1_time + pump2_rate * pump2_time

/-- Proves that the total gallons pumped is 1325 given the specified conditions -/
theorem test_pumping_result :
  total_gallons_pumped 180 250 6 3.5 = 1325 := by
  sorry

end NUMINAMATH_CALUDE_test_pumping_result_l596_59659


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_four_l596_59625

/-- Given acute angles x and y satisfying specific trigonometric equations,
    prove that their combination 2x + y equals π/4 radians. -/
theorem angle_sum_is_pi_over_four (x y : Real) (hx : 0 < x ∧ x < π/2) (hy : 0 < y ∧ y < π/2)
  (h1 : 2 * Real.cos x ^ 2 + 3 * Real.cos y ^ 2 = 1)
  (h2 : 2 * Real.sin (2 * x) + 3 * Real.sin (2 * y) = 0) :
  2 * x + y = π/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_four_l596_59625


namespace NUMINAMATH_CALUDE_symmetry_condition_l596_59627

/-- The necessary condition for y = 2x to be an axis of symmetry of y = (px + q)/(rx + s) --/
theorem symmetry_condition (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  (∀ x y : ℝ, y = (p * x + q) / (r * x + s) → 
    ∃ x' y', y' = (p * x' + q) / (r * x' + s) ∧ x = y' / 2 ∧ y = x') →
  p + s = 0 := by
sorry

end NUMINAMATH_CALUDE_symmetry_condition_l596_59627


namespace NUMINAMATH_CALUDE_petes_age_relation_l596_59607

/-- 
Given Pete's current age and his son's current age, 
this theorem proves how many years it will take for Pete 
to be exactly three times older than his son.
-/
theorem petes_age_relation (pete_age son_age : ℕ) 
  (h1 : pete_age = 35) (h2 : son_age = 9) : 
  ∃ (years : ℕ), pete_age + years = 3 * (son_age + years) ∧ years = 4 := by
  sorry

end NUMINAMATH_CALUDE_petes_age_relation_l596_59607


namespace NUMINAMATH_CALUDE_jasmine_weight_l596_59650

/-- The weight of a bag of chips in ounces -/
def bag_weight : ℕ := 20

/-- The weight of a tin of cookies in ounces -/
def tin_weight : ℕ := 9

/-- The number of bags of chips Jasmine buys -/
def num_bags : ℕ := 6

/-- The number of tins of cookies Jasmine buys -/
def num_tins : ℕ := 4 * num_bags

/-- The number of ounces in a pound -/
def ounces_per_pound : ℕ := 16

/-- The total weight Jasmine has to carry in pounds -/
def total_weight : ℕ := (bag_weight * num_bags + tin_weight * num_tins) / ounces_per_pound

theorem jasmine_weight : total_weight = 21 := by
  sorry

end NUMINAMATH_CALUDE_jasmine_weight_l596_59650


namespace NUMINAMATH_CALUDE_inequality_solution_l596_59675

theorem inequality_solution (x : ℝ) :
  x ≠ 5 → (x * (x - 2) / ((x - 5)^2) ≥ 15 ↔ x ∈ Set.Iio 5 ∪ Set.Ioi 5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l596_59675


namespace NUMINAMATH_CALUDE_bertha_family_without_children_bertha_family_consistent_l596_59647

/-- Represents the family structure of Bertha and her descendants -/
structure BerthaFamily where
  daughters : ℕ
  granddaughters : ℕ
  daughters_with_children : ℕ
  daughters_per_mother : ℕ

/-- The actual family structure of Bertha -/
def bertha_family : BerthaFamily :=
  { daughters := 8,
    granddaughters := 34,
    daughters_with_children := 5,
    daughters_per_mother := 6 }

/-- Theorem stating the number of Bertha's daughters and granddaughters without children -/
theorem bertha_family_without_children :
  bertha_family.daughters + bertha_family.granddaughters - bertha_family.daughters_with_children = 37 :=
by sorry

/-- Consistency check for the family structure -/
theorem bertha_family_consistent :
  bertha_family.daughters + bertha_family.granddaughters = 42 ∧
  bertha_family.granddaughters = bertha_family.daughters_with_children * bertha_family.daughters_per_mother :=
by sorry

end NUMINAMATH_CALUDE_bertha_family_without_children_bertha_family_consistent_l596_59647


namespace NUMINAMATH_CALUDE_sum_product_reciprocals_inequality_l596_59694

theorem sum_product_reciprocals_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  (a + b + c) * (1/a + 1/b + 1/c) > 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_reciprocals_inequality_l596_59694


namespace NUMINAMATH_CALUDE_incorrect_average_calculation_l596_59684

theorem incorrect_average_calculation (n : ℕ) (correct_avg : ℝ) (error : ℝ) :
  n = 10 ∧ correct_avg = 18 ∧ error = 20 →
  (n * correct_avg - error) / n = 16 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_average_calculation_l596_59684


namespace NUMINAMATH_CALUDE_number_order_l596_59630

theorem number_order (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (2 * a * b) / (a + b) ≤ Real.sqrt (a * b) ∧ Real.sqrt (a * b) ≤ (a + b) / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_order_l596_59630


namespace NUMINAMATH_CALUDE_circle_center_transformation_l596_59617

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def translate_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + d)

theorem circle_center_transformation :
  let S : ℝ × ℝ := (3, -4)
  let reflected := reflect_x S
  let translated := translate_up reflected 5
  translated = (3, 9) := by sorry

end NUMINAMATH_CALUDE_circle_center_transformation_l596_59617


namespace NUMINAMATH_CALUDE_school_students_count_l596_59626

theorem school_students_count (below_8_percent : ℝ) (age_8_count : ℕ) (above_8_ratio : ℝ) :
  below_8_percent = 0.2 →
  age_8_count = 60 →
  above_8_ratio = 2/3 →
  ∃ (total : ℕ), total = 125 ∧ 
    (total : ℝ) * below_8_percent + (age_8_count : ℝ) + (age_8_count : ℝ) * above_8_ratio = total := by
  sorry

end NUMINAMATH_CALUDE_school_students_count_l596_59626


namespace NUMINAMATH_CALUDE_meteorological_satellite_requires_comprehensive_survey_l596_59679

/-- Represents a type of survey --/
inductive SurveyType
| Comprehensive
| Sampling

/-- Represents a scenario for data collection --/
structure DataCollectionScenario where
  description : String
  requiredSurveyType : SurveyType

/-- Represents a component of a meteorological satellite --/
structure SatelliteComponent where
  id : Nat
  quality : Bool  -- True if the component meets quality standards, False otherwise

/-- Definition of a meteorological satellite --/
structure MeteorologicalSatellite where
  components : List SatelliteComponent

/-- Function to determine if a satellite is functional --/
def isSatelliteFunctional (satellite : MeteorologicalSatellite) : Bool :=
  satellite.components.all (fun c => c.quality)

/-- Theorem stating that the quality assessment of meteorological satellite components requires a comprehensive survey --/
theorem meteorological_satellite_requires_comprehensive_survey 
  (scenario : DataCollectionScenario) 
  (h1 : scenario.description = "The quality of components of a meteorological satellite about to be launched") :
  scenario.requiredSurveyType = SurveyType.Comprehensive :=
by
  sorry


end NUMINAMATH_CALUDE_meteorological_satellite_requires_comprehensive_survey_l596_59679


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l596_59673

theorem simplify_and_evaluate (x y : ℚ) 
  (hx : x = -1) 
  (hy : y = 1/3) : 
  (x + y) * (x - y) - (4 * x^3 * y - 8 * x * y^3) / (2 * x * y) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l596_59673


namespace NUMINAMATH_CALUDE_four_digit_number_at_1496_l596_59615

/-- Given a list of increasing positive integers starting with digit 2,
    returns the four-digit number formed by the nth to (n+3)th digits in the list -/
def fourDigitNumber (n : ℕ) : ℕ :=
  sorry

/-- The list of increasing positive integers starting with digit 2 -/
def digitTwoList : List ℕ :=
  sorry

theorem four_digit_number_at_1496 :
  fourDigitNumber 1496 = 5822 :=
sorry

end NUMINAMATH_CALUDE_four_digit_number_at_1496_l596_59615


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_mod_20_l596_59686

/-- The sum of an arithmetic sequence modulo 20 -/
theorem arithmetic_sequence_sum_mod_20 : ∃ m : ℕ, 
  let a := 2  -- first term
  let l := 102  -- last term
  let d := 5  -- common difference
  let n := (l - a) / d + 1  -- number of terms
  let S := n * (a + l) / 2  -- sum of arithmetic sequence
  S % 20 = m ∧ m < 20 ∧ m = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_mod_20_l596_59686


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l596_59620

theorem pure_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := (1 + a * Complex.I) / (1 - Complex.I)
  (∃ (b : ℝ), z = b * Complex.I) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l596_59620


namespace NUMINAMATH_CALUDE_ant_growth_rate_l596_59654

/-- The growth rate of an ant population over 5 hours -/
theorem ant_growth_rate (initial_population final_population : ℝ) 
  (h1 : initial_population = 50)
  (h2 : final_population = 1600)
  (h3 : final_population = initial_population * (growth_rate ^ 5)) :
  growth_rate = (32 : ℝ) ^ (1/5) :=
by sorry

end NUMINAMATH_CALUDE_ant_growth_rate_l596_59654


namespace NUMINAMATH_CALUDE_cosine_identity_l596_59656

/-- Proves that 2cos(16°)cos(29°) - cos(13°) = √2/2 -/
theorem cosine_identity : 
  2 * Real.cos (16 * π / 180) * Real.cos (29 * π / 180) - Real.cos (13 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_identity_l596_59656


namespace NUMINAMATH_CALUDE_max_sum_xyz_l596_59602

theorem max_sum_xyz (x y z c : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hc : c > 0)
  (h1 : x + c * y ≤ 36) (h2 : 2 * x + 3 * z ≤ 72) :
  (c ≥ 3 → ∃ (x' y' z' : ℝ), x' ≥ 0 ∧ y' ≥ 0 ∧ z' ≥ 0 ∧
    x' + c * y' ≤ 36 ∧ 2 * x' + 3 * z' ≤ 72 ∧ x' + y' + z' = 36 ∧
    ∀ (a b d : ℝ), a ≥ 0 → b ≥ 0 → d ≥ 0 → a + c * b ≤ 36 → 2 * a + 3 * d ≤ 72 → a + b + d ≤ 36) ∧
  (c < 3 → ∃ (x' y' z' : ℝ), x' ≥ 0 ∧ y' ≥ 0 ∧ z' ≥ 0 ∧
    x' + c * y' ≤ 36 ∧ 2 * x' + 3 * z' ≤ 72 ∧ x' + y' + z' = 24 + 36 / c ∧
    ∀ (a b d : ℝ), a ≥ 0 → b ≥ 0 → d ≥ 0 → a + c * b ≤ 36 → 2 * a + 3 * d ≤ 72 → a + b + d ≤ 24 + 36 / c) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_xyz_l596_59602


namespace NUMINAMATH_CALUDE_expression_defined_iff_in_interval_l596_59610

/-- The expression log(x-2) / sqrt(5-x) is defined if and only if x is in the open interval (2, 5) -/
theorem expression_defined_iff_in_interval (x : ℝ) :
  (∃ y : ℝ, y = (Real.log (x - 2)) / Real.sqrt (5 - x)) ↔ 2 < x ∧ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_defined_iff_in_interval_l596_59610


namespace NUMINAMATH_CALUDE_point_distance_theorem_l596_59665

/-- A point in the first quadrant with coordinates (3-a, 2a+2) -/
structure Point (a : ℝ) where
  x : ℝ := 3 - a
  y : ℝ := 2*a + 2
  first_quadrant : 0 < x ∧ 0 < y

/-- The theorem stating that if the distance from the point to the x-axis is twice
    the distance to the y-axis, then a = 1 -/
theorem point_distance_theorem (a : ℝ) (P : Point a) :
  P.y = 2 * P.x → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_distance_theorem_l596_59665


namespace NUMINAMATH_CALUDE_shift_linear_function_l596_59640

-- Define the original linear function
def original_function (x : ℝ) : ℝ := 5 * x - 8

-- Define the shift amount
def shift : ℝ := 4

-- Define the shifted function
def shifted_function (x : ℝ) : ℝ := original_function x + shift

-- Theorem statement
theorem shift_linear_function :
  ∀ x : ℝ, shifted_function x = 5 * x - 4 :=
by
  sorry

end NUMINAMATH_CALUDE_shift_linear_function_l596_59640


namespace NUMINAMATH_CALUDE_fruit_vendor_sales_l596_59609

/-- A fruit vendor's sales problem -/
theorem fruit_vendor_sales (apple_price orange_price : ℚ)
  (morning_apples morning_oranges : ℕ)
  (afternoon_oranges : ℕ)
  (total_sales : ℚ)
  (h1 : apple_price = 3/2)
  (h2 : orange_price = 1)
  (h3 : morning_apples = 40)
  (h4 : morning_oranges = 30)
  (h5 : afternoon_oranges = 40)
  (h6 : total_sales = 205) :
  ∃ (afternoon_apples : ℕ),
    afternoon_apples = 50 ∧
    total_sales = apple_price * (morning_apples + afternoon_apples) +
                  orange_price * (morning_oranges + afternoon_oranges) :=
by sorry

end NUMINAMATH_CALUDE_fruit_vendor_sales_l596_59609


namespace NUMINAMATH_CALUDE_floor_abs_sum_l596_59687

theorem floor_abs_sum : ⌊|(-5.7:ℝ)|⌋ + |⌊(-5.7:ℝ)⌋| = 11 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_sum_l596_59687


namespace NUMINAMATH_CALUDE_simplify_expression_l596_59678

theorem simplify_expression : (18 * 10^9) / (6 * 10^4) = 300000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l596_59678


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l596_59634

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (2*x - 3)^9 = a₀ + a₁*(x - 1) + a₂*(x - 1)^2 + a₃*(x - 1)^3 + 
    a₄*(x - 1)^4 + a₅*(x - 1)^5 + a₆*(x - 1)^6 + a₇*(x - 1)^7 + a₈*(x - 1)^8 + a₉*(x - 1)^9) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l596_59634


namespace NUMINAMATH_CALUDE_two_digit_number_value_l596_59696

/-- Represents a two-digit number with 5 as its tens digit and A as its units digit -/
def two_digit_number (A : ℕ) : ℕ := 50 + A

/-- The value of a two-digit number with 5 as its tens digit and A as its units digit -/
theorem two_digit_number_value (A : ℕ) (h : A < 10) : 
  two_digit_number A = 50 + A := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_value_l596_59696


namespace NUMINAMATH_CALUDE_smartphone_loss_percentage_l596_59655

theorem smartphone_loss_percentage (initial_cost selling_price : ℝ) :
  initial_cost = 300 →
  selling_price = 255 →
  (initial_cost - selling_price) / initial_cost * 100 = 15 := by
sorry

end NUMINAMATH_CALUDE_smartphone_loss_percentage_l596_59655


namespace NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_l596_59608

theorem least_subtrahend_for_divisibility (n : ℕ) (p q : ℕ) (h_p_prime : Nat.Prime p) (h_q_prime : Nat.Prime q) (h_p_ne_q : p ≠ q) :
  let m := n - (n / (p * q)) * (p * q)
  (∀ k < m, ¬((n - k) % p = 0 ∧ (n - k) % q = 0)) ∧
  (n - m) % p = 0 ∧ (n - m) % q = 0 :=
by sorry

-- The specific problem instance
example : 
  let n := 724946
  let p := 37
  let q := 53
  let m := n - (n / (p * q)) * (p * q)
  m = 1157 ∧
  (∀ k < m, ¬((n - k) % p = 0 ∧ (n - k) % q = 0)) ∧
  (n - m) % p = 0 ∧ (n - m) % q = 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_l596_59608


namespace NUMINAMATH_CALUDE_half_number_plus_seven_l596_59683

theorem half_number_plus_seven (n : ℝ) : n = 20 → (n / 2) + 7 = 17 := by
  sorry

end NUMINAMATH_CALUDE_half_number_plus_seven_l596_59683


namespace NUMINAMATH_CALUDE_flower_bed_fraction_is_one_sixth_l596_59662

/-- Represents a rectangular yard with flower beds -/
structure YardWithFlowerBeds where
  yard_width : ℝ
  trapezoid_short_side : ℝ
  trapezoid_long_side : ℝ

/-- The fraction of the yard occupied by the flower beds -/
def flower_bed_fraction (y : YardWithFlowerBeds) : ℚ :=
  1/6

/-- Theorem stating that the fraction of the yard occupied by the flower beds is 1/6 -/
theorem flower_bed_fraction_is_one_sixth (y : YardWithFlowerBeds)
    (h1 : y.trapezoid_short_side = 20)
    (h2 : y.trapezoid_long_side = 30) :
    flower_bed_fraction y = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_fraction_is_one_sixth_l596_59662


namespace NUMINAMATH_CALUDE_minimize_xy_sum_l596_59668

theorem minimize_xy_sum (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 11) :
  (∀ a b : ℕ+, (1 : ℚ) / a + (1 : ℚ) / (3 * b) = (1 : ℚ) / 11 → x * y ≤ a * b) →
  x * y = 176 ∧ x + y = 30 :=
sorry

end NUMINAMATH_CALUDE_minimize_xy_sum_l596_59668


namespace NUMINAMATH_CALUDE_solve_blueberry_problem_l596_59619

/-- The number of blueberry bushes Natalie needs to pick -/
def blueberry_problem (containers_per_bush : ℕ) (containers_per_zucchini : ℕ) (containers_to_keep : ℕ) (target_zucchinis : ℕ) : ℕ :=
  (target_zucchinis * containers_per_zucchini + containers_to_keep) / containers_per_bush

/-- Theorem stating the solution to Natalie's blueberry problem -/
theorem solve_blueberry_problem :
  blueberry_problem 10 4 20 60 = 26 := by
  sorry

end NUMINAMATH_CALUDE_solve_blueberry_problem_l596_59619


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l596_59688

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane : Plane → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_perpendicular_planes 
  (m n : Line) (α β : Plane) :
  perp_line_plane m α → 
  perp_line_plane n β → 
  perp_plane α β → 
  perp_line m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l596_59688


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_range_l596_59674

theorem quadratic_always_nonnegative_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 ≥ 0) ↔ -1 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_range_l596_59674


namespace NUMINAMATH_CALUDE_recipe_solution_l596_59693

def recipe_problem (sugar : ℕ) (flour_put : ℕ) (flour_sugar_diff : ℕ) : Prop :=
  let total_flour := sugar + flour_sugar_diff
  total_flour = flour_put + (total_flour - flour_put)

theorem recipe_solution :
  let sugar := 7
  let flour_put := 2
  let flour_sugar_diff := 2
  recipe_problem sugar flour_put flour_sugar_diff ∧
  (sugar + flour_sugar_diff = 9) :=
by sorry

end NUMINAMATH_CALUDE_recipe_solution_l596_59693


namespace NUMINAMATH_CALUDE_length_of_cd_l596_59605

/-- Given a line segment CD with points R and S on it, prove that CD has length 189 -/
theorem length_of_cd (C D R S : ℝ) : 
  (∃ (x y u v : ℝ), 
    C < R ∧ R < S ∧ S < D ∧  -- R and S are on the same side of midpoint
    4 * (R - C) = 3 * (D - R) ∧  -- R divides CD in ratio 3:4
    5 * (S - C) = 4 * (D - S) ∧  -- S divides CD in ratio 4:5
    S - R = 3) →  -- RS = 3
  D - C = 189 := by
sorry

end NUMINAMATH_CALUDE_length_of_cd_l596_59605


namespace NUMINAMATH_CALUDE_centroid_altitude_intersection_ratio_l596_59652

-- Define an orthocentric tetrahedron
structure OrthocentricTetrahedron where
  -- Add necessary fields

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a line in 3D space
structure Line3D where
  point : Point3D
  direction : Point3D

-- Define a sphere in 3D space
structure Sphere where
  center : Point3D
  radius : ℝ

def centroid (t : OrthocentricTetrahedron) : Point3D :=
  sorry

def altitudeFoot (t : OrthocentricTetrahedron) : Point3D :=
  sorry

def circumscribedSphere (t : OrthocentricTetrahedron) : Sphere :=
  sorry

def lineIntersectSphere (l : Line3D) (s : Sphere) : Point3D :=
  sorry

def distance (p1 p2 : Point3D) : ℝ :=
  sorry

def pointBetween (p1 p2 p3 : Point3D) : Prop :=
  sorry

theorem centroid_altitude_intersection_ratio 
  (t : OrthocentricTetrahedron)
  (G : Point3D)
  (F : Point3D)
  (K : Point3D) :
  G = centroid t →
  F = altitudeFoot t →
  K = lineIntersectSphere (Line3D.mk F (centroid t)) (circumscribedSphere t) →
  pointBetween K G F →
  distance K G = 3 * distance F G :=
sorry

end NUMINAMATH_CALUDE_centroid_altitude_intersection_ratio_l596_59652


namespace NUMINAMATH_CALUDE_line_plane_relationship_l596_59660

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two planes
variable (perpendicular_plane_plane : Plane → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the containment relation of a line in a plane
variable (contained_line_plane : Line → Plane → Prop)

-- Theorem statement
theorem line_plane_relationship
  (m : Line) (α β : Plane)
  (h1 : perpendicular_line_plane m β)
  (h2 : perpendicular_plane_plane α β) :
  parallel_line_plane m α ∨ contained_line_plane m α :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l596_59660


namespace NUMINAMATH_CALUDE_max_sum_square_roots_equality_condition_l596_59603

theorem max_sum_square_roots (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 25) :
  Real.sqrt (x + 16) + Real.sqrt (25 - x) + 2 * Real.sqrt x ≤ 11 + 4 * Real.sqrt 2 :=
by sorry

theorem equality_condition (x : ℝ) (h : x = 16) :
  Real.sqrt (x + 16) + Real.sqrt (25 - x) + 2 * Real.sqrt x = 11 + 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_square_roots_equality_condition_l596_59603


namespace NUMINAMATH_CALUDE_pencil_eraser_cost_l596_59677

/-- Represents the cost of pencils and erasers -/
structure PencilEraserCost where
  pencil : ℕ
  eraser : ℕ

/-- The total cost of 10 pencils and 4 erasers in cents -/
def totalCost : ℕ := 120

/-- Condition: A pencil costs more than an eraser -/
def pencilCostsMore (cost : PencilEraserCost) : Prop :=
  cost.pencil > cost.eraser

/-- Condition: The total cost equation must be satisfied -/
def satisfiesTotalCost (cost : PencilEraserCost) : Prop :=
  10 * cost.pencil + 4 * cost.eraser = totalCost

/-- The main theorem to prove -/
theorem pencil_eraser_cost :
  ∃ (cost : PencilEraserCost),
    pencilCostsMore cost ∧
    satisfiesTotalCost cost ∧
    cost.pencil + cost.eraser = 15 := by
  sorry

end NUMINAMATH_CALUDE_pencil_eraser_cost_l596_59677
