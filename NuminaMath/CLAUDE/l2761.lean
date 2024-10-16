import Mathlib

namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l2761_276138

theorem divisibility_implies_equality (a b n : ℕ) 
  (h : ∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) : 
  a = b^n := by
sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l2761_276138


namespace NUMINAMATH_CALUDE_triangle_area_arithmetic_progression_l2761_276165

/-- The area of a triangle with base 2a - d and height 2a + d is 2a^2 - d^2/2 -/
theorem triangle_area_arithmetic_progression (a d : ℝ) (h_a : a > 0) :
  let base := 2 * a - d
  let height := 2 * a + d
  (1 / 2 : ℝ) * base * height = 2 * a^2 - d^2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_arithmetic_progression_l2761_276165


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2761_276171

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 7 - 2 * a 4 = 6) 
  (h3 : a 3 = 2) : 
  ∃ d : ℝ, (∀ n, a (n + 1) = a n + d) ∧ d = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2761_276171


namespace NUMINAMATH_CALUDE_right_trapezoid_inscribed_circle_theorem_l2761_276136

/-- A right trapezoid with an inscribed circle -/
structure RightTrapezoidWithInscribedCircle where
  /-- Length of the longer base -/
  a : ℝ
  /-- Length of the shorter base -/
  c : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- The longer base is longer than the shorter base -/
  h1 : a > c
  /-- The bases are positive -/
  h2 : a > 0
  h3 : c > 0
  /-- The radius is positive -/
  h4 : r > 0
  /-- Relation between radius and bases -/
  h5 : r = (a * c) / (a + c)

/-- The theorem to be proved -/
theorem right_trapezoid_inscribed_circle_theorem (t : RightTrapezoidWithInscribedCircle) :
  (2 : ℝ) * t.r = 2 / ((1 / t.a) + (1 / t.c)) :=
by sorry

end NUMINAMATH_CALUDE_right_trapezoid_inscribed_circle_theorem_l2761_276136


namespace NUMINAMATH_CALUDE_squats_on_third_day_l2761_276181

/-- Calculates the number of squats on a given day, given the initial number and daily increase. -/
def squatsOnDay (initialSquats : ℕ) (dailyIncrease : ℕ) (day : ℕ) : ℕ :=
  initialSquats + (day * dailyIncrease)

/-- Theorem: Given an initial number of 30 squats and a daily increase of 5 squats,
    the number of squats on the third day will be 45. -/
theorem squats_on_third_day :
  squatsOnDay 30 5 2 = 45 := by
  sorry


end NUMINAMATH_CALUDE_squats_on_third_day_l2761_276181


namespace NUMINAMATH_CALUDE_concert_ticket_revenue_l2761_276198

theorem concert_ticket_revenue 
  (total_tickets : ℕ) 
  (total_revenue : ℕ) 
  (full_price : ℕ) 
  (discounted_price : ℕ) 
  (full_price_tickets : ℕ) 
  (discounted_tickets : ℕ) :
  total_tickets = 200 →
  total_revenue = 2800 →
  discounted_price = (3 * full_price) / 4 →
  total_tickets = full_price_tickets + discounted_tickets →
  total_revenue = full_price * full_price_tickets + discounted_price * discounted_tickets →
  full_price_tickets * full_price = 680 :=
by sorry

end NUMINAMATH_CALUDE_concert_ticket_revenue_l2761_276198


namespace NUMINAMATH_CALUDE_negative_square_power_two_l2761_276120

theorem negative_square_power_two (a b : ℝ) : (-a^2 * b)^2 = a^4 * b^2 := by sorry

end NUMINAMATH_CALUDE_negative_square_power_two_l2761_276120


namespace NUMINAMATH_CALUDE_least_number_with_remainder_four_l2761_276174

theorem least_number_with_remainder_four (n : ℕ) : 
  (∀ m : ℕ, m > 0 → n % m = 4) → 
  (n % 12 = 4) → 
  n ≥ 40 :=
by
  sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_four_l2761_276174


namespace NUMINAMATH_CALUDE_michael_twice_jacob_age_l2761_276143

theorem michael_twice_jacob_age (jacob_current_age : ℕ) (michael_current_age : ℕ) : 
  jacob_current_age = 11 - 4 →
  michael_current_age = jacob_current_age + 12 →
  ∃ x : ℕ, michael_current_age + x = 2 * (jacob_current_age + x) ∧ x = 5 :=
by sorry

end NUMINAMATH_CALUDE_michael_twice_jacob_age_l2761_276143


namespace NUMINAMATH_CALUDE_no_integer_roots_l2761_276196

theorem no_integer_roots (a b c : ℤ) (ha : a ≠ 0)
  (h0 : Odd (a * 0^2 + b * 0 + c))
  (h1 : Odd (a * 1^2 + b * 1 + c)) :
  ∀ t : ℤ, a * t^2 + b * t + c ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_integer_roots_l2761_276196


namespace NUMINAMATH_CALUDE_circle_center_distance_l2761_276168

theorem circle_center_distance (x y : ℝ) : 
  x^2 + y^2 = 6*x + 8*y + 3 →
  Real.sqrt ((10 - x)^2 + (5 - y)^2) = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_distance_l2761_276168


namespace NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l2761_276175

/-- Given a line L1 defined by 3x - 2y = 6, and a line L2 perpendicular to L1 with y-intercept 4,
    the x-intercept of L2 is 6. -/
theorem perpendicular_line_x_intercept :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 3 * x - 2 * y = 6
  let m1 : ℝ := 3 / 2  -- slope of L1
  let m2 : ℝ := -2 / 3  -- slope of L2 (perpendicular to L1)
  let L2 : ℝ → ℝ → Prop := λ x y ↦ y = m2 * x + 4  -- equation of L2
  let x_intercept : ℝ := 6
  (∀ x y, L2 x y ↔ y = m2 * x + 4) →  -- L2 is defined correctly
  (m1 * m2 = -1) →  -- L1 and L2 are perpendicular
  L2 x_intercept 0  -- (6, 0) satisfies the equation of L2
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l2761_276175


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2761_276194

theorem complex_equation_solution (z : ℂ) (a : ℝ) 
  (h1 : Complex.abs z = 2) 
  (h2 : (z - a)^2 = a) : 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2761_276194


namespace NUMINAMATH_CALUDE_closest_integer_to_sqrt_29_l2761_276179

theorem closest_integer_to_sqrt_29 :
  ∃ (n : ℤ), ∀ (m : ℤ), |n - Real.sqrt 29| ≤ |m - Real.sqrt 29| ∧ n = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_sqrt_29_l2761_276179


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2761_276123

theorem complex_equation_sum (a b : ℝ) : 
  (a : ℂ) + b * Complex.I = (11 - 7 * Complex.I) / (1 - 2 * Complex.I) → a + b = 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2761_276123


namespace NUMINAMATH_CALUDE_max_third_side_length_l2761_276104

theorem max_third_side_length (a b : ℝ) (ha : a = 7) (hb : b = 10) :
  ∃ (x : ℕ), x ≤ 16 ∧
    (∀ (y : ℕ), (y : ℝ) + a > b ∧ (y : ℝ) + b > a ∧ a + b > (y : ℝ) → y ≤ x) ∧
    ((16 : ℝ) + a > b ∧ (16 : ℝ) + b > a ∧ a + b > 16) :=
by sorry

end NUMINAMATH_CALUDE_max_third_side_length_l2761_276104


namespace NUMINAMATH_CALUDE_unique_four_digit_square_l2761_276155

/-- Represents a four-digit number --/
def FourDigitNumber := { n : ℕ // 1000 ≤ n ∧ n < 10000 }

/-- Extracts the thousands digit from a four-digit number --/
def thousandsDigit (n : FourDigitNumber) : ℕ := n.val / 1000

/-- Extracts the hundreds digit from a four-digit number --/
def hundredsDigit (n : FourDigitNumber) : ℕ := (n.val / 100) % 10

/-- Extracts the tens digit from a four-digit number --/
def tensDigit (n : FourDigitNumber) : ℕ := (n.val / 10) % 10

/-- Extracts the units digit from a four-digit number --/
def unitsDigit (n : FourDigitNumber) : ℕ := n.val % 10

/-- Checks if a natural number is a perfect square --/
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem unique_four_digit_square : 
  ∃! (n : FourDigitNumber), 
    isPerfectSquare n.val ∧ 
    thousandsDigit n = tensDigit n ∧ 
    hundredsDigit n = unitsDigit n + 1 ∧
    n.val = 8281 :=
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_square_l2761_276155


namespace NUMINAMATH_CALUDE_tate_education_ratio_tate_education_ratio_simplified_l2761_276160

/-- Represents the duration of Tate's education -/
structure TateEducation where
  normalHighSchool : ℕ  -- Normal duration of high school
  tateHighSchool : ℕ    -- Tate's actual duration in high school
  collegeFactor : ℕ     -- Factor of high school time spent in college
  totalYears : ℕ        -- Total years spent in education

/-- Conditions for Tate's education -/
def validTateEducation (e : TateEducation) : Prop :=
  e.tateHighSchool = e.normalHighSchool - 1 ∧
  e.totalYears = e.tateHighSchool + e.collegeFactor * e.tateHighSchool ∧
  e.totalYears = 12

/-- Theorem stating the ratio of college to high school time -/
theorem tate_education_ratio (e : TateEducation) 
  (h : validTateEducation e) : 
  e.collegeFactor = 3 := by
  sorry

/-- Corollary stating the ratio in simplified form -/
theorem tate_education_ratio_simplified (e : TateEducation) 
  (h : validTateEducation e) : 
  e.collegeFactor * e.tateHighSchool / e.tateHighSchool = 3 := by
  sorry

end NUMINAMATH_CALUDE_tate_education_ratio_tate_education_ratio_simplified_l2761_276160


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l2761_276133

theorem smallest_square_containing_circle (r : ℝ) (h : r = 7) : 
  (2 * r) ^ 2 = 196 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l2761_276133


namespace NUMINAMATH_CALUDE_fermat_coprime_and_infinite_primes_l2761_276154

def fermat (n : ℕ) : ℕ := 2^(2^n) + 1

theorem fermat_coprime_and_infinite_primes :
  (∀ n m : ℕ, n ≠ m → Nat.gcd (fermat n) (fermat m) = 1) ∧
  (¬ ∃ N : ℕ, ∀ p : ℕ, Prime p → p ≤ N) :=
sorry

end NUMINAMATH_CALUDE_fermat_coprime_and_infinite_primes_l2761_276154


namespace NUMINAMATH_CALUDE_least_five_digit_divisible_by_12_15_18_l2761_276161

theorem least_five_digit_divisible_by_12_15_18 :
  ∀ n : ℕ, 
    10000 ≤ n ∧ n < 100000 ∧ 
    12 ∣ n ∧ 15 ∣ n ∧ 18 ∣ n →
    10080 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_divisible_by_12_15_18_l2761_276161


namespace NUMINAMATH_CALUDE_binomial_square_difference_specific_case_l2761_276108

theorem binomial_square_difference (a b : ℕ) : (a + b)^2 - (a^2 + b^2) = 2 * a * b := by sorry

theorem specific_case : (45 + 15)^2 - (45^2 + 15^2) = 1350 := by sorry

end NUMINAMATH_CALUDE_binomial_square_difference_specific_case_l2761_276108


namespace NUMINAMATH_CALUDE_sunday_production_l2761_276151

/-- The number of toys produced on a given day of the week -/
def toysProduced (day : Nat) : Nat :=
  2500 + 25 * day

/-- The number of days in a week -/
def daysInWeek : Nat := 7

/-- Theorem stating that the number of toys produced on Sunday (day 6) is 2650 -/
theorem sunday_production :
  toysProduced (daysInWeek - 1) = 2650 := by
  sorry


end NUMINAMATH_CALUDE_sunday_production_l2761_276151


namespace NUMINAMATH_CALUDE_suit_price_increase_l2761_276158

/-- Proves that the percentage increase in the price of a suit was 25% --/
theorem suit_price_increase (original_price : ℝ) (final_price : ℝ) : 
  original_price = 200 →
  final_price = 187.5 →
  ∃ (increase_percentage : ℝ),
    increase_percentage = 25 ∧
    final_price = (original_price + original_price * (increase_percentage / 100)) * 0.75 :=
by sorry

end NUMINAMATH_CALUDE_suit_price_increase_l2761_276158


namespace NUMINAMATH_CALUDE_james_louise_ages_l2761_276188

/-- James and Louise's ages problem -/
theorem james_louise_ages (j l : ℝ) : 
  j = l + 9 →                   -- James is nine years older than Louise
  j + 8 = 3 * (l - 4) →         -- Eight years from now, James will be three times as old as Louise was four years ago
  j + l = 38 :=                 -- The sum of their current ages is 38
by
  sorry

end NUMINAMATH_CALUDE_james_louise_ages_l2761_276188


namespace NUMINAMATH_CALUDE_snail_return_time_is_integer_l2761_276184

/-- Represents the snail's position on the plane -/
structure SnailPosition :=
  (x : ℝ) (y : ℝ)

/-- Represents the snail's movement parameters -/
structure SnailMovement :=
  (speed : ℝ)
  (turnAngle : ℝ)
  (turnInterval : ℝ)

/-- Calculates the snail's position after a given time -/
def snailPositionAfterTime (initialPos : SnailPosition) (movement : SnailMovement) (time : ℝ) : SnailPosition :=
  sorry

/-- Checks if the snail has returned to the origin -/
def hasReturnedToOrigin (pos : SnailPosition) : Prop :=
  pos.x = 0 ∧ pos.y = 0

/-- Theorem: The snail can only return to the origin after an integer number of hours -/
theorem snail_return_time_is_integer 
  (movement : SnailMovement) 
  (h1 : movement.speed > 0)
  (h2 : movement.turnAngle = π / 3)
  (h3 : movement.turnInterval = 1 / 2) :
  ∀ t : ℝ, hasReturnedToOrigin (snailPositionAfterTime ⟨0, 0⟩ movement t) → ∃ n : ℕ, t = n :=
sorry

end NUMINAMATH_CALUDE_snail_return_time_is_integer_l2761_276184


namespace NUMINAMATH_CALUDE_selenes_purchase_cost_l2761_276101

/-- The total cost of Selene's purchase after discount -/
def total_cost_after_discount (camera_price : ℝ) (frame_price : ℝ) (num_cameras : ℕ) (num_frames : ℕ) (discount_rate : ℝ) : ℝ :=
  let total_before_discount := camera_price * num_cameras + frame_price * num_frames
  let discount := discount_rate * total_before_discount
  total_before_discount - discount

/-- Theorem stating that Selene's total payment is $551 -/
theorem selenes_purchase_cost : 
  total_cost_after_discount 110 120 2 3 0.05 = 551 := by
  sorry

#eval total_cost_after_discount 110 120 2 3 0.05

end NUMINAMATH_CALUDE_selenes_purchase_cost_l2761_276101


namespace NUMINAMATH_CALUDE_least_possible_value_z_minus_x_l2761_276140

theorem least_possible_value_z_minus_x 
  (x y z : ℤ) 
  (h1 : x < y ∧ y < z) 
  (h2 : y - x > 11) 
  (h3 : Even x) 
  (h4 : Odd y ∧ Odd z) :
  ∀ w, w = z - x → w ≥ 15 ∧ ∃ (x' y' z' : ℤ), 
    x' < y' ∧ y' < z' ∧ 
    y' - x' > 11 ∧ 
    Even x' ∧ Odd y' ∧ Odd z' ∧ 
    z' - x' = 15 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_value_z_minus_x_l2761_276140


namespace NUMINAMATH_CALUDE_square_side_increase_l2761_276189

theorem square_side_increase (p : ℝ) : 
  ((1 + p / 100) ^ 2 = 1.44) → p = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_side_increase_l2761_276189


namespace NUMINAMATH_CALUDE_inscribed_square_diagonal_l2761_276121

theorem inscribed_square_diagonal (length width : ℝ) (h1 : length = 8) (h2 : width = 6) :
  let inscribed_square_side := width
  let inscribed_square_area := inscribed_square_side ^ 2
  let third_square_area := 9 * inscribed_square_area
  let third_square_side := Real.sqrt third_square_area
  let third_square_diagonal := third_square_side * Real.sqrt 2
  third_square_diagonal = 18 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_diagonal_l2761_276121


namespace NUMINAMATH_CALUDE_beth_peas_cans_l2761_276103

/-- The number of cans of corn Beth bought -/
def corn_cans : ℕ := 10

/-- The number of cans of peas Beth bought -/
def peas_cans : ℕ := 2 * corn_cans + 15

theorem beth_peas_cans : peas_cans = 35 := by
  sorry

end NUMINAMATH_CALUDE_beth_peas_cans_l2761_276103


namespace NUMINAMATH_CALUDE_farmers_market_total_sales_l2761_276144

/-- Calculates the total sales from a farmers' market given specific conditions --/
theorem farmers_market_total_sales :
  let broccoli_sales : ℕ := 57
  let carrot_sales : ℕ := 2 * broccoli_sales
  let spinach_sales : ℕ := carrot_sales / 2 + 16
  let cauliflower_sales : ℕ := 136
  broccoli_sales + carrot_sales + spinach_sales + cauliflower_sales = 380 :=
by
  sorry


end NUMINAMATH_CALUDE_farmers_market_total_sales_l2761_276144


namespace NUMINAMATH_CALUDE_smallest_divisor_l2761_276193

theorem smallest_divisor (N D : ℕ) (q1 q2 k : ℕ) : 
  N = D * q1 + 75 →
  N = 37 * q2 + 1 →
  D > 75 →
  D = 37 * k + 38 →
  112 ≤ D :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_l2761_276193


namespace NUMINAMATH_CALUDE_fraction_meaningful_l2761_276170

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (x + 2) / (x - 1)) ↔ x ≠ 1 := by
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l2761_276170


namespace NUMINAMATH_CALUDE_cos_angle_POQ_l2761_276147

theorem cos_angle_POQ (P Q : ℝ × ℝ) : 
  (P.1^2 + P.2^2 = 1) →  -- P is on the unit circle
  (Q.1^2 + Q.2^2 = 1) →  -- Q is on the unit circle
  (P.1 ≥ 0 ∧ P.2 > 0) →  -- P is in the first quadrant
  (Q.1 > 0 ∧ Q.2 ≤ 0) →  -- Q is in the fourth quadrant
  P.2 = 4/5 →           -- vertical coordinate of P is 4/5
  Q.1 = 5/13 →          -- horizontal coordinate of Q is 5/13
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 2 - 2 * (P.1 * Q.1 + P.2 * Q.2) →  -- cosine formula
  P.1 * Q.1 + P.2 * Q.2 = -33/65 :=
by sorry

end NUMINAMATH_CALUDE_cos_angle_POQ_l2761_276147


namespace NUMINAMATH_CALUDE_alice_prob_three_turns_l2761_276113

/-- Represents the person holding the ball -/
inductive Person : Type
| Alice : Person
| Bob : Person

/-- The probability of tossing the ball for each person -/
def toss_prob (p : Person) : ℚ :=
  match p with
  | Person.Alice => 1/3
  | Person.Bob => 1/4

/-- The probability of keeping the ball for each person -/
def keep_prob (p : Person) : ℚ :=
  1 - toss_prob p

/-- The probability of Alice having the ball after n turns, given she starts with it -/
def alice_prob (n : ℕ) : ℚ :=
  sorry

theorem alice_prob_three_turns :
  alice_prob 3 = 227/432 :=
sorry

end NUMINAMATH_CALUDE_alice_prob_three_turns_l2761_276113


namespace NUMINAMATH_CALUDE_goldfinch_percentage_is_30_percent_l2761_276119

def number_of_goldfinches : ℕ := 6
def number_of_sparrows : ℕ := 9
def number_of_grackles : ℕ := 5

def total_birds : ℕ := number_of_goldfinches + number_of_sparrows + number_of_grackles

def goldfinch_percentage : ℚ := (number_of_goldfinches : ℚ) / (total_birds : ℚ) * 100

theorem goldfinch_percentage_is_30_percent : goldfinch_percentage = 30 := by
  sorry

end NUMINAMATH_CALUDE_goldfinch_percentage_is_30_percent_l2761_276119


namespace NUMINAMATH_CALUDE_five_digit_divisible_by_nine_l2761_276192

theorem five_digit_divisible_by_nine :
  ∀ x : ℕ, x < 10 →
  (738 * 10 + x) * 10 + 5 ≡ 0 [MOD 9] ↔ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_divisible_by_nine_l2761_276192


namespace NUMINAMATH_CALUDE_solid_volume_l2761_276152

/-- A solid with specific face dimensions -/
structure Solid where
  square_side : ℝ
  rect_length : ℝ
  rect_width : ℝ
  trapezoid_leg : ℝ

/-- The volume of the solid -/
noncomputable def volume (s : Solid) : ℝ := sorry

/-- Theorem stating that the volume of the specified solid is 552 dm³ -/
theorem solid_volume (s : Solid) 
  (h1 : s.square_side = 1) 
  (h2 : s.rect_length = 0.4)
  (h3 : s.rect_width = 0.2)
  (h4 : s.trapezoid_leg = 1.3) :
  volume s = 0.552 := by sorry

end NUMINAMATH_CALUDE_solid_volume_l2761_276152


namespace NUMINAMATH_CALUDE_billy_video_count_l2761_276178

theorem billy_video_count 
  (suggestions_per_round : ℕ) 
  (num_rounds : ℕ) 
  (final_pick : ℕ) :
  suggestions_per_round = 15 →
  num_rounds = 5 →
  final_pick = 5 →
  suggestions_per_round * num_rounds - (suggestions_per_round - final_pick) = 65 :=
by
  sorry

end NUMINAMATH_CALUDE_billy_video_count_l2761_276178


namespace NUMINAMATH_CALUDE_kamal_math_marks_l2761_276163

/-- Represents a student's marks in various subjects -/
structure StudentMarks where
  english : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ
  mathematics : ℕ
  average : ℕ
  total_subjects : ℕ

/-- Theorem stating that given Kamal's marks and average, his Mathematics marks must be 60 -/
theorem kamal_math_marks (kamal : StudentMarks) 
  (h1 : kamal.english = 76)
  (h2 : kamal.physics = 72)
  (h3 : kamal.chemistry = 65)
  (h4 : kamal.biology = 82)
  (h5 : kamal.average = 71)
  (h6 : kamal.total_subjects = 5) :
  kamal.mathematics = 60 := by
  sorry

end NUMINAMATH_CALUDE_kamal_math_marks_l2761_276163


namespace NUMINAMATH_CALUDE_multiply_and_simplify_l2761_276156

theorem multiply_and_simplify (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_simplify_l2761_276156


namespace NUMINAMATH_CALUDE_f_neg_a_value_l2761_276131

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x^2) - x) + 1

theorem f_neg_a_value (a : ℝ) (h : f a = 4) : f (-a) = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_a_value_l2761_276131


namespace NUMINAMATH_CALUDE_solution_of_equation_l2761_276195

theorem solution_of_equation (z : ℂ) : 
  (z^6 - 6*z^4 + 9*z^2 = 0) ↔ (z = -Real.sqrt 3 ∨ z = 0 ∨ z = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_solution_of_equation_l2761_276195


namespace NUMINAMATH_CALUDE_circular_coin_flip_probability_l2761_276107

def num_people : ℕ := 10

-- Function to calculate the number of valid arrangements
def valid_arrangements : ℕ → ℕ
| 0 => 1
| 1 => 2
| 2 => 3
| n + 3 => valid_arrangements (n + 1) + valid_arrangements (n + 2)

theorem circular_coin_flip_probability :
  (valid_arrangements num_people : ℚ) / 2^num_people = 123 / 1024 := by sorry

end NUMINAMATH_CALUDE_circular_coin_flip_probability_l2761_276107


namespace NUMINAMATH_CALUDE_problem_statement_l2761_276150

theorem problem_statement (n : ℤ) (a : ℝ) : 
  (6 * 11 * n > 0) → (a^(2*n) = 5) → (2 * a^(6*n) - 4 = 246) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2761_276150


namespace NUMINAMATH_CALUDE_problem_statement_l2761_276114

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

def g (a : ℝ) (x : ℝ) : ℝ := -x^2 + a*x - 3

theorem problem_statement :
  (∀ a : ℝ, (∀ x : ℝ, x > 0 → f x ≥ (1/2) * g a x) → a ≤ 4) ∧
  (∀ x : ℝ, x > 0 → log x > 1/exp x - 2/(exp 1 * x)) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2761_276114


namespace NUMINAMATH_CALUDE_infinite_primes_with_cubic_solution_l2761_276134

theorem infinite_primes_with_cubic_solution :
  ∃ (S : Set Nat), (∀ p ∈ S, Nat.Prime p) ∧ Set.Infinite S ∧
    (∀ p ∈ S, ∃ x : Nat, x^2 + x + 1 ≡ 0 [ZMOD p]) := by
  sorry

end NUMINAMATH_CALUDE_infinite_primes_with_cubic_solution_l2761_276134


namespace NUMINAMATH_CALUDE_total_rainfall_equals_1368_l2761_276129

def average_monthly_rainfall (year : ℕ) : ℝ :=
  35 + 3 * (year - 2010)

def yearly_rainfall (year : ℕ) : ℝ :=
  12 * average_monthly_rainfall year

def total_rainfall_2010_to_2012 : ℝ :=
  yearly_rainfall 2010 + yearly_rainfall 2011 + yearly_rainfall 2012

theorem total_rainfall_equals_1368 :
  total_rainfall_2010_to_2012 = 1368 := by sorry

end NUMINAMATH_CALUDE_total_rainfall_equals_1368_l2761_276129


namespace NUMINAMATH_CALUDE_pure_imaginary_solutions_l2761_276191

theorem pure_imaginary_solutions :
  let f : ℂ → ℂ := λ x => x^4 - 5*x^3 + 10*x^2 - 50*x - 75
  ∀ x : ℂ, (∃ k : ℝ, x = k * I) → (f x = 0 ↔ x = Complex.I * Real.sqrt 10 ∨ x = -Complex.I * Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_solutions_l2761_276191


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2761_276183

theorem smallest_n_congruence : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → ¬(528 * m ≡ 1068 * m [MOD 30])) ∧ 
  (528 * n ≡ 1068 * n [MOD 30]) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2761_276183


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2761_276112

/-- A quadratic function with vertex (m, k) and point (k, m) on its graph -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  m : ℝ
  k : ℝ
  a_nonzero : a ≠ 0
  vertex_condition : k = a * m^2 + b * m + c
  point_condition : m = a * k^2 + b * k + c

/-- Theorem stating that a(m - k) > 0 for a quadratic function with the given conditions -/
theorem quadratic_function_property (f : QuadraticFunction) : f.a * (f.m - f.k) > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2761_276112


namespace NUMINAMATH_CALUDE_first_positive_term_is_seventh_l2761_276153

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1 : ℚ) * d

theorem first_positive_term_is_seventh :
  let a₁ := -1
  let d := 1/5
  (∀ k < 7, arithmetic_sequence a₁ d k ≤ 0) ∧
  (arithmetic_sequence a₁ d 7 > 0) :=
by sorry

end NUMINAMATH_CALUDE_first_positive_term_is_seventh_l2761_276153


namespace NUMINAMATH_CALUDE_max_gaming_average_l2761_276169

theorem max_gaming_average (wednesday_hours : ℝ) (thursday_hours : ℝ) (tom_hours : ℝ) (fred_hours : ℝ) (additional_time : ℝ) :
  wednesday_hours = 2 →
  thursday_hours = 2 →
  tom_hours = 4 →
  fred_hours = 6 →
  additional_time = 0.5 →
  let total_hours := wednesday_hours + thursday_hours + max tom_hours fred_hours + additional_time
  let days := 3
  let average_hours := total_hours / days
  average_hours = 3.5 := by
sorry

end NUMINAMATH_CALUDE_max_gaming_average_l2761_276169


namespace NUMINAMATH_CALUDE_triangle_area_l2761_276110

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) : 
  (1/2) * a * b = 54 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2761_276110


namespace NUMINAMATH_CALUDE_parabola_distance_theorem_l2761_276102

/-- Parabola type -/
structure Parabola where
  /-- The equation of the parabola y^2 = 8x -/
  equation : ℝ → ℝ → Prop
  /-- The focus of the parabola -/
  focus : ℝ × ℝ
  /-- The directrix of the parabola -/
  directrix : ℝ → ℝ → Prop

/-- Point on the directrix -/
def PointOnDirectrix (p : Parabola) : Type := { point : ℝ × ℝ // p.directrix point.1 point.2 }

/-- Point on the parabola -/
def PointOnParabola (p : Parabola) : Type := { point : ℝ × ℝ // p.equation point.1 point.2 }

/-- Theorem: For a parabola y^2 = 8x, if FP = 4FQ, then |QF| = 3 -/
theorem parabola_distance_theorem (p : Parabola) 
  (hpeq : p.equation = fun x y ↦ y^2 = 8*x)
  (P : PointOnDirectrix p) 
  (Q : PointOnParabola p) 
  (hline : ∃ (t : ℝ), Q.val = p.focus + t • (P.val - p.focus))
  (hfp : ‖P.val - p.focus‖ = 4 * ‖Q.val - p.focus‖) :
  ‖Q.val - p.focus‖ = 3 := by sorry

end NUMINAMATH_CALUDE_parabola_distance_theorem_l2761_276102


namespace NUMINAMATH_CALUDE_conspiracy_split_l2761_276182

theorem conspiracy_split (S : Finset (Finset Nat)) :
  S.card = 6 →
  (∀ s ∈ S, s.card = 3) →
  (∃ T : Finset Nat, T ⊆ Finset.range 6 ∧ T.card = 3 ∧
    ∀ s ∈ S, (s ⊆ T → False) ∧ (s ⊆ (Finset.range 6 \ T) → False)) :=
by sorry

end NUMINAMATH_CALUDE_conspiracy_split_l2761_276182


namespace NUMINAMATH_CALUDE_xy_value_l2761_276130

theorem xy_value (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 126) : x * y = -5 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2761_276130


namespace NUMINAMATH_CALUDE_cos_75_cos_15_plus_sin_75_sin_15_l2761_276197

theorem cos_75_cos_15_plus_sin_75_sin_15 :
  Real.cos (75 * π / 180) * Real.cos (15 * π / 180) +
  Real.sin (75 * π / 180) * Real.sin (15 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_cos_15_plus_sin_75_sin_15_l2761_276197


namespace NUMINAMATH_CALUDE_vector_collinearity_l2761_276109

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem vector_collinearity :
  let a : ℝ × ℝ := (3, 6)
  let b : ℝ × ℝ := (x, 8)
  collinear a b → x = 4 :=
by sorry

end NUMINAMATH_CALUDE_vector_collinearity_l2761_276109


namespace NUMINAMATH_CALUDE_abs_x_minus_one_leq_two_solution_set_l2761_276126

theorem abs_x_minus_one_leq_two_solution_set :
  {x : ℝ | |x - 1| ≤ 2} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_abs_x_minus_one_leq_two_solution_set_l2761_276126


namespace NUMINAMATH_CALUDE_equation_solution_l2761_276148

theorem equation_solution : ∃ x : ℝ, (1 / (x - 3) - 2 = (x - 1) / (3 - x)) ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2761_276148


namespace NUMINAMATH_CALUDE_rebus_sum_l2761_276146

theorem rebus_sum : ∃ (EX OY AY OH : ℕ+), 
  EX = 4 * OY ∧ 
  AY = 4 * OH ∧ 
  EX + OY + AY + OH = 150 := by
  sorry

end NUMINAMATH_CALUDE_rebus_sum_l2761_276146


namespace NUMINAMATH_CALUDE_isabel_candy_theorem_l2761_276116

/-- The number of candy pieces Isabel has left after distribution -/
def remaining_candy (initial : ℕ) (friend : ℕ) (cousin : ℕ) (sister : ℕ) (distributed : ℕ) : ℤ :=
  (initial + friend + cousin + sister : ℤ) - distributed

/-- Theorem stating the number of candy pieces Isabel has left -/
theorem isabel_candy_theorem (x y z : ℕ) :
  remaining_candy 325 145 x y z = 470 + x + y - z := by
  sorry

end NUMINAMATH_CALUDE_isabel_candy_theorem_l2761_276116


namespace NUMINAMATH_CALUDE_interior_angle_non_integer_count_l2761_276167

theorem interior_angle_non_integer_count :
  ∃! (n : ℕ), 3 ≤ n ∧ n ≤ 10 ∧ ¬(∃ (k : ℕ), (180 * (n - 2)) / n = k) :=
by sorry

end NUMINAMATH_CALUDE_interior_angle_non_integer_count_l2761_276167


namespace NUMINAMATH_CALUDE_driver_net_pay_rate_l2761_276187

-- Define the parameters
def travel_time : ℝ := 3
def speed : ℝ := 50
def fuel_efficiency : ℝ := 25
def pay_rate : ℝ := 0.60
def gasoline_cost : ℝ := 2.50

-- Define the theorem
theorem driver_net_pay_rate :
  let total_distance := travel_time * speed
  let gasoline_used := total_distance / fuel_efficiency
  let gross_earnings := pay_rate * total_distance
  let gasoline_expense := gasoline_cost * gasoline_used
  let net_earnings := gross_earnings - gasoline_expense
  net_earnings / travel_time = 25 := by sorry

end NUMINAMATH_CALUDE_driver_net_pay_rate_l2761_276187


namespace NUMINAMATH_CALUDE_triangle_area_l2761_276142

/-- Given a triangle with perimeter 48 cm and inradius 2.5 cm, its area is 60 cm² -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) : 
  perimeter = 48 → inradius = 2.5 → area = perimeter / 2 * inradius → area = 60 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l2761_276142


namespace NUMINAMATH_CALUDE_multiply_by_three_plus_sqrt_five_equals_one_l2761_276100

theorem multiply_by_three_plus_sqrt_five_equals_one :
  let x := (3 - Real.sqrt 5) / 4
  x * (3 + Real.sqrt 5) = 1 := by
sorry

end NUMINAMATH_CALUDE_multiply_by_three_plus_sqrt_five_equals_one_l2761_276100


namespace NUMINAMATH_CALUDE_propositions_true_l2761_276199

theorem propositions_true : 
  (∀ (m : ℝ), (∀ (x : ℝ), x^2 + x + m > 0) → m > 1/4) ∧ 
  (∀ (A B C : ℝ), 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π → 
    (A > B ↔ Real.sin A > Real.sin B)) := by sorry

end NUMINAMATH_CALUDE_propositions_true_l2761_276199


namespace NUMINAMATH_CALUDE_math_problem_solutions_l2761_276176

theorem math_problem_solutions :
  (∃ (x : ℝ), x = Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24 ∧ x = 4 + Real.sqrt 6) ∧
  (∃ (y : ℝ), y = (Real.sqrt 27 - Real.sqrt 12) / Real.sqrt 3 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_math_problem_solutions_l2761_276176


namespace NUMINAMATH_CALUDE_four_intersection_points_l2761_276186

/-- The polynomial function representing the curve -/
def f (c : ℝ) (x : ℝ) : ℝ := x^4 + 9*x^3 + c*x^2 + 9*x + 4

/-- Theorem stating the condition for the existence of a line intersecting the curve in four distinct points -/
theorem four_intersection_points (c : ℝ) :
  (∃ (m n : ℝ), ∀ (x : ℝ), (f c x = m*x + n) → (∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    f c x₁ = m*x₁ + n ∧ f c x₂ = m*x₂ + n ∧ f c x₃ = m*x₃ + n ∧ f c x₄ = m*x₄ + n)) ↔
  c ≤ 243/8 :=
sorry

end NUMINAMATH_CALUDE_four_intersection_points_l2761_276186


namespace NUMINAMATH_CALUDE_optimal_output_l2761_276128

noncomputable section

/-- The defective rate as a function of daily output -/
def defective_rate (c : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ c then 1 / (6 - x) else 2 / 3

/-- The daily profit as a function of daily output -/
def daily_profit (c : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ c
  then (3 * (9 * x - 2 * x^2)) / (2 * (6 - x))
  else 0

/-- The theorem stating the optimal daily output for maximum profit -/
theorem optimal_output (c : ℝ) (h : 0 < c ∧ c < 6) :
  (∃ (x : ℝ), ∀ (y : ℝ), daily_profit c y ≤ daily_profit c x) →
  ((0 < c ∧ c < 3 → ∃ (x : ℝ), x = c ∧ ∀ (y : ℝ), daily_profit c y ≤ daily_profit c x) ∧
   (3 ≤ c ∧ c < 6 → ∃ (x : ℝ), x = 3 ∧ ∀ (y : ℝ), daily_profit c y ≤ daily_profit c x)) :=
by sorry

end

end NUMINAMATH_CALUDE_optimal_output_l2761_276128


namespace NUMINAMATH_CALUDE_sphere_volume_from_cross_section_l2761_276135

/-- Given a sphere with a circular cross-section of radius 4 and the distance
    from the sphere's center to the center of the cross-section is 3,
    prove that the volume of the sphere is (500/3)π. -/
theorem sphere_volume_from_cross_section (r : ℝ) (h : ℝ) :
  r^2 = 4^2 + 3^2 →
  (4 / 3) * π * r^3 = (500 / 3) * π := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_from_cross_section_l2761_276135


namespace NUMINAMATH_CALUDE_system_solution_unique_l2761_276190

theorem system_solution_unique :
  ∃! (x y z : ℝ), 
    x^2 - y*z = 1 ∧
    y^2 - x*z = 2 ∧
    z^2 - x*y = 3 ∧
    (x = 5*Real.sqrt 2/6 ∨ x = -5*Real.sqrt 2/6) ∧
    (y = -Real.sqrt 2/6 ∨ y = Real.sqrt 2/6) ∧
    (z = -7*Real.sqrt 2/6 ∨ z = 7*Real.sqrt 2/6) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_unique_l2761_276190


namespace NUMINAMATH_CALUDE_correct_average_calculation_l2761_276111

/-- Given a number of tables, women, and men, calculate the average number of customers per table. -/
def averageCustomersPerTable (tables : Float) (women : Float) (men : Float) : Float :=
  (women + men) / tables

/-- Theorem stating that for the given values, the average number of customers per table is correct. -/
theorem correct_average_calculation :
  averageCustomersPerTable 9.0 7.0 3.0 = (7.0 + 3.0) / 9.0 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_calculation_l2761_276111


namespace NUMINAMATH_CALUDE_min_value_sum_equality_condition_l2761_276115

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b)) + (b / (5 * c)) + (c / (6 * a)) ≥ 3 / Real.rpow 90 (1/3) :=
sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b)) + (b / (5 * c)) + (c / (6 * a)) = 3 / Real.rpow 90 (1/3) ↔
  (a / (3 * b)) = (b / (5 * c)) ∧ (b / (5 * c)) = (c / (6 * a)) ∧ 
  (c / (6 * a)) = Real.rpow (1/90) (1/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_equality_condition_l2761_276115


namespace NUMINAMATH_CALUDE_tangent_sum_equality_l2761_276173

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the triangle
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

-- Define the tangent line
def TangentLine (p : ℝ × ℝ) (c : Circle) : ℝ × ℝ → Prop := sorry

-- State that the circles are tangent
def CirclesTangent (c1 c2 : Circle) : Prop := sorry

-- State that the triangle is equilateral
def IsEquilateral (t : Triangle) : Prop := sorry

-- State that the triangle is inscribed in the larger circle
def Inscribed (t : Triangle) (c : Circle) : Prop := sorry

-- Define the length of a tangent line
def TangentLength (p : ℝ × ℝ) (c : Circle) : ℝ := sorry

-- Main theorem
theorem tangent_sum_equality 
  (c1 c2 : Circle) 
  (t : Triangle) 
  (h1 : CirclesTangent c1 c2) 
  (h2 : IsEquilateral t) 
  (h3 : Inscribed t c1) :
  ∃ (i j k : Fin 3), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    TangentLength (t.vertices i) c2 = 
    TangentLength (t.vertices j) c2 + TangentLength (t.vertices k) c2 :=
sorry

end NUMINAMATH_CALUDE_tangent_sum_equality_l2761_276173


namespace NUMINAMATH_CALUDE_hotel_reunions_l2761_276157

theorem hotel_reunions (total_guests : ℕ) (oates_attendees : ℕ) (hall_attendees : ℕ)
  (h1 : total_guests = 100)
  (h2 : oates_attendees = 50)
  (h3 : hall_attendees = 62)
  (h4 : ∀ g, g ≤ total_guests → (g ≤ oates_attendees ∨ g ≤ hall_attendees)) :
  oates_attendees + hall_attendees - total_guests = 12 := by
  sorry

end NUMINAMATH_CALUDE_hotel_reunions_l2761_276157


namespace NUMINAMATH_CALUDE_squares_in_6x6_grid_l2761_276105

/-- Calculates the number of squares in a grid with n+1 lines in each direction -/
def count_squares (n : ℕ) : ℕ := 
  (n * (n + 1) * (2 * n + 1)) / 6

/-- Theorem: In a 6x6 grid, the total number of squares is 55 -/
theorem squares_in_6x6_grid : count_squares 5 = 55 := by
  sorry

end NUMINAMATH_CALUDE_squares_in_6x6_grid_l2761_276105


namespace NUMINAMATH_CALUDE_ship_meetings_count_l2761_276124

/-- Represents the number of ships sailing in each direction -/
def num_ships_per_direction : ℕ := 5

/-- Represents the total number of ships -/
def total_ships : ℕ := 2 * num_ships_per_direction

/-- Calculates the total number of meetings between ships -/
def total_meetings : ℕ := num_ships_per_direction * num_ships_per_direction

/-- Theorem stating that the total number of meetings is 25 -/
theorem ship_meetings_count :
  total_meetings = 25 :=
by sorry

end NUMINAMATH_CALUDE_ship_meetings_count_l2761_276124


namespace NUMINAMATH_CALUDE_binomial_square_coeff_l2761_276159

theorem binomial_square_coeff (x : ℝ) : ∃ (r s : ℝ), 
  (r * x + s)^2 = (196 / 9) * x^2 + 28 * x + 9 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_coeff_l2761_276159


namespace NUMINAMATH_CALUDE_factorial_divisibility_l2761_276132

theorem factorial_divisibility (n : ℕ) (p : ℕ) (h_pos : n > 0) (h_prime : Nat.Prime p) 
  (h_div : p ^ p ∣ Nat.factorial n) : p ^ (p + 1) ∣ Nat.factorial n := by
  sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l2761_276132


namespace NUMINAMATH_CALUDE_shaded_area_rectangle_with_circles_l2761_276185

/-- Given a rectangle with width 30 inches and length 60 inches, and four identical circles
    each tangent to two adjacent sides of the rectangle and its neighboring circles,
    the total shaded area when the circles are excluded is 1800 - 225π square inches. -/
theorem shaded_area_rectangle_with_circles :
  let rectangle_width : ℝ := 30
  let rectangle_length : ℝ := 60
  let circle_radius : ℝ := rectangle_width / 4
  let rectangle_area : ℝ := rectangle_width * rectangle_length
  let circle_area : ℝ := π * circle_radius^2
  let total_circle_area : ℝ := 4 * circle_area
  let shaded_area : ℝ := rectangle_area - total_circle_area
  shaded_area = 1800 - 225 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_rectangle_with_circles_l2761_276185


namespace NUMINAMATH_CALUDE_triangle_properties_l2761_276125

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  A + B + C = π →
  c = Real.sqrt 3 →
  c * Real.tan C = Real.sqrt 3 * (a * Real.cos B + b * Real.cos A) →
  C = π/3 ∧ 0 < a - b/2 ∧ a - b/2 < 3/2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l2761_276125


namespace NUMINAMATH_CALUDE_equation_solution_l2761_276149

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  1 - 9 / x + 20 / x^2 = 0 → 2 / x = 1 / 2 ∨ 2 / x = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2761_276149


namespace NUMINAMATH_CALUDE_segment_sum_bound_l2761_276177

/-- A convex polygon in a 2D plane. -/
structure ConvexPolygon where
  -- We don't need to define the structure completely, just declare it
  area : ℝ

/-- A set of parallel lines in a 2D plane. -/
structure ParallelLines where
  -- Again, we don't need to define this completely
  count : ℕ
  spacing : ℝ

/-- The sum of lengths of segments cut by a polygon on parallel lines. -/
def sumOfSegments (polygon : ConvexPolygon) (lines : ParallelLines) : ℝ :=
  sorry -- Definition not provided, just declared

/-- Theorem statement -/
theorem segment_sum_bound
  (polygon : ConvexPolygon)
  (lines : ParallelLines)
  (h_area : polygon.area = 9)
  (h_lines : lines.count = 9)
  (h_spacing : lines.spacing = 1) :
  sumOfSegments polygon lines ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_segment_sum_bound_l2761_276177


namespace NUMINAMATH_CALUDE_geraint_on_time_speed_l2761_276162

/-- The distance Geraint cycles to work in kilometers. -/
def distance : ℝ := sorry

/-- The time in hours that Geraint's journey should take to arrive on time. -/
def on_time : ℝ := sorry

/-- The speed in km/h at which Geraint arrives on time. -/
def on_time_speed : ℝ := sorry

/-- Theorem stating that Geraint's on-time speed is 20 km/h. -/
theorem geraint_on_time_speed : 
  (distance / 15 = on_time + 1/6) →  -- At 15 km/h, he's 10 minutes (1/6 hour) late
  (distance / 30 = on_time - 1/6) →  -- At 30 km/h, he's 10 minutes (1/6 hour) early
  on_time_speed = 20 := by
  sorry

end NUMINAMATH_CALUDE_geraint_on_time_speed_l2761_276162


namespace NUMINAMATH_CALUDE_melissa_games_l2761_276166

/-- The number of games Melissa played -/
def number_of_games (total_points : ℕ) (points_per_game : ℕ) : ℕ :=
  total_points / points_per_game

/-- Proof that Melissa played 3 games -/
theorem melissa_games : number_of_games 21 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_melissa_games_l2761_276166


namespace NUMINAMATH_CALUDE_average_of_abc_l2761_276164

theorem average_of_abc (A B C : ℚ) 
  (eq1 : 2002 * C + 4004 * A = 8008)
  (eq2 : 3003 * B - 5005 * A = 7007) : 
  (A + B + C) / 3 = 22 / 9 := by sorry

end NUMINAMATH_CALUDE_average_of_abc_l2761_276164


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l2761_276180

theorem circle_center_and_radius :
  let eq := fun (x y : ℝ) => x^2 - 6*x + y^2 + 2*y - 9 = 0
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (3, -1) ∧ 
    radius = Real.sqrt 19 ∧
    ∀ (x y : ℝ), eq x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l2761_276180


namespace NUMINAMATH_CALUDE_log_2_bounds_l2761_276127

theorem log_2_bounds (h1 : 10^3 = 1000) (h2 : 10^4 = 10000) 
                     (h3 : 2^11 = 2048) (h4 : 2^14 = 16384) :
  3/11 < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < 2/7 := by
  sorry

end NUMINAMATH_CALUDE_log_2_bounds_l2761_276127


namespace NUMINAMATH_CALUDE_doubled_average_l2761_276118

theorem doubled_average (n : ℕ) (original_average : ℝ) (h1 : n = 30) (h2 : original_average = 45) :
  let new_average := 2 * original_average
  new_average = 90 := by
sorry

end NUMINAMATH_CALUDE_doubled_average_l2761_276118


namespace NUMINAMATH_CALUDE_triangle_vector_intersection_l2761_276137

/-- Given a triangle XYZ with points M, N, and Q satisfying specific conditions,
    prove that Q can be expressed as a linear combination of X, Y, and Z with specific coefficients. -/
theorem triangle_vector_intersection (X Y Z M N Q : ℝ × ℝ) : 
  (∃ (k : ℝ), M = k • Z + (1 - k) • Y ∧ k = 1/5) →  -- M lies on YZ extended
  (∃ (l : ℝ), N = l • X + (1 - l) • Z ∧ l = 3/5) →  -- N lies on XZ
  (∃ (s t : ℝ), Q = s • Y + (1 - s) • N ∧ Q = t • X + (1 - t) • M) →  -- Q is intersection of YN and XM
  Q = (12/23) • X + (3/23) • Y + (8/23) • Z :=
by sorry

end NUMINAMATH_CALUDE_triangle_vector_intersection_l2761_276137


namespace NUMINAMATH_CALUDE_positive_root_existence_l2761_276141

def f (x : ℝ) := x^5 - x - 1

theorem positive_root_existence :
  ∃ x ∈ Set.Icc 1 2, f x = 0 ∧ x > 0 :=
sorry

end NUMINAMATH_CALUDE_positive_root_existence_l2761_276141


namespace NUMINAMATH_CALUDE_kangaroo_cant_reach_far_l2761_276122

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines a valid jump for the kangaroo -/
def validJump (p q : Point) : Prop :=
  (q.x = p.x + 1 ∧ q.y = p.y - 1) ∨ (q.x = p.x - 5 ∧ q.y = p.y + 7)

/-- Defines if a point is in the first quadrant -/
def inFirstQuadrant (p : Point) : Prop :=
  p.x ≥ 0 ∧ p.y ≥ 0

/-- Defines if a point is at least 1000 units away from the origin -/
def farFromOrigin (p : Point) : Prop :=
  p.x^2 + p.y^2 ≥ 1000000

/-- Defines if a point can be reached through a sequence of valid jumps -/
def canReach (start target : Point) : Prop :=
  ∃ (n : ℕ) (path : ℕ → Point), 
    path 0 = start ∧ 
    path n = target ∧ 
    ∀ i < n, validJump (path i) (path (i+1)) ∧ inFirstQuadrant (path (i+1))

/-- The main theorem to be proved -/
theorem kangaroo_cant_reach_far (p : Point) 
  (h1 : inFirstQuadrant p) 
  (h2 : p.x + p.y ≤ 4) : 
  ¬∃ q : Point, canReach p q ∧ farFromOrigin q :=
sorry

end NUMINAMATH_CALUDE_kangaroo_cant_reach_far_l2761_276122


namespace NUMINAMATH_CALUDE_cubic_sum_inequality_l2761_276172

theorem cubic_sum_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a^3 + b^3 + a + b ≥ 4*a*b := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_inequality_l2761_276172


namespace NUMINAMATH_CALUDE_correct_arrival_times_l2761_276139

/-- Represents the train journey with given parameters -/
structure TrainJourney where
  totalDistance : ℝ
  uphillDistance1 : ℝ
  flatDistance : ℝ
  uphillDistance2 : ℝ
  speedDifference : ℝ
  stationDistances : List ℝ
  stopTime : ℝ
  departureTime : ℝ
  arrivalTime : ℝ

/-- Calculate arrival times at intermediate stations -/
def calculateArrivalTimes (journey : TrainJourney) : List ℝ :=
  sorry

/-- Main theorem: Arrival times at stations are correct -/
theorem correct_arrival_times (journey : TrainJourney)
  (h1 : journey.totalDistance = 185)
  (h2 : journey.uphillDistance1 = 40)
  (h3 : journey.flatDistance = 105)
  (h4 : journey.uphillDistance2 = 40)
  (h5 : journey.speedDifference = 10)
  (h6 : journey.stationDistances = [20, 70, 100, 161])
  (h7 : journey.stopTime = 3/60)
  (h8 : journey.departureTime = 8)
  (h9 : journey.arrivalTime = 10 + 22/60) :
  calculateArrivalTimes journey = [8 + 15/60, 8 + 53/60, 9 + 21/60, 10 + 34/60] :=
sorry

end NUMINAMATH_CALUDE_correct_arrival_times_l2761_276139


namespace NUMINAMATH_CALUDE_quadratic_roots_equivalence_l2761_276117

theorem quadratic_roots_equivalence (a b c : ℝ) (h : a ≠ 0) :
  (∀ x, a * x^2 - b * x + c = 0 → x > 0) → a * c > 0 ↔
  a * c ≤ 0 → ¬(∀ x, a * x^2 - b * x + c = 0 → x > 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_equivalence_l2761_276117


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2761_276106

theorem smallest_n_congruence :
  ∃ (n : ℕ), n > 0 ∧ (23 * n) % 11 = 5678 % 11 ∧
  ∀ (m : ℕ), m > 0 ∧ (23 * m) % 11 = 5678 % 11 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2761_276106


namespace NUMINAMATH_CALUDE_root_transformation_l2761_276145

theorem root_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 3*r₁^2 + 9 = 0) ∧ 
  (r₂^3 - 3*r₂^2 + 9 = 0) ∧ 
  (r₃^3 - 3*r₃^2 + 9 = 0) →
  ((3*r₁)^3 - 9*(3*r₁)^2 + 243 = 0) ∧
  ((3*r₂)^3 - 9*(3*r₂)^2 + 243 = 0) ∧
  ((3*r₃)^3 - 9*(3*r₃)^2 + 243 = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_transformation_l2761_276145
