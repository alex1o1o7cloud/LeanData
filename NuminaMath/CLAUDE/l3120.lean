import Mathlib

namespace NUMINAMATH_CALUDE_lcm_count_theorem_l3120_312088

theorem lcm_count_theorem : 
  ∃ (S : Finset ℕ), 
    S.card = 19 ∧ 
    (∀ k : ℕ, k > 0 → (Nat.lcm (Nat.lcm (9^9) (12^12)) k = 18^18 ↔ k ∈ S)) := by
  sorry

end NUMINAMATH_CALUDE_lcm_count_theorem_l3120_312088


namespace NUMINAMATH_CALUDE_trays_for_school_staff_l3120_312099

def small_oatmeal_cookies : ℕ := 276
def large_oatmeal_cookies : ℕ := 92
def large_choc_chip_cookies : ℕ := 150
def small_cookies_per_tray : ℕ := 12
def large_cookies_per_tray : ℕ := 6

theorem trays_for_school_staff : 
  (large_choc_chip_cookies + large_cookies_per_tray - 1) / large_cookies_per_tray = 25 := by
  sorry

end NUMINAMATH_CALUDE_trays_for_school_staff_l3120_312099


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3120_312057

def vector_a (t : ℝ) : Fin 2 → ℝ := ![t, 1]
def vector_b : Fin 2 → ℝ := ![2, 4]

def perpendicular (u v : Fin 2 → ℝ) : Prop :=
  (u 0) * (v 0) + (u 1) * (v 1) = 0

theorem perpendicular_vectors (t : ℝ) :
  perpendicular (vector_a t) vector_b → t = -2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3120_312057


namespace NUMINAMATH_CALUDE_pete_has_enough_money_l3120_312037

/-- Represents the amount of money Pete has and owes -/
structure PetesMoney where
  wallet_twenty : Nat -- number of $20 bills in wallet
  wallet_ten : Nat -- number of $10 bills in wallet
  wallet_pounds : Nat -- number of £5 notes in wallet
  pocket_ten : Nat -- number of $10 bills in pocket
  owed : Nat -- amount owed on the bike in dollars

/-- Calculates the total amount of money Pete has in dollars -/
def total_money (m : PetesMoney) : Nat :=
  m.wallet_twenty * 20 + m.wallet_ten * 10 + m.wallet_pounds * 7 + m.pocket_ten * 10

/-- Proves that Pete has enough money to pay off his bike debt -/
theorem pete_has_enough_money (m : PetesMoney) 
  (h1 : m.wallet_twenty = 2)
  (h2 : m.wallet_ten = 1)
  (h3 : m.wallet_pounds = 1)
  (h4 : m.pocket_ten = 4)
  (h5 : m.owed = 90) :
  total_money m ≥ m.owed :=
by sorry

end NUMINAMATH_CALUDE_pete_has_enough_money_l3120_312037


namespace NUMINAMATH_CALUDE_apple_students_l3120_312021

theorem apple_students (bananas apples both one_fruit : ℕ) 
  (h1 : bananas = 8)
  (h2 : one_fruit = 10)
  (h3 : both = 5)
  (h4 : one_fruit = (apples - both) + (bananas - both)) :
  apples = 12 := by
  sorry

end NUMINAMATH_CALUDE_apple_students_l3120_312021


namespace NUMINAMATH_CALUDE_three_card_draw_probability_l3120_312078

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of Aces in a standard deck -/
def NumAces : ℕ := 4

/-- Number of diamonds in a standard deck -/
def NumDiamonds : ℕ := 13

/-- Number of Jacks in a standard deck -/
def NumJacks : ℕ := 4

/-- Probability of drawing an Ace as the first card, a diamond as the second card, 
    and a Jack as the third card from a standard 52-card deck -/
theorem three_card_draw_probability : 
  (NumAces / StandardDeck) * (NumDiamonds / (StandardDeck - 1)) * (NumJacks / (StandardDeck - 2)) = 1 / 650 :=
by sorry

end NUMINAMATH_CALUDE_three_card_draw_probability_l3120_312078


namespace NUMINAMATH_CALUDE_is_rectangle_l3120_312064

/-- Given points A, B, C, and D in a 2D plane, prove that ABCD is a rectangle -/
theorem is_rectangle (A B C D : ℝ × ℝ) : 
  A = (-2, 0) → B = (1, 6) → C = (5, 4) → D = (2, -2) →
  (B.1 - A.1, B.2 - A.2) = (C.1 - D.1, C.2 - D.2) ∧
  (B.1 - A.1) * (D.1 - A.1) + (B.2 - A.2) * (D.2 - A.2) = 0 := by
  sorry

#check is_rectangle

end NUMINAMATH_CALUDE_is_rectangle_l3120_312064


namespace NUMINAMATH_CALUDE_elevator_min_trips_l3120_312033

def masses : List ℕ := [150, 60, 70, 71, 72, 100, 101, 102, 103]
def capacity : ℕ := 200

def is_valid_pair (m1 m2 : ℕ) : Prop := m1 + m2 ≤ capacity

def min_trips : ℕ := 5

theorem elevator_min_trips :
  (∀ (m1 m2 m3 : ℕ), m1 ∈ masses → m2 ∈ masses → m3 ∈ masses → m1 ≠ m2 → m2 ≠ m3 → m1 ≠ m3 → m1 + m2 + m3 > capacity) ∧
  (∃ (pairs : List (ℕ × ℕ)), 
    pairs.length = 4 ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ∈ masses ∧ p.2 ∈ masses ∧ p.1 ≠ p.2 ∧ is_valid_pair p.1 p.2) ∧
    (∀ (m : ℕ), m ∈ masses → m = 150 ∨ (∃ (p : ℕ × ℕ), p ∈ pairs ∧ (m = p.1 ∨ m = p.2)))) →
  min_trips = 5 :=
by sorry

end NUMINAMATH_CALUDE_elevator_min_trips_l3120_312033


namespace NUMINAMATH_CALUDE_sine_cosine_relation_l3120_312016

theorem sine_cosine_relation (θ : Real) (x : Real) 
  (h1 : 0 < θ ∧ θ < Real.pi / 2) 
  (h2 : x > 1) 
  (h3 : Real.cos (θ / 2) = Real.sqrt ((x + 1) / (2 * x))) : 
  Real.sin θ = Real.sqrt (x^2 - 1) / x := by
sorry

end NUMINAMATH_CALUDE_sine_cosine_relation_l3120_312016


namespace NUMINAMATH_CALUDE_scientific_notation_of_ten_million_two_hundred_thousand_l3120_312003

theorem scientific_notation_of_ten_million_two_hundred_thousand :
  ∃ (a : ℝ) (n : ℤ), 10200000 = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 10.2 ∧ n = 7 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_ten_million_two_hundred_thousand_l3120_312003


namespace NUMINAMATH_CALUDE_diophantine_equation_equivalence_l3120_312060

theorem diophantine_equation_equivalence (n k : ℕ) (h : n > k) :
  (∃ (x y z : ℕ+), x^n + y^n = z^k) ↔
  (∃ (x y z : ℕ+), x^n + y^n = z^(n-k)) :=
sorry

end NUMINAMATH_CALUDE_diophantine_equation_equivalence_l3120_312060


namespace NUMINAMATH_CALUDE_remainder_equality_l3120_312039

theorem remainder_equality (P P' Q D R R' s s' : ℕ) : 
  P > P' → 
  Q > 0 → 
  P < D → P' < D → Q < D →
  R = P % D →
  R' = P' % D →
  s = (P + P') % D →
  s' = (R + R') % D →
  s = s' :=
by sorry

end NUMINAMATH_CALUDE_remainder_equality_l3120_312039


namespace NUMINAMATH_CALUDE_slow_clock_theorem_l3120_312077

/-- Represents a clock with a specific overlap time between its minute and hour hands. -/
structure Clock where
  overlap_time : ℕ  -- Time in minutes between each overlap of minute and hour hands

/-- The number of overlaps in a 24-hour period for any clock -/
def num_overlaps : ℕ := 22

/-- Calculates the length of a 24-hour period for a given clock in minutes -/
def period_length (c : Clock) : ℕ :=
  num_overlaps * c.overlap_time

/-- The length of a standard 24-hour period in minutes -/
def standard_period : ℕ := 24 * 60

/-- Theorem stating that a clock with 66-minute overlaps is 12 minutes slower over 24 hours -/
theorem slow_clock_theorem (c : Clock) (h : c.overlap_time = 66) :
  period_length c - standard_period = 12 := by
  sorry


end NUMINAMATH_CALUDE_slow_clock_theorem_l3120_312077


namespace NUMINAMATH_CALUDE_tan_sin_ratio_equals_three_l3120_312020

theorem tan_sin_ratio_equals_three :
  (Real.tan (20 * π / 180) + 4 * Real.sin (20 * π / 180)) / Real.tan (30 * π / 180) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sin_ratio_equals_three_l3120_312020


namespace NUMINAMATH_CALUDE_investment_after_three_years_l3120_312097

def compound_interest (initial_investment : ℝ) (interest_rate : ℝ) (additional_investment : ℝ) (years : ℕ) : ℝ :=
  let rec helper (n : ℕ) (current_amount : ℝ) : ℝ :=
    if n = 0 then
      current_amount
    else
      helper (n - 1) ((current_amount * (1 + interest_rate)) + additional_investment)
  helper years initial_investment

theorem investment_after_three_years :
  let initial_investment : ℝ := 500
  let interest_rate : ℝ := 0.02
  let additional_investment : ℝ := 500
  let years : ℕ := 3
  compound_interest initial_investment interest_rate additional_investment years = 2060.80 := by
  sorry

end NUMINAMATH_CALUDE_investment_after_three_years_l3120_312097


namespace NUMINAMATH_CALUDE_sqrt_5_times_sqrt_6_minus_1_over_sqrt_5_range_l3120_312008

theorem sqrt_5_times_sqrt_6_minus_1_over_sqrt_5_range : 
  4 < Real.sqrt 5 * (Real.sqrt 6 - 1 / Real.sqrt 5) ∧ 
  Real.sqrt 5 * (Real.sqrt 6 - 1 / Real.sqrt 5) < 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_5_times_sqrt_6_minus_1_over_sqrt_5_range_l3120_312008


namespace NUMINAMATH_CALUDE_fourth_grade_students_l3120_312087

/-- The number of students in fourth grade at the end of the year -/
def final_students (initial : ℕ) (left : ℕ) (new : ℕ) : ℕ :=
  initial - left + new

/-- Theorem: Given the initial conditions, the final number of students is 11 -/
theorem fourth_grade_students : final_students 8 5 8 = 11 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l3120_312087


namespace NUMINAMATH_CALUDE_min_value_T_l3120_312070

theorem min_value_T (a b c : ℝ) 
  (h1 : ∀ x : ℝ, 1/a * x^2 + 6*x + c ≥ 0)
  (h2 : a*b > 1)
  (h3 : ∃ x : ℝ, 1/a * x^2 + 6*x + c = 0) :
  1/(2*(a*b - 1)) + a*(b + 2*c)/(a*b - 1) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_T_l3120_312070


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3120_312047

theorem contrapositive_equivalence :
  (∀ a : ℝ, a > 0 → a > 1) ↔ (∀ a : ℝ, a ≤ 1 → a ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3120_312047


namespace NUMINAMATH_CALUDE_count_symmetric_patterns_l3120_312015

/-- A symmetric digital pattern on an 8x8 grid --/
structure SymmetricPattern :=
  (grid : Fin 8 → Fin 8 → Bool)
  (symmetric : ∀ (i j : Fin 8), grid i j = grid (7 - i) j ∧ grid i j = grid i (7 - j) ∧ grid i j = grid j i)
  (not_monochrome : ∃ (i j k l : Fin 8), grid i j ≠ grid k l)

/-- The number of symmetric regions in an 8x8 grid --/
def num_symmetric_regions : Nat := 12

/-- The total number of possible symmetric digital patterns --/
def total_symmetric_patterns : Nat := 2^num_symmetric_regions - 2

theorem count_symmetric_patterns :
  total_symmetric_patterns = 4094 :=
sorry

end NUMINAMATH_CALUDE_count_symmetric_patterns_l3120_312015


namespace NUMINAMATH_CALUDE_max_sum_distance_from_line_max_sum_distance_from_line_tight_l3120_312066

theorem max_sum_distance_from_line (x₁ y₁ x₂ y₂ : ℝ) :
  x₁^2 + y₁^2 = 1 →
  x₂^2 + y₂^2 = 1 →
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 1 →
  |x₁ + y₁ - 1| + |x₂ + y₂ - 1| ≤ 2 + Real.sqrt 6 :=
by sorry

theorem max_sum_distance_from_line_tight :
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    x₁^2 + y₁^2 = 1 ∧
    x₂^2 + y₂^2 = 1 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 1 ∧
    |x₁ + y₁ - 1| + |x₂ + y₂ - 1| = 2 + Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_distance_from_line_max_sum_distance_from_line_tight_l3120_312066


namespace NUMINAMATH_CALUDE_infinitely_many_perfect_squares_in_sequence_l3120_312006

theorem infinitely_many_perfect_squares_in_sequence :
  ∀ k : ℕ, ∃ x y : ℕ+, x > k ∧ y^2 = ⌊x * Real.sqrt 2⌋ := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_perfect_squares_in_sequence_l3120_312006


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3120_312085

/-- For an arithmetic sequence with first term 13 and common difference -2,
    if the sum of the first n terms is 40, then n is either 4 or 10. -/
theorem arithmetic_sequence_sum (n : ℕ) : 
  let a : ℕ → ℤ := λ k => 13 - 2 * (k - 1)
  let S : ℕ → ℤ := λ m => (m * (2 * 13 + (m - 1) * (-2))) / 2
  S n = 40 → n = 4 ∨ n = 10 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3120_312085


namespace NUMINAMATH_CALUDE_max_candies_for_one_student_l3120_312005

theorem max_candies_for_one_student
  (num_students : ℕ)
  (mean_candies : ℕ)
  (h_num_students : num_students = 40)
  (h_mean_candies : mean_candies = 6)
  (h_at_least_one : ∀ student, student ≥ 1) :
  ∃ (max_candies : ℕ), max_candies = 201 ∧
    ∀ (student_candies : ℕ),
      student_candies ≤ max_candies ∧
      (num_students - 1) * 1 + student_candies ≤ num_students * mean_candies :=
by sorry

end NUMINAMATH_CALUDE_max_candies_for_one_student_l3120_312005


namespace NUMINAMATH_CALUDE_namek_clock_overlap_time_l3120_312013

/-- Represents the clock on Namek --/
structure NamekClock where
  minutes_per_hour : ℕ
  hour_hand_rate : ℚ
  minute_hand_rate : ℚ

/-- The time when the hour and minute hands overlap on Namek's clock --/
def overlap_time (clock : NamekClock) : ℚ :=
  360 / (clock.minute_hand_rate - clock.hour_hand_rate)

/-- Theorem stating that the overlap time for Namek's clock is 20/19 hours --/
theorem namek_clock_overlap_time :
  let clock : NamekClock := {
    minutes_per_hour := 100,
    hour_hand_rate := 360 / 20,
    minute_hand_rate := 360 / (100 / 60)
  }
  overlap_time clock = 20 / 19 := by sorry

end NUMINAMATH_CALUDE_namek_clock_overlap_time_l3120_312013


namespace NUMINAMATH_CALUDE_trapezoid_diagonal_inequality_l3120_312053

/-- A trapezoid with non-parallel sides b and d, and diagonals e and f -/
structure Trapezoid where
  b : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  b_pos : 0 < b
  d_pos : 0 < d
  e_pos : 0 < e
  f_pos : 0 < f

/-- The inequality |e - f| > |b - d| holds for a trapezoid -/
theorem trapezoid_diagonal_inequality (t : Trapezoid) : |t.e - t.f| > |t.b - t.d| := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_diagonal_inequality_l3120_312053


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_2012021_5_l3120_312069

/-- The area of a quadrilateral with vertices at (1, 2), (1, 1), (4, 1), and (2009, 2010) -/
def quadrilateral_area : ℝ :=
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (1, 1)
  let C : ℝ × ℝ := (4, 1)
  let D : ℝ × ℝ := (2009, 2010)
  -- Area calculation goes here
  0 -- Placeholder

/-- Theorem stating that the area of the quadrilateral is 2012021.5 square units -/
theorem quadrilateral_area_is_2012021_5 : quadrilateral_area = 2012021.5 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_2012021_5_l3120_312069


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3120_312065

theorem sufficient_not_necessary :
  (∀ x : ℝ, x - 1 > 0 → x^2 - 1 > 0) ∧
  (∃ x : ℝ, x^2 - 1 > 0 ∧ x - 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3120_312065


namespace NUMINAMATH_CALUDE_common_divisors_count_l3120_312048

/-- The number of positive divisors that 9240, 7920, and 8800 have in common -/
theorem common_divisors_count : Nat.card {d : ℕ | d > 0 ∧ d ∣ 9240 ∧ d ∣ 7920 ∧ d ∣ 8800} = 32 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_count_l3120_312048


namespace NUMINAMATH_CALUDE_largest_valid_number_l3120_312043

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c d e f : ℕ),
    n = a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    (c = a + b ∨ c = a - b ∨ c = b - a) ∧
    (d = b + c ∨ d = b - c ∨ d = c - b) ∧
    (e = c + d ∨ e = c - d ∨ e = d - c) ∧
    (f = d + e ∨ f = d - e ∨ f = e - d) ∧
    0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧
    0 ≤ d ∧ d ≤ 9 ∧ 0 ≤ e ∧ e ≤ 9 ∧ 0 ≤ f ∧ f ≤ 9

theorem largest_valid_number :
  ∀ n : ℕ, is_valid_number n → n ≤ 972538 :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_l3120_312043


namespace NUMINAMATH_CALUDE_playground_paint_ratio_l3120_312046

/-- Given a square playground with side length s and diagonal paint lines of width w,
    if one-third of the playground's area is covered in paint,
    then the ratio of s to w is 3/2. -/
theorem playground_paint_ratio (s w : ℝ) (h_positive : s > 0 ∧ w > 0) 
    (h_paint_area : w^2 + (s - w)^2 / 2 = s^2 / 3) : s / w = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_playground_paint_ratio_l3120_312046


namespace NUMINAMATH_CALUDE_eight_digit_increasing_count_M_is_correct_l3120_312028

/-- The number of 8-digit positive integers with strictly increasing digits using only 1 through 8 -/
def M : ℕ := 1

/-- The set of valid digits -/
def validDigits : Finset ℕ := Finset.range 8

theorem eight_digit_increasing_count : 
  (Finset.powerset validDigits).filter (fun s => s.card = 8) = {validDigits} := by sorry

/-- The main theorem stating that M is correct -/
theorem M_is_correct : 
  M = Finset.card ((Finset.powerset validDigits).filter (fun s => s.card = 8)) := by sorry

end NUMINAMATH_CALUDE_eight_digit_increasing_count_M_is_correct_l3120_312028


namespace NUMINAMATH_CALUDE_cobys_speed_l3120_312096

/-- Coby's road trip problem -/
theorem cobys_speed (d_WI d_IN : ℝ) (v_WI : ℝ) (t_total : ℝ) (h1 : d_WI = 640) (h2 : d_IN = 550) (h3 : v_WI = 80) (h4 : t_total = 19) :
  (d_IN / (t_total - d_WI / v_WI)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_cobys_speed_l3120_312096


namespace NUMINAMATH_CALUDE_pencils_left_over_l3120_312094

theorem pencils_left_over (total_pencils : ℕ) (students_class1 : ℕ) (students_class2 : ℕ) 
  (h1 : total_pencils = 210)
  (h2 : students_class1 = 30)
  (h3 : students_class2 = 20) :
  total_pencils - (students_class1 + students_class2) * (total_pencils / (students_class1 + students_class2)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_pencils_left_over_l3120_312094


namespace NUMINAMATH_CALUDE_depth_multiplier_is_fifteen_l3120_312035

/-- The depth of water in feet -/
def water_depth : ℕ := 255

/-- Ron's height in feet -/
def ron_height : ℕ := 13

/-- The difference between Dean's and Ron's heights in feet -/
def height_difference : ℕ := 4

/-- Dean's height in feet -/
def dean_height : ℕ := ron_height + height_difference

/-- The multiplier for Dean's height to find the depth of the water -/
def depth_multiplier : ℕ := water_depth / dean_height

theorem depth_multiplier_is_fifteen :
  depth_multiplier = 15 :=
by sorry

end NUMINAMATH_CALUDE_depth_multiplier_is_fifteen_l3120_312035


namespace NUMINAMATH_CALUDE_chocolate_eating_impossibility_l3120_312000

/-- Proves that it's impossible to eat enough of the remaining chocolates to reach 3/2 of all chocolates eaten --/
theorem chocolate_eating_impossibility (total : ℕ) (initial_percent : ℚ) : 
  total = 10000 →
  initial_percent = 1/5 →
  ¬∃ (remaining_percent : ℚ), 
    0 ≤ remaining_percent ∧ 
    remaining_percent ≤ 1 ∧
    (initial_percent * total + remaining_percent * (total - initial_percent * total) : ℚ) = 3/2 * total := by
  sorry


end NUMINAMATH_CALUDE_chocolate_eating_impossibility_l3120_312000


namespace NUMINAMATH_CALUDE_min_value_theorem_l3120_312040

theorem min_value_theorem (x y : ℝ) 
  (h : Real.exp x + x - 2023 = Real.exp 2023 / (y + 2023) - Real.log (y + 2023)) :
  (∀ x' y' : ℝ, Real.exp x' + x' - 2023 = Real.exp 2023 / (y' + 2023) - Real.log (y' + 2023) → 
    Real.exp x' + y' + 2024 ≥ Real.exp x + y + 2024) →
  Real.exp x + y + 2024 = 2 * Real.sqrt (Real.exp 2023) + 1 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3120_312040


namespace NUMINAMATH_CALUDE_bianca_roses_count_l3120_312090

theorem bianca_roses_count (tulips used extra : ℕ) : 
  tulips = 39 → used = 81 → extra = 7 → used + extra - tulips = 49 := by
  sorry

end NUMINAMATH_CALUDE_bianca_roses_count_l3120_312090


namespace NUMINAMATH_CALUDE_gavin_shirts_count_l3120_312042

theorem gavin_shirts_count (blue_shirts green_shirts : ℕ) :
  blue_shirts = 6 →
  green_shirts = 17 →
  blue_shirts + green_shirts = 23 :=
by sorry

end NUMINAMATH_CALUDE_gavin_shirts_count_l3120_312042


namespace NUMINAMATH_CALUDE_complex_multiplication_l3120_312004

/-- Given two complex numbers z₁ and z₂, prove that their product is equal to the specified result. -/
theorem complex_multiplication (z₁ z₂ : ℂ) : 
  z₁ = 1 - 3*I → z₂ = 6 - 8*I → z₁ * z₂ = -18 - 26*I := by
  sorry


end NUMINAMATH_CALUDE_complex_multiplication_l3120_312004


namespace NUMINAMATH_CALUDE_women_percentage_of_men_l3120_312098

theorem women_percentage_of_men (W M : ℝ) (h : M = 2 * W) : W / M * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_women_percentage_of_men_l3120_312098


namespace NUMINAMATH_CALUDE_batsman_performance_theorem_l3120_312061

/-- Represents a batsman's performance in a cricket tournament -/
structure BatsmanPerformance where
  innings : ℕ
  runsBeforeLastInning : ℕ
  runsInLastInning : ℕ
  averageIncrease : ℚ
  boundariesBeforeLastInning : ℕ
  boundariesInLastInning : ℕ

/-- Calculates the batting average after the last inning -/
def battingAverage (performance : BatsmanPerformance) : ℚ :=
  (performance.runsBeforeLastInning + performance.runsInLastInning) / performance.innings

/-- Calculates the batting efficiency factor -/
def battingEfficiencyFactor (performance : BatsmanPerformance) : ℚ :=
  (performance.boundariesBeforeLastInning + performance.boundariesInLastInning) / performance.innings

theorem batsman_performance_theorem (performance : BatsmanPerformance) 
  (h1 : performance.innings = 17)
  (h2 : performance.runsInLastInning = 84)
  (h3 : performance.averageIncrease = 5/2)
  (h4 : performance.boundariesInLastInning = 12)
  (h5 : performance.boundariesBeforeLastInning + performance.boundariesInLastInning = 72) :
  battingAverage performance = 44 ∧ battingEfficiencyFactor performance = 72/17 := by
  sorry

#eval (72 : ℚ) / 17

end NUMINAMATH_CALUDE_batsman_performance_theorem_l3120_312061


namespace NUMINAMATH_CALUDE_finish_tea_and_coffee_l3120_312044

/-- The number of days it takes for A and B to finish a can of coffee together -/
def coffee_together : ℝ := 10

/-- The number of days it takes for A to finish a can of coffee alone -/
def coffee_A : ℝ := 12

/-- The number of days it takes for A and B to finish a pound of tea together -/
def tea_together : ℝ := 12

/-- The number of days it takes for B to finish a pound of tea alone -/
def tea_B : ℝ := 20

/-- The time it takes for A and B to finish a pound of tea and a can of coffee -/
def total_time : ℝ := 35

theorem finish_tea_and_coffee :
  ∃ (coffee_B tea_A : ℝ),
    coffee_B > 0 ∧ tea_A > 0 ∧
    (1 / coffee_A + 1 / coffee_B) * coffee_together = 1 ∧
    (1 / tea_B + 1 / tea_A) * tea_together = 1 ∧
    tea_A + (1 / coffee_A + 1 / coffee_B) * (total_time - tea_A) = 1 :=
by sorry

end NUMINAMATH_CALUDE_finish_tea_and_coffee_l3120_312044


namespace NUMINAMATH_CALUDE_shekars_english_score_l3120_312026

def math_score : ℕ := 76
def science_score : ℕ := 65
def social_studies_score : ℕ := 82
def biology_score : ℕ := 95
def average_score : ℕ := 77
def total_subjects : ℕ := 5

theorem shekars_english_score :
  let known_scores_sum := math_score + science_score + social_studies_score + biology_score
  let total_score := average_score * total_subjects
  total_score - known_scores_sum = 67 := by
  sorry

end NUMINAMATH_CALUDE_shekars_english_score_l3120_312026


namespace NUMINAMATH_CALUDE_problem_solution_l3120_312083

-- Define the line y = ax - 2a + 4
def line_a (a : ℝ) (x : ℝ) : ℝ := a * x - 2 * a + 4

-- Define the point (2, 4)
def point_2_4 : ℝ × ℝ := (2, 4)

-- Define the line y + 1 = 3x
def line_3x (x : ℝ) : ℝ := 3 * x - 1

-- Define the line x + √3y + 1 = 0
def line_sqrt3 (x y : ℝ) : Prop := x + Real.sqrt 3 * y + 1 = 0

-- Define the point (-2, 3)
def point_neg2_3 : ℝ × ℝ := (-2, 3)

-- Define the line x - 2y + 3 = 0
def line_1 (x y : ℝ) : Prop := x - 2 * y + 3 = 0

-- Define the line 2x + y + 1 = 0
def line_2 (x y : ℝ) : Prop := 2 * x + y + 1 = 0

theorem problem_solution :
  (∀ a : ℝ, line_a a (point_2_4.1) = point_2_4.2) ∧
  (line_2 point_neg2_3.1 point_neg2_3.2 ∧
   ∀ x y : ℝ, line_1 x y → line_2 x y → x = y) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3120_312083


namespace NUMINAMATH_CALUDE_solve_for_b_l3120_312093

-- Define the functions p and q
def p (x : ℝ) := 3 * x + 5
def q (x b : ℝ) := 4 * x - b

-- State the theorem
theorem solve_for_b :
  ∀ b : ℝ, p (q 3 b) = 29 → b = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l3120_312093


namespace NUMINAMATH_CALUDE_right_triangle_segment_ratio_l3120_312089

theorem right_triangle_segment_ratio (a b c r s : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 → s > 0 →
  a^2 + b^2 = c^2 →  -- Right triangle condition
  r * s = a^2 →      -- Geometric mean theorem
  r + s = c →        -- Segments r and s form the hypotenuse
  a = (1/2) * b →    -- Given ratio a:b = 1:2
  r = (1/4) * s :=   -- Conclusion: ratio r:s = 1:4
by sorry

end NUMINAMATH_CALUDE_right_triangle_segment_ratio_l3120_312089


namespace NUMINAMATH_CALUDE_faster_walking_speed_l3120_312062

/-- Proves that given a person who walked 100 km at 10 km/hr, if they had walked at a faster speed
    for the same amount of time and covered an additional 20 km, their faster speed would be 12 km/hr. -/
theorem faster_walking_speed (actual_distance : ℝ) (actual_speed : ℝ) (additional_distance : ℝ) :
  actual_distance = 100 →
  actual_speed = 10 →
  additional_distance = 20 →
  (actual_distance + additional_distance) / (actual_distance / actual_speed) = 12 :=
by sorry

end NUMINAMATH_CALUDE_faster_walking_speed_l3120_312062


namespace NUMINAMATH_CALUDE_power_two_minus_one_div_seven_l3120_312051

theorem power_two_minus_one_div_seven (n : ℕ) :
  (7 ∣ (2^n - 1)) ↔ (3 ∣ n) := by
sorry

end NUMINAMATH_CALUDE_power_two_minus_one_div_seven_l3120_312051


namespace NUMINAMATH_CALUDE_yellow_preference_l3120_312052

theorem yellow_preference (total_students : ℕ) (total_girls : ℕ) 
  (h_total : total_students = 30)
  (h_girls : total_girls = 18)
  (h_green : total_students / 2 = total_students - (total_students / 2))
  (h_pink : total_girls / 3 = total_girls - (2 * (total_girls / 3))) :
  total_students - (total_students / 2 + total_girls / 3) = 9 := by
  sorry

end NUMINAMATH_CALUDE_yellow_preference_l3120_312052


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l3120_312012

theorem complex_sum_theorem (a b : ℂ) : 
  a = 5 - 3*I → b = 2 + 4*I → a + 2*b = 9 + 5*I := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l3120_312012


namespace NUMINAMATH_CALUDE_two_red_more_likely_than_one_four_l3120_312063

/-- The number of red balls in the box -/
def red_balls : ℕ := 4

/-- The number of white balls in the box -/
def white_balls : ℕ := 2

/-- The total number of balls in the box -/
def total_balls : ℕ := red_balls + white_balls

/-- The number of faces on each die -/
def die_faces : ℕ := 6

/-- The probability of drawing two red balls from the box -/
def prob_two_red : ℚ := (red_balls * (red_balls - 1)) / (total_balls * (total_balls - 1))

/-- The probability of rolling at least one 4 with two dice -/
def prob_at_least_one_four : ℚ := 1 - (die_faces - 1)^2 / die_faces^2

/-- Theorem stating that the probability of drawing two red balls is greater than
    the probability of rolling at least one 4 with two dice -/
theorem two_red_more_likely_than_one_four : prob_two_red > prob_at_least_one_four :=
sorry

end NUMINAMATH_CALUDE_two_red_more_likely_than_one_four_l3120_312063


namespace NUMINAMATH_CALUDE_amount_owed_l3120_312075

theorem amount_owed (rate_per_car : ℚ) (cars_washed : ℚ) (h1 : rate_per_car = 9/4) (h2 : cars_washed = 10/3) : 
  rate_per_car * cars_washed = 15/2 := by
sorry

end NUMINAMATH_CALUDE_amount_owed_l3120_312075


namespace NUMINAMATH_CALUDE_triangle_ratio_l3120_312038

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ) :
  A = 2 * π / 3 →  -- 120° in radians
  b = 1 →
  (1 / 2) * c * b * Real.sin A = Real.sqrt 3 →
  (b + c) / (Real.sin B + Real.sin C) = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_l3120_312038


namespace NUMINAMATH_CALUDE_peter_walking_time_l3120_312050

/-- The time required to walk a given distance at a given pace -/
def timeToWalk (distance : ℝ) (pace : ℝ) : ℝ := distance * pace

theorem peter_walking_time :
  let totalDistance : ℝ := 2.5
  let walkingPace : ℝ := 20
  let distanceWalked : ℝ := 1
  let remainingDistance : ℝ := totalDistance - distanceWalked
  timeToWalk remainingDistance walkingPace = 30 := by
sorry

end NUMINAMATH_CALUDE_peter_walking_time_l3120_312050


namespace NUMINAMATH_CALUDE_simplify_expression_l3120_312095

theorem simplify_expression (x t : ℝ) : (x^2 * t^3) * (x^3 * t^4) = x^5 * t^7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3120_312095


namespace NUMINAMATH_CALUDE_cube_congruence_implies_sum_divisibility_l3120_312036

theorem cube_congruence_implies_sum_divisibility (p x y z : ℕ) : 
  Prime p → 
  0 < x → x < y → y < z → z < p → 
  x^3 % p = y^3 % p → y^3 % p = z^3 % p → 
  (x^2 + y^2 + z^2) % (x + y + z) = 0 := by
sorry

end NUMINAMATH_CALUDE_cube_congruence_implies_sum_divisibility_l3120_312036


namespace NUMINAMATH_CALUDE_p_neither_sufficient_nor_necessary_for_q_l3120_312001

/-- Two lines are perpendicular if their slopes multiply to -1 -/
def perpendicular (a b : ℝ) : Prop := a * b = -1

/-- The condition p: ax + y + 1 = 0 is perpendicular to ax - y + 2 = 0 -/
def p (a : ℝ) : Prop := perpendicular a (-a)

/-- The condition q: a = 1 -/
def q : ℝ → Prop := (· = 1)

/-- p is neither sufficient nor necessary for q -/
theorem p_neither_sufficient_nor_necessary_for_q :
  (¬∀ a, p a → q a) ∧ (¬∀ a, q a → p a) := by sorry

end NUMINAMATH_CALUDE_p_neither_sufficient_nor_necessary_for_q_l3120_312001


namespace NUMINAMATH_CALUDE_line_symmetry_about_bisector_l3120_312079

/-- Given two lines l₁ and l₂ with angle bisector y = x, prove that if l₁ has equation ax + by + c = 0 (ab > 0), then l₂ has equation bx + ay + c = 0 -/
theorem line_symmetry_about_bisector (a b c : ℝ) (hab : a * b > 0) :
  let l₁ := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}
  let l₂ := {p : ℝ × ℝ | b * p.1 + a * p.2 + c = 0}
  let bisector := {p : ℝ × ℝ | p.1 = p.2}
  (∀ p : ℝ × ℝ, p ∈ bisector → (p ∈ l₁ ↔ p ∈ l₂)) →
  ∀ q : ℝ × ℝ, q ∈ l₂ := by
  sorry

end NUMINAMATH_CALUDE_line_symmetry_about_bisector_l3120_312079


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3120_312018

theorem polynomial_factorization (k : ℤ) :
  let N : ℕ := (4 * k^4 - 8 * k^2 + 2).toNat
  let p (x : ℝ) := x^8 + N * x^4 + 1
  let f (x : ℝ) := x^4 - 2*k*x^3 + 2*k^2*x^2 - 2*k*x + 1
  let g (x : ℝ) := x^4 + 2*k*x^3 + 2*k^2*x^2 + 2*k*x + 1
  ∀ x, p x = f x * g x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3120_312018


namespace NUMINAMATH_CALUDE_sequence_periodicity_l3120_312022

def M (m : ℕ) : Set ℕ :=
  {x | x ∈ Finset.range m ∨ (x > m ∧ x ≤ 2*m ∧ x % 2 = 1)}

def next_term (m : ℕ) (a : ℕ) : ℕ :=
  if a % 2 = 0 then a / 2 else a + m

def is_periodic (m : ℕ) (a : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, 
    (Nat.iterate (next_term m) k a) = (Nat.iterate (next_term m) n a)

theorem sequence_periodicity (m : ℕ) (h : m > 0) :
  ∀ a : ℕ, is_periodic m a ↔ a ∈ M m :=
sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l3120_312022


namespace NUMINAMATH_CALUDE_symmetric_probability_is_one_over_429_l3120_312084

/-- Represents a coloring of the 13-square array -/
def Coloring := Fin 13 → Bool

/-- The total number of squares in the array -/
def totalSquares : ℕ := 13

/-- The number of red squares -/
def redSquares : ℕ := 8

/-- The number of blue squares -/
def blueSquares : ℕ := 5

/-- Predicate to check if a coloring is symmetric under 90-degree rotation -/
def isSymmetric (c : Coloring) : Prop := sorry

/-- The number of symmetric colorings -/
def symmetricColorings : ℕ := 3

/-- The total number of possible colorings -/
def totalColorings : ℕ := Nat.choose totalSquares blueSquares

/-- The probability of selecting a symmetric coloring -/
def symmetricProbability : ℚ := symmetricColorings / totalColorings

theorem symmetric_probability_is_one_over_429 : 
  symmetricProbability = 1 / 429 := by sorry

end NUMINAMATH_CALUDE_symmetric_probability_is_one_over_429_l3120_312084


namespace NUMINAMATH_CALUDE_expression_value_l3120_312059

theorem expression_value (x y : ℝ) (h : y = 2 - x) : 4 * x + 4 * y - 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3120_312059


namespace NUMINAMATH_CALUDE_binomial_7_choose_4_l3120_312031

theorem binomial_7_choose_4 : Nat.choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_binomial_7_choose_4_l3120_312031


namespace NUMINAMATH_CALUDE_one_eighth_of_number_l3120_312092

theorem one_eighth_of_number (n : ℚ) (h : 6/11 * n = 48) : 1/8 * n = 11 := by
  sorry

end NUMINAMATH_CALUDE_one_eighth_of_number_l3120_312092


namespace NUMINAMATH_CALUDE_painting_perimeter_l3120_312091

/-- A rectangular painting with a frame -/
structure FramedPainting where
  /-- Width of the painting -/
  width : ℝ
  /-- Height of the painting -/
  height : ℝ
  /-- The frame extends 3 cm outward from each side -/
  frame_extension : ℝ := 3
  /-- The area of the frame not covered by the painting is 108 cm² -/
  frame_area : (width + 2 * frame_extension) * (height + 2 * frame_extension) - width * height = 108

/-- The perimeter of the painting is 24 cm -/
theorem painting_perimeter (p : FramedPainting) : 2 * (p.width + p.height) = 24 := by
  sorry

end NUMINAMATH_CALUDE_painting_perimeter_l3120_312091


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_l3120_312073

theorem quadratic_function_minimum (a c : ℝ) (ha : a > 0) (hc : c > 0) (hac : a * c = 1) :
  (a + 1) / c + (c + 1) / a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_minimum_l3120_312073


namespace NUMINAMATH_CALUDE_three_lines_theorem_l3120_312080

/-- Three lines in the plane -/
structure ThreeLines where
  l1 : Real → Real → Prop
  l2 : Real → Real → Prop
  l3 : Real → Real → Real → Prop

/-- The condition that the three lines divide the plane into six parts -/
def divides_into_six_parts (lines : ThreeLines) : Prop := sorry

/-- The main theorem -/
theorem three_lines_theorem (k : Real) :
  let lines : ThreeLines := {
    l1 := λ x y => x - 2*y + 1 = 0,
    l2 := λ x _ => x - 1 = 0,
    l3 := λ x y k => x + k*y = 0
  }
  divides_into_six_parts lines → k ∈ ({0, -1, -2} : Set Real) := by
  sorry

end NUMINAMATH_CALUDE_three_lines_theorem_l3120_312080


namespace NUMINAMATH_CALUDE_sector_central_angle_l3120_312032

/-- Given a sector with radius r and perimeter 3r, its central angle is 1. -/
theorem sector_central_angle (r : ℝ) (h : r > 0) :
  (∃ (l : ℝ), l > 0 ∧ 2 * r + l = 3 * r) →
  (∃ (α : ℝ), α = l / r ∧ α = 1) :=
by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3120_312032


namespace NUMINAMATH_CALUDE_canoe_kayak_difference_is_six_l3120_312056

/-- Represents the rental business scenario --/
structure RentalBusiness where
  canoe_price : ℕ
  kayak_price : ℕ
  canoe_kayak_ratio : ℚ
  total_revenue : ℕ

/-- The specific rental business instance --/
def our_business : RentalBusiness := {
  canoe_price := 9
  kayak_price := 12
  canoe_kayak_ratio := 4/3
  total_revenue := 432
}

/-- Calculates the difference between canoes and kayaks rented --/
def canoe_kayak_difference (rb : RentalBusiness) : ℕ :=
  sorry

/-- Theorem stating that the difference between canoes and kayaks rented is 6 --/
theorem canoe_kayak_difference_is_six :
  canoe_kayak_difference our_business = 6 := by
  sorry

end NUMINAMATH_CALUDE_canoe_kayak_difference_is_six_l3120_312056


namespace NUMINAMATH_CALUDE_sin_cos_from_tan_l3120_312086

theorem sin_cos_from_tan (α : Real) (h : Real.tan α = 3) : 
  Real.sin α * Real.cos α = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_from_tan_l3120_312086


namespace NUMINAMATH_CALUDE_nested_fraction_simplification_l3120_312045

theorem nested_fraction_simplification : 
  1 + (1 / (1 + (1 / (1 + (1 / (1 + 2)))))) = 11 / 7 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_simplification_l3120_312045


namespace NUMINAMATH_CALUDE_chinese_chess_probability_l3120_312007

theorem chinese_chess_probability (p_win p_draw : ℝ) 
  (h_win : p_win = 0.5) 
  (h_draw : p_draw = 0.2) : 
  p_win + p_draw = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_chinese_chess_probability_l3120_312007


namespace NUMINAMATH_CALUDE_cobbler_efficiency_l3120_312024

/-- Represents the cobbler's work schedule and output --/
structure CobblerSchedule where
  hours_per_day : ℕ -- Hours worked per day from Monday to Thursday
  friday_hours : ℕ  -- Hours worked on Friday
  shoes_per_week : ℕ -- Number of shoes mended in a week

/-- Calculates the number of shoes mended per hour --/
def shoes_per_hour (schedule : CobblerSchedule) : ℚ :=
  schedule.shoes_per_week / (4 * schedule.hours_per_day + schedule.friday_hours)

/-- Theorem stating that the cobbler mends 3 shoes per hour --/
theorem cobbler_efficiency (schedule : CobblerSchedule) 
  (h1 : schedule.hours_per_day = 8)
  (h2 : schedule.friday_hours = 3)
  (h3 : schedule.shoes_per_week = 105) :
  shoes_per_hour schedule = 3 := by
  sorry

#eval shoes_per_hour ⟨8, 3, 105⟩

end NUMINAMATH_CALUDE_cobbler_efficiency_l3120_312024


namespace NUMINAMATH_CALUDE_school_trip_classrooms_l3120_312049

theorem school_trip_classrooms 
  (students_per_classroom : ℕ) 
  (seats_per_bus : ℕ) 
  (buses_needed : ℕ) 
  (h1 : students_per_classroom = 66)
  (h2 : seats_per_bus = 6)
  (h3 : buses_needed = 737) :
  (buses_needed * seats_per_bus) / students_per_classroom = 67 := by
  sorry

end NUMINAMATH_CALUDE_school_trip_classrooms_l3120_312049


namespace NUMINAMATH_CALUDE_rope_cutting_problem_l3120_312067

theorem rope_cutting_problem (initial_length : ℝ) : 
  initial_length / 2 / 2 / 5 = 5 → initial_length = 100 := by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_problem_l3120_312067


namespace NUMINAMATH_CALUDE_perimeter_is_twelve_l3120_312041

/-- An equilateral triangle -/
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  base_positive : base > 0
  leg_positive : leg > 0

/-- A quadrilateral formed by cutting a corner from an equilateral triangle -/
def CutCornerQuadrilateral (et : EquilateralTriangle) (it : IsoscelesTriangle) :=
  it.leg < et.side ∧ it.base < et.side

/-- The perimeter of the quadrilateral formed by cutting a corner from an equilateral triangle -/
def perimeter (et : EquilateralTriangle) (it : IsoscelesTriangle) 
    (h : CutCornerQuadrilateral et it) : ℝ :=
  et.side + 2 * (et.side - it.leg) + it.base

/-- The main theorem -/
theorem perimeter_is_twelve 
    (et : EquilateralTriangle)
    (it : IsoscelesTriangle)
    (h : CutCornerQuadrilateral et it)
    (h_et_side : et.side = 4)
    (h_it_leg : it.leg = 0.5)
    (h_it_base : it.base = 1) :
    perimeter et it h = 12 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_is_twelve_l3120_312041


namespace NUMINAMATH_CALUDE_first_day_over_500_l3120_312055

def paperclips (day : ℕ) : ℕ :=
  match day with
  | 0 => 5  -- Monday
  | 1 => 10 -- Tuesday
  | n + 2 => 3 * paperclips (n + 1)

theorem first_day_over_500 :
  (∀ d < 6, paperclips d ≤ 500) ∧ (paperclips 6 > 500) := by
  sorry

end NUMINAMATH_CALUDE_first_day_over_500_l3120_312055


namespace NUMINAMATH_CALUDE_intersection_when_a_is_one_range_of_a_when_union_is_reals_l3120_312074

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | |x - a| < 4}
def B : Set ℝ := {x | x^2 - 4*x - 5 > 0}

-- Part 1
theorem intersection_when_a_is_one :
  A 1 ∩ B = {x | -3 < x ∧ x < -1} := by sorry

-- Part 2
theorem range_of_a_when_union_is_reals :
  (∃ a, A a ∪ B = Set.univ) → ∃ a, 1 < a ∧ a < 3 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_one_range_of_a_when_union_is_reals_l3120_312074


namespace NUMINAMATH_CALUDE_unique_solution_system_l3120_312025

theorem unique_solution_system : 
  ∃! (x y : ℕ+), (x : ℝ)^(y : ℝ) + 3 = (y : ℝ)^(x : ℝ) + 1 ∧ 
                 2 * (x : ℝ)^(y : ℝ) + 4 = (y : ℝ)^(x : ℝ) + 9 ∧
                 x = 3 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3120_312025


namespace NUMINAMATH_CALUDE_parallelogram_area_l3120_312030

/-- The area of a parallelogram with base 20 cm and height 16 cm is 320 cm². -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
  base = 20 → height = 16 → area = base * height → area = 320 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3120_312030


namespace NUMINAMATH_CALUDE_cheapest_lamp_cost_l3120_312029

theorem cheapest_lamp_cost (frank_money : ℕ) (remaining : ℕ) (price_ratio : ℕ) : 
  frank_money = 90 →
  remaining = 30 →
  price_ratio = 3 →
  (frank_money - remaining) / price_ratio = 20 :=
by sorry

end NUMINAMATH_CALUDE_cheapest_lamp_cost_l3120_312029


namespace NUMINAMATH_CALUDE_consecutive_non_primes_l3120_312017

theorem consecutive_non_primes (n : ℕ) (h : n ≥ 1) :
  ∃ (k : ℕ), ∀ (i : ℕ), i ∈ Finset.range n → 
    ¬ Nat.Prime (k + i) ∧ 
    (∀ (j : ℕ), j ∈ Finset.range n → k + i = k + j → i = j) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_non_primes_l3120_312017


namespace NUMINAMATH_CALUDE_land_sections_area_l3120_312023

theorem land_sections_area (x y z : ℝ) 
  (h1 : x = (2/5) * (x + y + z))
  (h2 : y / z = (3/2) / (4/3))
  (h3 : z = x - 16) :
  x + y + z = 136 := by
  sorry

end NUMINAMATH_CALUDE_land_sections_area_l3120_312023


namespace NUMINAMATH_CALUDE_square_areas_and_perimeters_l3120_312058

theorem square_areas_and_perimeters (x : ℝ) : 
  (∃ (s₁ s₂ : ℝ), 
    s₁^2 = x^2 + 12*x + 36 ∧ 
    s₂^2 = 4*x^2 - 12*x + 9 ∧ 
    4*s₁ + 4*s₂ = 64) → 
  x = 13/3 := by
sorry

end NUMINAMATH_CALUDE_square_areas_and_perimeters_l3120_312058


namespace NUMINAMATH_CALUDE_AB_BA_parallel_l3120_312002

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = k • w ∨ w = k • v

/-- Vector AB is defined as the difference between points B and A -/
def vector_AB (A B : ℝ × ℝ) : ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2)

/-- Vector BA is defined as the difference between points A and B -/
def vector_BA (A B : ℝ × ℝ) : ℝ × ℝ :=
  (A.1 - B.1, A.2 - B.2)

/-- Theorem: Vectors AB and BA are parallel -/
theorem AB_BA_parallel (A B : ℝ × ℝ) :
  are_parallel (vector_AB A B) (vector_BA A B) := by
  sorry

end NUMINAMATH_CALUDE_AB_BA_parallel_l3120_312002


namespace NUMINAMATH_CALUDE_inequality_proof_l3120_312076

theorem inequality_proof (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) :
  x + y + (1 / (x * y)) ≤ (1 / x) + (1 / y) + x * y := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3120_312076


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l3120_312081

theorem sqrt_equation_solutions :
  {x : ℝ | Real.sqrt (4 * x - 3) + 10 / Real.sqrt (4 * x - 3) = 7} = {7/4, 7} := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l3120_312081


namespace NUMINAMATH_CALUDE_find_x_and_y_l3120_312082

theorem find_x_and_y :
  ∀ x y : ℝ,
  x > y →
  x + y = 55 →
  x - y = 15 →
  x = 35 ∧ y = 20 := by
sorry

end NUMINAMATH_CALUDE_find_x_and_y_l3120_312082


namespace NUMINAMATH_CALUDE_power_function_even_l3120_312019

theorem power_function_even (α : ℤ) (h1 : 0 ≤ α) (h2 : α ≤ 5) :
  (∀ x : ℝ, (fun x => x^(3 - α)) (-x) = (fun x => x^(3 - α)) x) → α = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_function_even_l3120_312019


namespace NUMINAMATH_CALUDE_intersection_distance_product_l3120_312054

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := y^2 = 4*x

def C₂ (x y : ℝ) : Prop := Real.sqrt 3 * x - y - 2 * Real.sqrt 3 = 0

-- Define point P
def P : ℝ × ℝ := (2, 0)

-- Define the theorem
theorem intersection_distance_product :
  ∃ (A B : ℝ × ℝ),
    C₁ A.1 A.2 ∧ C₂ A.1 A.2 ∧
    C₁ B.1 B.2 ∧ C₂ B.1 B.2 ∧
    A ≠ B ∧
    (Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) *
     Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) = 32/3) :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_product_l3120_312054


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l3120_312072

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 3 * x + 2 * y = 6) 
  (eq2 : 2 * x + 3 * y = 7) : 
  14 * x^2 + 25 * x * y + 14 * y^2 = 85 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l3120_312072


namespace NUMINAMATH_CALUDE_moles_of_Na2SO4_formed_l3120_312014

-- Define the reactants and products
structure Compound where
  name : String
  coefficient : ℚ

-- Define the reaction
def reaction : List Compound → List Compound → Prop :=
  λ reactants products => reactants.length = 2 ∧ products.length = 2

-- Define the balanced equation
def balancedEquation : Prop :=
  reaction
    [⟨"H2SO4", 1⟩, ⟨"NaOH", 2⟩]
    [⟨"Na2SO4", 1⟩, ⟨"H2O", 2⟩]

-- Define the given amounts of reactants
def givenReactants : List Compound :=
  [⟨"H2SO4", 1⟩, ⟨"NaOH", 2⟩]

-- Theorem to prove
theorem moles_of_Na2SO4_formed
  (h1 : balancedEquation)
  (h2 : givenReactants = [⟨"H2SO4", 1⟩, ⟨"NaOH", 2⟩]) :
  ∃ (product : Compound),
    product.name = "Na2SO4" ∧ product.coefficient = 1 :=
  sorry

end NUMINAMATH_CALUDE_moles_of_Na2SO4_formed_l3120_312014


namespace NUMINAMATH_CALUDE_diana_hourly_wage_l3120_312011

/-- Represents Diana's work schedule and earnings --/
structure DianaWork where
  monday_hours : ℕ
  tuesday_hours : ℕ
  wednesday_hours : ℕ
  thursday_hours : ℕ
  friday_hours : ℕ
  weekly_earnings : ℕ

/-- Calculates Diana's hourly wage based on her work schedule and weekly earnings --/
def hourly_wage (d : DianaWork) : ℚ :=
  d.weekly_earnings / (d.monday_hours + d.tuesday_hours + d.wednesday_hours + d.thursday_hours + d.friday_hours)

/-- Theorem stating that Diana's hourly wage is $30 --/
theorem diana_hourly_wage :
  let d : DianaWork := {
    monday_hours := 10,
    tuesday_hours := 15,
    wednesday_hours := 10,
    thursday_hours := 15,
    friday_hours := 10,
    weekly_earnings := 1800
  }
  hourly_wage d = 30 := by sorry

end NUMINAMATH_CALUDE_diana_hourly_wage_l3120_312011


namespace NUMINAMATH_CALUDE_determinant_scaling_l3120_312068

theorem determinant_scaling (a b c d : ℝ) :
  Matrix.det ![![a, b], ![c, d]] = 5 →
  Matrix.det ![![2 * a, 2 * b], ![2 * c, 2 * d]] = 20 := by
  sorry

end NUMINAMATH_CALUDE_determinant_scaling_l3120_312068


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l3120_312009

theorem sum_of_reciprocals (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : 1/x - 1/y = -2) : 
  x + y = 4/3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l3120_312009


namespace NUMINAMATH_CALUDE_smallest_number_proof_l3120_312071

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_number_proof :
  ∃! x : ℕ, x > 0 ∧ 
    (∃ y : ℕ, y > 0 ∧ 
      x + y = 4728 ∧ 
      is_divisible_by (x + y) 27 ∧
      is_divisible_by (x + y) 35 ∧
      is_divisible_by (x + y) 25 ∧
      is_divisible_by (x + y) 21) ∧
    (∀ z : ℕ, z > 0 ∧ 
      (∃ w : ℕ, w > 0 ∧ 
        z + w = 4728 ∧ 
        is_divisible_by (z + w) 27 ∧
        is_divisible_by (z + w) 35 ∧
        is_divisible_by (z + w) 25 ∧
        is_divisible_by (z + w) 21) → 
      x ≤ z) ∧
  x = 4725 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l3120_312071


namespace NUMINAMATH_CALUDE_one_real_root_condition_l3120_312010

/-- Given the equation lg(kx) = 2lg(x+1), this theorem states the condition for k
    such that the equation has only one real root. -/
theorem one_real_root_condition (k : ℝ) : 
  (∃! x : ℝ, Real.log (k * x) = 2 * Real.log (x + 1)) ↔ (k < 0 ∨ k = 4) :=
by sorry

end NUMINAMATH_CALUDE_one_real_root_condition_l3120_312010


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3120_312027

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (x ≥ 2 ∧ y ≥ 2 → x + y ≥ 4) ∧
  (∃ x y : ℝ, x + y ≥ 4 ∧ (x < 2 ∨ y < 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3120_312027


namespace NUMINAMATH_CALUDE_bianca_carrots_l3120_312034

/-- Proves that Bianca threw out 10 carrots given the initial conditions -/
theorem bianca_carrots (initial : ℕ) (next_day : ℕ) (total : ℕ) (thrown_out : ℕ) 
  (h1 : initial = 23)
  (h2 : next_day = 47)
  (h3 : total = 60)
  (h4 : initial - thrown_out + next_day = total) : 
  thrown_out = 10 := by
  sorry

end NUMINAMATH_CALUDE_bianca_carrots_l3120_312034
