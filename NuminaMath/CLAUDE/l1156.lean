import Mathlib

namespace NUMINAMATH_CALUDE_rebecca_camping_items_l1156_115692

/-- The number of items Rebecca bought for her camping trip -/
def total_items (tent_stakes drink_mix water : ℕ) : ℕ :=
  tent_stakes + drink_mix + water

/-- Theorem stating the total number of items Rebecca bought -/
theorem rebecca_camping_items : ∃ (tent_stakes drink_mix water : ℕ),
  tent_stakes = 4 ∧
  drink_mix = 3 * tent_stakes ∧
  water = tent_stakes + 2 ∧
  total_items tent_stakes drink_mix water = 22 := by
  sorry

end NUMINAMATH_CALUDE_rebecca_camping_items_l1156_115692


namespace NUMINAMATH_CALUDE_smaller_mold_radius_prove_smaller_mold_radius_l1156_115673

/-- The radius of smaller hemisphere-shaped molds when jelly from a larger hemisphere
    is evenly distributed -/
theorem smaller_mold_radius (large_radius : ℝ) (num_small_molds : ℕ) : ℝ :=
  let large_volume := (2 / 3) * Real.pi * large_radius ^ 3
  let small_radius := (large_volume / (num_small_molds * ((2 / 3) * Real.pi))) ^ (1 / 3)
  small_radius

/-- Prove that the radius of each smaller mold is 1 / (2^(2/3)) feet -/
theorem prove_smaller_mold_radius :
  smaller_mold_radius 2 64 = 1 / (2 ^ (2 / 3)) := by
  sorry

end NUMINAMATH_CALUDE_smaller_mold_radius_prove_smaller_mold_radius_l1156_115673


namespace NUMINAMATH_CALUDE_pizza_combinations_l1156_115601

def num_toppings : ℕ := 8

theorem pizza_combinations (n : ℕ) (h : n = num_toppings) : 
  (n.choose 1) + (n.choose 2) + (n.choose 3) = 92 := by
  sorry

#eval num_toppings.choose 1 + num_toppings.choose 2 + num_toppings.choose 3

end NUMINAMATH_CALUDE_pizza_combinations_l1156_115601


namespace NUMINAMATH_CALUDE_probability_intersecting_diagonals_l1156_115672

/-- A regular decagon -/
structure RegularDecagon where
  -- Add any necessary properties

/-- Represents a diagonal in a regular decagon -/
structure Diagonal where
  -- Add any necessary properties

/-- The set of all diagonals in a regular decagon -/
def allDiagonals (d : RegularDecagon) : Set Diagonal :=
  sorry

/-- Predicate to check if two diagonals intersect inside the decagon -/
def intersectInside (d : RegularDecagon) (d1 d2 : Diagonal) : Prop :=
  sorry

/-- The number of ways to choose 3 diagonals from all diagonals -/
def numWaysChoose3Diagonals (d : RegularDecagon) : ℕ :=
  sorry

/-- The number of ways to choose 3 diagonals where at least two intersect -/
def numWaysChoose3IntersectingDiagonals (d : RegularDecagon) : ℕ :=
  sorry

theorem probability_intersecting_diagonals (d : RegularDecagon) :
    (numWaysChoose3IntersectingDiagonals d : ℚ) / (numWaysChoose3Diagonals d : ℚ) = 252 / 1309 := by
  sorry

end NUMINAMATH_CALUDE_probability_intersecting_diagonals_l1156_115672


namespace NUMINAMATH_CALUDE_difference_of_hypotenuse_numbers_l1156_115687

/-- A hypotenuse number is a natural number that can be represented as the sum of two squares of non-negative integers. -/
def is_hypotenuse (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 + b^2

/-- Any natural number greater than 10 can be represented as the difference of two hypotenuse numbers. -/
theorem difference_of_hypotenuse_numbers (n : ℕ) (h : n > 10) :
  ∃ m₁ m₂ : ℕ, is_hypotenuse m₁ ∧ is_hypotenuse m₂ ∧ n = m₁ - m₂ :=
sorry

end NUMINAMATH_CALUDE_difference_of_hypotenuse_numbers_l1156_115687


namespace NUMINAMATH_CALUDE_sqrt_8_to_6th_power_l1156_115626

theorem sqrt_8_to_6th_power : (Real.sqrt 8) ^ 6 = 512 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_8_to_6th_power_l1156_115626


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1156_115682

theorem right_triangle_hypotenuse (a b c : ℝ) :
  a = 36 → b = 48 → c^2 = a^2 + b^2 → c = 60 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1156_115682


namespace NUMINAMATH_CALUDE_correct_distribution_l1156_115645

/-- Represents the amount of coins each person receives -/
structure CoinDistribution where
  a : ℚ
  b : ℚ
  c : ℚ
  d : ℚ
  e : ℚ

/-- Checks if the distribution satisfies the problem conditions -/
def isValidDistribution (dist : CoinDistribution) : Prop :=
  -- The total amount is 5 coins
  dist.a + dist.b + dist.c + dist.d + dist.e = 5 ∧
  -- The difference between each person is equal
  (dist.b - dist.a = dist.c - dist.b) ∧
  (dist.c - dist.b = dist.d - dist.c) ∧
  (dist.d - dist.c = dist.e - dist.d) ∧
  -- The total amount received by A and B equals that received by C, D, and E
  dist.a + dist.b = dist.c + dist.d + dist.e

/-- The theorem stating the correct distribution -/
theorem correct_distribution :
  ∃ (dist : CoinDistribution),
    isValidDistribution dist ∧
    dist.a = 2/3 ∧
    dist.b = 5/6 ∧
    dist.c = 1 ∧
    dist.d = 7/6 ∧
    dist.e = 4/3 :=
  sorry

end NUMINAMATH_CALUDE_correct_distribution_l1156_115645


namespace NUMINAMATH_CALUDE_remainder_9_pow_2048_mod_50_l1156_115651

theorem remainder_9_pow_2048_mod_50 : 9^2048 % 50 = 21 := by
  sorry

end NUMINAMATH_CALUDE_remainder_9_pow_2048_mod_50_l1156_115651


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l1156_115686

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x < 2*x - 7 → x ≥ 8 ∧ 8 < 2*8 - 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l1156_115686


namespace NUMINAMATH_CALUDE_average_age_of_four_students_l1156_115620

theorem average_age_of_four_students
  (total_students : Nat)
  (average_age_all : ℝ)
  (num_group1 : Nat)
  (average_age_group1 : ℝ)
  (num_group2 : Nat)
  (age_last_student : ℝ)
  (h1 : total_students = 15)
  (h2 : average_age_all = 15)
  (h3 : num_group1 = 10)
  (h4 : average_age_group1 = 16)
  (h5 : num_group2 = 4)
  (h6 : age_last_student = 9)
  (h7 : total_students = num_group1 + num_group2 + 1) :
  (total_students * average_age_all - num_group1 * average_age_group1 - age_last_student) / num_group2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_four_students_l1156_115620


namespace NUMINAMATH_CALUDE_power_of_three_mod_five_l1156_115617

theorem power_of_three_mod_five : 3^2040 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_five_l1156_115617


namespace NUMINAMATH_CALUDE_shoe_pairs_problem_l1156_115640

theorem shoe_pairs_problem (n : ℕ) (h : n > 0) :
  (1 : ℚ) / (2 * n - 1 : ℚ) = 1 / 5 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_shoe_pairs_problem_l1156_115640


namespace NUMINAMATH_CALUDE_factorization_theorem_l1156_115648

theorem factorization_theorem (m n : ℝ) : 2 * m^3 * n - 32 * m * n = 2 * m * n * (m + 4) * (m - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_theorem_l1156_115648


namespace NUMINAMATH_CALUDE_smallest_divisible_by_2000_l1156_115618

def sequence_a (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → (n - 1 : ℤ) * a (n + 1) = (n + 1 : ℤ) * a n - 2 * (n - 1 : ℤ)

theorem smallest_divisible_by_2000 (a : ℕ → ℤ) (h : sequence_a a) (h2000 : 2000 ∣ a 1999) :
  (∃ n : ℕ, n ≥ 2 ∧ 2000 ∣ a n) ∧ (∀ m : ℕ, m ≥ 2 ∧ m < 249 → ¬(2000 ∣ a m)) ∧ 2000 ∣ a 249 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_2000_l1156_115618


namespace NUMINAMATH_CALUDE_log_inequality_may_not_hold_l1156_115654

theorem log_inequality_may_not_hold (a b : ℝ) (h : 1/a < 1/b ∧ 1/b < 0) :
  ¬ (∀ a b : ℝ, 1/a < 1/b ∧ 1/b < 0 → Real.log (-a) / Real.log (-b) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_may_not_hold_l1156_115654


namespace NUMINAMATH_CALUDE_trisha_works_52_weeks_l1156_115668

/-- Calculates the number of weeks worked in a year based on given parameters -/
def weeks_worked (hourly_rate : ℚ) (hours_per_week : ℚ) (withholding_rate : ℚ) (annual_take_home : ℚ) : ℚ :=
  annual_take_home / ((hourly_rate * hours_per_week) * (1 - withholding_rate))

/-- Proves that given the specified parameters, Trisha works 52 weeks in a year -/
theorem trisha_works_52_weeks :
  weeks_worked 15 40 (1/5) 24960 = 52 := by
  sorry

end NUMINAMATH_CALUDE_trisha_works_52_weeks_l1156_115668


namespace NUMINAMATH_CALUDE_remaining_water_l1156_115696

-- Define the initial amount of water
def initial_water : ℚ := 3

-- Define the first usage
def first_usage : ℚ := 5/4

-- Define the second usage
def second_usage : ℚ := 1/3

-- Theorem to prove
theorem remaining_water :
  initial_water - first_usage - second_usage = 17/12 := by
  sorry

end NUMINAMATH_CALUDE_remaining_water_l1156_115696


namespace NUMINAMATH_CALUDE_schooner_journey_l1156_115676

/-- Schooner journey problem -/
theorem schooner_journey 
  (c α β : ℝ) 
  (hc : c > 0) 
  (hα : α > 0) 
  (hβ : β > 0) 
  (h_time : ∃ (V : ℝ), V > 0 ∧ 
    α = (3*β/4) + (β*(c+V))/(4*(c-V)) ∧ 
    β = (3*β/4) + ((c+V)*β)/(12*c)) :
  ∃ (AB BC : ℝ),
    AB = (3*β*c)/4 ∧
    BC = (β*c*(4*α - 3*β))/(4*(2*α - β)) ∧
    AB > 0 ∧ BC > 0 := by
  sorry


end NUMINAMATH_CALUDE_schooner_journey_l1156_115676


namespace NUMINAMATH_CALUDE_lcm_24_30_40_l1156_115690

theorem lcm_24_30_40 : Nat.lcm (Nat.lcm 24 30) 40 = 120 := by
  sorry

end NUMINAMATH_CALUDE_lcm_24_30_40_l1156_115690


namespace NUMINAMATH_CALUDE_slide_count_l1156_115694

theorem slide_count (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  initial = 22 → additional = 13 → total = initial + additional → total = 35 := by
  sorry

end NUMINAMATH_CALUDE_slide_count_l1156_115694


namespace NUMINAMATH_CALUDE_constant_value_l1156_115697

def f (x : ℝ) : ℝ := 3 * x - 5

theorem constant_value : ∃ c : ℝ, 2 * f 3 - c = f (3 - 2) ∧ c = 10 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_l1156_115697


namespace NUMINAMATH_CALUDE_work_completion_l1156_115660

/-- Given a piece of work that requires 400 man-days to complete,
    prove that if it takes 26.666666666666668 days for a group of men to complete,
    then the number of men in that group is 15. -/
theorem work_completion (total_man_days : ℝ) (days_to_complete : ℝ) (num_men : ℝ) :
  total_man_days = 400 →
  days_to_complete = 26.666666666666668 →
  num_men * days_to_complete = total_man_days →
  num_men = 15 := by
sorry

end NUMINAMATH_CALUDE_work_completion_l1156_115660


namespace NUMINAMATH_CALUDE_incorrect_step_l1156_115693

theorem incorrect_step (a b : ℝ) (h : a < b) : ¬(2 * (a - b)^2 < (a - b)^2) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_step_l1156_115693


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l1156_115613

theorem binomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x, (x + 2)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  (a₁ + 3*a₃ + 5*a₅ + 7*a₇ + 9*a₉)^2 - (2*a₂ + 4*a₄ + 6*a₆ + 8*a₈)^2 = 3^12 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l1156_115613


namespace NUMINAMATH_CALUDE_total_fruits_l1156_115622

def persimmons : ℕ := 2
def apples : ℕ := 7

theorem total_fruits : persimmons + apples = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_l1156_115622


namespace NUMINAMATH_CALUDE_recurrence_sequence_properties_l1156_115600

/-- A sequence that satisfies the given recurrence relation -/
def RecurrenceSequence (x : ℕ → ℝ) (a : ℝ) : Prop :=
  ∀ n, x (n + 2) = 3 * x (n + 1) - 2 * x n + a

/-- An arithmetic progression -/
def ArithmeticProgression (x : ℕ → ℝ) (b c : ℝ) : Prop :=
  ∀ n, x n = b + (c - b) * (n - 1)

/-- A geometric progression -/
def GeometricProgression (x : ℕ → ℝ) (b q : ℝ) : Prop :=
  ∀ n, x n = b * q^(n - 1)

theorem recurrence_sequence_properties
  (x : ℕ → ℝ) (a b c : ℝ) (h : a < 0) :
  (RecurrenceSequence x a ∧ ArithmeticProgression x b c) →
    (a = c - b ∧ c < b) ∧
  (RecurrenceSequence x a ∧ GeometricProgression x b 2) →
    (a = 0 ∧ c = 2*b ∧ b > 0) :=
by sorry

end NUMINAMATH_CALUDE_recurrence_sequence_properties_l1156_115600


namespace NUMINAMATH_CALUDE_lizette_quiz_average_l1156_115604

theorem lizette_quiz_average (q1 q2 : ℝ) : 
  (q1 + q2 + 92) / 3 = 94 → (q1 + q2) / 2 = 95 := by
  sorry

end NUMINAMATH_CALUDE_lizette_quiz_average_l1156_115604


namespace NUMINAMATH_CALUDE_answer_key_combinations_l1156_115614

/-- Represents the number of answer choices for a multiple-choice question -/
def multipleChoiceOptions : ℕ := 4

/-- Represents the number of true-false questions -/
def trueFalseQuestions : ℕ := 3

/-- Represents the number of multiple-choice questions -/
def multipleChoiceQuestions : ℕ := 2

/-- Calculates the number of valid true-false combinations -/
def validTrueFalseCombinations : ℕ := 2^trueFalseQuestions - 2

/-- Calculates the number of multiple-choice combinations -/
def multipleChoiceCombinations : ℕ := multipleChoiceOptions^multipleChoiceQuestions

/-- Theorem stating the total number of ways to create the answer key -/
theorem answer_key_combinations :
  validTrueFalseCombinations * multipleChoiceCombinations = 96 := by
  sorry

end NUMINAMATH_CALUDE_answer_key_combinations_l1156_115614


namespace NUMINAMATH_CALUDE_lucas_pet_capacity_l1156_115679

/-- The number of pets Lucas can accommodate given his pet bed situation -/
def pets_accommodated (initial_beds : ℕ) (additional_beds : ℕ) (beds_per_pet : ℕ) : ℕ :=
  (initial_beds + additional_beds) / beds_per_pet

theorem lucas_pet_capacity : pets_accommodated 12 8 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_lucas_pet_capacity_l1156_115679


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1156_115680

/-- Given three rectangles with the following properties:
    Rectangle 1: length = 16 cm, width = 8 cm
    Rectangle 2: length = 1/2 of Rectangle 1's length, width = 1/2 of Rectangle 1's width
    Rectangle 3: length = 1/2 of Rectangle 2's length, width = 1/2 of Rectangle 2's width
    The perimeter of the figure formed by these rectangles is 60 cm. -/
theorem rectangle_perimeter (rect1_length rect1_width : ℝ) 
  (rect2_length rect2_width : ℝ) (rect3_length rect3_width : ℝ) :
  rect1_length = 16 ∧ 
  rect1_width = 8 ∧
  rect2_length = rect1_length / 2 ∧
  rect2_width = rect1_width / 2 ∧
  rect3_length = rect2_length / 2 ∧
  rect3_width = rect2_width / 2 →
  2 * (rect1_length + rect1_width + rect2_width + rect3_width) = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1156_115680


namespace NUMINAMATH_CALUDE_problem_statement_l1156_115606

theorem problem_statement (m : ℝ) : 
  let U : Set ℝ := Set.univ
  let A : Set ℝ := {x | x^2 + 3*x + 2 = 0}
  let B : Set ℝ := {x | x^2 + (m+1)*x + m = 0}
  (Set.compl A ∩ B = ∅) → (m = 1 ∨ m = 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1156_115606


namespace NUMINAMATH_CALUDE_min_pumps_needed_l1156_115667

/-- Represents the water pumping scenario -/
structure WaterPumping where
  x : ℝ  -- Amount of water already gushed out before pumping
  a : ℝ  -- Amount of water gushing out per minute
  b : ℝ  -- Amount of water each pump can pump out per minute

/-- The conditions of the water pumping problem -/
def water_pumping_conditions (w : WaterPumping) : Prop :=
  w.x + 40 * w.a = 2 * 40 * w.b ∧
  w.x + 16 * w.a = 4 * 16 * w.b ∧
  w.a > 0 ∧ w.b > 0

/-- The theorem stating the minimum number of pumps needed -/
theorem min_pumps_needed (w : WaterPumping) 
  (h : water_pumping_conditions w) : 
  ∀ n : ℕ, (w.x + 10 * w.a ≤ 10 * n * w.b) → n ≥ 6 := by
  sorry

#check min_pumps_needed

end NUMINAMATH_CALUDE_min_pumps_needed_l1156_115667


namespace NUMINAMATH_CALUDE_right_triangle_angles_l1156_115615

/-- A right-angled triangle with a specific property -/
structure RightTriangle where
  /-- The measure of the right angle in degrees -/
  right_angle : ℝ
  /-- The measure of the angle between the angle bisector of the right angle and the median to the hypotenuse, in degrees -/
  bisector_median_angle : ℝ
  /-- The right angle is 90 degrees -/
  right_angle_is_90 : right_angle = 90
  /-- The angle between the bisector and median is 16 degrees -/
  bisector_median_angle_is_16 : bisector_median_angle = 16

/-- The angles of the triangle given the specific conditions -/
def triangle_angles (t : RightTriangle) : (ℝ × ℝ × ℝ) :=
  (61, 29, 90)

/-- Theorem stating that the angles of the triangle are 61°, 29°, and 90° given the conditions -/
theorem right_triangle_angles (t : RightTriangle) :
  triangle_angles t = (61, 29, 90) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angles_l1156_115615


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_four_and_six_l1156_115629

/-- Represents a repeating decimal with a single digit repetend -/
def RepeatingDecimal (n : ℕ) : ℚ := n / 9

theorem sum_of_repeating_decimals_four_and_six :
  RepeatingDecimal 4 + RepeatingDecimal 6 = 10 / 9 := by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_four_and_six_l1156_115629


namespace NUMINAMATH_CALUDE_prop_a_neither_sufficient_nor_necessary_l1156_115653

-- Define propositions A and B
def PropA (a b : ℝ) : Prop := a + b ≠ 4
def PropB (a b : ℝ) : Prop := a ≠ 1 ∧ b ≠ 3

-- Theorem stating that Prop A is neither sufficient nor necessary for Prop B
theorem prop_a_neither_sufficient_nor_necessary :
  (∃ a b : ℝ, PropA a b ∧ ¬PropB a b) ∧
  (∃ a b : ℝ, PropB a b ∧ ¬PropA a b) :=
sorry

end NUMINAMATH_CALUDE_prop_a_neither_sufficient_nor_necessary_l1156_115653


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1156_115632

/-- 
Given a quadratic equation 2kx^2 + (8k+1)x + 8k = 0 with real coefficient k,
the equation has two distinct real roots if and only if k > -1/16 and k ≠ 0.
-/
theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 2 * k * x₁^2 + (8 * k + 1) * x₁ + 8 * k = 0 ∧
                          2 * k * x₂^2 + (8 * k + 1) * x₂ + 8 * k = 0) ↔
  (k > -1/16 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1156_115632


namespace NUMINAMATH_CALUDE_max_sections_five_lines_l1156_115627

/-- The maximum number of sections a rectangle can be divided into by n line segments -/
def maxSections (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | m + 1 => maxSections m + m + 1

/-- Theorem stating that 5 line segments can divide a rectangle into at most 16 sections -/
theorem max_sections_five_lines :
  maxSections 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_sections_five_lines_l1156_115627


namespace NUMINAMATH_CALUDE_range_of_a_l1156_115661

theorem range_of_a : ∀ a : ℝ, (∀ x : ℝ, x > 1 → x > a) ∧ (∃ x : ℝ, x > a ∧ x ≤ 1) ↔ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1156_115661


namespace NUMINAMATH_CALUDE_cos_sin_identity_l1156_115684

theorem cos_sin_identity :
  Real.cos (40 * π / 180) * Real.cos (160 * π / 180) + Real.sin (40 * π / 180) * Real.sin (20 * π / 180) = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_cos_sin_identity_l1156_115684


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1156_115642

theorem sqrt_inequality (x : ℝ) (h : x ≥ 4) : Real.sqrt (x - 3) + Real.sqrt (x - 2) > Real.sqrt (x - 4) + x - 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1156_115642


namespace NUMINAMATH_CALUDE_combined_capacity_after_transfer_l1156_115664

/-- Represents the capacity and fill level of a drum --/
structure Drum where
  capacity : ℝ
  fillLevel : ℝ

/-- Theorem stating the combined capacity of three drums --/
theorem combined_capacity_after_transfer
  (drumX : Drum)
  (drumY : Drum)
  (drumZ : Drum)
  (hX : drumX.capacity = A ∧ drumX.fillLevel = 1/2)
  (hY : drumY.capacity = 2*A ∧ drumY.fillLevel = 1/5)
  (hZ : drumZ.capacity = B ∧ drumZ.fillLevel = 1/4)
  : drumX.capacity + drumY.capacity + drumZ.capacity = 3*A + B :=
by
  sorry

#check combined_capacity_after_transfer

end NUMINAMATH_CALUDE_combined_capacity_after_transfer_l1156_115664


namespace NUMINAMATH_CALUDE_hedge_cost_and_quantity_l1156_115605

/-- Represents the cost and quantity of concrete blocks for a hedge --/
structure HedgeBlocks where
  cost_a : ℕ  -- Cost of Type A blocks
  cost_b : ℕ  -- Cost of Type B blocks
  cost_c : ℕ  -- Cost of Type C blocks
  qty_a : ℕ   -- Quantity of Type A blocks per section
  qty_b : ℕ   -- Quantity of Type B blocks per section
  qty_c : ℕ   -- Quantity of Type C blocks per section
  sections : ℕ -- Number of sections in the hedge

/-- Calculates the total cost and quantity of blocks for the entire hedge --/
def hedge_totals (h : HedgeBlocks) : ℕ × ℕ × ℕ × ℕ :=
  let total_cost := h.sections * (h.cost_a * h.qty_a + h.cost_b * h.qty_b + h.cost_c * h.qty_c)
  let total_a := h.sections * h.qty_a
  let total_b := h.sections * h.qty_b
  let total_c := h.sections * h.qty_c
  (total_cost, total_a, total_b, total_c)

theorem hedge_cost_and_quantity (h : HedgeBlocks) 
  (h_cost_a : h.cost_a = 2)
  (h_cost_b : h.cost_b = 3)
  (h_cost_c : h.cost_c = 4)
  (h_qty_a : h.qty_a = 20)
  (h_qty_b : h.qty_b = 10)
  (h_qty_c : h.qty_c = 5)
  (h_sections : h.sections = 8) :
  hedge_totals h = (720, 160, 80, 40) := by
  sorry

end NUMINAMATH_CALUDE_hedge_cost_and_quantity_l1156_115605


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l1156_115689

theorem quadratic_roots_ratio (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 8 = 0 → x₂^2 - 2*x₂ - 8 = 0 → (x₁ + x₂) / (x₁ * x₂) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l1156_115689


namespace NUMINAMATH_CALUDE_population_change_l1156_115634

theorem population_change (x : ℝ) : 
  let initial_population : ℝ := 10000
  let first_year_population : ℝ := initial_population * (1 + x / 100)
  let second_year_population : ℝ := first_year_population * (1 - 5 / 100)
  second_year_population = 9975 → x = 5 := by
sorry

end NUMINAMATH_CALUDE_population_change_l1156_115634


namespace NUMINAMATH_CALUDE_pill_supply_duration_l1156_115646

/-- Proves that a supply of pills lasts for a specific number of months -/
theorem pill_supply_duration (total_pills : ℕ) (days_per_pill : ℕ) (days_per_month : ℕ) 
  (h1 : total_pills = 120)
  (h2 : days_per_pill = 2)
  (h3 : days_per_month = 30) :
  (total_pills * days_per_pill) / days_per_month = 8 := by
  sorry

#check pill_supply_duration

end NUMINAMATH_CALUDE_pill_supply_duration_l1156_115646


namespace NUMINAMATH_CALUDE_johns_journey_speed_l1156_115678

theorem johns_journey_speed (total_distance : ℝ) (first_duration : ℝ) (second_duration : ℝ) (second_speed : ℝ) (S : ℝ) :
  total_distance = 240 →
  first_duration = 2 →
  second_duration = 3 →
  second_speed = 50 →
  total_distance = first_duration * S + second_duration * second_speed →
  S = 45 := by
  sorry

end NUMINAMATH_CALUDE_johns_journey_speed_l1156_115678


namespace NUMINAMATH_CALUDE_second_month_sale_l1156_115685

def sale_month1 : ℕ := 6235
def sale_month3 : ℕ := 6855
def sale_month4 : ℕ := 7230
def sale_month5 : ℕ := 6562
def sale_month6 : ℕ := 5191
def desired_average : ℕ := 6500
def num_months : ℕ := 6

theorem second_month_sale :
  ∃ (sale_month2 : ℕ),
    sale_month2 = desired_average * num_months - (sale_month1 + sale_month3 + sale_month4 + sale_month5 + sale_month6) ∧
    sale_month2 = 6927 := by
  sorry

end NUMINAMATH_CALUDE_second_month_sale_l1156_115685


namespace NUMINAMATH_CALUDE_inequality_range_l1156_115631

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*x > 2*a*x + a) ↔ -4 < a ∧ a < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l1156_115631


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1156_115636

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  d : ℚ      -- Common difference
  sum : ℕ → ℚ -- Sum function
  sum_formula : ∀ n, sum n = n * (2 * a 1 + (n - 1) * d) / 2
  term_formula : ∀ n, a n = a 1 + (n - 1) * d

/-- The common difference of the arithmetic sequence is 4 -/
theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence)
  (sum_5 : seq.sum 5 = -15)
  (sum_terms : seq.a 2 + seq.a 5 = -2) :
  seq.d = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1156_115636


namespace NUMINAMATH_CALUDE_function_inequality_condition_l1156_115699

open Real

theorem function_inequality_condition (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ →
    (a * exp x₁ / x₁ - x₁) / x₂ - (a * exp x₂ / x₂ - x₂) / x₁ < 0) ↔
  a ≥ -exp 1 :=
sorry

end NUMINAMATH_CALUDE_function_inequality_condition_l1156_115699


namespace NUMINAMATH_CALUDE_exists_ten_points_five_kites_l1156_115670

/-- A point on a 4x4 grid --/
structure GridPoint where
  x : Fin 4
  y : Fin 4

/-- A kite formed by four points on the grid --/
structure Kite where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint
  p4 : GridPoint

/-- Check if four points form a valid kite --/
def is_valid_kite (k : Kite) : Prop :=
  -- Two pairs of adjacent sides have equal length
  -- Diagonals intersect at a right angle
  -- One diagonal bisects the other
  sorry

/-- Count the number of kites formed by a set of points --/
def count_kites (points : Finset GridPoint) : Nat :=
  sorry

/-- Theorem stating that there exists an arrangement of 10 points forming exactly 5 kites --/
theorem exists_ten_points_five_kites :
  ∃ (points : Finset GridPoint),
    points.card = 10 ∧ count_kites points = 5 :=
  sorry

end NUMINAMATH_CALUDE_exists_ten_points_five_kites_l1156_115670


namespace NUMINAMATH_CALUDE_contrapositive_square_inequality_l1156_115630

theorem contrapositive_square_inequality (x y : ℝ) :
  (¬(x > y) → ¬(x^2 > y^2)) ↔ (x ≤ y → x^2 ≤ y^2) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_square_inequality_l1156_115630


namespace NUMINAMATH_CALUDE_households_with_car_l1156_115652

theorem households_with_car (total : Nat) (without_car_or_bike : Nat) (with_both : Nat) (with_bike_only : Nat)
  (h1 : total = 90)
  (h2 : without_car_or_bike = 11)
  (h3 : with_both = 18)
  (h4 : with_bike_only = 35) :
  total - without_car_or_bike - with_bike_only + with_both = 62 := by
sorry

end NUMINAMATH_CALUDE_households_with_car_l1156_115652


namespace NUMINAMATH_CALUDE_line_slope_m_values_l1156_115650

theorem line_slope_m_values (m : ℝ) : 
  (∃ a b c : ℝ, (m^2 + m - 4) * a + (m + 4) * b + (2 * m + 1) = c ∧ 
   (m^2 + m - 4) = -(m + 4) ∧ (m^2 + m - 4) ≠ 0) → 
  m = 0 ∨ m = -2 := by
sorry

end NUMINAMATH_CALUDE_line_slope_m_values_l1156_115650


namespace NUMINAMATH_CALUDE_sum_every_third_odd_integer_l1156_115666

/-- The sum of every third odd integer between 200 and 500 (inclusive) is 17400 -/
theorem sum_every_third_odd_integer : 
  (Finset.range 50).sum (fun i => 201 + 6 * i) = 17400 := by
  sorry

end NUMINAMATH_CALUDE_sum_every_third_odd_integer_l1156_115666


namespace NUMINAMATH_CALUDE_boat_drift_l1156_115656

/-- Calculate the drift of a boat crossing a river -/
theorem boat_drift (river_width : ℝ) (boat_speed : ℝ) (crossing_time : ℝ) :
  river_width = 400 ∧ boat_speed = 10 ∧ crossing_time = 50 →
  boat_speed * crossing_time - river_width = 100 := by
  sorry

end NUMINAMATH_CALUDE_boat_drift_l1156_115656


namespace NUMINAMATH_CALUDE_tangent_line_minimum_value_l1156_115649

/-- The minimum value of 1/a^2 + 1/b^2 for a line tangent to a circle -/
theorem tangent_line_minimum_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ a * x + 2 * b * y + 2 = 0 ∧ x^2 + y^2 = 2) :
  (1 / a^2 + 1 / b^2 : ℝ) ≥ 9/2 ∧ 
  (∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 
    (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ a₀ * x + 2 * b₀ * y + 2 = 0 ∧ x^2 + y^2 = 2) ∧
    1 / a₀^2 + 1 / b₀^2 = 9/2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_minimum_value_l1156_115649


namespace NUMINAMATH_CALUDE_lillian_cupcakes_l1156_115662

/-- Represents the number of dozen cupcakes Lillian can bake and ice --/
def cupcakes_dozen : ℕ := by sorry

theorem lillian_cupcakes :
  let initial_sugar : ℕ := 3
  let bags_bought : ℕ := 2
  let sugar_per_bag : ℕ := 6
  let sugar_for_batter : ℕ := 1
  let sugar_for_frosting : ℕ := 2
  
  let total_sugar : ℕ := initial_sugar + bags_bought * sugar_per_bag
  let sugar_per_dozen : ℕ := sugar_for_batter + sugar_for_frosting
  
  cupcakes_dozen = total_sugar / sugar_per_dozen ∧ cupcakes_dozen = 5 := by sorry

end NUMINAMATH_CALUDE_lillian_cupcakes_l1156_115662


namespace NUMINAMATH_CALUDE_odd_function_graph_point_l1156_115608

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_graph_point (f : ℝ → ℝ) (a : ℝ) :
  is_odd_function f → f (-a) = -f a :=
by
  sorry

end NUMINAMATH_CALUDE_odd_function_graph_point_l1156_115608


namespace NUMINAMATH_CALUDE_last_three_digits_are_218_l1156_115635

/-- A function that generates the list of positive integers starting with 2 -/
def digitsStartingWith2 (n : ℕ) : ℕ :=
  if n < 10 then 2
  else if n < 100 then 20 + (n - 10)
  else if n < 1000 then 200 + (n - 100)
  else 2000 + (n - 1000)

/-- A function that returns the nth digit in the list -/
def nthDigit (n : ℕ) : ℕ :=
  let number := digitsStartingWith2 ((n - 1) / 4 + 1)
  let digitPosition := (n - 1) % 4
  (number / (10 ^ (3 - digitPosition))) % 10

/-- The theorem to be proved -/
theorem last_three_digits_are_218 :
  (nthDigit 1198) * 100 + (nthDigit 1199) * 10 + nthDigit 1200 = 218 := by
  sorry


end NUMINAMATH_CALUDE_last_three_digits_are_218_l1156_115635


namespace NUMINAMATH_CALUDE_mangoes_per_box_l1156_115603

/-- Given a total of 4320 mangoes distributed equally among 36 boxes,
    prove that there are 10 dozens of mangoes in each box. -/
theorem mangoes_per_box (total_mangoes : Nat) (num_boxes : Nat) 
    (h1 : total_mangoes = 4320) (h2 : num_boxes = 36) :
    (total_mangoes / (12 * num_boxes) : Nat) = 10 := by
  sorry

end NUMINAMATH_CALUDE_mangoes_per_box_l1156_115603


namespace NUMINAMATH_CALUDE_fraction_product_equals_one_over_23426_l1156_115665

def fraction_product : ℕ → ℚ
  | 0 => 1
  | n + 1 => (n + 1 : ℚ) / (n + 5 : ℚ) * fraction_product n

theorem fraction_product_equals_one_over_23426 :
  fraction_product 49 = 1 / 23426 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equals_one_over_23426_l1156_115665


namespace NUMINAMATH_CALUDE_range_a_theorem_l1156_115647

/-- Proposition p: The solution set of the inequality x^2+(a-1)x+1≤0 is the empty set ∅ -/
def p (a : ℝ) : Prop :=
  ∀ x, x^2 + (a-1)*x + 1 > 0

/-- Proposition q: The function y=(a-1)^x is an increasing function -/
def q (a : ℝ) : Prop :=
  ∀ x y, x < y → (a-1)^x < (a-1)^y

/-- The range of a satisfying the given conditions -/
def range_a : Set ℝ :=
  {a | (-1 < a ∧ a ≤ 2) ∨ a ≥ 3}

/-- Theorem stating that given the conditions, the range of a is as specified -/
theorem range_a_theorem (a : ℝ) :
  (¬(p a ∧ q a)) → (p a ∨ q a) → a ∈ range_a :=
by sorry

end NUMINAMATH_CALUDE_range_a_theorem_l1156_115647


namespace NUMINAMATH_CALUDE_certain_event_at_least_one_genuine_l1156_115658

theorem certain_event_at_least_one_genuine :
  ∀ (total : ℕ) (genuine : ℕ) (defective : ℕ) (selected : ℕ),
    total = 12 →
    genuine = 10 →
    defective = 2 →
    total = genuine + defective →
    selected = 3 →
    (∀ outcome : Finset (Fin total),
      outcome.card = selected →
      ∃ i ∈ outcome, i.val < genuine) :=
by sorry

end NUMINAMATH_CALUDE_certain_event_at_least_one_genuine_l1156_115658


namespace NUMINAMATH_CALUDE_part_one_part_two_l1156_115609

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

-- Part 1
theorem part_one :
  ∀ x : ℝ, (p x 1 ∧ q x) → (2 < x ∧ x < 3) :=
sorry

-- Part 2
theorem part_two :
  (∀ x : ℝ, (2 < x ∧ x < 3) → ∃ a : ℝ, a > 0 ∧ a < x ∧ x < 3*a) →
  ∃ a : ℝ, 1 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1156_115609


namespace NUMINAMATH_CALUDE_percentage_problem_l1156_115624

theorem percentage_problem (X : ℝ) : 
  (28 / 100) * 400 + (45 / 100) * X = 224.5 → X = 250 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1156_115624


namespace NUMINAMATH_CALUDE_absolute_difference_l1156_115671

theorem absolute_difference (a x : ℝ) (h1 : a < 0) (h2 : |a| * x ≤ a) : 
  |x + 1| - |x - 3| = -4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_l1156_115671


namespace NUMINAMATH_CALUDE_coffee_machine_payoff_l1156_115657

/-- Calculates the number of days until a coffee machine pays for itself. --/
def coffee_machine_payoff_days (machine_price : ℕ) (discount : ℕ) (daily_cost : ℕ) (prev_coffees : ℕ) (prev_price : ℕ) : ℕ :=
  let actual_cost := machine_price - discount
  let prev_daily_expense := prev_coffees * prev_price
  let daily_savings := prev_daily_expense - daily_cost
  actual_cost / daily_savings

/-- Theorem stating that under the given conditions, the coffee machine pays for itself in 36 days. --/
theorem coffee_machine_payoff :
  coffee_machine_payoff_days 200 20 3 2 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_coffee_machine_payoff_l1156_115657


namespace NUMINAMATH_CALUDE_complex_power_110_deg_36_l1156_115659

theorem complex_power_110_deg_36 :
  (Complex.exp (110 * π / 180 * Complex.I)) ^ 36 = -1/2 + Complex.I * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_110_deg_36_l1156_115659


namespace NUMINAMATH_CALUDE_brownie_pieces_count_l1156_115633

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents a pan of brownies -/
structure BrowniePan where
  panDimensions : Dimensions
  pieceDimensions : Dimensions

/-- Calculates the number of brownie pieces that can be cut from the pan -/
def numberOfPieces (pan : BrowniePan) : ℕ :=
  (area pan.panDimensions) / (area pan.pieceDimensions)

theorem brownie_pieces_count :
  let pan : BrowniePan := {
    panDimensions := { length := 24, width := 15 },
    pieceDimensions := { length := 3, width := 2 }
  }
  numberOfPieces pan = 60 := by sorry

end NUMINAMATH_CALUDE_brownie_pieces_count_l1156_115633


namespace NUMINAMATH_CALUDE_allysons_age_l1156_115669

theorem allysons_age (hirams_age allyson_age : ℕ) : 
  hirams_age = 40 →
  hirams_age + 12 = 2 * allyson_age - 4 →
  allyson_age = 28 := by
sorry

end NUMINAMATH_CALUDE_allysons_age_l1156_115669


namespace NUMINAMATH_CALUDE_calculation_proof_l1156_115607

theorem calculation_proof : 
  ((0.8 + (1 / 5)) * 24 + 6.6) / (9 / 14) - 7.6 = 40 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1156_115607


namespace NUMINAMATH_CALUDE_george_total_blocks_l1156_115612

/-- The number of boxes George has -/
def num_boxes : ℕ := 2

/-- The number of blocks in each box -/
def blocks_per_box : ℕ := 6

/-- Theorem stating the total number of blocks George has -/
theorem george_total_blocks : num_boxes * blocks_per_box = 12 := by
  sorry

end NUMINAMATH_CALUDE_george_total_blocks_l1156_115612


namespace NUMINAMATH_CALUDE_product_mod_sixty_l1156_115623

theorem product_mod_sixty (m : ℕ) : 
  198 * 953 ≡ m [ZMOD 60] → 0 ≤ m → m < 60 → m = 54 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_sixty_l1156_115623


namespace NUMINAMATH_CALUDE_magazine_subscription_pigeonhole_l1156_115638

theorem magazine_subscription_pigeonhole 
  (total_students : Nat) 
  (subscription_combinations : Nat) 
  (h1 : total_students = 39) 
  (h2 : subscription_combinations = 7) :
  ∃ (combination : Nat), combination ≤ subscription_combinations ∧ 
    (total_students / subscription_combinations + 1 : Nat) ≤ 
      (λ i => (total_students / subscription_combinations : Nat) + 
        if i ≤ (total_students % subscription_combinations) then 1 else 0) combination :=
by
  sorry

end NUMINAMATH_CALUDE_magazine_subscription_pigeonhole_l1156_115638


namespace NUMINAMATH_CALUDE_solution_set_min_value_l1156_115611

-- Define the functions f and g
def f (x : ℝ) : ℝ := x - 2
def g (x : ℝ) : ℝ := 2*x - 5

-- Statement for the solution set
theorem solution_set : 
  {x : ℝ | |f x| + |g x| ≤ 2} = {x : ℝ | 5/3 ≤ x ∧ x ≤ 3} := by sorry

-- Statement for the minimum value
theorem min_value : 
  ∀ x : ℝ, |f (2*x)| + |g x| ≥ 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_min_value_l1156_115611


namespace NUMINAMATH_CALUDE_sachins_age_l1156_115619

theorem sachins_age (rahuls_age : ℝ) : 
  (rahuls_age + 7) / rahuls_age = 11 / 9 → rahuls_age + 7 = 38.5 := by
  sorry

end NUMINAMATH_CALUDE_sachins_age_l1156_115619


namespace NUMINAMATH_CALUDE_debt_payment_average_l1156_115674

theorem debt_payment_average (n : ℕ) (first_payment second_payment : ℚ) : 
  n = 40 →
  first_payment = 410 →
  second_payment = first_payment + 65 →
  (20 * first_payment + 20 * second_payment) / n = 442.50 := by
  sorry

end NUMINAMATH_CALUDE_debt_payment_average_l1156_115674


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l1156_115663

theorem quadratic_equation_result (y : ℝ) (h : 7 * y^2 + 6 = 5 * y + 14) : 
  (14 * y - 2)^2 = 258 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l1156_115663


namespace NUMINAMATH_CALUDE_laborer_income_l1156_115644

theorem laborer_income (
  avg_expenditure_6months : ℝ)
  (fell_into_debt : Prop)
  (reduced_expenses_4months : ℝ)
  (debt_cleared_and_saved : ℝ) :
  avg_expenditure_6months = 85 →
  fell_into_debt →
  reduced_expenses_4months = 60 →
  debt_cleared_and_saved = 30 →
  ∃ (monthly_income : ℝ), monthly_income = 78 :=
by sorry

end NUMINAMATH_CALUDE_laborer_income_l1156_115644


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l1156_115628

def calculate_total_cost (type_a_count : ℕ) (type_b_count : ℕ) (type_c_count : ℕ)
  (type_a_price : ℚ) (type_b_price : ℚ) (type_c_price : ℚ)
  (type_a_discount : ℚ) (type_b_discount : ℚ) (type_c_discount : ℚ)
  (type_a_discount_threshold : ℕ) (type_b_discount_threshold : ℕ) (type_c_discount_threshold : ℕ) : ℚ :=
  let type_a_cost := type_a_count * type_a_price
  let type_b_cost := type_b_count * type_b_price
  let type_c_cost := type_c_count * type_c_price
  let type_a_discounted_cost := if type_a_count > type_a_discount_threshold then type_a_cost * (1 - type_a_discount) else type_a_cost
  let type_b_discounted_cost := if type_b_count > type_b_discount_threshold then type_b_cost * (1 - type_b_discount) else type_b_cost
  let type_c_discounted_cost := if type_c_count > type_c_discount_threshold then type_c_cost * (1 - type_c_discount) else type_c_cost
  type_a_discounted_cost + type_b_discounted_cost + type_c_discounted_cost

theorem total_cost_is_correct :
  calculate_total_cost 150 90 60 2 3 5 0.2 0.15 0.1 100 50 30 = 739.5 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l1156_115628


namespace NUMINAMATH_CALUDE_inscribed_hexagon_area_l1156_115616

/-- A regular hexagon inscribed in a semicircle -/
structure InscribedHexagon where
  /-- The diameter of the semicircle -/
  diameter : ℝ
  /-- One side of the hexagon lies along the diameter -/
  side_on_diameter : Bool
  /-- Two opposite vertices of the hexagon are on the semicircle -/
  vertices_on_semicircle : Bool

/-- The area of an inscribed hexagon -/
def area (h : InscribedHexagon) : ℝ := sorry

/-- Theorem: The area of a regular hexagon inscribed in a semicircle of diameter 1 is 3√3/26 -/
theorem inscribed_hexagon_area :
  ∀ (h : InscribedHexagon), h.diameter = 1 → h.side_on_diameter = true → h.vertices_on_semicircle = true →
  area h = 3 * Real.sqrt 3 / 26 := by sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_area_l1156_115616


namespace NUMINAMATH_CALUDE_larger_integer_proof_l1156_115691

theorem larger_integer_proof (x y : ℕ) (h1 : y = 4 * x) (h2 : (x + 6) / y = 1 / 3) : y = 72 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_proof_l1156_115691


namespace NUMINAMATH_CALUDE_hybrid_car_trip_length_l1156_115655

theorem hybrid_car_trip_length : 
  ∀ (trip_length : ℝ),
  (trip_length / (0.02 * (trip_length - 40)) = 55) →
  trip_length = 440 :=
by sorry

end NUMINAMATH_CALUDE_hybrid_car_trip_length_l1156_115655


namespace NUMINAMATH_CALUDE_sample_size_is_100_l1156_115643

/-- A structure representing a statistical sampling process -/
structure SamplingProcess where
  totalStudents : Nat
  selectedStudents : Nat

/-- Definition of sample size for a SamplingProcess -/
def sampleSize (sp : SamplingProcess) : Nat := sp.selectedStudents

/-- Theorem stating that for the given sampling process, the sample size is 100 -/
theorem sample_size_is_100 (sp : SamplingProcess) 
  (h1 : sp.totalStudents = 1000) 
  (h2 : sp.selectedStudents = 100) : 
  sampleSize sp = 100 := by
  sorry

#check sample_size_is_100

end NUMINAMATH_CALUDE_sample_size_is_100_l1156_115643


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l1156_115695

/-- Represents the number of employees in each age group -/
structure EmployeeCount where
  middleAged : ℕ
  young : ℕ
  elderly : ℕ

/-- Calculates the stratified sample sizes for each age group -/
def stratifiedSample (total : ℕ) (ratio : EmployeeCount) (sampleSize : ℕ) : EmployeeCount :=
  let totalRatio := ratio.middleAged + ratio.young + ratio.elderly
  { middleAged := (sampleSize * ratio.middleAged) / totalRatio,
    young := (sampleSize * ratio.young) / totalRatio,
    elderly := (sampleSize * ratio.elderly) / totalRatio }

theorem correct_stratified_sample :
  let total := 3200
  let ratio := EmployeeCount.mk 5 3 2
  let sampleSize := 400
  let result := stratifiedSample total ratio sampleSize
  result.middleAged = 200 ∧ result.young = 120 ∧ result.elderly = 80 := by
  sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_l1156_115695


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1156_115683

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, ax^2 + b*x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → a - b = -10 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1156_115683


namespace NUMINAMATH_CALUDE_balloon_cost_difference_l1156_115610

/-- The cost of a helium balloon in dollars -/
def helium_cost : ℚ := 1.50

/-- The cost of a foil balloon in dollars -/
def foil_cost : ℚ := 2.50

/-- The number of helium balloons Allan bought -/
def allan_helium : ℕ := 2

/-- The number of foil balloons Allan bought -/
def allan_foil : ℕ := 3

/-- The number of helium balloons Jake bought -/
def jake_helium : ℕ := 4

/-- The number of foil balloons Jake bought -/
def jake_foil : ℕ := 2

/-- The total cost of Allan's balloons -/
def allan_total : ℚ := allan_helium * helium_cost + allan_foil * foil_cost

/-- The total cost of Jake's balloons -/
def jake_total : ℚ := jake_helium * helium_cost + jake_foil * foil_cost

/-- Theorem stating the difference in cost between Jake's and Allan's balloons -/
theorem balloon_cost_difference : jake_total - allan_total = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_balloon_cost_difference_l1156_115610


namespace NUMINAMATH_CALUDE_winwin_processing_fee_l1156_115637

/-- Calculates the processing fee for a lottery win -/
def processing_fee (total_win : ℝ) (tax_rate : ℝ) (take_home : ℝ) : ℝ :=
  total_win * (1 - tax_rate) - take_home

/-- Theorem: The processing fee for Winwin's lottery win is $5 -/
theorem winwin_processing_fee :
  processing_fee 50 0.2 35 = 5 := by
  sorry

end NUMINAMATH_CALUDE_winwin_processing_fee_l1156_115637


namespace NUMINAMATH_CALUDE_mike_remaining_cards_l1156_115681

/-- Calculates the number of baseball cards Mike has after Sam's purchase -/
def remaining_cards (initial : ℕ) (bought : ℕ) : ℕ :=
  initial - bought

/-- Theorem stating that Mike has 74 baseball cards after Sam's purchase -/
theorem mike_remaining_cards :
  remaining_cards 87 13 = 74 := by
  sorry

end NUMINAMATH_CALUDE_mike_remaining_cards_l1156_115681


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_is_27_l1156_115677

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  first : a 1 = 20
  last : ∃ n : ℕ, a n = 54
  sum : ∃ n : ℕ, (n : ℝ) / 2 * (a 1 + a n) = 999

/-- The number of terms in the arithmetic sequence is 27 -/
theorem arithmetic_sequence_n_is_27 (seq : ArithmeticSequence) : 
  ∃ n : ℕ, n = 27 ∧ seq.a n = 54 ∧ (n : ℝ) / 2 * (seq.a 1 + seq.a n) = 999 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_n_is_27_l1156_115677


namespace NUMINAMATH_CALUDE_good_apples_count_l1156_115639

theorem good_apples_count (total_apples unripe_apples : ℕ) 
  (h1 : total_apples = 14) 
  (h2 : unripe_apples = 6) : 
  total_apples - unripe_apples = 8 := by
sorry

end NUMINAMATH_CALUDE_good_apples_count_l1156_115639


namespace NUMINAMATH_CALUDE_theater_seat_count_l1156_115602

/-- Calculates the total number of seats in a theater with the given configuration. -/
def theater_seats (total_rows : ℕ) (odd_row_seats : ℕ) (even_row_seats : ℕ) : ℕ :=
  let odd_rows := (total_rows + 1) / 2
  let even_rows := total_rows / 2
  odd_rows * odd_row_seats + even_rows * even_row_seats

/-- Theorem stating that a theater with 11 rows, where odd rows have 15 seats
    and even rows have 16 seats, has a total of 170 seats. -/
theorem theater_seat_count :
  theater_seats 11 15 16 = 170 := by
  sorry

#eval theater_seats 11 15 16

end NUMINAMATH_CALUDE_theater_seat_count_l1156_115602


namespace NUMINAMATH_CALUDE_pokemon_card_paradox_l1156_115621

theorem pokemon_card_paradox (initial_cards sold_cards : ℕ) : 
  (∃ (received_cards bought_cards final_cards : ℕ), 
    received_cards = 41 ∧ 
    bought_cards = 20 ∧ 
    final_cards = 34 ∧ 
    initial_cards - sold_cards + received_cards + bought_cards = final_cards) → 
  sold_cards > initial_cards := by
  sorry

end NUMINAMATH_CALUDE_pokemon_card_paradox_l1156_115621


namespace NUMINAMATH_CALUDE_fraction_transformation_l1156_115641

theorem fraction_transformation (a b : ℝ) (h : a ≠ b) : -a / (a - b) = a / (b - a) := by
  sorry

end NUMINAMATH_CALUDE_fraction_transformation_l1156_115641


namespace NUMINAMATH_CALUDE_power_of_power_l1156_115688

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1156_115688


namespace NUMINAMATH_CALUDE_definite_integral_reciprocal_l1156_115675

theorem definite_integral_reciprocal : ∫ x in (1:ℝ)..2, (1:ℝ) / x = Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_reciprocal_l1156_115675


namespace NUMINAMATH_CALUDE_abc_product_l1156_115698

theorem abc_product (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_sum : a + b + c = 30)
  (h_eq : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + 672 / (a * b * c) = 1) :
  a * b * c = 2808 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l1156_115698


namespace NUMINAMATH_CALUDE_product_correction_l1156_115625

/-- Reverses the digits of a two-digit number -/
def reverseDigits (n : Nat) : Nat :=
  (n % 10) * 10 + (n / 10)

theorem product_correction (p q : Nat) :
  p ≥ 10 ∧ p < 100 →  -- p is a two-digit number
  q > 0 →  -- q is positive
  reverseDigits p * q = 221 →
  p * q = 923 := by
sorry

end NUMINAMATH_CALUDE_product_correction_l1156_115625
