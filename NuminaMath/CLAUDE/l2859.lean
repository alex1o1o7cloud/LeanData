import Mathlib

namespace NUMINAMATH_CALUDE_smallest_k_carboxylic_for_8002_l2859_285997

/-- A function that checks if a number has all digits the same --/
def allDigitsSame (n : ℕ) : Prop := sorry

/-- A function that checks if a list of natural numbers are all distinct --/
def allDistinct (list : List ℕ) : Prop := sorry

/-- A function that checks if all numbers in a list are greater than 9 --/
def allGreaterThan9 (list : List ℕ) : Prop := sorry

/-- A function that checks if a number is k-carboxylic --/
def isKCarboxylic (n k : ℕ) : Prop :=
  ∃ (list : List ℕ), 
    list.length = k ∧ 
    list.sum = n ∧ 
    allDistinct list ∧ 
    allGreaterThan9 list ∧ 
    ∀ m ∈ list, allDigitsSame m

/-- The main theorem --/
theorem smallest_k_carboxylic_for_8002 :
  (isKCarboxylic 8002 14) ∧ ∀ k < 14, ¬(isKCarboxylic 8002 k) := by sorry

end NUMINAMATH_CALUDE_smallest_k_carboxylic_for_8002_l2859_285997


namespace NUMINAMATH_CALUDE_equal_angle_locus_for_given_flagpoles_l2859_285933

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a flagpole -/
structure Flagpole where
  base : Point
  height : ℝ

/-- The locus of points with equal angles of elevation to two flagpoles -/
def equalAngleLocus (pole1 pole2 : Flagpole) : Set (Point) :=
  {p : Point | (p.x - 85/8)^2 + p.y^2 = (75/8)^2}

theorem equal_angle_locus_for_given_flagpoles :
  let pole1 : Flagpole := ⟨Point.mk (-5) 0, 5⟩
  let pole2 : Flagpole := ⟨Point.mk 5 0, 3⟩
  equalAngleLocus pole1 pole2 =
    {p : Point | (p.x - 85/8)^2 + p.y^2 = (75/8)^2} :=
by
  sorry

end NUMINAMATH_CALUDE_equal_angle_locus_for_given_flagpoles_l2859_285933


namespace NUMINAMATH_CALUDE_certain_number_addition_l2859_285910

theorem certain_number_addition (x : ℤ) : x + 36 = 71 → x + 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_addition_l2859_285910


namespace NUMINAMATH_CALUDE_bagel_count_l2859_285900

/-- Represents the number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Represents the cost of a bagel in cents -/
def bagel_cost : ℕ := 65

/-- Represents the cost of a muffin in cents -/
def muffin_cost : ℕ := 40

/-- Represents the number of days in the week -/
def days_in_week : ℕ := 7

/-- 
Given a 7-day period where either a 40-cent muffin or a 65-cent bagel is bought each day, 
and the total spending is a whole number of dollars, the number of bagels bought must be 4.
-/
theorem bagel_count : 
  ∀ (b : ℕ), 
  b ≤ days_in_week → 
  (bagel_cost * b + muffin_cost * (days_in_week - b)) % cents_per_dollar = 0 → 
  b = 4 := by
sorry

end NUMINAMATH_CALUDE_bagel_count_l2859_285900


namespace NUMINAMATH_CALUDE_max_consecutive_special_is_correct_l2859_285938

/-- A number is special if it's a 20-digit number that cannot be represented
    as a product of a 10-digit number and an 11-digit number. -/
def IsSpecial (n : ℕ) : Prop :=
  10^19 ≤ n ∧ n < 10^20 ∧
  ∀ a b : ℕ, 10^9 ≤ a ∧ a < 10^10 → 10^10 ≤ b ∧ b < 10^11 → n ≠ a * b

/-- The maximum quantity of consecutive special numbers -/
def MaxConsecutiveSpecial : ℕ := 10^9 - 1

/-- Theorem stating that MaxConsecutiveSpecial is indeed the maximum
    quantity of consecutive special numbers -/
theorem max_consecutive_special_is_correct :
  (∀ k : ℕ, k < MaxConsecutiveSpecial →
    ∀ i : ℕ, i < k → IsSpecial (10^19 + i + 1)) ∧
  (∀ k : ℕ, k > MaxConsecutiveSpecial →
    ∃ i j : ℕ, i < j ∧ j - i = k ∧ ¬IsSpecial j) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_special_is_correct_l2859_285938


namespace NUMINAMATH_CALUDE_b_investment_is_4200_l2859_285929

/-- Represents the investment and profit details of a partnership business -/
structure BusinessPartnership where
  a_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  a_profit_share : ℕ

/-- Calculates B's investment given the partnership details -/
def calculate_b_investment (bp : BusinessPartnership) : ℕ :=
  bp.total_profit * bp.a_investment / bp.a_profit_share - bp.a_investment - bp.c_investment

/-- Theorem stating that B's investment is 4200 given the specified conditions -/
theorem b_investment_is_4200 (bp : BusinessPartnership) 
  (h1 : bp.a_investment = 6300)
  (h2 : bp.c_investment = 10500)
  (h3 : bp.total_profit = 13000)
  (h4 : bp.a_profit_share = 3900) :
  calculate_b_investment bp = 4200 := by
  sorry

#eval calculate_b_investment ⟨6300, 10500, 13000, 3900⟩

end NUMINAMATH_CALUDE_b_investment_is_4200_l2859_285929


namespace NUMINAMATH_CALUDE_comic_book_stacking_theorem_l2859_285949

def num_spiderman : ℕ := 7
def num_archie : ℕ := 6
def num_garfield : ℕ := 4

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

def permutations_within_groups : ℕ := 
  factorial num_spiderman * factorial num_archie * factorial num_garfield

def group_arrangements : ℕ := 2 * 2

theorem comic_book_stacking_theorem :
  permutations_within_groups * group_arrangements = 19353600 := by
  sorry

end NUMINAMATH_CALUDE_comic_book_stacking_theorem_l2859_285949


namespace NUMINAMATH_CALUDE_rocket_max_altitude_l2859_285920

/-- The altitude function of the rocket -/
def h (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 25

/-- Theorem: The maximum altitude reached by the rocket is 45 meters -/
theorem rocket_max_altitude :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 45 := by
  sorry

end NUMINAMATH_CALUDE_rocket_max_altitude_l2859_285920


namespace NUMINAMATH_CALUDE_special_triangle_angle_difference_l2859_285947

/-- A triangle with special angle properties -/
structure SpecialTriangle where
  /-- The smallest angle of the triangle -/
  a : ℕ
  /-- The middle angle of the triangle -/
  b : ℕ
  /-- The largest angle of the triangle -/
  c : ℕ
  /-- One of the angles is a prime number -/
  h1 : Prime a ∨ Prime b ∨ Prime c
  /-- Two of the angles are squares of prime numbers -/
  h2 : ∃ p q : ℕ, Prime p ∧ Prime q ∧ 
       ((b = p^2 ∧ c = q^2) ∨ (a = p^2 ∧ c = q^2) ∨ (a = p^2 ∧ b = q^2))
  /-- The sum of the angles is 180 degrees -/
  h3 : a + b + c = 180
  /-- The angles are in ascending order -/
  h4 : a ≤ b ∧ b ≤ c

/-- The theorem stating the difference between the largest and smallest angles -/
theorem special_triangle_angle_difference (t : SpecialTriangle) : t.c - t.a = 167 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_angle_difference_l2859_285947


namespace NUMINAMATH_CALUDE_inequality_proof_l2859_285911

theorem inequality_proof (x : ℝ) (h1 : x > -1) (h2 : x ≠ 0) :
  (2 * |x|) / (2 + x) < |Real.log (1 + x)| ∧ |Real.log (1 + x)| < |x| / Real.sqrt (1 + x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2859_285911


namespace NUMINAMATH_CALUDE_treasure_probability_value_l2859_285971

/-- The probability of finding exactly 4 islands with treasure and no traps out of 8 islands -/
def treasure_probability : ℚ :=
  let n : ℕ := 8  -- Total number of islands
  let k : ℕ := 4  -- Number of islands with treasure
  let p_treasure : ℚ := 1/5  -- Probability of treasure and no traps
  let p_neither : ℚ := 7/10  -- Probability of neither treasure nor traps
  Nat.choose n k * p_treasure^k * p_neither^(n-k)

/-- The probability of finding exactly 4 islands with treasure and no traps out of 8 islands
    is equal to 673/25000 -/
theorem treasure_probability_value : treasure_probability = 673/25000 := by
  sorry

end NUMINAMATH_CALUDE_treasure_probability_value_l2859_285971


namespace NUMINAMATH_CALUDE_circle_area_increase_l2859_285967

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_increase_l2859_285967


namespace NUMINAMATH_CALUDE_arithmetic_progression_divisibility_l2859_285931

theorem arithmetic_progression_divisibility 
  (a : ℕ → ℕ) 
  (h_ap : ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d) 
  (h_div : ∀ n : ℕ, (a n * a (n + 31)) % 2005 = 0) : 
  ∀ n : ℕ, a n % 2005 = 0 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_divisibility_l2859_285931


namespace NUMINAMATH_CALUDE_ping_pong_meeting_l2859_285980

theorem ping_pong_meeting (total_legs : ℕ) (square_stool_legs round_stool_legs : ℕ) :
  total_legs = 33 ∧ square_stool_legs = 4 ∧ round_stool_legs = 3 →
  ∃ (total_members square_stools round_stools : ℕ),
    total_members = square_stools + round_stools ∧
    total_members * 2 + square_stools * square_stool_legs + round_stools * round_stool_legs = total_legs ∧
    total_members = 6 :=
by sorry

end NUMINAMATH_CALUDE_ping_pong_meeting_l2859_285980


namespace NUMINAMATH_CALUDE_digit_puzzle_proof_l2859_285993

theorem digit_puzzle_proof (P Q R S : ℕ) : 
  (P < 10 ∧ Q < 10 ∧ R < 10 ∧ S < 10) →
  (10 * P + Q) + (10 * R + P) = 10 * S + P →
  (10 * P + Q) - (10 * R + P) = P →
  S = 0 := by
  sorry

end NUMINAMATH_CALUDE_digit_puzzle_proof_l2859_285993


namespace NUMINAMATH_CALUDE_symmetry_coordinates_l2859_285982

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the origin -/
def symmetricToOrigin (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = -p.y

theorem symmetry_coordinates :
  let A : Point := ⟨3, -2⟩
  let A' : Point := ⟨-3, 2⟩
  symmetricToOrigin A A' → A'.x = -3 ∧ A'.y = 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_coordinates_l2859_285982


namespace NUMINAMATH_CALUDE_polynomial_equality_l2859_285945

theorem polynomial_equality : 110^5 - 5 * 110^4 + 10 * 110^3 - 10 * 110^2 + 5 * 110 - 1 = 161051000 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2859_285945


namespace NUMINAMATH_CALUDE_nine_gon_diagonals_l2859_285905

/-- The number of diagonals in a regular nine-sided polygon -/
def num_diagonals_nine_gon : ℕ :=
  (9 * (9 - 1)) / 2 - 9

theorem nine_gon_diagonals :
  num_diagonals_nine_gon = 27 := by
  sorry

end NUMINAMATH_CALUDE_nine_gon_diagonals_l2859_285905


namespace NUMINAMATH_CALUDE_largest_digit_sum_l2859_285972

theorem largest_digit_sum (a b c z : ℕ) : 
  (a < 10) → (b < 10) → (c < 10) → 
  (0 < z) → (z ≤ 12) → 
  (100 * a + 10 * b + c = 1000 / z) → 
  (a + b + c ≤ 8) :=
sorry

end NUMINAMATH_CALUDE_largest_digit_sum_l2859_285972


namespace NUMINAMATH_CALUDE_orange_marbles_count_l2859_285919

def total_marbles : ℕ := 24
def blue_marbles : ℕ := total_marbles / 2
def red_marbles : ℕ := 6

theorem orange_marbles_count : total_marbles - blue_marbles - red_marbles = 6 := by
  sorry

end NUMINAMATH_CALUDE_orange_marbles_count_l2859_285919


namespace NUMINAMATH_CALUDE_ab_value_l2859_285965

theorem ab_value (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 * b^2 + a^2 * b^3 = 20) : a * b = 2 ∨ a * b = -2 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2859_285965


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2859_285991

/-- The solution set of the inequality (x-2)(3-x) > 0 is the open interval (2, 3). -/
theorem solution_set_inequality (x : ℝ) : 
  (x - 2) * (3 - x) > 0 ↔ x ∈ Set.Ioo 2 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2859_285991


namespace NUMINAMATH_CALUDE_library_charge_calculation_l2859_285970

/-- Calculates the total amount paid for borrowed books --/
def total_amount_paid (daily_rate : ℚ) (book1_days : ℕ) (book2_days : ℕ) (num_books2 : ℕ) : ℚ :=
  daily_rate * book1_days + daily_rate * book2_days * num_books2

theorem library_charge_calculation :
  let daily_rate : ℚ := 50 / 100  -- 50 cents in dollars
  let book1_days : ℕ := 20
  let book2_days : ℕ := 31
  let num_books2 : ℕ := 2
  total_amount_paid daily_rate book1_days book2_days num_books2 = 41 := by
sorry

#eval total_amount_paid (50 / 100) 20 31 2

end NUMINAMATH_CALUDE_library_charge_calculation_l2859_285970


namespace NUMINAMATH_CALUDE_sum_A_and_B_l2859_285957

theorem sum_A_and_B : 
  let B := 278 + 365 * 3
  let A := 20 * 100 + 87 * 10
  A + B = 4243 := by
sorry

end NUMINAMATH_CALUDE_sum_A_and_B_l2859_285957


namespace NUMINAMATH_CALUDE_f_passes_through_2_8_f_neg_one_eq_neg_one_l2859_285923

/-- A power function passing through (2, 8) -/
def f (x : ℝ) : ℝ := x^3

/-- The function f passes through (2, 8) -/
theorem f_passes_through_2_8 : f 2 = 8 := by sorry

/-- The value of f(-1) is -1 -/
theorem f_neg_one_eq_neg_one : f (-1) = -1 := by sorry

end NUMINAMATH_CALUDE_f_passes_through_2_8_f_neg_one_eq_neg_one_l2859_285923


namespace NUMINAMATH_CALUDE_reeya_second_subject_score_l2859_285912

/-- Given Reeya's scores in 4 subjects and her average score, prove the score of the second subject. -/
theorem reeya_second_subject_score (score1 score2 score3 score4 : ℕ) (average : ℚ) :
  score1 = 55 →
  score3 = 82 →
  score4 = 55 →
  average = 67 →
  (score1 + score2 + score3 + score4 : ℚ) / 4 = average →
  score2 = 76 := by
sorry

end NUMINAMATH_CALUDE_reeya_second_subject_score_l2859_285912


namespace NUMINAMATH_CALUDE_insertion_possible_l2859_285934

/-- Represents a natural number with exactly 2007 digits -/
def Number2007 := { n : ℕ | 10^2006 ≤ n ∧ n < 10^2007 }

/-- Represents the operation of removing 7 digits from a number -/
def remove_seven_digits (n : Number2007) : ℕ := sorry

/-- Represents the operation of inserting 7 digits into a number -/
def insert_seven_digits (n : Number2007) : Number2007 := sorry

/-- The main theorem -/
theorem insertion_possible (a b : Number2007) :
  (∃ (c : ℕ), remove_seven_digits a = c ∧ remove_seven_digits b = c) →
  (∃ (d : Number2007), ∃ (f g : Number2007 → Number2007),
    f a = d ∧ g b = d ∧ 
    (∀ x : Number2007, ∃ y, insert_seven_digits x = f x ∧ insert_seven_digits y = g x)) :=
sorry

end NUMINAMATH_CALUDE_insertion_possible_l2859_285934


namespace NUMINAMATH_CALUDE_perpendicular_to_oblique_implies_perpendicular_to_projection_l2859_285921

/-- A plane in which we consider lines and their projections. -/
structure Plane where
  -- Add necessary fields here

/-- Represents a line in the plane. -/
structure Line (P : Plane) where
  -- Add necessary fields here

/-- Indicates that a line is oblique (not parallel or perpendicular to some reference). -/
def isOblique (P : Plane) (l : Line P) : Prop :=
  sorry

/-- The projection of a line onto the plane. -/
def projection (P : Plane) (l : Line P) : Line P :=
  sorry

/-- Indicates that two lines are perpendicular. -/
def isPerpendicular (P : Plane) (l1 l2 : Line P) : Prop :=
  sorry

/-- 
The main theorem: If a line is perpendicular to an oblique line in a plane,
then it is also perpendicular to the projection of the oblique line in this plane.
-/
theorem perpendicular_to_oblique_implies_perpendicular_to_projection
  (P : Plane) (l1 l2 : Line P) (h1 : isOblique P l1) (h2 : isPerpendicular P l1 l2) :
  isPerpendicular P (projection P l1) l2 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_oblique_implies_perpendicular_to_projection_l2859_285921


namespace NUMINAMATH_CALUDE_pear_problem_solution_l2859_285986

/-- Represents the pear selling problem --/
def PearProblem (initial_pears : ℝ) : Prop :=
  let sold_day1 := 0.20 * initial_pears
  let remaining_after_sale := initial_pears - sold_day1
  let thrown_day1 := 0.50 * remaining_after_sale
  let remaining_day2 := remaining_after_sale - thrown_day1
  let total_thrown := 0.72 * initial_pears
  let thrown_day2 := total_thrown - thrown_day1
  let sold_day2 := remaining_day2 - thrown_day2
  (sold_day2 / remaining_day2) = 0.20

/-- Theorem stating that the percentage of remaining pears sold on day 2 is 20% --/
theorem pear_problem_solution : 
  ∀ initial_pears : ℝ, initial_pears > 0 → PearProblem initial_pears :=
by
  sorry


end NUMINAMATH_CALUDE_pear_problem_solution_l2859_285986


namespace NUMINAMATH_CALUDE_total_weekly_time_l2859_285994

def parking_time : ℕ := 5
def walking_time : ℕ := 3
def long_wait_days : ℕ := 2
def short_wait_days : ℕ := 3
def long_wait_time : ℕ := 30
def short_wait_time : ℕ := 10
def work_days : ℕ := 5

theorem total_weekly_time :
  (parking_time + walking_time) * work_days +
  long_wait_days * long_wait_time +
  short_wait_days * short_wait_time = 130 := by
sorry

end NUMINAMATH_CALUDE_total_weekly_time_l2859_285994


namespace NUMINAMATH_CALUDE_insurance_compensation_l2859_285907

/-- Insurance compensation calculation --/
theorem insurance_compensation
  (insured_amount : ℝ)
  (deductible_percentage : ℝ)
  (actual_damage : ℝ)
  (h1 : insured_amount = 500000)
  (h2 : deductible_percentage = 0.01)
  (h3 : actual_damage = 4000)
  : min (max (actual_damage - insured_amount * deductible_percentage) 0) insured_amount = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_insurance_compensation_l2859_285907


namespace NUMINAMATH_CALUDE_question_1_question_2_l2859_285925

-- Define propositions p and q
def p (x : ℝ) : Prop := (x + 1) * (2 - x) ≥ 0
def q (m : ℝ) (x : ℝ) : Prop := x^2 + 2*m*x - m + 6 > 0

-- Theorem for question 1
theorem question_1 (m : ℝ) : (∀ x, q m x) → m ∈ Set.Ioo (-3 : ℝ) 2 :=
sorry

-- Theorem for question 2
theorem question_2 (m : ℝ) : 
  ((∀ x, p x → q m x) ∧ (∃ x, q m x ∧ ¬p x)) → m ∈ Set.Ioc (-3 : ℝ) 2 :=
sorry

end NUMINAMATH_CALUDE_question_1_question_2_l2859_285925


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2859_285952

theorem quadratic_factorization (x : ℝ) : 2 * x^2 - 12 * x + 18 = 2 * (x - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2859_285952


namespace NUMINAMATH_CALUDE_eighteenth_permutation_l2859_285909

def FourDigitPermutation : Type := Fin 4 → Fin 10

def isValidPermutation (p : FourDigitPermutation) : Prop :=
  (p 0 = 1 ∨ p 0 = 2 ∨ p 0 = 5 ∨ p 0 = 6) ∧
  (p 1 = 1 ∨ p 1 = 2 ∨ p 1 = 5 ∨ p 1 = 6) ∧
  (p 2 = 1 ∨ p 2 = 2 ∨ p 2 = 5 ∨ p 2 = 6) ∧
  (p 3 = 1 ∨ p 3 = 2 ∨ p 3 = 5 ∨ p 3 = 6) ∧
  (p 0 ≠ p 1) ∧ (p 0 ≠ p 2) ∧ (p 0 ≠ p 3) ∧
  (p 1 ≠ p 2) ∧ (p 1 ≠ p 3) ∧ (p 2 ≠ p 3)

def toInteger (p : FourDigitPermutation) : ℕ :=
  1000 * (p 0).val + 100 * (p 1).val + 10 * (p 2).val + (p 3).val

def isOrdered (p q : FourDigitPermutation) : Prop :=
  toInteger p ≤ toInteger q

theorem eighteenth_permutation :
  ∃ (perms : List FourDigitPermutation),
    (∀ p ∈ perms, isValidPermutation p) ∧
    (perms.length = 24) ∧
    (∀ i j, i < j → isOrdered (perms.get ⟨i, by sorry⟩) (perms.get ⟨j, by sorry⟩)) ∧
    (toInteger (perms.get ⟨17, by sorry⟩) = 5621) :=
  sorry

end NUMINAMATH_CALUDE_eighteenth_permutation_l2859_285909


namespace NUMINAMATH_CALUDE_cubic_identity_l2859_285999

theorem cubic_identity (y : ℝ) (h : y^3 + 1/y^3 = 110) : y + 1/y = 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_identity_l2859_285999


namespace NUMINAMATH_CALUDE_quadratic_roots_divisibility_l2859_285953

theorem quadratic_roots_divisibility
  (a b : ℤ) (u v : ℂ) (h1 : u^2 + a*u + b = 0)
  (h2 : v^2 + a*v + b = 0) (h3 : ∃ k : ℤ, a^2 = k * b) :
  ∀ n : ℕ, ∃ m : ℤ, u^(2*n) + v^(2*n) = m * b^n :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_divisibility_l2859_285953


namespace NUMINAMATH_CALUDE_divisibility_by_ten_l2859_285908

theorem divisibility_by_ten (x y : Nat) : 
  x < 10 → y < 10 → x + y = 2 → (65300 + 10 * x + y) % 10 = 0 → x = 2 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_ten_l2859_285908


namespace NUMINAMATH_CALUDE_complex_power_four_l2859_285903

theorem complex_power_four (i : ℂ) (h : i^2 = -1) : (2 + i)^4 = -7 + 24*i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_four_l2859_285903


namespace NUMINAMATH_CALUDE_enclosing_polygons_sides_l2859_285927

theorem enclosing_polygons_sides (m : ℕ) (n : ℕ) : 
  m = 12 →
  (360 / m : ℚ) / 2 = 360 / n →
  n = 24 := by
  sorry

end NUMINAMATH_CALUDE_enclosing_polygons_sides_l2859_285927


namespace NUMINAMATH_CALUDE_puppies_given_away_l2859_285985

def initial_puppies : ℕ := 12
def current_puppies : ℕ := 5

theorem puppies_given_away : initial_puppies - current_puppies = 7 := by
  sorry

end NUMINAMATH_CALUDE_puppies_given_away_l2859_285985


namespace NUMINAMATH_CALUDE_cubic_local_min_implies_a_range_l2859_285979

/-- A function f has a local minimum in the interval (1, 2) -/
def has_local_min_in_interval (f : ℝ → ℝ) : Prop :=
  ∃ x, 1 < x ∧ x < 2 ∧ ∀ y, 1 < y ∧ y < 2 → f x ≤ f y

/-- The cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x + a

theorem cubic_local_min_implies_a_range :
  ∀ a : ℝ, has_local_min_in_interval (f a) → 1 < a ∧ a < 4 :=
sorry

end NUMINAMATH_CALUDE_cubic_local_min_implies_a_range_l2859_285979


namespace NUMINAMATH_CALUDE_erased_odd_number_l2859_285951

theorem erased_odd_number (n : ℕ) (erased : ℕ) :
  (∃ k, n = k^2) ∧
  (∃ m, erased = 2*m - 1) ∧
  (n^2 - erased = 2008) →
  erased = 17 := by
sorry

end NUMINAMATH_CALUDE_erased_odd_number_l2859_285951


namespace NUMINAMATH_CALUDE_fraction_of_male_birds_l2859_285960

theorem fraction_of_male_birds (T : ℚ) (h1 : T > 0) : 
  let robins := (2 / 5 : ℚ) * T
  let bluejays := T - robins
  let female_robins := (1 / 3 : ℚ) * robins
  let female_bluejays := (2 / 3 : ℚ) * bluejays
  let male_birds := T - female_robins - female_bluejays
  male_birds / T = 7 / 15 := by
sorry

end NUMINAMATH_CALUDE_fraction_of_male_birds_l2859_285960


namespace NUMINAMATH_CALUDE_airline_problem_l2859_285943

/-- The number of airplanes owned by an airline company -/
def num_airplanes (rows_per_plane : ℕ) (seats_per_row : ℕ) (flights_per_day : ℕ) (total_passengers : ℕ) : ℕ :=
  total_passengers / (rows_per_plane * seats_per_row * flights_per_day)

/-- Theorem stating the number of airplanes owned by the company -/
theorem airline_problem :
  num_airplanes 20 7 2 1400 = 5 := by
  sorry

#eval num_airplanes 20 7 2 1400

end NUMINAMATH_CALUDE_airline_problem_l2859_285943


namespace NUMINAMATH_CALUDE_series_equation_solutions_l2859_285963

def series_sum (x : ℝ) : ℝ := 1 + 3*x + 7*x^2 + 11*x^3 + 15*x^4 + 19*x^5 + 23*x^6 + 27*x^7 + 31*x^8 + 35*x^9 + 39*x^10

theorem series_equation_solutions :
  ∃ (x₁ x₂ : ℝ), 
    x₁ ≠ x₂ ∧
    -1 < x₁ ∧ x₁ < 1 ∧
    -1 < x₂ ∧ x₂ < 1 ∧
    series_sum x₁ = 50 ∧
    series_sum x₂ = 50 ∧
    abs (x₁ - 0.959) < 0.001 ∧
    abs (x₂ - 0.021) < 0.001 :=
by sorry

end NUMINAMATH_CALUDE_series_equation_solutions_l2859_285963


namespace NUMINAMATH_CALUDE_min_value_quadratic_min_value_achieved_l2859_285928

theorem min_value_quadratic (x : ℝ) : x^2 - 6*x + 10 ≥ 1 := by sorry

theorem min_value_achieved : ∃ x : ℝ, x^2 - 6*x + 10 = 1 := by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_min_value_achieved_l2859_285928


namespace NUMINAMATH_CALUDE_no_solution_iff_m_equals_six_l2859_285975

theorem no_solution_iff_m_equals_six (m : ℝ) : 
  (∀ x : ℝ, (2 * x + m) / (x + 3) ≠ 1) ↔ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_equals_six_l2859_285975


namespace NUMINAMATH_CALUDE_part_one_part_two_l2859_285946

noncomputable section

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hA : 0 < A ∧ A < π)
  (hB : 0 < B ∧ B < π)
  (hC : 0 < C ∧ C < π)
  (hsum : A + B + C = π)
  (law_of_sines : a / Real.sin A = b / Real.sin B)
  (law_of_cosines : c^2 = a^2 + b^2 - 2*a*b*Real.cos C)

-- Define the specific triangle with the given condition
def SpecialTriangle (t : Triangle) : Prop :=
  3 * t.a = 2 * t.b

-- Part I
theorem part_one (t : Triangle) (h : SpecialTriangle t) (hB : t.B = π/3) :
  Real.sin t.C = (Real.sqrt 3 + 3 * Real.sqrt 2) / 6 :=
sorry

-- Part II
theorem part_two (t : Triangle) (h : SpecialTriangle t) (hC : Real.cos t.C = 2/3) :
  Real.sin (t.A - t.B) = -Real.sqrt 5 / 3 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2859_285946


namespace NUMINAMATH_CALUDE_train_speed_l2859_285914

/-- Given a train and platform with the following properties:
  * The train and platform have equal length
  * The train is 750 meters long
  * The train crosses the platform in one minute
  Prove that the speed of the train is 90 km/hr -/
theorem train_speed (train_length : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_length = 750 →
  platform_length = train_length →
  crossing_time = 1 →
  (train_length + platform_length) / crossing_time * 60 / 1000 = 90 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2859_285914


namespace NUMINAMATH_CALUDE_smallest_angle_for_complete_circle_sufficient_angle_for_complete_circle_exact_smallest_angle_for_complete_circle_l2859_285904

/-- The smallest angle needed to plot the entire circle for r = sin θ -/
theorem smallest_angle_for_complete_circle : 
  ∀ t : ℝ, (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t → ∃ r : ℝ, r = Real.sin θ ∧ 
    (∃ x y : ℝ, x = r * Real.cos θ ∧ y = r * Real.sin θ)) →
  (∀ x y : ℝ, x^2 + y^2 ≤ 1 → 
    ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ t ∧ x = (Real.sin θ) * (Real.cos θ) ∧ y = (Real.sin θ) * (Real.sin θ)) →
  t ≥ 3 * π / 2 :=
by sorry

/-- 3π/2 is sufficient to plot the entire circle for r = sin θ -/
theorem sufficient_angle_for_complete_circle :
  ∀ x y : ℝ, x^2 + y^2 ≤ 1 → 
    ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 3 * π / 2 ∧ x = (Real.sin θ) * (Real.cos θ) ∧ y = (Real.sin θ) * (Real.sin θ) :=
by sorry

/-- The smallest angle needed to plot the entire circle for r = sin θ is exactly 3π/2 -/
theorem exact_smallest_angle_for_complete_circle :
  (∀ t : ℝ, t < 3 * π / 2 → 
    ∃ x y : ℝ, x^2 + y^2 ≤ 1 ∧ 
      ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t → x ≠ (Real.sin θ) * (Real.cos θ) ∨ y ≠ (Real.sin θ) * (Real.sin θ)) ∧
  (∀ x y : ℝ, x^2 + y^2 ≤ 1 → 
    ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 3 * π / 2 ∧ x = (Real.sin θ) * (Real.cos θ) ∧ y = (Real.sin θ) * (Real.sin θ)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_for_complete_circle_sufficient_angle_for_complete_circle_exact_smallest_angle_for_complete_circle_l2859_285904


namespace NUMINAMATH_CALUDE_existence_of_index_l2859_285996

theorem existence_of_index (n : ℕ) (x : ℕ → ℝ) 
  (h_n : n ≥ 2)
  (h_x : ∀ i, i ≤ n → 0 ≤ x i ∧ x i ≤ 1) :
  ∃ i, 1 ≤ i ∧ i ≤ n - 1 ∧ x i * (1 - x (i + 1)) ≥ (1/4) * x 1 * (1 - x n) := by
sorry

end NUMINAMATH_CALUDE_existence_of_index_l2859_285996


namespace NUMINAMATH_CALUDE_base4_division_theorem_l2859_285961

/-- Converts a number from base 4 to base 10 -/
def base4ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 4 -/
def base10ToBase4 (n : ℕ) : ℕ := sorry

theorem base4_division_theorem :
  let dividend := 2313
  let divisor := 13
  let quotient := 122
  base10ToBase4 (base4ToBase10 dividend / base4ToBase10 divisor) = quotient := by
  sorry

end NUMINAMATH_CALUDE_base4_division_theorem_l2859_285961


namespace NUMINAMATH_CALUDE_total_length_of_T_l2859_285989

-- Define the set T
def T : Set (ℝ × ℝ) := {p : ℝ × ℝ | ‖‖|p.1| - 3‖ - 2‖ + ‖‖|p.2| - 3‖ - 2‖ = 2}

-- Define the total length of lines in T
def total_length (S : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem total_length_of_T : total_length T = 128 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_total_length_of_T_l2859_285989


namespace NUMINAMATH_CALUDE_valid_numbers_l2859_285918

def is_valid_number (n : ℕ) : Prop :=
  ∃ (k : ℕ) (a : Fin 10) (m n : ℕ),
    n = m + 10^k * a.val + 10^(k+1) * n ∧
    m < 10^k ∧
    m + 10^k * n = (m + 10^k * a.val + 10^(k+1) * n) / 6 ∧
    n % 10 ≠ 0

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {12, 24, 36, 48, 108} :=
sorry

end NUMINAMATH_CALUDE_valid_numbers_l2859_285918


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l2859_285968

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 - 8 * y - 6 = -2 * y^2 - 8 * x

-- Define the center and radius
def is_center_radius (c d s : ℝ) : Prop :=
  ∀ (x y : ℝ), circle_equation x y ↔ (x - c)^2 + (y - d)^2 = s^2

-- Theorem statement
theorem circle_center_radius_sum :
  ∃ (c d s : ℝ), is_center_radius c d s ∧ c + d + s = Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l2859_285968


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2859_285932

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - 4*x + 3
  (f 1 = 0 ∧ f 3 = 0) ∧ 
  (∀ x : ℝ, f x = 0 → x = 1 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2859_285932


namespace NUMINAMATH_CALUDE_registration_methods_count_l2859_285901

theorem registration_methods_count :
  let num_students : ℕ := 4
  let num_activities : ℕ := 3
  let students_choose_one (s : ℕ) (a : ℕ) : ℕ := a^s
  students_choose_one num_students num_activities = 81 := by
  sorry

end NUMINAMATH_CALUDE_registration_methods_count_l2859_285901


namespace NUMINAMATH_CALUDE_cubic_factorization_l2859_285954

theorem cubic_factorization (a : ℝ) : a^3 - 4*a = a*(a+2)*(a-2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2859_285954


namespace NUMINAMATH_CALUDE_max_unique_sums_l2859_285939

def coin_values : List ℕ := [1, 1, 1, 5, 10, 25]

def unique_sums (values : List ℕ) : Finset ℕ :=
  (values.map (λ x => values.map (λ y => x + y))).join.toFinset

theorem max_unique_sums :
  Finset.card (unique_sums coin_values) = 7 := by sorry

end NUMINAMATH_CALUDE_max_unique_sums_l2859_285939


namespace NUMINAMATH_CALUDE_recruitment_probability_one_pass_reinspection_probability_l2859_285942

/-- Probabilities of passing re-inspection for students A, B, and C -/
def p_reinspect_A : ℝ := 0.5
def p_reinspect_B : ℝ := 0.6
def p_reinspect_C : ℝ := 0.75

/-- Probabilities of passing cultural examination for students A, B, and C -/
def p_cultural_A : ℝ := 0.6
def p_cultural_B : ℝ := 0.5
def p_cultural_C : ℝ := 0.4

/-- All students pass political review -/
def p_political : ℝ := 1

/-- Assumption: Outcomes of the last three stages are independent -/
axiom independence : True

theorem recruitment_probability :
  p_reinspect_A * p_cultural_A * p_political = 0.3 :=
sorry

theorem one_pass_reinspection_probability :
  p_reinspect_A * (1 - p_reinspect_B) * (1 - p_reinspect_C) +
  (1 - p_reinspect_A) * p_reinspect_B * (1 - p_reinspect_C) +
  (1 - p_reinspect_A) * (1 - p_reinspect_B) * p_reinspect_C = 0.275 :=
sorry

end NUMINAMATH_CALUDE_recruitment_probability_one_pass_reinspection_probability_l2859_285942


namespace NUMINAMATH_CALUDE_iron_conductivity_is_deductive_l2859_285916

-- Define the universe of discourse
variable (U : Type)

-- Define the predicates
variable (Metal : U → Prop)
variable (ConductsElectricity : U → Prop)

-- Define iron as a constant in our universe
variable (iron : U)

-- State the premises and conclusion
variable (all_metals_conduct : ∀ x, Metal x → ConductsElectricity x)
variable (iron_is_metal : Metal iron)
variable (iron_conducts : ConductsElectricity iron)

-- Define deductive reasoning
def is_deductive_reasoning (premise1 premise2 conclusion : Prop) : Prop :=
  (premise1 ∧ premise2) → conclusion

-- Theorem stating that the given reasoning is deductive
theorem iron_conductivity_is_deductive :
  is_deductive_reasoning 
    (∀ x, Metal x → ConductsElectricity x)
    (Metal iron)
    (ConductsElectricity iron) :=
by sorry

end NUMINAMATH_CALUDE_iron_conductivity_is_deductive_l2859_285916


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l2859_285913

theorem divisibility_implies_equality (a b n : ℕ) 
  (h : ∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) : 
  a = b^n := by
sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l2859_285913


namespace NUMINAMATH_CALUDE_total_clothes_washed_l2859_285988

/-- The total number of clothes washed by Cally, Danny, and Emily -/
theorem total_clothes_washed (
  cally_white_shirts cally_colored_shirts cally_shorts cally_pants cally_jackets : ℕ)
  (danny_white_shirts danny_colored_shirts danny_shorts danny_pants danny_jackets : ℕ)
  (emily_white_shirts emily_colored_shirts emily_shorts emily_pants emily_jackets : ℕ)
  (cally_danny_socks emily_danny_socks : ℕ)
  (h1 : cally_white_shirts = 10)
  (h2 : cally_colored_shirts = 5)
  (h3 : cally_shorts = 7)
  (h4 : cally_pants = 6)
  (h5 : cally_jackets = 3)
  (h6 : danny_white_shirts = 6)
  (h7 : danny_colored_shirts = 8)
  (h8 : danny_shorts = 10)
  (h9 : danny_pants = 6)
  (h10 : danny_jackets = 4)
  (h11 : emily_white_shirts = 8)
  (h12 : emily_colored_shirts = 6)
  (h13 : emily_shorts = 9)
  (h14 : emily_pants = 5)
  (h15 : emily_jackets = 2)
  (h16 : cally_danny_socks = 3)
  (h17 : emily_danny_socks = 2) :
  cally_white_shirts + cally_colored_shirts + cally_shorts + cally_pants + cally_jackets +
  danny_white_shirts + danny_colored_shirts + danny_shorts + danny_pants + danny_jackets +
  emily_white_shirts + emily_colored_shirts + emily_shorts + emily_pants + emily_jackets +
  cally_danny_socks + emily_danny_socks = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_clothes_washed_l2859_285988


namespace NUMINAMATH_CALUDE_angle_sum_in_circle_l2859_285930

theorem angle_sum_in_circle (x : ℝ) : 
  3 * x + 6 * x + 2 * x + x = 360 → x = 30 := by
  sorry

#check angle_sum_in_circle

end NUMINAMATH_CALUDE_angle_sum_in_circle_l2859_285930


namespace NUMINAMATH_CALUDE_original_number_l2859_285906

theorem original_number : ∃ x : ℝ, 3 * (2 * x + 9) = 51 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l2859_285906


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l2859_285950

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 2*x - 3 = 0

-- Theorem statement
theorem point_in_second_quadrant (m n : ℝ) 
  (hm : quadratic_eq m) (hn : quadratic_eq n) (hlt : m < n) : 
  m < 0 ∧ n > 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l2859_285950


namespace NUMINAMATH_CALUDE_percentage_comparison_l2859_285940

theorem percentage_comparison (w x y z t : ℝ) 
  (hw : w = 0.6 * x)
  (hx : x = 0.6 * y)
  (hz : z = 0.54 * y)
  (ht : t = 0.48 * x) :
  (z - w) / w * 100 = 50 ∧ (w - t) / w * 100 = 20 :=
by sorry

end NUMINAMATH_CALUDE_percentage_comparison_l2859_285940


namespace NUMINAMATH_CALUDE_common_root_is_one_l2859_285924

/-- Given two quadratic equations with coefficients a and b that have exactly one common root, prove that this root is 1 -/
theorem common_root_is_one (a b : ℝ) 
  (h : ∃! x : ℝ, (x^2 + a*x + b = 0) ∧ (x^2 + b*x + a = 0)) : 
  ∃ x : ℝ, (x^2 + a*x + b = 0) ∧ (x^2 + b*x + a = 0) ∧ x = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_common_root_is_one_l2859_285924


namespace NUMINAMATH_CALUDE_parallel_condition_neither_sufficient_nor_necessary_l2859_285962

-- Define the types for lines and planes
variable (Line Plane : Type*)

-- Define the subset relation
variable (subset : Line → Plane → Prop)

-- Define the parallel relation for lines
variable (parallel_lines : Line → Line → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_condition_neither_sufficient_nor_necessary
  (l m : Line) (α : Plane) (h : subset m α) :
  ¬(∀ l m α, subset m α → (parallel_lines l m → parallel_line_plane l α)) ∧
  ¬(∀ l m α, subset m α → (parallel_line_plane l α → parallel_lines l m)) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_neither_sufficient_nor_necessary_l2859_285962


namespace NUMINAMATH_CALUDE_wine_equation_correctness_l2859_285990

/-- Represents the wine consumption and intoxication scenario --/
def wine_scenario (x y : ℚ) : Prop :=
  -- Total bottles of wine
  x + y = 19 ∧
  -- Intoxication effect
  3 * x + (1/3) * y = 33 ∧
  -- x represents good wine bottles
  x ≥ 0 ∧
  -- y represents inferior wine bottles
  y ≥ 0

/-- The system of equations correctly represents the wine scenario --/
theorem wine_equation_correctness :
  ∃ x y : ℚ, wine_scenario x y :=
sorry

end NUMINAMATH_CALUDE_wine_equation_correctness_l2859_285990


namespace NUMINAMATH_CALUDE_max_perimeter_of_rectangle_with_area_36_exists_rectangle_with_area_36_and_perimeter_74_l2859_285937

-- Define a rectangle with integer side lengths
structure Rectangle where
  length : ℕ
  width : ℕ

-- Define the area of a rectangle
def area (r : Rectangle) : ℕ := r.length * r.width

-- Define the perimeter of a rectangle
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

-- Theorem: The maximum perimeter of a rectangle with integer side lengths and area 36 is 74
theorem max_perimeter_of_rectangle_with_area_36 :
  ∀ r : Rectangle, area r = 36 → perimeter r ≤ 74 :=
by
  sorry

-- Theorem: There exists a rectangle with integer side lengths, area 36, and perimeter 74
theorem exists_rectangle_with_area_36_and_perimeter_74 :
  ∃ r : Rectangle, area r = 36 ∧ perimeter r = 74 :=
by
  sorry

end NUMINAMATH_CALUDE_max_perimeter_of_rectangle_with_area_36_exists_rectangle_with_area_36_and_perimeter_74_l2859_285937


namespace NUMINAMATH_CALUDE_intersection_M_N_l2859_285995

def M : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 5}
def N : Set ℝ := {x : ℝ | -5 < x ∧ x < 5}

theorem intersection_M_N : M ∩ N = {x : ℝ | -3 < x ∧ x < 5} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2859_285995


namespace NUMINAMATH_CALUDE_coffee_maker_capacity_l2859_285987

/-- Represents a cylindrical coffee maker -/
structure CoffeeMaker :=
  (capacity : ℝ)

/-- The coffee maker contains 45 cups when it is 36% full -/
def partially_filled (cm : CoffeeMaker) : Prop :=
  0.36 * cm.capacity = 45

/-- Theorem: A cylindrical coffee maker that contains 45 cups when 36% full has a capacity of 125 cups -/
theorem coffee_maker_capacity (cm : CoffeeMaker) 
  (h : partially_filled cm) : cm.capacity = 125 := by
  sorry

end NUMINAMATH_CALUDE_coffee_maker_capacity_l2859_285987


namespace NUMINAMATH_CALUDE_volcano_count_l2859_285955

theorem volcano_count (total : ℕ) (intact : ℕ) : 
  (intact : ℝ) = total * (1 - 0.2) * (1 - 0.4) * (1 - 0.5) ∧ intact = 48 → 
  total = 200 := by
sorry

end NUMINAMATH_CALUDE_volcano_count_l2859_285955


namespace NUMINAMATH_CALUDE_inscribed_square_area_l2859_285992

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 8 = 1

-- Define a square inscribed in the ellipse
def inscribed_square (s : ℝ) : Prop :=
  ∃ (x y : ℝ), ellipse x y ∧ s = 2 * x ∧ s = 2 * y

-- Theorem statement
theorem inscribed_square_area :
  ∃ (s : ℝ), inscribed_square s ∧ s^2 = 32/3 :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l2859_285992


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l2859_285981

theorem min_sum_of_squares (x y : ℝ) (h : (x + 3) * (y - 3) = 0) :
  ∃ (m : ℝ), (∀ a b : ℝ, (a + 3) * (b - 3) = 0 → x^2 + y^2 ≤ a^2 + b^2) ∧ m = 18 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l2859_285981


namespace NUMINAMATH_CALUDE_flower_seedling_problem_l2859_285958

/-- Represents the unit price of flower seedlings --/
structure FlowerPrice where
  typeA : ℝ
  typeB : ℝ

/-- Represents the cost function for purchasing flower seedlings --/
def cost_function (p : FlowerPrice) (a : ℝ) : ℝ :=
  p.typeA * (12 - a) + (p.typeB - a) * a

/-- The theorem statement for the flower seedling problem --/
theorem flower_seedling_problem (p : FlowerPrice) :
  (3 * p.typeA + 5 * p.typeB = 210) →
  (4 * p.typeA + 10 * p.typeB = 380) →
  p.typeA = 20 ∧ p.typeB = 30 ∧
  ∃ (a_min a_max : ℝ), 0 < a_min ∧ a_min < 12 ∧ 0 < a_max ∧ a_max < 12 ∧
    ∀ (a : ℝ), 0 < a ∧ a < 12 →
      229 ≤ cost_function p a ∧ cost_function p a ≤ 265 ∧
      cost_function p a_min = 229 ∧ cost_function p a_max = 265 := by
  sorry

end NUMINAMATH_CALUDE_flower_seedling_problem_l2859_285958


namespace NUMINAMATH_CALUDE_apple_pear_worth_l2859_285917

-- Define the worth of apples in terms of pears
def apple_worth (x : ℚ) : Prop := (3/4 : ℚ) * 16 * x = 6

-- Theorem to prove
theorem apple_pear_worth (x : ℚ) (h : apple_worth x) : (1/3 : ℚ) * 9 * x = (3/2 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_apple_pear_worth_l2859_285917


namespace NUMINAMATH_CALUDE_watch_cost_price_l2859_285964

theorem watch_cost_price (loss_percentage : ℝ) (gain_percentage : ℝ) (price_difference : ℝ) : 
  loss_percentage = 0.15 →
  gain_percentage = 0.10 →
  price_difference = 450 →
  ∃ (cost_price : ℝ),
    cost_price * (1 - loss_percentage) + price_difference = cost_price * (1 + gain_percentage) ∧
    cost_price = 1800 :=
by sorry

end NUMINAMATH_CALUDE_watch_cost_price_l2859_285964


namespace NUMINAMATH_CALUDE_promotion_savings_l2859_285973

/-- The price of a single pair of shoes -/
def shoe_price : ℕ := 40

/-- The discount amount for Promotion B -/
def promotion_b_discount : ℕ := 15

/-- Calculate the total cost using Promotion A -/
def cost_promotion_a (price : ℕ) : ℕ :=
  price + price / 2

/-- Calculate the total cost using Promotion B -/
def cost_promotion_b (price : ℕ) (discount : ℕ) : ℕ :=
  price + (price - discount)

/-- Theorem: The difference in cost between Promotion B and Promotion A is $5 -/
theorem promotion_savings : 
  cost_promotion_b shoe_price promotion_b_discount - cost_promotion_a shoe_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_promotion_savings_l2859_285973


namespace NUMINAMATH_CALUDE_find_m_value_l2859_285922

theorem find_m_value : ∃ m : ℝ, 
  (∀ x : ℝ, (m - 1) * x^2 + 5 * x + m^2 - 3 * m + 2 = 0) ∧ 
  (m^2 - 3 * m + 2 = 0) ∧ 
  (m - 1 ≠ 0) ∧ 
  (m = 2) := by
sorry

end NUMINAMATH_CALUDE_find_m_value_l2859_285922


namespace NUMINAMATH_CALUDE_decimal_49_to_binary_l2859_285998

/-- The binary representation of a natural number -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Convert a list of booleans to a natural number in binary representation -/
def fromBinary (l : List Bool) : ℕ :=
  l.foldr (fun b n => 2 * n + if b then 1 else 0) 0

theorem decimal_49_to_binary :
  toBinary 49 = [true, true, false, false, false, true] :=
by sorry

end NUMINAMATH_CALUDE_decimal_49_to_binary_l2859_285998


namespace NUMINAMATH_CALUDE_b_join_time_correct_l2859_285936

/-- Represents the number of months in a year -/
def monthsInYear : ℕ := 12

/-- A's initial investment in Rupees -/
def aInvestment : ℕ := 36000

/-- B's initial investment in Rupees -/
def bInvestment : ℕ := 54000

/-- Profit sharing ratio of A to B -/
def profitRatio : ℚ := 2 / 1

/-- Calculates the time B joined the business in months -/
def bJoinTime : ℕ := monthsInYear - 8

theorem b_join_time_correct :
  (aInvestment * monthsInYear : ℚ) / (bInvestment * bJoinTime) = profitRatio :=
sorry

end NUMINAMATH_CALUDE_b_join_time_correct_l2859_285936


namespace NUMINAMATH_CALUDE_green_bows_count_l2859_285969

theorem green_bows_count (total : ℕ) (white : ℕ) :
  white = 40 →
  (1 : ℚ) / 4 + (1 : ℚ) / 3 + (1 : ℚ) / 6 + (white : ℚ) / total = 1 →
  (1 : ℚ) / 6 * total = 27 :=
by sorry

end NUMINAMATH_CALUDE_green_bows_count_l2859_285969


namespace NUMINAMATH_CALUDE_like_terms_exponent_l2859_285926

theorem like_terms_exponent (a b : ℕ) : 
  (∀ x y : ℝ, ∃ k : ℝ, 2 * x^a * y^3 = k * (-x^2 * y^b)) → a^b = 8 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_l2859_285926


namespace NUMINAMATH_CALUDE_soda_cans_with_tax_l2859_285983

/-- Given:
  S : number of cans bought for Q quarters
  Q : number of quarters for S cans
  t : tax rate as a fraction of 1
  D : number of dollars available
-/
theorem soda_cans_with_tax (S Q : ℕ) (t : ℚ) (D : ℕ) :
  let cans_purchasable := (4 * D * S * (1 + t)) / Q
  cans_purchasable = (4 * D * S * (1 + t)) / Q :=
by sorry

end NUMINAMATH_CALUDE_soda_cans_with_tax_l2859_285983


namespace NUMINAMATH_CALUDE_yellow_marble_fraction_l2859_285976

theorem yellow_marble_fraction (n : ℝ) (h : n > 0) : 
  let initial_green := (2/3) * n
  let initial_yellow := n - initial_green
  let new_yellow := 3 * initial_yellow
  let new_total := initial_green + new_yellow
  new_yellow / new_total = 3/5 := by sorry

end NUMINAMATH_CALUDE_yellow_marble_fraction_l2859_285976


namespace NUMINAMATH_CALUDE_birdseed_mix_problem_l2859_285941

/-- Proves that Brand A contains 60% sunflower given the conditions of the birdseed mix problem -/
theorem birdseed_mix_problem (brand_a_millet : ℝ) (brand_b_millet : ℝ) (brand_b_safflower : ℝ)
  (mix_millet : ℝ) (mix_brand_a : ℝ) :
  brand_a_millet = 0.4 →
  brand_b_millet = 0.65 →
  brand_b_safflower = 0.35 →
  mix_millet = 0.5 →
  mix_brand_a = 0.6 →
  ∃ (brand_a_sunflower : ℝ),
    brand_a_sunflower = 0.6 ∧
    brand_a_millet + brand_a_sunflower = 1 ∧
    mix_brand_a * brand_a_millet + (1 - mix_brand_a) * brand_b_millet = mix_millet :=
by sorry

end NUMINAMATH_CALUDE_birdseed_mix_problem_l2859_285941


namespace NUMINAMATH_CALUDE_wendy_lost_lives_l2859_285959

/-- Represents the number of lives in Wendy's video game scenario -/
structure GameLives where
  initial : ℕ
  gained : ℕ
  final : ℕ
  lost : ℕ

/-- Theorem stating that Wendy lost 6 lives given the initial conditions -/
theorem wendy_lost_lives (game : GameLives) 
  (h1 : game.initial = 10)
  (h2 : game.gained = 37)
  (h3 : game.final = 41)
  (h4 : game.final = game.initial - game.lost + game.gained) :
  game.lost = 6 := by
  sorry

end NUMINAMATH_CALUDE_wendy_lost_lives_l2859_285959


namespace NUMINAMATH_CALUDE_fields_medal_stats_l2859_285915

def data_set : List ℕ := [29, 32, 33, 35, 35, 40]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem fields_medal_stats : 
  mode data_set = 35 ∧ median data_set = 34 := by sorry

end NUMINAMATH_CALUDE_fields_medal_stats_l2859_285915


namespace NUMINAMATH_CALUDE_complement_of_A_l2859_285935

def U : Set ℝ := Set.univ

def A : Set ℝ := {x : ℝ | 1 / x ≥ 1}

theorem complement_of_A : 
  Set.compl A = {x : ℝ | x ≤ 0 ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2859_285935


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2859_285978

theorem inequality_solution_range (m : ℝ) :
  (∃ x : ℝ, |x + 2| - |x + 3| > m) → m < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2859_285978


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2859_285984

/-- Two vectors are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (6, 2) (x, 3) → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2859_285984


namespace NUMINAMATH_CALUDE_divisible_by_15_20_25_between_1000_and_2000_l2859_285956

theorem divisible_by_15_20_25_between_1000_and_2000 : 
  ∃! n : ℕ, (∀ k : ℕ, 1000 < k ∧ k < 2000 ∧ 15 ∣ k ∧ 20 ∣ k ∧ 25 ∣ k → k ∈ Finset.range n) ∧ 
  (∀ k : ℕ, k ∈ Finset.range n → 1000 < k ∧ k < 2000 ∧ 15 ∣ k ∧ 20 ∣ k ∧ 25 ∣ k) :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_15_20_25_between_1000_and_2000_l2859_285956


namespace NUMINAMATH_CALUDE_sqrt_36_equals_6_l2859_285966

theorem sqrt_36_equals_6 : Real.sqrt 36 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_36_equals_6_l2859_285966


namespace NUMINAMATH_CALUDE_tshirt_cost_l2859_285902

theorem tshirt_cost (initial_amount : ℝ) (sweater_cost : ℝ) (shoes_cost : ℝ) 
                    (refund_percentage : ℝ) (final_amount : ℝ) :
  initial_amount = 74 →
  sweater_cost = 9 →
  shoes_cost = 30 →
  refund_percentage = 0.9 →
  final_amount = 51 →
  ∃ (tshirt_cost : ℝ),
    tshirt_cost = 14 ∧
    final_amount = initial_amount - sweater_cost - tshirt_cost - shoes_cost + refund_percentage * shoes_cost :=
by sorry

end NUMINAMATH_CALUDE_tshirt_cost_l2859_285902


namespace NUMINAMATH_CALUDE_binomial_coefficients_10_l2859_285977

theorem binomial_coefficients_10 : (Nat.choose 10 10 = 1) ∧ (Nat.choose 10 9 = 10) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficients_10_l2859_285977


namespace NUMINAMATH_CALUDE_heather_counts_209_l2859_285948

-- Define the range of numbers
def range : Set ℕ := {n | 1 ≤ n ∧ n ≤ 500}

-- Define Alice's skipping pattern
def aliceSkips (n : ℕ) : Prop := ∃ k, n = 5 * k - 2 ∧ 1 ≤ k ∧ k ≤ 100

-- Define the general skipping pattern for Barbara and the next 5 students
def otherSkips (n : ℕ) : Prop := ∃ m, n = 3 * m - 1 ∧ ¬(aliceSkips n)

-- Define Heather's number
def heatherNumber : ℕ := 209

-- Theorem statement
theorem heather_counts_209 :
  heatherNumber ∈ range ∧
  ¬(aliceSkips heatherNumber) ∧
  ¬(otherSkips heatherNumber) ∧
  ∀ n ∈ range, n ≠ heatherNumber → aliceSkips n ∨ otherSkips n :=
sorry

end NUMINAMATH_CALUDE_heather_counts_209_l2859_285948


namespace NUMINAMATH_CALUDE_two_digit_product_sum_l2859_285974

theorem two_digit_product_sum (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 8060 → 
  a + b = 127 := by
sorry

end NUMINAMATH_CALUDE_two_digit_product_sum_l2859_285974


namespace NUMINAMATH_CALUDE_book_input_time_l2859_285944

/-- The original time to complete a book input task given certain conditions on computer count and time changes. -/
theorem book_input_time : ∃ (n : ℕ) (T : ℚ),
  T > 0 ∧
  n > 3 ∧
  (n : ℚ) * T = (n + 3 : ℚ) * (3/4 * T) ∧
  (n - 3 : ℚ) * (T + 5/6) = (n : ℚ) * T ∧
  T = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_book_input_time_l2859_285944
