import Mathlib

namespace NUMINAMATH_CALUDE_binomial_expansion_with_arithmetic_sequence_coefficients_l3000_300048

/-- 
Given a binomial expansion (a+b)^n where the coefficients of the first three terms 
form an arithmetic sequence, this theorem proves that n = 8 and identifies 
the rational terms in the expansion when a = x and b = 1/2.
-/
theorem binomial_expansion_with_arithmetic_sequence_coefficients :
  ∀ n : ℕ,
  (∃ d : ℚ, (n.choose 1 : ℚ) = (n.choose 0 : ℚ) + d ∧ (n.choose 2 : ℚ) = (n.choose 1 : ℚ) + d) →
  (n = 8 ∧ 
   ∀ r : ℕ, r ≤ n → 
   (r = 0 ∨ r = 4 ∨ r = 8) ↔ ∃ q : ℚ, (n.choose r : ℚ) * (1 / 2 : ℚ)^r = q) :=
by sorry


end NUMINAMATH_CALUDE_binomial_expansion_with_arithmetic_sequence_coefficients_l3000_300048


namespace NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l3000_300059

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (((x^2 + y^2 + z^2) * (4*x^2 + 2*y^2 + 3*z^2)).sqrt) / (x*y*z) ≥ 2 + Real.sqrt 2 + Real.sqrt 3 :=
sorry

theorem lower_bound_achievable :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  (((x^2 + y^2 + z^2) * (4*x^2 + 2*y^2 + 3*z^2)).sqrt) / (x*y*z) = 2 + Real.sqrt 2 + Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l3000_300059


namespace NUMINAMATH_CALUDE_pauls_crayons_l3000_300002

theorem pauls_crayons (erasers_birthday : ℕ) (crayons_left : ℕ) (crayons_difference : ℕ) :
  erasers_birthday = 38 →
  crayons_left = 391 →
  crayons_difference = 353 →
  crayons_left = erasers_birthday + crayons_difference →
  crayons_left = 391 :=
by sorry

end NUMINAMATH_CALUDE_pauls_crayons_l3000_300002


namespace NUMINAMATH_CALUDE_ratio_equality_l3000_300093

theorem ratio_equality : 
  ∀ (a b c d x : ℚ), 
    a = 3 / 5 → 
    b = 6 / 7 → 
    c = 2 / 3 → 
    d = 7 / 15 → 
    (a / b = d / c) → 
    x = d := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3000_300093


namespace NUMINAMATH_CALUDE_no_natural_solution_for_x2_plus_y2_eq_7z2_l3000_300008

theorem no_natural_solution_for_x2_plus_y2_eq_7z2 :
  ¬ ∃ (x y z : ℕ), x^2 + y^2 = 7 * z^2 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_for_x2_plus_y2_eq_7z2_l3000_300008


namespace NUMINAMATH_CALUDE_thyme_leaves_theorem_l3000_300037

/-- The number of leaves per thyme plant -/
def thyme_leaves_per_plant : ℕ :=
  let basil_pots : ℕ := 3
  let rosemary_pots : ℕ := 9
  let thyme_pots : ℕ := 6
  let basil_leaves_per_pot : ℕ := 4
  let rosemary_leaves_per_pot : ℕ := 18
  let total_leaves : ℕ := 354
  let basil_leaves : ℕ := basil_pots * basil_leaves_per_pot
  let rosemary_leaves : ℕ := rosemary_pots * rosemary_leaves_per_pot
  let thyme_leaves : ℕ := total_leaves - basil_leaves - rosemary_leaves
  thyme_leaves / thyme_pots

theorem thyme_leaves_theorem : thyme_leaves_per_plant = 30 := by
  sorry

end NUMINAMATH_CALUDE_thyme_leaves_theorem_l3000_300037


namespace NUMINAMATH_CALUDE_train_count_l3000_300034

theorem train_count (carriages_per_train : ℕ) (rows_per_carriage : ℕ) (wheels_per_row : ℕ) (total_wheels : ℕ) :
  carriages_per_train = 4 →
  rows_per_carriage = 3 →
  wheels_per_row = 5 →
  total_wheels = 240 →
  total_wheels / (carriages_per_train * rows_per_carriage * wheels_per_row) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_train_count_l3000_300034


namespace NUMINAMATH_CALUDE_line_point_a_value_l3000_300006

/-- Given a line y = 0.75x + 1 and points (4, b), (a, 5), and (a, b + 1) on this line, prove that a = 16/3 -/
theorem line_point_a_value (b : ℝ) :
  (∃ (a : ℝ), (4 : ℝ) * (3/4) + 1 = b ∧ 
              a * (3/4) + 1 = 5 ∧ 
              a * (3/4) + 1 = b + 1) →
  ∃ (a : ℝ), a = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_line_point_a_value_l3000_300006


namespace NUMINAMATH_CALUDE_least_common_multiple_345667_l3000_300080

theorem least_common_multiple_345667 :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (3 ∣ m) ∧ (4 ∣ m) ∧ (5 ∣ m) ∧ (6 ∣ m) ∧ (7 ∣ m) → n ≤ m) ∧
  (3 ∣ n) ∧ (4 ∣ n) ∧ (5 ∣ n) ∧ (6 ∣ n) ∧ (7 ∣ n) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_345667_l3000_300080


namespace NUMINAMATH_CALUDE_cinnamon_tradition_duration_l3000_300030

/-- Represents the cinnamon ball tradition setup -/
structure CinnamonTradition where
  totalSocks : Nat
  extraSocks : Nat
  regularBalls : Nat
  extraBalls : Nat
  totalBalls : Nat

/-- Calculates the maximum number of full days the tradition can continue -/
def maxDays (ct : CinnamonTradition) : Nat :=
  ct.totalBalls / (ct.regularBalls * (ct.totalSocks - ct.extraSocks) + ct.extraBalls * ct.extraSocks)

/-- Theorem stating that for the given conditions, the tradition lasts 3 days -/
theorem cinnamon_tradition_duration :
  ∀ (ct : CinnamonTradition),
  ct.totalSocks = 9 →
  ct.extraSocks = 3 →
  ct.regularBalls = 2 →
  ct.extraBalls = 3 →
  ct.totalBalls = 75 →
  maxDays ct = 3 := by
  sorry

#eval maxDays { totalSocks := 9, extraSocks := 3, regularBalls := 2, extraBalls := 3, totalBalls := 75 }

end NUMINAMATH_CALUDE_cinnamon_tradition_duration_l3000_300030


namespace NUMINAMATH_CALUDE_arithmetic_sequence_increasing_iff_l3000_300086

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: For an arithmetic sequence, a₁ < a₃ if and only if aₙ < aₙ₊₁ for all n -/
theorem arithmetic_sequence_increasing_iff (a : ℕ → ℝ) :
  arithmetic_sequence a → (a 1 < a 3 ↔ ∀ n : ℕ, a n < a (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_increasing_iff_l3000_300086


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3000_300017

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, x > 0 → x^2020 > 0) ∧
  (∃ x, x^2020 > 0 ∧ ¬(x > 0)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3000_300017


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3000_300044

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
   x1^2 - x1 + 2*m - 2 = 0 ∧ 
   x2^2 - x2 + 2*m - 2 = 0) 
  ↔ m ≤ 9/8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3000_300044


namespace NUMINAMATH_CALUDE_village_population_l3000_300005

theorem village_population (P : ℕ) : 
  (P : ℝ) * (1 - 0.05) * (1 - 0.2) = 3553 → P = 4678 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l3000_300005


namespace NUMINAMATH_CALUDE_expression_equals_five_l3000_300079

theorem expression_equals_five :
  (1 - Real.sqrt 5) ^ 0 + |-Real.sqrt 2| - 2 * Real.cos (π / 4) + (1 / 4)⁻¹ = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_five_l3000_300079


namespace NUMINAMATH_CALUDE_rectangle_problem_l3000_300007

/-- Given a rectangle with length 4x + 1 and width x + 7, where the area is equal to twice the perimeter,
    the positive value of x is (-9 + √481) / 8. -/
theorem rectangle_problem (x : ℝ) : 
  (4*x + 1) * (x + 7) = 2 * (2*(4*x + 1) + 2*(x + 7)) → 
  x > 0 → 
  x = (-9 + Real.sqrt 481) / 8 := by
sorry

end NUMINAMATH_CALUDE_rectangle_problem_l3000_300007


namespace NUMINAMATH_CALUDE_five_people_seven_chairs_l3000_300015

/-- The number of ways to arrange n people in r chairs -/
def arrangements (n r : ℕ) : ℕ := sorry

/-- The number of ways to arrange n people in r chairs,
    with one person restricted to m specific chairs -/
def restricted_arrangements (n r m : ℕ) : ℕ := sorry

/-- Theorem: Five people can be arranged in a row of seven chairs in 2160 ways,
    given that the oldest must sit in one of the three chairs at the end of the row -/
theorem five_people_seven_chairs : restricted_arrangements 5 7 3 = 2160 := by sorry

end NUMINAMATH_CALUDE_five_people_seven_chairs_l3000_300015


namespace NUMINAMATH_CALUDE_cookie_recipe_total_cups_l3000_300004

/-- Represents the ratio of ingredients in the recipe -/
structure RecipeRatio where
  butter : ℕ
  flour : ℕ
  sugar : ℕ

/-- Calculates the total cups of ingredients given a ratio and the cups of sugar used -/
def totalCups (ratio : RecipeRatio) (sugarCups : ℕ) : ℕ :=
  let partSize := sugarCups / ratio.sugar
  (ratio.butter + ratio.flour + ratio.sugar) * partSize

/-- Theorem stating that for the given recipe ratio and sugar amount, the total cups is 18 -/
theorem cookie_recipe_total_cups :
  let ratio := RecipeRatio.mk 1 2 3
  let sugarCups := 9
  totalCups ratio sugarCups = 18 := by
  sorry

#check cookie_recipe_total_cups

end NUMINAMATH_CALUDE_cookie_recipe_total_cups_l3000_300004


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l3000_300067

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds_bound : hundreds < 10
  h_tens_bound : tens < 10
  h_ones_bound : ones < 10

/-- The value of a three-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- The reverse of a three-digit number -/
def ThreeDigitNumber.reverse (n : ThreeDigitNumber) : Nat :=
  100 * n.ones + 10 * n.tens + n.hundreds

theorem unique_three_digit_number :
  ∃! n : ThreeDigitNumber,
    (n.hundreds + n.tens + n.ones = 10) ∧
    (n.tens = n.hundreds + n.ones) ∧
    (n.reverse = n.value + 99) ∧
    (n.value = 253) := by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l3000_300067


namespace NUMINAMATH_CALUDE_ab_length_l3000_300046

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
axiom collinear : ∃ (t : ℝ), B = A + t • (D - A) ∧ C = A + t • (D - A)
axiom ab_eq_cd : dist A B = dist C D
axiom bc_length : dist B C = 15
axiom e_not_on_line : ¬∃ (t : ℝ), E = A + t • (D - A)
axiom be_eq_ce : dist B E = dist C E ∧ dist B E = 13

-- Define the perimeter function
def perimeter (X Y Z : ℝ × ℝ) : ℝ := dist X Y + dist Y Z + dist Z X

-- State the theorem
theorem ab_length :
  perimeter A E D = 1.5 * perimeter B E C →
  dist A B = 6.04 := by sorry

end NUMINAMATH_CALUDE_ab_length_l3000_300046


namespace NUMINAMATH_CALUDE_brent_baby_ruths_l3000_300069

/-- The number of Baby Ruths Brent received -/
def baby_ruths : ℕ := sorry

/-- The number of Kit-Kat bars Brent received -/
def kit_kat : ℕ := 5

/-- The number of Hershey kisses Brent received -/
def hershey_kisses : ℕ := 3 * kit_kat

/-- The number of Nerds boxes Brent received -/
def nerds : ℕ := 8

/-- The number of lollipops Brent received -/
def lollipops : ℕ := 11

/-- The number of Reese Peanut butter cups Brent received -/
def reese_cups : ℕ := baby_ruths / 2

/-- The number of lollipops Brent gave to his sister -/
def lollipops_given : ℕ := 5

/-- The total number of candies Brent had left after giving lollipops to his sister -/
def total_left : ℕ := 49

theorem brent_baby_ruths :
  kit_kat + hershey_kisses + nerds + (lollipops - lollipops_given) + baby_ruths + reese_cups = total_left ∧
  baby_ruths = 10 := by sorry

end NUMINAMATH_CALUDE_brent_baby_ruths_l3000_300069


namespace NUMINAMATH_CALUDE_square_perimeter_from_diagonal_l3000_300054

theorem square_perimeter_from_diagonal (d : ℝ) (h : d = 2 * Real.sqrt 2) :
  4 * (d / Real.sqrt 2) = 8 := by sorry

end NUMINAMATH_CALUDE_square_perimeter_from_diagonal_l3000_300054


namespace NUMINAMATH_CALUDE_jim_lamp_purchase_jim_lamp_purchase_correct_l3000_300081

theorem jim_lamp_purchase (lamp_cost : ℕ) (bulb_cost_difference : ℕ) (num_bulbs : ℕ) (total_paid : ℕ) : ℕ :=
  let bulb_cost := lamp_cost - bulb_cost_difference
  let num_lamps := (total_paid - num_bulbs * bulb_cost) / lamp_cost
  num_lamps

#check jim_lamp_purchase 7 4 6 32

theorem jim_lamp_purchase_correct :
  jim_lamp_purchase 7 4 6 32 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jim_lamp_purchase_jim_lamp_purchase_correct_l3000_300081


namespace NUMINAMATH_CALUDE_time_to_save_downpayment_l3000_300010

def salary : ℝ := 150000
def savings_rate : ℝ := 0.10
def house_cost : ℝ := 450000
def downpayment_rate : ℝ := 0.20

def yearly_savings : ℝ := salary * savings_rate
def required_downpayment : ℝ := house_cost * downpayment_rate

theorem time_to_save_downpayment :
  required_downpayment / yearly_savings = 6 := by sorry

end NUMINAMATH_CALUDE_time_to_save_downpayment_l3000_300010


namespace NUMINAMATH_CALUDE_line_passes_through_point_l3000_300042

theorem line_passes_through_point (a : ℝ) : (a + 2) * 1 + a * (-1) - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l3000_300042


namespace NUMINAMATH_CALUDE_exists_non_one_same_first_digit_l3000_300094

/-- Given a natural number n, returns the first digit of n -/
def firstDigit (n : ℕ) : ℕ := sorry

/-- Returns true if all numbers in the list start with the same digit -/
def sameFirstDigit (numbers : List ℕ) : Bool := sorry

theorem exists_non_one_same_first_digit :
  ∃ x : ℕ, x > 0 ∧ 
  let powers := List.range 2015 |>.map (λ i => x^(i+1))
  sameFirstDigit powers ∧ 
  firstDigit x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_non_one_same_first_digit_l3000_300094


namespace NUMINAMATH_CALUDE_sqrt_special_sum_l3000_300055

def digits_to_num (d : ℕ) (n : ℕ) : ℕ := (10^n - 1) / (10 - 1) * d

theorem sqrt_special_sum (n : ℕ) (h : n > 0) :
  Real.sqrt (digits_to_num 4 (2*n) + digits_to_num 1 (n+1) - digits_to_num 6 n) = 
  digits_to_num 6 (n-1) + 7 :=
sorry

end NUMINAMATH_CALUDE_sqrt_special_sum_l3000_300055


namespace NUMINAMATH_CALUDE_school_relationship_l3000_300038

/-- In a school with teachers and students, prove the relationship between the number of teachers,
    students, students per teacher, and common teachers between any two students. -/
theorem school_relationship (m n k l : ℕ) : 
  (∀ (teacher : Fin m), ∃! (students : Finset (Fin n)), students.card = k) →
  (∀ (student1 student2 : Fin n), student1 ≠ student2 → 
    ∃! (common_teachers : Finset (Fin m)), common_teachers.card = l) →
  m * k * (k - 1) = n * (n - 1) * l := by
  sorry

end NUMINAMATH_CALUDE_school_relationship_l3000_300038


namespace NUMINAMATH_CALUDE_converse_not_always_true_l3000_300045

theorem converse_not_always_true : ¬(∀ a b c : ℝ, a > b → a * c^2 > b * c^2) := by
  sorry

end NUMINAMATH_CALUDE_converse_not_always_true_l3000_300045


namespace NUMINAMATH_CALUDE_digits_of_powers_l3000_300077

/-- A number is even and not divisible by 10 -/
def IsEvenNotDivBy10 (n : ℕ) : Prop :=
  Even n ∧ ¬(10 ∣ n)

/-- The tens digit of a natural number -/
def TensDigit (n : ℕ) : ℕ :=
  (n / 10) % 10

/-- The hundreds digit of a natural number -/
def HundredsDigit (n : ℕ) : ℕ :=
  (n / 100) % 10

theorem digits_of_powers (N : ℕ) (h : IsEvenNotDivBy10 N) :
  TensDigit (N^20) = 7 ∧ HundredsDigit (N^200) = 3 := by
  sorry

end NUMINAMATH_CALUDE_digits_of_powers_l3000_300077


namespace NUMINAMATH_CALUDE_difference_between_sum_and_average_l3000_300011

def numbers : List ℕ := [44, 16, 2, 77, 241]

theorem difference_between_sum_and_average : 
  (numbers.sum : ℚ) - (numbers.sum : ℚ) / numbers.length = 304 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_sum_and_average_l3000_300011


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_240_360_l3000_300027

theorem lcm_gcf_ratio_240_360 : 
  (Nat.lcm 240 360) / (Nat.gcd 240 360) = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_240_360_l3000_300027


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l3000_300068

theorem cubic_equation_roots (k m : ℝ) : 
  (∃ a b c : ℤ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (∀ x : ℝ, x^3 - 8*x^2 + k*x - m = 0 ↔ (x = a ∨ x = b ∨ x = c))) →
  k + m = 27 := by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l3000_300068


namespace NUMINAMATH_CALUDE_parallelogram_below_line_l3000_300053

structure Point where
  x : ℝ
  y : ℝ

def Parallelogram (A B C D : Point) : Prop :=
  (B.x - A.x = D.x - C.x) ∧ (B.y - A.y = D.y - C.y)

def BelowOrOnLine (p : Point) (y0 : ℝ) : Prop :=
  p.y ≤ y0

theorem parallelogram_below_line :
  let A : Point := ⟨4, 2⟩
  let B : Point := ⟨-2, -4⟩
  let C : Point := ⟨-8, -4⟩
  let D : Point := ⟨0, 4⟩
  let y0 : ℝ := -2
  Parallelogram A B C D →
  ∀ p : Point, (∃ t u : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 0 ≤ u ∧ u ≤ 1 ∧
    p.x = A.x + t * (B.x - A.x) + u * (D.x - A.x) ∧
    p.y = A.y + t * (B.y - A.y) + u * (D.y - A.y)) →
  BelowOrOnLine p y0 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_below_line_l3000_300053


namespace NUMINAMATH_CALUDE_average_pages_of_books_l3000_300031

theorem average_pages_of_books (books : List ℕ) (h : books = [120, 150, 180, 210, 240]) : 
  (books.sum / books.length : ℚ) = 180 := by
  sorry

end NUMINAMATH_CALUDE_average_pages_of_books_l3000_300031


namespace NUMINAMATH_CALUDE_unique_non_range_value_l3000_300092

/-- The function f defined by the given properties -/
noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

/-- Theorem stating that 58 is the unique number not in the range of f -/
theorem unique_non_range_value
  (a b c d : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h_19 : f a b c d 19 = 19)
  (h_97 : f a b c d 97 = 97)
  (h_inverse : ∀ x ≠ -d/c, f a b c d (f a b c d x) = x) :
  ∃! y, ∀ x, f a b c d x ≠ y ∧ y = 58 := by
  sorry

end NUMINAMATH_CALUDE_unique_non_range_value_l3000_300092


namespace NUMINAMATH_CALUDE_smallest_prime_digit_sum_23_l3000_300071

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Check if a number is prime -/
def is_prime (n : ℕ) : Prop := sorry

/-- Theorem: 1997 is the smallest prime whose digits sum to 23 -/
theorem smallest_prime_digit_sum_23 :
  (is_prime 1997) ∧ 
  (digit_sum 1997 = 23) ∧ 
  (∀ n : ℕ, n < 1997 → (is_prime n ∧ digit_sum n = 23) → False) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_digit_sum_23_l3000_300071


namespace NUMINAMATH_CALUDE_collins_earnings_per_can_l3000_300040

/-- The amount of money Collin earns per aluminum can -/
def earnings_per_can (cans_home : ℕ) (cans_grandparents_multiplier : ℕ) (cans_neighbor : ℕ) (cans_dad_office : ℕ) (savings_amount : ℚ) : ℚ :=
  let total_cans := cans_home + cans_home * cans_grandparents_multiplier + cans_neighbor + cans_dad_office
  let total_earnings := 2 * savings_amount
  total_earnings / total_cans

/-- Theorem stating that Collin earns $0.25 per aluminum can -/
theorem collins_earnings_per_can :
  earnings_per_can 12 3 46 250 43 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_collins_earnings_per_can_l3000_300040


namespace NUMINAMATH_CALUDE_triangle_area_l3000_300003

theorem triangle_area (A B C : Real) (angleC : A + B + C = Real.pi) 
  (sideAC sideAB : Real) (h_angleC : C = Real.pi / 6) 
  (h_sideAC : sideAC = 3 * Real.sqrt 3) (h_sideAB : sideAB = 3) :
  let area := (1 / 2) * sideAC * sideAB * Real.sin A
  area = (9 * Real.sqrt 3) / 2 ∨ area = (9 * Real.sqrt 3) / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l3000_300003


namespace NUMINAMATH_CALUDE_probability_is_one_fourth_l3000_300095

/-- Represents a right triangle XYZ with given side lengths -/
structure RightTriangle where
  xy : ℝ
  xz : ℝ
  angle_x_is_right : xy > 0 ∧ xz > 0

/-- Calculates the probability of a randomly chosen point P inside the right triangle XYZ
    forming a triangle PYZ with an area less than one-third of the area of XYZ -/
def probability_small_area (t : RightTriangle) : ℝ :=
  sorry

/-- The main theorem stating that for a right triangle with sides 6 and 8,
    the probability of forming a smaller triangle with area less than one-third
    of the original triangle's area is 1/4 -/
theorem probability_is_one_fourth :
  let t : RightTriangle := ⟨6, 8, by norm_num⟩
  probability_small_area t = 1/4 :=
sorry

end NUMINAMATH_CALUDE_probability_is_one_fourth_l3000_300095


namespace NUMINAMATH_CALUDE_tim_dozens_of_golf_balls_l3000_300036

def total_golf_balls : ℕ := 156
def balls_per_dozen : ℕ := 12

theorem tim_dozens_of_golf_balls : 
  total_golf_balls / balls_per_dozen = 13 := by sorry

end NUMINAMATH_CALUDE_tim_dozens_of_golf_balls_l3000_300036


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_achievable_l3000_300060

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  (x^2 + 3*x + 1) / x ≥ 5 :=
by sorry

theorem min_value_achievable :
  ∃ x : ℝ, x > 0 ∧ (x^2 + 3*x + 1) / x = 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_achievable_l3000_300060


namespace NUMINAMATH_CALUDE_circle_bisection_l3000_300099

-- Define the two circles
def circle1 (a b x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = b^2 + 1
def circle2 (x y : ℝ) : Prop := (x + 1)^2 + (y + 1)^2 = 4

-- Define the bisection condition
def bisects (a b : ℝ) : Prop := 
  ∀ x y : ℝ, circle1 a b x y → circle2 x y → 
    ∃ x' y' : ℝ, x' ≠ x ∧ y' ≠ y ∧ circle1 a b x' y' ∧ circle2 x' y'

-- State the theorem
theorem circle_bisection (a b : ℝ) :
  bisects a b → a^2 + 2*a + 2*b + 5 = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_bisection_l3000_300099


namespace NUMINAMATH_CALUDE_b_days_proof_l3000_300051

/-- The number of days it takes A to complete the work alone -/
def a_days : ℝ := 6

/-- The total payment for the work -/
def total_payment : ℝ := 3600

/-- The number of days it takes A, B, and C to complete the work together -/
def abc_days : ℝ := 3

/-- The payment given to C -/
def c_payment : ℝ := 450

/-- The number of days it takes B to complete the work alone -/
def b_days : ℝ := 6

theorem b_days_proof :
  (1 / a_days + 1 / b_days) * abc_days = 1 ∧
  c_payment / total_payment = 1 - (1 / a_days + 1 / b_days) * abc_days :=
by sorry

end NUMINAMATH_CALUDE_b_days_proof_l3000_300051


namespace NUMINAMATH_CALUDE_bobs_final_salary_l3000_300098

/-- Calculates the final salary after two raises and a pay cut -/
def final_salary (initial_salary : ℝ) (first_raise : ℝ) (second_raise : ℝ) (pay_cut : ℝ) : ℝ :=
  let salary_after_first_raise := initial_salary * (1 + first_raise)
  let salary_after_second_raise := salary_after_first_raise * (1 + second_raise)
  salary_after_second_raise * (1 - pay_cut)

/-- Theorem stating that Bob's final salary is $2541 -/
theorem bobs_final_salary :
  final_salary 3000 0.1 0.1 0.3 = 2541 := by
  sorry

end NUMINAMATH_CALUDE_bobs_final_salary_l3000_300098


namespace NUMINAMATH_CALUDE_quadratic_inequality_not_always_negative_l3000_300096

theorem quadratic_inequality_not_always_negative :
  ¬ (∀ x : ℝ, x^2 + x - 1 < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_not_always_negative_l3000_300096


namespace NUMINAMATH_CALUDE_unique_prime_factor_count_l3000_300074

def count_prime_factors (n : ℕ) : ℕ := sorry

theorem unique_prime_factor_count : 
  ∃! x : ℕ, x > 0 ∧ count_prime_factors ((4^11) * (7^5) * (x^2)) = 29 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_factor_count_l3000_300074


namespace NUMINAMATH_CALUDE_angle_in_third_quadrant_l3000_300026

theorem angle_in_third_quadrant (α : Real) (h1 : π < α ∧ α < 3*π/2) : 
  (Real.sin (π/2 - α) * Real.cos (-α) * Real.tan (π + α)) / Real.cos (π - α) = 2 * Real.sqrt 5 / 5 →
  Real.cos α = -(Real.sqrt 5 / 5) := by
  sorry

end NUMINAMATH_CALUDE_angle_in_third_quadrant_l3000_300026


namespace NUMINAMATH_CALUDE_function_property_l3000_300013

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_periodic_neg_one (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) = -f x

def is_monotone_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem function_property (f : ℝ → ℝ) 
  (h1 : is_even_function f) 
  (h2 : is_periodic_neg_one f) 
  (h3 : is_monotone_increasing_on f (-1) 0) :
  f 2 > f (Real.sqrt 2) ∧ f (Real.sqrt 2) > f 3 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3000_300013


namespace NUMINAMATH_CALUDE_fraction_division_result_l3000_300090

theorem fraction_division_result (a : ℝ) 
  (h1 : a^2 + 4*a + 4 ≠ 0) 
  (h2 : a^2 + 5*a + 6 ≠ 0) : 
  (a^2 - 4) / (a^2 + 4*a + 4) / ((a^2 + a - 6) / (a^2 + 5*a + 6)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_result_l3000_300090


namespace NUMINAMATH_CALUDE_function_range_l3000_300058

theorem function_range (x : ℝ) :
  (∀ a ∈ Set.Icc (-1 : ℝ) 1, x^2 + (a - 4)*x + 4 - 2*a > 0) →
  x < 1 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_function_range_l3000_300058


namespace NUMINAMATH_CALUDE_sum_of_digits_n_plus_5_l3000_300063

-- Define S(n) as the sum of digits of n
def S (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_n_plus_5 (n : ℕ) (h1 : S n = 365) (h2 : n % 8 = S n % 8) :
  S (n + 5) = 370 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_n_plus_5_l3000_300063


namespace NUMINAMATH_CALUDE_election_winner_margin_l3000_300039

theorem election_winner_margin (total_votes : ℕ) (winner_votes : ℕ) : 
  (2 : ℕ) ≤ total_votes →
  winner_votes = (75 * total_votes) / 100 →
  winner_votes = 750 →
  winner_votes - (total_votes - winner_votes) = 500 :=
by sorry

end NUMINAMATH_CALUDE_election_winner_margin_l3000_300039


namespace NUMINAMATH_CALUDE_complex_product_theorem_l3000_300032

theorem complex_product_theorem (z₁ z₂ : ℂ) : 
  (z₁.re = 1 ∧ z₁.im = 1) → (z₂.re = 1 ∧ z₂.im = -1) → z₁ * z₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l3000_300032


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3000_300078

theorem greatest_divisor_with_remainders (a b r1 r2 : ℕ) (h1 : a > r1) (h2 : b > r2) : 
  Nat.gcd (a - r1) (b - r2) = 
    Nat.gcd (a % (Nat.gcd (a - r1) (b - r2))) r1 ∧ 
    Nat.gcd (a - r1) (b - r2) = 
    Nat.gcd (b % (Nat.gcd (a - r1) (b - r2))) r2 → 
  Nat.gcd (a - r1) (b - r2) = 
    Nat.gcd (1642 - 6) (1856 - 4) := by
  sorry

#eval Nat.gcd (1642 - 6) (1856 - 4)

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3000_300078


namespace NUMINAMATH_CALUDE_equation_solution_l3000_300016

theorem equation_solution (x : ℝ) : 
  (8 * x^2 + 120 * x + 7) / (3 * x + 10) = 4 * x + 2 ↔ 
  -4 * x^2 + 74 * x - 13 = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3000_300016


namespace NUMINAMATH_CALUDE_equidistant_points_count_l3000_300000

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane, represented by its normal vector and distance from origin --/
structure Line where
  normal : ℝ × ℝ
  distance : ℝ

/-- A point in a 2D plane --/
def Point := ℝ × ℝ

/-- Distance between a point and a line --/
def distancePointToLine (p : Point) (l : Line) : ℝ := sorry

/-- Distance between a point and a circle --/
def distancePointToCircle (p : Point) (c : Circle) : ℝ := sorry

/-- Check if a line is tangent to a circle --/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- The main theorem --/
theorem equidistant_points_count
  (c : Circle)
  (t1 t2 : Line)
  (h1 : c.radius = 4)
  (h2 : isTangent t1 c)
  (h3 : isTangent t2 c)
  (h4 : t1.distance = 4)
  (h5 : t2.distance = 6)
  (h6 : t1.normal = t2.normal) :
  ∃! (s : Finset Point), 
    (∀ p ∈ s, distancePointToCircle p c = distancePointToLine p t1 ∧ 
                distancePointToCircle p c = distancePointToLine p t2) ∧
    s.card = 2 := by sorry

end NUMINAMATH_CALUDE_equidistant_points_count_l3000_300000


namespace NUMINAMATH_CALUDE_milk_for_pizza_dough_l3000_300076

/-- Given a ratio of 50 mL of milk for every 250 mL of flour, 
    calculate the amount of milk needed for 1200 mL of flour. -/
theorem milk_for_pizza_dough (flour : ℝ) (milk : ℝ) : 
  flour = 1200 → 
  (milk / flour = 50 / 250) → 
  milk = 240 := by sorry

end NUMINAMATH_CALUDE_milk_for_pizza_dough_l3000_300076


namespace NUMINAMATH_CALUDE_martin_bell_ringing_l3000_300047

theorem martin_bell_ringing (small big : ℕ) : 
  small = (big / 3) + 4 →  -- Condition 1
  small + big = 52 →      -- Condition 2
  big = 36 :=             -- Conclusion
by sorry

end NUMINAMATH_CALUDE_martin_bell_ringing_l3000_300047


namespace NUMINAMATH_CALUDE_product_of_digits_of_non_divisible_by_four_l3000_300097

def numbers : List Nat := [3612, 3620, 3628, 3636, 3641]

def is_divisible_by_four (n : Nat) : Bool :=
  n % 4 = 0

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

theorem product_of_digits_of_non_divisible_by_four :
  ∃ n ∈ numbers, ¬is_divisible_by_four n ∧ 
  units_digit n * tens_digit n = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_digits_of_non_divisible_by_four_l3000_300097


namespace NUMINAMATH_CALUDE_power_sum_equality_l3000_300070

theorem power_sum_equality : 3^3 + 4^3 + 5^3 = 6^3 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l3000_300070


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3000_300056

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y + y * f x) = f x + f y + x * f y) →
  (∀ x : ℝ, f x = 0 ∨ f x = x) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3000_300056


namespace NUMINAMATH_CALUDE_sunset_colors_l3000_300023

/-- The number of colors the sky turns during a sunset -/
def sky_colors (sunset_duration : ℕ) (color_change_interval : ℕ) (minutes_per_hour : ℕ) : ℕ :=
  (sunset_duration * minutes_per_hour) / color_change_interval

/-- Theorem: During a 2-hour sunset, with the sky changing color every 10 minutes,
    and each hour being 60 minutes long, the sky turns 12 different colors. -/
theorem sunset_colors :
  sky_colors 2 10 60 = 12 := by
  sorry

#eval sky_colors 2 10 60

end NUMINAMATH_CALUDE_sunset_colors_l3000_300023


namespace NUMINAMATH_CALUDE_turtleneck_profit_l3000_300066

/-- Represents the pricing strategy and profit calculation for turtleneck sweaters -/
theorem turtleneck_profit (C : ℝ) (C_pos : C > 0) : 
  let initial_markup : ℝ := 0.20
  let new_year_markup : ℝ := 0.25
  let february_discount : ℝ := 0.09
  let SP1 : ℝ := C * (1 + initial_markup)
  let SP2 : ℝ := SP1 * (1 + new_year_markup)
  let SPF : ℝ := SP2 * (1 - february_discount)
  let profit : ℝ := SPF - C
  profit / C = 0.365 := by sorry

end NUMINAMATH_CALUDE_turtleneck_profit_l3000_300066


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3000_300018

def A : Set ℝ := {x | -1 < x ∧ x < 4}
def B : Set ℝ := {-4, 1, 3, 5}

theorem intersection_of_A_and_B : A ∩ B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3000_300018


namespace NUMINAMATH_CALUDE_rectangle_area_l3000_300062

/-- Proves that the area of a rectangle is 108 square inches, given that its length is 3 times its width and its width is 6 inches. -/
theorem rectangle_area (width : ℝ) (length : ℝ) : 
  width = 6 → length = 3 * width → width * length = 108 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3000_300062


namespace NUMINAMATH_CALUDE_parallelogram_roots_l3000_300083

/-- The polynomial whose roots we're investigating -/
def P (b : ℝ) (z : ℂ) : ℂ := z^4 - 8*z^3 + 13*b*z^2 - 3*(2*b^2 + 5*b - 3)*z - 1

/-- Predicate to check if a set of complex numbers forms a parallelogram -/
def formsParallelogram (roots : Finset ℂ) : Prop :=
  roots.card = 4 ∧ ∃ (w₁ w₂ : ℂ), roots = {w₁, -w₁, w₂, -w₂}

/-- The main theorem stating that 3/2 is the only real value of b for which
    the roots of P form a parallelogram -/
theorem parallelogram_roots :
  ∃! (b : ℝ), b = 3/2 ∧ 
    ∃ (roots : Finset ℂ), (∀ z ∈ roots, P b z = 0) ∧ formsParallelogram roots :=
sorry

end NUMINAMATH_CALUDE_parallelogram_roots_l3000_300083


namespace NUMINAMATH_CALUDE_min_addition_to_prime_l3000_300029

def is_valid_number (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧
  n = 10 * a + b ∧ 2 * a * b = n

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem min_addition_to_prime :
  ∃ n : ℕ, is_valid_number n ∧
  (∀ k : ℕ, k < 1 → ¬(is_prime (n + k))) ∧
  is_prime (n + 1) :=
sorry

end NUMINAMATH_CALUDE_min_addition_to_prime_l3000_300029


namespace NUMINAMATH_CALUDE_min_value_sum_product_l3000_300061

theorem min_value_sum_product (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / (a + b + c) + 1 / (a + b + d) + 1 / (a + c + d) + 1 / (b + c + d)) ≥ 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_product_l3000_300061


namespace NUMINAMATH_CALUDE_expression_evaluation_l3000_300043

theorem expression_evaluation : 12 - 10 + 8 * 7 + 6 - 5 * 4 + 3 / 3 - 2 = 43 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3000_300043


namespace NUMINAMATH_CALUDE_intersection_has_two_elements_l3000_300049

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | p.2 = p.1^2}
def B : Set (ℝ × ℝ) := {p | p.2 = 1 - |p.1|}

-- State the theorem
theorem intersection_has_two_elements :
  ∃ (p₁ p₂ : ℝ × ℝ), p₁ ≠ p₂ ∧ A ∩ B = {p₁, p₂} :=
sorry

end NUMINAMATH_CALUDE_intersection_has_two_elements_l3000_300049


namespace NUMINAMATH_CALUDE_aquafaba_to_egg_white_ratio_l3000_300021

theorem aquafaba_to_egg_white_ratio : 
  let num_cakes : ℕ := 2
  let egg_whites_per_cake : ℕ := 8
  let total_aquafaba : ℕ := 32
  let total_egg_whites : ℕ := num_cakes * egg_whites_per_cake
  (total_aquafaba : ℚ) / (total_egg_whites : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_aquafaba_to_egg_white_ratio_l3000_300021


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_general_quadratic_inequality_solution_l3000_300084

def quadratic_inequality (a b : ℝ) : Set ℝ :=
  {x | a * x^2 - 3 * x + 2 > 0}

def solution_set (b : ℝ) : Set ℝ :=
  {x | x < 1 ∨ x > b}

theorem quadratic_inequality_solution (a b : ℝ) :
  quadratic_inequality a b = solution_set b → a = 1 ∧ b = 2 :=
sorry

def general_quadratic_inequality (a b c : ℝ) : Set ℝ :=
  {x | x^2 - b * (a + c) * x + 4 * c > 0}

theorem general_quadratic_inequality_solution (a b c : ℝ) :
  a = 1 ∧ b = 2 →
  (c > 1 → general_quadratic_inequality a b c = {x | x < 2 ∨ x > 2 * c}) ∧
  (c = 1 → general_quadratic_inequality a b c = {x | x ≠ 2}) ∧
  (c < 1 → general_quadratic_inequality a b c = {x | x > 2 ∨ x < 2 * c}) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_general_quadratic_inequality_solution_l3000_300084


namespace NUMINAMATH_CALUDE_unique_number_existence_l3000_300072

def digit_sum (n : ℕ) : ℕ := sorry

def num_digits (n : ℕ) : ℕ := sorry

def has_digit (n d : ℕ) : Prop := sorry

def all_nines_except_one (n : ℕ) (pos : ℕ) : Prop := sorry

theorem unique_number_existence :
  ∃! N : ℕ,
    (num_digits N = 1112) ∧
    (2000 ∣ digit_sum N) ∧
    (2000 ∣ digit_sum (N + 1)) ∧
    (has_digit N 1) ∧
    (all_nines_except_one N 890) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_number_existence_l3000_300072


namespace NUMINAMATH_CALUDE_four_thirds_of_x_is_36_l3000_300085

theorem four_thirds_of_x_is_36 : ∃ x : ℚ, (4 / 3) * x = 36 ∧ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_four_thirds_of_x_is_36_l3000_300085


namespace NUMINAMATH_CALUDE_quadratic_transformation_l3000_300087

theorem quadratic_transformation (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c = 2 * (x - 4)^2 + 8) →
  ∃ n k, ∀ x, 3 * a * x^2 + 3 * b * x + 3 * c = n * (x - 4)^2 + k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l3000_300087


namespace NUMINAMATH_CALUDE_simple_interest_rate_l3000_300082

/-- Given a principal sum and a time period of 7 years, if the simple interest
    is one-fifth of the principal, prove that the annual interest rate is 20/7. -/
theorem simple_interest_rate (P : ℝ) (P_pos : P > 0) : 
  (P * 7 * (20 / 7) / 100 = P / 5) → (20 / 7 : ℝ) = 20 / 7 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l3000_300082


namespace NUMINAMATH_CALUDE_guessing_game_l3000_300014

theorem guessing_game (G C : ℕ) (h1 : G = 33) (h2 : 3 * G = 2 * C - 3) : C = 51 := by
  sorry

end NUMINAMATH_CALUDE_guessing_game_l3000_300014


namespace NUMINAMATH_CALUDE_computer_sales_ratio_l3000_300052

theorem computer_sales_ratio (total : ℕ) (laptops : ℕ) (desktops : ℕ) (netbooks : ℕ) :
  total = 72 →
  laptops = total / 2 →
  desktops = 12 →
  netbooks = total - laptops - desktops →
  (netbooks : ℚ) / total = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_computer_sales_ratio_l3000_300052


namespace NUMINAMATH_CALUDE_unique_divisible_by_six_l3000_300028

def is_divisible_by (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

theorem unique_divisible_by_six : 
  ∀ B : ℕ, B < 10 → 
    (is_divisible_by (7520 + B) 6 ↔ B = 4) :=
by sorry

end NUMINAMATH_CALUDE_unique_divisible_by_six_l3000_300028


namespace NUMINAMATH_CALUDE_stock_price_after_two_years_l3000_300050

/-- Calculates the final stock price after two years of growth -/
def final_stock_price (initial_price : ℝ) (first_year_growth : ℝ) (second_year_growth : ℝ) : ℝ :=
  initial_price * (1 + first_year_growth) * (1 + second_year_growth)

/-- Theorem stating that the stock price after two years of growth is $247.50 -/
theorem stock_price_after_two_years :
  final_stock_price 150 0.5 0.1 = 247.50 := by
  sorry

#eval final_stock_price 150 0.5 0.1

end NUMINAMATH_CALUDE_stock_price_after_two_years_l3000_300050


namespace NUMINAMATH_CALUDE_max_value_of_g_l3000_300012

noncomputable def f (c : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + c * x + 3

def tangent_perpendicular (c : ℝ) : Prop :=
  (deriv (f c) 0) * 1 = -1

noncomputable def g (c : ℝ) (x : ℝ) : ℝ := 4 * Real.log x - deriv (f c) x

theorem max_value_of_g (c : ℝ) :
  tangent_perpendicular c →
  ∃ (x_max : ℝ), x_max > 0 ∧ g c x_max = 2 * Real.log 2 - 1 ∧
  ∀ (x : ℝ), x > 0 → g c x ≤ g c x_max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l3000_300012


namespace NUMINAMATH_CALUDE_birds_flew_up_l3000_300019

theorem birds_flew_up (initial_birds final_birds : ℕ) 
  (h1 : initial_birds = 231)
  (h2 : final_birds = 312)
  : final_birds - initial_birds = 81 := by
  sorry

end NUMINAMATH_CALUDE_birds_flew_up_l3000_300019


namespace NUMINAMATH_CALUDE_correct_algebraic_notation_l3000_300041

/-- Predicate to check if an expression follows algebraic notation rules -/
def follows_algebraic_notation (expr : String) : Prop :=
  match expr with
  | "7/3 * x^2" => True
  | "a * 1/4" => False
  | "-2 1/6 * p" => False
  | "2y / z" => False
  | _ => False

/-- Theorem stating that 7/3 * x^2 follows algebraic notation rules -/
theorem correct_algebraic_notation :
  follows_algebraic_notation "7/3 * x^2" ∧
  ¬follows_algebraic_notation "a * 1/4" ∧
  ¬follows_algebraic_notation "-2 1/6 * p" ∧
  ¬follows_algebraic_notation "2y / z" :=
sorry

end NUMINAMATH_CALUDE_correct_algebraic_notation_l3000_300041


namespace NUMINAMATH_CALUDE_inequality_statements_truth_l3000_300075

theorem inequality_statements_truth :
  let statement1 := ∀ (a b c d : ℝ), a > b ∧ c > d → a - c > b - d
  let statement2 := ∀ (a b c d : ℝ), a > b ∧ b > 0 ∧ c > d ∧ d > 0 → a * c > b * d
  let statement3 := ∀ (a b : ℝ), a > b ∧ b > 0 → 3 * a > 3 * b
  let statement4 := ∀ (a b : ℝ), a > b ∧ b > 0 → 1 / (a^2) < 1 / (b^2)
  (¬statement1 ∧ statement2 ∧ statement3 ∧ statement4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_statements_truth_l3000_300075


namespace NUMINAMATH_CALUDE_max_band_members_l3000_300022

theorem max_band_members : ∃ (m : ℕ), m = 234 ∧
  (∃ (k : ℕ), m = k^2 + 9) ∧
  (∃ (n : ℕ), m = n * (n + 5)) ∧
  (∀ (m' : ℕ), m' > m →
    (∃ (k : ℕ), m' = k^2 + 9) →
    (∃ (n : ℕ), m' = n * (n + 5)) →
    False) :=
by sorry

end NUMINAMATH_CALUDE_max_band_members_l3000_300022


namespace NUMINAMATH_CALUDE_circle_radius_l3000_300065

/-- The radius of a circle satisfying the given condition -/
theorem circle_radius : ∃ (r : ℝ), r > 0 ∧ 3 * (2 * Real.pi * r) = 2 * (Real.pi * r^2) ∧ r = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l3000_300065


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l3000_300033

theorem rectangular_solid_volume (a b c : ℝ) 
  (h1 : a * b = 18)
  (h2 : b * c = 50)
  (h3 : a * c = 45) :
  a * b * c = 150 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l3000_300033


namespace NUMINAMATH_CALUDE_recurring_decimal_fraction_sum_l3000_300088

theorem recurring_decimal_fraction_sum (a b : ℕ+) : 
  (a : ℚ) / (b : ℚ) = 56 / 99 → 
  Nat.gcd a b = 1 → 
  (a : ℕ) + b = 155 := by
  sorry

end NUMINAMATH_CALUDE_recurring_decimal_fraction_sum_l3000_300088


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3000_300064

def M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def N : Set ℝ := {x | |x| < 2}

theorem intersection_of_M_and_N :
  M ∩ N = {x | -1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3000_300064


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3000_300091

/-- Definition of the sum of a geometric sequence -/
def geometric_sum (a : ℚ) (q : ℚ) (n : ℕ) : ℚ :=
  a * (1 - q^n) / (1 - q)

/-- Theorem: Given S_4 = 1 and S_8 = 17, the first term a_1 is either 1/15 or -1/5 -/
theorem geometric_sequence_first_term
  (a : ℚ) (q : ℚ)
  (h1 : geometric_sum a q 4 = 1)
  (h2 : geometric_sum a q 8 = 17) :
  a = 1/15 ∨ a = -1/5 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3000_300091


namespace NUMINAMATH_CALUDE_square_not_always_positive_l3000_300057

theorem square_not_always_positive : ¬ ∀ x : ℝ, x^2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_square_not_always_positive_l3000_300057


namespace NUMINAMATH_CALUDE_circle_area_theorem_l3000_300024

-- Define the circle ω
def ω : Set (ℝ × ℝ) := sorry

-- Define points A and B
def A : ℝ × ℝ := (4, 10)
def B : ℝ × ℝ := (10, 8)

-- State that A and B lie on circle ω
axiom A_on_ω : A ∈ ω
axiom B_on_ω : B ∈ ω

-- Define the tangent lines at A and B
def tangent_A : Set (ℝ × ℝ) := sorry
def tangent_B : Set (ℝ × ℝ) := sorry

-- Define the intersection point of tangent lines
def intersection : ℝ × ℝ := sorry

-- State that the intersection point is on the x-axis
axiom intersection_on_x_axis : intersection.2 = 0

-- Define the area of a circle
def circle_area (c : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem circle_area_theorem : circle_area ω = 100 * π / 9 := by sorry

end NUMINAMATH_CALUDE_circle_area_theorem_l3000_300024


namespace NUMINAMATH_CALUDE_square_difference_theorem_l3000_300001

theorem square_difference_theorem (a b A : ℝ) : 
  (5*a + 3*b)^2 = (5*a - 3*b)^2 + A → A = 60*a*b := by
  sorry

end NUMINAMATH_CALUDE_square_difference_theorem_l3000_300001


namespace NUMINAMATH_CALUDE_positive_integer_equation_l3000_300089

theorem positive_integer_equation (N : ℕ+) : 15^4 * 28^2 = 12^2 * N^2 ↔ N = 525 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_equation_l3000_300089


namespace NUMINAMATH_CALUDE_units_digit_of_seven_to_sixth_l3000_300009

theorem units_digit_of_seven_to_sixth (n : ℕ) : n = 7^6 → n % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_seven_to_sixth_l3000_300009


namespace NUMINAMATH_CALUDE_intersection_point_is_7_neg8_l3000_300035

/-- Two lines in 2D space --/
structure TwoLines where
  line1 : ℝ → ℝ × ℝ
  line2 : ℝ → ℝ × ℝ

/-- The given two lines from the problem --/
def givenLines : TwoLines where
  line1 := λ t => (1 + 2*t, 1 - 3*t)
  line2 := λ u => (5 + 4*u, -9 + 2*u)

/-- Definition of intersection point --/
def isIntersectionPoint (p : ℝ × ℝ) (lines : TwoLines) : Prop :=
  ∃ t u, lines.line1 t = p ∧ lines.line2 u = p

/-- Theorem stating that (7, -8) is the intersection point of the given lines --/
theorem intersection_point_is_7_neg8 :
  isIntersectionPoint (7, -8) givenLines := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_is_7_neg8_l3000_300035


namespace NUMINAMATH_CALUDE_white_area_calculation_l3000_300073

theorem white_area_calculation (total_area grey_area1 grey_area2 dark_grey_area : ℝ) 
  (h1 : total_area = 32)
  (h2 : grey_area1 = 16)
  (h3 : grey_area2 = 15)
  (h4 : dark_grey_area = 4) :
  total_area = grey_area1 + grey_area2 + (total_area - grey_area1 - grey_area2 + dark_grey_area) - dark_grey_area :=
by sorry

end NUMINAMATH_CALUDE_white_area_calculation_l3000_300073


namespace NUMINAMATH_CALUDE_largest_geometric_sequence_number_l3000_300020

/-- Checks if a three-digit number's digits form a geometric sequence -/
def is_geometric_sequence (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  ∃ (a b c : ℕ) (r : ℚ),
    n = 100 * a + 10 * b + c ∧
    a = 8 ∧
    b = Int.floor (8 * r) ∧
    c = Int.floor (8 * r^2) ∧
    r > 0 ∧ r < 1

/-- Checks if a three-digit number has distinct digits -/
def has_distinct_digits (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- The main theorem stating that 842 is the largest three-digit number
    satisfying the given conditions -/
theorem largest_geometric_sequence_number :
  ∀ n : ℕ, n ≥ 100 ∧ n < 1000 →
    is_geometric_sequence n →
    has_distinct_digits n →
    n ≤ 842 :=
by sorry

end NUMINAMATH_CALUDE_largest_geometric_sequence_number_l3000_300020


namespace NUMINAMATH_CALUDE_function_not_in_first_quadrant_l3000_300025

/-- The function f(x) = (1/5)^(x+1) + m does not pass through the first quadrant if and only if m ≤ -1/5 -/
theorem function_not_in_first_quadrant (m : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → (1/5)^(x+1) + m ≤ 0) ↔ m ≤ -1/5 := by sorry

end NUMINAMATH_CALUDE_function_not_in_first_quadrant_l3000_300025
