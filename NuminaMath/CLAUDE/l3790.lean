import Mathlib

namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3790_379006

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_monotone_increasing_on_positive (f : ℝ → ℝ) : Prop := 
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

-- State the theorem
theorem solution_set_of_inequality 
  (h_odd : is_odd f)
  (h_monotone : is_monotone_increasing_on_positive f)
  (h_f1 : f 1 = 0) :
  {x : ℝ | (f x - f (-x)) / x > 0} = {x : ℝ | x < -1 ∨ 1 < x} :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3790_379006


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l3790_379075

theorem quadratic_integer_roots (p q : ℤ) :
  ∀ n : ℕ, n ≤ 9 →
  ∃ x y : ℤ, x^2 + (p + n) * x + (q + n) = 0 ∧
             y^2 + (p + n) * y + (q + n) = 0 ∧
             x ≠ y :=
by sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l3790_379075


namespace NUMINAMATH_CALUDE_division_remainder_l3790_379011

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 760 →
  divisor = 36 →
  quotient = 21 →
  dividend = divisor * quotient + remainder →
  remainder = 4 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l3790_379011


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3790_379064

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a - 1)*x + 1 < 0) ↔ (a < -1 ∨ a > 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3790_379064


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3790_379008

theorem polynomial_divisibility (m : ℤ) : 
  ∃ k : ℤ, (4*m + 5)^2 - 9 = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3790_379008


namespace NUMINAMATH_CALUDE_max_value_AMC_l3790_379024

theorem max_value_AMC (A M C : ℕ) (sum_constraint : A + M + C = 10) :
  (∀ A' M' C' : ℕ, A' + M' + C' = 10 → 
    A' * M' * C' + A' * M' + M' * C' + C' * A' ≤ A * M * C + A * M + M * C + C * A) →
  A * M * C + A * M + M * C + C * A = 69 := by
  sorry

end NUMINAMATH_CALUDE_max_value_AMC_l3790_379024


namespace NUMINAMATH_CALUDE_increasing_sequence_condition_l3790_379005

theorem increasing_sequence_condition (a : ℕ → ℝ) (b : ℝ) :
  (∀ n : ℕ, n > 0 → a n < a (n + 1)) →
  (∀ n : ℕ, n > 0 → a n = n^2 + b*n) →
  b > -3 :=
sorry

end NUMINAMATH_CALUDE_increasing_sequence_condition_l3790_379005


namespace NUMINAMATH_CALUDE_simplify_fraction_l3790_379098

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3790_379098


namespace NUMINAMATH_CALUDE_factorial_15_not_divisible_by_17_l3790_379042

theorem factorial_15_not_divisible_by_17 : ¬(17 ∣ Nat.factorial 15) := by
  sorry

end NUMINAMATH_CALUDE_factorial_15_not_divisible_by_17_l3790_379042


namespace NUMINAMATH_CALUDE_next_joint_tutoring_day_l3790_379021

def jaclyn_schedule : ℕ := 3
def marcelle_schedule : ℕ := 4
def susanna_schedule : ℕ := 6
def wanda_schedule : ℕ := 7

theorem next_joint_tutoring_day :
  Nat.lcm jaclyn_schedule (Nat.lcm marcelle_schedule (Nat.lcm susanna_schedule wanda_schedule)) = 84 := by
  sorry

end NUMINAMATH_CALUDE_next_joint_tutoring_day_l3790_379021


namespace NUMINAMATH_CALUDE_equation_solution_l3790_379017

theorem equation_solution : 
  ∀ x : ℝ, x > 0 → (x^(Real.log x / Real.log 10) = x^5 / 10000 ↔ x = 10 ∨ x = 10000) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3790_379017


namespace NUMINAMATH_CALUDE_quadratic_solution_l3790_379084

theorem quadratic_solution (c : ℝ) : 
  ((-9 : ℝ)^2 + c * (-9) - 36 = 0) → c = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_l3790_379084


namespace NUMINAMATH_CALUDE_max_sum_xyz_l3790_379025

theorem max_sum_xyz (x y z : ℕ+) (h1 : x < y) (h2 : y < z) (h3 : x + x * y + x * y * z = 37) :
  x + y + z ≤ 20 :=
sorry

end NUMINAMATH_CALUDE_max_sum_xyz_l3790_379025


namespace NUMINAMATH_CALUDE_gift_contribution_theorem_l3790_379046

theorem gift_contribution_theorem (n : ℕ) (min_contribution max_contribution total : ℝ) :
  n = 12 →
  min_contribution = 1 →
  max_contribution = 9 →
  (∀ person, person ∈ Finset.range n → min_contribution ≤ person) →
  (∀ person, person ∈ Finset.range n → person ≤ max_contribution) →
  total = (n - 1) * min_contribution + max_contribution →
  total = 20 := by
  sorry

end NUMINAMATH_CALUDE_gift_contribution_theorem_l3790_379046


namespace NUMINAMATH_CALUDE_line_problem_l3790_379039

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if two lines are perpendicular -/
def are_perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem line_problem (m n : ℝ) :
  let l1 : Line := ⟨2, 2, -1⟩
  let l2 : Line := ⟨4, n, 3⟩
  let l3 : Line := ⟨m, 6, 1⟩
  are_parallel l1 l2 ∧ are_perpendicular l1 l3 → m + n = -2 := by
  sorry

end NUMINAMATH_CALUDE_line_problem_l3790_379039


namespace NUMINAMATH_CALUDE_price_difference_enhanced_basic_computer_l3790_379048

/-- Prove the price difference between enhanced and basic computers --/
theorem price_difference_enhanced_basic_computer :
  ∀ (basic_price enhanced_price printer_price : ℕ),
  basic_price = 1500 →
  basic_price + printer_price = 2500 →
  printer_price = (enhanced_price + printer_price) / 3 →
  enhanced_price - basic_price = 500 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_enhanced_basic_computer_l3790_379048


namespace NUMINAMATH_CALUDE_common_terms_is_geometric_l3790_379038

/-- Arithmetic sequence with sum of first n terms S_n = (3n^2 + 5n) / 2 -/
def arithmetic_sequence (n : ℕ) : ℚ :=
  3 * n + 1

/-- Geometric sequence with b_3 = 4 and b_6 = 32 -/
def geometric_sequence (n : ℕ) : ℚ :=
  2^(n - 1)

/-- Sequence of common terms between arithmetic_sequence and geometric_sequence -/
def common_terms (n : ℕ) : ℚ :=
  4^n

theorem common_terms_is_geometric :
  ∀ n : ℕ, n > 0 → ∃ k : ℕ, k > 0 ∧ 
    arithmetic_sequence k = geometric_sequence k ∧
    common_terms n = arithmetic_sequence k := by
  sorry

end NUMINAMATH_CALUDE_common_terms_is_geometric_l3790_379038


namespace NUMINAMATH_CALUDE_complex_fraction_square_l3790_379035

theorem complex_fraction_square (m n : ℝ) (h : m * (1 + Complex.I) = 1 + n * Complex.I) :
  ((m + n * Complex.I) / (m - n * Complex.I))^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_square_l3790_379035


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l3790_379043

/-- Given point A(-3, 1, 4), prove that its symmetric point B with respect to the origin has coordinates (3, -1, -4). -/
theorem symmetric_point_coordinates :
  let A : ℝ × ℝ × ℝ := (-3, 1, 4)
  let B : ℝ × ℝ × ℝ := (3, -1, -4)
  (∀ (x y z : ℝ), (x, y, z) = A → (-x, -y, -z) = B) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l3790_379043


namespace NUMINAMATH_CALUDE_wilson_family_seating_arrangements_l3790_379079

/-- The number of ways to seat a family with the given constraints -/
def seatingArrangements (numBoys numGirls : ℕ) : ℕ :=
  let numAdjacentBoys := 3
  let totalSeats := numBoys + numGirls
  let numRemainingBoys := numBoys - numAdjacentBoys
  let numEntities := numRemainingBoys + numGirls + 1  -- +1 for the block of 3 boys
  (numBoys.choose numAdjacentBoys) * (Nat.factorial numAdjacentBoys) *
  (Nat.factorial numEntities) * (Nat.factorial numRemainingBoys) *
  (Nat.factorial numGirls)

/-- Theorem stating that the number of seating arrangements for the Wilson family is 5760 -/
theorem wilson_family_seating_arrangements :
  seatingArrangements 5 2 = 5760 := by
  sorry

#eval seatingArrangements 5 2

end NUMINAMATH_CALUDE_wilson_family_seating_arrangements_l3790_379079


namespace NUMINAMATH_CALUDE_quadratic_two_roots_implies_a_le_two_l3790_379007

/-- 
Given a quadratic equation x^2 - 4x + 2a = 0 with parameter a,
if the equation has two real roots, then a ≤ 2.
-/
theorem quadratic_two_roots_implies_a_le_two : 
  ∀ (a : ℝ), (∃ (x y : ℝ), x ≠ y ∧ x^2 - 4*x + 2*a = 0 ∧ y^2 - 4*y + 2*a = 0) → a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_implies_a_le_two_l3790_379007


namespace NUMINAMATH_CALUDE_fish_count_l3790_379012

/-- The number of fish Lilly has -/
def lilly_fish : ℕ := 10

/-- The number of fish Rosy has -/
def rosy_fish : ℕ := 12

/-- The total number of fish Lilly and Rosy have together -/
def total_fish : ℕ := lilly_fish + rosy_fish

theorem fish_count : total_fish = 22 := by
  sorry

end NUMINAMATH_CALUDE_fish_count_l3790_379012


namespace NUMINAMATH_CALUDE_calculate_expression_l3790_379059

theorem calculate_expression : -3^2 + |(-5)| - 18 * (-1/3)^2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3790_379059


namespace NUMINAMATH_CALUDE_max_value_of_e_l3790_379056

def b (n : ℕ) : ℤ := (10^n - 1) / 7

def e (n : ℕ) : ℕ := Nat.gcd (Int.natAbs (b n)) (Int.natAbs (b (n + 2)))

theorem max_value_of_e : ∀ n : ℕ, e n ≤ 99 ∧ ∃ m : ℕ, e m = 99 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_e_l3790_379056


namespace NUMINAMATH_CALUDE_circle_equation_and_tangent_lines_l3790_379044

def circle_C (a b : ℝ) := {(x, y) : ℝ × ℝ | (x - a)^2 + (y - b)^2 = 2}

theorem circle_equation_and_tangent_lines :
  ∀ (a b : ℝ),
    b = a + 1 →
    (5 - a)^2 + (4 - b)^2 = 2 →
    (3 - a)^2 + (6 - b)^2 = 2 →
    (∃ (x y : ℝ), circle_C a b (x, y)) →
    (circle_C 4 5 = circle_C a b) ∧
    (∀ (k : ℝ),
      (k = 1 ∨ k = 23/7) ↔
      (∃ (x : ℝ), x ≠ 1 ∧ circle_C 4 5 (x, k*(x-1)) ∧
        ∀ (y : ℝ), y ≠ k*(x-1) → ¬ circle_C 4 5 (x, y))) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_and_tangent_lines_l3790_379044


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3790_379004

theorem isosceles_triangle_perimeter : ∀ a b : ℝ,
  (a^2 - 6*a + 8 = 0) →
  (b^2 - 6*b + 8 = 0) →
  (a ≠ b) →
  (∃ c : ℝ, c = max a b ∧ c = min a b + (max a b - min a b) ∧ a + b + c = 10) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3790_379004


namespace NUMINAMATH_CALUDE_fraction_value_l3790_379097

theorem fraction_value : (2200 - 2089)^2 / 196 = 63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3790_379097


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3790_379049

theorem trigonometric_identity : 
  Real.sin (4/3 * π) * Real.cos (11/6 * π) * Real.tan (3/4 * π) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3790_379049


namespace NUMINAMATH_CALUDE_b_10_equals_64_l3790_379026

/-- Sequences a and b satisfying the given conditions -/
def sequences_a_b (a b : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ 
  ∀ n : ℕ, (a n) * (a (n + 1)) = 2^n ∧
           (a n) + (a (n + 1)) = b n

/-- The main theorem to prove -/
theorem b_10_equals_64 (a b : ℕ → ℝ) (h : sequences_a_b a b) : 
  b 10 = 64 := by
  sorry

end NUMINAMATH_CALUDE_b_10_equals_64_l3790_379026


namespace NUMINAMATH_CALUDE_root_sum_magnitude_l3790_379099

theorem root_sum_magnitude (p : ℝ) (r₁ r₂ : ℝ) : 
  r₁ ≠ r₂ →
  r₁^2 + p*r₁ + 9 = 0 →
  r₂^2 + p*r₂ + 9 = 0 →
  |r₁ + r₂| > 6 := by
sorry

end NUMINAMATH_CALUDE_root_sum_magnitude_l3790_379099


namespace NUMINAMATH_CALUDE_units_digit_of_3_pow_1789_units_digit_of_1777_pow_1777_pow_1777_l3790_379092

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem for the first part
theorem units_digit_of_3_pow_1789 :
  unitsDigit (3^1789) = 3 := by sorry

-- Theorem for the second part
theorem units_digit_of_1777_pow_1777_pow_1777 :
  unitsDigit (1777^(1777^1777)) = 7 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_3_pow_1789_units_digit_of_1777_pow_1777_pow_1777_l3790_379092


namespace NUMINAMATH_CALUDE_quadratic_discriminant_condition_l3790_379037

theorem quadratic_discriminant_condition (a b c : ℝ) :
  (2 * a ≠ 0) →
  (ac = (9 * b^2 - 25) / 32) ↔ ((3 * b)^2 - 4 * (2 * a) * (4 * c) = 25) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_condition_l3790_379037


namespace NUMINAMATH_CALUDE_geometric_sequence_bound_l3790_379077

/-- Given two geometric sequences with specified properties, prove that the first term of the first sequence must be less than 4/3 -/
theorem geometric_sequence_bound (a b : ℝ) (r_a r_b : ℝ) : 
  (∑' i, a * r_a ^ i = 1) →
  (∑' i, b * r_b ^ i = 1) →
  (∑' i, (a * r_a ^ i) ^ 2) * (∑' i, (b * r_b ^ i) ^ 2) = ∑' i, (a * r_a ^ i) * (b * r_b ^ i) →
  a < 4/3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_bound_l3790_379077


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3790_379001

theorem algebraic_expression_value : 
  let x : ℝ := -1
  3 * x^2 + 2 * x - 1 = 0 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3790_379001


namespace NUMINAMATH_CALUDE_pet_shop_total_cost_l3790_379060

/-- Represents the cost of purchasing all pets in a pet shop given specific conditions. -/
def total_cost_of_pets (num_puppies num_kittens num_parakeets : ℕ) 
  (parakeet_cost : ℚ) 
  (puppy_parakeet_ratio kitten_parakeet_ratio : ℚ) : ℚ :=
  let puppy_cost := puppy_parakeet_ratio * parakeet_cost
  let kitten_cost := kitten_parakeet_ratio * parakeet_cost
  (num_puppies : ℚ) * puppy_cost + (num_kittens : ℚ) * kitten_cost + (num_parakeets : ℚ) * parakeet_cost

/-- Theorem stating that under given conditions, the total cost of pets is $130. -/
theorem pet_shop_total_cost : 
  total_cost_of_pets 2 2 3 10 3 2 = 130 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_total_cost_l3790_379060


namespace NUMINAMATH_CALUDE_problem_statement_l3790_379057

theorem problem_statement (a b : ℝ) (ha : a = 3) (hb : b = 2) :
  2 * (a^3 + b^3) / (a^2 - a*b + b^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3790_379057


namespace NUMINAMATH_CALUDE_sum_of_base7_series_l3790_379028

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 -/
def base10ToBase7 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of an arithmetic series -/
def arithmeticSeriesSum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_of_base7_series :
  let lastTerm := base7ToBase10 33
  let sum := arithmeticSeriesSum lastTerm
  base10ToBase7 sum = 606 := by sorry

end NUMINAMATH_CALUDE_sum_of_base7_series_l3790_379028


namespace NUMINAMATH_CALUDE_kekai_mms_packs_l3790_379089

/-- The number of sundaes made on Monday -/
def monday_sundaes : ℕ := 40

/-- The number of m&ms per sundae on Monday -/
def monday_mms_per_sundae : ℕ := 6

/-- The number of sundaes made on Tuesday -/
def tuesday_sundaes : ℕ := 20

/-- The number of m&ms per sundae on Tuesday -/
def tuesday_mms_per_sundae : ℕ := 10

/-- The number of m&ms in each pack -/
def mms_per_pack : ℕ := 40

/-- The total number of m&m packs used -/
def total_packs_used : ℕ := 11

theorem kekai_mms_packs :
  (monday_sundaes * monday_mms_per_sundae + tuesday_sundaes * tuesday_mms_per_sundae) / mms_per_pack = total_packs_used :=
by sorry

end NUMINAMATH_CALUDE_kekai_mms_packs_l3790_379089


namespace NUMINAMATH_CALUDE_angle_B_measure_l3790_379047

/-- In a triangle ABC, given that the measures of angles A, B, C form a geometric progression
    and b^2 - a^2 = a*c, prove that the measure of angle B is 2π/7 -/
theorem angle_B_measure (A B C : ℝ) (a b c : ℝ) :
  A > 0 → B > 0 → C > 0 →
  a > 0 → b > 0 → c > 0 →
  A + B + C = π →
  ∃ (q : ℝ), q > 0 ∧ B = q * A ∧ C = q * B →
  b^2 - a^2 = a * c →
  B = 2 * π / 7 := by
sorry

end NUMINAMATH_CALUDE_angle_B_measure_l3790_379047


namespace NUMINAMATH_CALUDE_house_cost_is_480000_l3790_379036

/-- Calculates the cost of a house given the following conditions:
  - A trailer costs $120,000
  - Each loan will be paid in monthly installments over 20 years
  - The monthly payment on the house is $1500 more than the trailer
-/
def house_cost (trailer_cost : ℕ) (loan_years : ℕ) (monthly_difference : ℕ) : ℕ :=
  let months : ℕ := loan_years * 12
  let trailer_monthly : ℕ := trailer_cost / months
  let house_monthly : ℕ := trailer_monthly + monthly_difference
  house_monthly * months

/-- Theorem stating that the cost of the house is $480,000 -/
theorem house_cost_is_480000 :
  house_cost 120000 20 1500 = 480000 := by
  sorry

end NUMINAMATH_CALUDE_house_cost_is_480000_l3790_379036


namespace NUMINAMATH_CALUDE_remaining_student_l3790_379030

theorem remaining_student (n : ℕ) (hn : n ≤ 2002) : n % 1331 = 0 ↔ n = 1331 :=
by sorry

#check remaining_student

end NUMINAMATH_CALUDE_remaining_student_l3790_379030


namespace NUMINAMATH_CALUDE_extremum_implies_a_in_open_interval_l3790_379063

open Set
open Function
open Real

/-- A function f has exactly one extremum point in an interval (a, b) if there exists
    exactly one point c in (a, b) where f'(c) = 0. -/
def has_exactly_one_extremum (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! c, a < c ∧ c < b ∧ deriv f c = 0

/-- The cubic function f(x) = x^3 + x^2 - ax - 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + x^2 - a*x - 4

theorem extremum_implies_a_in_open_interval :
  ∀ a : ℝ, has_exactly_one_extremum (f a) (-1) 1 → a ∈ Ioo 1 5 :=
sorry

end NUMINAMATH_CALUDE_extremum_implies_a_in_open_interval_l3790_379063


namespace NUMINAMATH_CALUDE_john_money_left_l3790_379010

/-- Proves that John has $65 left after giving money to his parents -/
theorem john_money_left (initial_amount : ℚ) : 
  initial_amount = 200 →
  initial_amount - (3/8 * initial_amount + 3/10 * initial_amount) = 65 := by
  sorry

end NUMINAMATH_CALUDE_john_money_left_l3790_379010


namespace NUMINAMATH_CALUDE_range_of_x_minus_cosy_l3790_379045

theorem range_of_x_minus_cosy (x y : ℝ) (h : x^2 + 2 * Real.cos y = 1) :
  -1 ≤ x - Real.cos y ∧ x - Real.cos y ≤ 1 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_minus_cosy_l3790_379045


namespace NUMINAMATH_CALUDE_comic_cost_theorem_l3790_379068

/-- Calculates the final cost of each comic book type after discount --/
def final_comic_cost (common_cards : ℕ) (uncommon_cards : ℕ) (rare_cards : ℕ)
  (common_value : ℚ) (uncommon_value : ℚ) (rare_value : ℚ)
  (standard_price : ℚ) (deluxe_price : ℚ) (limited_price : ℚ)
  (discount_threshold_low : ℚ) (discount_threshold_high : ℚ)
  (discount_low : ℚ) (discount_high : ℚ)
  (ratio_standard : ℕ) (ratio_deluxe : ℕ) (ratio_limited : ℕ) : ℚ :=
  sorry

theorem comic_cost_theorem (common_cards : ℕ) (uncommon_cards : ℕ) (rare_cards : ℕ)
  (common_value : ℚ) (uncommon_value : ℚ) (rare_value : ℚ)
  (standard_price : ℚ) (deluxe_price : ℚ) (limited_price : ℚ)
  (discount_threshold_low : ℚ) (discount_threshold_high : ℚ)
  (discount_low : ℚ) (discount_high : ℚ)
  (ratio_standard : ℕ) (ratio_deluxe : ℕ) (ratio_limited : ℕ) :
  common_cards = 1000 ∧ uncommon_cards = 750 ∧ rare_cards = 250 ∧
  common_value = 5/100 ∧ uncommon_value = 1/10 ∧ rare_value = 1/5 ∧
  standard_price = 4 ∧ deluxe_price = 8 ∧ limited_price = 12 ∧
  discount_threshold_low = 100 ∧ discount_threshold_high = 150 ∧
  discount_low = 5/100 ∧ discount_high = 1/10 ∧
  ratio_standard = 3 ∧ ratio_deluxe = 2 ∧ ratio_limited = 1 →
  final_comic_cost common_cards uncommon_cards rare_cards
    common_value uncommon_value rare_value
    standard_price deluxe_price limited_price
    discount_threshold_low discount_threshold_high
    discount_low discount_high
    ratio_standard ratio_deluxe ratio_limited = 6 :=
by sorry

end NUMINAMATH_CALUDE_comic_cost_theorem_l3790_379068


namespace NUMINAMATH_CALUDE_scores_analysis_l3790_379085

def scores : List ℕ := [7, 5, 9, 7, 4, 8, 9, 9, 7, 5]

def mode (l : List ℕ) : Set ℕ := sorry

def variance (l : List ℕ) : ℚ := sorry

def mean (l : List ℕ) : ℚ := sorry

def percentile (l : List ℕ) (p : ℚ) : ℚ := sorry

theorem scores_analysis :
  (mode scores = {7, 9}) ∧
  (variance scores = 3) ∧
  (mean scores = 7) ∧
  (percentile scores (70/100) = 17/2) := by sorry

end NUMINAMATH_CALUDE_scores_analysis_l3790_379085


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l3790_379013

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l3790_379013


namespace NUMINAMATH_CALUDE_sin_shift_l3790_379073

theorem sin_shift (x : ℝ) : Real.sin x = Real.sin (2 * (x - π / 4)) := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_l3790_379073


namespace NUMINAMATH_CALUDE_baseball_hits_percentage_l3790_379040

/-- 
Given a baseball player's hit statistics for a season:
- Total hits: 50
- Home runs: 2
- Triples: 3
- Doubles: 8

This theorem proves that the percentage of hits that were singles is 74%.
-/
theorem baseball_hits_percentage (total_hits home_runs triples doubles : ℕ) 
  (h1 : total_hits = 50)
  (h2 : home_runs = 2)
  (h3 : triples = 3)
  (h4 : doubles = 8) :
  (total_hits - (home_runs + triples + doubles)) / total_hits * 100 = 74 := by
  sorry

#eval (50 - (2 + 3 + 8)) / 50 * 100  -- Should output 74

end NUMINAMATH_CALUDE_baseball_hits_percentage_l3790_379040


namespace NUMINAMATH_CALUDE_map_scale_l3790_379067

/-- Given a map where 15 cm represents 90 km, prove that 20 cm represents 120 km -/
theorem map_scale (map_cm : ℝ) (real_km : ℝ) (h : map_cm / 15 = real_km / 90) :
  (20 * real_km) / map_cm = 120 :=
sorry

end NUMINAMATH_CALUDE_map_scale_l3790_379067


namespace NUMINAMATH_CALUDE_inscribed_circle_and_square_l3790_379088

theorem inscribed_circle_and_square (r : ℝ) (s : ℝ) : 
  -- Circle inscribed in a 3-4-5 right triangle
  r = 1 →
  -- Square concentric with circle and inside it
  s * Real.sqrt 2 = 2 →
  -- Side length of square is √2
  s = Real.sqrt 2 ∧
  -- Area between circle and square is π - 2
  π * r^2 - s^2 = π - 2 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_and_square_l3790_379088


namespace NUMINAMATH_CALUDE_subtract_negative_four_minus_negative_seven_l3790_379081

theorem subtract_negative (a b : ℤ) : a - (-b) = a + b := by sorry

theorem four_minus_negative_seven : 4 - (-7) = 11 := by sorry

end NUMINAMATH_CALUDE_subtract_negative_four_minus_negative_seven_l3790_379081


namespace NUMINAMATH_CALUDE_exam_mean_score_l3790_379095

theorem exam_mean_score (SD : ℝ) :
  (∃ M : ℝ, (58 = M - 2 * SD) ∧ (98 = M + 3 * SD)) → 
  (∃ M : ℝ, (58 = M - 2 * SD) ∧ (98 = M + 3 * SD) ∧ M = 74) :=
by sorry

end NUMINAMATH_CALUDE_exam_mean_score_l3790_379095


namespace NUMINAMATH_CALUDE_ellipse_parameter_sum_l3790_379082

-- Define the foci
def F₁ : ℝ × ℝ := (0, 4)
def F₂ : ℝ × ℝ := (6, 4)

-- Define the ellipse
def Ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  let (x₁, y₁) := F₁
  let (x₂, y₂) := F₂
  Real.sqrt ((x - x₁)^2 + (y - y₁)^2) + Real.sqrt ((x - x₂)^2 + (y - y₂)^2) = 10

-- Define the ellipse equation parameters
def h : ℝ := sorry
def k : ℝ := sorry
def a : ℝ := sorry
def b : ℝ := sorry

-- State the theorem
theorem ellipse_parameter_sum :
  h + k + a + b = 16 := by sorry

end NUMINAMATH_CALUDE_ellipse_parameter_sum_l3790_379082


namespace NUMINAMATH_CALUDE_gcd_factorial_ratio_l3790_379072

theorem gcd_factorial_ratio : Nat.gcd (Nat.factorial 6) ((Nat.factorial 9) / (Nat.factorial 4)) = 480 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_ratio_l3790_379072


namespace NUMINAMATH_CALUDE_M_equals_N_set_order_irrelevant_l3790_379069

-- Define the sets M and N
def M : Set ℕ := {3, 2}
def N : Set ℕ := {2, 3}

-- Theorem stating that M and N are equal
theorem M_equals_N : M = N := by
  sorry

-- Additional theorem to emphasize that order doesn't matter in sets
theorem set_order_irrelevant (A B : Set α) : 
  (∀ x, x ∈ A ↔ x ∈ B) → A = B := by
  sorry

end NUMINAMATH_CALUDE_M_equals_N_set_order_irrelevant_l3790_379069


namespace NUMINAMATH_CALUDE_heejin_has_most_volleyballs_l3790_379051

/-- The number of basketballs Heejin has -/
def basketballs : ℕ := 3

/-- The number of volleyballs Heejin has -/
def volleyballs : ℕ := 5

/-- The number of baseballs Heejin has -/
def baseball : ℕ := 1

/-- Theorem stating that Heejin has more volleyballs than any other type of ball -/
theorem heejin_has_most_volleyballs : 
  volleyballs > basketballs ∧ volleyballs > baseball :=
sorry

end NUMINAMATH_CALUDE_heejin_has_most_volleyballs_l3790_379051


namespace NUMINAMATH_CALUDE_distance_between_foci_l3790_379053

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y - 3)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 25

/-- The first focus of the ellipse -/
def F1 : ℝ × ℝ := (4, 3)

/-- The second focus of the ellipse -/
def F2 : ℝ × ℝ := (-6, 9)

/-- The theorem stating the distance between the foci -/
theorem distance_between_foci :
  Real.sqrt ((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2) = 2 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_foci_l3790_379053


namespace NUMINAMATH_CALUDE_set_A_characterization_l3790_379066

def A : Set ℝ := {a | ∃! x, (x + a) / (x^2 - 1) = 1}

theorem set_A_characterization : A = {-1, 1, -5/4} := by sorry

end NUMINAMATH_CALUDE_set_A_characterization_l3790_379066


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3790_379019

theorem rationalize_denominator :
  ∀ (x : ℝ), x > 0 → (5 / (x^(1/3) + (27 * x)^(1/3))) = (5 * (9 * x)^(1/3)) / 12 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3790_379019


namespace NUMINAMATH_CALUDE_alexander_rearrangements_l3790_379000

theorem alexander_rearrangements (name_length : ℕ) (rearrangements_per_minute : ℕ) : 
  name_length = 9 → rearrangements_per_minute = 15 → 
  (Nat.factorial name_length / rearrangements_per_minute : ℚ) / 60 = 403.2 := by
  sorry

end NUMINAMATH_CALUDE_alexander_rearrangements_l3790_379000


namespace NUMINAMATH_CALUDE_bicycle_sales_theorem_l3790_379055

/-- Represents the sales and pricing data for bicycle types A and B -/
structure BicycleSales where
  lastYearTotalSalesA : ℕ
  priceIncreaseA : ℕ
  purchasePriceA : ℕ
  purchasePriceB : ℕ
  sellingPriceB : ℕ
  totalPurchase : ℕ

/-- Calculates the selling price of type A bicycles this year -/
def sellingPriceA (data : BicycleSales) : ℕ :=
  sorry

/-- Calculates the optimal purchase plan to maximize profit -/
def optimalPurchasePlan (data : BicycleSales) : ℕ × ℕ :=
  sorry

/-- Main theorem stating the selling price of type A bicycles and the optimal purchase plan -/
theorem bicycle_sales_theorem (data : BicycleSales) 
  (h1 : data.lastYearTotalSalesA = 32000)
  (h2 : data.priceIncreaseA = 400)
  (h3 : data.purchasePriceA = 1100)
  (h4 : data.purchasePriceB = 1400)
  (h5 : data.sellingPriceB = 2400)
  (h6 : data.totalPurchase = 50)
  (h7 : ∀ (x y : ℕ), x + y = data.totalPurchase → y ≤ 2 * x) :
  sellingPriceA data = 2000 ∧ optimalPurchasePlan data = (17, 33) :=
sorry

end NUMINAMATH_CALUDE_bicycle_sales_theorem_l3790_379055


namespace NUMINAMATH_CALUDE_greatest_number_with_special_remainder_l3790_379070

theorem greatest_number_with_special_remainder : ∃ n : ℕ, 
  (n % 91 = (n / 91) ^ 2) ∧ 
  (∀ m : ℕ, m > n → m % 91 ≠ (m / 91) ^ 2) ∧
  n = 900 := by
sorry

end NUMINAMATH_CALUDE_greatest_number_with_special_remainder_l3790_379070


namespace NUMINAMATH_CALUDE_ufo_convention_attendees_l3790_379074

theorem ufo_convention_attendees (total : ℕ) (difference : ℕ) : 
  total = 120 → difference = 4 → 
  ∃ (male female : ℕ), 
    male + female = total ∧ 
    male = female + difference ∧ 
    male = 62 := by
  sorry

end NUMINAMATH_CALUDE_ufo_convention_attendees_l3790_379074


namespace NUMINAMATH_CALUDE_flour_needed_proof_l3790_379078

/-- The amount of flour Katie needs in pounds -/
def katie_flour : ℝ := 3

/-- The additional amount of flour Sheila needs compared to Katie in pounds -/
def sheila_additional : ℝ := 2

/-- The multiplier for John's flour needs compared to Sheila's -/
def john_multiplier : ℝ := 1.5

/-- The amount of flour Sheila needs in pounds -/
def sheila_flour : ℝ := katie_flour + sheila_additional

/-- The amount of flour John needs in pounds -/
def john_flour : ℝ := john_multiplier * sheila_flour

/-- The total amount of flour needed by Katie, Sheila, and John -/
def total_flour : ℝ := katie_flour + sheila_flour + john_flour

theorem flour_needed_proof : total_flour = 15.5 := by
  sorry

end NUMINAMATH_CALUDE_flour_needed_proof_l3790_379078


namespace NUMINAMATH_CALUDE_min_occupied_seats_for_150_proof_37_seats_for_150_l3790_379032

/-- Given a row of seats, returns the minimum number of occupied seats required
    to ensure the next person must sit next to someone. -/
def min_occupied_seats (total_seats : ℕ) : ℕ :=
  (total_seats + 3) / 4

theorem min_occupied_seats_for_150 :
  min_occupied_seats 150 = 37 := by
  sorry

/-- Proves that 37 is the minimum number of occupied seats required
    for 150 total seats to ensure the next person sits next to someone. -/
theorem proof_37_seats_for_150 :
  ∀ n : ℕ, n < min_occupied_seats 150 →
    ∃ arrangement : Fin 150 → Bool,
      (∀ i : Fin 150, arrangement i = true → i.val < n) ∧
      ∃ j : Fin 150, (∀ k : Fin 150, k.val = j.val - 1 ∨ k.val = j.val + 1 → arrangement k = false) := by
  sorry

end NUMINAMATH_CALUDE_min_occupied_seats_for_150_proof_37_seats_for_150_l3790_379032


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3790_379050

theorem polynomial_evaluation :
  let f : ℝ → ℝ := λ x ↦ 2*x^4 + 3*x^3 - x^2 + 2*x + 5
  f (-2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3790_379050


namespace NUMINAMATH_CALUDE_gis_not_just_computer_system_l3790_379003

/-- Represents a Geographic Information System (GIS) -/
structure GIS where
  provides_decision_info : Bool
  used_in_urban_management : Bool
  has_data_functions : Bool
  is_computer_system : Bool

/-- The properties of a valid GIS based on the given conditions -/
def is_valid_gis (g : GIS) : Prop :=
  g.provides_decision_info ∧
  g.used_in_urban_management ∧
  g.has_data_functions ∧
  ¬g.is_computer_system

/-- The statement to be proven false -/
def incorrect_statement (g : GIS) : Prop :=
  g.is_computer_system

theorem gis_not_just_computer_system :
  ∃ (g : GIS), is_valid_gis g ∧ ¬incorrect_statement g :=
sorry

end NUMINAMATH_CALUDE_gis_not_just_computer_system_l3790_379003


namespace NUMINAMATH_CALUDE_sum_of_powers_of_two_l3790_379009

theorem sum_of_powers_of_two : 1 + 1/2 + 1/4 + 1/8 = 15/8 := by sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_two_l3790_379009


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3790_379022

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℚ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_a6 : a 6 = 5) 
  (h_a10 : a 10 = 6) : 
  ∃ d : ℚ, d = 1/4 ∧ ∀ n : ℕ, a (n + 1) = a n + d := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3790_379022


namespace NUMINAMATH_CALUDE_quadratic_real_equal_roots_l3790_379080

/-- 
For a quadratic equation of the form 3x^2 + 6kx + 9 = 0, 
the roots are real and equal if and only if k = ± √3.
-/
theorem quadratic_real_equal_roots (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + 6 * k * x + 9 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 + 6 * k * y + 9 = 0 → y = x) ↔ 
  k = Real.sqrt 3 ∨ k = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_equal_roots_l3790_379080


namespace NUMINAMATH_CALUDE_abc_inequality_l3790_379029

theorem abc_inequality (a b c : ℝ) 
  (ha : a < 0) (hb : b < 0) (hc : c < 0)
  (ha' : a > -3) (hb' : b > -3) (hc' : c > -3) : 
  a * b * c > -27 := by
sorry

end NUMINAMATH_CALUDE_abc_inequality_l3790_379029


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3790_379091

/-- Represents a hyperbola with parameters a and b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0
  h_rel : a = Real.sqrt 2 * b

/-- The equation of the asymptotes of a hyperbola -/
def asymptote_equation (h : Hyperbola) : Set (ℝ × ℝ) :=
  {(x, y) | y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x}

/-- Theorem: The asymptotes of the given hyperbola are y = ±√2x -/
theorem hyperbola_asymptotes (h : Hyperbola) : 
  asymptote_equation h = {(x, y) | y^2 / h.a^2 - x^2 / h.b^2 = 1} := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3790_379091


namespace NUMINAMATH_CALUDE_money_distribution_l3790_379093

theorem money_distribution (a b c : ℕ) : 
  a = 3 * b →
  b > c →
  a + b + c = 645 →
  b = 134 →
  b - c = 25 :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_l3790_379093


namespace NUMINAMATH_CALUDE_souvenir_shop_optimal_solution_souvenir_shop_max_profit_l3790_379034

/-- Represents the cost and profit structure for souvenir types A and B -/
structure SouvenirShop where
  cost_A : ℝ
  cost_B : ℝ
  profit_A : ℝ
  profit_B : ℝ

/-- Theorem stating the optimal solution for the souvenir shop problem -/
theorem souvenir_shop_optimal_solution (shop : SouvenirShop) 
  (h1 : 7 * shop.cost_A + 8 * shop.cost_B = 380)
  (h2 : 10 * shop.cost_A + 6 * shop.cost_B = 380)
  (h3 : shop.profit_A = 5)
  (h4 : shop.profit_B = 7) : 
  (shop.cost_A = 20 ∧ shop.cost_B = 30) ∧ 
  (∀ a b : ℕ, a + b = 40 → a * shop.cost_A + b * shop.cost_B ≤ 900 → 
    a * shop.profit_A + b * shop.profit_B ≥ 216 → 
    a * shop.profit_A + b * shop.profit_B ≤ 30 * shop.profit_A + 10 * shop.profit_B) :=
sorry

/-- Corollary stating the maximum profit -/
theorem souvenir_shop_max_profit (shop : SouvenirShop) 
  (h : shop.cost_A = 20 ∧ shop.cost_B = 30 ∧ shop.profit_A = 5 ∧ shop.profit_B = 7) :
  30 * shop.profit_A + 10 * shop.profit_B = 220 :=
sorry

end NUMINAMATH_CALUDE_souvenir_shop_optimal_solution_souvenir_shop_max_profit_l3790_379034


namespace NUMINAMATH_CALUDE_fraction_addition_l3790_379016

theorem fraction_addition (c : ℝ) : (4 + 3 * c) / 7 + 2 = (18 + 3 * c) / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3790_379016


namespace NUMINAMATH_CALUDE_at_least_seven_zeros_l3790_379015

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem at_least_seven_zeros (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period f 3) 
  (h_zero : f 2 = 0) : 
  ∃ (S : Finset ℝ), S.card ≥ 7 ∧ (∀ x ∈ S, 0 < x ∧ x < 6 ∧ f x = 0) :=
sorry

end NUMINAMATH_CALUDE_at_least_seven_zeros_l3790_379015


namespace NUMINAMATH_CALUDE_quadratic_inequality_minimum_l3790_379065

theorem quadratic_inequality_minimum (a b : ℝ) (h1 : a > b) 
  (h2 : ∀ x, (a*x^2 + 2*x + b > 0) ↔ (x ≠ -1/a)) :
  ∃ m : ℝ, m = 6 ∧ ∀ x, x = (a^2 + b^2 + 7)/(a - b) → x ≥ m := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_minimum_l3790_379065


namespace NUMINAMATH_CALUDE_right_triangle_sets_l3790_379094

/-- A function that checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- The theorem stating that among the given sets, only (6, 8, 10) forms a right triangle --/
theorem right_triangle_sets :
  ¬ is_right_triangle 2 3 4 ∧
  ¬ is_right_triangle (Real.sqrt 7) 3 5 ∧
  is_right_triangle 6 8 10 ∧
  ¬ is_right_triangle 5 12 12 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l3790_379094


namespace NUMINAMATH_CALUDE_number_wall_solution_l3790_379033

/-- Represents a number wall with four layers --/
structure NumberWall :=
  (bottom_row : Fin 4 → ℕ)
  (second_row : Fin 3 → ℕ)
  (third_row : Fin 2 → ℕ)
  (top : ℕ)

/-- Checks if a number wall follows the addition rule --/
def is_valid_wall (wall : NumberWall) : Prop :=
  (∀ i : Fin 3, wall.second_row i = wall.bottom_row i + wall.bottom_row (i + 1)) ∧
  (∀ i : Fin 2, wall.third_row i = wall.second_row i + wall.second_row (i + 1)) ∧
  (wall.top = wall.third_row 0 + wall.third_row 1)

/-- The theorem to be proved --/
theorem number_wall_solution (m : ℕ) : 
  (∃ wall : NumberWall, 
    wall.bottom_row = ![m, 4, 10, 9] ∧ 
    wall.top = 52 ∧ 
    is_valid_wall wall) → 
  m = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_wall_solution_l3790_379033


namespace NUMINAMATH_CALUDE_largest_divided_by_smallest_l3790_379002

theorem largest_divided_by_smallest : 
  let numbers : List ℝ := [10, 11, 12, 13]
  (List.maximum numbers).get! / (List.minimum numbers).get! = 1.3 := by
sorry

end NUMINAMATH_CALUDE_largest_divided_by_smallest_l3790_379002


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l3790_379023

theorem gain_percent_calculation (marked_price : ℝ) (marked_price_positive : marked_price > 0) :
  let cost_price := 0.25 * marked_price
  let discount := 0.5 * marked_price
  let selling_price := marked_price - discount
  let gain := selling_price - cost_price
  let gain_percent := (gain / cost_price) * 100
  gain_percent = 100 := by
sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l3790_379023


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l3790_379086

-- Define the diamond operation
def diamond (a b : ℝ) : ℝ := 3 * a + 3 * b^2

-- Theorem statement
theorem diamond_equation_solution :
  ∀ a : ℝ, diamond a 4 = 75 → a = 9 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l3790_379086


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_property_l3790_379058

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Sum of three consecutive terms in a sequence -/
def SumOfThree (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  a n + a (n + 1) + a (n + 2)

theorem geometric_sequence_sum_property
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_sum1 : SumOfThree a 1 = 8)
  (h_sum2 : SumOfThree a 4 = -4) :
  SumOfThree a 7 = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_property_l3790_379058


namespace NUMINAMATH_CALUDE_not_even_and_composite_two_l3790_379031

/-- Definition of an even number -/
def IsEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

/-- Definition of a composite number -/
def IsComposite (n : ℕ) : Prop := ∃ a b : ℕ, a ≠ 1 ∧ b ≠ 1 ∧ n = a * b

/-- Theorem: It is false that 2 is both an even number and a composite number -/
theorem not_even_and_composite_two : ¬(IsEven 2 ∧ IsComposite 2) := by
  sorry

end NUMINAMATH_CALUDE_not_even_and_composite_two_l3790_379031


namespace NUMINAMATH_CALUDE_taxi_ride_distance_is_8_miles_l3790_379020

/-- Calculates the distance of a taxi ride given the fare structure and total charge -/
def taxi_ride_distance (initial_charge : ℚ) (additional_charge : ℚ) (total_charge : ℚ) : ℚ :=
  let remaining_charge := total_charge - initial_charge
  let additional_increments := remaining_charge / additional_charge
  (additional_increments + 1) * (1 / 5)

/-- Proves that the taxi ride distance is 8 miles given the specified fare structure and total charge -/
theorem taxi_ride_distance_is_8_miles :
  let initial_charge : ℚ := 21/10  -- $2.10
  let additional_charge : ℚ := 4/10  -- $0.40
  let total_charge : ℚ := 177/10  -- $17.70
  taxi_ride_distance initial_charge additional_charge total_charge = 8 := by
  sorry

#eval taxi_ride_distance (21/10) (4/10) (177/10)

end NUMINAMATH_CALUDE_taxi_ride_distance_is_8_miles_l3790_379020


namespace NUMINAMATH_CALUDE_min_sum_given_log_condition_l3790_379076

theorem min_sum_given_log_condition (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : Real.log a / Real.log 4 + Real.log b / Real.log 4 ≥ 5) : 
  a + b ≥ 64 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 
    Real.log a₀ / Real.log 4 + Real.log b₀ / Real.log 4 ≥ 5 ∧ a₀ + b₀ = 64 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_given_log_condition_l3790_379076


namespace NUMINAMATH_CALUDE_shopping_solution_l3790_379041

/-- The cost of Liz's shopping trip -/
def shopping_problem (recipe_book_cost : ℝ) : Prop :=
  let baking_dish_cost := 2 * recipe_book_cost
  let ingredients_cost := 5 * 3
  let apron_cost := recipe_book_cost + 1
  recipe_book_cost + baking_dish_cost + ingredients_cost + apron_cost = 40

/-- The solution to the shopping problem -/
theorem shopping_solution : ∃ (recipe_book_cost : ℝ), 
  shopping_problem recipe_book_cost ∧ recipe_book_cost = 6 := by
  sorry

end NUMINAMATH_CALUDE_shopping_solution_l3790_379041


namespace NUMINAMATH_CALUDE_class_receives_reward_l3790_379062

def standard_jumps : ℕ := 160

def performance_records : List ℤ := [16, -1, 20, -2, -5, 11, -7, 6, 9, 13]

def score (x : ℤ) : ℚ :=
  if x ≥ 0 then x
  else -0.5 * x.natAbs

def total_score (records : List ℤ) : ℚ :=
  (records.map score).sum

theorem class_receives_reward (records : List ℤ) :
  records = performance_records →
  total_score records > 65 := by
  sorry

end NUMINAMATH_CALUDE_class_receives_reward_l3790_379062


namespace NUMINAMATH_CALUDE_y_value_l3790_379027

theorem y_value (m : ℕ) (y : ℝ) 
  (h1 : ((1 ^ m) / (y ^ m)) * ((1 ^ 16) / (4 ^ 16)) = 1 / (2 * (10 ^ 31)))
  (h2 : m = 31) : 
  y = 5 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l3790_379027


namespace NUMINAMATH_CALUDE_parallel_planes_theorem_l3790_379014

-- Define the types for our objects
variable (Point Line Plane : Type)

-- Define the relations we need
variable (subset : Line → Plane → Prop)
variable (intersect : Line → Line → Point → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- State the theorem
theorem parallel_planes_theorem 
  (α β : Plane) (a b : Line) (A : Point) :
  subset a α →
  subset b α →
  intersect a b A →
  parallel_line_plane a β →
  parallel_line_plane b β →
  parallel_plane α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_theorem_l3790_379014


namespace NUMINAMATH_CALUDE_line_contains_point_l3790_379087

/-- A line in the xy-plane represented by the equation 2 - kx = 5y -/
def line (k : ℝ) (x y : ℝ) : Prop := 2 - k * x = 5 * y

/-- The point (2, -1) -/
def point : ℝ × ℝ := (2, -1)

/-- Theorem: The line contains the point (2, -1) if and only if k = 7/2 -/
theorem line_contains_point :
  ∀ k : ℝ, line k point.1 point.2 ↔ k = 7/2 := by sorry

end NUMINAMATH_CALUDE_line_contains_point_l3790_379087


namespace NUMINAMATH_CALUDE_sin_1_lt_log_3_sqrt_7_l3790_379052

theorem sin_1_lt_log_3_sqrt_7 :
  ∀ (sin : ℝ → ℝ) (log : ℝ → ℝ → ℝ),
  (0 < 1 ∧ 1 < π/3 ∧ π/3 < π/2) →
  sin (π/3) = Real.sqrt 3 / 2 →
  3^7 < 7^4 →
  sin 1 < log 3 (Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_sin_1_lt_log_3_sqrt_7_l3790_379052


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3790_379054

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 6*x + 5
  ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3790_379054


namespace NUMINAMATH_CALUDE_parallel_to_plane_not_always_parallel_l3790_379018

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallelLine : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- State the theorem
theorem parallel_to_plane_not_always_parallel :
  ∃ (m n : Line) (α : Plane),
    parallelLinePlane m α ∧ parallelLinePlane n α ∧ ¬ parallelLine m n := by
  sorry

end NUMINAMATH_CALUDE_parallel_to_plane_not_always_parallel_l3790_379018


namespace NUMINAMATH_CALUDE_bus_problem_l3790_379061

/-- Calculates the number of children who got on the bus -/
def children_got_on (initial : ℕ) (got_off : ℕ) (final : ℕ) : ℕ :=
  final - (initial - got_off)

/-- Proves that 5 children got on the bus given the initial, final, and number of children who got off -/
theorem bus_problem : children_got_on 21 10 16 = 5 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_l3790_379061


namespace NUMINAMATH_CALUDE_roots_magnitude_l3790_379096

theorem roots_magnitude (p : ℝ) (r₁ r₂ : ℝ) : 
  (r₁ ≠ r₂) →  -- r₁ and r₂ are distinct
  (r₁^2 + p*r₁ + 12 = 0) →  -- r₁ is a root of the equation
  (r₂^2 + p*r₂ + 12 = 0) →  -- r₂ is a root of the equation
  (abs r₁ > 4 ∨ abs r₂ > 4) :=
by sorry

end NUMINAMATH_CALUDE_roots_magnitude_l3790_379096


namespace NUMINAMATH_CALUDE_evaluate_expression_l3790_379083

theorem evaluate_expression (b x : ℝ) (h : x = b + 9) : 2*x - b + 5 = b + 23 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3790_379083


namespace NUMINAMATH_CALUDE_pet_food_cost_differences_l3790_379071

/-- Calculates the total cost including tax -/
def totalCostWithTax (quantity : Float) (price : Float) (taxRate : Float) : Float :=
  quantity * price * (1 + taxRate)

/-- Theorem: Sum of differences between pet food costs -/
theorem pet_food_cost_differences (dogQuantity catQuantity birdQuantity fishQuantity : Float)
  (dogPrice catPrice birdPrice fishPrice : Float) (taxRate : Float)
  (h1 : dogQuantity = 600.5)
  (h2 : catQuantity = 327.25)
  (h3 : birdQuantity = 415.75)
  (h4 : fishQuantity = 248.5)
  (h5 : dogPrice = 24.99)
  (h6 : catPrice = 19.49)
  (h7 : birdPrice = 15.99)
  (h8 : fishPrice = 13.89)
  (h9 : taxRate = 0.065) :
  let dogCost := totalCostWithTax dogQuantity dogPrice taxRate
  let catCost := totalCostWithTax catQuantity catPrice taxRate
  let birdCost := totalCostWithTax birdQuantity birdPrice taxRate
  let fishCost := totalCostWithTax fishQuantity fishPrice taxRate
  (dogCost - catCost) + (catCost - birdCost) + (birdCost - fishCost) = 12301.9002 :=
by sorry

end NUMINAMATH_CALUDE_pet_food_cost_differences_l3790_379071


namespace NUMINAMATH_CALUDE_circle_area_radius_increase_l3790_379090

theorem circle_area_radius_increase : 
  ∀ (r : ℝ) (r' : ℝ), r > 0 → r' > 0 → 
  (π * r' ^ 2 = 4 * π * r ^ 2) → 
  (r' = 2 * r) := by
sorry

end NUMINAMATH_CALUDE_circle_area_radius_increase_l3790_379090
