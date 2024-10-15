import Mathlib

namespace NUMINAMATH_CALUDE_sesame_mass_scientific_notation_l2159_215911

theorem sesame_mass_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.00000201 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.01 ∧ n = -6 :=
sorry

end NUMINAMATH_CALUDE_sesame_mass_scientific_notation_l2159_215911


namespace NUMINAMATH_CALUDE_ceiling_sqrt_224_l2159_215985

theorem ceiling_sqrt_224 : ⌈Real.sqrt 224⌉ = 15 := by sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_224_l2159_215985


namespace NUMINAMATH_CALUDE_isosceles_if_root_one_right_angled_if_equal_roots_l2159_215979

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)

-- Define the quadratic equation
def quadratic_equation (t : Triangle) (x : ℝ) : Prop :=
  (t.a - t.c) * x^2 - 2 * t.b * x + (t.a + t.c) = 0

-- Define isosceles triangle
def is_isosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

-- Define right-angled triangle
def is_right_angled (t : Triangle) : Prop :=
  t.a^2 = t.b^2 + t.c^2 ∨ t.b^2 = t.a^2 + t.c^2 ∨ t.c^2 = t.a^2 + t.b^2

-- Theorem 1
theorem isosceles_if_root_one (t : Triangle) :
  quadratic_equation t 1 → is_isosceles t :=
sorry

-- Theorem 2
theorem right_angled_if_equal_roots (t : Triangle) :
  (∃ x : ℝ, ∀ y : ℝ, quadratic_equation t y ↔ y = x) → is_right_angled t :=
sorry

end NUMINAMATH_CALUDE_isosceles_if_root_one_right_angled_if_equal_roots_l2159_215979


namespace NUMINAMATH_CALUDE_solve_equation_l2159_215930

theorem solve_equation (Q : ℝ) : (Q^3)^(1/2) = 9 * 729^(1/6) → Q = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2159_215930


namespace NUMINAMATH_CALUDE_monotonic_f_implies_a_eq_one_l2159_215959

/-- Piecewise function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ -1 then -x^2 + 2*a else a*x + 4

/-- Theorem stating that if f is monotonic on ℝ, then a = 1 -/
theorem monotonic_f_implies_a_eq_one (a : ℝ) :
  Monotone (f a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_f_implies_a_eq_one_l2159_215959


namespace NUMINAMATH_CALUDE_current_task2_hours_proof_l2159_215946

/-- Calculates the current hours spent on task 2 per day given work conditions -/
def current_task2_hours (total_weekly_hours : ℕ) (work_days : ℕ) (task1_daily_hours : ℕ) (task1_reduction : ℕ) : ℕ :=
  let task1_weekly_hours := task1_daily_hours * work_days
  let new_task1_weekly_hours := task1_weekly_hours - task1_reduction
  let task2_weekly_hours := total_weekly_hours - new_task1_weekly_hours
  task2_weekly_hours / work_days

theorem current_task2_hours_proof :
  current_task2_hours 40 5 5 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_current_task2_hours_proof_l2159_215946


namespace NUMINAMATH_CALUDE_jake_weight_loss_l2159_215909

def jake_weight : ℝ := 93
def total_weight : ℝ := 132

theorem jake_weight_loss : ∃ (x : ℝ), 
  x ≥ 0 ∧ 
  jake_weight - x = 2 * (total_weight - jake_weight) ∧ 
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_jake_weight_loss_l2159_215909


namespace NUMINAMATH_CALUDE_incorrect_combination_not_equivalent_l2159_215988

-- Define the polynomial
def original_polynomial (a b : ℚ) : ℚ := 2 * a * b - 4 * a^2 - 5 * a * b + 9 * a^2

-- Define the incorrect combination
def incorrect_combination (a b : ℚ) : ℚ := (2 * a * b - 5 * a * b) - (4 * a^2 + 9 * a^2)

-- Theorem stating that the incorrect combination is not equivalent to the original polynomial
theorem incorrect_combination_not_equivalent :
  ∃ a b : ℚ, original_polynomial a b ≠ incorrect_combination a b :=
sorry

end NUMINAMATH_CALUDE_incorrect_combination_not_equivalent_l2159_215988


namespace NUMINAMATH_CALUDE_show_length_is_52_hours_l2159_215977

-- Define the number of hours in a day
def hours_in_day : ℕ := 24

-- Define the watching time for each day
def monday_hours : ℕ := hours_in_day / 2
def tuesday_hours : ℕ := 4
def wednesday_hours : ℕ := hours_in_day / 4
def thursday_hours : ℕ := (monday_hours + tuesday_hours + wednesday_hours) / 2
def friday_hours : ℕ := 19

-- Define the total show length
def total_show_length : ℕ := monday_hours + tuesday_hours + wednesday_hours + thursday_hours + friday_hours

-- Theorem to prove
theorem show_length_is_52_hours : total_show_length = 52 := by
  sorry

end NUMINAMATH_CALUDE_show_length_is_52_hours_l2159_215977


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2159_215992

theorem polynomial_remainder (x : ℝ) : 
  let p := fun x : ℝ => 8*x^4 - 10*x^3 + 16*x^2 - 18*x + 5
  let d := fun x : ℝ => 4*x - 8
  (p x) % (d x) = 81 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2159_215992


namespace NUMINAMATH_CALUDE_max_vector_sum_is_6_l2159_215981

-- Define the points in R²
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (0, 4)
def C : ℝ × ℝ := (1, 0)

-- Define the set of points D that satisfy |CD| = 1
def D : Set (ℝ × ℝ) := {d | ‖C - d‖ = 1}

-- Define the vector sum OA + OB + OD
def vectorSum (d : ℝ × ℝ) : ℝ × ℝ := A + B + d

-- Theorem statement
theorem max_vector_sum_is_6 :
  ∃ (m : ℝ), m = 6 ∧ ∀ (d : ℝ × ℝ), d ∈ D → ‖vectorSum d‖ ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_vector_sum_is_6_l2159_215981


namespace NUMINAMATH_CALUDE_solution_product_l2159_215961

theorem solution_product (r s : ℝ) : 
  r ≠ s ∧ 
  (r - 7) * (3 * r + 11) = r^2 - 16 * r + 55 ∧ 
  (s - 7) * (3 * s + 11) = s^2 - 16 * s + 55 →
  (r + 4) * (s + 4) = 25 := by sorry

end NUMINAMATH_CALUDE_solution_product_l2159_215961


namespace NUMINAMATH_CALUDE_stamp_collection_value_l2159_215923

theorem stamp_collection_value (total_stamps : ℕ) (sample_stamps : ℕ) (sample_value : ℚ) 
  (h1 : total_stamps = 18)
  (h2 : sample_stamps = 6)
  (h3 : sample_value = 15) : 
  (total_stamps : ℚ) * (sample_value / sample_stamps) = 45 := by
  sorry

end NUMINAMATH_CALUDE_stamp_collection_value_l2159_215923


namespace NUMINAMATH_CALUDE_word_reduction_l2159_215962

-- Define the alphabet
inductive Letter
| A
| B
| C

-- Define a word as a list of letters
def Word := List Letter

-- Define the equivalence relation
def equivalent : Word → Word → Prop := sorry

-- Define the duplication operation
def duplicate : Word → Word := sorry

-- Define the removal operation
def remove : Word → Word := sorry

-- Main theorem
theorem word_reduction (w : Word) : 
  ∃ (w' : Word), equivalent w w' ∧ w'.length ≤ 8 := by sorry

end NUMINAMATH_CALUDE_word_reduction_l2159_215962


namespace NUMINAMATH_CALUDE_boston_distance_l2159_215954

/-- The distance between Cincinnati and Atlanta in miles -/
def distance_to_atlanta : ℕ := 440

/-- The maximum distance the cyclists can bike in a day -/
def max_daily_distance : ℕ := 40

/-- The number of days it takes to reach Atlanta -/
def days_to_atlanta : ℕ := distance_to_atlanta / max_daily_distance

/-- The distance between Cincinnati and Boston in miles -/
def distance_to_boston : ℕ := days_to_atlanta * max_daily_distance

/-- Theorem stating that the distance to Boston is 440 miles -/
theorem boston_distance : distance_to_boston = 440 := by
  sorry

end NUMINAMATH_CALUDE_boston_distance_l2159_215954


namespace NUMINAMATH_CALUDE_least_possible_smallest_integer_l2159_215996

theorem least_possible_smallest_integer 
  (a b c d e f : ℤ) -- Six different integers
  (h_diff : a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f) -- Integers are different and in ascending order
  (h_median : (c + d) / 2 = 75) -- Median is 75
  (h_largest : f = 120) -- Largest is 120
  (h_smallest_neg : a < 0) -- Smallest is negative
  : a ≥ -1 := by
sorry

end NUMINAMATH_CALUDE_least_possible_smallest_integer_l2159_215996


namespace NUMINAMATH_CALUDE_arithmetic_pattern_l2159_215932

theorem arithmetic_pattern (n : ℕ) : 
  (10^n - 1) * 9 + (n + 1) = 10^(n+1) - 1 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_pattern_l2159_215932


namespace NUMINAMATH_CALUDE_ratio_chain_l2159_215901

theorem ratio_chain (a b c d e : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7)
  (h4 : d / e = 2)
  : e / a = 1 / 17.5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_chain_l2159_215901


namespace NUMINAMATH_CALUDE_probability_multiple_6_or_8_l2159_215967

def is_multiple_of_6_or_8 (n : ℕ) : Bool :=
  n % 6 = 0 ∨ n % 8 = 0

def count_multiples (n : ℕ) : ℕ :=
  (List.range n).filter is_multiple_of_6_or_8 |>.length

theorem probability_multiple_6_or_8 :
  (count_multiples 60 : ℚ) / 60 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_multiple_6_or_8_l2159_215967


namespace NUMINAMATH_CALUDE_solution_set_equality_max_value_g_l2159_215910

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Define the inequality condition
def inequality_condition (x : ℝ) : Prop := f x ≥ 1

-- Define the solution set
def solution_set : Set ℝ := {x | x ≥ 1}

-- Define the function g
def g (x : ℝ) : ℝ := f x - x^2 + x

-- Theorem 1: The solution set of f(x) ≥ 1 is {x | x ≥ 1}
theorem solution_set_equality : 
  {x : ℝ | inequality_condition x} = solution_set := by sorry

-- Theorem 2: The maximum value of g(x) is 5/4
theorem max_value_g : 
  ∃ (x : ℝ), g x = 5/4 ∧ ∀ (y : ℝ), g y ≤ 5/4 := by sorry

end NUMINAMATH_CALUDE_solution_set_equality_max_value_g_l2159_215910


namespace NUMINAMATH_CALUDE_john_total_needed_l2159_215921

/-- The amount of money John has, in dollars. -/
def john_has : ℚ := 0.75

/-- The additional amount John needs, in dollars. -/
def john_needs_more : ℚ := 1.75

/-- The total amount John needs is the sum of what he has and what he needs more. -/
theorem john_total_needed : john_has + john_needs_more = 2.50 := by
  sorry

end NUMINAMATH_CALUDE_john_total_needed_l2159_215921


namespace NUMINAMATH_CALUDE_division_problem_l2159_215936

theorem division_problem (x y z : ℝ) (h1 : x / y = 3) (h2 : y / z = 5/2) : 
  z / x = 2/15 := by sorry

end NUMINAMATH_CALUDE_division_problem_l2159_215936


namespace NUMINAMATH_CALUDE_product_factors_l2159_215960

/-- Given three different natural numbers, each with exactly three factors,
    the product a³b⁴c⁵ has 693 factors. -/
theorem product_factors (a b c : ℕ) (ha : a.factors.length = 3)
    (hb : b.factors.length = 3) (hc : c.factors.length = 3)
    (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
    (a^3 * b^4 * c^5).factors.length = 693 := by
  sorry

end NUMINAMATH_CALUDE_product_factors_l2159_215960


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2159_215983

/-- Given a polynomial g(x) = 3x^4 - 20x^3 + 30x^2 - 35x - 75, prove that g(6) = 363 -/
theorem polynomial_evaluation (x : ℝ) :
  let g := fun x => 3 * x^4 - 20 * x^3 + 30 * x^2 - 35 * x - 75
  g 6 = 363 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2159_215983


namespace NUMINAMATH_CALUDE_manuscript_cost_calculation_l2159_215929

def manuscript_typing_cost (first_typing_rate : ℕ) (revision_rate : ℕ) 
  (total_pages : ℕ) (pages_revised_once : ℕ) (pages_revised_twice : ℕ) : ℕ :=
  let pages_not_revised := total_pages - pages_revised_once - pages_revised_twice
  let first_typing_cost := total_pages * first_typing_rate
  let revision_cost_once := pages_revised_once * revision_rate
  let revision_cost_twice := pages_revised_twice * (2 * revision_rate)
  first_typing_cost + revision_cost_once + revision_cost_twice

theorem manuscript_cost_calculation :
  manuscript_typing_cost 5 4 100 30 20 = 780 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_cost_calculation_l2159_215929


namespace NUMINAMATH_CALUDE_march14_is_tuesday_l2159_215968

/-- 
Represents days of the week.
-/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- 
Represents a specific date in February or March.
-/
structure Date where
  month : Nat
  day : Nat

/-- 
Returns the number of days between two dates, assuming they are in the same year
and the year is not a leap year.
-/
def daysBetween (d1 d2 : Date) : Nat :=
  sorry

/-- 
Returns the day of the week that occurs 'n' days after a given day of the week.
-/
def dayAfter (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  sorry

/-- 
Theorem: If February 14th is on a Tuesday, then March 14th is also on a Tuesday.
-/
theorem march14_is_tuesday (h : dayAfter DayOfWeek.Tuesday 
  (daysBetween ⟨2, 14⟩ ⟨3, 14⟩) = DayOfWeek.Tuesday) :
  dayAfter DayOfWeek.Tuesday (daysBetween ⟨2, 14⟩ ⟨3, 14⟩) = DayOfWeek.Tuesday := by
  sorry

end NUMINAMATH_CALUDE_march14_is_tuesday_l2159_215968


namespace NUMINAMATH_CALUDE_inequalities_for_positive_reals_l2159_215978

theorem inequalities_for_positive_reals (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (1 / (a * c) + a / (b^2 * c) + b * c ≥ 2 * Real.sqrt 2) ∧
  (a + b + c ≥ Real.sqrt (2 * a * b) + Real.sqrt (2 * a * c)) ∧
  (a^2 + b^2 + c^2 ≥ 2 * a * b + 2 * b * c - 2 * a * c) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_for_positive_reals_l2159_215978


namespace NUMINAMATH_CALUDE_sum_of_base3_digits_345_l2159_215957

/-- Converts a natural number to its base-3 representation -/
def toBase3 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 3) :: aux (m / 3)
  aux n

/-- Sums the digits in a list of natural numbers -/
def sumDigits (digits : List ℕ) : ℕ :=
  digits.foldl (·+·) 0

theorem sum_of_base3_digits_345 :
  sumDigits (toBase3 345) = 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_base3_digits_345_l2159_215957


namespace NUMINAMATH_CALUDE_bank_account_difference_l2159_215989

theorem bank_account_difference (bob_amount jenna_amount phil_amount : ℝ) : 
  bob_amount = 60 →
  phil_amount = (1/3) * bob_amount →
  jenna_amount = 2 * phil_amount →
  bob_amount - jenna_amount = 20 := by
sorry

end NUMINAMATH_CALUDE_bank_account_difference_l2159_215989


namespace NUMINAMATH_CALUDE_max_value_polynomial_l2159_215948

theorem max_value_polynomial (a b : ℝ) (h : a + b = 4) :
  (∃ x y : ℝ, x + y = 4 ∧ 
    a^4*b + a^3*b + a^2*b + a*b + a*b^2 + a*b^3 + a*b^4 ≤ 
    x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4) ∧
  (∀ x y : ℝ, x + y = 4 → 
    x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 ≤ 7225/56) ∧
  (∃ x y : ℝ, x + y = 4 ∧ 
    x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 = 7225/56) :=
by sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l2159_215948


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2159_215966

theorem inequality_equivalence (x : ℝ) : x - 1 ≤ (1 + x) / 3 ↔ x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2159_215966


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_min_value_achievable_l2159_215900

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_one : x + y + z = 1) : 
  1/x + 4/y + 9/z ≥ 36 :=
by
  sorry

theorem min_value_achievable : 
  ∃ (x y z : ℝ), 0 < x ∧ 0 < y ∧ 0 < z ∧ x + y + z = 1 ∧ 1/x + 4/y + 9/z = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_min_value_achievable_l2159_215900


namespace NUMINAMATH_CALUDE_right_side_difference_l2159_215924

/-- A triangle with specific side lengths -/
structure Triangle where
  left : ℝ
  right : ℝ
  base : ℝ

/-- The properties of our specific triangle -/
def special_triangle (t : Triangle) : Prop :=
  t.left = 12 ∧ 
  t.base = 24 ∧ 
  t.left + t.right + t.base = 50 ∧
  t.right > t.left

theorem right_side_difference (t : Triangle) (h : special_triangle t) : 
  t.right - t.left = 2 := by
  sorry

end NUMINAMATH_CALUDE_right_side_difference_l2159_215924


namespace NUMINAMATH_CALUDE_max_distance_complex_l2159_215999

theorem max_distance_complex (z : ℂ) (h : Complex.abs z = 3) :
  ∃ (max_dist : ℝ), max_dist = (36 * Real.sqrt 26) / 5 ∧
  ∀ w : ℂ, Complex.abs w = 3 → Complex.abs ((2 + Complex.I) * w^2 - w^4) ≤ max_dist :=
by sorry

end NUMINAMATH_CALUDE_max_distance_complex_l2159_215999


namespace NUMINAMATH_CALUDE_grass_seed_bags_for_park_lot_l2159_215974

/-- Calculates the number of grass seed bags needed for a rectangular lot with a concrete section --/
def grassSeedBags (lotLength lotWidth concreteSize seedCoverage : ℕ) : ℕ :=
  let totalArea := lotLength * lotWidth
  let concreteArea := concreteSize * concreteSize
  let grassyArea := totalArea - concreteArea
  (grassyArea + seedCoverage - 1) / seedCoverage

/-- Theorem stating that 100 bags of grass seeds are needed for the given lot specifications --/
theorem grass_seed_bags_for_park_lot : 
  grassSeedBags 120 60 40 56 = 100 := by
  sorry

#eval grassSeedBags 120 60 40 56

end NUMINAMATH_CALUDE_grass_seed_bags_for_park_lot_l2159_215974


namespace NUMINAMATH_CALUDE_complement_of_A_l2159_215920

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}

theorem complement_of_A (A B : Set ℕ) 
  (h1 : A ∪ B = {1, 2, 3, 4, 5})
  (h2 : A ∩ B = {3, 4, 5}) :
  Aᶜ = {6} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l2159_215920


namespace NUMINAMATH_CALUDE_calculation_proof_l2159_215935

theorem calculation_proof : 
  (5^(2/3) - 5^(3/2)) / 5^(1/2) = 60 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2159_215935


namespace NUMINAMATH_CALUDE_arithmetic_sequence_index_l2159_215994

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

theorem arithmetic_sequence_index :
  ∀ n : ℕ,
  arithmetic_sequence 1 3 n = 2014 → n = 672 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_index_l2159_215994


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_negation_of_square_not_equal_self_l2159_215964

theorem negation_of_universal_proposition (p : ℝ → Prop) :
  (¬∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬p x) :=
by sorry

theorem negation_of_square_not_equal_self :
  (¬∀ x : ℝ, x^2 ≠ x) ↔ (∃ x : ℝ, x^2 = x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_negation_of_square_not_equal_self_l2159_215964


namespace NUMINAMATH_CALUDE_central_cell_value_l2159_215913

theorem central_cell_value (a b c d e f g h i : ℝ) 
  (row_prod : a * b * c = 10 ∧ d * e * f = 10 ∧ g * h * i = 10)
  (col_prod : a * d * g = 10 ∧ b * e * h = 10 ∧ c * f * i = 10)
  (square_prod : a * b * d * e = 3 ∧ b * c * e * f = 3 ∧ d * e * g * h = 3 ∧ e * f * h * i = 3) :
  e = 0.00081 := by
sorry

end NUMINAMATH_CALUDE_central_cell_value_l2159_215913


namespace NUMINAMATH_CALUDE_altons_rental_cost_l2159_215984

/-- Calculates the weekly rental cost for Alton's business. -/
theorem altons_rental_cost
  (daily_earnings : ℝ)  -- Alton's daily earnings
  (weekly_profit : ℝ)   -- Alton's weekly profit
  (h1 : daily_earnings = 8)  -- Alton earns $8 per day
  (h2 : weekly_profit = 36)  -- Alton's total profit every week is $36
  : ∃ (rental_cost : ℝ), rental_cost = 20 ∧ 7 * daily_earnings - rental_cost = weekly_profit :=
by
  sorry

#check altons_rental_cost

end NUMINAMATH_CALUDE_altons_rental_cost_l2159_215984


namespace NUMINAMATH_CALUDE_greatest_multiple_under_1000_l2159_215927

theorem greatest_multiple_under_1000 : ∃ (n : ℕ), n = 945 ∧ 
  n < 1000 ∧ 
  3 ∣ n ∧ 
  5 ∣ n ∧ 
  7 ∣ n ∧ 
  ∀ m : ℕ, m < 1000 ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_under_1000_l2159_215927


namespace NUMINAMATH_CALUDE_six_grades_assignments_l2159_215902

/-- The number of ways to assign n grades, where grades are 2, 3, or 4, and no two consecutive 2s are allowed. -/
def gradeAssignments (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 3
  | n + 2 => 2 * gradeAssignments (n + 1) + 2 * gradeAssignments n

/-- The theorem stating that there are 448 ways to assign 6 grades under the given conditions. -/
theorem six_grades_assignments : gradeAssignments 6 = 448 := by
  sorry

end NUMINAMATH_CALUDE_six_grades_assignments_l2159_215902


namespace NUMINAMATH_CALUDE_max_ratio_three_digit_number_l2159_215931

theorem max_ratio_three_digit_number :
  ∀ (a b c : ℕ),
    1 ≤ a ∧ a ≤ 9 →
    0 ≤ b ∧ b ≤ 9 →
    0 ≤ c ∧ c ≤ 9 →
    let N := 100 * a + 10 * b + c
    let S := a + b + c
    (N : ℚ) / S ≤ 100 ∧ 
    (∃ a' b' c', 
      1 ≤ a' ∧ a' ≤ 9 ∧ 
      0 ≤ b' ∧ b' ≤ 9 ∧ 
      0 ≤ c' ∧ c' ≤ 9 ∧ 
      let N' := 100 * a' + 10 * b' + c'
      let S' := a' + b' + c'
      (N' : ℚ) / S' = 100) :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_three_digit_number_l2159_215931


namespace NUMINAMATH_CALUDE_fuel_fraction_proof_l2159_215925

def road_trip_fuel_calculation (total_fuel : ℝ) (first_third : ℝ) (second_third_fraction : ℝ) : Prop :=
  let second_third := total_fuel * second_third_fraction
  let final_third := total_fuel - first_third - second_third
  final_third / second_third = 1 / 2

theorem fuel_fraction_proof :
  road_trip_fuel_calculation 60 30 (1/3) :=
by
  sorry

end NUMINAMATH_CALUDE_fuel_fraction_proof_l2159_215925


namespace NUMINAMATH_CALUDE_complex_equation_result_l2159_215969

theorem complex_equation_result (a b : ℝ) (h : (a + 4 * Complex.I) * Complex.I = b + Complex.I) : a - b = -5 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_result_l2159_215969


namespace NUMINAMATH_CALUDE_g_sum_property_l2159_215971

def g (x : ℝ) : ℝ := 2 * x^8 + 3 * x^6 - 4 * x^4 + 5

theorem g_sum_property : g 5 = 7 → g 5 + g (-5) = 14 := by sorry

end NUMINAMATH_CALUDE_g_sum_property_l2159_215971


namespace NUMINAMATH_CALUDE_opposite_expressions_l2159_215937

theorem opposite_expressions (x : ℝ) : (4 * x - 8 = -(3 * x - 6)) ↔ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_expressions_l2159_215937


namespace NUMINAMATH_CALUDE_complex_modulus_sqrt_5_l2159_215919

theorem complex_modulus_sqrt_5 (z : ℂ) (x y : ℝ) :
  z = x + y * I →
  x / (1 - I) = 1 + y * I →
  Complex.abs z = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_sqrt_5_l2159_215919


namespace NUMINAMATH_CALUDE_airplane_passengers_l2159_215958

theorem airplane_passengers (total_passengers men : ℕ) 
  (h1 : total_passengers = 80)
  (h2 : men = 30)
  (h3 : ∃ women : ℕ, women = men) :
  ∃ children : ℕ, children = 20 ∧ total_passengers = men + men + children :=
by sorry

end NUMINAMATH_CALUDE_airplane_passengers_l2159_215958


namespace NUMINAMATH_CALUDE_sallys_peaches_l2159_215917

/-- Given that Sally had 13 peaches initially and ended up with 55 peaches,
    prove that she picked 42 peaches. -/
theorem sallys_peaches (initial : ℕ) (final : ℕ) (h1 : initial = 13) (h2 : final = 55) :
  final - initial = 42 := by sorry

end NUMINAMATH_CALUDE_sallys_peaches_l2159_215917


namespace NUMINAMATH_CALUDE_buffet_price_theorem_l2159_215947

/-- Represents the price of an adult buffet ticket -/
def adult_price : ℝ := 30

/-- Represents the price of a child buffet ticket -/
def child_price : ℝ := 15

/-- Represents the discount rate for senior citizens -/
def senior_discount : ℝ := 0.1

/-- Calculates the total cost for the family's buffet -/
def total_cost (adult_price : ℝ) : ℝ :=
  2 * adult_price +  -- Cost for 2 adults
  2 * (1 - senior_discount) * adult_price +  -- Cost for 2 senior citizens
  3 * child_price  -- Cost for 3 children

theorem buffet_price_theorem :
  total_cost adult_price = 159 :=
by sorry

end NUMINAMATH_CALUDE_buffet_price_theorem_l2159_215947


namespace NUMINAMATH_CALUDE_batsmans_average_increase_l2159_215995

theorem batsmans_average_increase 
  (score_17th : ℕ) 
  (average_after_17th : ℚ) 
  (h1 : score_17th = 66) 
  (h2 : average_after_17th = 18) : 
  average_after_17th - (((17 : ℕ) * average_after_17th - score_17th) / 16 : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_batsmans_average_increase_l2159_215995


namespace NUMINAMATH_CALUDE_ivans_age_l2159_215956

-- Define the age components
def years : ℕ := 48
def months : ℕ := 48
def weeks : ℕ := 48
def days : ℕ := 48
def hours : ℕ := 48

-- Define conversion factors
def monthsPerYear : ℕ := 12
def daysPerWeek : ℕ := 7
def daysPerYear : ℕ := 365
def hoursPerDay : ℕ := 24

-- Theorem to prove
theorem ivans_age : 
  (years + months / monthsPerYear + 
   (weeks * daysPerWeek + days + hours / hoursPerDay) / daysPerYear) = 53 := by
  sorry

end NUMINAMATH_CALUDE_ivans_age_l2159_215956


namespace NUMINAMATH_CALUDE_aquarium_animals_l2159_215993

theorem aquarium_animals (num_aquariums : ℕ) (total_animals : ℕ) 
  (h1 : num_aquariums = 26)
  (h2 : total_animals = 52)
  (h3 : ∃ (animals_per_aquarium : ℕ), 
    animals_per_aquarium > 1 ∧ 
    animals_per_aquarium % 2 = 1 ∧
    num_aquariums * animals_per_aquarium = total_animals) :
  ∃ (animals_per_aquarium : ℕ), 
    animals_per_aquarium = 13 ∧
    animals_per_aquarium > 1 ∧ 
    animals_per_aquarium % 2 = 1 ∧
    num_aquariums * animals_per_aquarium = total_animals :=
by sorry

end NUMINAMATH_CALUDE_aquarium_animals_l2159_215993


namespace NUMINAMATH_CALUDE_divisibility_problem_l2159_215997

/-- A number is a five-digit number if it's between 10000 and 99999 -/
def IsFiveDigit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- A number starts with 4 if its first digit is 4 -/
def StartsWithFour (n : ℕ) : Prop := ∃ k, n = 40000 + k ∧ k < 10000

/-- A number ends with 7 if its last digit is 7 -/
def EndsWithSeven (n : ℕ) : Prop := ∃ k, n = 10 * k + 7

/-- A number starts with 9 if its first digit is 9 -/
def StartsWithNine (n : ℕ) : Prop := ∃ k, n = 90000 + k ∧ k < 10000

/-- A number ends with 3 if its last digit is 3 -/
def EndsWithThree (n : ℕ) : Prop := ∃ k, n = 10 * k + 3

theorem divisibility_problem (x y z : ℕ) 
  (hx_five : IsFiveDigit x) (hy_five : IsFiveDigit y) (hz_five : IsFiveDigit z)
  (hx_start : StartsWithFour x) (hx_end : EndsWithSeven x)
  (hy_start : StartsWithNine y) (hy_end : EndsWithThree y)
  (hxz : z ∣ x) (hyz : z ∣ y) : 
  11 ∣ (2 * y - x) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l2159_215997


namespace NUMINAMATH_CALUDE_girls_percentage_is_60_percent_l2159_215973

/-- Represents the number of students in the school -/
def total_students : ℕ := 150

/-- Represents the number of boys who did not join varsity clubs -/
def boys_not_in_varsity : ℕ := 40

/-- Represents the fraction of boys who joined varsity clubs -/
def boys_varsity_fraction : ℚ := 1/3

/-- Calculates the percentage of girls in the school -/
def girls_percentage : ℚ :=
  let total_boys : ℕ := boys_not_in_varsity * 3 / 2
  let total_girls : ℕ := total_students - total_boys
  (total_girls : ℚ) / total_students * 100

/-- Theorem stating that the percentage of girls in the school is 60% -/
theorem girls_percentage_is_60_percent : girls_percentage = 60 := by
  sorry

end NUMINAMATH_CALUDE_girls_percentage_is_60_percent_l2159_215973


namespace NUMINAMATH_CALUDE_gcd_1037_425_l2159_215945

theorem gcd_1037_425 : Nat.gcd 1037 425 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1037_425_l2159_215945


namespace NUMINAMATH_CALUDE_factorial_difference_l2159_215986

theorem factorial_difference : Nat.factorial 8 - Nat.factorial 7 = 35280 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l2159_215986


namespace NUMINAMATH_CALUDE_quadratic_root_range_l2159_215941

theorem quadratic_root_range (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    a * x₁^2 - 2*(a+1)*x₁ + (a-1) = 0 ∧
    a * x₂^2 - 2*(a+1)*x₂ + (a-1) = 0 ∧
    x₁ > 2 ∧ x₂ < 2) →
  (0 < a ∧ a < 5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l2159_215941


namespace NUMINAMATH_CALUDE_cone_base_circumference_l2159_215943

theorem cone_base_circumference (r : ℝ) (sector_angle : ℝ) : 
  r = 6 → sector_angle = 300 → 
  2 * π * r * (360 - sector_angle) / 360 = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l2159_215943


namespace NUMINAMATH_CALUDE_monkeys_count_l2159_215914

theorem monkeys_count (termites : ℕ) (total_workers : ℕ) (h1 : termites = 622) (h2 : total_workers = 861) :
  total_workers - termites = 239 := by
  sorry

end NUMINAMATH_CALUDE_monkeys_count_l2159_215914


namespace NUMINAMATH_CALUDE_franks_age_l2159_215953

theorem franks_age (frank_age : ℕ) (gabriel_age : ℕ) : 
  gabriel_age = frank_age - 3 →
  frank_age + gabriel_age = 17 →
  frank_age = 10 := by
sorry

end NUMINAMATH_CALUDE_franks_age_l2159_215953


namespace NUMINAMATH_CALUDE_green_marble_probability_l2159_215975

theorem green_marble_probability (total : ℕ) (p_white p_red_or_blue : ℚ) : 
  total = 84 →
  p_white = 1/4 →
  p_red_or_blue = 463/1000 →
  (total : ℚ) * (1 - p_white - p_red_or_blue) / total = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_green_marble_probability_l2159_215975


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l2159_215907

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : GeometricSequence a)
  (h_relation : a 7 = a 6 + 2 * a 5)
  (h_exist : ∃ m n : ℕ, Real.sqrt (a m * a n) = 2 * Real.sqrt 2 * a 1) :
  (∃ m n : ℕ, 1 / m + 4 / n = 11 / 6) ∧
  (∀ m n : ℕ, 1 / m + 4 / n ≥ 11 / 6) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l2159_215907


namespace NUMINAMATH_CALUDE_ratio_of_linear_system_l2159_215926

theorem ratio_of_linear_system (x y a b : ℝ) (h1 : 4 * x - 2 * y = a) 
  (h2 : 5 * y - 10 * x = b) (h3 : b ≠ 0) : a / b = -1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_linear_system_l2159_215926


namespace NUMINAMATH_CALUDE_building_floor_ratio_l2159_215912

/-- Given three buildings A, B, and C, where:
  * Building A has 4 floors
  * Building B has 9 more floors than Building A
  * Building C has 59 floors
Prove that the ratio of floors in Building C to Building B is 59/13 -/
theorem building_floor_ratio : 
  (floors_A : ℕ) → 
  (floors_B : ℕ) → 
  (floors_C : ℕ) → 
  floors_A = 4 →
  floors_B = floors_A + 9 →
  floors_C = 59 →
  (floors_C : ℚ) / floors_B = 59 / 13 := by
sorry

end NUMINAMATH_CALUDE_building_floor_ratio_l2159_215912


namespace NUMINAMATH_CALUDE_certain_time_in_seconds_l2159_215955

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- The given time in minutes -/
def given_time : ℕ := 4

/-- The certain time in seconds -/
def certain_time : ℕ := given_time * seconds_per_minute

theorem certain_time_in_seconds : certain_time = 240 := by
  sorry

end NUMINAMATH_CALUDE_certain_time_in_seconds_l2159_215955


namespace NUMINAMATH_CALUDE_square_circle_radius_l2159_215972

/-- Given a square with a circumscribed circle, if the sum of the lengths of all sides
    of the square equals the area of the circumscribed circle, then the radius of the
    circle is 4√2/π. -/
theorem square_circle_radius (s : ℝ) (r : ℝ) (h : s > 0) (h' : r > 0) :
  4 * s = π * r^2 → r = 4 * Real.sqrt 2 / π :=
by sorry

end NUMINAMATH_CALUDE_square_circle_radius_l2159_215972


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_a_geq_3_l2159_215903

theorem empty_solution_set_implies_a_geq_3 (a : ℝ) : 
  (∀ x : ℝ, ¬((x - 2) / 5 + 2 > x - 4 / 5 ∧ x > a)) → a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_implies_a_geq_3_l2159_215903


namespace NUMINAMATH_CALUDE_cow_chicken_problem_l2159_215963

theorem cow_chicken_problem (C H : ℕ) : 
  4 * C + 2 * H = 2 * (C + H) + 12 → C = 6 := by
  sorry

end NUMINAMATH_CALUDE_cow_chicken_problem_l2159_215963


namespace NUMINAMATH_CALUDE_duke_record_breaking_l2159_215933

/-- Duke's basketball record breaking proof --/
theorem duke_record_breaking (points_to_tie : ℕ) (old_record : ℕ) 
  (free_throws : ℕ) (regular_baskets : ℕ) (normal_three_pointers : ℕ) :
  points_to_tie = 17 →
  old_record = 257 →
  free_throws = 5 →
  regular_baskets = 4 →
  normal_three_pointers = 2 →
  (free_throws * 1 + regular_baskets * 2 + (normal_three_pointers + 1) * 3) - points_to_tie = 5 := by
  sorry

#check duke_record_breaking

end NUMINAMATH_CALUDE_duke_record_breaking_l2159_215933


namespace NUMINAMATH_CALUDE_loan_amount_proof_l2159_215950

/-- Calculates the total loan amount given the loan terms -/
def total_loan_amount (down_payment : ℕ) (monthly_payment : ℕ) (years : ℕ) : ℕ :=
  down_payment + monthly_payment * years * 12

/-- Proves that the total loan amount is correct given the specified conditions -/
theorem loan_amount_proof (down_payment monthly_payment years : ℕ) 
  (h1 : down_payment = 10000)
  (h2 : monthly_payment = 600)
  (h3 : years = 5) :
  total_loan_amount down_payment monthly_payment years = 46000 := by
  sorry

end NUMINAMATH_CALUDE_loan_amount_proof_l2159_215950


namespace NUMINAMATH_CALUDE_vector_subtraction_scalar_multiplication_l2159_215940

theorem vector_subtraction_scalar_multiplication :
  (3 : ℝ) • (((⟨-3, 2, -5⟩ : ℝ × ℝ × ℝ) - ⟨1, 6, 2⟩) : ℝ × ℝ × ℝ) = ⟨-12, -12, -21⟩ := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_scalar_multiplication_l2159_215940


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l2159_215904

theorem gcd_lcm_product (a b : ℕ) : Nat.gcd a b * Nat.lcm a b = a * b := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l2159_215904


namespace NUMINAMATH_CALUDE_years_between_second_and_third_car_l2159_215939

def year_first_car : ℕ := 1970
def years_between_first_and_second : ℕ := 10
def year_third_car : ℕ := 2000

theorem years_between_second_and_third_car : 
  year_third_car - (year_first_car + years_between_first_and_second) = 20 := by
  sorry

end NUMINAMATH_CALUDE_years_between_second_and_third_car_l2159_215939


namespace NUMINAMATH_CALUDE_peter_has_25_candies_l2159_215987

/-- The number of candies each person has after sharing equally -/
def shared_candies : ℕ := 30

/-- The number of candies Mark has -/
def mark_candies : ℕ := 30

/-- The number of candies John has -/
def john_candies : ℕ := 35

/-- The number of candies Peter has -/
def peter_candies : ℕ := shared_candies * 3 - mark_candies - john_candies

theorem peter_has_25_candies : peter_candies = 25 := by
  sorry

end NUMINAMATH_CALUDE_peter_has_25_candies_l2159_215987


namespace NUMINAMATH_CALUDE_dog_burrs_problem_l2159_215918

theorem dog_burrs_problem (burrs ticks : ℕ) : 
  ticks = 6 * burrs → 
  burrs + ticks = 84 → 
  burrs = 12 := by sorry

end NUMINAMATH_CALUDE_dog_burrs_problem_l2159_215918


namespace NUMINAMATH_CALUDE_probability_failed_chinese_given_failed_math_l2159_215938

theorem probability_failed_chinese_given_failed_math 
  (total_students : ℕ) 
  (failed_math : ℕ) 
  (failed_chinese : ℕ) 
  (failed_both : ℕ) 
  (h1 : failed_math = (25 : ℕ) * total_students / 100)
  (h2 : failed_chinese = (10 : ℕ) * total_students / 100)
  (h3 : failed_both = (5 : ℕ) * total_students / 100)
  (h4 : total_students > 0) :
  (failed_both : ℚ) / failed_math = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_failed_chinese_given_failed_math_l2159_215938


namespace NUMINAMATH_CALUDE_not_square_among_powers_l2159_215934

theorem not_square_among_powers : 
  (∃ n : ℕ, 1^6 = n^2) ∧
  (∃ n : ℕ, 3^4 = n^2) ∧
  (∃ n : ℕ, 4^3 = n^2) ∧
  (∃ n : ℕ, 5^2 = n^2) ∧
  (¬ ∃ n : ℕ, 2^5 = n^2) := by
  sorry

end NUMINAMATH_CALUDE_not_square_among_powers_l2159_215934


namespace NUMINAMATH_CALUDE_cuboid_diagonal_range_l2159_215990

theorem cuboid_diagonal_range (d1 d2 x : ℝ) :
  d1 = 5 →
  d2 = 4 →
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
    a^2 + b^2 = d1^2 ∧
    a^2 + c^2 = d2^2 ∧
    b^2 + c^2 = x^2) →
  3 < x ∧ x < Real.sqrt 41 := by
sorry

end NUMINAMATH_CALUDE_cuboid_diagonal_range_l2159_215990


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2159_215942

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, 2 * x^3 + x * y - 7 = 0 ↔ 
    (x = -7 ∧ y = -99) ∨ (x = -1 ∧ y = -9) ∨ (x = 1 ∧ y = 5) ∨ (x = 7 ∧ y = -97) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2159_215942


namespace NUMINAMATH_CALUDE_star_five_two_l2159_215951

def star (a b : ℚ) : ℚ := a^2 + a/b

theorem star_five_two : star 5 2 = 55/2 := by
  sorry

end NUMINAMATH_CALUDE_star_five_two_l2159_215951


namespace NUMINAMATH_CALUDE_sum_of_two_primes_10003_l2159_215952

/-- A function that returns true if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- The number of ways 10003 can be written as the sum of two primes -/
theorem sum_of_two_primes_10003 :
  ∃! (p q : ℕ), isPrime p ∧ isPrime q ∧ p + q = 10003 :=
sorry

end NUMINAMATH_CALUDE_sum_of_two_primes_10003_l2159_215952


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l2159_215944

/-- The minimum value of a quadratic function -/
theorem quadratic_minimum_value (a k c : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + (a + k) * x + c
  ∃ m : ℝ, (∀ x, f x ≥ m) ∧ (m = (-a^2 - 2*a*k - k^2 + 4*a*c) / (4*a)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l2159_215944


namespace NUMINAMATH_CALUDE_ball_count_l2159_215991

theorem ball_count (white green yellow red purple : ℕ)
  (h1 : white = 22)
  (h2 : green = 18)
  (h3 : yellow = 2)
  (h4 : red = 15)
  (h5 : purple = 3)
  (h6 : (white + green + yellow : ℚ) / (white + green + yellow + red + purple) = 7/10) :
  white + green + yellow + red + purple = 60 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_l2159_215991


namespace NUMINAMATH_CALUDE_product_inspection_probability_l2159_215916

theorem product_inspection_probability : 
  let p_good_as_defective : ℝ := 0.02
  let p_defective_as_good : ℝ := 0.01
  let num_good : ℕ := 3
  let num_defective : ℕ := 1
  let p_correct_good : ℝ := 1 - p_good_as_defective
  let p_correct_defective : ℝ := 1 - p_defective_as_good
  (p_correct_good ^ num_good) * (p_correct_defective ^ num_defective) = 0.932 :=
by sorry

end NUMINAMATH_CALUDE_product_inspection_probability_l2159_215916


namespace NUMINAMATH_CALUDE_ninth_grade_classes_l2159_215998

theorem ninth_grade_classes (total_matches : ℕ) (h : total_matches = 28) :
  ∃ x : ℕ, x * (x - 1) / 2 = total_matches ∧ x = 8 :=
by sorry

end NUMINAMATH_CALUDE_ninth_grade_classes_l2159_215998


namespace NUMINAMATH_CALUDE_tan_45_degrees_l2159_215976

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l2159_215976


namespace NUMINAMATH_CALUDE_total_taco_combinations_l2159_215949

/-- The number of optional toppings available for tacos. -/
def num_toppings : ℕ := 8

/-- The number of meat options available for tacos. -/
def meat_options : ℕ := 3

/-- The number of shell options available for tacos. -/
def shell_options : ℕ := 2

/-- Calculates the total number of different taco combinations. -/
def taco_combinations : ℕ := 2^num_toppings * meat_options * shell_options

/-- Theorem stating that the total number of taco combinations is 1536. -/
theorem total_taco_combinations : taco_combinations = 1536 := by
  sorry

end NUMINAMATH_CALUDE_total_taco_combinations_l2159_215949


namespace NUMINAMATH_CALUDE_simplify_expression_constant_sum_l2159_215928

/-- Given expressions for A and B in terms of a and b -/
def A (a b : ℝ) : ℝ := 2 * a^2 + a * b - 2 * b - 1

/-- Given expressions for A and B in terms of a and b -/
def B (a b : ℝ) : ℝ := -a^2 + a * b - 2

/-- Theorem 1: Simplification of 3A - (2A - 2B) -/
theorem simplify_expression (a b : ℝ) :
  3 * A a b - (2 * A a b - 2 * B a b) = 3 * a * b - 2 * b - 5 := by sorry

/-- Theorem 2: Value of a when A + 2B is constant for any b -/
theorem constant_sum (a : ℝ) :
  (∀ b : ℝ, ∃ k : ℝ, A a b + 2 * B a b = k) → a = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_constant_sum_l2159_215928


namespace NUMINAMATH_CALUDE_simplify_expression_l2159_215965

theorem simplify_expression : (4 + 3) + (8 - 3 - 1) = 11 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2159_215965


namespace NUMINAMATH_CALUDE_base7_to_base10_conversion_l2159_215905

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base 7 representation of the number --/
def base7Number : List Nat := [6, 5, 4, 3, 2]

theorem base7_to_base10_conversion :
  base7ToBase10 base7Number = 6068 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_conversion_l2159_215905


namespace NUMINAMATH_CALUDE_two_axes_implies_center_symmetry_l2159_215980

/-- A figure in a plane --/
structure Figure where
  -- Add necessary fields here
  -- This is just a placeholder structure

/-- Represents an axis of symmetry for a figure --/
structure AxisOfSymmetry where
  -- Add necessary fields here
  -- This is just a placeholder structure

/-- Represents a center of symmetry for a figure --/
structure CenterOfSymmetry where
  -- Add necessary fields here
  -- This is just a placeholder structure

/-- A function to determine if a figure has exactly two axes of symmetry --/
def has_exactly_two_axes_of_symmetry (f : Figure) : Prop :=
  ∃ (a1 a2 : AxisOfSymmetry), a1 ≠ a2 ∧
    (∀ (a : AxisOfSymmetry), a = a1 ∨ a = a2)

/-- A function to determine if a figure has a center of symmetry --/
def has_center_of_symmetry (f : Figure) : Prop :=
  ∃ (c : CenterOfSymmetry), true  -- Placeholder, replace with actual condition

/-- Theorem: If a figure has exactly two axes of symmetry, it must have a center of symmetry --/
theorem two_axes_implies_center_symmetry (f : Figure) :
  has_exactly_two_axes_of_symmetry f → has_center_of_symmetry f :=
by sorry

end NUMINAMATH_CALUDE_two_axes_implies_center_symmetry_l2159_215980


namespace NUMINAMATH_CALUDE_largest_circle_radius_l2159_215908

/-- Represents a standard chessboard --/
structure Chessboard :=
  (size : ℕ)
  (is_standard : size = 8)

/-- Represents a circle on the chessboard --/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- Checks if a circle intersects any white square on the chessboard --/
def intersects_white_square (c : Circle) (b : Chessboard) : Prop :=
  sorry

/-- The largest circle that doesn't intersect any white square --/
def largest_circle (b : Chessboard) : Circle :=
  sorry

theorem largest_circle_radius (b : Chessboard) :
  (largest_circle b).radius = (Real.sqrt 10) / 2 :=
sorry

end NUMINAMATH_CALUDE_largest_circle_radius_l2159_215908


namespace NUMINAMATH_CALUDE_pokemon_card_ratio_l2159_215906

theorem pokemon_card_ratio : 
  ∀ (jenny orlando richard : ℕ),
    jenny = 6 →
    orlando = jenny + 2 →
    ∃ k : ℕ, richard = k * orlando →
    jenny + orlando + richard = 38 →
    richard / orlando = 3 := by
  sorry

end NUMINAMATH_CALUDE_pokemon_card_ratio_l2159_215906


namespace NUMINAMATH_CALUDE_expression_simplification_l2159_215922

theorem expression_simplification (x : ℝ) : 
  2 * x - 3 * (2 - x) + 4 * (2 + x) - 5 * (1 - 3 * x) = 24 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2159_215922


namespace NUMINAMATH_CALUDE_function_ordering_l2159_215915

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of being an even function
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the property of being monotonically decreasing on an interval
def is_monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

-- State the theorem
theorem function_ordering (h1 : is_even f) (h2 : is_monotone_decreasing_on (fun x ↦ f (x - 2)) 0 2) :
  f 0 < f (-1) ∧ f (-1) < f 2 :=
sorry

end NUMINAMATH_CALUDE_function_ordering_l2159_215915


namespace NUMINAMATH_CALUDE_oranges_per_box_l2159_215970

theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℕ) 
  (h1 : total_oranges = 42) (h2 : num_boxes = 7) :
  total_oranges / num_boxes = 6 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_box_l2159_215970


namespace NUMINAMATH_CALUDE_negation_of_all_divisible_by_seven_are_odd_l2159_215982

theorem negation_of_all_divisible_by_seven_are_odd :
  (¬ ∀ n : ℤ, 7 ∣ n → Odd n) ↔ (∃ n : ℤ, 7 ∣ n ∧ ¬ Odd n) := by sorry

end NUMINAMATH_CALUDE_negation_of_all_divisible_by_seven_are_odd_l2159_215982
