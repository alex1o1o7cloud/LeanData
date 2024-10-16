import Mathlib

namespace NUMINAMATH_CALUDE_least_four_digit_7_heavy_l1338_133896

def is_7_heavy (n : ℕ) : Prop := n % 7 > 3

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem least_four_digit_7_heavy : 
  (∀ n : ℕ, is_four_digit n → is_7_heavy n → 1000 ≤ n) ∧ 
  is_four_digit 1000 ∧ 
  is_7_heavy 1000 :=
sorry

end NUMINAMATH_CALUDE_least_four_digit_7_heavy_l1338_133896


namespace NUMINAMATH_CALUDE_solve_for_a_l1338_133881

def A (a : ℝ) : Set ℝ := {a - 2, a^2 + 4*a, 10}

theorem solve_for_a : ∀ a : ℝ, -3 ∈ A a → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l1338_133881


namespace NUMINAMATH_CALUDE_x_can_be_any_real_value_l1338_133878

theorem x_can_be_any_real_value
  (x y z w : ℝ)
  (h1 : x / y > z / w)
  (h2 : y ≠ 0 ∧ w ≠ 0)
  (h3 : y * w > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b < 0 ∧ c = 0 ∧
    (x = a ∨ x = b ∨ x = c) :=
sorry

end NUMINAMATH_CALUDE_x_can_be_any_real_value_l1338_133878


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_range_of_c_l1338_133841

-- Define the quadratic function
def f (a b x : ℝ) := x^2 - a*x + b

-- Define the inequality
def inequality (a b : ℝ) := {x : ℝ | f a b x < 0}

-- Define the second quadratic function
def g (b c x : ℝ) := -x^2 + b*x + c

-- Theorem 1
theorem sum_of_a_and_b (a b : ℝ) : 
  inequality a b = {x | 2 < x ∧ x < 3} → a + b = 11 := by sorry

-- Theorem 2
theorem range_of_c (c : ℝ) : 
  (∀ x, g 6 c x ≤ 0) → c ≤ -9 := by sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_range_of_c_l1338_133841


namespace NUMINAMATH_CALUDE_g_inequality_solution_set_range_of_a_l1338_133853

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - 5*a| + |2*x + 1|
def g (x : ℝ) : ℝ := |x - 1| + 3

-- Theorem for the solution set of |g(x)| < 8
theorem g_inequality_solution_set :
  {x : ℝ | |g x| < 8} = {x : ℝ | -4 < x ∧ x < 6} := by sorry

-- Theorem for the range of a
theorem range_of_a (h : ∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) :
  a ≥ 0.4 ∨ a ≤ -0.8 := by sorry

end NUMINAMATH_CALUDE_g_inequality_solution_set_range_of_a_l1338_133853


namespace NUMINAMATH_CALUDE_evaluate_expression_l1338_133820

theorem evaluate_expression (a b c : ℚ) (ha : a = 1/2) (hb : b = 3/4) (hc : c = 8) :
  a^3 * b^2 * c = 9/16 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1338_133820


namespace NUMINAMATH_CALUDE_green_shirt_pairs_green_green_pairs_count_l1338_133800

theorem green_shirt_pairs (total_students : ℕ) (red_students : ℕ) (green_students : ℕ) 
  (total_pairs : ℕ) (red_red_pairs : ℕ) : ℕ :=
  let green_green_pairs := 
    have _ : total_students = 144 := by sorry
    have _ : red_students = 63 := by sorry
    have _ : green_students = 81 := by sorry
    have _ : total_pairs = 72 := by sorry
    have _ : red_red_pairs = 27 := by sorry
    have _ : total_students = red_students + green_students := by sorry
    have _ : red_students * 2 ≥ red_red_pairs * 2 := by sorry
    let red_in_mixed_pairs := red_students - (red_red_pairs * 2)
    let remaining_green := green_students - red_in_mixed_pairs
    remaining_green / 2
  green_green_pairs

theorem green_green_pairs_count : 
  green_shirt_pairs 144 63 81 72 27 = 36 := by sorry

end NUMINAMATH_CALUDE_green_shirt_pairs_green_green_pairs_count_l1338_133800


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1338_133824

theorem inequality_solution_set (x : ℝ) : 2 * x - 3 > 7 - x ↔ x > 10 / 3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1338_133824


namespace NUMINAMATH_CALUDE_second_integer_value_l1338_133856

theorem second_integer_value (a b c d : ℤ) : 
  (∃ x : ℤ, a = x ∧ b = x + 2 ∧ c = x + 4 ∧ d = x + 6) →  -- consecutive even integers
  (a + d = 156) →                                        -- sum of first and fourth is 156
  b = 77                                                 -- second integer is 77
:= by sorry

end NUMINAMATH_CALUDE_second_integer_value_l1338_133856


namespace NUMINAMATH_CALUDE_all_flowers_bloom_monday_l1338_133842

-- Define the days of the week
inductive Day : Type
| monday : Day
| tuesday : Day
| wednesday : Day
| thursday : Day
| friday : Day
| saturday : Day
| sunday : Day

-- Define the flower types
inductive Flower : Type
| sunflower : Flower
| lily : Flower
| peony : Flower

-- Define a function to check if a flower blooms on a given day
def blooms (f : Flower) (d : Day) : Prop := sorry

-- Define the conditions
axiom one_day_all_bloom : ∃! d : Day, ∀ f : Flower, blooms f d

axiom no_three_consecutive_days : 
  ∀ f : Flower, ∀ d1 d2 d3 : Day, 
    (blooms f d1 ∧ blooms f d2 ∧ blooms f d3) → 
    (d1 ≠ Day.monday ∨ d2 ≠ Day.tuesday ∨ d3 ≠ Day.wednesday) ∧
    (d1 ≠ Day.tuesday ∨ d2 ≠ Day.wednesday ∨ d3 ≠ Day.thursday) ∧
    (d1 ≠ Day.wednesday ∨ d2 ≠ Day.thursday ∨ d3 ≠ Day.friday) ∧
    (d1 ≠ Day.thursday ∨ d2 ≠ Day.friday ∨ d3 ≠ Day.saturday) ∧
    (d1 ≠ Day.friday ∨ d2 ≠ Day.saturday ∨ d3 ≠ Day.sunday) ∧
    (d1 ≠ Day.saturday ∨ d2 ≠ Day.sunday ∨ d3 ≠ Day.monday) ∧
    (d1 ≠ Day.sunday ∨ d2 ≠ Day.monday ∨ d3 ≠ Day.tuesday)

axiom two_flowers_not_bloom : 
  ∀ f1 f2 : Flower, f1 ≠ f2 → 
    (∃! d : Day, ¬(blooms f1 d ∧ blooms f2 d))

axiom sunflowers_not_bloom : 
  ¬blooms Flower.sunflower Day.tuesday ∧ 
  ¬blooms Flower.sunflower Day.thursday ∧ 
  ¬blooms Flower.sunflower Day.sunday

axiom lilies_not_bloom : 
  ¬blooms Flower.lily Day.thursday ∧ 
  ¬blooms Flower.lily Day.saturday

axiom peonies_not_bloom : 
  ¬blooms Flower.peony Day.sunday

-- The theorem to prove
theorem all_flowers_bloom_monday : 
  ∀ f : Flower, blooms f Day.monday ∧ 
  (∀ d : Day, d ≠ Day.monday → ¬(∀ f : Flower, blooms f d)) :=
by sorry

end NUMINAMATH_CALUDE_all_flowers_bloom_monday_l1338_133842


namespace NUMINAMATH_CALUDE_intersection_segment_length_l1338_133854

-- Define the parabola and line
def parabola (x y : ℝ) : Prop := x^2 = -4*y
def line (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ line A.1 A.2 ∧
  parabola B.1 B.2 ∧ line B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem intersection_segment_length :
  ∀ A B : ℝ × ℝ, intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 := by sorry

end NUMINAMATH_CALUDE_intersection_segment_length_l1338_133854


namespace NUMINAMATH_CALUDE_not_cheap_necessary_for_good_quality_l1338_133803

-- Define the universe of items
variable (Item : Type)

-- Define the properties
variable (not_cheap : Item → Prop)
variable (good_quality : Item → Prop)

-- Define the "You get what you pay for" principle
variable (you_get_what_you_pay_for : ∀ (x : Item), good_quality x → ¬(not_cheap x) → False)

-- Theorem: "not cheap" is a necessary condition for "good quality"
theorem not_cheap_necessary_for_good_quality :
  ∀ (x : Item), good_quality x → not_cheap x :=
by
  sorry

end NUMINAMATH_CALUDE_not_cheap_necessary_for_good_quality_l1338_133803


namespace NUMINAMATH_CALUDE_school_bus_capacity_l1338_133850

/-- Calculates the total number of students that can be seated on a bus --/
def bus_capacity (rows : ℕ) (sections_per_row : ℕ) (students_per_section : ℕ) : ℕ :=
  rows * sections_per_row * students_per_section

/-- Theorem: A bus with 13 rows, 2 sections per row, and 2 students per section can seat 52 students --/
theorem school_bus_capacity : bus_capacity 13 2 2 = 52 := by
  sorry

end NUMINAMATH_CALUDE_school_bus_capacity_l1338_133850


namespace NUMINAMATH_CALUDE_banana_groups_l1338_133804

theorem banana_groups (total_bananas : ℕ) (group_size : ℕ) (h1 : total_bananas = 180) (h2 : group_size = 18) :
  total_bananas / group_size = 10 := by
  sorry

end NUMINAMATH_CALUDE_banana_groups_l1338_133804


namespace NUMINAMATH_CALUDE_outfit_choices_l1338_133898

theorem outfit_choices (shirts : ℕ) (skirts : ℕ) (dresses : ℕ) : 
  shirts = 4 → skirts = 3 → dresses = 2 → shirts * skirts + dresses = 14 := by
  sorry

end NUMINAMATH_CALUDE_outfit_choices_l1338_133898


namespace NUMINAMATH_CALUDE_suitcase_lock_settings_l1338_133832

/-- Represents a lock with a specified number of dials and digits per dial -/
structure Lock :=
  (numDials : ℕ)
  (digitsPerDial : ℕ)

/-- Calculates the number of different settings for a lock with all digits different -/
def countDifferentSettings (lock : Lock) : ℕ :=
  sorry

/-- The specific lock in the problem -/
def suitcaseLock : Lock :=
  { numDials := 3,
    digitsPerDial := 10 }

/-- Theorem stating that the number of different settings for the suitcase lock is 720 -/
theorem suitcase_lock_settings :
  countDifferentSettings suitcaseLock = 720 :=
sorry

end NUMINAMATH_CALUDE_suitcase_lock_settings_l1338_133832


namespace NUMINAMATH_CALUDE_exercise_time_is_1910_l1338_133889

/-- The total exercise time for Javier, Sanda, Luis, and Nita -/
def total_exercise_time : ℕ :=
  let javier := 50 * 10
  let sanda := 90 * 3 + 75 * 2 + 45 * 4
  let luis := 60 * 5 + 30 * 3
  let nita := 100 * 2 + 55 * 4
  javier + sanda + luis + nita

/-- Theorem stating that the total exercise time is 1910 minutes -/
theorem exercise_time_is_1910 : total_exercise_time = 1910 := by
  sorry

end NUMINAMATH_CALUDE_exercise_time_is_1910_l1338_133889


namespace NUMINAMATH_CALUDE_sequence_divisibility_l1338_133802

def u : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 2 * u (n + 1) - 3 * u n

def v (a b c : ℤ) : ℕ → ℤ
  | 0 => a
  | 1 => b
  | 2 => c
  | (n + 3) => v a b c (n + 2) - 3 * v a b c (n + 1) + 27 * v a b c n

theorem sequence_divisibility (a b c : ℤ) :
  (∃ N : ℕ, ∀ n > N, ∃ k : ℤ, v a b c n = k * u n) →
  3 * a = 2 * b + c := by
  sorry

end NUMINAMATH_CALUDE_sequence_divisibility_l1338_133802


namespace NUMINAMATH_CALUDE_stating_min_natives_correct_stating_min_natives_sufficient_l1338_133826

/-- Represents the minimum number of natives required for the joke-sharing problem. -/
def min_natives (k : ℕ) : ℕ := 2^k

/-- 
Theorem stating that min_natives(k) is the smallest number of natives needed
for each native to know at least k jokes (apart from their own) after crossing the river.
-/
theorem min_natives_correct (k : ℕ) :
  ∀ N : ℕ, (∀ native : Fin N, 
    (∃ known_jokes : Finset (Fin N), 
      known_jokes.card ≥ k ∧ 
      native ∉ known_jokes ∧
      (∀ joke ∈ known_jokes, joke ≠ native))) 
    → N ≥ min_natives k :=
by
  sorry

/-- 
Theorem stating that min_natives(k) is sufficient for each native to know
at least k jokes (apart from their own) after crossing the river.
-/
theorem min_natives_sufficient (k : ℕ) :
  ∃ crossing_strategy : Unit,
    ∀ native : Fin (min_natives k),
      ∃ known_jokes : Finset (Fin (min_natives k)),
        known_jokes.card ≥ k ∧
        native ∉ known_jokes ∧
        (∀ joke ∈ known_jokes, joke ≠ native) :=
by
  sorry

end NUMINAMATH_CALUDE_stating_min_natives_correct_stating_min_natives_sufficient_l1338_133826


namespace NUMINAMATH_CALUDE_multiples_of_seven_square_l1338_133815

theorem multiples_of_seven_square (a b : ℕ) : 
  (∀ k : ℕ, k ≤ a → (7 * k < 50)) ∧ 
  (∀ k : ℕ, k > a → (7 * k ≥ 50)) ∧
  (∀ k : ℕ, k ≤ b → (k * 7 < 50 ∧ k > 0)) ∧
  (∀ k : ℕ, k > b → (k * 7 ≥ 50 ∨ k ≤ 0)) →
  (a + b)^2 = 196 := by
sorry

end NUMINAMATH_CALUDE_multiples_of_seven_square_l1338_133815


namespace NUMINAMATH_CALUDE_sin_cos_cube_sum_l1338_133814

theorem sin_cos_cube_sum (θ : ℝ) (h : Real.sin θ + Real.cos θ = 1/2) :
  Real.sin θ ^ 3 + Real.cos θ ^ 3 = 11/16 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_cube_sum_l1338_133814


namespace NUMINAMATH_CALUDE_base5_to_base8_conversion_l1338_133816

/-- Converts a base-5 number to base-10 -/
def base5_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 -/
def base10_to_base8 (n : ℕ) : ℕ := sorry

theorem base5_to_base8_conversion :
  base10_to_base8 (base5_to_base10 1234) = 302 := by sorry

end NUMINAMATH_CALUDE_base5_to_base8_conversion_l1338_133816


namespace NUMINAMATH_CALUDE_weight_of_b_l1338_133887

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 41) :
  b = 27 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_b_l1338_133887


namespace NUMINAMATH_CALUDE_cubic_equation_root_b_value_l1338_133843

theorem cubic_equation_root_b_value :
  ∀ (a b : ℚ),
  (∃ (x : ℂ), x = 1 + Real.sqrt 2 ∧ x^3 + a*x^2 + b*x + 6 = 0) →
  b = 11 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_b_value_l1338_133843


namespace NUMINAMATH_CALUDE_solution_set_is_correct_l1338_133838

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x : ℝ, (f.deriv.deriv) x < f x)
variable (h2 : f 2 = 1)

-- Define the solution set
def solution_set := {x : ℝ | f x > Real.exp (x - 2)}

-- State the theorem
theorem solution_set_is_correct : solution_set f = Set.Iio 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_is_correct_l1338_133838


namespace NUMINAMATH_CALUDE_infinite_integer_solutions_l1338_133846

theorem infinite_integer_solutions :
  ∃ (S : Set ℕ+), (Set.Infinite S) ∧
  (∀ n ∈ S, ¬ ∃ m : ℕ, n = m^3) ∧
  (∀ n ∈ S,
    let a : ℝ := (n : ℝ)^(1/3)
    let b : ℝ := 1 / (a - ⌊a⌋)
    let c : ℝ := 1 / (b - ⌊b⌋)
    ∃ r s t : ℤ, (r ≠ 0 ∨ s ≠ 0 ∨ t ≠ 0) ∧ r * a + s * b + t * c = 0) :=
by sorry

end NUMINAMATH_CALUDE_infinite_integer_solutions_l1338_133846


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1338_133868

/-- A geometric sequence where the sum of every two consecutive terms forms another geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 2) + a (n + 3) = r * (a n + a (n + 1))

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  a 1 + a 2 = 1 →
  a 3 + a 4 = 2 →
  a 9 + a 10 = 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1338_133868


namespace NUMINAMATH_CALUDE_ten_thousandths_digit_of_437_div_128_l1338_133860

theorem ten_thousandths_digit_of_437_div_128 :
  (437 : ℚ) / 128 = 3 + 4/10 + 1/100 + 4/1000 + 6/10000 + 8/100000 + 7/1000000 + 5/10000000 :=
by sorry

end NUMINAMATH_CALUDE_ten_thousandths_digit_of_437_div_128_l1338_133860


namespace NUMINAMATH_CALUDE_inscribed_squares_segment_product_l1338_133805

theorem inscribed_squares_segment_product (c d : ℝ) : 
  (∃ (small_square_area large_square_area : ℝ),
    small_square_area = 9 ∧ 
    large_square_area = 18 ∧ 
    c + d = (large_square_area).sqrt ∧ 
    c^2 + d^2 = large_square_area) → 
  c * d = 0 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_segment_product_l1338_133805


namespace NUMINAMATH_CALUDE_total_students_l1338_133892

theorem total_students (students_3rd : ℕ) (students_4th : ℕ) (boys_2nd : ℕ) (girls_2nd : ℕ) :
  students_3rd = 19 →
  students_4th = 2 * students_3rd →
  boys_2nd = 10 →
  girls_2nd = 19 →
  students_3rd + students_4th + (boys_2nd + girls_2nd) = 86 :=
by sorry

end NUMINAMATH_CALUDE_total_students_l1338_133892


namespace NUMINAMATH_CALUDE_line_circle_intersection_range_l1338_133882

/-- The equation of the circle C -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 8*x + 12 = 0

/-- The equation of the line -/
def line_equation (x y k : ℝ) : Prop :=
  y = k*x - 2

/-- The condition for the line to have at least one common point with the circle -/
def has_common_point (k : ℝ) : Prop :=
  ∃ x y : ℝ, circle_equation x y ∧ line_equation x y k

/-- The theorem stating the range of k for which the line has at least one common point with the circle -/
theorem line_circle_intersection_range :
  ∀ k : ℝ, has_common_point k ↔ -4/3 ≤ k ∧ k ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_range_l1338_133882


namespace NUMINAMATH_CALUDE_cos_sin_transformation_l1338_133831

theorem cos_sin_transformation (x : ℝ) : 
  3 * Real.cos x = 3 * Real.sin (2 * (x + 2 * Real.pi / 3) - Real.pi / 6) := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_transformation_l1338_133831


namespace NUMINAMATH_CALUDE_sin_15_cos_75_plus_cos_15_sin_105_eq_1_l1338_133866

theorem sin_15_cos_75_plus_cos_15_sin_105_eq_1 :
  Real.sin (15 * π / 180) * Real.cos (75 * π / 180) + 
  Real.cos (15 * π / 180) * Real.sin (105 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_cos_75_plus_cos_15_sin_105_eq_1_l1338_133866


namespace NUMINAMATH_CALUDE_two_numbers_with_difference_and_quotient_l1338_133872

theorem two_numbers_with_difference_and_quotient :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a - b = 157 ∧ a / b = 2 ∧ a = 314 ∧ b = 157 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_with_difference_and_quotient_l1338_133872


namespace NUMINAMATH_CALUDE_power_sum_prime_l1338_133848

theorem power_sum_prime (p a n : ℕ) : 
  Prime p → a > 0 → n > 0 → (2^p + 3^p = a^n) → n = 1 := by
sorry

end NUMINAMATH_CALUDE_power_sum_prime_l1338_133848


namespace NUMINAMATH_CALUDE_james_weekly_nut_spending_l1338_133880

/-- Represents the cost and consumption of nuts -/
structure NutInfo where
  price : ℚ
  weight : ℚ
  consumption : ℚ
  days : ℕ

/-- Calculates the weekly cost for a type of nut -/
def weeklyCost (nut : NutInfo) : ℚ :=
  (nut.consumption / nut.days) * 7 * (nut.price / nut.weight)

/-- Theorem stating James' weekly spending on nuts -/
theorem james_weekly_nut_spending :
  let pistachios : NutInfo := ⟨10, 5, 30, 5⟩
  let almonds : NutInfo := ⟨8, 4, 24, 4⟩
  let walnuts : NutInfo := ⟨12, 6, 18, 3⟩
  weeklyCost pistachios + weeklyCost almonds + weeklyCost walnuts = 252 := by
  sorry

end NUMINAMATH_CALUDE_james_weekly_nut_spending_l1338_133880


namespace NUMINAMATH_CALUDE_greatest_q_minus_r_l1338_133801

theorem greatest_q_minus_r : ∃ (q r : ℕ), 
  1043 = 23 * q + r ∧ 
  r < 23 ∧
  ∀ (q' r' : ℕ), 1043 = 23 * q' + r' ∧ r' < 23 → q' - r' ≤ q - r :=
by sorry

end NUMINAMATH_CALUDE_greatest_q_minus_r_l1338_133801


namespace NUMINAMATH_CALUDE_existence_of_b₁_b₂_l1338_133823

theorem existence_of_b₁_b₂ (a₁ a₂ : ℝ) 
  (h₁ : a₁ ≥ 0) (h₂ : a₂ ≥ 0) (h₃ : a₁ + a₂ = 1) : 
  ∃ b₁ b₂ : ℝ, b₁ ≥ 0 ∧ b₂ ≥ 0 ∧ b₁ + b₂ = 1 ∧ 
  (5/4 - a₁) * b₁ + 3 * (5/4 - a₂) * b₂ > 1 := by
sorry

end NUMINAMATH_CALUDE_existence_of_b₁_b₂_l1338_133823


namespace NUMINAMATH_CALUDE_fashion_show_total_time_l1338_133857

/-- Represents the different types of clothing in the fashion show -/
inductive ClothingType
  | EveningWear
  | BathingSuit
  | FormalWear
  | CasualWear

/-- Returns the time in minutes for a runway walk based on the clothing type -/
def walkTime (c : ClothingType) : ℝ :=
  match c with
  | ClothingType.EveningWear => 4
  | ClothingType.BathingSuit => 2
  | ClothingType.FormalWear => 3
  | ClothingType.CasualWear => 2.5

/-- The number of models in the show -/
def numModels : ℕ := 10

/-- Returns the number of sets for each clothing type -/
def numSets (c : ClothingType) : ℕ :=
  match c with
  | ClothingType.EveningWear => 4
  | ClothingType.BathingSuit => 2
  | ClothingType.FormalWear => 3
  | ClothingType.CasualWear => 5

/-- Calculates the total time for all runway walks of a specific clothing type -/
def totalTimeForClothingType (c : ClothingType) : ℝ :=
  (walkTime c) * (numSets c : ℝ) * (numModels : ℝ)

/-- Theorem: The total time for all runway trips during the fashion show is 415 minutes -/
theorem fashion_show_total_time :
  (totalTimeForClothingType ClothingType.EveningWear) +
  (totalTimeForClothingType ClothingType.BathingSuit) +
  (totalTimeForClothingType ClothingType.FormalWear) +
  (totalTimeForClothingType ClothingType.CasualWear) = 415 := by
  sorry


end NUMINAMATH_CALUDE_fashion_show_total_time_l1338_133857


namespace NUMINAMATH_CALUDE_no_students_in_both_l1338_133879

/-- Represents the number of students in different language classes -/
structure LanguageClasses where
  total : ℕ
  onlyFrench : ℕ
  onlySpanish : ℕ
  neither : ℕ

/-- Calculates the number of students taking both French and Spanish -/
def studentsInBoth (classes : LanguageClasses) : ℕ :=
  classes.total - (classes.onlyFrench + classes.onlySpanish + classes.neither)

/-- Theorem: In the given scenario, no students are taking both French and Spanish -/
theorem no_students_in_both (classes : LanguageClasses)
  (h_total : classes.total = 28)
  (h_french : classes.onlyFrench = 5)
  (h_spanish : classes.onlySpanish = 10)
  (h_neither : classes.neither = 13) :
  studentsInBoth classes = 0 := by
  sorry

#eval studentsInBoth { total := 28, onlyFrench := 5, onlySpanish := 10, neither := 13 }

end NUMINAMATH_CALUDE_no_students_in_both_l1338_133879


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1338_133859

def M : Set ℝ := {y | ∃ x, y = Real.sin x}
def N : Set ℝ := {0, 1, 2}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1338_133859


namespace NUMINAMATH_CALUDE_vann_teeth_cleaning_l1338_133864

/-- The number of teeth a dog has -/
def dog_teeth : ℕ := 42

/-- The number of teeth a cat has -/
def cat_teeth : ℕ := 30

/-- The number of teeth a pig has -/
def pig_teeth : ℕ := 28

/-- The number of dogs Vann will clean -/
def num_dogs : ℕ := 5

/-- The number of cats Vann will clean -/
def num_cats : ℕ := 10

/-- The number of pigs Vann will clean -/
def num_pigs : ℕ := 7

/-- The total number of teeth Vann will clean -/
def total_teeth : ℕ := dog_teeth * num_dogs + cat_teeth * num_cats + pig_teeth * num_pigs

theorem vann_teeth_cleaning :
  total_teeth = 706 := by
  sorry

end NUMINAMATH_CALUDE_vann_teeth_cleaning_l1338_133864


namespace NUMINAMATH_CALUDE_continuous_fraction_solution_l1338_133894

theorem continuous_fraction_solution :
  ∃ y : ℝ, y > 0 ∧ y = 3 + 3 / (2 + 3 / y) ∧ y = (3 + 3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_continuous_fraction_solution_l1338_133894


namespace NUMINAMATH_CALUDE_parabola_focus_l1338_133875

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := x^2 = -4*y

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (0, -1)

/-- Theorem: The focus of the parabola x^2 = -4y is (0, -1) -/
theorem parabola_focus :
  ∀ x y : ℝ, parabola_equation x y → (x, y) = focus :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l1338_133875


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l1338_133847

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -7) : x^3 + 1/x^3 = -322 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l1338_133847


namespace NUMINAMATH_CALUDE_g_lower_bound_l1338_133883

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a|

-- Define the function g
def g (a x : ℝ) : ℝ := f a x + f a (x + 2)

-- Theorem statement
theorem g_lower_bound (a : ℝ) (h : ∀ x, f a x ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2) :
  ∀ x, g a x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_g_lower_bound_l1338_133883


namespace NUMINAMATH_CALUDE_roden_fish_count_l1338_133810

/-- The number of gold fish Roden bought -/
def gold_fish : ℕ := 15

/-- The number of blue fish Roden bought -/
def blue_fish : ℕ := 7

/-- The total number of fish Roden bought -/
def total_fish : ℕ := gold_fish + blue_fish

theorem roden_fish_count : total_fish = 22 := by
  sorry

end NUMINAMATH_CALUDE_roden_fish_count_l1338_133810


namespace NUMINAMATH_CALUDE_rebeccas_haircut_price_l1338_133852

/-- Rebecca's hair salon pricing and earnings --/
theorem rebeccas_haircut_price 
  (perm_price : ℕ) 
  (dye_job_price : ℕ) 
  (dye_cost : ℕ) 
  (haircuts : ℕ) 
  (perms : ℕ) 
  (dye_jobs : ℕ) 
  (tips : ℕ) 
  (total_earnings : ℕ) 
  (h : perm_price = 40) 
  (i : dye_job_price = 60) 
  (j : dye_cost = 10) 
  (k : haircuts = 4) 
  (l : perms = 1) 
  (m : dye_jobs = 2) 
  (n : tips = 50) 
  (o : total_earnings = 310) : 
  ∃ (haircut_price : ℕ), 
    haircut_price * haircuts + 
    perm_price * perms + 
    dye_job_price * dye_jobs + 
    tips - 
    dye_cost * dye_jobs = total_earnings ∧ 
    haircut_price = 30 := by
  sorry

end NUMINAMATH_CALUDE_rebeccas_haircut_price_l1338_133852


namespace NUMINAMATH_CALUDE_max_value_inequality_equality_case_l1338_133870

theorem max_value_inequality (a : ℝ) : (∀ x > 1, x + 1 / (x - 1) ≥ a) → a ≤ 3 := by sorry

theorem equality_case : ∃ x > 1, x + 1 / (x - 1) = 3 := by sorry

end NUMINAMATH_CALUDE_max_value_inequality_equality_case_l1338_133870


namespace NUMINAMATH_CALUDE_rectangle_area_l1338_133885

/-- The area of a rectangle with dimensions 0.5 meters and 0.36 meters is 1800 square centimeters. -/
theorem rectangle_area : 
  let length_m : ℝ := 0.5
  let width_m : ℝ := 0.36
  let cm_per_m : ℝ := 100
  let length_cm := length_m * cm_per_m
  let width_cm := width_m * cm_per_m
  length_cm * width_cm = 1800 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1338_133885


namespace NUMINAMATH_CALUDE_calculator_prices_and_relations_l1338_133861

/-- The price of two A-brand and three B-brand calculators -/
def total_price_1 : ℝ := 156

/-- The price of three A-brand and one B-brand calculator -/
def total_price_2 : ℝ := 122

/-- The discount rate for A-brand calculators during promotion -/
def discount_rate_A : ℝ := 0.8

/-- The discount rate for B-brand calculators during promotion -/
def discount_rate_B : ℝ := 0.875

/-- The unit price of A-brand calculators -/
def price_A : ℝ := 30

/-- The unit price of B-brand calculators -/
def price_B : ℝ := 32

/-- The function relation for A-brand calculators during promotion -/
def y1 (x : ℝ) : ℝ := 24 * x

/-- The function relation for B-brand calculators during promotion -/
def y2 (x : ℝ) : ℝ := 28 * x

theorem calculator_prices_and_relations :
  (2 * price_A + 3 * price_B = total_price_1) ∧
  (3 * price_A + price_B = total_price_2) ∧
  (∀ x, y1 x = discount_rate_A * price_A * x) ∧
  (∀ x, y2 x = discount_rate_B * price_B * x) :=
sorry

end NUMINAMATH_CALUDE_calculator_prices_and_relations_l1338_133861


namespace NUMINAMATH_CALUDE_seven_balls_three_boxes_l1338_133828

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 365 ways to distribute 7 distinguishable balls into 3 indistinguishable boxes -/
theorem seven_balls_three_boxes : distribute_balls 7 3 = 365 := by sorry

end NUMINAMATH_CALUDE_seven_balls_three_boxes_l1338_133828


namespace NUMINAMATH_CALUDE_arrival_time_difference_l1338_133811

/-- The distance to the campsite in miles -/
def distance : ℝ := 3

/-- Jill's hiking speed in miles per hour -/
def jill_speed : ℝ := 6

/-- Jack's hiking speed in miles per hour -/
def jack_speed : ℝ := 3

/-- Conversion factor from hours to minutes -/
def minutes_per_hour : ℝ := 60

/-- The time difference in minutes between Jill and Jack's arrival at the campsite -/
theorem arrival_time_difference : 
  (distance / jack_speed - distance / jill_speed) * minutes_per_hour = 30 := by
  sorry

end NUMINAMATH_CALUDE_arrival_time_difference_l1338_133811


namespace NUMINAMATH_CALUDE_no_special_pentagon_l1338_133858

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a pentagon as a set of 5 points
def Pentagon : Type := { p : Finset Point3D // p.card = 5 }

-- Define a function to check if three points are colinear
def areColinear (p q r : Point3D) : Prop := sorry

-- Define a function to check if a point is in the interior of a triangle
def isInteriorPoint (p : Point3D) (t1 t2 t3 : Point3D) : Prop := sorry

-- Define a function to check if a line segment intersects a plane at an interior point of a triangle
def intersectsTriangleInterior (p1 p2 t1 t2 t3 : Point3D) : Prop := sorry

-- Main theorem
theorem no_special_pentagon : 
  ¬ ∃ (pent : Pentagon), 
    ∀ (v1 v2 v3 v4 v5 : Point3D),
      v1 ∈ pent.val → v2 ∈ pent.val → v3 ∈ pent.val → v4 ∈ pent.val → v5 ∈ pent.val →
      v1 ≠ v2 → v1 ≠ v3 → v1 ≠ v4 → v1 ≠ v5 → v2 ≠ v3 → v2 ≠ v4 → v2 ≠ v5 → v3 ≠ v4 → v3 ≠ v5 → v4 ≠ v5 →
      (intersectsTriangleInterior v1 v3 v2 v4 v5 ∧
       intersectsTriangleInterior v1 v4 v2 v3 v5 ∧
       intersectsTriangleInterior v2 v4 v1 v3 v5 ∧
       intersectsTriangleInterior v2 v5 v1 v3 v4 ∧
       intersectsTriangleInterior v3 v5 v1 v2 v4) :=
by sorry


end NUMINAMATH_CALUDE_no_special_pentagon_l1338_133858


namespace NUMINAMATH_CALUDE_sausage_cutting_theorem_l1338_133899

/-- Represents the number of pieces produced when cutting along rings of a single color -/
def PiecesFromSingleColor : ℕ → ℕ := λ n => n + 1

/-- Represents the total number of pieces produced when cutting along rings of multiple colors -/
def TotalPieces (cuts : List ℕ) : ℕ :=
  (cuts.sum) + 1

theorem sausage_cutting_theorem (red yellow green : ℕ) 
  (h_red : PiecesFromSingleColor red = 5)
  (h_yellow : PiecesFromSingleColor yellow = 7)
  (h_green : PiecesFromSingleColor green = 11) :
  TotalPieces [red, yellow, green] = 21 := by
  sorry

#check sausage_cutting_theorem

end NUMINAMATH_CALUDE_sausage_cutting_theorem_l1338_133899


namespace NUMINAMATH_CALUDE_periodic_function_equality_l1338_133817

/-- Given a function f(x) = a*sin(πx + α) + b*cos(πx + β), if f(3) = 3, then f(2016) = -3 -/
theorem periodic_function_equality (a b α β : ℝ) :
  let f : ℝ → ℝ := λ x => a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β)
  f 3 = 3 → f 2016 = -3 := by sorry

end NUMINAMATH_CALUDE_periodic_function_equality_l1338_133817


namespace NUMINAMATH_CALUDE_smallest_term_proof_l1338_133827

def arithmetic_sequence (n : ℕ) : ℕ := 7 * n

theorem smallest_term_proof :
  ∀ k : ℕ, 
    (arithmetic_sequence k > 150 ∧ arithmetic_sequence k % 5 = 0) → 
    arithmetic_sequence k ≥ 175 :=
by sorry

end NUMINAMATH_CALUDE_smallest_term_proof_l1338_133827


namespace NUMINAMATH_CALUDE_alyssa_pears_l1338_133821

theorem alyssa_pears (total_pears nancy_pears : ℕ) 
  (h1 : total_pears = 59) 
  (h2 : nancy_pears = 17) : 
  total_pears - nancy_pears = 42 := by
sorry

end NUMINAMATH_CALUDE_alyssa_pears_l1338_133821


namespace NUMINAMATH_CALUDE_remaining_slices_l1338_133807

def total_slices : ℕ := 2 * 8

def slices_after_friends : ℕ := total_slices - (total_slices / 4)

def slices_after_family : ℕ := slices_after_friends - (slices_after_friends / 3)

def slices_after_alex : ℕ := slices_after_family - 3

theorem remaining_slices : slices_after_alex = 5 := by
  sorry

end NUMINAMATH_CALUDE_remaining_slices_l1338_133807


namespace NUMINAMATH_CALUDE_number_division_l1338_133877

theorem number_division : ∃ x : ℝ, x / 0.04 = 400.90000000000003 ∧ x = 16.036 := by
  sorry

end NUMINAMATH_CALUDE_number_division_l1338_133877


namespace NUMINAMATH_CALUDE_alternating_sum_of_squares_100_to_1_l1338_133808

/-- The sum of alternating differences of squares from 100² to 1² -/
def alternatingSumOfSquares : ℕ → ℤ
  | 0 => 0
  | n + 1 => (n + 1)^2 - alternatingSumOfSquares n

/-- The main theorem stating that the alternating sum of squares from 100² to 1² equals 5050 -/
theorem alternating_sum_of_squares_100_to_1 :
  alternatingSumOfSquares 100 = 5050 := by
  sorry


end NUMINAMATH_CALUDE_alternating_sum_of_squares_100_to_1_l1338_133808


namespace NUMINAMATH_CALUDE_max_items_is_nine_l1338_133830

-- Define the constants
def total_budget : ℚ := 50
def sandwich_cost : ℚ := 6
def drink_cost : ℚ := 1.5
def discount : ℚ := 5

-- Define the function to calculate the total cost
def total_cost (sandwiches : ℕ) (drinks : ℕ) : ℚ :=
  if sandwiches > 5 then
    sandwich_cost * sandwiches - discount + drink_cost * drinks
  else
    sandwich_cost * sandwiches + drink_cost * drinks

-- Define the function to check if a purchase is valid
def is_valid_purchase (sandwiches : ℕ) (drinks : ℕ) : Prop :=
  total_cost sandwiches drinks ≤ total_budget

-- Define the function to calculate the total number of items
def total_items (sandwiches : ℕ) (drinks : ℕ) : ℕ :=
  sandwiches + drinks

-- Theorem statement
theorem max_items_is_nine :
  ∀ s d : ℕ, is_valid_purchase s d → total_items s d ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_max_items_is_nine_l1338_133830


namespace NUMINAMATH_CALUDE_compound_weight_l1338_133849

/-- Given a compound with a molecular weight of 1188, prove that the total weight of 4 moles is 4752,
    while the molecular weight remains constant. -/
theorem compound_weight (molecular_weight : ℕ) (num_moles : ℕ) :
  molecular_weight = 1188 → num_moles = 4 →
  (num_moles * molecular_weight = 4752) ∧ (molecular_weight = 1188) := by
  sorry

#check compound_weight

end NUMINAMATH_CALUDE_compound_weight_l1338_133849


namespace NUMINAMATH_CALUDE_restaurant_combinations_l1338_133809

/-- The number of main dishes on the menu -/
def main_dishes : ℕ := 15

/-- The number of appetizer options -/
def appetizer_options : ℕ := 5

/-- The number of people ordering -/
def num_people : ℕ := 2

theorem restaurant_combinations :
  (main_dishes ^ num_people) * appetizer_options = 1125 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_combinations_l1338_133809


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l1338_133893

/-- A quadratic function f(x) = x^2 + (k+2)x + k + 5 -/
def f (k : ℝ) (x : ℝ) : ℝ := x^2 + (k+2)*x + k + 5

theorem quadratic_roots_condition (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ f k x₁ = 0 ∧ f k x₂ = 0) →
  -5 < k ∧ k < -4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l1338_133893


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l1338_133839

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 5| = |x + 3| := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l1338_133839


namespace NUMINAMATH_CALUDE_feet_per_mile_l1338_133813

/-- Proves that if an object travels 200 feet in 2 seconds with a speed of 68.18181818181819 miles per hour, then there are 5280 feet in one mile. -/
theorem feet_per_mile (distance : ℝ) (time : ℝ) (speed : ℝ) (feet_per_mile : ℝ) :
  distance = 200 →
  time = 2 →
  speed = 68.18181818181819 →
  distance / time = speed * feet_per_mile / 3600 →
  feet_per_mile = 5280 := by
  sorry

end NUMINAMATH_CALUDE_feet_per_mile_l1338_133813


namespace NUMINAMATH_CALUDE_fourth_circle_radius_l1338_133819

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup (O₁ O₂ O₃ O₄ : Circle) : Prop :=
  -- O₁ and O₂ are externally tangent
  let (x₁, y₁) := O₁.center
  let (x₂, y₂) := O₂.center
  ((x₂ - x₁)^2 + (y₂ - y₁)^2 = (O₁.radius + O₂.radius)^2) ∧
  -- Radii of O₁ and O₂
  (O₁.radius = 7) ∧
  (O₂.radius = 14) ∧
  -- O₃ is tangent to both O₁ and O₂
  let (x₃, y₃) := O₃.center
  ((x₃ - x₁)^2 + (y₃ - y₁)^2 = (O₁.radius + O₃.radius)^2) ∧
  ((x₃ - x₂)^2 + (y₃ - y₂)^2 = (O₂.radius + O₃.radius)^2) ∧
  -- Center of O₃ is on the line connecting centers of O₁ and O₂
  ((y₃ - y₁) * (x₂ - x₁) = (x₃ - x₁) * (y₂ - y₁)) ∧
  -- O₄ is tangent to O₁, O₂, and O₃
  let (x₄, y₄) := O₄.center
  ((x₄ - x₁)^2 + (y₄ - y₁)^2 = (O₁.radius + O₄.radius)^2) ∧
  ((x₄ - x₂)^2 + (y₄ - y₂)^2 = (O₂.radius + O₄.radius)^2) ∧
  ((x₄ - x₃)^2 + (y₄ - y₃)^2 = (O₃.radius - O₄.radius)^2)

theorem fourth_circle_radius (O₁ O₂ O₃ O₄ : Circle) :
  problem_setup O₁ O₂ O₃ O₄ → O₄.radius = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_fourth_circle_radius_l1338_133819


namespace NUMINAMATH_CALUDE_brother_age_problem_l1338_133822

theorem brother_age_problem (younger_age older_age : ℕ) : 
  younger_age + older_age = 26 → 
  older_age = younger_age + 2 → 
  older_age = 14 := by
sorry

end NUMINAMATH_CALUDE_brother_age_problem_l1338_133822


namespace NUMINAMATH_CALUDE_galaxy_distance_in_miles_l1338_133886

/-- The number of miles in one light-year -/
def miles_per_light_year : ℝ := 6 * 10^12

/-- The distance to the observed galaxy in thousand million light-years -/
def galaxy_distance_thousand_million_light_years : ℝ := 13.4

/-- Conversion factor from thousand million to billion -/
def thousand_million_to_billion : ℝ := 1

theorem galaxy_distance_in_miles :
  let distance_light_years := galaxy_distance_thousand_million_light_years * thousand_million_to_billion * 10^9
  let distance_miles := distance_light_years * miles_per_light_year
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |distance_miles - 8 * 10^22| < ε * (8 * 10^22) :=
sorry

end NUMINAMATH_CALUDE_galaxy_distance_in_miles_l1338_133886


namespace NUMINAMATH_CALUDE_first_day_price_is_four_l1338_133876

/-- Represents the pen sales scenario over three days -/
structure PenSales where
  day1_price : ℝ
  day1_quantity : ℝ

/-- The revenue is the same for all three days -/
def same_revenue (s : PenSales) : Prop :=
  s.day1_price * s.day1_quantity = 
  (s.day1_price - 1) * (s.day1_quantity + 100) ∧
  s.day1_price * s.day1_quantity = 
  (s.day1_price + 2) * (s.day1_quantity - 100)

/-- The price on the first day is 4 yuan -/
theorem first_day_price_is_four :
  ∃ (s : PenSales), same_revenue s ∧ s.day1_price = 4 := by
  sorry

end NUMINAMATH_CALUDE_first_day_price_is_four_l1338_133876


namespace NUMINAMATH_CALUDE_sin_180_degrees_l1338_133869

theorem sin_180_degrees : Real.sin (180 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_180_degrees_l1338_133869


namespace NUMINAMATH_CALUDE_product_xy_equals_nine_sqrt_three_l1338_133865

-- Define the variables
variable (x y a b : ℝ)

-- State the theorem
theorem product_xy_equals_nine_sqrt_three
  (h1 : x = b^(3/2))
  (h2 : y = a)
  (h3 : a + a = b^2)
  (h4 : y = b)
  (h5 : a + a = b^(3/2))
  (h6 : b = 3) :
  x * y = 9 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_product_xy_equals_nine_sqrt_three_l1338_133865


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_l1338_133845

theorem cubic_polynomial_root (a b c : ℚ) :
  (∃ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ∧ x = 2 - Real.sqrt 5) →
  (∀ x y : ℝ, x^3 + a*x^2 + b*x + c = 0 ∧ y^3 + a*y^2 + b*y + c = 0 → x + y = 4) →
  (-4 : ℝ)^3 + a*(-4 : ℝ)^2 + b*(-4 : ℝ) + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_l1338_133845


namespace NUMINAMATH_CALUDE_zachary_crunches_l1338_133897

/-- Proves that Zachary did 14 crunches given the conditions -/
theorem zachary_crunches : 
  ∀ (zachary_pushups zachary_total : ℕ),
  zachary_pushups = 53 →
  zachary_total = 67 →
  zachary_total - zachary_pushups = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_zachary_crunches_l1338_133897


namespace NUMINAMATH_CALUDE_slurpee_change_l1338_133825

/-- Calculates the change received when buying Slurpees -/
theorem slurpee_change (money_given : ℕ) (slurpee_cost : ℕ) (slurpees_bought : ℕ) : 
  money_given = 20 → slurpee_cost = 2 → slurpees_bought = 6 →
  money_given - (slurpee_cost * slurpees_bought) = 8 := by
  sorry

end NUMINAMATH_CALUDE_slurpee_change_l1338_133825


namespace NUMINAMATH_CALUDE_complete_square_l1338_133829

theorem complete_square (x : ℝ) : x^2 - 8*x + 15 = 0 ↔ (x - 4)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_l1338_133829


namespace NUMINAMATH_CALUDE_second_player_can_win_l1338_133891

/-- A function representing a player's strategy for choosing digits. -/
def Strategy := Nat → Fin 5

/-- The result of a game where two players alternate choosing digits. -/
def GameResult (s1 s2 : Strategy) : Fin 9 :=
  (List.range 30).foldl
    (λ acc i => (acc + if i % 2 = 0 then s1 i else s2 i) % 9)
    0

/-- Theorem stating that the second player can always ensure divisibility by 9. -/
theorem second_player_can_win :
  ∀ s1 : Strategy, ∃ s2 : Strategy, GameResult s1 s2 = 0 :=
sorry

end NUMINAMATH_CALUDE_second_player_can_win_l1338_133891


namespace NUMINAMATH_CALUDE_smallest_longer_leg_length_l1338_133888

/-- Represents a 30-60-90 triangle --/
structure Triangle30_60_90 where
  hypotenuse : ℝ
  shorterLeg : ℝ
  longerLeg : ℝ
  hypotenuse_eq : hypotenuse = 2 * shorterLeg
  longerLeg_eq : longerLeg = shorterLeg * Real.sqrt 3

/-- Represents a sequence of three connected 30-60-90 triangles --/
structure TriangleSequence where
  largest : Triangle30_60_90
  middle : Triangle30_60_90
  smallest : Triangle30_60_90
  connection1 : largest.longerLeg = middle.hypotenuse
  connection2 : middle.longerLeg = smallest.hypotenuse
  largest_hypotenuse : largest.hypotenuse = 12
  middle_special : middle.hypotenuse = middle.longerLeg

theorem smallest_longer_leg_length (seq : TriangleSequence) : 
  seq.smallest.longerLeg = 4.5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_longer_leg_length_l1338_133888


namespace NUMINAMATH_CALUDE_series_convergence_l1338_133873

def series_term (n : ℕ) : ℚ :=
  (4 * n + 3) / ((4 * n - 2)^2 * (4 * n + 2)^2)

def series_sum : ℚ := 1 / 128

theorem series_convergence : 
  (∑' n, series_term n) = series_sum :=
sorry

end NUMINAMATH_CALUDE_series_convergence_l1338_133873


namespace NUMINAMATH_CALUDE_expression_equality_l1338_133835

theorem expression_equality : 
  -21 * (2/3) + 3 * (1/4) - (-2/3) - (1/4) = -18 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1338_133835


namespace NUMINAMATH_CALUDE_matching_jelly_bean_probability_l1338_133834

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  green : ℕ
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Calculates the total number of jelly beans a person has -/
def total_jelly_beans (jb : JellyBeans) : ℕ :=
  jb.green + jb.red + jb.blue + jb.yellow

/-- Abe's jelly bean distribution -/
def abe_jelly_beans : JellyBeans :=
  { green := 1, red := 1, blue := 1, yellow := 0 }

/-- Bob's jelly bean distribution -/
def bob_jelly_beans : JellyBeans :=
  { green := 2, red := 3, blue := 0, yellow := 2 }

/-- Calculates the probability of two people showing the same color jelly bean -/
def matching_color_probability (person1 person2 : JellyBeans) : ℚ :=
  let total1 := total_jelly_beans person1
  let total2 := total_jelly_beans person2
  (person1.green * person2.green + person1.red * person2.red + person1.blue * person2.blue) / (total1 * total2)

theorem matching_jelly_bean_probability :
  matching_color_probability abe_jelly_beans bob_jelly_beans = 5 / 21 := by
  sorry

end NUMINAMATH_CALUDE_matching_jelly_bean_probability_l1338_133834


namespace NUMINAMATH_CALUDE_function_value_at_pi_over_four_l1338_133844

theorem function_value_at_pi_over_four (φ : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sin (x + 2 * φ) - 2 * Real.sin φ * Real.cos (x + φ)
  f (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_pi_over_four_l1338_133844


namespace NUMINAMATH_CALUDE_carla_teaches_23_students_l1338_133884

/-- The number of students Carla teaches -/
def total_students : ℕ :=
  let students_in_restroom : ℕ := 2
  let absent_students : ℕ := 3 * students_in_restroom - 1
  let total_desks : ℕ := 4 * 6
  let occupied_desks : ℕ := (2 * total_desks) / 3
  occupied_desks + students_in_restroom + absent_students

/-- Theorem stating that Carla teaches 23 students -/
theorem carla_teaches_23_students : total_students = 23 := by
  sorry

end NUMINAMATH_CALUDE_carla_teaches_23_students_l1338_133884


namespace NUMINAMATH_CALUDE_size_relationship_l1338_133890

theorem size_relationship (x : ℝ) : 
  let a := x^2 + x + Real.sqrt 2
  let b := Real.log 3 / Real.log 10
  let c := Real.exp (-1/2)
  b < c ∧ c < a := by sorry

end NUMINAMATH_CALUDE_size_relationship_l1338_133890


namespace NUMINAMATH_CALUDE_sphere_centers_distance_l1338_133863

/-- The distance between the centers of two spheres with masses M and m, 
    where a point B exists such that both spheres exert equal gravitational force on it,
    and A is a point between the centers with distance d from B. -/
theorem sphere_centers_distance (M m d : ℝ) (hM : M > 0) (hm : m > 0) (hd : d > 0) : 
  ∃ (distance : ℝ), distance = d / 2 * (M - m) / Real.sqrt (M * m) :=
sorry

end NUMINAMATH_CALUDE_sphere_centers_distance_l1338_133863


namespace NUMINAMATH_CALUDE_factorization_x_squared_minus_2x_l1338_133874

theorem factorization_x_squared_minus_2x (x : ℝ) : x^2 - 2*x = x*(x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_minus_2x_l1338_133874


namespace NUMINAMATH_CALUDE_min_distance_ellipse_to_N_l1338_133855

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x + 3)^2 + y^2) + Real.sqrt ((x - 3)^2 + y^2) = 10

/-- The fixed point N -/
def N : ℝ × ℝ := (-6, 0)

/-- The minimum distance from a point on the ellipse to N -/
def min_distance_to_N : ℝ := 1

/-- Theorem stating the minimum distance from any point on the ellipse to N is 1 -/
theorem min_distance_ellipse_to_N :
  ∀ x y : ℝ, ellipse_equation x y →
  ∃ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y ∧
  ∀ (q : ℝ × ℝ), ellipse_equation q.1 q.2 →
  dist p N ≤ dist q N ∧ dist p N = min_distance_to_N :=
sorry

end NUMINAMATH_CALUDE_min_distance_ellipse_to_N_l1338_133855


namespace NUMINAMATH_CALUDE_new_person_weight_l1338_133818

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℚ) (replaced_weight : ℚ) :
  initial_count = 10 →
  weight_increase = 5/2 →
  replaced_weight = 50 →
  ∃ (new_weight : ℚ),
    new_weight = replaced_weight + (initial_count * weight_increase) ∧
    new_weight = 75 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1338_133818


namespace NUMINAMATH_CALUDE_max_distance_difference_l1338_133862

-- Define the curve C₂
def C₂ (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (1, 1)

-- Define the distance function
def dist_squared (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

theorem max_distance_difference :
  ∃ (max : ℝ), max = 2 + 2 * Real.sqrt 39 ∧
  ∀ (P : ℝ × ℝ), C₂ P.1 P.2 →
    dist_squared P A - dist_squared P B ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_distance_difference_l1338_133862


namespace NUMINAMATH_CALUDE_modified_chessboard_no_tiling_l1338_133871

/-- Represents a chessboard cell --/
inductive Cell
| White
| Black

/-- Represents a 2x1 tile --/
structure Tile :=
  (first : Cell)
  (second : Cell)

/-- Represents the modified chessboard --/
def ModifiedChessboard : Type :=
  Fin 8 → Fin 8 → Option Cell

/-- A valid 2x1 tile covers one white and one black cell --/
def isValidTile (t : Tile) : Prop :=
  (t.first = Cell.White ∧ t.second = Cell.Black) ∨
  (t.first = Cell.Black ∧ t.second = Cell.White)

/-- A tiling of the modified chessboard --/
def Tiling : Type :=
  List Tile

/-- Checks if a tiling is valid for the modified chessboard --/
def isValidTiling (t : Tiling) (mb : ModifiedChessboard) : Prop :=
  sorry

theorem modified_chessboard_no_tiling :
  ∀ (mb : ModifiedChessboard),
    (mb 0 0 = none) →  -- Bottom-left square removed
    (mb 7 7 = none) →  -- Top-right square removed
    (∀ i j, i ≠ 0 ∨ j ≠ 0 → i ≠ 7 ∨ j ≠ 7 → mb i j ≠ none) →  -- All other squares present
    (∀ i j, (i + j) % 2 = 0 → mb i j = some Cell.White) →  -- White cells
    (∀ i j, (i + j) % 2 = 1 → mb i j = some Cell.Black) →  -- Black cells
    ¬∃ (t : Tiling), isValidTiling t mb :=
by
  sorry

end NUMINAMATH_CALUDE_modified_chessboard_no_tiling_l1338_133871


namespace NUMINAMATH_CALUDE_jogging_challenge_l1338_133806

theorem jogging_challenge (monday_distance : Real) (daily_increase : Real) 
  (saturday_multiplier : Real) (weekly_goal : Real) :
  let tuesday_distance := monday_distance * (1 + daily_increase)
  let thursday_distance := tuesday_distance * (1 + daily_increase)
  let saturday_distance := thursday_distance * saturday_multiplier
  let sunday_distance := weekly_goal - (monday_distance + tuesday_distance + thursday_distance + saturday_distance)
  monday_distance = 3 ∧ 
  daily_increase = 0.1 ∧ 
  saturday_multiplier = 2.5 ∧ 
  weekly_goal = 40 →
  tuesday_distance = 3.3 ∧ 
  thursday_distance = 3.63 ∧ 
  saturday_distance = 9.075 ∧ 
  sunday_distance = 21.995 := by
  sorry

end NUMINAMATH_CALUDE_jogging_challenge_l1338_133806


namespace NUMINAMATH_CALUDE_power_of_three_equivalence_l1338_133812

theorem power_of_three_equivalence : 
  (1 / 2 : ℝ) * (3 : ℝ)^21 - (1 / 3 : ℝ) * (3 : ℝ)^20 = (7 / 6 : ℝ) * (3 : ℝ)^20 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_equivalence_l1338_133812


namespace NUMINAMATH_CALUDE_expression_evaluation_l1338_133833

theorem expression_evaluation : 2 - (-3) - 4 + (-5) + 6 - (-7) - 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1338_133833


namespace NUMINAMATH_CALUDE_outfit_combinations_l1338_133836

theorem outfit_combinations (shirts : ℕ) (pants : ℕ) (hats : ℕ) : 
  shirts = 5 → pants = 4 → hats = 2 → shirts * pants * hats = 40 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l1338_133836


namespace NUMINAMATH_CALUDE_jesse_money_left_l1338_133867

def jesse_shopping (initial_amount : ℕ) (novel_cost : ℕ) : ℕ :=
  let lunch_cost := 2 * novel_cost
  initial_amount - (novel_cost + lunch_cost)

theorem jesse_money_left : jesse_shopping 50 7 = 29 := by
  sorry

end NUMINAMATH_CALUDE_jesse_money_left_l1338_133867


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l1338_133837

/-- The line x + ay = 3 is tangent to the circle (x-1)² + y² = 2 if and only if a = ±1 -/
theorem line_tangent_to_circle (a : ℝ) : 
  (∀ x y : ℝ, (x + a * y = 3) → ((x - 1)^2 + y^2 = 2) → 
   (∀ x' y' : ℝ, (x' + a * y' = 3) → ((x' - 1)^2 + y'^2 ≥ 2))) ↔ 
  (a = 1 ∨ a = -1) :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l1338_133837


namespace NUMINAMATH_CALUDE_triplet_satisfies_equations_l1338_133851

theorem triplet_satisfies_equations : ∃ (x y z : ℂ),
  x + y + z = 5 ∧
  x^2 + y^2 + z^2 = 19 ∧
  x^3 + y^3 + z^3 = 53 ∧
  x = -1 ∧ y = Complex.I * Real.sqrt 3 ∧ z = -Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triplet_satisfies_equations_l1338_133851


namespace NUMINAMATH_CALUDE_inequalities_proof_l1338_133840

theorem inequalities_proof (a b c : ℝ) (h1 : a > 0) (h2 : a > b) (h3 : b > c) : 
  (a * b > b * c) ∧ (a * c > b * c) ∧ (a * b > a * c) ∧ (a + b > b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l1338_133840


namespace NUMINAMATH_CALUDE_even_mono_decreasing_range_l1338_133895

/-- A function f: ℝ → ℝ is even -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is monotonically decreasing on [0,+∞) -/
def IsMonoDecreasingOnNonnegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

theorem even_mono_decreasing_range (f : ℝ → ℝ) (m : ℝ) 
    (h_even : IsEven f) 
    (h_mono : IsMonoDecreasingOnNonnegative f) 
    (h_ineq : f m > f (1 - m)) : 
  m < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_even_mono_decreasing_range_l1338_133895
