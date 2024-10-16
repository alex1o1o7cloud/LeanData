import Mathlib

namespace NUMINAMATH_CALUDE_range_of_x_range_of_m_l2487_248774

-- Problem 1
theorem range_of_x (x : ℝ) : (4*x - 3)^2 ≤ 1 → 1/2 ≤ x ∧ x ≤ 1 := by sorry

-- Problem 2
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, 4*x + m < 0 → x^2 - x - 2 > 0) → m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_range_of_x_range_of_m_l2487_248774


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2487_248734

theorem quadratic_roots_property (m n : ℝ) : 
  (m^2 + 2*m - 5 = 0) → (n^2 + 2*n - 5 = 0) → (m^2 + m*n + 2*m = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2487_248734


namespace NUMINAMATH_CALUDE_no_prime_square_diff_4048_l2487_248785

theorem no_prime_square_diff_4048 : ¬ ∃ (p q : ℕ), Prime p ∧ Prime q ∧ p^2 - q^2 = 4048 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_square_diff_4048_l2487_248785


namespace NUMINAMATH_CALUDE_slopes_equal_implies_parallel_false_l2487_248790

/-- A line in a 2D plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Two lines are parallel if they have the same slope --/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

/-- Statement: If two lines have the same slope, they are parallel --/
theorem slopes_equal_implies_parallel_false :
  ¬ (∀ (l1 l2 : Line), l1.slope = l2.slope → parallel l1 l2) := by
  sorry

end NUMINAMATH_CALUDE_slopes_equal_implies_parallel_false_l2487_248790


namespace NUMINAMATH_CALUDE_point_B_in_third_quadrant_l2487_248799

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def is_in_second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Definition of the third quadrant -/
def is_in_third_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Theorem: If A(m, -n) is in the second quadrant, then B(-mn, m) is in the third quadrant -/
theorem point_B_in_third_quadrant 
  (m n : ℝ) 
  (h : is_in_second_quadrant ⟨m, -n⟩) : 
  is_in_third_quadrant ⟨-m*n, m⟩ := by
  sorry

end NUMINAMATH_CALUDE_point_B_in_third_quadrant_l2487_248799


namespace NUMINAMATH_CALUDE_sum_of_arithmetic_sequence_l2487_248759

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem sum_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a5 : a 5 = 3)
  (h_a6 : a 6 = -2) :
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_arithmetic_sequence_l2487_248759


namespace NUMINAMATH_CALUDE_back_seat_holds_eight_l2487_248726

/-- Represents the seating capacity of a bus with specific arrangements --/
structure BusSeating where
  left_seats : Nat
  right_seats : Nat
  people_per_seat : Nat
  total_capacity : Nat

/-- Calculates the number of people that can be seated at the back of the bus --/
def back_seat_capacity (bus : BusSeating) : Nat :=
  bus.total_capacity - (bus.left_seats + bus.right_seats) * bus.people_per_seat

/-- Theorem stating that for the given bus configuration, the back seat can hold 8 people --/
theorem back_seat_holds_eight :
  let bus : BusSeating := {
    left_seats := 15,
    right_seats := 12,
    people_per_seat := 3,
    total_capacity := 89
  }
  back_seat_capacity bus = 8 := by
  sorry

#eval back_seat_capacity {
  left_seats := 15,
  right_seats := 12,
  people_per_seat := 3,
  total_capacity := 89
}

end NUMINAMATH_CALUDE_back_seat_holds_eight_l2487_248726


namespace NUMINAMATH_CALUDE_fraction_percent_of_x_l2487_248716

theorem fraction_percent_of_x (x : ℝ) (h : x > 0) : (x / 10 + x / 25) / x * 100 = 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_percent_of_x_l2487_248716


namespace NUMINAMATH_CALUDE_tan_seven_pi_sixths_l2487_248727

theorem tan_seven_pi_sixths : Real.tan (7 * π / 6) = 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_seven_pi_sixths_l2487_248727


namespace NUMINAMATH_CALUDE_max_value_of_f_l2487_248750

/-- The sum of the first n terms of the sequence a_n -/
def S (n : ℕ) : ℕ := 2^n - 1

/-- The nth term of the sequence a_n -/
def a (n : ℕ) : ℕ :=
  if n = 1 then 1 else 2^(n-1)

/-- The expression to be maximized -/
def f (n : ℕ) : ℚ :=
  (a n : ℚ) / ((a n * S n : ℕ) + a 6 : ℚ)

theorem max_value_of_f :
  ∀ n : ℕ, n ≥ 1 → f n ≤ 1/15 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2487_248750


namespace NUMINAMATH_CALUDE_room_length_calculation_l2487_248752

theorem room_length_calculation (area : ℝ) (width : ℝ) (length : ℝ) 
  (h1 : area = 10) 
  (h2 : width = 2) 
  (h3 : area = length * width) : 
  length = 5 := by
sorry

end NUMINAMATH_CALUDE_room_length_calculation_l2487_248752


namespace NUMINAMATH_CALUDE_inequalities_proof_l2487_248763

theorem inequalities_proof :
  (((12 : ℝ) / 11) ^ 11 > ((11 : ℝ) / 10) ^ 10) ∧
  (((12 : ℝ) / 11) ^ 12 < ((11 : ℝ) / 10) ^ 11) ∧
  (((12 : ℝ) / 11) ^ 10 > ((11 : ℝ) / 10) ^ 9) ∧
  (((11 : ℝ) / 10) ^ 12 > ((12 : ℝ) / 11) ^ 13) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_proof_l2487_248763


namespace NUMINAMATH_CALUDE_square_of_98_l2487_248770

theorem square_of_98 : (98 : ℕ) ^ 2 = 9604 := by sorry

end NUMINAMATH_CALUDE_square_of_98_l2487_248770


namespace NUMINAMATH_CALUDE_toad_ratio_proof_l2487_248767

/-- Proves that the ratio of Sarah's toads to Jim's toads is 2 --/
theorem toad_ratio_proof (tim_toads jim_toads sarah_toads : ℕ) : 
  tim_toads = 30 →
  jim_toads = tim_toads + 20 →
  sarah_toads = 100 →
  sarah_toads / jim_toads = 2 := by
sorry

end NUMINAMATH_CALUDE_toad_ratio_proof_l2487_248767


namespace NUMINAMATH_CALUDE_interior_exterior_angle_ratio_octagon_l2487_248741

/-- The ratio of an interior angle to an exterior angle in a regular octagon is 3:1 -/
theorem interior_exterior_angle_ratio_octagon : 
  ∀ (interior_angle exterior_angle : ℝ),
  interior_angle > 0 → 
  exterior_angle > 0 →
  (∀ (n : ℕ), n = 8 → interior_angle = (n - 2) * 180 / n) →
  (∀ (n : ℕ), n = 8 → exterior_angle = 360 / n) →
  interior_angle / exterior_angle = 3 := by
sorry

end NUMINAMATH_CALUDE_interior_exterior_angle_ratio_octagon_l2487_248741


namespace NUMINAMATH_CALUDE_three_divided_by_p_l2487_248703

theorem three_divided_by_p (p q : ℝ) 
  (h1 : 3 / q = 18) 
  (h2 : p - q = 0.33333333333333337) : 
  3 / p = 6 := by
  sorry

end NUMINAMATH_CALUDE_three_divided_by_p_l2487_248703


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2487_248749

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 3| = |x - 5| :=
by
  -- The unique solution is x = 4
  use 4
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2487_248749


namespace NUMINAMATH_CALUDE_parallelogram_count_l2487_248789

/-- 
Given an equilateral triangle ABC where each side is divided into n equal parts
and lines are drawn parallel to each side through these division points,
the total number of parallelograms formed is 3 * (n+1)^2 * n^2 / 4.
-/
theorem parallelogram_count (n : ℕ) : 
  (3 : ℚ) * (n + 1)^2 * n^2 / 4 = 3 * Nat.choose (n + 2) 4 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_count_l2487_248789


namespace NUMINAMATH_CALUDE_triangle_acuteness_l2487_248761

theorem triangle_acuteness (a b c : ℝ) (n : ℕ) (h1 : n > 2) 
  (h2 : a > 0) (h3 : b > 0) (h4 : c > 0)
  (h5 : a + b > c) (h6 : b + c > a) (h7 : c + a > b)
  (h8 : a^n + b^n = c^n) : 
  a^2 + b^2 > c^2 := by
  sorry

#check triangle_acuteness

end NUMINAMATH_CALUDE_triangle_acuteness_l2487_248761


namespace NUMINAMATH_CALUDE_necessary_condition_range_l2487_248737

theorem necessary_condition_range (a : ℝ) : 
  (∀ x : ℝ, x < a + 2 → x ≤ 2) → a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_necessary_condition_range_l2487_248737


namespace NUMINAMATH_CALUDE_triangle_properties_l2487_248715

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that under certain conditions, angle C and the sum of sines of A and B
    have specific values. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  c^2 = a^2 + b^2 + a*b →
  c = 4 * Real.sqrt 7 →
  a + b + c = 12 + 4 * Real.sqrt 7 →
  C = 2 * Real.pi / 3 ∧
  Real.sin A + Real.sin B = 3 * Real.sqrt 21 / 28 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2487_248715


namespace NUMINAMATH_CALUDE_range_of_a_l2487_248721

def p (a : ℝ) : Prop := ∃ x ∈ Set.Icc 0 (Real.pi / 2), Real.cos x ^ 2 + 2 * Real.cos x - a = 0

def q (a : ℝ) : Prop := ∀ x : ℝ, x ^ 2 + 2 * a * x - 8 + 6 * a ≥ 0

theorem range_of_a :
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a ∈ Set.Ioo 0 2 ∪ Set.Ioo 3 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2487_248721


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l2487_248778

/-- The ratio of area to perimeter for an equilateral triangle with side length 6 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let s : ℝ := 6
  let area : ℝ := s^2 * Real.sqrt 3 / 4
  let perimeter : ℝ := 3 * s
  area / perimeter = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l2487_248778


namespace NUMINAMATH_CALUDE_initial_snatch_weight_l2487_248747

/-- Represents John's weightlifting progress --/
structure Weightlifter where
  initialCleanAndJerk : ℝ
  initialSnatch : ℝ
  newCleanAndJerk : ℝ
  newSnatch : ℝ
  newTotal : ℝ

/-- Theorem stating that given the conditions, John's initial Snatch weight was 50 kg --/
theorem initial_snatch_weight (john : Weightlifter) :
  john.initialCleanAndJerk = 80 ∧
  john.newCleanAndJerk = 2 * john.initialCleanAndJerk ∧
  john.newSnatch = 1.8 * john.initialSnatch ∧
  john.newTotal = 250 ∧
  john.newTotal = john.newCleanAndJerk + john.newSnatch →
  john.initialSnatch = 50 := by
  sorry

#check initial_snatch_weight

end NUMINAMATH_CALUDE_initial_snatch_weight_l2487_248747


namespace NUMINAMATH_CALUDE_total_paintable_area_l2487_248765

/-- Represents the dimensions of a bedroom -/
structure Bedroom where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the wall area of a bedroom -/
def wallArea (b : Bedroom) : ℝ :=
  2 * (b.length * b.height + b.width * b.height)

/-- Calculates the paintable area of a bedroom -/
def paintableArea (b : Bedroom) (unpaintableArea : ℝ) : ℝ :=
  wallArea b - unpaintableArea

/-- The four bedrooms in Isabella's house -/
def isabellasBedrooms : List Bedroom := [
  { length := 14, width := 12, height := 9 },
  { length := 13, width := 11, height := 9 },
  { length := 15, width := 10, height := 9 },
  { length := 12, width := 12, height := 9 }
]

/-- The area occupied by doorways and windows in each bedroom -/
def unpaintableAreaPerRoom : ℝ := 70

/-- Theorem: The total area of walls to be painted in Isabella's house is 1502 square feet -/
theorem total_paintable_area :
  (isabellasBedrooms.map (fun b => paintableArea b unpaintableAreaPerRoom)).sum = 1502 := by
  sorry

end NUMINAMATH_CALUDE_total_paintable_area_l2487_248765


namespace NUMINAMATH_CALUDE_not_equal_1990_l2487_248797

/-- Count of positive integers ≤ pqn that have a common divisor with pq -/
def f (p q n : ℕ) : ℕ := 
  (n * p) + (n * q) - n

theorem not_equal_1990 (p q n : ℕ) (hp : Prime p) (hq : Prime q) (hpq : p ≠ q) (hn : n > 0) :
  (f p q n : ℚ) / n ≠ 1990 := by
  sorry

end NUMINAMATH_CALUDE_not_equal_1990_l2487_248797


namespace NUMINAMATH_CALUDE_budgets_equal_in_1996_l2487_248755

/-- Represents the year when the budgets of two projects become equal -/
def year_budgets_equal (initial_q initial_v increase_q decrease_v : ℕ) : ℕ :=
  let n : ℕ := (initial_v - initial_q) / (increase_q + decrease_v)
  1990 + n

/-- Theorem stating that the budgets become equal in 1996 -/
theorem budgets_equal_in_1996 :
  year_budgets_equal 540000 780000 30000 10000 = 1996 := by
  sorry

end NUMINAMATH_CALUDE_budgets_equal_in_1996_l2487_248755


namespace NUMINAMATH_CALUDE_a_share_of_profit_l2487_248796

/-- Calculates the share of profit for an investor in a partnership business -/
def calculate_share_of_profit (investment_A investment_B investment_C total_profit : ℚ) : ℚ :=
  (investment_A / (investment_A + investment_B + investment_C)) * total_profit

/-- Theorem: A's share of the profit is 3780 given the investments and total profit -/
theorem a_share_of_profit (investment_A investment_B investment_C total_profit : ℚ) 
  (h1 : investment_A = 6300)
  (h2 : investment_B = 4200)
  (h3 : investment_C = 10500)
  (h4 : total_profit = 12600) :
  calculate_share_of_profit investment_A investment_B investment_C total_profit = 3780 := by
  sorry

end NUMINAMATH_CALUDE_a_share_of_profit_l2487_248796


namespace NUMINAMATH_CALUDE_bread_cost_l2487_248784

/-- Prove that the cost of the bread is $1.25 given the conditions --/
theorem bread_cost (total_cost change_nickels : ℚ) 
  (h1 : total_cost = 205/100)  -- Total cost is $2.05
  (h2 : change_nickels = 8 * 5/100)  -- 8 nickels in change
  (h3 : ∃ (change_quarter change_dime : ℚ), 
    change_quarter = 25/100 ∧ 
    change_dime = 10/100 ∧ 
    700/100 - total_cost = change_quarter + change_dime + change_nickels + 420/100) 
  : ∃ (bread_cost cheese_cost : ℚ), 
    bread_cost = 125/100 ∧ 
    cheese_cost = 80/100 ∧ 
    bread_cost + cheese_cost = total_cost := by
  sorry

end NUMINAMATH_CALUDE_bread_cost_l2487_248784


namespace NUMINAMATH_CALUDE_intersection_when_a_10_subset_condition_l2487_248723

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 22}

-- Theorem for part 1
theorem intersection_when_a_10 :
  A 10 ∩ B = {x | 21 ≤ x ∧ x ≤ 22} := by sorry

-- Theorem for part 2
theorem subset_condition :
  ∀ a : ℝ, A a ⊆ B ↔ a ≤ 9 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_10_subset_condition_l2487_248723


namespace NUMINAMATH_CALUDE_smallest_d_value_l2487_248764

def no_triangle (a b c : ℝ) : Prop :=
  a + b ≤ c ∨ a + c ≤ b ∨ b + c ≤ a

theorem smallest_d_value (c d : ℝ) 
  (h1 : 2 < c ∧ c < d)
  (h2 : no_triangle 2 c d)
  (h3 : no_triangle (1/d) (1/c) 2) :
  d = 2 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_smallest_d_value_l2487_248764


namespace NUMINAMATH_CALUDE_probability_prime_and_power_of_2_l2487_248793

/-- The set of prime numbers between 1 and 8 (inclusive) -/
def primes_1_to_8 : Finset Nat := {2, 3, 5, 7}

/-- The set of powers of 2 between 1 and 8 (inclusive) -/
def powers_of_2_1_to_8 : Finset Nat := {1, 2, 4, 8}

/-- The number of sides on each die -/
def die_sides : Nat := 8

theorem probability_prime_and_power_of_2 :
  (Finset.card primes_1_to_8 * Finset.card powers_of_2_1_to_8) / (die_sides * die_sides) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_prime_and_power_of_2_l2487_248793


namespace NUMINAMATH_CALUDE_negate_positive_negate_negative_positive_negative_positive_positive_l2487_248744

-- Define the operations
def negate (x : ℝ) : ℝ := -x
def positive (x : ℝ) : ℝ := x

-- Theorem statements
theorem negate_positive (x : ℝ) : negate (positive x) = -x := by sorry

theorem negate_negative (x : ℝ) : negate (negate x) = x := by sorry

theorem positive_negative (x : ℝ) : positive (negate x) = -x := by sorry

theorem positive_positive (x : ℝ) : positive (positive x) = x := by sorry

end NUMINAMATH_CALUDE_negate_positive_negate_negative_positive_negative_positive_positive_l2487_248744


namespace NUMINAMATH_CALUDE_eunji_has_most_marbles_l2487_248783

def minyoung_marbles : ℕ := 4
def yujeong_marbles : ℕ := 2
def eunji_marbles : ℕ := minyoung_marbles + 1

theorem eunji_has_most_marbles :
  eunji_marbles > minyoung_marbles ∧ eunji_marbles > yujeong_marbles :=
by
  sorry

end NUMINAMATH_CALUDE_eunji_has_most_marbles_l2487_248783


namespace NUMINAMATH_CALUDE_tim_score_l2487_248717

/-- Represents the scores of players in a basketball game -/
structure BasketballScores where
  joe : ℕ
  tim : ℕ
  ken : ℕ

/-- Theorem: Tim's score is 30 points given the conditions of the basketball game -/
theorem tim_score (scores : BasketballScores) : scores.tim = 30 :=
  by
  have h1 : scores.tim = scores.joe + 20 := by sorry
  have h2 : scores.tim * 2 = scores.ken := by sorry
  have h3 : scores.joe + scores.tim + scores.ken = 100 := by sorry
  sorry

#check tim_score

end NUMINAMATH_CALUDE_tim_score_l2487_248717


namespace NUMINAMATH_CALUDE_parallel_transitivity_counterexample_l2487_248772

-- Define the types for lines and planes in 3D space
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Define the subset relation for a line being in a plane
variable (subset : Line → Plane → Prop)

theorem parallel_transitivity_counterexample 
  (m n : Line) (α β : Plane) :
  ¬(∀ (m n : Line) (α β : Plane), 
    parallel m n → 
    parallel_line_plane n α → 
    parallel_plane α β → 
    parallel_line_plane m β) :=
sorry

end NUMINAMATH_CALUDE_parallel_transitivity_counterexample_l2487_248772


namespace NUMINAMATH_CALUDE_rhombus_matches_l2487_248768

/-- Represents the number of matches needed for a rhombus -/
def matches_for_rhombus (s : ℕ) : ℕ := s * (s + 3)

/-- Theorem: The number of matches needed for a rhombus with side length s,
    divided into unit triangles, is s(s+3) -/
theorem rhombus_matches (s : ℕ) : 
  matches_for_rhombus s = s * (s + 3) := by
  sorry

#eval matches_for_rhombus 10  -- Should evaluate to 320

end NUMINAMATH_CALUDE_rhombus_matches_l2487_248768


namespace NUMINAMATH_CALUDE_circle_equation_solution_l2487_248791

theorem circle_equation_solution :
  ∃! (x y : ℝ), (x - 11)^2 + (y - 12)^2 + (x - y)^2 = 1/3 ∧ x = 11 + 1/3 ∧ y = 11 + 2/3 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_solution_l2487_248791


namespace NUMINAMATH_CALUDE_tom_build_time_l2487_248748

theorem tom_build_time (avery_time : ℝ) (joint_work_time : ℝ) (tom_finish_time : ℝ) :
  avery_time = 3 →
  joint_work_time = 1 →
  tom_finish_time = 39.99999999999999 / 60 →
  ∃ (tom_solo_time : ℝ),
    (1 / avery_time + 1 / tom_solo_time) * joint_work_time + 
    (1 / tom_solo_time) * tom_finish_time = 1 ∧
    tom_solo_time = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_tom_build_time_l2487_248748


namespace NUMINAMATH_CALUDE_product_of_four_consecutive_integers_divisible_by_24_l2487_248745

theorem product_of_four_consecutive_integers_divisible_by_24 (n : ℕ) :
  ∃ k : ℕ, n * (n + 1) * (n + 2) * (n + 3) = 24 * k ∧
  ∀ m : ℕ, m > 24 → ¬(∀ n : ℕ, ∃ k : ℕ, n * (n + 1) * (n + 2) * (n + 3) = m * k) :=
by sorry

end NUMINAMATH_CALUDE_product_of_four_consecutive_integers_divisible_by_24_l2487_248745


namespace NUMINAMATH_CALUDE_branch_fractions_sum_l2487_248775

theorem branch_fractions_sum : 
  (1/3 : ℚ) + (2/3 : ℚ) + (1/5 : ℚ) + (2/5 : ℚ) + (3/5 : ℚ) + (4/5 : ℚ) + 
  (1/7 : ℚ) + (2/7 : ℚ) + (3/7 : ℚ) + (4/7 : ℚ) + (5/7 : ℚ) + (6/7 : ℚ) = 1 :=
by sorry

end NUMINAMATH_CALUDE_branch_fractions_sum_l2487_248775


namespace NUMINAMATH_CALUDE_min_diameter_bounds_l2487_248709

/-- The minimum diameter of n points on a plane where the distance between any two points is at least 1 -/
def min_diameter (n : ℕ) : ℝ :=
  sorry

/-- The distance between any two points is at least 1 -/
axiom min_distance (n : ℕ) (i j : Fin n) (points : Fin n → ℝ × ℝ) :
  i ≠ j → dist (points i) (points j) ≥ 1

theorem min_diameter_bounds :
  (∀ n : ℕ, n = 2 ∨ n = 3 → min_diameter n ≥ 1) ∧
  (min_diameter 4 ≥ Real.sqrt 2) ∧
  (min_diameter 5 ≥ (1 + Real.sqrt 5) / 2) :=
sorry

end NUMINAMATH_CALUDE_min_diameter_bounds_l2487_248709


namespace NUMINAMATH_CALUDE_range_of_a_l2487_248792

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 1 - 2*a

def g (a : ℝ) (x : ℝ) : ℝ := |x - a| - a*x

def has_two_distinct_intersections (a : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc 0 1 ∧ x₂ ∈ Set.Icc 0 1 ∧ f a x₁ = 0 ∧ f a x₂ = 0

def has_minimum_value (a : ℝ) : Prop :=
  ∃ x₀, ∀ x, g a x₀ ≤ g a x

theorem range_of_a (a : ℝ) :
  a > 0 ∧ ¬(has_two_distinct_intersections a) ∧ has_minimum_value a →
  a ∈ Set.Ioo 0 (Real.sqrt 2 - 1) ∪ Set.Ioo (1/2) 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2487_248792


namespace NUMINAMATH_CALUDE_circle_equation_l2487_248706

/-- A circle in the 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane represented by ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a line --/
def Line.contains (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Checks if a circle is tangent to a line at a given point --/
def Circle.tangentTo (c : Circle) (l : Line) (p : ℝ × ℝ) : Prop :=
  l.contains p ∧
  (c.center.1 - p.1) ^ 2 + (c.center.2 - p.2) ^ 2 = c.radius ^ 2 ∧
  (c.center.1 - p.1) * l.a + (c.center.2 - p.2) * l.b = 0

/-- The main theorem --/
theorem circle_equation (c : Circle) :
  (c.center.2 = -4 * c.center.1) →  -- Center lies on y = -4x
  (c.tangentTo (Line.mk 1 1 (-1)) (3, -2)) →  -- Tangent to x + y - 1 = 0 at (3, -2)
  (∀ x y : ℝ, (x - 1)^2 + (y + 4)^2 = 8 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2487_248706


namespace NUMINAMATH_CALUDE_expression_simplification_l2487_248724

theorem expression_simplification (x y z : ℝ) 
  (hx : x ≠ 2) (hy : y ≠ 3) (hz : z ≠ 4) :
  (x - 2) / (6 - z) * (y - 3) / (2 - x) * (z - 4) / (3 - y) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2487_248724


namespace NUMINAMATH_CALUDE_number_classification_l2487_248735

-- Define a number type that can represent both decimal and natural numbers
inductive Number
  | Decimal (integerPart : Int) (fractionalPart : Nat)
  | Natural (value : Nat)

-- Define a function to check if a number is decimal
def isDecimal (n : Number) : Prop :=
  match n with
  | Number.Decimal _ _ => True
  | Number.Natural _ => False

-- Define a function to check if a number is natural
def isNatural (n : Number) : Prop :=
  match n with
  | Number.Decimal _ _ => False
  | Number.Natural _ => True

-- Theorem statement
theorem number_classification (n : Number) :
  (isDecimal n ∧ ¬isNatural n) ∨ (¬isDecimal n ∧ isNatural n) :=
by sorry

end NUMINAMATH_CALUDE_number_classification_l2487_248735


namespace NUMINAMATH_CALUDE_students_with_no_books_l2487_248742

/-- Represents the number of students who borrowed a specific number of books -/
structure BookBorrowers where
  zero : ℕ
  one : ℕ
  two : ℕ
  threeOrMore : ℕ

/-- The total number of students in the class -/
def totalStudents : ℕ := 40

/-- The average number of books borrowed per student -/
def averageBooks : ℚ := 2

/-- Calculates the total number of books borrowed -/
def totalBooksBorrowed (b : BookBorrowers) : ℕ :=
  0 * b.zero + 1 * b.one + 2 * b.two + 3 * b.threeOrMore

/-- Theorem stating the number of students who did not borrow books -/
theorem students_with_no_books (b : BookBorrowers) : 
  b.zero = 1 ∧ 
  b.one = 12 ∧ 
  b.two = 13 ∧ 
  b.zero + b.one + b.two + b.threeOrMore = totalStudents ∧
  (totalBooksBorrowed b : ℚ) / totalStudents = averageBooks :=
by
  sorry


end NUMINAMATH_CALUDE_students_with_no_books_l2487_248742


namespace NUMINAMATH_CALUDE_gold_hoard_problem_l2487_248730

theorem gold_hoard_problem (total_per_brother : ℝ) (eldest_gold : ℝ) (eldest_silver_fraction : ℝ)
  (total_silver : ℝ) (h1 : total_per_brother = 100)
  (h2 : eldest_gold = 30)
  (h3 : eldest_silver_fraction = 1/5)
  (h4 : total_silver = 350) :
  eldest_gold + (total_silver - eldest_silver_fraction * total_silver) = 50 := by
  sorry


end NUMINAMATH_CALUDE_gold_hoard_problem_l2487_248730


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l2487_248798

theorem divisibility_implies_equality (a b n : ℕ) :
  (∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) →
  a = b^n :=
by sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l2487_248798


namespace NUMINAMATH_CALUDE_price_reduction_l2487_248731

theorem price_reduction (x : ℝ) : 
  (100 - x) * 0.9 = 85.5 → x = 5 := by sorry

end NUMINAMATH_CALUDE_price_reduction_l2487_248731


namespace NUMINAMATH_CALUDE_symmetry_of_f_l2487_248738

def arithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def f (x a₁ a₂ : ℝ) : ℝ := |x - a₁| + |x - a₂|

theorem symmetry_of_f (a : ℕ → ℝ) (d : ℝ) (h : d ≠ 0) :
  arithmeticSequence a d →
  ∀ x : ℝ, f (((a 1) + (a 2)) / 2 - x) ((a 1) : ℝ) ((a 2) : ℝ) = 
           f (((a 1) + (a 2)) / 2 + x) ((a 1) : ℝ) ((a 2) : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_of_f_l2487_248738


namespace NUMINAMATH_CALUDE_decreasing_interval_of_f_l2487_248711

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x

def monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y ≤ f x

theorem decreasing_interval_of_f :
  ∃ (a b : ℝ), a = 5 * Real.pi / 6 ∧ b = Real.pi ∧
  monotonically_decreasing f a b ∧
  ∀ c d, 0 ≤ c ∧ d ≤ Real.pi ∧ c < d ∧ monotonically_decreasing f c d →
    a ≤ c ∧ d ≤ b :=
by sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_f_l2487_248711


namespace NUMINAMATH_CALUDE_police_force_ratio_l2487_248780

/-- Given a police force with female officers and officers on duty, prove the ratio of female officers to total officers on duty. -/
theorem police_force_ratio (total_female : ℕ) (total_on_duty : ℕ) (female_duty_percent : ℚ) : 
  total_female = 300 →
  total_on_duty = 240 →
  female_duty_percent = 2/5 →
  (female_duty_percent * total_female) / total_on_duty = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_police_force_ratio_l2487_248780


namespace NUMINAMATH_CALUDE_characterization_of_n_l2487_248722

/-- A bijection from {1, ..., n} to itself -/
def Bijection (n : ℕ) := { f : Fin n → Fin n // Function.Bijective f }

/-- The main theorem -/
theorem characterization_of_n (m : ℕ) (h_m : Even m) (h_m_pos : 0 < m) :
  ∀ n : ℕ, (∃ f : Bijection n,
    ∀ x y : Fin n, (m * x.val - y.val) % n = 0 →
      (n + 1) ∣ (f.val x).val^m - (f.val y).val) ↔
  Nat.Prime (n + 1) :=
sorry

end NUMINAMATH_CALUDE_characterization_of_n_l2487_248722


namespace NUMINAMATH_CALUDE_ab_equals_twelve_l2487_248754

-- Define the set A
def A (a b : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ b}

-- Define the complement of A
def complement_A : Set ℝ := {x | x < 3 ∨ x > 4}

-- Theorem statement
theorem ab_equals_twelve (a b : ℝ) : 
  A a b ∪ complement_A = Set.univ → a * b = 12 := by
  sorry

end NUMINAMATH_CALUDE_ab_equals_twelve_l2487_248754


namespace NUMINAMATH_CALUDE_whittlesworth_band_size_l2487_248720

theorem whittlesworth_band_size (n : ℕ) : 
  (20 * n % 28 = 6) →
  (20 * n % 19 = 5) →
  (20 * n < 1200) →
  (∀ m : ℕ, (20 * m % 28 = 6) → (20 * m % 19 = 5) → (20 * m < 1200) → m ≤ n) →
  20 * n = 2000 :=
by sorry

end NUMINAMATH_CALUDE_whittlesworth_band_size_l2487_248720


namespace NUMINAMATH_CALUDE_fourth_side_length_l2487_248736

/-- A quadrilateral inscribed in a circle with radius 300, where three sides have lengths 300, 300, and 150√2 -/
structure InscribedQuadrilateral where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The length of the first side -/
  side1 : ℝ
  /-- The length of the second side -/
  side2 : ℝ
  /-- The length of the third side -/
  side3 : ℝ
  /-- The length of the fourth side -/
  side4 : ℝ
  /-- Condition that the quadrilateral is inscribed in a circle with radius 300 -/
  radius_eq : radius = 300
  /-- Condition that two sides have length 300 -/
  side1_eq : side1 = 300
  side2_eq : side2 = 300
  /-- Condition that one side has length 150√2 -/
  side3_eq : side3 = 150 * Real.sqrt 2

/-- Theorem stating that the fourth side of the inscribed quadrilateral has length 450 -/
theorem fourth_side_length (q : InscribedQuadrilateral) : q.side4 = 450 := by
  sorry

end NUMINAMATH_CALUDE_fourth_side_length_l2487_248736


namespace NUMINAMATH_CALUDE_class_size_l2487_248758

theorem class_size (total : ℕ) (girls_ratio : ℚ) (boys : ℕ) : 
  girls_ratio = 5 / 8 → boys = 60 → total = 160 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l2487_248758


namespace NUMINAMATH_CALUDE_parallelogram_area_l2487_248766

/-- The area of a parallelogram with longer diagonal 5 and heights 2 and 3 -/
theorem parallelogram_area (d : ℝ) (h₁ h₂ : ℝ) (hd : d = 5) (hh₁ : h₁ = 2) (hh₂ : h₂ = 3) :
  (h₁ * h₂) / (((3 * Real.sqrt 21 + 8) / 25) : ℝ) = 150 / (3 * Real.sqrt 21 + 8) := by
sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2487_248766


namespace NUMINAMATH_CALUDE_train_crossing_tree_time_l2487_248712

/-- Given a train and a platform with specified lengths and the time it takes for the train to pass the platform, 
    calculate the time it takes for the train to cross a tree. -/
theorem train_crossing_tree_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (time_to_pass_platform : ℝ) 
  (h1 : train_length = 1200) 
  (h2 : platform_length = 300) 
  (h3 : time_to_pass_platform = 150) : 
  (train_length / ((train_length + platform_length) / time_to_pass_platform)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_tree_time_l2487_248712


namespace NUMINAMATH_CALUDE_simplify_product_l2487_248776

theorem simplify_product (a : ℝ) : 
  (2 * a) * (3 * a^2) * (5 * a^3) * (7 * a^4) * (11 * a^5) * (13 * a^6) = 30030 * a^21 := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_l2487_248776


namespace NUMINAMATH_CALUDE_problem_two_l2487_248757

theorem problem_two : -2.5 / (5/16) * (-1/8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_two_l2487_248757


namespace NUMINAMATH_CALUDE_muffin_division_l2487_248746

theorem muffin_division (num_friends : ℕ) (total_muffins : ℕ) : 
  num_friends = 4 → total_muffins = 20 → (total_muffins / (num_friends + 1) : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_muffin_division_l2487_248746


namespace NUMINAMATH_CALUDE_fraction_cube_equality_l2487_248760

theorem fraction_cube_equality : 
  (81000 : ℝ)^3 / (27000 : ℝ)^3 = 27 :=
by
  have h : (81000 : ℝ) = 3 * 27000 := by norm_num
  sorry

end NUMINAMATH_CALUDE_fraction_cube_equality_l2487_248760


namespace NUMINAMATH_CALUDE_john_number_theorem_l2487_248743

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def switch_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem john_number_theorem :
  ∃! x : ℕ, is_two_digit x ∧
    84 ≤ switch_digits (5 * x - 7) ∧
    switch_digits (5 * x - 7) ≤ 90 ∧
    x = 11 := by
  sorry

end NUMINAMATH_CALUDE_john_number_theorem_l2487_248743


namespace NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l2487_248740

-- Define the conditions P and Q
def P (x : ℝ) : Prop := |2*x - 3| < 1
def Q (x : ℝ) : Prop := x*(x - 3) < 0

-- Theorem stating that P is sufficient but not necessary for Q
theorem P_sufficient_not_necessary_for_Q :
  (∀ x : ℝ, P x → Q x) ∧ 
  (∃ x : ℝ, Q x ∧ ¬(P x)) :=
sorry

end NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l2487_248740


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_l2487_248771

theorem unique_number_with_three_prime_divisors (x n : ℕ) : 
  x = 6^n + 1 →
  Odd n →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ x = 11 * p * q) →
  (∀ r : ℕ, Prime r → r ∣ x → r = 11 ∨ r = p ∨ r = q) →
  x = 7777 := by
sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_l2487_248771


namespace NUMINAMATH_CALUDE_cone_volume_from_semicircle_l2487_248788

/-- The volume of a cone formed by rolling a semicircle -/
theorem cone_volume_from_semicircle (R : ℝ) (R_pos : R > 0) :
  ∃ (V : ℝ), V = (Real.sqrt 3 / 24) * π * R^3 ∧ 
  V = (1/3) * π * (R/2)^2 * (Real.sqrt 3 * R / 2) :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_from_semicircle_l2487_248788


namespace NUMINAMATH_CALUDE_correct_num_pigs_l2487_248725

/-- The number of pigs Randy has -/
def num_pigs : ℕ := 2

/-- The amount of feed per pig per day in pounds -/
def feed_per_pig_per_day : ℕ := 10

/-- The total amount of feed for all pigs per week in pounds -/
def total_feed_per_week : ℕ := 140

/-- Theorem stating that the number of pigs is correct given the feeding conditions -/
theorem correct_num_pigs : 
  num_pigs * feed_per_pig_per_day * 7 = total_feed_per_week := by
  sorry


end NUMINAMATH_CALUDE_correct_num_pigs_l2487_248725


namespace NUMINAMATH_CALUDE_sin_graph_shift_l2487_248769

noncomputable def f (x : ℝ) := Real.sin (2 * x)
noncomputable def g (x : ℝ) := Real.sin (2 * x + 1)

theorem sin_graph_shift :
  ∀ x : ℝ, g x = f (x + 1/2) := by sorry

end NUMINAMATH_CALUDE_sin_graph_shift_l2487_248769


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l2487_248704

theorem sum_of_solutions_is_zero (y : ℝ) (x₁ x₂ : ℝ) : 
  y = 10 → 
  x₁^2 + y^2 = 200 → 
  x₂^2 + y^2 = 200 → 
  x₁ + x₂ = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l2487_248704


namespace NUMINAMATH_CALUDE_tangent_point_implies_base_l2487_248753

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem tangent_point_implies_base (a : ℝ) :
  (∃ m : ℝ, f a m = (1/3) * m ∧ 
    (∀ x : ℝ, x > 0 → HasDerivAt (f a) ((1/3) : ℝ) m)) →
  a = Real.exp ((3:ℝ) / Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_tangent_point_implies_base_l2487_248753


namespace NUMINAMATH_CALUDE_min_distance_curve_to_line_l2487_248700

/-- The minimum distance from a point on the curve y = x^2 - ln x to the line y = x - 2 is √2 --/
theorem min_distance_curve_to_line :
  let f (x : ℝ) := x^2 - Real.log x
  let g (x : ℝ) := x - 2
  ∀ x > 0, ∃ y : ℝ, y = f x ∧
    (∀ x' > 0, ∃ y' : ℝ, y' = f x' →
      Real.sqrt 2 ≤ |y' - g x'|) ∧
    |y - g x| = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_curve_to_line_l2487_248700


namespace NUMINAMATH_CALUDE_oplus_problem_l2487_248781

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the operation ⊕
def oplus : Element → Element → Element
  | Element.one, Element.one => Element.two
  | Element.one, Element.two => Element.three
  | Element.one, Element.three => Element.four
  | Element.one, Element.four => Element.one
  | Element.two, Element.one => Element.three
  | Element.two, Element.two => Element.one
  | Element.two, Element.three => Element.two
  | Element.two, Element.four => Element.four
  | Element.three, Element.one => Element.four
  | Element.three, Element.two => Element.two
  | Element.three, Element.three => Element.one
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.one
  | Element.four, Element.two => Element.four
  | Element.four, Element.three => Element.three
  | Element.four, Element.four => Element.two

theorem oplus_problem :
  oplus (oplus Element.three Element.two) (oplus Element.four Element.one) = Element.three :=
by sorry

end NUMINAMATH_CALUDE_oplus_problem_l2487_248781


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2487_248707

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2487_248707


namespace NUMINAMATH_CALUDE_m_positive_l2487_248714

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x^2) - x) + 2 / (2^x + 1) + 1

theorem m_positive (m : ℝ) (h : f (m - 1) + f (1 - 2*m) > 4) : m > 0 := by
  sorry

end NUMINAMATH_CALUDE_m_positive_l2487_248714


namespace NUMINAMATH_CALUDE_garage_sale_pricing_l2487_248713

theorem garage_sale_pricing (total_items : ℕ) (n : ℕ) 
  (h1 : total_items = 42)
  (h2 : n < total_items)
  (h3 : n = total_items - 24) : n = 19 := by
  sorry

end NUMINAMATH_CALUDE_garage_sale_pricing_l2487_248713


namespace NUMINAMATH_CALUDE_exponential_function_extrema_l2487_248719

theorem exponential_function_extrema (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x
  let max_val := max (f 1) (f 2)
  let min_val := min (f 1) (f 2)
  max_val + min_val = 12 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_extrema_l2487_248719


namespace NUMINAMATH_CALUDE_union_of_sets_l2487_248794

theorem union_of_sets : 
  let A : Set ℤ := {0, 1, 2}
  let B : Set ℤ := {-1, 0}
  A ∪ B = {-1, 0, 1, 2} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l2487_248794


namespace NUMINAMATH_CALUDE_unripe_oranges_per_day_is_24_l2487_248701

/-- The number of sacks of unripe oranges harvested per day -/
def unripe_oranges_per_day : ℕ := 24

/-- The total number of sacks of unripe oranges after the harvest period -/
def total_unripe_oranges : ℕ := 1080

/-- The number of days in the harvest period -/
def harvest_days : ℕ := 45

/-- Theorem stating that the number of sacks of unripe oranges harvested per day is 24 -/
theorem unripe_oranges_per_day_is_24 : 
  unripe_oranges_per_day = total_unripe_oranges / harvest_days :=
by sorry

end NUMINAMATH_CALUDE_unripe_oranges_per_day_is_24_l2487_248701


namespace NUMINAMATH_CALUDE_matrix_multiplication_example_l2487_248705

theorem matrix_multiplication_example :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![5, -3; 2, 2]
  A * B = !![17, -7; 16, -16] := by
  sorry

end NUMINAMATH_CALUDE_matrix_multiplication_example_l2487_248705


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2487_248729

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- Properties of a specific arithmetic sequence -/
def SpecificSequence (seq : ArithmeticSequence) : Prop :=
  seq.S 5 < seq.S 6 ∧ seq.S 6 = seq.S 7 ∧ seq.S 7 > seq.S 8

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
  (h : SpecificSequence seq) : 
  (∃ d, ∀ n, seq.a (n + 1) - seq.a n = d ∧ d < 0) ∧ 
  seq.S 9 < seq.S 5 ∧
  seq.a 7 = 0 ∧
  (∀ n, seq.S n ≤ seq.S 6) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2487_248729


namespace NUMINAMATH_CALUDE_curve_C_extrema_l2487_248739

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 6 = 0

-- Define the function we want to maximize/minimize
def f (x y : ℝ) : ℝ := x + 2*y

-- State the theorem
theorem curve_C_extrema :
  (∀ x y : ℝ, C x y → 10 - Real.sqrt 6 ≤ f x y) ∧
  (∀ x y : ℝ, C x y → f x y ≤ 10 + Real.sqrt 6) ∧
  (∃ x₁ y₁ : ℝ, C x₁ y₁ ∧ f x₁ y₁ = 10 - Real.sqrt 6) ∧
  (∃ x₂ y₂ : ℝ, C x₂ y₂ ∧ f x₂ y₂ = 10 + Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_curve_C_extrema_l2487_248739


namespace NUMINAMATH_CALUDE_tub_drain_time_l2487_248779

/-- Represents the time it takes to drain a tub -/
def drainTime (initialFraction : ℚ) (drainedFraction : ℚ) (initialTime : ℚ) : ℚ :=
  (drainedFraction * initialTime) / initialFraction

theorem tub_drain_time :
  let initialFraction : ℚ := 5 / 7
  let remainingFraction : ℚ := 1 - initialFraction
  let initialTime : ℚ := 4
  drainTime initialFraction remainingFraction initialTime = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tub_drain_time_l2487_248779


namespace NUMINAMATH_CALUDE_parabola_vertex_y_zero_l2487_248708

/-- The parabola y = x^2 - 10x + d has its vertex at y = 0 when d = 25 -/
theorem parabola_vertex_y_zero (x y d : ℝ) : 
  y = x^2 - 10*x + d → 
  (∃ x₀, ∀ x, x^2 - 10*x + d ≥ x₀^2 - 10*x₀ + d) → 
  d = 25 ↔ x^2 - 10*x + d ≥ 0 ∧ ∃ x₁, x₁^2 - 10*x₁ + d = 0 := by
sorry


end NUMINAMATH_CALUDE_parabola_vertex_y_zero_l2487_248708


namespace NUMINAMATH_CALUDE_sector_area_l2487_248710

/-- Given a circular sector with arc length 3π and central angle 135°, prove its area is 6π. -/
theorem sector_area (r : ℝ) (θ : ℝ) : 
  r * θ = 3 * Real.pi → 
  θ = 135 * Real.pi / 180 →
  (1 / 2) * r^2 * θ = 6 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sector_area_l2487_248710


namespace NUMINAMATH_CALUDE_valid_numbers_l2487_248787

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (let a := n / 100
   let b := (n / 10) % 10
   let c := n % 10
   let new_n := n + 3
   let new_a := new_n / 100
   let new_b := (new_n / 10) % 10
   let new_c := new_n % 10
   a + b + c = 3 * (new_a + new_b + new_c))

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {108, 117, 207} :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l2487_248787


namespace NUMINAMATH_CALUDE_cube_path_exists_l2487_248756

/-- Represents a cell on the chessboard --/
structure Cell :=
  (x : Fin 8)
  (y : Fin 8)

/-- Represents a face of the cube --/
inductive Face
  | Top
  | Bottom
  | North
  | South
  | East
  | West

/-- Represents the state of the cube on the board --/
structure CubeState :=
  (position : Cell)
  (topFace : Face)

/-- Represents a move of the cube --/
inductive Move
  | North
  | South
  | East
  | West

/-- Function to apply a move to a cube state --/
def applyMove (state : CubeState) (move : Move) : CubeState :=
  sorry

/-- Predicate to check if a cell has been visited --/
def hasVisited (cell : Cell) (path : List CubeState) : Prop :=
  sorry

/-- Theorem: There exists a path for the cube that visits all cells while keeping one face never touching the board --/
theorem cube_path_exists : 
  ∃ (initialState : CubeState) (path : List Move),
    (∀ cell : Cell, hasVisited cell (initialState :: (List.scanl applyMove initialState path))) ∧
    (∃ face : Face, ∀ state ∈ (initialState :: (List.scanl applyMove initialState path)), state.topFace ≠ face) :=
  sorry

end NUMINAMATH_CALUDE_cube_path_exists_l2487_248756


namespace NUMINAMATH_CALUDE_opposite_solutions_k_value_l2487_248728

theorem opposite_solutions_k_value (x y k : ℝ) : 
  (2 * x + 5 * y = k) → 
  (x - 4 * y = 15) → 
  (x + y = 0) → 
  k = -9 := by
sorry

end NUMINAMATH_CALUDE_opposite_solutions_k_value_l2487_248728


namespace NUMINAMATH_CALUDE_two_children_gender_combinations_l2487_248702

-- Define the possible genders
inductive Gender
| Male
| Female

-- Define a type for a pair of children's genders
def ChildrenGenders := (Gender × Gender)

-- Define the set of all possible gender combinations
def allGenderCombinations : Set ChildrenGenders :=
  {(Gender.Male, Gender.Male), (Gender.Male, Gender.Female),
   (Gender.Female, Gender.Male), (Gender.Female, Gender.Female)}

-- Theorem statement
theorem two_children_gender_combinations :
  ∀ (family : ChildrenGenders), family ∈ allGenderCombinations :=
by sorry

end NUMINAMATH_CALUDE_two_children_gender_combinations_l2487_248702


namespace NUMINAMATH_CALUDE_line_hyperbola_intersection_l2487_248733

theorem line_hyperbola_intersection (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ 
    x₁^2 - (k*x₁ + 1)^2 = 1 ∧ 
    x₂^2 - (k*x₂ + 1)^2 = 1) → 
  k > 1 ∧ k < Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_line_hyperbola_intersection_l2487_248733


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_l2487_248762

/-- The complex number under consideration -/
def z : ℂ := Complex.I * (-2 + 3 * Complex.I)

/-- A complex number is in the third quadrant if its real and imaginary parts are both negative -/
def is_in_third_quadrant (w : ℂ) : Prop :=
  w.re < 0 ∧ w.im < 0

/-- Theorem stating that z is in the third quadrant -/
theorem z_in_third_quadrant : is_in_third_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_l2487_248762


namespace NUMINAMATH_CALUDE_fraction_sum_l2487_248773

theorem fraction_sum : (3 : ℚ) / 8 + (9 : ℚ) / 12 = (9 : ℚ) / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l2487_248773


namespace NUMINAMATH_CALUDE_circle_polar_rectangular_equivalence_l2487_248751

/-- The polar coordinate equation of a circle is equivalent to its rectangular coordinate equation -/
theorem circle_polar_rectangular_equivalence (x y ρ θ : ℝ) :
  (x^2 + y^2 - 2*x = 0) ↔ (ρ = 2*Real.cos θ ∧ x = ρ*Real.cos θ ∧ y = ρ*Real.sin θ) :=
sorry

end NUMINAMATH_CALUDE_circle_polar_rectangular_equivalence_l2487_248751


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2487_248782

theorem solve_linear_equation :
  ∃ x : ℤ, 9773 + x = 13200 ∧ x = 3427 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2487_248782


namespace NUMINAMATH_CALUDE_sum_of_roots_equation_l2487_248795

theorem sum_of_roots_equation (x : ℝ) : 
  (∃ a b : ℝ, x^2 - 5*x + 7 = 9 ∧ x = a ∨ x = b) → a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equation_l2487_248795


namespace NUMINAMATH_CALUDE_smallest_valid_number_divisible_by_51_l2487_248732

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  (n / 1000 = (n / 100) % 10) ∧
  ((n / 10) % 10 = n % 10)

theorem smallest_valid_number_divisible_by_51 :
  ∃ (A : ℕ), is_valid_number A ∧ A % 51 = 0 ∧
  ∀ (B : ℕ), is_valid_number B ∧ B % 51 = 0 → A ≤ B ∧ A = 1122 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_divisible_by_51_l2487_248732


namespace NUMINAMATH_CALUDE_nested_triangle_perimeter_sum_l2487_248777

/-- Given a circle of radius r, we define a sequence of nested equilateral triangles 
    where each subsequent triangle is formed by joining the midpoints of the sides 
    of the previous triangle, starting with an equilateral triangle inscribed in the circle. 
    This theorem states that the limit of the sum of the perimeters of all these triangles 
    is 6r√3. -/
theorem nested_triangle_perimeter_sum (r : ℝ) (h : r > 0) : 
  let first_perimeter := 3 * r * Real.sqrt 3
  let perimeter_sequence := fun n => first_perimeter * (1 / 2) ^ n
  (∑' n, perimeter_sequence n) = 6 * r * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_nested_triangle_perimeter_sum_l2487_248777


namespace NUMINAMATH_CALUDE_stamp_problem_solution_l2487_248718

def stamp_problem (aj kj cj : ℕ) (m : ℚ) : Prop :=
  aj = 370 ∧
  kj = aj / 2 ∧
  aj + kj + cj = 930 ∧
  cj = m * kj + 5

theorem stamp_problem_solution :
  ∃ (aj kj cj : ℕ) (m : ℚ), stamp_problem aj kj cj m ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_stamp_problem_solution_l2487_248718


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2487_248786

theorem quadratic_roots_property (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ - 4 = 0 → x₂^2 - 3*x₂ - 4 = 0 → x₁^2 - 4*x₁ - x₂ + 2*x₁*x₂ = -7 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2487_248786
