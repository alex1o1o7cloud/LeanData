import Mathlib

namespace NUMINAMATH_CALUDE_total_molecular_weight_l2687_268792

/-- Atomic weight of Aluminium in g/mol -/
def Al : ℝ := 26.98

/-- Atomic weight of Oxygen in g/mol -/
def O : ℝ := 16.00

/-- Atomic weight of Hydrogen in g/mol -/
def H : ℝ := 1.01

/-- Atomic weight of Sodium in g/mol -/
def Na : ℝ := 22.99

/-- Atomic weight of Chlorine in g/mol -/
def Cl : ℝ := 35.45

/-- Atomic weight of Calcium in g/mol -/
def Ca : ℝ := 40.08

/-- Atomic weight of Carbon in g/mol -/
def C : ℝ := 12.01

/-- Molecular weight of Aluminium hydroxide in g/mol -/
def Al_OH_3 : ℝ := Al + 3 * O + 3 * H

/-- Molecular weight of Sodium chloride in g/mol -/
def NaCl : ℝ := Na + Cl

/-- Molecular weight of Calcium carbonate in g/mol -/
def CaCO_3 : ℝ := Ca + C + 3 * O

/-- Total molecular weight of the given compounds in grams -/
def total_weight : ℝ := 4 * Al_OH_3 + 2 * NaCl + 3 * CaCO_3

theorem total_molecular_weight : total_weight = 729.19 := by
  sorry

end NUMINAMATH_CALUDE_total_molecular_weight_l2687_268792


namespace NUMINAMATH_CALUDE_square_sum_xy_l2687_268793

theorem square_sum_xy (x y : ℝ) 
  (h1 : x * (x + y) = 30) 
  (h2 : y * (x + y) = 60) : 
  (x + y)^2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_xy_l2687_268793


namespace NUMINAMATH_CALUDE_solution_product_l2687_268758

theorem solution_product (r s : ℝ) : 
  (r - 7) * (3 * r + 11) = r^2 - 16 * r + 63 →
  (s - 7) * (3 * s + 11) = s^2 - 16 * s + 63 →
  r ≠ s →
  (r + 4) * (s + 4) = -66 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l2687_268758


namespace NUMINAMATH_CALUDE_sequence_contains_even_number_l2687_268763

def is_valid_sequence (x : ℕ → ℕ) : Prop :=
  ∀ n, ∃ d, d > 0 ∧ d < 10 ∧ x (n + 1) = x n + d ∧ ∃ k, x n / 10^k % 10 = d

theorem sequence_contains_even_number (x : ℕ → ℕ) (h : is_valid_sequence x) :
  ∃ n, Even (x n) := by
  sorry

end NUMINAMATH_CALUDE_sequence_contains_even_number_l2687_268763


namespace NUMINAMATH_CALUDE_tina_total_time_l2687_268714

/-- Calculates the total time for Tina to clean keys, let them dry, take breaks, and complete her assignment -/
def total_time (total_keys : ℕ) (keys_to_clean : ℕ) (clean_time_per_key : ℕ) (dry_time_per_key : ℕ) (break_interval : ℕ) (break_duration : ℕ) (assignment_time : ℕ) : ℕ :=
  let cleaning_time := keys_to_clean * clean_time_per_key
  let drying_time := total_keys * dry_time_per_key
  let break_count := total_keys / break_interval
  let break_time := break_count * break_duration
  cleaning_time + drying_time + break_time + assignment_time

/-- Proves that given the conditions in the problem, the total time is 541 minutes -/
theorem tina_total_time :
  total_time 30 29 7 10 5 3 20 = 541 := by
  sorry

end NUMINAMATH_CALUDE_tina_total_time_l2687_268714


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l2687_268725

/-- Given a square with diagonal length 14√2 cm, its area is 196 cm² -/
theorem square_area_from_diagonal : ∀ s : ℝ,
  s > 0 →
  s * s * 2 = (14 * Real.sqrt 2) ^ 2 →
  s * s = 196 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l2687_268725


namespace NUMINAMATH_CALUDE_rtl_grouping_equivalence_l2687_268702

/-- Right-to-left grouping evaluation function -/
noncomputable def rtlEval (a b c d e : ℝ) : ℝ := a / (b * c - (d + e))

/-- Standard algebraic notation representation -/
noncomputable def standardNotation (a b c d e : ℝ) : ℝ := a / (b * c - d - e)

/-- Theorem stating the equivalence of right-to-left grouping and standard notation -/
theorem rtl_grouping_equivalence (a b c d e : ℝ) :
  rtlEval a b c d e = standardNotation a b c d e :=
sorry

end NUMINAMATH_CALUDE_rtl_grouping_equivalence_l2687_268702


namespace NUMINAMATH_CALUDE_range_of_f_l2687_268724

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x

-- Define the domain
def domain : Set ℝ := {x | 1 ≤ x ∧ x < 5}

-- Theorem statement
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | -4 ≤ y ∧ y < 5} :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l2687_268724


namespace NUMINAMATH_CALUDE_min_value_theorem_l2687_268757

theorem min_value_theorem (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) 
  (hm : m > 0) (hn : n > 0) (h_line : 3*m + n = 1) : 
  (∃ (x : ℝ), ∀ (m n : ℝ), m > 0 → n > 0 → 3*m + n = 1 → 3/m + 1/n ≥ x) ∧ 
  (∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ 3*m + n = 1 ∧ 3/m + 1/n = 16) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2687_268757


namespace NUMINAMATH_CALUDE_function_inequality_l2687_268766

theorem function_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_cond : ∀ x, x * deriv f x ≥ 0) : 
  f (-1) + f 1 ≥ 2 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2687_268766


namespace NUMINAMATH_CALUDE_polynomial_equality_implies_specific_a_l2687_268709

theorem polynomial_equality_implies_specific_a (a b c : ℤ) :
  (∀ x : ℤ, (x - a) * (x - 15) + 4 = (x + b) * (x + c)) →
  (a = 10 ∨ a = 25) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_implies_specific_a_l2687_268709


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2687_268765

/-- Given a quadratic equation ax^2 + 10x + c = 0 with exactly one solution,
    where a + c = 12 and a < c, prove that a = 6 - √11 and c = 6 + √11 -/
theorem unique_solution_quadratic (a c : ℝ) : 
  (∃! x, a * x^2 + 10 * x + c = 0) → 
  a + c = 12 → 
  a < c → 
  (a = 6 - Real.sqrt 11 ∧ c = 6 + Real.sqrt 11) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2687_268765


namespace NUMINAMATH_CALUDE_determinant_minimization_l2687_268717

theorem determinant_minimization (a b : ℤ) : 
  let Δ := 36 * a - 81 * b
  ∃ (c : ℕ+), 
    (∀ k : ℕ+, Δ = k → c ≤ k) ∧ 
    Δ = c ∧
    c = 9 ∧
    (∀ a' b' : ℕ+, 36 * a' - 81 * b' = c → a + b ≤ a' + b') ∧
    a = 7 ∧ 
    b = 3 := by
  sorry

end NUMINAMATH_CALUDE_determinant_minimization_l2687_268717


namespace NUMINAMATH_CALUDE_coin_toss_total_l2687_268753

theorem coin_toss_total (head_count tail_count : ℕ) :
  let total_tosses := head_count + tail_count
  total_tosses = head_count + tail_count := by
  sorry

#check coin_toss_total 3 7

end NUMINAMATH_CALUDE_coin_toss_total_l2687_268753


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l2687_268784

/-- 
Given a triangle with sides in the ratio 5 : 6 : 7 and a perimeter of 720 cm,
prove that the longest side has a length of 280 cm.
-/
theorem longest_side_of_triangle (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (ratio : a / 5 = b / 6 ∧ b / 6 = c / 7)
  (perimeter : a + b + c = 720) :
  c = 280 := by
  sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l2687_268784


namespace NUMINAMATH_CALUDE_probability_wait_two_minutes_expected_wait_time_l2687_268779

def total_suitcases : ℕ := 200
def business_suitcases : ℕ := 10
def suitcase_interval : ℕ := 2  -- seconds

-- Part a
theorem probability_wait_two_minutes :
  (Nat.choose 59 9 : ℚ) / (Nat.choose total_suitcases business_suitcases) =
  ↑(Nat.choose 59 9) / ↑(Nat.choose total_suitcases business_suitcases) := by sorry

-- Part b
theorem expected_wait_time :
  (4020 : ℚ) / 11 = 2 * (business_suitcases * (total_suitcases + 1) / (business_suitcases + 1)) := by sorry

end NUMINAMATH_CALUDE_probability_wait_two_minutes_expected_wait_time_l2687_268779


namespace NUMINAMATH_CALUDE_quadratic_roots_range_quadratic_roots_value_l2687_268742

/-- The quadratic equation x^2 + 3x + k - 2 = 0 -/
def quadratic (x k : ℝ) : Prop := x^2 + 3*x + k - 2 = 0

/-- The equation has real roots -/
def has_real_roots (k : ℝ) : Prop := ∃ x : ℝ, quadratic x k

/-- The roots of the equation satisfy (x_1 + 1)(x_2 + 1) = -1 -/
def roots_condition (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, quadratic x₁ k ∧ quadratic x₂ k ∧ (x₁ + 1) * (x₂ + 1) = -1

theorem quadratic_roots_range (k : ℝ) :
  has_real_roots k → k ≤ 17/4 :=
sorry

theorem quadratic_roots_value (k : ℝ) :
  has_real_roots k → roots_condition k → k = 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_quadratic_roots_value_l2687_268742


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_103_l2687_268749

theorem last_three_digits_of_7_to_103 :
  7^103 ≡ 343 [ZMOD 1000] := by
sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_103_l2687_268749


namespace NUMINAMATH_CALUDE_journey_distance_proof_l2687_268745

def total_journey_time : Real := 2.5
def first_segment_time : Real := 0.5
def first_segment_speed : Real := 20
def break_time : Real := 0.25
def second_segment_time : Real := 1
def second_segment_speed : Real := 30
def third_segment_speed : Real := 15

theorem journey_distance_proof :
  let first_segment_distance := first_segment_time * first_segment_speed
  let second_segment_distance := second_segment_time * second_segment_speed
  let third_segment_time := total_journey_time - (first_segment_time + break_time + second_segment_time)
  let third_segment_distance := third_segment_time * third_segment_speed
  let total_distance := first_segment_distance + second_segment_distance + third_segment_distance
  total_distance = 51.25 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_proof_l2687_268745


namespace NUMINAMATH_CALUDE_intersection_A_B_l2687_268723

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | ∃ m : ℝ, x = m^2 + 1}

theorem intersection_A_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2687_268723


namespace NUMINAMATH_CALUDE_remainder_theorem_l2687_268731

-- Define the polynomial q(x)
def q (D E F : ℝ) (x : ℝ) : ℝ := D * x^4 + E * x^2 + F * x + 7

-- State the theorem
theorem remainder_theorem (D E F : ℝ) :
  q D E F 2 = 5 → q D E F (-2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2687_268731


namespace NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l2687_268788

theorem angle_in_fourth_quadrant (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (2 * Real.pi))
  (h2 : Real.sin α < 0)
  (h3 : Real.cos α > 0) : 
  α ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l2687_268788


namespace NUMINAMATH_CALUDE_bicycle_inventory_solution_l2687_268736

/-- Represents the bicycle inventory changes in Hank's store over three days -/
def bicycle_inventory_problem (initial_stock : ℕ) (saturday_bought : ℕ) : Prop :=
  let friday_change : ℤ := 15 - 10
  let saturday_change : ℤ := saturday_bought - 12
  let sunday_change : ℤ := 11 - 9
  (friday_change + saturday_change + sunday_change : ℤ) = 3

/-- The solution to the bicycle inventory problem -/
theorem bicycle_inventory_solution :
  ∃ (initial_stock : ℕ), bicycle_inventory_problem initial_stock 8 :=
sorry

end NUMINAMATH_CALUDE_bicycle_inventory_solution_l2687_268736


namespace NUMINAMATH_CALUDE_quadratic_point_ordering_l2687_268715

/-- 
Given a quadratic function y = ax² + 6ax - 5 where a > 0, 
and points A(-4, y₁), B(-3, y₂), and C(1, y₃) on this function's graph,
prove that y₂ < y₁ < y₃.
-/
theorem quadratic_point_ordering (a y₁ y₂ y₃ : ℝ) 
  (ha : a > 0)
  (hA : y₁ = a * (-4)^2 + 6 * a * (-4) - 5)
  (hB : y₂ = a * (-3)^2 + 6 * a * (-3) - 5)
  (hC : y₃ = a * 1^2 + 6 * a * 1 - 5) :
  y₂ < y₁ ∧ y₁ < y₃ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_ordering_l2687_268715


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2687_268751

theorem inequality_equivalence (x : ℝ) : (x - 8) / (x^2 - 4*x + 13) ≥ 0 ↔ x ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2687_268751


namespace NUMINAMATH_CALUDE_favorite_fruit_strawberries_l2687_268770

theorem favorite_fruit_strawberries (total students_oranges students_pears students_apples : ℕ) 
  (h_total : total = 450)
  (h_oranges : students_oranges = 70)
  (h_pears : students_pears = 120)
  (h_apples : students_apples = 147) :
  total - (students_oranges + students_pears + students_apples) = 113 := by
  sorry

end NUMINAMATH_CALUDE_favorite_fruit_strawberries_l2687_268770


namespace NUMINAMATH_CALUDE_prime_factorial_divisibility_l2687_268700

theorem prime_factorial_divisibility (p k n : ℕ) (hp : Prime p) :
  p^k ∣ n! → (p!)^k ∣ n! := by
  sorry

end NUMINAMATH_CALUDE_prime_factorial_divisibility_l2687_268700


namespace NUMINAMATH_CALUDE_simultaneous_arrival_l2687_268738

/-- Represents a point on the shore of the circular lake -/
structure Pier where
  point : ℝ × ℝ

/-- Represents a boat with a starting position and speed -/
structure Boat where
  start : Pier
  speed : ℝ

/-- Represents the circular lake with four piers -/
structure Lake where
  k : Pier
  l : Pier
  p : Pier
  q : Pier

/-- Represents the collision point of two boats -/
def collision_point (b1 b2 : Boat) (dest1 dest2 : Pier) : ℝ × ℝ := sorry

/-- Time taken for a boat to reach its destination -/
def time_to_destination (b : Boat) (dest : Pier) : ℝ := sorry

/-- Main theorem: If boats collide when going to opposite piers,
    they will reach swapped destinations simultaneously -/
theorem simultaneous_arrival (lake : Lake) (boat : Boat) (rowboat : Boat) :
  let x := collision_point boat rowboat lake.p lake.q
  boat.start = lake.k →
  rowboat.start = lake.l →
  time_to_destination boat lake.q = time_to_destination rowboat lake.p := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_arrival_l2687_268738


namespace NUMINAMATH_CALUDE_one_approval_probability_l2687_268727

/-- The probability of a voter approving the council's measures -/
def approval_rate : ℝ := 0.6

/-- The number of voters polled -/
def num_polled : ℕ := 4

/-- The probability of exactly one voter approving out of the polled voters -/
def prob_one_approval : ℝ := 4 * (approval_rate * (1 - approval_rate)^3)

/-- Theorem stating that the probability of exactly one voter approving is 0.1536 -/
theorem one_approval_probability : prob_one_approval = 0.1536 := by
  sorry

end NUMINAMATH_CALUDE_one_approval_probability_l2687_268727


namespace NUMINAMATH_CALUDE_initial_girls_count_l2687_268737

theorem initial_girls_count (initial_total : ℕ) (initial_girls : ℕ) : 
  (initial_girls = initial_total / 2) →
  (initial_girls - 3) * 10 = 4 * (initial_total + 1) →
  (initial_girls - 4) * 20 = 7 * (initial_total + 2) →
  initial_girls = 17 := by
sorry

end NUMINAMATH_CALUDE_initial_girls_count_l2687_268737


namespace NUMINAMATH_CALUDE_max_square_side_length_is_correct_l2687_268707

/-- The width of the blackboard in centimeters. -/
def blackboardWidth : ℕ := 120

/-- The length of the blackboard in centimeters. -/
def blackboardLength : ℕ := 96

/-- The maximum side length of a square picture that can fit on the blackboard without remainder. -/
def maxSquareSideLength : ℕ := 24

/-- Theorem stating that the maximum side length of a square that can fit both the width and length of the blackboard without remainder is 24 cm. -/
theorem max_square_side_length_is_correct :
  maxSquareSideLength = Nat.gcd blackboardWidth blackboardLength ∧
  blackboardWidth % maxSquareSideLength = 0 ∧
  blackboardLength % maxSquareSideLength = 0 ∧
  ∀ n : ℕ, n > maxSquareSideLength →
    (blackboardWidth % n ≠ 0 ∨ blackboardLength % n ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_max_square_side_length_is_correct_l2687_268707


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2687_268783

/-- Given a train of length 360 meters passing a bridge of length 240 meters in 4 minutes,
    prove that the speed of the train is 2.5 m/s. -/
theorem train_speed_calculation (train_length : ℝ) (bridge_length : ℝ) (time_minutes : ℝ) :
  train_length = 360 →
  bridge_length = 240 →
  time_minutes = 4 →
  (train_length + bridge_length) / (time_minutes * 60) = 2.5 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l2687_268783


namespace NUMINAMATH_CALUDE_b_joined_after_ten_months_l2687_268790

/-- Represents the business scenario --/
structure Business where
  a_investment : ℕ
  b_investment : ℕ
  profit_ratio_a : ℕ
  profit_ratio_b : ℕ
  total_duration : ℕ

/-- Calculates the number of months after which B joined the business --/
def months_before_b_joined (b : Business) : ℕ :=
  b.total_duration - (b.a_investment * b.total_duration * b.profit_ratio_b) / 
    (b.b_investment * b.profit_ratio_a)

/-- Theorem stating that B joined after 10 months --/
theorem b_joined_after_ten_months (b : Business) 
  (h1 : b.a_investment = 3500)
  (h2 : b.b_investment = 31500)
  (h3 : b.profit_ratio_a = 2)
  (h4 : b.profit_ratio_b = 3)
  (h5 : b.total_duration = 12) :
  months_before_b_joined b = 10 := by
  sorry

end NUMINAMATH_CALUDE_b_joined_after_ten_months_l2687_268790


namespace NUMINAMATH_CALUDE_base_10_648_equals_base_7_1614_l2687_268713

/-- Converts a base-10 integer to its representation in base 7 --/
def toBase7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else (n % 7) :: toBase7 (n / 7)

/-- Converts a list of digits in base 7 to its decimal representation --/
def fromBase7 (digits : List ℕ) : ℕ :=
  digits.foldr (λ d acc => d + 7 * acc) 0

theorem base_10_648_equals_base_7_1614 :
  fromBase7 [4, 1, 6, 1] = 648 :=
by sorry

end NUMINAMATH_CALUDE_base_10_648_equals_base_7_1614_l2687_268713


namespace NUMINAMATH_CALUDE_magic_8_ball_probability_l2687_268756

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

theorem magic_8_ball_probability : 
  binomial_probability 7 3 (3/7) = 241920/823543 := by sorry

end NUMINAMATH_CALUDE_magic_8_ball_probability_l2687_268756


namespace NUMINAMATH_CALUDE_complex_power_four_l2687_268795

theorem complex_power_four (i : ℂ) : i^2 = -1 → (1 - i)^4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_four_l2687_268795


namespace NUMINAMATH_CALUDE_line_through_point_equal_intercepts_l2687_268754

/-- A line passing through (1,2) with equal intercepts has equation 2x - y = 0 or x + y - 3 = 0 -/
theorem line_through_point_equal_intercepts :
  ∀ (a b c : ℝ), 
    (∀ x y : ℝ, a * x + b * y + c = 0 → (x = 1 ∧ y = 2)) →  -- Line passes through (1,2)
    (∃ k : ℝ, k ≠ 0 ∧ a = k ∧ b = k) →                      -- Equal intercepts condition
    ((a = 2 ∧ b = -1 ∧ c = 0) ∨ (a = 1 ∧ b = 1 ∧ c = -3)) := by
  sorry


end NUMINAMATH_CALUDE_line_through_point_equal_intercepts_l2687_268754


namespace NUMINAMATH_CALUDE_permutation_remainder_cardinality_l2687_268706

theorem permutation_remainder_cardinality 
  (a : Fin 100 → Fin 100) 
  (h_perm : Function.Bijective a) :
  let b : Fin 100 → ℕ := fun i => (Finset.range i.succ).sum (fun j => (a j).val + 1)
  let r : Fin 100 → Fin 100 := fun i => (b i) % 100
  Finset.card (Finset.image r (Finset.univ : Finset (Fin 100))) ≥ 11 :=
by
  sorry

end NUMINAMATH_CALUDE_permutation_remainder_cardinality_l2687_268706


namespace NUMINAMATH_CALUDE_m_range_l2687_268718

def f (x : ℝ) : ℝ := x^5 + x^3

theorem m_range (m : ℝ) (h1 : m ∈ Set.Icc (-2 : ℝ) 2) 
  (h2 : (m - 1) ∈ Set.Icc (-2 : ℝ) 2) (h3 : f m + f (m - 1) > 0) : 
  m ∈ Set.Ioo (1/2 : ℝ) 2 := by
sorry

end NUMINAMATH_CALUDE_m_range_l2687_268718


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l2687_268734

/-- The length of the bridge in meters -/
def bridge_length : ℝ := 200

/-- The time it takes for the train to cross the bridge in seconds -/
def bridge_crossing_time : ℝ := 10

/-- The time it takes for the train to pass a lamp post on the bridge in seconds -/
def lamppost_passing_time : ℝ := 5

/-- The length of the train in meters -/
def train_length : ℝ := 200

theorem bridge_length_calculation :
  bridge_length = train_length := by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l2687_268734


namespace NUMINAMATH_CALUDE_larger_tv_diagonal_l2687_268712

theorem larger_tv_diagonal (d : ℝ) : 
  d > 0 → 
  (d / Real.sqrt 2) ^ 2 = (25 / Real.sqrt 2) ^ 2 + 79.5 → 
  d = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_larger_tv_diagonal_l2687_268712


namespace NUMINAMATH_CALUDE_rectangle_area_l2687_268785

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 36 →
  rectangle_width^2 = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2687_268785


namespace NUMINAMATH_CALUDE_sqrt_t6_plus_t4_l2687_268744

theorem sqrt_t6_plus_t4 (t : ℝ) : Real.sqrt (t^6 + t^4) = t^2 * Real.sqrt (t^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_t6_plus_t4_l2687_268744


namespace NUMINAMATH_CALUDE_hostel_cost_23_days_l2687_268704

/-- Calculate the cost of staying in a student youth hostel for a given number of days. -/
def hostelCost (days : ℕ) : ℚ :=
  let firstWeekRate : ℚ := 18
  let additionalWeekRate : ℚ := 11
  let firstWeekDays : ℕ := min days 7
  let additionalDays : ℕ := days - firstWeekDays
  firstWeekRate * firstWeekDays + additionalWeekRate * additionalDays

/-- The cost of staying for 23 days in the student youth hostel is $302.00. -/
theorem hostel_cost_23_days : hostelCost 23 = 302 := by
  sorry

#eval hostelCost 23

end NUMINAMATH_CALUDE_hostel_cost_23_days_l2687_268704


namespace NUMINAMATH_CALUDE_sheela_bank_deposit_l2687_268726

theorem sheela_bank_deposit (monthly_income : ℝ) (deposit_percentage : ℝ) (deposit_amount : ℝ) :
  monthly_income = 11875 →
  deposit_percentage = 32 →
  deposit_amount = (deposit_percentage / 100) * monthly_income →
  deposit_amount = 3796 := by
  sorry

end NUMINAMATH_CALUDE_sheela_bank_deposit_l2687_268726


namespace NUMINAMATH_CALUDE_complex_power_four_l2687_268748

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_four (h : i^2 = -1) : (1 - i)^4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_four_l2687_268748


namespace NUMINAMATH_CALUDE_spherical_coordinates_rotation_l2687_268791

/-- Given a point with rectangular coordinates (3, -4, 2) and spherical coordinates (ρ, θ, φ),
    prove that the point with spherical coordinates (ρ, θ + π, φ) has rectangular coordinates (-3, 4, 2) -/
theorem spherical_coordinates_rotation (ρ θ φ : ℝ) :
  (3 = ρ * Real.sin φ * Real.cos θ) →
  (-4 = ρ * Real.sin φ * Real.sin θ) →
  (2 = ρ * Real.cos φ) →
  (ρ * Real.sin φ * Real.cos (θ + π) = -3) ∧
  (ρ * Real.sin φ * Real.sin (θ + π) = 4) ∧
  (ρ * Real.cos φ = 2) := by
  sorry

end NUMINAMATH_CALUDE_spherical_coordinates_rotation_l2687_268791


namespace NUMINAMATH_CALUDE_y_coord_Q_l2687_268752

/-- A line passing through the origin with slope 0.8 -/
def line (x : ℝ) : ℝ := 0.8 * x

/-- The x-coordinate of point Q -/
def x_coord_Q : ℝ := 6

/-- Theorem: The y-coordinate of point Q is 4.8 -/
theorem y_coord_Q : line x_coord_Q = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_y_coord_Q_l2687_268752


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l2687_268743

theorem r_value_when_n_is_3 :
  let n : ℕ := 3
  let s : ℕ := 2^n + n
  let r : ℕ := 3^s - n^2
  r = 177138 := by sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l2687_268743


namespace NUMINAMATH_CALUDE_banana_problem_solution_l2687_268708

/-- Represents the banana purchase and sale problem --/
def banana_problem (purchase_pounds : ℚ) (purchase_price : ℚ) 
                   (sale_pounds : ℚ) (sale_price : ℚ) 
                   (profit : ℚ) (total_pounds : ℚ) : Prop :=
  -- Cost price per pound
  let cp_per_pound := purchase_price / purchase_pounds
  -- Selling price per pound
  let sp_per_pound := sale_price / sale_pounds
  -- Total cost
  let total_cost := total_pounds * cp_per_pound
  -- Total revenue
  let total_revenue := total_pounds * sp_per_pound
  -- Profit calculation
  (total_revenue - total_cost = profit) ∧
  -- Ensure the total pounds is positive
  (total_pounds > 0)

/-- Theorem stating the solution to the banana problem --/
theorem banana_problem_solution :
  banana_problem 3 0.5 4 1 6 432 := by
  sorry

end NUMINAMATH_CALUDE_banana_problem_solution_l2687_268708


namespace NUMINAMATH_CALUDE_library_tables_count_l2687_268710

/-- The number of pupils that can be seated at a rectangular table -/
def rectangular_table_capacity : ℕ := 10

/-- The number of pupils that can be seated at a square table -/
def square_table_capacity : ℕ := 4

/-- The number of square tables needed in the library -/
def square_tables_needed : ℕ := 5

/-- The total number of pupils that need to be seated -/
def total_pupils : ℕ := 90

/-- The number of rectangular tables in the library -/
def rectangular_tables : ℕ := 7

theorem library_tables_count :
  rectangular_tables * rectangular_table_capacity +
  square_tables_needed * square_table_capacity = total_pupils :=
by sorry

end NUMINAMATH_CALUDE_library_tables_count_l2687_268710


namespace NUMINAMATH_CALUDE_snowdrift_solution_l2687_268719

def snowdrift_problem (initial_depth : ℝ) : Prop :=
  let day2_depth := initial_depth / 2
  let day3_depth := day2_depth + 6
  let day4_depth := day3_depth + 18
  day4_depth = 34 ∧ initial_depth = 20

theorem snowdrift_solution :
  ∃ (initial_depth : ℝ), snowdrift_problem initial_depth :=
sorry

end NUMINAMATH_CALUDE_snowdrift_solution_l2687_268719


namespace NUMINAMATH_CALUDE_customers_who_tried_sample_l2687_268750

/-- Given a store that puts out product samples, this theorem calculates
    the number of customers who tried a sample based on the given conditions. -/
theorem customers_who_tried_sample
  (samples_per_box : ℕ)
  (boxes_opened : ℕ)
  (samples_left : ℕ)
  (h1 : samples_per_box = 20)
  (h2 : boxes_opened = 12)
  (h3 : samples_left = 5) :
  samples_per_box * boxes_opened - samples_left = 235 :=
by sorry

end NUMINAMATH_CALUDE_customers_who_tried_sample_l2687_268750


namespace NUMINAMATH_CALUDE_gcd_9009_13860_l2687_268728

theorem gcd_9009_13860 : Nat.gcd 9009 13860 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_9009_13860_l2687_268728


namespace NUMINAMATH_CALUDE_cos_sin_shift_l2687_268782

theorem cos_sin_shift (x : ℝ) : 
  Real.cos (x/2 - Real.pi/4) = Real.sin (x/2 + Real.pi/4) := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_shift_l2687_268782


namespace NUMINAMATH_CALUDE_divisors_of_expression_l2687_268747

theorem divisors_of_expression (n : ℕ+) : 
  ∃ (d : Finset ℕ+), 
    (∀ k : ℕ+, k ∈ d ↔ ∀ m : ℕ+, k ∣ (m * (m^2 - 1) * (m^2 + 3) * (m^2 + 5))) ∧
    d.card = 16 :=
sorry

end NUMINAMATH_CALUDE_divisors_of_expression_l2687_268747


namespace NUMINAMATH_CALUDE_laborer_income_l2687_268746

/-- Represents the financial situation of a laborer over a 10-month period -/
structure LaborerFinances where
  monthly_income : ℝ
  initial_expenditure : ℝ
  initial_months : ℕ
  reduced_expenditure : ℝ
  reduced_months : ℕ
  savings : ℝ

/-- The theorem stating the laborer's monthly income given the conditions -/
theorem laborer_income (lf : LaborerFinances) 
  (h1 : lf.initial_expenditure = 85)
  (h2 : lf.initial_months = 6)
  (h3 : lf.reduced_expenditure = 60)
  (h4 : lf.reduced_months = 4)
  (h5 : lf.savings = 30)
  (h6 : ∃ d : ℝ, d > 0 ∧ 
        lf.monthly_income * lf.initial_months = lf.initial_expenditure * lf.initial_months - d ∧
        lf.monthly_income * lf.reduced_months = lf.reduced_expenditure * lf.reduced_months + d + lf.savings) :
  lf.monthly_income = 78 := by
sorry

end NUMINAMATH_CALUDE_laborer_income_l2687_268746


namespace NUMINAMATH_CALUDE_vector_at_negative_three_l2687_268799

/-- A parameterized line in 2D space -/
structure ParameterizedLine where
  -- The vector on the line at parameter t
  vector : ℝ → ℝ × ℝ

/-- The given parameterized line satisfying the problem conditions -/
def given_line : ParameterizedLine :=
  { vector := sorry }

theorem vector_at_negative_three :
  given_line.vector 1 = (4, 5) →
  given_line.vector 5 = (12, -11) →
  given_line.vector (-3) = (-4, 21) := by
  sorry

end NUMINAMATH_CALUDE_vector_at_negative_three_l2687_268799


namespace NUMINAMATH_CALUDE_common_chord_equation_l2687_268739

/-- Given two circles in the xy-plane, this theorem states the equation of the line
    on which their common chord lies. -/
theorem common_chord_equation (x y : ℝ) :
  (x^2 + y^2 - 10*x - 10*y = 0) →
  (x^2 + y^2 + 6*x - 2*y - 40 = 0) →
  (2*x + y - 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_common_chord_equation_l2687_268739


namespace NUMINAMATH_CALUDE_trig_function_slope_angle_l2687_268740

/-- Given a trigonometric function f(x) = a*sin(x) - b*cos(x) with the property
    that f(π/4 - x) = f(π/4 + x) for all x, prove that the slope angle of the line
    ax - by + c = 0 is 3π/4. -/
theorem trig_function_slope_angle (a b c : ℝ) :
  (∀ x, a * Real.sin x - b * Real.cos x = a * Real.sin (Real.pi/4 - x) - b * Real.cos (Real.pi/4 - x)) →
  (∃ k : ℝ, k > 0 ∧ a = k ∧ b = k) →
  Real.arctan (a / b) = 3 * Real.pi / 4 :=
sorry

end NUMINAMATH_CALUDE_trig_function_slope_angle_l2687_268740


namespace NUMINAMATH_CALUDE_total_dog_legs_l2687_268760

/-- The standard number of legs for a dog -/
def standard_dog_legs : ℕ := 4

/-- The number of dogs in the park -/
def dogs_in_park : ℕ := 109

/-- Theorem: The total number of dog legs in the park is 436 -/
theorem total_dog_legs : dogs_in_park * standard_dog_legs = 436 := by
  sorry

end NUMINAMATH_CALUDE_total_dog_legs_l2687_268760


namespace NUMINAMATH_CALUDE_repeating_decimal_multiplication_l2687_268730

theorem repeating_decimal_multiplication (x : ℝ) : 
  x = 3.131 / 9999 → (10^5 - 10^3) * x = 309.969 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_multiplication_l2687_268730


namespace NUMINAMATH_CALUDE_nearest_integer_to_power_l2687_268778

theorem nearest_integer_to_power : ∃ n : ℤ, 
  n = 3707 ∧ 
  ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 2)^6 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 2)^6 - (m : ℝ)| :=
by sorry

end NUMINAMATH_CALUDE_nearest_integer_to_power_l2687_268778


namespace NUMINAMATH_CALUDE_summer_camp_boys_l2687_268775

theorem summer_camp_boys (total : ℕ) (teachers : ℕ) (boy_ratio girl_ratio : ℕ) :
  total = 65 →
  teachers = 5 →
  boy_ratio = 3 →
  girl_ratio = 4 →
  ∃ (boys girls : ℕ),
    boys + girls + teachers = total ∧
    boys * girl_ratio = girls * boy_ratio ∧
    boys = 26 :=
by sorry

end NUMINAMATH_CALUDE_summer_camp_boys_l2687_268775


namespace NUMINAMATH_CALUDE_total_answer_key_ways_l2687_268777

/-- Represents a sequence of true-false answers -/
def TFSequence := List Bool

/-- Represents a sequence of multiple-choice answers -/
def MCSequence := List Nat

/-- Checks if a TFSequence is valid (no more than 3 consecutive true or false answers) -/
def isValidTFSequence (seq : TFSequence) : Bool :=
  sorry

/-- Checks if a MCSequence is valid (no consecutive answers are the same) -/
def isValidMCSequence (seq : MCSequence) : Bool :=
  sorry

/-- Counts the number of valid TFSequences of length 10 -/
def countValidTFSequences : Nat :=
  sorry

/-- Counts the number of valid MCSequences of length 5 with 6 choices each -/
def countValidMCSequences : Nat :=
  sorry

/-- The main theorem stating the total number of ways to write the answer key -/
theorem total_answer_key_ways :
  (countValidTFSequences * countValidMCSequences) =
  (countValidTFSequences * 3750) :=
by
  sorry

end NUMINAMATH_CALUDE_total_answer_key_ways_l2687_268777


namespace NUMINAMATH_CALUDE_lecture_arrangements_l2687_268789

/-- Represents the number of lecturers --/
def n : ℕ := 7

/-- Represents the number of lecturers with specific ordering constraints --/
def k : ℕ := 3

/-- Calculates the number of valid arrangements for k lecturers with ordering constraints --/
def valid_arrangements (k : ℕ) : ℕ :=
  (k - 1) * k / 2

/-- Calculates the number of ways to arrange the remaining lecturers --/
def remaining_arrangements (n k : ℕ) : ℕ :=
  Nat.factorial (n - k)

/-- Theorem stating the total number of possible lecture arrangements --/
theorem lecture_arrangements :
  valid_arrangements k * remaining_arrangements n k = 240 :=
sorry

end NUMINAMATH_CALUDE_lecture_arrangements_l2687_268789


namespace NUMINAMATH_CALUDE_jake_alcohol_consumption_l2687_268773

-- Define the given constants
def total_shots : ℚ := 8
def ounces_per_shot : ℚ := 3/2
def alcohol_percentage : ℚ := 1/2

-- Define Jake's share of shots
def jakes_shots : ℚ := total_shots / 2

-- Define the function to calculate pure alcohol consumed
def pure_alcohol_consumed : ℚ :=
  jakes_shots * ounces_per_shot * alcohol_percentage

-- Theorem statement
theorem jake_alcohol_consumption :
  pure_alcohol_consumed = 3 := by sorry

end NUMINAMATH_CALUDE_jake_alcohol_consumption_l2687_268773


namespace NUMINAMATH_CALUDE_not_equal_to_seven_thirds_l2687_268711

theorem not_equal_to_seven_thirds : ∃ x, x ≠ 7/3 ∧ 
  (x = 3 + 1/9) ∧ 
  (14/6 = 7/3) ∧ 
  (2 + 1/3 = 7/3) ∧ 
  (2 + 4/12 = 7/3) := by
  sorry

end NUMINAMATH_CALUDE_not_equal_to_seven_thirds_l2687_268711


namespace NUMINAMATH_CALUDE_cubic_inequality_iff_open_interval_l2687_268781

theorem cubic_inequality_iff_open_interval :
  ∀ x : ℝ, x * (x^2 - 9) < 0 ↔ x ∈ Set.Ioo (-4 : ℝ) 3 := by sorry

end NUMINAMATH_CALUDE_cubic_inequality_iff_open_interval_l2687_268781


namespace NUMINAMATH_CALUDE_tangent_line_sin_at_pi_l2687_268786

theorem tangent_line_sin_at_pi (x y : ℝ) :
  let f : ℝ → ℝ := λ t => Real.sin t
  let f' : ℝ → ℝ := λ t => Real.cos t
  let tangent_point : ℝ × ℝ := (π, 0)
  let slope : ℝ := f' tangent_point.1
  x + y - π = 0 ↔ y - tangent_point.2 = slope * (x - tangent_point.1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_sin_at_pi_l2687_268786


namespace NUMINAMATH_CALUDE_expression_evaluation_l2687_268721

theorem expression_evaluation (m : ℝ) (h : m = 2) : 
  (m^2 - 9) / (m^2 - 6*m + 9) / (1 - 2/(m - 3)) = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2687_268721


namespace NUMINAMATH_CALUDE_sum_of_digits_888_base8_l2687_268764

/-- Converts a natural number from base 10 to base 8 -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the sum of digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

theorem sum_of_digits_888_base8 : sumDigits (toBase8 888) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_888_base8_l2687_268764


namespace NUMINAMATH_CALUDE_sum_simplification_l2687_268768

theorem sum_simplification : -1^2004 + (-1)^2005 + 1^2006 - 1^2007 = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_simplification_l2687_268768


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l2687_268705

theorem sqrt_sum_inequality : Real.sqrt 2 + Real.sqrt 11 < Real.sqrt 3 + Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l2687_268705


namespace NUMINAMATH_CALUDE_batsman_innings_count_l2687_268729

theorem batsman_innings_count (total_runs : ℕ → ℕ) (highest_score lowest_score : ℕ) :
  (∃ n : ℕ, n > 0 ∧
    total_runs n / n = 62 ∧
    highest_score - lowest_score = 150 ∧
    (total_runs n - highest_score - lowest_score) / (n - 2) = 58 ∧
    highest_score = 225) →
  ∃ n : ℕ, n = 104 ∧
    total_runs n / n = 62 ∧
    highest_score - lowest_score = 150 ∧
    (total_runs n - highest_score - lowest_score) / (n - 2) = 58 ∧
    highest_score = 225 :=
by sorry

end NUMINAMATH_CALUDE_batsman_innings_count_l2687_268729


namespace NUMINAMATH_CALUDE_committee_age_difference_l2687_268796

theorem committee_age_difference (n : ℕ) (A : ℝ) (O N : ℝ) : 
  n = 20 → 
  n * A = n * A + O - N → 
  O - N = 160 :=
by
  sorry

end NUMINAMATH_CALUDE_committee_age_difference_l2687_268796


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2687_268776

theorem complex_equation_solution : ∃ (z : ℂ), 3 - 2 * Complex.I * z = 7 + 4 * Complex.I * z ∧ z = (2 * Complex.I) / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2687_268776


namespace NUMINAMATH_CALUDE_no_solution_iff_k_equals_four_l2687_268772

theorem no_solution_iff_k_equals_four :
  ∀ k : ℝ, (∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → (x - 3) / (x - 4) ≠ (x - k) / (x - 8)) ↔ k = 4 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_k_equals_four_l2687_268772


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2687_268780

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, x^2 - a*x - b < 0 ↔ 2 < x ∧ x < 3) → 
  a = 5 ∧ b = -6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2687_268780


namespace NUMINAMATH_CALUDE_rearrangements_without_substring_l2687_268720

def string_length : ℕ := 8
def h_count : ℕ := 2
def m_count : ℕ := 4
def t_count : ℕ := 2

def total_arrangements : ℕ := string_length.factorial / (h_count.factorial * m_count.factorial * t_count.factorial)

def substring_length : ℕ := 4
def remaining_string_length : ℕ := string_length - substring_length + 1

def arrangements_with_substring : ℕ := 
  (remaining_string_length.factorial / (h_count.pred.factorial * m_count.pred.pred.pred.factorial)) * 
  (substring_length.factorial / m_count.pred.factorial)

theorem rearrangements_without_substring : 
  total_arrangements - arrangements_with_substring + 1 = 361 := by sorry

end NUMINAMATH_CALUDE_rearrangements_without_substring_l2687_268720


namespace NUMINAMATH_CALUDE_first_discount_percentage_l2687_268761

/-- Proves that the first discount percentage is 20% given the conditions of the problem -/
theorem first_discount_percentage (original_price : ℝ) (second_discount : ℝ) (final_price : ℝ)
  (h1 : original_price = 400)
  (h2 : second_discount = 15)
  (h3 : final_price = 272)
  (h4 : final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100)) :
  first_discount = 20 := by
  sorry


end NUMINAMATH_CALUDE_first_discount_percentage_l2687_268761


namespace NUMINAMATH_CALUDE_inverse_composition_value_l2687_268735

open Function

-- Define the functions h and k
variable (h k : ℝ → ℝ)

-- Define the condition given in the problem
axiom h_k_relation : ∀ x, h⁻¹ (k x) = 3 * x - 4

-- State the theorem to be proved
theorem inverse_composition_value : k⁻¹ (h 5) = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_composition_value_l2687_268735


namespace NUMINAMATH_CALUDE_dog_drying_time_l2687_268733

/-- Time to dry a short-haired dog in minutes -/
def short_hair_time : ℕ := 10

/-- Time to dry a full-haired dog in minutes -/
def full_hair_time : ℕ := 2 * short_hair_time

/-- Number of short-haired dogs -/
def num_short_hair : ℕ := 6

/-- Number of full-haired dogs -/
def num_full_hair : ℕ := 9

/-- Total time to dry all dogs in hours -/
def total_time_hours : ℚ := (num_short_hair * short_hair_time + num_full_hair * full_hair_time) / 60

theorem dog_drying_time : total_time_hours = 4 := by
  sorry

end NUMINAMATH_CALUDE_dog_drying_time_l2687_268733


namespace NUMINAMATH_CALUDE_weight_gain_ratio_l2687_268755

/-- The weight gain problem at the family reunion -/
theorem weight_gain_ratio (jose_gain orlando_gain fernando_gain : ℚ) : 
  orlando_gain = 5 →
  fernando_gain = jose_gain / 2 - 3 →
  jose_gain + orlando_gain + fernando_gain = 20 →
  jose_gain / orlando_gain = 12 / 5 := by
sorry

end NUMINAMATH_CALUDE_weight_gain_ratio_l2687_268755


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_240_l2687_268794

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_is_240 :
  rectangle_area 3600 10 = 240 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_240_l2687_268794


namespace NUMINAMATH_CALUDE_fraction_integer_iff_specific_p_l2687_268771

theorem fraction_integer_iff_specific_p (p : ℕ+) :
  (∃ (k : ℕ+), (4 * p + 40 : ℚ) / (3 * p - 7 : ℚ) = k) ↔ p ∈ ({5, 8, 18, 50} : Set ℕ+) :=
sorry

end NUMINAMATH_CALUDE_fraction_integer_iff_specific_p_l2687_268771


namespace NUMINAMATH_CALUDE_walker_speed_l2687_268716

-- Define the speed of person B
def speed_B : ℝ := 3

-- Define the number of crossings
def num_crossings : ℕ := 5

-- Define the time period in hours
def time_period : ℝ := 1

-- Theorem statement
theorem walker_speed (speed_A : ℝ) : 
  (num_crossings : ℝ) / (speed_A + speed_B) = time_period → 
  speed_A = 2 := by
sorry

end NUMINAMATH_CALUDE_walker_speed_l2687_268716


namespace NUMINAMATH_CALUDE_sixth_term_seq1_sixth_term_seq2_l2687_268701

-- Define the first sequence
def seq1 (n : ℕ) : ℕ := 3 * n

-- Define the second sequence
def seq2 (n : ℕ) : ℕ := n * n

-- Theorem for the first sequence
theorem sixth_term_seq1 : seq1 5 = 15 := by sorry

-- Theorem for the second sequence
theorem sixth_term_seq2 : seq2 6 = 36 := by sorry

end NUMINAMATH_CALUDE_sixth_term_seq1_sixth_term_seq2_l2687_268701


namespace NUMINAMATH_CALUDE_triangle_cut_theorem_l2687_268787

theorem triangle_cut_theorem : 
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → 
    ((9 - y : ℤ) + (12 - y) ≤ (20 - y) ∧
     (9 - y : ℤ) + (20 - y) ≤ (12 - y) ∧
     (12 - y : ℤ) + (20 - y) ≤ (9 - y)) → 
    y ≥ x) ∧
  (9 - x : ℤ) + (12 - x) ≤ (20 - x) ∧
  (9 - x : ℤ) + (20 - x) ≤ (12 - x) ∧
  (12 - x : ℤ) + (20 - x) ≤ (9 - x) ∧
  x = 17 :=
by sorry

end NUMINAMATH_CALUDE_triangle_cut_theorem_l2687_268787


namespace NUMINAMATH_CALUDE_percentage_of_men_employees_l2687_268762

theorem percentage_of_men_employees (men_attendance : ℝ) (women_attendance : ℝ) (total_attendance : ℝ) :
  men_attendance = 0.2 →
  women_attendance = 0.4 →
  total_attendance = 0.34 →
  ∃ (men_percentage : ℝ),
    men_percentage + (1 - men_percentage) = 1 ∧
    men_attendance * men_percentage + women_attendance * (1 - men_percentage) = total_attendance ∧
    men_percentage = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_men_employees_l2687_268762


namespace NUMINAMATH_CALUDE_number_multiplied_by_three_twice_l2687_268741

theorem number_multiplied_by_three_twice (x : ℝ) : (3 * (3 * x) = 18) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplied_by_three_twice_l2687_268741


namespace NUMINAMATH_CALUDE_problem_statement_l2687_268797

theorem problem_statement (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2687_268797


namespace NUMINAMATH_CALUDE_mathilda_debt_repayment_l2687_268769

/-- Mathilda's debt repayment problem -/
theorem mathilda_debt_repayment 
  (original_debt : ℝ) 
  (remaining_percentage : ℝ) 
  (initial_installment : ℝ) :
  original_debt = 500 ∧ 
  remaining_percentage = 75 ∧ 
  initial_installment = original_debt * (100 - remaining_percentage) / 100 →
  initial_installment = 125 := by
sorry

end NUMINAMATH_CALUDE_mathilda_debt_repayment_l2687_268769


namespace NUMINAMATH_CALUDE_fraction_irreducible_l2687_268703

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l2687_268703


namespace NUMINAMATH_CALUDE_max_dot_product_ellipse_l2687_268732

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

def center : ℝ × ℝ := (0, 0)

noncomputable def left_focus : ℝ × ℝ := (-1, 0)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem max_dot_product_ellipse :
  ∃ (max : ℝ), max = 6 ∧
  ∀ (P : ℝ × ℝ), ellipse P.1 P.2 →
  dot_product (P.1 - center.1, P.2 - center.2) (P.1 - left_focus.1, P.2 - left_focus.2) ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_dot_product_ellipse_l2687_268732


namespace NUMINAMATH_CALUDE_amusement_park_total_cost_l2687_268798

/-- Represents the total cost for a group of children at an amusement park -/
def amusement_park_cost (num_children : ℕ) 
  (ferris_wheel_cost ferris_wheel_participants : ℕ)
  (roller_coaster_cost roller_coaster_participants : ℕ)
  (merry_go_round_cost : ℕ)
  (bumper_cars_cost bumper_cars_participants : ℕ)
  (haunted_house_cost haunted_house_participants : ℕ)
  (log_flume_cost log_flume_participants : ℕ)
  (ice_cream_cost ice_cream_participants : ℕ)
  (hot_dog_cost hot_dog_participants : ℕ)
  (pizza_cost pizza_participants : ℕ)
  (pretzel_cost pretzel_participants : ℕ)
  (cotton_candy_cost cotton_candy_participants : ℕ)
  (soda_cost soda_participants : ℕ) : ℕ :=
  ferris_wheel_cost * ferris_wheel_participants +
  roller_coaster_cost * roller_coaster_participants +
  merry_go_round_cost * num_children +
  bumper_cars_cost * bumper_cars_participants +
  haunted_house_cost * haunted_house_participants +
  log_flume_cost * log_flume_participants +
  ice_cream_cost * ice_cream_participants +
  hot_dog_cost * hot_dog_participants +
  pizza_cost * pizza_participants +
  pretzel_cost * pretzel_participants +
  cotton_candy_cost * cotton_candy_participants +
  soda_cost * soda_participants

/-- The total cost for the group of children at the amusement park is $286 -/
theorem amusement_park_total_cost : 
  amusement_park_cost 10 5 6 7 4 3 4 7 6 5 8 3 8 4 6 5 4 3 5 2 3 6 2 7 = 286 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_total_cost_l2687_268798


namespace NUMINAMATH_CALUDE_coffee_blend_weight_l2687_268722

/-- Proves that the total weight of a coffee blend is 20 pounds -/
theorem coffee_blend_weight (price_blend1 price_blend2 price_final : ℚ) 
  (weight_blend1 : ℚ) : 
  price_blend1 = 9 ∧ 
  price_blend2 = 8 ∧ 
  price_final = 21/5 ∧ 
  weight_blend1 = 8 →
  ∃ weight_blend2 : ℚ, 
    (weight_blend1 * price_blend1 + weight_blend2 * price_blend2) / (weight_blend1 + weight_blend2) = price_final ∧
    weight_blend1 + weight_blend2 = 20 :=
by sorry

end NUMINAMATH_CALUDE_coffee_blend_weight_l2687_268722


namespace NUMINAMATH_CALUDE_existence_of_four_integers_l2687_268759

theorem existence_of_four_integers : ∃ (a b c d : ℤ),
  (abs a > 1000000) ∧ 
  (abs b > 1000000) ∧ 
  (abs c > 1000000) ∧ 
  (abs d > 1000000) ∧ 
  (1 / a + 1 / b + 1 / c + 1 / d : ℚ) = 1 / (a * b * c * d) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_four_integers_l2687_268759


namespace NUMINAMATH_CALUDE_symmetric_rook_placements_8x8_l2687_268767

/-- Represents a chessboard configuration with rooks placed symmetrically --/
structure SymmetricRookPlacement where
  board_size : Nat
  num_rooks : Nat
  is_symmetric : Bool

/-- Counts the number of symmetric rook placements on a chessboard --/
def count_symmetric_rook_placements (config : SymmetricRookPlacement) : Nat :=
  sorry

/-- Theorem stating the number of symmetric rook placements for 8 rooks on an 8x8 chessboard --/
theorem symmetric_rook_placements_8x8 :
  count_symmetric_rook_placements ⟨8, 8, true⟩ = 139448 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_rook_placements_8x8_l2687_268767


namespace NUMINAMATH_CALUDE_specific_polygon_perimeter_l2687_268774

/-- The perimeter of a polygon consisting of a rectangle and a right triangle -/
def polygon_perimeter (rect_side1 rect_side2 triangle_hypotenuse : ℝ) : ℝ :=
  2 * (rect_side1 + rect_side2) - rect_side2 + triangle_hypotenuse

/-- Theorem: The perimeter of the specific polygon is 21 units -/
theorem specific_polygon_perimeter :
  polygon_perimeter 6 4 5 = 21 := by
  sorry

#eval polygon_perimeter 6 4 5

end NUMINAMATH_CALUDE_specific_polygon_perimeter_l2687_268774
