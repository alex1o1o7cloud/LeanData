import Mathlib

namespace NUMINAMATH_CALUDE_circle_condition_l1740_174067

/-- A circle in the xy-plane can be represented by an equation of the form
    (x - h)^2 + (y - k)^2 = r^2, where (h, k) is the center and r is the radius. --/
def is_circle (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (h k r : ℝ), r > 0 ∧ ∀ x y, f x y = 0 ↔ (x - h)^2 + (y - k)^2 = r^2

/-- The main theorem: if x^2 + y^2 - x + y + m = 0 represents a circle, then m < 1/2 --/
theorem circle_condition (m : ℝ) 
  (h : is_circle (fun x y => x^2 + y^2 - x + y + m)) : 
  m < (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_circle_condition_l1740_174067


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l1740_174092

/-- Given a circle C' with equation x^2 - 14x + y^2 + 16y + 100 = 0,
    prove that the sum of its center coordinates and radius is -1 + √13 -/
theorem circle_center_radius_sum :
  ∃ (a' b' r' : ℝ),
    (∀ (x y : ℝ), x^2 - 14*x + y^2 + 16*y + 100 = 0 ↔ (x - a')^2 + (y - b')^2 = r'^2) →
    a' + b' + r' = -1 + Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l1740_174092


namespace NUMINAMATH_CALUDE_probability_red_and_purple_or_yellow_l1740_174093

def total_balls : ℕ := 120
def white_balls : ℕ := 25
def green_balls : ℕ := 22
def yellow_balls : ℕ := 15
def red_balls : ℕ := 12
def purple_balls : ℕ := 18
def blue_balls : ℕ := 10
def orange_balls : ℕ := 18

theorem probability_red_and_purple_or_yellow :
  let purple_or_yellow_balls := purple_balls + yellow_balls
  let remaining_balls := total_balls - 1
  (red_balls : ℚ) / total_balls * purple_or_yellow_balls / remaining_balls = 33 / 1190 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_and_purple_or_yellow_l1740_174093


namespace NUMINAMATH_CALUDE_symmetric_difference_A_B_l1740_174038

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 1}
def B : Set ℝ := {x : ℝ | x < 0}

-- Define set difference
def set_difference (M N : Set ℝ) : Set ℝ := {x : ℝ | x ∈ M ∧ x ∉ N}

-- Define symmetric difference
def symmetric_difference (M N : Set ℝ) : Set ℝ := 
  (set_difference M N) ∪ (set_difference N M)

-- State the theorem
theorem symmetric_difference_A_B : 
  symmetric_difference A B = {x : ℝ | x < -1 ∨ (0 ≤ x ∧ x < 1)} := by sorry

end NUMINAMATH_CALUDE_symmetric_difference_A_B_l1740_174038


namespace NUMINAMATH_CALUDE_largest_integer_times_eleven_less_than_150_l1740_174008

theorem largest_integer_times_eleven_less_than_150 :
  ∀ x : ℤ, x ≤ 13 ↔ 11 * x < 150 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_times_eleven_less_than_150_l1740_174008


namespace NUMINAMATH_CALUDE_wooden_toy_price_is_20_l1740_174068

/-- The original price of each wooden toy -/
def wooden_toy_price : ℝ := 20

/-- The number of paintings bought -/
def num_paintings : ℕ := 10

/-- The original price of each painting -/
def painting_price : ℝ := 40

/-- The number of wooden toys bought -/
def num_toys : ℕ := 8

/-- The discount rate for paintings -/
def painting_discount : ℝ := 0.1

/-- The discount rate for wooden toys -/
def toy_discount : ℝ := 0.15

/-- The total loss from the sale -/
def total_loss : ℝ := 64

theorem wooden_toy_price_is_20 :
  (num_paintings * painting_price + num_toys * wooden_toy_price) -
  (num_paintings * painting_price * (1 - painting_discount) +
   num_toys * wooden_toy_price * (1 - toy_discount)) = total_loss :=
by sorry

end NUMINAMATH_CALUDE_wooden_toy_price_is_20_l1740_174068


namespace NUMINAMATH_CALUDE_one_intersection_implies_a_range_l1740_174014

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a^2*x + 1

-- State the theorem
theorem one_intersection_implies_a_range (a : ℝ) :
  (∃! x : ℝ, f a x = 3) → -1 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_one_intersection_implies_a_range_l1740_174014


namespace NUMINAMATH_CALUDE_max_consecutive_semi_primes_l1740_174099

/-- A natural number is semi-prime if it is greater than 25 and is the sum of two distinct prime numbers. -/
def IsSemiPrime (n : ℕ) : Prop :=
  n > 25 ∧ ∃ p q : ℕ, p.Prime ∧ q.Prime ∧ p ≠ q ∧ n = p + q

/-- The maximum number of consecutive semi-prime natural numbers is 5. -/
theorem max_consecutive_semi_primes :
  ∀ n : ℕ, (∀ k : ℕ, k ∈ Finset.range 6 → IsSemiPrime (n + k)) →
    ¬∀ k : ℕ, k ∈ Finset.range 7 → IsSemiPrime (n + k) :=
by sorry

end NUMINAMATH_CALUDE_max_consecutive_semi_primes_l1740_174099


namespace NUMINAMATH_CALUDE_single_layer_cake_cost_l1740_174087

/-- The cost of a single layer cake slice -/
def single_layer_cost : ℝ := 4

/-- The cost of a double layer cake slice -/
def double_layer_cost : ℝ := 7

/-- The number of single layer cake slices bought -/
def single_layer_count : ℕ := 7

/-- The number of double layer cake slices bought -/
def double_layer_count : ℕ := 5

/-- The total amount spent -/
def total_spent : ℝ := 63

theorem single_layer_cake_cost :
  single_layer_cost * single_layer_count + double_layer_cost * double_layer_count = total_spent :=
by sorry

end NUMINAMATH_CALUDE_single_layer_cake_cost_l1740_174087


namespace NUMINAMATH_CALUDE_max_xy_given_constraint_l1740_174004

theorem max_xy_given_constraint (x y : ℝ) (h : 2 * x + y = 1) : 
  ∃ (max : ℝ), max = (1/8 : ℝ) ∧ ∀ (x' y' : ℝ), 2 * x' + y' = 1 → x' * y' ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_xy_given_constraint_l1740_174004


namespace NUMINAMATH_CALUDE_composite_numbers_l1740_174010

theorem composite_numbers (n : ℕ) (h : n > 2) : 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n^4 + 2*n^2 + 1 = a * b) ∧ 
  (∃ c d : ℕ, c > 1 ∧ d > 1 ∧ n^4 + n^2 + 1 = c * d) := by
  sorry

#check composite_numbers

end NUMINAMATH_CALUDE_composite_numbers_l1740_174010


namespace NUMINAMATH_CALUDE_function_value_at_two_l1740_174078

/-- Given a function f(x) = x^5 + px^3 + qx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem function_value_at_two (p q : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^5 + p*x^3 + q*x - 8)
  (h2 : f (-2) = 10) : 
  f 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l1740_174078


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_50_l1740_174044

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The sum of factorials from 1 to n -/
def sumFactorials (n : ℕ) : ℕ := 
  (List.range n).map (λ i => factorial (i + 1)) |> List.sum

/-- The units digit of the sum of factorials from 1! to 50! is 3 -/
theorem units_digit_sum_factorials_50 : 
  unitsDigit (sumFactorials 50) = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_50_l1740_174044


namespace NUMINAMATH_CALUDE_g_of_3_equals_5_l1740_174070

-- Define the function g
def g (x : ℝ) : ℝ := 2 * (x - 2) + 3

-- State the theorem
theorem g_of_3_equals_5 : g 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_equals_5_l1740_174070


namespace NUMINAMATH_CALUDE_fraction_problem_l1740_174071

theorem fraction_problem (x : ℚ) : 
  (80 / 100 * 45 : ℚ) - (x * 25) = 16 ↔ x = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_fraction_problem_l1740_174071


namespace NUMINAMATH_CALUDE_vector_parallel_perpendicular_l1740_174009

def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)

theorem vector_parallel_perpendicular (x : ℝ) :
  (∃ k : ℝ, a + 2 • b x = k • (2 • a - b x) → x = 1/2) ∧
  ((a + 2 • b x) • (2 • a - b x) = 0 → x = -2 ∨ x = 7/2) :=
sorry

end NUMINAMATH_CALUDE_vector_parallel_perpendicular_l1740_174009


namespace NUMINAMATH_CALUDE_cube_root_eight_over_sqrt_two_equals_sqrt_two_l1740_174097

theorem cube_root_eight_over_sqrt_two_equals_sqrt_two : 
  (8 : ℝ)^(1/3) / (2 : ℝ)^(1/2) = (2 : ℝ)^(1/2) := by sorry

end NUMINAMATH_CALUDE_cube_root_eight_over_sqrt_two_equals_sqrt_two_l1740_174097


namespace NUMINAMATH_CALUDE_range_of_m_l1740_174000

/-- The range of m given specific conditions on the roots of a quadratic equation -/
theorem range_of_m (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) ∧ 
  ¬(1 < m ∧ m < 3) ∧
  ¬¬(∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) →
  m ≥ 3 ∨ m < -2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1740_174000


namespace NUMINAMATH_CALUDE_find_v5_l1740_174053

def sequence_relation (v : ℕ → ℝ) : Prop :=
  ∀ n, v (n + 2) = 2 * v (n + 1) + v n

theorem find_v5 (v : ℕ → ℝ) (h1 : sequence_relation v) (h2 : v 4 = 15) (h3 : v 7 = 255) :
  v 5 = 45 := by
  sorry

end NUMINAMATH_CALUDE_find_v5_l1740_174053


namespace NUMINAMATH_CALUDE_olympiads_spellings_l1740_174043

def word_length : Nat := 9

-- Function to calculate the number of valid spellings
def valid_spellings (n : Nat) : Nat :=
  if n = 0 then 1
  else if n = word_length then 2^(n-1)
  else 2 * valid_spellings (n-1)

theorem olympiads_spellings :
  valid_spellings word_length = 256 :=
by sorry

end NUMINAMATH_CALUDE_olympiads_spellings_l1740_174043


namespace NUMINAMATH_CALUDE_vacation_savings_time_l1740_174082

/-- 
Given:
- goal_amount: The total amount needed for the vacation
- current_savings: The amount currently saved
- monthly_savings: The amount that can be saved each month

Prove that the number of months needed to reach the goal is 3.
-/
theorem vacation_savings_time (goal_amount current_savings monthly_savings : ℕ) 
  (h1 : goal_amount = 5000)
  (h2 : current_savings = 2900)
  (h3 : monthly_savings = 700) :
  (goal_amount - current_savings + monthly_savings - 1) / monthly_savings = 3 := by
  sorry


end NUMINAMATH_CALUDE_vacation_savings_time_l1740_174082


namespace NUMINAMATH_CALUDE_smallest_multiple_of_2_3_5_l1740_174049

theorem smallest_multiple_of_2_3_5 : ∀ n : ℕ, n > 0 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n → n ≥ 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_2_3_5_l1740_174049


namespace NUMINAMATH_CALUDE_only_tiger_leopard_valid_l1740_174047

-- Define the animals
inductive Animal : Type
| Lion : Animal
| Tiger : Animal
| Leopard : Animal
| Elephant : Animal

-- Define a pair of animals
def AnimalPair := (Animal × Animal)

-- Define the conditions
def validPair (pair : AnimalPair) : Prop :=
  -- Two different animals are sent
  pair.1 ≠ pair.2 ∧
  -- If lion is sent, tiger must be sent
  (pair.1 = Animal.Lion ∨ pair.2 = Animal.Lion) → 
    (pair.1 = Animal.Tiger ∨ pair.2 = Animal.Tiger) ∧
  -- If leopard is not sent, tiger cannot be sent
  (pair.1 ≠ Animal.Leopard ∧ pair.2 ≠ Animal.Leopard) → 
    (pair.1 ≠ Animal.Tiger ∧ pair.2 ≠ Animal.Tiger) ∧
  -- If leopard is sent, elephant cannot be sent
  (pair.1 = Animal.Leopard ∨ pair.2 = Animal.Leopard) → 
    (pair.1 ≠ Animal.Elephant ∧ pair.2 ≠ Animal.Elephant)

-- Theorem: The only valid pair is Tiger and Leopard
theorem only_tiger_leopard_valid :
  ∀ (pair : AnimalPair), validPair pair ↔ 
    ((pair.1 = Animal.Tiger ∧ pair.2 = Animal.Leopard) ∨
     (pair.1 = Animal.Leopard ∧ pair.2 = Animal.Tiger)) :=
by sorry

end NUMINAMATH_CALUDE_only_tiger_leopard_valid_l1740_174047


namespace NUMINAMATH_CALUDE_batsman_average_after_17th_innings_l1740_174021

/-- Represents a batsman's performance over multiple innings -/
structure BatsmanPerformance where
  innings : Nat
  totalScore : Nat
  averageIncrease : Nat
  lastInningsScore : Nat

/-- Calculates the average score of a batsman -/
def calculateAverage (performance : BatsmanPerformance) : Rat :=
  performance.totalScore / performance.innings

theorem batsman_average_after_17th_innings
  (performance : BatsmanPerformance)
  (h1 : performance.innings = 17)
  (h2 : performance.lastInningsScore = 85)
  (h3 : calculateAverage performance - calculateAverage { performance with
    innings := performance.innings - 1
    totalScore := performance.totalScore - performance.lastInningsScore
  } = performance.averageIncrease)
  (h4 : performance.averageIncrease = 3) :
  calculateAverage performance = 37 := by
  sorry

#eval calculateAverage {
  innings := 17,
  totalScore := 17 * 37,
  averageIncrease := 3,
  lastInningsScore := 85
}

end NUMINAMATH_CALUDE_batsman_average_after_17th_innings_l1740_174021


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1740_174017

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given vectors a and b in ℝ², if a is parallel to b and a = (m, 4) and b = (3, -2), then m = -6 -/
theorem parallel_vectors_m_value (m : ℝ) :
  let a : ℝ × ℝ := (m, 4)
  let b : ℝ × ℝ := (3, -2)
  parallel a b → m = -6 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1740_174017


namespace NUMINAMATH_CALUDE_mc_question_time_l1740_174066

-- Define the total number of questions
def total_questions : ℕ := 60

-- Define the number of multiple-choice questions
def mc_questions : ℕ := 30

-- Define the number of fill-in-the-blank questions
def fib_questions : ℕ := 30

-- Define the time to learn each fill-in-the-blank question (in minutes)
def fib_time : ℕ := 25

-- Define the total study time (in minutes)
def total_study_time : ℕ := 20 * 60

-- Define the function to calculate the time for multiple-choice questions
def mc_time (x : ℕ) : ℕ := x * mc_questions

-- Define the function to calculate the time for fill-in-the-blank questions
def fib_total_time : ℕ := fib_questions * fib_time

-- Theorem: The time to learn each multiple-choice question is 15 minutes
theorem mc_question_time : 
  ∃ (x : ℕ), mc_time x + fib_total_time = total_study_time ∧ x = 15 :=
by sorry

end NUMINAMATH_CALUDE_mc_question_time_l1740_174066


namespace NUMINAMATH_CALUDE_quadratic_sum_reciprocal_l1740_174098

theorem quadratic_sum_reciprocal (x : ℝ) (h : x^2 - 4*x + 2 = 0) : x + 2/x = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_reciprocal_l1740_174098


namespace NUMINAMATH_CALUDE_inverse_A_times_B_l1740_174029

theorem inverse_A_times_B : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 1]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; -1, 5]
  A⁻¹ * B = !![1/2, 1; -1/2, 5] := by sorry

end NUMINAMATH_CALUDE_inverse_A_times_B_l1740_174029


namespace NUMINAMATH_CALUDE_ben_pea_picking_l1740_174090

/-- Ben's pea-picking problem -/
theorem ben_pea_picking (P : ℕ) : ∃ (T : ℚ), T = P / 8 :=
  by
  -- Define Ben's picking rates
  have rate1 : (56 : ℚ) / 7 = 8 := by sorry
  have rate2 : (72 : ℚ) / 9 = 8 := by sorry

  -- Prove the theorem
  sorry

end NUMINAMATH_CALUDE_ben_pea_picking_l1740_174090


namespace NUMINAMATH_CALUDE_cheese_warehouse_problem_l1740_174030

theorem cheese_warehouse_problem (total_rats : ℕ) (cheese_first_night : ℕ) (rats_second_night : ℕ) :
  total_rats > rats_second_night →
  cheese_first_night = 10 →
  rats_second_night = 7 →
  (rats_second_night : ℚ) * (cheese_first_night : ℚ) / (2 * total_rats : ℚ) = 1 →
  cheese_first_night + 1 = 11 := by
  sorry

#check cheese_warehouse_problem

end NUMINAMATH_CALUDE_cheese_warehouse_problem_l1740_174030


namespace NUMINAMATH_CALUDE_marble_probability_l1740_174042

theorem marble_probability (total : ℕ) (p_white p_green : ℚ) :
  total = 100 →
  p_white = 1/4 →
  p_green = 1/5 →
  ∃ (p_red_blue : ℚ), p_red_blue = 11/20 ∧ 
    p_white + p_green + p_red_blue = 1 :=
sorry

end NUMINAMATH_CALUDE_marble_probability_l1740_174042


namespace NUMINAMATH_CALUDE_min_value_theorem_l1740_174094

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 3) :
  (2 * a^2 + 1) / a + (b^2 - 2) / (b + 2) ≥ 13/5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1740_174094


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1740_174011

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 → 4/x + 1/y ≥ 9/2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 2 ∧ 4/x + 1/y = 9/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1740_174011


namespace NUMINAMATH_CALUDE_triangle_side_length_l1740_174065

/-- Theorem: In a triangle ABC where side b = 2, angle A = 45°, and angle C = 75°, 
    the length of side a is equal to (2/3)√6. -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  b = 2 → 
  A = 45 * π / 180 → 
  C = 75 * π / 180 → 
  a = (2 / 3) * Real.sqrt 6 := by
sorry


end NUMINAMATH_CALUDE_triangle_side_length_l1740_174065


namespace NUMINAMATH_CALUDE_tim_payment_proof_l1740_174020

/-- Represents the available bills Tim has -/
structure AvailableBills where
  tens : Nat
  fives : Nat
  ones : Nat

/-- Calculates the minimum number of bills needed to pay a given amount -/
def minBillsNeeded (bills : AvailableBills) (amount : Nat) : Nat :=
  sorry

theorem tim_payment_proof (bills : AvailableBills) (amount : Nat) :
  bills.tens = 13 ∧ bills.fives = 11 ∧ bills.ones = 17 ∧ amount = 128 →
  minBillsNeeded bills amount = 16 := by
  sorry

end NUMINAMATH_CALUDE_tim_payment_proof_l1740_174020


namespace NUMINAMATH_CALUDE_parabola_vertex_in_second_quadrant_l1740_174083

/-- Represents a parabola of the form y = 2(x-m-1)^2 + 2m + 4 -/
def Parabola (m : ℝ) := λ x : ℝ => 2 * (x - m - 1)^2 + 2 * m + 4

/-- The x-coordinate of the vertex of the parabola -/
def vertex_x (m : ℝ) : ℝ := m + 1

/-- The y-coordinate of the vertex of the parabola -/
def vertex_y (m : ℝ) : ℝ := 2 * m + 4

/-- Predicate for a point being in the second quadrant -/
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem parabola_vertex_in_second_quadrant (m : ℝ) :
  in_second_quadrant (vertex_x m) (vertex_y m) ↔ -2 < m ∧ m < -1 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_in_second_quadrant_l1740_174083


namespace NUMINAMATH_CALUDE_cookies_divisible_by_bags_l1740_174089

/-- Represents the number of snack bags Destiny can make -/
def num_bags : ℕ := 6

/-- Represents the total number of chocolate candy bars -/
def total_candy_bars : ℕ := 18

/-- Represents the number of cookies Destiny received -/
def num_cookies : ℕ := sorry

/-- Theorem stating that the number of cookies is divisible by the number of bags -/
theorem cookies_divisible_by_bags : num_bags ∣ num_cookies := by sorry

end NUMINAMATH_CALUDE_cookies_divisible_by_bags_l1740_174089


namespace NUMINAMATH_CALUDE_plate_cutting_theorem_l1740_174085

def can_measure (weights : List ℕ) (target : ℕ) : Prop :=
  ∃ (pos neg : List ℕ), pos.sum - neg.sum = target ∧ pos.toFinset ∪ neg.toFinset ⊆ weights.toFinset

theorem plate_cutting_theorem :
  let weights := [1, 3, 7]
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 11 → can_measure weights n :=
by sorry

end NUMINAMATH_CALUDE_plate_cutting_theorem_l1740_174085


namespace NUMINAMATH_CALUDE_gcf_seven_eight_factorial_l1740_174072

theorem gcf_seven_eight_factorial :
  Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcf_seven_eight_factorial_l1740_174072


namespace NUMINAMATH_CALUDE_point_on_terminal_side_l1740_174022

theorem point_on_terminal_side (x : ℝ) (α : ℝ) :
  (∃ P : ℝ × ℝ, P = (x, 2) ∧ P.2 / Real.sqrt (P.1^2 + P.2^2) = 2/3) →
  x = Real.sqrt 5 ∨ x = -Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_point_on_terminal_side_l1740_174022


namespace NUMINAMATH_CALUDE_katie_game_difference_l1740_174013

def katie_new_games : ℕ := 57
def katie_old_games : ℕ := 39
def friends_new_games : ℕ := 34

theorem katie_game_difference :
  (katie_new_games + katie_old_games) - friends_new_games = 62 := by
  sorry

end NUMINAMATH_CALUDE_katie_game_difference_l1740_174013


namespace NUMINAMATH_CALUDE_cosh_leq_exp_squared_l1740_174058

theorem cosh_leq_exp_squared (k : ℝ) :
  (∀ x : ℝ, Real.cosh x ≤ Real.exp (k * x^2)) ↔ k ≥ (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_cosh_leq_exp_squared_l1740_174058


namespace NUMINAMATH_CALUDE_boxwood_charge_theorem_l1740_174077

/-- Calculates the total charge for trimming and shaping boxwoods -/
def total_charge (num_boxwoods : ℕ) (num_shaped : ℕ) (trim_cost : ℚ) (shape_cost : ℚ) : ℚ :=
  (num_boxwoods * trim_cost) + (num_shaped * shape_cost)

/-- Proves that the total charge for trimming 30 boxwoods and shaping 4 of them is $210.00 -/
theorem boxwood_charge_theorem :
  total_charge 30 4 5 15 = 210 := by
  sorry

#eval total_charge 30 4 5 15

end NUMINAMATH_CALUDE_boxwood_charge_theorem_l1740_174077


namespace NUMINAMATH_CALUDE_jacket_price_restoration_l1740_174073

theorem jacket_price_restoration : 
  ∀ (original_price : ℝ), original_price > 0 →
  let price_after_first_reduction := original_price * (1 - 0.25)
  let price_after_second_reduction := price_after_first_reduction * (1 - 0.30)
  let restoration_factor := original_price / price_after_second_reduction
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  |restoration_factor - 1 - 0.9048| < ε :=
by sorry

end NUMINAMATH_CALUDE_jacket_price_restoration_l1740_174073


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l1740_174060

theorem solution_satisfies_system :
  ∃ (x y z : ℝ), 
    (3 * x + 2 * y - z = 1) ∧
    (4 * x - 5 * y + 3 * z = 11) ∧
    (x = 1 ∧ y = 1 ∧ z = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l1740_174060


namespace NUMINAMATH_CALUDE_distance_between_points_l1740_174069

/-- Given two points M(-2, a) and N(a, 4) on a line with slope -1/2, 
    prove that the distance between M and N is 6√3. -/
theorem distance_between_points (a : ℝ) : 
  (4 - a) / (a + 2) = -1/2 →
  Real.sqrt ((a + 2)^2 + (4 - a)^2) = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1740_174069


namespace NUMINAMATH_CALUDE_modulo_equivalence_l1740_174075

theorem modulo_equivalence : ∃ n : ℕ, 173 * 927 ≡ n [ZMOD 50] ∧ n < 50 ∧ n = 21 := by
  sorry

end NUMINAMATH_CALUDE_modulo_equivalence_l1740_174075


namespace NUMINAMATH_CALUDE_four_students_line_arrangement_l1740_174041

/-- The number of ways to arrange 4 students in a line with restrictions -/
def restricted_arrangements : ℕ := 12

/-- The total number of unrestricted arrangements of 4 students -/
def total_arrangements : ℕ := 24

/-- The number of arrangements where the fourth student is next to at least one other -/
def invalid_arrangements : ℕ := 12

theorem four_students_line_arrangement :
  restricted_arrangements = total_arrangements - invalid_arrangements :=
by sorry

end NUMINAMATH_CALUDE_four_students_line_arrangement_l1740_174041


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l1740_174034

theorem express_y_in_terms_of_x (x y : ℝ) :
  4 * x - y = 7 → y = 4 * x - 7 := by
  sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l1740_174034


namespace NUMINAMATH_CALUDE_vacation_cost_l1740_174032

theorem vacation_cost (C : ℝ) : 
  (C / 6 - C / 8 = 120) → C = 2880 := by
sorry

end NUMINAMATH_CALUDE_vacation_cost_l1740_174032


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l1740_174076

/-- Converts a number from base 7 to base 10 -/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 -/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Returns the units digit of a number in base 7 -/
def unitsDigitBase7 (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem units_digit_of_expression :
  let a := 43
  let b := 124
  let c := 15
  unitsDigitBase7 ((toBase7 (toBase10 a + toBase10 b)) * c) = 6 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l1740_174076


namespace NUMINAMATH_CALUDE_triangle_area_l1740_174095

theorem triangle_area (a b c : ℝ) : 
  (a + 4) / 3 = (b + 3) / 2 ∧ 
  (b + 3) / 2 = (c + 8) / 4 ∧ 
  a + b + c = 12 → 
  (1 / 2) * b * c = 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l1740_174095


namespace NUMINAMATH_CALUDE_meal_price_calculation_meal_price_correct_l1740_174055

/-- Calculate the entire price of a meal given individual costs, tax rate, and tip rate -/
theorem meal_price_calculation (appetizer : ℚ) (buffy_entree : ℚ) (oz_entree : ℚ) 
  (side1 : ℚ) (side2 : ℚ) (dessert : ℚ) (drink_price : ℚ) 
  (tax_rate : ℚ) (tip_rate : ℚ) : ℚ :=
  let total_before_tax := appetizer + buffy_entree + oz_entree + side1 + side2 + dessert + 2 * drink_price
  let tax := total_before_tax * tax_rate
  let total_with_tax := total_before_tax + tax
  let tip := total_with_tax * tip_rate
  let total_price := total_with_tax + tip
  total_price

/-- The entire price of the meal is $120.66 -/
theorem meal_price_correct : 
  meal_price_calculation 9 20 25 6 8 11 (13/2) (3/40) (11/50) = 12066/100 := by
  sorry

end NUMINAMATH_CALUDE_meal_price_calculation_meal_price_correct_l1740_174055


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1740_174001

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 3, 5}
def N : Set ℕ := {2, 3, 4}

theorem intersection_complement_equality :
  M ∩ (U \ N) = {1, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1740_174001


namespace NUMINAMATH_CALUDE_third_person_investment_range_l1740_174050

theorem third_person_investment_range (total : ℝ) (ratio_high_low : ℝ) :
  total = 143 ∧ ratio_high_low = 5 / 3 →
  ∃ (max min : ℝ),
    max = 55 ∧ min = 39 ∧
    ∀ (third : ℝ),
      (∃ (high low : ℝ),
        high + low + third = total ∧
        high / low = ratio_high_low ∧
        high ≥ third ∧ third ≥ low) →
      third ≤ max ∧ third ≥ min :=
by sorry

end NUMINAMATH_CALUDE_third_person_investment_range_l1740_174050


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1740_174080

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def M : Set Int := {0, 1, 2}
def N : Set Int := {0, 1, 2, 3}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1740_174080


namespace NUMINAMATH_CALUDE_initial_ratio_new_ratio_partners_count_l1740_174074

/-- Represents the number of partners in a firm -/
def partners : ℕ := 18

/-- Represents the number of associates in a firm -/
def associates : ℕ := (63 * partners) / 2

/-- The ratio of partners to associates is 2:63 -/
theorem initial_ratio : partners * 63 = associates * 2 := by sorry

/-- Adding 45 associates changes the ratio to 1:34 -/
theorem new_ratio : partners * 34 = (associates + 45) * 1 := by sorry

/-- The number of partners in the firm is 18 -/
theorem partners_count : partners = 18 := by sorry

end NUMINAMATH_CALUDE_initial_ratio_new_ratio_partners_count_l1740_174074


namespace NUMINAMATH_CALUDE_c_death_year_l1740_174036

structure Mathematician where
  name : String
  birth_year : ℕ
  death_year : ℕ

def arithmetic_sequence (a b c : ℕ) : Prop :=
  b - a = c - b

theorem c_death_year (a b c : Mathematician) (d : String) :
  a.name = "A" →
  b.name = "B" →
  c.name = "C" →
  d = "D" →
  a.death_year = 1980 →
  a.death_year - a.birth_year = 50 →
  b.death_year - b.birth_year < 50 →
  c.death_year - c.birth_year = 60 →
  a.death_year - b.death_year < 10 →
  a.death_year - b.death_year > 0 →
  b.death_year - b.birth_year = c.death_year - b.death_year →
  arithmetic_sequence a.birth_year b.birth_year c.birth_year →
  c.death_year = 1986 := by
  sorry

#check c_death_year

end NUMINAMATH_CALUDE_c_death_year_l1740_174036


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1740_174088

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a 2 →
  a 1 + a 3 = 5 →
  a 3 + a 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1740_174088


namespace NUMINAMATH_CALUDE_sams_initial_money_l1740_174084

/-- Calculates the initial amount of money given the number of books bought, 
    cost per book, and money left after purchase. -/
def initial_money (num_books : ℕ) (cost_per_book : ℕ) (money_left : ℕ) : ℕ :=
  num_books * cost_per_book + money_left

/-- Theorem stating that given the specific conditions of Sam's purchase,
    his initial amount of money was 79 dollars. -/
theorem sams_initial_money : 
  initial_money 9 7 16 = 79 := by
  sorry

#eval initial_money 9 7 16

end NUMINAMATH_CALUDE_sams_initial_money_l1740_174084


namespace NUMINAMATH_CALUDE_triangle_properties_l1740_174079

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  (a - c) / (a * Real.cos C + c * Real.cos A) = (b - c) / (a + c) →
  a + b + c ≤ 3 * Real.sqrt 3 →
  A = π / 3 ∧ a / (2 * Real.sin (π / 3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1740_174079


namespace NUMINAMATH_CALUDE_survey_problem_l1740_174028

theorem survey_problem (A B : ℕ) : 
  (A * 20 = B * 50) →                   -- 20% of A equals 50% of B (students who read both books)
  (A - A * 20 / 100) - (B - B * 50 / 100) = 150 →  -- Difference between those who read only A and only B
  (A - A * 20 / 100) + (B - B * 50 / 100) + (A * 20 / 100) = 300 :=  -- Total number of students
by sorry

end NUMINAMATH_CALUDE_survey_problem_l1740_174028


namespace NUMINAMATH_CALUDE_one_greater_one_less_than_one_l1740_174057

theorem one_greater_one_less_than_one (a b : ℝ) (h : ((1 + a * b) / (a + b))^2 < 1) :
  (a > 1 ∧ -1 < b ∧ b < 1) ∨ (-1 < a ∧ a < 1 ∧ b > 1) :=
sorry

end NUMINAMATH_CALUDE_one_greater_one_less_than_one_l1740_174057


namespace NUMINAMATH_CALUDE_rainy_days_calculation_l1740_174006

/-- Calculates the number of rainy days in a week given cycling conditions --/
def rainy_days (rain_speed : ℕ) (snow_speed : ℕ) (snow_days : ℕ) (total_distance : ℕ) : ℕ :=
  let snow_distance := snow_speed * snow_days
  (total_distance - snow_distance) / rain_speed

theorem rainy_days_calculation :
  rainy_days 90 30 4 390 = 3 := by
  sorry

end NUMINAMATH_CALUDE_rainy_days_calculation_l1740_174006


namespace NUMINAMATH_CALUDE_union_S_T_l1740_174096

-- Define the sets S and T
def S : Set ℝ := {x | x > -2}
def T : Set ℝ := {x | x^2 + 3*x - 4 ≤ 0}

-- State the theorem
theorem union_S_T : S ∪ T = {x : ℝ | x ≥ -4} := by sorry

end NUMINAMATH_CALUDE_union_S_T_l1740_174096


namespace NUMINAMATH_CALUDE_circle_intersection_distance_l1740_174086

-- Define the circle M
def CircleM (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 6*y = 0

-- Define the points on the circle
def PointO : ℝ × ℝ := (0, 0)
def PointA : ℝ × ℝ := (1, 1)
def PointB : ℝ × ℝ := (4, 2)

-- Define the intersection points
def PointS : ℝ × ℝ := (8, 0)
def PointT : ℝ × ℝ := (0, -6)

-- Theorem statement
theorem circle_intersection_distance :
  CircleM PointO.1 PointO.2 ∧
  CircleM PointA.1 PointA.2 ∧
  CircleM PointB.1 PointB.2 ∧
  CircleM PointS.1 PointS.2 ∧
  CircleM PointT.1 PointT.2 ∧
  PointS.1 > 0 ∧
  PointT.2 < 0 →
  Real.sqrt ((PointS.1 - PointT.1)^2 + (PointS.2 - PointT.2)^2) = 10 :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_distance_l1740_174086


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1740_174063

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem arithmetic_sequence_problem (a₁ d : ℝ) :
  (∃ (a : ℕ → ℝ), 
    (∀ n, a n = arithmetic_sequence a₁ d n) ∧
    ((Real.sin (a 3))^2 * (Real.cos (a 6))^2 - (Real.sin (a 6))^2 * (Real.cos (a 3))^2) / 
      Real.sin (a 4 + a 5) = 1 ∧
    d ∈ Set.Ioo (-1 : ℝ) 0 ∧
    (∀ n : ℕ, n ≠ 9 → 
      (n * a₁ + n * (n - 1) / 2 * d) ≤ (9 * a₁ + 9 * 8 / 2 * d))) →
  a₁ = 17 * Real.pi / 12 ∧ a₁ ∈ Set.Ioo (4 * Real.pi / 3) (3 * Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1740_174063


namespace NUMINAMATH_CALUDE_roots_greater_than_five_implies_k_range_l1740_174005

theorem roots_greater_than_five_implies_k_range (k : ℝ) : 
  (∀ x : ℝ, x^2 - 11*x + (30 + k) = 0 → x > 5) → 
  (0 < k ∧ k ≤ 1/4) :=
by sorry

end NUMINAMATH_CALUDE_roots_greater_than_five_implies_k_range_l1740_174005


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_union_of_A_and_B_l1740_174064

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 7}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}

-- Theorem for intersection
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 7} := by sorry

-- Theorem for union
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 < x ∧ x < 10} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_union_of_A_and_B_l1740_174064


namespace NUMINAMATH_CALUDE_largest_sum_proof_l1740_174061

theorem largest_sum_proof : 
  let sums : List ℚ := [1/3 + 1/4, 1/3 + 1/5, 1/3 + 1/6, 1/3 + 1/9, 1/3 + 1/8]
  ∀ x ∈ sums, x ≤ 1/3 + 1/4 ∧ 1/3 + 1/4 = 7/12 := by
sorry

end NUMINAMATH_CALUDE_largest_sum_proof_l1740_174061


namespace NUMINAMATH_CALUDE_square_flag_side_length_l1740_174045

theorem square_flag_side_length (total_fabric : ℝ) (square_flags : ℕ) (wide_flags : ℕ) (tall_flags : ℕ) (remaining_fabric : ℝ) :
  total_fabric = 1000 ∧ 
  square_flags = 16 ∧ 
  wide_flags = 20 ∧ 
  tall_flags = 10 ∧ 
  remaining_fabric = 294 →
  ∃ (side_length : ℝ),
    side_length = 4 ∧
    side_length^2 * square_flags + 15 * (wide_flags + tall_flags) = total_fabric - remaining_fabric :=
by sorry

end NUMINAMATH_CALUDE_square_flag_side_length_l1740_174045


namespace NUMINAMATH_CALUDE_mixture_weight_l1740_174081

/-- The weight of a mixture of green tea and coffee given specific price changes and costs. -/
theorem mixture_weight (june_cost green_tea_july coffee_july mixture_cost : ℝ) : 
  june_cost > 0 →
  green_tea_july = 0.1 * june_cost →
  coffee_july = 2 * june_cost →
  mixture_cost = 3.15 →
  (mixture_cost / ((green_tea_july + coffee_july) / 2)) = 3 := by
  sorry

#check mixture_weight

end NUMINAMATH_CALUDE_mixture_weight_l1740_174081


namespace NUMINAMATH_CALUDE_evaluate_expression_l1740_174033

theorem evaluate_expression : (-3)^4 / 3^2 - 2^5 + 7^2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1740_174033


namespace NUMINAMATH_CALUDE_max_value_z_minus_i_l1740_174091

theorem max_value_z_minus_i (z : ℂ) (h : Complex.abs z = 2) :
  ∃ (M : ℝ), M = 3 ∧ ∀ w, Complex.abs w = 2 → Complex.abs (w - Complex.I) ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_z_minus_i_l1740_174091


namespace NUMINAMATH_CALUDE_fraction_equality_l1740_174040

theorem fraction_equality (a b : ℝ) (h : b ≠ 0) : (2 * a) / (2 * b) = a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1740_174040


namespace NUMINAMATH_CALUDE_sqrt_two_times_sqrt_eight_equals_four_l1740_174052

theorem sqrt_two_times_sqrt_eight_equals_four : Real.sqrt 2 * Real.sqrt 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_times_sqrt_eight_equals_four_l1740_174052


namespace NUMINAMATH_CALUDE_find_n_l1740_174051

theorem find_n : ∃ n : ℤ, (15 : ℝ) ^ (2 * n) = (1 / 15 : ℝ) ^ (3 * n - 30) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l1740_174051


namespace NUMINAMATH_CALUDE_set_A_theorem_l1740_174027

def A (a : ℝ) := {x : ℝ | 2 * x + a > 0}

theorem set_A_theorem (a : ℝ) :
  (1 ∉ A a) → (2 ∈ A a) → -4 < a ∧ a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_set_A_theorem_l1740_174027


namespace NUMINAMATH_CALUDE_households_B_and_C_eq_22_l1740_174039

/-- A residential building where each household subscribes to exactly two different newspapers. -/
structure Building where
  /-- The number of subscriptions for newspaper A -/
  subscriptions_A : ℕ
  /-- The number of subscriptions for newspaper B -/
  subscriptions_B : ℕ
  /-- The number of subscriptions for newspaper C -/
  subscriptions_C : ℕ
  /-- The total number of households in the building -/
  total_households : ℕ
  /-- Each household subscribes to exactly two different newspapers -/
  two_subscriptions : subscriptions_A + subscriptions_B + subscriptions_C = 2 * total_households

/-- The number of households subscribing to both newspaper B and C in a given building -/
def households_B_and_C (b : Building) : ℕ :=
  b.total_households - b.subscriptions_A

theorem households_B_and_C_eq_22 (b : Building) 
  (h_A : b.subscriptions_A = 30)
  (h_B : b.subscriptions_B = 34)
  (h_C : b.subscriptions_C = 40) :
  households_B_and_C b = 22 := by
  sorry

#eval households_B_and_C ⟨30, 34, 40, 52, by norm_num⟩

end NUMINAMATH_CALUDE_households_B_and_C_eq_22_l1740_174039


namespace NUMINAMATH_CALUDE_sum_of_squares_16_to_30_l1740_174062

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_squares_16_to_30 :
  sum_of_squares 30 - sum_of_squares 15 = 8215 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_16_to_30_l1740_174062


namespace NUMINAMATH_CALUDE_expression_value_l1740_174024

theorem expression_value (x y : ℝ) (h1 : 2 * x + 3 * y = 5) (h2 : x = 4) :
  3 * x^2 + 12 * x * y + y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1740_174024


namespace NUMINAMATH_CALUDE_instructor_schedule_lcm_l1740_174015

theorem instructor_schedule_lcm : Nat.lcm 9 (Nat.lcm 5 (Nat.lcm 8 (Nat.lcm 10 12))) = 360 := by
  sorry

end NUMINAMATH_CALUDE_instructor_schedule_lcm_l1740_174015


namespace NUMINAMATH_CALUDE_zero_subset_A_l1740_174037

def A : Set ℝ := {x | x > -3}

theorem zero_subset_A : {0} ⊆ A := by sorry

end NUMINAMATH_CALUDE_zero_subset_A_l1740_174037


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1740_174035

theorem sum_of_fractions (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : ∃ n : ℤ, (a / b + b / c + c / a : ℚ) = n)
  (h2 : ∃ m : ℤ, (b / a + c / b + a / c : ℚ) = m) :
  (a / b + b / c + c / a : ℚ) = 3 ∨ (a / b + b / c + c / a : ℚ) = -3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1740_174035


namespace NUMINAMATH_CALUDE_range_of_z_plus_4_minus_3i_l1740_174059

/-- The range of |z+4-3i| when |z| = 2 -/
theorem range_of_z_plus_4_minus_3i (z : ℂ) (h : Complex.abs z = 2) :
  ∃ (w : ℂ), Complex.abs w = 2 ∧ Complex.abs (w + 4 - 3*Complex.I) = 3 ∧
  ∃ (v : ℂ), Complex.abs v = 2 ∧ Complex.abs (v + 4 - 3*Complex.I) = 7 ∧
  ∀ (u : ℂ), Complex.abs u = 2 → 3 ≤ Complex.abs (u + 4 - 3*Complex.I) ∧ Complex.abs (u + 4 - 3*Complex.I) ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_range_of_z_plus_4_minus_3i_l1740_174059


namespace NUMINAMATH_CALUDE_race_results_l1740_174002

-- Define the race parameters
def race_distance : ℝ := 200
def time_A : ℝ := 40
def time_B : ℝ := 50
def time_C : ℝ := 45

-- Define the time differences
def time_diff_AB : ℝ := time_B - time_A
def time_diff_AC : ℝ := time_C - time_A
def time_diff_BC : ℝ := time_C - time_B

-- Theorem statement
theorem race_results :
  (time_diff_AB = 10) ∧
  (time_diff_AC = 5) ∧
  (time_diff_BC = -5) := by
  sorry

end NUMINAMATH_CALUDE_race_results_l1740_174002


namespace NUMINAMATH_CALUDE_brick_wall_bottom_row_l1740_174031

/-- Represents a brick wall with a decreasing number of bricks per row from bottom to top -/
structure BrickWall where
  numRows : Nat
  totalBricks : Nat
  bottomRowBricks : Nat

/-- Calculates the total number of bricks in the wall given the number of bricks in the bottom row -/
def sumBricks (n : Nat) : Nat :=
  List.range n |> List.map (fun i => n - i) |> List.sum

/-- Theorem: A brick wall with 5 rows and 50 total bricks, where each row above the bottom
    has one less brick than the row below, has 12 bricks in the bottom row -/
theorem brick_wall_bottom_row : 
  ∀ (wall : BrickWall), 
    wall.numRows = 5 → 
    wall.totalBricks = 50 → 
    (sumBricks wall.bottomRowBricks = wall.totalBricks) → 
    wall.bottomRowBricks = 12 := by
  sorry

end NUMINAMATH_CALUDE_brick_wall_bottom_row_l1740_174031


namespace NUMINAMATH_CALUDE_batsman_average_l1740_174003

/-- Represents a batsman's performance --/
structure Batsman where
  innings : ℕ
  runsInLastInning : ℕ
  averageIncrease : ℕ

/-- Calculates the average runs per inning after the last inning --/
def finalAverage (b : Batsman) : ℕ :=
  let previousAverage := b.runsInLastInning - b.averageIncrease
  previousAverage + b.averageIncrease

/-- Theorem: The batsman's average after 17 innings is 40 runs --/
theorem batsman_average (b : Batsman) 
  (h1 : b.innings = 17)
  (h2 : b.runsInLastInning = 200)
  (h3 : b.averageIncrease = 10) : 
  finalAverage b = 40 := by
  sorry

#eval finalAverage { innings := 17, runsInLastInning := 200, averageIncrease := 10 }

end NUMINAMATH_CALUDE_batsman_average_l1740_174003


namespace NUMINAMATH_CALUDE_jose_share_of_profit_l1740_174046

/-- Calculates the share of profit for an investor given the total profit and investment ratios -/
def calculate_share_of_profit (total_profit : ℚ) (investment_ratio : ℚ) (total_investment_ratio : ℚ) : ℚ :=
  (investment_ratio / total_investment_ratio) * total_profit

theorem jose_share_of_profit (tom_investment : ℚ) (jose_investment : ℚ) 
  (tom_duration : ℚ) (jose_duration : ℚ) (total_profit : ℚ) :
  tom_investment = 3000 →
  jose_investment = 4500 →
  tom_duration = 12 →
  jose_duration = 10 →
  total_profit = 5400 →
  let tom_investment_ratio := tom_investment * tom_duration
  let jose_investment_ratio := jose_investment * jose_duration
  let total_investment_ratio := tom_investment_ratio + jose_investment_ratio
  calculate_share_of_profit total_profit jose_investment_ratio total_investment_ratio = 3000 := by
sorry

end NUMINAMATH_CALUDE_jose_share_of_profit_l1740_174046


namespace NUMINAMATH_CALUDE_linear_function_value_l1740_174007

theorem linear_function_value (f : ℝ → ℝ) (a b : ℝ) 
    (h1 : f = fun x ↦ a * x + b)
    (h2 : f 1 = 2017)
    (h3 : f 2 = 2018) : 
  f 2019 = 4035 := by
sorry

end NUMINAMATH_CALUDE_linear_function_value_l1740_174007


namespace NUMINAMATH_CALUDE_b_range_l1740_174025

/-- A cubic function with a parameter b -/
def f (b : ℝ) (x : ℝ) : ℝ := -x^3 + b*x

/-- The derivative of f with respect to x -/
def f_deriv (b : ℝ) (x : ℝ) : ℝ := -3*x^2 + b

theorem b_range (b : ℝ) :
  (∀ x ∈ Set.Ioo 0 1, Monotone (f b)) →
  (∀ x, f b x = 0 → x ∈ Set.Icc (-2) 2) →
  b ∈ Set.Icc 3 4 := by
  sorry

end NUMINAMATH_CALUDE_b_range_l1740_174025


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1740_174018

theorem necessary_but_not_sufficient (a b : ℝ) : 
  (∀ a b, a > b + 1 → a > b) ∧ 
  (∃ a b, a > b ∧ ¬(a > b + 1)) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1740_174018


namespace NUMINAMATH_CALUDE_yimin_orchard_tree_count_l1740_174048

/-- The number of trees in Yimin Orchard -/
theorem yimin_orchard_tree_count : 
  let pear_rows : ℕ := 15
  let apple_rows : ℕ := 34
  let trees_per_row : ℕ := 21
  (pear_rows + apple_rows) * trees_per_row = 1029 := by
sorry

end NUMINAMATH_CALUDE_yimin_orchard_tree_count_l1740_174048


namespace NUMINAMATH_CALUDE_find_a_l1740_174019

-- Define the solution set
def solutionSet (x : ℝ) : Prop :=
  (-3 < x ∧ x < -1) ∨ x > 2

-- Define the inequality
def inequality (a x : ℝ) : Prop :=
  (x + a) / (x^2 + 4*x + 3) > 0

theorem find_a :
  (∃ a : ℝ, ∀ x : ℝ, inequality a x ↔ solutionSet x) →
  (∃ a : ℝ, a = -2 ∧ ∀ x : ℝ, inequality a x ↔ solutionSet x) :=
by sorry

end NUMINAMATH_CALUDE_find_a_l1740_174019


namespace NUMINAMATH_CALUDE_cubic_expression_equality_l1740_174054

theorem cubic_expression_equality : 6^3 - 4 * 6^2 + 4 * 6 - 1 = 95 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_equality_l1740_174054


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l1740_174012

def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel_points :
  ∀ x y : ℝ, f x = y →
    (∃ k : ℝ, (3 * x^2 + 1) * k = 1 ∧ 4 * k = 1) ↔ 
    ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4)) := by
  sorry

#check tangent_parallel_points

end NUMINAMATH_CALUDE_tangent_parallel_points_l1740_174012


namespace NUMINAMATH_CALUDE_complex_division_result_l1740_174016

theorem complex_division_result : Complex.I * 2 / (1 + Complex.I) = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_result_l1740_174016


namespace NUMINAMATH_CALUDE_max_divisible_integers_l1740_174056

theorem max_divisible_integers (n : ℕ) : ℕ := by
  -- Let S be the set of 2n consecutive integers
  -- Let D be the set of divisors {n+1, n+2, ..., 2n}
  -- max_divisible is the maximum number of integers in S divisible by at least one number in D
  -- We want to prove that max_divisible = n + ⌊n/2⌋
  sorry

#check max_divisible_integers

end NUMINAMATH_CALUDE_max_divisible_integers_l1740_174056


namespace NUMINAMATH_CALUDE_total_visitors_is_440_l1740_174023

/-- Represents the survey results of visitors to a Picasso painting exhibition -/
structure SurveyResults where
  totalVisitors : ℕ
  didNotEnjoyOrUnderstand : ℕ
  enjoyedAndUnderstood : ℕ

/-- The conditions of the survey results -/
def surveyConditions (results : SurveyResults) : Prop :=
  results.didNotEnjoyOrUnderstand = 110 ∧
  results.enjoyedAndUnderstood = 3 * results.totalVisitors / 4 ∧
  results.totalVisitors = results.enjoyedAndUnderstood + results.didNotEnjoyOrUnderstand

/-- The theorem stating that given the survey conditions, the total number of visitors is 440 -/
theorem total_visitors_is_440 (results : SurveyResults) :
  surveyConditions results → results.totalVisitors = 440 := by
  sorry

#check total_visitors_is_440

end NUMINAMATH_CALUDE_total_visitors_is_440_l1740_174023


namespace NUMINAMATH_CALUDE_quadratic_triple_root_relation_l1740_174026

/-- For a quadratic equation px^2 + qx + r = 0, if one root is triple the other, 
    then 3q^2 = 16pr -/
theorem quadratic_triple_root_relation (p q r : ℝ) (x₁ x₂ : ℝ) : 
  (p * x₁^2 + q * x₁ + r = 0) →
  (p * x₂^2 + q * x₂ + r = 0) →
  (x₂ = 3 * x₁) →
  (3 * q^2 = 16 * p * r) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_triple_root_relation_l1740_174026
