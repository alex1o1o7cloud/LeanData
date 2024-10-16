import Mathlib

namespace NUMINAMATH_CALUDE_unique_number_l1067_106740

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def middle_digits_39 (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a * 1000 + 390 + b ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9

theorem unique_number :
  ∃! n : ℕ, is_four_digit n ∧ middle_digits_39 n ∧ n % 45 = 0 ∧ n ≤ 5000 ∧ n = 1395 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_l1067_106740


namespace NUMINAMATH_CALUDE_find_a_l1067_106703

theorem find_a : ∃ a : ℤ, 
  (∃ x : ℤ, (2 * x - a = 3) ∧ 
    (∀ y : ℤ, (1 - (y - 2) / 2 : ℚ) < ((1 + y) / 3 : ℚ) → y ≥ x) ∧
    (1 - (x - 2) / 2 : ℚ) < ((1 + x) / 3 : ℚ)) →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_find_a_l1067_106703


namespace NUMINAMATH_CALUDE_subtract_29_result_l1067_106749

theorem subtract_29_result (x : ℤ) (h : x - 46 = 15) : x - 29 = 32 := by
  sorry

end NUMINAMATH_CALUDE_subtract_29_result_l1067_106749


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1067_106729

/-- Given an arithmetic sequence 3, 7, 11, ..., x, y, 35, prove that x + y = 58 -/
theorem arithmetic_sequence_sum (x y : ℝ) : 
  (∃ (n : ℕ), n ≥ 5 ∧ 
    (∀ k : ℕ, k ≤ n → 
      (if k = 1 then 3
       else if k = 2 then 7
       else if k = 3 then 11
       else if k = n - 1 then x
       else if k = n then y
       else if k = n + 1 then 35
       else 0) = 3 + (k - 1) * 4)) →
  x + y = 58 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1067_106729


namespace NUMINAMATH_CALUDE_cos_2015_eq_neg_sin_55_l1067_106751

theorem cos_2015_eq_neg_sin_55 (m : ℝ) (h : Real.sin (55 * π / 180) = m) :
  Real.cos (2015 * π / 180) = -m := by
  sorry

end NUMINAMATH_CALUDE_cos_2015_eq_neg_sin_55_l1067_106751


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_negative_l1067_106797

theorem sum_of_reciprocals_negative (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (product_eight : a * b * c = 8) : 
  1/a + 1/b + 1/c < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_negative_l1067_106797


namespace NUMINAMATH_CALUDE_cycle_loss_percentage_l1067_106788

/-- Calculates the percentage of loss given the cost price and selling price. -/
def loss_percentage (cost_price selling_price : ℚ) : ℚ :=
  ((cost_price - selling_price) / cost_price) * 100

/-- Proves that the loss percentage for a cycle with cost price 1400 and selling price 1190 is 15%. -/
theorem cycle_loss_percentage :
  let cost_price : ℚ := 1400
  let selling_price : ℚ := 1190
  loss_percentage cost_price selling_price = 15 := by
sorry

end NUMINAMATH_CALUDE_cycle_loss_percentage_l1067_106788


namespace NUMINAMATH_CALUDE_washer_cost_l1067_106748

/-- Given a washer-dryer combination costing $1,200, where the washer costs $220 more than the dryer,
    prove that the cost of the washer is $710. -/
theorem washer_cost (total_cost dryer_cost washer_cost : ℕ) : 
  total_cost = 1200 →
  washer_cost = dryer_cost + 220 →
  total_cost = washer_cost + dryer_cost →
  washer_cost = 710 := by
sorry

end NUMINAMATH_CALUDE_washer_cost_l1067_106748


namespace NUMINAMATH_CALUDE_class_size_proof_l1067_106786

theorem class_size_proof (original_average : ℝ) (new_students : ℕ) (new_students_average : ℝ) (average_decrease : ℝ) :
  original_average = 40 →
  new_students = 8 →
  new_students_average = 32 →
  average_decrease = 4 →
  ∃ (original_size : ℕ), 
    (original_size * original_average + new_students * new_students_average) / (original_size + new_students) = original_average - average_decrease ∧
    original_size = 8 :=
by sorry

end NUMINAMATH_CALUDE_class_size_proof_l1067_106786


namespace NUMINAMATH_CALUDE_sin_plus_cos_range_l1067_106718

theorem sin_plus_cos_range (A B C : Real) (a b c : Real) :
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  b^2 = a * c →            -- Given condition
  ∃ (x : Real), 1 < x ∧ x ≤ Real.sqrt 2 ∧ x = Real.sin B + Real.cos B :=
by sorry

end NUMINAMATH_CALUDE_sin_plus_cos_range_l1067_106718


namespace NUMINAMATH_CALUDE_bernoulli_expectation_and_variance_l1067_106759

/-- A random variable with Bernoulli distribution -/
structure BernoulliRV where
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- Probability mass function for Bernoulli distribution -/
def prob (ξ : BernoulliRV) (k : ℕ) : ℝ :=
  if k = 0 then 1 - ξ.p
  else if k = 1 then ξ.p
  else 0

/-- Expected value of a Bernoulli random variable -/
def expectation (ξ : BernoulliRV) : ℝ := ξ.p

/-- Variance of a Bernoulli random variable -/
def variance (ξ : BernoulliRV) : ℝ := (1 - ξ.p) * ξ.p

/-- Theorem: The expected value and variance of a Bernoulli random variable -/
theorem bernoulli_expectation_and_variance (ξ : BernoulliRV) :
  expectation ξ = ξ.p ∧ variance ξ = (1 - ξ.p) * ξ.p := by sorry

end NUMINAMATH_CALUDE_bernoulli_expectation_and_variance_l1067_106759


namespace NUMINAMATH_CALUDE_not_always_equal_to_self_l1067_106755

-- Define the ❤ operation
def heartsuit (x y : ℝ) : ℝ := |x - y|

-- Theorem stating that the statement "x ❤ 0 = x for all x" is false
theorem not_always_equal_to_self : ¬ ∀ x : ℝ, heartsuit x 0 = x := by
  sorry

end NUMINAMATH_CALUDE_not_always_equal_to_self_l1067_106755


namespace NUMINAMATH_CALUDE_exactly_one_number_satisfies_condition_l1067_106789

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  n < 700 ∧ n = 7 * sum_of_digits n

theorem exactly_one_number_satisfies_condition : 
  ∃! n : ℕ, satisfies_condition n :=
sorry

end NUMINAMATH_CALUDE_exactly_one_number_satisfies_condition_l1067_106789


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1067_106706

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 3)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1067_106706


namespace NUMINAMATH_CALUDE_maria_coffee_order_l1067_106735

-- Define the variables
def visits_per_day : ℕ := 2
def total_cups_per_day : ℕ := 6

-- Define the function to calculate cups per visit
def cups_per_visit : ℕ := total_cups_per_day / visits_per_day

-- Theorem statement
theorem maria_coffee_order :
  cups_per_visit = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_maria_coffee_order_l1067_106735


namespace NUMINAMATH_CALUDE_minimum_b_value_l1067_106741

theorem minimum_b_value (a b : ℕ) : 
  a = 23 →
  (a + b) % 10 = 5 →
  (a + b) % 7 = 4 →
  b ≥ 2 ∧ ∃ (b' : ℕ), b' ≥ 2 → b ≤ b' :=
by sorry

end NUMINAMATH_CALUDE_minimum_b_value_l1067_106741


namespace NUMINAMATH_CALUDE_polynomial_coefficient_b_l1067_106796

theorem polynomial_coefficient_b (a b c : ℚ) : 
  (∀ x, (5*x^2 - 3*x + 7/3) * (a*x^2 + b*x + c) = 
        15*x^4 - 14*x^3 + 20*x^2 - 25/3*x + 14/3) →
  b = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_b_l1067_106796


namespace NUMINAMATH_CALUDE_infinitely_many_composites_l1067_106738

/-- A strictly increasing sequence of natural numbers where each number from
    the third one onwards is the sum of some two preceding numbers. -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧
  (∀ n ≥ 2, ∃ i j, i < n ∧ j < n ∧ a n = a i + a j)

/-- A number is composite if it's not prime and greater than 1. -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ Nat.Prime n

/-- The main theorem stating that there are infinitely many composite numbers
    in a special sequence. -/
theorem infinitely_many_composites (a : ℕ → ℕ) (h : SpecialSequence a) :
    ∀ N, ∃ n > N, IsComposite (a n) :=
  sorry

end NUMINAMATH_CALUDE_infinitely_many_composites_l1067_106738


namespace NUMINAMATH_CALUDE_mutuallyExclusiveNotContradictoryPairs_l1067_106783

-- Define the events
inductive Event : Type
| Miss : Event
| Hit : Event
| MoreThan4 : Event
| AtLeast5 : Event

-- Define mutual exclusivity
def mutuallyExclusive (e1 e2 : Event) : Prop := sorry

-- Define contradictory events
def contradictory (e1 e2 : Event) : Prop := sorry

-- Define a function to count pairs of events that are mutually exclusive but not contradictory
def countMutuallyExclusiveNotContradictory (events : List Event) : Nat := sorry

-- Theorem to prove
theorem mutuallyExclusiveNotContradictoryPairs :
  let events := [Event.Miss, Event.Hit, Event.MoreThan4, Event.AtLeast5]
  countMutuallyExclusiveNotContradictory events = 2 := by sorry

end NUMINAMATH_CALUDE_mutuallyExclusiveNotContradictoryPairs_l1067_106783


namespace NUMINAMATH_CALUDE_triangle_side_length_l1067_106742

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  c = 2 → b = 6 → B = 2 * π / 3 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1067_106742


namespace NUMINAMATH_CALUDE_isabel_weekly_distance_l1067_106791

/-- Calculates the total distance run in a week given a circuit length, 
    number of morning and afternoon runs, and number of days in a week. -/
def total_weekly_distance (circuit_length : ℕ) (morning_runs : ℕ) (afternoon_runs : ℕ) (days_in_week : ℕ) : ℕ :=
  (circuit_length * (morning_runs + afternoon_runs) * days_in_week)

/-- Theorem stating that running a 365-meter circuit 7 times in the morning and 3 times in the afternoon
    for 7 days results in a total distance of 25550 meters. -/
theorem isabel_weekly_distance :
  total_weekly_distance 365 7 3 7 = 25550 := by
  sorry

end NUMINAMATH_CALUDE_isabel_weekly_distance_l1067_106791


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l1067_106746

/-- Given two vectors a and b in a real inner product space such that 
    |a| = |b| = |a - 2b| = 1, prove that |a + 2b| = 3. -/
theorem vector_magnitude_problem (a b : EuclideanSpace ℝ (Fin n)) 
  (h1 : ‖a‖ = 1) 
  (h2 : ‖b‖ = 1) 
  (h3 : ‖a - 2 • b‖ = 1) : 
  ‖a + 2 • b‖ = 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l1067_106746


namespace NUMINAMATH_CALUDE_car_dealer_sales_l1067_106712

theorem car_dealer_sales (x : ℕ) (a b : ℤ) : 
  x > 0 ∧ 
  (7 : ℚ) = (x : ℚ)⁻¹ * (7 * x : ℚ) ∧ 
  (8 : ℚ) = ((x - 1) : ℚ)⁻¹ * ((7 * x - a) : ℚ) ∧ 
  (5 : ℚ) = ((x - 1) : ℚ)⁻¹ * ((7 * x - b) : ℚ) ∧ 
  (23 : ℚ) / 4 = ((x - 2) : ℚ)⁻¹ * ((7 * x - a - b) : ℚ) →
  7 * x = 42 := by
  sorry

end NUMINAMATH_CALUDE_car_dealer_sales_l1067_106712


namespace NUMINAMATH_CALUDE_inequalities_hold_for_all_reals_l1067_106713

-- Define the two quadratic functions
def f (x : ℝ) := x^2 + 6*x + 10
def g (x : ℝ) := -x^2 + x - 2

-- Theorem stating that both inequalities hold for all real numbers
theorem inequalities_hold_for_all_reals :
  (∀ x : ℝ, f x > 0) ∧ (∀ x : ℝ, g x < 0) :=
sorry

end NUMINAMATH_CALUDE_inequalities_hold_for_all_reals_l1067_106713


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1067_106772

theorem polynomial_simplification (x : ℝ) : 
  (3*x^2 - 2*x + 5)*(x - 2) - (x - 2)*(2*x^2 - 5*x + 42) + (2*x - 7)*(x - 2)*(x + 3) = 
  3*x^3 - 4*x^2 - 62*x + 116 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1067_106772


namespace NUMINAMATH_CALUDE_addition_to_reach_91_l1067_106722

theorem addition_to_reach_91 : ∃ x : ℚ, (5 * 12) / (180 / 3) + x = 91 :=
by
  sorry

end NUMINAMATH_CALUDE_addition_to_reach_91_l1067_106722


namespace NUMINAMATH_CALUDE_absolute_value_inequality_supremum_l1067_106757

theorem absolute_value_inequality_supremum :
  (∀ k : ℝ, (∀ x : ℝ, |x + 3| + |x - 1| > k) → k < 4) ∧
  ∀ ε > 0, ∃ x : ℝ, |x + 3| + |x - 1| < 4 + ε :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_supremum_l1067_106757


namespace NUMINAMATH_CALUDE_max_b_value_l1067_106771

theorem max_b_value (a b c : ℕ) (h1 : a * b * c = 360) (h2 : 1 < c) (h3 : c < b) (h4 : b < a) :
  b ≤ 12 ∧ ∃ (a' b' c' : ℕ), a' * b' * c' = 360 ∧ 1 < c' ∧ c' < b' ∧ b' < a' ∧ b' = 12 :=
by sorry

end NUMINAMATH_CALUDE_max_b_value_l1067_106771


namespace NUMINAMATH_CALUDE_marks_trees_l1067_106711

theorem marks_trees (initial_trees : ℕ) (new_trees_per_existing : ℕ) : 
  initial_trees = 93 → new_trees_per_existing = 8 → 
  initial_trees + initial_trees * new_trees_per_existing = 837 := by
sorry

end NUMINAMATH_CALUDE_marks_trees_l1067_106711


namespace NUMINAMATH_CALUDE_classroom_ratio_l1067_106775

/-- Represents a classroom with two portions of students with different GPAs -/
structure Classroom where
  portion_a : ℝ  -- Size of portion A (GPA 15)
  portion_b : ℝ  -- Size of portion B (GPA 18)
  gpa_a : ℝ      -- GPA of portion A
  gpa_b : ℝ      -- GPA of portion B
  gpa_total : ℝ  -- Total GPA of the class

/-- The ratio of portion A to the whole class is 1:3 given the conditions -/
theorem classroom_ratio (c : Classroom) 
  (h1 : c.gpa_a = 15)
  (h2 : c.gpa_b = 18)
  (h3 : c.gpa_total = 17)
  (h4 : c.gpa_a * c.portion_a + c.gpa_b * c.portion_b = c.gpa_total * (c.portion_a + c.portion_b)) :
  c.portion_a / (c.portion_a + c.portion_b) = 1 / 3 := by
  sorry

#check classroom_ratio

end NUMINAMATH_CALUDE_classroom_ratio_l1067_106775


namespace NUMINAMATH_CALUDE_intersection_implies_m_range_l1067_106787

-- Define the sets A and B
def A (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1^2 + m * p.1 + 2}

def B : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 + 1 = 0 ∧ 0 ≤ p.1 ∧ p.1 ≤ 2}

-- State the theorem
theorem intersection_implies_m_range (m : ℝ) :
  (A m ∩ B).Nonempty → m ≤ -1 := by
  sorry


end NUMINAMATH_CALUDE_intersection_implies_m_range_l1067_106787


namespace NUMINAMATH_CALUDE_opponent_total_score_l1067_106726

def TeamScores : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def LostGames (scores : List ℕ) : List ℕ := 
  scores.filter (λ x => x % 2 = 1 ∧ x ≤ 13)

def WonGames (scores : List ℕ) (lostGames : List ℕ) : List ℕ :=
  scores.filter (λ x => x ∉ lostGames)

def OpponentScoresInLostGames (lostGames : List ℕ) : List ℕ :=
  lostGames.map (λ x => x + 1)

def OpponentScoresInWonGames (wonGames : List ℕ) : List ℕ :=
  wonGames.map (λ x => x / 2)

theorem opponent_total_score :
  let lostGames := LostGames TeamScores
  let wonGames := WonGames TeamScores lostGames
  let opponentLostScores := OpponentScoresInLostGames lostGames
  let opponentWonScores := OpponentScoresInWonGames wonGames
  (opponentLostScores.sum + opponentWonScores.sum) = 75 :=
sorry

end NUMINAMATH_CALUDE_opponent_total_score_l1067_106726


namespace NUMINAMATH_CALUDE_product_equals_one_l1067_106765

theorem product_equals_one (n : ℕ) (x : ℕ → ℝ) (f : ℕ → ℝ) :
  n > 2 →
  (∀ i j, i % n = j % n → x i = x j) →
  (∀ i, f i = x i + x i * x (i + 1) + x i * x (i + 1) * x (i + 2) + 
    x i * x (i + 1) * x (i + 2) * x (i + 3) + 
    x i * x (i + 1) * x (i + 2) * x (i + 3) * x (i + 4) + 
    x i * x (i + 1) * x (i + 2) * x (i + 3) * x (i + 4) * x (i + 5) +
    x i * x (i + 1) * x (i + 2) * x (i + 3) * x (i + 4) * x (i + 5) * x (i + 6) +
    x i * x (i + 1) * x (i + 2) * x (i + 3) * x (i + 4) * x (i + 5) * x (i + 6) * x (i + 7)) →
  (∀ i j, f i = f j) →
  (∃ i j, x i ≠ x j) →
  (x 1 * x 2 * x 3 * x 4 * x 5 * x 6 * x 7 * x 8) = 1 := by
sorry

end NUMINAMATH_CALUDE_product_equals_one_l1067_106765


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l1067_106736

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant (m : ℝ) :
  second_quadrant (-1) (m^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l1067_106736


namespace NUMINAMATH_CALUDE_total_cost_of_suits_l1067_106799

def cost_of_first_suit : ℕ := 300

def cost_of_second_suit (first_suit_cost : ℕ) : ℕ :=
  3 * first_suit_cost + 200

def total_cost (first_suit_cost : ℕ) : ℕ :=
  first_suit_cost + cost_of_second_suit first_suit_cost

theorem total_cost_of_suits :
  total_cost cost_of_first_suit = 1400 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_suits_l1067_106799


namespace NUMINAMATH_CALUDE_eggs_taken_l1067_106785

theorem eggs_taken (initial_eggs : ℕ) (remaining_eggs : ℕ) (h1 : initial_eggs = 47) (h2 : remaining_eggs = 42) :
  initial_eggs - remaining_eggs = 5 := by
  sorry

end NUMINAMATH_CALUDE_eggs_taken_l1067_106785


namespace NUMINAMATH_CALUDE_max_food_per_guest_l1067_106747

theorem max_food_per_guest (total_food : ℕ) (min_guests : ℕ) (h1 : total_food = 323) (h2 : min_guests = 162) :
  (total_food / min_guests : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_food_per_guest_l1067_106747


namespace NUMINAMATH_CALUDE_f_greater_g_iff_a_geq_half_l1067_106756

noncomputable section

open Real

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - a - log x

def g (x : ℝ) : ℝ := 1/x - Real.exp 1 / (Real.exp x)

-- State the theorem
theorem f_greater_g_iff_a_geq_half (a : ℝ) :
  (∀ x > 1, f a x > g x) ↔ a ≥ 1/2 := by sorry

end

end NUMINAMATH_CALUDE_f_greater_g_iff_a_geq_half_l1067_106756


namespace NUMINAMATH_CALUDE_addition_of_integers_l1067_106758

theorem addition_of_integers : -10 + 3 = -7 := by
  sorry

end NUMINAMATH_CALUDE_addition_of_integers_l1067_106758


namespace NUMINAMATH_CALUDE_son_work_time_l1067_106727

-- Define the work rates
def man_rate : ℚ := 1 / 10
def combined_rate : ℚ := 1 / 5

-- Define the son's work rate
def son_rate : ℚ := combined_rate - man_rate

-- Theorem to prove
theorem son_work_time :
  son_rate = 1 / 10 ∧ (1 / son_rate : ℚ) = 10 :=
by sorry

end NUMINAMATH_CALUDE_son_work_time_l1067_106727


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1067_106719

theorem polynomial_factorization :
  ∃ (a b c d : ℤ), ∀ (x : ℝ),
    x^4 + x^3 + x^2 + x + 12 = (x^2 + a*x + b) * (x^2 + c*x + d) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1067_106719


namespace NUMINAMATH_CALUDE_min_value_theorem_l1067_106792

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 4 ∧
  (∀ (c d : ℝ), c > 0 → d > 0 → c + d = 4 → 1 / (c + 1) + 1 / (d + 3) ≥ 1 / (x + 1) + 1 / (y + 3)) ∧
  1 / (x + 1) + 1 / (y + 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1067_106792


namespace NUMINAMATH_CALUDE_cost_of_type_B_books_l1067_106745

/-- The cost of purchasing type B books given the total number of books and the number of type A books purchased -/
theorem cost_of_type_B_books (total_books : ℕ) (x : ℕ) (price_B : ℕ) 
  (h_total : total_books = 100)
  (h_price : price_B = 6)
  (h_x_le_total : x ≤ total_books) :
  price_B * (total_books - x) = 6 * (100 - x) :=
by sorry

end NUMINAMATH_CALUDE_cost_of_type_B_books_l1067_106745


namespace NUMINAMATH_CALUDE_scenario_one_registration_methods_scenario_two_registration_methods_scenario_three_registration_methods_l1067_106705

/- Define the number of students and events -/
def num_students : ℕ := 6
def num_events : ℕ := 3

/- Theorem for scenario 1 -/
theorem scenario_one_registration_methods :
  (num_events ^ num_students : ℕ) = 729 := by sorry

/- Theorem for scenario 2 -/
theorem scenario_two_registration_methods :
  (num_students * (num_students - 1) * (num_students - 2) : ℕ) = 120 := by sorry

/- Theorem for scenario 3 -/
theorem scenario_three_registration_methods :
  (num_students ^ num_events : ℕ) = 216 := by sorry

end NUMINAMATH_CALUDE_scenario_one_registration_methods_scenario_two_registration_methods_scenario_three_registration_methods_l1067_106705


namespace NUMINAMATH_CALUDE_unique_integer_divisible_by_18_with_sqrt_between_26_and_26_2_l1067_106709

theorem unique_integer_divisible_by_18_with_sqrt_between_26_and_26_2 :
  ∃! (N : ℕ), 
    N > 0 ∧ 
    N % 18 = 0 ∧ 
    (26 : ℝ) < Real.sqrt N ∧ 
    Real.sqrt N < 26.2 ∧ 
    N = 684 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_divisible_by_18_with_sqrt_between_26_and_26_2_l1067_106709


namespace NUMINAMATH_CALUDE_supermarket_prices_l1067_106779

/-- The price of sugar per kilogram -/
def sugar_price : ℝ := sorry

/-- The price of salt per kilogram -/
def salt_price : ℝ := sorry

/-- The price of rice per kilogram -/
def rice_price : ℝ := sorry

/-- The total price of given quantities of sugar, salt, and rice -/
def total_price (sugar_kg salt_kg rice_kg : ℝ) : ℝ :=
  sugar_kg * sugar_price + salt_kg * salt_price + rice_kg * rice_price

theorem supermarket_prices :
  (total_price 5 3 2 = 28) ∧
  (total_price 4 2 1 = 22) ∧
  (sugar_price = 2 * salt_price) ∧
  (rice_price = 3 * salt_price) →
  total_price 6 4 3 = 36.75 := by
sorry

end NUMINAMATH_CALUDE_supermarket_prices_l1067_106779


namespace NUMINAMATH_CALUDE_transformed_area_theorem_l1067_106743

/-- A 2x2 matrix representing the transformation --/
def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 5, -2]

/-- The area of the original region T --/
def original_area : ℝ := 12

/-- Theorem stating that applying the transformation matrix to a region with area 12 results in a new region with area 312 --/
theorem transformed_area_theorem :
  abs (Matrix.det transformation_matrix) * original_area = 312 := by
  sorry

#check transformed_area_theorem

end NUMINAMATH_CALUDE_transformed_area_theorem_l1067_106743


namespace NUMINAMATH_CALUDE_disjoint_quadratic_sets_l1067_106700

theorem disjoint_quadratic_sets (A B : ℤ) : ∃ C : ℤ,
  ∀ x y : ℤ, x^2 + A*x + B ≠ 2*y^2 + 2*y + C :=
by sorry

end NUMINAMATH_CALUDE_disjoint_quadratic_sets_l1067_106700


namespace NUMINAMATH_CALUDE_sum_of_specific_terms_l1067_106766

def tangent_sequence (a : ℕ → ℝ) : Prop :=
  ∀ k : ℕ, k > 0 → a (k + 1) = (3 / 2) * a k

theorem sum_of_specific_terms 
  (a : ℕ → ℝ) 
  (h1 : tangent_sequence a) 
  (h2 : a 1 = 16) : 
  a 1 + a 3 + a 5 = 133 := by
sorry

end NUMINAMATH_CALUDE_sum_of_specific_terms_l1067_106766


namespace NUMINAMATH_CALUDE_geometric_series_sum_of_cubes_l1067_106768

theorem geometric_series_sum_of_cubes 
  (a : ℝ) (r : ℝ) (hr : -1 < r ∧ r < 1) 
  (h1 : a / (1 - r) = 2) 
  (h2 : a^2 / (1 - r^2) = 6) : 
  a^3 / (1 - r^3) = 96/7 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_of_cubes_l1067_106768


namespace NUMINAMATH_CALUDE_swimmers_passing_theorem_l1067_106777

/-- Represents the number of times two swimmers pass each other in a pool -/
def swimmers_passing_count (pool_length : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ) : ℕ :=
  -- The actual implementation is not provided, as per the instructions
  sorry

/-- Theorem stating the number of times the swimmers pass each other under given conditions -/
theorem swimmers_passing_theorem :
  let pool_length : ℝ := 120
  let speed1 : ℝ := 4
  let speed2 : ℝ := 3
  let total_time : ℝ := 15 * 60  -- 15 minutes in seconds
  swimmers_passing_count pool_length speed1 speed2 total_time = 53 := by
  sorry

end NUMINAMATH_CALUDE_swimmers_passing_theorem_l1067_106777


namespace NUMINAMATH_CALUDE_regular_price_is_0_15_l1067_106732

/-- The regular price per can of soda -/
def regular_price : ℝ := sorry

/-- The discounted price per can when purchased in 24-can cases -/
def discounted_price : ℝ := 0.9 * regular_price

/-- The price of 75 cans purchased in 24-can cases -/
def price_75_cans : ℝ := 10.125

theorem regular_price_is_0_15 : regular_price = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_regular_price_is_0_15_l1067_106732


namespace NUMINAMATH_CALUDE_inequality_always_holds_l1067_106707

theorem inequality_always_holds (m : ℝ) :
  (∀ x : ℝ, 2 * m * x^2 + m * x - 3/4 < 0) → -6 < m ∧ m ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_holds_l1067_106707


namespace NUMINAMATH_CALUDE_not_or_implies_both_false_l1067_106784

theorem not_or_implies_both_false (p q : Prop) :
  ¬(p ∨ q) → ¬p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_not_or_implies_both_false_l1067_106784


namespace NUMINAMATH_CALUDE_floor_x_width_l1067_106770

/-- Represents a rectangular floor with a length and width in feet. -/
structure RectangularFloor where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular floor. -/
def area (floor : RectangularFloor) : ℝ :=
  floor.length * floor.width

theorem floor_x_width
  (x y : RectangularFloor)
  (h1 : area x = area y)
  (h2 : x.length = 18)
  (h3 : y.width = 9)
  (h4 : y.length = 20) :
  x.width = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_x_width_l1067_106770


namespace NUMINAMATH_CALUDE_g_is_odd_l1067_106767

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define g as f(x) - f(-x)
def g (x : ℝ) : ℝ := f x - f (-x)

-- Theorem: g is an odd function
theorem g_is_odd : ∀ x : ℝ, g f (-x) = -(g f x) := by
  sorry

end NUMINAMATH_CALUDE_g_is_odd_l1067_106767


namespace NUMINAMATH_CALUDE_article_selling_price_l1067_106750

theorem article_selling_price (cost_price : ℝ) (profit_percentage : ℝ) (selling_price : ℝ) : 
  cost_price = 90.91 →
  profit_percentage = 10 →
  selling_price = cost_price * (1 + profit_percentage / 100) →
  selling_price = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_article_selling_price_l1067_106750


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1067_106754

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 → 
    ∃ t : ℝ, (b*x + a*y = 0 ∨ b*x - a*y = 0) ∧ 
    x^2 + y^2 - 6*x + 5 = t^2) →
  (3 : ℝ)^2 = a^2 - b^2 →
  a^2 = 5 ∧ b^2 = 4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1067_106754


namespace NUMINAMATH_CALUDE_set_union_problem_l1067_106763

theorem set_union_problem (a b : ℝ) : 
  let M : Set ℝ := {3, 2*a}
  let N : Set ℝ := {a, b}
  M ∩ N = {2} → M ∪ N = {1, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l1067_106763


namespace NUMINAMATH_CALUDE_expression_simplification_l1067_106774

theorem expression_simplification (m : ℝ) (h : m^2 - m - 1 = 0) :
  (m - 1) / (m^2 - 2*m) / (m + 1/(m - 2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1067_106774


namespace NUMINAMATH_CALUDE_oplus_2_3_4_l1067_106776

-- Define the operation ⊕
def oplus (a b c : ℝ) : ℝ := a * b - 4 * a + c^2

-- Theorem statement
theorem oplus_2_3_4 : oplus 2 3 4 = 14 := by sorry

end NUMINAMATH_CALUDE_oplus_2_3_4_l1067_106776


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_a_geq_one_l1067_106761

theorem quadratic_inequality_implies_a_geq_one (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * x + a ≥ 0) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_a_geq_one_l1067_106761


namespace NUMINAMATH_CALUDE_function_inequality_solutions_l1067_106782

noncomputable def f (x : ℝ) : ℝ := (Real.log (2 * x)) / x

theorem function_inequality_solutions (a : ℝ) :
  (∃! (s : Finset ℤ), s.card = 2 ∧ ∀ x : ℤ, (x : ℝ) > 0 → (x ∈ s ↔ f x ^ 2 + a * f x > 0)) ↔
  a ∈ Set.Ioo (-Real.log 2) (-1/3 * Real.log 6) ∪ {-1/3 * Real.log 6} :=
sorry

end NUMINAMATH_CALUDE_function_inequality_solutions_l1067_106782


namespace NUMINAMATH_CALUDE_power_mod_seventeen_l1067_106734

theorem power_mod_seventeen : 7^2023 % 17 = 15 := by sorry

end NUMINAMATH_CALUDE_power_mod_seventeen_l1067_106734


namespace NUMINAMATH_CALUDE_smallest_special_number_after_3429_l1067_106778

/-- A function that returns true if a natural number uses exactly four different digits -/
def uses_four_different_digits (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  n = a * 1000 + b * 100 + c * 10 + d ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10

theorem smallest_special_number_after_3429 :
  ∀ k : ℕ, k > 3429 ∧ k < 3450 → ¬(uses_four_different_digits k) ∧
  uses_four_different_digits 3450 :=
by sorry

end NUMINAMATH_CALUDE_smallest_special_number_after_3429_l1067_106778


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1067_106723

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 6 + a 8 = 10) 
  (h_a3 : a 3 = 1) : 
  a 11 = 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1067_106723


namespace NUMINAMATH_CALUDE_problem_figure_perimeter_l1067_106731

/-- Represents a rectangle with integer side lengths -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the figure described in the problem -/
structure Figure where
  bottom_row : Vector Rectangle 3
  middle_square : Rectangle
  side_rectangles : Vector Rectangle 2

/-- The figure described in the problem -/
def problem_figure : Figure := {
  bottom_row := ⟨[{width := 1, height := 1}, {width := 1, height := 1}, {width := 1, height := 1}], rfl⟩
  middle_square := {width := 1, height := 1}
  side_rectangles := ⟨[{width := 1, height := 2}, {width := 1, height := 2}], rfl⟩
}

/-- Calculates the perimeter of the given figure -/
def perimeter (f : Figure) : ℕ :=
  sorry

/-- Theorem stating that the perimeter of the problem figure is 13 -/
theorem problem_figure_perimeter : perimeter problem_figure = 13 :=
  sorry

end NUMINAMATH_CALUDE_problem_figure_perimeter_l1067_106731


namespace NUMINAMATH_CALUDE_largest_w_value_l1067_106798

theorem largest_w_value (w x y z : ℝ) 
  (sum_eq : w + x + y + z = 25)
  (prod_sum_eq : w*x + w*y + w*z + x*y + x*z + y*z = 2*y + 2*z + 193) :
  w ≤ 25/2 ∧ ∃ (w' : ℝ), w' = 25/2 ∧ 
    w' + x' + y' + z' = 25 ∧ 
    w'*x' + w'*y' + w'*z' + x'*y' + x'*z' + y'*z' = 2*y' + 2*z' + 193 :=
sorry

end NUMINAMATH_CALUDE_largest_w_value_l1067_106798


namespace NUMINAMATH_CALUDE_least_years_to_double_l1067_106702

theorem least_years_to_double (rate : ℝ) (h : rate = 0.5) : 
  (∃ t : ℕ, (1 + rate)^t > 2) ∧ 
  (∀ t : ℕ, (1 + rate)^t > 2 → t ≥ 2) :=
by
  sorry

#check least_years_to_double

end NUMINAMATH_CALUDE_least_years_to_double_l1067_106702


namespace NUMINAMATH_CALUDE_copper_percentage_second_alloy_l1067_106764

/-- Calculates the percentage of copper in the second alloy -/
theorem copper_percentage_second_alloy 
  (desired_percentage : Real) 
  (first_alloy_percentage : Real)
  (first_alloy_weight : Real)
  (total_weight : Real) :
  let second_alloy_weight := total_weight - first_alloy_weight
  let desired_copper := desired_percentage * total_weight / 100
  let first_alloy_copper := first_alloy_percentage * first_alloy_weight / 100
  let second_alloy_copper := desired_copper - first_alloy_copper
  second_alloy_copper / second_alloy_weight * 100 = 21 :=
by
  sorry

#check copper_percentage_second_alloy 19.75 18 45 108

end NUMINAMATH_CALUDE_copper_percentage_second_alloy_l1067_106764


namespace NUMINAMATH_CALUDE_num_outfits_l1067_106715

/-- Number of shirts available -/
def num_shirts : ℕ := 8

/-- Number of ties available -/
def num_ties : ℕ := 5

/-- Number of pairs of pants available -/
def num_pants : ℕ := 4

/-- Number of jackets available -/
def num_jackets : ℕ := 2

/-- Number of tie options (including no tie) -/
def tie_options : ℕ := num_ties + 1

/-- Number of jacket options (including no jacket) -/
def jacket_options : ℕ := num_jackets + 1

/-- Theorem stating the number of distinct outfits -/
theorem num_outfits : num_shirts * num_pants * tie_options * jacket_options = 576 := by
  sorry

end NUMINAMATH_CALUDE_num_outfits_l1067_106715


namespace NUMINAMATH_CALUDE_angle_at_point_l1067_106733

theorem angle_at_point (y : ℝ) : 
  y > 0 ∧ y + y + 140 = 360 → y = 110 := by
  sorry

end NUMINAMATH_CALUDE_angle_at_point_l1067_106733


namespace NUMINAMATH_CALUDE_function_inequality_l1067_106781

open Real

theorem function_inequality (f : ℝ → ℝ) (f_deriv : Differentiable ℝ f) 
  (h1 : ∀ x, f x + deriv f x > 1) (h2 : f 0 = 4) :
  ∀ x, f x > 3 / exp x + 1 ↔ x > 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1067_106781


namespace NUMINAMATH_CALUDE_smallest_number_with_2020_divisors_l1067_106704

def number_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem smallest_number_with_2020_divisors :
  ∀ m : ℕ, m < 2^100 * 3^4 * 5 * 7 →
    number_of_divisors m ≠ 2020 ∧
    number_of_divisors (2^100 * 3^4 * 5 * 7) = 2020 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_2020_divisors_l1067_106704


namespace NUMINAMATH_CALUDE_goldbach_multiplication_counterexample_l1067_106739

theorem goldbach_multiplication_counterexample :
  ∃ p : ℕ, Prime p ∧ p > 5 ∧
  (∀ q r : ℕ, Prime q → Prime r → Odd q → Odd r → p ≠ q * r) ∧
  (∀ q : ℕ, Prime q → Odd q → p ≠ q^2) := by
  sorry

end NUMINAMATH_CALUDE_goldbach_multiplication_counterexample_l1067_106739


namespace NUMINAMATH_CALUDE_total_reading_materials_l1067_106730

def magazines : ℕ := 425
def newspapers : ℕ := 275

theorem total_reading_materials : magazines + newspapers = 700 := by
  sorry

end NUMINAMATH_CALUDE_total_reading_materials_l1067_106730


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l1067_106737

theorem complete_square_quadratic (x : ℝ) : 
  (∃ c d : ℝ, x^2 + 6*x - 3 = 0 ↔ (x + c)^2 = d) → 
  (∃ c : ℝ, (x + c)^2 = 12) :=
by sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l1067_106737


namespace NUMINAMATH_CALUDE_additional_lawn_to_mow_l1067_106793

/-- The problem of calculating additional square feet to mow -/
theorem additional_lawn_to_mow 
  (rate : ℚ) 
  (book_cost : ℚ) 
  (lawns_mowed : ℕ) 
  (lawn_length : ℕ) 
  (lawn_width : ℕ) : 
  rate = 1/10 → 
  book_cost = 150 → 
  lawns_mowed = 3 → 
  lawn_length = 20 → 
  lawn_width = 15 → 
  (book_cost - lawns_mowed * lawn_length * lawn_width * rate) / rate = 600 := by
  sorry

#check additional_lawn_to_mow

end NUMINAMATH_CALUDE_additional_lawn_to_mow_l1067_106793


namespace NUMINAMATH_CALUDE_quotient_change_l1067_106762

theorem quotient_change (a b : ℝ) (h : b ≠ 0) :
  ((100 * a) / (b / 10)) = 1000 * (a / b) := by
sorry

end NUMINAMATH_CALUDE_quotient_change_l1067_106762


namespace NUMINAMATH_CALUDE_tv_production_last_period_avg_l1067_106701

/-- Represents the production of TVs in a factory over a month -/
structure TVProduction where
  totalDays : Nat
  firstPeriodDays : Nat
  firstPeriodAvg : Nat
  monthlyAvg : Nat

/-- Calculates the average production for the last period of the month -/
def lastPeriodAvg (p : TVProduction) : Rat :=
  let lastPeriodDays := p.totalDays - p.firstPeriodDays
  let totalProduction := p.totalDays * p.monthlyAvg
  let firstPeriodProduction := p.firstPeriodDays * p.firstPeriodAvg
  (totalProduction - firstPeriodProduction) / lastPeriodDays

/-- Theorem stating that given the conditions, the average production for the last 5 days is 20 TVs per day -/
theorem tv_production_last_period_avg 
  (p : TVProduction) 
  (h1 : p.totalDays = 30) 
  (h2 : p.firstPeriodDays = 25) 
  (h3 : p.firstPeriodAvg = 50) 
  (h4 : p.monthlyAvg = 45) : 
  lastPeriodAvg p = 20 := by
  sorry

end NUMINAMATH_CALUDE_tv_production_last_period_avg_l1067_106701


namespace NUMINAMATH_CALUDE_number_problem_l1067_106717

theorem number_problem : ∃ x : ℝ, (x / 3) * 12 = 9 ∧ x = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1067_106717


namespace NUMINAMATH_CALUDE_unique_function_property_l1067_106753

theorem unique_function_property (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = 0 ↔ x = 0)
  (h2 : ∀ x y, f (x^2 + y * f x) + f (y^2 + x * f y) = (f (x + y))^2) :
  ∀ x, f x = x := by
  sorry

end NUMINAMATH_CALUDE_unique_function_property_l1067_106753


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l1067_106760

theorem jelly_bean_probability 
  (red_prob : ℝ) 
  (orange_prob : ℝ) 
  (green_prob : ℝ) 
  (h1 : red_prob = 0.15)
  (h2 : orange_prob = 0.4)
  (h3 : green_prob = 0.1)
  (h4 : ∃ yellow_prob : ℝ, red_prob + orange_prob + yellow_prob + green_prob = 1) :
  ∃ yellow_prob : ℝ, yellow_prob = 0.35 ∧ red_prob + orange_prob + yellow_prob + green_prob = 1 := by
sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l1067_106760


namespace NUMINAMATH_CALUDE_fourth_root_power_eight_l1067_106773

theorem fourth_root_power_eight : (2^6)^(1/4)^8 = 4096 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_power_eight_l1067_106773


namespace NUMINAMATH_CALUDE_max_small_boxes_in_large_box_l1067_106725

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- The dimensions of the large box -/
def largeBox : BoxDimensions :=
  { length := 12, width := 14, height := 16 }

/-- The dimensions of the small box -/
def smallBox : BoxDimensions :=
  { length := 3, width := 7, height := 2 }

/-- Theorem stating the maximum number of small boxes that fit into the large box -/
theorem max_small_boxes_in_large_box :
  boxVolume largeBox / boxVolume smallBox = 64 := by
  sorry

end NUMINAMATH_CALUDE_max_small_boxes_in_large_box_l1067_106725


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1067_106708

theorem polynomial_evaluation :
  let f : ℝ → ℝ := λ x => 2*x^4 + 3*x^3 + x^2 + 2*x + 3
  f 2 = 67 := by
sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1067_106708


namespace NUMINAMATH_CALUDE_game_download_proof_l1067_106720

/-- Proves that the amount downloaded before the connection slowed down is 310 MB -/
theorem game_download_proof (total_size : ℕ) (current_speed : ℕ) (remaining_time : ℕ) 
  (h1 : total_size = 880)
  (h2 : current_speed = 3)
  (h3 : remaining_time = 190) :
  total_size - current_speed * remaining_time = 310 := by
  sorry

end NUMINAMATH_CALUDE_game_download_proof_l1067_106720


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l1067_106769

/-- Given an arithmetic sequence where the 3rd term is 23 and the 5th term is 43,
    the 9th term is 83. -/
theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h1 : a 3 = 23)  -- The 3rd term is 23
  (h2 : a 5 = 43)  -- The 5th term is 43
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)  -- The sequence is arithmetic
  : a 9 = 83 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l1067_106769


namespace NUMINAMATH_CALUDE_square_area_proof_l1067_106714

/-- Given a square with side length equal to both 5x - 21 and 29 - 2x,
    prove that its area is 10609/49 square meters. -/
theorem square_area_proof (x : ℝ) (h : 5 * x - 21 = 29 - 2 * x) :
  (5 * x - 21) ^ 2 = 10609 / 49 := by
  sorry

end NUMINAMATH_CALUDE_square_area_proof_l1067_106714


namespace NUMINAMATH_CALUDE_sequence_2017th_term_l1067_106710

theorem sequence_2017th_term (a : ℕ+ → ℚ) 
  (h1 : a 1 = 0)
  (h2 : ∀ n : ℕ+, n ≥ 2 → (1 / (1 - a n) - 1 / (1 - a (n - 1)) = 1)) :
  a 2017 = 2016 / 2017 := by
  sorry

end NUMINAMATH_CALUDE_sequence_2017th_term_l1067_106710


namespace NUMINAMATH_CALUDE_connected_vessels_equilibrium_l1067_106790

/-- Represents the final levels of liquids in two connected vessels after equilibrium -/
def FinalLevels (H : ℝ) : ℝ × ℝ :=
  (0.69 * H, H)

/-- Proves that the given final levels are correct for the connected vessels problem -/
theorem connected_vessels_equilibrium 
  (H : ℝ) 
  (h_positive : H > 0) 
  (ρ_water : ℝ) 
  (ρ_gasoline : ℝ) 
  (h_water_density : ρ_water = 1000) 
  (h_gasoline_density : ρ_gasoline = 600) 
  (h_initial_level : ℝ) 
  (h_initial : h_initial = 0.9 * H) 
  (h_tap_height : ℝ) 
  (h_tap : h_tap_height = 0.2 * H) : 
  FinalLevels H = (0.69 * H, H) :=
sorry

#check connected_vessels_equilibrium

end NUMINAMATH_CALUDE_connected_vessels_equilibrium_l1067_106790


namespace NUMINAMATH_CALUDE_min_packages_correct_min_packages_value_l1067_106724

/-- The number of t-shirts in each package -/
def package_size : ℕ := 6

/-- The number of t-shirts Mom wants to buy -/
def desired_shirts : ℕ := 71

/-- The minimum number of packages needed to buy at least the desired number of shirts -/
def min_packages : ℕ := (desired_shirts + package_size - 1) / package_size

theorem min_packages_correct : 
  min_packages * package_size ≥ desired_shirts ∧ 
  ∀ k : ℕ, k * package_size ≥ desired_shirts → k ≥ min_packages :=
by sorry

theorem min_packages_value : min_packages = 12 :=
by sorry

end NUMINAMATH_CALUDE_min_packages_correct_min_packages_value_l1067_106724


namespace NUMINAMATH_CALUDE_number_of_digits_in_N_l1067_106716

theorem number_of_digits_in_N : ∃ (N : ℕ), 
  N = 2^12 * 5^8 ∧ (Nat.log 10 N + 1 = 10) := by sorry

end NUMINAMATH_CALUDE_number_of_digits_in_N_l1067_106716


namespace NUMINAMATH_CALUDE_bicycle_price_calculation_l1067_106794

theorem bicycle_price_calculation (initial_cost : ℝ) (profit1 : ℝ) (profit2 : ℝ) : 
  initial_cost = 150 ∧ profit1 = 0.20 ∧ profit2 = 0.25 →
  (initial_cost * (1 + profit1)) * (1 + profit2) = 225 :=
by
  sorry

end NUMINAMATH_CALUDE_bicycle_price_calculation_l1067_106794


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l1067_106721

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {4, 5}

theorem complement_of_M_in_U :
  (U \ M) = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l1067_106721


namespace NUMINAMATH_CALUDE_g_of_3_l1067_106780

def g (x : ℝ) : ℝ := 5 * x^3 - 6 * x^2 - 3 * x + 5

theorem g_of_3 : g 3 = 77 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_l1067_106780


namespace NUMINAMATH_CALUDE_square_difference_l1067_106795

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1067_106795


namespace NUMINAMATH_CALUDE_difference_of_squares_l1067_106752

theorem difference_of_squares : 65^2 - 55^2 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1067_106752


namespace NUMINAMATH_CALUDE_neither_direct_nor_inverse_proportional_l1067_106728

/-- A function representing the relationship between x and y --/
def Relationship (f : ℝ → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ x y, y = f x → (y = k * x ∨ x * y = k)

/-- Equation A: 2x + y = 5 --/
def EquationA (x y : ℝ) : Prop := 2 * x + y = 5

/-- Equation B: 4xy = 12 --/
def EquationB (x y : ℝ) : Prop := 4 * x * y = 12

/-- Equation C: x = 4y --/
def EquationC (x y : ℝ) : Prop := x = 4 * y

/-- Equation D: 2x + 3y = 15 --/
def EquationD (x y : ℝ) : Prop := 2 * x + 3 * y = 15

/-- Equation E: x/y = 4 --/
def EquationE (x y : ℝ) : Prop := x / y = 4

theorem neither_direct_nor_inverse_proportional :
  (¬ Relationship (λ x => 5 - 2 * x)) ∧
  (Relationship (λ x => 3 / (4 * x))) ∧
  (Relationship (λ x => x / 4)) ∧
  (¬ Relationship (λ x => (15 - 2 * x) / 3)) ∧
  (Relationship (λ x => x / 4)) :=
sorry

end NUMINAMATH_CALUDE_neither_direct_nor_inverse_proportional_l1067_106728


namespace NUMINAMATH_CALUDE_min_turtle_distance_l1067_106744

/-- Represents an observer watching the turtle --/
structure Observer where
  startTime : ℕ
  endTime : ℕ
  distanceObserved : ℕ

/-- Represents the turtle's movement --/
def TurtleMovement (observers : List Observer) : Prop :=
  -- The observation lasts for 6 minutes
  (∀ o ∈ observers, o.startTime ≥ 0 ∧ o.endTime ≤ 6 * 60) ∧
  -- Each observer watches for 1 minute continuously
  (∀ o ∈ observers, o.endTime - o.startTime = 60) ∧
  -- Each observer notes 1 meter of movement
  (∀ o ∈ observers, o.distanceObserved = 1) ∧
  -- The turtle is always being observed
  (∀ t : ℕ, t ≥ 0 ∧ t ≤ 6 * 60 → ∃ o ∈ observers, o.startTime ≤ t ∧ t < o.endTime)

/-- The theorem stating the minimum distance the turtle could have traveled --/
theorem min_turtle_distance (observers : List Observer) 
  (h : TurtleMovement observers) : 
  ∃ d : ℕ, d = 4 ∧ (∀ d' : ℕ, (∃ obs : List Observer, TurtleMovement obs ∧ d' = obs.length) → d ≤ d') :=
sorry

end NUMINAMATH_CALUDE_min_turtle_distance_l1067_106744
