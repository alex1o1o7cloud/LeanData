import Mathlib

namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l2321_232112

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l2321_232112


namespace NUMINAMATH_CALUDE_min_nSn_l2321_232125

/-- An arithmetic sequence with sum properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  sum_10 : S 10 = 0
  sum_15 : S 15 = 25

/-- The product of n and S_n for an arithmetic sequence -/
def nSn (seq : ArithmeticSequence) (n : ℕ) : ℝ := n * seq.S n

/-- The minimum value of nS_n for the given arithmetic sequence -/
theorem min_nSn (seq : ArithmeticSequence) :
  ∃ (min : ℝ), min = -49 ∧ ∀ (n : ℕ), n ≠ 0 → min ≤ nSn seq n :=
sorry

end NUMINAMATH_CALUDE_min_nSn_l2321_232125


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2321_232165

theorem quadratic_inequality_solution_sets 
  (a b : ℝ) 
  (h : Set.Ioo 2 3 = {x | x^2 + a*x + b < 0}) : 
  {x : ℝ | b*x^2 + a*x + 1 > 0} = Set.Iic (1/3) ∪ Set.Ioi (1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2321_232165


namespace NUMINAMATH_CALUDE_total_value_calculation_l2321_232136

/-- Calculates the total value of coins and paper money with a certificate bonus --/
def totalValue (goldWorth silverWorth bronzeWorth titaniumWorth : ℝ)
                (banknoteWorth couponWorth voucherWorth : ℝ)
                (goldCount silverCount bronzeCount titaniumCount : ℕ)
                (banknoteCount couponCount voucherCount : ℕ)
                (certificateBonus : ℝ) : ℝ :=
  let goldValue := goldWorth * goldCount
  let silverValue := silverWorth * silverCount
  let bronzeValue := bronzeWorth * bronzeCount
  let titaniumValue := titaniumWorth * titaniumCount
  let banknoteValue := banknoteWorth * banknoteCount
  let couponValue := couponWorth * couponCount
  let voucherValue := voucherWorth * voucherCount
  let baseTotal := goldValue + silverValue + bronzeValue + titaniumValue +
                   banknoteValue + couponValue + voucherValue
  let bonusAmount := certificateBonus * (goldValue + silverValue)
  baseTotal + bonusAmount

theorem total_value_calculation :
  totalValue 80 45 25 10 50 10 20 7 9 12 5 3 6 4 0.05 = 1653.25 := by
  sorry

end NUMINAMATH_CALUDE_total_value_calculation_l2321_232136


namespace NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l2321_232193

theorem smallest_m_for_integral_solutions : 
  (∀ m : ℕ, m > 0 ∧ m < 160 → ¬∃ x : ℤ, 10 * x^2 - m * x + 630 = 0) ∧ 
  (∃ x : ℤ, 10 * x^2 - 160 * x + 630 = 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l2321_232193


namespace NUMINAMATH_CALUDE_choose_starters_with_triplet_l2321_232145

/-- The number of players in the soccer team -/
def total_players : ℕ := 16

/-- The number of triplets -/
def num_triplets : ℕ := 3

/-- The number of starters to be chosen -/
def num_starters : ℕ := 7

/-- The number of ways to choose 7 starters from 16 players with at least one triplet -/
def ways_to_choose_starters : ℕ := 9721

/-- Theorem stating that the number of ways to choose 7 starters from 16 players,
    including a set of triplets, such that at least one of the triplets is in the
    starting lineup, is equal to 9721 -/
theorem choose_starters_with_triplet :
  (Nat.choose num_triplets 1 * Nat.choose (total_players - num_triplets) (num_starters - 1) +
   Nat.choose num_triplets 2 * Nat.choose (total_players - num_triplets) (num_starters - 2) +
   Nat.choose num_triplets 3 * Nat.choose (total_players - num_triplets) (num_starters - 3)) =
  ways_to_choose_starters :=
by sorry

end NUMINAMATH_CALUDE_choose_starters_with_triplet_l2321_232145


namespace NUMINAMATH_CALUDE_age_ratio_problem_l2321_232103

theorem age_ratio_problem (ann_age : ℕ) (x : ℚ) : 
  ann_age = 6 →
  (ann_age + 10) + (x * ann_age + 10) = 38 →
  x * ann_age / ann_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l2321_232103


namespace NUMINAMATH_CALUDE_count_sequences_with_at_least_three_heads_l2321_232184

/-- The number of distinct sequences of 10 coin flips containing at least 3 heads -/
def sequences_with_at_least_three_heads : ℕ :=
  2^10 - (Nat.choose 10 0 + Nat.choose 10 1 + Nat.choose 10 2)

/-- Theorem stating that the number of sequences with at least 3 heads is 968 -/
theorem count_sequences_with_at_least_three_heads :
  sequences_with_at_least_three_heads = 968 := by
  sorry

end NUMINAMATH_CALUDE_count_sequences_with_at_least_three_heads_l2321_232184


namespace NUMINAMATH_CALUDE_marks_lost_per_wrong_answer_l2321_232134

theorem marks_lost_per_wrong_answer 
  (total_questions : ℕ)
  (marks_per_correct : ℕ)
  (total_marks : ℕ)
  (correct_answers : ℕ)
  (h1 : total_questions = 60)
  (h2 : marks_per_correct = 4)
  (h3 : total_marks = 160)
  (h4 : correct_answers = 44)
  : ℕ :=
by
  sorry

#check marks_lost_per_wrong_answer

end NUMINAMATH_CALUDE_marks_lost_per_wrong_answer_l2321_232134


namespace NUMINAMATH_CALUDE_employed_females_percentage_l2321_232122

theorem employed_females_percentage (population : ℝ) 
  (h1 : population > 0)
  (employed : ℝ) 
  (h2 : employed = 0.6 * population)
  (employed_males : ℝ) 
  (h3 : employed_males = 0.42 * population) :
  (employed - employed_males) / employed = 0.3 := by
sorry

end NUMINAMATH_CALUDE_employed_females_percentage_l2321_232122


namespace NUMINAMATH_CALUDE_least_reducible_fraction_l2321_232111

/-- A fraction a/b is reducible if gcd(a,b) > 1 -/
def IsReducible (a b : ℤ) : Prop := Int.gcd a b > 1

/-- The numerator of our fraction -/
def Numerator (m : ℕ) : ℤ := m - 17

/-- The denominator of our fraction -/
def Denominator (m : ℕ) : ℤ := 6 * m + 7

theorem least_reducible_fraction :
  (∀ m : ℕ, m < 126 → ¬(IsReducible (Numerator m) (Denominator m))) ∧
  IsReducible (Numerator 126) (Denominator 126) := by
  sorry

end NUMINAMATH_CALUDE_least_reducible_fraction_l2321_232111


namespace NUMINAMATH_CALUDE_max_individual_score_l2321_232119

theorem max_individual_score (n : ℕ) (total : ℕ) (min_score : ℕ) 
  (h1 : n = 12)
  (h2 : total = 100)
  (h3 : min_score = 7)
  (h4 : ∀ p : ℕ, p ≤ n → min_score ≤ p) :
  ∃ max_score : ℕ, 
    (∀ p : ℕ, p ≤ n → p ≤ max_score) ∧ 
    (∃ player : ℕ, player ≤ n ∧ player = max_score) ∧
    max_score = 23 :=
sorry

end NUMINAMATH_CALUDE_max_individual_score_l2321_232119


namespace NUMINAMATH_CALUDE_reading_homework_pages_l2321_232181

theorem reading_homework_pages
  (math_pages : ℕ)
  (problems_per_page : ℕ)
  (total_problems : ℕ)
  (h1 : math_pages = 6)
  (h2 : problems_per_page = 3)
  (h3 : total_problems = 30) :
  (total_problems - math_pages * problems_per_page) / problems_per_page = 4 :=
by sorry

end NUMINAMATH_CALUDE_reading_homework_pages_l2321_232181


namespace NUMINAMATH_CALUDE_unique_solution_system_l2321_232154

theorem unique_solution_system : 
  ∃! (a b c : ℕ+), 
    (a.val : ℤ)^3 - (b.val : ℤ)^3 - (c.val : ℤ)^3 = 3 * (a.val : ℤ) * (b.val : ℤ) * (c.val : ℤ) ∧ 
    (a.val : ℤ)^2 = 2 * ((b.val : ℤ) + (c.val : ℤ)) ∧
    a.val = 2 ∧ b.val = 1 ∧ c.val = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l2321_232154


namespace NUMINAMATH_CALUDE_cosine_period_problem_l2321_232106

theorem cosine_period_problem (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x : ℝ, ∃ y : ℝ, y = a * Real.cos (b * x + c) + d) →
  (∀ x : ℝ, ∃ y : ℝ, y = a * Real.cos (b * (x + 2 * π) + c) + d) →
  b = 2 := by
sorry

end NUMINAMATH_CALUDE_cosine_period_problem_l2321_232106


namespace NUMINAMATH_CALUDE_integer_roots_count_l2321_232156

/-- Represents a fourth-degree polynomial with integer coefficients -/
structure IntPolynomial where
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ

/-- The number of integer roots of an IntPolynomial, counting multiplicity -/
def num_integer_roots (p : IntPolynomial) : ℕ := sorry

/-- Theorem stating the possible values for the number of integer roots -/
theorem integer_roots_count (p : IntPolynomial) :
  num_integer_roots p = 0 ∨ num_integer_roots p = 1 ∨ num_integer_roots p = 2 ∨ num_integer_roots p = 4 :=
sorry

end NUMINAMATH_CALUDE_integer_roots_count_l2321_232156


namespace NUMINAMATH_CALUDE_classroom_pairing_probability_l2321_232123

/-- The probability of two specific students being paired in a classroom. -/
def pairProbability (n : ℕ) : ℚ :=
  1 / (n - 1)

/-- Theorem: In a classroom of 24 students where each student is randomly paired
    with another, the probability of a specific student being paired with
    another specific student is 1/23. -/
theorem classroom_pairing_probability :
  pairProbability 24 = 1 / 23 := by
  sorry

#eval pairProbability 24

end NUMINAMATH_CALUDE_classroom_pairing_probability_l2321_232123


namespace NUMINAMATH_CALUDE_equation_solution_l2321_232115

theorem equation_solution : ∃ s : ℚ, 
  (s^2 - 6*s + 8) / (s^2 - 9*s + 14) = (s^2 - 3*s - 18) / (s^2 - 2*s - 24) ∧ 
  s = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2321_232115


namespace NUMINAMATH_CALUDE_arrangements_count_l2321_232150

/-- The number of different arrangements for 6 students where two specific students cannot stand together -/
def number_of_arrangements : ℕ := 480

/-- The total number of students -/
def total_students : ℕ := 6

/-- The number of students that can be arranged freely -/
def free_students : ℕ := 4

/-- The number of gaps after arranging the free students -/
def number_of_gaps : ℕ := 5

/-- The number of students that cannot stand together -/
def restricted_students : ℕ := 2

theorem arrangements_count :
  number_of_arrangements = 
    (Nat.factorial free_students) * (number_of_gaps * (number_of_gaps - 1)) :=
by sorry

end NUMINAMATH_CALUDE_arrangements_count_l2321_232150


namespace NUMINAMATH_CALUDE_watch_cost_price_l2321_232159

/-- Proves that the cost price of a watch is 1500 Rs. given the conditions of the problem -/
theorem watch_cost_price (loss_percentage : ℚ) (gain_percentage : ℚ) (price_difference : ℚ) :
  loss_percentage = 10 / 100 →
  gain_percentage = 5 / 100 →
  price_difference = 225 →
  ∃ (cost_price : ℚ), 
    (1 - loss_percentage) * cost_price + price_difference = (1 + gain_percentage) * cost_price ∧
    cost_price = 1500 := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_price_l2321_232159


namespace NUMINAMATH_CALUDE_typist_salary_problem_l2321_232102

/-- Given a salary that is first increased by 10% and then decreased by 5%,
    resulting in Rs. 2090, prove that the original salary was Rs. 2000. -/
theorem typist_salary_problem (S : ℝ) : 
  S * 1.1 * 0.95 = 2090 → S = 2000 := by
  sorry

#check typist_salary_problem

end NUMINAMATH_CALUDE_typist_salary_problem_l2321_232102


namespace NUMINAMATH_CALUDE_no_solution_arcsin_arccos_squared_l2321_232104

theorem no_solution_arcsin_arccos_squared (x : ℝ) : 
  (Real.arcsin x + Real.arccos x = π / 2) → (Real.arcsin x)^2 + (Real.arccos x)^2 ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_arcsin_arccos_squared_l2321_232104


namespace NUMINAMATH_CALUDE_area_of_S_l2321_232135

/-- A regular octagon in the complex plane -/
structure RegularOctagon where
  center : ℂ
  side_distance : ℝ
  parallel_to_real_axis : Prop

/-- The region outside the octagon -/
def R (octagon : RegularOctagon) : Set ℂ :=
  sorry

/-- The set S defined by the inversion of R -/
def S (octagon : RegularOctagon) : Set ℂ :=
  {w | ∃ z ∈ R octagon, w = 1 / z}

/-- The area of a set in the complex plane -/
noncomputable def area (s : Set ℂ) : ℝ :=
  sorry

theorem area_of_S (octagon : RegularOctagon) 
    (h1 : octagon.center = 0)
    (h2 : octagon.side_distance = 1.5)
    (h3 : octagon.parallel_to_real_axis) :
  area (S octagon) = 9 * Real.pi + 4.5 * Real.sqrt 2 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_area_of_S_l2321_232135


namespace NUMINAMATH_CALUDE_orange_count_proof_l2321_232148

/-- The number of apples in the basket -/
def num_apples : ℕ := 10

/-- The number of oranges added to the basket -/
def added_oranges : ℕ := 5

/-- The initial number of oranges in the basket -/
def initial_oranges : ℕ := 5

theorem orange_count_proof :
  (num_apples : ℚ) = (1 / 2 : ℚ) * ((num_apples : ℚ) + (initial_oranges : ℚ) + (added_oranges : ℚ)) :=
by sorry

end NUMINAMATH_CALUDE_orange_count_proof_l2321_232148


namespace NUMINAMATH_CALUDE_triangle_angle_value_l2321_232146

/-- Given a triangle ABC with angle C = 60°, angle A = x, and angle B = 2x,
    where x is also an alternate interior angle formed by a line intersecting two parallel lines,
    prove that x = 40°. -/
theorem triangle_angle_value (A B C : ℝ) (x : ℝ) : 
  A = x → B = 2*x → C = 60 → A + B + C = 180 → x = 40 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_value_l2321_232146


namespace NUMINAMATH_CALUDE_pages_copied_for_35_dollars_l2321_232187

-- Define the cost per 3 pages in cents
def cost_per_3_pages : ℚ := 7

-- Define the budget in dollars
def budget : ℚ := 35

-- Define the function to calculate the number of pages
def pages_copied (cost_per_3_pages budget : ℚ) : ℚ :=
  (budget * 100) * (3 / cost_per_3_pages)

-- Theorem statement
theorem pages_copied_for_35_dollars :
  pages_copied cost_per_3_pages budget = 1500 := by
  sorry

end NUMINAMATH_CALUDE_pages_copied_for_35_dollars_l2321_232187


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_4_and_5_l2321_232109

theorem smallest_four_digit_divisible_by_4_and_5 :
  ∀ n : ℕ, 1000 ≤ n → n < 10000 → n % 4 = 0 → n % 5 = 0 → 1000 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_4_and_5_l2321_232109


namespace NUMINAMATH_CALUDE_max_attendees_is_three_tuesday_has_three_friday_has_three_saturday_has_three_no_day_exceeds_three_l2321_232118

-- Define the days of the week
inductive Day
| Mon | Tues | Wed | Thurs | Fri | Sat

-- Define the people
inductive Person
| Amy | Bob | Charlie | Diana | Evan

-- Define the availability function
def available : Person → Day → Bool
| Person.Amy, Day.Mon => false
| Person.Amy, Day.Tues => true
| Person.Amy, Day.Wed => false
| Person.Amy, Day.Thurs => false
| Person.Amy, Day.Fri => true
| Person.Amy, Day.Sat => true
| Person.Bob, Day.Mon => true
| Person.Bob, Day.Tues => false
| Person.Bob, Day.Wed => true
| Person.Bob, Day.Thurs => true
| Person.Bob, Day.Fri => false
| Person.Bob, Day.Sat => true
| Person.Charlie, Day.Mon => false
| Person.Charlie, Day.Tues => false
| Person.Charlie, Day.Wed => false
| Person.Charlie, Day.Thurs => true
| Person.Charlie, Day.Fri => true
| Person.Charlie, Day.Sat => false
| Person.Diana, Day.Mon => true
| Person.Diana, Day.Tues => true
| Person.Diana, Day.Wed => false
| Person.Diana, Day.Thurs => false
| Person.Diana, Day.Fri => true
| Person.Diana, Day.Sat => false
| Person.Evan, Day.Mon => false
| Person.Evan, Day.Tues => true
| Person.Evan, Day.Wed => true
| Person.Evan, Day.Thurs => false
| Person.Evan, Day.Fri => false
| Person.Evan, Day.Sat => true

-- Count the number of available people for a given day
def countAvailable (d : Day) : Nat :=
  (List.filter (λ p => available p d) [Person.Amy, Person.Bob, Person.Charlie, Person.Diana, Person.Evan]).length

-- Find the maximum number of available people across all days
def maxAvailable : Nat :=
  List.foldl max 0 (List.map countAvailable [Day.Mon, Day.Tues, Day.Wed, Day.Thurs, Day.Fri, Day.Sat])

-- Theorem: The maximum number of attendees is 3
theorem max_attendees_is_three : maxAvailable = 3 := by sorry

-- Theorem: Tuesday has 3 attendees
theorem tuesday_has_three : countAvailable Day.Tues = 3 := by sorry

-- Theorem: Friday has 3 attendees
theorem friday_has_three : countAvailable Day.Fri = 3 := by sorry

-- Theorem: Saturday has 3 attendees
theorem saturday_has_three : countAvailable Day.Sat = 3 := by sorry

-- Theorem: No other day has more than 3 attendees
theorem no_day_exceeds_three : ∀ d : Day, countAvailable d ≤ 3 := by sorry

end NUMINAMATH_CALUDE_max_attendees_is_three_tuesday_has_three_friday_has_three_saturday_has_three_no_day_exceeds_three_l2321_232118


namespace NUMINAMATH_CALUDE_panda_bamboo_consumption_l2321_232139

/-- The amount of bamboo eaten by bigger pandas each day -/
def bigger_panda_bamboo : ℝ := 275

/-- The number of small pandas -/
def small_pandas : ℕ := 4

/-- The number of bigger pandas -/
def bigger_pandas : ℕ := 5

/-- The amount of bamboo eaten by small pandas each day -/
def small_panda_bamboo : ℝ := 25

/-- The total amount of bamboo eaten by all pandas in a week -/
def total_weekly_bamboo : ℝ := 2100

/-- The number of days in a week -/
def days_in_week : ℕ := 7

theorem panda_bamboo_consumption :
  bigger_panda_bamboo * bigger_pandas * days_in_week +
  small_panda_bamboo * small_pandas * days_in_week =
  total_weekly_bamboo :=
sorry

end NUMINAMATH_CALUDE_panda_bamboo_consumption_l2321_232139


namespace NUMINAMATH_CALUDE_jakesDrinkVolume_l2321_232163

/-- Represents the composition of a drink mixture -/
structure DrinkMixture where
  coke : ℕ
  sprite : ℕ
  mountainDew : ℕ

/-- Calculates the total parts in a drink mixture -/
def totalParts (d : DrinkMixture) : ℕ := d.coke + d.sprite + d.mountainDew

/-- Represents Jake's drink mixture -/
def jakesDrink : DrinkMixture := { coke := 2, sprite := 1, mountainDew := 3 }

/-- The volume of Coke in Jake's drink in ounces -/
def cokeVolume : ℕ := 6

/-- Theorem: Jake's drink has a total volume of 18 ounces -/
theorem jakesDrinkVolume : 
  (cokeVolume * totalParts jakesDrink) / jakesDrink.coke = 18 := by
  sorry

end NUMINAMATH_CALUDE_jakesDrinkVolume_l2321_232163


namespace NUMINAMATH_CALUDE_percentage_decrease_l2321_232186

theorem percentage_decrease (initial : ℝ) (increase_percent : ℝ) (final : ℝ) :
  initial = 1500 →
  increase_percent = 20 →
  final = 1080 →
  ∃ (decrease_percent : ℝ),
    final = (initial * (1 + increase_percent / 100)) * (1 - decrease_percent / 100) ∧
    decrease_percent = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_decrease_l2321_232186


namespace NUMINAMATH_CALUDE_square_congruent_one_count_l2321_232189

/-- For n ≥ 2, the number of integers x with 0 ≤ x < n such that x² ≡ 1 (mod n) 
    is equal to 2 times the number of pairs (a, b) such that ab = n and gcd(a, b) = 1 -/
theorem square_congruent_one_count (n : ℕ) (h : n ≥ 2) :
  (Finset.filter (fun x => x^2 % n = 1) (Finset.range n)).card =
  2 * (Finset.filter (fun p : ℕ × ℕ => p.1 * p.2 = n ∧ Nat.gcd p.1 p.2 = 1) 
    (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1)))).card := by
  sorry

end NUMINAMATH_CALUDE_square_congruent_one_count_l2321_232189


namespace NUMINAMATH_CALUDE_geometric_progression_arcsin_least_t_l2321_232170

theorem geometric_progression_arcsin_least_t : 
  ∃ (t : ℝ), t > 0 ∧ 
  (∀ (α : ℝ), 0 < α → α < π / 2 → 
    ∃ (r : ℝ), r > 0 ∧
    (Real.arcsin (Real.sin α) = α) ∧
    (Real.arcsin (Real.sin (3 * α)) = r * α) ∧
    (Real.arcsin (Real.sin (8 * α)) = r^2 * α) ∧
    (Real.arcsin (Real.sin (t * α)) = r^3 * α)) ∧
  (∀ (t' : ℝ), t' > 0 → 
    (∀ (α : ℝ), 0 < α → α < π / 2 → 
      ∃ (r : ℝ), r > 0 ∧
      (Real.arcsin (Real.sin α) = α) ∧
      (Real.arcsin (Real.sin (3 * α)) = r * α) ∧
      (Real.arcsin (Real.sin (8 * α)) = r^2 * α) ∧
      (Real.arcsin (Real.sin (t' * α)) = r^3 * α)) →
    t ≤ t') ∧
  t = 16 * Real.sqrt 6 / 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_arcsin_least_t_l2321_232170


namespace NUMINAMATH_CALUDE_complement_of_70_degrees_l2321_232172

theorem complement_of_70_degrees :
  let given_angle : ℝ := 70
  let complement_sum : ℝ := 90
  let complement_angle : ℝ := complement_sum - given_angle
  complement_angle = 20 := by sorry

end NUMINAMATH_CALUDE_complement_of_70_degrees_l2321_232172


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l2321_232138

theorem gcd_lcm_product_24_36 : Nat.gcd 24 36 * Nat.lcm 24 36 = 864 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l2321_232138


namespace NUMINAMATH_CALUDE_choir_size_after_new_members_l2321_232101

theorem choir_size_after_new_members (original : Nat) (new : Nat) : 
  original = 36 → new = 9 → original + new = 45 := by
  sorry

end NUMINAMATH_CALUDE_choir_size_after_new_members_l2321_232101


namespace NUMINAMATH_CALUDE_tree_height_after_two_years_l2321_232114

/-- Given a tree that triples its height every year and reaches 81 feet after 4 years,
    this function calculates its height after a given number of years. -/
def tree_height (years : ℕ) : ℚ :=
  81 / (3 ^ (4 - years))

/-- Theorem stating that the height of the tree after 2 years is 9 feet. -/
theorem tree_height_after_two_years :
  tree_height 2 = 9 := by sorry

end NUMINAMATH_CALUDE_tree_height_after_two_years_l2321_232114


namespace NUMINAMATH_CALUDE_pyramid_sphere_inequality_l2321_232127

/-- A regular quadrilateral pyramid -/
structure RegularQuadrilateralPyramid where
  /-- The radius of the circumscribed sphere -/
  R : ℝ
  /-- The radius of the inscribed sphere -/
  r : ℝ
  /-- R is positive -/
  R_pos : 0 < R
  /-- r is positive -/
  r_pos : 0 < r

/-- 
For a regular quadrilateral pyramid inscribed in a sphere with radius R 
and circumscribed around a sphere with radius r, R ≥ (√2 + 1)r holds.
-/
theorem pyramid_sphere_inequality (p : RegularQuadrilateralPyramid) : 
  p.R ≥ (Real.sqrt 2 + 1) * p.r := by
  sorry

end NUMINAMATH_CALUDE_pyramid_sphere_inequality_l2321_232127


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_to_9000_l2321_232169

theorem last_four_digits_of_5_to_9000 (h : 5^300 ≡ 1 [ZMOD 1250]) :
  5^9000 ≡ 1 [ZMOD 1250] := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_of_5_to_9000_l2321_232169


namespace NUMINAMATH_CALUDE_return_speed_calculation_l2321_232130

/-- Proves that the return speed is 100/3 mph given the conditions of the problem -/
theorem return_speed_calculation (distance : ℝ) (outbound_speed : ℝ) (average_speed : ℝ) 
  (h1 : distance = 150)
  (h2 : outbound_speed = 50)
  (h3 : average_speed = 40) :
  let return_speed := (2 * distance * outbound_speed) / (2 * distance - average_speed * (distance / outbound_speed))
  return_speed = 100 / 3 := by
sorry

#eval (2 * 150 * 50) / (2 * 150 - 40 * (150 / 50))

end NUMINAMATH_CALUDE_return_speed_calculation_l2321_232130


namespace NUMINAMATH_CALUDE_min_value_f_in_interval_l2321_232128

def f (x : ℝ) : ℝ := x^4 - 4*x + 3

theorem min_value_f_in_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) 3 ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-2 : ℝ) 3 → f x ≤ f y) ∧
  f x = 0 :=
sorry

end NUMINAMATH_CALUDE_min_value_f_in_interval_l2321_232128


namespace NUMINAMATH_CALUDE_a_2007_equals_4_l2321_232140

def f : ℕ → ℕ
  | 1 => 4
  | 2 => 1
  | 3 => 3
  | 4 => 5
  | 5 => 2
  | _ => 0

def a : ℕ → ℕ
  | 0 => 5
  | n + 1 => f (a n)

theorem a_2007_equals_4 : a 2007 = 4 := by
  sorry

end NUMINAMATH_CALUDE_a_2007_equals_4_l2321_232140


namespace NUMINAMATH_CALUDE_five_people_arrangement_l2321_232121

/-- The number of arrangements of n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of arrangements of n people in a row where two specific people are next to each other -/
def adjacentArrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 1)

/-- The number of arrangements of n people in a row where two specific people are not next to each other -/
def nonAdjacentArrangements (n : ℕ) : ℕ := totalArrangements n - adjacentArrangements n

theorem five_people_arrangement :
  nonAdjacentArrangements 5 = 72 :=
by sorry

end NUMINAMATH_CALUDE_five_people_arrangement_l2321_232121


namespace NUMINAMATH_CALUDE_birds_on_fence_l2321_232137

theorem birds_on_fence : ∃ x : ℕ, (2 * x + 10 = 50) ∧ (x = 20) :=
by sorry

end NUMINAMATH_CALUDE_birds_on_fence_l2321_232137


namespace NUMINAMATH_CALUDE_roots_sum_abs_l2321_232147

theorem roots_sum_abs (a b c m : ℤ) : 
  (∀ x : ℤ, x^3 - 2023*x + m = 0 ↔ x = a ∨ x = b ∨ x = c) →
  abs a + abs b + abs c = 94 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_abs_l2321_232147


namespace NUMINAMATH_CALUDE_investment_roi_difference_l2321_232117

def emma_investment : ℝ := 300
def briana_investment : ℝ := 500
def emma_yield_rate : ℝ := 0.15
def briana_yield_rate : ℝ := 0.10
def time_period : ℕ := 2

theorem investment_roi_difference :
  briana_investment * briana_yield_rate * time_period - 
  emma_investment * emma_yield_rate * time_period = 10 := by
  sorry

end NUMINAMATH_CALUDE_investment_roi_difference_l2321_232117


namespace NUMINAMATH_CALUDE_probability_of_letter_in_mathematics_l2321_232144

def alphabet : Finset Char := sorry

def mathematics : Finset Char := sorry

theorem probability_of_letter_in_mathematics :
  (mathematics.card : ℚ) / (alphabet.card : ℚ) = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_letter_in_mathematics_l2321_232144


namespace NUMINAMATH_CALUDE_convex_polygon_division_theorem_l2321_232162

-- Define a type for polygons
def Polygon : Type := Set (ℝ × ℝ)

-- Define a type for motions (transformations)
def Motion : Type := (ℝ × ℝ) → (ℝ × ℝ)

-- Define a predicate for convex polygons
def IsConvex (p : Polygon) : Prop := sorry

-- Define a predicate for orientation-preserving motions
def IsOrientationPreserving (m : Motion) : Prop := sorry

-- Define a predicate for a polygon being dividable by a broken line into two polygons
def DividableByBrokenLine (p : Polygon) (p1 p2 : Polygon) : Prop := sorry

-- Define a predicate for a polygon being dividable by a segment into two polygons
def DividableBySegment (p : Polygon) (p1 p2 : Polygon) : Prop := sorry

-- Define a predicate for two polygons being transformable into each other by a motion
def Transformable (p1 p2 : Polygon) (m : Motion) : Prop := sorry

-- State the theorem
theorem convex_polygon_division_theorem (p : Polygon) :
  IsConvex p →
  (∃ (p1 p2 : Polygon) (m : Motion), 
    DividableByBrokenLine p p1 p2 ∧ 
    IsOrientationPreserving m ∧ 
    Transformable p1 p2 m) →
  (∃ (q1 q2 : Polygon) (n : Motion), 
    DividableBySegment p q1 q2 ∧ 
    IsOrientationPreserving n ∧ 
    Transformable q1 q2 n) :=
by sorry

end NUMINAMATH_CALUDE_convex_polygon_division_theorem_l2321_232162


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2321_232197

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, a ∈ Set.Iic 0 → ∃ y : ℝ, y^2 - y + a ≤ 0) ∧
  (∃ b : ℝ, b ∉ Set.Iic 0 ∧ ∃ z : ℝ, z^2 - z + b ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2321_232197


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2321_232180

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 10) : x^3 + 1/x^3 = 970 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2321_232180


namespace NUMINAMATH_CALUDE_sphere_volume_to_surface_area_l2321_232196

theorem sphere_volume_to_surface_area :
  ∀ (r : ℝ), 
    (4 / 3 * π * r^3 = 32 * π / 3) →
    (4 * π * r^2 = 16 * π) :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_to_surface_area_l2321_232196


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l2321_232152

theorem complex_expression_simplification :
  (0.7264 * 0.4329 * 0.5478) + (0.1235 * 0.3412 * 0.6214) - 
  (0.1289 * 0.5634 * 0.3921) / (0.3785 * 0.4979 * 0.2884) - 
  (0.2956 * 0.3412 * 0.6573) = -0.3902 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l2321_232152


namespace NUMINAMATH_CALUDE_area_of_S_l2321_232151

-- Define the set S
def S : Set (ℝ × ℝ) := {(a, b) | ∀ x, x^2 + 2*b*x + 1 ≠ 2*a*(x + b)}

-- State the theorem
theorem area_of_S : MeasureTheory.volume S = π := by sorry

end NUMINAMATH_CALUDE_area_of_S_l2321_232151


namespace NUMINAMATH_CALUDE_rectangle_perimeter_relation_l2321_232185

/-- Given a figure divided into equal squares, this theorem proves the relationship
    between the perimeters of two rectangles formed by these squares. -/
theorem rectangle_perimeter_relation (square_side : ℝ) 
  (h1 : square_side > 0)
  (h2 : 3 * square_side * 2 + 2 * square_side = 112) : 
  4 * square_side * 2 + 2 * square_side = 140 := by
  sorry

#check rectangle_perimeter_relation

end NUMINAMATH_CALUDE_rectangle_perimeter_relation_l2321_232185


namespace NUMINAMATH_CALUDE_jackson_meat_problem_l2321_232131

theorem jackson_meat_problem (M : ℝ) : 
  M > 0 → 
  M - (1/4 * M) - 3 = 12 → 
  M = 20 :=
by sorry

end NUMINAMATH_CALUDE_jackson_meat_problem_l2321_232131


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2321_232120

/-- A line passing through (-1, 2) and perpendicular to 2x - 3y + 4 = 0 has the equation 3x + 2y - 1 = 0 -/
theorem perpendicular_line_equation :
  ∀ (l : Set (ℝ × ℝ)),
  ((-1, 2) ∈ l) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ ∃ (t : ℝ), x = -1 + 3*t ∧ y = 2 - 2*t) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ 3*x + 2*y - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2321_232120


namespace NUMINAMATH_CALUDE_cylinder_height_l2321_232108

/-- Represents a right cylinder with given dimensions -/
structure RightCylinder where
  radius : ℝ
  height : ℝ
  lateralSurfaceArea : ℝ
  endArea : ℝ

/-- Theorem stating the height of a specific cylinder -/
theorem cylinder_height (c : RightCylinder) 
  (h_radius : c.radius = 2)
  (h_lsa : c.lateralSurfaceArea = 16 * Real.pi)
  (h_ea : c.endArea = 8 * Real.pi) :
  c.height = 4 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_l2321_232108


namespace NUMINAMATH_CALUDE_cylinder_volume_relation_l2321_232166

/-- Given two cylinders A and B, where A's radius is r and height is h,
    B's height is r and radius is h, and A's volume is twice B's volume,
    prove that A's volume can be expressed as 4π h^3. -/
theorem cylinder_volume_relation (r h : ℝ) (h_pos : h > 0) :
  let volume_A := π * r^2 * h
  let volume_B := π * h^2 * r
  volume_A = 2 * volume_B → r = 2 * h → volume_A = 4 * π * h^3 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_relation_l2321_232166


namespace NUMINAMATH_CALUDE_recommended_sleep_hours_l2321_232182

theorem recommended_sleep_hours (total_sleep : ℝ) (short_sleep : ℝ) (short_days : ℕ) 
  (normal_days : ℕ) (normal_sleep_percentage : ℝ) 
  (h1 : total_sleep = 30)
  (h2 : short_sleep = 3)
  (h3 : short_days = 2)
  (h4 : normal_days = 5)
  (h5 : normal_sleep_percentage = 0.6)
  (h6 : total_sleep = short_sleep * short_days + normal_sleep_percentage * normal_days * recommended_sleep) :
  recommended_sleep = 8 := by
  sorry

end NUMINAMATH_CALUDE_recommended_sleep_hours_l2321_232182


namespace NUMINAMATH_CALUDE_max_profit_at_three_l2321_232155

/-- Represents the annual operating cost for a given year -/
def annual_cost (n : ℕ) : ℚ := 2 * n

/-- Represents the total operating cost for n years -/
def total_cost (n : ℕ) : ℚ := n^2 + n

/-- Represents the annual operating income -/
def annual_income : ℚ := 11

/-- Represents the initial cost of the car -/
def initial_cost : ℚ := 9

/-- Represents the annual average profit for n years -/
def annual_average_profit (n : ℕ+) : ℚ := 
  annual_income - (total_cost n + initial_cost) / n

/-- Theorem stating that the annual average profit is maximized when n = 3 -/
theorem max_profit_at_three : 
  ∀ (m : ℕ+), annual_average_profit 3 ≥ annual_average_profit m :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_three_l2321_232155


namespace NUMINAMATH_CALUDE_jeff_matches_won_l2321_232167

/-- Represents the duration of the tennis competition in minutes -/
def total_playtime : ℕ := 225

/-- Represents the time in minutes it takes Jeff to score a point -/
def minutes_per_point : ℕ := 7

/-- Represents the minimum number of points required to win a match -/
def points_to_win : ℕ := 12

/-- Represents the break time in minutes between matches -/
def break_time : ℕ := 5

/-- Calculates the total number of points Jeff scored during the competition -/
def total_points : ℕ := total_playtime / minutes_per_point

/-- Calculates the duration of a single match in minutes, including playtime and break time -/
def match_duration : ℕ := points_to_win * minutes_per_point + break_time

/-- Represents the number of matches Jeff won during the competition -/
def matches_won : ℕ := total_playtime / match_duration

theorem jeff_matches_won : matches_won = 2 := by sorry

end NUMINAMATH_CALUDE_jeff_matches_won_l2321_232167


namespace NUMINAMATH_CALUDE_puppy_discount_percentage_l2321_232143

/-- Calculates the discount percentage given the total cost before discount and the amount spent after discount -/
def discount_percentage (total_cost : ℚ) (amount_spent : ℚ) : ℚ :=
  (total_cost - amount_spent) / total_cost * 100

/-- Proves that the new-customer discount percentage is 20% for Julia's puppy purchases -/
theorem puppy_discount_percentage :
  let adoption_fee : ℚ := 20
  let dog_food : ℚ := 20
  let treats : ℚ := 2 * 2.5
  let toys : ℚ := 15
  let crate : ℚ := 20
  let bed : ℚ := 20
  let collar_leash : ℚ := 15
  let total_cost : ℚ := dog_food + treats + toys + crate + bed + collar_leash
  let total_spent : ℚ := 96
  let store_spent : ℚ := total_spent - adoption_fee
  discount_percentage total_cost store_spent = 20 := by
sorry

#eval discount_percentage 95 76

end NUMINAMATH_CALUDE_puppy_discount_percentage_l2321_232143


namespace NUMINAMATH_CALUDE_toothpick_grid_theorem_l2321_232149

/-- Calculates the number of unique toothpicks in a rectangular grid frame. -/
def unique_toothpicks (height width : ℕ) : ℕ :=
  let horizontal_toothpicks := (height + 1) * width
  let vertical_toothpicks := height * (width + 1)
  let intersections := (height + 1) * (width + 1)
  horizontal_toothpicks + vertical_toothpicks - intersections

/-- Theorem stating that a 15x8 toothpick grid uses 119 unique toothpicks. -/
theorem toothpick_grid_theorem :
  unique_toothpicks 15 8 = 119 := by
  sorry

#eval unique_toothpicks 15 8

end NUMINAMATH_CALUDE_toothpick_grid_theorem_l2321_232149


namespace NUMINAMATH_CALUDE_only_expr4_is_equation_l2321_232142

-- Define the four expressions
def expr1 : ℝ → Prop := λ x ↦ 3 + x < 1
def expr2 : ℝ → ℝ := λ x ↦ x - 67 + 63
def expr3 : ℝ → ℝ := λ x ↦ 4.8 + x
def expr4 : ℝ → Prop := λ x ↦ x + 0.7 = 12

-- Theorem stating that only expr4 is an equation
theorem only_expr4_is_equation :
  (∃ (x : ℝ), expr4 x) ∧
  (¬∃ (x : ℝ), expr1 x = (3 + x < 1)) ∧
  (∀ (x : ℝ), ¬∃ (y : ℝ), expr2 x = y) ∧
  (∀ (x : ℝ), ¬∃ (y : ℝ), expr3 x = y) :=
sorry

end NUMINAMATH_CALUDE_only_expr4_is_equation_l2321_232142


namespace NUMINAMATH_CALUDE_perpendicular_unit_vector_to_BC_l2321_232107

def AB : ℝ × ℝ := (-1, 3)
def BC : ℝ → ℝ × ℝ := λ k => (3, k)
def CD : ℝ → ℝ × ℝ := λ k => (k, 2)

def AC (k : ℝ) : ℝ × ℝ := (AB.1 + (BC k).1, AB.2 + (BC k).2)

def parallel (v w : ℝ × ℝ) : Prop := v.1 * w.2 = v.2 * w.1

def perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

def is_unit_vector (v : ℝ × ℝ) : Prop := v.1^2 + v.2^2 = 1

theorem perpendicular_unit_vector_to_BC (k : ℝ) :
  parallel (AC k) (CD k) →
  ∃ v : ℝ × ℝ, perpendicular v (BC k) ∧ is_unit_vector v ∧
    (v = (Real.sqrt 10 / 10, -3 * Real.sqrt 10 / 10) ∨
     v = (-Real.sqrt 10 / 10, 3 * Real.sqrt 10 / 10)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_unit_vector_to_BC_l2321_232107


namespace NUMINAMATH_CALUDE_smallest_divisible_n_l2321_232173

theorem smallest_divisible_n : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m^3 % 450 = 0 ∧ m^4 % 2560 = 0 → n ≤ m) ∧
  n^3 % 450 = 0 ∧ n^4 % 2560 = 0 ∧ n = 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_n_l2321_232173


namespace NUMINAMATH_CALUDE_stratified_sample_medium_stores_l2321_232133

/-- Given a population of stores with a known number of medium-sized stores,
    calculate the number of medium-sized stores in a stratified sample. -/
theorem stratified_sample_medium_stores
  (total_stores : ℕ)
  (medium_stores : ℕ)
  (sample_size : ℕ)
  (h1 : total_stores = 300)
  (h2 : medium_stores = 75)
  (h3 : sample_size = 20) :
  (medium_stores : ℚ) / total_stores * sample_size = 5 := by
sorry

end NUMINAMATH_CALUDE_stratified_sample_medium_stores_l2321_232133


namespace NUMINAMATH_CALUDE_math_homework_pages_l2321_232110

-- Define the variables
def reading_pages : ℕ := 4
def problems_per_page : ℕ := 3
def total_problems : ℕ := 30

-- Define the theorem
theorem math_homework_pages :
  ∃ (math_pages : ℕ), 
    math_pages * problems_per_page + reading_pages * problems_per_page = total_problems ∧
    math_pages = 6 := by
  sorry

end NUMINAMATH_CALUDE_math_homework_pages_l2321_232110


namespace NUMINAMATH_CALUDE_equal_angle_implies_equal_side_l2321_232183

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Represents the orthocenter of a triangle -/
def orthocenter (t : Triangle) : Point :=
  sorry

/-- Reflects a point with respect to a line segment -/
def reflect (p : Point) (a : Point) (b : Point) : Point :=
  sorry

/-- Checks if two triangles have an equal angle -/
def have_equal_angle (t1 t2 : Triangle) : Prop :=
  sorry

/-- Checks if two triangles have an equal side -/
def have_equal_side (t1 t2 : Triangle) : Prop :=
  sorry

/-- Checks if a triangle is acute -/
def is_acute (t : Triangle) : Prop :=
  sorry

theorem equal_angle_implies_equal_side 
  (ABC : Triangle) 
  (h_acute : is_acute ABC) 
  (H : Point) 
  (h_ortho : H = orthocenter ABC) 
  (A' B' C' : Point) 
  (h_A' : A' = reflect H ABC.B ABC.C) 
  (h_B' : B' = reflect H ABC.C ABC.A) 
  (h_C' : C' = reflect H ABC.A ABC.B) 
  (A'B'C' : Triangle) 
  (h_A'B'C' : A'B'C' = Triangle.mk A' B' C') 
  (h_equal_angle : have_equal_angle ABC A'B'C') :
  have_equal_side ABC A'B'C' :=
sorry

end NUMINAMATH_CALUDE_equal_angle_implies_equal_side_l2321_232183


namespace NUMINAMATH_CALUDE_second_derivative_zero_l2321_232113

open Real

/-- Given a differentiable function f and a point x₀ such that 
    the limit of (f(x₀) - f(x₀ + 2Δx)) / Δx as Δx approaches 0 is 2,
    prove that the second derivative of f at x₀ is 0. -/
theorem second_derivative_zero (f : ℝ → ℝ) (x₀ : ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_limit : ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |((f x₀ - f (x₀ + 2*Δx)) / Δx) - 2| < ε) :
  deriv (deriv f) x₀ = 0 := by
  sorry

end NUMINAMATH_CALUDE_second_derivative_zero_l2321_232113


namespace NUMINAMATH_CALUDE_same_number_of_heads_probability_p_plus_q_l2321_232174

-- Define the probability of heads for a fair coin
def fair_coin_prob : ℚ := 1/2

-- Define the probability of heads for the biased coin
def biased_coin_prob : ℚ := 2/5

-- Define the function to calculate the probability of getting k heads when flipping both coins
def prob_k_heads (k : ℕ) : ℚ :=
  match k with
  | 0 => (1 - fair_coin_prob) * (1 - biased_coin_prob)
  | 1 => fair_coin_prob * (1 - biased_coin_prob) + (1 - fair_coin_prob) * biased_coin_prob
  | 2 => fair_coin_prob * biased_coin_prob
  | _ => 0

-- State the theorem
theorem same_number_of_heads_probability :
  (prob_k_heads 0)^2 + (prob_k_heads 1)^2 + (prob_k_heads 2)^2 = 19/50 := by
  sorry

-- Define p and q
def p : ℕ := 19
def q : ℕ := 50

-- State the theorem for p + q
theorem p_plus_q : p + q = 69 := by
  sorry

end NUMINAMATH_CALUDE_same_number_of_heads_probability_p_plus_q_l2321_232174


namespace NUMINAMATH_CALUDE_tangent_circles_theorem_l2321_232178

/-- Given two circles with centers E and F tangent to segment BD and semicircles with diameters AB, BC, and AC,
    where r1, r2, and r are the radii of semicircles with diameters AB, BC, and AC respectively,
    and l1 and l2 are the radii of circles with centers E and F respectively. -/
theorem tangent_circles_theorem 
  (r1 r2 r l1 l2 : ℝ) 
  (h_r : r = r1 + r2) 
  (h_positive : r1 > 0 ∧ r2 > 0 ∧ l1 > 0 ∧ l2 > 0) :
  (∃ (distance_E_to_AC : ℝ), distance_E_to_AC = Real.sqrt ((r1 + l1)^2 - (r1 - l1)^2)) ∧ 
  l1 = (r1 * r2) / (r1 + r2) := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_theorem_l2321_232178


namespace NUMINAMATH_CALUDE_molecular_weight_constant_l2321_232194

-- Define the molecular weight of Aluminum carbonate
def aluminum_carbonate_mw : ℝ := 233.99

-- Define temperature and pressure
def temperature : ℝ := 298
def pressure : ℝ := 1

-- Define compressibility and thermal expansion coefficients
-- (We don't use these in the theorem, but they're mentioned in the problem)
def compressibility : ℝ := sorry
def thermal_expansion : ℝ := sorry

-- Theorem stating that the molecular weight remains constant
theorem molecular_weight_constant (T P : ℝ) :
  aluminum_carbonate_mw = 233.99 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_constant_l2321_232194


namespace NUMINAMATH_CALUDE_smaller_number_problem_l2321_232176

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 22) (h2 : x - y = 16) : 
  min x y = 3 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l2321_232176


namespace NUMINAMATH_CALUDE_simplify_expression_1_expand_expression_2_simplify_expression_3_l2321_232157

-- Part 1
theorem simplify_expression_1 (m n : ℝ) :
  15 * m * n^2 + 5 * m * n * m^3 * n = 15 * m * n^2 + 5 * m^4 * n^2 := by sorry

-- Part 2
theorem expand_expression_2 (x : ℝ) :
  (3 * x + 1) * (2 * x - 5) = 6 * x^2 - 13 * x - 5 := by sorry

-- Part 3
theorem simplify_expression_3 :
  (-0.25)^2024 * 4^2023 = 0.25 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_expand_expression_2_simplify_expression_3_l2321_232157


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_divisible_by_eight_l2321_232188

theorem consecutive_even_numbers_divisible_by_eight (n : ℤ) : 
  ∃ k : ℤ, 4 * n * (n + 1) = 8 * k := by
sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_divisible_by_eight_l2321_232188


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_h_l2321_232179

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the vertex form
def vertex_form (n h k : ℝ) (x : ℝ) : ℝ := n * (x - h)^2 + k

-- Theorem statement
theorem quadratic_vertex_form_h (a b c : ℝ) :
  (∃ n k : ℝ, ∀ x : ℝ, 4 * f a b c x = vertex_form n 3 k x) →
  (∀ x : ℝ, f a b c x = 3 * (x - 3)^2 + 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_h_l2321_232179


namespace NUMINAMATH_CALUDE_symmetric_circle_correct_symmetric_circle_equation_l2321_232129

/-- Represents a circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Defines symmetry with respect to the x-axis -/
def symmetricXAxis (c : Circle) : Circle :=
  { center := (c.center.1, -c.center.2), radius := c.radius }

/-- The original circle -/
def originalCircle : Circle :=
  { center := (-2, -1), radius := 2 }

/-- The symmetric circle -/
def symmetricCircle : Circle :=
  { center := (-2, 1), radius := 2 }

/-- Theorem stating that symmetricCircle is the result of applying x-axis symmetry to originalCircle -/
theorem symmetric_circle_correct : 
  symmetricXAxis originalCircle = symmetricCircle := by
  sorry

/-- Function to generate the equation of a circle -/
def circleEquation (c : Circle) : ℝ → ℝ → Prop :=
  fun x y => (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- Theorem stating that the equation of the symmetric circle is (x+2)^2 + (y-1)^2 = 4 -/
theorem symmetric_circle_equation :
  circleEquation symmetricCircle = fun x y => (x + 2)^2 + (y - 1)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_circle_correct_symmetric_circle_equation_l2321_232129


namespace NUMINAMATH_CALUDE_ratio_pq_qr_l2321_232116

-- Define the circle and points
def Circle : Type := Unit
def Point : Type := Unit

-- Define the radius of the circle
def radius : ℝ := 2

-- Define the points P, Q, and R on the circle
def P : Point := sorry
def Q : Point := sorry
def R : Point := sorry

-- Define the distance between two points
def distance : Point → Point → ℝ := sorry

-- Define the length of an arc
def arcLength : Point → Point → ℝ := sorry

-- State the theorem
theorem ratio_pq_qr (h1 : distance P Q = distance P R)
                    (h2 : distance P Q > radius)
                    (h3 : arcLength Q R = 2 * Real.pi) :
  distance P Q / arcLength Q R = 2 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ratio_pq_qr_l2321_232116


namespace NUMINAMATH_CALUDE_gumballs_last_42_days_l2321_232132

/-- The number of gumballs Kim gets for each pair of earrings -/
def gumballs_per_pair : ℕ := 9

/-- The number of pairs of earrings Kim brings on day 1 -/
def earrings_day1 : ℕ := 3

/-- The number of pairs of earrings Kim brings on day 2 -/
def earrings_day2 : ℕ := 2 * earrings_day1

/-- The number of pairs of earrings Kim brings on day 3 -/
def earrings_day3 : ℕ := earrings_day2 - 1

/-- The number of gumballs Kim eats per day -/
def gumballs_eaten_per_day : ℕ := 3

/-- The total number of gumballs Kim receives -/
def total_gumballs : ℕ := 
  gumballs_per_pair * (earrings_day1 + earrings_day2 + earrings_day3)

/-- The number of days the gumballs will last -/
def days_gumballs_last : ℕ := total_gumballs / gumballs_eaten_per_day

theorem gumballs_last_42_days : days_gumballs_last = 42 := by
  sorry

end NUMINAMATH_CALUDE_gumballs_last_42_days_l2321_232132


namespace NUMINAMATH_CALUDE_painter_rooms_problem_l2321_232100

theorem painter_rooms_problem (hours_per_room : ℕ) (rooms_painted : ℕ) (remaining_hours : ℕ) :
  hours_per_room = 7 →
  rooms_painted = 5 →
  remaining_hours = 49 →
  rooms_painted + remaining_hours / hours_per_room = 12 :=
by sorry

end NUMINAMATH_CALUDE_painter_rooms_problem_l2321_232100


namespace NUMINAMATH_CALUDE_money_left_after_shopping_l2321_232141

def bread_price : ℝ := 2
def butter_original_price : ℝ := 3
def butter_discount : ℝ := 0.1
def juice_price_multiplier : ℝ := 2
def cookies_original_price : ℝ := 4
def cookies_discount : ℝ := 0.2
def vat_rate : ℝ := 0.05
def initial_money : ℝ := 20

def calculate_discounted_price (original_price discount : ℝ) : ℝ :=
  original_price * (1 - discount)

def calculate_total_cost (bread butter juice cookies : ℝ) : ℝ :=
  bread + butter + juice + cookies

def apply_vat (total_cost vat_rate : ℝ) : ℝ :=
  total_cost * (1 + vat_rate)

theorem money_left_after_shopping :
  let butter_price := calculate_discounted_price butter_original_price butter_discount
  let cookies_price := calculate_discounted_price cookies_original_price cookies_discount
  let juice_price := bread_price * juice_price_multiplier
  let total_cost := calculate_total_cost bread_price butter_price juice_price cookies_price
  let final_cost := apply_vat total_cost vat_rate
  initial_money - final_cost = 7.5 := by sorry

end NUMINAMATH_CALUDE_money_left_after_shopping_l2321_232141


namespace NUMINAMATH_CALUDE_inner_polygon_perimeter_less_than_outer_l2321_232105

-- Define a type for convex polygons
structure ConvexPolygon where
  -- Add necessary fields (this is a simplified representation)
  perimeter : ℝ

-- Define a relation for one polygon being inside another
def IsInside (inner outer : ConvexPolygon) : Prop :=
  -- Add necessary conditions for one polygon being inside another
  sorry

-- Theorem statement
theorem inner_polygon_perimeter_less_than_outer
  (inner outer : ConvexPolygon)
  (h : IsInside inner outer) :
  inner.perimeter < outer.perimeter :=
sorry

end NUMINAMATH_CALUDE_inner_polygon_perimeter_less_than_outer_l2321_232105


namespace NUMINAMATH_CALUDE_max_hardcover_books_l2321_232161

/-- A number is composite if it has a factor other than 1 and itself -/
def IsComposite (n : ℕ) : Prop :=
  ∃ m, 1 < m ∧ m < n ∧ n % m = 0

/-- The problem statement -/
theorem max_hardcover_books :
  ∀ (hardcover paperback : ℕ),
  hardcover + paperback = 36 →
  IsComposite (paperback - hardcover) →
  hardcover ≤ 16 :=
by sorry

end NUMINAMATH_CALUDE_max_hardcover_books_l2321_232161


namespace NUMINAMATH_CALUDE_equation_solution_l2321_232177

theorem equation_solution : 
  ∃! x : ℝ, x ≠ 2 ∧ x - 5 = (3 * |x - 2|) / (x - 2) ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2321_232177


namespace NUMINAMATH_CALUDE_rectangular_parallelepiped_volume_l2321_232168

/-- The volume of a rectangular parallelepiped -/
def volume (width length height : ℝ) : ℝ := width * length * height

/-- Theorem: The volume of a rectangular parallelepiped with width 15 cm, length 6 cm, and height 4 cm is 360 cubic centimeters -/
theorem rectangular_parallelepiped_volume :
  volume 15 6 4 = 360 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_parallelepiped_volume_l2321_232168


namespace NUMINAMATH_CALUDE_intersection_point_parametric_equation_l2321_232160

/-- Given a triangle ABC with points D and E such that:
    - D lies on BC extended past C with BD:DC = 2:1
    - E lies on AC with AE:EC = 2:1
    - P is the intersection of BE and AD
    This theorem proves that P can be expressed as (1/7)A + (2/7)B + (4/7)C -/
theorem intersection_point_parametric_equation 
  (A B C D E P : ℝ × ℝ) : 
  (∃ t : ℝ, D = (1 - t) • B + t • C ∧ t = 2/3) →
  (∃ s : ℝ, E = (1 - s) • A + s • C ∧ s = 2/3) →
  (∃ u v : ℝ, P = (1 - u) • A + u • D ∧ P = (1 - v) • B + v • E) →
  P = (1/7) • A + (2/7) • B + (4/7) • C :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_parametric_equation_l2321_232160


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l2321_232158

/-- Given two lines L1 and L2, returns true if they are parallel but not coincident -/
def are_parallel_not_coincident (L1 L2 : ℝ → ℝ → Prop) : Prop :=
  (∃ k : ℝ, k ≠ 0 ∧ ∀ x y, L1 x y ↔ L2 (k * x) (k * y)) ∧
  ¬(∀ x y, L1 x y ↔ L2 x y)

/-- The first line: ax + 2y + 6 = 0 -/
def L1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 6 = 0

/-- The second line: x + (a - 1)y + (a^2 - 1) = 0 -/
def L2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y + (a^2 - 1) = 0

theorem parallel_lines_a_value :
  ∃ a : ℝ, are_parallel_not_coincident (L1 a) (L2 a) ∧ a = -1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l2321_232158


namespace NUMINAMATH_CALUDE_square_ratio_sum_l2321_232199

theorem square_ratio_sum (area_ratio : ℚ) (a b c : ℕ) : 
  area_ratio = 300 / 75 →
  (a : ℚ) * Real.sqrt b / c = Real.sqrt area_ratio →
  a + b + c = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_sum_l2321_232199


namespace NUMINAMATH_CALUDE_statement_d_is_incorrect_l2321_232171

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation for planes and lines
variable (perp : Plane → Plane → Prop)
variable (perpLine : Line → Plane → Prop)
variable (perpLines : Line → Line → Prop)

-- State the theorem
theorem statement_d_is_incorrect
  (α β : Plane) (l m n : Line)
  (h_diff_planes : α ≠ β)
  (h_diff_lines : l ≠ m ∧ m ≠ n ∧ l ≠ n)
  (h_perp_planes : perp α β)
  (h_perp_m_α : perpLine m α)
  (h_perp_n_β : perpLine n β) :
  ¬ (∀ m n, perpLines m n) :=
sorry

end NUMINAMATH_CALUDE_statement_d_is_incorrect_l2321_232171


namespace NUMINAMATH_CALUDE_total_balls_l2321_232198

theorem total_balls (S V B : ℕ) : 
  S = 68 ∧ 
  S = V - 12 ∧ 
  S = B + 23 → 
  S + V + B = 193 := by
sorry

end NUMINAMATH_CALUDE_total_balls_l2321_232198


namespace NUMINAMATH_CALUDE_sum_solution_equation_value_l2321_232191

/-- A sum solution equation is an equation of the form a/x = b where the solution for x is 1/(a+b) -/
def IsSumSolutionEquation (a b : ℚ) : Prop :=
  ∀ x, a / x = b ↔ x = 1 / (a + b)

/-- The main theorem: if n/x = 3-n is a sum solution equation, then n = 3/4 -/
theorem sum_solution_equation_value (n : ℚ) :
  IsSumSolutionEquation n (3 - n) → n = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_solution_equation_value_l2321_232191


namespace NUMINAMATH_CALUDE_decimal_expansion_irrational_l2321_232124

/-- Decimal expansion function -/
def decimal_expansion (f : ℕ → ℕ) : ℚ :=
  sorry

/-- Power function -/
def f (n : ℕ) (x : ℕ) : ℕ :=
  x^n

/-- Theorem: The decimal expansion α is irrational for all positive integers n -/
theorem decimal_expansion_irrational (n : ℕ) (h : n > 0) :
  ¬ ∃ (q : ℚ), q = decimal_expansion (f n) :=
sorry

end NUMINAMATH_CALUDE_decimal_expansion_irrational_l2321_232124


namespace NUMINAMATH_CALUDE_bill_difference_l2321_232126

theorem bill_difference : 
  ∀ (alice_bill bob_bill : ℝ),
  alice_bill * 0.25 = 5 →
  bob_bill * 0.10 = 4 →
  bob_bill - alice_bill = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_bill_difference_l2321_232126


namespace NUMINAMATH_CALUDE_unique_five_digit_number_l2321_232153

def is_valid_increment (n m : ℕ) : Prop :=
  ∃ (d₁ d₂ d₃ d₄ d₅ : ℕ),
    n = 10000 * d₁ + 1000 * d₂ + 100 * d₃ + 10 * d₄ + d₅ ∧
    m = 10000 * (d₁ + 2) + 1000 * (d₂ + 4) + 100 * (d₃ + 2) + 10 * (d₄ + 4) + (d₅ + 4) ∧
    d₁ < 10 ∧ d₂ < 10 ∧ d₃ < 10 ∧ d₄ < 10 ∧ d₅ < 10

theorem unique_five_digit_number :
  ∀ n : ℕ,
    10000 ≤ n ∧ n < 100000 →
    (∃ m : ℕ, is_valid_increment n m ∧ m = 4 * n) →
    n = 14074 :=
by sorry

end NUMINAMATH_CALUDE_unique_five_digit_number_l2321_232153


namespace NUMINAMATH_CALUDE_parabola_properties_l2321_232175

/-- A parabola with given properties -/
structure Parabola where
  vertex : ℝ × ℝ
  axis_vertical : Bool
  passing_point : ℝ × ℝ

/-- Shift vector -/
def shift_vector : ℝ × ℝ := (2, 3)

/-- Our specific parabola -/
def our_parabola : Parabola := {
  vertex := (3, -2),
  axis_vertical := true,
  passing_point := (5, 6)
}

/-- The equation of our parabola -/
def parabola_equation (x : ℝ) : ℝ := 2 * x^2 - 12 * x + 16

/-- The new vertex after shifting -/
def new_vertex : ℝ × ℝ := (5, 1)

theorem parabola_properties :
  (∀ x, parabola_equation x = 2 * (x - our_parabola.vertex.1)^2 + our_parabola.vertex.2) ∧
  parabola_equation our_parabola.passing_point.1 = our_parabola.passing_point.2 ∧
  new_vertex = (our_parabola.vertex.1 + shift_vector.1, our_parabola.vertex.2 + shift_vector.2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l2321_232175


namespace NUMINAMATH_CALUDE_nina_running_distance_l2321_232164

/-- Conversion factor from kilometers to miles -/
def km_to_miles : ℝ := 0.621371

/-- Conversion factor from yards to miles -/
def yard_to_miles : ℝ := 0.000568182

/-- Distance Nina ran in miles for her initial run -/
def initial_run : ℝ := 0.08

/-- Distance Nina ran in kilometers for her second run (done twice) -/
def second_run_km : ℝ := 3

/-- Distance Nina ran in yards for her third run -/
def third_run_yards : ℝ := 1200

/-- Distance Nina ran in kilometers for her final run -/
def final_run_km : ℝ := 6

/-- Total distance Nina ran in miles -/
def total_distance : ℝ := 
  initial_run + 
  2 * (second_run_km * km_to_miles) + 
  (third_run_yards * yard_to_miles) + 
  (final_run_km * km_to_miles)

theorem nina_running_distance : 
  ∃ ε > 0, |total_distance - 8.22| < ε :=
sorry

end NUMINAMATH_CALUDE_nina_running_distance_l2321_232164


namespace NUMINAMATH_CALUDE_unique_solution_l2321_232192

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def is_solution (a x y p : ℕ) : Prop :=
  is_single_digit a ∧ is_single_digit x ∧ is_single_digit y ∧ is_single_digit p ∧
  a ≠ x ∧ a ≠ y ∧ a ≠ p ∧ x ≠ y ∧ x ≠ p ∧ y ≠ p ∧
  10 * a + x + 10 * y + x = 100 * y + 10 * p + a

theorem unique_solution :
  ∀ a x y p : ℕ, is_solution a x y p → a = 8 ∧ x = 9 ∧ y = 1 ∧ p = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l2321_232192


namespace NUMINAMATH_CALUDE_unknown_number_value_l2321_232195

theorem unknown_number_value (y : ℝ) : (12 : ℝ)^3 * y^4 / 432 = 5184 → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_value_l2321_232195


namespace NUMINAMATH_CALUDE_circle_radius_l2321_232190

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 4*y + 16 = 0

-- State the theorem
theorem circle_radius : ∃ (h k r : ℝ), r = 2 ∧
  ∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_l2321_232190
