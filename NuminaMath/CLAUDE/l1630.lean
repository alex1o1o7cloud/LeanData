import Mathlib

namespace NUMINAMATH_CALUDE_candy_bar_problem_l1630_163004

theorem candy_bar_problem (fred : ℕ) (bob : ℕ) (jacqueline : ℕ) : 
  fred = 12 →
  bob = fred + 6 →
  jacqueline = 10 * (fred + bob) →
  (40 : ℚ) / 100 * jacqueline = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_bar_problem_l1630_163004


namespace NUMINAMATH_CALUDE_circle_line_distance_l1630_163046

theorem circle_line_distance (a : ℝ) : 
  let circle : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 4)^2 = 4}
  let line : Set (ℝ × ℝ) := {p | a * p.1 + p.2 - 1 = 0}
  let center : ℝ × ℝ := (1, 4)
  (∀ p ∈ line, ((p.1 - center.1)^2 + (p.2 - center.2)^2).sqrt ≥ 1) ∧
  (∃ p ∈ line, ((p.1 - center.1)^2 + (p.2 - center.2)^2).sqrt = 1) →
  a = -4/3 := by
sorry


end NUMINAMATH_CALUDE_circle_line_distance_l1630_163046


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1630_163067

theorem polynomial_factorization (a b : ℝ) :
  (a^2 + 10*a + 25) - b^2 = (a + 5 + b) * (a + 5 - b) := by
  sorry

#check polynomial_factorization

end NUMINAMATH_CALUDE_polynomial_factorization_l1630_163067


namespace NUMINAMATH_CALUDE_set_operations_l1630_163091

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | (x - 1) / (x - 6) < 0}

theorem set_operations :
  (A ∩ B = {x : ℝ | 2 ≤ x ∧ x < 6}) ∧
  (A ∪ B = {x : ℝ | 1 < x ∧ x ≤ 8}) ∧
  ((Aᶜ ∩ B) = {x : ℝ | 1 < x ∧ x < 2}) := by sorry

end NUMINAMATH_CALUDE_set_operations_l1630_163091


namespace NUMINAMATH_CALUDE_dog_paws_on_ground_l1630_163005

theorem dog_paws_on_ground (total_dogs : ℕ) (h1 : total_dogs = 12) : 
  (total_dogs / 2) * 2 + (total_dogs / 2) * 4 = 36 :=
by sorry

#check dog_paws_on_ground

end NUMINAMATH_CALUDE_dog_paws_on_ground_l1630_163005


namespace NUMINAMATH_CALUDE_unique_matching_number_l1630_163077

/-- A function that checks if two numbers match in exactly one digit position -/
def match_one_digit (a b : ℕ) : Prop :=
  ∃! i : Fin 3, (a / 10^i.val % 10) = (b / 10^i.val % 10)

/-- The theorem stating that 729 is the only three-digit number matching one digit with each guess -/
theorem unique_matching_number : 
  ∀ n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧ 
    match_one_digit n 109 ∧ 
    match_one_digit n 704 ∧ 
    match_one_digit n 124 
    → n = 729 := by
  sorry


end NUMINAMATH_CALUDE_unique_matching_number_l1630_163077


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1630_163061

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  arithmetic_sequence b →
  a 1 = 25 →
  b 1 = 75 →
  a 2 + b 2 = 100 →
  a 37 + b 37 = 100 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1630_163061


namespace NUMINAMATH_CALUDE_triangle_cosine_ratio_l1630_163038

/-- In any triangle ABC, (b * cos C + c * cos B) / a = 1 -/
theorem triangle_cosine_ratio (A B C a b c : ℝ) : 
  0 < a → 0 < b → 0 < c →
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  (b * Real.cos C + c * Real.cos B) / a = 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_cosine_ratio_l1630_163038


namespace NUMINAMATH_CALUDE_problem_solution_l1630_163033

theorem problem_solution :
  let x : ℝ := 88 + (4/3) * 88
  let y : ℝ := x + (3/5) * x
  let z : ℝ := (1/2) * (x + y)
  z = 266.9325 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1630_163033


namespace NUMINAMATH_CALUDE_meet_once_l1630_163088

/-- Represents the movement of Michael and the garbage truck --/
structure Movement where
  michael_speed : ℝ
  truck_speed : ℝ
  pail_distance : ℝ
  truck_stop_time : ℝ

/-- Calculates the number of meetings between Michael and the truck --/
def count_meetings (m : Movement) : ℕ :=
  sorry

/-- The specific scenario described in the problem --/
def problem_scenario : Movement where
  michael_speed := 4
  truck_speed := 12
  pail_distance := 200
  truck_stop_time := 20

/-- Theorem stating that Michael and the truck meet exactly once --/
theorem meet_once :
  count_meetings problem_scenario = 1 :=
sorry

end NUMINAMATH_CALUDE_meet_once_l1630_163088


namespace NUMINAMATH_CALUDE_smallest_positive_a_for_two_roots_in_unit_interval_l1630_163066

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Predicate to check if a quadratic function has two distinct roots in (0,1) -/
def has_two_distinct_roots_in_unit_interval (f : QuadraticFunction) : Prop :=
  ∃ (r s : ℝ), 0 < r ∧ r < 1 ∧ 0 < s ∧ s < 1 ∧ r ≠ s ∧
    f.a * r^2 + f.b * r + f.c = 0 ∧
    f.a * s^2 + f.b * s + f.c = 0

/-- The main theorem stating the smallest positive integer a -/
theorem smallest_positive_a_for_two_roots_in_unit_interval :
  ∃ (a : ℤ), a > 0 ∧
    (∀ (f : QuadraticFunction), f.a = a → has_two_distinct_roots_in_unit_interval f) ∧
    (∀ (a' : ℤ), 0 < a' → a' < a →
      ∃ (f : QuadraticFunction), f.a = a' ∧ ¬has_two_distinct_roots_in_unit_interval f) ∧
    a = 5 :=
  sorry

end NUMINAMATH_CALUDE_smallest_positive_a_for_two_roots_in_unit_interval_l1630_163066


namespace NUMINAMATH_CALUDE_equal_square_difference_subsequence_equal_square_difference_and_arithmetic_is_constant_l1630_163012

-- Define the property of being an "equal square difference sequence"
def is_equal_square_difference (a : ℕ → ℝ) : Prop :=
  ∃ p : ℝ, ∀ n : ℕ, n ≥ 2 → a n ^ 2 - a (n - 1) ^ 2 = p

-- Define the property of being an arithmetic sequence
def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

-- Theorem 1
theorem equal_square_difference_subsequence
  (a : ℕ → ℝ) (k : ℕ) (hk : k > 0) (ha : is_equal_square_difference a) :
  is_equal_square_difference (fun n ↦ a (k * n)) :=
sorry

-- Theorem 2
theorem equal_square_difference_and_arithmetic_is_constant
  (a : ℕ → ℝ) (ha1 : is_equal_square_difference a) (ha2 : is_arithmetic a) :
  ∃ c : ℝ, ∀ n : ℕ, a n = c :=
sorry

end NUMINAMATH_CALUDE_equal_square_difference_subsequence_equal_square_difference_and_arithmetic_is_constant_l1630_163012


namespace NUMINAMATH_CALUDE_largest_band_size_l1630_163014

theorem largest_band_size :
  ∀ m r x : ℕ,
  m = r * x + 3 →
  m = (r - 1) * (x + 2) →
  m < 100 →
  ∃ m_max : ℕ,
  m_max = 69 ∧
  ∀ m' : ℕ,
  (∃ r' x' : ℕ, m' = r' * x' + 3 ∧ m' = (r' - 1) * (x' + 2) ∧ m' < 100) →
  m' ≤ m_max :=
sorry

end NUMINAMATH_CALUDE_largest_band_size_l1630_163014


namespace NUMINAMATH_CALUDE_waiter_earnings_l1630_163074

def lunch_shift (total_customers : ℕ) (tipping_customers : ℕ) 
  (tip_8 : ℕ) (tip_10 : ℕ) (tip_12 : ℕ) (meal_cost : ℕ) : ℕ :=
  let total_tips := 8 * tip_8 + 10 * tip_10 + 12 * tip_12
  total_tips - meal_cost

theorem waiter_earnings : 
  lunch_shift 12 6 3 2 1 5 = 51 := by sorry

end NUMINAMATH_CALUDE_waiter_earnings_l1630_163074


namespace NUMINAMATH_CALUDE_quadratic_range_solution_set_l1630_163003

/-- A quadratic function with real coefficients -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

/-- The theorem stating the value of c given the conditions -/
theorem quadratic_range_solution_set (a b m : ℝ) :
  (∀ x, f a b x ≥ 0) →  -- range of f is [0, +∞)
  (∃ c, ∀ x, f a b x < c ↔ m < x ∧ x < m + 6) →  -- solution set of f(x) < c is (m, m+6)
  ∃ c, c = 9 := by sorry

end NUMINAMATH_CALUDE_quadratic_range_solution_set_l1630_163003


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1630_163016

theorem quadratic_inequality_equivalence :
  ∀ x : ℝ, x * (2 * x + 3) < 15 ↔ x ∈ Set.Ioo (-5/2) 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1630_163016


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l1630_163073

/-- Given a line l: 4x + 5y - 8 = 0 and a point A (3, 2), 
    the perpendicular line through A has the equation 4y - 5x + 7 = 0 -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let l : Set (ℝ × ℝ) := {(x, y) | 4 * x + 5 * y - 8 = 0}
  let A : ℝ × ℝ := (3, 2)
  let perpendicular_line : Set (ℝ × ℝ) := {(x, y) | 4 * y - 5 * x + 7 = 0}
  (∀ (p q : ℝ × ℝ), p ∈ l ∧ q ∈ l ∧ p ≠ q →
    (A.1 - p.1) * (q.1 - p.1) + (A.2 - p.2) * (q.2 - p.2) = 0) ∧
  A ∈ perpendicular_line :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l1630_163073


namespace NUMINAMATH_CALUDE_lamp_position_probability_l1630_163095

/-- The probability that a randomly chosen point on a line segment of length 6
    is at least 2 units away from both endpoints is 1/3. -/
theorem lamp_position_probability : 
  let total_length : ℝ := 6
  let min_distance : ℝ := 2
  let favorable_length : ℝ := total_length - 2 * min_distance
  favorable_length / total_length = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_lamp_position_probability_l1630_163095


namespace NUMINAMATH_CALUDE_prop_truth_values_l1630_163006

-- Define a structure for a line
structure Line where
  slope : ℝ
  x_intercept : ℝ
  y_intercept : ℝ

-- Define the original proposition
def original_prop (l : Line) : Prop :=
  l.slope = -1 → l.x_intercept = l.y_intercept

-- Define the converse
def converse_prop (l : Line) : Prop :=
  l.x_intercept = l.y_intercept → l.slope = -1

-- Define the inverse
def inverse_prop (l : Line) : Prop :=
  l.slope ≠ -1 → l.x_intercept ≠ l.y_intercept

-- Define the contrapositive
def contrapositive_prop (l : Line) : Prop :=
  l.x_intercept ≠ l.y_intercept → l.slope ≠ -1

-- Theorem stating the truth values of the propositions
theorem prop_truth_values :
  ∃ l : Line, original_prop l ∧
  ¬(∀ l : Line, converse_prop l) ∧
  ¬(∀ l : Line, inverse_prop l) ∧
  (∀ l : Line, contrapositive_prop l) :=
sorry

end NUMINAMATH_CALUDE_prop_truth_values_l1630_163006


namespace NUMINAMATH_CALUDE_sports_competition_results_l1630_163010

/-- Represents the outcome of a single event -/
inductive EventOutcome
| SchoolAWins
| SchoolBWins

/-- Represents the outcome of the entire championship -/
inductive ChampionshipOutcome
| SchoolAWins
| SchoolBWins

/-- The probability of School A winning each event -/
def probSchoolAWins : Fin 3 → ℝ
| 0 => 0.5
| 1 => 0.4
| 2 => 0.8

/-- The score awarded for winning an event -/
def winningScore : ℕ := 10

/-- Calculate the probability of School A winning the championship -/
def probSchoolAWinsChampionship : ℝ := sorry

/-- Calculate the expectation of School B's total score -/
def expectationSchoolBScore : ℝ := sorry

/-- Theorem stating the main results -/
theorem sports_competition_results :
  probSchoolAWinsChampionship = 0.6 ∧ expectationSchoolBScore = 13 := by sorry

end NUMINAMATH_CALUDE_sports_competition_results_l1630_163010


namespace NUMINAMATH_CALUDE_teddy_hamburger_count_l1630_163076

/-- The number of hamburgers Teddy bought -/
def teddy_hamburgers : ℕ := 5

/-- The total amount spent by Robert and Teddy -/
def total_spent : ℕ := 106

/-- The cost of a pizza box -/
def pizza_cost : ℕ := 10

/-- The cost of a soft drink -/
def drink_cost : ℕ := 2

/-- The cost of a hamburger -/
def hamburger_cost : ℕ := 3

/-- The number of pizza boxes Robert bought -/
def robert_pizza : ℕ := 5

/-- The number of soft drinks Robert bought -/
def robert_drinks : ℕ := 10

/-- The number of soft drinks Teddy bought -/
def teddy_drinks : ℕ := 10

theorem teddy_hamburger_count :
  total_spent = 
    robert_pizza * pizza_cost + 
    (robert_drinks + teddy_drinks) * drink_cost + 
    teddy_hamburgers * hamburger_cost := by
  sorry

end NUMINAMATH_CALUDE_teddy_hamburger_count_l1630_163076


namespace NUMINAMATH_CALUDE_credits_to_graduate_l1630_163064

/-- The number of semesters in college -/
def semesters : ℕ := 8

/-- The number of classes taken per semester -/
def classes_per_semester : ℕ := 5

/-- The number of credits per class -/
def credits_per_class : ℕ := 3

/-- The total number of credits needed to graduate -/
def total_credits : ℕ := semesters * classes_per_semester * credits_per_class

theorem credits_to_graduate : total_credits = 120 := by
  sorry

end NUMINAMATH_CALUDE_credits_to_graduate_l1630_163064


namespace NUMINAMATH_CALUDE_ratio_problem_l1630_163060

theorem ratio_problem (first_term second_term : ℚ) : 
  first_term = 15 → 
  first_term / second_term = 60 / 100 → 
  second_term = 25 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l1630_163060


namespace NUMINAMATH_CALUDE_paper_plates_and_cups_cost_l1630_163002

theorem paper_plates_and_cups_cost (plate_cost cup_cost : ℝ) : 
  100 * plate_cost + 200 * cup_cost = 7.5 → 
  20 * plate_cost + 40 * cup_cost = 1.5 := by
sorry

end NUMINAMATH_CALUDE_paper_plates_and_cups_cost_l1630_163002


namespace NUMINAMATH_CALUDE_multiplier_is_five_l1630_163096

/-- Given a number that equals some times the difference between itself and 4,
    prove that when the number is 5, the multiplier is also 5. -/
theorem multiplier_is_five (n m : ℝ) : n = m * (n - 4) → n = 5 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_is_five_l1630_163096


namespace NUMINAMATH_CALUDE_ceiling_sqrt_225_l1630_163079

theorem ceiling_sqrt_225 : ⌈Real.sqrt 225⌉ = 15 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_225_l1630_163079


namespace NUMINAMATH_CALUDE_alex_jimmy_yellow_ratio_l1630_163068

-- Define the number of marbles each person has
def lorin_black : ℕ := 4
def jimmy_yellow : ℕ := 22
def alex_total : ℕ := 19

-- Define Alex's black marbles as twice Lorin's
def alex_black : ℕ := 2 * lorin_black

-- Define Alex's yellow marbles
def alex_yellow : ℕ := alex_total - alex_black

-- Theorem to prove
theorem alex_jimmy_yellow_ratio :
  (alex_yellow : ℚ) / jimmy_yellow = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_alex_jimmy_yellow_ratio_l1630_163068


namespace NUMINAMATH_CALUDE_sin_cos_difference_equals_neg_sqrt_two_over_two_l1630_163019

theorem sin_cos_difference_equals_neg_sqrt_two_over_two :
  Real.sin (18 * π / 180) * Real.cos (63 * π / 180) -
  Real.sin (72 * π / 180) * Real.sin (117 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_equals_neg_sqrt_two_over_two_l1630_163019


namespace NUMINAMATH_CALUDE_tangent_line_problem_l1630_163034

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := (k + 1) * x^3 - 3 * (k + 2) * x^2 - k^2 - 2 * k

-- Define the derivative of f
def f' (k : ℝ) (x : ℝ) : ℝ := 3 * (k + 1) * x^2 - 6 * (k + 2) * x

theorem tangent_line_problem (k : ℝ) (h1 : k > -1) :
  (∀ x ∈ Set.Ioo 0 4, f' k x < 0) →
  (k = 0 ∧ 
   ∃ t : ℝ, t = f' 0 1 ∧ 9 * 1 + (-5) + 4 = 0 ∧ 
   ∀ x y : ℝ, y = t * (x - 1) + (-5) ↔ 9 * x + y + 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l1630_163034


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1630_163031

/-- Given a geometric sequence {a_n} where the sum of the first n terms is S_n = 2^n + r,
    prove that r = -1 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (r : ℝ) :
  (∀ n : ℕ, S n = 2^n + r) →
  (∀ n : ℕ, n ≥ 1 → a n = S n - S (n-1)) →
  (∀ n : ℕ, n ≥ 2 → a n = 2 * a (n-1)) →
  r = -1 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sum_l1630_163031


namespace NUMINAMATH_CALUDE_cos_negative_nineteen_pi_sixths_tan_x_given_sin_x_l1630_163085

-- Part I
theorem cos_negative_nineteen_pi_sixths :
  Real.cos (-19 * π / 6) = -Real.sqrt 3 / 2 := by sorry

-- Part II
theorem tan_x_given_sin_x (x : Real) :
  x ∈ Set.Icc (π / 2) (3 * π / 2) →
  Real.sin x = -3 / 5 →
  Real.tan x = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_cos_negative_nineteen_pi_sixths_tan_x_given_sin_x_l1630_163085


namespace NUMINAMATH_CALUDE_dice_roll_probability_l1630_163047

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The number of favorable outcomes on the first die (less than 3) -/
def favorableFirst : ℕ := 2

/-- The number of favorable outcomes on the second die (greater than 3) -/
def favorableSecond : ℕ := 3

/-- The probability of the desired outcome when rolling two dice -/
def probability : ℚ := (favorableFirst / numSides) * (favorableSecond / numSides)

theorem dice_roll_probability :
  probability = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l1630_163047


namespace NUMINAMATH_CALUDE_line_equation_through_point_l1630_163086

/-- The equation of a line with slope 2 passing through the point (2, 3) is 2x - y - 1 = 0 -/
theorem line_equation_through_point (x y : ℝ) :
  let slope : ℝ := 2
  let point : ℝ × ℝ := (2, 3)
  (y - point.2 = slope * (x - point.1)) ↔ (2 * x - y - 1 = 0) := by
sorry

end NUMINAMATH_CALUDE_line_equation_through_point_l1630_163086


namespace NUMINAMATH_CALUDE_triangle_area_l1630_163030

theorem triangle_area (a b : ℝ) (C : ℝ) (h1 : a = 7) (h2 : b = 11) (h3 : C = 60 * π / 180) :
  (1 / 2) * a * b * Real.sin C = (77 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1630_163030


namespace NUMINAMATH_CALUDE_total_vehicles_l1630_163048

theorem total_vehicles (lanes : Nat) (trucks_per_lane : Nat) : 
  lanes = 4 → 
  trucks_per_lane = 60 → 
  (lanes * trucks_per_lane * 2 + lanes * trucks_per_lane) = 2160 :=
by
  sorry

end NUMINAMATH_CALUDE_total_vehicles_l1630_163048


namespace NUMINAMATH_CALUDE_alice_bob_number_sum_l1630_163000

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem alice_bob_number_sum :
  ∀ (A B : ℕ),
  A ∈ Finset.range 50 →
  B ∈ Finset.range 50 →
  A ≠ B →
  A ≠ 1 →
  A ≠ 50 →
  is_prime B →
  (∃ k : ℕ, 120 * B + A = k * k) →
  A + B = 43 :=
by sorry

end NUMINAMATH_CALUDE_alice_bob_number_sum_l1630_163000


namespace NUMINAMATH_CALUDE_profit_calculation_l1630_163071

/-- The number of pencils purchased -/
def total_pencils : ℕ := 2000

/-- The purchase price per pencil in dollars -/
def purchase_price : ℚ := 1/5

/-- The selling price per pencil in dollars -/
def selling_price : ℚ := 2/5

/-- The desired profit in dollars -/
def desired_profit : ℚ := 160

/-- The number of pencils that must be sold to achieve the desired profit -/
def pencils_to_sell : ℕ := 1400

theorem profit_calculation :
  (pencils_to_sell : ℚ) * selling_price - (total_pencils : ℚ) * purchase_price = desired_profit :=
sorry

end NUMINAMATH_CALUDE_profit_calculation_l1630_163071


namespace NUMINAMATH_CALUDE_orchids_sold_correct_l1630_163008

/-- The number of orchids sold by a plant supplier -/
def orchids_sold : ℕ := 20

/-- The price of each orchid -/
def orchid_price : ℕ := 50

/-- The number of potted Chinese money plants sold -/
def money_plants_sold : ℕ := 15

/-- The price of each potted Chinese money plant -/
def money_plant_price : ℕ := 25

/-- The number of workers -/
def workers : ℕ := 2

/-- The wage paid to each worker -/
def worker_wage : ℕ := 40

/-- The cost of new pots -/
def new_pots_cost : ℕ := 150

/-- The amount left after all transactions -/
def amount_left : ℕ := 1145

/-- Theorem stating that the number of orchids sold is correct given the problem conditions -/
theorem orchids_sold_correct :
  orchids_sold * orchid_price + 
  money_plants_sold * money_plant_price - 
  (workers * worker_wage + new_pots_cost) = 
  amount_left := by sorry

end NUMINAMATH_CALUDE_orchids_sold_correct_l1630_163008


namespace NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l1630_163041

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem tenth_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = 5) :
  a 10 = 19 := by
sorry

end NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l1630_163041


namespace NUMINAMATH_CALUDE_soda_production_in_8_hours_l1630_163070

/-- Represents the production rate of a soda machine -/
structure SodaMachine where
  cans_per_interval : ℕ
  interval_minutes : ℕ

/-- Calculates the number of cans produced in a given number of hours -/
def cans_produced (machine : SodaMachine) (hours : ℕ) : ℕ :=
  let intervals_per_hour : ℕ := 60 / machine.interval_minutes
  let total_intervals : ℕ := hours * intervals_per_hour
  machine.cans_per_interval * total_intervals

theorem soda_production_in_8_hours (machine : SodaMachine)
    (h1 : machine.cans_per_interval = 30)
    (h2 : machine.interval_minutes = 30) :
    cans_produced machine 8 = 480 := by
  sorry

end NUMINAMATH_CALUDE_soda_production_in_8_hours_l1630_163070


namespace NUMINAMATH_CALUDE_michaels_dogs_l1630_163080

theorem michaels_dogs (num_cats : ℕ) (cost_per_animal : ℕ) (total_cost : ℕ) :
  num_cats = 2 →
  cost_per_animal = 13 →
  total_cost = 65 →
  ∃ num_dogs : ℕ, num_dogs = 3 ∧ total_cost = cost_per_animal * (num_cats + num_dogs) :=
by sorry

end NUMINAMATH_CALUDE_michaels_dogs_l1630_163080


namespace NUMINAMATH_CALUDE_curve_is_two_lines_l1630_163028

-- Define the equation of the curve
def curve_equation (x y : ℝ) : Prop := x^2 + x*y = x

-- Theorem stating that the curve equation represents two lines
theorem curve_is_two_lines :
  ∃ (m₁ b₁ m₂ b₂ : ℝ), 
    (∀ x y, curve_equation x y ↔ (y = m₁ * x + b₁ ∨ y = m₂ * x + b₂)) :=
sorry

end NUMINAMATH_CALUDE_curve_is_two_lines_l1630_163028


namespace NUMINAMATH_CALUDE_triangle_max_area_l1630_163040

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  (Real.sin A - Real.sin B) / Real.sin C = (c - b) / (2 + b) →
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  ∃ (area : ℝ), area ≤ Real.sqrt 3 ∧
    area = (1/2) * a * b * Real.sin C ∧
    ∀ (area' : ℝ), area' = (1/2) * a * b * Real.sin C → area' ≤ area :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l1630_163040


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l1630_163057

/-- An ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  left_focus : ℝ × ℝ
  right_vertex : ℝ × ℝ

/-- A line with a given slope -/
structure Line where
  slope : ℝ

/-- The standard form of an ellipse equation -/
def standard_equation (a b : ℝ) : (ℝ × ℝ) → Prop :=
  fun p => p.1^2 / (a^2) + p.2^2 / (b^2) = 1

/-- The equation of a line -/
def line_equation (m b : ℝ) : (ℝ × ℝ) → Prop :=
  fun p => p.2 = m * p.1 + b

theorem ellipse_intersection_theorem (C : Ellipse) (l : Line) :
  C.center = (0, 0) ∧ C.left_focus = (-Real.sqrt 3, 0) ∧ C.right_vertex = (2, 0) ∧ l.slope = 1/2 →
  (∃ a b : ℝ, standard_equation a b = standard_equation 2 1) ∧
  (∃ chord_length : ℝ, chord_length ≤ Real.sqrt 10 ∧
    ∀ other_length : ℝ, other_length ≤ chord_length) ∧
  (∃ b : ℝ, line_equation (1/2) b = line_equation (1/2) 0 →
    ∀ other_b : ℝ, ∃ length : ℝ, length ≤ Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l1630_163057


namespace NUMINAMATH_CALUDE_max_d_value_l1630_163022

def a (n : ℕ) : ℕ := 100 + n^2

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value : ∃ (n : ℕ), d n = 401 ∧ ∀ (m : ℕ), d m ≤ 401 :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l1630_163022


namespace NUMINAMATH_CALUDE_least_seven_digit_binary_l1630_163069

/-- The least positive base ten number that requires seven digits for its binary representation -/
def leastSevenDigitBinary : ℕ := 64

/-- A function that returns the number of digits in the binary representation of a natural number -/
def binaryDigits (n : ℕ) : ℕ := sorry

theorem least_seven_digit_binary :
  (∀ m : ℕ, m < leastSevenDigitBinary → binaryDigits m < 7) ∧
  binaryDigits leastSevenDigitBinary = 7 := by sorry

end NUMINAMATH_CALUDE_least_seven_digit_binary_l1630_163069


namespace NUMINAMATH_CALUDE_combined_average_score_l1630_163024

theorem combined_average_score (g₁ g₂ : ℕ) (avg₁ avg₂ : ℚ) :
  g₁ > 0 → g₂ > 0 →
  avg₁ = 88 →
  avg₂ = 76 →
  g₁ = (4 * g₂) / 5 →
  let total_score := g₁ * avg₁ + g₂ * avg₂
  let total_students := g₁ + g₂
  (total_score / total_students : ℚ) = 81 := by
  sorry

end NUMINAMATH_CALUDE_combined_average_score_l1630_163024


namespace NUMINAMATH_CALUDE_sum_coordinates_reflection_l1630_163025

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define reflection over x-axis
def reflect_x (p : Point) : Point :=
  (p.1, -p.2)

theorem sum_coordinates_reflection (y : ℝ) :
  let C : Point := (3, y)
  let D : Point := reflect_x C
  C.1 + C.2 + D.1 + D.2 = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_coordinates_reflection_l1630_163025


namespace NUMINAMATH_CALUDE_min_product_of_prime_sum_l1630_163018

theorem min_product_of_prime_sum (m n p : ℕ) : 
  Prime m → Prime n → Prime p → 
  m ≠ n → m ≠ p → n ≠ p →
  m + n = p →
  (∀ m' n' p' : ℕ, Prime m' → Prime n' → Prime p' → 
    m' ≠ n' → m' ≠ p' → n' ≠ p' →
    m' + n' = p' → m' * n' * p' ≥ m * n * p) →
  m * n * p = 30 := by
sorry

end NUMINAMATH_CALUDE_min_product_of_prime_sum_l1630_163018


namespace NUMINAMATH_CALUDE_white_ball_probability_l1630_163021

/-- Represents the number of balls initially in the bag -/
def initial_balls : ℕ := 6

/-- Represents the total number of balls after adding the white ball -/
def total_balls : ℕ := initial_balls + 1

/-- Represents the number of white balls added -/
def white_balls : ℕ := 1

/-- The probability of extracting the white ball -/
def prob_white : ℚ := white_balls / total_balls

theorem white_ball_probability :
  prob_white = 1 / 7 := by sorry

end NUMINAMATH_CALUDE_white_ball_probability_l1630_163021


namespace NUMINAMATH_CALUDE_sqrt_real_condition_l1630_163039

theorem sqrt_real_condition (x : ℝ) : (∃ y : ℝ, y ^ 2 = (x - 1) / 9) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_real_condition_l1630_163039


namespace NUMINAMATH_CALUDE_typists_for_180_letters_l1630_163058

/-- The number of typists needed to type a certain number of letters in a given time, 
    given a known typing rate. -/
def typists_needed 
  (known_typists : ℕ) 
  (known_letters : ℕ) 
  (known_minutes : ℕ) 
  (target_letters : ℕ) 
  (target_minutes : ℕ) : ℕ :=
  sorry

theorem typists_for_180_letters 
  (h1 : typists_needed 20 40 20 180 60 = 30) : 
  typists_needed 20 40 20 180 60 = 30 := by
  sorry

end NUMINAMATH_CALUDE_typists_for_180_letters_l1630_163058


namespace NUMINAMATH_CALUDE_total_marbles_l1630_163007

theorem total_marbles (mary_marbles joan_marbles : ℕ) 
  (h1 : mary_marbles = 9) 
  (h2 : joan_marbles = 3) : 
  mary_marbles + joan_marbles = 12 := by
sorry

end NUMINAMATH_CALUDE_total_marbles_l1630_163007


namespace NUMINAMATH_CALUDE_hat_shoppe_pricing_l1630_163099

theorem hat_shoppe_pricing (x : ℝ) (h : x > 0) : 
  0.75 * (1.3 * x) = 0.975 * x := by sorry

end NUMINAMATH_CALUDE_hat_shoppe_pricing_l1630_163099


namespace NUMINAMATH_CALUDE_function_always_positive_l1630_163015

theorem function_always_positive
  (f : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (h : ∀ x : ℝ, 2 * f x + x * (deriv f x) > x^2) :
  ∀ x : ℝ, f x > 0 :=
sorry

end NUMINAMATH_CALUDE_function_always_positive_l1630_163015


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1630_163059

theorem sufficient_condition_for_inequality (a : ℝ) :
  0 < a ∧ a < (1/5) → (1/a) > 3 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1630_163059


namespace NUMINAMATH_CALUDE_permutations_of_seven_distinct_objects_l1630_163026

theorem permutations_of_seven_distinct_objects : Nat.factorial 7 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_seven_distinct_objects_l1630_163026


namespace NUMINAMATH_CALUDE_coins_per_roll_is_25_l1630_163087

/-- Represents the number of coins in a single roll -/
def coins_per_roll : ℕ := sorry

/-- The number of rolls each bank teller has -/
def rolls_per_teller : ℕ := 10

/-- The number of bank tellers -/
def number_of_tellers : ℕ := 4

/-- The total number of coins among all tellers -/
def total_coins : ℕ := 1000

theorem coins_per_roll_is_25 : 
  coins_per_roll * rolls_per_teller * number_of_tellers = total_coins →
  coins_per_roll = 25 := by
  sorry

end NUMINAMATH_CALUDE_coins_per_roll_is_25_l1630_163087


namespace NUMINAMATH_CALUDE_taxi_fare_equation_l1630_163049

/-- Taxi fare calculation -/
theorem taxi_fare_equation 
  (x : ℝ) 
  (h_distance : x > 3) 
  (starting_price : ℝ := 6) 
  (price_per_km : ℝ := 2.4) 
  (total_fare : ℝ := 13.2) :
  starting_price + price_per_km * (x - 3) = total_fare := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_equation_l1630_163049


namespace NUMINAMATH_CALUDE_basketball_win_rate_l1630_163050

theorem basketball_win_rate (total_games : ℕ) (first_part_games : ℕ) (first_part_wins : ℕ) 
  (remaining_games : ℕ) (target_percentage : ℚ) :
  total_games = first_part_games + remaining_games →
  (first_part_wins : ℚ) / (first_part_games : ℚ) > target_percentage →
  ∃ (remaining_wins : ℕ), 
    remaining_wins ≤ remaining_games ∧ 
    ((first_part_wins + remaining_wins : ℚ) / (total_games : ℚ) ≥ target_percentage) ∧
    (∀ (x : ℕ), x < remaining_wins → 
      (first_part_wins + x : ℚ) / (total_games : ℚ) < target_percentage) :=
by
  sorry

-- Example usage
example : 
  ∃ (remaining_wins : ℕ), 
    remaining_wins ≤ 35 ∧ 
    ((45 + remaining_wins : ℚ) / 90 ≥ 3/4) ∧
    (∀ (x : ℕ), x < remaining_wins → (45 + x : ℚ) / 90 < 3/4) :=
basketball_win_rate 90 55 45 35 (3/4)
  (by norm_num)
  (by norm_num)

end NUMINAMATH_CALUDE_basketball_win_rate_l1630_163050


namespace NUMINAMATH_CALUDE_max_sum_product_sqrt_max_value_quarter_equality_condition_l1630_163082

theorem max_sum_product_sqrt (x1 x2 x3 x4 : ℝ) 
  (non_neg : x1 ≥ 0 ∧ x2 ≥ 0 ∧ x3 ≥ 0 ∧ x4 ≥ 0) 
  (sum_constraint : x1 + x2 + x3 + x4 = 1) :
  let sum_prod := (x1 + x2) * Real.sqrt (x1 * x2) + 
                  (x1 + x3) * Real.sqrt (x1 * x3) + 
                  (x1 + x4) * Real.sqrt (x1 * x4) + 
                  (x2 + x3) * Real.sqrt (x2 * x3) + 
                  (x2 + x4) * Real.sqrt (x2 * x4) + 
                  (x3 + x4) * Real.sqrt (x3 * x4)
  ∀ y1 y2 y3 y4 : ℝ, 
    y1 ≥ 0 → y2 ≥ 0 → y3 ≥ 0 → y4 ≥ 0 → 
    y1 + y2 + y3 + y4 = 1 → 
    sum_prod ≥ (y1 + y2) * Real.sqrt (y1 * y2) + 
               (y1 + y3) * Real.sqrt (y1 * y3) + 
               (y1 + y4) * Real.sqrt (y1 * y4) + 
               (y2 + y3) * Real.sqrt (y2 * y3) + 
               (y2 + y4) * Real.sqrt (y2 * y4) + 
               (y3 + y4) * Real.sqrt (y3 * y4) :=
by sorry

theorem max_value_quarter (x1 x2 x3 x4 : ℝ) 
  (non_neg : x1 ≥ 0 ∧ x2 ≥ 0 ∧ x3 ≥ 0 ∧ x4 ≥ 0) 
  (sum_constraint : x1 + x2 + x3 + x4 = 1) :
  let sum_prod := (x1 + x2) * Real.sqrt (x1 * x2) + 
                  (x1 + x3) * Real.sqrt (x1 * x3) + 
                  (x1 + x4) * Real.sqrt (x1 * x4) + 
                  (x2 + x3) * Real.sqrt (x2 * x3) + 
                  (x2 + x4) * Real.sqrt (x2 * x4) + 
                  (x3 + x4) * Real.sqrt (x3 * x4)
  sum_prod ≤ 3/4 :=
by sorry

theorem equality_condition (x1 x2 x3 x4 : ℝ) 
  (non_neg : x1 ≥ 0 ∧ x2 ≥ 0 ∧ x3 ≥ 0 ∧ x4 ≥ 0) 
  (sum_constraint : x1 + x2 + x3 + x4 = 1) :
  let sum_prod := (x1 + x2) * Real.sqrt (x1 * x2) + 
                  (x1 + x3) * Real.sqrt (x1 * x3) + 
                  (x1 + x4) * Real.sqrt (x1 * x4) + 
                  (x2 + x3) * Real.sqrt (x2 * x3) + 
                  (x2 + x4) * Real.sqrt (x2 * x4) + 
                  (x3 + x4) * Real.sqrt (x3 * x4)
  sum_prod = 3/4 ↔ x1 = 1/4 ∧ x2 = 1/4 ∧ x3 = 1/4 ∧ x4 = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_product_sqrt_max_value_quarter_equality_condition_l1630_163082


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1630_163093

/-- Two quantities vary inversely if their product is constant -/
def VaryInversely (a b : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, a x * b x = k

theorem inverse_variation_problem (a b : ℝ → ℝ) 
  (h1 : VaryInversely a b) 
  (h2 : a 1 = 1500) 
  (h3 : b 1 = 0.25) 
  (h4 : a 2 = 3000) : 
  b 2 = 0.125 := by
sorry


end NUMINAMATH_CALUDE_inverse_variation_problem_l1630_163093


namespace NUMINAMATH_CALUDE_dividend_calculation_l1630_163020

theorem dividend_calculation (divisor quotient remainder dividend : ℕ) : 
  divisor = 17 →
  quotient = 10 →
  remainder = 2 →
  dividend = divisor * quotient + remainder →
  dividend = 172 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1630_163020


namespace NUMINAMATH_CALUDE_garden_area_increase_l1630_163098

/-- Proves that changing a rectangular garden to a square garden with the same perimeter increases the area by 100 square feet. -/
theorem garden_area_increase (length width : ℝ) (h1 : length = 40) (h2 : width = 20) :
  let rectangle_area := length * width
  let perimeter := 2 * (length + width)
  let square_side := perimeter / 4
  let square_area := square_side * square_side
  square_area - rectangle_area = 100 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_increase_l1630_163098


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2011_l1630_163013

/-- The last four digits of 5^n -/
def lastFourDigits (n : ℕ) : ℕ := 5^n % 10000

/-- Given conditions -/
axiom base_case_5 : lastFourDigits 5 = 3125
axiom base_case_6 : lastFourDigits 6 = 5625
axiom base_case_7 : lastFourDigits 7 = 8125

/-- Theorem statement -/
theorem last_four_digits_of_5_pow_2011 : lastFourDigits 2011 = 8125 := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2011_l1630_163013


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_2023_l1630_163044

theorem smallest_prime_factor_of_2023 : Nat.minFac 2023 = 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_2023_l1630_163044


namespace NUMINAMATH_CALUDE_product_in_first_quadrant_l1630_163081

def complex_multiply (a b c d : ℝ) : ℂ :=
  Complex.mk (a * c - b * d) (a * d + b * c)

theorem product_in_first_quadrant :
  let z : ℂ := complex_multiply 1 3 3 (-1)
  (z.re > 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_product_in_first_quadrant_l1630_163081


namespace NUMINAMATH_CALUDE_ngon_area_division_l1630_163083

/-- Represents a convex n-gon -/
structure ConvexNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  convex : sorry -- Condition for convexity

/-- Represents a point on the boundary of the n-gon -/
structure BoundaryPoint (n : ℕ) (polygon : ConvexNGon n) where
  point : ℝ × ℝ
  on_boundary : sorry -- Condition for being on the boundary
  not_vertex : sorry -- Condition for not being a vertex

/-- Predicate to check if a line divides the polygon's area in half -/
def divides_area_in_half (n : ℕ) (polygon : ConvexNGon n) (a b : ℝ × ℝ) : Prop := sorry

/-- The number of sides on which the boundary points lie -/
def sides_with_points (n : ℕ) (polygon : ConvexNGon n) (points : Fin n → BoundaryPoint n polygon) : ℕ := sorry

theorem ngon_area_division (n : ℕ) (polygon : ConvexNGon n) 
  (points : Fin n → BoundaryPoint n polygon)
  (h_divide : ∀ i : Fin n, divides_area_in_half n polygon (polygon.vertices i) (points i).point) :
  (3 ≤ sides_with_points n polygon points) ∧ 
  (sides_with_points n polygon points ≤ if n % 2 = 0 then n - 1 else n) := sorry

end NUMINAMATH_CALUDE_ngon_area_division_l1630_163083


namespace NUMINAMATH_CALUDE_prob_two_heads_is_one_fourth_l1630_163009

/-- The probability of getting heads on a single flip of a fair coin -/
def prob_heads : ℚ := 1/2

/-- The probability of getting heads on both of the first two flips of a fair coin -/
def prob_two_heads : ℚ := prob_heads * prob_heads

/-- Theorem stating that the probability of getting heads on both of the first two flips of a fair coin is 1/4 -/
theorem prob_two_heads_is_one_fourth : prob_two_heads = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_heads_is_one_fourth_l1630_163009


namespace NUMINAMATH_CALUDE_saltwater_aquariums_l1630_163052

theorem saltwater_aquariums (total_saltwater_animals : ℕ) (animals_per_aquarium : ℕ) 
  (h1 : total_saltwater_animals = 1012)
  (h2 : animals_per_aquarium = 46) :
  total_saltwater_animals / animals_per_aquarium = 22 := by
  sorry

end NUMINAMATH_CALUDE_saltwater_aquariums_l1630_163052


namespace NUMINAMATH_CALUDE_triple_base_double_exponent_l1630_163042

theorem triple_base_double_exponent (a b x : ℝ) (hb : b ≠ 0) :
  let r := (3 * a) ^ (2 * b)
  r = a ^ b * x ^ b → x = 9 * a := by
sorry

end NUMINAMATH_CALUDE_triple_base_double_exponent_l1630_163042


namespace NUMINAMATH_CALUDE_problem_statement_l1630_163027

theorem problem_statement (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_eq : Real.exp a + a = Real.log (b * Real.exp b) ∧ Real.log (b * Real.exp b) = 2) :
  (b * Real.exp b = Real.exp 2) ∧
  (a + b = 2) ∧
  (Real.exp a + Real.log b = 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1630_163027


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1630_163051

/-- A sector OAB is a third of a circle with radius 6 cm. 
    An inscribed circle is tangent to the sector at three points. -/
def sector_with_inscribed_circle (r : ℝ) : Prop :=
  r > 0 ∧ 
  ∃ (R : ℝ), R = 6 ∧
  ∃ (θ : ℝ), θ = 2 * Real.pi / 3 ∧
  ∃ (x y : ℝ), x^2 + y^2 = r^2 ∧
  x = R * Real.sin θ ∧
  y = R * (1 - Real.cos θ)

/-- The radius of the inscribed circle in the sector described above is 6√2 - 6 cm. -/
theorem inscribed_circle_radius :
  ∀ r : ℝ, sector_with_inscribed_circle r → r = 6 * (Real.sqrt 2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1630_163051


namespace NUMINAMATH_CALUDE_inequality_property_l1630_163075

theorem inequality_property (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_property_l1630_163075


namespace NUMINAMATH_CALUDE_prime_equation_solutions_l1630_163055

theorem prime_equation_solutions (p : ℕ) :
  (Prime p ∧ ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p) ↔ p = 2 ∨ p = 3 ∨ p = 7 := by
  sorry

end NUMINAMATH_CALUDE_prime_equation_solutions_l1630_163055


namespace NUMINAMATH_CALUDE_value_of_expression_l1630_163036

theorem value_of_expression (x : ℝ) (h : x = 5) : 4 * x - 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1630_163036


namespace NUMINAMATH_CALUDE_sin_330_degrees_l1630_163065

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l1630_163065


namespace NUMINAMATH_CALUDE_maxwell_brad_meeting_time_l1630_163053

theorem maxwell_brad_meeting_time 
  (distance : ℝ) 
  (maxwell_speed : ℝ) 
  (brad_speed : ℝ) 
  (maxwell_head_start : ℝ) :
  distance = 34 →
  maxwell_speed = 4 →
  brad_speed = 6 →
  maxwell_head_start = 1 →
  ∃ (total_time : ℝ), 
    total_time = 4 ∧
    maxwell_speed * total_time + brad_speed * (total_time - maxwell_head_start) = distance :=
by
  sorry

end NUMINAMATH_CALUDE_maxwell_brad_meeting_time_l1630_163053


namespace NUMINAMATH_CALUDE_existence_of_solution_l1630_163062

theorem existence_of_solution : ∃ (a b : ℕ), 
  a > 1 ∧ b > 1 ∧ a^13 * b^31 = 6^2015 ∧ a = 2^155 ∧ b = 3^65 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_solution_l1630_163062


namespace NUMINAMATH_CALUDE_years_before_aziz_birth_l1630_163029

def current_year : ℕ := 2021
def aziz_age : ℕ := 36
def parents_move_year : ℕ := 1982

theorem years_before_aziz_birth : 
  current_year - aziz_age - parents_move_year = 3 := by sorry

end NUMINAMATH_CALUDE_years_before_aziz_birth_l1630_163029


namespace NUMINAMATH_CALUDE_linear_function_proof_l1630_163023

/-- A linear function passing through (-2, -1) and parallel to y = 2x - 3 -/
def f (x : ℝ) : ℝ := 2 * x + 3

/-- The slope of the line y = 2x - 3 -/
def slope_parallel : ℝ := 2

theorem linear_function_proof :
  (∀ x, f x = 2 * x + 3) ∧
  f (-2) = -1 ∧
  (∀ x y, f y - f x = slope_parallel * (y - x)) :=
sorry

end NUMINAMATH_CALUDE_linear_function_proof_l1630_163023


namespace NUMINAMATH_CALUDE_coin_problem_l1630_163001

theorem coin_problem (x y z : ℕ) : 
  x + y + z = 900 →
  x + 2*y + 5*z = 1950 →
  z = x / 2 →
  y = 450 := by
sorry

end NUMINAMATH_CALUDE_coin_problem_l1630_163001


namespace NUMINAMATH_CALUDE_graph_connectivity_probability_l1630_163011

/-- A complete graph with 20 vertices -/
def complete_graph : Nat := 20

/-- Number of edges removed -/
def removed_edges : Nat := 35

/-- Total number of edges in the complete graph -/
def total_edges : Nat := complete_graph * (complete_graph - 1) / 2

/-- Number of edges remaining after removal -/
def remaining_edges : Nat := total_edges - removed_edges

/-- Probability that the graph remains connected after edge removal -/
def connected_probability : ℚ :=
  1 - (complete_graph * (Nat.choose (total_edges - complete_graph + 1) (removed_edges - complete_graph + 1))) / 
      (Nat.choose total_edges removed_edges)

theorem graph_connectivity_probability :
  connected_probability = 1 - (20 * (Nat.choose 171 16)) / (Nat.choose 190 35) :=
sorry

end NUMINAMATH_CALUDE_graph_connectivity_probability_l1630_163011


namespace NUMINAMATH_CALUDE_min_power_cycles_mod1024_l1630_163092

/-- A power cycle is a set of nonnegative integer powers of an integer a. -/
def PowerCycle (a : ℤ) : Set ℤ :=
  {k : ℤ | ∃ n : ℕ, k = a ^ n}

/-- A set of power cycles covers all odd integers modulo 1024 if for any odd integer n,
    there exists a power cycle in the set and an integer k in that cycle
    such that n ≡ k (mod 1024). -/
def CoverAllOddMod1024 (S : Set (Set ℤ)) : Prop :=
  ∀ n : ℤ, Odd n → ∃ C ∈ S, ∃ k ∈ C, n ≡ k [ZMOD 1024]

/-- The theorem states that the minimum number of power cycles required
    to cover all odd integers modulo 1024 is 10. -/
theorem min_power_cycles_mod1024 :
  ∃ S : Set (Set ℤ),
    (∀ C ∈ S, ∃ a : ℤ, C = PowerCycle a) ∧
    CoverAllOddMod1024 S ∧
    S.ncard = 10 ∧
    ∀ T : Set (Set ℤ),
      (∀ C ∈ T, ∃ a : ℤ, C = PowerCycle a) →
      CoverAllOddMod1024 T →
      T.ncard ≥ 10 :=
by sorry

end NUMINAMATH_CALUDE_min_power_cycles_mod1024_l1630_163092


namespace NUMINAMATH_CALUDE_john_needs_thirteen_l1630_163017

/-- The amount of additional money John needs to buy a pogo stick -/
def additional_money_needed (saturday_earnings sunday_earnings previous_weekend_earnings pogo_stick_cost : ℕ) : ℕ :=
  pogo_stick_cost - (saturday_earnings + sunday_earnings + previous_weekend_earnings)

/-- Theorem stating how much additional money John needs -/
theorem john_needs_thirteen : 
  ∀ (saturday_earnings sunday_earnings previous_weekend_earnings pogo_stick_cost : ℕ),
    saturday_earnings = 18 →
    sunday_earnings = saturday_earnings / 2 →
    previous_weekend_earnings = 20 →
    pogo_stick_cost = 60 →
    additional_money_needed saturday_earnings sunday_earnings previous_weekend_earnings pogo_stick_cost = 13 := by
  sorry

end NUMINAMATH_CALUDE_john_needs_thirteen_l1630_163017


namespace NUMINAMATH_CALUDE_third_bakery_needs_twelve_sacks_l1630_163045

/-- The number of weeks Antoine supplies strawberries -/
def weeks : ℕ := 4

/-- The total number of sacks Antoine supplies in 4 weeks -/
def total_sacks : ℕ := 72

/-- The number of sacks the first bakery needs per week -/
def first_bakery_sacks : ℕ := 2

/-- The number of sacks the second bakery needs per week -/
def second_bakery_sacks : ℕ := 4

/-- The number of sacks the third bakery needs per week -/
def third_bakery_sacks : ℕ := total_sacks / weeks - (first_bakery_sacks + second_bakery_sacks)

theorem third_bakery_needs_twelve_sacks : third_bakery_sacks = 12 := by
  sorry

end NUMINAMATH_CALUDE_third_bakery_needs_twelve_sacks_l1630_163045


namespace NUMINAMATH_CALUDE_sqrt_eight_and_nine_sixteenths_l1630_163035

theorem sqrt_eight_and_nine_sixteenths (x : ℝ) : 
  x = Real.sqrt (8 + 9/16) → x = Real.sqrt 137 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_and_nine_sixteenths_l1630_163035


namespace NUMINAMATH_CALUDE_cost_price_is_1000_l1630_163056

/-- The cost price of a toy, given the selling conditions -/
def cost_price_of_toy (total_sold : ℕ) (selling_price : ℕ) (gain_in_toys : ℕ) : ℕ :=
  selling_price / (total_sold + gain_in_toys)

/-- Theorem stating the cost price of a toy under the given conditions -/
theorem cost_price_is_1000 :
  cost_price_of_toy 18 21000 3 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_is_1000_l1630_163056


namespace NUMINAMATH_CALUDE_exists_points_with_midpoint_l1630_163078

-- Define the hyperbola equation
def is_on_hyperbola (x y : ℝ) : Prop := x^2 - y^2/9 = 1

-- Define the midpoint of two points
def is_midpoint (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2

-- Theorem statement
theorem exists_points_with_midpoint :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_on_hyperbola x₁ y₁ ∧
    is_on_hyperbola x₂ y₂ ∧
    is_midpoint (-1) (-4) x₁ y₁ x₂ y₂ :=
by
  sorry

end NUMINAMATH_CALUDE_exists_points_with_midpoint_l1630_163078


namespace NUMINAMATH_CALUDE_like_terms_exponent_value_l1630_163037

theorem like_terms_exponent_value (x y : ℝ) (m n : ℕ) :
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * x^(m+3) * y^6 = b * x^5 * y^(2*n)) →
  m^n = 8 := by
sorry

end NUMINAMATH_CALUDE_like_terms_exponent_value_l1630_163037


namespace NUMINAMATH_CALUDE_max_k_value_l1630_163054

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (h : 4 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + k * ((x / y) + (y / x))) :
  k ≤ 1 ∧ ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 4 = 1^2 * ((x^2 / y^2) + (y^2 / x^2)) + 1 * ((x / y) + (y / x)) :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l1630_163054


namespace NUMINAMATH_CALUDE_total_employees_l1630_163072

/-- Given a corporation with part-time and full-time employees, 
    calculate the total number of employees. -/
theorem total_employees (part_time full_time : ℕ) :
  part_time = 2041 →
  full_time = 63093 →
  part_time + full_time = 65134 := by
  sorry

end NUMINAMATH_CALUDE_total_employees_l1630_163072


namespace NUMINAMATH_CALUDE_segment_ratio_l1630_163084

/-- Given four distinct points on a plane with segments of lengths a, a, b, a+√3b, 2a, and 2b
    connecting them, the ratio of b to a is 2 + √3. -/
theorem segment_ratio (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  ∃ (p1 p2 p3 p4 : ℝ × ℝ), 
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
    ({dist p1 p2, dist p1 p3, dist p1 p4, dist p2 p3, dist p2 p4, dist p3 p4} : Finset ℝ) = 
      {a, a, b, a + Real.sqrt 3 * b, 2 * a, 2 * b} →
    b / a = 2 + Real.sqrt 3 := by
  sorry

#check segment_ratio

end NUMINAMATH_CALUDE_segment_ratio_l1630_163084


namespace NUMINAMATH_CALUDE_xyz_values_l1630_163094

theorem xyz_values (x y z : ℝ) 
  (eq1 : x * y - 5 * y = 20)
  (eq2 : y * z - 5 * z = 20)
  (eq3 : z * x - 5 * x = 20) :
  x * y * z = 340 ∨ x * y * z = -62.5 := by
sorry

end NUMINAMATH_CALUDE_xyz_values_l1630_163094


namespace NUMINAMATH_CALUDE_distance_A_O_min_distance_O_line_l1630_163097

-- Define the polyline distance function
def polyline_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

-- Define point A
def A : ℝ × ℝ := (-1, 3)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the line equation
def on_line (x y : ℝ) : Prop :=
  2 * x + y - 2 * Real.sqrt 5 = 0

-- Theorem 1: The polyline distance between A and O is 4
theorem distance_A_O :
  polyline_distance A.1 A.2 O.1 O.2 = 4 := by sorry

-- Theorem 2: The minimum polyline distance between O and any point on the line is √5
theorem min_distance_O_line :
  ∃ (x y : ℝ), on_line x y ∧
  ∀ (x' y' : ℝ), on_line x' y' →
  polyline_distance O.1 O.2 x y ≤ polyline_distance O.1 O.2 x' y' ∧
  polyline_distance O.1 O.2 x y = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_distance_A_O_min_distance_O_line_l1630_163097


namespace NUMINAMATH_CALUDE_cube_root_equation_sum_l1630_163032

theorem cube_root_equation_sum (x y z : ℕ+) :
  (4 : ℝ) * ((7 : ℝ)^(1/3) - (6 : ℝ)^(1/3))^(1/2) = (x : ℝ)^(1/3) + (y : ℝ)^(1/3) - (z : ℝ)^(1/3) →
  x + y + z = 75 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_sum_l1630_163032


namespace NUMINAMATH_CALUDE_sea_hidden_by_cloud_l1630_163089

theorem sea_hidden_by_cloud (total_landscape visible_island cloud_cover : ℚ) :
  cloud_cover = 1/2 ∧ 
  visible_island = 1/4 ∧ 
  visible_island = 3/4 * (visible_island + (cloud_cover - 1/2)) →
  cloud_cover - (cloud_cover - 1/2) - visible_island = 5/12 :=
by sorry

end NUMINAMATH_CALUDE_sea_hidden_by_cloud_l1630_163089


namespace NUMINAMATH_CALUDE_growth_percentage_calculation_l1630_163043

def previous_height : Real := 139.65
def current_height : Real := 147.0

theorem growth_percentage_calculation :
  let difference := current_height - previous_height
  let growth_rate := difference / previous_height
  let growth_percentage := growth_rate * 100
  ∃ ε > 0, abs (growth_percentage - 5.26) < ε :=
sorry

end NUMINAMATH_CALUDE_growth_percentage_calculation_l1630_163043


namespace NUMINAMATH_CALUDE_common_root_pairs_l1630_163090

theorem common_root_pairs (n : ℕ) (hn : n > 1) :
  ∀ s t : ℤ, (∃ x : ℝ, x^n + s*x = 2007 ∧ x^n + t*x = 2008) ↔ 
  ((s = 2006 ∧ t = 2007) ∨ 
   (s = -2008 ∧ t = -2009 ∧ Even n) ∨ 
   (s = -2006 ∧ t = -2007 ∧ Odd n)) :=
by sorry

end NUMINAMATH_CALUDE_common_root_pairs_l1630_163090


namespace NUMINAMATH_CALUDE_triangle_side_equations_l1630_163063

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the altitude and median
def altitude_equation (x y : ℝ) : Prop := x + 2*y - 4 = 0
def median_equation (x y : ℝ) : Prop := 2*x + y - 3 = 0

-- Define the equations of the sides
def side_AB_equation (x y : ℝ) : Prop := 2*x - y + 1 = 0
def side_BC_equation (x y : ℝ) : Prop := 2*x + 3*y - 7 = 0
def side_AC_equation (x y : ℝ) : Prop := y = 1

theorem triangle_side_equations 
  (tri : Triangle)
  (h1 : tri.A = (0, 1))
  (h2 : ∀ x y, altitude_equation x y → (x - tri.A.1) * (tri.B.2 - tri.A.2) = -(y - tri.A.2) * (tri.B.1 - tri.A.1))
  (h3 : ∀ x y, median_equation x y → 2 * x = tri.A.1 + tri.C.1 ∧ 2 * y = tri.A.2 + tri.C.2) :
  (∀ x y, side_AB_equation x y ↔ (y - tri.A.2) = ((tri.B.2 - tri.A.2) / (tri.B.1 - tri.A.1)) * (x - tri.A.1)) ∧
  (∀ x y, side_BC_equation x y ↔ (y - tri.B.2) = ((tri.C.2 - tri.B.2) / (tri.C.1 - tri.B.1)) * (x - tri.B.1)) ∧
  (∀ x y, side_AC_equation x y ↔ (y - tri.A.2) = ((tri.C.2 - tri.A.2) / (tri.C.1 - tri.A.1)) * (x - tri.A.1)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_equations_l1630_163063
