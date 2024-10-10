import Mathlib

namespace divisible_by_seven_last_digit_l1469_146946

theorem divisible_by_seven_last_digit :
  ∃! d : ℕ, d < 10 ∧ ∀ n : ℕ, n % 10 = d → (7 ∣ n ↔ 7 ∣ d) :=
by
  -- The proof goes here
  sorry

end divisible_by_seven_last_digit_l1469_146946


namespace total_marbles_is_76_l1469_146917

/-- The total number of marbles in an arrangement where 9 rows have 8 marbles each and 1 row has 4 marbles -/
def total_marbles : ℕ := 
  let rows_with_eight := 9
  let marbles_per_row_eight := 8
  let rows_with_four := 1
  let marbles_per_row_four := 4
  rows_with_eight * marbles_per_row_eight + rows_with_four * marbles_per_row_four

/-- Theorem stating that the total number of marbles is 76 -/
theorem total_marbles_is_76 : total_marbles = 76 := by
  sorry

end total_marbles_is_76_l1469_146917


namespace amount_second_shop_is_340_l1469_146938

/-- The amount spent on books from the second shop -/
def amount_second_shop (books_first : ℕ) (amount_first : ℕ) (books_second : ℕ) (total_books : ℕ) (avg_price : ℕ) : ℕ :=
  total_books * avg_price - amount_first

/-- Theorem: The amount spent on the second shop is 340 -/
theorem amount_second_shop_is_340 :
  amount_second_shop 55 1500 60 115 16 = 340 := by
  sorry

end amount_second_shop_is_340_l1469_146938


namespace church_cookie_baking_l1469_146941

theorem church_cookie_baking (members : ℕ) (sheets_per_member : ℕ) (cookies_per_sheet : ℕ)
  (h1 : members = 100)
  (h2 : sheets_per_member = 10)
  (h3 : cookies_per_sheet = 16) :
  members * sheets_per_member * cookies_per_sheet = 16000 := by
  sorry

end church_cookie_baking_l1469_146941


namespace vector_expression_l1469_146983

/-- Given vectors a, b, and c in ℝ², prove that c = 2a + b -/
theorem vector_expression (a b c : ℝ × ℝ) 
  (h1 : a = (1, 2)) 
  (h2 : a + b = (0, 3)) 
  (h3 : c = (1, 5)) : 
  c = 2 • a + b := by sorry

end vector_expression_l1469_146983


namespace distance_to_y_axis_l1469_146900

theorem distance_to_y_axis (x : ℝ) :
  let P : ℝ × ℝ := (x, -5)
  (|P.2| = (1/2 : ℝ) * |P.1|) → |P.1| = 10 :=
by sorry

end distance_to_y_axis_l1469_146900


namespace perfect_square_completion_l1469_146943

theorem perfect_square_completion (ε : ℝ) (hε : ε > 0) : 
  ∃ x : ℝ, ∃ y : ℝ, 
    (12.86 * 12.86 + 12.86 * x + 0.14 * 0.14 = y * y) ∧ 
    (|x - 0.28| < ε) := by
  sorry

end perfect_square_completion_l1469_146943


namespace first_stick_length_l1469_146982

theorem first_stick_length (stick1 stick2 stick3 : ℝ) : 
  stick2 = 2 * stick1 →
  stick3 = stick2 - 1 →
  stick1 + stick2 + stick3 = 14 →
  stick1 = 3 := by
sorry

end first_stick_length_l1469_146982


namespace camp_participants_equality_l1469_146906

structure CampParticipants where
  mathOrange : ℕ
  mathPurple : ℕ
  physicsOrange : ℕ
  physicsPurple : ℕ

theorem camp_participants_equality (p : CampParticipants) 
  (h : p.physicsOrange = p.mathPurple) : 
  p.mathOrange + p.mathPurple = p.mathOrange + p.physicsOrange :=
by
  sorry

#check camp_participants_equality

end camp_participants_equality_l1469_146906


namespace escalator_walking_rate_l1469_146915

/-- Given an escalator moving upwards at a certain rate with a specified length,
    prove that a person walking on it at a certain rate will take a specific time
    to cover the entire length. -/
theorem escalator_walking_rate
  (escalator_speed : ℝ)
  (escalator_length : ℝ)
  (time_taken : ℝ)
  (h1 : escalator_speed = 12)
  (h2 : escalator_length = 196)
  (h3 : time_taken = 14)
  : ∃ (walking_rate : ℝ),
    escalator_length = (walking_rate + escalator_speed) * time_taken ∧
    walking_rate = 2 :=
by sorry

end escalator_walking_rate_l1469_146915


namespace correct_operation_l1469_146930

theorem correct_operation (x y : ℝ) : 4 * x^3 * y^2 * (x^2 * y^3) = 4 * x^5 * y^5 := by
  sorry

end correct_operation_l1469_146930


namespace abs_neg_two_eq_two_l1469_146944

theorem abs_neg_two_eq_two : |(-2 : ℝ)| = 2 := by
  sorry

end abs_neg_two_eq_two_l1469_146944


namespace intersection_of_A_and_B_l1469_146945

def A : Set ℝ := {x | x ≤ 1}
def B : Set ℝ := {-3, 1, 2, 4}

theorem intersection_of_A_and_B : A ∩ B = {-3, 1} := by
  sorry

end intersection_of_A_and_B_l1469_146945


namespace bag_price_problem_l1469_146998

theorem bag_price_problem (P : ℝ) : 
  (P - P * 0.95 * 0.96 = 44) → P = 500 := by
  sorry

end bag_price_problem_l1469_146998


namespace leap_year_53_sundays_l1469_146959

/-- The number of days in a leap year -/
def leap_year_days : ℕ := 366

/-- The number of complete weeks in a leap year -/
def complete_weeks : ℕ := leap_year_days / 7

/-- The number of extra days beyond complete weeks in a leap year -/
def extra_days : ℕ := leap_year_days % 7

/-- The number of possible combinations for the extra days -/
def extra_day_combinations : ℕ := 7

/-- The number of combinations that include a Sunday -/
def sunday_combinations : ℕ := 2

/-- The probability of a randomly chosen leap year having 53 Sundays -/
def prob_53_sundays : ℚ := sunday_combinations / extra_day_combinations

theorem leap_year_53_sundays : 
  prob_53_sundays = 2 / 7 :=
sorry

end leap_year_53_sundays_l1469_146959


namespace max_profit_l1469_146986

noncomputable def T (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x < 6 then (9*x - 2*x^2) / (6 - x)
  else 0

theorem max_profit :
  ∃ (x : ℝ), 1 ≤ x ∧ x < 6 ∧ T x = 3 ∧ ∀ y, T y ≤ T x :=
by
  sorry

end max_profit_l1469_146986


namespace salad_price_proof_l1469_146908

/-- Proves that the price of each salad is $2.50 given the problem conditions --/
theorem salad_price_proof (hot_dog_price : ℝ) (hot_dog_count : ℕ) (salad_count : ℕ) 
  (payment : ℝ) (change : ℝ) :
  hot_dog_price = 1.5 →
  hot_dog_count = 5 →
  salad_count = 3 →
  payment = 20 →
  change = 5 →
  (payment - change - hot_dog_price * hot_dog_count) / salad_count = 2.5 := by
sorry

#eval (20 - 5 - 1.5 * 5) / 3

end salad_price_proof_l1469_146908


namespace profit_sharing_ratio_l1469_146963

/-- Represents the investment and time period for a partner --/
structure Partner where
  investment : ℕ
  months : ℕ

/-- Calculates the effective capital of a partner --/
def effectiveCapital (p : Partner) : ℕ := p.investment * p.months

/-- Calculates the ratio of two numbers --/
def ratio (a b : ℕ) : ℕ × ℕ :=
  let gcd := a.gcd b
  (a / gcd, b / gcd)

/-- Theorem stating the profit sharing ratio between P and Q --/
theorem profit_sharing_ratio (p q : Partner)
  (h1 : p.investment = 4000)
  (h2 : p.months = 12)
  (h3 : q.investment = 9000)
  (h4 : q.months = 8) :
  ratio (effectiveCapital p) (effectiveCapital q) = (2, 3) := by
  sorry

#check profit_sharing_ratio

end profit_sharing_ratio_l1469_146963


namespace statement_C_is_false_l1469_146909

def f (x : ℝ) := 3 - 4*x - 2*x^2

theorem statement_C_is_false :
  ¬(∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 1 2, f x ≤ max ∧ f x ≥ min) ∧ 
    (max = -3) ∧ (min = -13) ∧
    (∃ x₁ ∈ Set.Icc 1 2, f x₁ = max) ∧
    (∃ x₂ ∈ Set.Icc 1 2, f x₂ = min)) :=
by
  sorry


end statement_C_is_false_l1469_146909


namespace pencils_broken_l1469_146992

theorem pencils_broken (initial bought found misplaced final : ℕ) : 
  initial = 20 → 
  bought = 2 → 
  found = 4 → 
  misplaced = 7 → 
  final = 16 → 
  initial + bought + found - misplaced - final = 3 := by
  sorry

end pencils_broken_l1469_146992


namespace count_distinct_sums_of_special_fractions_l1469_146913

def is_special_fraction (a b : ℕ+) : Prop := a.val + b.val = 18

def sum_of_special_fractions (n : ℤ) : Prop :=
  ∃ (a₁ b₁ a₂ b₂ : ℕ+), 
    is_special_fraction a₁ b₁ ∧ 
    is_special_fraction a₂ b₂ ∧ 
    n = (a₁.val : ℤ) * b₂.val + (a₂.val : ℤ) * b₁.val

theorem count_distinct_sums_of_special_fractions : 
  ∃! (s : Finset ℤ), 
    (∀ n, n ∈ s ↔ sum_of_special_fractions n) ∧ 
    s.card = 3 :=
sorry

end count_distinct_sums_of_special_fractions_l1469_146913


namespace quadratic_inequality_solution_set_l1469_146931

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 9 > 0} = {x : ℝ | x < -3 ∨ x > 3} := by sorry

end quadratic_inequality_solution_set_l1469_146931


namespace min_sum_absolute_values_l1469_146967

theorem min_sum_absolute_values :
  (∀ x : ℝ, |x - 3| + |x - 1| + |x + 6| ≥ 9) ∧
  (∃ x : ℝ, |x - 3| + |x - 1| + |x + 6| = 9) := by
  sorry

end min_sum_absolute_values_l1469_146967


namespace third_degree_polynomial_property_l1469_146925

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial := ℝ → ℝ

/-- The property that the absolute value of g at certain points equals 10 -/
def HasSpecificValues (g : ThirdDegreePolynomial) : Prop :=
  |g 0| = 10 ∧ |g 1| = 10 ∧ |g 3| = 10 ∧ |g 4| = 10 ∧ |g 5| = 10 ∧ |g 8| = 10

theorem third_degree_polynomial_property (g : ThirdDegreePolynomial) 
  (h : HasSpecificValues g) : |g (-1)| = 70 := by
  sorry

end third_degree_polynomial_property_l1469_146925


namespace uncertain_roots_l1469_146940

/-- Given that mx² - 2(m+2)x + m + 5 = 0 has no real roots, 
    prove that the number of real roots of (m-5)x² - 2(m+2)x + m = 0 is uncertain. -/
theorem uncertain_roots (m : ℝ) 
  (h : ∀ x : ℝ, m * x^2 - 2*(m+2)*x + m + 5 ≠ 0) : 
  ∃ m₁ m₂ : ℝ, 
    (∃! x : ℝ, (m₁-5) * x^2 - 2*(m₁+2)*x + m₁ = 0) ∧ 
    (∃ x y : ℝ, x ≠ y ∧ (m₂-5) * x^2 - 2*(m₂+2)*x + m₂ = 0 ∧ (m₂-5) * y^2 - 2*(m₂+2)*y + m₂ = 0) :=
by
  sorry


end uncertain_roots_l1469_146940


namespace paintbrush_cost_calculation_l1469_146976

/-- The cost of the paintbrush Rose wants to buy -/
def paintbrush_cost (paints_cost easel_cost rose_has rose_needs : ℚ) : ℚ :=
  (rose_has + rose_needs) - (paints_cost + easel_cost)

/-- Theorem stating the cost of the paintbrush Rose wants to buy -/
theorem paintbrush_cost_calculation :
  paintbrush_cost 9.20 6.50 7.10 11 = 2.40 := by sorry

end paintbrush_cost_calculation_l1469_146976


namespace absent_boys_l1469_146995

/-- Proves the number of absent boys in a class with given conditions -/
theorem absent_boys (total_students : ℕ) (girls_present : ℕ) : 
  total_students = 250 →
  girls_present = 140 →
  girls_present = 2 * (total_students - (total_students - (girls_present + girls_present / 2))) →
  total_students - (girls_present + girls_present / 2) = 40 :=
by sorry

end absent_boys_l1469_146995


namespace parallelogram_vertex_sum_l1469_146975

/-- A parallelogram with vertices A, B, C, D in 2D space -/
structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The sum of coordinates of a point -/
def sumCoordinates (p : ℝ × ℝ) : ℝ := p.1 + p.2

/-- Theorem: In parallelogram ABCD with A(-1,2), B(3,-4), C(7,3), and A,C opposite, 
    the sum of coordinates of D is 12 -/
theorem parallelogram_vertex_sum (ABCD : Parallelogram) 
    (hA : ABCD.A = (-1, 2))
    (hB : ABCD.B = (3, -4))
    (hC : ABCD.C = (7, 3))
    (hAC_opposite : ABCD.A = (-ABCD.C.1, -ABCD.C.2)) :
    sumCoordinates ABCD.D = 12 := by
  sorry

end parallelogram_vertex_sum_l1469_146975


namespace percentage_problem_l1469_146926

theorem percentage_problem (x : ℝ) (a : ℝ) (h1 : (x / 100) * 170 = 85) (h2 : a = 170) : x = 50 := by
  sorry

end percentage_problem_l1469_146926


namespace total_profit_is_5400_l1469_146918

/-- Represents the profit sharing scenario between Tom and Jose -/
structure ProfitSharing where
  tom_investment : ℕ
  tom_months : ℕ
  jose_investment : ℕ
  jose_months : ℕ
  jose_profit : ℕ

/-- Calculates the total profit earned by Tom and Jose -/
def total_profit (ps : ProfitSharing) : ℕ :=
  sorry

/-- Theorem stating that the total profit is 5400 given the specified conditions -/
theorem total_profit_is_5400 (ps : ProfitSharing) 
  (h1 : ps.tom_investment = 3000)
  (h2 : ps.tom_months = 12)
  (h3 : ps.jose_investment = 4500)
  (h4 : ps.jose_months = 10)
  (h5 : ps.jose_profit = 3000) :
  total_profit ps = 5400 :=
sorry

end total_profit_is_5400_l1469_146918


namespace min_max_sum_x_l1469_146922

theorem min_max_sum_x (x y z : ℝ) 
  (sum_eq : x + y + z = 6)
  (sum_sq_eq : x^2 + y^2 + z^2 = 14) :
  ∃ (m M : ℝ), (∀ t, m ≤ t ∧ t ≤ M → ∃ u v, t + u + v = 6 ∧ t^2 + u^2 + v^2 = 14) ∧
                (∀ s, (∃ u v, s + u + v = 6 ∧ s^2 + u^2 + v^2 = 14) → m ≤ s ∧ s ≤ M) ∧
                m + M = 10/3 :=
sorry

end min_max_sum_x_l1469_146922


namespace hat_color_game_l1469_146990

/-- Represents the maximum number of correct guesses in the hat color game -/
def max_correct_guesses (n k : ℕ) : ℕ :=
  n - k - 1

/-- Theorem stating the maximum number of guaranteed correct guesses in the hat color game -/
theorem hat_color_game (n k : ℕ) (h1 : k < n) :
  max_correct_guesses n k = n - k - 1 :=
by sorry

end hat_color_game_l1469_146990


namespace wednesday_earnings_l1469_146977

/-- Represents the working hours and earnings of Jack and Bob on a particular Wednesday -/
structure WorkDay where
  t : ℝ
  jack_hours : ℝ := t - 2
  jack_rate : ℝ := 3 * t - 2
  bob_hours : ℝ := 1.5 * (t - 2)
  bob_rate : ℝ := (3 * t - 2) - (2 * t - 7)
  tax : ℝ := 10

/-- The theorem stating that t = 19/3 is the only valid solution -/
theorem wednesday_earnings (w : WorkDay) : 
  (w.jack_hours * w.jack_rate - w.tax = w.bob_hours * w.bob_rate - w.tax) ∧ 
  (w.jack_hours > 0) ∧ (w.bob_hours > 0) → 
  w.t = 19/3 := by
  sorry

#check wednesday_earnings

end wednesday_earnings_l1469_146977


namespace auction_bids_l1469_146952

theorem auction_bids (initial_price final_price : ℕ) (price_increase : ℕ) (num_bidders : ℕ) :
  initial_price = 15 →
  final_price = 65 →
  price_increase = 5 →
  num_bidders = 2 →
  (final_price - initial_price) / price_increase / num_bidders = 5 :=
by sorry

end auction_bids_l1469_146952


namespace proposition_implication_l1469_146960

theorem proposition_implication (P : ℕ+ → Prop) 
  (h1 : ∀ k : ℕ+, P k → P (k + 1)) 
  (h2 : ¬ P 9) : 
  ¬ P 8 := by
  sorry

end proposition_implication_l1469_146960


namespace constant_if_average_property_l1469_146974

/-- A function from ℤ² to ℕ -/
def GridFunction := ℤ × ℤ → ℕ

/-- The property that f(x, y) is the average of its four neighbors -/
def HasAverageProperty (f : GridFunction) : Prop :=
  ∀ x y : ℤ, f (x, y) = (f (x+1, y) + f (x-1, y) + f (x, y+1) + f (x, y-1)) / 4

/-- Main theorem: if f has the average property, then it is constant -/
theorem constant_if_average_property (f : GridFunction) (h : HasAverageProperty f) :
  ∃ c : ℕ, ∀ x y : ℤ, f (x, y) = c := by
  sorry

end constant_if_average_property_l1469_146974


namespace hexagon_perimeter_l1469_146928

/-- The perimeter of a hexagon with side length 7 inches is 42 inches. -/
theorem hexagon_perimeter : 
  ∀ (hexagon_side_length : ℝ), 
  hexagon_side_length = 7 → 
  6 * hexagon_side_length = 42 := by
  sorry

end hexagon_perimeter_l1469_146928


namespace find_y_l1469_146921

theorem find_y (n x y : ℝ) : 
  (100 + 200 + n + x) / 4 = 250 ∧ 
  (n + 150 + 100 + x + y) / 5 = 200 → 
  y = 50 := by
sorry

end find_y_l1469_146921


namespace quadratic_no_rational_solution_l1469_146953

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluation of a quadratic polynomial at a point x -/
def evaluate (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Statement: For any quadratic polynomial with real coefficients, 
    there exists a natural number n such that p(x) = 1/n has no rational solutions -/
theorem quadratic_no_rational_solution (p : QuadraticPolynomial) :
  ∃ n : ℕ, ∀ x : ℚ, evaluate p x ≠ 1 / n := by sorry

end quadratic_no_rational_solution_l1469_146953


namespace increase_amount_is_four_l1469_146933

/-- Represents a set of numbers with a known size and average -/
structure NumberSet where
  size : ℕ
  average : ℝ

/-- Calculates the sum of elements in a NumberSet -/
def NumberSet.sum (s : NumberSet) : ℝ := s.size * s.average

/-- The original set of numbers -/
def original_set : NumberSet := { size := 10, average := 6.2 }

/-- The new set of numbers after increasing one element -/
def new_set : NumberSet := { size := 10, average := 6.6 }

/-- The theorem to be proved -/
theorem increase_amount_is_four :
  new_set.sum - original_set.sum = 4 := by sorry

end increase_amount_is_four_l1469_146933


namespace power_sum_division_equals_seventeen_l1469_146951

theorem power_sum_division_equals_seventeen :
  1^234 + 4^6 / 4^4 = 17 := by
  sorry

end power_sum_division_equals_seventeen_l1469_146951


namespace unique_operator_assignment_l1469_146935

-- Define the arithmetic operators
inductive Operator
| Plus
| Minus
| Multiply
| Divide
| Equals

-- Define a function to apply an operator
def apply_operator (op : Operator) (a b : ℕ) : Prop :=
  match op with
  | Operator.Plus => a + b = b
  | Operator.Minus => a - b = b
  | Operator.Multiply => a * b = b
  | Operator.Divide => a / b = b
  | Operator.Equals => a = b

-- Define the theorem
theorem unique_operator_assignment :
  ∃! (A B C D E : Operator),
    apply_operator A 4 2 ∧
    apply_operator B 2 2 ∧
    apply_operator B 8 (4 * 2) ∧
    apply_operator C 4 2 ∧
    apply_operator D 2 3 ∧
    apply_operator B 5 5 ∧
    apply_operator B 4 (5 - 1) ∧
    apply_operator E 5 1 ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
    C ≠ D ∧ C ≠ E ∧
    D ≠ E :=
sorry

end unique_operator_assignment_l1469_146935


namespace xyz_equals_seven_cubed_l1469_146993

theorem xyz_equals_seven_cubed 
  (x y z : ℝ) 
  (h1 : x^2 * y * z^3 = 7^4) 
  (h2 : x * y^2 = 7^5) : 
  x * y * z = 7^3 := by
  sorry

end xyz_equals_seven_cubed_l1469_146993


namespace weight_of_doubled_cube_l1469_146910

/-- Given a cube of metal weighing 7 pounds, prove that another cube of the same metal
    with sides twice as long will weigh 56 pounds. -/
theorem weight_of_doubled_cube (s : ℝ) (ρ : ℝ) (h : ρ * s^3 = 7) :
  ρ * (2*s)^3 = 56 := by
sorry

end weight_of_doubled_cube_l1469_146910


namespace round_trip_average_speed_l1469_146978

/-- The average speed of a round trip given outbound and return speeds -/
theorem round_trip_average_speed
  (outbound_speed : ℝ)
  (return_speed : ℝ)
  (h1 : outbound_speed = 60)
  (h2 : return_speed = 40)
  : (2 / (1 / outbound_speed + 1 / return_speed)) = 48 := by
  sorry

end round_trip_average_speed_l1469_146978


namespace quadratic_equal_roots_l1469_146927

theorem quadratic_equal_roots (a : ℝ) :
  (∃ x : ℝ, x * (x + 1) + a * x = 0 ∧
   ∀ y : ℝ, y * (y + 1) + a * y = 0 → y = x) →
  a = -1 :=
sorry

end quadratic_equal_roots_l1469_146927


namespace triangle_side_length_l1469_146947

theorem triangle_side_length (a b c : ℝ) (A : ℝ) : 
  A = Real.pi / 3 →  -- 60 degrees in radians
  a * b = 11 →
  a + b = 7 →
  a > c ∧ c > b →
  c = 4 :=
by sorry

end triangle_side_length_l1469_146947


namespace existence_of_special_set_l1469_146956

theorem existence_of_special_set (n : ℕ) (h : n ≥ 2) :
  ∃ (S : Finset ℤ), Finset.card S = n ∧
    ∀ (a b : ℤ), a ∈ S → b ∈ S → a ≠ b → (a - b)^2 ∣ (a * b) := by
  sorry

end existence_of_special_set_l1469_146956


namespace f_of_tan_squared_l1469_146954

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem f_of_tan_squared (t : ℝ) (h1 : 0 ≤ t) (h2 : t ≤ π/4) :
  (∀ x : ℝ, x ≠ 0 → x ≠ 1 → f (x / (x - 1)) = 1 / x) →
  f (Real.tan t ^ 2) = Real.tan t ^ 2 - 1 := by
  sorry

end f_of_tan_squared_l1469_146954


namespace johns_grocery_spending_l1469_146999

theorem johns_grocery_spending (total_spent : ℚ) 
  (meat_fraction : ℚ) (bakery_fraction : ℚ) (candy_spent : ℚ) :
  total_spent = 24 →
  meat_fraction = 1/3 →
  bakery_fraction = 1/6 →
  candy_spent = 6 →
  total_spent - (meat_fraction * total_spent + bakery_fraction * total_spent) - candy_spent = 1/4 * total_spent :=
by sorry

end johns_grocery_spending_l1469_146999


namespace euro_puzzle_l1469_146980

theorem euro_puzzle (E M n : ℕ) : 
  (M + 3 = n * (E - 3)) →
  (E + n = 3 * (M - n)) →
  n > 0 →
  (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 7) :=
by sorry

end euro_puzzle_l1469_146980


namespace equal_perimeter_ratio_l1469_146996

/-- Given a square and an equilateral triangle with equal perimeters, 
    the ratio of the triangle's side length to the square's side length is 4/3 -/
theorem equal_perimeter_ratio (s t : ℝ) (hs : s > 0) (ht : t > 0) 
  (h_equal_perimeter : 4 * s = 3 * t) : t / s = 4 / 3 := by
  sorry

end equal_perimeter_ratio_l1469_146996


namespace option_d_is_deductive_reasoning_l1469_146958

/-- A predicate representing periodic functions --/
def IsPeriodic (f : ℝ → ℝ) : Prop := sorry

/-- A predicate representing trigonometric functions --/
def IsTrigonometric (f : ℝ → ℝ) : Prop := sorry

/-- Definition of deductive reasoning --/
def IsDeductiveReasoning (premise1 premise2 conclusion : Prop) : Prop :=
  (premise1 ∧ premise2) → conclusion

/-- The tangent function --/
noncomputable def tan : ℝ → ℝ := sorry

/-- Theorem stating that the reasoning in option D is deductive --/
theorem option_d_is_deductive_reasoning :
  IsDeductiveReasoning
    (∀ f, IsTrigonometric f → IsPeriodic f)
    (IsTrigonometric tan)
    (IsPeriodic tan) :=
sorry

end option_d_is_deductive_reasoning_l1469_146958


namespace equation_solution_l1469_146962

theorem equation_solution : ∀ x : ℚ, 
  (Real.sqrt (6 * x) / Real.sqrt (4 * (x - 1)) = 3) → x = 24 / 23 := by
  sorry

end equation_solution_l1469_146962


namespace total_sword_weight_l1469_146907

/-- The number of squads the Dark Lord has -/
def num_squads : ℕ := 10

/-- The number of orcs in each squad -/
def orcs_per_squad : ℕ := 8

/-- The weight of swords each orc carries (in pounds) -/
def sword_weight_per_orc : ℕ := 15

/-- Theorem stating the total weight of swords to be transported -/
theorem total_sword_weight : 
  num_squads * orcs_per_squad * sword_weight_per_orc = 1200 := by
  sorry

end total_sword_weight_l1469_146907


namespace diagonal_crosses_820_cubes_l1469_146991

/-- The number of unit cubes crossed by an internal diagonal in a rectangular solid. -/
def cubesCrossedByDiagonal (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

/-- Theorem stating that the number of cubes crossed by the diagonal in a 200 × 330 × 360 solid is 820. -/
theorem diagonal_crosses_820_cubes :
  cubesCrossedByDiagonal 200 330 360 = 820 := by
  sorry

end diagonal_crosses_820_cubes_l1469_146991


namespace is_projection_matrix_l1469_146994

def projection_matrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P * P = P

theorem is_projection_matrix : 
  let P : Matrix (Fin 2) (Fin 2) ℚ := !![2368/2401, 16/49; 33*2401/2240, 33/49]
  projection_matrix P := by sorry

end is_projection_matrix_l1469_146994


namespace greatest_3digit_base8_divisible_by_7_l1469_146920

/-- Converts a base 8 number to decimal --/
def base8ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base 8 --/
def decimalToBase8 (n : ℕ) : ℕ := sorry

/-- Checks if a number is a 3-digit base 8 number --/
def isThreeDigitBase8 (n : ℕ) : Prop := 
  100 ≤ n ∧ n ≤ 777

theorem greatest_3digit_base8_divisible_by_7 :
  ∃ (n : ℕ), isThreeDigitBase8 n ∧ 
             n % 7 = 0 ∧
             base8ToDecimal n = 511 ∧
             decimalToBase8 511 = 777 ∧
             ∀ (m : ℕ), isThreeDigitBase8 m ∧ m % 7 = 0 → m ≤ n :=
by sorry

end greatest_3digit_base8_divisible_by_7_l1469_146920


namespace division_in_base4_l1469_146981

/-- Converts a base 4 number to base 10 -/
def base4ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a base 10 number to base 4 -/
def base10ToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- Represents division in base 4 -/
def divBase4 (a b : List Nat) : List Nat :=
  base10ToBase4 ((base4ToBase10 a) / (base4ToBase10 b))

theorem division_in_base4 :
  divBase4 [3, 1, 2, 2] [3, 1] = [3, 5] := by sorry

end division_in_base4_l1469_146981


namespace function_relationship_l1469_146961

def f (b c : ℝ) (x : ℝ) : ℝ := x^2 - b*x + c

theorem function_relationship (b c : ℝ) :
  (∀ x, f b c (1 + x) = f b c (1 - x)) →
  f b c 0 = 3 →
  ∀ x, f b c (3^x) ≥ f b c (2^x) := by
  sorry

end function_relationship_l1469_146961


namespace circle_area_through_DEF_l1469_146929

-- Define the triangle DEF
def triangle_DEF (D E F : ℝ × ℝ) : Prop :=
  let d_e := Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2)
  let d_f := Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2)
  d_e = d_f ∧ d_e = 5 * Real.sqrt 3

-- Define the tangent circle
def tangent_circle (D E F : ℝ × ℝ) (G : ℝ × ℝ) : Prop :=
  let g_e := Real.sqrt ((E.1 - G.1)^2 + (E.2 - G.2)^2)
  let g_f := Real.sqrt ((F.1 - G.1)^2 + (F.2 - G.2)^2)
  g_e = 6 ∧ g_f = 6

-- Define the altitude condition
def altitude_condition (D E F : ℝ × ℝ) (G : ℝ × ℝ) : Prop :=
  let m_ef := (F.2 - E.2) / (F.1 - E.1)
  let m_dg := (G.2 - D.2) / (G.1 - D.1)
  m_ef * m_dg = -1

-- Theorem statement
theorem circle_area_through_DEF 
  (D E F : ℝ × ℝ) 
  (G : ℝ × ℝ) 
  (h1 : triangle_DEF D E F) 
  (h2 : tangent_circle D E F G) 
  (h3 : altitude_condition D E F G) :
  let R := Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) / 2
  Real.pi * R^2 = 36 * Real.pi := by sorry

end circle_area_through_DEF_l1469_146929


namespace max_both_writers_and_editors_l1469_146950

/-- Represents the number of people at the newspaper conference --/
def total_people : ℕ := 150

/-- Represents the number of writers at the conference --/
def writers : ℕ := 50

/-- Represents the number of editors at the conference --/
def editors : ℕ := 66

/-- Represents the number of people who are both writers and editors --/
def both (x : ℕ) : ℕ := x

/-- Represents the number of people who are neither writers nor editors --/
def neither (x : ℕ) : ℕ := 3 * x

/-- States that the number of editors is more than 65 --/
axiom editors_more_than_65 : editors > 65

/-- Theorem stating that the maximum number of people who are both writers and editors is 17 --/
theorem max_both_writers_and_editors :
  ∃ (x : ℕ), x ≤ 17 ∧
  total_people = writers + editors - both x + neither x ∧
  ∀ (y : ℕ), y > x →
    total_people ≠ writers + editors - both y + neither y :=
sorry

end max_both_writers_and_editors_l1469_146950


namespace z_takes_at_most_two_values_l1469_146934

/-- Given two distinct real numbers x and y with absolute values not less than 2,
    prove that z = uv + (uv)⁻¹ can take at most 2 distinct values,
    where u + u⁻¹ = x and v + v⁻¹ = y. -/
theorem z_takes_at_most_two_values (x y : ℝ) (hx : |x| ≥ 2) (hy : |y| ≥ 2) (hxy : x ≠ y) :
  ∃ (z₁ z₂ : ℝ), ∀ (u v : ℝ),
    (u + u⁻¹ = x) → (v + v⁻¹ = y) → (u * v + (u * v)⁻¹ = z₁ ∨ u * v + (u * v)⁻¹ = z₂) :=
by sorry

end z_takes_at_most_two_values_l1469_146934


namespace seating_arrangement_l1469_146901

theorem seating_arrangement (total_people : ℕ) (row_sizes : List ℕ) : 
  total_people = 65 →
  (∀ x ∈ row_sizes, x = 7 ∨ x = 8 ∨ x = 9) →
  (List.sum row_sizes = total_people) →
  (List.count 9 row_sizes = 1) :=
by sorry

end seating_arrangement_l1469_146901


namespace marie_erasers_l1469_146968

def initial_erasers : ℕ := 95
def lost_erasers : ℕ := 42

theorem marie_erasers : initial_erasers - lost_erasers = 53 := by
  sorry

end marie_erasers_l1469_146968


namespace original_price_l1469_146905

/-- Given an article with price changes and final price, calculate the original price -/
theorem original_price (q r : ℚ) : 
  (∃ (x : ℚ), x * (1 + q / 100) * (1 - r / 100) = 2) →
  (∃ (x : ℚ), x = 200 / (100 + q - r - q * r / 100)) :=
by sorry

end original_price_l1469_146905


namespace complement_A_intersect_B_l1469_146966

-- Define the sets A and B
def A : Set ℝ := {x | |x - 1| > 2}
def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- Define the complement of A in the universal set ℝ
def C_UA : Set ℝ := (Set.univ : Set ℝ) \ A

-- State the theorem
theorem complement_A_intersect_B : (C_UA ∩ B) = Set.Ioc 2 3 := by sorry

end complement_A_intersect_B_l1469_146966


namespace unique_f_exists_l1469_146914

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The function f(n) to be proved unique -/
def f (n : ℕ) : ℕ := sorry

/-- Main theorem statement -/
theorem unique_f_exists (n : ℕ) (h1 : n > 1) (h2 : n ≠ 10) :
  ∃! fn : ℕ, fn ≥ 2 ∧ ∀ k : ℕ, 0 < k → k < fn →
    sum_of_digits k + sum_of_digits (fn - k) = n :=
sorry

end unique_f_exists_l1469_146914


namespace quadratic_minimum_point_l1469_146985

/-- The x-coordinate of the minimum point of a quadratic function f(x) = x^2 - 2px + 4q,
    where p and q are positive real numbers, is p. -/
theorem quadratic_minimum_point (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  let f : ℝ → ℝ := fun x ↦ x^2 - 2*p*x + 4*q
  (∀ x, f p ≤ f x) ∧ (∃ x, f p < f x) := by
  sorry

end quadratic_minimum_point_l1469_146985


namespace exam_question_distribution_l1469_146971

theorem exam_question_distribution :
  ∃ (P M E : ℕ),
    P + M + E = 50 ∧
    P ≥ 39 ∧ P ≤ 41 ∧
    M ≥ 7 ∧ M ≤ 8 ∧
    E ≥ 2 ∧ E ≤ 3 ∧
    P = 40 ∧ M = 7 ∧ E = 3 :=
by sorry

end exam_question_distribution_l1469_146971


namespace sugar_solution_percentage_l1469_146984

theorem sugar_solution_percentage (x : ℝ) :
  (3/4 * x + 1/4 * 50 = 20) → x = 10 := by
  sorry

end sugar_solution_percentage_l1469_146984


namespace fourth_sample_is_twenty_l1469_146902

/-- Represents a systematic sampling scheme. -/
structure SystematicSample where
  population : ℕ
  sample_size : ℕ
  first_sample : ℕ
  interval : ℕ

/-- Generates the nth sample number in a systematic sampling scheme. -/
def nth_sample (s : SystematicSample) (n : ℕ) : ℕ :=
  s.first_sample + (n - 1) * s.interval

/-- The theorem stating that 20 is the fourth sample number. -/
theorem fourth_sample_is_twenty
  (total_students : ℕ)
  (h_total : total_students = 56)
  (sample : SystematicSample)
  (h_population : sample.population = total_students)
  (h_sample_size : sample.sample_size = 4)
  (h_first_sample : sample.first_sample = 6)
  (h_interval : sample.interval = total_students / sample.sample_size)
  (h_third_sample : nth_sample sample 3 = 34)
  (h_fourth_sample : nth_sample sample 4 = 48) :
  nth_sample sample 2 = 20 := by
  sorry


end fourth_sample_is_twenty_l1469_146902


namespace distinct_remainders_l1469_146903

def sequence_a : ℕ → ℕ
  | 0 => 1
  | n + 1 => 2^(sequence_a n) + sequence_a n

theorem distinct_remainders (n m : ℕ) (hn : n < 243) (hm : m < 243) (hnm : n ≠ m) :
  sequence_a n % 243 ≠ sequence_a m % 243 := by
  sorry

end distinct_remainders_l1469_146903


namespace imaginary_part_of_i_over_one_plus_i_l1469_146924

theorem imaginary_part_of_i_over_one_plus_i (i : ℂ) (h : i * i = -1) :
  Complex.im (i / (1 + i)) = 1 / 2 := by
  sorry

end imaginary_part_of_i_over_one_plus_i_l1469_146924


namespace percentage_of_125_equal_to_70_l1469_146965

theorem percentage_of_125_equal_to_70 : 
  ∃ p : ℝ, p * 125 = 70 ∧ p = 56 / 100 := by sorry

end percentage_of_125_equal_to_70_l1469_146965


namespace min_sum_and_inequality_l1469_146936

theorem min_sum_and_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3 * a * b) :
  (∃ (min : ℝ), min = 4/3 ∧ ∀ x y, x > 0 → y > 0 → x + y = 3 * x * y → x + y ≥ min) ∧
  (a / b + b / a ≥ 8 / (9 * a * b)) := by
sorry

end min_sum_and_inequality_l1469_146936


namespace max_product_is_48_l1469_146939

def max_product (x y z : ℕ+) : Prop :=
  (x : ℕ) + y + z = 12 ∧
  x ≤ y ∧ y ≤ z ∧
  z ≤ 3 * x ∧
  x * y * z ≤ 48

theorem max_product_is_48 :
  ∀ x y z : ℕ+, max_product x y z → x * y * z = 48 :=
sorry

end max_product_is_48_l1469_146939


namespace seminar_attendees_seminar_attendees_solution_l1469_146923

theorem seminar_attendees (total : ℕ) (company_a : ℕ) : ℕ :=
  let company_b := 2 * company_a
  let company_c := company_a + 10
  let company_d := company_c - 5
  let from_companies := company_a + company_b + company_c + company_d
  total - from_companies

theorem seminar_attendees_solution :
  seminar_attendees 185 30 = 20 := by
  sorry

end seminar_attendees_seminar_attendees_solution_l1469_146923


namespace license_plate_count_l1469_146942

/-- The number of consonants available for the license plate. -/
def num_consonants : ℕ := 20

/-- The number of vowels available for the license plate. -/
def num_vowels : ℕ := 6

/-- The number of digits available for the license plate. -/
def num_digits : ℕ := 10

/-- The number of special symbols available for the license plate. -/
def num_special_symbols : ℕ := 2

/-- The total number of possible license plates. -/
def total_license_plates : ℕ := num_consonants * num_vowels * num_consonants * num_digits * num_special_symbols

/-- Theorem stating that the total number of license plates is 48,000. -/
theorem license_plate_count : total_license_plates = 48000 := by
  sorry

end license_plate_count_l1469_146942


namespace percentage_difference_l1469_146969

theorem percentage_difference (p j t : ℝ) 
  (hj : j = 0.75 * p) 
  (ht : t = 0.9375 * p) : 
  (t - j) / t = 0.2 := by
sorry

end percentage_difference_l1469_146969


namespace least_positive_integer_with_remainders_l1469_146937

theorem least_positive_integer_with_remainders : ∃ b : ℕ, 
  b > 0 ∧ 
  b % 2 = 1 ∧ 
  b % 5 = 2 ∧ 
  b % 7 = 3 ∧ 
  ∀ c : ℕ, c > 0 ∧ c % 2 = 1 ∧ c % 5 = 2 ∧ c % 7 = 3 → b ≤ c :=
by
  use 17
  sorry

end least_positive_integer_with_remainders_l1469_146937


namespace initial_mean_calculation_l1469_146919

theorem initial_mean_calculation (n : ℕ) (M initial_wrong corrected_value new_mean : ℝ) :
  n = 50 ∧
  initial_wrong = 23 ∧
  corrected_value = 30 ∧
  new_mean = 36.5 ∧
  (n : ℝ) * new_mean = (n : ℝ) * M + (corrected_value - initial_wrong) →
  M = 36.36 := by
sorry

end initial_mean_calculation_l1469_146919


namespace geometry_relationships_l1469_146955

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (line_parallel : Line → Line → Prop)
variable (line_perpendicular : Line → Line → Prop)

-- State the theorem
theorem geometry_relationships 
  (l m : Line) (a : Plane) (h_diff : l ≠ m) :
  (perpendicular l a ∧ contains a m → line_perpendicular l m) ∧
  (perpendicular l a ∧ line_parallel l m → perpendicular m a) ∧
  ¬(parallel l a ∧ contains a m → line_parallel l m) ∧
  ¬(parallel l a ∧ parallel m a → line_parallel l m) :=
sorry

end geometry_relationships_l1469_146955


namespace fraction_division_addition_l1469_146973

theorem fraction_division_addition : (5 : ℚ) / 6 / ((9 : ℚ) / 10) + (1 : ℚ) / 15 = (402 : ℚ) / 405 := by
  sorry

end fraction_division_addition_l1469_146973


namespace max_sum_xy_l1469_146916

-- Define the constraint function
def constraint (x y : ℝ) : Prop :=
  Real.log y / Real.log ((x^2 + y^2) / 2) ≥ 1 ∧ (x ≠ 0 ∨ y ≠ 0) ∧ x^2 + y^2 ≠ 2

-- State the theorem
theorem max_sum_xy :
  ∃ (max : ℝ), max = 1 + Real.sqrt 2 ∧
  (∀ x y : ℝ, constraint x y → x + y ≤ max) ∧
  (∃ x y : ℝ, constraint x y ∧ x + y = max) :=
sorry

end max_sum_xy_l1469_146916


namespace parabola_y_intercepts_l1469_146979

-- Define the quadratic equation
def quadratic_equation (y : ℝ) : ℝ := 3 * y^2 - 5 * y + 2

-- State the theorem
theorem parabola_y_intercepts :
  ∃ (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧ quadratic_equation y₁ = 0 ∧ quadratic_equation y₂ = 0 ∧
  ∀ (y : ℝ), quadratic_equation y = 0 → y = y₁ ∨ y = y₂ :=
sorry

end parabola_y_intercepts_l1469_146979


namespace roots_quadratic_equation_l1469_146970

theorem roots_quadratic_equation (a b : ℝ) : 
  (a^2 - a - 2013 = 0) → (b^2 - b - 2013 = 0) → (a^2 + 2*a + 3*b - 2 = 2014) := by
  sorry

end roots_quadratic_equation_l1469_146970


namespace base7_to_base49_conversion_l1469_146972

/-- Converts a list of digits in base 7 to a natural number -/
def fromBase7 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 7 * acc + d) 0

/-- Converts a natural number to a list of digits in base 49 -/
def toBase49 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 49) ((m % 49) :: acc)
    aux n []

theorem base7_to_base49_conversion :
  toBase49 (fromBase7 [6, 2, 6]) = [0, 6, 0, 2, 0, 6] := by
  sorry

end base7_to_base49_conversion_l1469_146972


namespace g_50_eq_zero_l1469_146964

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → x * g y + y * g x = g (x * y)

/-- The main theorem stating that g(50) = 0 for any function satisfying the functional equation -/
theorem g_50_eq_zero (g : ℝ → ℝ) (h : FunctionalEquation g) : g 50 = 0 := by
  sorry

end g_50_eq_zero_l1469_146964


namespace first_group_selection_is_five_l1469_146932

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  group_count : ℕ
  group_size : ℕ
  selected_number : ℕ
  selected_group : ℕ

/-- Calculates the position of a number within its group -/
def position_in_group (s : SystematicSampling) : ℕ :=
  s.selected_number - (s.selected_group - 1) * s.group_size

/-- Calculates the number selected from the first group -/
def first_group_selection (s : SystematicSampling) : ℕ :=
  position_in_group s

/-- Theorem stating the correct number selected from the first group -/
theorem first_group_selection_is_five (s : SystematicSampling) 
  (h1 : s.total_students = 160)
  (h2 : s.sample_size = 20)
  (h3 : s.group_count = 20)
  (h4 : s.group_size = 8)
  (h5 : s.selected_number = 125)
  (h6 : s.selected_group = 16) : 
  first_group_selection s = 5 := by
  sorry

end first_group_selection_is_five_l1469_146932


namespace circle_chord_intersection_l1469_146987

theorem circle_chord_intersection (r : ℝ) (chord_length : ℝ) :
  r = 8 →
  chord_length = 12 →
  ∃ (ak kb : ℝ),
    ak = 8 - 2 * Real.sqrt 7 ∧
    kb = 8 + 2 * Real.sqrt 7 ∧
    ak + kb = 2 * r :=
by sorry

end circle_chord_intersection_l1469_146987


namespace peanut_mixture_proof_l1469_146949

/-- Given the following:
    - 10 pounds of Virginia peanuts cost $3.50 per pound
    - Spanish peanuts cost $3.00 per pound
    - The desired mixture should cost $3.40 per pound
    Prove that 2.5 pounds of Spanish peanuts should be used to create the mixture. -/
theorem peanut_mixture_proof (virginia_weight : ℝ) (virginia_price : ℝ) (spanish_price : ℝ) 
  (mixture_price : ℝ) (spanish_weight : ℝ) :
  virginia_weight = 10 →
  virginia_price = 3.5 →
  spanish_price = 3 →
  mixture_price = 3.4 →
  spanish_weight = 2.5 →
  (virginia_weight * virginia_price + spanish_weight * spanish_price) / (virginia_weight + spanish_weight) = mixture_price :=
by sorry

end peanut_mixture_proof_l1469_146949


namespace polynomial_division_remainder_l1469_146911

-- Define the polynomial and the divisor
def f (x : ℝ) : ℝ := x^6 - 2*x^5 - 3*x^4 + 4*x^3 + 5*x^2 - x - 2
def g (x : ℝ) : ℝ := (x-3)*(x^2-1)

-- Define the remainder
def r (x : ℝ) : ℝ := 18*x^2 + x - 17

-- Theorem statement
theorem polynomial_division_remainder :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, f x = g x * q x + r x :=
sorry

end polynomial_division_remainder_l1469_146911


namespace function_range_l1469_146989

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x - 1

-- Define the domain
def domain : Set ℝ := {x | -2 < x ∧ x < 1}

-- State the theorem
theorem function_range :
  {y | ∃ x ∈ domain, f x = y} = {y | -2 ≤ y ∧ y < 2} := by sorry

end function_range_l1469_146989


namespace rectangular_box_surface_area_l1469_146904

/-- Given a rectangular box with dimensions a, b, and c, if the sum of the lengths of its twelve edges
    is 156 and the distance from one corner to the farthest corner is 25, then its total surface area is 896. -/
theorem rectangular_box_surface_area
  (a b c : ℝ)
  (edge_sum : 4 * a + 4 * b + 4 * c = 156)
  (diagonal : Real.sqrt (a^2 + b^2 + c^2) = 25) :
  2 * (a * b + b * c + c * a) = 896 := by
  sorry

end rectangular_box_surface_area_l1469_146904


namespace smallest_factorial_with_43_zeroes_l1469_146948

/-- Count the number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 175 is the smallest positive integer k such that k! ends in at least 43 zeroes -/
theorem smallest_factorial_with_43_zeroes :
  (∀ k : ℕ, k > 0 → k < 175 → trailingZeroes k < 43) ∧ trailingZeroes 175 = 43 := by
  sorry

#eval trailingZeroes 175  -- Should output 43

end smallest_factorial_with_43_zeroes_l1469_146948


namespace sum_of_first_four_terms_l1469_146957

/-- Geometric sequence with given properties -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem sum_of_first_four_terms (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 = 9 →
  a 5 = 243 →
  (a 1) + (a 2) + (a 3) + (a 4) = 120 := by
  sorry

end sum_of_first_four_terms_l1469_146957


namespace binomial_15_4_l1469_146988

theorem binomial_15_4 : Nat.choose 15 4 = 1365 := by sorry

end binomial_15_4_l1469_146988


namespace symmetric_line_equation_l1469_146997

/-- A line in the 2D plane represented by its slope-intercept form -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.yIntercept

/-- Defines symmetry of two lines with respect to the y-axis -/
def symmetricAboutYAxis (l₁ l₂ : Line) : Prop :=
  ∀ x y, l₁.contains x y ↔ l₂.contains (-x) y

/-- The main theorem -/
theorem symmetric_line_equation (l₁ l₂ : Line) :
  l₁.slope = 3 →
  l₁.contains 1 2 →
  symmetricAboutYAxis l₁ l₂ →
  ∀ x y, l₂.contains x y ↔ 3 * x + y + 1 = 0 :=
sorry

end symmetric_line_equation_l1469_146997


namespace distance_to_place_l1469_146912

/-- The distance to the place given the man's rowing speed, river speed, and total time -/
theorem distance_to_place (mans_speed : ℝ) (river_speed : ℝ) (total_time : ℝ) :
  mans_speed = 4 →
  river_speed = 2 →
  total_time = 1.5 →
  (1 / (mans_speed + river_speed) + 1 / (mans_speed - river_speed)) * total_time = 2.25 :=
by sorry

end distance_to_place_l1469_146912
