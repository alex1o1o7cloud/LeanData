import Mathlib

namespace division_problem_l2924_292462

theorem division_problem (A : ℕ) : A / 3 = 8 ∧ A % 3 = 2 → A = 26 := by
  sorry

end division_problem_l2924_292462


namespace division_ratio_l2924_292455

theorem division_ratio (divisor quotient remainder : ℕ) 
  (h1 : divisor = 10 * quotient)
  (h2 : ∃ n : ℕ, divisor = n * remainder)
  (h3 : remainder = 46)
  (h4 : 5290 = divisor * quotient + remainder) :
  divisor / remainder = 5 :=
sorry

end division_ratio_l2924_292455


namespace quadratic_roots_range_l2924_292483

/-- A quadratic equation (k-1)x^2 + 4x + 1 = 0 has two distinct real roots -/
def has_two_distinct_real_roots (k : ℝ) : Prop :=
  (k - 1 ≠ 0) ∧ (16 - 4 * k + 4 > 0)

/-- The range of k for which the quadratic equation has two distinct real roots -/
theorem quadratic_roots_range :
  ∀ k : ℝ, has_two_distinct_real_roots k ↔ (k < 5 ∧ k ≠ 1) :=
by sorry

end quadratic_roots_range_l2924_292483


namespace A_B_red_mutually_exclusive_not_contradictory_l2924_292421

-- Define the set of cards
inductive Card : Type
| Black : Card
| Red : Card
| White : Card

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person

-- Define a distribution of cards to people
def Distribution := Person → Card

-- Define the event "A gets the red card"
def A_gets_red (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "B gets the red card"
def B_gets_red (d : Distribution) : Prop := d Person.B = Card.Red

-- Theorem stating that "A gets the red card" and "B gets the red card" are mutually exclusive but not contradictory
theorem A_B_red_mutually_exclusive_not_contradictory :
  (∀ d : Distribution, ¬(A_gets_red d ∧ B_gets_red d)) ∧
  (∃ d1 d2 : Distribution, A_gets_red d1 ∧ B_gets_red d2) :=
sorry

end A_B_red_mutually_exclusive_not_contradictory_l2924_292421


namespace expression_equality_l2924_292461

theorem expression_equality : 201 * 5 + 1220 - 2 * 3 * 5 * 7 = 2015 := by
  sorry

end expression_equality_l2924_292461


namespace divisibility_condition_l2924_292422

theorem divisibility_condition (a b : ℤ) : 
  (∃ d : ℕ, d ≥ 2 ∧ ∀ n : ℕ, n > 0 → (d : ℤ) ∣ (a^n + b^n + 1)) ↔ 
  ((a % 2 = 0 ∧ b % 2 = 1) ∨ (a % 3 = 1 ∧ b % 3 = 1)) := by
sorry

end divisibility_condition_l2924_292422


namespace lucky_larry_coincidence_l2924_292444

theorem lucky_larry_coincidence (a b c d e : ℤ) : 
  a = 2 → b = 4 → c = 3 → d = 5 → e = -15 →
  a - (b - (c * (d + e))) = a - b - c * d + e := by sorry

end lucky_larry_coincidence_l2924_292444


namespace reflection_sum_theorem_l2924_292440

def point (x y : ℝ) := (x, y)

def reflect_over_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def sum_coordinates (p1 p2 : ℝ × ℝ) : ℝ :=
  p1.1 + p1.2 + p2.1 + p2.2

theorem reflection_sum_theorem :
  let C : ℝ × ℝ := point 5 (-3)
  let D : ℝ × ℝ := reflect_over_x_axis C
  sum_coordinates C D = 10 := by sorry

end reflection_sum_theorem_l2924_292440


namespace mike_oil_changes_l2924_292420

/-- Represents the time in minutes for various car maintenance tasks and Mike's work schedule --/
structure CarMaintenance where
  wash_time : Nat  -- Time to wash a car in minutes
  oil_change_time : Nat  -- Time to change oil in minutes
  tire_change_time : Nat  -- Time to change a set of tires in minutes
  total_work_time : Nat  -- Total work time in minutes
  cars_washed : Nat  -- Number of cars washed
  tire_sets_changed : Nat  -- Number of tire sets changed

/-- Calculates the number of cars Mike changed oil on given the car maintenance data --/
def calculate_oil_changes (data : CarMaintenance) : Nat :=
  let total_wash_time := data.wash_time * data.cars_washed
  let total_tire_change_time := data.tire_change_time * data.tire_sets_changed
  let remaining_time := data.total_work_time - total_wash_time - total_tire_change_time
  remaining_time / data.oil_change_time

/-- Theorem stating that given the problem conditions, Mike changed oil on 6 cars --/
theorem mike_oil_changes (data : CarMaintenance) 
  (h1 : data.wash_time = 10)
  (h2 : data.oil_change_time = 15)
  (h3 : data.tire_change_time = 30)
  (h4 : data.total_work_time = 4 * 60)
  (h5 : data.cars_washed = 9)
  (h6 : data.tire_sets_changed = 2) :
  calculate_oil_changes data = 6 := by
  sorry

end mike_oil_changes_l2924_292420


namespace bus_seat_difference_l2924_292434

/-- Represents a bus with seats on both sides and a special seat at the back. -/
structure Bus where
  leftSeats : Nat
  rightSeats : Nat
  backSeatCapacity : Nat
  regularSeatCapacity : Nat
  totalCapacity : Nat

/-- The difference in the number of seats between the left and right sides of the bus. -/
def seatDifference (bus : Bus) : Nat :=
  bus.leftSeats - bus.rightSeats

/-- Theorem stating the difference in seats for a specific bus configuration. -/
theorem bus_seat_difference :
  ∃ (bus : Bus),
    bus.leftSeats = 15 ∧
    bus.backSeatCapacity = 11 ∧
    bus.regularSeatCapacity = 3 ∧
    bus.totalCapacity = 92 ∧
    seatDifference bus = 3 := by
  sorry

#check bus_seat_difference

end bus_seat_difference_l2924_292434


namespace gcd_90_250_l2924_292406

theorem gcd_90_250 : Nat.gcd 90 250 = 10 := by
  sorry

end gcd_90_250_l2924_292406


namespace residue_of_power_mod_13_l2924_292441

theorem residue_of_power_mod_13 : (5 ^ 1234 : ℕ) % 13 = 12 := by
  sorry

end residue_of_power_mod_13_l2924_292441


namespace savings_from_discount_l2924_292471

def initial_price : ℚ := 475
def discounted_price : ℚ := 199

theorem savings_from_discount :
  initial_price - discounted_price = 276 := by
  sorry

end savings_from_discount_l2924_292471


namespace product_25_sum_0_l2924_292445

theorem product_25_sum_0 (a b c d : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d → 
  a * b * c * d = 25 → 
  a + b + c + d = 0 := by
sorry

end product_25_sum_0_l2924_292445


namespace sally_net_earnings_two_months_l2924_292412

-- Define the given values
def last_month_work_income : ℝ := 1000
def last_month_work_expenses : ℝ := 200
def last_month_side_hustle : ℝ := 150
def work_income_increase : ℝ := 0.1
def work_expenses_increase : ℝ := 0.15
def side_hustle_increase : ℝ := 0.2
def tax_rate : ℝ := 0.25

-- Define the calculation functions
def calculate_net_work_income (income : ℝ) (expenses : ℝ) : ℝ :=
  income - expenses - (tax_rate * income)

def calculate_total_net_earnings (work_income : ℝ) (side_hustle : ℝ) : ℝ :=
  calculate_net_work_income work_income last_month_work_expenses + side_hustle

-- Theorem statement
theorem sally_net_earnings_two_months :
  let last_month := calculate_total_net_earnings last_month_work_income last_month_side_hustle
  let this_month := calculate_total_net_earnings 
    (last_month_work_income * (1 + work_income_increase))
    (last_month_side_hustle * (1 + side_hustle_increase))
  last_month + this_month = 1475 := by sorry

end sally_net_earnings_two_months_l2924_292412


namespace negation_of_universal_proposition_l2924_292413

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 0 → x^3 + 2*x ≥ 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^3 + 2*x < 0) :=
by sorry

end negation_of_universal_proposition_l2924_292413


namespace ping_pong_paddles_sold_l2924_292405

/-- Given the total sales and average price per pair of ping pong paddles,
    prove the number of pairs sold. -/
theorem ping_pong_paddles_sold
  (total_sales : ℝ)
  (avg_price : ℝ)
  (h1 : total_sales = 735)
  (h2 : avg_price = 9.8) :
  total_sales / avg_price = 75 := by
  sorry

end ping_pong_paddles_sold_l2924_292405


namespace prob_three_red_large_deck_l2924_292493

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)
  (hsum : total_cards = red_cards + black_cards)

/-- Probability of drawing three red cards in a row -/
def prob_three_red (d : Deck) : ℚ :=
  (d.red_cards : ℚ) / d.total_cards *
  ((d.red_cards - 1) : ℚ) / (d.total_cards - 1) *
  ((d.red_cards - 2) : ℚ) / (d.total_cards - 2)

/-- The main theorem -/
theorem prob_three_red_large_deck :
  let d : Deck := ⟨104, 52, 52, rfl⟩
  prob_three_red d = 425 / 3502 := by sorry

end prob_three_red_large_deck_l2924_292493


namespace circle_intersection_angle_l2924_292428

-- Define the circle equation
def circle_equation (x y c : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 2*y + c = 0

-- Define the center of the circle
def center : ℝ × ℝ := (2, -1)

-- Define the angle APB
def angle_APB : ℝ := 120

-- Theorem statement
theorem circle_intersection_angle (c : ℝ) :
  ∃ (A B : ℝ × ℝ),
    (A.1 = 0 ∧ circle_equation A.1 A.2 c) ∧
    (B.1 = 0 ∧ circle_equation B.1 B.2 c) ∧
    (angle_APB = 120) →
    c = -11 := by
  sorry

end circle_intersection_angle_l2924_292428


namespace product_units_digit_in_base_6_l2924_292468

/-- The units digit of the product of two numbers in a given base -/
def unitsDigitInBase (a b : ℕ) (base : ℕ) : ℕ :=
  (a * b) % base

/-- 314 in base 10 -/
def num1 : ℕ := 314

/-- 59 in base 10 -/
def num2 : ℕ := 59

/-- The base we're converting to -/
def targetBase : ℕ := 6

theorem product_units_digit_in_base_6 :
  unitsDigitInBase num1 num2 targetBase = 4 := by
  sorry

end product_units_digit_in_base_6_l2924_292468


namespace smallest_norm_l2924_292402

open Real
open InnerProductSpace

/-- Given a vector v such that ‖v + (4, 2)‖ = 10, the smallest possible value of ‖v‖ is 10 - 2√5 -/
theorem smallest_norm (v : ℝ × ℝ) (h : ‖v + (4, 2)‖ = 10) :
  ∃ (w : ℝ × ℝ), ‖w‖ = 10 - 2 * Real.sqrt 5 ∧ ∀ u : ℝ × ℝ, ‖u + (4, 2)‖ = 10 → ‖w‖ ≤ ‖u‖ :=
by sorry

end smallest_norm_l2924_292402


namespace arctan_sum_three_seven_l2924_292467

theorem arctan_sum_three_seven : Real.arctan (3/7) + Real.arctan (7/3) = π / 2 := by
  sorry

end arctan_sum_three_seven_l2924_292467


namespace square_binomial_divided_by_negative_square_l2924_292400

theorem square_binomial_divided_by_negative_square (m : ℝ) (hm : m ≠ 0) :
  (2 * m^2 - m)^2 / (-m^2) = -4 * m^2 + 4 * m - 1 := by
  sorry

end square_binomial_divided_by_negative_square_l2924_292400


namespace line_points_k_value_l2924_292429

theorem line_points_k_value (m n k : ℝ) : 
  (∀ x y, x - 5/2 * y + 1 = 0 ↔ y = 2/5 * x + 2/5) →
  (m - 5/2 * n + 1 = 0) →
  ((m + 1/2) - 5/2 * (n + 1/k) + 1 = 0) →
  (n + 1/k = n + 1) →
  k = 1 := by
  sorry

end line_points_k_value_l2924_292429


namespace total_work_hours_l2924_292408

theorem total_work_hours (hours_per_day : ℕ) (days_worked : ℕ) : 
  hours_per_day = 3 → days_worked = 5 → hours_per_day * days_worked = 15 :=
by sorry

end total_work_hours_l2924_292408


namespace softball_team_size_l2924_292492

theorem softball_team_size (men women : ℕ) : 
  women = men + 6 →
  (men : ℝ) / (women : ℝ) = 0.45454545454545453 →
  men + women = 16 := by
sorry

end softball_team_size_l2924_292492


namespace unique_prime_solution_l2924_292458

theorem unique_prime_solution : ∃! (p q r : ℕ), 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p + q^2 = r^4 := by
  sorry

end unique_prime_solution_l2924_292458


namespace chessboard_coverage_l2924_292411

/-- An L-shaped piece covers exactly 3 squares -/
def L_shape_coverage : ℕ := 3

/-- A unit square piece covers exactly 1 square -/
def unit_square_coverage : ℕ := 1

/-- Predicate to determine if an n×n chessboard can be covered -/
def can_cover (n : ℕ) : Prop :=
  ∃ k : ℕ, n^2 = k * L_shape_coverage ∨ n^2 = k * L_shape_coverage + unit_square_coverage

theorem chessboard_coverage (n : ℕ) :
  ¬(can_cover n) ↔ n % 3 = 2 :=
sorry

end chessboard_coverage_l2924_292411


namespace divisibility_criterion_l2924_292401

theorem divisibility_criterion (n : ℕ+) : 
  (n + 2 : ℕ) ∣ (n^3 + 3*n + 29 : ℕ) ↔ n = 1 ∨ n = 3 ∨ n = 13 :=
sorry

end divisibility_criterion_l2924_292401


namespace range_of_x_when_a_is_one_range_of_a_when_q_sufficient_not_necessary_l2924_292404

-- Define the propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Theorem for part (1)
theorem range_of_x_when_a_is_one (x : ℝ) (h : p 1 x ∧ q x) : 2 < x ∧ x < 3 := by
  sorry

-- Theorem for part (2)
theorem range_of_a_when_q_sufficient_not_necessary (a : ℝ) 
  (h1 : a > 0)
  (h2 : ∀ x, q x → p a x)
  (h3 : ∃ x, p a x ∧ ¬q x) : 
  1 < a ∧ a ≤ 2 := by
  sorry

end range_of_x_when_a_is_one_range_of_a_when_q_sufficient_not_necessary_l2924_292404


namespace unique_solution_implies_a_equals_10_l2924_292431

-- Define the equation
def equation (a : ℝ) (x : ℝ) : Prop :=
  (x * Real.log a ^ 2 - 1) / (x + Real.log a) = x

-- Theorem statement
theorem unique_solution_implies_a_equals_10 :
  (∃! x : ℝ, equation a x) → a = 10 :=
by sorry

end unique_solution_implies_a_equals_10_l2924_292431


namespace cos_negative_nineteen_pi_sixths_l2924_292485

theorem cos_negative_nineteen_pi_sixths :
  Real.cos (-19 * π / 6) = Real.sqrt 3 / 2 := by
  sorry

end cos_negative_nineteen_pi_sixths_l2924_292485


namespace set_operations_l2924_292435

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 4}

-- Define the intervals for the expected results
def open_interval (a b : ℝ) : Set ℝ := {x | a < x ∧ x < b}
def closed_open_interval (a b : ℝ) : Set ℝ := {x | a ≤ x ∧ x < b}
def open_closed_interval (a b : ℝ) : Set ℝ := {x | a < x ∧ x ≤ b}
def left_ray (a : ℝ) : Set ℝ := {x | x ≤ a}
def right_ray (a : ℝ) : Set ℝ := {x | a < x}

-- State the theorem
theorem set_operations :
  (A ∩ B = open_interval 0 3) ∧
  (A ∪ B = open_interval (-1) 4) ∧
  ((Aᶜ ∩ Bᶜ) = left_ray (-1) ∪ right_ray 4) :=
by sorry

end set_operations_l2924_292435


namespace problem_solution_l2924_292452

theorem problem_solution (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_prod : x * y * z = 1)
  (h_eq1 : x + 1 / z = 3)
  (h_eq2 : y + 1 / x = 31) : 
  z + 1 / y = 9 / 23 := by
sorry

end problem_solution_l2924_292452


namespace completing_square_result_l2924_292499

theorem completing_square_result (x : ℝ) :
  x^2 - 6*x + 4 = 0 ↔ (x - 3)^2 = 5 :=
sorry

end completing_square_result_l2924_292499


namespace gcd_3150_9800_l2924_292439

theorem gcd_3150_9800 : Nat.gcd 3150 9800 = 350 := by sorry

end gcd_3150_9800_l2924_292439


namespace remainder_101_37_mod_100_l2924_292480

theorem remainder_101_37_mod_100 : (101^37) % 100 = 1 := by
  sorry

end remainder_101_37_mod_100_l2924_292480


namespace quadratic_one_solution_l2924_292454

theorem quadratic_one_solution (n : ℝ) : 
  (∃! x : ℝ, 4 * x^2 + n * x + 4 = 0) ↔ (n = 8 ∧ n > 0) := by
  sorry

end quadratic_one_solution_l2924_292454


namespace expression_value_l2924_292476

theorem expression_value (a : ℚ) (h : a = 1/3) : (2 * a⁻¹ + a⁻¹ / 3) / a^2 = 63 := by
  sorry

end expression_value_l2924_292476


namespace truth_teller_liar_arrangement_l2924_292496

def is_valid_arrangement (n k : ℕ) : Prop :=
  n > 0 ∧ k > 0 ∧ k < n ∧ 
  ∃ (m : ℕ), 2^m * k < n ∧ n ≤ 2^(m+1) * k

theorem truth_teller_liar_arrangement (n k : ℕ) :
  is_valid_arrangement n k →
  ∃ (m : ℕ), n = 2^m * (n.gcd k) ∧ 2^m > (k / (n.gcd k)) :=
sorry

end truth_teller_liar_arrangement_l2924_292496


namespace inequality_and_bound_l2924_292472

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1|

-- State the theorem
theorem inequality_and_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = Real.sqrt 2) :
  (∀ x, f x > 3 - |x + 2| ↔ x < -3 ∨ x > 0) ∧
  (∀ x, f x - |x| ≤ Real.sqrt (a^2 + 4*b^2)) := by
  sorry

end inequality_and_bound_l2924_292472


namespace inequality_solution_set_l2924_292497

theorem inequality_solution_set (x : ℝ) : 
  (|x + 1| - 2 > 0) ↔ (x < -3 ∨ x > 1) :=
by sorry

end inequality_solution_set_l2924_292497


namespace no_solution_implies_m_leq_2_l2924_292426

theorem no_solution_implies_m_leq_2 (m : ℝ) :
  (∀ x : ℝ, ¬(x - 2 < 3*x - 6 ∧ x < m)) → m ≤ 2 := by
  sorry

end no_solution_implies_m_leq_2_l2924_292426


namespace a_eq_neg_one_sufficient_not_necessary_l2924_292479

-- Define the complex number z
def z (a : ℝ) : ℂ := (a - 2*Complex.I)*Complex.I

-- Define the point M in the complex plane
def M (a : ℝ) : ℂ := z a

-- Define what it means for a point to be in the fourth quadrant
def in_fourth_quadrant (c : ℂ) : Prop := 0 < c.re ∧ c.im < 0

-- State the theorem
theorem a_eq_neg_one_sufficient_not_necessary :
  (∀ a : ℝ, a = -1 → in_fourth_quadrant (M a)) ∧
  (∃ a : ℝ, a ≠ -1 ∧ in_fourth_quadrant (M a)) :=
sorry

end a_eq_neg_one_sufficient_not_necessary_l2924_292479


namespace senior_titles_in_sample_l2924_292488

/-- Represents the number of staff members with senior titles in a stratified sample -/
def seniorTitlesInSample (totalStaff : ℕ) (seniorStaff : ℕ) (sampleSize : ℕ) : ℕ :=
  (seniorStaff * sampleSize) / totalStaff

/-- Theorem: In a company with 150 staff members, including 15 with senior titles,
    a stratified sample of size 30 will contain 3 staff members with senior titles. -/
theorem senior_titles_in_sample :
  seniorTitlesInSample 150 15 30 = 3 := by
  sorry

end senior_titles_in_sample_l2924_292488


namespace f_composition_value_l2924_292450

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x else 2 * Real.sqrt 2 * Real.cos x

theorem f_composition_value : f (f (-Real.pi/4)) = 4 := by
  sorry

end f_composition_value_l2924_292450


namespace power_function_property_l2924_292459

/-- Given a power function f(x) = x^α where α ∈ ℝ, 
    if f(2) = √2, then f(4) = 2 -/
theorem power_function_property (α : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x > 0, f x = x ^ α) 
  (h2 : f 2 = Real.sqrt 2) : 
  f 4 = 2 := by
  sorry

end power_function_property_l2924_292459


namespace optimal_transportation_plan_l2924_292478

structure VehicleType where
  capacity : ℕ
  cost : ℕ

def total_supply : ℕ := 120

def vehicle_a : VehicleType := ⟨5, 300⟩
def vehicle_b : VehicleType := ⟨8, 400⟩
def vehicle_c : VehicleType := ⟨10, 500⟩

def total_vehicles : ℕ := 18

def transportation_plan (a b c : ℕ) : Prop :=
  a + b + c = total_vehicles ∧
  a * vehicle_a.capacity + b * vehicle_b.capacity + c * vehicle_c.capacity ≥ total_supply

def transportation_cost (a b c : ℕ) : ℕ :=
  a * vehicle_a.cost + b * vehicle_b.cost + c * vehicle_c.cost

theorem optimal_transportation_plan :
  ∀ (a b c : ℕ),
    transportation_plan a b c →
    transportation_cost a b c ≥ transportation_cost 8 10 0 :=
by sorry

end optimal_transportation_plan_l2924_292478


namespace smallest_n_for_multiples_l2924_292442

theorem smallest_n_for_multiples : ∃ (a : Fin 15 → ℕ), 
  (∀ i : Fin 15, 16 ≤ a i ∧ a i ≤ 34) ∧ 
  (∀ i : Fin 15, a i % (i.val + 1) = 0) ∧
  (∀ i j : Fin 15, i ≠ j → a i ≠ a j) ∧
  (∀ n : ℕ, n < 34 → ¬∃ (b : Fin 15 → ℕ), 
    (∀ i : Fin 15, 16 ≤ b i ∧ b i ≤ n) ∧ 
    (∀ i : Fin 15, b i % (i.val + 1) = 0) ∧
    (∀ i j : Fin 15, i ≠ j → b i ≠ b j)) :=
by sorry

end smallest_n_for_multiples_l2924_292442


namespace nara_height_l2924_292475

/-- Given the heights of Sangheon, Chiho, and Nara, prove Nara's height -/
theorem nara_height (sangheon_height : Real) (chiho_diff : Real) (nara_diff : Real)
  (h1 : sangheon_height = 1.56)
  (h2 : chiho_diff = 0.14)
  (h3 : nara_diff = 0.27) :
  sangheon_height - chiho_diff + nara_diff = 1.69 := by
  sorry


end nara_height_l2924_292475


namespace car_speed_problem_l2924_292423

theorem car_speed_problem (v : ℝ) : 
  (∀ (t : ℝ), t = 3 → (70 - v) * t = 60) → v = 50 := by
  sorry

end car_speed_problem_l2924_292423


namespace mixture_weight_approx_140_l2924_292403

/-- Represents the weight ratio of almonds to walnuts in the mixture -/
def almond_to_walnut_ratio : ℚ := 5

/-- Represents the weight of almonds in the mixture in pounds -/
def almond_weight : ℚ := 116.67

/-- Calculates the total weight of the mixture -/
def total_mixture_weight : ℚ :=
  almond_weight + (almond_weight / almond_to_walnut_ratio)

/-- Theorem stating that the total weight of the mixture is approximately 140 pounds -/
theorem mixture_weight_approx_140 :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ |total_mixture_weight - 140| < ε :=
sorry

end mixture_weight_approx_140_l2924_292403


namespace coin_and_die_probability_l2924_292446

/-- Probability of getting heads on a biased coin -/
def prob_heads : ℚ := 2/3

/-- Probability of getting an even number on a regular six-sided die -/
def prob_even_die : ℚ := 1/2

/-- Theorem: The probability of getting heads on a biased coin with 2/3 probability for heads
    and an even number on a regular six-sided die is 1/3 -/
theorem coin_and_die_probability :
  prob_heads * prob_even_die = 1/3 := by
  sorry

end coin_and_die_probability_l2924_292446


namespace swordfish_difference_l2924_292449

/-- The number of times Shelly and Sam go fishing -/
def fishing_trips : ℕ := 5

/-- The total number of swordfish caught in all trips -/
def total_swordfish : ℕ := 25

/-- The number of swordfish Shelly catches each time -/
def shelly_catch : ℕ := 5 - 2

/-- The number of swordfish Sam catches each time -/
def sam_catch : ℕ := (total_swordfish - fishing_trips * shelly_catch) / fishing_trips

theorem swordfish_difference : shelly_catch - sam_catch = 1 := by
  sorry

end swordfish_difference_l2924_292449


namespace car_resale_gain_l2924_292437

/-- Calculates the percentage gain when reselling a car -/
theorem car_resale_gain (original_price selling_price_2 : ℝ) (loss_percent : ℝ) : 
  original_price = 50561.80 →
  loss_percent = 11 →
  selling_price_2 = 54000 →
  (selling_price_2 - (original_price * (1 - loss_percent / 100))) / (original_price * (1 - loss_percent / 100)) * 100 = 20 :=
by sorry

end car_resale_gain_l2924_292437


namespace disk_count_l2924_292498

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green

/-- Represents the bag of disks -/
structure DiskBag where
  blue : ℕ
  yellow : ℕ
  green : ℕ

/-- The ratio of blue:yellow:green disks is 3:7:8 -/
def ratio_condition (bag : DiskBag) : Prop :=
  ∃ (x : ℕ), bag.blue = 3 * x ∧ bag.yellow = 7 * x ∧ bag.green = 8 * x

/-- There are 20 more green disks than blue disks -/
def green_blue_difference (bag : DiskBag) : Prop :=
  bag.green = bag.blue + 20

/-- The total number of disks in the bag -/
def total_disks (bag : DiskBag) : ℕ :=
  bag.blue + bag.yellow + bag.green

/-- Theorem: The total number of disks in the bag is 72 -/
theorem disk_count (bag : DiskBag) 
  (h1 : ratio_condition bag) 
  (h2 : green_blue_difference bag) : 
  total_disks bag = 72 := by
  sorry


end disk_count_l2924_292498


namespace circle_center_and_radius_l2924_292481

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y - 6 = 0

-- State the theorem
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ (x y : ℝ), circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    center = (-1, 2) ∧
    radius = Real.sqrt 11 :=
by sorry

end circle_center_and_radius_l2924_292481


namespace dans_car_mpg_l2924_292407

/-- Calculates the miles per gallon of Dan's car given the cost of gas and distance traveled on a certain amount of money. -/
theorem dans_car_mpg (gas_cost : ℝ) (miles : ℝ) (spent : ℝ) : 
  gas_cost = 4 → miles = 432 → spent = 54 → (miles / (spent / gas_cost)) = 32 :=
by sorry

end dans_car_mpg_l2924_292407


namespace female_students_count_l2924_292453

theorem female_students_count (total_average : ℚ) (male_count : ℕ) (male_average : ℚ) (female_average : ℚ) :
  total_average = 90 →
  male_count = 8 →
  male_average = 87 →
  female_average = 92 →
  ∃ (female_count : ℕ), 
    (male_count * male_average + female_count * female_average) / (male_count + female_count) = total_average ∧
    female_count = 12 := by
  sorry

end female_students_count_l2924_292453


namespace gcd_102_238_l2924_292419

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end gcd_102_238_l2924_292419


namespace win_sector_area_l2924_292489

/-- The area of the WIN sector on a circular spinner with given radius and winning probability -/
theorem win_sector_area (r : ℝ) (p : ℝ) (h_r : r = 10) (h_p : p = 3/7) :
  p * π * r^2 = 300 * π / 7 := by
  sorry

end win_sector_area_l2924_292489


namespace hyperbola_focus_l2924_292460

/-- Given a hyperbola with equation x²/a² - y²/2 = 1, where one of its asymptotes
    passes through the point (√2, 1), prove that one of its foci has coordinates (√6, 0) -/
theorem hyperbola_focus (a : ℝ) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / 2 = 1) →  -- Hyperbola equation
  (∃ (m : ℝ), m * Real.sqrt 2 = 1 ∧ m = Real.sqrt 2 / a) →  -- Asymptote through (√2, 1)
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / 2 = 1 ∧ x = Real.sqrt 6 ∧ y = 0) :=  -- Focus at (√6, 0)
by sorry

end hyperbola_focus_l2924_292460


namespace pamphlet_cost_l2924_292463

theorem pamphlet_cost : ∃ p : ℝ, p = 1.10 ∧ 8 * p < 9 ∧ 11 * p > 12 := by
  sorry

end pamphlet_cost_l2924_292463


namespace inequality_range_l2924_292448

theorem inequality_range (a : ℝ) : 
  (∀ x > a, 2 * x + 1 / (x - a) ≥ 2 * Real.sqrt 2) ↔ a ≥ 0 := by sorry

end inequality_range_l2924_292448


namespace louis_age_proof_l2924_292495

/-- Carla's age in 6 years -/
def carla_future_age : ℕ := 30

/-- Number of years until Carla reaches her future age -/
def years_until_future : ℕ := 6

/-- Sum of Carla's and Louis's current ages -/
def sum_of_ages : ℕ := 55

/-- Louis's current age -/
def louis_age : ℕ := 31

theorem louis_age_proof :
  louis_age = sum_of_ages - (carla_future_age - years_until_future) :=
by sorry

end louis_age_proof_l2924_292495


namespace domino_pigeonhole_l2924_292425

/-- Represents a domino with two halves -/
structure Domino :=
  (half1 : Fin 7)
  (half2 : Fin 7)

/-- Represents the state of dominoes after cutting -/
structure DominoState :=
  (row : List Domino)
  (cut_halves : List (Fin 7))

/-- The theorem statement -/
theorem domino_pigeonhole 
  (dominoes : List Domino)
  (h1 : dominoes.length = 28)
  (h2 : ∀ i : Fin 7, (dominoes.map Domino.half1 ++ dominoes.map Domino.half2).count i = 7)
  (state : DominoState)
  (h3 : state.row.length = 26)
  (h4 : state.cut_halves.length = 4)
  (h5 : ∀ d ∈ dominoes, d ∈ state.row ∨ (d.half1 ∈ state.cut_halves ∧ d.half2 ∈ state.cut_halves)) :
  ∃ i j : Fin 4, i ≠ j ∧ state.cut_halves[i] = state.cut_halves[j] :=
sorry

end domino_pigeonhole_l2924_292425


namespace solve_equation_one_solve_equation_two_l2924_292457

-- Equation 1
theorem solve_equation_one : 
  ∃ x : ℚ, (1 / 6) * (3 * x - 6) = (2 / 5) * x - 3 ∧ x = -20 := by sorry

-- Equation 2
theorem solve_equation_two : 
  ∃ x : ℚ, (1 - 2 * x) / 3 = (3 * x + 1) / 7 - 3 ∧ x = 67 / 23 := by sorry

end solve_equation_one_solve_equation_two_l2924_292457


namespace arithmetic_sequence_sum_l2924_292470

/-- An arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a_5 + a_8 = 24, then a_6 + a_7 = 24 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 5 + a 8 = 24) : 
  a 6 + a 7 = 24 := by
  sorry

end arithmetic_sequence_sum_l2924_292470


namespace kids_wearing_shoes_l2924_292418

theorem kids_wearing_shoes (total : ℕ) (socks : ℕ) (both : ℕ) (barefoot : ℕ) 
  (h_total : total = 22)
  (h_socks : socks = 12)
  (h_both : both = 6)
  (h_barefoot : barefoot = 8) :
  total - barefoot - (socks - both) = 8 := by
  sorry

end kids_wearing_shoes_l2924_292418


namespace only_set_C_forms_triangle_l2924_292433

/-- Triangle inequality theorem: The sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if a set of three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- The given sets of line segments -/
def set_A : (ℝ × ℝ × ℝ) := (3, 5, 8)
def set_B : (ℝ × ℝ × ℝ) := (8, 8, 18)
def set_C : (ℝ × ℝ × ℝ) := (1, 1, 1)
def set_D : (ℝ × ℝ × ℝ) := (3, 4, 8)

/-- Theorem: Among the given sets, only set C can form a triangle -/
theorem only_set_C_forms_triangle :
  ¬(can_form_triangle set_A.1 set_A.2.1 set_A.2.2) ∧
  ¬(can_form_triangle set_B.1 set_B.2.1 set_B.2.2) ∧
  can_form_triangle set_C.1 set_C.2.1 set_C.2.2 ∧
  ¬(can_form_triangle set_D.1 set_D.2.1 set_D.2.2) :=
by sorry

end only_set_C_forms_triangle_l2924_292433


namespace angle_sum_is_pi_half_l2924_292427

theorem angle_sum_is_pi_half (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : Real.sin α ^ 2 + Real.sin β ^ 2 = Real.sin (α + β)) : 
  α + β = π/2 := by
sorry

end angle_sum_is_pi_half_l2924_292427


namespace circle_triangle_problem_l2924_292491

/-- Given that a triangle equals three circles and a triangle plus a circle equals 40,
    prove that the circle equals 10 and the triangle equals 30. -/
theorem circle_triangle_problem (circle triangle : ℕ) 
    (h1 : triangle = 3 * circle)
    (h2 : triangle + circle = 40) :
    circle = 10 ∧ triangle = 30 := by
  sorry

end circle_triangle_problem_l2924_292491


namespace stripe_ratio_l2924_292477

/-- Given the conditions about stripes on tennis shoes, prove the ratio of Hortense's to Olga's stripes -/
theorem stripe_ratio (olga_stripes_per_shoe : ℕ) (rick_stripes_per_shoe : ℕ) (total_stripes : ℕ)
  (h1 : olga_stripes_per_shoe = 3)
  (h2 : rick_stripes_per_shoe = olga_stripes_per_shoe - 1)
  (h3 : total_stripes = 22)
  (h4 : total_stripes = 2 * olga_stripes_per_shoe + 2 * rick_stripes_per_shoe + hortense_stripes) :
  hortense_stripes / (2 * olga_stripes_per_shoe) = 2 :=
by sorry

end stripe_ratio_l2924_292477


namespace square_roots_and_cube_root_problem_l2924_292436

theorem square_roots_and_cube_root_problem (x y a : ℝ) :
  x > 0 ∧
  (a + 3)^2 = x ∧
  (2*a - 15)^2 = x ∧
  (x + y - 2)^(1/3) = 4 →
  x - 2*y + 2 = 17 := by
  sorry

end square_roots_and_cube_root_problem_l2924_292436


namespace min_ceiling_height_for_illumination_l2924_292490

/-- The minimum ceiling height for complete illumination of a rectangular field. -/
theorem min_ceiling_height_for_illumination (length width : ℝ) 
  (h : ℝ) (multiple : ℝ) : 
  length = 100 →
  width = 80 →
  multiple = 0.1 →
  (∃ (n : ℕ), h = n * multiple) →
  (2 * h ≥ Real.sqrt (length^2 + width^2)) →
  (∀ (h' : ℝ), (∃ (n : ℕ), h' = n * multiple) → 
    (2 * h' ≥ Real.sqrt (length^2 + width^2)) → h' ≥ h) →
  h = 32.1 :=
by sorry

end min_ceiling_height_for_illumination_l2924_292490


namespace arrangement_count_l2924_292486

def number_of_arrangements (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  Nat.choose n k * Nat.choose m k * Nat.factorial (n + m - 2 * k)

theorem arrangement_count :
  let total_people : ℕ := 6
  let people_per_row : ℕ := 3
  number_of_arrangements total_people people_per_row 2 = 216 :=
by sorry

end arrangement_count_l2924_292486


namespace solve_for_a_l2924_292414

theorem solve_for_a : ∀ a : ℝ, (∃ x : ℝ, x = 1 ∧ 2 * x - a = 0) → a = 2 := by
  sorry

end solve_for_a_l2924_292414


namespace probability_of_two_defective_in_two_tests_l2924_292430

/-- The number of electronic components -/
def total_components : ℕ := 6

/-- The number of defective components -/
def defective_components : ℕ := 2

/-- The number of qualified components -/
def qualified_components : ℕ := 4

/-- The probability of finding exactly 2 defective components after 2 tests -/
def probability_two_defective_in_two_tests : ℚ := 1 / 15

/-- Theorem stating the probability of finding exactly 2 defective components after 2 tests -/
theorem probability_of_two_defective_in_two_tests :
  probability_two_defective_in_two_tests = 1 / 15 :=
by sorry

end probability_of_two_defective_in_two_tests_l2924_292430


namespace score_difference_l2924_292424

theorem score_difference (score : ℕ) (h : score = 15) : 3 * score - 2 * score = 15 := by
  sorry

end score_difference_l2924_292424


namespace weight_difference_l2924_292469

def bridget_weight : ℕ := 39
def martha_weight : ℕ := 2

theorem weight_difference : bridget_weight - martha_weight = 37 := by
  sorry

end weight_difference_l2924_292469


namespace min_side_difference_in_triangle_l2924_292494

theorem min_side_difference_in_triangle (xy xz yz : ℕ) : 
  xy + xz + yz = 3021 →
  xy < xz →
  xz < yz →
  2 ≤ yz - xy :=
by sorry

end min_side_difference_in_triangle_l2924_292494


namespace johns_allowance_l2924_292417

/-- John's weekly allowance problem -/
theorem johns_allowance (A : ℝ) : A = 2.40 :=
  let arcade_spent := (3 : ℝ) / 5 * A
  let remaining_after_arcade := A - arcade_spent
  let toy_store_spent := (1 : ℝ) / 3 * remaining_after_arcade
  let remaining_after_toy_store := remaining_after_arcade - toy_store_spent
  by
    have h1 : remaining_after_toy_store = 0.64
    sorry
    -- Proof goes here
    sorry

#check johns_allowance

end johns_allowance_l2924_292417


namespace bike_rides_total_l2924_292415

/-- The number of times Billy rode his bike -/
def billy_rides : ℕ := 17

/-- The number of times John rode his bike -/
def john_rides : ℕ := 2 * billy_rides

/-- The number of times their mother rode her bike -/
def mother_rides : ℕ := john_rides + 10

/-- The total number of times they rode their bikes -/
def total_rides : ℕ := billy_rides + john_rides + mother_rides

theorem bike_rides_total : total_rides = 95 := by
  sorry

end bike_rides_total_l2924_292415


namespace amanda_peaches_difference_l2924_292466

/-- The number of peaches each person has -/
structure Peaches where
  jill : ℕ
  steven : ℕ
  jake : ℕ
  amanda : ℕ

/-- The conditions of the problem -/
def peach_conditions (p : Peaches) : Prop :=
  p.jill = 12 ∧
  p.steven = p.jill + 15 ∧
  p.jake = p.steven - 16 ∧
  p.amanda = 2 * p.jill

/-- The average number of peaches Jake, Steven, and Jill have -/
def average (p : Peaches) : ℚ :=
  (p.jake + p.steven + p.jill : ℚ) / 3

/-- The theorem to be proved -/
theorem amanda_peaches_difference (p : Peaches) (h : peach_conditions p) :
  p.amanda - average p = 7.33 := by
  sorry

end amanda_peaches_difference_l2924_292466


namespace five_fridays_in_august_l2924_292482

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- Given a month and a day of the week, count how many times that day appears -/
def countDayInMonth (m : Month) (d : DayOfWeek) : Nat :=
  sorry

/-- The next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  sorry

theorem five_fridays_in_august 
  (july : Month)
  (august : Month)
  (h1 : july.days = 31)
  (h2 : august.days = 31)
  (h3 : countDayInMonth july DayOfWeek.Tuesday = 5) :
  countDayInMonth august DayOfWeek.Friday = 5 :=
sorry

end five_fridays_in_august_l2924_292482


namespace fraction_chain_l2924_292464

theorem fraction_chain (a b c d e : ℝ) 
  (h1 : a/b = 5)
  (h2 : b/c = 1/2)
  (h3 : c/d = 4)
  (h4 : d/e = 1/3)
  : e/a = 3/10 := by
  sorry

end fraction_chain_l2924_292464


namespace cubic_equation_root_c_value_l2924_292451

theorem cubic_equation_root_c_value : ∃ (c d : ℚ),
  ((-2 : ℝ) - 3 * Real.sqrt 5) ^ 3 + c * ((-2 : ℝ) - 3 * Real.sqrt 5) ^ 2 + 
  d * ((-2 : ℝ) - 3 * Real.sqrt 5) + 50 = 0 → c = 114 / 41 := by
sorry

end cubic_equation_root_c_value_l2924_292451


namespace penguins_fed_correct_l2924_292438

/-- The number of penguins that have already gotten a fish -/
def penguins_fed (total_penguins : ℕ) (penguins_to_feed : ℕ) : ℕ :=
  total_penguins - penguins_to_feed

theorem penguins_fed_correct (total_fish : ℕ) (total_penguins : ℕ) (penguins_to_feed : ℕ) :
  total_fish = 68 →
  total_penguins = 36 →
  penguins_to_feed = 17 →
  penguins_fed total_penguins penguins_to_feed = 19 :=
by
  sorry

#eval penguins_fed 36 17

end penguins_fed_correct_l2924_292438


namespace min_points_tenth_game_l2924_292410

def points_four_games : List ℕ := [18, 22, 15, 19]

def average_greater_than_19 (total_points : ℕ) : Prop :=
  (total_points : ℚ) / 10 > 19

theorem min_points_tenth_game 
  (h1 : (points_four_games.sum : ℚ) / 4 > (List.sum (List.take 6 points_four_games) : ℚ) / 6)
  (h2 : ∃ (p : ℕ), average_greater_than_19 (points_four_games.sum + List.sum (List.take 6 points_four_games) + p)) :
  ∃ (p : ℕ), p ≥ 9 ∧ average_greater_than_19 (points_four_games.sum + List.sum (List.take 6 points_four_games) + p) ∧
  ∀ (q : ℕ), q < 9 → ¬average_greater_than_19 (points_four_games.sum + List.sum (List.take 6 points_four_games) + q) :=
sorry

end min_points_tenth_game_l2924_292410


namespace total_haircut_time_l2924_292473

/-- The time it takes to cut a woman's hair in minutes -/
def womanHairCutTime : ℕ := 50

/-- The time it takes to cut a man's hair in minutes -/
def manHairCutTime : ℕ := 15

/-- The time it takes to cut a kid's hair in minutes -/
def kidHairCutTime : ℕ := 25

/-- The number of women's haircuts Joe performed -/
def numWomenHaircuts : ℕ := 3

/-- The number of men's haircuts Joe performed -/
def numMenHaircuts : ℕ := 2

/-- The number of kids' haircuts Joe performed -/
def numKidsHaircuts : ℕ := 3

/-- Theorem stating the total time Joe spent cutting hair -/
theorem total_haircut_time :
  numWomenHaircuts * womanHairCutTime +
  numMenHaircuts * manHairCutTime +
  numKidsHaircuts * kidHairCutTime = 255 := by
  sorry

end total_haircut_time_l2924_292473


namespace ellen_legos_l2924_292465

theorem ellen_legos (initial_legos : ℕ) (lost_legos : ℕ) 
  (h1 : initial_legos = 2080) 
  (h2 : lost_legos = 17) : 
  initial_legos - lost_legos = 2063 := by
sorry

end ellen_legos_l2924_292465


namespace probability_cousins_names_l2924_292447

/-- Represents the number of letters in each cousin's name -/
structure NameLengths where
  amelia : ℕ
  bethany : ℕ
  claire : ℕ

/-- The probability of selecting two cards from different cousins' names -/
def probability_different_names (nl : NameLengths) : ℚ :=
  let total := nl.amelia + nl.bethany + nl.claire
  2 * (nl.amelia * nl.bethany + nl.amelia * nl.claire + nl.bethany * nl.claire) / (total * (total - 1))

/-- Theorem stating the probability of selecting two cards from different cousins' names -/
theorem probability_cousins_names :
  let nl : NameLengths := { amelia := 6, bethany := 7, claire := 6 }
  probability_different_names nl = 40 / 57 := by
  sorry


end probability_cousins_names_l2924_292447


namespace ellipse_and_rhombus_problem_l2924_292443

-- Define the ellipse C₁
def C₁ (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the parabola C₂
def C₂ (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the line BD
def BD (x y : ℝ) : Prop := 7 * x - 7 * y + 1 = 0

-- Define the rhombus ABCD
structure Rhombus where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Main theorem
theorem ellipse_and_rhombus_problem 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a > b) 
  (F₁ F₂ M : ℝ × ℝ) 
  (hF₂ : F₂ = (1, 0)) 
  (hM : C₁ a b M.1 M.2 ∧ C₂ M.1 M.2) 
  (hMF₂ : Real.sqrt ((M.1 - F₂.1)^2 + (M.2 - F₂.2)^2) = 2) 
  (ABCD : Rhombus) 
  (hAC : C₁ a b ABCD.A.1 ABCD.A.2 ∧ C₁ a b ABCD.C.1 ABCD.C.2) 
  (hBD : BD ABCD.B.1 ABCD.B.2 ∧ BD ABCD.D.1 ABCD.D.2) :
  (∀ x y, C₁ a b x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (ABCD.A.2 = -ABCD.A.1 - 1/14 ∧ ABCD.C.2 = -ABCD.C.1 - 1/14) := by
  sorry

end ellipse_and_rhombus_problem_l2924_292443


namespace power_five_addition_l2924_292432

theorem power_five_addition (a : ℝ) : a^5 + a^5 = 2*a^5 := by
  sorry

end power_five_addition_l2924_292432


namespace unique_solution_cube_root_system_l2924_292487

theorem unique_solution_cube_root_system :
  ∃! (x y z : ℝ),
    Real.sqrt (x^3 - y) = z - 1 ∧
    Real.sqrt (y^3 - z) = x - 1 ∧
    Real.sqrt (z^3 - x) = y - 1 ∧
    x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end unique_solution_cube_root_system_l2924_292487


namespace zero_last_to_appear_l2924_292474

/-- Modified Fibonacci sequence -/
def modFib : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => modFib (n + 1) + modFib n

/-- The set of digits that have appeared in the units position up to the nth term -/
def digitsAppeared (n : ℕ) : Finset ℕ :=
  Finset.filter (fun d => ∃ k ≤ n, modFib k % 10 = d) (Finset.range 10)

/-- The proposition that 0 is the last digit to appear in the units position -/
theorem zero_last_to_appear : ∃ N : ℕ, 
  (∀ n ≥ N, 0 ∈ digitsAppeared n) ∧ 
  (∀ d : ℕ, d < 10 → d ≠ 0 → ∃ n < N, d ∈ digitsAppeared n) :=
sorry

end zero_last_to_appear_l2924_292474


namespace chicken_difference_l2924_292456

theorem chicken_difference (mary john ray : ℕ) 
  (h1 : john = mary + 5)
  (h2 : ray + 6 = mary)
  (h3 : ray = 10) : 
  john - ray = 11 :=
by sorry

end chicken_difference_l2924_292456


namespace rectangular_field_area_l2924_292484

/-- Given a rectangular field with one uncovered side of 30 feet and three sides
    requiring 70 feet of fencing, the area of the field is 600 square feet. -/
theorem rectangular_field_area (L W : ℝ) : 
  L = 30 →
  L + 2 * W = 70 →
  L * W = 600 := by sorry

end rectangular_field_area_l2924_292484


namespace uncle_dave_ice_cream_l2924_292409

/-- The number of ice cream sandwiches Uncle Dave bought -/
def total_ice_cream_sandwiches : ℕ := sorry

/-- The number of Uncle Dave's nieces -/
def number_of_nieces : ℕ := 11

/-- The number of ice cream sandwiches each niece would get -/
def sandwiches_per_niece : ℕ := 13

/-- Theorem stating that the total number of ice cream sandwiches is 143 -/
theorem uncle_dave_ice_cream : total_ice_cream_sandwiches = number_of_nieces * sandwiches_per_niece := by
  sorry

end uncle_dave_ice_cream_l2924_292409


namespace equation_solution_l2924_292416

theorem equation_solution : 
  let eq := fun x : ℝ => 81 * (1 - x)^2 - 64
  ∃ (x1 x2 : ℝ), x1 = 1/9 ∧ x2 = 17/9 ∧ eq x1 = 0 ∧ eq x2 = 0 ∧
  ∀ (x : ℝ), eq x = 0 → x = x1 ∨ x = x2 :=
by
  sorry

#check equation_solution

end equation_solution_l2924_292416
