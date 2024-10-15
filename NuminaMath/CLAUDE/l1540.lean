import Mathlib

namespace NUMINAMATH_CALUDE_regression_y_change_l1540_154081

/-- Represents a simple linear regression equation -/
structure LinearRegression where
  intercept : ℝ
  slope : ℝ

/-- Represents the change in y for a unit change in x -/
def yChange (reg : LinearRegression) : ℝ := -reg.slope

theorem regression_y_change (reg : LinearRegression) 
  (h : reg = { intercept := 3, slope := 5 }) : 
  yChange reg = -5 := by sorry

end NUMINAMATH_CALUDE_regression_y_change_l1540_154081


namespace NUMINAMATH_CALUDE_product_of_good_is_good_l1540_154056

/-- A positive integer is good if it can be represented as ax^2 + bxy + cy^2 
    with b^2 - 4ac = -20 for some integers a, b, c, x, y -/
def is_good (n : ℕ+) : Prop :=
  ∃ (a b c x y : ℤ), (n : ℤ) = a * x^2 + b * x * y + c * y^2 ∧ b^2 - 4 * a * c = -20

/-- The product of two good numbers is also a good number -/
theorem product_of_good_is_good (n1 n2 : ℕ+) (h1 : is_good n1) (h2 : is_good n2) :
  is_good (n1 * n2) :=
sorry

end NUMINAMATH_CALUDE_product_of_good_is_good_l1540_154056


namespace NUMINAMATH_CALUDE_window_length_l1540_154080

/-- Given a rectangular window with width 10 feet and area 60 square feet, its length is 6 feet -/
theorem window_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 10 → area = 60 → area = length * width → length = 6 := by
  sorry

end NUMINAMATH_CALUDE_window_length_l1540_154080


namespace NUMINAMATH_CALUDE_max_divisors_sympathetic_l1540_154061

/-- A number is sympathetic if for each of its divisors d, d+2 is prime. -/
def Sympathetic (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → Nat.Prime (d + 2)

/-- The maximum number of divisors a sympathetic number can have is 8. -/
theorem max_divisors_sympathetic :
  ∃ n : ℕ, Sympathetic n ∧ (∀ m : ℕ, Sympathetic m → Nat.card (Nat.divisors m) ≤ Nat.card (Nat.divisors n)) ∧
    Nat.card (Nat.divisors n) = 8 :=
sorry

end NUMINAMATH_CALUDE_max_divisors_sympathetic_l1540_154061


namespace NUMINAMATH_CALUDE_segments_can_be_commensurable_l1540_154022

/-- Represents a geometric segment -/
structure Segment where
  length : ℝ
  pos : length > 0

/-- Two segments are commensurable if their ratio is rational -/
def commensurable (a b : Segment) : Prop :=
  ∃ (q : ℚ), a.length = q * b.length

/-- Segment m fits into a an integer number of times -/
def fits_integer_times (m a : Segment) : Prop :=
  ∃ (k : ℤ), a.length = k * m.length

/-- No segment m/(10^n) fits into b an integer number of times -/
def no_submultiple_fits (m b : Segment) : Prop :=
  ∀ (n : ℕ), ¬∃ (j : ℤ), b.length = j * (m.length / (10^n : ℝ))

theorem segments_can_be_commensurable
  (a b m : Segment)
  (h1 : fits_integer_times m a)
  (h2 : no_submultiple_fits m b) :
  commensurable a b :=
sorry

end NUMINAMATH_CALUDE_segments_can_be_commensurable_l1540_154022


namespace NUMINAMATH_CALUDE_lose_condition_win_condition_rattle_count_l1540_154077

/-- The number of rattles Twalley has -/
def t : ℕ := 7

/-- The number of rattles Tweerley has -/
def r : ℕ := 5

/-- If Twalley loses the bet, he will have the same number of rattles as Tweerley -/
theorem lose_condition : t - 1 = r + 1 := by sorry

/-- If Twalley wins the bet, he will have twice as many rattles as Tweerley -/
theorem win_condition : t + 1 = 2 * (r - 1) := by sorry

/-- Prove that given the conditions of the bet, Twalley must have 7 rattles and Tweerley must have 5 rattles -/
theorem rattle_count : t = 7 ∧ r = 5 := by sorry

end NUMINAMATH_CALUDE_lose_condition_win_condition_rattle_count_l1540_154077


namespace NUMINAMATH_CALUDE_rectangular_cards_are_squares_l1540_154096

/-- Represents a rectangular card with dimensions width and height -/
structure Card where
  width : ℕ+
  height : ℕ+

/-- Represents the result of a child cutting their card into squares -/
structure CutResult where
  squareCount : ℕ+

theorem rectangular_cards_are_squares
  (n : ℕ+)
  (h_n : n > 1)
  (cards : Fin n → Card)
  (h_identical : ∀ i j : Fin n, cards i = cards j)
  (cuts : Fin n → CutResult)
  (h_prime_total : Nat.Prime (Finset.sum (Finset.range n) (λ i => (cuts i).squareCount))) :
  ∀ i : Fin n, (cards i).width = (cards i).height :=
sorry

end NUMINAMATH_CALUDE_rectangular_cards_are_squares_l1540_154096


namespace NUMINAMATH_CALUDE_medium_pizzas_ordered_l1540_154069

/-- Represents the number of slices in different pizza sizes --/
structure PizzaSlices where
  small : Nat
  medium : Nat
  large : Nat

/-- Represents the number of pizzas ordered --/
structure PizzaOrder where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculates the total number of slices for a given order --/
def totalSlices (slices : PizzaSlices) (order : PizzaOrder) : Nat :=
  slices.small * order.small + slices.medium * order.medium + slices.large * order.large

/-- The main theorem to prove --/
theorem medium_pizzas_ordered 
  (slices : PizzaSlices) 
  (order : PizzaOrder) 
  (h1 : slices.small = 6)
  (h2 : slices.medium = 8)
  (h3 : slices.large = 12)
  (h4 : order.small + order.medium + order.large = 15)
  (h5 : order.small = 4)
  (h6 : totalSlices slices order = 136) :
  order.medium = 5 := by
  sorry

end NUMINAMATH_CALUDE_medium_pizzas_ordered_l1540_154069


namespace NUMINAMATH_CALUDE_photo_shoot_count_l1540_154055

/-- The number of photos taken during a photo shoot, given initial conditions and final count --/
theorem photo_shoot_count (initial : ℕ) (deleted_first : ℕ) (added_first : ℕ)
  (deleted_friend1 : ℕ) (added_friend1 : ℕ)
  (deleted_friend2 : ℕ) (added_friend2 : ℕ)
  (added_friend3 : ℕ)
  (deleted_last : ℕ) (final : ℕ) :
  initial = 63 →
  deleted_first = 7 →
  added_first = 15 →
  deleted_friend1 = 3 →
  added_friend1 = 5 →
  deleted_friend2 = 1 →
  added_friend2 = 4 →
  added_friend3 = 6 →
  deleted_last = 2 →
  final = 112 →
  ∃ x : ℕ, x = 32 ∧
    final = initial - deleted_first + added_first + x - deleted_friend1 + added_friend1 - deleted_friend2 + added_friend2 + added_friend3 - deleted_last :=
by sorry

end NUMINAMATH_CALUDE_photo_shoot_count_l1540_154055


namespace NUMINAMATH_CALUDE_solve_textbook_problems_l1540_154087

/-- The number of days it takes to solve all problems -/
def solve_duration (total_problems : ℕ) (problems_left_day3 : ℕ) : ℕ :=
  let problems_solved_day3 := total_problems - problems_left_day3
  let z := problems_solved_day3 / 3
  let daily_problems := List.range 7 |>.map (fun i => z + 1 - i)
  daily_problems.length

/-- Theorem stating that it takes 7 days to solve all problems under given conditions -/
theorem solve_textbook_problems :
  solve_duration 91 46 = 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_textbook_problems_l1540_154087


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l1540_154007

theorem negation_of_universal_statement :
  ¬(∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) ↔ ∃ x : ℝ, x^2 + 2*x + 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l1540_154007


namespace NUMINAMATH_CALUDE_solve_pq_system_l1540_154052

theorem solve_pq_system (p q : ℝ) (hp : p > 1) (hq : q > 1) 
  (h1 : 1/p + 1/q = 1) (h2 : p * q = 9) : 
  q = (9 + 3 * Real.sqrt 5) / 2 ∨ q = (9 - 3 * Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_pq_system_l1540_154052


namespace NUMINAMATH_CALUDE_inequality_solution_l1540_154028

theorem inequality_solution (x : ℝ) : 
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1 / 4) ↔ 
  (x < -2 ∨ (-1 < x ∧ x < 0) ∨ 2 < x) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1540_154028


namespace NUMINAMATH_CALUDE_largest_alpha_l1540_154060

theorem largest_alpha : ∃ (α : ℝ), (α = 3) ∧ 
  (∀ (m n : ℕ+), (m : ℝ) / n < Real.sqrt 7 → α / n^2 ≤ 7 - (m : ℝ)^2 / n^2) ∧
  (∀ (β : ℝ), β > α → 
    ∃ (m n : ℕ+), (m : ℝ) / n < Real.sqrt 7 ∧ β / n^2 > 7 - (m : ℝ)^2 / n^2) :=
sorry

end NUMINAMATH_CALUDE_largest_alpha_l1540_154060


namespace NUMINAMATH_CALUDE_third_day_temperature_l1540_154029

/-- Given three temperatures in Fahrenheit, calculates their average -/
def average (t1 t2 t3 : ℚ) : ℚ := (t1 + t2 + t3) / 3

/-- Proves that given an average temperature of -7°F for three days, 
    with temperatures of -8°F and +1°F on two of the days, 
    the temperature on the third day must be -14°F -/
theorem third_day_temperature 
  (t1 t2 t3 : ℚ) 
  (h1 : t1 = -8)
  (h2 : t2 = 1)
  (h_avg : average t1 t2 t3 = -7) :
  t3 = -14 := by
  sorry

#eval average (-8) 1 (-14) -- Should output -7

end NUMINAMATH_CALUDE_third_day_temperature_l1540_154029


namespace NUMINAMATH_CALUDE_masha_floor_number_l1540_154057

/-- Represents a multi-story apartment building -/
structure ApartmentBuilding where
  floors : ℕ
  entrances : ℕ
  apartments_per_floor : ℕ

/-- Calculates the floor number given an apartment number and building structure -/
def floor_number (building : ApartmentBuilding) (apartment_number : ℕ) : ℕ :=
  sorry

theorem masha_floor_number :
  let building := ApartmentBuilding.mk 17 4 5
  let masha_apartment := 290
  floor_number building masha_apartment = 7 := by
  sorry

end NUMINAMATH_CALUDE_masha_floor_number_l1540_154057


namespace NUMINAMATH_CALUDE_tax_calculation_l1540_154009

/-- Calculate the tax amount given gross pay and net pay -/
def calculate_tax (gross_pay : ℝ) (net_pay : ℝ) : ℝ :=
  gross_pay - net_pay

theorem tax_calculation :
  let gross_pay : ℝ := 450
  let net_pay : ℝ := 315
  calculate_tax gross_pay net_pay = 135 := by
sorry

end NUMINAMATH_CALUDE_tax_calculation_l1540_154009


namespace NUMINAMATH_CALUDE_choir_average_age_l1540_154016

/-- The average age of people in a choir given the number and average age of females and males -/
theorem choir_average_age 
  (num_females : ℕ) 
  (num_males : ℕ) 
  (avg_age_females : ℚ) 
  (avg_age_males : ℚ) 
  (h1 : num_females = 12) 
  (h2 : num_males = 18) 
  (h3 : avg_age_females = 28) 
  (h4 : avg_age_males = 32) :
  let total_people := num_females + num_males
  let total_age := num_females * avg_age_females + num_males * avg_age_males
  total_age / total_people = 30.4 := by
sorry


end NUMINAMATH_CALUDE_choir_average_age_l1540_154016


namespace NUMINAMATH_CALUDE_p_or_q_iff_m_in_range_l1540_154051

def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 - m*x + 3/2 > 0

def q (m : ℝ) : Prop := 
  (m - 1 > 0) ∧ (3 - m > 0) ∧ 
  ∃ c : ℝ, c^2 = (m - 1)*(3 - m) ∧ 
  ∀ x y : ℝ, x^2/(m-1) + y^2/(3-m) = 1 → x^2 + y^2 = (m-1)^2/(m-1) ∨ x^2 + y^2 = (3-m)^2/(3-m)

theorem p_or_q_iff_m_in_range (m : ℝ) : 
  p m ∨ q m ↔ m > -Real.sqrt 6 ∧ m < 3 :=
sorry

end NUMINAMATH_CALUDE_p_or_q_iff_m_in_range_l1540_154051


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_three_l1540_154068

theorem fraction_zero_implies_x_equals_three (x : ℝ) :
  (x^2 - 9) / (x + 3) = 0 ∧ x + 3 ≠ 0 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_three_l1540_154068


namespace NUMINAMATH_CALUDE_local_minimum_at_two_l1540_154036

-- Define the function f
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

-- State the theorem
theorem local_minimum_at_two (c : ℝ) : 
  (∀ h, h > 0 → ∃ δ > 0, ∀ x, 0 < |x - 2| ∧ |x - 2| < δ → f c x ≥ f c 2) → 
  c = 2 := by
  sorry

end NUMINAMATH_CALUDE_local_minimum_at_two_l1540_154036


namespace NUMINAMATH_CALUDE_triangle_cosine_identity_l1540_154043

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  -- Ensure A + B + C = π (180 degrees)
  angle_sum : A + B + C = Real.pi
  -- Ensure all angles are positive
  A_pos : A > 0
  B_pos : B > 0
  C_pos : C > 0
  -- Ensure the given condition 2b = a + c
  side_condition : 2 * b = a + c

-- Theorem statement
theorem triangle_cosine_identity (t : Triangle) :
  5 * Real.cos t.A - 4 * Real.cos t.A * Real.cos t.C + 5 * Real.cos t.C = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_identity_l1540_154043


namespace NUMINAMATH_CALUDE_role_assignment_combinations_l1540_154054

def number_of_friends : ℕ := 6

theorem role_assignment_combinations (maria_is_cook : Bool) 
  (h1 : maria_is_cook = true) 
  (h2 : number_of_friends = 6) : 
  (Nat.choose (number_of_friends - 1) 1) * (Nat.choose (number_of_friends - 2) 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_role_assignment_combinations_l1540_154054


namespace NUMINAMATH_CALUDE_log_sum_equality_l1540_154040

theorem log_sum_equality : Real.log 50 + Real.log 30 = 3 + Real.log 1.5 := by sorry

end NUMINAMATH_CALUDE_log_sum_equality_l1540_154040


namespace NUMINAMATH_CALUDE_union_of_sets_l1540_154034

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {2, 3, 4}
  A ∪ B = {1, 2, 3, 4} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l1540_154034


namespace NUMINAMATH_CALUDE_semicircles_to_circle_area_ratio_l1540_154041

/-- The ratio of the combined areas of two semicircles with radius r/2 inscribed in a circle with radius r to the area of the circle is 1/4. -/
theorem semicircles_to_circle_area_ratio (r : ℝ) (h : r > 0) : 
  (2 * (π * (r/2)^2 / 2)) / (π * r^2) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_semicircles_to_circle_area_ratio_l1540_154041


namespace NUMINAMATH_CALUDE_cookie_batches_for_workshop_l1540_154049

/-- Calculates the minimum number of full batches of cookies needed for a math competition workshop --/
def min_cookie_batches (base_students : ℕ) (additional_students : ℕ) (cookies_per_student : ℕ) (cookies_per_batch : ℕ) : ℕ :=
  let total_students := base_students + additional_students
  let total_cookies_needed := total_students * cookies_per_student
  (total_cookies_needed + cookies_per_batch - 1) / cookies_per_batch

/-- Proves that 16 batches are needed for the given conditions --/
theorem cookie_batches_for_workshop : 
  min_cookie_batches 90 15 3 20 = 16 := by
sorry

end NUMINAMATH_CALUDE_cookie_batches_for_workshop_l1540_154049


namespace NUMINAMATH_CALUDE_two_layer_coverage_is_zero_l1540_154078

/-- Represents the area covered by rugs with different layers of overlap -/
structure RugCoverage where
  total_rug_area : ℝ
  total_floor_coverage : ℝ
  multilayer_coverage : ℝ
  three_layer_coverage : ℝ

/-- Calculates the area covered by exactly two layers of rug -/
def two_layer_coverage (rc : RugCoverage) : ℝ :=
  rc.multilayer_coverage - rc.three_layer_coverage

/-- Theorem stating that under the given conditions, the area covered by exactly two layers of rug is 0 -/
theorem two_layer_coverage_is_zero (rc : RugCoverage)
  (h1 : rc.total_rug_area = 212)
  (h2 : rc.total_floor_coverage = 140)
  (h3 : rc.multilayer_coverage = 24)
  (h4 : rc.three_layer_coverage = 24) :
  two_layer_coverage rc = 0 := by
  sorry

end NUMINAMATH_CALUDE_two_layer_coverage_is_zero_l1540_154078


namespace NUMINAMATH_CALUDE_computers_probability_l1540_154092

def CAMPUS : Finset Char := {'C', 'A', 'M', 'P', 'U', 'S'}
def THREADS : Finset Char := {'T', 'H', 'R', 'E', 'A', 'D', 'S'}
def GLOW : Finset Char := {'G', 'L', 'O', 'W'}
def COMPUTERS : Finset Char := {'C', 'O', 'M', 'P', 'U', 'T', 'E', 'R', 'S'}

def probability_CAMPUS : ℚ := 1 / (CAMPUS.card.choose 3)
def probability_THREADS : ℚ := 1 / (THREADS.card.choose 5)
def probability_GLOW : ℚ := (GLOW.card - 1).choose 1 / (GLOW.card.choose 2)

theorem computers_probability :
  probability_CAMPUS * probability_THREADS * probability_GLOW = 1 / 840 := by
  sorry

end NUMINAMATH_CALUDE_computers_probability_l1540_154092


namespace NUMINAMATH_CALUDE_money_ratio_l1540_154097

theorem money_ratio (bob phil jenna : ℚ) : 
  bob = 60 →
  phil = (1/3) * bob →
  jenna = bob - 20 →
  jenna / phil = 2 := by
sorry

end NUMINAMATH_CALUDE_money_ratio_l1540_154097


namespace NUMINAMATH_CALUDE_a_in_range_l1540_154031

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem a_in_range (a : ℝ) (h : a ∈ A) : -1 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_a_in_range_l1540_154031


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l1540_154019

/-- Given a car's speed in the first hour and its average speed over two hours,
    calculate its speed in the second hour. -/
theorem car_speed_second_hour
  (speed_first_hour : ℝ)
  (average_speed : ℝ)
  (h1 : speed_first_hour = 90)
  (h2 : average_speed = 65)
  : ∃ (speed_second_hour : ℝ),
    speed_second_hour = 40 ∧
    (speed_first_hour + speed_second_hour) / 2 = average_speed :=
by
  sorry

#check car_speed_second_hour

end NUMINAMATH_CALUDE_car_speed_second_hour_l1540_154019


namespace NUMINAMATH_CALUDE_good_goods_sufficient_for_not_cheap_l1540_154018

-- Define the propositions
def good_goods : Prop := sorry
def not_cheap : Prop := sorry

-- Define Sister Qian's statement
def sister_qian_statement : Prop := good_goods → not_cheap

-- Theorem to prove
theorem good_goods_sufficient_for_not_cheap :
  sister_qian_statement → (∃ p q : Prop, (p → q) ∧ (p = good_goods) ∧ (q = not_cheap)) :=
by sorry

end NUMINAMATH_CALUDE_good_goods_sufficient_for_not_cheap_l1540_154018


namespace NUMINAMATH_CALUDE_sequence_seventh_term_l1540_154024

theorem sequence_seventh_term : 
  ∀ (a : ℕ → ℕ), 
    a 1 = 1 ∧ 
    (∀ n : ℕ, a (n + 1) = 2 * a n + 2) → 
    a 7 = 190 := by
  sorry

end NUMINAMATH_CALUDE_sequence_seventh_term_l1540_154024


namespace NUMINAMATH_CALUDE_tangent_fraction_equality_l1540_154083

theorem tangent_fraction_equality (α : Real) (h : Real.tan α = 3) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 3 * Real.cos α) = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_tangent_fraction_equality_l1540_154083


namespace NUMINAMATH_CALUDE_smallest_k_divisible_by_power_of_three_l1540_154015

theorem smallest_k_divisible_by_power_of_three : ∃ k : ℕ, 
  (∀ m : ℕ, m < k → ¬(3^67 ∣ 2016^m)) ∧ (3^67 ∣ 2016^k) ∧ k = 34 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_divisible_by_power_of_three_l1540_154015


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1540_154030

theorem polynomial_factorization (x y : ℝ) :
  -2 * x^2 * y + 8 * x * y - 6 * y = -2 * y * (x - 1) * (x - 3) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1540_154030


namespace NUMINAMATH_CALUDE_substitution_remainder_l1540_154098

/-- Number of players in a soccer team --/
def total_players : ℕ := 22

/-- Number of starting players --/
def starting_players : ℕ := 11

/-- Number of substitute players --/
def substitute_players : ℕ := 11

/-- Maximum number of substitutions allowed --/
def max_substitutions : ℕ := 4

/-- Function to calculate the number of ways to make k substitutions --/
def substitution_ways (k : ℕ) : ℕ :=
  match k with
  | 0 => 1
  | 1 => starting_players * substitute_players
  | k+1 => starting_players * (substitute_players - k) * substitution_ways k

/-- Total number of substitution scenarios --/
def total_scenarios : ℕ :=
  List.sum (List.map substitution_ways (List.range (max_substitutions + 1)))

/-- Theorem stating the remainder when total scenarios is divided by 2000 --/
theorem substitution_remainder :
  total_scenarios % 2000 = 942 := by sorry

end NUMINAMATH_CALUDE_substitution_remainder_l1540_154098


namespace NUMINAMATH_CALUDE_igor_lied_l1540_154033

-- Define the set of boys
inductive Boy : Type
| andrey : Boy
| maxim : Boy
| igor : Boy
| kolya : Boy

-- Define the possible positions in the race
inductive Position : Type
| first : Position
| second : Position
| third : Position
| fourth : Position

-- Define a function to represent the actual position of each boy
def actual_position : Boy → Position := sorry

-- Define a function to represent whether a boy is telling the truth
def is_truthful : Boy → Prop := sorry

-- State the conditions of the problem
axiom three_truthful : ∃ (a b c : Boy), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  is_truthful a ∧ is_truthful b ∧ is_truthful c ∧ 
  ∀ (d : Boy), d ≠ a ∧ d ≠ b ∧ d ≠ c → ¬is_truthful d

axiom andrey_claim : is_truthful Boy.andrey ↔ 
  actual_position Boy.andrey ≠ Position.first ∧ 
  actual_position Boy.andrey ≠ Position.fourth

axiom maxim_claim : is_truthful Boy.maxim ↔ 
  actual_position Boy.maxim ≠ Position.fourth

axiom igor_claim : is_truthful Boy.igor ↔ 
  actual_position Boy.igor = Position.first

axiom kolya_claim : is_truthful Boy.kolya ↔ 
  actual_position Boy.kolya = Position.fourth

-- Theorem to prove
theorem igor_lied : ¬is_truthful Boy.igor := by sorry

end NUMINAMATH_CALUDE_igor_lied_l1540_154033


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l1540_154074

/-- An arithmetic sequence with specified properties -/
structure ArithmeticSequence where
  /-- The sequence terms -/
  a : ℕ → ℝ
  /-- The number of terms -/
  n : ℕ
  /-- Sum of first 3 terms is 20 -/
  first_three_sum : a 1 + a 2 + a 3 = 20
  /-- Sum of last 3 terms is 130 -/
  last_three_sum : a (n - 2) + a (n - 1) + a n = 130
  /-- Sum of all terms is 200 -/
  total_sum : (Finset.range n).sum a = 200

/-- The number of terms in the arithmetic sequence is 8 -/
theorem arithmetic_sequence_length (seq : ArithmeticSequence) : seq.n = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l1540_154074


namespace NUMINAMATH_CALUDE_largest_number_l1540_154044

def a : ℚ := 8.23455
def b : ℚ := 8 + 234 / 1000 + 5 / 9000
def c : ℚ := 8 + 23 / 100 + 45 / 9900
def d : ℚ := 8 + 2 / 10 + 345 / 999
def e : ℚ := 8 + 2345 / 9999

theorem largest_number : b > a ∧ b > c ∧ b > d ∧ b > e := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l1540_154044


namespace NUMINAMATH_CALUDE_squats_on_fourth_day_l1540_154053

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def squats_on_day (initial_squats : ℕ) (day : ℕ) : ℕ :=
  match day with
  | 0 => initial_squats
  | n + 1 => squats_on_day initial_squats n + factorial n

theorem squats_on_fourth_day (initial_squats : ℕ) :
  initial_squats = 30 → squats_on_day initial_squats 3 = 39 := by
  sorry

end NUMINAMATH_CALUDE_squats_on_fourth_day_l1540_154053


namespace NUMINAMATH_CALUDE_sin_cubed_identity_l1540_154091

theorem sin_cubed_identity (θ : Real) : 
  Real.sin θ ^ 3 = -1/4 * Real.sin (3 * θ) + 3/4 * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_sin_cubed_identity_l1540_154091


namespace NUMINAMATH_CALUDE_divisibility_condition_l1540_154017

theorem divisibility_condition (M : ℕ) : 
  M > 0 ∧ M < 10 →
  (5 ∣ (1989^M + M^1989)) ↔ (M = 1 ∨ M = 4) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1540_154017


namespace NUMINAMATH_CALUDE_additional_round_trips_l1540_154038

/-- Represents the number of passengers on a one-way trip -/
def one_way_passengers : ℕ := 100

/-- Represents the number of passengers on a return trip -/
def return_passengers : ℕ := 60

/-- Represents the total number of passengers transported that day -/
def total_passengers : ℕ := 640

/-- Calculates the number of passengers in one round trip -/
def passengers_per_round_trip : ℕ := one_way_passengers + return_passengers

/-- Theorem: The number of additional round trips is 3 -/
theorem additional_round_trips :
  (total_passengers - passengers_per_round_trip) / passengers_per_round_trip = 3 := by
  sorry

end NUMINAMATH_CALUDE_additional_round_trips_l1540_154038


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1540_154084

theorem quadratic_minimum : 
  (∃ (y : ℝ), y^2 - 6*y + 5 = -4) ∧ 
  (∀ (y : ℝ), y^2 - 6*y + 5 ≥ -4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1540_154084


namespace NUMINAMATH_CALUDE_circle_area_special_condition_l1540_154093

theorem circle_area_special_condition (r : ℝ) (h : (2 * r)^2 = 8 * (2 * π * r)) :
  π * r^2 = 16 * π^3 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_special_condition_l1540_154093


namespace NUMINAMATH_CALUDE_problem_solution_l1540_154021

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log (exp x + a)

def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem problem_solution :
  (∃ a, is_odd (f a)) →
  (∃ a, a = 0 ∧ is_odd (f a)) ∧
  (∀ m : ℝ,
    (m > 1/exp 1 + exp 2 → ¬∃ x, (log x) / x = x^2 - 2 * (exp 1) * x + m) ∧
    (m = 1/exp 1 + exp 2 → ∃! x, x = exp 1 ∧ (log x) / x = x^2 - 2 * (exp 1) * x + m) ∧
    (m < 1/exp 1 + exp 2 → ∃ x y, x ≠ y ∧ (log x) / x = x^2 - 2 * (exp 1) * x + m ∧ (log y) / y = y^2 - 2 * (exp 1) * y + m))
    := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1540_154021


namespace NUMINAMATH_CALUDE_school_pupils_l1540_154045

theorem school_pupils (girls : ℕ) (boys : ℕ) (h1 : girls = 692) (h2 : girls = boys + 458) :
  girls + boys = 926 := by
  sorry

end NUMINAMATH_CALUDE_school_pupils_l1540_154045


namespace NUMINAMATH_CALUDE_first_interest_rate_is_five_percent_l1540_154062

/-- Proves that the first interest rate is 5% given the problem conditions --/
theorem first_interest_rate_is_five_percent
  (total_amount : ℝ)
  (first_part : ℝ)
  (second_part : ℝ)
  (second_interest_rate : ℝ)
  (total_income : ℝ)
  (h1 : total_amount = 2500)
  (h2 : first_part = 1000)
  (h3 : second_part = total_amount - first_part)
  (h4 : second_interest_rate = 6)
  (h5 : total_income = 140)
  (h6 : total_income = (first_part * first_interest_rate / 100) + (second_part * second_interest_rate / 100)) :
  first_interest_rate = 5 := by
  sorry

#check first_interest_rate_is_five_percent

end NUMINAMATH_CALUDE_first_interest_rate_is_five_percent_l1540_154062


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1540_154072

-- Problem 1
theorem problem_1 : (π - 2023) ^ 0 - 3 * Real.tan (π / 6) + |1 - Real.sqrt 3| = 0 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) :
  ((2 * x + 1) / (x - 1) - 1) / ((2 * x + x^2) / (x^2 - 2 * x + 1)) = (x - 1) / x := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1540_154072


namespace NUMINAMATH_CALUDE_chris_age_l1540_154014

theorem chris_age (a b c : ℕ) : 
  (a + b + c) / 3 = 9 →  -- The average of their ages is 9
  c - 4 = a →            -- Four years ago, Chris was Amy's current age
  b + 3 = 2 * (a + 3) / 3 →  -- In 3 years, Ben's age will be 2/3 of Amy's age
  c = 13 :=               -- Chris's current age is 13
by sorry

end NUMINAMATH_CALUDE_chris_age_l1540_154014


namespace NUMINAMATH_CALUDE_factorization_equality_l1540_154089

theorem factorization_equality (x y : ℝ) : y - 2*x*y + x^2*y = y*(1-x)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1540_154089


namespace NUMINAMATH_CALUDE_boat_trip_time_l1540_154071

theorem boat_trip_time (v : ℝ) :
  (90 = (v - 3) * (T + 0.5)) →
  (90 = (v + 3) * T) →
  (T > 0) →
  T = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_boat_trip_time_l1540_154071


namespace NUMINAMATH_CALUDE_william_bottle_caps_l1540_154058

/-- The number of bottle caps William has in total -/
def total_bottle_caps (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that William has 43 bottle caps in total -/
theorem william_bottle_caps : 
  total_bottle_caps 2 41 = 43 := by
  sorry

end NUMINAMATH_CALUDE_william_bottle_caps_l1540_154058


namespace NUMINAMATH_CALUDE_intersection_A_B_union_complement_A_B_l1540_154000

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | |x - 1| < 2}

-- Define set B
def B : Set ℝ := {x | x^2 - 3*x + 2 > 0}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x | -1 < x ∧ x < 1 ∨ 2 < x ∧ x < 3} := by sorry

-- Theorem for (C_U A) ∪ B
theorem union_complement_A_B : (Set.compl A) ∪ B = {x | x < 1 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_complement_A_B_l1540_154000


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l1540_154020

theorem rectangular_to_polar_conversion :
  let x : ℝ := 3
  let y : ℝ := -3
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := if x > 0 ∧ y < 0 then 2 * π + Real.arctan (y / x) else Real.arctan (y / x)
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π →
  r = 3 * Real.sqrt 2 ∧ θ = 7 * π / 4 := by sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l1540_154020


namespace NUMINAMATH_CALUDE_problems_left_to_grade_l1540_154065

/-- Calculates the number of problems left to grade for a teacher grading worksheets from three subjects. -/
theorem problems_left_to_grade
  (math_problems_per_sheet : ℕ)
  (science_problems_per_sheet : ℕ)
  (english_problems_per_sheet : ℕ)
  (total_math_sheets : ℕ)
  (total_science_sheets : ℕ)
  (total_english_sheets : ℕ)
  (graded_math_sheets : ℕ)
  (graded_science_sheets : ℕ)
  (graded_english_sheets : ℕ)
  (h_math : math_problems_per_sheet = 5)
  (h_science : science_problems_per_sheet = 3)
  (h_english : english_problems_per_sheet = 7)
  (h_total_math : total_math_sheets = 10)
  (h_total_science : total_science_sheets = 15)
  (h_total_english : total_english_sheets = 12)
  (h_graded_math : graded_math_sheets = 6)
  (h_graded_science : graded_science_sheets = 10)
  (h_graded_english : graded_english_sheets = 5) :
  (total_math_sheets * math_problems_per_sheet - graded_math_sheets * math_problems_per_sheet) +
  (total_science_sheets * science_problems_per_sheet - graded_science_sheets * science_problems_per_sheet) +
  (total_english_sheets * english_problems_per_sheet - graded_english_sheets * english_problems_per_sheet) = 84 :=
by sorry

end NUMINAMATH_CALUDE_problems_left_to_grade_l1540_154065


namespace NUMINAMATH_CALUDE_fractional_equation_integer_solution_l1540_154010

theorem fractional_equation_integer_solution (m : ℤ) : 
  (∃ x : ℤ, (m * x - 1) / (x - 2) + 1 / (2 - x) = 2 ∧ x ≠ 2) ↔ 
  (m = 4 ∨ m = 3 ∨ m = 0) :=
sorry

end NUMINAMATH_CALUDE_fractional_equation_integer_solution_l1540_154010


namespace NUMINAMATH_CALUDE_range_of_a_lower_bound_of_f_l1540_154032

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a - 1| + |x - 2*a|

-- Theorem for the range of a
theorem range_of_a (a : ℝ) : f 1 a < 3 → -2/3 < a ∧ a < 4/3 := by sorry

-- Theorem for the lower bound of f(x)
theorem lower_bound_of_f (a x : ℝ) : a ≥ 1 → f x a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_lower_bound_of_f_l1540_154032


namespace NUMINAMATH_CALUDE_problem_solution_l1540_154006

def f (x : ℝ) : ℝ := 2 * x^2 + 7

def g (x : ℝ) : ℝ := x^3 - 4

theorem problem_solution (a : ℝ) (h1 : a > 0) (h2 : f (g a) = 23) : 
  a = (2 * Real.sqrt 2 + 4) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1540_154006


namespace NUMINAMATH_CALUDE_sheet_width_calculation_l1540_154001

/-- Proves that a sheet with given dimensions and margins has a width of 20 cm when 64% is used for typing -/
theorem sheet_width_calculation (w : ℝ) : 
  w > 0 ∧ 
  (w - 4) * 24 = 0.64 * w * 30 → 
  w = 20 := by
  sorry

end NUMINAMATH_CALUDE_sheet_width_calculation_l1540_154001


namespace NUMINAMATH_CALUDE_cos_squared_pi_eighth_minus_one_l1540_154095

theorem cos_squared_pi_eighth_minus_one (π : Real) : 2 * Real.cos (π / 8) ^ 2 - 1 = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_pi_eighth_minus_one_l1540_154095


namespace NUMINAMATH_CALUDE_probability_grape_star_l1540_154094

/-- A tablet shape -/
inductive Shape
| Square
| Triangle
| Star

/-- A tablet flavor -/
inductive Flavor
| Strawberry
| Grape
| Orange

/-- The number of tablets of each shape -/
def tablets_per_shape : ℕ := 60

/-- The number of flavors -/
def num_flavors : ℕ := 3

/-- The total number of tablets -/
def total_tablets : ℕ := tablets_per_shape * 3

/-- The number of grape star tablets -/
def grape_star_tablets : ℕ := tablets_per_shape / num_flavors

theorem probability_grape_star :
  (grape_star_tablets : ℚ) / total_tablets = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_grape_star_l1540_154094


namespace NUMINAMATH_CALUDE_average_of_xyz_l1540_154008

theorem average_of_xyz (x y z : ℝ) (h : (5 / 4) * (x + y + z) = 20) :
  (x + y + z) / 3 = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_xyz_l1540_154008


namespace NUMINAMATH_CALUDE_condition_equivalent_to_a_range_l1540_154011

/-- The function f(x) = ax - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1

/-- The function g(x) = -x^2 + 2x + 1 -/
def g (x : ℝ) : ℝ := -x^2 + 2*x + 1

/-- The theorem stating the equivalence between the condition and the range of a -/
theorem condition_equivalent_to_a_range :
  ∀ a : ℝ, (∀ x₁ ∈ Set.Icc (-1) 1, ∃ x₂ ∈ Set.Icc 0 2, f a x₁ < g x₂) ↔ a ∈ Set.Ioo (-3) 3 :=
by sorry

end NUMINAMATH_CALUDE_condition_equivalent_to_a_range_l1540_154011


namespace NUMINAMATH_CALUDE_scientific_notation_120000_l1540_154004

theorem scientific_notation_120000 :
  (120000 : ℝ) = 1.2 * (10 ^ 5) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_120000_l1540_154004


namespace NUMINAMATH_CALUDE_jose_initial_caps_l1540_154064

/-- The number of bottle caps Jose started with -/
def initial_caps : ℕ := sorry

/-- The number of bottle caps Jose received from Rebecca -/
def received_caps : ℕ := 2

/-- The total number of bottle caps Jose ended up with -/
def total_caps : ℕ := 9

/-- Theorem stating that Jose started with 7 bottle caps -/
theorem jose_initial_caps : initial_caps = 7 := by
  sorry

end NUMINAMATH_CALUDE_jose_initial_caps_l1540_154064


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1540_154086

/-- The constant term in the expansion of (x - 2/x^2)^9 is -672 -/
theorem constant_term_expansion : 
  let f : ℝ → ℝ := fun x ↦ (x - 2 / x^2)^9
  ∃ c : ℝ, ∀ x : ℝ, x ≠ 0 → f x = c + x * (f x - c) / x ∧ c = -672 :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1540_154086


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l1540_154026

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 2 / 1)
  (hdb : d / b = 1 / 8) :
  a / c = 5 / 1 := by
sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l1540_154026


namespace NUMINAMATH_CALUDE_anthony_initial_pencils_l1540_154035

-- Define the variables
def pencils_given : ℝ := 9.0
def pencils_left : ℕ := 47

-- State the theorem
theorem anthony_initial_pencils : 
  pencils_given + pencils_left = 56 := by sorry

end NUMINAMATH_CALUDE_anthony_initial_pencils_l1540_154035


namespace NUMINAMATH_CALUDE_shared_rest_days_count_l1540_154048

/-- Chris's work cycle in days -/
def chris_cycle : ℕ := 7

/-- Dana's work cycle in days -/
def dana_cycle : ℕ := 7

/-- Total number of days -/
def total_days : ℕ := 1200

/-- Number of rest days Chris has in a cycle -/
def chris_rest_days : ℕ := 2

/-- Number of rest days Dana has in a cycle -/
def dana_rest_days : ℕ := 1

/-- The day in the cycle when both Chris and Dana rest -/
def common_rest_day : ℕ := 7

/-- The number of times Chris and Dana share a rest day in the given period -/
def shared_rest_days : ℕ := total_days / chris_cycle

theorem shared_rest_days_count :
  shared_rest_days = 171 :=
sorry

end NUMINAMATH_CALUDE_shared_rest_days_count_l1540_154048


namespace NUMINAMATH_CALUDE_train_crossing_time_l1540_154090

/-- Proves that a train with given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_km_hr : ℝ) (crossing_time : ℝ) : 
  train_length = 400 →
  train_speed_km_hr = 144 →
  crossing_time = train_length / (train_speed_km_hr * 1000 / 3600) →
  crossing_time = 10 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l1540_154090


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1540_154047

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = q * a n) →  -- geometric sequence condition
  q = -3 →                         -- given common ratio
  (a 1 + a 3 + a 5 + a 7) / (a 2 + a 4 + a 6 + a 8) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1540_154047


namespace NUMINAMATH_CALUDE_correct_writers_l1540_154082

/-- Represents the group of students and their writing task -/
structure StudentGroup where
  total : Nat
  cat_writers : Nat
  rat_writers : Nat
  crocodile_writers : Nat
  correct_cat : Nat
  correct_rat : Nat

/-- Theorem stating the number of students who wrote their word correctly -/
theorem correct_writers (group : StudentGroup) 
  (h1 : group.total = 50)
  (h2 : group.cat_writers = 10)
  (h3 : group.rat_writers = 18)
  (h4 : group.crocodile_writers = group.total - group.cat_writers - group.rat_writers)
  (h5 : group.correct_cat = 15)
  (h6 : group.correct_rat = 15) :
  group.correct_cat + group.correct_rat - (group.cat_writers + group.rat_writers) + group.crocodile_writers = 8 := by
  sorry

end NUMINAMATH_CALUDE_correct_writers_l1540_154082


namespace NUMINAMATH_CALUDE_sugar_substitute_box_cost_l1540_154067

theorem sugar_substitute_box_cost 
  (packets_per_coffee : ℕ)
  (coffees_per_day : ℕ)
  (packets_per_box : ℕ)
  (days_supply : ℕ)
  (total_cost : ℝ)
  (h1 : packets_per_coffee = 1)
  (h2 : coffees_per_day = 2)
  (h3 : packets_per_box = 30)
  (h4 : days_supply = 90)
  (h5 : total_cost = 24) :
  total_cost / (days_supply * coffees_per_day * packets_per_coffee / packets_per_box) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sugar_substitute_box_cost_l1540_154067


namespace NUMINAMATH_CALUDE_picnic_attendance_theorem_l1540_154013

/-- The percentage of men who attended the picnic -/
def men_attendance_rate : ℝ := 0.20

/-- The percentage of women who attended the picnic -/
def women_attendance_rate : ℝ := 0.40

/-- The percentage of employees who are men -/
def men_employee_rate : ℝ := 0.55

/-- The percentage of all employees who attended the picnic -/
def total_attendance_rate : ℝ := men_employee_rate * men_attendance_rate + (1 - men_employee_rate) * women_attendance_rate

theorem picnic_attendance_theorem :
  total_attendance_rate = 0.29 := by sorry

end NUMINAMATH_CALUDE_picnic_attendance_theorem_l1540_154013


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1540_154085

theorem contrapositive_equivalence (p q : Prop) :
  (p → q) → (¬q → ¬p) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1540_154085


namespace NUMINAMATH_CALUDE_pilot_miles_theorem_l1540_154079

theorem pilot_miles_theorem (tuesday_miles : ℕ) (total_miles : ℕ) :
  tuesday_miles = 1134 →
  total_miles = 7827 →
  ∃ (thursday_miles : ℕ),
    3 * (tuesday_miles + thursday_miles) = total_miles ∧
    thursday_miles = 1475 :=
by
  sorry

end NUMINAMATH_CALUDE_pilot_miles_theorem_l1540_154079


namespace NUMINAMATH_CALUDE_power_difference_square_sum_l1540_154039

theorem power_difference_square_sum (m n : ℕ+) : 
  2^(m : ℕ) - 2^(n : ℕ) = 1792 → m^2 + n^2 = 185 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_square_sum_l1540_154039


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1540_154046

/-- Given a sphere whose surface area increases by 4π cm² when cut in half,
    prove that its original surface area was 8π cm². -/
theorem sphere_surface_area (R : ℝ) (h : 2 * Real.pi * R^2 = 4 * Real.pi) :
  4 * Real.pi * R^2 = 8 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l1540_154046


namespace NUMINAMATH_CALUDE_vector_multiplication_and_addition_l1540_154037

theorem vector_multiplication_and_addition :
  (3 : ℝ) • ((-3 : ℝ), (2 : ℝ), (-5 : ℝ)) + ((4 : ℝ), (10 : ℝ), (-6 : ℝ)) = 
  ((-5 : ℝ), (16 : ℝ), (-21 : ℝ)) := by sorry

end NUMINAMATH_CALUDE_vector_multiplication_and_addition_l1540_154037


namespace NUMINAMATH_CALUDE_girls_from_maple_grove_l1540_154066

-- Define the total number of students
def total_students : ℕ := 150

-- Define the number of boys
def num_boys : ℕ := 82

-- Define the number of girls
def num_girls : ℕ := 68

-- Define the number of students from Pine Ridge School
def pine_ridge_students : ℕ := 70

-- Define the number of students from Maple Grove School
def maple_grove_students : ℕ := 80

-- Define the number of boys from Pine Ridge School
def pine_ridge_boys : ℕ := 36

-- Theorem to prove
theorem girls_from_maple_grove :
  total_students = num_boys + num_girls ∧
  total_students = pine_ridge_students + maple_grove_students ∧
  num_boys = pine_ridge_boys + (num_boys - pine_ridge_boys) →
  maple_grove_students - (num_boys - pine_ridge_boys) = 34 :=
by sorry

end NUMINAMATH_CALUDE_girls_from_maple_grove_l1540_154066


namespace NUMINAMATH_CALUDE_notebook_cost_l1540_154012

theorem notebook_cost (total_cost cover_cost notebook_cost : ℝ) : 
  total_cost = 3.60 →
  notebook_cost = 1.5 * cover_cost →
  total_cost = notebook_cost + cover_cost →
  notebook_cost = 2.16 := by
sorry

end NUMINAMATH_CALUDE_notebook_cost_l1540_154012


namespace NUMINAMATH_CALUDE_roberts_extra_chocolates_l1540_154059

theorem roberts_extra_chocolates (robert_chocolates nickel_chocolates : ℕ) 
  (h1 : robert_chocolates = 9) 
  (h2 : nickel_chocolates = 2) : 
  robert_chocolates - nickel_chocolates = 7 := by
sorry

end NUMINAMATH_CALUDE_roberts_extra_chocolates_l1540_154059


namespace NUMINAMATH_CALUDE_odd_numbers_perfect_square_l1540_154003

/-- Sum of first n odd numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n^2

/-- The n-th odd number -/
def nthOddNumber (n : ℕ) : ℕ := 2*n - 1

theorem odd_numbers_perfect_square (K : ℕ) :
  K % 2 = 1 →  -- K is odd
  (∃ (N : ℕ), N < 50 ∧ sumOddNumbers N = N^2 ∧ nthOddNumber N = K) →
  1 ≤ K ∧ K ≤ 97 :=
by sorry

end NUMINAMATH_CALUDE_odd_numbers_perfect_square_l1540_154003


namespace NUMINAMATH_CALUDE_science_club_membership_l1540_154042

theorem science_club_membership (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ)
  (h1 : total = 120)
  (h2 : math = 75)
  (h3 : physics = 50)
  (h4 : both = 15) :
  total - (math + physics - both) = 10 := by
sorry

end NUMINAMATH_CALUDE_science_club_membership_l1540_154042


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1540_154070

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℕ, x^3 > x^2) ↔ (∃ x : ℕ, x^3 ≤ x^2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1540_154070


namespace NUMINAMATH_CALUDE_largest_value_when_x_is_quarter_l1540_154002

theorem largest_value_when_x_is_quarter (x : ℝ) (h : x = 1/4) : 
  (1/x > x) ∧ (1/x > x^2) ∧ (1/x > (1/2)*x) ∧ (1/x > Real.sqrt x) := by
  sorry

end NUMINAMATH_CALUDE_largest_value_when_x_is_quarter_l1540_154002


namespace NUMINAMATH_CALUDE_janes_leave_days_l1540_154073

theorem janes_leave_days (jane_rate ashley_rate total_days extra_days : ℝ) 
  (h1 : jane_rate = 1 / 10)
  (h2 : ashley_rate = 1 / 40)
  (h3 : total_days = 15.2)
  (h4 : extra_days = 4) : 
  ∃ leave_days : ℝ, 
    (jane_rate + ashley_rate) * (total_days - leave_days) + 
    ashley_rate * leave_days + 
    jane_rate * extra_days = 1 ∧ 
    leave_days = 13 := by
sorry

end NUMINAMATH_CALUDE_janes_leave_days_l1540_154073


namespace NUMINAMATH_CALUDE_eight_jaguars_arrangement_l1540_154005

/-- The number of ways to arrange n different objects in a line -/
def linearArrangements (n : ℕ) : ℕ := n.factorial

/-- The number of ways to arrange n different objects in a line with the largest and smallest at the ends -/
def arrangementsWithExtremes (n : ℕ) : ℕ :=
  2 * linearArrangements (n - 2)

/-- Theorem: There are 1440 ways to arrange 8 different objects in a line with the largest and smallest at the ends -/
theorem eight_jaguars_arrangement :
  arrangementsWithExtremes 8 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_eight_jaguars_arrangement_l1540_154005


namespace NUMINAMATH_CALUDE_sequence_properties_l1540_154076

def a (n : ℕ+) : ℤ := n * (n - 8) - 20

theorem sequence_properties :
  (∃ (k : ℕ), k = 9 ∧ ∀ n : ℕ+, a n < 0 ↔ n.val ≤ k) ∧
  (∀ n : ℕ+, n ≥ 4 → a (n + 1) > a n) ∧
  (∀ n : ℕ+, a n ≥ a 4 ∧ a 4 = -36) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l1540_154076


namespace NUMINAMATH_CALUDE_traditionalist_progressive_ratio_l1540_154063

/-- Represents a country with provinces, progressives, and traditionalists -/
structure Country where
  num_provinces : ℕ
  total_population : ℝ
  fraction_traditionalist : ℝ
  progressives : ℝ
  traditionalists_per_province : ℝ

/-- The theorem stating the ratio of traditionalists in one province to total progressives -/
theorem traditionalist_progressive_ratio (c : Country) 
  (h1 : c.num_provinces = 4)
  (h2 : c.fraction_traditionalist = 0.75)
  (h3 : c.total_population = c.progressives + c.num_provinces * c.traditionalists_per_province)
  (h4 : c.fraction_traditionalist * c.total_population = c.num_provinces * c.traditionalists_per_province) :
  c.traditionalists_per_province / c.progressives = 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_traditionalist_progressive_ratio_l1540_154063


namespace NUMINAMATH_CALUDE_rectangle_existence_l1540_154050

theorem rectangle_existence (s d : ℝ) (hs : s > 0) (hd : d > 0) :
  ∃ (a b : ℝ), 2 * (a + b) = s ∧ a^2 + b^2 = d^2 ∧ a > 0 ∧ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_existence_l1540_154050


namespace NUMINAMATH_CALUDE_cooking_and_weaving_count_l1540_154025

/-- Represents the number of people in various curriculum combinations -/
structure CurriculumParticipation where
  yoga : ℕ
  cooking : ℕ
  weaving : ℕ
  cookingOnly : ℕ
  cookingAndYoga : ℕ
  allCurriculums : ℕ

/-- Theorem stating the number of people studying both cooking and weaving -/
theorem cooking_and_weaving_count (cp : CurriculumParticipation)
  (h1 : cp.yoga = 35)
  (h2 : cp.cooking = 20)
  (h3 : cp.weaving = 15)
  (h4 : cp.cookingOnly = 7)
  (h5 : cp.cookingAndYoga = 5)
  (h6 : cp.allCurriculums = 3) :
  cp.cooking - cp.cookingOnly - cp.cookingAndYoga + cp.allCurriculums = 5 := by
  sorry


end NUMINAMATH_CALUDE_cooking_and_weaving_count_l1540_154025


namespace NUMINAMATH_CALUDE_ages_ratio_years_ago_sum_of_ages_correct_years_ago_l1540_154023

/-- The number of years ago when the ages of A, B, and C were in the ratio 1 : 2 : 3 -/
def years_ago : ℕ := 3

/-- The present age of A -/
def A_age : ℕ := 11

/-- The present age of B -/
def B_age : ℕ := 22

/-- The present age of C -/
def C_age : ℕ := 24

/-- The theorem stating that the ages were in ratio 1:2:3 some years ago -/
theorem ages_ratio_years_ago : 
  (A_age - years_ago) * 2 = B_age - years_ago ∧
  (A_age - years_ago) * 3 = C_age - years_ago :=
sorry

/-- The theorem stating that the sum of present ages is 57 -/
theorem sum_of_ages : A_age + B_age + C_age = 57 :=
sorry

/-- The main theorem proving that 'years_ago' is correct -/
theorem correct_years_ago : 
  ∃ (y : ℕ), y = years_ago ∧
  (A_age - y) * 2 = B_age - y ∧
  (A_age - y) * 3 = C_age - y ∧
  A_age + B_age + C_age = 57 ∧
  A_age = 11 :=
sorry

end NUMINAMATH_CALUDE_ages_ratio_years_ago_sum_of_ages_correct_years_ago_l1540_154023


namespace NUMINAMATH_CALUDE_pi_is_irrational_l1540_154088

theorem pi_is_irrational : Irrational Real.pi := by sorry

end NUMINAMATH_CALUDE_pi_is_irrational_l1540_154088


namespace NUMINAMATH_CALUDE_inequality_proof_l1540_154099

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + 8/(x*y) + y^2 ≥ 8 ∧
  (x^2 + 8/(x*y) + y^2 = 8 ↔ x = Real.sqrt 2 ∧ y = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1540_154099


namespace NUMINAMATH_CALUDE_laundry_synchronization_l1540_154027

def ronald_cycle : ℕ := 6
def tim_cycle : ℕ := 9
def laura_cycle : ℕ := 12
def dani_cycle : ℕ := 15
def laura_birthday : ℕ := 35

theorem laundry_synchronization (ronald_cycle tim_cycle laura_cycle dani_cycle laura_birthday : ℕ) 
  (h1 : ronald_cycle = 6)
  (h2 : tim_cycle = 9)
  (h3 : laura_cycle = 12)
  (h4 : dani_cycle = 15)
  (h5 : laura_birthday = 35) :
  ∃ (next_sync : ℕ), next_sync - laura_birthday = 145 ∧ 
  next_sync % ronald_cycle = 0 ∧
  next_sync % tim_cycle = 0 ∧
  next_sync % laura_cycle = 0 ∧
  next_sync % dani_cycle = 0 :=
by sorry

end NUMINAMATH_CALUDE_laundry_synchronization_l1540_154027


namespace NUMINAMATH_CALUDE_angle_in_first_quadrant_l1540_154075

theorem angle_in_first_quadrant (α : Real) 
  (h1 : Real.tan α > 0) 
  (h2 : Real.sin α + Real.cos α > 0) : 
  0 < α ∧ α < Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_in_first_quadrant_l1540_154075
