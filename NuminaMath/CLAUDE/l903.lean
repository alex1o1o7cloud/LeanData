import Mathlib

namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l903_90318

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} :
  (∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- The value of a for which the given lines are parallel -/
theorem parallel_lines_a_value :
  ∀ a : ℝ, (∀ x y : ℝ, 2 * x + a * y + 2 = 0 ↔ a * x + (a + 4) * y - 1 = 0) ↔ (a = 4 ∨ a = -2) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l903_90318


namespace NUMINAMATH_CALUDE_inequality_equivalence_l903_90376

theorem inequality_equivalence (y : ℝ) :
  3/40 + |y - 17/80| < 1/8 ↔ 13/80 < y ∧ y < 21/80 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l903_90376


namespace NUMINAMATH_CALUDE_probability_of_four_given_different_numbers_l903_90354

/-- Two fair dice are rolled once each -/
def roll_two_dice : Type := Unit

/-- Event A: The two dice show different numbers -/
def event_A (roll : roll_two_dice) : Prop := sorry

/-- Event B: A 4 is rolled -/
def event_B (roll : roll_two_dice) : Prop := sorry

/-- P(B|A) is the conditional probability of event B given event A -/
def conditional_probability (A B : roll_two_dice → Prop) : ℝ := sorry

theorem probability_of_four_given_different_numbers :
  conditional_probability event_A event_B = 1/3 := by sorry

end NUMINAMATH_CALUDE_probability_of_four_given_different_numbers_l903_90354


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l903_90369

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane, represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Predicate to check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  ∃ (p : ℝ × ℝ), (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
                 p.2 = l.slope * p.1 + l.yIntercept

theorem tangent_line_y_intercept : 
  ∀ (l : Line) (c1 c2 : Circle),
    c1.center = (3, 0) →
    c1.radius = 3 →
    c2.center = (8, 0) →
    c2.radius = 2 →
    isTangent l c1 →
    isTangent l c2 →
    (∃ (p1 p2 : ℝ × ℝ), 
      isTangent l c1 ∧ 
      isTangent l c2 ∧ 
      p1.2 > 0 ∧ 
      p2.2 > 0) →
    l.yIntercept = Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l903_90369


namespace NUMINAMATH_CALUDE_contract_completion_hours_l903_90353

/-- Represents the contract completion problem -/
structure ContractProblem where
  total_days : ℕ
  initial_men : ℕ
  initial_hours : ℕ
  days_passed : ℕ
  work_completed : ℚ
  additional_men : ℕ

/-- Calculates the required daily work hours to complete the contract on time -/
def required_hours (p : ContractProblem) : ℚ :=
  let total_man_hours := p.initial_men * p.initial_hours * p.total_days
  let remaining_man_hours := (1 - p.work_completed) * total_man_hours
  let remaining_days := p.total_days - p.days_passed
  let total_men := p.initial_men + p.additional_men
  remaining_man_hours / (total_men * remaining_days)

/-- Theorem stating that the required work hours for the given problem is approximately 7.16 -/
theorem contract_completion_hours (p : ContractProblem) 
  (h1 : p.total_days = 46)
  (h2 : p.initial_men = 117)
  (h3 : p.initial_hours = 8)
  (h4 : p.days_passed = 33)
  (h5 : p.work_completed = 4/7)
  (h6 : p.additional_men = 81) :
  ∃ ε > 0, abs (required_hours p - 7.16) < ε :=
sorry

end NUMINAMATH_CALUDE_contract_completion_hours_l903_90353


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_a_range_l903_90382

open Real

/-- The function f(x) = e^x - (a-1)x + 1 is monotonically decreasing on [0,1] -/
def is_monotone_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ (Set.Icc 0 1) → y ∈ (Set.Icc 0 1) → x ≤ y → f x ≥ f y

/-- The main theorem -/
theorem monotone_decreasing_implies_a_range 
  (a : ℝ) 
  (f : ℝ → ℝ) 
  (h : f = fun x ↦ exp x - (a - 1) * x + 1) 
  (h_monotone : is_monotone_decreasing f) : 
  a ∈ Set.Ici (exp 1 + 1) := by
sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_a_range_l903_90382


namespace NUMINAMATH_CALUDE_complement_union_equals_set_l903_90366

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3, 5}
def N : Set ℕ := {4, 5}

theorem complement_union_equals_set : 
  (U \ (M ∪ N)) = {1, 6} := by sorry

end NUMINAMATH_CALUDE_complement_union_equals_set_l903_90366


namespace NUMINAMATH_CALUDE_bryan_shelves_count_l903_90305

/-- The number of mineral samples per shelf -/
def samples_per_shelf : ℕ := 65

/-- The total number of mineral samples -/
def total_samples : ℕ := 455

/-- The number of shelves Bryan has -/
def number_of_shelves : ℕ := total_samples / samples_per_shelf

theorem bryan_shelves_count : number_of_shelves = 7 := by
  sorry

end NUMINAMATH_CALUDE_bryan_shelves_count_l903_90305


namespace NUMINAMATH_CALUDE_representation_3060_l903_90351

def representationsCount (n : ℕ) : ℕ := 
  (n / 6) + 1

theorem representation_3060 : representationsCount 3060 = 511 := by
  sorry

end NUMINAMATH_CALUDE_representation_3060_l903_90351


namespace NUMINAMATH_CALUDE_f_is_odd_l903_90322

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
axiom not_identically_zero : ∃ x, f x ≠ 0

-- The functional equation
axiom functional_equation : ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a * Real.cos b

-- The theorem to prove
theorem f_is_odd : (∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a * Real.cos b) → 
  (∀ x : ℝ, f (-x) = -f x) :=
sorry

end NUMINAMATH_CALUDE_f_is_odd_l903_90322


namespace NUMINAMATH_CALUDE_percentage_married_students_l903_90345

theorem percentage_married_students (T : ℝ) (T_pos : T > 0) : 
  let male_students := 0.7 * T
  let female_students := 0.3 * T
  let married_male_students := (2/7) * male_students
  let married_female_students := (1/3) * female_students
  let total_married_students := married_male_students + married_female_students
  (total_married_students / T) * 100 = 30 := by sorry

end NUMINAMATH_CALUDE_percentage_married_students_l903_90345


namespace NUMINAMATH_CALUDE_half_distance_time_l903_90310

/-- Represents the total distance of Tony's errands in miles -/
def total_distance : ℝ := 10 + 15 + 5 + 20 + 25

/-- Represents Tony's constant speed in miles per hour -/
def speed : ℝ := 50

/-- Theorem stating that the time taken to drive half the total distance at the given speed is 0.75 hours -/
theorem half_distance_time : (total_distance / 2) / speed = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_half_distance_time_l903_90310


namespace NUMINAMATH_CALUDE_solve_factorial_equation_l903_90326

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem solve_factorial_equation : ∃ n : ℕ, n * factorial n + factorial n = 720 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_factorial_equation_l903_90326


namespace NUMINAMATH_CALUDE_magical_red_knights_fraction_l903_90361

/-- Represents the fraction of knights of a certain color who are magical -/
structure MagicalFraction where
  numerator : ℕ
  denominator : ℕ
  denominator_pos : denominator > 0

/-- Represents the distribution of knights in the kingdom -/
structure KnightDistribution where
  total : ℕ
  red : ℕ
  blue : ℕ
  magical : ℕ
  red_fraction : red = (3 * total) / 8
  blue_fraction : blue = total - red
  magical_fraction : magical = total / 4

/-- The main theorem about magical knights -/
theorem magical_red_knights_fraction 
  (dist : KnightDistribution) 
  (red_magical : MagicalFraction) 
  (blue_magical : MagicalFraction) :
  (3 * dist.total) / 8 * red_magical.numerator / red_magical.denominator + 
  (5 * dist.total) / 8 * blue_magical.numerator / blue_magical.denominator = dist.total / 4 →
  red_magical.numerator * blue_magical.denominator = 
  3 * blue_magical.numerator * red_magical.denominator →
  red_magical.numerator = 6 ∧ red_magical.denominator = 19 :=
sorry

end NUMINAMATH_CALUDE_magical_red_knights_fraction_l903_90361


namespace NUMINAMATH_CALUDE_shirt_price_calculation_l903_90373

/-- The price of a single shirt -/
def shirt_price : ℝ := 45

/-- The price of a single pair of pants -/
def pants_price : ℝ := 25

/-- The total cost of the initial purchase -/
def total_cost : ℝ := 120

/-- The refund percentage -/
def refund_percentage : ℝ := 0.25

theorem shirt_price_calculation :
  (∀ s₁ s₂ : ℝ, s₁ = s₂ → s₁ = shirt_price) →  -- All shirts have the same price
  (∀ p₁ p₂ : ℝ, p₁ = p₂ → p₁ = pants_price) →  -- All pants have the same price
  shirt_price ≠ pants_price →                  -- Shirt price ≠ pants price
  2 * shirt_price + 3 * pants_price = total_cost →  -- 2 shirts + 3 pants = $120
  3 * pants_price = refund_percentage * total_cost →  -- Refund for 3 pants = 25% of $120
  shirt_price = 45 := by sorry

end NUMINAMATH_CALUDE_shirt_price_calculation_l903_90373


namespace NUMINAMATH_CALUDE_original_price_is_20_l903_90368

/-- Represents the ticket pricing scenario for a concert --/
structure ConcertTickets where
  original_price : ℝ
  total_revenue : ℝ
  total_tickets : ℕ
  discount_40_count : ℕ
  discount_15_count : ℕ

/-- The concert ticket scenario satisfies the given conditions --/
def valid_scenario (c : ConcertTickets) : Prop :=
  c.total_tickets = 50 ∧
  c.discount_40_count = 10 ∧
  c.discount_15_count = 20 ∧
  c.total_revenue = 860 ∧
  c.total_revenue = (c.discount_40_count : ℝ) * (0.6 * c.original_price) +
                    (c.discount_15_count : ℝ) * (0.85 * c.original_price) +
                    ((c.total_tickets - c.discount_40_count - c.discount_15_count) : ℝ) * c.original_price

/-- The original ticket price is $20 --/
theorem original_price_is_20 (c : ConcertTickets) (h : valid_scenario c) :
  c.original_price = 20 := by
  sorry

end NUMINAMATH_CALUDE_original_price_is_20_l903_90368


namespace NUMINAMATH_CALUDE_car_wash_price_l903_90343

theorem car_wash_price (oil_change_price : ℕ) (repair_price : ℕ) (oil_changes : ℕ) (repairs : ℕ) (car_washes : ℕ) (total_earnings : ℕ) :
  oil_change_price = 20 →
  repair_price = 30 →
  oil_changes = 5 →
  repairs = 10 →
  car_washes = 15 →
  total_earnings = 475 →
  (oil_change_price * oil_changes + repair_price * repairs + car_washes * 5 = total_earnings) :=
by
  sorry


end NUMINAMATH_CALUDE_car_wash_price_l903_90343


namespace NUMINAMATH_CALUDE_endangered_animal_population_l903_90344

/-- The population of an endangered animal after n years, given an initial population and annual decrease rate. -/
def population (m : ℝ) (r : ℝ) (n : ℕ) : ℝ := m * (1 - r) ^ n

/-- Theorem stating that given specific conditions, the population after 3 years will be 5832. -/
theorem endangered_animal_population :
  let m : ℝ := 8000  -- Initial population
  let r : ℝ := 0.1   -- Annual decrease rate (10%)
  let n : ℕ := 3     -- Number of years
  population m r n = 5832 := by
  sorry

end NUMINAMATH_CALUDE_endangered_animal_population_l903_90344


namespace NUMINAMATH_CALUDE_exists_irrational_between_3_and_4_l903_90304

theorem exists_irrational_between_3_and_4 : ∃ x : ℝ, Irrational x ∧ 3 < x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_exists_irrational_between_3_and_4_l903_90304


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l903_90390

/-- The discriminant of a quadratic equation ax² + bx + c is b² - 4ac -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 5x² + (5 + 1/5)x - 2/5 -/
def a : ℚ := 5
def b : ℚ := 5 + 1/5
def c : ℚ := -2/5

theorem quadratic_discriminant : discriminant a b c = 876/25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l903_90390


namespace NUMINAMATH_CALUDE_functional_equation_solution_l903_90328

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f x + f y) = y + (f x)^2

/-- The main theorem stating that any function satisfying the equation must be either the identity function or its negation -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l903_90328


namespace NUMINAMATH_CALUDE_profit_margin_increase_l903_90378

theorem profit_margin_increase (initial_margin : ℝ) (final_margin : ℝ) 
  (price_increase : ℝ) : 
  initial_margin = 0.25 →
  final_margin = 0.40 →
  price_increase = (1 + final_margin) / (1 + initial_margin) - 1 →
  price_increase = 0.12 := by
  sorry

#check profit_margin_increase

end NUMINAMATH_CALUDE_profit_margin_increase_l903_90378


namespace NUMINAMATH_CALUDE_all_transformations_correct_l903_90329

-- Define the transformations
def transformation_A (a b : ℝ) : Prop := a = b → a + 5 = b + 5

def transformation_B (x y a : ℝ) : Prop := x = y → x / a = y / a

def transformation_C (m n : ℝ) : Prop := m = n → 1 - 3 * m = 1 - 3 * n

def transformation_D (x y c : ℝ) : Prop := x = y → x * c = y * c

-- Theorem stating all transformations are correct
theorem all_transformations_correct :
  (∀ a b : ℝ, transformation_A a b) ∧
  (∀ x y a : ℝ, a ≠ 0 → transformation_B x y a) ∧
  (∀ m n : ℝ, transformation_C m n) ∧
  (∀ x y c : ℝ, transformation_D x y c) :=
sorry

end NUMINAMATH_CALUDE_all_transformations_correct_l903_90329


namespace NUMINAMATH_CALUDE_circle_and_trajectory_l903_90341

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 4)^2 + p.2^2 = 5}

-- Define points M and N
def M : ℝ × ℝ := (5, 2)
def N : ℝ × ℝ := (3, 2)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem circle_and_trajectory :
  (M ∈ circle_C) ∧ 
  (N ∈ circle_C) ∧ 
  (∀ p ∈ circle_C, p.1 = 4 → p.2 = 0) →
  (∀ A ∈ circle_C, 
    ∃ P : ℝ × ℝ, 
      (P.1 - O.1 = 2 * (A.1 - O.1)) ∧ 
      (P.2 - O.2 = 2 * (A.2 - O.2)) ∧
      (P.1 - 8)^2 + P.2^2 = 20) := by
  sorry

end NUMINAMATH_CALUDE_circle_and_trajectory_l903_90341


namespace NUMINAMATH_CALUDE_charity_donation_l903_90364

/-- The number of pennies collected by Cassandra -/
def cassandra_pennies : ℕ := 5000

/-- The difference in pennies collected between Cassandra and James -/
def difference : ℕ := 276

/-- The number of pennies collected by James -/
def james_pennies : ℕ := cassandra_pennies - difference

/-- The total number of pennies donated to charity -/
def total_donated : ℕ := cassandra_pennies + james_pennies

theorem charity_donation :
  total_donated = 9724 :=
sorry

end NUMINAMATH_CALUDE_charity_donation_l903_90364


namespace NUMINAMATH_CALUDE_nate_optimal_speed_l903_90384

/-- The speed at which Nate should drive to arrive just in time -/
def optimal_speed : ℝ := 48

/-- The time it takes for Nate to arrive on time -/
def on_time : ℝ := 5

/-- The distance Nate needs to travel -/
def distance : ℝ := 240

theorem nate_optimal_speed :
  (distance = 40 * (on_time + 1)) ∧
  (distance = 60 * (on_time - 1)) →
  optimal_speed = distance / on_time :=
by sorry

end NUMINAMATH_CALUDE_nate_optimal_speed_l903_90384


namespace NUMINAMATH_CALUDE_streetlight_combinations_l903_90334

/-- Represents the number of streetlights -/
def total_lights : ℕ := 12

/-- Represents the number of lights that can be turned off -/
def lights_off : ℕ := 3

/-- Represents the number of positions where lights can be turned off -/
def eligible_positions : ℕ := 8

/-- The number of ways to turn off lights under the given conditions -/
def ways_to_turn_off : ℕ := Nat.choose eligible_positions lights_off

theorem streetlight_combinations : ways_to_turn_off = 56 := by
  sorry

end NUMINAMATH_CALUDE_streetlight_combinations_l903_90334


namespace NUMINAMATH_CALUDE_original_number_exists_and_unique_l903_90360

theorem original_number_exists_and_unique : ∃! x : ℝ, 3 * (2 * x + 9) = 51 := by
  sorry

end NUMINAMATH_CALUDE_original_number_exists_and_unique_l903_90360


namespace NUMINAMATH_CALUDE_expression_value_l903_90395

theorem expression_value (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : abs a ≠ abs b) :
  let expr1 := b^2 / a^2 + a^2 / b^2 - 2
  let expr2 := (a + b) / (b - a) + (b - a) / (a + b)
  let expr3 := (1 / a^2 + 1 / b^2) / (1 / b^2 - 1 / a^2) - (1 / b^2 - 1 / a^2) / (1 / a^2 + 1 / b^2)
  expr1 * expr2 * expr3 = -8 := by sorry

end NUMINAMATH_CALUDE_expression_value_l903_90395


namespace NUMINAMATH_CALUDE_problem_solution_l903_90352

theorem problem_solution (a b k : ℝ) 
  (h1 : 4^a = k) 
  (h2 : 9^b = k) 
  (h3 : 1/a + 1/b = 2) : k = 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l903_90352


namespace NUMINAMATH_CALUDE_talia_father_age_l903_90363

-- Define Talia's current age
def talia_age : ℕ := 20 - 7

-- Define Talia's mom's current age
def mom_age : ℕ := 3 * talia_age

-- Define Talia's father's current age
def father_age : ℕ := mom_age - 3

-- Theorem statement
theorem talia_father_age : father_age = 36 := by
  sorry

end NUMINAMATH_CALUDE_talia_father_age_l903_90363


namespace NUMINAMATH_CALUDE_range_of_x_when_m_is_2_range_of_m_when_p_necessary_not_sufficient_l903_90324

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 6*x + 5 ≤ 0

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Part 1
theorem range_of_x_when_m_is_2 :
  ∀ x : ℝ, (p x ∧ q x 2) → (1 ≤ x ∧ x ≤ 3) :=
sorry

-- Part 2
theorem range_of_m_when_p_necessary_not_sufficient :
  ∀ m : ℝ, (m > 0 ∧ (∀ x : ℝ, q x m → p x) ∧ (∃ x : ℝ, p x ∧ ¬(q x m))) → m ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_m_is_2_range_of_m_when_p_necessary_not_sufficient_l903_90324


namespace NUMINAMATH_CALUDE_integer_solutions_for_equation_l903_90358

theorem integer_solutions_for_equation : 
  {(x, y) : ℤ × ℤ | x^2 - y^4 = 2009} = {(45, 2), (45, -2), (-45, 2), (-45, -2)} :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_for_equation_l903_90358


namespace NUMINAMATH_CALUDE_somin_solved_most_l903_90371

def suhyeon_remaining : ℚ := 1/4
def somin_remaining : ℚ := 1/8
def jisoo_remaining : ℚ := 1/5

theorem somin_solved_most : 
  somin_remaining < suhyeon_remaining ∧ somin_remaining < jisoo_remaining :=
sorry

end NUMINAMATH_CALUDE_somin_solved_most_l903_90371


namespace NUMINAMATH_CALUDE_property_sale_outcome_l903_90321

/-- Calculates the net outcome for a property seller in a specific scenario --/
theorem property_sale_outcome (initial_value : ℝ) (profit_rate : ℝ) (loss_rate : ℝ) (fee_rate : ℝ) : 
  initial_value = 20000 ∧ 
  profit_rate = 0.15 ∧ 
  loss_rate = 0.15 ∧ 
  fee_rate = 0.05 → 
  (initial_value * (1 + profit_rate)) - 
  (initial_value * (1 + profit_rate) * (1 - loss_rate) * (1 + fee_rate)) = 2472.5 := by
sorry

end NUMINAMATH_CALUDE_property_sale_outcome_l903_90321


namespace NUMINAMATH_CALUDE_least_possible_value_z_minus_x_l903_90379

theorem least_possible_value_z_minus_x 
  (x y z : ℤ) 
  (h1 : x < y ∧ y < z)
  (h2 : y - x > 11)
  (h3 : Even x)
  (h4 : Odd y ∧ Odd z) :
  ∀ w : ℤ, w = z - x → w ≥ 14 ∧ ∃ (x' y' z' : ℤ), 
    x' < y' ∧ y' < z' ∧ 
    y' - x' > 11 ∧ 
    Even x' ∧ Odd y' ∧ Odd z' ∧ 
    z' - x' = 14 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_value_z_minus_x_l903_90379


namespace NUMINAMATH_CALUDE_find_a_interest_rate_l903_90331

-- Define constants
def total_amount : ℝ := 10000
def years : ℝ := 2
def b_interest_rate : ℝ := 18
def interest_difference : ℝ := 360
def b_amount : ℝ := 4000

-- Define variables
variable (a_amount : ℝ) (a_interest_rate : ℝ)

-- Theorem statement
theorem find_a_interest_rate :
  a_amount + b_amount = total_amount →
  (a_amount * a_interest_rate * years) / 100 = (b_amount * b_interest_rate * years) / 100 + interest_difference →
  a_interest_rate = 15 := by
  sorry

end NUMINAMATH_CALUDE_find_a_interest_rate_l903_90331


namespace NUMINAMATH_CALUDE_acid_solution_replacement_l903_90357

theorem acid_solution_replacement (V : ℝ) (h : V > 0) :
  let x : ℝ := 0.5
  let initial_concentration : ℝ := 0.5
  let replacement_concentration : ℝ := 0.3
  let final_concentration : ℝ := 0.4
  initial_concentration * V - initial_concentration * x * V + replacement_concentration * x * V = final_concentration * V :=
by sorry

end NUMINAMATH_CALUDE_acid_solution_replacement_l903_90357


namespace NUMINAMATH_CALUDE_problem_solution_l903_90362

theorem problem_solution (m n : ℚ) (h : m - n = -2/3) : 7 - 3*m + 3*n = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l903_90362


namespace NUMINAMATH_CALUDE_house_transaction_net_worth_change_l903_90302

/-- Represents the financial state of a person -/
structure FinancialState where
  cash : Int
  houseValue : Int

/-- Calculates the net worth of a person -/
def netWorth (state : FinancialState) : Int :=
  state.cash + state.houseValue

/-- Represents a house transaction between two people -/
def houseTransaction (buyer seller : FinancialState) (price : Int) : (FinancialState × FinancialState) :=
  ({ cash := buyer.cash - price, houseValue := seller.houseValue },
   { cash := seller.cash + price, houseValue := 0 })

theorem house_transaction_net_worth_change 
  (initialA initialB : FinancialState)
  (houseValue firstPrice secondPrice : Int) :
  initialA.cash = 15000 →
  initialA.houseValue = 12000 →
  initialB.cash = 13000 →
  initialB.houseValue = 0 →
  houseValue = 12000 →
  firstPrice = 14000 →
  secondPrice = 10000 →
  let (afterFirstA, afterFirstB) := houseTransaction initialB initialA firstPrice
  let (finalB, finalA) := houseTransaction afterFirstA afterFirstB secondPrice
  netWorth finalA - netWorth initialA = 4000 ∧
  netWorth finalB - netWorth initialB = -4000 := by sorry


end NUMINAMATH_CALUDE_house_transaction_net_worth_change_l903_90302


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_intersection_l903_90314

-- Define the two circles
def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y + 3)^2 = 13
def circle2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Define the perpendicular bisector equation
def perp_bisector (x y : ℝ) : Prop := 3*x - y - 9 = 0

-- Theorem statement
theorem perpendicular_bisector_of_intersection :
  ∃ (A B : ℝ × ℝ), 
    (circle1 A.1 A.2 ∧ circle2 A.1 A.2) ∧ 
    (circle1 B.1 B.2 ∧ circle2 B.1 B.2) ∧ 
    A ≠ B ∧
    (∀ (x y : ℝ), perp_bisector x y ↔ 
      (x - (A.1 + B.1)/2)^2 + (y - (A.2 + B.2)/2)^2 = 
      ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_intersection_l903_90314


namespace NUMINAMATH_CALUDE_book_arrangement_proof_l903_90309

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem book_arrangement_proof :
  let total_books : ℕ := 9
  let arabic_books : ℕ := 2
  let french_books : ℕ := 3
  let english_books : ℕ := 4
  let arabic_group : ℕ := 1
  let english_group : ℕ := 1
  let total_groups : ℕ := arabic_group + english_group + french_books

  (factorial total_groups) * (factorial arabic_books) * (factorial english_books) = 5760 :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_proof_l903_90309


namespace NUMINAMATH_CALUDE_gcd_problem_l903_90313

theorem gcd_problem (b : ℤ) (h : ∃ (k : ℤ), b = 7 * (2 * k + 1)) :
  Int.gcd (3 * b^2 + 34 * b + 76) (b + 16) = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l903_90313


namespace NUMINAMATH_CALUDE_jenny_money_l903_90385

theorem jenny_money (original : ℚ) : 
  (4/7 : ℚ) * original = 24 → (1/2 : ℚ) * original = 21 := by
sorry

end NUMINAMATH_CALUDE_jenny_money_l903_90385


namespace NUMINAMATH_CALUDE_area_of_specific_quadrilateral_l903_90349

/-- The area of a quadrilateral can be calculated using the Shoelace formula -/
def quadrilateralArea (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  let x1 := v1.1
  let y1 := v1.2
  let x2 := v2.1
  let y2 := v2.2
  let x3 := v3.1
  let y3 := v3.2
  let x4 := v4.1
  let y4 := v4.2
  0.5 * abs ((x1*y2 + x2*y3 + x3*y4 + x4*y1) - (y1*x2 + y2*x3 + y3*x4 + y4*x1))

theorem area_of_specific_quadrilateral :
  quadrilateralArea (2, 1) (4, 3) (7, 1) (4, 6) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_area_of_specific_quadrilateral_l903_90349


namespace NUMINAMATH_CALUDE_four_inch_cube_value_l903_90375

/-- Represents the properties of a gold cube -/
structure GoldCube where
  edge : ℝ  -- Edge length in inches
  weight : ℝ  -- Weight in pounds
  value : ℝ  -- Value in dollars

/-- The properties of gold cubes are directly proportional to their volume -/
axiom prop_proportional_to_volume (c1 c2 : GoldCube) :
  c2.weight = c1.weight * (c2.edge / c1.edge)^3 ∧
  c2.value = c1.value * (c2.edge / c1.edge)^3

/-- Given information about a one-inch gold cube -/
def one_inch_cube : GoldCube :=
  { edge := 1
  , weight := 0.5
  , value := 1000 }

/-- Theorem: A four-inch cube of gold is worth $64000 -/
theorem four_inch_cube_value :
  ∃ (c : GoldCube), c.edge = 4 ∧ c.value = 64000 :=
sorry

end NUMINAMATH_CALUDE_four_inch_cube_value_l903_90375


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l903_90392

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ x y : ℝ, x ≥ 1 ∧ y ≥ 1 → x + y ≥ 2) ∧
  (∃ x y : ℝ, x + y ≥ 2 ∧ ¬(x ≥ 1 ∧ y ≥ 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l903_90392


namespace NUMINAMATH_CALUDE_total_intersection_points_l903_90398

/-- Regular polygon inscribed in a circle -/
structure RegularPolygon where
  sides : ℕ
  inscribed : Bool

/-- Represents the configuration of regular polygons in a circle -/
structure PolygonConfiguration where
  square : RegularPolygon
  hexagon : RegularPolygon
  octagon : RegularPolygon
  shared_vertices : ℕ
  no_triple_intersections : Bool

/-- Calculates the number of intersection points between two polygons -/
def intersectionPoints (p1 p2 : RegularPolygon) (shared : Bool) : ℕ :=
  sorry

/-- Theorem stating the total number of intersection points -/
theorem total_intersection_points (config : PolygonConfiguration) : 
  config.square.sides = 4 ∧ 
  config.hexagon.sides = 6 ∧ 
  config.octagon.sides = 8 ∧
  config.square.inscribed ∧
  config.hexagon.inscribed ∧
  config.octagon.inscribed ∧
  config.shared_vertices ≤ 3 ∧
  config.no_triple_intersections →
  intersectionPoints config.square config.hexagon (config.shared_vertices > 0) +
  intersectionPoints config.square config.octagon (config.shared_vertices > 1) +
  intersectionPoints config.hexagon config.octagon (config.shared_vertices > 2) = 164 :=
sorry

end NUMINAMATH_CALUDE_total_intersection_points_l903_90398


namespace NUMINAMATH_CALUDE_sum_of_F_at_4_and_neg_2_l903_90393

noncomputable def F (x : ℝ) : ℝ := Real.sqrt (abs (x + 2)) + (10 / Real.pi) * Real.arctan (Real.sqrt (abs x))

theorem sum_of_F_at_4_and_neg_2 : F 4 + F (-2) = Real.sqrt 6 + 3.529 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_F_at_4_and_neg_2_l903_90393


namespace NUMINAMATH_CALUDE_sum_coordinates_of_D_l903_90300

/-- Given that N(6,2) is the midpoint of line segment CD and C(10,-2), 
    prove that the sum of coordinates of D is 8 -/
theorem sum_coordinates_of_D (N C D : ℝ × ℝ) : 
  N = (6, 2) → 
  C = (10, -2) → 
  N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → 
  D.1 + D.2 = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_coordinates_of_D_l903_90300


namespace NUMINAMATH_CALUDE_contradiction_proof_l903_90336

theorem contradiction_proof (x a b : ℝ) : 
  x^2 - (a + b)*x - a*b ≠ 0 → x ≠ a ∧ x ≠ b := by
  sorry

end NUMINAMATH_CALUDE_contradiction_proof_l903_90336


namespace NUMINAMATH_CALUDE_smallest_b_in_arithmetic_series_l903_90338

theorem smallest_b_in_arithmetic_series (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- a, b, c are positive
  ∃ d : ℝ, a = b - d ∧ c = b + d →  -- a, b, c form an arithmetic series
  a * b * c = 216 →  -- product is 216
  b ≥ 6 ∧ ∀ x : ℝ, (0 < x ∧ ∃ y z : ℝ, 0 < y ∧ 0 < z ∧ ∃ d : ℝ, y = x - d ∧ z = x + d ∧ y * x * z = 216) → x ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_in_arithmetic_series_l903_90338


namespace NUMINAMATH_CALUDE_pages_left_to_read_l903_90320

theorem pages_left_to_read (total_pages : ℕ) (percent_read : ℚ) (pages_left : ℕ) : 
  total_pages = 1250 → 
  percent_read = 37/100 → 
  pages_left = total_pages - Int.floor (percent_read * total_pages) → 
  pages_left = 788 := by
sorry

end NUMINAMATH_CALUDE_pages_left_to_read_l903_90320


namespace NUMINAMATH_CALUDE_ants_meet_after_11_laps_l903_90347

/-- The number of laps on the small circle before the ants meet again -/
def num_laps_to_meet (large_radius small_radius : ℕ) : ℕ :=
  Nat.lcm large_radius small_radius / small_radius

theorem ants_meet_after_11_laps :
  num_laps_to_meet 33 9 = 11 := by sorry

end NUMINAMATH_CALUDE_ants_meet_after_11_laps_l903_90347


namespace NUMINAMATH_CALUDE_initial_peaches_count_l903_90303

/-- Represents the state of the fruit bowl on a given day -/
structure FruitBowl :=
  (day : Nat)
  (ripe : Nat)
  (unripe : Nat)

/-- Updates the fruit bowl state for the next day -/
def nextDay (bowl : FruitBowl) : FruitBowl :=
  let newRipe := bowl.ripe + 2
  let newUnripe := bowl.unripe - 2
  { day := bowl.day + 1, ripe := newRipe, unripe := newUnripe }

/-- Represents eating 3 peaches on day 3 -/
def eatPeaches (bowl : FruitBowl) : FruitBowl :=
  { bowl with ripe := bowl.ripe - 3 }

/-- The initial state of the fruit bowl -/
def initialBowl : FruitBowl :=
  { day := 0, ripe := 4, unripe := 13 }

/-- The final state of the fruit bowl after 5 days -/
def finalBowl : FruitBowl :=
  (nextDay ∘ nextDay ∘ eatPeaches ∘ nextDay ∘ nextDay ∘ nextDay) initialBowl

/-- Theorem stating that the initial number of peaches was 17 -/
theorem initial_peaches_count :
  initialBowl.ripe + initialBowl.unripe = 17 ∧
  finalBowl.ripe = finalBowl.unripe + 7 :=
by sorry


end NUMINAMATH_CALUDE_initial_peaches_count_l903_90303


namespace NUMINAMATH_CALUDE_other_number_proof_l903_90315

/-- Given two positive integers with known HCF, LCM, and one of the numbers, prove the value of the other number -/
theorem other_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 16) (h2 : Nat.lcm a b = 396) (h3 : a = 36) : b = 176 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l903_90315


namespace NUMINAMATH_CALUDE_farmer_earnings_l903_90388

/-- Calculates the total earnings from selling potatoes and carrots given the harvest quantities and pricing. -/
theorem farmer_earnings (potato_count : ℕ) (potato_bundle_size : ℕ) (potato_bundle_price : ℚ)
                        (carrot_count : ℕ) (carrot_bundle_size : ℕ) (carrot_bundle_price : ℚ) :
  potato_count = 250 →
  potato_bundle_size = 25 →
  potato_bundle_price = 190 / 100 →
  carrot_count = 320 →
  carrot_bundle_size = 20 →
  carrot_bundle_price = 2 →
  (potato_count / potato_bundle_size * potato_bundle_price +
   carrot_count / carrot_bundle_size * carrot_bundle_price : ℚ) = 51 := by
  sorry

#eval (250 / 25 * (190 / 100) + 320 / 20 * 2 : ℚ)

end NUMINAMATH_CALUDE_farmer_earnings_l903_90388


namespace NUMINAMATH_CALUDE_expression_evaluation_l903_90356

theorem expression_evaluation : 
  2016 / ((13 + 5/7) - (8 + 8/11)) * (5/7 - 5/11) = 105 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l903_90356


namespace NUMINAMATH_CALUDE_mixture_weight_l903_90301

/-- Calculates the weight of a mixture of two brands of vegetable ghee -/
theorem mixture_weight 
  (weight_a : ℝ) 
  (weight_b : ℝ) 
  (ratio_a : ℝ) 
  (ratio_b : ℝ) 
  (total_volume : ℝ) 
  (h1 : weight_a = 900) 
  (h2 : weight_b = 850) 
  (h3 : ratio_a = 3) 
  (h4 : ratio_b = 2) 
  (h5 : total_volume = 4) : 
  (weight_a * (ratio_a / (ratio_a + ratio_b)) * total_volume + 
   weight_b * (ratio_b / (ratio_a + ratio_b)) * total_volume) / 1000 = 3.52 := by
sorry

end NUMINAMATH_CALUDE_mixture_weight_l903_90301


namespace NUMINAMATH_CALUDE_sin_cos_sum_equal_shifted_cos_l903_90372

theorem sin_cos_sum_equal_shifted_cos (x : ℝ) : 
  Real.sin (3 * x) + Real.cos (3 * x) = Real.sqrt 2 * Real.cos (3 * x - π / 4) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equal_shifted_cos_l903_90372


namespace NUMINAMATH_CALUDE_probability_both_truth_l903_90389

theorem probability_both_truth (prob_A prob_B : ℝ) 
  (h_A : prob_A = 0.7) 
  (h_B : prob_B = 0.6) : 
  prob_A * prob_B = 0.42 := by
sorry

end NUMINAMATH_CALUDE_probability_both_truth_l903_90389


namespace NUMINAMATH_CALUDE_bank_coins_l903_90386

theorem bank_coins (total_coins dimes quarters : ℕ) (h1 : total_coins = 11) (h2 : dimes = 2) (h3 : quarters = 7) :
  ∃ nickels : ℕ, nickels = total_coins - dimes - quarters :=
by
  sorry

end NUMINAMATH_CALUDE_bank_coins_l903_90386


namespace NUMINAMATH_CALUDE_absolute_value_inequalities_l903_90377

theorem absolute_value_inequalities (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 2| > a → a < 3) ∧
  (∀ x : ℝ, |x - 1| - |x + 3| < a → a > 4) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequalities_l903_90377


namespace NUMINAMATH_CALUDE_simplify_trig_fraction_l903_90333

theorem simplify_trig_fraction (x : ℝ) :
  (3 + 2 * Real.sin x + 2 * Real.cos x) / (3 + 2 * Real.sin x - 2 * Real.cos x) = 
  3 / 5 + 2 / 5 * Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_fraction_l903_90333


namespace NUMINAMATH_CALUDE_project_completion_time_l903_90381

/-- The number of days it takes for person A to complete the project alone -/
def A_days : ℕ := 20

/-- The number of days it takes for both A and B to complete the project together,
    with A quitting 10 days before completion -/
def total_days : ℕ := 18

/-- The number of days A works before quitting -/
def A_work_days : ℕ := total_days - 10

/-- The rate at which person A completes the project per day -/
def A_rate : ℚ := 1 / A_days

theorem project_completion_time (B_days : ℕ) :
  (A_work_days : ℚ) * (A_rate + 1 / B_days) + (10 : ℚ) * (1 / B_days) = 1 →
  B_days = 30 := by sorry

end NUMINAMATH_CALUDE_project_completion_time_l903_90381


namespace NUMINAMATH_CALUDE_complement_of_union_M_P_l903_90383

open Set

-- Define the universal set U as the real numbers
def U : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x | x ≤ 1}

-- Define set P
def P : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem complement_of_union_M_P : 
  (M ∪ P)ᶜ = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_M_P_l903_90383


namespace NUMINAMATH_CALUDE_complex_fraction_bounds_l903_90370

theorem complex_fraction_bounds (z w : ℂ) (hz : z ≠ 0) (hw : w ≠ 0) :
  ∃ (min max : ℝ),
    (∀ z w : ℂ, z ≠ 0 → w ≠ 0 → min ≤ Complex.abs (z + w) / (Complex.abs z + Complex.abs w) ∧
                        Complex.abs (z + w) / (Complex.abs z + Complex.abs w) ≤ max) ∧
    min = 0 ∧ max = 1 ∧ max - min = 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_bounds_l903_90370


namespace NUMINAMATH_CALUDE_robotics_club_max_participants_l903_90335

theorem robotics_club_max_participants 
  (physics : Finset ℕ)
  (math : Finset ℕ)
  (programming : Finset ℕ)
  (h1 : physics.card = 8)
  (h2 : math.card = 7)
  (h3 : programming.card = 11)
  (h4 : (physics ∩ math).card ≥ 2)
  (h5 : (math ∩ programming).card ≥ 3)
  (h6 : (physics ∩ programming).card ≥ 4) :
  (physics ∪ math ∪ programming).card ≤ 19 :=
sorry

end NUMINAMATH_CALUDE_robotics_club_max_participants_l903_90335


namespace NUMINAMATH_CALUDE_nine_solutions_mod_455_l903_90306

theorem nine_solutions_mod_455 : 
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, 1 ≤ n ∧ n ≤ 455 ∧ n^3 % 455 = 1) ∧ 
    (∀ n, 1 ≤ n ∧ n ≤ 455 ∧ n^3 % 455 = 1 → n ∈ s) ∧ 
    s.card = 9 :=
by sorry

end NUMINAMATH_CALUDE_nine_solutions_mod_455_l903_90306


namespace NUMINAMATH_CALUDE_new_persons_weight_l903_90348

theorem new_persons_weight (original_count : ℕ) (weight_increase : ℝ) 
  (replaced_weight1 : ℝ) (replaced_weight2 : ℝ) : 
  original_count = 20 →
  weight_increase = 5 →
  replaced_weight1 = 58 →
  replaced_weight2 = 64 →
  (original_count : ℝ) * weight_increase + replaced_weight1 + replaced_weight2 = 222 :=
by sorry

end NUMINAMATH_CALUDE_new_persons_weight_l903_90348


namespace NUMINAMATH_CALUDE_closest_ratio_l903_90340

/-- The annual interest rate -/
def interest_rate : ℝ := 0.05

/-- The number of years -/
def years : ℕ := 10

/-- The ratio of final amount to initial amount after compound interest -/
def ratio : ℝ := (1 + interest_rate) ^ years

/-- The given options for the ratio -/
def options : List ℝ := [1.5, 1.6, 1.7, 1.8]

/-- Theorem stating that 1.6 is the closest option to the actual ratio -/
theorem closest_ratio : 
  ∃ (x : ℝ), x ∈ options ∧ ∀ (y : ℝ), y ∈ options → |ratio - x| ≤ |ratio - y| ∧ x = 1.6 :=
sorry

end NUMINAMATH_CALUDE_closest_ratio_l903_90340


namespace NUMINAMATH_CALUDE_books_return_to_initial_config_l903_90339

/-- Represents the position of a book in the stack -/
def BookPosition := ℕ

/-- Represents the state of the book stack -/
def BookStack := List BookPosition

/-- Performs a single reversal operation on the top k books of the stack -/
def reverseTop (k : ℕ) (stack : BookStack) : BookStack :=
  (stack.take k).reverse ++ stack.drop k

/-- Performs a full round of reversal operations on the stack -/
def fullRound (stack : BookStack) : BookStack :=
  (List.range stack.length).foldl (fun s i => reverseTop (i + 1) s) stack

/-- Theorem: For any finite number of books, there exists a finite number of movements
    such that the books return to their initial configuration -/
theorem books_return_to_initial_config (n : ℕ) :
  ∃ M : ℕ, ∃ initial : BookStack,
    initial.length = n ∧
    (List.range M).foldl (fun s _ => fullRound s) initial = initial :=
  sorry

end NUMINAMATH_CALUDE_books_return_to_initial_config_l903_90339


namespace NUMINAMATH_CALUDE_road_trip_cost_sharing_l903_90399

/-- A road trip cost-sharing scenario -/
theorem road_trip_cost_sharing
  (alice_paid bob_paid carlos_paid : ℤ)
  (h_alice : alice_paid = 90)
  (h_bob : bob_paid = 150)
  (h_carlos : carlos_paid = 210)
  (h_split_evenly : alice_paid + bob_paid + carlos_paid = 3 * ((alice_paid + bob_paid + carlos_paid) / 3)) :
  let total := alice_paid + bob_paid + carlos_paid
  let share := total / 3
  let alice_owes := share - alice_paid
  let bob_owes := share - bob_paid
  alice_owes - bob_owes = 60 := by
sorry

end NUMINAMATH_CALUDE_road_trip_cost_sharing_l903_90399


namespace NUMINAMATH_CALUDE_select_cards_probability_l903_90327

-- Define the total number of cards
def total_cards : ℕ := 12

-- Define the number of cards for Alex's name
def alex_cards : ℕ := 4

-- Define the number of cards for Jamie's name
def jamie_cards : ℕ := 8

-- Define the number of cards to be selected
def selected_cards : ℕ := 3

-- Define the probability of selecting 2 cards from Alex's name and 1 from Jamie's name
def probability : ℚ := 12 / 55

-- Theorem statement
theorem select_cards_probability :
  (Nat.choose alex_cards 2 * Nat.choose jamie_cards 1) / Nat.choose total_cards selected_cards = probability :=
sorry

end NUMINAMATH_CALUDE_select_cards_probability_l903_90327


namespace NUMINAMATH_CALUDE_line_intersection_theorem_l903_90317

/-- The line of intersection of two planes --/
def line_of_intersection (a₁ b₁ c₁ d₁ a₂ b₂ c₂ d₂ : ℝ) :=
  {(x, y, z) : ℝ × ℝ × ℝ | a₁ * x + b₁ * y + c₁ * z + d₁ = 0 ∧
                            a₂ * x + b₂ * y + c₂ * z + d₂ = 0}

/-- The system of equations representing a line --/
def line_equation (p q r s t u : ℝ) :=
  {(x, y, z) : ℝ × ℝ × ℝ | x / p + y / q + z / r = 1 ∧
                            x / s + y / t + z / u = 1}

theorem line_intersection_theorem :
  line_of_intersection 2 3 3 (-9) 4 2 1 (-8) =
  line_equation 4.5 3 3 2 4 8 := by sorry

end NUMINAMATH_CALUDE_line_intersection_theorem_l903_90317


namespace NUMINAMATH_CALUDE_gcd_lcm_product_180_l903_90319

theorem gcd_lcm_product_180 (a b : ℕ+) :
  (Nat.gcd a b) * (Nat.lcm a b) = 180 →
  (∃ s : Finset ℕ+, s.card = 9 ∧ ∀ x, x ∈ s ↔ ∃ c d : ℕ+, (Nat.gcd c d) * (Nat.lcm c d) = 180 ∧ Nat.gcd c d = x) :=
by sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_180_l903_90319


namespace NUMINAMATH_CALUDE_intersection_at_midpoint_l903_90342

/-- Given a line segment from (3,6) to (5,10) and a line x + y = b that
    intersects this segment at its midpoint, prove that b = 12. -/
theorem intersection_at_midpoint (b : ℝ) : 
  (∃ (x y : ℝ), x + y = b ∧ 
    x = (3 + 5) / 2 ∧ 
    y = (6 + 10) / 2) → 
  b = 12 := by
sorry

end NUMINAMATH_CALUDE_intersection_at_midpoint_l903_90342


namespace NUMINAMATH_CALUDE_rental_problem_l903_90312

/-- Rental problem theorem -/
theorem rental_problem (first_hour_rate : ℕ) (additional_hour_rate : ℕ) (total_paid : ℕ) :
  first_hour_rate = 25 →
  additional_hour_rate = 10 →
  total_paid = 125 →
  ∃ (hours : ℕ), hours = 11 ∧ 
    total_paid = first_hour_rate + (hours - 1) * additional_hour_rate :=
by
  sorry

end NUMINAMATH_CALUDE_rental_problem_l903_90312


namespace NUMINAMATH_CALUDE_dave_train_books_l903_90391

/-- The number of books about trains Dave bought -/
def num_train_books (num_animal_books num_space_books cost_per_book total_spent : ℕ) : ℕ :=
  (total_spent - (num_animal_books + num_space_books) * cost_per_book) / cost_per_book

theorem dave_train_books :
  num_train_books 8 6 6 102 = 3 :=
sorry

end NUMINAMATH_CALUDE_dave_train_books_l903_90391


namespace NUMINAMATH_CALUDE_horner_v3_value_l903_90394

def horner_v3 (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ) (x : ℤ) : ℤ :=
  let v := a₆
  let v₁ := v * x + a₅
  let v₂ := v₁ * x + a₄
  v₂ * x + a₃

theorem horner_v3_value :
  horner_v3 12 35 (-8) 79 6 5 3 (-4) = -57 := by
  sorry

end NUMINAMATH_CALUDE_horner_v3_value_l903_90394


namespace NUMINAMATH_CALUDE_fraction_sum_integer_l903_90380

theorem fraction_sum_integer (n : ℕ) (h1 : n > 0) 
  (h2 : ∃ (k : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / n = k) : 
  n = 24 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_integer_l903_90380


namespace NUMINAMATH_CALUDE_room_volume_l903_90346

/-- Given a room with length three times its breadth, height twice its breadth,
    and floor area of 12 sq.m, prove that its volume is 48 cubic meters. -/
theorem room_volume (b l h : ℝ) (h1 : l = 3 * b) (h2 : h = 2 * b) (h3 : l * b = 12) :
  l * b * h = 48 := by
  sorry

end NUMINAMATH_CALUDE_room_volume_l903_90346


namespace NUMINAMATH_CALUDE_valid_placements_count_l903_90355

/-- Represents a valid placement of letters in the grid -/
def ValidPlacement := Fin 16 → Fin 2

/-- The total number of cells in the grid -/
def gridSize : Nat := 16

/-- The number of rows (or columns) in the grid -/
def gridDimension : Nat := 4

/-- The number of each letter to be placed -/
def letterCount : Nat := 2

/-- Checks if a placement is valid (no same letter in any row or column) -/
def isValidPlacement (p : ValidPlacement) : Prop := sorry

/-- Counts the number of valid placements -/
def countValidPlacements : Nat := sorry

/-- The main theorem stating the correct number of valid placements -/
theorem valid_placements_count : countValidPlacements = 3960 := by sorry

end NUMINAMATH_CALUDE_valid_placements_count_l903_90355


namespace NUMINAMATH_CALUDE_hand_towels_per_set_l903_90307

/-- The number of hand towels in a set -/
def h : ℕ := sorry

/-- The number of bath towels in a set -/
def bath_towels_per_set : ℕ := 6

/-- The smallest number of each type of towel sold -/
def min_towels_sold : ℕ := 102

theorem hand_towels_per_set :
  (∃ (n : ℕ), h * n = bath_towels_per_set * n ∧ h * n = min_towels_sold) →
  h = 17 := by sorry

end NUMINAMATH_CALUDE_hand_towels_per_set_l903_90307


namespace NUMINAMATH_CALUDE_finite_minimal_elements_l903_90367

def is_minimal {n : ℕ} (A : Set (Fin n → ℕ+)) (a : Fin n → ℕ+) : Prop :=
  a ∈ A ∧ ∀ b ∈ A, (∀ i, b i ≤ a i) → b = a

theorem finite_minimal_elements {n : ℕ} (A : Set (Fin n → ℕ+)) :
  Set.Finite {a ∈ A | is_minimal A a} := by
  sorry

end NUMINAMATH_CALUDE_finite_minimal_elements_l903_90367


namespace NUMINAMATH_CALUDE_total_paintable_area_is_1200_l903_90350

/-- Represents the dimensions of a bedroom --/
structure BedroomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total wall area of a bedroom --/
def totalWallArea (dim : BedroomDimensions) : ℝ :=
  2 * (dim.length * dim.height + dim.width * dim.height)

/-- Calculates the paintable wall area of a bedroom --/
def paintableWallArea (dim : BedroomDimensions) (nonPaintableArea : ℝ) : ℝ :=
  totalWallArea dim - nonPaintableArea

/-- The main theorem stating the total paintable area of all bedrooms --/
theorem total_paintable_area_is_1200 
  (bedroom1 : BedroomDimensions)
  (bedroom2 : BedroomDimensions)
  (bedroom3 : BedroomDimensions)
  (nonPaintable1 nonPaintable2 nonPaintable3 : ℝ) :
  bedroom1.length = 14 ∧ bedroom1.width = 11 ∧ bedroom1.height = 9 ∧
  bedroom2.length = 13 ∧ bedroom2.width = 12 ∧ bedroom2.height = 9 ∧
  bedroom3.length = 15 ∧ bedroom3.width = 10 ∧ bedroom3.height = 9 ∧
  nonPaintable1 = 50 ∧ nonPaintable2 = 55 ∧ nonPaintable3 = 45 →
  paintableWallArea bedroom1 nonPaintable1 + 
  paintableWallArea bedroom2 nonPaintable2 + 
  paintableWallArea bedroom3 nonPaintable3 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_total_paintable_area_is_1200_l903_90350


namespace NUMINAMATH_CALUDE_yoongi_has_fewer_apples_l903_90359

def jungkook_initial_apples : ℕ := 6
def jungkook_received_apples : ℕ := 3
def yoongi_apples : ℕ := 4

theorem yoongi_has_fewer_apples :
  yoongi_apples < jungkook_initial_apples + jungkook_received_apples :=
by
  sorry

end NUMINAMATH_CALUDE_yoongi_has_fewer_apples_l903_90359


namespace NUMINAMATH_CALUDE_root_transformation_l903_90323

theorem root_transformation {b : ℝ} (a b c d : ℝ) :
  (a^4 - b*a - 3 = 0) ∧
  (b^4 - b*b - 3 = 0) ∧
  (c^4 - b*c - 3 = 0) ∧
  (d^4 - b*d - 3 = 0) →
  (3*(-1/a)^4 - b*(-1/a)^3 - 1 = 0) ∧
  (3*(-1/b)^4 - b*(-1/b)^3 - 1 = 0) ∧
  (3*(-1/c)^4 - b*(-1/c)^3 - 1 = 0) ∧
  (3*(-1/d)^4 - b*(-1/d)^3 - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_transformation_l903_90323


namespace NUMINAMATH_CALUDE_pencils_left_l903_90316

/-- The number of pencils in a dozen -/
def pencils_per_dozen : ℕ := 12

/-- The initial number of dozens of pencils -/
def initial_dozens : ℕ := 3

/-- The number of students -/
def num_students : ℕ := 11

/-- The number of pencils each student takes -/
def pencils_per_student : ℕ := 3

/-- Theorem stating that after students take pencils, 3 pencils are left -/
theorem pencils_left : 
  initial_dozens * pencils_per_dozen - num_students * pencils_per_student = 3 := by
  sorry

end NUMINAMATH_CALUDE_pencils_left_l903_90316


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l903_90337

theorem smallest_n_congruence (n : ℕ+) : 
  (∀ k : ℕ+, k < n → (7 ^ k.val) % 4 ≠ (k.val ^ 7) % 4) ∧ 
  (7 ^ n.val) % 4 = (n.val ^ 7) % 4 → 
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l903_90337


namespace NUMINAMATH_CALUDE_max_plates_on_table_l903_90325

/-- The radius of the table in meters -/
def table_radius : ℝ := 1

/-- The radius of each plate in meters -/
def plate_radius : ℝ := 0.15

/-- The maximum number of plates that can fit on the table -/
def max_plates : ℕ := 44

/-- Theorem stating that the maximum number of plates that can fit on the table is 44 -/
theorem max_plates_on_table :
  ∀ k : ℕ, 
    (k : ℝ) * π * plate_radius^2 ≤ π * table_radius^2 ↔ k ≤ max_plates :=
by sorry

end NUMINAMATH_CALUDE_max_plates_on_table_l903_90325


namespace NUMINAMATH_CALUDE_video_game_pricing_l903_90311

theorem video_game_pricing (total_games : ℕ) (non_working_games : ℕ) (total_earnings : ℕ) :
  total_games = 16 →
  non_working_games = 8 →
  total_earnings = 56 →
  (total_earnings : ℚ) / (total_games - non_working_games : ℚ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_video_game_pricing_l903_90311


namespace NUMINAMATH_CALUDE_repeating_decimal_three_thirty_six_l903_90374

/-- The repeating decimal 3.363636... is equal to 10/3 -/
theorem repeating_decimal_three_thirty_six : ∃ (x : ℚ), x = 10 / 3 ∧ x = 3 + (36 / 99) := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_three_thirty_six_l903_90374


namespace NUMINAMATH_CALUDE_kates_hair_length_l903_90308

theorem kates_hair_length (logan_hair emily_hair kate_hair : ℝ) : 
  logan_hair = 20 →
  emily_hair = logan_hair + 6 →
  kate_hair = emily_hair / 2 →
  kate_hair = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_kates_hair_length_l903_90308


namespace NUMINAMATH_CALUDE_sum_of_squares_l903_90387

/-- Given a sequence {aₙ} where the sum of its first n terms S = 2n - 1,
    T is the sum of the first n terms of the sequence {aₙ²} -/
def T (n : ℕ) : ℚ :=
  (16^n - 1) / 15

/-- The sum of the first n terms of the original sequence -/
def S (n : ℕ) : ℕ :=
  2 * n - 1

/-- Theorem stating that T is the correct sum for the sequence {aₙ²} -/
theorem sum_of_squares (n : ℕ) : T n = (16^n - 1) / 15 :=
  by sorry

end NUMINAMATH_CALUDE_sum_of_squares_l903_90387


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l903_90396

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecular_weight (ca_count : ℕ) (o_count : ℕ) (h_count : ℕ) 
                     (ca_weight : ℝ) (o_weight : ℝ) (h_weight : ℝ) : ℝ :=
  ca_count * ca_weight + o_count * o_weight + h_count * h_weight

/-- Theorem stating that the molecular weight of the given compound is 74.094 g/mol -/
theorem compound_molecular_weight : 
  molecular_weight 1 2 2 40.08 15.999 1.008 = 74.094 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l903_90396


namespace NUMINAMATH_CALUDE_infinitely_many_primes_composite_l903_90330

theorem infinitely_many_primes_composite (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ¬Nat.Prime (a * p + b)} :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_composite_l903_90330


namespace NUMINAMATH_CALUDE_sin_4theta_from_exp_itheta_l903_90397

theorem sin_4theta_from_exp_itheta (θ : ℝ) :
  Complex.exp (θ * Complex.I) = (2 + Complex.I * Real.sqrt 5) / 3 →
  Real.sin (4 * θ) = -8 * Real.sqrt 5 / 81 := by
  sorry

end NUMINAMATH_CALUDE_sin_4theta_from_exp_itheta_l903_90397


namespace NUMINAMATH_CALUDE_line_slope_l903_90365

/-- The slope of a line given by the equation 3y - (1/2)x = 9 is 1/6 -/
theorem line_slope (x y : ℝ) : 3 * y - (1/2) * x = 9 → (y - 3) / x = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l903_90365


namespace NUMINAMATH_CALUDE_rhombuses_in_grid_of_25_l903_90332

/-- Represents a triangular grid of equilateral triangles -/
structure TriangularGrid where
  side_length : ℕ
  total_triangles : ℕ

/-- Calculates the number of rhombuses in a triangular grid -/
def count_rhombuses (grid : TriangularGrid) : ℕ :=
  3 * (grid.side_length - 1) * grid.side_length

/-- Theorem: In a triangular grid with 25 triangles (5 per side), there are 30 rhombuses -/
theorem rhombuses_in_grid_of_25 :
  let grid : TriangularGrid := { side_length := 5, total_triangles := 25 }
  count_rhombuses grid = 30 := by
  sorry


end NUMINAMATH_CALUDE_rhombuses_in_grid_of_25_l903_90332
