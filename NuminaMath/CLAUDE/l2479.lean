import Mathlib

namespace NUMINAMATH_CALUDE_prime_between_squares_l2479_247901

theorem prime_between_squares : ∃! p : ℕ, 
  Prime p ∧ 
  ∃ n : ℕ, p - 5 = n^2 ∧ p + 8 = (n + 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_prime_between_squares_l2479_247901


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2479_247970

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {(3, -1)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2479_247970


namespace NUMINAMATH_CALUDE_f_five_zeros_a_range_l2479_247938

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 2 then
    Real.log x ^ 2 - floor (Real.log x) - 2
  else if x ≤ 0 then
    Real.exp (-x) - a * x - 1
  else
    0  -- This case is not specified in the original problem, so we set it to 0

-- State the theorem
theorem f_five_zeros_a_range (a : ℝ) :
  (∃ (s : Finset ℝ), s.card = 5 ∧ ∀ x ∈ s, f a x = 0) →
  a ∈ Set.Iic (-1 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_f_five_zeros_a_range_l2479_247938


namespace NUMINAMATH_CALUDE_equation_solution_l2479_247967

theorem equation_solution :
  let y : ℚ := 20 / 7
  2 / y + (3 / y) / (6 / y) = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2479_247967


namespace NUMINAMATH_CALUDE_committee_formation_count_l2479_247954

/-- The number of ways to form a committee with specific requirements -/
theorem committee_formation_count : ∀ (n m k : ℕ),
  n ≥ m ∧ m ≥ k ∧ k ≥ 2 →
  (Nat.choose (n - 2) (k - 2) : ℕ) = Nat.choose n m →
  n = 12 ∧ m = 5 ∧ k = 3 →
  Nat.choose (n - 2) (k - 2) = 120 := by
  sorry

#check committee_formation_count

end NUMINAMATH_CALUDE_committee_formation_count_l2479_247954


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2479_247960

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (∀ a b, a + b ≠ 3 → (a ≠ 1 ∨ b ≠ 2)) ∧
  (∃ a b, (a ≠ 1 ∨ b ≠ 2) ∧ a + b = 3) := by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2479_247960


namespace NUMINAMATH_CALUDE_inequality_proof_l2479_247904

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a * b + b * c + c * a = 1) : 
  1 / (a^2 + 1) + 1 / (b^2 + 1) + 1 / (c^2 + 1) ≤ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2479_247904


namespace NUMINAMATH_CALUDE_white_is_lightest_l2479_247912

-- Define the puppy type
inductive Puppy
| White
| Black
| Yellowy
| Spotted

-- Define the "lighter than" relation
def lighterThan : Puppy → Puppy → Prop := sorry

-- State the theorem
theorem white_is_lightest :
  (lighterThan Puppy.White Puppy.Black) ∧
  (lighterThan Puppy.Black Puppy.Yellowy) ∧
  (lighterThan Puppy.Yellowy Puppy.Spotted) →
  ∀ p : Puppy, p ≠ Puppy.White → lighterThan Puppy.White p :=
sorry

end NUMINAMATH_CALUDE_white_is_lightest_l2479_247912


namespace NUMINAMATH_CALUDE_calculator_purchase_theorem_l2479_247985

/-- Represents the unit price of a type A calculator -/
def price_A : ℝ := 110

/-- Represents the unit price of a type B calculator -/
def price_B : ℝ := 120

/-- Represents the total number of calculators to be purchased -/
def total_calculators : ℕ := 100

/-- Theorem stating the properties of calculator prices and minimum purchase cost -/
theorem calculator_purchase_theorem :
  (price_B = price_A + 10) ∧
  (550 / price_A = 600 / price_B) ∧
  (∀ a b : ℕ, a + b = total_calculators → b ≤ 3 * a →
    price_A * a + price_B * b ≥ 11000) :=
by sorry

end NUMINAMATH_CALUDE_calculator_purchase_theorem_l2479_247985


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2479_247924

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (5 * x₁^2 + 20 * x₁ - 25 = 0) →
  (5 * x₂^2 + 20 * x₂ - 25 = 0) →
  (x₁ ≠ x₂) →
  x₁^2 + x₂^2 = 26 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2479_247924


namespace NUMINAMATH_CALUDE_base_k_representation_l2479_247977

/-- Converts a base-k number represented as a list of digits to its decimal value. -/
def baseKToDecimal (digits : List Nat) (k : Nat) : Nat :=
  digits.foldr (fun d acc => d + k * acc) 0

/-- Checks if a given list of digits is a valid representation in base-k. -/
def isValidBaseK (digits : List Nat) (k : Nat) : Prop :=
  digits.all (· < k)

theorem base_k_representation (k : Nat) :
  (k > 0 ∧ baseKToDecimal [1, 3, 2] k = 30 ∧ isValidBaseK [1, 3, 2] k) ↔ k = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_k_representation_l2479_247977


namespace NUMINAMATH_CALUDE_division_problem_l2479_247969

theorem division_problem (dividend quotient remainder divisor : ℕ) 
  (h1 : dividend = 271)
  (h2 : quotient = 9)
  (h3 : remainder = 1)
  (h4 : dividend = divisor * quotient + remainder) :
  divisor = 30 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2479_247969


namespace NUMINAMATH_CALUDE_chord_length_implies_a_values_l2479_247956

theorem chord_length_implies_a_values (a : ℝ) : 
  (∃ (x y : ℝ), (x - a)^2 + y^2 = 4 ∧ x - y = 1) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁ - a)^2 + y₁^2 = 4 ∧ x₁ - y₁ = 1 ∧
    (x₂ - a)^2 + y₂^2 = 4 ∧ x₂ - y₂ = 1 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 8) →
  a = -1 ∨ a = 3 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_implies_a_values_l2479_247956


namespace NUMINAMATH_CALUDE_mean_of_quiz_scores_l2479_247930

def quiz_scores : List ℝ := [86, 91, 89, 95, 88, 94]

theorem mean_of_quiz_scores :
  (quiz_scores.sum / quiz_scores.length : ℝ) = 90.5 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_quiz_scores_l2479_247930


namespace NUMINAMATH_CALUDE_graduates_not_both_l2479_247927

def biotechnology_class (total_graduates : ℕ) (both_job_and_degree : ℕ) : Prop :=
  total_graduates - both_job_and_degree = 60

theorem graduates_not_both : biotechnology_class 73 13 :=
  sorry

end NUMINAMATH_CALUDE_graduates_not_both_l2479_247927


namespace NUMINAMATH_CALUDE_youngest_child_age_l2479_247944

/-- Represents the age of the youngest child in a group of 5 children -/
def youngest_age (total_age : ℕ) : ℕ :=
  (total_age - 20) / 5

/-- Theorem stating that if the sum of ages of 5 children born at 2-year intervals is 50,
    then the age of the youngest child is 6 years -/
theorem youngest_child_age :
  youngest_age 50 = 6 := by
  sorry

end NUMINAMATH_CALUDE_youngest_child_age_l2479_247944


namespace NUMINAMATH_CALUDE_bagel_store_spending_l2479_247948

theorem bagel_store_spending :
  ∀ (B D : ℝ),
  D = (7/10) * B →
  B = D + 15 →
  B + D = 85 :=
by
  sorry

end NUMINAMATH_CALUDE_bagel_store_spending_l2479_247948


namespace NUMINAMATH_CALUDE_expression_evaluation_l2479_247976

theorem expression_evaluation :
  let x : ℚ := -2
  let y : ℚ := 2
  (1/2 * x - 2 * (x - 1/3 * y^2) + (-3/2 * x + 1/3 * y^2)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2479_247976


namespace NUMINAMATH_CALUDE_probability_two_sunny_days_out_of_five_l2479_247962

theorem probability_two_sunny_days_out_of_five :
  let n : ℕ := 5  -- total number of days
  let k : ℕ := 2  -- number of sunny days we want
  let p : ℚ := 1/4  -- probability of a sunny day (1 - probability of rain)
  let q : ℚ := 3/4  -- probability of a rainy day
  (n.choose k : ℚ) * p^k * q^(n - k) = 135/512 :=
sorry

end NUMINAMATH_CALUDE_probability_two_sunny_days_out_of_five_l2479_247962


namespace NUMINAMATH_CALUDE_ship_distance_constant_l2479_247992

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a semicircular path -/
structure SemicircularPath where
  center : Point
  radius : ℝ

/-- Represents the ship's journey -/
structure ShipJourney where
  path1 : SemicircularPath
  path2 : SemicircularPath

/-- Represents the ship's position along its journey -/
structure ShipPosition where
  t : ℝ  -- Time parameter (0 ≤ t ≤ 2)
  isOnFirstPath : Bool

/-- Distance function for the ship's position -/
def distance (journey : ShipJourney) (pos : ShipPosition) : ℝ :=
  if pos.isOnFirstPath then journey.path1.radius else journey.path2.radius

theorem ship_distance_constant (journey : ShipJourney) :
  ∀ t1 t2 : ℝ, 0 ≤ t1 ∧ t1 ≤ 1 → 0 ≤ t2 ∧ t2 ≤ 1 →
    distance journey { t := t1, isOnFirstPath := true } =
    distance journey { t := t2, isOnFirstPath := true } ∧
  ∀ t3 t4 : ℝ, 1 < t3 ∧ t3 ≤ 2 → 1 < t4 ∧ t4 ≤ 2 →
    distance journey { t := t3, isOnFirstPath := false } =
    distance journey { t := t4, isOnFirstPath := false } ∧
  journey.path1.radius ≠ journey.path2.radius →
    ∃ t5 t6 : ℝ, 0 ≤ t5 ∧ t5 ≤ 1 ∧ 1 < t6 ∧ t6 ≤ 2 ∧
      distance journey { t := t5, isOnFirstPath := true } ≠
      distance journey { t := t6, isOnFirstPath := false } :=
by
  sorry

end NUMINAMATH_CALUDE_ship_distance_constant_l2479_247992


namespace NUMINAMATH_CALUDE_train_length_l2479_247996

/-- The length of a train given its speed, time to cross a bridge, and bridge length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  bridge_length = 265 →
  train_speed * crossing_time - bridge_length = 110 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2479_247996


namespace NUMINAMATH_CALUDE_ramon_age_in_twenty_years_ramon_current_age_l2479_247984

/-- Ramon's current age -/
def ramon_age : ℕ := 26

/-- Loui's current age -/
def loui_age : ℕ := 23

/-- In twenty years, Ramon will be twice as old as Loui is today -/
theorem ramon_age_in_twenty_years (ramon_age loui_age : ℕ) :
  ramon_age + 20 = 2 * loui_age := by sorry

theorem ramon_current_age : ramon_age = 26 := by sorry

end NUMINAMATH_CALUDE_ramon_age_in_twenty_years_ramon_current_age_l2479_247984


namespace NUMINAMATH_CALUDE_solve_equation_l2479_247966

theorem solve_equation (x y : ℤ) 
  (h1 : x^2 - 3*x + 6 = y + 2) 
  (h2 : x = -8) : 
  y = 92 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l2479_247966


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_l2479_247936

theorem sum_of_fifth_powers (a b c d : ℝ) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (sum_condition : a + b + c + d = 3) 
  (sum_of_squares : a^2 + b^2 + c^2 + d^2 = 45) : 
  (a^5 / ((a-b)*(a-c)*(a-d))) + (b^5 / ((b-a)*(b-c)*(b-d))) + 
  (c^5 / ((c-a)*(c-b)*(c-d))) + (d^5 / ((d-a)*(d-b)*(d-c))) = -9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_l2479_247936


namespace NUMINAMATH_CALUDE_customers_left_l2479_247941

theorem customers_left (initial_customers : ℕ) (remaining_tables : ℕ) (people_per_table : ℕ) : 
  initial_customers = 21 → 
  remaining_tables = 3 → 
  people_per_table = 3 → 
  initial_customers - (remaining_tables * people_per_table) = 12 := by
sorry

end NUMINAMATH_CALUDE_customers_left_l2479_247941


namespace NUMINAMATH_CALUDE_quadratic_function_ellipse_l2479_247932

/-- Given a quadratic function y = ax^2 + bx + c where ac ≠ 0,
    with vertex (-b/(2a), -1/(4a)),
    and intersections with x-axis on opposite sides of y-axis,
    prove that (b, c) lies on the ellipse b^2 + c^2/4 = 1 --/
theorem quadratic_function_ellipse (a b c : ℝ) (h1 : a ≠ 0) (h2 : c ≠ 0) :
  (∃ (p q : ℝ), p < 0 ∧ q > 0 ∧ a * p^2 + b * p + c = 0 ∧ a * q^2 + b * q + c = 0) →
  (∃ (m : ℝ), m = -4 ∧ (b / (2 * a))^2 + m^2 = ((b^2 - 4 * a * c) / (4 * a^2))) →
  b^2 + c^2 / 4 = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_ellipse_l2479_247932


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2479_247925

theorem solution_set_inequality (x : ℝ) :
  (Set.Icc 1 2 : Set ℝ) = {x | -x^2 + 3*x - 2 ≥ 0} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2479_247925


namespace NUMINAMATH_CALUDE_bag_problem_l2479_247950

/-- Represents the number of balls in the bag -/
def total_balls : ℕ := 6

/-- Represents the probability of drawing at least 1 white ball when drawing 2 balls -/
def prob_at_least_one_white : ℚ := 4/5

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- Represents the number of white balls in the bag -/
def num_white_balls : ℕ := sorry

/-- Calculates the probability of drawing exactly k white balls when drawing 2 balls -/
def prob_k_white (k : ℕ) : ℚ := sorry

/-- Calculates the mathematical expectation of the number of white balls drawn -/
def expectation : ℚ := sorry

theorem bag_problem :
  (1 - (choose (total_balls - num_white_balls) 2 : ℚ) / (choose total_balls 2 : ℚ) = prob_at_least_one_white) →
  (num_white_balls = 3 ∧ expectation = 1) :=
by sorry

end NUMINAMATH_CALUDE_bag_problem_l2479_247950


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2479_247922

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 2 - y^2 / 4 = -1

-- Define the asymptote equation
def asymptote (x y : ℝ) : Prop := y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x

-- Theorem statement
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, (∃ ε > 0, ∀ x' y' : ℝ, hyperbola x' y' ∧ x'^2 + y'^2 > 1/ε^2 →
    |y' - (Real.sqrt 2 * x')| < ε ∨ |y' - (-Real.sqrt 2 * x')| < ε) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2479_247922


namespace NUMINAMATH_CALUDE_parallel_planes_lines_relationship_l2479_247935

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallel : Plane → Plane → Prop)

-- Define the containment relation for lines in planes
variable (contained_in : Line → Plane → Prop)

-- Define the positional relationships between lines
variable (is_parallel : Line → Line → Prop)
variable (is_skew : Line → Line → Prop)

-- State the theorem
theorem parallel_planes_lines_relationship 
  (α β : Plane) (a b : Line) 
  (h1 : parallel α β) 
  (h2 : contained_in a α) 
  (h3 : contained_in b β) : 
  is_parallel a b ∨ is_skew a b :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_lines_relationship_l2479_247935


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2479_247952

theorem polynomial_evaluation (f : ℝ → ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, f x = (1 - 3*x) * (1 + x)^5) →
  (∀ x, f x = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₀ + (1/3)*a₁ + (1/3^2)*a₂ + (1/3^3)*a₃ + (1/3^4)*a₄ + (1/3^5)*a₅ + (1/3^6)*a₆ = 0 :=
by sorry


end NUMINAMATH_CALUDE_polynomial_evaluation_l2479_247952


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2479_247974

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the properties of the sequence
theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 3 * a 7 = 2 * (a 4)^2 →
  a 3 = 1 →
  a 2 = Real.sqrt 2 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2479_247974


namespace NUMINAMATH_CALUDE_philips_school_days_l2479_247914

/-- Given the following conditions:
  - The distance from Philip's house to school is 2.5 miles
  - The distance from Philip's house to the market is 2 miles
  - Philip makes two round trips to school each day he goes to school
  - Philip makes one round trip to the market during weekends
  - Philip's car's mileage for a typical week is 44 miles

  Prove that Philip makes round trips to school 4 days a week. -/
theorem philips_school_days :
  ∀ (school_distance market_distance : ℚ)
    (daily_school_trips weekly_market_trips : ℕ)
    (weekly_mileage : ℚ),
  school_distance = 5/2 →
  market_distance = 2 →
  daily_school_trips = 2 →
  weekly_market_trips = 1 →
  weekly_mileage = 44 →
  ∃ (days : ℕ),
    days = 4 ∧
    weekly_mileage = (2 * school_distance * daily_school_trips * days : ℚ) + (2 * market_distance * weekly_market_trips) :=
by sorry

end NUMINAMATH_CALUDE_philips_school_days_l2479_247914


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2479_247971

open Set

/-- The solution set of the inequality -x^2 + ax + b ≥ 0 -/
def SolutionSet (a b : ℝ) : Set ℝ := {x | -x^2 + a*x + b ≥ 0}

/-- The theorem stating the equivalence of the solution sets -/
theorem solution_set_equivalence (a b : ℝ) :
  SolutionSet a b = Icc (-2) 3 →
  {x : ℝ | x^2 - 5*a*x + b > 0} = {x : ℝ | x < 2 ∨ x > 3} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2479_247971


namespace NUMINAMATH_CALUDE_fraction_product_one_l2479_247965

def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem fraction_product_one : 
  ∃ (a b c d e f : ℕ), 
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    Nat.gcd a b = 1 ∧ Nat.gcd c d = 1 ∧ Nat.gcd e f = 1 ∧
    (a * c * e : ℚ) / (b * d * f : ℚ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_one_l2479_247965


namespace NUMINAMATH_CALUDE_max_value_sqrt_quadratic_l2479_247990

theorem max_value_sqrt_quadratic :
  ∃ (max : ℝ), max = 9/2 ∧
  ∀ a : ℝ, -6 ≤ a ∧ a ≤ 3 →
    Real.sqrt ((3 - a) * (a + 6)) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_quadratic_l2479_247990


namespace NUMINAMATH_CALUDE_ellipse_tangent_intersection_l2479_247926

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line that P moves along
def line_P (x y : ℝ) : Prop := x + y = 3

-- Define a point on the ellipse
def point_on_ellipse (x y : ℝ) : Prop := ellipse x y

-- Define the tangent line at a point (x₀, y₀) on the ellipse
def tangent_line (x₀ y₀ x y : ℝ) : Prop :=
  point_on_ellipse x₀ y₀ → x₀*x/4 + y₀*y/3 = 1

-- Theorem statement
theorem ellipse_tangent_intersection :
  ∀ x₀ y₀ x₁ y₁ x₂ y₂,
    line_P x₀ y₀ →
    point_on_ellipse x₁ y₁ →
    point_on_ellipse x₂ y₂ →
    tangent_line x₁ y₁ x₀ y₀ →
    tangent_line x₂ y₂ x₀ y₀ →
    ∃ t, t*x₁ + (1-t)*x₂ = 4/3 ∧ t*y₁ + (1-t)*y₂ = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_tangent_intersection_l2479_247926


namespace NUMINAMATH_CALUDE_tessellating_nonagon_angles_l2479_247991

/-- A nonagon that tessellates the plane and can be decomposed into seven triangles -/
structure TessellatingNonagon where
  /-- The vertices of the nonagon -/
  vertices : Fin 9 → ℝ × ℝ
  /-- The nonagon tessellates the plane -/
  tessellates : sorry
  /-- The nonagon can be decomposed into seven triangles -/
  decomposable : sorry
  /-- Some sides of the nonagon form rhombuses with equal side lengths -/
  has_rhombuses : sorry

/-- The angles of a tessellating nonagon -/
def nonagon_angles (n : TessellatingNonagon) : Fin 9 → ℝ := sorry

/-- Theorem stating the angles of the tessellating nonagon -/
theorem tessellating_nonagon_angles (n : TessellatingNonagon) :
  nonagon_angles n = ![105, 60, 195, 195, 195, 15, 165, 165, 165] := by sorry

end NUMINAMATH_CALUDE_tessellating_nonagon_angles_l2479_247991


namespace NUMINAMATH_CALUDE_problems_per_page_problems_per_page_is_four_l2479_247945

theorem problems_per_page : ℕ → Prop :=
  fun p =>
    let math_pages : ℕ := 4
    let reading_pages : ℕ := 6
    let total_pages : ℕ := math_pages + reading_pages
    let total_problems : ℕ := 40
    total_pages * p = total_problems → p = 4

-- The proof is omitted
theorem problems_per_page_is_four : problems_per_page 4 := by sorry

end NUMINAMATH_CALUDE_problems_per_page_problems_per_page_is_four_l2479_247945


namespace NUMINAMATH_CALUDE_orthocenter_tangents_collinear_l2479_247943

/-- Representation of a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Representation of a triangle -/
structure Triangle :=
  (A : Point) (B : Point) (C : Point)

/-- Definition of an acute-angled triangle -/
def isAcuteAngled (t : Triangle) : Prop := sorry

/-- Definition of the orthocenter of a triangle -/
def orthocenter (t : Triangle) : Point := sorry

/-- Definition of a circle -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Definition of a tangent line to a circle -/
def isTangent (p : Point) (c : Circle) : Prop := sorry

/-- Definition of collinearity -/
def areCollinear (p1 p2 p3 : Point) : Prop := sorry

/-- Main theorem -/
theorem orthocenter_tangents_collinear 
  (t : Triangle) 
  (h_acute : isAcuteAngled t) 
  (H : Point) 
  (h_ortho : H = orthocenter t) 
  (c : Circle) 
  (h_circle : c.center = Point.mk ((t.B.x + t.C.x) / 2) ((t.B.y + t.C.y) / 2) ∧ 
              c.radius = (((t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2)^(1/2)) / 2) 
  (P Q : Point) 
  (h_tangent_P : isTangent P c ∧ (∃ k : ℝ, P = Point.mk (t.A.x + k * (P.x - t.A.x)) (t.A.y + k * (P.y - t.A.y))))
  (h_tangent_Q : isTangent Q c ∧ (∃ k : ℝ, Q = Point.mk (t.A.x + k * (Q.x - t.A.x)) (t.A.y + k * (Q.y - t.A.y))))
  : areCollinear P H Q := 
sorry

end NUMINAMATH_CALUDE_orthocenter_tangents_collinear_l2479_247943


namespace NUMINAMATH_CALUDE_extremum_of_f_on_M_l2479_247957

def M : Set ℝ := {x | x^2 + 4*x ≤ 0}

def f (x : ℝ) : ℝ := -x^2 - 6*x + 1

theorem extremum_of_f_on_M :
  ∃ (min max : ℝ), 
    (∀ x ∈ M, f x ≥ min) ∧ 
    (∃ x ∈ M, f x = min) ∧
    (∀ x ∈ M, f x ≤ max) ∧ 
    (∃ x ∈ M, f x = max) ∧
    min = 1 ∧ max = 10 :=
sorry

end NUMINAMATH_CALUDE_extremum_of_f_on_M_l2479_247957


namespace NUMINAMATH_CALUDE_cube_occupation_percentage_l2479_247993

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the number of cubes that can fit along a given dimension -/
def cubesFit (dimension cubeSide : ℕ) : ℕ :=
  dimension / cubeSide

/-- Calculates the volume occupied by cubes in the box -/
def occupiedVolume (d : BoxDimensions) (cubeSide : ℕ) : ℕ :=
  (cubesFit d.length cubeSide) * (cubesFit d.width cubeSide) * (cubesFit d.height cubeSide) * (cubeSide ^ 3)

/-- Theorem: The percentage of volume occupied by 4-inch cubes in a 
    8x7x12 inch box is equal to 4/7 -/
theorem cube_occupation_percentage :
  let boxDim : BoxDimensions := { length := 8, width := 7, height := 12 }
  let cubeSide : ℕ := 4
  (occupiedVolume boxDim cubeSide : ℚ) / (boxVolume boxDim : ℚ) = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_cube_occupation_percentage_l2479_247993


namespace NUMINAMATH_CALUDE_hockey_tournament_points_l2479_247918

/-- The number of teams in the tournament -/
def num_teams : ℕ := 2016

/-- The number of points awarded for a win -/
def points_per_win : ℕ := 3

/-- The total number of games played in the tournament -/
def total_games : ℕ := num_teams * (num_teams - 1) / 2

/-- The total number of points awarded in the tournament -/
def total_points : ℕ := total_games * points_per_win

theorem hockey_tournament_points :
  total_points = 6093360 := by sorry

end NUMINAMATH_CALUDE_hockey_tournament_points_l2479_247918


namespace NUMINAMATH_CALUDE_first_month_sale_is_3435_l2479_247947

/-- Calculates the sale in the first month given the sales for the next 5 months and the average sale -/
def first_month_sale (sale2 sale3 sale4 sale5 sale6 average : ℕ) : ℕ :=
  6 * average - (sale2 + sale3 + sale4 + sale5 + sale6)

/-- The sale in the first month is 3435 given the conditions of the problem -/
theorem first_month_sale_is_3435 :
  first_month_sale 3927 3855 4230 3562 1991 3500 = 3435 := by
sorry

#eval first_month_sale 3927 3855 4230 3562 1991 3500

end NUMINAMATH_CALUDE_first_month_sale_is_3435_l2479_247947


namespace NUMINAMATH_CALUDE_identical_cuts_different_shapes_l2479_247937

/-- Represents a polygon --/
structure Polygon where
  area : ℝ

/-- Represents a square --/
structure Square where
  side : ℝ

/-- Represents a triangle --/
structure Triangle where
  base : ℝ
  height : ℝ

/-- The theorem stating that it's possible to cut identical pieces from two identical polygons
    such that one remaining shape is a square and the other is a triangle --/
theorem identical_cuts_different_shapes (original : Polygon) :
  ∃ (cut_piece : ℝ) (square : Square) (triangle : Triangle),
    original.area = square.side ^ 2 + cut_piece ∧
    original.area = (1 / 2) * triangle.base * triangle.height + cut_piece ∧
    square.side ^ 2 = (1 / 2) * triangle.base * triangle.height :=
sorry

end NUMINAMATH_CALUDE_identical_cuts_different_shapes_l2479_247937


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l2479_247995

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem largest_prime_divisor_of_factorial_sum :
  ∃ (p : ℕ), is_prime p ∧ 
    p ∣ (factorial 10 + factorial 11) ∧ 
    ∀ (q : ℕ), is_prime q → q ∣ (factorial 10 + factorial 11) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l2479_247995


namespace NUMINAMATH_CALUDE_symmetric_increasing_function_property_l2479_247981

/-- A function that is increasing on (-∞, 2) and its graph shifted by 2 is symmetric about x=0 -/
def symmetric_increasing_function (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y ∧ y < 2 → f x < f y) ∧
  (∀ x, f (x + 2) = f (2 - x))

/-- If f is a symmetric increasing function, then f(0) < f(3) -/
theorem symmetric_increasing_function_property (f : ℝ → ℝ) 
  (h : symmetric_increasing_function f) : f 0 < f 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_increasing_function_property_l2479_247981


namespace NUMINAMATH_CALUDE_prob_different_suits_l2479_247972

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the suits in a standard deck -/
inductive Suit
  | Hearts | Diamonds | Clubs | Spades

/-- Function to get the suit of a card -/
def getSuit (card : Fin 52) : Suit := sorry

/-- Probability of drawing three cards of different suits -/
def probDifferentSuits (d : Deck) : ℚ :=
  (39 : ℚ) / 51 * (26 : ℚ) / 50

/-- Theorem stating the probability of drawing three cards of different suits -/
theorem prob_different_suits (d : Deck) :
  probDifferentSuits d = 169 / 425 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_suits_l2479_247972


namespace NUMINAMATH_CALUDE_boy_walking_time_l2479_247917

/-- If a boy walks at 3/2 of his usual rate and arrives 4 minutes early, 
    his usual time to reach school is 12 minutes. -/
theorem boy_walking_time (usual_rate : ℝ) (usual_time : ℝ) 
    (h1 : usual_rate > 0) (h2 : usual_time > 0) : 
    (3 / 2 : ℝ) * usual_rate * (usual_time - 4) = usual_rate * usual_time → 
    usual_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_boy_walking_time_l2479_247917


namespace NUMINAMATH_CALUDE_chimney_bricks_l2479_247973

/-- The number of bricks in the chimney -/
def h : ℕ := 360

/-- Brenda's time to build the chimney alone (in hours) -/
def brenda_time : ℕ := 8

/-- Brandon's time to build the chimney alone (in hours) -/
def brandon_time : ℕ := 12

/-- Efficiency decrease when working together (in bricks per hour) -/
def efficiency_decrease : ℕ := 15

/-- Time taken to build the chimney together (in hours) -/
def time_together : ℕ := 6

theorem chimney_bricks : 
  time_together * ((h / brenda_time + h / brandon_time) - efficiency_decrease) = h := by
  sorry

#check chimney_bricks

end NUMINAMATH_CALUDE_chimney_bricks_l2479_247973


namespace NUMINAMATH_CALUDE_freezer_temperature_l2479_247911

/-- Given a refrigerator with a refrigeration compartment and a freezer compartment,
    this theorem proves that if the refrigeration compartment is at 4°C and
    the freezer is 22°C colder, then the freezer temperature is -18°C. -/
theorem freezer_temperature
  (temp_refrigeration : ℝ)
  (temp_difference : ℝ)
  (h1 : temp_refrigeration = 4)
  (h2 : temp_difference = 22)
  : temp_refrigeration - temp_difference = -18 := by
  sorry

end NUMINAMATH_CALUDE_freezer_temperature_l2479_247911


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2479_247933

theorem absolute_value_equation_solution :
  ∀ x : ℝ, |x - 5| = 3 * x + 1 ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2479_247933


namespace NUMINAMATH_CALUDE_jackson_and_williams_money_l2479_247949

theorem jackson_and_williams_money (jackson_money : ℝ) (williams_money : ℝ) :
  jackson_money = 125 →
  jackson_money = 5 * williams_money →
  jackson_money + williams_money = 145.83 :=
by sorry

end NUMINAMATH_CALUDE_jackson_and_williams_money_l2479_247949


namespace NUMINAMATH_CALUDE_sixth_finger_is_one_l2479_247988

def f : ℕ → ℕ
| 2 => 1
| 1 => 8
| 8 => 7
| 7 => 2
| _ => 0  -- Default case for other inputs

def finger_sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 2  -- Start with 2 on the first finger (index 0)
  | n + 1 => f (finger_sequence n)

theorem sixth_finger_is_one : finger_sequence 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sixth_finger_is_one_l2479_247988


namespace NUMINAMATH_CALUDE_no_natural_solution_for_equation_l2479_247923

theorem no_natural_solution_for_equation : ¬∃ (a b : ℕ), a^2 - 3*b^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_for_equation_l2479_247923


namespace NUMINAMATH_CALUDE_binary_arithmetic_equality_l2479_247919

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binaryToNat (bits : List Bool) : Nat :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Represents a binary number as a list of bits. -/
def binary (bits : List Bool) : Nat := binaryToNat bits

theorem binary_arithmetic_equality : 
  (binary [true, false, true, true, true, false] + binary [true, false, true, false, true]) -
  (binary [true, true, true, false, false, false] - binary [true, true, false, true, false, true]) +
  binary [true, true, true, false, true] =
  binary [true, false, true, true, true, false, true] := by
  sorry

#eval binary [true, false, true, true, true, false, true]

end NUMINAMATH_CALUDE_binary_arithmetic_equality_l2479_247919


namespace NUMINAMATH_CALUDE_sum_of_real_and_imaginary_parts_l2479_247951

theorem sum_of_real_and_imaginary_parts : ∃ (z : ℂ), z = 3 - 4*I ∧ z.re + z.im = -1 :=
sorry

end NUMINAMATH_CALUDE_sum_of_real_and_imaginary_parts_l2479_247951


namespace NUMINAMATH_CALUDE_probability_at_least_one_boy_one_girl_l2479_247989

theorem probability_at_least_one_boy_one_girl (p : ℝ) : 
  p = 1/2 → (1 - 2 * p^4) = 7/8 := by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_boy_one_girl_l2479_247989


namespace NUMINAMATH_CALUDE_sum_of_abs_roots_l2479_247986

/-- Given a polynomial x^3 - 2023x + n with integer roots p, q, and r, 
    prove that the sum of their absolute values is 84 -/
theorem sum_of_abs_roots (n : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 2023*x + n = 0 ↔ x = p ∨ x = q ∨ x = r) → 
  |p| + |q| + |r| = 84 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_abs_roots_l2479_247986


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l2479_247905

/-- Given a triangle ABC with sides a, b, and c, prove that if a = b + 1, b = c + 1, 
    and the perimeter is 21, then a = 8, b = 7, and c = 6. -/
theorem triangle_side_lengths 
  (a b c : ℝ) 
  (h1 : a = b + 1) 
  (h2 : b = c + 1) 
  (h3 : a + b + c = 21) : 
  a = 8 ∧ b = 7 ∧ c = 6 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_lengths_l2479_247905


namespace NUMINAMATH_CALUDE_solve_equation_l2479_247929

theorem solve_equation (x : ℝ) (h : Real.sqrt ((2 / x) + 3) = 2) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2479_247929


namespace NUMINAMATH_CALUDE_ramesh_profit_share_l2479_247953

/-- Calculates the share of profit for a partner in a business partnership -/
def calculateProfitShare (investment1 : ℕ) (investment2 : ℕ) (totalProfit : ℕ) : ℕ :=
  (investment2 * totalProfit) / (investment1 + investment2)

/-- Theorem stating that Ramesh's share of the profit is 11,875 -/
theorem ramesh_profit_share :
  calculateProfitShare 24000 40000 19000 = 11875 := by
  sorry

end NUMINAMATH_CALUDE_ramesh_profit_share_l2479_247953


namespace NUMINAMATH_CALUDE_exists_class_with_at_least_35_students_l2479_247999

/-- Proves that in a school with 33 classes and 1150 students, there exists at least one class with 35 or more students. -/
theorem exists_class_with_at_least_35_students 
  (num_classes : ℕ) 
  (total_students : ℕ) 
  (h1 : num_classes = 33) 
  (h2 : total_students = 1150) : 
  ∃ (class_size : ℕ), class_size ≥ 35 ∧ class_size ≤ total_students := by
  sorry

#check exists_class_with_at_least_35_students

end NUMINAMATH_CALUDE_exists_class_with_at_least_35_students_l2479_247999


namespace NUMINAMATH_CALUDE_function_condition_implies_b_bound_l2479_247903

theorem function_condition_implies_b_bound (b : ℝ) :
  (∃ x ∈ Set.Icc (1/2 : ℝ) 2, 
    Real.exp x * (x - b) + x * (Real.exp x * (x - b + 2)) > 0) →
  b < 8/3 := by
  sorry

end NUMINAMATH_CALUDE_function_condition_implies_b_bound_l2479_247903


namespace NUMINAMATH_CALUDE_third_term_of_x_plus_two_pow_five_l2479_247955

/-- The coefficient of the r-th term in the expansion of (a + b)^n -/
def binomial_coefficient (n : ℕ) (r : ℕ) : ℕ :=
  Nat.choose n r

/-- The r-th term in the expansion of (a + b)^n -/
def binomial_term (n : ℕ) (r : ℕ) (a b : ℚ) : ℚ :=
  (binomial_coefficient n r : ℚ) * a^(n - r) * b^r

/-- The third term of (x + 2)^5 is 40x^3 -/
theorem third_term_of_x_plus_two_pow_five (x : ℚ) :
  binomial_term 5 2 x 2 = 40 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_third_term_of_x_plus_two_pow_five_l2479_247955


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2479_247900

theorem complex_equation_solution (a b : ℝ) (z : ℂ) :
  z = a + 4*Complex.I ∧ z / (z + b) = 4*Complex.I → b = 17 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2479_247900


namespace NUMINAMATH_CALUDE_no_rational_solution_l2479_247908

theorem no_rational_solution : ¬∃ (x y z : ℚ), 
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x^5 + 2*y^5 + 5*z^5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_l2479_247908


namespace NUMINAMATH_CALUDE_stock_price_increase_percentage_l2479_247907

theorem stock_price_increase_percentage (total_stocks : ℕ) (higher_stocks : ℕ) : 
  total_stocks = 1980 →
  higher_stocks = 1080 →
  higher_stocks > (total_stocks - higher_stocks) →
  (((higher_stocks : ℝ) - (total_stocks - higher_stocks)) / (total_stocks - higher_stocks : ℝ)) * 100 = 20 :=
by sorry

end NUMINAMATH_CALUDE_stock_price_increase_percentage_l2479_247907


namespace NUMINAMATH_CALUDE_cube_expansion_coefficient_sum_l2479_247964

theorem cube_expansion_coefficient_sum (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (Real.sqrt 3 * x - 1)^3 = a₀ + a₁*x + a₂*x^2 + a₃*x^3) →
  (a₀ + a₂)^2 - (a₁ + a₃)^2 = -8 := by
  sorry

end NUMINAMATH_CALUDE_cube_expansion_coefficient_sum_l2479_247964


namespace NUMINAMATH_CALUDE_family_trip_arrangements_l2479_247958

theorem family_trip_arrangements (n : Nat) (k : Nat) : 
  n = 4 ∧ k = 3 → k^n = 81 := by
  sorry

end NUMINAMATH_CALUDE_family_trip_arrangements_l2479_247958


namespace NUMINAMATH_CALUDE_partial_fraction_sum_zero_l2479_247963

theorem partial_fraction_sum_zero (x : ℝ) (A B C D E : ℝ) :
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) =
  A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5) →
  A + B + C + D + E = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_zero_l2479_247963


namespace NUMINAMATH_CALUDE_limit_f_zero_l2479_247968

-- Define the function f
noncomputable def f (x y : ℝ) : ℝ :=
  if x^2 + y^2 ≠ 0 then x * Real.sin (1 / y) + y * Real.sin (1 / x)
  else 0

-- State the theorem
theorem limit_f_zero :
  ∀ ε > 0, ∃ δ > 0, ∀ x y : ℝ, Real.sqrt (x^2 + y^2) < δ → |f x y| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_f_zero_l2479_247968


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2479_247942

/-- Given a triangle ABC with side lengths a, b, and c, if a^2 + b^2 - c^2 = ab, 
    then the measure of angle C is 60°. -/
theorem triangle_angle_measure (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
    (h_eq : a^2 + b^2 - c^2 = a * b) : 
    Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2479_247942


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2479_247921

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 and right focus at (c, 0),
    if a circle with radius 4 centered at the right focus passes through
    the origin and the point (a, b) on the asymptote, then a² = 4 and b² = 12 -/
theorem hyperbola_equation (a b c : ℝ) (h1 : c > 0) (h2 : a > 0) (h3 : b > 0) :
  (c = 4) →
  ((a - c)^2 + b^2 = 16) →
  (a^2 + b^2 = c^2) →
  (a^2 = 4 ∧ b^2 = 12) := by
  sorry

#check hyperbola_equation

end NUMINAMATH_CALUDE_hyperbola_equation_l2479_247921


namespace NUMINAMATH_CALUDE_value_of_x_l2479_247928

theorem value_of_x (x y z : ℚ) : 
  x = (1 / 3) * y → 
  y = (1 / 4) * z → 
  z = 96 → 
  x = 8 := by
sorry

end NUMINAMATH_CALUDE_value_of_x_l2479_247928


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisible_by_prime_l2479_247975

theorem binomial_coefficient_divisible_by_prime (p k : ℕ) :
  Nat.Prime p → 1 ≤ k → k ≤ p - 1 → p ∣ Nat.choose p k := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisible_by_prime_l2479_247975


namespace NUMINAMATH_CALUDE_equation_solution_l2479_247931

theorem equation_solution :
  ∃ x : ℚ, (5 * x - 3) / (6 * x - 6) = 4 / 3 ∧ x = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2479_247931


namespace NUMINAMATH_CALUDE_negation_equivalence_l2479_247913

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2479_247913


namespace NUMINAMATH_CALUDE_jack_payback_l2479_247920

/-- The amount borrowed by Jack -/
def principal : ℚ := 1200

/-- The interest rate as a decimal -/
def interestRate : ℚ := 1/10

/-- The amount Jack will pay back -/
def amountToPay : ℚ := principal * (1 + interestRate)

/-- Theorem stating that the amount Jack will pay back is 1320 -/
theorem jack_payback : amountToPay = 1320 := by sorry

end NUMINAMATH_CALUDE_jack_payback_l2479_247920


namespace NUMINAMATH_CALUDE_angle_BOK_formula_l2479_247983

/-- Represents a trihedral angle with vertex O and edges OA, OB, and OC -/
structure TrihedralAngle where
  α : ℝ  -- Angle BOC
  β : ℝ  -- Angle COA
  γ : ℝ  -- Angle AOB

/-- Represents a sphere inscribed in a trihedral angle -/
structure InscribedSphere (t : TrihedralAngle) where
  K : Point₃  -- Point where the sphere touches face BOC

/-- The angle BOK in a trihedral angle with an inscribed sphere -/
noncomputable def angleBOK (t : TrihedralAngle) (s : InscribedSphere t) : ℝ :=
  sorry

/-- Theorem stating that the angle BOK is equal to (α + γ - β) / 2 -/
theorem angle_BOK_formula (t : TrihedralAngle) (s : InscribedSphere t) :
  angleBOK t s = (t.α + t.γ - t.β) / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_BOK_formula_l2479_247983


namespace NUMINAMATH_CALUDE_shoe_cost_difference_l2479_247902

/-- Proves that the percentage difference between the average cost per year of new shoes
    and the cost of repairing used shoes is 10.34%, given the specified conditions. -/
theorem shoe_cost_difference (used_repair_cost : ℝ) (used_repair_duration : ℝ)
    (new_shoe_cost : ℝ) (new_shoe_duration : ℝ)
    (h1 : used_repair_cost = 14.50)
    (h2 : used_repair_duration = 1)
    (h3 : new_shoe_cost = 32.00)
    (h4 : new_shoe_duration = 2) :
    let used_cost_per_year := used_repair_cost / used_repair_duration
    let new_cost_per_year := new_shoe_cost / new_shoe_duration
    let percentage_difference := (new_cost_per_year - used_cost_per_year) / used_cost_per_year * 100
    percentage_difference = 10.34 := by
  sorry

end NUMINAMATH_CALUDE_shoe_cost_difference_l2479_247902


namespace NUMINAMATH_CALUDE_correct_paint_time_equation_l2479_247961

/-- Represents the time needed for three people to paint a room together, given their individual rates and a break time. -/
def paint_time (rate1 rate2 rate3 break_time : ℝ) (t : ℝ) : Prop :=
  (1 / rate1 + 1 / rate2 + 1 / rate3) * (t - break_time) = 1

/-- Theorem stating that the equation correctly represents the painting time for Doug, Dave, and Ralph. -/
theorem correct_paint_time_equation :
  ∀ t : ℝ, paint_time 6 8 12 1.5 t ↔ (1/6 + 1/8 + 1/12) * (t - 1.5) = 1 :=
by sorry

end NUMINAMATH_CALUDE_correct_paint_time_equation_l2479_247961


namespace NUMINAMATH_CALUDE_f_of_four_equals_thirteen_l2479_247940

/-- Given a function f where f(2x) = 3x^2 + 1 for all x, prove that f(4) = 13 -/
theorem f_of_four_equals_thirteen (f : ℝ → ℝ) (h : ∀ x, f (2 * x) = 3 * x^2 + 1) : 
  f 4 = 13 := by
  sorry

end NUMINAMATH_CALUDE_f_of_four_equals_thirteen_l2479_247940


namespace NUMINAMATH_CALUDE_find_number_l2479_247978

theorem find_number (x : ℝ) : 5 + 2 * (x - 3) = 15 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2479_247978


namespace NUMINAMATH_CALUDE_circle_radius_proof_l2479_247910

/-- Representation of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Representation of a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Given geometric configuration -/
theorem circle_radius_proof (C1 C2 : Circle) (X Y Z : Point) :
  -- C1's center O is on C2
  C2.center.x^2 + C2.center.y^2 = C2.radius^2 →
  -- X and Y are intersection points of C1 and C2
  (X.x - C1.center.x)^2 + (X.y - C1.center.y)^2 = C1.radius^2 →
  (X.x - C2.center.x)^2 + (X.y - C2.center.y)^2 = C2.radius^2 →
  (Y.x - C1.center.x)^2 + (Y.y - C1.center.y)^2 = C1.radius^2 →
  (Y.x - C2.center.x)^2 + (Y.y - C2.center.y)^2 = C2.radius^2 →
  -- Z is on C2 but outside C1
  (Z.x - C2.center.x)^2 + (Z.y - C2.center.y)^2 = C2.radius^2 →
  (Z.x - C1.center.x)^2 + (Z.y - C1.center.y)^2 > C1.radius^2 →
  -- Given distances
  (X.x - Z.x)^2 + (X.y - Z.y)^2 = 15^2 →
  (C1.center.x - Z.x)^2 + (C1.center.y - Z.y)^2 = 13^2 →
  (Y.x - Z.x)^2 + (Y.y - Z.y)^2 = 8^2 →
  -- Conclusion
  C1.radius = Real.sqrt 394 := by
    sorry


end NUMINAMATH_CALUDE_circle_radius_proof_l2479_247910


namespace NUMINAMATH_CALUDE_trapezoid_area_l2479_247946

/-- A trapezoid with given dimensions -/
structure Trapezoid where
  AD : ℝ  -- Length of longer base
  BC : ℝ  -- Length of shorter base
  AC : ℝ  -- Length of one diagonal
  BD : ℝ  -- Length of other diagonal

/-- The area of a trapezoid with the given dimensions is 80 -/
theorem trapezoid_area (T : Trapezoid)
    (h1 : T.AD = 24)
    (h2 : T.BC = 8)
    (h3 : T.AC = 13)
    (h4 : T.BD = 5 * Real.sqrt 17) :
    (T.AD + T.BC) * Real.sqrt (T.AC ^ 2 - ((T.AD - T.BC) / 2 + T.BC) ^ 2) / 2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l2479_247946


namespace NUMINAMATH_CALUDE_unique_m_exists_l2479_247959

/-- Given a positive integer m, converts the hexadecimal number Im05 to decimal --/
def hex_to_decimal (m : ℕ+) : ℕ :=
  16 * 16 * 16 * m.val + 16 * 16 * 13 + 16 * 0 + 5

/-- Theorem stating that there exists a unique positive integer m 
    such that Im05 in hexadecimal equals 293 in decimal --/
theorem unique_m_exists : ∃! (m : ℕ+), hex_to_decimal m = 293 := by
  sorry

end NUMINAMATH_CALUDE_unique_m_exists_l2479_247959


namespace NUMINAMATH_CALUDE_octagon_diagonals_l2479_247909

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular octagon has 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: The number of diagonals in a regular octagon is 20 -/
theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l2479_247909


namespace NUMINAMATH_CALUDE_rabbit_nuts_count_l2479_247994

theorem rabbit_nuts_count :
  ∀ (rabbit_holes fox_holes : ℕ),
    rabbit_holes = fox_holes + 5 →
    4 * rabbit_holes = 6 * fox_holes →
    4 * rabbit_holes = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_rabbit_nuts_count_l2479_247994


namespace NUMINAMATH_CALUDE_champion_wins_39_l2479_247998

/-- Represents a basketball championship. -/
structure BasketballChampionship where
  n : ℕ                -- Number of teams
  totalPoints : ℕ      -- Total points of non-champion teams
  champPoints : ℕ      -- Points of the champion

/-- The number of matches won by the champion. -/
def championWins (championship : BasketballChampionship) : ℕ :=
  championship.champPoints - (championship.n - 1) * 2

/-- Theorem stating the number of matches won by the champion. -/
theorem champion_wins_39 (championship : BasketballChampionship) :
  championship.n = 27 ∧
  championship.totalPoints = 2015 ∧
  championship.champPoints = 3 * championship.n^2 - 3 * championship.n - championship.totalPoints →
  championWins championship = 39 := by
  sorry

#eval championWins { n := 27, totalPoints := 2015, champPoints := 91 }

end NUMINAMATH_CALUDE_champion_wins_39_l2479_247998


namespace NUMINAMATH_CALUDE_quadratic_function_largest_m_l2479_247916

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

def symmetric_about_neg_one (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x - 4) = f (2 - x)

def greater_than_or_equal_x (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x ≥ x

def less_than_or_equal_square (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Ioo 0 2 → f x ≤ ((x + 1) / 2)^2

def min_value_zero (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y ≥ f x

theorem quadratic_function_largest_m (a b c : ℝ) (h_a : a ≠ 0) :
  let f := quadratic_function a b c
  symmetric_about_neg_one f ∧
  greater_than_or_equal_x f ∧
  less_than_or_equal_square f ∧
  min_value_zero f →
  (∃ m : ℝ, m > 1 ∧
    (∃ t : ℝ, ∀ x : ℝ, x ∈ Set.Icc 1 m → f (x + t) ≤ x) ∧
    (∀ n : ℝ, n > m →
      ¬(∃ t : ℝ, ∀ x : ℝ, x ∈ Set.Icc 1 n → f (x + t) ≤ x))) ∧
  (∀ m : ℝ, m > 1 ∧
    (∃ t : ℝ, ∀ x : ℝ, x ∈ Set.Icc 1 m → f (x + t) ≤ x) ∧
    (∀ n : ℝ, n > m →
      ¬(∃ t : ℝ, ∀ x : ℝ, x ∈ Set.Icc 1 n → f (x + t) ≤ x)) →
    m = 9) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_largest_m_l2479_247916


namespace NUMINAMATH_CALUDE_chicken_rabbit_problem_has_unique_solution_l2479_247939

/-- Represents the number of chickens and rabbits in a cage. -/
structure AnimalCount where
  chickens : ℕ
  rabbits : ℕ

/-- Checks if the given animal count satisfies the problem conditions. -/
def satisfiesConditions (count : AnimalCount) : Prop :=
  count.chickens = 2 * (4 * count.rabbits) - 5 ∧
  2 * count.chickens + count.rabbits = 92

/-- There exists a unique solution to the chicken and rabbit problem. -/
theorem chicken_rabbit_problem_has_unique_solution :
  ∃! count : AnimalCount, satisfiesConditions count :=
sorry

end NUMINAMATH_CALUDE_chicken_rabbit_problem_has_unique_solution_l2479_247939


namespace NUMINAMATH_CALUDE_cattle_milk_production_l2479_247980

/-- Given a herd of dairy cows, calculates the daily milk production per cow -/
def daily_milk_per_cow (num_cows : ℕ) (weekly_milk : ℕ) : ℚ :=
  (weekly_milk : ℚ) / 7 / num_cows

theorem cattle_milk_production :
  daily_milk_per_cow 52 364000 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_cattle_milk_production_l2479_247980


namespace NUMINAMATH_CALUDE_complex_division_result_l2479_247934

theorem complex_division_result : 
  let i : ℂ := Complex.I
  (1 + i) / (-2 * i) = -1/2 + 1/2 * i := by sorry

end NUMINAMATH_CALUDE_complex_division_result_l2479_247934


namespace NUMINAMATH_CALUDE_teresa_age_at_birth_l2479_247982

/-- Calculates Teresa's age when Michiko was born given current ages and Morio's age at Michiko's birth -/
def teresaAgeAtBirth (teresaCurrentAge marioCurrentAge marioAgeAtBirth : ℕ) : ℕ :=
  marioAgeAtBirth - (marioCurrentAge - teresaCurrentAge)

theorem teresa_age_at_birth :
  teresaAgeAtBirth 59 71 38 = 26 := by
  sorry

end NUMINAMATH_CALUDE_teresa_age_at_birth_l2479_247982


namespace NUMINAMATH_CALUDE_parabola_values_l2479_247979

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_coord (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_values (p : Parabola) : 
  (p.y_coord 4 = 5) ∧ 
  (p.y_coord 2 = -3) ∧ 
  (p.y_coord 6 = 3) ∧
  (∀ x : ℝ, p.y_coord x = p.y_coord (8 - x)) →
  p.a = -2 ∧ p.b = 16 ∧ p.c = -27 := by
  sorry

end NUMINAMATH_CALUDE_parabola_values_l2479_247979


namespace NUMINAMATH_CALUDE_male_avg_is_58_l2479_247906

/-- Represents an association with male and female members selling raffle tickets -/
structure Association where
  total_avg : ℝ
  female_avg : ℝ
  male_female_ratio : ℝ

/-- The average number of tickets sold by male members -/
def male_avg (a : Association) : ℝ :=
  (3 * a.total_avg - 2 * a.female_avg)

/-- Theorem stating the average number of tickets sold by male members -/
theorem male_avg_is_58 (a : Association) 
  (h1 : a.total_avg = 66)
  (h2 : a.female_avg = 70)
  (h3 : a.male_female_ratio = 1/2) :
  male_avg a = 58 := by
  sorry


end NUMINAMATH_CALUDE_male_avg_is_58_l2479_247906


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_parallel_lines_l2479_247987

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (planeParallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_parallel_lines 
  (m n : Line) (α β : Plane) :
  m ≠ n →
  α ≠ β →
  parallel m n →
  perpendicular m α →
  perpendicular n β →
  planeParallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_parallel_lines_l2479_247987


namespace NUMINAMATH_CALUDE_xy_value_l2479_247915

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 8) : x * y = 8 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2479_247915


namespace NUMINAMATH_CALUDE_polyhedron_sum_l2479_247997

def convex_polyhedron (V : ℕ) (t h : ℕ) : Prop :=
  t + h = 40 ∧ 
  3 * t + 6 * h = 5 * V ∧
  3 * t + 6 * h = 2 * (3 * t + 6 * h) / 2 - 38

theorem polyhedron_sum : ∃ V : ℕ, convex_polyhedron V 20 20 ∧ 100 * 2 + 10 * 3 + V = 266 :=
sorry

end NUMINAMATH_CALUDE_polyhedron_sum_l2479_247997
