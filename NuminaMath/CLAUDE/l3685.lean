import Mathlib

namespace NUMINAMATH_CALUDE_bhaskara_solution_l3685_368574

/-- The number of people in Bhaskara's money distribution problem -/
def bhaskara_problem (n : ℕ) : Prop :=
  let initial_sum := n * (2 * 3 + (n - 1) * 1) / 2
  let redistribution_sum := 100 * n
  initial_sum = redistribution_sum

theorem bhaskara_solution :
  ∃ n : ℕ, n > 0 ∧ bhaskara_problem n ∧ n = 195 := by
  sorry

end NUMINAMATH_CALUDE_bhaskara_solution_l3685_368574


namespace NUMINAMATH_CALUDE_cubic_function_range_l3685_368518

/-- A cubic function f(x) = ax³ + bx² + cx + d satisfying given conditions -/
structure CubicFunction where
  f : ℝ → ℝ
  cubic : ∃ (a b c d : ℝ), ∀ x, f x = a * x^3 + b * x^2 + c * x + d
  cond1 : 1 ≤ f (-1) ∧ f (-1) ≤ 2
  cond2 : 1 ≤ f 1 ∧ f 1 ≤ 3
  cond3 : 2 ≤ f 2 ∧ f 2 ≤ 4
  cond4 : -1 ≤ f 3 ∧ f 3 ≤ 1

/-- The value of f(4) is always within the range [-21¾, 1] for any CubicFunction -/
theorem cubic_function_range (cf : CubicFunction) :
  -21.75 ≤ cf.f 4 ∧ cf.f 4 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_range_l3685_368518


namespace NUMINAMATH_CALUDE_cricket_bat_cost_price_l3685_368516

theorem cricket_bat_cost_price 
  (profit_a_to_b : Real) 
  (profit_b_to_c : Real) 
  (price_c_pays : Real) : 
  profit_a_to_b = 0.20 →
  profit_b_to_c = 0.25 →
  price_c_pays = 234 →
  ∃ (cost_price_a : Real), 
    cost_price_a = 156 ∧ 
    price_c_pays = (1 + profit_b_to_c) * ((1 + profit_a_to_b) * cost_price_a) :=
by sorry

end NUMINAMATH_CALUDE_cricket_bat_cost_price_l3685_368516


namespace NUMINAMATH_CALUDE_even_periodic_function_property_l3685_368595

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem even_periodic_function_property
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_period : has_period f 2)
  (h_interval : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-2) 0, f x = 3 - |x + 1| := by
  sorry

end NUMINAMATH_CALUDE_even_periodic_function_property_l3685_368595


namespace NUMINAMATH_CALUDE_right_triangle_ratio_bound_l3685_368504

/-- A non-degenerate right triangle with sides a, b, and c (c being the hypotenuse) -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  right_angle : a^2 + b^2 = c^2
  c_largest : c ≥ a ∧ c ≥ b

/-- The theorem stating that the least upper bound of (a^2 + b^2 + c^2) / a^2 for all non-degenerate right triangles is 4 -/
theorem right_triangle_ratio_bound :
  ∃ N : ℝ, (∀ t : RightTriangle, (t.a^2 + t.b^2 + t.c^2) / t.a^2 ≤ N) ∧
  (∀ ε > 0, ∃ t : RightTriangle, N - ε < (t.a^2 + t.b^2 + t.c^2) / t.a^2) ∧
  N = 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_bound_l3685_368504


namespace NUMINAMATH_CALUDE_soaking_solution_l3685_368561

/-- Represents the time needed to soak clothes for each type of stain -/
structure SoakingTime where
  grass : ℕ
  marinara : ℕ

/-- Conditions for the soaking problem -/
def soaking_problem (t : SoakingTime) : Prop :=
  t.marinara = t.grass + 7 ∧ 
  3 * t.grass + t.marinara = 19

/-- Theorem stating the solution to the soaking problem -/
theorem soaking_solution :
  ∃ (t : SoakingTime), soaking_problem t ∧ t.grass = 3 := by
  sorry

end NUMINAMATH_CALUDE_soaking_solution_l3685_368561


namespace NUMINAMATH_CALUDE_triangle_area_l3685_368598

/-- Given a right triangle with side lengths in the ratio 2:3:4 inscribed in a circle of radius 4,
    prove that its area is 12. -/
theorem triangle_area (a b c : ℝ) (r : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Ensure positive side lengths
  r = 4 →  -- Circle radius
  a^2 + b^2 = c^2 →  -- Right triangle condition
  c = 2 * r →  -- Hypotenuse is diameter
  b / a = 3 / 2 →  -- Side ratio condition
  c / a = 2 →  -- Side ratio condition
  (1 / 2) * a * b = 12 :=  -- Area formula
by sorry

end NUMINAMATH_CALUDE_triangle_area_l3685_368598


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l3685_368591

theorem largest_prime_factor_of_expression : 
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ (16^4 + 2 * 16^2 + 1 - 15^4) ∧ 
    ∀ q : ℕ, Nat.Prime q → q ∣ (16^4 + 2 * 16^2 + 1 - 15^4) → q ≤ p) ∧
  (Nat.Prime 241 ∧ 241 ∣ (16^4 + 2 * 16^2 + 1 - 15^4)) := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l3685_368591


namespace NUMINAMATH_CALUDE_subset_condition_l3685_368586

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def B (a : ℝ) : Set ℝ := {x : ℝ | x > a}

theorem subset_condition (a : ℝ) : A ⊆ B a → a < -2 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l3685_368586


namespace NUMINAMATH_CALUDE_roll_five_probability_l3685_368592

/-- A cube with six faces -/
structure Cube where
  faces : Fin 6 → ℕ

/-- The specific cube described in the problem -/
def problemCube : Cube :=
  { faces := λ i => match i with
    | ⟨0, _⟩ => 1
    | ⟨1, _⟩ => 1
    | ⟨2, _⟩ => 2
    | ⟨3, _⟩ => 4
    | ⟨4, _⟩ => 5
    | ⟨5, _⟩ => 5
    | _ => 0 }

/-- The probability of rolling a specific number on the cube -/
def rollProbability (c : Cube) (n : ℕ) : ℚ :=
  (Finset.filter (λ i => c.faces i = n) Finset.univ).card / 6

/-- Theorem stating that the probability of rolling a 5 on the problem cube is 1/3 -/
theorem roll_five_probability :
  rollProbability problemCube 5 = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_roll_five_probability_l3685_368592


namespace NUMINAMATH_CALUDE_ratio_theorem_max_coeff_theorem_l3685_368550

open Real

/-- The ratio of the sum of all coefficients to the sum of all binomial coefficients
    in the expansion of (x^(2/3) + 3x^2)^n is 32 -/
def ratio_condition (n : ℕ) : Prop :=
  (4 : ℝ)^n / (2 : ℝ)^n = 32

/-- The value of n that satisfies the ratio condition -/
def n_value : ℕ := 5

/-- Theorem stating that n_value satisfies the ratio condition -/
theorem ratio_theorem : ratio_condition n_value := by
  sorry

/-- The terms with maximum binomial coefficient in the expansion -/
def max_coeff_terms (x : ℝ) : ℝ × ℝ :=
  (90 * x^6, 270 * x^(22/3))

/-- Theorem stating that max_coeff_terms gives the correct terms -/
theorem max_coeff_theorem (x : ℝ) :
  max_coeff_terms x = (90 * x^6, 270 * x^(22/3)) := by
  sorry

end NUMINAMATH_CALUDE_ratio_theorem_max_coeff_theorem_l3685_368550


namespace NUMINAMATH_CALUDE_quadratic_polynomials_exist_l3685_368548

/-- A quadratic polynomial ax^2 + bx + c -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The number of real roots of a quadratic polynomial -/
def num_real_roots (p : QuadraticPolynomial) : ℕ :=
  sorry

/-- The sum of two quadratic polynomials -/
def add (p q : QuadraticPolynomial) : QuadraticPolynomial :=
  ⟨p.a + q.a, p.b + q.b, p.c + q.c⟩

theorem quadratic_polynomials_exist : ∃ (f g h : QuadraticPolynomial),
  (num_real_roots f = 2) ∧
  (num_real_roots g = 2) ∧
  (num_real_roots h = 2) ∧
  (num_real_roots (add f g) = 1) ∧
  (num_real_roots (add f h) = 1) ∧
  (num_real_roots (add g h) = 1) ∧
  (num_real_roots (add (add f g) h) = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_polynomials_exist_l3685_368548


namespace NUMINAMATH_CALUDE_smallest_integer_y_minus_five_smallest_l3685_368597

theorem smallest_integer_y (y : ℤ) : (7 - 3 * y < 25) ↔ y ≥ -5 := by sorry

theorem minus_five_smallest : ∃ (y : ℤ), (7 - 3 * y < 25) ∧ (∀ (z : ℤ), z < y → (7 - 3 * z ≥ 25)) := by sorry

end NUMINAMATH_CALUDE_smallest_integer_y_minus_five_smallest_l3685_368597


namespace NUMINAMATH_CALUDE_albert_needs_twelve_dollars_l3685_368572

/-- The amount of additional money Albert needs to buy his art supplies -/
def additional_money_needed (paintbrush_cost paint_cost easel_cost current_money : ℚ) : ℚ :=
  paintbrush_cost + paint_cost + easel_cost - current_money

/-- Theorem stating that Albert needs $12 more -/
theorem albert_needs_twelve_dollars :
  additional_money_needed 1.50 4.35 12.65 6.50 = 12 := by
  sorry

end NUMINAMATH_CALUDE_albert_needs_twelve_dollars_l3685_368572


namespace NUMINAMATH_CALUDE_most_cost_effective_plan_l3685_368581

/-- Represents the capacity and rental cost of a truck type -/
structure TruckType where
  capacity : ℕ
  rentalCost : ℕ

/-- Represents a rental plan -/
structure RentalPlan where
  typeA : ℕ
  typeB : ℕ

def totalCapacity (a b : TruckType) (plan : RentalPlan) : ℕ :=
  a.capacity * plan.typeA + b.capacity * plan.typeB

def totalCost (a b : TruckType) (plan : RentalPlan) : ℕ :=
  a.rentalCost * plan.typeA + b.rentalCost * plan.typeB

/-- The main theorem stating the most cost-effective rental plan -/
theorem most_cost_effective_plan 
  (typeA typeB : TruckType)
  (h1 : 2 * typeA.capacity + typeB.capacity = 10)
  (h2 : typeA.capacity + 2 * typeB.capacity = 11)
  (h3 : typeA.rentalCost = 100)
  (h4 : typeB.rentalCost = 120) :
  ∃ (plan : RentalPlan),
    totalCapacity typeA typeB plan = 31 ∧
    (∀ (otherPlan : RentalPlan),
      totalCapacity typeA typeB otherPlan = 31 →
      totalCost typeA typeB plan ≤ totalCost typeA typeB otherPlan) ∧
    plan.typeA = 1 ∧
    plan.typeB = 7 ∧
    totalCost typeA typeB plan = 940 :=
  sorry

end NUMINAMATH_CALUDE_most_cost_effective_plan_l3685_368581


namespace NUMINAMATH_CALUDE_four_digit_square_modification_l3685_368508

/-- A function that returns the first digit of a natural number -/
def firstDigit (n : ℕ) : ℕ :=
  if n < 10 then n else firstDigit (n / 10)

/-- A function that returns the last digit of a natural number -/
def lastDigit (n : ℕ) : ℕ := n % 10

/-- A function that modifies a number by decreasing its first digit by 3
    and increasing its last digit by 3 -/
def modifyNumber (n : ℕ) : ℕ :=
  n - 3000 + 3

theorem four_digit_square_modification :
  ∃ n : ℕ,
    1000 ≤ n ∧ n < 10000 ∧  -- four-digit number
    ∃ a : ℕ, n = a^2 ∧      -- perfect square
    ∃ b : ℕ, modifyNumber n = b^2  -- modified number is also a perfect square
  := by sorry

end NUMINAMATH_CALUDE_four_digit_square_modification_l3685_368508


namespace NUMINAMATH_CALUDE_smallest_q_is_31_l3685_368529

theorem smallest_q_is_31 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : q = 15 * p + 1) :
  q ≥ 31 :=
sorry

end NUMINAMATH_CALUDE_smallest_q_is_31_l3685_368529


namespace NUMINAMATH_CALUDE_lisas_marbles_problem_l3685_368524

/-- The minimum number of additional marbles needed -/
def min_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_marbles

/-- Theorem stating the minimum number of additional marbles needed for Lisa's problem -/
theorem lisas_marbles_problem (num_friends : ℕ) (initial_marbles : ℕ) 
    (h1 : num_friends = 12) (h2 : initial_marbles = 34) : 
    min_additional_marbles num_friends initial_marbles = 44 := by
  sorry

#eval min_additional_marbles 12 34

end NUMINAMATH_CALUDE_lisas_marbles_problem_l3685_368524


namespace NUMINAMATH_CALUDE_find_p_value_l3685_368576

theorem find_p_value (a b c p : ℝ) 
  (h1 : 9 / (a + b) = p / (a + c)) 
  (h2 : p / (a + c) = 13 / (c - b)) : 
  p = 22 := by
sorry

end NUMINAMATH_CALUDE_find_p_value_l3685_368576


namespace NUMINAMATH_CALUDE_average_decrease_l3685_368525

theorem average_decrease (initial_count : ℕ) (initial_avg : ℚ) (new_obs : ℚ) :
  initial_count = 6 →
  initial_avg = 13 →
  new_obs = 6 →
  let total_sum := initial_count * initial_avg
  let new_sum := total_sum + new_obs
  let new_count := initial_count + 1
  let new_avg := new_sum / new_count
  initial_avg - new_avg = 1 := by
sorry

end NUMINAMATH_CALUDE_average_decrease_l3685_368525


namespace NUMINAMATH_CALUDE_updated_mean_after_decrement_l3685_368559

theorem updated_mean_after_decrement (n : ℕ) (original_mean decrement : ℝ) :
  n > 0 →
  n = 50 →
  original_mean = 200 →
  decrement = 6 →
  (n * original_mean - n * decrement) / n = 194 := by
  sorry

end NUMINAMATH_CALUDE_updated_mean_after_decrement_l3685_368559


namespace NUMINAMATH_CALUDE_jane_albert_same_committee_l3685_368500

def total_people : ℕ := 6
def committee_size : ℕ := 3

def probability_same_committee : ℚ := 1 / 5

theorem jane_albert_same_committee :
  let total_combinations := Nat.choose total_people committee_size
  let favorable_combinations := Nat.choose (total_people - 2) (committee_size - 2)
  (favorable_combinations : ℚ) / total_combinations = probability_same_committee :=
sorry

end NUMINAMATH_CALUDE_jane_albert_same_committee_l3685_368500


namespace NUMINAMATH_CALUDE_counterexample_exists_l3685_368503

theorem counterexample_exists : ∃ (a b : ℝ), a^2 > b^2 ∧ a ≤ b := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3685_368503


namespace NUMINAMATH_CALUDE_mary_sheep_theorem_l3685_368568

def initial_sheep : ℕ := 1500

def sister_percentage : ℚ := 1/4
def brother_percentage : ℚ := 3/10
def cousin_fraction : ℚ := 1/7

def remaining_sheep : ℕ := 676

theorem mary_sheep_theorem :
  let sheep_after_sister := initial_sheep - ⌊initial_sheep * sister_percentage⌋
  let sheep_after_brother := sheep_after_sister - ⌊sheep_after_sister * brother_percentage⌋
  let sheep_after_cousin := sheep_after_brother - ⌊sheep_after_brother * cousin_fraction⌋
  sheep_after_cousin = remaining_sheep := by sorry

end NUMINAMATH_CALUDE_mary_sheep_theorem_l3685_368568


namespace NUMINAMATH_CALUDE_tank_capacity_l3685_368587

theorem tank_capacity
  (bucket1_capacity bucket2_capacity : ℕ)
  (bucket1_uses bucket2_uses : ℕ)
  (h1 : bucket1_capacity = 4)
  (h2 : bucket2_capacity = 3)
  (h3 : bucket2_uses = bucket1_uses + 4)
  (h4 : bucket1_capacity * bucket1_uses = bucket2_capacity * bucket2_uses) :
  bucket1_capacity * bucket1_uses = 48 :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_l3685_368587


namespace NUMINAMATH_CALUDE_quadrilateral_inequality_l3685_368509

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the length of a line segment
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the measure of an angle
def angle_measure (p q r : ℝ × ℝ) : ℝ := sorry

theorem quadrilateral_inequality (ABCD : Quadrilateral) :
  length ABCD.A ABCD.D = length ABCD.B ABCD.C →
  angle_measure ABCD.A ABCD.D ABCD.C > angle_measure ABCD.B ABCD.C ABCD.D →
  length ABCD.A ABCD.C > length ABCD.B ABCD.D :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_inequality_l3685_368509


namespace NUMINAMATH_CALUDE_johnny_first_job_hours_l3685_368590

/-- Represents Johnny's work schedule and earnings --/
structure WorkSchedule where
  hourlyRate1 : ℝ
  hourlyRate2 : ℝ
  hourlyRate3 : ℝ
  hours2 : ℝ
  hours3 : ℝ
  daysWorked : ℝ
  totalEarnings : ℝ

/-- Theorem stating that given the conditions, Johnny worked 3 hours on the first job each day --/
theorem johnny_first_job_hours (schedule : WorkSchedule)
  (h1 : schedule.hourlyRate1 = 7)
  (h2 : schedule.hourlyRate2 = 10)
  (h3 : schedule.hourlyRate3 = 12)
  (h4 : schedule.hours2 = 2)
  (h5 : schedule.hours3 = 4)
  (h6 : schedule.daysWorked = 5)
  (h7 : schedule.totalEarnings = 445) :
  ∃ (x : ℝ), x = 3 ∧ 
    schedule.daysWorked * (schedule.hourlyRate1 * x + 
      schedule.hourlyRate2 * schedule.hours2 + 
      schedule.hourlyRate3 * schedule.hours3) = schedule.totalEarnings :=
by
  sorry

end NUMINAMATH_CALUDE_johnny_first_job_hours_l3685_368590


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l3685_368564

/-- Given a geometric sequence {aₙ} where a₁ = -2 and a₅ = -4, prove that a₃ = -2√2 -/
theorem geometric_sequence_third_term (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1) 
  (h_a1 : a 1 = -2) (h_a5 : a 5 = -4) : a 3 = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l3685_368564


namespace NUMINAMATH_CALUDE_max_square_plots_l3685_368540

/-- Represents the dimensions of the park and available fencing --/
structure ParkData where
  width : ℕ
  length : ℕ
  fencing : ℕ

/-- Represents a potential partitioning of the park --/
structure Partitioning where
  sideLength : ℕ
  numPlots : ℕ

/-- Checks if a partitioning is valid for the given park data --/
def isValidPartitioning (park : ParkData) (part : Partitioning) : Prop :=
  part.sideLength > 0 ∧
  park.width % part.sideLength = 0 ∧
  park.length % part.sideLength = 0 ∧
  part.numPlots = (park.width / part.sideLength) * (park.length / part.sideLength) ∧
  (park.width / part.sideLength - 1) * park.length + (park.length / part.sideLength - 1) * park.width ≤ park.fencing

/-- Theorem stating that the maximum number of square plots is 2 --/
theorem max_square_plots (park : ParkData) 
  (h_width : park.width = 30)
  (h_length : park.length = 60)
  (h_fencing : park.fencing = 2400) :
  (∀ p : Partitioning, isValidPartitioning park p → p.numPlots ≤ 2) ∧
  (∃ p : Partitioning, isValidPartitioning park p ∧ p.numPlots = 2) :=
sorry

end NUMINAMATH_CALUDE_max_square_plots_l3685_368540


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_sum_l3685_368536

theorem arithmetic_geometric_sequence_sum (a : ℕ → ℤ) :
  (∀ n, a (n + 1) = a n + 2) →  -- arithmetic sequence with common difference 2
  (a 3)^2 = a 1 * a 4 →         -- a_1, a_3, a_4 form a geometric sequence
  a 2 + a 3 = -10 :=            -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_sum_l3685_368536


namespace NUMINAMATH_CALUDE_hexagon_diagonals_l3685_368588

/-- The number of internal diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A hexagon is a polygon with 6 sides -/
def hexagon_sides : ℕ := 6

/-- Theorem: A hexagon has 9 internal diagonals -/
theorem hexagon_diagonals : num_diagonals hexagon_sides = 9 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_diagonals_l3685_368588


namespace NUMINAMATH_CALUDE_series_sum_is_zero_l3685_368535

open Real
open Topology
open Tendsto

noncomputable def series_sum : ℝ := ∑' n, (3 * n + 4) / ((n + 1) * (n + 2) * (n + 3))

theorem series_sum_is_zero : series_sum = 0 := by sorry

end NUMINAMATH_CALUDE_series_sum_is_zero_l3685_368535


namespace NUMINAMATH_CALUDE_inequality_proof_l3685_368534

theorem inequality_proof (x : Real) (h : 0 < x ∧ x < Real.pi / 2) :
  Real.sin (Real.cos x) < Real.cos x ∧ Real.cos x < Real.cos (Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3685_368534


namespace NUMINAMATH_CALUDE_car_trip_speed_l3685_368579

/-- Proves that given the conditions of the car trip, the speed for the remaining part is 20 mph -/
theorem car_trip_speed (D : ℝ) (h_D_pos : D > 0) : 
  let first_part := 0.8 * D
  let second_part := 0.2 * D
  let first_speed := 80
  let total_avg_speed := 50
  let v := (first_speed * total_avg_speed * second_part) / 
           (first_speed * D - total_avg_speed * first_part)
  v = 20 := by
  sorry

end NUMINAMATH_CALUDE_car_trip_speed_l3685_368579


namespace NUMINAMATH_CALUDE_locus_is_hexagon_l3685_368556

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a triangle in 3D space
structure Triangle3D where
  A : Point3D
  B : Point3D
  C : Point3D

-- Define a function to check if a triangle is acute-angled
def isAcuteTriangle (t : Triangle3D) : Prop :=
  sorry

-- Define a function to check if a point forms acute-angled triangles with all sides of the base triangle
def formsAcuteTriangles (P : Point3D) (base : Triangle3D) : Prop :=
  sorry

-- Define the locus of points
def locusOfPoints (base : Triangle3D) : Set Point3D :=
  {P | formsAcuteTriangles P base}

-- Theorem statement
theorem locus_is_hexagon (base : Triangle3D) 
  (h : isAcuteTriangle base) : 
  ∃ (hexagon : Set Point3D), locusOfPoints base = hexagon :=
sorry

end NUMINAMATH_CALUDE_locus_is_hexagon_l3685_368556


namespace NUMINAMATH_CALUDE_buffalo_count_l3685_368507

/-- Represents the number of buffaloes in the group -/
def num_buffaloes : ℕ → ℕ → ℕ := sorry

/-- Represents the number of ducks in the group -/
def num_ducks : ℕ → ℕ → ℕ := sorry

/-- The total number of legs in the group -/
def total_legs (b d : ℕ) : ℕ := 4 * b + 2 * d

/-- The total number of heads in the group -/
def total_heads (b d : ℕ) : ℕ := b + d

theorem buffalo_count (b d : ℕ) : 
  total_legs b d = 2 * total_heads b d + 24 → num_buffaloes b d = 12 := by
  sorry

end NUMINAMATH_CALUDE_buffalo_count_l3685_368507


namespace NUMINAMATH_CALUDE_spaceDivisions_correct_l3685_368573

/-- The number of parts that n planes can divide space into, given that
    each group of three planes intersects at one point and no group of
    four planes has a common point. -/
def spaceDivisions (n : ℕ) : ℚ :=
  (n^3 + 5*n + 6) / 6

/-- Theorem stating that spaceDivisions correctly calculates the number
    of parts that n planes can divide space into. -/
theorem spaceDivisions_correct (n : ℕ) :
  spaceDivisions n = (n^3 + 5*n + 6) / 6 :=
by sorry

end NUMINAMATH_CALUDE_spaceDivisions_correct_l3685_368573


namespace NUMINAMATH_CALUDE_more_cars_difference_l3685_368578

/-- The number of cars Tommy has -/
def tommy_cars : ℕ := 3

/-- The number of cars Jessie has -/
def jessie_cars : ℕ := 3

/-- The total number of cars all three have -/
def total_cars : ℕ := 17

/-- The number of cars Jessie's older brother has -/
def brother_cars : ℕ := total_cars - (tommy_cars + jessie_cars)

theorem more_cars_difference : brother_cars - (tommy_cars + jessie_cars) = 5 := by
  sorry

end NUMINAMATH_CALUDE_more_cars_difference_l3685_368578


namespace NUMINAMATH_CALUDE_point_quadrant_l3685_368599

/-- Given that point P(-4a, 2+b) is in the third quadrant, prove that point Q(a, b) is in the fourth quadrant -/
theorem point_quadrant (a b : ℝ) :
  (-4 * a < 0 ∧ 2 + b < 0) → (a > 0 ∧ b < 0) :=
by sorry

end NUMINAMATH_CALUDE_point_quadrant_l3685_368599


namespace NUMINAMATH_CALUDE_units_digit_of_quotient_l3685_368558

theorem units_digit_of_quotient (n : ℕ) (h : 5 ∣ (2^1993 + 3^1993)) :
  (2^1993 + 3^1993) / 5 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_quotient_l3685_368558


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l3685_368512

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 2)^2 = 25

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  4*x + 3*y - 35 = 0

-- Theorem statement
theorem circle_tangent_to_line :
  ∃ (x y : ℝ), circle_equation x y ∧ line_equation x y ∧
  ∀ (x' y' : ℝ), circle_equation x' y' ∧ line_equation x' y' → (x', y') = (x, y) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l3685_368512


namespace NUMINAMATH_CALUDE_twins_age_product_difference_l3685_368521

theorem twins_age_product_difference : 
  ∀ (current_age : ℕ), 
    current_age = 6 → 
    (current_age + 1) * (current_age + 1) - current_age * current_age = 13 := by
  sorry

end NUMINAMATH_CALUDE_twins_age_product_difference_l3685_368521


namespace NUMINAMATH_CALUDE_x1_value_l3685_368539

theorem x1_value (x1 x2 x3 x4 : Real) 
  (h_order : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1) 
  (h_eq : (1-x1)^2 + (x1-x2)^2 + (x2-x3)^2 + (x3-x4)^2 + x4^2 = 1/5) :
  x1 = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_x1_value_l3685_368539


namespace NUMINAMATH_CALUDE_graph_not_in_first_quadrant_l3685_368545

-- Define the function
def f (k x : ℝ) : ℝ := k * (x - k)

-- Theorem statement
theorem graph_not_in_first_quadrant (k : ℝ) (h : k < 0) :
  ∀ x y : ℝ, f k x = y → ¬(x > 0 ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_graph_not_in_first_quadrant_l3685_368545


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3685_368565

-- Define the sets A and B
def A : Set ℕ := {2, 1, 3}
def B : Set ℕ := {2, 3, 5}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3685_368565


namespace NUMINAMATH_CALUDE_set_equality_implies_values_l3685_368542

theorem set_equality_implies_values (x y : ℝ) : 
  ({1, x, y} : Set ℝ) = {x, x^2, x*y} → x = -1 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_values_l3685_368542


namespace NUMINAMATH_CALUDE_license_plate_difference_l3685_368526

/-- The number of possible digits in a license plate -/
def num_digits : ℕ := 10

/-- The number of possible letters in a license plate -/
def num_letters : ℕ := 26

/-- The number of possible license plates for Alpha (LLDDDDLL format) -/
def alpha_plates : ℕ := num_letters^4 * num_digits^4

/-- The number of possible license plates for Beta (LLLDDDD format) -/
def beta_plates : ℕ := num_letters^3 * num_digits^4

/-- The theorem stating the difference in number of license plates between Alpha and Beta -/
theorem license_plate_difference :
  alpha_plates - beta_plates = num_digits^4 * num_letters^3 * 25 := by
  sorry

#eval alpha_plates - beta_plates
#eval num_digits^4 * num_letters^3 * 25

end NUMINAMATH_CALUDE_license_plate_difference_l3685_368526


namespace NUMINAMATH_CALUDE_master_craftsman_production_l3685_368505

/-- The number of parts manufactured by the master craftsman during the shift -/
def total_parts : ℕ := 210

/-- The number of parts manufactured in the first hour -/
def first_hour_parts : ℕ := 35

/-- The increase in production rate (parts per hour) -/
def rate_increase : ℕ := 15

/-- The time saved by increasing the production rate (in hours) -/
def time_saved : ℚ := 3/2

theorem master_craftsman_production :
  ∃ (remaining_parts : ℕ),
    remaining_parts / first_hour_parts - remaining_parts / (first_hour_parts + rate_increase) = time_saved ∧
    total_parts = first_hour_parts + remaining_parts :=
by sorry

end NUMINAMATH_CALUDE_master_craftsman_production_l3685_368505


namespace NUMINAMATH_CALUDE_lex_apple_count_l3685_368510

/-- The total number of apples Lex picked -/
def total_apples : ℕ := 85

/-- The number of apples with worms -/
def wormy_apples : ℕ := total_apples / 5

/-- The number of bruised apples -/
def bruised_apples : ℕ := total_apples / 5 + 9

/-- The number of apples left to eat raw -/
def raw_apples : ℕ := 42

theorem lex_apple_count :
  wormy_apples + bruised_apples + raw_apples = total_apples :=
by sorry

end NUMINAMATH_CALUDE_lex_apple_count_l3685_368510


namespace NUMINAMATH_CALUDE_find_B_l3685_368523

theorem find_B (A C B : ℤ) (h1 : A = 520) (h2 : C = A + 204) (h3 : C = B + 179) : B = 545 := by
  sorry

end NUMINAMATH_CALUDE_find_B_l3685_368523


namespace NUMINAMATH_CALUDE_simplify_expression_l3685_368569

theorem simplify_expression (x : ℝ) : 5*x + 6 - x + 12 = 4*x + 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3685_368569


namespace NUMINAMATH_CALUDE_minimize_function_l3685_368596

theorem minimize_function (x y z : ℝ) : 
  (3 * x + 2 * y + z = 3) →
  (x^2 + y^2 + 2 * z^2 ≥ 2/3) →
  (x^2 + y^2 + 2 * z^2 = 2/3 → x * y / z = 8/3) :=
by sorry

end NUMINAMATH_CALUDE_minimize_function_l3685_368596


namespace NUMINAMATH_CALUDE_weight_change_problem_l3685_368515

/-- Represents the scenario of replacing a man in a group and the resulting weight change -/
structure WeightChangeScenario where
  initial_count : ℕ
  initial_average : ℝ
  replaced_weight : ℝ
  new_weight : ℝ
  average_increase : ℝ

/-- The theorem representing the weight change problem -/
theorem weight_change_problem (scenario : WeightChangeScenario) 
  (h1 : scenario.initial_count = 10)
  (h2 : scenario.replaced_weight = 58)
  (h3 : scenario.average_increase = 2.5) :
  scenario.new_weight = 83 ∧ 
  ∀ (x : ℝ), ∃ (scenario' : WeightChangeScenario), 
    scenario'.initial_average = x ∧
    scenario'.initial_count = scenario.initial_count ∧
    scenario'.replaced_weight = scenario.replaced_weight ∧
    scenario'.new_weight = scenario.new_weight ∧
    scenario'.average_increase = scenario.average_increase :=
by sorry

end NUMINAMATH_CALUDE_weight_change_problem_l3685_368515


namespace NUMINAMATH_CALUDE_decimal_point_problem_l3685_368527

theorem decimal_point_problem (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 5 * (1 / x)) : 
  x = Real.sqrt 2 / 20 := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l3685_368527


namespace NUMINAMATH_CALUDE_algebraic_expressions_proof_l3685_368522

theorem algebraic_expressions_proof (a b : ℝ) 
  (ha : a = Real.sqrt 5 + 1) 
  (hb : b = Real.sqrt 5 - 1) : 
  (a^2 * b + a * b^2 = 8 * Real.sqrt 5) ∧ 
  (a^2 - a * b + b^2 = 8) := by
sorry

end NUMINAMATH_CALUDE_algebraic_expressions_proof_l3685_368522


namespace NUMINAMATH_CALUDE_three_non_congruent_triangles_l3685_368513

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  perimeter_eq : a + b + c = 11
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

/-- Two triangles are congruent if they have the same side lengths (up to permutation) -/
def congruent (t1 t2 : IntTriangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

/-- The set of all valid integer triangles with perimeter 11 -/
def valid_triangles : Set IntTriangle :=
  {t : IntTriangle | True}

/-- The theorem to be proved -/
theorem three_non_congruent_triangles :
  ∃ (t1 t2 t3 : IntTriangle),
    t1 ∈ valid_triangles ∧ t2 ∈ valid_triangles ∧ t3 ∈ valid_triangles ∧
    ¬(congruent t1 t2) ∧ ¬(congruent t2 t3) ∧ ¬(congruent t1 t3) ∧
    ∀ (t : IntTriangle), t ∈ valid_triangles →
      congruent t t1 ∨ congruent t t2 ∨ congruent t t3 :=
by sorry

end NUMINAMATH_CALUDE_three_non_congruent_triangles_l3685_368513


namespace NUMINAMATH_CALUDE_drama_club_subjects_l3685_368506

theorem drama_club_subjects (total : ℕ) (math : ℕ) (physics : ℕ) (chem : ℕ)
  (math_physics : ℕ) (math_chem : ℕ) (physics_chem : ℕ) (all_three : ℕ)
  (h_total : total = 70)
  (h_math : math = 42)
  (h_physics : physics = 35)
  (h_chem : chem = 25)
  (h_math_physics : math_physics = 18)
  (h_math_chem : math_chem = 10)
  (h_physics_chem : physics_chem = 8)
  (h_all_three : all_three = 5) :
  total - (math + physics + chem - math_physics - math_chem - physics_chem + all_three) = 0 :=
by sorry

end NUMINAMATH_CALUDE_drama_club_subjects_l3685_368506


namespace NUMINAMATH_CALUDE_triangle_side_length_l3685_368557

/-- Given a triangle ABC with side lengths a = 2, b = 1, and angle C = 60°, 
    the length of side c is √3. -/
theorem triangle_side_length (a b c : ℝ) (C : ℝ) : 
  a = 2 → b = 1 → C = Real.pi / 3 → c = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3685_368557


namespace NUMINAMATH_CALUDE_mice_ratio_l3685_368552

theorem mice_ratio (white_mice brown_mice : ℕ) 
  (hw : white_mice = 14) 
  (hb : brown_mice = 7) : 
  (white_mice : ℚ) / (white_mice + brown_mice) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_mice_ratio_l3685_368552


namespace NUMINAMATH_CALUDE_number_of_molecules_value_l3685_368567

/-- The number of molecules in a given substance -/
def number_of_molecules : ℕ := 3 * 10^26

/-- Theorem stating that the number of molecules is 3 · 10^26 -/
theorem number_of_molecules_value : number_of_molecules = 3 * 10^26 := by
  sorry

end NUMINAMATH_CALUDE_number_of_molecules_value_l3685_368567


namespace NUMINAMATH_CALUDE_sqrt_sum_bounds_l3685_368544

theorem sqrt_sum_bounds : 
  let n : ℝ := Real.sqrt 4 + Real.sqrt 7
  4 < n ∧ n < 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_sum_bounds_l3685_368544


namespace NUMINAMATH_CALUDE_brothers_combined_age_l3685_368593

theorem brothers_combined_age : 
  ∀ (x y : ℕ), (x - 6 + y - 6 = 100) → (x + y = 112) :=
by
  sorry

end NUMINAMATH_CALUDE_brothers_combined_age_l3685_368593


namespace NUMINAMATH_CALUDE_sum_of_possible_values_l3685_368517

theorem sum_of_possible_values (e f : ℚ) : 
  (2 * |2 - e| = 5 ∧ |3 * e + f| = 7) → 
  (∃ e₁ f₁ e₂ f₂ : ℚ, 
    (2 * |2 - e₁| = 5 ∧ |3 * e₁ + f₁| = 7) ∧
    (2 * |2 - e₂| = 5 ∧ |3 * e₂ + f₂| = 7) ∧
    e₁ + f₁ + e₂ + f₂ = 6) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_possible_values_l3685_368517


namespace NUMINAMATH_CALUDE_constant_term_exists_l3685_368519

/-- Represents the derivative of a function q with respect to some variable -/
def derivative (q : ℝ → ℝ) : ℝ → ℝ := sorry

/-- The equation q' = 3q - 3 -/
def equation (q : ℝ → ℝ) : Prop :=
  ∀ x, derivative q x = 3 * q x - 3

/-- The value of (4')' is 72 -/
def condition (q : ℝ → ℝ) : Prop :=
  derivative (derivative q) 4 = 72

/-- There exists a constant term in the equation -/
theorem constant_term_exists (q : ℝ → ℝ) (h1 : equation q) (h2 : condition q) :
  ∃ c : ℝ, ∀ x, derivative q x = 3 * q x + c :=
sorry

end NUMINAMATH_CALUDE_constant_term_exists_l3685_368519


namespace NUMINAMATH_CALUDE_complex_equation_proof_l3685_368531

def complex_i : ℂ := Complex.I

theorem complex_equation_proof (z : ℂ) (h : z = 1 + complex_i) : 
  2 / z + z^2 = 1 + complex_i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_proof_l3685_368531


namespace NUMINAMATH_CALUDE_same_remainder_mod_27_l3685_368577

/-- Given a six-digit number X, Y is formed by moving the first three digits of X after the last three digits -/
def form_Y (X : ℕ) : ℕ :=
  let a := X / 1000
  let b := X % 1000
  1000 * b + a

/-- Theorem: For any six-digit number X, X and Y (formed from X) have the same remainder when divided by 27 -/
theorem same_remainder_mod_27 (X : ℕ) (h : 100000 ≤ X ∧ X < 1000000) :
  X % 27 = form_Y X % 27 := by
  sorry


end NUMINAMATH_CALUDE_same_remainder_mod_27_l3685_368577


namespace NUMINAMATH_CALUDE_range_of_f_less_than_zero_l3685_368554

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def decreasing_on_nonpositive (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 0 → f y ≤ f x

-- State the theorem
theorem range_of_f_less_than_zero
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_decreasing : decreasing_on_nonpositive f)
  (h_f_neg_two : f (-2) = 0) :
  {x : ℝ | f x < 0} = Set.Ioo (-2) 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_less_than_zero_l3685_368554


namespace NUMINAMATH_CALUDE_greg_trousers_count_l3685_368543

/-- The cost of a shirt -/
def shirtCost : ℝ := sorry

/-- The cost of a pair of trousers -/
def trousersCost : ℝ := sorry

/-- The cost of a tie -/
def tieCost : ℝ := sorry

/-- The number of trousers Greg bought in the first scenario -/
def firstScenarioTrousers : ℕ := sorry

theorem greg_trousers_count :
  (6 * shirtCost + firstScenarioTrousers * trousersCost + 2 * tieCost = 80) ∧
  (4 * shirtCost + 2 * trousersCost + 2 * tieCost = 140) ∧
  (5 * shirtCost + 3 * trousersCost + 2 * tieCost = 110) →
  firstScenarioTrousers = 4 := by
  sorry

end NUMINAMATH_CALUDE_greg_trousers_count_l3685_368543


namespace NUMINAMATH_CALUDE_rectangle_existence_l3685_368514

theorem rectangle_existence (m : ℕ) (h : m > 12) :
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x * y > m ∧ x * (y - 1) < m ∧ x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_rectangle_existence_l3685_368514


namespace NUMINAMATH_CALUDE_greg_granola_bars_l3685_368511

/-- Proves that Greg set aside 1 granola bar for each day of the week --/
theorem greg_granola_bars (total : ℕ) (traded : ℕ) (sisters : ℕ) (bars_per_sister : ℕ) (days : ℕ)
  (h_total : total = 20)
  (h_traded : traded = 3)
  (h_sisters : sisters = 2)
  (h_bars_per_sister : bars_per_sister = 5)
  (h_days : days = 7) :
  (total - traded - sisters * bars_per_sister) / days = 1 := by
  sorry

end NUMINAMATH_CALUDE_greg_granola_bars_l3685_368511


namespace NUMINAMATH_CALUDE_expression_evaluation_l3685_368541

theorem expression_evaluation :
  let a : ℝ := 40
  let c : ℝ := 4
  1891 - (1600 / a + 8040 / a) * c = 927 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3685_368541


namespace NUMINAMATH_CALUDE_binomial_square_constant_l3685_368551

theorem binomial_square_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 60*x + c = (x + a)^2) → c = 900 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l3685_368551


namespace NUMINAMATH_CALUDE_basketball_lineup_count_l3685_368538

def total_players : ℕ := 16
def twins : ℕ := 2
def seniors : ℕ := 5
def lineup_size : ℕ := 7

/-- The number of ways to choose a lineup of 7 players from a team of 16 players,
    including a set of twins and 5 seniors, where exactly one twin must be in the lineup
    and at least two seniors must be selected. -/
theorem basketball_lineup_count : 
  (Nat.choose twins 1) *
  (Nat.choose seniors 2 * Nat.choose (total_players - twins - seniors) 4 +
   Nat.choose seniors 3 * Nat.choose (total_players - twins - seniors) 3) = 4200 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_count_l3685_368538


namespace NUMINAMATH_CALUDE_inequality_proof_l3685_368560

theorem inequality_proof (a b c d : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
  (h_sum : a + b + c + d = 1) : 
  (a^2 / (1 + a)) + (b^2 / (1 + b)) + (c^2 / (1 + c)) + (d^2 / (1 + d)) ≥ 1/5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3685_368560


namespace NUMINAMATH_CALUDE_sum_norms_gt_sum_pairwise_norms_l3685_368580

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [CompleteSpace V]

/-- Given four pairwise non-parallel vectors whose sum is zero, 
    the sum of their norms is greater than the sum of the norms of their pairwise sums with the first vector -/
theorem sum_norms_gt_sum_pairwise_norms (a b c d : V) 
    (h_sum : a + b + c + d = 0)
    (h_ab : ¬ ∃ (k : ℝ), b = k • a)
    (h_ac : ¬ ∃ (k : ℝ), c = k • a)
    (h_ad : ¬ ∃ (k : ℝ), d = k • a)
    (h_bc : ¬ ∃ (k : ℝ), c = k • b)
    (h_bd : ¬ ∃ (k : ℝ), d = k • b)
    (h_cd : ¬ ∃ (k : ℝ), d = k • c) :
  ‖a‖ + ‖b‖ + ‖c‖ + ‖d‖ > ‖a + b‖ + ‖a + c‖ + ‖a + d‖ := by
  sorry

end NUMINAMATH_CALUDE_sum_norms_gt_sum_pairwise_norms_l3685_368580


namespace NUMINAMATH_CALUDE_stephanies_internet_bill_l3685_368537

/-- Stephanie's household budget problem -/
theorem stephanies_internet_bill :
  let electricity_bill : ℕ := 60
  let gas_bill : ℕ := 40
  let water_bill : ℕ := 40
  let gas_paid : ℚ := 3/4 * gas_bill + 5
  let water_paid : ℚ := 1/2 * water_bill
  let internet_payments : ℕ := 4
  let internet_payment_amount : ℕ := 5
  let total_remaining : ℕ := 30
  
  ∃ (internet_bill : ℕ),
    internet_bill = internet_payments * internet_payment_amount + 
      (total_remaining - (gas_bill - gas_paid) - (water_bill - water_paid)) :=
by
  sorry


end NUMINAMATH_CALUDE_stephanies_internet_bill_l3685_368537


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l3685_368532

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ := 5 * x - 2 * y + 2 * x * y

-- State the theorem
theorem diamond_equation_solution :
  ∃! y : ℝ, diamond 4 y = 30 ∧ y = 5/3 := by sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l3685_368532


namespace NUMINAMATH_CALUDE_petes_age_proof_l3685_368571

/-- Pete's current age -/
def petes_age : ℕ := 35

/-- Pete's son's current age -/
def sons_age : ℕ := 9

/-- Years into the future when the age comparison is made -/
def years_later : ℕ := 4

theorem petes_age_proof :
  petes_age = 35 ∧
  sons_age = 9 ∧
  petes_age + years_later = 3 * (sons_age + years_later) :=
by sorry

end NUMINAMATH_CALUDE_petes_age_proof_l3685_368571


namespace NUMINAMATH_CALUDE_nine_million_squared_zeros_l3685_368575

/-- For a positive integer n, represent a number composed of n nines -/
def all_nines (n : ℕ) : ℕ := 10^n - 1

/-- The number of zeros in the expansion of (all_nines n)² -/
def num_zeros (n : ℕ) : ℕ := n - 1

theorem nine_million_squared_zeros :
  ∃ (k : ℕ), all_nines 7 ^ 2 = k * 10^6 + m ∧ m < 10^6 :=
sorry

end NUMINAMATH_CALUDE_nine_million_squared_zeros_l3685_368575


namespace NUMINAMATH_CALUDE_circle_center_from_diameter_endpoints_l3685_368582

/-- The center of a circle given the endpoints of its diameter -/
theorem circle_center_from_diameter_endpoints (x₁ y₁ x₂ y₂ : ℝ) :
  let endpoint1 : ℝ × ℝ := (x₁, y₁)
  let endpoint2 : ℝ × ℝ := (x₂, y₂)
  let center : ℝ × ℝ := ((x₁ + x₂) / 2, (y₁ + y₂) / 2)
  endpoint1 = (2, -3) → endpoint2 = (8, 9) → center = (5, 3) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_from_diameter_endpoints_l3685_368582


namespace NUMINAMATH_CALUDE_candy_distribution_l3685_368562

theorem candy_distribution (total_candy : ℕ) (num_students : ℕ) (pieces_per_student : ℕ) : 
  total_candy = 344 → num_students = 43 → 
  pieces_per_student * num_students = total_candy →
  pieces_per_student = 8 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l3685_368562


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_12_l3685_368585

theorem smallest_divisible_by_1_to_12 : ∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 12 → k ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 12 → k ∣ m) → n ≤ m) ∧ n = 27720 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_12_l3685_368585


namespace NUMINAMATH_CALUDE_sqrt_inequality_sum_reciprocal_inequality_l3685_368533

-- Problem 1
theorem sqrt_inequality : Real.sqrt 3 + Real.sqrt 8 < 2 + Real.sqrt 7 := by sorry

-- Problem 2
theorem sum_reciprocal_inequality {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 1/a + 1/b + 1/c ≥ 9 := by sorry

end NUMINAMATH_CALUDE_sqrt_inequality_sum_reciprocal_inequality_l3685_368533


namespace NUMINAMATH_CALUDE_max_pyramid_volume_l3685_368530

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a pyramid with vertex O and base ABC -/
structure Pyramid where
  O : Point3D
  A : Point3D
  B : Point3D
  C : Point3D

/-- Calculates the angle between two vectors -/
def angle (v1 v2 : Point3D) : ℝ := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point3D) : ℝ := sorry

/-- Calculates the volume of a pyramid -/
def pyramidVolume (p : Pyramid) : ℝ := sorry

/-- Checks if a point is on the surface of a sphere -/
def isOnSphere (center : Point3D) (radius : ℝ) (point : Point3D) : Prop :=
  distance center point = radius

theorem max_pyramid_volume (p : Pyramid) (r : ℝ) :
  r = 3 →
  isOnSphere p.O r p.A →
  isOnSphere p.O r p.B →
  isOnSphere p.O r p.C →
  angle (p.A) (p.B) = 150 * π / 180 →
  ∀ (q : Pyramid), 
    isOnSphere p.O r q.A →
    isOnSphere p.O r q.B →
    isOnSphere p.O r q.C →
    pyramidVolume q ≤ 9/2 :=
by sorry

end NUMINAMATH_CALUDE_max_pyramid_volume_l3685_368530


namespace NUMINAMATH_CALUDE_intersection_distance_l3685_368570

/-- The distance between the points of intersection of x^2 + y = 12 and x + y = 12 is √2 -/
theorem intersection_distance : ∃ (p1 p2 : ℝ × ℝ),
  (p1.1^2 + p1.2 = 12 ∧ p1.1 + p1.2 = 12) ∧
  (p2.1^2 + p2.2 = 12 ∧ p2.1 + p2.2 = 12) ∧
  p1 ≠ p2 ∧
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_intersection_distance_l3685_368570


namespace NUMINAMATH_CALUDE_square_brush_ratio_l3685_368549

theorem square_brush_ratio (s w : ℝ) (h_positive_s : 0 < s) (h_positive_w : 0 < w) :
  (w^2 + 2 * (s^2 / 2 - w^2) = s^2 / 3) → (s / w = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_square_brush_ratio_l3685_368549


namespace NUMINAMATH_CALUDE_problem_solution_l3685_368501

theorem problem_solution (p q : Prop) (hp : 1 > -2) (hq : Even 2) : 
  (p ∨ q) ∧ (p ∧ q) ∧ ¬(¬p) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3685_368501


namespace NUMINAMATH_CALUDE_boat_rowing_probability_l3685_368555

theorem boat_rowing_probability : 
  let p_left1 : ℚ := 3/5  -- Probability of first left oar working
  let p_left2 : ℚ := 2/5  -- Probability of second left oar working
  let p_right1 : ℚ := 4/5  -- Probability of first right oar working
  let p_right2 : ℚ := 3/5  -- Probability of second right oar working
  
  -- Probability of both left oars failing
  let p_left_fail : ℚ := (1 - p_left1) * (1 - p_left2)
  
  -- Probability of both right oars failing
  let p_right_fail : ℚ := (1 - p_right1) * (1 - p_right2)
  
  -- Probability of all four oars failing
  let p_all_fail : ℚ := p_left_fail * p_right_fail
  
  -- Probability of being able to row the boat
  let p_row : ℚ := 1 - (p_left_fail + p_right_fail - p_all_fail)
  
  p_row = 437/625 := by sorry

end NUMINAMATH_CALUDE_boat_rowing_probability_l3685_368555


namespace NUMINAMATH_CALUDE_distinct_triangles_in_grid_l3685_368583

/-- The number of points in each row of the grid -/
def rows : ℕ := 3

/-- The number of points in each column of the grid -/
def columns : ℕ := 4

/-- The total number of points in the grid -/
def total_points : ℕ := rows * columns

/-- The number of degenerate cases (collinear points) -/
def degenerate_cases : ℕ := rows + columns + 2

theorem distinct_triangles_in_grid : 
  (total_points.choose 3) - degenerate_cases = 76 := by sorry

end NUMINAMATH_CALUDE_distinct_triangles_in_grid_l3685_368583


namespace NUMINAMATH_CALUDE_matrix_is_square_iff_a_eq_zero_l3685_368553

def A (a : ℚ) : Matrix (Fin 4) (Fin 4) ℚ :=
  !![a,   -a,  -1,   0;
     a,   -a,   0,  -1;
     1,    0,   a,  -a;
     0,    1,   a,  -a]

theorem matrix_is_square_iff_a_eq_zero (a : ℚ) :
  (∃ C : Matrix (Fin 4) (Fin 4) ℚ, A a = C ^ 2) ↔ a = 0 := by sorry

end NUMINAMATH_CALUDE_matrix_is_square_iff_a_eq_zero_l3685_368553


namespace NUMINAMATH_CALUDE_quadratic_completing_square_l3685_368594

theorem quadratic_completing_square :
  ∀ x : ℝ, x^2 - 4*x - 5 = 0 ↔ (x - 2)^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_l3685_368594


namespace NUMINAMATH_CALUDE_extremum_implies_a_equals_negative_four_l3685_368563

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 - b*x + a^2 - 6*a

-- Define the derivative of f(x)
def f' (a b x : ℝ) : ℝ := 3*x^2 - 2*a*x - b

-- Theorem statement
theorem extremum_implies_a_equals_negative_four (a b : ℝ) :
  f' a b 2 = 0 ∧ f a b 2 = 8 → a = -4 :=
by sorry

end NUMINAMATH_CALUDE_extremum_implies_a_equals_negative_four_l3685_368563


namespace NUMINAMATH_CALUDE_smallest_number_l3685_368520

theorem smallest_number (a b c d : ℤ) (h1 : a = 1) (h2 : b = 0) (h3 : c = -1) (h4 : d = -3) :
  d ≤ a ∧ d ≤ b ∧ d ≤ c :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3685_368520


namespace NUMINAMATH_CALUDE_simplify_fraction_l3685_368502

theorem simplify_fraction (x : ℝ) (h : x ≠ 2) :
  (1 + 1 / (x - 2)) / ((x - x^2) / (x - 2)) = -(x - 1) / x := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3685_368502


namespace NUMINAMATH_CALUDE_total_amount_is_2500_l3685_368528

/-- Proves that the total amount of money divided into two parts is 2500,
    given the conditions from the original problem. -/
theorem total_amount_is_2500 
  (total : ℝ) 
  (part1 : ℝ) 
  (part2 : ℝ) 
  (h1 : total = part1 + part2)
  (h2 : part1 = 1000)
  (h3 : 0.05 * part1 + 0.06 * part2 = 140) :
  total = 2500 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_is_2500_l3685_368528


namespace NUMINAMATH_CALUDE_no_common_solution_l3685_368589

theorem no_common_solution :
  ¬∃ x : ℚ, (6 * (x - 2/3) - (x + 7) = 11) ∧ ((2*x - 1)/3 = (2*x + 1)/6 - 2) := by
  sorry

end NUMINAMATH_CALUDE_no_common_solution_l3685_368589


namespace NUMINAMATH_CALUDE_toaster_pricing_theorem_l3685_368584

/-- Represents the relationship between cost and number of purchasers for toasters -/
def toaster_relation (c p : ℝ) : Prop := c * p = 6000

theorem toaster_pricing_theorem :
  -- Given condition
  toaster_relation 300 20 →
  -- Proofs to show
  (toaster_relation 600 10 ∧ toaster_relation 400 15) :=
by
  sorry

end NUMINAMATH_CALUDE_toaster_pricing_theorem_l3685_368584


namespace NUMINAMATH_CALUDE_third_term_geometric_sequence_l3685_368547

theorem third_term_geometric_sequence
  (q : ℝ)
  (h_q_abs : |q| < 1)
  (h_sum : (a : ℕ → ℝ) → (∀ n, a (n + 1) = q * a n) → (∑' n, a n) = 8/5)
  (h_second_term : ∃ a : ℕ → ℝ, (∀ n, a (n + 1) = q * a n) ∧ a 1 = -1/2) :
  ∃ a : ℕ → ℝ, (∀ n, a (n + 1) = q * a n) ∧ a 1 = -1/2 ∧ a 2 = 1/8 :=
sorry

end NUMINAMATH_CALUDE_third_term_geometric_sequence_l3685_368547


namespace NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l3685_368546

theorem unique_solution_trigonometric_equation :
  ∃! (x : ℝ), 0 < x ∧ x < 1 ∧ Real.sin (Real.arccos (Real.tan (Real.arcsin x))) = x :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l3685_368546


namespace NUMINAMATH_CALUDE_max_k_value_l3685_368566

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (heq : 4 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + k * ((x / y) + (y / x))) :
  k ≤ (3/2) ∧ ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 4 = (3/2)^2 * ((x^2 / y^2) + (y^2 / x^2)) + (3/2) * ((x / y) + (y / x)) :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l3685_368566
