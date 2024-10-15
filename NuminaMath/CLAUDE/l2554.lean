import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_simplification_l2554_255423

theorem polynomial_simplification (x : ℝ) : 
  (2 * x^4 + 3 * x^3 - 5 * x^2 + 6 * x - 8) + (-5 * x^4 - 2 * x^3 + 4 * x^2 - 6 * x + 7) = 
  -3 * x^4 + x^3 - x^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2554_255423


namespace NUMINAMATH_CALUDE_common_difference_is_two_l2554_255458

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_arithmetic : ∀ n, a (n + 1) = a n + d
  h_nonzero : d ≠ 0
  h_sum : a 1 + a 2 + a 3 = 9
  h_geometric : ∃ r : ℝ, r ≠ 0 ∧ a 2 = r * a 1 ∧ a 5 = r * a 2

/-- The common difference of the arithmetic sequence is 2 -/
theorem common_difference_is_two (seq : ArithmeticSequence) : seq.d = 2 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_two_l2554_255458


namespace NUMINAMATH_CALUDE_quadratic_monotonicity_l2554_255496

/-- A function f is monotonic on an open interval (a, b) if it is either
    strictly increasing or strictly decreasing on that interval. -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a < x ∧ x < y ∧ y < b → f x < f y) ∨
  (∀ x y, a < x ∧ x < y ∧ y < b → f y < f x)

theorem quadratic_monotonicity (a : ℝ) :
  IsMonotonic (fun x ↦ x^2 + 2*(a-1)*x + 2) 2 4 →
  a ≤ -3 ∨ a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_monotonicity_l2554_255496


namespace NUMINAMATH_CALUDE_boa_constrictor_length_alberts_boa_length_l2554_255494

/-- The length of Albert's boa constrictor given the length of his garden snake and their relative sizes. -/
theorem boa_constrictor_length (garden_snake_length : ℕ) (relative_size : ℕ) : ℕ :=
  garden_snake_length * relative_size

/-- Proof that Albert's boa constrictor is 70 inches long. -/
theorem alberts_boa_length : boa_constrictor_length 10 7 = 70 := by
  sorry

end NUMINAMATH_CALUDE_boa_constrictor_length_alberts_boa_length_l2554_255494


namespace NUMINAMATH_CALUDE_distance_after_two_hours_l2554_255422

/-- Anna's jogging speed in miles per minute -/
def anna_speed : ℚ := 1 / 20

/-- Mark's running speed in miles per minute -/
def mark_speed : ℚ := 3 / 40

/-- The time period in minutes -/
def time_period : ℚ := 2 * 60

/-- The theorem stating the distance between Anna and Mark after 2 hours -/
theorem distance_after_two_hours :
  anna_speed * time_period + mark_speed * time_period = 9 := by sorry

end NUMINAMATH_CALUDE_distance_after_two_hours_l2554_255422


namespace NUMINAMATH_CALUDE_abs_diff_eq_sum_abs_iff_product_nonpositive_l2554_255410

theorem abs_diff_eq_sum_abs_iff_product_nonpositive (a b : ℝ) :
  |a - b| = |a| + |b| ↔ a * b ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_eq_sum_abs_iff_product_nonpositive_l2554_255410


namespace NUMINAMATH_CALUDE_greatest_x_value_l2554_255450

theorem greatest_x_value (x : ℤ) (h : (2.134 : ℝ) * (10 : ℝ) ^ (x : ℝ) < 210000) :
  x ≤ 4 ∧ ∃ y : ℤ, y > 4 → (2.134 : ℝ) * (10 : ℝ) ^ (y : ℝ) ≥ 210000 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l2554_255450


namespace NUMINAMATH_CALUDE_epidemic_duration_l2554_255469

structure Dwarf :=
  (id : ℕ)
  (status : ℕ → Nat)  -- 0: healthy, 1: sick, 2: immune

def Epidemic (population : List Dwarf) (day : ℕ) : Prop :=
  ∃ (d : Dwarf), d ∈ population ∧ d.status day = 1

def VisitsSickFriends (d : Dwarf) (population : List Dwarf) (day : ℕ) : Prop :=
  d.status day = 0 → ∃ (sick : Dwarf), sick ∈ population ∧ sick.status day = 1

def BecomeSick (d : Dwarf) (population : List Dwarf) (day : ℕ) : Prop :=
  VisitsSickFriends d population day → d.status (day + 1) = 1

def ImmunityPeriod (d : Dwarf) (day : ℕ) : Prop :=
  d.status day = 1 → d.status (day + 1) = 2

def CannotInfectImmune (d : Dwarf) (day : ℕ) : Prop :=
  d.status day = 2 → d.status (day + 1) ≠ 1

theorem epidemic_duration (population : List Dwarf) :
  (∃ (d : Dwarf), d ∈ population ∧ d.status 0 = 2) →
    ∀ (n : ℕ), ∃ (m : ℕ), m ≥ n ∧ Epidemic population m
  ∧
  (∀ (d : Dwarf), d ∈ population → d.status 0 ≠ 2) →
    ∃ (n : ℕ), ∀ (m : ℕ), m ≥ n → ¬(Epidemic population m) :=
by sorry

end NUMINAMATH_CALUDE_epidemic_duration_l2554_255469


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2554_255441

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(2*x - 1)
  f (1/2) = 1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2554_255441


namespace NUMINAMATH_CALUDE_saree_stripe_ratio_l2554_255477

theorem saree_stripe_ratio :
  ∀ (gold brown blue : ℕ),
    gold = brown →
    blue = 5 * gold →
    brown = 4 →
    blue = 60 →
    (gold : ℚ) / brown = 3 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_saree_stripe_ratio_l2554_255477


namespace NUMINAMATH_CALUDE_interest_rate_is_four_percent_l2554_255439

/-- Given a principal sum and an interest rate, if the simple interest
    for 5 years is one-fifth of the principal, then the interest rate is 4% -/
theorem interest_rate_is_four_percent 
  (P : ℝ) -- Principal sum
  (R : ℝ) -- Interest rate as a percentage
  (h : P > 0) -- Assumption that principal is positive
  (h_interest : P / 5 = (P * R * 5) / 100) -- Condition that interest is one-fifth of principal
  : R = 4 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_is_four_percent_l2554_255439


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2554_255493

theorem geometric_sequence_sum (a₁ a₂ a₃ a₄ a₅ : ℕ) (q : ℚ) :
  (a₁ > 0) →
  (a₂ > a₁) → (a₃ > a₂) → (a₄ > a₃) → (a₅ > a₄) →
  (a₂ = a₁ * q) → (a₃ = a₂ * q) → (a₄ = a₃ * q) → (a₅ = a₄ * q) →
  (a₁ + a₂ + a₃ + a₄ + a₅ = 211) →
  (a₁ = 16 ∧ q = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2554_255493


namespace NUMINAMATH_CALUDE_one_black_two_white_reachable_l2554_255421

-- Define the urn state as a pair of natural numbers (white, black)
def UrnState := ℕ × ℕ

-- Define the initial state
def initial_state : UrnState := (50, 150)

-- Define the four operations
def op1 (s : UrnState) : UrnState := (s.1, s.2 - 2)
def op2 (s : UrnState) : UrnState := (s.1, s.2 - 1)
def op3 (s : UrnState) : UrnState := (s.1, s.2)
def op4 (s : UrnState) : UrnState := (s.1 - 3, s.2 + 2)

-- Define a predicate for valid states (non-negative marbles)
def valid_state (s : UrnState) : Prop := s.1 ≥ 0 ∧ s.2 ≥ 0

-- Define the reachability relation
inductive reachable : UrnState → Prop where
  | initial : reachable initial_state
  | op1 : ∀ s, reachable s → valid_state (op1 s) → reachable (op1 s)
  | op2 : ∀ s, reachable s → valid_state (op2 s) → reachable (op2 s)
  | op3 : ∀ s, reachable s → valid_state (op3 s) → reachable (op3 s)
  | op4 : ∀ s, reachable s → valid_state (op4 s) → reachable (op4 s)

-- Theorem stating that the configuration (2, 1) is reachable
theorem one_black_two_white_reachable : reachable (2, 1) := by sorry

end NUMINAMATH_CALUDE_one_black_two_white_reachable_l2554_255421


namespace NUMINAMATH_CALUDE_factorization_problems_l2554_255451

theorem factorization_problems (x y : ℝ) : 
  (x^3 - 6*x^2 + 9*x = x*(x-3)^2) ∧ 
  ((x-2)^2 - x + 2 = (x-2)*(x-3)) ∧ 
  ((x^2 + y^2)^2 - 4*x^2*y^2 = (x+y)^2*(x-y)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l2554_255451


namespace NUMINAMATH_CALUDE_furniture_pricing_l2554_255480

/-- The cost price of furniture before markup and discount -/
def cost_price : ℝ := 7777.78

/-- The markup percentage applied to the cost price -/
def markup_percentage : ℝ := 0.20

/-- The discount percentage applied to the total price -/
def discount_percentage : ℝ := 0.10

/-- The final price paid by the customer after markup and discount -/
def final_price : ℝ := 8400

theorem furniture_pricing :
  final_price = (1 - discount_percentage) * (1 + markup_percentage) * cost_price := by
  sorry

end NUMINAMATH_CALUDE_furniture_pricing_l2554_255480


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2554_255482

theorem complex_number_quadrant (z : ℂ) (h : (3 + 4*I)*z = 25) : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2554_255482


namespace NUMINAMATH_CALUDE_existence_of_invariant_sequences_l2554_255483

/-- A binary sequence is a function from ℕ to {0, 1} -/
def BinarySeq := ℕ → Fin 2

/-- Remove odd-indexed elements from a sequence -/
def removeOdd (s : BinarySeq) : BinarySeq :=
  fun n => s (2 * n + 1)

/-- Remove even-indexed elements from a sequence -/
def removeEven (s : BinarySeq) : BinarySeq :=
  fun n => s (2 * n)

/-- A sequence is invariant under odd removal if removing odd-indexed elements results in the same sequence -/
def invariantUnderOddRemoval (s : BinarySeq) : Prop :=
  ∀ n, s n = removeOdd s n

/-- A sequence is invariant under even removal if removing even-indexed elements results in the same sequence -/
def invariantUnderEvenRemoval (s : BinarySeq) : Prop :=
  ∀ n, s n = removeEven s n

theorem existence_of_invariant_sequences :
  (∃ s : BinarySeq, invariantUnderOddRemoval s) ∧
  (∃ s : BinarySeq, invariantUnderEvenRemoval s) :=
sorry

end NUMINAMATH_CALUDE_existence_of_invariant_sequences_l2554_255483


namespace NUMINAMATH_CALUDE_product_173_240_l2554_255419

theorem product_173_240 : 
  (∃ n : ℕ, n * 12 = 173 * 240 ∧ n = 3460) → 173 * 240 = 41520 := by
  sorry

end NUMINAMATH_CALUDE_product_173_240_l2554_255419


namespace NUMINAMATH_CALUDE_equation_solutions_l2554_255442

theorem equation_solutions :
  (∀ x : ℝ, 2 * x^2 - 1 = 49 ↔ x = 5 ∨ x = -5) ∧
  (∀ x : ℝ, (x + 3)^3 = 64 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2554_255442


namespace NUMINAMATH_CALUDE_smallest_seating_arrangement_l2554_255474

/-- Represents a circular seating arrangement -/
structure CircularSeating where
  total_chairs : ℕ
  seated_people : ℕ

/-- Checks if a seating arrangement satisfies the condition that any new person must sit next to someone -/
def satisfies_condition (seating : CircularSeating) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ seating.total_chairs → 
    ∃ i j, i ≠ j ∧ 
           (i % seating.total_chairs + 1 = k ∨ (i + 1) % seating.total_chairs + 1 = k) ∧
           (j % seating.total_chairs + 1 = k ∨ (j + 1) % seating.total_chairs + 1 = k)

/-- The main theorem to prove -/
theorem smallest_seating_arrangement :
  ∀ n < 25, ¬(satisfies_condition ⟨100, n⟩) ∧ 
  satisfies_condition ⟨100, 25⟩ := by
  sorry

#check smallest_seating_arrangement

end NUMINAMATH_CALUDE_smallest_seating_arrangement_l2554_255474


namespace NUMINAMATH_CALUDE_book_pages_l2554_255434

/-- The number of pages Ceasar has already read -/
def pages_read : ℕ := 147

/-- The number of pages Ceasar has left to read -/
def pages_left : ℕ := 416

/-- The total number of pages in the book -/
def total_pages : ℕ := pages_read + pages_left

/-- Theorem stating that the total number of pages in the book is 563 -/
theorem book_pages : total_pages = 563 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_l2554_255434


namespace NUMINAMATH_CALUDE_unique_prime_pair_sum_53_l2554_255495

/-- A function that checks if a natural number is prime -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- The theorem stating that there is exactly one pair of primes summing to 53 -/
theorem unique_prime_pair_sum_53 : 
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 53 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_pair_sum_53_l2554_255495


namespace NUMINAMATH_CALUDE_solve_for_y_l2554_255461

theorem solve_for_y (x y : ℝ) (h1 : x^2 - 3*x + 2 = y + 6) (h2 : x = -4) : y = 24 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2554_255461


namespace NUMINAMATH_CALUDE_manager_salary_is_4200_l2554_255475

/-- Calculates the manager's salary given the number of employees, their average salary,
    and the increase in average salary when the manager's salary is added. -/
def managerSalary (numEmployees : ℕ) (avgSalary : ℚ) (avgIncrease : ℚ) : ℚ :=
  (avgSalary + avgIncrease) * (numEmployees + 1) - avgSalary * numEmployees

/-- Proves that the manager's salary is 4200 given the problem conditions. -/
theorem manager_salary_is_4200 :
  managerSalary 15 1800 150 = 4200 := by
  sorry

#eval managerSalary 15 1800 150

end NUMINAMATH_CALUDE_manager_salary_is_4200_l2554_255475


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l2554_255460

theorem sqrt_equation_solutions :
  ∀ x : ℝ, (Real.sqrt x = 18 / (11 - Real.sqrt x)) ↔ (x = 81 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l2554_255460


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l2554_255400

/-- Proves the length of a bridge given train specifications and crossing times -/
theorem bridge_length_calculation (train_length : ℝ) (signal_post_time : ℝ) (bridge_time : ℝ) :
  train_length = 600 →
  signal_post_time = 40 →
  bridge_time = 600 →
  let train_speed := train_length / signal_post_time
  let bridge_length := train_speed * bridge_time - train_length
  bridge_length = 8400 := by
sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l2554_255400


namespace NUMINAMATH_CALUDE_ants_sugar_harvesting_l2554_255484

def sugar_harvesting (initial_sugar : ℝ) (removal_rate : ℝ) (time_passed : ℝ) : Prop :=
  let remaining_sugar := initial_sugar - removal_rate * time_passed
  let remaining_time := remaining_sugar / removal_rate
  remaining_time = 3

theorem ants_sugar_harvesting :
  sugar_harvesting 24 4 3 :=
sorry

end NUMINAMATH_CALUDE_ants_sugar_harvesting_l2554_255484


namespace NUMINAMATH_CALUDE_curve_condition_iff_l2554_255440

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A curve defined by a function f(x, y) = 0 -/
structure Curve where
  f : ℝ → ℝ → ℝ

/-- Predicate for a point being on a curve -/
def IsOnCurve (p : Point) (c : Curve) : Prop :=
  c.f p.x p.y = 0

/-- Theorem stating that f(x, y) = 0 is a necessary and sufficient condition
    for a point P(x, y) to be on the curve f(x, y) = 0 -/
theorem curve_condition_iff (c : Curve) (p : Point) :
  IsOnCurve p c ↔ c.f p.x p.y = 0 := by sorry

end NUMINAMATH_CALUDE_curve_condition_iff_l2554_255440


namespace NUMINAMATH_CALUDE_derivative_of_two_sin_x_l2554_255446

theorem derivative_of_two_sin_x (x : ℝ) :
  deriv (λ x => 2 * Real.sin x) x = 2 * Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_two_sin_x_l2554_255446


namespace NUMINAMATH_CALUDE_monday_sunday_speed_ratio_l2554_255413

/-- Proves that the ratio of speeds on Monday (first 32 miles) to Sunday is 2:1 -/
theorem monday_sunday_speed_ratio 
  (total_distance : ℝ) 
  (sunday_speed : ℝ) 
  (monday_first_distance : ℝ) 
  (monday_first_speed : ℝ) :
  total_distance = 120 →
  monday_first_distance = 32 →
  (total_distance / sunday_speed) * 1.6 = 
    (monday_first_distance / monday_first_speed) + 
    ((total_distance - monday_first_distance) / (sunday_speed / 2)) →
  monday_first_speed / sunday_speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_monday_sunday_speed_ratio_l2554_255413


namespace NUMINAMATH_CALUDE_gum_pack_size_l2554_255427

/-- The number of pieces of banana gum Luke has initially -/
def banana_gum : ℕ := 28

/-- The number of pieces of apple gum Luke has initially -/
def apple_gum : ℕ := 36

/-- The number of pieces of gum in each complete pack -/
def y : ℕ := 14

theorem gum_pack_size :
  (banana_gum - 2 * y) * (apple_gum + 3 * y) = banana_gum * apple_gum := by
  sorry

#check gum_pack_size

end NUMINAMATH_CALUDE_gum_pack_size_l2554_255427


namespace NUMINAMATH_CALUDE_car_savings_calculation_l2554_255426

theorem car_savings_calculation 
  (monthly_earnings : ℕ) 
  (car_cost : ℕ) 
  (total_earnings : ℕ) 
  (h1 : monthly_earnings = 4000)
  (h2 : car_cost = 45000)
  (h3 : total_earnings = 360000) :
  car_cost / (total_earnings / monthly_earnings) = 500 := by
sorry

end NUMINAMATH_CALUDE_car_savings_calculation_l2554_255426


namespace NUMINAMATH_CALUDE_cos_48_degrees_l2554_255473

theorem cos_48_degrees :
  ∃ x : ℝ, 4 * x^3 - 3 * x - (1 + Real.sqrt 5) / 4 = 0 ∧
  Real.cos (48 * π / 180) = (1 / 2) * x + (Real.sqrt 3 / 2) * Real.sqrt (1 - x^2) := by
  sorry

end NUMINAMATH_CALUDE_cos_48_degrees_l2554_255473


namespace NUMINAMATH_CALUDE_max_teams_advancing_l2554_255412

/-- The number of teams in the tournament -/
def num_teams : ℕ := 7

/-- The minimum number of points required to advance -/
def min_points_to_advance : ℕ := 13

/-- The number of points awarded for a win -/
def win_points : ℕ := 3

/-- The number of points awarded for a draw -/
def draw_points : ℕ := 1

/-- The number of points awarded for a loss -/
def loss_points : ℕ := 0

/-- The total number of games played in the tournament -/
def total_games : ℕ := (num_teams * (num_teams - 1)) / 2

/-- The maximum total points that can be awarded in the tournament -/
def max_total_points : ℕ := total_games * win_points

/-- Theorem stating the maximum number of teams that can advance -/
theorem max_teams_advancing :
  ∀ n : ℕ, (n * min_points_to_advance ≤ max_total_points) →
  (∀ m : ℕ, m > n → m * min_points_to_advance > max_total_points) →
  n = 4 := by sorry

end NUMINAMATH_CALUDE_max_teams_advancing_l2554_255412


namespace NUMINAMATH_CALUDE_parabola_vertex_l2554_255457

/-- The vertex of the parabola y = x^2 - 6x + 1 has coordinates (3, -8) -/
theorem parabola_vertex (x y : ℝ) : 
  y = x^2 - 6*x + 1 → ∃ (h k : ℝ), h = 3 ∧ k = -8 ∧ ∀ x, y = (x - h)^2 + k := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2554_255457


namespace NUMINAMATH_CALUDE_exhibition_arrangement_l2554_255407

/-- The number of display stands -/
def n : ℕ := 9

/-- The number of exhibits -/
def k : ℕ := 3

/-- The number of ways to arrange k distinct objects in n positions,
    where the objects cannot be placed at the ends or adjacent to each other -/
def arrangement_count (n k : ℕ) : ℕ :=
  if n < 2 * k + 1 then 0
  else (n - k - 1).choose k * k.factorial

theorem exhibition_arrangement :
  arrangement_count n k = 60 := by sorry

end NUMINAMATH_CALUDE_exhibition_arrangement_l2554_255407


namespace NUMINAMATH_CALUDE_difference_of_squares_l2554_255405

theorem difference_of_squares (m : ℝ) : m^2 - 4 = (m + 2) * (m - 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2554_255405


namespace NUMINAMATH_CALUDE_first_digit_of_5_to_n_l2554_255403

theorem first_digit_of_5_to_n (n : ℕ) : 
  (∃ k : ℕ, 7 * 10^k ≤ 2^n ∧ 2^n < 8 * 10^k) → 
  (∃ m : ℕ, 10^m ≤ 5^n ∧ 5^n < 2 * 10^m) :=
by sorry

end NUMINAMATH_CALUDE_first_digit_of_5_to_n_l2554_255403


namespace NUMINAMATH_CALUDE_tangent_triangle_area_l2554_255476

-- Define the curve
def curve (x y : ℝ) : Prop := x * y - x + 2 * y - 5 = 0

-- Define the point A
def point_A : ℝ × ℝ := (1, 2)

-- Define the tangent line at point A
def tangent_line (x y : ℝ) : Prop := x + 3 * y - 7 = 0

-- Theorem statement
theorem tangent_triangle_area : 
  curve point_A.1 point_A.2 →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    tangent_line x₁ y₁ ∧ 
    tangent_line x₂ y₂ ∧ 
    x₁ = 0 ∧ 
    y₂ = 0 ∧ 
    (1/2 * x₂ * y₁ = 49/6)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_triangle_area_l2554_255476


namespace NUMINAMATH_CALUDE_lamp_arrangements_count_l2554_255411

/-- The number of ways to select k items from n items. -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to turn off 3 lamps in a row of 10 lamps,
    where the end lamps must remain on and no two consecutive lamps can be off. -/
def lamp_arrangements : ℕ := choose 6 3

theorem lamp_arrangements_count : lamp_arrangements = 20 := by sorry

end NUMINAMATH_CALUDE_lamp_arrangements_count_l2554_255411


namespace NUMINAMATH_CALUDE_angle_measure_proof_l2554_255425

theorem angle_measure_proof (C D : ℝ) : 
  C + D = 180 →  -- Angles are supplementary
  C = 7 * D →    -- C is 7 times D
  C = 157.5 :=   -- Measure of angle C
by sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l2554_255425


namespace NUMINAMATH_CALUDE_circle_equation_l2554_255472

/-- Given a circle with center (-1, 2) passing through the point (2, -2),
    its standard equation is (x+1)^2 + (y-2)^2 = 25 -/
theorem circle_equation (x y : ℝ) : 
  let center := (-1, 2)
  let point_on_circle := (2, -2)
  (x + 1)^2 + (y - 2)^2 = 25 ↔ 
    (∃ (r : ℝ), r > 0 ∧
      (x - center.1)^2 + (y - center.2)^2 = r^2 ∧
      (point_on_circle.1 - center.1)^2 + (point_on_circle.2 - center.2)^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2554_255472


namespace NUMINAMATH_CALUDE_triangle_area_problem_l2554_255471

theorem triangle_area_problem (x : ℝ) (h1 : x > 0) 
  (h2 : (1/2) * x * 3*x = 72) : x = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_problem_l2554_255471


namespace NUMINAMATH_CALUDE_planning_committee_combinations_l2554_255429

theorem planning_committee_combinations (n : ℕ) (k : ℕ) : n = 20 ∧ k = 3 → Nat.choose n k = 1140 := by
  sorry

end NUMINAMATH_CALUDE_planning_committee_combinations_l2554_255429


namespace NUMINAMATH_CALUDE_fraction_addition_l2554_255497

theorem fraction_addition (d : ℝ) : (6 + 5 * d) / 9 + 3 = (33 + 5 * d) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l2554_255497


namespace NUMINAMATH_CALUDE_father_son_age_sum_father_son_age_sum_proof_l2554_255430

/-- Given that:
  1) Eighteen years ago, the father was 3 times as old as his son.
  2) Now, the father is twice as old as his son.
  Prove that the sum of their current ages is 108 years. -/
theorem father_son_age_sum : ℕ → ℕ → Prop :=
  fun (son_age father_age : ℕ) =>
    (father_age - 18 = 3 * (son_age - 18)) →
    (father_age = 2 * son_age) →
    (son_age + father_age = 108)

/-- Proof of the theorem -/
theorem father_son_age_sum_proof : ∃ (son_age father_age : ℕ),
  father_son_age_sum son_age father_age :=
by
  sorry

end NUMINAMATH_CALUDE_father_son_age_sum_father_son_age_sum_proof_l2554_255430


namespace NUMINAMATH_CALUDE_integers_between_sqrt3_and_sqrt13_two_and_three_between_sqrt3_and_sqrt13_only_two_and_three_between_sqrt3_and_sqrt13_l2554_255417

theorem integers_between_sqrt3_and_sqrt13 :
  ∃ (n : ℤ), (↑n : ℝ) > Real.sqrt 3 ∧ (↑n : ℝ) < Real.sqrt 13 :=
by
  sorry

theorem two_and_three_between_sqrt3_and_sqrt13 :
  (2 : ℝ) > Real.sqrt 3 ∧ (2 : ℝ) < Real.sqrt 13 ∧
  (3 : ℝ) > Real.sqrt 3 ∧ (3 : ℝ) < Real.sqrt 13 :=
by
  sorry

theorem only_two_and_three_between_sqrt3_and_sqrt13 :
  ∀ (n : ℤ), (↑n : ℝ) > Real.sqrt 3 ∧ (↑n : ℝ) < Real.sqrt 13 → n = 2 ∨ n = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_integers_between_sqrt3_and_sqrt13_two_and_three_between_sqrt3_and_sqrt13_only_two_and_three_between_sqrt3_and_sqrt13_l2554_255417


namespace NUMINAMATH_CALUDE_placement_theorem_l2554_255449

def number_of_placements (n : ℕ) : ℕ := 
  Nat.choose 4 2 * (n * (n - 1))

theorem placement_theorem : number_of_placements 4 = 72 := by
  sorry

end NUMINAMATH_CALUDE_placement_theorem_l2554_255449


namespace NUMINAMATH_CALUDE_quadratic_function_solution_set_l2554_255402

/-- Given a quadratic function f(x) = x^2 + bx + 1 where f(-1) = f(3),
    prove that the solution set of f(x) > 0 is {x ∈ ℝ | x ≠ 1} -/
theorem quadratic_function_solution_set
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = x^2 + b*x + 1)
  (h2 : f (-1) = f 3)
  : {x : ℝ | f x > 0} = {x : ℝ | x ≠ 1} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_solution_set_l2554_255402


namespace NUMINAMATH_CALUDE_counterfeit_banknote_theorem_l2554_255433

/-- Represents a banknote with a natural number denomination -/
structure Banknote where
  denomination : ℕ

/-- Represents a collection of banknotes -/
def BanknoteCollection := List Banknote

/-- The detector's reading of the total sum -/
def detectorSum (collection : BanknoteCollection) : ℕ := sorry

/-- The actual sum of genuine banknotes -/
def genuineSum (collection : BanknoteCollection) : ℕ := sorry

/-- Predicate to check if a collection has pairwise different denominations -/
def hasPairwiseDifferentDenominations (collection : BanknoteCollection) : Prop := sorry

/-- Predicate to check if a collection has exactly one counterfeit banknote -/
def hasExactlyOneCounterfeit (collection : BanknoteCollection) : Prop := sorry

/-- The denomination of the counterfeit banknote -/
def counterfeitDenomination (collection : BanknoteCollection) : ℕ := sorry

theorem counterfeit_banknote_theorem (collection : BanknoteCollection) 
  (h1 : hasPairwiseDifferentDenominations collection)
  (h2 : hasExactlyOneCounterfeit collection) :
  detectorSum collection - genuineSum collection = counterfeitDenomination collection := by
  sorry

end NUMINAMATH_CALUDE_counterfeit_banknote_theorem_l2554_255433


namespace NUMINAMATH_CALUDE_final_distance_after_checkpoints_l2554_255492

/-- Represents the state of a car on the highway -/
structure CarState where
  position : ℝ
  speed : ℝ

/-- Represents a checkpoint on the highway -/
structure Checkpoint where
  position : ℝ
  new_speed : ℝ

/-- Updates the car state after passing a checkpoint -/
def update_car_state (car : CarState) (checkpoint : Checkpoint) : CarState :=
  { position := checkpoint.position, speed := checkpoint.new_speed }

/-- Calculates the final distance between two cars after passing checkpoints -/
def final_distance (initial_distance : ℝ) (initial_speed : ℝ) (checkpoints : List Checkpoint) : ℝ :=
  sorry

/-- Theorem stating the final distance between the cars -/
theorem final_distance_after_checkpoints :
  let initial_distance := 100
  let initial_speed := 60
  let checkpoints := [
    { position := 1000, new_speed := 80 },
    { position := 2000, new_speed := 100 },
    { position := 3000, new_speed := 120 }
  ]
  final_distance initial_distance initial_speed checkpoints = 200 := by
  sorry

end NUMINAMATH_CALUDE_final_distance_after_checkpoints_l2554_255492


namespace NUMINAMATH_CALUDE_largest_angle_right_in_special_triangle_l2554_255481

/-- A triangle with sides a, b, c and semiperimeter s -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  s : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  semiperimeter_def : s = (a + b + c) / 2

/-- The radii of the circles tangent to the sides of the triangle -/
structure TriangleRadii (t : Triangle) where
  r : ℝ  -- inradius
  ra : ℝ  -- exradius opposite to side a
  rb : ℝ  -- exradius opposite to side b
  rc : ℝ  -- exradius opposite to side c
  radii_relations : 
    t.s * r = (t.s - t.a) * ra ∧
    t.s * r = (t.s - t.b) * rb ∧
    t.s * r = (t.s - t.c) * rc

/-- The radii form a geometric progression -/
def radii_in_geometric_progression (t : Triangle) (tr : TriangleRadii t) : Prop :=
  ∃ q : ℝ, q > 1 ∧ tr.ra = q * tr.r ∧ tr.rb = q^2 * tr.r ∧ tr.rc = q^3 * tr.r

/-- The largest angle in a triangle is 90 degrees -/
def largest_angle_is_right (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.b^2 + t.c^2 = t.a^2 ∨ t.c^2 + t.a^2 = t.b^2

theorem largest_angle_right_in_special_triangle (t : Triangle) (tr : TriangleRadii t) 
  (h : radii_in_geometric_progression t tr) : 
  largest_angle_is_right t :=
sorry

end NUMINAMATH_CALUDE_largest_angle_right_in_special_triangle_l2554_255481


namespace NUMINAMATH_CALUDE_complex_expression_equality_l2554_255466

theorem complex_expression_equality : 
  let x := (11 + 6 * Real.sqrt 2) * Real.sqrt (11 - 6 * Real.sqrt 2) - 
           (11 - 6 * Real.sqrt 2) * Real.sqrt (11 + 6 * Real.sqrt 2)
  let y := Real.sqrt (Real.sqrt 5 + 2) + Real.sqrt (Real.sqrt 5 - 2) - 
           Real.sqrt (Real.sqrt 5 + 1)
  x / y = 28 + 14 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l2554_255466


namespace NUMINAMATH_CALUDE_rihannas_initial_money_l2554_255401

/-- Represents the shopping scenario and calculates Rihanna's initial money --/
def rihannas_shopping (mango_price apple_juice_price : ℕ) (mango_quantity apple_juice_quantity : ℕ) (money_left : ℕ) : ℕ :=
  let total_cost := mango_price * mango_quantity + apple_juice_price * apple_juice_quantity
  total_cost + money_left

/-- Theorem stating that Rihanna's initial money was $50 --/
theorem rihannas_initial_money :
  rihannas_shopping 3 3 6 6 14 = 50 := by
  sorry

end NUMINAMATH_CALUDE_rihannas_initial_money_l2554_255401


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2554_255470

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 20) :
  1 / x + 1 / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2554_255470


namespace NUMINAMATH_CALUDE_triangle_properties_l2554_255453

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : (2 * t.b - t.c) / t.a = Real.cos t.C / Real.cos t.A) 
  (h2 : t.a = Real.sqrt 5) 
  (h3 : (1 / 2) * t.b * t.c * Real.sin t.A = Real.sqrt 3 / 2) :
  t.A = π / 3 ∧ t.a + t.b + t.c = Real.sqrt 5 + Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2554_255453


namespace NUMINAMATH_CALUDE_pi_half_irrational_l2554_255404

theorem pi_half_irrational : Irrational (π / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_pi_half_irrational_l2554_255404


namespace NUMINAMATH_CALUDE_xyz_sum_root_l2554_255490

theorem xyz_sum_root (x y z : ℝ) 
  (eq1 : y + z = 24)
  (eq2 : z + x = 26)
  (eq3 : x + y = 28) :
  Real.sqrt (x * y * z * (x + y + z)) = Real.sqrt 83655 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_root_l2554_255490


namespace NUMINAMATH_CALUDE_angle_D_measure_l2554_255489

/-- Given a geometric figure with angles A, B, C, and D, prove that when 
    m∠A = 50°, m∠B = 35°, and m∠C = 35°, then m∠D = 120°. -/
theorem angle_D_measure (A B C D : Real) 
    (hA : A = 50) 
    (hB : B = 35)
    (hC : C = 35) : 
  D = 120 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l2554_255489


namespace NUMINAMATH_CALUDE_howlers_lineup_count_l2554_255498

def total_players : Nat := 15
def lineup_size : Nat := 6
def excluded_players : Nat := 3

theorem howlers_lineup_count :
  (Nat.choose (total_players - excluded_players) lineup_size) +
  (excluded_players * Nat.choose (total_players - excluded_players) (lineup_size - 1)) = 3300 :=
by sorry

end NUMINAMATH_CALUDE_howlers_lineup_count_l2554_255498


namespace NUMINAMATH_CALUDE_at_least_three_same_purchase_l2554_255428

/-- Represents a purchase combination of items -/
structure Purchase where
  threeYuanItems : Nat
  fiveYuanItems : Nat
  deriving Repr

/-- The set of all valid purchase combinations -/
def validPurchases : Finset Purchase :=
  sorry

/-- The number of valid purchase combinations -/
def numCombinations : Nat :=
  Finset.card validPurchases

theorem at_least_three_same_purchase (n : Nat) (h : n = 25) :
  ∀ (purchases : Fin n → Purchase),
    (∀ i, purchases i ∈ validPurchases) →
    ∃ (p : Purchase) (i j k : Fin n),
      i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
      purchases i = p ∧ purchases j = p ∧ purchases k = p :=
  sorry

end NUMINAMATH_CALUDE_at_least_three_same_purchase_l2554_255428


namespace NUMINAMATH_CALUDE_part_one_part_two_l2554_255488

-- Define propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

-- Part 1
theorem part_one :
  ∀ x : ℝ, (1 < x ∧ x < 3) ∧ (2 < x ∧ x ≤ 3) ↔ (2 < x ∧ x < 3) :=
sorry

-- Part 2
theorem part_two :
  ∀ a : ℝ, a > 0 →
  ((∀ x : ℝ, 2 < x ∧ x ≤ 3 → a < x ∧ x < 3*a) ∧
   (∃ x : ℝ, a < x ∧ x < 3*a ∧ ¬(2 < x ∧ x ≤ 3))) →
  (1 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2554_255488


namespace NUMINAMATH_CALUDE_go_pieces_probability_l2554_255459

theorem go_pieces_probability (p_black p_white : ℝ) 
  (h_black : p_black = 1/7)
  (h_white : p_white = 12/35) :
  p_black + p_white = 17/35 := by
  sorry

end NUMINAMATH_CALUDE_go_pieces_probability_l2554_255459


namespace NUMINAMATH_CALUDE_inequality_solution_l2554_255420

theorem inequality_solution (x : ℝ) : x^2 - 3*x - 10 < 0 ∧ x > 1 → 1 < x ∧ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2554_255420


namespace NUMINAMATH_CALUDE_solve_for_a_l2554_255464

theorem solve_for_a : ∀ a : ℝ, (a * 1 - (-3) = 1) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2554_255464


namespace NUMINAMATH_CALUDE_smallest_special_number_l2554_255416

def is_special (n : ℕ) : Prop :=
  (n > 3429) ∧ (∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    n = a * 1000 + b * 100 + c * 10 + d)

theorem smallest_special_number :
  ∀ m : ℕ, is_special m → m ≥ 3450 :=
by sorry

end NUMINAMATH_CALUDE_smallest_special_number_l2554_255416


namespace NUMINAMATH_CALUDE_arithmetic_sequence_b_formula_l2554_255499

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def b (a : ℕ → ℝ) (n : ℕ) : ℝ := a (3^n)

theorem arithmetic_sequence_b_formula (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 = 8 →
  a 8 = 26 →
  ∀ n : ℕ, b a n = 3^(n+1) + 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_b_formula_l2554_255499


namespace NUMINAMATH_CALUDE_subset_implies_a_nonpositive_l2554_255455

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x + a ≤ 0}
def B : Set ℝ := {x : ℝ | x^2 - 3*x + 2 ≤ 0}

-- Theorem statement
theorem subset_implies_a_nonpositive (a : ℝ) (h : B ⊆ A a) : a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_nonpositive_l2554_255455


namespace NUMINAMATH_CALUDE_corn_amount_approx_l2554_255486

/-- The cost of corn per pound -/
def corn_cost : ℝ := 1.05

/-- The cost of beans per pound -/
def bean_cost : ℝ := 0.39

/-- The total pounds of corn and beans bought -/
def total_pounds : ℝ := 30

/-- The total cost of the purchase -/
def total_cost : ℝ := 23.10

/-- The amount of corn bought (in pounds) -/
noncomputable def corn_amount : ℝ := 
  (total_cost - bean_cost * total_pounds) / (corn_cost - bean_cost)

theorem corn_amount_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |corn_amount - 17.3| < ε :=
sorry

end NUMINAMATH_CALUDE_corn_amount_approx_l2554_255486


namespace NUMINAMATH_CALUDE_snowball_theorem_l2554_255418

def snowball_distribution (lucy_snowballs : ℕ) (charlie_extra : ℕ) : Prop :=
  let charlie_initial := lucy_snowballs + charlie_extra
  let linus_received := charlie_initial / 2
  let charlie_final := charlie_initial / 2
  let sally_received := linus_received / 3
  let linus_final := linus_received - sally_received
  charlie_final = 25 ∧ lucy_snowballs = 19 ∧ linus_final = 17 ∧ sally_received = 8

theorem snowball_theorem : snowball_distribution 19 31 := by
  sorry

end NUMINAMATH_CALUDE_snowball_theorem_l2554_255418


namespace NUMINAMATH_CALUDE_equation_solutions_l2554_255479

theorem equation_solutions : 
  let f (x : ℝ) := (18*x - x^2) / (x + 2) * (x + (18 - x) / (x + 2))
  ∀ x : ℝ, f x = 56 ↔ x = 4 ∨ x = -14/17 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2554_255479


namespace NUMINAMATH_CALUDE_expression_value_l2554_255454

theorem expression_value : ∃ x : ℕ, (8000 * 6000 : ℕ) = 480 * x ∧ x = 100000 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2554_255454


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l2554_255431

/-- Given a geometric sequence {a_n} with a₁ = 1 and a₄ = 8, prove that a₇ = 64 -/
theorem geometric_sequence_seventh_term (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 1 →                                -- First term condition
  a 4 = 8 →                                -- Fourth term condition
  a 7 = 64 :=                              -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l2554_255431


namespace NUMINAMATH_CALUDE_excess_amount_l2554_255456

theorem excess_amount (x : ℝ) (a : ℝ) : 
  x = 0.16 * x + a → x = 50 → a = 42 := by
  sorry

end NUMINAMATH_CALUDE_excess_amount_l2554_255456


namespace NUMINAMATH_CALUDE_probability_one_genuine_one_defective_l2554_255447

/-- The probability of selecting exactly one genuine product and one defective product
    when randomly selecting two products from a set of 5 genuine products and 1 defective product. -/
theorem probability_one_genuine_one_defective :
  let total_products : ℕ := 5 + 1
  let genuine_products : ℕ := 5
  let defective_products : ℕ := 1
  let total_selections : ℕ := Nat.choose total_products 2
  let favorable_selections : ℕ := genuine_products * defective_products
  (favorable_selections : ℚ) / total_selections = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_genuine_one_defective_l2554_255447


namespace NUMINAMATH_CALUDE_seven_fourth_mod_hundred_l2554_255443

theorem seven_fourth_mod_hundred : 7^4 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_seven_fourth_mod_hundred_l2554_255443


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l2554_255438

/-- The length of the major axis of the ellipse x^2 + 4y^2 = 100 is 20 -/
theorem ellipse_major_axis_length :
  let ellipse := {(x, y) : ℝ × ℝ | x^2 + 4*y^2 = 100}
  ∃ a b : ℝ, a > b ∧ b > 0 ∧
    (∀ (x y : ℝ), (x, y) ∈ ellipse ↔ x^2/a^2 + y^2/b^2 = 1) ∧
    2*a = 20 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l2554_255438


namespace NUMINAMATH_CALUDE_athlete_track_arrangements_l2554_255435

/-- The number of ways to arrange 5 athletes on 5 tracks with exactly two in their numbered tracks -/
def athleteArrangements : ℕ := 20

/-- The number of ways to choose 2 items from a set of 5 -/
def choose5_2 : ℕ := 10

/-- The number of derangements of 3 objects -/
def derangement3 : ℕ := 2

theorem athlete_track_arrangements :
  athleteArrangements = choose5_2 * derangement3 :=
sorry

end NUMINAMATH_CALUDE_athlete_track_arrangements_l2554_255435


namespace NUMINAMATH_CALUDE_Tricia_age_is_5_l2554_255448

-- Define the ages as natural numbers
def Vincent_age : ℕ := 22
def Rupert_age : ℕ := Vincent_age - 2
def Khloe_age : ℕ := Rupert_age - 10
def Eugene_age : ℕ := Khloe_age * 3
def Yorick_age : ℕ := Eugene_age * 2
def Amilia_age : ℕ := Yorick_age / 4
def Tricia_age : ℕ := Amilia_age / 3

-- Theorem statement
theorem Tricia_age_is_5 : Tricia_age = 5 := by
  sorry

end NUMINAMATH_CALUDE_Tricia_age_is_5_l2554_255448


namespace NUMINAMATH_CALUDE_point_not_on_transformed_plane_l2554_255452

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies similarity transformation to a plane -/
def transformPlane (p : Plane) (k : ℝ) : Plane :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point lies on a plane -/
def pointOnPlane (point : Point3D) (plane : Plane) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- The main theorem -/
theorem point_not_on_transformed_plane :
  let originalPlane : Plane := { a := 1, b := -1, c := -1, d := -1 }
  let k : ℝ := 4
  let transformedPlane := transformPlane originalPlane k
  let point : Point3D := { x := 7, y := 0, z := -1 }
  ¬(pointOnPlane point transformedPlane) := by
  sorry


end NUMINAMATH_CALUDE_point_not_on_transformed_plane_l2554_255452


namespace NUMINAMATH_CALUDE_smallest_b_for_nonprime_cubic_l2554_255467

theorem smallest_b_for_nonprime_cubic (x : ℤ) : ∃ (b : ℕ+), ∀ (x : ℤ), ¬ Prime (x^3 + b^2) ∧ ∀ (k : ℕ+), k < b → ∃ (y : ℤ), Prime (y^3 + k^2) :=
sorry

end NUMINAMATH_CALUDE_smallest_b_for_nonprime_cubic_l2554_255467


namespace NUMINAMATH_CALUDE_sector_area_l2554_255444

/-- The area of a circular sector given its arc length and radius -/
theorem sector_area (l r : ℝ) (hl : l > 0) (hr : r > 0) : 
  (l * r) / 2 = (l * r) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2554_255444


namespace NUMINAMATH_CALUDE_late_fee_is_150_l2554_255437

/-- Calculates the late fee for electricity payment -/
def calculate_late_fee (cost_per_watt : ℝ) (watts_used : ℝ) (total_paid : ℝ) : ℝ :=
  total_paid - cost_per_watt * watts_used

/-- Proves that the late fee is $150 given the problem conditions -/
theorem late_fee_is_150 :
  let cost_per_watt : ℝ := 4
  let watts_used : ℝ := 300
  let total_paid : ℝ := 1350
  calculate_late_fee cost_per_watt watts_used total_paid = 150 := by
  sorry

end NUMINAMATH_CALUDE_late_fee_is_150_l2554_255437


namespace NUMINAMATH_CALUDE_crayons_given_to_friends_l2554_255409

theorem crayons_given_to_friends (crayons_lost : ℕ) (total_crayons_lost_or_given : ℕ) 
  (h1 : crayons_lost = 535)
  (h2 : total_crayons_lost_or_given = 587) :
  total_crayons_lost_or_given - crayons_lost = 52 := by
  sorry

end NUMINAMATH_CALUDE_crayons_given_to_friends_l2554_255409


namespace NUMINAMATH_CALUDE_subscription_difference_l2554_255465

/-- Represents the subscription amounts and profit distribution for a business venture. -/
structure BusinessVenture where
  total_subscription : ℕ
  total_profit : ℕ
  a_profit : ℕ
  b_subscription : ℕ
  c_subscription : ℕ

/-- Theorem stating the difference between b's and c's subscriptions given the problem conditions. -/
theorem subscription_difference (bv : BusinessVenture) : 
  bv.total_subscription = 50000 ∧
  bv.total_profit = 36000 ∧
  bv.a_profit = 15120 ∧
  bv.b_subscription + 4000 + bv.b_subscription + bv.c_subscription = bv.total_subscription ∧
  bv.a_profit * bv.total_subscription = bv.total_profit * (bv.b_subscription + 4000) →
  bv.b_subscription - bv.c_subscription = 5000 := by
  sorry

#check subscription_difference

end NUMINAMATH_CALUDE_subscription_difference_l2554_255465


namespace NUMINAMATH_CALUDE_tank_length_proof_l2554_255491

/-- Proves that the length of a rectangular tank is 3 feet given specific conditions -/
theorem tank_length_proof (l : ℝ) : 
  let w : ℝ := 6
  let h : ℝ := 2
  let cost_per_sqft : ℝ := 20
  let total_cost : ℝ := 1440
  let surface_area : ℝ := 2 * l * w + 2 * l * h + 2 * w * h
  total_cost = cost_per_sqft * surface_area → l = 3 := by
  sorry

end NUMINAMATH_CALUDE_tank_length_proof_l2554_255491


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_endpoint_coordinate_sum_proof_l2554_255462

/-- Given a line segment with one endpoint (6,4) and midpoint (3,10),
    the sum of the coordinates of the other endpoint is 16. -/
theorem endpoint_coordinate_sum : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → Prop :=
  fun endpoint1 midpoint endpoint2 =>
    endpoint1 = (6, 4) ∧
    midpoint = (3, 10) ∧
    midpoint = ((endpoint1.1 + endpoint2.1) / 2, (endpoint1.2 + endpoint2.2) / 2) →
    endpoint2.1 + endpoint2.2 = 16

/-- Proof of the theorem -/
theorem endpoint_coordinate_sum_proof : ∃ (endpoint2 : ℝ × ℝ),
  endpoint_coordinate_sum (6, 4) (3, 10) endpoint2 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_endpoint_coordinate_sum_proof_l2554_255462


namespace NUMINAMATH_CALUDE_inscribed_box_sphere_radius_l2554_255432

theorem inscribed_box_sphere_radius (a b c s : ℝ) : 
  a > 0 → b > 0 → c > 0 → s > 0 →
  (a + b + c = 18) →
  (2 * a * b + 2 * b * c + 2 * a * c = 216) →
  (4 * s^2 = a^2 + b^2 + c^2) →
  s = Real.sqrt 27 := by
sorry

end NUMINAMATH_CALUDE_inscribed_box_sphere_radius_l2554_255432


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l2554_255406

/-- A rectangle with perimeter 60 meters and area 221 square meters has dimensions 17 meters and 13 meters. -/
theorem rectangle_dimensions (l w : ℝ) (h_perimeter : 2 * l + 2 * w = 60) (h_area : l * w = 221) :
  (l = 17 ∧ w = 13) ∨ (l = 13 ∧ w = 17) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l2554_255406


namespace NUMINAMATH_CALUDE_gcd_1734_816_l2554_255463

theorem gcd_1734_816 : Nat.gcd 1734 816 = 102 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1734_816_l2554_255463


namespace NUMINAMATH_CALUDE_find_m_l2554_255414

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define the set A
def A (m : ℕ) : Set ℕ := {x ∈ U | x^2 - 5*x + m = 0}

-- Define the complement of A in U
def complement_A (m : ℕ) : Set ℕ := U \ A m

-- Theorem statement
theorem find_m : ∃ m : ℕ, complement_A m = {2, 3} ∧ m = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l2554_255414


namespace NUMINAMATH_CALUDE_system_solution_ratio_l2554_255424

theorem system_solution_ratio (x y c d : ℝ) (h1 : 4 * x - 2 * y = c)
    (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) : c / d = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l2554_255424


namespace NUMINAMATH_CALUDE_f_at_negative_four_l2554_255408

/-- The polynomial f(x) = 12 + 35x − 8x^2 + 79x^3 + 6x^4 + 5x^5 + 3x^6 -/
def f (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

/-- Theorem: The value of f(-4) is 3392 -/
theorem f_at_negative_four : f (-4) = 3392 := by
  sorry

end NUMINAMATH_CALUDE_f_at_negative_four_l2554_255408


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_l2554_255445

def f (x : ℝ) : ℝ := -x^2 + 4*x - 3

theorem f_monotone_decreasing : 
  MonotoneOn f (Set.Ici 2) := by sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_l2554_255445


namespace NUMINAMATH_CALUDE_select_team_count_l2554_255436

/-- The number of ways to select a team of 8 members (4 boys and 4 girls) from a group of 10 boys and 12 girls -/
def selectTeam (totalBoys : ℕ) (totalGirls : ℕ) (teamSize : ℕ) (boysInTeam : ℕ) (girlsInTeam : ℕ) : ℕ :=
  Nat.choose totalBoys boysInTeam * Nat.choose totalGirls girlsInTeam

/-- Theorem stating that the number of ways to select the team is 103950 -/
theorem select_team_count :
  selectTeam 10 12 8 4 4 = 103950 := by
  sorry

end NUMINAMATH_CALUDE_select_team_count_l2554_255436


namespace NUMINAMATH_CALUDE_triangle_altitude_reciprocal_sum_bounds_l2554_255485

/-- For any triangle, the sum of the reciprocals of two altitudes lies between the reciprocal of the radius of the inscribed circle and the reciprocal of its diameter. -/
theorem triangle_altitude_reciprocal_sum_bounds (a b c m_a m_b m_c ρ s t : ℝ) 
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_altitudes : m_a > 0 ∧ m_b > 0 ∧ m_c > 0)
  (h_perimeter : a + b + c = 2 * s)
  (h_area : t > 0)
  (h_inscribed_radius : ρ > 0)
  (h_altitude_a : a * m_a = 2 * t)
  (h_altitude_b : b * m_b = 2 * t)
  (h_altitude_c : c * m_c = 2 * t)
  (h_inscribed_radius_def : s * ρ = t) :
  1 / (2 * ρ) < 1 / m_a + 1 / m_b ∧ 1 / m_a + 1 / m_b < 1 / ρ :=
by sorry

end NUMINAMATH_CALUDE_triangle_altitude_reciprocal_sum_bounds_l2554_255485


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_slope_product_l2554_255468

/-- An ellipse passing through (2,0) with eccentricity √3/2 -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_eq : a^2 = 4
  h_ecc : (a^2 - b^2) / a^2 = 3/4

/-- A line passing through (1,0) with non-zero slope -/
structure Line where
  k : ℝ
  h_k_nonzero : k ≠ 0

/-- The theorem statement -/
theorem ellipse_line_intersection_slope_product (C : Ellipse) (l : Line) :
  ∃ k' : ℝ, l.k * k' = -1/4 := by sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_slope_product_l2554_255468


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2554_255478

theorem complex_equation_solution (a : ℝ) : (a - Complex.I)^2 = 2 * Complex.I → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2554_255478


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l2554_255487

theorem unique_quadratic_solution :
  ∃! (q : ℝ), q ≠ 0 ∧ (∃! x, q * x^2 - 8 * x + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l2554_255487


namespace NUMINAMATH_CALUDE_inverse_function_value_l2554_255415

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 4*x

-- Define the domain of f
def domain (x : ℝ) : Prop := x < -2

-- Define the inverse function property
def is_inverse (f : ℝ → ℝ) (f_inv : ℝ → ℝ) : Prop :=
  ∀ x, domain x → f (f_inv (f x)) = f x ∧ f_inv (f x) = x

-- Theorem statement
theorem inverse_function_value :
  ∃ f_inv : ℝ → ℝ, is_inverse f f_inv ∧ f_inv 12 = -6 :=
sorry

end NUMINAMATH_CALUDE_inverse_function_value_l2554_255415
