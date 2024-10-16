import Mathlib

namespace NUMINAMATH_CALUDE_min_k_for_inequality_l2317_231792

theorem min_k_for_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ k : ℝ, (1 / a + 1 / b + k / (a + b) ≥ 0) → k ≥ -4) ∧
  (∃ k : ℝ, k = -4 ∧ 1 / a + 1 / b + k / (a + b) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_min_k_for_inequality_l2317_231792


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l2317_231725

theorem smallest_number_with_remainders : ∃ N : ℕ, 
  N > 0 ∧ 
  N % 13 = 2 ∧ 
  N % 15 = 4 ∧ 
  (∀ M : ℕ, M > 0 → M % 13 = 2 → M % 15 = 4 → N ≤ M) ∧
  N = 184 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l2317_231725


namespace NUMINAMATH_CALUDE_intersection_point_of_function_and_inverse_l2317_231784

theorem intersection_point_of_function_and_inverse
  (b a : ℤ) (g : ℝ → ℝ) (g_inv : ℝ → ℝ)
  (h1 : ∀ x, g x = 4 * x + b)
  (h2 : Function.LeftInverse g_inv g ∧ Function.RightInverse g_inv g)
  (h3 : g (-4) = a)
  (h4 : g_inv (-4) = a) :
  a = -4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_function_and_inverse_l2317_231784


namespace NUMINAMATH_CALUDE_scissors_freedom_theorem_l2317_231785

/-- Represents the state of the rope and scissors system -/
structure RopeScissorsState where
  loopThroughScissors : Bool
  ropeEndsFixed : Bool
  noKnotsUntied : Bool

/-- Represents a single manipulation of the rope -/
inductive RopeManipulation
  | PullLoop
  | PassLoopAroundEnds
  | ReverseDirection

/-- Defines a sequence of rope manipulations -/
def ManipulationSequence := List RopeManipulation

/-- Predicate to check if a manipulation sequence frees the scissors -/
def freesScissors (seq : ManipulationSequence) : Prop := sorry

/-- The main theorem stating that there exists a sequence of manipulations that frees the scissors -/
theorem scissors_freedom_theorem (initialState : RopeScissorsState) 
  (h1 : initialState.loopThroughScissors = true)
  (h2 : initialState.ropeEndsFixed = true)
  (h3 : initialState.noKnotsUntied = true) :
  ∃ (seq : ManipulationSequence), freesScissors seq := by
  sorry


end NUMINAMATH_CALUDE_scissors_freedom_theorem_l2317_231785


namespace NUMINAMATH_CALUDE_curve_to_line_equation_l2317_231711

/-- Given a curve parameterized by (x, y) = (3t + 6, 5t - 7), where t is a real number,
    prove that the equation of the line in the form y = mx + b is y = (5/3)x - 17. -/
theorem curve_to_line_equation :
  ∀ (t x y : ℝ), x = 3 * t + 6 ∧ y = 5 * t - 7 →
  ∃ (m b : ℝ), m = 5 / 3 ∧ b = -17 ∧ y = m * x + b :=
by sorry

end NUMINAMATH_CALUDE_curve_to_line_equation_l2317_231711


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l2317_231770

-- Define the diamond operation
def diamond (A B : ℝ) : ℝ := 4 * A + 3 * B * 2

-- Theorem statement
theorem diamond_equation_solution :
  ∃! A : ℝ, diamond A 5 = 64 ∧ A = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l2317_231770


namespace NUMINAMATH_CALUDE_f_at_5_l2317_231746

def f (x : ℝ) : ℝ := x^5 - 2*x^4 + x^3 + x^2 - x - 5

theorem f_at_5 : f 5 = 2015 := by
  sorry

end NUMINAMATH_CALUDE_f_at_5_l2317_231746


namespace NUMINAMATH_CALUDE_quadratic_diophantine_equation_solution_l2317_231797

theorem quadratic_diophantine_equation_solution 
  (a b c : ℕ+) 
  (h : (a * c : ℕ) = b^2 + b + 1) : 
  ∃ (x y : ℤ), (a : ℤ) * x^2 - (2 * (b : ℤ) + 1) * x * y + (c : ℤ) * y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_diophantine_equation_solution_l2317_231797


namespace NUMINAMATH_CALUDE_sequence_sum_property_l2317_231796

def sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

theorem sequence_sum_property (a : ℕ → ℝ) :
  (∀ n : ℕ, n > 0 → (sequence_sum a n - 1)^2 = a n * sequence_sum a n) →
  (∀ n : ℕ, n > 0 → sequence_sum a n = n / (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_property_l2317_231796


namespace NUMINAMATH_CALUDE_basketball_cricket_students_l2317_231729

theorem basketball_cricket_students (B C : Finset Nat) : 
  (B.card = 7) → 
  (C.card = 8) → 
  ((B ∩ C).card = 5) → 
  ((B ∪ C).card = 10) :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_cricket_students_l2317_231729


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2317_231750

/-- An isosceles triangle with side lengths 4 and 8 has a perimeter of 20 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ), 
  a = 8 ∧ b = 8 ∧ c = 4 → -- Two sides are 8, one side is 4
  a + b > c ∧ b + c > a ∧ c + a > b → -- Triangle inequality
  a + b + c = 20 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2317_231750


namespace NUMINAMATH_CALUDE_total_matches_is_seventeen_l2317_231767

/-- Calculates the number of matches in a round-robin tournament for n teams -/
def roundRobinMatches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Represents a football competition with the given structure -/
structure FootballCompetition where
  totalTeams : ℕ
  groupSize : ℕ
  numGroups : ℕ
  semiFinalistPerGroup : ℕ
  semiFinalsLegs : ℕ
  finalMatches : ℕ

/-- Calculates the total number of matches in the competition -/
def totalMatches (comp : FootballCompetition) : ℕ :=
  (comp.numGroups * roundRobinMatches comp.groupSize) +
  (comp.numGroups * comp.semiFinalistPerGroup * comp.semiFinalsLegs / 2) +
  comp.finalMatches

/-- The specific football competition described in the problem -/
def specificCompetition : FootballCompetition :=
  { totalTeams := 8
  , groupSize := 4
  , numGroups := 2
  , semiFinalistPerGroup := 2
  , semiFinalsLegs := 2
  , finalMatches := 1 }

theorem total_matches_is_seventeen :
  totalMatches specificCompetition = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_matches_is_seventeen_l2317_231767


namespace NUMINAMATH_CALUDE_sock_selection_theorem_l2317_231732

def total_socks : ℕ := 7
def blue_socks : ℕ := 2
def other_socks : ℕ := 5
def socks_to_choose : ℕ := 4

def valid_combinations : ℕ := 30

theorem sock_selection_theorem :
  (Nat.choose blue_socks 2 * Nat.choose other_socks 2) +
  (Nat.choose blue_socks 2 * Nat.choose other_socks 1) +
  (Nat.choose blue_socks 1 * Nat.choose other_socks 2) = valid_combinations :=
by sorry

end NUMINAMATH_CALUDE_sock_selection_theorem_l2317_231732


namespace NUMINAMATH_CALUDE_boat_license_plates_l2317_231776

def letter_choices : ℕ := 3
def digit_choices : ℕ := 10
def num_digits : ℕ := 4

theorem boat_license_plates :
  letter_choices * digit_choices^num_digits = 30000 :=
sorry

end NUMINAMATH_CALUDE_boat_license_plates_l2317_231776


namespace NUMINAMATH_CALUDE_min_value_sqrt_plus_reciprocal_l2317_231704

theorem min_value_sqrt_plus_reciprocal (x : ℝ) (h : x > 0) :
  3 * Real.sqrt x + 2 / x ≥ 5 ∧
  ∃ y > 0, 3 * Real.sqrt y + 2 / y = 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_sqrt_plus_reciprocal_l2317_231704


namespace NUMINAMATH_CALUDE_polar_line_properties_l2317_231772

/-- A line in polar coordinates passing through (4,0) and perpendicular to the polar axis -/
def polar_line (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 4

theorem polar_line_properties (ρ θ : ℝ) :
  polar_line ρ θ →
  (ρ * Real.cos θ = 4 ∧ ρ * Real.sin θ = 0) ∧
  (∀ (x y : ℝ), x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ → x = 4) :=
sorry

end NUMINAMATH_CALUDE_polar_line_properties_l2317_231772


namespace NUMINAMATH_CALUDE_no_14_consecutive_divisible_by_primes_lt_13_exist_21_consecutive_divisible_by_primes_lt_17_l2317_231741

/-- The set of primes less than 13 -/
def primes_lt_13 : Set Nat := {p | Nat.Prime p ∧ p < 13}

/-- The set of primes less than 17 -/
def primes_lt_17 : Set Nat := {p | Nat.Prime p ∧ p < 17}

/-- A function that checks if a number is divisible by any prime in a given set -/
def divisible_by_any_prime (n : Nat) (primes : Set Nat) : Prop :=
  ∃ p ∈ primes, n % p = 0

/-- Theorem stating that there do not exist 14 consecutive positive integers
    each divisible by a prime less than 13 -/
theorem no_14_consecutive_divisible_by_primes_lt_13 :
  ¬ ∃ start : Nat, ∀ i ∈ Finset.range 14, divisible_by_any_prime (start + i) primes_lt_13 :=
sorry

/-- Theorem stating that there exist 21 consecutive positive integers
    each divisible by a prime less than 17 -/
theorem exist_21_consecutive_divisible_by_primes_lt_17 :
  ∃ start : Nat, ∀ i ∈ Finset.range 21, divisible_by_any_prime (start + i) primes_lt_17 :=
sorry

end NUMINAMATH_CALUDE_no_14_consecutive_divisible_by_primes_lt_13_exist_21_consecutive_divisible_by_primes_lt_17_l2317_231741


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2317_231789

/-- A geometric sequence with its sum function -/
structure GeometricSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)
  sum_property : ∀ n : ℕ, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))

/-- The main theorem -/
theorem geometric_sequence_property (seq : GeometricSequence) 
  (h1 : seq.S 4 = 1) (h2 : seq.S 8 = 3) : 
  seq.a 17 + seq.a 18 + seq.a 19 + seq.a 20 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2317_231789


namespace NUMINAMATH_CALUDE_pants_cost_l2317_231718

theorem pants_cost (total_spent shirt_cost tie_cost : ℕ) 
  (h1 : total_spent = 198)
  (h2 : shirt_cost = 43)
  (h3 : tie_cost = 15) : 
  total_spent - (shirt_cost + tie_cost) = 140 := by
  sorry

end NUMINAMATH_CALUDE_pants_cost_l2317_231718


namespace NUMINAMATH_CALUDE_solve_for_p_l2317_231723

theorem solve_for_p (p q : ℚ) 
  (eq1 : 5 * p - 2 * q = 14) 
  (eq2 : 6 * p + q = 31) : 
  p = 76 / 17 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_p_l2317_231723


namespace NUMINAMATH_CALUDE_symmetry_shift_l2317_231769

/-- Given a function f(x) = √3 cos x - sin x, this theorem states that
    the smallest positive value of θ such that the graph of f(x-θ) is
    symmetrical about the line x = π/6 is π/3. -/
theorem symmetry_shift (f : ℝ → ℝ) (h : ∀ x, f x = Real.sqrt 3 * Real.cos x - Real.sin x) :
  ∃ θ : ℝ, θ > 0 ∧
    (∀ θ' > 0, (∀ x, f (x - θ') = f (π / 3 - (x - π / 6))) → θ ≤ θ') ∧
    (∀ x, f (x - θ) = f (π / 3 - (x - π / 6))) ∧
    θ = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_shift_l2317_231769


namespace NUMINAMATH_CALUDE_theater_rows_count_l2317_231744

/-- Represents a theater with a specific seating arrangement -/
structure Theater where
  total_seats : ℕ
  num_rows : ℕ
  first_row_seats : ℕ

/-- Calculates the total number of seats in the theater based on the seating arrangement -/
def total_seats_calc (t : Theater) : ℕ :=
  (t.first_row_seats + (t.first_row_seats + t.num_rows - 1)) * t.num_rows / 2

/-- Theorem stating that a theater with 1000 seats and the given seating arrangement has 25 rows -/
theorem theater_rows_count (t : Theater) 
  (h1 : t.total_seats = 1000)
  (h2 : t.num_rows > 16)
  (h3 : total_seats_calc t = t.total_seats) : 
  t.num_rows = 25 := by
  sorry

#check theater_rows_count

end NUMINAMATH_CALUDE_theater_rows_count_l2317_231744


namespace NUMINAMATH_CALUDE_probability_x_greater_9y_l2317_231743

/-- The probability that a randomly chosen point (x,y) from a rectangle
    with vertices (0,0), (2017,0), (2017,2018), and (0,2018) satisfies x > 9y -/
theorem probability_x_greater_9y : ℝ := by
  -- Define the rectangle
  let rectangle_width : ℝ := 2017
  let rectangle_height : ℝ := 2018
  
  -- Define the area of the rectangle
  let rectangle_area : ℝ := rectangle_width * rectangle_height
  
  -- Define the area of the region where x > 9y
  let region_area : ℝ := (rectangle_width^2) / 18
  
  -- Calculate the probability
  let probability : ℝ := region_area / rectangle_area
  
  -- Prove that the probability is equal to 2017/36324
  sorry

#eval (2017 : ℚ) / 36324

end NUMINAMATH_CALUDE_probability_x_greater_9y_l2317_231743


namespace NUMINAMATH_CALUDE_digit_equation_sum_l2317_231713

theorem digit_equation_sum (A B C D U : ℕ) : 
  (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ U) ∧
  (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ U) ∧
  (C ≠ D) ∧ (C ≠ U) ∧
  (D ≠ U) ∧
  (A < 10) ∧ (B < 10) ∧ (C < 10) ∧ (D < 10) ∧ (U < 10) ∧ (U > 0) ∧
  ((10 * A + B) * (10 * C + D) = 111 * U) →
  A + B + C + D + U = 17 := by
sorry

end NUMINAMATH_CALUDE_digit_equation_sum_l2317_231713


namespace NUMINAMATH_CALUDE_quadratic_solution_l2317_231779

/-- The quadratic equation ax^2 + 10x + c = 0 has exactly one solution, a + c = 12, and a < c -/
def quadratic_equation (a c : ℝ) : Prop :=
  ∃! x, a * x^2 + 10 * x + c = 0 ∧ a + c = 12 ∧ a < c

/-- The solution to the quadratic equation is (6-√11, 6+√11) -/
theorem quadratic_solution :
  ∀ a c : ℝ, quadratic_equation a c → a = 6 - Real.sqrt 11 ∧ c = 6 + Real.sqrt 11 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_l2317_231779


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2317_231794

theorem arithmetic_mean_problem (a b c : ℝ) :
  let numbers := [a, b, c, 108]
  (numbers.sum / numbers.length = 92) →
  ((a + b + c) / 3 = 260 / 3) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2317_231794


namespace NUMINAMATH_CALUDE_polynomial_sum_l2317_231735

-- Define the polynomials
def p (x : ℝ) : ℝ := -2 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

-- State the theorem
theorem polynomial_sum (x : ℝ) : p x + q x + r x = 12 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l2317_231735


namespace NUMINAMATH_CALUDE_bob_initial_pennies_l2317_231724

theorem bob_initial_pennies :
  ∀ (a b : ℕ),
  (b + 2 = 4 * (a - 2)) →
  (b - 2 = 3 * (a + 2)) →
  b = 62 := by
  sorry

end NUMINAMATH_CALUDE_bob_initial_pennies_l2317_231724


namespace NUMINAMATH_CALUDE_branch_fraction_l2317_231701

theorem branch_fraction (L : ℝ) (F : ℝ) : 
  L = 3 →  -- The branch length is 3 meters
  0 < F → F < 1 →  -- F is a proper fraction
  L - (L / 3 + F * L) = 0.6 * L →  -- Remaining length after removal
  F = 1 / 15 := by
sorry

end NUMINAMATH_CALUDE_branch_fraction_l2317_231701


namespace NUMINAMATH_CALUDE_monotonic_increasing_condition_l2317_231720

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x

-- State the theorem
theorem monotonic_increasing_condition (a : ℝ) : 
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, Monotone (fun x => f a x)) → a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_condition_l2317_231720


namespace NUMINAMATH_CALUDE_sum_of_three_integers_l2317_231760

theorem sum_of_three_integers (large medium small : ℕ+) 
  (sum_large_medium : large + medium = 2003)
  (diff_medium_small : medium - small = 1000) :
  large + medium + small = 2004 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_integers_l2317_231760


namespace NUMINAMATH_CALUDE_sum_of_max_min_S_l2317_231787

theorem sum_of_max_min_S (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0)
  (h1 : x + y = 10) (h2 : y + z = 8) : 
  let S := x + z
  ∃ (S_min S_max : ℝ), 
    (∀ s, (∃ x' z', x' ≥ 0 ∧ z' ≥ 0 ∧ ∃ y', y' ≥ 0 ∧ x' + y' = 10 ∧ y' + z' = 8 ∧ s = x' + z') → s ≥ S_min) ∧
    (∀ s, (∃ x' z', x' ≥ 0 ∧ z' ≥ 0 ∧ ∃ y', y' ≥ 0 ∧ x' + y' = 10 ∧ y' + z' = 8 ∧ s = x' + z') → s ≤ S_max) ∧
    S_min + S_max = 20 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_max_min_S_l2317_231787


namespace NUMINAMATH_CALUDE_min_y_value_l2317_231782

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 18*x + 54*y) :
  ∃ (y_min : ℝ), y_min = 27 - Real.sqrt 810 ∧ ∀ (y' : ℝ), ∃ (x' : ℝ), x'^2 + y'^2 = 18*x' + 54*y' → y' ≥ y_min :=
sorry

end NUMINAMATH_CALUDE_min_y_value_l2317_231782


namespace NUMINAMATH_CALUDE_order_of_a_l2317_231709

theorem order_of_a (a : ℝ) (h : a^2 + a < 0) : -a > a^2 ∧ a^2 > -a^2 ∧ -a^2 > a := by
  sorry

end NUMINAMATH_CALUDE_order_of_a_l2317_231709


namespace NUMINAMATH_CALUDE_seating_arrangements_l2317_231765

/-- Represents the number of people sitting around the table -/
def total_people : ℕ := 8

/-- Represents the number of people with specific roles (leader, deputy leader, recorder) -/
def specific_roles : ℕ := 3

/-- Represents the number of ways the recorder can sit between the leader and deputy leader -/
def recorder_arrangements : ℕ := 2

/-- Calculates the factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Theorem stating the number of distinct seating arrangements -/
theorem seating_arrangements :
  (factorial (total_people - specific_roles + 1 - 1)) * recorder_arrangements = 240 := by
  sorry

#eval (factorial (total_people - specific_roles + 1 - 1)) * recorder_arrangements

end NUMINAMATH_CALUDE_seating_arrangements_l2317_231765


namespace NUMINAMATH_CALUDE_one_third_of_twelve_x_plus_five_l2317_231726

theorem one_third_of_twelve_x_plus_five (x : ℚ) : (1 / 3) * (12 * x + 5) = 4 * x + 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_twelve_x_plus_five_l2317_231726


namespace NUMINAMATH_CALUDE_max_value_implies_a_l2317_231721

theorem max_value_implies_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, 2 * a^x - 5 ≤ 10) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 2, 2 * a^x - 5 = 10) →
  a = Real.sqrt (15/2) ∨ a = 15/2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l2317_231721


namespace NUMINAMATH_CALUDE_vector_sum_squared_norms_l2317_231763

theorem vector_sum_squared_norms (a b : ℝ × ℝ) :
  let m : ℝ × ℝ := (4, 10)  -- midpoint
  (∀ (x : ℝ) (y : ℝ), m = ((a.1 + b.1) / 2, (a.2 + b.2) / 2)) →  -- midpoint condition
  (a.1 * b.1 + a.2 * b.2 = 12) →  -- dot product condition
  (a.1^2 + a.2^2) + (b.1^2 + b.2^2) = 440 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_squared_norms_l2317_231763


namespace NUMINAMATH_CALUDE_hyperbola_properties_l2317_231771

/-- Properties of a hyperbola -/
theorem hyperbola_properties (x y : ℝ) :
  (y^2 / 9 - x^2 / 16 = 1) →
  (∃ (imaginary_axis_length : ℝ) (asymptote_slope : ℝ) (focus_y : ℝ) (eccentricity : ℝ),
    imaginary_axis_length = 8 ∧
    asymptote_slope = 3/4 ∧
    focus_y = 5 ∧
    eccentricity = 5/3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l2317_231771


namespace NUMINAMATH_CALUDE_graph_above_condition_l2317_231751

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2|
def g (x m : ℝ) : ℝ := -|x + 3| + m

-- State the theorem
theorem graph_above_condition (m : ℝ) :
  (∀ x : ℝ, f x > g x m) → m < 5 := by
  sorry

end NUMINAMATH_CALUDE_graph_above_condition_l2317_231751


namespace NUMINAMATH_CALUDE_video_game_points_l2317_231722

/-- 
Given a video game level with the following conditions:
- There are 6 enemies in total
- Each defeated enemy gives 3 points
- 2 enemies are not defeated

Prove that the total points earned is 12.
-/
theorem video_game_points (total_enemies : ℕ) (points_per_enemy : ℕ) (undefeated_enemies : ℕ) :
  total_enemies = 6 →
  points_per_enemy = 3 →
  undefeated_enemies = 2 →
  (total_enemies - undefeated_enemies) * points_per_enemy = 12 :=
by sorry

end NUMINAMATH_CALUDE_video_game_points_l2317_231722


namespace NUMINAMATH_CALUDE_inequality_iff_p_in_unit_interval_l2317_231752

/-- The function f(x) = x^2 + ax + b -/
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

/-- The proposition that pf(x) + qf(y) ≥ f(px + qy) for all real x, y -/
def inequality_holds (a b p q : ℝ) : Prop :=
  ∀ x y : ℝ, p * f a b x + q * f a b y ≥ f a b (p*x + q*y)

theorem inequality_iff_p_in_unit_interval (a b : ℝ) :
  ∀ p q : ℝ, p + q = 1 →
    (inequality_holds a b p q ↔ 0 ≤ p ∧ p ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_iff_p_in_unit_interval_l2317_231752


namespace NUMINAMATH_CALUDE_max_value_of_s_l2317_231788

theorem max_value_of_s (p q r s : ℝ) 
  (sum_eq : p + q + r + s = 10)
  (sum_prod_eq : p*q + p*r + p*s + q*r + q*s + r*s = 22) :
  s ≤ (5 + Real.sqrt 93) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_s_l2317_231788


namespace NUMINAMATH_CALUDE_log_relation_l2317_231795

theorem log_relation (a b : ℝ) : 
  a = Real.log 400 / Real.log 16 → b = Real.log 20 / Real.log 2 → a = b / 2 := by
  sorry

end NUMINAMATH_CALUDE_log_relation_l2317_231795


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2317_231734

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 2 + a 10 = 16 → a 4 + a 6 + a 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2317_231734


namespace NUMINAMATH_CALUDE_min_abs_z_plus_2i_l2317_231790

theorem min_abs_z_plus_2i (z : ℂ) (h : Complex.abs (z^2 + 9) = Complex.abs (z * (z + 3*I))) :
  ∃ (w : ℂ), Complex.abs (w + 2*I) = 5/2 ∧ ∀ (z : ℂ), Complex.abs (z^2 + 9) = Complex.abs (z * (z + 3*I)) → Complex.abs (z + 2*I) ≥ 5/2 :=
by sorry

end NUMINAMATH_CALUDE_min_abs_z_plus_2i_l2317_231790


namespace NUMINAMATH_CALUDE_tangent_through_origin_l2317_231777

/-- Given a curve y = x^a + 1 where a is a real number,
    if the tangent line to this curve at the point (1, 2) passes through the origin,
    then a = 2. -/
theorem tangent_through_origin (a : ℝ) : 
  (∀ x y : ℝ, y = x^a + 1) →
  (∃ m b : ℝ, ∀ x y : ℝ, y = m * (x - 1) + 2 ∧ y = m * x + b) →
  (0 = 0 * 0 + b) →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_through_origin_l2317_231777


namespace NUMINAMATH_CALUDE_min_product_under_constraints_l2317_231708

theorem min_product_under_constraints (x y z : ℝ) :
  x > 0 → y > 0 → z > 0 →
  x + y + z = 2 →
  x ≤ 3*y ∧ x ≤ 3*z ∧ y ≤ 3*x ∧ y ≤ 3*z ∧ z ≤ 3*x ∧ z ≤ 3*y →
  x * y * z ≥ 32/81 :=
by sorry

end NUMINAMATH_CALUDE_min_product_under_constraints_l2317_231708


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l2317_231714

theorem ferris_wheel_capacity (num_seats : ℕ) (people_per_seat : ℕ) 
  (h1 : num_seats = 4) (h2 : people_per_seat = 5) :
  num_seats * people_per_seat = 20 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l2317_231714


namespace NUMINAMATH_CALUDE_hayden_ironing_time_l2317_231799

/-- The total time Hayden spends ironing over 4 weeks -/
def total_ironing_time (shirt_time pants_time days_per_week num_weeks : ℕ) : ℕ :=
  (shirt_time + pants_time) * days_per_week * num_weeks

/-- Proof that Hayden spends 160 minutes ironing over 4 weeks -/
theorem hayden_ironing_time :
  total_ironing_time 5 3 5 4 = 160 := by
  sorry

end NUMINAMATH_CALUDE_hayden_ironing_time_l2317_231799


namespace NUMINAMATH_CALUDE_quadratic_trinomial_equality_l2317_231780

/-- A quadratic trinomial function -/
def quadratic_trinomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_trinomial_equality 
  (a b c : ℝ) (h_a : a ≠ 0) :
  (∀ x, quadratic_trinomial a b c (3.8 * x - 1) = quadratic_trinomial a b c (-3.8 * x)) →
  (∀ x, quadratic_trinomial a b c x = quadratic_trinomial a a c x) :=
sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_equality_l2317_231780


namespace NUMINAMATH_CALUDE_johnnys_third_job_rate_l2317_231753

/-- Given Johnny's work schedule and earnings, prove the hourly rate of his third job. -/
theorem johnnys_third_job_rate (hours_job1 hours_job2 hours_job3 : ℕ)
                               (rate_job1 rate_job2 : ℕ)
                               (days : ℕ)
                               (total_earnings : ℕ) :
  hours_job1 = 3 →
  hours_job2 = 2 →
  hours_job3 = 4 →
  rate_job1 = 7 →
  rate_job2 = 10 →
  days = 5 →
  total_earnings = 445 →
  ∃ (rate_job3 : ℕ), 
    rate_job3 = 12 ∧
    total_earnings = (hours_job1 * rate_job1 + hours_job2 * rate_job2 + hours_job3 * rate_job3) * days :=
by sorry

end NUMINAMATH_CALUDE_johnnys_third_job_rate_l2317_231753


namespace NUMINAMATH_CALUDE_prob_six_diff_tens_digits_l2317_231730

/-- The probability of selecting 6 different integers between 10 and 99 (inclusive) 
    with different tens digits -/
def prob_diff_tens_digits : ℚ :=
  8000 / 5895

/-- The number of integers between 10 and 99, inclusive -/
def total_integers : ℕ := 90

/-- The number of possible tens digits -/
def num_tens_digits : ℕ := 9

/-- The number of integers to be selected -/
def num_selected : ℕ := 6

/-- The number of integers for each tens digit -/
def integers_per_tens : ℕ := 10

theorem prob_six_diff_tens_digits :
  prob_diff_tens_digits = 
    (Nat.choose num_tens_digits num_selected * integers_per_tens ^ num_selected) / 
    Nat.choose total_integers num_selected :=
sorry

end NUMINAMATH_CALUDE_prob_six_diff_tens_digits_l2317_231730


namespace NUMINAMATH_CALUDE_faye_coloring_books_l2317_231703

theorem faye_coloring_books :
  ∀ (initial : ℕ), 
    (initial - 3 + 48 = 79) → 
    initial = 34 := by
  sorry

end NUMINAMATH_CALUDE_faye_coloring_books_l2317_231703


namespace NUMINAMATH_CALUDE_expression_simplification_l2317_231727

theorem expression_simplification (a b : ℚ) (ha : a = -2) (hb : b = 3) :
  (((a - b) / (a^2 - 2*a*b + b^2) - a / (a^2 - 2*a*b)) / (b / (a - 2*b))) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2317_231727


namespace NUMINAMATH_CALUDE_square_area_from_two_points_square_area_specific_case_l2317_231740

/-- The area of a square given two points on the same side -/
theorem square_area_from_two_points (x1 y1 x2 y2 : ℝ) (h : x1 = x2) :
  (y1 - y2) ^ 2 = 225 → 
  let side_length := |y1 - y2|
  (side_length ^ 2 : ℝ) = 225 := by
  sorry

/-- The specific case for the given coordinates -/
theorem square_area_specific_case : 
  let x1 : ℝ := 20
  let y1 : ℝ := 20
  let x2 : ℝ := 20
  let y2 : ℝ := 5
  (y1 - y2) ^ 2 = 225 ∧ 
  let side_length := |y1 - y2|
  (side_length ^ 2 : ℝ) = 225 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_two_points_square_area_specific_case_l2317_231740


namespace NUMINAMATH_CALUDE_max_sum_red_green_balls_l2317_231700

theorem max_sum_red_green_balls :
  ∀ (total red green blue : ℕ),
    total = 28 →
    green = 12 →
    red + green + blue = total →
    red ≤ 11 →
    red + green ≤ 23 ∧ ∃ (red' : ℕ), red' ≤ 11 ∧ red' + green = 23 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_red_green_balls_l2317_231700


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l2317_231719

/-- The length of a train given its speed and the time it takes to cross a platform. -/
theorem train_length (platform_length : ℝ) (crossing_time : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (5 / 18)
  let total_distance := train_speed_mps * crossing_time
  total_distance - platform_length

/-- Proof that the train length is approximately 110 meters. -/
theorem train_length_proof :
  ∃ ε > 0, abs (train_length 165 7.499400047996161 132 - 110) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_train_length_train_length_proof_l2317_231719


namespace NUMINAMATH_CALUDE_david_pushups_count_l2317_231798

def zachary_pushups : ℕ := 35

def david_pushups : ℕ := zachary_pushups + 9

theorem david_pushups_count : david_pushups = 44 := by sorry

end NUMINAMATH_CALUDE_david_pushups_count_l2317_231798


namespace NUMINAMATH_CALUDE_tv_discounted_price_l2317_231706

def original_price : ℝ := 500.00
def discount1 : ℝ := 0.10
def discount2 : ℝ := 0.20
def discount3 : ℝ := 0.15

def final_price : ℝ := original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)

theorem tv_discounted_price : final_price = 306.00 := by
  sorry

end NUMINAMATH_CALUDE_tv_discounted_price_l2317_231706


namespace NUMINAMATH_CALUDE_multiply_and_simplify_l2317_231738

theorem multiply_and_simplify (x : ℝ) : 
  (x^4 + 49*x^2 + 2401) * (x^2 - 49) = x^6 - 117649 := by
sorry

end NUMINAMATH_CALUDE_multiply_and_simplify_l2317_231738


namespace NUMINAMATH_CALUDE_abc_is_246_l2317_231748

/-- Represents a base-8 number with two digits --/
def BaseEight (a b : ℕ) : ℕ := 8 * a + b

/-- Converts a three-digit number to its decimal representation --/
def ToDecimal (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

theorem abc_is_246 (A B C : ℕ) 
  (h1 : A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0)
  (h2 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h3 : A < 8 ∧ B < 8)
  (h4 : C < 6)
  (h5 : BaseEight A B + C = BaseEight C 2)
  (h6 : BaseEight A B + BaseEight B A = BaseEight C C) :
  ToDecimal A B C = 246 := by
  sorry

end NUMINAMATH_CALUDE_abc_is_246_l2317_231748


namespace NUMINAMATH_CALUDE_valid_C_characterization_l2317_231736

/-- A sequence of integers -/
def IntegerSequence := ℕ → ℤ

/-- A sequence is bounded below -/
def BoundedBelow (a : IntegerSequence) : Prop :=
  ∃ M : ℤ, ∀ n : ℕ, M ≤ a n

/-- A sequence satisfies the given inequality for a given C -/
def SatisfiesInequality (a : IntegerSequence) (C : ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → (0 : ℝ) ≤ a (n - 1) + C * a n + a (n + 1) ∧ 
                   a (n - 1) + C * a n + a (n + 1) < 1

/-- A sequence is periodic -/
def Periodic (a : IntegerSequence) : Prop :=
  ∃ p : ℕ, p > 0 ∧ ∀ n : ℕ, a (n + p) = a n

/-- The set of all C that satisfy the conditions -/
def ValidC : Set ℝ :=
  {C : ℝ | ∀ a : IntegerSequence, BoundedBelow a → SatisfiesInequality a C → Periodic a}

theorem valid_C_characterization : ValidC = Set.Ici (-2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_valid_C_characterization_l2317_231736


namespace NUMINAMATH_CALUDE_ellipse_from_hyperbola_l2317_231733

/-- Given a hyperbola with equation x²/4 - y²/12 = -1, prove that the equation of the ellipse
    whose foci are the vertices of the hyperbola and whose vertices are the foci of the hyperbola
    is x²/4 + y²/16 = 1 -/
theorem ellipse_from_hyperbola (x y : ℝ) :
  (x^2 / 4 - y^2 / 12 = -1) →
  ∃ (a b : ℝ), (a > 0 ∧ b > 0 ∧ a > b) ∧
  ((x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 4 + y^2 / 16 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_from_hyperbola_l2317_231733


namespace NUMINAMATH_CALUDE_complex_sum_equals_negative_ten_i_l2317_231728

/-- Prove that the sum of complex numbers (5-5i)+(-2-i)-(3+4i) equals -10i -/
theorem complex_sum_equals_negative_ten_i : (5 - 5*I) + (-2 - I) - (3 + 4*I) = -10*I := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equals_negative_ten_i_l2317_231728


namespace NUMINAMATH_CALUDE_f_derivative_l2317_231766

def f (x : ℝ) : ℝ := -3 * x - 1

theorem f_derivative : 
  deriv f = fun _ => -3 := by sorry

end NUMINAMATH_CALUDE_f_derivative_l2317_231766


namespace NUMINAMATH_CALUDE_quadratic_equation_real_roots_m_range_l2317_231759

theorem quadratic_equation_real_roots_m_range 
  (m : ℝ) 
  (has_real_roots : ∃ x : ℝ, (m - 2) * x^2 + 2 * m * x + m + 3 = 0) :
  m ≤ 6 ∧ m ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_roots_m_range_l2317_231759


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_increase_l2317_231747

theorem equilateral_triangle_area_increase :
  ∀ s : ℝ,
  s > 0 →
  (s^2 * Real.sqrt 3) / 4 = 36 * Real.sqrt 3 →
  let new_s := s + 2
  let new_area := (new_s^2 * Real.sqrt 3) / 4
  let original_area := (s^2 * Real.sqrt 3) / 4
  new_area - original_area = 13 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_increase_l2317_231747


namespace NUMINAMATH_CALUDE_specific_triangle_angle_l2317_231761

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the properties of the specific triangle -/
theorem specific_triangle_angle (t : Triangle) 
  (h1 : t.a = 2)
  (h2 : t.b = 2)
  (h3 : t.A = 45) :
  t.B = 67.5 := by
  sorry


end NUMINAMATH_CALUDE_specific_triangle_angle_l2317_231761


namespace NUMINAMATH_CALUDE_trefoil_cases_l2317_231783

theorem trefoil_cases (total_boxes : ℕ) (boxes_per_case : ℕ) (h1 : total_boxes = 24) (h2 : boxes_per_case = 8) :
  total_boxes / boxes_per_case = 3 := by
  sorry

end NUMINAMATH_CALUDE_trefoil_cases_l2317_231783


namespace NUMINAMATH_CALUDE_weight_of_new_person_l2317_231762

/-- Proves that if the average weight of 6 persons increases by 1.5 kg when a person
    weighing 65 kg is replaced by a new person, then the weight of the new person is 74 kg. -/
theorem weight_of_new_person
  (num_persons : ℕ)
  (avg_increase : ℝ)
  (old_weight : ℝ)
  (new_weight : ℝ)
  (h1 : num_persons = 6)
  (h2 : avg_increase = 1.5)
  (h3 : old_weight = 65)
  (h4 : new_weight = num_persons * avg_increase + old_weight) :
  new_weight = 74 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l2317_231762


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_value_l2317_231737

/-- 
Given two lines in the xy-plane defined by their equations,
this theorem states that if these lines are perpendicular,
then the parameter m must equal 1/2.
-/
theorem perpendicular_lines_m_value (m : ℝ) : 
  (∀ x y : ℝ, x - m * y + 2 * m = 0 → x + 2 * y - m = 0 → 
    (1 : ℝ) / m * (-1 / 2 : ℝ) = -1) → 
  m = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_value_l2317_231737


namespace NUMINAMATH_CALUDE_geometric_sequence_S_3_range_l2317_231731

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Define the sum of the first three terms
def S_3 (a : ℕ → ℝ) : ℝ := a 1 + a 2 + a 3

-- Theorem statement
theorem geometric_sequence_S_3_range
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_a2 : a 2 = 1) :
  ∃ y : ℝ, S_3 a = y ↔ y ∈ Set.Ici 3 ∪ Set.Iic (-1) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_S_3_range_l2317_231731


namespace NUMINAMATH_CALUDE_proposition_values_l2317_231707

theorem proposition_values (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬p) : 
  ¬p ∧ (q ∨ ¬q) :=
sorry

end NUMINAMATH_CALUDE_proposition_values_l2317_231707


namespace NUMINAMATH_CALUDE_soap_cost_for_year_l2317_231793

/-- The cost of soap for a year given the duration and price of a single bar -/
theorem soap_cost_for_year (months_per_bar : ℕ) (price_per_bar : ℕ) : 
  months_per_bar = 2 → price_per_bar = 8 → (12 / months_per_bar) * price_per_bar = 48 := by
  sorry

#check soap_cost_for_year

end NUMINAMATH_CALUDE_soap_cost_for_year_l2317_231793


namespace NUMINAMATH_CALUDE_two_intersections_l2317_231717

/-- A line in a plane represented by ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if two lines are identical -/
def identical (l1 l2 : Line) : Prop :=
  parallel l1 l2 ∧ l1.a * l2.c = l1.c * l2.a

/-- Check if two lines intersect -/
def intersect (l1 l2 : Line) : Prop :=
  ¬(parallel l1 l2) ∨ identical l1 l2

/-- The number of distinct intersection points of at least two lines -/
def num_intersections (lines : List Line) : ℕ :=
  sorry

/-- The three lines from the problem -/
def line1 : Line := ⟨3, 2, 4⟩
def line2 : Line := ⟨-1, 3, 3⟩
def line3 : Line := ⟨6, -4, 8⟩

/-- The main theorem -/
theorem two_intersections :
  num_intersections [line1, line2, line3] = 2 := by sorry

end NUMINAMATH_CALUDE_two_intersections_l2317_231717


namespace NUMINAMATH_CALUDE_shelby_heavy_rain_time_l2317_231705

/-- Represents the speeds and durations of Shelby's scooter ride --/
structure ScooterRide where
  sunnySpeed : ℝ
  lightRainSpeed : ℝ
  heavyRainSpeed : ℝ
  totalDistance : ℝ
  totalTime : ℝ
  heavyRainTime : ℝ

/-- Theorem stating that given the conditions of Shelby's ride, she spent 20 minutes in heavy rain --/
theorem shelby_heavy_rain_time (ride : ScooterRide) 
  (h1 : ride.sunnySpeed = 35)
  (h2 : ride.lightRainSpeed = 25)
  (h3 : ride.heavyRainSpeed = 15)
  (h4 : ride.totalDistance = 50)
  (h5 : ride.totalTime = 100) :
  ride.heavyRainTime = 20 := by
  sorry

#check shelby_heavy_rain_time

end NUMINAMATH_CALUDE_shelby_heavy_rain_time_l2317_231705


namespace NUMINAMATH_CALUDE_proportion_theorem_l2317_231702

theorem proportion_theorem (A B C p q r : ℝ) 
  (h1 : A / B = p) 
  (h2 : B / C = q) 
  (h3 : C / A = r) : 
  ∃ k : ℝ, k > 0 ∧ 
    A = k * (p^2 * q / r)^(1/3) ∧ 
    B = k * (q^2 * r / p)^(1/3) ∧ 
    C = k * (r^2 * p / q)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_proportion_theorem_l2317_231702


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_10_l2317_231715

theorem binomial_coefficient_20_10 
  (h1 : Nat.choose 18 8 = 43758)
  (h2 : Nat.choose 18 9 = 48620)
  (h3 : Nat.choose 18 10 = 43758) :
  Nat.choose 20 10 = 184756 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_10_l2317_231715


namespace NUMINAMATH_CALUDE_renovation_project_materials_l2317_231755

theorem renovation_project_materials (sand dirt cement gravel stone : ℝ) 
  (h1 : sand = 0.17)
  (h2 : dirt = 0.33)
  (h3 : cement = 0.17)
  (h4 : gravel = 0.25)
  (h5 : stone = 0.08) :
  sand + dirt + cement + gravel + stone = 1 := by
  sorry

end NUMINAMATH_CALUDE_renovation_project_materials_l2317_231755


namespace NUMINAMATH_CALUDE_stratified_sample_young_employees_l2317_231742

/-- Calculates the number of employees to be drawn from a specific age group in a stratified sample. -/
def stratifiedSampleSize (totalEmployees : ℕ) (groupSize : ℕ) (sampleSize : ℕ) : ℕ :=
  (groupSize * sampleSize) / totalEmployees

/-- Proves that the number of employees no older than 45 to be drawn in a stratified sample is 15. -/
theorem stratified_sample_young_employees :
  let totalEmployees : ℕ := 200
  let youngEmployees : ℕ := 120
  let sampleSize : ℕ := 25
  stratifiedSampleSize totalEmployees youngEmployees sampleSize = 15 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sample_young_employees_l2317_231742


namespace NUMINAMATH_CALUDE_contribution_ratio_l2317_231791

-- Define the contributions and profit
def robi_contribution : ℝ := 4000
def rudy_contribution : ℝ → ℝ := λ x => robi_contribution + x
def total_contribution : ℝ → ℝ := λ x => robi_contribution + rudy_contribution x
def profit_rate : ℝ := 0.20
def individual_profit : ℝ := 900

-- Define the theorem
theorem contribution_ratio :
  ∃ x : ℝ,
    x > 0 ∧
    profit_rate * total_contribution x = 2 * individual_profit ∧
    rudy_contribution x / robi_contribution = 5 / 4 := by
  sorry


end NUMINAMATH_CALUDE_contribution_ratio_l2317_231791


namespace NUMINAMATH_CALUDE_carlos_pesos_sum_of_digits_l2317_231710

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Represents the exchange rate from dollars to pesos -/
def exchangeRate : ℚ := 12 / 8

theorem carlos_pesos_sum_of_digits :
  ∀ d : ℕ,
  (exchangeRate * d - 72 : ℚ) = d →
  sumOfDigits d = 9 := by sorry

end NUMINAMATH_CALUDE_carlos_pesos_sum_of_digits_l2317_231710


namespace NUMINAMATH_CALUDE_chicken_farm_proof_l2317_231758

/-- The number of chickens Michael has now -/
def initial_chickens : ℕ := 550

/-- The annual increase in the number of chickens -/
def annual_increase : ℕ := 150

/-- The number of years -/
def years : ℕ := 9

/-- The number of chickens after 9 years -/
def final_chickens : ℕ := 1900

/-- Theorem stating that the initial number of chickens plus the total increase over 9 years equals the final number of chickens -/
theorem chicken_farm_proof : 
  initial_chickens + (annual_increase * years) = final_chickens := by
  sorry


end NUMINAMATH_CALUDE_chicken_farm_proof_l2317_231758


namespace NUMINAMATH_CALUDE_watch_cost_price_l2317_231754

/-- The cost price of a watch satisfying certain conditions -/
theorem watch_cost_price : ∃ (cp : ℚ), 
  cp > 0 ∧ 
  (0.9 * cp = cp - 0.1 * cp) ∧ 
  (1.04 * cp = cp + 0.04 * cp) ∧ 
  (1.04 * cp - 0.9 * cp = 168) ∧ 
  cp = 1200 := by
sorry

end NUMINAMATH_CALUDE_watch_cost_price_l2317_231754


namespace NUMINAMATH_CALUDE_max_sum_cubes_l2317_231775

theorem max_sum_cubes (a b c d : ℝ) 
  (sum_squares : a^2 + b^2 + c^2 + d^2 = 16)
  (distinct : a ≠ b ∧ b ≠ c ∧ c ≠ d) :
  a^3 + b^3 + c^3 + d^3 ≤ 64 ∧ 
  ∃ (x y z w : ℝ), x^2 + y^2 + z^2 + w^2 = 16 ∧ 
                   x ≠ y ∧ y ≠ z ∧ z ≠ w ∧
                   x^3 + y^3 + z^3 + w^3 = 64 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_cubes_l2317_231775


namespace NUMINAMATH_CALUDE_problem_statement_l2317_231749

theorem problem_statement (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 2*y = 1 ∧ a^2 + b^2 ≥ x^2 + y^2) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y = 1 → a^2 + b^2 ≤ x^2 + y^2) ∧
  (a^2 + b^2 = 1/5) ∧
  (a*b ≤ 1/8) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y = 1 → x*y ≤ 1/8) ∧
  (1/a + 1/b ≥ 3 + 2*Real.sqrt 2) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y = 1 → 1/x + 1/y ≥ 3 + 2*Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2317_231749


namespace NUMINAMATH_CALUDE_five_T_three_equals_38_l2317_231745

-- Define the operation T
def T (a b : ℝ) : ℝ := 4 * a + 6 * b

-- Theorem to prove
theorem five_T_three_equals_38 : T 5 3 = 38 := by
  sorry

end NUMINAMATH_CALUDE_five_T_three_equals_38_l2317_231745


namespace NUMINAMATH_CALUDE_unused_streetlights_l2317_231768

/-- Given the number of streetlights bought by the New York City Council, 
    the number of squares in New York, and the number of streetlights per square, 
    calculate the number of unused streetlights. -/
theorem unused_streetlights (total : ℕ) (squares : ℕ) (per_square : ℕ) 
    (h1 : total = 200) (h2 : squares = 15) (h3 : per_square = 12) : 
  total - squares * per_square = 20 := by
  sorry

end NUMINAMATH_CALUDE_unused_streetlights_l2317_231768


namespace NUMINAMATH_CALUDE_equal_pairwise_products_l2317_231764

theorem equal_pairwise_products (n : ℕ) : 
  (¬ ∃ n : ℕ, n > 0 ∧ n < 1000 ∧ n^2 - 1000*n + 499500 = 0) ∧
  (∃ n : ℕ, n > 0 ∧ n < 10000 ∧ n^2 - 10000*n + 49995000 = 0) := by
  sorry

end NUMINAMATH_CALUDE_equal_pairwise_products_l2317_231764


namespace NUMINAMATH_CALUDE_car_average_speed_l2317_231778

/-- Given a car traveling at different speeds for two hours, 
    calculate its average speed. -/
theorem car_average_speed 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (h1 : speed1 = 20) 
  (h2 : speed2 = 60) : 
  (speed1 + speed2) / 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_car_average_speed_l2317_231778


namespace NUMINAMATH_CALUDE_storage_b_has_five_pieces_l2317_231716

/-- Represents a storage device with a number of data pieces -/
structure StorageDevice :=
  (pieces : ℕ)

/-- Represents the state of three storage devices A, B, and C -/
structure StorageState :=
  (A : StorageDevice)
  (B : StorageDevice)
  (C : StorageDevice)

/-- Performs the described operations on the storage devices -/
def performOperations (n : ℕ) (initial : StorageState) : StorageState :=
  { A := ⟨2 * (n - 2)⟩,
    B := ⟨n + 3 - (n - 2)⟩,
    C := ⟨n - 1⟩ }

/-- The theorem stating that after the operations, storage device B has 5 data pieces -/
theorem storage_b_has_five_pieces (n : ℕ) (h : n ≥ 2) :
  (performOperations n { A := ⟨0⟩, B := ⟨0⟩, C := ⟨0⟩ }).B.pieces = 5 := by
  sorry

#check storage_b_has_five_pieces

end NUMINAMATH_CALUDE_storage_b_has_five_pieces_l2317_231716


namespace NUMINAMATH_CALUDE_x_plus_y_values_l2317_231774

theorem x_plus_y_values (x y : ℝ) 
  (eq1 : x^2 + x*y + 2*y = 10) 
  (eq2 : y^2 + x*y + 2*x = 14) : 
  x + y = 4 ∨ x + y = -6 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l2317_231774


namespace NUMINAMATH_CALUDE_card_area_after_shortening_l2317_231756

/-- Given a rectangle with initial dimensions 5 × 7 inches, 
    prove that shortening both sides by 1 inch results in an area of 24 square inches. -/
theorem card_area_after_shortening :
  let initial_length : ℝ := 7
  let initial_width : ℝ := 5
  let shortened_length : ℝ := initial_length - 1
  let shortened_width : ℝ := initial_width - 1
  shortened_length * shortened_width = 24 := by
  sorry

end NUMINAMATH_CALUDE_card_area_after_shortening_l2317_231756


namespace NUMINAMATH_CALUDE_five_dice_not_same_probability_l2317_231739

theorem five_dice_not_same_probability : 
  let n : ℕ := 5  -- number of dice
  let s : ℕ := 6  -- number of sides on each die
  let total_outcomes : ℕ := s^n
  let same_number_outcomes : ℕ := s
  let prob_not_same : ℚ := 1 - (same_number_outcomes : ℚ) / total_outcomes
  prob_not_same = 1295 / 1296 :=
by sorry

end NUMINAMATH_CALUDE_five_dice_not_same_probability_l2317_231739


namespace NUMINAMATH_CALUDE_doubled_b_cost_percentage_l2317_231773

-- Define the cost function
def cost (t : ℝ) (b : ℝ) : ℝ := t * b^4

-- Theorem statement
theorem doubled_b_cost_percentage (t : ℝ) (b : ℝ) (h : t > 0) (h' : b > 0) :
  cost t (2*b) = 16 * cost t b := by
  sorry

end NUMINAMATH_CALUDE_doubled_b_cost_percentage_l2317_231773


namespace NUMINAMATH_CALUDE_unique_solution_when_a_is_seven_l2317_231781

/-- The equation has exactly one solution when a = 7 -/
theorem unique_solution_when_a_is_seven (x : ℝ) (a : ℝ) : 
  (a ≠ 1 ∧ x ≠ -3) →
  (∃! x, ((|((a*x^2 - a*x - 12*a + x^2 + x + 12) / (a*x + 3*a - x - 3))| - a) * |4*a - 3*x - 19| = 0)) ↔
  a = 7 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_when_a_is_seven_l2317_231781


namespace NUMINAMATH_CALUDE_circle_diameter_l2317_231712

theorem circle_diameter (C : ℝ) (h : C = 100) : C / π = 100 / π := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_l2317_231712


namespace NUMINAMATH_CALUDE_open_box_volume_l2317_231757

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet -/
theorem open_box_volume (sheet_length sheet_width cut_length : ℝ) 
  (h1 : sheet_length = 48)
  (h2 : sheet_width = 36)
  (h3 : cut_length = 4) : 
  (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length = 4480 := by
  sorry

end NUMINAMATH_CALUDE_open_box_volume_l2317_231757


namespace NUMINAMATH_CALUDE_point_A_coordinates_l2317_231786

theorem point_A_coordinates :
  ∀ a : ℤ,
  (a + 1 < 0) →
  (2 * a + 6 > 0) →
  (a + 1, 2 * a + 6) = (-1, 2) :=
by
  sorry

end NUMINAMATH_CALUDE_point_A_coordinates_l2317_231786
