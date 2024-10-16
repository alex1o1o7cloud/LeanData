import Mathlib

namespace NUMINAMATH_CALUDE_no_solution_exists_l2944_294446

theorem no_solution_exists : ∀ k : ℕ, k^6 + k^4 + k^2 ≠ 10^(k+1) + 9 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2944_294446


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l2944_294498

theorem vector_magnitude_proof (a b : ℝ × ℝ) : 
  a = (-1, 2) → b = (1, 3) → ‖(2 • a) - b‖ = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l2944_294498


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_characterization_l2944_294409

/-- A quadrilateral is cyclic if and only if the sum of products of opposite angles equals π². -/
theorem cyclic_quadrilateral_characterization (α β γ δ : Real) 
  (h_angles : α + β + γ + δ = 2 * Real.pi) : 
  (α + γ = Real.pi ∧ β + δ = Real.pi) ↔ α * β + α * δ + γ * β + γ * δ = Real.pi ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_characterization_l2944_294409


namespace NUMINAMATH_CALUDE_fifth_store_cars_l2944_294470

def store_count : Nat := 5
def car_counts : Vector Nat 4 := ⟨[30, 14, 14, 21], rfl⟩
def mean : Rat := 104/5

theorem fifth_store_cars : 
  ∃ x : Nat, (car_counts.toList.sum + x) / store_count = mean :=
by
  sorry

end NUMINAMATH_CALUDE_fifth_store_cars_l2944_294470


namespace NUMINAMATH_CALUDE_eggs_sold_equals_450_l2944_294459

/-- The number of eggs in one tray -/
def eggs_per_tray : ℕ := 30

/-- The initial number of trays to be collected -/
def initial_trays : ℕ := 10

/-- The number of trays dropped (lost) -/
def dropped_trays : ℕ := 2

/-- The number of additional trays added after the accident -/
def additional_trays : ℕ := 7

/-- The total number of eggs sold -/
def eggs_sold : ℕ := (initial_trays - dropped_trays + additional_trays) * eggs_per_tray

theorem eggs_sold_equals_450 : eggs_sold = 450 := by
  sorry

end NUMINAMATH_CALUDE_eggs_sold_equals_450_l2944_294459


namespace NUMINAMATH_CALUDE_two_and_half_dozens_eq_30_l2944_294441

/-- The number of items in a dozen -/
def dozen : ℕ := 12

/-- The number of pens in two and one-half dozens -/
def two_and_half_dozens : ℕ := 2 * dozen + dozen / 2

/-- Theorem stating that two and one-half dozens of pens is equal to 30 pens -/
theorem two_and_half_dozens_eq_30 : two_and_half_dozens = 30 := by
  sorry

end NUMINAMATH_CALUDE_two_and_half_dozens_eq_30_l2944_294441


namespace NUMINAMATH_CALUDE_least_integer_square_48_more_than_double_l2944_294495

theorem least_integer_square_48_more_than_double :
  ∃ x : ℤ, x^2 = 2*x + 48 ∧ ∀ y : ℤ, y^2 = 2*y + 48 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_integer_square_48_more_than_double_l2944_294495


namespace NUMINAMATH_CALUDE_negation_of_existence_square_positive_negation_l2944_294431

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, x < 0 ∧ p x) ↔ (∀ x, x < 0 → ¬ p x) := by sorry

theorem square_positive_negation :
  (¬ ∃ x : ℝ, x < 0 ∧ x^2 > 0) ↔ (∀ x : ℝ, x < 0 → x^2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_square_positive_negation_l2944_294431


namespace NUMINAMATH_CALUDE_set_operations_l2944_294434

-- Define the sets
def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def C_U_N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define N (this is what we need to prove)
def N : Set ℝ := {x | (-3 ≤ x ∧ x ≤ 0) ∨ (2 ≤ x ∧ x ≤ 3)}

-- State the theorem
theorem set_operations :
  (N = {x : ℝ | (-3 ≤ x ∧ x ≤ 0) ∨ (2 ≤ x ∧ x ≤ 3)}) ∧
  (M ∩ C_U_N = {x : ℝ | 0 < x ∧ x < 1}) ∧
  (M ∪ N = {x : ℝ | (-3 ≤ x ∧ x < 1) ∨ (2 ≤ x ∧ x ≤ 3)}) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_l2944_294434


namespace NUMINAMATH_CALUDE_f_properties_l2944_294430

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem f_properties :
  (∃ (x_min : ℝ), f 1 x_min = 2 ∧ ∀ x, f 1 x ≥ f 1 x_min) ∧
  (∀ a ≤ 0, ∀ x y, x < y → f a x > f a y) ∧
  (∀ a > 0, ∀ x y, x < y → 
    ((x < -Real.log a ∧ y < -Real.log a) → f a x > f a y) ∧
    ((x > -Real.log a ∧ y > -Real.log a) → f a x < f a y)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2944_294430


namespace NUMINAMATH_CALUDE_min_distance_between_points_l2944_294414

/-- Given four points P, Q, R, and S in a metric space, with distances PQ = 12, QR = 5, and RS = 8,
    the minimum possible distance between P and S is 1. -/
theorem min_distance_between_points (X : Type*) [MetricSpace X] 
  (P Q R S : X) 
  (h_PQ : dist P Q = 12)
  (h_QR : dist Q R = 5)
  (h_RS : dist R S = 8) : 
  ∃ (configuration : X → X), dist (configuration P) (configuration S) = 1 ∧ 
    (∀ (config : X → X), dist (config P) (config S) ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_min_distance_between_points_l2944_294414


namespace NUMINAMATH_CALUDE_inequality_proof_l2944_294412

theorem inequality_proof (x y z t : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (ht : t > 0) :
  (x + y + z + t) / 2 + 4 / (x*y + y*z + z*t + t*x) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2944_294412


namespace NUMINAMATH_CALUDE_equation_system_solution_l2944_294478

theorem equation_system_solution (a b c d : ℝ) :
  (a + b = c + d) →
  (a^3 + b^3 = c^3 + d^3) →
  ((a = c ∧ b = d) ∨ (a = d ∧ b = c)) :=
by sorry

end NUMINAMATH_CALUDE_equation_system_solution_l2944_294478


namespace NUMINAMATH_CALUDE_mean_proportional_of_segments_l2944_294473

theorem mean_proportional_of_segments (a b : ℝ) (ha : a = 2) (hb : b = 6) :
  ∃ c : ℝ, c > 0 ∧ c^2 = a * b ∧ c = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_of_segments_l2944_294473


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2944_294405

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + bx - 2 > 0 ↔ -4 < x ∧ x < 1) → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2944_294405


namespace NUMINAMATH_CALUDE_scientific_notation_of_58000000000_l2944_294464

theorem scientific_notation_of_58000000000 :
  (58000000000 : ℝ) = 5.8 * (10 : ℝ) ^ 10 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_58000000000_l2944_294464


namespace NUMINAMATH_CALUDE_investment_growth_l2944_294468

/-- Given an initial investment that grows to $400 after 4 years at 25% simple interest per year,
    prove that the value after 6 years is $500. -/
theorem investment_growth (P : ℝ) : 
  P + P * 0.25 * 4 = 400 → 
  P + P * 0.25 * 6 = 500 := by
  sorry

end NUMINAMATH_CALUDE_investment_growth_l2944_294468


namespace NUMINAMATH_CALUDE_max_identical_end_digits_of_square_l2944_294406

theorem max_identical_end_digits_of_square (n : ℕ) (h : n % 10 ≠ 0) :
  ∀ k : ℕ, k > 4 → ∃ d : ℕ, d < 10 ∧ (n^2) % (10^k) ≠ d * ((10^k - 1) / 9) :=
sorry

end NUMINAMATH_CALUDE_max_identical_end_digits_of_square_l2944_294406


namespace NUMINAMATH_CALUDE_josette_bought_three_bottles_l2944_294415

/-- The number of bottles Josette bought for €1.50, given that 4 bottles cost €2 -/
def bottles_bought (cost_four_bottles : ℚ) (amount_spent : ℚ) : ℚ :=
  amount_spent / (cost_four_bottles / 4)

/-- Theorem stating that Josette bought 3 bottles -/
theorem josette_bought_three_bottles : 
  bottles_bought 2 (3/2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_josette_bought_three_bottles_l2944_294415


namespace NUMINAMATH_CALUDE_trig_expression_equality_l2944_294456

theorem trig_expression_equality : 
  (Real.sin (38 * π / 180) * Real.sin (38 * π / 180) + 
   Real.cos (38 * π / 180) * Real.sin (52 * π / 180) - 
   Real.tan (15 * π / 180) ^ 2) / 
  (3 * Real.tan (15 * π / 180)) = 
  (2 + Real.sqrt 3) / 9 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l2944_294456


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2944_294400

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2 - x| < 1} = Set.Ioo 1 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2944_294400


namespace NUMINAMATH_CALUDE_quadratic_equation_two_roots_l2944_294469

-- Define the geometric progression
def is_geometric_progression (a b c : ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ b = a * q ∧ c = a * q^2

-- Define the quadratic equation
def has_two_distinct_roots (a b c : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + 2 * Real.sqrt 2 * b * x₁ + c = 0 ∧
                        a * x₂^2 + 2 * Real.sqrt 2 * b * x₂ + c = 0

-- Theorem statement
theorem quadratic_equation_two_roots
  (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_geom : is_geometric_progression a b c) :
  has_two_distinct_roots a b c :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_two_roots_l2944_294469


namespace NUMINAMATH_CALUDE_smallest_odd_between_2_and_7_l2944_294427

theorem smallest_odd_between_2_and_7 : 
  ∀ n : ℕ, (2 < n ∧ n < 7 ∧ Odd n) → 3 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_odd_between_2_and_7_l2944_294427


namespace NUMINAMATH_CALUDE_cube_sum_preceding_integers_l2944_294477

theorem cube_sum_preceding_integers : ∃ n : ℤ, n = 6 ∧ n^3 = (n-1)^3 + (n-2)^3 + (n-3)^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_preceding_integers_l2944_294477


namespace NUMINAMATH_CALUDE_total_mission_time_is_11_days_l2944_294486

/-- Calculates the total time spent on missions given the planned duration of the first mission,
    the percentage increase in duration, and the duration of the second mission. -/
def total_mission_time (planned_duration : ℝ) (percentage_increase : ℝ) (second_mission_duration : ℝ) : ℝ :=
  (planned_duration * (1 + percentage_increase)) + second_mission_duration

/-- Proves that the total time spent on missions is 11 days. -/
theorem total_mission_time_is_11_days : 
  total_mission_time 5 0.6 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_mission_time_is_11_days_l2944_294486


namespace NUMINAMATH_CALUDE_yoque_borrowed_amount_l2944_294444

/-- The amount Yoque borrowed -/
def borrowed_amount : ℝ := 150

/-- The number of months for repayment -/
def repayment_period : ℕ := 11

/-- The monthly payment amount -/
def monthly_payment : ℝ := 15

/-- The interest rate as a decimal -/
def interest_rate : ℝ := 0.1

theorem yoque_borrowed_amount :
  borrowed_amount = (monthly_payment * repayment_period) / (1 + interest_rate) :=
by sorry

end NUMINAMATH_CALUDE_yoque_borrowed_amount_l2944_294444


namespace NUMINAMATH_CALUDE_auto_dealer_sales_l2944_294426

theorem auto_dealer_sales (trucks : ℕ) (cars : ℕ) : 
  trucks = 21 →
  cars = trucks + 27 →
  cars + trucks = 69 := by
sorry

end NUMINAMATH_CALUDE_auto_dealer_sales_l2944_294426


namespace NUMINAMATH_CALUDE_intersection_condition_l2944_294485

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M (a : ℝ) : Set ℝ := {x | x + a ≥ 0}

-- Define set N
def N : Set ℝ := {x | x - 2 < 1}

-- Theorem statement
theorem intersection_condition (a : ℝ) :
  M a ∩ (Set.compl N) = {x | x ≥ 3} → a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l2944_294485


namespace NUMINAMATH_CALUDE_min_students_in_class_l2944_294490

theorem min_students_in_class (b g : ℕ) : 
  b > 0 → 
  g > 0 → 
  2 * (b / 2) = 3 * (g / 3) → 
  b + g ≥ 7 ∧ 
  ∃ (b' g' : ℕ), b' > 0 ∧ g' > 0 ∧ 2 * (b' / 2) = 3 * (g' / 3) ∧ b' + g' = 7 :=
by sorry

end NUMINAMATH_CALUDE_min_students_in_class_l2944_294490


namespace NUMINAMATH_CALUDE_x_plus_twice_y_l2944_294403

theorem x_plus_twice_y (x y z : ℚ) : 
  x = y / 3 → y = z / 4 → z = 100 → x + 2 * y = 175 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_twice_y_l2944_294403


namespace NUMINAMATH_CALUDE_matchsticks_20th_stage_l2944_294476

def matchsticks (n : ℕ) : ℕ :=
  5 + 3 * (n - 1) + (n - 1) / 5

theorem matchsticks_20th_stage :
  matchsticks 20 = 66 := by
  sorry

end NUMINAMATH_CALUDE_matchsticks_20th_stage_l2944_294476


namespace NUMINAMATH_CALUDE_sleep_increase_l2944_294497

theorem sleep_increase (initial_sleep : ℝ) (increase_fraction : ℝ) (final_sleep : ℝ) :
  initial_sleep = 6 →
  increase_fraction = 1/3 →
  final_sleep = initial_sleep + initial_sleep * increase_fraction →
  final_sleep = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_sleep_increase_l2944_294497


namespace NUMINAMATH_CALUDE_sum_of_max_min_g_l2944_294408

-- Define the function g(x)
def g (x : ℝ) : ℝ := |x - 2| + |x - 4| + |x - 6| - |2*x - 6|

-- Define the domain of x
def domain (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 10

-- Theorem statement
theorem sum_of_max_min_g :
  ∃ (max min : ℝ), 
    (∀ x, domain x → g x ≤ max) ∧
    (∃ x, domain x ∧ g x = max) ∧
    (∀ x, domain x → min ≤ g x) ∧
    (∃ x, domain x ∧ g x = min) ∧
    max + min = 14 :=
sorry

end NUMINAMATH_CALUDE_sum_of_max_min_g_l2944_294408


namespace NUMINAMATH_CALUDE_cookie_chip_ratio_l2944_294474

/-- Proves that the ratio of cookie tins to chip bags is 4:1 given the problem conditions -/
theorem cookie_chip_ratio :
  let chip_weight : ℕ := 20  -- weight of a bag of chips in ounces
  let cookie_weight : ℕ := 9  -- weight of a tin of cookies in ounces
  let chip_bags : ℕ := 6  -- number of bags of chips Jasmine buys
  let total_weight : ℕ := 21 * 16  -- total weight Jasmine carries in ounces

  let cookie_tins : ℕ := (total_weight - chip_weight * chip_bags) / cookie_weight

  (cookie_tins : ℚ) / chip_bags = 4 / 1 :=
by sorry

end NUMINAMATH_CALUDE_cookie_chip_ratio_l2944_294474


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2944_294451

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 81

-- Define the factored form
def f (x : ℝ) : ℝ := (x-3)*(x+3)*(x^2+9)

-- Theorem stating the equality of the polynomial and its factored form
theorem polynomial_factorization :
  ∀ x : ℝ, p x = f x :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2944_294451


namespace NUMINAMATH_CALUDE_runner_problem_l2944_294424

theorem runner_problem (v : ℝ) (h1 : v > 0) :
  (40 / v = 20 / v + 4) →
  (40 / (v / 2) = 8) :=
by sorry

end NUMINAMATH_CALUDE_runner_problem_l2944_294424


namespace NUMINAMATH_CALUDE_log_equation_l2944_294450

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equation : (log10 5)^2 + log10 2 * log10 50 = 1 := by sorry

end NUMINAMATH_CALUDE_log_equation_l2944_294450


namespace NUMINAMATH_CALUDE_sin_30_degrees_l2944_294407

theorem sin_30_degrees : Real.sin (30 * π / 180) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_30_degrees_l2944_294407


namespace NUMINAMATH_CALUDE_equation_solutions_l2944_294472

theorem equation_solutions :
  ∀ x y : ℤ, y ≥ 0 → (24 * y + 1 = (4 * y^2 - x^2)^2) →
    ((x = 1 ∨ x = -1) ∧ y = 0) ∨
    ((x = 3 ∨ x = -3) ∧ y = 1) ∨
    ((x = 3 ∨ x = -3) ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2944_294472


namespace NUMINAMATH_CALUDE_distribute_four_to_three_l2944_294462

/-- The number of ways to distribute n distinct objects into k distinct containers,
    with each container having at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose r objects from n distinct objects. -/
def choose (n r : ℕ) : ℕ := sorry

/-- The number of ways to arrange n distinct objects in k positions. -/
def arrange (n k : ℕ) : ℕ := sorry

theorem distribute_four_to_three :
  distribute 4 3 = 36 :=
by
  have h1 : distribute 4 3 = choose 4 2 * arrange 3 3 := sorry
  sorry


end NUMINAMATH_CALUDE_distribute_four_to_three_l2944_294462


namespace NUMINAMATH_CALUDE_product_zero_l2944_294449

theorem product_zero (r : ℂ) (h1 : r^4 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_zero_l2944_294449


namespace NUMINAMATH_CALUDE_intersection_integer_iff_k_valid_l2944_294418

/-- A point in the Cartesian plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Check if a point is the intersection of two lines -/
def is_intersection (p : Point) (k : ℤ) : Prop :=
  p.y = p.x - 2 ∧ p.y = k * p.x + k

/-- The set of valid k values -/
def valid_k : Set ℤ := {-2, 0, 2, 4}

/-- Main theorem: The intersection is an integer point iff k is in the valid set -/
theorem intersection_integer_iff_k_valid (k : ℤ) :
  (∃ p : Point, is_intersection p k) ↔ k ∈ valid_k :=
sorry

end NUMINAMATH_CALUDE_intersection_integer_iff_k_valid_l2944_294418


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_inequality_l2944_294422

theorem sum_of_reciprocals_inequality (a b c : ℝ) (h : a + b + c = 3) :
  (1 / (5 * a^2 - 4 * a + 1) + 1 / (5 * b^2 - 4 * b + 1) + 1 / (5 * c^2 - 4 * c + 1)) ≤ 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_inequality_l2944_294422


namespace NUMINAMATH_CALUDE_expression_simplification_l2944_294494

theorem expression_simplification (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 - 2) / x) * ((y^2 - 2) / y) - ((x^2 + 2) / y) * ((y^2 + 2) / x) = -4 * (x / y + y / x) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2944_294494


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2944_294440

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 16) = 5 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2944_294440


namespace NUMINAMATH_CALUDE_floor_abs_negative_real_l2944_294410

theorem floor_abs_negative_real : ⌊|(-54.7 : ℝ)|⌋ = 54 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_real_l2944_294410


namespace NUMINAMATH_CALUDE_no_valid_two_digit_number_l2944_294435

theorem no_valid_two_digit_number : ¬ ∃ (N : ℕ), 
  (10 ≤ N ∧ N < 100) ∧ 
  (∃ (x : ℕ), 
    x > 3 ∧ 
    N - (10 * (N % 10) + N / 10) = x^3) :=
sorry

end NUMINAMATH_CALUDE_no_valid_two_digit_number_l2944_294435


namespace NUMINAMATH_CALUDE_function_decomposition_l2944_294492

/-- A function is α-periodic if f(x + α) = f(x) for all x -/
def Periodic (f : ℝ → ℝ) (α : ℝ) : Prop :=
  ∀ x, f (x + α) = f x

/-- A function is linear if f(x) = ax for some constant a -/
def Linear (f : ℝ → ℝ) : Prop :=
  ∃ a, ∀ x, f x = a * x

theorem function_decomposition (f : ℝ → ℝ) (α β : ℝ) (hα : α ≠ 0)
    (h : ∀ x, f (x + α) = f x + β) :
    ∃ (g h : ℝ → ℝ), Periodic g α ∧ Linear h ∧ ∀ x, f x = g x + h x := by
  sorry

end NUMINAMATH_CALUDE_function_decomposition_l2944_294492


namespace NUMINAMATH_CALUDE_min_value_of_x_plus_3y_min_value_is_16_min_value_achieved_l2944_294475

theorem min_value_of_x_plus_3y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 3 * x + y) :
  ∀ a b : ℝ, a > 0 → b > 0 → a * b = 3 * a + b → x + 3 * y ≤ a + 3 * b :=
by sorry

theorem min_value_is_16 (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 3 * x + y) :
  x + 3 * y ≥ 16 :=
by sorry

theorem min_value_achieved (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 3 * x + y) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a * b = 3 * a + b ∧ a + 3 * b = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_x_plus_3y_min_value_is_16_min_value_achieved_l2944_294475


namespace NUMINAMATH_CALUDE_unique_remainder_mod_ten_l2944_294419

theorem unique_remainder_mod_ten :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -1345 [ZMOD 10] ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_remainder_mod_ten_l2944_294419


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_seventeen_sixths_l2944_294458

theorem sum_of_solutions_eq_seventeen_sixths :
  let f : ℝ → ℝ := λ x => (3*x + 5) * (2*x - 9)
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) →
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = 17/6) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_seventeen_sixths_l2944_294458


namespace NUMINAMATH_CALUDE_remainder_theorem_l2944_294432

theorem remainder_theorem (x y u v : ℕ) (hx : x > 0) (hy : y > 0) 
  (hu : u = x / y) (hv : v = x % y) (hv_bound : v < y) : 
  (x + 3 * u * y + y) % y = v := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2944_294432


namespace NUMINAMATH_CALUDE_odd_function_root_property_l2944_294436

/-- A function f : ℝ → ℝ is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- x₀ is a root of f(x) + exp(x) = 0 -/
def IsRootOf (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f x₀ + Real.exp x₀ = 0

theorem odd_function_root_property (f : ℝ → ℝ) (x₀ : ℝ) 
    (h_odd : IsOdd f) (h_root : IsRootOf f x₀) :
    Real.exp (-x₀) * f (-x₀) - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_root_property_l2944_294436


namespace NUMINAMATH_CALUDE_smallest_base_for_90_l2944_294499

theorem smallest_base_for_90 : 
  ∃ (b : ℕ), b = 5 ∧ 
  (∀ (x : ℕ), x < b → ¬(∃ (d₁ d₂ d₃ : ℕ), d₁ < x ∧ d₂ < x ∧ d₃ < x ∧ 
    90 = d₁ * x^2 + d₂ * x + d₃)) ∧
  (∃ (d₁ d₂ d₃ : ℕ), d₁ < b ∧ d₂ < b ∧ d₃ < b ∧ 
    90 = d₁ * b^2 + d₂ * b + d₃) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_90_l2944_294499


namespace NUMINAMATH_CALUDE_kaleb_toy_purchase_l2944_294489

def max_toys_purchasable (initial_savings : ℕ) (allowance : ℕ) (toy_cost : ℕ) : ℕ :=
  (initial_savings + allowance) / toy_cost

theorem kaleb_toy_purchase :
  max_toys_purchasable 21 15 6 = 6 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_toy_purchase_l2944_294489


namespace NUMINAMATH_CALUDE_lines_skew_iff_b_neq_l2944_294439

/-- Two lines in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Definition of skew lines -/
def are_skew (l1 l2 : Line3D) : Prop :=
  ∀ t u : ℝ, l1.point + t • l1.direction ≠ l2.point + u • l2.direction

/-- The problem statement -/
theorem lines_skew_iff_b_neq (b : ℝ) :
  let l1 : Line3D := ⟨(1, 2, b), (2, 3, 4)⟩
  let l2 : Line3D := ⟨(3, 0, -1), (5, 3, 1)⟩
  are_skew l1 l2 ↔ b ≠ 11/3 := by
  sorry

end NUMINAMATH_CALUDE_lines_skew_iff_b_neq_l2944_294439


namespace NUMINAMATH_CALUDE_class_average_approx_76_percent_l2944_294416

def class_average (group1_percent : ℝ) (group1_score : ℝ) 
                  (group2_percent : ℝ) (group2_score : ℝ) 
                  (group3_percent : ℝ) (group3_score : ℝ) : ℝ :=
  group1_percent * group1_score + group2_percent * group2_score + group3_percent * group3_score

theorem class_average_approx_76_percent :
  let group1_percent : ℝ := 0.15
  let group1_score : ℝ := 100
  let group2_percent : ℝ := 0.50
  let group2_score : ℝ := 78
  let group3_percent : ℝ := 0.35
  let group3_score : ℝ := 63
  let average := class_average group1_percent group1_score group2_percent group2_score group3_percent group3_score
  ∃ ε > 0, |average - 76| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_class_average_approx_76_percent_l2944_294416


namespace NUMINAMATH_CALUDE_power_equation_solution_l2944_294417

theorem power_equation_solution (m : ℝ) : 2^m = (64 : ℝ)^(1/3) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2944_294417


namespace NUMINAMATH_CALUDE_polynomial_no_real_roots_l2944_294483

theorem polynomial_no_real_roots (a b c : ℝ) (h : |a| + |b| + |c| ≤ Real.sqrt 2) :
  ∀ x : ℝ, x^4 + a*x^3 + b*x^2 + c*x + 1 > 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_no_real_roots_l2944_294483


namespace NUMINAMATH_CALUDE_circle_proof_l2944_294484

-- Define the points A and B
def A : ℝ × ℝ := (-1, 3)
def B : ℝ × ℝ := (3, -1)

-- Define the line on which the center lies
def center_line (x y : ℝ) : Prop := x - 2*y - 1 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y + 1)^2 = 16

-- Theorem statement
theorem circle_proof :
  ∃ (center : ℝ × ℝ),
    center_line center.1 center.2 ∧
    circle_equation A.1 A.2 ∧
    circle_equation B.1 B.2 :=
by sorry

end NUMINAMATH_CALUDE_circle_proof_l2944_294484


namespace NUMINAMATH_CALUDE_non_decreasing_integers_count_l2944_294421

/-- The number of digits in the integers we're considering -/
def n : ℕ := 11

/-- The number of possible digit values (1 to 9) -/
def k : ℕ := 9

/-- The number of 11-digit positive integers with non-decreasing digits -/
def non_decreasing_integers : ℕ := Nat.choose (n + k - 1) (k - 1)

theorem non_decreasing_integers_count : non_decreasing_integers = 75582 := by
  sorry

end NUMINAMATH_CALUDE_non_decreasing_integers_count_l2944_294421


namespace NUMINAMATH_CALUDE_square_of_negative_sqrt_five_l2944_294423

theorem square_of_negative_sqrt_five : (-Real.sqrt 5)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_sqrt_five_l2944_294423


namespace NUMINAMATH_CALUDE_sum_of_roots_l2944_294454

theorem sum_of_roots (a b c d : ℝ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (∀ x : ℝ, x^2 - 10*a*x - 11*b = 0 ↔ (x = c ∨ x = d)) →
  (∀ x : ℝ, x^2 - 10*c*x - 11*d = 0 ↔ (x = a ∨ x = b)) →
  a + b + c + d = 1210 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2944_294454


namespace NUMINAMATH_CALUDE_cubic_root_in_interval_l2944_294496

/-- Given a cubic equation with three real roots and a condition on its coefficients,
    prove that at least one root belongs to the interval [0, 2]. -/
theorem cubic_root_in_interval
  (a b c : ℝ)
  (has_three_real_roots : ∃ x y z : ℝ, ∀ t : ℝ, t^3 + a*t^2 + b*t + c = 0 ↔ t = x ∨ t = y ∨ t = z)
  (coef_sum_bound : 2 ≤ a + b + c ∧ a + b + c ≤ 0) :
  ∃ r : ℝ, r^3 + a*r^2 + b*r + c = 0 ∧ 0 ≤ r ∧ r ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_cubic_root_in_interval_l2944_294496


namespace NUMINAMATH_CALUDE_symmetric_point_y_axis_l2944_294448

/-- Given a point P(4, -5) and its symmetric point P1 with respect to the y-axis, 
    prove that P1 has coordinates (-4, -5) -/
theorem symmetric_point_y_axis : 
  let P : ℝ × ℝ := (4, -5)
  let P1 : ℝ × ℝ := (-P.1, P.2)  -- Definition of symmetry with respect to y-axis
  P1 = (-4, -5) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_y_axis_l2944_294448


namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_of_20_factorial_l2944_294481

-- Define n as 20!
def n : ℕ := (List.range 20).foldl (· * ·) 1

-- Define the property of k being the greatest integer for which 2^k divides n
def is_greatest_power_of_two_factor (k : ℕ) : Prop :=
  2^k ∣ n ∧ ∀ m : ℕ, 2^m ∣ n → m ≤ k

-- Theorem statement
theorem greatest_power_of_two_factor_of_20_factorial :
  is_greatest_power_of_two_factor 18 :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_of_20_factorial_l2944_294481


namespace NUMINAMATH_CALUDE_sarahs_weeds_total_l2944_294411

theorem sarahs_weeds_total (tuesday : ℕ) (wednesday : ℕ) (thursday : ℕ) (friday : ℕ) : 
  tuesday = 25 →
  wednesday = 3 * tuesday →
  thursday = wednesday / 5 →
  friday = thursday - 10 →
  tuesday + wednesday + thursday + friday = 120 :=
by sorry

end NUMINAMATH_CALUDE_sarahs_weeds_total_l2944_294411


namespace NUMINAMATH_CALUDE_exists_q_and_distance_l2944_294465

/-- Square ABCD with side length 6, point P on AB, and intersecting circles -/
structure SquareWithCircles where
  /-- Side length of the square -/
  side_length : ℝ
  /-- Distance of P from A on side AB -/
  p_distance : ℝ
  /-- Radius of circle centered at P -/
  p_radius : ℝ
  /-- Radius of circle centered at D -/
  d_radius : ℝ
  /-- Assertion that the structure represents a valid configuration -/
  h_side : side_length = 6
  h_p : p_distance = 3
  h_p_radius : p_radius = 3
  h_d_radius : d_radius = 5

/-- Theorem stating the existence of point Q and its distance from BC -/
theorem exists_q_and_distance (s : SquareWithCircles) : 
  ∃ (q : ℝ × ℝ) (dist : ℝ), 
    (q.1 - s.p_distance)^2 + (q.2 - s.side_length)^2 = s.p_radius^2 ∧ 
    q.1^2 + q.2^2 = s.d_radius^2 ∧
    0 ≤ q.1 ∧ q.1 ≤ s.side_length ∧
    0 ≤ q.2 ∧ q.2 ≤ s.side_length ∧
    dist = q.2 := by
  sorry

end NUMINAMATH_CALUDE_exists_q_and_distance_l2944_294465


namespace NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l2944_294429

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 13) : 
  x^3 + y^3 = 35 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l2944_294429


namespace NUMINAMATH_CALUDE_target_hitting_probability_l2944_294457

theorem target_hitting_probability (miss_prob : ℝ) (hit_prob : ℝ) :
  miss_prob = 0.20 →
  hit_prob = 1 - miss_prob →
  hit_prob = 0.80 :=
by
  sorry

end NUMINAMATH_CALUDE_target_hitting_probability_l2944_294457


namespace NUMINAMATH_CALUDE_plate_arrangement_theorem_l2944_294460

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def circular_permutations (n : ℕ) (groups : List ℕ) : ℕ :=
  factorial (n - 1) / (groups.map factorial).prod

theorem plate_arrangement_theorem : 
  let total_plates := 14
  let blue_plates := 6
  let red_plates := 3
  let green_plates := 3
  let orange_plates := 2
  let total_arrangements := circular_permutations total_plates [blue_plates, red_plates, green_plates, orange_plates]
  let adjacent_green_arrangements := circular_permutations (total_plates - green_plates + 1) [blue_plates, red_plates, 1, orange_plates]
  total_arrangements - adjacent_green_arrangements = 1349070 := by
  sorry

end NUMINAMATH_CALUDE_plate_arrangement_theorem_l2944_294460


namespace NUMINAMATH_CALUDE_problem_statement_l2944_294428

theorem problem_statement (x y : ℝ) 
  (hx : x = 1 / (Real.sqrt 2 + 1)) 
  (hy : y = 1 / (Real.sqrt 2 - 1)) : 
  x^2 - 3*x*y + y^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2944_294428


namespace NUMINAMATH_CALUDE_product_zero_l2944_294425

theorem product_zero (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_zero_l2944_294425


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2944_294413

/-- Arithmetic sequence with first term 20 and common difference -2 -/
def arithmetic_sequence (n : ℕ) : ℤ := 20 - 2 * (n - 1)

/-- Sum of first n terms of the arithmetic sequence -/
def sum_arithmetic_sequence (n : ℕ) : ℤ := -n^2 + 21*n

theorem arithmetic_sequence_properties :
  ∀ n : ℕ,
  (arithmetic_sequence n = -2*n + 22) ∧
  (sum_arithmetic_sequence n = -n^2 + 21*n) ∧
  (∀ k : ℕ, sum_arithmetic_sequence k ≤ 110) ∧
  (∃ m : ℕ, sum_arithmetic_sequence m = 110) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2944_294413


namespace NUMINAMATH_CALUDE_restaurant_bill_proof_l2944_294402

theorem restaurant_bill_proof : 
  ∀ (n : ℕ) (total_friends : ℕ) (paying_friends : ℕ) (extra_amount : ℕ),
    total_friends = 10 →
    paying_friends = 9 →
    extra_amount = 3 →
    n = (paying_friends * (n / total_friends + extra_amount)) →
    n = 270 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_proof_l2944_294402


namespace NUMINAMATH_CALUDE_pentagon_one_side_theorem_l2944_294420

/-- A non-self-intersecting pentagon in 2D space -/
structure Pentagon where
  vertices : Fin 5 → ℝ × ℝ
  non_self_intersecting : sorry  -- Condition for non-self-intersection

/-- Half-plane defined by a line segment -/
def HalfPlane (a b : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | sorry}  -- Definition of half-plane

/-- Theorem: For any non-self-intersecting pentagon, there exists a side such that
    all vertices lie in the same half-plane defined by that side -/
theorem pentagon_one_side_theorem (p : Pentagon) :
  ∃ (i : Fin 5), ∀ (j : Fin 5),
    p.vertices j ∈ HalfPlane (p.vertices i) (p.vertices ((i + 1) % 5)) :=
by sorry

end NUMINAMATH_CALUDE_pentagon_one_side_theorem_l2944_294420


namespace NUMINAMATH_CALUDE_graphs_intersection_l2944_294493

/-- 
Given non-zero real numbers k and b, this theorem states that the graphs of 
y = kx + b and y = kb/x can only intersect in the first and third quadrants when kb > 0.
-/
theorem graphs_intersection (k b : ℝ) (hk : k ≠ 0) (hb : b ≠ 0) :
  (∀ x y : ℝ, y = k * x + b ∧ y = k * b / x → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) ↔ k * b > 0 :=
by sorry

end NUMINAMATH_CALUDE_graphs_intersection_l2944_294493


namespace NUMINAMATH_CALUDE_probability_sum_20_l2944_294433

/-- A die is represented as a finite set of natural numbers from 1 to 12 -/
def TwelveSidedDie : Finset ℕ := Finset.range 12 

/-- The sum of two dice rolls -/
def DiceSum (roll1 roll2 : ℕ) : ℕ := roll1 + roll2

/-- The set of all possible outcomes when rolling two dice -/
def AllOutcomes : Finset (ℕ × ℕ) := TwelveSidedDie.product TwelveSidedDie

/-- The set of favorable outcomes (sum of 20) -/
def FavorableOutcomes : Finset (ℕ × ℕ) :=
  AllOutcomes.filter (fun p => DiceSum p.1 p.2 = 20)

/-- The probability of rolling a sum of 20 with two twelve-sided dice -/
theorem probability_sum_20 : 
  (FavorableOutcomes.card : ℚ) / AllOutcomes.card = 5 / 144 := by
  sorry


end NUMINAMATH_CALUDE_probability_sum_20_l2944_294433


namespace NUMINAMATH_CALUDE_visitation_problem_l2944_294445

/-- Represents the visitation schedule of a friend --/
structure VisitSchedule where
  period : ℕ+

/-- Calculates the number of days in a given period when exactly two friends visit --/
def exactlyTwoVisits (alice beatrix claire : VisitSchedule) (totalDays : ℕ) : ℕ :=
  sorry

/-- Theorem statement for the visitation problem --/
theorem visitation_problem :
  let alice : VisitSchedule := ⟨1⟩
  let beatrix : VisitSchedule := ⟨5⟩
  let claire : VisitSchedule := ⟨7⟩
  let totalDays : ℕ := 180
  exactlyTwoVisits alice beatrix claire totalDays = 51 := by sorry

end NUMINAMATH_CALUDE_visitation_problem_l2944_294445


namespace NUMINAMATH_CALUDE_smallest_multiple_of_1_to_10_l2944_294482

def is_multiple_of_all (n : ℕ) : Prop :=
  ∀ i : ℕ, 1 ≤ i → i ≤ 10 → n % i = 0

theorem smallest_multiple_of_1_to_10 :
  ∃ (n : ℕ), n > 0 ∧ is_multiple_of_all n ∧ ∀ m : ℕ, m > 0 → is_multiple_of_all m → n ≤ m :=
by
  use 2520
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_1_to_10_l2944_294482


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l2944_294479

theorem hyperbola_asymptote_angle (c d : ℝ) (h1 : c > d) (h2 : c > 0) (h3 : d > 0) :
  (∀ x y : ℝ, x^2 / c^2 - y^2 / d^2 = 1) →
  (Real.arctan (d / c) - Real.arctan (-d / c) = π / 4) →
  c / d = 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l2944_294479


namespace NUMINAMATH_CALUDE_syllogism_cos_periodic_l2944_294487

-- Define the properties
def IsTrigonometric (f : ℝ → ℝ) : Prop := sorry
def IsPeriodic (f : ℝ → ℝ) : Prop := sorry

-- Define the cosine function
def cos : ℝ → ℝ := sorry

-- Theorem to prove
theorem syllogism_cos_periodic :
  (∀ f : ℝ → ℝ, IsTrigonometric f → IsPeriodic f) →
  (IsTrigonometric cos) →
  (IsPeriodic cos) := by sorry

end NUMINAMATH_CALUDE_syllogism_cos_periodic_l2944_294487


namespace NUMINAMATH_CALUDE_range_of_f_l2944_294467

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then |x| - 1 else Real.sin x ^ 2

theorem range_of_f :
  Set.range f = Set.Ioi (-1) := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2944_294467


namespace NUMINAMATH_CALUDE_plant_supplier_earnings_l2944_294453

theorem plant_supplier_earnings :
  let orchid_count : ℕ := 20
  let orchid_price : ℕ := 50
  let money_plant_count : ℕ := 15
  let money_plant_price : ℕ := 25
  let worker_count : ℕ := 2
  let worker_pay : ℕ := 40
  let pot_cost : ℕ := 150
  let total_earnings := orchid_count * orchid_price + money_plant_count * money_plant_price
  let total_expenses := worker_count * worker_pay + pot_cost
  total_earnings - total_expenses = 1145 :=
by
  sorry

#check plant_supplier_earnings

end NUMINAMATH_CALUDE_plant_supplier_earnings_l2944_294453


namespace NUMINAMATH_CALUDE_coffee_cost_calculation_l2944_294491

/-- Proves that the cost of the second coffee is $2.50 given the conditions --/
theorem coffee_cost_calculation (daily_coffees : ℕ) (espresso_cost : ℚ) (days : ℕ) (total_spent : ℚ) :
  daily_coffees = 2 →
  espresso_cost = 3 →
  days = 20 →
  total_spent = 110 →
  ∃ (iced_coffee_cost : ℚ), iced_coffee_cost = (5/2 : ℚ) ∧ 
    days * (espresso_cost + iced_coffee_cost) = total_spent :=
by
  sorry

end NUMINAMATH_CALUDE_coffee_cost_calculation_l2944_294491


namespace NUMINAMATH_CALUDE_union_with_complement_l2944_294488

theorem union_with_complement (I A B : Set ℕ) : 
  I = {1, 2, 3, 4} →
  A = {1} →
  B = {2, 4} →
  A ∪ (I \ B) = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_with_complement_l2944_294488


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_length_l2944_294404

/-- Represents an isosceles right triangle -/
structure IsoscelesRightTriangle where
  /-- The length of a leg of the triangle -/
  leg : ℝ
  /-- The length of the hypotenuse of the triangle -/
  hypotenuse : ℝ
  /-- Condition that the hypotenuse is √2 times the leg -/
  hyp_leg_relation : hypotenuse = leg * Real.sqrt 2
  /-- Condition that the leg is positive -/
  leg_pos : leg > 0

/-- The theorem stating that for an isosceles right triangle with area 64, its hypotenuse is 16 -/
theorem isosceles_right_triangle_hypotenuse_length
  (t : IsoscelesRightTriangle)
  (area_eq : t.leg * t.leg / 2 = 64) :
  t.hypotenuse = 16 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_length_l2944_294404


namespace NUMINAMATH_CALUDE_polygon_sides_count_l2944_294452

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The theorem stating that a polygon satisfying the given condition has 9 sides -/
theorem polygon_sides_count : 
  ∃ (n : ℕ), n > 0 ∧ 
  (num_diagonals (2*n) - 2*n) - (num_diagonals n - n) = 99 ∧ 
  n = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l2944_294452


namespace NUMINAMATH_CALUDE_mean_of_playground_counts_l2944_294447

def playground_counts : List ℕ := [6, 12, 1, 12, 7, 3, 8]

theorem mean_of_playground_counts :
  (playground_counts.sum : ℚ) / playground_counts.length = 7 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_playground_counts_l2944_294447


namespace NUMINAMATH_CALUDE_solution_set_f_max_value_g_range_of_m_l2944_294466

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Define the function g
def g (x : ℝ) : ℝ := f x - x^2 + x

-- Theorem for the solution set of f(x) ≥ 1
theorem solution_set_f (x : ℝ) : f x ≥ 1 ↔ x ≥ 1 := by sorry

-- Theorem for the maximum value of g(x)
theorem max_value_g : ∃ (x : ℝ), ∀ (y : ℝ), g y ≤ g x ∧ g x = 5/4 := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) : (∃ (x : ℝ), f x ≥ x^2 - x + m) ↔ m ≤ 5/4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_max_value_g_range_of_m_l2944_294466


namespace NUMINAMATH_CALUDE_matt_first_quarter_score_l2944_294463

/-- Calculates the total score in basketball given the number of 2-point and 3-point shots made. -/
def totalScore (twoPointShots threePointShots : ℕ) : ℕ :=
  2 * twoPointShots + 3 * threePointShots

/-- Proves that Matt's score in the first quarter is 14 points. -/
theorem matt_first_quarter_score :
  totalScore 4 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_matt_first_quarter_score_l2944_294463


namespace NUMINAMATH_CALUDE_divisibility_and_infinite_pairs_l2944_294401

theorem divisibility_and_infinite_pairs (c d : ℤ) :
  (∃ f : ℕ → ℤ × ℤ, Function.Injective f ∧
    ∀ n, (f n).1 ∣ (c * (f n).2 + d) ∧ (f n).2 ∣ (c * (f n).1 + d)) ↔
  c ∣ d :=
by sorry

end NUMINAMATH_CALUDE_divisibility_and_infinite_pairs_l2944_294401


namespace NUMINAMATH_CALUDE_area_FGCD_l2944_294461

/-- Represents a trapezoid ABCD with the given properties -/
structure Trapezoid where
  ab : ℝ
  cd : ℝ
  altitude : ℝ
  ab_positive : 0 < ab
  cd_positive : 0 < cd
  altitude_positive : 0 < altitude

/-- Theorem stating the area of quadrilateral FGCD in the given trapezoid -/
theorem area_FGCD (t : Trapezoid) (h1 : t.ab = 10) (h2 : t.cd = 26) (h3 : t.altitude = 15) :
  let fg := (t.ab + t.cd) / 2 - 5 / 2
  (fg + t.cd) / 2 * t.altitude = 311.25 := by sorry

end NUMINAMATH_CALUDE_area_FGCD_l2944_294461


namespace NUMINAMATH_CALUDE_correct_result_largest_negative_integer_result_l2944_294471

/-- Given polynomial A -/
def A (x : ℝ) : ℝ := 3 * x^2 - x + 1

/-- Given polynomial B -/
def B (x : ℝ) : ℝ := -x^2 - 2*x - 3

/-- Theorem stating the correct result of A - B -/
theorem correct_result (x : ℝ) : A x - B x = 4 * x^2 + x + 4 := by sorry

/-- Theorem stating the value of A - B when x is the largest negative integer -/
theorem largest_negative_integer_result : A (-1) - B (-1) = 7 := by sorry

end NUMINAMATH_CALUDE_correct_result_largest_negative_integer_result_l2944_294471


namespace NUMINAMATH_CALUDE_custom_op_difference_l2944_294455

-- Define the custom operation
def customOp (x y : ℝ) : ℝ := x * y - 3 * x

-- State the theorem
theorem custom_op_difference : (customOp 7 4) - (customOp 4 7) = -9 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_difference_l2944_294455


namespace NUMINAMATH_CALUDE_percent_of_y_l2944_294480

theorem percent_of_y (y : ℝ) (h : y > 0) : ((7 * y) / 20 + (3 * y) / 10) / y = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_y_l2944_294480


namespace NUMINAMATH_CALUDE_trailing_zeroes_500_factorial_l2944_294438

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeroes in 500! is 124 -/
theorem trailing_zeroes_500_factorial :
  trailingZeroes 500 = 124 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeroes_500_factorial_l2944_294438


namespace NUMINAMATH_CALUDE_no_triangle_exists_l2944_294443

-- Define the triangle
structure Triangle :=
  (a b : ℝ)
  (angleBisector : ℝ)

-- Define the conditions
def triangleConditions (t : Triangle) : Prop :=
  t.a = 12 ∧ t.b = 20 ∧ t.angleBisector = 15

-- Theorem statement
theorem no_triangle_exists :
  ¬ ∃ (t : Triangle), triangleConditions t ∧ 
  ∃ (c : ℝ), c > 0 ∧ 
  (c + t.a > t.b) ∧ (c + t.b > t.a) ∧ (t.a + t.b > c) ∧
  t.angleBisector = Real.sqrt (t.a * t.b * (1 - (c^2 / (t.a + t.b)^2))) :=
sorry

end NUMINAMATH_CALUDE_no_triangle_exists_l2944_294443


namespace NUMINAMATH_CALUDE_triplet_transformation_theorem_l2944_294442

/-- Represents a triplet of integers -/
structure Triplet where
  a : Int
  b : Int
  c : Int

/-- Represents an operation on a triplet -/
inductive Operation
  | IncrementA (k : Int) (i : Fin 3) : Operation
  | DecrementA (k : Int) (i : Fin 3) : Operation
  | IncrementB (k : Int) (i : Fin 3) : Operation
  | DecrementB (k : Int) (i : Fin 3) : Operation
  | IncrementC (k : Int) (i : Fin 3) : Operation
  | DecrementC (k : Int) (i : Fin 3) : Operation

/-- Applies an operation to a triplet -/
def applyOperation (t : Triplet) (op : Operation) : Triplet :=
  match op with
  | Operation.IncrementA k i => { t with a := t.a + k * (if i = 0 then t.a else if i = 1 then t.b else t.c) }
  | Operation.DecrementA k i => { t with a := t.a - k * (if i = 0 then t.a else if i = 1 then t.b else t.c) }
  | Operation.IncrementB k i => { t with b := t.b + k * (if i = 0 then t.a else if i = 1 then t.b else t.c) }
  | Operation.DecrementB k i => { t with b := t.b - k * (if i = 0 then t.a else if i = 1 then t.b else t.c) }
  | Operation.IncrementC k i => { t with c := t.c + k * (if i = 0 then t.a else if i = 1 then t.b else t.c) }
  | Operation.DecrementC k i => { t with c := t.c - k * (if i = 0 then t.a else if i = 1 then t.b else t.c) }

theorem triplet_transformation_theorem (a b c : Int) (h : Int.gcd a (Int.gcd b c) = 1) :
  ∃ (ops : List Operation), ops.length ≤ 5 ∧
    (ops.foldl applyOperation (Triplet.mk a b c) = Triplet.mk 1 0 0) := by
  sorry

end NUMINAMATH_CALUDE_triplet_transformation_theorem_l2944_294442


namespace NUMINAMATH_CALUDE_semicircle_radius_l2944_294437

theorem semicircle_radius (x y z : ℝ) (h_right_angle : x^2 + y^2 = z^2)
  (h_xy_area : π * x^2 / 2 = 12.5 * π) (h_xz_arc : π * y = 9 * π) :
  z / 2 = Real.sqrt 424 / 2 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_l2944_294437
