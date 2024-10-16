import Mathlib

namespace NUMINAMATH_CALUDE_min_value_of_z_l1670_167080

theorem min_value_of_z (x y : ℝ) (h : x^2 + 2*x*y - 3*y^2 = 1) : 
  ∃ (z_min : ℝ), z_min = (1 + Real.sqrt 5) / 4 ∧ ∀ z, z = x^2 + y^2 → z ≥ z_min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_z_l1670_167080


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1670_167096

/-- The lateral surface area of a cone with base radius 3 and height 4 is 15π. -/
theorem cone_lateral_surface_area :
  let r : ℝ := 3  -- base radius
  let h : ℝ := 4  -- height
  let l : ℝ := (r^2 + h^2).sqrt  -- slant height
  let S : ℝ := π * r * l  -- lateral surface area formula
  S = 15 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1670_167096


namespace NUMINAMATH_CALUDE_first_month_sale_l1670_167093

def sales_data : List ℕ := [6927, 6855, 7230, 6562]
def required_sixth_month_sale : ℕ := 5591
def target_average : ℕ := 6600
def num_months : ℕ := 6

theorem first_month_sale (sales : List ℕ) (sixth_sale target_avg n_months : ℕ)
  (h1 : sales = sales_data)
  (h2 : sixth_sale = required_sixth_month_sale)
  (h3 : target_avg = target_average)
  (h4 : n_months = num_months) :
  ∃ (first_sale : ℕ), 
    (first_sale + sales.sum + sixth_sale) / n_months = target_avg ∧ 
    first_sale = 6435 := by
  sorry

end NUMINAMATH_CALUDE_first_month_sale_l1670_167093


namespace NUMINAMATH_CALUDE_our_system_is_linear_l1670_167040

/-- Represents a linear equation in two variables -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  -- ax + by = c

/-- Represents a system of two equations -/
structure EquationSystem where
  eq1 : LinearEquation
  eq2 : LinearEquation

/-- Checks if an equation is linear -/
def isLinear (eq : LinearEquation) : Prop :=
  eq.a ≠ 0 ∨ eq.b ≠ 0

/-- Checks if a system consists of two linear equations -/
def isSystemOfTwoLinearEquations (sys : EquationSystem) : Prop :=
  isLinear sys.eq1 ∧ isLinear sys.eq2

/-- The specific system we want to prove is a system of two linear equations -/
def ourSystem : EquationSystem :=
  { eq1 := { a := 1, b := 1, c := 5 }  -- x + y = 5
    eq2 := { a := 0, b := 1, c := 2 }  -- y = 2
  }

/-- Theorem stating that our system is a system of two linear equations -/
theorem our_system_is_linear : isSystemOfTwoLinearEquations ourSystem := by
  sorry


end NUMINAMATH_CALUDE_our_system_is_linear_l1670_167040


namespace NUMINAMATH_CALUDE_swimming_pool_count_l1670_167018

theorem swimming_pool_count (total : ℕ) (garage : ℕ) (both : ℕ) (neither : ℕ) : 
  total = 90 → garage = 50 → both = 35 → neither = 35 → 
  ∃ (pool : ℕ), pool = 40 ∧ 
    total = garage + pool - both + neither :=
by
  sorry

end NUMINAMATH_CALUDE_swimming_pool_count_l1670_167018


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1670_167004

theorem quadratic_real_roots (a b c : ℤ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ x : ℝ, x^2 + 2*x*(a : ℝ) + 3*((b : ℝ) + (c : ℝ)) = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1670_167004


namespace NUMINAMATH_CALUDE_sol_earnings_l1670_167039

/-- Calculates the earnings from selling candy bars over a week -/
def candy_bar_earnings (initial_sales : ℕ) (daily_increase : ℕ) (days : ℕ) (price_cents : ℕ) : ℚ :=
  let total_bars := (List.range days).map (λ i => initial_sales + i * daily_increase) |>.sum
  (total_bars * price_cents : ℚ) / 100

/-- Theorem stating that Sol's earnings from selling candy bars over a week is $12.00 -/
theorem sol_earnings : candy_bar_earnings 10 4 6 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sol_earnings_l1670_167039


namespace NUMINAMATH_CALUDE_puppies_brought_in_puppies_brought_in_solution_l1670_167034

theorem puppies_brought_in (initial_puppies : ℕ) (adoption_rate : ℕ) (adoption_days : ℕ) : ℕ :=
  let total_adopted := adoption_rate * adoption_days
  total_adopted - initial_puppies

theorem puppies_brought_in_solution :
  puppies_brought_in 2 4 9 = 34 := by
  sorry

end NUMINAMATH_CALUDE_puppies_brought_in_puppies_brought_in_solution_l1670_167034


namespace NUMINAMATH_CALUDE_probability_of_black_ball_l1670_167061

theorem probability_of_black_ball 
  (p_red : ℝ) 
  (p_white : ℝ) 
  (h1 : p_red = 0.42) 
  (h2 : p_white = 0.28) 
  (h3 : 0 ≤ p_red ∧ p_red ≤ 1) 
  (h4 : 0 ≤ p_white ∧ p_white ≤ 1) : 
  1 - p_red - p_white = 0.30 := by
sorry

end NUMINAMATH_CALUDE_probability_of_black_ball_l1670_167061


namespace NUMINAMATH_CALUDE_arithmetic_mean_odd_primes_under_30_l1670_167036

def odd_primes_under_30 : List Nat := [3, 5, 7, 11, 13, 17, 19, 23, 29]

def arithmetic_mean (list : List Nat) : Rat :=
  (list.sum : Rat) / list.length

theorem arithmetic_mean_odd_primes_under_30 :
  arithmetic_mean odd_primes_under_30 = 14 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_odd_primes_under_30_l1670_167036


namespace NUMINAMATH_CALUDE_optimal_price_and_units_l1670_167050

-- Define the problem parameters
def initial_cost : ℝ := 40
def initial_price : ℝ := 50
def initial_units : ℝ := 500
def price_range_low : ℝ := 50
def price_range_high : ℝ := 70
def target_profit : ℝ := 8000

-- Define the price-demand relationship
def units_sold (price : ℝ) : ℝ :=
  initial_units - 10 * (price - initial_price)

-- Define the profit function
def profit (price : ℝ) : ℝ :=
  (price - initial_cost) * units_sold price

-- State the theorem
theorem optimal_price_and_units :
  ∃ (price : ℝ) (units : ℝ),
    price_range_low ≤ price ∧
    price ≤ price_range_high ∧
    units = units_sold price ∧
    profit price = target_profit ∧
    price = 60 ∧
    units = 400 := by
  sorry

end NUMINAMATH_CALUDE_optimal_price_and_units_l1670_167050


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l1670_167026

theorem quadratic_inequality_empty_solution_set (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 2 ≥ 0) → 0 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l1670_167026


namespace NUMINAMATH_CALUDE_painting_cost_in_cny_l1670_167097

-- Define exchange rates
def usd_to_nad : ℝ := 7
def usd_to_cny : ℝ := 6

-- Define the cost of the painting in Namibian dollars
def painting_cost_nad : ℝ := 105

-- Theorem to prove
theorem painting_cost_in_cny :
  (painting_cost_nad / usd_to_nad) * usd_to_cny = 90 := by
  sorry

end NUMINAMATH_CALUDE_painting_cost_in_cny_l1670_167097


namespace NUMINAMATH_CALUDE_weight_of_b_l1670_167073

theorem weight_of_b (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 46) :
  B = 37 := by sorry

end NUMINAMATH_CALUDE_weight_of_b_l1670_167073


namespace NUMINAMATH_CALUDE_job_completion_theorem_l1670_167042

/-- Represents the job completion problem -/
structure JobCompletion where
  total_days : ℕ
  initial_workers : ℕ
  days_worked : ℕ
  work_completed : ℚ
  workers_fired : ℕ

/-- Calculates the remaining days to complete the job -/
def remaining_days (job : JobCompletion) : ℕ :=
  sorry

/-- Theorem stating that for the given conditions, the remaining work will be completed in 75 days -/
theorem job_completion_theorem (job : JobCompletion) 
  (h1 : job.total_days = 100)
  (h2 : job.initial_workers = 10)
  (h3 : job.days_worked = 20)
  (h4 : job.work_completed = 1/4)
  (h5 : job.workers_fired = 2) :
  remaining_days job = 75 := by sorry

end NUMINAMATH_CALUDE_job_completion_theorem_l1670_167042


namespace NUMINAMATH_CALUDE_opposite_number_theorem_l1670_167059

theorem opposite_number_theorem (a : ℝ) : (-(-a) = -2) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_number_theorem_l1670_167059


namespace NUMINAMATH_CALUDE_cooking_time_is_five_l1670_167000

def recommended_cooking_time (cooked_time seconds_remaining : ℕ) : ℚ :=
  (cooked_time + seconds_remaining) / 60

theorem cooking_time_is_five :
  recommended_cooking_time 45 255 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cooking_time_is_five_l1670_167000


namespace NUMINAMATH_CALUDE_initial_boarders_count_l1670_167033

theorem initial_boarders_count (B D : ℕ) : 
  (B : ℚ) / D = 2 / 5 →  -- Original ratio
  ((B + 15 : ℚ) / D = 1 / 2) →  -- New ratio after 15 boarders joined
  B = 60 := by
sorry

end NUMINAMATH_CALUDE_initial_boarders_count_l1670_167033


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1670_167057

theorem inequality_solution_set (a : ℝ) : 
  (∀ x, x ∈ Set.Ioo (-1 : ℝ) 2 ↔ |a * x + 2| < 6) → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1670_167057


namespace NUMINAMATH_CALUDE_mary_next_birthday_l1670_167002

/-- Represents the ages of Mary, Sally, and Danielle -/
structure Ages where
  mary : ℝ
  sally : ℝ
  danielle : ℝ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.mary = 1.2 * ages.sally ∧
  ages.sally = 0.6 * ages.danielle ∧
  ages.mary + ages.sally + ages.danielle = 23.2

/-- The theorem to be proved -/
theorem mary_next_birthday (ages : Ages) :
  problem_conditions ages → ⌊ages.mary⌋ + 1 = 8 :=
sorry

end NUMINAMATH_CALUDE_mary_next_birthday_l1670_167002


namespace NUMINAMATH_CALUDE_point_distance_on_line_l1670_167005

/-- Given a line with equation x - 5/2y + 1 = 0 and two points on this line,
    if the x-coordinate of the second point is 1/2 unit more than the x-coordinate of the first point,
    then the difference between their x-coordinates is 1/2. -/
theorem point_distance_on_line (m n a : ℝ) : 
  (m - (5/2) * n + 1 = 0) →  -- First point (m, n) satisfies the line equation
  (m + a - (5/2) * (n + 1) + 1 = 0) →  -- Second point (m + a, n + 1) satisfies the line equation
  (m + a = m + 1/2) →  -- x-coordinate of second point is 1/2 more than first point
  a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_point_distance_on_line_l1670_167005


namespace NUMINAMATH_CALUDE_trapezoid_bases_solutions_l1670_167099

theorem trapezoid_bases_solutions :
  let valid_pair : ℕ × ℕ → Prop := fun (b₁, b₂) =>
    b₁ + b₂ = 60 ∧ 
    b₁ % 9 = 0 ∧ 
    b₂ % 9 = 0 ∧ 
    b₁ > 0 ∧ 
    b₂ > 0 ∧ 
    (60 : ℝ) * (b₁ + b₂) / 2 = 1800
  ∃! (solutions : List (ℕ × ℕ)),
    solutions.length = 3 ∧ 
    ∀ pair, pair ∈ solutions ↔ valid_pair pair :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_bases_solutions_l1670_167099


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1670_167006

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + 2*x - 3 < 0} = {x : ℝ | -3 < x ∧ x < 1} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1670_167006


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_22_l1670_167016

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

theorem smallest_prime_with_digit_sum_22 :
  ∃ (p : ℕ), is_prime p ∧ digit_sum p = 22 ∧
  ∀ (q : ℕ), is_prime q → digit_sum q = 22 → p ≤ q :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_22_l1670_167016


namespace NUMINAMATH_CALUDE_part_one_part_two_l1670_167032

-- Define the propositions p and q
def p (x : ℝ) : Prop := (x + 2) * (x - 6) ≤ 0
def q (m x : ℝ) : Prop := 2 - m ≤ x ∧ x ≤ 2 + m

-- Part 1
theorem part_one (m : ℝ) (h : m > 0) :
  (∀ x, p x → q m x) → m ∈ Set.Ici 4 := by sorry

-- Part 2
theorem part_two (x : ℝ) :
  (p x ∨ q 5 x) ∧ ¬(p x ∧ q 5 x) →
  x ∈ Set.Ioc (-3) (-2) ∪ Set.Ioc 6 7 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1670_167032


namespace NUMINAMATH_CALUDE_tan_sum_given_sin_cos_sum_l1670_167062

theorem tan_sum_given_sin_cos_sum (x y : ℝ) 
  (h1 : Real.sin x + Real.sin y = 5/13)
  (h2 : Real.cos x + Real.cos y = 12/13) : 
  Real.tan x + Real.tan y = 240/119 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_given_sin_cos_sum_l1670_167062


namespace NUMINAMATH_CALUDE_min_value_expression_l1670_167014

theorem min_value_expression (x : ℝ) :
  Real.sqrt (x^2 - 2 * Real.sqrt 3 * abs x + 4) +
  Real.sqrt (x^2 + 2 * Real.sqrt 3 * abs x + 12) ≥ 2 * Real.sqrt 7 ∧
  (Real.sqrt (x^2 - 2 * Real.sqrt 3 * abs x + 4) +
   Real.sqrt (x^2 + 2 * Real.sqrt 3 * abs x + 12) = 2 * Real.sqrt 7 ↔
   x = Real.sqrt 3 / 2 ∨ x = -Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1670_167014


namespace NUMINAMATH_CALUDE_min_value_theorem_l1670_167029

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x + 3 / (x + 1) ≥ 2 * Real.sqrt 3 - 1 ∧
  (x + 3 / (x + 1) = 2 * Real.sqrt 3 - 1 ↔ x = Real.sqrt 3 - 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1670_167029


namespace NUMINAMATH_CALUDE_solve_equation_l1670_167025

theorem solve_equation (X : ℝ) : (X^3).sqrt = 81 * (81^(1/9)) → X = 3^(80/27) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1670_167025


namespace NUMINAMATH_CALUDE_smallest_angle_theorem_l1670_167068

-- Define the equation
def equation (x : ℝ) : Prop :=
  Real.sin (3 * x) * Real.sin (4 * x) = Real.cos (3 * x) * Real.cos (4 * x)

-- Define the theorem
theorem smallest_angle_theorem :
  ∃ (x : ℝ), x > 0 ∧ x < π ∧ equation x ∧
  (∀ (y : ℝ), y > 0 ∧ y < x → ¬equation y) ∧
  x = 90 * (π / 180) / 7 :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_theorem_l1670_167068


namespace NUMINAMATH_CALUDE_range_of_f_on_interval_solution_sets_f_positive_l1670_167085

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a + 1) * x + a

-- Part 1: Range of f(x) when a = 3 on [-1, 3]
theorem range_of_f_on_interval :
  ∀ x ∈ Set.Icc (-1 : ℝ) 3, -1 ≤ f 3 x ∧ f 3 x ≤ 8 :=
sorry

-- Part 2: Solution sets for f(x) > 0
theorem solution_sets_f_positive (a : ℝ) :
  (∀ x, f a x > 0 ↔ 
    (a > 1 ∧ (x < 1 ∨ x > a)) ∨
    (a < 1 ∧ (x < a ∨ x > 1)) ∨
    (a = 1 ∧ x ≠ 1)) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_on_interval_solution_sets_f_positive_l1670_167085


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1670_167011

theorem algebraic_expression_value (m n : ℝ) (h : -2*m + 3*n^2 = -7) : 
  12*n^2 - 8*m + 4 = -24 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1670_167011


namespace NUMINAMATH_CALUDE_touching_values_are_zero_and_neg_four_l1670_167082

/-- Two linear functions with parallel, non-vertical graphs -/
structure ParallelLinearFunctions where
  f : ℝ → ℝ
  g : ℝ → ℝ
  parallel : ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b ∧ g x = a * x + c
  not_vertical : ∃ (a : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + (f 0)

/-- Condition that (f x)^2 touches 4(g x) -/
def touches_squared_to_scaled (p : ParallelLinearFunctions) : Prop :=
  ∃! x, (p.f x)^2 = 4 * (p.g x)

/-- Values of A for which (g x)^2 touches A(f x) -/
def touching_values (p : ParallelLinearFunctions) : Set ℝ :=
  {A | ∃! x, (p.g x)^2 = A * (p.f x)}

/-- Main theorem -/
theorem touching_values_are_zero_and_neg_four 
    (p : ParallelLinearFunctions) 
    (h : touches_squared_to_scaled p) : 
    touching_values p = {0, -4} := by
  sorry


end NUMINAMATH_CALUDE_touching_values_are_zero_and_neg_four_l1670_167082


namespace NUMINAMATH_CALUDE_ninetieth_term_is_13_l1670_167021

def sequence_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem ninetieth_term_is_13 :
  ∃ (seq : ℕ → ℕ),
    (∀ n : ℕ, ∀ k : ℕ, k > sequence_sum n → k ≤ sequence_sum (n + 1) → seq k = n + 1) →
    seq 90 = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_ninetieth_term_is_13_l1670_167021


namespace NUMINAMATH_CALUDE_sum_of_functions_l1670_167066

theorem sum_of_functions (x : ℝ) (hx : x ≠ 2) :
  let f : ℝ → ℝ := λ x => x^2 - 1/(x-2)
  let g : ℝ → ℝ := λ x => 1/(x-2) + 1
  f x + g x = x^2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_functions_l1670_167066


namespace NUMINAMATH_CALUDE_tom_apples_l1670_167028

/-- The number of apples Phillip has -/
def phillip_apples : ℕ := 40

/-- The number of apples Ben has more than Phillip -/
def ben_extra_apples : ℕ := 8

/-- The fraction of Ben's apples that Tom has -/
def tom_fraction : ℚ := 3 / 8

/-- Theorem stating that Tom has 18 apples -/
theorem tom_apples : ℕ := by sorry

end NUMINAMATH_CALUDE_tom_apples_l1670_167028


namespace NUMINAMATH_CALUDE_practice_coincidence_l1670_167094

def trumpet_interval : ℕ := 11
def flute_interval : ℕ := 3

theorem practice_coincidence : Nat.lcm trumpet_interval flute_interval = 33 := by
  sorry

end NUMINAMATH_CALUDE_practice_coincidence_l1670_167094


namespace NUMINAMATH_CALUDE_no_natural_solution_l1670_167044

theorem no_natural_solution :
  ¬∃ (x y : ℕ), x^2 + y^2 + 1 = 6*x*y := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_l1670_167044


namespace NUMINAMATH_CALUDE_expansion_coefficient_remainder_counts_l1670_167008

/-- 
Given a natural number n, Tᵣ(n) represents the number of coefficients in the expansion of (1+x)ⁿ 
that give a remainder of r when divided by 3, where r ∈ {0,1,2}.
-/
def T (r n : ℕ) : ℕ := sorry

/-- The theorem states the values of T₀(2006), T₁(2006), and T₂(2006) for the expansion of (1+x)²⁰⁰⁶. -/
theorem expansion_coefficient_remainder_counts : 
  T 0 2006 = 1764 ∧ T 1 2006 = 122 ∧ T 2 2006 = 121 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_remainder_counts_l1670_167008


namespace NUMINAMATH_CALUDE_class_funds_calculation_l1670_167067

/-- Proves that the class funds amount to $14 given the problem conditions -/
theorem class_funds_calculation (total_contribution student_count student_contribution : ℕ) 
  (h1 : total_contribution = 90)
  (h2 : student_count = 19)
  (h3 : student_contribution = 4) :
  total_contribution - (student_count * student_contribution) = 14 := by
  sorry

#check class_funds_calculation

end NUMINAMATH_CALUDE_class_funds_calculation_l1670_167067


namespace NUMINAMATH_CALUDE_no_real_roots_condition_implies_inequality_g_no_intersect_l1670_167030

/-- A quadratic function that doesn't intersect with y = x -/
structure NoIntersectQuadratic where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  no_intersect : ∀ x : ℝ, a * x^2 + b * x + c ≠ x

def f (q : NoIntersectQuadratic) (x : ℝ) : ℝ := q.a * x^2 + q.b * x + q.c

theorem no_real_roots (q : NoIntersectQuadratic) : ∀ x : ℝ, f q (f q x) ≠ x := by sorry

theorem condition_implies_inequality (q : NoIntersectQuadratic) (h : q.a + q.b + q.c = 0) :
  ∀ x : ℝ, f q (f q x) < x := by sorry

def g (q : NoIntersectQuadratic) (x : ℝ) : ℝ := q.a * x^2 - q.b * x + q.c

theorem g_no_intersect (q : NoIntersectQuadratic) : ∀ x : ℝ, g q x ≠ -x := by sorry

end NUMINAMATH_CALUDE_no_real_roots_condition_implies_inequality_g_no_intersect_l1670_167030


namespace NUMINAMATH_CALUDE_equation_solution_l1670_167053

theorem equation_solution (k : ℝ) : 
  (∀ x : ℝ, -x^2 - (k + 7)*x - 8 = -(x - 2)*(x - 4)) ↔ k = -13 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1670_167053


namespace NUMINAMATH_CALUDE_car_travel_time_l1670_167071

theorem car_travel_time (distance : ℝ) (new_speed : ℝ) (time_ratio : ℝ) (t : ℝ) 
  (h1 : distance = 630)
  (h2 : new_speed = 70)
  (h3 : time_ratio = 3/2)
  (h4 : distance = (distance / t) * (time_ratio * t))
  (h5 : distance = new_speed * (time_ratio * t)) :
  t = 6 := by
sorry

end NUMINAMATH_CALUDE_car_travel_time_l1670_167071


namespace NUMINAMATH_CALUDE_kimberly_peanuts_l1670_167078

/-- The number of times Kimberly went to the store last month -/
def store_visits : ℕ := 3

/-- The number of peanuts Kimberly buys each time she goes to the store -/
def peanuts_per_visit : ℕ := 7

/-- The total number of peanuts Kimberly bought last month -/
def total_peanuts : ℕ := store_visits * peanuts_per_visit

theorem kimberly_peanuts : total_peanuts = 21 := by
  sorry

end NUMINAMATH_CALUDE_kimberly_peanuts_l1670_167078


namespace NUMINAMATH_CALUDE_general_equation_pattern_l1670_167095

theorem general_equation_pattern (n : ℝ) : n ≠ 4 ∧ (8 - n) ≠ 4 →
  n / (n - 4) + (8 - n) / ((8 - n) - 4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_general_equation_pattern_l1670_167095


namespace NUMINAMATH_CALUDE_factor_implies_c_equals_three_l1670_167076

theorem factor_implies_c_equals_three (c : ℝ) : 
  (∀ x : ℝ, (x + 7) ∣ (c * x^3 + 19 * x^2 - 4 * c * x + 20)) → c = 3 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_c_equals_three_l1670_167076


namespace NUMINAMATH_CALUDE_solar_project_analysis_l1670_167089

/-- Represents the net profit of a solar power generation project over n years -/
def net_profit (n : ℕ+) : ℚ :=
  -4 * n^2 + 80 * n - 144

/-- Represents the average annual profit of the project over n years -/
def avg_annual_profit (n : ℕ+) : ℚ :=
  net_profit n / n

theorem solar_project_analysis :
  ∀ n : ℕ+,
  -- 1. Net profit function
  net_profit n = -4 * n^2 + 80 * n - 144 ∧
  -- 2. Project starts making profit from the 3rd year
  (∀ k : ℕ+, k ≥ 3 → net_profit k > 0) ∧
  (∀ k : ℕ+, k < 3 → net_profit k ≤ 0) ∧
  -- 3. Maximum average annual profit occurs when n = 6
  (∀ k : ℕ+, avg_annual_profit k ≤ avg_annual_profit 6) ∧
  -- 4. Maximum net profit occurs when n = 10
  (∀ k : ℕ+, net_profit k ≤ net_profit 10) ∧
  -- 5. Both options result in the same total profit
  net_profit 6 + 72 = net_profit 10 + 8 ∧
  net_profit 6 + 72 = 264 :=
by sorry


end NUMINAMATH_CALUDE_solar_project_analysis_l1670_167089


namespace NUMINAMATH_CALUDE_end_of_week_stock_l1670_167098

def pencils_per_day : ℕ := 100
def working_days_per_week : ℕ := 5
def initial_stock : ℕ := 80
def pencils_sold : ℕ := 350

theorem end_of_week_stock : 
  pencils_per_day * working_days_per_week + initial_stock - pencils_sold = 230 := by
  sorry

end NUMINAMATH_CALUDE_end_of_week_stock_l1670_167098


namespace NUMINAMATH_CALUDE_max_gcd_of_coprime_linear_combination_l1670_167024

theorem max_gcd_of_coprime_linear_combination (m n : ℕ) :
  Nat.gcd m n = 1 →
  ∃ a b : ℕ, Nat.gcd (m + 2000 * n) (n + 2000 * m) = 2000^2 - 1 ∧
            ∀ c d : ℕ, Nat.gcd (c + 2000 * d) (d + 2000 * c) ≤ 2000^2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_of_coprime_linear_combination_l1670_167024


namespace NUMINAMATH_CALUDE_student_pairs_l1670_167051

theorem student_pairs (n : ℕ) (h : n = 12) : (n.choose 2) = 66 := by
  sorry

end NUMINAMATH_CALUDE_student_pairs_l1670_167051


namespace NUMINAMATH_CALUDE_sweets_problem_l1670_167038

/-- The number of sweets initially on the table -/
def initial_sweets : ℕ := 50

/-- The number of sweets Jack took -/
def jack_sweets (total : ℕ) : ℕ := total / 2 + 4

/-- The number of sweets remaining after Jack -/
def after_jack (total : ℕ) : ℕ := total - jack_sweets total

/-- The number of sweets Paul took -/
def paul_sweets (remaining : ℕ) : ℕ := remaining / 3 + 5

/-- The number of sweets remaining after Paul -/
def after_paul (remaining : ℕ) : ℕ := remaining - paul_sweets remaining

/-- Olivia took the last 9 sweets -/
def olivia_sweets : ℕ := 9

theorem sweets_problem :
  after_paul (after_jack initial_sweets) = olivia_sweets :=
sorry

end NUMINAMATH_CALUDE_sweets_problem_l1670_167038


namespace NUMINAMATH_CALUDE_polynomial_product_no_x4_x3_terms_l1670_167083

theorem polynomial_product_no_x4_x3_terms :
  let P (x : ℝ) := 2 * x^3 - 5 * x^2 + 7 * x - 8
  let Q (x : ℝ) := a * x^2 + b * x + 11
  (∀ x, (P x) * (Q x) = 8 * x^5 - 17 * x^2 - 3 * x - 88) →
  a = 4 ∧ b = 10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_product_no_x4_x3_terms_l1670_167083


namespace NUMINAMATH_CALUDE_jays_change_is_twenty_l1670_167012

/-- The change Jay received after purchasing items and paying with a fifty-dollar bill -/
def jays_change (book_price pen_price ruler_price paid_amount : ℕ) : ℕ :=
  paid_amount - (book_price + pen_price + ruler_price)

/-- Theorem stating that Jay's change is $20 given the specific prices and payment amount -/
theorem jays_change_is_twenty :
  jays_change 25 4 1 50 = 20 := by
  sorry

end NUMINAMATH_CALUDE_jays_change_is_twenty_l1670_167012


namespace NUMINAMATH_CALUDE_yard_length_26_trees_l1670_167019

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (tree_distance : ℝ) : ℝ :=
  (num_trees - 1) * tree_distance

/-- Theorem: The length of a yard with 26 equally spaced trees,
    where the distance between consecutive trees is 15 meters, is 375 meters -/
theorem yard_length_26_trees :
  yard_length 26 15 = 375 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_26_trees_l1670_167019


namespace NUMINAMATH_CALUDE_right_rectangular_prism_diagonal_ratio_bound_right_rectangular_prism_diagonal_ratio_bound_tight_l1670_167041

theorem right_rectangular_prism_diagonal_ratio_bound 
  (a b h d : ℝ) (ha : a > 0) (hb : b > 0) (hh : h > 0) 
  (hd : d^2 = a^2 + b^2 + h^2) : 
  (a + b + h) / d ≤ Real.sqrt 3 := by
sorry

theorem right_rectangular_prism_diagonal_ratio_bound_tight : 
  ∃ (a b h d : ℝ), a > 0 ∧ b > 0 ∧ h > 0 ∧ d^2 = a^2 + b^2 + h^2 ∧ 
  (a + b + h) / d = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_diagonal_ratio_bound_right_rectangular_prism_diagonal_ratio_bound_tight_l1670_167041


namespace NUMINAMATH_CALUDE_log_5_125_l1670_167037

-- Define the logarithm function
noncomputable def log (a : ℝ) (N : ℝ) : ℝ :=
  Real.log N / Real.log a

-- Theorem statement
theorem log_5_125 : log 5 125 = 3 := by
  sorry


end NUMINAMATH_CALUDE_log_5_125_l1670_167037


namespace NUMINAMATH_CALUDE_max_xyz_value_l1670_167072

theorem max_xyz_value (x y z : ℝ) (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (eq_cond : (2*x * 2*y) + 3*z = (x + 2*z) * (y + 2*z))
  (sum_cond : x + y + z = 2) :
  x * y * z ≤ 8 / 27 := by
sorry

end NUMINAMATH_CALUDE_max_xyz_value_l1670_167072


namespace NUMINAMATH_CALUDE_farm_problem_l1670_167055

theorem farm_problem :
  ∃ (l g : ℕ), l > 0 ∧ g > 0 ∧ 30 * l + 32 * g = 1200 ∧ l > g :=
by sorry

end NUMINAMATH_CALUDE_farm_problem_l1670_167055


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l1670_167079

/-- A convex nonagon is a 9-sided polygon -/
def ConvexNonagon := Nat

/-- The number of distinct diagonals in a convex nonagon -/
def num_diagonals (n : ConvexNonagon) : Nat :=
  27

theorem nonagon_diagonals :
  ∀ n : ConvexNonagon, num_diagonals n = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l1670_167079


namespace NUMINAMATH_CALUDE_series_convergence_implies_scaled_convergence_l1670_167013

theorem series_convergence_implies_scaled_convergence 
  (a : ℕ → ℝ) (h : Summable a) : Summable (fun n => a n / n) := by
  sorry

end NUMINAMATH_CALUDE_series_convergence_implies_scaled_convergence_l1670_167013


namespace NUMINAMATH_CALUDE_lizzy_money_after_loan_l1670_167091

def calculate_final_amount (initial_amount : ℝ) (loan_amount : ℝ) (interest_rate : ℝ) : ℝ :=
  initial_amount - loan_amount + loan_amount * (1 + interest_rate)

theorem lizzy_money_after_loan (initial_amount loan_amount interest_rate : ℝ) 
  (h1 : initial_amount = 30)
  (h2 : loan_amount = 15)
  (h3 : interest_rate = 0.2) :
  calculate_final_amount initial_amount loan_amount interest_rate = 33 := by
  sorry

end NUMINAMATH_CALUDE_lizzy_money_after_loan_l1670_167091


namespace NUMINAMATH_CALUDE_greg_age_l1670_167003

/-- Given the ages and relationships of Cindy, Jan, Marcia, and Greg, prove Greg's age. -/
theorem greg_age (cindy_age : ℕ) (jan_age : ℕ) (marcia_age : ℕ) (greg_age : ℕ)
  (h1 : cindy_age = 5)
  (h2 : jan_age = cindy_age + 2)
  (h3 : marcia_age = 2 * jan_age)
  (h4 : greg_age = marcia_age + 2) :
  greg_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_greg_age_l1670_167003


namespace NUMINAMATH_CALUDE_rectangle_area_l1670_167017

theorem rectangle_area (width : ℝ) (length : ℝ) : 
  length = 4 * width → 
  2 * length + 2 * width = 200 → 
  length * width = 1600 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1670_167017


namespace NUMINAMATH_CALUDE_evelyns_bottle_caps_l1670_167086

/-- The problem of Evelyn's bottle caps -/
theorem evelyns_bottle_caps
  (initial : ℕ)            -- Initial number of bottle caps
  (found : ℕ)              -- Number of bottle caps found
  (total : ℕ)              -- Total number of bottle caps at the end
  (h1 : found = 63)        -- Evelyn found 63 bottle caps
  (h2 : total = 81)        -- Evelyn ended up with 81 bottle caps in total
  (h3 : total = initial + found) -- The total is the sum of initial and found bottle caps
  : initial = 18 :=
by sorry

end NUMINAMATH_CALUDE_evelyns_bottle_caps_l1670_167086


namespace NUMINAMATH_CALUDE_not_divisible_by_product_l1670_167049

theorem not_divisible_by_product (a₁ a₂ b₁ b₂ : ℕ) 
  (h1 : 1 < b₁) (h2 : b₁ < a₁) (h3 : 1 < b₂) (h4 : b₂ < a₂) 
  (h5 : b₁ ∣ a₁) (h6 : b₂ ∣ a₂) : 
  ¬(a₁ * a₂ ∣ a₁ * b₁ + a₂ * b₂ - 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_product_l1670_167049


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1670_167047

theorem simplify_and_evaluate (x y : ℝ) (h : x / y = 3) :
  (1 + y^2 / (x^2 - y^2)) * ((x - y) / x) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1670_167047


namespace NUMINAMATH_CALUDE_count_integer_ratios_eq_five_l1670_167007

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  sorry

/-- Given two arithmetic sequences and a property of their sums,
    counts the number of positive integers that make the ratio of their terms an integer -/
def count_integer_ratios (a b : ArithmeticSequence) : ℕ :=
  sorry

theorem count_integer_ratios_eq_five
  (a b : ArithmeticSequence)
  (h : ∀ n : ℕ+, sum_n a n / sum_n b n = (7 * n + 45) / (n + 3)) :
  count_integer_ratios a b = 5 :=
sorry

end NUMINAMATH_CALUDE_count_integer_ratios_eq_five_l1670_167007


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1670_167064

/-- Given an arithmetic sequence where the eighth term is 20 and the common difference is 3,
    prove that the sum of the first three terms is 6. -/
theorem arithmetic_sequence_sum (a : ℕ → ℤ) :
  (∀ n, a (n + 1) - a n = 3) →  -- Common difference is 3
  a 8 = 20 →                   -- Eighth term is 20
  a 1 + a 2 + a 3 = 6 :=        -- Sum of first three terms is 6
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1670_167064


namespace NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l1670_167087

theorem product_of_sum_and_cube_sum (p q : ℝ) 
  (h1 : p + q = 10) 
  (h2 : p^3 + q^3 = 370) : 
  p * q = 21 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l1670_167087


namespace NUMINAMATH_CALUDE_jenny_jump_distance_l1670_167027

/-- The sum of the first n terms of a geometric series with first term a and common ratio r -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The number of jumps Jenny makes -/
def num_jumps : ℕ := 7

/-- The fraction of remaining distance Jenny jumps each time -/
def jump_fraction : ℚ := 1/4

/-- The common ratio of the geometric series representing Jenny's jumps -/
def common_ratio : ℚ := 1 - jump_fraction

theorem jenny_jump_distance :
  geometric_sum jump_fraction common_ratio num_jumps = 14197/16384 := by
  sorry

end NUMINAMATH_CALUDE_jenny_jump_distance_l1670_167027


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_150_l1670_167065

theorem closest_integer_to_cube_root_150 : 
  ∀ n : ℤ, |n - (150 : ℝ)^(1/3)| ≥ |5 - (150 : ℝ)^(1/3)| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_150_l1670_167065


namespace NUMINAMATH_CALUDE_sum_even_divisors_140_l1670_167070

/-- Sum of even positive divisors of a natural number n -/
def sumEvenDivisors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of all even positive divisors of 140 is 288 -/
theorem sum_even_divisors_140 : sumEvenDivisors 140 = 288 := by sorry

end NUMINAMATH_CALUDE_sum_even_divisors_140_l1670_167070


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1670_167074

theorem quadratic_inequality_solution_set (a : ℝ) :
  let solution_set := {x : ℝ | a * x^2 - 2 * (a - 1) * x + a ≤ 0}
  (a > 1/2 → solution_set = ∅) ∧
  (a = 1/2 → solution_set = {-1}) ∧
  (0 < a ∧ a < 1/2 → solution_set = Set.Icc ((a - 1 - Real.sqrt (1 - 2*a)) / a) ((a - 1 + Real.sqrt (1 - 2*a)) / a)) ∧
  (a = 0 → solution_set = Set.Iic 0) ∧
  (a < 0 → solution_set = Set.Iic ((a - 1 + Real.sqrt (1 - 2*a)) / a) ∪ Set.Ici ((a - 1 - Real.sqrt (1 - 2*a)) / a)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1670_167074


namespace NUMINAMATH_CALUDE_table_capacity_l1670_167031

theorem table_capacity (invited : ℕ) (no_show : ℕ) (tables : ℕ) : 
  invited = 45 → no_show = 35 → tables = 5 → (invited - no_show) / tables = 2 := by
  sorry

end NUMINAMATH_CALUDE_table_capacity_l1670_167031


namespace NUMINAMATH_CALUDE_building_meets_safety_regulations_l1670_167045

/-- Represents the school building configuration and safety requirements -/
structure SchoolBuilding where
  floors : Nat
  classrooms_per_floor : Nat
  main_doors : Nat
  side_doors : Nat
  students_all_doors_2min : Nat
  students_half_doors_4min : Nat
  emergency_efficiency_decrease : Rat
  evacuation_time_limit : Nat
  students_per_classroom : Nat

/-- Calculates the flow rate of students through doors -/
def calculate_flow_rates (building : SchoolBuilding) : Nat × Nat :=
  sorry

/-- Checks if the building meets safety regulations -/
def meets_safety_regulations (building : SchoolBuilding) : Bool :=
  sorry

/-- Theorem stating that the given building configuration meets safety regulations -/
theorem building_meets_safety_regulations :
  let building : SchoolBuilding := {
    floors := 4,
    classrooms_per_floor := 8,
    main_doors := 2,
    side_doors := 2,
    students_all_doors_2min := 560,
    students_half_doors_4min := 800,
    emergency_efficiency_decrease := 1/5,
    evacuation_time_limit := 5,
    students_per_classroom := 45
  }
  meets_safety_regulations building = true :=
sorry

end NUMINAMATH_CALUDE_building_meets_safety_regulations_l1670_167045


namespace NUMINAMATH_CALUDE_parallel_tangents_intersection_l1670_167043

theorem parallel_tangents_intersection (x₀ : ℝ) : 
  (∃ (y₁ y₂ : ℝ), y₁ = x₀^2 - 1 ∧ y₂ = 1 - x₀^3 ∧ 
   (2 * x₀) = -(3 * x₀^2)) → 
  (x₀ = 0 ∨ x₀ = -2/3) := by sorry

end NUMINAMATH_CALUDE_parallel_tangents_intersection_l1670_167043


namespace NUMINAMATH_CALUDE_congruence_system_solution_l1670_167069

theorem congruence_system_solution :
  ∃ x : ℤ, (x ≡ 1 [ZMOD 6] ∧ x ≡ 9 [ZMOD 14] ∧ x ≡ 7 [ZMOD 15]) ↔ x ≡ 37 [ZMOD 210] :=
by sorry

end NUMINAMATH_CALUDE_congruence_system_solution_l1670_167069


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l1670_167023

theorem min_value_expression (a b c d : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0)
  (sum_eq_4 : a + b + c + d = 4) : 
  (a^8 / ((a^2+b)*(a^2+c)*(a^2+d))) + 
  (b^8 / ((b^2+c)*(b^2+d)*(b^2+a))) + 
  (c^8 / ((c^2+d)*(c^2+a)*(c^2+b))) + 
  (d^8 / ((d^2+a)*(d^2+b)*(d^2+c))) ≥ (1/2) := by
  sorry

theorem equality_condition (a b c d : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0)
  (sum_eq_4 : a + b + c + d = 4) :
  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) ↔ 
  (a^8 / ((a^2+b)*(a^2+c)*(a^2+d))) + 
  (b^8 / ((b^2+c)*(b^2+d)*(b^2+a))) + 
  (c^8 / ((c^2+d)*(c^2+a)*(c^2+b))) + 
  (d^8 / ((d^2+a)*(d^2+b)*(d^2+c))) = (1/2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l1670_167023


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1670_167048

/-- Simple interest calculation -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Problem statement -/
theorem interest_rate_calculation (principal time interest : ℝ) 
  (h1 : principal = 500)
  (h2 : time = 4)
  (h3 : interest = 90) :
  ∃ (rate : ℝ), simple_interest principal rate time = interest ∧ rate = 0.045 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l1670_167048


namespace NUMINAMATH_CALUDE_ninas_running_drill_l1670_167058

/-- Nina's running drill problem -/
theorem ninas_running_drill 
  (initial_run : ℝ) 
  (total_distance : ℝ) 
  (h1 : initial_run = 0.08333333333333333)
  (h2 : total_distance = 0.8333333333333334) :
  total_distance - 2 * initial_run = 0.6666666666666667 := by
  sorry

end NUMINAMATH_CALUDE_ninas_running_drill_l1670_167058


namespace NUMINAMATH_CALUDE_age_of_b_l1670_167090

/-- Given three people a, b, and c, prove that if their average age is 25 years
    and the average age of a and c is 29 years, then the age of b is 17 years. -/
theorem age_of_b (a b c : ℕ) : 
  (a + b + c) / 3 = 25 → (a + c) / 2 = 29 → b = 17 := by
  sorry

end NUMINAMATH_CALUDE_age_of_b_l1670_167090


namespace NUMINAMATH_CALUDE_equation_solution_l1670_167063

theorem equation_solution : ∃! x : ℝ, (2 / (x - 1) = 3 / (x - 2)) ∧ (x = -1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1670_167063


namespace NUMINAMATH_CALUDE_scientific_notation_of_56_99_million_l1670_167010

def million : ℝ := 1000000

theorem scientific_notation_of_56_99_million :
  56.99 * million = 5.699 * (10 : ℝ) ^ 7 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_56_99_million_l1670_167010


namespace NUMINAMATH_CALUDE_tenth_term_of_arithmetic_progression_l1670_167035

/-- The nth term of an arithmetic progression -/
def arithmeticProgressionTerm (a : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

/-- Theorem: The 10th term of an arithmetic progression with first term 8 and common difference 2 is 26 -/
theorem tenth_term_of_arithmetic_progression :
  arithmeticProgressionTerm 8 2 10 = 26 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_arithmetic_progression_l1670_167035


namespace NUMINAMATH_CALUDE_tangent_circles_count_l1670_167077

-- Define a type for lines in a plane
structure Line where
  -- Add necessary properties for a line

-- Define a type for circles in a plane
structure Circle where
  -- Add necessary properties for a circle

-- Define a function to check if a circle is tangent to a line
def is_tangent (c : Circle) (l : Line) : Prop :=
  sorry

-- Define a function to count the number of circles tangent to three lines
def count_tangent_circles (l1 l2 l3 : Line) : ℕ :=
  sorry

-- Define predicates for different line configurations
def general_position (l1 l2 l3 : Line) : Prop :=
  sorry

def intersect_at_point (l1 l2 l3 : Line) : Prop :=
  sorry

def all_parallel (l1 l2 l3 : Line) : Prop :=
  sorry

def two_parallel_one_intersecting (l1 l2 l3 : Line) : Prop :=
  sorry

theorem tangent_circles_count 
  (l1 l2 l3 : Line) : 
  (general_position l1 l2 l3 → count_tangent_circles l1 l2 l3 = 4) ∧
  (intersect_at_point l1 l2 l3 → count_tangent_circles l1 l2 l3 = 0) ∧
  (all_parallel l1 l2 l3 → count_tangent_circles l1 l2 l3 = 0) ∧
  (two_parallel_one_intersecting l1 l2 l3 → count_tangent_circles l1 l2 l3 = 2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_circles_count_l1670_167077


namespace NUMINAMATH_CALUDE_javelin_throw_distance_l1670_167022

theorem javelin_throw_distance (first second third : ℝ) 
  (h1 : first = 2 * second)
  (h2 : first = (1 / 2) * third)
  (h3 : first + second + third = 1050) :
  first = 300 := by
  sorry

end NUMINAMATH_CALUDE_javelin_throw_distance_l1670_167022


namespace NUMINAMATH_CALUDE_expression_evaluation_l1670_167009

theorem expression_evaluation : 72 + (120 / 15) + (15 * 12) - 250 - (480 / 8) = -50 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1670_167009


namespace NUMINAMATH_CALUDE_equation_representations_l1670_167001

-- Define the equations
def equation1 (x y : ℝ) : Prop := x * (x^2 + y^2 - 4) = 0
def equation2 (x y : ℝ) : Prop := x^2 + (x^2 + y^2 - 4)^2 = 0

-- Define what it means for an equation to represent a line and a circle
def represents_line_and_circle (f : ℝ → ℝ → Prop) : Prop :=
  (∃ (a : ℝ), ∀ y, f a y) ∧ 
  (∃ (h k r : ℝ), ∀ x y, f x y ↔ (x - h)^2 + (y - k)^2 = r^2)

-- Define what it means for an equation to represent two points
def represents_two_points (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), x1 ≠ x2 ∨ y1 ≠ y2 ∧ 
    (∀ x y, f x y ↔ (x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2))

-- State the theorem
theorem equation_representations : 
  represents_line_and_circle equation1 ∧ represents_two_points equation2 := by
  sorry

end NUMINAMATH_CALUDE_equation_representations_l1670_167001


namespace NUMINAMATH_CALUDE_base_12_remainder_div_7_l1670_167084

-- Define the base-12 number
def base_12_num : ℕ := 2 * 12^3 + 5 * 12^2 + 4 * 12 + 3

-- Theorem statement
theorem base_12_remainder_div_7 : base_12_num % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_base_12_remainder_div_7_l1670_167084


namespace NUMINAMATH_CALUDE_mass_of_compound_l1670_167015

/-- The mass of a compound given its molecular weight and number of moles. -/
def mass (molecular_weight : ℝ) (moles : ℝ) : ℝ :=
  molecular_weight * moles

/-- Theorem: The mass of 7 moles of a compound with a molecular weight of 588 g/mol is 4116 g. -/
theorem mass_of_compound : mass 588 7 = 4116 := by
  sorry

end NUMINAMATH_CALUDE_mass_of_compound_l1670_167015


namespace NUMINAMATH_CALUDE_train_journey_time_l1670_167052

/-- The usual time for a train journey, given reduced speed and delay -/
theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) (h2 : usual_time > 0) : 
  (4 / 5 * usual_speed) * (usual_time + 1 / 2) = usual_speed * usual_time → 
  usual_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_time_l1670_167052


namespace NUMINAMATH_CALUDE_integral_equals_six_implies_b_equals_e_to_four_l1670_167088

theorem integral_equals_six_implies_b_equals_e_to_four (b : ℝ) :
  (∫ (x : ℝ) in Set.Icc (Real.exp 1) b, 2 / x) = 6 →
  b = Real.exp 4 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_six_implies_b_equals_e_to_four_l1670_167088


namespace NUMINAMATH_CALUDE_daily_increase_amount_l1670_167060

def fine_sequence (x : ℚ) : ℕ → ℚ
  | 0 => 0.05
  | n + 1 => min (fine_sequence x n + x) (2 * fine_sequence x n)

theorem daily_increase_amount :
  ∃ x : ℚ, x > 0 ∧ fine_sequence x 4 = 0.70 ∧ 
  ∀ n : ℕ, n > 0 → fine_sequence x n = fine_sequence x (n-1) + x :=
by sorry

end NUMINAMATH_CALUDE_daily_increase_amount_l1670_167060


namespace NUMINAMATH_CALUDE_muffins_for_sale_is_108_l1670_167054

/-- Calculate the number of muffins for sale given the following conditions:
  * 3 boys each make 12 muffins
  * 2 girls each make 20 muffins
  * 1 girl makes 15 muffins
  * 2 boys each make 18 muffins
  * 15% of all muffins will not make it to the sale
-/
def muffinsForSale : ℕ :=
  let boys_group1 := 3 * 12
  let boys_group2 := 2 * 18
  let girls_group1 := 2 * 20
  let girls_group2 := 1 * 15
  let total_muffins := boys_group1 + boys_group2 + girls_group1 + girls_group2
  let muffins_not_for_sale := (total_muffins : ℚ) * (15 : ℚ) / (100 : ℚ)
  ⌊(total_muffins : ℚ) - muffins_not_for_sale⌋.toNat

/-- Theorem stating that the number of muffins for sale is 108 -/
theorem muffins_for_sale_is_108 : muffinsForSale = 108 := by
  sorry

end NUMINAMATH_CALUDE_muffins_for_sale_is_108_l1670_167054


namespace NUMINAMATH_CALUDE_fruit_stand_problem_l1670_167075

/-- Represents the number of fruits Mary selects -/
structure FruitSelection where
  apples : ℕ
  oranges : ℕ
  bananas : ℕ

/-- Calculates the total cost of fruits in cents -/
def totalCost (s : FruitSelection) : ℕ :=
  40 * s.apples + 60 * s.oranges + 80 * s.bananas

/-- Calculates the average cost of fruits in cents -/
def averageCost (s : FruitSelection) : ℚ :=
  (totalCost s : ℚ) / (s.apples + s.oranges + s.bananas : ℚ)

theorem fruit_stand_problem (s : FruitSelection) 
  (total_fruits : s.apples + s.oranges + s.bananas = 12)
  (initial_avg : averageCost s = 55) :
  let new_selection := FruitSelection.mk s.apples (s.oranges - 6) s.bananas
  averageCost new_selection = 50 := by
  sorry

end NUMINAMATH_CALUDE_fruit_stand_problem_l1670_167075


namespace NUMINAMATH_CALUDE_max_regions_six_chords_l1670_167056

/-- The number of regions created by drawing k chords in a circle -/
def num_regions (k : ℕ) : ℕ := 1 + k * (k + 1) / 2

/-- Theorem: The maximum number of regions created by drawing 6 chords in a circle is 22 -/
theorem max_regions_six_chords : num_regions 6 = 22 := by
  sorry

end NUMINAMATH_CALUDE_max_regions_six_chords_l1670_167056


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1670_167046

theorem inequality_system_solution (a : ℝ) (h : a < 0) :
  {x : ℝ | x > -2*a ∧ x > 3*a} = {x : ℝ | x > -2*a} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1670_167046


namespace NUMINAMATH_CALUDE_intersection_area_is_three_sqrt_three_half_l1670_167081

/-- Regular tetrahedron with edge length 6 -/
structure RegularTetrahedron where
  edgeLength : ℝ
  edgeLength_eq : edgeLength = 6

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Plane passing through three points -/
structure Plane where
  a : Point3D  -- Vertex A
  m : Point3D  -- Midpoint M
  n : Point3D  -- Point N

/-- The area of intersection between a regular tetrahedron and a plane -/
def intersectionArea (t : RegularTetrahedron) (p : Plane) : ℝ := sorry

/-- Theorem stating that the area of intersection is 3√3/2 -/
theorem intersection_area_is_three_sqrt_three_half (t : RegularTetrahedron) (p : Plane) :
  intersectionArea t p = 3 * Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_intersection_area_is_three_sqrt_three_half_l1670_167081


namespace NUMINAMATH_CALUDE_continued_fraction_solution_l1670_167020

/-- The continued fraction equation representing the given expression -/
def continued_fraction_equation (x : ℝ) : Prop :=
  x = 3 + 5 / (2 + 5 / x)

/-- The theorem stating that 5 is the solution to the continued fraction equation -/
theorem continued_fraction_solution :
  ∃ (x : ℝ), continued_fraction_equation x ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_solution_l1670_167020


namespace NUMINAMATH_CALUDE_range_of_a_for_false_quadratic_inequality_l1670_167092

theorem range_of_a_for_false_quadratic_inequality :
  (∃ a : ℝ, ∀ x : ℝ, x^2 - a*x + 1 > 0) ↔ 
  (∃ a : ℝ, -2 < a ∧ a < 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_false_quadratic_inequality_l1670_167092
