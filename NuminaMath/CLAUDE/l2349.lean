import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l2349_234946

theorem quadratic_no_real_roots : ¬ ∃ (x : ℝ), x^2 + x + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l2349_234946


namespace NUMINAMATH_CALUDE_books_loaned_out_l2349_234976

/-- The number of books in the special collection at the beginning of the month -/
def initial_books : ℕ := 150

/-- The percentage of loaned books that are returned -/
def return_rate : ℚ := 85 / 100

/-- The number of books in the special collection at the end of the month -/
def final_books : ℕ := 135

/-- The number of books damaged or lost and replaced -/
def replaced_books : ℕ := 5

/-- The number of books loaned out during the month -/
def loaned_books : ℕ := 133

theorem books_loaned_out : 
  initial_books - loaned_books + (return_rate * loaned_books).floor + replaced_books = final_books :=
sorry

end NUMINAMATH_CALUDE_books_loaned_out_l2349_234976


namespace NUMINAMATH_CALUDE_pagoda_lights_l2349_234958

theorem pagoda_lights (n : ℕ) (total : ℕ) (h1 : n = 7) (h2 : total = 381) :
  ∃ (a : ℕ), 
    a * (1 - (1/2)^n) / (1 - 1/2) = total ∧ 
    a * (1/2)^(n-1) = 3 :=
sorry

end NUMINAMATH_CALUDE_pagoda_lights_l2349_234958


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l2349_234947

def diamond (a b : ℤ) : ℤ := 2 * a + b

theorem diamond_equation_solution :
  ∃ y : ℤ, diamond 4 (diamond 3 y) = 17 ∧ y = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l2349_234947


namespace NUMINAMATH_CALUDE_frank_bags_theorem_l2349_234905

/-- Given that Frank has a total number of candy pieces and puts an equal number of pieces in each bag, 
    calculate the number of bags used. -/
def bags_used (total_candy : ℕ) (candy_per_bag : ℕ) : ℕ :=
  total_candy / candy_per_bag

/-- Theorem stating that Frank used 2 bags given the problem conditions -/
theorem frank_bags_theorem (total_candy : ℕ) (candy_per_bag : ℕ) 
  (h1 : total_candy = 16) (h2 : candy_per_bag = 8) : 
  bags_used total_candy candy_per_bag = 2 := by
  sorry

end NUMINAMATH_CALUDE_frank_bags_theorem_l2349_234905


namespace NUMINAMATH_CALUDE_tara_savings_loss_l2349_234963

/-- The amount Tara had saved before losing all her savings -/
def amount_lost : ℕ := by sorry

theorem tara_savings_loss :
  let clarinet_cost : ℕ := 90
  let initial_savings : ℕ := 10
  let book_price : ℕ := 5
  let total_books_sold : ℕ := 25
  amount_lost = 45 := by sorry

end NUMINAMATH_CALUDE_tara_savings_loss_l2349_234963


namespace NUMINAMATH_CALUDE_tire_pricing_ratio_l2349_234934

/-- Represents the daily tire production capacity --/
def daily_production : ℕ := 1000

/-- Represents the daily tire demand --/
def daily_demand : ℕ := 1200

/-- Represents the production cost of each tire in cents --/
def production_cost : ℕ := 25000

/-- Represents the weekly loss in cents due to limited production capacity --/
def weekly_loss : ℕ := 17500000

/-- Represents the ratio of selling price to production cost --/
def selling_price_ratio : ℚ := 3/2

theorem tire_pricing_ratio :
  daily_production = 1000 →
  daily_demand = 1200 →
  production_cost = 25000 →
  weekly_loss = 17500000 →
  selling_price_ratio = 3/2 := by sorry

end NUMINAMATH_CALUDE_tire_pricing_ratio_l2349_234934


namespace NUMINAMATH_CALUDE_function_bound_l2349_234956

theorem function_bound (a x : ℝ) (h1 : |a| ≤ 1) (h2 : |x| ≤ 1) : 
  |a * x^2 + x - a| ≤ 5/4 := by sorry

end NUMINAMATH_CALUDE_function_bound_l2349_234956


namespace NUMINAMATH_CALUDE_work_multiple_l2349_234914

/-- If a person can complete one unit of work in 5 days, and takes 15 days to complete 
    a certain amount of the same type of work, then the amount of work completed in 15 days 
    is 3 times the original unit of work. -/
theorem work_multiple (original_days : ℕ) (new_days : ℕ) (work_multiple : ℚ) :
  original_days = 5 →
  new_days = 15 →
  work_multiple = (new_days : ℚ) / (original_days : ℚ) →
  work_multiple = 3 := by
sorry

end NUMINAMATH_CALUDE_work_multiple_l2349_234914


namespace NUMINAMATH_CALUDE_paige_pencils_at_home_l2349_234973

/-- The number of pencils Paige had in her backpack -/
def pencils_in_backpack : ℕ := 2

/-- The difference between the number of pencils at home and in the backpack -/
def pencil_difference : ℕ := 13

/-- The number of pencils Paige had at home -/
def pencils_at_home : ℕ := pencils_in_backpack + pencil_difference

theorem paige_pencils_at_home :
  pencils_at_home = 15 := by sorry

end NUMINAMATH_CALUDE_paige_pencils_at_home_l2349_234973


namespace NUMINAMATH_CALUDE_monotonicity_condition_even_function_condition_minimum_value_l2349_234987

-- Define the function f(x) = x^2 + 2ax
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x

-- Define the domain [-5, 5]
def domain : Set ℝ := Set.Icc (-5) 5

-- Statement 1: Monotonicity condition
theorem monotonicity_condition (a : ℝ) :
  (∀ x ∈ domain, ∀ y ∈ domain, x < y → f a x < f a y) ∨
  (∀ x ∈ domain, ∀ y ∈ domain, x < y → f a x > f a y) ↔
  a ≤ -5 ∨ a ≥ 5 :=
sorry

-- Statement 2: Even function condition and extrema
theorem even_function_condition (a : ℝ) :
  (∀ x ∈ domain, f a x - 2*x = f a (-x) - 2*(-x)) →
  a = 1 ∧ 
  (∀ x ∈ domain, f a x ≤ 35) ∧
  (∀ x ∈ domain, f a x ≥ -1) ∧
  (∃ x ∈ domain, f a x = 35) ∧
  (∃ x ∈ domain, f a x = -1) :=
sorry

-- Statement 3: Minimum value
theorem minimum_value (a : ℝ) :
  (a ≥ 5 → ∀ x ∈ domain, f a x ≥ 25 - 10*a) ∧
  (a ≤ -5 → ∀ x ∈ domain, f a x ≥ 25 + 10*a) ∧
  (-5 < a ∧ a < 5 → ∀ x ∈ domain, f a x ≥ -a^2) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_condition_even_function_condition_minimum_value_l2349_234987


namespace NUMINAMATH_CALUDE_house_value_correct_l2349_234966

/-- Represents the inheritance distribution problem --/
structure InheritanceProblem where
  totalBrothers : Nat
  housesCount : Nat
  moneyPaidPerOlderBrother : Nat
  totalInheritance : Nat

/-- Calculates the value of one house given the inheritance problem --/
def houseValue (problem : InheritanceProblem) : Nat :=
  let olderBrothersCount := problem.housesCount
  let youngerBrothersCount := problem.totalBrothers - olderBrothersCount
  let totalMoneyPaid := olderBrothersCount * problem.moneyPaidPerOlderBrother
  let inheritancePerBrother := problem.totalInheritance / problem.totalBrothers
  (inheritancePerBrother * problem.totalBrothers - totalMoneyPaid) / problem.housesCount

/-- Theorem stating that the house value is correct for the given problem --/
theorem house_value_correct (problem : InheritanceProblem) :
  problem.totalBrothers = 5 →
  problem.housesCount = 3 →
  problem.moneyPaidPerOlderBrother = 2000 →
  problem.totalInheritance = 15000 →
  houseValue problem = 3000 := by
  sorry

#eval houseValue { totalBrothers := 5, housesCount := 3, moneyPaidPerOlderBrother := 2000, totalInheritance := 15000 }

end NUMINAMATH_CALUDE_house_value_correct_l2349_234966


namespace NUMINAMATH_CALUDE_puzzle_solution_l2349_234904

/-- Represents a chip in the puzzle -/
def Chip := Fin 25

/-- Represents the arrangement of chips -/
def Arrangement := Fin 25 → Chip

/-- The initial arrangement of chips -/
def initial_arrangement : Arrangement := sorry

/-- The target arrangement of chips (in order) -/
def target_arrangement : Arrangement := sorry

/-- Represents a swap of two chips -/
def Swap := Chip × Chip

/-- Applies a swap to an arrangement -/
def apply_swap (a : Arrangement) (s : Swap) : Arrangement := sorry

/-- A sequence of swaps -/
def SwapSequence := List Swap

/-- Applies a sequence of swaps to an arrangement -/
def apply_swap_sequence (a : Arrangement) (ss : SwapSequence) : Arrangement := sorry

/-- The optimal swap sequence to solve the puzzle -/
def optimal_swap_sequence : SwapSequence := sorry

theorem puzzle_solution :
  apply_swap_sequence initial_arrangement optimal_swap_sequence = target_arrangement ∧
  optimal_swap_sequence.length = 19 := by sorry

end NUMINAMATH_CALUDE_puzzle_solution_l2349_234904


namespace NUMINAMATH_CALUDE_min_max_sum_of_a_l2349_234990

theorem min_max_sum_of_a (a b c : ℝ) (sum_eq : a + b + c = 5) (sum_sq_eq : a^2 + b^2 + c^2 = 8) :
  ∃ (m M : ℝ), (∀ x, (∃ y z, x + y + z = 5 ∧ x^2 + y^2 + z^2 = 8) → m ≤ x ∧ x ≤ M) ∧ m + M = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_max_sum_of_a_l2349_234990


namespace NUMINAMATH_CALUDE_smallest_sum_of_sequence_l2349_234924

theorem smallest_sum_of_sequence (A B C D : ℕ) : 
  A > 0 → B > 0 → C > 0 →  -- A, B, C are positive integers
  (C - B = B - A) →  -- A, B, C form an arithmetic sequence
  (C * C = B * D) →  -- B, C, D form a geometric sequence
  (C : ℚ) / B = 7 / 4 →  -- C/B = 7/4
  (∀ A' B' C' D' : ℕ, 
    A' > 0 → B' > 0 → C' > 0 → 
    (C' - B' = B' - A') → 
    (C' * C' = B' * D') → 
    (C' : ℚ) / B' = 7 / 4 → 
    A + B + C + D ≤ A' + B' + C' + D') →
  A + B + C + D = 97 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_sequence_l2349_234924


namespace NUMINAMATH_CALUDE_dividend_calculation_l2349_234968

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 18) 
  (h2 : quotient = 9) 
  (h3 : remainder = 3) : 
  divisor * quotient + remainder = 165 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2349_234968


namespace NUMINAMATH_CALUDE_factors_of_180_l2349_234922

/-- The number of positive factors of 180 is 18 -/
theorem factors_of_180 : Nat.card (Nat.divisors 180) = 18 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_180_l2349_234922


namespace NUMINAMATH_CALUDE_parabola_intercepts_sum_l2349_234988

/-- Represents a parabola of the form x = 3y² - 9y + 5 --/
def Parabola (x y : ℝ) : Prop := x = 3 * y^2 - 9 * y + 5

/-- Theorem stating that for the given parabola, the sum of its x-intercept and y-intercepts is 8 --/
theorem parabola_intercepts_sum (a b c : ℝ) 
  (h_x_intercept : Parabola a 0)
  (h_y_intercept1 : Parabola 0 b)
  (h_y_intercept2 : Parabola 0 c)
  : a + b + c = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intercepts_sum_l2349_234988


namespace NUMINAMATH_CALUDE_basketball_highlight_film_avg_player_footage_l2349_234994

/-- Calculates the average player footage in minutes for a basketball highlight film --/
theorem basketball_highlight_film_avg_player_footage
  (point_guard_footage : ℕ)
  (shooting_guard_footage : ℕ)
  (small_forward_footage : ℕ)
  (power_forward_footage : ℕ)
  (center_footage : ℕ)
  (game_footage : ℕ)
  (interview_footage : ℕ)
  (opening_closing_footage : ℕ)
  (h1 : point_guard_footage = 130)
  (h2 : shooting_guard_footage = 145)
  (h3 : small_forward_footage = 85)
  (h4 : power_forward_footage = 60)
  (h5 : center_footage = 180)
  (h6 : game_footage = 120)
  (h7 : interview_footage = 90)
  (h8 : opening_closing_footage = 30) :
  (point_guard_footage + shooting_guard_footage + small_forward_footage + power_forward_footage + center_footage) / (5 * 60) = 2 :=
by sorry

end NUMINAMATH_CALUDE_basketball_highlight_film_avg_player_footage_l2349_234994


namespace NUMINAMATH_CALUDE_remove_parentheses_first_step_l2349_234919

/-- Represents the steps in solving a linear equation -/
inductive SolvingStep
  | RemoveParentheses
  | EliminateDenominator
  | MoveTerms
  | CombineTerms

/-- Represents a linear equation -/
structure LinearEquation where
  lhs : ℝ → ℝ
  rhs : ℝ → ℝ

/-- The given equation: 2x + 3(2x - 1) = 16 - (x + 1) -/
def givenEquation : LinearEquation :=
  { lhs := λ x ↦ 2*x + 3*(2*x - 1)
    rhs := λ x ↦ 16 - (x + 1) }

/-- The first step in solving the given linear equation -/
def firstSolvingStep (eq : LinearEquation) : SolvingStep := sorry

/-- Theorem stating that removing parentheses is the first step for the given equation -/
theorem remove_parentheses_first_step :
  firstSolvingStep givenEquation = SolvingStep.RemoveParentheses := sorry

end NUMINAMATH_CALUDE_remove_parentheses_first_step_l2349_234919


namespace NUMINAMATH_CALUDE_at_least_one_solution_l2349_234902

-- Define the polynomials
variable (P S T : ℂ → ℂ)

-- Define the properties of the polynomials
axiom P_degree : ∃ (a b c : ℂ), ∀ z, P z = z^3 + a*z^2 + b*z + 4
axiom S_degree : ∃ (a b c d : ℂ), ∀ z, S z = z^4 + a*z^3 + b*z^2 + c*z + 5
axiom T_degree : ∃ (a b c d e f g : ℂ), ∀ z, T z = z^7 + a*z^6 + b*z^5 + c*z^4 + d*z^3 + e*z^2 + f*z + 20

-- Theorem statement
theorem at_least_one_solution :
  ∃ z : ℂ, P z * S z = T z :=
sorry

end NUMINAMATH_CALUDE_at_least_one_solution_l2349_234902


namespace NUMINAMATH_CALUDE_line_segment_has_measurable_length_l2349_234927

-- Define the characteristics of geometric objects
structure GeometricObject where
  has_endpoints : Bool
  is_infinite : Bool

-- Define specific geometric objects
def line : GeometricObject :=
  { has_endpoints := false, is_infinite := true }

def ray : GeometricObject :=
  { has_endpoints := true, is_infinite := true }

def line_segment : GeometricObject :=
  { has_endpoints := true, is_infinite := false }

-- Define a property for having measurable length
def has_measurable_length (obj : GeometricObject) : Prop :=
  obj.has_endpoints ∧ ¬obj.is_infinite

-- Theorem statement
theorem line_segment_has_measurable_length :
  has_measurable_length line_segment ∧
  ¬has_measurable_length line ∧
  ¬has_measurable_length ray :=
sorry

end NUMINAMATH_CALUDE_line_segment_has_measurable_length_l2349_234927


namespace NUMINAMATH_CALUDE_gcd_lcm_power_equation_l2349_234931

/-- Given positive integers m and n, if m^(gcd m n) = n^(lcm m n), then m = 1 and n = 1 -/
theorem gcd_lcm_power_equation (m n : ℕ+) :
  m ^ (Nat.gcd m.val n.val) = n ^ (Nat.lcm m.val n.val) → m = 1 ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_power_equation_l2349_234931


namespace NUMINAMATH_CALUDE_circle_tangent_sum_l2349_234911

def circle_radius_sum : ℝ := 14

theorem circle_tangent_sum (C : ℝ × ℝ) (r : ℝ) :
  (C.1 = r ∧ C.2 = r) →  -- Circle center C is at (r, r)
  ((C.1 - 5)^2 + C.2^2 = (r + 2)^2) →  -- External tangency condition
  (∃ (r1 r2 : ℝ), r1 + r2 = circle_radius_sum ∧ 
    ((C.1 = r1 ∧ C.2 = r1) ∨ (C.1 = r2 ∧ C.2 = r2))) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_sum_l2349_234911


namespace NUMINAMATH_CALUDE_minimum_nickels_needed_l2349_234998

def sneaker_cost : ℚ := 42.5
def tax_rate : ℚ := 0.08
def five_dollar_bills : ℕ := 4
def one_dollar_bills : ℕ := 6
def quarters : ℕ := 10

def total_cost : ℚ := sneaker_cost * (1 + tax_rate)

def money_without_nickels : ℚ := 
  (five_dollar_bills * 5) + one_dollar_bills + (quarters * 0.25)

theorem minimum_nickels_needed :
  ∃ n : ℕ, 
    (money_without_nickels + n * 0.05 ≥ total_cost) ∧
    (∀ m : ℕ, m < n → money_without_nickels + m * 0.05 < total_cost) ∧
    n = 348 := by
  sorry

end NUMINAMATH_CALUDE_minimum_nickels_needed_l2349_234998


namespace NUMINAMATH_CALUDE_linear_function_problem_l2349_234960

-- Define a linear function
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

-- State the theorem
theorem linear_function_problem (f : ℝ → ℝ) 
  (h_linear : LinearFunction f)
  (h_diff : f 10 - f 5 = 20)
  (h_f0 : f 0 = 3) :
  f 15 - f 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_problem_l2349_234960


namespace NUMINAMATH_CALUDE_sum_256_64_base_8_l2349_234909

def to_base_8 (n : ℕ) : ℕ := sorry

theorem sum_256_64_base_8 : 
  to_base_8 (256 + 64) = 500 := by sorry

end NUMINAMATH_CALUDE_sum_256_64_base_8_l2349_234909


namespace NUMINAMATH_CALUDE_planes_perpendicular_to_line_are_parallel_lines_perpendicular_to_plane_are_parallel_not_all_lines_perpendicular_to_line_are_parallel_not_all_planes_perpendicular_to_plane_are_parallel_or_intersect_l2349_234985

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relationships
variable (perpendicular_line_line : Line → Line → Prop)
variable (perpendicular_plane_line : Plane → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)
variable (parallel_line : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (intersect_plane : Plane → Plane → Prop)

-- Theorem statements
theorem planes_perpendicular_to_line_are_parallel 
  (p1 p2 : Plane) (l : Line) 
  (h1 : perpendicular_plane_line p1 l) 
  (h2 : perpendicular_plane_line p2 l) : 
  parallel_plane p1 p2 :=
sorry

theorem lines_perpendicular_to_plane_are_parallel 
  (l1 l2 : Line) (p : Plane) 
  (h1 : perpendicular_line_plane l1 p) 
  (h2 : perpendicular_line_plane l2 p) : 
  parallel_line l1 l2 :=
sorry

theorem not_all_lines_perpendicular_to_line_are_parallel : 
  ∃ (l1 l2 l3 : Line), 
    perpendicular_line_line l1 l3 ∧ 
    perpendicular_line_line l2 l3 ∧ 
    ¬(parallel_line l1 l2) :=
sorry

theorem not_all_planes_perpendicular_to_plane_are_parallel_or_intersect : 
  ∃ (p1 p2 p3 : Plane), 
    perpendicular_plane_plane p1 p3 ∧ 
    perpendicular_plane_plane p2 p3 ∧ 
    ¬(parallel_plane p1 p2 ∨ intersect_plane p1 p2) :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_to_line_are_parallel_lines_perpendicular_to_plane_are_parallel_not_all_lines_perpendicular_to_line_are_parallel_not_all_planes_perpendicular_to_plane_are_parallel_or_intersect_l2349_234985


namespace NUMINAMATH_CALUDE_point_not_in_reflected_rectangle_l2349_234978

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflect_y (p : Point) : Point :=
  ⟨-p.x, p.y⟩

/-- The set of vertices of the original rectangle -/
def original_vertices : Set Point :=
  {⟨1, 3⟩, ⟨1, 1⟩, ⟨4, 1⟩, ⟨4, 3⟩}

/-- The set of vertices of the reflected rectangle -/
def reflected_vertices : Set Point :=
  original_vertices.image reflect_y

/-- The point in question -/
def point_to_check : Point :=
  ⟨-3, 4⟩

theorem point_not_in_reflected_rectangle :
  point_to_check ∉ reflected_vertices :=
sorry

end NUMINAMATH_CALUDE_point_not_in_reflected_rectangle_l2349_234978


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2349_234983

theorem sum_of_coefficients (b₅ b₄ b₃ b₂ b₁ b₀ : ℝ) :
  (∀ x : ℝ, (2*x + 3)^5 = b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 3125 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2349_234983


namespace NUMINAMATH_CALUDE_bill_omelet_time_l2349_234932

/-- Represents the time Bill spends on preparing and cooking omelets -/
def total_time (
  pepper_chop_time : ℕ)
  (onion_chop_time : ℕ)
  (cheese_grate_time : ℕ)
  (omelet_cook_time : ℕ)
  (num_peppers : ℕ)
  (num_onions : ℕ)
  (num_omelets : ℕ) : ℕ :=
  pepper_chop_time * num_peppers +
  onion_chop_time * num_onions +
  cheese_grate_time * num_omelets +
  omelet_cook_time * num_omelets

/-- Theorem stating that Bill spends 50 minutes preparing and cooking omelets -/
theorem bill_omelet_time : 
  total_time 3 4 1 5 4 2 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_bill_omelet_time_l2349_234932


namespace NUMINAMATH_CALUDE_fibCoeff_symmetry_l2349_234937

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Fibonacci coefficient -/
def fibCoeff (n k : ℕ) : ℚ :=
  if k ≤ n then
    (List.range k).foldl (λ acc i => acc * fib (n - i)) 1 /
    (List.range k).foldl (λ acc i => acc * fib (k - i)) 1
  else 0

/-- Symmetry property of Fibonacci coefficients -/
theorem fibCoeff_symmetry (n k : ℕ) (h : k ≤ n) :
  fibCoeff n k = fibCoeff n (n - k) := by
  sorry

end NUMINAMATH_CALUDE_fibCoeff_symmetry_l2349_234937


namespace NUMINAMATH_CALUDE_max_work_hours_l2349_234920

/-- Represents Mary's work schedule and pay structure --/
structure WorkSchedule where
  regularHours : ℕ := 20
  regularRate : ℚ := 8
  overtimeRate : ℚ := 10
  maxEarnings : ℚ := 760

/-- Calculates the total hours worked given regular and overtime hours --/
def totalHours (regular : ℕ) (overtime : ℕ) : ℕ :=
  regular + overtime

/-- Calculates the total earnings given regular and overtime hours --/
def totalEarnings (schedule : WorkSchedule) (overtime : ℕ) : ℚ :=
  (schedule.regularHours : ℚ) * schedule.regularRate + (overtime : ℚ) * schedule.overtimeRate

/-- Theorem: The maximum number of hours Mary can work in a week is 80 --/
theorem max_work_hours (schedule : WorkSchedule) : 
  ∃ (overtime : ℕ), 
    totalHours schedule.regularHours overtime = 80 ∧ 
    totalEarnings schedule overtime ≤ schedule.maxEarnings ∧
    ∀ (h : ℕ), totalEarnings schedule h ≤ schedule.maxEarnings → 
      totalHours schedule.regularHours h ≤ 80 :=
by
  sorry

end NUMINAMATH_CALUDE_max_work_hours_l2349_234920


namespace NUMINAMATH_CALUDE_three_number_sum_l2349_234908

theorem three_number_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →  -- Ordering of numbers
  b = 8 →  -- Median is 8
  (a + b + c) / 3 = a + 8 →  -- Mean is 8 more than least
  (a + b + c) / 3 = c - 20 →  -- Mean is 20 less than greatest
  a + b + c = 60 := by sorry

end NUMINAMATH_CALUDE_three_number_sum_l2349_234908


namespace NUMINAMATH_CALUDE_regular_polygon_properties_l2349_234933

theorem regular_polygon_properties :
  ∀ (n : ℕ) (interior_angle exterior_angle : ℝ),
  n > 2 →
  interior_angle - exterior_angle = 90 →
  interior_angle + exterior_angle = 180 →
  n * exterior_angle = 360 →
  (n - 2) * 180 = n * interior_angle →
  (n - 2) * 180 = 1080 ∧ n = 8 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_properties_l2349_234933


namespace NUMINAMATH_CALUDE_car_wash_group_composition_l2349_234918

theorem car_wash_group_composition (total : ℕ) (girls : ℕ) : 
  girls = (2 * total : ℚ) / 5 →    -- Initially 40% of the group are girls
  ((girls : ℚ) - 2) / total = 3 / 10 →   -- After changes, 30% of the group are girls
  girls = 8 := by
sorry

end NUMINAMATH_CALUDE_car_wash_group_composition_l2349_234918


namespace NUMINAMATH_CALUDE_cantaloupes_left_total_l2349_234943

/-- The total number of cantaloupes left after each person's changes -/
def total_cantaloupes_left (fred_initial fred_eaten tim_initial tim_lost susan_initial susan_given nancy_initial nancy_traded : ℕ) : ℕ :=
  (fred_initial - fred_eaten) + (tim_initial - tim_lost) + (susan_initial - susan_given) + (nancy_initial - nancy_traded)

/-- Theorem stating the total number of cantaloupes left is 138 -/
theorem cantaloupes_left_total :
  total_cantaloupes_left 38 4 44 7 57 10 25 5 = 138 := by
  sorry

end NUMINAMATH_CALUDE_cantaloupes_left_total_l2349_234943


namespace NUMINAMATH_CALUDE_triangle_with_arithmetic_sides_is_right_angled_and_inradius_equals_diff_l2349_234901

/-- A triangle with sides in arithmetic progression including its semiperimeter -/
structure TriangleWithArithmeticSides where
  /-- The common difference of the arithmetic progression -/
  d : ℝ
  /-- The middle term of the arithmetic progression -/
  a : ℝ
  /-- Ensures that the sides are positive -/
  d_pos : 0 < d
  a_pos : 0 < a
  /-- Ensures that the triangle inequality holds -/
  triangle_ineq : 2 * d < a

theorem triangle_with_arithmetic_sides_is_right_angled_and_inradius_equals_diff 
  (t : TriangleWithArithmeticSides) : 
  /- The triangle is right-angled -/
  (3 * t.a / 4) ^ 2 + (4 * t.a / 4) ^ 2 = (5 * t.a / 4) ^ 2 ∧ 
  /- The common difference equals the inradius -/
  t.d = (3 * t.a / 4 + 4 * t.a / 4 - 5 * t.a / 4) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_with_arithmetic_sides_is_right_angled_and_inradius_equals_diff_l2349_234901


namespace NUMINAMATH_CALUDE_all_choose_same_house_probability_l2349_234942

/-- The probability that all 3 persons choose the same house when there are 3 houses
    and each person independently chooses a house with equal probability. -/
theorem all_choose_same_house_probability :
  let num_houses : ℕ := 3
  let num_persons : ℕ := 3
  let prob_choose_house : ℚ := 1 / 3
  (num_houses * (prob_choose_house ^ num_persons)) = 1 / 9 :=
by sorry

end NUMINAMATH_CALUDE_all_choose_same_house_probability_l2349_234942


namespace NUMINAMATH_CALUDE_inequalities_proof_l2349_234992

theorem inequalities_proof (a b : ℝ) (h : a + b > 0) :
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧
  (a^21 + b^21 > 0) ∧
  ((a+2)*(b+2) > a*b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l2349_234992


namespace NUMINAMATH_CALUDE_function_composition_result_l2349_234923

theorem function_composition_result (a b : ℝ) :
  (∀ x, (3 * ((a * x) + b) - 4) = 4 * x + 3) →
  a + b = 11 / 3 :=
by sorry

end NUMINAMATH_CALUDE_function_composition_result_l2349_234923


namespace NUMINAMATH_CALUDE_statement_B_is_incorrect_l2349_234989

-- Define the basic types
def Chromosome : Type := String
def Allele : Type := String
def Genotype : Type := List Allele

-- Define the meiosis process
def meiosis (g : Genotype) : List Genotype := sorry

-- Define the normal chromosome distribution
def normalChromosomeDistribution (g : Genotype) : Prop := sorry

-- Define the statement B
def statementB : Prop :=
  ∃ (parent : Genotype) (sperm : Genotype),
    parent = ["A", "a", "X^b", "Y"] ∧
    sperm ∈ meiosis parent ∧
    sperm = ["A", "A", "a", "Y"] ∧
    (∃ (other_sperms : List Genotype),
      other_sperms.length = 3 ∧
      (∀ s ∈ other_sperms, s ∈ meiosis parent) ∧
      other_sperms = [["a", "Y"], ["X^b"], ["X^b"]])

-- Theorem stating that B is incorrect
theorem statement_B_is_incorrect :
  ¬statementB :=
sorry

end NUMINAMATH_CALUDE_statement_B_is_incorrect_l2349_234989


namespace NUMINAMATH_CALUDE_unique_a_value_l2349_234930

def A (a : ℝ) : Set ℝ := {a - 2, 2 * a^2 + 5 * a, 12}

theorem unique_a_value : ∀ a : ℝ, -3 ∈ A a ↔ a = -3/2 := by sorry

end NUMINAMATH_CALUDE_unique_a_value_l2349_234930


namespace NUMINAMATH_CALUDE_dice_arithmetic_progression_probability_l2349_234935

def num_dice : ℕ := 4
def faces_per_die : ℕ := 6

def is_arithmetic_progression (nums : Finset ℕ) : Prop :=
  ∃ (a d : ℕ), ∀ i ∈ nums, ∃ k : ℕ, i = a + k * d

def favorable_outcomes : Finset (Finset ℕ) :=
  {{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}}

theorem dice_arithmetic_progression_probability :
  (Finset.card favorable_outcomes) / (faces_per_die ^ num_dice : ℚ) = 1 / 432 := by
  sorry

end NUMINAMATH_CALUDE_dice_arithmetic_progression_probability_l2349_234935


namespace NUMINAMATH_CALUDE_school_boys_count_l2349_234993

theorem school_boys_count :
  ∀ (boys girls : ℕ),
  (boys : ℚ) / girls = 5 / 13 →
  girls = boys + 80 →
  boys = 50 := by
sorry

end NUMINAMATH_CALUDE_school_boys_count_l2349_234993


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2349_234921

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_3 : a 3 = 4) (h_6 : a 6 = 1/2) : a 4 + a 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2349_234921


namespace NUMINAMATH_CALUDE_monotonicity_condition_equiv_a_range_l2349_234938

/-- Definition of the piecewise function f -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 5 else a/x

/-- Theorem stating the equivalence between the monotonicity condition and the range of a -/
theorem monotonicity_condition_equiv_a_range :
  ∀ a : ℝ, (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₂ - f a x₁) / (x₂ - x₁) > 0) ↔ a ∈ Set.Icc (-3) (-2) :=
by sorry

end NUMINAMATH_CALUDE_monotonicity_condition_equiv_a_range_l2349_234938


namespace NUMINAMATH_CALUDE_vectors_orthogonal_l2349_234910

def vector1 : Fin 2 → ℝ := ![2, 5]
def vector2 (x : ℝ) : Fin 2 → ℝ := ![x, -3]

theorem vectors_orthogonal :
  let x : ℝ := 15/2
  (vector1 0 * vector2 x 0 + vector1 1 * vector2 x 1 = 0) := by sorry

end NUMINAMATH_CALUDE_vectors_orthogonal_l2349_234910


namespace NUMINAMATH_CALUDE_triangular_number_difference_l2349_234944

/-- The nth triangular number -/
def triangularNumber (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The difference between the 2010th and 2009th triangular numbers is 2010 -/
theorem triangular_number_difference : triangularNumber 2010 - triangularNumber 2009 = 2010 := by
  sorry

end NUMINAMATH_CALUDE_triangular_number_difference_l2349_234944


namespace NUMINAMATH_CALUDE_ivy_morning_cupcakes_l2349_234995

/-- The number of cupcakes Ivy baked in the morning -/
def morning_cupcakes : ℕ := sorry

/-- The number of cupcakes Ivy baked in the afternoon -/
def afternoon_cupcakes : ℕ := morning_cupcakes + 15

/-- The total number of cupcakes Ivy baked -/
def total_cupcakes : ℕ := 55

/-- Theorem stating that Ivy baked 20 cupcakes in the morning -/
theorem ivy_morning_cupcakes : 
  morning_cupcakes = 20 ∧ 
  afternoon_cupcakes = morning_cupcakes + 15 ∧ 
  total_cupcakes = morning_cupcakes + afternoon_cupcakes := by
  sorry

end NUMINAMATH_CALUDE_ivy_morning_cupcakes_l2349_234995


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l2349_234982

theorem min_value_x_plus_y (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : 9 / (x + 1) + 1 / (y + 1) = 1) : 
  x + y ≥ 14 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 9 / (x₀ + 1) + 1 / (y₀ + 1) = 1 ∧ x₀ + y₀ = 14 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l2349_234982


namespace NUMINAMATH_CALUDE_lidia_remaining_money_l2349_234961

/-- Proves the remaining money after Lidia buys her needed apps -/
theorem lidia_remaining_money 
  (app_cost : ℕ) 
  (apps_needed : ℕ) 
  (available_money : ℕ) 
  (h1 : app_cost = 4)
  (h2 : apps_needed = 15)
  (h3 : available_money = 66) :
  available_money - (app_cost * apps_needed) = 6 :=
by sorry

end NUMINAMATH_CALUDE_lidia_remaining_money_l2349_234961


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2349_234997

/-- Given that x is inversely proportional to y, prove that when x = 8 and y = 16,
    then x = -4 when y = -32 -/
theorem inverse_proportion_problem (x y : ℝ) (c : ℝ) 
    (h1 : x * y = c)  -- x is inversely proportional to y
    (h2 : 8 * 16 = c) -- When x = 8, y = 16
    (h3 : y = -32)    -- Given y = -32
    : x = -4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2349_234997


namespace NUMINAMATH_CALUDE_toy_position_l2349_234926

theorem toy_position (total_toys : ℕ) (position_from_right : ℕ) (position_from_left : ℕ) :
  total_toys = 19 →
  position_from_right = 8 →
  position_from_left = total_toys - (position_from_right - 1) →
  position_from_left = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_toy_position_l2349_234926


namespace NUMINAMATH_CALUDE_pi_half_not_in_M_l2349_234957

-- Define the set M
def M : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}

-- State the theorem
theorem pi_half_not_in_M : π / 2 ∉ M := by
  sorry

end NUMINAMATH_CALUDE_pi_half_not_in_M_l2349_234957


namespace NUMINAMATH_CALUDE_weight_of_smaller_cube_l2349_234981

/-- Given two cubes of the same material, where the second cube has sides twice as long as the first
and weighs 64 pounds, the weight of the first cube is 8 pounds. -/
theorem weight_of_smaller_cube (s : ℝ) (weight_first : ℝ) (weight_second : ℝ) : 
  s > 0 → 
  weight_second = 64 → 
  (2 * s)^3 / s^3 * weight_first = weight_second → 
  weight_first = 8 :=
by sorry

end NUMINAMATH_CALUDE_weight_of_smaller_cube_l2349_234981


namespace NUMINAMATH_CALUDE_sum_first_ten_even_numbers_l2349_234971

-- Define the first 10 even numbers
def firstTenEvenNumbers : List Nat := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

-- Theorem: The sum of the first 10 even numbers is 110
theorem sum_first_ten_even_numbers :
  firstTenEvenNumbers.sum = 110 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_ten_even_numbers_l2349_234971


namespace NUMINAMATH_CALUDE_denominator_numerator_difference_l2349_234974

/-- The repeating decimal 0.868686... -/
def F : ℚ := 86 / 99

/-- F expressed as a decimal is 0.868686... (infinitely repeating) -/
axiom F_decimal : F = 0.868686

theorem denominator_numerator_difference :
  (F.den : ℤ) - (F.num : ℤ) = 13 := by sorry

end NUMINAMATH_CALUDE_denominator_numerator_difference_l2349_234974


namespace NUMINAMATH_CALUDE_trip_time_at_new_speed_l2349_234972

-- Define the original speed, time, and new speed
def original_speed : ℝ := 80
def original_time : ℝ := 3
def new_speed : ℝ := 50

-- Define the constant distance
def distance : ℝ := original_speed * original_time

-- Theorem to prove
theorem trip_time_at_new_speed :
  distance / new_speed = 4.8 := by sorry

end NUMINAMATH_CALUDE_trip_time_at_new_speed_l2349_234972


namespace NUMINAMATH_CALUDE_tangent_circles_theorem_l2349_234916

/-- Given two circles with centers E and F tangent to segment BD and semicircles with diameters AB, BC, and AC,
    where r1, r2, and r are the radii of semicircles with diameters AB, BC, and AC respectively,
    and l1 and l2 are the radii of circles with centers E and F respectively. -/
theorem tangent_circles_theorem 
  (r1 r2 r l1 l2 : ℝ) 
  (h_r : r = r1 + r2) 
  (h_positive : r1 > 0 ∧ r2 > 0 ∧ l1 > 0 ∧ l2 > 0) :
  (∃ (distance_E_to_AC : ℝ), distance_E_to_AC = Real.sqrt ((r1 + l1)^2 - (r1 - l1)^2)) ∧ 
  l1 = (r1 * r2) / (r1 + r2) := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_theorem_l2349_234916


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l2349_234986

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

/-- The equation of the first circle: x^2 + y^2 = 1 -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- The equation of the second circle: x^2 + y^2 - 6x - 8y + 9 = 0 -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + 9 = 0

theorem circles_externally_tangent :
  externally_tangent (0, 0) (3, 4) 1 4 := by
  sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l2349_234986


namespace NUMINAMATH_CALUDE_mikes_initial_cards_l2349_234945

theorem mikes_initial_cards (initial_cards current_cards cards_sold : ℕ) :
  current_cards = 74 →
  cards_sold = 13 →
  initial_cards = current_cards + cards_sold →
  initial_cards = 87 := by
sorry

end NUMINAMATH_CALUDE_mikes_initial_cards_l2349_234945


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2349_234900

theorem solve_linear_equation (x : ℝ) : (3 * x - 8 = -2 * x + 17) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2349_234900


namespace NUMINAMATH_CALUDE_impossible_valid_arrangement_l2349_234949

/-- Represents the colors of chips -/
inductive Color
| Blue
| Red
| Green

/-- Represents a circular arrangement of chips -/
def CircularArrangement := List Color

/-- Represents a swap operation -/
inductive SwapOperation
| BlueRed
| BlueGreen

/-- Initial arrangement of chips -/
def initial_arrangement : CircularArrangement :=
  (List.replicate 40 Color.Blue) ++ (List.replicate 30 Color.Red) ++ (List.replicate 20 Color.Green)

/-- Checks if an arrangement has no adjacent chips of the same color -/
def is_valid_arrangement (arr : CircularArrangement) : Bool :=
  sorry

/-- Applies a swap operation to an arrangement -/
def apply_swap (arr : CircularArrangement) (op : SwapOperation) : CircularArrangement :=
  sorry

/-- Theorem stating that it's impossible to achieve a valid arrangement -/
theorem impossible_valid_arrangement :
  ∀ (ops : List SwapOperation),
    let final_arrangement := ops.foldl apply_swap initial_arrangement
    ¬ (is_valid_arrangement final_arrangement) :=
  sorry

end NUMINAMATH_CALUDE_impossible_valid_arrangement_l2349_234949


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l2349_234953

theorem quadratic_root_problem (v : ℚ) : 
  (3 * ((-12 - Real.sqrt 400) / 15)^2 + 12 * ((-12 - Real.sqrt 400) / 15) + v = 0) → 
  v = 704/75 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l2349_234953


namespace NUMINAMATH_CALUDE_g_composition_of_three_l2349_234991

def g (n : ℤ) : ℤ :=
  if n < 5 then 2 * n^2 + 3 else 4 * n + 1

theorem g_composition_of_three : g (g (g 3)) = 341 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_three_l2349_234991


namespace NUMINAMATH_CALUDE_pet_store_siamese_cats_l2349_234913

/-- The number of Siamese cats initially in the pet store. -/
def initial_siamese_cats : ℕ := 13

/-- The number of house cats initially in the pet store. -/
def initial_house_cats : ℕ := 5

/-- The number of cats sold during the sale. -/
def cats_sold : ℕ := 10

/-- The number of cats left after the sale. -/
def cats_remaining : ℕ := 8

/-- Theorem stating that the initial number of Siamese cats is correct. -/
theorem pet_store_siamese_cats :
  initial_siamese_cats + initial_house_cats - cats_sold = cats_remaining :=
by sorry

end NUMINAMATH_CALUDE_pet_store_siamese_cats_l2349_234913


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_h_l2349_234917

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the vertex form
def vertex_form (n h k : ℝ) (x : ℝ) : ℝ := n * (x - h)^2 + k

-- Theorem statement
theorem quadratic_vertex_form_h (a b c : ℝ) :
  (∃ n k : ℝ, ∀ x : ℝ, 4 * f a b c x = vertex_form n 3 k x) →
  (∀ x : ℝ, f a b c x = 3 * (x - 3)^2 + 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_h_l2349_234917


namespace NUMINAMATH_CALUDE_equal_split_contribution_l2349_234928

def earnings : List ℝ := [18, 22, 30, 38, 45]

theorem equal_split_contribution (total : ℝ) (equal_share : ℝ) :
  total = earnings.sum →
  equal_share = total / 5 →
  45 - equal_share = 14.4 := by
  sorry

end NUMINAMATH_CALUDE_equal_split_contribution_l2349_234928


namespace NUMINAMATH_CALUDE_club_average_age_l2349_234967

theorem club_average_age (women : ℕ) (men : ℕ) (children : ℕ)
  (women_avg : ℝ) (men_avg : ℝ) (children_avg : ℝ)
  (h_women : women = 12)
  (h_men : men = 18)
  (h_children : children = 10)
  (h_women_avg : women_avg = 32)
  (h_men_avg : men_avg = 38)
  (h_children_avg : children_avg = 10) :
  (women * women_avg + men * men_avg + children * children_avg) / (women + men + children) = 29.2 := by
  sorry

end NUMINAMATH_CALUDE_club_average_age_l2349_234967


namespace NUMINAMATH_CALUDE_inequality_proof_l2349_234948

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : 
  (Real.sqrt (b^2 - a*c)) / a < Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2349_234948


namespace NUMINAMATH_CALUDE_toms_candy_problem_l2349_234952

/-- Tom's candy problem -/
theorem toms_candy_problem (initial : Nat) (bought : Nat) (total : Nat) (friend_gave : Nat) : 
  initial = 2 → 
  bought = 10 → 
  total = 19 → 
  initial + bought + friend_gave = total → 
  friend_gave = 7 := by
  sorry

#check toms_candy_problem

end NUMINAMATH_CALUDE_toms_candy_problem_l2349_234952


namespace NUMINAMATH_CALUDE_unique_solution_l2349_234939

-- Define the equation
def equation (x : ℝ) : Prop :=
  (3 * x^2 - 15 * x + 18) / (x^2 - 5 * x + 6) = x - 2

-- Theorem statement
theorem unique_solution :
  ∃! x : ℝ, equation x ∧ x^2 - 5 * x + 6 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2349_234939


namespace NUMINAMATH_CALUDE_most_accurate_estimate_l2349_234954

-- Define the temperature range
def lower_bound : Float := 98.6
def upper_bound : Float := 99.1

-- Define a type for temperature readings
structure TemperatureReading where
  value : Float
  is_within_range : lower_bound ≤ value ∧ value ≤ upper_bound

-- Define a function to determine if a reading is closer to the upper bound
def closer_to_upper_bound (reading : TemperatureReading) : Prop :=
  reading.value > (lower_bound + upper_bound) / 2

-- Theorem statement
theorem most_accurate_estimate (reading : TemperatureReading) 
  (h : closer_to_upper_bound reading) : 
  upper_bound = 99.1 ∧ upper_bound - reading.value < reading.value - lower_bound :=
by
  sorry

end NUMINAMATH_CALUDE_most_accurate_estimate_l2349_234954


namespace NUMINAMATH_CALUDE_sum_le_one_plus_product_l2349_234941

theorem sum_le_one_plus_product (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  a + b ≤ 1 + a * b :=
by sorry

end NUMINAMATH_CALUDE_sum_le_one_plus_product_l2349_234941


namespace NUMINAMATH_CALUDE_rational_equation_solution_l2349_234955

theorem rational_equation_solution :
  ∃ x : ℚ, (x^2 - 7*x + 10) / (x^2 - 6*x + 5) = (x^2 - 4*x - 21) / (x^2 - 3*x - 18) ∧ x = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l2349_234955


namespace NUMINAMATH_CALUDE_books_not_sold_percentage_l2349_234925

def initial_stock : ℕ := 800
def monday_sales : ℕ := 62
def tuesday_sales : ℕ := 62
def wednesday_sales : ℕ := 60
def thursday_sales : ℕ := 48
def friday_sales : ℕ := 40

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales

def books_not_sold : ℕ := initial_stock - total_sales

def percentage_not_sold : ℚ := (books_not_sold : ℚ) / (initial_stock : ℚ) * 100

theorem books_not_sold_percentage :
  percentage_not_sold = 66 := by sorry

end NUMINAMATH_CALUDE_books_not_sold_percentage_l2349_234925


namespace NUMINAMATH_CALUDE_sum_of_consecutive_odd_primes_l2349_234940

/-- Two natural numbers are consecutive primes if they are both prime and there are no primes between them. -/
def ConsecutivePrimes (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ p < q ∧ ∀ k, p < k → k < q → ¬Nat.Prime k

theorem sum_of_consecutive_odd_primes (p q : ℕ) (h : ConsecutivePrimes p q) (hp_odd : Odd p) (hq_odd : Odd q) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ p + q = 2 * a * b :=
sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_odd_primes_l2349_234940


namespace NUMINAMATH_CALUDE_matrix_equality_implies_fraction_l2349_234915

/-- Given two 2x2 matrices A and B, where A is [[2, 5], [3, 7]] and B is [[a, b], [c, d]],
    if AB = BA and 5b ≠ c, then (a - d) / (c - 5b) = 6c / (5a + 22c) -/
theorem matrix_equality_implies_fraction (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 5; 3, 7]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  (A * B = B * A) → (5 * b ≠ c) → 
  (a - d) / (c - 5 * b) = 6 * c / (5 * a + 22 * c) := by
  sorry

end NUMINAMATH_CALUDE_matrix_equality_implies_fraction_l2349_234915


namespace NUMINAMATH_CALUDE_product_of_one_plus_tangents_sine_double_angle_l2349_234959

-- Part I
theorem product_of_one_plus_tangents (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2) (h3 : α + β = π/4) : 
  (1 + Real.tan α) * (1 + Real.tan β) = 2 := by
  sorry

-- Part II
theorem sine_double_angle (α β : Real) 
  (h1 : π/2 < β ∧ β < α ∧ α < 3*π/4) 
  (h2 : Real.cos (α - β) = 12/13) 
  (h3 : Real.sin (α + β) = -3/5) : 
  Real.sin (2 * α) = -56/65 := by
  sorry

end NUMINAMATH_CALUDE_product_of_one_plus_tangents_sine_double_angle_l2349_234959


namespace NUMINAMATH_CALUDE_solve_equation_l2349_234912

theorem solve_equation (x : ℝ) (h : 5 - 5 / x = 4 + 4 / x) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2349_234912


namespace NUMINAMATH_CALUDE_max_value_trig_sum_l2349_234969

theorem max_value_trig_sum (x : ℝ) : 3 * Real.cos x + 4 * Real.sin x ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_trig_sum_l2349_234969


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l2349_234977

/-- Given a right triangle with acute angles in the ratio 5:4 and hypotenuse 10 cm,
    the length of the side opposite the smaller angle is 10 * sin(40°) -/
theorem right_triangle_side_length (a b c : ℝ) (θ₁ θ₂ : Real) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem (right triangle condition)
  c = 10 →  -- hypotenuse length
  θ₁ / θ₂ = 5 / 4 →  -- ratio of acute angles
  θ₁ + θ₂ = π / 2 →  -- sum of acute angles in a right triangle
  θ₂ < θ₁ →  -- θ₂ is the smaller angle
  b = 10 * Real.sin (40 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l2349_234977


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l2349_234984

/-- Isosceles trapezoid with given properties -/
structure IsoscelesTrapezoid where
  /-- Length of lateral sides AB and CD -/
  lateral_side : ℝ
  /-- Ratio of AH : AK : AC -/
  ratio_ah : ℝ
  ratio_ak : ℝ
  ratio_ac : ℝ
  /-- Conditions -/
  lateral_positive : lateral_side > 0
  ratio_positive : ratio_ah > 0 ∧ ratio_ak > 0 ∧ ratio_ac > 0
  ratio_order : ratio_ah < ratio_ak ∧ ratio_ak < ratio_ac

/-- The area of the isosceles trapezoid with given properties is 180 -/
theorem isosceles_trapezoid_area
  (t : IsoscelesTrapezoid)
  (h1 : t.lateral_side = 10)
  (h2 : t.ratio_ah = 5 ∧ t.ratio_ak = 14 ∧ t.ratio_ac = 15) :
  ∃ (area : ℝ), area = 180 :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l2349_234984


namespace NUMINAMATH_CALUDE_optimal_solution_l2349_234936

/-- Represents a container with its size and count -/
structure Container where
  size : Nat
  count : Nat

/-- Calculates the total volume of water from a list of containers -/
def totalVolume (containers : List Container) : Nat :=
  containers.foldl (fun acc c => acc + c.size * c.count) 0

/-- Calculates the total number of trips for a list of containers -/
def totalTrips (containers : List Container) : Nat :=
  containers.foldl (fun acc c => acc + c.count) 0

/-- Theorem stating that the given solution is optimal -/
theorem optimal_solution (initialVolume timeLimit : Nat) : 
  let targetVolume : Nat := 823
  let containers : List Container := [
    { size := 8, count := 18 },
    { size := 2, count := 1 },
    { size := 5, count := 1 }
  ]
  (initialVolume = 676) →
  (timeLimit = 45) →
  (totalVolume containers + initialVolume ≥ targetVolume) ∧
  (totalTrips containers ≤ timeLimit) ∧
  (∀ (otherContainers : List Container),
    (totalVolume otherContainers + initialVolume ≥ targetVolume) →
    (totalTrips otherContainers ≤ timeLimit) →
    (totalTrips containers ≤ totalTrips otherContainers)) :=
by sorry


end NUMINAMATH_CALUDE_optimal_solution_l2349_234936


namespace NUMINAMATH_CALUDE_psychiatric_sessions_l2349_234965

theorem psychiatric_sessions 
  (total_patients : ℕ) 
  (total_sessions : ℕ) 
  (first_patient_sessions : ℕ) 
  (second_patient_additional_sessions : ℕ) :
  total_patients = 4 →
  total_sessions = 25 →
  first_patient_sessions = 6 →
  second_patient_additional_sessions = 5 →
  total_sessions - (first_patient_sessions + (first_patient_sessions + second_patient_additional_sessions)) = 8 :=
by sorry

end NUMINAMATH_CALUDE_psychiatric_sessions_l2349_234965


namespace NUMINAMATH_CALUDE_square_diff_div_81_l2349_234980

theorem square_diff_div_81 : (2500 - 2409)^2 / 81 = 102 := by sorry

end NUMINAMATH_CALUDE_square_diff_div_81_l2349_234980


namespace NUMINAMATH_CALUDE_greater_number_problem_l2349_234999

theorem greater_number_problem (a b : ℝ) (h1 : a + b = 40) (h2 : a - b = 12) (h3 : a > b) : a = 26 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_problem_l2349_234999


namespace NUMINAMATH_CALUDE_fry_costs_60_cents_l2349_234950

-- Define the costs in cents
def burger_cost : ℕ := 80
def soda_cost : ℕ := 60

-- Define the total costs of Alice's and Bill's purchases in cents
def alice_total : ℕ := 420
def bill_total : ℕ := 340

-- Define the function to calculate the cost of a fry
def fry_cost : ℕ :=
  alice_total - 3 * burger_cost - 2 * soda_cost

-- Theorem to prove
theorem fry_costs_60_cents :
  fry_cost = 60 ∧
  2 * burger_cost + soda_cost + 2 * fry_cost = bill_total :=
by sorry

end NUMINAMATH_CALUDE_fry_costs_60_cents_l2349_234950


namespace NUMINAMATH_CALUDE_gas_used_l2349_234979

theorem gas_used (initial_gas final_gas : ℝ) (h1 : initial_gas = 0.5) (h2 : final_gas = 0.17) :
  initial_gas - final_gas = 0.33 := by
sorry

end NUMINAMATH_CALUDE_gas_used_l2349_234979


namespace NUMINAMATH_CALUDE_tea_mixture_ratio_l2349_234975

/-- Proves that the ratio of tea at Rs. 64 per kg to tea at Rs. 74 per kg is 1:1 in a mixture worth Rs. 69 per kg -/
theorem tea_mixture_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  64 * x + 74 * y = 69 * (x + y) → x = y := by
  sorry

end NUMINAMATH_CALUDE_tea_mixture_ratio_l2349_234975


namespace NUMINAMATH_CALUDE_store_shelves_theorem_l2349_234951

/-- Calculates the number of shelves needed to display coloring books -/
def shelves_needed (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) : ℕ :=
  (initial_stock - books_sold) / books_per_shelf

/-- Theorem: Given the specific conditions, the number of shelves used is 9 -/
theorem store_shelves_theorem :
  shelves_needed 87 33 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_store_shelves_theorem_l2349_234951


namespace NUMINAMATH_CALUDE_area_of_folded_rectangle_l2349_234903

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a rectangle -/
structure Rectangle :=
  (A B C D : Point)

/-- Represents the folded configuration -/
structure FoldedConfig :=
  (rect : Rectangle)
  (E F B' C' : Point)

/-- The main theorem -/
theorem area_of_folded_rectangle 
  (config : FoldedConfig) 
  (h1 : config.rect.A.x < config.E.x) -- E is on AB
  (h2 : config.rect.C.x > config.F.x) -- F is on CD
  (h3 : config.E.x - config.rect.B.x < config.rect.C.x - config.F.x) -- BE < CF
  (h4 : config.C'.y = config.rect.A.y) -- C' is on AD
  (h5 : (config.B'.x - config.rect.A.x) * (config.C'.y - config.E.y) = 
        (config.C'.x - config.rect.A.x) * (config.B'.y - config.E.y)) -- ∠AB'C' ≅ ∠B'EA
  (h6 : Real.sqrt ((config.B'.x - config.rect.A.x)^2 + (config.B'.y - config.rect.A.y)^2) = 7) -- AB' = 7
  (h7 : config.E.x - config.rect.B.x = 17) -- BE = 17
  : (config.rect.B.x - config.rect.A.x) * (config.rect.C.y - config.rect.A.y) = 
    (1372 + 833 * Real.sqrt 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_area_of_folded_rectangle_l2349_234903


namespace NUMINAMATH_CALUDE_min_n_for_constant_term_l2349_234906

theorem min_n_for_constant_term (x : ℝ) (x_ne_zero : x ≠ 0) : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), (n.choose k) * (-1)^k * x^(n - 8*k) = 1) ∧
  (∀ (m : ℕ), m > 0 ∧ m < n → 
    ¬(∃ (k : ℕ), (m.choose k) * (-1)^k * x^(m - 8*k) = 1)) ∧
  n = 8 :=
sorry

end NUMINAMATH_CALUDE_min_n_for_constant_term_l2349_234906


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2349_234970

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Check if a point lies on a line -/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem perpendicular_line_through_point : 
  ∃ (l : Line), 
    perpendicular l { a := 3, b := -5, c := 6 } ∧ 
    point_on_line (-1) 2 l ∧
    l = { a := 5, b := 3, c := -1 } :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2349_234970


namespace NUMINAMATH_CALUDE_center_coordinates_sum_l2349_234907

/-- The sum of the coordinates of the center of the circle given by x^2 + y^2 = 6x - 8y + 18 is -1 -/
theorem center_coordinates_sum (x y : ℝ) : 
  (x^2 + y^2 = 6*x - 8*y + 18) → (∃ a b : ℝ, (x - a)^2 + (y - b)^2 = (x^2 + y^2 - 6*x + 8*y - 18) ∧ a + b = -1) :=
by sorry

end NUMINAMATH_CALUDE_center_coordinates_sum_l2349_234907


namespace NUMINAMATH_CALUDE_gymnastics_competition_participants_l2349_234962

/-- Represents the structure of a gymnastics competition layout --/
structure GymnasticsCompetition where
  rows : ℕ
  columns : ℕ
  front_position : ℕ
  back_position : ℕ
  left_position : ℕ
  right_position : ℕ

/-- Calculates the total number of participants in the gymnastics competition --/
def total_participants (gc : GymnasticsCompetition) : ℕ :=
  gc.rows * gc.columns

/-- Theorem stating that the total number of participants is 425 --/
theorem gymnastics_competition_participants :
  ∀ (gc : GymnasticsCompetition),
    gc.front_position = 6 →
    gc.back_position = 12 →
    gc.left_position = 15 →
    gc.right_position = 11 →
    gc.columns = gc.front_position + gc.back_position - 1 →
    gc.rows = gc.left_position + gc.right_position - 1 →
    total_participants gc = 425 := by
  sorry

#check gymnastics_competition_participants

end NUMINAMATH_CALUDE_gymnastics_competition_participants_l2349_234962


namespace NUMINAMATH_CALUDE_absolute_value_equation_solutions_l2349_234929

theorem absolute_value_equation_solutions :
  let S : Set ℝ := {x | |x + 1| * |x - 2| * |x + 3| * |x - 4| = |x - 1| * |x + 2| * |x - 3| * |x + 4|}
  S = {0, Real.sqrt 7, -Real.sqrt 7, 
       Real.sqrt ((13 + Real.sqrt 73) / 2), -Real.sqrt ((13 + Real.sqrt 73) / 2),
       Real.sqrt ((13 - Real.sqrt 73) / 2), -Real.sqrt ((13 - Real.sqrt 73) / 2)} := by
  sorry


end NUMINAMATH_CALUDE_absolute_value_equation_solutions_l2349_234929


namespace NUMINAMATH_CALUDE_problem_figure_total_triangles_l2349_234964

/-- Represents a triangular figure composed of equilateral triangles --/
structure TriangularFigure where
  rows : ℕ
  bottom_row_count : ℕ

/-- Calculates the total number of triangles in the figure --/
def total_triangles (figure : TriangularFigure) : ℕ :=
  sorry

/-- The specific triangular figure described in the problem --/
def problem_figure : TriangularFigure :=
  { rows := 4
  , bottom_row_count := 4 }

/-- Theorem stating that the total number of triangles in the problem figure is 16 --/
theorem problem_figure_total_triangles :
  total_triangles problem_figure = 16 := by sorry

end NUMINAMATH_CALUDE_problem_figure_total_triangles_l2349_234964


namespace NUMINAMATH_CALUDE_least_exponent_sum_for_2000_l2349_234996

def is_valid_representation (powers : List ℤ) : Prop :=
  (2000 : ℚ) = (powers.map (λ x => (2 : ℚ) ^ x)).sum ∧
  powers.Nodup ∧
  ∃ x ∈ powers, x < 0

theorem least_exponent_sum_for_2000 :
  ∃ (powers : List ℤ),
    is_valid_representation powers ∧
    ∀ (other_powers : List ℤ),
      is_valid_representation other_powers →
      (powers.sum ≤ other_powers.sum) :=
by sorry

end NUMINAMATH_CALUDE_least_exponent_sum_for_2000_l2349_234996
