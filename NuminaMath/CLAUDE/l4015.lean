import Mathlib

namespace NUMINAMATH_CALUDE_max_profit_at_optimal_price_l4015_401503

/-- Represents the product pricing and sales model -/
structure ProductModel where
  initial_price : ℝ
  initial_sales : ℝ
  price_demand_slope : ℝ
  cost_price : ℝ

/-- Calculates the profit function for a given price decrease -/
def profit_function (model : ProductModel) (x : ℝ) : ℝ :=
  let new_price := model.initial_price - x
  let new_sales := model.initial_sales + model.price_demand_slope * x
  (new_price - model.cost_price) * new_sales

/-- Theorem stating the maximum profit and optimal price decrease -/
theorem max_profit_at_optimal_price (model : ProductModel) 
  (h_initial_price : model.initial_price = 60)
  (h_initial_sales : model.initial_sales = 300)
  (h_price_demand_slope : model.price_demand_slope = 30)
  (h_cost_price : model.cost_price = 40) :
  ∃ (x : ℝ), 
    x = 5 ∧ 
    profit_function model x = 6750 ∧ 
    ∀ (y : ℝ), profit_function model y ≤ profit_function model x :=
  sorry

#eval profit_function ⟨60, 300, 30, 40⟩ 5

end NUMINAMATH_CALUDE_max_profit_at_optimal_price_l4015_401503


namespace NUMINAMATH_CALUDE_least_abaaba_six_primes_l4015_401523

def is_abaaba_form (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ a < 10 ∧ b < 10 ∧
  n = a * 100000 + b * 10000 + a * 1000 + a * 100 + b * 10 + a

def is_product_of_six_distinct_primes (n : ℕ) : Prop :=
  ∃ (p₁ p₂ p₃ p₄ p₅ p₆ : ℕ),
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧ Prime p₆ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ p₁ ≠ p₆ ∧
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ p₂ ≠ p₆ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ p₃ ≠ p₆ ∧
    p₄ ≠ p₅ ∧ p₄ ≠ p₆ ∧
    p₅ ≠ p₆ ∧
    n = p₁ * p₂ * p₃ * p₄ * p₅ * p₆

theorem least_abaaba_six_primes :
  (is_abaaba_form 282282 ∧ is_product_of_six_distinct_primes 282282) ∧
  (∀ n : ℕ, n < 282282 → ¬(is_abaaba_form n ∧ is_product_of_six_distinct_primes n)) :=
by sorry

end NUMINAMATH_CALUDE_least_abaaba_six_primes_l4015_401523


namespace NUMINAMATH_CALUDE_problem_2005_squared_minus_2003_times_2007_l4015_401548

theorem problem_2005_squared_minus_2003_times_2007 : 2005^2 - 2003 * 2007 = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_2005_squared_minus_2003_times_2007_l4015_401548


namespace NUMINAMATH_CALUDE_frog_jump_probability_l4015_401551

/-- Represents a point on the grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- The grid dimensions -/
def gridWidth : ℕ := 7
def gridHeight : ℕ := 5

/-- The jump distance -/
def jumpDistance : ℕ := 2

/-- The starting point of the frog -/
def startPoint : Point := ⟨2, 3⟩

/-- Predicate to check if a point is on a horizontal edge -/
def isOnHorizontalEdge (p : Point) : Prop :=
  p.y = 0 ∨ p.y = gridHeight

/-- Predicate to check if a point is on the grid -/
def isOnGrid (p : Point) : Prop :=
  p.x ≤ gridWidth ∧ p.y ≤ gridHeight

/-- The probability of reaching a horizontal edge from a given point -/
noncomputable def probReachHorizontalEdge (p : Point) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem frog_jump_probability :
  probReachHorizontalEdge startPoint = 3/4 :=
sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l4015_401551


namespace NUMINAMATH_CALUDE_digitCubeSequence_1729th_term_l4015_401514

/-- Sum of cubes of digits of a natural number -/
def sumCubesOfDigits (n : ℕ) : ℕ := sorry

/-- The sequence defined by the sum of cubes of digits -/
def digitCubeSequence : ℕ → ℕ
  | 0 => 1729
  | n + 1 => sumCubesOfDigits (digitCubeSequence n)

/-- The 1729th term of the digit cube sequence is 370 -/
theorem digitCubeSequence_1729th_term :
  digitCubeSequence 1728 = 370 := by sorry

end NUMINAMATH_CALUDE_digitCubeSequence_1729th_term_l4015_401514


namespace NUMINAMATH_CALUDE_multiplicative_inverse_modulo_l4015_401515

def A : Nat := 123456
def B : Nat := 162738
def M : Nat := 1000000
def N : Nat := 503339

theorem multiplicative_inverse_modulo :
  (A * B * N) % M = 1 := by
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_modulo_l4015_401515


namespace NUMINAMATH_CALUDE_hat_and_glasses_probability_l4015_401508

theorem hat_and_glasses_probability
  (total_hats : ℕ)
  (total_glasses : ℕ)
  (prob_hat_given_glasses : ℚ)
  (h1 : total_hats = 60)
  (h2 : total_glasses = 40)
  (h3 : prob_hat_given_glasses = 1 / 4) :
  (total_glasses : ℚ) * prob_hat_given_glasses / total_hats = 1 / 6 :=
by sorry

end NUMINAMATH_CALUDE_hat_and_glasses_probability_l4015_401508


namespace NUMINAMATH_CALUDE_megan_folders_l4015_401599

theorem megan_folders (initial_files : ℕ) (deleted_files : ℕ) (files_per_folder : ℕ) : 
  initial_files = 93 → 
  deleted_files = 21 → 
  files_per_folder = 8 → 
  (initial_files - deleted_files) / files_per_folder = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_megan_folders_l4015_401599


namespace NUMINAMATH_CALUDE_f_has_two_zeros_l4015_401568

-- Define the function f
def f (x : ℝ) : ℝ := 2*x - 3*x

-- Theorem statement
theorem f_has_two_zeros :
  ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b :=
sorry

end NUMINAMATH_CALUDE_f_has_two_zeros_l4015_401568


namespace NUMINAMATH_CALUDE_symmetry_coordinates_l4015_401532

/-- Given two points are symmetric with respect to the origin, 
    their coordinates are negatives of each other -/
def symmetric_wrt_origin (A B : ℝ × ℝ) : Prop :=
  B.1 = -A.1 ∧ B.2 = -A.2

theorem symmetry_coordinates :
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (2, -3)
  symmetric_wrt_origin A B → B = (2, -3) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_coordinates_l4015_401532


namespace NUMINAMATH_CALUDE_bob_distance_walked_l4015_401552

theorem bob_distance_walked (total_distance : ℝ) (yolanda_rate : ℝ) (bob_rate : ℝ) 
  (h1 : total_distance = 80)
  (h2 : yolanda_rate = 8)
  (h3 : bob_rate = 9) : 
  ∃ t : ℝ, t * (yolanda_rate + bob_rate) = total_distance - yolanda_rate ∧ 
  bob_rate * t = 648 / 17 := by
sorry

end NUMINAMATH_CALUDE_bob_distance_walked_l4015_401552


namespace NUMINAMATH_CALUDE_divisibility_of_power_plus_exponent_l4015_401531

theorem divisibility_of_power_plus_exponent (n : ℕ) (hn : 0 < n) :
  ∃ m : ℕ, n ∣ (2^m + m) :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_power_plus_exponent_l4015_401531


namespace NUMINAMATH_CALUDE_rectangular_plot_ratio_l4015_401595

/-- Proves that for a rectangular plot with given area and breadth, the ratio of length to breadth is 3:1 -/
theorem rectangular_plot_ratio (area : ℝ) (breadth : ℝ) (length : ℝ) : 
  area = 972 →
  breadth = 18 →
  area = length * breadth →
  length / breadth = 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_ratio_l4015_401595


namespace NUMINAMATH_CALUDE_problem_solution_l4015_401507

noncomputable section

def m (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (x / 3), Real.cos (x / 3))
def n (x : ℝ) : ℝ × ℝ := (Real.cos (x / 3), Real.cos (x / 3))
def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

def a : ℝ := 2

variable (A B C : ℝ)
variable (b c : ℝ)

axiom triangle_condition : (2 * a - b) * Real.cos C = c * Real.cos B
axiom f_of_A : f A = 3 / 2

theorem problem_solution :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ ∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧
  (∃ k : ℤ, ∀ x : ℝ, f ((-π/4 + 3*π/2*↑k) + x) = f ((-π/4 + 3*π/2*↑k) - x)) ∧
  c = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4015_401507


namespace NUMINAMATH_CALUDE_percentage_of_female_brunettes_l4015_401516

theorem percentage_of_female_brunettes 
  (total_students : ℕ) 
  (female_percentage : ℚ)
  (short_brunette_percentage : ℚ)
  (short_brunette_count : ℕ) :
  total_students = 200 →
  female_percentage = 3/5 →
  short_brunette_percentage = 1/2 →
  short_brunette_count = 30 →
  (short_brunette_count : ℚ) / (short_brunette_percentage * (female_percentage * total_students)) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_female_brunettes_l4015_401516


namespace NUMINAMATH_CALUDE_braden_winnings_l4015_401558

/-- The amount of money Braden has after winning two bets, given his initial amount --/
def final_amount (initial_amount : ℕ) : ℕ :=
  initial_amount + 2 * initial_amount + 2 * initial_amount

/-- Theorem stating that Braden's final amount is $2000 given an initial amount of $400 --/
theorem braden_winnings :
  final_amount 400 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_braden_winnings_l4015_401558


namespace NUMINAMATH_CALUDE_bryan_annual_commute_hours_l4015_401530

/-- Represents the time in minutes for each segment of Bryan's commute -/
structure CommuteSegment where
  walk_to_bus : ℕ
  bus_ride : ℕ
  walk_to_work : ℕ

/-- Represents Bryan's daily commute -/
def daily_commute : CommuteSegment :=
  { walk_to_bus := 5
  , bus_ride := 20
  , walk_to_work := 5 }

/-- Calculates the total time for a one-way commute in minutes -/
def one_way_commute_time (c : CommuteSegment) : ℕ :=
  c.walk_to_bus + c.bus_ride + c.walk_to_work

/-- Calculates the total daily commute time in hours -/
def daily_commute_hours (c : CommuteSegment) : ℚ :=
  (2 * one_way_commute_time c : ℚ) / 60

/-- The number of days Bryan works per year -/
def work_days_per_year : ℕ := 365

/-- Theorem stating that Bryan spends 365 hours per year commuting -/
theorem bryan_annual_commute_hours :
  (daily_commute_hours daily_commute * work_days_per_year : ℚ) = 365 := by
  sorry


end NUMINAMATH_CALUDE_bryan_annual_commute_hours_l4015_401530


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l4015_401555

theorem smallest_fraction_between (p q : ℕ) : 
  p > 0 → q > 0 → (7 : ℚ)/12 < p/q → p/q < 5/8 → 
  (∀ p' q' : ℕ, p' > 0 → q' > 0 → q' < q → (7 : ℚ)/12 < p'/q' → p'/q' < 5/8 → False) →
  q - p = 2 := by
sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l4015_401555


namespace NUMINAMATH_CALUDE_max_quarters_sasha_l4015_401524

/-- Represents the value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- Represents the value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- Represents the total amount of money Sasha has in dollars -/
def total_money : ℚ := 4.80

/-- 
Given that Sasha has $4.80 in U.S. coins and three times as many nickels as quarters,
prove that the maximum number of quarters she could have is 12.
-/
theorem max_quarters_sasha : 
  ∃ (q : ℕ), q ≤ 12 ∧ 
  q * quarter_value + 3 * q * nickel_value = total_money ∧
  ∀ (n : ℕ), n * quarter_value + 3 * n * nickel_value = total_money → n ≤ q :=
by sorry

end NUMINAMATH_CALUDE_max_quarters_sasha_l4015_401524


namespace NUMINAMATH_CALUDE_bobby_jump_improvement_l4015_401577

/-- Bobby's jump rope ability as a child and adult -/
def bobby_jumps : ℕ × ℕ := (30, 60)

/-- The difference in jumps per minute between Bobby as an adult and as a child -/
def jump_difference : ℕ := bobby_jumps.2 - bobby_jumps.1

theorem bobby_jump_improvement : jump_difference = 30 := by
  sorry

end NUMINAMATH_CALUDE_bobby_jump_improvement_l4015_401577


namespace NUMINAMATH_CALUDE_combinatorial_equation_l4015_401591

theorem combinatorial_equation (n : ℕ) : (Nat.choose (n + 1) (n - 1) = 28) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_combinatorial_equation_l4015_401591


namespace NUMINAMATH_CALUDE_project_popularity_order_l4015_401561

def park_renovation : ℚ := 9 / 24
def new_library : ℚ := 10 / 30
def street_lighting : ℚ := 7 / 21
def community_garden : ℚ := 8 / 24

theorem project_popularity_order :
  park_renovation > community_garden ∧
  community_garden = new_library ∧
  new_library = street_lighting ∧
  park_renovation > new_library :=
by sorry

end NUMINAMATH_CALUDE_project_popularity_order_l4015_401561


namespace NUMINAMATH_CALUDE_ned_earnings_l4015_401586

/-- Calculates the total earnings from selling working video games -/
def calculate_earnings (total_games : ℕ) (non_working_games : ℕ) (price_per_game : ℕ) : ℕ :=
  (total_games - non_working_games) * price_per_game

/-- Proves that Ned's earnings from selling his working video games is $63 -/
theorem ned_earnings :
  calculate_earnings 15 6 7 = 63 := by
  sorry

end NUMINAMATH_CALUDE_ned_earnings_l4015_401586


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l4015_401553

theorem quadratic_form_ratio (k : ℝ) : ∃ (d r s : ℝ),
  5 * k^2 - 6 * k + 15 = d * (k + r)^2 + s ∧ s / r = -22 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l4015_401553


namespace NUMINAMATH_CALUDE_common_roots_product_l4015_401583

/-- Given two polynomials that share exactly two roots, prove that the product of these common roots is 1/3 -/
theorem common_roots_product (C D : ℝ) : 
  ∃ (p q r s t u : ℝ),
    (∀ x : ℝ, x^4 - 3*x^3 + C*x + 24 = (x - p)*(x - q)*(x - r)*(x - s)) ∧
    (∀ x : ℝ, x^4 - D*x^3 + 4*x^2 + 72 = (x - p)*(x - q)*(x - t)*(x - u)) ∧
    p ≠ q ∧ 
    p * q = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_common_roots_product_l4015_401583


namespace NUMINAMATH_CALUDE_hyperbola_equation_from_parameters_l4015_401520

/-- Represents a hyperbola with center at the origin and foci on the x-axis -/
structure Hyperbola where
  focal_width : ℝ
  eccentricity : ℝ

/-- The equation of a hyperbola given its parameters -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 12 = 1

/-- Theorem stating that a hyperbola with given parameters has the specified equation -/
theorem hyperbola_equation_from_parameters (h : Hyperbola) 
  (hw : h.focal_width = 8) 
  (he : h.eccentricity = 2) : 
  ∀ x y : ℝ, hyperbola_equation h x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_from_parameters_l4015_401520


namespace NUMINAMATH_CALUDE_grace_age_l4015_401597

/-- Represents the ages of the people in the problem -/
structure Ages where
  grace : ℕ
  faye : ℕ
  chad : ℕ
  eduardo : ℕ
  diana : ℕ

/-- Defines the age relationships between the people -/
def valid_ages (a : Ages) : Prop :=
  a.faye = a.grace + 6 ∧
  a.faye = a.chad + 2 ∧
  a.eduardo = a.chad + 3 ∧
  a.eduardo = a.diana + 4 ∧
  a.diana = 17

/-- Theorem stating that if the ages are valid, Grace's age is 14 -/
theorem grace_age (a : Ages) : valid_ages a → a.grace = 14 := by
  sorry

end NUMINAMATH_CALUDE_grace_age_l4015_401597


namespace NUMINAMATH_CALUDE_ray_gave_peter_30_cents_l4015_401529

/-- Given that Ray has 175 cents in nickels, gives twice as many cents to Randi as to Peter,
    and Randi has 6 more nickels than Peter, prove that Ray gave 30 cents to Peter. -/
theorem ray_gave_peter_30_cents (total : ℕ) (peter_cents : ℕ) (randi_cents : ℕ) : 
  total = 175 →
  randi_cents = 2 * peter_cents →
  randi_cents = peter_cents + 6 * 5 →
  peter_cents = 30 := by
sorry

end NUMINAMATH_CALUDE_ray_gave_peter_30_cents_l4015_401529


namespace NUMINAMATH_CALUDE_roots_of_quadratic_equation_l4015_401501

theorem roots_of_quadratic_equation :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x
  (f 0 = 0) ∧ (f 2 = 0) ∧ 
  (∀ x : ℝ, f x = 0 → x = 0 ∨ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_equation_l4015_401501


namespace NUMINAMATH_CALUDE_no_perfect_square_in_range_l4015_401560

theorem no_perfect_square_in_range : 
  ∀ n : ℕ, 5 ≤ n ∧ n ≤ 15 → ¬∃ m : ℕ, 2 * n^2 + 3 * n + 2 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_in_range_l4015_401560


namespace NUMINAMATH_CALUDE_paint_remaining_rooms_l4015_401582

def time_to_paint_remaining (total_rooms : ℕ) (time_per_room : ℕ) (painted_rooms : ℕ) : ℕ :=
  (total_rooms - painted_rooms) * time_per_room

theorem paint_remaining_rooms 
  (total_rooms : ℕ) 
  (time_per_room : ℕ) 
  (painted_rooms : ℕ) 
  (h1 : total_rooms = 11) 
  (h2 : time_per_room = 7) 
  (h3 : painted_rooms = 2) : 
  time_to_paint_remaining total_rooms time_per_room painted_rooms = 63 := by
sorry

end NUMINAMATH_CALUDE_paint_remaining_rooms_l4015_401582


namespace NUMINAMATH_CALUDE_A_equals_nine_l4015_401521

/-- A3 is a two-digit number -/
def A3 : ℕ := sorry

/-- A is the tens digit of A3 -/
def A : ℕ := A3 / 10

/-- The ones digit of A3 -/
def B : ℕ := A3 % 10

/-- A3 is a two-digit number -/
axiom A3_two_digit : 10 ≤ A3 ∧ A3 ≤ 99

/-- A3 - 41 = 52 -/
axiom A3_equation : A3 - 41 = 52

theorem A_equals_nine : A = 9 := by
  sorry

end NUMINAMATH_CALUDE_A_equals_nine_l4015_401521


namespace NUMINAMATH_CALUDE_cost_of_one_sandwich_and_juice_l4015_401592

/-- Given the cost of multiple items, calculate the cost of one item and one juice -/
theorem cost_of_one_sandwich_and_juice 
  (juice_cost : ℝ) 
  (juice_count : ℕ) 
  (sandwich_cost : ℝ) 
  (sandwich_count : ℕ) : 
  juice_cost / juice_count + sandwich_cost / sandwich_count = 5 :=
by
  sorry

#check cost_of_one_sandwich_and_juice 10 5 6 2

end NUMINAMATH_CALUDE_cost_of_one_sandwich_and_juice_l4015_401592


namespace NUMINAMATH_CALUDE_polynomial_equality_l4015_401522

theorem polynomial_equality (P : ℝ → ℝ) :
  (∀ m : ℝ, P m - (4 * m^3 + m^2 + 5) = 3 * m^4 - 4 * m^3 - m^2 + m - 8) →
  (∀ m : ℝ, P m = 3 * m^4 + m - 3) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l4015_401522


namespace NUMINAMATH_CALUDE_jelly_cost_for_sandwiches_l4015_401556

/-- The cost of jelly for N sandwiches -/
def jelly_cost (N B J : ℕ+) : ℚ :=
  (N * J * 7 : ℚ) / 100

/-- The total cost of peanut butter and jelly for N sandwiches -/
def total_cost (N B J : ℕ+) : ℚ :=
  (N * (3 * B + 7 * J) : ℚ) / 100

theorem jelly_cost_for_sandwiches
  (N B J : ℕ+)
  (h1 : total_cost N B J = 252 / 100)
  (h2 : N > 1) :
  jelly_cost N B J = 168 / 100 :=
by sorry

end NUMINAMATH_CALUDE_jelly_cost_for_sandwiches_l4015_401556


namespace NUMINAMATH_CALUDE_corn_stalks_per_row_l4015_401537

/-- Proves that given 5 rows of corn, 8 corn stalks per bushel, and a total harvest of 50 bushels,
    the number of corn stalks in each row is 80. -/
theorem corn_stalks_per_row 
  (rows : ℕ) 
  (stalks_per_bushel : ℕ) 
  (total_bushels : ℕ) 
  (h1 : rows = 5)
  (h2 : stalks_per_bushel = 8)
  (h3 : total_bushels = 50) :
  (total_bushels * stalks_per_bushel) / rows = 80 := by
  sorry

end NUMINAMATH_CALUDE_corn_stalks_per_row_l4015_401537


namespace NUMINAMATH_CALUDE_medicine_supply_duration_l4015_401543

/-- Represents the duration of the medicine supply in months -/
def medicine_duration (pills : ℕ) (pill_fraction : ℚ) (days_between_doses : ℕ) : ℚ :=
  let days_per_pill := (days_between_doses : ℚ) / pill_fraction
  let total_days := (pills : ℚ) * days_per_pill
  let days_per_month := 30
  total_days / days_per_month

/-- The theorem stating that the given medicine supply lasts 18 months -/
theorem medicine_supply_duration :
  medicine_duration 60 (1/3) 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_medicine_supply_duration_l4015_401543


namespace NUMINAMATH_CALUDE_function_property_l4015_401557

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 4

theorem function_property (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ 2 ∧ 0 ≤ x₂ ∧ x₂ ≤ 2 → |f a x₁ - f a x₂| < 4) →
  0 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l4015_401557


namespace NUMINAMATH_CALUDE_rhombus_area_l4015_401510

/-- A rhombus with perimeter 20cm and diagonals in ratio 4:3 has an area of 24cm². -/
theorem rhombus_area (d₁ d₂ : ℝ) : 
  d₁ > 0 → d₂ > 0 →  -- diagonals are positive
  d₁ / d₂ = 4 / 3 →  -- ratio of diagonals is 4:3
  (d₁^2 + d₂^2) / 2 = 25 →  -- perimeter is 20cm (side length is 5cm)
  d₁ * d₂ / 2 = 24 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_l4015_401510


namespace NUMINAMATH_CALUDE_point_c_coordinates_l4015_401580

/-- Given points A and B in ℝ³, if vector AC is half of vector AB, then C has specific coordinates -/
theorem point_c_coordinates (A B C : ℝ × ℝ × ℝ) : 
  A = (2, 2, 7) → 
  B = (-2, 4, 3) → 
  C - A = (1 / 2 : ℝ) • (B - A) → 
  C = (0, 3, 5) := by
  sorry

end NUMINAMATH_CALUDE_point_c_coordinates_l4015_401580


namespace NUMINAMATH_CALUDE_age_ratio_problem_l4015_401533

theorem age_ratio_problem (amy jeremy chris : ℕ) : 
  amy + jeremy + chris = 132 →
  amy = jeremy / 3 →
  jeremy = 66 →
  ∃ k : ℕ, chris = k * amy →
  chris / amy = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l4015_401533


namespace NUMINAMATH_CALUDE_spherical_coord_transformation_l4015_401540

/-- Given a point with rectangular coordinates (a, b, c) and 
    spherical coordinates (3, 3π/4, π/6), prove that the point 
    with rectangular coordinates (a, -b, c) has spherical 
    coordinates (3, 7π/4, π/6) -/
theorem spherical_coord_transformation 
  (a b c : ℝ) 
  (h1 : a = 3 * Real.sin (π/6) * Real.cos (3*π/4))
  (h2 : b = 3 * Real.sin (π/6) * Real.sin (3*π/4))
  (h3 : c = 3 * Real.cos (π/6)) :
  ∃ (ρ θ φ : ℝ), 
    ρ = 3 ∧ 
    θ = 7*π/4 ∧ 
    φ = π/6 ∧
    a = ρ * Real.sin φ * Real.cos θ ∧
    -b = ρ * Real.sin φ * Real.sin θ ∧
    c = ρ * Real.cos φ :=
by sorry

end NUMINAMATH_CALUDE_spherical_coord_transformation_l4015_401540


namespace NUMINAMATH_CALUDE_square_construction_l4015_401593

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane defined by two points -/
structure Line2D where
  p1 : Point2D
  p2 : Point2D

/-- A square in a 2D plane -/
structure Square where
  vertices : Fin 4 → Point2D

/-- Check if a point lies on a line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop := sorry

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line2D) : Prop := sorry

/-- Check if all sides of a square are equal length -/
def equalSides (s : Square) : Prop := sorry

/-- The main theorem -/
theorem square_construction (A B C D : Point2D) :
  ∃ (s : Square),
    (∀ i : Fin 4, ∃ p ∈ [A, B, C, D], pointOnLine (s.vertices i) (Line2D.mk (s.vertices i) (s.vertices ((i + 1) % 4)))) ∧
    (∀ i : Fin 4, perpendicular (Line2D.mk (s.vertices i) (s.vertices ((i + 1) % 4))) (Line2D.mk (s.vertices ((i + 1) % 4)) (s.vertices ((i + 2) % 4)))) ∧
    equalSides s :=
sorry

end NUMINAMATH_CALUDE_square_construction_l4015_401593


namespace NUMINAMATH_CALUDE_sum_of_specific_sequences_l4015_401564

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a₁ + i * d)

def sum_of_sequences (seq1 seq2 : List ℕ) : ℕ :=
  (seq1 ++ seq2).sum

theorem sum_of_specific_sequences :
  let seq1 := arithmetic_sequence 3 10 5
  let seq2 := arithmetic_sequence 7 10 5
  sum_of_sequences seq1 seq2 = 250 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_sequences_l4015_401564


namespace NUMINAMATH_CALUDE_roses_problem_l4015_401572

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := 21

/-- The number of roses Jessica cut from her garden -/
def cut_roses : ℕ := 28

/-- The number of roses Jessica threw away -/
def thrown_roses : ℕ := 34

/-- The number of roses currently in the vase -/
def current_roses : ℕ := 15

theorem roses_problem :
  initial_roses = 21 ∧
  thrown_roses = cut_roses + 6 ∧
  current_roses = initial_roses + cut_roses - thrown_roses :=
by sorry

end NUMINAMATH_CALUDE_roses_problem_l4015_401572


namespace NUMINAMATH_CALUDE_carol_final_score_is_negative_nineteen_l4015_401511

-- Define the scores and multipliers for each round
def first_round_score : Int := 17
def second_round_base_score : Int := 6
def second_round_multiplier : Int := 2
def last_round_base_loss : Int := 16
def last_round_multiplier : Int := 3

-- Define Carol's final score
def carol_final_score : Int := 
  first_round_score + 
  (second_round_base_score * second_round_multiplier) - 
  (last_round_base_loss * last_round_multiplier)

-- Theorem to prove Carol's final score
theorem carol_final_score_is_negative_nineteen : 
  carol_final_score = -19 := by
  sorry

end NUMINAMATH_CALUDE_carol_final_score_is_negative_nineteen_l4015_401511


namespace NUMINAMATH_CALUDE_min_cuts_for_daily_payment_min_cuts_for_all_lengths_l4015_401596

/-- Represents a chain of links -/
structure Chain where
  length : ℕ

/-- Represents a cut strategy for a chain -/
structure CutStrategy where
  cuts : ℕ

/-- Checks if a cut strategy is valid for daily payments -/
def is_valid_daily_payment_strategy (chain : Chain) (strategy : CutStrategy) : Prop :=
  ∀ day : ℕ, day ≤ chain.length → ∃ payment : ℕ, payment = day

/-- Checks if a cut strategy can produce any number of links up to the chain length -/
def can_produce_all_lengths (chain : Chain) (strategy : CutStrategy) : Prop :=
  ∀ n : ℕ, n ≤ chain.length → ∃ combination : List ℕ, combination.sum = n

/-- Theorem for the minimum cuts needed for daily payments -/
theorem min_cuts_for_daily_payment (chain : Chain) (strategy : CutStrategy) : 
  chain.length = 7 → 
  strategy.cuts = 1 → 
  is_valid_daily_payment_strategy chain strategy :=
sorry

/-- Theorem for the minimum cuts needed to produce all lengths -/
theorem min_cuts_for_all_lengths (chain : Chain) (strategy : CutStrategy) :
  chain.length = 2000 →
  strategy.cuts = 7 →
  can_produce_all_lengths chain strategy :=
sorry

end NUMINAMATH_CALUDE_min_cuts_for_daily_payment_min_cuts_for_all_lengths_l4015_401596


namespace NUMINAMATH_CALUDE_egg_collection_theorem_l4015_401527

/-- The number of dozen eggs Benjamin collects per day -/
def benjamin_eggs : ℕ := 6

/-- The number of dozen eggs Carla collects per day -/
def carla_eggs : ℕ := 3 * benjamin_eggs

/-- The number of dozen eggs Trisha collects per day -/
def trisha_eggs : ℕ := benjamin_eggs - 4

/-- The total number of dozen eggs collected by Benjamin, Carla, and Trisha -/
def total_eggs : ℕ := benjamin_eggs + carla_eggs + trisha_eggs

theorem egg_collection_theorem : total_eggs = 26 := by
  sorry

end NUMINAMATH_CALUDE_egg_collection_theorem_l4015_401527


namespace NUMINAMATH_CALUDE_constant_sum_l4015_401504

/-- The number of distinct roots of a rational function -/
noncomputable def distinctRoots (num : ℝ → ℝ) (denom : ℝ → ℝ) : ℕ := sorry

theorem constant_sum (a b : ℝ) : 
  distinctRoots (λ x => (x+a)*(x+b)*(x+10)) (λ x => (x+4)^2) = 3 →
  distinctRoots (λ x => (x+2*a)*(x+4)*(x+5)) (λ x => (x+b)*(x+10)) = 1 →
  100*a + b = 205 := by sorry

end NUMINAMATH_CALUDE_constant_sum_l4015_401504


namespace NUMINAMATH_CALUDE_marble_probability_l4015_401565

/-- The number of blue marbles initially in the bag -/
def blue_marbles : ℕ := 5

/-- The number of white marbles initially in the bag -/
def white_marbles : ℕ := 7

/-- The number of red marbles initially in the bag -/
def red_marbles : ℕ := 4

/-- The total number of marbles initially in the bag -/
def total_marbles : ℕ := blue_marbles + white_marbles + red_marbles

/-- The number of marbles to be drawn -/
def marbles_drawn : ℕ := total_marbles - 2

/-- The probability of having one white and one blue marble remaining after randomly drawing marbles until only two are left -/
theorem marble_probability : 
  (Nat.choose blue_marbles blue_marbles * Nat.choose white_marbles (white_marbles - 1) * Nat.choose red_marbles red_marbles) / 
  Nat.choose total_marbles marbles_drawn = 7 / 120 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l4015_401565


namespace NUMINAMATH_CALUDE_mary_sold_at_least_12_boxes_l4015_401549

/-- The number of cases Mary needs to deliver -/
def cases : ℕ := 2

/-- The number of boxes in each case -/
def boxes_per_case : ℕ := 6

/-- The minimum number of boxes Mary sold -/
def min_boxes_sold : ℕ := cases * boxes_per_case

/-- Mary has some extra boxes (number unspecified) -/
axiom has_extra_boxes : ∃ n : ℕ, n > 0

theorem mary_sold_at_least_12_boxes :
  min_boxes_sold ≥ 12 ∧ ∃ total : ℕ, total > min_boxes_sold :=
sorry

end NUMINAMATH_CALUDE_mary_sold_at_least_12_boxes_l4015_401549


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l4015_401539

noncomputable def U : Set ℝ := Set.univ

def A : Set ℝ := {x | Real.log (x - 2) < 1}

def B : Set ℝ := {x | x^2 - x - 2 < 0}

theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | 2 < x ∧ x < 12} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l4015_401539


namespace NUMINAMATH_CALUDE_sphere_volume_from_box_diagonal_l4015_401570

theorem sphere_volume_from_box_diagonal (a b c : ℝ) (ha : a = 3 * Real.sqrt 2) (hb : b = 4 * Real.sqrt 2) (hc : c = 5 * Real.sqrt 2) :
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  (4 / 3) * Real.pi * (diagonal / 2)^3 = 500 * Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_sphere_volume_from_box_diagonal_l4015_401570


namespace NUMINAMATH_CALUDE_thursday_to_tuesday_ratio_l4015_401518

/-- Represents the number of baseball cards Buddy has on each day --/
structure BuddysCards where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- Represents the number of cards Buddy bought on Thursday --/
def thursdayPurchase (cards : BuddysCards) : ℕ :=
  cards.thursday - cards.wednesday

/-- The theorem stating the ratio of Thursday's purchase to Tuesday's amount --/
theorem thursday_to_tuesday_ratio (cards : BuddysCards) :
  cards.monday = 30 →
  cards.tuesday = cards.monday / 2 →
  cards.wednesday = cards.tuesday + 12 →
  cards.thursday = 32 →
  thursdayPurchase cards * 3 = cards.tuesday := by
  sorry

end NUMINAMATH_CALUDE_thursday_to_tuesday_ratio_l4015_401518


namespace NUMINAMATH_CALUDE_number_of_parents_attending_l4015_401517

/-- The number of parents attending a school meeting -/
theorem number_of_parents_attending (S R B N : ℕ) : 
  S = 25 →  -- number of parents volunteering to supervise
  B = 11 →  -- number of parents volunteering for both supervising and bringing refreshments
  R = 42 →  -- number of parents volunteering to bring refreshments
  R = (3 * N) / 2 →  -- R is 1.5 times N
  S + R - B + N = 95 :=  -- total number of parents
by sorry

end NUMINAMATH_CALUDE_number_of_parents_attending_l4015_401517


namespace NUMINAMATH_CALUDE_maximize_product_l4015_401525

theorem maximize_product (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : x + y = 50) :
  x^3 * y^4 ≤ 30^3 * 20^4 ∧
  (x^3 * y^4 = 30^3 * 20^4 ↔ x = 30 ∧ y = 20) :=
by sorry

end NUMINAMATH_CALUDE_maximize_product_l4015_401525


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l4015_401546

theorem quadratic_inequality_solution (m : ℝ) :
  (∀ x, mx^2 + 8*m*x + 28 < 0 ↔ -7 < x ∧ x < -1) →
  m = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l4015_401546


namespace NUMINAMATH_CALUDE_valid_words_count_l4015_401584

def alphabet_size : ℕ := 15
def max_word_length : ℕ := 5

def total_words (n : ℕ) : ℕ := 
  (alphabet_size ^ 1) + (alphabet_size ^ 2) + (alphabet_size ^ 3) + 
  (alphabet_size ^ 4) + (alphabet_size ^ 5)

def words_without_letter (n : ℕ) : ℕ := 
  ((alphabet_size - 1) ^ 1) + ((alphabet_size - 1) ^ 2) + 
  ((alphabet_size - 1) ^ 3) + ((alphabet_size - 1) ^ 4) + 
  ((alphabet_size - 1) ^ 5)

def words_without_two_letters (n : ℕ) : ℕ := 
  ((alphabet_size - 2) ^ 1) + ((alphabet_size - 2) ^ 2) + 
  ((alphabet_size - 2) ^ 3) + ((alphabet_size - 2) ^ 4) + 
  ((alphabet_size - 2) ^ 5)

theorem valid_words_count : 
  total_words alphabet_size - 2 * words_without_letter alphabet_size + 
  words_without_two_letters alphabet_size = 62460 := by
  sorry

end NUMINAMATH_CALUDE_valid_words_count_l4015_401584


namespace NUMINAMATH_CALUDE_badge_exchange_l4015_401506

theorem badge_exchange (vasya_initial : ℕ) (tolya_initial : ℕ) : 
  (vasya_initial = tolya_initial + 5) →
  (vasya_initial - (24 * vasya_initial) / 100 + (20 * tolya_initial) / 100 = 
   tolya_initial - (20 * tolya_initial) / 100 + (24 * vasya_initial) / 100 - 1) →
  (vasya_initial = 50 ∧ tolya_initial = 45) :=
by sorry

#check badge_exchange

end NUMINAMATH_CALUDE_badge_exchange_l4015_401506


namespace NUMINAMATH_CALUDE_problem_statement_l4015_401536

theorem problem_statement : 3 * 3^4 - 9^32 / 9^30 = 162 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4015_401536


namespace NUMINAMATH_CALUDE_negative_two_cubed_equality_l4015_401566

theorem negative_two_cubed_equality : (-2)^3 = -2^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_cubed_equality_l4015_401566


namespace NUMINAMATH_CALUDE_rainfall_second_week_l4015_401576

theorem rainfall_second_week (total_rainfall : ℝ) (ratio : ℝ) : 
  total_rainfall = 20 →
  ratio = 1.5 →
  ∃ (first_week : ℝ) (second_week : ℝ),
    first_week + second_week = total_rainfall ∧
    second_week = ratio * first_week ∧
    second_week = 12 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_second_week_l4015_401576


namespace NUMINAMATH_CALUDE_complete_square_with_integer_l4015_401535

theorem complete_square_with_integer : 
  ∃ (k : ℤ) (b : ℝ), ∀ (x : ℝ), x^2 + 8*x + 20 = (x + b)^2 + k := by
  sorry

end NUMINAMATH_CALUDE_complete_square_with_integer_l4015_401535


namespace NUMINAMATH_CALUDE_ln_exp_equals_id_l4015_401562

theorem ln_exp_equals_id : ∀ x : ℝ, Real.log (Real.exp x) = x := by sorry

end NUMINAMATH_CALUDE_ln_exp_equals_id_l4015_401562


namespace NUMINAMATH_CALUDE_school_choir_members_l4015_401545

theorem school_choir_members :
  ∃! n : ℕ, 120 ≤ n ∧ n ≤ 300 ∧
  n % 6 = 1 ∧ n % 8 = 5 ∧ n % 9 = 2 ∧ n = 241 :=
by sorry

end NUMINAMATH_CALUDE_school_choir_members_l4015_401545


namespace NUMINAMATH_CALUDE_reeyas_average_score_l4015_401554

theorem reeyas_average_score : 
  let scores : List ℕ := [65, 67, 76, 82, 85]
  (scores.sum / scores.length : ℚ) = 75 := by
  sorry

end NUMINAMATH_CALUDE_reeyas_average_score_l4015_401554


namespace NUMINAMATH_CALUDE_infinite_solutions_and_sum_of_exceptions_l4015_401587

theorem infinite_solutions_and_sum_of_exceptions :
  let A : ℚ := 3
  let B : ℚ := 5
  let C : ℚ := 40/3
  let f (x : ℚ) := (x + B) * (A * x + 40) / ((x + C) * (x + 5))
  (∀ x, x ≠ -C → x ≠ -5 → f x = 3) ∧
  (-5 + (-C) = -55/3) := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_and_sum_of_exceptions_l4015_401587


namespace NUMINAMATH_CALUDE_locus_of_circumcenter_l4015_401509

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcenter
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the product OA · OB · OC
def product (c : Circle) (t : Triangle) : ℝ := sorry

theorem locus_of_circumcenter 
  (c : Circle) 
  (t : Triangle) 
  (p : ℝ) 
  (h : product c t = p^3) :
  ∃ (P : ℝ × ℝ), 
    P = circumcenter t ∧ 
    distance c.center P = (p / (4 * c.radius^2)) * Real.sqrt (p * (p^3 - 8 * c.radius^3)) :=
sorry

end NUMINAMATH_CALUDE_locus_of_circumcenter_l4015_401509


namespace NUMINAMATH_CALUDE_m_range_l4015_401534

-- Define the set A
def A (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 1 < 0}

-- Theorem statement
theorem m_range (m : ℝ) : 1 ∈ A m ∧ 3 ∉ A m → 0 < m ∧ m < 2 :=
by
  sorry

end NUMINAMATH_CALUDE_m_range_l4015_401534


namespace NUMINAMATH_CALUDE_velocity_center_of_mass_before_collision_l4015_401547

/-- Velocity of the center of mass of a two-cart system before collision -/
theorem velocity_center_of_mass_before_collision 
  (m : ℝ) -- mass of cart 1
  (v1_initial : ℝ) -- initial velocity of cart 1
  (m2 : ℝ) -- mass of cart 2
  (v2_initial : ℝ) -- initial velocity of cart 2
  (v1_final : ℝ) -- final velocity of cart 1
  (h1 : v1_initial = 12) -- initial velocity of cart 1 is 12 m/s
  (h2 : m2 = 4) -- mass of cart 2 is 4 kg
  (h3 : v2_initial = 0) -- cart 2 is initially at rest
  (h4 : v1_final = -6) -- final velocity of cart 1 is 6 m/s to the left
  (h5 : m > 0) -- mass of cart 1 is positive
  (h6 : m2 > 0) -- mass of cart 2 is positive
  : (m * v1_initial + m2 * v2_initial) / (m + m2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_velocity_center_of_mass_before_collision_l4015_401547


namespace NUMINAMATH_CALUDE_polynomial_simplification_l4015_401598

theorem polynomial_simplification (x y : ℝ) :
  (10 * x^12 + 8 * x^9 + 5 * x^7) + (11 * x^9 + 3 * x^7 + 4 * x^3 + 6 * y^2 + 7 * x + 9) =
  10 * x^12 + 19 * x^9 + 8 * x^7 + 4 * x^3 + 6 * y^2 + 7 * x + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l4015_401598


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l4015_401589

/-- Given a sector with a central angle of 60° and an arc length of 2π,
    its inscribed circle has a radius of 2. -/
theorem inscribed_circle_radius (θ : ℝ) (arc_length : ℝ) (R : ℝ) (r : ℝ) :
  θ = π / 3 →
  arc_length = 2 * π →
  arc_length = θ * R →
  3 * r = R →
  r = 2 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l4015_401589


namespace NUMINAMATH_CALUDE_square_split_into_pentagons_or_hexagons_l4015_401563

/-- A polygon in 2D space -/
structure Polygon :=
  (vertices : List (ℝ × ℝ))

/-- The number of sides of a polygon -/
def Polygon.sides (p : Polygon) : ℕ := p.vertices.length

/-- A concave polygon -/
def ConcavePolygon (p : Polygon) : Prop := sorry

/-- The area of a polygon -/
def Polygon.area (p : Polygon) : ℝ := sorry

/-- A square with side length 1 -/
def UnitSquare : Polygon := sorry

/-- Two polygons are equal in area -/
def EqualArea (p1 p2 : Polygon) : Prop := p1.area = p2.area

/-- A polygon is contained within another polygon -/
def ContainedIn (p1 p2 : Polygon) : Prop := sorry

/-- The union of two polygons -/
def PolygonUnion (p1 p2 : Polygon) : Polygon := sorry

theorem square_split_into_pentagons_or_hexagons :
  ∃ (p1 p2 : Polygon),
    (p1.sides = 5 ∧ p2.sides = 5 ∨ p1.sides = 6 ∧ p2.sides = 6) ∧
    ConcavePolygon p1 ∧
    ConcavePolygon p2 ∧
    EqualArea p1 p2 ∧
    ContainedIn p1 UnitSquare ∧
    ContainedIn p2 UnitSquare ∧
    PolygonUnion p1 p2 = UnitSquare :=
sorry

end NUMINAMATH_CALUDE_square_split_into_pentagons_or_hexagons_l4015_401563


namespace NUMINAMATH_CALUDE_additional_dividend_calculation_l4015_401500

/-- Calculates the additional dividend per share given expected and actual earnings -/
def additional_dividend (expected_earnings : ℚ) (actual_earnings : ℚ) : ℚ :=
  let earnings_difference := actual_earnings - expected_earnings
  let additional_earnings := max earnings_difference 0
  additional_earnings / 2

/-- Proves that the additional dividend is $0.15 per share given the problem conditions -/
theorem additional_dividend_calculation :
  let expected_earnings : ℚ := 80 / 100
  let actual_earnings : ℚ := 110 / 100
  additional_dividend expected_earnings actual_earnings = 15 / 100 := by
  sorry


end NUMINAMATH_CALUDE_additional_dividend_calculation_l4015_401500


namespace NUMINAMATH_CALUDE_book_reading_rate_l4015_401567

/-- Calculates the number of pages read per day given the total number of pages and days spent reading. -/
def pages_per_day (total_pages : ℕ) (total_days : ℕ) : ℕ :=
  total_pages / total_days

/-- Theorem stating that reading 12518 pages over 569 days results in 22 pages per day. -/
theorem book_reading_rate :
  pages_per_day 12518 569 = 22 := by
  sorry

end NUMINAMATH_CALUDE_book_reading_rate_l4015_401567


namespace NUMINAMATH_CALUDE_card_selection_ways_l4015_401575

-- Define the number of suits in a standard deck
def num_suits : ℕ := 4

-- Define the number of cards per suit in a standard deck
def cards_per_suit : ℕ := 13

-- Define the total number of cards in a standard deck
def total_cards : ℕ := num_suits * cards_per_suit

-- Define the number of cards to choose
def cards_to_choose : ℕ := 4

-- Define the number of cards to keep after discarding
def cards_to_keep : ℕ := 3

-- Theorem statement
theorem card_selection_ways :
  (num_suits.choose cards_to_choose) * (cards_per_suit ^ cards_to_choose) * cards_to_choose = 114244 := by
  sorry

end NUMINAMATH_CALUDE_card_selection_ways_l4015_401575


namespace NUMINAMATH_CALUDE_units_digit_period_four_units_digit_2_power_2012_l4015_401581

/-- The units digit of 2^n -/
def unitsDigit (n : ℕ) : ℕ := 2^n % 10

/-- The pattern of units digits for powers of 2 repeats every 4 steps -/
theorem units_digit_period_four (n : ℕ) : 
  unitsDigit n = unitsDigit (n + 4) :=
sorry

/-- The units digit of 2^2012 is 6 -/
theorem units_digit_2_power_2012 : unitsDigit 2012 = 6 :=
sorry

end NUMINAMATH_CALUDE_units_digit_period_four_units_digit_2_power_2012_l4015_401581


namespace NUMINAMATH_CALUDE_negative_fraction_range_l4015_401594

theorem negative_fraction_range (x : ℝ) : (x - 1) / x^2 < 0 → x < 1 ∧ x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_range_l4015_401594


namespace NUMINAMATH_CALUDE_range_of_m_l4015_401526

def A : Set ℝ := {x | x ≤ -2 ∨ x ≥ -1}
def B (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x ≤ 2*m}

theorem range_of_m (m : ℝ) :
  (A ∩ B m = ∅) → (A ∪ B m = A) → m ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l4015_401526


namespace NUMINAMATH_CALUDE_largest_divisor_of_p_cubed_minus_p_l4015_401512

theorem largest_divisor_of_p_cubed_minus_p (p : ℕ) (hp : Prime p) (hp_ge_5 : p ≥ 5) :
  (∃ (k : ℕ), k * 12 = p^3 - p) ∧
  (∀ (d : ℕ), d > 12 → ¬(∀ (q : ℕ), Prime q → q ≥ 5 → ∃ (k : ℕ), k * d = q^3 - q)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_p_cubed_minus_p_l4015_401512


namespace NUMINAMATH_CALUDE_smallest_number_l4015_401513

theorem smallest_number (s : Set ℝ) (hs : s = {0, -2, 1, (1/2 : ℝ)}) :
  ∃ x ∈ s, ∀ y ∈ s, x ≤ y ∧ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l4015_401513


namespace NUMINAMATH_CALUDE_rectangle_parallelogram_relationship_l4015_401559

-- Define the types
def Parallelogram : Type := sorry
def Rectangle : Type := sorry

-- Define the relationship between Rectangle and Parallelogram
axiom rectangle_is_parallelogram : Rectangle → Parallelogram

-- State the theorem
theorem rectangle_parallelogram_relationship :
  (∀ r : Rectangle, ∃ p : Parallelogram, p = rectangle_is_parallelogram r) ∧
  ¬(∀ p : Parallelogram, ∃ r : Rectangle, p = rectangle_is_parallelogram r) :=
sorry

end NUMINAMATH_CALUDE_rectangle_parallelogram_relationship_l4015_401559


namespace NUMINAMATH_CALUDE_grading_implications_l4015_401573

-- Define the type for grades
inductive Grade : Type
| A : Grade
| B : Grade
| C : Grade
| D : Grade
| F : Grade

-- Define the ordering on grades
instance : LE Grade where
  le := λ g₁ g₂ => match g₁, g₂ with
    | Grade.F, _ => true
    | Grade.D, Grade.D | Grade.D, Grade.C | Grade.D, Grade.B | Grade.D, Grade.A => true
    | Grade.C, Grade.C | Grade.C, Grade.B | Grade.C, Grade.A => true
    | Grade.B, Grade.B | Grade.B, Grade.A => true
    | Grade.A, Grade.A => true
    | _, _ => false

instance : LT Grade where
  lt := λ g₁ g₂ => g₁ ≤ g₂ ∧ g₁ ≠ g₂

-- Define the grading function
def grading_function (score : ℚ) : Grade :=
  if score ≥ 90 then Grade.B
  else if score < 70 then Grade.C
  else Grade.C  -- Default case, can be any grade between B and C

-- State the theorem
theorem grading_implications :
  (∀ (score : ℚ) (grade : Grade),
    (grading_function score = grade → 
      (grade < Grade.B → score < 90) ∧
      (grade > Grade.C → score ≥ 70))) :=
sorry

end NUMINAMATH_CALUDE_grading_implications_l4015_401573


namespace NUMINAMATH_CALUDE_canteen_distance_l4015_401574

theorem canteen_distance (girls_camp_distance boys_camp_distance : ℝ) 
  (h1 : girls_camp_distance = 600)
  (h2 : boys_camp_distance = 800) :
  let hypotenuse := Real.sqrt (girls_camp_distance ^ 2 + boys_camp_distance ^ 2)
  let canteen_distance := Real.sqrt ((girls_camp_distance ^ 2 + (hypotenuse / 2) ^ 2))
  ⌊canteen_distance⌋ = 781 := by
  sorry

end NUMINAMATH_CALUDE_canteen_distance_l4015_401574


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l4015_401579

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∃ P : ℝ × ℝ, (P.1^2 / a^2 - P.2^2 / b^2 = 1) ∧
       (|P.1 - (-c)| + |P.1 - c| = 3*b) ∧
       (|P.1 - (-c)| * |P.1 - c| = 9/4 * a * b))
  (h4 : c^2 = a^2 + b^2) : 
  (c / a : ℝ) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l4015_401579


namespace NUMINAMATH_CALUDE_sum_of_p_and_q_l4015_401541

theorem sum_of_p_and_q (p q : ℤ) : 
  p > 1 → 
  q > 1 → 
  (2 * q - 1) % p = 0 → 
  (2 * p - 1) % q = 0 → 
  p + q = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_p_and_q_l4015_401541


namespace NUMINAMATH_CALUDE_prism_pyramid_height_relation_l4015_401502

/-- Given an equilateral triangle with side length a, prove that if a prism and a pyramid
    are constructed on this triangle with height m, and the lateral surface area of the prism
    equals the lateral surface area of the pyramid, then m = a/6 -/
theorem prism_pyramid_height_relation (a : ℝ) (m : ℝ) (h_pos : a > 0) : 
  (3 * a * m = (3 * a / 2) * Real.sqrt (m^2 + a^2 / 12)) → m = a / 6 := by
  sorry

end NUMINAMATH_CALUDE_prism_pyramid_height_relation_l4015_401502


namespace NUMINAMATH_CALUDE_hexagon_angle_sum_l4015_401542

-- Define the hexagon and its properties
structure Hexagon where
  A : ℝ
  B : ℝ
  C : ℝ
  x : ℝ
  y : ℝ
  h : A = 34
  i : B = 74
  j : C = 32

-- State the theorem
theorem hexagon_angle_sum (H : Hexagon) : H.x + H.y = 40 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angle_sum_l4015_401542


namespace NUMINAMATH_CALUDE_points_on_line_l4015_401528

/-- A line in the 2D plane defined by the equation 7x + 2y = 41 -/
def line (x y : ℝ) : Prop := 7 * x + 2 * y = 41

/-- Point A with coordinates (5, 3) -/
def point_A : ℝ × ℝ := (5, 3)

/-- Point B with coordinates (-5, 38) -/
def point_B : ℝ × ℝ := (-5, 38)

/-- Theorem stating that points A and B lie on the given line -/
theorem points_on_line :
  line point_A.1 point_A.2 ∧ line point_B.1 point_B.2 := by
  sorry

end NUMINAMATH_CALUDE_points_on_line_l4015_401528


namespace NUMINAMATH_CALUDE_daily_earnings_a_and_c_l4015_401538

/-- Given three workers a, b, and c, with their daily earnings, prove that a and c together earn $400 per day. -/
theorem daily_earnings_a_and_c (a b c : ℕ) : 
  a + b + c = 600 →  -- Total earnings of a, b, and c
  b + c = 300 →      -- Combined earnings of b and c
  c = 100 →          -- Earnings of c
  a + c = 400 :=     -- Combined earnings of a and c
by
  sorry

end NUMINAMATH_CALUDE_daily_earnings_a_and_c_l4015_401538


namespace NUMINAMATH_CALUDE_triangle_area_l4015_401571

-- Define the curve
def curve (x : ℝ) : ℝ := (x - 4)^2 * (x + 3)

-- Define the x-intercepts
def x_intercept1 : ℝ := 4
def x_intercept2 : ℝ := -3

-- Define the y-intercept
def y_intercept : ℝ := curve 0

-- Theorem statement
theorem triangle_area : 
  let base := x_intercept1 - x_intercept2
  let height := y_intercept
  (1/2 : ℝ) * base * height = 168 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l4015_401571


namespace NUMINAMATH_CALUDE_apple_box_weight_l4015_401578

theorem apple_box_weight (n : ℕ) (w : ℝ) (h1 : n = 5) (h2 : w > 30) 
  (h3 : n * (w - 30) = 2 * w) : w = 50 := by
  sorry

end NUMINAMATH_CALUDE_apple_box_weight_l4015_401578


namespace NUMINAMATH_CALUDE_quadratic_function_unique_coefficients_l4015_401569

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 - 2 * a * x + 1 + b

theorem quadratic_function_unique_coefficients 
  (a b : ℝ) 
  (h_a_pos : a > 0) 
  (h_max : ∀ x ∈ Set.Icc 2 3, f a b x ≤ 4) 
  (h_min : ∀ x ∈ Set.Icc 2 3, f a b x ≥ 1) 
  (h_max_achieved : ∃ x ∈ Set.Icc 2 3, f a b x = 4) 
  (h_min_achieved : ∃ x ∈ Set.Icc 2 3, f a b x = 1) : 
  a = 1 ∧ b = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_coefficients_l4015_401569


namespace NUMINAMATH_CALUDE_unique_perpendicular_line_l4015_401544

-- Define the concept of a plane in 3D space
class Plane :=
  (normal : ℝ → ℝ → ℝ → ℝ)

-- Define the concept of a line in 3D space
class Line :=
  (direction : ℝ → ℝ → ℝ)

-- Define a point in 3D space
def Point := ℝ × ℝ × ℝ

-- Define what it means for a point to be outside a plane
def PointOutsidePlane (p : Point) (plane : Plane) : Prop := sorry

-- Define what it means for a line to pass through a point
def LinePassesThroughPoint (l : Line) (p : Point) : Prop := sorry

-- Define perpendicularity between a line and a plane
def LinePerpendicular (l : Line) (plane : Plane) : Prop := sorry

-- State the theorem
theorem unique_perpendicular_line 
  (plane : Plane) (p : Point) (h : PointOutsidePlane p plane) :
  ∃! l : Line, LinePassesThroughPoint l p ∧ LinePerpendicular l plane :=
sorry

end NUMINAMATH_CALUDE_unique_perpendicular_line_l4015_401544


namespace NUMINAMATH_CALUDE_function_value_at_m_l4015_401505

/-- Given a function f(x) = x³ + ax + 3 where f(-m) = 1, prove that f(m) = 5 -/
theorem function_value_at_m (a m : ℝ) : 
  (fun x : ℝ ↦ x^3 + a*x + 3) (-m) = 1 → 
  (fun x : ℝ ↦ x^3 + a*x + 3) m = 5 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_m_l4015_401505


namespace NUMINAMATH_CALUDE_money_left_over_l4015_401588

-- Define the given conditions
def video_game_cost : ℝ := 60
def discount_rate : ℝ := 0.15
def candy_cost : ℝ := 5
def sales_tax_rate : ℝ := 0.10
def shipping_fee : ℝ := 3
def babysitting_rate : ℝ := 8
def hours_worked : ℝ := 9

-- Define the theorem
theorem money_left_over :
  let discounted_price := video_game_cost * (1 - discount_rate)
  let video_game_total := discounted_price + shipping_fee
  let video_game_with_tax := video_game_total * (1 + sales_tax_rate)
  let candy_with_tax := candy_cost * (1 + sales_tax_rate)
  let total_cost := video_game_with_tax + candy_with_tax
  let earnings := babysitting_rate * hours_worked
  earnings - total_cost = 7.10 := by
  sorry

end NUMINAMATH_CALUDE_money_left_over_l4015_401588


namespace NUMINAMATH_CALUDE_king_ducats_distribution_l4015_401519

theorem king_ducats_distribution (n : ℕ) (total_ducats : ℕ) :
  (∃ (a : ℕ),
    -- The eldest son receives 'a' ducats in the first round
    a + n = 21 ∧
    -- Total ducats in the first round
    n * a - (n - 1) * n / 2 +
    -- Total ducats in the second round
    n * (n + 1) / 2 = total_ducats) →
  n = 7 ∧ total_ducats = 105 := by
sorry

end NUMINAMATH_CALUDE_king_ducats_distribution_l4015_401519


namespace NUMINAMATH_CALUDE_rational_roots_imply_rational_roots_l4015_401550

theorem rational_roots_imply_rational_roots (c : ℝ) (p q : ℚ) :
  (p^2 - p + c = 0) → (q^2 - q + c = 0) →
  ∃ (r s : ℚ), r^2 + p*r - q = 0 ∧ s^2 + p*s - q = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_roots_imply_rational_roots_l4015_401550


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l4015_401585

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_equation (given_line : Line) (point : Point) :
  given_line.a = 1 ∧ given_line.b = -2 ∧ given_line.c = 1 ∧
  point.x = 2 ∧ point.y = -1 →
  ∃ (l : Line), l.perpendicular given_line ∧ point.liesOn l ∧
  l.a = 2 ∧ l.b = 1 ∧ l.c = -3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l4015_401585


namespace NUMINAMATH_CALUDE_unique_zero_point_condition_l4015_401590

/-- The function f(x) = ax³ - 3x² + 2 has only one zero point if and only if a ∈ (-∞, -√2) ∪ (√2, +∞) -/
theorem unique_zero_point_condition (a : ℝ) :
  (∃! x, a * x^3 - 3 * x^2 + 2 = 0) ↔ a < -Real.sqrt 2 ∨ a > Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_zero_point_condition_l4015_401590
