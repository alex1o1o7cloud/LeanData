import Mathlib

namespace NUMINAMATH_CALUDE_springfield_population_difference_l2064_206411

/-- The population difference between two cities given the population of one city and their total population -/
def population_difference (population_springfield : ℕ) (total_population : ℕ) : ℕ :=
  population_springfield - (total_population - population_springfield)

/-- Theorem stating that the population difference between Springfield and the other city is 119,666 -/
theorem springfield_population_difference :
  population_difference 482653 845640 = 119666 := by
  sorry

end NUMINAMATH_CALUDE_springfield_population_difference_l2064_206411


namespace NUMINAMATH_CALUDE_sin_cube_identity_l2064_206401

theorem sin_cube_identity (θ : Real) : 
  Real.sin θ ^ 3 = (-1/4) * Real.sin (3 * θ) + (3/4) * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_sin_cube_identity_l2064_206401


namespace NUMINAMATH_CALUDE_function_value_proof_l2064_206447

/-- Given a function f(x) = ax^5 + bx^3 + cx + 8, prove that if f(-2) = 10, then f(2) = 6 -/
theorem function_value_proof (a b c : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^5 + b * x^3 + c * x + 8
  f (-2) = 10 → f 2 = 6 := by
sorry

end NUMINAMATH_CALUDE_function_value_proof_l2064_206447


namespace NUMINAMATH_CALUDE_polyhedron_exists_l2064_206478

/-- A vertex of the polyhedron -/
inductive Vertex : Type
| A | B | C | D | E | F | G | H

/-- An edge of the polyhedron -/
inductive Edge : Type
| AB | AC | AH | BC | BD | CD | DE | EF | EG | FG | FH | GH

/-- A polyhedron structure -/
structure Polyhedron :=
  (vertices : List Vertex)
  (edges : List Edge)

/-- The specific polyhedron we're interested in -/
def specificPolyhedron : Polyhedron :=
  { vertices := [Vertex.A, Vertex.B, Vertex.C, Vertex.D, Vertex.E, Vertex.F, Vertex.G, Vertex.H],
    edges := [Edge.AB, Edge.AC, Edge.AH, Edge.BC, Edge.BD, Edge.CD, Edge.DE, Edge.EF, Edge.EG, Edge.FG, Edge.FH, Edge.GH] }

/-- Theorem stating the existence of the polyhedron -/
theorem polyhedron_exists : ∃ (p : Polyhedron), p = specificPolyhedron :=
sorry

end NUMINAMATH_CALUDE_polyhedron_exists_l2064_206478


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2064_206432

def M : Set ℕ := {0, 1}
def N : Set ℕ := {1, 2}

theorem union_of_M_and_N : M ∪ N = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2064_206432


namespace NUMINAMATH_CALUDE_money_found_at_mall_l2064_206446

def prove_money_found (initial_amount : ℚ) (mother_gave : ℚ) (toy_cost : ℚ) (money_left : ℚ) : ℚ :=
  (toy_cost + money_left) - (initial_amount + mother_gave)

theorem money_found_at_mall :
  let initial_amount := 0.85
  let mother_gave := 0.40
  let toy_cost := 1.60
  let money_left := 0.15
  prove_money_found initial_amount mother_gave toy_cost money_left = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_money_found_at_mall_l2064_206446


namespace NUMINAMATH_CALUDE_sum_first_150_remainder_l2064_206440

theorem sum_first_150_remainder (n : Nat) (h : n = 150) :
  (List.range n).sum % 8000 = 3325 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_150_remainder_l2064_206440


namespace NUMINAMATH_CALUDE_tennis_tournament_rounds_l2064_206461

theorem tennis_tournament_rounds :
  ∀ (rounds : ℕ)
    (games_per_round : List ℕ)
    (cans_per_game : ℕ)
    (balls_per_can : ℕ)
    (total_balls : ℕ),
  games_per_round = [8, 4, 2, 1] →
  cans_per_game = 5 →
  balls_per_can = 3 →
  total_balls = 225 →
  (List.sum games_per_round * cans_per_game * balls_per_can = total_balls) →
  rounds = 4 := by
sorry

end NUMINAMATH_CALUDE_tennis_tournament_rounds_l2064_206461


namespace NUMINAMATH_CALUDE_min_cost_theorem_min_cost_value_l2064_206423

def volleyball_price : ℕ := 50
def basketball_price : ℕ := 80

def total_balls : ℕ := 60
def max_cost : ℕ := 3800
def max_volleyballs : ℕ := 38

def cost_function (m : ℕ) : ℕ := volleyball_price * m + basketball_price * (total_balls - m)

theorem min_cost_theorem (m : ℕ) (h1 : m ≤ max_volleyballs) (h2 : cost_function m ≤ max_cost) :
  cost_function max_volleyballs ≤ cost_function m :=
sorry

theorem min_cost_value : cost_function max_volleyballs = 3660 :=
sorry

end NUMINAMATH_CALUDE_min_cost_theorem_min_cost_value_l2064_206423


namespace NUMINAMATH_CALUDE_loan_interest_rate_l2064_206487

theorem loan_interest_rate (principal time_period rate interest : ℝ) : 
  principal = 800 →
  time_period = rate →
  interest = 632 →
  interest = (principal * rate * time_period) / 100 →
  rate = Real.sqrt 79 :=
by sorry

end NUMINAMATH_CALUDE_loan_interest_rate_l2064_206487


namespace NUMINAMATH_CALUDE_prism_path_lengths_l2064_206474

/-- Regular triangular prism with given properties -/
structure RegularTriangularPrism where
  -- Base edge length
  ab : ℝ
  -- Height
  aa1 : ℝ
  -- Point on base edge BC
  p : ℝ × ℝ × ℝ
  -- Shortest path length from P to M
  shortest_path : ℝ

/-- Theorem stating the lengths of PC and NC in the given prism -/
theorem prism_path_lengths (prism : RegularTriangularPrism)
  (h_ab : prism.ab = 3)
  (h_aa1 : prism.aa1 = 4)
  (h_path : prism.shortest_path = Real.sqrt 29) :
  ∃ (pc nc : ℝ), pc = 2 ∧ nc = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_prism_path_lengths_l2064_206474


namespace NUMINAMATH_CALUDE_factorization_m_squared_minus_3m_l2064_206467

theorem factorization_m_squared_minus_3m (m : ℝ) : m^2 - 3*m = m*(m-3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_m_squared_minus_3m_l2064_206467


namespace NUMINAMATH_CALUDE_tower_height_difference_l2064_206489

theorem tower_height_difference (grace_height clyde_height : ℕ) :
  grace_height = 40 ∧ grace_height = 8 * clyde_height →
  grace_height - clyde_height = 35 := by
  sorry

end NUMINAMATH_CALUDE_tower_height_difference_l2064_206489


namespace NUMINAMATH_CALUDE_ivanov_net_worth_calculation_l2064_206449

/-- The net worth of the Ivanov family -/
def ivanov_net_worth : ℕ := by sorry

/-- The value of the Ivanov family's apartment in rubles -/
def apartment_value : ℕ := 3000000

/-- The value of the Ivanov family's car in rubles -/
def car_value : ℕ := 900000

/-- The amount in the Ivanov family's bank deposit in rubles -/
def bank_deposit : ℕ := 300000

/-- The value of the Ivanov family's securities in rubles -/
def securities_value : ℕ := 200000

/-- The amount of liquid cash the Ivanov family has in rubles -/
def liquid_cash : ℕ := 100000

/-- The Ivanov family's mortgage balance in rubles -/
def mortgage_balance : ℕ := 1500000

/-- The Ivanov family's car loan balance in rubles -/
def car_loan_balance : ℕ := 500000

/-- The Ivanov family's debt to relatives in rubles -/
def debt_to_relatives : ℕ := 200000

/-- Theorem stating that the Ivanov family's net worth is 2,300,000 rubles -/
theorem ivanov_net_worth_calculation :
  ivanov_net_worth = 
    (apartment_value + car_value + bank_deposit + securities_value + liquid_cash) -
    (mortgage_balance + car_loan_balance + debt_to_relatives) :=
by sorry

end NUMINAMATH_CALUDE_ivanov_net_worth_calculation_l2064_206449


namespace NUMINAMATH_CALUDE_trig_sum_equals_two_l2064_206495

theorem trig_sum_equals_two :
  Real.cos (0 : ℝ) ^ 4 +
  Real.cos (Real.pi / 2) ^ 4 +
  Real.sin (Real.pi / 4) ^ 4 +
  Real.sin (3 * Real.pi / 4) ^ 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_equals_two_l2064_206495


namespace NUMINAMATH_CALUDE_least_k_equals_2_pow_q_l2064_206499

/-- Represents a polynomial with integer coefficients -/
def IntPolynomial := ℕ → ℤ

/-- Given an even positive integer n, this function returns the least k₀ such that
    k₀ = f(x) · (x+1)^n + g(x) · (x^n + 1) for some polynomials f(x) and g(x) with integer coefficients -/
noncomputable def least_k (n : ℕ) : ℕ :=
  sorry

theorem least_k_equals_2_pow_q (n : ℕ) (q r : ℕ) (hn : Even n) (hq : Odd q) (hnqr : n = q * 2^r) :
  least_k n = 2^q :=
by sorry

end NUMINAMATH_CALUDE_least_k_equals_2_pow_q_l2064_206499


namespace NUMINAMATH_CALUDE_stratified_sampling_size_l2064_206451

/-- Represents the number of units produced by each workshop -/
structure WorkshopProduction where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the sampling information -/
structure SamplingInfo where
  total_sample : ℕ
  workshop_b_sample : ℕ

/-- Theorem stating the correct sample size for the given scenario -/
theorem stratified_sampling_size 
  (prod : WorkshopProduction)
  (sample : SamplingInfo)
  (h1 : prod.a = 96)
  (h2 : prod.b = 84)
  (h3 : prod.c = 60)
  (h4 : sample.workshop_b_sample = 7)
  (h5 : sample.workshop_b_sample / sample.total_sample = prod.b / (prod.a + prod.b + prod.c)) :
  sample.total_sample = 70 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_size_l2064_206451


namespace NUMINAMATH_CALUDE_smallest_number_with_properties_l2064_206454

def ends_with_six (n : ℕ) : Prop :=
  n % 10 = 6

def move_six_to_front (n : ℕ) : ℕ :=
  let k := (Nat.log 10 n).succ
  6 * 10^k + (n - 6) / 10

theorem smallest_number_with_properties :
  ∃ (N : ℕ), N = 153846 ∧
  ends_with_six N ∧
  move_six_to_front N = 4 * N ∧
  ∀ (m : ℕ), m < N →
    ¬(ends_with_six m ∧ move_six_to_front m = 4 * m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_properties_l2064_206454


namespace NUMINAMATH_CALUDE_least_squares_for_25x25_l2064_206482

theorem least_squares_for_25x25 (n : Nat) (h1 : n = 25) (h2 : n * n = 625) :
  ∃ f : Nat → Nat, f n ≥ (n^2 - 1) / 2 ∧ f n ≥ 312 := by
  sorry

end NUMINAMATH_CALUDE_least_squares_for_25x25_l2064_206482


namespace NUMINAMATH_CALUDE_poly_simplification_poly_evaluation_l2064_206427

-- Define the original polynomial
def original_poly (x : ℝ) : ℝ :=
  (2*x^5 - 3*x^4 + 5*x^3 - 9*x^2 + 8*x - 15) + (5*x^4 - 2*x^3 + 3*x^2 - 4*x + 9)

-- Define the simplified polynomial
def simplified_poly (x : ℝ) : ℝ :=
  2*x^5 + 2*x^4 + 3*x^3 - 6*x^2 + 4*x - 6

-- Theorem stating that the original polynomial equals the simplified polynomial
theorem poly_simplification (x : ℝ) : original_poly x = simplified_poly x := by
  sorry

-- Theorem stating that the simplified polynomial evaluated at x = 2 equals 98
theorem poly_evaluation : simplified_poly 2 = 98 := by
  sorry

end NUMINAMATH_CALUDE_poly_simplification_poly_evaluation_l2064_206427


namespace NUMINAMATH_CALUDE_simplify_expression_l2064_206494

theorem simplify_expression (x : ℝ) : 120 * x - 75 * x = 45 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2064_206494


namespace NUMINAMATH_CALUDE_tent_production_equation_correct_l2064_206430

/-- Represents the tent production scenario -/
structure TentProduction where
  original_plan : ℕ
  increase_percentage : ℚ
  days_ahead : ℕ
  daily_increase : ℕ

/-- The equation representing the tent production scenario -/
def production_equation (tp : TentProduction) (x : ℚ) : Prop :=
  (tp.original_plan : ℚ) / (x - tp.daily_increase) - 
  (tp.original_plan * (1 + tp.increase_percentage)) / x = tp.days_ahead

/-- Theorem stating that the equation correctly represents the given conditions -/
theorem tent_production_equation_correct (tp : TentProduction) (x : ℚ) 
  (h1 : tp.original_plan = 7200)
  (h2 : tp.increase_percentage = 1/5)
  (h3 : tp.days_ahead = 4)
  (h4 : tp.daily_increase = 720)
  (h5 : x > tp.daily_increase) :
  production_equation tp x := by
  sorry

end NUMINAMATH_CALUDE_tent_production_equation_correct_l2064_206430


namespace NUMINAMATH_CALUDE_find_number_l2064_206418

theorem find_number : ∃ x : ℕ, x * 99999 = 65818408915 ∧ x = 658185 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2064_206418


namespace NUMINAMATH_CALUDE_nested_G_evaluation_l2064_206498

def G (x : ℝ) : ℝ := (x - 2)^2 - 1

theorem nested_G_evaluation : G (G (G (G (G 2)))) = 1179395 := by
  sorry

end NUMINAMATH_CALUDE_nested_G_evaluation_l2064_206498


namespace NUMINAMATH_CALUDE_largest_two_digit_power_ending_l2064_206409

/-- A number is a two-digit number if it's between 10 and 99, inclusive. -/
def IsTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A number satisfies the power condition if all its positive integer powers end with itself modulo 100. -/
def SatisfiesPowerCondition (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 0 → n^k % 100 = n % 100

/-- 76 is the largest two-digit number divisible by 4 that satisfies the power condition. -/
theorem largest_two_digit_power_ending : 
  IsTwoDigit 76 ∧ 
  76 % 4 = 0 ∧ 
  SatisfiesPowerCondition 76 ∧ 
  ∀ n : ℕ, IsTwoDigit n → n % 4 = 0 → SatisfiesPowerCondition n → n ≤ 76 :=
sorry

end NUMINAMATH_CALUDE_largest_two_digit_power_ending_l2064_206409


namespace NUMINAMATH_CALUDE_regular_polygon_with_160_degree_angles_has_18_sides_l2064_206471

/-- A regular polygon with interior angles measuring 160° has 18 sides. -/
theorem regular_polygon_with_160_degree_angles_has_18_sides :
  ∀ n : ℕ, n ≥ 3 →
  (∀ θ : ℝ, θ = 160 → (n : ℝ) * θ = (n - 2 : ℝ) * 180) →
  n = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_160_degree_angles_has_18_sides_l2064_206471


namespace NUMINAMATH_CALUDE_sqrt_six_div_sqrt_two_eq_sqrt_three_l2064_206410

theorem sqrt_six_div_sqrt_two_eq_sqrt_three :
  Real.sqrt 6 / Real.sqrt 2 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_div_sqrt_two_eq_sqrt_three_l2064_206410


namespace NUMINAMATH_CALUDE_problem_solution_l2064_206453

theorem problem_solution (t : ℝ) :
  let x := 3 - 2*t
  let y := t^2 + 3*t + 6
  x = -1 → y = 16 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2064_206453


namespace NUMINAMATH_CALUDE_medicine_price_reduction_l2064_206444

theorem medicine_price_reduction (x : ℝ) : 
  (25 : ℝ) * (1 - x)^2 = 16 ↔ 
  (∃ (price_after_first_reduction : ℝ),
    price_after_first_reduction = 25 * (1 - x) ∧
    16 = price_after_first_reduction * (1 - x)) :=
by sorry

end NUMINAMATH_CALUDE_medicine_price_reduction_l2064_206444


namespace NUMINAMATH_CALUDE_weight_division_l2064_206490

theorem weight_division (n : ℕ) : 
  (∃ (a b c : ℕ), a + b + c = n * (n + 1) / 2 ∧ a = b ∧ b = c) ↔ 
  (n > 3 ∧ (n % 3 = 0 ∨ n % 3 = 2)) :=
by sorry

end NUMINAMATH_CALUDE_weight_division_l2064_206490


namespace NUMINAMATH_CALUDE_alex_has_48_shells_l2064_206419

/-- The number of seashells in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of seashells Mimi picked up -/
def mimi_dozens : ℕ := 2

/-- The number of seashells Mimi picked up -/
def mimi_shells : ℕ := mimi_dozens * dozen

/-- The number of seashells Kyle found -/
def kyle_shells : ℕ := 2 * mimi_shells

/-- The number of seashells Leigh grabbed -/
def leigh_shells : ℕ := kyle_shells / 3

/-- The number of seashells Alex unearthed -/
def alex_shells : ℕ := 3 * leigh_shells

/-- Theorem stating that Alex had 48 seashells -/
theorem alex_has_48_shells : alex_shells = 48 := by
  sorry

end NUMINAMATH_CALUDE_alex_has_48_shells_l2064_206419


namespace NUMINAMATH_CALUDE_max_value_expression_l2064_206457

theorem max_value_expression (a b : ℝ) (ha : a ≥ 1) (hb : b ≥ 1) :
  (|7*a + 8*b - a*b| + |2*a + 8*b - 6*a*b|) / (a * Real.sqrt (1 + b^2)) ≤ 9 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l2064_206457


namespace NUMINAMATH_CALUDE_absolute_value_of_T_l2064_206438

def i : ℂ := Complex.I

def T : ℂ := (1 + i)^18 + (1 - i)^18

theorem absolute_value_of_T : Complex.abs T = 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_T_l2064_206438


namespace NUMINAMATH_CALUDE_log_equation_solution_l2064_206458

theorem log_equation_solution :
  ∃ y : ℝ, (2 * Real.log y + 3 * Real.log 2 = 1) ∧ (y = Real.sqrt 5 / 2) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2064_206458


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2064_206425

def U : Set Nat := {0, 2, 4, 6, 8, 10}
def A : Set Nat := {2, 4, 6}
def B : Set Nat := {1}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 1, 8, 10} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2064_206425


namespace NUMINAMATH_CALUDE_opposite_of_2023_l2064_206465

theorem opposite_of_2023 : 
  ∀ x : ℤ, x + 2023 = 0 ↔ x = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l2064_206465


namespace NUMINAMATH_CALUDE_solution_set_part1_a_range_part2_l2064_206431

-- Define the function f
def f (x a : ℝ) : ℝ := x^2 + |2*x - 4| + a

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | (f x (-3)) > x^2 + |x|} = {x : ℝ | x < 1/3 ∨ x > 7} := by sorry

-- Part 2
theorem a_range_part2 :
  (∀ x : ℝ, f x a ≥ 0) → a ≥ -3 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_a_range_part2_l2064_206431


namespace NUMINAMATH_CALUDE_exponent_division_l2064_206437

theorem exponent_division (a : ℝ) : a^6 / a^4 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2064_206437


namespace NUMINAMATH_CALUDE_freshman_count_proof_l2064_206417

theorem freshman_count_proof :
  ∃! n : ℕ, n < 600 ∧ n % 25 = 24 ∧ n % 19 = 10 ∧ n = 574 := by
  sorry

end NUMINAMATH_CALUDE_freshman_count_proof_l2064_206417


namespace NUMINAMATH_CALUDE_multiply_add_theorem_l2064_206442

theorem multiply_add_theorem : 15 * 30 + 45 * 15 = 1125 := by
  sorry

end NUMINAMATH_CALUDE_multiply_add_theorem_l2064_206442


namespace NUMINAMATH_CALUDE_problem_statement_l2064_206415

theorem problem_statement : 2 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 1600 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2064_206415


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2064_206492

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ∈ Set.Ici (0 : ℝ) → x^3 + 2*x ≥ 0) ↔
  (∃ x : ℝ, x ∈ Set.Ici (0 : ℝ) ∧ x^3 + 2*x < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2064_206492


namespace NUMINAMATH_CALUDE_gloria_has_23_maple_trees_l2064_206428

/-- Represents the problem of calculating Gloria's maple trees --/
def GloriasMapleTrees (cabin_price cash_on_hand leftover cypress_count pine_count cypress_price pine_price maple_price : ℕ) : Prop :=
  let total_needed := cabin_price - cash_on_hand
  let cypress_income := cypress_count * cypress_price
  let pine_income := pine_count * pine_price
  let maple_income := total_needed - cypress_income - pine_income
  ∃ (maple_count : ℕ), 
    maple_count * maple_price = maple_income ∧
    maple_count * maple_price + cypress_income + pine_income + cash_on_hand = cabin_price + leftover

theorem gloria_has_23_maple_trees : 
  GloriasMapleTrees 129000 150 350 20 600 100 200 300 → 
  ∃ (maple_count : ℕ), maple_count = 23 :=
sorry

end NUMINAMATH_CALUDE_gloria_has_23_maple_trees_l2064_206428


namespace NUMINAMATH_CALUDE_f_monotonicity_and_extreme_value_l2064_206450

noncomputable section

def f (x : ℝ) := Real.log x - x

theorem f_monotonicity_and_extreme_value :
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₁ < f x₂) ∧
  (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ x, 0 < x → f x ≤ f 1) ∧
  f 1 = -1 :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_extreme_value_l2064_206450


namespace NUMINAMATH_CALUDE_four_number_sequence_l2064_206479

theorem four_number_sequence (a b c d : ℝ) 
  (h1 : b^2 = a*c)
  (h2 : a*b*c = 216)
  (h3 : 2*c = b + d)
  (h4 : b + c + d = 12) :
  a = 9 ∧ b = 6 ∧ c = 4 ∧ d = 2 := by
sorry

end NUMINAMATH_CALUDE_four_number_sequence_l2064_206479


namespace NUMINAMATH_CALUDE_min_modulus_m_is_two_l2064_206403

/-- Given a quadratic equation with complex coefficients that has a real root,
    prove that the minimum value of the modulus of m is 2. -/
theorem min_modulus_m_is_two (m : ℂ) :
  (∃ x : ℝ, (1 + 2*I : ℂ) * x^2 + m * x + (1 - 2*I : ℂ) = 0) →
  (∀ m' : ℂ, (∃ x : ℝ, (1 + 2*I : ℂ) * x^2 + m' * x + (1 - 2*I : ℂ) = 0) →
    Complex.abs m' ≥ 2) ∧
  (∃ m₀ : ℂ, (∃ x : ℝ, (1 + 2*I : ℂ) * x^2 + m₀ * x + (1 - 2*I : ℂ) = 0) ∧
    Complex.abs m₀ = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_modulus_m_is_two_l2064_206403


namespace NUMINAMATH_CALUDE_kevins_phone_repair_l2064_206472

/-- Given the initial conditions of Kevin's phone repair scenario, 
    prove that the number of phones each person needs to repair is 9. -/
theorem kevins_phone_repair 
  (initial_phones : ℕ) 
  (repaired_phones : ℕ) 
  (new_phones : ℕ) 
  (h1 : initial_phones = 15)
  (h2 : repaired_phones = 3)
  (h3 : new_phones = 6) :
  (initial_phones - repaired_phones + new_phones) / 2 = 9 := by
sorry

end NUMINAMATH_CALUDE_kevins_phone_repair_l2064_206472


namespace NUMINAMATH_CALUDE_apple_harvest_l2064_206466

/-- Proves that the initial number of apples is 569 given the harvesting conditions -/
theorem apple_harvest (new_apples : ℕ) (rotten_apples : ℕ) (current_apples : ℕ)
  (h1 : new_apples = 419)
  (h2 : rotten_apples = 263)
  (h3 : current_apples = 725) :
  current_apples + rotten_apples - new_apples = 569 := by
  sorry

#check apple_harvest

end NUMINAMATH_CALUDE_apple_harvest_l2064_206466


namespace NUMINAMATH_CALUDE_tangent_line_at_zero_l2064_206486

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.exp x + 2

theorem tangent_line_at_zero (x : ℝ) :
  ∃ (m b : ℝ), (∀ h : ℝ, f h = m * h + b) ∧ m = 2 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_zero_l2064_206486


namespace NUMINAMATH_CALUDE_A_subset_B_l2064_206452

variable {X : Type*} -- Domain of functions f and g
variable (f g : X → ℝ) -- Real-valued functions f and g
variable (a : ℝ) -- Real number a

def A (f g : X → ℝ) (a : ℝ) : Set X :=
  {x : X | |f x| + |g x| < a}

def B (f g : X → ℝ) (a : ℝ) : Set X :=
  {x : X | |f x + g x| < a}

theorem A_subset_B (h : a > 0) : A f g a ⊆ B f g a := by
  sorry

end NUMINAMATH_CALUDE_A_subset_B_l2064_206452


namespace NUMINAMATH_CALUDE_vacation_cost_l2064_206475

theorem vacation_cost (C : ℝ) : 
  (C / 3 - C / 4 = 60) → C = 720 := by
sorry

end NUMINAMATH_CALUDE_vacation_cost_l2064_206475


namespace NUMINAMATH_CALUDE_tammy_earnings_l2064_206483

/-- Calculates Tammy's earnings from selling oranges over a period of time. -/
def orange_earnings (num_trees : ℕ) (oranges_per_tree : ℕ) (oranges_per_pack : ℕ) 
  (price_per_pack : ℕ) (num_days : ℕ) : ℕ :=
  let oranges_per_day := num_trees * oranges_per_tree
  let packs_per_day := oranges_per_day / oranges_per_pack
  let total_packs := packs_per_day * num_days
  total_packs * price_per_pack

/-- Proves that Tammy's earnings after 3 weeks equal $840. -/
theorem tammy_earnings : 
  orange_earnings 10 12 6 2 (3 * 7) = 840 := by
  sorry

end NUMINAMATH_CALUDE_tammy_earnings_l2064_206483


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l2064_206480

/-- Given two cubes with edge lengths a and b, where a/b = 3/1 and the volume of the cube
    with edge length a is 27 units, prove that the volume of the cube with edge length b is 1 unit. -/
theorem cube_volume_ratio (a b : ℝ) (h1 : a / b = 3 / 1) (h2 : a^3 = 27) : b^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l2064_206480


namespace NUMINAMATH_CALUDE_advantages_of_early_license_l2064_206468

-- Define the type for advantages
inductive Advantage
  | CostSavings
  | RentalFlexibility
  | EmploymentOpportunities

-- Define a function to check if an advantage applies to getting a license at 18
def is_advantage_at_18 (a : Advantage) : Prop :=
  match a with
  | Advantage.CostSavings => true
  | Advantage.RentalFlexibility => true
  | Advantage.EmploymentOpportunities => true

-- Define a function to check if an advantage applies to getting a license at 30
def is_advantage_at_30 (a : Advantage) : Prop :=
  match a with
  | Advantage.CostSavings => false
  | Advantage.RentalFlexibility => false
  | Advantage.EmploymentOpportunities => false

-- Theorem stating that there are at least three distinct advantages
-- of getting a license at 18 compared to 30
theorem advantages_of_early_license :
  ∃ (a b c : Advantage), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  is_advantage_at_18 a ∧ is_advantage_at_18 b ∧ is_advantage_at_18 c ∧
  ¬is_advantage_at_30 a ∧ ¬is_advantage_at_30 b ∧ ¬is_advantage_at_30 c :=
sorry

end NUMINAMATH_CALUDE_advantages_of_early_license_l2064_206468


namespace NUMINAMATH_CALUDE_subset_difference_theorem_l2064_206473

theorem subset_difference_theorem (n k m : ℕ) (A : Finset ℕ) 
  (h1 : k ≥ 2)
  (h2 : n ≤ m)
  (h3 : m < ((2 * k - 1) * n) / k)
  (h4 : A.card = n)
  (h5 : ∀ a ∈ A, a ≤ m) :
  ∀ x : ℤ, 0 < x ∧ x < n / (k - 1) → 
    ∃ a a' : ℕ, a ∈ A ∧ a' ∈ A ∧ (a : ℤ) - (a' : ℤ) = x :=
by sorry

end NUMINAMATH_CALUDE_subset_difference_theorem_l2064_206473


namespace NUMINAMATH_CALUDE_rectangle_diagonals_equal_diagonals_equal_not_always_rectangle_not_rectangle_diagonals_not_equal_not_always_diagonals_not_equal_not_rectangle_l2064_206462

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define what it means for a quadrilateral to be a rectangle
def is_rectangle (q : Quadrilateral) : Prop :=
  sorry

-- Define what it means for diagonals to be equal
def diagonals_equal (q : Quadrilateral) : Prop :=
  sorry

-- Theorem statements
theorem rectangle_diagonals_equal (q : Quadrilateral) :
  is_rectangle q → diagonals_equal q :=
sorry

theorem diagonals_equal_not_always_rectangle :
  ∃ q : Quadrilateral, diagonals_equal q ∧ ¬is_rectangle q :=
sorry

theorem not_rectangle_diagonals_not_equal_not_always :
  ∃ q : Quadrilateral, ¬is_rectangle q ∧ diagonals_equal q :=
sorry

theorem diagonals_not_equal_not_rectangle (q : Quadrilateral) :
  ¬diagonals_equal q → ¬is_rectangle q :=
sorry

end NUMINAMATH_CALUDE_rectangle_diagonals_equal_diagonals_equal_not_always_rectangle_not_rectangle_diagonals_not_equal_not_always_diagonals_not_equal_not_rectangle_l2064_206462


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l2064_206429

theorem negation_of_proposition (p : ℝ → Prop) : 
  (¬∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬p x) :=
by sorry

theorem negation_of_specific_proposition : 
  (¬∀ x : ℝ, x^2 - 2*x > 0) ↔ (∃ x : ℝ, x^2 - 2*x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l2064_206429


namespace NUMINAMATH_CALUDE_trees_survived_vs_died_l2064_206413

theorem trees_survived_vs_died (initial_trees dead_trees : ℕ) : 
  initial_trees = 11 → dead_trees = 2 → 
  (initial_trees - dead_trees) - dead_trees = 7 := by
  sorry

end NUMINAMATH_CALUDE_trees_survived_vs_died_l2064_206413


namespace NUMINAMATH_CALUDE_total_cost_of_seeds_bottles_not_enough_l2064_206469

-- Define the given values
def seed_price : ℝ := 9.48
def seed_amount : ℝ := 3.3
def bottle_capacity : ℝ := 0.35
def num_bottles : ℕ := 9

-- Theorem for the total cost of grass seeds
theorem total_cost_of_seeds : seed_price * seed_amount = 31.284 := by sorry

-- Theorem for the insufficiency of 9 bottles
theorem bottles_not_enough : seed_amount > (bottle_capacity * num_bottles) := by sorry

end NUMINAMATH_CALUDE_total_cost_of_seeds_bottles_not_enough_l2064_206469


namespace NUMINAMATH_CALUDE_team_games_theorem_l2064_206422

theorem team_games_theorem (first_games : Nat) (win_rate_first : Real) 
  (win_rate_remaining : Real) (total_win_rate : Real) :
  first_games = 30 →
  win_rate_first = 0.4 →
  win_rate_remaining = 0.8 →
  total_win_rate = 0.6 →
  ∃ (total_games : Nat),
    total_games = 60 ∧
    (first_games : Real) * win_rate_first + 
    (total_games - first_games : Real) * win_rate_remaining = 
    (total_games : Real) * total_win_rate :=
by sorry

#check team_games_theorem

end NUMINAMATH_CALUDE_team_games_theorem_l2064_206422


namespace NUMINAMATH_CALUDE_fraction_simplification_l2064_206407

theorem fraction_simplification (a b : ℝ) (h : a ≠ 0) :
  (a^2 + 2*a*b + b^2) / (a^2 + a*b) = (a + b) / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2064_206407


namespace NUMINAMATH_CALUDE_range_of_a_l2064_206459

theorem range_of_a (a b c : ℝ) 
  (eq1 : a^2 - b*c - 8*a + 7 = 0)
  (eq2 : b^2 + c^2 + b*c - 6*a + 6 = 0) :
  1 ≤ a ∧ a ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2064_206459


namespace NUMINAMATH_CALUDE_train_meet_time_l2064_206464

/-- The time (in hours after midnight) when the trains meet -/
def meet_time : ℝ := 11

/-- The time (in hours after midnight) when the train from B starts -/
def start_time_B : ℝ := 8

/-- The distance between stations A and B in kilometers -/
def distance : ℝ := 155

/-- The speed of the train from A in km/h -/
def speed_A : ℝ := 20

/-- The speed of the train from B in km/h -/
def speed_B : ℝ := 25

/-- The time (in hours after midnight) when the train from A starts -/
def start_time_A : ℝ := 7

theorem train_meet_time :
  start_time_A = meet_time - (distance - speed_B * (meet_time - start_time_B)) / speed_A :=
by sorry

end NUMINAMATH_CALUDE_train_meet_time_l2064_206464


namespace NUMINAMATH_CALUDE_problem_statement_l2064_206477

theorem problem_statement (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -15)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 20) :
  b / (a + b) + c / (b + c) + a / (c + a) = 19 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2064_206477


namespace NUMINAMATH_CALUDE_min_variance_product_l2064_206455

theorem min_variance_product (a b : ℝ) : 
  2 ≤ 3 ∧ 3 ≤ 3 ∧ 3 ≤ 7 ∧ 7 ≤ a ∧ a ≤ b ∧ b ≤ 12 ∧ 12 ≤ 13.7 ∧ 13.7 ≤ 18.3 ∧ 18.3 ≤ 21 →
  (2 + 3 + 3 + 7 + a + b + 12 + 13.7 + 18.3 + 21) / 10 = 10 →
  a + b = 20 →
  (∀ x y : ℝ, x + y = 20 → (x - 10)^2 + (y - 10)^2 ≥ (a - 10)^2 + (b - 10)^2) →
  a * b = 100 :=
by sorry


end NUMINAMATH_CALUDE_min_variance_product_l2064_206455


namespace NUMINAMATH_CALUDE_greatest_multiple_24_unique_digits_remainder_l2064_206434

/-- 
M is the greatest integer multiple of 24 with no two digits being the same.
-/
def M : ℕ := sorry

/-- 
A function that checks if a natural number has all unique digits.
-/
def has_unique_digits (n : ℕ) : Prop := sorry

theorem greatest_multiple_24_unique_digits_remainder (h1 : M % 24 = 0) 
  (h2 : has_unique_digits M) 
  (h3 : ∀ k : ℕ, k > M → k % 24 = 0 → ¬(has_unique_digits k)) : 
  M % 1000 = 720 := by sorry

end NUMINAMATH_CALUDE_greatest_multiple_24_unique_digits_remainder_l2064_206434


namespace NUMINAMATH_CALUDE_min_abs_phi_l2064_206456

/-- Given a function y = 2sin(2x - φ) whose graph is symmetric about the point (4π/3, 0),
    the minimum value of |φ| is π/3 -/
theorem min_abs_phi (φ : ℝ) : 
  (∀ x : ℝ, 2 * Real.sin (2 * x - φ) = 2 * Real.sin (2 * (8 * π / 3 - x) - φ)) →
  ∃ k : ℤ, φ = 8 * π / 3 - k * π →
  |φ| ≥ π / 3 ∧ ∃ φ₀ : ℝ, |φ₀| = π / 3 ∧ 
    (∀ x : ℝ, 2 * Real.sin (2 * x - φ₀) = 2 * Real.sin (2 * (8 * π / 3 - x) - φ₀)) :=
by sorry

end NUMINAMATH_CALUDE_min_abs_phi_l2064_206456


namespace NUMINAMATH_CALUDE_inequality_solution_existence_condition_l2064_206448

-- Define the functions f and g
def f (a x : ℝ) := |2 * x + a| - |2 * x + 3|
def g (x : ℝ) := |x - 1| - 3

-- Theorem for the first part of the problem
theorem inequality_solution (x : ℝ) :
  |g x| < 2 ↔ -4 < x ∧ x < 6 := by sorry

-- Theorem for the second part of the problem
theorem existence_condition (a : ℝ) :
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) ↔ 0 ≤ a ∧ a ≤ 6 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_existence_condition_l2064_206448


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_negative_two_ninths_l2064_206497

-- Define the function f
def f (x : ℝ) : ℝ := (3*x)^2 + 2*(3*x) + 2

-- State the theorem
theorem sum_of_roots_equals_negative_two_ninths :
  ∃ (z₁ z₂ : ℝ), z₁ ≠ z₂ ∧ f z₁ = 10 ∧ f z₂ = 10 ∧ z₁ + z₂ = -2/9 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_negative_two_ninths_l2064_206497


namespace NUMINAMATH_CALUDE_project_duration_proof_l2064_206496

/-- The original duration of the project in months -/
def original_duration : ℝ := 30

/-- The reduction in project duration when efficiency is increased -/
def duration_reduction : ℝ := 6

/-- The factor by which efficiency is increased -/
def efficiency_increase : ℝ := 1.25

theorem project_duration_proof :
  (original_duration - duration_reduction) / original_duration = 1 / efficiency_increase :=
by sorry

#check project_duration_proof

end NUMINAMATH_CALUDE_project_duration_proof_l2064_206496


namespace NUMINAMATH_CALUDE_sqrt_trig_identity_l2064_206402

theorem sqrt_trig_identity : 
  Real.sqrt (2 - Real.sin 2 ^ 2 + Real.cos 4) = -Real.sqrt 3 * Real.cos 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_trig_identity_l2064_206402


namespace NUMINAMATH_CALUDE_lattice_points_form_square_l2064_206493

-- Define a structure for a point in the plane
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define a function to calculate the squared distance between two points
def squaredDistance (p q : Point) : ℤ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

-- Define a function to calculate the area of a triangle given three points
def areaOfTriangle (p q r : Point) : ℚ :=
  let a := squaredDistance p q
  let b := squaredDistance q r
  let c := squaredDistance r p
  ((a + b + c)^2 - 2 * (a^2 + b^2 + c^2)) / 16

-- Theorem statement
theorem lattice_points_form_square (p q r : Point) 
  (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p)
  (h_inequality : (squaredDistance p q).sqrt + (squaredDistance q r).sqrt < (8 * areaOfTriangle p q r + 1).sqrt) :
  ∃ s : Point, s ≠ p ∧ s ≠ q ∧ s ≠ r ∧ 
    squaredDistance p q = squaredDistance q r ∧
    squaredDistance r s = squaredDistance s p ∧
    squaredDistance p q = squaredDistance r s :=
sorry

end NUMINAMATH_CALUDE_lattice_points_form_square_l2064_206493


namespace NUMINAMATH_CALUDE_largest_number_with_property_l2064_206488

/-- A function that returns true if a natural number has all distinct digits --/
def has_distinct_digits (n : ℕ) : Prop := sorry

/-- A function that returns true if a number is not divisible by 11 --/
def not_divisible_by_11 (n : ℕ) : Prop := sorry

/-- A function that returns true if all subsequences of digits in a number are not divisible by 11 --/
def all_subsequences_not_divisible_by_11 (n : ℕ) : Prop := sorry

/-- The main theorem stating that 987654321 is the largest natural number 
    with all distinct digits and all subsequences not divisible by 11 --/
theorem largest_number_with_property : 
  ∀ n : ℕ, n > 987654321 → 
  ¬(has_distinct_digits n ∧ all_subsequences_not_divisible_by_11 n) :=
sorry

end NUMINAMATH_CALUDE_largest_number_with_property_l2064_206488


namespace NUMINAMATH_CALUDE_expression_evaluation_l2064_206439

theorem expression_evaluation : 
  (3^2015 + 3^2013 + 3^2012) / (3^2015 - 3^2013 + 3^2012) = 31/25 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2064_206439


namespace NUMINAMATH_CALUDE_smallest_class_size_seventeen_satisfies_conditions_smallest_class_size_is_seventeen_l2064_206443

theorem smallest_class_size (n : ℕ) : 
  (n % 4 = 1) ∧ (n % 5 = 2) ∧ (n % 7 = 3) → n ≥ 17 :=
by sorry

theorem seventeen_satisfies_conditions : 
  (17 % 4 = 1) ∧ (17 % 5 = 2) ∧ (17 % 7 = 3) :=
by sorry

theorem smallest_class_size_is_seventeen : 
  ∃ (n : ℕ), n = 17 ∧ (n % 4 = 1) ∧ (n % 5 = 2) ∧ (n % 7 = 3) ∧ 
  (∀ m : ℕ, (m % 4 = 1) ∧ (m % 5 = 2) ∧ (m % 7 = 3) → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_class_size_seventeen_satisfies_conditions_smallest_class_size_is_seventeen_l2064_206443


namespace NUMINAMATH_CALUDE_hyperbola_decreasing_condition_l2064_206406

/-- For a hyperbola y = (1-m)/x, y decreases as x increases when x > 0 if and only if m < 1 -/
theorem hyperbola_decreasing_condition (m : ℝ) : 
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ → (1-m)/x₁ > (1-m)/x₂) ↔ m < 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_decreasing_condition_l2064_206406


namespace NUMINAMATH_CALUDE_cone_height_l2064_206404

/-- Given a cone whose lateral surface development is a sector with radius 2 and central angle 180°,
    the height of the cone is √3. -/
theorem cone_height (r : ℝ) (l : ℝ) (h : ℝ) :
  r = 1 →  -- radius of the base (derived from the sector's arc length)
  l = 2 →  -- slant height (radius of the sector)
  h^2 + r^2 = l^2 →  -- Pythagorean theorem
  h = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cone_height_l2064_206404


namespace NUMINAMATH_CALUDE_herring_fat_proof_l2064_206405

/-- The amount of fat in ounces for a herring -/
def herring_fat : ℝ := 40

/-- The amount of fat in ounces for an eel -/
def eel_fat : ℝ := 20

/-- The amount of fat in ounces for a pike -/
def pike_fat : ℝ := eel_fat + 10

/-- The number of each type of fish cooked -/
def fish_count : ℕ := 40

/-- The total amount of fat served in ounces -/
def total_fat : ℝ := 3600

theorem herring_fat_proof : 
  herring_fat * fish_count + eel_fat * fish_count + pike_fat * fish_count = total_fat :=
by sorry

end NUMINAMATH_CALUDE_herring_fat_proof_l2064_206405


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l2064_206445

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 →
  ∃ k : ℕ, k > 0 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) ∧
  ∀ m : ℕ, m > k → ¬(∀ i : ℕ, i > 0 → m ∣ (i * (i + 1) * (i + 2) * (i + 3))) →
  k = 24 :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l2064_206445


namespace NUMINAMATH_CALUDE_sum_of_factors_l2064_206481

theorem sum_of_factors (m n p q : ℤ) : 
  m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q →
  (5 - m) * (5 - n) * (5 - p) * (5 - q) = 9 →
  m + n + p + q = 20 := by
sorry

end NUMINAMATH_CALUDE_sum_of_factors_l2064_206481


namespace NUMINAMATH_CALUDE_circle_radius_is_three_l2064_206433

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x + m*y - 4 = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  2*x + y = 0

-- Define symmetry with respect to a line
def symmetric_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  line_equation ((x₁ + x₂)/2) ((y₁ + y₂)/2)

-- Theorem statement
theorem circle_radius_is_three (m : ℝ) 
  (h₁ : ∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_equation x₁ y₁ m ∧ 
    circle_equation x₂ y₂ m ∧ 
    symmetric_points x₁ y₁ x₂ y₂) :
  (let center_x := 1
   let center_y := -m/2
   let radius := Real.sqrt ((center_x - 0)^2 + (center_y - 0)^2)
   radius = 3) := by sorry

end NUMINAMATH_CALUDE_circle_radius_is_three_l2064_206433


namespace NUMINAMATH_CALUDE_chase_travel_time_l2064_206470

/-- Represents the travel time between Granville and Salisbury -/
def travel_time (speed : ℝ) : ℝ := sorry

theorem chase_travel_time :
  let chase_speed : ℝ := 1
  let cameron_speed : ℝ := 2 * chase_speed
  let danielle_speed : ℝ := 3 * cameron_speed
  let danielle_time : ℝ := 30

  travel_time chase_speed = 180 := by sorry

end NUMINAMATH_CALUDE_chase_travel_time_l2064_206470


namespace NUMINAMATH_CALUDE_cherry_weekly_earnings_l2064_206416

/-- Represents Cherry's delivery service earnings --/
def cherry_earnings : ℝ → ℝ → ℝ → ℝ → ℝ := λ price_small price_large num_small num_large =>
  (price_small * num_small + price_large * num_large) * 7

/-- Theorem stating Cherry's weekly earnings --/
theorem cherry_weekly_earnings :
  let price_small := 2.5
  let price_large := 4
  let num_small := 4
  let num_large := 2
  cherry_earnings price_small price_large num_small num_large = 126 :=
by sorry

end NUMINAMATH_CALUDE_cherry_weekly_earnings_l2064_206416


namespace NUMINAMATH_CALUDE_usual_time_calculation_l2064_206424

/-- Given a man who walks at P% of his usual speed and takes T minutes more than usual,
    his usual time U (in minutes) to cover the distance is (P * T) / (100 - P). -/
theorem usual_time_calculation (P T : ℝ) (h1 : 0 < P) (h2 : P < 100) (h3 : 0 < T) :
  ∃ U : ℝ, U > 0 ∧ U = (P * T) / (100 - P) :=
sorry

end NUMINAMATH_CALUDE_usual_time_calculation_l2064_206424


namespace NUMINAMATH_CALUDE_some_seniors_not_club_members_l2064_206463

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Senior : U → Prop)
variable (Punctual : U → Prop)
variable (ClubMember : U → Prop)

-- State the theorem
theorem some_seniors_not_club_members
  (h1 : ∃ x, Senior x ∧ ¬Punctual x)
  (h2 : ∀ x, ClubMember x → Punctual x) :
  ∃ x, Senior x ∧ ¬ClubMember x :=
by
  sorry


end NUMINAMATH_CALUDE_some_seniors_not_club_members_l2064_206463


namespace NUMINAMATH_CALUDE_printing_presses_count_l2064_206460

/-- The number of papers printed -/
def num_papers : ℕ := 500000

/-- The time taken in the first scenario (in hours) -/
def time1 : ℝ := 12

/-- The time taken in the second scenario (in hours) -/
def time2 : ℝ := 13.999999999999998

/-- The number of printing presses in the second scenario -/
def presses2 : ℕ := 30

/-- The number of printing presses in the first scenario -/
def presses1 : ℕ := 26

theorem printing_presses_count :
  (num_papers : ℝ) / time1 / (num_papers / time2) = presses1 / presses2 :=
sorry

end NUMINAMATH_CALUDE_printing_presses_count_l2064_206460


namespace NUMINAMATH_CALUDE_no_perfect_square_n_n_plus_one_l2064_206441

theorem no_perfect_square_n_n_plus_one (n : ℕ) (hn : n > 0) : 
  ¬∃ (k : ℕ), n * (n + 1) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_n_n_plus_one_l2064_206441


namespace NUMINAMATH_CALUDE_initial_order_size_l2064_206412

/-- The number of cogs produced per hour in the initial phase -/
def initial_rate : ℕ := 36

/-- The number of cogs produced per hour in the second phase -/
def second_rate : ℕ := 60

/-- The number of additional cogs produced in the second phase -/
def additional_cogs : ℕ := 60

/-- The overall average output in cogs per hour -/
def average_output : ℝ := 45

/-- The theorem stating that the initial order was for 60 cogs -/
theorem initial_order_size :
  ∃ x : ℕ, 
    (x + additional_cogs) / (x / initial_rate + 1 : ℝ) = average_output →
    x = 60 := by
  sorry

end NUMINAMATH_CALUDE_initial_order_size_l2064_206412


namespace NUMINAMATH_CALUDE_divisibility_property_l2064_206400

theorem divisibility_property (a b n : ℕ) (h : a^n ∣ b) : a^(n+1) ∣ (a+1)^b - 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l2064_206400


namespace NUMINAMATH_CALUDE_select_workers_count_l2064_206421

/-- The number of ways to select two workers from a group of three for day and night shifts -/
def select_workers : ℕ :=
  let workers := 3
  let day_shift_choices := workers
  let night_shift_choices := workers - 1
  day_shift_choices * night_shift_choices

/-- Theorem: The number of ways to select two workers from a group of three for day and night shifts is 6 -/
theorem select_workers_count : select_workers = 6 := by
  sorry

end NUMINAMATH_CALUDE_select_workers_count_l2064_206421


namespace NUMINAMATH_CALUDE_conic_section_eccentricity_l2064_206414

/-- Given that real numbers 4, m, 9 form a geometric sequence,
    prove that the eccentricity of the conic section x^2/m + y^2 = 1
    is either √30/6 or √7 -/
theorem conic_section_eccentricity (m : ℝ) :
  (4 * m = m * 9) →
  let e := if m > 0
           then Real.sqrt (1 - m / 6) / Real.sqrt (m / 6)
           else Real.sqrt (1 + 6 / m) / 1
  (e = Real.sqrt 30 / 6 ∨ e = Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_conic_section_eccentricity_l2064_206414


namespace NUMINAMATH_CALUDE_divisible_by_18_sqrt_between_30_and_30_5_l2064_206426

theorem divisible_by_18_sqrt_between_30_and_30_5 :
  ∃ (n : ℕ), n > 0 ∧ n % 18 = 0 ∧ 30 < Real.sqrt n ∧ Real.sqrt n < 30.5 ∧
  (n = 900 ∨ n = 918) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_18_sqrt_between_30_and_30_5_l2064_206426


namespace NUMINAMATH_CALUDE_total_students_accommodated_l2064_206435

/-- Represents a bus with its seating configuration and broken seats -/
structure Bus where
  columns : Nat
  rows : Nat
  broken_seats : Nat

/-- Calculates the number of usable seats in a bus -/
def usable_seats (bus : Bus) : Nat :=
  bus.columns * bus.rows - bus.broken_seats

/-- The list of buses with their configurations -/
def buses : List Bus := [
  ⟨4, 10, 2⟩,
  ⟨5, 8, 4⟩,
  ⟨3, 12, 3⟩,
  ⟨4, 12, 1⟩,
  ⟨6, 8, 5⟩,
  ⟨5, 10, 2⟩
]

/-- Theorem: The total number of students that can be accommodated is 245 -/
theorem total_students_accommodated : (buses.map usable_seats).sum = 245 := by
  sorry


end NUMINAMATH_CALUDE_total_students_accommodated_l2064_206435


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2064_206484

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  d > 0 →  -- positive common difference
  a 1 + a 2 + a 3 = 15 →  -- first condition
  a 1 * a 2 * a 3 = 80 →  -- second condition
  a 11 + a 12 + a 13 = 105 :=  -- conclusion
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2064_206484


namespace NUMINAMATH_CALUDE_inequality_theorem_l2064_206420

theorem inequality_theorem (a : ℚ) (x : ℝ) :
  ((a > 1 ∨ a < 0) ∧ x > 0 ∧ x ≠ 1 → x^(a : ℝ) - a * x + a - 1 > 0) ∧
  (0 < a ∧ a < 1 ∧ x > 0 ∧ x ≠ 1 → x^(a : ℝ) - a * x + a - 1 < 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2064_206420


namespace NUMINAMATH_CALUDE_intersection_point_unique_l2064_206408

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (-3/5, -4/5)

/-- First line equation: y = 3x + 1 -/
def line1 (x y : ℚ) : Prop := y = 3 * x + 1

/-- Second line equation: y + 5 = -7x -/
def line2 (x y : ℚ) : Prop := y + 5 = -7 * x

theorem intersection_point_unique :
  let (x, y) := intersection_point
  (line1 x y ∧ line2 x y) ∧
  ∀ x' y', line1 x' y' ∧ line2 x' y' → (x', y') = (x, y) := by sorry

end NUMINAMATH_CALUDE_intersection_point_unique_l2064_206408


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2064_206476

theorem polynomial_factorization (x y z : ℝ) :
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  (x - y) * (y - z) * (z - x) * (x*y + x*z + y*z) := by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2064_206476


namespace NUMINAMATH_CALUDE_sum_of_disk_areas_l2064_206491

/-- The number of disks placed on the circle -/
def n : ℕ := 15

/-- The radius of the large circle -/
def R : ℝ := 1

/-- Represents the arrangement of disks on the circle -/
structure DiskArrangement where
  /-- The radius of each small disk -/
  r : ℝ
  /-- The disks cover the entire circle -/
  covers_circle : r > 0
  /-- The disks do not overlap -/
  no_overlap : 2 * n * r ≤ 2 * π * R
  /-- Each disk is tangent to its neighbors -/
  tangent_neighbors : 2 * n * r = 2 * π * R

/-- The theorem stating the sum of areas of the disks -/
theorem sum_of_disk_areas (arrangement : DiskArrangement) :
  n * π * arrangement.r^2 = 105 * π - 60 * π * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_disk_areas_l2064_206491


namespace NUMINAMATH_CALUDE_socks_cost_proof_l2064_206436

/-- The cost of a uniform item without discount -/
structure UniformItem where
  shirt : ℝ
  pants : ℝ
  socks : ℝ

/-- The cost of a uniform item with discount -/
structure DiscountedUniformItem where
  shirt : ℝ
  pants : ℝ
  socks : ℝ

def team_size : ℕ := 12
def team_savings : ℝ := 36

def regular_uniform : UniformItem :=
  { shirt := 7.5,
    pants := 15,
    socks := 4.5 }  -- We use the answer here as we're proving this value

def discounted_uniform : DiscountedUniformItem :=
  { shirt := 6.75,
    pants := 13.5,
    socks := 3.75 }

theorem socks_cost_proof :
  let regular_total := team_size * (regular_uniform.shirt + regular_uniform.pants + regular_uniform.socks)
  let discounted_total := team_size * (discounted_uniform.shirt + discounted_uniform.pants + discounted_uniform.socks)
  regular_total - discounted_total = team_savings :=
by sorry

end NUMINAMATH_CALUDE_socks_cost_proof_l2064_206436


namespace NUMINAMATH_CALUDE_rent_percentage_calculation_l2064_206485

theorem rent_percentage_calculation (last_year_earnings last_year_rent_percentage this_year_earnings_percentage this_year_rent_amount_percentage : ℝ) 
  (h1 : last_year_rent_percentage = 20)
  (h2 : this_year_earnings_percentage = 125)
  (h3 : this_year_rent_amount_percentage = 187.5) : 
  (this_year_rent_amount_percentage / 100) * (last_year_rent_percentage / 100) / (this_year_earnings_percentage / 100) = 0.3 := by
sorry

end NUMINAMATH_CALUDE_rent_percentage_calculation_l2064_206485
