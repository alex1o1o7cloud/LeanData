import Mathlib

namespace NUMINAMATH_CALUDE_monomial_coefficient_l1094_109448

/-- The coefficient of a monomial is the numerical factor that multiplies the variables. -/
def coefficient (m : ℚ) (x y : ℚ) : ℚ := m

/-- The monomial -9/4 * x^2 * y -/
def monomial (x y : ℚ) : ℚ := -9/4 * x^2 * y

theorem monomial_coefficient :
  coefficient (-9/4) x y = -9/4 :=
by sorry

end NUMINAMATH_CALUDE_monomial_coefficient_l1094_109448


namespace NUMINAMATH_CALUDE_lcm_24_90_35_l1094_109459

theorem lcm_24_90_35 : Nat.lcm 24 (Nat.lcm 90 35) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_lcm_24_90_35_l1094_109459


namespace NUMINAMATH_CALUDE_xyz_inequality_l1094_109495

theorem xyz_inequality (x y z : ℝ) (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 0) :
  x * y > x * z :=
by sorry

end NUMINAMATH_CALUDE_xyz_inequality_l1094_109495


namespace NUMINAMATH_CALUDE_range_of_a_l1094_109435

def star_op (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (a : ℝ) :
  (∀ x, star_op x (x - a) > 0 → -1 ≤ x ∧ x ≤ 1) →
  -2 ≤ a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1094_109435


namespace NUMINAMATH_CALUDE_train_crossing_time_l1094_109422

/-- Given a train traveling at a certain speed that crosses a platform of known length in a specific time,
    calculate the time it takes for the train to cross a man standing on the platform. -/
theorem train_crossing_time
  (train_speed_kmph : ℝ)
  (train_speed_mps : ℝ)
  (platform_length : ℝ)
  (platform_crossing_time : ℝ)
  (h1 : train_speed_kmph = 72)
  (h2 : train_speed_mps = 20)
  (h3 : platform_length = 220)
  (h4 : platform_crossing_time = 30)
  (h5 : train_speed_mps = train_speed_kmph * (1000 / 3600)) :
  (train_speed_mps * platform_crossing_time - platform_length) / train_speed_mps = 19 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1094_109422


namespace NUMINAMATH_CALUDE_least_sum_m_n_l1094_109497

theorem least_sum_m_n (m n : ℕ+) (h1 : Nat.gcd (m + n) 330 = 1)
  (h2 : ∃ k : ℕ, m^(m : ℕ) = k * n^(n : ℕ)) (h3 : ¬∃ k : ℕ, m = k * n) :
  m + n ≥ 377 ∧ ∃ m' n' : ℕ+, m' + n' = 377 ∧ 
    Nat.gcd (m' + n') 330 = 1 ∧ 
    (∃ k : ℕ, (m' : ℕ)^(m' : ℕ) = k * (n' : ℕ)^(n' : ℕ)) ∧ 
    ¬∃ k : ℕ, (m' : ℕ) = k * (n' : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_least_sum_m_n_l1094_109497


namespace NUMINAMATH_CALUDE_chess_tournament_players_l1094_109410

theorem chess_tournament_players (total_games : ℕ) (h : total_games = 380) :
  ∃ n : ℕ, n > 0 ∧ 2 * n * (n - 1) = total_games :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l1094_109410


namespace NUMINAMATH_CALUDE_function_properties_l1094_109405

noncomputable section

variable (a : ℝ)
variable (f : ℝ → ℝ)

def has_extremum_in (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x ∈ Set.Ioo a b, ∀ y ∈ Set.Ioo a b, f x ≤ f y ∨ f x ≥ f y

def has_unique_zero (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = 0

theorem function_properties (h1 : a > 0) 
    (h2 : f = λ x => Real.exp (2*x) + 2 / Real.exp x - a*x) :
  (has_extremum_in f 0 1 → 0 < a ∧ a < 2 * Real.exp 2 - 2 / Real.exp 1) ∧
  (has_unique_zero f → ∃ x₀, f x₀ = 0 ∧ Real.log 2 < x₀ ∧ x₀ < 1) := by
  sorry

end

end NUMINAMATH_CALUDE_function_properties_l1094_109405


namespace NUMINAMATH_CALUDE_remaining_money_l1094_109444

def base_8_to_10 (n : Nat) : Nat :=
  (n / 1000) * 512 + ((n / 100) % 10) * 64 + ((n / 10) % 10) * 8 + (n % 10)

def savings : Nat := 5377
def airline_ticket : Nat := 1200
def travel_pass : Nat := 600

theorem remaining_money :
  base_8_to_10 savings - airline_ticket - travel_pass = 1015 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l1094_109444


namespace NUMINAMATH_CALUDE_conference_handshakes_l1094_109412

/-- The number of handshakes in a conference with multiple companies --/
def num_handshakes (num_companies : ℕ) (representatives_per_company : ℕ) : ℕ :=
  let total_people := num_companies * representatives_per_company
  let handshakes_per_person := total_people - representatives_per_company
  (total_people * handshakes_per_person) / 2

/-- Theorem: In a conference with 5 companies, each having 5 representatives,
    where every person shakes hands once with every person except those from
    their own company, the total number of handshakes is 250. --/
theorem conference_handshakes :
  num_handshakes 5 5 = 250 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l1094_109412


namespace NUMINAMATH_CALUDE_dice_probability_l1094_109469

def num_dice : ℕ := 6
def num_success : ℕ := 3
def prob_success : ℚ := 1/3
def prob_failure : ℚ := 2/3

theorem dice_probability : 
  (Nat.choose num_dice num_success : ℚ) * prob_success^num_success * prob_failure^(num_dice - num_success) = 160/729 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l1094_109469


namespace NUMINAMATH_CALUDE_grid_drawing_theorem_l1094_109487

/-- Represents a grid configuration -/
structure GridConfiguration (n : ℕ+) :=
  (has_diagonal : Fin n → Fin n → Bool)
  (start_vertex : Fin n × Fin n)
  (is_valid : Bool)

/-- Checks if a grid configuration is valid according to the problem conditions -/
def is_valid_configuration (n : ℕ+) (config : GridConfiguration n) : Prop :=
  -- Adjacent cells have diagonals in different directions
  (∀ i j, config.has_diagonal i j → ¬(config.has_diagonal (i+1) j ∧ config.has_diagonal i (j+1))) ∧
  -- Can be drawn in one stroke starting from bottom-left vertex
  (config.start_vertex = (0, 0)) ∧
  -- Each edge or diagonal is traversed exactly once
  config.is_valid

/-- The main theorem stating that only n = 1, 2, 3 satisfy the conditions -/
theorem grid_drawing_theorem :
  ∀ n : ℕ+, (∃ config : GridConfiguration n, is_valid_configuration n config) ↔ n = 1 ∨ n = 2 ∨ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_grid_drawing_theorem_l1094_109487


namespace NUMINAMATH_CALUDE_larger_ball_radius_larger_ball_radius_proof_l1094_109430

theorem larger_ball_radius : ℝ → Prop :=
  fun r : ℝ =>
    -- Volume of a sphere: (4/3) * π * r^3
    let volume_sphere (radius : ℝ) := (4/3) * Real.pi * (radius ^ 3)
    -- Volume of 10 balls with radius 2
    let volume_ten_balls := 10 * volume_sphere 2
    -- Volume of 2 balls with radius 1
    let volume_two_small_balls := 2 * volume_sphere 1
    -- Volume of the larger ball with radius r
    let volume_larger_ball := volume_sphere r
    -- The total volume equality
    volume_ten_balls = volume_larger_ball + volume_two_small_balls →
    -- The radius of the larger ball is ∛78
    r = Real.rpow 78 (1/3)

-- The proof is omitted
theorem larger_ball_radius_proof : larger_ball_radius (Real.rpow 78 (1/3)) := by sorry

end NUMINAMATH_CALUDE_larger_ball_radius_larger_ball_radius_proof_l1094_109430


namespace NUMINAMATH_CALUDE_units_digit_problem_l1094_109443

/-- Given a positive even integer with a positive units digit,
    if the units digit of its cube minus the units digit of its square is 0,
    then the number needed to be added to its units digit to get 10 is 4. -/
theorem units_digit_problem (p : ℕ) : 
  p > 0 → 
  Even p → 
  p % 10 > 0 → 
  p % 10 < 10 → 
  (p^3 % 10) - (p^2 % 10) = 0 → 
  10 - (p % 10) = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l1094_109443


namespace NUMINAMATH_CALUDE_polygon_sides_from_diagonals_l1094_109433

theorem polygon_sides_from_diagonals :
  ∃ (n : ℕ), n > 2 ∧ (n * (n - 3)) / 2 = 15 ∧ 
  (∀ (m : ℕ), m > 2 → (m * (m - 3)) / 2 = 15 → m = n) ∧
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_from_diagonals_l1094_109433


namespace NUMINAMATH_CALUDE_zach_babysitting_hours_l1094_109418

def bike_cost : ℕ := 100
def weekly_allowance : ℕ := 5
def lawn_mowing_pay : ℕ := 10
def babysitting_rate : ℕ := 7
def current_savings : ℕ := 65
def additional_needed : ℕ := 6

theorem zach_babysitting_hours :
  ∃ (hours : ℕ),
    bike_cost = current_savings + weekly_allowance + lawn_mowing_pay + babysitting_rate * hours + additional_needed ∧
    hours = 2 := by
  sorry

end NUMINAMATH_CALUDE_zach_babysitting_hours_l1094_109418


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1094_109439

theorem fraction_evaluation : 
  let numerator := (12^4 + 288) * (24^4 + 288) * (36^4 + 288) * (48^4 + 288) * (60^4 + 288)
  let denominator := (6^4 + 288) * (18^4 + 288) * (30^4 + 288) * (42^4 + 288) * (54^4 + 288)
  numerator / denominator = -332 := by
sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1094_109439


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l1094_109423

/-- Proves that for a journey of given distance and original time,
    the average speed required to complete the same journey in a 
    multiple of the original time is as calculated. -/
theorem journey_speed_calculation 
  (distance : ℝ) 
  (original_time : ℝ) 
  (time_multiplier : ℝ) 
  (h1 : distance = 378) 
  (h2 : original_time = 6) 
  (h3 : time_multiplier = 3/2) :
  distance / (original_time * time_multiplier) = 42 :=
by
  sorry

#check journey_speed_calculation

end NUMINAMATH_CALUDE_journey_speed_calculation_l1094_109423


namespace NUMINAMATH_CALUDE_tangent_line_at_one_monotonicity_intervals_minimum_value_on_interval_l1094_109425

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 1

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 6 * x^2 + 6 * a * x

-- Theorem for the tangent line equation when a = 0
theorem tangent_line_at_one (x y : ℝ) :
  f 0 1 = 3 ∧ f' 0 1 = 6 → (6 * x - y - 3 = 0 ↔ y - 3 = 6 * (x - 1)) :=
sorry

-- Theorem for monotonicity intervals
theorem monotonicity_intervals (a : ℝ) :
  (a = 0 → ∀ x, f' a x ≥ 0) ∧
  (a > 0 → ∀ x, (x < -a ∨ x > 0) ↔ f' a x > 0) ∧
  (a < 0 → ∀ x, (x < 0 ∨ x > -a) ↔ f' a x > 0) :=
sorry

-- Theorem for minimum value on [0, 2]
theorem minimum_value_on_interval (a : ℝ) :
  (a ≥ 0 → ∀ x ∈ Set.Icc 0 2, f a x ≥ f a 0) ∧
  (-2 < a ∧ a < 0 → ∀ x ∈ Set.Icc 0 2, f a x ≥ f a (-a)) ∧
  (a ≤ -2 → ∀ x ∈ Set.Icc 0 2, f a x ≥ f a 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_monotonicity_intervals_minimum_value_on_interval_l1094_109425


namespace NUMINAMATH_CALUDE_value_of_N_l1094_109499

theorem value_of_N : ∃ N : ℝ, (0.20 * N = 0.30 * 5000) ∧ (N = 7500) := by
  sorry

end NUMINAMATH_CALUDE_value_of_N_l1094_109499


namespace NUMINAMATH_CALUDE_inequality_must_hold_l1094_109456

theorem inequality_must_hold (a b c : ℝ) (h : a > b ∧ b > c) : a - |c| > b - |c| := by
  sorry

end NUMINAMATH_CALUDE_inequality_must_hold_l1094_109456


namespace NUMINAMATH_CALUDE_salt_solution_problem_l1094_109413

/-- Proves the initial water mass and percentage increase given final conditions --/
theorem salt_solution_problem (final_mass : ℝ) (final_concentration : ℝ) 
  (h_final_mass : final_mass = 850)
  (h_final_concentration : final_concentration = 0.36) : 
  ∃ (initial_mass : ℝ) (percentage_increase : ℝ),
    initial_mass = 544 ∧ 
    percentage_increase = 25 ∧
    final_mass = initial_mass * (1 + percentage_increase / 100)^2 ∧
    final_concentration = 1 - (initial_mass / final_mass) :=
by
  sorry


end NUMINAMATH_CALUDE_salt_solution_problem_l1094_109413


namespace NUMINAMATH_CALUDE_lower_limit_of_range_l1094_109493

theorem lower_limit_of_range (x : ℕ) : 
  x ≤ 100 ∧ 
  (∃ (S : Finset ℕ), S.card = 13 ∧ 
    (∀ n ∈ S, x ≤ n ∧ n ≤ 100 ∧ n % 6 = 0) ∧
    (∀ n, x ≤ n ∧ n ≤ 100 ∧ n % 6 = 0 → n ∈ S)) →
  x = 24 :=
by sorry

end NUMINAMATH_CALUDE_lower_limit_of_range_l1094_109493


namespace NUMINAMATH_CALUDE_tangent_line_at_one_max_value_min_value_f_properties_l1094_109442

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - x^2 + 1

-- Theorem for the tangent line equation
theorem tangent_line_at_one : 
  ∃ (m b : ℝ), ∀ x y, y = m * x + b ↔ (x = 1 ∧ y = f 1) ∨ (y - f 1 = m * (x - 1)) :=
sorry

-- Theorem for the maximum value
theorem max_value :
  ∃ x_max, f x_max = 1 ∧ ∀ x, f x ≤ 1 :=
sorry

-- Theorem for the minimum value
theorem min_value :
  ∃ x_min, f x_min = 23/27 ∧ ∀ x, f x ≥ 23/27 :=
sorry

-- Theorem combining all results
theorem f_properties :
  (∃ (m b : ℝ), ∀ x y, y = m * x + b ↔ (x = 1 ∧ y = f 1) ∨ (y - f 1 = m * (x - 1))) ∧
  (∃ x_max, f x_max = 1 ∧ ∀ x, f x ≤ 1) ∧
  (∃ x_min, f x_min = 23/27 ∧ ∀ x, f x ≥ 23/27) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_max_value_min_value_f_properties_l1094_109442


namespace NUMINAMATH_CALUDE_equation_solution_l1094_109480

theorem equation_solution : ∃ x : ℚ, 3 * x - 6 = |(-21 + 8 - 3)| ∧ x = 22 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1094_109480


namespace NUMINAMATH_CALUDE_linear_relationship_scaling_l1094_109492

/-- Given a linear relationship where an increase of 4 units in x results in an increase of 10 units in y,
    prove that an increase of 12 units in x results in an increase of 30 units in y. -/
theorem linear_relationship_scaling (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 4) - f x = 10) :
  ∀ x : ℝ, f (x + 12) - f x = 30 := by
sorry

end NUMINAMATH_CALUDE_linear_relationship_scaling_l1094_109492


namespace NUMINAMATH_CALUDE_cake_two_sided_icing_count_l1094_109498

/-- Represents a cube cake with icing on specific faces -/
structure CakeCube where
  size : Nat
  icedFaces : Finset (Fin 3)

/-- Counts the number of 1×1×1 subcubes with icing on exactly two sides -/
def countTwoSidedIcingCubes (cake : CakeCube) : Nat :=
  sorry

/-- The main theorem stating that a 5×5×5 cake with icing on top, front, and back
    has exactly 12 subcubes with icing on two sides when cut into 1×1×1 cubes -/
theorem cake_two_sided_icing_count :
  let cake : CakeCube := { size := 5, icedFaces := {0, 1, 2} }
  countTwoSidedIcingCubes cake = 12 := by
  sorry

end NUMINAMATH_CALUDE_cake_two_sided_icing_count_l1094_109498


namespace NUMINAMATH_CALUDE_student_count_l1094_109406

theorem student_count (n : ℕ) (rank_top : ℕ) (rank_bottom : ℕ) 
  (h1 : rank_top = 30) 
  (h2 : rank_bottom = 30) : 
  n = 59 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l1094_109406


namespace NUMINAMATH_CALUDE_coffee_mixture_cost_l1094_109453

theorem coffee_mixture_cost (cost_A : ℝ) (cost_mixture : ℝ) (total_weight : ℝ) (weight_A : ℝ) (weight_B : ℝ) :
  cost_A = 10 →
  cost_mixture = 11 →
  total_weight = 480 →
  weight_A = 240 →
  weight_B = 240 →
  (total_weight * cost_mixture - weight_A * cost_A) / weight_B = 12 :=
by sorry

end NUMINAMATH_CALUDE_coffee_mixture_cost_l1094_109453


namespace NUMINAMATH_CALUDE_m_range_theorem_l1094_109490

/-- The range of values for m satisfying the given conditions -/
def m_range (m : ℝ) : Prop :=
  (-2 ≤ m ∧ m < 1) ∨ m > 2

/-- Condition p: The solution set of x^2 + mx + 1 < 0 is empty -/
def condition_p (m : ℝ) : Prop :=
  ∀ x, x^2 + m*x + 1 ≥ 0

/-- Condition q: The function 4x^2 + 4(m-1)x + 3 has no extreme value -/
def condition_q (m : ℝ) : Prop :=
  ∀ x, 8*x + 4*(m-1) ≠ 0

/-- Theorem stating the range of m given the conditions -/
theorem m_range_theorem (m : ℝ)
  (h1 : condition_p m ∨ condition_q m)
  (h2 : ¬(condition_p m ∧ condition_q m)) :
  m_range m :=
sorry

end NUMINAMATH_CALUDE_m_range_theorem_l1094_109490


namespace NUMINAMATH_CALUDE_total_chairs_moved_l1094_109427

/-- The total number of chairs agreed to be moved is equal to the sum of
    chairs moved by Carey, chairs moved by Pat, and chairs left to move. -/
theorem total_chairs_moved (carey_chairs pat_chairs left_chairs : ℕ)
  (h1 : carey_chairs = 28)
  (h2 : pat_chairs = 29)
  (h3 : left_chairs = 17) :
  carey_chairs + pat_chairs + left_chairs = 74 := by
  sorry

end NUMINAMATH_CALUDE_total_chairs_moved_l1094_109427


namespace NUMINAMATH_CALUDE_root_sum_absolute_value_l1094_109404

theorem root_sum_absolute_value (m : ℝ) (α β : ℝ) 
  (h1 : α^2 - 22*α + m = 0)
  (h2 : β^2 - 22*β + m = 0)
  (h3 : m ≤ 121) : 
  |α| + |β| = if 0 ≤ m then 22 else Real.sqrt (484 - 4*m) :=
by sorry

end NUMINAMATH_CALUDE_root_sum_absolute_value_l1094_109404


namespace NUMINAMATH_CALUDE_intersection_when_m_is_5_range_of_m_for_necessary_not_sufficient_l1094_109440

def A : Set ℝ := {x : ℝ | x^2 - 8*x + 7 ≤ 0}

def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem intersection_when_m_is_5 : 
  A ∩ B 5 = {x : ℝ | 6 ≤ x ∧ x ≤ 7} := by sorry

theorem range_of_m_for_necessary_not_sufficient :
  (∀ m : ℝ, (B m).Nonempty → (B m ⊆ A ∧ B m ≠ A)) ↔ 2 ≤ m ∧ m ≤ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_5_range_of_m_for_necessary_not_sufficient_l1094_109440


namespace NUMINAMATH_CALUDE_pressure_change_pressure_at_4m3_l1094_109481

/-- Represents the pressure-volume relationship for a gas -/
structure GasPV where
  k : ℝ
  pressure : ℝ → ℝ
  volume : ℝ → ℝ
  inverse_square_relation : ∀ v, pressure v = k / (volume v)^2

/-- The theorem stating the pressure when volume changes -/
theorem pressure_change (gas : GasPV) (v₁ v₂ : ℝ) (p₁ : ℝ) 
    (h₁ : gas.pressure v₁ = p₁)
    (h₂ : v₁ > 0)
    (h₃ : v₂ > 0)
    (h₄ : gas.volume v₁ = v₁)
    (h₅ : gas.volume v₂ = v₂) :
  gas.pressure v₂ = p₁ * (v₁ / v₂)^2 :=
by sorry

/-- The specific problem instance -/
theorem pressure_at_4m3 (gas : GasPV) 
    (h₁ : gas.pressure 2 = 25)
    (h₂ : gas.volume 2 = 2)
    (h₃ : gas.volume 4 = 4) :
  gas.pressure 4 = 6.25 :=
by sorry

end NUMINAMATH_CALUDE_pressure_change_pressure_at_4m3_l1094_109481


namespace NUMINAMATH_CALUDE_interest_rate_calculation_interest_rate_value_l1094_109463

/-- The interest rate for Rs 100 over 8 years that produces the same interest as Rs 200 at 10% for 2 years -/
def interest_rate : ℝ := sorry

/-- The initial amount in rupees -/
def initial_amount : ℝ := 100

/-- The time period in years -/
def time_period : ℝ := 8

/-- The comparison amount in rupees -/
def comparison_amount : ℝ := 200

/-- The comparison interest rate -/
def comparison_rate : ℝ := 0.1

/-- The comparison time period in years -/
def comparison_time : ℝ := 2

theorem interest_rate_calculation : 
  initial_amount * interest_rate * time_period = 
  comparison_amount * comparison_rate * comparison_time :=
sorry

theorem interest_rate_value : interest_rate = 0.05 :=
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_interest_rate_value_l1094_109463


namespace NUMINAMATH_CALUDE_sequence_sum_unique_value_l1094_109414

def is_strictly_increasing (s : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → s n < s m

theorem sequence_sum_unique_value
  (a b : ℕ → ℕ)
  (h_a_incr : is_strictly_increasing a)
  (h_b_incr : is_strictly_increasing b)
  (h_eq : a 10 = b 10)
  (h_lt : a 10 < 2017)
  (h_a_rec : ∀ n : ℕ, a (n + 2) = a (n + 1) + a n)
  (h_b_rec : ∀ n : ℕ, b (n + 1) = 2 * b n) :
  a 1 + b 1 = 5 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_unique_value_l1094_109414


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1094_109485

theorem inequality_and_equality_condition (a b c : ℝ) (h : a^2 + b^2 + c^2 = 3) :
  (a^2 / (2 + b + c^2) + b^2 / (2 + c + a^2) + c^2 / (2 + a + b^2) ≥ (a + b + c)^2 / 12) ∧
  ((a^2 / (2 + b + c^2) + b^2 / (2 + c + a^2) + c^2 / (2 + a + b^2) = (a + b + c)^2 / 12) ↔ 
   (a = 1 ∧ b = 1 ∧ c = 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1094_109485


namespace NUMINAMATH_CALUDE_greater_number_proof_l1094_109450

theorem greater_number_proof (x y : ℝ) (h1 : x > y) (h2 : x + y = 40) (h3 : x - y = 10) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_proof_l1094_109450


namespace NUMINAMATH_CALUDE_gary_chickens_l1094_109483

theorem gary_chickens (initial_chickens : ℕ) : 
  (∃ (current_chickens : ℕ), 
    current_chickens = 8 * initial_chickens ∧ 
    6 * 7 * current_chickens = 1344) → 
  initial_chickens = 4 := by
sorry

end NUMINAMATH_CALUDE_gary_chickens_l1094_109483


namespace NUMINAMATH_CALUDE_census_population_scientific_notation_l1094_109457

/-- 
Given a positive integer n, its scientific notation is a representation of the form a × 10^b, 
where 1 ≤ a < 10 and b is an integer.
-/
def scientific_notation (n : ℕ+) : ℝ × ℤ := sorry

theorem census_population_scientific_notation :
  scientific_notation 932700 = (9.327, 5) := by sorry

end NUMINAMATH_CALUDE_census_population_scientific_notation_l1094_109457


namespace NUMINAMATH_CALUDE_log_sum_and_product_implies_arithmetic_mean_l1094_109419

theorem log_sum_and_product_implies_arithmetic_mean (x y : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : Real.log x / Real.log y + Real.log y / Real.log x = 10/3) 
  (h4 : x * y = 144) : 
  (x + y) / 2 = 13 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_and_product_implies_arithmetic_mean_l1094_109419


namespace NUMINAMATH_CALUDE_problem_statement_l1094_109489

theorem problem_statement (x y : ℝ) 
  (h1 : 3 * x + 4 * y = 5) 
  (h2 : 2 * x + 5 * y = 8) : 
  9 * x^2 + 38 * x * y + 41 * y^2 = 153 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1094_109489


namespace NUMINAMATH_CALUDE_percentage_problem_l1094_109482

theorem percentage_problem (x : ℝ) :
  (15 / 100) * (30 / 100) * (50 / 100) * x = 126 →
  x = 5600 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l1094_109482


namespace NUMINAMATH_CALUDE_common_point_sum_mod_9_l1094_109460

theorem common_point_sum_mod_9 : ∃ (x : ℤ), 
  (∀ (y : ℤ), (y ≡ 3*x + 5 [ZMOD 9] ↔ y ≡ 7*x + 3 [ZMOD 9])) ∧ 
  (x ≡ 5 [ZMOD 9]) := by
  sorry

end NUMINAMATH_CALUDE_common_point_sum_mod_9_l1094_109460


namespace NUMINAMATH_CALUDE_max_value_constraint_l1094_109496

theorem max_value_constraint (x y : ℝ) (h : 3 * x^2 + 4 * y^2 = 12) :
  ∃ (max : ℝ), max = 4 * Real.sqrt 3 ∧ ∀ (a b : ℝ), 3 * a^2 + 4 * b^2 = 12 → 3 * a + 2 * b ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_constraint_l1094_109496


namespace NUMINAMATH_CALUDE_book_arrangement_count_l1094_109445

/-- Represents the number of ways to arrange books on a shelf. -/
def arrange_books (math_books : ℕ) (english_books : ℕ) (science_books : ℕ) : ℕ :=
  let group_arrangements := 6
  let math_arrangements := Nat.factorial math_books
  let english_arrangements := Nat.factorial english_books
  let science_arrangements := Nat.factorial science_books
  group_arrangements * math_arrangements * english_arrangements * science_arrangements

/-- Theorem stating the number of ways to arrange the books on the shelf. -/
theorem book_arrangement_count :
  arrange_books 4 6 2 = 207360 :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l1094_109445


namespace NUMINAMATH_CALUDE_circles_intersect_l1094_109403

/-- Two circles in a 2D plane --/
structure TwoCircles where
  c1_center : ℝ × ℝ
  c2_center : ℝ × ℝ
  c1_radius : ℝ
  c2_radius : ℝ

/-- Definition of intersecting circles --/
def are_intersecting (tc : TwoCircles) : Prop :=
  let d := Real.sqrt ((tc.c2_center.1 - tc.c1_center.1)^2 + (tc.c2_center.2 - tc.c1_center.2)^2)
  (tc.c1_radius + tc.c2_radius > d) ∧ (d > abs (tc.c1_radius - tc.c2_radius))

/-- The main theorem --/
theorem circles_intersect (tc : TwoCircles) 
  (h1 : tc.c1_center = (-2, 2))
  (h2 : tc.c2_center = (2, 5))
  (h3 : tc.c1_radius = 2)
  (h4 : tc.c2_radius = 4)
  (h5 : Real.sqrt ((tc.c2_center.1 - tc.c1_center.1)^2 + (tc.c2_center.2 - tc.c1_center.2)^2) = 5)
  : are_intersecting tc := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l1094_109403


namespace NUMINAMATH_CALUDE_consecutive_integers_problem_l1094_109471

theorem consecutive_integers_problem (n : ℕ) (avg : ℚ) (max : ℕ) 
  (h_consecutive : ∃ (start : ℤ), ∀ i : ℕ, i < n → start + i ∈ (Set.range (fun i => start + i) : Set ℤ))
  (h_average : avg = (↑(n * (2 * max - n + 1)) / (2 * n) : ℚ))
  (h_max : max = 36)
  (h_avg : avg = 33) :
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_problem_l1094_109471


namespace NUMINAMATH_CALUDE_second_investment_value_l1094_109428

theorem second_investment_value (x : ℝ) : 
  (0.07 * 500 + 0.09 * x = 0.085 * (500 + x)) → x = 1500 := by
  sorry

end NUMINAMATH_CALUDE_second_investment_value_l1094_109428


namespace NUMINAMATH_CALUDE_lego_castle_ratio_l1094_109454

/-- Proves that the ratio of Legos used for the castle to the total number of Legos is 1:2 --/
theorem lego_castle_ratio :
  let total_legos : ℕ := 500
  let legos_put_back : ℕ := 245
  let missing_legos : ℕ := 5
  let castle_legos : ℕ := total_legos - legos_put_back - missing_legos
  (castle_legos : ℚ) / total_legos = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_lego_castle_ratio_l1094_109454


namespace NUMINAMATH_CALUDE_lottery_probability_theorem_l1094_109437

def megaBallCount : ℕ := 30
def winnerBallsTotal : ℕ := 50
def winnerBallsPicked : ℕ := 5
def bonusBallCount : ℕ := 15

def lotteryProbability : ℚ :=
  1 / (megaBallCount * (Nat.choose winnerBallsTotal winnerBallsPicked) * bonusBallCount)

theorem lottery_probability_theorem :
  lotteryProbability = 1 / 95673600 := by
  sorry

end NUMINAMATH_CALUDE_lottery_probability_theorem_l1094_109437


namespace NUMINAMATH_CALUDE_heartsuit_four_six_l1094_109420

-- Define the ♡ operation
def heartsuit (x y : ℝ) : ℝ := 5*x + 3*y

-- Theorem statement
theorem heartsuit_four_six : heartsuit 4 6 = 38 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_four_six_l1094_109420


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l1094_109424

theorem simplify_sqrt_sum : 
  Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3) = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l1094_109424


namespace NUMINAMATH_CALUDE_x_value_from_equation_l1094_109470

theorem x_value_from_equation (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 6 * x^2 + 18 * x * y = 2 * x^3 + 3 * x^2 * y^2) : 
  x = (3 + Real.sqrt 153) / 4 := by
  sorry

end NUMINAMATH_CALUDE_x_value_from_equation_l1094_109470


namespace NUMINAMATH_CALUDE_fraction_and_decimal_representation_l1094_109431

theorem fraction_and_decimal_representation :
  (7 : ℚ) / 16 = 7 / 16 ∧ (100.45 : ℝ) = 100 + 4/10 + 5/100 :=
by sorry

end NUMINAMATH_CALUDE_fraction_and_decimal_representation_l1094_109431


namespace NUMINAMATH_CALUDE_complex_sum_problem_l1094_109472

theorem complex_sum_problem (a b c d e f : ℝ) : 
  b = 1 → 
  e = -a - c → 
  (a + b * Complex.I) + (c + d * Complex.I) + (e + f * Complex.I) = -Complex.I → 
  d + f = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l1094_109472


namespace NUMINAMATH_CALUDE_same_color_probability_l1094_109407

def total_plates : ℕ := 11
def red_plates : ℕ := 6
def blue_plates : ℕ := 5
def selected_plates : ℕ := 3

theorem same_color_probability :
  (Nat.choose red_plates selected_plates : ℚ) / (Nat.choose total_plates selected_plates) = 4 / 33 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l1094_109407


namespace NUMINAMATH_CALUDE_sequence_arrangement_count_l1094_109476

theorem sequence_arrangement_count : ℕ := by
  -- Define the length of the sequence
  let n : ℕ := 6

  -- Define the counts of each number in the sequence
  let count_of_ones : ℕ := 3
  let count_of_twos : ℕ := 2
  let count_of_threes : ℕ := 1

  -- Assert that the sum of counts equals the sequence length
  have h_sum_counts : count_of_ones + count_of_twos + count_of_threes = n := by sorry

  -- Define the number of ways to arrange the sequence
  let arrangement_count : ℕ := n.choose count_of_threes * (n - count_of_threes).choose count_of_twos

  -- Prove that the arrangement count equals 60
  have h_arrangement_count : arrangement_count = 60 := by sorry

  -- Return the final result
  exact 60

end NUMINAMATH_CALUDE_sequence_arrangement_count_l1094_109476


namespace NUMINAMATH_CALUDE_worker_b_days_l1094_109441

/-- The number of days it takes for worker b to complete a job alone,
    given that worker a is twice as efficient as worker b and
    together they complete the job in 6 days. -/
theorem worker_b_days (efficiency_a : ℝ) (efficiency_b : ℝ) (days_together : ℝ) : 
  efficiency_a = 2 * efficiency_b →
  days_together = 6 →
  efficiency_a + efficiency_b = 1 / days_together →
  1 / efficiency_b = 18 := by
sorry

end NUMINAMATH_CALUDE_worker_b_days_l1094_109441


namespace NUMINAMATH_CALUDE_matrix_problem_l1094_109402

def A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -1; -4, 3]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![2, -2; -4, 6]

theorem matrix_problem :
  (∃ X : Matrix (Fin 2) (Fin 2) ℚ, A * X = B) ∧
  A⁻¹ = !![3/2, 1/2; 2, 1] ∧
  A * !![1, 0; 0, 2] = B := by sorry

end NUMINAMATH_CALUDE_matrix_problem_l1094_109402


namespace NUMINAMATH_CALUDE_jason_money_calculation_l1094_109488

/-- Represents the value of coins in cents -/
inductive Coin
  | quarter
  | dime
  | nickel

/-- The value of a coin in cents -/
def coin_value (c : Coin) : ℕ :=
  match c with
  | Coin.quarter => 25
  | Coin.dime => 10
  | Coin.nickel => 5

/-- Calculates the total value of coins in dollars -/
def coins_value (quarters dimes nickels : ℕ) : ℚ :=
  (quarters * coin_value Coin.quarter + dimes * coin_value Coin.dime + nickels * coin_value Coin.nickel) / 100

/-- Converts euros to US dollars -/
def euros_to_dollars (euros : ℚ) : ℚ :=
  euros * 1.20

theorem jason_money_calculation (initial_quarters initial_dimes initial_nickels : ℕ)
    (initial_euros : ℚ)
    (additional_quarters additional_dimes additional_nickels : ℕ)
    (additional_euros : ℚ) :
    let initial_coins := coins_value initial_quarters initial_dimes initial_nickels
    let initial_dollars := initial_coins + euros_to_dollars initial_euros
    let additional_coins := coins_value additional_quarters additional_dimes additional_nickels
    let additional_dollars := additional_coins + euros_to_dollars additional_euros
    let total_dollars := initial_dollars + additional_dollars
    initial_quarters = 49 →
    initial_dimes = 32 →
    initial_nickels = 18 →
    initial_euros = 22.50 →
    additional_quarters = 25 →
    additional_dimes = 15 →
    additional_nickels = 10 →
    additional_euros = 12 →
    total_dollars = 66 := by
  sorry

end NUMINAMATH_CALUDE_jason_money_calculation_l1094_109488


namespace NUMINAMATH_CALUDE_circle_equation_sum_l1094_109438

/-- Given a circle equation, prove the sum of center coordinates and radius -/
theorem circle_equation_sum (x y : ℝ) :
  (∀ x y, x^2 + 14*y + 65 = -y^2 - 8*x) →
  ∃ a b r : ℝ,
    (∀ x y, (x - a)^2 + (y - b)^2 = r^2) ∧
    a + b + r = -11 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_sum_l1094_109438


namespace NUMINAMATH_CALUDE_inequality_proof_l1094_109484

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b = (a + 1/a^3)/2) 
  (hc : c = (b + 1/b^3)/2) 
  (hb_lt_1 : b < 1) : 
  1 < c ∧ c < a :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1094_109484


namespace NUMINAMATH_CALUDE_probability_of_three_in_six_sevenths_l1094_109474

def decimal_representation (n d : ℕ) : List ℕ := sorry

theorem probability_of_three_in_six_sevenths : 
  let rep := decimal_representation 6 7
  ∀ k, k ∈ rep → k ≠ 3 :=
sorry

end NUMINAMATH_CALUDE_probability_of_three_in_six_sevenths_l1094_109474


namespace NUMINAMATH_CALUDE_nicole_fish_tanks_l1094_109421

/-- Represents the number of fish tanks Nicole has -/
def num_tanks : ℕ := 4

/-- Represents the amount of water (in gallons) needed for each of the first two tanks -/
def water_first_two : ℕ := 8

/-- Represents the amount of water (in gallons) needed for each of the other two tanks -/
def water_other_two : ℕ := water_first_two - 2

/-- Represents the total amount of water (in gallons) needed for all tanks in one week -/
def total_water_per_week : ℕ := 2 * water_first_two + 2 * water_other_two

/-- Represents the number of weeks -/
def num_weeks : ℕ := 4

/-- Represents the total amount of water (in gallons) needed for all tanks in four weeks -/
def total_water_four_weeks : ℕ := 112

theorem nicole_fish_tanks :
  num_tanks = 4 ∧
  water_first_two = 8 ∧
  water_other_two = water_first_two - 2 ∧
  total_water_per_week = 2 * water_first_two + 2 * water_other_two ∧
  total_water_four_weeks = num_weeks * total_water_per_week :=
by sorry

end NUMINAMATH_CALUDE_nicole_fish_tanks_l1094_109421


namespace NUMINAMATH_CALUDE_employee_pay_l1094_109436

theorem employee_pay (x y : ℝ) (h1 : x + y = 616) (h2 : x = 1.2 * y) : y = 280 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_l1094_109436


namespace NUMINAMATH_CALUDE_parabola_point_comparison_l1094_109494

/-- Theorem: For a parabola y = ax^2 - 4ax + 2 where a > 0, 
    and points (-1, y₁) and (1, y₂) on the parabola, y₁ > y₂ -/
theorem parabola_point_comparison 
  (a : ℝ) 
  (h_a_pos : a > 0) 
  (y₁ y₂ : ℝ) 
  (h_y₁ : y₁ = a * (-1)^2 - 4 * a * (-1) + 2) 
  (h_y₂ : y₂ = a * 1^2 - 4 * a * 1 + 2) : 
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_comparison_l1094_109494


namespace NUMINAMATH_CALUDE_henrys_age_l1094_109464

theorem henrys_age (h s : ℕ) : 
  h + 8 = 3 * (s - 1) →
  (h - 25) + (s - 25) = 83 →
  h = 97 :=
by sorry

end NUMINAMATH_CALUDE_henrys_age_l1094_109464


namespace NUMINAMATH_CALUDE_x_minus_p_equals_3_minus_2p_l1094_109451

theorem x_minus_p_equals_3_minus_2p (x p : ℝ) (h1 : |x - 3| = p) (h2 : x < 3) :
  x - p = 3 - 2*p := by
  sorry

end NUMINAMATH_CALUDE_x_minus_p_equals_3_minus_2p_l1094_109451


namespace NUMINAMATH_CALUDE_computer_profit_percentage_l1094_109461

theorem computer_profit_percentage (cost : ℝ) 
  (h1 : 2240 = cost * 1.4) 
  (h2 : 2400 > cost) : 
  (2400 - cost) / cost = 0.5 := by
sorry

end NUMINAMATH_CALUDE_computer_profit_percentage_l1094_109461


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l1094_109491

theorem quadratic_solution_property (p q : ℝ) : 
  (3 * p^2 + 4 * p - 7 = 0) → 
  (3 * q^2 + 4 * q - 7 = 0) → 
  (p - 2) * (q - 2) = 13/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l1094_109491


namespace NUMINAMATH_CALUDE_gcd_12345_67890_l1094_109416

theorem gcd_12345_67890 : Nat.gcd 12345 67890 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12345_67890_l1094_109416


namespace NUMINAMATH_CALUDE_not_necessarily_right_triangle_l1094_109479

theorem not_necessarily_right_triangle (A B C : ℝ) : 
  A > 0 → B > 0 → C > 0 →
  A + B + C = 180 →
  A / 3 = B / 4 →
  B / 4 = C / 5 →
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) :=
by sorry

end NUMINAMATH_CALUDE_not_necessarily_right_triangle_l1094_109479


namespace NUMINAMATH_CALUDE_smallest_integer_l1094_109429

theorem smallest_integer (m n x : ℕ) : 
  m = 72 →
  x > 0 →
  Nat.gcd m n = x + 8 →
  Nat.lcm m n = x * (x + 8) →
  n ≥ 8 ∧ (∃ (y : ℕ), y > 0 ∧ y + 8 ∣ 72 ∧ y < x → False) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_l1094_109429


namespace NUMINAMATH_CALUDE_fred_money_left_l1094_109455

/-- Calculates the amount of money Fred has left after spending half his allowance on movies and earning money from washing a car. -/
def money_left (allowance : ℕ) (car_wash_earnings : ℕ) : ℕ :=
  allowance / 2 + car_wash_earnings

/-- Proves that Fred has 14 dollars left given his allowance and car wash earnings. -/
theorem fred_money_left :
  money_left 16 6 = 14 := by
  sorry

end NUMINAMATH_CALUDE_fred_money_left_l1094_109455


namespace NUMINAMATH_CALUDE_sequence_count_16_l1094_109465

/-- Represents the number of valid sequences of length n -/
def validSequences : ℕ → ℕ
  | 0 => 0
  | 1 => 0
  | 2 => 2
  | n + 2 => 2 * validSequences n

/-- The problem statement -/
theorem sequence_count_16 : validSequences 16 = 256 := by
  sorry

end NUMINAMATH_CALUDE_sequence_count_16_l1094_109465


namespace NUMINAMATH_CALUDE_fraction_arrangement_equals_two_l1094_109415

theorem fraction_arrangement_equals_two : ∃ (f : ℚ → ℚ → ℚ → ℚ → ℚ), f (1/4) (1/4) (1/4) (1/4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_arrangement_equals_two_l1094_109415


namespace NUMINAMATH_CALUDE_no_50_cell_crossing_l1094_109486

/-- The maximum number of cells a straight line can cross on an m × n grid -/
def maxCrossedCells (m n : ℕ) : ℕ := m + n - Nat.gcd m n

/-- Theorem: On a 20 × 30 grid, it's impossible to draw a straight line that crosses 50 cells -/
theorem no_50_cell_crossing :
  maxCrossedCells 20 30 < 50 := by
  sorry

end NUMINAMATH_CALUDE_no_50_cell_crossing_l1094_109486


namespace NUMINAMATH_CALUDE_compound_ratio_l1094_109468

theorem compound_ratio (total_weight : ℝ) (weight_B : ℝ) :
  total_weight = 108 →
  weight_B = 90 →
  let weight_A := total_weight - weight_B
  (weight_A / weight_B) = (1 / 5 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_compound_ratio_l1094_109468


namespace NUMINAMATH_CALUDE_polynomial_sum_l1094_109411

-- Define the polynomials f and g
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

-- State the theorem
theorem polynomial_sum (a b c d : ℝ) : 
  (∃ (x : ℝ), f a b x = g c d x ∧ f a b x = -25 ∧ x = 50) →  -- f and g intersect at (50, -25)
  (∀ (x : ℝ), f a b x ≥ -25) →  -- minimum value of f is -25
  (∀ (x : ℝ), g c d x ≥ -25) →  -- minimum value of g is -25
  g c d (-a/2) = 0 →  -- vertex of f is root of g
  f a b (-c/2) = 0 →  -- vertex of g is root of f
  a ≠ c →  -- f and g are distinct
  a + c = -101 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_l1094_109411


namespace NUMINAMATH_CALUDE_zeros_of_specific_quadratic_range_of_a_for_distinct_zeros_l1094_109449

-- Define the quadratic function
def f (a b x : ℝ) := a * x^2 + b * x + (b - 1)

-- Part 1
theorem zeros_of_specific_quadratic :
  let f₁ := f 1 (-2)
  (f₁ 3 = 0) ∧ (f₁ (-1) = 0) ∧ (∀ x, f₁ x = 0 → x = 3 ∨ x = -1) := by sorry

-- Part 2
theorem range_of_a_for_distinct_zeros (a : ℝ) :
  (a ≠ 0) →
  (∀ b : ℝ, ∃ x y : ℝ, x ≠ y ∧ f a b x = 0 ∧ f a b y = 0) →
  (0 < a ∧ a < 1) := by sorry

end NUMINAMATH_CALUDE_zeros_of_specific_quadratic_range_of_a_for_distinct_zeros_l1094_109449


namespace NUMINAMATH_CALUDE_range_of_a_l1094_109446

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + a*x + 1 < 0) = False → a < -2 ∨ a > 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1094_109446


namespace NUMINAMATH_CALUDE_square_root_product_plus_one_l1094_109478

theorem square_root_product_plus_one : 
  Real.sqrt ((34 : ℝ) * 33 * 32 * 31 + 1) = 1055 := by sorry

end NUMINAMATH_CALUDE_square_root_product_plus_one_l1094_109478


namespace NUMINAMATH_CALUDE_poster_area_l1094_109462

theorem poster_area (x y : ℕ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : (3 * x + 4) * (y + 3) = 63) : x * y = 15 := by
  sorry

end NUMINAMATH_CALUDE_poster_area_l1094_109462


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1094_109401

theorem absolute_value_inequality (x : ℝ) :
  |2 * x + 6| < 10 ↔ -8 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1094_109401


namespace NUMINAMATH_CALUDE_partnership_annual_gain_l1094_109426

/-- Represents the annual gain of a partnership given the following conditions:
    - A invests x at the beginning of the year
    - B invests 2x after 6 months
    - C invests 3x after 8 months
    - A's share is 6200
    - Profit is divided based on investment amount and time
-/
theorem partnership_annual_gain (x : ℝ) (total_gain : ℝ) : 
  x > 0 →
  (x * 12) / (x * 12 + 2 * x * 6 + 3 * x * 4) = 6200 / total_gain →
  total_gain = 18600 := by
  sorry

#check partnership_annual_gain

end NUMINAMATH_CALUDE_partnership_annual_gain_l1094_109426


namespace NUMINAMATH_CALUDE_passing_percentage_l1094_109458

theorem passing_percentage (marks_obtained : ℕ) (marks_short : ℕ) (total_marks : ℕ) :
  marks_obtained = 125 →
  marks_short = 40 →
  total_marks = 500 →
  (((marks_obtained + marks_short : ℚ) / total_marks) * 100 : ℚ) = 33 := by
  sorry

end NUMINAMATH_CALUDE_passing_percentage_l1094_109458


namespace NUMINAMATH_CALUDE_number_of_walls_proof_correct_l1094_109473

/-- Proves that the number of walls in a room is 5, given specific conditions about wall size and painting time. -/
theorem number_of_walls (wall_width : ℝ) (wall_height : ℝ) (painting_rate : ℝ) (total_time : ℝ) (spare_time : ℝ) : ℕ :=
  by
  -- Define the given conditions
  have h1 : wall_width = 2 := by sorry
  have h2 : wall_height = 3 := by sorry
  have h3 : painting_rate = 1 / 10 := by sorry  -- 1 square meter per 10 minutes
  have h4 : total_time = 10 := by sorry  -- 10 hours
  have h5 : spare_time = 5 := by sorry  -- 5 hours

  -- Calculate the number of walls
  let wall_area := wall_width * wall_height
  let available_time := total_time - spare_time
  let paintable_area := available_time * 60 * painting_rate  -- Convert hours to minutes
  let number_of_walls := ⌊paintable_area / wall_area⌋  -- Floor division

  -- Prove that the number of walls is 5
  sorry

/-- The number of walls in the room -/
def solution : ℕ := 5

/-- Proves that the calculated number of walls matches the solution -/
theorem proof_correct : number_of_walls 2 3 (1/10) 10 5 = solution := by sorry

end NUMINAMATH_CALUDE_number_of_walls_proof_correct_l1094_109473


namespace NUMINAMATH_CALUDE_inequality_problem_l1094_109447

theorem inequality_problem (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_prod : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
sorry

end NUMINAMATH_CALUDE_inequality_problem_l1094_109447


namespace NUMINAMATH_CALUDE_regular_tetrahedron_iff_l1094_109475

/-- A tetrahedron -/
structure Tetrahedron where
  /-- The base of the tetrahedron -/
  base : Triangle
  /-- The apex of the tetrahedron -/
  apex : Point

/-- A regular tetrahedron -/
def RegularTetrahedron (t : Tetrahedron) : Prop :=
  sorry

/-- The base of the tetrahedron is an equilateral triangle -/
def HasEquilateralBase (t : Tetrahedron) : Prop :=
  sorry

/-- The dihedral angles between the lateral faces and the base are equal -/
def HasEqualDihedralAngles (t : Tetrahedron) : Prop :=
  sorry

/-- All lateral edges form equal angles with the base -/
def HasEqualLateralEdgeAngles (t : Tetrahedron) : Prop :=
  sorry

/-- Theorem: A tetrahedron is regular if and only if it satisfies certain conditions -/
theorem regular_tetrahedron_iff (t : Tetrahedron) : 
  RegularTetrahedron t ↔ 
  (HasEquilateralBase t ∧ HasEqualDihedralAngles t) ∨
  (HasEqualLateralEdgeAngles t ∧ HasEqualDihedralAngles t) :=
sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_iff_l1094_109475


namespace NUMINAMATH_CALUDE_class_size_class_size_problem_l1094_109477

theorem class_size (total_stickers : ℕ) (num_friends : ℕ) (stickers_per_friend : ℕ) 
  (stickers_per_other : ℕ) (leftover_stickers : ℕ) : ℕ :=
  let stickers_to_friends := num_friends * stickers_per_friend
  let remaining_stickers := total_stickers - stickers_to_friends - leftover_stickers
  let other_students := remaining_stickers / stickers_per_other
  other_students + num_friends + 1

theorem class_size_problem : 
  class_size 50 5 4 2 8 = 17 := by
  sorry

end NUMINAMATH_CALUDE_class_size_class_size_problem_l1094_109477


namespace NUMINAMATH_CALUDE_thompson_exam_rule_l1094_109452

-- Define the universe of discourse
variable (Student : Type)

-- Define predicates
variable (answered_all_correctly : Student → Prop)
variable (received_C_or_higher : Student → Prop)

-- State the theorem
theorem thompson_exam_rule 
  (h : ∀ s : Student, ¬(answered_all_correctly s) → ¬(received_C_or_higher s)) :
  ∀ s : Student, received_C_or_higher s → answered_all_correctly s :=
by sorry

end NUMINAMATH_CALUDE_thompson_exam_rule_l1094_109452


namespace NUMINAMATH_CALUDE_M_intersect_N_l1094_109467

def M : Set ℝ := {x | x^2 - 4 < 0}
def N : Set ℝ := {x | ∃ n : ℤ, x = 2*n + 1}

theorem M_intersect_N : M ∩ N = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_l1094_109467


namespace NUMINAMATH_CALUDE_laura_age_l1094_109466

def is_divisible_by (a b : ℕ) : Prop := a % b = 0

theorem laura_age :
  ∀ (L A : ℕ),
  is_divisible_by (L - 1) 8 →
  is_divisible_by (A - 1) 8 →
  is_divisible_by (L + 1) 7 →
  is_divisible_by (A + 1) 7 →
  A < 100 →
  L = 41 :=
by sorry

end NUMINAMATH_CALUDE_laura_age_l1094_109466


namespace NUMINAMATH_CALUDE_football_gear_cost_l1094_109434

theorem football_gear_cost (x : ℝ) 
  (h1 : x + x = 2 * x)  -- Shorts + T-shirt costs twice as much as shorts
  (h2 : x + 4 * x = 5 * x)  -- Shorts + boots costs five times as much as shorts
  (h3 : x + 2 * x = 3 * x)  -- Shorts + shin guards costs three times as much as shorts
  : x + x + 4 * x + 2 * x = 8 * x :=  -- Total cost is 8 times the cost of shorts
by sorry

end NUMINAMATH_CALUDE_football_gear_cost_l1094_109434


namespace NUMINAMATH_CALUDE_auction_bidding_l1094_109409

theorem auction_bidding (price_increase : ℕ) (start_price : ℕ) (end_price : ℕ) (num_bidders : ℕ) :
  price_increase = 5 →
  start_price = 15 →
  end_price = 65 →
  num_bidders = 2 →
  (end_price - start_price) / price_increase / num_bidders = 5 :=
by sorry

end NUMINAMATH_CALUDE_auction_bidding_l1094_109409


namespace NUMINAMATH_CALUDE_unique_m_for_power_function_l1094_109408

/-- A function f is a power function if it has the form f(x) = ax^b for some constants a and b, where a ≠ 0 -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^b

/-- A function f is increasing on (0, +∞) if for all x₁, x₂ > 0, x₁ < x₂ implies f(x₁) < f(x₂) -/
def is_increasing_on_positive_reals (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ < f x₂

/-- The main theorem -/
theorem unique_m_for_power_function :
  ∃! m : ℝ, 
    is_power_function (fun x ↦ (m^2 - m - 1) * x^m) ∧
    is_increasing_on_positive_reals (fun x ↦ (m^2 - m - 1) * x^m) ∧
    m = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_m_for_power_function_l1094_109408


namespace NUMINAMATH_CALUDE_hyperbola_properties_l1094_109400

/-- The equation of a hyperbola passing through (1, 0) with asymptotes y = ±2x -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 - y^2/4 = 1

/-- The focus of the parabola y² = 4x -/
def parabola_focus : ℝ × ℝ := (1, 0)

/-- The asymptotes of the hyperbola -/
def asymptote_pos (x y : ℝ) : Prop := y = 2 * x
def asymptote_neg (x y : ℝ) : Prop := y = -2 * x

theorem hyperbola_properties :
  (∀ x y : ℝ, hyperbola_equation x y → (asymptote_pos x y ∨ asymptote_neg x y)) ∧
  hyperbola_equation parabola_focus.1 parabola_focus.2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l1094_109400


namespace NUMINAMATH_CALUDE_sum_ratio_equals_half_l1094_109417

theorem sum_ratio_equals_half : (1 + 2 + 3 + 4 + 5) / (2 + 4 + 6 + 8 + 10) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_ratio_equals_half_l1094_109417


namespace NUMINAMATH_CALUDE_correct_answer_l1094_109432

-- Define proposition p
def p : Prop := ∀ a b c : ℝ, a < b → a * c^2 < b * c^2

-- Define proposition q
def q : Prop := ∃ x : ℝ, x^2 - x + 1 ≤ 0

-- Theorem to prove
theorem correct_answer : p ∨ (¬q) := by sorry

end NUMINAMATH_CALUDE_correct_answer_l1094_109432
