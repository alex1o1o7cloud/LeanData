import Mathlib

namespace NUMINAMATH_CALUDE_equivalence_condition_l2515_251575

theorem equivalence_condition (a b : ℝ) (h : a * b > 0) :
  (a > b) ↔ (1 / a < 1 / b) :=
by sorry

end NUMINAMATH_CALUDE_equivalence_condition_l2515_251575


namespace NUMINAMATH_CALUDE_OPRQ_shapes_l2515_251506

-- Define the points
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the quadrilateral OPRQ
structure Quadrilateral where
  O : Point2D
  P : Point2D
  R : Point2D
  Q : Point2D

-- Define the conditions for parallelogram, rectangle, and rhombus
def is_parallelogram (quad : Quadrilateral) : Prop :=
  ∃ k l : ℝ, k ≠ 0 ∧ l ≠ 0 ∧
  quad.R.x = k * quad.P.x + l * quad.Q.x ∧
  quad.R.y = k * quad.P.y + l * quad.Q.y

def is_rectangle (quad : Quadrilateral) : Prop :=
  is_parallelogram quad ∧
  quad.P.x * quad.Q.x + quad.P.y * quad.Q.y = 0

def is_rhombus (quad : Quadrilateral) : Prop :=
  is_parallelogram quad ∧
  quad.P.x^2 + quad.P.y^2 = quad.Q.x^2 + quad.Q.y^2

-- Main theorem
theorem OPRQ_shapes (P Q : Point2D) (h : P ≠ Q) :
  ∃ (R : Point2D) (quad : Quadrilateral),
    quad.O = ⟨0, 0⟩ ∧ quad.P = P ∧ quad.Q = Q ∧ quad.R = R ∧
    (is_parallelogram quad ∨ is_rectangle quad ∨ is_rhombus quad) :=
  sorry

end NUMINAMATH_CALUDE_OPRQ_shapes_l2515_251506


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2515_251520

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 3 + Real.sqrt 3 ∧ x₂ = 3 - Real.sqrt 3 ∧
    x₁^2 - 6*x₁ + 6 = 0 ∧ x₂^2 - 6*x₂ + 6 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = 5 ∧
    (x₁ - 1) * (x₁ - 3) = 8 ∧ (x₂ - 1) * (x₂ - 3) = 8) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2515_251520


namespace NUMINAMATH_CALUDE_square_pyramid_frustum_volume_fraction_l2515_251580

/-- The volume of a square pyramid frustum as a fraction of the original pyramid --/
theorem square_pyramid_frustum_volume_fraction 
  (base_edge : ℝ) 
  (altitude : ℝ) 
  (h_base : base_edge = 40) 
  (h_alt : altitude = 18) :
  let original_volume := (1/3) * base_edge^2 * altitude
  let small_base_edge := (1/5) * base_edge
  let small_altitude := (1/5) * altitude
  let small_volume := (1/3) * small_base_edge^2 * small_altitude
  let frustum_volume := original_volume - small_volume
  frustum_volume / original_volume = 2383 / 2400 := by
sorry

end NUMINAMATH_CALUDE_square_pyramid_frustum_volume_fraction_l2515_251580


namespace NUMINAMATH_CALUDE_evaluate_64_to_5_6th_power_l2515_251574

theorem evaluate_64_to_5_6th_power : (64 : ℝ) ^ (5/6) = 32 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_64_to_5_6th_power_l2515_251574


namespace NUMINAMATH_CALUDE_chord_line_equation_l2515_251569

/-- The equation of a line containing a chord of a parabola -/
theorem chord_line_equation (x y : ℝ → ℝ) :
  (∀ t : ℝ, (y t)^2 = -8 * (x t)) →  -- parabola equation
  (∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ 
    (x t₁ + x t₂) / 2 = -1 ∧ 
    (y t₁ + y t₂) / 2 = 1) →  -- midpoint condition
  ∃ a b c : ℝ, a ≠ 0 ∧ 
    (∀ t : ℝ, a * (x t) + b * (y t) + c = 0) ∧ 
    (4 * a = -b ∧ 3 * a = -c) :=  -- line equation
by sorry

end NUMINAMATH_CALUDE_chord_line_equation_l2515_251569


namespace NUMINAMATH_CALUDE_digit_sum_property_l2515_251539

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem digit_sum_property (M : ℕ) :
  (∀ k : ℕ, k > 0 → k ≤ M → S (M * k) = S M) ↔
  ∃ n : ℕ, n > 0 ∧ M = 10^n - 1 :=
sorry

end NUMINAMATH_CALUDE_digit_sum_property_l2515_251539


namespace NUMINAMATH_CALUDE_line_inclination_angle_ratio_l2515_251551

theorem line_inclination_angle_ratio (θ : Real) : 
  (2 : Real) * Real.tan θ + 1 = 0 →
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_angle_ratio_l2515_251551


namespace NUMINAMATH_CALUDE_rock_paper_scissors_wins_l2515_251545

/-- Represents the outcome of a single round --/
inductive RoundResult
| Win
| Lose
| Tie

/-- Represents a player's position and game results --/
structure PlayerState :=
  (position : Int)
  (wins : Nat)
  (losses : Nat)
  (ties : Nat)

/-- Updates a player's state based on the round result --/
def updatePlayerState (state : PlayerState) (result : RoundResult) : PlayerState :=
  match result with
  | RoundResult.Win => { state with position := state.position + 3, wins := state.wins + 1 }
  | RoundResult.Lose => { state with position := state.position - 2, losses := state.losses + 1 }
  | RoundResult.Tie => { state with position := state.position + 1, ties := state.ties + 1 }

/-- Represents the state of the game --/
structure GameState :=
  (playerA : PlayerState)
  (playerB : PlayerState)
  (rounds : Nat)

/-- Updates the game state based on the round result for Player A --/
def updateGameState (state : GameState) (result : RoundResult) : GameState :=
  { state with
    playerA := updatePlayerState state.playerA result,
    playerB := updatePlayerState state.playerB (match result with
      | RoundResult.Win => RoundResult.Lose
      | RoundResult.Lose => RoundResult.Win
      | RoundResult.Tie => RoundResult.Tie),
    rounds := state.rounds + 1 }

/-- The main theorem to prove --/
theorem rock_paper_scissors_wins
  (initialDistance : Nat)
  (totalRounds : Nat)
  (finalPositionA : Int)
  (finalPositionB : Int)
  (h1 : initialDistance = 30)
  (h2 : totalRounds = 15)
  (h3 : finalPositionA = 17)
  (h4 : finalPositionB = 2) :
  ∃ (gameResults : List RoundResult),
    let finalState := gameResults.foldl updateGameState
      { playerA := ⟨0, 0, 0, 0⟩,
        playerB := ⟨initialDistance, 0, 0, 0⟩,
        rounds := 0 }
    finalState.rounds = totalRounds ∧
    finalState.playerA.position = finalPositionA ∧
    finalState.playerB.position = finalPositionB ∧
    finalState.playerA.wins = 7 :=
sorry

end NUMINAMATH_CALUDE_rock_paper_scissors_wins_l2515_251545


namespace NUMINAMATH_CALUDE_parabola_constant_l2515_251587

theorem parabola_constant (c : ℝ) : 
  (∃ (x y : ℝ), y = x^2 - c ∧ x = 3 ∧ y = 8) → c = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_constant_l2515_251587


namespace NUMINAMATH_CALUDE_complex_number_location_l2515_251531

theorem complex_number_location (z : ℂ) (h : z * (1 + Complex.I) * (-2 * Complex.I) = z) :
  Complex.re z > 0 ∧ Complex.im z < 0 :=
sorry

end NUMINAMATH_CALUDE_complex_number_location_l2515_251531


namespace NUMINAMATH_CALUDE_function_bounds_l2515_251526

theorem function_bounds 
  (f : ℕ+ → ℕ+) 
  (h_increasing : ∀ n m : ℕ+, n < m → f n < f m) 
  (k : ℕ+) 
  (h_composition : ∀ n : ℕ+, f (f n) = k * n) :
  ∀ n : ℕ+, (2 * k * n : ℚ) / (k + 1) ≤ f n ∧ (f n : ℚ) ≤ ((k + 1) * n) / 2 :=
sorry

end NUMINAMATH_CALUDE_function_bounds_l2515_251526


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l2515_251547

theorem geometric_series_ratio (r : ℝ) (h : r ≠ 1) :
  (∃ (a : ℝ), a / (1 - r) = 64 * (a * r^4) / (1 - r)) →
  r = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l2515_251547


namespace NUMINAMATH_CALUDE_carpenter_woodblocks_l2515_251542

theorem carpenter_woodblocks (total_needed : ℕ) (current_logs : ℕ) (additional_logs : ℕ) 
  (h1 : total_needed = 80)
  (h2 : current_logs = 8)
  (h3 : additional_logs = 8) :
  (total_needed / (current_logs + additional_logs) : ℕ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_carpenter_woodblocks_l2515_251542


namespace NUMINAMATH_CALUDE_alpha_more_cost_effective_regular_l2515_251504

/-- Represents a fitness club with a monthly fee -/
structure FitnessClub where
  name : String
  monthlyFee : ℕ

/-- Calculates the yearly cost for a fitness club -/
def yearlyCost (club : FitnessClub) : ℕ :=
  club.monthlyFee * 12

/-- Calculates the cost per visit for a given number of visits -/
def costPerVisit (club : FitnessClub) (visits : ℕ) : ℚ :=
  (yearlyCost club : ℚ) / visits

/-- Represents the two attendance scenarios -/
inductive AttendancePattern
  | Regular
  | Sporadic

/-- Calculates the number of visits per year based on the attendance pattern -/
def visitsPerYear (pattern : AttendancePattern) : ℕ :=
  match pattern with
  | .Regular => 96
  | .Sporadic => 48

/-- The main theorem stating that Alpha is more cost-effective for regular attendance -/
theorem alpha_more_cost_effective_regular :
  let alpha : FitnessClub := ⟨"Alpha", 999⟩
  let beta : FitnessClub := ⟨"Beta", 1299⟩
  let regularVisits := visitsPerYear AttendancePattern.Regular
  costPerVisit alpha regularVisits < costPerVisit beta regularVisits :=
by sorry

end NUMINAMATH_CALUDE_alpha_more_cost_effective_regular_l2515_251504


namespace NUMINAMATH_CALUDE_solution_set_properties_l2515_251525

def M (k : ℝ) : Set ℝ :=
  {x : ℝ | (k^2 + 2*k - 3)*x^2 + (k + 3)*x - 1 > 0}

theorem solution_set_properties (k : ℝ) :
  (M k = ∅ → k ∈ Set.Icc (-3 : ℝ) (1/5)) ∧
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ M k = Set.Ioo a b → k ∈ Set.Ioo (1/5 : ℝ) 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_properties_l2515_251525


namespace NUMINAMATH_CALUDE_interest_group_count_l2515_251530

/-- The number of students who joined at least one interest group -/
def students_in_interest_groups (science_tech : ℕ) (speech : ℕ) (both : ℕ) : ℕ :=
  science_tech + speech - both

theorem interest_group_count : 
  students_in_interest_groups 65 35 20 = 80 := by
sorry

end NUMINAMATH_CALUDE_interest_group_count_l2515_251530


namespace NUMINAMATH_CALUDE_no_valid_m_l2515_251521

/-- The trajectory of point M -/
def trajectory (x y m : ℝ) : Prop :=
  x^2 / 4 - y^2 / (m^2 - 4) = 1 ∧ x ≥ 2

/-- Line L -/
def line_L (x y : ℝ) : Prop :=
  y = (1/2) * x - 3

/-- Intersection points of trajectory and line L -/
def intersection_points (m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    trajectory x₁ y₁ m ∧ trajectory x₂ y₂ m ∧
    line_L x₁ y₁ ∧ line_L x₂ y₂

/-- Vector dot product condition -/
def dot_product_condition (m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    trajectory x₁ y₁ m ∧ trajectory x₂ y₂ m ∧
    line_L x₁ y₁ ∧ line_L x₂ y₂ ∧
    (x₁ * x₂ + (y₁ - 1) * (y₂ - 1) = 9/2)

theorem no_valid_m :
  ¬∃ m : ℝ, m > 2 ∧ intersection_points m ∧ dot_product_condition m :=
sorry

end NUMINAMATH_CALUDE_no_valid_m_l2515_251521


namespace NUMINAMATH_CALUDE_profit_maximized_at_optimal_price_l2515_251502

/-- The profit function for a product with a cost of 30 yuan per item,
    where x is the selling price and (200 - x) is the quantity sold. -/
def profit_function (x : ℝ) : ℝ := -x^2 + 230*x - 6000

/-- The selling price that maximizes the profit. -/
def optimal_price : ℝ := 115

theorem profit_maximized_at_optimal_price :
  ∀ x : ℝ, profit_function x ≤ profit_function optimal_price :=
sorry

end NUMINAMATH_CALUDE_profit_maximized_at_optimal_price_l2515_251502


namespace NUMINAMATH_CALUDE_building_height_calculation_l2515_251553

/-- Given a building and a pole, calculate the height of the building using similar triangles. -/
theorem building_height_calculation (building_shadow : ℝ) (pole_height : ℝ) (pole_shadow : ℝ)
  (h_building_shadow : building_shadow = 20)
  (h_pole_height : pole_height = 2)
  (h_pole_shadow : pole_shadow = 3) :
  (pole_height / pole_shadow) * building_shadow = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_building_height_calculation_l2515_251553


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_sum_l2515_251523

theorem partial_fraction_decomposition_sum (p q r A B C : ℝ) : 
  (p ≠ q ∧ p ≠ r ∧ q ≠ r) →
  (∀ (x : ℝ), x^3 - 16*x^2 + 72*x - 27 = (x - p) * (x - q) * (x - r)) →
  (∀ (s : ℝ), s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 16*s^2 + 72*s - 27) = A / (s - p) + B / (s - q) + C / (s - r)) →
  A + B + C = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_sum_l2515_251523


namespace NUMINAMATH_CALUDE_greatest_integer_radius_of_circle_l2515_251585

theorem greatest_integer_radius_of_circle (r : ℝ) : 
  (π * r^2 < 100 * π) → (∀ n : ℕ, n > 9 → π * (n : ℝ)^2 ≥ 100 * π) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_of_circle_l2515_251585


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l2515_251560

-- Define the hyperbola equation
def is_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (|k| - 2) + y^2 / (5 - k) = 1

-- Define the range of k
def k_range (k : ℝ) : Prop :=
  (-2 < k ∧ k < 2) ∨ k > 5

-- Theorem stating the relationship between the hyperbola equation and the range of k
theorem hyperbola_k_range :
  ∀ k : ℝ, is_hyperbola k ↔ k_range k :=
sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l2515_251560


namespace NUMINAMATH_CALUDE_cary_earnings_l2515_251534

/-- Calculates the total net earnings over three years for an employee named Cary --/
def total_net_earnings (initial_wage : ℚ) : ℚ :=
  let year1_base_wage := initial_wage
  let year1_hours := 40 * 50
  let year1_gross := year1_hours * year1_base_wage + 500
  let year1_net := year1_gross * (1 - 0.2)

  let year2_base_wage := year1_base_wage * 1.2 * 0.75
  let year2_regular_hours := 40 * 51
  let year2_overtime_hours := 10 * 51
  let year2_gross := year2_regular_hours * year2_base_wage + 
                     year2_overtime_hours * (year2_base_wage * 1.5) - 300
  let year2_net := year2_gross * (1 - 0.22)

  let year3_base_wage := year2_base_wage * 1.1
  let year3_hours := 40 * 50
  let year3_gross := year3_hours * year3_base_wage + 1000
  let year3_net := year3_gross * (1 - 0.18)

  year1_net + year2_net + year3_net

/-- Theorem stating that Cary's total net earnings over three years equals $52,913.10 --/
theorem cary_earnings : total_net_earnings 10 = 52913.1 := by
  sorry

end NUMINAMATH_CALUDE_cary_earnings_l2515_251534


namespace NUMINAMATH_CALUDE_square_position_after_2023_transformations_l2515_251536

-- Define a square as a list of four vertices
def Square := List Char

-- Define the transformations
def rotate90CW (s : Square) : Square :=
  match s with
  | [a, b, c, d] => [d, a, b, c]
  | _ => s

def reflectVertical (s : Square) : Square :=
  match s with
  | [a, b, c, d] => [c, b, a, d]
  | _ => s

def rotate180 (s : Square) : Square :=
  match s with
  | [a, b, c, d] => [c, d, a, b]
  | _ => s

-- Define the sequence of transformations
def transform (s : Square) (n : Nat) : Square :=
  match n % 3 with
  | 0 => rotate180 s
  | 1 => rotate90CW s
  | _ => reflectVertical s

-- Main theorem
theorem square_position_after_2023_transformations (initial : Square) :
  initial = ['A', 'B', 'C', 'D'] →
  (transform initial 2023) = ['C', 'B', 'A', 'D'] := by
  sorry


end NUMINAMATH_CALUDE_square_position_after_2023_transformations_l2515_251536


namespace NUMINAMATH_CALUDE_olympic_torch_relay_schemes_l2515_251588

/-- The number of segments in the Olympic torch relay -/
def num_segments : ℕ := 6

/-- The number of torchbearers -/
def num_torchbearers : ℕ := 6

/-- The number of choices for the first torchbearer -/
def first_choices : ℕ := 3

/-- The number of choices for the last torchbearer -/
def last_choices : ℕ := 2

/-- The number of choices for each middle segment -/
def middle_choices : ℕ := num_torchbearers

/-- The number of middle segments -/
def num_middle_segments : ℕ := num_segments - 2

/-- The total number of different relay schemes -/
def total_schemes : ℕ := first_choices * (middle_choices ^ num_middle_segments) * last_choices

theorem olympic_torch_relay_schemes :
  total_schemes = 7776 := by
  sorry

end NUMINAMATH_CALUDE_olympic_torch_relay_schemes_l2515_251588


namespace NUMINAMATH_CALUDE_lindas_savings_l2515_251508

theorem lindas_savings (furniture_fraction : Real) (tv_cost : Real) 
  (refrigerator_percent : Real) (furniture_discount : Real) (tv_tax : Real) :
  furniture_fraction = 3/4 →
  tv_cost = 210 →
  refrigerator_percent = 20/100 →
  furniture_discount = 7/100 →
  tv_tax = 6/100 →
  ∃ (savings : Real),
    savings = 1898.40 ∧
    (furniture_fraction * savings * (1 - furniture_discount) + 
     tv_cost * (1 + tv_tax) + 
     tv_cost * (1 + refrigerator_percent)) = savings :=
by
  sorry


end NUMINAMATH_CALUDE_lindas_savings_l2515_251508


namespace NUMINAMATH_CALUDE_octagon_area_in_square_l2515_251573

/-- The area of a regular octagon inscribed in a square -/
theorem octagon_area_in_square (s : ℝ) (h : s = 4 + 2 * Real.sqrt 2) :
  let octagon_area := s^2 - 8
  octagon_area = 16 + 16 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_octagon_area_in_square_l2515_251573


namespace NUMINAMATH_CALUDE_sum_and_count_theorem_l2515_251512

def sum_of_range (a b : ℕ) : ℕ := ((b - a + 1) * (a + b)) / 2

def count_even_in_range (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_theorem : 
  sum_of_range 40 60 + count_even_in_range 40 60 = 1061 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_count_theorem_l2515_251512


namespace NUMINAMATH_CALUDE_factorization_perfect_square_factorization_difference_of_cubes_l2515_251509

/-- Proves the factorization of a^2 + 2a + 1 -/
theorem factorization_perfect_square (a : ℝ) : a^2 + 2*a + 1 = (a + 1)^2 := by
  sorry

/-- Proves the factorization of a^3 - ab^2 -/
theorem factorization_difference_of_cubes (a b : ℝ) : a^3 - a*b^2 = a*(a + b)*(a - b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_perfect_square_factorization_difference_of_cubes_l2515_251509


namespace NUMINAMATH_CALUDE_tangent_line_equation_max_integer_k_l2515_251559

noncomputable section

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := (Real.log x + k) / Real.exp x

-- Define the derivative of f
def f_derivative (k : ℝ) (x : ℝ) : ℝ := (1 - k*x - x * Real.log x) / (x * Real.exp x)

-- Theorem for the tangent line equation
theorem tangent_line_equation (x y : ℝ) :
  k = 2 →
  y = f 2 x →
  x = 1 →
  (x + Real.exp y - 3 = 0) :=
sorry

-- Theorem for the maximum integer value of k
theorem max_integer_k (k : ℤ) :
  (∀ x > 1, x * Real.exp x * f_derivative k x + (2 * ↑k - 1) * x < 1 + ↑k) →
  k ≤ 3 :=
sorry

end

end NUMINAMATH_CALUDE_tangent_line_equation_max_integer_k_l2515_251559


namespace NUMINAMATH_CALUDE_quadratic_points_range_l2515_251598

/-- Given a quadratic function f(x) = -x^2 - 2x + 3, prove that if (a, m) and (a+2, n) are points on the graph of f, and m ≥ n, then a ≥ -2. -/
theorem quadratic_points_range (a m n : ℝ) : 
  (m = -a^2 - 2*a + 3) → 
  (n = -(a+2)^2 - 2*(a+2) + 3) → 
  (m ≥ n) → 
  (a ≥ -2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_points_range_l2515_251598


namespace NUMINAMATH_CALUDE_equation_solutions_l2515_251555

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(1, 2, 26), (1, 8, 8), (2, 2, 19), (2, 4, 12), (2, 5, 10), (4, 4, 8)}

def satisfies_equation (triple : ℕ × ℕ × ℕ) : Prop :=
  let (x, y, z) := triple
  x * y + y * z + z * x = 80 ∧ x ≤ y ∧ y ≤ z

theorem equation_solutions :
  ∀ (x y z : ℕ), satisfies_equation (x, y, z) ↔ (x, y, z) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2515_251555


namespace NUMINAMATH_CALUDE_problem_solution_l2515_251529

theorem problem_solution (m n : ℝ) (h : |m - n - 5| + (2*m + n - 4)^2 = 0) : 
  3*m + n = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2515_251529


namespace NUMINAMATH_CALUDE_product_of_solutions_abs_value_equation_l2515_251586

theorem product_of_solutions_abs_value_equation :
  ∃ (x₁ x₂ : ℝ), (|x₁| = 3 * (|x₁| - 2) ∧ |x₂| = 3 * (|x₂| - 2) ∧ x₁ ≠ x₂) ∧ x₁ * x₂ = -9 :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_abs_value_equation_l2515_251586


namespace NUMINAMATH_CALUDE_min_value_expression_l2515_251537

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (heq : a * b = 1 / 2) :
  (4 * a^2 + b^2 + 3) / (2 * a - b) ≥ 2 * Real.sqrt 5 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ > b₀ ∧ a₀ * b₀ = 1 / 2 ∧
    (4 * a₀^2 + b₀^2 + 3) / (2 * a₀ - b₀) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2515_251537


namespace NUMINAMATH_CALUDE_clothing_factory_production_adjustment_l2515_251596

/-- Represents the scenario of a clothing factory adjusting its production rate -/
theorem clothing_factory_production_adjustment 
  (total_pieces : ℕ) 
  (original_rate : ℕ) 
  (days_earlier : ℕ) 
  (x : ℝ) 
  (h1 : total_pieces = 720)
  (h2 : original_rate = 48)
  (h3 : days_earlier = 5) :
  (total_pieces : ℝ) / original_rate - total_pieces / (x + original_rate) = days_earlier :=
by sorry

end NUMINAMATH_CALUDE_clothing_factory_production_adjustment_l2515_251596


namespace NUMINAMATH_CALUDE_even_function_implies_m_equals_two_l2515_251558

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = (m-1)x^2 + (m-2)x + (m^2 - 7m + 12) -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  (m - 1) * x^2 + (m - 2) * x + (m^2 - 7*m + 12)

theorem even_function_implies_m_equals_two :
  ∀ m : ℝ, IsEven (f m) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_m_equals_two_l2515_251558


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l2515_251594

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := circle_radius / 4
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_proof (h1 : square_area = 784) (h2 : rectangle_breadth = 5) :
  rectangle_area square_area rectangle_breadth = 35 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l2515_251594


namespace NUMINAMATH_CALUDE_bed_weight_difference_bed_weight_difference_proof_l2515_251571

theorem bed_weight_difference : ℝ → ℝ → Prop :=
  fun single_bed_weight double_bed_weight =>
    (5 * single_bed_weight = 50) →
    (2 * single_bed_weight + 4 * double_bed_weight = 100) →
    (double_bed_weight - single_bed_weight = 10)

-- The proof is omitted
theorem bed_weight_difference_proof : ∃ (s d : ℝ), bed_weight_difference s d :=
  sorry

end NUMINAMATH_CALUDE_bed_weight_difference_bed_weight_difference_proof_l2515_251571


namespace NUMINAMATH_CALUDE_not_perfect_square_l2515_251505

theorem not_perfect_square (n : ℕ+) : ¬∃ (m : ℕ), 4 * n^2 + 4 * n + 4 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l2515_251505


namespace NUMINAMATH_CALUDE_binomial_12_10_equals_66_l2515_251517

theorem binomial_12_10_equals_66 : Nat.choose 12 10 = 66 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_10_equals_66_l2515_251517


namespace NUMINAMATH_CALUDE_cyclic_pentagon_area_diagonal_ratio_l2515_251566

/-- A cyclic pentagon is a pentagon inscribed in a circle -/
structure CyclicPentagon where
  vertices : Fin 5 → ℝ × ℝ
  center : ℝ × ℝ
  radius : ℝ
  is_cyclic : ∀ i : Fin 5, dist (vertices i) center = radius

/-- The area of a cyclic pentagon -/
def area (p : CyclicPentagon) : ℝ := sorry

/-- The sum of the diagonals of a cyclic pentagon -/
def sum_diagonals (p : CyclicPentagon) : ℝ := sorry

/-- The theorem stating that the ratio of a cyclic pentagon's area to the sum of its diagonals
    is not greater than a quarter of its circumradius -/
theorem cyclic_pentagon_area_diagonal_ratio (p : CyclicPentagon) :
  area p / sum_diagonals p ≤ p.radius / 4 := by sorry

end NUMINAMATH_CALUDE_cyclic_pentagon_area_diagonal_ratio_l2515_251566


namespace NUMINAMATH_CALUDE_bakery_items_l2515_251579

theorem bakery_items (total : ℕ) (bread_rolls : ℕ) (bagels : ℕ) (croissants : ℕ)
  (h1 : total = 90)
  (h2 : bread_rolls = 49)
  (h3 : bagels = 22)
  (h4 : total = bread_rolls + croissants + bagels) :
  croissants = 19 := by
sorry

end NUMINAMATH_CALUDE_bakery_items_l2515_251579


namespace NUMINAMATH_CALUDE_first_class_product_rate_l2515_251543

/-- Given a product with a pass rate and a rate of first-class products among qualified products,
    prove that the overall rate of first-class products is their product. -/
theorem first_class_product_rate
  (pass_rate : ℝ)
  (first_class_rate_among_qualified : ℝ)
  (h1 : 0 ≤ pass_rate ∧ pass_rate ≤ 1)
  (h2 : 0 ≤ first_class_rate_among_qualified ∧ first_class_rate_among_qualified ≤ 1) :
  pass_rate * first_class_rate_among_qualified =
  pass_rate * first_class_rate_among_qualified :=
by sorry

end NUMINAMATH_CALUDE_first_class_product_rate_l2515_251543


namespace NUMINAMATH_CALUDE_situp_competition_result_l2515_251515

/-- Adam's sit-up performance -/
def adam_situps (round : ℕ) : ℕ :=
  40 - 8 * (round - 1)

/-- Barney's sit-up performance -/
def barney_situps : ℕ := 45

/-- Carrie's sit-up performance -/
def carrie_situps : ℕ := 2 * barney_situps

/-- Jerrie's sit-up performance -/
def jerrie_situps : ℕ := carrie_situps + 5

/-- Total sit-ups for Adam -/
def adam_total : ℕ :=
  (adam_situps 1) + (adam_situps 2) + (adam_situps 3)

/-- Total sit-ups for Barney -/
def barney_total : ℕ := barney_situps * 5

/-- Total sit-ups for Carrie -/
def carrie_total : ℕ := carrie_situps * 4

/-- Total sit-ups for Jerrie -/
def jerrie_total : ℕ := jerrie_situps * 6

/-- The combined total of sit-ups -/
def combined_total : ℕ :=
  adam_total + barney_total + carrie_total + jerrie_total

theorem situp_competition_result :
  combined_total = 1251 := by
  sorry

end NUMINAMATH_CALUDE_situp_competition_result_l2515_251515


namespace NUMINAMATH_CALUDE_koshchey_chest_count_l2515_251591

/-- Represents the number of chests Koshchey has -/
structure KoshcheyChests where
  large : ℕ
  medium : ℕ
  small : ℕ
  empty : ℕ

/-- The total number of chests Koshchey has -/
def total_chests (k : KoshcheyChests) : ℕ :=
  k.large + k.medium + k.small

/-- Koshchey's chest configuration satisfies the problem conditions -/
def is_valid_configuration (k : KoshcheyChests) : Prop :=
  k.large = 11 ∧
  k.empty = 102 ∧
  ∃ (x : ℕ), x ≤ k.large ∧ k.medium = 8 * x

theorem koshchey_chest_count (k : KoshcheyChests) 
  (h : is_valid_configuration k) : total_chests k = 115 :=
by sorry

end NUMINAMATH_CALUDE_koshchey_chest_count_l2515_251591


namespace NUMINAMATH_CALUDE_second_class_size_l2515_251567

theorem second_class_size (students1 : ℕ) (avg1 : ℚ) (avg2 : ℚ) (avg_total : ℚ) :
  students1 = 25 →
  avg1 = 40 →
  avg2 = 60 →
  avg_total = 50.90909090909091 →
  ∃ students2 : ℕ, 
    students2 = 30 ∧
    (students1 * avg1 + students2 * avg2) / (students1 + students2 : ℚ) = avg_total :=
by sorry

end NUMINAMATH_CALUDE_second_class_size_l2515_251567


namespace NUMINAMATH_CALUDE_total_sales_proof_l2515_251548

def window_screen_sales (march_sales : ℕ) : ℕ :=
  let february_sales := march_sales / 4
  let january_sales := february_sales / 2
  january_sales + february_sales + march_sales

theorem total_sales_proof (march_sales : ℕ) (h : march_sales = 8800) :
  window_screen_sales march_sales = 12100 := by
  sorry

end NUMINAMATH_CALUDE_total_sales_proof_l2515_251548


namespace NUMINAMATH_CALUDE_digit_mean_is_four_point_five_l2515_251562

/-- The period length of the repeating decimal expansion of 1/(98^2) -/
def period_length : ℕ := 9604

/-- The sum of all digits in one period of the repeating decimal expansion of 1/(98^2) -/
def digit_sum : ℕ := 432180

/-- The mean of the digits in one complete period of the repeating decimal expansion of 1/(98^2) -/
def digit_mean : ℚ := digit_sum / period_length

theorem digit_mean_is_four_point_five :
  digit_mean = 4.5 := by sorry

end NUMINAMATH_CALUDE_digit_mean_is_four_point_five_l2515_251562


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l2515_251549

theorem more_girls_than_boys (total_students : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_students = 650 → boys = 272 → girls = total_students - boys → 
  girls - boys = 106 := by sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l2515_251549


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l2515_251532

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- An ellipse with axes parallel to coordinate axes -/
structure Ellipse where
  center : Point
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis

/-- Check if a point lies on an ellipse -/
def onEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - e.center.x)^2 / e.a^2 + (p.y - e.center.y)^2 / e.b^2 = 1

theorem ellipse_major_axis_length :
  let p1 : Point := ⟨0, 0⟩
  let p2 : Point := ⟨2, 2⟩
  let p3 : Point := ⟨-2, 2⟩
  let p4 : Point := ⟨4, 0⟩
  let p5 : Point := ⟨4, 4⟩
  ∃ (e : Ellipse),
    (¬ collinear p1 p2 p3 ∧ ¬ collinear p1 p2 p4 ∧ ¬ collinear p1 p2 p5 ∧
     ¬ collinear p1 p3 p4 ∧ ¬ collinear p1 p3 p5 ∧ ¬ collinear p1 p4 p5 ∧
     ¬ collinear p2 p3 p4 ∧ ¬ collinear p2 p3 p5 ∧ ¬ collinear p2 p4 p5 ∧
     ¬ collinear p3 p4 p5) →
    (onEllipse p1 e ∧ onEllipse p2 e ∧ onEllipse p3 e ∧ onEllipse p4 e ∧ onEllipse p5 e) →
    2 * e.a = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ellipse_major_axis_length_l2515_251532


namespace NUMINAMATH_CALUDE_angle_C_measure_l2515_251589

/-- Given a triangle ABC where sin²A - sin²C = (sin A - sin B) sin B, prove that the measure of angle C is π/3 -/
theorem angle_C_measure (A B C : ℝ) (hABC : A + B + C = π) 
  (h : Real.sin A ^ 2 - Real.sin C ^ 2 = (Real.sin A - Real.sin B) * Real.sin B) : 
  C = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l2515_251589


namespace NUMINAMATH_CALUDE_positive_integer_solution_for_exponential_equation_l2515_251524

theorem positive_integer_solution_for_exponential_equation :
  ∀ (n a b c : ℕ), 
    n > 1 → a > 0 → b > 0 → c > 0 →
    n^a + n^b = n^c →
    (n = 2 ∧ b = a ∧ c = a + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_positive_integer_solution_for_exponential_equation_l2515_251524


namespace NUMINAMATH_CALUDE_min_sum_squares_given_cubic_constraint_l2515_251592

/-- Given real numbers x, y, and z satisfying x^3 + y^3 + z^3 - 3xyz = 1,
    the sum of their squares x^2 + y^2 + z^2 is always greater than or equal to 1 -/
theorem min_sum_squares_given_cubic_constraint (x y z : ℝ) 
    (h : x^3 + y^3 + z^3 - 3*x*y*z = 1) : 
    x^2 + y^2 + z^2 ≥ 1 := by
  sorry

#check min_sum_squares_given_cubic_constraint

end NUMINAMATH_CALUDE_min_sum_squares_given_cubic_constraint_l2515_251592


namespace NUMINAMATH_CALUDE_division_practice_time_l2515_251599

-- Define the given conditions
def total_training_time : ℕ := 5 * 60  -- 5 hours in minutes
def training_days : ℕ := 10
def daily_multiplication_time : ℕ := 10

-- Define the theorem
theorem division_practice_time :
  (total_training_time - training_days * daily_multiplication_time) / training_days = 20 := by
  sorry

end NUMINAMATH_CALUDE_division_practice_time_l2515_251599


namespace NUMINAMATH_CALUDE_point_M_in_first_quadrant_l2515_251500

/-- If point P(0,m) lies on the negative half-axis of the y-axis, 
    then point M(-m,-m+1) lies in the first quadrant. -/
theorem point_M_in_first_quadrant (m : ℝ) : 
  m < 0 → -m > 0 ∧ -m + 1 > 0 := by sorry

end NUMINAMATH_CALUDE_point_M_in_first_quadrant_l2515_251500


namespace NUMINAMATH_CALUDE_white_bellied_minnows_count_l2515_251564

/-- Proves the number of white-bellied minnows in a pond given the percentages of red, green, and white-bellied minnows and the number of red-bellied minnows. -/
theorem white_bellied_minnows_count 
  (red_percent : ℚ) 
  (green_percent : ℚ) 
  (red_count : ℕ) 
  (h_red_percent : red_percent = 40 / 100)
  (h_green_percent : green_percent = 30 / 100)
  (h_red_count : red_count = 20)
  : ∃ (total : ℕ) (white_count : ℕ),
    red_percent * total = red_count ∧
    (1 - red_percent - green_percent) * total = white_count ∧
    white_count = 15 := by
  sorry

#check white_bellied_minnows_count

end NUMINAMATH_CALUDE_white_bellied_minnows_count_l2515_251564


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2515_251565

theorem simplify_and_evaluate : 
  let x : ℝ := 1
  let y : ℝ := -2
  7 * x * y - 2 * (5 * x * y - 2 * x^2 * y) + 3 * x * y = -8 := by
sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2515_251565


namespace NUMINAMATH_CALUDE_remainder_r17_plus_1_div_r_plus_1_l2515_251528

theorem remainder_r17_plus_1_div_r_plus_1 (r : ℤ) : (r^17 + 1) % (r + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_r17_plus_1_div_r_plus_1_l2515_251528


namespace NUMINAMATH_CALUDE_average_first_100_odd_numbers_l2515_251563

theorem average_first_100_odd_numbers : 
  let n := 100
  let nth_odd (k : ℕ) := 2 * k - 1
  let first_odd := nth_odd 1
  let last_odd := nth_odd n
  let sum := (n / 2) * (first_odd + last_odd)
  sum / n = 100 := by
sorry

end NUMINAMATH_CALUDE_average_first_100_odd_numbers_l2515_251563


namespace NUMINAMATH_CALUDE_transmission_time_is_three_minutes_l2515_251576

/-- The number of blocks to be sent -/
def num_blocks : ℕ := 150

/-- The number of chunks per block -/
def chunks_per_block : ℕ := 256

/-- The transmission rate in chunks per second -/
def transmission_rate : ℕ := 200

/-- The time it takes to send all blocks in minutes -/
def transmission_time : ℚ :=
  (num_blocks * chunks_per_block : ℚ) / transmission_rate / 60

theorem transmission_time_is_three_minutes :
  transmission_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_transmission_time_is_three_minutes_l2515_251576


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2515_251516

/-- Computes the annual interest rate given the principal, time, compounding frequency, and final amount -/
def calculate_interest_rate (principal : ℝ) (time : ℝ) (compounding_frequency : ℕ) (final_amount : ℝ) : ℝ :=
  sorry

theorem interest_rate_calculation (principal : ℝ) (time : ℝ) (compounding_frequency : ℕ) (final_amount : ℝ) 
  (h1 : principal = 6000)
  (h2 : time = 1.5)
  (h3 : compounding_frequency = 2)
  (h4 : final_amount = 6000 + 945.75) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |calculate_interest_rate principal time compounding_frequency final_amount - 0.099| < ε :=
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l2515_251516


namespace NUMINAMATH_CALUDE_lottery_ratio_l2515_251597

def lottery_problem (lottery_winnings : ℕ) (savings : ℕ) (fun_money : ℕ) : Prop :=
  let taxes := lottery_winnings / 2
  let after_taxes := lottery_winnings - taxes
  let investment := savings / 5
  let student_loans := after_taxes - (savings + investment + fun_money)
  (lottery_winnings = 12006 ∧ savings = 1000 ∧ fun_money = 2802) →
  (student_loans : ℚ) / after_taxes = 1 / 3

theorem lottery_ratio : 
  lottery_problem 12006 1000 2802 :=
sorry

end NUMINAMATH_CALUDE_lottery_ratio_l2515_251597


namespace NUMINAMATH_CALUDE_zero_in_interval_one_two_l2515_251552

noncomputable def f (x : ℝ) := Real.exp x + 2 * x - 6

theorem zero_in_interval_one_two :
  ∃ z ∈ Set.Ioo 1 2, f z = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_one_two_l2515_251552


namespace NUMINAMATH_CALUDE_factorial_plus_twelve_square_l2515_251572

theorem factorial_plus_twelve_square (m n : ℕ) : m.factorial + 12 = n^2 ↔ m = 4 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_factorial_plus_twelve_square_l2515_251572


namespace NUMINAMATH_CALUDE_shorter_side_length_l2515_251544

-- Define the rectangle
def Rectangle (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a ≥ b

-- Theorem statement
theorem shorter_side_length (a b : ℝ) 
  (h_rect : Rectangle a b) 
  (h_perim : 2 * a + 2 * b = 62) 
  (h_area : a * b = 240) : 
  b = 15 := by
  sorry

end NUMINAMATH_CALUDE_shorter_side_length_l2515_251544


namespace NUMINAMATH_CALUDE_tv_screen_area_difference_l2515_251593

theorem tv_screen_area_difference : 
  let diagonal_large : ℝ := 22
  let diagonal_small : ℝ := 20
  let area_large := diagonal_large ^ 2
  let area_small := diagonal_small ^ 2
  area_large - area_small = 84 := by
  sorry

end NUMINAMATH_CALUDE_tv_screen_area_difference_l2515_251593


namespace NUMINAMATH_CALUDE_calendar_date_theorem_l2515_251557

/-- Represents a monthly calendar with dates behind letters --/
structure MonthlyCalendar where
  C : ℤ  -- Date behind C
  A : ℤ  -- Date behind A
  B : ℤ  -- Date behind B
  Q : ℤ  -- Date behind Q

/-- Theorem: The difference between dates behind C and Q equals the sum of dates behind A and B --/
theorem calendar_date_theorem (cal : MonthlyCalendar) 
  (hC : cal.C = x)
  (hA : cal.A = x + 2)
  (hB : cal.B = x + 14)
  (hQ : cal.Q = -x - 16)
  : cal.C - cal.Q = cal.A + cal.B :=
by sorry

end NUMINAMATH_CALUDE_calendar_date_theorem_l2515_251557


namespace NUMINAMATH_CALUDE_fly_path_length_l2515_251514

theorem fly_path_length (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c = 5) 
  (h4 : a^2 + b^2 = c^2) : ∃ (path_length : ℝ), path_length > 10 ∧ 
  path_length = 5 * c := by sorry

end NUMINAMATH_CALUDE_fly_path_length_l2515_251514


namespace NUMINAMATH_CALUDE_factorization_equivalence_l2515_251554

theorem factorization_equivalence (x y : ℝ) : -(2*x - y) * (2*x + y) = -4*x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equivalence_l2515_251554


namespace NUMINAMATH_CALUDE_initial_puppies_count_l2515_251503

/-- The number of puppies Alyssa gave to her friends -/
def puppies_given : ℕ := 7

/-- The number of puppies Alyssa has now -/
def puppies_remaining : ℕ := 5

/-- The initial number of puppies Alyssa had -/
def initial_puppies : ℕ := puppies_given + puppies_remaining

theorem initial_puppies_count : initial_puppies = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_puppies_count_l2515_251503


namespace NUMINAMATH_CALUDE_investment_percentage_proof_l2515_251507

theorem investment_percentage_proof (total_investment : ℝ) (first_investment : ℝ) 
  (second_investment : ℝ) (second_rate : ℝ) (third_rate : ℝ) (yearly_income : ℝ) :
  total_investment = 10000 ∧ 
  first_investment = 4000 ∧ 
  second_investment = 3500 ∧ 
  second_rate = 0.04 ∧ 
  third_rate = 0.064 ∧ 
  yearly_income = 500 →
  ∃ x : ℝ, 
    x = 5 ∧ 
    first_investment * (x / 100) + second_investment * second_rate + 
    (total_investment - first_investment - second_investment) * third_rate = yearly_income :=
by sorry

end NUMINAMATH_CALUDE_investment_percentage_proof_l2515_251507


namespace NUMINAMATH_CALUDE_people_sitting_on_benches_l2515_251501

theorem people_sitting_on_benches (num_benches : ℕ) (bench_capacity : ℕ) (available_spaces : ℕ) : 
  num_benches = 50 → bench_capacity = 4 → available_spaces = 120 → 
  num_benches * bench_capacity - available_spaces = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_people_sitting_on_benches_l2515_251501


namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l2515_251561

/-- Represents a digit in a given base -/
def Digit (d : ℕ) := {n : ℕ // n < d}

/-- Converts a two-digit number in base d to its decimal representation -/
def toDecimal (d : ℕ) (tens : Digit d) (ones : Digit d) : ℕ :=
  d * tens.val + ones.val

theorem digit_difference_in_base_d 
  (d : ℕ) (hd : d > 8) 
  (C : Digit d) (D : Digit d) 
  (h : toDecimal d C D + toDecimal d C C = d * d + 5 * d + 3) :
  C.val - D.val = 1 := by
sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l2515_251561


namespace NUMINAMATH_CALUDE_probability_one_red_one_green_l2515_251538

def total_marbles : ℕ := 4 + 6 + 11

def prob_red_then_green : ℚ :=
  (4 : ℚ) / total_marbles * 6 / (total_marbles - 1)

def prob_green_then_red : ℚ :=
  (6 : ℚ) / total_marbles * 4 / (total_marbles - 1)

theorem probability_one_red_one_green :
  prob_red_then_green + prob_green_then_red = 4 / 35 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_red_one_green_l2515_251538


namespace NUMINAMATH_CALUDE_point_in_same_region_l2515_251590

/-- The line equation -/
def line_equation (x y : ℝ) : ℝ := 3*x + 2*y + 5

/-- Definition of being in the same region -/
def same_region (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (line_equation x₁ y₁ > 0 ∧ line_equation x₂ y₂ > 0) ∨
  (line_equation x₁ y₁ < 0 ∧ line_equation x₂ y₂ < 0)

/-- Theorem stating that (-3,4) is in the same region as (0,0) -/
theorem point_in_same_region : same_region (-3) 4 0 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_same_region_l2515_251590


namespace NUMINAMATH_CALUDE_complex_multiplication_l2515_251568

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 + i) = -1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2515_251568


namespace NUMINAMATH_CALUDE_linear_congruence_intercepts_l2515_251533

/-- Proves the properties of x-intercept and y-intercept for the linear congruence equation 5x ≡ 3y + 2 (mod 27) -/
theorem linear_congruence_intercepts :
  ∃ (x₀ y₀ : ℕ),
    x₀ < 27 ∧
    y₀ < 27 ∧
    (5 * x₀) % 27 = 2 ∧
    (3 * y₀) % 27 = 25 ∧
    x₀ + y₀ = 40 := by
  sorry

end NUMINAMATH_CALUDE_linear_congruence_intercepts_l2515_251533


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2515_251550

/-- The function f(x) = x^2 - 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- The derivative of f(x) -/
def f_deriv (x : ℝ) : ℝ := 2 * x

/-- The point of tangency -/
def tangent_point : ℝ := 1

/-- The slope of the tangent line at x = 1 -/
def tangent_slope : ℝ := f_deriv tangent_point

/-- The y-intercept of the tangent line -/
def y_intercept : ℝ := -(tangent_slope * tangent_point - f tangent_point)

theorem tangent_line_equation :
  ∀ x y : ℝ, y = tangent_slope * x + y_intercept ↔ y = 2 * x - 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2515_251550


namespace NUMINAMATH_CALUDE_f_x_plus_2_l2515_251584

/-- Given a function f where f(x) = x(x-1)/2, prove that f(x+2) = (x+2)(x+1)/2 -/
theorem f_x_plus_2 (f : ℝ → ℝ) (h : ∀ x, f x = x * (x - 1) / 2) :
  ∀ x, f (x + 2) = (x + 2) * (x + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_x_plus_2_l2515_251584


namespace NUMINAMATH_CALUDE_friendly_match_schemes_l2515_251582

/-- The number of ways to form two teams from teachers and students -/
def formTeams (numTeachers numStudents : ℕ) : ℕ :=
  let teacherCombinations := 1 -- Always select both teachers
  let studentCombinations := numStudents.choose 3
  let studentDistributions := 3 -- Ways to distribute 3 students into 2 teams
  teacherCombinations * studentCombinations * studentDistributions

/-- Theorem stating the number of ways to form teams in the given scenario -/
theorem friendly_match_schemes :
  formTeams 2 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_friendly_match_schemes_l2515_251582


namespace NUMINAMATH_CALUDE_cos_two_pi_thirds_plus_two_alpha_l2515_251535

theorem cos_two_pi_thirds_plus_two_alpha (α : Real) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  Real.cos ((2 * π) / 3 + 2 * α) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_pi_thirds_plus_two_alpha_l2515_251535


namespace NUMINAMATH_CALUDE_cookies_in_fridge_l2515_251578

/-- The number of cookies Uncle Jude baked -/
def total_cookies : ℕ := 1024

/-- The number of cookies given to Tim -/
def tim_cookies : ℕ := 48

/-- The number of cookies given to Mike -/
def mike_cookies : ℕ := 58

/-- The number of cookies given to Sarah -/
def sarah_cookies : ℕ := 78

/-- The number of cookies given to Anna -/
def anna_cookies : ℕ := 2 * (tim_cookies + mike_cookies) - sarah_cookies / 2

/-- The number of cookies Uncle Jude put in the fridge -/
def fridge_cookies : ℕ := total_cookies - (tim_cookies + mike_cookies + sarah_cookies + anna_cookies)

theorem cookies_in_fridge : fridge_cookies = 667 := by
  sorry

end NUMINAMATH_CALUDE_cookies_in_fridge_l2515_251578


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2515_251577

def arithmetic_sequence (a : ℝ) (n : ℕ) : ℝ := a + (n - 1) * (a + 1 - (a - 1))

theorem arithmetic_sequence_formula (a : ℝ) :
  (arithmetic_sequence a 1 = a - 1) ∧
  (arithmetic_sequence a 2 = a + 1) ∧
  (arithmetic_sequence a 3 = 2 * a + 3) →
  ∀ n : ℕ, arithmetic_sequence a n = 2 * n - 3 :=
by
  sorry

#check arithmetic_sequence_formula

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2515_251577


namespace NUMINAMATH_CALUDE_rectangle_formation_count_l2515_251510

theorem rectangle_formation_count :
  let horizontal_lines := 5
  let vertical_lines := 4
  let horizontal_choices := Nat.choose horizontal_lines 2
  let vertical_choices := Nat.choose vertical_lines 2
  horizontal_choices * vertical_choices = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_formation_count_l2515_251510


namespace NUMINAMATH_CALUDE_greatest_base_six_digit_sum_l2515_251511

/-- Represents a positive integer in base 6 as a list of digits (least significant first) -/
def BaseNRepr (n : ℕ) : List ℕ :=
  sorry

/-- Computes the sum of digits in a base 6 representation -/
def sumDigits (repr : List ℕ) : ℕ :=
  sorry

theorem greatest_base_six_digit_sum :
  (∀ n : ℕ, n > 0 → n < 2401 → sumDigits (BaseNRepr n) ≤ 12) ∧
  (∃ n : ℕ, n > 0 ∧ n < 2401 ∧ sumDigits (BaseNRepr n) = 12) :=
sorry

end NUMINAMATH_CALUDE_greatest_base_six_digit_sum_l2515_251511


namespace NUMINAMATH_CALUDE_product_of_numbers_l2515_251595

theorem product_of_numbers (x y : ℝ) : x + y = 40 ∧ x - y = 16 → x * y = 336 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2515_251595


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2515_251546

/-- An ellipse with parametric equations x = 3cos(φ) and y = 5sin(φ) -/
structure Ellipse where
  x : ℝ → ℝ
  y : ℝ → ℝ
  h_x : ∀ φ, x φ = 3 * Real.cos φ
  h_y : ∀ φ, y φ = 5 * Real.sin φ

/-- The eccentricity of an ellipse -/
def eccentricity (e : Ellipse) : ℝ :=
  sorry

/-- Theorem stating that the eccentricity of the given ellipse is 4/5 -/
theorem ellipse_eccentricity (e : Ellipse) : eccentricity e = 4/5 :=
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2515_251546


namespace NUMINAMATH_CALUDE_land_to_cabin_ratio_example_l2515_251527

/-- Given a total cost and cabin cost, calculate the ratio of land cost to cabin cost -/
def land_to_cabin_ratio (total_cost cabin_cost : ℕ) : ℚ :=
  (total_cost - cabin_cost) / cabin_cost

/-- Theorem: The ratio of land cost to cabin cost is 4 when the total cost is $30,000 and the cabin cost is $6,000 -/
theorem land_to_cabin_ratio_example : land_to_cabin_ratio 30000 6000 = 4 := by
  sorry

end NUMINAMATH_CALUDE_land_to_cabin_ratio_example_l2515_251527


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2515_251541

def triangle_abc (a b c A B C : ℝ) : Prop :=
  b = c * (2 * Real.sin A + Real.cos A) ∧ 
  a = Real.sqrt 2 ∧ 
  B = 3 * Real.pi / 4

theorem triangle_abc_properties (a b c A B C : ℝ) 
  (h : triangle_abc a b c A B C) :
  Real.sin C = Real.sqrt 5 / 5 ∧ 
  (1/2) * a * c * Real.sin B = 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2515_251541


namespace NUMINAMATH_CALUDE_specific_pentagon_area_l2515_251570

/-- Represents a pentagon that can be decomposed into two triangles and a trapezoid -/
structure DecomposablePentagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  triangle1_area : ℝ
  triangle2_area : ℝ
  trapezoid_area : ℝ
  decomposable : side1 > 0 ∧ side2 > 0 ∧ side3 > 0 ∧ side4 > 0 ∧ side5 > 0

/-- Calculate the area of a decomposable pentagon -/
def area (p : DecomposablePentagon) : ℝ :=
  p.triangle1_area + p.triangle2_area + p.trapezoid_area

/-- Theorem stating that a specific pentagon has an area of 848 square units -/
theorem specific_pentagon_area :
  ∃ (p : DecomposablePentagon),
    p.side1 = 18 ∧ p.side2 = 22 ∧ p.side3 = 30 ∧ p.side4 = 26 ∧ p.side5 = 22 ∧
    area p = 848 := by
  sorry


end NUMINAMATH_CALUDE_specific_pentagon_area_l2515_251570


namespace NUMINAMATH_CALUDE_fourth_grade_students_left_l2515_251519

/-- The number of students who left during the year -/
def students_left (initial : ℕ) (new : ℕ) (final : ℕ) : ℕ :=
  initial + new - final

theorem fourth_grade_students_left : students_left 11 42 47 = 6 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_left_l2515_251519


namespace NUMINAMATH_CALUDE_max_value_2x_minus_y_l2515_251522

theorem max_value_2x_minus_y (x y : ℝ) 
  (h1 : x + y - 1 < 0) 
  (h2 : x - y ≤ 0) 
  (h3 : x ≥ 0) : 
  ∀ z, z = 2*x - y → z ≤ (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_max_value_2x_minus_y_l2515_251522


namespace NUMINAMATH_CALUDE_max_y_coordinate_l2515_251513

theorem max_y_coordinate (x y : ℝ) : 
  (x^2 / 49) + ((y - 3)^2 / 25) = 0 → y ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_y_coordinate_l2515_251513


namespace NUMINAMATH_CALUDE_angle_AEC_measure_l2515_251540

-- Define the angles in the triangle
def angle_ABE' : ℝ := 150
def angle_BAC : ℝ := 108

-- Define the property of supplementary angles
def supplementary (a b : ℝ) : Prop := a + b = 180

-- Theorem statement
theorem angle_AEC_measure :
  ∀ angle_ABE angle_AEC,
  supplementary angle_ABE angle_ABE' →
  angle_ABE + angle_BAC + angle_AEC = 180 →
  angle_AEC = 42 := by
    sorry

end NUMINAMATH_CALUDE_angle_AEC_measure_l2515_251540


namespace NUMINAMATH_CALUDE_equation_solution_l2515_251556

theorem equation_solution (z : ℝ) (some_number : ℝ) :
  (14 * (-1 + z) + some_number = -14 * (1 - z) - 10) →
  some_number = -10 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2515_251556


namespace NUMINAMATH_CALUDE_square_triangle_perimeter_relation_l2515_251518

/-- Given a square with perimeter 40 and a larger equilateral triangle with 
    perimeter a + b√p (where p is prime), prove that if a = 30, b = 10, 
    and p = 3, then 7a + 5b + 3p = 269. -/
theorem square_triangle_perimeter_relation 
  (square_perimeter : ℝ) 
  (a b : ℝ) 
  (p : ℕ) 
  (h1 : square_perimeter = 40)
  (h2 : Nat.Prime p)
  (h3 : a = 30)
  (h4 : b = 10)
  (h5 : p = 3)
  : 7 * a + 5 * b + 3 * ↑p = 269 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_perimeter_relation_l2515_251518


namespace NUMINAMATH_CALUDE_inequality_solution_l2515_251581

theorem inequality_solution (a : ℝ) (h : a ≠ 0) :
  let f := fun x => x^2 - 5*a*x + 6*a^2
  (∀ x, f x > 0 ↔ (a > 0 ∧ (x < 2*a ∨ x > 3*a)) ∨ (a < 0 ∧ (x < 3*a ∨ x > 2*a))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2515_251581


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2515_251583

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + b*x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → 
  a + b = -14 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2515_251583
