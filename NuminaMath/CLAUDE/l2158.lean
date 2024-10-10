import Mathlib

namespace triangle_properties_l2158_215818

/-- Represents a triangle with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about properties of an acute triangle -/
theorem triangle_properties (t : Triangle) 
  (acute : t.A > 0 ∧ t.A < π ∧ t.B > 0 ∧ t.B < π ∧ t.C > 0 ∧ t.C < π)
  (m : ℝ × ℝ) (n : ℝ × ℝ)
  (h_m : m = (Real.sqrt 3, 2 * Real.sin t.A))
  (h_n : n = (t.c, t.a))
  (h_parallel : ∃ (k : ℝ), m.1 * n.2 = k * m.2 * n.1)
  (h_c : t.c = Real.sqrt 7)
  (h_area : 1/2 * t.a * t.b * Real.sin t.C = 3 * Real.sqrt 3 / 2) :
  t.C = π/3 ∧ t.a + t.b = 5 := by
  sorry

end triangle_properties_l2158_215818


namespace anya_lost_games_l2158_215826

/-- Represents a girl playing table tennis -/
inductive Girl
| Anya
| Bella
| Valya
| Galya
| Dasha

/-- Represents the state of a girl (playing or resting) -/
inductive State
| Playing
| Resting

/-- The number of games each girl played -/
def games_played (g : Girl) : Nat :=
  match g with
  | Girl.Anya => 4
  | Girl.Bella => 6
  | Girl.Valya => 7
  | Girl.Galya => 10
  | Girl.Dasha => 11

/-- The total number of games played -/
def total_games : Nat := 19

/-- Predicate to check if a girl lost a specific game -/
def lost_game (g : Girl) (game_number : Nat) : Prop := sorry

/-- Theorem stating that Anya lost in games 4, 8, 12, and 16 -/
theorem anya_lost_games :
  lost_game Girl.Anya 4 ∧
  lost_game Girl.Anya 8 ∧
  lost_game Girl.Anya 12 ∧
  lost_game Girl.Anya 16 :=
by sorry

end anya_lost_games_l2158_215826


namespace cost_price_calculation_l2158_215816

def selling_price : ℝ := 270
def profit_percentage : ℝ := 0.20

theorem cost_price_calculation :
  ∃ (cost_price : ℝ), 
    cost_price * (1 + profit_percentage) = selling_price ∧ 
    cost_price = 225 :=
by sorry

end cost_price_calculation_l2158_215816


namespace grade12_population_l2158_215865

/-- Represents the number of students in each grade -/
structure GradePopulation where
  grade10 : Nat
  grade11 : Nat
  grade12 : Nat

/-- Represents the number of students sampled from each grade -/
structure SampleSize where
  grade10 : Nat
  total : Nat

/-- Check if the sampling is proportional to the population -/
def isProportionalSampling (pop : GradePopulation) (sample : SampleSize) : Prop :=
  sample.grade10 * (pop.grade10 + pop.grade11 + pop.grade12) = 
  sample.total * pop.grade10

theorem grade12_population 
  (pop : GradePopulation)
  (sample : SampleSize)
  (h1 : pop.grade10 = 1000)
  (h2 : pop.grade11 = 1200)
  (h3 : sample.total = 66)
  (h4 : sample.grade10 = 20)
  (h5 : isProportionalSampling pop sample) :
  pop.grade12 = 1100 := by
  sorry

#check grade12_population

end grade12_population_l2158_215865


namespace cos_135_degrees_l2158_215856

theorem cos_135_degrees : Real.cos (135 * π / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_degrees_l2158_215856


namespace sqrt_comparison_quadratic_inequality_solution_l2158_215869

-- Part 1
theorem sqrt_comparison : Real.sqrt 7 + Real.sqrt 10 > Real.sqrt 3 + Real.sqrt 14 := by sorry

-- Part 2
theorem quadratic_inequality_solution (m : ℝ) :
  (∀ x, -1/2 * x^2 + 2*x > m*x ↔ 0 < x ∧ x < 2) → m = 1 := by sorry

end sqrt_comparison_quadratic_inequality_solution_l2158_215869


namespace fourth_root_equation_l2158_215840

theorem fourth_root_equation (y : ℝ) :
  (y * (y^5)^(1/2))^(1/4) = 4 → y = 2^(16/7) := by
  sorry

end fourth_root_equation_l2158_215840


namespace trig_identity_proof_l2158_215823

theorem trig_identity_proof : 
  Real.cos (15 * π / 180) * Real.cos (105 * π / 180) - 
  Real.cos (75 * π / 180) * Real.sin (105 * π / 180) = -1/2 := by
  sorry

end trig_identity_proof_l2158_215823


namespace linear_function_monotonicity_linear_function_parity_linear_function_y_intercept_linear_function_x_intercept_l2158_215810

-- Define a linear function
def linearFunction (a b x : ℝ) : ℝ := a * x + b

-- Theorem about monotonicity of linear functions
theorem linear_function_monotonicity (a b : ℝ) :
  (∀ x y : ℝ, x < y → linearFunction a b x < linearFunction a b y) ↔ a > 0 :=
sorry

-- Theorem about parity of linear functions
theorem linear_function_parity (a b : ℝ) :
  (∀ x : ℝ, linearFunction a b (-x) = -linearFunction a b x + 2*b) ↔ b = 0 :=
sorry

-- Theorem about y-intercept of linear functions
theorem linear_function_y_intercept (a b : ℝ) :
  linearFunction a b 0 = b :=
sorry

-- Theorem about x-intercept of linear functions (when it exists)
theorem linear_function_x_intercept (a b : ℝ) (h : a ≠ 0) :
  linearFunction a b (-b/a) = 0 :=
sorry

end linear_function_monotonicity_linear_function_parity_linear_function_y_intercept_linear_function_x_intercept_l2158_215810


namespace complement_of_A_in_U_l2158_215890

universe u

def U : Set ℕ := {0, 1, 2, 3}
def A : Set ℕ := {1, 3}

theorem complement_of_A_in_U : 
  (U \ A) = {0, 2} := by sorry

end complement_of_A_in_U_l2158_215890


namespace fraction_problem_l2158_215815

theorem fraction_problem : 
  let number : ℝ := 14.500000000000002
  let result : ℝ := 126.15
  let fraction : ℝ := result / (number ^ 2)
  fraction = 0.6 := by sorry

end fraction_problem_l2158_215815


namespace point_coordinates_l2158_215806

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The second quadrant of the Cartesian coordinate system -/
def second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The distance from a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- The distance from a point to the y-axis -/
def distance_to_y_axis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates :
  ∀ (p : Point),
    second_quadrant p →
    distance_to_x_axis p = 3 →
    distance_to_y_axis p = 7 →
    p = Point.mk (-7) 3 := by
  sorry

end point_coordinates_l2158_215806


namespace relation_between_exponents_l2158_215850

/-- Given real numbers a, b, c, d, x, y, p satisfying certain equations,
    prove that y = (3 * p^2) / 2 -/
theorem relation_between_exponents
  (a b c d x y p : ℝ)
  (h1 : a^x = c^(3*p))
  (h2 : c^(3*p) = b^2)
  (h3 : c^y = b^p)
  (h4 : b^p = d^3)
  : y = (3 * p^2) / 2 := by
  sorry

end relation_between_exponents_l2158_215850


namespace z_sum_zero_implies_x_squared_minus_y_squared_eq_neg_three_z_times_one_plus_i_purely_imaginary_implies_modulus_eq_two_sqrt_two_l2158_215848

-- Define complex numbers z₁ and z₂
def z₁ (x : ℝ) : ℂ := (2 * x + 1) + 2 * Complex.I
def z₂ (x y : ℝ) : ℂ := -x - y * Complex.I

-- Theorem 1
theorem z_sum_zero_implies_x_squared_minus_y_squared_eq_neg_three
  (x y : ℝ) (h : z₁ x + z₂ x y = 0) :
  x^2 - y^2 = -3 := by sorry

-- Theorem 2
theorem z_times_one_plus_i_purely_imaginary_implies_modulus_eq_two_sqrt_two
  (x : ℝ) (h : (Complex.I + 1) * z₁ x = Complex.I * (Complex.im ((Complex.I + 1) * z₁ x))) :
  Complex.abs (z₁ x) = 2 * Real.sqrt 2 := by sorry

end z_sum_zero_implies_x_squared_minus_y_squared_eq_neg_three_z_times_one_plus_i_purely_imaginary_implies_modulus_eq_two_sqrt_two_l2158_215848


namespace senior_field_trip_l2158_215819

theorem senior_field_trip :
  ∃! n : ℕ, n < 300 ∧ n % 17 = 15 ∧ n % 19 = 12 ∧ n = 202 := by
  sorry

end senior_field_trip_l2158_215819


namespace polynomial_factorization_l2158_215846

theorem polynomial_factorization (x : ℝ) :
  9 * (x + 3) * (x + 7) * (x + 11) * (x + 13) - 4 * x^2 =
  (3 * x^2 + 52 * x + 231) * (3 * x^2 + 56 * x + 231) := by
  sorry

end polynomial_factorization_l2158_215846


namespace divisibility_problem_l2158_215860

theorem divisibility_problem (a b : ℕ) :
  (∃ k : ℕ, a = k * (b + 1)) ∧
  (∃ m : ℕ, 43 = m * (a + b)) →
  ((a = 22 ∧ b = 21) ∨
   (a = 33 ∧ b = 10) ∨
   (a = 40 ∧ b = 3) ∨
   (a = 42 ∧ b = 1)) :=
by sorry

end divisibility_problem_l2158_215860


namespace calculate_expression_solve_inequalities_l2158_215845

-- Part 1
theorem calculate_expression : 
  |3 - Real.pi| - (-2)⁻¹ + 4 * Real.cos (60 * π / 180) = Real.pi - 1/2 := by sorry

-- Part 2
theorem solve_inequalities (x : ℝ) : 
  (5*x - 1 > 3*(x + 1) ∧ 1 + 2*x ≥ x - 1) ↔ x > 2 := by sorry

end calculate_expression_solve_inequalities_l2158_215845


namespace infinite_perfect_squares_in_sequence_l2158_215851

theorem infinite_perfect_squares_in_sequence :
  ∃ f : ℕ → ℕ × ℕ, 
    (∀ i : ℕ, (f i).1^2 = 1 + 17 * (f i).2^2) ∧ 
    (∀ i j : ℕ, i ≠ j → f i ≠ f j) := by
  sorry

end infinite_perfect_squares_in_sequence_l2158_215851


namespace roots_equation_sum_l2158_215861

theorem roots_equation_sum (a b : ℝ) : 
  (a^2 + a - 2022 = 0) → 
  (b^2 + b - 2022 = 0) → 
  (a ≠ b) →
  (a^2 + 2*a + b = 2021) := by
sorry

end roots_equation_sum_l2158_215861


namespace quadratic_inequality_range_l2158_215893

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, x^2 + m*x + 1 ≥ 0) ↔ -2 ≤ m ∧ m ≤ 2 :=
by sorry

end quadratic_inequality_range_l2158_215893


namespace f_equals_g_l2158_215847

def f (x : ℝ) : ℝ := x^2 - 2*x - 1
def g (t : ℝ) : ℝ := t^2 - 2*t - 1

theorem f_equals_g : f = g := by sorry

end f_equals_g_l2158_215847


namespace remaining_money_l2158_215894

/-- Calculates the remaining money after purchasing bread and peanut butter -/
theorem remaining_money 
  (bread_cost : ℝ) 
  (peanut_butter_cost : ℝ) 
  (initial_money : ℝ) 
  (num_loaves : ℕ) : 
  bread_cost = 2.25 →
  peanut_butter_cost = 2 →
  initial_money = 14 →
  num_loaves = 3 →
  initial_money - (num_loaves * bread_cost + peanut_butter_cost) = 5.25 := by
sorry

end remaining_money_l2158_215894


namespace f_equals_g_l2158_215808

theorem f_equals_g (f g : Nat → Nat)
  (h1 : ∀ n : Nat, n > 0 → f (g n) = f n + 1)
  (h2 : ∀ n : Nat, n > 0 → g (f n) = g n + 1) :
  ∀ n : Nat, n > 0 → f n = g n :=
by sorry

end f_equals_g_l2158_215808


namespace intersection_empty_union_real_l2158_215824

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 1}

-- Theorem 1
theorem intersection_empty (a : ℝ) : A a ∩ B = ∅ ↔ a > 3 := by sorry

-- Theorem 2
theorem union_real (a : ℝ) : A a ∪ B = Set.univ ↔ -2 ≤ a ∧ a ≤ -1/2 := by sorry

end intersection_empty_union_real_l2158_215824


namespace base8_to_base10_547_l2158_215897

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The base-8 representation of the number --/
def base8Number : List Nat := [7, 4, 5]

theorem base8_to_base10_547 :
  base8ToBase10 base8Number = 359 := by
  sorry

end base8_to_base10_547_l2158_215897


namespace problem_solution_l2158_215853

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + 2 * m * x - 1

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (3 * f m x + 4) / (x - 2)

theorem problem_solution (m : ℝ) :
  m > 0 →
  (∀ x, f m x < 0 ↔ -3 < x ∧ x < 1) →
  (∃ min_g : ℝ, ∀ x > 2, g m x ≥ min_g ∧ ∃ x₀ > 2, g m x₀ = min_g) ∧
  min_g = 12 ∧
  (∃ x₁ x₂ : ℝ, x₁ ∈ [-3, 0] ∧ x₂ ∈ [-3, 0] ∧ |f m x₁ - f m x₂| ≥ 4 → m ≥ 1) :=
by sorry

end problem_solution_l2158_215853


namespace lisas_eggs_per_child_l2158_215852

theorem lisas_eggs_per_child (breakfasts_per_year : ℕ) (num_children : ℕ) 
  (husband_eggs : ℕ) (self_eggs : ℕ) (total_eggs : ℕ) :
  breakfasts_per_year = 260 →
  num_children = 4 →
  husband_eggs = 3 →
  self_eggs = 2 →
  total_eggs = 3380 →
  ∃ (eggs_per_child : ℕ), 
    eggs_per_child = 2 ∧
    total_eggs = breakfasts_per_year * (num_children * eggs_per_child + husband_eggs + self_eggs) :=
by sorry

end lisas_eggs_per_child_l2158_215852


namespace jackson_deduction_l2158_215805

/-- Calculates the deduction in cents given an hourly wage in dollars and a deduction rate. -/
def calculate_deduction (hourly_wage : ℚ) (deduction_rate : ℚ) : ℚ :=
  hourly_wage * 100 * deduction_rate

theorem jackson_deduction :
  let hourly_wage : ℚ := 25
  let deduction_rate : ℚ := 25 / 1000  -- 2.5% expressed as a rational number
  calculate_deduction hourly_wage deduction_rate = 62.5 := by
  sorry

#eval calculate_deduction 25 (25/1000)

end jackson_deduction_l2158_215805


namespace smallest_whole_dollar_price_with_tax_l2158_215858

theorem smallest_whole_dollar_price_with_tax (n : ℕ) (x : ℕ) : n = 21 ↔ 
  n > 0 ∧ 
  x > 0 ∧
  (105 * x) % 100 = 0 ∧
  (105 * x) / 100 = n ∧
  ∀ m : ℕ, m > 0 → m < n → ¬∃ y : ℕ, y > 0 ∧ (105 * y) % 100 = 0 ∧ (105 * y) / 100 = m :=
sorry

end smallest_whole_dollar_price_with_tax_l2158_215858


namespace shenille_score_l2158_215803

/-- Represents the number of points Shenille scored in a basketball game -/
def points_scored (three_point_attempts : ℕ) (two_point_attempts : ℕ) : ℝ :=
  (0.6 : ℝ) * three_point_attempts + (0.6 : ℝ) * two_point_attempts

/-- Theorem stating that Shenille scored 18 points given the conditions -/
theorem shenille_score :
  ∀ x y : ℕ,
  x + y = 30 →
  points_scored x y = 18 :=
by sorry

end shenille_score_l2158_215803


namespace number_division_proof_l2158_215873

theorem number_division_proof (a b c d : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →  -- Ensure all parts are positive
  a / 5 = b / 7 ∧ a / 5 = c / 4 ∧ a / 5 = d / 8 →  -- Parts are proportional
  c = 60 →  -- Smallest part is 60
  a + b + c + d = 360 :=  -- Total number is 360
by sorry

end number_division_proof_l2158_215873


namespace first_player_wins_l2158_215822

-- Define the game state
structure GameState where
  hour : Nat

-- Define player moves
def firstPlayerMove (state : GameState) : GameState :=
  { hour := (state.hour + 2) % 12 }

def secondPlayerMoveInitial (state : GameState) : GameState :=
  { hour := 5 }

def secondPlayerMoveSubsequent (state : GameState) (move : Nat) : GameState :=
  { hour := (state.hour + move) % 12 }

def firstPlayerMoveSubsequent (state : GameState) (move : Nat) : GameState :=
  { hour := (state.hour + move) % 12 }

-- Define the game sequence
def gameSequence (secondPlayerLastMove : Nat) : GameState :=
  let initial := { hour := 0 }  -- 12 o'clock
  let afterFirstMove := firstPlayerMove initial
  let afterSecondMove := secondPlayerMoveInitial afterFirstMove
  let afterThirdMove := firstPlayerMoveSubsequent afterSecondMove 3
  let afterFourthMove := secondPlayerMoveSubsequent afterThirdMove secondPlayerLastMove
  let finalState := 
    if secondPlayerLastMove = 2 then
      firstPlayerMoveSubsequent afterFourthMove 3
    else
      firstPlayerMoveSubsequent afterFourthMove 2

  finalState

-- Theorem statement
theorem first_player_wins (secondPlayerLastMove : Nat) 
  (h : secondPlayerLastMove = 2 ∨ secondPlayerLastMove = 3) : 
  (gameSequence secondPlayerLastMove).hour = 6 := by
  sorry

end first_player_wins_l2158_215822


namespace hemisphere_volume_l2158_215828

/-- The volume of a hemisphere with diameter 8 cm is (128/3)π cubic centimeters. -/
theorem hemisphere_volume (π : ℝ) (hemisphere_diameter : ℝ) (hemisphere_volume : ℝ → ℝ → ℝ) :
  hemisphere_diameter = 8 →
  hemisphere_volume π hemisphere_diameter = (128 / 3) * π := by
  sorry

end hemisphere_volume_l2158_215828


namespace no_interchange_possible_l2158_215829

/-- Represents the circular arrangement of three tiles -/
inductive CircularArrangement
  | ABC
  | BCA
  | CAB

/-- Represents a move that slides a tile to an adjacent vacant space -/
inductive Move
  | Left
  | Right

/-- Applies a move to a circular arrangement -/
def applyMove (arr : CircularArrangement) (m : Move) : CircularArrangement :=
  match arr, m with
  | CircularArrangement.ABC, Move.Right => CircularArrangement.BCA
  | CircularArrangement.BCA, Move.Right => CircularArrangement.CAB
  | CircularArrangement.CAB, Move.Right => CircularArrangement.ABC
  | CircularArrangement.ABC, Move.Left => CircularArrangement.CAB
  | CircularArrangement.BCA, Move.Left => CircularArrangement.ABC
  | CircularArrangement.CAB, Move.Left => CircularArrangement.BCA

/-- Applies a sequence of moves to a circular arrangement -/
def applyMoves (arr : CircularArrangement) (moves : List Move) : CircularArrangement :=
  match moves with
  | [] => arr
  | m :: ms => applyMoves (applyMove arr m) ms

/-- Theorem stating that it's impossible to interchange 1 and 3 -/
theorem no_interchange_possible (moves : List Move) :
  applyMoves CircularArrangement.ABC moves ≠ CircularArrangement.BCA :=
sorry

end no_interchange_possible_l2158_215829


namespace fraction_domain_l2158_215825

theorem fraction_domain (x : ℝ) : 
  (∃ y : ℝ, y = 3 / (x - 2)) ↔ x ≠ 2 :=
sorry

end fraction_domain_l2158_215825


namespace cubic_derivative_equality_l2158_215809

theorem cubic_derivative_equality (f : ℝ → ℝ) (x : ℝ) :
  (f = fun x ↦ x^3) →
  (deriv f x = 3) →
  (x = 1 ∨ x = -1) := by
  sorry

end cubic_derivative_equality_l2158_215809


namespace tom_completion_time_l2158_215882

/-- Represents the duration of a combined BS and Ph.D. program -/
structure Program where
  bs_duration : ℕ
  phd_duration : ℕ

/-- Calculates the time taken by a student to complete the program given a completion ratio -/
def completion_time (p : Program) (ratio : ℚ) : ℚ :=
  ratio * (p.bs_duration + p.phd_duration)

theorem tom_completion_time :
  let p : Program := { bs_duration := 3, phd_duration := 5 }
  let ratio : ℚ := 3/4
  completion_time p ratio = 6 := by
  sorry

end tom_completion_time_l2158_215882


namespace min_value_theorem_l2158_215862

theorem min_value_theorem (α₁ α₂ : ℝ) 
  (h : (2 + Real.sin α₁)⁻¹ + (2 + Real.sin (2 * α₂))⁻¹ = 2) : 
  ∃ (k₁ k₂ : ℤ), ∀ (α₁' α₂' : ℝ), 
    (2 + Real.sin α₁')⁻¹ + (2 + Real.sin (2 * α₂'))⁻¹ = 2 →
    |10 * Real.pi - α₁' - α₂'| ≥ |10 * Real.pi - ((-π/2 : ℝ) + 2 * ↑k₁ * π) - ((-π/4 : ℝ) + ↑k₂ * π)| ∧
    |10 * Real.pi - ((-π/2 : ℝ) + 2 * ↑k₁ * π) - ((-π/4 : ℝ) + ↑k₂ * π)| = π/4 :=
by sorry

end min_value_theorem_l2158_215862


namespace not_sufficient_not_necessary_l2158_215866

theorem not_sufficient_not_necessary (p q : Prop) : 
  (¬(p ∧ q → p ∨ q)) ∧ (¬(p ∨ q → ¬(p ∧ q))) := by
  sorry

end not_sufficient_not_necessary_l2158_215866


namespace fraction_of_single_men_l2158_215889

theorem fraction_of_single_men (total : ℝ) (h1 : total > 0) : 
  let women := 0.64 * total
  let men := total - women
  let married := 0.60 * total
  let married_women := 0.75 * women
  let married_men := married - married_women
  let single_men := men - married_men
  single_men / men = 2/3 := by sorry

end fraction_of_single_men_l2158_215889


namespace sum_of_composite_functions_l2158_215843

def p (x : ℝ) : ℝ := |x| - 3

def q (x : ℝ) : ℝ := -|x|

def x_values : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

theorem sum_of_composite_functions :
  (x_values.map (λ x => q (p x))).sum = -15 := by sorry

end sum_of_composite_functions_l2158_215843


namespace subtract_like_terms_l2158_215821

theorem subtract_like_terms (a : ℝ) : 2 * a - 3 * a = -a := by
  sorry

end subtract_like_terms_l2158_215821


namespace quadratic_coincidence_l2158_215863

/-- A quadratic function with vertex at the origin -/
def QuadraticAtOrigin (a : ℝ) : ℝ → ℝ := λ x ↦ a * x^2

/-- The translated quadratic function -/
def TranslatedQuadratic : ℝ → ℝ := λ x ↦ 2 * x^2 + x - 1

/-- Theorem stating that if a quadratic function with vertex at the origin
    can be translated to coincide with y = 2x² + x - 1,
    then its analytical expression is y = 2x² -/
theorem quadratic_coincidence (a : ℝ) :
  (∃ h k : ℝ, ∀ x, QuadraticAtOrigin a (x - h) + k = TranslatedQuadratic x) →
  a = 2 :=
sorry

end quadratic_coincidence_l2158_215863


namespace two_not_units_digit_of_square_l2158_215871

def units_digit (n : ℕ) : ℕ := n % 10

theorem two_not_units_digit_of_square : ∀ n : ℕ, units_digit (n^2) ≠ 2 := by
  sorry

end two_not_units_digit_of_square_l2158_215871


namespace triangle_properties_l2158_215886

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.c - t.b = 2 * t.b * Real.cos t.A) :
  (t.a = 2 * Real.sqrt 6 ∧ t.b = 3 → t.c = 5) ∧
  (t.C = Real.pi / 2 → t.B = Real.pi / 6) := by
  sorry

end triangle_properties_l2158_215886


namespace x_less_than_neg_two_sufficient_not_necessary_for_x_leq_zero_l2158_215895

theorem x_less_than_neg_two_sufficient_not_necessary_for_x_leq_zero :
  (∀ x : ℝ, x < -2 → x ≤ 0) ∧
  (∃ x : ℝ, x ≤ 0 ∧ x ≥ -2) :=
by sorry

end x_less_than_neg_two_sufficient_not_necessary_for_x_leq_zero_l2158_215895


namespace max_boats_in_river_l2158_215802

theorem max_boats_in_river (river_width : ℝ) (boat_width : ℝ) (min_space : ℝ) :
  river_width = 42 →
  boat_width = 3 →
  min_space = 2 →
  ⌊(river_width - 2 * min_space) / (boat_width + 2 * min_space)⌋ = 5 :=
by
  sorry

end max_boats_in_river_l2158_215802


namespace ariella_daniella_savings_difference_l2158_215839

theorem ariella_daniella_savings_difference :
  ∀ (ariella_initial daniella_savings : ℝ),
    daniella_savings = 400 →
    ariella_initial + ariella_initial * 0.1 * 2 = 720 →
    ariella_initial > daniella_savings →
    ariella_initial - daniella_savings = 200 :=
by
  sorry

end ariella_daniella_savings_difference_l2158_215839


namespace volunteer_selection_theorem_l2158_215842

/-- The number of volunteers --/
def n : ℕ := 5

/-- The number of roles to be filled --/
def k : ℕ := 4

/-- The number of ways to arrange k people in k positions --/
def arrange (k : ℕ) : ℕ := Nat.factorial k

/-- The number of ways to choose k people from n people --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of ways to select and arrange volunteers for roles --/
def totalWays : ℕ :=
  arrange (k - 1) + choose (n - 1) (k - 1) * (k - 1) * arrange (k - 1)

theorem volunteer_selection_theorem : totalWays = 96 := by
  sorry

end volunteer_selection_theorem_l2158_215842


namespace half_sqrt_is_one_l2158_215812

theorem half_sqrt_is_one (x : ℝ) : (1/2 : ℝ) * Real.sqrt x = 1 → x = 4 := by
  sorry

end half_sqrt_is_one_l2158_215812


namespace arithmetic_progression_sum_l2158_215800

theorem arithmetic_progression_sum (a d : ℚ) :
  (let S₂₀ := 20 * (2 * a + 19 * d) / 2
   let S₅₀ := 50 * (2 * a + 49 * d) / 2
   let S₇₀ := 70 * (2 * a + 69 * d) / 2
   S₂₀ = 200 ∧ S₅₀ = 150) →
  S₇₀ = -350 / 3 := by
  sorry

end arithmetic_progression_sum_l2158_215800


namespace morning_travel_time_l2158_215844

/-- Proves that the time taken to move in the morning is 1 hour -/
theorem morning_travel_time (v_morning v_afternoon : ℝ) (time_diff : ℝ) 
  (h1 : v_morning = 20)
  (h2 : v_afternoon = 10)
  (h3 : time_diff = 1) :
  ∃ (t_morning : ℝ), t_morning = 1 ∧ t_morning * v_morning = (t_morning + time_diff) * v_afternoon :=
by sorry

end morning_travel_time_l2158_215844


namespace zacks_marbles_l2158_215872

theorem zacks_marbles (initial_marbles : ℕ) (kept_marbles : ℕ) (num_friends : ℕ) : 
  initial_marbles = 65 → 
  kept_marbles = 5 → 
  num_friends = 3 → 
  (initial_marbles - kept_marbles) / num_friends = 20 := by
sorry

end zacks_marbles_l2158_215872


namespace female_officers_count_l2158_215854

/-- The total number of police officers on duty that night -/
def total_on_duty : ℕ := 160

/-- The fraction of officers on duty that were female -/
def female_fraction : ℚ := 1/2

/-- The percentage of all female officers that were on duty -/
def female_on_duty_percentage : ℚ := 16/100

/-- The total number of female officers on the police force -/
def total_female_officers : ℕ := 500

theorem female_officers_count :
  total_female_officers = 
    (total_on_duty * female_fraction) / female_on_duty_percentage := by
  sorry

end female_officers_count_l2158_215854


namespace complex_equation_solution_l2158_215841

theorem complex_equation_solution (z : ℂ) : z * (1 + Complex.I) = 1 - Complex.I → z = -Complex.I := by
  sorry

end complex_equation_solution_l2158_215841


namespace fish_count_approximation_l2158_215868

/-- Approximates the total number of fish in a pond based on a tagging and recapture experiment. -/
def approximate_fish_count (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) : ℕ :=
  (initial_tagged * second_catch) / tagged_in_second

/-- Theorem stating that under the given conditions, the approximate number of fish in the pond is 313. -/
theorem fish_count_approximation :
  let initial_tagged := 50
  let second_catch := 50
  let tagged_in_second := 8
  approximate_fish_count initial_tagged second_catch tagged_in_second = 313 :=
by
  sorry

#eval approximate_fish_count 50 50 8

end fish_count_approximation_l2158_215868


namespace intersection_of_perpendicular_lines_l2158_215877

/-- Given two lines in a plane, where one is y = 3x + 4 and the other is perpendicular to it
    passing through the point (3, 2), their intersection point is (3/10, 49/10). -/
theorem intersection_of_perpendicular_lines 
  (line1 : ℝ → ℝ → Prop) 
  (line2 : ℝ → ℝ → Prop) 
  (h1 : ∀ x y, line1 x y ↔ y = 3 * x + 4)
  (h2 : ∀ x y, line2 x y → (y - 2) = -(1/3) * (x - 3))
  (h3 : line2 3 2)
  : ∃ x y, line1 x y ∧ line2 x y ∧ x = 3/10 ∧ y = 49/10 :=
sorry

end intersection_of_perpendicular_lines_l2158_215877


namespace parallelogram_sum_impossibility_l2158_215880

theorem parallelogram_sum_impossibility :
  ¬ ∃ (a b h : ℕ+), (b * h : ℕ) + 2 * a + 2 * b + 6 = 102 :=
by sorry

end parallelogram_sum_impossibility_l2158_215880


namespace gcd_of_squares_sum_l2158_215836

theorem gcd_of_squares_sum : Nat.gcd 
  (122^2 + 234^2 + 346^2 + 458^2) 
  (121^2 + 233^2 + 345^2 + 457^2) = 1 := by
  sorry

end gcd_of_squares_sum_l2158_215836


namespace length_of_mn_l2158_215814

/-- Given four collinear points A, B, C, D in order on a line,
    with M as the midpoint of AC and N as the midpoint of BD,
    prove that the length of MN is 24 when AD = 68 and BC = 20. -/
theorem length_of_mn (A B C D M N : ℝ) : 
  (A < B) → (B < C) → (C < D) →  -- Points are in order
  (M = (A + C) / 2) →            -- M is midpoint of AC
  (N = (B + D) / 2) →            -- N is midpoint of BD
  (D - A = 68) →                 -- AD = 68
  (C - B = 20) →                 -- BC = 20
  (N - M = 24) :=                -- MN = 24
by sorry

end length_of_mn_l2158_215814


namespace ratio_problem_l2158_215834

theorem ratio_problem (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 16 := by
  sorry

end ratio_problem_l2158_215834


namespace taxi_theorem_l2158_215857

def taxi_distances : List ℤ := [9, -3, -5, 4, -8, 6]

def fuel_consumption : ℚ := 0.08
def gasoline_price : ℚ := 6
def starting_price : ℚ := 6
def additional_charge : ℚ := 1.5
def starting_distance : ℕ := 3

def total_distance (distances : List ℤ) : ℕ :=
  (distances.map (Int.natAbs)).sum

def fuel_cost (distance : ℕ) : ℚ :=
  distance * fuel_consumption * gasoline_price

def segment_income (distance : ℕ) : ℚ :=
  if distance ≤ starting_distance then
    starting_price
  else
    starting_price + (distance - starting_distance) * additional_charge

def total_income (distances : List ℤ) : ℚ :=
  (distances.map (Int.natAbs)).map segment_income |>.sum

def net_income (distances : List ℤ) : ℚ :=
  total_income distances - fuel_cost (total_distance distances)

theorem taxi_theorem :
  total_distance taxi_distances = 35 ∧
  fuel_cost (total_distance taxi_distances) = 16.8 ∧
  net_income taxi_distances = 44.7 := by
  sorry

end taxi_theorem_l2158_215857


namespace polynomial_division_remainder_l2158_215887

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^3 + 2*x^2 = (x^2 + 3*x + 2) * q + (x + 2) := by sorry

end polynomial_division_remainder_l2158_215887


namespace solution_of_system_l2158_215830

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x^((Real.log y)^(Real.log (Real.log x))) = 10^(y^2)
def equation2 (x y : ℝ) : Prop := y^((Real.log x)^(Real.log (Real.log y))) = y^y

-- State the theorem
theorem solution_of_system :
  ∃ (x y : ℝ), x > 1 ∧ y > 1 ∧ equation1 x y ∧ equation2 x y ∧ x = 10^(10^10) ∧ y = 10^10 :=
sorry

end solution_of_system_l2158_215830


namespace rotation_volume_sum_l2158_215892

/-- Given a square ABCD with side length a and a point M at distance b from its center,
    the sum of volumes of solids obtained by rotating triangles ABM, BCM, CDM, and DAM
    around lines AB, BC, CD, and DA respectively is equal to 3a^3/8 -/
theorem rotation_volume_sum (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) :
  let square := {A : ℝ × ℝ | ∃ (x y : ℝ), x ∈ [0, a] ∧ y ∈ [0, a] ∧ A = (x, y)}
  let center := (a/2, a/2)
  let M : ℝ × ℝ := sorry -- A point at distance b from the center
  let volume_sum := sorry -- Sum of volumes of rotated triangles
  volume_sum = 3 * a^3 / 8 := by
sorry

end rotation_volume_sum_l2158_215892


namespace initial_people_count_initial_people_count_proof_l2158_215849

theorem initial_people_count : ℕ → Prop :=
  fun n => 
    (n / 3 : ℚ) / 2 = 15 → n = 90

-- The proof goes here
theorem initial_people_count_proof : initial_people_count 90 := by
  sorry

end initial_people_count_initial_people_count_proof_l2158_215849


namespace road_trip_distance_l2158_215804

/-- Calculates the total distance of a road trip given specific conditions --/
theorem road_trip_distance 
  (speed : ℝ) 
  (break_duration : ℝ) 
  (time_between_breaks : ℝ) 
  (hotel_search_time : ℝ) 
  (total_trip_time : ℝ) 
  (h1 : speed = 62) 
  (h2 : break_duration = 0.5) 
  (h3 : time_between_breaks = 5) 
  (h4 : hotel_search_time = 0.5) 
  (h5 : total_trip_time = 50) :
  ∃ (distance : ℝ), distance = 2790 := by
  sorry

#check road_trip_distance

end road_trip_distance_l2158_215804


namespace divisibility_by_three_l2158_215859

theorem divisibility_by_three (a b c : ℤ) (h : (9 : ℤ) ∣ (a^3 + b^3 + c^3)) :
  (3 : ℤ) ∣ a ∨ (3 : ℤ) ∣ b ∨ (3 : ℤ) ∣ c :=
by sorry

end divisibility_by_three_l2158_215859


namespace circle_rotation_invariance_l2158_215874

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a rotation
def rotate (θ : ℝ) (O : ℝ × ℝ) (p : ℝ × ℝ) : ℝ × ℝ := sorry

-- Theorem statement
theorem circle_rotation_invariance (S : Circle) (θ : ℝ) (O : ℝ × ℝ) :
  ∃ (S' : Circle), S'.radius = S.radius ∧
    (∀ (p : ℝ × ℝ), (p.1 - S.center.1)^2 + (p.2 - S.center.2)^2 = S.radius^2 →
      let p' := rotate θ O p
      (p'.1 - S'.center.1)^2 + (p'.2 - S'.center.2)^2 = S'.radius^2) :=
sorry

end circle_rotation_invariance_l2158_215874


namespace value_of_y_l2158_215878

theorem value_of_y (x y : ℚ) : 
  x = 51 → x^3*y - 2*x^2*y + x*y = 127500 → y = 1/51 := by sorry

end value_of_y_l2158_215878


namespace set_equality_implies_a_value_l2158_215899

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {a, a^2}
def B (b : ℝ) : Set ℝ := {1, b}

-- State the theorem
theorem set_equality_implies_a_value (a b : ℝ) :
  A a = B b → a = -1 := by
  sorry

end set_equality_implies_a_value_l2158_215899


namespace train_speed_with_stoppages_train_problem_l2158_215891

/-- Calculates the speed of a train including stoppages -/
theorem train_speed_with_stoppages 
  (speed_without_stoppages : ℝ) 
  (stoppage_time : ℝ) 
  (total_time : ℝ) :
  speed_without_stoppages * (total_time - stoppage_time) / total_time = 
  speed_without_stoppages * (1 - stoppage_time / total_time) := by
  sorry

/-- The speed of a train including stoppages, given its speed without stoppages and stoppage time -/
theorem train_problem 
  (speed_without_stoppages : ℝ) 
  (stoppage_time : ℝ) :
  speed_without_stoppages = 45 →
  stoppage_time = 1/3 →
  speed_without_stoppages * (1 - stoppage_time) = 30 := by
  sorry

end train_speed_with_stoppages_train_problem_l2158_215891


namespace inequality_proof_l2158_215817

theorem inequality_proof (x : ℝ) (h : x ≥ 5) :
  Real.sqrt (x - 2) - Real.sqrt (x - 3) < Real.sqrt (x - 4) - Real.sqrt (x - 5) := by
  sorry

end inequality_proof_l2158_215817


namespace evaluate_expression_l2158_215855

theorem evaluate_expression : 3000 * (3000 ^ 3000) = 3000 ^ 3001 := by
  sorry

end evaluate_expression_l2158_215855


namespace triangle_ABC_properties_l2158_215879

theorem triangle_ABC_properties (A B C : ℝ) :
  0 < A ∧ A < 2 * π / 3 →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  Real.cos C + (Real.cos A - Real.sqrt 3 * Real.sin A) * Real.cos B = 0 →
  Real.sin (A - π / 3) = 3 / 5 →
  B = π / 3 ∧ Real.sin (2 * C) = (24 + 7 * Real.sqrt 3) / 50 := by
  sorry

end triangle_ABC_properties_l2158_215879


namespace expression_value_l2158_215883

theorem expression_value (x y : ℝ) (hx : x = 1) (hy : y = -2) :
  3 * y^2 - x^2 + 2 * (2 * x^2 - 3 * x * y) - 3 * (x^2 + y^2) = 12 := by
  sorry

end expression_value_l2158_215883


namespace right_triangle_third_side_product_l2158_215835

theorem right_triangle_third_side_product (a b c : ℝ) : 
  a = 4 → b = 5 → (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) → 
  (c = 3 ∨ c = Real.sqrt 41) → 
  3 * Real.sqrt 41 = (if c = 3 then 3 else Real.sqrt 41) * (if c = Real.sqrt 41 then Real.sqrt 41 else 3) :=
sorry

end right_triangle_third_side_product_l2158_215835


namespace stevens_peaches_l2158_215820

theorem stevens_peaches (jake steven jill : ℕ) 
  (h1 : jake = steven - 18)
  (h2 : steven = jill + 13)
  (h3 : jill = 6) : 
  steven = 19 := by
sorry

end stevens_peaches_l2158_215820


namespace two_million_six_hundred_thousand_scientific_notation_l2158_215801

/-- Scientific notation representation -/
def scientific_notation (n : ℝ) (x : ℝ) (p : ℤ) : Prop :=
  1 ≤ x ∧ x < 10 ∧ n = x * (10 : ℝ) ^ p

/-- Theorem: 2,600,000 in scientific notation -/
theorem two_million_six_hundred_thousand_scientific_notation :
  ∃ (x : ℝ) (p : ℤ), scientific_notation 2600000 x p ∧ x = 2.6 ∧ p = 6 := by
  sorry

end two_million_six_hundred_thousand_scientific_notation_l2158_215801


namespace least_possible_xy_l2158_215867

theorem least_possible_xy (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 8) : 
  ∃ (min_xy : ℕ), (x * y : ℕ) ≥ min_xy ∧ 
  (∃ (x' y' : ℕ+), (1 : ℚ) / x' + (1 : ℚ) / (3 * y') = (1 : ℚ) / 8 ∧ (x' * y' : ℕ) = min_xy) :=
sorry

end least_possible_xy_l2158_215867


namespace hyperbola_point_outside_circle_l2158_215813

theorem hyperbola_point_outside_circle 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hyperbola_eq : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (focus : c > 0)
  (eccentricity : c / a = 2)
  (x₁ x₂ : ℝ)
  (roots : a * x₁^2 + b * x₁ - c = 0 ∧ a * x₂^2 + b * x₂ - c = 0) :
  x₁^2 + x₂^2 > 2 :=
by sorry

end hyperbola_point_outside_circle_l2158_215813


namespace det_special_matrix_l2158_215875

theorem det_special_matrix (k a b : ℝ) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![![1, k * Real.sin (a - b), Real.sin a],
                                        ![k * Real.sin (a - b), 1, k * Real.sin b],
                                        ![Real.sin a, k * Real.sin b, 1]]
  Matrix.det M = 1 - Real.sin a ^ 2 - k ^ 2 * Real.sin b ^ 2 := by
  sorry

end det_special_matrix_l2158_215875


namespace die_roll_probability_l2158_215864

def roll_die : ℕ := 6
def num_trials : ℕ := 6
def min_success : ℕ := 5
def success_probability : ℚ := 1/3

theorem die_roll_probability : 
  (success_probability ^ num_trials) + 
  (Nat.choose num_trials min_success * success_probability ^ min_success * (1 - success_probability) ^ (num_trials - min_success)) = 13/729 := by
  sorry

end die_roll_probability_l2158_215864


namespace radius_ratio_in_regular_hexagonal_pyramid_l2158_215870

/-- A regular hexagonal pyramid with a circumscribed sphere and an inscribed sphere. -/
structure RegularHexagonalPyramid where
  /-- The radius of the circumscribed sphere -/
  R_c : ℝ
  /-- The radius of the inscribed sphere -/
  R_i : ℝ
  /-- The center of the circumscribed sphere lies on the surface of the inscribed sphere -/
  center_on_surface : R_c = R_i + R_i

/-- The ratio of the radius of the circumscribed sphere to the radius of the inscribed sphere
    in a regular hexagonal pyramid where the center of the circumscribed sphere lies on
    the surface of the inscribed sphere is equal to 1 + √(7/3). -/
theorem radius_ratio_in_regular_hexagonal_pyramid (p : RegularHexagonalPyramid) :
  p.R_c / p.R_i = 1 + Real.sqrt (7/3) := by
  sorry

end radius_ratio_in_regular_hexagonal_pyramid_l2158_215870


namespace least_positive_integer_multiple_of_43_l2158_215884

theorem least_positive_integer_multiple_of_43 :
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → y < x → ¬((3*y)^2 + 3*29*3*y + 29^2) % 43 = 0) ∧
  ((3*x)^2 + 3*29*3*x + 29^2) % 43 = 0 ∧
  x = 19 := by
sorry

end least_positive_integer_multiple_of_43_l2158_215884


namespace marie_binders_count_l2158_215888

theorem marie_binders_count :
  let notebooks_count : ℕ := 4
  let stamps_per_notebook : ℕ := 20
  let stamps_per_binder : ℕ := 50
  let kept_fraction : ℚ := 1/4
  let stamps_given_away : ℕ := 135
  ∃ binders_count : ℕ,
    (notebooks_count * stamps_per_notebook + binders_count * stamps_per_binder) * (1 - kept_fraction) = stamps_given_away ∧
    binders_count = 2 :=
by sorry

end marie_binders_count_l2158_215888


namespace bill_caroline_age_difference_l2158_215896

theorem bill_caroline_age_difference (bill_age caroline_age : ℕ) : 
  bill_age + caroline_age = 26 →
  bill_age = 17 →
  ∃ x : ℕ, bill_age = 2 * caroline_age - x →
  2 * caroline_age - bill_age = 1 := by
sorry

end bill_caroline_age_difference_l2158_215896


namespace consecutive_circle_selections_l2158_215807

/-- Represents the arrangement of circles in the figure -/
structure CircleArrangement where
  total_circles : Nat
  long_side_rows : Nat
  long_side_ways : Nat
  diagonal_ways : Nat

/-- The specific arrangement for our problem -/
def problem_arrangement : CircleArrangement :=
  { total_circles := 33
  , long_side_rows := 6
  , long_side_ways := 21
  , diagonal_ways := 18 }

/-- Calculates the total number of ways to select three consecutive circles -/
def count_consecutive_selections (arr : CircleArrangement) : Nat :=
  arr.long_side_ways + 2 * arr.diagonal_ways

/-- Theorem stating that there are 57 ways to select three consecutive circles -/
theorem consecutive_circle_selections :
  count_consecutive_selections problem_arrangement = 57 := by
  sorry

end consecutive_circle_selections_l2158_215807


namespace angle_bisector_inequalities_l2158_215832

/-- Given a triangle with side lengths a, b, and c, and semiperimeter p,
    prove properties about the lengths of its angle bisectors. -/
theorem angle_bisector_inequalities
  (a b c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (p : ℝ) (hp : p = (a + b + c) / 2)
  (l_a l_b l_c : ℝ)
  (hl_a : l_a^2 ≤ p * (p - a))
  (hl_b : l_b^2 ≤ p * (p - b))
  (hl_c : l_c^2 ≤ p * (p - c)) :
  (l_a^2 + l_b^2 + l_c^2 ≤ p^2) ∧
  (l_a + l_b + l_c ≤ Real.sqrt 3 * p) := by
  sorry

end angle_bisector_inequalities_l2158_215832


namespace total_marbles_l2158_215876

theorem total_marbles (num_boxes : ℕ) (marbles_per_box : ℕ) 
  (h1 : num_boxes = 10) (h2 : marbles_per_box = 100) : 
  num_boxes * marbles_per_box = 1000 := by
  sorry

end total_marbles_l2158_215876


namespace triangle_side_length_l2158_215811

theorem triangle_side_length 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_area : (1/2) * a * c * Real.sin B = Real.sqrt 3)
  (h_angle : B = Real.pi / 3)
  (h_sides : a^2 + c^2 = 3*a*c) :
  b = 2 * Real.sqrt 2 := by
sorry

end triangle_side_length_l2158_215811


namespace students_present_l2158_215885

theorem students_present (total : ℕ) (absent_fraction : ℚ) : total = 28 → absent_fraction = 2/7 → total - (absent_fraction * total).floor = 20 := by
  sorry

end students_present_l2158_215885


namespace shoes_sold_day1_l2158_215881

/-- Represents the sales data for a shoe store --/
structure ShoeSales where
  shoe_price : ℕ
  boot_price : ℕ
  day1_shoes : ℕ
  day1_boots : ℕ
  day2_shoes : ℕ
  day2_boots : ℕ

/-- Theorem stating the number of shoes sold on day 1 given the sales conditions --/
theorem shoes_sold_day1 (s : ShoeSales) : 
  s.boot_price = s.shoe_price + 15 →
  s.day1_shoes * s.shoe_price + s.day1_boots * s.boot_price = 460 →
  s.day2_shoes * s.shoe_price + s.day2_boots * s.boot_price = 560 →
  s.day1_boots = 16 →
  s.day2_shoes = 8 →
  s.day2_boots = 32 →
  s.day1_shoes = 94 := by
  sorry

#check shoes_sold_day1

end shoes_sold_day1_l2158_215881


namespace equation_solution_l2158_215833

theorem equation_solution : ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = -1 ∧
  ∀ x : ℝ, x * (x - 2) + x - 2 = 0 ↔ x = x₁ ∨ x = x₂ := by
  sorry

end equation_solution_l2158_215833


namespace probability_theorem_l2158_215831

/-- Represents the number of buttons in a jar -/
structure JarContents where
  red : ℕ
  blue : ℕ

/-- Represents the state of both jars -/
structure JarState where
  jarA : JarContents
  jarB : JarContents

def initial_jarA : JarContents := { red := 6, blue := 14 }

def initial_jarB : JarContents := { red := 0, blue := 0 }

def initial_state : JarState := { jarA := initial_jarA, jarB := initial_jarB }

def buttons_removed (state : JarState) : ℕ :=
  initial_jarA.red + initial_jarA.blue - (state.jarA.red + state.jarA.blue)

def same_number_removed (state : JarState) : Prop :=
  state.jarB.red = state.jarB.blue

def fraction_remaining (state : JarState) : ℚ :=
  (state.jarA.red + state.jarA.blue) / (initial_jarA.red + initial_jarA.blue)

def probability_both_red (state : JarState) : ℚ :=
  (state.jarA.red / (state.jarA.red + state.jarA.blue)) *
  (state.jarB.red / (state.jarB.red + state.jarB.blue))

theorem probability_theorem (final_state : JarState) :
  buttons_removed final_state > 0 ∧
  same_number_removed final_state ∧
  fraction_remaining final_state = 5/7 →
  probability_both_red final_state = 3/28 := by
  sorry

#check probability_theorem

end probability_theorem_l2158_215831


namespace cos_neg_600_degrees_l2158_215837

theorem cos_neg_600_degrees : Real.cos ((-600 : ℝ) * Real.pi / 180) = -1/2 := by
  sorry

end cos_neg_600_degrees_l2158_215837


namespace power_equality_l2158_215827

theorem power_equality (q : ℕ) : 81^7 = 3^q → q = 28 := by
  sorry

end power_equality_l2158_215827


namespace betty_order_cost_l2158_215838

/-- The total cost of Betty's order -/
def total_cost (slippers_quantity : ℕ) (slippers_price : ℚ) 
               (lipstick_quantity : ℕ) (lipstick_price : ℚ)
               (hair_color_quantity : ℕ) (hair_color_price : ℚ) : ℚ :=
  slippers_quantity * slippers_price + 
  lipstick_quantity * lipstick_price + 
  hair_color_quantity * hair_color_price

/-- Theorem stating that Betty's total order cost is $44 -/
theorem betty_order_cost : 
  total_cost 6 (5/2) 4 (5/4) 8 3 = 44 := by
  sorry

#eval total_cost 6 (5/2) 4 (5/4) 8 3

end betty_order_cost_l2158_215838


namespace cubic_function_property_l2158_215898

theorem cubic_function_property (p q r s : ℝ) : 
  let g := fun (x : ℝ) => p * x^3 + q * x^2 + r * x + s
  (g (-1) = 2) → (g (-2) = -1) → (g 1 = -2) → 
  (9*p - 3*q + 3*r - s = -2) := by
sorry

end cubic_function_property_l2158_215898
