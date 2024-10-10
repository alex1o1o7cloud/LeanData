import Mathlib

namespace minimum_value_problem_l2711_271116

theorem minimum_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y) * (x + 1/y - 2020) + (y + 1/x) * (y + 1/x - 2020) ≥ -2040200 ∧
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (a + 1/b) * (a + 1/b - 2020) + (b + 1/a) * (b + 1/a - 2020) = -2040200 :=
by sorry

end minimum_value_problem_l2711_271116


namespace unique_k_for_prime_roots_l2711_271168

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem unique_k_for_prime_roots : 
  ∃! k : ℕ, ∃ p q : ℕ, 
    is_prime p ∧ 
    is_prime q ∧ 
    p + q = 78 ∧ 
    p * q = k ∧ 
    p^2 - 78*p + k = 0 ∧ 
    q^2 - 78*q + k = 0 ∧
    k = 146 :=
sorry

end unique_k_for_prime_roots_l2711_271168


namespace meeting_distance_meeting_distance_is_correct_l2711_271149

/-- The distance between two people moving towards each other -/
theorem meeting_distance (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
  2 * (a + b)
where
  /-- Person A's speed in km/h -/
  speed_a : ℝ := a
  /-- Person B's speed in km/h -/
  speed_b : ℝ := b
  /-- Time taken for them to meet in hours -/
  meeting_time : ℝ := 2
  /-- The two people start from different locations -/
  different_start_locations : Prop := True
  /-- The two people start at the same time -/
  same_start_time : Prop := True
  /-- The two people move towards each other -/
  move_towards_each_other : Prop := True

theorem meeting_distance_is_correct (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  meeting_distance a b ha hb = 2 * (a + b) := by sorry

end meeting_distance_meeting_distance_is_correct_l2711_271149


namespace arithmetic_sequence_bounds_l2711_271123

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A six-term arithmetic sequence containing 4 and 20 (in that order) -/
structure ArithSeqWithFourAndTwenty where
  a : ℕ → ℝ
  is_arithmetic : IsArithmeticSequence a
  has_four_and_twenty : ∃ i j : ℕ, i < j ∧ j < 6 ∧ a i = 4 ∧ a j = 20

/-- The theorem stating the largest and smallest possible values of z-r -/
theorem arithmetic_sequence_bounds (seq : ArithSeqWithFourAndTwenty) :
  (∃ zr : ℝ, zr = seq.a 5 - seq.a 0 ∧ zr ≤ 80 ∧ 
  ∀ zr' : ℝ, zr' = seq.a 5 - seq.a 0 → zr' ≤ zr) ∧
  (∃ zr : ℝ, zr = seq.a 5 - seq.a 0 ∧ zr ≥ 16 ∧ 
  ∀ zr' : ℝ, zr' = seq.a 5 - seq.a 0 → zr' ≥ zr) :=
sorry

end arithmetic_sequence_bounds_l2711_271123


namespace negation_of_implication_for_all_negation_of_zero_product_l2711_271197

theorem negation_of_implication_for_all (P Q : ℝ → ℝ → Prop) :
  (¬ ∀ a b : ℝ, P a b → Q a b) ↔ (∃ a b : ℝ, P a b ∧ ¬ Q a b) :=
by sorry

theorem negation_of_zero_product :
  (¬ ∀ a b : ℝ, a = 0 → a * b = 0) ↔ (∃ a b : ℝ, a = 0 ∧ a * b ≠ 0) :=
by sorry

end negation_of_implication_for_all_negation_of_zero_product_l2711_271197


namespace shaded_squares_area_sum_square_division_problem_l2711_271187

theorem shaded_squares_area_sum (initial_area : ℝ) (ratio : ℝ) :
  initial_area > 0 →
  ratio > 0 →
  ratio < 1 →
  let series_sum := initial_area / (1 - ratio)
  series_sum = initial_area / (1 - ratio) :=
by sorry

theorem square_division_problem :
  let initial_side_length : ℝ := 8
  let initial_area : ℝ := initial_side_length ^ 2
  let ratio : ℝ := 1 / 4
  let series_sum := initial_area / (1 - ratio)
  series_sum = 64 / 3 :=
by sorry

end shaded_squares_area_sum_square_division_problem_l2711_271187


namespace f_at_negative_three_l2711_271107

def f (x : ℝ) : ℝ := -2 * x^3 + 5 * x^2 - 3 * x + 2

theorem f_at_negative_three : f (-3) = 110 := by
  sorry

end f_at_negative_three_l2711_271107


namespace sugar_water_concentration_increases_l2711_271144

theorem sugar_water_concentration_increases 
  (a b m : ℝ) 
  (h1 : b > a) 
  (h2 : a > 0) 
  (h3 : m > 0) : 
  a / b < (a + m) / (b + m) := by
sorry

end sugar_water_concentration_increases_l2711_271144


namespace fraction_sum_l2711_271176

theorem fraction_sum (a b : ℚ) (h : a / b = 1 / 2) : (a + b) / b = 3 / 2 := by
  sorry

end fraction_sum_l2711_271176


namespace max_popsicles_with_10_dollars_l2711_271134

/-- Represents the available popsicle purchase options -/
structure PopsicleOption where
  quantity : ℕ
  price : ℕ

/-- Finds the maximum number of popsicles that can be purchased with a given budget -/
def maxPopsicles (options : List PopsicleOption) (budget : ℕ) : ℕ :=
  sorry

/-- The main theorem proving that 23 is the maximum number of popsicles that can be purchased -/
theorem max_popsicles_with_10_dollars :
  let options : List PopsicleOption := [
    ⟨1, 1⟩,  -- Single popsicle
    ⟨3, 2⟩,  -- 3-popsicle box
    ⟨5, 3⟩,  -- 5-popsicle box
    ⟨10, 4⟩  -- 10-popsicle box
  ]
  let budget := 10
  maxPopsicles options budget = 23 := by
  sorry

end max_popsicles_with_10_dollars_l2711_271134


namespace quadratic_function_minimum_l2711_271148

theorem quadratic_function_minimum (a b c : ℝ) (h₁ : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  let f' : ℝ → ℝ := λ x ↦ 2 * a * x + b
  (f' 0 > 0) →
  (∀ x : ℝ, f x ≥ 0) →
  f 1 / f' 0 ≥ 2 := by
  sorry

end quadratic_function_minimum_l2711_271148


namespace divisibility_theorem_l2711_271156

theorem divisibility_theorem (n : ℕ) (a : ℝ) (h : n > 0) :
  ∃ k : ℤ, a^(2*n + 1) + (a - 1)^(n + 2) = k * (a^2 - a + 1) := by
sorry

end divisibility_theorem_l2711_271156


namespace team_games_count_l2711_271160

/-- The number of games the team plays -/
def total_games : ℕ := 14

/-- The number of shots John gets per foul -/
def shots_per_foul : ℕ := 2

/-- The number of times John gets fouled per game -/
def fouls_per_game : ℕ := 5

/-- The percentage of games John plays -/
def games_played_percentage : ℚ := 4/5

/-- The total number of free throws John gets -/
def total_free_throws : ℕ := 112

theorem team_games_count :
  total_games = 14 ∧
  shots_per_foul = 2 ∧
  fouls_per_game = 5 ∧
  games_played_percentage = 4/5 ∧
  total_free_throws = 112 →
  (total_games : ℚ) * games_played_percentage * (shots_per_foul * fouls_per_game) = total_free_throws := by
  sorry

end team_games_count_l2711_271160


namespace sqrt_eight_times_sqrt_two_l2711_271169

theorem sqrt_eight_times_sqrt_two : Real.sqrt 8 * Real.sqrt 2 = 4 := by
  sorry

end sqrt_eight_times_sqrt_two_l2711_271169


namespace distance_when_parallel_max_distance_l2711_271165

/-- A parabola with vertex at the origin -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1^2}

/-- Two points on the parabola -/
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

/-- The origin -/
def O : ℝ × ℝ := (0, 0)

/-- Assumption that P and Q are on the parabola -/
axiom h_P_on_parabola : P ∈ Parabola
axiom h_Q_on_parabola : Q ∈ Parabola

/-- Assumption that OP is perpendicular to OQ -/
axiom h_perpendicular : (P.1 * Q.1 + P.2 * Q.2) = 0

/-- Distance from a point to a line -/
def distanceToLine (point : ℝ × ℝ) (line : Set (ℝ × ℝ)) : ℝ := sorry

/-- The line PQ -/
def LinePQ : Set (ℝ × ℝ) := sorry

/-- Statement: When PQ is parallel to x-axis, distance from O to PQ is 1 -/
theorem distance_when_parallel : 
  P.2 = Q.2 → distanceToLine O LinePQ = 1 := sorry

/-- Statement: The maximum distance from O to PQ is 1 -/
theorem max_distance : 
  ∀ P Q : ℝ × ℝ, P ∈ Parabola → Q ∈ Parabola → 
  (P.1 * Q.1 + P.2 * Q.2) = 0 → 
  distanceToLine O LinePQ ≤ 1 := sorry

end distance_when_parallel_max_distance_l2711_271165


namespace convex_pentagon_integer_point_l2711_271166

-- Define a point in 2D space
structure Point where
  x : ℤ
  y : ℤ

-- Define a pentagon as a list of 5 points
def Pentagon := List Point

-- Define a predicate to check if a pentagon is convex
def isConvex (p : Pentagon) : Prop := sorry

-- Define a predicate to check if a point is inside or on the boundary of a pentagon
def isInsideOrOnBoundary (point : Point) (p : Pentagon) : Prop := sorry

-- The main theorem
theorem convex_pentagon_integer_point 
  (p : Pentagon) 
  (h1 : p.length = 5) 
  (h2 : isConvex p) : 
  ∃ (point : Point), isInsideOrOnBoundary point p := by
  sorry

end convex_pentagon_integer_point_l2711_271166


namespace max_true_statements_l2711_271181

theorem max_true_statements (c d : ℝ) : 
  let statements := [
    (1 / c > 1 / d),
    (c^2 < d^2),
    (c > d),
    (c > 0),
    (d > 0)
  ]
  ∃ (trueStatements : Finset (Fin 5)), 
    (∀ i ∈ trueStatements, statements[i] = true) ∧ 
    trueStatements.card ≤ 3 ∧
    ∀ (otherStatements : Finset (Fin 5)), 
      (∀ i ∈ otherStatements, statements[i] = true) →
      otherStatements.card ≤ 3 :=
by
  sorry

end max_true_statements_l2711_271181


namespace A_power_101_l2711_271199

def A : Matrix (Fin 3) (Fin 3) ℤ := !![0, 0, 1; 1, 0, 0; 0, 1, 0]

theorem A_power_101 : A ^ 101 = !![0, 1, 0; 0, 0, 1; 1, 0, 0] := by
  sorry

end A_power_101_l2711_271199


namespace quadratic_equations_solutions_l2711_271171

theorem quadratic_equations_solutions :
  (∀ x : ℝ, x^2 + 10*x = 56 ↔ x = 4 ∨ x = -14) ∧
  (∀ x : ℝ, 4*x^2 + 48 = 32*x ↔ x = 6 ∨ x = 2) ∧
  (∀ x : ℝ, x^2 + 20 = 12*x ↔ x = 10 ∨ x = 2) ∧
  (∀ x : ℝ, 3*x^2 - 36 = 32*x - x^2 ↔ x = 9 ∨ x = -1) ∧
  (∀ x : ℝ, x^2 + 8*x = 20) ∧
  (∀ x : ℝ, 3*x^2 = 12*x + 63) ∧
  (∀ x : ℝ, x^2 + 16 = 8*x) ∧
  (∀ x : ℝ, 6*x^2 + 12*x = 90) ∧
  (∀ x : ℝ, (1/2)*x^2 + x = 7.5) :=
by sorry


end quadratic_equations_solutions_l2711_271171


namespace inequality_equivalence_l2711_271121

theorem inequality_equivalence (x : ℝ) : -1/2 * x - 1 < 0 ↔ x > -2 := by
  sorry

end inequality_equivalence_l2711_271121


namespace line_intercepts_sum_l2711_271182

theorem line_intercepts_sum (d : ℚ) : 
  (∃ (x y : ℚ), 6 * x + 5 * y + d = 0 ∧ x + y = 15) → d = -450 / 11 := by
  sorry

end line_intercepts_sum_l2711_271182


namespace cosine_amplitude_l2711_271154

theorem cosine_amplitude (c d : ℝ) (hc : c < 0) (hd : d > 0) 
  (hmax : ∀ x, c * Real.cos (d * x) ≤ 3) 
  (hmin : ∀ x, -3 ≤ c * Real.cos (d * x)) 
  (hmax_achieved : ∃ x, c * Real.cos (d * x) = 3) 
  (hmin_achieved : ∃ x, c * Real.cos (d * x) = -3) : 
  c = -3 := by
sorry

end cosine_amplitude_l2711_271154


namespace unique_solution_proof_l2711_271138

/-- The value of q for which the quadratic equation qx^2 - 16x + 8 = 0 has exactly one solution -/
def unique_solution_q : ℝ := 8

/-- The quadratic equation qx^2 - 16x + 8 = 0 -/
def quadratic_equation (q x : ℝ) : ℝ := q * x^2 - 16 * x + 8

theorem unique_solution_proof :
  ∀ q : ℝ, q ≠ 0 →
  (∃! x : ℝ, quadratic_equation q x = 0) ↔ q = unique_solution_q :=
sorry

end unique_solution_proof_l2711_271138


namespace square_length_CD_l2711_271145

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = -3 * x^2 + 2 * x + 5

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the problem statement
theorem square_length_CD (C D : PointOnParabola) : 
  (C.x = -D.x ∧ C.y = -D.y) → (C.x - D.x)^2 + (C.y - D.y)^2 = 100/3 := by
  sorry

end square_length_CD_l2711_271145


namespace max_a_value_l2711_271110

theorem max_a_value (a : ℝ) : (∀ x : ℝ, x^2 + |2*x - 6| ≥ a) → a ≤ 5 :=
by sorry

end max_a_value_l2711_271110


namespace chlorine_original_cost_l2711_271109

/-- The original cost of a liter of chlorine -/
def chlorine_cost : ℝ := sorry

/-- The sale price of chlorine as a percentage of its original price -/
def chlorine_sale_percent : ℝ := 0.80

/-- The original price of a box of soap -/
def soap_original_price : ℝ := 16

/-- The sale price of a box of soap -/
def soap_sale_price : ℝ := 12

/-- The number of liters of chlorine bought -/
def chlorine_quantity : ℕ := 3

/-- The number of boxes of soap bought -/
def soap_quantity : ℕ := 5

/-- The total savings when buying chlorine and soap at sale prices -/
def total_savings : ℝ := 26

theorem chlorine_original_cost :
  chlorine_cost = 10 :=
by
  sorry

end chlorine_original_cost_l2711_271109


namespace equation_solution_l2711_271172

theorem equation_solution : 
  ∃! y : ℝ, 7 * (2 * y - 3) + 5 = -3 * (4 - 5 * y) ∧ y = -4 := by
  sorry

end equation_solution_l2711_271172


namespace range_of_m_l2711_271133

def has_two_distinct_real_roots (m : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def is_increasing_on_reals (m : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → (m^2 - m + 1)^x < (m^2 - m + 1)^y

def p (m : ℝ) : Prop := has_two_distinct_real_roots m

def q (m : ℝ) : Prop := is_increasing_on_reals m

theorem range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m) →
  ((-2 ≤ m ∧ m < 0) ∨ (1 < m ∧ m ≤ 2)) :=
sorry

end range_of_m_l2711_271133


namespace points_on_line_l2711_271104

/-- Given three points (8, 10), (0, m), and (-8, 6) on a straight line, prove that m = 8 -/
theorem points_on_line (m : ℝ) : 
  (∀ (t : ℝ), ∃ (s : ℝ), (8 * (1 - t) + 0 * t, 10 * (1 - t) + m * t) = 
    (0 * (1 - s) + (-8) * s, m * (1 - s) + 6 * s)) → 
  m = 8 := by
  sorry

end points_on_line_l2711_271104


namespace aaron_can_lids_l2711_271186

theorem aaron_can_lids (num_boxes : ℕ) (lids_per_box : ℕ) (total_lids : ℕ) :
  num_boxes = 3 →
  lids_per_box = 13 →
  total_lids = 53 →
  total_lids - (num_boxes * lids_per_box) = 14 := by
  sorry

end aaron_can_lids_l2711_271186


namespace perfect_square_sum_l2711_271193

theorem perfect_square_sum (x y : ℕ) 
  (h : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / (x + 2) + (1 : ℚ) / (y - 2)) : 
  ∃ n : ℕ, x * y + 1 = n ^ 2 := by
sorry

end perfect_square_sum_l2711_271193


namespace gcd_and_prime_check_l2711_271108

theorem gcd_and_prime_check : 
  (Nat.gcd 7854 15246 = 6) ∧ ¬(Nat.Prime 6) := by sorry

end gcd_and_prime_check_l2711_271108


namespace problem_statement_l2711_271179

theorem problem_statement (a b : ℝ) : 
  (Real.sqrt (a - 2) + abs (b + 3) = 0) → ((a + b) ^ 2023 = -1) := by
  sorry

end problem_statement_l2711_271179


namespace function_minimum_value_l2711_271188

theorem function_minimum_value (x : ℝ) (h : x > -1) :
  (x^2 + 7*x + 10) / (x + 1) ≥ 9 := by
  sorry

end function_minimum_value_l2711_271188


namespace factor_polynomial_l2711_271189

theorem factor_polynomial (x : ℝ) : 75 * x^3 - 300 * x^7 = 75 * x^3 * (1 - 4 * x^4) := by
  sorry

end factor_polynomial_l2711_271189


namespace sqrt_four_fourth_powers_sum_l2711_271129

theorem sqrt_four_fourth_powers_sum (h : ℝ) : 
  h = Real.sqrt (4^4 + 4^4 + 4^4 + 4^4) → h = 32 :=
by
  sorry

end sqrt_four_fourth_powers_sum_l2711_271129


namespace rachel_made_18_dollars_l2711_271174

/-- The amount of money Rachel made selling chocolate bars -/
def rachel_money (total_bars : ℕ) (unsold_bars : ℕ) (price_per_bar : ℕ) : ℕ :=
  (total_bars - unsold_bars) * price_per_bar

/-- Theorem stating that Rachel made $18 -/
theorem rachel_made_18_dollars :
  rachel_money 13 4 2 = 18 := by
  sorry

end rachel_made_18_dollars_l2711_271174


namespace probability_even_balls_correct_l2711_271106

def probability_even_balls (n : ℕ) : ℚ :=
  1/2 - 1/(2*(2^n - 1))

theorem probability_even_balls_correct (n : ℕ) :
  probability_even_balls n = 1/2 - 1/(2*(2^n - 1)) :=
sorry

end probability_even_balls_correct_l2711_271106


namespace consecutive_even_sum_representation_l2711_271127

theorem consecutive_even_sum_representation (n k : ℕ) (hn : n > 2) (hk : k > 2) :
  ∃ m : ℕ, n * (n - 1)^(k - 1) = n * (2 * m + (n - 1)) := by
  sorry

end consecutive_even_sum_representation_l2711_271127


namespace albert_took_five_candies_l2711_271141

/-- The number of candies Albert took away -/
def candies_taken (initial final : ℕ) : ℕ := initial - final

/-- Proof that Albert took 5 candies -/
theorem albert_took_five_candies :
  candies_taken 76 71 = 5 := by
  sorry

end albert_took_five_candies_l2711_271141


namespace sean_whistles_l2711_271142

/-- Given that Sean has 32 more whistles than Charles and Charles has 13 whistles,
    prove that Sean has 45 whistles. -/
theorem sean_whistles (charles_whistles : ℕ) (sean_extra_whistles : ℕ) 
  (h1 : charles_whistles = 13)
  (h2 : sean_extra_whistles = 32) :
  charles_whistles + sean_extra_whistles = 45 := by
  sorry

end sean_whistles_l2711_271142


namespace remaining_pennies_l2711_271157

def initial_pennies : ℕ := 989
def spent_pennies : ℕ := 728

theorem remaining_pennies :
  initial_pennies - spent_pennies = 261 :=
by sorry

end remaining_pennies_l2711_271157


namespace scale_model_height_l2711_271185

/-- The scale ratio of the model -/
def scale_ratio : ℚ := 1 / 25

/-- The actual height of the Eiffel Tower in feet -/
def actual_height : ℕ := 1063

/-- The height of the scale model before rounding -/
def model_height : ℚ := actual_height * scale_ratio

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

theorem scale_model_height :
  round_to_nearest model_height = 43 := by sorry

end scale_model_height_l2711_271185


namespace fuel_consumption_model_initial_fuel_fuel_decrease_rate_non_negative_fuel_l2711_271146

/-- Represents the remaining fuel in a car's tank as a function of time. -/
def remaining_fuel (x : ℝ) : ℝ :=
  80 - 10 * x

theorem fuel_consumption_model (x : ℝ) (hx : x ≥ 0) :
  remaining_fuel x = 80 - 10 * x :=
by
  sorry

/-- Verifies that the remaining fuel is 80 at time 0. -/
theorem initial_fuel : remaining_fuel 0 = 80 :=
by
  sorry

/-- Proves that the fuel decreases by 10 units for each unit of time. -/
theorem fuel_decrease_rate (x : ℝ) :
  remaining_fuel (x + 1) = remaining_fuel x - 10 :=
by
  sorry

/-- Confirms that the remaining fuel is non-negative for non-negative time. -/
theorem non_negative_fuel (x : ℝ) (hx : x ≥ 0) :
  remaining_fuel x ≥ 0 :=
by
  sorry

end fuel_consumption_model_initial_fuel_fuel_decrease_rate_non_negative_fuel_l2711_271146


namespace gcd_of_12_and_20_l2711_271158

theorem gcd_of_12_and_20 : Nat.gcd 12 20 = 4 := by
  sorry

end gcd_of_12_and_20_l2711_271158


namespace smallest_pencil_count_l2711_271150

theorem smallest_pencil_count (p : ℕ) : 
  (p > 0) →
  (p % 6 = 5) → 
  (p % 7 = 3) → 
  (p % 8 = 7) → 
  (∀ q : ℕ, q > 0 → q % 6 = 5 → q % 7 = 3 → q % 8 = 7 → p ≤ q) →
  p = 35 := by
sorry

end smallest_pencil_count_l2711_271150


namespace l_shape_area_l2711_271132

/-- The area of an L shape formed by removing a smaller rectangle from a larger rectangle -/
theorem l_shape_area (big_length big_width small_length small_width : ℕ) : 
  big_length = 8 →
  big_width = 5 →
  small_length = big_length - 2 →
  small_width = big_width - 2 →
  big_length * big_width - small_length * small_width = 22 :=
by sorry

end l_shape_area_l2711_271132


namespace employee_hire_year_l2711_271137

/-- Rule of 70 retirement provision -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- The year an employee was hired -/
def hire_year : ℕ := 1970

/-- The age at which the employee was hired -/
def hire_age : ℕ := 32

/-- The year the employee becomes eligible to retire -/
def retirement_year : ℕ := 2008

theorem employee_hire_year :
  rule_of_70 (hire_age + (retirement_year - hire_year)) (retirement_year - hire_year) ∧
  ∀ y, y > hire_year →
    ¬rule_of_70 (hire_age + (retirement_year - y)) (retirement_year - y) :=
sorry

end employee_hire_year_l2711_271137


namespace track_length_proof_l2711_271105

/-- The length of the circular track in meters -/
def track_length : ℝ := 180

/-- The distance Brenda runs before their first meeting in meters -/
def brenda_first_meeting : ℝ := 120

/-- The additional distance Sally runs after their first meeting before their second meeting in meters -/
def sally_additional : ℝ := 180

theorem track_length_proof :
  ∃ (brenda_speed sally_speed : ℝ),
    brenda_speed > 0 ∧ sally_speed > 0 ∧
    brenda_speed ≠ sally_speed ∧
    (sally_speed * track_length = brenda_speed * (track_length + brenda_first_meeting)) ∧
    (sally_speed * (track_length + sally_additional) = brenda_speed * (2 * track_length + brenda_first_meeting)) :=
sorry

end track_length_proof_l2711_271105


namespace cubic_diff_linear_diff_mod_six_l2711_271136

theorem cubic_diff_linear_diff_mod_six (x y : ℤ) : 
  (x^3 - y^3) % 6 = (x - y) % 6 := by sorry

end cubic_diff_linear_diff_mod_six_l2711_271136


namespace movie_of_the_year_threshold_l2711_271161

def total_members : ℕ := 775
def threshold : ℚ := 1/4

theorem movie_of_the_year_threshold : 
  ∀ n : ℕ, (n : ℚ) ≥ threshold * total_members ∧ 
  ∀ m : ℕ, m < n → (m : ℚ) < threshold * total_members → n = 194 := by
  sorry

end movie_of_the_year_threshold_l2711_271161


namespace min_value_of_reciprocal_sum_min_value_achievable_l2711_271198

theorem min_value_of_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : a + 3*b = 1) :
  1/a + 3/b ≥ 16 :=
sorry

theorem min_value_achievable (ε : ℝ) (hε : ε > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 3*b = 1 ∧ 1/a + 3/b < 16 + ε :=
sorry

end min_value_of_reciprocal_sum_min_value_achievable_l2711_271198


namespace opposite_reciprocal_sum_l2711_271153

theorem opposite_reciprocal_sum (m n c d : ℝ) : 
  m = -n → c * d = 1 → m + n + 3 * c * d - 10 = -7 := by sorry

end opposite_reciprocal_sum_l2711_271153


namespace linear_function_range_l2711_271164

theorem linear_function_range (x y : ℝ) :
  y = -2 * x + 3 →
  y ≤ 6 →
  x ≥ -3/2 :=
by
  sorry

end linear_function_range_l2711_271164


namespace tuesday_equals_friday_l2711_271163

def total_weekly_time : ℝ := 5
def monday_time : ℝ := 1.5
def wednesday_time : ℝ := 1.5
def friday_time : ℝ := 1

def tuesday_time : ℝ := total_weekly_time - (monday_time + wednesday_time + friday_time)

theorem tuesday_equals_friday : tuesday_time = friday_time := by
  sorry

end tuesday_equals_friday_l2711_271163


namespace hotel_stay_duration_l2711_271119

theorem hotel_stay_duration (cost_per_night_per_person : ℕ) (num_people : ℕ) (total_cost : ℕ) : 
  cost_per_night_per_person = 40 →
  num_people = 3 →
  total_cost = 360 →
  total_cost = cost_per_night_per_person * num_people * 3 :=
by
  sorry

end hotel_stay_duration_l2711_271119


namespace semicircle_area_with_inscribed_rectangle_radius_from_inscribed_rectangle_l2711_271140

/-- The area of a semicircle with an inscribed 1×3 rectangle -/
theorem semicircle_area_with_inscribed_rectangle (r : ℝ) : 
  (r^2 = 5/4) → -- The radius squared equals 5/4
  (π * r^2 / 2 = 5*π/4) := -- The area of the semicircle equals 5π/4
by sorry

/-- The relationship between the radius and the inscribed rectangle -/
theorem radius_from_inscribed_rectangle (r : ℝ) :
  (r^2 = 5/4) ↔ -- The radius squared equals 5/4
  (∃ (w h : ℝ), w = 1 ∧ h = 3 ∧ w^2 + (h/2)^2 = r^2) := -- There exists a 1×3 rectangle inscribed
by sorry

end semicircle_area_with_inscribed_rectangle_radius_from_inscribed_rectangle_l2711_271140


namespace grid_erasing_game_strategies_l2711_271180

/-- Represents the possible outcomes of the grid erasing game -/
inductive GameOutcome
  | FirstPlayerWins
  | SecondPlayerWins

/-- Defines the grid erasing game -/
def GridErasingGame (rows : Nat) (cols : Nat) : GameOutcome :=
  sorry

/-- Theorem stating the winning strategies for different grid sizes -/
theorem grid_erasing_game_strategies :
  (GridErasingGame 10 12 = GameOutcome.SecondPlayerWins) ∧
  (GridErasingGame 9 10 = GameOutcome.FirstPlayerWins) ∧
  (GridErasingGame 9 11 = GameOutcome.SecondPlayerWins) := by
  sorry

/-- Lemma: In a grid with even dimensions, the second player has a winning strategy -/
lemma even_dimensions_second_player_wins (m n : Nat) 
  (hm : Even m) (hn : Even n) : 
  GridErasingGame m n = GameOutcome.SecondPlayerWins := by
  sorry

/-- Lemma: In a grid with one odd and one even dimension, the first player has a winning strategy -/
lemma odd_even_dimensions_first_player_wins (m n : Nat) 
  (hm : Odd m) (hn : Even n) : 
  GridErasingGame m n = GameOutcome.FirstPlayerWins := by
  sorry

/-- Lemma: In a grid with both odd dimensions, the second player has a winning strategy -/
lemma odd_dimensions_second_player_wins (m n : Nat) 
  (hm : Odd m) (hn : Odd n) : 
  GridErasingGame m n = GameOutcome.SecondPlayerWins := by
  sorry

end grid_erasing_game_strategies_l2711_271180


namespace g_sum_property_l2711_271120

-- Define the function g
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^8 + b * x^6 - c * x^4 + 5

-- State the theorem
theorem g_sum_property (a b c : ℝ) : g a b c 10 = 3 → g a b c 10 + g a b c (-10) = 6 := by
  sorry

end g_sum_property_l2711_271120


namespace gcd_lcm_product_l2711_271117

theorem gcd_lcm_product (a b : ℕ) (ha : a = 150) (hb : b = 90) :
  (Nat.gcd a b) * (Nat.lcm a b) = 13500 := by
  sorry

end gcd_lcm_product_l2711_271117


namespace no_thirty_consecutive_zeros_l2711_271173

/-- For any natural number n, the last 100 digits of 5^n do not contain 30 consecutive zeros. -/
theorem no_thirty_consecutive_zeros (n : ℕ) : 
  ¬ (∃ k : ℕ, k + 29 < 100 ∧ ∀ i : ℕ, i < 30 → (5^n / 10^k) % 10^(100-k) % 10 = 0) := by
  sorry

end no_thirty_consecutive_zeros_l2711_271173


namespace smallest_resolvable_debt_l2711_271167

theorem smallest_resolvable_debt (pig_value chicken_value : ℕ) 
  (h_pig : pig_value = 250) (h_chicken : chicken_value = 175) :
  ∃ (debt : ℕ), debt > 0 ∧ 
  (∃ (p c : ℤ), debt = pig_value * p + chicken_value * c) ∧
  (∀ (d : ℕ), d > 0 → d < debt → 
    ¬∃ (p c : ℤ), d = pig_value * p + chicken_value * c) :=
by
  -- The proof goes here
  sorry

end smallest_resolvable_debt_l2711_271167


namespace complex_calculation_l2711_271139

theorem complex_calculation : (26.3 * 12 * 20) / 3 + 125 - Real.sqrt 576 = 21141 := by
  sorry

end complex_calculation_l2711_271139


namespace average_people_per_hour_rounded_l2711_271155

/-- The number of people moving to Alaska in 5 days -/
def total_people : ℕ := 4000

/-- The number of days -/
def num_days : ℕ := 5

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Calculate the average number of people moving to Alaska per hour -/
def average_per_hour : ℚ :=
  total_people / (num_days * hours_per_day)

/-- Round a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem average_people_per_hour_rounded : 
  round_to_nearest average_per_hour = 33 := by
  sorry

end average_people_per_hour_rounded_l2711_271155


namespace intersection_line_of_circles_l2711_271101

/-- Definition of a circle with center (h, k) and radius r -/
def Circle (h k r : ℝ) := {(x, y) : ℝ × ℝ | (x - h)^2 + (y - k)^2 = r^2}

/-- The intersection line of two circles -/
def IntersectionLine (c1 c2 : ℝ × ℝ × ℝ) : ℝ × ℝ → Prop :=
  let (h1, k1, r1) := c1
  let (h2, k2, r2) := c2
  λ (x, y) => x + y = -59/34

theorem intersection_line_of_circles :
  let c1 : ℝ × ℝ × ℝ := (-12, -6, 15)
  let c2 : ℝ × ℝ × ℝ := (4, 11, 9)
  ∀ (p : ℝ × ℝ), p ∈ Circle c1.1 c1.2.1 c1.2.2 ∩ Circle c2.1 c2.2.1 c2.2.2 →
    IntersectionLine c1 c2 p :=
by
  sorry

#check intersection_line_of_circles

end intersection_line_of_circles_l2711_271101


namespace factorial_division_l2711_271135

theorem factorial_division : Nat.factorial 5 / Nat.factorial (5 - 3) = 60 := by sorry

end factorial_division_l2711_271135


namespace max_students_planting_trees_l2711_271114

theorem max_students_planting_trees :
  ∀ (a b : ℕ),
  3 * a + 5 * b = 115 →
  ∀ (x y : ℕ),
  3 * x + 5 * y = 115 →
  a + b ≥ x + y →
  a + b = 37 :=
by sorry

end max_students_planting_trees_l2711_271114


namespace equation_solution_l2711_271178

noncomputable def f (x : ℝ) : ℝ := x + Real.arctan x * Real.sqrt (x^2 + 1)

theorem equation_solution :
  ∃! x : ℝ, 2*x + 2 + f x + f (x + 2) = 0 ∧ x = -1 :=
sorry

end equation_solution_l2711_271178


namespace system_solution_l2711_271102

theorem system_solution :
  ∃ (x y₁ y₂ : ℝ),
    (x / 5 + 3 = 4) ∧
    (x^2 - 4*x*y₁ + 3*y₁^2 = 36) ∧
    (x^2 - 4*x*y₂ + 3*y₂^2 = 36) ∧
    (x = 5) ∧
    (y₁ = 10/3 + Real.sqrt 133 / 3) ∧
    (y₂ = 10/3 - Real.sqrt 133 / 3) :=
by sorry


end system_solution_l2711_271102


namespace price_reduction_for_target_profit_max_profit_price_reduction_l2711_271170

/-- Profit function given price reduction x -/
def profit (x : ℝ) : ℝ := (80 - x) * (40 + 2 * x)

/-- Theorem for part 1 of the problem -/
theorem price_reduction_for_target_profit :
  profit 40 = 4800 := by sorry

/-- Theorem for part 2 of the problem -/
theorem max_profit_price_reduction :
  ∀ x : ℝ, profit x ≤ profit 30 ∧ profit 30 = 5000 := by sorry

end price_reduction_for_target_profit_max_profit_price_reduction_l2711_271170


namespace min_value_expression_l2711_271190

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ((3 * a * b - 6 * b + a * (1 - a))^2 + (9 * b^2 + 2 * a + 3 * b * (1 - a))^2) / (a^2 + 9 * b^2) ≥ 4 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧
    ((3 * a₀ * b₀ - 6 * b₀ + a₀ * (1 - a₀))^2 + (9 * b₀^2 + 2 * a₀ + 3 * b₀ * (1 - a₀))^2) / (a₀^2 + 9 * b₀^2) = 4 :=
by sorry

end min_value_expression_l2711_271190


namespace clock_hand_positions_l2711_271159

/-- Represents the number of minutes after 12:00 when the clock hands overlap -/
def overlap_time : ℚ := 720 / 11

/-- Represents the number of times the clock hands overlap in 12 hours -/
def overlap_count : ℕ := 11

/-- Represents the number of times the clock hands form right angles in 12 hours -/
def right_angle_count : ℕ := 22

/-- Represents the number of times the clock hands form straight angles in 12 hours -/
def straight_angle_count : ℕ := 11

/-- Proves that the clock hands overlap, form right angles, and straight angles
    the specified number of times in a 12-hour period -/
theorem clock_hand_positions :
  overlap_count = 11 ∧
  right_angle_count = 22 ∧
  straight_angle_count = 11 :=
by sorry

end clock_hand_positions_l2711_271159


namespace f_g_3_equals_28_l2711_271118

-- Define the functions f and g
def g (x : ℝ) : ℝ := x^2 + 1
def f (x : ℝ) : ℝ := 3*x - 2

-- State the theorem
theorem f_g_3_equals_28 : f (g 3) = 28 := by
  sorry

end f_g_3_equals_28_l2711_271118


namespace triangle_angle_measure_l2711_271103

theorem triangle_angle_measure (P Q R : ℝ) (h1 : R = 3 * Q) (h2 : Q = 30) :
  P + Q + R = 180 → P = 60 := by
  sorry

end triangle_angle_measure_l2711_271103


namespace monochromatic_triangle_in_K17_l2711_271125

/-- A coloring of the edges of a complete graph -/
def Coloring (n : ℕ) := Fin n → Fin n → Fin 3

/-- A triangle in a graph is a set of three distinct vertices -/
def Triangle (n : ℕ) := { t : Fin n × Fin n × Fin n // t.1 ≠ t.2.1 ∧ t.1 ≠ t.2.2 ∧ t.2.1 ≠ t.2.2 }

/-- A triangle is monochromatic if all its edges have the same color -/
def IsMonochromatic (n : ℕ) (c : Coloring n) (t : Triangle n) : Prop :=
  c t.val.1 t.val.2.1 = c t.val.1 t.val.2.2 ∧ 
  c t.val.1 t.val.2.1 = c t.val.2.1 t.val.2.2

/-- The main theorem: any 3-coloring of K_17 contains a monochromatic triangle -/
theorem monochromatic_triangle_in_K17 :
  ∀ (c : Coloring 17), ∃ (t : Triangle 17), IsMonochromatic 17 c t :=
sorry


end monochromatic_triangle_in_K17_l2711_271125


namespace crank_slider_motion_l2711_271152

/-- Crank-slider mechanism -/
structure CrankSlider where
  oa : ℝ
  ab : ℝ
  mb : ℝ
  ω : ℝ

/-- Point coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Velocity vector -/
structure Velocity where
  vx : ℝ
  vy : ℝ

/-- Motion equations for point M -/
def motionEquations (cs : CrankSlider) (t : ℝ) : Point :=
  sorry

/-- Trajectory equation for point M -/
def trajectoryEquation (cs : CrankSlider) (x : ℝ) (y : ℝ) : Prop :=
  sorry

/-- Velocity of point M -/
def velocityM (cs : CrankSlider) (t : ℝ) : Velocity :=
  sorry

theorem crank_slider_motion 
  (cs : CrankSlider) 
  (h1 : cs.oa = 90) 
  (h2 : cs.ab = 90) 
  (h3 : cs.mb = cs.ab / 3) 
  (h4 : cs.ω = 10) :
  ∃ (me : ℝ → Point) (te : ℝ → ℝ → Prop) (ve : ℝ → Velocity),
    me = motionEquations cs ∧
    te = trajectoryEquation cs ∧
    ve = velocityM cs :=
  sorry

end crank_slider_motion_l2711_271152


namespace max_value_theorem_l2711_271184

/-- Given a quadratic function y = ax² + x - b where a > 0 and b > 1,
    if the solution set P of y > 0 intersects with Q = {x | -2-t < x < -2+t}
    for all positive t, then the maximum value of 1/a - 1/b is 1/2. -/
theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 1) :
  (∀ t > 0, ∃ x, (a * x^2 + x - b > 0 ∧ -2 - t < x ∧ x < -2 + t)) →
  (∃ m, m = 1/a - 1/b ∧ ∀ a' b', a' > 0 → b' > 1 →
    (∀ t > 0, ∃ x, (a' * x^2 + x - b' > 0 ∧ -2 - t < x ∧ x < -2 + t)) →
    1/a' - 1/b' ≤ m) ∧
  m = 1/2 := by
  sorry

end max_value_theorem_l2711_271184


namespace unique_triplet_solution_l2711_271162

theorem unique_triplet_solution : 
  ∃! (x y z : ℕ+), 
    (x^y.val + y.val^x.val = z.val^y.val) ∧ 
    (x^y.val + 2012 = y.val^(z.val + 1)) ∧
    x = 6 ∧ y = 2 ∧ z = 10 := by
  sorry

end unique_triplet_solution_l2711_271162


namespace paulines_garden_rows_l2711_271122

/-- Represents Pauline's garden --/
structure Garden where
  tomatoes : ℕ
  cucumbers : ℕ
  potatoes : ℕ
  extra_capacity : ℕ
  spaces_per_row : ℕ

/-- Calculates the number of rows in the garden --/
def number_of_rows (g : Garden) : ℕ :=
  (g.tomatoes + g.cucumbers + g.potatoes + g.extra_capacity) / g.spaces_per_row

/-- Theorem: The number of rows in Pauline's garden is 10 --/
theorem paulines_garden_rows :
  let g : Garden := {
    tomatoes := 3 * 5,
    cucumbers := 5 * 4,
    potatoes := 30,
    extra_capacity := 85,
    spaces_per_row := 15
  }
  number_of_rows g = 10 := by
  sorry

end paulines_garden_rows_l2711_271122


namespace permutation_difference_l2711_271130

def permutation (n : ℕ) (r : ℕ) : ℕ :=
  (n - r + 1).factorial / (n - r).factorial

theorem permutation_difference : permutation 8 4 - 2 * permutation 8 2 = 1568 := by
  sorry

end permutation_difference_l2711_271130


namespace max_a_2016_gt_44_l2711_271128

/-- Definition of the sequence a_{n,k} -/
def a (n k : ℕ) : ℝ :=
  sorry

/-- The maximum value of a_{n,k} for a given n -/
def m (n : ℕ) : ℝ :=
  sorry

/-- The main theorem to be proved -/
theorem max_a_2016_gt_44 
  (h1 : ∀ k, 1 ≤ k ∧ k ≤ 2016 → 0 < a 0 k)
  (h2 : ∀ n k, n ≥ 0 ∧ 1 ≤ k ∧ k < 2016 → a (n+1) k = a n k + 1 / (2 * a n (k+1)))
  (h3 : ∀ n, n ≥ 0 → a (n+1) 2016 = a n 2016 + 1 / (2 * a n 1)) :
  m 2016 > 44 :=
sorry

end max_a_2016_gt_44_l2711_271128


namespace existence_of_sequence_l2711_271147

/-- Given positive integers a and b where b > a > 1 and a does not divide b,
    as well as a sequence of positive integers b_n such that b_{n+1} ≥ 2b_n for all n,
    there exists a sequence of positive integers a_n satisfying certain conditions. -/
theorem existence_of_sequence (a b : ℕ) (b_seq : ℕ → ℕ) 
  (h_a_pos : a > 0) (h_b_pos : b > 0) (h_b_gt_a : b > a) (h_a_gt_1 : a > 1)
  (h_a_not_div_b : ¬ (b % a = 0))
  (h_b_seq_growth : ∀ n : ℕ, b_seq (n + 1) ≥ 2 * b_seq n) :
  ∃ a_seq : ℕ → ℕ, 
    (∀ n : ℕ, (a_seq (n + 1) - a_seq n = a) ∨ (a_seq (n + 1) - a_seq n = b)) ∧
    (∀ m l : ℕ, ∀ n : ℕ, a_seq m + a_seq l ≠ b_seq n) :=
sorry

end existence_of_sequence_l2711_271147


namespace expense_reduction_equation_l2711_271196

/-- Represents the average monthly reduction rate as a real number between 0 and 1 -/
def reduction_rate : ℝ := sorry

/-- The initial monthly expenses in yuan -/
def initial_expenses : ℝ := 2500

/-- The final monthly expenses after two months in yuan -/
def final_expenses : ℝ := 1600

/-- The number of months over which the reduction occurred -/
def num_months : ℕ := 2

theorem expense_reduction_equation :
  initial_expenses * (1 - reduction_rate) ^ num_months = final_expenses :=
sorry

end expense_reduction_equation_l2711_271196


namespace purple_balls_count_l2711_271112

/-- Represents the number of green balls in the bin -/
def green_balls : ℕ := 5

/-- Represents the win amount for drawing a green ball -/
def green_win : ℚ := 2

/-- Represents the loss amount for drawing a purple ball -/
def purple_loss : ℚ := 2

/-- Represents the expected winnings -/
def expected_win : ℚ := (1 : ℚ) / 2

/-- 
Given a bin with 5 green balls and k purple balls, where k is a positive integer,
and a game where drawing a green ball wins 2 dollars and drawing a purple ball loses 2 dollars,
if the expected amount won is 50 cents, then k must equal 3.
-/
theorem purple_balls_count (k : ℕ+) : 
  (green_balls : ℚ) / (green_balls + k) * green_win + 
  (k : ℚ) / (green_balls + k) * (-purple_loss) = expected_win → 
  k = 3 := by
  sorry


end purple_balls_count_l2711_271112


namespace equation_solution_l2711_271191

theorem equation_solution : ∃ x : ℚ, (2/7) * (1/8) * x - 4 = 12 ∧ x = 448 := by
  sorry

end equation_solution_l2711_271191


namespace stratified_sampling_group_size_l2711_271195

theorem stratified_sampling_group_size 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (group_a_sample : ℕ) :
  total_population = 200 →
  sample_size = 40 →
  group_a_sample = 16 →
  (total_population - (total_population * group_a_sample / sample_size) : ℕ) = 120 :=
by sorry

end stratified_sampling_group_size_l2711_271195


namespace jordan_rectangle_length_l2711_271151

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem jordan_rectangle_length : 
  ∀ (carol jordan : Rectangle),
    carol.length = 5 →
    carol.width = 24 →
    jordan.width = 60 →
    area carol = area jordan →
    jordan.length = 2 := by
  sorry

end jordan_rectangle_length_l2711_271151


namespace max_value_of_f_l2711_271115

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 4) * (x - a)

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x - 4

theorem max_value_of_f (a : ℝ) :
  (f' a (-1) = 0) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 4, f a x = 42) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 4, f a x ≤ 42) :=
by
  sorry

#check max_value_of_f

end max_value_of_f_l2711_271115


namespace pauline_snow_shoveling_l2711_271183

/-- Calculates the total volume of snow shoveled up to a given hour -/
def snowShoveled (hour : ℕ) : ℕ :=
  (20 * hour) - (hour * (hour - 1) / 2)

/-- Represents Pauline's snow shoveling problem -/
theorem pauline_snow_shoveling (drivewayWidth drivewayLength snowDepth : ℕ) 
  (h1 : drivewayWidth = 5)
  (h2 : drivewayLength = 10)
  (h3 : snowDepth = 4) :
  ∃ (hour : ℕ), hour = 13 ∧ snowShoveled hour ≥ drivewayWidth * drivewayLength * snowDepth ∧ 
  snowShoveled (hour - 1) < drivewayWidth * drivewayLength * snowDepth :=
by
  sorry


end pauline_snow_shoveling_l2711_271183


namespace pears_cost_l2711_271111

theorem pears_cost (initial_amount : ℕ) (banana_cost : ℕ) (banana_packs : ℕ) (asparagus_cost : ℕ) (chicken_cost : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 55 →
  banana_cost = 4 →
  banana_packs = 2 →
  asparagus_cost = 6 →
  chicken_cost = 11 →
  remaining_amount = 28 →
  initial_amount - (banana_cost * banana_packs + asparagus_cost + chicken_cost + remaining_amount) = 2 :=
by
  sorry

#check pears_cost

end pears_cost_l2711_271111


namespace negative_three_hash_six_l2711_271175

/-- The '#' operation for rational numbers -/
def hash (a b : ℚ) : ℚ := a^2 + a*b - 5

/-- Theorem: (-3)#6 = -14 -/
theorem negative_three_hash_six : hash (-3) 6 = -14 := by sorry

end negative_three_hash_six_l2711_271175


namespace abs_sum_range_l2711_271143

theorem abs_sum_range : 
  (∀ x : ℝ, |x + 2| + |x + 3| ≥ 1) ∧ 
  (∃ y : ℝ, ∀ ε > 0, ∃ x : ℝ, |x + 2| + |x + 3| < y + ε) ∧ 
  y = 1 := by
  sorry

end abs_sum_range_l2711_271143


namespace geometric_series_first_term_l2711_271194

theorem geometric_series_first_term 
  (r : ℝ) (S : ℝ) (a : ℝ) 
  (h1 : r = 1/4) 
  (h2 : S = 80) 
  (h3 : S = a / (1 - r)) : 
  a = 60 := by sorry

end geometric_series_first_term_l2711_271194


namespace at_op_difference_l2711_271100

-- Define the @ operation
def at_op (x y : ℤ) : ℤ := x * y - 2 * x

-- State the theorem
theorem at_op_difference : (at_op 5 3) - (at_op 3 5) = -4 := by
  sorry

end at_op_difference_l2711_271100


namespace petyas_journey_fraction_l2711_271124

/-- The fraction of the journey Petya completed before remembering his pen -/
def journey_fraction (total_time walking_time early_arrival late_arrival : ℚ) : ℚ :=
  walking_time / total_time

theorem petyas_journey_fraction :
  let total_time : ℚ := 20
  let early_arrival : ℚ := 3
  let late_arrival : ℚ := 7
  ∃ (walking_time : ℚ),
    journey_fraction total_time walking_time early_arrival late_arrival = 1/4 :=
by sorry

end petyas_journey_fraction_l2711_271124


namespace joe_lifts_l2711_271177

theorem joe_lifts (total_weight first_lift : ℕ) 
  (h1 : total_weight = 900)
  (h2 : first_lift = 400) :
  total_weight - first_lift = first_lift + 100 := by
  sorry

end joe_lifts_l2711_271177


namespace log_expression_equality_l2711_271113

theorem log_expression_equality : 
  Real.sqrt (Real.log 18 / Real.log 4 - Real.log 18 / Real.log 9 + Real.log 9 / Real.log 2) = 
  (3 * Real.log 3 - Real.log 2) / Real.sqrt (2 * Real.log 3 * Real.log 2) := by
  sorry

end log_expression_equality_l2711_271113


namespace expression_simplification_l2711_271131

theorem expression_simplification 
  (a b c x : ℝ) 
  (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) : 
  (x - a)^3 + a*x / ((a - b)*(a - c)) + 
  (x - b)^3 + b*x / ((b - a)*(b - c)) + 
  (x - c)^3 + c*x / ((c - a)*(c - b)) = 
  a + b + c + 3*x + 1 := by
  sorry

end expression_simplification_l2711_271131


namespace initially_tagged_fish_count_l2711_271126

/-- The number of fish initially caught and tagged in a pond -/
def initially_tagged_fish (total_fish : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) : ℕ :=
  (tagged_in_second * total_fish) / second_catch

/-- Theorem stating that the number of initially tagged fish is 50 -/
theorem initially_tagged_fish_count :
  initially_tagged_fish 250 50 10 = 50 := by
  sorry

#eval initially_tagged_fish 250 50 10

end initially_tagged_fish_count_l2711_271126


namespace inverse_inequality_l2711_271192

theorem inverse_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 1 / a > 1 / b := by
  sorry

end inverse_inequality_l2711_271192
