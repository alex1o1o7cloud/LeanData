import Mathlib

namespace triangle_vector_parallel_l2932_293255

theorem triangle_vector_parallel (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  (a, Real.sqrt 3 * b) = (Real.cos A, Real.sin B) →
  A = π / 3 ∧
  (a = 2 → 2 < b + c ∧ b + c ≤ 4) :=
by sorry

end triangle_vector_parallel_l2932_293255


namespace min_value_theorem_min_value_achieved_l2932_293248

theorem min_value_theorem (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a + b = 1) :
  (1 / (a + 2*b)) + (4 / (2*a + b)) ≥ 3 :=
by sorry

theorem min_value_achieved (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a + b = 1) :
  ∃ (a₀ b₀ : ℝ), a₀ ≥ 0 ∧ b₀ ≥ 0 ∧ a₀ + b₀ = 1 ∧ (1 / (a₀ + 2*b₀)) + (4 / (2*a₀ + b₀)) = 3 :=
by sorry

end min_value_theorem_min_value_achieved_l2932_293248


namespace probability_different_classes_l2932_293283

/-- The probability of selecting two students from different language classes -/
theorem probability_different_classes (total : ℕ) (german : ℕ) (chinese : ℕ) 
  (h1 : total = 30)
  (h2 : german = 22)
  (h3 : chinese = 19)
  (h4 : german + chinese - total ≥ 0) : 
  (Nat.choose total 2 - (Nat.choose (german + chinese - total) 2 + 
   Nat.choose (german - (german + chinese - total)) 2 + 
   Nat.choose (chinese - (german + chinese - total)) 2)) / Nat.choose total 2 = 352 / 435 := by
  sorry

end probability_different_classes_l2932_293283


namespace decagon_diagonals_l2932_293216

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular decagon has 35 diagonals -/
theorem decagon_diagonals : num_diagonals 10 = 35 := by
  sorry

end decagon_diagonals_l2932_293216


namespace sqrt_two_times_sqrt_six_equals_two_sqrt_three_l2932_293203

theorem sqrt_two_times_sqrt_six_equals_two_sqrt_three :
  Real.sqrt 2 * Real.sqrt 6 = 2 * Real.sqrt 3 := by
  sorry

end sqrt_two_times_sqrt_six_equals_two_sqrt_three_l2932_293203


namespace prove_weights_l2932_293209

/-- Represents a weighing device that signals when the total weight is 46 kg -/
def WeighingDevice (weights : List Nat) : Bool :=
  weights.sum = 46

/-- Represents the set of ingots with weights from 1 to 13 kg -/
def Ingots : List Nat := List.range 13 |>.map (· + 1)

/-- Checks if a given list of weights is a subset of the Ingots -/
def IsValidSelection (selection : List Nat) : Bool :=
  selection.all (· ∈ Ingots) ∧ selection.length ≤ Ingots.length

theorem prove_weights :
  ∃ (selection1 selection2 : List Nat),
    IsValidSelection selection1 ∧
    IsValidSelection selection2 ∧
    WeighingDevice selection1 ∧
    WeighingDevice selection2 ∧
    (9 ∈ selection1 ∨ 9 ∈ selection2) ∧
    (10 ∈ selection1 ∨ 10 ∈ selection2) :=
  sorry

end prove_weights_l2932_293209


namespace regular_polygon_perimeter_l2932_293240

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) :
  side_length = 7 ∧ exterior_angle = 45 →
  (360 / exterior_angle) * side_length = 56 :=
by sorry

end regular_polygon_perimeter_l2932_293240


namespace h2so4_equals_khso4_l2932_293254

/-- Represents the balanced chemical equation for the reaction between KOH and H2SO4 to form KHSO4 -/
structure ChemicalReaction where
  koh : ℝ
  h2so4 : ℝ
  khso4 : ℝ

/-- The theorem states that the number of moles of H2SO4 needed is equal to the number of moles of KHSO4 formed,
    given that the number of moles of KOH initially present is equal to the number of moles of KHSO4 formed -/
theorem h2so4_equals_khso4 (reaction : ChemicalReaction) 
    (h : reaction.koh = reaction.khso4) : reaction.h2so4 = reaction.khso4 := by
  sorry

#check h2so4_equals_khso4

end h2so4_equals_khso4_l2932_293254


namespace william_max_moves_l2932_293235

/-- Represents a player in the game -/
inductive Player : Type
| Mark : Player
| William : Player

/-- Represents a move in the game -/
inductive Move : Type
| Double : Move  -- Multiply by 2 and add 1
| Quadruple : Move  -- Multiply by 4 and add 3

/-- Applies a move to the current value -/
def applyMove (value : ℕ) (move : Move) : ℕ :=
  match move with
  | Move.Double => 2 * value + 1
  | Move.Quadruple => 4 * value + 3

/-- Checks if the game is over -/
def isGameOver (value : ℕ) : Prop :=
  value > 2^100

/-- Represents the state of the game -/
structure GameState :=
  (value : ℕ)
  (currentPlayer : Player)

/-- Represents an optimal strategy for the game -/
def OptimalStrategy : Type :=
  GameState → Move

/-- The maximum number of moves William can make -/
def maxWilliamMoves : ℕ := 33

/-- The main theorem to be proved -/
theorem william_max_moves 
  (strategy : OptimalStrategy) : 
  ∃ (game : List Move), 
    game.length = 2 * maxWilliamMoves + 1 ∧ 
    isGameOver (game.foldl applyMove 1) ∧
    ∀ (game' : List Move), 
      game'.length > 2 * maxWilliamMoves + 1 → 
      ¬isGameOver (game'.foldl applyMove 1) :=
sorry

end william_max_moves_l2932_293235


namespace cubic_equation_implies_square_l2932_293288

theorem cubic_equation_implies_square (y : ℝ) : 
  2 * y^3 + 3 * y^2 - 2 * y - 8 = 0 → (5 * y - 2)^2 = 64 := by
  sorry

end cubic_equation_implies_square_l2932_293288


namespace girls_fraction_at_joint_event_l2932_293242

/-- Represents a middle school with a given number of students and boy-to-girl ratio --/
structure MiddleSchool where
  total_students : ℕ
  boy_ratio : ℕ
  girl_ratio : ℕ

/-- Calculates the number of girls in a middle school --/
def girls_count (school : MiddleSchool) : ℚ :=
  (school.total_students : ℚ) * school.girl_ratio / (school.boy_ratio + school.girl_ratio)

/-- The fraction of girls at a joint event of two middle schools --/
def girls_fraction (school1 school2 : MiddleSchool) : ℚ :=
  (girls_count school1 + girls_count school2) / (school1.total_students + school2.total_students)

theorem girls_fraction_at_joint_event :
  let jasper_creek : MiddleSchool := { total_students := 360, boy_ratio := 7, girl_ratio := 5 }
  let brookstone : MiddleSchool := { total_students := 240, boy_ratio := 3, girl_ratio := 5 }
  girls_fraction jasper_creek brookstone = 1/2 := by
  sorry

end girls_fraction_at_joint_event_l2932_293242


namespace oliver_video_games_l2932_293234

/-- The number of working video games Oliver bought -/
def working_games : ℕ := 6

/-- The number of bad video games Oliver bought -/
def bad_games : ℕ := 5

/-- The total number of video games Oliver bought -/
def total_games : ℕ := working_games + bad_games

theorem oliver_video_games : 
  total_games = working_games + bad_games := by sorry

end oliver_video_games_l2932_293234


namespace smallest_b_for_divisibility_l2932_293295

def is_single_digit (n : ℕ) : Prop := n < 10

def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem smallest_b_for_divisibility : 
  ∃ (B : ℕ), is_single_digit B ∧ 
             is_divisible_by_13 (200 + 10 * B + 5) ∧ 
             (∀ (k : ℕ), k < B → ¬(is_divisible_by_13 (200 + 10 * k + 5))) ∧
             B = 12 := by
  sorry

end smallest_b_for_divisibility_l2932_293295


namespace geometric_progression_value_l2932_293226

def is_geometric_progression (a b c : ℝ) : Prop :=
  b ^ 2 = a * c

theorem geometric_progression_value :
  ∃ x : ℝ, is_geometric_progression (30 + x) (70 + x) (150 + x) ∧ x = 10 := by
  sorry

end geometric_progression_value_l2932_293226


namespace no_valid_subset_exists_l2932_293224

/-- The set M defined as the intersection of (0,1) and ℚ -/
def M : Set ℚ := Set.Ioo 0 1 ∩ Set.range Rat.cast

/-- Definition of a valid subset A -/
def is_valid_subset (A : Set ℚ) : Prop :=
  A ⊆ M ∧
  ∀ x ∈ M, ∃! (S : Finset ℚ), (S : Set ℚ) ⊆ A ∧ x = S.sum id

/-- Theorem stating that no valid subset A exists -/
theorem no_valid_subset_exists : ¬∃ A : Set ℚ, is_valid_subset A := by
  sorry

end no_valid_subset_exists_l2932_293224


namespace binomial_6_choose_2_l2932_293269

theorem binomial_6_choose_2 : Nat.choose 6 2 = 15 := by
  sorry

end binomial_6_choose_2_l2932_293269


namespace swimming_pool_volume_l2932_293270

/-- Calculate the volume of a trapezoidal prism-shaped swimming pool -/
theorem swimming_pool_volume 
  (width : ℝ) 
  (length : ℝ) 
  (shallow_depth : ℝ) 
  (deep_depth : ℝ) 
  (h_width : width = 9) 
  (h_length : length = 12) 
  (h_shallow : shallow_depth = 1) 
  (h_deep : deep_depth = 4) : 
  (1 / 2) * (shallow_depth + deep_depth) * width * length = 270 := by
sorry

end swimming_pool_volume_l2932_293270


namespace quadratic_equation_coefficients_l2932_293261

theorem quadratic_equation_coefficients (b c : ℝ) :
  (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = 1 ∨ x = -7) →
  (∀ x : ℝ, |x + 3| = 4 ↔ x = 1 ∨ x = -7) →
  b = 6 ∧ c = -7 := by
  sorry

end quadratic_equation_coefficients_l2932_293261


namespace shaded_area_hexagon_with_semicircles_l2932_293250

/-- The area of the shaded region in a regular hexagon with inscribed semicircles -/
theorem shaded_area_hexagon_with_semicircles (s : ℝ) (h : s = 3) :
  let hexagon_area := 3 * Real.sqrt 3 / 2 * s^2
  let semicircle_area := π * (s/2)^2 / 2
  let total_semicircle_area := 3 * semicircle_area
  hexagon_area - total_semicircle_area = 13.5 * Real.sqrt 3 - 27 * π / 8 := by
  sorry

end shaded_area_hexagon_with_semicircles_l2932_293250


namespace expression_evaluation_l2932_293201

theorem expression_evaluation (a b : ℝ) 
  (h : (a + 1/2)^2 + |b - 2| = 0) : 
  5*(3*a^2*b - a*b^2) - (a*b^2 + 3*a^2*b) = 18 := by
sorry

end expression_evaluation_l2932_293201


namespace sequence_properties_l2932_293215

def sequence_a (n : ℕ) : ℝ := sorry

def sum_S (n : ℕ) : ℝ := sorry

def sequence_T (n : ℕ) : ℝ := sorry

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → sum_S n = 2 * sequence_a n - sequence_a 1) ∧
  (sequence_a 1 + sequence_a 3 = 2 * (sequence_a 2 + 1)) →
  (∀ n : ℕ, n > 0 → sequence_a n = 2^n) ∧
  (∀ n : ℕ, n > 0 → sequence_T n = 2 - (n + 2) / 2^n) :=
by sorry

end sequence_properties_l2932_293215


namespace spinner_probability_l2932_293299

theorem spinner_probability (pA pB pC pD : ℚ) : 
  pA = 1/4 → pB = 1/3 → pD = 1/6 → pA + pB + pC + pD = 1 → pC = 1/4 := by
  sorry

end spinner_probability_l2932_293299


namespace jenny_spent_fraction_l2932_293273

theorem jenny_spent_fraction (initial_amount : ℚ) : 
  (initial_amount / 2 = 21) →
  (initial_amount - 24 > 0) →
  ((initial_amount - 24) / initial_amount = 3/7) := by
  sorry

end jenny_spent_fraction_l2932_293273


namespace custom_op_example_l2932_293241

/-- Custom operation ⊕ for rational numbers -/
def custom_op (a b : ℚ) : ℚ := a * b + (a - b)

/-- Theorem stating that (-5) ⊕ 4 = -29 -/
theorem custom_op_example : custom_op (-5) 4 = -29 := by
  sorry

end custom_op_example_l2932_293241


namespace z3_magnitude_range_l2932_293276

/-- Given complex numbers satisfying certain conditions, prove the range of the magnitude of z₃ -/
theorem z3_magnitude_range (z₁ z₂ z₃ : ℂ) 
  (h1 : Complex.abs z₁ = Real.sqrt 2)
  (h2 : Complex.abs z₂ = Real.sqrt 2)
  (h3 : (z₁.re * z₂.re + z₁.im * z₂.im) = 0)
  (h4 : Complex.abs (z₁ + z₂ - z₃) = 2) :
  0 ≤ Complex.abs z₃ ∧ Complex.abs z₃ ≤ 4 := by sorry

end z3_magnitude_range_l2932_293276


namespace number_problem_l2932_293259

theorem number_problem (x : ℝ) : 0.65 * x = 0.8 * x - 21 → x = 140 := by
  sorry

end number_problem_l2932_293259


namespace glasses_cost_l2932_293278

theorem glasses_cost (frame_cost : ℝ) (coupon : ℝ) (insurance_coverage : ℝ) (total_cost : ℝ) :
  frame_cost = 200 →
  coupon = 50 →
  insurance_coverage = 0.8 →
  total_cost = 250 →
  ∃ (lens_cost : ℝ), lens_cost = 500 ∧
    total_cost = (frame_cost - coupon) + (1 - insurance_coverage) * lens_cost :=
by sorry

end glasses_cost_l2932_293278


namespace no_special_subset_exists_l2932_293200

theorem no_special_subset_exists : ¬∃ (M : Set ℝ), 
  (M.Nonempty) ∧ 
  (∀ (r : ℝ) (a : ℝ), r > 0 → a ∈ M → 
    ∃! (b : ℝ), b ∈ M ∧ |a - b| = r) := by
  sorry

end no_special_subset_exists_l2932_293200


namespace continuous_fraction_value_l2932_293204

theorem continuous_fraction_value :
  ∃ x : ℝ, x = 1 + 1 / (2 + 1 / x) ∧ x = (Real.sqrt 3 + 1) / 2 := by
  sorry

end continuous_fraction_value_l2932_293204


namespace find_number_l2932_293298

theorem find_number : ∃ (x : ℝ), 5 + x * (8 - 3) = 15 ∧ x = 2 := by
  sorry

end find_number_l2932_293298


namespace three_over_x_equals_one_l2932_293289

theorem three_over_x_equals_one (x : ℝ) (h : 1 - 6/x + 9/x^2 = 0) : 3/x = 1 := by
  sorry

end three_over_x_equals_one_l2932_293289


namespace product_of_fractions_l2932_293284

theorem product_of_fractions : 
  (10 : ℚ) / 6 * 4 / 20 * 20 / 12 * 16 / 32 * 40 / 24 * 8 / 40 * 60 / 36 * 32 / 64 = 25 / 324 := by
  sorry

end product_of_fractions_l2932_293284


namespace sneakers_cost_l2932_293239

/-- The cost of sneakers given initial savings, action figure sales, and remaining money --/
theorem sneakers_cost
  (initial_savings : ℕ)
  (action_figures_sold : ℕ)
  (price_per_figure : ℕ)
  (money_left : ℕ)
  (h1 : initial_savings = 15)
  (h2 : action_figures_sold = 10)
  (h3 : price_per_figure = 10)
  (h4 : money_left = 25) :
  initial_savings + action_figures_sold * price_per_figure - money_left = 90 := by
  sorry

end sneakers_cost_l2932_293239


namespace range_of_a_l2932_293292

def p (x : ℝ) : Prop := |x + 1| > 3

def q (x a : ℝ) : Prop := x > a

theorem range_of_a (h1 : ∀ x, q x a → p x) 
                   (h2 : ∃ x, p x ∧ ¬q x a) : 
  a ≥ 2 := by sorry

end range_of_a_l2932_293292


namespace cyclic_power_inequality_l2932_293253

theorem cyclic_power_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a^4*b + b^4*c + c^4*d + d^4*a ≥ a*b*c*d*(a + b + c + d) := by
  sorry

end cyclic_power_inequality_l2932_293253


namespace total_coins_last_month_l2932_293268

/-- The number of coins Mathilde had at the start of this month -/
def mathilde_this_month : ℕ := 100

/-- The number of coins Salah had at the start of this month -/
def salah_this_month : ℕ := 100

/-- The percentage increase in Mathilde's coins from last month to this month -/
def mathilde_increase : ℚ := 25/100

/-- The percentage decrease in Salah's coins from last month to this month -/
def salah_decrease : ℚ := 20/100

/-- Theorem stating that the total number of coins Mathilde and Salah had at the start of last month was 205 -/
theorem total_coins_last_month : 
  ∃ (mathilde_last_month salah_last_month : ℕ),
    (mathilde_this_month : ℚ) = mathilde_last_month * (1 + mathilde_increase) ∧
    (salah_this_month : ℚ) = salah_last_month * (1 - salah_decrease) ∧
    mathilde_last_month + salah_last_month = 205 := by
  sorry

end total_coins_last_month_l2932_293268


namespace button_fraction_proof_l2932_293208

theorem button_fraction_proof (mari kendra sue : ℕ) : 
  mari = 5 * kendra + 4 →
  mari = 64 →
  sue = 6 →
  sue / kendra = 1 / 2 := by
sorry

end button_fraction_proof_l2932_293208


namespace apples_to_eat_raw_l2932_293296

theorem apples_to_eat_raw (total : ℕ) (wormy : ℕ) (bruised : ℕ) : 
  total = 85 → 
  wormy = total / 5 →
  bruised = wormy + 9 →
  total - wormy - bruised = 42 := by
sorry

end apples_to_eat_raw_l2932_293296


namespace cubic_root_sum_product_l2932_293256

theorem cubic_root_sum_product (α β : ℝ) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let p : ℝ → ℝ := λ x => α * x^3 - α * x^2 + β * x + β
  ∀ x₁ x₂ x₃ : ℝ, (p x₁ = 0 ∧ p x₂ = 0 ∧ p x₃ = 0 ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) →
    (x₁ + x₂ + x₃) * (1/x₁ + 1/x₂ + 1/x₃) = -1 := by
  sorry

end cubic_root_sum_product_l2932_293256


namespace reciprocal_of_point_two_l2932_293212

theorem reciprocal_of_point_two (x : ℝ) : x = 0.2 → 1 / x = 5 := by
  sorry

end reciprocal_of_point_two_l2932_293212


namespace geometric_sequence_sum_l2932_293206

/-- The sum of all terms in the geometric sequence {(2/3)^n, n ∈ ℕ*} is 2. -/
theorem geometric_sequence_sum : 
  let a : ℕ → ℝ := fun n => (2/3)^n
  ∑' n, a n = 2 := by sorry

end geometric_sequence_sum_l2932_293206


namespace rationalize_and_simplify_l2932_293281

theorem rationalize_and_simplify :
  (32 / Real.sqrt 8) + (8 / Real.sqrt 32) = 9 * Real.sqrt 2 := by
  sorry

end rationalize_and_simplify_l2932_293281


namespace probability_sum_seven_l2932_293221

def die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}

def sum_is_seven (roll : ℕ × ℕ) : Prop :=
  roll.1 + roll.2 = 7

def favorable_outcomes : Finset (ℕ × ℕ) :=
  {(1, 6), (6, 1), (2, 5), (5, 2), (3, 4), (4, 3)}

theorem probability_sum_seven :
  (Finset.card favorable_outcomes : ℚ) / (Finset.card (die_faces.product die_faces) : ℚ) = 1 / 6 := by
  sorry

end probability_sum_seven_l2932_293221


namespace train_length_calculation_l2932_293243

/-- Prove that a train traveling at 60 km/hour that takes 30 seconds to pass a bridge of 140 meters in length has a length of approximately 360.1 meters. -/
theorem train_length_calculation (train_speed : ℝ) (bridge_pass_time : ℝ) (bridge_length : ℝ) :
  train_speed = 60 →
  bridge_pass_time = 30 →
  bridge_length = 140 →
  ∃ (train_length : ℝ), abs (train_length - 360.1) < 0.1 :=
by sorry

end train_length_calculation_l2932_293243


namespace max_sum_of_factors_48_l2932_293287

theorem max_sum_of_factors_48 :
  ∃ (a b : ℕ), a * b = 48 ∧ a + b = 49 ∧
  ∀ (x y : ℕ), x * y = 48 → x + y ≤ 49 := by
  sorry

end max_sum_of_factors_48_l2932_293287


namespace three_solutions_at_plus_minus_one_two_solutions_at_plus_minus_sqrt_two_four_solutions_between_neg_sqrt_two_and_sqrt_two_no_solutions_outside_sqrt_two_l2932_293219

/-- The number of solutions to the system of equations
    x^2 - y^2 = 0 and (x-a)^2 + y^2 = 1 -/
def num_solutions (a : ℝ) : ℕ :=
  sorry

/-- The system has three solutions when a = ±1 -/
theorem three_solutions_at_plus_minus_one :
  (num_solutions 1 = 3) ∧ (num_solutions (-1) = 3) :=
sorry

/-- The system has two solutions when a = ±√2 -/
theorem two_solutions_at_plus_minus_sqrt_two :
  (num_solutions (Real.sqrt 2) = 2) ∧ (num_solutions (-(Real.sqrt 2)) = 2) :=
sorry

/-- The system has four solutions for all other values of a in (-√2, √2) except ±1 -/
theorem four_solutions_between_neg_sqrt_two_and_sqrt_two (a : ℝ) :
  a ∈ Set.Ioo (-(Real.sqrt 2)) (Real.sqrt 2) ∧ a ≠ 1 ∧ a ≠ -1 →
  num_solutions a = 4 :=
sorry

/-- The system has no solutions for |a| > √2 -/
theorem no_solutions_outside_sqrt_two (a : ℝ) :
  |a| > Real.sqrt 2 → num_solutions a = 0 :=
sorry

end three_solutions_at_plus_minus_one_two_solutions_at_plus_minus_sqrt_two_four_solutions_between_neg_sqrt_two_and_sqrt_two_no_solutions_outside_sqrt_two_l2932_293219


namespace randy_baseball_gloves_l2932_293264

theorem randy_baseball_gloves (bats : ℕ) (gloves : ℕ) : 
  bats = 4 → gloves = 7 * bats + 1 → gloves = 29 := by
  sorry

end randy_baseball_gloves_l2932_293264


namespace point_coordinate_sum_l2932_293275

/-- Given two points A(0,0) and B(x,-3) where the slope of AB is 4/5, 
    the sum of B's coordinates is -27/4 -/
theorem point_coordinate_sum (x : ℚ) : 
  let A : ℚ × ℚ := (0, 0)
  let B : ℚ × ℚ := (x, -3)
  let slope : ℚ := (B.2 - A.2) / (B.1 - A.1)
  slope = 4/5 → x + B.2 = -27/4 := by
  sorry

end point_coordinate_sum_l2932_293275


namespace max_true_statements_l2932_293260

theorem max_true_statements (a b : ℝ) : 
  let statements := [
    (1/a > 1/b),
    (a^2 < b^2),
    (a > b),
    (a > 0),
    (b > 0)
  ]
  ∃ (trueStatements : List Bool), 
    trueStatements.length ≤ 3 ∧ 
    ∀ (i : Nat), i < statements.length → 
      (trueStatements.get? i = some true → statements.get! i) ∧
      (statements.get! i → trueStatements.get? i = some true) :=
sorry

end max_true_statements_l2932_293260


namespace solve_for_S_l2932_293246

theorem solve_for_S : ∃ S : ℚ, (1/3 : ℚ) * (1/8 : ℚ) * S = (1/4 : ℚ) * (1/6 : ℚ) * 180 ∧ S = 180 := by
  sorry

end solve_for_S_l2932_293246


namespace boat_license_combinations_l2932_293213

/-- The number of possible letters for a boat license. -/
def num_letters : ℕ := 4

/-- The number of digits in a boat license. -/
def num_digits : ℕ := 6

/-- The number of possible digits for each position (0-9). -/
def digits_per_position : ℕ := 10

/-- The total number of possible boat license combinations. -/
def total_combinations : ℕ := num_letters * (digits_per_position ^ num_digits)

/-- Theorem stating that the total number of boat license combinations is 4,000,000. -/
theorem boat_license_combinations :
  total_combinations = 4000000 := by
  sorry

end boat_license_combinations_l2932_293213


namespace fox_invasion_count_l2932_293290

/-- The number of foxes that invaded the forest region --/
def num_foxes : ℕ := 3

/-- The initial number of rodents in the forest --/
def initial_rodents : ℕ := 150

/-- The number of rodents each fox catches per week --/
def rodents_per_fox_per_week : ℕ := 6

/-- The number of weeks the foxes hunted --/
def weeks : ℕ := 3

/-- The number of rodents remaining after the foxes hunted --/
def remaining_rodents : ℕ := 96

theorem fox_invasion_count :
  num_foxes * (rodents_per_fox_per_week * weeks) = initial_rodents - remaining_rodents :=
by sorry

end fox_invasion_count_l2932_293290


namespace james_comics_count_l2932_293285

/-- The number of days in a non-leap year -/
def daysInYear : ℕ := 365

/-- The number of years James writes comics -/
def numYears : ℕ := 4

/-- The frequency of James writing comics (every other day) -/
def comicFrequency : ℕ := 2

/-- The total number of comics James writes in 4 non-leap years -/
def totalComics : ℕ := (daysInYear * numYears) / comicFrequency

theorem james_comics_count : totalComics = 730 := by
  sorry

end james_comics_count_l2932_293285


namespace card_passing_game_theorem_l2932_293214

/-- Represents the state of the card-passing game -/
structure GameState where
  num_students : ℕ
  num_cards : ℕ
  card_distribution : List ℕ

/-- Defines a valid game state -/
def valid_game_state (state : GameState) : Prop :=
  state.num_students = 1994 ∧
  state.card_distribution.length = state.num_students ∧
  state.card_distribution.sum = state.num_cards

/-- Defines the condition for the game to end -/
def game_ends (state : GameState) : Prop :=
  ∀ n, n ∈ state.card_distribution → n ≤ 1

/-- Defines the ability to continue the game -/
def can_continue (state : GameState) : Prop :=
  ∃ n, n ∈ state.card_distribution ∧ n ≥ 2

/-- Main theorem about the card-passing game -/
theorem card_passing_game_theorem (state : GameState) 
  (h_valid : valid_game_state state) :
  (state.num_cards ≥ state.num_students → 
    ∃ (game_sequence : ℕ → GameState), ∀ n, can_continue (game_sequence n)) ∧
  (state.num_cards < state.num_students → 
    ∃ (game_sequence : ℕ → GameState) (end_state : ℕ), game_ends (game_sequence end_state)) :=
sorry

end card_passing_game_theorem_l2932_293214


namespace modified_cube_edge_count_l2932_293280

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℝ

/-- Represents the result of removing smaller cubes from the corners of a larger cube -/
structure ModifiedCube where
  originalCube : Cube
  removedCube : Cube

/-- Calculates the number of edges in the modified cube structure -/
def edgeCount (mc : ModifiedCube) : ℕ :=
  12 + 8 * 6

/-- Theorem stating that removing cubes of side length 5 from each corner of a cube 
    with side length 10 results in a solid with 60 edges -/
theorem modified_cube_edge_count :
  let largeCube := Cube.mk 10
  let smallCube := Cube.mk 5
  let modifiedCube := ModifiedCube.mk largeCube smallCube
  edgeCount modifiedCube = 60 := by
  sorry

end modified_cube_edge_count_l2932_293280


namespace calculate_expression_l2932_293244

theorem calculate_expression : (-3)^25 + 2^(4^2 + 5^2 - 7^2) + 3^3 = -3^25 + 27 + 1/256 := by
  sorry

end calculate_expression_l2932_293244


namespace fib_div_by_five_l2932_293210

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem fib_div_by_five (n : ℕ) : 5 ∣ n → 5 ∣ fib n := by
  sorry

end fib_div_by_five_l2932_293210


namespace flour_needed_l2932_293228

/-- Given a recipe requiring 12 cups of flour and 10 cups already added,
    prove that the additional cups of flour needed is 2. -/
theorem flour_needed (recipe_flour : ℕ) (added_flour : ℕ) : 
  recipe_flour = 12 → added_flour = 10 → recipe_flour - added_flour = 2 := by
  sorry

end flour_needed_l2932_293228


namespace power_product_eq_7776_l2932_293231

theorem power_product_eq_7776 : 3^5 * 2^5 = 7776 := by
  sorry

end power_product_eq_7776_l2932_293231


namespace ratio_of_squares_nonnegative_l2932_293265

theorem ratio_of_squares_nonnegative (x : ℝ) (h : x ≠ 5) : (x^2) / ((x - 5)^2) ≥ 0 := by
  sorry

end ratio_of_squares_nonnegative_l2932_293265


namespace seven_digit_divisibility_l2932_293252

def is_divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

theorem seven_digit_divisibility :
  ∀ (a b c d e f : ℕ),
    (is_divisible_by_8 (2300000 + a * 10000 + b * 1000 + 372) = false) ∧
    (is_divisible_by_8 (5300000 + c * 10000 + d * 1000 + 164) = false) ∧
    (is_divisible_by_8 (5000000 + e * 10000 + f * 1000 + 3416) = true) ∧
    (is_divisible_by_8 (7100000 + a * 10000 + b * 1000 + 172) = false) :=
by
  sorry

#check seven_digit_divisibility

end seven_digit_divisibility_l2932_293252


namespace triangle_angle_proof_l2932_293233

theorem triangle_angle_proof (a b c : ℝ) (A : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Ensure positive side lengths
  0 < A ∧ A < π →  -- Angle A is between 0 and π
  a^2 = b^2 + Real.sqrt 3 * b * c + c^2 →  -- Given condition
  A = 5 * π / 6 := by
  sorry

end triangle_angle_proof_l2932_293233


namespace product_of_sines_equals_one_eighth_l2932_293236

theorem product_of_sines_equals_one_eighth :
  (1 + Real.sin (π / 12)) * (1 + Real.sin (5 * π / 12)) *
  (1 + Real.sin (7 * π / 12)) * (1 + Real.sin (11 * π / 12)) = 1 / 8 := by
  sorry

end product_of_sines_equals_one_eighth_l2932_293236


namespace trigonometric_product_bounds_l2932_293271

theorem trigonometric_product_bounds (x y z : Real) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π/12) 
  (h4 : x + y + z = π/2) : 
  1/8 ≤ Real.cos x * Real.sin y * Real.cos z ∧ 
  Real.cos x * Real.sin y * Real.cos z ≤ (2 + Real.sqrt 3) / 8 := by
  sorry

end trigonometric_product_bounds_l2932_293271


namespace circle_radius_satisfies_condition_l2932_293247

/-- The radius of a circle satisfying the given condition -/
def circle_radius : ℝ := 8

/-- The condition that the product of four inches and the circumference equals the area -/
def circle_condition (r : ℝ) : Prop := 4 * (2 * Real.pi * r) = Real.pi * r^2

/-- Theorem stating that the radius satisfies the condition -/
theorem circle_radius_satisfies_condition : 
  circle_condition circle_radius := by sorry

end circle_radius_satisfies_condition_l2932_293247


namespace cube_root_of_2_plus_11i_l2932_293251

def complex_cube_root (z : ℂ) : Prop :=
  z^3 = (2 : ℂ) + Complex.I * 11

theorem cube_root_of_2_plus_11i :
  complex_cube_root (2 + Complex.I) ∧
  ∃ (z₁ z₂ : ℂ), 
    complex_cube_root z₁ ∧
    complex_cube_root z₂ ∧
    z₁ ≠ z₂ ∧
    z₁ ≠ (2 + Complex.I) ∧
    z₂ ≠ (2 + Complex.I) :=
by sorry

end cube_root_of_2_plus_11i_l2932_293251


namespace triangle_side_ratio_range_l2932_293229

open Real

theorem triangle_side_ratio_range (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  C = 3 * B ∧  -- Given condition
  a / sin A = b / sin B ∧  -- Sine rule
  a / sin A = c / sin C →  -- Sine rule
  1 < a / b ∧ a / b < 3 := by
sorry

end triangle_side_ratio_range_l2932_293229


namespace polygon_interior_exterior_angle_relation_l2932_293263

theorem polygon_interior_exterior_angle_relation (n : ℕ) : 
  (n - 2) * 180 = 2 * 360 → n = 6 := by
  sorry

end polygon_interior_exterior_angle_relation_l2932_293263


namespace slower_speed_calculation_l2932_293266

/-- Proves that the slower speed is 8.4 km/hr given the conditions of the problem -/
theorem slower_speed_calculation (actual_distance : ℝ) (faster_speed : ℝ) (additional_distance : ℝ)
  (h1 : actual_distance = 50)
  (h2 : faster_speed = 14)
  (h3 : additional_distance = 20)
  : ∃ slower_speed : ℝ,
    slower_speed = 8.4 ∧
    (actual_distance / faster_speed = (actual_distance - additional_distance) / slower_speed) := by
  sorry

end slower_speed_calculation_l2932_293266


namespace exactly_one_mean_value_point_l2932_293294

-- Define the function f(x) = x³ + 2x
def f (x : ℝ) : ℝ := x^3 + 2*x

-- Define the mean value point condition
def is_mean_value_point (f : ℝ → ℝ) (x₀ : ℝ) (a b : ℝ) : Prop :=
  x₀ ∈ Set.Icc a b ∧ f x₀ = (∫ (x : ℝ) in a..b, f x) / (b - a)

-- Theorem statement
theorem exactly_one_mean_value_point :
  ∃! x₀ : ℝ, is_mean_value_point f x₀ (-1) 1 :=
sorry

end exactly_one_mean_value_point_l2932_293294


namespace simplest_form_sqrt_l2932_293223

/-- A square root is in its simplest form if the number under the root has no perfect square factors other than 1. -/
def is_simplest_form (x : ℝ) : Prop :=
  ∀ n : ℕ, n > 1 → (n^2 : ℝ) ∣ x → False

/-- Given four square roots, prove that √15 is in its simplest form while the others are not. -/
theorem simplest_form_sqrt : 
  is_simplest_form 15 ∧ 
  ¬is_simplest_form 24 ∧ 
  ¬is_simplest_form (7/3) ∧ 
  ¬is_simplest_form 0.9 :=
sorry

end simplest_form_sqrt_l2932_293223


namespace yard_length_with_11_trees_l2932_293258

/-- The length of a yard with equally spaced trees -/
def yardLength (numTrees : ℕ) (distanceBetweenTrees : ℝ) : ℝ :=
  (numTrees - 1) * distanceBetweenTrees

/-- Theorem: The length of a yard with 11 equally spaced trees, 
    with 15 meters between consecutive trees, is 150 meters -/
theorem yard_length_with_11_trees : 
  yardLength 11 15 = 150 := by
  sorry

end yard_length_with_11_trees_l2932_293258


namespace money_difference_l2932_293293

-- Define the amounts for each day
def tuesday_amount : ℝ := 8.5
def wednesday_amount : ℝ := 5.5 * tuesday_amount
def thursday_amount : ℝ := wednesday_amount * 1.1
def friday_amount : ℝ := thursday_amount * 0.75

-- Define the difference
def difference : ℝ := friday_amount - tuesday_amount

-- Theorem statement
theorem money_difference : difference = 30.06875 := by
  sorry

end money_difference_l2932_293293


namespace inequality_proof_l2932_293232

theorem inequality_proof (x : ℝ) (hx : x ≠ 0) :
  max 0 (Real.log (abs x)) ≥ 
    ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (abs x) + 
    (1 / (2 * Real.sqrt 5)) * Real.log (abs (x^2 - 1)) + 
    (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2) ∧
  (max 0 (Real.log (abs x)) = 
    ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (abs x) + 
    (1 / (2 * Real.sqrt 5)) * Real.log (abs (x^2 - 1)) + 
    (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2) ↔ 
    x = (Real.sqrt 5 + 1) / 2 ∨ x = (Real.sqrt 5 - 1) / 2 ∨ 
    x = -(Real.sqrt 5 + 1) / 2 ∨ x = -(Real.sqrt 5 - 1) / 2) :=
by sorry

end inequality_proof_l2932_293232


namespace largest_three_digit_arithmetic_sequence_l2932_293257

/-- Checks if a three-digit number has distinct digits -/
def has_distinct_digits (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 100) ≠ ((n / 10) % 10) ∧
  (n / 100) ≠ (n % 10) ∧
  ((n / 10) % 10) ≠ (n % 10)

/-- Checks if the digits of a three-digit number form an arithmetic sequence -/
def digits_form_arithmetic_sequence (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 100) - ((n / 10) % 10) = ((n / 10) % 10) - (n % 10)

/-- The main theorem stating that 789 is the largest three-digit number
    with distinct digits forming an arithmetic sequence -/
theorem largest_three_digit_arithmetic_sequence :
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 →
    has_distinct_digits m ∧ digits_form_arithmetic_sequence m →
    m ≤ 789) ∧
  has_distinct_digits 789 ∧ digits_form_arithmetic_sequence 789 :=
sorry

end largest_three_digit_arithmetic_sequence_l2932_293257


namespace partnership_investment_l2932_293237

/-- Partnership investment problem -/
theorem partnership_investment (x : ℝ) (m : ℝ) : 
  x > 0 ∧ m > 0 →
  18900 * (12 * x) / (12 * x + 2 * x * (12 - m) + 3 * x * 4) = 6300 →
  m = 6 := by
  sorry

end partnership_investment_l2932_293237


namespace solution_set_for_a_equals_4_solution_existence_condition_l2932_293291

-- Define the function f
def f (a x : ℝ) : ℝ := |2*x - a|

-- Theorem for the first part of the problem
theorem solution_set_for_a_equals_4 :
  {x : ℝ | |2*x - 4| < 8 - |x - 1|} = Set.Ioo (-1) (13/3) := by sorry

-- Theorem for the second part of the problem
theorem solution_existence_condition (a : ℝ) :
  (∃ x, f a x > 8 + |2*x - 1|) ↔ (a > 9 ∨ a < -7) := by sorry

end solution_set_for_a_equals_4_solution_existence_condition_l2932_293291


namespace shirt_cost_l2932_293286

/-- Given the cost of jeans and shirts in two scenarios, prove the cost of one shirt. -/
theorem shirt_cost (j s : ℚ) 
  (scenario1 : 3 * j + 2 * s = 69)
  (scenario2 : 2 * j + 3 * s = 66) :
  s = 12 := by
  sorry

end shirt_cost_l2932_293286


namespace greatest_area_difference_l2932_293238

/-- Represents a rectangle with integer dimensions. -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- Calculates the area of a rectangle. -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Represents the maximum length available for the rotated rectangle's diagonal. -/
def maxDiagonal : ℕ := 50

theorem greatest_area_difference : 
  ∃ (r1 r2 : Rectangle), 
    perimeter r1 = 100 ∧ 
    perimeter r2 = 100 ∧ 
    r2.length * r2.length + r2.width * r2.width ≤ maxDiagonal * maxDiagonal ∧
    ∀ (s1 s2 : Rectangle), 
      perimeter s1 = 100 → 
      perimeter s2 = 100 → 
      s2.length * s2.length + s2.width * s2.width ≤ maxDiagonal * maxDiagonal →
      (area r1 - area r2) ≥ (area s1 - area s2) ∧
      (area r1 - area r2) = 373 :=
by
  sorry

end greatest_area_difference_l2932_293238


namespace cylinder_intersection_angle_l2932_293211

theorem cylinder_intersection_angle (r b a : ℝ) (h_r : r = 1) (h_b : b = r) 
  (h_e : (Real.sqrt 5) / 3 = Real.sqrt (1 - (b / a)^2)) :
  Real.arccos (2 / 3) = Real.arccos (b / a) := by sorry

end cylinder_intersection_angle_l2932_293211


namespace cycle_price_proof_l2932_293222

/-- Proves that given a cycle sold at a 20% loss for Rs. 1280, the original price was Rs. 1600 -/
theorem cycle_price_proof (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1280)
  (h2 : loss_percentage = 20) : 
  ∃ (original_price : ℝ), 
    original_price * (1 - loss_percentage / 100) = selling_price ∧ 
    original_price = 1600 :=
by sorry

end cycle_price_proof_l2932_293222


namespace cubic_stone_weight_l2932_293202

/-- The weight of a cubic stone -/
def stone_weight (edge_length : ℝ) (weight_per_unit : ℝ) : ℝ :=
  edge_length ^ 3 * weight_per_unit

/-- Theorem: The weight of a cubic stone with edge length 8 decimeters,
    where each cubic decimeter weighs 3.5 kilograms, is 1792 kilograms. -/
theorem cubic_stone_weight :
  stone_weight 8 3.5 = 1792 := by
  sorry

end cubic_stone_weight_l2932_293202


namespace largest_x_value_l2932_293205

theorem largest_x_value (x : ℝ) : 
  (16 * x^2 - 40 * x + 15) / (4 * x - 3) + 7 * x = 8 * x - 2 →
  x ≤ 9/4 ∧ ∃ y, y > 9/4 → (16 * y^2 - 40 * y + 15) / (4 * y - 3) + 7 * y ≠ 8 * y - 2 :=
by sorry

end largest_x_value_l2932_293205


namespace mork_mindy_tax_rate_l2932_293249

/-- The combined tax rate for Mork and Mindy -/
theorem mork_mindy_tax_rate 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (mindy_income_ratio : ℝ) 
  (h1 : mork_rate = 0.4) 
  (h2 : mindy_rate = 0.3) 
  (h3 : mindy_income_ratio = 2) : 
  (mork_rate + mindy_rate * mindy_income_ratio) / (1 + mindy_income_ratio) = 1/3 :=
by sorry

end mork_mindy_tax_rate_l2932_293249


namespace harry_last_mile_water_consumption_l2932_293274

/-- Represents the hike scenario --/
structure HikeScenario where
  totalDistance : ℝ
  initialWater : ℝ
  finalWater : ℝ
  timeTaken : ℝ
  leakRate : ℝ
  waterConsumptionFirstThreeMiles : ℝ

/-- Calculates the water consumed in the last mile of the hike --/
def waterConsumedLastMile (h : HikeScenario) : ℝ :=
  h.initialWater - h.finalWater - (h.leakRate * h.timeTaken) - (h.waterConsumptionFirstThreeMiles * (h.totalDistance - 1))

/-- Theorem stating that Harry drank 3 cups of water in the last mile --/
theorem harry_last_mile_water_consumption :
  let h : HikeScenario := {
    totalDistance := 4
    initialWater := 10
    finalWater := 2
    timeTaken := 2
    leakRate := 1
    waterConsumptionFirstThreeMiles := 1
  }
  waterConsumedLastMile h = 3 := by
  sorry


end harry_last_mile_water_consumption_l2932_293274


namespace solution_in_interval_l2932_293217

theorem solution_in_interval (x : ℝ) : 
  (Real.log x + x = 2) → (1.5 < x ∧ x < 2) := by
  sorry

end solution_in_interval_l2932_293217


namespace diana_eraser_sharing_l2932_293230

theorem diana_eraser_sharing (total_erasers : ℕ) (erasers_per_friend : ℕ) (h1 : total_erasers = 3840) (h2 : erasers_per_friend = 80) :
  total_erasers / erasers_per_friend = 48 := by
  sorry

end diana_eraser_sharing_l2932_293230


namespace table_problem_l2932_293218

theorem table_problem :
  (∀ x : ℤ, (2 * x - 7 : ℤ) = -5 ↔ x = 1) ∧
  (∀ x : ℤ, (-3 * x - 1 : ℤ) = 5 ↔ x = -2) ∧
  (∀ x : ℤ, (3 * x + 2 : ℤ) - (-2 * x + 5) = 7 ↔ x = 2) ∧
  (∀ m n : ℤ, (∀ x : ℤ, (m * (x + 1) + n : ℤ) - (m * x + n) = -4) →
              (m * 3 + n : ℤ) = -5 →
              (m * 7 + n : ℤ) = -21) :=
by sorry

end table_problem_l2932_293218


namespace remainder_of_product_div_12_l2932_293267

theorem remainder_of_product_div_12 : (1125 * 1127 * 1129) % 12 = 3 := by
  sorry

end remainder_of_product_div_12_l2932_293267


namespace journey_distance_l2932_293262

/-- Proves that the total distance of a journey is 70 km given specific travel conditions. -/
theorem journey_distance (v1 v2 : ℝ) (t_late : ℝ) : 
  v1 = 40 →  -- Average speed for on-time arrival (km/h)
  v2 = 35 →  -- Average speed for late arrival (km/h)
  t_late = 0.25 →  -- Time of late arrival (hours)
  ∃ (d t : ℝ), 
    d = v1 * t ∧  -- Distance equation for on-time arrival
    d = v2 * (t + t_late) ∧  -- Distance equation for late arrival
    d = 70  -- Total distance of the journey (km)
  := by sorry

end journey_distance_l2932_293262


namespace slower_time_is_692_l2932_293227

/-- The number of stories in the building -/
def num_stories : ℕ := 50

/-- The time Lola takes to run up one story (in seconds) -/
def lola_time_per_story : ℕ := 12

/-- The time the elevator takes to go up one story (in seconds) -/
def elevator_time_per_story : ℕ := 10

/-- The time the elevator stops on each floor (in seconds) -/
def elevator_stop_time : ℕ := 4

/-- The number of floors where the elevator stops -/
def num_elevator_stops : ℕ := num_stories - 2

/-- The time Lola takes to reach the top floor -/
def lola_total_time : ℕ := num_stories * lola_time_per_story

/-- The time Tara takes to reach the top floor -/
def tara_total_time : ℕ := num_stories * elevator_time_per_story + num_elevator_stops * elevator_stop_time

theorem slower_time_is_692 : max lola_total_time tara_total_time = 692 := by sorry

end slower_time_is_692_l2932_293227


namespace edward_money_theorem_l2932_293279

/-- Represents the amount of money Edward had before spending --/
def initial_amount : ℝ := 22

/-- Represents the amount Edward spent on books --/
def spent_amount : ℝ := 16

/-- Represents the amount Edward has left --/
def remaining_amount : ℝ := 6

/-- Represents the number of books Edward bought --/
def number_of_books : ℕ := 92

/-- Theorem stating that the initial amount equals the sum of spent and remaining amounts --/
theorem edward_money_theorem :
  initial_amount = spent_amount + remaining_amount := by sorry

end edward_money_theorem_l2932_293279


namespace y_minimum_value_and_interval_l2932_293282

def y (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem y_minimum_value_and_interval :
  (∃ (m : ℝ), ∀ (x : ℝ), y x ≥ m ∧ (∃ (x₀ : ℝ), y x₀ = m)) ∧
  (∀ (x : ℝ), y x = 2 ↔ -1 < x ∧ x < 1) :=
sorry

end y_minimum_value_and_interval_l2932_293282


namespace double_wardrobe_socks_l2932_293207

/-- Represents the number of items in Jonas' wardrobe -/
structure Wardrobe :=
  (socks : ℕ)
  (shoes : ℕ)
  (pants : ℕ)
  (tshirts : ℕ)

/-- Calculates the total number of individual items in the wardrobe -/
def totalItems (w : Wardrobe) : ℕ :=
  w.socks * 2 + w.shoes * 2 + w.pants + w.tshirts

/-- Calculates the number of sock pairs needed to double the wardrobe -/
def sockPairsNeeded (w : Wardrobe) : ℕ :=
  totalItems w

theorem double_wardrobe_socks (w : Wardrobe) 
  (h1 : w.socks = 20)
  (h2 : w.shoes = 5)
  (h3 : w.pants = 10)
  (h4 : w.tshirts = 10) :
  sockPairsNeeded w = 35 := by
  sorry

#eval sockPairsNeeded { socks := 20, shoes := 5, pants := 10, tshirts := 10 }

end double_wardrobe_socks_l2932_293207


namespace initial_people_is_ten_l2932_293225

/-- Represents the job completion scenario with given conditions -/
structure JobCompletion where
  initialDays : ℕ
  initialWorkDone : ℚ
  daysBeforeFiring : ℕ
  peopleFired : ℕ
  remainingDays : ℕ

/-- Calculates the initial number of people hired -/
def initialPeopleHired (job : JobCompletion) : ℕ :=
  sorry

/-- Theorem stating that the initial number of people hired is 10 -/
theorem initial_people_is_ten (job : JobCompletion) 
  (h1 : job.initialDays = 100)
  (h2 : job.initialWorkDone = 1/4)
  (h3 : job.daysBeforeFiring = 20)
  (h4 : job.peopleFired = 2)
  (h5 : job.remainingDays = 75) :
  initialPeopleHired job = 10 :=
sorry

end initial_people_is_ten_l2932_293225


namespace largest_decimal_l2932_293220

theorem largest_decimal : ∀ (a b c d e : ℚ),
  a = 0.936 ∧ b = 0.9358 ∧ c = 0.9361 ∧ d = 0.935 ∧ e = 0.921 →
  c ≥ a ∧ c ≥ b ∧ c ≥ d ∧ c ≥ e :=
by sorry

end largest_decimal_l2932_293220


namespace square_triangle_area_ratio_l2932_293245

theorem square_triangle_area_ratio : 
  ∀ (s t : ℝ), s > 0 → t > 0 → 
  s^2 = (t^2 * Real.sqrt 3) / 4 → 
  s / t = (Real.sqrt (Real.sqrt 3)) / 2 := by
sorry

end square_triangle_area_ratio_l2932_293245


namespace journey_time_approx_24_hours_l2932_293277

/-- Represents a segment of the journey --/
structure Segment where
  distance : Float
  speed : Float
  stay : Float

/-- Calculates the time taken for a segment --/
def segmentTime (s : Segment) : Float :=
  s.distance / s.speed + s.stay

/-- Represents Manex's journey --/
def manexJourney : List Segment := [
  { distance := 70, speed := 60, stay := 1 },
  { distance := 50, speed := 35, stay := 3 },
  { distance := 20, speed := 60, stay := 0 },
  { distance := 20, speed := 30, stay := 2 },
  { distance := 30, speed := 40, stay := 0 },
  { distance := 60, speed := 70, stay := 2.5 },
  { distance := 60, speed := 35, stay := 0.75 }
]

/-- Calculates the total outbound distance --/
def outboundDistance : Float :=
  (manexJourney.map (·.distance)).sum

/-- Represents the return journey --/
def returnJourney : Segment :=
  { distance := outboundDistance + 100, speed := 55, stay := 0 }

/-- Calculates the total journey time --/
def totalJourneyTime : Float :=
  (manexJourney.map segmentTime).sum + segmentTime returnJourney

/-- Theorem stating that the total journey time is approximately 24 hours --/
theorem journey_time_approx_24_hours :
  (totalJourneyTime).round = 24 := by
  sorry


end journey_time_approx_24_hours_l2932_293277


namespace min_steps_parallel_line_l2932_293297

/-- A line in a plane -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- A point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A circle in a plane -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents a construction step using either a line or a circle -/
inductive ConstructionStep
  | line : Line → ConstructionStep
  | circle : Circle → ConstructionStep

/-- Checks if a point is on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Checks if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- The main theorem stating that the minimum number of construction steps to create a parallel line is 3 -/
theorem min_steps_parallel_line 
  (a : Line) (O : Point) (h : ¬ O.onLine a) :
  ∃ (steps : List ConstructionStep) (l : Line),
    steps.length = 3 ∧
    O.onLine l ∧
    l.parallel a ∧
    (∀ (steps' : List ConstructionStep) (l' : Line),
      steps'.length < 3 →
      ¬(O.onLine l' ∧ l'.parallel a)) :=
sorry

end min_steps_parallel_line_l2932_293297


namespace min_draws_for_twenty_l2932_293272

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls to guarantee at least n of a single color -/
def minDraws (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The actual ball counts in the problem -/
def problemCounts : BallCounts :=
  { red := 35, green := 25, yellow := 22, blue := 15, white := 12, black := 10 }

/-- The theorem to be proved -/
theorem min_draws_for_twenty (counts : BallCounts) :
  counts = problemCounts → minDraws counts 20 = 95 := by
  sorry

end min_draws_for_twenty_l2932_293272
