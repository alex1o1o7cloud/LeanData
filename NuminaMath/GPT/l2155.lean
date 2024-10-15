import Mathlib

namespace NUMINAMATH_GPT_total_games_played_l2155_215579

theorem total_games_played (points_per_game_winner : ℕ) (points_per_game_loser : ℕ) (jack_games_won : ℕ)
  (jill_total_points : ℕ) (total_games : ℕ)
  (h1 : points_per_game_winner = 2)
  (h2 : points_per_game_loser = 1)
  (h3 : jack_games_won = 4)
  (h4 : jill_total_points = 10)
  (h5 : ∀ games_won_by_jill : ℕ, jill_total_points = games_won_by_jill * points_per_game_winner +
           (jack_games_won * points_per_game_loser)) :
  total_games = jack_games_won + (jill_total_points - jack_games_won * points_per_game_loser) / points_per_game_winner := by
  sorry

end NUMINAMATH_GPT_total_games_played_l2155_215579


namespace NUMINAMATH_GPT_range_of_independent_variable_of_sqrt_l2155_215521

theorem range_of_independent_variable_of_sqrt (x : ℝ) : (2 * x - 3 ≥ 0) ↔ (x ≥ 3 / 2) := sorry

end NUMINAMATH_GPT_range_of_independent_variable_of_sqrt_l2155_215521


namespace NUMINAMATH_GPT_least_number_subtracted_l2155_215574

theorem least_number_subtracted (n m : ℕ) (h₁ : m = 2590) (h₂ : n = 2590 - 16) :
  (n % 9 = 6) ∧ (n % 11 = 6) ∧ (n % 13 = 6) :=
by
  sorry

end NUMINAMATH_GPT_least_number_subtracted_l2155_215574


namespace NUMINAMATH_GPT_geometric_progression_identity_l2155_215512

theorem geometric_progression_identity (a b c : ℝ) (h : b^2 = a * c) : 
  (a + b + c) * (a - b + c) = a^2 + b^2 + c^2 := 
by
  sorry

end NUMINAMATH_GPT_geometric_progression_identity_l2155_215512


namespace NUMINAMATH_GPT_non_zero_digits_fraction_l2155_215517

def count_non_zero_digits (n : ℚ) : ℕ :=
  -- A placeholder for the actual implementation.
  sorry

theorem non_zero_digits_fraction : count_non_zero_digits (120 / (2^4 * 5^9 : ℚ)) = 3 :=
  sorry

end NUMINAMATH_GPT_non_zero_digits_fraction_l2155_215517


namespace NUMINAMATH_GPT_confectioner_pastry_l2155_215578

theorem confectioner_pastry (P : ℕ) (h : P / 28 - 6 = P / 49) : P = 378 :=
sorry

end NUMINAMATH_GPT_confectioner_pastry_l2155_215578


namespace NUMINAMATH_GPT_smartphone_cost_l2155_215546

theorem smartphone_cost :
  let current_savings : ℕ := 40
  let weekly_saving : ℕ := 15
  let num_months : ℕ := 2
  let weeks_in_month : ℕ := 4 
  let total_weeks := num_months * weeks_in_month
  let total_savings := weekly_saving * total_weeks
  let total_money := current_savings + total_savings
  total_money = 160 := by
  sorry

end NUMINAMATH_GPT_smartphone_cost_l2155_215546


namespace NUMINAMATH_GPT_solve_equation_l2155_215537

theorem solve_equation :
  {x : ℝ | (x + 1) * (x + 3) = x + 1} = {-1, -2} :=
sorry

end NUMINAMATH_GPT_solve_equation_l2155_215537


namespace NUMINAMATH_GPT_lara_harvest_raspberries_l2155_215587

-- Define measurements of the garden
def length : ℕ := 10
def width : ℕ := 7

-- Define planting and harvesting constants
def plants_per_sq_ft : ℕ := 5
def raspberries_per_plant : ℕ := 12

-- Calculate expected number of raspberries
theorem lara_harvest_raspberries :  length * width * plants_per_sq_ft * raspberries_per_plant = 4200 := 
by sorry

end NUMINAMATH_GPT_lara_harvest_raspberries_l2155_215587


namespace NUMINAMATH_GPT_merchant_gross_profit_l2155_215557

theorem merchant_gross_profit :
  ∃ S : ℝ, (42 + 0.30 * S = S) ∧ ((0.80 * S) - 42 = 6) :=
by
  sorry

end NUMINAMATH_GPT_merchant_gross_profit_l2155_215557


namespace NUMINAMATH_GPT_Bills_age_proof_l2155_215588

variable {b t : ℚ}

theorem Bills_age_proof (h1 : b = 4 * t / 3) (h2 : b + 30 = 9 * (t + 30) / 8) : b = 24 := by 
  sorry

end NUMINAMATH_GPT_Bills_age_proof_l2155_215588


namespace NUMINAMATH_GPT_xy_product_l2155_215522

theorem xy_product (x y z : ℝ) (h : x^2 + y^2 = x * y * (z + 1 / z)) :
  x = y * z ∨ y = x * z := 
by
  sorry

end NUMINAMATH_GPT_xy_product_l2155_215522


namespace NUMINAMATH_GPT_molecular_weight_BaBr2_l2155_215553

theorem molecular_weight_BaBr2 (w: ℝ) (h: w = 2376) : w / 8 = 297 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_BaBr2_l2155_215553


namespace NUMINAMATH_GPT_decrease_of_negative_distance_l2155_215506

theorem decrease_of_negative_distance (x : Int) (increase : Int → Int) (decrease : Int → Int) :
  (increase 30 = 30) → (decrease 5 = -5) → (decrease 5 = -5) :=
by
  intros
  sorry

end NUMINAMATH_GPT_decrease_of_negative_distance_l2155_215506


namespace NUMINAMATH_GPT_equivalent_expression_l2155_215585

-- Define the conditions and the statement that needs to be proven
theorem equivalent_expression (x : ℝ) (h : x^2 - 2 * x + 1 = 0) : 2 * x^2 - 4 * x = -2 := 
  by
    sorry

end NUMINAMATH_GPT_equivalent_expression_l2155_215585


namespace NUMINAMATH_GPT_apples_chosen_l2155_215536

def total_fruits : ℕ := 12
def bananas : ℕ := 4
def oranges : ℕ := 5
def total_other_fruits := bananas + oranges

theorem apples_chosen : total_fruits - total_other_fruits = 3 :=
by sorry

end NUMINAMATH_GPT_apples_chosen_l2155_215536


namespace NUMINAMATH_GPT_num_diagonals_tetragon_l2155_215582

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem num_diagonals_tetragon : num_diagonals_in_polygon 4 = 2 := by
  sorry

end NUMINAMATH_GPT_num_diagonals_tetragon_l2155_215582


namespace NUMINAMATH_GPT_sqrt_x_plus_sqrt_inv_x_l2155_215543

theorem sqrt_x_plus_sqrt_inv_x (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) : 
  (Real.sqrt x + 1 / Real.sqrt x) = Real.sqrt 52 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_x_plus_sqrt_inv_x_l2155_215543


namespace NUMINAMATH_GPT_correct_propositions_l2155_215530

namespace ProofProblem

-- Define Curve C
def curve_C (x y t : ℝ) : Prop :=
  (x^2 / (4 - t)) + (y^2 / (t - 1)) = 1

-- Proposition ①
def proposition_1 (t : ℝ) : Prop :=
  ¬(1 < t ∧ t < 4 ∧ t ≠ 5 / 2)

-- Proposition ②
def proposition_2 (t : ℝ) : Prop :=
  t > 4 ∨ t < 1

-- Proposition ③
def proposition_3 (t : ℝ) : Prop :=
  t ≠ 5 / 2

-- Proposition ④
def proposition_4 (t : ℝ) : Prop :=
  1 < t ∧ t < (5 / 2)

-- The theorem we need to prove
theorem correct_propositions (t : ℝ) :
  (proposition_1 t = false) ∧
  (proposition_2 t = true) ∧
  (proposition_3 t = false) ∧
  (proposition_4 t = true) :=
by
  sorry

end ProofProblem

end NUMINAMATH_GPT_correct_propositions_l2155_215530


namespace NUMINAMATH_GPT_annulus_divide_l2155_215509

theorem annulus_divide (r : ℝ) (h₁ : 2 < 14) (h₂ : 2 > 0) (h₃ : 14 > 0)
    (h₄ : π * 196 - π * r^2 = π * r^2 - π * 4) : r = 10 := 
sorry

end NUMINAMATH_GPT_annulus_divide_l2155_215509


namespace NUMINAMATH_GPT_sum_of_perimeters_l2155_215529

theorem sum_of_perimeters (x y : Real) 
  (h1 : x^2 + y^2 = 85)
  (h2 : x^2 - y^2 = 45) :
  4 * (Real.sqrt 65 + 2 * Real.sqrt 5) = 4 * x + 4 * y := by
  sorry

end NUMINAMATH_GPT_sum_of_perimeters_l2155_215529


namespace NUMINAMATH_GPT_smallest_b_1111_is_square_l2155_215562

theorem smallest_b_1111_is_square : 
  ∃ b : ℕ, b > 0 ∧ (∀ n : ℕ, (b^3 + b^2 + b + 1 = n^2 → b = 7)) :=
by
  sorry

end NUMINAMATH_GPT_smallest_b_1111_is_square_l2155_215562


namespace NUMINAMATH_GPT_solution_inequality_1_range_of_a_l2155_215510

noncomputable def f (x : ℝ) : ℝ := abs x + abs (x - 2)

theorem solution_inequality_1 :
  {x : ℝ | f x < 3} = {x : ℝ | - (1/2) < x ∧ x < (5/2)} :=
by
  sorry

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x < a) → a > 2 :=
by
  sorry

end NUMINAMATH_GPT_solution_inequality_1_range_of_a_l2155_215510


namespace NUMINAMATH_GPT_asymptote_hole_sum_l2155_215577

noncomputable def number_of_holes (f : ℝ → ℝ) : ℕ := -- Assume a definition to count holes
sorry

noncomputable def number_of_vertical_asymptotes (f : ℝ → ℝ) : ℕ := -- Assume a definition to count vertical asymptotes
sorry

noncomputable def number_of_horizontal_asymptotes (f : ℝ → ℝ) : ℕ := -- Assume a definition to count horizontal asymptotes
sorry

noncomputable def number_of_oblique_asymptotes (f : ℝ → ℝ) : ℕ := -- Assume a definition to count oblique asymptotes
sorry

theorem asymptote_hole_sum :
  let f := λ x => (x^2 + 4*x + 3) / (x^3 - 2*x^2 - x + 2)
  let a := number_of_holes f
  let b := number_of_vertical_asymptotes f
  let c := number_of_horizontal_asymptotes f
  let d := number_of_oblique_asymptotes f
  a + 2 * b + 3 * c + 4 * d = 8 :=
by
  sorry

end NUMINAMATH_GPT_asymptote_hole_sum_l2155_215577


namespace NUMINAMATH_GPT_solution_replacement_concentration_l2155_215584

theorem solution_replacement_concentration :
  ∀ (init_conc replaced_fraction new_conc replaced_conc : ℝ),
    init_conc = 0.45 → replaced_fraction = 0.5 → replaced_conc = 0.25 → new_conc = 35 →
    (init_conc - replaced_fraction * init_conc + replaced_fraction * replaced_conc) * 100 = new_conc :=
by
  intro init_conc replaced_fraction new_conc replaced_conc
  intros h_init h_frac h_replaced h_new
  rw [h_init, h_frac, h_replaced, h_new]
  sorry

end NUMINAMATH_GPT_solution_replacement_concentration_l2155_215584


namespace NUMINAMATH_GPT_Susan_has_10_dollars_left_l2155_215547

def initial_amount : ℝ := 80
def food_expense : ℝ := 15
def rides_expense : ℝ := 3 * food_expense
def games_expense : ℝ := 10
def total_expense : ℝ := food_expense + rides_expense + games_expense
def remaining_amount : ℝ := initial_amount - total_expense

theorem Susan_has_10_dollars_left : remaining_amount = 10 := by
  sorry

end NUMINAMATH_GPT_Susan_has_10_dollars_left_l2155_215547


namespace NUMINAMATH_GPT_ratio_of_areas_l2155_215581

noncomputable def side_length_C := 24 -- cm
noncomputable def side_length_D := 54 -- cm
noncomputable def ratio_areas := (side_length_C / side_length_D) ^ 2

theorem ratio_of_areas : ratio_areas = 16 / 81 := sorry

end NUMINAMATH_GPT_ratio_of_areas_l2155_215581


namespace NUMINAMATH_GPT_total_gain_is_19200_l2155_215519

noncomputable def total_annual_gain_of_partnership (x : ℝ) (A_share : ℝ) (B_investment_after : ℕ) (C_investment_after : ℕ) : ℝ :=
  let A_investment_time := 12
  let B_investment_time := 12 - B_investment_after
  let C_investment_time := 12 - C_investment_after
  let proportional_sum := x * A_investment_time + 2 * x * B_investment_time + 3 * x * C_investment_time
  let individual_proportion := proportional_sum / A_investment_time
  3 * A_share

theorem total_gain_is_19200 (x A_share : ℝ) (B_investment_after C_investment_after : ℕ) :
  A_share = 6400 →
  B_investment_after = 6 →
  C_investment_after = 8 →
  total_annual_gain_of_partnership x A_share B_investment_after C_investment_after = 19200 :=
by
  intros hA hB hC
  have x_pos : x > 0 := by sorry   -- Additional assumptions if required
  have A_share_pos : A_share > 0 := by sorry -- Additional assumptions if required
  sorry

end NUMINAMATH_GPT_total_gain_is_19200_l2155_215519


namespace NUMINAMATH_GPT_remainder_when_divided_by_15_l2155_215570

theorem remainder_when_divided_by_15 (N : ℕ) (h1 : N % 60 = 49) : N % 15 = 4 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_15_l2155_215570


namespace NUMINAMATH_GPT_james_fish_weight_l2155_215500

theorem james_fish_weight :
  let trout := 200
  let salmon := trout + (trout * 0.5)
  let tuna := 2 * salmon
  trout + salmon + tuna = 1100 := 
by
  sorry

end NUMINAMATH_GPT_james_fish_weight_l2155_215500


namespace NUMINAMATH_GPT_trapezoid_diagonals_l2155_215595

theorem trapezoid_diagonals (AD BC : ℝ) (angle_DAB angle_BCD : ℝ)
  (hAD : AD = 8) (hBC : BC = 6) (h_angle_DAB : angle_DAB = 90)
  (h_angle_BCD : angle_BCD = 120) :
  ∃ AC BD : ℝ, AC = 4 * Real.sqrt 3 ∧ BD = 2 * Real.sqrt 19 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_diagonals_l2155_215595


namespace NUMINAMATH_GPT_g_h_2_equals_584_l2155_215524

def g (x : ℝ) : ℝ := 3 * x^2 - 4
def h (x : ℝ) : ℝ := -2 * x^3 + 2

theorem g_h_2_equals_584 : g (h 2) = 584 := 
by 
  sorry

end NUMINAMATH_GPT_g_h_2_equals_584_l2155_215524


namespace NUMINAMATH_GPT_maximum_x_minus_y_l2155_215538

theorem maximum_x_minus_y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) : x - y ≤ 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_GPT_maximum_x_minus_y_l2155_215538


namespace NUMINAMATH_GPT_max_f_and_sin_alpha_l2155_215507

noncomputable def f (x : ℝ) : ℝ := Real.sin x + 2 * Real.cos x

theorem max_f_and_sin_alpha :
  (∀ x : ℝ, f x ≤ Real.sqrt 5) ∧ (∃ α : ℝ, (α + Real.arccos (1 / Real.sqrt 5) = π / 2 + 2 * π * some_integer) ∧ (f α = Real.sqrt 5) ∧ (Real.sin α = 1 / Real.sqrt 5)) :=
by
  sorry

end NUMINAMATH_GPT_max_f_and_sin_alpha_l2155_215507


namespace NUMINAMATH_GPT_max_value_of_expression_l2155_215555

theorem max_value_of_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h_sum : x + y + z = 1) :
  x^4 * y^3 * z^2 ≤ 1024 / 14348907 :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l2155_215555


namespace NUMINAMATH_GPT_find_inner_circle_radius_of_trapezoid_l2155_215568

noncomputable def radius_of_inner_circle (k m n p : ℤ) : ℝ :=
  (-k + m * Real.sqrt n) / p

def is_equivalent (a b : ℝ) : Prop := a = b

theorem find_inner_circle_radius_of_trapezoid :
  ∃ (r : ℝ), is_equivalent r (radius_of_inner_circle 123 104 3 29) :=
by
  let r := radius_of_inner_circle 123 104 3 29
  have h1 :  (4^2 + (Real.sqrt (r^2 + 8 * r))^2 = (r + 4)^2) := sorry
  have h2 :  (3^2 + (Real.sqrt (r^2 + 6 * r))^2 = (r + 3)^2) := sorry
  have height_eq : Real.sqrt 13 = (Real.sqrt (r^2 + 6 * r) + Real.sqrt (r^2 + 8 * r)) := sorry
  use r
  exact sorry

end NUMINAMATH_GPT_find_inner_circle_radius_of_trapezoid_l2155_215568


namespace NUMINAMATH_GPT_two_digit_numbers_reverse_square_condition_l2155_215572

theorem two_digit_numbers_reverse_square_condition :
  ∀ (a b : ℕ), 0 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 →
  (∃ n : ℕ, 10 * a + b + 10 * b + a = n^2) ↔ 
  (10 * a + b = 29 ∨ 10 * a + b = 38 ∨ 10 * a + b = 47 ∨ 10 * a + b = 56 ∨ 
   10 * a + b = 65 ∨ 10 * a + b = 74 ∨ 10 * a + b = 83 ∨ 10 * a + b = 92) :=
by {
  sorry
}

end NUMINAMATH_GPT_two_digit_numbers_reverse_square_condition_l2155_215572


namespace NUMINAMATH_GPT_evaluate_expression_l2155_215559

theorem evaluate_expression : 4 * 6 * 8 + 24 / 4 - 2 = 196 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2155_215559


namespace NUMINAMATH_GPT_edge_length_of_cubical_box_l2155_215565

noncomputable def volume_of_cube (edge_length_cm : ℝ) : ℝ :=
  edge_length_cm ^ 3

noncomputable def number_of_cubes : ℝ := 8000
noncomputable def edge_of_small_cube_cm : ℝ := 5

noncomputable def total_volume_of_cubes_cm3 : ℝ :=
  volume_of_cube edge_of_small_cube_cm * number_of_cubes

noncomputable def volume_of_box_cm3 : ℝ := total_volume_of_cubes_cm3
noncomputable def edge_length_of_box_m : ℝ :=
  (volume_of_box_cm3)^(1 / 3) / 100

theorem edge_length_of_cubical_box :
  edge_length_of_box_m = 1 := by 
  sorry

end NUMINAMATH_GPT_edge_length_of_cubical_box_l2155_215565


namespace NUMINAMATH_GPT_arcsin_arccos_interval_l2155_215571

open Real
open Set

theorem arcsin_arccos_interval (x y : ℝ) (h : x^2 + y^2 = 1) : 
  ∃ t ∈ Icc (-3 * π / 2) (π / 2), 2 * arcsin x - arccos y = t := 
sorry

end NUMINAMATH_GPT_arcsin_arccos_interval_l2155_215571


namespace NUMINAMATH_GPT_problems_left_to_grade_l2155_215576

theorem problems_left_to_grade 
  (problems_per_worksheet : ℕ)
  (total_worksheets : ℕ)
  (graded_worksheets : ℕ)
  (h1 : problems_per_worksheet = 2)
  (h2 : total_worksheets = 14)
  (h3 : graded_worksheets = 7) : 
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 14 :=
by 
  sorry

end NUMINAMATH_GPT_problems_left_to_grade_l2155_215576


namespace NUMINAMATH_GPT_find_expression_l2155_215556

variables (x y z : ℝ) (ω : ℂ)

theorem find_expression
  (h1 : x ≠ -1) (h2 : y ≠ -1) (h3 : z ≠ -1)
  (h4 : ω^3 = 1) (h5 : ω ≠ 1)
  (h6 : (1 / (x + ω) + 1 / (y + ω) + 1 / (z + ω) = ω)) :
  1 / (x + 1) + 1 / (y + 1) + 1 / (z + 1) = -1 / 3 :=
sorry

end NUMINAMATH_GPT_find_expression_l2155_215556


namespace NUMINAMATH_GPT_arithmetic_sequence_formula_l2155_215526

theorem arithmetic_sequence_formula (a : ℕ → ℤ) (d : ℤ) :
  (a 3 = 4) → (d = -2) → ∀ n : ℕ, a n = 10 - 2 * n :=
by
  intros h1 h2 n
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_formula_l2155_215526


namespace NUMINAMATH_GPT_liu_xing_statement_incorrect_l2155_215542

-- Definitions of the initial statistics of the classes
def avg_score_class_91 : ℝ := 79.5
def avg_score_class_92 : ℝ := 80.2

-- Definitions of corrections applied
def correction_gain_class_91 : ℝ := 0.6 * 3
def correction_loss_class_91 : ℝ := 0.2 * 3
def correction_gain_class_92 : ℝ := 0.5 * 3
def correction_loss_class_92 : ℝ := 0.3 * 3

-- Definitions of corrected averages
def corrected_avg_class_91 : ℝ := avg_score_class_91 + correction_gain_class_91 - correction_loss_class_91
def corrected_avg_class_92 : ℝ := avg_score_class_92 + correction_gain_class_92 - correction_loss_class_92

-- Proof statement
theorem liu_xing_statement_incorrect : corrected_avg_class_91 ≤ corrected_avg_class_92 :=
by {
  -- Additional hints and preliminary calculations could be done here.
  sorry
}

end NUMINAMATH_GPT_liu_xing_statement_incorrect_l2155_215542


namespace NUMINAMATH_GPT_unique_integer_sequence_exists_l2155_215589

open Nat

def a (n : ℕ) : ℤ := sorry

theorem unique_integer_sequence_exists :
  ∃ (a : ℕ → ℤ), a 1 = 1 ∧ a 2 > 1 ∧ (∀ n ≥ 1, (a (n+1))^3 + 1 = a n * a (n+2)) ∧
  (∀ b, (b 1 = 1) → (b 2 > 1) → (∀ n ≥ 1, (b (n+1))^3 + 1 = b n * b (n+2)) → b = a) :=
by
  sorry

end NUMINAMATH_GPT_unique_integer_sequence_exists_l2155_215589


namespace NUMINAMATH_GPT_discount_savings_difference_l2155_215548

def cover_price : ℝ := 30
def discount_amount : ℝ := 5
def discount_percentage : ℝ := 0.25

theorem discount_savings_difference :
  let price_after_discount := cover_price - discount_amount
  let price_after_percentage_first := cover_price * (1 - discount_percentage)
  let new_price_after_percentage := price_after_discount * (1 - discount_percentage)
  let new_price_after_discount := price_after_percentage_first - discount_amount
  (new_price_after_percentage - new_price_after_discount) * 100 = 125 :=
by
  sorry

end NUMINAMATH_GPT_discount_savings_difference_l2155_215548


namespace NUMINAMATH_GPT_purple_shoes_count_l2155_215514

-- Define the conditions
def total_shoes : ℕ := 1250
def blue_shoes : ℕ := 540
def remaining_shoes : ℕ := total_shoes - blue_shoes
def green_shoes := remaining_shoes / 2
def purple_shoes := green_shoes

-- State the theorem to be proven
theorem purple_shoes_count : purple_shoes = 355 := 
by
-- Proof can be filled in here (not needed for the task)
sorry

end NUMINAMATH_GPT_purple_shoes_count_l2155_215514


namespace NUMINAMATH_GPT_O_l2155_215567

theorem O'Hara_triple_49_16_y : 
  (∃ y : ℕ, (49 : ℕ).sqrt + (16 : ℕ).sqrt = y) → y = 11 :=
by
  sorry

end NUMINAMATH_GPT_O_l2155_215567


namespace NUMINAMATH_GPT_number_division_remainder_l2155_215598

theorem number_division_remainder (N k m : ℤ) (h1 : N = 281 * k + 160) (h2 : N = D * m + 21) : D = 139 :=
by sorry

end NUMINAMATH_GPT_number_division_remainder_l2155_215598


namespace NUMINAMATH_GPT_find_gamma_l2155_215518

variable (γ δ : ℝ)

def directly_proportional (γ δ : ℝ) : Prop := ∃ c : ℝ, γ = c * δ

theorem find_gamma (h1 : directly_proportional γ δ) (h2 : γ = 5) (h3 : δ = -10) : δ = 25 → γ = -25 / 2 := by
  sorry

end NUMINAMATH_GPT_find_gamma_l2155_215518


namespace NUMINAMATH_GPT_border_area_correct_l2155_215593

noncomputable def area_of_border (poster_height poster_width border_width : ℕ) : ℕ :=
  let framed_height := poster_height + 2 * border_width
  let framed_width := poster_width + 2 * border_width
  (framed_height * framed_width) - (poster_height * poster_width)

theorem border_area_correct :
  area_of_border 12 16 4 = 288 :=
by
  rfl

end NUMINAMATH_GPT_border_area_correct_l2155_215593


namespace NUMINAMATH_GPT_length_AC_and_area_OAC_l2155_215591

open Real EuclideanGeometry

def ellipse (x y : ℝ) : Prop :=
  x^2 + 2 * y^2 = 2

def line_1 (x y : ℝ) : Prop :=
  y = x + 1

def line_2 (B : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  B.fst = 3 * P.fst ∧ B.snd = 3 * P.snd

theorem length_AC_and_area_OAC 
  (A C : ℝ × ℝ) 
  (B P : ℝ × ℝ) 
  (O : ℝ × ℝ := (0, 0)) 
  (h1 : ellipse A.fst A.snd) 
  (h2 : ellipse C.fst C.snd) 
  (h3 : line_1 A.fst A.snd) 
  (h4 : line_1 C.fst C.snd) 
  (h5 : line_2 B P) 
  (h6 : (P.fst = (A.fst + C.fst) / 2) ∧ (P.snd = (A.snd + C.snd) / 2)) : 
  |(dist A C)| = 4/3 * sqrt 2 ∧
  (1/2 * abs (A.fst * C.snd - C.fst * A.snd)) = 4/9 := sorry

end NUMINAMATH_GPT_length_AC_and_area_OAC_l2155_215591


namespace NUMINAMATH_GPT_find_a_l2155_215544

-- The conditions converted to Lean definitions
variable (a : ℝ)
variable (α : ℝ)
variable (point_on_terminal_side : a ≠ 0 ∧ (∃ α, tan α = -1 / 2 ∧ ∀ y : ℝ, y = -1 → a = 2 * y) )

-- The theorem statement
theorem find_a (H : point_on_terminal_side): a = 2 := by
  sorry

end NUMINAMATH_GPT_find_a_l2155_215544


namespace NUMINAMATH_GPT_runner_time_second_half_l2155_215534

theorem runner_time_second_half (v : ℝ) (h1 : 20 / v + 4 = 40 / v) : 40 / v = 8 :=
by
  sorry

end NUMINAMATH_GPT_runner_time_second_half_l2155_215534


namespace NUMINAMATH_GPT_ceil_minus_eq_zero_l2155_215592

theorem ceil_minus_eq_zero (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 0) : ⌈x⌉ - x = 0 :=
sorry

end NUMINAMATH_GPT_ceil_minus_eq_zero_l2155_215592


namespace NUMINAMATH_GPT_expression_evaluation_l2155_215541

theorem expression_evaluation :
  5 * 402 + 4 * 402 + 3 * 402 + 401 = 5225 := by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l2155_215541


namespace NUMINAMATH_GPT_even_diagonal_moves_l2155_215590

def King_Moves (ND D : ℕ) :=
  ND + D = 63 ∧ ND % 2 = 0

theorem even_diagonal_moves (ND D : ℕ) (traverse_board : King_Moves ND D) : D % 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_even_diagonal_moves_l2155_215590


namespace NUMINAMATH_GPT_total_savings_l2155_215539

def liam_oranges : ℕ := 40
def liam_price_per_two : ℝ := 2.50
def claire_oranges : ℕ := 30
def claire_price_per_one : ℝ := 1.20

theorem total_savings : (liam_oranges / 2 * liam_price_per_two) + (claire_oranges * claire_price_per_one) = 86 := by
  sorry

end NUMINAMATH_GPT_total_savings_l2155_215539


namespace NUMINAMATH_GPT_find_b_l2155_215554

-- Define complex numbers z1 and z2
def z1 (b : ℝ) : Complex := Complex.mk 3 (-b)

def z2 : Complex := Complex.mk 1 (-2)

-- Statement that needs to be proved
theorem find_b (b : ℝ) (h : (z1 b / z2).re = 0) : b = -3 / 2 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_find_b_l2155_215554


namespace NUMINAMATH_GPT_parallel_line_with_intercept_sum_l2155_215573

theorem parallel_line_with_intercept_sum (c : ℝ) :
  (∀ x y : ℝ, 2 * x + 3 * y + 5 = 0 → 2 * x + 3 * y + c = 0) ∧ 
  (-c / 3 - c / 2 = 6) → 
  (10 * x + 15 * y - 36 = 0) :=
by
  sorry

end NUMINAMATH_GPT_parallel_line_with_intercept_sum_l2155_215573


namespace NUMINAMATH_GPT_fraction_of_students_getting_F_l2155_215597

theorem fraction_of_students_getting_F
  (students_A students_B students_C students_D passing_fraction : ℚ) 
  (hA : students_A = 1/4)
  (hB : students_B = 1/2)
  (hC : students_C = 1/8)
  (hD : students_D = 1/12)
  (hPassing : passing_fraction = 0.875) :
  (1 - (students_A + students_B + students_C + students_D)) = 1/24 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_students_getting_F_l2155_215597


namespace NUMINAMATH_GPT_negation_exists_implication_l2155_215596

theorem negation_exists_implication (x : ℝ) : (¬ ∃ y > 0, y^2 - 2*y - 3 ≤ 0) ↔ ∀ y > 0, y^2 - 2*y - 3 > 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_exists_implication_l2155_215596


namespace NUMINAMATH_GPT_dealer_is_cheating_l2155_215558

variable (w a : ℝ)
noncomputable def measured_weight (w : ℝ) (a : ℝ) : ℝ :=
  (a * w + w / a) / 2

theorem dealer_is_cheating (h : a > 0) : measured_weight w a ≥ w :=
by
  sorry

end NUMINAMATH_GPT_dealer_is_cheating_l2155_215558


namespace NUMINAMATH_GPT_problem1_problem2_l2155_215528

theorem problem1 (a b x y : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < x ∧ 0 < y) : 
  (a^2 / x + b^2 / y) ≥ ((a + b)^2 / (x + y)) ∧ (a * y = b * x → (a^2 / x + b^2 / y) = ((a + b)^2 / (x + y))) :=
sorry

theorem problem2 (x : ℝ) (h : 0 < x ∧ x < 1 / 2) :
  (∀ x, 0 < x ∧ x < 1 / 2 → ((2 / x + 9 / (1 - 2 * x)) ≥ 25)) ∧ (2 * (1 - 2 * (1 / 5)) = 9 * (1 / 5) → (2 / (1 / 5) + 9 / (1 - 2 * (1 / 5)) = 25)) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l2155_215528


namespace NUMINAMATH_GPT_angle_A_is_30_degrees_l2155_215502

theorem angle_A_is_30_degrees {A : ℝ} (hA_acute : 0 < A ∧ A < π / 2) (hA_sin : Real.sin A = 1 / 2) : A = π / 6 :=
sorry

end NUMINAMATH_GPT_angle_A_is_30_degrees_l2155_215502


namespace NUMINAMATH_GPT_Micheal_work_rate_l2155_215580

theorem Micheal_work_rate 
    (M A : ℕ) 
    (h1 : 1 / M + 1 / A = 1 / 20)
    (h2 : 9 / 200 = 1 / A) : M = 200 :=
by
    sorry

end NUMINAMATH_GPT_Micheal_work_rate_l2155_215580


namespace NUMINAMATH_GPT_luke_earning_problem_l2155_215550

variable (WeedEarning Weeks SpendPerWeek MowingEarning : ℤ)

theorem luke_earning_problem
  (h1 : WeedEarning = 18)
  (h2 : Weeks = 9)
  (h3 : SpendPerWeek = 3)
  (h4 : MowingEarning + WeedEarning = Weeks * SpendPerWeek) :
  MowingEarning = 9 := by
  sorry

end NUMINAMATH_GPT_luke_earning_problem_l2155_215550


namespace NUMINAMATH_GPT_unique_root_of_linear_equation_l2155_215511

theorem unique_root_of_linear_equation (a b : ℝ) (h : a ≠ 0) : ∃! x : ℝ, a * x = b :=
by
  sorry

end NUMINAMATH_GPT_unique_root_of_linear_equation_l2155_215511


namespace NUMINAMATH_GPT_boxes_remaining_to_sell_l2155_215527

-- Define the conditions
def first_customer_boxes : ℕ := 5 
def second_customer_boxes : ℕ := 4 * first_customer_boxes
def third_customer_boxes : ℕ := second_customer_boxes / 2
def fourth_customer_boxes : ℕ := 3 * third_customer_boxes
def final_customer_boxes : ℕ := 10
def sales_goal : ℕ := 150

-- Total boxes sold
def total_boxes_sold : ℕ := first_customer_boxes + second_customer_boxes + third_customer_boxes + fourth_customer_boxes + final_customer_boxes

-- Boxes left to sell to hit the sales goal
def boxes_left_to_sell : ℕ := sales_goal - total_boxes_sold

-- Prove the number of boxes left to sell is 75
theorem boxes_remaining_to_sell : boxes_left_to_sell = 75 :=
by
  -- Step to prove goes here
  sorry

end NUMINAMATH_GPT_boxes_remaining_to_sell_l2155_215527


namespace NUMINAMATH_GPT_smallest_possible_value_of_sum_l2155_215531

theorem smallest_possible_value_of_sum (a b : ℤ) (h1 : a > 6) (h2 : ∃ a' b', a' - b' = 4) : a + b < 11 := 
sorry

end NUMINAMATH_GPT_smallest_possible_value_of_sum_l2155_215531


namespace NUMINAMATH_GPT_rectangle_length_from_square_thread_l2155_215560

theorem rectangle_length_from_square_thread (side_of_square width_of_rectangle : ℝ) (same_thread : Bool) 
  (h1 : side_of_square = 20) (h2 : width_of_rectangle = 14) (h3 : same_thread) : 
  ∃ length_of_rectangle : ℝ, length_of_rectangle = 26 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_length_from_square_thread_l2155_215560


namespace NUMINAMATH_GPT_taqeesha_grade_correct_l2155_215540

-- Definitions for conditions
def total_score_of_24_students := 24 * 82
def total_score_of_25_students (T: ℕ) := 25 * 84
def taqeesha_grade := 132

-- Theorem statement forming the proof problem
theorem taqeesha_grade_correct
    (h1: total_score_of_24_students + taqeesha_grade = total_score_of_25_students taqeesha_grade): 
    taqeesha_grade = 132 :=
by
  sorry

end NUMINAMATH_GPT_taqeesha_grade_correct_l2155_215540


namespace NUMINAMATH_GPT_circumradius_eq_l2155_215566

noncomputable def circumradius (r : ℂ) (t1 t2 t3 : ℂ) : ℂ :=
  (2 * r ^ 4) / Complex.abs ((t1 + t2) * (t2 + t3) * (t3 + t1))

theorem circumradius_eq (r t1 t2 t3 : ℂ) (h_pos_r : r ≠ 0) :
  circumradius r t1 t2 t3 = (2 * r ^ 4) / Complex.abs ((t1 + t2) * (t2 + t3) * (t3 + t1)) :=
  by sorry

end NUMINAMATH_GPT_circumradius_eq_l2155_215566


namespace NUMINAMATH_GPT_standard_concession_l2155_215515

theorem standard_concession (x : ℝ) : 
  (∀ (x : ℝ), (2000 - (x / 100) * 2000) - 0.2 * (2000 - (x / 100) * 2000) = 1120) → x = 30 := 
by 
  sorry

end NUMINAMATH_GPT_standard_concession_l2155_215515


namespace NUMINAMATH_GPT_cream_strawberry_prices_l2155_215549

noncomputable def price_flavor_B : ℝ := 30
noncomputable def price_flavor_A : ℝ := 40

theorem cream_strawberry_prices (x y : ℝ) 
  (h1 : y = x + 10) 
  (h2 : 800 / y = 600 / x) : 
  x = price_flavor_B ∧ y = price_flavor_A :=
by 
  sorry

end NUMINAMATH_GPT_cream_strawberry_prices_l2155_215549


namespace NUMINAMATH_GPT_g_at_5_l2155_215513

variable (g : ℝ → ℝ)

-- Define the condition on g
def functional_condition : Prop :=
  ∀ x : ℝ, g x + 3 * g (1 - x) = 2 * x ^ 2 + 1

-- The statement proven should be g(5) = 8 given functional_condition
theorem g_at_5 (h : functional_condition g) : g 5 = 8 := by
  sorry

end NUMINAMATH_GPT_g_at_5_l2155_215513


namespace NUMINAMATH_GPT_trains_clear_time_l2155_215599

theorem trains_clear_time
  (length_train1 : ℕ) (length_train2 : ℕ)
  (speed_train1_kmph : ℕ) (speed_train2_kmph : ℕ)
  (conversion_factor : ℕ) -- 5/18 as a rational number (for clarity)
  (approx_rel_speed : ℚ) -- Approximate relative speed 
  (total_distance : ℕ) 
  (total_time : ℚ) :
  length_train1 = 160 →
  length_train2 = 280 →
  speed_train1_kmph = 42 →
  speed_train2_kmph = 30 →
  conversion_factor = 5 / 18 →
  approx_rel_speed = (42 * (5 / 18) + 30 * (5 / 18)) →
  total_distance = length_train1 + length_train2 →
  total_time = total_distance / approx_rel_speed →
  total_time = 22 := 
by
  sorry

end NUMINAMATH_GPT_trains_clear_time_l2155_215599


namespace NUMINAMATH_GPT_prob_three_friends_same_group_l2155_215501

theorem prob_three_friends_same_group :
  let students := 800
  let groups := 4
  let group_size := students / groups
  let p_same_group := 1 / groups
  p_same_group * p_same_group = 1 / 16 := 
by
  sorry

end NUMINAMATH_GPT_prob_three_friends_same_group_l2155_215501


namespace NUMINAMATH_GPT_find_white_daisies_l2155_215503

theorem find_white_daisies (W P R : ℕ) 
  (h1 : P = 9 * W) 
  (h2 : R = 4 * P - 3) 
  (h3 : W + P + R = 273) : 
  W = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_white_daisies_l2155_215503


namespace NUMINAMATH_GPT_only_possible_b_l2155_215594

theorem only_possible_b (b : ℕ) (h : ∃ a k l : ℕ, k ≠ l ∧ (b > 0) ∧ (a > 0) ∧ (b ^ (k + l)) ∣ (a ^ k + b ^ l) ∧ (b ^ (k + l)) ∣ (a ^ l + b ^ k)) : 
  b = 1 :=
sorry

end NUMINAMATH_GPT_only_possible_b_l2155_215594


namespace NUMINAMATH_GPT_find_a_equiv_l2155_215516

theorem find_a_equiv (a x : ℝ) (h : ∀ x, (a * x^2 + 20 * x + 25) = (2 * x + 5) * (2 * x + 5)) : a = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_a_equiv_l2155_215516


namespace NUMINAMATH_GPT_total_amount_of_money_l2155_215545

def one_rupee_note_value := 1
def five_rupee_note_value := 5
def ten_rupee_note_value := 10

theorem total_amount_of_money (n : ℕ) 
  (h : 3 * n = 90) : n * one_rupee_note_value + n * five_rupee_note_value + n * ten_rupee_note_value = 480 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_of_money_l2155_215545


namespace NUMINAMATH_GPT_imaginary_part_of_z_l2155_215535

-- Let 'z' be the complex number \(\frac {2i}{1-i}\)
noncomputable def z : ℂ := (2 * Complex.I) / (1 - Complex.I)

theorem imaginary_part_of_z :
  z.im = 1 :=
sorry

end NUMINAMATH_GPT_imaginary_part_of_z_l2155_215535


namespace NUMINAMATH_GPT_find_sinD_l2155_215564

variable (DE DF : ℝ)

-- Conditions
def area_of_triangle (DE DF : ℝ) (sinD : ℝ) : Prop :=
  1 / 2 * DE * DF * sinD = 72

def geometric_mean (DE DF : ℝ) : Prop :=
  Real.sqrt (DE * DF) = 15

theorem find_sinD (DE DF sinD : ℝ) (h1 : area_of_triangle DE DF sinD) (h2 : geometric_mean DE DF) :
  sinD = 16 / 25 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_sinD_l2155_215564


namespace NUMINAMATH_GPT_find_swimming_speed_l2155_215586

variable (S : ℝ)

def is_average_speed (x y avg : ℝ) : Prop :=
  avg = 2 * x * y / (x + y)

theorem find_swimming_speed
  (running_speed : ℝ := 7)
  (average_speed : ℝ := 4)
  (h : is_average_speed S running_speed average_speed) :
  S = 2.8 :=
by sorry

end NUMINAMATH_GPT_find_swimming_speed_l2155_215586


namespace NUMINAMATH_GPT_max_value_of_function_l2155_215552

noncomputable def function (x : ℝ) : ℝ := (Real.cos x) ^ 2 - Real.sin x

theorem max_value_of_function : ∃ x : ℝ, function x = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_function_l2155_215552


namespace NUMINAMATH_GPT_Carlson_max_jars_l2155_215583

theorem Carlson_max_jars (n a : ℕ) (hn : 13 * n = 5 * (8 * n + 9 * a)) : ∃ k : ℕ, k ≤ 23 := 
sorry

end NUMINAMATH_GPT_Carlson_max_jars_l2155_215583


namespace NUMINAMATH_GPT_theo_cookies_per_sitting_l2155_215551

-- Definitions from conditions
def sittings_per_day : ℕ := 3
def days_per_month : ℕ := 20
def cookies_in_3_months : ℕ := 2340

-- Calculation based on conditions
def sittings_per_month : ℕ := sittings_per_day * days_per_month
def sittings_in_3_months : ℕ := sittings_per_month * 3

-- Target statement
theorem theo_cookies_per_sitting :
  cookies_in_3_months / sittings_in_3_months = 13 :=
sorry

end NUMINAMATH_GPT_theo_cookies_per_sitting_l2155_215551


namespace NUMINAMATH_GPT_triangle_sides_consecutive_and_angle_relationship_l2155_215505

theorem triangle_sides_consecutive_and_angle_relationship (a b c : ℕ) 
  (h1 : a < b) (h2 : b < c) (h3 : b = a + 1) (h4 : c = b + 1) 
  (angle_A angle_B angle_C : ℝ) 
  (h_angle_sum : angle_A + angle_B + angle_C = π) 
  (h_angle_relation : angle_B = 2 * angle_A) : 
  (a, b, c) = (4, 5, 6) :=
sorry

end NUMINAMATH_GPT_triangle_sides_consecutive_and_angle_relationship_l2155_215505


namespace NUMINAMATH_GPT_find_m_of_parallel_vectors_l2155_215532

theorem find_m_of_parallel_vectors (m : ℝ) 
  (a : ℝ × ℝ := (1, 2)) 
  (b : ℝ × ℝ := (m, m + 1))
  (parallel : a.1 * b.2 = a.2 * b.1) :
  m = 1 :=
by
  -- We assume a parallel condition and need to prove m = 1
  sorry

end NUMINAMATH_GPT_find_m_of_parallel_vectors_l2155_215532


namespace NUMINAMATH_GPT_sawing_steel_bar_time_l2155_215520

theorem sawing_steel_bar_time (pieces : ℕ) (time_per_cut : ℕ) : 
  pieces = 6 → time_per_cut = 2 → (pieces - 1) * time_per_cut = 10 := 
by
  intros
  sorry

end NUMINAMATH_GPT_sawing_steel_bar_time_l2155_215520


namespace NUMINAMATH_GPT_find_original_number_l2155_215533

/-- The difference between a number increased by 18.7% and the same number decreased by 32.5% is 45. -/
theorem find_original_number (w : ℝ) (h : 1.187 * w - 0.675 * w = 45) : w = 45 / 0.512 :=
by
  sorry

end NUMINAMATH_GPT_find_original_number_l2155_215533


namespace NUMINAMATH_GPT_cookies_per_child_l2155_215561

theorem cookies_per_child 
  (total_cookies : ℕ) 
  (children : ℕ) 
  (x : ℚ) 
  (adults_fraction : total_cookies * x = total_cookies / 4) 
  (remaining_cookies : total_cookies - total_cookies * x = 180) 
  (correct_fraction : x = 1 / 4) 
  (correct_children : children = 6) :
  (total_cookies - total_cookies * x) / children = 30 := by
  sorry

end NUMINAMATH_GPT_cookies_per_child_l2155_215561


namespace NUMINAMATH_GPT_platform_length_is_150_l2155_215563

noncomputable def length_of_platform
  (train_length : ℝ)
  (time_to_cross_platform : ℝ)
  (time_to_cross_pole : ℝ)
  (L : ℝ) : Prop :=
  train_length + L = (train_length / time_to_cross_pole) * time_to_cross_platform

theorem platform_length_is_150 :
  length_of_platform 300 27 18 150 :=
by 
  -- Proof omitted, but the statement is ready for proving
  sorry

end NUMINAMATH_GPT_platform_length_is_150_l2155_215563


namespace NUMINAMATH_GPT_cyclist_final_speed_l2155_215569

def u : ℝ := 16
def a : ℝ := 0.5
def t : ℕ := 7200

theorem cyclist_final_speed : 
  (u + a * t) * 3.6 = 13017.6 := by
  sorry

end NUMINAMATH_GPT_cyclist_final_speed_l2155_215569


namespace NUMINAMATH_GPT_correct_option_is_B_l2155_215523

noncomputable def correct_calculation (x : ℝ) : Prop :=
  (x ≠ 1) → (x ≠ 0) → (x ≠ -1) → (-2 / (2 * x - 2) = 1 / (1 - x))

theorem correct_option_is_B (x : ℝ) : correct_calculation x := by
  intros hx1 hx2 hx3
  sorry

end NUMINAMATH_GPT_correct_option_is_B_l2155_215523


namespace NUMINAMATH_GPT_smallest_of_seven_consecutive_even_numbers_l2155_215575

theorem smallest_of_seven_consecutive_even_numbers (a b c d e f g : ℤ)
  (h₁ : a + b + c + d + e + f + g = 700)
  (h₂ : b = a + 2)
  (h₃ : c = a + 4)
  (h₄ : d = a + 6)
  (h₅ : e = a + 8)
  (h₆ : f = a + 10)
  (h₇ : g = a + 12)
  : a = 94 :=
by
  -- Proof is omitted, this is just the statement.
  sorry

end NUMINAMATH_GPT_smallest_of_seven_consecutive_even_numbers_l2155_215575


namespace NUMINAMATH_GPT_polynomial_factors_sum_l2155_215508

theorem polynomial_factors_sum (a b : ℝ) 
  (h : ∃ c : ℝ, (∀ x: ℝ, x^3 + a * x^2 + b * x + 8 = (x + 1) * (x + 2) * (x + c))) : 
  a + b = 21 :=
sorry

end NUMINAMATH_GPT_polynomial_factors_sum_l2155_215508


namespace NUMINAMATH_GPT_range_of_a_l2155_215504

theorem range_of_a (a : ℝ) :
  (∀ x, (x < -1 ∨ x > 5) ∨ (a < x ∧ x < a + 8)) ↔ (-3 < a ∧ a < -1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2155_215504


namespace NUMINAMATH_GPT_average_payment_is_460_l2155_215525

theorem average_payment_is_460 :
  let n := 52
  let first_payment := 410
  let extra := 65
  let num_first_payments := 12
  let num_rest_payments := n - num_first_payments
  let rest_payment := first_payment + extra
  (num_first_payments * first_payment + num_rest_payments * rest_payment) / n = 460 := by
  sorry

end NUMINAMATH_GPT_average_payment_is_460_l2155_215525
