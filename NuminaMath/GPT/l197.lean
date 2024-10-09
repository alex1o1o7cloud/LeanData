import Mathlib

namespace root_interval_k_l197_19779

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem root_interval_k (k : ℤ) (h_cont : Continuous f) (h_mono : Monotone f)
  (h1 : f 2 < 0) (h2 : f 3 > 0) : k = 4 :=
by
  -- The proof part is omitted as per instruction.
  sorry

end root_interval_k_l197_19779


namespace theo_cookies_per_sitting_l197_19706

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

end theo_cookies_per_sitting_l197_19706


namespace Bills_age_proof_l197_19777

variable {b t : ℚ}

theorem Bills_age_proof (h1 : b = 4 * t / 3) (h2 : b + 30 = 9 * (t + 30) / 8) : b = 24 := by 
  sorry

end Bills_age_proof_l197_19777


namespace num_diagonals_tetragon_l197_19787

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem num_diagonals_tetragon : num_diagonals_in_polygon 4 = 2 := by
  sorry

end num_diagonals_tetragon_l197_19787


namespace trains_clear_time_l197_19761

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

end trains_clear_time_l197_19761


namespace count_no_carry_pairs_l197_19757

theorem count_no_carry_pairs : 
  ∃ n, n = 1125 ∧ ∀ (a b : ℕ), (2000 ≤ a ∧ a < 2999 ∧ b = a + 1) → 
  (∀ i, (0 ≤ i ∧ i < 4) → ((a / (10 ^ i) % 10 + b / (10 ^ i) % 10) < 10)) := sorry

end count_no_carry_pairs_l197_19757


namespace solve_trig_equation_l197_19753

open Real

theorem solve_trig_equation (x : ℝ) (n : ℤ) :
  (2 * tan (6 * x) ^ 4 + 4 * sin (4 * x) * sin (8 * x) - cos (8 * x) - cos (16 * x) + 2) / sqrt (cos x - sqrt 3 * sin x) = 0 
  ∧ cos x - sqrt 3 * sin x > 0 →
  ∃ (k : ℤ), x = 2 * π * k ∨ x = -π / 6 + 2 * π * k ∨ x = -π / 3 + 2 * π * k ∨ x = -π / 2 + 2 * π * k ∨ x = -2 * π / 3 + 2 * π * k :=
sorry

end solve_trig_equation_l197_19753


namespace sum_of_coefficients_l197_19799

theorem sum_of_coefficients (a b c : ℝ) (w : ℂ) (h_roots : ∃ w : ℂ, (∃ i : ℂ, i^2 = -1) ∧ 
  (x + ax^2 + bx + c)^3 = (w + 3*im)* (w + 9*im)*(2*w - 4)) :
  a + b + c = -136 :=
sorry

end sum_of_coefficients_l197_19799


namespace sawing_steel_bar_time_l197_19727

theorem sawing_steel_bar_time (pieces : ℕ) (time_per_cut : ℕ) : 
  pieces = 6 → time_per_cut = 2 → (pieces - 1) * time_per_cut = 10 := 
by
  intros
  sorry

end sawing_steel_bar_time_l197_19727


namespace xy_product_l197_19725

theorem xy_product (x y z : ℝ) (h : x^2 + y^2 = x * y * (z + 1 / z)) :
  x = y * z ∨ y = x * z := 
by
  sorry

end xy_product_l197_19725


namespace edge_length_of_cubical_box_l197_19742

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

end edge_length_of_cubical_box_l197_19742


namespace circumradius_eq_l197_19735

noncomputable def circumradius (r : ℂ) (t1 t2 t3 : ℂ) : ℂ :=
  (2 * r ^ 4) / Complex.abs ((t1 + t2) * (t2 + t3) * (t3 + t1))

theorem circumradius_eq (r t1 t2 t3 : ℂ) (h_pos_r : r ≠ 0) :
  circumradius r t1 t2 t3 = (2 * r ^ 4) / Complex.abs ((t1 + t2) * (t2 + t3) * (t3 + t1)) :=
  by sorry

end circumradius_eq_l197_19735


namespace fraction_of_students_getting_F_l197_19751

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

end fraction_of_students_getting_F_l197_19751


namespace integer_solutions_of_system_l197_19764

theorem integer_solutions_of_system :
  {x : ℤ | - 2 * x + 7 < 10 ∧ (7 * x + 1) / 5 - 1 ≤ x} = {-1, 0, 1, 2} :=
by
  sorry

end integer_solutions_of_system_l197_19764


namespace exists_equilateral_triangle_l197_19776

variables {d1 d2 d3 : AffineSubspace ℝ (EuclideanSpace ℝ (Fin 2))}

theorem exists_equilateral_triangle (hne1 : d1 ≠ d2) (hne2 : d2 ≠ d3) (hne3 : d1 ≠ d3) : 
  ∃ (A1 A2 A3 : EuclideanSpace ℝ (Fin 2)), 
  (A1 ∈ d1 ∧ A2 ∈ d2 ∧ A3 ∈ d3) ∧ 
  dist A1 A2 = dist A2 A3 ∧ dist A2 A3 = dist A3 A1 := 
sorry

end exists_equilateral_triangle_l197_19776


namespace range_of_k_l197_19758

theorem range_of_k (k : ℤ) (x : ℤ) 
  (h1 : -4 * x - k ≤ 0) 
  (h2 : x = -1 ∨ x = -2) : 
  8 ≤ k ∧ k < 12 :=
sorry

end range_of_k_l197_19758


namespace correct_propositions_l197_19714

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

end correct_propositions_l197_19714


namespace smartphone_cost_l197_19729

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

end smartphone_cost_l197_19729


namespace least_number_subtracted_l197_19709

theorem least_number_subtracted (n m : ℕ) (h₁ : m = 2590) (h₂ : n = 2590 - 16) :
  (n % 9 = 6) ∧ (n % 11 = 6) ∧ (n % 13 = 6) :=
by
  sorry

end least_number_subtracted_l197_19709


namespace last_three_digits_7_pow_105_l197_19785

theorem last_three_digits_7_pow_105 : (7^105) % 1000 = 783 :=
  sorry

end last_three_digits_7_pow_105_l197_19785


namespace greatest_length_of_pieces_l197_19771

/-- Alicia has three ropes with lengths of 28 inches, 42 inches, and 70 inches.
She wants to cut these ropes into equal length pieces for her art project, and she doesn't want any leftover pieces.
Prove that the greatest length of each piece she can cut is 7 inches. -/
theorem greatest_length_of_pieces (a b c : ℕ) (h1 : a = 28) (h2 : b = 42) (h3 : c = 70) :
  ∃ (d : ℕ), d > 0 ∧ d ∣ a ∧ d ∣ b ∧ d ∣ c ∧ ∀ e : ℕ, e > 0 ∧ e ∣ a ∧ e ∣ b ∧ e ∣ c → e ≤ d := sorry

end greatest_length_of_pieces_l197_19771


namespace chemical_reaction_proof_l197_19769

-- Define the given number of moles for each reactant
def moles_NaOH : ℕ := 4
def moles_NH4Cl : ℕ := 3

-- Define the balanced chemical equation stoichiometry
def stoichiometry_ratio_NaOH_NH4Cl : ℕ := 1

-- Define the product formation based on the limiting reactant
theorem chemical_reaction_proof
  (moles_NaOH : ℕ)
  (moles_NH4Cl : ℕ)
  (stoichiometry_ratio_NaOH_NH4Cl : ℕ)
  (h1 : moles_NaOH = 4)
  (h2 : moles_NH4Cl = 3)
  (h3 : stoichiometry_ratio_NaOH_NH4Cl = 1):
  (3 = 3 * 1) ∧
  (3 = 3 * 1) ∧
  (3 = 3 * 1) ∧
  (3 = moles_NH4Cl) ∧
  (1 = moles_NaOH - moles_NH4Cl) :=
by {
  -- Provide assumptions based on the problem
  sorry
}

end chemical_reaction_proof_l197_19769


namespace age_of_youngest_child_l197_19756

theorem age_of_youngest_child (x : ℕ) (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 70) : x = 8 :=
by
  sorry

end age_of_youngest_child_l197_19756


namespace integer_pairs_solution_l197_19773

theorem integer_pairs_solution (x y : ℤ) (k : ℤ) :
  2 * x^2 - 6 * x * y + 3 * y^2 = -1 ↔
  ∃ n : ℤ, x = (2 + Real.sqrt 3)^k / 2 ∨ x = -(2 + Real.sqrt 3)^k / 2 ∧
           y = x + (2 + Real.sqrt 3)^k / (2 * Real.sqrt 3) ∨ 
           y = x - (2 + Real.sqrt 3)^k / (2 * Real.sqrt 3) :=
sorry

end integer_pairs_solution_l197_19773


namespace positive_difference_l197_19795

theorem positive_difference (x y : ℚ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 20) : y - x = 80 / 7 :=
by
  sorry

end positive_difference_l197_19795


namespace solution_replacement_concentration_l197_19774

theorem solution_replacement_concentration :
  ∀ (init_conc replaced_fraction new_conc replaced_conc : ℝ),
    init_conc = 0.45 → replaced_fraction = 0.5 → replaced_conc = 0.25 → new_conc = 35 →
    (init_conc - replaced_fraction * init_conc + replaced_fraction * replaced_conc) * 100 = new_conc :=
by
  intro init_conc replaced_fraction new_conc replaced_conc
  intros h_init h_frac h_replaced h_new
  rw [h_init, h_frac, h_replaced, h_new]
  sorry

end solution_replacement_concentration_l197_19774


namespace inequality_hold_l197_19791

variables (x y : ℝ)

theorem inequality_hold (h : x^4 + y^4 ≥ 2) : |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := 
sorry

end inequality_hold_l197_19791


namespace perpendicular_line_through_point_l197_19781

open Real

theorem perpendicular_line_through_point (B : ℝ × ℝ) (x y : ℝ) (c : ℝ)
  (hB : B = (3, 0)) (h_perpendicular : 2 * x + y - 5 = 0) :
  x - 2 * y + 3 = 0 :=
sorry

end perpendicular_line_through_point_l197_19781


namespace total_amount_of_money_l197_19744

def one_rupee_note_value := 1
def five_rupee_note_value := 5
def ten_rupee_note_value := 10

theorem total_amount_of_money (n : ℕ) 
  (h : 3 * n = 90) : n * one_rupee_note_value + n * five_rupee_note_value + n * ten_rupee_note_value = 480 :=
by
  sorry

end total_amount_of_money_l197_19744


namespace find_constants_monotonicity_l197_19783

noncomputable def f (x a b : ℝ) := (x^2 + a * x) * Real.exp x + b

theorem find_constants (a b : ℝ) (h_tangent : (f 0 a b = 1) ∧ (deriv (f · a b) 0 = -2)) :
  a = -2 ∧ b = 1 := by
  sorry

theorem monotonicity (a b : ℝ) (h_constants : a = -2 ∧ b = 1) :
  (∀ x : ℝ, (Real.exp x * (x^2 - 2) > 0 → x > Real.sqrt 2 ∨ x < -Real.sqrt 2)) ∧
  (∀ x : ℝ, (Real.exp x * (x^2 - 2) < 0 → -Real.sqrt 2 < x ∧ x < Real.sqrt 2)) := by
  sorry

end find_constants_monotonicity_l197_19783


namespace smallest_possible_value_of_sum_l197_19703

theorem smallest_possible_value_of_sum (a b : ℤ) (h1 : a > 6) (h2 : ∃ a' b', a' - b' = 4) : a + b < 11 := 
sorry

end smallest_possible_value_of_sum_l197_19703


namespace smallest_of_seven_consecutive_even_numbers_l197_19710

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

end smallest_of_seven_consecutive_even_numbers_l197_19710


namespace principal_amount_l197_19782

theorem principal_amount (r : ℝ) (n : ℕ) (t : ℕ) (A : ℝ) :
    r = 0.12 → n = 2 → t = 20 →
    ∃ P : ℝ, A = P * (1 + r / n)^(n * t) :=
by
  intros hr hn ht
  have P := A / (1 + r / n)^(n * t)
  use P
  sorry

end principal_amount_l197_19782


namespace total_games_played_l197_19716

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

end total_games_played_l197_19716


namespace ermias_balls_more_is_5_l197_19796

-- Define the conditions
def time_per_ball : ℕ := 20
def alexia_balls : ℕ := 20
def total_time : ℕ := 900

-- Define Ermias's balls
def ermias_balls_more (x : ℕ) : ℕ := alexia_balls + x

-- Alexia's total inflation time
def alexia_total_time : ℕ := alexia_balls * time_per_ball

-- Ermias's total inflation time given x more balls than Alexia
def ermias_total_time (x : ℕ) : ℕ := (ermias_balls_more x) * time_per_ball

-- Total time taken by both Alexia and Ermias
def combined_time (x : ℕ) : ℕ := alexia_total_time + ermias_total_time x

-- Proven that Ermias inflated 5 more balls than Alexia given the total time condition
theorem ermias_balls_more_is_5 : (∃ x : ℕ, combined_time x = total_time) := 
by {
  sorry
}

end ermias_balls_more_is_5_l197_19796


namespace greatest_divisible_by_13_l197_19770

def is_distinct_nonzero_digits (A B C : ℕ) : Prop :=
  0 < A ∧ A < 10 ∧ 0 < B ∧ B < 10 ∧ 0 < C ∧ C < 10 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C

def number (A B C : ℕ) : ℕ :=
  10000 * A + 1000 * B + 100 * C + 10 * B + A

theorem greatest_divisible_by_13 :
  ∃ (A B C : ℕ), is_distinct_nonzero_digits A B C ∧ number A B C % 13 = 0 ∧ number A B C = 96769 :=
sorry

end greatest_divisible_by_13_l197_19770


namespace sqrt_x_plus_sqrt_inv_x_l197_19724

theorem sqrt_x_plus_sqrt_inv_x (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) : 
  (Real.sqrt x + 1 / Real.sqrt x) = Real.sqrt 52 := 
by
  sorry

end sqrt_x_plus_sqrt_inv_x_l197_19724


namespace discount_savings_difference_l197_19745

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

end discount_savings_difference_l197_19745


namespace imaginary_part_of_z_l197_19721

-- Let 'z' be the complex number \(\frac {2i}{1-i}\)
noncomputable def z : ℂ := (2 * Complex.I) / (1 - Complex.I)

theorem imaginary_part_of_z :
  z.im = 1 :=
sorry

end imaginary_part_of_z_l197_19721


namespace angle_equality_l197_19792

variables {Point Circle : Type}
variables (K O1 O2 P1 P2 Q1 Q2 M1 M2 : Point)
variables (W1 W2 : Circle)
variables (midpoint : Point → Point → Point)
variables (is_center : Point → Circle → Prop)
variables (intersects_at : Circle → Circle → Point → Prop)
variables (common_tangent_points : Circle → Circle → (Point × Point) × (Point × Point) → Prop)
variables (intersect_circle_at : Circle → Line → Point → Point → Prop)
variables (angle : Point → Point → Point → ℝ) -- to denote the angle measure between three points

-- Conditions
axiom K_intersection : intersects_at W1 W2 K
axiom O1_center : is_center O1 W1
axiom O2_center : is_center O2 W2
axiom tangents_meet_at : common_tangent_points W1 W2 ((P1, Q1), (P2, Q2))
axiom M1_midpoint : M1 = midpoint P1 Q1
axiom M2_midpoint : M2 = midpoint P2 Q2

-- The statement to prove
theorem angle_equality : angle O1 K O2 = angle M1 K M2 := 
  sorry

end angle_equality_l197_19792


namespace two_digit_numbers_reverse_square_condition_l197_19737

theorem two_digit_numbers_reverse_square_condition :
  ∀ (a b : ℕ), 0 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 →
  (∃ n : ℕ, 10 * a + b + 10 * b + a = n^2) ↔ 
  (10 * a + b = 29 ∨ 10 * a + b = 38 ∨ 10 * a + b = 47 ∨ 10 * a + b = 56 ∨ 
   10 * a + b = 65 ∨ 10 * a + b = 74 ∨ 10 * a + b = 83 ∨ 10 * a + b = 92) :=
by {
  sorry
}

end two_digit_numbers_reverse_square_condition_l197_19737


namespace probability_at_least_one_shows_one_is_correct_l197_19798

/-- Two fair 8-sided dice are rolled. What is the probability that at least one of the dice shows a 1? -/
def probability_at_least_one_shows_one : ℚ :=
  let total_outcomes := 8 * 8
  let neither_one := 7 * 7
  let at_least_one := total_outcomes - neither_one
  at_least_one / total_outcomes

theorem probability_at_least_one_shows_one_is_correct :
  probability_at_least_one_shows_one = 15 / 64 :=
by
  unfold probability_at_least_one_shows_one
  sorry

end probability_at_least_one_shows_one_is_correct_l197_19798


namespace sin_neg_135_eq_neg_sqrt_2_over_2_l197_19775

theorem sin_neg_135_eq_neg_sqrt_2_over_2 :
  Real.sin (-135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_neg_135_eq_neg_sqrt_2_over_2_l197_19775


namespace total_gain_is_19200_l197_19726

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

end total_gain_is_19200_l197_19726


namespace ceil_minus_eq_zero_l197_19763

theorem ceil_minus_eq_zero (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 0) : ⌈x⌉ - x = 0 :=
sorry

end ceil_minus_eq_zero_l197_19763


namespace arcsin_arccos_interval_l197_19739

open Real
open Set

theorem arcsin_arccos_interval (x y : ℝ) (h : x^2 + y^2 = 1) : 
  ∃ t ∈ Icc (-3 * π / 2) (π / 2), 2 * arcsin x - arccos y = t := 
sorry

end arcsin_arccos_interval_l197_19739


namespace molecular_weight_BaBr2_l197_19700

theorem molecular_weight_BaBr2 (w: ℝ) (h: w = 2376) : w / 8 = 297 :=
by
  sorry

end molecular_weight_BaBr2_l197_19700


namespace cyclist_final_speed_l197_19741

def u : ℝ := 16
def a : ℝ := 0.5
def t : ℕ := 7200

theorem cyclist_final_speed : 
  (u + a * t) * 3.6 = 13017.6 := by
  sorry

end cyclist_final_speed_l197_19741


namespace negation_exists_implication_l197_19750

theorem negation_exists_implication (x : ℝ) : (¬ ∃ y > 0, y^2 - 2*y - 3 ≤ 0) ↔ ∀ y > 0, y^2 - 2*y - 3 > 0 :=
by
  sorry

end negation_exists_implication_l197_19750


namespace stamps_per_book_type2_eq_15_l197_19749

-- Defining the conditions
def num_books_type1 : ℕ := 4
def stamps_per_book_type1 : ℕ := 10
def num_books_type2 : ℕ := 6
def total_stamps : ℕ := 130

-- Stating the theorem to prove the number of stamps in each book of the second type is 15
theorem stamps_per_book_type2_eq_15 : 
  ∀ (x : ℕ), 
    (num_books_type1 * stamps_per_book_type1 + num_books_type2 * x = total_stamps) → 
    x = 15 :=
by
  sorry

end stamps_per_book_type2_eq_15_l197_19749


namespace sum_of_areas_of_triangles_l197_19760

noncomputable def triangle_sum_of_box (a b c : ℝ) :=
  let face_triangles_area := 4 * ((a * b + a * c + b * c) / 2)
  let perpendicular_triangles_area := 4 * ((a * c + b * c) / 2)
  let oblique_triangles_area := 8 * Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))
  face_triangles_area + perpendicular_triangles_area + oblique_triangles_area

theorem sum_of_areas_of_triangles :
  triangle_sum_of_box 2 3 4 = 168 + k * Real.sqrt p := sorry

end sum_of_areas_of_triangles_l197_19760


namespace problem_solution_l197_19797

section
variables (a b : ℝ)

-- Definition of the \* operation
def star_op (a b : ℝ) : ℝ := (a + 1) * (b - 1)

-- Definition of a^{*2} as a \* a
def star_square (a : ℝ) : ℝ := star_op a a

-- Define the specific problem instance with x = 2
def problem_expr : ℝ := star_op 3 (star_square 2) - star_op 2 2 + 1

-- Theorem stating the correct answer
theorem problem_solution : problem_expr = 6 := by
  -- Proof steps, marked as 'sorry'
  sorry

end

end problem_solution_l197_19797


namespace only_possible_b_l197_19790

theorem only_possible_b (b : ℕ) (h : ∃ a k l : ℕ, k ≠ l ∧ (b > 0) ∧ (a > 0) ∧ (b ^ (k + l)) ∣ (a ^ k + b ^ l) ∧ (b ^ (k + l)) ∣ (a ^ l + b ^ k)) : 
  b = 1 :=
sorry

end only_possible_b_l197_19790


namespace arithmetic_sequence_formula_l197_19718

theorem arithmetic_sequence_formula (a : ℕ → ℤ) (d : ℤ) :
  (a 3 = 4) → (d = -2) → ∀ n : ℕ, a n = 10 - 2 * n :=
by
  intros h1 h2 n
  sorry

end arithmetic_sequence_formula_l197_19718


namespace apples_chosen_l197_19722

def total_fruits : ℕ := 12
def bananas : ℕ := 4
def oranges : ℕ := 5
def total_other_fruits := bananas + oranges

theorem apples_chosen : total_fruits - total_other_fruits = 3 :=
by sorry

end apples_chosen_l197_19722


namespace number_division_remainder_l197_19752

theorem number_division_remainder (N k m : ℤ) (h1 : N = 281 * k + 160) (h2 : N = D * m + 21) : D = 139 :=
by sorry

end number_division_remainder_l197_19752


namespace dealer_is_cheating_l197_19702

variable (w a : ℝ)
noncomputable def measured_weight (w : ℝ) (a : ℝ) : ℝ :=
  (a * w + w / a) / 2

theorem dealer_is_cheating (h : a > 0) : measured_weight w a ≥ w :=
by
  sorry

end dealer_is_cheating_l197_19702


namespace length_AC_and_area_OAC_l197_19754

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

end length_AC_and_area_OAC_l197_19754


namespace g_h_2_equals_584_l197_19746

def g (x : ℝ) : ℝ := 3 * x^2 - 4
def h (x : ℝ) : ℝ := -2 * x^3 + 2

theorem g_h_2_equals_584 : g (h 2) = 584 := 
by 
  sorry

end g_h_2_equals_584_l197_19746


namespace find_m_of_parallel_vectors_l197_19704

theorem find_m_of_parallel_vectors (m : ℝ) 
  (a : ℝ × ℝ := (1, 2)) 
  (b : ℝ × ℝ := (m, m + 1))
  (parallel : a.1 * b.2 = a.2 * b.1) :
  m = 1 :=
by
  -- We assume a parallel condition and need to prove m = 1
  sorry

end find_m_of_parallel_vectors_l197_19704


namespace range_of_independent_variable_of_sqrt_l197_19715

theorem range_of_independent_variable_of_sqrt (x : ℝ) : (2 * x - 3 ≥ 0) ↔ (x ≥ 3 / 2) := sorry

end range_of_independent_variable_of_sqrt_l197_19715


namespace find_a_l197_19728

-- The conditions converted to Lean definitions
variable (a : ℝ)
variable (α : ℝ)
variable (point_on_terminal_side : a ≠ 0 ∧ (∃ α, tan α = -1 / 2 ∧ ∀ y : ℝ, y = -1 → a = 2 * y) )

-- The theorem statement
theorem find_a (H : point_on_terminal_side): a = 2 := by
  sorry

end find_a_l197_19728


namespace product_of_roots_l197_19768

variable {x1 x2 : ℝ}

theorem product_of_roots (h : ∀ x, -x^2 + 3*x = 0 → (x = x1 ∨ x = x2)) :
  x1 * x2 = 0 :=
by
  sorry

end product_of_roots_l197_19768


namespace average_speed_jeffrey_l197_19784
-- Import the necessary Lean library.

-- Initial conditions in the problem, restated as Lean definitions.
def distance_jog (d : ℝ) : Prop := d = 3
def speed_jog (s : ℝ) : Prop := s = 4
def distance_walk (d : ℝ) : Prop := d = 4
def speed_walk (s : ℝ) : Prop := s = 3

-- Target statement to prove using Lean.
theorem average_speed_jeffrey :
  ∀ (dj sj dw sw : ℝ), distance_jog dj → speed_jog sj → distance_walk dw → speed_walk sw →
    (dj + dw) / ((dj / sj) + (dw / sw)) = 3.36 := 
  by
    intros dj sj dw sw hj hs hw hw
    sorry

end average_speed_jeffrey_l197_19784


namespace smallest_b_1111_is_square_l197_19711

theorem smallest_b_1111_is_square : 
  ∃ b : ℕ, b > 0 ∧ (∀ n : ℕ, (b^3 + b^2 + b + 1 = n^2 → b = 7)) :=
by
  sorry

end smallest_b_1111_is_square_l197_19711


namespace find_b_l197_19701

-- Define complex numbers z1 and z2
def z1 (b : ℝ) : Complex := Complex.mk 3 (-b)

def z2 : Complex := Complex.mk 1 (-2)

-- Statement that needs to be proved
theorem find_b (b : ℝ) (h : (z1 b / z2).re = 0) : b = -3 / 2 :=
by
  -- proof goes here
  sorry

end find_b_l197_19701


namespace smallest_gcd_l197_19786

theorem smallest_gcd (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (H1 : Nat.gcd x y = 270) (H2 : Nat.gcd x z = 105) : Nat.gcd y z = 15 :=
sorry

end smallest_gcd_l197_19786


namespace parallel_line_with_intercept_sum_l197_19740

theorem parallel_line_with_intercept_sum (c : ℝ) :
  (∀ x y : ℝ, 2 * x + 3 * y + 5 = 0 → 2 * x + 3 * y + c = 0) ∧ 
  (-c / 3 - c / 2 = 6) → 
  (10 * x + 15 * y - 36 = 0) :=
by
  sorry

end parallel_line_with_intercept_sum_l197_19740


namespace cream_strawberry_prices_l197_19748

noncomputable def price_flavor_B : ℝ := 30
noncomputable def price_flavor_A : ℝ := 40

theorem cream_strawberry_prices (x y : ℝ) 
  (h1 : y = x + 10) 
  (h2 : 800 / y = 600 / x) : 
  x = price_flavor_B ∧ y = price_flavor_A :=
by 
  sorry

end cream_strawberry_prices_l197_19748


namespace O_l197_19736

theorem O'Hara_triple_49_16_y : 
  (∃ y : ℕ, (49 : ℕ).sqrt + (16 : ℕ).sqrt = y) → y = 11 :=
by
  sorry

end O_l197_19736


namespace sequence_6th_term_l197_19766

theorem sequence_6th_term 
    (a₁ a₂ a₃ a₄ a₅ a₆ : ℚ)
    (h₁ : a₁ = 3)
    (h₅ : a₅ = 54)
    (h₂ : a₂ = (a₁ + a₃) / 3)
    (h₃ : a₃ = (a₂ + a₄) / 3)
    (h₄ : a₄ = (a₃ + a₅) / 3)
    (h₆ : a₅ = (a₄ + a₆) / 3) :
    a₆ = 1133 / 7 :=
by
  sorry

end sequence_6th_term_l197_19766


namespace wrapping_cube_wrapping_prism_a_wrapping_prism_b_l197_19788

theorem wrapping_cube (ways_cube : ℕ) :
  ways_cube = 3 :=
  sorry

theorem wrapping_prism_a (ways_prism_a : ℕ) (a : ℝ) :
  (ways_prism_a = 5) ↔ (a > 0) :=
  sorry

theorem wrapping_prism_b (ways_prism_b : ℕ) (b : ℝ) :
  (ways_prism_b = 7) ↔ (b > 0) :=
  sorry

end wrapping_cube_wrapping_prism_a_wrapping_prism_b_l197_19788


namespace max_value_of_expression_l197_19708

theorem max_value_of_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h_sum : x + y + z = 1) :
  x^4 * y^3 * z^2 ≤ 1024 / 14348907 :=
sorry

end max_value_of_expression_l197_19708


namespace max_value_of_function_l197_19707

noncomputable def function (x : ℝ) : ℝ := (Real.cos x) ^ 2 - Real.sin x

theorem max_value_of_function : ∃ x : ℝ, function x = 5 / 4 :=
by
  sorry

end max_value_of_function_l197_19707


namespace ratio_of_areas_l197_19719

noncomputable def side_length_C := 24 -- cm
noncomputable def side_length_D := 54 -- cm
noncomputable def ratio_areas := (side_length_C / side_length_D) ^ 2

theorem ratio_of_areas : ratio_areas = 16 / 81 := sorry

end ratio_of_areas_l197_19719


namespace farmer_land_acres_l197_19755

theorem farmer_land_acres
  (initial_ratio_corn : Nat)
  (initial_ratio_sugar_cane : Nat)
  (initial_ratio_tobacco : Nat)
  (new_ratio_corn : Nat)
  (new_ratio_sugar_cane : Nat)
  (new_ratio_tobacco : Nat)
  (additional_tobacco_acres : Nat)
  (total_land_acres : Nat) :
  initial_ratio_corn = 5 →
  initial_ratio_sugar_cane = 2 →
  initial_ratio_tobacco = 2 →
  new_ratio_corn = 2 →
  new_ratio_sugar_cane = 2 →
  new_ratio_tobacco = 5 →
  additional_tobacco_acres = 450 →
  total_land_acres = 1350 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end farmer_land_acres_l197_19755


namespace customers_in_each_car_l197_19772

-- Conditions given in the problem
def cars : ℕ := 10
def purchases_sports : ℕ := 20
def purchases_music : ℕ := 30

-- Total purchases are equal to the total number of customers
def total_purchases : ℕ := purchases_sports + purchases_music
def total_customers (C : ℕ) : ℕ := cars * C

-- Lean statement to prove that the number of customers in each car is 5
theorem customers_in_each_car : (∃ C : ℕ, total_customers C = total_purchases) ∧ (∀ C : ℕ, total_customers C = total_purchases → C = 5) :=
by
  sorry

end customers_in_each_car_l197_19772


namespace Susan_has_10_dollars_left_l197_19747

def initial_amount : ℝ := 80
def food_expense : ℝ := 15
def rides_expense : ℝ := 3 * food_expense
def games_expense : ℝ := 10
def total_expense : ℝ := food_expense + rides_expense + games_expense
def remaining_amount : ℝ := initial_amount - total_expense

theorem Susan_has_10_dollars_left : remaining_amount = 10 := by
  sorry

end Susan_has_10_dollars_left_l197_19747


namespace average_payment_is_460_l197_19717

theorem average_payment_is_460 :
  let n := 52
  let first_payment := 410
  let extra := 65
  let num_first_payments := 12
  let num_rest_payments := n - num_first_payments
  let rest_payment := first_payment + extra
  (num_first_payments * first_payment + num_rest_payments * rest_payment) / n = 460 := by
  sorry

end average_payment_is_460_l197_19717


namespace trapezoid_diagonals_l197_19762

theorem trapezoid_diagonals (AD BC : ℝ) (angle_DAB angle_BCD : ℝ)
  (hAD : AD = 8) (hBC : BC = 6) (h_angle_DAB : angle_DAB = 90)
  (h_angle_BCD : angle_BCD = 120) :
  ∃ AC BD : ℝ, AC = 4 * Real.sqrt 3 ∧ BD = 2 * Real.sqrt 19 :=
by
  sorry

end trapezoid_diagonals_l197_19762


namespace equivalent_expression_l197_19794

-- Define the conditions and the statement that needs to be proven
theorem equivalent_expression (x : ℝ) (h : x^2 - 2 * x + 1 = 0) : 2 * x^2 - 4 * x = -2 := 
  by
    sorry

end equivalent_expression_l197_19794


namespace rectangle_length_from_square_thread_l197_19731

theorem rectangle_length_from_square_thread (side_of_square width_of_rectangle : ℝ) (same_thread : Bool) 
  (h1 : side_of_square = 20) (h2 : width_of_rectangle = 14) (h3 : same_thread) : 
  ∃ length_of_rectangle : ℝ, length_of_rectangle = 26 := 
by
  sorry

end rectangle_length_from_square_thread_l197_19731


namespace problem1_problem2_l197_19720

theorem problem1 (a b x y : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < x ∧ 0 < y) : 
  (a^2 / x + b^2 / y) ≥ ((a + b)^2 / (x + y)) ∧ (a * y = b * x → (a^2 / x + b^2 / y) = ((a + b)^2 / (x + y))) :=
sorry

theorem problem2 (x : ℝ) (h : 0 < x ∧ x < 1 / 2) :
  (∀ x, 0 < x ∧ x < 1 / 2 → ((2 / x + 9 / (1 - 2 * x)) ≥ 25)) ∧ (2 * (1 - 2 * (1 / 5)) = 9 * (1 / 5) → (2 / (1 / 5) + 9 / (1 - 2 * (1 / 5)) = 25)) :=
sorry

end problem1_problem2_l197_19720


namespace remainder_when_divided_by_15_l197_19738

theorem remainder_when_divided_by_15 (N : ℕ) (h1 : N % 60 = 49) : N % 15 = 4 :=
by
  sorry

end remainder_when_divided_by_15_l197_19738


namespace find_sinD_l197_19733

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

end find_sinD_l197_19733


namespace Carlson_max_jars_l197_19767

theorem Carlson_max_jars (n a : ℕ) (hn : 13 * n = 5 * (8 * n + 9 * a)) : ∃ k : ℕ, k ≤ 23 := 
sorry

end Carlson_max_jars_l197_19767


namespace sum_of_perimeters_l197_19713

theorem sum_of_perimeters (x y : Real) 
  (h1 : x^2 + y^2 = 85)
  (h2 : x^2 - y^2 = 45) :
  4 * (Real.sqrt 65 + 2 * Real.sqrt 5) = 4 * x + 4 * y := by
  sorry

end sum_of_perimeters_l197_19713


namespace cookies_per_child_l197_19732

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

end cookies_per_child_l197_19732


namespace border_area_correct_l197_19765

noncomputable def area_of_border (poster_height poster_width border_width : ℕ) : ℕ :=
  let framed_height := poster_height + 2 * border_width
  let framed_width := poster_width + 2 * border_width
  (framed_height * framed_width) - (poster_height * poster_width)

theorem border_area_correct :
  area_of_border 12 16 4 = 288 :=
by
  rfl

end border_area_correct_l197_19765


namespace residue_7_1234_mod_13_l197_19780

theorem residue_7_1234_mod_13 :
  (7 : ℤ) ^ 1234 % 13 = 12 :=
by
  -- given conditions as definitions
  have h1 : (7 : ℤ) % 13 = 7 := by norm_num
  
  -- auxiliary calculations
  have h2 : (49 : ℤ) % 13 = 10 := by norm_num
  have h3 : (100 : ℤ) % 13 = 9 := by norm_num
  have h4 : (81 : ℤ) % 13 = 3 := by norm_num
  have h5 : (729 : ℤ) % 13 = 1 := by norm_num

  -- the actual problem we want to prove
  sorry

end residue_7_1234_mod_13_l197_19780


namespace biking_days_in_week_l197_19789

def onurDistancePerDay : ℕ := 250
def hanilDistanceMorePerDay : ℕ := 40
def weeklyDistance : ℕ := 2700

theorem biking_days_in_week : (weeklyDistance / (onurDistancePerDay + hanilDistanceMorePerDay + onurDistancePerDay)) = 5 :=
by
  sorry

end biking_days_in_week_l197_19789


namespace unique_integer_sequence_exists_l197_19778

open Nat

def a (n : ℕ) : ℤ := sorry

theorem unique_integer_sequence_exists :
  ∃ (a : ℕ → ℤ), a 1 = 1 ∧ a 2 > 1 ∧ (∀ n ≥ 1, (a (n+1))^3 + 1 = a n * a (n+2)) ∧
  (∀ b, (b 1 = 1) → (b 2 > 1) → (∀ n ≥ 1, (b (n+1))^3 + 1 = b n * b (n+2)) → b = a) :=
by
  sorry

end unique_integer_sequence_exists_l197_19778


namespace platform_length_is_150_l197_19712

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

end platform_length_is_150_l197_19712


namespace liu_xing_statement_incorrect_l197_19723

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

end liu_xing_statement_incorrect_l197_19723


namespace total_savings_l197_19734

def liam_oranges : ℕ := 40
def liam_price_per_two : ℝ := 2.50
def claire_oranges : ℕ := 30
def claire_price_per_one : ℝ := 1.20

theorem total_savings : (liam_oranges / 2 * liam_price_per_two) + (claire_oranges * claire_price_per_one) = 86 := by
  sorry

end total_savings_l197_19734


namespace maximum_x_minus_y_l197_19705

theorem maximum_x_minus_y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) : x - y ≤ 2 * Real.sqrt 3 / 3 := by
  sorry

end maximum_x_minus_y_l197_19705


namespace problem_eqn_l197_19793

theorem problem_eqn (a b c : ℝ) :
  (∃ r₁ r₂ : ℝ, r₁^2 + 3 * r₁ - 1 = 0 ∧ r₂^2 + 3 * r₂ - 1 = 0) ∧
  (∀ x : ℝ, (x^2 + 3 * x - 1 = 0) → (x^4 + a * x^2 + b * x + c = 0)) →
  a + b + 4 * c = -7 :=
by
  sorry

end problem_eqn_l197_19793


namespace evaluate_expression_l197_19730

theorem evaluate_expression : 4 * 6 * 8 + 24 / 4 - 2 = 196 := by
  sorry

end evaluate_expression_l197_19730


namespace time_to_return_l197_19759

-- Given conditions
def distance : ℝ := 1000
def return_speed : ℝ := 142.85714285714286

-- Goal to prove
theorem time_to_return : distance / return_speed = 7 := 
by
  sorry

end time_to_return_l197_19759


namespace Micheal_work_rate_l197_19743

theorem Micheal_work_rate 
    (M A : ℕ) 
    (h1 : 1 / M + 1 / A = 1 / 20)
    (h2 : 9 / 200 = 1 / A) : M = 200 :=
by
    sorry

end Micheal_work_rate_l197_19743
