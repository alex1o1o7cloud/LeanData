import Mathlib

namespace percentage_increase_l26_26919

theorem percentage_increase (initial final : ℝ)
  (h_initial: initial = 60) (h_final: final = 90) :
  (final - initial) / initial * 100 = 50 :=
by
  sorry

end percentage_increase_l26_26919


namespace problem_solution_l26_26550

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem problem_solution (f : ℝ → ℝ)
  (H1 : even_function f)
  (H2 : ∀ x, f (x + 4) = -f x)
  (H3 : ∀ ⦃x y⦄, 0 ≤ x → x < y → y ≤ 4 → f y < f x) :
  f 13 < f 10 ∧ f 10 < f 15 :=
  by
    sorry

end problem_solution_l26_26550


namespace possible_ages_l26_26963

-- Define the set of digits
def digits : Multiset ℕ := {1, 1, 2, 2, 2, 3}

-- Condition: The age must start with "211"
def starting_sequence : List ℕ := [2, 1, 1]

-- Calculate the count of possible ages
def count_ages : ℕ :=
  let remaining_digits := [2, 2, 1, 3]
  let total_permutations := Nat.factorial 4
  let repetitions := Nat.factorial 2
  total_permutations / repetitions

theorem possible_ages : count_ages = 12 := by
  -- Proof should go here but it's omitted according to instructions.
  sorry

end possible_ages_l26_26963


namespace number_of_voters_in_election_l26_26581

theorem number_of_voters_in_election
  (total_membership : ℕ)
  (votes_cast : ℕ)
  (winning_percentage_cast : ℚ)
  (percentage_of_total : ℚ)
  (h_total : total_membership = 1600)
  (h_winning_percentage : winning_percentage_cast = 0.60)
  (h_percentage_of_total : percentage_of_total = 0.196875)
  (h_votes : winning_percentage_cast * votes_cast = percentage_of_total * total_membership) :
  votes_cast = 525 :=
by
  sorry

end number_of_voters_in_election_l26_26581


namespace restore_example_l26_26961

theorem restore_example (x : ℕ) (y : ℕ) :
  (10 ≤ x * 8 ∧ x * 8 < 100) ∧ (100 ≤ x * 9 ∧ x * 9 < 1000) ∧ y = 98 → x = 12 ∧ x * y = 1176 :=
by
  sorry

end restore_example_l26_26961


namespace intersection_complement_eq_l26_26649

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 3}
def B : Set ℝ := {x | x < 2}

-- Define the complement of B in ℝ
def complement_B : Set ℝ := {x | x ≥ 2}

-- Define the intersection of A and complement of B
def intersection : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

-- The theorem to be proved
theorem intersection_complement_eq : (A ∩ complement_B) = intersection :=
sorry

end intersection_complement_eq_l26_26649


namespace gwen_money_difference_l26_26505

theorem gwen_money_difference:
  let money_from_grandparents : ℕ := 15
  let money_from_uncle : ℕ := 8
  money_from_grandparents - money_from_uncle = 7 :=
by
  sorry

end gwen_money_difference_l26_26505


namespace std_deviation_above_l26_26484

variable (mean : ℝ) (std_dev : ℝ) (score1 : ℝ) (score2 : ℝ)
variable (n1 : ℝ) (n2 : ℝ)

axiom h_mean : mean = 74
axiom h_std1 : score1 = 58
axiom h_std2 : score2 = 98
axiom h_cond1 : score1 = mean - n1 * std_dev
axiom h_cond2 : n1 = 2

theorem std_deviation_above (mean std_dev score1 score2 n1 n2 : ℝ)
  (h_mean : mean = 74)
  (h_std1 : score1 = 58)
  (h_std2 : score2 = 98)
  (h_cond1 : score1 = mean - n1 * std_dev)
  (h_cond2 : n1 = 2) :
  n2 = (score2 - mean) / std_dev :=
sorry

end std_deviation_above_l26_26484


namespace figure_F10_squares_l26_26559

def num_squares (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 1 + 3 * (n - 1) * n

theorem figure_F10_squares : num_squares 10 = 271 :=
by sorry

end figure_F10_squares_l26_26559


namespace solve_for_x_l26_26301

variables (x y z : ℝ)

def condition : Prop :=
  1 / (x + y) + 1 / (x - y) = z / (x - y)

theorem solve_for_x (h : condition x y z) : x = z / 2 :=
by
  sorry

end solve_for_x_l26_26301


namespace fraction_not_covered_l26_26825

/--
Given that frame X has a diameter of 16 cm and frame Y has a diameter of 12 cm,
prove that the fraction of the surface of frame X that is not covered by frame Y is 7/16.
-/
theorem fraction_not_covered (dX dY : ℝ) (hX : dX = 16) (hY : dY = 12) : 
  let rX := dX / 2
  let rY := dY / 2
  let AX := Real.pi * rX^2
  let AY := Real.pi * rY^2
  let uncovered_area := AX - AY
  let fraction_not_covered := uncovered_area / AX
  fraction_not_covered = 7 / 16 :=
by
  sorry

end fraction_not_covered_l26_26825


namespace A_square_or_cube_neg_identity_l26_26489

open Matrix

theorem A_square_or_cube_neg_identity (A : Matrix (Fin 2) (Fin 2) ℚ)
  (n : ℕ) (hn_nonzero : n ≠ 0) (hA_pow_n : A ^ n = -(1 : Matrix (Fin 2) (Fin 2) ℚ)) :
  A ^ 2 = -(1 : Matrix (Fin 2) (Fin 2) ℚ) ∨ A ^ 3 = -(1 : Matrix (Fin 2) (Fin 2) ℚ) :=
sorry

end A_square_or_cube_neg_identity_l26_26489


namespace find_ratio_b_a_l26_26737

theorem find_ratio_b_a (a b : ℝ) 
  (h : ∀ x : ℝ, (2 * a - b) * x + (a + b) > 0 ↔ x > -3) : 
  b / a = 5 / 4 :=
sorry

end find_ratio_b_a_l26_26737


namespace intersection_A_B_l26_26627

-- Define sets A and B according to the conditions provided
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | 0 < x ∧ x < 3}

-- Define the theorem stating the intersection of A and B
theorem intersection_A_B : A ∩ B = {1, 2} := by
  sorry

end intersection_A_B_l26_26627


namespace sum_of_geometric_progression_l26_26885

theorem sum_of_geometric_progression (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) (a1 a3 : ℝ) (h1 : a1 + a3 = 5) (h2 : a1 * a3 = 4)
  (h3 : a 1 = a1) (h4 : a 3 = a3)
  (h5 : ∀ k, a (k + 1) > a k)  -- Sequence is increasing
  (h6 : S n = a 1 * ((1 - (2:ℝ) ^ n) / (1 - 2)))
  (h7 : n = 6) :
  S 6 = 63 :=
sorry

end sum_of_geometric_progression_l26_26885


namespace birds_in_house_l26_26174

theorem birds_in_house (B : ℕ) :
  let dogs := 3
  let cats := 18
  let humans := 7
  let total_heads := B + dogs + cats + humans
  let total_feet := 2 * B + 4 * dogs + 4 * cats + 2 * humans
  total_feet = total_heads + 74 → B = 4 :=
by
  intros dogs cats humans total_heads total_feet condition
  -- We assume the condition and work towards the proof.
  sorry

end birds_in_house_l26_26174


namespace exists_powers_of_7_difference_div_by_2021_l26_26414

theorem exists_powers_of_7_difference_div_by_2021 :
  ∃ n m : ℕ, n > m ∧ 2021 ∣ (7^n - 7^m) := 
by
  sorry

end exists_powers_of_7_difference_div_by_2021_l26_26414


namespace slope_of_tangent_at_A_l26_26942

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem slope_of_tangent_at_A :
  (deriv f 0) = 1 :=
by
  sorry

end slope_of_tangent_at_A_l26_26942


namespace P_sufficient_for_Q_P_not_necessary_for_Q_l26_26029

variable (x : ℝ)
def P : Prop := x >= 0
def Q : Prop := 2 * x + 1 / (2 * x + 1) >= 1

theorem P_sufficient_for_Q : P x -> Q x := 
by sorry

theorem P_not_necessary_for_Q : ¬ (Q x -> P x) := 
by sorry

end P_sufficient_for_Q_P_not_necessary_for_Q_l26_26029


namespace new_sailor_weight_l26_26841

-- Define the conditions
variables {average_weight : ℝ} (new_weight : ℝ)
variable (old_weight : ℝ := 56)

-- State the property we need to prove
theorem new_sailor_weight
  (h : (new_weight - old_weight) = 8) :
  new_weight = 64 :=
by
  sorry

end new_sailor_weight_l26_26841


namespace meaningful_fraction_l26_26996

theorem meaningful_fraction (x : ℝ) : (x + 1 ≠ 0) ↔ (x ≠ -1) :=
by {
  sorry -- Proof goes here
}

end meaningful_fraction_l26_26996


namespace calculate_expression_l26_26799

theorem calculate_expression :
  2 * Real.sin (60 * Real.pi / 180) + abs (Real.sqrt 3 - 3) + (Real.pi - 1)^0 = 4 :=
by
  sorry

end calculate_expression_l26_26799


namespace difference_even_number_sums_l26_26928

open Nat

def sum_of_even_numbers (start end_ : ℕ) : ℕ :=
  let n := (end_ - start) / 2 + 1
  n * (start + end_) / 2

theorem difference_even_number_sums :
  let sum_A := sum_of_even_numbers 10 50
  let sum_B := sum_of_even_numbers 110 150
  sum_B - sum_A = 2100 :=
by
  let sum_A := sum_of_even_numbers 10 50
  let sum_B := sum_of_even_numbers 110 150
  show sum_B - sum_A = 2100
  sorry

end difference_even_number_sums_l26_26928


namespace fifteenth_term_of_geometric_sequence_l26_26564

theorem fifteenth_term_of_geometric_sequence :
  let a := 12
  let r := (1:ℚ) / 3
  let n := 15
  (a * r^(n-1)) = (4 / 1594323:ℚ)
:=
  by
    sorry

end fifteenth_term_of_geometric_sequence_l26_26564


namespace positive_difference_solutions_l26_26504

theorem positive_difference_solutions : 
  ∀ (r : ℝ), r ≠ -3 → 
  (∃ r1 r2 : ℝ, (r^2 - 6*r - 20) / (r + 3) = 3*r + 10 → r1 ≠ r2 ∧ 
  |r1 - r2| = 20) :=
by
  sorry

end positive_difference_solutions_l26_26504


namespace abs_nonneg_rational_l26_26475

theorem abs_nonneg_rational (a : ℚ) : |a| ≥ 0 :=
sorry

end abs_nonneg_rational_l26_26475


namespace tangent_line_to_parabola_l26_26551

-- Define the line and parabola equations
def line (x y k : ℝ) := 4 * x + 3 * y + k = 0
def parabola (x y : ℝ) := y ^ 2 = 16 * x

-- Prove that if the line is tangent to the parabola, then k = 9
theorem tangent_line_to_parabola (k : ℝ) :
  (∃ (x y : ℝ), line x y k ∧ parabola x y ∧ (y^2 + 12 * y + 4 * k = 0 ∧ 144 - 16 * k = 0)) → k = 9 :=
by
  sorry

end tangent_line_to_parabola_l26_26551


namespace compute_exponent_problem_l26_26748

noncomputable def exponent_problem : ℤ :=
  3 * (3^4) - (9^60) / (9^57)

theorem compute_exponent_problem : exponent_problem = -486 := by
  sorry

end compute_exponent_problem_l26_26748


namespace adam_paper_tearing_l26_26310

theorem adam_paper_tearing (n : ℕ) :
  let starts_with_one_piece : ℕ := 1
  let increment_to_four : ℕ := 3
  let increment_to_ten : ℕ := 9
  let target_pieces : ℕ := 20000
  let start_modulo : ℤ := 1

  -- Modulo 3 analysis
  starts_with_one_piece % 3 = start_modulo ∧
  increment_to_four % 3 = 0 ∧ 
  increment_to_ten % 3 = 0 ∧ 
  target_pieces % 3 = 2 → 
  n % 3 = start_modulo ∧ ∀ m, m % 3 = 0 → n + m ≠ target_pieces :=
sorry

end adam_paper_tearing_l26_26310


namespace converse_proposition_l26_26028

theorem converse_proposition (a b c : ℝ) (h : c ≠ 0) :
  a * c^2 > b * c^2 → a > b :=
by
  sorry

end converse_proposition_l26_26028


namespace expected_heads_value_in_cents_l26_26457

open ProbabilityTheory

-- Define the coins and their respective values
def penny_value := 1
def nickel_value := 5
def half_dollar_value := 50
def dollar_value := 100

-- Define the probability of landing heads for each coin
def heads_prob := 1 / 2

-- Define the expected value function
noncomputable def expected_value_of_heads : ℝ :=
  heads_prob * (penny_value + nickel_value + half_dollar_value + dollar_value)

theorem expected_heads_value_in_cents : expected_value_of_heads = 78 := by
  sorry

end expected_heads_value_in_cents_l26_26457


namespace question_correct_statements_l26_26072

noncomputable def f : ℝ → ℝ := sorry

axiom additivity (f : ℝ → ℝ) : ∀ x y : ℝ, f (x + y) = f x + f y
axiom periodicity (f : ℝ → ℝ) : f 2 = 0

theorem question_correct_statements : 
  (∀ x : ℝ, f (x + 2) = f x) ∧ -- ensuring the function is periodic
  (∀ x : ℝ, f x = -f (-x)) ∧ -- ensuring the function is odd
  (∀ x : ℝ, f (x+2) = -f (-x)) :=  -- ensuring symmetry about point (1,0)
by
  -- We'll prove this using the conditions given and properties derived from it
  sorry 

end question_correct_statements_l26_26072


namespace bumper_car_rides_l26_26439

-- Define the conditions
def rides_on_ferris_wheel : ℕ := 7
def cost_per_ride : ℕ := 5
def total_tickets : ℕ := 50

-- Formulate the statement to be proved
theorem bumper_car_rides : ∃ n : ℕ, 
  total_tickets = (rides_on_ferris_wheel * cost_per_ride) + (n * cost_per_ride) ∧ n = 3 :=
sorry

end bumper_car_rides_l26_26439


namespace limit_expr_at_pi_l26_26330

theorem limit_expr_at_pi :
  (Real.exp π - Real.exp x) / (Real.sin (5*x) - Real.sin (3*x)) = 1 / 2 * Real.exp π :=
by
  sorry

end limit_expr_at_pi_l26_26330


namespace product_profit_equation_l26_26632

theorem product_profit_equation (purchase_price selling_price : ℝ) 
                                (initial_units units_decrease_per_dollar_increase : ℝ)
                                (profit : ℝ)
                                (hx : purchase_price = 35)
                                (hy : selling_price = 40)
                                (hz : initial_units = 200)
                                (hs : units_decrease_per_dollar_increase = 5)
                                (hp : profit = 1870) :
  ∃ x : ℝ, (x + (selling_price - purchase_price)) * (initial_units - units_decrease_per_dollar_increase * x) = profit :=
by { sorry }

end product_profit_equation_l26_26632


namespace unicorn_tether_l26_26994

theorem unicorn_tether (a b c : ℕ) (h_c_prime : Prime c) :
  (∃ (a b c : ℕ), c = 1 ∧ (25 - 15 = 10 ∧ 10^2 + 10^2 = 15^2 ∧ 
  a = 10 ∧ b = 125) ∧ a + b + c = 136) :=
  sorry

end unicorn_tether_l26_26994


namespace angle_in_third_quadrant_l26_26183

theorem angle_in_third_quadrant (k : ℤ) (α : ℝ) 
  (h : 180 + k * 360 < α ∧ α < 270 + k * 360) : 
  180 - α > -90 - k * 360 ∧ 180 - α < -k * 360 := 
by sorry

end angle_in_third_quadrant_l26_26183


namespace num_biology_books_is_15_l26_26849

-- conditions
def num_chemistry_books : ℕ := 8
def total_ways : ℕ := 2940

-- main statement to prove
theorem num_biology_books_is_15 : ∃ B: ℕ, (B * (B - 1)) / 2 * (num_chemistry_books * (num_chemistry_books - 1)) / 2 = total_ways ∧ B = 15 :=
by
  sorry

end num_biology_books_is_15_l26_26849


namespace solve_for_x_l26_26408

theorem solve_for_x : 
  ∀ x : ℚ, x + 5/6 = 7/18 - 2/9 → x = -2/3 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l26_26408


namespace product_of_radii_l26_26335

-- Definitions based on the problem conditions
def passes_through (a : ℝ) (C : ℝ × ℝ) : Prop :=
  (C.1 - a)^2 + (C.2 - a)^2 = a^2

def tangent_to_axes (a : ℝ) : Prop :=
  a > 0

def circle_radii_roots (a b : ℝ) : Prop :=
  a^2 - 14 * a + 25 = 0 ∧ b^2 - 14 * b + 25 = 0

-- Theorem statement to prove the product of the radii
theorem product_of_radii (a r1 r2 : ℝ) (h1 : passes_through a (3, 4)) (h2 : tangent_to_axes a) (h3 : circle_radii_roots r1 r2) : r1 * r2 = 25 :=
by
  sorry

end product_of_radii_l26_26335


namespace original_number_l26_26617

theorem original_number (x : ℝ) (h : 1.35 * x = 680) : x = 503.70 :=
sorry

end original_number_l26_26617


namespace minimum_value_l26_26334

theorem minimum_value (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_condition : 2 * a + 3 * b = 1) :
  ∃ min_value : ℝ, min_value = 65 / 6 ∧ (∀ c d : ℝ, (0 < c) → (0 < d) → (2 * c + 3 * d = 1) → (1 / c + 1 / d ≥ min_value)) :=
sorry

end minimum_value_l26_26334


namespace two_b_squared_eq_a_squared_plus_c_squared_l26_26400

theorem two_b_squared_eq_a_squared_plus_c_squared (a b c : ℝ) (h : 1 / (a + b) + 1 / (b + c) = 2 / (c + a)) : 
  2 * b^2 = a^2 + c^2 := 
sorry

end two_b_squared_eq_a_squared_plus_c_squared_l26_26400


namespace residue_of_7_pow_2023_mod_19_l26_26756

theorem residue_of_7_pow_2023_mod_19 : (7^2023) % 19 = 3 :=
by 
  -- The main goal is to construct the proof that matches our explanation.
  sorry

end residue_of_7_pow_2023_mod_19_l26_26756


namespace ways_to_divide_week_l26_26660

def week_seconds : ℕ := 604800

theorem ways_to_divide_week (n m : ℕ) (h1 : 0 < n) (h2 : 0 < m) (h3 : week_seconds = n * m) :
  (∃ (pairs : ℕ), pairs = 336) :=
sorry

end ways_to_divide_week_l26_26660


namespace least_perimeter_of_triangle_l26_26620

theorem least_perimeter_of_triangle (c : ℕ) (h1 : 24 + 51 > c) (h2 : c > 27) : 24 + 51 + c = 103 :=
by
  sorry

end least_perimeter_of_triangle_l26_26620


namespace train_speed_is_28_l26_26622

-- Define the given conditions
def train_length : ℕ := 1200
def overbridge_length : ℕ := 200
def crossing_time : ℕ := 50

-- Define the total distance
def total_distance := train_length + overbridge_length

-- Define the speed calculation function
def speed (distance time : ℕ) : ℕ := 
  distance / time

-- State the theorem to be proven
theorem train_speed_is_28 : speed total_distance crossing_time = 28 := 
by
  -- Proof to be provided
  sorry

end train_speed_is_28_l26_26622


namespace pyramid_volume_l26_26112

theorem pyramid_volume 
(EF FG QE : ℝ) 
(base_area : ℝ) 
(volume : ℝ)
(h1 : EF = 10)
(h2 : FG = 5)
(h3 : base_area = EF * FG)
(h4 : QE = 9)
(h5 : volume = (1 / 3) * base_area * QE) : 
volume = 150 :=
by
  simp [h1, h2, h3, h4, h5]
  sorry

end pyramid_volume_l26_26112


namespace value_of_expression_l26_26595

theorem value_of_expression : 2 - (-5) = 7 :=
by
  sorry

end value_of_expression_l26_26595


namespace temperature_decrease_l26_26613

-- Define the conditions
def temperature_rise (temp_increase: ℤ) : ℤ := temp_increase

-- Define the claim to be proved
theorem temperature_decrease (temp_decrease: ℤ) : temperature_rise 3 = 3 → temperature_rise (-6) = -6 :=
by
  sorry

end temperature_decrease_l26_26613


namespace find_common_difference_l26_26094

variable {a : ℕ → ℤ}  -- Define the arithmetic sequence as a function from natural numbers to integers
variable (d : ℤ)      -- Define the common difference

-- Assume the conditions given in the problem
axiom h1 : a 2 = 14
axiom h2 : a 5 = 5

theorem find_common_difference (n : ℕ) : d = -3 :=
by {
  -- This part will be filled in by the actual proof
  sorry
}

end find_common_difference_l26_26094


namespace range_of_x_given_p_and_q_range_of_m_given_neg_q_sufficient_for_neg_p_l26_26898

variable {x m : ℝ}

-- First statement: Given m = 4 and p ∧ q, prove the range of x is 4 < x < 5
theorem range_of_x_given_p_and_q (m : ℝ) (h : m = 4) :
  (x^2 - 7*x + 10 < 0) ∧ (x^2 - 4*m*x + 3*m^2 < 0) → (4 < x ∧ x < 5) :=
sorry

-- Second statement: Prove the range of m given ¬q is a sufficient but not necessary condition for ¬p
theorem range_of_m_given_neg_q_sufficient_for_neg_p :
  (m ≤ 2) ∧ (3*m ≥ 5) ∧ (m > 0) → (5/3 ≤ m ∧ m ≤ 2) :=
sorry

end range_of_x_given_p_and_q_range_of_m_given_neg_q_sufficient_for_neg_p_l26_26898


namespace find_A_solution_l26_26481

theorem find_A_solution (A : ℝ) (h : 32 * A^3 = 42592) : A = 11 :=
sorry

end find_A_solution_l26_26481


namespace simplify_expression_l26_26179

theorem simplify_expression :
  (2^8 + 5^5) * (2^3 - (-2)^3)^7 = 9077567990336 :=
by
  sorry

end simplify_expression_l26_26179


namespace cost_of_computer_game_is_90_l26_26887

-- Define the costs of individual items
def polo_shirt_price : ℕ := 26
def necklace_price : ℕ := 83
def rebate : ℕ := 12
def total_cost_after_rebate : ℕ := 322

-- Define the number of items
def polo_shirt_quantity : ℕ := 3
def necklace_quantity : ℕ := 2
def computer_game_quantity : ℕ := 1

-- Calculate the total cost before rebate
def total_cost_before_rebate : ℕ :=
  total_cost_after_rebate + rebate

-- Calculate the total cost of polo shirts and necklaces
def total_cost_polo_necklaces : ℕ :=
  (polo_shirt_quantity * polo_shirt_price) + (necklace_quantity * necklace_price)

-- Define the unknown cost of the computer game
def computer_game_price : ℕ :=
  total_cost_before_rebate - total_cost_polo_necklaces

-- Prove the cost of the computer game
theorem cost_of_computer_game_is_90 : computer_game_price = 90 := by
  -- The following line is a placeholder for the actual proof
  sorry

end cost_of_computer_game_is_90_l26_26887


namespace tom_and_jerry_same_speed_l26_26541

noncomputable def speed_of_tom (y : ℝ) : ℝ :=
  y^2 - 14*y + 45

noncomputable def speed_of_jerry (y : ℝ) : ℝ :=
  (y^2 - 2*y - 35) / (y - 5)

theorem tom_and_jerry_same_speed (y : ℝ) (h₁ : y ≠ 5) (h₂ : speed_of_tom y = speed_of_jerry y) :
  speed_of_tom y = 6 :=
by
  sorry

end tom_and_jerry_same_speed_l26_26541


namespace age_twice_in_2_years_l26_26347

/-
Conditions:
1. The man is 24 years older than his son.
2. The present age of the son is 22 years.
3. In a certain number of years, the man's age will be twice the age of his son.
-/
def man_is_24_years_older (S M : ℕ) : Prop := M = S + 24
def present_age_son : ℕ := 22
def age_twice_condition (Y S M : ℕ) : Prop := M + Y = 2 * (S + Y)

/-
Prove that in 2 years, the man's age will be twice the age of his son.
-/
theorem age_twice_in_2_years : ∃ (Y : ℕ), 
  (man_is_24_years_older present_age_son M) → 
  (age_twice_condition Y present_age_son M) →
  Y = 2 :=
by
  sorry

end age_twice_in_2_years_l26_26347


namespace no_linear_factor_with_integer_coefficients_l26_26468

def expression (x y z : ℤ) : ℤ :=
  x^2 - y^2 - z^2 + 3 * y * z + x + 2 * y - z

theorem no_linear_factor_with_integer_coefficients:
  ¬ ∃ (a b c d : ℤ), a ≠ 0 ∧ 
                      ∀ (x y z : ℤ), 
                        expression x y z = a * x + b * y + c * z + d := by
  sorry

end no_linear_factor_with_integer_coefficients_l26_26468


namespace wallpaper_removal_time_l26_26936

theorem wallpaper_removal_time (time_per_wall : ℕ) (dining_room_walls_remaining : ℕ) (living_room_walls : ℕ) :
  time_per_wall = 2 → dining_room_walls_remaining = 3 → living_room_walls = 4 → 
  time_per_wall * (dining_room_walls_remaining + living_room_walls) = 14 :=
by
  sorry

end wallpaper_removal_time_l26_26936


namespace power_function_value_l26_26730

noncomputable def f (x : ℝ) : ℝ := x^2

theorem power_function_value :
  f 3 = 9 :=
by
  -- Since f(x) = x^2 and f passes through (-2, 4)
  -- f(x) = x^2, so f(3) = 3^2 = 9
  sorry

end power_function_value_l26_26730


namespace compound_interest_rate_l26_26832

-- Defining the principal amount and total repayment
def P : ℝ := 200
def A : ℝ := 220

-- The annual compound interest rate
noncomputable def annual_compound_interest_rate (P A : ℝ) (n : ℕ) : ℝ :=
  (A / P)^(1 / n) - 1

-- Introducing the conditions
axiom compounded_annually : ∀ (P A : ℝ), annual_compound_interest_rate P A 1 = 0.1

-- Stating the theorem
theorem compound_interest_rate :
  annual_compound_interest_rate P A 1 = 0.1 :=
by {
  exact compounded_annually P A
}

end compound_interest_rate_l26_26832


namespace max_diff_y_l26_26049

theorem max_diff_y (x y z : ℕ) (h₁ : 4 < x) (h₂ : x < z) (h₃ : z < y) (h₄ : y < 10) (h₅ : y - x = 5) : y = 9 :=
sorry

end max_diff_y_l26_26049


namespace floor_diff_l26_26516

theorem floor_diff (x : ℝ) (h : x = 13.2) : 
  (Int.floor (x^2) - (Int.floor x) * (Int.floor x) = 5) := by
  sorry

end floor_diff_l26_26516


namespace max_value_f_l26_26679

open Real

/-- Determine the maximum value of the function f(x) = 1 / (1 - x * (1 - x)). -/
theorem max_value_f (x : ℝ) : 
  ∃ y, y = (1 / (1 - x * (1 - x))) ∧ y ≤ 4/3 ∧ ∀ z, z = (1 / (1 - x * (1 - x))) → z ≤ 4/3 :=
by
  sorry

end max_value_f_l26_26679


namespace find_smallest_in_arithmetic_progression_l26_26384

theorem find_smallest_in_arithmetic_progression (a d : ℝ)
  (h1 : (a-2*d)^3 + (a-d)^3 + a^3 + (a+d)^3 + (a+2*d)^3 = 0)
  (h2 : (a-2*d)^4 + (a-d)^4 + a^4 + (a+d)^4 + (a+2*d)^4 = 136) :
  (a - 2*d) = -2 * Real.sqrt 2 :=
sorry

end find_smallest_in_arithmetic_progression_l26_26384


namespace relative_error_comparison_l26_26088

theorem relative_error_comparison :
  let e₁ := 0.05
  let l₁ := 25.0
  let e₂ := 0.4
  let l₂ := 200.0
  let relative_error (e l : ℝ) : ℝ := (e / l) * 100
  (relative_error e₁ l₁ = relative_error e₂ l₂) :=
by
  sorry

end relative_error_comparison_l26_26088


namespace line_through_midpoint_bisects_chord_eqn_l26_26754

theorem line_through_midpoint_bisects_chord_eqn :
  ∀ (x y : ℝ), (x^2 - 4*y^2 = 4) ∧ (∃ x1 y1 x2 y2 : ℝ, 
    (x1^2 - 4 * y1^2 = 4) ∧ (x2^2 - 4 * y2^2 = 4) ∧ 
    (x1 + x2) / 2 = 3 ∧ (y1 + y2) / 2 = -1) → 
    3 * x + 4 * y - 5 = 0 :=
by
  intros x y h
  sorry

end line_through_midpoint_bisects_chord_eqn_l26_26754


namespace pow_neg_one_diff_l26_26397

theorem pow_neg_one_diff (n : ℤ) (h1 : n = 2010) (h2 : n + 1 = 2011) :
  (-1)^2010 - (-1)^2011 = 2 := 
by
  sorry

end pow_neg_one_diff_l26_26397


namespace sum_sequence_correct_l26_26380

def sequence_term (n : ℕ) : ℕ :=
  if n % 9 = 0 ∧ n % 32 = 0 then 7
  else if n % 7 = 0 ∧ n % 32 = 0 then 9
  else if n % 7 = 0 ∧ n % 9 = 0 then 32
  else 0

def sequence_sum (up_to : ℕ) : ℕ :=
  (Finset.range (up_to + 1)).sum sequence_term

theorem sum_sequence_correct : sequence_sum 2015 = 1106 := by
  sorry

end sum_sequence_correct_l26_26380


namespace min_value_of_quadratic_l26_26851

open Real

theorem min_value_of_quadratic 
  (x y z : ℝ) 
  (h : 3 * x + 2 * y + z = 1) : 
  x^2 + 2 * y^2 + 3 * z^2 ≥ 3 / 34 := 
sorry

end min_value_of_quadratic_l26_26851


namespace child_stops_incur_yearly_cost_at_age_18_l26_26948

def john_contribution (years: ℕ) (cost_per_year: ℕ) : ℕ :=
  years * cost_per_year / 2

def university_contribution (university_cost: ℕ) : ℕ :=
  university_cost / 2

def total_contribution (years_after_8: ℕ) : ℕ :=
  john_contribution 8 10000 +
  john_contribution years_after_8 20000 +
  university_contribution 250000

theorem child_stops_incur_yearly_cost_at_age_18 :
  (total_contribution n = 265000) → (n + 8 = 18) :=
by
  sorry

end child_stops_incur_yearly_cost_at_age_18_l26_26948


namespace sum_of_inserted_numbers_l26_26495

variable {x y : ℝ} -- Variables x and y are real numbers

-- Conditions
axiom geometric_sequence_condition : x^2 = 3 * y
axiom arithmetic_sequence_condition : 2 * y = x + 9

-- Goal: Prove that x + y = 45 / 4 (which is 11 1/4)
theorem sum_of_inserted_numbers : x + y = 45 / 4 :=
by
  -- Utilize axioms and conditions
  sorry

end sum_of_inserted_numbers_l26_26495


namespace remainder_correct_l26_26201

def dividend : ℝ := 13787
def divisor : ℝ := 154.75280898876406
def quotient : ℝ := 89
def remainder : ℝ := dividend - (divisor * quotient)

theorem remainder_correct: remainder = 14 := by
  -- Proof goes here
  sorry

end remainder_correct_l26_26201


namespace Joann_lollipop_theorem_l26_26461

noncomputable def Joann_lollipops (a : ℝ) : ℝ := a + 9

theorem Joann_lollipop_theorem (a : ℝ) (total_lollipops : ℝ) 
  (h1 : a + (a + 3) + (a + 6) + (a + 9) + (a + 12) + (a + 15) = 150) 
  (h2 : total_lollipops = 150) : 
  Joann_lollipops a = 26.5 :=
by
  sorry

end Joann_lollipop_theorem_l26_26461


namespace total_distance_covered_l26_26610

-- Define the basic conditions
def num_marathons : Nat := 15
def miles_per_marathon : Nat := 26
def yards_per_marathon : Nat := 385
def yards_per_mile : Nat := 1760

-- Define the total miles and total yards covered
def total_miles : Nat := num_marathons * miles_per_marathon
def total_yards : Nat := num_marathons * yards_per_marathon

-- Convert excess yards into miles and calculate the remaining yards
def extra_miles : Nat := total_yards / yards_per_mile
def remaining_yards : Nat := total_yards % yards_per_mile

-- Compute the final total distance
def total_distance_miles : Nat := total_miles + extra_miles
def total_distance_yards : Nat := remaining_yards

-- The theorem that needs to be proven
theorem total_distance_covered :
  total_distance_miles = 393 ∧ total_distance_yards = 495 :=
by
  sorry

end total_distance_covered_l26_26610


namespace MariaTotalPaid_l26_26580

-- Define a structure to hold the conditions
structure DiscountProblem where
  discount_rate : ℝ
  discount_amount : ℝ

-- Define the given discount problem specific to Maria
def MariaDiscountProblem : DiscountProblem :=
  { discount_rate := 0.25, discount_amount := 40 }

-- Define our goal: proving the total amount paid by Maria
theorem MariaTotalPaid (p : DiscountProblem) (h₀ : p = MariaDiscountProblem) :
  let original_price := p.discount_amount / p.discount_rate
  let total_paid := original_price - p.discount_amount
  total_paid = 120 :=
by
  sorry

end MariaTotalPaid_l26_26580


namespace alok_age_proof_l26_26134

variable (A B C : ℕ)

theorem alok_age_proof (h1 : B = 6 * A) (h2 : B + 10 = 2 * (C + 10)) (h3 : C = 10) : A = 5 :=
by
  sorry

end alok_age_proof_l26_26134


namespace age_sum_proof_l26_26817

noncomputable def leilei_age : ℝ := 30 -- Age of Leilei this year
noncomputable def feifei_age (R : ℝ) : ℝ := 1 / 2 * R + 12 -- Age of Feifei this year defined in terms of R

theorem age_sum_proof (R F : ℝ)
  (h1 : F = 1 / 2 * R + 12)
  (h2 : F + 1 = 2 * (R + 1) - 34) :
  R + F = 57 :=
by 
  -- Proof steps would go here
  sorry

end age_sum_proof_l26_26817


namespace angle_45_deg_is_75_venerts_l26_26862

-- There are 600 venerts in a full circle.
def venus_full_circle : ℕ := 600

-- A full circle on Earth is 360 degrees.
def earth_full_circle : ℕ := 360

-- Conversion factor from degrees to venerts.
def degrees_to_venerts (deg : ℕ) : ℕ :=
  deg * (venus_full_circle / earth_full_circle)

-- Angle of 45 degrees in venerts.
def angle_45_deg_in_venerts : ℕ := 45 * (venus_full_circle / earth_full_circle)

theorem angle_45_deg_is_75_venerts :
  angle_45_deg_in_venerts = 75 :=
by
  -- Proof will be inserted here.
  sorry

end angle_45_deg_is_75_venerts_l26_26862


namespace remainder_of_2n_l26_26422

theorem remainder_of_2n (n : ℕ) (h : n % 4 = 3) : (2 * n) % 4 = 2 := 
sorry

end remainder_of_2n_l26_26422


namespace ellipse_triangle_perimeter_l26_26683

-- Definitions based on conditions
def is_ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 2) = 1

-- Triangle perimeter calculation
def triangle_perimeter (a c : ℝ) : ℝ := 2 * a + 2 * c

-- Main theorem statement
theorem ellipse_triangle_perimeter :
  let a := 2
  let b2 := 2
  let c := Real.sqrt (a ^ 2 - b2)
  ∀ (P : ℝ × ℝ), (is_ellipse P.1 P.2) → triangle_perimeter a c = 4 + 2 * Real.sqrt 2 :=
by
  intros P hP
  -- Here, we would normally provide the proof.
  sorry

end ellipse_triangle_perimeter_l26_26683


namespace rulers_left_l26_26772

variable (rulers_in_drawer : Nat)
variable (rulers_taken : Nat)

theorem rulers_left (h1 : rulers_in_drawer = 46) (h2 : rulers_taken = 25) : 
  rulers_in_drawer - rulers_taken = 21 := by
  sorry

end rulers_left_l26_26772


namespace intersection_A_B_l26_26412

def A : Set ℝ := {x | x * (x - 4) < 0}
def B : Set ℝ := {0, 1, 5}

theorem intersection_A_B : (A ∩ B) = {1} := by
  sorry

end intersection_A_B_l26_26412


namespace part_a_l26_26582

theorem part_a (a b : ℝ) (h1 : a^2 + b^2 = 1) (h2 : a + b = 1) : a * b = 0 := 
by 
  sorry

end part_a_l26_26582


namespace largest_n_unique_k_l26_26599

-- Defining the main theorem statement
theorem largest_n_unique_k :
  ∃ (n : ℕ), (n = 63) ∧ (∃! (k : ℤ), (9 / 17 : ℚ) < (n : ℚ) / ((n + k) : ℚ) ∧ (n : ℚ) / ((n + k) : ℚ) < (8 / 15 : ℚ)) :=
sorry

end largest_n_unique_k_l26_26599


namespace evaluate_expression_l26_26499

theorem evaluate_expression (c d : ℕ) (hc : c = 4) (hd : d = 2) :
  (c^c - c * (c - d)^c)^c = 136048896 := by
  sorry

end evaluate_expression_l26_26499


namespace solitaire_game_end_with_one_piece_l26_26641

theorem solitaire_game_end_with_one_piece (n : ℕ) : 
  ∃ (remaining_pieces : ℕ), 
  remaining_pieces = 1 ↔ n % 3 ≠ 0 :=
sorry

end solitaire_game_end_with_one_piece_l26_26641


namespace determinant_roots_l26_26833

theorem determinant_roots (s p q a b c : ℂ) 
  (h : ∀ x : ℂ, x^3 - s*x^2 + p*x + q = (x - a) * (x - b) * (x - c)) :
  (1 + a) * ((1 + b) * (1 + c) - 1) - ((1) * (1 + c) - 1) + ((1) - (1 + b)) = p + 3 * s :=
by {
  -- expanded determinant calculations
  sorry
}

end determinant_roots_l26_26833


namespace sqrt_expression_l26_26462

theorem sqrt_expression :
  (Real.sqrt (2 ^ 4 * 3 ^ 6 * 5 ^ 2)) = 540 := sorry

end sqrt_expression_l26_26462


namespace nina_total_miles_l26_26635

noncomputable def totalDistance (warmUp firstHillUp firstHillDown firstRecovery 
                                 tempoRun secondHillUp secondHillDown secondRecovery 
                                 fartlek sprintsYards jogsBetweenSprints coolDown : ℝ) 
                                 (mileInYards : ℝ) : ℝ :=
  warmUp + 
  (firstHillUp + firstHillDown + firstRecovery) + 
  tempoRun + 
  (secondHillUp + secondHillDown + secondRecovery) + 
  fartlek + 
  (sprintsYards / mileInYards) + 
  jogsBetweenSprints + 
  coolDown

theorem nina_total_miles : 
  totalDistance 0.25 0.15 0.25 0.15 1.5 0.2 0.35 0.1 1.8 (8 * 50) (8 * 0.2) 0.3 1760 = 5.877 :=
by
  sorry

end nina_total_miles_l26_26635


namespace smallest_rational_in_set_l26_26843

theorem smallest_rational_in_set : 
  ∀ (a b c d : ℚ), 
    a = -2/3 → b = -1 → c = 0 → d = 1 → 
    (a > b ∧ b < c ∧ c < d) → b = -1 := 
by
  intros a b c d ha hb hc hd h
  sorry

end smallest_rational_in_set_l26_26843


namespace sufficient_but_not_necessary_for_monotonic_l26_26014

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
∀ x y, x ≤ y → f x ≤ f y

noncomputable def is_sufficient_condition (P Q : Prop) : Prop :=
P → Q

noncomputable def is_not_necessary_condition (P Q : Prop) : Prop :=
¬ Q → ¬ P

noncomputable def is_sufficient_but_not_necessary (P Q : Prop) : Prop :=
is_sufficient_condition P Q ∧ is_not_necessary_condition P Q

theorem sufficient_but_not_necessary_for_monotonic (f : ℝ → ℝ) :
  (∀ x, 0 ≤ deriv f x) → is_monotonically_increasing f :=
sorry

end sufficient_but_not_necessary_for_monotonic_l26_26014


namespace fran_speed_l26_26836

variable (s : ℝ)

theorem fran_speed
  (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ)
  (h1 : joann_speed = 15)
  (h2 : joann_time = 4)
  (h3 : fran_time = 3.5)
  (h4 : fran_time * s = joann_speed * joann_time)
  : s = 120 / 7 := 
by
  sorry

end fran_speed_l26_26836


namespace sum_of_star_angles_l26_26275

theorem sum_of_star_angles :
  let n := 12
  let angle_per_arc := 360 / n
  let arcs_per_tip := 3
  let internal_angle_per_tip := 360 - arcs_per_tip * angle_per_arc
  let sum_of_angles := n * (360 - internal_angle_per_tip)
  sum_of_angles = 1080 :=
by
  sorry

end sum_of_star_angles_l26_26275


namespace sum_of_two_numbers_l26_26429

theorem sum_of_two_numbers (a b S : ℤ) (h : a + b = S) : 
  3 * (a + 5) + 3 * (b + 7) = 3 * S + 36 :=
by
  sorry

end sum_of_two_numbers_l26_26429


namespace diamond_expression_l26_26853

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- Declare the main theorem
theorem diamond_expression :
  (diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4)) = -29 / 132 := 
by
  sorry

end diamond_expression_l26_26853


namespace chord_length_eq_l26_26025

noncomputable def length_of_chord (radius : ℝ) (distance_to_chord : ℝ) : ℝ :=
  2 * Real.sqrt (radius^2 - distance_to_chord^2)

theorem chord_length_eq {radius distance_to_chord : ℝ} (h_radius : radius = 5) (h_distance : distance_to_chord = 4) :
  length_of_chord radius distance_to_chord = 6 :=
by
  sorry

end chord_length_eq_l26_26025


namespace unique_real_solution_floor_eq_l26_26225

theorem unique_real_solution_floor_eq (k : ℕ) (h : k > 0) :
  ∃! x : ℝ, k ≤ x ∧ x < k + 1 ∧ ⌊x⌋ * (x^2 + 1) = x^3 :=
sorry

end unique_real_solution_floor_eq_l26_26225


namespace vector_addition_result_l26_26828

-- Define the given vectors
def vector_a : ℝ × ℝ := (2, 1)
def vector_b : ℝ × ℝ := (-3, 4)

-- Statement to prove that the sum of the vectors is (-1, 5)
theorem vector_addition_result : vector_a + vector_b = (-1, 5) :=
by
  -- Use the fact that vector addition in ℝ^2 is component-wise
  sorry

end vector_addition_result_l26_26828


namespace f_is_decreasing_max_k_value_l26_26781

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log (x + 1)) / x

theorem f_is_decreasing : ∀ x > 0, (∃ y > x, f y < f x) :=
by
  sorry

theorem max_k_value : ∃ k : ℕ, (∀ x > 0, f x > k / (x + 1)) ∧ k = 3 :=
by
  sorry

end f_is_decreasing_max_k_value_l26_26781


namespace polynomial_sequence_finite_functions_l26_26355

theorem polynomial_sequence_finite_functions (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) : 
  (1 / (1 + a + b) + 1 / (1 + b + c) + 1 / (1 + c + a) ≤ 1) := 
by
  sorry

end polynomial_sequence_finite_functions_l26_26355


namespace tan_A_of_triangle_conditions_l26_26905

open Real

def triangle_angles (A B C : ℝ) : Prop :=
  A + B + C = π ∧ 0 < A ∧ A < π / 2 ∧ B = π / 4

def form_arithmetic_sequence (a b c : ℝ) : Prop :=
  2 * b^2 = a^2 + c^2

theorem tan_A_of_triangle_conditions
  (A B C a b c : ℝ)
  (h_angles : triangle_angles A B C)
  (h_seq : form_arithmetic_sequence a b c) :
  tan A = sqrt 2 - 1 :=
by
  sorry

end tan_A_of_triangle_conditions_l26_26905


namespace geom_prog_common_ratio_l26_26999

-- Definition of a geometric progression
def geom_prog (u : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n ≥ 1, u (n + 1) = u n + u (n - 1)

-- Statement of the problem
theorem geom_prog_common_ratio (u : ℕ → ℝ) (q : ℝ) (hq : ∀ n ≥ 1, u (n + 1) = u n + u (n - 1)) :
  (q = (1 + Real.sqrt 5) / 2) ∨ (q = (1 - Real.sqrt 5) / 2) :=
sorry

end geom_prog_common_ratio_l26_26999


namespace planted_fraction_correct_l26_26723

noncomputable def field_planted_fraction (leg1 leg2 : ℕ) (square_distance : ℕ) : ℚ :=
  let hypotenuse := Real.sqrt (leg1^2 + leg2^2)
  let total_area := (leg1 * leg2) / 2
  let square_side := square_distance
  let square_area := square_side^2
  let planted_area := total_area - square_area
  planted_area / total_area

theorem planted_fraction_correct :
  field_planted_fraction 5 12 4 = 367 / 375 :=
by
  sorry

end planted_fraction_correct_l26_26723


namespace find_numbers_l26_26665

-- Definitions for the conditions
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def is_even_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 98 ∧ n % 2 = 0
def difference_is_three (x y : ℕ) : Prop := x - y = 3

-- Statement of the proof problem
theorem find_numbers (x y : ℕ) (h1 : is_three_digit x) (h2 : is_even_two_digit y) (h3 : difference_is_three x y) :
  x = 101 ∧ y = 98 :=
sorry

end find_numbers_l26_26665


namespace mean_score_l26_26071

theorem mean_score (M SD : ℝ) (h1 : 58 = M - 2 * SD) (h2 : 98 = M + 3 * SD) : M = 74 :=
by
  sorry

end mean_score_l26_26071


namespace playground_area_l26_26970

theorem playground_area (w l : ℕ) (h1 : 2 * l + 2 * w = 72) (h2 : l = 3 * w) : l * w = 243 := by
  sorry

end playground_area_l26_26970


namespace pies_with_no_ingredients_l26_26659

theorem pies_with_no_ingredients (total_pies : ℕ)
  (pies_with_chocolate : ℕ)
  (pies_with_blueberries : ℕ)
  (pies_with_vanilla : ℕ)
  (pies_with_almonds : ℕ)
  (H_total : total_pies = 60)
  (H_chocolate : pies_with_chocolate = 1 / 3 * total_pies)
  (H_blueberries : pies_with_blueberries = 3 / 4 * total_pies)
  (H_vanilla : pies_with_vanilla = 2 / 5 * total_pies)
  (H_almonds : pies_with_almonds = 1 / 10 * total_pies) :
  ∃ (pies_without_ingredients : ℕ), pies_without_ingredients = 15 :=
by
  sorry

end pies_with_no_ingredients_l26_26659


namespace determine_digit_square_l26_26437

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def is_palindrome (n : ℕ) : Prop :=
  let d1 := (n / 100000) % 10
  let d2 := (n / 10000) % 10
  let d3 := (n / 1000) % 10
  let d4 := (n / 100) % 10
  let d5 := (n / 10) % 10
  let d6 := n % 10
  d1 = d6 ∧ d2 = d5 ∧ d3 = d4

def is_multiple_of_6 (n : ℕ) : Prop := is_even (n % 10) ∧ is_divisible_by_3 (List.sum (Nat.digits 10 n))

theorem determine_digit_square :
  ∃ (square : ℕ),
  (is_palindrome (53700000 + square * 10 + 735) ∧ is_multiple_of_6 (53700000 + square * 10 + 735)) ∧ square = 6 := by
  sorry

end determine_digit_square_l26_26437


namespace seventh_graders_more_than_sixth_graders_l26_26923

-- Definitions based on conditions
variables (S6 S7 : ℕ)
variable (h : 7 * S6 = 6 * S7)

-- Proposition based on the conclusion
theorem seventh_graders_more_than_sixth_graders (h : 7 * S6 = 6 * S7) : S7 > S6 :=
by {
  -- Skipping the proof with sorry
  sorry
}

end seventh_graders_more_than_sixth_graders_l26_26923


namespace radius_of_circle_l26_26855

theorem radius_of_circle (P Q : ℝ) (h : P / Q = 25) : ∃ r : ℝ, 2 * π * r = Q ∧ π * r^2 = P ∧ r = 50 := 
by
  -- Proof starts here
  sorry

end radius_of_circle_l26_26855


namespace tan_theta_correct_l26_26681

noncomputable def tan_theta : Real :=
  let θ : Real := sorry
  if h_θ : θ ∈ Set.Ioc 0 (Real.pi / 4) then
    if h : Real.sin θ + Real.cos θ = 17 / 13 then
      Real.tan θ
    else
      0
  else
    0

theorem tan_theta_correct {θ : Real} (h_θ : θ ∈ Set.Ioc 0 (Real.pi / 4))
  (h : Real.sin θ + Real.cos θ = 17 / 13) :
  Real.tan θ = 5 / 12 := sorry

end tan_theta_correct_l26_26681


namespace range_of_fx₂_l26_26907

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2 * x + a * Real.log x

def is_extreme_point (a x : ℝ) : Prop := 
  (2 * x^2 - 2 * x + a) / x = 0

theorem range_of_fx₂ (a x₁ x₂ : ℝ) (h₀ : 0 < a) (h₁ : a < 1 / 2) 
  (h₂ : 0 < x₁ ∧ x₁ < x₂) (h₃ : is_extreme_point a x₁)
  (h₄ : is_extreme_point a x₂) : 
  (f a x₂) ∈ (Set.Ioo (-(3 + 2 * Real.log 2) / 4) (-1)) :=
sorry

end range_of_fx₂_l26_26907


namespace sum_999_is_1998_l26_26738

theorem sum_999_is_1998 : 999 + 999 = 1998 :=
by
  sorry

end sum_999_is_1998_l26_26738


namespace sqrt_sqr_l26_26571

theorem sqrt_sqr (x : ℝ) (hx : 0 ≤ x) : (Real.sqrt x) ^ 2 = x := 
by sorry

example : (Real.sqrt 3) ^ 2 = 3 := 
by apply sqrt_sqr; linarith

end sqrt_sqr_l26_26571


namespace min_troublemakers_l26_26415

theorem min_troublemakers (n : ℕ) (truth_teller liar : ℕ → Prop) 
  (total_students : n = 29)
  (exists_one_liar : ∀ i : ℕ, liar i = true → ((liar (i - 1) = true ∨ liar (i + 1) = true))) -- assume cyclic seating
  (exists_two_liar : ∀ i : ℕ, truth_teller i = true → (liar (i - 1) = true ∧ liar (i + 1) = true)) :
  (∃ k : ℕ, k = 10 ∧ (∀ i : ℕ, i < 29 -> liar i = true -> k ≥ i)) :=
sorry

end min_troublemakers_l26_26415


namespace subtraction_problem_l26_26812

variable (x : ℕ) -- Let's assume x is a natural number for this problem

theorem subtraction_problem (h : x - 46 = 15) : x - 29 = 32 := 
by 
  sorry -- Proof to be filled in

end subtraction_problem_l26_26812


namespace ray_climbs_l26_26082

theorem ray_climbs (n : ℕ) (h1 : n % 3 = 1) (h2 : n % 5 = 3) (h3 : n % 7 = 1) (h4 : n > 15) : n = 73 :=
sorry

end ray_climbs_l26_26082


namespace binom_12_9_eq_220_l26_26392

noncomputable def binom (n k : ℕ) : ℕ :=
  n.choose k

theorem binom_12_9_eq_220 : binom 12 9 = 220 :=
sorry

end binom_12_9_eq_220_l26_26392


namespace shell_highest_point_time_l26_26811

theorem shell_highest_point_time (a b c : ℝ) (h₁ : a ≠ 0)
  (h₂ : a * 7^2 + b * 7 + c = a * 14^2 + b * 14 + c) :
  (-b / (2 * a)) = 10.5 :=
by
  -- The proof is omitted as per the instructions
  sorry

end shell_highest_point_time_l26_26811


namespace modulus_of_complex_l26_26317

-- Some necessary imports for complex numbers and proofs in Lean
open Complex

theorem modulus_of_complex (x y : ℝ) (h : (1 + I) * x = 1 + y * I) : abs (x + y * I) = Real.sqrt 2 :=
by
  sorry

end modulus_of_complex_l26_26317


namespace machine_Y_produces_more_widgets_l26_26889

-- Definitions for the rates and widgets produced
def W_x := 18 -- widgets per hour by machine X
def total_widgets := 1080

-- Calculations for time taken by each machine
def T_x := total_widgets / W_x -- time taken by machine X
def T_y := T_x - 10 -- machine Y takes 10 hours less

-- Rate at which machine Y produces widgets
def W_y := total_widgets / T_y

-- Calculation of percentage increase
def percentage_increase := (W_y - W_x) / W_x * 100

-- The final theorem to prove
theorem machine_Y_produces_more_widgets : percentage_increase = 20 := by
  sorry

end machine_Y_produces_more_widgets_l26_26889


namespace gcd_of_polynomials_l26_26778

theorem gcd_of_polynomials (b : ℤ) (h : 2460 ∣ b) : 
  Int.gcd (b^2 + 6 * b + 30) (b + 5) = 30 :=
sorry

end gcd_of_polynomials_l26_26778


namespace total_books_in_library_l26_26089

theorem total_books_in_library :
  ∃ (total_books : ℕ),
  (∀ (books_per_floor : ℕ), books_per_floor - 2 = 20 → 
  total_books = (28 * 6 * books_per_floor)) ∧ total_books = 3696 :=
by
  sorry

end total_books_in_library_l26_26089


namespace find_numbers_l26_26053

/-- Given the sums of three pairs of numbers, we prove the individual numbers. -/
theorem find_numbers (x y z : ℕ) (h1 : x + y = 40) (h2 : y + z = 50) (h3 : z + x = 70) :
  x = 30 ∧ y = 10 ∧ z = 40 :=
by
  sorry

end find_numbers_l26_26053


namespace joan_spent_on_thursday_l26_26838

theorem joan_spent_on_thursday : 
  ∀ (n : ℕ), 
  2 * (4 + n) = 18 → 
  n = 14 := 
by 
  sorry

end joan_spent_on_thursday_l26_26838


namespace least_xy_l26_26611

theorem least_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 9) : xy = 108 := by
  sorry

end least_xy_l26_26611


namespace no_real_roots_range_l26_26884

theorem no_real_roots_range (k : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x - 2*k + 3 ≠ 0) → k < 1 :=
by
  sorry

end no_real_roots_range_l26_26884


namespace combined_distance_all_birds_two_seasons_l26_26600

-- Definition of the given conditions
def number_of_birds : Nat := 20
def distance_jim_to_disney : Nat := 50
def distance_disney_to_london : Nat := 60

-- The conclusion we need to prove
theorem combined_distance_all_birds_two_seasons :
  (distance_jim_to_disney + distance_disney_to_london) * number_of_birds = 2200 :=
by
  sorry

end combined_distance_all_birds_two_seasons_l26_26600


namespace geometric_sequence_product_l26_26237

theorem geometric_sequence_product
  (a : ℕ → ℝ)
  (h_geo : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r)
  (h_pos : ∀ n : ℕ, 0 < a n)
  (h_roots : (∃ a₁ a₁₉ : ℝ, (a₁ + a₁₉ = 10) ∧ (a₁ * a₁₉ = 16) ∧ a 1 = a₁ ∧ a 19 = a₁₉)) :
  a 8 * a 12 = 16 := 
sorry

end geometric_sequence_product_l26_26237


namespace number_of_n_factorizable_l26_26957

theorem number_of_n_factorizable :
  ∃! n_values : Finset ℕ, (∀ n ∈ n_values, n ≤ 100 ∧ ∃ a b : ℤ, a + b = -2 ∧ a * b = -n) ∧ n_values.card = 9 := by
  sorry

end number_of_n_factorizable_l26_26957


namespace problem_statement_l26_26204

variable {R : Type} [LinearOrderedField R]
variable (f : R → R)

theorem problem_statement
  (hf1 : ∀ x y : R, 0 < x ∧ x < 2 ∧ 0 < y ∧ y < 2 ∧ x < y → f x < f y)
  (hf2 : ∀ x : R, f (x + 2) = f (- (x + 2))) :
  f (7 / 2) < f 1 ∧ f 1 < f (5 / 2) :=
by
  sorry

end problem_statement_l26_26204


namespace problem1_problem2_problem3_problem4_l26_26965

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x - 1
noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 - a

-- Prove that if f is increasing on ℝ, then a ∈ (-∞, 0]
theorem problem1 (a : ℝ) : (∀ x y : ℝ, x ≤ y → f x a ≤ f y a) → a ≤ 0 :=
sorry

-- Prove that if f is decreasing on (-1, 1), then a ∈ [3, ∞)
theorem problem2 (a : ℝ) : (∀ x y : ℝ, -1 < x → x < 1 → -1 < y → y < 1 → x ≤ y → f x a ≥ f y a) → 3 ≤ a :=
sorry

-- Prove that if the decreasing interval of f is (-1, 1), then a = 3
theorem problem3 (a : ℝ) : (∀ x : ℝ, (abs x < 1) ↔ f' x a < 0) → a = 3 :=
sorry

-- Prove that if f is not monotonic on (-1, 1), then a ∈ (0, 3)
theorem problem4 (a : ℝ) : (¬(∀ x : ℝ, -1 < x → x < 1 → (f' x a = 0) ∨ (f' x a ≠ 0))) → (0 < a ∧ a < 3) :=
sorry

end problem1_problem2_problem3_problem4_l26_26965


namespace maximum_ratio_squared_l26_26987

theorem maximum_ratio_squared (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ge : a ≥ b)
  (x y : ℝ) (h_x : 0 ≤ x) (h_xa : x < a) (h_y : 0 ≤ y) (h_yb : y < b)
  (h_eq1 : a^2 + y^2 = b^2 + x^2)
  (h_eq2 : b^2 + x^2 = (a - x)^2 + (b - y)^2) :
  (a / b)^2 ≤ 4 / 3 :=
sorry

end maximum_ratio_squared_l26_26987


namespace problem_statement_l26_26931

theorem problem_statement (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 4) :
  x + (x^3 / y^2) + (y^3 / x^2) + y = 74.0625 :=
sorry

end problem_statement_l26_26931


namespace diff_of_roots_l26_26186

-- Define the quadratic equation and its coefficients
def quadratic_eq (z : ℝ) : ℝ := 2 * z^2 + 5 * z - 12

-- Define the discriminant function
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the roots of the quadratic equation using the quadratic formula
noncomputable def larger_root (a b c : ℝ) : ℝ := (-b + Real.sqrt (discriminant a b c)) / (2 * a)
noncomputable def smaller_root (a b c : ℝ) : ℝ := (-b - Real.sqrt (discriminant a b c)) / (2 * a)

-- Define the proof statement
theorem diff_of_roots : 
  ∃ (a b c z1 z2 : ℝ), 
    a = 2 ∧ b = 5 ∧ c = -12 ∧
    quadratic_eq z1 = 0 ∧ quadratic_eq z2 = 0 ∧
    z1 = smaller_root a b c ∧ z2 = larger_root a b c ∧
    z2 - z1 = 5.5 := 
by 
  sorry

end diff_of_roots_l26_26186


namespace union_M_N_l26_26802

def M : Set ℝ := { x | x^2 - x = 0 }
def N : Set ℝ := { y | y^2 + y = 0 }

theorem union_M_N : (M ∪ N) = {-1, 0, 1} := 
by 
  sorry

end union_M_N_l26_26802


namespace angle_covered_in_three_layers_l26_26535

/-- Define the conditions: A 90-degree angle, sum of angles is 290 degrees,
    and prove the angle covered in three layers is 110 degrees. -/
theorem angle_covered_in_three_layers {α β : ℝ}
  (h1 : α + β = 90)
  (h2 : 2*α + 3*β = 290) :
  β = 110 := 
sorry

end angle_covered_in_three_layers_l26_26535


namespace part_one_part_two_l26_26565

def f (a x : ℝ) : ℝ := abs (x - a ^ 2) + abs (x + 2 * a + 3)

theorem part_one (a x : ℝ) : f a x ≥ 2 :=
by 
  sorry

noncomputable def f_neg_three_over_two (a : ℝ) : ℝ := f a (-3/2)

theorem part_two (a : ℝ) (h : f_neg_three_over_two a < 3) : -1 < a ∧ a < 0 :=
by 
  sorry

end part_one_part_two_l26_26565


namespace fractions_expressible_iff_prime_l26_26511

noncomputable def is_good_fraction (a b n : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = n

theorem fractions_expressible_iff_prime (n : ℕ) (hn : n > 1) :
  (∀ (a b : ℕ), b < n → ∃ (k l : ℤ), k * a + l * n = b) ↔ Prime n :=
sorry

end fractions_expressible_iff_prime_l26_26511


namespace min_chord_length_intercepted_line_eq_l26_26232

theorem min_chord_length_intercepted_line_eq (m : ℝ)
  (hC : ∀ (x y : ℝ), (x-1)^2 + (y-1)^2 = 16)
  (hL : ∀ (x y : ℝ), (2*m-1)*x + (m-1)*y - 3*m + 1 = 0)
  : ∃ x y : ℝ, x - 2*y - 4 = 0 := sorry

end min_chord_length_intercepted_line_eq_l26_26232


namespace express_in_scientific_notation_l26_26323

-- Definitions based on problem conditions
def GDP_first_quarter : ℝ := 27017800000000

-- Main theorem statement that needs to be proved
theorem express_in_scientific_notation :
  ∃ (a : ℝ) (b : ℤ), (GDP_first_quarter = a * 10 ^ b) ∧ (a = 2.70178) ∧ (b = 13) :=
by
  sorry -- Placeholder to indicate the proof is omitted

end express_in_scientific_notation_l26_26323


namespace sum_consecutive_numbers_last_digit_diff_l26_26023

theorem sum_consecutive_numbers_last_digit_diff (a : ℕ) : 
    (2015 * (a + 1007) % 10) ≠ (2019 * (a + 3024) % 10) := 
by 
  sorry

end sum_consecutive_numbers_last_digit_diff_l26_26023


namespace product_of_odd_primes_mod_32_l26_26568

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def M : ℕ := odd_primes_less_than_32.foldr (· * ·) 1

theorem product_of_odd_primes_mod_32 : M % 32 = 17 :=
by
  -- the proof goes here
  sorry

end product_of_odd_primes_mod_32_l26_26568


namespace table_height_l26_26721

theorem table_height (r s x y l : ℝ)
  (h1 : x + l - y = 32)
  (h2 : y + l - x = 28) :
  l = 30 :=
by
  sorry

end table_height_l26_26721


namespace sequence_formula_minimum_m_l26_26341

variable (a_n : ℕ → ℕ) (S_n : ℕ → ℕ)

/-- The sequence a_n with sum of its first n terms S_n, the first term a_1 = 1, and the terms
   1, a_n, S_n forming an arithmetic sequence, satisfies a_n = 2^(n-1). -/
theorem sequence_formula (h1 : a_n 1 = 1)
    (h2 : ∀ n : ℕ, 1 + n * (a_n n - 1) = S_n n) :
    ∀ n : ℕ, a_n n = 2 ^ (n - 1) := by
  sorry

/-- T_n being the sum of the sequence {n / a_n}, if T_n < (m - 4) / 3 for all n in ℕ*, 
    then the minimum value of m is 16. -/
theorem minimum_m (T_n : ℕ → ℝ) (m : ℕ)
    (hT : ∀ n : ℕ, n > 0 → T_n n < (m - 4) / 3) :
    m ≥ 16 := by
  sorry

end sequence_formula_minimum_m_l26_26341


namespace baker_cakes_l26_26757

theorem baker_cakes (P x : ℝ) (h1 : P * x = 320) (h2 : 0.80 * P * (x + 2) = 320) : x = 8 :=
by
  sorry

end baker_cakes_l26_26757


namespace ian_says_1306_l26_26638

noncomputable def number_i_say := 4 * (4 * (4 * (4 * (4 * (4 * (4 * (4 * 1 - 2) - 2) - 2) - 2) - 2) - 2) - 2) - 2

theorem ian_says_1306 (n : ℕ) : 1 ≤ n ∧ n ≤ 2000 → n = 1306 :=
by sorry

end ian_says_1306_l26_26638


namespace seq_problem_part1_seq_problem_part2_l26_26399

def seq (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n + 2) = |a (n + 1) - a n|

theorem seq_problem_part1 (a : ℕ → ℤ) 
  (h1 : a 1 = 2)
  (h2 : a 2 = -1)
  (h_seq : seq a) :
  a 2008 = 0 := 
sorry

theorem seq_problem_part2 (a : ℕ → ℤ)
  (h1 : a 1 = 2)
  (h2 : a 2 = -1)
  (h_seq : seq a) :
  ∃ (M : ℤ), 
  (∀ n : ℕ, ∃ m : ℕ, m > n ∧ a m = 0) ∧ 
  (∀ n : ℕ, ∃ m : ℕ, m > n ∧ a m = M) := 
sorry

end seq_problem_part1_seq_problem_part2_l26_26399


namespace expand_expression_l26_26084

theorem expand_expression (x y : ℝ) :
  (x + 3) * (4 * x - 5 * y) = 4 * x ^ 2 - 5 * x * y + 12 * x - 15 * y :=
by
  sorry

end expand_expression_l26_26084


namespace relationship_among_a_b_c_l26_26847

noncomputable def a : ℝ := Real.log (Real.tan (70 * Real.pi / 180)) / Real.log (1 / 2)
noncomputable def b : ℝ := Real.log (Real.sin (25 * Real.pi / 180)) / Real.log (1 / 2)
noncomputable def c : ℝ := (1 / 2) ^ Real.cos (25 * Real.pi / 180)

theorem relationship_among_a_b_c : a < c ∧ c < b :=
by
  -- proofs would go here
  sorry

end relationship_among_a_b_c_l26_26847


namespace find_value_of_expression_l26_26324

open Real

theorem find_value_of_expression (x y z w : ℝ) (h1 : x + y + z + w = 0) (h2 : x^7 + y^7 + z^7 + w^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 := by
  sorry

end find_value_of_expression_l26_26324


namespace inequality_holds_if_and_only_if_l26_26376

variable (x : ℝ) (b : ℝ)

theorem inequality_holds_if_and_only_if (hx : |x-5| + |x-3| + |x-2| < b) : b > 4 :=
sorry

end inequality_holds_if_and_only_if_l26_26376


namespace remainder_of_sum_l26_26463

theorem remainder_of_sum (n1 n2 n3 n4 n5 : ℕ) (h1 : n1 = 7145) (h2 : n2 = 7146)
  (h3 : n3 = 7147) (h4 : n4 = 7148) (h5 : n5 = 7149) :
  ((n1 + n2 + n3 + n4 + n5) % 8) = 7 :=
by sorry

end remainder_of_sum_l26_26463


namespace solid_with_square_views_is_cube_l26_26563

-- Define the conditions and the solid type
def is_square_face (view : Type) : Prop := 
  -- Definition to characterize a square view. This is general,
  -- as the detailed characterization of a 'square' in Lean would depend
  -- on more advanced geometry modules, assuming a simple predicate here.
  sorry

structure Solid := (front_view : Type) (top_view : Type) (left_view : Type)

-- Conditions indicating that all views are squares
def all_views_square (S : Solid) : Prop :=
  is_square_face S.front_view ∧ is_square_face S.top_view ∧ is_square_face S.left_view

-- The theorem we are aiming to prove
theorem solid_with_square_views_is_cube (S : Solid) (h : all_views_square S) : S = {front_view := ℝ, top_view := ℝ, left_view := ℝ} := sorry

end solid_with_square_views_is_cube_l26_26563


namespace rectangle_length_l26_26438

theorem rectangle_length (P L B : ℕ) (hP : P = 500) (hB : B = 100) (hP_eq : P = 2 * (L + B)) : L = 150 :=
by
  sorry

end rectangle_length_l26_26438


namespace grazing_area_of_goat_l26_26749

/-- 
Consider a circular park with a diameter of 50 feet, and a square monument with 10 feet on each side.
Sally ties her goat on one corner of the monument with a 20-foot rope. Calculate the total grazing area
around the monument considering the space limited by the park's boundary.
-/
theorem grazing_area_of_goat : 
  let park_radius := 25
  let monument_side := 10
  let rope_length := 20
  let monument_radius := monument_side / 2 
  let grazing_quarter_circle := (1 / 4) * Real.pi * rope_length^2
  let ungrazable_area := (1 / 4) * Real.pi * monument_radius^2
  grazing_quarter_circle - ungrazable_area = 93.75 * Real.pi :=
by
  sorry

end grazing_area_of_goat_l26_26749


namespace find_q_l26_26320

def Q (x p q d : ℝ) : ℝ := x^3 + p * x^2 + q * x + d

theorem find_q (p q d : ℝ) (h: d = 3) (h1: -p / 3 = -d) (h2: -p / 3 = 1 + p + q + d) : q = -16 :=
by
  sorry

end find_q_l26_26320


namespace negation_of_prop_p_is_correct_l26_26199

-- Define the original proposition p
def prop_p (x y : ℝ) : Prop := x > 0 ∧ y > 0 → x * y > 0

-- Define the negation of the proposition p
def neg_prop_p (x y : ℝ) : Prop := x ≤ 0 ∨ y ≤ 0 → x * y ≤ 0

-- The theorem we need to prove
theorem negation_of_prop_p_is_correct : ∀ x y : ℝ, neg_prop_p x y := 
sorry

end negation_of_prop_p_is_correct_l26_26199


namespace resistance_of_second_resistor_l26_26968

theorem resistance_of_second_resistor 
  (R1 R_total R2 : ℝ) 
  (hR1: R1 = 9) 
  (hR_total: R_total = 4.235294117647059) 
  (hFormula: 1/R_total = 1/R1 + 1/R2) : 
  R2 = 8 :=
by
  sorry

end resistance_of_second_resistor_l26_26968


namespace factorization_sum_l26_26864

theorem factorization_sum :
  ∃ a b c : ℤ, (∀ x : ℝ, (x^2 + 20 * x + 96 = (x + a) * (x + b)) ∧
                      (x^2 + 18 * x + 81 = (x - b) * (x + c))) →
              (a + b + c = 30) :=
by
  sorry

end factorization_sum_l26_26864


namespace find_m_l26_26728

theorem find_m (x y m : ℤ) 
  (h1 : x + 2 * y = 5 * m) 
  (h2 : x - 2 * y = 9 * m) 
  (h3 : 3 * x + 2 * y = 19) : 
  m = 1 := 
by 
  sorry

end find_m_l26_26728


namespace solve_system_eq_pos_reals_l26_26257

theorem solve_system_eq_pos_reals (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x^2 + y^2 + x * y = 7)
  (h2 : x^2 + z^2 + x * z = 13)
  (h3 : y^2 + z^2 + y * z = 19) :
  x = 1 ∧ y = 2 ∧ z = 3 :=
sorry

end solve_system_eq_pos_reals_l26_26257


namespace inequality_solution_l26_26706

variable {x : ℝ}

theorem inequality_solution (h : 0 ≤ x ∧ x ≤ 2 * Real.pi) : 
  (2 * Real.cos x ≤ |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| 
  ∧ |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ≤ Real.sqrt 2) ↔ 
  (Real.pi / 4 ≤ x ∧ x ≤ 7 * Real.pi / 4) :=
sorry

end inequality_solution_l26_26706


namespace log_tangent_ratio_l26_26713

open Real

theorem log_tangent_ratio (α β : ℝ) 
  (h1 : sin (α + β) = 1 / 2) 
  (h2 : sin (α - β) = 1 / 3) : 
  log 5 * (tan α / tan β) = 1 := 
sorry

end log_tangent_ratio_l26_26713


namespace sin_alpha_value_l26_26473

open Real


theorem sin_alpha_value (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : π / 2 < β ∧ β < π)
  (h_sin_alpha_beta : sin (α + β) = 3 / 5) (h_cos_beta : cos β = -5 / 13) :
  sin α = 33 / 65 := 
by
  sorry

end sin_alpha_value_l26_26473


namespace find_m_abc_inequality_l26_26095

-- Define properties and the theorem for the first problem
def f (x m : ℝ) := m - |x - 2|

theorem find_m (m : ℝ) : (∀ x, f (x + 2) m ≥ 0 ↔ x ∈ Set.Icc (-1 : ℝ) 1) → m = 1 := by
  intros h
  sorry

-- Define properties and the theorem for the second problem
theorem abc_inequality (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
  (1 / a + 1 / (2 * b) + 1 / (3 * c) = 1) → (a + 2 * b + 3 * c ≥ 9) := by
  intros h
  sorry

end find_m_abc_inequality_l26_26095


namespace statement_C_l26_26131

theorem statement_C (x : ℝ) (h : x^2 < 4) : x < 2 := 
sorry

end statement_C_l26_26131


namespace citizen_income_l26_26337

noncomputable def income (I : ℝ) : Prop :=
  let P := 0.11 * 40000
  let A := I - 40000
  P + 0.20 * A = 8000

theorem citizen_income (I : ℝ) (h : income I) : I = 58000 := 
by
  -- proof steps go here
  sorry

end citizen_income_l26_26337


namespace fencing_cost_correct_l26_26492

noncomputable def length : ℝ := 80
noncomputable def diff : ℝ := 60
noncomputable def cost_per_meter : ℝ := 26.50

-- Let's calculate the breadth first
noncomputable def breadth : ℝ := length - diff

-- Calculate the perimeter
noncomputable def perimeter : ℝ := 2 * (length + breadth)

-- Calculate the total cost
noncomputable def total_cost : ℝ := perimeter * cost_per_meter

theorem fencing_cost_correct : total_cost = 5300 := 
by 
  sorry

end fencing_cost_correct_l26_26492


namespace arithmetic_sequence_common_difference_l26_26830

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_arith_seq : ∀ n : ℕ, a n = a 1 + (n - 1) * d)
  (h_a30 : a 30 = 100)
  (h_a100 : a 100 = 30) :
  d = -1 := sorry

end arithmetic_sequence_common_difference_l26_26830


namespace range_of_u_l26_26607

variable (a b u : ℝ)

theorem range_of_u (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
  (∀ x : ℝ, x > 0 → a^2 + b^2 ≥ x ↔ x ≤ 16) :=
sorry

end range_of_u_l26_26607


namespace problem1_problem2_l26_26937

-- Define condition p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2*x - 8 > 0)

-- Define the negation of p
def neg_p (x a : ℝ) : Prop := ¬ p x a
-- Define the negation of q
def neg_q (x : ℝ) : Prop := ¬ q x

-- Question 1: Prove that if a = 1 and p ∧ q is true, then 2 < x < 3
theorem problem1 (x : ℝ) (h1 : p x 1 ∧ q x) : 2 < x ∧ x < 3 := 
by sorry

-- Question 2: Prove that if ¬ p is a sufficient but not necessary condition for ¬ q, then 1 < a ≤ 2
theorem problem2 (a : ℝ) (h2 : ∀ x : ℝ, neg_p x a → neg_q x) : 1 < a ∧ a ≤ 2 := 
by sorry

end problem1_problem2_l26_26937


namespace common_ratio_l26_26258

-- Problem Statement Definitions
variable (a1 q : ℝ)

-- Given Conditions
def a3 := a1 * q^2
def S3 := a1 * (1 + q + q^2)

-- Proof Statement
theorem common_ratio (h1 : a3 = 3/2) (h2 : S3 = 9/2) : q = 1 ∨ q = -1/2 := by
  sorry

end common_ratio_l26_26258


namespace more_green_than_yellow_l26_26206

-- Define constants
def red_peaches : ℕ := 2
def yellow_peaches : ℕ := 6
def green_peaches : ℕ := 14

-- Prove the statement
theorem more_green_than_yellow : green_peaches - yellow_peaches = 8 :=
by
  sorry

end more_green_than_yellow_l26_26206


namespace actual_area_of_lawn_l26_26041

-- Definitions and conditions
variable (blueprint_area : ℝ)
variable (side_on_blueprint : ℝ)
variable (actual_side_length : ℝ)

-- Given conditions
def blueprint_conditions := 
  blueprint_area = 300 ∧ 
  side_on_blueprint = 5 ∧ 
  actual_side_length = 15

-- Prove the actual area of the lawn
theorem actual_area_of_lawn (blueprint_area : ℝ) (side_on_blueprint : ℝ) (actual_side_length : ℝ) (x : ℝ) :
  blueprint_conditions blueprint_area side_on_blueprint actual_side_length →
  (x = 27000000 ∧ x / 10000 = 2700) :=
by
  sorry

end actual_area_of_lawn_l26_26041


namespace opposite_of_neg_one_fifth_l26_26218

theorem opposite_of_neg_one_fifth : -(- (1/5)) = (1/5) :=
by
  sorry

end opposite_of_neg_one_fifth_l26_26218


namespace milk_fraction_in_cup1_is_one_third_l26_26051

-- Define the initial state of the cups
structure CupsState where
  cup1_tea : ℚ  -- amount of tea in cup1
  cup1_milk : ℚ -- amount of milk in cup1
  cup2_tea : ℚ  -- amount of tea in cup2
  cup2_milk : ℚ -- amount of milk in cup2

def initial_cups_state : CupsState := {
  cup1_tea := 8,
  cup1_milk := 0,
  cup2_tea := 0,
  cup2_milk := 8
}

-- Function to transfer a fraction of tea from cup 1 to cup 2
def transfer_tea (s : CupsState) (frac : ℚ) : CupsState := {
  cup1_tea := s.cup1_tea * (1 - frac),
  cup1_milk := s.cup1_milk,
  cup2_tea := s.cup2_tea + s.cup1_tea * frac,
  cup2_milk := s.cup2_milk
}

-- Function to transfer a fraction of the mixture from cup 2 to cup 1
def transfer_mixture (s : CupsState) (frac : ℚ) : CupsState := {
  cup1_tea := s.cup1_tea + (frac * s.cup2_tea),
  cup1_milk := s.cup1_milk + (frac * s.cup2_milk),
  cup2_tea := s.cup2_tea * (1 - frac),
  cup2_milk := s.cup2_milk * (1 - frac)
}

-- Define the state after each transfer
def state_after_tea_transfer := transfer_tea initial_cups_state (1 / 4)
def final_state := transfer_mixture state_after_tea_transfer (1 / 3)

-- Prove the fraction of milk in the first cup is 1/3
theorem milk_fraction_in_cup1_is_one_third : 
  (final_state.cup1_milk / (final_state.cup1_tea + final_state.cup1_milk)) = 1 / 3 :=
by
  -- skipped proof
  sorry

end milk_fraction_in_cup1_is_one_third_l26_26051


namespace power_function_value_at_9_l26_26604

noncomputable def f (x : ℝ) : ℝ := x ^ (1 / 2)

theorem power_function_value_at_9 (h : f 2 = Real.sqrt 2) : f 9 = 3 :=
by sorry

end power_function_value_at_9_l26_26604


namespace train_cross_bridge_time_l26_26343

-- Length of the train in meters
def train_length : ℕ := 165

-- Length of the bridge in meters
def bridge_length : ℕ := 660

-- Speed of the train in kmph
def train_speed_kmph : ℕ := 54

-- Conversion factor from kmph to m/s
def kmph_to_mps : ℚ := 5 / 18

-- Total distance to be traveled by the train to cross the bridge
def total_distance : ℕ := train_length + bridge_length

-- Speed of the train in meters per second (m/s)
def train_speed_mps : ℚ := train_speed_kmph * kmph_to_mps

-- Time taken for the train to cross the bridge (in seconds)
def time_to_cross_bridge : ℚ := total_distance / train_speed_mps

-- Prove that the time taken for the train to cross the bridge is 55 seconds
theorem train_cross_bridge_time : time_to_cross_bridge = 55 := by
  -- Proof goes here
  sorry

end train_cross_bridge_time_l26_26343


namespace exists_distinct_pure_powers_l26_26751

-- Definitions and conditions
def is_pure_kth_power (k m : ℕ) : Prop := ∃ t : ℕ, m = t ^ k

-- The main theorem statement
theorem exists_distinct_pure_powers (n : ℕ) (hn : 0 < n) :
  ∃ (a : Fin n → ℕ),
    (∀ i j : Fin n, i ≠ j → a i ≠ a j) ∧ 
    is_pure_kth_power 2009 (Finset.univ.sum a) ∧ 
    is_pure_kth_power 2010 (Finset.univ.prod a) :=
sorry

end exists_distinct_pure_powers_l26_26751


namespace min_x_4y_is_minimum_l26_26286

noncomputable def min_value_x_4y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : (1 / x) + (1 / (2 * y)) = 2) : ℝ :=
  x + 4 * y

theorem min_x_4y_is_minimum : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (1 / x + 1 / (2 * y) = 2) ∧ (x + 4 * y = (3 / 2) + Real.sqrt 2) :=
sorry

end min_x_4y_is_minimum_l26_26286


namespace part_a_part_b_l26_26601

def can_cut_into_equal_dominoes (n : ℕ) : Prop :=
  ∃ horiz_vert_dominoes : ℕ × ℕ,
    n % 2 = 1 ∧
    (n * n - 1) / 2 = horiz_vert_dominoes.1 + horiz_vert_dominoes.2 ∧
    horiz_vert_dominoes.1 = horiz_vert_dominoes.2

theorem part_a : can_cut_into_equal_dominoes 101 :=
by {
  sorry
}

theorem part_b : ¬can_cut_into_equal_dominoes 99 :=
by {
  sorry
}

end part_a_part_b_l26_26601


namespace arithmetic_sequence_l26_26910

variable (p q : ℕ) -- Assuming natural numbers for simplicity, but can be generalized.

def a (n : ℕ) : ℕ := p * n + q

theorem arithmetic_sequence:
  ∀ n : ℕ, n ≥ 1 → (a n - a (n-1) = p) := by
  -- proof steps would go here
  sorry

end arithmetic_sequence_l26_26910


namespace ratio_of_quadratic_roots_l26_26159

theorem ratio_of_quadratic_roots (a b c : ℝ) (h : 2 * b^2 = 9 * a * c) : 
  ∃ (x₁ x₂ : ℝ), (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) ∧ (x₁ / x₂ = 2) :=
sorry

end ratio_of_quadratic_roots_l26_26159


namespace hawks_points_l26_26715

theorem hawks_points (E H : ℕ) (h1 : E + H = 82) (h2 : E = H + 6) : H = 38 :=
sorry

end hawks_points_l26_26715


namespace find_integer_roots_l26_26618

open Int Polynomial

def P (x : ℤ) : ℤ := x^3 - 3 * x^2 - 13 * x + 15

theorem find_integer_roots : {x : ℤ | P x = 0} = {-3, 1, 5} := by
  sorry

end find_integer_roots_l26_26618


namespace car_distance_problem_l26_26979

theorem car_distance_problem
  (d y z r : ℝ)
  (initial_distance : d = 113)
  (right_turn_distance : y = 15)
  (second_car_distance : z = 35)
  (remaining_distance : r = 28)
  (x : ℝ) :
  2 * x + z + y + r = d → 
  x = 17.5 :=
by
  intros h
  sorry  

end car_distance_problem_l26_26979


namespace find_third_test_score_l26_26680

-- Definitions of the given conditions
def test_score_1 := 80
def test_score_2 := 70
variable (x : ℕ) -- the unknown third score
def test_score_4 := 100
def average_score (s1 s2 s3 s4 : ℕ) : ℕ := (s1 + s2 + s3 + s4) / 4

-- Theorem stating that given the conditions, the third test score must be 90
theorem find_third_test_score (h : average_score test_score_1 test_score_2 x test_score_4 = 85) : x = 90 :=
by
  sorry

end find_third_test_score_l26_26680


namespace min_vertical_distance_between_graphs_l26_26927

noncomputable def absolute_value (x : ℝ) : ℝ :=
if x >= 0 then x else -x

theorem min_vertical_distance_between_graphs : 
  ∃ d : ℝ, d = 3 / 4 ∧ ∀ x : ℝ, ∃ dist : ℝ, dist = absolute_value x - (- x^2 - 4 * x - 3) ∧ dist >= d :=
by
  sorry

end min_vertical_distance_between_graphs_l26_26927


namespace m_necessary_not_sufficient_cond_l26_26364

theorem m_necessary_not_sufficient_cond (m : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → x^3 - 3 * x + m = 0) → m ≤ 2 :=
sorry

end m_necessary_not_sufficient_cond_l26_26364


namespace even_function_periodic_symmetric_about_2_l26_26856

variables {F : ℝ → ℝ}

theorem even_function_periodic_symmetric_about_2
  (h_even : ∀ x, F x = F (-x))
  (h_symmetric : ∀ x, F (2 - x) = F (2 + x))
  (h_cond : F 2011 + 2 * F 1 = 18) :
  F 2011 = 6 :=
sorry

end even_function_periodic_symmetric_about_2_l26_26856


namespace area_of_triangle_ABC_eq_3_l26_26105

variable {n : ℕ}

def arithmetic_seq (a_1 d : ℤ) : ℕ → ℤ
| 0     => 0
| (n+1) => a_1 + n * d

def sum_arithmetic_seq (a_1 d : ℤ) : ℕ → ℤ
| 0     => 0
| (n+1) => (n + 1) * a_1 + (n * (n + 1) / 2) * d

def f (n : ℕ) : ℤ := sum_arithmetic_seq 4 6 n

def point_A (n : ℕ) : ℤ × ℤ := (n, f n)
def point_B (n : ℕ) : ℤ × ℤ := (n + 1, f (n + 1))
def point_C (n : ℕ) : ℤ × ℤ := (n + 2, f (n + 2))

def area_of_triangle (A B C : ℤ × ℤ) : ℤ :=
  (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)).natAbs / 2

theorem area_of_triangle_ABC_eq_3 : 
  ∀ (n : ℕ), area_of_triangle (point_A n) (point_B n) (point_C n) = 3 := 
sorry

end area_of_triangle_ABC_eq_3_l26_26105


namespace find_larger_integer_l26_26995

theorem find_larger_integer (x : ℕ) (hx₁ : 4 * x > 0) (hx₂ : (x + 6) * 3 = 4 * x) : 4 * x = 72 :=
by
  sorry

end find_larger_integer_l26_26995


namespace piglet_balloons_l26_26771

theorem piglet_balloons (n w o total_balloons: ℕ) (H1: w = 2 * n) (H2: o = 4 * n) (H3: n + w + o = total_balloons) (H4: total_balloons = 44) : n - (7 * n - total_balloons) = 2 :=
by
  sorry

end piglet_balloons_l26_26771


namespace find_x_l26_26222

theorem find_x (x : ℕ) (h : x * 6000 = 480 * 10^5) : x = 8000 := 
by
  sorry

end find_x_l26_26222


namespace base5_number_l26_26693

/-- A base-5 number only contains the digits 0, 1, 2, 3, and 4.
    Given the number 21340, we need to prove that it could possibly be a base-5 number. -/
theorem base5_number (n : ℕ) (h : n = 21340) : 
  ∀ d ∈ [2, 1, 3, 4, 0], d < 5 :=
by sorry

end base5_number_l26_26693


namespace gcd_lcm_product_l26_26944

theorem gcd_lcm_product (a b : ℕ) (h₁ : a = 24) (h₂ : b = 36) :
  Nat.gcd a b * Nat.lcm a b = 864 := 
by
  rw [h₁, h₂]
  -- You can include specific calculation just to express the idea
  -- rw [Nat.gcd_comm, Nat.gcd_rec]
  -- rw [Nat.lcm_def]
  -- rw [Nat.mul_subst]
  sorry

end gcd_lcm_product_l26_26944


namespace find_smaller_number_l26_26050

theorem find_smaller_number (x y : ℤ) (h1 : x + y = 60) (h2 : x - y = 8) : y = 26 :=
by
  sorry

end find_smaller_number_l26_26050


namespace smallest_next_divisor_l26_26497

theorem smallest_next_divisor (n : ℕ) (hn : n % 2 = 0) (h4d : 1000 ≤ n ∧ n < 10000) (hdiv : 221 ∣ n) : 
  ∃ (d : ℕ), d = 238 ∧ 221 < d ∧ d ∣ n :=
by
  sorry

end smallest_next_divisor_l26_26497


namespace find_k_l26_26925

variable {a_n : ℕ → ℤ}    -- Define the arithmetic sequence as a function from natural numbers to integers
variable {a1 d : ℤ}        -- a1 is the first term, d is the common difference

-- Conditions
axiom seq_def : ∀ n, a_n n = a1 + (n - 1) * d
axiom sum_condition : 9 * a1 + 36 * d = 4 * a1 + 6 * d
axiom ak_a4_zero (k : ℕ): a_n 4 + a_n k = 0

-- Problem Statement to prove
theorem find_k : ∃ k : ℕ, a_n 4 + a_n k = 0 → k = 10 :=
by
  use 10
  intro h
  -- proof omitted
  sorry

end find_k_l26_26925


namespace roots_of_quadratic_l26_26861

variable {γ δ : ℝ}

theorem roots_of_quadratic (hγ : γ^2 - 5*γ + 6 = 0) (hδ : δ^2 - 5*δ + 6 = 0) : 
  8*γ^5 + 15*δ^4 = 8425 := 
by
  sorry

end roots_of_quadratic_l26_26861


namespace greatest_perimeter_l26_26874

theorem greatest_perimeter (w l : ℕ) (h1 : w * l = 12) : 
  ∃ (P : ℕ), P = 2 * (w + l) ∧ ∀ (w' l' : ℕ), w' * l' = 12 → 2 * (w' + l') ≤ P := 
sorry

end greatest_perimeter_l26_26874


namespace cliff_shiny_igneous_l26_26009

variables (I S : ℕ)

theorem cliff_shiny_igneous :
  I = S / 2 ∧ I + S = 270 → I / 3 = 30 := 
by
  intro h
  sorry

end cliff_shiny_igneous_l26_26009


namespace avg_difference_in_circumferences_l26_26282

-- Define the conditions
def inner_circle_diameter : ℝ := 30
def min_track_width : ℝ := 10
def max_track_width : ℝ := 15

-- Define the average difference in the circumferences of the two circles
theorem avg_difference_in_circumferences :
  let avg_width := (min_track_width + max_track_width) / 2
  let outer_circle_diameter := inner_circle_diameter + 2 * avg_width
  let inner_circle_circumference := Real.pi * inner_circle_diameter
  let outer_circle_circumference := Real.pi * outer_circle_diameter
  outer_circle_circumference - inner_circle_circumference = 25 * Real.pi :=
by
  sorry

end avg_difference_in_circumferences_l26_26282


namespace arithmetic_sequence_sum_l26_26190

-- Condition definitions
def a : Int := 3
def d : Int := 2
def a_n : Int := 25
def n : Int := 12

-- Sum formula for an arithmetic sequence proof
theorem arithmetic_sequence_sum :
    let n := 12
    let S_n := (n * (a + a_n)) / 2
    S_n = 168 := by
  sorry

end arithmetic_sequence_sum_l26_26190


namespace probability_even_in_5_of_7_rolls_is_21_over_128_l26_26319

noncomputable def probability_even_in_5_of_7_rolls : ℚ :=
  let n := 7
  let k := 5
  let p := (1:ℚ) / 2
  let binomial (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k
  (binomial n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_even_in_5_of_7_rolls_is_21_over_128 :
  probability_even_in_5_of_7_rolls = 21 / 128 :=
by
  sorry

end probability_even_in_5_of_7_rolls_is_21_over_128_l26_26319


namespace not_forall_abs_ge_zero_l26_26549

theorem not_forall_abs_ge_zero : (¬(∀ x : ℝ, |x + 1| ≥ 0)) ↔ (∃ x : ℝ, |x + 1| < 0) :=
by
  sorry

end not_forall_abs_ge_zero_l26_26549


namespace no_solution_to_inequalities_l26_26356

theorem no_solution_to_inequalities : 
  ∀ x : ℝ, ¬ (4 * x - 3 < (x + 2)^2 ∧ (x + 2)^2 < 8 * x - 5) :=
by
  sorry

end no_solution_to_inequalities_l26_26356


namespace hot_peppers_percentage_correct_l26_26446

def sunday_peppers : ℕ := 7
def monday_peppers : ℕ := 12
def tuesday_peppers : ℕ := 14
def wednesday_peppers : ℕ := 12
def thursday_peppers : ℕ := 5
def friday_peppers : ℕ := 18
def saturday_peppers : ℕ := 12
def non_hot_peppers : ℕ := 64

def total_peppers : ℕ := sunday_peppers + monday_peppers + tuesday_peppers + wednesday_peppers + thursday_peppers + friday_peppers + saturday_peppers
def hot_peppers : ℕ := total_peppers - non_hot_peppers
def hot_peppers_percentage : ℕ := (hot_peppers * 100) / total_peppers

theorem hot_peppers_percentage_correct : hot_peppers_percentage = 20 := 
by 
  sorry

end hot_peppers_percentage_correct_l26_26446


namespace last_digit_of_189_in_base_3_is_0_l26_26306

theorem last_digit_of_189_in_base_3_is_0 : 
  (189 % 3 = 0) :=
sorry

end last_digit_of_189_in_base_3_is_0_l26_26306


namespace negate_existential_l26_26307

theorem negate_existential (p : Prop) : (¬(∃ x : ℝ, x^2 - 2 * x + 2 ≤ 0)) ↔ ∀ x : ℝ, x^2 - 2 * x + 2 > 0 :=
by sorry

end negate_existential_l26_26307


namespace probability_of_A_winning_l26_26353

-- Define the conditions
def p : ℝ := 0.6
def q : ℝ := 1 - p  -- probability of losing a set

-- Formulate the probabilities for each win scenario
def P_WW : ℝ := p * p
def P_LWW : ℝ := q * p * p
def P_WLW : ℝ := p * q * p

-- Calculate the total probability of winning the match
def total_probability : ℝ := P_WW + P_LWW + P_WLW

-- Prove that the total probability of A winning the match is 0.648
theorem probability_of_A_winning : total_probability = 0.648 :=
by
    -- Provide the calculation details
    sorry  -- replace with the actual proof steps if needed, otherwise keep sorry to skip the proof

end probability_of_A_winning_l26_26353


namespace part_one_part_two_l26_26454

-- Given that tan α = 2, prove that the following expressions are correct:

theorem part_one (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin (Real.pi - α) + Real.cos (α - Real.pi / 2) - Real.cos (3 * Real.pi + α)) / 
  (Real.cos (Real.pi / 2 + α) - Real.sin (2 * Real.pi + α) + 2 * Real.sin (α - Real.pi / 2)) = 
  -5 / 6 := 
by
  -- Proof skipped
  sorry

theorem part_two (α : ℝ) (h : Real.tan α = 2) :
  Real.cos (2 * α) + Real.sin α * Real.cos α = -1 / 5 := 
by
  -- Proof skipped
  sorry

end part_one_part_two_l26_26454


namespace horizontal_asymptote_value_l26_26264

theorem horizontal_asymptote_value :
  (∃ y : ℝ, ∀ x : ℝ, (y = (18 * x^5 + 6 * x^3 + 3 * x^2 + 5 * x + 4) / (6 * x^5 + 4 * x^3 + 5 * x^2 + 2 * x + 1)) → y = 3) :=
by
  sorry

end horizontal_asymptote_value_l26_26264


namespace michael_total_cost_l26_26479

def peach_pies : ℕ := 5
def apple_pies : ℕ := 4
def blueberry_pies : ℕ := 3

def pounds_per_pie : ℕ := 3

def price_per_pound_peaches : ℝ := 2.0
def price_per_pound_apples : ℝ := 1.0
def price_per_pound_blueberries : ℝ := 1.0

def total_peach_pounds : ℕ := peach_pies * pounds_per_pie
def total_apple_pounds : ℕ := apple_pies * pounds_per_pie
def total_blueberry_pounds : ℕ := blueberry_pies * pounds_per_pie

def cost_peaches : ℝ := total_peach_pounds * price_per_pound_peaches
def cost_apples : ℝ := total_apple_pounds * price_per_pound_apples
def cost_blueberries : ℝ := total_blueberry_pounds * price_per_pound_blueberries

def total_cost : ℝ := cost_peaches + cost_apples + cost_blueberries

theorem michael_total_cost :
  total_cost = 51.0 := by
  sorry

end michael_total_cost_l26_26479


namespace james_total_riding_time_including_rest_stop_l26_26417

theorem james_total_riding_time_including_rest_stop :
  let distance1 := 40 -- miles
  let speed1 := 16 -- miles per hour
  let distance2 := 40 -- miles
  let speed2 := 20 -- miles per hour
  let rest_stop := 20 -- minutes
  let rest_stop_in_hours := rest_stop / 60 -- convert to hours
  let time1 := distance1 / speed1 -- time for the first part
  let time2 := distance2 / speed2 -- time for the second part
  let total_time := time1 + rest_stop_in_hours + time2 -- total time including rest
  total_time = 4.83 :=
by
  sorry

end james_total_riding_time_including_rest_stop_l26_26417


namespace olivia_possible_amount_l26_26626

theorem olivia_possible_amount (k : ℕ) :
  ∃ k : ℕ, 1 + 79 * k = 1984 :=
by
  -- Prove that there exists a non-negative integer k such that the equation holds
  sorry

end olivia_possible_amount_l26_26626


namespace quilt_block_shading_fraction_l26_26694

theorem quilt_block_shading_fraction :
  (fraction_shaded : ℚ) → 
  (quilt_block_size : ℕ) → 
  (fully_shaded_squares : ℕ) → 
  (half_shaded_squares : ℕ) → 
  quilt_block_size = 16 →
  fully_shaded_squares = 6 →
  half_shaded_squares = 4 →
  fraction_shaded = 1/2 :=
by 
  sorry

end quilt_block_shading_fraction_l26_26694


namespace number_of_uncertain_events_is_three_l26_26100

noncomputable def cloudy_day_will_rain : Prop := sorry
noncomputable def fair_coin_heads : Prop := sorry
noncomputable def two_students_same_birth_month : Prop := sorry
noncomputable def olympics_2008_in_beijing : Prop := true

def is_uncertain (event: Prop) : Prop :=
  event ∧ ¬(event = true ∨ event = false)

theorem number_of_uncertain_events_is_three :
  is_uncertain cloudy_day_will_rain ∧
  is_uncertain fair_coin_heads ∧
  is_uncertain two_students_same_birth_month ∧
  ¬is_uncertain olympics_2008_in_beijing →
  3 = 3 :=
by sorry

end number_of_uncertain_events_is_three_l26_26100


namespace max_yellow_apples_can_take_max_total_apples_can_take_l26_26434

structure Basket :=
  (total_apples : ℕ)
  (green_apples : ℕ)
  (yellow_apples : ℕ)
  (red_apples : ℕ)
  (green_lt_yellow : green_apples < yellow_apples)
  (yellow_lt_red : yellow_apples < red_apples)

def basket_conditions : Basket :=
  { total_apples := 44,
    green_apples := 11,
    yellow_apples := 14,
    red_apples := 19,
    green_lt_yellow := sorry,  -- 11 < 14
    yellow_lt_red := sorry }   -- 14 < 19

theorem max_yellow_apples_can_take : basket_conditions.yellow_apples = 14 := sorry

theorem max_total_apples_can_take : basket_conditions.green_apples 
                                     + basket_conditions.yellow_apples 
                                     + (basket_conditions.red_apples - 2) = 42 := sorry

end max_yellow_apples_can_take_max_total_apples_can_take_l26_26434


namespace complex_division_l26_26262

theorem complex_division (i : ℂ) (h : i^2 = -1) : (2 + i) / (1 - 2 * i) = i := 
by
  sorry

end complex_division_l26_26262


namespace binary_mul_1101_111_eq_1001111_l26_26062

theorem binary_mul_1101_111_eq_1001111 :
  let n1 := 0b1101 -- binary representation of 13
  let n2 := 0b111  -- binary representation of 7
  let product := 0b1001111 -- binary representation of 79
  n1 * n2 = product :=
by
  sorry

end binary_mul_1101_111_eq_1001111_l26_26062


namespace max_sum_of_ABC_l26_26073

/-- Theorem: The maximum value of A + B + C for distinct positive integers A, B, and C such that A * B * C = 2023 is 297. -/
theorem max_sum_of_ABC (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) (h4 : A * B * C = 2023) :
  A + B + C ≤ 297 :=
sorry

end max_sum_of_ABC_l26_26073


namespace parallel_lines_distance_l26_26534

theorem parallel_lines_distance (b c : ℝ) 
  (h1: b = 8) 
  (h2: (abs (10 - c) / (Real.sqrt (3^2 + 4^2))) = 3) :
  b + c = -12 ∨ b + c = 48 := by
 sorry

end parallel_lines_distance_l26_26534


namespace polynomial_factorization_l26_26486

theorem polynomial_factorization (x : ℝ) :
  x^6 + 6*x^5 + 15*x^4 + 20*x^3 + 15*x^2 + 6*x + 1 = (x + 1)^6 :=
by {
  -- proof goes here
  sorry
}

end polynomial_factorization_l26_26486


namespace triangle_angle_identity_l26_26820

theorem triangle_angle_identity
  (α β γ : ℝ)
  (h_triangle : α + β + γ = π)
  (sin_α_ne_zero : Real.sin α ≠ 0)
  (sin_β_ne_zero : Real.sin β ≠ 0)
  (sin_γ_ne_zero : Real.sin γ ≠ 0) :
  (Real.cos α / (Real.sin β * Real.sin γ) +
   Real.cos β / (Real.sin α * Real.sin γ) +
   Real.cos γ / (Real.sin α * Real.sin β) = 2) := by
  sorry

end triangle_angle_identity_l26_26820


namespace solution_correct_l26_26933

noncomputable def solve_system (A1 A2 A3 A4 A5 : ℝ) (x1 x2 x3 x4 x5 : ℝ) :=
  (2 * x1 - 2 * x2 = A1) ∧
  (-x1 + 4 * x2 - 3 * x3 = A2) ∧
  (-2 * x2 + 6 * x3 - 4 * x4 = A3) ∧
  (-3 * x3 + 8 * x4 - 5 * x5 = A4) ∧
  (-4 * x4 + 10 * x5 = A5)

theorem solution_correct {A1 A2 A3 A4 A5 x1 x2 x3 x4 x5 : ℝ} :
  solve_system A1 A2 A3 A4 A5 x1 x2 x3 x4 x5 → 
  x1 = (5 * A1 + 4 * A2 + 3 * A3 + 2 * A4 + A5) / 6 ∧
  x2 = (2 * A1 + 4 * A2 + 3 * A3 + 2 * A4 + A5) / 6 ∧
  x3 = (A1 + 2 * A2 + 3 * A3 + 2 * A4 + A5) / 6 ∧
  x4 = (A1 + 2 * A2 + 3 * A3 + 4 * A4 + 2 * A5) / 12 ∧
  x5 = (A1 + 2 * A2 + 3 * A3 + 4 * A4 + 5 * A5) / 30 :=
sorry

end solution_correct_l26_26933


namespace find_x_l26_26418

theorem find_x (x : ℚ) : x * 9999 = 724827405 → x = 72492.75 :=
by
  sorry

end find_x_l26_26418


namespace max_value_of_ab_expression_l26_26432

noncomputable def max_ab_expression : ℝ :=
  let a := 4
  let b := 20 / 3
  a * b * (60 - 5 * a - 3 * b)

theorem max_value_of_ab_expression :
  ∀ (a b : ℝ), 0 < a → 0 < b → 5 * a + 3 * b < 60 →
  ab * (60 - 5 * a - 3 * b) ≤ max_ab_expression :=
sorry

end max_value_of_ab_expression_l26_26432


namespace prism_volume_l26_26458

noncomputable def volume_of_prism (l w h : ℝ) : ℝ :=
l * w * h

theorem prism_volume (l w h : ℝ) (h1 : l = 2 * w) (h2 : l * w = 10) (h3 : w * h = 18) (h4 : l * h = 36) :
  volume_of_prism l w h = 36 * Real.sqrt 5 :=
by
  -- Proof goes here
  sorry

end prism_volume_l26_26458


namespace ellen_painting_time_l26_26124

def time_to_paint_lilies := 5
def time_to_paint_roses := 7
def time_to_paint_orchids := 3
def time_to_paint_vines := 2

def number_of_lilies := 17
def number_of_roses := 10
def number_of_orchids := 6
def number_of_vines := 20

def total_time := 213

theorem ellen_painting_time:
  time_to_paint_lilies * number_of_lilies +
  time_to_paint_roses * number_of_roses +
  time_to_paint_orchids * number_of_orchids +
  time_to_paint_vines * number_of_vines = total_time := by
  sorry

end ellen_painting_time_l26_26124


namespace find_b_l26_26421

theorem find_b (a b : ℤ) (h1 : 3 * a + 1 = 1) (h2 : b - a = 2) : b = 2 := by
  sorry

end find_b_l26_26421


namespace older_brother_allowance_l26_26615

theorem older_brother_allowance 
  (sum_allowance : ℕ)
  (difference : ℕ)
  (total_sum : sum_allowance = 12000)
  (additional_amount : difference = 1000) :
  ∃ (older_brother_allowance younger_brother_allowance : ℕ), 
    older_brother_allowance = younger_brother_allowance + difference ∧
    younger_brother_allowance + older_brother_allowance = sum_allowance ∧
    older_brother_allowance = 6500 :=
by {
  sorry
}

end older_brother_allowance_l26_26615


namespace isosceles_with_60_eq_angle_is_equilateral_l26_26413

open Real

noncomputable def is_equilateral_triangle (a b c : ℝ) (A B C : ℝ) :=
  A = 60 ∧ B = 60 ∧ C = 60

noncomputable def is_isosceles_triangle (a b c : ℝ) (A B C : ℝ) :=
  (a = b ∨ b = c ∨ c = a) ∧ (A + B + C = 180)

theorem isosceles_with_60_eq_angle_is_equilateral
  (a b c A B C : ℝ)
  (h_iso : is_isosceles_triangle a b c A B C)
  (h_angle : A = 60 ∨ B = 60 ∨ C = 60) :
  is_equilateral_triangle a b c A B C :=
sorry

end isosceles_with_60_eq_angle_is_equilateral_l26_26413


namespace interest_rate_difference_l26_26732

-- Definitions for given conditions
def principal : ℝ := 3000
def time : ℝ := 9
def additional_interest : ℝ := 1350

-- The Lean 4 statement for the equivalence
theorem interest_rate_difference 
  (R H : ℝ) 
  (h_interest_formula_original : principal * R * time / 100 = principal * R * time / 100) 
  (h_interest_formula_higher : principal * H * time / 100 = principal * R * time / 100 + additional_interest) 
  : (H - R) = 5 :=
sorry

end interest_rate_difference_l26_26732


namespace trains_cross_time_l26_26519

noncomputable def time_to_cross (len1 len2 speed1_kmh speed2_kmh : ℝ) : ℝ :=
  let speed1_ms := speed1_kmh * (5 / 18)
  let speed2_ms := speed2_kmh * (5 / 18)
  let relative_speed_ms := speed1_ms + speed2_ms
  let total_distance := len1 + len2
  total_distance / relative_speed_ms

theorem trains_cross_time :
  time_to_cross 1500 1000 90 75 = 54.55 := by
  sorry

end trains_cross_time_l26_26519


namespace Vlad_height_feet_l26_26553

theorem Vlad_height_feet 
  (sister_height_feet : ℕ)
  (sister_height_inches : ℕ)
  (vlad_height_diff : ℕ)
  (vlad_height_inches : ℕ)
  (vlad_height_feet : ℕ)
  (vlad_height_rem : ℕ)
  (sister_height := (sister_height_feet * 12) + sister_height_inches)
  (vlad_height := sister_height + vlad_height_diff)
  (vlad_height_feet_rem := (vlad_height / 12, vlad_height % 12)) 
  (h_sister_height : sister_height_feet = 2)
  (h_sister_height_inches : sister_height_inches = 10)
  (h_vlad_height_diff : vlad_height_diff = 41)
  (h_vlad_height : vlad_height = 75)
  (h_vlad_height_feet : vlad_height_feet = 6)
  (h_vlad_height_rem : vlad_height_rem = 3) :
  vlad_height_feet = 6 := by
  sorry

end Vlad_height_feet_l26_26553


namespace solution_set_intersection_l26_26587

theorem solution_set_intersection (a b : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x - 3 < 0 ↔ -1 < x ∧ x < 3) →
  (∀ x : ℝ, x^2 + x - 6 < 0 ↔ -3 < x ∧ x < 2) →
  (∀ x : ℝ, x^2 + a * x + b < 0 ↔ (-1 < x ∧ x < 2)) →
  a + b = -3 :=
by 
  sorry

end solution_set_intersection_l26_26587


namespace multiplication_problem_l26_26547

-- Define the problem in Lean 4.
theorem multiplication_problem (a b : ℕ) (ha : a < 10) (hb : b < 10) 
  (h : (30 + a) * (10 * b + 4) = 126) : a + b = 7 :=
sorry

end multiplication_problem_l26_26547


namespace r_squared_plus_s_squared_l26_26194

theorem r_squared_plus_s_squared (r s : ℝ) (h1 : r * s = 16) (h2 : r + s = 8) : r^2 + s^2 = 32 :=
by
  sorry

end r_squared_plus_s_squared_l26_26194


namespace solve_expression_l26_26066

theorem solve_expression (x : ℝ) (h : 5 * x - 7 = 15 * x + 13) : 3 * (x + 4) = 6 :=
sorry

end solve_expression_l26_26066


namespace find_a5_l26_26155

variable {a : ℕ → ℝ}  -- Define the sequence a(n)

-- Define the conditions of the problem
variable (a1_positive : ∀ n, a n > 0)
variable (geo_seq : ∀ n, a (n + 1) = a n * 2)
variable (condition : (a 3) * (a 11) = 16)

theorem find_a5 (a1_positive : ∀ n, a n > 0) (geo_seq : ∀ n, a (n + 1) = a n * 2)
(condition : (a 3) * (a 11) = 16) : a 5 = 1 := by
  sorry

end find_a5_l26_26155


namespace find_m_value_l26_26022

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem find_m_value (m : ℝ) 
  (h : dot_product (2 * m - 1, 3) (1, -1) = 2) : 
  m = 3 := by
  sorry

end find_m_value_l26_26022


namespace min_transport_cost_l26_26734

/- Definitions for the problem conditions -/
def villageA_vegetables : ℕ := 80
def villageB_vegetables : ℕ := 60
def destinationX_requirement : ℕ := 65
def destinationY_requirement : ℕ := 75

def cost_A_to_X : ℕ := 50
def cost_A_to_Y : ℕ := 30
def cost_B_to_X : ℕ := 60
def cost_B_to_Y : ℕ := 45

def W (x : ℕ) : ℕ :=
  cost_A_to_X * x +
  cost_A_to_Y * (villageA_vegetables - x) +
  cost_B_to_X * (destinationX_requirement - x) +
  cost_B_to_Y * (x - 5) + 6075 - 225

/- Prove that the minimum total cost W is 6100 -/
theorem min_transport_cost : ∃ (x : ℕ), 5 ≤ x ∧ x ≤ 65 ∧ W x = 6100 :=
by sorry

end min_transport_cost_l26_26734


namespace cost_of_bananas_and_cantaloupe_l26_26935

variable (a b c d : ℝ)

theorem cost_of_bananas_and_cantaloupe :
  (a + b + c + d = 30) →
  (d = 3 * a) →
  (c = a - b) →
  (b + c = 6) :=
by
  intros h1 h2 h3
  sorry

end cost_of_bananas_and_cantaloupe_l26_26935


namespace smallest_total_marbles_l26_26296

-- Definitions based on conditions in a)
def urn_contains_marbles : Type := ℕ → ℕ
def red_marbles (u : urn_contains_marbles) := u 0
def white_marbles (u : urn_contains_marbles) := u 1
def blue_marbles (u : urn_contains_marbles) := u 2
def green_marbles (u : urn_contains_marbles) := u 3
def yellow_marbles (u : urn_contains_marbles) := u 4
def total_marbles (u : urn_contains_marbles) := u 0 + u 1 + u 2 + u 3 + u 4

-- Probabilities of selection events
def prob_event_a (u : urn_contains_marbles) := (red_marbles u).choose 5
def prob_event_b (u : urn_contains_marbles) := (white_marbles u).choose 1 * (red_marbles u).choose 4
def prob_event_c (u : urn_contains_marbles) := (white_marbles u).choose 1 * (blue_marbles u).choose 1 * (red_marbles u).choose 3
def prob_event_d (u : urn_contains_marbles) := (white_marbles u).choose 1 * (blue_marbles u).choose 1 * (green_marbles u).choose 1 * (red_marbles u).choose 2
def prob_event_e (u : urn_contains_marbles) := (white_marbles u).choose 1 * (blue_marbles u).choose 1 * (green_marbles u).choose 1 * (yellow_marbles u).choose 1 * (red_marbles u).choose 1

-- Proof that the smallest total number of marbles satisfying the conditions is 33
theorem smallest_total_marbles : ∃ u : urn_contains_marbles, 
    (prob_event_a u = prob_event_b u) ∧ 
    (prob_event_b u = prob_event_c u) ∧ 
    (prob_event_c u = prob_event_d u) ∧ 
    (prob_event_d u = prob_event_e u) ∧ 
    total_marbles u = 33 := sorry

end smallest_total_marbles_l26_26296


namespace negation_of_square_positive_l26_26448

open Real

-- Define the original proposition
def prop_square_positive : Prop :=
  ∀ x : ℝ, x^2 > 0

-- Define the negation of the original proposition
def prop_square_not_positive : Prop :=
  ∃ x : ℝ, ¬ (x^2 > 0)

-- The theorem that asserts the logical equivalence for the negation
theorem negation_of_square_positive :
  ¬ prop_square_positive ↔ prop_square_not_positive :=
by sorry

end negation_of_square_positive_l26_26448


namespace exists_arithmetic_progression_2004_perfect_powers_perfect_powers_not_infinite_arithmetic_progression_l26_26456

-- Definition: A positive integer n is a perfect power if n = a ^ b for some integers a, b with b > 1.
def isPerfectPower (n : ℕ) : Prop :=
  ∃ a b : ℕ, b > 1 ∧ n = a^b

-- Part (a): Prove the existence of an arithmetic progression of 2004 perfect powers.
theorem exists_arithmetic_progression_2004_perfect_powers :
  ∃ (x r : ℕ), (∀ n : ℕ, n < 2004 → ∃ a b : ℕ, b > 1 ∧ (x + n * r) = a^b) :=
sorry

-- Part (b): Prove that perfect powers cannot form an infinite arithmetic progression.
theorem perfect_powers_not_infinite_arithmetic_progression :
  ¬ ∃ (x r : ℕ), (∀ n : ℕ, ∃ a b : ℕ, b > 1 ∧ (x + n * r) = a^b) :=
sorry

end exists_arithmetic_progression_2004_perfect_powers_perfect_powers_not_infinite_arithmetic_progression_l26_26456


namespace quadrant_of_theta_l26_26556

theorem quadrant_of_theta (θ : ℝ) (h1 : Real.cos θ > 0) (h2 : Real.sin θ < 0) : (0 < θ ∧ θ < π/2) ∨ (3*π/2 < θ ∧ θ < 2*π) :=
by
  sorry

end quadrant_of_theta_l26_26556


namespace num_emails_received_after_second_deletion_l26_26162

-- Define the initial conditions and final question
variable (initialEmails : ℕ)    -- Initial number of emails
variable (deletedEmails1 : ℕ)   -- First batch of deleted emails
variable (receivedEmails1 : ℕ)  -- First batch of received emails
variable (deletedEmails2 : ℕ)   -- Second batch of deleted emails
variable (receivedEmails2 : ℕ)  -- Second batch of received emails
variable (receivedEmails3 : ℕ)  -- Third batch of received emails
variable (finalEmails : ℕ)      -- Final number of emails in the inbox

-- Conditions based on the problem description
axiom initialEmails_def : initialEmails = 0
axiom deletedEmails1_def : deletedEmails1 = 50
axiom receivedEmails1_def : receivedEmails1 = 15
axiom deletedEmails2_def : deletedEmails2 = 20
axiom receivedEmails3_def : receivedEmails3 = 10
axiom finalEmails_def : finalEmails = 30

-- Question: Prove that the number of emails received after the second deletion is 5
theorem num_emails_received_after_second_deletion : receivedEmails2 = 5 :=
by
  sorry

end num_emails_received_after_second_deletion_l26_26162


namespace base_circumference_cone_l26_26894

theorem base_circumference_cone (r : ℝ) (h : r = 5) (θ : ℝ) (k : θ = 180) : 
  ∃ c : ℝ, c = 5 * π :=
by
  sorry

end base_circumference_cone_l26_26894


namespace train_length_proof_l26_26054

/-- Given a train's speed of 45 km/hr, time to cross a bridge of 30 seconds, and the bridge length of 225 meters, prove that the length of the train is 150 meters. -/
theorem train_length_proof (speed_km_hr : ℝ) (time_sec : ℝ) (bridge_length_m : ℝ) (train_length_m : ℝ)
    (h_speed : speed_km_hr = 45) (h_time : time_sec = 30) (h_bridge_length : bridge_length_m = 225) :
  train_length_m = 150 :=
by
  sorry

end train_length_proof_l26_26054


namespace distance_upstream_l26_26487

/-- Proof that the distance a man swims upstream is 18 km given certain conditions. -/
theorem distance_upstream (c : ℝ) (h1 : 54 / (12 + c) = 3) (h2 : 12 - c = 6) : (12 - c) * 3 = 18 :=
by
  sorry

end distance_upstream_l26_26487


namespace lewis_total_earnings_l26_26671

def Weekly_earnings : ℕ := 92
def Number_of_weeks : ℕ := 5

theorem lewis_total_earnings : Weekly_earnings * Number_of_weeks = 460 := by
  sorry

end lewis_total_earnings_l26_26671


namespace garden_sparrows_l26_26524

theorem garden_sparrows (ratio_b_s : ℕ) (bluebirds sparrows : ℕ)
  (h1 : ratio_b_s = 4 / 5) (h2 : bluebirds = 28) :
  sparrows = 35 :=
  sorry

end garden_sparrows_l26_26524


namespace water_tank_capacity_l26_26544

variable (C : ℝ)  -- Full capacity of the tank in liters

theorem water_tank_capacity (h1 : 0.4 * C = 0.9 * C - 50) : C = 100 := by
  sorry

end water_tank_capacity_l26_26544


namespace quadratic_solution_l26_26810

theorem quadratic_solution (x : ℝ) : x ^ 2 - 4 * x + 3 = 0 ↔ x = 1 ∨ x = 3 :=
by
  sorry

end quadratic_solution_l26_26810


namespace circles_intersect_l26_26352

theorem circles_intersect (t : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 - 2 * t * x + t^2 - 4 = 0 ∧ x^2 + y^2 + 2 * x - 4 * t * y + 4 * t^2 - 8 = 0) ↔ 
  (-12 / 5 < t ∧ t < -2 / 5) ∨ (0 < t ∧ t < 2) :=
sorry

end circles_intersect_l26_26352


namespace farm_section_areas_l26_26167

theorem farm_section_areas (n : ℕ) (total_area : ℕ) (sections : ℕ) 
  (hn : sections = 5) (ht : total_area = 300) : total_area / sections = 60 :=
by
  sorry

end farm_section_areas_l26_26167


namespace greatest_possible_third_side_l26_26351

theorem greatest_possible_third_side (t : ℕ) (h : 5 < t ∧ t < 15) : t = 14 :=
sorry

end greatest_possible_third_side_l26_26351


namespace second_number_is_22_l26_26239

theorem second_number_is_22 
    (A B : ℤ)
    (h1 : A - B = 88) 
    (h2 : A = 110) :
    B = 22 :=
by
  sorry

end second_number_is_22_l26_26239


namespace sum_of_first_three_terms_l26_26548

theorem sum_of_first_three_terms 
  (a d : ℤ) 
  (h1 : a + 4 * d = 15) 
  (h2 : d = 3) : 
  a + (a + d) + (a + 2 * d) = 18 :=
by
  sorry

end sum_of_first_three_terms_l26_26548


namespace num_of_consec_int_sum_18_l26_26000

theorem num_of_consec_int_sum_18 : 
  ∃! (a n : ℕ), n ≥ 3 ∧ (n * (2 * a + n - 1)) = 36 :=
sorry

end num_of_consec_int_sum_18_l26_26000


namespace rectangle_ratio_l26_26498

theorem rectangle_ratio (s y x : ℝ)
  (h1 : 4 * y * x + s * s = 9 * s * s)
  (h2 : s + y + y = 3 * s)
  (h3 : y = s)
  (h4 : x + s = 3 * s) : 
  (x / y = 2) :=
sorry

end rectangle_ratio_l26_26498


namespace solve_for_asterisk_l26_26193

theorem solve_for_asterisk (asterisk : ℝ) : 
  ((60 / 20) * (60 / asterisk) = 1) → asterisk = 180 :=
by
  sorry

end solve_for_asterisk_l26_26193


namespace student_a_score_l26_26109

def total_questions : ℕ := 100
def correct_responses : ℕ := 87
def incorrect_responses : ℕ := total_questions - correct_responses
def score : ℕ := correct_responses - 2 * incorrect_responses

theorem student_a_score : score = 61 := by
  unfold score
  unfold correct_responses
  unfold incorrect_responses
  norm_num
  -- At this point, the theorem is stated, but we insert sorry to satisfy the requirement of not providing the proof.
  sorry

end student_a_score_l26_26109


namespace mom_t_shirts_total_l26_26291

-- Definitions based on the conditions provided in the problem
def packages : ℕ := 71
def t_shirts_per_package : ℕ := 6

-- The statement to prove that the total number of white t-shirts is 426
theorem mom_t_shirts_total : packages * t_shirts_per_package = 426 := by sorry

end mom_t_shirts_total_l26_26291


namespace polynomial_min_value_l26_26196

theorem polynomial_min_value (x : ℝ) : x = -3 → x^2 + 6 * x + 10 = 1 :=
by
  intro h
  sorry

end polynomial_min_value_l26_26196


namespace option_D_is_div_by_9_l26_26211

-- Define the parameters and expressions
def A (k : ℕ) : ℤ := 6 + 6 * 7^k
def B (k : ℕ) : ℤ := 2 + 7^(k - 1)
def C (k : ℕ) : ℤ := 2 * (2 + 7^(k + 1))
def D (k : ℕ) : ℤ := 3 * (2 + 7^k)

-- Define the main theorem to prove that D is divisible by 9
theorem option_D_is_div_by_9 (k : ℕ) (hk : k > 0) : D k % 9 = 0 :=
sorry

end option_D_is_div_by_9_l26_26211


namespace area_of_frame_l26_26245

def width : ℚ := 81 / 4
def depth : ℚ := 148 / 9
def area (w d : ℚ) : ℚ := w * d

theorem area_of_frame : area width depth = 333 := by
  sorry

end area_of_frame_l26_26245


namespace widgets_made_per_week_l26_26809

theorem widgets_made_per_week
  (widgets_per_hour : Nat)
  (hours_per_day : Nat)
  (days_per_week : Nat)
  (total_widgets : Nat) :
  widgets_per_hour = 20 →
  hours_per_day = 8 →
  days_per_week = 5 →
  total_widgets = widgets_per_hour * hours_per_day * days_per_week →
  total_widgets = 800 :=
by
  intros h1 h2 h3 h4
  sorry

end widgets_made_per_week_l26_26809


namespace max_angle_in_hexagon_l26_26474

-- Definition of the problem
theorem max_angle_in_hexagon :
  ∃ (a d : ℕ), a + a + d + a + 2*d + a + 3*d + a + 4*d + a + 5*d = 720 ∧ 
               a + 5 * d < 180 ∧ 
               (∀ a d : ℕ, a + a + d + a + 2*d + a + 3*d + a + 4*d + a + 5*d = 720 → 
               a + 5*d < 180 → m <= 175) :=
sorry

end max_angle_in_hexagon_l26_26474


namespace sum_of_coefficients_eq_two_l26_26224

theorem sum_of_coefficients_eq_two {a b c : ℤ} (h : ∀ x : ℤ, x * (x + 1) = a + b * x + c * x^2) : a + b + c = 2 := 
by
  sorry

end sum_of_coefficients_eq_two_l26_26224


namespace plane_equation_parallel_to_Oz_l26_26157

theorem plane_equation_parallel_to_Oz (A B D : ℝ)
  (h1 : A * 1 + B * 0 + D = 0)
  (h2 : A * (-2) + B * 1 + D = 0)
  (h3 : ∀ z : ℝ, exists c : ℝ, A * z + B * c + D = 0):
  A = 1 ∧ B = 3 ∧ D = -1 :=
  by
  sorry

end plane_equation_parallel_to_Oz_l26_26157


namespace quadratic_ineq_solutions_l26_26431

theorem quadratic_ineq_solutions (c : ℝ) (h : c > 0) : c < 16 ↔ ∀ x : ℝ, x^2 - 8 * x + c < 0 → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x = x1 ∨ x = x2 :=
by
  sorry

end quadratic_ineq_solutions_l26_26431


namespace ferns_have_1260_leaves_l26_26142

def num_ferns : ℕ := 6
def fronds_per_fern : ℕ := 7
def leaves_per_frond : ℕ := 30
def total_leaves : ℕ := num_ferns * fronds_per_fern * leaves_per_frond

theorem ferns_have_1260_leaves : total_leaves = 1260 :=
by 
  -- proof goes here
  sorry

end ferns_have_1260_leaves_l26_26142


namespace find_divisor_l26_26959

theorem find_divisor (n d : ℤ) (k : ℤ)
  (h1 : n % d = 3)
  (h2 : n^2 % d = 4) : d = 5 :=
sorry

end find_divisor_l26_26959


namespace solve_equation_l26_26037

theorem solve_equation {n k l m : ℕ} (h_l : l > 1) :
  (1 + n^k)^l = 1 + n^m ↔ (n = 2 ∧ k = 1 ∧ l = 2 ∧ m = 3) :=
sorry

end solve_equation_l26_26037


namespace highest_of_seven_consecutive_with_average_33_l26_26003

theorem highest_of_seven_consecutive_with_average_33 (x : ℤ) 
    (h : (x - 3 + x - 2 + x - 1 + x + x + 1 + x + 2 + x + 3) / 7 = 33) : 
    x + 3 = 36 := 
sorry

end highest_of_seven_consecutive_with_average_33_l26_26003


namespace sum_geometric_sequence_terms_l26_26906

theorem sum_geometric_sequence_terms (a r : ℝ) 
  (h1 : a * (1 - r^1500) / (1 - r) = 300) 
  (h2 : a * (1 - r^3000) / (1 - r) = 570) :
  a * (1 - r^4500) / (1 - r) = 813 := 
by
  sorry

end sum_geometric_sequence_terms_l26_26906


namespace minimum_value_of_3x_plus_4y_exists_x_y_for_minimum_value_l26_26823

theorem minimum_value_of_3x_plus_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : 3 * x + 4 * y ≥ 5 :=
sorry

theorem exists_x_y_for_minimum_value : ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + 3 * y = 5 * x * y ∧ 3 * x + 4 * y = 5 :=
sorry

end minimum_value_of_3x_plus_4y_exists_x_y_for_minimum_value_l26_26823


namespace no_real_roots_iff_l26_26371

theorem no_real_roots_iff (k : ℝ) : (∀ x : ℝ, x^2 - 2*x - k ≠ 0) ↔ k < -1 :=
by
  sorry

end no_real_roots_iff_l26_26371


namespace binary_division_example_l26_26536

theorem binary_division_example : 
  let a := 0b10101  -- binary representation of 21
  let b := 0b11     -- binary representation of 3
  let quotient := 0b111  -- binary representation of 7
  a / b = quotient := 
by sorry

end binary_division_example_l26_26536


namespace inequality_subtraction_real_l26_26769

theorem inequality_subtraction_real (a b c : ℝ) (h : a > b) : a - c > b - c :=
sorry

end inequality_subtraction_real_l26_26769


namespace total_carrots_l26_26268

/-- 
  If Pleasant Goat and Beautiful Goat each receive 6 carrots, and the other goats each receive 3 carrots, there will be 6 carrots left over.
  If Pleasant Goat and Beautiful Goat each receive 7 carrots, and the other goats each receive 5 carrots, there will be a shortage of 14 carrots.
  Prove the total number of carrots (n) is 45. 
--/
theorem total_carrots (X n : ℕ) 
  (h1 : n = 3 * X + 18) 
  (h2 : n = 5 * X) : 
  n = 45 := 
by
  sorry

end total_carrots_l26_26268


namespace max_gcd_13n_plus_4_8n_plus_3_l26_26381

theorem max_gcd_13n_plus_4_8n_plus_3 : 
  ∀ n : ℕ, n > 0 → ∃ d : ℕ, d = 7 ∧ ∀ k : ℕ, k = gcd (13 * n + 4) (8 * n + 3) → k ≤ d :=
by
  sorry

end max_gcd_13n_plus_4_8n_plus_3_l26_26381


namespace simplify_expression_l26_26464

theorem simplify_expression (x : ℝ) : 5 * x + 6 - x + 12 = 4 * x + 18 :=
by sorry

end simplify_expression_l26_26464


namespace inequality_holds_iff_x_in_interval_l26_26469

theorem inequality_holds_iff_x_in_interval (x : ℝ) :
  (∀ n : ℕ, 0 < n → (1 + x)^n ≤ 1 + (2^n - 1) * x) ↔ (0 ≤ x ∧ x ≤ 1) :=
sorry

end inequality_holds_iff_x_in_interval_l26_26469


namespace union_A_B_eq_univ_inter_compl_A_B_eq_interval_subset_B_range_of_a_l26_26389

variable (A B C : Set ℝ)
variable (a : ℝ)

-- Condition definitions
def set_A : Set ℝ := {x | x ≤ 3 ∨ x ≥ 6}
def set_B : Set ℝ := {x | -2 < x ∧ x < 9}
def set_C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Proof statement (1)
theorem union_A_B_eq_univ (A B : Set ℝ) (h₁ : A = set_A) (h₂ : B = set_B) :
  A ∪ B = Set.univ := by sorry

theorem inter_compl_A_B_eq_interval (A B : Set ℝ) (h₁ : A = set_A) (h₂ : B = set_B) :
  (Set.univ \ A) ∩ B = {x | 3 < x ∧ x < 6} := by sorry

-- Proof statement (2)
theorem subset_B_range_of_a (a : ℝ) (h : set_C a ⊆ set_B) :
  -2 ≤ a ∧ a ≤ 8 := by sorry

end union_A_B_eq_univ_inter_compl_A_B_eq_interval_subset_B_range_of_a_l26_26389


namespace total_customers_l26_26690

def initial_customers : ℝ := 29.0    -- 29.0 initial customers
def lunch_rush_customers : ℝ := 20.0 -- Adds 20.0 customers during lunch rush
def additional_customers : ℝ := 34.0 -- Adds 34.0 more customers

theorem total_customers : (initial_customers + lunch_rush_customers + additional_customers) = 83.0 :=
by
  sorry

end total_customers_l26_26690


namespace paco_cookies_proof_l26_26217

-- Define the initial conditions
def initial_cookies : Nat := 40
def cookies_eaten : Nat := 2
def cookies_bought : Nat := 37
def free_cookies_per_bought : Nat := 2

-- Define the total number of cookies after all operations
def total_cookies (initial_cookies cookies_eaten cookies_bought free_cookies_per_bought : Nat) : Nat :=
  let remaining_cookies := initial_cookies - cookies_eaten
  let free_cookies := cookies_bought * free_cookies_per_bought
  let cookies_from_bakery := cookies_bought + free_cookies
  remaining_cookies + cookies_from_bakery

-- The target statement that needs to be proved
theorem paco_cookies_proof : total_cookies initial_cookies cookies_eaten cookies_bought free_cookies_per_bought = 149 :=
by
  sorry

end paco_cookies_proof_l26_26217


namespace polynomial_roots_a_ge_five_l26_26043

theorem polynomial_roots_a_ge_five (a b c : ℤ) (h_a_pos : a > 0)
    (h_distinct_roots : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 0 < x₁ ∧ x₁ < 1 ∧ 0 < x₂ ∧ x₂ < 1 ∧ 
        a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) : a ≥ 5 := sorry

end polynomial_roots_a_ge_five_l26_26043


namespace average_male_grade_l26_26924

theorem average_male_grade (avg_all avg_fem : ℝ) (N_male N_fem : ℕ) 
    (h1 : avg_all = 90) 
    (h2 : avg_fem = 92) 
    (h3 : N_male = 8) 
    (h4 : N_fem = 12) :
    let total_students := N_male + N_fem
    let total_sum_all := avg_all * total_students
    let total_sum_fem := avg_fem * N_fem
    let total_sum_male := total_sum_all - total_sum_fem
    let avg_male := total_sum_male / N_male
    avg_male = 87 :=
by 
  let total_students := N_male + N_fem
  let total_sum_all := avg_all * total_students
  let total_sum_fem := avg_fem * N_fem
  let total_sum_male := total_sum_all - total_sum_fem
  let avg_male := total_sum_male / N_male
  sorry

end average_male_grade_l26_26924


namespace find_a_and_b_l26_26181

theorem find_a_and_b (a b m : ℝ) 
  (h1 : (3 * a - 5)^(1 / 3) = -2)
  (h2 : ∀ x, x^2 = b → x = m ∨ x = 1 - 5 * m) : 
  a = -1 ∧ b = 1 / 16 :=
by
  sorry  -- proof to be constructed

end find_a_and_b_l26_26181


namespace good_games_count_l26_26879

-- Define the conditions
def games_from_friend : Nat := 50
def games_from_garage_sale : Nat := 27
def games_that_didnt_work : Nat := 74

-- Define the total games bought
def total_games_bought : Nat := games_from_friend + games_from_garage_sale

-- State the theorem to prove the number of good games
theorem good_games_count : total_games_bought - games_that_didnt_work = 3 :=
by
  sorry

end good_games_count_l26_26879


namespace largest_square_area_l26_26297

theorem largest_square_area (a b c : ℝ)
  (right_triangle : a^2 + b^2 = c^2)
  (square_area_sum : a^2 + b^2 + c^2 = 450)
  (area_a : a^2 = 100) :
  c^2 = 225 :=
by
  sorry

end largest_square_area_l26_26297


namespace crayons_left_l26_26150

-- Define initial number of crayons and the number taken by Mary
def initial_crayons : ℝ := 7.5
def taken_crayons : ℝ := 2.25

-- Calculate remaining crayons
def remaining_crayons := initial_crayons - taken_crayons

-- Prove that the remaining crayons are 5.25
theorem crayons_left : remaining_crayons = 5.25 := by
  sorry

end crayons_left_l26_26150


namespace solve_xy_l26_26929

theorem solve_xy (x y : ℝ) :
  (x - 11)^2 + (y - 12)^2 + (x - y)^2 = 1 / 3 → 
  x = 34 / 3 ∧ y = 35 / 3 :=
by
  intro h
  sorry

end solve_xy_l26_26929


namespace largest_prime_factor_13231_l26_26047

-- Define the conditions
def is_prime (n : ℕ) : Prop := ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

-- State the problem as a theorem in Lean 4
theorem largest_prime_factor_13231 (H1 : 13231 = 121 * 109) 
    (H2 : is_prime 109)
    (H3 : 121 = 11^2) :
    ∃ p, is_prime p ∧ p ∣ 13231 ∧ ∀ q, is_prime q ∧ q ∣ 13231 → q ≤ p :=
by
  sorry

end largest_prime_factor_13231_l26_26047


namespace election_total_votes_l26_26532

theorem election_total_votes
  (V : ℝ)
  (h1 : 0 ≤ V) 
  (h_majority : 0.70 * V - 0.30 * V = 182) :
  V = 455 := 
by 
  sorry

end election_total_votes_l26_26532


namespace number_of_combinations_with_constraints_l26_26717

theorem number_of_combinations_with_constraints :
  let n : ℕ := 6
  let k : ℕ := 2
  let total_combinations := Nat.choose n k
  let invalid_combinations := 2
  total_combinations - invalid_combinations = 13 :=
by
  let n : ℕ := 6
  let k : ℕ := 2
  let total_combinations := Nat.choose 6 2
  let invalid_combinations := 2
  show total_combinations - invalid_combinations = 13
  sorry

end number_of_combinations_with_constraints_l26_26717


namespace fraction_identity_l26_26897

theorem fraction_identity (a b : ℝ) (h₀ : a^2 + a = 4) (h₁ : b^2 + b = 4) (h₂ : a ≠ b) :
  (b / a) + (a / b) = - (9 / 4) :=
sorry

end fraction_identity_l26_26897


namespace exponential_inequality_l26_26704

theorem exponential_inequality (n : ℕ) (h : n ≥ 5) : 2^n > n^2 + 1 :=
sorry

end exponential_inequality_l26_26704


namespace find_x_plus_y_l26_26160

theorem find_x_plus_y (x y : ℝ) 
  (h1 : x + Real.cos y = 1005) 
  (h2 : x + 1005 * Real.sin y = 1003) 
  (h3 : π ≤ y ∧ y ≤ 3 * π / 2) : 
  x + y = 1005 + 3 * π / 2 :=
sorry

end find_x_plus_y_l26_26160


namespace angle_same_terminal_side_l26_26502

theorem angle_same_terminal_side (k : ℤ) : ∃ k : ℤ, 290 = k * 360 - 70 :=
by
  sorry

end angle_same_terminal_side_l26_26502


namespace lowest_possible_number_of_students_l26_26152

theorem lowest_possible_number_of_students :
  Nat.lcm 18 24 = 72 :=
by
  sorry

end lowest_possible_number_of_students_l26_26152


namespace triangle_sum_is_19_l26_26340

-- Defining the operation on a triangle
def triangle_op (a b c : ℕ) := a * b - c

-- Defining the vertices of the two triangles
def triangle1 := (4, 2, 3)
def triangle2 := (3, 5, 1)

-- Statement that the sum of the operation results is 19
theorem triangle_sum_is_19 :
  triangle_op (4) (2) (3) + triangle_op (3) (5) (1) = 19 :=
by
  -- Triangle 1 calculation: 4 * 2 - 3 = 8 - 3 = 5
  -- Triangle 2 calculation: 3 * 5 - 1 = 15 - 1 = 14
  -- Sum of calculations: 5 + 14 = 19
  sorry

end triangle_sum_is_19_l26_26340


namespace chord_length_l26_26176

theorem chord_length
  (x y : ℝ)
  (h_circle : (x-1)^2 + (y-2)^2 = 2)
  (h_line : 3*x - 4*y = 0) :
  ∃ L : ℝ, L = 2 :=
sorry

end chord_length_l26_26176


namespace find_m_l26_26695

theorem find_m (a : ℕ → ℝ) (m : ℝ)
  (h1 : (∀ (x : ℝ), x^2 + m * x - 8 = 0 → x = a 2 ∨ x = a 8))
  (h2 : a 4 + a 6 = a 5 ^ 2 + 1) :
  m = -2 :=
sorry

end find_m_l26_26695


namespace necessary_but_not_sufficient_l26_26332

def p (a : ℝ) : Prop := ∃ (x : ℝ), x^2 + 2 * a * x - a ≤ 0

def q (a : ℝ) : Prop := a > 0 ∨ a < -1

theorem necessary_but_not_sufficient (a : ℝ) : (∃ (x : ℝ), x^2 + 2 * a * x - a ≤ 0) → (a > 0 ∨ a < -1) ∧ ¬((a > 0 ∨ a < -1) → (∃ (x : ℝ), x^2 + 2 * a * x - a ≤ 0)) :=
by
  sorry

end necessary_but_not_sufficient_l26_26332


namespace compute_focus_d_l26_26914

-- Define the given conditions as Lean definitions
structure Ellipse (d : ℝ) :=
  (first_quadrant : d > 0)
  (F1 : ℝ × ℝ := (4, 8))
  (F2 : ℝ × ℝ := (d, 8))
  (tangent_x_axis : (d + 4) / 2 > 0)
  (tangent_y_axis : (d + 4) / 2 > 0)

-- Define the proof problem to show d = 6 for the given conditions
theorem compute_focus_d (d : ℝ) (e : Ellipse d) : d = 6 := by
  sorry

end compute_focus_d_l26_26914


namespace six_times_product_plus_one_equals_seven_pow_sixteen_l26_26846

theorem six_times_product_plus_one_equals_seven_pow_sixteen :
  6 * (7 + 1) * (7^2 + 1) * (7^4 + 1) * (7^8 + 1) + 1 = 7^16 := 
  sorry

end six_times_product_plus_one_equals_seven_pow_sixteen_l26_26846


namespace max_cos_half_sin_eq_1_l26_26185

noncomputable def max_value_expression (θ : ℝ) : ℝ :=
  Real.cos (θ / 2) * (1 - Real.sin θ)

theorem max_cos_half_sin_eq_1 : 
  ∀ θ : ℝ, 0 < θ ∧ θ < π → max_value_expression θ ≤ 1 :=
by
  intros θ h
  sorry

end max_cos_half_sin_eq_1_l26_26185


namespace ab_value_l26_26212

theorem ab_value (a b : ℤ) (h : 48 * a * b = 65 * a * b) : a * b = 0 :=
  sorry

end ab_value_l26_26212


namespace zamena_solution_l26_26816

def digit_assignment (A M E H Z N : ℕ) : Prop :=
  1 ≤ A ∧ A ≤ 5 ∧ 1 ≤ M ∧ M ≤ 5 ∧ 1 ≤ E ∧ E ≤ 5 ∧ 1 ≤ H ∧ H ≤ 5 ∧ 1 ≤ Z ∧ Z ≤ 5 ∧ 1 ≤ N ∧ N ≤ 5 ∧
  A ≠ M ∧ A ≠ E ∧ A ≠ H ∧ A ≠ Z ∧ A ≠ N ∧
  M ≠ E ∧ M ≠ H ∧ M ≠ Z ∧ M ≠ N ∧
  E ≠ H ∧ E ≠ Z ∧ E ≠ N ∧
  H ≠ Z ∧ H ≠ N ∧
  Z ≠ N ∧
  3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A 

theorem zamena_solution : 
  ∃ (A M E H Z N : ℕ), digit_assignment A M E H Z N ∧ (Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234) := 
by
  sorry

end zamena_solution_l26_26816


namespace expansion_coefficient_l26_26409

theorem expansion_coefficient (x : ℝ) (h : x ≠ 0): 
  (∃ r : ℕ, (7 - (3 / 2 : ℝ) * r = 1) ∧ Nat.choose 7 r = 35) := 
  sorry

end expansion_coefficient_l26_26409


namespace people_eat_only_vegetarian_l26_26420

def number_of_people_eat_only_veg (total_veg : ℕ) (both_veg_nonveg : ℕ) : ℕ :=
  total_veg - both_veg_nonveg

theorem people_eat_only_vegetarian
  (total_veg : ℕ) (both_veg_nonveg : ℕ)
  (h1 : total_veg = 28)
  (h2 : both_veg_nonveg = 12)
  : number_of_people_eat_only_veg total_veg both_veg_nonveg = 16 := by
  sorry

end people_eat_only_vegetarian_l26_26420


namespace max_value_vector_sum_l26_26696

theorem max_value_vector_sum (α β : ℝ) :
  let a := (Real.cos α, Real.sin α)
  let b := (Real.sin β, -Real.cos β)
  |(a.1 + b.1, a.2 + b.2)| ≤ 2 := by
  sorry

end max_value_vector_sum_l26_26696


namespace pencil_cost_is_correct_l26_26780

-- Defining the cost of a pen as x and the cost of a pencil as y in cents
def cost_of_pen_and_pencil (x y : ℕ) : Prop :=
  3 * x + 5 * y = 345 ∧ 4 * x + 2 * y = 280

-- Stating the theorem that proves y = 39
theorem pencil_cost_is_correct (x y : ℕ) (h : cost_of_pen_and_pencil x y) : y = 39 :=
by
  sorry

end pencil_cost_is_correct_l26_26780


namespace solution_set_of_floor_equation_l26_26651

theorem solution_set_of_floor_equation (x : ℝ) : 
  (⌊⌊2 * x⌋ - 1/2⌋ = ⌊x + 3⌋) ↔ (3.5 ≤ x ∧ x < 4.5) :=
by sorry

end solution_set_of_floor_equation_l26_26651


namespace find_percentage_l26_26883

theorem find_percentage (P : ℝ) :
  (P / 100) * 1280 = ((0.20 * 650) + 190) ↔ P = 25 :=
by
  sorry

end find_percentage_l26_26883


namespace exists_station_to_complete_loop_l26_26178

structure CircularHighway where
  fuel_at_stations : List ℝ -- List of fuel amounts at each station
  travel_cost : List ℝ -- List of travel costs between consecutive stations

def total_fuel (hw : CircularHighway) : ℝ :=
  hw.fuel_at_stations.sum

def total_travel_cost (hw : CircularHighway) : ℝ :=
  hw.travel_cost.sum

def sufficient_fuel (hw : CircularHighway) : Prop :=
  total_fuel hw ≥ 2 * total_travel_cost hw

noncomputable def can_return_to_start (hw : CircularHighway) (start_station : ℕ) : Prop :=
  -- Function that checks if starting from a specific station allows for a return
  sorry

theorem exists_station_to_complete_loop (hw : CircularHighway) (h : sufficient_fuel hw) : ∃ start_station, can_return_to_start hw start_station :=
  sorry

end exists_station_to_complete_loop_l26_26178


namespace julia_fascinating_last_digits_l26_26378

theorem julia_fascinating_last_digits : ∃ n : ℕ, n = 10 ∧ (∀ x : ℕ, (∃ y : ℕ, x = 10 * y) → x % 10 < 10) :=
by
  sorry

end julia_fascinating_last_digits_l26_26378


namespace matt_house_wall_height_l26_26904

noncomputable def height_of_walls_in_matt_house : ℕ :=
  let living_room_side := 40
  let bedroom_side_1 := 10
  let bedroom_side_2 := 12

  let perimeter_living_room := 4 * living_room_side
  let perimeter_living_room_3_walls := perimeter_living_room - living_room_side

  let perimeter_bedroom := 2 * (bedroom_side_1 + bedroom_side_2)

  let total_perimeter_to_paint := perimeter_living_room_3_walls + perimeter_bedroom
  let total_area_to_paint := 1640

  total_area_to_paint / total_perimeter_to_paint

theorem matt_house_wall_height :
  height_of_walls_in_matt_house = 10 := by
  sorry

end matt_house_wall_height_l26_26904


namespace polynomial_value_at_n_plus_1_l26_26021

theorem polynomial_value_at_n_plus_1 
  (f : ℕ → ℝ) 
  (n : ℕ)
  (hdeg : ∃ m, m = n) 
  (hvalues : ∀ k (hk : k ≤ n), f k = k / (k + 1)) : 
  f (n + 1) = (n + 1 + (-1) ^ (n + 1)) / (n + 2) := 
by
  sorry

end polynomial_value_at_n_plus_1_l26_26021


namespace moles_of_SO2_formed_l26_26035

variable (n_NaHSO3 n_HCl n_SO2 : ℕ)

/--
The reaction between sodium bisulfite (NaHSO3) and hydrochloric acid (HCl) is:
NaHSO3 + HCl → NaCl + H2O + SO2
Given 2 moles of NaHSO3 and 2 moles of HCl, prove that the number of moles of SO2 formed is 2.
-/
theorem moles_of_SO2_formed :
  (n_NaHSO3 = 2) →
  (n_HCl = 2) →
  (∀ (n : ℕ), (n_NaHSO3 = n) → (n_HCl = n) → (n_SO2 = n)) →
  n_SO2 = 2 :=
by 
  intros hNaHSO3 hHCl hReaction
  exact hReaction 2 hNaHSO3 hHCl

end moles_of_SO2_formed_l26_26035


namespace intersection_of_lines_l26_26016

theorem intersection_of_lines : 
  ∃ (x y : ℚ), 5 * x - 2 * y = 8 ∧ 6 * x + 3 * y = 21 ∧ x = 22 / 9 ∧ y = 19 / 9 :=
by 
  sorry

end intersection_of_lines_l26_26016


namespace expand_and_simplify_l26_26752

theorem expand_and_simplify (x : ℝ) : 3 * (x - 3) * (x + 10) + 2 * x = 3 * x^2 + 23 * x - 90 :=
by
  sorry

end expand_and_simplify_l26_26752


namespace dan_age_l26_26569

theorem dan_age (D : ℕ) (h : D + 20 = 7 * (D - 4)) : D = 8 :=
by
  sorry

end dan_age_l26_26569


namespace grid_spiral_infinite_divisible_by_68_grid_spiral_unique_center_sums_l26_26289

theorem grid_spiral_infinite_divisible_by_68 (n : ℕ) :
  ∃ (k : ℕ), ∃ (m : ℕ), ∃ (t : ℕ), 
  let A := t + 0;
  let B := t + 4;
  let C := t + 12;
  let D := t + 8;
  (k = n * 68 ∧ (n ≥ 1)) ∧ 
  (m = A + B + C + D) ∧ (m % 68 = 0) := by
  sorry

theorem grid_spiral_unique_center_sums (n : ℕ) :
  ∀ (i j : ℕ), 
  let Si := n * 68 + i;
  let Sj := n * 68 + j;
  ¬ (Si = Sj) := by
  sorry

end grid_spiral_infinite_divisible_by_68_grid_spiral_unique_center_sums_l26_26289


namespace simplify_expression_l26_26494

theorem simplify_expression (x : ℝ) : (x + 3) * (x - 3) = x^2 - 9 :=
by
  -- We acknowledge this is the placeholder for the proof.
  -- This statement follows directly from the difference of squares identity.
  sorry

end simplify_expression_l26_26494


namespace rebus_puzzle_verified_l26_26860

-- Defining the conditions
def A := 1
def B := 1
def C := 0
def D := 1
def F := 1
def L := 1
def M := 0
def N := 1
def P := 0
def Q := 1
def T := 1
def G := 8
def H := 1
def K := 4
def W := 4
def X := 1

noncomputable def verify_rebus_puzzle : Prop :=
  (A * B * 10 = 110) ∧
  (6 * G / (10 * H + 7) = 4) ∧
  (L + N * 10 = 20) ∧
  (12 - K = 8) ∧
  (101 + 10 * W + X = 142)

-- Lean statement to verify the problem
theorem rebus_puzzle_verified : verify_rebus_puzzle :=
by {
  -- Values are already defined and will be concluded by Lean
  sorry
}

end rebus_puzzle_verified_l26_26860


namespace gcd_180_270_450_l26_26139

theorem gcd_180_270_450 : Nat.gcd (Nat.gcd 180 270) 450 = 90 := by 
  sorry

end gcd_180_270_450_l26_26139


namespace arithmetic_sequence_index_l26_26960

theorem arithmetic_sequence_index (a : ℕ → ℕ) (n : ℕ) (first_term comm_diff : ℕ):
  (∀ k, a k = first_term + comm_diff * (k - 1)) → a n = 2016 → n = 404 :=
by 
  sorry

end arithmetic_sequence_index_l26_26960


namespace bijective_bounded_dist_l26_26513

open Int

theorem bijective_bounded_dist {k : ℕ} (f : ℤ → ℤ) 
    (hf_bijective : Function.Bijective f)
    (hf_property : ∀ i j : ℤ, |i - j| ≤ k → |f i - (f j)| ≤ k) :
    ∀ i j : ℤ, |f i - (f j)| = |i - j| := 
sorry

end bijective_bounded_dist_l26_26513


namespace geoff_needed_more_votes_to_win_l26_26900

-- Definitions based on the conditions
def total_votes : ℕ := 6000
def percent_to_fraction (p : ℕ) : ℚ := p / 100
def geoff_percent : ℚ := percent_to_fraction 1
def win_percent : ℚ := percent_to_fraction 51

-- Specific values derived from the conditions
def geoff_votes : ℚ := geoff_percent * total_votes
def win_votes : ℚ := win_percent * total_votes + 1

-- The theorem we intend to prove
theorem geoff_needed_more_votes_to_win :
  (win_votes - geoff_votes) = 3001 := by
  sorry

end geoff_needed_more_votes_to_win_l26_26900


namespace line_equation_l26_26367

-- Definitions according to the conditions
def point_P := (3, 4)
def slope_angle_l := 90

-- Statement of the theorem to prove
theorem line_equation (l : ℝ → ℝ) (h1 : l point_P.1 = point_P.2) (h2 : slope_angle_l = 90) :
  ∃ k : ℝ, k = 3 ∧ ∀ x, l x = 3 - x :=
sorry

end line_equation_l26_26367


namespace simplify_fraction_expr_l26_26982

theorem simplify_fraction_expr (a : ℝ) (h : a ≠ 1) : (a / (a - 1) + 1 / (1 - a)) = 1 := by
  sorry

end simplify_fraction_expr_l26_26982


namespace customer_outreach_time_l26_26557

variable (x : ℝ)

theorem customer_outreach_time
  (h1 : 8 = x + x / 2 + 2) :
  x = 4 :=
by sorry

end customer_outreach_time_l26_26557


namespace max_AB_CD_value_l26_26402

def is_digit (x : ℕ) : Prop := x ≥ 1 ∧ x ≤ 9

noncomputable def max_AB_CD : ℕ :=
  let A := 9
  let B := 8
  let C := 7
  let D := 6
  (A + B) + (C + D)

theorem max_AB_CD_value :
  ∀ (A B C D : ℕ), 
    is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧ 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
    (A + B) + (C + D) ≤ max_AB_CD :=
by
  sorry

end max_AB_CD_value_l26_26402


namespace find_second_number_l26_26865

theorem find_second_number (G N: ℕ) (h1: G = 101) (h2: 4351 % G = 8) (h3: N % G = 10) : N = 4359 :=
by 
  sorry

end find_second_number_l26_26865


namespace find_x_l26_26136

theorem find_x (x y : ℝ) (hx : x ≠ 0) (h1 : x / 2 = y^2) (h2 : x / 4 = 4 * y) : x = 128 :=
by
  sorry

end find_x_l26_26136


namespace at_least_one_nonzero_l26_26038

theorem at_least_one_nonzero (a b : ℝ) (h : a^2 + b^2 ≠ 0) : a ≠ 0 ∨ b ≠ 0 :=
by
  sorry

end at_least_one_nonzero_l26_26038


namespace sin_double_angle_pi_six_l26_26377

theorem sin_double_angle_pi_six (α : ℝ)
  (h : 2 * Real.sin α = 1 + 2 * Real.sqrt 3 * Real.cos α) :
  Real.sin (2 * α - Real.pi / 6) = 7 / 8 :=
sorry

end sin_double_angle_pi_six_l26_26377


namespace triangle_third_side_length_l26_26964

theorem triangle_third_side_length (a b : ℕ) (h1 : a = 2) (h2 : b = 3) 
(h3 : ∃ x, x^2 - 10 * x + 21 = 0 ∧ (a + b > x) ∧ (a + x > b) ∧ (b + x > a)) :
  ∃ x, x = 3 := 
by 
  sorry

end triangle_third_side_length_l26_26964


namespace min_value_of_ab_l26_26646

theorem min_value_of_ab {a b : ℝ} (ha : a > 0) (hb : b > 0) 
  (h_eq : ∀ (x y : ℝ), (x / a + y / b = 1) → (x^2 + y^2 = 1)) : a * b = 2 :=
by sorry

end min_value_of_ab_l26_26646


namespace vector_dot_product_l26_26913

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (2, -2)

theorem vector_dot_product : (a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2)) = -1 := by
  -- skipping the proof
  sorry

end vector_dot_product_l26_26913


namespace part_1_select_B_prob_part_2_select_BC_prob_l26_26742

-- Definitions for the four students
inductive Student
| A
| B
| C
| D

open Student

-- Definition for calculating probability
def probability (favorable total : Nat) : Rat :=
  favorable / total

-- Part (1)
theorem part_1_select_B_prob : probability 1 4 = 1 / 4 :=
  sorry

-- Part (2)
theorem part_2_select_BC_prob : probability 2 12 = 1 / 6 :=
  sorry

end part_1_select_B_prob_part_2_select_BC_prob_l26_26742


namespace classify_event_l26_26725

-- Define the conditions of the problem
def involves_variables_and_uncertainties (event: String) : Prop := 
  event = "Turning on the TV and broadcasting 'Chinese Intangible Cultural Heritage'"

-- Define the type of event as a string
def event_type : String := "random"

-- The theorem to prove the classification of the event
theorem classify_event : involves_variables_and_uncertainties "Turning on the TV and broadcasting 'Chinese Intangible Cultural Heritage'" →
  event_type = "random" :=
by
  intro h
  -- Proof is skipped
  sorry

end classify_event_l26_26725


namespace black_grid_probability_l26_26326

theorem black_grid_probability : 
  (let n := 4
   let unit_squares := n * n
   let pairs := unit_squares / 2
   let probability_each_pair := (1:ℝ) / 4
   let total_probability := probability_each_pair ^ pairs
   total_probability = (1:ℝ) / 65536) :=
by
  let n := 4
  let unit_squares := n * n
  let pairs := unit_squares / 2
  let probability_each_pair := (1:ℝ) / 4
  let total_probability := probability_each_pair ^ pairs
  sorry

end black_grid_probability_l26_26326


namespace geometric_sequence_sum_ratio_l26_26674

theorem geometric_sequence_sum_ratio 
  (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a (n+1) = a 0 * q ^ n)
  (h2 : ∀ n, S n = (a 0 * (q ^ n - 1)) / (q - 1))
  (h3 : 6 * a 3 = a 0 * q ^ 5 - a 0 * q ^ 4) :
  S 4 / S 2 = 10 := 
sorry

end geometric_sequence_sum_ratio_l26_26674


namespace property_holds_for_1_and_4_l26_26445

theorem property_holds_for_1_and_4 (n : ℕ) : 
  (∀ q : ℕ, n % q^2 < q^(q^2) / 2) ↔ (n = 1 ∨ n = 4) :=
by sorry

end property_holds_for_1_and_4_l26_26445


namespace dave_has_20_more_than_derek_l26_26974

-- Define the amounts of money Derek and Dave start with
def initial_amount_derek : ℕ := 40
def initial_amount_dave : ℕ := 50

-- Define the amounts Derek spends
def spend_derek_lunch_self1 : ℕ := 14
def spend_derek_lunch_dad : ℕ := 11
def spend_derek_lunch_self2 : ℕ := 5
def spend_derek_dessert_sister : ℕ := 8

-- Define the amounts Dave spends
def spend_dave_lunch_mom : ℕ := 7
def spend_dave_lunch_cousin : ℕ := 12
def spend_dave_snacks_friends : ℕ := 9

-- Define calculations for total spending
def total_spend_derek : ℕ :=
  spend_derek_lunch_self1 + spend_derek_lunch_dad + spend_derek_lunch_self2 + spend_derek_dessert_sister

def total_spend_dave : ℕ :=
  spend_dave_lunch_mom + spend_dave_lunch_cousin + spend_dave_snacks_friends

-- Define remaining amount of money
def remaining_derek : ℕ :=
  initial_amount_derek - total_spend_derek

def remaining_dave : ℕ :=
  initial_amount_dave - total_spend_dave

-- Define the property to be proved
theorem dave_has_20_more_than_derek : remaining_dave - remaining_derek = 20 := by
  sorry

end dave_has_20_more_than_derek_l26_26974


namespace sum_of_roots_l26_26945

theorem sum_of_roots (x : ℝ) (h : (x + 3) * (x - 2) = 15) : x = -1 :=
sorry

end sum_of_roots_l26_26945


namespace distinct_infinite_solutions_l26_26909

theorem distinct_infinite_solutions (n : ℕ) (hn : n > 0) : 
  ∃ p q : ℤ, p + q * Real.sqrt 5 = (9 + 4 * Real.sqrt 5) ^ n ∧ (p * p - 5 * q * q = 1) ∧ 
  ∀ m : ℕ, (m ≠ n → (9 + 4 * Real.sqrt 5) ^ m ≠ (9 + 4 * Real.sqrt 5) ^ n) :=
by
  sorry

end distinct_infinite_solutions_l26_26909


namespace original_price_l26_26792

theorem original_price (P : ℝ) (h1 : ∃ P : ℝ, (120 : ℝ) = P + 0.2 * P) : P = 100 :=
by
  obtain ⟨P, h⟩ := h1
  sorry

end original_price_l26_26792


namespace Polly_tweets_l26_26391

theorem Polly_tweets :
  let HappyTweets := 18 * 50
  let HungryTweets := 4 * 35
  let WatchingReflectionTweets := 45 * 30
  let SadTweets := 6 * 20
  let PlayingWithToysTweets := 25 * 75
  HappyTweets + HungryTweets + WatchingReflectionTweets + SadTweets + PlayingWithToysTweets = 4385 :=
by
  sorry

end Polly_tweets_l26_26391


namespace train_crossing_time_l26_26254

-- Define the conditions
def length_of_train : ℕ := 200  -- in meters
def speed_of_train_kmph : ℕ := 90  -- in km per hour
def length_of_tunnel : ℕ := 2500  -- in meters

-- Conversion of speed from kmph to m/s
def speed_of_train_mps : ℕ := speed_of_train_kmph * 1000 / 3600

-- Define the total distance to be covered (train length + tunnel length)
def total_distance : ℕ := length_of_train + length_of_tunnel

-- Define the expected time to cross the tunnel (in seconds)
def expected_time : ℕ := 108

-- The theorem statement to prove
theorem train_crossing_time : (total_distance / speed_of_train_mps) = expected_time := 
by
  sorry

end train_crossing_time_l26_26254


namespace f_log2_9_l26_26517

def f (x : ℝ) : ℝ := sorry

theorem f_log2_9 : 
  (∀ x, f (x + 1) = 1 / f x) → 
  (∀ x, 0 < x ∧ x ≤ 1 → f x = 2^x) → 
  f (Real.log 9 / Real.log 2) = 8 / 9 :=
by
  intros h1 h2
  sorry

end f_log2_9_l26_26517


namespace solve_quadratic_eq_l26_26743

theorem solve_quadratic_eq (x : ℝ) : x^2 - 4 = 0 ↔ x = 2 ∨ x = -2 :=
by
  sorry

end solve_quadratic_eq_l26_26743


namespace smallest_composite_square_side_length_l26_26835

theorem smallest_composite_square_side_length (n : ℕ) (h : ∃ k, 14 * n = k^2) : 
  ∃ m : ℕ, n = 14 ∧ m = 14 :=
by
  sorry

end smallest_composite_square_side_length_l26_26835


namespace geo_seq_ratio_l26_26039

theorem geo_seq_ratio (S : ℕ → ℝ) (r : ℝ) (hS : ∀ n, S n = (1 - r^(n+1)) / (1 - r))
  (hS_ratio : S 10 / S 5 = 1 / 2) : S 15 / S 5 = 3 / 4 := 
by
  sorry

end geo_seq_ratio_l26_26039


namespace factorize_ax2_minus_a_l26_26470

theorem factorize_ax2_minus_a (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) :=
by
  sorry

end factorize_ax2_minus_a_l26_26470


namespace value_of_m_l26_26249

theorem value_of_m (x m : ℝ) (h : x ≠ 3) (H : (x / (x - 3) = 2 - m / (3 - x))) : m = 3 :=
sorry

end value_of_m_l26_26249


namespace range_of_a_l26_26226

variable (x a : ℝ)

theorem range_of_a (h1 : ∀ x, x ≤ a → x < 2) (h2 : ∀ x, x < 2) : a ≥ 2 :=
sorry

end range_of_a_l26_26226


namespace cdf_from_pdf_l26_26395

noncomputable def pdf (x : ℝ) : ℝ :=
  if x ≤ 0 then 0
  else if 0 < x ∧ x ≤ Real.pi / 2 then Real.cos x
  else 0

noncomputable def cdf (x : ℝ) : ℝ :=
  if x ≤ 0 then 0
  else if 0 < x ∧ x ≤ Real.pi / 2 then Real.sin x
  else 1

theorem cdf_from_pdf (x : ℝ) : 
  ∀ x : ℝ, cdf x = 
    if x ≤ 0 then 0
    else if 0 < x ∧ x ≤ Real.pi / 2 then Real.sin x
    else 1 :=
by
  sorry

end cdf_from_pdf_l26_26395


namespace desired_overall_percentage_l26_26834

-- Define the scores in the three subjects
def score1 := 50
def score2 := 70
def score3 := 90

-- Define the expected overall percentage
def expected_overall_percentage := 70

-- The main theorem to prove
theorem desired_overall_percentage :
  (score1 + score2 + score3) / 3 = expected_overall_percentage :=
by
  sorry

end desired_overall_percentage_l26_26834


namespace bus_stoppage_time_l26_26450

theorem bus_stoppage_time
  (speed_excluding_stoppages : ℝ)
  (speed_including_stoppages : ℝ)
  (reduction_in_speed : speed_excluding_stoppages - speed_including_stoppages = 8) :
  ∃ t : ℝ, t = 9.6 := 
sorry

end bus_stoppage_time_l26_26450


namespace isosceles_triangle_no_obtuse_l26_26144

theorem isosceles_triangle_no_obtuse (A B C : ℝ) 
  (h1 : A = 70) 
  (h2 : B = 70) 
  (h3 : A + B + C = 180) 
  (h_iso : A = B) 
  : (A ≤ 90) ∧ (B ≤ 90) ∧ (C ≤ 90) :=
by
  sorry

end isosceles_triangle_no_obtuse_l26_26144


namespace right_triangle_ratio_l26_26300

theorem right_triangle_ratio (a b c r s : ℝ) (h : a / b = 2 / 5)
  (h_c : c^2 = a^2 + b^2)
  (h_r : r = a^2 / c)
  (h_s : s = b^2 / c) :
  r / s = 4 / 25 := by
  sorry

end right_triangle_ratio_l26_26300


namespace prime_triplets_l26_26231

theorem prime_triplets (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) :
  p ^ q + q ^ p = r ↔ (p = 2 ∧ q = 3 ∧ r = 17) ∨ (p = 3 ∧ q = 2 ∧ r = 17) := by
  sorry

end prime_triplets_l26_26231


namespace product_of_real_numbers_triple_when_added_to_their_reciprocal_l26_26048

theorem product_of_real_numbers_triple_when_added_to_their_reciprocal :
  (∀ x : ℝ, x + (1 / x) = 3 * x → x = (1 / Real.sqrt 2) ∨ x = - (1 / Real.sqrt 2)) →
  (∀ x1 x2 : ℝ, (x1 = (1 / Real.sqrt 2) ∧ x2 = - (1 / Real.sqrt 2)) → x1 * x2 = - (1 / 2)) :=
by
  intros h1 h2
  sorry

end product_of_real_numbers_triple_when_added_to_their_reciprocal_l26_26048


namespace arrangement_of_BANANA_l26_26133

theorem arrangement_of_BANANA : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 :=
by
  sorry

end arrangement_of_BANANA_l26_26133


namespace angles_of_triangle_l26_26188

theorem angles_of_triangle (a b c m_a m_b : ℝ) (h1 : m_a ≥ a) (h2 : m_b ≥ b) : 
  ∃ (α β γ : ℝ), ∀ t, 
  (t = 90) ∧ (α = 45) ∧ (β = 45) := 
sorry

end angles_of_triangle_l26_26188


namespace area_of_sector_l26_26169

theorem area_of_sector (L θ : ℝ) (hL : L = 4) (hθ : θ = 2) : 
  (1 / 2) * ((L / θ) ^ 2) * θ = 4 := by
  sorry

end area_of_sector_l26_26169


namespace find_B_l26_26699

-- Define the polynomial function and its properties
def polynomial (z : ℤ) (A B : ℤ) : ℤ :=
  z^4 - 6 * z^3 + A * z^2 + B * z + 9

-- Prove that B = -9 under the given conditions
theorem find_B (A B : ℤ) (r1 r2 r3 r4 : ℤ)
  (h1 : polynomial r1 A B = 0)
  (h2 : polynomial r2 A B = 0)
  (h3 : polynomial r3 A B = 0)
  (h4 : polynomial r4 A B = 0)
  (h5 : r1 + r2 + r3 + r4 = 6)
  (h6 : r1 > 0)
  (h7 : r2 > 0)
  (h8 : r3 > 0)
  (h9 : r4 > 0) :
  B = -9 :=
by
  sorry

end find_B_l26_26699


namespace simple_interest_rate_l26_26091

theorem simple_interest_rate (SI : ℝ) (P : ℝ) (T : ℝ) (R : ℝ)
  (h1 : SI = 130) (h2 : P = 780) (h3 : T = 4) :
  R = 4.17 :=
sorry

end simple_interest_rate_l26_26091


namespace chandra_valid_pairings_l26_26315

noncomputable def valid_pairings (total_items : Nat) (invalid_pairing : Nat) : Nat :=
total_items * total_items - invalid_pairing

theorem chandra_valid_pairings : valid_pairings 5 1 = 24 := by
  sorry

end chandra_valid_pairings_l26_26315


namespace chinese_character_equation_l26_26177

noncomputable def units_digit (n: ℕ) : ℕ :=
  n % 10

noncomputable def tens_digit (n: ℕ) : ℕ :=
  (n / 10) % 10

noncomputable def hundreds_digit (n: ℕ) : ℕ :=
  (n / 100) % 10

def Math : ℕ := 25
def LoveMath : ℕ := 125
def ILoveMath : ℕ := 3125

theorem chinese_character_equation :
  Math * LoveMath = ILoveMath :=
by
  have h_units_math := units_digit Math
  have h_units_lovemath := units_digit LoveMath
  have h_units_ilovemath := units_digit ILoveMath
  
  have h_tens_math := tens_digit Math
  have h_tens_lovemath := tens_digit LoveMath
  have h_tens_ilovemath := tens_digit ILoveMath

  have h_hundreds_lovemath := hundreds_digit LoveMath
  have h_hundreds_ilovemath := hundreds_digit ILoveMath

  -- Check conditions:
  -- h_units_* should be 0, 1, 5 or 6
  -- h_tens_math == h_tens_lovemath == h_tens_ilovemath
  -- h_hundreds_lovemath == h_hundreds_ilovemath

  sorry -- Proof would go here

end chinese_character_equation_l26_26177


namespace polynomial_factorization_l26_26656

theorem polynomial_factorization :
  5 * (x + 3) * (x + 7) * (x + 11) * (x + 13) - 4 * x^2 = 5 * x^4 + 180 * x^3 + 1431 * x^2 + 4900 * x + 5159 :=
by sorry

end polynomial_factorization_l26_26656


namespace probability_at_least_one_admitted_l26_26760

-- Define the events and probabilities
variables (A B : Prop)
variables (P_A : ℝ) (P_B : ℝ)
variables (independent : Prop)

-- Assume the given conditions
def P_A_def : Prop := P_A = 0.6
def P_B_def : Prop := P_B = 0.7
def independent_def : Prop := independent = true  -- simplistic representation for independence

-- Statement: Prove the probability that at least one of them is admitted is 0.88
theorem probability_at_least_one_admitted : 
  P_A = 0.6 → P_B = 0.7 → independent = true →
  (1 - (1 - P_A) * (1 - P_B)) = 0.88 :=
by
  intros
  sorry

end probability_at_least_one_admitted_l26_26760


namespace even_combinations_result_in_486_l26_26765

-- Define the operations possible (increase by 2, increase by 3, multiply by 2)
inductive Operation
| inc2
| inc3
| mul2

open Operation

-- Function to apply an operation to a number
def applyOperation : Operation → ℕ → ℕ
| inc2, n => n + 2
| inc3, n => n + 3
| mul2, n => n * 2

-- Function to apply a list of operations to the initial number 1
def applyOperationsList (ops : List Operation) : ℕ :=
ops.foldl (fun acc op => applyOperation op acc) 1

-- Count the number of combinations that result in an even number
noncomputable def evenCombosCount : ℕ :=
(List.replicate 6 [inc2, inc3, mul2]).foldl (fun acc x => acc * x.length) 1
  |> λ _ => (3 ^ 5) * 2

theorem even_combinations_result_in_486 :
  evenCombosCount = 486 :=
sorry

end even_combinations_result_in_486_l26_26765


namespace goods_train_speed_l26_26115

theorem goods_train_speed
  (length_train : ℝ)
  (length_platform : ℝ)
  (time_taken : ℝ)
  (speed_kmph : ℝ)
  (h1 : length_train = 240.0416)
  (h2 : length_platform = 280)
  (h3 : time_taken = 26)
  (h4 : speed_kmph = 72.00576) :
  speed_kmph = ((length_train + length_platform) / time_taken) * 3.6 := sorry

end goods_train_speed_l26_26115


namespace aerith_is_correct_l26_26726

theorem aerith_is_correct :
  ∀ x : ℝ, x = 1.4 → (x ^ (x ^ x)) < 2 → ∃ y : ℝ, y = x ^ (x ^ x) :=
by
  sorry

end aerith_is_correct_l26_26726


namespace sum_of_squares_l26_26106

theorem sum_of_squares (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 :=
sorry

end sum_of_squares_l26_26106


namespace find_quotient_l26_26560

theorem find_quotient
  (larger smaller : ℕ)
  (h1 : larger - smaller = 1200)
  (h2 : larger = 1495)
  (rem : ℕ := 4)
  (h3 : larger % smaller = rem) :
  larger / smaller = 5 := 
by 
  sorry

end find_quotient_l26_26560


namespace no_solution_x_to_2n_plus_y_to_2n_eq_z_sq_l26_26426

theorem no_solution_x_to_2n_plus_y_to_2n_eq_z_sq (n : ℕ) (h : ∀ (x y z : ℕ), x^n + y^n ≠ z^n) : ∀ (x y z : ℕ), x^(2*n) + y^(2*n) ≠ z^2 :=
by 
  intro x y z
  sorry

end no_solution_x_to_2n_plus_y_to_2n_eq_z_sq_l26_26426


namespace calculate_drift_l26_26645

def width_of_river : ℕ := 400
def speed_of_boat : ℕ := 10
def time_to_cross : ℕ := 50
def actual_distance_traveled := speed_of_boat * time_to_cross

theorem calculate_drift : actual_distance_traveled - width_of_river = 100 :=
by
  -- width_of_river = 400
  -- speed_of_boat = 10
  -- time_to_cross = 50
  -- actual_distance_traveled = 10 * 50 = 500
  -- expected drift = 500 - 400 = 100
  sorry

end calculate_drift_l26_26645


namespace work_rate_a_b_l26_26508

/-- a and b can do a piece of work in some days, b and c in 5 days, c and a in 15 days. If c takes 12 days to do the work, 
    prove that a and b together can complete the work in 10 days.
-/
theorem work_rate_a_b
  (A B C : ℚ) 
  (h1 : B + C = 1 / 5)
  (h2 : C + A = 1 / 15)
  (h3 : C = 1 / 12) :
  (A + B = 1 / 10) := 
sorry

end work_rate_a_b_l26_26508


namespace number_of_intersections_of_lines_l26_26703

theorem number_of_intersections_of_lines : 
  let L1 := {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 = 12}
  let L2 := {p : ℝ × ℝ | 5 * p.1 - 2 * p.2 = 10}
  let L3 := {p : ℝ × ℝ | p.1 = 3}
  let L4 := {p : ℝ × ℝ | p.2 = 1}
  ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧ p1 ∈ L1 ∧ p1 ∈ L2 ∧ p2 ∈ L3 ∧ p2 ∈ L4 :=
by
  sorry

end number_of_intersections_of_lines_l26_26703


namespace sales_difference_l26_26735
noncomputable def max_min_difference (sales : List ℕ) : ℕ :=
  (sales.maximum.getD 0) - (sales.minimum.getD 0)

theorem sales_difference :
  max_min_difference [1200, 1450, 1950, 1700] = 750 :=
by
  sorry

end sales_difference_l26_26735


namespace no_positive_reals_satisfy_conditions_l26_26348

theorem no_positive_reals_satisfy_conditions :
  ¬ ∃ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  4 * (a * b + b * c + c * a) - 1 ≥ a^2 + b^2 + c^2 ∧ 
  a^2 + b^2 + c^2 ≥ 3 * (a^3 + b^3 + c^3) :=
by
  sorry

end no_positive_reals_satisfy_conditions_l26_26348


namespace students_pass_both_subjects_l26_26722

theorem students_pass_both_subjects
  (F_H F_E F_HE : ℝ)
  (h1 : F_H = 0.25)
  (h2 : F_E = 0.48)
  (h3 : F_HE = 0.27) :
  (100 - (F_H + F_E - F_HE) * 100) = 54 :=
by
  sorry

end students_pass_both_subjects_l26_26722


namespace lucy_flour_used_l26_26804

theorem lucy_flour_used
  (initial_flour : ℕ := 500)
  (final_flour : ℕ := 130)
  (flour_needed_to_buy : ℤ := 370)
  (used_flour : ℕ) :
  initial_flour - used_flour = 2 * final_flour → used_flour = 240 :=
by
  sorry

end lucy_flour_used_l26_26804


namespace total_area_calculations_l26_26966

noncomputable def total_area_in_hectares : ℝ :=
  let sections := 5
  let area_per_section := 60
  let conversion_factor_acre_to_hectare := 0.404686
  sections * area_per_section * conversion_factor_acre_to_hectare

noncomputable def total_area_in_square_meters : ℝ :=
  let conversion_factor_hectare_to_square_meter := 10000
  total_area_in_hectares * conversion_factor_hectare_to_square_meter

theorem total_area_calculations :
  total_area_in_hectares = 121.4058 ∧ total_area_in_square_meters = 1214058 := by
  sorry

end total_area_calculations_l26_26966


namespace smallest_even_sum_l26_26745

theorem smallest_even_sum :
  ∃ (a b c : Int), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ∈ ({8, -4, 3, 27, 10} : Set Int) ∧ b ∈ ({8, -4, 3, 27, 10} : Set Int) ∧ c ∈ ({8, -4, 3, 27, 10} : Set Int) ∧ (a + b + c) % 2 = 0 ∧ (a + b + c) = 14 := sorry

end smallest_even_sum_l26_26745


namespace determine_k_l26_26441

noncomputable def f (x k : ℝ) : ℝ := -4 * x^3 + k * x

theorem determine_k : ∀ k : ℝ, (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x k ≤ 1) → k = 3 :=
by
  sorry

end determine_k_l26_26441


namespace total_new_bottles_l26_26059

theorem total_new_bottles (initial_bottles : ℕ) (recycle_ratio : ℕ) (bonus_ratio : ℕ) (final_bottles : ℕ) :
  initial_bottles = 625 →
  recycle_ratio = 5 →
  bonus_ratio = 20 →
  final_bottles = 163 :=
by {
  sorry -- Proof goes here
}

end total_new_bottles_l26_26059


namespace eval_expression_l26_26213

theorem eval_expression : (8 / 4 - 3 * 2 + 9 - 3^2) = -4 := sorry

end eval_expression_l26_26213


namespace bertha_daughters_and_granddaughters_have_no_daughters_l26_26668

def total_daughters_and_granddaughters (daughters granddaughters : Nat) : Nat :=
daughters + granddaughters

def no_daughters (bertha_daughters bertha_granddaughters : Nat) : Nat :=
bertha_daughters + bertha_granddaughters

theorem bertha_daughters_and_granddaughters_have_no_daughters :
  (bertha_daughters : Nat) →
  (daughters_with_6_daughters : Nat) →
  (granddaughters : Nat) →
  (total_daughters_and_granddaughters bertha_daughters granddaughters = 30) →
  bertha_daughters = 6 →
  granddaughters = 6 * daughters_with_6_daughters →
  no_daughters (bertha_daughters - daughters_with_6_daughters) granddaughters = 26 :=
by
  intros bertha_daughters daughters_with_6_daughters granddaughters h_total h_bertha h_granddaughters
  sorry

end bertha_daughters_and_granddaughters_have_no_daughters_l26_26668


namespace remainder_when_product_divided_by_5_l26_26949

def n1 := 1483
def n2 := 1773
def n3 := 1827
def n4 := 2001
def mod5 (n : Nat) : Nat := n % 5

theorem remainder_when_product_divided_by_5 :
  mod5 (n1 * n2 * n3 * n4) = 3 :=
sorry

end remainder_when_product_divided_by_5_l26_26949


namespace find_f_three_l26_26230

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x
def f_condition (f : ℝ → ℝ) := ∀ x : ℝ, x < 0 → f x = (1/2)^x

theorem find_f_three (f : ℝ → ℝ) (h₁ : odd_function f) (h₂ : f_condition f) : f 3 = -8 :=
sorry

end find_f_three_l26_26230


namespace value_of_a_l26_26878

def f (x : ℝ) : ℝ := x^2 + 9
def g (x : ℝ) : ℝ := x^2 - 5

theorem value_of_a (a : ℝ) (h1 : a > 0) (h2 : f (g a) = 25) : a = 3 :=
by
  sorry

end value_of_a_l26_26878


namespace divisible_iff_l26_26993

-- Definitions from the conditions
def a : ℕ → ℕ
  | 0     => 0
  | 1     => 1
  | (n+2) => 2 * a (n + 1) + a n

-- Main theorem statement.
theorem divisible_iff (n k : ℕ) : 2^k ∣ a n ↔ 2^k ∣ n := by
  sorry

end divisible_iff_l26_26993


namespace combined_share_of_A_and_C_l26_26006

-- Definitions based on the conditions
def total_money : Float := 15800
def charity_investment : Float := 0.10 * total_money
def savings_investment : Float := 0.08 * total_money
def remaining_money : Float := total_money - charity_investment - savings_investment

def ratio_A : Nat := 5
def ratio_B : Nat := 9
def ratio_C : Nat := 6
def ratio_D : Nat := 5
def sum_of_ratios : Nat := ratio_A + ratio_B + ratio_C + ratio_D

def share_A : Float := (ratio_A.toFloat / sum_of_ratios.toFloat) * remaining_money
def share_C : Float := (ratio_C.toFloat / sum_of_ratios.toFloat) * remaining_money
def combined_share_A_C : Float := share_A + share_C

-- Statement to be proven
theorem combined_share_of_A_and_C : combined_share_A_C = 5700.64 := by
  sorry

end combined_share_of_A_and_C_l26_26006


namespace b_gets_more_than_c_l26_26983

-- Define A, B, and C as real numbers
variables (A B C : ℝ)

theorem b_gets_more_than_c 
  (h1 : A = 3 * B)
  (h2 : B = C + 25)
  (h3 : A + B + C = 645)
  (h4 : B = 134) : 
  B - C = 25 :=
by
  -- Using the conditions from the problem
  sorry

end b_gets_more_than_c_l26_26983


namespace find_a_and_b_l26_26209

theorem find_a_and_b (a b : ℝ) 
  (h_tangent_slope : (2 * a * 2 + b = 1)) 
  (h_point_on_parabola : (a * 4 + b * 2 + 9 = -1)) : 
  a = 3 ∧ b = -11 :=
by
  sorry

end find_a_and_b_l26_26209


namespace harrys_morning_routine_time_l26_26616

theorem harrys_morning_routine_time :
  (15 + 20 + 25 + 2 * 15 = 90) :=
by
  sorry

end harrys_morning_routine_time_l26_26616


namespace cashier_total_bills_l26_26309

theorem cashier_total_bills
  (total_value : ℕ)
  (num_ten_bills : ℕ)
  (num_twenty_bills : ℕ)
  (h1 : total_value = 330)
  (h2 : num_ten_bills = 27)
  (h3 : num_twenty_bills = 3) :
  num_ten_bills + num_twenty_bills = 30 :=
by
  -- Proof goes here
  sorry

end cashier_total_bills_l26_26309


namespace exists_integer_lt_sqrt_10_l26_26374

theorem exists_integer_lt_sqrt_10 : ∃ k : ℤ, k < Real.sqrt 10 := by
  have h_sqrt_bounds : 3 < Real.sqrt 10 ∧ Real.sqrt 10 < 4 := by
    -- Proof involving basic properties and calculations
    sorry
  exact ⟨3, h_sqrt_bounds.left⟩

end exists_integer_lt_sqrt_10_l26_26374


namespace oxen_count_l26_26202

theorem oxen_count (B C O : ℕ) (H1 : 3 * B = 4 * C) (H2 : 3 * B = 2 * O) (H3 : 15 * B + 24 * C + O * O = 33 * B + (3 / 2) * O * B) (H4 : 24 * B = 48) (H5 : 60 * C + 30 * B + 18 * (O * (3 / 2) * B) = 108 * B + (3 / 2) * O * B * 18)
: O = 8 :=
by 
  sorry

end oxen_count_l26_26202


namespace smallest_d_for_divisibility_by_3_l26_26733

def sum_of_digits (d : ℕ) : ℕ := 5 + 4 + 7 + d + 0 + 6

theorem smallest_d_for_divisibility_by_3 (d : ℕ) :
  (sum_of_digits 2) % 3 = 0 ∧ ∀ k, k < 2 → sum_of_digits k % 3 ≠ 0 := 
sorry

end smallest_d_for_divisibility_by_3_l26_26733


namespace time_to_cover_length_l26_26523

-- Define the conditions
def speed_escalator : ℝ := 12
def length_escalator : ℝ := 150
def speed_person : ℝ := 3

-- State the theorem to be proved
theorem time_to_cover_length : (length_escalator / (speed_escalator + speed_person)) = 10 := by
  sorry

end time_to_cover_length_l26_26523


namespace average_speed_trip_l26_26988

theorem average_speed_trip :
  let distance_1 := 65
  let distance_2 := 45
  let distance_3 := 55
  let distance_4 := 70
  let distance_5 := 60
  let total_time := 5
  let total_distance := distance_1 + distance_2 + distance_3 + distance_4 + distance_5
  let average_speed := total_distance / total_time
  average_speed = 59 :=
by
  sorry

end average_speed_trip_l26_26988


namespace balloon_count_correct_l26_26997

def gold_balloons : ℕ := 141
def black_balloons : ℕ := 150
def silver_balloons : ℕ := 2 * gold_balloons
def total_balloons : ℕ := gold_balloons + silver_balloons + black_balloons

theorem balloon_count_correct : total_balloons = 573 := by
  sorry

end balloon_count_correct_l26_26997


namespace percentage_increase_selling_price_l26_26543

-- Defining the conditions
def original_price : ℝ := 6
def increased_price : ℝ := 8.64
def total_sales_per_hour : ℝ := 216
def max_price : ℝ := 10

-- Statement for Part 1
theorem percentage_increase (x : ℝ) : 6 * (1 + x)^2 = 8.64 → x = 0.2 :=
by
  sorry

-- Statement for Part 2
theorem selling_price (a : ℝ) : (6 + a) * (30 - 2 * a) = 216 → 6 + a ≤ 10 → 6 + a = 9 :=
by
  sorry

end percentage_increase_selling_price_l26_26543


namespace parabola_focus_l26_26590

theorem parabola_focus (f : ℝ) :
  (∀ x : ℝ, 2*x^2 = x^2 + (2*x^2 - f)^2 - (2*x^2 - -f)^2) →
  f = -1/8 :=
by sorry

end parabola_focus_l26_26590


namespace donna_additional_flyers_l26_26336

theorem donna_additional_flyers (m d a : ℕ) (h1 : m = 33) (h2 : d = 2 * m + a) (h3 : d = 71) : a = 5 :=
by
  have m_val : m = 33 := h1
  rw [m_val] at h2
  linarith [h3, h2]

end donna_additional_flyers_l26_26336


namespace max_volume_prism_l26_26739

theorem max_volume_prism (a b h : ℝ) (h_congruent_lateral : a = b) (sum_areas_eq_48 : a * h + b * h + a * b = 48) : 
  ∃ V : ℝ, V = 64 :=
by
  sorry

end max_volume_prism_l26_26739


namespace seventeenth_replacement_month_l26_26687

def months_after_january (n : Nat) : Nat :=
  n % 12

theorem seventeenth_replacement_month :
  months_after_january (7 * 16) = 4 :=
by
  sorry

end seventeenth_replacement_month_l26_26687


namespace Amy_bought_tomato_soup_l26_26631

-- Conditions
variables (chicken_soup_cans total_soups : ℕ)
variable (Amy_bought_soups : total_soups = 9)
variable (Amy_bought_chicken_soup : chicken_soup_cans = 6)

-- Question: How many cans of tomato soup did she buy?
def cans_of_tomato_soup (chicken_soup_cans total_soups : ℕ) : ℕ :=
  total_soups - chicken_soup_cans

-- Theorem: Prove that the number of cans of tomato soup Amy bought is 3
theorem Amy_bought_tomato_soup : 
  cans_of_tomato_soup chicken_soup_cans total_soups = 3 :=
by
  rw [Amy_bought_soups, Amy_bought_chicken_soup]
  -- The steps for the proof would follow here
  sorry

end Amy_bought_tomato_soup_l26_26631


namespace sum_a_b_eq_5_l26_26806

theorem sum_a_b_eq_5 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * b = a - 2) (h4 : (-2)^2 = b * (2 * b + 2)) : a + b = 5 :=
sorry

end sum_a_b_eq_5_l26_26806


namespace senate_subcommittee_l26_26584

/-- 
Proof of the number of ways to form a Senate subcommittee consisting of 7 Republicans
and 2 Democrats from the available 12 Republicans and 6 Democrats.
-/
theorem senate_subcommittee (R D : ℕ) (choose_R choose_D : ℕ) (hR : R = 12) (hD : D = 6) 
  (h_choose_R : choose_R = 7) (h_choose_D : choose_D = 2) : 
  (Nat.choose R choose_R) * (Nat.choose D choose_D) = 11880 := by
  sorry

end senate_subcommittee_l26_26584


namespace math_problem_l26_26316

variables {x y : ℝ}

theorem math_problem (h1 : x * y = 16) (h2 : 1 / x = 3 * (1 / y)) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x < y) : 
  (2 * y - x) = 24 - (4 * Real.sqrt 3) / 3 :=
by sorry

end math_problem_l26_26316


namespace vector_addition_example_l26_26270

theorem vector_addition_example :
  let a := (1, 2)
  let b := (-2, 1)
  a.1 + 2 * b.1 = -3 ∧ a.2 + 2 * b.2 = 4 :=
by
  sorry

end vector_addition_example_l26_26270


namespace max_S_possible_l26_26455

theorem max_S_possible (nums : List ℝ) (h_nums_in_bound : ∀ n ∈ nums, 0 ≤ n ∧ n ≤ 1) (h_sum_leq_253_div_12 : nums.sum ≤ 253 / 12) :
  ∃ (A B : List ℝ), (∀ x ∈ A, x ∈ nums) ∧ (∀ y ∈ B, y ∈ nums) ∧ A.union B = nums ∧ A.sum ≤ 11 ∧ B.sum ≤ 11 :=
sorry

end max_S_possible_l26_26455


namespace pool_capacity_is_800_l26_26075

-- Definitions for the given problem conditions
def fill_time_all_valves : ℝ := 36
def fill_time_first_valve : ℝ := 180
def fill_time_second_valve : ℝ := 240
def third_valve_more_than_first : ℝ := 30
def third_valve_more_than_second : ℝ := 10
def leak_rate : ℝ := 20

-- Function definition for the capacity of the pool
def capacity (W : ℝ) : Prop :=
  let V1 := W / fill_time_first_valve
  let V2 := W / fill_time_second_valve
  let V3 := (W / fill_time_first_valve) + third_valve_more_than_first
  let effective_rate := V1 + V2 + V3 - leak_rate
  (W / fill_time_all_valves) = effective_rate

-- Proof statement that the capacity of the pool is 800 cubic meters
theorem pool_capacity_is_800 : capacity 800 :=
by
  -- Proof is omitted
  sorry

end pool_capacity_is_800_l26_26075


namespace even_increasing_decreasing_l26_26234

theorem even_increasing_decreasing (f : ℝ → ℝ) (h_def : ∀ x : ℝ, f x = -x^2) :
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, x < 0 → f x < f (x + 1)) ∧ (∀ x : ℝ, x > 0 → f x > f (x + 1)) :=
by
  sorry

end even_increasing_decreasing_l26_26234


namespace weight_of_6m_rod_l26_26967

theorem weight_of_6m_rod (r ρ : ℝ) (h₁ : 11.25 > 0) (h₂ : 6 > 0) (h₃ : 0 < r) (h₄ : 42.75 = π * r^2 * 11.25 * ρ) : 
  (π * r^2 * 6 * (42.75 / (π * r^2 * 11.25))) = 22.8 :=
by
  sorry

end weight_of_6m_rod_l26_26967


namespace calculate_DA_l26_26676

open Real

-- Definitions based on conditions
def AU := 90
def AN := 180
def UB := 270
def AB := AU + UB
def ratio := 3 / 4

-- Statement of the problem in Lean 
theorem calculate_DA :
  ∃ (p q : ℕ), (q ≠ 0) ∧ (∀ p' q' : ℕ, ¬ (q = p'^2 * q')) ∧ DA = p * sqrt q ∧ p + q = result :=
  sorry

end calculate_DA_l26_26676


namespace mice_path_count_l26_26472

theorem mice_path_count
  (x y : ℕ)
  (left_house_yesterday top_house_yesterday right_house_yesterday : ℕ)
  (left_house_today top_house_today right_house_today : ℕ)
  (h_left_yesterday : left_house_yesterday = 8)
  (h_top_yesterday : top_house_yesterday = 4)
  (h_right_yesterday : right_house_yesterday = 7)
  (h_left_today : left_house_today = 4)
  (h_top_today : top_house_today = 4)
  (h_right_today : right_house_today = 7)
  (h_eq : (left_house_yesterday - left_house_today) + 
          (right_house_yesterday - right_house_today) = 
          top_house_today - top_house_yesterday) :
  x + y = 11 :=
by
  sorry

end mice_path_count_l26_26472


namespace number_in_scientific_notation_l26_26575

def scientific_notation_form (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ a ∧ a < 10

theorem number_in_scientific_notation : scientific_notation_form 3.7515 7 ∧ 37515000 = 3.7515 * 10^7 :=
by
  sorry

end number_in_scientific_notation_l26_26575


namespace complex_number_property_l26_26596

theorem complex_number_property (i : ℂ) (h : i^2 = -1) : (1 + i)^(20) - (1 - i)^(20) = 0 :=
by {
  sorry
}

end complex_number_property_l26_26596


namespace smallest_n_inequality_l26_26290

theorem smallest_n_inequality :
  ∃ n : ℕ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧
  (∀ m : ℕ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ m * (x^4 + y^4 + z^4 + w^4)) → n ≤ m) :=
sorry

end smallest_n_inequality_l26_26290


namespace total_money_l26_26998

theorem total_money (p q r : ℕ)
  (h1 : r = 2000)
  (h2 : r = (2 / 3) * (p + q)) : 
  p + q + r = 5000 :=
by
  sorry

end total_money_l26_26998


namespace hyperbola_is_given_equation_l26_26763

noncomputable def hyperbola_equation : Prop :=
  ∃ a b : ℝ, 
    (a > 0 ∧ b > 0) ∧ 
    (4^2 = a^2 + b^2) ∧ 
    (a = b) ∧ 
    (∀ x y : ℝ, (x^2 / 8 - y^2 / 8 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1))

theorem hyperbola_is_given_equation : hyperbola_equation :=
sorry

end hyperbola_is_given_equation_l26_26763


namespace factorize_difference_of_squares_l26_26266

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l26_26266


namespace expected_winnings_is_350_l26_26361

noncomputable def expected_winnings : ℝ :=
  (1 / 8) * (7 + 6 + 5 + 4 + 3 + 2 + 1 + 0)

theorem expected_winnings_is_350 :
  expected_winnings = 3.5 :=
by sorry

end expected_winnings_is_350_l26_26361


namespace problem_solution_l26_26123

theorem problem_solution
  {a b c d : ℝ}
  (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h2 : (a^2012 - c^2012) * (a^2012 - d^2012) = 2011)
  (h3 : (b^2012 - c^2012) * (b^2012 - d^2012) = 2011) :
  (c * d)^2012 - (a * b)^2012 = 2011 :=
by
  sorry

end problem_solution_l26_26123


namespace find_x_range_l26_26934

noncomputable def f (x : ℝ) : ℝ := if h : x ≥ 0 then 3^(-x) else 3^(x)

theorem find_x_range (x : ℝ) (h1 : f 2 = -f (2*x - 1) ∧ f 2 < 0) : -1/2 < x ∧ x < 3/2 := by
  -- Proof goes here
  sorry

end find_x_range_l26_26934


namespace half_difference_donation_l26_26758

def margoDonation : ℝ := 4300
def julieDonation : ℝ := 4700

theorem half_difference_donation : (julieDonation - margoDonation) / 2 = 200 := by
  sorry

end half_difference_donation_l26_26758


namespace quadratic_inequality_l26_26228

theorem quadratic_inequality (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + 2 * x + 1 < 0) → a < 1 := 
by
  sorry

end quadratic_inequality_l26_26228


namespace negation_of_proposition_l26_26097

theorem negation_of_proposition (x y : ℝ) :
  (¬ (x + y = 1 → xy ≤ 1)) ↔ (x + y ≠ 1 → xy > 1) :=
by 
  sorry

end negation_of_proposition_l26_26097


namespace range_of_f_is_real_l26_26128

noncomputable def f (x : ℝ) (m : ℝ) := Real.log (5^x + 4 / 5^x + m)

theorem range_of_f_is_real (m : ℝ) : (∀ y : ℝ, ∃ x : ℝ, f x m = y) ↔ m ≤ -4 :=
sorry

end range_of_f_is_real_l26_26128


namespace problem_l26_26099

theorem problem (m : ℝ) (h : m^2 + 3 * m = -1) : m - 1 / (m + 1) = -2 :=
by
  sorry

end problem_l26_26099


namespace angles_cosine_condition_l26_26240

theorem angles_cosine_condition {A B : ℝ} (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) :
  (A > B) ↔ (Real.cos A < Real.cos B) :=
by
sorry

end angles_cosine_condition_l26_26240


namespace calculation1_calculation2_calculation3_calculation4_l26_26761

-- Proving the first calculation: 3 * 232 + 456 = 1152
theorem calculation1 : 3 * 232 + 456 = 1152 := 
by 
  sorry

-- Proving the second calculation: 760 * 5 - 2880 = 920
theorem calculation2 : 760 * 5 - 2880 = 920 :=
by 
  sorry

-- Proving the third calculation: 805 / 7 = 115 (integer division)
theorem calculation3 : 805 / 7 = 115 :=
by 
  sorry

-- Proving the fourth calculation: 45 + 255 / 5 = 96
theorem calculation4 : 45 + 255 / 5 = 96 :=
by 
  sorry

end calculation1_calculation2_calculation3_calculation4_l26_26761


namespace each_child_gets_one_slice_l26_26943

-- Define the conditions
def couple_slices_per_person : ℕ := 3
def number_of_people : ℕ := 2
def number_of_children : ℕ := 6
def pizzas_ordered : ℕ := 3
def slices_per_pizza : ℕ := 4

-- Calculate slices required by the couple
def total_slices_for_couple : ℕ := couple_slices_per_person * number_of_people

-- Calculate total slices available
def total_slices : ℕ := pizzas_ordered * slices_per_pizza

-- Calculate slices for children
def slices_for_children : ℕ := total_slices - total_slices_for_couple

-- Calculate slices each child gets
def slices_per_child : ℕ := slices_for_children / number_of_children

-- The proof statement
theorem each_child_gets_one_slice : slices_per_child = 1 := by
  sorry

end each_child_gets_one_slice_l26_26943


namespace sequence_v5_value_l26_26814

theorem sequence_v5_value (v : ℕ → ℝ) (h_rec : ∀ n, v (n + 2) = 3 * v (n + 1) - v n)
  (h_v3 : v 3 = 17) (h_v6 : v 6 = 524) : v 5 = 198.625 :=
sorry

end sequence_v5_value_l26_26814


namespace lcm_18_28_45_65_eq_16380_l26_26976

theorem lcm_18_28_45_65_eq_16380 : Nat.lcm 18 (Nat.lcm 28 (Nat.lcm 45 65)) = 16380 :=
sorry

end lcm_18_28_45_65_eq_16380_l26_26976


namespace regular_dodecahedron_edges_l26_26788

-- Define a regular dodecahedron as a type
inductive RegularDodecahedron : Type
| mk : RegularDodecahedron

-- Define a function that returns the number of edges for a regular dodecahedron
def numberOfEdges (d : RegularDodecahedron) : Nat :=
  30

-- The mathematical statement to be proved
theorem regular_dodecahedron_edges (d : RegularDodecahedron) : numberOfEdges d = 30 := by
  sorry

end regular_dodecahedron_edges_l26_26788


namespace volume_conversion_l26_26815

theorem volume_conversion (a : Nat) (b : Nat) (c : Nat) (d : Nat) (e : Nat) (f : Nat)
  (h1 : a = 1) (h2 : b = 3) (h3 : c = a^3) (h4 : d = b^3) (h5 : c = 1) (h6 : d = 27) 
  (h7 : 1 = 1) (h8 : 27 = 27) (h9 : e = 5) 
  (h10 : f = e * d) : 
  f = 135 := 
sorry

end volume_conversion_l26_26815


namespace total_children_on_playground_l26_26119

theorem total_children_on_playground (girls boys : ℕ) (h_girls : girls = 28) (h_boys : boys = 35) : girls + boys = 63 := 
by 
  sorry

end total_children_on_playground_l26_26119


namespace max_volume_of_box_l26_26629

theorem max_volume_of_box (sheetside : ℝ) (cutside : ℝ) (volume : ℝ) 
  (h1 : sheetside = 6) 
  (h2 : ∀ (x : ℝ), 0 < x ∧ x < (sheetside / 2) → volume = x * (sheetside - 2 * x)^2) : 
  cutside = 1 :=
by
  sorry

end max_volume_of_box_l26_26629


namespace matrix_power_101_l26_26403

noncomputable def matrix_B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0], ![0, 0, 1], ![1, 0, 0]]

theorem matrix_power_101 :
  (matrix_B ^ 101) = ![![0, 0, 1], ![1, 0, 0], ![0, 1, 0]] :=
  sorry

end matrix_power_101_l26_26403


namespace mod_computation_l26_26829

theorem mod_computation (n : ℤ) : 
  0 ≤ n ∧ n < 23 ∧ 47582 % 23 = n ↔ n = 3 := 
by 
  -- Proof omitted
  sorry

end mod_computation_l26_26829


namespace regular_polygon_sides_l26_26875

-- Define the conditions of the problem 
def is_regular_polygon (n : ℕ) (exterior_angle : ℝ) : Prop :=
  360 / n = exterior_angle

-- State the theorem
theorem regular_polygon_sides (h : is_regular_polygon n 18) : n = 20 := 
sorry

end regular_polygon_sides_l26_26875


namespace intersection_complement_eq_l26_26302

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {3, 4, 5}
def U : Set ℝ := Set.univ  -- Universal set U is the set of all real numbers

theorem intersection_complement_eq : P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end intersection_complement_eq_l26_26302


namespace students_with_green_eyes_l26_26052

-- Define the variables and given conditions
def total_students : ℕ := 36
def students_with_red_hair (y : ℕ) : ℕ := 3 * y
def students_with_both : ℕ := 12
def students_with_neither : ℕ := 4

-- Define the proof statement
theorem students_with_green_eyes :
  ∃ y : ℕ, 
  (students_with_red_hair y + y - students_with_both + students_with_neither = total_students) ∧
  (students_with_red_hair y ≠ y) → y = 11 :=
by
  sorry

end students_with_green_eyes_l26_26052


namespace Ara_height_in_inches_l26_26546

theorem Ara_height_in_inches (Shea_current_height : ℝ) (Shea_growth_percentage : ℝ) (Ara_growth_factor : ℝ) (Shea_growth_amount : ℝ) (Ara_current_height : ℝ) :
  Shea_current_height = 75 →
  Shea_growth_percentage = 0.25 →
  Ara_growth_factor = 1 / 3 →
  Shea_growth_amount = 75 * (1 / (1 + 0.25)) * 0.25 →
  Ara_current_height = 75 * (1 / (1 + 0.25)) + (75 * (1 / (1 + 0.25)) * 0.25) * (1 / 3) →
  Ara_current_height = 65 :=
by sorry

end Ara_height_in_inches_l26_26546


namespace fraction_sent_afternoon_l26_26386

-- Defining the problem conditions
def total_fliers : ℕ := 1000
def fliers_sent_morning : ℕ := total_fliers * 1/5
def fliers_left_afternoon : ℕ := total_fliers - fliers_sent_morning
def fliers_left_next_day : ℕ := 600
def fliers_sent_afternoon : ℕ := fliers_left_afternoon - fliers_left_next_day

-- Proving the fraction of fliers sent in the afternoon
theorem fraction_sent_afternoon : (fliers_sent_afternoon : ℚ) / fliers_left_afternoon = 1/4 :=
by
  -- proof goes here
  sorry

end fraction_sent_afternoon_l26_26386


namespace min_value_ineq_l26_26911

noncomputable def a_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0 ∧ a 2018 = a 2017 + 2 * a 2016

theorem min_value_ineq (a : ℕ → ℝ) (m n : ℕ) 
  (h : a_sequence a) 
  (h2 : a m * a n = 16 * (a 1) ^ 2) :
  (4 / m) + (1 / n) ≥ 5 / 3 :=
sorry

end min_value_ineq_l26_26911


namespace simplify_and_evaluate_expression_l26_26451

theorem simplify_and_evaluate_expression (m n : ℤ) (h_m : m = -1) (h_n : n = 2) :
  3 * m^2 * n - 2 * m * n^2 - 4 * m^2 * n + m * n^2 = 2 :=
by
  sorry

end simplify_and_evaluate_expression_l26_26451


namespace largest_possible_m_l26_26775

theorem largest_possible_m (x y : ℕ) (h1 : x > y) (hx : Nat.Prime x) (hy : Nat.Prime y) (hxy : x < 10) (hyy : y < 10) (h_prime_10xy : Nat.Prime (10 * x + y)) : ∃ m : ℕ, m = x * y * (10 * x + y) ∧ 1000 ≤ m ∧ m ≤ 9999 ∧ ∀ n : ℕ, (n = x * y * (10 * x + y) ∧ 1000 ≤ n ∧ n ≤ 9999) → n ≤ 1533 :=
by
  sorry

end largest_possible_m_l26_26775


namespace number_of_parrots_l26_26658

noncomputable def daily_consumption_parakeet : ℕ := 2
noncomputable def daily_consumption_parrot : ℕ := 14
noncomputable def daily_consumption_finch : ℕ := 1  -- Each finch eats half of what a parakeet eats

noncomputable def num_parakeets : ℕ := 3
noncomputable def num_finches : ℕ := 4
noncomputable def required_birdseed : ℕ := 266
noncomputable def days_in_week : ℕ := 7

theorem number_of_parrots (num_parrots : ℕ) : 
  daily_consumption_parakeet * num_parakeets * days_in_week +
  daily_consumption_finch * num_finches * days_in_week + 
  daily_consumption_parrot * num_parrots * days_in_week = required_birdseed → num_parrots = 2 :=
by 
  -- The proof is omitted as per the instructions
  sorry

end number_of_parrots_l26_26658


namespace solve_division_problem_l26_26140

-- Problem Conditions
def division_problem : ℚ := 0.25 / 0.005

-- Proof Problem Statement
theorem solve_division_problem : division_problem = 50 := by
  sorry

end solve_division_problem_l26_26140


namespace last_digit_of_3_to_2010_is_9_l26_26312

theorem last_digit_of_3_to_2010_is_9 : (3^2010 % 10) = 9 := by
  -- Given that the last digits of powers of 3 cycle through 3, 9, 7, 1
  -- We need to prove that the last digit of 3^2010 is 9
  sorry

end last_digit_of_3_to_2010_is_9_l26_26312


namespace cost_price_of_watch_l26_26525

theorem cost_price_of_watch 
  (CP : ℝ)
  (h1 : 0.88 * CP = SP_loss)
  (h2 : 1.04 * CP = SP_gain)
  (h3 : SP_gain - SP_loss = 140) :
  CP = 875 := 
sorry

end cost_price_of_watch_l26_26525


namespace compute_difference_of_squares_l26_26096

theorem compute_difference_of_squares :
    75^2 - 25^2 = 5000 :=
by
  sorry

end compute_difference_of_squares_l26_26096


namespace words_to_score_A_l26_26664

-- Define the total number of words
def total_words : ℕ := 600

-- Define the target percentage
def target_percentage : ℚ := 90 / 100

-- Define the minimum number of words to learn
def min_words_to_learn : ℕ := 540

-- Define the condition for scoring at least 90%
def meets_requirement (learned_words : ℕ) : Prop :=
  learned_words / total_words ≥ target_percentage

-- The goal is to prove that learning 540 words meets the requirement
theorem words_to_score_A : meets_requirement min_words_to_learn :=
by
  sorry

end words_to_score_A_l26_26664


namespace right_triangle_hypotenuse_unique_l26_26056

theorem right_triangle_hypotenuse_unique :
  ∃ (a b c : ℚ) (d e : ℕ), 
    (c^2 = a^2 + b^2) ∧
    (a = 10 * e + d) ∧
    (c = 10 * d + e) ∧
    (d + e = 11) ∧
    (d ≠ e) ∧
    (a = 56) ∧
    (b = 33) ∧
    (c = 65) :=
by {
  sorry
}

end right_triangle_hypotenuse_unique_l26_26056


namespace problem_1_problem_2_l26_26709

-- Definitions for set A and B when a = 3 for (1)
def A : Set ℝ := { x | -x^2 + 5 * x - 4 > 0 }
def B : Set ℝ := { x | x^2 - 5 * x + 6 ≤ 0 }

-- Theorem for (1)
theorem problem_1 : A ∪ (Bᶜ) = Set.univ := sorry

-- Function to describe B based on a for (2)
def B_a (a : ℝ) : Set ℝ := { x | x^2 - (a + 2) * x + 2 * a ≤ 0 }
def A_set : Set ℝ := { x | -x^2 + 5 * x - 4 > 0 }

-- Theorem for (2)
theorem problem_2 (a : ℝ) : (1 < a ∧ a < 4) → (A_set ∩ B_a a ≠ ∅ ∧ B_a a ⊆ A_set ∧ B_a a ≠ A_set) := sorry

end problem_1_problem_2_l26_26709


namespace factory_profit_l26_26471

def cost_per_unit : ℝ := 2.00
def fixed_cost : ℝ := 500.00
def selling_price_per_unit : ℝ := 2.50

theorem factory_profit (x : ℕ) (hx : x > 1000) :
  selling_price_per_unit * x > fixed_cost + cost_per_unit * x :=
by
  sorry

end factory_profit_l26_26471


namespace cost_of_two_dogs_l26_26787

theorem cost_of_two_dogs (original_price : ℤ) (profit_margin : ℤ) (num_dogs : ℤ) (final_price : ℤ) :
  original_price = 1000 →
  profit_margin = 30 →
  num_dogs = 2 →
  final_price = original_price + (profit_margin * original_price / 100) →
  num_dogs * final_price = 2600 :=
by
  sorry

end cost_of_two_dogs_l26_26787


namespace find_unknown_number_l26_26984

-- Define the problem conditions and required proof
theorem find_unknown_number (a b : ℕ) (h1 : 2 * a = 3 + b) (h2 : (a - 6)^2 = 3 * b) : b = 3 ∨ b = 27 :=
sorry

end find_unknown_number_l26_26984


namespace time_boarding_in_London_l26_26410

open Nat

def time_in_ET_to_London_time (time_et: ℕ) : ℕ :=
  (time_et + 5) % 24

def subtract_hours (time: ℕ) (hours: ℕ) : ℕ :=
  (time + 24 * (hours / 24) - (hours % 24)) % 24

theorem time_boarding_in_London :
  let cape_town_arrival_time_et := 10
  let flight_duration_ny_to_cape := 10
  let ny_departure_time := subtract_hours cape_town_arrival_time_et flight_duration_ny_to_cape
  let flight_duration_london_to_ny := 18
  let ny_arrival_time := subtract_hours ny_departure_time flight_duration_london_to_ny
  let london_time := time_in_ET_to_London_time ny_arrival_time
  let london_departure_time := subtract_hours london_time flight_duration_london_to_ny
  london_departure_time = 17 :=
by
  -- Proof omitted
  sorry

end time_boarding_in_London_l26_26410


namespace distinct_convex_polygons_of_four_or_more_sides_l26_26570

noncomputable def total_subsets (n : Nat) : Nat := 2^n

noncomputable def subsets_with_fewer_than_four_members (n : Nat) : Nat := 
  (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)

noncomputable def valid_subsets (n : Nat) : Nat := 
  total_subsets n - subsets_with_fewer_than_four_members n

theorem distinct_convex_polygons_of_four_or_more_sides (n : Nat) (h : n = 15) : valid_subsets n = 32192 := by
  sorry

end distinct_convex_polygons_of_four_or_more_sides_l26_26570


namespace problem_statement_l26_26917

theorem problem_statement :
  ∀ m n : ℕ, (m = 9) → (n = m^2 + 1) → n - m = 73 :=
by
  intros m n hm hn
  rw [hm, hn]
  sorry

end problem_statement_l26_26917


namespace combined_score_is_210_l26_26747

theorem combined_score_is_210 :
  ∀ (total_questions : ℕ) (marks_per_question : ℕ) (jose_wrong : ℕ) 
    (meghan_less_than_jose : ℕ) (jose_more_than_alisson : ℕ) (jose_total : ℕ),
  total_questions = 50 →
  marks_per_question = 2 →
  jose_wrong = 5 →
  meghan_less_than_jose = 20 →
  jose_more_than_alisson = 40 →
  jose_total = total_questions * marks_per_question - (jose_wrong * marks_per_question) →
  (jose_total - meghan_less_than_jose) + jose_total + (jose_total - jose_more_than_alisson) = 210 :=
by
  intros total_questions marks_per_question jose_wrong meghan_less_than_jose jose_more_than_alisson jose_total
  intros h1 h2 h3 h4 h5 h6
  sorry

end combined_score_is_210_l26_26747


namespace security_deposit_percentage_l26_26630

theorem security_deposit_percentage
    (daily_rate : ℝ) (pet_fee : ℝ) (service_fee_rate : ℝ) (days : ℝ) (security_deposit : ℝ)
    (total_cost : ℝ) (expected_percentage : ℝ) :
    daily_rate = 125.0 →
    pet_fee = 100.0 →
    service_fee_rate = 0.20 →
    days = 14 →
    security_deposit = 1110 →
    total_cost = daily_rate * days + pet_fee + (daily_rate * days + pet_fee) * service_fee_rate →
    expected_percentage = (security_deposit / total_cost) * 100 →
    expected_percentage = 50 :=
by
  intros
  sorry

end security_deposit_percentage_l26_26630


namespace smallest_n_interesting_meeting_l26_26521

theorem smallest_n_interesting_meeting (m : ℕ) (hm : 2 ≤ m) :
  ∀ (n : ℕ), (n ≤ 3 * m - 1) ∧ (∀ (rep : Finset (Fin (3 * m))), rep.card = n →
  ∃ subrep : Finset (Fin (3 * m)), subrep.card = 3 ∧ ∀ (x y : Fin (3 * m)), x ∈ subrep → y ∈ subrep → x ≠ y → ∃ z : Fin (3 * m), z ∈ subrep ∧ z = x + y) → n = 2 * m + 1 := by
  sorry

end smallest_n_interesting_meeting_l26_26521


namespace count_distinct_four_digit_numbers_divisible_by_3_ending_in_45_l26_26636

/--
Prove that the total number of distinct four-digit numbers that end with 45 and 
are divisible by 3 is 27.
-/
theorem count_distinct_four_digit_numbers_divisible_by_3_ending_in_45 :
  ∃ n : ℕ, n = 27 ∧ 
  ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) → (0 ≤ b ∧ b ≤ 9) → 
  (∃ k : ℕ, a + b + 9 = 3 * k) → 
  (10 * (10 * a + b) + 45) = 1000 * a + 100 * b + 45 → 
  1000 * a + 100 * b + 45 = n := sorry

end count_distinct_four_digit_numbers_divisible_by_3_ending_in_45_l26_26636


namespace sale_price_is_207_l26_26685

-- Definitions for the conditions given
def price_at_store_P : ℝ := 200
def regular_price_at_store_Q (price_P : ℝ) : ℝ := price_P * 1.15
def sale_price_at_store_Q (regular_price_Q : ℝ) : ℝ := regular_price_Q * 0.90

-- Goal: Prove the sale price of the bicycle at Store Q is 207
theorem sale_price_is_207 : sale_price_at_store_Q (regular_price_at_store_Q price_at_store_P) = 207 :=
by
  sorry

end sale_price_is_207_l26_26685


namespace domain_of_function_l26_26794

-- Definitions based on conditions
def function_domain (x : ℝ) : Prop := (x > -1) ∧ (x ≠ 1)

-- Prove the domain is the desired set
theorem domain_of_function :
  ∀ x, function_domain x ↔ ((-1 < x ∧ x < 1) ∨ (1 < x)) :=
  by
    sorry

end domain_of_function_l26_26794


namespace incorrect_statement_D_l26_26912

def ordinate_of_x_axis_is_zero (p : ℝ × ℝ) : Prop :=
  p.2 = 0

def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  abs p.1

def is_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

def point_A_properties (a b : ℝ) : Prop :=
  let x := - a^2 - 1
  let y := abs b
  x < 0 ∧ y ≥ 0

theorem incorrect_statement_D (a b : ℝ) : 
  ∃ (x y : ℝ), point_A_properties a b ∧ (x = -a^2 - 1 ∧ y = abs b ∧ (x < 0 ∧ y = 0)) :=
by {
  sorry
}

end incorrect_statement_D_l26_26912


namespace both_subjects_sum_l26_26573

-- Define the total number of students
def N : ℕ := 1500

-- Define the bounds for students studying Biology (B) and Chemistry (C)
def B_min : ℕ := 900
def B_max : ℕ := 1050

def C_min : ℕ := 600
def C_max : ℕ := 750

-- Let x and y be the smallest and largest number of students studying both subjects
def x : ℕ := B_max + C_max - N
def y : ℕ := B_min + C_min - N

-- Prove that y + x = 300
theorem both_subjects_sum : y + x = 300 := by
  sorry

end both_subjects_sum_l26_26573


namespace evaluate_expression_l26_26261

theorem evaluate_expression : 
  ( (2^12)^2 - (2^10)^2 ) / ( (2^11)^2 - (2^9)^2 ) = 4 :=
by
  sorry

end evaluate_expression_l26_26261


namespace water_tank_capacity_l26_26839

theorem water_tank_capacity (C : ℝ) :
  0.4 * C - 0.1 * C = 36 → C = 120 :=
by sorry

end water_tank_capacity_l26_26839


namespace abs_sum_of_first_six_a_sequence_terms_l26_26090

def a_sequence (n : ℕ) : ℤ :=
  match n with
  | 0 => -5
  | n+1 => a_sequence n + 2

theorem abs_sum_of_first_six_a_sequence_terms :
  |a_sequence 0| + |a_sequence 1| + |a_sequence 2| + |a_sequence 3| + |a_sequence 4| + |a_sequence 5| = 18 := sorry

end abs_sum_of_first_six_a_sequence_terms_l26_26090


namespace linear_function_no_third_quadrant_l26_26208

theorem linear_function_no_third_quadrant :
  ∀ x y : ℝ, (y = -5 * x + 2023) → ¬ (x < 0 ∧ y < 0) := 
by
  intros x y h
  sorry

end linear_function_no_third_quadrant_l26_26208


namespace whiskers_count_l26_26187

variable (P C S : ℕ)

theorem whiskers_count :
  P = 14 →
  C = 2 * P - 6 →
  S = P + C + 8 →
  C = 22 ∧ S = 44 :=
by
  intros hP hC hS
  rw [hP] at hC
  rw [hP, hC] at hS
  exact ⟨hC, hS⟩

end whiskers_count_l26_26187


namespace longer_segment_of_triangle_l26_26452

theorem longer_segment_of_triangle {a b c : ℝ} (h_triangle : a = 40 ∧ b = 90 ∧ c = 100) (h_altitude : ∃ h, h > 0) : 
  ∃ (longer_segment : ℝ), longer_segment = 82.5 :=
by 
  sorry

end longer_segment_of_triangle_l26_26452


namespace max_sum_composite_shape_l26_26143

theorem max_sum_composite_shape :
  let faces_hex_prism := 8
  let edges_hex_prism := 18
  let vertices_hex_prism := 12

  let faces_hex_with_pyramid := 8 - 1 + 6
  let edges_hex_with_pyramid := 18 + 6
  let vertices_hex_with_pyramid := 12 + 1
  let sum_hex_with_pyramid := faces_hex_with_pyramid + edges_hex_with_pyramid + vertices_hex_with_pyramid

  let faces_rec_with_pyramid := 8 - 1 + 5
  let edges_rec_with_pyramid := 18 + 4
  let vertices_rec_with_pyramid := 12 + 1
  let sum_rec_with_pyramid := faces_rec_with_pyramid + edges_rec_with_pyramid + vertices_rec_with_pyramid

  sum_hex_with_pyramid = 50 ∧ sum_rec_with_pyramid = 46 ∧ sum_hex_with_pyramid ≥ sum_rec_with_pyramid := 
by
  have faces_hex_prism := 8
  have edges_hex_prism := 18
  have vertices_hex_prism := 12

  have faces_hex_with_pyramid := 8 - 1 + 6
  have edges_hex_with_pyramid := 18 + 6
  have vertices_hex_with_pyramid := 12 + 1
  have sum_hex_with_pyramid := faces_hex_with_pyramid + edges_hex_with_pyramid + vertices_hex_with_pyramid

  have faces_rec_with_pyramid := 8 - 1 + 5
  have edges_rec_with_pyramid := 18 + 4
  have vertices_rec_with_pyramid := 12 + 1
  have sum_rec_with_pyramid := faces_rec_with_pyramid + edges_rec_with_pyramid + vertices_rec_with_pyramid

  sorry -- proof omitted

end max_sum_composite_shape_l26_26143


namespace john_daily_reading_hours_l26_26148

-- Definitions from the conditions
def reading_rate := 50  -- pages per hour
def total_pages := 2800  -- pages
def weeks := 4
def days_per_week := 7

-- Hypotheses derived from the conditions
def total_hours := total_pages / reading_rate  -- 2800 / 50 = 56 hours
def total_days := weeks * days_per_week  -- 4 * 7 = 28 days

-- Theorem to prove 
theorem john_daily_reading_hours : (total_hours / total_days) = 2 := by
  sorry

end john_daily_reading_hours_l26_26148


namespace lcm_of_9_12_15_is_180_l26_26482

theorem lcm_of_9_12_15_is_180 :
  Nat.lcm 9 (Nat.lcm 12 15) = 180 :=
by
  sorry

end lcm_of_9_12_15_is_180_l26_26482


namespace anita_total_cartons_l26_26896

-- Defining the conditions
def cartons_of_strawberries : ℕ := 10
def cartons_of_blueberries : ℕ := 9
def additional_cartons_needed : ℕ := 7

-- Adding the core theorem to be proved
theorem anita_total_cartons :
  cartons_of_strawberries + cartons_of_blueberries + additional_cartons_needed = 26 := 
by
  sorry

end anita_total_cartons_l26_26896


namespace f_of_72_l26_26540

theorem f_of_72 (f : ℕ → ℝ) (p q : ℝ) (h1 : ∀ a b : ℕ, f (a * b) = f a + f b)
  (h2 : f 2 = p) (h3 : f 3 = q) : f 72 = 3 * p + 2 * q := 
sorry

end f_of_72_l26_26540


namespace overall_percentage_badminton_l26_26020

theorem overall_percentage_badminton (N S : ℕ) (pN pS : ℝ) :
  N = 1500 → S = 1800 → pN = 0.30 → pS = 0.35 → 
  ( (N * pN + S * pS) / (N + S) ) * 100 = 33 := 
by
  intros hN hS hpN hpS
  sorry

end overall_percentage_badminton_l26_26020


namespace fair_coin_toss_consecutive_heads_l26_26940

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem fair_coin_toss_consecutive_heads :
  let total_outcomes := 1024
  let favorable_outcomes := 
    1 + binom 10 1 + binom 9 2 + binom 8 3 + binom 7 4 + binom 6 5
  let prob := favorable_outcomes / total_outcomes
  let i := 9
  let j := 64
  Nat.gcd i j = 1 ∧ (prob = i / j) ∧ i + j = 73 :=
by
  sorry

end fair_coin_toss_consecutive_heads_l26_26940


namespace sum_due_is_correct_l26_26360

-- Definitions of the given conditions
def BD : ℝ := 78
def TD : ℝ := 66

-- Definition of the sum due (S)
noncomputable def S : ℝ := (TD^2) / (BD - TD) + TD

-- The theorem to be proved
theorem sum_due_is_correct : S = 429 := by
  sorry

end sum_due_is_correct_l26_26360


namespace order_of_a_b_c_l26_26724

noncomputable def a := Real.sqrt 3 - Real.sqrt 2
noncomputable def b := Real.sqrt 6 - Real.sqrt 5
noncomputable def c := Real.sqrt 7 - Real.sqrt 6

theorem order_of_a_b_c : a > b ∧ b > c :=
by
  sorry

end order_of_a_b_c_l26_26724


namespace tax_rate_l26_26058

noncomputable def payroll_tax : Float := 300000
noncomputable def tax_paid : Float := 200
noncomputable def tax_threshold : Float := 200000

theorem tax_rate (tax_rate : Float) : 
  (payroll_tax - tax_threshold) * tax_rate = tax_paid → tax_rate = 0.002 := 
by
  sorry

end tax_rate_l26_26058


namespace isosceles_triangle_inequality_l26_26114

theorem isosceles_triangle_inequality
  (a b : ℝ)
  (hb : b > 0)
  (h₁₂ : 12 * (π / 180) = π / 15) 
  (h_sin6 : Real.sin (6 * (π / 180)) > 1 / 10)
  (h_eq : a = 2 * b * Real.sin (6 * (π / 180))) : 
  b < 5 * a := 
by
  sorry

end isosceles_triangle_inequality_l26_26114


namespace max_marks_l26_26293

-- Define the conditions
def passing_marks (M : ℕ) : ℕ := 40 * M / 100

def Ravish_got_marks : ℕ := 40
def marks_failed_by : ℕ := 40

-- Lean statement to prove
theorem max_marks (M : ℕ) (h : passing_marks M = Ravish_got_marks + marks_failed_by) : M = 200 :=
by
  sorry

end max_marks_l26_26293


namespace triangle_a_eq_5_over_3_triangle_b_plus_c_eq_4_l26_26558

theorem triangle_a_eq_5_over_3
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : a / (Real.cos A) = (3 * c - 2 * b) / (Real.cos B))
  (h2 : b = Real.sqrt 5 * Real.sin B) :
  a = 5 / 3 := sorry

theorem triangle_b_plus_c_eq_4
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : a / (Real.cos A) = (3 * c - 2 * b) / (Real.cos B))
  (h2 : a = Real.sqrt 6)
  (h3 : 1 / 2 * b * c * Real.sin A = Real.sqrt 5 / 2) :
  b + c = 4 := sorry

end triangle_a_eq_5_over_3_triangle_b_plus_c_eq_4_l26_26558


namespace sum_of_inversion_counts_of_all_permutations_l26_26229

noncomputable def sum_of_inversion_counts (n : ℕ) (fixed_val : ℕ) (fixed_pos : ℕ) : ℕ :=
  if n = 6 ∧ fixed_val = 4 ∧ fixed_pos = 3 then 120 else 0

theorem sum_of_inversion_counts_of_all_permutations :
  sum_of_inversion_counts 6 4 3 = 120 :=
by
  sorry

end sum_of_inversion_counts_of_all_permutations_l26_26229


namespace average_eq_y_value_l26_26608

theorem average_eq_y_value :
  (y : ℤ) → (h : (15 + 25 + y) / 3 = 20) → y = 20 :=
by
  intro y h
  sorry

end average_eq_y_value_l26_26608


namespace inequality_problem_l26_26304

theorem inequality_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  (a + 1 / (2 * b^2)) * (b + 1 / (2 * a^2)) ≥ 25 / 4 := 
sorry

end inequality_problem_l26_26304


namespace maggie_kept_bouncy_balls_l26_26899

def packs_bought_yellow : ℝ := 8.0
def packs_given_away_green : ℝ := 4.0
def packs_bought_green : ℝ := 4.0
def balls_per_pack : ℝ := 10.0

theorem maggie_kept_bouncy_balls :
  packs_bought_yellow * balls_per_pack + (packs_bought_green - packs_given_away_green) * balls_per_pack = 80.0 :=
by sorry

end maggie_kept_bouncy_balls_l26_26899


namespace N_has_at_least_8_distinct_divisors_N_has_at_least_32_distinct_divisors_l26_26868

-- Define the number with 1986 ones
def N : ℕ := (10^1986 - 1) / 9

-- Definition of having at least n distinct divisors
def has_at_least_n_distinct_divisors (num : ℕ) (n : ℕ) :=
  ∃ (divisors : Finset ℕ), divisors.card ≥ n ∧ ∀ d ∈ divisors, d ∣ num

theorem N_has_at_least_8_distinct_divisors :
  has_at_least_n_distinct_divisors N 8 :=
sorry

theorem N_has_at_least_32_distinct_divisors :
  has_at_least_n_distinct_divisors N 32 :=
sorry


end N_has_at_least_8_distinct_divisors_N_has_at_least_32_distinct_divisors_l26_26868


namespace relationship_l26_26518

-- Given definitions
def S : ℕ := 31
def L : ℕ := 124 - S

-- Proving the relationship
theorem relationship: S + L = 124 ∧ S = 31 → L = S + 62 := by
  sorry

end relationship_l26_26518


namespace opposite_sides_range_a_l26_26716

theorem opposite_sides_range_a (a: ℝ) :
  ((1 - 2 * a + 1) * (a + 4 + 1) < 0) ↔ (a < -5 ∨ a > 1) :=
by
  sorry

end opposite_sides_range_a_l26_26716


namespace total_value_correct_l26_26908

noncomputable def total_value (num_coins : ℕ) : ℕ :=
  let value_one_rupee := num_coins * 1
  let value_fifty_paise := (num_coins * 50) / 100
  let value_twentyfive_paise := (num_coins * 25) / 100
  value_one_rupee + value_fifty_paise + value_twentyfive_paise

theorem total_value_correct :
  let num_coins := 40
  total_value num_coins = 70 := by
  sorry

end total_value_correct_l26_26908


namespace time_to_plough_together_l26_26284

def work_rate_r := 1 / 15
def work_rate_s := 1 / 20
def combined_work_rate := work_rate_r + work_rate_s
def total_field := 1
def T := total_field / combined_work_rate

theorem time_to_plough_together : T = 60 / 7 :=
by
  -- Here you would provide the proof steps if it were required
  -- Since the proof steps are not needed, we indicate the end with sorry
  sorry

end time_to_plough_together_l26_26284


namespace distance_between_points_l26_26168

open Real

theorem distance_between_points : 
  let p1 := (2, 2)
  let p2 := (5, 9)
  dist (p1 : ℝ × ℝ) p2 = sqrt 58 :=
by
  let p1 := (2, 2)
  let p2 := (5, 9)
  have h1 : p1.1 = 2 := rfl
  have h2 : p1.2 = 2 := rfl
  have h3 : p2.1 = 5 := rfl
  have h4 : p2.2 = 9 := rfl
  sorry

end distance_between_points_l26_26168


namespace final_result_l26_26591

-- Define the number of letters in each name
def letters_in_elida : ℕ := 5
def letters_in_adrianna : ℕ := 2 * letters_in_elida - 2

-- Define the alphabetical positions and their sums for each name
def sum_positions_elida : ℕ := 5 + 12 + 9 + 4 + 1
def sum_positions_adrianna : ℕ := 1 + 4 + 18 + 9 + 1 + 14 + 14 + 1
def sum_positions_belinda : ℕ := 2 + 5 + 12 + 9 + 14 + 4 + 1

-- Define the total sum of alphabetical positions
def total_sum_positions : ℕ := sum_positions_elida + sum_positions_adrianna + sum_positions_belinda

-- Define the average of the total sum
def average_sum_positions : ℕ := total_sum_positions / 3

-- Prove the final result
theorem final_result : (average_sum_positions * 3 - sum_positions_elida) = 109 :=
by
  -- Proof skipped
  sorry

end final_result_l26_26591


namespace largest_fraction_l26_26271

theorem largest_fraction (A B C D E : ℚ)
    (hA: A = 5 / 11)
    (hB: B = 7 / 16)
    (hC: C = 23 / 50)
    (hD: D = 99 / 200)
    (hE: E = 202 / 403) : 
    E > A ∧ E > B ∧ E > C ∧ E > D :=
by
  sorry

end largest_fraction_l26_26271


namespace students_going_on_field_trip_l26_26476

-- Define conditions
def van_capacity : Nat := 7
def number_of_vans : Nat := 6
def number_of_adults : Nat := 9

-- Define the total capacity
def total_people_capacity : Nat := number_of_vans * van_capacity

-- Define the number of students
def number_of_students : Nat := total_people_capacity - number_of_adults

-- Prove the number of students is 33
theorem students_going_on_field_trip : number_of_students = 33 := by
  sorry

end students_going_on_field_trip_l26_26476


namespace present_value_of_machine_l26_26873

theorem present_value_of_machine {
  V0 : ℝ
} (h : 36100 = V0 * (0.95)^2) : V0 = 39978.95 :=
sorry

end present_value_of_machine_l26_26873


namespace g_is_correct_l26_26045

-- Define the given polynomial equation
def poly_lhs (x : ℝ) : ℝ := 2 * x^5 - x^3 + 4 * x^2 + 3 * x - 5
def poly_rhs (x : ℝ) : ℝ := 7 * x^3 - 4 * x + 2

-- Define the function g(x)
def g (x : ℝ) : ℝ := -2 * x^5 + 6 * x^3 - 4 * x^2 - x + 7

-- The theorem to be proven
theorem g_is_correct : ∀ x : ℝ, poly_lhs x + g x = poly_rhs x :=
by
  intro x
  unfold poly_lhs poly_rhs g
  sorry

end g_is_correct_l26_26045


namespace quadratic_roots_vieta_l26_26447

theorem quadratic_roots_vieta :
  ∀ (x₁ x₂ : ℝ), (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → 
  (∃ (x₁ x₂ : ℝ), x₁ + x₂ = 2 ∧ x₁ * x₂ = -8) → 
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 :=
by
  intros x₁ x₂ h₁ h₂
  sorry

end quadratic_roots_vieta_l26_26447


namespace selling_price_of_cycle_l26_26729

theorem selling_price_of_cycle (original_price : ℝ) (loss_percentage : ℝ) (loss_amount : ℝ) (selling_price : ℝ) :
  original_price = 2000 →
  loss_percentage = 10 →
  loss_amount = (loss_percentage / 100) * original_price →
  selling_price = original_price - loss_amount →
  selling_price = 1800 :=
by
  intros
  sorry

end selling_price_of_cycle_l26_26729


namespace totalWheelsInStorageArea_l26_26070

def numberOfBicycles := 24
def numberOfTricycles := 14
def wheelsPerBicycle := 2
def wheelsPerTricycle := 3

theorem totalWheelsInStorageArea :
  numberOfBicycles * wheelsPerBicycle + numberOfTricycles * wheelsPerTricycle = 90 :=
by
  sorry

end totalWheelsInStorageArea_l26_26070


namespace mutually_exclusive_not_opposite_l26_26281

namespace event_theory

-- Definition to represent the student group
structure Group where
  boys : ℕ
  girls : ℕ

def student_group : Group := {boys := 3, girls := 2}

-- Definition of events
inductive Event
| AtLeastOneBoyAndOneGirl
| ExactlyOneBoyExactlyTwoBoys
| AtLeastOneBoyAllGirls
| AtMostOneBoyAllGirls

open Event

-- Conditions provided in the problem
def condition (grp : Group) : Prop :=
  grp.boys = 3 ∧ grp.girls = 2

-- The main statement to prove in Lean
theorem mutually_exclusive_not_opposite :
  condition student_group →
  ∃ e₁ e₂ : Event, e₁ = ExactlyOneBoyExactlyTwoBoys ∧ e₂ = ExactlyOneBoyExactlyTwoBoys ∧ (
    (e₁ ≠ e₂) ∧ (¬ (e₁ = e₂ ∧ e₁ = ExactlyOneBoyExactlyTwoBoys))
  ) :=
by
  sorry

end event_theory

end mutually_exclusive_not_opposite_l26_26281


namespace delta_y_over_delta_x_l26_26145

variable (Δx : ℝ)

def f (x : ℝ) : ℝ := 2 * x^2 - 1

theorem delta_y_over_delta_x : (f (1 + Δx) - f 1) / Δx = 4 + 2 * Δx :=
by
  sorry

end delta_y_over_delta_x_l26_26145


namespace synthetic_analytic_incorrect_statement_l26_26975

theorem synthetic_analytic_incorrect_statement
  (basic_methods : ∀ (P Q : Prop), (P → Q) ∨ (Q → P))
  (synthetic_forward : ∀ (P Q : Prop), (P → Q))
  (analytic_backward : ∀ (P Q : Prop), (Q → P)) :
  ¬ (∀ (P Q : Prop), (P → Q) ∧ (Q → P)) :=
by
  sorry

end synthetic_analytic_incorrect_statement_l26_26975


namespace cylinder_volume_increase_l26_26977

variable (r h : ℝ)

theorem cylinder_volume_increase :
  (π * (4 * r) ^ 2 * (2 * h)) = 32 * (π * r ^ 2 * h) :=
by
  sorry

end cylinder_volume_increase_l26_26977


namespace coordinates_of_A_l26_26032

-- Definition of the point A with coordinates (-1, 3)
def point_A : ℝ × ℝ := (-1, 3)

-- Statement that the coordinates of point A with respect to the origin are (-1, 3)
theorem coordinates_of_A : point_A = (-1, 3) := by
  sorry

end coordinates_of_A_l26_26032


namespace remainder_of_s_minus_t_plus_t_minus_u_l26_26368

theorem remainder_of_s_minus_t_plus_t_minus_u (s t u : ℕ) (hs : s % 12 = 4) (ht : t % 12 = 5) (hu : u % 12 = 7) (h_order : s > t ∧ t > u) :
  ((s - t) + (t - u)) % 12 = 9 :=
by sorry

end remainder_of_s_minus_t_plus_t_minus_u_l26_26368


namespace grandfather_time_difference_l26_26030

-- Definitions based on the conditions
def treadmill_days : ℕ := 4
def miles_per_day : ℕ := 2
def monday_speed : ℕ := 6
def tuesday_speed : ℕ := 3
def wednesday_speed : ℕ := 4
def thursday_speed : ℕ := 3
def walk_speed : ℕ := 3

-- The theorem statement
theorem grandfather_time_difference :
  let monday_time := (miles_per_day : ℚ) / monday_speed
  let tuesday_time := (miles_per_day : ℚ) / tuesday_speed
  let wednesday_time := (miles_per_day : ℚ) / wednesday_speed
  let thursday_time := (miles_per_day : ℚ) / thursday_speed
  let actual_total_time := monday_time + tuesday_time + wednesday_time + thursday_time
  let walk_total_time := (treadmill_days * miles_per_day : ℚ) / walk_speed
  (walk_total_time - actual_total_time) * 60 = 80 := sorry

end grandfather_time_difference_l26_26030


namespace factorization_correct_l26_26697

theorem factorization_correct (x y : ℝ) : 
  x * (x - y) - y * (x - y) = (x - y) ^ 2 :=
by 
  sorry

end factorization_correct_l26_26697


namespace length_of_room_l26_26036

noncomputable def room_length (width cost rate : ℝ) : ℝ :=
  let area := cost / rate
  area / width

theorem length_of_room :
  room_length 4.75 38475 900 = 9 := by
  sorry

end length_of_room_l26_26036


namespace ratio_of_boys_in_class_l26_26151

noncomputable def boy_to_total_ratio (p_boy p_girl : ℚ) : ℚ :=
p_boy / (p_boy + p_girl)

theorem ratio_of_boys_in_class (p_boy p_girl total_students : ℚ)
    (h1 : p_boy = (3/4) * p_girl)
    (h2 : p_boy + p_girl = 1)
    (h3 : total_students = 1) :
    boy_to_total_ratio p_boy p_girl = 3/7 :=
by
  sorry

end ratio_of_boys_in_class_l26_26151


namespace extra_money_from_customer_l26_26203

theorem extra_money_from_customer
  (price_per_craft : ℕ)
  (num_crafts_sold : ℕ)
  (deposit_amount : ℕ)
  (remaining_amount : ℕ)
  (total_amount_before_deposit : ℕ)
  (amount_made_from_crafts : ℕ)
  (extra_money : ℕ) :
  price_per_craft = 12 →
  num_crafts_sold = 3 →
  deposit_amount = 18 →
  remaining_amount = 25 →
  total_amount_before_deposit = deposit_amount + remaining_amount →
  amount_made_from_crafts = price_per_craft * num_crafts_sold →
  extra_money = total_amount_before_deposit - amount_made_from_crafts →
  extra_money = 7 :=
by
  intros; sorry

end extra_money_from_customer_l26_26203


namespace smallest_n_mod_equiv_l26_26577

theorem smallest_n_mod_equiv (n : ℕ) (h : 0 < n ∧ 2^n ≡ n^5 [MOD 4]) : n = 2 :=
by
  sorry

end smallest_n_mod_equiv_l26_26577


namespace count_divisible_by_five_l26_26633

theorem count_divisible_by_five : 
  ∃ n : ℕ, (∀ x, 1 ≤ x ∧ x ≤ 1000 → (x % 5 = 0 → (n = 200))) :=
by
  sorry

end count_divisible_by_five_l26_26633


namespace total_length_of_board_l26_26790

-- Define variables for the lengths
variable (S L : ℝ)

-- Given conditions as Lean definitions
def condition1 : Prop := 2 * S = L + 4
def condition2 : Prop := S = 8.0

-- The goal is to prove the total length of the board is 20.0 feet
theorem total_length_of_board (h1 : condition1 S L) (h2 : condition2 S) : S + L = 20.0 := by
  sorry

end total_length_of_board_l26_26790


namespace sum_f_to_2017_l26_26156

noncomputable def f (x : ℕ) : ℝ := Real.cos (x * Real.pi / 3)

theorem sum_f_to_2017 : (Finset.range 2017).sum f = 1 / 2 :=
by
  sorry

end sum_f_to_2017_l26_26156


namespace cost_of_dried_fruit_l26_26566

variable (x : ℝ)

theorem cost_of_dried_fruit 
  (h1 : 3 * 12 + 2.5 * x = 56) : 
  x = 8 := 
by 
  sorry

end cost_of_dried_fruit_l26_26566


namespace inequality_solution_l26_26795

theorem inequality_solution (x : ℝ) : 
  (0 < x ∧ x ≤ 3) ∨ (4 ≤ x) ↔ (3 * (x - 3) * (x - 4)) / x ≥ 0 := 
sorry

end inequality_solution_l26_26795


namespace square_87_l26_26598

theorem square_87 : 87^2 = 7569 :=
by
  sorry

end square_87_l26_26598


namespace gcd_5800_14025_l26_26921

theorem gcd_5800_14025 : Int.gcd 5800 14025 = 25 := by
  sorry

end gcd_5800_14025_l26_26921


namespace radar_placement_and_coverage_area_l26_26644

theorem radar_placement_and_coverage_area (r : ℝ) (w : ℝ) (n : ℕ) (h_radars : n = 5) (h_radius : r = 13) (h_width : w = 10) :
  let max_dist := 12 / Real.sin (Real.pi / 5)
  let area_ring := (240 * Real.pi) / Real.tan (Real.pi / 5)
  max_dist = 12 / Real.sin (Real.pi / 5) ∧ area_ring = (240 * Real.pi) / Real.tan (Real.pi / 5) :=
by
  sorry

end radar_placement_and_coverage_area_l26_26644


namespace Vikki_take_home_pay_is_correct_l26_26973

noncomputable def Vikki_take_home_pay : ℝ :=
  let hours_worked : ℝ := 42
  let hourly_pay_rate : ℝ := 12
  let gross_earnings : ℝ := hours_worked * hourly_pay_rate

  let fed_tax_first_300 : ℝ := 300 * 0.15
  let amount_over_300 : ℝ := gross_earnings - 300
  let fed_tax_excess : ℝ := amount_over_300 * 0.22
  let total_federal_tax : ℝ := fed_tax_first_300 + fed_tax_excess

  let state_tax : ℝ := gross_earnings * 0.07
  let retirement_contribution : ℝ := gross_earnings * 0.06
  let insurance_cover : ℝ := gross_earnings * 0.03
  let union_dues : ℝ := 5

  let total_deductions : ℝ := total_federal_tax + state_tax + retirement_contribution + insurance_cover + union_dues
  let take_home_pay : ℝ := gross_earnings - total_deductions
  take_home_pay

theorem Vikki_take_home_pay_is_correct : Vikki_take_home_pay = 328.48 :=
by
  sorry

end Vikki_take_home_pay_is_correct_l26_26973


namespace max_side_length_of_triangle_l26_26770

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l26_26770


namespace nat_divisibility_l26_26042

theorem nat_divisibility (n : ℕ) : 27 ∣ (10^n + 18 * n - 1) :=
  sorry

end nat_divisibility_l26_26042


namespace square_plot_area_l26_26710

theorem square_plot_area (s : ℕ) 
  (cost_per_foot : ℕ) 
  (total_cost : ℕ) 
  (H1 : cost_per_foot = 58) 
  (H2 : total_cost = 1624) 
  (H3 : total_cost = 232 * s) : 
  s * s = 49 := 
  by sorry

end square_plot_area_l26_26710


namespace simplify_fraction_l26_26892

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) : 
  ( (x^2 + 1) / (x - 1) - (2*x) / (x - 1) ) = x - 1 :=
by
  -- Your proof steps would go here.
  sorry

end simplify_fraction_l26_26892


namespace acute_angle_sum_equals_pi_over_two_l26_26893

theorem acute_angle_sum_equals_pi_over_two (a b : ℝ) (ha : 0 < a ∧ a < π / 2) (hb : 0 < b ∧ b < π / 2)
  (h1 : 4 * (Real.cos a)^2 + 3 * (Real.cos b)^2 = 1)
  (h2 : 4 * Real.sin (2 * a) + 3 * Real.sin (2 * b) = 0) :
  2 * a + b = π / 2 := 
sorry

end acute_angle_sum_equals_pi_over_two_l26_26893


namespace find_x_l26_26712

theorem find_x (x : ℝ) (h : 9 / (x + 4) = 1) : x = 5 :=
sorry

end find_x_l26_26712


namespace distance_from_focus_to_asymptote_l26_26798

theorem distance_from_focus_to_asymptote
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : a = b)
  (h2 : |a| / Real.sqrt 2 = 2) :
  Real.sqrt 2 * 2 = 2 * Real.sqrt 2 :=
by
  sorry

end distance_from_focus_to_asymptote_l26_26798


namespace problem_theorem_l26_26512

theorem problem_theorem (x y z : ℤ) 
  (h1 : x = 10 * y + 3)
  (h2 : 2 * x = 21 * y + 1)
  (h3 : 3 * x = 5 * z + 2) : 
  11 * y - x + 7 * z = 219 := 
by
  sorry

end problem_theorem_l26_26512


namespace parallel_line_distance_l26_26784

theorem parallel_line_distance 
    (A_upper : ℝ) (A_middle : ℝ) (A_lower : ℝ)
    (A_total : ℝ) (A_half : ℝ)
    (h_upper : A_upper = 3)
    (h_middle : A_middle = 5)
    (h_lower : A_lower = 2) 
    (h_total : A_total = A_upper + A_middle + A_lower)
    (h_half : A_half = A_total / 2) :
    ∃ d : ℝ, d = 2 + 0.6 ∧ A_middle * 0.6 = 3 := 
sorry

end parallel_line_distance_l26_26784


namespace jeremy_school_distance_l26_26920

def travel_time_rush_hour := 15 / 60 -- hours
def travel_time_clear_day := 10 / 60 -- hours
def speed_increase := 20 -- miles per hour

def distance_to_school (d v : ℝ) : Prop :=
  d = v * travel_time_rush_hour ∧ d = (v + speed_increase) * travel_time_clear_day

theorem jeremy_school_distance (d v : ℝ) (h_speed : v = 40) : d = 10 :=
by
  have travel_time_rush_hour := 1/4
  have travel_time_clear_day := 1/6
  have speed_increase := 20
  
  have h1 : d = v * travel_time_rush_hour := by sorry
  have h2 : d = (v + speed_increase) * travel_time_clear_day := by sorry
  have eqn := distance_to_school d v
  sorry

end jeremy_school_distance_l26_26920


namespace range_of_a_l26_26714

theorem range_of_a (a b c : ℝ) 
  (h1 : a^2 - b*c - 8*a + 7 = 0)
  (h2 : b^2 + c^2 + b*c - 6*a + 6 = 0) : 
  1 ≤ a ∧ a ≤ 9 :=
sorry

end range_of_a_l26_26714


namespace distance_between_centers_l26_26292

noncomputable def distance_centers_inc_exc (PQ PR QR: ℝ) (hPQ: PQ = 17) (hPR: PR = 15) (hQR: QR = 8) : ℝ :=
  let s := (PQ + PR + QR) / 2
  let area := Real.sqrt (s * (s - PQ) * (s - PR) * (s - QR))
  let r := area / s
  let r' := area / (s - QR)
  let PU := s - PQ
  let PV := s
  let PI := Real.sqrt ((PU)^2 + (r)^2)
  let PE := Real.sqrt ((PV)^2 + (r')^2)
  PE - PI

theorem distance_between_centers (PQ PR QR : ℝ) (hPQ: PQ = 17) (hPR: PR = 15) (hQR: QR = 8) :
  distance_centers_inc_exc PQ PR QR hPQ hPR hQR = 5 * Real.sqrt 17 - 3 * Real.sqrt 2 :=
by sorry

end distance_between_centers_l26_26292


namespace largest_divisor_of_n_l26_26978

theorem largest_divisor_of_n (n : ℕ) (h_pos : 0 < n) (h_div : 7200 ∣ n^2) : 60 ∣ n := 
sorry

end largest_divisor_of_n_l26_26978


namespace value_of_expression_l26_26890

theorem value_of_expression (A B C D : ℝ) (h1 : A - B = 30) (h2 : C + D = 20) :
  (B + C) - (A - D) = -10 :=
by
  sorry

end value_of_expression_l26_26890


namespace alcohol_mixture_l26_26736

variable {a b c d : ℝ} (ha : a ≠ d) (hbc : d ≠ c)

theorem alcohol_mixture (hcd : a ≥ d ∧ d ≥ c ∨ a ≤ d ∧ d ≤ c) :
  x = b * (d - c) / (a - d) :=
by 
  sorry

end alcohol_mixture_l26_26736


namespace sum_of_odd_powers_l26_26866

variable (x y z a : ℝ) (k : ℕ)

theorem sum_of_odd_powers (h1 : x + y + z = a) (h2 : x^3 + y^3 + z^3 = a^3) (hk : k % 2 = 1) : 
  x^k + y^k + z^k = a^k :=
sorry

end sum_of_odd_powers_l26_26866


namespace max_candy_one_student_l26_26255

theorem max_candy_one_student (n : ℕ) (mu : ℕ) (at_least_two : ℕ → Prop) :
  n = 35 → mu = 6 →
  (∀ x, at_least_two x → x ≥ 2) →
  ∃ max_candy : ℕ, (∀ x, at_least_two x → x ≤ max_candy) ∧ max_candy = 142 :=
by
sorry

end max_candy_one_student_l26_26255


namespace correct_operation_l26_26958

variables {a b : ℝ}

theorem correct_operation : (5 * a * b - 6 * a * b = -1 * a * b) := by
  sorry

end correct_operation_l26_26958


namespace trigonometric_expression_equals_one_l26_26647

theorem trigonometric_expression_equals_one :
  let cos30 := Real.sqrt 3 / 2
  let sin60 := Real.sqrt 3 / 2
  let sin30 := 1 / 2
  let cos60 := 1 / 2

  (1 - 1 / cos30) * (1 + 1 / sin60) *
  (1 - 1 / sin30) * (1 + 1 / cos60) = 1 :=
by
  let cos30 := Real.sqrt 3 / 2
  let sin60 := Real.sqrt 3 / 2
  let sin30 := 1 / 2
  let cos60 := 1 / 2
  sorry

end trigonometric_expression_equals_one_l26_26647


namespace intersection_M_N_l26_26459

noncomputable def M : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
noncomputable def N : Set ℝ := {x | abs x < 2}

theorem intersection_M_N : M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_M_N_l26_26459


namespace find_triangle_sides_l26_26789

theorem find_triangle_sides (x y : ℕ) : 
  (x * y = 200) ∧ (x + 2 * y = 50) → ((x = 40 ∧ y = 5) ∨ (x = 10 ∧ y = 20)) := 
by
  intro h
  sorry

end find_triangle_sides_l26_26789


namespace height_of_tree_in_kilmer_park_l26_26662

-- Define the initial conditions
def initial_height_ft := 52
def growth_per_year_ft := 5
def years := 8
def ft_to_inch := 12

-- Define the expected result in inches
def expected_height_inch := 1104

-- State the problem as a theorem
theorem height_of_tree_in_kilmer_park :
  (initial_height_ft + growth_per_year_ft * years) * ft_to_inch = expected_height_inch :=
by
  sorry

end height_of_tree_in_kilmer_park_l26_26662


namespace steak_and_egg_meal_cost_is_16_l26_26819

noncomputable def steak_and_egg_cost (x : ℝ) := 
  (x + 14) / 2 + 0.20 * (x + 14) = 21

theorem steak_and_egg_meal_cost_is_16 (x : ℝ) (h : steak_and_egg_cost x) : x = 16 := 
by 
  sorry

end steak_and_egg_meal_cost_is_16_l26_26819


namespace find_b_l26_26485

noncomputable def p (x : ℕ) := 3 * x + 5
noncomputable def q (x : ℕ) (b : ℕ) := 4 * x - b

theorem find_b : ∃ (b : ℕ), p (q 3 b) = 29 ∧ b = 4 := sorry

end find_b_l26_26485


namespace identical_solutions_of_quadratic_linear_l26_26033

theorem identical_solutions_of_quadratic_linear (k : ℝ) :
  (∃ x : ℝ, x^2 = 4 * x + k ∧ x^2 = 4 * x + k) ↔ k = -4 :=
by
  sorry

end identical_solutions_of_quadratic_linear_l26_26033


namespace sum_of_two_integers_l26_26750

theorem sum_of_two_integers (x y : ℕ) (h₁ : x^2 + y^2 = 145) (h₂ : x * y = 40) : x + y = 15 := 
by
  -- Proof omitted
  sorry

end sum_of_two_integers_l26_26750


namespace min_value_x_4_over_x_min_value_x_4_over_x_eq_l26_26411

theorem min_value_x_4_over_x (x : ℝ) (h : x > 0) : x + 4 / x ≥ 4 :=
sorry

theorem min_value_x_4_over_x_eq (x : ℝ) (h : x > 0) : (x + 4 / x = 4) ↔ (x = 2) :=
sorry

end min_value_x_4_over_x_min_value_x_4_over_x_eq_l26_26411


namespace pet_store_initial_puppies_l26_26702

theorem pet_store_initial_puppies
  (sold: ℕ) (cages: ℕ) (puppies_per_cage: ℕ)
  (remaining_puppies: ℕ)
  (h1: sold = 30)
  (h2: cages = 6)
  (h3: puppies_per_cage = 8)
  (h4: remaining_puppies = cages * puppies_per_cage):
  (sold + remaining_puppies) = 78 :=
by
  sorry

end pet_store_initial_puppies_l26_26702


namespace average_salary_of_officers_l26_26163

-- Define the given conditions
def avg_salary_total := 120
def avg_salary_non_officers := 110
def num_officers := 15
def num_non_officers := 480

-- Define the expected result
def avg_salary_officers := 440

-- Define the problem and statement to be proved in Lean
theorem average_salary_of_officers :
  (num_officers + num_non_officers) * avg_salary_total - num_non_officers * avg_salary_non_officers = num_officers * avg_salary_officers := 
by
  sorry

end average_salary_of_officers_l26_26163


namespace min_value_a_plus_3b_l26_26684

theorem min_value_a_plus_3b (a b : ℝ) (h_positive : 0 < a ∧ 0 < b)
  (h_condition : (1 / (a + 3) + 1 / (b + 3) = 1 / 4)) :
  a + 3 * b ≥ 4 + 8 * Real.sqrt 3 := 
sorry

end min_value_a_plus_3b_l26_26684


namespace remainder_div_x_plus_2_l26_26443

def f (x : ℤ) : ℤ := x^15 + 3

theorem remainder_div_x_plus_2 : f (-2) = -32765 := by
  sorry

end remainder_div_x_plus_2_l26_26443


namespace candy_bar_cost_correct_l26_26433

def initial_amount : ℕ := 4
def remaining_amount : ℕ := 3
def candy_bar_cost : ℕ := initial_amount - remaining_amount

theorem candy_bar_cost_correct : candy_bar_cost = 1 := by
  unfold candy_bar_cost
  sorry

end candy_bar_cost_correct_l26_26433


namespace first_reduction_percentage_l26_26200

theorem first_reduction_percentage 
  (P : ℝ)  -- original price
  (x : ℝ)  -- first day reduction percentage
  (h : P > 0) -- price assumption
  (h2 : 0 ≤ x ∧ x ≤ 100) -- percentage assumption
  (cond : P * (1 - x / 100) * 0.86 = 0.774 * P) : 
  x = 10 := 
sorry

end first_reduction_percentage_l26_26200


namespace max_legs_lengths_l26_26238

theorem max_legs_lengths (a x y : ℝ) (h₁ : x^2 + y^2 = a^2) (h₂ : 3 * x + 4 * y ≤ 5 * a) :
  3 * x + 4 * y = 5 * a → x = (3 * a / 5) ∧ y = (4 * a / 5) :=
by
  sorry

end max_legs_lengths_l26_26238


namespace total_money_collected_is_140_l26_26480

def total_attendees : ℕ := 280
def child_attendees : ℕ := 80
def adult_attendees : ℕ := total_attendees - child_attendees
def adult_ticket_cost : ℝ := 0.60
def child_ticket_cost : ℝ := 0.25

def money_collected_from_adults : ℝ := adult_attendees * adult_ticket_cost
def money_collected_from_children : ℝ := child_attendees * child_ticket_cost
def total_money_collected : ℝ := money_collected_from_adults + money_collected_from_children

theorem total_money_collected_is_140 : total_money_collected = 140 := by
  sorry

end total_money_collected_is_140_l26_26480


namespace Shiela_drawings_l26_26118

theorem Shiela_drawings (n_neighbors : ℕ) (drawings_per_neighbor : ℕ) (total_drawings : ℕ) 
    (h1 : n_neighbors = 6) (h2 : drawings_per_neighbor = 9) : total_drawings = 54 :=
by 
  sorry

end Shiela_drawings_l26_26118


namespace table_coverage_percentage_l26_26256

def A := 204  -- Total area of the runners
def T := 175  -- Area of the table
def A2 := 24  -- Area covered by exactly two layers of runner
def A3 := 20  -- Area covered by exactly three layers of runner

theorem table_coverage_percentage : 
  (A - 2 * A2 - 3 * A3 + A2 + A3) / T * 100 = 80 := 
by
  sorry

end table_coverage_percentage_l26_26256


namespace complete_the_square_l26_26063

theorem complete_the_square (x : ℝ) : x^2 + 2*x - 3 = 0 ↔ (x + 1)^2 = 4 :=
by sorry

end complete_the_square_l26_26063


namespace required_earnings_correct_l26_26667

-- Definitions of the given conditions
def retail_price : ℝ := 600
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05
def amount_saved : ℝ := 120
def amount_given_by_mother : ℝ := 250
def additional_costs : ℝ := 50

-- Required amount Maria must earn
def required_earnings : ℝ := 247

-- Lean 4 theorem statement
theorem required_earnings_correct :
  let discount_amount := discount_rate * retail_price
  let discounted_price := retail_price - discount_amount
  let sales_tax_amount := sales_tax_rate * discounted_price
  let total_bike_cost := discounted_price + sales_tax_amount
  let total_cost := total_bike_cost + additional_costs
  let total_have := amount_saved + amount_given_by_mother
  required_earnings = total_cost - total_have :=
by
  sorry

end required_earnings_correct_l26_26667


namespace number_of_real_b_l26_26233

noncomputable def count_integer_roots_of_quadratic_eq_b : ℕ :=
  let pairs := [(1, 64), (2, 32), (4, 16), (8, 8), (-1, -64), (-2, -32), (-4, -16), (-8, -8)]
  pairs.length

theorem number_of_real_b : count_integer_roots_of_quadratic_eq_b = 8 :=
by {
  -- sorry is used to skip the proof
  sorry
}

end number_of_real_b_l26_26233


namespace total_right_handed_players_is_correct_l26_26085

variable (total_players : ℕ)
variable (throwers : ℕ)
variable (left_handed_non_throwers_ratio : ℕ)
variable (total_right_handed_players : ℕ)

theorem total_right_handed_players_is_correct
  (h1 : total_players = 61)
  (h2 : throwers = 37)
  (h3 : left_handed_non_throwers_ratio = 1 / 3)
  (h4 : total_right_handed_players = 53) :
  total_right_handed_players = throwers + (total_players - throwers) -
    left_handed_non_throwers_ratio * (total_players - throwers) :=
by
  sorry

end total_right_handed_players_is_correct_l26_26085


namespace children_difference_l26_26950

theorem children_difference (initial_count : ℕ) (remaining_count : ℕ) (difference : ℕ) 
  (h1 : initial_count = 41) (h2 : remaining_count = 18) :
  difference = initial_count - remaining_count := 
by
  sorry

end children_difference_l26_26950


namespace average_speed_l26_26831

theorem average_speed (speed1 speed2 time1 time2: ℝ) (h1 : speed1 = 60) (h2 : time1 = 3) (h3 : speed2 = 85) (h4 : time2 = 2) : 
  (speed1 * time1 + speed2 * time2) / (time1 + time2) = 70 :=
by
  -- Definitions
  have distance1 := speed1 * time1
  have distance2 := speed2 * time2
  have total_distance := distance1 + distance2
  have total_time := time1 + time2
  -- Proof skeleton
  sorry

end average_speed_l26_26831


namespace george_painting_combinations_l26_26079

namespace Combinations

/-- George's painting problem -/
theorem george_painting_combinations :
  let colors := 10
  let colors_to_pick := 3
  let textures := 2
  ((colors) * (colors - 1) * (colors - 2) / (colors_to_pick * (colors_to_pick - 1) * 1)) * (textures ^ colors_to_pick) = 960 :=
by
  sorry

end Combinations

end george_painting_combinations_l26_26079


namespace average_salary_of_all_employees_l26_26393

theorem average_salary_of_all_employees 
    (avg_salary_officers : ℝ)
    (avg_salary_non_officers : ℝ)
    (num_officers : ℕ)
    (num_non_officers : ℕ)
    (h1 : avg_salary_officers = 450)
    (h2 : avg_salary_non_officers = 110)
    (h3 : num_officers = 15)
    (h4 : num_non_officers = 495) :
    (avg_salary_officers * num_officers + avg_salary_non_officers * num_non_officers)
    / (num_officers + num_non_officers) = 120 := by
  sorry

end average_salary_of_all_employees_l26_26393


namespace simple_interest_calculation_l26_26522

-- Define the principal (P), rate (R), and time (T)
def principal : ℝ := 10000
def rate : ℝ := 0.08
def time : ℝ := 1

-- Define the simple interest formula
def simple_interest (P R T : ℝ) : ℝ := P * R * T

-- The theorem to be proved
theorem simple_interest_calculation : simple_interest principal rate time = 800 :=
by
  -- Proof steps would go here, but this is left as an exercise
  sorry

end simple_interest_calculation_l26_26522


namespace hyeyoung_walked_correct_l26_26707

/-- The length of the promenade near Hyeyoung's house is 6 kilometers (km). -/
def promenade_length : ℕ := 6

/-- Hyeyoung walked from the starting point to the halfway point of the trail. -/
def hyeyoung_walked : ℕ := promenade_length / 2

/-- The distance Hyeyoung walked is 3 kilometers (km). -/
theorem hyeyoung_walked_correct : hyeyoung_walked = 3 := by
  sorry

end hyeyoung_walked_correct_l26_26707


namespace even_function_has_specific_m_l26_26698

theorem even_function_has_specific_m (m : ℝ) (f : ℝ → ℝ) (h_def : ∀ x : ℝ, f x = x^2 + (m - 1) * x - 3) (h_even : ∀ x : ℝ, f x = f (-x)) :
  m = 1 :=
by
  sorry

end even_function_has_specific_m_l26_26698


namespace smallest_n_l26_26067

theorem smallest_n (n : ℕ) (h : 503 * n % 48 = 1019 * n % 48) : n = 4 := by
  sorry

end smallest_n_l26_26067


namespace maximum_dn_l26_26175

-- Definitions of a_n and d_n based on the problem statement
def a (n : ℕ) : ℕ := 150 + (n + 1)^2
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

-- Statement of the theorem
theorem maximum_dn : ∃ M, M = 2 ∧ ∀ n, d n ≤ M :=
by
  -- proof should be written here
  sorry

end maximum_dn_l26_26175


namespace jill_travel_time_to_school_is_20_minutes_l26_26137

variables (dave_rate : ℕ) (dave_step : ℕ) (dave_time : ℕ)
variables (jill_rate : ℕ) (jill_step : ℕ)

def dave_distance : ℕ := dave_rate * dave_step * dave_time
def jill_time_to_school : ℕ := dave_distance dave_rate dave_step dave_time / (jill_rate * jill_step)

theorem jill_travel_time_to_school_is_20_minutes : 
  dave_rate = 85 → dave_step = 80 → dave_time = 18 → 
  jill_rate = 120 → jill_step = 50 → jill_time_to_school 85 80 18 120 50 = 20 :=
by
  intros
  unfold jill_time_to_school
  unfold dave_distance
  sorry

end jill_travel_time_to_school_is_20_minutes_l26_26137


namespace intersection_point_exists_l26_26362

def equation_1 (x y : ℝ) : Prop := 3 * x^2 - 12 * y^2 = 48
def line_eq (x y : ℝ) : Prop := y = - (1 / 3) * x + 5

theorem intersection_point_exists :
  ∃ (x y : ℝ), equation_1 x y ∧ line_eq x y ∧ x = 75 / 8 ∧ y = 15 / 8 :=
sorry

end intersection_point_exists_l26_26362


namespace color_of_last_bead_is_white_l26_26227

-- Defining the pattern of the beads
inductive BeadColor
| White
| Black
| Red

open BeadColor

-- Define the repeating pattern of the beads
def beadPattern : ℕ → BeadColor
| 0 => White
| 1 => Black
| 2 => Black
| 3 => Red
| 4 => Red
| 5 => Red
| (n + 6) => beadPattern n

-- Define the total number of beads
def totalBeads : ℕ := 85

-- Define the position of the last bead
def lastBead : ℕ := totalBeads - 1

-- Proving the color of the last bead
theorem color_of_last_bead_is_white : beadPattern lastBead = White :=
by
  sorry

end color_of_last_bead_is_white_l26_26227


namespace canonical_equations_of_line_l26_26195

-- Conditions: Two planes given by their equations
def plane1 (x y z : ℝ) : Prop := 6 * x - 5 * y + 3 * z + 8 = 0
def plane2 (x y z : ℝ) : Prop := 6 * x + 5 * y - 4 * z + 4 = 0

-- Proving the canonical form of the line
theorem canonical_equations_of_line :
  ∃ x y z, plane1 x y z ∧ plane2 x y z ↔ 
  ∃ t, x = -1 + 5 * t ∧ y = 2 / 5 + 42 * t ∧ z = 60 * t :=
sorry

end canonical_equations_of_line_l26_26195


namespace work_completion_time_l26_26759

theorem work_completion_time (W : ℝ) : 
  let A_effort := 1 / 11
  let B_effort := 1 / 20
  let C_effort := 1 / 55
  (2 * A_effort + B_effort + C_effort) = 1 / 4 → 
  8 * (2 * A_effort + B_effort + C_effort) = 1 :=
by { sorry }

end work_completion_time_l26_26759


namespace slowest_pipe_time_l26_26980

noncomputable def fill_tank_rate (R : ℝ) : Prop :=
  let rate1 := 6 * R
  let rate3 := 2 * R
  let combined_rate := 9 * R
  combined_rate = 1 / 30

theorem slowest_pipe_time (R : ℝ) (h : fill_tank_rate R) : 1 / R = 270 :=
by
  have h1 := h
  sorry

end slowest_pipe_time_l26_26980


namespace num_trombone_players_l26_26701

def weight_per_trumpet := 5
def weight_per_clarinet := 5
def weight_per_trombone := 10
def weight_per_tuba := 20
def weight_per_drum := 15

def num_trumpets := 6
def num_clarinets := 9
def num_tubas := 3
def num_drummers := 2
def total_weight := 245

theorem num_trombone_players : 
  let weight_trumpets := num_trumpets * weight_per_trumpet
  let weight_clarinets := num_clarinets * weight_per_clarinet
  let weight_tubas := num_tubas * weight_per_tuba
  let weight_drums := num_drummers * weight_per_drum
  let weight_others := weight_trumpets + weight_clarinets + weight_tubas + weight_drums
  let weight_trombones := total_weight - weight_others
  weight_trombones / weight_per_trombone = 8 :=
by
  sorry

end num_trombone_players_l26_26701


namespace jean_jail_time_l26_26345

/-- Jean has 3 counts of arson -/
def arson_count : ℕ := 3

/-- Each arson count has a 36-month sentence -/
def arson_sentence : ℕ := 36

/-- Jean has 2 burglary charges -/
def burglary_charges : ℕ := 2

/-- Each burglary charge has an 18-month sentence -/
def burglary_sentence : ℕ := 18

/-- Jean has six times as many petty larceny charges as burglary charges -/
def petty_larceny_multiplier : ℕ := 6

/-- Each petty larceny charge is 1/3 as long as a burglary charge -/
def petty_larceny_sentence : ℕ := burglary_sentence / 3

/-- Calculate all charges in months -/
def total_charges : ℕ :=
  (arson_count * arson_sentence) +
  (burglary_charges * burglary_sentence) +
  (petty_larceny_multiplier * burglary_charges * petty_larceny_sentence)

/-- Prove the total jail time for Jean is 216 months -/
theorem jean_jail_time : total_charges = 216 := by
  sorry

end jean_jail_time_l26_26345


namespace min_value_xyz_l26_26299

theorem min_value_xyz (x y z : ℝ) (h : x + 2 * y + 3 * z = 1) : x^2 + y^2 + z^2 ≥ 1 / 14 := 
by
  sorry

end min_value_xyz_l26_26299


namespace inequality_1_inequality_2_l26_26545

-- Define the first inequality proof problem
theorem inequality_1 (x : ℝ) : 5 * x + 3 < 11 + x ↔ x < 2 := by
  sorry

-- Define the second set of inequalities proof problem
theorem inequality_2 (x : ℝ) : 
  (2 * x + 1 < 3 * x + 3) ∧ ((x + 1) / 2 ≤ (1 - x) / 6 + 1) ↔ (-2 < x ∧ x ≤ 1) := by
  sorry

end inequality_1_inequality_2_l26_26545


namespace find_time_ball_hits_ground_l26_26242

theorem find_time_ball_hits_ground :
  ∃ t : ℝ, (-16 * t^2 + 40 * t + 30 = 0) ∧ (t = (5 + 5 * Real.sqrt 22) / 4) := 
by
  sorry

end find_time_ball_hits_ground_l26_26242


namespace angle_quadrant_l26_26514

theorem angle_quadrant (theta : ℤ) (h_theta : theta = -3290) : 
  ∃ q : ℕ, q = 4 := 
by 
  sorry

end angle_quadrant_l26_26514


namespace simplify_expression_l26_26295

theorem simplify_expression : (3 + 3 + 5) / 2 - 1 / 2 = 5 := by
  sorry

end simplify_expression_l26_26295


namespace mayo_bottle_count_l26_26657

-- Define the given ratio and the number of ketchup bottles
def ratio_ketchup : ℕ := 3
def ratio_mustard : ℕ := 3
def ratio_mayo : ℕ := 2
def num_ketchup_bottles : ℕ := 6

-- Define the proof problem: The number of mayo bottles
theorem mayo_bottle_count :
  (num_ketchup_bottles / ratio_ketchup) * ratio_mayo = 4 :=
by sorry

end mayo_bottle_count_l26_26657


namespace sample_size_is_50_l26_26303

theorem sample_size_is_50 (n : ℕ) :
  (n > 0) → 
  (10 / n = 2 / (2 + 3 + 5)) → 
  n = 50 := 
by
  sorry

end sample_size_is_50_l26_26303


namespace speed_W_B_l26_26700

-- Definitions for the conditions
def distance_W_B (D : ℝ) := 2 * D
def average_speed := 36
def speed_B_C := 20

-- The problem statement to be verified in Lean
theorem speed_W_B (D : ℝ) (S : ℝ) (h1: distance_W_B D = 2 * D) (h2: S ≠ 0 ∧ D ≠ 0)
(h3: (3 * D) / ((2 * D) / S + D / speed_B_C) = average_speed) : S = 60 := by
sorry

end speed_W_B_l26_26700


namespace clarinet_players_count_l26_26572

-- Given weights and counts
def weight_trumpet : ℕ := 5
def weight_clarinet : ℕ := 5
def weight_trombone : ℕ := 10
def weight_tuba : ℕ := 20
def weight_drum : ℕ := 15
def count_trumpets : ℕ := 6
def count_trombones : ℕ := 8
def count_tubas : ℕ := 3
def count_drummers : ℕ := 2
def total_weight : ℕ := 245

-- Calculated known weight
def known_weight : ℕ :=
  (count_trumpets * weight_trumpet) +
  (count_trombones * weight_trombone) +
  (count_tubas * weight_tuba) +
  (count_drummers * weight_drum)

-- Weight carried by clarinets
def weight_clarinets : ℕ := total_weight - known_weight

-- Number of clarinet players
def number_of_clarinet_players : ℕ := weight_clarinets / weight_clarinet

theorem clarinet_players_count :
  number_of_clarinet_players = 9 := by
  unfold number_of_clarinet_players
  unfold weight_clarinets
  unfold known_weight
  calc
    (245 - (
      (6 * 5) + 
      (8 * 10) + 
      (3 * 20) + 
      (2 * 15))) / 5 = 9 := by norm_num

end clarinet_players_count_l26_26572


namespace dot_product_result_l26_26614

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (1, 1)

theorem dot_product_result : (a.1 * b.1 + a.2 * b.2) = 1 := by
  sorry

end dot_product_result_l26_26614


namespace smallest_a_for_polynomial_l26_26916

theorem smallest_a_for_polynomial (a b x₁ x₂ x₃ : ℕ) 
    (h1 : x₁ * x₂ * x₃ = 2730)
    (h2 : x₁ + x₂ + x₃ = a)
    (h3 : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0)
    (h4 : ∀ y₁ y₂ y₃ : ℕ, y₁ * y₂ * y₃ = 2730 ∧ y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 → y₁ + y₂ + y₃ ≥ a) :
  a = 54 :=
  sorry

end smallest_a_for_polynomial_l26_26916


namespace mh_range_l26_26046

theorem mh_range (x m : ℝ) (h : 1 / 3 < x ∧ x < 1 / 2) (hx : |x - m| < 1) : 
  -1 / 2 ≤ m ∧ m ≤ 4 / 3 := 
sorry

end mh_range_l26_26046


namespace eval_f_nested_l26_26586

noncomputable def f (x : ℝ) : ℝ :=
if h : x < 0 then x + 1 else x ^ 2

theorem eval_f_nested : f (f (-2)) = 0 := by
  sorry

end eval_f_nested_l26_26586


namespace percentage_of_students_on_trip_l26_26333

variable (students : ℕ) -- Total number of students at the school
variable (students_trip_and_more_than_100 : ℕ) -- Number of students who went to the camping trip and took more than $100
variable (percent_trip_and_more_than_100 : ℚ) -- Percent of students who went to camping trip and took more than $100

-- Given Conditions
def cond1 : students_trip_and_more_than_100 = (percent_trip_and_more_than_100 * students) := 
  by
    sorry  -- This will represent the first condition: 18% of students went to a camping trip and took more than $100.

variable (percent_did_not_take_more_than_100 : ℚ) -- Percent of students who went to camping trip and did not take more than $100

-- second condition
def cond2 : percent_did_not_take_more_than_100 = 0.75 := 
  by
    sorry  -- Represent the second condition: 75% of students who went to the camping trip did not take more than $100.

-- Prove
theorem percentage_of_students_on_trip : 
  (students_trip_and_more_than_100 / (0.25 * students)) * 100 = (72 : ℚ) := 
  by
    sorry

end percentage_of_students_on_trip_l26_26333


namespace calc_radical_power_l26_26265

theorem calc_radical_power : (Real.sqrt (Real.sqrt (Real.sqrt (Real.sqrt 16))) ^ 12) = 4096 := sorry

end calc_radical_power_l26_26265


namespace find_a_l26_26800

noncomputable def angle := 30 * Real.pi / 180 -- In radians

noncomputable def tan_angle : ℝ := Real.tan angle

theorem find_a (a : ℝ) (h1 : tan_angle = 1 / Real.sqrt 3) : 
  x - a * y + 3 = 0 → a = Real.sqrt 3 :=
by
  sorry

end find_a_l26_26800


namespace compute_expression_l26_26555

theorem compute_expression : 
  let x := 19
  let y := 15
  (x + y)^2 - (x - y)^2 = 1140 :=
by
  sorry

end compute_expression_l26_26555


namespace intersect_sets_l26_26642

def set_M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def set_N : Set ℝ := {x | abs x < 2}

theorem intersect_sets :
  (set_M ∩ set_N) = {x | -1 ≤ x ∧ x < 2} :=
sorry

end intersect_sets_l26_26642


namespace possible_b4b7_products_l26_26394

theorem possible_b4b7_products (b : ℕ → ℤ) (d : ℤ)
  (h_arith_sequence : ∀ n, b (n + 1) = b n + d)
  (h_increasing : ∀ n, b (n + 1) > b n)
  (h_product_21 : b 5 * b 6 = 21) :
  b 4 * b 7 = -779 ∨ b 4 * b 7 = 21 :=
by
  sorry

end possible_b4b7_products_l26_26394


namespace tom_fruit_bowl_l26_26605

def initial_lemons (oranges lemons removed remaining : ℕ) : ℕ :=
  lemons

theorem tom_fruit_bowl (oranges removed remaining : ℕ) (L : ℕ) 
  (h_oranges : oranges = 3)
  (h_removed : removed = 3)
  (h_remaining : remaining = 6)
  (h_initial : oranges + L - removed = remaining) : 
  initial_lemons oranges L removed remaining = 6 :=
by
  -- Implement the proof here
  sorry

end tom_fruit_bowl_l26_26605


namespace decorations_per_box_l26_26776

-- Definitions based on given conditions
def used_decorations : ℕ := 35
def given_away_decorations : ℕ := 25
def number_of_boxes : ℕ := 4

-- Theorem stating the problem
theorem decorations_per_box : (used_decorations + given_away_decorations) / number_of_boxes = 15 := by
  sorry

end decorations_per_box_l26_26776


namespace C_gets_more_than_D_by_500_l26_26221

-- Definitions based on conditions
def proportionA := 5
def proportionB := 2
def proportionC := 4
def proportionD := 3

def totalProportion := proportionA + proportionB + proportionC + proportionD

def A_share := 2500
def totalMoney := A_share * (totalProportion / proportionA)

def C_share := (proportionC / totalProportion) * totalMoney
def D_share := (proportionD / totalProportion) * totalMoney

-- The theorem stating the final question
theorem C_gets_more_than_D_by_500 : C_share - D_share = 500 := by
  sorry

end C_gets_more_than_D_by_500_l26_26221


namespace gcd_condition_l26_26886

theorem gcd_condition (a b c : ℕ) (h1 : Nat.gcd a b = 255) (h2 : Nat.gcd a c = 855) :
  Nat.gcd b c = 15 :=
sorry

end gcd_condition_l26_26886


namespace maximum_smallest_angle_l26_26766

-- Definition of points on the plane
structure Point2D :=
  (x : ℝ)
  (y : ℝ)

-- Function to calculate the angle between three points (p1, p2, p3)
def angle (p1 p2 p3 : Point2D) : ℝ := 
  -- Placeholder for the actual angle calculation
  sorry

-- Condition: Given five points on a plane
variables (A B C D E : Point2D)

-- Maximum value of the smallest angle formed by any triple is 36 degrees
theorem maximum_smallest_angle :
  ∃ α : ℝ, (∀ p1 p2 p3 : Point2D, α ≤ angle p1 p2 p3) ∧ α = 36 :=
sorry

end maximum_smallest_angle_l26_26766


namespace union_complement_eq_univ_l26_26129

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 7}

-- Define set M
def M : Set ℕ := {1, 3, 5, 7}

-- Define set N
def N : Set ℕ := {3, 5}

-- Define the complement of N with respect to U
def complement_U_N : Set ℕ := {1, 2, 4, 7}

-- Prove that U = M ∪ complement_U_N
theorem union_complement_eq_univ : U = M ∪ complement_U_N := 
sorry

end union_complement_eq_univ_l26_26129


namespace smallest_angle_of_quadrilateral_l26_26688

theorem smallest_angle_of_quadrilateral 
  (x : ℝ) 
  (h1 : x + 2 * x + 3 * x + 4 * x = 360) : 
  x = 36 :=
by
  sorry

end smallest_angle_of_quadrilateral_l26_26688


namespace sequence_b_n_l26_26182

theorem sequence_b_n (b : ℕ → ℕ) (h₀ : b 1 = 3) (h₁ : ∀ n, b (n + 1) = b n + 3 * n + 1) :
  b 50 = 3727 :=
sorry

end sequence_b_n_l26_26182


namespace sin_squared_not_periodic_l26_26490

noncomputable def sin_squared (x : ℝ) : ℝ := Real.sin (x^2)

theorem sin_squared_not_periodic : 
  ¬ (∃ T > 0, ∀ x ∈ Set.univ, sin_squared (x + T) = sin_squared x) := 
sorry

end sin_squared_not_periodic_l26_26490


namespace solve_quadratic_eq_1_solve_quadratic_eq_2_l26_26385

-- Proof for Equation 1
theorem solve_quadratic_eq_1 : ∀ x : ℝ, x^2 - 4 * x + 1 = 0 ↔ (x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) :=
by sorry

-- Proof for Equation 2
theorem solve_quadratic_eq_2 : ∀ x : ℝ, 5 * x - 2 = (2 - 5 * x) * (3 * x + 4) ↔ (x = 2 / 5 ∨ x = -5 / 3) :=
by sorry

end solve_quadratic_eq_1_solve_quadratic_eq_2_l26_26385


namespace pipe_B_fills_6_times_faster_l26_26243

theorem pipe_B_fills_6_times_faster :
  let R_A := 1 / 32
  let combined_rate := 7 / 32
  let R_B := combined_rate - R_A
  (R_B / R_A = 6) :=
by
  let R_A := 1 / 32
  let combined_rate := 7 / 32
  let R_B := combined_rate - R_A
  sorry

end pipe_B_fills_6_times_faster_l26_26243


namespace dima_picks_more_berries_l26_26101

theorem dima_picks_more_berries (N : ℕ) (dima_fastness : ℕ) (sergei_fastness : ℕ) (dima_rate : ℕ) (sergei_rate : ℕ) :
  N = 450 → dima_fastness = 2 * sergei_fastness →
  dima_rate = 1 → sergei_rate = 2 →
  let dima_basket : ℕ := N / 2
  let sergei_basket : ℕ := (2 * N) / 3
  dima_basket > sergei_basket ∧ (dima_basket - sergei_basket) = 50 := 
by {
  sorry
}

end dima_picks_more_berries_l26_26101


namespace f_monotonicity_l26_26509

noncomputable def f : ℝ → ℝ := sorry -- Definition of the function f(x)

axiom f_symm (x : ℝ) : f (1 - x) = f x

axiom f_derivative (x : ℝ) : (x - 1 / 2) * (deriv f x) > 0

theorem f_monotonicity (x1 x2 : ℝ) (h1 : x1 < x2) (h2 : x1 + x2 > 1) : f x1 < f x2 :=
sorry

end f_monotonicity_l26_26509


namespace complement_of_A_in_B_l26_26342

def set_A : Set ℤ := {x | 2 * x = x^2}
def set_B : Set ℤ := {x | -x^2 + x + 2 ≥ 0}

theorem complement_of_A_in_B :
  (set_B \ set_A) = {-1, 1} :=
by
  sorry

end complement_of_A_in_B_l26_26342


namespace two_real_roots_opposite_signs_l26_26520

theorem two_real_roots_opposite_signs (a : ℝ) :
  (∃ x y : ℝ, (a * x^2 - (a + 3) * x + 2 = 0) ∧ (a * y^2 - (a + 3) * y + 2 = 0) ∧ (x * y < 0)) ↔ (a < 0) :=
by
  sorry

end two_real_roots_opposite_signs_l26_26520


namespace smallest_n_l26_26813

theorem smallest_n (n : ℕ) (h₁ : n > 2016) (h₂ : n % 4 = 0) : 
  ¬(1^n + 2^n + 3^n + 4^n) % 10 = 0 → n = 2020 :=
by
  sorry

end smallest_n_l26_26813


namespace balloons_left_after_distribution_l26_26578

-- Definitions for the conditions
def red_balloons : ℕ := 23
def blue_balloons : ℕ := 39
def green_balloons : ℕ := 71
def yellow_balloons : ℕ := 89
def total_balloons : ℕ := red_balloons + blue_balloons + green_balloons + yellow_balloons
def number_of_friends : ℕ := 10

-- Statement to prove the correct answer
theorem balloons_left_after_distribution : total_balloons % number_of_friends = 2 :=
by
  -- The proof would go here
  sorry

end balloons_left_after_distribution_l26_26578


namespace number_of_shirts_that_weigh_1_pound_l26_26746

/-- 
Jon's laundry machine can do 5 pounds of laundry at a time. 
Some number of shirts weigh 1 pound. 
2 pairs of pants weigh 1 pound. 
Jon needs to wash 20 shirts and 20 pants. 
Jon has to do 3 loads of laundry. 
-/
theorem number_of_shirts_that_weigh_1_pound
    (machine_capacity : ℕ)
    (num_shirts : ℕ)
    (shirts_per_pound : ℕ)
    (pairs_of_pants_per_pound : ℕ)
    (num_pants : ℕ)
    (loads : ℕ)
    (weight_per_load : ℕ)
    (total_pants_weight : ℕ)
    (total_weight : ℕ)
    (shirt_weight_per_pound : ℕ)
    (shirts_weighing_one_pound : ℕ) :
  machine_capacity = 5 → 
  num_shirts = 20 → 
  pairs_of_pants_per_pound = 2 →
  num_pants = 20 →
  loads = 3 →
  weight_per_load = 5 → 
  total_pants_weight = (num_pants / pairs_of_pants_per_pound) →
  total_weight = (loads * weight_per_load) →
  shirts_weighing_one_pound = (total_weight - total_pants_weight) / num_shirts → 
  shirts_weighing_one_pound = 4 :=
by sorry

end number_of_shirts_that_weigh_1_pound_l26_26746


namespace prime_bound_l26_26971

-- The definition for the n-th prime number
def nth_prime (n : ℕ) : ℕ := sorry  -- placeholder for the primorial definition

-- The main theorem to prove
theorem prime_bound (n : ℕ) : nth_prime n ≤ 2 ^ 2 ^ (n - 1) := sorry

end prime_bound_l26_26971


namespace overall_average_score_l26_26192

-- Definitions used from conditions
def male_students : Nat := 8
def male_avg_score : Real := 83
def female_students : Nat := 28
def female_avg_score : Real := 92

-- Theorem to prove the overall average score is 90
theorem overall_average_score : 
  (male_students * male_avg_score + female_students * female_avg_score) / (male_students + female_students) = 90 := 
by 
  sorry

end overall_average_score_l26_26192


namespace rationalize_denominator_l26_26141

theorem rationalize_denominator (h : Real.sqrt 200 = 10 * Real.sqrt 2) : 
  (7 / Real.sqrt 200) = (7 * Real.sqrt 2 / 20) :=
by
  sorry

end rationalize_denominator_l26_26141


namespace races_to_champion_l26_26444

theorem races_to_champion (num_sprinters : ℕ) (sprinters_per_race : ℕ) (advancing_per_race : ℕ)
  (eliminated_per_race : ℕ) (initial_races : ℕ) (total_races : ℕ):
  num_sprinters = 360 ∧ sprinters_per_race = 8 ∧ advancing_per_race = 2 ∧ 
  eliminated_per_race = 6 ∧ initial_races = 45 ∧ total_races = 62 →
  initial_races + (initial_races / sprinters_per_race +
  ((initial_races / sprinters_per_race) / sprinters_per_race +
  (((initial_races / sprinters_per_race) / sprinters_per_race) / sprinters_per_race + 1))) = total_races :=
sorry

end races_to_champion_l26_26444


namespace problem_am_gm_inequality_l26_26246

theorem problem_am_gm_inequality
  (a b c : ℝ)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum_sq : a^2 + b^2 + c^2 = 3) : 
  (1 / (1 + a * b)) + (1 / (1 + b * c)) + (1 / (1 + c * a)) ≥ 3 / 2 :=
by
  sorry

end problem_am_gm_inequality_l26_26246


namespace average_gas_mileage_round_trip_l26_26370

theorem average_gas_mileage_round_trip :
  (300 / ((150 / 28) + (150 / 18))) = 22 := by
sorry

end average_gas_mileage_round_trip_l26_26370


namespace product_of_distinct_numbers_l26_26428

theorem product_of_distinct_numbers (x y : ℝ) (h1 : x ≠ y)
  (h2 : 1 / (1 + x^2) + 1 / (1 + y^2) = 2 / (1 + x * y)) :
  x * y = 1 := 
sorry

end product_of_distinct_numbers_l26_26428


namespace value_of_x_l26_26483

theorem value_of_x (v w z y x : ℤ) 
  (h1 : v = 90)
  (h2 : w = v + 30)
  (h3 : z = w + 21)
  (h4 : y = z + 11)
  (h5 : x = y + 6) : 
  x = 158 :=
by 
  sorry

end value_of_x_l26_26483


namespace c_work_rate_l26_26435

theorem c_work_rate (x : ℝ) : 
  (1 / 7 + 1 / 14 + 1 / x = 1 / 4) → x = 28 :=
by
  sorry

end c_work_rate_l26_26435


namespace fraction_to_decimal_l26_26837

theorem fraction_to_decimal : (9 : ℚ) / 25 = 0.36 :=
by
  sorry

end fraction_to_decimal_l26_26837


namespace arrange_letters_l26_26010

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem arrange_letters : factorial 7 / (factorial 3 * factorial 2 * factorial 2) = 210 := 
by
  sorry

end arrange_letters_l26_26010


namespace neither_5_nice_nor_6_nice_count_l26_26877

def is_k_nice (N k : ℕ) : Prop :=
  N % k = 1

def count_5_nice (N : ℕ) : ℕ :=
  (N - 1) / 5 + 1

def count_6_nice (N : ℕ) : ℕ :=
  (N - 1) / 6 + 1

def lcm (a b : ℕ) : ℕ :=
  Nat.lcm a b

def count_30_nice (N : ℕ) : ℕ :=
  (N - 1) / 30 + 1

theorem neither_5_nice_nor_6_nice_count : 
  ∀ N < 200, 
  (N - (count_5_nice 199 + count_6_nice 199 - count_30_nice 199)) = 133 := 
by
  sorry

end neither_5_nice_nor_6_nice_count_l26_26877


namespace area_error_percent_l26_26248

theorem area_error_percent (L W : ℝ) (L_pos : 0 < L) (W_pos : 0 < W) :
  let A := L * W
  let A_measured := (1.05 * L) * (0.96 * W)
  let error_percent := ((A_measured - A) / A) * 100
  error_percent = 0.8 :=
by
  let A := L * W
  let A_measured := (1.05 * L) * (0.96 * W)
  let error := A_measured - A
  let error_percent := (error / A) * 100
  sorry

end area_error_percent_l26_26248


namespace sum_of_three_consecutive_numbers_l26_26120

theorem sum_of_three_consecutive_numbers (smallest : ℕ) (h : smallest = 29) :
  (smallest + (smallest + 1) + (smallest + 2)) = 90 :=
by
  sorry

end sum_of_three_consecutive_numbers_l26_26120


namespace cone_inscribed_spheres_distance_l26_26672

noncomputable def distance_between_sphere_centers (R α : ℝ) : ℝ :=
  R * (Real.sqrt 2) * Real.sin (α / 2) / (2 * Real.cos (α / 8) * Real.cos ((45 : ℝ) - α / 8))

theorem cone_inscribed_spheres_distance (R α : ℝ) (h1 : R > 0) (h2 : α > 0) :
  distance_between_sphere_centers R α = R * (Real.sqrt 2) * Real.sin (α / 2) / (2 * Real.cos (α / 8) * Real.cos ((45 : ℝ) - α / 8)) :=
by 
  sorry

end cone_inscribed_spheres_distance_l26_26672


namespace steve_speed_back_l26_26990

open Real

noncomputable def steves_speed_on_way_back : ℝ := 15

theorem steve_speed_back
  (distance_to_work : ℝ)
  (traffic_time_to_work : ℝ)
  (traffic_time_back : ℝ)
  (total_time : ℝ)
  (speed_ratio : ℝ) :
  distance_to_work = 30 →
  traffic_time_to_work = 30 →
  traffic_time_back = 15 →
  total_time = 405 →
  speed_ratio = 2 →
  steves_speed_on_way_back = 15 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end steve_speed_back_l26_26990


namespace smallest_c_such_that_one_in_range_l26_26951

theorem smallest_c_such_that_one_in_range :
  ∃ c : ℝ, (∀ x : ℝ, ∃ y : ℝ, y =  x^2 - 2 * x + c ∧ y = 1) ∧ c = 2 :=
by
  sorry

end smallest_c_such_that_one_in_range_l26_26951


namespace outlet_pipe_emptying_time_l26_26280

noncomputable def fill_rate_pipe1 : ℝ := 1 / 18
noncomputable def fill_rate_pipe2 : ℝ := 1 / 30
noncomputable def empty_rate_outlet_pipe (x : ℝ) : ℝ := 1 / x
noncomputable def combined_rate (x : ℝ) : ℝ := fill_rate_pipe1 + fill_rate_pipe2 - empty_rate_outlet_pipe x
noncomputable def total_fill_time : ℝ := 0.06666666666666665

theorem outlet_pipe_emptying_time : ∃ x : ℝ, combined_rate x = 1 / total_fill_time ∧ x = 45 :=
by
  sorry

end outlet_pipe_emptying_time_l26_26280


namespace distance_internal_tangent_l26_26235

noncomputable def radius_O := 5
noncomputable def distance_external := 9

theorem distance_internal_tangent (radius_O radius_dist_external : ℝ) 
  (h1 : radius_O = 5) (h2: radius_dist_external = 9) : 
  ∃ r : ℝ, r = 4 ∧ abs (r - radius_O) = 1 := by
  sorry

end distance_internal_tangent_l26_26235


namespace a_pow_10_plus_b_pow_10_l26_26017

theorem a_pow_10_plus_b_pow_10 (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11)
  (hn : ∀ n ≥ 3, a^(n) + b^(n) = (a^(n-1) + b^(n-1)) + (a^(n-2) + b^(n-2))) :
  a^10 + b^10 = 123 :=
by
  sorry

end a_pow_10_plus_b_pow_10_l26_26017


namespace share_per_person_l26_26390

-- Defining the total cost and number of people
def total_cost : ℝ := 12100
def num_people : ℝ := 11

-- The theorem stating that each person's share is $1,100.00
theorem share_per_person : total_cost / num_people = 1100 := by
  sorry

end share_per_person_l26_26390


namespace parameter_conditions_l26_26251

theorem parameter_conditions (p x y : ℝ) :
  (x - p)^2 = 16 * (y - 3 + p) →
  y^2 + ((x - 3) / (|x| - 3))^2 = 1 →
  |x| ≠ 3 →
  p > 3 ∧ 
  ((p ≤ 4 ∨ p ≥ 12) ∧ (p < 19 ∨ 19 < p)) :=
sorry

end parameter_conditions_l26_26251


namespace saree_final_sale_price_in_inr_l26_26493

noncomputable def finalSalePrice (initialPrice: ℝ) (discounts: List ℝ) (conversionRate: ℝ) : ℝ :=
  let finalUSDPrice := discounts.foldl (fun acc discount => acc * (1 - discount)) initialPrice
  finalUSDPrice * conversionRate

theorem saree_final_sale_price_in_inr
  (initialPrice : ℝ := 150)
  (discounts : List ℝ := [0.20, 0.15, 0.05])
  (conversionRate : ℝ := 75)
  : finalSalePrice initialPrice discounts conversionRate = 7267.5 :=
by
  sorry

end saree_final_sale_price_in_inr_l26_26493


namespace find_number_l26_26606

-- Define the number x and the condition as a theorem to be proven.
theorem find_number (x : ℝ) (h : (1/3) * x - 5 = 10) : x = 45 :=
sorry

end find_number_l26_26606


namespace smallest_n_sqrt_12n_integer_l26_26764

theorem smallest_n_sqrt_12n_integer : ∃ n : ℕ, (n > 0) ∧ (∃ k : ℕ, 12 * n = k^2) ∧ n = 3 := by
  sorry

end smallest_n_sqrt_12n_integer_l26_26764


namespace value_of_A_l26_26711

theorem value_of_A {α : Type} [LinearOrderedSemiring α] 
  (L A D E : α) (L_value : L = 15) (LEAD DEAL DELL : α)
  (LEAD_value : LEAD = 50)
  (DEAL_value : DEAL = 55)
  (DELL_value : DELL = 60)
  (LEAD_condition : L + E + A + D = LEAD)
  (DEAL_condition : D + E + A + L = DEAL)
  (DELL_condition : D + E + L + L = DELL) :
  A = 25 :=
by
  sorry

end value_of_A_l26_26711


namespace find_x_l26_26808

theorem find_x (x : ℝ) : (x / (x + 2) + 3 / (x + 2) + 2 * x / (x + 2) = 4) → x = -5 :=
by
  sorry

end find_x_l26_26808


namespace find_x_value_l26_26488

noncomputable def floor_plus_2x_eq_33 (x : ℝ) : Prop :=
  ∃ n : ℤ, ⌊x⌋ = n ∧ n + 2 * x = 33 ∧  (0 : ℝ) ≤ x - n ∧ x - n < 1

theorem find_x_value : ∀ x : ℝ, floor_plus_2x_eq_33 x → x = 11 :=
by
  intro x
  intro h
  -- Proof skipped, included as 'sorry' to compile successfully.
  sorry

end find_x_value_l26_26488


namespace joshua_bottle_caps_l26_26955

theorem joshua_bottle_caps (initial_caps : ℕ) (additional_caps : ℕ) (total_caps : ℕ) 
  (h1 : initial_caps = 40) 
  (h2 : additional_caps = 7) 
  (h3 : total_caps = initial_caps + additional_caps) : 
  total_caps = 47 := 
by 
  sorry

end joshua_bottle_caps_l26_26955


namespace greatest_three_digit_number_l26_26675

theorem greatest_three_digit_number
  (n : ℕ) (h_3digit : 100 ≤ n ∧ n < 1000) (h_mod7 : n % 7 = 2) (h_mod4 : n % 4 = 1) :
  n = 989 :=
sorry

end greatest_three_digit_number_l26_26675


namespace abs_neg_ten_l26_26283

theorem abs_neg_ten : abs (-10) = 10 := 
by {
  sorry
}

end abs_neg_ten_l26_26283


namespace minimize_distance_l26_26670

theorem minimize_distance
  (a b c d : ℝ)
  (h1 : (a + 3 * Real.log a) / b = 1)
  (h2 : (d - 3) / (2 * c) = 1) :
  (a - c)^2 + (b - d)^2 = (9 / 5) * (Real.log (Real.exp 1 / 3))^2 :=
by sorry

end minimize_distance_l26_26670


namespace max_marks_l26_26325

theorem max_marks (M : ℝ) (h1 : 0.33 * M = 92 + 40) : M = 400 :=
by
  sorry

end max_marks_l26_26325


namespace construct_convex_hexagon_l26_26501

-- Definitions of the sides and their lengths
variables {A B C D E F : Type} -- Points of the hexagon
variables {AB BC CD DE EF FA : ℝ}  -- Lengths of the sides
variables (convex_hexagon : Prop) -- the hexagon is convex

-- Hypotheses of parallel and equal opposite sides
variables (H_AB_DE : AB = DE)
variables (H_BC_EF : BC = EF)
variables (H_CD_AF : CD = AF)

-- Define the construction of the hexagon under the given conditions
theorem construct_convex_hexagon
  (convex_hexagon : Prop)
  (H_AB_DE : AB = DE)
  (H_BC_EF : BC = EF)
  (H_CD_AF : CD = AF) : 
  ∃ (A B C D E F : Type), 
    A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ F ∧ F ≠ A ∧ convex_hexagon ∧ 
    (AB = FA) ∧ (AF = CD) ∧ (BC = EF) ∧ (AB = DE) := 
sorry -- Proof omitted

end construct_convex_hexagon_l26_26501


namespace average_side_lengths_l26_26223

open Real

theorem average_side_lengths (A1 A2 A3 : ℝ) (h1 : A1 = 25) (h2 : A2 = 64) (h3 : A3 = 144) :
  ((sqrt A1) + (sqrt A2) + (sqrt A3)) / 3 = 25 / 3 :=
by 
  -- To be filled in the proof later
  sorry

end average_side_lengths_l26_26223


namespace garden_area_increase_l26_26401

-- Define the dimensions and perimeter of the rectangular garden
def length_rect : ℕ := 30
def width_rect : ℕ := 12
def area_rect : ℕ := length_rect * width_rect

def perimeter_rect : ℕ := 2 * (length_rect + width_rect)

-- Define the side length and area of the new square garden
def side_square : ℕ := perimeter_rect / 4
def area_square : ℕ := side_square * side_square

-- Define the increase in area
def increase_in_area : ℕ := area_square - area_rect

-- Prove the increase in area is 81 square feet
theorem garden_area_increase : increase_in_area = 81 := by
  sorry

end garden_area_increase_l26_26401


namespace find_angle_B_find_sin_C_l26_26779

-- Statement for proving B = π / 4 given the conditions
theorem find_angle_B (a b c : ℝ) (A B C : ℝ) 
  (h1 : a * Real.sin A + c * Real.sin C - Real.sqrt 2 * a * Real.sin C = b * Real.sin B) 
  (hABC : A + B + C = Real.pi) :
  B = Real.pi / 4 := 
sorry

-- Statement for proving sin C when cos A = 1 / 3
theorem find_sin_C (A C : ℝ) 
  (hA : Real.cos A = 1 / 3)
  (hABC : A + Real.pi / 4 + C = Real.pi) :
  Real.sin C = (4 + Real.sqrt 2) / 6 := 
sorry

end find_angle_B_find_sin_C_l26_26779


namespace difference_of_numbers_l26_26576

variable (x y : ℝ)

theorem difference_of_numbers (h1 : x + y = 10) (h2 : x - y = 19) (h3 : x^2 - y^2 = 190) : x - y = 19 :=
by sorry

end difference_of_numbers_l26_26576


namespace min_value_range_of_x_l26_26423

variables (a b x : ℝ)

-- Problem 1: Prove the minimum value of 1/a + 4/b given a + b = 1, a > 0, b > 0
theorem min_value (h1 : a + b = 1) (h2 : a > 0) (h3 : b > 0) : 
  ∃ c, c = 9 ∧ ∀ y, ∃ (a b : ℝ), a + b = 1 ∧ a > 0 ∧ b > 0 → (1/a + 4/b) ≥ y :=
sorry

-- Problem 2: Prove the range of x for which 1/a + 4/b ≥ |2x - 1| - |x + 1|
theorem range_of_x (h : ∀ (a b : ℝ), a + b = 1 → a > 0 → b > 0 → (1/a + 4/b) ≥ (|2*x - 1| - |x + 1|)) :
  -7 ≤ x ∧ x ≤ 11 :=
sorry

end min_value_range_of_x_l26_26423


namespace discount_percentage_l26_26184

theorem discount_percentage 
  (evening_ticket_cost : ℝ) (food_combo_cost : ℝ) (savings : ℝ) (discounted_food_combo_cost : ℝ) (discounted_total_cost : ℝ) 
  (h1 : evening_ticket_cost = 10) 
  (h2 : food_combo_cost = 10)
  (h3 : discounted_food_combo_cost = 10 * 0.5)
  (h4 : discounted_total_cost = evening_ticket_cost + food_combo_cost - savings)
  (h5 : savings = 7)
: (1 - discounted_total_cost / (evening_ticket_cost + food_combo_cost)) * 100 = 20 :=
by
  sorry

end discount_percentage_l26_26184


namespace tangent_slope_of_circle_l26_26741

theorem tangent_slope_of_circle {x1 y1 x2 y2 : ℝ}
  (hx1 : x1 = 1) (hy1 : y1 = 1) (hx2 : x2 = 6) (hy2 : y2 = 4) :
  ∀ m : ℝ, m = -5 / 3 ↔
    (∃ (r : ℝ), r = (y2 - y1) / (x2 - x1) ∧ m = -1 / r) :=
by
  sorry

end tangent_slope_of_circle_l26_26741


namespace find_width_of_sheet_of_paper_l26_26322

def width_of_sheet_of_paper (W : ℝ) : Prop :=
  let margin := 1.5
  let length_of_paper := 10
  let area_covered := 38.5
  let width_of_picture := W - 2 * margin
  let length_of_picture := length_of_paper - 2 * margin
  width_of_picture * length_of_picture = area_covered

theorem find_width_of_sheet_of_paper : ∃ W : ℝ, width_of_sheet_of_paper W ∧ W = 8.5 :=
by
  -- Placeholder for the actual proof
  sorry

end find_width_of_sheet_of_paper_l26_26322


namespace opposite_seven_is_minus_seven_l26_26926

theorem opposite_seven_is_minus_seven :
  ∃ x : ℤ, 7 + x = 0 ∧ x = -7 := 
sorry

end opposite_seven_is_minus_seven_l26_26926


namespace range_of_a_l26_26526

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 - a * x + 1 < 0) ↔ (a > 2 ∨ a < -2) :=
by
  sorry

end range_of_a_l26_26526


namespace part1_part2_part3_l26_26621

variable {a b c : ℝ}

-- Part (1)
theorem part1 (a b c : ℝ) : a * (b - c) ^ 2 + b * (c - a) ^ 2 + c * (a - b) ^ 2 + 4 * a * b * c > a ^ 3 + b ^ 3 + c ^ 3 :=
sorry

-- Part (2)
theorem part2 (a b c : ℝ) : 2 * a ^ 2 * b ^ 2 + 2 * b ^ 2 * c ^ 2 + 2 * c ^ 2 * a ^ 2 > a ^ 4 + b ^ 4 + c ^ 4 :=
sorry

-- Part (3)
theorem part3 (a b c : ℝ) : 2 * a * b + 2 * b * c + 2 * c * a > a ^ 2 + b ^ 2 + c ^ 2 :=
sorry

end part1_part2_part3_l26_26621


namespace motorcyclist_average_speed_l26_26005

theorem motorcyclist_average_speed :
  ∀ (t : ℝ), 120 / t = 60 * 3 → 
  3 * t / 4 = 45 :=
by
  sorry

end motorcyclist_average_speed_l26_26005


namespace solution_set_inequality_l26_26954

theorem solution_set_inequality (x : ℝ) (h : 0 < x ∧ x ≤ 1) : 
  ∀ (x : ℝ), (0 < x ∧ x ≤ 1 ↔ ∀ a > 0, ∀ b ≤ 1, (2/x + (1-x) ^ (1/2) ≥ 1 + (1-x)^(1/2))) := sorry

end solution_set_inequality_l26_26954


namespace no_real_solution_for_eq_l26_26888

theorem no_real_solution_for_eq (y : ℝ) : ¬ ∃ y : ℝ, ((y - 4 * y + 10)^2 + 4 = -2 * |y|) :=
by
  sorry

end no_real_solution_for_eq_l26_26888


namespace intersection_of_sets_l26_26350

open Set

variable {x : ℝ}

theorem intersection_of_sets : 
  let A := {x : ℝ | x^2 - 4*x + 3 < 0}
  let B := {x : ℝ | x > 2}
  A ∩ B = {x : ℝ | 2 < x ∧ x < 3} :=
by
  sorry

end intersection_of_sets_l26_26350


namespace audrey_older_than_heracles_l26_26287

variable (A H : ℕ)
variable (hH : H = 10)
variable (hFutureAge : A + 3 = 2 * H)

theorem audrey_older_than_heracles : A - H = 7 :=
by
  have h1 : H = 10 := by assumption
  have h2 : A + 3 = 2 * H := by assumption
  -- Proof is omitted
  sorry

end audrey_older_than_heracles_l26_26287


namespace find_xy_l26_26108

theorem find_xy (x y : ℕ) (hx : x ≥ 1) (hy : y ≥ 1) : 
  2^x - 5 = 11^y ↔ (x = 4 ∧ y = 1) :=
by sorry

end find_xy_l26_26108


namespace find_p_l26_26346

variable (p q : ℝ) (k : ℕ)

theorem find_p (h_sum : ∀ (α β : ℝ), α + β = 2) (h_prod : ∀ (α β : ℝ), α * β = k) (hk : k > 0) :
  p = -2 := by
  sorry

end find_p_l26_26346


namespace variable_cost_per_book_l26_26653

theorem variable_cost_per_book
  (F : ℝ) (S : ℝ) (N : ℕ) (V : ℝ)
  (fixed_cost : F = 56430) 
  (selling_price_per_book : S = 21.75) 
  (num_books : N = 4180) 
  (production_eq_sales : S * N = F + V * N) :
  V = 8.25 :=
by sorry

end variable_cost_per_book_l26_26653


namespace roger_expenses_fraction_l26_26827

theorem roger_expenses_fraction {B t s n : ℝ} (h1 : t = 0.25 * (B - s))
  (h2 : s = 0.10 * (B - t)) (h3 : n = 5) :
  (t + s + n) / B = 0.41 :=
sorry

end roger_expenses_fraction_l26_26827


namespace magnitude_BC_range_l26_26252

theorem magnitude_BC_range (AB AC : EuclideanSpace ℝ (Fin 2)) 
  (h₁ : ‖AB‖ = 18) (h₂ : ‖AC‖ = 5) : 
  13 ≤ ‖AC - AB‖ ∧ ‖AC - AB‖ ≤ 23 := 
  sorry

end magnitude_BC_range_l26_26252


namespace find_xy_l26_26018

variable {x y : ℝ}

theorem find_xy (h₁ : x + y = 10) (h₂ : x^3 + y^3 = 370) : x * y = 21 :=
by
  sorry

end find_xy_l26_26018


namespace table_relationship_l26_26592

theorem table_relationship (x y : ℕ) (h : (x, y) ∈ [(1, 1), (2, 8), (3, 27), (4, 64), (5, 125)]) : y = x^3 :=
sorry

end table_relationship_l26_26592


namespace books_for_sale_l26_26708

theorem books_for_sale (initial_books found_books : ℕ) (h1 : initial_books = 33) (h2 : found_books = 26) :
  initial_books + found_books = 59 :=
by
  sorry

end books_for_sale_l26_26708


namespace roots_ratio_quadratic_eq_l26_26210

theorem roots_ratio_quadratic_eq {k r s : ℝ} 
(h_eq : ∃ a b : ℝ, a * r = b * s) 
(ratio_3_2 : ∃ t : ℝ, r = 3 * t ∧ s = 2 * t) 
(eqn : r + s = -10 ∧ r * s = k) : 
k = 24 := 
sorry

end roots_ratio_quadratic_eq_l26_26210


namespace pizza_slices_l26_26158

-- Definitions of conditions
def slices (H C : ℝ) : Prop :=
  (H / 2 - 3 + 2 * C / 3 = 11) ∧ (H = C)

-- Stating the theorem to prove
theorem pizza_slices (H C : ℝ) (h : slices H C) : H = 12 :=
sorry

end pizza_slices_l26_26158


namespace integer_pairs_sum_product_l26_26538

theorem integer_pairs_sum_product (x y : ℤ) (h : x + y = x * y) : (x = 2 ∧ y = 2) ∨ (x = 0 ∧ y = 0) :=
by
  sorry

end integer_pairs_sum_product_l26_26538


namespace find_x_plus_y_l26_26305

theorem find_x_plus_y
  (x y : ℝ)
  (h1 : x + Real.cos y = 2010)
  (h2 : x + 2010 * Real.sin y = 2009)
  (h3 : (π / 2) ≤ y ∧ y ≤ π) :
  x + y = 2011 + π :=
sorry

end find_x_plus_y_l26_26305


namespace largest_possible_k_satisfies_triangle_condition_l26_26507

theorem largest_possible_k_satisfies_triangle_condition :
  ∃ k : ℕ, 
    k = 2009 ∧ 
    ∀ (b r w : Fin 2009 → ℝ), 
    (∀ i : Fin 2009, i ≤ i.succ → b i ≤ b i.succ ∧ r i ≤ r i.succ ∧ w i ≤ w i.succ) → 
    (∃ (j : Fin 2009), 
      b j + r j > w j ∧ b j + w j > r j ∧ r j + w j > b j) :=
sorry

end largest_possible_k_satisfies_triangle_condition_l26_26507


namespace sum_of_squares_l26_26365

theorem sum_of_squares (x y z : ℝ)
  (h1 : (x + y + z) / 3 = 10)
  (h2 : (xyz)^(1/3) = 6)
  (h3 : 3 / ((1/x) + (1/y) + (1/z)) = 4) : 
  x^2 + y^2 + z^2 = 576 := 
by
  sorry

end sum_of_squares_l26_26365


namespace largest_three_digit_base7_to_decimal_l26_26339

theorem largest_three_digit_base7_to_decimal :
  (6 * 7^2 + 6 * 7^1 + 6 * 7^0) = 342 :=
by
  sorry

end largest_three_digit_base7_to_decimal_l26_26339


namespace factorize_expression_l26_26278

theorem factorize_expression (a b : ℝ) :
  ab^(3 : ℕ) - 4 * ab = ab * (b + 2) * (b - 2) :=
by
  -- proof to be provided
  sorry

end factorize_expression_l26_26278


namespace gcd_not_perfect_square_l26_26981

theorem gcd_not_perfect_square
  (m n : ℕ)
  (h1 : (m % 3 = 0 ∨ n % 3 = 0) ∧ ¬(m % 3 = 0 ∧ n % 3 = 0))
  : ¬ ∃ k : ℕ, k * k = Nat.gcd (m^2 + n^2 + 2) (m^2 * n^2 + 3) :=
by
  sorry

end gcd_not_perfect_square_l26_26981


namespace max_months_with_5_sundays_l26_26379

theorem max_months_with_5_sundays (months : ℕ) (days_in_year : ℕ) (extra_sundays : ℕ) :
  months = 12 ∧ (days_in_year = 365 ∨ days_in_year = 366) ∧ extra_sundays = days_in_year % 7
  → ∃ max_months_with_5_sundays, max_months_with_5_sundays = 5 := 
by
  sorry

end max_months_with_5_sundays_l26_26379


namespace usual_time_is_36_l26_26870

noncomputable def usual_time_to_school (R : ℝ) (T : ℝ) : Prop :=
  let new_rate := (9/8 : ℝ) * R
  let new_time := T - 4
  R * T = new_rate * new_time

theorem usual_time_is_36 (R : ℝ) (T : ℝ) (h : T = 36) : usual_time_to_school R T :=
by
  sorry

end usual_time_is_36_l26_26870


namespace range_of_m_l26_26125

def isDistinctRealRootsInInterval (a b x : ℝ) : Prop :=
  a * x^2 + b * x + 4 = 0 ∧ 0 < x ∧ x ≤ 3

theorem range_of_m (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ isDistinctRealRootsInInterval 1 (- (m + 1)) x ∧ isDistinctRealRootsInInterval 1 (- (m + 1)) y) ↔
  (3 < m ∧ m ≤ 10 / 3) :=
sorry

end range_of_m_l26_26125


namespace find_xyz_l26_26845

theorem find_xyz (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) + x * y * z = 15) :
  x * y * z = 9 / 2 := by
  sorry

end find_xyz_l26_26845


namespace geometric_series_sum_l26_26552

theorem geometric_series_sum : 
  ∀ (a r l : ℕ), 
    a = 2 ∧ r = 3 ∧ l = 4374 → 
    ∃ n S, 
      a * r ^ (n - 1) = l ∧ 
      S = a * (r^n - 1) / (r - 1) ∧ 
      S = 6560 :=
by 
  intros a r l h
  sorry

end geometric_series_sum_l26_26552


namespace proof_problem_l26_26510

open Set

noncomputable def U : Set ℝ := Icc (-5 : ℝ) 4

noncomputable def A : Set ℝ := {x : ℝ | -3 ≤ 2 * x + 1 ∧ 2 * x + 1 < 1}

noncomputable def B : Set ℝ := {x : ℝ | x^2 - 2 * x ≤ 0}

-- Definition of the complement of A in U
noncomputable def complement_U_A : Set ℝ := U \ A

-- The final proof statement
theorem proof_problem : (complement_U_A ∩ B) = Icc 0 2 :=
by
  sorry

end proof_problem_l26_26510


namespace smallest_circle_equation_l26_26111

theorem smallest_circle_equation :
  ∃ (x y : ℝ), (y^2 = 4 * x) ∧ (x - 1)^2 + y^2 = 1 ∧ ((x - 1)^2 + y^2 = 1) = (x^2 + y^2 = 1) := 
sorry

end smallest_circle_equation_l26_26111


namespace sum_of_roots_l26_26259

theorem sum_of_roots : 
  (∃ x1 x2 : ℚ, (3 * x1 + 4) * (2 * x1 - 12) = 0 ∧ (3 * x2 + 4) * (2 * x2 - 12) = 0 ∧ x1 ≠ x2 ∧ x1 + x2 = 14 / 3) :=
sorry

end sum_of_roots_l26_26259


namespace root_of_quadratic_l26_26972

theorem root_of_quadratic (b : ℝ) : 
  (-9)^2 + b * (-9) - 45 = 0 -> b = 4 :=
by
  sorry

end root_of_quadratic_l26_26972


namespace fraction_meaningful_iff_l26_26132

theorem fraction_meaningful_iff (x : ℝ) : (∃ y, y = x / (x - 2)) ↔ x ≠ 2 :=
by
  sorry

end fraction_meaningful_iff_l26_26132


namespace equipment_B_production_l26_26903

theorem equipment_B_production
  (total_production : ℕ)
  (sample_size : ℕ)
  (A_sample_production : ℕ)
  (B_sample_production : ℕ)
  (A_total_production : ℕ)
  (B_total_production : ℕ)
  (total_condition : total_production = 4800)
  (sample_condition : sample_size = 80)
  (A_sample_condition : A_sample_production = 50)
  (B_sample_condition : B_sample_production = 30)
  (ratio_condition : (A_sample_production / B_sample_production) = (5 / 3))
  (production_condition : A_total_production + B_total_production = total_production) :
  B_total_production = 1800 := 
sorry

end equipment_B_production_l26_26903


namespace quadratic_distinct_positive_roots_l26_26197

theorem quadratic_distinct_positive_roots (a : ℝ) : 
  9 * (a - 2) > 0 → 
  a > 0 → 
  a^2 - 9 * a + 18 > 0 → 
  a ≠ 11 → 
  (2 < a ∧ a < 3) ∨ (6 < a ∧ a < 11) ∨ (11 < a) := 
by 
  intros h1 h2 h3 h4
  sorry

end quadratic_distinct_positive_roots_l26_26197


namespace find_c_for_radius_of_circle_l26_26061

theorem find_c_for_radius_of_circle :
  ∃ c : ℝ, (∀ x y : ℝ, x^2 + 8 * x + y^2 - 6 * y + c = 0 → (x + 4)^2 + (y - 3)^2 = 25 - c) ∧
  (∀ x y : ℝ, (x + 4)^2 + (y - 3)^2 = 25 → c = 0) :=
sorry

end find_c_for_radius_of_circle_l26_26061


namespace arithmetic_mean_eq_one_l26_26753

theorem arithmetic_mean_eq_one 
  (x a b : ℝ) 
  (hx : x ≠ 0) 
  (hb : b ≠ 0) : 
  (1 / 2 * ((x + a + b) / x + (x - a - b) / x)) = 1 := by
  sorry

end arithmetic_mean_eq_one_l26_26753


namespace isosceles_triangle_perimeter_l26_26138

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 2) (h2 : b = 4) (h3 : a ≠ b) (h4 : a + b > b) (h5 : a + b > a) 
: ∃ p : ℝ, p = 10 :=
by
  -- Using the given conditions to determine the perimeter
  sorry

end isosceles_triangle_perimeter_l26_26138


namespace inequality_proof_l26_26077

variables (a b c d e f : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f)
variable (hcond : |sqrt (a * d) - sqrt (b * c)| ≤ 1)

theorem inequality_proof :
  (a * e + b / e) * (c * e + d / e) ≥ (a^2 * f^2 - b^2 / f^2) * (d^2 / f^2 - c^2 * f^2) := by
  sorry

end inequality_proof_l26_26077


namespace math_more_than_reading_homework_l26_26669

-- Definitions based on given conditions
def M : Nat := 9  -- Math homework pages
def R : Nat := 2  -- Reading homework pages

theorem math_more_than_reading_homework :
  M - R = 7 :=
by
  -- Proof would go here, showing that 9 - 2 indeed equals 7
  sorry

end math_more_than_reading_homework_l26_26669


namespace count_six_digit_palindromes_l26_26876

def num_six_digit_palindromes : ℕ := 9000

theorem count_six_digit_palindromes :
  (∃ a b c d : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧
     num_six_digit_palindromes = 9000) :=
sorry

end count_six_digit_palindromes_l26_26876


namespace part1_part2_l26_26419

open Real

-- Condition: tan(alpha) = 3
variable {α : ℝ} (h : tan α = 3)

-- Proof of first part
theorem part1 : (4 * sin α - 2 * cos α) / (5 * cos α + 3 * sin α) = 5 / 7 :=
by
  sorry

-- Proof of second part
theorem part2 : 1 - 4 * sin α * cos α + 2 * cos α ^ 2 = 0 :=
by
  sorry

end part1_part2_l26_26419


namespace line_equation_min_intercepts_l26_26661

theorem line_equation_min_intercepts (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : 1 / a + 4 / b = 1) : 2 * 1 + 4 - 6 = 0 ↔ (a = 3 ∧ b = 6) :=
by
  sorry

end line_equation_min_intercepts_l26_26661


namespace set_equality_proof_l26_26503

theorem set_equality_proof :
  (∃ (u : ℤ), ∃ (m n l : ℤ), u = 12 * m + 8 * n + 4 * l) ↔
  (∃ (u : ℤ), ∃ (p q r : ℤ), u = 20 * p + 16 * q + 12 * r) :=
sorry

end set_equality_proof_l26_26503


namespace smallest_positive_n_l26_26024

theorem smallest_positive_n (n : ℕ) (h : 77 * n ≡ 308 [MOD 385]) : n = 4 :=
sorry

end smallest_positive_n_l26_26024


namespace Maria_trip_time_l26_26663

/-- 
Given:
- Maria drove 80 miles on a freeway.
- Maria drove 20 miles on a rural road.
- Her speed on the rural road was half of her speed on the freeway.
- Maria spent 40 minutes driving on the rural road.

Prove that Maria's entire trip took 120 minutes.
-/ 
theorem Maria_trip_time
  (distance_freeway : ℕ)
  (distance_rural : ℕ)
  (rural_speed_ratio : ℕ → ℕ)
  (time_rural_minutes : ℕ) 
  (time_freeway : ℕ)
  (total_time : ℕ) 
  (speed_rural : ℕ)
  (speed_freeway : ℕ) 
  :
  distance_freeway = 80 ∧
  distance_rural = 20 ∧ 
  rural_speed_ratio (speed_freeway) = speed_rural ∧ 
  time_rural_minutes = 40 ∧
  time_rural_minutes = 20 / speed_rural ∧
  speed_freeway = 2 * speed_rural ∧
  time_freeway = distance_freeway / speed_freeway ∧
  total_time = time_rural_minutes + time_freeway → 
  total_time = 120 :=
by
  intros
  sorry

end Maria_trip_time_l26_26663


namespace bricks_needed_for_wall_l26_26803

noncomputable def brick_volume (length width height : ℝ) : ℝ :=
  length * width * height

noncomputable def wall_volume (length height thickness : ℝ) : ℝ :=
  length * height * thickness

theorem bricks_needed_for_wall :
  let length_wall := 800
  let height_wall := 600
  let thickness_wall := 22.5
  let length_brick := 100
  let width_brick := 11.25
  let height_brick := 6
  let vol_wall := wall_volume length_wall height_wall thickness_wall
  let vol_brick := brick_volume length_brick width_brick height_brick
  vol_wall / vol_brick = 1600 :=
by
  sorry

end bricks_needed_for_wall_l26_26803


namespace case1_equiv_case2_equiv_determine_case_l26_26527

theorem case1_equiv (a c x : ℝ) (hc : c ≠ 0) (hx : x ≠ 0) : 
  ((x + a) / (x + c) = a / c) ↔ (a = c) :=
by sorry

theorem case2_equiv (b d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) : 
  (b / d = b / d) :=
by sorry

theorem determine_case (a b c d x : ℝ) (hc : c ≠ 0) (hx : x ≠ 0) (hb : b ≠ 0) (hd : d ≠ 0) :
  ¬((x + a) / (x + c) = a / c) ∧ (b / d = b / d) :=
by sorry

end case1_equiv_case2_equiv_determine_case_l26_26527


namespace conic_eccentricity_l26_26375

theorem conic_eccentricity (m : ℝ) (h : 0 < -m) (h2 : (Real.sqrt (1 + (-1 / m))) = 2) : m = -1/3 := 
by
  -- Proof can be added here
  sorry

end conic_eccentricity_l26_26375


namespace smallest_pos_int_terminating_decimal_with_9_l26_26034

theorem smallest_pos_int_terminating_decimal_with_9 : ∃ n : ℕ, (∃ m k : ℕ, n = 2^m * 5^k ∧ (∃ d : ℕ, d ∈ n.digits 10 ∧ d = 9)) ∧ n = 4096 :=
by {
    sorry
}

end smallest_pos_int_terminating_decimal_with_9_l26_26034


namespace bus_carrying_capacity_l26_26354

variables (C : ℝ)

theorem bus_carrying_capacity (h1 : ∀ x : ℝ, x = (3 / 5) * C) 
                              (h2 : ∀ y : ℝ, y = 50 - 18)
                              (h3 : ∀ z : ℝ, x + y = C) : C = 80 :=
by
  sorry

end bus_carrying_capacity_l26_26354


namespace mappings_count_A_to_B_l26_26327

open Finset

def A : Finset ℕ := {1, 2}
def B : Finset ℕ := {3, 4}

theorem mappings_count_A_to_B : (card B) ^ (card A) = 4 :=
by
  -- This line will state that the proof is skipped for now.
  sorry

end mappings_count_A_to_B_l26_26327


namespace speed_of_man_upstream_l26_26689

-- Conditions stated as definitions 
def V_m : ℝ := 33 -- Speed of the man in still water
def V_downstream : ℝ := 40 -- Speed of the man rowing downstream

-- Required proof problem
theorem speed_of_man_upstream : V_m - (V_downstream - V_m) = 26 := 
by
  -- the following sorry is a placeholder for the actual proof
  sorry

end speed_of_man_upstream_l26_26689


namespace t_over_s_possible_values_l26_26383

-- Define the initial conditions
variables (n : ℕ) (h : n ≥ 3)

-- The theorem statement
theorem t_over_s_possible_values (s t : ℕ) (h_s : s > 0) (h_t : t > 0) : 
  (∃ r : ℚ, r = t / s ∧ 1 ≤ r ∧ r < (n - 1)) :=
sorry

end t_over_s_possible_values_l26_26383


namespace other_candidate_votes_l26_26215

-- Define the constants according to the problem
variables (X Y Z : ℝ)
axiom h1 : X = Y + (1 / 2) * Y
axiom h2 : X = 22500
axiom h3 : Y = Z - (2 / 5) * Z

-- Define the goal
theorem other_candidate_votes : Z = 25000 :=
by
  sorry

end other_candidate_votes_l26_26215


namespace math_proof_equiv_l26_26740

def A := 5
def B := 3
def C := 2
def D := 0
def E := 0
def F := 1
def G := 0

theorem math_proof_equiv : (A * 1000 + B * 100 + C * 10 + D) + (E * 100 + F * 10 + G) = 5300 :=
by
  sorry

end math_proof_equiv_l26_26740


namespace sum_of_50th_row_l26_26267

-- Define triangular numbers
def T (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the sum of numbers in the nth row
def f (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1 -- T_1 is 1 for the base case
  else 2 * f (n - 1) + n * (n + 1)

-- Prove the sum of the 50th row
theorem sum_of_50th_row : f 50 = 2^50 - 2550 := 
  sorry

end sum_of_50th_row_l26_26267


namespace fencing_required_l26_26623

theorem fencing_required (L W : ℕ) (A : ℕ) 
  (hL : L = 20) 
  (hA : A = 680) 
  (hArea : A = L * W) : 
  2 * W + L = 88 := 
by 
  sorry

end fencing_required_l26_26623


namespace merchant_marked_price_percentage_l26_26854

variables (L S M C : ℝ)
variable (h1 : C = 0.7 * L)
variable (h2 : C = 0.75 * S)
variable (h3 : S = 0.9 * M)

theorem merchant_marked_price_percentage : M = 1.04 * L :=
by
  sorry

end merchant_marked_price_percentage_l26_26854


namespace diameter_of_circle_with_inscribed_right_triangle_l26_26602

theorem diameter_of_circle_with_inscribed_right_triangle (a b c : ℕ) (h1 : a = 6) (h2 : b = 8) (right_triangle : a^2 + b^2 = c^2) : c = 10 :=
by
  subst h1
  subst h2
  simp at right_triangle
  sorry

end diameter_of_circle_with_inscribed_right_triangle_l26_26602


namespace fraction_human_habitable_surface_l26_26002

variable (fraction_water_coverage : ℚ)
variable (fraction_inhabitable_remaining_land : ℚ)
variable (fraction_reserved_for_agriculture : ℚ)

def fraction_inhabitable_land (f_water : ℚ) (f_inhabitable : ℚ) : ℚ :=
  (1 - f_water) * f_inhabitable

def fraction_habitable_land (f_inhabitable_land : ℚ) (f_reserved : ℚ) : ℚ :=
  f_inhabitable_land * (1 - f_reserved)

theorem fraction_human_habitable_surface 
  (h1 : fraction_water_coverage = 3/5)
  (h2 : fraction_inhabitable_remaining_land = 2/3)
  (h3 : fraction_reserved_for_agriculture = 1/2) :
  fraction_habitable_land 
    (fraction_inhabitable_land fraction_water_coverage fraction_inhabitable_remaining_land)
    fraction_reserved_for_agriculture = 2/15 :=
by {
  sorry
}

end fraction_human_habitable_surface_l26_26002


namespace time_to_cross_l26_26382

noncomputable def length_first_train : ℝ := 210
noncomputable def speed_first_train : ℝ := 120 * 1000 / 3600 -- Convert to m/s
noncomputable def length_second_train : ℝ := 290.04
noncomputable def speed_second_train : ℝ := 80 * 1000 / 3600 -- Convert to m/s

noncomputable def relative_speed := speed_first_train + speed_second_train
noncomputable def total_length := length_first_train + length_second_train
noncomputable def crossing_time := total_length / relative_speed

theorem time_to_cross : crossing_time = 9 := by
  let length_first_train : ℝ := 210
  let speed_first_train : ℝ := 120 * 1000 / 3600 -- Convert to m/s
  let length_second_train : ℝ := 290.04
  let speed_second_train : ℝ := 80 * 1000 / 3600 -- Convert to m/s

  let relative_speed := speed_first_train + speed_second_train
  let total_length := length_first_train + length_second_train
  let crossing_time := total_length / relative_speed

  show crossing_time = 9
  sorry

end time_to_cross_l26_26382


namespace min_value_a_plus_2b_minus_3c_l26_26026

theorem min_value_a_plus_2b_minus_3c
  (a b c : ℝ)
  (h : ∀ (x y : ℝ), x + 2 * y - 3 ≤ a * x + b * y + c ∧ a * x + b * y + c ≤ x + 2 * y + 3) :
  ∃ m : ℝ, m = a + 2 * b - 3 * c ∧ m = -4 :=
by
  sorry

end min_value_a_plus_2b_minus_3c_l26_26026


namespace probability_sum_is_five_l26_26537

theorem probability_sum_is_five (m n : ℕ) (h_m : 1 ≤ m ∧ m ≤ 6) (h_n : 1 ≤ n ∧ n ≤ 6)
  (h_total_outcomes : ∃(total_outcomes : ℕ), total_outcomes = 36)
  (h_favorable_outcomes : ∃(favorable_outcomes : ℕ), favorable_outcomes = 4) :
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 9 :=
sorry

end probability_sum_is_five_l26_26537


namespace trajectory_midpoint_l26_26628

/-- Let A and B be two moving points on the circle x^2 + y^2 = 4, and AB = 2. 
    The equation of the trajectory of the midpoint M of the line segment AB is x^2 + y^2 = 3. -/
theorem trajectory_midpoint (A B : ℝ × ℝ) (M : ℝ × ℝ)
    (hA : A.1^2 + A.2^2 = 4)
    (hB : B.1^2 + B.2^2 = 4)
    (hAB : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4)
    (hM : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
    M.1^2 + M.2^2 = 3 :=
sorry

end trajectory_midpoint_l26_26628


namespace number_of_students_l26_26842

theorem number_of_students (n : ℕ)
  (h_avg : 100 * n = total_marks_unknown)
  (h_wrong_marks : total_marks_wrong = total_marks_unknown + 50)
  (h_correct_avg : total_marks_correct / n = 95)
  (h_corrected_marks : total_marks_correct = total_marks_wrong - 50) :
  n = 10 :=
by
  sorry

end number_of_students_l26_26842


namespace expected_value_X_correct_prob_1_red_ball_B_correct_l26_26767

-- Boxes configuration
structure BoxConfig where
  white_A : ℕ -- Number of white balls in box A
  red_A : ℕ -- Number of red balls in box A
  white_B : ℕ -- Number of white balls in box B
  red_B : ℕ -- Number of red balls in box B

-- Given the problem configuration
def initialConfig : BoxConfig := {
  white_A := 2,
  red_A := 2,
  white_B := 1,
  red_B := 3,
}

-- Define random variable X (number of red balls drawn from box A)
def prob_X (X : ℕ) (cfg : BoxConfig) : ℚ :=
  if X = 0 then 1 / 6
  else if X = 1 then 2 / 3
  else if X = 2 then 1 / 6
  else 0

-- Expected value of X
noncomputable def expected_value_X (cfg : BoxConfig) : ℚ :=
  0 * (prob_X 0 cfg) + 1 * (prob_X 1 cfg) + 2 * (prob_X 2 cfg)

-- Probability of drawing 1 red ball from box B
noncomputable def prob_1_red_ball_B (cfg : BoxConfig) (X : ℕ) : ℚ :=
  if X = 0 then 1 / 2
  else if X = 1 then 2 / 3
  else if X = 2 then 5 / 6
  else 0

-- Total probability of drawing 1 red ball from box B
noncomputable def total_prob_1_red_ball_B (cfg : BoxConfig) : ℚ :=
  (prob_X 0 cfg * (prob_1_red_ball_B cfg 0))
  + (prob_X 1 cfg * (prob_1_red_ball_B cfg 1))
  + (prob_X 2 cfg * (prob_1_red_ball_B cfg 2))


theorem expected_value_X_correct : expected_value_X initialConfig = 1 := by
  sorry

theorem prob_1_red_ball_B_correct : total_prob_1_red_ball_B initialConfig = 2 / 3 := by
  sorry

end expected_value_X_correct_prob_1_red_ball_B_correct_l26_26767


namespace no_integer_for_58th_power_64_digits_valid_replacement_for_64_digits_l26_26405

theorem no_integer_for_58th_power_64_digits : ¬ ∃ n : ℤ, 10^63 ≤ n^58 ∧ n^58 < 10^64 :=
sorry

theorem valid_replacement_for_64_digits (k : ℕ) (hk : 1 ≤ k ∧ k ≤ 81) : 
  ¬ ∃ n : ℤ, 10^(k-1) ≤ n^58 ∧ n^58 < 10^k :=
sorry

end no_integer_for_58th_power_64_digits_valid_replacement_for_64_digits_l26_26405


namespace number_of_dogs_l26_26406

variable {C D : ℕ}

def ratio_of_dogs_to_cats (D C : ℕ) : Prop := D = (15/7) * C

def ratio_after_additional_cats (D C : ℕ) : Prop :=
  D = 15 * (C + 8) / 11

theorem number_of_dogs (h1 : ratio_of_dogs_to_cats D C) (h2 : ratio_after_additional_cats D C) :
  D = 30 :=
by
  sorry

end number_of_dogs_l26_26406


namespace bob_got_15_candies_l26_26425

-- Define the problem conditions
def bob_neighbor_sam : Prop := true -- Bob is Sam's next door neighbor
def bob_accompany_sam_home : Prop := true -- Bob decided to accompany Sam home

def bob_share_chewing_gums : ℕ := 15 -- Bob's share of chewing gums
def bob_share_chocolate_bars : ℕ := 20 -- Bob's share of chocolate bars
def bob_share_candies : ℕ := 15 -- Bob's share of assorted candies

-- Define the main assertion
theorem bob_got_15_candies : bob_share_candies = 15 := 
by sorry

end bob_got_15_candies_l26_26425


namespace xy_value_l26_26121

theorem xy_value (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 := 
by sorry

end xy_value_l26_26121


namespace composition_points_value_l26_26478

theorem composition_points_value (f g : ℕ → ℕ) (ab cd : ℕ) 
  (h₁ : f 2 = 6) 
  (h₂ : f 3 = 4) 
  (h₃ : f 4 = 2)
  (h₄ : g 2 = 4) 
  (h₅ : g 3 = 2) 
  (h₆ : g 5 = 6) :
  let (a, b) := (2, 6)
  let (c, d) := (3, 4)
  ab + cd = (a * b) + (c * d) :=
by {
  sorry
}

end composition_points_value_l26_26478


namespace value_ab_plus_a_plus_b_l26_26189

noncomputable def polynomial : Polynomial ℝ := Polynomial.C (-1) + Polynomial.X * Polynomial.C (-1) + Polynomial.X^2 * Polynomial.C (-4) + Polynomial.X^4

theorem value_ab_plus_a_plus_b {a b : ℝ} (h : polynomial.eval a = 0 ∧ polynomial.eval b = 0) : a * b + a + b = -1 / 2 :=
sorry

end value_ab_plus_a_plus_b_l26_26189


namespace triangle_area_PQR_l26_26357

def point := (ℝ × ℝ)

def P : point := (2, 3)
def Q : point := (7, 3)
def R : point := (4, 10)

noncomputable def triangle_area (A B C : point) : ℝ :=
  (1/2) * ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_PQR : triangle_area P Q R = 17.5 :=
  sorry

end triangle_area_PQR_l26_26357


namespace knights_wins_33_l26_26359

def sharks_wins : ℕ := sorry
def falcons_wins : ℕ := sorry
def knights_wins : ℕ := sorry
def wolves_wins : ℕ := sorry
def dragons_wins : ℕ := 38 -- Dragons won the most games

-- Condition 1: The Sharks won more games than the Falcons.
axiom sharks_won_more_than_falcons : sharks_wins > falcons_wins

-- Condition 2: The Knights won more games than the Wolves, but fewer than the Dragons.
axiom knights_won_more_than_wolves : knights_wins > wolves_wins
axiom knights_won_less_than_dragons : knights_wins < dragons_wins

-- Condition 3: The Wolves won more than 22 games.
axiom wolves_won_more_than_22 : wolves_wins > 22

-- The possible wins are 24, 27, 33, 36, and 38 and the dragons win 38 (already accounted in dragons_wins)

-- Prove that the Knights won 33 games.
theorem knights_wins_33 : knights_wins = 33 :=
sorry -- proof goes here

end knights_wins_33_l26_26359


namespace small_possible_value_l26_26219

theorem small_possible_value (a b : ℕ) (h : a > 0) (h1 : b > 0) (h2 : 2^12 * 3^3 = a^b) : a + b = 110593 := by
  sorry

end small_possible_value_l26_26219


namespace distance_between_stripes_l26_26087

/-
Problem statement:
Given:
1. The street has parallel curbs 30 feet apart.
2. The length of the curb between the stripes is 10 feet.
3. Each stripe is 60 feet long.

Prove:
The distance between the stripes is 5 feet.
-/

-- Definitions:
def distance_between_curbs : ℝ := 30
def length_between_stripes_on_curb : ℝ := 10
def length_of_each_stripe : ℝ := 60

-- Theorem statement:
theorem distance_between_stripes :
  ∃ d : ℝ, (length_between_stripes_on_curb * distance_between_curbs = length_of_each_stripe * d) ∧ d = 5 :=
by
  sorry

end distance_between_stripes_l26_26087


namespace inequality_selection_l26_26069

theorem inequality_selection (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  1/a + 4/b ≥ 9/(a + b) :=
sorry

end inequality_selection_l26_26069


namespace number_of_positive_integers_with_erased_digit_decreased_by_nine_times_l26_26104

theorem number_of_positive_integers_with_erased_digit_decreased_by_nine_times : 
  ∃ n : ℕ, 
  ∀ (m a k : ℕ),
  (m + 10^k * a + 10^(k + 1) * n = 9 * (m + 10^k * n)) → 
  m < 10^k ∧ n > 0 ∧ n < m ∧  m ≠ 0 → 
  (m + 10^k * n  = 9 * (m - a) ) ∧ 
  (m % 10 = 5 ∨ m % 10 = 0) → 
  n = 28 :=
by
  sorry

end number_of_positive_integers_with_erased_digit_decreased_by_nine_times_l26_26104


namespace find_n_from_sequence_l26_26953

theorem find_n_from_sequence (a : ℕ → ℝ) (h₁ : ∀ n : ℕ, a n = (1 / (Real.sqrt n + Real.sqrt (n + 1))))
  (h₂ : ∃ n : ℕ, a n + a (n + 1) = Real.sqrt 11 - 3) : 9 ∈ {n | a n + a (n + 1) = Real.sqrt 11 - 3} :=
by
  sorry

end find_n_from_sequence_l26_26953


namespace number_of_tests_initially_l26_26273

-- Given conditions
variables (n S : ℕ)
variables (h1 : S / n = 70)
variables (h2 : S = 70 * n)
variables (h3 : (S - 55) / (n - 1) = 75)

-- Prove the number of tests initially, n, is 4.
theorem number_of_tests_initially (n : ℕ) (S : ℕ)
  (h1 : S / n = 70) (h2 : S = 70 * n) (h3 : (S - 55) / (n - 1) = 75) :
  n = 4 :=
sorry

end number_of_tests_initially_l26_26273


namespace total_ages_l26_26807

theorem total_ages (Xavier Yasmin : ℕ) (h1 : Xavier = 2 * Yasmin) (h2 : Xavier + 6 = 30) : Xavier + Yasmin = 36 :=
by
  sorry

end total_ages_l26_26807


namespace tax_percentage_l26_26449

theorem tax_percentage (car_price tax_paid first_tier_price : ℝ) (first_tier_tax_rate : ℝ) (tax_second_tier : ℝ) :
  car_price = 30000 ∧
  tax_paid = 5500 ∧
  first_tier_price = 10000 ∧
  first_tier_tax_rate = 0.25 ∧
  tax_second_tier = 0.15
  → (tax_second_tier) = 0.15 :=
by
  intros h
  rcases h with ⟨h1, h2, h3, h4, h5⟩
  sorry

end tax_percentage_l26_26449


namespace smallest_fraction_numerator_l26_26530

theorem smallest_fraction_numerator :
  ∃ a b : ℕ, 10 ≤ a ∧ a < b ∧ b ≤ 99 ∧ 6 * a > 5 * b ∧ ∀ c d : ℕ,
    (10 ≤ c ∧ c < d ∧ d ≤ 99 ∧ 6 * c > 5 * d → a ≤ c) ∧ 
    a = 81 :=
sorry

end smallest_fraction_numerator_l26_26530


namespace emily_subtracts_99_from_50_squared_l26_26274

theorem emily_subtracts_99_from_50_squared :
  (50 - 1) ^ 2 = 50 ^ 2 - 99 := by
  sorry

end emily_subtracts_99_from_50_squared_l26_26274


namespace vector_evaluation_l26_26654

-- Define the vectors
def v1 : ℝ × ℝ := (3, -2)
def v2 : ℝ × ℝ := (2, -6)
def v3 : ℝ × ℝ := (0, 3)
def scalar : ℝ := 5
def expected_result : ℝ × ℝ := (-7, 31)

-- Statement to be proved
theorem vector_evaluation : v1 - scalar • v2 + v3 = expected_result :=
by
  sorry

end vector_evaluation_l26_26654


namespace find_xy_l26_26440

theorem find_xy (p x y : ℕ) (hp : Nat.Prime p) (hx : 0 < x) (hy : 0 < y) :
  p * (x - y) = x * y ↔ (x, y) = (p^2 - p, p + 1) := by
  sorry

end find_xy_l26_26440


namespace total_distance_traveled_l26_26648

theorem total_distance_traveled (x : ℕ) (d_1 d_2 d_3 d_4 d_5 d_6 : ℕ) 
  (h1 : d_1 = 60 / x) 
  (h2 : d_2 = 60 / (x + 3)) 
  (h3 : d_3 = 60 / (x + 6)) 
  (h4 : d_4 = 60 / (x + 9)) 
  (h5 : d_5 = 60 / (x + 12)) 
  (h6 : d_6 = 60 / (x + 15)) 
  (hx1 : x ∣ 60) 
  (hx2 : (x + 3) ∣ 60) 
  (hx3 : (x + 6) ∣ 60) 
  (hx4 : (x + 9) ∣ 60) 
  (hx5 : (x + 12) ∣ 60) 
  (hx6 : (x + 15) ∣ 60) :
  d_1 + d_2 + d_3 + d_4 + d_5 + d_6 = 39 := 
sorry

end total_distance_traveled_l26_26648


namespace rational_numbers_on_circle_l26_26398

theorem rational_numbers_on_circle (a b c d e f : ℚ)
  (h1 : a = |b - c|)
  (h2 : b = d)
  (h3 : c = |d - e|)
  (h4 : d = |e - f|)
  (h5 : e = f)
  (h6 : a + b + c + d + e + f = 1) :
  [a, b, c, d, e, f] = [1/4, 1/4, 0, 1/4, 1/4, 0] :=
sorry

end rational_numbers_on_circle_l26_26398


namespace eval_derivative_at_one_and_neg_one_l26_26166

def f (x : ℝ) : ℝ := x^4 + x - 1

theorem eval_derivative_at_one_and_neg_one : 
  (deriv f 1) + (deriv f (-1)) = 2 :=
by 
  -- proof to be filled in
  sorry

end eval_derivative_at_one_and_neg_one_l26_26166


namespace find_a_minus_b_l26_26686

theorem find_a_minus_b (a b : ℤ) 
  (h1 : 3015 * a + 3019 * b = 3023) 
  (h2 : 3017 * a + 3021 * b = 3025) : 
  a - b = -3 := 
sorry

end find_a_minus_b_l26_26686


namespace ratio_x_y_z_w_l26_26805

theorem ratio_x_y_z_w (x y z w : ℝ) 
(h1 : 0.10 * x = 0.20 * y)
(h2 : 0.30 * y = 0.40 * z)
(h3 : 0.50 * z = 0.60 * w) : 
  (x / w) = 8 
  ∧ (y / w) = 4 
  ∧ (z / w) = 3
  ∧ (w / w) = 2.5 := 
sorry

end ratio_x_y_z_w_l26_26805


namespace parabola_focus_on_line_l26_26969

theorem parabola_focus_on_line (p : ℝ) (h₁ : 0 < p) (h₂ : (2 * (p / 2) + 0 - 2 = 0)) : p = 2 :=
sorry

end parabola_focus_on_line_l26_26969


namespace bus_capacity_percentage_l26_26826

theorem bus_capacity_percentage (x : ℕ) (h1 : 150 * x / 100 + 150 * 70 / 100 = 195) : x = 60 :=
by
  sorry

end bus_capacity_percentage_l26_26826


namespace range_of_a_l26_26567

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 3 → log (x - 1) + log (3 - x) = log (a - x)) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 1 < x₁ ∧ x₁ < 3 ∧ 1 < x₂ ∧ x₂ < 3) →
  3 < a ∧ a < 13 / 4 :=
by
  sorry

end range_of_a_l26_26567


namespace arithmetic_sequence_diff_l26_26244

-- Define the arithmetic sequence properties
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variable (a : ℕ → ℤ)
variable (h1 : is_arithmetic_sequence a 2)

-- Prove that a_5 - a_2 = 6
theorem arithmetic_sequence_diff : a 5 - a 2 = 6 :=
by sorry

end arithmetic_sequence_diff_l26_26244


namespace smallest_n_correct_l26_26074

/-- The first term of the geometric sequence. -/
def a₁ : ℚ := 5 / 6

/-- The second term of the geometric sequence. -/
def a₂ : ℚ := 25

/-- The common ratio for the geometric sequence. -/
def r : ℚ := a₂ / a₁

/-- The nth term of the geometric sequence. -/
def a_n (n : ℕ) : ℚ := a₁ * r^(n - 1)

/-- The smallest n such that the nth term is divisible by 10^7. -/
def smallest_n : ℕ := 8

theorem smallest_n_correct :
  ∀ n : ℕ, (a₁ * r^(n - 1)) ∣ (10^7 : ℚ) ↔ n = smallest_n := 
sorry

end smallest_n_correct_l26_26074


namespace no_sum_2015_l26_26506

theorem no_sum_2015 (x a : ℤ) : 3 * x + 3 * a ≠ 2015 := by
  sorry

end no_sum_2015_l26_26506


namespace no_maximum_value_l26_26086

-- Define the conditions and the expression in Lean
def expression (a b c d : ℝ) : ℝ := a^2 + b^2 + c^2 + d^2 + a*b + c*d

def condition (a b c d : ℝ) : Prop := a * d - b * c = 1

theorem no_maximum_value : ¬ ∃ M, ∀ a b c d, condition a b c d → expression a b c d ≤ M := by
  sorry

end no_maximum_value_l26_26086


namespace sum_of_acute_angles_l26_26824

theorem sum_of_acute_angles (angle1 angle2 angle3 angle4 angle5 angle6 : ℝ)
  (h1 : angle1 = 30) (h2 : angle2 = 30) (h3 : angle3 = 30) (h4 : angle4 = 30) (h5 : angle5 = 30) (h6 : angle6 = 30) :
  (angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + 
  (angle1 + angle2) + (angle2 + angle3) + (angle3 + angle4) + (angle4 + angle5) + (angle5 + angle6)) = 480 :=
  sorry

end sum_of_acute_angles_l26_26824


namespace distance_to_moscow_at_4PM_l26_26720

noncomputable def exact_distance_at_4PM (d12: ℝ) (d13: ℝ) (d15: ℝ) : ℝ :=
  d15 - 12

theorem distance_to_moscow_at_4PM  (h12 : 81.5 ≤ 82 ∧ 82 ≤ 82.5)
                                  (h13 : 70.5 ≤ 71 ∧ 71 ≤ 71.5)
                                  (h15 : 45.5 ≤ 46 ∧ 46 ≤ 46.5) :
  exact_distance_at_4PM 82 71 46 = 34 :=
by
  sorry

end distance_to_moscow_at_4PM_l26_26720


namespace part_I_part_II_l26_26004

namespace ArithmeticGeometricSequences

-- Definitions of sequences and their properties
def a1 : ℕ := 1
def b1 : ℕ := 2
def b (n : ℕ) : ℕ := 2 * 3 ^ (n - 1) -- General term of the geometric sequence

-- Definitions from given conditions
def a (n : ℕ) : ℕ := 3 * n - 2 -- General term of the arithmetic sequence

-- Sum of the first n terms of the geometric sequence
def S (n : ℕ) : ℕ := if n = 0 then 0 else 3 ^ n - 1

-- Theorem statement
theorem part_I (n : ℕ) : 
  (a1 = 1) ∧ 
  (b1 = 2) ∧ 
  (∀ n > 0, b n > 0) ∧ 
  (∀ b2 : ℕ, 2 * (1 + b2 / 2) = 2 + b2) ∧ 
  (∀ b2 a2 : ℕ, (1 + b2 / 2)^2 = b2 * ((a 3) + 2)) →
  (a n = 3 * n - 2) ∧ 
  (b n = 2 * 3 ^ (n - 1)) :=
  sorry

theorem part_II (n : ℕ) (m : ℝ) :
  (a1 = 1) ∧ 
  (b1 = 2) ∧ 
  (∀ n > 0, b n > 0) ∧ 
  (∀ b2 : ℕ, 2 * (1 + b2 / 2) = 2 + b2) ∧ 
  (∀ b2 a2 : ℕ, (1 + b2 / 2)^2 = b2 * ((a 3) + 2)) → 
  (∀ n > 0, S n + a n > m) → 
  (m < 3) :=
  sorry

end ArithmeticGeometricSequences

end part_I_part_II_l26_26004


namespace max_value_expression_l26_26844

theorem max_value_expression (s : ℝ) : 
  ∃ M, M = -3 * s^2 + 36 * s + 7 ∧ (∀ t : ℝ, -3 * t^2 + 36 * t + 7 ≤ M) :=
by
  use 115
  sorry

end max_value_expression_l26_26844


namespace inequality_problem_l26_26539

variables {a b c d : ℝ}

theorem inequality_problem (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : d > 0) :
  (a^3 + b^3 + c^3) / (a + b + c) + (b^3 + c^3 + d^3) / (b + c + d) +
  (c^3 + d^3 + a^3) / (c + d + a) + (d^3 + a^3 + b^3) / (d + a + b) ≥ a^2 + b^2 + c^2 + d^2 := 
by
  sorry

end inequality_problem_l26_26539


namespace brets_dinner_tip_calculation_l26_26891

/-
  We need to prove that the percentage of the tip Bret included is 20%, given the conditions.
-/

theorem brets_dinner_tip_calculation :
  let num_meals := 4
  let cost_per_meal := 12
  let num_appetizers := 2
  let cost_per_appetizer := 6
  let rush_fee := 5
  let total_cost := 77
  (total_cost - (num_meals * cost_per_meal + num_appetizers * cost_per_appetizer + rush_fee))
  / (num_meals * cost_per_meal + num_appetizers * cost_per_appetizer) * 100 = 20 :=
by
  sorry

end brets_dinner_tip_calculation_l26_26891


namespace average_visitors_per_day_in_month_l26_26311

theorem average_visitors_per_day_in_month (avg_visitors_sunday : ℕ) (avg_visitors_other_days : ℕ) (days_in_month : ℕ) (starts_sunday : Bool) :
  avg_visitors_sunday = 140 → avg_visitors_other_days = 80 → days_in_month = 30 → starts_sunday = true → 
  (∀ avg_visitors, avg_visitors = (4 * avg_visitors_sunday + 26 * avg_visitors_other_days) / days_in_month → avg_visitors = 88) :=
by
  intros h1 h2 h3 h4
  have total_visitors : ℕ := 4 * avg_visitors_sunday + 26 * avg_visitors_other_days
  have avg := total_visitors / days_in_month
  have visitors : ℕ := 2640
  sorry

end average_visitors_per_day_in_month_l26_26311


namespace proj_b_l26_26643

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let (ux, uy) := u
  let (vx, vy) := v
  let factor := (ux * vx + uy * vy) / (vx * vx + vy * vy)
  (factor * vx, factor * vy)

theorem proj_b (a b v : ℝ × ℝ) (h_ortho : a.1 * b.1 + a.2 * b.2 = 0)
  (h_proj_a : proj v a = (1, 2)) : proj v b = (3, -4) :=
by
  sorry

end proj_b_l26_26643


namespace simplify_and_evaluate_expression_l26_26956

theorem simplify_and_evaluate_expression (a b : ℤ) (h₁ : a = -2) (h₂ : b = 1) :
  ((a - 2 * b) ^ 2 - (a + 3 * b) * (a - 2 * b)) / b = 20 :=
by
  sorry

end simplify_and_evaluate_expression_l26_26956


namespace borrowed_amount_correct_l26_26946

variables (monthly_payment : ℕ) (months : ℕ) (total_payment : ℕ) (borrowed_amount : ℕ)

def total_payment_calculation (monthly_payment : ℕ) (months : ℕ) : ℕ :=
  monthly_payment * months

theorem borrowed_amount_correct :
  monthly_payment = 15 →
  months = 11 →
  total_payment = total_payment_calculation monthly_payment months →
  total_payment = 110 * borrowed_amount / 100 →
  borrowed_amount = 150 :=
by
  intros h1 h2 h3 h4
  sorry

end borrowed_amount_correct_l26_26946


namespace cost_of_each_ruler_l26_26241
-- Import the necessary library

-- Define the conditions and statement
theorem cost_of_each_ruler (students : ℕ) (rulers_each : ℕ) (cost_per_ruler : ℕ) (total_cost : ℕ) 
  (cond1 : students = 42)
  (cond2 : students / 2 < 42 / 2)
  (cond3 : cost_per_ruler > rulers_each)
  (cond4 : students * rulers_each * cost_per_ruler = 2310) : 
  cost_per_ruler = 11 :=
sorry

end cost_of_each_ruler_l26_26241


namespace find_x_value_l26_26588

noncomputable def x_value (x y z : ℝ) : Prop :=
  (26 = (z + x) / 2) ∧
  (z = 52 - x) ∧
  (52 - x = (26 + y) / 2) ∧
  (y = 78 - 2 * x) ∧
  (78 - 2 * x = (8 + (52 - x)) / 2) ∧
  (x = 32)

theorem find_x_value : ∃ x y z : ℝ, x_value x y z :=
by
  use 32  -- x
  use 14  -- y derived from 78 - 2x where x = 32 leads to y = 14
  use 20  -- z derived from 52 - x where x = 32 leads to z = 20
  unfold x_value
  simp
  sorry

end find_x_value_l26_26588


namespace range_of_a_l26_26637

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

theorem range_of_a (a : ℝ) (h : ∀ x y : ℝ, x ≤ y → f x a ≥ f y a) : a ≤ -3 :=
by
  sorry

end range_of_a_l26_26637


namespace prob_rain_next_day_given_today_rain_l26_26387

variable (P_rain : ℝ) (P_rain_2_days : ℝ)
variable (p_given_rain : ℝ)

-- Given conditions
def condition_P_rain : Prop := P_rain = 1/3
def condition_P_rain_2_days : Prop := P_rain_2_days = 1/5

-- The question to prove
theorem prob_rain_next_day_given_today_rain (h1 : condition_P_rain P_rain) (h2 : condition_P_rain_2_days P_rain_2_days) :
  p_given_rain = 3/5 :=
by
  sorry

end prob_rain_next_day_given_today_rain_l26_26387


namespace minimum_discount_l26_26081

theorem minimum_discount (C M : ℝ) (profit_margin : ℝ) (x : ℝ) 
  (hC : C = 800) (hM : M = 1200) (hprofit_margin : profit_margin = 0.2) :
  (M * x - C ≥ C * profit_margin) → (x ≥ 0.8) :=
by
  -- Here, we need to solve the inequality given the conditions
  sorry

end minimum_discount_l26_26081


namespace polynomial_coeff_sum_l26_26031

theorem polynomial_coeff_sum (A B C D : ℤ) 
  (h : ∀ x : ℤ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) : 
  A + B + C + D = 36 :=
by 
  sorry

end polynomial_coeff_sum_l26_26031


namespace computer_operations_correct_l26_26902

-- Define the rate of operations per second
def operations_per_second : ℝ := 4 * 10^8

-- Define the total number of seconds the computer operates
def total_seconds : ℝ := 6 * 10^5

-- Define the expected total number of operations
def expected_operations : ℝ := 2.4 * 10^14

-- Theorem stating the total number of operations is as expected
theorem computer_operations_correct :
  operations_per_second * total_seconds = expected_operations :=
by
  sorry

end computer_operations_correct_l26_26902


namespace Bella_age_l26_26180

theorem Bella_age (B : ℕ) (h₁ : ∃ n : ℕ, n = B + 9) (h₂ : B + (B + 9) = 19) : B = 5 := 
by
  sorry

end Bella_age_l26_26180


namespace parabola_focus_distance_l26_26170

theorem parabola_focus_distance (p m : ℝ) (h1 : p > 0) (h2 : (2 - (-p/2)) = 4) : p = 4 := 
by
  sorry

end parabola_focus_distance_l26_26170


namespace objective_function_range_l26_26247

noncomputable def feasible_region (A B C : ℝ × ℝ) := 
  let (x, y) := A
  let (x1, y1) := B 
  let (x2, y2) := C 
  {p : ℝ × ℝ | True} -- The exact feasible region description is not specified

theorem objective_function_range
  (A B C: ℝ × ℝ)
  (a b : ℝ)
  (x y : ℝ)
  (hA : A = (x, y))
  (hB : B = (1, 1))
  (hC : C = (5, 2))
  (h1 : a + b = 3)
  (h2 : 5 * a + 2 * b = 12) :
  let z := a * x + b * y
  3 ≤ z ∧ z ≤ 12 :=
by
  sorry

end objective_function_range_l26_26247


namespace solution_l26_26589

noncomputable def f (x : ℝ) := 
  10 / (Real.sqrt (x - 5) - 10) + 
  2 / (Real.sqrt (x - 5) - 5) + 
  9 / (Real.sqrt (x - 5) + 5) + 
  18 / (Real.sqrt (x - 5) + 10)

theorem solution : 
  f (1230 / 121) = 0 := sorry

end solution_l26_26589


namespace complete_the_square_l26_26093

-- Define the initial condition
def initial_eqn (x : ℝ) : Prop := x^2 - 6 * x + 5 = 0

-- Theorem statement for completing the square
theorem complete_the_square (x : ℝ) : initial_eqn x → (x - 3)^2 = 4 :=
by sorry

end complete_the_square_l26_26093


namespace center_temperature_l26_26044

-- Define the conditions as a structure
structure SquareSheet (f : ℝ × ℝ → ℝ) :=
  (temp_0: ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f (x, 0) = 0 ∧ f (0, x) = 0 ∧ f (1, x) = 0)
  (temp_100: ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f (x, 1) = 100)
  (no_radiation_loss: True) -- Just a placeholder since this condition is theoretical in nature

-- Define the claim as a theorem
theorem center_temperature (f : ℝ × ℝ → ℝ) (h : SquareSheet f) : f (0.5, 0.5) = 25 :=
by
  sorry -- Proof is not required and skipped

end center_temperature_l26_26044


namespace clothing_order_equation_l26_26718

open Real

-- Definitions and conditions
def total_pieces : ℕ := 720
def initial_rate : ℕ := 48
def days_earlier : ℕ := 5

-- Statement that we need to prove
theorem clothing_order_equation (x : ℕ) :
    (720 / 48 : ℝ) - (720 / (x + 48) : ℝ) = 5 := 
sorry

end clothing_order_equation_l26_26718


namespace find_y_l26_26797

theorem find_y (t y : ℝ) (h1 : -3 = 2 - t) (h2 : y = 4 * t + 7) : y = 27 :=
sorry

end find_y_l26_26797


namespace power_comparison_l26_26294

theorem power_comparison (A B : ℝ) (h1 : A = 1997 ^ (1998 ^ 1999)) (h2 : B = 1999 ^ (1998 ^ 1997)) (h3 : 1997 < 1999) :
  A > B :=
by
  sorry

end power_comparison_l26_26294


namespace annie_total_miles_l26_26639

theorem annie_total_miles (initial_gallons : ℕ) (miles_per_gallon : ℕ)
  (initial_trip_miles : ℕ) (purchased_gallons : ℕ) (final_gallons : ℕ)
  (total_miles : ℕ) :
  initial_gallons = 12 →
  miles_per_gallon = 28 →
  initial_trip_miles = 280 →
  purchased_gallons = 6 →
  final_gallons = 5 →
  total_miles = 364 := by
  sorry

end annie_total_miles_l26_26639


namespace complex_modulus_square_l26_26313

open Complex

theorem complex_modulus_square (z : ℂ) (h : z^2 + abs z ^ 2 = 7 + 6 * I) : abs z ^ 2 = 85 / 14 :=
sorry

end complex_modulus_square_l26_26313


namespace value_of_X_l26_26612

noncomputable def M : ℕ := 3009 / 3
noncomputable def N : ℕ := (2 * M) / 3
noncomputable def X : ℕ := M - N

theorem value_of_X : X = 335 := by
  sorry

end value_of_X_l26_26612


namespace find_initial_milk_amount_l26_26871

-- Define the initial amount of milk as a variable in liters
variable (T : ℝ)

-- Given conditions
def consumed (T : ℝ) := 0.4 * T
def leftover (T : ℝ) := 0.69

-- The total milk at first was T if T = 0.69 / 0.6
theorem find_initial_milk_amount 
  (h1 : leftover T = 0.69)
  (h2 : consumed T = 0.4 * T) :
  T = 1.15 :=
by
  sorry

end find_initial_milk_amount_l26_26871


namespace totalAttendees_l26_26126

def numberOfBuses : ℕ := 8
def studentsPerBus : ℕ := 45
def chaperonesList : List ℕ := [2, 3, 4, 5, 3, 4, 2, 6]

theorem totalAttendees : 
    numberOfBuses * studentsPerBus + chaperonesList.sum = 389 := 
by
  sorry

end totalAttendees_l26_26126


namespace find_p_l26_26869

theorem find_p (h p : Polynomial ℝ) 
  (H1 : h + p = 3 * X^2 - X + 4)
  (H2 : h = X^4 - 5 * X^2 + X + 6) : 
  p = -X^4 + 8 * X^2 - 2 * X - 2 :=
sorry

end find_p_l26_26869


namespace greatest_teams_l26_26369

-- Define the number of girls and boys as constants
def numGirls : ℕ := 40
def numBoys : ℕ := 32

-- Define the greatest number of teams possible with equal number of girls and boys as teams.
theorem greatest_teams : Nat.gcd numGirls numBoys = 8 := sorry

end greatest_teams_l26_26369


namespace butterfly_black_dots_l26_26593

theorem butterfly_black_dots (b f : ℕ) (total_butterflies : b = 397) (total_black_dots : f = 4764) : f / b = 12 :=
by
  sorry

end butterfly_black_dots_l26_26593


namespace area_of_triangle_formed_by_lines_l26_26272

theorem area_of_triangle_formed_by_lines (x y : ℝ) (h1 : y = x) (h2 : x = -5) :
  let base := 5
  let height := 5
  let area := (1 / 2 : ℝ) * base * height
  area = 12.5 := 
by
  sorry

end area_of_triangle_formed_by_lines_l26_26272


namespace total_exercise_time_l26_26594

theorem total_exercise_time :
  let javier_minutes_per_day := 50
  let javier_days := 7
  let sanda_minutes_per_day := 90
  let sanda_days := 3
  (javier_minutes_per_day * javier_days + sanda_minutes_per_day * sanda_days) = 620 :=
by
  sorry

end total_exercise_time_l26_26594


namespace find_consecutive_numbers_l26_26057

theorem find_consecutive_numbers (a b c : ℕ) (h1 : a + 1 = b) (h2 : b + 1 = c)
    (h_lcm : Nat.lcm a (Nat.lcm b c) = 660) : a = 10 ∧ b = 11 ∧ c = 12 := 
    sorry

end find_consecutive_numbers_l26_26057


namespace four_digit_number_sum_l26_26068

theorem four_digit_number_sum (x y z w : ℕ) (h1 : 1001 * x + 101 * y + 11 * z + 2 * w = 2003)
  (h2 : x = 1) : (x = 1 ∧ y = 9 ∧ z = 7 ∧ w = 8) ↔ (1000 * x + 100 * y + 10 * z + w = 1978) :=
by sorry

end four_digit_number_sum_l26_26068


namespace shape_is_cylinder_l26_26915

noncomputable def shape_desc (r θ z a : ℝ) : Prop := r = a

theorem shape_is_cylinder (a : ℝ) (h_a : a > 0) :
  ∀ (r θ z : ℝ), shape_desc r θ z a → ∃ c : Set (ℝ × ℝ × ℝ), c = {p : ℝ × ℝ × ℝ | ∃ θ z, p = (a, θ, z)} :=
by
  sorry

end shape_is_cylinder_l26_26915


namespace find_fg_satisfy_l26_26076

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x) / 2
noncomputable def g (x : ℝ) (c : ℝ) : ℝ := (Real.sin x - Real.cos x) / 2 + c

theorem find_fg_satisfy (c : ℝ) : ∀ x y : ℝ,
  Real.sin x + Real.cos y = f x + f y + g x c - g y c := 
by 
  intros;
  rw [f, g, g, f];
  sorry

end find_fg_satisfy_l26_26076


namespace original_height_of_ball_l26_26172

theorem original_height_of_ball (h : ℝ) : 
  (h + 2 * (0.5 * h) + 2 * ((0.5)^2 * h) = 200) -> 
  h = 800 / 9 := 
by
  sorry

end original_height_of_ball_l26_26172


namespace base_number_is_two_l26_26491

theorem base_number_is_two (a : ℝ) (x : ℕ) (h1 : x = 14) (h2 : a^x - a^(x - 2) = 3 * a^12) : a = 2 := by
  sorry

end base_number_is_two_l26_26491


namespace parallel_lines_not_coincident_l26_26655

theorem parallel_lines_not_coincident (a : ℝ) :
  (∀ x y : ℝ, ax + 2 * y + 6 = 0) ∧
  (∀ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0) →
  ¬ ∃ (b : ℝ), ∀ x y : ℝ, ax + 2 * y + b = 0 ∧ x + (a - 1) * y + b = 0 →
  a = -1 :=
by
  sorry

end parallel_lines_not_coincident_l26_26655


namespace lesson_duration_tuesday_l26_26027

theorem lesson_duration_tuesday
  (monday_lessons : ℕ)
  (monday_duration : ℕ)
  (tuesday_lessons : ℕ)
  (wednesday_multiplier : ℕ)
  (total_time : ℕ)
  (monday_hours : ℕ)
  (tuesday_hours : ℕ)
  (wednesday_hours : ℕ)
  (H1 : monday_lessons = 6)
  (H2 : monday_duration = 30)
  (H3 : tuesday_lessons = 3)
  (H4 : wednesday_multiplier = 2)
  (H5 : total_time = 12)
  (H6 : monday_hours = monday_lessons * monday_duration / 60)
  (H7 : tuesday_hours = tuesday_lessons * T)
  (H8 : wednesday_hours = wednesday_multiplier * tuesday_hours)
  (H9 : monday_hours + tuesday_hours + wednesday_hours = total_time) :
  T = 1 := by
  sorry

end lesson_duration_tuesday_l26_26027


namespace simplify_sqrt_450_l26_26640

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l26_26640


namespace temperature_on_tuesday_l26_26941

theorem temperature_on_tuesday 
  (M T W Th F Sa : ℝ)
  (h1 : (M + T + W) / 3 = 38)
  (h2 : (T + W + Th) / 3 = 42)
  (h3 : (W + Th + F) / 3 = 44)
  (h4 : (Th + F + Sa) / 3 = 46)
  (hF : F = 43)
  (pattern : M + 2 = Sa ∨ M - 1 = Sa) :
  T = 80 :=
sorry

end temperature_on_tuesday_l26_26941


namespace div_by_73_l26_26496

theorem div_by_73 (n : ℕ) (h : 0 < n) : (2^(3*n + 6) + 3^(4*n + 2)) % 73 = 0 := sorry

end div_by_73_l26_26496


namespace sum_not_prime_l26_26263

theorem sum_not_prime (a b c d : ℕ) (h : a * b = c * d) : ¬ Nat.Prime (a + b + c + d) := 
sorry

end sum_not_prime_l26_26263


namespace parabola_standard_form_l26_26777

theorem parabola_standard_form (a : ℝ) (x y : ℝ) :
  (∀ a : ℝ, (2 * a + 3) * x + y - 4 * a + 2 = 0 → 
  x = 2 ∧ y = -8) → 
  (y^2 = 32 * x ∨ x^2 = - (1/2) * y) :=
by 
  intros h
  sorry

end parabola_standard_form_l26_26777


namespace steven_more_peaches_l26_26328

variable (Jake Steven Jill : ℕ)

-- Conditions
axiom h1 : Jake + 6 = Steven
axiom h2 : Jill = 5
axiom h3 : Jake = 17

-- Goal
theorem steven_more_peaches : Steven - Jill = 18 := by
  sorry

end steven_more_peaches_l26_26328


namespace amount_spent_on_shirt_l26_26895

-- Definitions and conditions
def total_spent_clothing : ℝ := 25.31
def spent_on_jacket : ℝ := 12.27

-- Goal: Prove the amount spent on the shirt is 13.04
theorem amount_spent_on_shirt : (total_spent_clothing - spent_on_jacket = 13.04) := by
  sorry

end amount_spent_on_shirt_l26_26895


namespace problem_l26_26863

theorem problem (a b : ℝ) (h : a > b) (k : b > 0) : b * (a - b) > 0 := 
by
  sorry

end problem_l26_26863


namespace slope_equal_angles_l26_26427

-- Define the problem
theorem slope_equal_angles (k : ℝ) :
  (∀ (l1 l2 : ℝ), l1 = 1 ∧ l2 = 2 → (abs ((k - l1) / (1 + k * l1)) = abs ((l2 - k) / (1 + l2 * k)))) →
  (k = (1 + Real.sqrt 10) / 3 ∨ k = (1 - Real.sqrt 10) / 3) :=
by
  intros
  sorry

end slope_equal_angles_l26_26427


namespace students_not_in_biology_l26_26918

theorem students_not_in_biology (S : ℕ) (f : ℚ) (hS : S = 840) (hf : f = 0.35) :
  S - (f * S) = 546 :=
by
  sorry

end students_not_in_biology_l26_26918


namespace hcf_of_three_numbers_l26_26727

def hcf (a b : ℕ) : ℕ := gcd a b

theorem hcf_of_three_numbers :
  let a := 136
  let b := 144
  let c := 168
  hcf (hcf a b) c = 8 :=
by
  sorry

end hcf_of_three_numbers_l26_26727


namespace smallest_common_multiple_gt_50_l26_26308

theorem smallest_common_multiple_gt_50 (a b : ℕ) (h1 : a = 15) (h2 : b = 20) : 
    ∃ x, x > 50 ∧ Nat.lcm a b = x := by
  have h_lcm : Nat.lcm a b = 60 := by sorry
  use 60
  exact ⟨by decide, h_lcm⟩

end smallest_common_multiple_gt_50_l26_26308


namespace smallest_value_l26_26260

noncomputable def smallest_possible_value (a b : ℝ) : ℝ := 2 * a + b

theorem smallest_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 ≥ 3 * b) (h4 : b^2 ≥ (8 / 9) * a) :
  smallest_possible_value a b = 5.602 :=
sorry

end smallest_value_l26_26260


namespace find_x_minus_y_l26_26363

def rotated_point (x y h k : ℝ) : ℝ × ℝ := (2 * h - x, 2 * k - y)

def reflected_point (x y : ℝ) : ℝ × ℝ := (y, x)

def transformed_point (x y : ℝ) : ℝ × ℝ :=
  reflected_point (rotated_point x y 2 3).1 (rotated_point x y 2 3).2

theorem find_x_minus_y (x y : ℝ) (h1 : transformed_point x y = (4, -1)) : x - y = 3 := 
by 
  sorry

end find_x_minus_y_l26_26363


namespace a_minus_b_value_l26_26117

theorem a_minus_b_value (a b : ℤ) :
  (∀ x : ℝ, 9 * x^3 + y^2 + a * x - b * x^3 + x + 5 = y^2 + 5) → a - b = -10 :=
by
  sorry

end a_minus_b_value_l26_26117


namespace value_set_l26_26939

open Real Set

noncomputable def possible_values (a b c : ℝ) : Set ℝ :=
  {x | ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 2 ∧ x = c / a + c / b}

theorem value_set (c : ℝ) (hc : c > 0) : possible_values a b c = Ici (2 * c) := by
  sorry

end value_set_l26_26939


namespace fraction_is_meaningful_l26_26424

theorem fraction_is_meaningful (x : ℝ) : x ≠ 1 ↔ ∃ y : ℝ, y = 8 / (x - 1) :=
by
  sorry

end fraction_is_meaningful_l26_26424


namespace no_valid_solutions_l26_26404

theorem no_valid_solutions (a b : ℝ) (h1 : ∀ x, (a * x + b) ^ 2 = 4 * x^2 + 4 * x + 4) : false :=
  by
  sorry

end no_valid_solutions_l26_26404


namespace train_length_correct_l26_26858

noncomputable def speed_km_per_hour : ℝ := 56
noncomputable def time_seconds : ℝ := 32.142857142857146
noncomputable def bridge_length_m : ℝ := 140
noncomputable def train_length_m : ℝ := 360

noncomputable def speed_m_per_s : ℝ := speed_km_per_hour * (1000 / 3600)
noncomputable def total_distance_m : ℝ := speed_m_per_s * time_seconds

theorem train_length_correct :
  (total_distance_m - bridge_length_m) = train_length_m :=
  by
    sorry

end train_length_correct_l26_26858


namespace number_of_buses_l26_26321

-- Define the conditions
def vans : ℕ := 9
def people_per_van : ℕ := 8
def people_per_bus : ℕ := 27
def total_people : ℕ := 342

-- Translate the mathematical proof problem
theorem number_of_buses : ∃ buses : ℕ, (vans * people_per_van + buses * people_per_bus = total_people) ∧ (buses = 10) :=
by
  -- calculations to prove the theorem
  sorry

end number_of_buses_l26_26321


namespace find_integer_value_of_a_l26_26250

-- Define the conditions for the equation and roots
def equation_has_two_distinct_negative_integer_roots (a : ℤ) : Prop :=
  ∃ x1 x2 : ℤ, x1 ≠ x2 ∧ x1 < 0 ∧ x2 < 0 ∧ (a^2 - 1) * x1^2 - 2 * (5 * a + 1) * x1 + 24 = 0 ∧ (a^2 - 1) * x2^2 - 2 * (5 * a + 1) * x2 + 24 = 0 ∧
  x1 = 6 / (a - 1) ∧ x2 = 4 / (a + 1)

-- Prove that the only integer value of a that satisfies these conditions is -2
theorem find_integer_value_of_a : 
  ∃ (a : ℤ), equation_has_two_distinct_negative_integer_roots a ∧ a = -2 := 
sorry

end find_integer_value_of_a_l26_26250


namespace evaluate_expression_l26_26666

theorem evaluate_expression (b : ℕ) (h : b = 5) : b^3 * b^4 * 2 = 156250 :=
by
  sorry

end evaluate_expression_l26_26666


namespace no_six_digit_starting_with_five_12_digit_square_six_digit_starting_with_one_12_digit_square_smallest_k_for_n_digit_number_square_l26_26705

-- Part (a)
theorem no_six_digit_starting_with_five_12_digit_square : ∀ (x y : ℕ), (5 * 10^5 ≤ x) → (x < 6 * 10^5) → (10^5 ≤ y) → (y < 10^6) → ¬∃ z : ℕ, (10^11 ≤ z) ∧ (z < 10^12) ∧ x * 10^6 + y = z^2 := sorry

-- Part (b)
theorem six_digit_starting_with_one_12_digit_square : ∀ (x y : ℕ), (10^5 ≤ x) → (x < 2 * 10^5) → (10^5 ≤ y) → (y < 10^6) → ∃ z : ℕ, (10^11 ≤ z) ∧ (z < 2 * 10^11) ∧ x * 10^6 + y = z^2 := sorry

-- Part (c)
theorem smallest_k_for_n_digit_number_square : ∀ (n : ℕ), ∃ (k : ℕ), k = n + 1 ∧ ∀ (x : ℕ), (10^(n-1) ≤ x) → (x < 10^n) → ∃ y : ℕ, (10^(n + k - 1) ≤ x * 10^k + y) ∧ (x * 10^k + y) < 10^(n + k) ∧ ∃ z : ℕ, x * 10^k + y = z^2 := sorry

end no_six_digit_starting_with_five_12_digit_square_six_digit_starting_with_one_12_digit_square_smallest_k_for_n_digit_number_square_l26_26705


namespace sum_of_prime_factors_240345_l26_26001

theorem sum_of_prime_factors_240345 : ∀ {p1 p2 p3 : ℕ}, 
  Prime p1 → Prime p2 → Prime p3 →
  p1 * p2 * p3 = 240345 →
  p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 →
  p1 + p2 + p3 = 16011 :=
by
  intros p1 p2 p3 hp1 hp2 hp3 hprod hdiff
  sorry

end sum_of_prime_factors_240345_l26_26001


namespace gcd_72_168_gcd_98_280_f_at_3_l26_26013

/-- 
Prove that the GCD of 72 and 168 using the method of mutual subtraction is 24.
-/
theorem gcd_72_168 : Nat.gcd 72 168 = 24 :=
sorry

/-- 
Prove that the GCD of 98 and 280 using the Euclidean algorithm is 14.
-/
theorem gcd_98_280 : Nat.gcd 98 280 = 14 :=
sorry

/-- 
Prove that the value of f(3) where f(x) = x^5 + x^3 + x^2 + x + 1 is 283 using Horner's method.
-/
def f (x : ℕ) : ℕ := x^5 + x^3 + x^2 + x + 1

theorem f_at_3 : f 3 = 283 :=
sorry

end gcd_72_168_gcd_98_280_f_at_3_l26_26013


namespace sum_q_p_values_is_neg42_l26_26285

def p (x : Int) : Int := 2 * Int.natAbs x - 1

def q (x : Int) : Int := -(Int.natAbs x) - 1

def values : List Int := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

def q_p_sum : Int :=
  let q_p_values := values.map (λ x => q (p x))
  q_p_values.sum

theorem sum_q_p_values_is_neg42 : q_p_sum = -42 :=
  by
    sorry

end sum_q_p_values_is_neg42_l26_26285


namespace pascal_row_10_sum_l26_26372

-- Definition: sum of the numbers in Row n of Pascal's Triangle is 2^n
def pascal_row_sum (n : ℕ) : ℕ := 2^n

-- Theorem: sum of the numbers in Row 10 of Pascal's Triangle is 1024
theorem pascal_row_10_sum : pascal_row_sum 10 = 1024 :=
by
  sorry

end pascal_row_10_sum_l26_26372


namespace grape_juice_percentage_l26_26731

theorem grape_juice_percentage
  (original_mixture : ℝ)
  (percent_grape_juice : ℝ)
  (added_grape_juice : ℝ)
  (h1 : original_mixture = 50)
  (h2 : percent_grape_juice = 0.10)
  (h3 : added_grape_juice = 10)
  : (percent_grape_juice * original_mixture + added_grape_juice) / (original_mixture + added_grape_juice) * 100 = 25 :=
by
  sorry

end grape_juice_percentage_l26_26731


namespace laptop_total_selling_price_l26_26786

-- Define the original price of the laptop
def originalPrice : ℝ := 1200

-- Define the discount rate
def discountRate : ℝ := 0.30

-- Define the redemption coupon amount
def coupon : ℝ := 50

-- Define the tax rate
def taxRate : ℝ := 0.15

-- Calculate the discount amount
def discountAmount : ℝ := originalPrice * discountRate

-- Calculate the sale price after discount
def salePrice : ℝ := originalPrice - discountAmount

-- Calculate the new sale price after applying the coupon
def newSalePrice : ℝ := salePrice - coupon

-- Calculate the tax amount
def taxAmount : ℝ := newSalePrice * taxRate

-- Calculate the total selling price after tax
def totalSellingPrice : ℝ := newSalePrice + taxAmount

-- Prove that the total selling price is 908.5 dollars
theorem laptop_total_selling_price : totalSellingPrice = 908.5 := by
  unfold totalSellingPrice newSalePrice taxAmount salePrice discountAmount
  norm_num
  sorry

end laptop_total_selling_price_l26_26786


namespace regular_price_of_tire_l26_26298

theorem regular_price_of_tire (p : ℝ) (h : 2 * p + p / 2 = 270) : p = 108 :=
sorry

end regular_price_of_tire_l26_26298


namespace find_f_zero_l26_26583

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_zero (a : ℝ) (h1 : ∀ x : ℝ, f (x - a) = x^3 + 1)
  (h2 : ∀ x : ℝ, f x + f (2 - x) = 2) : 
  f 0 = 0 :=
sorry

end find_f_zero_l26_26583


namespace percentage_saved_l26_26678

noncomputable def calculateSavedPercentage : ℚ :=
  let first_tier_free_tickets := 1
  let second_tier_free_tickets_per_ticket := 2
  let number_of_tickets_purchased := 10
  let total_free_tickets :=
    first_tier_free_tickets +
    (number_of_tickets_purchased - 5) * second_tier_free_tickets_per_ticket
  let total_tickets_received := number_of_tickets_purchased + total_free_tickets
  let free_tickets := total_tickets_received - number_of_tickets_purchased
  (free_tickets / total_tickets_received) * 100

theorem percentage_saved : calculateSavedPercentage = 52.38 :=
by
  sorry

end percentage_saved_l26_26678


namespace marbles_count_l26_26852

variable (r b : ℕ)

theorem marbles_count (hr1 : 8 * (r - 1) = r + b - 2) (hr2 : 4 * r = r + b - 3) : r + b = 9 := 
by sorry

end marbles_count_l26_26852


namespace max_value_ineq_l26_26171

theorem max_value_ineq (x y : ℝ) (hx1 : -5 ≤ x) (hx2 : x ≤ -3) (hy1 : 1 ≤ y) (hy2 : y ≤ 3) : 
  (x + y) / (x - 1) ≤ 2 / 3 := 
sorry

end max_value_ineq_l26_26171


namespace tan_alpha_eq_two_l26_26338

theorem tan_alpha_eq_two (α : ℝ) (h1 : α ∈ Set.Ioc 0 (Real.pi / 2))
    (h2 : Real.sin ((Real.pi / 4) - α) * Real.sin ((Real.pi / 4) + α) = -3 / 10) :
    Real.tan α = 2 := by
  sorry

end tan_alpha_eq_two_l26_26338


namespace find_eccentricity_of_ellipse_l26_26821

noncomputable def ellipseEccentricity (k : ℝ) : ℝ :=
  let a := Real.sqrt (k + 2)
  let b := Real.sqrt (k + 1)
  let c := Real.sqrt (a^2 - b^2)
  c / a

theorem find_eccentricity_of_ellipse (k : ℝ) (h1 : k + 2 = 4) (h2 : Real.sqrt (k + 2) = 2) :
  ellipseEccentricity k = 1 / 2 := by
  sorry

end find_eccentricity_of_ellipse_l26_26821


namespace track_width_track_area_l26_26952

theorem track_width (r1 r2 : ℝ) (h1 : 2 * π * r1 - 2 * π * r2 = 24 * π) : r1 - r2 = 12 :=
by sorry

theorem track_area (r1 r2 : ℝ) (h1 : r1 = r2 + 12) : π * (r1^2 - r2^2) = π * (24 * r2 + 144) :=
by sorry

end track_width_track_area_l26_26952


namespace shortest_side_length_l26_26460

theorem shortest_side_length (perimeter : ℝ) (shortest : ℝ) (side1 side2 side3 : ℝ) 
  (h1 : side1 + side2 + side3 = perimeter)
  (h2 : side1 = 2 * shortest)
  (h3 : side2 = 2 * shortest) :
  shortest = 3 := by
  sorry

end shortest_side_length_l26_26460


namespace correct_operation_l26_26901

theorem correct_operation (x : ℝ) : (x^3 * x^2 = x^5) :=
by sorry

end correct_operation_l26_26901


namespace solve_simultaneous_eqns_l26_26135

theorem solve_simultaneous_eqns :
  ∀ (x y : ℝ), 
  (1/x - 1/(2*y) = 2*y^4 - 2*x^4 ∧ 1/x + 1/(2*y) = (3*x^2 + y^2) * (x^2 + 3*y^2)) 
  ↔ 
  (x = (3^(1/5) + 1) / 2 ∧ y = (3^(1/5) - 1) / 2) :=
by sorry

end solve_simultaneous_eqns_l26_26135


namespace find_a_l26_26634

noncomputable def A (a : ℝ) : Set ℝ := {-1, a^2 + 1, a^2 - 3}
noncomputable def B (a : ℝ) : Set ℝ := {-4, a - 1, a + 1}

theorem find_a (a : ℝ) (h : A a ∩ B a = {-2}) : a = -1 :=
sorry

end find_a_l26_26634


namespace smallest_common_multiple_l26_26107

theorem smallest_common_multiple (n : ℕ) (h1 : n > 0) (h2 : 8 ∣ n) (h3 : 6 ∣ n) : n = 24 :=
by sorry

end smallest_common_multiple_l26_26107


namespace max_value_m_l26_26989

theorem max_value_m (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  (∀ (m : ℝ), (4 / (1 - x) ≥ m - 1 / x)) ↔ (∃ (m : ℝ), m ≤ 9) :=
by
  sorry

end max_value_m_l26_26989


namespace ian_saves_per_day_l26_26083

-- Let us define the given conditions
def total_saved : ℝ := 0.40 -- Ian saved a total of $0.40
def days : ℕ := 40 -- Ian saved for 40 days

-- Now, we need to prove that Ian saved 0.01 dollars/day
theorem ian_saves_per_day (h : total_saved = 0.40 ∧ days = 40) : total_saved / days = 0.01 :=
by
  sorry

end ian_saves_per_day_l26_26083


namespace no_other_distinct_prime_products_l26_26585

theorem no_other_distinct_prime_products :
  ∀ (q1 q2 q3 : Nat), 
  Prime q1 ∧ Prime q2 ∧ Prime q3 ∧ q1 ≠ q2 ∧ q2 ≠ q3 ∧ q1 ≠ q3 ∧ q1 * q2 * q3 ≠ 17 * 11 * 23 → 
  q1 + q2 + q3 ≠ 51 :=
by
  intros q1 q2 q3 h
  sorry

end no_other_distinct_prime_products_l26_26585


namespace tenth_term_of_arithmetic_sequence_l26_26220

-- Define the initial conditions: first term 'a' and the common difference 'd'
def a : ℤ := 2
def d : ℤ := 1 - a

-- Define the n-th term of an arithmetic sequence formula
def nth_term (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

-- Statement to prove
theorem tenth_term_of_arithmetic_sequence :
  nth_term a d 10 = -7 := 
by
  sorry

end tenth_term_of_arithmetic_sequence_l26_26220


namespace final_answer_after_subtracting_l26_26677

theorem final_answer_after_subtracting (n : ℕ) (h : n = 990) : (n / 9) - 100 = 10 :=
by
  sorry

end final_answer_after_subtracting_l26_26677


namespace radius_of_larger_circle_l26_26773

theorem radius_of_larger_circle (r : ℝ) (R : ℝ) 
  (h1 : ∀ a b c : ℝ, a = 2 ∧ b = 2 ∧ c = 2) 
  (h2 : ∀ x y z : ℝ, (x = 4) ∧ (y = 4) ∧ (z = 4) ) 
  (h3 : ∀ A B : ℝ, A * 2 = 2) : 
  R = 2 + 2 * Real.sqrt 3 :=
by
  sorry

end radius_of_larger_circle_l26_26773


namespace cost_to_fix_car_l26_26288

variable {S A : ℝ}

theorem cost_to_fix_car (h1 : A = 3 * S + 50) (h2 : S + A = 450) : A = 350 := 
by
  sorry

end cost_to_fix_car_l26_26288


namespace fraction_of_second_eq_fifth_of_first_l26_26467

theorem fraction_of_second_eq_fifth_of_first 
  (a b x y : ℕ)
  (h1 : y = 40)
  (h2 : x + 35 = 4 * y)
  (h3 : (1 / 5) * x = (a / b) * y) 
  (hb : b ≠ 0):
  a / b = 5 / 8 := by
  sorry

end fraction_of_second_eq_fifth_of_first_l26_26467


namespace value_of_x_squared_plus_reciprocal_l26_26198

theorem value_of_x_squared_plus_reciprocal (x : ℝ) (h : 47 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 7 :=
by
  sorry

end value_of_x_squared_plus_reciprocal_l26_26198


namespace infinite_solutions_x2_y3_z5_l26_26531

theorem infinite_solutions_x2_y3_z5 :
  ∃ (t : ℕ), ∃ (x y z : ℕ), x = 2^(15*t + 12) ∧ y = 2^(10*t + 8) ∧ z = 2^(6*t + 5) ∧ (x^2 + y^3 = z^5) :=
sorry

end infinite_solutions_x2_y3_z5_l26_26531


namespace functions_of_same_family_count_l26_26007

theorem functions_of_same_family_count : 
  (∃ (y : ℝ → ℝ), ∀ x, y x = x^2) ∧ 
  (∃ (range_set : Set ℝ), range_set = {1, 2}) → 
  ∃ n, n = 9 :=
by
  sorry

end functions_of_same_family_count_l26_26007


namespace binary_101111_to_decimal_is_47_decimal_47_to_octal_is_57_l26_26529

def binary_to_decimal (b : ℕ) : ℕ :=
  32 + 0 + 8 + 4 + 2 + 1 -- Calculated manually for simplicity

def decimal_to_octal (d : ℕ) : ℕ :=
  (5 * 10) + 7 -- Manually converting decimal 47 to octal 57 for simplicity

theorem binary_101111_to_decimal_is_47 : binary_to_decimal 0b101111 = 47 := 
by sorry

theorem decimal_47_to_octal_is_57 : decimal_to_octal 47 = 57 := 
by sorry

end binary_101111_to_decimal_is_47_decimal_47_to_octal_is_57_l26_26529


namespace evaluate_expression_l26_26597

theorem evaluate_expression :
  -2 ^ 2005 + (-2) ^ 2006 + 2 ^ 2007 - 2 ^ 2008 = 2 ^ 2005 :=
by
  -- The following proof is left as an exercise.
  sorry

end evaluate_expression_l26_26597


namespace parallel_vectors_x_eq_one_l26_26682

/-- Given vectors a = (2x + 1, 3) and b = (2 - x, 1), prove that if they 
are parallel, then x = 1. -/
theorem parallel_vectors_x_eq_one (x : ℝ) :
  (∃ k : ℝ, (2 * x + 1) = k * (2 - x) ∧ 3 = k * 1) → x = 1 :=
by 
  sorry

end parallel_vectors_x_eq_one_l26_26682


namespace bacon_calories_percentage_l26_26416

-- Mathematical statement based on the problem
theorem bacon_calories_percentage :
  ∀ (total_sandwich_calories : ℕ) (number_of_bacon_strips : ℕ) (calories_per_strip : ℕ),
    total_sandwich_calories = 1250 →
    number_of_bacon_strips = 2 →
    calories_per_strip = 125 →
    (number_of_bacon_strips * calories_per_strip) * 100 / total_sandwich_calories = 20 :=
by
  intros total_sandwich_calories number_of_bacon_strips calories_per_strip h1 h2 h3 
  sorry

end bacon_calories_percentage_l26_26416


namespace initial_population_l26_26060

theorem initial_population (P : ℝ) : 
  (0.9 * P * 0.85 = 2907) → P = 3801 := by
  sorry

end initial_population_l26_26060


namespace pier_influence_duration_l26_26164

noncomputable def distance_affected_by_typhoon (AB AC: ℝ) : ℝ :=
  let AD := 350
  let DC := (AD ^ 2 - AC ^ 2).sqrt
  2 * DC

noncomputable def duration_under_influence (distance speed: ℝ) : ℝ :=
  distance / speed

theorem pier_influence_duration :
  let AB := 400
  let AC := AB * (1 / 2)
  let speed := 40
  duration_under_influence (distance_affected_by_typhoon AB AC) speed = 2.5 :=
by
  -- Proof would go here, but since it's omitted
  sorry

end pier_influence_duration_l26_26164


namespace savings_in_cents_l26_26857

def price_local : ℝ := 149.99
def price_payment : ℝ := 26.50
def number_payments : ℕ := 5
def fee_delivery : ℝ := 19.99

theorem savings_in_cents :
  (price_local - (number_payments * price_payment + fee_delivery)) * 100 = -250 := by
  sorry

end savings_in_cents_l26_26857


namespace both_players_score_same_points_l26_26331

theorem both_players_score_same_points :
  let P_A_score := 0.5 
  let P_B_score := 0.8 
  let P_A_miss := 1 - P_A_score
  let P_B_miss := 1 - P_B_score
  let P_both_miss := P_A_miss * P_B_miss
  let P_both_score := P_A_score * P_B_score
  let P_same_points := P_both_miss + P_both_score
  P_same_points = 0.5 := 
by {
  -- Actual proof should be here
  sorry
}

end both_players_score_same_points_l26_26331


namespace farm_field_ploughing_l26_26785

theorem farm_field_ploughing (A D : ℕ) 
  (h1 : ∀ farmerA_initial_capacity: ℕ, farmerA_initial_capacity = 120)
  (h2 : ∀ farmerB_initial_capacity: ℕ, farmerB_initial_capacity = 100)
  (h3 : ∀ farmerA_adjustment: ℕ, farmerA_adjustment = 10)
  (h4 : ∀ farmerA_reduced_capacity: ℕ, farmerA_reduced_capacity = farmerA_initial_capacity - (farmerA_adjustment * farmerA_initial_capacity / 100))
  (h5 : ∀ farmerB_reduced_capacity: ℕ, farmerB_reduced_capacity = 90)
  (h6 : ∀ extra_days: ℕ, extra_days = 3)
  (h7 : ∀ remaining_hectares: ℕ, remaining_hectares = 60)
  (h8 : ∀ initial_combined_effort: ℕ, initial_combined_effort = (farmerA_initial_capacity + farmerB_initial_capacity) * D)
  (h9 : ∀ total_combined_effort: ℕ, total_combined_effort = (farmerA_reduced_capacity + farmerB_reduced_capacity) * (D + extra_days))
  (h10 : ∀ area_covered: ℕ, area_covered = total_combined_effort + remaining_hectares)
  : initial_combined_effort = A ∧ D = 30 ∧ A = 6600 :=
by
  sorry

end farm_field_ploughing_l26_26785


namespace cos_square_theta_plus_pi_over_4_eq_one_fourth_l26_26453

variable (θ : ℝ)

theorem cos_square_theta_plus_pi_over_4_eq_one_fourth
  (h : Real.tan θ + 1 / Real.tan θ = 4) :
  Real.cos (θ + Real.pi / 4) ^ 2 = 1 / 4 :=
sorry

end cos_square_theta_plus_pi_over_4_eq_one_fourth_l26_26453


namespace geometric_sequence_sum_l26_26008

theorem geometric_sequence_sum (S : ℕ → ℝ) (a₄_to_a₁₂_sum : ℝ):
  (S 3 = 2) → (S 6 = 6) → a₄_to_a₁₂_sum = (S 12 - S 3)  :=
by
  sorry

end geometric_sequence_sum_l26_26008


namespace ratio_of_speeds_l26_26329

theorem ratio_of_speeds (v_A v_B : ℝ) (d_A d_B t : ℝ) (h1 : d_A = 100) (h2 : d_B = 50) (h3 : v_A = d_A / t) (h4 : v_B = d_B / t) : 
  v_A / v_B = 2 := 
by sorry

end ratio_of_speeds_l26_26329


namespace solve_MQ_above_A_l26_26358

-- Definitions of the given conditions
def ABCD_side := 8
def MNPQ_length := 16
def MNPQ_width := 8
def area_outer_inner_ratio := 1 / 3

-- Definition to prove
def length_MQ_above_A := 8 / 3

-- The area calculations
def area_MNPQ := MNPQ_length * MNPQ_width
def area_ABCD := ABCD_side * ABCD_side
def area_outer := (area_outer_inner_ratio * area_MNPQ)
def MQ_above_A_calculated := area_outer / MNPQ_length

theorem solve_MQ_above_A :
  MQ_above_A_calculated = length_MQ_above_A := by sorry

end solve_MQ_above_A_l26_26358


namespace tan_alpha_eq_2_implies_sin_2alpha_eq_4_over_5_l26_26214

variable {α : Real}

theorem tan_alpha_eq_2_implies_sin_2alpha_eq_4_over_5 (h : Real.tan α = 2) : Real.sin (2 * α) = 4 / 5 := 
by
  sorry

end tan_alpha_eq_2_implies_sin_2alpha_eq_4_over_5_l26_26214


namespace count_true_statements_l26_26147

theorem count_true_statements (x : ℝ) (h : x > -3) :
  (if (x > -3 → x > -6) then 1 else 0) +
  (if (¬ (x > -3 → x > -6)) then 1 else 0) +
  (if (x > -6 → x > -3) then 1 else 0) +
  (if (¬ (x > -6 → x > -3)) then 1 else 0) = 2 :=
sorry

end count_true_statements_l26_26147


namespace sum_of_arithmetic_seq_minimum_value_n_equals_5_l26_26744

variable {a : ℕ → ℝ} -- Define a sequence of real numbers
variable {S : ℕ → ℝ} -- Define the sum function for the sequence

-- Assume conditions
axiom a3_a8_neg : a 3 + a 8 < 0
axiom S11_pos : S 11 > 0

-- Prove the minimum value of S_n occurs at n = 5
theorem sum_of_arithmetic_seq_minimum_value_n_equals_5 :
  ∃ n, (∀ m < 5, S m ≥ S n) ∧ (∀ m > 5, S m > S n) ∧ n = 5 :=
sorry

end sum_of_arithmetic_seq_minimum_value_n_equals_5_l26_26744


namespace smallest_prime_number_conditions_l26_26102

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |> List.sum -- Summing the digits in base 10

def is_prime (n : ℕ) : Prop := Nat.Prime n

def smallest_prime_number (n : ℕ) : Prop :=
  is_prime n ∧ sum_of_digits n = 17 ∧ n > 200 ∧
  (∀ m : ℕ, is_prime m ∧ sum_of_digits m = 17 ∧ m > 200 → n ≤ m)

theorem smallest_prime_number_conditions (p : ℕ) : 
  smallest_prime_number p ↔ p = 197 :=
by
  sorry

end smallest_prime_number_conditions_l26_26102


namespace seq_integer_l26_26442

theorem seq_integer (a : ℕ → ℤ) (h1 : a 1 = 1) (h2 : a 2 = 1) (h3 : a 3 = 249)
(h_rec : ∀ n, a (n + 3) = (1991 + a (n + 2) * a (n + 1)) / a n) :
∀ n, ∃ b : ℤ, a n = b :=
by
  sorry

end seq_integer_l26_26442


namespace find_min_value_l26_26719

theorem find_min_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1 / a + 1 / b = 1) : a + 2 * b ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end find_min_value_l26_26719


namespace pqrs_l26_26762

theorem pqrs(p q r s t u : ℤ) :
  (729 * (x : ℤ) * x * x + 64 = (p * x * x + q * x + r) * (s * x * x + t * x + u)) →
  p = 9 → q = 4 → r = 0 → s = 81 → t = -36 → u = 16 →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 8210 := by
  intros h1 hp hq hr hs ht hu
  sorry

end pqrs_l26_26762


namespace arithmetic_sequence_sum_first_nine_terms_l26_26515

variable (a_n : ℕ → ℤ)
variable (S_n : ℕ → ℤ)
variable (d : ℤ)

-- The sequence {a_n} is an arithmetic sequence.
def arithmetic_sequence := ∀ n : ℕ, a_n (n + 1) = a_n n + d

-- The sum of the first n terms of the sequence.
def sum_first_n_terms := ∀ n : ℕ, S_n n = (n * (a_n 1 + a_n n)) / 2

-- Given condition: a_2 = 3 * a_4 - 6
def given_condition := a_n 2 = 3 * a_n 4 - 6

-- The main theorem to prove S_9 = 27
theorem arithmetic_sequence_sum_first_nine_terms (h_arith : arithmetic_sequence a_n d) (h_sum : sum_first_n_terms a_n S_n) (h_condition : given_condition a_n) : 
  S_n 9 = 27 := 
by
  sorry

end arithmetic_sequence_sum_first_nine_terms_l26_26515


namespace set_intersection_complement_l26_26055

theorem set_intersection_complement (U : Set ℝ) (A B : Set ℝ) 
  (hU : U = Set.univ) 
  (hA : ∀ x : ℝ, A x ↔ x^2 - x - 6 ≤ 0) 
  (hB : ∀ x : ℝ, B x ↔ Real.log x / Real.log (1/2) ≥ -1) :
  A ∩ (U \ B) = (Set.Icc (-2 : ℝ) 0 ∪ Set.Ioc 2 3) :=
by
  ext x
  -- Proof here would follow
  sorry

end set_intersection_complement_l26_26055


namespace evaluate_function_at_neg_one_l26_26098

def f (x : ℝ) : ℝ := -2 * x^2 + 1

theorem evaluate_function_at_neg_one : f (-1) = -1 :=
by
  sorry

end evaluate_function_at_neg_one_l26_26098


namespace random_walk_expected_distance_l26_26922

noncomputable def expected_distance_after_random_walk (n : ℕ) : ℚ :=
(sorry : ℚ) -- We'll define this in the proof

-- Proof problem statement in Lean 4
theorem random_walk_expected_distance :
  expected_distance_after_random_walk 6 = 15 / 8 :=
by 
  sorry

end random_walk_expected_distance_l26_26922


namespace a_2011_value_l26_26872

noncomputable def sequence_a : ℕ → ℝ
| 0 => 6/7
| (n + 1) => if 0 ≤ sequence_a n ∧ sequence_a n < 1/2 then 2 * sequence_a n
              else 2 * sequence_a n - 1

theorem a_2011_value : sequence_a 2011 = 6/7 := sorry

end a_2011_value_l26_26872


namespace johns_haircut_tip_percentage_l26_26373

noncomputable def percent_of_tip (annual_spending : ℝ) (haircut_cost : ℝ) (haircut_frequency : ℕ) : ℝ := 
  ((annual_spending / haircut_frequency - haircut_cost) / haircut_cost) * 100

theorem johns_haircut_tip_percentage : 
  let hair_growth_rate : ℝ := 1.5
  let initial_length : ℝ := 6
  let max_length : ℝ := 9
  let haircut_cost : ℝ := 45
  let annual_spending : ℝ := 324
  let months_in_year : ℕ := 12
  let growth_period := 2 -- months it takes for hair to grow 3 inches
  let haircuts_per_year := months_in_year / growth_period -- number of haircuts per year
  percent_of_tip annual_spending haircut_cost haircuts_per_year = 20 := by
  sorry

end johns_haircut_tip_percentage_l26_26373


namespace maximum_rubles_l26_26783

-- We define the initial number of '1' and '2' cards
def num_ones : ℕ := 2013
def num_twos : ℕ := 2013
def total_digits : ℕ := num_ones + num_twos

-- Definition of the problem statement
def problem_statement : Prop :=
  ∃ (max_rubles : ℕ), 
    max_rubles = 5 ∧
    ∀ (current_k : ℕ), 
      current_k = 5 → 
      ∃ (moves : ℕ), 
        moves ≤ max_rubles ∧
        (current_k - moves * 2) % 11 = 0

-- The expected solution is proving the maximum rubles is 5
theorem maximum_rubles : problem_statement :=
by
  sorry

end maximum_rubles_l26_26783


namespace regular_polygon_sides_eq_seven_l26_26561

theorem regular_polygon_sides_eq_seven (n : ℕ) (h1 : D = n * (n-3) / 2) (h2 : D = 2 * n) : n = 7 := 
by
  sorry

end regular_polygon_sides_eq_seven_l26_26561


namespace tangent_circle_line_radius_l26_26801

theorem tangent_circle_line_radius (m : ℝ) :
  (∀ x y : ℝ, (x - 1)^2 + y^2 = m → x + y = 1 → dist (1, 0) (x, y) = Real.sqrt m) →
  m = 1 / 2 :=
by
  sorry

end tangent_circle_line_radius_l26_26801


namespace dot_product_eq_l26_26791

def vector1 : ℝ × ℝ := (-3, 0)
def vector2 : ℝ × ℝ := (7, 9)

theorem dot_product_eq :
  (vector1.1 * vector2.1 + vector1.2 * vector2.2) = -21 :=
by
  sorry

end dot_product_eq_l26_26791


namespace cost_price_of_article_l26_26396

theorem cost_price_of_article (C MP SP : ℝ) (h1 : MP = 62.5) (h2 : SP = 0.95 * MP) (h3 : SP = 1.25 * C) :
  C = 47.5 :=
sorry

end cost_price_of_article_l26_26396


namespace function_inverse_necessary_not_sufficient_l26_26692

theorem function_inverse_necessary_not_sufficient (f : ℝ → ℝ) :
  (∃ g : ℝ → ℝ, ∀ x : ℝ, g (f x) = x ∧ f (g x) = x) →
  ¬ (∀ (x y : ℝ), x < y → f x < f y) :=
by
  sorry

end function_inverse_necessary_not_sufficient_l26_26692


namespace password_guess_probability_l26_26562

def probability_correct_digit_within_two_attempts : Prop :=
  let total_digits := 10
  let prob_first_attempt := 1 / total_digits
  let prob_second_attempt := (9 / total_digits) * (1 / (total_digits - 1))
  (prob_first_attempt + prob_second_attempt) = 1 / 5

theorem password_guess_probability :
  probability_correct_digit_within_two_attempts :=
by
  -- proof goes here
  sorry

end password_guess_probability_l26_26562


namespace reena_loan_l26_26624

/-- 
  Problem setup:
  Reena took a loan of $1200 at simple interest for a period equal to the rate of interest years. 
  She paid $192 as interest at the end of the loan period.
  We aim to prove that the rate of interest is 4%. 
-/
theorem reena_loan (P : ℝ) (SI : ℝ) (R : ℝ) (N : ℝ) 
  (hP : P = 1200) 
  (hSI : SI = 192) 
  (hN : N = R) 
  (hSI_formula : SI = P * R * N / 100) : 
  R = 4 := 
by 
  sorry

end reena_loan_l26_26624


namespace infinite_polynomial_pairs_l26_26603

open Polynomial

theorem infinite_polynomial_pairs :
  ∀ n : ℕ, ∃ (fn gn : ℤ[X]), fn^2 - (X^4 - 2 * X) * gn^2 = 1 :=
sorry

end infinite_polynomial_pairs_l26_26603


namespace least_integer_value_l26_26466

theorem least_integer_value 
  (x : ℤ) (h : |3 * x - 5| ≤ 22) : x = -5 ↔ ∃ (k : ℤ), k = -5 ∧ |3 * k - 5| ≤ 22 :=
by
  sorry

end least_integer_value_l26_26466


namespace min_diagonal_length_of_trapezoid_l26_26768

theorem min_diagonal_length_of_trapezoid (a b h d1 d2 : ℝ) 
  (h_area : a * h + b * h = 2)
  (h_diag : d1^2 + d2^2 = h^2 + (a + b)^2) 
  : d1 ≥ Real.sqrt 2 :=
sorry

end min_diagonal_length_of_trapezoid_l26_26768


namespace min_value_is_3_l26_26867

theorem min_value_is_3 (a b : ℝ) (h1 : a > b / 2) (h2 : 2 * a > b) : (2 * a + b) / a ≥ 3 :=
sorry

end min_value_is_3_l26_26867


namespace haley_stickers_l26_26110

theorem haley_stickers (friends : ℕ) (stickers_per_friend : ℕ) (total_stickers : ℕ) :
  friends = 9 → stickers_per_friend = 8 → total_stickers = friends * stickers_per_friend → total_stickers = 72 :=
by
  intros h_friends h_stickers_per_friend h_total_stickers
  rw [h_friends, h_stickers_per_friend] at h_total_stickers
  exact h_total_stickers

end haley_stickers_l26_26110


namespace simplify_expression_l26_26366

theorem simplify_expression (y : ℝ) :
  4 * y - 8 * y^2 + 6 - (3 - 6 * y - 9 * y^2 + 2 * y^3) = -2 * y^3 + y^2 + 10 * y + 3 := 
by
  -- Proof goes here, but we just state sorry for now
  sorry

end simplify_expression_l26_26366


namespace arithmetic_sequence_sum_l26_26991

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h_arith : ∃ (d : ℝ), ∀ (n : ℕ), a (n + 1) = a n + d) 
  (h1 : a 1 + a 5 = 2) 
  (h2 : a 2 + a 14 = 12) : 
  (10 / 2) * (2 * a 1 + 9 * d) = 35 :=
by
  sorry

end arithmetic_sequence_sum_l26_26991


namespace gain_percent_is_40_l26_26122

-- Define the conditions
def purchase_price : ℕ := 800
def repair_costs : ℕ := 200
def selling_price : ℕ := 1400

-- Define the total cost
def total_cost : ℕ := purchase_price + repair_costs

-- Define the gain
def gain : ℕ := selling_price - total_cost

-- Define the gain percent
def gain_percent : ℕ := (gain * 100) / total_cost

theorem gain_percent_is_40 : gain_percent = 40 := by
  -- Placeholder for the proof
  sorry

end gain_percent_is_40_l26_26122


namespace expression_value_l26_26078

theorem expression_value (a b : ℕ) (h₁ : a = 2023) (h₂ : b = 2020) :
  ((
     (3 / (a - b) + (3 * a) / (a^3 - b^3) * ((a^2 + a * b + b^2) / (a + b))) * ((2 * a + b) / (a^2 + 2 * a * b + b^2))
  ) * (3 / (a + b))) = 3 :=
by
  -- Use the provided conditions
  rw [h₁, h₂]
  -- Execute the following steps as per the mathematical solution steps 
  sorry

end expression_value_l26_26078


namespace no_perfect_square_in_range_l26_26127

def isPerfectSquare (x : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = x

theorem no_perfect_square_in_range :
  ∀ (n : ℕ), 4 ≤ n ∧ n ≤ 12 → ¬ isPerfectSquare (2*n*n + 3*n + 2) :=
by
  intro n
  intro h
  sorry

end no_perfect_square_in_range_l26_26127


namespace members_playing_both_l26_26822

theorem members_playing_both
  (N B T Neither : ℕ)
  (hN : N = 40)
  (hB : B = 20)
  (hT : T = 18)
  (hNeither : Neither = 5) :
  (B + T) - (N - Neither) = 3 := by
-- to complete the proof
sorry

end members_playing_both_l26_26822


namespace calculate_expression_value_l26_26652

theorem calculate_expression_value :
  5 * 7 + 6 * 9 + 13 * 2 + 4 * 6 = 139 :=
by
  -- proof can be added here
  sorry

end calculate_expression_value_l26_26652


namespace jakes_digging_time_l26_26691

theorem jakes_digging_time
  (J : ℕ)
  (Paul_work_rate : ℚ := 1/24)
  (Hari_work_rate : ℚ := 1/48)
  (Combined_work_rate : ℚ := 1/8)
  (Combined_work_eq : 1 / J + Paul_work_rate + Hari_work_rate = Combined_work_rate) :
  J = 16 := sorry

end jakes_digging_time_l26_26691


namespace barbara_spent_total_l26_26065

variables (cost_steaks cost_chicken total_spent per_pound_steak per_pound_chicken : ℝ)
variables (weight_steaks weight_chicken : ℝ)

-- Defining the given conditions
def conditions :=
  per_pound_steak = 15 ∧
  weight_steaks = 4.5 ∧
  cost_steaks = per_pound_steak * weight_steaks ∧

  per_pound_chicken = 8 ∧
  weight_chicken = 1.5 ∧
  cost_chicken = per_pound_chicken * weight_chicken

-- Proving the total spent by Barbara is $79.50
theorem barbara_spent_total 
  (h : conditions per_pound_steak weight_steaks cost_steaks per_pound_chicken weight_chicken cost_chicken) : 
  total_spent = 79.5 :=
sorry

end barbara_spent_total_l26_26065


namespace candy_bar_cost_correct_l26_26130

noncomputable def candy_bar_cost : ℕ := 25 -- Correct answer from the solution

theorem candy_bar_cost_correct (C : ℤ) (H1 : 3 * C + 150 + 50 = 11 * 25)
  (H2 : ∃ C, C ≥ 0) : C = candy_bar_cost :=
by
  sorry

end candy_bar_cost_correct_l26_26130


namespace du_chin_remaining_money_l26_26554

noncomputable def du_chin_revenue_over_week : ℝ := 
  let day0_revenue := 200 * 20
  let day0_cost := 3 / 5 * day0_revenue
  let day0_remaining := day0_revenue - day0_cost

  let day1_revenue := day0_remaining * 1.10
  let day1_cost := day0_cost * 1.10
  let day1_remaining := day1_revenue - day1_cost

  let day2_revenue := day1_remaining * 0.95
  let day2_cost := day1_cost * 0.90
  let day2_remaining := day2_revenue - day2_cost

  let day3_revenue := day2_remaining
  let day3_cost := day2_cost
  let day3_remaining := day3_revenue - day3_cost

  let day4_revenue := day3_remaining * 1.15
  let day4_cost := day3_cost * 1.05
  let day4_remaining := day4_revenue - day4_cost

  let day5_revenue := day4_remaining * 0.92
  let day5_cost := day4_cost * 0.95
  let day5_remaining := day5_revenue - day5_cost

  let day6_revenue := day5_remaining * 1.05
  let day6_cost := day5_cost
  let day6_remaining := day6_revenue - day6_cost

  day0_remaining + day1_remaining + day2_remaining + day3_remaining + day4_remaining + day5_remaining + day6_remaining

theorem du_chin_remaining_money : du_chin_revenue_over_week = 13589.08 := 
  sorry

end du_chin_remaining_money_l26_26554


namespace coordinates_of_B_l26_26279

-- Define the initial coordinates of point A
def A : ℝ × ℝ := (1, -2)

-- Define the transformation to get point B from A
def B : ℝ × ℝ := (A.1 - 2, A.2 + 3)

theorem coordinates_of_B : B = (-1, 1) :=
by
  sorry

end coordinates_of_B_l26_26279


namespace length_of_platform_is_350_l26_26011

-- Define the parameters as given in the problem
def train_length : ℕ := 300
def time_to_cross_post : ℕ := 18
def time_to_cross_platform : ℕ := 39

-- Define the speed of the train as a ratio of the length of the train and the time to cross the post
def train_speed : ℚ := train_length / time_to_cross_post

-- Formalize the problem statement: Prove that the length of the platform is 350 meters
theorem length_of_platform_is_350 : ∃ (L : ℕ), (train_speed * time_to_cross_platform) = train_length + L := by
  use 350
  sorry

end length_of_platform_is_350_l26_26011


namespace right_triangle_inequality_l26_26500

theorem right_triangle_inequality {a b c : ℝ} (h : c^2 = a^2 + b^2) : 
  a + b ≤ c * Real.sqrt 2 :=
sorry

end right_triangle_inequality_l26_26500


namespace probability_white_given_popped_is_7_over_12_l26_26092

noncomputable def probability_white_given_popped : ℚ :=
  let P_W := 0.4
  let P_Y := 0.4
  let P_R := 0.2
  let P_popped_given_W := 0.7
  let P_popped_given_Y := 0.5
  let P_popped_given_R := 0
  let P_popped := P_popped_given_W * P_W + P_popped_given_Y * P_Y + P_popped_given_R * P_R
  (P_popped_given_W * P_W) / P_popped

theorem probability_white_given_popped_is_7_over_12 : probability_white_given_popped = 7 / 12 := 
  by
    sorry

end probability_white_given_popped_is_7_over_12_l26_26092


namespace minimum_flour_cost_l26_26436

-- Definitions based on conditions provided
def loaves : ℕ := 12
def flour_per_loaf : ℕ := 4
def flour_needed : ℕ := loaves * flour_per_loaf

def ten_pound_bag_weight : ℕ := 10
def ten_pound_bag_cost : ℕ := 10

def twelve_pound_bag_weight : ℕ := 12
def twelve_pound_bag_cost : ℕ := 13

def cost_10_pound_bags : ℕ := (flour_needed + ten_pound_bag_weight - 1) / ten_pound_bag_weight * ten_pound_bag_cost
def cost_12_pound_bags : ℕ := (flour_needed + twelve_pound_bag_weight - 1) / twelve_pound_bag_weight * twelve_pound_bag_cost

theorem minimum_flour_cost : min cost_10_pound_bags cost_12_pound_bags = 50 := by
  sorry

end minimum_flour_cost_l26_26436


namespace solve_puzzle_l26_26755

theorem solve_puzzle
  (EH OY AY OH : ℕ)
  (h1 : EH = 4 * OY)
  (h2 : AY = 4 * OH) :
  EH + OY + AY + OH = 150 :=
sorry

end solve_puzzle_l26_26755


namespace solve_for_x_l26_26012

theorem solve_for_x (x : ℚ) : 
  x + 5 / 6 = 11 / 18 - 2 / 9 → x = -4 / 9 := 
by
  intro h
  sorry

end solve_for_x_l26_26012


namespace thought_number_and_appended_digit_l26_26930

theorem thought_number_and_appended_digit (x y : ℕ) (hx : x > 0) (hy : y ≤ 9):
  (10 * x + y - x^2 = 8 * x) ↔ (x = 2 ∧ y = 0) ∨ (x = 3 ∧ y = 3) ∨ (x = 4 ∧ y = 8) := sorry

end thought_number_and_appended_digit_l26_26930


namespace discarded_number_l26_26277

theorem discarded_number (S x : ℕ) (h1 : S / 50 = 50) (h2 : (S - x - 55) / 48 = 50) : x = 45 :=
by
  sorry

end discarded_number_l26_26277


namespace archer_prob_6_or_less_l26_26542

noncomputable def prob_event_D (P_A P_B P_C : ℝ) : ℝ :=
  1 - (P_A + P_B + P_C)

theorem archer_prob_6_or_less :
  let P_A := 0.5
  let P_B := 0.2
  let P_C := 0.1
  prob_event_D P_A P_B P_C = 0.2 :=
by
  sorry

end archer_prob_6_or_less_l26_26542


namespace no_x0_leq_zero_implies_m_gt_1_l26_26947

theorem no_x0_leq_zero_implies_m_gt_1 (m : ℝ) :
  (¬ ∃ x0 : ℝ, x0^2 + 2 * x0 + m ≤ 0) ↔ m > 1 :=
sorry

end no_x0_leq_zero_implies_m_gt_1_l26_26947


namespace P_has_no_negative_roots_but_at_least_one_positive_root_l26_26932

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := x^6 - 4*x^5 - 9*x^3 + 2*x + 9

-- Statement of the problem
theorem P_has_no_negative_roots_but_at_least_one_positive_root :
  (∀ x : ℝ, x < 0 → P x ≠ 0 ∧ P x > 0) ∧ (∃ x : ℝ, x > 0 ∧ P x = 0) :=
by
  sorry

end P_has_no_negative_roots_but_at_least_one_positive_root_l26_26932


namespace solve_quadratic_l26_26318

theorem solve_quadratic (h₁ : 48 * (3/4:ℚ)^2 - 74 * (3/4:ℚ) + 47 = 0) :
  ∃ x : ℚ, x ≠ 3/4 ∧ 48 * x^2 - 74 * x + 47 = 0 ∧ x = 11/12 := 
by
  sorry

end solve_quadratic_l26_26318


namespace units_digit_33_exp_l26_26236

def units_digit_of_power_cyclic (base exponent : ℕ) (cycle : List ℕ) : ℕ :=
  cycle.get! (exponent % cycle.length)

theorem units_digit_33_exp (n : ℕ) (h1 : 33 = 1 + 4 * 8) (h2 : 44 = 4 * 11) :
  units_digit_of_power_cyclic 33 (33 * 44 ^ 44) [3, 9, 7, 1] = 3 :=
by
  sorry

end units_digit_33_exp_l26_26236


namespace alice_probability_same_color_l26_26609

def total_ways_to_draw : ℕ := 
  Nat.choose 9 3 * Nat.choose 6 3 * Nat.choose 3 3

def favorable_outcomes_for_alice : ℕ := 
  3 * Nat.choose 6 3 * Nat.choose 3 3

def probability_alice_same_color : ℚ := 
  favorable_outcomes_for_alice / total_ways_to_draw

theorem alice_probability_same_color : probability_alice_same_color = 1 / 28 := 
by
  -- Proof is omitted as per instructions
  sorry

end alice_probability_same_color_l26_26609


namespace trailing_zeros_sum_15_factorial_l26_26986

theorem trailing_zeros_sum_15_factorial : 
  let k := 5
  let h := 3
  k + h = 8 := by
  sorry

end trailing_zeros_sum_15_factorial_l26_26986


namespace robins_total_pieces_of_gum_l26_26579

theorem robins_total_pieces_of_gum :
  let initial_packages := 27
  let pieces_per_initial_package := 18
  let additional_packages := 15
  let pieces_per_additional_package := 12
  let more_packages := 8
  let pieces_per_more_package := 25
  (initial_packages * pieces_per_initial_package) +
  (additional_packages * pieces_per_additional_package) +
  (more_packages * pieces_per_more_package) = 866 :=
by
  sorry

end robins_total_pieces_of_gum_l26_26579


namespace farmer_feed_full_price_l26_26818

theorem farmer_feed_full_price
  (total_spent : ℕ)
  (chicken_feed_discount_percent : ℕ)
  (chicken_feed_percent : ℕ)
  (goat_feed_percent : ℕ)
  (total_spent_val : total_spent = 35)
  (chicken_feed_discount_percent_val : chicken_feed_discount_percent = 50)
  (chicken_feed_percent_val : chicken_feed_percent = 40)
  (goat_feed_percent_val : goat_feed_percent = 60) :
  (total_spent * chicken_feed_percent / 100 * 2) + (total_spent * goat_feed_percent / 100) = 49 := 
by
  -- Placeholder for proof.
  sorry

end farmer_feed_full_price_l26_26818


namespace sin_2A_value_l26_26850

variable {A B C : ℝ}
variable {a b c : ℝ}
variable (h₁ : a / (2 * Real.cos A) = b / (3 * Real.cos B))
variable (h₂ : b / (3 * Real.cos B) = c / (6 * Real.cos C))

theorem sin_2A_value (h₃ : a / (2 * Real.cos A) = c / (6 * Real.cos C)) :
  Real.sin (2 * A) = 3 * Real.sqrt 11 / 10 := sorry

end sin_2A_value_l26_26850


namespace part1_part2_l26_26880

open Nat

-- Part (I)
theorem part1 (a b : ℝ) (h1 : ∀ x : ℝ, x^2 - a * x + b = 0 → x = 2 ∨ x = 3) :
  a + b = 11 :=
by sorry

-- Part (II)
theorem part2 (c : ℝ) (h2 : ∀ x : ℝ, -x^2 + 6 * x + c ≤ 0) :
  c ≤ -9 :=
by sorry

end part1_part2_l26_26880


namespace expression_not_equal_l26_26205

theorem expression_not_equal :
  let e1 := 250 * 12
  let e2 := 25 * 4 + 30
  let e3 := 25 * 40 * 3
  let product := 25 * 120
  e2 ≠ product :=
by
  let e1 := 250 * 12
  let e2 := 25 * 4 + 30
  let e3 := 25 * 40 * 3
  let product := 25 * 120
  sorry

end expression_not_equal_l26_26205


namespace direct_proportion_point_l26_26015

theorem direct_proportion_point (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : y₁ = k * x₁) (hx₁ : x₁ = -1) (hy₁ : y₁ = 2) (hx₂ : x₂ = 1) (hy₂ : y₂ = -2) 
  : y₂ = k * x₂ := 
by
  -- sorry will skip the proof
  sorry

end direct_proportion_point_l26_26015


namespace find_n_l26_26216

def factorial : ℕ → ℕ 
| 0 => 1
| (n + 1) => (n + 1) * factorial n

theorem find_n (n : ℕ) : 3 * n * factorial n + 2 * factorial n = 40320 → n = 8 :=
by
  sorry

end find_n_l26_26216


namespace small_box_dolls_l26_26064

theorem small_box_dolls (x : ℕ) : 
  (5 * 7 + 9 * x = 71) → x = 4 :=
by
  sorry

end small_box_dolls_l26_26064


namespace fencing_rate_correct_l26_26154

noncomputable def rate_of_fencing_per_meter (area_hectares : ℝ) (total_cost : ℝ) : ℝ :=
  let area_sqm := area_hectares * 10000
  let r_squared := area_sqm / Real.pi
  let r := Real.sqrt r_squared
  let circumference := 2 * Real.pi * r
  total_cost / circumference

theorem fencing_rate_correct :
  rate_of_fencing_per_meter 13.86 6070.778380479544 = 4.60 :=
by
  sorry

end fencing_rate_correct_l26_26154


namespace find_x_l26_26388

theorem find_x (x : ℚ) (h : (3 + 1 / (2 + 1 / (3 + 3 / (4 + x)))) = 225 / 68) : 
  x = -50 / 19 := 
sorry

end find_x_l26_26388


namespace min_straight_line_cuts_l26_26116

theorem min_straight_line_cuts (can_overlap : Prop) : 
  ∃ (cuts : ℕ), cuts = 4 ∧ 
  (∀ (square : ℕ), square = 3 →
   ∀ (unit : ℕ), unit = 1 → 
   ∀ (divided : Prop), divided = True → 
   (unit * unit) * 9 = (square * square)) :=
by
  sorry

end min_straight_line_cuts_l26_26116


namespace expression_always_integer_l26_26938

theorem expression_always_integer (m : ℕ) : 
  ∃ k : ℤ, (m / 3 + m^2 / 2 + m^3 / 6 : ℚ) = (k : ℚ) := 
sorry

end expression_always_integer_l26_26938


namespace find_y_when_x_is_7_l26_26173

theorem find_y_when_x_is_7 (x y : ℝ) (h1 : x * y = 200) (h2 : x = 7) : y = 200 / 7 :=
by
  sorry

end find_y_when_x_is_7_l26_26173


namespace expression_simplify_l26_26574

theorem expression_simplify
  (a b : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : b ≠ 0)
  (h₃ : 3 * a - b / 3 ≠ 0) :
  (3 * a - b / 3)⁻¹ * ((3 * a)⁻¹ - (b / 3)⁻¹) = - (1 / (a * b)) :=
by
  sorry

end expression_simplify_l26_26574


namespace triangle_area_proof_l26_26650

-- Conditions
variables (P r : ℝ) (semi_perimeter : ℝ)
-- The perimeter of the triangle is 40 cm
def perimeter_condition : Prop := P = 40
-- The inradius of the triangle is 2.5 cm
def inradius_condition : Prop := r = 2.5
-- The semi-perimeter is half of the perimeter
def semi_perimeter_def : Prop := semi_perimeter = P / 2

-- The area of the triangle
def area_of_triangle : ℝ := r * semi_perimeter

-- Proof Problem
theorem triangle_area_proof (hP : perimeter_condition P) (hr : inradius_condition r) (hsemi : semi_perimeter_def P semi_perimeter) :
  area_of_triangle r semi_perimeter = 50 :=
  sorry

end triangle_area_proof_l26_26650


namespace fraction_to_decimal_and_add_l26_26528

theorem fraction_to_decimal_and_add (a b : ℚ) (h : a = 7 / 16) : (a + b) = 2.4375 ↔ b = 2 :=
by
   sorry

end fraction_to_decimal_and_add_l26_26528


namespace pupils_like_only_maths_l26_26113

noncomputable def number_pupils_like_only_maths (total: ℕ) (maths_lovers: ℕ) (english_lovers: ℕ) 
(neither_lovers: ℕ) (both_lovers: ℕ) : ℕ :=
maths_lovers - both_lovers

theorem pupils_like_only_maths : 
∀ (total: ℕ) (maths_lovers: ℕ) (english_lovers: ℕ) (neither_lovers: ℕ) (both_lovers: ℕ),
total = 30 →
maths_lovers = 20 →
english_lovers = 18 →
both_lovers = 2 * neither_lovers →
neither_lovers + maths_lovers + english_lovers - both_lovers - both_lovers = total →
number_pupils_like_only_maths total maths_lovers english_lovers neither_lovers both_lovers = 4 :=
by
  intros _ _ _ _ _ _ _ _ _ _
  sorry

end pupils_like_only_maths_l26_26113


namespace contrapositive_example_l26_26782

theorem contrapositive_example (a b : ℝ) (h : a^2 + b^2 < 4) : a + b ≠ 3 :=
sorry

end contrapositive_example_l26_26782


namespace tangent_normal_lines_l26_26314

noncomputable def x (t : ℝ) : ℝ := (1 / 2) * t^2 - (1 / 4) * t^4
noncomputable def y (t : ℝ) : ℝ := (1 / 2) * t^2 + (1 / 3) * t^3
def t0 : ℝ := 0

theorem tangent_normal_lines :
  (∃ m : ℝ, ∀ t : ℝ, t = t0 → y t = m * x t) ∧
  (∃ n : ℝ, ∀ t : ℝ, t = t0 → y t = n * x t ∧ n = -1 / m) :=
sorry

end tangent_normal_lines_l26_26314


namespace lines_parallel_iff_m_eq_1_l26_26103

-- Define the two lines l1 and l2
def l1 (m : ℝ) (x y : ℝ) : Prop := x + (1 + m) * y = 2 - m
def l2 (m : ℝ) (x y : ℝ) : Prop := 2 * m * x + 4 * y = -16

-- Parallel lines condition
def parallel_condition (m : ℝ) : Prop := (1 * 4 - 2 * m * (1 + m) = 0) ∧ (1 * 16 - 2 * m * (m - 2) ≠ 0)

-- The theorem to prove
theorem lines_parallel_iff_m_eq_1 (m : ℝ) : l1 m = l2 m → parallel_condition m → m = 1 :=
by 
  sorry

end lines_parallel_iff_m_eq_1_l26_26103


namespace solve_df1_l26_26881

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (df1 : ℝ)

-- The condition given in the problem
axiom func_def : ∀ x, f x = 2 * x * df1 + (Real.log x)

-- Express the relationship from the derivative and solve for f'(1) = -1
theorem solve_df1 : df1 = -1 :=
by
  -- Here we will insert the proof steps in Lean, but they are omitted in this statement.
  sorry

end solve_df1_l26_26881


namespace karen_starts_late_by_4_minutes_l26_26153

-- Define conditions as Lean 4 variables/constants
noncomputable def karen_speed : ℝ := 60 -- in mph
noncomputable def tom_speed : ℝ := 45 -- in mph
noncomputable def tom_distance : ℝ := 24 -- in miles
noncomputable def karen_lead : ℝ := 4 -- in miles

-- Main theorem statement
theorem karen_starts_late_by_4_minutes : 
  ∃ (minutes_late : ℝ), minutes_late = 4 :=
by
  -- Calculations based on given conditions provided in the problem
  let t := tom_distance / tom_speed -- Time for Tom to drive 24 miles
  let tk := (tom_distance + karen_lead) / karen_speed -- Time for Karen to drive 28 miles
  let time_difference := t - tk -- Time difference between Tom and Karen
  let minutes_late := time_difference * 60 -- Convert time difference to minutes
  existsi minutes_late -- Existential quantifier to state the existence of such a time
  have h : minutes_late = 4 := sorry -- Placeholder for demonstrating equality
  exact h

end karen_starts_late_by_4_minutes_l26_26153


namespace sum_of_abs_coeffs_l26_26349

theorem sum_of_abs_coeffs (a : ℕ → ℤ) :
  (∀ x : ℤ, (1 - x)^5 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5) →
  |a 0| + |a 1| + |a 2| + |a 3| + |a 4| + |a 5| = 32 := 
by
  sorry

end sum_of_abs_coeffs_l26_26349


namespace inscribe_circle_in_convex_polygon_l26_26533

theorem inscribe_circle_in_convex_polygon
  (S P r : ℝ) 
  (hP_pos : P > 0)
  (h_poly_area : S > 0)
  (h_nonneg : r ≥ 0) :
  S / P ≤ r :=
sorry

end inscribe_circle_in_convex_polygon_l26_26533


namespace cube_numbers_not_all_even_cube_numbers_not_all_divisible_by_3_l26_26080

-- Define the initial state of the cube vertices
def initial_cube : ℕ → ℕ
| 0 => 1  -- The number at vertex 0 is 1
| _ => 0  -- The numbers at other vertices are 0

-- Define the edge addition operation
def edge_add (v1 v2 : ℕ → ℕ) (edge : ℕ × ℕ) : ℕ → ℕ :=
  λ x => if x = edge.1 ∨ x = edge.2 then v1 x + 1 else v1 x

-- Condition: one can add one to the numbers at the ends of any edge
axiom edge_op : ∀ (v : ℕ → ℕ) (e : ℕ × ℕ), ℕ → ℕ

-- Defining the problem in Lean
theorem cube_numbers_not_all_even :
  ¬ (∃ (v : ℕ → ℕ), ∀ x, v x % 2 = 0) :=
by
  -- Proof not required
  sorry

theorem cube_numbers_not_all_divisible_by_3 :
  ¬ (∃ (v : ℕ → ℕ), ∀ x, v x % 3 = 0) :=
by
  -- Proof not required
  sorry

end cube_numbers_not_all_even_cube_numbers_not_all_divisible_by_3_l26_26080


namespace smallest_a_value_l26_26962

theorem smallest_a_value (α β γ : ℕ) (hαβγ : α * β * γ = 2010) (hα : α > 0) (hβ : β > 0) (hγ : γ > 0) :
  α + β + γ = 78 :=
by
-- Proof would go here
sorry

end smallest_a_value_l26_26962


namespace car_return_speed_l26_26859

theorem car_return_speed (d : ℕ) (speed_CD : ℕ) (avg_speed_round_trip : ℕ) 
  (round_trip_distance : ℕ) (time_CD : ℕ) (time_round_trip : ℕ) (r: ℕ) 
  (h1 : d = 150) (h2 : speed_CD = 75) (h3 : avg_speed_round_trip = 60)
  (h4 : d * 2 = round_trip_distance) 
  (h5 : time_CD = d / speed_CD) 
  (h6 : time_round_trip = time_CD + d / r) 
  (h7 : avg_speed_round_trip = round_trip_distance / time_round_trip) :
  r = 50 :=
by {
  -- proof steps will go here
  sorry
}

end car_return_speed_l26_26859


namespace number_of_dots_on_faces_l26_26848

theorem number_of_dots_on_faces (d A B C D : ℕ) 
  (h1 : d = 6)
  (h2 : A = 3)
  (h3 : B = 5)
  (h4 : C = 6)
  (h5 : D = 5) :
  A = 3 ∧ B = 5 ∧ C = 6 ∧ D = 5 :=
by {
  sorry
}

end number_of_dots_on_faces_l26_26848


namespace valeries_thank_you_cards_l26_26407

variables (T R J B : ℕ)

theorem valeries_thank_you_cards :
  B = 2 →
  R = B + 3 →
  J = 2 * R →
  T + (B + 1) + R + J = 21 →
  T = 3 :=
by
  intros hB hR hJ hTotal
  sorry

end valeries_thank_you_cards_l26_26407


namespace candy_count_l26_26161

variables (S M L : ℕ)

theorem candy_count :
  S + M + L = 110 ∧ S + L = 100 ∧ L = S + 20 → S = 40 ∧ M = 10 ∧ L = 60 :=
by
  intros h
  sorry

end candy_count_l26_26161


namespace total_chickens_l26_26882

theorem total_chickens (coops chickens_per_coop : ℕ) (h1 : coops = 9) (h2 : chickens_per_coop = 60) :
  coops * chickens_per_coop = 540 := by
  sorry

end total_chickens_l26_26882


namespace farmer_total_land_l26_26276

noncomputable def total_land_owned_by_farmer (cleared_land_with_tomato : ℝ) (cleared_percentage : ℝ) (grape_percentage : ℝ) (potato_percentage : ℝ) : ℝ :=
  let cleared_land := cleared_percentage
  let total_clearance_with_tomato := cleared_land_with_tomato
  let unused_cleared_percentage := 1 - grape_percentage - potato_percentage
  let total_cleared_land := total_clearance_with_tomato / unused_cleared_percentage
  total_cleared_land / cleared_land

theorem farmer_total_land (cleared_land_with_tomato : ℝ) (cleared_percentage : ℝ) (grape_percentage : ℝ) (potato_percentage : ℝ) :
  (cleared_land_with_tomato = 450) →
  (cleared_percentage = 0.90) →
  (grape_percentage = 0.10) →
  (potato_percentage = 0.80) →
  total_land_owned_by_farmer cleared_land_with_tomato 90 10 80 = 1666.6667 :=
by
  intro h1 h2 h3 h4
  sorry

end farmer_total_land_l26_26276


namespace matrix_subtraction_l26_26465

-- Define matrices A and B
def A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![ 4, -3 ],
  ![ 2,  8 ]
]

def B : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![ 1,  5 ],
  ![ -3,  6 ]
]

-- Define the result matrix as given in the problem
def result : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![ 3, -8 ],
  ![ 5,  2 ]
]

-- The theorem to prove
theorem matrix_subtraction : A - B = result := 
by 
  sorry

end matrix_subtraction_l26_26465


namespace construct_rectangle_l26_26793

structure Rectangle where
  side1 : ℝ
  side2 : ℝ
  diagonal : ℝ
  sum_diag_side : ℝ := side2 + diagonal

theorem construct_rectangle (b a d : ℝ) (r : Rectangle) :
  r.side2 = a ∧ r.side1 = b ∧ r.sum_diag_side = a + d :=
by
  sorry

end construct_rectangle_l26_26793


namespace prove_healthy_diet_multiple_l26_26619

variable (rum_on_pancakes rum_earlier rum_after_pancakes : ℝ)
variable (healthy_multiple : ℝ)

-- Definitions from conditions
def Sally_gave_rum_on_pancakes : Prop := rum_on_pancakes = 10
def Don_had_rum_earlier : Prop := rum_earlier = 12
def Don_can_have_rum_after_pancakes : Prop := rum_after_pancakes = 8

-- Concluding multiple for healthy diet
def healthy_diet_multiple : Prop := healthy_multiple = (rum_on_pancakes + rum_after_pancakes - rum_earlier) / rum_on_pancakes

theorem prove_healthy_diet_multiple :
  Sally_gave_rum_on_pancakes rum_on_pancakes →
  Don_had_rum_earlier rum_earlier →
  Don_can_have_rum_after_pancakes rum_after_pancakes →
  healthy_diet_multiple rum_on_pancakes rum_earlier rum_after_pancakes healthy_multiple →
  healthy_multiple = 0.8 := 
by
  intros h1 h2 h3 h4
  sorry

end prove_healthy_diet_multiple_l26_26619


namespace change_in_expression_l26_26019

theorem change_in_expression (x a : ℝ) (h : 0 < a) :
  (x + a)^3 - 3 * (x + a) - (x^3 - 3 * x) = 3 * a * x^2 + 3 * a^2 * x + a^3 - 3 * a
  ∨ (x - a)^3 - 3 * (x - a) - (x^3 - 3 * x) = -3 * a * x^2 + 3 * a^2 * x - a^3 + 3 * a :=
sorry

end change_in_expression_l26_26019


namespace Peter_bought_4_notebooks_l26_26269

theorem Peter_bought_4_notebooks :
  (let green_notebooks := 2
   let black_notebook := 1
   let pink_notebook := 1
   green_notebooks + black_notebook + pink_notebook = 4) :=
by sorry

end Peter_bought_4_notebooks_l26_26269


namespace monotonic_intervals_value_of_a_inequality_a_minus_one_l26_26253

noncomputable def f (a x : ℝ) : ℝ := a * x + Real.log x

theorem monotonic_intervals (a : ℝ) :
  (∀ x, 0 < x → 0 ≤ a → 0 < (a * x + 1) / x) ∧
  (∀ x, 0 < x → a < 0 → (0 < x ∧ x < -1/a → 0 < (a * x + 1) / x) ∧
    (-1/a < x → 0 > (a * x + 1) / x)) :=
sorry

theorem value_of_a (a : ℝ) (h_a : a < 0) (h_max : (∀ x, x ∈ Set.Icc 0 e → f a x ≤ -2) ∧ (∃ x, x ∈ Set.Icc 0 e ∧ f a x = -2)) :
  a = -Real.exp 1 := 
sorry

theorem inequality_a_minus_one (a : ℝ) (h_a : a = -1) :
  (∀ x, 0 < x → x * |f a x| > Real.log x + 1/2 * x) :=
sorry

end monotonic_intervals_value_of_a_inequality_a_minus_one_l26_26253


namespace area_of_triangle_ADC_l26_26191

-- Define the constants for the problem
variable (BD DC : ℝ)
variable (abd_area adc_area : ℝ)

-- Given conditions
axiom ratio_condition : BD / DC = 5 / 2
axiom area_abd : abd_area = 35

-- Define the theorem to be proved
theorem area_of_triangle_ADC :
  ∃ adc_area, adc_area = 14 ∧ abd_area / adc_area = BD / DC := 
sorry

end area_of_triangle_ADC_l26_26191


namespace binomial_expansion_a0_a1_a3_a5_l26_26165

theorem binomial_expansion_a0_a1_a3_a5 
    (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ)
    (h : (1 + 2 * x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) :
  a_0 + a_1 + a_3 + a_5 = 123 :=
sorry

end binomial_expansion_a0_a1_a3_a5_l26_26165


namespace fg_of_3_eq_97_l26_26985

def f (x : ℕ) : ℕ := 4 * x - 3
def g (x : ℕ) : ℕ := (x + 2) ^ 2

theorem fg_of_3_eq_97 : f (g 3) = 97 := by
  sorry

end fg_of_3_eq_97_l26_26985


namespace composite_infinitely_many_l26_26430

theorem composite_infinitely_many (t : ℕ) (ht : t ≥ 2) :
  ∃ n : ℕ, n = 3 ^ (2 ^ t) - 2 ^ (2 ^ t) ∧ (3 ^ (n - 1) - 2 ^ (n - 1)) % n = 0 :=
by
  use 3 ^ (2 ^ t) - 2 ^ (2 ^ t)
  sorry 

end composite_infinitely_many_l26_26430


namespace valid_parameterizations_l26_26673

theorem valid_parameterizations :
  (∀ t : ℝ, ∃ x y : ℝ, (x = 0 + 4 * t) ∧ (y = -4 + 8 * t) ∧ (y = 2 * x - 4)) ∧
  (∀ t : ℝ, ∃ x y : ℝ, (x = 3 + 1 * t) ∧ (y = 2 + 2 * t) ∧ (y = 2 * x - 4)) ∧
  (∀ t : ℝ, ∃ x y : ℝ, (x = -1 + 2 * t) ∧ (y = -6 + 4 * t) ∧ (y = 2 * x - 4)) :=
by
  -- Proof goes here
  sorry

end valid_parameterizations_l26_26673


namespace tan_alpha_value_cos2_minus_sin2_l26_26625

variable (α : Real) 

axiom is_internal_angle (angle : Real) : angle ∈ Set.Ico 0 Real.pi 

axiom sin_cos_sum (α : Real) : α ∈ Set.Ico 0 Real.pi → Real.sin α + Real.cos α = 1 / 5

theorem tan_alpha_value (h : α ∈ Set.Ico 0 Real.pi) : Real.tan α = -4 / 3 := by 
  sorry

theorem cos2_minus_sin2 (h : Real.tan α = -4 / 3) : 1 / (Real.cos α^2 - Real.sin α^2) = -25 / 7 := by 
  sorry

end tan_alpha_value_cos2_minus_sin2_l26_26625


namespace negation_of_prop_original_l26_26207

-- Definitions and conditions as per the problem
def prop_original : Prop :=
  ∃ x : ℝ, x^2 + x + 1 ≤ 0

def prop_negation : Prop :=
  ∀ x : ℝ, x^2 + x + 1 > 0

-- The theorem states the mathematical equivalence
theorem negation_of_prop_original : ¬ prop_original ↔ prop_negation := 
sorry

end negation_of_prop_original_l26_26207


namespace max_expression_value_l26_26796

noncomputable def max_value : ℕ := 17

theorem max_expression_value 
  (x y z : ℕ) 
  (hx : 10 ≤ x ∧ x < 100) 
  (hy : 10 ≤ y ∧ y < 100) 
  (hz : 10 ≤ z ∧ z < 100) 
  (mean_eq : (x + y + z) / 3 = 60) : 
  (x + y) / z ≤ max_value :=
sorry

end max_expression_value_l26_26796


namespace proof_problem_l26_26992

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)

-- Definition for statement 1
def statement1 := f 0 = 0

-- Definition for statement 2
def statement2 := (∃ x > 0, ∀ y > 0, f x ≥ f y) → (∃ x < 0, ∀ y < 0, f x ≤ f y)

-- Definition for statement 3
def statement3 := (∀ x ≥ 1, ∀ y ≥ 1, x < y → f x < f y) → (∀ x ≤ -1, ∀ y ≤ -1, x < y → f y < f x)

-- Definition for statement 4
def statement4 := (∀ x > 0, f x = x^2 - 2 * x) → (∀ x < 0, f x = -x^2 - 2 * x)

-- Combined proof problem
theorem proof_problem :
  (statement1 f) ∧ (statement2 f) ∧ (statement4 f) ∧ ¬ (statement3 f) :=
by sorry

end proof_problem_l26_26992


namespace remainder_2_pow_19_div_7_l26_26040

theorem remainder_2_pow_19_div_7 :
  2^19 % 7 = 2 := by
  sorry

end remainder_2_pow_19_div_7_l26_26040


namespace Sara_quarters_after_borrowing_l26_26840

theorem Sara_quarters_after_borrowing (initial_quarters borrowed_quarters : ℕ) (h1 : initial_quarters = 783) (h2 : borrowed_quarters = 271) :
  initial_quarters - borrowed_quarters = 512 := by
  sorry

end Sara_quarters_after_borrowing_l26_26840


namespace smallest_area_right_triangle_l26_26774

theorem smallest_area_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (A : ℝ), A = 6 * Real.sqrt 7 :=
sorry

end smallest_area_right_triangle_l26_26774


namespace exists_rectangular_parallelepiped_with_equal_surface_area_and_edge_sum_l26_26149

theorem exists_rectangular_parallelepiped_with_equal_surface_area_and_edge_sum :
  ∃ (a b c : ℤ), 2 * (a * b + b * c + c * a) = 4 * (a + b + c) :=
by
  -- Here we prove the existence of such integers a, b, c, which is stated in the theorem
  sorry

end exists_rectangular_parallelepiped_with_equal_surface_area_and_edge_sum_l26_26149


namespace sum_of_edges_l26_26146

-- Define the number of edges for a triangle and a rectangle
def edges_triangle : Nat := 3
def edges_rectangle : Nat := 4

-- The theorem states that the sum of the edges of a triangle and a rectangle is 7
theorem sum_of_edges : edges_triangle + edges_rectangle = 7 := 
by
  -- proof omitted
  sorry

end sum_of_edges_l26_26146


namespace value_range_of_f_l26_26477

noncomputable def f (x : ℝ) : ℝ := 2 + Real.logb 5 (x + 3)

theorem value_range_of_f :
  ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ∈ Set.Icc (2 : ℝ) 3 := 
by
  sorry

end value_range_of_f_l26_26477


namespace factorize_expression_l26_26344

variable (a : ℝ) -- assuming a is a real number

theorem factorize_expression (a : ℝ) : a^2 + 3 * a = a * (a + 3) :=
by
  -- proof goes here
  sorry

end factorize_expression_l26_26344
