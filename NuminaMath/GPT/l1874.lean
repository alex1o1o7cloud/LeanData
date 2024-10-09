import Mathlib

namespace range_of_m_l1874_187439

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (6 - 3 * (x + 1) < x - 9) ∧ (x - m > -1) ↔ (x > 3)) → (m ≤ 4) :=
by
  sorry

end range_of_m_l1874_187439


namespace tasks_to_shower_l1874_187410

-- Definitions of the conditions
def tasks_to_clean_house : Nat := 7
def tasks_to_make_dinner : Nat := 4
def minutes_per_task : Nat := 10
def total_minutes : Nat := 2 * 60

-- The theorem we want to prove
theorem tasks_to_shower (x : Nat) :
  total_minutes = (tasks_to_clean_house + tasks_to_make_dinner + x) * minutes_per_task →
  x = 1 := by
  sorry

end tasks_to_shower_l1874_187410


namespace mul_point_five_point_three_l1874_187438

theorem mul_point_five_point_three : 0.5 * 0.3 = 0.15 := 
by  sorry

end mul_point_five_point_three_l1874_187438


namespace garden_roller_diameter_l1874_187411

theorem garden_roller_diameter
  (l : ℝ) (A : ℝ) (r : ℕ) (pi : ℝ)
  (h_l : l = 2)
  (h_A : A = 44)
  (h_r : r = 5)
  (h_pi : pi = 22 / 7) :
  ∃ d : ℝ, d = 1.4 :=
by {
  sorry
}

end garden_roller_diameter_l1874_187411


namespace substance_same_number_of_atoms_l1874_187453

def molecule (kind : String) (atom_count : ℕ) := (kind, atom_count)

def H3PO4 := molecule "H₃PO₄" 8
def H2O2 := molecule "H₂O₂" 4
def H2SO4 := molecule "H₂SO₄" 7
def NaCl := molecule "NaCl" 2 -- though it consists of ions, let's denote it as 2 for simplicity
def HNO3 := molecule "HNO₃" 5

def mol_atoms (mol : ℝ) (molecule : ℕ) : ℝ := mol * molecule

theorem substance_same_number_of_atoms :
  mol_atoms 0.2 H3PO4.2 = mol_atoms 0.4 H2O2.2 :=
by
  unfold H3PO4 H2O2 mol_atoms
  sorry

end substance_same_number_of_atoms_l1874_187453


namespace snail_total_distance_l1874_187462

-- Conditions
def initial_pos : ℤ := 0
def pos1 : ℤ := 4
def pos2 : ℤ := -3
def pos3 : ℤ := 6

-- Total distance traveled by the snail
def distance_traveled : ℤ :=
  abs (pos1 - initial_pos) +
  abs (pos2 - pos1) +
  abs (pos3 - pos2)

-- Theorem statement
theorem snail_total_distance : distance_traveled = 20 :=
by
  -- Proof is omitted, as per request
  sorry

end snail_total_distance_l1874_187462


namespace solution_set_of_equation_l1874_187415

theorem solution_set_of_equation (x : ℝ) : 
  (abs (2 * x - 1) = abs x + abs (x - 1)) ↔ (x ≤ 0 ∨ x ≥ 1) := 
by 
  sorry

end solution_set_of_equation_l1874_187415


namespace tangent_expression_l1874_187461

theorem tangent_expression :
  (Real.tan (10 * Real.pi / 180) + Real.tan (50 * Real.pi / 180) + Real.tan (120 * Real.pi / 180))
  / (Real.tan (10 * Real.pi / 180) * Real.tan (50 * Real.pi / 180)) = -Real.sqrt 3 := by
  sorry

end tangent_expression_l1874_187461


namespace smallest_period_of_f_l1874_187474

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x + Real.cos x) ^ 2 + 1

theorem smallest_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi :=
by
  sorry

end smallest_period_of_f_l1874_187474


namespace proof_problem_l1874_187437

theorem proof_problem
  (x y : ℤ)
  (hx : ∃ m : ℤ, x = 6 * m)
  (hy : ∃ n : ℤ, y = 12 * n) :
  (x + y) % 2 = 0 ∧ (x + y) % 6 = 0 ∧ ¬ (x + y) % 12 = 0 → ¬ (x + y) % 12 = 0 :=
  sorry

end proof_problem_l1874_187437


namespace cost_of_each_pack_l1874_187472

theorem cost_of_each_pack (num_packs : ℕ) (total_paid : ℝ) (change_received : ℝ) 
(h1 : num_packs = 3) (h2 : total_paid = 20) (h3 : change_received = 11) : 
(total_paid - change_received) / num_packs = 3 := by
  sorry

end cost_of_each_pack_l1874_187472


namespace angle_AXC_angle_ACB_l1874_187465

-- Definitions of the problem conditions
variables (A B C D X : Type) [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty D] [Nonempty X]
variables (AD DC: Type) [Nonempty AD] [Nonempty DC]
variables (angleB angleXDC angleAXC angleACB : ℝ)
variables (AB BX: ℝ)

-- Given conditions
axiom equal_sides: AD = DC
axiom pointX: BX = AB
axiom given_angleB: angleB = 34
axiom given_angleXDC: angleXDC = 52

-- Proof goals (no proof included, only the statements)
theorem angle_AXC: angleAXC = 107 :=
sorry

theorem angle_ACB: angleACB = 47 :=
sorry

end angle_AXC_angle_ACB_l1874_187465


namespace q_investment_l1874_187459

theorem q_investment (p_investment : ℝ) (profit_ratio_p : ℝ) (profit_ratio_q : ℝ) (q_investment : ℝ) 
  (h1 : p_investment = 40000) 
  (h2 : profit_ratio_p / profit_ratio_q = 2 / 3) 
  : q_investment = 60000 := 
sorry

end q_investment_l1874_187459


namespace whiteboard_ink_cost_l1874_187450

/-- 
There are 5 classes: A, B, C, D, E
Class A: 3 whiteboards
Class B: 2 whiteboards
Class C: 4 whiteboards
Class D: 1 whiteboard
Class E: 3 whiteboards
The ink usage per whiteboard in each class:
Class A: 20ml per whiteboard
Class B: 25ml per whiteboard
Class C: 15ml per whiteboard
Class D: 30ml per whiteboard
Class E: 20ml per whiteboard
The cost of ink is 50 cents per ml
-/
def total_cost_in_dollars : ℕ :=
  let ink_usage_A := 3 * 20
  let ink_usage_B := 2 * 25
  let ink_usage_C := 4 * 15
  let ink_usage_D := 1 * 30
  let ink_usage_E := 3 * 20
  let total_ink_usage := ink_usage_A + ink_usage_B + ink_usage_C + ink_usage_D + ink_usage_E
  let total_cost_in_cents := total_ink_usage * 50
  total_cost_in_cents / 100

theorem whiteboard_ink_cost : total_cost_in_dollars = 130 := 
  by 
    sorry -- Proof needs to be implemented

end whiteboard_ink_cost_l1874_187450


namespace two_people_lying_l1874_187499

def is_lying (A B C D : Prop) : Prop :=
  (A ↔ ¬B) ∧ (B ↔ ¬C) ∧ (C ↔ ¬B) ∧ (D ↔ ¬A)

theorem two_people_lying (A B C D : Prop) (LA LB LC LD : Prop) :
  is_lying A B C D → (LA → ¬A) → (LB → ¬B) → (LC → ¬C) → (LD → ¬D) → (LA ∧ LC ∧ ¬LB ∧ ¬LD) :=
by
  sorry

end two_people_lying_l1874_187499


namespace work_completion_l1874_187489

theorem work_completion (a b : ℕ) (h1 : a + b = 5) (h2 : a = 10) : b = 10 := by
  sorry

end work_completion_l1874_187489


namespace cos_45_minus_cos_90_eq_sqrt2_over_2_l1874_187414

theorem cos_45_minus_cos_90_eq_sqrt2_over_2 :
  (Real.cos (45 * Real.pi / 180) - Real.cos (90 * Real.pi / 180)) = (Real.sqrt 2 / 2) :=
by
  have h1 : Real.cos (90 * Real.pi / 180) = 0 := by sorry
  have h2 : Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by sorry
  sorry

end cos_45_minus_cos_90_eq_sqrt2_over_2_l1874_187414


namespace find_y_from_equation_l1874_187485

theorem find_y_from_equation (y : ℕ) 
  (h : (12 ^ 2) * (6 ^ 3) / y = 72) : 
  y = 432 :=
  sorry

end find_y_from_equation_l1874_187485


namespace find_root_interval_l1874_187457

noncomputable def f : ℝ → ℝ := sorry

theorem find_root_interval :
  f 2 < 0 ∧ f 3 > 0 ∧ f 2.5 < 0 ∧ f 2.75 > 0 ∧ f 2.625 > 0 ∧ f 2.5625 > 0 →
  ∃ x, 2.5 < x ∧ x < 2.5625 ∧ f x = 0 := sorry

end find_root_interval_l1874_187457


namespace expand_polynomial_l1874_187463

theorem expand_polynomial (x : ℝ) : (5 * x + 3) * (6 * x ^ 2 + 2) = 30 * x ^ 3 + 18 * x ^ 2 + 10 * x + 6 :=
by
  sorry

end expand_polynomial_l1874_187463


namespace range_of_m_for_distinct_real_roots_of_quadratic_l1874_187486

theorem range_of_m_for_distinct_real_roots_of_quadratic (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 4*x1 - m = 0 ∧ x2^2 + 4*x2 - m = 0) ↔ m > -4 :=
by
  sorry

end range_of_m_for_distinct_real_roots_of_quadratic_l1874_187486


namespace octal_to_decimal_equiv_l1874_187495

-- Definitions for the octal number 724
def d0 := 4
def d1 := 2
def d2 := 7

-- Definition for the base
def base := 8

-- Calculation of the decimal equivalent
def calc_decimal : ℕ :=
  d0 * base^0 + d1 * base^1 + d2 * base^2

-- The proof statement
theorem octal_to_decimal_equiv : calc_decimal = 468 := by
  sorry

end octal_to_decimal_equiv_l1874_187495


namespace inverse_of_matrix_l1874_187471

open Matrix

def mat : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![4, 9], ![2, 5]]

def inv_mat : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![5/2, -9/2], ![-1, 2]]

theorem inverse_of_matrix :
  ∃ (inv : Matrix (Fin 2) (Fin 2) ℚ), 
    inv * mat = 1 ∧ mat * inv = 1 :=
  ⟨inv_mat, by
    -- Providing the proof steps here is beyond the scope
    sorry⟩

end inverse_of_matrix_l1874_187471


namespace bus_initial_count_l1874_187473

theorem bus_initial_count (x : ℕ) (got_off : ℕ) (remained : ℕ) (h1 : got_off = 47) (h2 : remained = 43) (h3 : x - got_off = remained) : x = 90 :=
by
  rw [h1, h2] at h3
  sorry

end bus_initial_count_l1874_187473


namespace odd_factors_count_l1874_187454

-- Definition of the number to factorize
def n : ℕ := 252

-- The prime factorization of 252 (ignoring the even factor 2)
def p1 : ℕ := 3
def p2 : ℕ := 7
def e1 : ℕ := 2  -- exponent of 3 in the factorization
def e2 : ℕ := 1  -- exponent of 7 in the factorization

-- The statement to prove
theorem odd_factors_count : 
  let odd_factor_count := (e1 + 1) * (e2 + 1)
  odd_factor_count = 6 :=
by
  sorry

end odd_factors_count_l1874_187454


namespace cos_angle_identity_l1874_187460

theorem cos_angle_identity (a : ℝ) (h : Real.sin (π / 6 - a) - Real.cos a = 1 / 3) :
  Real.cos (2 * a + π / 3) = 7 / 9 :=
by
  sorry

end cos_angle_identity_l1874_187460


namespace distance_apart_l1874_187400

def race_total_distance : ℕ := 1000
def distance_Arianna_ran : ℕ := 184

theorem distance_apart :
  race_total_distance - distance_Arianna_ran = 816 :=
by
  sorry

end distance_apart_l1874_187400


namespace area_of_isosceles_trapezoid_l1874_187496

variable (a b c d : ℝ) -- Variables for sides and bases of the trapezoid

-- Define isosceles trapezoid with given sides and bases
def is_isosceles_trapezoid (a b c d : ℝ) (h : ℝ) :=
  a = b ∧ c = 10 ∧ d = 16 ∧ (∃ (h : ℝ), a^2 = h^2 + ((d - c) / 2)^2 ∧ a = 5)

-- Lean theorem for the area of the isosceles trapezoid
theorem area_of_isosceles_trapezoid :
  ∀ (a b c d : ℝ) (h : ℝ), is_isosceles_trapezoid a b c d h
  → (1 / 2) * (c + d) * h = 52 :=
by
  sorry

end area_of_isosceles_trapezoid_l1874_187496


namespace players_on_team_are_4_l1874_187404

noncomputable def number_of_players (score_old_record : ℕ) (rounds : ℕ) (score_first_9_rounds : ℕ) (final_round_diff : ℕ) :=
  let points_needed := score_old_record * rounds
  let points_final_needed := score_old_record - final_round_diff
  let total_points_needed := points_needed * 1
  let final_round_points_needed := total_points_needed - score_first_9_rounds
  let P := final_round_points_needed / points_final_needed
  P

theorem players_on_team_are_4 :
  number_of_players 287 10 10440 27 = 4 :=
by
  sorry

end players_on_team_are_4_l1874_187404


namespace frances_card_value_l1874_187407

theorem frances_card_value (x : ℝ) (hx : 90 < x ∧ x < 180) :
  (∃ f : ℝ → ℝ, f = sin ∨ f = cos ∨ f = tan ∧
    f x = -1 ∧
    (∃ y : ℝ, y ≠ x ∧ (sin y ≠ -1 ∧ cos y ≠ -1 ∧ tan y ≠ -1))) :=
sorry

end frances_card_value_l1874_187407


namespace insert_digits_identical_l1874_187494

theorem insert_digits_identical (A B : List Nat) (hA : A.length = 2007) (hB : B.length = 2007)
  (hErase : ∃ (C : List Nat) (erase7A : List Nat → List Nat) (erase7B : List Nat → List Nat),
    (erase7A A = C) ∧ (erase7B B = C) ∧ (C.length = 2000)) :
  ∃ (D : List Nat) (insert7A : List Nat → List Nat) (insert7B : List Nat → List Nat),
    (insert7A A = D) ∧ (insert7B B = D) ∧ (D.length = 2014) := sorry

end insert_digits_identical_l1874_187494


namespace Jake_weight_loss_l1874_187408

variable (J S: ℕ) (x : ℕ)

theorem Jake_weight_loss:
  J = 93 -> J + S = 132 -> J - x = 2 * S -> x = 15 :=
by
  intros hJ hJS hCondition
  sorry

end Jake_weight_loss_l1874_187408


namespace factor_polynomial_l1874_187449

theorem factor_polynomial (x : ℝ) : 75 * x^5 - 300 * x^10 = 75 * x^5 * (1 - 4 * x^5) :=
by
  sorry

end factor_polynomial_l1874_187449


namespace find_rth_term_l1874_187421

def S (n : ℕ) : ℕ := 2 * n + 3 * (n^3)

def a (r : ℕ) : ℕ := S r - S (r - 1)

theorem find_rth_term (r : ℕ) : a r = 9 * r^2 - 9 * r + 5 := by
  sorry

end find_rth_term_l1874_187421


namespace rowing_problem_l1874_187434

theorem rowing_problem (R S x y : ℝ) 
  (h1 : R = y + x) 
  (h2 : S = y - x) : 
  x = (R - S) / 2 ∧ y = (R + S) / 2 :=
by
  sorry

end rowing_problem_l1874_187434


namespace Zilla_savings_l1874_187446

-- Define the conditions
def rent_expense (E : ℝ) (R : ℝ) := R = 0.07 * E
def other_expenses (E : ℝ) := 0.5 * E
def amount_saved (E : ℝ) (R : ℝ) (S : ℝ) := S = E - (R + other_expenses E)

-- Define the main problem statement
theorem Zilla_savings (E R S: ℝ) 
    (hR : rent_expense E R)
    (hR_val : R = 133)
    (hS : amount_saved E R S) : 
    S = 817 := by
  sorry

end Zilla_savings_l1874_187446


namespace inequality_true_l1874_187455

noncomputable def f : ℝ → ℝ := sorry -- f is a function defined on (0, +∞)

axiom f_derivative (x : ℝ) (hx : 0 < x) : ∃ f'' : ℝ → ℝ, f'' x * x + 2 * f x = 1 / x^2

theorem inequality_true : (f 2) / 9 < (f 3) / 4 :=
  sorry

end inequality_true_l1874_187455


namespace area_difference_l1874_187442

-- Definitions of the given conditions
structure Triangle :=
(base : ℝ)
(height : ℝ)

def area (t : Triangle) : ℝ :=
  0.5 * t.base * t.height

-- Conditions of the problem
def EFG : Triangle := {base := 8, height := 4}
def EFG' : Triangle := {base := 4, height := 2}

-- Proof statement
theorem area_difference :
  area EFG - area EFG' = 12 :=
by
  sorry

end area_difference_l1874_187442


namespace min_value_of_quadratic_l1874_187440

theorem min_value_of_quadratic (x y : ℝ) : (x^2 + 2*x*y + y^2) ≥ 0 ∧ ∃ x y, x = -y ∧ x^2 + 2*x*y + y^2 = 0 := by
  sorry

end min_value_of_quadratic_l1874_187440


namespace sin_alpha_given_cos_alpha_plus_pi_over_3_l1874_187435

theorem sin_alpha_given_cos_alpha_plus_pi_over_3 
  (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos (α + π / 3) = 1 / 5) : 
  Real.sin α = (2 * Real.sqrt 6 - Real.sqrt 3) / 10 := 
by 
  sorry

end sin_alpha_given_cos_alpha_plus_pi_over_3_l1874_187435


namespace divide_gray_area_l1874_187466

-- The conditions
variables {A_rectangle A_square : ℝ} (h : 0 ≤ A_square ∧ A_square ≤ A_rectangle)

-- The main statement
theorem divide_gray_area : ∃ l : ℝ → ℝ → Prop, (∀ (x : ℝ), l x (A_rectangle / 2)) ∧ (∀ (y : ℝ), l (A_square / 2) y) ∧ (A_rectangle - A_square) / 2 = (A_rectangle - A_square) / 2 := by sorry

end divide_gray_area_l1874_187466


namespace sum_of_first_33_terms_arith_seq_l1874_187497

noncomputable def sum_arith_prog (a_1 d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a_1 + (n - 1) * d)

theorem sum_of_first_33_terms_arith_seq :
  ∃ (a_1 d : ℝ), (4 * a_1 + 64 * d = 28) → (sum_arith_prog a_1 d 33 = 231) :=
by
  sorry

end sum_of_first_33_terms_arith_seq_l1874_187497


namespace polynomial_divisibility_l1874_187482

theorem polynomial_divisibility : 
  ∃ k : ℤ, (k = 8) ∧ (∀ x : ℂ, (4 * x^3 - 8 * x^2 + k * x - 16) % (x - 2) = 0) ∧ 
           (∀ x : ℂ, (4 * x^3 - 8 * x^2 + k * x - 16) % (x^2 + 1) = 0) :=
sorry

end polynomial_divisibility_l1874_187482


namespace largest_square_area_correct_l1874_187428

noncomputable def area_of_largest_square (x y z : ℝ) : Prop := 
  ∃ (area : ℝ), (z^2 = area) ∧ 
                 (x^2 + y^2 = z^2) ∧ 
                 (x^2 + y^2 + 2*z^2 = 722) ∧ 
                 (area = 722 / 3)

theorem largest_square_area_correct (x y z : ℝ) :
  area_of_largest_square x y z :=
  sorry

end largest_square_area_correct_l1874_187428


namespace purchasing_methods_count_l1874_187469

def material_cost : ℕ := 40
def instrument_cost : ℕ := 60
def budget : ℕ := 400
def min_materials : ℕ := 4
def min_instruments : ℕ := 2

theorem purchasing_methods_count : 
  (∃ (n_m m : ℕ), 
    n_m ≥ min_materials ∧ m ≥ min_instruments ∧ 
    n_m * material_cost + m * instrument_cost ≤ budget) → 
  (∃ (count : ℕ), count = 7) :=
by 
  sorry

end purchasing_methods_count_l1874_187469


namespace rectangular_board_area_l1874_187420

variable (length width : ℕ)

theorem rectangular_board_area
  (h1 : length = 2 * width)
  (h2 : 2 * length + 2 * width = 84) :
  length * width = 392 := 
by
  sorry

end rectangular_board_area_l1874_187420


namespace quadratic_square_binomial_l1874_187445

theorem quadratic_square_binomial (k : ℝ) : (∃ b : ℝ, x^2 - 18 * x + k = (x + b) ^ 2) ↔ k = 81 := by
  sorry

end quadratic_square_binomial_l1874_187445


namespace melanie_attended_games_l1874_187481

/-- Melanie attended 5 football games if there were 12 total games and she missed 7. -/
theorem melanie_attended_games (totalGames : ℕ) (missedGames : ℕ) (h₁ : totalGames = 12) (h₂ : missedGames = 7) :
  totalGames - missedGames = 5 := 
sorry

end melanie_attended_games_l1874_187481


namespace domain_h_l1874_187441

noncomputable def h (x : ℝ) : ℝ := (x^2 + 5 * x + 6) / (|x - 2| + |x + 2|)

theorem domain_h : ∀ x : ℝ, ∃ y : ℝ, y = h x :=
by
  sorry

end domain_h_l1874_187441


namespace radio_price_rank_l1874_187447

theorem radio_price_rank (total_items : ℕ) (radio_position_highest : ℕ) (radio_position_lowest : ℕ) 
  (h1 : total_items = 40) (h2 : radio_position_highest = 17) : 
  radio_position_lowest = total_items - radio_position_highest + 1 :=
by
  sorry

end radio_price_rank_l1874_187447


namespace total_laundry_time_correct_l1874_187452

-- Define the washing and drying times for each load
def whites_washing_time : Nat := 72
def whites_drying_time : Nat := 50
def darks_washing_time : Nat := 58
def darks_drying_time : Nat := 65
def colors_washing_time : Nat := 45
def colors_drying_time : Nat := 54

-- Define total times for each load
def whites_total_time : Nat := whites_washing_time + whites_drying_time
def darks_total_time : Nat := darks_washing_time + darks_drying_time
def colors_total_time : Nat := colors_washing_time + colors_drying_time

-- Define the total time for all three loads
def total_laundry_time : Nat := whites_total_time + darks_total_time + colors_total_time

-- The proof statement
theorem total_laundry_time_correct : total_laundry_time = 344 := by
  unfold total_laundry_time
  unfold whites_total_time darks_total_time colors_total_time
  unfold whites_washing_time whites_drying_time
  unfold darks_washing_time darks_drying_time
  unfold colors_washing_time colors_drying_time
  sorry

end total_laundry_time_correct_l1874_187452


namespace cistern_total_wet_surface_area_l1874_187467

-- Define the length, width, and depth of water in the cistern
def length : ℝ := 9
def width : ℝ := 4
def depth : ℝ := 1.25

-- Define the bottom surface area
def bottom_surface_area : ℝ := length * width

-- Define the longer side surface area
def longer_side_surface_area_each : ℝ := depth * length

-- Define the shorter end surface area
def shorter_end_surface_area_each : ℝ := depth * width

-- Calculate the total wet surface area
def total_wet_surface_area : ℝ := bottom_surface_area + 2 * longer_side_surface_area_each + 2 * shorter_end_surface_area_each

-- The theorem to be proved
theorem cistern_total_wet_surface_area :
  total_wet_surface_area = 68.5 :=
by
  -- since bottom_surface_area = 36,
  -- 2 * longer_side_surface_area_each = 22.5, and
  -- 2 * shorter_end_surface_area_each = 10
  -- the total will be equal to 68.5
  sorry

end cistern_total_wet_surface_area_l1874_187467


namespace solve_eq1_solve_eq2_l1874_187431

theorem solve_eq1 (x : ℝ) : (x+1)^2 = 4 ↔ x = 1 ∨ x = -3 := 
by sorry

theorem solve_eq2 (x : ℝ) : 3*x^2 - 2*x - 1 = 0 ↔ x = 1 ∨ x = -1/3 := 
by sorry

end solve_eq1_solve_eq2_l1874_187431


namespace sachin_age_l1874_187429

theorem sachin_age (S R : ℕ) (h1 : R = S + 18) (h2 : S * 9 = R * 7) : S = 63 := 
by
  sorry

end sachin_age_l1874_187429


namespace maximum_distance_correct_l1874_187470

noncomputable def maximum_distance 
  (m : ℝ) 
  (θ : ℝ) 
  (P : ℝ × ℝ) 
  (intersection : (x y : ℝ) → (x + m * y = 0) ∧ (m * x - y - 2 * m + 4 = 0) → P = (x, y)) 
  (distance : (x y : ℝ) → (x - 1) * Real.cos θ + (y - 2) * Real.sin θ = 3) : 
  ℝ :=
3 + Real.sqrt 5

theorem maximum_distance_correct 
  (m : ℝ) 
  (θ : ℝ) 
  (P : ℝ × ℝ) 
  (intersection : (x y : ℝ) → (x + m * y = 0) ∧ (m * x - y - 2 * m + 4 = 0) → P = (x, y)) 
  (distance : (x y : ℝ) → (x - 1) * Real.cos θ + (y - 2) * Real.sin θ = 3) : 
  maximum_distance m θ P intersection distance = 3 + Real.sqrt 5 := 
sorry

end maximum_distance_correct_l1874_187470


namespace find_min_max_value_l1874_187476

open Real

theorem find_min_max_value (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_pos_d : 0 < d) (h_det : b^2 - 4 * a * c < 0) :
  ∃ (min_val max_val: ℝ),
    min_val = (2 * d * sqrt (a * c)) / (b + 2 * sqrt (a * c)) ∧ 
    max_val = (2 * d * sqrt (a * c)) / (b - 2 * sqrt (a * c)) ∧
    (∀ x y : ℝ, a * x^2 + c * y^2 ≥ min_val ∧ a * x^2 + c * y^2 ≤ max_val) :=
by
  -- Proof goes here
  sorry

end find_min_max_value_l1874_187476


namespace average_minutes_run_is_44_over_3_l1874_187423

open BigOperators

def average_minutes_run (s : ℕ) : ℚ :=
  let sixth_graders := 3 * s
  let seventh_graders := s
  let eighth_graders := s / 2
  let total_students := sixth_graders + seventh_graders + eighth_graders
  let total_minutes_run := 20 * sixth_graders + 12 * eighth_graders
  total_minutes_run / total_students

theorem average_minutes_run_is_44_over_3 (s : ℕ) (h1 : 0 < s) : 
  average_minutes_run s = 44 / 3 := 
by
  sorry

end average_minutes_run_is_44_over_3_l1874_187423


namespace sum_of_numbers_l1874_187412

theorem sum_of_numbers : 72.52 + 12.23 + 5.21 = 89.96 :=
by sorry

end sum_of_numbers_l1874_187412


namespace product_combination_count_l1874_187403

-- Definitions of the problem

-- There are 6 different types of cookies
def num_cookies : Nat := 6

-- There are 4 different types of milk
def num_milks : Nat := 4

-- Charlie will not order more than one of the same type
def charlie_order_limit : Nat := 1

-- Delta will only order cookies, including repeats of types
def delta_only_cookies : Bool := true

-- Prove that there are 2531 ways for Charlie and Delta to leave the store with 4 products collectively
theorem product_combination_count : 
  (number_of_ways : Nat) = 2531 
  := sorry

end product_combination_count_l1874_187403


namespace equal_x_l1874_187402

theorem equal_x (x y : ℝ) (h : x / (x + 2) = (y^2 + 3 * y - 2) / (y^2 + 3 * y + 1)) :
  x = (2 * y^2 + 6 * y - 4) / 3 :=
sorry

end equal_x_l1874_187402


namespace cuboid_third_edge_length_l1874_187427

theorem cuboid_third_edge_length
  (l w : ℝ)
  (A : ℝ)
  (h : ℝ)
  (hl : l = 4)
  (hw : w = 5)
  (hA : A = 148)
  (surface_area_formula : A = 2 * (l * w + l * h + w * h)) :
  h = 6 :=
by
  sorry

end cuboid_third_edge_length_l1874_187427


namespace distance_to_first_museum_l1874_187418

theorem distance_to_first_museum (x : ℝ) 
  (dist_second_museum : ℝ) 
  (total_distance : ℝ) 
  (h1 : dist_second_museum = 15) 
  (h2 : total_distance = 40) 
  (h3 : 2 * x + 2 * dist_second_museum = total_distance) : x = 5 :=
by 
  sorry

end distance_to_first_museum_l1874_187418


namespace max_a_for_necessary_not_sufficient_condition_l1874_187443

theorem max_a_for_necessary_not_sufficient_condition {x a : ℝ} (h : ∀ x, x^2 > 1 → x < a) : a = -1 :=
by sorry

end max_a_for_necessary_not_sufficient_condition_l1874_187443


namespace perfect_square_condition_l1874_187451

theorem perfect_square_condition (n : ℤ) : 
    ∃ k : ℤ, n^2 + 6*n + 1 = k^2 ↔ n = 0 ∨ n = -6 := by
  sorry

end perfect_square_condition_l1874_187451


namespace max_subset_no_ap_l1874_187491

theorem max_subset_no_ap (n : ℕ) (H : n ≥ 4) :
  ∃ (s : Finset ℝ), (s.card ≥ ⌊Real.sqrt (2 * n / 3)⌋₊ + 1) ∧
  ∀ (a b c : ℝ), a ∈ s → b ∈ s → c ∈ s → a ≠ b → a ≠ c → b ≠ c → (a, b, c) ≠ (a + b - c, b, c) :=
sorry

end max_subset_no_ap_l1874_187491


namespace part1_beef_noodles_mix_sauce_purchased_l1874_187405

theorem part1_beef_noodles_mix_sauce_purchased (x y : ℕ) (h1 : x + y = 170) (h2 : 15 * x + 20 * y = 3000) :
  x = 80 ∧ y = 90 :=
sorry

end part1_beef_noodles_mix_sauce_purchased_l1874_187405


namespace number_of_ones_and_zeros_not_perfect_square_l1874_187409

open Int

theorem number_of_ones_and_zeros_not_perfect_square (k : ℕ) : 
  let N := (10^k) * (10^300 - 1) / 9
  ¬ ∃ m : ℤ, m^2 = N :=
by
  sorry

end number_of_ones_and_zeros_not_perfect_square_l1874_187409


namespace solve_system_of_equations_l1874_187416

theorem solve_system_of_equations :
  ∀ (x1 x2 x3 x4 x5: ℝ), 
  (x3 + x4 + x5)^5 = 3 * x1 ∧ 
  (x4 + x5 + x1)^5 = 3 * x2 ∧ 
  (x5 + x1 + x2)^5 = 3 * x3 ∧ 
  (x1 + x2 + x3)^5 = 3 * x4 ∧ 
  (x2 + x3 + x4)^5 = 3 * x5 →
  (x1 = 0 ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 0 ∧ x5 = 0) ∨ 
  (x1 = 1/3 ∧ x2 = 1/3 ∧ x3 = 1/3 ∧ x4 = 1/3 ∧ x5 = 1/3) ∨ 
  (x1 = -1/3 ∧ x2 = -1/3 ∧ x3 = -1/3 ∧ x4 = -1/3 ∧ x5 = -1/3) := 
by 
  sorry

end solve_system_of_equations_l1874_187416


namespace average_chemistry_mathematics_l1874_187468

noncomputable def marks (P C M B : ℝ) : Prop := 
  P + C + M + B = (P + B) + 180 ∧ P = 1.20 * B

theorem average_chemistry_mathematics 
  (P C M B : ℝ) (h : marks P C M B) : (C + M) / 2 = 90 :=
by
  sorry

end average_chemistry_mathematics_l1874_187468


namespace min_value_of_2x_plus_y_l1874_187413

theorem min_value_of_2x_plus_y 
  (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : 1 / (x + 1) + 1 / (x + 2 * y) = 1) : 
  (2 * x + y) = 1 / 2 + Real.sqrt 3 := 
sorry

end min_value_of_2x_plus_y_l1874_187413


namespace travel_remaining_distance_l1874_187417

-- Definitions of given conditions
def total_distance : ℕ := 369
def amoli_speed : ℕ := 42
def amoli_time : ℕ := 3
def anayet_speed : ℕ := 61
def anayet_time : ℕ := 2

-- Define the distances each person traveled
def amoli_distance := amoli_speed * amoli_time
def anayet_distance := anayet_speed * anayet_time

-- Define the total distance covered
def total_covered := amoli_distance + anayet_distance

-- Define the remaining distance
def remaining_distance := total_distance - total_covered

-- Prove the remaining distance is 121 miles
theorem travel_remaining_distance : remaining_distance = 121 := by
  sorry

end travel_remaining_distance_l1874_187417


namespace infinite_solutions_eq_a_l1874_187444

variable (a x y: ℝ)

-- Define the two equations
def eq1 : Prop := a * x + y - 1 = 0
def eq2 : Prop := 4 * x + a * y - 2 = 0

theorem infinite_solutions_eq_a (h : ∃ x y, eq1 a x y ∧ eq2 a x y) :
  a = 2 := 
sorry

end infinite_solutions_eq_a_l1874_187444


namespace transform_point_c_l1874_187492

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

def reflect_diag (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

theorem transform_point_c :
  let C := (3, 2)
  let C' := reflect_x C
  let C'' := reflect_y C'
  let C''' := reflect_diag C''
  C''' = (-2, -3) :=
by
  sorry

end transform_point_c_l1874_187492


namespace slope_of_line_l1874_187432

theorem slope_of_line (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (1, 2)) (h2 : (x2, y2) = (4, 8)) :
  (y2 - y1) / (x2 - x1) = 2 := 
by
  sorry

end slope_of_line_l1874_187432


namespace not_sufficient_nor_necessary_l1874_187433

theorem not_sufficient_nor_necessary (a b : ℝ) : ¬((a > b) → (a^2 > b^2)) ∧ ¬((a^2 > b^2) → (a > b)) := 
by 
  sorry

end not_sufficient_nor_necessary_l1874_187433


namespace less_than_subtraction_l1874_187419

-- Define the numbers as real numbers
def a : ℝ := 47.2
def b : ℝ := 0.5

-- Theorem statement
theorem less_than_subtraction : a - b = 46.7 :=
by
  sorry

end less_than_subtraction_l1874_187419


namespace roots_of_polynomial_l1874_187483

theorem roots_of_polynomial :
  (∀ x : ℝ, (x^2 - 5 * x + 6) * x * (x - 5) = 0 ↔ x = 0 ∨ x = 2 ∨ x = 3 ∨ x = 5) :=
by
  sorry

end roots_of_polynomial_l1874_187483


namespace problem_l1874_187436

theorem problem (p q : ℕ) (hp : 0 < p) (hq : 0 < q) (h1 : p + 5 < q)
  (h2 : (p + (p + 2) + (p + 5) + q + (q + 1) + (2 * q - 1)) / 6 = q)
  (h3 : (p + 5 + q) / 2 = q) : p + q = 11 :=
by sorry

end problem_l1874_187436


namespace diophantine_equation_solvable_l1874_187430

theorem diophantine_equation_solvable (a : ℕ) (ha : 0 < a) : 
  ∃ (x y : ℤ), x^2 - y^2 = a^3 :=
by
  let x := (a * (a + 1)) / 2
  let y := (a * (a - 1)) / 2
  have hx : x^2 = (a * (a + 1) / 2 : ℤ)^2 := sorry
  have hy : y^2 = (a * (a - 1) / 2 : ℤ)^2 := sorry
  use x
  use y
  sorry

end diophantine_equation_solvable_l1874_187430


namespace distinct_arrangements_l1874_187484

-- Defining the conditions as constants
def num_women : ℕ := 9
def num_men : ℕ := 3
def total_slots : ℕ := num_women + num_men

-- Using the combination formula directly as part of the statement
theorem distinct_arrangements : Nat.choose total_slots num_men = 220 := by
  sorry

end distinct_arrangements_l1874_187484


namespace farm_needs_12880_ounces_of_horse_food_per_day_l1874_187480

-- Define the given conditions
def ratio_sheep_to_horses : ℕ × ℕ := (1, 7)
def food_per_horse_per_day : ℕ := 230
def number_of_sheep : ℕ := 8

-- Define the proof goal
theorem farm_needs_12880_ounces_of_horse_food_per_day :
  let number_of_horses := number_of_sheep * ratio_sheep_to_horses.2
  number_of_horses * food_per_horse_per_day = 12880 :=
by
  sorry

end farm_needs_12880_ounces_of_horse_food_per_day_l1874_187480


namespace drying_time_l1874_187475

theorem drying_time
  (time_short : ℕ := 10) -- Time to dry a short-haired dog in minutes
  (time_full : ℕ := time_short * 2) -- Time to dry a full-haired dog in minutes, which is twice as long
  (num_short : ℕ := 6) -- Number of short-haired dogs
  (num_full : ℕ := 9) -- Number of full-haired dogs
  : (time_short * num_short + time_full * num_full) / 60 = 4 := 
by
  sorry

end drying_time_l1874_187475


namespace inverse_proportionality_l1874_187477

theorem inverse_proportionality:
  (∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → y = k / x) ∧ y = 1 ∧ x = 2 →
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → y = 2 / x :=
by
  sorry

end inverse_proportionality_l1874_187477


namespace max_hours_is_70_l1874_187479

-- Define the conditions
def regular_hourly_rate : ℕ := 8
def first_20_hours : ℕ := 20
def max_weekly_earnings : ℕ := 660
def overtime_rate_multiplier : ℕ := 25

-- Define the overtime hourly rate
def overtime_hourly_rate : ℕ := regular_hourly_rate + (regular_hourly_rate * overtime_rate_multiplier / 100)

-- Define the earnings for the first 20 hours
def earnings_first_20_hours : ℕ := regular_hourly_rate * first_20_hours

-- Define the maximum overtime earnings
def max_overtime_earnings : ℕ := max_weekly_earnings - earnings_first_20_hours

-- Define the maximum overtime hours
def max_overtime_hours : ℕ := max_overtime_earnings / overtime_hourly_rate

-- Define the maximum total hours
def max_total_hours : ℕ := first_20_hours + max_overtime_hours

-- Theorem to prove that the maximum number of hours is 70
theorem max_hours_is_70 : max_total_hours = 70 :=
by
  sorry

end max_hours_is_70_l1874_187479


namespace volume_parallelepiped_l1874_187406

noncomputable def volume_of_parallelepiped (m n p d : ℝ) : ℝ :=
  if h : m > 0 ∧ n > 0 ∧ p > 0 ∧ d > 0 then
    m * n * p * (d^3) / (m^2 + n^2 + p^2)^(3/2)
  else 0

theorem volume_parallelepiped (m n p d : ℝ) (hm : m > 0) (hn : n > 0) (hp : p > 0) (hd : d > 0) :
  volume_of_parallelepiped m n p d = m * n * p * (d^3) / (m^2 + n^2 + p^2)^(3/2) := by
  sorry

end volume_parallelepiped_l1874_187406


namespace system_solution_l1874_187493

theorem system_solution (x y z : ℝ) :
    x + y + z = 2 ∧ 
    x^2 + y^2 + z^2 = 26 ∧
    x^3 + y^3 + z^3 = 38 →
    (x = 1 ∧ y = 4 ∧ z = -3) ∨
    (x = 1 ∧ y = -3 ∧ z = 4) ∨
    (x = 4 ∧ y = 1 ∧ z = -3) ∨
    (x = 4 ∧ y = -3 ∧ z = 1) ∨
    (x = -3 ∧ y = 1 ∧ z = 4) ∨
    (x = -3 ∧ y = 4 ∧ z = 1) := by
  sorry

end system_solution_l1874_187493


namespace number_of_younger_siblings_l1874_187401

-- Definitions based on the problem conditions
def Nicole_cards : ℕ := 400
def Cindy_cards : ℕ := 2 * Nicole_cards
def Combined_cards : ℕ := Nicole_cards + Cindy_cards
def Rex_cards : ℕ := Combined_cards / 2
def Rex_remaining_cards : ℕ := 150
def Total_shares : ℕ := Rex_cards / Rex_remaining_cards
def Rex_share : ℕ := 1

-- The theorem to prove how many younger siblings Rex has
theorem number_of_younger_siblings :
  Total_shares - Rex_share = 3 :=
  by
    sorry

end number_of_younger_siblings_l1874_187401


namespace adjacent_product_negative_l1874_187498

noncomputable def a_seq : ℕ → ℚ
| 0 => 15
| (n+1) => (a_seq n) - (2 / 3)

theorem adjacent_product_negative :
  ∃ n : ℕ, a_seq 22 * a_seq 23 < 0 :=
by
  -- From the conditions, it is known that a_seq satisfies the recursive definition
  --
  -- We seek to prove that a_seq 22 * a_seq 23 < 0
  sorry

end adjacent_product_negative_l1874_187498


namespace equivalent_operation_l1874_187425

theorem equivalent_operation (x : ℚ) :
  (x / (5 / 6) * (4 / 7)) = x * (24 / 35) :=
by
  sorry

end equivalent_operation_l1874_187425


namespace remaining_slices_correct_l1874_187422

-- Define initial slices of pie and cake
def initial_pie_slices : Nat := 2 * 8
def initial_cake_slices : Nat := 12

-- Define slices eaten on Friday
def friday_pie_slices_eaten : Nat := 2
def friday_cake_slices_eaten : Nat := 2

-- Define slices eaten on Saturday
def saturday_pie_slices_eaten (remaining: Nat) : Nat := remaining / 2 -- 50%
def saturday_cake_slices_eaten (remaining: Nat) : Nat := remaining / 4 -- 25%

-- Define slices eaten on Sunday morning
def sunday_morning_pie_slices_eaten : Nat := 2
def sunday_morning_cake_slices_eaten : Nat := 3

-- Define slices eaten on Sunday evening
def sunday_evening_pie_slices_eaten : Nat := 4
def sunday_evening_cake_slices_eaten : Nat := 1

-- Function to calculate remaining slices
def remaining_slices : Nat × Nat :=
  let after_friday_pies := initial_pie_slices - friday_pie_slices_eaten
  let after_friday_cake := initial_cake_slices - friday_cake_slices_eaten
  let after_saturday_pies := after_friday_pies - saturday_pie_slices_eaten after_friday_pies
  let after_saturday_cake := after_friday_cake - saturday_cake_slices_eaten after_friday_cake
  let after_sunday_morning_pies := after_saturday_pies - sunday_morning_pie_slices_eaten
  let after_sunday_morning_cake := after_saturday_cake - sunday_morning_cake_slices_eaten
  let final_pies := after_sunday_morning_pies - sunday_evening_pie_slices_eaten
  let final_cake := after_sunday_morning_cake - sunday_evening_cake_slices_eaten
  (final_pies, final_cake)

theorem remaining_slices_correct :
  remaining_slices = (1, 4) :=
  by {
    sorry -- Proof is omitted
  }

end remaining_slices_correct_l1874_187422


namespace avg_weight_difference_l1874_187448

-- Define the weights of the boxes following the given conditions.
def box1_weight : ℕ := 200
def box3_weight : ℕ := box1_weight + (25 * box1_weight / 100)
def box2_weight : ℕ := box3_weight + (20 * box3_weight / 100)
def box4_weight : ℕ := 350
def box5_weight : ℕ := box4_weight * 100 / 70

-- Define the average weight of the four heaviest boxes.
def avg_heaviest : ℕ := (box2_weight + box3_weight + box4_weight + box5_weight) / 4

-- Define the average weight of the four lightest boxes.
def avg_lightest : ℕ := (box1_weight + box2_weight + box3_weight + box4_weight) / 4

-- Define the difference between the average weights of the heaviest and lightest boxes.
def avg_difference : ℕ := avg_heaviest - avg_lightest

-- State the theorem with the expected result.
theorem avg_weight_difference : avg_difference = 75 :=
by
  -- Proof is not provided.
  sorry

end avg_weight_difference_l1874_187448


namespace part1_part2_l1874_187487

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Given conditions
axiom a_3a_5 : a 3 * a 5 = 63
axiom a_2a_6 : a 2 + a 6 = 16

-- Part (1) Proving the general formula
theorem part1 : 
  (∀ n : ℕ, a n = 12 - n) :=
sorry

-- Part (2) Proving the maximum value of S_n
theorem part2 :
  (∃ n : ℕ, (S n = (n * (12 - (n - 1) / 2)) → (n = 11 ∨ n = 12) ∧ (S n = 66))) :=
sorry

end part1_part2_l1874_187487


namespace commute_days_l1874_187458

-- Definitions of the variables
variables (a b c x : ℕ)

-- Given conditions
def condition1 : Prop := a + c = 12
def condition2 : Prop := b + c = 20
def condition3 : Prop := a + b = 14

-- The theorem to prove
theorem commute_days (h1 : condition1 a c) (h2 : condition2 b c) (h3 : condition3 a b) : a + b + c = 23 :=
sorry

end commute_days_l1874_187458


namespace volume_of_prism_l1874_187464

theorem volume_of_prism (l w h : ℝ) (hlw : l * w = 15) (hwh : w * h = 20) (hlh : l * h = 24) : l * w * h = 60 := 
sorry

end volume_of_prism_l1874_187464


namespace value_of_a_plus_c_l1874_187426

variables {a b c : ℝ}

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

def f_inv (x : ℝ) : ℝ := c * x^2 + b * x + a

theorem value_of_a_plus_c : a + c = -1 :=
sorry

end value_of_a_plus_c_l1874_187426


namespace find_number_l1874_187424

theorem find_number (x : ℕ) (h : 5 + 2 * (8 - x) = 15) : x = 3 :=
sorry

end find_number_l1874_187424


namespace exists_y_less_than_half_p_l1874_187456

theorem exists_y_less_than_half_p (p : ℕ) (hp : Nat.Prime p) (hp_gt3 : p > 3) :
  ∃ (y : ℕ), y < p / 2 ∧ ∀ (a b : ℕ), p * y + 1 ≠ a * b ∨ a ≤ y ∨ b ≤ y :=
by sorry

end exists_y_less_than_half_p_l1874_187456


namespace triangle_angle_contradiction_l1874_187478

theorem triangle_angle_contradiction (α β γ : ℝ) (h1 : α + β + γ = 180)
(h2 : α > 60) (h3 : β > 60) (h4 : γ > 60) : false :=
sorry

end triangle_angle_contradiction_l1874_187478


namespace sum_of_marked_angles_l1874_187488

theorem sum_of_marked_angles (sum_of_angles_around_vertex : ℕ := 360) 
    (vertices : ℕ := 7) (triangles : ℕ := 3) 
    (sum_of_interior_angles_triangle : ℕ := 180) :
    (vertices * sum_of_angles_around_vertex - triangles * sum_of_interior_angles_triangle) = 1980 :=
by
  sorry

end sum_of_marked_angles_l1874_187488


namespace initial_men_count_l1874_187490

-- Definitions based on problem conditions
def initial_days : ℝ := 18
def extra_men : ℝ := 400
def final_days : ℝ := 12.86

-- Proposition to show the initial number of men based on conditions
theorem initial_men_count (M : ℝ) (h : M * initial_days = (M + extra_men) * final_days) : M = 1000 := by
  sorry

end initial_men_count_l1874_187490
