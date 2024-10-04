import Mathlib
import Mathlib.*
import Mathlib.Algebra.BigOperators
import Mathlib.Analysis.Complex.Trigonometry
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Permutation
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Face
import Mathlib.Geometry.Tetrahedron
import Mathlib.Geometry.Triangle
import Mathlib.NumberTheory.Basic
import Mathlib.NumberTheory.GCD
import Mathlib.NumberTheory.ModularArithmetic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith
import Real
import Real.Basic

namespace min_students_in_class_l0_747

def minimum_possible_students (b g : ℕ) (boyspassing girlspassing : ℕ):
  minimum_possible_students = b + g := sorry

theorem min_students_in_class :
  ∃ (b g : ℕ), b + g = 11 ∧ (boyspassing = b / 2) ∧ (girlspassing = (2 * g) / 3) ∧ 2 * girlspassing = boyspassing := sorry

end min_students_in_class_l0_747


namespace PlayerA_has_winning_strategy_l0_248

-- Definitions to represent the conditions
structure Chessboard :=
  (size : ℕ)
  (total_positions : ℕ := size ^ 2)

-- Moves for Player A and Player B
inductive Move
| horizontal : Move
| vertical   : Move

-- Game state definition
structure GameState :=
  (board : Chessboard)
  (occupied_positions : Finset (Fin 1994 × Fin 1994))
  (next_move : Move)

-- Initial state for player A
def initialState : GameState :=
  ⟨⟨1994⟩, ∅, Move.horizontal⟩

-- Theorem statement for Player A having a winning strategy
theorem PlayerA_has_winning_strategy (gs : GameState) :
  ∃ strat : (Fin 1994 × Fin 1994) → (Fin 1994 × Fin 1994), 
  ∀ move : Move, move = Move.horizontal → (finspace : Finset (Fin 1994 × Fin 1994)) → 
  (strat gs.board.occupied_positions) ∉ gs.board.occupied_positions :=
sorry

end PlayerA_has_winning_strategy_l0_248


namespace new_average_weight_l0_947

theorem new_average_weight (original_average_weight : ℕ) (num_original_players : ℕ)
  (new_player1_weight : ℕ) (new_player2_weight : ℕ) :
  original_average_weight = 112 →
  num_original_players = 7 →
  new_player1_weight = 110 →
  new_player2_weight = 60 →
  (original_average_weight * num_original_players + new_player1_weight + new_player2_weight) / (num_original_players + 2) = 106 :=
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  exact sorry

end new_average_weight_l0_947


namespace sum_of_angles_l0_659

theorem sum_of_angles (k : ℝ) :
  (k = 1) →
  (∀ x ∈ set.Icc 0 360, 
    sin x ^ 3 - cos x ^ 3 = k * ((1 / cos x) - (1 / sin x))) →
  ∃ (angles : list ℝ), 
    (∀ x ∈ angles, x ∈ set.Icc 0 360 ∧ sin x ^ 3 - cos x ^ 3 = k * ((1 / cos x) - (1 / sin x))) ∧ 
    angles.sum = 270 :=
begin
  intros hk h,
  -- sorry, the proof should be placed here.
  sorry,
end

end sum_of_angles_l0_659


namespace cos_135_eq_neg_sqrt2_div_2_l0_485

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by 
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_485


namespace valid_3_word_sentences_count_l0_855
noncomputable def countValidSentences : ℕ :=
  let words := ["splargh", "glumph", "amr", "blargh"];
  let sentences := list.product (list.product words words) words;
  let validSentences := sentences.filter (λ s, 
      ¬ list.pairwise (λ x y => (x = "splargh" ∧ y = "glumph") ∨ 
                                  (x = "amr" ∧ y = "blargh")) s)
  in validSentences.length

theorem valid_3_word_sentences_count : countValidSentences = 48 :=
begin
  sorry
end

end valid_3_word_sentences_count_l0_855


namespace positive_difference_l0_907

theorem positive_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
sorry

end positive_difference_l0_907


namespace johns_total_due_l0_213

noncomputable def total_amount_due (initial_amount : ℝ) (first_charge_rate : ℝ) 
  (second_charge_rate : ℝ) (third_charge_rate : ℝ) : ℝ := 
  let after_first_charge := initial_amount * first_charge_rate
  let after_second_charge := after_first_charge * second_charge_rate
  let after_third_charge := after_second_charge * third_charge_rate
  after_third_charge

theorem johns_total_due : total_amount_due 500 1.02 1.03 1.025 = 538.43 := 
  by
    -- The proof would go here.
    sorry

end johns_total_due_l0_213


namespace positive_difference_of_numbers_l0_921

theorem positive_difference_of_numbers
  (a b : ℝ)
  (h1 : a + b = 10)
  (h2 : a^2 - b^2 = 40) :
  |a - b| = 4 :=
by 
  sorry

end positive_difference_of_numbers_l0_921


namespace minimum_value_of_function_l0_785

theorem minimum_value_of_function (x : ℝ) :
  let y := abs (x - 1) + abs (x - 2) - abs (x - 3)
  in ∃ (x0 : ℝ), (y x0 = -1) ∧ ∀ (x : ℝ), y x ≥ -1 :=
by sorry

end minimum_value_of_function_l0_785


namespace sum_series_correct_l0_660

-- Define the ranges and increments
def n_range := (1 : Finset ℕ)..(12 : Finset ℕ)
def m_range := (3 : Finset ℕ).bind (λ x, if x % 2 == 1 && x <= 8 then {x} else ∅)
def p_range := {2, 6}

-- Define the function for the series term
def series_term (n m p : ℕ) : ℕ := 2 * n + 3 * m + 4 * p

-- Define the sum of the series under given conditions
def sum_series : ℕ := 
  ∑ n in n_range, ∑ m in m_range, ∑ p in p_range, series_term n m p

-- The theorem statement
theorem sum_series_correct : sum_series = 257 :=
by sorry

end sum_series_correct_l0_660


namespace koshchey_total_chests_l0_217

-- Define the problem and conditions
def Koshchey := 11 -- Total number of large chests
def medium_chests_in_large := 8 -- Number of medium chests in a non-empty large chest
def total_empty_chests := 102 -- Total number of empty chests

-- Construct the theorem
theorem koshchey_total_chests : 
  ∀ (large_chests : ℕ) (medium_chests : ℕ) (empty_chests : ℕ), 
    large_chests = Koshchey →
    medium_chests = medium_chests_in_large →
    empty_chests = total_empty_chests →
    ∃ (x : ℕ), (large_chests + 7 * x = empty_chests) ∧ 
               (11 + 8 * x = 115) :=
by
  intros large_chests medium_chests empty_chests
  intro h1 h2 h3
  existsi 13 -- x is 13
  split
  sorry
  sorry

end koshchey_total_chests_l0_217


namespace cuberoot_inequality_l0_226

theorem cuberoot_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
    Real.cbrt (a * b) + Real.cbrt (c * d) ≤ Real.cbrt ((a + b + c) * (b + c + d)) := 
sorry

end cuberoot_inequality_l0_226


namespace ellis_family_water_bottles_l0_83

theorem ellis_family_water_bottles :
  (let num_people := 4
       bottle_per_hour_per_person := 1/2
       total_hours := 8 + 8 in
   num_people * bottle_per_hour_per_person * total_hours = 32) :=
by
  let num_people := 4
  let bottle_per_hour_per_person := 1/2
  let total_hours := 8 + 8
  show num_people * bottle_per_hour_per_person * total_hours = 32
  sorry

end ellis_family_water_bottles_l0_83


namespace cos_135_eq_neg_inv_sqrt_2_l0_518

theorem cos_135_eq_neg_inv_sqrt_2 : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_518


namespace cos_135_eq_neg_inv_sqrt_2_l0_574

theorem cos_135_eq_neg_inv_sqrt_2 :
  cos (135 * Real.pi / 180) = - (1 / Real.sqrt 2) :=
by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_574


namespace cos_135_eq_neg_sqrt_two_div_two_l0_452

theorem cos_135_eq_neg_sqrt_two_div_two : real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt_two_div_two_l0_452


namespace cos_135_eq_neg_sqrt2_div_2_l0_599

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  -- Conditions incorporated directly into the requirement due to the nature of trigonometric evaluations.
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_599


namespace problem_statement_l0_794

def prop_p (x : ℝ) : Prop := x^2 >= x
def prop_q : Prop := ∃ x : ℝ, x^2 >= x

theorem problem_statement : (∀ x : ℝ, prop_p x) = false ∧ prop_q = true :=
by 
  sorry

end problem_statement_l0_794


namespace find_N_sum_e_l0_172

theorem find_N_sum_e (N : ℝ) (e1 e2 : ℝ) :
  (2 * abs (2 - e1) = N) ∧
  (2 * abs (2 - e2) = N) ∧
  (e1 ≠ e2) ∧
  (e1 + e2 = 4) →
  N = 0 :=
by
  sorry

end find_N_sum_e_l0_172


namespace notebook_price_constant_and_quantity_variable_l0_386

-- Define the price of one notebook as a constant
def price_per_notebook : ℕ := 5

-- Define the total cost in yuan and the number of notebooks as variables
variables (x y : ℕ)

-- The proof statement
theorem notebook_price_constant_and_quantity_variable (h : y = 5 * x) :
  constant price_per_notebook ∧ variable x :=
by
  -- Proof will go here
  sorry

end notebook_price_constant_and_quantity_variable_l0_386


namespace checkout_speed_ratio_l0_418

theorem checkout_speed_ratio (n x y : ℝ) 
  (h1 : 40 * x = 20 * y + n)
  (h2 : 36 * x = 12 * y + n) : 
  x = 2 * y := 
sorry

end checkout_speed_ratio_l0_418


namespace parking_lot_problem_l0_266

variable (M S : Nat)

theorem parking_lot_problem (h1 : M + S = 30) (h2 : 15 * M + 8 * S = 324) :
  M = 12 ∧ S = 18 :=
by
  -- proof omitted
  sorry

end parking_lot_problem_l0_266


namespace choir_average_age_l0_857

theorem choir_average_age 
  (avg_f : ℝ) (n_f : ℕ)
  (avg_m : ℝ) (n_m : ℕ)
  (h_f : avg_f = 28) 
  (h_nf : n_f = 12) 
  (h_m : avg_m = 40) 
  (h_nm : n_m = 18) 
  : (n_f * avg_f + n_m * avg_m) / (n_f + n_m) = 35.2 := 
by 
  sorry

end choir_average_age_l0_857


namespace ball_color_arrangement_l0_830

-- Definitions for the conditions
variable (balls_in_red_box balls_in_white_box balls_in_yellow_box : Nat)
variable (red_balls white_balls yellow_balls : Nat)

-- Conditions as assumptions
axiom more_balls_in_yellow_box_than_yellow_balls : balls_in_yellow_box > yellow_balls
axiom different_balls_in_red_box_than_white_balls : balls_in_red_box ≠ white_balls
axiom fewer_white_balls_than_balls_in_white_box : white_balls < balls_in_white_box

-- The main theorem to prove
theorem ball_color_arrangement
  (more_balls_in_yellow_box_than_yellow_balls : balls_in_yellow_box > yellow_balls)
  (different_balls_in_red_box_than_white_balls : balls_in_red_box ≠ white_balls)
  (fewer_white_balls_than_balls_in_white_box : white_balls < balls_in_white_box) :
  (balls_in_red_box, balls_in_white_box, balls_in_yellow_box) = (yellow_balls, red_balls, white_balls) :=
sorry

end ball_color_arrangement_l0_830


namespace partial_deriv_f_alpha_eval_partial_deriv_f_beta_eval_partial_deriv_z_x_eval_partial_deriv_z_y_eval_l0_56

noncomputable def partial_deriv_f_alpha : ℝ := sorry

theorem partial_deriv_f_alpha_eval (m n : ℝ) :
  partial_deriv_f_alpha (cos (m * (π / (2 * m)) - n * 0)) α (π / (2 * m)) β 0 = -m :=
by sorry

noncomputable def partial_deriv_f_beta : ℝ := sorry

theorem partial_deriv_f_beta_eval (m n : ℝ) :
  partial_deriv_f_beta (cos (m * (π / (2 * m)) - n * 0)) α (π / (2 * m)) β 0 = n :=
by sorry

noncomputable def partial_deriv_z_x : ℝ := sorry

theorem partial_deriv_z_x_eval (x y : ℝ) :
  partial_deriv_z_x (ln (x^2 - y^2)) x 2 y (-1) = 4/3 :=
by sorry

noncomputable def partial_deriv_z_y : ℝ := sorry

theorem partial_deriv_z_y_eval (x y : ℝ) :
  partial_deriv_z_y (ln (x^2 - y^2)) x 2 y (-1) = 2/3 :=
by sorry

end partial_deriv_f_alpha_eval_partial_deriv_f_beta_eval_partial_deriv_z_x_eval_partial_deriv_z_y_eval_l0_56


namespace cos_135_eq_neg_sqrt2_div_2_l0_616

theorem cos_135_eq_neg_sqrt2_div_2 : 
  Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_616


namespace equilateral_triangle_points_l0_85

theorem equilateral_triangle_points (n : ℕ) (h : 2 < n) :
  (∃ (points : set (ℝ × ℝ)), points.card = n ∧
    (∀ p1 ∈ points, ∀ p2 ∈ points, p1 ≠ p2 →
      (∃ p3 ∈ points, p1 ≠ p3 ∧ p2 ≠ p3 ∧
       dist p1 p2 = dist p1 p3 ∧ dist p1 p3 = dist p2 p3))) ↔ n = 3 :=
sorry

end equilateral_triangle_points_l0_85


namespace option2_is_cheaper_l0_398

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def final_price_option1 (initial_price : ℝ) : ℝ :=
  let price_after_first_discount := apply_discount initial_price 0.30
  let price_after_second_discount := apply_discount price_after_first_discount 0.10
  apply_discount price_after_second_discount 0.05

def final_price_option2 (initial_price : ℝ) : ℝ :=
  let price_after_first_discount := apply_discount initial_price 0.30
  let price_after_second_discount := apply_discount price_after_first_discount 0.05
  apply_discount price_after_second_discount 0.15

theorem option2_is_cheaper (initial_price : ℝ) (h : initial_price = 12000) :
  final_price_option2 initial_price = 6783 ∧ final_price_option1 initial_price = 7182 → 6783 < 7182 :=
by
  intros
  sorry

end option2_is_cheaper_l0_398


namespace car_payment_percentage_l0_52

def car_payment : ℝ := 400
def tax_fraction : ℝ := 1/3
def gross_income : ℝ := 3000

def after_tax_income : ℝ := gross_income * (1 - tax_fraction)

def percentage_spent_on_car (c : ℝ) (ati : ℝ) : ℝ := (c / ati) * 100

theorem car_payment_percentage :
  percentage_spent_on_car car_payment after_tax_income = 20 := by
  sorry

end car_payment_percentage_l0_52


namespace figure_enclosed_in_hexagon_figure_enclosed_in_octagon_l0_3

-- Problem 1 (a): Define the necessary structures and theorem
structure Square (Q : Type) :=
(a1 a2 a3 a4 : Q)
(side_length : ℝ)

structure Circle (k : Type) :=
(inscribed_in : Square k)
(diameter : ℝ)

structure Triangle (Δ : Type) :=
(tangent_to : Circle Δ)
(perpendicular_to : fin 4 → Δ)

structure Hexagon (U : Type) :=
(formed_from : Square U)
(cut_triangles : Triangle U)

theorem figure_enclosed_in_hexagon {U Q k Δ : Type} 
  (Q_square: Square Q)
  (circle_k: Circle k)
  (triangles: fin 2 → Triangle Δ)
  (hexagon_U: Hexagon U) :
  (∀ (figure : Type), diameter figure = 1 → enclosed_in figure U) :=
  sorry

-- Problem 1 (b): Define the necessary structures and theorem
structure RegularHexagon (P : Type) :=
(a1 a2 a3 a4 a5 a6 : P)
(side_length : ℝ)

structure IrregularOctagon (V : Type) :=
(formed_from : RegularHexagon V)
(cut_triangles : Triangle V)

theorem figure_enclosed_in_octagon {V P k Δ : Type}
  (P_hexagon: RegularHexagon P)
  (circle_k: Circle k)
  (triangles: fin 2 → Triangle Δ)
  (octagon_V: IrregularOctagon V) :
  (∀ (figure : Type), diameter figure = 1 → enclosed_in figure V) :=
  sorry

end figure_enclosed_in_hexagon_figure_enclosed_in_octagon_l0_3


namespace probability_of_black_square_at_2017_moves_l0_838

noncomputable def coordinate_sum (moves : Nat → Nat) (n : Nat) : Nat :=
  -- Define the sum of coordinates after N moves based on the given movement probability
  n - moves n

theorem probability_of_black_square_at_2017_moves :
  let p_black := 1 / 3
  -- Ryan starts on a black square (0, 0)
  let initial_square := (0, 0)
  -- Coloring based on (x + y) % 3
  ∀ (x y : Nat), (x + y) % 3 = 0 ↔ black (x, y)
  -- Ryan makes 2017 moves with given probability
  -- Movement rules:
  -- Moving up or right increases the coordinate sum by 1.
  -- Moving diagonally up-right increases the coordinate sum by 2.
  ∀(n : Nat), 0 ≤ n ∧ n ≤ 2017 →
  (coordinate_sum n 2017) % 3 = 0 → 
  -- The probability of ending on a black square is 1/3
  p_black = 1 / 3 :=
sorry

end probability_of_black_square_at_2017_moves_l0_838


namespace cos_135_degree_l0_494

theorem cos_135_degree : 
  let Q := (Complex.exp (135 * Real.pi / 180 * Complex.I)) in 
  Complex.re Q = -Real.sqrt 2 / 2 :=
by sorry

end cos_135_degree_l0_494


namespace chess_tournament_l0_191

-- Define the main theorem statement
theorem chess_tournament {n : ℕ} (G : Fin n → Fin n → Bool) (h : ∀ i j, i ≠ j → G i j ∨ G j i) :
  ∃ (f : Fin n → Fin n), ∀ i, i.val < n - 1 → ¬G (f i) (f (i + 1)) :=
by
  sorry

end chess_tournament_l0_191


namespace cos_135_eq_neg_sqrt2_div_2_l0_486

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by 
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_486


namespace find_largest_p_plus_q_l0_953

noncomputable def largest_possible_p_plus_q : ℝ :=
  let B := (12 : ℝ, 19 : ℝ)
  let C := (23 : ℝ, 20 : ℝ)
  let area_ABC : ℝ := 70
  let slope_median : ℝ := -5 in
  47

theorem find_largest_p_plus_q (p q : ℝ) (hA : (p, q) = A) (h_area : 2 * area_ABC = abs (p * (B.2 - C.2) + B.1 * (C.2 - q) + C.1 * (q - B.2)))
(h_slope : q = -5 * p + 107) : p + q = largest_possible_p_plus_q := 
  sorry

end find_largest_p_plus_q_l0_953


namespace max_natural_S_n_pos_l0_679

-- Defining the arithmetic sequence and its properties
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Conditions given in the problem
variables (a : ℕ → ℤ)
variable (h_seq : arithmetic_sequence a)
variable (h_a1 : a 1 > 0)
variable (h_a3_a10 : a 3 + a 10 > 0)
variable (h_a6_a7 : a 6 * a 7 < 0)

-- Sum of the first n terms
def S (n : ℕ) : ℤ := ∑ i in finset.range n, a (i + 1)

-- The theorem we need to prove
theorem max_natural_S_n_pos : 
  ∃ n : ℕ, (∀ m : ℕ, m > n → S a m ≤ 0) ∧ (∀ m : ℕ, m ≤ n → S a m > 0) :=
begin
  -- Proof is omitted
  sorry
end

end max_natural_S_n_pos_l0_679


namespace probability_of_odd_numbers_exactly_five_times_in_seven_rolls_l0_330

theorem probability_of_odd_numbers_exactly_five_times_in_seven_rolls :
  (nat.choose 7 5 * (1/2)^5 * (1/2)^2) = (21 / 128) := by
  sorry

end probability_of_odd_numbers_exactly_five_times_in_seven_rolls_l0_330


namespace problem_equivalent_proof_l0_42

theorem problem_equivalent_proof (A B : Set) (P : Bool) :
  -- Conditions:
  -- 1) The necessary condition for A ∩ B = A to hold is A ⊊ B.
  ((A ∩ B = A) → (A ⊆ B)) ∧ ¬((A ∩ B = A) → (A ⊂ B)) → false
  ∧
  -- 2) The negation of "If x^2 + y^2 = 0, then both x and y are 0".
  (∀ (x y : Real), (¬(x^2 + y^2 = 0) → (x ≠ 0 ∨ y ≠ 0)))
  ∧
  -- 3) The converse of "Congruent triangles are similar triangles".
  (∀ (T1 T2 : Triangle), T1 ≅ T2 ↔ T1 ∼ T2 → false)
  ∧
  -- 4) The contrapositive of "The opposite angles of a cyclic quadrilateral are supplementary".
  (∀ (Q : Quadrilateral), (cyclic Q → angles_supplementary (opposite_angles Q)))
  →
  -- Answer:
  (P = ② ∨ P = ④) :=
begin
  intros,
  sorry,
end

end problem_equivalent_proof_l0_42


namespace refined_poincare_inequality_l0_787

variable (f : ℝ → ℝ) (X : ℝ)
variable [is_normal X 0 1]
variable [SmoothFunction f]
variable [FiniteExpectation (fun x => f x ^ 2)]

-- Define the expectation E and variance D
noncomputable def E (x : ℝ → ℝ) := sorry
noncomputable def D (x : ℝ → ℝ) := sorry

-- Statement of the refined inequality
theorem refined_poincare_inequality :
  D (fun x => f x) X ≤ (E (fun x => (f' x) ^ 2) X + abs (E (fun x => f' x) X) ^ 2) / 2 :=
sorry

end refined_poincare_inequality_l0_787


namespace cos_135_eq_neg_inv_sqrt_2_l0_581

theorem cos_135_eq_neg_inv_sqrt_2 :
  cos (135 * Real.pi / 180) = - (1 / Real.sqrt 2) :=
by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_581


namespace cos_135_eq_neg_inv_sqrt_2_l0_575

theorem cos_135_eq_neg_inv_sqrt_2 :
  cos (135 * Real.pi / 180) = - (1 / Real.sqrt 2) :=
by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_575


namespace solution_fraction_l0_992

-- Conditions and definition of x
def initial_quantity : ℝ := 1
def concentration_70 : ℝ := 0.70
def concentration_25 : ℝ := 0.25
def concentration_new : ℝ := 0.35

-- Definition of the fraction of the solution replaced
def x (fraction : ℝ) : Prop :=
  concentration_70 * initial_quantity - concentration_70 * fraction + concentration_25 * fraction = concentration_new * initial_quantity

-- The theorem we need to prove
theorem solution_fraction : ∃ (fraction : ℝ), x fraction ∧ fraction = 7 / 9 :=
by
  use 7 / 9
  simp [x]
  sorry  -- Proof steps would be filled here

end solution_fraction_l0_992


namespace pencils_more_than_pens_l0_295

theorem pencils_more_than_pens 
  (pencils pens : ℕ) 
  (ratio_pens_to_pencils : 5 * pencils = 6 * pens) 
  (total_pencils : 54) : 
  pencils > pens := 
begin
  have group_count_lemma : 54 / 6 = 9 := by sorry,
  have total_pens : 9 * 5 = 45 := by sorry,
  show 54 - 45 = 9 := by sorry,
end

end pencils_more_than_pens_l0_295


namespace sector_area_l0_186

theorem sector_area
  (r : ℝ) (s : ℝ) (h_r : r = 1) (h_s : s = 1) : 
  (1 / 2) * r * s = 1 / 2 := by
  sorry

end sector_area_l0_186


namespace units_digit_of_k_l0_173

theorem units_digit_of_k
  (k : ℤ) (h1 : 1 < k)
  (α : ℂ) (h2 : α ^ 2 - k * α + 1 = 0)
  (h3 : ∀ n : ℕ, 10 < n → (α ^ (2 ^ n) + α ^ (- (2 ^ n))) % 10 = 7) :
  k % 10 = 3 :=
sorry

end units_digit_of_k_l0_173


namespace positive_difference_l0_910

theorem positive_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
sorry

end positive_difference_l0_910


namespace solution_set_of_inequality_l0_302

theorem solution_set_of_inequality : {x : ℝ | x^2 + x - 2 < 0} = {x : ℝ | -2 < x ∧ x < 1} :=
by
  sorry

end solution_set_of_inequality_l0_302


namespace cos_135_eq_correct_l0_601

noncomputable def cos_135_eq: Prop :=
  real.cos ((135 * real.pi) / 180) = -real.sqrt 2 / 2

theorem cos_135_eq_correct : cos_135_eq :=
  sorry

end cos_135_eq_correct_l0_601


namespace ana_trip_additional_miles_l0_411

theorem ana_trip_additional_miles :
  ∀ (D1 D2 : ℕ) (S1 S2 AvgSp : ℕ),
    D1 = 20 →
    S1 = 40 →
    AvgSp = 55 →
    S2 = 60 →
    D2 = 90 →
    let time1 := D1 / S1 in
    let time2 := D2 / S2 in
    let total_time := time1 + time2 in
    let total_distance := D1 + D2 in
    total_distance / total_time = AvgSp := 
by
  intros,
  sorry

end ana_trip_additional_miles_l0_411


namespace probability_of_at_least_9_points_l0_885

namespace ShooterProbability

def P (event: String) : ℝ :=
  if event = "A" then 0.4
  else if event = "B" then 0.3
  else if event = "C" then 0.3
  else 0

theorem probability_of_at_least_9_points :
  P("A") + P("B") = 0.7 :=
by
  sorry

end ShooterProbability

end probability_of_at_least_9_points_l0_885


namespace green_balls_to_remove_l0_368

theorem green_balls_to_remove : 
  ∀ (total_balls redPercentage targetPercentage : ℕ),
  total_balls = 150 → 
  redPercentage = 40 → 
  targetPercentage = 80 →
  let red_balls := (redPercentage * total_balls) / 100 in
  let green_balls_initial := total_balls - red_balls in
  let target_red_balls := (red_balls * 100) / targetPercentage in
  total_balls - green_balls_initial - target_red_balls = 75 :=
by
  intros total_balls redPercentage targetPercentage
  intro h1 h2 h3
  let red_balls := (redPercentage * total_balls) / 100
  let green_balls_initial := total_balls - red_balls
  let target_red_balls := (red_balls * 100) / targetPercentage
  have red_balls_calc : red_balls = 60 := by sorry -- from 0.40 * 150
  have green_balls_calc : green_balls_initial = 90 := by sorry -- from 150 - 60
  have target_red_balls_calc : target_red_balls = 75 := by sorry -- from (60 * 100) / 80
  have target_balls := total_balls - target_red_balls -- total_balls - 75
  exact target_ballsโช riesoroiry.calc

end green_balls_to_remove_l0_368


namespace positive_difference_of_two_numbers_l0_900

theorem positive_difference_of_two_numbers 
  (a b : ℝ) 
  (h₁ : a + b = 10) 
  (h₂ : a^2 - b^2 = 40) : 
  |a - b| = 4 := 
sorry

end positive_difference_of_two_numbers_l0_900


namespace sum_of_digits_next_l0_223

def sum_of_digits (n : ℕ) : ℕ :=
  (Nat.digits 10 n).sum

theorem sum_of_digits_next (n : ℕ) (h : sum_of_digits n = 1399) : 
  sum_of_digits (n + 1) = 1402 :=
sorry

end sum_of_digits_next_l0_223


namespace grapefruit_orchards_l0_999

theorem grapefruit_orchards (total_orchards lemons_orchards oranges_factor remaining_orchards : ℕ) 
    (H1 : total_orchards = 16)
    (H2 : lemons_orchards = 8)
    (H3 : oranges_factor = 2)
    (H4 : oranges_orchards = lemons_orchards / oranges_factor)
    (H5 : remaining_orchards = total_orchards - lemons_orchards - oranges_orchards)
    (H6 : grapefruit_orchards = remaining_orchards / 2) : 
  grapefruit_orchards = 2 := by
  sorry

end grapefruit_orchards_l0_999


namespace parity_of_f_and_h_l0_152

-- Define function f
def f (x : ℝ) : ℝ := x^2

-- Define function h
def h (x : ℝ) : ℝ := x

-- Define even and odd function
def even_fun (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x
def odd_fun (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = - g x

-- Theorem statement
theorem parity_of_f_and_h :
  even_fun f ∧ odd_fun h :=
by {
  sorry
}

end parity_of_f_and_h_l0_152


namespace probability_divisor_twenty_l0_11

/-
Given a circular spinner divided into 20 equal sections and an arrow that is spun once,
prove that the probability that the arrow stops in a section containing a number
that is a divisor of 20 is \(\frac{3}{10}\).
-/

theorem probability_divisor_twenty : 
  let total_sections : ℕ := 20 in
  let divisors_of_20 := {1, 2, 4, 5, 10, 20}.toFinset in
  (divisors_of_20.card : ℚ) / total_sections = 3 / 10 :=
by
  sorry

end probability_divisor_twenty_l0_11


namespace cos_135_eq_neg_sqrt_two_div_two_l0_454

theorem cos_135_eq_neg_sqrt_two_div_two : real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt_two_div_two_l0_454


namespace center_of_symmetry_g_l0_275

-- Define the original function
def f (x : ℝ) : ℝ := √3 * Real.sin x + Real.cos x

-- Define the transformation: shift left by π/12 and then shorten x-coordinates to half
def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x + π / 4)

-- Prove that the new function g(x) has a center of symmetry at the point (-π / 8, 0)
theorem center_of_symmetry_g : g (-π / 8) = 0 :=
by
  -- This would be where the proof steps go if they were required
  sorry

end center_of_symmetry_g_l0_275


namespace problem1_tangent_line_at_a_eq_2_problem2_f_1_div_a_leq_0_problem3_unique_zero_a_eq_1_l0_237

-- Given function f
def f (a : ℝ) (x : ℝ) := real.log x - a * x^2 + a * x

-- Problem 1: Tangent line when a = 2
theorem problem1_tangent_line_at_a_eq_2 :
  let a := 2 in ∀ x : ℝ, (x, f a x) = (1, f a 1) → x + (f a 1) - 1 = 0 :=
by
    let a := 2
    intro x
    intro h
    sorry

-- Problem 2: Prove that f(1/a) <= 0 for a > 0
theorem problem2_f_1_div_a_leq_0 (a : ℝ) (ha : 0 < a) :
  f a (1 / a) ≤ 0 :=
by
    sorry

-- Problem 3: If f(x) has exactly one zero, then a = 1
theorem problem3_unique_zero_a_eq_1 (a : ℝ) :
  (∃! x : ℝ, f a x = 0) ↔ a = 1 :=
by
    sorry

end problem1_tangent_line_at_a_eq_2_problem2_f_1_div_a_leq_0_problem3_unique_zero_a_eq_1_l0_237


namespace largest_divisor_of_q_l0_409

theorem largest_divisor_of_q (die_faces : Finset ℕ) (h : die_faces = Finset.range 1 9) (Q : ℕ) 
    (hQ : Q = ∏ x in (die_faces \ {missing_die}), x) : ∃ d, d = 192 ∧ d ∣ Q :=
by
  have fact8 : 8! = 40320 := by norm_num
  have prime_factors : Nat.prime_factors 40320 = [2, 2, 2, 2, 2, 2, 2, 3, 3, 5, 7] := by norm_num
  sorry

end largest_divisor_of_q_l0_409


namespace sum_of_roots_of_quadratic_l0_728

open Polynomial

theorem sum_of_roots_of_quadratic :
  ∀ (m n : ℝ), (m ≠ n ∧ (∀ x, x^2 + 2*x - 1 = 0 → x = m ∨ x = n)) → m + n = -2 :=
by
  sorry

end sum_of_roots_of_quadratic_l0_728


namespace number_of_real_solutions_eq_63_l0_658

theorem number_of_real_solutions_eq_63 :
  (setOf (λ x : ℝ, x / 100 = sin x ∧ -100 ≤ x ∧ x ≤ 100)).count = 63 := 
sorry

end number_of_real_solutions_eq_63_l0_658


namespace positive_difference_of_two_numbers_l0_936

theorem positive_difference_of_two_numbers (x y : ℝ)
  (h1 : x + y = 10) (h2 : x ^ 2 - y ^ 2 = 40) : abs (x - y) = 4 := 
by
  sorry

end positive_difference_of_two_numbers_l0_936


namespace melissa_total_score_l0_806

theorem melissa_total_score (games : ℕ) (points_per_game : ℕ) 
  (h_games : games = 3) (h_points_per_game : points_per_game = 27) : 
  points_per_game * games = 81 := 
by 
  sorry

end melissa_total_score_l0_806


namespace pond_length_is_9_l0_872

-- Define the length and width of the field
variables (l w : ℕ) (l_eq : l = 2 * w) (l_val : l = 36)

-- Define the area of the field and the pond
noncomputable def A_field := l * w
noncomputable def A_pond := A_field / 8

-- Define the length of the side of the pond
noncomputable def s := Nat.sqrt A_pond

-- State the theorem to prove
theorem pond_length_is_9 : s = 9 :=
by 
  -- Use the conditions and prove the statement
  sorry

end pond_length_is_9_l0_872


namespace number_of_obtuse_triangles_in_120_gon_l0_313

-- Definitions based on conditions
def regular_polygon (n : Nat) := true

def vertices (n : Nat) := List.range n

def selecting_three_vertices_makes_obtuse_triangle (n : Nat) (k l m : Nat) :=
  k < l ∧ l < m ∧ m - k < 60

-- Theorem statement based on the question and correct answer
theorem number_of_obtuse_triangles_in_120_gon :
  let n := 120
  ∑ (k : Fin n) (l : Fin n) (m : Fin n),
  if selecting_three_vertices_makes_obtuse_triangle n k l m then 1 else 0 = 205320 := 
  by
    sorry

end number_of_obtuse_triangles_in_120_gon_l0_313


namespace maria_younger_than_ann_l0_801

variable (M A : ℕ)

def maria_current_age : Prop := M = 7

def age_relation_four_years_ago : Prop := M - 4 = (1 / 2) * (A - 4)

theorem maria_younger_than_ann :
  maria_current_age M → age_relation_four_years_ago M A → A - M = 3 :=
by
  sorry

end maria_younger_than_ann_l0_801


namespace number_of_even_integers_satisfying_conditions_l0_724

open Int

def satisfies_conditions (n : Int) : Prop :=
  (n + 5) * (n - 9) ≤ 0

def count_even_satisfying_conditions : Int :=
  List.length (
    List.filter 
      (λ x => (satisfies_conditions x) && (x % 2 = 0))
      (List.range' (-5) (15)) -- list of integers from -5 to 9 inclusive
  )

theorem number_of_even_integers_satisfying_conditions :
  count_even_satisfying_conditions = 7 :=
by
  sorry

end number_of_even_integers_satisfying_conditions_l0_724


namespace cos_135_eq_correct_l0_606

noncomputable def cos_135_eq: Prop :=
  real.cos ((135 * real.pi) / 180) = -real.sqrt 2 / 2

theorem cos_135_eq_correct : cos_135_eq :=
  sorry

end cos_135_eq_correct_l0_606


namespace sqrt_inequality_l0_62

theorem sqrt_inequality :
  real.sqrt 11 - real.sqrt 12 < real.sqrt 12 - real.sqrt 13 :=
sorry

end sqrt_inequality_l0_62


namespace cubical_cake_l0_12

noncomputable def cubical_cake_properties : Prop :=
  let a : ℝ := 3
  let top_area := (1 / 2) * 3 * 1.5
  let height := 3
  let volume := top_area * height
  let vertical_triangles_area := 2 * ((1 / 2) * 1.5 * 3)
  let vertical_rectangular_area := 3 * 3
  let iced_area := top_area + vertical_triangles_area + vertical_rectangular_area
  volume + iced_area = 22.5

theorem cubical_cake : cubical_cake_properties := sorry

end cubical_cake_l0_12


namespace matrix_non_invertible_fraction_sum_eq_one_l0_100

theorem matrix_non_invertible_fraction_sum_eq_one
  (x y z : ℝ)
  (h : ¬ (det (matrix.of ![
                          ![x, x + y, x + z],
                          ![x + y, y, y + z],
                          ![x + z, y + z, z]
                         ]) ≠ 0)) :
  (x + y + z ≠ 0) -> 
  (x / (x + y + z) + y / (x + y + z) + z / (x + y + z) = 1) :=
  sorry

end matrix_non_invertible_fraction_sum_eq_one_l0_100


namespace vectors_not_coplanar_l0_412

def a : ℝ × ℝ × ℝ := (4, 1, 1)
def b : ℝ × ℝ × ℝ := (-9, -4, -9)
def c : ℝ × ℝ × ℝ := (6, 2, 6)

def scalarTripleProduct (u v w : ℝ × ℝ × ℝ) : ℝ :=
  let (u1, u2, u3) := u
  let (v1, v2, v3) := v
  let (w1, w2, w3) := w
  u1 * (v2 * w3 - v3 * w2) - u2 * (v1 * w3 - v3 * w1) + u3 * (v1 * w2 - v2 * w1)

theorem vectors_not_coplanar : scalarTripleProduct a b c = -18 := by
  sorry

end vectors_not_coplanar_l0_412


namespace sum_remainder_982_l0_780

def distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  list.nodup digits

theorem sum_remainder_982 :
  let S := ∑ i in finset.filter (λ n => n ≥ 1000 ∧ n ≤ 2019 ∧ distinct_digits n) (finset.range 2020), i
  S % 1000 = 982 :=
by 
  sorry

end sum_remainder_982_l0_780


namespace fence_cost_correct_l0_888

noncomputable def solve_fence_cost (length_ratio : ℝ) (width_ratio : ℝ) (area : ℝ)
    (cost_type_A : ℝ) (fraction_type_A : ℝ) (cost_type_B : ℝ) (fraction_type_B : ℝ) : ℝ :=
  let x := real.sqrt (area / (length_ratio * width_ratio))
  let length := length_ratio * x
  let width := width_ratio * x
  let perimeter := 2 * (length + width)
  let cost_A := cost_type_A * fraction_type_A * perimeter
  let cost_B := cost_type_B * fraction_type_B * perimeter
  cost_A + cost_B

theorem fence_cost_correct :
  solve_fence_cost 5 3 8220 0.80 (3/5) 1.20 (2/5) = 359.42 :=
by
  sorry

end fence_cost_correct_l0_888


namespace find_k_l0_384

variable {V : Type} [AddCommGroup V] [Module ℝ V]
variable (u v : V)

theorem find_k (h : ∃ t : ℝ, t ≠ 0 ∧ t ≠ 1 ∧ u + t • (v - u) = k • u + (5 / 8) • v) :
  k = 3 / 8 := sorry

end find_k_l0_384


namespace cos_135_eq_neg_inv_sqrt2_l0_434

theorem cos_135_eq_neg_inv_sqrt2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt2_l0_434


namespace yellow_paint_amount_l0_799

theorem yellow_paint_amount (b y : ℕ) (h_ratio : y * 7 = 3 * b) (h_blue_amount : b = 21) : y = 9 :=
by
  sorry

end yellow_paint_amount_l0_799


namespace perpendicular_condition_l0_123

/-- Two lines l1 and l2 are defined as follows:
     l1 : mx + y - 1 = 0
     l2 : (m - 2)x + my - 1 = 0
    Prove that m = 1 is a sufficient but not necessary condition for l1 to be perpendicular to l2 -/
theorem perpendicular_condition (m : ℝ) :
  (m = 1) → is_perpendicular l1 l2 ∧ ∃ m', m' ≠ 1 ∧ is_perpendicular (line1 m') (line2 m') :=
sorry

def line1 (m : ℝ) : ℝ × ℝ → ℝ
  | (x, y) => m * x + y - 1

def line2 (m : ℝ) : ℝ × ℝ → ℝ
  | (x, y) => (m - 2) * x + m * y - 1

def is_perpendicular (l1 l2 : ℝ × ℝ → ℝ) : Prop :=
  ∀ (x1 y1 x2 y2 : ℝ), l1 (x1, y1) = 0 → l2 (x2, y2) = 0 →
    ((x1 - x2) * (y1 - y2) = -1)

end perpendicular_condition_l0_123


namespace classify_books_ways_l0_259

def num_classify_ways (books : Finset ℕ) (intersection : Finset ℕ): ℕ :=
  if books.card = 6 ∧ intersection.card = 3 ∧ intersection ⊆ books then
  270 else 0

theorem classify_books_ways (books : Finset ℕ) (intersection : Finset ℕ)
  (h₁ : books.card = 6) (h₂ : intersection.card = 3) (h₃ : intersection ⊆ books) :
  num_classify_ways books intersection = 270 :=
by 
  rw num_classify_ways 
  split_ifs 
  . { refl }
  . { exfalso, exact h.ne (ne_of_eq_of_ne h₁ (ne_of_eq_of_not h₂ rfl h₃))}

end classify_books_ways_l0_259


namespace time_difference_between_A_and_B_is_9_value_of_x_for_participant_C_l0_748

noncomputable def participant_A_time := 36
noncomputable def participant_B_time := 45

noncomputable def participant_A_handicap := 10
noncomputable def participant_B_handicap := 5
noncomputable def participant_C_handicap := 0

noncomputable def total_distance := 110

def participant_time_difference : ℕ := participant_B_time - participant_A_time

def participant_c_time (x : ℝ) : Prop :=
  let speed_A := (total_distance + participant_A_handicap : ℝ) / participant_A_time
  let speed_B := (total_distance + participant_B_handicap : ℝ) / participant_B_time
  let distance_A_in_B_time := speed_A * participant_B_time
  let halfway_distance := (distance_A_in_B_time + (total_distance + participant_B_handicap)) / 2
  let actual_halfway_point := halfway_distance - participant_A_handicap
  x = actual_halfway_point / speed_A

theorem time_difference_between_A_and_B_is_9 : participant_time_difference = 9 := 
by 
  unfold participant_time_difference
  exact Nat.sub_self 36

theorem value_of_x_for_participant_C (x : ℝ) : participant_c_time x → x = 36.75 := 
by
  intro h
  sorry

end time_difference_between_A_and_B_is_9_value_of_x_for_participant_C_l0_748


namespace smallest_number_divisible_l0_961

   theorem smallest_number_divisible (d n : ℕ) (h₁ : (n + 7) % 11 = 0) (h₂ : (n + 7) % 24 = 0) (h₃ : (n + 7) % d = 0) (h₄ : (n + 7) = 257) : n = 250 :=
   by
     sorry
   
end smallest_number_divisible_l0_961


namespace denote_below_warning_level_l0_886

-- Conditions
def warning_water_level : ℝ := 905.7
def exceed_by_10 : ℝ := 10
def below_by_5 : ℝ := -5

-- Problem statement
theorem denote_below_warning_level : below_by_5 = -5 := 
by
  sorry

end denote_below_warning_level_l0_886


namespace squirrel_travel_time_l0_27

theorem squirrel_travel_time :
  ∀ (speed distance : ℝ), speed = 5 → distance = 3 →
  (distance / speed) * 60 = 36 := by
  intros speed distance h_speed h_distance
  rw [h_speed, h_distance]
  norm_num

end squirrel_travel_time_l0_27


namespace bisector_plane_division_l0_827

def bisector_plane_divides (tetra : Tetrahedron) (AB : Edge) (E : Point) : Prop :=
  let ABC := face_of tetra AB
  let ABD := opposite_face_of tetra ABC
  let CD := opposite_edge_of tetra AB
  E ∈ CD ∧
  ∃ CE ED : ℝ,
    CE / ED = area_of_face ABC / area_of_face ABD

theorem bisector_plane_division (tetra : Tetrahedron) (AB : Edge) (E : Point) (faces : face_of tetra AB) :
  bisector_plane_divides tetra AB E :=
by
  sorry

end bisector_plane_division_l0_827


namespace projection_identity_l0_683

variables (P : ℝ × ℝ × ℝ) (x1 y1 z1 x2 y2 z2 x3 y3 z3 : ℝ)

-- Define point P as (-1, 3, -4)
def point_P := (-1, 3, -4) = P

-- Define projections on the coordinate planes
def projection_yoz := (x1, y1, z1) = (0, 3, -4)
def projection_zox := (x2, y2, z2) = (-1, 0, -4)
def projection_xoy := (x3, y3, z3) = (-1, 3, 0)

-- Prove that x1^2 + y2^2 + z3^2 = 0 under the given conditions
theorem projection_identity :
  point_P P ∧ projection_yoz x1 y1 z1 ∧ projection_zox x2 y2 z2 ∧ projection_xoy x3 y3 z3 →
  (x1^2 + y2^2 + z3^2 = 0) :=
by
  sorry

end projection_identity_l0_683


namespace card_probability_multiple_3_4_or_7_l0_38

theorem card_probability_multiple_3_4_or_7 :
  let cards := Finset.range 121 
  let multiples_of_3 := cards.filter (λ x, x % 3 = 0)
  let multiples_of_4 := cards.filter (λ x, x % 4 = 0)
  let multiples_of_7 := cards.filter (λ x, x % 7 = 0)
  let multiples_of_12 := cards.filter (λ x, x % 12 = 0)
  let multiples_of_21 := cards.filter (λ x, x % 21 = 0)
  let multiples_of_28 := cards.filter (λ x, x % 28 = 0)
  let multiples_of_84 := cards.filter (λ x, x % 84 = 0)
  
  let total_multiples := multiples_of_3.card + multiples_of_4.card + multiples_of_7.card -
                        multiples_of_12.card - multiples_of_21.card - multiples_of_28.card +
                        multiples_of_84.card in
  total_multiples = 69 →
  (69 / 120 : ℚ) = 69 / 120 :=
by
  sorry

end card_probability_multiple_3_4_or_7_l0_38


namespace F_participated_in_4_games_l0_843

noncomputable def number_of_games_f_participated : Prop :=
  ∃ (A B C D E F : ℕ), 
    -- Conditions
    A = 3 ∧ B = 3 ∧ C = 4 ∧ D = 4 ∧ E = 2 ∧ 
    -- A did not play against C
    ∀ games : list (ℕ × ℕ), (A, C) ∉ games ∧
    -- B did not play against D
    (B, D) ∉ games ∧
    -- Total number of games F participated in on the first day
    F = 4

theorem F_participated_in_4_games : number_of_games_f_participated := 
  sorry

end F_participated_in_4_games_l0_843


namespace foldable_shape_is_axisymmetric_l0_734

def is_axisymmetric_shape (shape : Type) : Prop :=
  (∃ l : (shape → shape), (∀ x, l x = x))

theorem foldable_shape_is_axisymmetric (shape : Type) (l : shape → shape) 
  (h1 : ∀ x, l x = x) : is_axisymmetric_shape shape := by
  sorry

end foldable_shape_is_axisymmetric_l0_734


namespace greatest_gcd_of_rope_lengths_l0_767

theorem greatest_gcd_of_rope_lengths : Nat.gcd (Nat.gcd 39 52) 65 = 13 := by
  sorry

end greatest_gcd_of_rope_lengths_l0_767


namespace clea_ride_time_l0_61

-- Definitions from conditions 
variables {c s d t : ℝ}

-- Given conditions as definitions
def non_operating_escalator : Prop := d = 120 * c
def operating_escalator : Prop := d = 48 * (c + s)

-- Target proof statement
theorem clea_ride_time 
  (h₁ : non_operating_escalator) 
  (h₂ : operating_escalator) : 
  t = 80 :=
by sorry

end clea_ride_time_l0_61


namespace reciprocals_log_AP_l0_731

variables {a b c r k n : ℕ}
-- Conditions:
def condition1 := b = a * r^k
def condition2 := c = a * r^(2*k)
def condition3 := r > 0
def condition4 := k > 0
def condition5 := n > 1

-- Statement:
theorem reciprocals_log_AP (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) (h5 : condition5) :
  let y1 := 1 / log a n,
      y2 := 1 / log b n,
      y3 := 1 / log c n
  in (y2 - y1) = (y3 - y2) :=
by sorry

end reciprocals_log_AP_l0_731


namespace problem_theorem_l0_232

open scoped Nat

-- Define the function f(n) as the largest prime factor of n
noncomputable def largestPrimeFactor (n : ℕ) : ℕ :=
  if h : n > 1 then
    Nat.find_greatest (λ p, Nat.Prime p ∧ p ∣ n) n
  else 1

-- Define the sum of f(n) over the given range
noncomputable def sum_f_n2_minus_1 (start stop : ℕ) : ℕ :=
  (Finset.range (stop - start + 1)).sum (λ i, largestPrimeFactor ((i + start)^2 - 1))

noncomputable def sum_f_n (start stop : ℕ) : ℕ :=
  (Finset.range (stop - start + 1)).sum (λ i, largestPrimeFactor (i + start))

-- Define the target value N
noncomputable def N : ℕ :=
  Nat.floor (10^4 * (sum_f_n2_minus_1 2 1000000) / (sum_f_n 2 1000000))

-- The theorem stating the value of N
theorem problem_theorem : N = 18215 :=
  by
  unfold N
  sorry

end problem_theorem_l0_232


namespace largest_p_q_sum_l0_951

theorem largest_p_q_sum 
  (p q : ℝ)
  (A := (p, q))
  (B := (12, 19))
  (C := (23, 20))
  (area_ABC : ℝ := 70)
  (slope_median : ℝ := -5)
  (midpoint_BC := ((12 + 23) / 2, (19 + 20) / 2))
  (eq_median : (q - midpoint_BC.2) = slope_median * (p - midpoint_BC.1))
  (area_eq : 140 = 240 - 437 - 20 * p + 23 * q + 19 * p - 12 * q) :
  p + q ≤ 47 :=
sorry

end largest_p_q_sum_l0_951


namespace paris_time_is_correct_l0_311

def time_difference_beijing_paris : ℤ := -7

def beijing_time_at_moment : ℤ := 5

def paris_time_at_moment : ℤ := beijing_time_at_moment + time_difference_beijing_paris

theorem paris_time_is_correct :
  paris_time_at_moment = 22 ∧ "October 25" :=
  sorry

end paris_time_is_correct_l0_311


namespace arithmetic_sequence_property_l0_850

-- Define the arithmetic sequence and the given conditions
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n+1) - a n = a 1 - a 0

def problemStatement (a : ℕ → ℝ) :=
isArithmeticSequence a ∧
((∑ i in Finset.range 150, a i) = 150) ∧
((∑ i in Finset.range (300-150) + 150, a i) - (∑ i in Finset.range 150, a i) = 300)

-- The desired constant difference between the first two terms
def answer : ℝ := 1 / 150

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : problemStatement a) :
  a 1 - a 0 = answer :=
sorry

end arithmetic_sequence_property_l0_850


namespace cosine_135_eq_neg_sqrt_2_div_2_l0_543

theorem cosine_135_eq_neg_sqrt_2_div_2 (cos180 : Real := -1) (cos90 : Real := 0) (cos45 : Real := Real.sqrt 2 / 2)
    (theta1 : Real := 180) (theta2 : Real := 45) (theta3 : Real := 135) :
    theta3 = theta1 - theta2 ∧ 
    Real.cos theta1 = cos180 ∧ 
    Real.cos theta2 = cos45 → 
    Real.cos theta3 = -cos45 := by
  sorry

end cosine_135_eq_neg_sqrt_2_div_2_l0_543


namespace product_sequence_fraction_l0_55

theorem product_sequence_fraction :
  let seq := list.range' 1 503 in
  let product := list.prod (seq.map (λ n, (n : ℚ) / (n + 1))) in
  product = 1 / 504 :=
by
  sorry

end product_sequence_fraction_l0_55


namespace grapefruits_count_l0_996

def citrus_orchards : Prop :=
  ∃ (total lemons oranges limes grapefruits : ℕ),
    (total = 16) ∧ (lemons = 8) ∧ (oranges = lemons / 2) ∧ 
    (limes + grapefruits = total - (lemons + oranges)) ∧ 
    (limes = grapefruits) ∧ (grapefruits = 2)

theorem grapefruits_count : citrus_orchards :=
begin
  unfold citrus_orchards,
  use 16,
  use 8,
  use 4,
  use 2,
  use 2,
  split,
  { refl },
  split,
  { refl },
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { refl },
  refl,
end

end grapefruits_count_l0_996


namespace infinite_square_free_gaps_l0_47

open Nat

/--
  Arrange all square-free positive integers in ascending order \(a_1, a_2, a_3, \ldots, a_n, \ldots \).
  Prove that there are infinitely many positive integers \( n \) such that \( a_{n+1} - a_n = 2020 \).
-/
theorem infinite_square_free_gaps : ∃ᶠ n in at_top, ∃ k > n, square_free_seq (k + 1) - square_free_seq k = 2020 := 
sorry

def square_free_seq : ℕ → ℕ := sorry

def is_square_free (n : ℕ) : Prop :=
  ∀ d : ℕ, d * d ∣ n → d = 1

/--
  Given a positive integer sequence defined by \( a_n \), where \( a_k \) is the \( k \)-th square-free number.
-/
def square_free_seq (n : ℕ) : ℕ :=
  (Finset.range (n * n)).filter is_square_free

end infinite_square_free_gaps_l0_47


namespace cos_135_eq_neg_sqrt_two_div_two_l0_453

theorem cos_135_eq_neg_sqrt_two_div_two : real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt_two_div_two_l0_453


namespace census_suitable_survey_l0_966

-- Definitions corresponding to conditions
def surveyA : Prop := "Survey on the quality and safety of local grain processing conducted by the Market Supervision Administration."
def surveyB : Prop := "Survey on the viewership ratings of the 2023 CCTV Spring Festival Gala."
def surveyC : Prop := "Survey on the weekly duration of physical exercise for a ninth-grade class."
def surveyD : Prop := "Survey on the household chores participation of junior high school students in the entire city."

-- The statement we need to prove
theorem census_suitable_survey (A B C D : Prop) : C :=
by sorry

end census_suitable_survey_l0_966


namespace probability_real_roots_of_quadratic_l0_116

theorem probability_real_roots_of_quadratic (b : ℝ) (hb : 0 < b ∧ b < 1) : 
  let favorable_interval_length := (1 / 4 : ℝ) - 0
      total_interval_length := 1 - 0
  in (favorable_interval_length / total_interval_length = 1 / 4) :=
by
  sorry

end probability_real_roots_of_quadratic_l0_116


namespace max_red_points_l0_944

theorem max_red_points (n : ℕ) (h_n : n = 1600) : 
  max_red_dyable_points n = 26 := 
sorry

/-- 
Helper definition that represents the function for calculating
the maximum number of red-dyeable points based on the given rule.
-/
noncomputable def max_red_dyable_points (n : ℕ) : ℕ := 
if n = 25 then 20
else if n = 50 then 21
else if n = 100 then 22
else if n = 200 then 23
else if n = 400 then 24
else if n = 800 then 25
else if n = 1600 then 26
else 0 -- non-interesting case not required by the problem

end max_red_points_l0_944


namespace find_r_l0_136

theorem find_r 
  (r : ℤ) 
  (h1 : -1 ≤ r ∧ r ≤ 5)
  (h2 : coefficient (x^r) ((1 - 1/x)*(1 + x)^5) = 0) 
  : r = 2 :=
sorry

end find_r_l0_136


namespace cos_135_eq_neg_inv_sqrt_2_l0_528

theorem cos_135_eq_neg_inv_sqrt_2 : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_528


namespace compute_division_l0_337

variable (a b c : ℕ)
variable (ha : a = 3)
variable (hb : b = 2)
variable (hc : c = 2)

theorem compute_division : (c * a^3 + c * b^3) / (a^2 - a * b + b^2) = 10 := by
  sorry

end compute_division_l0_337


namespace intersect_point_on_BC_l0_183

open EuclideanGeometry

theorem intersect_point_on_BC (A B C M N X : Point) :
  is_midpoint M A B →
  is_midpoint N A C →
  tangent AX (circumcircle A B C) →
  ∃ ω_B : Circle, centroid BC M B ∧ tangent_at M (circumcircle M B X) ω_B ∧
  ∃ ω_C : Circle, centroid BC N C ∧ tangent_at N (circumcircle N C X) ω_C ∧
  ∃ D : Point, D ∈ intersection_points ω_B ω_C ∧ collinear B C D :=
sorry

end intersect_point_on_BC_l0_183


namespace carla_laundry_l0_429

def totalLaundry(startHour : ℕ, endHour : ℕ, piecesPerHour : ℕ) : ℕ :=
  (endHour - startHour) * piecesPerHour

theorem carla_laundry (h_start : 8 = 8) (h_end : 12 = 12) (h_pieces : 20 = 20) : totalLaundry 8 12 20 = 80 :=
  by
  rw [totalLaundry]
  sorry

end carla_laundry_l0_429


namespace problem_solution_l0_779

variables {A B C A_1 B_1 C_1 O α β γ α_1 β_1 γ_1 : Type} 
[InscirbedTriangles : CongruentTrianglesInscribed ABC A_1B_1C_1 O]
  
def pedal_triangle_similar : Prop :=
  SimilarTriangles α β γ (PedalTriangle ABC O)

def circumcenter_properties : Prop :=
  ∀ P, circumcenter α β γ O ∧ orthocenter α_1 β_1 γ_1 O

def maximal_triangle : Prop :=
  maximized α β γ ∧ minimized α_1 β_1 γ_1 ↔ coincident ABC A_1B_1C_1

def proof : Prop :=
  pedal_triangle_similar ∧ circumcenter_properties ∧ maximal_triangle

theorem problem_solution: proof := 
sorry

end problem_solution_l0_779


namespace solution_to_equation_l0_828

theorem solution_to_equation (x y : ℕ → ℕ) (h1 : x 1 = 2) (h2 : y 1 = 3)
  (h3 : ∀ k, x (k + 1) = 3 * x k + 2 * y k)
  (h4 : ∀ k, y (k + 1) = 4 * x k + 3 * y k) :
  ∀ n, 2 * (x n)^2 + 1 = (y n)^2 := 
by
  sorry

end solution_to_equation_l0_828


namespace common_factor_polynomial_l0_269

variable {R : Type*} [CommRing R]
variable (a b c : R)

-- Define the terms of the polynomial
def term1 := 8 * a^3 * b^2
def term2 := 12 * a * b^3 * c

-- Define the GCD of the numerical coefficients
def gcd_num_coeffs : R := 4

-- Define the common factor with the smallest power present in both terms
def common_factor : R := gcd_num_coeffs * a * b^2

theorem common_factor_polynomial :
  common_factor = 4 * a * b^2 :=
sorry

end common_factor_polynomial_l0_269


namespace number_of_players_l0_39

variable (total_socks : ℕ) (socks_per_player : ℕ)

theorem number_of_players (h1 : total_socks = 16) (h2 : socks_per_player = 2) :
  total_socks / socks_per_player = 8 := by
  -- proof steps will go here
  sorry

end number_of_players_l0_39


namespace modulus_complex_z_l0_140

noncomputable def complex_z : ℂ := (1 - 3 * complex.i) / (1 + 2 * complex.i)

theorem modulus_complex_z : complex.abs complex_z = real.sqrt 2 := 
by
  -- Proof to be filled in later
  sorry

end modulus_complex_z_l0_140


namespace evaluate_fraction_l0_642

theorem evaluate_fraction:
  (125 : ℝ)^(1/3) / (64 : ℝ)^(1/2) * (81 : ℝ)^(1/4) = 15 / 8 := 
by
  sorry

end evaluate_fraction_l0_642


namespace find_b_l0_111

-- Define the function f
def f (x : ℝ) (b : ℝ) := 2^x + b

-- Define the condition that inverse of f passes through (5, 2)
def inverse_passes_through (b : ℝ) := ∃ (y : ℝ), f (2) b = 5

theorem find_b : ∃ (b : ℝ), inverse_passes_through b ∧ b = 1 :=
by
  use 1
  sorry

end find_b_l0_111


namespace range_of_a_l0_704

noncomputable def tangent_slopes (a x0 : ℝ) : ℝ × ℝ :=
  let k1 := (a * x0 + a - 1) * Real.exp x0
  let k2 := (x0 - 2) * Real.exp (-x0)
  (k1, k2)

theorem range_of_a (a x0 : ℝ) (h : x0 ∈ Set.Icc 0 (3 / 2))
  (h_perpendicular : (tangent_slopes a x0).1 * (tangent_slopes a x0).2 = -1)
  : 1 ≤ a ∧ a ≤ 3 / 2 :=
sorry

end range_of_a_l0_704


namespace total_ounces_of_drink_l0_210

-- Definitions based on the conditions
def coke_parts : ℕ := 2
def sprite_parts : ℕ := 1
def mountain_dew_parts : ℕ := 3
def coke_ounces : ℕ := 6

-- Theorem stating the total ounces in the drink
theorem total_ounces_of_drink : 
  let total_parts := coke_parts + sprite_parts + mountain_dew_parts
  let ounces_per_part := coke_ounces / coke_parts
in total_parts * ounces_per_part = 18 := by
  sorry

end total_ounces_of_drink_l0_210


namespace five_p_coins_problem_l0_408

theorem five_p_coins_problem :
  ∃ (N : ℕ), (N < 20001) ∧
            ∃ (x : ℕ), (N = 11 * x) ∧
            ((11 * x) % 12 = 3) ∧
            ((11 * x) % 18 = 3) ∧
            ((11 * x) % 45 = 3) ∧
            (N % 5 = 0) ∧
            (N = 363 + 1980 * (N / 1980) ) :=
begin
  sorry
end

end five_p_coins_problem_l0_408


namespace cos_135_eq_neg_inv_sqrt_2_l0_573

theorem cos_135_eq_neg_inv_sqrt_2 :
  cos (135 * Real.pi / 180) = - (1 / Real.sqrt 2) :=
by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_573


namespace hair_diameter_scientific_notation_l0_863

theorem hair_diameter_scientific_notation :
  let diameter_in_meters := 0.0000597
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ diameter_in_meters = a * 10 ^ n ∧ a = 5.97 ∧ n = -5 :=
by {
  let diameter_in_meters := 0.0000597,
  use [5.97, -5],
  split,
  {split,
    {simp, norm_num},
    {simp, norm_num}},
  {split,
    {ring_nf},
    {simp}}
}

end hair_diameter_scientific_notation_l0_863


namespace greatest_divisor_540_less_than_60_and_factor_of_126_l0_340

theorem greatest_divisor_540_less_than_60_and_factor_of_126 : ∃ d ∈ (finset.Ico 1 60), d ∣ 540 ∧ d ∣ 126 ∧ ∀ e ∈ (finset.Ico 1 60), e ∣ 540 ∧ e ∣ 126 → e ≤ d :=
sorry

end greatest_divisor_540_less_than_60_and_factor_of_126_l0_340


namespace lila_position_seventh_l0_189

open Finset

def person := {x : string // x ∈ ["Lila", "Esther", "Noel", "Ivan", "Jessica", "Omar"]}

def position : person → ℕ 
| ⟨"Lila", _⟩ := 7
| ⟨"Esther", _⟩ := 3
| ⟨"Noel", _⟩ := 2
| ⟨"Ivan", _⟩ := 4
| ⟨"Jessica", _⟩ := 10
| ⟨"Omar", _⟩ := 6

lemma jessica_behind_esther : position ⟨"Jessica", sorry⟩ = position ⟨"Esther", sorry⟩ + 7 := by sorry
lemma ivan_behind_noel : position ⟨"Ivan", sorry⟩ = position ⟨"Noel", sorry⟩ + 2 := by sorry
lemma lila_behind_esther : position ⟨"Lila", sorry⟩ = position ⟨"Esther", sorry⟩ + 4 := by sorry
lemma noel_behind_omar : position ⟨"Noel", sorry⟩ = position ⟨"Omar", sorry⟩ + 4 := by sorry
lemma omar_behind_esther : position ⟨"Omar", sorry⟩ = position ⟨"Esther", sorry⟩ + 3 := by sorry
lemma ivan_position : position ⟨"Ivan", sorry⟩ = 4 := by sorry

theorem lila_position_seventh : position ⟨"Lila", sorry⟩ = 7 := by sorry

end lila_position_seventh_l0_189


namespace tangent_line_at_A_increasing_intervals_decreasing_interval_l0_711

noncomputable def f (x : ℝ) := 2 * x^3 + 3 * x^2 + 1

-- Define the derivatives at x
noncomputable def f' (x : ℝ) := 6 * x^2 + 6 * x

-- Define the tangent line equation at a point
noncomputable def tangent_line (x : ℝ) := 12 * x - 6

theorem tangent_line_at_A :
  tangent_line 1 = 6 :=
  by
    -- proof omitted
    sorry

theorem increasing_intervals :
  (∀ x ∈ Set.Ioi 0, f' x > 0) ∧
  (∀ x ∈ Set.Iio (-1), f' x > 0) :=
  by
    -- proof omitted
    sorry

theorem decreasing_interval :
  ∀ x ∈ Set.Ioo (-1) 0, f' x < 0 :=
  by
    -- proof omitted
    sorry

end tangent_line_at_A_increasing_intervals_decreasing_interval_l0_711


namespace g_of_f_of_3_is_217_l0_782

-- Definitions of f and g
def f (x : ℝ) : ℝ := x^2 - 4
def g (x : ℝ) : ℝ := x^3 + 3 * x^2 + 3 * x + 2

-- The theorem we need to prove
theorem g_of_f_of_3_is_217 : g (f 3) = 217 := by
  sorry

end g_of_f_of_3_is_217_l0_782


namespace coefficient_x10_l0_958

theorem coefficient_x10 (n k : ℕ) (a b : ℕ) (h1 : n = 12) (h2 : a = 2) (h3 : b = 10) : 
  (binomial n k * a ^ k = 264) ↔ (n = 12 ∧ k = 2 ∧ a = 2 ∧ binomial 12 2 * 2^2 = 264) :=
by
  sorry

end coefficient_x10_l0_958


namespace construct_triangle_l0_632

structure Triangle where
  A B C : Point

def is_median (A B M : Point) : Prop :=
  (M = midpoint B C)

def is_centroid (A B C O : Point) (G : Point) : Prop :=
  G = centroid A B C

def is_constructible (a m1 m2 : ℝ) : Prop :=
  ∃ (A B C M N O : Point),
    (BC_len = a) ∧
    (AM_len = m1) ∧
    (BN_len = m2) ∧
    is_median A B M ∧
    is_median B C N ∧
    is_centroid A B C O ∧
    ((BM = a / 2) ∧ (BO = (2 / 3) * m2) ∧ (MO = (1 / 3) * m1))

theorem construct_triangle (a m1 m2 : ℝ) :
  is_constructible a m1 m2 :=
sorry

end construct_triangle_l0_632


namespace area_of_triangle_MNP_l0_48

-- Definitions of the problem's conditions
variables (A B C D M N P : Type) [AddGroup A][AddGroup B][AddGroup C][AddGroup D][AddGroup M][AddGroup N][AddGroup P]
variables (MN AD BC h : ℝ)

def trapezoid_area (AD BC h : ℝ) : ℝ := (AD + BC) / 2 * h

def is_midline (AD BC : ℝ) (MN : ℝ) : Prop := MN = (AD + BC) / 2

noncomputable def height_of_trapezoid (AD BC area : ℝ) : ℝ := (2 * area) / (AD + BC)

def area_triangle_MNP (MN PH : ℝ) : ℝ := (MN * PH) / 2

-- Translated Lean statement
theorem area_of_triangle_MNP 
  (H : trapezoid_area AD BC h = 76)
  (is_midline AD BC MN)
  (area_MN_h : MN * h = 76) 
  : area_triangle_MNP MN (38/MN) = 19 :=
by 
  sorry -- Proof not required

end area_of_triangle_MNP_l0_48


namespace simplify_expression_l0_258

variable (x : ℝ)

theorem simplify_expression : 3 * x + 4 * x^3 + 2 - (7 - 3 * x - 4 * x^3) = 8 * x^3 + 6 * x - 5 := 
by 
  sorry

end simplify_expression_l0_258


namespace cos_135_eq_correct_l0_609

noncomputable def cos_135_eq: Prop :=
  real.cos ((135 * real.pi) / 180) = -real.sqrt 2 / 2

theorem cos_135_eq_correct : cos_135_eq :=
  sorry

end cos_135_eq_correct_l0_609


namespace cos_135_eq_neg_inv_sqrt_2_l0_524

theorem cos_135_eq_neg_inv_sqrt_2 : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_524


namespace positive_difference_l0_897

theorem positive_difference (a b : ℝ) (h₁ : a + b = 10) (h₂ : a^2 - b^2 = 40) : |a - b| = 4 :=
by
  sorry

end positive_difference_l0_897


namespace min_total_cost_min_total_cost_ge_144_l0_324

variable (s a : ℝ)

def total_cost (v : ℝ) : ℝ :=
  s * ((a / v) + (v / 25))

theorem min_total_cost (h1 : 0 < a) (h2 : a < 144) : 
  is_minimum (total_cost s a) (5 * real.sqrt a) (set.Icc 0 60) := sorry

theorem min_total_cost_ge_144 (h1 : 144 ≤ a) : 
  is_minimum (total_cost s a) 60 (set.Icc 0 60) := sorry

end min_total_cost_min_total_cost_ge_144_l0_324


namespace resulting_solution_percentage_l0_22

theorem resulting_solution_percentage :
  ∀ (C_init R C_replace : ℚ), 
  C_init = 0.85 → 
  R = 0.6923076923076923 → 
  C_replace = 0.2 → 
  (C_init * (1 - R) + C_replace * R) = 0.4 :=
by
  intros C_init R C_replace hC_init hR hC_replace
  -- Omitted proof here
  sorry

end resulting_solution_percentage_l0_22


namespace cos_135_eq_neg_sqrt2_div_2_l0_594

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  -- Conditions incorporated directly into the requirement due to the nature of trigonometric evaluations.
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_594


namespace probability_of_odd_numbers_exactly_five_times_in_seven_rolls_l0_329

theorem probability_of_odd_numbers_exactly_five_times_in_seven_rolls :
  (nat.choose 7 5 * (1/2)^5 * (1/2)^2) = (21 / 128) := by
  sorry

end probability_of_odd_numbers_exactly_five_times_in_seven_rolls_l0_329


namespace cos_135_eq_neg_inv_sqrt_2_l0_523

theorem cos_135_eq_neg_inv_sqrt_2 : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_523


namespace sum_of_solutions_eq_zero_l0_343

theorem sum_of_solutions_eq_zero :
  (∃ x : ℝ, (3 * x / 15) = (4 / x) ∧ x ≠ 0 → (Σ s in (finset.image (λ (x), (x : ℝ)) {2 * real.sqrt 5, -2 * real.sqrt 5}), s) = 0) :=
sorry

end sum_of_solutions_eq_zero_l0_343


namespace find_counterexample_l0_73

-- Define the set of candidate numbers
def candidate_numbers : List ℕ := [15, 18, 24, 28, 30]

-- To state if a number is composite, we state it is not prime
def is_composite (n : ℕ) : Prop := ¬ (Nat.prime n)

-- To state if a number is a counterexample
def is_counterexample (n : ℕ) : Prop := is_composite n ∧ is_composite (n - 3)

-- Stating the theorem
theorem find_counterexample : ∃ n ∈ candidate_numbers, is_counterexample n :=
by
  use 15
  sorry

end find_counterexample_l0_73


namespace geometric_sequence_general_term_and_arithmetic_sequence_max_sum_l0_125

theorem geometric_sequence_general_term_and_arithmetic_sequence_max_sum :
  (∃ a_n : ℕ → ℕ, ∃ b_n : ℕ → ℤ, ∃ T_n : ℕ → ℤ,
    (∀ n, a_n n = 2^(n-1)) ∧
    (a_n 1 + a_n 2 = 3) ∧
    (b_n 2 = a_n 3) ∧
    (b_n 3 = -b_n 5) ∧
    (∀ n, T_n n = n * (b_n 1 + b_n n) / 2) ∧
    (T_n 3 = 12) ∧
    (T_n 4 = 12)) :=
by
  sorry

end geometric_sequence_general_term_and_arithmetic_sequence_max_sum_l0_125


namespace range_of_fraction_l0_726

theorem range_of_fraction (x y : ℝ) (hx : 1 ≤ x ∧ x ≤ 4) (hy : 3 ≤ y ∧ y ≤ 6) :
  ∀ z, z = x / y → (1 / 6 ≤ z ∧ z ≤ 4 / 3) :=
sorry

end range_of_fraction_l0_726


namespace true_propositions_l0_239

variables {α β : Type*} [plane α] [plane β]

/-- Proposition 1: If two intersecting lines within plane α are parallel to two lines within plane β respectively, then plane α is parallel to plane β. -/
def prop1 (intersecting_lines_within_alpha : line α) (parallel_to_lines_within_beta : line β) : Prop :=
∀ (line1 line2 : line α), intersecting line1 line2 → (parallel line1 (line β)) ∧ (parallel line2 (line β)) → parallel α β

/-- Proposition 2: If a line l outside of plane α is parallel to a line within plane α, then line l and plane α are parallel. -/
def prop2 (l : line) (parallel_to_line_within_alpha : line α) : Prop :=
(∀ (line_in_alpha : line α), parallel l line_in_alpha) → parallel l α

/-- Proposition 3: If planes α and β intersect along line l, and if there is a line within plane α that is perpendicular to l, then planes α and β are perpendicular to each other. -/
def prop3 (l : line) (line_within_alpha_perpendicular_to_l : line α) : Prop :=
(intersects_along α β l) → (∃ (line_in_alpha : line α), perpendicular line_in_alpha l) → perpendicular_planes α β

/-- Proposition 4: The necessary and sufficient condition for line l to be perpendicular to plane α is for line l to be perpendicular to two lines within plane α. -/
def prop4 (l : line) (perpendicular_to_two_lines_within_alpha : line α) : Prop :=
(∀ (line1 line2 : line α), perpendicular l line1 ∧ perpendicular l line2) ↔ perpendicular l α

theorem true_propositions :
  prop1 ∧ prop2 ∧ ¬prop3 ∧ ¬prop4 :=
by sorry

end true_propositions_l0_239


namespace ticket_difference_l0_49

/-- 
  Define the initial number of tickets Billy had,
  the number of tickets after buying a yoyo,
  and state the proof that the difference is 16.
--/

theorem ticket_difference (initial_tickets : ℕ) (remaining_tickets : ℕ) 
  (h₁ : initial_tickets = 48) (h₂ : remaining_tickets = 32) : 
  initial_tickets - remaining_tickets = 16 :=
by
  /- This is where the prover would go, 
     no need to implement it as we know the expected result -/
  sorry

end ticket_difference_l0_49


namespace cos_135_eq_neg_sqrt2_div_2_l0_589

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  -- Conditions incorporated directly into the requirement due to the nature of trigonometric evaluations.
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_589


namespace product_of_areas_equal_l0_21

-- Define the given distances from vertices to tangency points and heights
variables {n : ℕ} {a : Fin n → ℝ} {h : Fin n → ℝ}

theorem product_of_areas_equal :
  (∏ i in Finset.filter (λ i => i % 2 = 1) (Finset.range n), (1/2 * a i * h i)) =
  (∏ i in Finset.filter (λ i => i % 2 = 0) (Finset.range n), (1/2 * a i * h i)) :=
sorry

end product_of_areas_equal_l0_21


namespace anna_age_when_married_l0_215

-- Define constants for the conditions
def j_married : ℕ := 22
def m : ℕ := 30
def combined_age_today : ℕ := 5 * j_married
def j_current : ℕ := j_married + m

-- Define Anna's current age based on the combined age today and Josh's current age
def a_current : ℕ := combined_age_today - j_current

-- Define Anna's age when married
def a_married : ℕ := a_current - m

-- Statement of the theorem to be proved
theorem anna_age_when_married : a_married = 28 :=
by
  sorry

end anna_age_when_married_l0_215


namespace prove_problem_l0_219

noncomputable def proof_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) : Prop :=
  (1 + 1 / x) * (1 + 1 / y) ≥ 9

theorem prove_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : proof_problem x y hx hy h :=
  sorry

end prove_problem_l0_219


namespace total_height_of_buildings_l0_309

-- Define the heights of the buildings based on the given conditions
def height_1 : ℝ := 100
def height_2 : ℝ := 1 / 2 * height_1
def height_3 : ℝ := 1 / 3 * height_2
def height_4 : ℝ := 1 / 4 * height_3
def height_5 : ℝ := 2 / 5 * height_4
def height_6 : ℝ := 3 / 4 * height_5

-- The total height 173.75 feet
theorem total_height_of_buildings : 
  height_1 + height_2 + height_3 + height_4 + height_5 + height_6 = 173.75 :=
by
  sorry

end total_height_of_buildings_l0_309


namespace garret_age_now_l0_321

variable (G : ℕ)
variable (Shane_current_age : ℕ) 
variable (twenty_years_ago : ℕ)

-- Define the current age of Shane.
def Shane_current_age := 44

-- Define the age of Shane twenty years ago.
def twenty_years_ago := Shane_current_age - 20

-- Statement: if twenty years ago, Shane's age was twice Garret's current age, prove Garret's age is 12.
theorem garret_age_now :
  (twenty_years_ago = 2 * G) → (G = 12) :=
by
  sorry

end garret_age_now_l0_321


namespace union_complement_A_B_eq_l0_159

-- Define the universal set U, set A, and set B
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Define the complement of A with respect to U
def complement_U_A : Set ℕ := {x | x ∈ U ∧ x ∉ A}

-- The statement to be proved
theorem union_complement_A_B_eq {U A B : Set ℕ} (hU : U = {0, 1, 2, 3, 4}) 
  (hA : A = {0, 1, 2, 3}) (hB : B = {2, 3, 4}) :
  (complement_U_A) ∪ B = {2, 3, 4} := 
by
  sorry

end union_complement_A_B_eq_l0_159


namespace find_magical_points_on_specific_quad_find_t_for_unique_magical_point_l0_741

-- Define what it means to be a "magical point"
def is_magical_point (m : ℝ) (A : ℝ × ℝ) : Prop :=
  A = (m, 2 * m)

-- Specialize for the specific quadratic function y = x^2 - x - 4
def on_specific_quadratic (m : ℝ) (A : ℝ × ℝ) : Prop :=
  A = (m, m^2 - m - 4)

-- Theorem for part 1: Find the magical points on y = x^2 - x - 4
theorem find_magical_points_on_specific_quad (m : ℝ) (A : ℝ × ℝ) :
  is_magical_point m A ∧ on_specific_quadratic m A →
  (A = (4, 8) ∨ A = (-1, -2)) :=
sorry

-- Define the quadratic function for part 2
def on_general_quadratic (t m : ℝ) (A : ℝ × ℝ) : Prop :=
  A = (m, t * m^2 + (t-2) * m - 4)

-- Theorem for part 2: Find the t values for unique magical points
theorem find_t_for_unique_magical_point (t m : ℝ) (A : ℝ × ℝ) :
  ( ∀ m, is_magical_point m A ∧ on_general_quadratic t m A → 
    (t * m^2 + (t-4) * m - 4 = 0) ) → 
  ( ∃! m, is_magical_point m A ∧ on_general_quadratic t m A ) →
  t = -4 :=
sorry

end find_magical_points_on_specific_quad_find_t_for_unique_magical_point_l0_741


namespace sum_of_digits_product_7777_5555_l0_91

theorem sum_of_digits_product_7777_5555 :
  let num_sevens := 72
  let num_fives := 72
  let sevens := list.repeat 7 num_sevens
  let fives := list.repeat 5 num_fives
  let product := (list.foldl (λ acc d, acc * 10 + d) 0 sevens) * (list.foldl (λ acc d, acc * 10 + d) 0 fives)
  let sum_of_digits := list.foldl (λ acc d, acc + d) 0 (list.map (λ n, int.of_nat n) (int.to_nat.product.digits 10))
  sum_of_digits = 576 := sorry

end sum_of_digits_product_7777_5555_l0_91


namespace hypotenuse_length_13_l0_675

noncomputable def right_angled_triangle_hypotenuse : ℝ :=
  let roots := (5, 12)
  let (a, b) := roots
  real.sqrt (a^2 + b^2)

theorem hypotenuse_length_13 :
  ∃ a b : ℝ, (a = 5 ∧ b = 12) → right_angled_triangle_hypotenuse = 13 :=
by
  sorry

end hypotenuse_length_13_l0_675


namespace find_added_number_l0_387

theorem find_added_number 
  (initial_number : ℕ)
  (final_result : ℕ)
  (h : initial_number = 8)
  (h_result : 3 * (2 * initial_number + final_result) = 75) : 
  final_result = 9 := by
  sorry

end find_added_number_l0_387


namespace cos_135_eq_neg_sqrt2_div_2_l0_506

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = - Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_506


namespace liars_positions_l0_811

structure Islander :=
  (position : Nat)
  (statement : String)

-- Define our islanders
def A : Islander := { position := 1, statement := "My closest tribesman in this line is 3 meters away from me." }
def D : Islander := { position := 4, statement := "My closest tribesman in this line is 1 meter away from me." }
def E : Islander := { position := 5, statement := "My closest tribesman in this line is 2 meters away from me." }

-- Define the other islanders with dummy statements
def B : Islander := { position := 2, statement := "" }
def C : Islander := { position := 3, statement := "" }
def F : Islander := { position := 6, statement := "" }

-- Define the main theorem
theorem liars_positions (knights_count : Nat) (liars_count : Nat) (is_knight : Islander → Bool)
  (is_lair : Islander → Bool) : 
  ( ∀ x, is_knight x ↔ ¬is_lair x ) → -- Knight and liar are mutually exclusive
  knights_count = 3 → 
  liars_count = 3 →
  is_knight A = false → 
  is_knight D = false → 
  is_knight E = false → 
  is_lair A = true ∧
  is_lair D = true ∧
  is_lair E = true := by
  sorry

end liars_positions_l0_811


namespace positive_difference_l0_913

theorem positive_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
sorry

end positive_difference_l0_913


namespace line_contains_point_iff_k_eq_neg1_l0_103

theorem line_contains_point_iff_k_eq_neg1 (k : ℝ) :
  (∃ x y : ℝ, x = 2 ∧ y = -1 ∧ (2 - k * x = -4 * y)) ↔ k = -1 :=
by
  sorry

end line_contains_point_iff_k_eq_neg1_l0_103


namespace isosceles_right_triangle_hypotenuse_l0_881

noncomputable def length_of_hypotenuse (s : ℝ) : ℝ := s * Real.sqrt 2

theorem isosceles_right_triangle_hypotenuse :
  ∀ (s : ℝ), 
    (2 * s + s * Real.sqrt 2 = (Real.pi * ((s * Real.sqrt 2 / 2) ^ 2) / 2)) →
    length_of_hypotenuse s = 2 * Real.sqrt 2 * (2 + Real.sqrt 2) / Real.pi :=
by
  -- conditions stating the given perimeter equals the circumscribed circle area
  intros s condition
  -- declaring the length of the hypotenuse under the given condition
  sorry

end isosceles_right_triangle_hypotenuse_l0_881


namespace jake_pure_alcohol_l0_772

-- Definitions based on the conditions
def shots : ℕ := 8
def ounces_per_shot : ℝ := 1.5
def vodka_purity : ℝ := 0.5
def friends : ℕ := 2

-- Statement to prove the amount of pure alcohol Jake drank
theorem jake_pure_alcohol : (shots * ounces_per_shot * vodka_purity) / friends = 3 := by
  sorry

end jake_pure_alcohol_l0_772


namespace cos_135_eq_neg_inv_sqrt_2_l0_517

theorem cos_135_eq_neg_inv_sqrt_2 : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_517


namespace cos_135_eq_neg_sqrt2_div_2_l0_628

theorem cos_135_eq_neg_sqrt2_div_2 : 
  Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_628


namespace cos_alpha_add_pi_over_12_l0_729

theorem cos_alpha_add_pi_over_12 (α : ℝ) (h : Real.tan (α + π / 3) = -2) :
  Real.cos (α + π / 12) = (sqrt 10 / 10) ∨ Real.cos (α + π / 12) = -(sqrt 10 / 10) := sorry

end cos_alpha_add_pi_over_12_l0_729


namespace probability_ABABA_l0_318

-- Define the conditions
def num_A : ℕ := 3
def num_B : ℕ := 2
def total_tiles : ℕ := 5
def desired_sequence : list char := ['A', 'B', 'A', 'B', 'A']

-- Main theorem stating the probability
theorem probability_ABABA : 
  (1 : ℚ) / (Nat.factorial total_tiles / (Nat.factorial num_A * Nat.factorial num_B) : ℚ) = 1 / 10 := 
by 
  -- Initial steps to prepare the proof structure
  sorry

end probability_ABABA_l0_318


namespace sin_6_deg_approx_l0_798

theorem sin_6_deg_approx : 
  let r := 1 in  -- Assume the radius of the circle is 1 for simplicity.
  let area_sector := (6 / 360 * Real.pi * r^2) in
  let area_triangle := (1 / 2 * r^2 * Real.sin (6 * Real.pi / 180)) in
  area_sector ≈ area_triangle →
  Real.sin (6 * Real.pi / 180) ≈ Real.pi / 30 := 
by
  intros r area_sector area_triangle h
  -- Use the given condition and compute sin 6 degrees
  sorry

end sin_6_deg_approx_l0_798


namespace find_a_l0_184

noncomputable def triangle (A B C a b c : ℝ) : Prop :=
  ∃ (α β γ : ℝ), α = A ∧ β = B ∧ γ = C ∧
                 a = sqrt (b^2 + c^2 - 2 * b * c * cos A) ∧
                 b = sqrt (a^2 + c^2 - 2 * a * c * cos B) ∧
                 c = sqrt (a^2 + b^2 - 2 * a * b * cos C) ∧
                 α ∈ (0, π) ∧ β ∈ (0, π) ∧ γ ∈ (0, π)

variables (A B C a b c : ℝ)
variables (condition1: cos A^2 - cos B^2 + sin C^2 = sin B * sin C ∧ sin B * sin C = 1/4)
variables (condition2: area_of_triangle = sqrt 3)

noncomputable def area_of_triangle := 1/2 * b * c * sin A

theorem find_a : triangle A B C a b c ∧ condition1 ∧ condition2 → a = 2 * sqrt 3 :=
by sorry

end find_a_l0_184


namespace none_form_pair_one_pair_others_not_exactly_two_pairs_l0_362

theorem none_form_pair : (choose 10 4) * (2^4) = 3360 := 
by sorry

theorem one_pair_others_not : (choose 10 2) * (2^2) * 8 = 1440 := 
by sorry

theorem exactly_two_pairs : (choose 10 2) = 45 := 
by sorry

end none_form_pair_one_pair_others_not_exactly_two_pairs_l0_362


namespace find_sinusoidal_constants_l0_51

noncomputable def sinusoidal_function_values (a b c d : ℝ) :=
  ∃ y : ℝ → ℝ, 
    (y = λ x, a * Real.sin (b * x + c) + d) ∧ 
    (y 0 = 5) ∧ 
    (y (2 * Real.pi / 5) = y (-3)) ∧ -- Completes 5 cycles
    (Real.abs(a) = 4) ∧ 
    (d = 1) ∧ 
    (c = 0) ∧ 
    (b = 5)

theorem find_sinusoidal_constants:
  sinusoidal_function_values 4 5 0 1 :=
by sorry

end find_sinusoidal_constants_l0_51


namespace cos_135_degree_l0_493

theorem cos_135_degree : 
  let Q := (Complex.exp (135 * Real.pi / 180 * Complex.I)) in 
  Complex.re Q = -Real.sqrt 2 / 2 :=
by sorry

end cos_135_degree_l0_493


namespace positive_difference_l0_908

theorem positive_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
sorry

end positive_difference_l0_908


namespace cos_135_degree_l0_490

theorem cos_135_degree : 
  let Q := (Complex.exp (135 * Real.pi / 180 * Complex.I)) in 
  Complex.re Q = -Real.sqrt 2 / 2 :=
by sorry

end cos_135_degree_l0_490


namespace exists_i_for_inequality_l0_221

theorem exists_i_for_inequality (n : ℕ) (x : ℕ → ℝ) (h1 : 2 ≤ n) 
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n → 0 ≤ x i ∧ x i ≤ 1) :
  ∃ i, 1 ≤ i ∧ i ≤ n - 1 ∧ x i * (1 - x (i + 1)) ≥ (1 / 4) * x 1 * (1 - x n) :=
by
  sorry

end exists_i_for_inequality_l0_221


namespace michael_number_l0_244

theorem michael_number (m : ℕ) (h1 : m % 75 = 0) (h2 : m % 40 = 0) (h3 : 1000 < m) (h4 : m < 3000) :
  m = 1800 ∨ m = 2400 ∨ m = 3000 :=
sorry

end michael_number_l0_244


namespace students_passed_both_tests_l0_315

theorem students_passed_both_tests (total_students : ℕ) (failed_both_tests : ℕ) 
  (passed_chinese : ℕ) (passed_english : ℕ) : 
  total_students = 50 → failed_both_tests = 4 → passed_chinese = 40 → passed_english = 31 →
  (∃ x : ℕ, passed_chinese + passed_english - (total_students - failed_both_tests) = x ∧ x = 25) :=
begin
  intros h1 h2 h3 h4,
  use (passed_chinese + passed_english - (total_students - failed_both_tests)),
  split,
  { refl },
  { rw [h1, h2, h3, h4],
    norm_num },
end

end students_passed_both_tests_l0_315


namespace cos_135_eq_neg_inv_sqrt_2_l0_582

theorem cos_135_eq_neg_inv_sqrt_2 :
  cos (135 * Real.pi / 180) = - (1 / Real.sqrt 2) :=
by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_582


namespace good_number_condition_l0_182

-- Definition of what it means for a number n to be a "good number"
def is_good_number (n : ℕ) : Prop :=
  ∃ (a : ℕ → ℕ) (h_perm : ∀ i, i ∈ finset.range n → a i ∈ finset.range n),
    ∀ k, k ∈ finset.range n → ∃ m, (k + a k) = m * m

-- Define the set of "not-good" numbers
def not_good_numbers := {1, 2, 4, 6, 7, 9, 11}

-- Prove that for all n not in the set of "not-good" numbers, n is a "good number"
theorem good_number_condition (n : ℕ) :
  n ∉ not_good_numbers ↔ is_good_number n :=
by
  sorry

end good_number_condition_l0_182


namespace cos_135_eq_neg_sqrt2_div_2_l0_595

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  -- Conditions incorporated directly into the requirement due to the nature of trigonometric evaluations.
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_595


namespace modulus_of_complex_l0_285

/--
  Prove that the modulus of the complex number \( \frac{i^{2017}}{1+i} \) (where \( i \) is the imaginary unit) is equal to \( \frac{\sqrt{2}}{2} \)
-/
theorem modulus_of_complex (i : ℂ) (h : i^2 = -1) : 
  complex.abs (i^2017 / (1 + i)) = real.sqrt 2 / 2 :=
sorry

end modulus_of_complex_l0_285


namespace percent_of_a_is_20_l0_175

variable {a b c : ℝ}

theorem percent_of_a_is_20 (h1 : c = (x / 100) * a)
                          (h2 : c = 0.1 * b)
                          (h3 : b = 2 * a) :
  c = 0.2 * a := sorry

end percent_of_a_is_20_l0_175


namespace expression_evaluation_l0_79

theorem expression_evaluation : 
  1000 + (∑ k in finset.range 998, ((999 - k) / (3^k : ℝ))) + (2 / (3^998 : ℝ)) = 999.5 + 1498.5 * 3^997 :=
by
  sorry

end expression_evaluation_l0_79


namespace valid_pairs_count_l0_865

theorem valid_pairs_count :
  let f := λ A B α β : ℕ, α + β = A * B ∧ α * β = 10 * B ∧ 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9 ∧ A > B
  let pairs := { (A, B) | ∃ α β, f A B α β }
  pairs.count = 36 :=
  sorry

end valid_pairs_count_l0_865


namespace cos_135_eq_neg_sqrt2_div_2_l0_484

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by 
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_484


namespace maximum_profit_and_volume_l0_378

-- Definition of additional cost function C(x)
def C (x : ℝ) : ℝ :=
  if x < 80 then (1 / 3) * x^2 + 10 * x
  else 51 * x + 10000 / x - 1450

-- Definition of annual profit function L(x)
def L (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 80 then - (1 / 3) * x^2 + 40 * x - 250
  else 1200 - (x + 10000 / x)

-- Theorem to prove the maximum profit and optimal production volume
theorem maximum_profit_and_volume :
  ∃ x : ℝ, (x = 100 ∧ L x = 1000) := 
sorry

end maximum_profit_and_volume_l0_378


namespace cos_135_degree_l0_492

theorem cos_135_degree : 
  let Q := (Complex.exp (135 * Real.pi / 180 * Complex.I)) in 
  Complex.re Q = -Real.sqrt 2 / 2 :=
by sorry

end cos_135_degree_l0_492


namespace negation_of_universal_proposition_l0_875

def int_divisible_by_5 (n : ℤ) := ∃ k : ℤ, n = 5 * k
def int_odd (n : ℤ) := ∃ k : ℤ, n = 2 * k + 1

theorem negation_of_universal_proposition :
  (¬ ∀ n : ℤ, int_divisible_by_5 n → int_odd n) ↔ (∃ n : ℤ, int_divisible_by_5 n ∧ ¬ int_odd n) :=
by
  sorry

end negation_of_universal_proposition_l0_875


namespace find_sum_of_terms_l0_680

noncomputable def arithmetic_sequence (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d
noncomputable def sum_of_first_n_terms (a₁ d n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem find_sum_of_terms (a₁ d : ℕ) (S : ℕ → ℕ) (h1 : S 4 = 8) (h2 : S 8 = 20) :
    S 4 = 4 * (2 * a₁ + 3 * d) / 2 → S 8 = 8 * (2 * a₁ + 7 * d) / 2 →
    a₁ = 13 / 8 ∧ d = 1 / 4 →
    a₁ + 10 * d + a₁ + 11 * d + a₁ + 12 * d + a₁ + 13 * d = 18 :=
by 
  sorry

end find_sum_of_terms_l0_680


namespace tetrahedron_volume_l0_647

open Real

-- Given conditions
variables (A B C D : Point) (BC : ℝ) (angleABC_BCD : ℝ) (areaABC areaBCD : ℝ)
hypothesis hBC : BC = 12
hypothesis hangle : angleABC_BCD = π / 4 -- 45 degrees in radians
hypothesis hAreaABC : areaABC = 150
hypothesis hAreaBCD : areaBCD = 90

-- The goal is to compute the volume of tetrahedron ABCD
theorem tetrahedron_volume (hBC : BC = 12) (hangle : angleABC_BCD = π / 4) (hAreaABC : areaABC = 150) (hAreaBCD : areaBCD = 90) :
  volume_of_tetrahedron A B C D = 375 * sqrt 2 :=
sorry

end tetrahedron_volume_l0_647


namespace log_prod_eq_l0_421

variables {a b c k x : ℝ}
variables (logs : list ℝ) (n : ℕ)

noncomputable def log_abc_to_k (logs : list ℝ) (x : ℝ) : ℝ :=
1 / (list.sum (logs.map (λ l, 1 / l)))

theorem log_prod_eq :
  (∀ i : ℕ, i < logs.length → logs.nth i ≠ some 0) →
  x ≠ 1 →
  ∀ log_fun, (∀ i : ℕ, i < logs.length → log_fun (logs.nth_le i sorry) = logs.nth_le i sorry) →
  log_fun (list.prod logs) x = log_abc_to_k logs x :=
sorry

end log_prod_eq_l0_421


namespace train_cross_signal_pole_l0_984

theorem train_cross_signal_pole 
    (length_train : ℕ) 
    (length_platform : ℕ)
    (time_cross_platform : ℕ)
    (speed_train : ℚ)
    (h_length_train : length_train = 300)
    (h_length_platform : length_platform = 150)
    (h_time_cross_platform : time_cross_platform = 39)
    (h_speed_train : speed_train = (length_train + length_platform) / time_cross_platform) :
  (300 / speed_train) = 26 :=
begin
  rw [h_length_train, h_length_platform, h_time_cross_platform] at h_speed_train,
  rw [h_length_train, h_speed_train],
  norm_num,
end

end train_cross_signal_pole_l0_984


namespace total_ounces_of_drink_l0_211

-- Definitions based on the conditions
def coke_parts : ℕ := 2
def sprite_parts : ℕ := 1
def mountain_dew_parts : ℕ := 3
def coke_ounces : ℕ := 6

-- Theorem stating the total ounces in the drink
theorem total_ounces_of_drink : 
  let total_parts := coke_parts + sprite_parts + mountain_dew_parts
  let ounces_per_part := coke_ounces / coke_parts
in total_parts * ounces_per_part = 18 := by
  sorry

end total_ounces_of_drink_l0_211


namespace visible_length_red_bus_l0_280

-- Definitions from the problem
def length_red_bus : ℝ := 48
def length_orange_car : ℝ := length_red_bus / 4
def length_yellow_bus : ℝ := length_orange_car * 3.5

-- Proof statement
theorem visible_length_red_bus : (length_red_bus - length_yellow_bus) = 6 := by
  have h1 : length_orange_car = 12 := by sorry
  have h2 : length_yellow_bus = 42 := by sorry
  have h3 : length_red_bus = 48 := by sorry
  sorry

end visible_length_red_bus_l0_280


namespace cos_135_eq_neg_sqrt2_div_2_l0_476

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by 
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_476


namespace positive_difference_of_two_numbers_l0_929

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l0_929


namespace sum_real_roots_series_equation_l0_90

theorem sum_real_roots_series_equation :
  let f := λ x : ℝ, x^3 - 2 * x + 2
  let S := {x : ℝ | |x| < 1 ∧ f x = 0}
  ∑ x in S, x = -- Correct answer here, but we cannot numerically simplify so we'll state it as a property of S
sorry

end sum_real_roots_series_equation_l0_90


namespace all_walking_same_direction_l0_41

-- Given conditions in the Lean code
constants (alley_length : ℝ) (speed1 speed2 speed3 : ℝ)
constant (time_interval : ℝ)

-- Setting values according to the problem
axiom h_alley_length : alley_length = 0.1
axiom h_speed1 : speed1 = 1
axiom h_speed2 : speed2 = 2
axiom h_speed3 : speed3 = 3
axiom h_time_interval : time_interval = 1 / 60 -- 1 minute in hours

-- The theorem stating the question's answer.
theorem all_walking_same_direction :
  ∃ t : ℝ, ∀ t' ∈ Icc t (t + time_interval), 
    (∃ d1 d2 d3 : ℝ,
      d1 = (t' * speed1) % (2 * alley_length) ∧
      d2 = (t' * speed2) % (2 * alley_length) ∧
      d3 = (t' * speed3) % (2 * alley_length) ∧
      sign (d1 - alley_length) = sign (d2 - alley_length) ∧
      sign (d1 - alley_length) = sign (d3 - alley_length)) :=
sorry

end all_walking_same_direction_l0_41


namespace cos_135_eq_neg_sqrt2_div_2_l0_467

theorem cos_135_eq_neg_sqrt2_div_2 : cos (135 * real.pi / 180) = - (real.sqrt 2 / 2) :=
by
  have h1 : 135 = 180 - 45 := by norm_num
  have h2 : cos (180 * real.pi / 180 - 45 * real.pi / 180) = - cos (45 * real.pi / 180) :=
    by rw [cos_sub, cos_pi, cos_pi_div_four, sin_pi, sin_pi_div_four]; norm_num
  have h3 : cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by norm_num
  rw [h1, ←h2, h3]
  norm_num
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_467


namespace solve_and_sum_solutions_l0_848

theorem solve_and_sum_solutions :
  (∀ x : ℝ, x^2 > 9 → 
  (∃ S : ℝ, (∀ x : ℝ, (x / 3 + x / Real.sqrt (x^2 - 9) = 35 / 12) → S = 8.75))) :=
sorry

end solve_and_sum_solutions_l0_848


namespace distance_condition_l0_292

noncomputable def distance_from_O_to_ABC (A B C O : Point) (r AB BC CA : ℝ) : ℝ :=
let K := (1/2) * 24 * 7 in
let R := (AB * BC * CA) / (4 * K) in
let OD := Real.sqrt (r^2 - R^2) in
OD

theorem distance_condition (A B C O : Point) (AB BC CA : ℝ) (OD : ℝ) (r : ℝ)
  (h1 : AB = 7) (h2 : BC = 24) (h3 : CA = 25) (h4 : r = 15) :
  OD = (5 * Real.sqrt 119) / 2 ↔ m + n + k = 126 := by
  let K := (1/2) * 24 * 7
  let R := (AB * BC * CA) / (4 * K)
  have hOD : OD = Real.sqrt (r^2 - R^2)
  sorry

end distance_condition_l0_292


namespace cos_135_eq_neg_sqrt_two_div_two_l0_460

theorem cos_135_eq_neg_sqrt_two_div_two : real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt_two_div_two_l0_460


namespace exists_geometric_sequence_number_l0_96

noncomputable def geometric_sequence_number : ℝ :=
  let x := (1 + Real.sqrt 5) / 2 in
  x

theorem exists_geometric_sequence_number :
  ∃ (x : ℝ), 0 < x ∧
             ∃ (a b c : ℝ), a = Real.floor x ∧ b = (x - Real.floor x) ∧ c = x ∧
             ((b / a) = (c / b) ∧ (a / b) = (b / c)) ∧
             x = geometric_sequence_number :=
by
  sorry

end exists_geometric_sequence_number_l0_96


namespace domain_tan_2x_pi_3_l0_864

theorem domain_tan_2x_pi_3 :
  (∀ x : ℝ, y = tan (2 * x - π / 3) → (∀ k : ℤ, 2 * x - π / 3 ≠ k * π + π / 2) → x ∉ (λ k : ℤ, {x | x = (k * π) / 2 + 5 * π / 12})) :=
sorry

end domain_tan_2x_pi_3_l0_864


namespace cos_135_eq_neg_sqrt2_div_2_l0_509

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = - Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_509


namespace smallest_period_sin_cos_l0_889

theorem smallest_period_sin_cos (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin x + Real.cos x) :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧ T = 2 * Real.pi :=
sorry

end smallest_period_sin_cos_l0_889


namespace quadratic_inequality_solution_l0_651

theorem quadratic_inequality_solution (k : ℝ) :
  (-1 < k ∧ k < 7) ↔ ∀ x : ℝ, x^2 - (k - 5) * x - k + 8 > 0 :=
by
  sorry

end quadratic_inequality_solution_l0_651


namespace trigonometric_identity_l0_427

theorem trigonometric_identity :
  let sin_30 := 1 / 2
  let cos_30 := Real.sqrt 3 / 2
  let tan_60 := Real.sqrt 3
  2 * sin_30 + cos_30 * tan_60 = 5 / 2 :=
by
  let sin_30 := 1 / 2
  let cos_30 := Real.sqrt 3 / 2
  let tan_60 := Real.sqrt 3
  have h1 : 2 * sin_30 = 1 := by norm_num
  have h2 : cos_30 * tan_60 = 3 / 2 := by norm_num
  calc
    2 * sin_30 + cos_30 * tan_60
        = 1 + 3 / 2 : by rw [h1, h2]
    ... = 5 / 2 : by norm_num

end trigonometric_identity_l0_427


namespace new_bucket_capacity_is_9_l0_983

def total_capacity_old := 24 * 13.5
def total_capacity_new (x : ℝ) : ℝ := 36 * x

theorem new_bucket_capacity_is_9 :
  total_capacity_new 9 = total_capacity_old := 
by
  sorry

end new_bucket_capacity_is_9_l0_983


namespace true_propositions_l0_407

-- Definitions used in conditions

-- Proposition (1): Negation of "If \( x^2 + y^2 \ne 0 \), then \( x \) and \( y \) are not both zero"
def proposition1 : Prop := (¬ (x^2 + y^2 ≠ 0) → (x = 0 ∧ y = 0))

-- Proposition (2): Converse of "Similar triangles have equal areas"
def similar_triangles (A B C A' B' C' : Point) : Prop := true -- Placeholder actual definition depends on geometry libraries
def equal_areas (A B C A' B' C' : Point) : Prop := true -- Placeholder actual definition depends on geometry libraries
def proposition2 : Prop := ∀ (A B C A' B' C' : Point), equal_areas A B C A' B' C' → similar_triangles A B C A' B' C'

-- Proposition (3): Contrapositive of "If \( m > 0 \), then the equation \( x^2 + x - m = 0 \) has real roots"
def proposition3 : Prop := (∀ (m : ℝ), m > 0 → (∃ x₁ x₂ : ℝ, x₁^2 + x₁ - m = 0 ∧ x₂^2 + x₂ - m = 0))

-- Proposition (4): Contrapositive of "If \( x - \sqrt{3} \) is a rational number, then \( x \) is an irrational number"
def proposition4 : Prop := (∀ x : ℝ, (∃ r : ℚ, x = r + Real.sqrt 3) → ¬(∃ q : ℚ, x = q))

-- Proof that propositions 1, 3, and 4 are true
theorem true_propositions : proposition1 ∧ proposition3 ∧ proposition4 :=
by
  sorry

end true_propositions_l0_407


namespace cosine_values_l0_769

noncomputable theory

open Real

theorem cosine_values (x y : ℝ)
  (h1 : (cos (3 * x)) / ((2 * cos (2 * x) - 1) * cos y) = (2 / 3) + cos (x - y) ^ 2)
  (h2 : (sin (3 * x)) / ((2 * cos (2 * x) + 1) * sin y) = -(1 / 3) - sin (x - y) ^ 2) :
  cos (x - 3 * y) = -1 ∨ cos (x - 3 * y) = -1 / 3 :=
sorry

end cosine_values_l0_769


namespace cos_135_eq_neg_inv_sqrt_2_l0_521

theorem cos_135_eq_neg_inv_sqrt_2 : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_521


namespace intersection_common_point_l0_413

-- Definitions for quadrilateral, parallelogram, and midpoint
variables {A B C D E F G H E' F' G' H' : Type*}

-- Let Quadrilateral ABCD be inscribed in parallelogram EFGH
-- with sides of EFGH parallel to the diagonals of ABCD.
def inscribed_parallelogram (A B C D E F G H : Type*) :=
  parallelogram E F G H ∧
  parallel (line E F) (diagonal A C) ∧
  parallel (line F G) (diagonal B D)

-- Define Midpoints E', F', G', H'
def midpoints (A B C D E' F' G' H' : Type*) :=
  midpoint E' A B ∧
  midpoint F' B C ∧
  midpoint G' C D ∧
  midpoint H' D A

-- Connecting midpoints to corresponding vertices of the parallelogram
def connect_midpoints (E F G H E' F' G' H' : Type*) :=
  connected E E' ∧
  connected F F' ∧
  connected G G' ∧
  connected H H'

-- The main statement to prove:
theorem intersection_common_point 
  (h_inscribed_parallelogram : inscribed_parallelogram A B C D E F G H)
  (h_midpoints : midpoints A B C D E' F' G' H')
  (h_connect_midpoints : connect_midpoints E F G H E' F' G' H') :
  ∃ O, intersect_at_common_point O (line_segment E E') (line_segment F F') (line_segment G G') (line_segment H H') :=
sorry

end intersection_common_point_l0_413


namespace cos_135_eq_neg_inv_sqrt_2_l0_551

theorem cos_135_eq_neg_inv_sqrt_2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 :=
sorry

end cos_135_eq_neg_inv_sqrt_2_l0_551


namespace proving_C_D_l0_706

variables (a b c : Type)
variables (a b : Vector)
variables (A B M N : Point)

-- Define the given propositions
def proposition_A : Prop := ∀ (u v w : Vector), ∃ (B : Bool), B = False
def proposition_B : Prop := ∀ (a b : Vector), (a ⟂ b → ¬ ∃ (c : Vector), ∀ x : Vector, x = a ∨ x = b ∨ x = c)
def proposition_C : Prop := ∀ (a b : Vector), (a ∥ b → ¬ ∃ (c : Vector), ∀ x : Vector, x = a ∨ x = b ∨ x = c)
def proposition_D : Prop := ∀ (A B M N : Point), (¬ ∃ (u v w : Vector), u = vector_BA(A B) ∧ v = vector_BM(B M) ∧ w = vector_BN(B N) ∧ linearly_independent u v w → coplanar A B M N)

-- State the problem: proving propositions C and D given the conditions
theorem proving_C_D : proposition_C a b ∧ proposition_D A B M N := sorry

end proving_C_D_l0_706


namespace cos_135_eq_neg_sqrt2_div_2_l0_503

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = - Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_503


namespace proof_cos_135_degree_l0_566

noncomputable def cos_135_degree_is_negative_sqrt2_over_2 : Prop :=
  real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2

theorem proof_cos_135_degree : cos_135_degree_is_negative_sqrt2_over_2 :=
by
  sorry

end proof_cos_135_degree_l0_566


namespace find_x_l0_758

noncomputable def x_value : ℝ :=
  let x := 24
  x

theorem find_x (x : ℝ) (h : 7 * x + 3 * x + 4 * x + x = 360) : x = 24 := by
  sorry

end find_x_l0_758


namespace integer_valued_polynomial_of_integer_values_l0_250

-- Define the statement in Lean 4
theorem integer_valued_polynomial_of_integer_values :
  ∀ (m : ℕ) (f : ℤ → ℤ), 
  (∃ (a : ℤ), ∀ i : ℕ, i ≤ m → f (a + i) ∈ ℤ) →
  (∀ x : ℤ, f x ∈ ℤ) :=
begin
  sorry 
end

end integer_valued_polynomial_of_integer_values_l0_250


namespace average_increase_l0_216

def first_four_scores : List ℕ := [85, 90, 82, 89]
def fifth_score : ℕ := 92

def average (scores : List ℕ) : ℚ :=
  (scores.foldl (+) 0 : ℚ) / scores.length

theorem average_increase :
  average first_four_scores = (85 + 90 + 82 + 89) / 4 ∧
  average (first_four_scores ++ [fifth_score]) = (85 + 90 + 82 + 89 + 92) / 5 →
  ( (average (first_four_scores ++ [fifth_score]) - average first_four_scores) = 1.1 ) :=
by
  intros h
  sorry 

end average_increase_l0_216


namespace cos_135_eq_neg_sqrt_two_div_two_l0_457

theorem cos_135_eq_neg_sqrt_two_div_two : real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt_two_div_two_l0_457


namespace cos_135_eq_neg_inv_sqrt_2_l0_530

theorem cos_135_eq_neg_inv_sqrt_2 : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_530


namespace find_LCM_of_numbers_l0_308

def HCF (a b : ℕ) : ℕ := sorry  -- A placeholder definition for HCF
def LCM (a b : ℕ) : ℕ := sorry  -- A placeholder definition for LCM

theorem find_LCM_of_numbers (a b : ℕ) 
  (h1 : a + b = 55) 
  (h2 : HCF a b = 5) 
  (h3 : (1 : ℚ) / a + (1 : ℚ) / b = 0.09166666666666666) : 
  LCM a b = 120 := 
by 
  sorry

end find_LCM_of_numbers_l0_308


namespace largest_p_q_sum_l0_952

theorem largest_p_q_sum 
  (p q : ℝ)
  (A := (p, q))
  (B := (12, 19))
  (C := (23, 20))
  (area_ABC : ℝ := 70)
  (slope_median : ℝ := -5)
  (midpoint_BC := ((12 + 23) / 2, (19 + 20) / 2))
  (eq_median : (q - midpoint_BC.2) = slope_median * (p - midpoint_BC.1))
  (area_eq : 140 = 240 - 437 - 20 * p + 23 * q + 19 * p - 12 * q) :
  p + q ≤ 47 :=
sorry

end largest_p_q_sum_l0_952


namespace inequality_solution_set_l0_298

theorem inequality_solution_set : 
  { x : ℝ | x^2 + x - 2 < 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end inequality_solution_set_l0_298


namespace T_perimeter_l0_874

theorem T_perimeter (l w : ℝ) (h1 : l = 4) (h2 : w = 2) :
  let rect_perimeter := 2 * l + 2 * w
  let overlap := 2 * w
  2 * rect_perimeter - overlap = 20 :=
by
  -- Proof will be added here
  sorry

end T_perimeter_l0_874


namespace positive_difference_of_two_numbers_l0_915

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l0_915


namespace sequence_20th_term_is_6_over_7_l0_296

-- Define the sequence
def sequence_term (n k : ℕ) : ℚ :=
  if k ≤ n then (k : ℚ) / (n + 1 : ℚ) else 0

-- Find the n-th term in the sequence
noncomputable def nth_term (n : ℕ) : ℚ :=
  let m := Nat.find (λ m => ((m * (m + 1)) / 2 : ℕ) >= n)
  let term_index := n - (m * (m - 1) / 2 : ℕ)
  (term_index : ℚ) / (m + 1 : ℚ)

-- Statement of the problem
theorem sequence_20th_term_is_6_over_7 : nth_term 20 = 6 / 7 :=
by
  sorry

end sequence_20th_term_is_6_over_7_l0_296


namespace min_c_plus_3d_l0_781

theorem min_c_plus_3d (c d : ℝ) (hc : 0 < c) (hd : 0 < d) 
    (h1 : c^2 ≥ 12 * d) (h2 : 9 * d^2 ≥ 4 * c) : 
  c + 3 * d ≥ 8 :=
  sorry

end min_c_plus_3d_l0_781


namespace largest_possible_b_is_12_l0_943

open Nat

noncomputable def largest_b (a b c : ℕ) : ℕ :=
  if 1 < c ∧ c < b ∧ b < a ∧ prime c ∧ a * b * c = 360 then b else 0

theorem largest_possible_b_is_12 : ∃ a c, 1 < c ∧ c < 12 ∧ 12 < a ∧ prime c ∧ a * 12 * c = 360 :=
begin
  -- Since the proof is not required, we add sorry here.
  sorry
end

end largest_possible_b_is_12_l0_943


namespace edward_initial_money_l0_81

theorem edward_initial_money (spent_books : ℕ) (spent_pens : ℕ) (remaining_money : ℕ) :
  spent_books = 6 → spent_pens = 16 → remaining_money = 19 →
  spent_books + spent_pens + remaining_money = 41 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num

end edward_initial_money_l0_81


namespace relationship_among_abc_l0_668

theorem relationship_among_abc : 
  let a := (0.7: ℝ) ^ (0.8: ℝ)
  let b := Real.log 0.8 / Real.log 2
  let c := (1.1: ℝ) ^ (0.8: ℝ)
  in b < a ∧ a < c :=
by 
  let a := (0.7: ℝ) ^ (0.8: ℝ)
  let b := Real.log 0.8 / Real.log 2
  let c := (1.1: ℝ) ^ (0.8: ℝ)
  sorry

end relationship_among_abc_l0_668


namespace share_of_C_l0_837

variable (A B C x : ℝ)

theorem share_of_C (hA : A = (2/3) * B) 
(hB : B = (1/4) * C) 
(hTotal : A + B + C = 595) 
(hC : C = x) : x = 420 :=
by
  -- Proof will follow here
  sorry

end share_of_C_l0_837


namespace six_digit_numbers_count_l0_948

theorem six_digit_numbers_count :
  (∑ x in Finset.univ.filter (λ x, 
    let odd_candidates := [1, 3, 5, 7, 9],
    let positions := Finset.range 6,
    let zero_positions := Finset.range 5, -- zero cannot be in the first position
    let two_positions := Finset.range 5,
    let odd_positions := Finset.range 3,
    let zero_ways := 5,
    let two_ways := Nat.choose 5 2,
    let odd_count := Nat.choose 5 3,
    let odd_arrangements := Nat.factorial 3 in
    zero_ways * two_ways * odd_count * odd_arrangements = 3000)
  ) = 3000 := sorry

end six_digit_numbers_count_l0_948


namespace solution_set_of_inequality_l0_305

theorem solution_set_of_inequality :
  { x : ℝ | x^2 + x - 2 < 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end solution_set_of_inequality_l0_305


namespace cos_135_degree_l0_489

theorem cos_135_degree : 
  let Q := (Complex.exp (135 * Real.pi / 180 * Complex.I)) in 
  Complex.re Q = -Real.sqrt 2 / 2 :=
by sorry

end cos_135_degree_l0_489


namespace cos_135_eq_neg_inv_sqrt_2_l0_585

theorem cos_135_eq_neg_inv_sqrt_2 :
  cos (135 * Real.pi / 180) = - (1 / Real.sqrt 2) :=
by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_585


namespace solution_set_l0_883

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' (x : ℝ) : ℝ := sorry
def π : ℝ := Real.pi

axiom odd_f : ∀ x : ℝ, f x = -f (-x)
axiom domain_f : ∀ x : ℝ, (x ∈ Ioo (-π) 0 ∨ x ∈ Ioo 0 π) → true
axiom deriv_condition : ∀ x : ℝ, (0 < x ∧ x < π) → (f' x) * (Real.sin x) - (f x) * (Real.cos x) < 0

theorem solution_set :
  {x : ℝ | f x < Real.sqrt 2 * f (π/4) * (Real.sin x)} = Ioo (-π/4) 0 ∪ Ioo (π/4) π :=
sorry

end solution_set_l0_883


namespace cos_135_eq_neg_sqrt2_div_2_l0_469

theorem cos_135_eq_neg_sqrt2_div_2 : cos (135 * real.pi / 180) = - (real.sqrt 2 / 2) :=
by
  have h1 : 135 = 180 - 45 := by norm_num
  have h2 : cos (180 * real.pi / 180 - 45 * real.pi / 180) = - cos (45 * real.pi / 180) :=
    by rw [cos_sub, cos_pi, cos_pi_div_four, sin_pi, sin_pi_div_four]; norm_num
  have h3 : cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by norm_num
  rw [h1, ←h2, h3]
  norm_num
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_469


namespace visible_length_red_bus_l0_279

-- Definitions from the problem
def length_red_bus : ℝ := 48
def length_orange_car : ℝ := length_red_bus / 4
def length_yellow_bus : ℝ := length_orange_car * 3.5

-- Proof statement
theorem visible_length_red_bus : (length_red_bus - length_yellow_bus) = 6 := by
  have h1 : length_orange_car = 12 := by sorry
  have h2 : length_yellow_bus = 42 := by sorry
  have h3 : length_red_bus = 48 := by sorry
  sorry

end visible_length_red_bus_l0_279


namespace infinite_solutions_pos_int_description_of_solutions_l0_630

-- Conditions
def equation (x y : ℕ) : Prop := (3 * x ^ 3 + x * y ^ 2) * (x ^ 2 * y + 3 * y ^ 3) = (x - y) ^ 7

-- Part (a)
theorem infinite_solutions_pos_int : ∃^∞ (x y : ℕ), x > 0 ∧ y > 0 ∧ equation x y := 
sorry

-- Part (b)
theorem description_of_solutions :
  ∀ (x y : ℕ), (x > 0 ∧ y > 0 ∧ equation x y) ↔
  (∃ n : ℕ, y = n ^ 7 ∧ x = (1 + 1 / n) * y) := 
sorry

end infinite_solutions_pos_int_description_of_solutions_l0_630


namespace prove_axisymmetric_char4_l0_344

-- Predicates representing whether a character is an axisymmetric figure
def is_axisymmetric (ch : Char) : Prop := sorry

-- Definitions for the conditions given in the problem
def char1 := '月'
def char2 := '右'
def char3 := '同'
def char4 := '干'

-- Statement that needs to be proven
theorem prove_axisymmetric_char4 (h1 : ¬ is_axisymmetric char1) 
                                  (h2 : ¬ is_axisymmetric char2) 
                                  (h3 : ¬ is_axisymmetric char3) : 
                                  is_axisymmetric char4 :=
sorry

end prove_axisymmetric_char4_l0_344


namespace order_of_fractions_l0_109

-- Define the conditions and prove the required inequality.
theorem order_of_fractions 
  (x y z : ℝ)
  (h1 : log 2 x = log 3 y)
  (h2 : log 3 y = log 5 z)
  (h3 : log 2 x < 0) :
  2 / x < 3 / y ∧ 3 / y < 5 / z := 
by
  sorry

end order_of_fractions_l0_109


namespace num_correct_statements_l0_691

variable {S : ℕ → ℝ} {d : ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def S_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n * (2*a 1 + (n-1)*d)) / 2

theorem num_correct_statements (S : ℕ → ℝ) (d : ℝ) (h1 : ∀ n, S n = (n * (2*(S 1) + (n-1)*d)) / 2)
  (h2 : S 5 < S 6) (h3 : S 6 > S 7) : 
  (d < 0) ∧ 
  (∀ n, S 6 ≥ S n) ∧ 
  (S 11 > 0) ∧ 
  ¬ (S 12 < 0) :=
by 
    sorry

end num_correct_statements_l0_691


namespace find_potential_l0_89

noncomputable def vector_field := sorry -- Define the vector field \(\mathbf{a}\) here
noncomputable def potential (ρ φ z : ℝ) := arctan z * log ρ + ρ * cos φ + C

theorem find_potential : 
  (∇ (λ (ρ φ z : ℝ), potential ρ φ z) = vector_field) :=
sorry

end find_potential_l0_89


namespace negation_of_universal_quantifier_l0_288

theorem negation_of_universal_quantifier :
  (∀ (q : Type) [quadrilateral q], has_circumcircle q) ↔ ∃ (q : Type) [quadrilateral q], ¬has_circumcircle q := 
sorry

end negation_of_universal_quantifier_l0_288


namespace positive_difference_of_two_numbers_l0_940

theorem positive_difference_of_two_numbers (x y : ℝ)
  (h1 : x + y = 10) (h2 : x ^ 2 - y ^ 2 = 40) : abs (x - y) = 4 := 
by
  sorry

end positive_difference_of_two_numbers_l0_940


namespace probability_of_odd_numbers_l0_336

def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_of_5_odd_numbers_in_7_rolls : ℚ :=
  (binomial 7 5) * (1 / 2)^5 * (1 / 2)^(7-5)

theorem probability_of_odd_numbers :
  probability_of_5_odd_numbers_in_7_rolls = 21 / 128 :=
by
  sorry

end probability_of_odd_numbers_l0_336


namespace mass_percentage_O_in_C6H8O6_l0_86

theorem mass_percentage_O_in_C6H8O6 :
  let atomic_mass_C := 12.01
  let atomic_mass_H := 1.01
  let atomic_mass_O := 16.00
  let molar_mass_C6H8O6 := (6 * atomic_mass_C) + (8 * atomic_mass_H) + (6 * atomic_mass_O)
  let mass_of_oxygen := 8 * atomic_mass_O
  let mass_percentage_O := (mass_of_oxygen / molar_mass_C6H8O6) * 100
  mass_percentage_O = 72.67 :=
by
  -- Definitions
  let atomic_mass_C := 12.01
  let atomic_mass_H := 1.01
  let atomic_mass_O := 16.00
  let molar_mass_C6H8O6 := (6 * atomic_mass_C) + (8 * atomic_mass_H) + (6 * atomic_mass_O)
  let mass_of_oxygen := 8 * atomic_mass_O
  let mass_percentage_O := (mass_of_oxygen / molar_mass_C6H8O6) * 100
  -- Proof
  sorry

end mass_percentage_O_in_C6H8O6_l0_86


namespace positive_difference_of_two_numbers_l0_931

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l0_931


namespace cos_135_degree_l0_491

theorem cos_135_degree : 
  let Q := (Complex.exp (135 * Real.pi / 180 * Complex.I)) in 
  Complex.re Q = -Real.sqrt 2 / 2 :=
by sorry

end cos_135_degree_l0_491


namespace drink_total_amount_l0_209

theorem drink_total_amount (parts_coke parts_sprite parts_mountain_dew ounces_coke total_parts : ℕ)
  (h1 : parts_coke = 2) (h2 : parts_sprite = 1) (h3 : parts_mountain_dew = 3)
  (h4 : total_parts = parts_coke + parts_sprite + parts_mountain_dew)
  (h5 : ounces_coke = 6) :
  ( ounces_coke * total_parts ) / parts_coke = 18 :=
by
  sorry

end drink_total_amount_l0_209


namespace cos_135_degree_l0_501

theorem cos_135_degree : 
  let Q := (Complex.exp (135 * Real.pi / 180 * Complex.I)) in 
  Complex.re Q = -Real.sqrt 2 / 2 :=
by sorry

end cos_135_degree_l0_501


namespace cosine_135_eq_neg_sqrt_2_div_2_l0_538

theorem cosine_135_eq_neg_sqrt_2_div_2 (cos180 : Real := -1) (cos90 : Real := 0) (cos45 : Real := Real.sqrt 2 / 2)
    (theta1 : Real := 180) (theta2 : Real := 45) (theta3 : Real := 135) :
    theta3 = theta1 - theta2 ∧ 
    Real.cos theta1 = cos180 ∧ 
    Real.cos theta2 = cos45 → 
    Real.cos theta3 = -cos45 := by
  sorry

end cosine_135_eq_neg_sqrt_2_div_2_l0_538


namespace triangle_area_l0_180

open Real

-- Define the parabola and its focus
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus_of_parabola : ℝ × ℝ := (1, 0)

-- Define the points A, B and O such that O is the origin and A, B lie on the parabola
def origin : ℝ × ℝ := (0, 0)
def point_A (a : ℝ) : ℝ × ℝ := (a^2, 2 * a)
def point_B (a : ℝ) : ℝ × ℝ := (a^2, -2 * a)

-- Define the condition that the orthocenter of triangle OAB is the focus of the parabola
def orthocenter_condition (a : ℝ) : Prop :=
  let O := origin in
  let A := point_A a in
  let B := point_B a in
  let F := focus_of_parabola in
  -- Some condition linking orthocenter to focus (mathematical detail omitted here)
  sorry

-- Prove that if the orthocenter condition holds, the area of ΔOAB is 10√5
theorem triangle_area (a : ℝ) 
  (hA : parabola (point_A a).fst (point_A a).snd)
  (hB : parabola (point_B a).fst (point_B a).snd)
  (hOrthocenter : orthocenter_condition a) : 
  let O := origin in
  let A := point_A a in
  let B := point_B a in
  let AB := dist A B in
  let height := (dist O (a^2, 0)) in
  0.5 * AB * height = 10 * sqrt 5 := 
by
  -- Lean doesn't directly assist with concluding real numbers calculations, so we use 'sorry' 
  sorry

end triangle_area_l0_180


namespace cos_135_degree_l0_500

theorem cos_135_degree : 
  let Q := (Complex.exp (135 * Real.pi / 180 * Complex.I)) in 
  Complex.re Q = -Real.sqrt 2 / 2 :=
by sorry

end cos_135_degree_l0_500


namespace journey_time_ratio_l0_370

theorem journey_time_ratio (D : ℝ) (h₁ : D > 0) :
  let T1 := D / 48
  let T2 := D / 32
  (T2 / T1) = 3 / 2 :=
by
  sorry

end journey_time_ratio_l0_370


namespace cos_135_eq_neg_sqrt2_div_2_l0_516

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = - Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_516


namespace cos_135_eq_neg_inv_sqrt_2_l0_529

theorem cos_135_eq_neg_inv_sqrt_2 : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_529


namespace distance_PQ_ge_5_l0_816

-- Definitions based on conditions
def point (P : Type) : Prop := ∃ Q : Type, True

-- Assuming P lies on the angle bisector of AOB
def on_angle_bisector (P : Type) (OA OB : Type → Prop) : Prop := 
  ∀ Q, (PQ_to_OA P OA Q) = (PQ_to_OB P OB Q)

-- P lies on angle bisector of ∠AOB
axiom P_on_angle_bisector : ∀ (P OA OB : Type), on_angle_bisector P OA OB

-- Distance from point P to side OA is 5
def dist_to_side_OA (P OA : Type) : ℝ := 5

-- Point Q is any point on side OB
def point_on_OB (Q OB : Type → Prop) : Prop := OB Q

-- Goal is to prove that for any point Q on side OB, distance PQ is ≥ 5
theorem distance_PQ_ge_5 (P Q OA OB : Type) 
  (P_bisector : on_angle_bisector P OA OB) (dist_P_OA : dist_to_side_OA P OA) 
  (Q_on_OB : point_on_OB Q OB) : dist PQ P Q ≥ 5 := 
by 
  sorry -- Proof to be filled in

end distance_PQ_ge_5_l0_816


namespace delivery_order_count_l0_264

-- Define the groups and hotels
def Group := {1, 2, 3, 4, 5}
def Hotel := {"Druzhba", "Rossiya", "Minsk", "FourthGroup", "FifthGroup"}

-- Number of ways to deliver considering the constraints
theorem delivery_order_count : 
  let total_permutations := Nat.factorial 5 in
  let valid_permutations := total_permutations / 2 / 2 in
  valid_permutations = 30 :=
by
  sorry

end delivery_order_count_l0_264


namespace smallest_possible_value_of_e_l0_115

noncomputable def polynomial_with_roots (a b c d e : ℤ) (x : ℂ) : Prop :=
  x^4 + b * x^3 + c * x^2 + d * x + e = 0

noncomputable def integer_roots (p : ℤ) : Prop :=
  p = -3 ∨ p = 7 ∨ p = 11 ∨ p = -1/4

noncomputable def positive_integer (e : ℤ) : Prop :=
  0 < e

noncomputable def product_of_roots (a e : ℤ) : Prop :=
  e = a * (-3) * 7 * 11 * (-1/4)

theorem smallest_possible_value_of_e :
  ∃ e : ℤ, polynomial_with_roots 1 _ _ _ e ∧ 
            positive_integer e ∧ 
            e = 231 :=
by
  sorry

end smallest_possible_value_of_e_l0_115


namespace find_vector_l0_74

noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (2 + 3 * t, 3 + 2 * t)

noncomputable def line_m (s : ℝ) : ℝ × ℝ :=
  (-4 + 3 * s, 5 + 2 * s)

def vector_condition (v1 v2 : ℝ) : Prop :=
  v1 - v2 = 1

theorem find_vector :
  ∃ (v1 v2 : ℝ), vector_condition v1 v2 ∧ (v1, v2) = (3, 2) :=
sorry

end find_vector_l0_74


namespace positive_difference_of_two_numbers_l0_903

theorem positive_difference_of_two_numbers 
  (a b : ℝ) 
  (h₁ : a + b = 10) 
  (h₂ : a^2 - b^2 = 40) : 
  |a - b| = 4 := 
sorry

end positive_difference_of_two_numbers_l0_903


namespace cos_135_eq_correct_l0_607

noncomputable def cos_135_eq: Prop :=
  real.cos ((135 * real.pi) / 180) = -real.sqrt 2 / 2

theorem cos_135_eq_correct : cos_135_eq :=
  sorry

end cos_135_eq_correct_l0_607


namespace range_of_f_l0_262

noncomputable def f : ℝ → ℝ := λ x, 2^x + 3

theorem range_of_f : set.range (λ x : {x : ℝ // 0 < x}, f x) = set.Ici 4 := by
  sorry

end range_of_f_l0_262


namespace min_even_integers_l0_323

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem min_even_integers (x y a b m n : ℤ)
  (h1 : x + y = 24)
  (h2 : x + y + a + b = 39)
  (h3 : x + y + a + b + m + n = 58) :
  ∃ e1 e2 : ℤ, (is_even e1 ∧ is_even e2) ∧ 
  ({x, y, a, b, m, n} - {e1, e2} ⊆ {is_odd}) :=
sorry

end min_even_integers_l0_323


namespace increase_in_circumference_by_2_cm_l0_267

noncomputable def radius_increase_by_two (r : ℝ) : ℝ := r + 2
noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

theorem increase_in_circumference_by_2_cm (r : ℝ) : 
    circumference (radius_increase_by_two r) - circumference r = 12.56 :=
by sorry

end increase_in_circumference_by_2_cm_l0_267


namespace O_is_center_of_prism_l0_197

noncomputable def prism_center (n : ℕ) (h : n ≥ 3) : Prop :=
  ∀ (p : RegularPrism n) (O : Point)
    (H : ∃ A B C A_1 B_1 C_1 : Point, 
      ∃ (d₁ : Diagonal p A A_1) (d₂ : Diagonal p B B_1) (d₃ : Diagonal p C C_1), 
      intersect_at d₁ d₂ d₃ O), 
    ∀ V : Point, (V ∈ vertices_of_prism p) → dist O V = dist O (some_vertex p)

-- The theorem we need to prove
theorem O_is_center_of_prism (n : ℕ) (h : n ≥ 3) :
  prism_center n h :=
sorry

end O_is_center_of_prism_l0_197


namespace cos_135_eq_neg_sqrt2_div_2_l0_593

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  -- Conditions incorporated directly into the requirement due to the nature of trigonometric evaluations.
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_593


namespace dot_product_a_b_equals_neg5_l0_721

-- Defining vectors and conditions
structure vector2 := (x : ℝ) (y : ℝ)

def a : vector2 := ⟨2, 1⟩
def b (x : ℝ) : vector2 := ⟨x, -1⟩

-- Collinearity condition
def parallel (v w : vector2) : Prop :=
  v.x * w.y = v.y * w.x

-- Dot product definition
def dot_product (v w : vector2) : ℝ :=
  v.x * w.x + v.y * w.y

-- Given condition
theorem dot_product_a_b_equals_neg5 (x : ℝ) (h : parallel a ⟨a.x - x, a.y - (-1)⟩) : dot_product a (b x) = -5 :=
sorry

end dot_product_a_b_equals_neg5_l0_721


namespace solution_set_l0_634

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ) -- Function for the derivative of f

axiom f_deriv : ∀ x, f' x = (deriv f) x

axiom f_condition1 : ∀ x, f x > 1 - f' x
axiom f_condition2 : f 0 = 0
  
theorem solution_set (x : ℝ) : (e^x * f x > e^x - 1) ↔ (x > 0) := 
  sorry

end solution_set_l0_634


namespace equation_of_line_l0_701

theorem equation_of_line (l : ℝ → ℝ) :
  (l 1 = 2 ∧ (∃ a : ℝ, l 0 = 2 * a ∧ a ≠ 0 ∧ ∀ x : ℝ, l x = (2 * l a / a) * x))
  ∨ (l 1 = 2 ∧ (∃ a : ℝ, l 0 = 2 * a ∧ a ≠ 0 ∧ ∀ x y : ℝ, 2 * x + y - 4 = 0)) := sorry

end equation_of_line_l0_701


namespace range_b_minus_a_l0_143

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem range_b_minus_a (a b : ℝ) (h : ∀ y ∈ Icc a b, f y ∈ Icc (-1 : ℝ) 3) :
  2 ≤ b - a ∧ b - a ≤ 4 :=
sorry

end range_b_minus_a_l0_143


namespace problem_l0_289

theorem problem (a b c : ℝ) (h1 : a > b) (h2 : (a - b) * (b - c) * (c - a) > 0) : a > c :=
sorry

end problem_l0_289


namespace Rachel_average_speed_l0_831

noncomputable def total_distance : ℝ := 2 + 4 + 6

noncomputable def time_to_Alicia : ℝ := 2 / 3
noncomputable def time_to_Lisa : ℝ := 4 / 5
noncomputable def time_to_Nicholas : ℝ := 1 / 2

noncomputable def total_time : ℝ := (20 / 30) + (24 / 30) + (15 / 30)

noncomputable def average_speed : ℝ := total_distance / total_time

theorem Rachel_average_speed : average_speed = 360 / 59 :=
by
  sorry

end Rachel_average_speed_l0_831


namespace train_speed_l0_970

def length_of_train : ℝ := 160
def time_to_cross : ℝ := 18
def speed_in_kmh : ℝ := 32

theorem train_speed :
  (length_of_train / time_to_cross) * 3.6 = speed_in_kmh :=
by
  sorry

end train_speed_l0_970


namespace sum_of_squares_of_real_roots_of_equation_l0_661

theorem sum_of_squares_of_real_roots_of_equation :
  (∃ (x : ℝ), x^256 - 256^32 = 0) → 
  (∑ r in {x : ℝ | x^256 - 256^32 = 0}, r^2) = 8 :=
by
  sorry

end sum_of_squares_of_real_roots_of_equation_l0_661


namespace solve_for_x_l0_842

-- Defining the primary condition.
def equation (x : ℝ) : Prop := 2^(x + 5) = 288

-- Lean statement to prove the value of x that satisfies the equation.
theorem solve_for_x : ∃ x : ℝ, equation x ∧ x ≈ 3.17 := 
by sorry

end solve_for_x_l0_842


namespace cos_135_eq_neg_inv_sqrt_2_l0_552

theorem cos_135_eq_neg_inv_sqrt_2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 :=
sorry

end cos_135_eq_neg_inv_sqrt_2_l0_552


namespace arctan_cos_solution_l0_844

theorem arctan_cos_solution (x : ℝ) (hx : -real.pi ≤ x ∧ x ≤ real.pi) : 
  (∃ y : ℝ, y = arccos (real.sqrt ((-1 + real.sqrt 5) / 2)) ∧ (x = y ∨ x = -y)) ↔ arctan (cos x) = x / 3 :=
sorry

end arctan_cos_solution_l0_844


namespace cos_135_eq_neg_sqrt2_div_2_l0_597

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  -- Conditions incorporated directly into the requirement due to the nature of trigonometric evaluations.
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_597


namespace max_correct_answers_l0_17

theorem max_correct_answers :
  ∃ (c w b : ℤ), 
      c + w + b = 30 ∧ 
      4 * c - 3 * w = 54 ∧ 
      14 ≤ c ∧ 
      c ≤ 20 ∧ 
      ∀ x, (14 ≤ x ∧ x ≤ 20 ∧ ∃ wx bx, x + wx + bx = 30 ∧ 4 * x - 3 * wx = 54) → x ≤ c :=
begin
  use [20, 6, 4],
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  sorry
end

end max_correct_answers_l0_17


namespace eleonora_age_l0_979

-- Definitions
def age_eleonora (e m : ℕ) : Prop :=
m - e = 3 * (2 * e - m) ∧ 3 * e + (m + 2 * e) = 100

-- Theorem stating that Eleonora's age is 15
theorem eleonora_age (e m : ℕ) (h : age_eleonora e m) : e = 15 :=
sorry

end eleonora_age_l0_979


namespace probability_of_odd_numbers_l0_334

def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_of_5_odd_numbers_in_7_rolls : ℚ :=
  (binomial 7 5) * (1 / 2)^5 * (1 / 2)^(7-5)

theorem probability_of_odd_numbers :
  probability_of_5_odd_numbers_in_7_rolls = 21 / 128 :=
by
  sorry

end probability_of_odd_numbers_l0_334


namespace seq_arithmetic_product_position_l0_117

noncomputable def a : ℕ → ℝ
| 0 := 1/5
| n + 1 := if h : n + 1 > 0 then
  let a_nm1 := a n in
  1 / (4 * (n + 1) + 1)
else 0

theorem seq_arithmetic : ∀ n : ℕ, n > 0 → (a n) ≠ 0 →
  1 / (a n) - 1 / (a (n - 1)) = 4 := 
by
  intro n hn hne
  induction n with
  | zero => contradiction
  | succ n ih => sorry

theorem product_position : a 1 * a 2 = a 11 :=
by
  sorry

end seq_arithmetic_product_position_l0_117


namespace graph_symmetry_l0_276

noncomputable def f (x : ℝ) := Real.log x / Real.log 4
noncomputable def g (x : ℝ) := 4 ^ x

theorem graph_symmetry :
  ∀ x y : ℝ, (y = f x) ↔ (x = g y) :=
by
  -- sorry serves as a placeholder for the proof
  sorry

end graph_symmetry_l0_276


namespace grapefruit_orchards_l0_998

theorem grapefruit_orchards (total_orchards lemons_orchards oranges_factor remaining_orchards : ℕ) 
    (H1 : total_orchards = 16)
    (H2 : lemons_orchards = 8)
    (H3 : oranges_factor = 2)
    (H4 : oranges_orchards = lemons_orchards / oranges_factor)
    (H5 : remaining_orchards = total_orchards - lemons_orchards - oranges_orchards)
    (H6 : grapefruit_orchards = remaining_orchards / 2) : 
  grapefruit_orchards = 2 := by
  sorry

end grapefruit_orchards_l0_998


namespace projection_distance_range_l0_138

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

def is_projection (M P : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  ∃ Q, l Q.1 Q.2 ∧ (M.1 = (P.1 + Q.1) / 2) ∧ (M.2 = (P.2 + Q.2) / 2)

def arithmetic_seq (a b c : ℝ) : Prop :=
  2 * b = a + c

theorem projection_distance_range (a b c : ℝ) (P : ℝ × ℝ) (N : ℝ × ℝ) :
  arithmetic_seq a b c →
  P = (-3, 0) →
  N = (2, 3) →
  ∃ M, is_projection M P (λ x y, a * x + b * y + c = 0) ∧ 5 - real.sqrt 5 ≤ distance M N ∧ distance M N ≤ 5 + real.sqrt 5 :=
by
  sorry

end projection_distance_range_l0_138


namespace hyperbola_eccentricity_l0_153

-- Defining the conditions given in the problem
variable (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)

-- Defining the condition that asymptote equations are given by y = ±2x
def asymptote_condition : Prop := b = 2 * a

-- Defining the formula for the eccentricity of a hyperbola
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + (b / a) ^ 2)

-- The theorem to be proved: Given the hyperbola and its conditions, prove its eccentricity is √5
theorem hyperbola_eccentricity (h_asymptote : asymptote_condition a b) : eccentricity a b = Real.sqrt 5 :=
by
  -- This 'by' block is empty and will typically contain the proof
  sorry

end hyperbola_eccentricity_l0_153


namespace cos_135_eq_neg_sqrt2_div_2_l0_588

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  -- Conditions incorporated directly into the requirement due to the nature of trigonometric evaluations.
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_588


namespace group_division_l0_768

theorem group_division :
  ∃ (A B : Finset ℕ), 
    A.card = 4 ∧
    B.card = 4 ∧ 
    A ∪ B = {1, 2, 3, 4, 5, 6, 7, 8} ∧ 
    A ∩ B = ∅ ∧ 
    (A.sum id = B.sum id) ∧ 
    (A.sum (λ x, x ^ 2) = B.sum (λ x, x ^ 2)) :=
sorry

end group_division_l0_768


namespace B_work_days_eq_40_div_7_l0_980

-- Definitions
def A_work_days : ℝ := 10
def A_work_rate : ℝ := 1 / A_work_days
def efficiency_factor : ℝ := 1.75
def B_work_rate : ℝ := efficiency_factor * A_work_rate

-- Proof statement
theorem B_work_days_eq_40_div_7 : (1 / B_work_rate) = 40 / 7 :=
by
  -- We skip the proof here
  sorry

end B_work_days_eq_40_div_7_l0_980


namespace divisors_with_different_parity_l0_358

/--
Let n be a positive integer, g(n) be the number of positive divisors of n 
of the form 6k + 1 and h(n) be the number of positive divisors of n of 
the form 6k - 1, where k is a nonnegative integer.
Prove that if g(n) and h(n) have different parity, 
then n must be of the form n = 2^a * 3^b * t^2 where 
a and b are nonnegative integers and t is an integer.
-/
theorem divisors_with_different_parity (n : ℕ) 
  (g h : ℕ → ℕ)
  (g_def : ∀ (n : ℕ), g n = (nat.divisors n).count (λ d, ∃ k, d = 6*k + 1))
  (h_def : ∀ (n : ℕ), h n = (nat.divisors n).count (λ d, ∃ k, d = 6*k - 1))
  (g_h_parity_diff : (g n % 2) ≠ (h n % 2)) : 
  ∃ (a b : ℕ) (t : ℤ), n = 2^a * 3^b * (t^2).natAbs :=
sorry

end divisors_with_different_parity_l0_358


namespace parallel_lines_in_plane_l0_664

variables {α : Type} [plane α] (m n : line α) 

-- Define the properties and relationships between m, n, and α.
def perpendicular (l1 l2 : line α) : Prop := -- Definition for l1 perpendicular to l2
sorry

def parallel (l1 l2 : line α) : Prop := -- Definition for l1 parallel to l2
sorry

def subset (l : line α) (p : plane α) : Prop := -- Definition for l ⊂ p
sorry

-- The main theorem to be proved
theorem parallel_lines_in_plane
  (h1 : subset m α)
  (h2 : parallel n α) :
  parallel m n :=
sorry

end parallel_lines_in_plane_l0_664


namespace sec_315_eq_sqrt2_l0_649

theorem sec_315_eq_sqrt2 : sec 315 = √2 :=
  by
    -- Definitions and conditions
    let θ := 315
    have h1 : sec θ = 1 / cos θ := by sorry
    have h2 : cos (360 - 45) = cos 45 := by sorry
    have h3 : cos 45 = sqrt(2) / 2 := by sorry
    -- Proof leads to the conclusion
    sorry -- replace this with the actual proof

end sec_315_eq_sqrt2_l0_649


namespace scientific_notation_of_neg_small_num_l0_646

-- Define the numbers involved in the problem
def num : ℝ := -0.0000406
def number_in_scientific_notation (a : ℝ) (n : ℤ) : Prop := num = a * 10 ^ n

-- State the theorem to be proven
theorem scientific_notation_of_neg_small_num :
  ∃ (a : ℝ) (n : ℤ), number_in_scientific_notation a n ∧ 1 ≤ |a| ∧ |a| < 10 :=
begin
  use -4.06,
  use -5,
  split,
  { -- Prove the number in scientific notation is equal to the original number
    rw number_in_scientific_notation,
    norm_num,
  },
  split,
  { -- Prove the absolute value of the significant figure is at least 1
    norm_num,
  },
  { -- Prove the absolute value of the significant figure is less than 10
    norm_num,
  }
end

end scientific_notation_of_neg_small_num_l0_646


namespace parabola_equation_circle_equation_l0_716

theorem parabola_equation (p : ℝ) (h : p > 0)
  (C : ∀ (x y : ℝ), y^2 = 2 * p * x)
  (l : ∀ (x y : ℝ), y = x - 1)
  (A B : ℝ × ℝ) (h_intersect : C A.1 A.2 ∧ C B.1 B.2 ∧ l A.1 A.2 ∧ l B.1 B.2)
  (h_AB : dist A B = 8) :
  ∃ p, C = (λ x y, y^2 = 4 * x) := 
begin
  sorry
end

theorem circle_equation (C : ∀ (x y : ℝ), y^2 = 4 * x)
  (l : ∀ (x y : ℝ), y = x - 1)
  (A B : ℝ × ℝ) (h_intersect : C A.1 A.2 ∧ C B.1 B.2 ∧ l A.1 A.2 ∧ l B.1 B.2)
  (h_AB : dist A B = 8)
  (h_tangent : ∀ (x₀ y₀ : ℝ), y₀ = -x₀ + 5 → (x₀ + 1)^2 = (y₀ - x₀ + 1)^2 / 2 + 16) :
  (∀ x y, (x - 3)^2 + (y - 2)^2 = 16) ∨ (∀ x y, (x - 11)^2 + (y + 6)^2 = 144) :=
begin
  sorry
end

end parabola_equation_circle_equation_l0_716


namespace find_a_find_sin_A_l0_744

-- Definitions of the given problem conditions
variables {A B C : Type}
variables [triangle ABC]

-- Given conditions
def b : ℝ := 2
def cos_C : ℝ := 3 / 4
def area_ABC : ℝ := √7 / 4

-- Question 1: Prove a = 1
theorem find_a (h1 : b = 2) (h2 : cos_C = 3 / 4) (h3 : area_ABC = √7 / 4) : a = 1 := sorry

-- Question 2: Prove sin A = √(14) / 8
theorem find_sin_A (h1 : b = 2) (h2 : cos_C = 3 / 4) (h3 : area_ABC = √7 / 4) : sin A = √(14) / 8 := sorry

end find_a_find_sin_A_l0_744


namespace cos_135_eq_neg_inv_sqrt2_l0_437

theorem cos_135_eq_neg_inv_sqrt2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt2_l0_437


namespace cos_135_eq_neg_sqrt2_div_2_l0_625

theorem cos_135_eq_neg_sqrt2_div_2 : 
  Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_625


namespace profit_percentage_is_correct_l0_396

noncomputable def shopkeeper_profit_percentage : ℚ :=
  let cost_A : ℚ := 12 * (15/16)
  let cost_B : ℚ := 18 * (47/50)
  let profit_A : ℚ := 12 - cost_A
  let profit_B : ℚ := 18 - cost_B
  let total_profit : ℚ := profit_A + profit_B
  let total_cost : ℚ := cost_A + cost_B
  (total_profit / total_cost) * 100

theorem profit_percentage_is_correct :
  shopkeeper_profit_percentage = 6.5 := by
  sorry

end profit_percentage_is_correct_l0_396


namespace cos_135_eq_neg_inv_sqrt_2_l0_554

theorem cos_135_eq_neg_inv_sqrt_2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 :=
sorry

end cos_135_eq_neg_inv_sqrt_2_l0_554


namespace triangle_cos_identity_l0_205

variable {A B C : ℝ} -- Angle A, B, C are real numbers representing the angles of the triangle
variable {a b c : ℝ} -- Sides a, b, c are real numbers representing the lengths of the sides of the triangle

theorem triangle_cos_identity (h : 2 * b = a + c) : 5 * (Real.cos A) - 4 * (Real.cos A) * (Real.cos C) + 5 * (Real.cos C) = 4 :=
by
  sorry

end triangle_cos_identity_l0_205


namespace circle_center_range_l0_201

open Real

theorem circle_center_range (a : ℝ) : 
  (∀ (M : ℝ × ℝ), (M.1 - a)^2 + (M.2 - (2 * a - 4))^2 = 1 → M.1^2 + (M.2 + 1)^2 = 4) → 0 ≤ a ∧ a ≤ 12 / 5 := 
begin
  intro H,
  -- The necessary proof steps will go here, but are omitted.
  sorry
end

end circle_center_range_l0_201


namespace coprime_less_than_prime_coprime_less_than_prime_square_l0_1

theorem coprime_less_than_prime (p : ℕ) (hp : Nat.Prime p) : 
  Nat.card { n : ℕ | n < p ∧ Nat.coprime n p } = p - 1 := 
sorry

theorem coprime_less_than_prime_square (p : ℕ) (hp : Nat.Prime p) : 
  Nat.card { n : ℕ | n < p^2 ∧ Nat.coprime n (p^2) } = p * (p - 1) := 
sorry

end coprime_less_than_prime_coprime_less_than_prime_square_l0_1


namespace fourth_vertex_of_square_l0_757

theorem fourth_vertex_of_square :
  let z1 := (3 + complex.i) / (1 - complex.i)
  let z2 := -2 + complex.i
  let z3 := (0:ℂ)
  ∃ z4: ℂ, z4 = -1 + 3 * complex.i ∧
    z1 ≠ z2 ∧ z2 ≠ z3 ∧ z1 ≠ z3 ∧
    ((z4 - z1 = z2 - z3) ∨ (z1 - z4 = z2 - z3) ∨ 
     (z4 - z1 = z3 - z2) ∨ (z1 - z4 = z3 - z2)) := 
by 
  sorry

end fourth_vertex_of_square_l0_757


namespace total_students_appeared_l0_975

-- Define the given conditions
def percent_passed : ℝ := 35 / 100
def num_failed : ℝ := 455
def percent_failed : ℝ := 1 - percent_passed

-- Statement to prove
theorem total_students_appeared :
  (num_failed / percent_failed) = 700 :=
by
  -- We can eventually write our proof here
  sorry

end total_students_appeared_l0_975


namespace problem_1_problem_2_i_problem_2_ii_l0_122

noncomputable section

-- Definition for the standard equation of the ellipse given its eccentricity and a vertex from the parabola
def ellipse_standard_eq (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (a^2 = 16) ∧ (b^2 = 12) ∧ (1 / 2 = 1 / 2) -- centering the eccentricity and vertex at parabola

theorem problem_1 :
  ellipse_standard_eq 4 (2 * Real.sqrt 3) → (∀ x y, x^2 / 16 + y^2 / 12 = 1) :=
sorry

-- Define the intersections P and Q, and condition of maximum area for the quadrilateral APBQ
def intersect_points (P Q : ℝ × ℝ) : Prop :=
  P = (-2, 3) ∧ Q = (-2, -3) ∨ P = (-2, -3) ∧ Q = (-2, 3)

def slope_condition (A B : ℝ × ℝ) : Prop :=
  ∀ x y, slope AB = 1 / 2 → ∃ m, slope AB = m → (-2 < m < 4)

theorem problem_2_i (P Q A B : ℝ × ℝ) (hPQ : intersect_points P Q) (hAB : slope_condition A B) :
  max_area_quadrilateral P Q A B = 12 * Real.sqrt 3 :=
sorry

-- Condition for the angles and the slope fixed value logic
def angle_condition (A B : ℝ × ℝ) (P Q : ℝ × ℝ) : Prop :=
  ∠ APQ = ∠ BPQ

theorem problem_2_ii (P Q A B : ℝ × ℝ) (hPQ : intersect_points P Q)
  (hAngle : angle_condition A B P Q) :
  slope AB = -1 / 2 ∨ slope AB = 1 / 2 :=
sorry

end problem_1_problem_2_i_problem_2_ii_l0_122


namespace probability_at_least_one_male_l0_104

theorem probability_at_least_one_male (total_students male_students female_students chosen_students : ℕ) 
  (total_comb : ℕ := Nat.choose total_students chosen_students)
  (female_comb : ℕ := Nat.choose female_students chosen_students)
  (prob_at_least_one_male : ℚ := 1 - (female_comb : ℚ) / (total_comb : ℚ)) :
  (total_students = 5) ∧ (male_students = 3) ∧ (female_students = 2) ∧ (chosen_students = 2) → 
  prob_at_least_one_male = 9 / 10 :=
by
  sorry

end probability_at_least_one_male_l0_104


namespace running_laps_l0_987

theorem running_laps (A B : ℕ)
  (h_ratio : ∀ t : ℕ, (A * t) = 5 * (B * t) / 3)
  (h_start : A = 5 ∧ B = 3 ∧ ∀ t : ℕ, (A * t) - (B * t) = 4) :
  (B * 2 = 6) ∧ (A * 2 = 10) :=
by
  sorry

end running_laps_l0_987


namespace same_flips_probability_calc_l0_950

-- Definitions corresponding to the conditions in a)
def fair_coin_flip_probability (n : ℕ) : ℝ :=
  if n < 2 then 0 else (n-1)^3 * (1/2)^(3*n)

-- Main statement: The sum representing the probability that all three flip the same number of times
noncomputable def total_probability : ℝ :=
  ∑' (n : ℕ), fair_coin_flip_probability n

-- The theorem we want to prove
theorem same_flips_probability_calc : total_probability = sorry :=
sorry

end same_flips_probability_calc_l0_950


namespace cos_135_eq_correct_l0_605

noncomputable def cos_135_eq: Prop :=
  real.cos ((135 * real.pi) / 180) = -real.sqrt 2 / 2

theorem cos_135_eq_correct : cos_135_eq :=
  sorry

end cos_135_eq_correct_l0_605


namespace inradius_theorem_l0_133

-- Define points D, E, F on sides BC, CA, and AB of triangle ABC
variables {A B C D E F : Type} [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
variables [Incircle.triangle_incircle ℝ (Triangle.mk A B C) (Incircle.mk_incircle)]
variables [Incircle.triangle_incircle ℝ (Triangle.mk A E F) (Incircle.mk_incircle)]
variables [Incircle.triangle_incircle ℝ (Triangle.mk B F D) (Incircle.mk_incircle)]
variables [Incircle.triangle_incircle ℝ (Triangle.mk C D E) (Incircle.mk_incircle)]

-- Define the radii fields
variables (r r_0 R : ℝ)

-- Assume the incircles of triangles AEF, BFD, CDE have the same radius r
axiom incircle_equiv : ∀ r : ℝ, 
  Incircle.radius (Triangle.mk A E F).incircle = r ∧
  Incircle.radius (Triangle.mk B F D).incircle = r ∧
  Incircle.radius (Triangle.mk C D E).incircle = r →

-- Let r0 be the inradius of DEF and R be the inradius of ABC
axiom def_inradius : Incircle.radius (Triangle.mk D E F).incircle = r_0
axiom abc_inradius : Incircle.radius (Triangle.mk A B C).incircle = R

-- The theorem to be proven
theorem inradius_theorem :
  r + r_0 = R :=
sorry

end inradius_theorem_l0_133


namespace positive_difference_of_two_numbers_l0_930

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l0_930


namespace parabola_larger_zero_l0_290

theorem parabola_larger_zero (a b c : ℝ)
  (vertex_condition : ∃ h k : ℝ, h = 3 ∧ k = -3 ∧ ∀ x : ℝ, y = a * (x - h) ^ 2 + k)
  (point_condition : ∃ p q : ℝ, p = 5 ∧ q = 17 ∧ ∀ x2 : ℝ, q = a * (x2 - h) ^ 2 + k) :
  let h := 3 in
  let x := 3 + real.sqrt (3/5) in
  x - h = real.sqrt (3/5) :=
sorry

end parabola_larger_zero_l0_290


namespace ellipse_value_l0_759

noncomputable def a_c_ratio (a c : ℝ) : ℝ :=
  (a + c) / (a - c)

theorem ellipse_value (a b c : ℝ) 
  (h1 : a^2 = b^2 + c^2) 
  (h2 : a^2 + b^2 - 3 * c^2 = 0) :
  a_c_ratio a c = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end ellipse_value_l0_759


namespace table_height_l0_968

variable (l h w : ℝ)

-- Given conditions:
def conditionA := l + h - w = 36
def conditionB := w + h - l = 30

-- Proof that height of the table h is 33 inches
theorem table_height {l h w : ℝ} 
  (h1 : l + h - w = 36) 
  (h2 : w + h - l = 30) : 
  h = 33 := 
by
  sorry

end table_height_l0_968


namespace cosine_135_eq_neg_sqrt_2_div_2_l0_536

theorem cosine_135_eq_neg_sqrt_2_div_2 (cos180 : Real := -1) (cos90 : Real := 0) (cos45 : Real := Real.sqrt 2 / 2)
    (theta1 : Real := 180) (theta2 : Real := 45) (theta3 : Real := 135) :
    theta3 = theta1 - theta2 ∧ 
    Real.cos theta1 = cos180 ∧ 
    Real.cos theta2 = cos45 → 
    Real.cos theta3 = -cos45 := by
  sorry

end cosine_135_eq_neg_sqrt_2_div_2_l0_536


namespace min_f_value_zodiac_numbers_l0_735

def is_unity_number (m : ℕ) : Prop :=
  let h := m / 100
  let t := (m / 10) % 10
  let u := m % 10
  h = t + u

def f (m : ℕ) : ℕ :=
  let n := (m / 100) + (m % 100) * 10
  (m - n) / 9

theorem min_f_value : ∃ m, is_unity_number m ∧ f(m) = 1 :=
by 
  exists 110
  sorry

theorem zodiac_numbers : ∀ m, is_unity_number m → (f(m) % 12 = 0) → f(m) ∈ {12, 24, 36, 72} :=
by 
  intro m 
  intro is_unity
  intro f_multiple_of_twelve
  sorry

end min_f_value_zodiac_numbers_l0_735


namespace roots_of_cubic_poly_with_rational_coeffs_and_tangency_are_rational_l0_225

theorem roots_of_cubic_poly_with_rational_coeffs_and_tangency_are_rational
  (P : ℚ[X])
  (h_deg : degree P = 3)
  (h_tangency : ∃ a : ℚ, P.eval a = 0 ∧ (derivative P).eval a = 0) :
  ∀ root : ℚ, is_root P root :=
sorry

end roots_of_cubic_poly_with_rational_coeffs_and_tangency_are_rational_l0_225


namespace Rohan_earning_after_6_months_l0_834

def farm_area : ℕ := 20
def trees_per_sqm : ℕ := 2
def coconuts_per_tree : ℕ := 6
def harvest_interval : ℕ := 3
def sale_price : ℝ := 0.50
def total_months : ℕ := 6

theorem Rohan_earning_after_6_months :
  farm_area * trees_per_sqm * coconuts_per_tree * (total_months / harvest_interval) * sale_price 
    = 240 := by
  sorry

end Rohan_earning_after_6_months_l0_834


namespace eccentricity_range_value_of_a_l0_71
open Real

-- Define the conditions of the hyperbola and line
def hyperbola (a : ℝ) : set (ℝ × ℝ) := { p | p.1^2 / a^2 - p.2^2 = 1 }
def line (x y : ℝ) : Prop := x + y = 1

-- Define the points A and B, and point P
def is_intersection (p : ℝ × ℝ) (a : ℝ) : Prop :=
  hyperbola a p ∧ line p.1 p.2

-- Eccentricity of the hyperbola
def eccentricity (a : ℝ) : ℝ := (sqrt (a^2 + 1)) / a

-- Proof of the range of eccentricity
theorem eccentricity_range (a : ℝ) (h1 : 0 < a) (h2 : a < sqrt 2) (h3 : a ≠ 1) :
  eccentricity a > sqrt 6 / 2 ∧ eccentricity a ≠ sqrt 2 := 
by
  sorry

-- Define the vector equation for point P
def vector_eq (A B P : ℝ × ℝ) : Prop := 
  let PA := (A.1 - P.1, A.2 - P.2)
  let PB := (B.1 - P.1, B.2 - P.2)
  PA.1 = (5 / 12) * PB.1 ∧ PA.2 = (5 / 12) * PB.2

-- Proof of the value of a
theorem value_of_a (a : ℝ) (A B P : ℝ × ℝ) (h : vector_eq A B P) (h1 : is_intersection A a) (h2 : is_intersection B a) :
  a = 17 / 13 :=
by
  sorry

end eccentricity_range_value_of_a_l0_71


namespace marla_errand_total_time_l0_804

theorem marla_errand_total_time :
  let d1 := 20 -- Driving to son's school
  let b := 30  -- Taking a bus to the grocery store
  let s := 15  -- Shopping at the grocery store
  let w := 10  -- Walking to the gas station
  let g := 5   -- Filling up gas
  let r := 25  -- Riding a bicycle to the school
  let p := 70  -- Attending parent-teacher night
  let c := 30  -- Catching up with a friend at a coffee shop
  let sub := 40-- Taking the subway home
  let d2 := 20 -- Driving home
  d1 + b + s + w + g + r + p + c + sub + d2 = 265 := by
  sorry

end marla_errand_total_time_l0_804


namespace value_of_a_l0_158

theorem value_of_a (a : ℝ) :
  (A : Set ℝ := { x | 1 < x ∧ x < 7 }) ∧
  (B : Set ℝ := { x | a + 1 < x ∧ x < 2a + 5 }) ∧
  (A_inter_B : Set ℝ := A ∩ B) ∧
  (A_inter_B = { x | 3 < x ∧ x < 7 }) →
  a = 2 :=
by
  sorry

end value_of_a_l0_158


namespace points_concyclic_l0_10

theorem points_concyclic (k₁ k₂ : Circle) (A B C D E F G : Point)
  (h1 : k₁ ∈ interior k₂)
  (h2 : k₁.tangents k₂ = {A})
  (h3 : ∃ l : Line, A ∈ l ∧ l ∩ k₁ = {B} ∧ l ∩ k₂ = {C})
  (h4 : ∃ t : Line, t ∈ tangents_of_point B k₁ ∧ t ∩ k₂ = {D, E})
  (h5 : ∃ s : Line, s ∈ tangents_of_point C k₁ ∧ s ∩ k₁ = {F, G}) :
  concyclic {D, E, F, G} :=
begin
  sorry -- Proof omitted
end

end points_concyclic_l0_10


namespace journey_total_distance_l0_733

theorem journey_total_distance (D : ℝ) 
  (train_fraction : ℝ := 3/5) 
  (bus_fraction : ℝ := 7/20) 
  (walk_distance : ℝ := 6.5) 
  (total_fraction : ℝ := 1) : 
  (1 - (train_fraction + bus_fraction)) * D = walk_distance → D = 130 := 
by
  sorry

end journey_total_distance_l0_733


namespace positive_difference_of_two_numbers_l0_938

theorem positive_difference_of_two_numbers (x y : ℝ)
  (h1 : x + y = 10) (h2 : x ^ 2 - y ^ 2 = 40) : abs (x - y) = 4 := 
by
  sorry

end positive_difference_of_two_numbers_l0_938


namespace coefficient_x2_in_expansion_l0_76

def polynomial := (2 - x) * (1 - 3 * x) ^ 4

theorem coefficient_x2_in_expansion 
  (x : ℝ) : 
  (polynomial.coeff x^2) = 120 := sorry

end coefficient_x2_in_expansion_l0_76


namespace proof_cos_135_degree_l0_568

noncomputable def cos_135_degree_is_negative_sqrt2_over_2 : Prop :=
  real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2

theorem proof_cos_135_degree : cos_135_degree_is_negative_sqrt2_over_2 :=
by
  sorry

end proof_cos_135_degree_l0_568


namespace polar_equation_of_circle_length_of_segment_PQ_l0_755

variables {t α α₀ θ θ₁ θ₂ : ℝ}
noncomputable def polar_coordinate_circle_eq : Prop :=
  let x := 1 + cos t
  let y := sin t
  let ρ := 2 * cos θ
  (x - 1)^2 + y^2 = 1

noncomputable def line_eq (ρ : ℝ) : Prop :=
  2 * ρ * sin (α + π / 4) = 2 * sqrt 2

noncomputable def curve_eq (θ : ℝ) : Prop :=
  θ = α₀

def segment_length (ρ₁ ρ₂ : ℝ) : ℝ :=
  abs (ρ₁ - ρ₂)

theorem polar_equation_of_circle :
  ∃ ρ θ, polar_coordinate_circle_eq :=
sorry

theorem length_of_segment_PQ (ρ₁ ρ₂ : ℝ) :
  line_eq ρ₂ ∧ curve_eq θ₁ ∧ curve_eq θ₂ ∧ tan θ₁ = 2 ∧ tan θ₂ = 2 ∧ ρ₁ = (2 / 5) * sqrt 5 ∧ ρ₂ = (2 / 3) * sqrt 5 → segment_length ρ₁ ρ₂ = (4 / 15) * sqrt 5 :=
sorry

end polar_equation_of_circle_length_of_segment_PQ_l0_755


namespace range_of_a_for_monotonicity_l0_739

theorem range_of_a_for_monotonicity (a : ℝ) :
  (∀ x ∈ Set.Iic (-1/4 : ℝ), deriv (λ x, x^2 - 2*a*x + a - 3) x ≤ 0) → 
  a ≤ -3 := sorry

end range_of_a_for_monotonicity_l0_739


namespace find_emma_money_l0_365

-- Defining the given conditions
def brenda_money : ℝ := 8
def brenda_jeff_difference : ℝ := 4
def jeff_money : ℝ := brenda_money - brenda_jeff_difference
def daya_jeff_ratio : ℝ := 2/5
def daya_money : ℝ := jeff_money / daya_jeff_ratio
def daya_emma_ratio : ℝ := 1.25

-- Defining the proof problem
theorem find_emma_money : (daya_money = 10) → (daya_money = daya_emma_ratio * emma_money) → emma_money = 8 :=
by
  sorry

end find_emma_money_l0_365


namespace increasing_necessary_not_sufficient_l0_796

variable {a b : ℝ} {f : ℝ → ℝ}

/-- Let f be differentiable on (a, b). Then f(x) being an increasing function on (a, b)
    is a necessary but not sufficient condition for f'(x) > 0. -/
theorem increasing_necessary_not_sufficient (h_diff : ∀ x ∈ set.Ioo a b, differentiable_at ℝ f x) :
  (∀ x y ∈ set.Ioo a b, x < y → f x < f y) → (∀ x ∈ set.Ioo a b, 0 < deriv f x) → 
  (∃ x ∈ set.Ioo a b, deriv f x = 0) :=
sorry

end increasing_necessary_not_sufficient_l0_796


namespace marks_in_english_eq_l0_774

def avg := 71
def math_marks := 60
def physics_marks := 72
def chemistry_marks := 65
def biology_marks := 82
def subjects_count := 5
def total_marks := avg * subjects_count
def known_marks := math_marks + physics_marks + chemistry_marks + biology_marks

theorem marks_in_english_eq :
  let E := total_marks - known_marks in
  E = 76 := 
by
  sorry

end marks_in_english_eq_l0_774


namespace vector_product_magnitude_l0_134

variables (u v : EuclideanSpace ℝ (Fin 2))
noncomputable def vector_u : EuclideanSpace ℝ (Fin 2) := ![4, 0]
noncomputable def vector_v : EuclideanSpace ℝ (Fin 2) := ![2 - 4, 2 * Real.sqrt 3 - 0]

theorem vector_product_magnitude : ‖u × (vector_u - v)‖ = 8 * Real.sqrt 3 :=
by
  -- Defining u and v as given in the conditions
  let u := ![4, 0]
  let v := ![2 - 4, 2 * Real.sqrt 3 - 0]
  
  -- Condition: u + v = (2, 2√3)
  have h1 : u + v = ![2, 2 * Real.sqrt 3] := by
    simp [vector_u, vector_v]
  
  -- Calculate ‖u × (u - v)‖
  calc
    ‖u × (u - v)‖
      = ‖u‡)
      = ‖u × ( ![4, 0] - (![2 - 4, 2 * Real.sqrt 3 - 0]))‖ : by sorry
      = 8 * Real.sqrt 3

end vector_product_magnitude_l0_134


namespace matrix_identity_l0_87

noncomputable def M := λ : matrix (fin 3) (fin 3) ℝ,
  ![ ![ (1 / real.sqrt 2), (1 / real.sqrt 2), 0 ],
     ![ (-1 / real.sqrt 2), (1 / real.sqrt 2), 0 ],
     ![ 0, 0, (1 / 2) ] ]

def R := λ (θ : ℝ) : matrix (fin 3) (fin 3) ℝ,
  ![ ![ real.cos θ, -real.sin θ, 0 ],
     ![ real.sin θ, real.cos θ, 0 ],
     ![ 0, 0, 2 ] ]

theorem matrix_identity :
  M ⬝ (R (real.pi / 4)) = 1 := by
  sorry

end matrix_identity_l0_87


namespace cos_135_eq_neg_inv_sqrt2_l0_435

theorem cos_135_eq_neg_inv_sqrt2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt2_l0_435


namespace find_s_l0_730

theorem find_s (n : ℤ) (hn : n ≠ 0) (s : ℝ)
  (hs : s = (20 / (2^(2*n+4) + 2^(2*n+2)))^(1 / n)) :
  s = 1 / 4 :=
by
  sorry

end find_s_l0_730


namespace cos_135_eq_neg_inv_sqrt_2_l0_576

theorem cos_135_eq_neg_inv_sqrt_2 :
  cos (135 * Real.pi / 180) = - (1 / Real.sqrt 2) :=
by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_576


namespace cos_135_eq_neg_inv_sqrt_2_l0_579

theorem cos_135_eq_neg_inv_sqrt_2 :
  cos (135 * Real.pi / 180) = - (1 / Real.sqrt 2) :=
by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_579


namespace positive_difference_l0_912

theorem positive_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
sorry

end positive_difference_l0_912


namespace second_term_arithmetic_sequence_l0_742

theorem second_term_arithmetic_sequence (a d : ℝ) (h : a + (a + 2 * d) = 10) : 
  a + d = 5 :=
by
  sorry

end second_term_arithmetic_sequence_l0_742


namespace jesse_max_correct_answers_l0_188

theorem jesse_max_correct_answers :
  ∃ a b c : ℕ, a + b + c = 60 ∧ 5 * a - 2 * c = 150 ∧ a ≤ 38 :=
sorry

end jesse_max_correct_answers_l0_188


namespace number_of_boys_in_class_l0_858

theorem number_of_boys_in_class (n : ℕ) (h : 182 * n - 166 + 106 = 180 * n) : n = 30 :=
by {
  sorry
}

end number_of_boys_in_class_l0_858


namespace domain_of_f_parity_of_f_range_of_f_l0_670

noncomputable def f (a x : ℝ) := Real.log (1 + x) / Real.log a - Real.log (1 - x) / Real.log a

variables {a x : ℝ}

-- The properties derived:
theorem domain_of_f (ha : a > 0) (ha1 : a ≠ 1) : 
  (-1 < x ∧ x < 1) ↔ ∃ y, f a x = y :=
sorry

theorem parity_of_f (ha : a > 0) (ha1 : a ≠ 1) : 
  f a (-x) = -f a x :=
sorry

theorem range_of_f (ha : a > 0) (ha1 : a ≠ 1) : 
  (f a x > 0 ↔ (a > 1 ∧ 0 < x ∧ x < 1) ∨ (0 < a ∧ a < 1 ∧ -1 < x ∧ x < 0)) :=
sorry

end domain_of_f_parity_of_f_range_of_f_l0_670


namespace equal_medians_equal_angle_bisectors_l0_2

variables {A B C A1 B1 C1 M M1 D D1 : Type}

open Triangle

-- Definitions for congruence of two triangles
def congruent (ΔABC ΔA1B1C1 : Triangle) : Prop :=
  ΔABC.AB = ΔA1B1C1.AB ∧ ΔABC.AC = ΔA1B1C1.AC ∧ ΔABC.BC = ΔA1B1C1.BC ∧ 
  ΔABC.∡A = ΔA1B1C1.∡A ∧ ΔABC.∡B = ΔA1B1C1.∡B ∧ ΔABC.∡C = ΔA1B1C1.∡C

-- Definition for medians
def is_median (X Y Z M : Triangle) : Prop :=
  midpoint Y Z M ∧ collinear X Y M ∧ collinear X Z M

-- Definition for angle bisectors
def is_angle_bisector (X Y Z D : Triangle) : Prop :=
  angle_bisector X Y Z D

variables (ΔABC ΔA1B1C1 : Triangle)

-- Given condition: the two triangles are congruent
axiom H : congruent ΔABC ΔA1B1C1

-- Medians from vertices A and A1
axiom HM1 : is_median ΔABC.A ΔABC.B ΔABC.C ΔABC.M
axiom HM2 : is_median ΔA1B1C1.A ΔA1B1C1.B ΔA1B1C1.C ΔA1B1C1.M1

-- Angle bisectors from vertices A and A1
axiom HA1 : is_angle_bisector ΔABC.A ΔABC.B ΔABC.C ΔABC.D
axiom HA2 : is_angle_bisector ΔA1B1C1.A ΔA1B1C1.B ΔA1B1C1.C ΔA1B1C1.D1

theorem equal_medians : ΔABC.M = ΔA1B1C1.M1 :=
by sorry

theorem equal_angle_bisectors : ΔABC.D = ΔA1B1C1.D1 :=
by sorry

end equal_medians_equal_angle_bisectors_l0_2


namespace ratio_of_term_to_difference_l0_307

def arithmetic_progression_sum (n a d : ℕ) : ℕ := (n * (2 * a + (n - 1) * d)) / 2

theorem ratio_of_term_to_difference (a d : ℕ) 
  (h1: arithmetic_progression_sum 7 a d = arithmetic_progression_sum 3 a d + 20)
  (h2 : d ≠ 0) : a / d = 1 / 2 := 
by 
  sorry

end ratio_of_term_to_difference_l0_307


namespace max_leap_years_in_200_years_l0_416

theorem max_leap_years_in_200_years (leap_interval : ℕ) (period : ℕ) (leap_years : ℕ) :
  leap_interval = 5 → period = 200 → leap_years = period / leap_interval → leap_years = 40 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3,
  exact h3.symm
  sorry

end max_leap_years_in_200_years_l0_416


namespace cos_135_eq_neg_inv_sqrt_2_l0_557

theorem cos_135_eq_neg_inv_sqrt_2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 :=
sorry

end cos_135_eq_neg_inv_sqrt_2_l0_557


namespace monotonic_intervals_range_of_a_for_inequality_l0_712

noncomputable def f (a x : ℝ) : ℝ := (x + a) / (a * Real.exp x)

theorem monotonic_intervals (a : ℝ) :
  (if a > 0 then
    ∀ x, (x < (1 - a) → 0 < deriv (f a) x) ∧ ((1 - a) < x → deriv (f a) x < 0)
  else
    ∀ x, (x < (1 - a) → deriv (f a) x < 0) ∧ ((1 - a) < x → 0 < deriv (f a) x)) := 
sorry

theorem range_of_a_for_inequality (a : ℝ) :
  (∀ x, 0 < x → (3 + 2 * Real.log x) / Real.exp x ≤ f a x + 2 * x) ↔
  a ∈ Set.Iio (-1/2) ∪ Set.Ioi 0 :=
sorry

end monotonic_intervals_range_of_a_for_inequality_l0_712


namespace problem1_problem2_l0_713

open Real

-- Define the function f(x) under the given condition a < 1
def f (x a : ℝ) : ℝ := x^2 + a * x + a / x

-- Define the function g(x) with constants k and a
def g (x a k : ℝ) : ℝ := x * f x a + abs (x^2 - 1) + (k - a) * x - a

-- Problem 1: Prove monotonicity of f(x) on [1, +∞) given a < 1
theorem problem1 (a : ℝ) (h_a : a < 1) : ∀ (x1 x2 : ℝ), 1 ≤ x1 → x1 < x2 → f x1 a < f x2 a :=
by
  sorry

-- Problem 2: Given g(x) = 0 has two solutions in (0, 2), compare 1/x1 + 1/x2 with 4
theorem problem2 (a k : ℝ) (x1 x2 : ℝ) (h_a : a < 1) (h_k : -7/2 < k ∧ k < -1)
  (h_solutions : g x1 a k = 0 ∧ g x2 a k = 0 ∧ 0 < x1 ∧ x1 ≤ 1 ∧ 1 < x2 ∧ x2 < 2) :
  (1 / x1 + 1 / x2) < 4 :=
by
  sorry

end problem1_problem2_l0_713


namespace eighth_grade_girls_l0_878

theorem eighth_grade_girls
  (G : ℕ) 
  (boys : ℕ) 
  (h1 : boys = 2 * G - 16) 
  (h2 : G + boys = 68) : 
  G = 28 :=
by
  sorry

end eighth_grade_girls_l0_878


namespace amplitude_of_sine_function_l0_420

theorem amplitude_of_sine_function 
  (a b c d : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_max_min : ∀ x : ℝ, -3 ≤ a * real.sin (b * x + c) + d ∧ a * real.sin (b * x + c) + d ≤ 5) :
  a = 4 :=
sorry

end amplitude_of_sine_function_l0_420


namespace thirtieth_number_l0_212

theorem thirtieth_number (n : ℕ) : 
  let first_sequence := list.range' 2 ((20 - 2) / 2 + 1) |>.map (λ n, n * 2)
  let second_sequence_start := 23
  let second_sequence := list.range' second_sequence_start (n - (1 + (second_sequence_start - 20) / 3)) |>.map (λ n, second_sequence_start + 3 * (n - 1))
  (first_sequence.length + second_sequence.length >= 30) -> (first_sequence ++ second_sequence).nth! (30 - 1) = 50 := 
by
  sorry

end thirtieth_number_l0_212


namespace johns_class_l0_746

theorem johns_class (k : ℕ) (h1 : 4 * k + 5 * k = 36) : 4 * k = 16 :=
by
  have h : 9 * k = 36 := by linarith
  have k_eq : k = 4 := by linarith
  rw [k_eq]
  rfl

end johns_class_l0_746


namespace polynomial_remainder_l0_786

theorem polynomial_remainder (P : Polynomial ℝ) (H1 : P.eval 1 = 2) (H2 : P.eval 2 = 1) :
  ∃ Q : Polynomial ℝ, P = Q * (Polynomial.X - 1) * (Polynomial.X - 2) + (3 - Polynomial.X) :=
by
  sorry

end polynomial_remainder_l0_786


namespace percentage_w_less_x_l0_293

theorem percentage_w_less_x 
    (z : ℝ) 
    (y : ℝ) 
    (x : ℝ) 
    (w : ℝ) 
    (hy : y = 1.20 * z)
    (hx : x = 1.20 * y)
    (hw : w = 1.152 * z) 
    : (x - w) / x * 100 = 20 :=
by
  sorry

end percentage_w_less_x_l0_293


namespace correct_calculation_result_l0_167

theorem correct_calculation_result (x : ℤ) (h : x - 32 = 25) : x - 23 = 34 :=
by
  sorry

end correct_calculation_result_l0_167


namespace perimeter_is_32_l0_32

-- Define the side lengths of the triangle
def a : ℕ := 13
def b : ℕ := 9
def c : ℕ := 10

-- Definition of the perimeter of the triangle
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- Theorem stating the perimeter is 32
theorem perimeter_is_32 : perimeter a b c = 32 :=
by
  sorry

end perimeter_is_32_l0_32


namespace positive_difference_of_two_numbers_l0_937

theorem positive_difference_of_two_numbers (x y : ℝ)
  (h1 : x + y = 10) (h2 : x ^ 2 - y ^ 2 = 40) : abs (x - y) = 4 := 
by
  sorry

end positive_difference_of_two_numbers_l0_937


namespace permutation_probability_l0_749

theorem permutation_probability (total_digits: ℕ) (zeros: ℕ) (ones: ℕ) 
  (total_permutations: ℕ) (favorable_permutations: ℕ) (probability: ℚ)
  (h1: total_digits = 6) 
  (h2: zeros = 2) 
  (h3: ones = 4) 
  (h4: total_permutations = 2 ^ total_digits) 
  (h5: favorable_permutations = Nat.choose total_digits zeros) 
  (h6: probability = favorable_permutations / total_permutations) : 
  probability = 15 / 64 := 
sorry

end permutation_probability_l0_749


namespace ellipse_equation_and_trajectory_l0_681

-- Definitions for the problem conditions
def ellipse_center : ℝ × ℝ := (0, 0)
def left_focus : ℝ × ℝ := (-real.sqrt 3, 0)
def right_vertex : ℝ × ℝ := (2, 0)
def point_A : ℝ × ℝ := (1, 1 / 2)

-- Standard form of ellipse using given conditions
def standard_ellipse_equation : Prop :=
  ∀ x y : ℝ, (x^2 / 4 + y^2 = 1 ↔ (x, y) = (x, y))

-- Trajectory equation for the midpoint of PA
def midpoint_trajectory_equation : Prop :=
  ∀ x y : ℝ, ((x - 1 / 2)^2 + 4 * (y - 1 / 4)^2 = 1 ↔ (x, y) = (x, y))

-- Lean 4 statement for proving the equations
theorem ellipse_equation_and_trajectory (x y : ℝ) : standard_ellipse_equation ∧ midpoint_trajectory_equation :=
by
  sorry  -- Proof of the theorem goes here


end ellipse_equation_and_trajectory_l0_681


namespace cos_135_eq_neg_sqrt2_div_2_l0_483

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by 
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_483


namespace positive_difference_l0_895

theorem positive_difference (a b : ℝ) (h₁ : a + b = 10) (h₂ : a^2 - b^2 = 40) : |a - b| = 4 :=
by
  sorry

end positive_difference_l0_895


namespace regular_tetrahedron_medians_angle_distance_l0_652

theorem regular_tetrahedron_medians_angle_distance (a : ℝ) :
  ∃ θ d : ℝ, 
    θ = Real.arccos (1 / 6) ∧ d = a * Real.sqrt (1 / 3) :=
begin
  sorry
end

end regular_tetrahedron_medians_angle_distance_l0_652


namespace initial_apples_l0_245

-- Definitions of the conditions
def Minseok_ate : Nat := 3
def Jaeyoon_ate : Nat := 3
def apples_left : Nat := 2

-- The proposition we need to prove
theorem initial_apples : Minseok_ate + Jaeyoon_ate + apples_left = 8 := by
  sorry

end initial_apples_l0_245


namespace number_composite_l0_251

theorem number_composite (n : ℕ) : 
  n = 10^(2^1974 + 2^1000 - 1) + 1 →
  ∃ a b : ℕ, 1 < a ∧ a < n ∧ n = a * b :=
by sorry

end number_composite_l0_251


namespace cos_135_eq_neg_sqrt2_div_2_l0_617

theorem cos_135_eq_neg_sqrt2_div_2 : 
  Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_617


namespace determine_knights_liars_l0_401

noncomputable def natives_circle (n : ℕ) : Prop :=
  ∃ (knights liars : Fin n → Bool), 
  ∀ i : Fin n,
    (knights i → 
      (statement_age_left i = age (left_neighbor i) ∧ statement_age_right i = age (right_neighbor i))) ∧ 
    (liars i → 
      ((statement_age_left i = age (left_neighbor i) + 1 ∨ statement_age_left i = age (left_neighbor i) - 1) ∧
       (statement_age_right i = age (right_neighbor i) + 1 ∨ statement_age_right i = age (right_neighbor i) - 1)))

theorem determine_knights_liars :
  natives_circle 50 →
  ∃ (f : Fin 50 → Bool), 
  (∀ i : Fin 50, f i = true ↔ is_knight i) ∧ (∀ i : Fin 50, f i = false ↔ is_liar i) :=
sorry

end determine_knights_liars_l0_401


namespace space_needed_between_apple_trees_l0_253

-- Definitions based on conditions
def apple_tree_width : ℕ := 10
def peach_tree_width : ℕ := 12
def space_between_peach_trees : ℕ := 15
def total_space : ℕ := 71
def number_of_apple_trees : ℕ := 2
def number_of_peach_trees : ℕ := 2

-- Lean 4 theorem statement
theorem space_needed_between_apple_trees :
  (total_space 
   - (number_of_peach_trees * peach_tree_width + space_between_peach_trees))
  - (number_of_apple_trees * apple_tree_width) 
  = 12 := by
  sorry

end space_needed_between_apple_trees_l0_253


namespace number_of_girls_l0_879

-- Define the number of girls and boys
variables (G B : ℕ)

-- Define the conditions
def condition1 : Prop := B = 2 * G - 16
def condition2 : Prop := G + B = 68

-- The theorem we want to prove
theorem number_of_girls (h1 : condition1 G B) (h2 : condition2 G B) : G = 28 :=
by
  sorry

end number_of_girls_l0_879


namespace monotonic_increasing_interval_l0_286

theorem monotonic_increasing_interval : ∀ x : ℝ, (x > 2) → ((x-3) * Real.exp x > 0) :=
sorry

end monotonic_increasing_interval_l0_286


namespace problem_1_problem_2a_problem_2b_problem_3_l0_149

def f (x : ℝ) (a : ℝ) : ℝ := x - 1 + a / Real.exp(x)

-- Proof Problem 1
theorem problem_1 (a : ℝ) (h : ∀ x, f x a = x - 1 + a / Real.exp(x)) (h1 : (∃ x, ∀ y, (f' x = 0 -> y = f x a)))
    (hx : (1 - a / Real.e) = 0) :
    a = Real.e :=
sorry

-- Proof Problem 2a: When a ≤ 0
theorem problem_2a (a : ℝ) (ha : a ≤ 0) : (∀ x, f' x a > 0) → ∀ x, ¬(∃ y, (f x a = y ∧ f' x a = 0)) :=
sorry

-- Proof Problem 2b: When a > 0
theorem problem_2b (a : ℝ) (ha : a > 0) :
  ∃ x, ∀ y, (y = f (Real.log a) a) ↔ y = Real.log a ∧ 
  (∀ z, (z < Real.log a → f' z a < 0) ∧ (z > Real.log a → f' z a > 0)) :=
sorry

-- Proof Problem 3
def g (x : ℝ) (k : ℝ) := f x 1 - (k * x - 1)

theorem problem_3 (k : ℝ) (h : ∀ x, g x k <> 0) :
  k ≤ 1 :=
sorry

end problem_1_problem_2a_problem_2b_problem_3_l0_149


namespace min_b1_b2_l0_717

theorem min_b1_b2 (b : ℕ → ℕ) (h : ∀ n ≥ 1, b (n + 2) = (b n + 2210) / (1 + b (n + 1))) :
  ∃ (b1 b2 : ℕ), b1 b2 = 2210 ∧ (∀ n, b n = ite (n % 2 = 0) b1 b2) ∧ b1 + b2 = 147 :=
by
  sorry

end min_b1_b2_l0_717


namespace determine_truthfulness_of_natives_l0_403

-- Definitions of the problem in Lean
/-- Each native has a position in the circle and an age in years. -/
structure Native :=
  (pos : Nat) -- position in the circle (1 through 50)
  (leftAge : Nat) -- age of left neighbor as declared by this native
  (rightAge : Nat) -- age of right neighbor as declared by this native
  (truthfulness : Bool) -- True if the native is a knight, False if a liar

/-- Given conditions: There are 50 natives in a circle. -/
constant natives : Fin 50 → Native

/-- Each knight declares the age of their neighbors correctly. -/
axiom knights_correct (i : Fin 50) : (natives i).truthfulness = true →
  (natives i).leftAge = (natives ((i.val + 49) % 50)).pos ∧
  (natives i).rightAge = (natives ((i.val + 1) % 50)).pos

/-- Each liar increases one age by 1 and decreases the other by 1 of their choice. -/
axiom liars_incorrect (i : Fin 50) : (natives i).truthfulness = false →
  ((natives i).leftAge = (natives ((i.val + 49) % 50)).pos + 1 ∧
   (natives i).rightAge = (natives ((i.val + 1) % 50)).pos - 1) ∨
  ((natives i).leftAge = (natives ((i.val + 49) % 50)).pos - 1 ∧
   (natives i).rightAge = (natives ((i.val + 1) % 50)).pos + 1)

-- Statement of the problem to prove
theorem determine_truthfulness_of_natives :
  ∃ f : Fin 50 → Bool, ∀ i : Fin 50, f i = (natives i).truthfulness :=
sorry -- proof omitted

end determine_truthfulness_of_natives_l0_403


namespace solve_diamond_eq_l0_957

noncomputable def diamond (a b : ℝ) : ℝ := a / b

theorem solve_diamond_eq :
  ∃ x : ℝ, diamond 504 (diamond 12 x) = 50 ∧ x = 25 / 21 :=
by
  let x := 25 / 21
  have H1 : diamond 12 x = 12 / x := rfl
  have H2 : diamond 504 (diamond 12 x) = 504 / (12 / x) := rfl
  use x
  split
  { exact H2 }
  { exact rfl }
  sorry

end solve_diamond_eq_l0_957


namespace trigonometric_identity_l0_428

theorem trigonometric_identity :
  let sin_30 := 1 / 2
  let cos_30 := Real.sqrt 3 / 2
  let tan_60 := Real.sqrt 3
  2 * sin_30 + cos_30 * tan_60 = 5 / 2 :=
by
  let sin_30 := 1 / 2
  let cos_30 := Real.sqrt 3 / 2
  let tan_60 := Real.sqrt 3
  have h1 : 2 * sin_30 = 1 := by norm_num
  have h2 : cos_30 * tan_60 = 3 / 2 := by norm_num
  calc
    2 * sin_30 + cos_30 * tan_60
        = 1 + 3 / 2 : by rw [h1, h2]
    ... = 5 / 2 : by norm_num

end trigonometric_identity_l0_428


namespace total_weight_of_watermelons_and_pineapples_l0_314

theorem total_weight_of_watermelons_and_pineapples :
  (∀ w p, 4 * w = 5200 ∧ w = p + 850 → 3 * w + 4 * p = 5700) :=
by 
  intros w p h
  suffices: 3 * w + 4 * p = 5700
  sorry

end total_weight_of_watermelons_and_pineapples_l0_314


namespace sum_b_eq_l0_156

open BigOperators

-- Given conditions from the problem
def a (n : ℕ) : ℝ := (∑ i in finset.range n, (i + 1) : ℝ) / (n + 1)
def b (n : ℕ) : ℝ := 4 * (1 / n - 1 / (n + 1))

-- Statement to prove the sum of the first n terms of the sequence {b_n}
theorem sum_b_eq (n : ℕ) : (∑ i in finset.range n, b (i + 1)) = 4 * n / (n + 1) :=
sorry

end sum_b_eq_l0_156


namespace total_earnings_after_six_months_l0_836

def area_of_farm : ℕ := 20
def trees_per_square_meter : ℕ := 2
def coconuts_per_tree : ℕ := 6
def harvest_interval_months : ℕ := 3
def price_per_coconut : ℕ := 0.5 -- Note: Lean requires specific handling for non-integer values, so we can use a fractional form instead

theorem total_earnings_after_six_months :
  let total_trees := area_of_farm * trees_per_square_meter in
  let total_coconuts := total_trees * coconuts_per_tree in
  let number_of_harvests := 6 / harvest_interval_months in
  let total_coconuts_six_months := total_coconuts * number_of_harvests in
  let earnings := total_coconuts_six_months * price_per_coconut in
  earnings = 240 :=
by {
  sorry
}

end total_earnings_after_six_months_l0_836


namespace no_possible_stack_of_1997_sum_l0_28

theorem no_possible_stack_of_1997_sum :
  ¬ ∃ k : ℕ, 6 * k = 3 * 1997 := by
  sorry

end no_possible_stack_of_1997_sum_l0_28


namespace variance_seven_points_l0_135

variables {a_1 a_2 a_3 a_4 a_5 a_6 : ℝ}

def average (a1 a2 a3 a4 a5 a6 : ℝ) : ℝ :=
  (a1 + a2 + a3 + a4 + a5 + a6) / 6

def variance (a1 a2 a3 a4 a5 a6 : ℝ) (mean : ℝ) : ℝ :=
  ((a1 - mean)^2 + (a2 - mean)^2 + (a3 - mean)^2 + 
   (a4 - mean)^2 + (a5 - mean)^2 + (a6 - mean)^2) / 6

theorem variance_seven_points :
  let x := average a_1 a_2 a_3 a_4 a_5 a_6 in
  variance a_1 a_2 a_3 a_4 a_5 a_6 x = 0.2 →
  let v := ((a_1 - x)^2 + (a_2 - x)^2 + (a_3 - x)^2 + 
            (a_4 - x)^2 + (a_5 - x)^2 + (a_6 - x)^2) in
  ((a_1 - x)^2 + (a_2 - x)^2 + (a_3 - x)^2 + 
   (a_4 - x)^2 + (a_5 - x)^2 + (a_6 - x)^2 + (x - x)^2) / 7 = 6 / 35 :=
begin
  sorry
end

end variance_seven_points_l0_135


namespace cos_135_eq_neg_inv_sqrt_2_l0_586

theorem cos_135_eq_neg_inv_sqrt_2 :
  cos (135 * Real.pi / 180) = - (1 / Real.sqrt 2) :=
by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_586


namespace train_speed_km_per_hr_l0_399

theorem train_speed_km_per_hr (train_length bridge_length : ℕ) (cross_time : ℕ):
  train_length = 140 → bridge_length = 235 → cross_time = 30 →
  let total_distance := train_length + bridge_length in
  let speed_m_s := total_distance / cross_time in
  let speed_km_hr := speed_m_s * 3.6 in
  speed_km_hr = 45 :=
by
  intros h1 h2 h3
  let total_distance := train_length + bridge_length
  let speed_m_s := total_distance / cross_time
  let speed_km_hr := speed_m_s * 3.6
  have h_total_distance : total_distance = 375 := by
    simp [h1, h2]
  have h_speed_m_s : speed_m_s = 12.5 := by
    simp [h_total_distance, h3]
    norm_num
  have h_speed_km_hr : speed_km_hr = 45 := by
    simp [h_speed_m_s]
    norm_num
  exact h_speed_km_hr

end train_speed_km_per_hr_l0_399


namespace parallelogram_area_l0_725

theorem parallelogram_area :
  let base := 5
  let height := 7
  let area := base * height
  area = 35 :=
by
  have base_def : base = 5 := rfl
  have height_def : height = 7 := rfl
  have area_def : area = base * height := rfl
  rw [base_def, height_def] at area_def
  exact area_def.symm

end parallelogram_area_l0_725


namespace mod_inverse_sum_l0_78

theorem mod_inverse_sum :
  (2⁻¹ : ℤ) + (4⁻¹ : ℤ) ≡ 5 [MOD 17] :=
by
  have h1 : 2 * 9 ≡ 1 [MOD 17], by norm_num,
  have h2 : 4 * 13 ≡ 1 [MOD 17], by norm_num,
  have inv2 : 2⁻¹ ≡ 9 [MOD 17], from (inv_of_eq_inv h1).symm,
  have inv4 : 4⁻¹ ≡ 13 [MOD 17], from (inv_of_eq_inv h2).symm,
  calc
    (2⁻¹ : ℤ) + (4⁻¹ : ℤ) ≡ 9 + 13 [MOD 17] : by rw [inv2, inv4]
    ... ≡ 22 [MOD 17] : by norm_num
    ... ≡ 5 [MOD 17] : by norm_num

end mod_inverse_sum_l0_78


namespace cos_135_degree_l0_495

theorem cos_135_degree : 
  let Q := (Complex.exp (135 * Real.pi / 180 * Complex.I)) in 
  Complex.re Q = -Real.sqrt 2 / 2 :=
by sorry

end cos_135_degree_l0_495


namespace cos_135_eq_correct_l0_604

noncomputable def cos_135_eq: Prop :=
  real.cos ((135 * real.pi) / 180) = -real.sqrt 2 / 2

theorem cos_135_eq_correct : cos_135_eq :=
  sorry

end cos_135_eq_correct_l0_604


namespace fraction_is_7_over_22_l0_14

theorem fraction_is_7_over_22 (n m : ℕ) (h1 : n / (m - 1) = 1 / 3) (h2 : (n + 4) / m = 1 / 2) : n = 7 ∧ m = 22 :=
by
  -- We use integer version of equality here to allow arithmetic operations
  have h3 : 3 * n = m - 1, from sorry,
  have h4 : 2 * (n + 4) = m, from sorry,
  have h5 : 3 * n + 1 = 2 * n + 8, from sorry,
  have h6 : n = 7, from sorry,
  have h7 : m = 22, from sorry,
  exact ⟨h6, h7⟩


end fraction_is_7_over_22_l0_14


namespace largest_k_no_rooks_l0_791

namespace Chessboard

-- Defining a peaceful rook placement on an n x n chessboard
def isPeaceful (n : ℕ) (rooks : Fin n → Fin n) : Prop :=
  Function.Injective rooks

-- The main theorem to express the equivalence
theorem largest_k_no_rooks (n : ℕ) (h : n ≥ 2) :
  ∃ k : ℕ, (∃ i : ℕ, i^2 < n ∧ n ≤ (i+1)^2 ∧ k = i) ∧
    ∀ rooks : Fin n → Fin n, isPeaceful n rooks → 
    ∃ (r1 r2 : Fin n), ∀ i j : ℕ, i < k → j < k → (r1 + i ≤ n - 1) ∧ (r2 + j ≤ n - 1) ∧ 
      rooks (Fin.ofNat (r1 + i)) ≠ Fin.ofNat (r2 + j) :=
sorry

end Chessboard

end largest_k_no_rooks_l0_791


namespace count_M_eq_count_N_l0_797

-- Define increasing function on ℝ
def is_increasing (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f(x) < f(y)

-- Define sets M and N
def M (f : ℝ → ℝ) := {x : ℝ | f(x) = x}
def N (f : ℝ → ℝ) := {x : ℝ | f(f(x)) = x}

-- The theorem statement
theorem count_M_eq_count_N (f : ℝ → ℝ) (h : is_increasing f) : 
  (M f).to_finset.card = (N f).to_finset.card := 
by
  sorry

end count_M_eq_count_N_l0_797


namespace ratio_area_triangle_quadrilateral_l0_763

variables {A B C D E : Type*}
variables [linear_ordered_field A]
variables [linear_ordered_field B]
variables [linear_ordered_field C]

-- Given conditions
variables (AB BC AC AD AE : ℝ)
variables (h1 : AB = 40) (h2 : BC = 50) (h3 : AC = 58)
variables (h4 : AD = 30) (h5 : AE = 20)

-- Prove that the ratio of the area of triangle ADE to the area of the quadrilateral BCED is 9/7
theorem ratio_area_triangle_quadrilateral 
	(h1 : AB = 40) (h2 : BC = 50) (h3 : AC = 58)
    (h4 : AD = 30) (h5 : AE = 20) :
    (Area(△ ADE) / Area(BCED)) = 9 / 7 :=
by
  sorry

end ratio_area_triangle_quadrilateral_l0_763


namespace max_halls_visited_l0_287

/-- 
  Given a museum with 16 halls where half have paintings and the other half have sculptures,
  arranged alternatingly starting from paintings in hall A, find the maximum number of halls
  a tourist can visit starting from hall A and ending at hall B without visiting any hall more than once.
-/
theorem max_halls_visited : 
  ∃ (route : list ℕ), length route = 15 ∧ valid_route route ∧ route.head = A ∧ route.last = B
:= 
sorry

-- Definitions needed for the above theorem
def A : ℕ := 1   -- assuming hall A is represented by 1
def B : ℕ := 16  -- assuming hall B is represented by 16

-- Function to check if a route is valid considering alternation & visiting constraints
def valid_route (route : list ℕ) : Prop :=
  (∀ i, i < route.length - 1 → route.nth i ≠ route.nth (i + 1)) ∧
  (∀ hall, list.count route hall ≤ 1) ∧
  (∀ hall, (hall ≤ 8 → nth route hall = 1) ∧ (hall > 8 → nth route hall = 2) ) -- alternate


end max_halls_visited_l0_287


namespace proof_cos_135_degree_l0_563

noncomputable def cos_135_degree_is_negative_sqrt2_over_2 : Prop :=
  real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2

theorem proof_cos_135_degree : cos_135_degree_is_negative_sqrt2_over_2 :=
by
  sorry

end proof_cos_135_degree_l0_563


namespace cos_alpha_value_l0_703

variable (α : ℝ)
variable (x y r : ℝ)

-- Conditions
def point_condition : Prop := (x = 1 ∧ y = -Real.sqrt 3 ∧ r = 2 ∧ r = Real.sqrt (x^2 + y^2))

-- Question/Proof Statement
theorem cos_alpha_value (h : point_condition x y r) : Real.cos α = 1 / 2 :=
sorry

end cos_alpha_value_l0_703


namespace equation_has_solution_l0_98

noncomputable def solve_equation (a x : ℝ) : Prop :=
  log 2 ((3 * sqrt 3 + cos a * (sin x + 4)) / (3 * sin a * cos x)) = 
  abs (3 * sin a * cos x) - abs (cos a * (sin x + 4) + 3 * sqrt 3)

def is_solution (a x : ℝ) : Prop :=
  (a = 5 * π / 6 + 2 * π * n ∧ x = π / 6 + 2 * π * k) 
  ∨ (a = -5 * π / 6 + 2 * π * n ∧ x = 5 * π / 6 + 2 * π * k)

theorem equation_has_solution (a x : ℝ) (n k : ℤ) : 
  (3 * sqrt 3 + cos a * (sin x + 4)) / (3 * sin a * cos x) > 0 →
  solve_equation a x ↔ is_solution a x :=
sorry

end equation_has_solution_l0_98


namespace steins_lemma_steins_lemma_multivariable_l0_978

-- Definitions for almost differentiable functions and Gaussian distributions
variable {f : ℝ → ℝ} {d : ℕ} {η : Vector ℝ d} {ξ : ℝ}

noncomputable def almost_differentiable (f : ℝ → ℝ) :=
  ∃ (g : ℝ → ℝ), ∀ a b : ℝ, f(b) - f(a) = ∫ x in 0..1, (b - a) * g(a + x * (b - a))

noncomputable def almost_differentiable_d (f : Vector ℝ d → ℝ) :=
  ∃ (g : Vector ℝ d → Vector ℝ d), ∀ a b : Vector ℝ d, f b - f a = ∫ x in 0..1, (b - a) ⬝ g(a + x • (b - a))

-- Stein's lemma for single-variable function
theorem steins_lemma
  (hf : almost_differentiable f)
  (hE' : ∫ ξ, |f' ξ| * PDF(ξ) < ∞)
  (ξ : ℝ) (hξ : has_pdf ξ (λ x, 1 / sqrt (2 * π) * exp (-x^2 / 2))) :
  (∫ ξ, |ξ * f ξ| * PDF(ξ) < ∞) ∧ (∫ ξ, ξ * f ξ = ∫ ξ, f' ξ) :=
sorry

-- Generalized Stein's lemma for multivariable function
theorem steins_lemma_multivariable
  (hf : almost_differentiable_d (f : Vector ℝ d → ℝ))
  (hE' : ∫ η, ∑ i in Finset.range d, |f'_i η| * PDF(η) < ∞)
  (ξ : ℝ) (hξ : has_pdf ξ (λ x, 1 / sqrt (2 * π) * exp (-x^2 / 2)))
  (η : Vector ℝ d) 
  (hη : has_pdf η (multivariate_normal PDF with mean 0 covariance 1)):
  (∫ η, ∫ ξ, |ξ * f η| * PDF(η,ξ) < ∞) ∧ 
  (∫ η, ∫ ξ, ξ * f η * PDF(η,ξ) = ∫ η, ∇f η * ξη * PDF(η,ξ)) :=
sorry

end steins_lemma_steins_lemma_multivariable_l0_978


namespace probability_twice_correct_l0_60

noncomputable def probability_at_least_twice (x y : ℝ) : ℝ :=
if (0 ≤ x ∧ x ≤ 1000) ∧ (0 ≤ y ∧ y ≤ 3000) then
  if y ≥ 2*x then (1/6 : ℝ) else 0
else 0

theorem probability_twice_correct : probability_at_least_twice 500 1000 = (1/6 : ℝ) :=
sorry

end probability_twice_correct_l0_60


namespace cosine_135_eq_neg_sqrt_2_div_2_l0_540

theorem cosine_135_eq_neg_sqrt_2_div_2 (cos180 : Real := -1) (cos90 : Real := 0) (cos45 : Real := Real.sqrt 2 / 2)
    (theta1 : Real := 180) (theta2 : Real := 45) (theta3 : Real := 135) :
    theta3 = theta1 - theta2 ∧ 
    Real.cos theta1 = cos180 ∧ 
    Real.cos theta2 = cos45 → 
    Real.cos theta3 = -cos45 := by
  sorry

end cosine_135_eq_neg_sqrt_2_div_2_l0_540


namespace find_t_value_l0_203

theorem find_t_value (k t : ℤ) (h1 : 0 < k) (h2 : k < 10) (h3 : 0 < t) (h4 : t < 10) : t = 6 :=
by
  sorry

end find_t_value_l0_203


namespace angle_POF_impossible_l0_690

theorem angle_POF_impossible (a : ℝ) (ha : a > 0)
    (F : ℝ × ℝ) (hF : F ∈ {p : ℝ × ℝ | p.1 ^ 2 / (3 * a ^ 2) - p.2 ^ 2 / (a ^ 2) = 1}) 
    (O : ℝ × ℝ) (hO : O = (0, 0))
    (P : ℝ × ℝ) (hP : P ∈ {p : ℝ × ℝ | p.1 ^ 2 / (3 * a ^ 2) - p.2 ^ 2 / (a ^ 2) = 1}) :
    ¬(∠POF = 60) := 
sorry

end angle_POF_impossible_l0_690


namespace last_digit_p_minus_q_not_5_l0_325

theorem last_digit_p_minus_q_not_5 (p q : ℕ) (n : ℕ) 
  (h1 : p * q = 10^n) 
  (h2 : ¬ (p % 10 = 0))
  (h3 : ¬ (q % 10 = 0))
  (h4 : p > q) : (p - q) % 10 ≠ 5 :=
by sorry

end last_digit_p_minus_q_not_5_l0_325


namespace smallest_positive_period_intervals_of_monotonic_increase_range_of_g_l0_146

def f (x : ℝ) : ℝ := (sin x + cos x)^2 - 2 * (cos x)^2 + (sqrt 2) / 2
def g (x : ℝ) : ℝ := f (x + π / 24)

theorem smallest_positive_period : 
  is_periodic f π :=
sorry

theorem intervals_of_monotonic_increase (k : ℤ) : 
  strictly_increasing_on f (set.Icc (k * π - π / 8) (k * π + 3 * π / 8)) :=
sorry

theorem range_of_g : 
  set.range_on g (set.Icc (π / 6) (2 * π / 3)) = set.Icc 0 (3 * sqrt 2 / 2) :=
sorry

end smallest_positive_period_intervals_of_monotonic_increase_range_of_g_l0_146


namespace sqrt_expression_evaluation_l0_312

theorem sqrt_expression_evaluation : (Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 1)))^4 = 3 + 2 * Real.sqrt 2 := 
by
  sorry

end sqrt_expression_evaluation_l0_312


namespace equality_of_a_and_b_l0_778

theorem equality_of_a_and_b (a b : ℕ) (h : 1 < 4ab - 1 (a b) ∧ (4a^2 - 1)^2 ∣ (4ab - 1))
-- Assuming the necessary conditions directly from the problem.
: a = b := 
sorry -- Proof to be done

end equality_of_a_and_b_l0_778


namespace QC_eq_QD_l0_185

variable (A B C D E F Q : Type) 
variables [EuclideanGeometry A B C D E F Q]

-- Conditions
axiom angle_C_eq_90 : ∠C = 90
axiom CD_perp_AB : ⟦ CD ⟧ ⊥ ⟦ AB ⟧
axiom foot_D : foot CD AB = D
axiom E_incenter_ADC : incenter E (triangle ADC)
axiom F_incenter_BDC : incenter F (triangle BDC)
axiom Q_nine_point_center_CEF : nine_point_circle Q (triangle CEF)

-- Goal
theorem QC_eq_QD : distance Q C = distance Q D :=
sorry

end QC_eq_QD_l0_185


namespace complex_binom_sum_l0_424

theorem complex_binom_sum :
  (1 / (2:ℝ)^1998) * (1 - 3 * (nat.choose 1998 2 : ℕ) + 3^2 * (nat.choose 1998 4 : ℕ) - 3^3 * (nat.choose 1998 6 : ℕ) + 
  ... +
  3^998 * (nat.choose 1998 1996 : ℕ) - 3^999 * (nat.choose 1998 1998 : ℕ)) = 1 := 
by {
  sorry
}

end complex_binom_sum_l0_424


namespace interval_length_f_le_5_l0_75

def f (a : ℝ) : ℕ :=
  -- f(a) counts the number of distinct solutions to the equation
  -- sin (a π x / (x^2 + 1)) + cos (π (x^2 + 4ax + 1) / (4x^2 + 4)) = sqrt(2 - sqrt(2))
  sorry

theorem interval_length_f_le_5 :
  let I := set.Icc (-17/4 : ℝ) (17/4 : ℝ)
  f(a) ≤ 5 -> I.length = 8.5 :=
by
  sorry

end interval_length_f_le_5_l0_75


namespace volume_of_solid_correct_l0_719

-- Definitions based on given conditions
def solid_dimensions : List (ℝ × ℝ × ℝ) := [
  (a, b, c), -- these would be specific dimensional views in cm
  (d, e, f),
  (g, h, i)
]

-- Statement to prove
theorem volume_of_solid_correct : ∃ π: ℝ, volume_of_solid solid_dimensions = 12 * π := sorry

end volume_of_solid_correct_l0_719


namespace smallest_value_of_N_l0_753

theorem smallest_value_of_N :
  ∃ N : ℕ, ∀ (P1 P2 P3 P4 P5 : ℕ) (x1 x2 x3 x4 x5 y1 y2 y3 y4 y5 : ℕ),
    (P1 = 1 ∧ P2 = 2 ∧ P3 = 3 ∧ P4 = 4 ∧ P5 = 5) →
    (x1 = a_1 ∧ x2 = N + a_2 ∧ x3 = 2 * N + a_3 ∧ x4 = 3 * N + a_4 ∧ x5 = 4 * N + a_5) →
    (y1 = 5 * (a_1 - 1) + 1 ∧ y2 = 5 * (a_2 - 1) + 2 ∧ y3 = 5 * (a_3 - 1) + 3 ∧ y4 = 5 * (a_4 - 1) + 4 ∧ y5 = 5 * (a_5 - 1) + 5) →
    (x1 = y2 ∧ x2 = y1 ∧ x3 = y4 ∧ x4 = y5 ∧ x5 = y3) →
    N = 149 :=
sorry

end smallest_value_of_N_l0_753


namespace interval_intersection_exists_l0_777

noncomputable def T (a x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ a then x + (1 - a)
else if a < x ∧ x ≤ 1 then x - a
else 0

theorem interval_intersection_exists (a : ℝ) (ha : 0 < a ∧ a < 1) (J : set ℝ) (hJ : ∃ x y, J = set.Icc x y ∧ x ∈ set.Ioo 0 1 ∧ y ∈ set.Ioo 0 1 ∧ x < y) :
  ∃ n : ℕ, (set.image (λ x, (T^n a x)) J) ∩ J ≠ ∅ :=
sorry

end interval_intersection_exists_l0_777


namespace subset_ratio_l0_139

theorem subset_ratio (S T : ℕ) (hS : S = 256) (hT : T = 56) :
  (T / S : ℚ) = 7 / 32 := by
sorry

end subset_ratio_l0_139


namespace probability_of_odd_numbers_l0_335

def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_of_5_odd_numbers_in_7_rolls : ℚ :=
  (binomial 7 5) * (1 / 2)^5 * (1 / 2)^(7-5)

theorem probability_of_odd_numbers :
  probability_of_5_odd_numbers_in_7_rolls = 21 / 128 :=
by
  sorry

end probability_of_odd_numbers_l0_335


namespace cos_135_eq_neg_sqrt_two_div_two_l0_451

theorem cos_135_eq_neg_sqrt_two_div_two : real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt_two_div_two_l0_451


namespace necessary_but_not_sufficient_l0_130

-- Define α as an interior angle of triangle ABC
def is_interior_angle_of_triangle (α : ℝ) : Prop :=
  0 < α ∧ α < 180

-- Define the sine condition
def sine_condition (α : ℝ) : Prop :=
  Real.sin α = Real.sqrt 2 / 2

-- Define the main theorem
theorem necessary_but_not_sufficient (α : ℝ) (h1 : is_interior_angle_of_triangle α) (h2 : sine_condition α) :
  (sine_condition α) ↔ (α = 45) ∨ (α = 135) := by
  sorry

end necessary_but_not_sufficient_l0_130


namespace positive_difference_of_two_numbers_l0_928

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l0_928


namespace slope_positive_probability_square_l0_849

theorem slope_positive_probability_square :
  let vertices := [(1,0), (0,1), (-1,0), (0,-1)]
  let P := Classical.arbitrary (fin 4)
  let Q := Classical.arbitrary (fin 4)
  ∃ (t u : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 0 ≤ u ∧ u ≤ 1 ∧ 
  ((vertices.nth P).get_or_else (0,0)).1 * (vertices.nth Q).get_or_else (0,0).2 -
  ((vertices.nth P).get_or_else (0,0)).2 * (vertices.nth Q).get_or_else (0,0).1 > 0 :=
  1 / 2 := sorry

end slope_positive_probability_square_l0_849


namespace percent_increase_cube_surface_area_l0_964

theorem percent_increase_cube_surface_area :
  ∀ (s : ℝ), let new_edge := 1.2 * s in
             let original_surface_area := 6 * s^2 in
             let new_surface_area := 6 * (new_edge)^2 in
             ((new_surface_area - original_surface_area) / original_surface_area) * 100 = 44 :=
by
  intro s
  let new_edge := 1.2 * s
  let original_surface_area := 6 * s^2
  let new_surface_area := 6 * (new_edge)^2
  sorry

end percent_increase_cube_surface_area_l0_964


namespace edward_rides_l0_640

theorem edward_rides (initial_tickets spent_tickets ride_cost remaining_tickets rides : ℕ)
  (h1 : initial_tickets = 79)
  (h2 : spent_tickets = 23)
  (h3 : ride_cost = 7)
  (h4 : remaining_tickets = initial_tickets - spent_tickets)
  (h5 : rides = remaining_tickets / ride_cost) :
  rides = 8 :=
by
  rw [h1, h2, h3] at h4 h5
  simp [h4, h5]
  sorry

end edward_rides_l0_640


namespace validate_shots_statistics_l0_882

-- Define the scores and their frequencies
def scores : List ℕ := [6, 7, 8, 9, 10]
def times : List ℕ := [4, 10, 11, 9, 6]

-- Condition 1: Calculate the mode
def mode := 8

-- Condition 2: Calculate the median
def median := 8

-- Condition 3: Calculate the 35th percentile
def percentile_35 := ¬(35 * 40 / 100 = 7)

-- Condition 4: Calculate the average
def average := 8.075

theorem validate_shots_statistics :
  mode = 8
  ∧ median = 8
  ∧ percentile_35
  ∧ average = 8.075 :=
by
  sorry

end validate_shots_statistics_l0_882


namespace suitable_for_census_survey_l0_77

-- Define the conditions
def conditionA : Prop :=
  ∃ t: Type, (t = "Understanding the lethal radius of the newest batch of artillery shells in 2016")

def conditionB : Prop :=
  ∃ t: Type, (t = "Understanding the viewer ratings of Yangquan TV Station's program ⟨XX⟩")

def conditionC : Prop :=
  ∃ t: Type, (t = "Understanding the species of fish in the Yellow River")

def conditionD : Prop :=
  ∃ t: Type, (t = "Understanding the awareness rate of 'Shanxi Spirit' among students in a particular class")

-- Proof statement that option D is suitable for a census survey
theorem suitable_for_census_survey: conditionD :=
begin
  sorry
end

end suitable_for_census_survey_l0_77


namespace max_tied_teams_for_most_wins_l0_754

-- Definitions based on conditions
def num_teams : ℕ := 7
def total_games_played : ℕ := num_teams * (num_teams - 1) / 2

-- Proposition stating the problem and the expected answer
theorem max_tied_teams_for_most_wins : 
  (∀ (t : ℕ), t ≤ num_teams → ∃ w : ℕ, t * w = total_games_played / num_teams) → 
  t = 7 :=
by
  sorry

end max_tied_teams_for_most_wins_l0_754


namespace min_sum_cubic_abs_l0_357

theorem min_sum_cubic_abs (n : ℕ) (x : Fin n → ℝ) (h_n : 3 ≤ n)
  (h_min : ∀ i j : Fin n, i ≠ j → abs (x i - x j) ≥ 1) :
  (∑ k, (abs (x k))^3) ≥ if n % 2 = 1 then ((n^2 - 1)^2) / 32 else (n^2 * (n^2 - 2)) / 32 := 
sorry

end min_sum_cubic_abs_l0_357


namespace positive_difference_l0_909

theorem positive_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
sorry

end positive_difference_l0_909


namespace circle_equation_line_equation_fixed_points_l0_770

-- (1) Prove the equation of circle C
theorem circle_equation (a : ℝ) (r : ℝ) (h1 : 4 * a + 17 = 5 * real.sqrt(16 + 9)) :
  (x - 2)^2 + y^2 = 25 := sorry

-- (2) Prove the equation of line l that intersects circle C at points A and B with AB = 8
theorem line_equation (P : ℝ × ℝ) (hP : P = (-1, 3 / 2)) (AB : ℝ) (hAB : AB = 8) :
  ∃ k : ℝ, 3 * x - 4 * y + 9 = 0 := sorry

-- (3) Prove the fixed points through which the circle passing through A, P, and C must pass
theorem fixed_points (P : ℝ × ℝ) (hp : P.1 + P.2 + 6 = 0)
  (tangent_lines : ∃ A B : ℝ × ℝ, is_tangent (x - 2)^2 + y^2 = 25 P A ∧ is_tangent (x - 2)^2 + y^2 = 25 P B) :
  ∃ (F : ℝ × ℝ) (y1 : ℝ) (y2 : ℝ), 
    F = (2, 0) ∨ F = (-2, -4) := sorry

end circle_equation_line_equation_fixed_points_l0_770


namespace solution_set_f_x_plus_2_lt_5_l0_695

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 4 * x else (abs x)^2 - 4 * abs x

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Lean 4 statement for the proof problem
theorem solution_set_f_x_plus_2_lt_5 :
  even_function f →
  (∀ x, x ≥ 0 → f x = x^2 - 4 * x) →
  { x : ℝ | f (x + 2) < 5 } = set.Ioo (-7 : ℝ) (3 : ℝ) :=
by
  intros h_even h_f
  sorry -- Proof not required

end solution_set_f_x_plus_2_lt_5_l0_695


namespace total_students_mrs_mcgillicuddy_l0_814

-- Define the conditions as variables
def students_registered_morning : ℕ := 25
def students_absent_morning : ℕ := 3
def students_registered_afternoon : ℕ := 24
def students_absent_afternoon : ℕ := 4

-- Prove the total number of students present over the two sessions
theorem total_students_mrs_mcgillicuddy : 
  students_registered_morning - students_absent_morning + students_registered_afternoon - students_absent_afternoon = 42 :=
by
  sorry

end total_students_mrs_mcgillicuddy_l0_814


namespace rectangle_ratio_l0_199

theorem rectangle_ratio 
  (A B C D E F P Q : Point)
  (h1 : is_rectangle A B C D)
  (h2 : distance A B = 8)
  (h3 : distance B C = 4)
  (h4 : E ∈ segment B C)
  (h5 : distance B E = 1)
  (h6 : F ∈ segment B C)
  (h7 : distance E F = 2)
  (h8 : P ∈ line_through A E)
  (h9 : P ∈ line_through B D)
  (h10 : Q ∈ line_through A F)
  (h11 : Q ∈ line_through B D) :
  let r : ℕ := 3
  let s : ℕ := 2
  let t : ℕ := 1
  r + s + t = 6 :=
by sorry

end rectangle_ratio_l0_199


namespace cos_135_eq_neg_inv_sqrt_2_l0_546

theorem cos_135_eq_neg_inv_sqrt_2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 :=
sorry

end cos_135_eq_neg_inv_sqrt_2_l0_546


namespace distance_B_to_E_l0_684

-- Define the problem setup
noncomputable def equilateral_triangle_side : ℝ := 1
noncomputable def distance_B_to_C : ℝ := 2
noncomputable def distance_B_to_D : ℝ := 3

-- State the theorem to prove
theorem distance_B_to_E :
  ∃ E : ℝ, E = sqrt 7 :=
sorry

end distance_B_to_E_l0_684


namespace goods_train_crossing_time_l0_381

def speed_kmh : ℕ := 72
def train_length_m : ℕ := 230
def platform_length_m : ℕ := 290

noncomputable def crossing_time_seconds (speed_kmh train_length_m platform_length_m : ℕ) : ℕ :=
  let distance_m := train_length_m + platform_length_m
  let speed_ms := speed_kmh * 1000 / 3600
  distance_m / speed_ms

theorem goods_train_crossing_time :
  crossing_time_seconds speed_kmh train_length_m platform_length_m = 26 :=
by
  -- The proof should be filled in here
  sorry

end goods_train_crossing_time_l0_381


namespace isosceles_trapezoid_l0_825

theorem isosceles_trapezoid (A B C D : Point) (AC BD : Line) :
  trapezoid A B C D → parallel A B C D → equal_length AC BD → is_isosceles A B C D :=
  by sorry

end isosceles_trapezoid_l0_825


namespace schools_in_pythagoras_town_l0_644

-- Assuming the conditions stated in the problem
constant num_students_per_school : ℕ := 3
constant student_count : ℕ
constant rank_beth : ℕ := 40
constant rank_carla : ℕ := 75

def is_median (rank : ℕ) : Prop := 
  rank > 1 ∧
  ∀ r, r < rank → (rank - 1).even ∧
  ∀ r, r > rank → (r - rank).even

def meets_conditions (rank : ℕ) : Prop :=
  ∃ n, student_count = 2 * rank - 1 ∧ 
  n = student_count / num_students_per_school ∧
  student_count ≥ 75 ∧ 
  rank < 40 ∧ rank > 0 ∧
  ∃ k, 2 * rank - 1 = 3 * k + 2 

-- Andrea's rank is the median, and she scored the highest on her team
def andrea_rank_median (rank : ℕ) : Prop := 
  is_median(rank) ∧ 
  meets_conditions(rank)

def number_of_schools : ℕ :=
  student_count / num_students_per_school

-- Proof problem
theorem schools_in_pythagoras_town : number_of_schools = 25 :=
by 
  sorry

end schools_in_pythagoras_town_l0_644


namespace smallest_positive_n_l0_342

theorem smallest_positive_n : ∃ (n : ℕ), n > 0 ∧ (gcd (5 * n - 3) (11 * n + 4)) > 1 ∧ ∀ (m : ℕ), m > 0 → (gcd (5 * m - 3) (11 * m + 4)) > 1 → n ≤ m :=
begin
  sorry
end

end smallest_positive_n_l0_342


namespace distance_PQ_ge_5_l0_817

-- Definitions based on conditions
def point (P : Type) : Prop := ∃ Q : Type, True

-- Assuming P lies on the angle bisector of AOB
def on_angle_bisector (P : Type) (OA OB : Type → Prop) : Prop := 
  ∀ Q, (PQ_to_OA P OA Q) = (PQ_to_OB P OB Q)

-- P lies on angle bisector of ∠AOB
axiom P_on_angle_bisector : ∀ (P OA OB : Type), on_angle_bisector P OA OB

-- Distance from point P to side OA is 5
def dist_to_side_OA (P OA : Type) : ℝ := 5

-- Point Q is any point on side OB
def point_on_OB (Q OB : Type → Prop) : Prop := OB Q

-- Goal is to prove that for any point Q on side OB, distance PQ is ≥ 5
theorem distance_PQ_ge_5 (P Q OA OB : Type) 
  (P_bisector : on_angle_bisector P OA OB) (dist_P_OA : dist_to_side_OA P OA) 
  (Q_on_OB : point_on_OB Q OB) : dist PQ P Q ≥ 5 := 
by 
  sorry -- Proof to be filled in

end distance_PQ_ge_5_l0_817


namespace cos_135_eq_neg_sqrt2_div_2_l0_482

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by 
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_482


namespace sum_first_8_terms_arith_seq_l0_756

theorem sum_first_8_terms_arith_seq (a : ℕ → ℝ) (a3_eq_5 : a 3 = 5)
  (a4_a8_sum_22 : a 4 + a 8 = 22) :
  ( ∑ i in Finset.range 8, a i) = 64 :=
by
  -- Skipping the proof steps
  sorry

end sum_first_8_terms_arith_seq_l0_756


namespace anand_present_age_l0_6

theorem anand_present_age (A B : ℕ) 
  (h1 : B = A + 10)
  (h2 : A - 10 = (B - 10) / 3) :
  A = 15 :=
sorry

end anand_present_age_l0_6


namespace clothing_price_after_increase_and_discount_l0_19

theorem clothing_price_after_increase_and_discount :
  let original_price := 120
  let increased_price := original_price * (1 + 0.2)
  let final_price := increased_price * 0.8
  final_price < original_price :=
by
  let original_price := 120
  let increased_price := original_price * 1.2
  let final_price := increased_price * 0.8
  sorry

end clothing_price_after_increase_and_discount_l0_19


namespace cos_135_eq_neg_sqrt_two_div_two_l0_456

theorem cos_135_eq_neg_sqrt_two_div_two : real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt_two_div_two_l0_456


namespace positive_difference_of_numbers_l0_926

theorem positive_difference_of_numbers
  (a b : ℝ)
  (h1 : a + b = 10)
  (h2 : a^2 - b^2 = 40) :
  |a - b| = 4 :=
by 
  sorry

end positive_difference_of_numbers_l0_926


namespace cos_shifted_eq_l0_132

noncomputable def cos_shifted (theta : ℝ) (h1 : Real.cos theta = -12 / 13) (h2 : theta ∈ Set.Ioo Real.pi (3 / 2 * Real.pi)) : Real :=
  Real.cos (theta + Real.pi / 4)

theorem cos_shifted_eq (theta : ℝ) (h1 : Real.cos theta = -12 / 13) (h2 : theta ∈ Set.Ioo Real.pi (3 / 2 * Real.pi)) :
  cos_shifted theta h1 h2 = -7 * Real.sqrt 2 / 26 := 
by
  sorry

end cos_shifted_eq_l0_132


namespace solution_set_of_inequality_l0_303

theorem solution_set_of_inequality :
  { x : ℝ | x^2 + x - 2 < 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end solution_set_of_inequality_l0_303


namespace max_value_of_a_l0_674

theorem max_value_of_a (a : ℝ) (h : ∀ k : ℝ, k ∈ set.Icc (-1 : ℝ) 1 → ∀ x : ℝ, x ∈ set.Ioo 0 6 → 6 * real.log x + x^2 - 8 * x + a ≤ k * x) : 
  a ≤ 6 * real.log 6 - 6 :=
sorry

end max_value_of_a_l0_674


namespace cos_135_eq_neg_sqrt2_div_2_l0_622

theorem cos_135_eq_neg_sqrt2_div_2 : 
  Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_622


namespace isosceles_trapezoid_l0_826

theorem isosceles_trapezoid (A B C D : Point) (AC BD : Line) :
  trapezoid A B C D → parallel A B C D → equal_length AC BD → is_isosceles A B C D :=
  by sorry

end isosceles_trapezoid_l0_826


namespace min_val_f1_f2_l0_150

noncomputable def f1 (a : ℝ) (x : ℝ) := Real.exp (|x - 2 * a + 1|)
noncomputable def f2 (a : ℝ) (x : ℝ) := Real.exp (|x - a| + 1)
noncomputable def f (a : ℝ) (x : ℝ) := f1 a x + f2 a x

theorem min_val_f1_f2 (a : ℝ) (h : a = 2) : ∃ x ∈ set.Icc (2 : ℝ) (3 : ℝ), f a x = 2 * Real.exp 1 :=
sorry

end min_val_f1_f2_l0_150


namespace positive_difference_of_two_numbers_l0_934

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l0_934


namespace current_average_age_of_seven_persons_l0_366

theorem current_average_age_of_seven_persons (T : ℕ)
  (h1 : T + 12 = 6 * 43)
  (h2 : 69 = 69)
  : (T + 69) / 7 = 45 := by
  sorry

end current_average_age_of_seven_persons_l0_366


namespace line_plane_intersection_l0_740

variable {Point : Type} [AffineSpace α Point]
variables (l : Line Point) (α : Plane Point)

-- Define that l is not parallel to α as a hypothesis
def not_parallel (l : Line Point) (α : Plane Point) : Prop :=
  ∀ (P Q : Point), P ∈ l → Q ∈ α → (P ≠ Q)

-- Define that l shares common points with α as the conclusion
def shares_common_points (l : Line Point) (α : Plane Point) : Prop :=
  ∃ (P : Point), (P ∈ l) ∧ (P ∈ α)

theorem line_plane_intersection (h : not_parallel l α) : shares_common_points l α :=
by
  sorry

end line_plane_intersection_l0_740


namespace cube_circumscribed_sphere_surface_area_l0_40

theorem cube_circumscribed_sphere_surface_area (a : ℝ) (h : a = 2) : 
  let r := (a * real.sqrt 3) / 2
  let S := 4 * real.pi * r^2
  S = 12 * real.pi :=
by
  have : r = real.sqrt 3
    sorry
  have : S = 4 * real.pi * (real.sqrt 3)^2
    sorry
  rw [this]
  norm_num
  sorry

end cube_circumscribed_sphere_surface_area_l0_40


namespace cos_135_eq_neg_sqrt2_div_2_l0_621

theorem cos_135_eq_neg_sqrt2_div_2 : 
  Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_621


namespace total_moles_NaCl_l0_635

-- Define the moles of reactants
def moles_NaOH : ℕ := 3
def moles_Cl2 : ℕ := 2
def moles_HCl : ℕ := 4

-- Define the reactions
-- Reaction 1: 2NaOH + Cl2 → 2NaCl + H2O
-- Reaction 2: NaOH + HCl → NaCl + H2O

-- Lean statement to prove the total moles of NaCl formed
theorem total_moles_NaCl (moles_NaOH : ℕ) (moles_Cl2 : ℕ) (moles_HCl : ℕ) :
  moles_NaOH = 3 → moles_Cl2 = 2 → moles_HCl = 4 →
  let moles_NaCl1 := min (moles_Cl2 * 2) (moles_NaOH * (2 / 2)) in
  let moles_NaCl2 := min moles_NaOH moles_HCl in
  moles_NaCl1 + moles_NaCl2 = 7 :=
by {
  intros h1 h2 h3,
  let moles_NaCl1 := 2 * 2,
  let moles_NaCl2 := 3,
  show moles_NaCl1 + moles_NaCl2 = 7,
  sorry
}

end total_moles_NaCl_l0_635


namespace cos_135_eq_neg_sqrt_two_div_two_l0_458

theorem cos_135_eq_neg_sqrt_two_div_two : real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt_two_div_two_l0_458


namespace evaluate_f_at_2_l0_112

def f (x : ℕ) : ℕ := 5 * x + 2

theorem evaluate_f_at_2 : f 2 = 12 := by
  sorry

end evaluate_f_at_2_l0_112


namespace find_range_of_product_l0_708

noncomputable def f : ℝ → ℝ :=
λ x, if 0 < x ∧ x < 3 then |Real.log x / Real.log 3| else -Real.cos (π / 3 * x)

theorem find_range_of_product (x₁ x₂ x₃ x₄ : ℝ)
  (h1 : 0 < x₁ ∧ x₁ < 3)
  (h2 : 3 ≤ x₃ ∧ x₃ ≤ 9)
  (h3 : 3 ≤ x₄ ∧ x₄ ≤ 9)
  (hx₁ : f x₁ = f 1)
  (hx₂ : f x₂ = f 1)
  (hx₃ : f x₃ = f 1)
  (hx₄ : f x₄ = f 1) :
  (27 < x₁ * 2 * x₃ * x₄ ∧ x₁ * 2 * x₃ * x₄ < 135 / 4) :=
  sorry

end find_range_of_product_l0_708


namespace extremum_at_zero_f_ge_x_squared_l0_709

-- Definition of the function f(x) for when a = 1
def f (x : ℝ) : ℝ := (Real.exp x - 1) * Real.log (x + 1)

-- Condition: f(x) attains an extremum at x = 0
theorem extremum_at_zero : (deriv f 0) = 0 := by
  -- Proof here
  sorry

-- Condition: f(x) >= x^2 for x >= 0
theorem f_ge_x_squared (x : ℝ) (hx : 0 ≤ x) : f x ≥ x^2 := by
  -- Proof here
  sorry

end extremum_at_zero_f_ge_x_squared_l0_709


namespace sum_a_b_l0_181

theorem sum_a_b (a b : ℕ) 
  (h1 : (∀ n m, (∏ x in Finset.range (b - 3 + 1), (x + 4) / (x + 3)) = (a / 3))) 
  (h2 : (a / 3 = 12)) 
  : a + b = 71 :=
  by
  sorry

end sum_a_b_l0_181


namespace value_of_m_l0_178

theorem value_of_m (m x : ℝ) (h : x - 4 ≠ 0) (hx_pos : x > 0) 
  (eqn : m / (x - 4) - (1 - x) / (4 - x) = 0) : m = 3 := 
by
  sorry

end value_of_m_l0_178


namespace total_students_in_school_l0_974

theorem total_students_in_school 
  (below_8_percent : ℝ) (above_8_ratio : ℝ) (students_8 : ℕ) : 
  below_8_percent = 0.20 → above_8_ratio = 2/3 → students_8 = 12 → 
  (∃ T : ℕ, T = 25) :=
by
  sorry

end total_students_in_school_l0_974


namespace cos_135_eq_neg_sqrt2_div_2_l0_619

theorem cos_135_eq_neg_sqrt2_div_2 : 
  Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_619


namespace cos_135_eq_neg_sqrt2_div_2_l0_591

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  -- Conditions incorporated directly into the requirement due to the nature of trigonometric evaluations.
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_591


namespace unique_solution_l0_260

theorem unique_solution (x : ℝ) (hx : x ≥ 0) : 2021 * x = 2022 * x ^ (2021 / 2022) - 1 → x = 1 :=
by
  intros h
  sorry

end unique_solution_l0_260


namespace triangle_inequality_l0_252

theorem triangle_inequality (a b c R : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b)
  (hR : ∃ (O : ℝ × ℝ × ℝ), 
  ((O.1)^2 + (O.2)^2 + (O.3)^2 ≤ 9 * R^2)) :
  a^2 + b^2 + c^2 ≤ 9 * R^2 :=
sorry

end triangle_inequality_l0_252


namespace total_charge_3_hours_l0_371

-- Define the charges for the first hour (F) and additional hours (A)
variable (F A : ℝ)

-- Given conditions
axiom charge_relation : F = A + 20
axiom total_charge_5_hours : F + 4 * A = 300

-- The theorem stating the total charge for 3 hours of therapy
theorem total_charge_3_hours : 
  (F + 2 * A) = 188 :=
by
  -- Insert the proof here
  sorry

end total_charge_3_hours_l0_371


namespace bus_passes_man_in_time_l0_971

open Real

def length_bus : ℝ := 15
def speed_bus : ℝ := 40
def speed_man : ℝ := 8

noncomputable def time_to_pass (length_bus speed_bus speed_man : ℝ) : ℝ :=
  let relative_speed := (speed_bus + speed_man) * (5 / 18)
  length_bus / relative_speed

theorem bus_passes_man_in_time :
  abs (time_to_pass length_bus speed_bus speed_man - 1.125) < 0.01 :=
by
  sorry

end bus_passes_man_in_time_l0_971


namespace lcm_105_360_eq_2520_l0_959

theorem lcm_105_360_eq_2520 :
  Nat.lcm 105 360 = 2520 :=
by
  have h1 : 105 = 3 * 5 * 7 := by norm_num
  have h2 : 360 = 2^3 * 3^2 * 5 := by norm_num
  rw [h1, h2]
  sorry

end lcm_105_360_eq_2520_l0_959


namespace solution_set_of_inequality_l0_301

theorem solution_set_of_inequality : {x : ℝ | x^2 + x - 2 < 0} = {x : ℝ | -2 < x ∧ x < 1} :=
by
  sorry

end solution_set_of_inequality_l0_301


namespace trapezoid_is_isosceles_l0_823

variable (A B C D : Type) [AffineSpace ℝ A] (a b c d : A)

-- Define that ABCD is a trapezoid with AD and BC as bases
def trapezoid (A B C D: A) [AffineSpace ℝ A]: Prop :=
  ∃ (AD BC: AffineSubspace ℝ A), 
    AD ∈ lineThrough A D ∧ BC ∈ lineThrough B C ∧ AD ∥ BC


-- Define that the diagonals in the trapezoid are of equal length
def equal_diagonals (A B C D: A) [AffineSpace ℝ A]: Prop :=
  dist A C = dist B D

-- Define the theorem statement
theorem trapezoid_is_isosceles (A B C D : A) [AffineSpace ℝ A] :
  trapezoid A B C D → equal_diagonals A B C D → dist A B = dist C D :=
by
  sorry

end trapezoid_is_isosceles_l0_823


namespace cos_135_eq_neg_sqrt2_div_2_l0_505

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = - Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_505


namespace box_volume_l0_354

theorem box_volume
  (L W H : ℝ)
  (h1 : L * W = 120)
  (h2 : W * H = 72)
  (h3 : L * H = 60) :
  L * W * H = 720 :=
by
  -- The proof goes here
  sorry

end box_volume_l0_354


namespace find_h_plus_k_l0_92

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 14*y - 11 = 0

-- State the problem: Prove h + k = -4 given (h, k) is the center of the circle
theorem find_h_plus_k : (∃ h k, (∀ x y, circle_eq x y ↔ (x - h)^2 + (y - k)^2 = 69) ∧ h + k = -4) :=
by {
  sorry
}

end find_h_plus_k_l0_92


namespace frame_interior_edge_sum_l0_23

theorem frame_interior_edge_sum (y : ℝ) :
  ( ∀ outer_edge1 : ℝ, outer_edge1 = 7 →
    ∀ frame_width : ℝ, frame_width = 2 →
    ∀ frame_area : ℝ, frame_area = 30 →
    7 * y - (3 * (y - 4)) = 30) → 
  (7 * y - (4 * y - 12) ) / 4 = 4.5 → 
  (3 + (y - 4)) * 2 = 7 :=
sorry

end frame_interior_edge_sum_l0_23


namespace positive_difference_l0_898

theorem positive_difference (a b : ℝ) (h₁ : a + b = 10) (h₂ : a^2 - b^2 = 40) : |a - b| = 4 :=
by
  sorry

end positive_difference_l0_898


namespace largeville_running_distance_l0_187

theorem largeville_running_distance :
  ∀ (street_width block_side_length : ℕ),
  street_width = 25 →
  block_side_length = 500 →
  let sarah_distance := 4 * block_side_length in
  let sam_distance := 4 * (block_side_length + 2 * street_width) in
  sam_distance - sarah_distance = 200 :=
by
  intros street_width block_side_length h_street_width h_block_side_length
  let sarah_distance := 4 * block_side_length
  let sam_distance := 4 * (block_side_length + 2 * street_width)
  rw [h_street_width, h_block_side_length]
  exact sorry -- Placeholder for the actual proof

end largeville_running_distance_l0_187


namespace proof_cos_135_degree_l0_570

noncomputable def cos_135_degree_is_negative_sqrt2_over_2 : Prop :=
  real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2

theorem proof_cos_135_degree : cos_135_degree_is_negative_sqrt2_over_2 :=
by
  sorry

end proof_cos_135_degree_l0_570


namespace equation_of_line_AB_l0_374

-- Definition of the given circle
def circle1 : Type := { p : ℝ × ℝ // p.1^2 + (p.2 - 2)^2 = 4 }

-- Definition of the center and point on the second circle
def center : ℝ × ℝ := (0, 2)
def point : ℝ × ℝ := (-2, 6)

-- Definition of the second circle with diameter endpoints
def circle2_eq (x y : ℝ) : Prop := (x + 1)^2 + (y - 4)^2 = 5

-- Statement to be proved
theorem equation_of_line_AB :
  ∃ x y : ℝ, (x^2 + (y - 2)^2 = 4) ∧ ((x + 1)^2 + (y - 4)^2 = 5) ∧ (x - 2*y + 6 = 0) := 
sorry

end equation_of_line_AB_l0_374


namespace basketball_team_won_games_l0_988

noncomputable def number_of_games_won (total_games : ℕ) (win_loss_diff : ℕ) : ℕ :=
  let L := (total_games - win_loss_diff) / 2
  let W := L + win_loss_diff
  W

theorem basketball_team_won_games (total_games win_loss_diff : ℕ) (h : total_games = 62) (h_diff : win_loss_diff = 28) :
  number_of_games_won total_games win_loss_diff = 45 :=
by
  rw [h, h_diff]
  unfold number_of_games_won
  norm_num
  sorry

end basketball_team_won_games_l0_988


namespace find_angle_RPB_l0_760

theorem find_angle_RPB (AB_diameter : ∀ (O A B : Point), diameter O A B)
                        (APQ_RBQ_straight : ∀ (P Q R : Point), straight_line P Q ∧ straight_line R B Q)
                        (angle_PAB : ∀ (P A B : Point), angle_deg P A B = 35) :
    ∃ p : ℝ, p = 40 := sorry

end find_angle_RPB_l0_760


namespace bridge_length_l0_281

-- Definitions based on the conditions
def train_length : ℝ := 100
def speed_km_per_hr : ℝ := 45
def crossing_time_seconds : ℝ := 30

-- Conversion factor and derived speed in m/s
def speed_m_per_s : ℝ := (speed_km_per_hr * 1000) / 3600

-- The distance covered in 30 seconds includes length of train and bridge.
def total_distance : ℝ := speed_m_per_s * crossing_time_seconds

-- Stating the theorem to show the length of the bridge
theorem bridge_length (h1 : speed_m_per_s = 12.5)
                      (h2 : total_distance = 375) :
  total_distance = train_length + 275 := by
sorry

end bridge_length_l0_281


namespace minimum_books_borrowed_by_rest_l0_751

variable (TotalStudents : ℕ) (NoBook : ℕ) (OneBook : ℕ) (TwoBooks : ℕ) (AvgBooks : ℕ)
variable (BooksAccounted : ℕ) (RemainingStudents : ℕ) (RemainingBooks : ℕ)

def total_students : Prop := TotalStudents = 20
def no_books_students : Prop := NoBook = 2
def one_book_students : Prop := OneBook = 10
def two_books_students : Prop := TwoBooks = 5
def avg_books_per_student : Prop := AvgBooks = 2

def total_books_required : Prop := TotalStudents * AvgBooks = 40

def books_accounted : Prop := (NoBook * 0) + (OneBook * 1) + (TwoBooks * 2) = 20
def remaining_students : Prop := TotalStudents - (NoBook + OneBook + TwoBooks) = 3
def remaining_books : Prop := TotalBooksRequired - BooksAccounted = 20

theorem minimum_books_borrowed_by_rest :
  total_students →
  no_books_students →
  one_book_students →
  two_books_students →
  avg_books_per_student →
  total_books_required →
  books_accounted →
  remaining_students →
  remaining_books →
  RemainingBooks = 20
:= by
  sorry

end minimum_books_borrowed_by_rest_l0_751


namespace M_transformation_projective_l0_161

-- Definitions for the geometric setup
structure Point :=
(x : ℝ)
(y : ℝ)

structure Line :=
(a : ℝ) -- slope
(b : ℝ) -- y-intercept

-- Parallel lines a and b
def line_a : Line := sorry -- define the actual equation
def line_b : Line := sorry -- define the actual equation

-- A fixed point O
def O : Point := sorry

-- Given any point M
variable (M : Point)

-- Function to get intersection of line with y = mx + c form with another line
def intersection (l1 l2 : Line) : Point := sorry

-- Arbitrary line l through M that does not pass through O, intersects a at A and b at B
def l : Line := sorry
def A : Point := intersection l line_a
def B : Point := intersection l line_b

-- Define the line OM
def line_OM : Line := sorry

-- Define line parallel to OB passing through A
def parallel_to_OB_through_A : Line := sorry

-- Define point M' as the intersection of OM with line parallel to OB through A
def M' : Point := intersection line_OM parallel_to_OB_through_A

-- Theorem: M' does not depend on the choice of the line l
theorem M'_independent_of_l (M : Point) : ∀ l1 l2 : Line, l1 ≠ line_OM ∧ l2 ≠ line_OM →
    intersection line_OM (parallel_to_OB_through_A) = M' :=
sorry

-- Theorem: The transformation mapping M to M' is projective
theorem transformation_projective (M : Point) : ∀ (M' : Point),
    ∃ (P : Point → Point), P(M) = M' ∧
    (∀ (l : Line), ∃ (l' : Line), ∀ (x : Point), x ∈ l → P(x) ∈ l') :=
sorry

end M_transformation_projective_l0_161


namespace time_for_water_level_change_l0_955

variable (S s g V H h : ℝ)

def T : ℝ := 
  2 * (S / (0.6 * s * sqrt (2 * g))) *
  (sqrt H - sqrt (H - h) + (V / (0.6 * s * sqrt (2 * g))) * log (abs ((sqrt H - (V / (0.6 * s * sqrt (2 * g)))) / (sqrt (H - h) - (V / (0.6 * s * sqrt (2 * g)))))))

theorem time_for_water_level_change :
  ∀ (S s g V H h : ℝ), 
  T S s g V H h =
    2 * (S / (0.6 * s * sqrt (2 * g))) *
    (sqrt H - sqrt (H - h) + (V / (0.6 * s * sqrt (2 * g))) * log (abs ((sqrt H - (V / (0.6 * s * sqrt (2 * g)))) / (sqrt (H - h) - (V / (0.6 * s * sqrt (2 * g)))))))
  :=
begin
  sorry
end

end time_for_water_level_change_l0_955


namespace cos_135_eq_neg_sqrt2_div_2_l0_508

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = - Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_508


namespace cosine_135_eq_neg_sqrt_2_div_2_l0_533

theorem cosine_135_eq_neg_sqrt_2_div_2 (cos180 : Real := -1) (cos90 : Real := 0) (cos45 : Real := Real.sqrt 2 / 2)
    (theta1 : Real := 180) (theta2 : Real := 45) (theta3 : Real := 135) :
    theta3 = theta1 - theta2 ∧ 
    Real.cos theta1 = cos180 ∧ 
    Real.cos theta2 = cos45 → 
    Real.cos theta3 = -cos45 := by
  sorry

end cosine_135_eq_neg_sqrt_2_div_2_l0_533


namespace a_and_b_values_f_decreasing_range_of_k_l0_699

variable {θ : ℝ}
noncomputable def f (x : ℝ) : ℝ := (-2^x + 1) / (2^(x + 1) + 2)

theorem a_and_b_values :
  (∀ x : ℝ, f (-x) = -f x) → (2 = 2 ∧ 1 = 1) :=
sorry

theorem f_decreasing :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂) :=
sorry

theorem range_of_k (k : ℝ) :
  (∀ θ ∈ set.Ioo (-(π / 2)) (π / 2), f k + f (cos θ ^ 2 - 2 * sin θ) ≤ 0) → k > -2 :=
sorry

end a_and_b_values_f_decreasing_range_of_k_l0_699


namespace circle_center_and_radius_locus_of_midpoint_l0_202

-- Part 1: Prove the equation of the circle C:
theorem circle_center_and_radius (a b r: ℝ) (hc: a + b = 2):
  (4 - a)^2 + b^2 = r^2 →
  (2 - a)^2 + (2 - b)^2 = r^2 →
  a = 2 ∧ b = 0 ∧ r = 2 := by
  sorry

-- Part 2: Prove the locus of the midpoint M:
theorem locus_of_midpoint (x y : ℝ) :
  ∃ (x1 y1 : ℝ), (x1 - 2)^2 + y1^2 = 4 ∧ x = (x1 + 5) / 2 ∧ y = y1 / 2 →
  x^2 - 7*x + y^2 + 45/4 = 0 := by
  sorry

end circle_center_and_radius_locus_of_midpoint_l0_202


namespace area_inside_circle_outside_square_l0_750

theorem area_inside_circle_outside_square :
  ∀ (r : ℝ), r = 5 →
  let circle_area := π * r^2,
      square_side := r * √2,
      square_area := (square_side)^2 in
  (circle_area - square_area) = 25 * π - 50 :=
by
  intros r hr,
  simp [hr, sqr],
  sorry

end area_inside_circle_outside_square_l0_750


namespace even_function_inequality_l0_148

theorem even_function_inequality (a b : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : ∀ x, f x = a ^ abs (x + b)) (h₃ : ∀ x, f (-x) = f x) :
  f (b - 3) < f (a + 2) :=
by
  sorry

end even_function_inequality_l0_148


namespace cos_135_eq_neg_sqrt_two_div_two_l0_455

theorem cos_135_eq_neg_sqrt_two_div_two : real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt_two_div_two_l0_455


namespace grapefruits_count_l0_997

def citrus_orchards : Prop :=
  ∃ (total lemons oranges limes grapefruits : ℕ),
    (total = 16) ∧ (lemons = 8) ∧ (oranges = lemons / 2) ∧ 
    (limes + grapefruits = total - (lemons + oranges)) ∧ 
    (limes = grapefruits) ∧ (grapefruits = 2)

theorem grapefruits_count : citrus_orchards :=
begin
  unfold citrus_orchards,
  use 16,
  use 8,
  use 4,
  use 2,
  use 2,
  split,
  { refl },
  split,
  { refl },
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { refl },
  refl,
end

end grapefruits_count_l0_997


namespace sector_area_ratio_l0_247

theorem sector_area_ratio (O : Type) [MetricSpace O] [InnerProductSpace ℝ O]
  (A E F : O) (h1 : ∠ O A E = 40) (h2 : ∠ F O A = 60) :
  ∃ (r : ℝ), (r = 13 / 18) :=
by
  sorry

end sector_area_ratio_l0_247


namespace number_of_girls_l0_880

-- Define the number of girls and boys
variables (G B : ℕ)

-- Define the conditions
def condition1 : Prop := B = 2 * G - 16
def condition2 : Prop := G + B = 68

-- The theorem we want to prove
theorem number_of_girls (h1 : condition1 G B) (h2 : condition2 G B) : G = 28 :=
by
  sorry

end number_of_girls_l0_880


namespace max_hop_sum_l0_65

-- Define the problem context
def frog_hops_set (n : ℕ) : set ℕ := { k | 1 ≤ k ∧ k < 2^n }

-- Conditions of the problem
def frog_conditions (n : ℕ) : Prop :=
  (∀ k ∈ frog_hops_set n, ∃ l : ℕ, k = 2^l) ∧
  ∀ k ∈ frog_hops_set n, k < 2^n

-- Formally stating the problem
theorem max_hop_sum (n : ℕ) (hn : n > 0) :
  frog_conditions n → max_sum n = (4^n - 1) / 3 :=
sorry

end max_hop_sum_l0_65


namespace cos_135_eq_correct_l0_614

noncomputable def cos_135_eq: Prop :=
  real.cos ((135 * real.pi) / 180) = -real.sqrt 2 / 2

theorem cos_135_eq_correct : cos_135_eq :=
  sorry

end cos_135_eq_correct_l0_614


namespace determine_knights_liars_l0_402

noncomputable def natives_circle (n : ℕ) : Prop :=
  ∃ (knights liars : Fin n → Bool), 
  ∀ i : Fin n,
    (knights i → 
      (statement_age_left i = age (left_neighbor i) ∧ statement_age_right i = age (right_neighbor i))) ∧ 
    (liars i → 
      ((statement_age_left i = age (left_neighbor i) + 1 ∨ statement_age_left i = age (left_neighbor i) - 1) ∧
       (statement_age_right i = age (right_neighbor i) + 1 ∨ statement_age_right i = age (right_neighbor i) - 1)))

theorem determine_knights_liars :
  natives_circle 50 →
  ∃ (f : Fin 50 → Bool), 
  (∀ i : Fin 50, f i = true ↔ is_knight i) ∧ (∀ i : Fin 50, f i = false ↔ is_liar i) :=
sorry

end determine_knights_liars_l0_402


namespace wrongly_noted_mark_l0_265

/-- 
  The average marks of 10 students in a class is 100.
  A student's mark is wrongly noted instead of 10.
  The correct average marks is 96.
  We need to prove that the wrongly noted mark is 50.
-/
theorem wrongly_noted_mark 
  (n : ℕ)
  (avg_wrong : ℕ) 
  (correct_mark : ℕ) 
  (avg_correct : ℕ) 
  (wrongly_noted : ℕ) 
  (h1 : n = 10)
  (h2 : avg_wrong = 100)
  (h3 : correct_mark = 10)
  (h4 : avg_correct = 96)
  (h5 : wrongly_noted = 50)
  : 
  let sum_wrong := n * avg_wrong,
      sum_correct := n * avg_correct in
  sum_wrong - sum_correct = wrongly_noted - correct_mark :=
by
  -- Proof goes here
  sorry

end wrongly_noted_mark_l0_265


namespace domain_of_f_value_of_f_l0_669

noncomputable def f (x : ℝ) : ℝ := (x^2 - 1) / 2

theorem domain_of_f : ∀ (x : ℝ), f (x) ∈ (-∞, ∞) → x ∈ [-REAL.sqrt 2, REAL.sqrt 2] :=
sorry

theorem value_of_f :
  f (Real.sin (Real.pi / 6)) = -3 / 8 :=
by
  sorry

end domain_of_f_value_of_f_l0_669


namespace cos_135_eq_neg_sqrt2_div_2_l0_473

theorem cos_135_eq_neg_sqrt2_div_2 : cos (135 * real.pi / 180) = - (real.sqrt 2 / 2) :=
by
  have h1 : 135 = 180 - 45 := by norm_num
  have h2 : cos (180 * real.pi / 180 - 45 * real.pi / 180) = - cos (45 * real.pi / 180) :=
    by rw [cos_sub, cos_pi, cos_pi_div_four, sin_pi, sin_pi_div_four]; norm_num
  have h3 : cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by norm_num
  rw [h1, ←h2, h3]
  norm_num
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_473


namespace cos_135_eq_neg_inv_sqrt_2_l0_525

theorem cos_135_eq_neg_inv_sqrt_2 : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_525


namespace cos_135_eq_neg_sqrt2_div_2_l0_507

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = - Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_507


namespace circle_chord_area_l0_192

noncomputable def part_circle_area_between_chords (R : ℝ) : ℝ :=
  (R^2 * (Real.pi + Real.sqrt 3)) / 2

theorem circle_chord_area (R : ℝ) :
  ∀ (a₃ a₆ : ℝ),
    a₃ = Real.sqrt 3 * R →
    a₆ = R →
    part_circle_area_between_chords R = (R^2 * (Real.pi + Real.sqrt 3)) / 2 :=
by
  intros a₃ a₆ h₁ h₂
  sorry

end circle_chord_area_l0_192


namespace cos_135_eq_correct_l0_610

noncomputable def cos_135_eq: Prop :=
  real.cos ((135 * real.pi) / 180) = -real.sqrt 2 / 2

theorem cos_135_eq_correct : cos_135_eq :=
  sorry

end cos_135_eq_correct_l0_610


namespace number_of_real_values_of_c_l0_99

theorem number_of_real_values_of_c : 
  ∃ c1 c2 : ℝ, 
    (abs ((1 / 3 : ℝ) + complex.i * (- c1)) = 2 / 3 ∧
     abs ((1 / 3 : ℝ) + complex.i * (- c2)) = 2 / 3) ∧ 
    c1 ≠ c2 :=
sorry

end number_of_real_values_of_c_l0_99


namespace determine_truthfulness_of_natives_l0_404

-- Definitions of the problem in Lean
/-- Each native has a position in the circle and an age in years. -/
structure Native :=
  (pos : Nat) -- position in the circle (1 through 50)
  (leftAge : Nat) -- age of left neighbor as declared by this native
  (rightAge : Nat) -- age of right neighbor as declared by this native
  (truthfulness : Bool) -- True if the native is a knight, False if a liar

/-- Given conditions: There are 50 natives in a circle. -/
constant natives : Fin 50 → Native

/-- Each knight declares the age of their neighbors correctly. -/
axiom knights_correct (i : Fin 50) : (natives i).truthfulness = true →
  (natives i).leftAge = (natives ((i.val + 49) % 50)).pos ∧
  (natives i).rightAge = (natives ((i.val + 1) % 50)).pos

/-- Each liar increases one age by 1 and decreases the other by 1 of their choice. -/
axiom liars_incorrect (i : Fin 50) : (natives i).truthfulness = false →
  ((natives i).leftAge = (natives ((i.val + 49) % 50)).pos + 1 ∧
   (natives i).rightAge = (natives ((i.val + 1) % 50)).pos - 1) ∨
  ((natives i).leftAge = (natives ((i.val + 49) % 50)).pos - 1 ∧
   (natives i).rightAge = (natives ((i.val + 1) % 50)).pos + 1)

-- Statement of the problem to prove
theorem determine_truthfulness_of_natives :
  ∃ f : Fin 50 → Bool, ∀ i : Fin 50, f i = (natives i).truthfulness :=
sorry -- proof omitted

end determine_truthfulness_of_natives_l0_404


namespace hex_area_eq_156_l0_271

open set function

/-- Define the areas of the triangles and the extended points as conditions -/
def area_ABC : ℝ := 12
def PA_eq_AB_eq_BS (A B S : ℝ) : Prop := PA = AB ∧ AB = BS
def QA_eq_AC_eq_CT (A C T : ℝ) : Prop := QA = AC ∧ AC = CT
def RB_eq_BC_eq_CU (B C U : ℝ) : Prop := RB = BC ∧ BC = CU

/-- Prove that the area of hexagon PQRSTU is 156 square centimeters -/
theorem hex_area_eq_156
  (PA AB BS : ℝ) (QA AC CT : ℝ) (RB BC CU : ℝ)
  (h1 : PA_eq_AB_eq_BS PA AB BS)
  (h2 : QA_eq_AC_eq_CT QA AC CT)
  (h3 : RB_eq_BC_eq_CU RB BC CU)
  (hABC : area_ABC = 12) :
  hexagon_area P Q R S T U = 156 :=
sorry

end hex_area_eq_156_l0_271


namespace max_value_m_l0_852

noncomputable def exists_triangle_with_sides (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_value_m (a b c : ℝ) (m : ℝ) (h1 : 0 < m) (h2 : abc ≤ 1/4) (h3 : 1/(a^2) + 1/(b^2) + 1/(c^2) < m) :
  m ≤ 9 ↔ exists_triangle_with_sides a b c :=
sorry

end max_value_m_l0_852


namespace arithmetic_difference_l0_176

variables (p q r : ℝ)

theorem arithmetic_difference (h1 : (p + q) / 2 = 10) (h2 : (q + r) / 2 = 27) : r - p = 34 :=
by
  sorry

end arithmetic_difference_l0_176


namespace employed_females_percent_l0_762

theorem employed_females_percent (total_employable_population : ℕ) (total_employed : ℕ)
    (employed_males : ℕ) (H1 : total_employed = 120 * total_employable_population / 100)
    (H2 : employed_males = 80 * total_employed / 100) :
    (total_employed - employed_males) * 100 / total_employed = 33.33 :=
by
    sorry

end employed_females_percent_l0_762


namespace cos_135_eq_neg_sqrt2_div_2_l0_479

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by 
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_479


namespace max_distance_from_P_to_A_l0_391

-- Define the points and distances
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, 0)
def C : ℝ × ℝ := (2, 2)
def D : ℝ × ℝ := (0, 2)

-- Distance functions
def v (x y : ℝ) : ℝ := real.sqrt ((x - 2)^2 + y^2)
def w (x y : ℝ) : ℝ := real.sqrt ((x - 2)^2 + (y - 2)^2)
def t (x y : ℝ) : ℝ := real.sqrt (x^2 + (y - 2)^2)

-- Given condition
def condition (x y : ℝ) : Prop := (v x y)^2 + (w x y)^2 = (t x y)^2

-- Prove the greatest distance from P to A is sqrt(10), given the condition
theorem max_distance_from_P_to_A (x y : ℝ) (h : condition x y) : real.sqrt (x^2 + y^2) ≤ real.sqrt 10 :=
sorry

end max_distance_from_P_to_A_l0_391


namespace range_of_m_l0_795

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3*x + 2

theorem range_of_m (m : ℝ) :
  (∀ θ : ℝ, f (3 + 2 * Real.sin θ) < m) → m > 4 :=
sorry

end range_of_m_l0_795


namespace solve_for_x_l0_174

theorem solve_for_x {x : ℤ} (h : x - 2 * x + 3 * x - 4 * x = 120) : x = -60 :=
sorry

end solve_for_x_l0_174


namespace equal_share_of_land_l0_0

variables {AB BC : ℝ} (O : ℝ → ℝ) -- AB and BC are the lengths, O is a point function
def area_parallelogram (AB HK : ℝ) : ℝ := AB * HK

def area_triangle (base height : ℝ) : ℝ := 1 / 2 * base * height

theorem equal_share_of_land (AB BC HK OH OK : ℝ) (h_AB_longer : AB > BC) 
    (h_AB_CD : AB = HK) -- since AB = CD
    (h_OH_OK : OH + OK = HK) :
    let area_total := area_parallelogram AB HK in
    let area_pierre := area_triangle AB OH + area_triangle AB OK in
    let area_jean := area_total - area_pierre in
    area_pierre = area_jean :=
by
    sorry

end equal_share_of_land_l0_0


namespace acme_horseshoes_production_l0_405

theorem acme_horseshoes_production
  (profit : ℝ)
  (initial_outlay : ℝ)
  (cost_per_set : ℝ)
  (selling_price : ℝ)
  (number_of_sets : ℕ) :
  profit = selling_price * number_of_sets - (initial_outlay + cost_per_set * number_of_sets) →
  profit = 15337.5 →
  initial_outlay = 12450 →
  cost_per_set = 20.75 →
  selling_price = 50 →
  number_of_sets = 950 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end acme_horseshoes_production_l0_405


namespace major_axis_length_l0_832

noncomputable def rectangle_area (m n : ℝ) : ℝ := m * n
noncomputable def ellipse_area (a b : ℝ) : ℝ := π * a * b
noncomputable def ellipse_condition_1 (m n a : ℝ) : Prop := m + n = 2 * a
noncomputable def ellipse_condition_2 (m n a b : ℝ) : Prop := m^2 + n^2 = 4 * (a^2 - b^2)

theorem major_axis_length 
  (m n a b : ℝ) 
  (h1 : rectangle_area m n = 4050) 
  (h2 : ellipse_area a b = 3240 * π) 
  (h3 : ellipse_condition_1 m n a) 
  (h4 : ellipse_condition_2 m n a b) : 
  2 * a = 144 := 
by 
  sorry

end major_axis_length_l0_832


namespace cos_135_eq_neg_sqrt2_div_2_l0_514

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = - Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_514


namespace cos_135_eq_neg_inv_sqrt2_l0_446

theorem cos_135_eq_neg_inv_sqrt2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt2_l0_446


namespace artist_paints_33_square_meters_l0_44

/-
Conditions:
1. The artist has 14 cubes.
2. Each cube has an edge of 1 meter.
3. The cubes are arranged in a pyramid-like structure with three layers.
4. The top layer has 1 cube, the middle layer has 4 cubes, and the bottom layer has 9 cubes.
-/

def exposed_surface_area (num_cubes : Nat) (layer1 : Nat) (layer2 : Nat) (layer3 : Nat) : Nat :=
  let layer1_area := 5 -- Each top layer cube has 5 faces exposed
  let layer2_edge_cubes := 4 -- Count of cubes on the edge in middle layer
  let layer2_area := layer2_edge_cubes * 3 -- Each middle layer edge cube has 3 faces exposed
  let layer3_area := 9 -- Each bottom layer cube has 1 face exposed
  let top_faces := layer1 + layer2 + layer3 -- All top faces exposed
  layer1_area + layer2_area + layer3_area + top_faces

theorem artist_paints_33_square_meters :
  exposed_surface_area 14 1 4 9 = 33 := 
sorry

end artist_paints_33_square_meters_l0_44


namespace no_solution_for_x_l0_352

theorem no_solution_for_x (x : ℝ) :
  (1 / (x + 4) + 1 / (x - 4) = 1 / (x - 4)) → False :=
by
  sorry

end no_solution_for_x_l0_352


namespace cos_135_eq_correct_l0_608

noncomputable def cos_135_eq: Prop :=
  real.cos ((135 * real.pi) / 180) = -real.sqrt 2 / 2

theorem cos_135_eq_correct : cos_135_eq :=
  sorry

end cos_135_eq_correct_l0_608


namespace cos_135_eq_neg_inv_sqrt_2_l0_583

theorem cos_135_eq_neg_inv_sqrt_2 :
  cos (135 * Real.pi / 180) = - (1 / Real.sqrt 2) :=
by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_583


namespace smallest_n_must_contain_median_triangle_l0_415

def no_collinear_three_points (S : Set Point) : Prop :=
  ∀ (A B C : Point), A ∈ S → B ∈ S → C ∈ S → ¬ collinear A B C

def distinct_distances (S : Set Point) : Prop :=
  ∀ (A B C D : Point), A ∈ S → B ∈ S → C ∈ S → D ∈ S → (A ≠ B → distance A B ≠ distance C D)

def median_edge_of_S (S : Set Point) (A B : Point) : Prop :=
  ∃ (C : Point), C ∈ S ∧ |A - C| < |A - B| ∧ |A - B| < |B - C|

def median_triangle_of_S (S : Set Point) (A B C : Point) : Prop :=
  median_edge_of_S S A B ∧ median_edge_of_S S A C ∧ median_edge_of_S S B C

def contains_median_triangle (S : Set Point) : Prop :=
  ∃ (A B C : Point), A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ median_triangle_of_S S A B C

theorem smallest_n_must_contain_median_triangle :
  ∀ (S : Set Point), (no_collinear_three_points S ∧ distinct_distances S) → card S = 6 → contains_median_triangle S :=
sorry

end smallest_n_must_contain_median_triangle_l0_415


namespace inequality_solution_set_l0_297

theorem inequality_solution_set : 
  { x : ℝ | x^2 + x - 2 < 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end inequality_solution_set_l0_297


namespace minimum_value_amgm_l0_228

theorem minimum_value_amgm (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 27) : (a + 3 * b + 9 * c) ≥ 27 :=
by
  sorry

end minimum_value_amgm_l0_228


namespace seller_loss_l0_13

-- Define the values involved
def item_cost := 20
def fake_banknote := 100
def real_banknote := 100
def change_given := 80

-- Define the conditions and the final loss calculation
theorem seller_loss :
  let loss := item_cost + change_given + real_banknote
  loss = 200 :=
by
  -- Introduce the definitions
  let loss := item_cost + change_given + real_banknote
  -- Assert that the total loss equals 200
  show loss = 200 from sorry

end seller_loss_l0_13


namespace eggs_in_each_basket_is_15_l0_800
open Nat

theorem eggs_in_each_basket_is_15 :
  ∃ n : ℕ, (n ∣ 30) ∧ (n ∣ 45) ∧ (n ≥ 5) ∧ (n = 15) :=
sorry

end eggs_in_each_basket_is_15_l0_800


namespace total_students_mrs_mcgillicuddy_l0_813

-- Define the conditions as variables
def students_registered_morning : ℕ := 25
def students_absent_morning : ℕ := 3
def students_registered_afternoon : ℕ := 24
def students_absent_afternoon : ℕ := 4

-- Prove the total number of students present over the two sessions
theorem total_students_mrs_mcgillicuddy : 
  students_registered_morning - students_absent_morning + students_registered_afternoon - students_absent_afternoon = 42 :=
by
  sorry

end total_students_mrs_mcgillicuddy_l0_813


namespace sin_ratio_BAD_CAD_l0_764

-- Definitions corresponding to conditions
def AngleB : ℝ := 45
def AngleC : ℝ := 60
def ratioBDCD := 2 / 1

-- Theorem statement corresponding to the question and answer
theorem sin_ratio_BAD_CAD (A B C D : Type) [Triangle A B C] [angle_eq B 45] [angle_eq C 60] [point D divides_segment (segment B C) 2 1] :
  (sin (angle A B D) / sin (angle A C D)) = (2 * Real.sqrt 6 / 3) := by
  sorry

end sin_ratio_BAD_CAD_l0_764


namespace ratio_boys_total_l0_193

theorem ratio_boys_total (p : ℝ) (q : ℝ) (h : q = 1 - p) (h_prob : p = 3 / 5 * q) :
  p = 3 / 8 :=
by
  rw [h] at h_prob,
  have h': p = 3 / 5 * (1 - p), by assumption,
  -- Proof steps omitted
  sorry

end ratio_boys_total_l0_193


namespace cos_135_eq_neg_sqrt2_div_2_l0_511

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = - Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_511


namespace ratio_of_first_term_to_common_difference_l0_306

theorem ratio_of_first_term_to_common_difference (a d : ℕ) (h : 15 * a + 105 * d = 3 * (5 * a + 10 * d)) : a = 5 * d :=
by
  sorry

end ratio_of_first_term_to_common_difference_l0_306


namespace brother_book_pages_l0_345

variable (Ryan_pages_per_week : ℕ) (days_per_week : ℕ) 
variable (Ryan_average : ℕ) (brother_average : ℕ)

theorem brother_book_pages 
  (Ryan_pages_eq : Ryan_pages_per_week = 2100) 
  (days_eq : days_per_week = 7) 
  (Ryan_average_eq : Ryan_average = Ryan_pages_per_week / days_per_week) 
  (Ryan_brother_diff_eq : Ryan_average = brother_average + 100) 
  : brother_average = 200 :=
by
  have Ryan_pages : Ryan_average = 2100 / 7, from sorry
  have brother_average_calc : brother_average + 100 = 300, from sorry
  have brother_average_eq : brother_average = 200, from sorry
  exact brother_average_eq

end brother_book_pages_l0_345


namespace percentage_reduced_l0_990

theorem percentage_reduced (P : ℝ) : (200 * (P / 100)) - 12 = 178 → P = 95 := 
by 
  intro h 
  have : 200 * (P / 100) = 190 := by linarith
  have : P / 100 = 0.95 := by linarith
  have : P = 0.95 * 100 := by linarith
  linarith

end percentage_reduced_l0_990


namespace beaver_paths_l0_37

noncomputable def catalan (n : ℕ) : ℕ :=
  nat.div (nat.choose (2 * n) n) (n + 1)

theorem beaver_paths : catalan 4 = 14 :=
by
  sorry

end beaver_paths_l0_37


namespace cone_fraction_l0_394

theorem cone_fraction
  (r h : ℝ)
  (nonzero_r : r ≠ 0)
  (cone_condition : r^2 + h^2 = 400 * r^2) :
  (h / r = real.sqrt 399) ∧ (1 + 399 = 400) :=
by
  sorry

end cone_fraction_l0_394


namespace min_abs_of_complex_number_l0_229

-- Define the conditions
variable (z : ℂ)
axiom condition : |z - 6| + |z - 𝓘 * 5| = 7

-- Define the theorem we need to prove
theorem min_abs_of_complex_number :
  (∃ z : ℂ, |z - 6| + |z - 𝓘 * 5| = 7) → ∃ z : ℂ, |z| = 30 / 7 :=
by
  sorry

end min_abs_of_complex_number_l0_229


namespace cos_135_eq_neg_inv_sqrt_2_l0_527

theorem cos_135_eq_neg_inv_sqrt_2 : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_527


namespace work_together_time_l0_348

/--
Given that:
1. Worker A can complete a task in 13 days.
2. Worker B can complete the same task in approximately 11.142857142857144 days.

We want to prove that together worker A and worker B can complete the task in approximately 6 days.
-/
theorem work_together_time :
  let A_time := 13
  let B_time := 11.142857142857144
  let combined_time := 6 in
  1 / (1 / A_time + 1 / B_time) = combined_time :=
by
  sorry

end work_together_time_l0_348


namespace line_through_points_l0_272

theorem line_through_points (m b: ℝ) 
  (h1: ∃ m, ∀ x y : ℝ, ((x, y) = (1, 3) ∨ (x, y) = (3, 7)) → y = m * x + b) 
  (h2: ∀ x y : ℝ, ((x, y) = (1, 3) ∨ (x, y) = (3, 7)) → y = m * x + b):
  m + b = 3 :=
by
  -- proof goes here
  sorry

end line_through_points_l0_272


namespace polynomial_form_l0_783

variable {R : Type*} [Field R]

theorem polynomial_form (g : R[X]) (h_monic : g.leading_coeff = 1) (h_deg : g.natDegree = 2) (h_g0 : g.eval 0 = 6) (h_g1 : g.eval 1 = 12) :
  g = X^2 + 5 * X + 6 := by
  sorry

end polynomial_form_l0_783


namespace pizza_eaten_after_six_trips_l0_346

theorem pizza_eaten_after_six_trips :
  let a := (1 : ℝ) / 3
  let r := (1 : ℝ) / 3
  let n := 6
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 364 / 729 :=
by
  let a := (1 : ℝ) / 3
  let r := (1 : ℝ) / 3
  let n := 6
  let S_n := a * (1 - r^n) / (1 - r)
  have : S_n = (1 / 3) * (1 - (1 / 3)^6) / (1 - 1 / 3) := by sorry
  have : S_n = 364 / 729 := by sorry
  exact this

end pizza_eaten_after_six_trips_l0_346


namespace black_pens_count_l0_808

theorem black_pens_count (red_pens blue_pens total_pens : ℕ) (h1 : red_pens = 65) (h2 : blue_pens = 45) (h3 : total_pens = 168) : total_pens - (red_pens + blue_pens) = 58 :=
by
  rw [h1, h2, h3]
  sorry

end black_pens_count_l0_808


namespace distinct_distances_l0_114

theorem distinct_distances (k : ℕ) (h :  k ≥ 1) (points : fin (3 * k + 2) → ℝ × ℝ)
  (h_no_collinear : ∀ i j l, i ≠ j → j ≠ l → i ≠ l → ¬collinear ({points i, points j, points l} : set (ℝ × ℝ))) :
  ∃ p ∈ (fin (3 * k + 2) → ℝ × ℝ), 
  (set.card {dist | ∃ q, q ≠ p ∧ dist = dist (points p) (points q)} ≥ k + 1) :=
sorry

end distinct_distances_l0_114


namespace smallest_discount_l0_66

theorem smallest_discount 
  (m : ℕ)
  (m_effective : ℕ → ℕ := λ x, x) -- m_effective will represent the discount provided by m%
  (discount1_effective : ℕ := 28) -- 28% for the first discount scheme
  (discount2_effective : ℕ := 22) -- An integer approximation for 22.1312%
  (discount3_effective : ℕ := 22) -- An integer approximation for 22.56%
  (h1 : m > discount1_effective)
  (h2 : m > discount2_effective)
  (h3 : m > discount3_effective)
  : m > 28 := by
  sorry

end smallest_discount_l0_66


namespace least_four_digit_11_heavy_l0_35

def is_11_heavy (n : ℕ) : Prop := (n % 11) > 7

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

theorem least_four_digit_11_heavy : ∃ n : ℕ, is_four_digit n ∧ is_11_heavy n ∧ 
  (∀ m : ℕ, is_four_digit m ∧ is_11_heavy m → 1000 ≤ n) := 
sorry

end least_four_digit_11_heavy_l0_35


namespace cos_135_eq_neg_sqrt2_div_2_l0_512

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = - Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_512


namespace nine_sided_convex_polygon_diagonals_l0_18

theorem nine_sided_convex_polygon_diagonals :
  ∃ (n : ℕ), n = 9 ∧ (n * (n - 3)) / 2 = 27 :=
by
  existsi 9
  split
  · rfl
  · norm_num
  sorry

end nine_sided_convex_polygon_diagonals_l0_18


namespace no_valid_m_for_monotonically_decreasing_l0_154

theorem no_valid_m_for_monotonically_decreasing :
  ∀ (f : ℝ → ℝ) (m : ℤ),
  f = λ x, (m-1)^2 * x^(m^2 - 4*m + 2) →
  ¬ (∀ x > 0, ∀ y > x, f y < f x) :=
begin
  intros f m h,
  sorry -- Proof omitted
end

end no_valid_m_for_monotonically_decreasing_l0_154


namespace teal_more_green_count_l0_363

open Set

-- Define the survey data structure
def Survey : Type := {p : ℕ // p ≤ 150}

def people_surveyed : ℕ := 150
def more_blue (s : Survey) : Prop := sorry
def more_green (s : Survey) : Prop := sorry

-- Define the given conditions
def count_more_blue : ℕ := 90
def count_more_both : ℕ := 40
def count_neither : ℕ := 20

-- Define the proof statement
theorem teal_more_green_count :
  (count_more_both + (people_surveyed - (count_neither + (count_more_blue - count_more_both)))) = 80 :=
by {
  -- Sorry is used as a placeholder for the proof
  sorry
}

end teal_more_green_count_l0_363


namespace right_triangle_distance_l0_204

theorem right_triangle_distance (x h d : ℝ) :
  x + Real.sqrt ((x + 2 * h) ^ 2 + d ^ 2) = 2 * h + d → 
  x = (h * d) / (2 * h + d) :=
by
  intros h_eq_d
  sorry

end right_triangle_distance_l0_204


namespace sum_x_coordinates_Q4_l0_629

variable (n : ℕ)
variable (Q : Fin n → ℝ × ℝ)

def sum_x_coordinates (polygon : Fin n → ℝ × ℝ) : ℝ :=
  ∑ i, (polygon i).fst

theorem sum_x_coordinates_Q4 (h1 : n = 40)
  (h2 : sum_x_coordinates Q = 120)
  (Q2 : Fin n → ℝ × ℝ := λ i, ((Q i).fst + (Q ((i + 1) % n)).fst) / 2, (Q i).snd + (Q ((i + 1) % n)).snd) / 2)
  (Q3 : Fin n → ℝ × ℝ := λ i, ((Q2 i).fst + (Q2 ((i + 1) % n)).fst) / 2, (Q2 i).snd + (Q2 ((i + 1) % n)).snd) / 2)
  (Q4 : Fin n → ℝ × ℝ := λ i, ((Q3 i).fst + (Q3 ((i + 1) % n)).fst) / 2, (Q3 i).snd + (Q3 ((i + 1) % n)).snd) / 2) :
  sum_x_coordinates Q4 = 120 := by
  sorry

end sum_x_coordinates_Q4_l0_629


namespace hyperbola_eccentricity_l0_977

variables {a b : ℝ} (h_a_gt0 : a > 0) (h_b_gt0 : b > 0)

def hyperbola (x y : ℝ) : Prop := 
  x^2 / a^2 - y^2 / b^2 = 1

def F1 (c : ℝ) := (-c, 0)
def F2 (c : ℝ) := (c, 0)
def c (a b : ℝ) := Real.sqrt (a^2 + b^2)

-- Definitions for points and conditions on the intersection of the line
variables (c_gt_a: c a b > a)
          (A B : ℝ × ℝ)
          (h_AF1 : ((A.1 + c a b)^2 + A.2^2) = (2 * c a b)^2)
          (h_BF2 : ((B.1 - c a b)^2 + B.2^2) = (2 * ((A.1 - c a b)^2 + A.2^2)))

theorem hyperbola_eccentricity : 
  (eccentricity : ℝ) := c a b / a = 5 / 3 :=
sorry

end hyperbola_eccentricity_l0_977


namespace sum_of_squares_of_distances_l0_356

theorem sum_of_squares_of_distances (R a : ℝ) :
  ∀ (M : ℝ) (hM : abs M ≤ R),
  let d1 := real.sqrt ((a ^ 2) + (R ^ 2) - 2 * a * R * (cos (0:ℝ))) in
  let d2 := real.sqrt ((a ^ 2) + (R ^ 2) + 2 * a * R * (cos (0:ℝ))) in
  d1 ^ 2 + d2 ^ 2 = 2 * (a ^ 2 + R ^ 2) := 
by sorry

end sum_of_squares_of_distances_l0_356


namespace cos_135_eq_neg_inv_sqrt_2_l0_549

theorem cos_135_eq_neg_inv_sqrt_2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 :=
sorry

end cos_135_eq_neg_inv_sqrt_2_l0_549


namespace ball_distribution_l0_815

theorem ball_distribution: 
  let n := 18 in
  let k := 5 in
  -- placing at least 3 balls in each of 5 boxes
  ∃ ways, ways = Nat.choose (n - 3 * k + k - 1) (k - 1) ∧ ways = 35 :=
begin
  let n := 18,
  let k := 5,
  -- calculating the remaining balls
  let remaining_balls := n - 3 * k,
  -- using the stars and bars method
  let ways := Nat.choose (remaining_balls + k - 1) (k - 1),
  use ways,
  split,
  {
    refl,
  },
  {
    -- calculation of the binomial coefficient
    have h : remaining_balls = 3 := rfl, -- 18 - 15 = 3
    rw h,
    have h' : Nat.choose (7) (4) = 35 := by rnorm,
    rw h',
  },
end

end ball_distribution_l0_815


namespace oranges_in_box_l0_190

theorem oranges_in_box :
  ∃ (A P O : ℕ), A + P + O = 60 ∧ A = 3 * (P + O) ∧ P = (A + O) / 5 ∧ O = 5 :=
by
  sorry

end oranges_in_box_l0_190


namespace cos_135_eq_neg_sqrt2_div_2_l0_615

theorem cos_135_eq_neg_sqrt2_div_2 : 
  Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_615


namespace cos_135_degree_l0_496

theorem cos_135_degree : 
  let Q := (Complex.exp (135 * Real.pi / 180 * Complex.I)) in 
  Complex.re Q = -Real.sqrt 2 / 2 :=
by sorry

end cos_135_degree_l0_496


namespace cos_135_eq_neg_sqrt2_div_2_l0_624

theorem cos_135_eq_neg_sqrt2_div_2 : 
  Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_624


namespace cos_135_eq_neg_sqrt2_div_2_l0_475

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by 
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_475


namespace g_symmetric_y_axis_l0_70

-- Definition of the function g(x)
def g (x : ℝ) : ℝ :=
  abs ⌈x⌉ - abs ⌈x + 1⌉

-- The theorem stating the symmetry about the y-axis
theorem g_symmetric_y_axis : ∀ x, g(-x) = g(x) :=
by
  sorry

end g_symmetric_y_axis_l0_70


namespace remainder_of_3_to_40_plus_5_mod_5_l0_960

theorem remainder_of_3_to_40_plus_5_mod_5 : (3^40 + 5) % 5 = 1 :=
by
  sorry

end remainder_of_3_to_40_plus_5_mod_5_l0_960


namespace solution_set_of_inequality_l0_890

theorem solution_set_of_inequality :
  {x : ℝ | 2 ≥ 1 / (x - 1)} = {x : ℝ | x < 1} ∪ {x : ℝ | x ≥ 3 / 2} :=
by
  sorry

end solution_set_of_inequality_l0_890


namespace vector_operation_l0_637

open Matrix

def u : Matrix (Fin 2) (Fin 1) ℝ := ![![3], ![-6]]
def v : Matrix (Fin 2) (Fin 1) ℝ := ![![1], ![-9]]
def w : Matrix (Fin 2) (Fin 1) ℝ := ![![-1], ![4]]

--\mathbf{u} - 5\mathbf{v} + \mathbf{w} = \begin{pmatrix} = \begin{pmatrix} -3 \\ 43 \end{pmatrix}
theorem vector_operation : u - (5 : ℝ) • v + w = ![![-3], ![43]] :=
by
  sorry

end vector_operation_l0_637


namespace prank_combinations_l0_414

theorem prank_combinations :
  let monday_choices := 1
  let tuesday_choices := 2
  let wednesday_choices := 5
  let thursday_choices := 4
  let friday_choices := 1
  (monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices) = 40 :=
by
  sorry

end prank_combinations_l0_414


namespace seating_arrangement_l0_367

theorem seating_arrangement (A4_2 : ℕ) (A4_1 : ℕ) (fact5 : ℕ) :
  A4_2 = 12 → A4_1 = 4 → fact5 = 120 → 
  A4_2 * A4_1 * fact5 = 5760 :=
by
  intros hA4_2 hA4_1 hfact5
  rw [hA4_2, hA4_1, hfact5]
  norm_num
  done

end seating_arrangement_l0_367


namespace circle_area_60_degree_angle_l0_20

open Real

theorem circle_area_60_degree_angle {A B C D K : Point} (O : Point) (R : ℝ)
  (h_angle : ∠(A - K) = 60)
  (h_tangent : circle O R is_tangent_to A)
  (h_intersect_AB : (circle O R).intersection_path (line_through K) = {A, B})
  (h_intersect_CD : (circle O R).intersection_path (line_bisector ∠(A - K)) = {C, D})
  (h_AB: dist A B = sqrt 6)
  (h_CD: dist C D = sqrt 6) :
  π * R^2 = π * sqrt 3 := 
sorry

end circle_area_60_degree_angle_l0_20


namespace positive_difference_l0_911

theorem positive_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
sorry

end positive_difference_l0_911


namespace max_value_vector_sum_l0_160

theorem max_value_vector_sum (α β : ℝ) :
  let a := (Real.cos α, Real.sin α)
  let b := (Real.sin β, -Real.cos β)
  |(a.1 + b.1, a.2 + b.2)| ≤ 2 := by
  sorry

end max_value_vector_sum_l0_160


namespace units_digit_of_five_consecutive_product_is_zero_l0_942

theorem units_digit_of_five_consecutive_product_is_zero (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 10 = 0 :=
by
  sorry

end units_digit_of_five_consecutive_product_is_zero_l0_942


namespace area_enclosed_by_S_l0_235

noncomputable def four_presentable {z w : ℂ} : Prop :=
  |w| = 4 ∧ z = w - 1 / w

def S : set ℂ := {z | ∃ w : ℂ, four_presentable z w}

theorem area_enclosed_by_S : 
  ∃ (area : ℝ), area = (255 / 16) * Real.pi := 
begin
  use (255 / 16) * Real.pi,
  sorry
end

end area_enclosed_by_S_l0_235


namespace minimal_abs_difference_l0_170

theorem minimal_abs_difference : ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ xy - 2x + 5y = 111 ∧ |x - y| = 93 := 
sorry

end minimal_abs_difference_l0_170


namespace fran_avg_speed_l0_773

theorem fran_avg_speed (Joann_speed : ℕ) (Joann_time : ℚ) (Fran_time : ℕ) (distance : ℕ) (s : ℚ) : 
  Joann_speed = 16 → 
  Joann_time = 3.5 → 
  Fran_time = 4 → 
  distance = Joann_speed * Joann_time → 
  distance = Fran_time * s → 
  s = 14 :=
by
  intros hJs hJt hFt hD hF
  sorry

end fran_avg_speed_l0_773


namespace units_digit_2_pow_2015_minus_1_l0_54

theorem units_digit_2_pow_2015_minus_1 : (2^2015 - 1) % 10 = 7 := by
  sorry

end units_digit_2_pow_2015_minus_1_l0_54


namespace at_least_two_equal_radii_l0_765

-- Given variables representing the radii
variables {x y z t : ℝ}
-- Establish our conditions: the circles touch each other externally and touch two adjacent sides of the quadrilateral
axiom radii_property : ∃ x y z t > 0, (∃ quadrilateral_can_be_inscribed: (x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0))

-- The main theorem to be proven
theorem at_least_two_equal_radii
  (h : ∃ x y z t > 0, QuadrilateralInscribed := radii_property) :
  ∃ (r1 r2 : ℝ), r1 = r2 := 
sorry

end at_least_two_equal_radii_l0_765


namespace shaded_area_correct_l0_994

noncomputable def area_shaded_region : ℝ :=
  let radius_small := 2
  let radius_large := 6
  let area_large := Real.pi * radius_large^2
  let area_small := Real.pi * radius_small^2
  area_large - area_small

theorem shaded_area_correct :
  let radius_small := 2
  let radius_large := 6
  ∃ A B C : ℝ → Prop, (
    A center_radius_small ∧
    B center_radius_large ∧
    Tangent (circle A radius_small) (circle B radius_large) at C ∧
    B on circle A radius_small
  ) → area_shaded_region = 32 * Real.pi :=
by
  sorry

end shaded_area_correct_l0_994


namespace find_m_n_sum_l0_220

-- Definitions representing the conditions given
variables {x y z : ℂ}
variable (h1 : (x / (y + z)) + (y / (z + x)) + (z / (x + y)) = 9)
variable (h2 : (x^2 / (y + z)) + (y^2 / (z + x)) + (z^2 / (x + y)) = 64)
variable (h3 : (x^3 / (y + z)) + (y^3 / (z + x)) + (z^3 / (x + y)) = 488)

-- Expressing the main condition to prove
noncomputable def prove_m_n_sum : ℂ := 
  let frac_expr := (x / (y*z)) + (y / (z*x)) + (z / (x*y))
  let m_n := (3 : ℕ, 13 : ℕ) -- the pair (m, n) where GCD(3, 13) = 1
  in if frac_expr = m_n.1 / m_n.2 then m_n.1 + m_n.2 else 0

theorem find_m_n_sum : (h1 ∧ h2 ∧ h3) → prove_m_n_sum = 16 :=
by {
  sorry
}

end find_m_n_sum_l0_220


namespace positive_difference_of_numbers_l0_923

theorem positive_difference_of_numbers
  (a b : ℝ)
  (h1 : a + b = 10)
  (h2 : a^2 - b^2 = 40) :
  |a - b| = 4 :=
by 
  sorry

end positive_difference_of_numbers_l0_923


namespace cosine_135_eq_neg_sqrt_2_div_2_l0_542

theorem cosine_135_eq_neg_sqrt_2_div_2 (cos180 : Real := -1) (cos90 : Real := 0) (cos45 : Real := Real.sqrt 2 / 2)
    (theta1 : Real := 180) (theta2 : Real := 45) (theta3 : Real := 135) :
    theta3 = theta1 - theta2 ∧ 
    Real.cos theta1 = cos180 ∧ 
    Real.cos theta2 = cos45 → 
    Real.cos theta3 = -cos45 := by
  sorry

end cosine_135_eq_neg_sqrt_2_div_2_l0_542


namespace positive_difference_l0_893

theorem positive_difference (a b : ℝ) (h₁ : a + b = 10) (h₂ : a^2 - b^2 = 40) : |a - b| = 4 :=
by
  sorry

end positive_difference_l0_893


namespace root_in_interval_l0_871

-- Define the function f
def f (x : ℝ) : ℝ := log x + 2 * x - 6

-- State the theorem
theorem root_in_interval : ∃ x ∈ Ioo 2 3, f x = 0 :=
by {
  -- Define the specific conditions given in the problem
  have f2 : f 2 < 0 := by simp [f, log_pos, lt_sub_iff_add_lt]; norm_num1,
  have f3 : f 3 > 0 := by simp [f, log_pos, sub_lt_iff_lt_add]; norm_num1,

  -- Claim that there is a root in the interval (2, 3)
  apply exists_root_of_ivt,
  exact f2,
  exact f3,
} sorry

end root_in_interval_l0_871


namespace Caitlin_correct_age_l0_50

def Aunt_Anna_age := 48
def Brianna_age := Aunt_Anna_age / 2
def Caitlin_age := Brianna_age - 7

theorem Caitlin_correct_age : Caitlin_age = 17 := by
  /- Condon: Aunt Anna is 48 years old. -/
  let ha := Aunt_Anna_age
  /- Condon: Brianna is half as old as Aunt Anna. -/
  let hb := Brianna_age
  /- Condon: Caitlin is 7 years younger than Brianna. -/
  let hc := Caitlin_age
  /- Question: How old is Caitlin? Proof: -/
  sorry

end Caitlin_correct_age_l0_50


namespace cos_135_eq_neg_sqrt_two_div_two_l0_447

theorem cos_135_eq_neg_sqrt_two_div_two : real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt_two_div_two_l0_447


namespace evaluate_3f3_minus_2f9_l0_69

def f (x : ℝ) : ℝ := x^2 - 3 * real.sqrt x

theorem evaluate_3f3_minus_2f9 : 3 * f 3 - 2 * f 9 = -117 - 9 * real.sqrt 3 :=
by
  sorry

end evaluate_3f3_minus_2f9_l0_69


namespace cos_135_eq_neg_sqrt2_div_2_l0_477

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by 
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_477


namespace positive_difference_of_two_numbers_l0_904

theorem positive_difference_of_two_numbers 
  (a b : ℝ) 
  (h₁ : a + b = 10) 
  (h₂ : a^2 - b^2 = 40) : 
  |a - b| = 4 := 
sorry

end positive_difference_of_two_numbers_l0_904


namespace find_f_of_2_l0_129

theorem find_f_of_2 (f : ℝ → ℝ) (h : ∀ (x : ℝ), x > 0 → f (Real.log x / Real.log 2) = 2 ^ x) : f 2 = 16 :=
by
  sorry

end find_f_of_2_l0_129


namespace positive_difference_of_two_numbers_l0_932

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l0_932


namespace min_value_of_f_l0_685

-- Define the function f
def f (a b c x y z : ℤ) : ℤ := a * x + b * y + c * z

-- Define the gcd function for three integers
def gcd3 (a b c : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) c

-- Define the main theorem to prove
theorem min_value_of_f (a b c : ℕ) (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0) :
  ∃ (x y z : ℤ), f a b c x y z = gcd3 a b c := 
by
  sorry

end min_value_of_f_l0_685


namespace cos_135_eq_neg_inv_sqrt_2_l0_555

theorem cos_135_eq_neg_inv_sqrt_2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 :=
sorry

end cos_135_eq_neg_inv_sqrt_2_l0_555


namespace cos_135_eq_neg_sqrt2_div_2_l0_487

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by 
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_487


namespace cosine_135_eq_neg_sqrt_2_div_2_l0_532

theorem cosine_135_eq_neg_sqrt_2_div_2 (cos180 : Real := -1) (cos90 : Real := 0) (cos45 : Real := Real.sqrt 2 / 2)
    (theta1 : Real := 180) (theta2 : Real := 45) (theta3 : Real := 135) :
    theta3 = theta1 - theta2 ∧ 
    Real.cos theta1 = cos180 ∧ 
    Real.cos theta2 = cos45 → 
    Real.cos theta3 = -cos45 := by
  sorry

end cosine_135_eq_neg_sqrt_2_div_2_l0_532


namespace leopards_arrangement_l0_802

theorem leopards_arrangement (leopards : Fin 8 → ℕ) (h : ∀ i j, i < j → leopards i < leopards j) :
  (∃ shortest tallest, shortest ≠ tallest ∧ 
    (shortest = 0 ∨ shortest = 7) ∧ (tallest = 0 ∨ tallest = 7) ∧ shortest ≠ tallest) →
  ∃ (!remaining_leopards), 
    (2 * (List.permutations (!remaining_leopards)).length = 1440) :=
by
  sorry

end leopards_arrangement_l0_802


namespace marked_price_is_35_percent_above_cost_l0_25

/-- If an item has a cost price CP, and the shopkeeper wants to gain 8%, 
while offering a 20% discount on the marked price, the marked price 
is 35% above the cost price. -/
theorem marked_price_is_35_percent_above_cost (CP : ℝ) (hCP_pos : 0 < CP) (SP : ℝ) (hSP : SP = CP * 1.08) (MP : ℝ) (hMP : SP = 0.8 * MP) :
  (MP - CP) / CP * 100 = 35 :=
begin
  have h : CP ≠ 0, from ne_of_gt hCP_pos,
  calc (MP - CP) / CP * 100
      = (MP / CP - 1) * 100 : by { rw sub_div, rw div_self h, }
  ... = ((SP / 0.8) / CP - 1) * 100 : by { rw ← hMP, }
  ... = ((CP * 1.08 / 0.8) / CP - 1) * 100 : by { rw hSP, }
  ... = ((1.08 / 0.8) - 1) * 100 : by { rw div_mul_cancel _ h, }
  ... = (1.35 - 1) * 100 : by norm_num
  ... = 35 : by norm_num,
end

end marked_price_is_35_percent_above_cost_l0_25


namespace positive_difference_of_numbers_l0_925

theorem positive_difference_of_numbers
  (a b : ℝ)
  (h1 : a + b = 10)
  (h2 : a^2 - b^2 = 40) :
  |a - b| = 4 :=
by 
  sorry

end positive_difference_of_numbers_l0_925


namespace function_form_l0_789

theorem function_form (C : ℕ) (C_pos : C > 0) (f : ℕ → ℕ) 
  (hf : ∀ a b : ℕ, a > 0 → b > 0 → a + b > C → a + f b ∣ a^2 + b * f a) : 
  ∃ k : ℕ, k > 0 ∧ ∀ a : ℕ, a > 0 → f a = k * a :=
begin
  sorry
end

end function_form_l0_789


namespace exinscribed_sphere_inequality_l0_761

variable (r r_A r_B r_C r_D : ℝ)

theorem exinscribed_sphere_inequality 
  (hr : 0 < r) 
  (hrA : 0 < r_A) 
  (hrB : 0 < r_B) 
  (hrC : 0 < r_C) 
  (hrD : 0 < r_D) :
  1 / Real.sqrt (r_A^2 - r_A * r_B + r_B^2) +
  1 / Real.sqrt (r_B^2 - r_B * r_C + r_C^2) +
  1 / Real.sqrt (r_C^2 - r_C * r_D + r_D^2) +
  1 / Real.sqrt (r_D^2 - r_D * r_A + r_A^2) ≤
  2 / r := by
  sorry

end exinscribed_sphere_inequality_l0_761


namespace half_of_number_l0_364

theorem half_of_number (N : ℝ)
  (h1 : (4 / 15) * (5 / 7) * N = (4 / 9) * (2 / 5) * N + 8) : 
  (N / 2) = 315 := 
sorry

end half_of_number_l0_364


namespace apple_cost_l0_46

theorem apple_cost (x l q : ℝ) 
  (h1 : 10 * l = 3.62) 
  (h2 : x * l + (33 - x) * q = 11.67)
  (h3 : x * l + (36 - x) * q = 12.48) : 
  x = 30 :=
by
  sorry

end apple_cost_l0_46


namespace log_10_50_between_consecutive_integers_l0_93

theorem log_10_50_between_consecutive_integers : 
  ∃ (a b : ℤ), a + b = 3 ∧ a < log 50 / log 10 ∧ log 50 / log 10 < b ∧ b = a + 1 :=
by
  use 1, 2
  split; norm_num
  split; exact one_lt_log_one_div.ten51
  split; exact log_one_div.ten52
  rfl

end log_10_50_between_consecutive_integers_l0_93


namespace problem_proof_equality_cases_l0_788

theorem problem_proof (x y : ℝ) (h : (x + 1) * (y + 2) = 8) : (x * y - 10) ^ 2 ≥ 64 := sorry

theorem equality_cases (x y : ℝ) (h : (x + 1) * (y + 2) = 8) : 
  (x * y - 10) ^ 2 = 64 ↔ ((x,y) = (1, 2) ∨ (x,y) = (-3, -6)) := sorry

end problem_proof_equality_cases_l0_788


namespace trapezoid_is_isosceles_l0_824

variable (A B C D : Type) [AffineSpace ℝ A] (a b c d : A)

-- Define that ABCD is a trapezoid with AD and BC as bases
def trapezoid (A B C D: A) [AffineSpace ℝ A]: Prop :=
  ∃ (AD BC: AffineSubspace ℝ A), 
    AD ∈ lineThrough A D ∧ BC ∈ lineThrough B C ∧ AD ∥ BC


-- Define that the diagonals in the trapezoid are of equal length
def equal_diagonals (A B C D: A) [AffineSpace ℝ A]: Prop :=
  dist A C = dist B D

-- Define the theorem statement
theorem trapezoid_is_isosceles (A B C D : A) [AffineSpace ℝ A] :
  trapezoid A B C D → equal_diagonals A B C D → dist A B = dist C D :=
by
  sorry

end trapezoid_is_isosceles_l0_824


namespace positive_difference_of_two_numbers_l0_941

theorem positive_difference_of_two_numbers (x y : ℝ)
  (h1 : x + y = 10) (h2 : x ^ 2 - y ^ 2 = 40) : abs (x - y) = 4 := 
by
  sorry

end positive_difference_of_two_numbers_l0_941


namespace shifted_sine_is_monotonic_in_interval_l0_179

theorem shifted_sine_is_monotonic_in_interval :
  (∀ x y, -π/12 ≤ x → x ≤ 5*π/12 → -π/12 ≤ y → y ≤ 5*π/12 → x < y → sin (2*x - π/3) < sin (2*y - π/3)) :=
by
  sorry

end shifted_sine_is_monotonic_in_interval_l0_179


namespace distance_min_value_l0_687

theorem distance_min_value (a b c d : ℝ) 
  (h₁ : |b - (Real.log a) / a| + |c - d + 2| = 0) : 
  (a - c)^2 + (b - d)^2 = 9 / 2 :=
by {
  sorry
}

end distance_min_value_l0_687


namespace complex_quadrant_l0_101

theorem complex_quadrant (Z : ℂ) (hZ : Z = 1 + complex.i) : 
  let point := (1 / Z + Z : ℂ) in
  0 < point.re ∧ 0 < point.im :=
by
  sorry

end complex_quadrant_l0_101


namespace angle_of_inclination_of_line_2x_minus_2y_plus_1_eq_0_l0_338

theorem angle_of_inclination_of_line_2x_minus_2y_plus_1_eq_0 :
  ∃ α : ℝ, α = 45 ∧ ∀ x y : ℝ, 2 * x - 2 * y + 1 = 0 -> tan α = 1 :=
by
  sorry

end angle_of_inclination_of_line_2x_minus_2y_plus_1_eq_0_l0_338


namespace number_of_equidistant_points_l0_196

-- Let T be a regular tetrahedron with vertices A, B, C, and D
-- Define a regular tetrahedron and points on its surface
noncomputable def regular_tetrahedron := sorry -- This would define the regular tetrahedron structurally
noncomputable def point_on_surface (T : Tetrahedron) (p : Point) : Prop := sorry

-- Define the equidistance condition
def equidistant_from_edge_and_points (T : Tetrahedron) (p : Point) (e : Edge) (p1 p2 : Point) : Prop :=
  distance p (midpoint e) = distance p p1 ∧ distance p (midpoint e) = distance p p2

-- The proof problem
theorem number_of_equidistant_points (T : Tetrahedron) (A B C D : Point) (H : regular_tetrahedron T A B C D):
  ∃ p1 p2, point_on_surface T p1 ∧ point_on_surface T p2 ∧ 
  equidistant_from_edge_and_points T p1 (edge A B) C D ∧ 
  equidistant_from_edge_and_points T p2 (edge A B) C D ∧ 
  p1 ≠ p2 :=
by
  sorry

end number_of_equidistant_points_l0_196


namespace a6_value_l0_678

noncomputable def a_seq : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+1) := 3 * (Nat.sum (List.range n).map a_seq)

theorem a6_value : a_seq 6 = 768 := by
  sorry

end a6_value_l0_678


namespace particle_distance_l0_389

def parabola (x : ℝ) : ℝ :=
  x^2 + 2 * x - 3

def point_P_x (y : ℝ) : set ℝ := {x | parabola x = y}

def horizontal_distance (x1 x2 : ℝ) : ℝ :=
  abs (x1 - x2)

theorem particle_distance :
  ∃ (x1 x2 : ℝ), parabola x1 = -3 ∧ parabola x2 = 7 ∧ horizontal_distance x1 x2 = 1 + sqrt 11 :=
sorry

end particle_distance_l0_389


namespace proof_cos_135_degree_l0_572

noncomputable def cos_135_degree_is_negative_sqrt2_over_2 : Prop :=
  real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2

theorem proof_cos_135_degree : cos_135_degree_is_negative_sqrt2_over_2 :=
by
  sorry

end proof_cos_135_degree_l0_572


namespace cosine_135_eq_neg_sqrt_2_div_2_l0_544

theorem cosine_135_eq_neg_sqrt_2_div_2 (cos180 : Real := -1) (cos90 : Real := 0) (cos45 : Real := Real.sqrt 2 / 2)
    (theta1 : Real := 180) (theta2 : Real := 45) (theta3 : Real := 135) :
    theta3 = theta1 - theta2 ∧ 
    Real.cos theta1 = cos180 ∧ 
    Real.cos theta2 = cos45 → 
    Real.cos theta3 = -cos45 := by
  sorry

end cosine_135_eq_neg_sqrt_2_div_2_l0_544


namespace ice_cream_sales_not_less_than_6500_in_5_months_l0_373

def ice_cream_sales (x : ℕ) (A ω ϕ B : ℝ) : ℝ :=
  A * Real.cos (ω * x + ϕ) + B

theorem ice_cream_sales_not_less_than_6500_in_5_months :
  ∃ A ω ϕ B : ℝ,
    (A > 0) ∧ (ω > 0) ∧ (|ϕ| < Real.pi) ∧
    (∀ x : ℕ, (1 ≤ x ∧ x ≤ 12) → ice_cream_sales x A ω ϕ B ≥ 500 ∧ ice_cream_sales x A ω ϕ B ≤ 8500) ∧
    (ice_cream_sales 8 A ω ϕ B = 8500) ∧ 
    (ice_cream_sales 2 A ω ϕ B = 500) ∧
    (finset.filter (λ x, ice_cream_sales x A ω ϕ B ≥ 6500) (finset.range 13) = finset.range 5) :=
by
  sorry

end ice_cream_sales_not_less_than_6500_in_5_months_l0_373


namespace sum_of_coefficients_is_7_l0_793

noncomputable def v (n : ℕ) : ℕ := sorry

theorem sum_of_coefficients_is_7 : 
  (∀ n : ℕ, v (n + 1) - v n = 3 * n + 2) → (v 1 = 7) → (∃ a b c : ℝ, (a * n^2 + b * n + c = v n) ∧ (a + b + c = 7)) := 
by
  intros H1 H2
  sorry

end sum_of_coefficients_is_7_l0_793


namespace number_of_sets_C_l0_124

def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}

def B : Set ℕ := {x | -1 < (x : ℤ) ∧ (x : ℤ) < 4}

def num_sets_C (A B : Set ℕ) := 
  {C : Set ℕ | A ⊆ C ∧ C ⊆ B}.to_finset.card

theorem number_of_sets_C : num_sets_C {1, 2} {0, 1, 2, 3} = 4 := by
  sorry

end number_of_sets_C_l0_124


namespace find_a_l0_861

theorem find_a (a : ℚ) : 
  (∑ r in finset.range 7, (if (12 - 3 * r = 3) then (-1)^r * a^(6 - r) * (nat.choose 6 r) else 0)) = 160 → 
  a = -2 :=
by
  sorry

end find_a_l0_861


namespace train_length_eq_l0_400

-- Definitions
def train_speed_kmh : Float := 45
def crossing_time_s : Float := 30
def total_length_m : Float := 245

-- Theorem statement
theorem train_length_eq :
  ∃ (train_length bridge_length: Float),
  bridge_length = total_length_m - train_length ∧
  train_speed_kmh * 1000 / 3600 * crossing_time_s = train_length + bridge_length ∧
  train_length = 130 :=
by
  sorry

end train_length_eq_l0_400


namespace proof_problem_l0_234

noncomputable def a : ℝ := -sqrt (9 / 27)
noncomputable def b : ℝ := sqrt ((3 + sqrt 8)^2 / 12)

theorem proof_problem :
  a^2 = 9 / 27 ∧ b^2 = (3 + sqrt 8)^2 / 12 ∧ a < 0 ∧ b > 0 →
  (a + b)^4 = 9 * sqrt 2 / 16 ∧ 9 + 2 + 16 = 27 :=
by
  sorry

end proof_problem_l0_234


namespace a0_and_Sn_l0_107

theorem a0_and_Sn (n : ℕ) (hn : 0 < n) :
  (let a0 := (x + 1) ^ n := 2^n in
   let Sn := n * 3^(n - 1) in
   (a0 = 2^n) ∧
   (Sn = n * 3^(n - 1)) ∧
   (n = 1 ∨ n = 2 → Sn ≤ n^3) ∧
   (n = 3 → Sn = n^3) ∧
   (n ≥ 4 → Sn > n^3)
  ) :=
sorry

end a0_and_Sn_l0_107


namespace circles_sum_reciprocal_sqrt_eq_l0_375

noncomputable def initial_radii : ℕ × ℕ := (60^2, 80^2)

-- Defining the radius calculation for recursively added circles
noncomputable def radius (r1 r2 : ℕ) : ℕ :=
  r1 * r2 / ((Nat.sqrt r1 + Nat.sqrt r2) * (Nat.sqrt r1 + Nat.sqrt r2))

-- Defining a function to calculate the sum of reciprocals of square roots of radii
noncomputable def reciprocal_sqrt_sum (S : List ℕ) : ℚ :=
  S.sum (fun r => (1 : ℚ) / Nat.sqrt r)

-- Given conditions
def mk_circle_layers : ℕ → List ℕ
| 0 => [60^2, 80^2]
| (k + 1) => mk_circle_layers k ++ List.map2 radius (dropLast $ mk_circle_layers k) (tail $ mk_circle_layers k)

noncomputable def S : List ℕ := List.join (List.map mk_circle_layers (List.range 7))

-- Final statement
theorem circles_sum_reciprocal_sqrt_eq :
  reciprocal_sqrt_sum S = (2555 : ℚ) / 336 :=
  sorry

end circles_sum_reciprocal_sqrt_eq_l0_375


namespace birthday_gift_package_l0_84

theorem birthday_gift_package 
  (spa_voucher_usd : ℝ := 250)
  (birthday_cake_usd : ℝ := 25)
  (flowers_usd : ℝ := 35)
  (skincare_set_usd : ℝ := 45)
  (books_usd : ℝ := 90)
  (discount_rate : ℝ := 0.10)
  (erika_savings_usd : ℝ := 155)
  (sam_savings_usd : ℝ := 175)
  (exchange_rate : ℝ := 1.2) :
  let spa_voucher_discounted := spa_voucher_usd * (1 - discount_rate),
      books_discounted := books_usd * (1 - discount_rate),
      total_cost_usd := spa_voucher_discounted + birthday_cake_usd + flowers_usd + skincare_set_usd + books_discounted,
      rick_savings_usd := total_cost_usd / 2,
      amy_savings_usd := 2 * (birthday_cake_usd + flowers_usd + skincare_set_usd),
      erika_contribution_usd := erika_savings_usd * 1.2,
      rick_contribution_usd := rick_savings_usd * 1.15,
      sam_contribution_usd := sam_savings_usd * 1.10,
      amy_contribution_usd := amy_savings_usd * 1.05,
      total_contribution_usd := erika_contribution_usd + rick_contribution_usd + sam_contribution_usd + amy_contribution_usd,
      total_cost_eur := total_cost_usd / exchange_rate,
      total_contribution_eur := total_contribution_usd / exchange_rate,
      remaining_amount_eur := total_contribution_eur - total_cost_eur
  in remaining_amount_eur ≈ 353.60 :=
by 
  sorry

end birthday_gift_package_l0_84


namespace cosine_135_eq_neg_sqrt_2_div_2_l0_541

theorem cosine_135_eq_neg_sqrt_2_div_2 (cos180 : Real := -1) (cos90 : Real := 0) (cos45 : Real := Real.sqrt 2 / 2)
    (theta1 : Real := 180) (theta2 : Real := 45) (theta3 : Real := 135) :
    theta3 = theta1 - theta2 ∧ 
    Real.cos theta1 = cos180 ∧ 
    Real.cos theta2 = cos45 → 
    Real.cos theta3 = -cos45 := by
  sorry

end cosine_135_eq_neg_sqrt_2_div_2_l0_541


namespace max_f_value_area_of_triangle_l0_110

open Real

-- Definition of vectors a and b
def a (x : ℝ) : ℝ × ℝ := (2 * cos x, sin x - cos x)
def b (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin x, sin x + cos x)

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := (a x).fst * (b x).fst + (a x).snd * (b x).snd

-- Definition of the conditions
def triangle_cond (a b c A B C : ℝ) := a + b = 2 * sqrt 3 ∧ c = sqrt 6 ∧ f C = 2

-- Tuple proving the maximum value condition for f(x)
theorem max_f_value (x : ℝ) : (∃ k : ℤ, x = k * π + π / 3) ↔ f x = 2 :=
sorry

-- Area calculation for triangle ABC given the conditions
theorem area_of_triangle (a b c A B C : ℝ) (h : triangle_cond a b c A B C) : 
  let s := (a + b + c) / 2 in
  sqrt (s * (s - a) * (s - b) * (s - c)) = sqrt 3 / 2 :=
sorry

end max_f_value_area_of_triangle_l0_110


namespace proof_cos_135_degree_l0_561

noncomputable def cos_135_degree_is_negative_sqrt2_over_2 : Prop :=
  real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2

theorem proof_cos_135_degree : cos_135_degree_is_negative_sqrt2_over_2 :=
by
  sorry

end proof_cos_135_degree_l0_561


namespace length_diagonal_AC_l0_673

theorem length_diagonal_AC 
  (A B C D : Point)
  (h₁ : distance A B = 1)
  (h₂ : distance A D = 1)
  (h₃ : angle A B D = 100°)
  (h₄ : angle A D C = 130°)
  : distance A C = real.sqrt(2 - 2 * real.cos (100 * real.pi / 180)) :=
sorry

end length_diagonal_AC_l0_673


namespace cos_135_eq_correct_l0_611

noncomputable def cos_135_eq: Prop :=
  real.cos ((135 * real.pi) / 180) = -real.sqrt 2 / 2

theorem cos_135_eq_correct : cos_135_eq :=
  sorry

end cos_135_eq_correct_l0_611


namespace mean_temperature_l0_854

theorem mean_temperature :
  let temperatures := [-8, -5, -5, -2, 0, 4, 5, 3, 6, 1] in
  (temperatures.sum : ℚ) / temperatures.length = -0.6 :=
by
  sorry

end mean_temperature_l0_854


namespace delayed_rulings_count_l0_383

theorem delayed_rulings_count (total_cases dismissed_cases ruled_guilty innocent_fraction : ℕ) 
    (h_total_cases : total_cases = 17)
    (h_dismissed_cases : dismissed_cases = 2)
    (h_ruled_guilty : ruled_guilty = 4)
    (h_innocent_fraction : innocent_fraction = 2 / 3) :
    let remaining_cases := total_cases - dismissed_cases in
    let ruled_innocent := innocent_fraction * remaining_cases in
    let accounted_cases := ruled_innocent + ruled_guilty in
    let delayed_cases := remaining_cases - accounted_cases in
    delayed_cases = 1 := 
by 
  sorry

end delayed_rulings_count_l0_383


namespace find_eggs_per_turtle_l0_80

variable (num_turtles num_hatchlings : ℕ) (hatch_rate : ℚ) (E : ℕ)

axiom condition_1 : num_turtles = 6
axiom condition_2 : hatch_rate = 0.40
axiom condition_3 : num_hatchlings = 48

theorem find_eggs_per_turtle : E = 20 :=
by
  have h1 : E * (num_turtles : ℚ) = num_hatchlings / hatch_rate,
    from sorry,
  have h2 : E = num_hatchlings / (hatch_rate * num_turtles),
    from sorry,
  rw [condition_1, condition_2, condition_3] at h1 h2,
  exact sorry

end find_eggs_per_turtle_l0_80


namespace count_perfect_cubes_between_1000_and_10000_l0_164

theorem count_perfect_cubes_between_1000_and_10000 :
  {n : ℕ | 1000 < n ∧ n < 10000 ∧ ∃ k : ℕ, n = k^3}.card = 11 := by
sorry

end count_perfect_cubes_between_1000_and_10000_l0_164


namespace cos_135_eq_neg_inv_sqrt_2_l0_578

theorem cos_135_eq_neg_inv_sqrt_2 :
  cos (135 * Real.pi / 180) = - (1 / Real.sqrt 2) :=
by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_578


namespace Rohan_earning_after_6_months_l0_833

def farm_area : ℕ := 20
def trees_per_sqm : ℕ := 2
def coconuts_per_tree : ℕ := 6
def harvest_interval : ℕ := 3
def sale_price : ℝ := 0.50
def total_months : ℕ := 6

theorem Rohan_earning_after_6_months :
  farm_area * trees_per_sqm * coconuts_per_tree * (total_months / harvest_interval) * sale_price 
    = 240 := by
  sorry

end Rohan_earning_after_6_months_l0_833


namespace line_through_origin_and_intersection_eq_x_y_l0_655

theorem line_through_origin_and_intersection_eq_x_y :
  ∀ (x y : ℝ), (x - 2 * y + 2 = 0) ∧ (2 * x - y - 2 = 0) →
  ∃ m b : ℝ, m = 1 ∧ b = 0 ∧ (y = m * x + b) :=
by
  sorry

end line_through_origin_and_intersection_eq_x_y_l0_655


namespace increasing_interval_iff_a_ge_neg_3_l0_867

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x - 2

theorem increasing_interval_iff_a_ge_neg_3 {a : ℝ} : 
  (∀ x ≥ 1, (derivative (f a) x) ≥ 0) ↔ a ≥ -3 :=
by
  sorry

end increasing_interval_iff_a_ge_neg_3_l0_867


namespace time_saved_calculator_l0_206

-- Define the conditions
def time_with_calculator (n : ℕ) : ℕ := 2 * n
def time_without_calculator (n : ℕ) : ℕ := 5 * n
def total_problems : ℕ := 20

-- State the theorem to prove the time saved is 60 minutes
theorem time_saved_calculator : 
  time_without_calculator total_problems - time_with_calculator total_problems = 60 :=
sorry

end time_saved_calculator_l0_206


namespace cos_135_eq_neg_sqrt2_div_2_l0_488

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by 
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_488


namespace perpendicular_and_equal_length_l0_105

variables {a b c d : ℂ}

def G1 := (a + d) / 2
def O1 := G1 + (d - a) / 2 * complex.I

def G2 := (a + b) / 2
def O2 := G2 + (b - a) / 2 * complex.I

def G3 := (b + c) / 2
def O3 := G3 + (c - b) / 2 * complex.I

def G4 := (c + d) / 2
def O4 := G4 + (d - c) / 2 * complex.I

noncomputable def O1O3 := O3 - O1
noncomputable def O2O4 := O4 - O2

theorem perpendicular_and_equal_length : 
  (O1O3 * ⟨0, 1⟩ = 0) ∧ (complex.abs O1O3 = complex.abs O2O4) := sorry

end perpendicular_and_equal_length_l0_105


namespace num_cookies_sixth_plate_l0_310

theorem num_cookies_sixth_plate :
  let cookies : ℕ → ℕ := λ n,
    if n = 1 then 5
    else if n = 2 then 7
    else if n = 3 then 10
    else if n = 4 then 14
    else if n = 5 then 19
    else cookies (n-1) + (n-1)
  in cookies 6 = 25 :=
by
  sorry

end num_cookies_sixth_plate_l0_310


namespace time_to_cross_tree_l0_985

def train_length : ℕ := 600
def platform_length : ℕ := 450
def time_to_pass_platform : ℕ := 105

-- Definition of the condition that leads to the speed of the train
def speed_of_train : ℚ := (train_length + platform_length) / time_to_pass_platform

-- Statement to prove the time to cross the tree
theorem time_to_cross_tree :
  (train_length : ℚ) / speed_of_train = 60 :=
by
  sorry

end time_to_cross_tree_l0_985


namespace minimum_value_amgm_l0_227

theorem minimum_value_amgm (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 27) : (a + 3 * b + 9 * c) ≥ 27 :=
by
  sorry

end minimum_value_amgm_l0_227


namespace ratio_XQ_QY_l0_862

-- Define the conditions of the problem
def hexagon_area : ℝ := 7
def bisected_area : ℝ := hexagon_area / 2
def rectangle_area : ℝ := 4
def triangle_base : ℝ := 4
def triangle_area : ℝ := bisected_area - rectangle_area
def triangle_height : ℝ := (2 * triangle_area) / triangle_base
def segment_length : ℝ := 4

-- Prove the ratio of XQ to QY given these conditions
theorem ratio_XQ_QY : 
  (hexagon_area = 7) →
  (bisected_area = 3.5) →
  (rectangle_area = 4) →
  (triangle_base = 4) →
  (triangle_area = 0.5) →
  (triangle_height = 0.25) →
  (segment_length = 4) →
  (∀ XQ QY, XQ + QY = segment_length → XQ / QY = 3) :=
begin
  sorry
end

end ratio_XQ_QY_l0_862


namespace proof_cos_135_degree_l0_560

noncomputable def cos_135_degree_is_negative_sqrt2_over_2 : Prop :=
  real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2

theorem proof_cos_135_degree : cos_135_degree_is_negative_sqrt2_over_2 :=
by
  sorry

end proof_cos_135_degree_l0_560


namespace original_number_is_144_l0_119

theorem original_number_is_144 :
  ∃ (A B C : ℕ), A ≠ 0 ∧
  (100 * A + 11 * B = 144) ∧
  (A * B^2 = 10 * A + C) ∧
  (A * C = C) ∧
  A = 1 ∧ B = 4 ∧ C = 6 :=
by
  sorry

end original_number_is_144_l0_119


namespace example1_condition1_example1_condition2_l0_645

-- Define the simple propositions for condition 1
def prop1_p : Prop := ∀ A B C D : Type, is_rhombus A ∧ has_diagonals B C D A → is_perpendicular B C
def prop1_q : Prop := ∀ A B C D : Type, is_rhombus A ∧ has_diagonals B C D A → bisect B C

-- Define the combined proposition for condition 1
def prop1 : Prop := prop1_p ∧ prop1_q

-- Define the simple propositions for condition 2
def prop2_p : Prop := 2 < 3
def prop2_q : Prop := 2 = 3

-- Define the combined proposition for condition 2
def prop2 : Prop := prop2_p ∨ prop2_q

-- Theorems to be proven
theorem example1_condition1 : prop1 = True := by
  sorry

theorem example1_condition2 : prop2 = True := by
  sorry

end example1_condition1_example1_condition2_l0_645


namespace isosceles_triangle_divides_area_l0_319

theorem isosceles_triangle_divides_area
  (a α β : ℝ) (h1 : a > 0) (h2 : α ∈ (0, π/2)) (h3 : β ∈ (0, π/2)) :
  let S_amb := (1 / 2) * a * (AM) * Real.sin (α - β)
  let S_amc := a * Real.cos α * (AM) * Real.sin β
  (S_amb / S_amc) = (Real.sin (α - β) / (2 * Real.cos α * Real.sin β)) :=
by
  sorry

end isosceles_triangle_divides_area_l0_319


namespace find_height_of_second_triangular_sail_l0_243

-- Define the dimensions of the rectangular sail
def rect_length : ℝ := 5 
def rect_width : ℝ := 8

-- Define the dimensions of the first triangular sail
def tri1_base : ℝ := 3
def tri1_height : ℝ := 4

-- Define the dimensions of the second triangular sail
def tri2_base : ℝ := 4

-- Define the total canvas needed
def total_canvas_needed : ℝ := 58

-- Calculate the areas
def area_rect : ℝ := rect_length * rect_width
def area_tri1 : ℝ := (tri1_base * tri1_height) / 2
def total_area_so_far : ℝ := area_rect + area_tri1

-- The area needed for second triangular sail
def area_tri2_needed : ℝ := total_canvas_needed - total_area_so_far

-- To find the height of the second triangular sail
def height_of_second_triangular_sail (h : ℝ) : Prop :=
  area_tri2_needed = (tri2_base * h) / 2

-- Statement to prove
theorem find_height_of_second_triangular_sail : ∃ h : ℝ, height_of_second_triangular_sail h ∧ h = 6 :=
by
  use 6
  -- proof goes here
  sorry

end find_height_of_second_triangular_sail_l0_243


namespace cos_135_eq_neg_inv_sqrt_2_l0_548

theorem cos_135_eq_neg_inv_sqrt_2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 :=
sorry

end cos_135_eq_neg_inv_sqrt_2_l0_548


namespace cos_135_eq_neg_sqrt2_div_2_l0_598

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  -- Conditions incorporated directly into the requirement due to the nature of trigonometric evaluations.
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_598


namespace cos_135_eq_neg_sqrt_two_div_two_l0_459

theorem cos_135_eq_neg_sqrt_two_div_two : real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt_two_div_two_l0_459


namespace bob_distance_when_they_meet_l0_353

-- Define the conditions
def distance_XY : ℝ := 10
def yolanda_rate : ℝ := 3
def bob_rate : ℝ := 4
def yolanda_start_time : ℝ := 0
def bob_start_time : ℝ := 1

-- The statement we want to prove
theorem bob_distance_when_they_meet : 
  ∃ t : ℝ, (yolanda_rate * (t + 1) + bob_rate * t = distance_XY) ∧ (bob_rate * t = 4) :=
sorry

end bob_distance_when_they_meet_l0_353


namespace cosine_135_eq_neg_sqrt_2_div_2_l0_535

theorem cosine_135_eq_neg_sqrt_2_div_2 (cos180 : Real := -1) (cos90 : Real := 0) (cos45 : Real := Real.sqrt 2 / 2)
    (theta1 : Real := 180) (theta2 : Real := 45) (theta3 : Real := 135) :
    theta3 = theta1 - theta2 ∧ 
    Real.cos theta1 = cos180 ∧ 
    Real.cos theta2 = cos45 → 
    Real.cos theta3 = -cos45 := by
  sorry

end cosine_135_eq_neg_sqrt_2_div_2_l0_535


namespace Winnie_lollipops_leftover_l0_967

-- Definitions corresponding to the conditions
def cherry_lollipops : ℕ := 32
def wintergreen_lollipops : ℕ := 150
def grape_lollipops : ℕ := 7
def shrimp_cocktail_lollipops : ℕ := 280
def friends : ℕ := 14

-- Statement of the problem to be proved
theorem Winnie_lollipops_leftover :
  let total_lollipops := cherry_lollipops + wintergreen_lollipops + grape_lollipops + shrimp_cocktail_lollipops in
  total_lollipops % friends = 7 :=
by
  -- proof steps to be filled
  sorry

end Winnie_lollipops_leftover_l0_967


namespace min_distance_proof_l0_688

noncomputable def min_distance_squared : ℝ :=
  let a : ℝ := sorry -- some positive real number
  let b : ℝ := ln a / a
  let d : ℝ := sorry -- another real number
  let c : ℝ := d - 2
  in (a - c)^2 + (b - d)^2

theorem min_distance_proof : min_distance_squared = 9 / 2 :=
by
  sorry

end min_distance_proof_l0_688


namespace gcd_9155_4892_l0_339

theorem gcd_9155_4892 : Nat.gcd 9155 4892 = 1 := 
by 
  sorry

end gcd_9155_4892_l0_339


namespace smallest_m_for_X_l0_230

def is_power_of_2 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def smallest_m (X : Set ℕ) (m : ℕ) : Prop :=
  ∀ W : Finset ℕ, W.card = m → W ⊆ X →
  ∃ u v ∈ W, is_power_of_2 (u + v)

theorem smallest_m_for_X :
  smallest_m ({ n | 1 ≤ n ∧ n ≤ 2001 }) 999 :=
by
  sorry

end smallest_m_for_X_l0_230


namespace first_player_always_wins_l0_246

theorem first_player_always_wins (grid_size : ℕ) (grid_odd : grid_size % 2 = 1)
  (first_pos : ℕ × ℕ) (initial_pos : first_pos = (grid_size // 2 + 1, grid_size // 2 + 1)) :
  ∃ (path : list (ℕ × ℕ)), path.head = first_pos ∧
  ((∀ (move : ℕ × ℕ), move ∈ path → move.1 < grid_size ∧ move.2 < grid_size) ∧ 
  ((1, 1) ∈ path ∨ (1, grid_size) ∈ path ∨ 
    (grid_size, 1) ∈ path ∨ (grid_size, grid_size) ∈ path)) :=
by
  sorry

end first_player_always_wins_l0_246


namespace number_of_arrangments_l0_5

theorem number_of_arrangments : 
  let hearts := [2, 3, 4, 5]
  let clubs := [2, 3, 4, 5]
  let cards := hearts ++ clubs
  let valid_sums := {combo | (combo : list ℕ) ∈ cards.combinations 4 ∧ combo.sum = 14}
  let arrangement_count := valid_sums.to_finset.sum (λ combo, (multiset.to_finset combo.to_multiset).card.factorial / 
    (multiset.to_finset (multiset.filter (=combo.head) combo.to_multiset)).card.factorial *
    (multiset.to_finset (multiset.filter (=combo.tail.head) combo.to_multiset)).card.factorial)
  arrangement_count = 396 :=
sorry

end number_of_arrangments_l0_5


namespace _l0_829

noncomputable def simpson_line_theorem {A B C P X Y Z : Type}
  [InCircle A B C] [OnCircumcircle P A B C] 
  (hperp_X : Perpendicular P X (Side B C))
  (hperp_Y : Perpendicular P Y (Side C A))
  (hperp_Z : Perpendicular P Z (Side A B))
  : Collinear X Y Z :=
sorry

end _l0_829


namespace series_convergence_l0_956

theorem series_convergence : 
  (∑' n : ℕ, (1 : ℝ) / (n + 1) / (n + 2)) = 1 := 
begin
  sorry
end

end series_convergence_l0_956


namespace cos_135_eq_neg_inv_sqrt_2_l0_550

theorem cos_135_eq_neg_inv_sqrt_2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 :=
sorry

end cos_135_eq_neg_inv_sqrt_2_l0_550


namespace problem_1_problem_2_problem_3_l0_145

open Real

def f (x a : ℝ) : ℝ := x / exp x - a * x * log x

theorem problem_1 {a : ℝ} (h1 : deriv (λ (x : ℝ), f x a) 1 = -1) : a = 1 :=
  sorry

theorem problem_2 : ∀ x : ℝ, f x 1 < 2 / exp 1 :=
  sorry

variables {m n : ℝ}

theorem problem_3 (h1 : m * n = 1) : 1 / exp m + 1 / exp n < 2 * (m + n) :=
  sorry

end problem_1_problem_2_problem_3_l0_145


namespace cos_135_eq_neg_inv_sqrt2_l0_442

theorem cos_135_eq_neg_inv_sqrt2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt2_l0_442


namespace proof_cos_135_degree_l0_559

noncomputable def cos_135_degree_is_negative_sqrt2_over_2 : Prop :=
  real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2

theorem proof_cos_135_degree : cos_135_degree_is_negative_sqrt2_over_2 :=
by
  sorry

end proof_cos_135_degree_l0_559


namespace solution_set_of_inequality_l0_304

theorem solution_set_of_inequality :
  { x : ℝ | x^2 + x - 2 < 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end solution_set_of_inequality_l0_304


namespace proof_cos_135_degree_l0_564

noncomputable def cos_135_degree_is_negative_sqrt2_over_2 : Prop :=
  real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2

theorem proof_cos_135_degree : cos_135_degree_is_negative_sqrt2_over_2 :=
by
  sorry

end proof_cos_135_degree_l0_564


namespace sequence_formula_correct_l0_656

-- Define the sequence
def sequence (n : ℕ) : ℝ :=
  match n % 4 with
  | 0 => 0
  | 1 => 1
  | 2 => 0
  | 3 => -1
  | _ => 0 -- This should never be hit

-- General term formula in the statement
def general_term (n : ℕ) : ℝ :=
  Real.cos ((n + 2) * Real.pi / 2)

-- The theorem stating the equivalence of the sequence and the general term
theorem sequence_formula_correct : ∀ n : ℕ, sequence n = general_term n :=
by
  intro n
  sorry

end sequence_formula_correct_l0_656


namespace cos_135_eq_neg_sqrt2_div_2_l0_462

theorem cos_135_eq_neg_sqrt2_div_2 : cos (135 * real.pi / 180) = - (real.sqrt 2 / 2) :=
by
  have h1 : 135 = 180 - 45 := by norm_num
  have h2 : cos (180 * real.pi / 180 - 45 * real.pi / 180) = - cos (45 * real.pi / 180) :=
    by rw [cos_sub, cos_pi, cos_pi_div_four, sin_pi, sin_pi_div_four]; norm_num
  have h3 : cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by norm_num
  rw [h1, ←h2, h3]
  norm_num
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_462


namespace solveEquation_l0_846

theorem solveEquation (x : ℝ) (hx : |x| ≥ 3) : (∃ x₁ x₂ : ℝ, (x₁ ≠ x₂ ∧ (x₁ / 3 + x₁ / Real.sqrt (x₁ ^ 2 - 9) = 35 / 12) ∧ (x₂ / 3 + x₂ / Real.sqrt (x₂ ^ 2 - 9) = 35 / 12)) ∧ x₁ + x₂ = 8.75) :=
sorry

end solveEquation_l0_846


namespace fraction_area_above_line_l0_282

theorem fraction_area_above_line (square : set (ℝ × ℝ))
                                 (A B : ℝ × ℝ)
                                 (H_square : square = {(4, 0), (7, 0), (7, 3), (4, 3)})
                                 (H_A : A = (4, 3))
                                 (H_B : B = (7, 1)) :
  ∃ (fraction : ℝ), fraction = 5 / 6 :=
by
  sorry

end fraction_area_above_line_l0_282


namespace binom_inequality_l0_131

-- Defining the conditions as non-computable functions
def is_nonneg_integer := ℕ

-- Defining the binomial coefficient function
noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

-- The statement of the theorem
theorem binom_inequality (n k h : ℕ) (hn : n ≥ k + h) : binom n (k + h) ≥ binom (n - k) h :=
  sorry

end binom_inequality_l0_131


namespace find_two_heaviest_l0_945

theorem find_two_heaviest (a b c d : ℝ) : 
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d) →
  ∃ x y : ℝ, (x ≠ y) ∧ (x = max (max (max a b) c) d) ∧ (y = max (max (min (max a b) c) d) d) :=
by sorry

end find_two_heaviest_l0_945


namespace total_number_of_bills_l0_256

-- Definitions based on conditions given
def total_amount : ℕ := 150
def num_20_bills : ℕ := 4
def total_amount_10_bills : ℕ := 50
def value_5_bill : ℕ := 5
def value_10_bill : ℕ := 10
def value_20_bill : ℕ := 20

-- The total number of bills Samuel has
theorem total_number_of_bills : 
  let amount_20_bills := num_20_bills * value_20_bill in
  let remaining_amount := total_amount - (total_amount_10_bills + amount_20_bills) in
  let num_10_bills := total_amount_10_bills / value_10_bill in
  let num_5_bills := remaining_amount / value_5_bill in
  num_5_bills + num_10_bills + num_20_bills = 13 :=
by
  -- Proof is left as an exercise
  sorry

end total_number_of_bills_l0_256


namespace no_ordered_quadruples_l0_636

def matrix_inverse_condition (a b c d : ℝ) : Prop :=
  matrix.inv ![![a, b], ![c, d]] = ![![1/d, 1/c], ![1/b, 1/a]]

theorem no_ordered_quadruples : ∀ (a b c d : ℝ), ¬ matrix_inverse_condition a b c d :=
by
  intros a b c d
  unfold matrix_inverse_condition
  sorry

end no_ordered_quadruples_l0_636


namespace cos_135_eq_neg_inv_sqrt_2_l0_584

theorem cos_135_eq_neg_inv_sqrt_2 :
  cos (135 * Real.pi / 180) = - (1 / Real.sqrt 2) :=
by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_584


namespace number_of_cows_is_11_l0_194

variables (D C : ℕ)

def total_number_of_legs := 2 * D + 4 * C
def total_number_of_heads := D + C
def legs_condition := total_number_of_legs = 22 + 2 * total_number_of_heads

theorem number_of_cows_is_11 (h : legs_condition D C) : C = 11 := 
sorry

end number_of_cows_is_11_l0_194


namespace ratio_area_triangle_to_quadrilateral_l0_887

theorem ratio_area_triangle_to_quadrilateral (A B C D E H G : Point) (convex : convex_quadrilateral A B C D)
  (midpoint_H : midpoint H B D)
  (midpoint_G : midpoint G A C)
  (extension_AD_BC : extension AD BC E) :
  (area (triangle E H G) / area (quadrilateral A B C D) = 1/4) := 
sorry

end ratio_area_triangle_to_quadrilateral_l0_887


namespace cos_135_eq_neg_inv_sqrt2_l0_433

theorem cos_135_eq_neg_inv_sqrt2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt2_l0_433


namespace cos_135_eq_neg_inv_sqrt2_l0_441

theorem cos_135_eq_neg_inv_sqrt2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt2_l0_441


namespace cos_135_eq_neg_sqrt2_div_2_l0_461

theorem cos_135_eq_neg_sqrt2_div_2 : cos (135 * real.pi / 180) = - (real.sqrt 2 / 2) :=
by
  have h1 : 135 = 180 - 45 := by norm_num
  have h2 : cos (180 * real.pi / 180 - 45 * real.pi / 180) = - cos (45 * real.pi / 180) :=
    by rw [cos_sub, cos_pi, cos_pi_div_four, sin_pi, sin_pi_div_four]; norm_num
  have h3 : cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by norm_num
  rw [h1, ←h2, h3]
  norm_num
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_461


namespace count_integers_between_roots_l0_163

theorem count_integers_between_roots :
  let lower_bound := nat.ceil (Real.sqrt 5)
  let upper_bound := nat.floor (Real.sqrt 88)
  upper_bound - lower_bound + 1 = 7 :=
by
  sorry

end count_integers_between_roots_l0_163


namespace positive_difference_of_two_numbers_l0_917

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l0_917


namespace students_strawberries_correct_l0_839

def total_students := 900
def students_oranges := 130
def students_pears := 210
def students_apples := 275
def students_bananas := 93
def students_grapes := 119

def students_strawberries := total_students - (students_oranges + students_pears + students_apples + students_bananas + students_grapes)

theorem students_strawberries_correct : students_strawberries = 73 :=
by
  unfold students_strawberries
  unfold total_students students_oranges students_pears students_apples students_bananas students_grapes
  simp
  sorry

end students_strawberries_correct_l0_839


namespace cos_135_eq_neg_inv_sqrt2_l0_445

theorem cos_135_eq_neg_inv_sqrt2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt2_l0_445


namespace arithmetic_sequence_sum_l0_737

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h : a 3 + a 4 + a 5 = 12) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 := 
by
  sorry

end arithmetic_sequence_sum_l0_737


namespace num_of_n_satisfies_l0_165

open Nat

theorem num_of_n_satisfies (h : ∀ n : ℕ, (n > 0 ∧ 4 ∣ n) → lcm 24 n = 4 * gcd 40320 n) :
    ∃ (N : ℕ), N = 72 ∧ ∀ n : ℕ, (n > 0 ∧ 4 ∣ n ∧ lcm 24 n = 4 * gcd 40320 n) ↔ (n ≤ N) :=
sorry

end num_of_n_satisfies_l0_165


namespace angle_between_vectors_l0_722

variable (a b : ℝ^3)
variable (norm_a : ℝ) (norm_b : ℝ) (dot_ab : ℝ)

-- Assuming the given conditions
axioms 
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 2)
  (hab : a ⬝ b = -sqrt 3)

-- Prove that the angle θ between a and b is 5π/6
theorem angle_between_vectors (a b : ℝ^3)
  (ha : ∥a∥ = 1) (hb : ∥b∥ = 2) 
  (hab : a ⬝ b = -sqrt 3) : 
  ∃ θ : ℝ, θ = 5 * Real.pi / 6 := 
sorry

end angle_between_vectors_l0_722


namespace ratio_of_logs_l0_263

noncomputable def log_base (b x : ℝ) := (Real.log x) / (Real.log b)

theorem ratio_of_logs (a b : ℝ) 
    (h1 : 0 < a) (h2 : 0 < b) 
    (h3 : log_base 8 a = log_base 18 b)
    (h4 : log_base 18 b = log_base 32 (a + b)) : 
    b / a = (1 + Real.sqrt 5) / 2 :=
by sorry

end ratio_of_logs_l0_263


namespace no_such_m_exists_l0_95

-- Define the sequence S_n as the LCM of numbers 1 to n
def S (n : ℕ) : ℕ := nat.lcm_list (list.range (n + 1))

-- Prove that there isn't a natural number m such that S (m + 1) = 4 * S m
theorem no_such_m_exists : ¬ ∃ (m : ℕ), S (m + 1) = 4 * S m :=
by 
  -- The detailed proof is omitted here
  sorry

end no_such_m_exists_l0_95


namespace parallel_planes_transitivity_l0_692

-- Define different planes α, β, γ
variables (α β γ : Plane)

-- Define the parallel relation between planes
axiom parallel : Plane → Plane → Prop

-- Conditions
axiom diff_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ
axiom β_parallel_α : parallel β α
axiom γ_parallel_α : parallel γ α

-- Statement to prove
theorem parallel_planes_transitivity (α β γ : Plane) 
  (h1 : parallel β α) 
  (h2 : parallel γ α) 
  (h3 : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) : parallel β γ :=
sorry

end parallel_planes_transitivity_l0_692


namespace rectangle_length_approx_35_l0_94

theorem rectangle_length_approx_35 (w x : ℝ) (h_wx : 3 * w = 2 * x) (h_area : 5 * ((x * (2 / 3) * x) = 4000)) : x ≈ 35 :=
by
  sorry

end rectangle_length_approx_35_l0_94


namespace cos_135_eq_neg_inv_sqrt_2_l0_545

theorem cos_135_eq_neg_inv_sqrt_2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 :=
sorry

end cos_135_eq_neg_inv_sqrt_2_l0_545


namespace cos_135_eq_neg_inv_sqrt_2_l0_556

theorem cos_135_eq_neg_inv_sqrt_2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 :=
sorry

end cos_135_eq_neg_inv_sqrt_2_l0_556


namespace value_of_7_prime_prime_l0_705

-- Define the function q' (written as q_prime in Lean)
def q_prime (q : ℕ) : ℕ := 3 * q - 3

-- Define the specific value problem
theorem value_of_7_prime_prime : q_prime (q_prime 7) = 51 := by
  sorry

end value_of_7_prime_prime_l0_705


namespace proof_cos_135_degree_l0_571

noncomputable def cos_135_degree_is_negative_sqrt2_over_2 : Prop :=
  real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2

theorem proof_cos_135_degree : cos_135_degree_is_negative_sqrt2_over_2 :=
by
  sorry

end proof_cos_135_degree_l0_571


namespace gcd_7163_209_is_19_l0_108

theorem gcd_7163_209_is_19 :
  ∃ d : ℕ, d = Nat.gcd 7163 209 ∧ d = 19 :=
begin
  have eq1 : 7163 = 209 * 34 + 57 := by norm_num,
  have eq2 : 209 = 57 * 3 + 38 := by norm_num,
  have eq3 : 57 = 38 * 1 + 19 := by norm_num,
  have eq4 : 38 = 19 * 2 := by norm_num,
  sorry
end

end gcd_7163_209_is_19_l0_108


namespace trig_identity_proof_l0_426

theorem trig_identity_proof :
  2 * (1 / 2) + (Real.sqrt 3 / 2) * Real.sqrt 3 = 5 / 2 :=
by
  sorry

end trig_identity_proof_l0_426


namespace solution_set_inequality_l0_891

theorem solution_set_inequality (x : ℝ) : 2 * x^2 - x - 3 > 0 ↔ x > 3 / 2 ∨ x < -1 := by
  sorry

end solution_set_inequality_l0_891


namespace cos_135_eq_neg_sqrt2_div_2_l0_480

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by 
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_480


namespace medians_intersect_single_point_isosceles_if_two_medians_equal_l0_120

variable {α β γ : Type} [LinearOrder α] [LinearOrder β] [LinearOrder γ]

-- Definitions for the medians
structure Triangle where
  A B C : α

structure Medians (K1 K2 K3 : α) (T : Triangle) where
  K1 : α
  K2 : α
  K3 : α

-- Lean statement to prove that the medians intersect at a single point
theorem medians_intersect_single_point {T : Triangle} {K1 K2 K3 : α}
  (m : Medians K1 K2 K3 T) : ∃ P : α, P ∈ {K1, K2, K3} := 
  sorry

-- Constructing a triangle from its medians (definition)
def construct_triangle_from_medians {K1 K2 K3 : α} :
  Triangle :=
  sorry

-- Calculating the sides and area of a triangle from its medians (definition)
def calculate_sides_and_area (K1 K2 K3 : α) : (α × α × α) × α :=
  sorry

-- Prove that a triangle is isosceles if two medians are equal
theorem isosceles_if_two_medians_equal {T : Triangle} {K1 K2 K3 : α} (m : Medians K1 K2 K3 T)
  (h : K2 = K3) : 
  T.A = T.B ∨ T.B = T.C ∨ T.C = T.A := 
  sorry

end medians_intersect_single_point_isosceles_if_two_medians_equal_l0_120


namespace solution_set_of_inequality_l0_300

theorem solution_set_of_inequality : {x : ℝ | x^2 + x - 2 < 0} = {x : ℝ | -2 < x ∧ x < 1} :=
by
  sorry

end solution_set_of_inequality_l0_300


namespace copper_wire_diameter_l0_417

theorem copper_wire_diameter 
  (L : ℝ)    -- distance in meters
  (P : ℝ)    -- power in watts
  (V : ℝ)    -- voltage in volts
  (Ploss : ℝ)    -- maximum allowable power loss in watts
  (ρ : ℝ)    -- resistance per meter for a copper wire with a cross-sectional area of 1 mm^2 in ohms
  (H_L : L = 68000)
  (H_P : P = 223710)
  (H_V : V = 20000)
  (H_Ploss : Ploss = 22371)
  (H_ρ : ρ = 1 / 55)
  : ∃ d : ℝ, d ≈ 2.96 := 
sorry

end copper_wire_diameter_l0_417


namespace positive_difference_of_two_numbers_l0_919

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l0_919


namespace card_statements_l0_810

def statement_1 (s : Fin 5 → Bool) : Prop := (∃! i, s i = false)
def statement_2 (s : Fin 5 → Bool) : Prop := (∃! i1 i2, s i1 = false ∧ s i2 = false ∧ i1 ≠ i2)
def statement_3 (s : Fin 5 → Bool) : Prop := (∃! i1 i2 i3, s i1 = false ∧ s i2 = false ∧ s i3 = false ∧ i1 ≠ i2 ∧ i2 ≠ i3 ∧ i1 ≠ i3)
def statement_4 (s : Fin 5 → Bool) : Prop := (∃! i1 i2 i3 i4, s i1 = false ∧ s i2 = false ∧ s i3 = false ∧ s i4 = false ∧ i1 ≠ i2 ∧ i2 ≠ i3 ∧ i3 ≠ i4 ∧ i1 ≠ i4)
def statement_5 (s : Fin 5 → Bool) : Prop := (∃! i, s i = true)

noncomputable def number_false_statements (s : Fin 5 → Bool) : ℕ :=
  5 - (Finset.filter (λ i, s i) (Finset.univ : Finset (Fin 5))).card

theorem card_statements (s : Fin 5 → Bool) :
  statement_1 s ∨ statement_2 s ∨ statement_3 s ∨ statement_4 s ∨ statement_5 s →
  number_false_statements s = 4 :=
begin
  sorry
end

end card_statements_l0_810


namespace difference_in_male_and_female_l0_294

open Nat

variable (boys girls : ℕ)
variable (ratio_condition : 3 * girls = 4 * boys)
variable (total_condition : boys + girls = 42)
variable (diff := girls - boys)

theorem difference_in_male_and_female : diff = 6 :=
by
  have boy_ratio : girls = (4 * boys) / 3 := by sorry
  have girl_ratio : boys = (3 * girls) / 4 := by sorry
  calc diff
       = girls - boys : by sorry
    ... = 6 : by sorry

end difference_in_male_and_female_l0_294


namespace cos_135_eq_neg_inv_sqrt2_l0_443

theorem cos_135_eq_neg_inv_sqrt2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt2_l0_443


namespace cos_135_eq_neg_sqrt2_div_2_l0_515

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = - Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_515


namespace problem_statement_l0_776

theorem problem_statement (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^2 + y^2 + z^2 + x * y + y * z + z * x) / 6 ≤ 
  ((x + y + z) / 3) * sqrt((x^2 + y^2 + z^2) / 3) :=
by
  sorry

end problem_statement_l0_776


namespace two_lines_parallel_plane_not_imply_planes_parallel_l0_743

-- Definitions of lines, planes, and parallelism.
structure Line := (p1 p2 : Point) -- A line can be represented by two points.
structure Plane := (p1 p2 p3 : Point) -- A plane can be represented by three points.
def parallel (P : Plane) (Q : Plane) : Prop := sorry -- Parallelism between two planes
def parallel_to_plane (l : Line) (P : Plane) : Prop := sorry -- A line is parallel to a plane

-- Points and conditions
variable (P1 P2 : Plane)
variable (l1 l2 : Line)
variable (not_parallel : ¬ (parallel P1 P2))
variable (h1 : parallel_to_plane l1 P2)
variable (h2 : parallel_to_plane l2 P2)

-- Prove the statement
theorem two_lines_parallel_plane_not_imply_planes_parallel : 
  (l1 p1, l1 p2 are in P1) → (l2 p1, l2 p2 are in P1) → 
  parallel_to_plane l1 P2 → parallel_to_plane l2 P2 → 
  ¬ parallel P1 P2 :=
begin
  intros, 
  exact not_parallel,
end

end two_lines_parallel_plane_not_imply_planes_parallel_l0_743


namespace sasha_studies_more_avg_4_l0_869

-- Define the differences recorded over the five days
def differences : List ℤ := [20, 0, 30, -20, -10]

-- Calculate the average difference
def average_difference (diffs : List ℤ) : ℚ :=
  (List.sum diffs : ℚ) / (List.length diffs : ℚ)

-- The statement to prove
theorem sasha_studies_more_avg_4 :
  average_difference differences = 4 := by
  sorry

end sasha_studies_more_avg_4_l0_869


namespace volume_of_prism_l0_278

noncomputable def volume_rectangular_prism : Float :=
  let b := 48 / 5.5
  let l := 3 * b
  let h := (1 / 2) * l
  l * b * h

theorem volume_of_prism :
  (volume_rectangular_prism ≈ 2992.727272) :=
by sorry

end volume_of_prism_l0_278


namespace tan_addition_formula_tan_product_l0_359

-- Definition of tangent addition formula as a condition
theorem tan_addition_formula (a b : ℝ) :
  tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := sorry

-- Given conditions for specific degrees
def tan_1_deg := tan (π / 180) -- π/180 rad is equal to 1°
def tan_44_deg := tan (44 * π / 180) -- 44° in radians

-- The main theorem to be proven
theorem tan_product :
  (1 + tan_1_deg) * (1 + tan_44_deg) = 2 := sorry

end tan_addition_formula_tan_product_l0_359


namespace shift_lemma_phi_value_l0_257

noncomputable def shifted_function (x : ℝ) : ℝ :=
  sin (4 * (x + π / 12))

axiom original_function (x : ℝ) : ℝ :=
  sin (4 * x)

theorem shift_lemma (x : ℝ) : shifted_function x = sin (4 * x + π / 3) :=
sorry

theorem phi_value : ∀ x, shifted_function x = sin(4 * x + π / 3) → π / 3 = π / 3 :=
by
  intros _ h
  assumption

end shift_lemma_phi_value_l0_257


namespace area_ratio_l0_255

theorem area_ratio (P1 P2 P4 : ℕ) (h1 : P1 = 16) (h2 : P2 = 32) (h4 : P4 = 20) : 
  (P2 / 4) ^ 2 / (P4 / 4) ^ 2 = 64 / 25 :=
by
  -- Definitions based on given perimeters
  set s2 := P2 / 4 with hs2
  set s4 := P4 / 4 with hs4
  -- Calculations of areas
  have a2 : s2 ^ 2 = 64 := by
    calc
      s2 ^ 2 = (32 / 4) ^ 2 : by rw h2
           ... = 8 ^ 2      : by norm_num
           ... = 64         : by norm_num
  have a4 : s4 ^ 2 = 25 := by
    calc
      s4 ^ 2 = (20 / 4) ^ 2 : by rw h4
           ... = 5 ^ 2      : by norm_num
           ... = 25         : by norm_num
  -- Final ratio
  calc
    (P2 / 4) ^ 2 / (P4 / 4) ^ 2 = 64 / 25 : by
      rw [←a2, ←a4]

end area_ratio_l0_255


namespace percentage_insurance_is_5_l0_242

-- Given conditions as Lean definitions
def salary : ℝ := 2000
def tax_rate : ℝ := 0.20
def tax_amount : ℝ := tax_rate * salary
def remaining_after_tax : ℝ := salary - tax_amount
def utility_fraction : ℝ := 1 / 4
def remaining_after_utility (P : ℝ) : ℝ := remaining_after_tax - (P / 100 * salary) - utility_fraction * (remaining_after_tax - (P / 100 * salary))
def remaining_amount : ℝ := 1125

-- Prove that the percentage of Maria's salary going to insurance is 5%
theorem percentage_insurance_is_5 : ∃ P : ℝ, remaining_after_utility P = remaining_amount ∧ P = 5 :=
by
  sorry

end percentage_insurance_is_5_l0_242


namespace isosceles_triangle_congruent_side_length_l0_859

theorem isosceles_triangle_congruent_side_length (base height: ℝ) (area: ℝ)
  (h_base: base = 18)
  (h_area: area = 72)
  (h_area_formula: 2 * area = base * height) : 
  ∃ (congruent_side : ℝ), congruent_side ^ 2 = 145 :=
by {
  have height_eq : height = 8, 
  {
    calc
      2 * 72 = 18 * height : by rw [h_area, h_area_formula, h_base]
      144 = 18 * height
      144 / 18 = height
      8 = height
  },
  use sqrt(9^2 + 8^2),
  have h1: 9^2 + 8^2 = 81 + 64, 
  sorry,
  have h2: 81 + 64 = 145,
  sorry,
  rw [sqrt_sq (by norm_num : 0 ≤ 145)], -- sqrt and sqr are inverses for non-negative 145
  rw [← h2, h1],
}

end isosceles_triangle_congruent_side_length_l0_859


namespace drink_total_amount_l0_208

theorem drink_total_amount (parts_coke parts_sprite parts_mountain_dew ounces_coke total_parts : ℕ)
  (h1 : parts_coke = 2) (h2 : parts_sprite = 1) (h3 : parts_mountain_dew = 3)
  (h4 : total_parts = parts_coke + parts_sprite + parts_mountain_dew)
  (h5 : ounces_coke = 6) :
  ( ounces_coke * total_parts ) / parts_coke = 18 :=
by
  sorry

end drink_total_amount_l0_208


namespace positive_difference_of_two_numbers_l0_935

theorem positive_difference_of_two_numbers (x y : ℝ)
  (h1 : x + y = 10) (h2 : x ^ 2 - y ^ 2 = 40) : abs (x - y) = 4 := 
by
  sorry

end positive_difference_of_two_numbers_l0_935


namespace polygon_sides_eq_eight_l0_177

theorem polygon_sides_eq_eight (n : ℕ) 
  (h_diff : (n - 2) * 180 - 360 = 720) :
  n = 8 := 
by 
  sorry

end polygon_sides_eq_eight_l0_177


namespace cos_135_eq_neg_sqrt2_div_2_l0_504

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = - Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_504


namespace smallest_positive_period_l0_236

noncomputable def f (A ω φ : ℝ) (x : ℝ) := A * Real.sin (ω * x + φ)

theorem smallest_positive_period 
  (A ω φ T : ℝ) 
  (hA : A > 0) 
  (hω : ω > 0)
  (h1 : f A ω φ (π / 2) = f A ω φ (2 * π / 3))
  (h2 : f A ω φ (π / 6) = -f A ω φ (π / 2))
  (h3 : ∀ x1 x2, (π / 6) ≤ x1 → x1 ≤ x2 → x2 ≤ (π / 2) → f A ω φ x1 ≤ f A ω φ x2) :
  T = π :=
sorry

end smallest_positive_period_l0_236


namespace sum_abc_is_eight_l0_851

theorem sum_abc_is_eight (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : (a + b + c)^3 - a^3 - b^3 - c^3 = 294) : a + b + c = 8 :=
by
  sorry

end sum_abc_is_eight_l0_851


namespace negative_card_probability_l0_316

def cards : List ℤ := [0, 1, -1, 2, -2, 3]

def number_of_cards : ℕ := cards.length

def number_of_negative_cards : ℕ := cards.count (λ x : ℤ, x < 0)

def probability_of_negative_card : ℚ := number_of_negative_cards / number_of_cards

theorem negative_card_probability : probability_of_negative_card = 1 / 3 :=
by
  sorry

end negative_card_probability_l0_316


namespace adam_money_given_l0_406

theorem adam_money_given (original_money : ℕ) (final_money : ℕ) (money_given : ℕ) :
  original_money = 79 →
  final_money = 92 →
  money_given = final_money - original_money →
  money_given = 13 := by
sorry

end adam_money_given_l0_406


namespace percentage_plane_enclosed_by_hexagons_l0_291

noncomputable
def percentageEnclosedByHexagons (a : ℝ) : ℝ :=
  let area_total := 16 * a^2
  let area_hexagons := 8 * a^2
  (area_hexagons / area_total) * 100

theorem percentage_plane_enclosed_by_hexagons (a : ℝ) :
  percentageEnclosedByHexagons(a) = 50 := by
  sorry

end percentage_plane_enclosed_by_hexagons_l0_291


namespace andy_demerits_l0_45

theorem andy_demerits (x : ℕ) :
  (∀ x, 6 * x + 15 = 27 → x = 2) :=
by
  intro
  sorry

end andy_demerits_l0_45


namespace final_theorem_l0_102

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := (x - 2) ^ 2

-- Definition of an even function
def is_even_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g x = g (-x)

-- Proposition A to prove f(x + 2) is an even function
lemma PropositionA : is_even_function (λ x, f (x + 2)) :=
sorry

-- Proposition B to check if the function is decreasing on (-∞, 2) and increasing on (2, +∞)
def is_decreasing_on (g : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y : ℝ, x < y → x ∈ I → y ∈ I → g x > g y

def is_increasing_on (g : ℝ → ℝ) (I: Set ℝ) : Prop :=
  ∀ x y : ℝ, x < y → x ∈ I → y ∈ I → g x < g y

lemma PropositionB :
  is_decreasing_on f {x | x < 2} ∧ is_increasing_on f {x | 2 < x} :=
sorry

-- Final theorem combining both propositions.
theorem final_theorem :
  (is_even_function (λ x, f (x + 2)) ∧
   is_decreasing_on f {x | x < 2} ∧
   is_increasing_on f {x | 2 < x}) :=
  by
  { 
    exact ⟨PropositionA, PropositionB⟩
  }

end final_theorem_l0_102


namespace min_points_dodecahedron_min_points_icosahedron_l0_341

-- Definitions for the dodecahedron
def dodecahedron_faces : ℕ := 12
def vertices_per_face_dodecahedron : ℕ := 3

-- Prove the minimum number of points to mark each face of a dodecahedron
theorem min_points_dodecahedron (n : ℕ) (h : 3 * n >= dodecahedron_faces) : n >= 4 :=
sorry

-- Definitions for the icosahedron
def icosahedron_faces : ℕ := 20
def icosahedron_vertices : ℕ := 12

-- Prove the minimum number of points to mark each face of an icosahedron
theorem min_points_icosahedron (n : ℕ) (h : n >= 6) : n = 6 :=
sorry

end min_points_dodecahedron_min_points_icosahedron_l0_341


namespace two_estates_problem_l0_261

theorem two_estates_problem :
  ∃ (x y : ℕ), 
    (xy = 156000) ∧ 
    ((x - 7000) * (y - 0.5) = 104500) ∧ 
    ((x = 26000) ∧ (y = 6) ∨ 
     (x = 84000) ∧ (y = 1.857)) :=
by {
  sorry
}

end two_estates_problem_l0_261


namespace smallest_positive_integer_l0_962

theorem smallest_positive_integer (x : ℕ) :
  (x % 5 = 2) ∧ (x % 7 = 3) ∧ (x % 9 = 4) → x = 157 :=
begin
  sorry
end

end smallest_positive_integer_l0_962


namespace cos_135_eq_neg_sqrt2_div_2_l0_620

theorem cos_135_eq_neg_sqrt2_div_2 : 
  Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_620


namespace problem1_problem2_l0_157

variables (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ)

def a_defs : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 2 * a n + n - 1

def b_defs : Prop :=
  ∀ n, b n = a n + n

def b_geometric : Prop :=
  ∃ q, q ≠ 0 ∧ ∀ n, b (n + 1) = q * b n

def S_n_formula (n : ℕ) : ℕ :=
  2^(n+1) - 2 - (n^2 + n) / 2

def S_n_defs : Prop :=
  S 0 = 0 ∧ ∀ n, S (n + 1) = S n + a (n + 1)

def S_n_correct : Prop :=
  ∀ n, S n = S_n_formula n

theorem problem1 (a : ℕ → ℕ) (b : ℕ → ℕ) :
  a_defs a → b_defs a b → b_geometric b :=
by sorry

theorem problem2 (a : ℕ → ℕ) (S : ℕ → ℕ) :
  a_defs a → S_n_defs a S → S_n_correct a S :=
by sorry

end problem1_problem2_l0_157


namespace prime_count_inequality_l0_821

-- Conditions: n is a natural number and π(n) represents the number of prime numbers less than or equal to n.

noncomputable def pi (n : ℕ) : ℕ :=
  Nat.Prime.pi n

theorem prime_count_inequality (n : ℕ) (h : 1 ≤ n) : 
  pi(n) ≥ Int.toNat (Real.log n / Real.log 4) :=
by
  sorry

end prime_count_inequality_l0_821


namespace positive_difference_of_two_numbers_l0_914

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l0_914


namespace max_profit_l0_949

noncomputable def profit_A (x : ℝ) : ℝ := -x^2 + 4 * x

noncomputable def profit_B (x : ℝ) : ℝ := 2 * x

noncomputable def total_profit (x : ℝ) : ℝ :=
  profit_A x + profit_B (3 - x)

theorem max_profit :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 3 ∧ total_profit x = 7 :=
by
  use 1
  split
  · linarith
  split
  · linarith
  · simp [total_profit, profit_A, profit_B]
  sorry  -- Provide the detailed proof of maximum calculation and verification

end max_profit_l0_949


namespace range_of_m_l0_672

noncomputable def p (x : ℝ) : Prop := abs (1 - (x - 1) / 3) ≤ 2
noncomputable def q (x : ℝ) (m : ℝ) : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) : (∀ x : ℝ, ¬p x → ¬q x m) → (m ≥ 9) :=
by
  sorry

end range_of_m_l0_672


namespace sum_of_coordinates_after_reflections_l0_249

theorem sum_of_coordinates_after_reflections :
  let A := (3, 2)
  let B := (9, 18)
  let N := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let reflect_y (P : ℤ × ℤ) := (-P.1, P.2)
  let reflect_x (P : ℤ × ℤ) := (P.1, -P.2)
  let N' := reflect_y N
  let N'' := reflect_x N'
  N''.1 + N''.2 = -16 := by sorry

end sum_of_coordinates_after_reflections_l0_249


namespace pairs_satisfying_int_l0_650

theorem pairs_satisfying_int (a b : ℕ) :
  ∃ n : ℕ, a = 2 * n^2 + 1 ∧ b = n ↔ (2 * a * b^2 + 1) ∣ (a^3 + 1) := by
  sorry

end pairs_satisfying_int_l0_650


namespace function_properties_l0_113

variable (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def even_translated_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-(x) + π/2) = f (x + π/2)

theorem function_properties 
  (h1 : odd_function f)
  (h2 : even_translated_function f) :
  (∀ x, f (x + 2 * π) = f (x)) ∧ -- periodicity
  (∀ k : ℤ, (∀ x, f (-x + (π / 2 + 2 * k * π)) = f (x + (π / 2 + 2 * k * π)))) ∧ -- symmetry axis
  (f (π/2) = f (π/2) ∧ f (0) = 0) := -- symmetry center
sorry

end function_properties_l0_113


namespace cos_135_eq_neg_inv_sqrt_2_l0_522

theorem cos_135_eq_neg_inv_sqrt_2 : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_522


namespace sale_in_fifth_month_l0_382

theorem sale_in_fifth_month 
  (S1 S2 S3 S4 S6 : ℕ)
  (h1 : S1 = 6735)
  (h2 : S2 = 6927)
  (h3 : S3 = 6855)
  (h4 : S4 = 7230)
  (h5 : S6 = 4691)
  (avg_sale : ℕ)
  (h_avg_sale : avg_sale = 6500) :
  let total_sales_required := avg_sale * 6,
      total_sales_first_four := S1 + S2 + S3 + S4,
      total_sales_sixth := total_sales_first_four + S6 in
  total_sales_required - total_sales_sixth = 6562 :=
by
  sorry

end sale_in_fifth_month_l0_382


namespace cos_135_eq_neg_sqrt2_div_2_l0_472

theorem cos_135_eq_neg_sqrt2_div_2 : cos (135 * real.pi / 180) = - (real.sqrt 2 / 2) :=
by
  have h1 : 135 = 180 - 45 := by norm_num
  have h2 : cos (180 * real.pi / 180 - 45 * real.pi / 180) = - cos (45 * real.pi / 180) :=
    by rw [cos_sub, cos_pi, cos_pi_div_four, sin_pi, sin_pi_div_four]; norm_num
  have h3 : cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by norm_num
  rw [h1, ←h2, h3]
  norm_num
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_472


namespace min_value_arithmetic_sequence_l0_702

theorem min_value_arithmetic_sequence (a : ℕ → ℝ) 
  (h_arith_seq : ∀ n m, a (n + 1) - a n = a (m+1) - a m)
  (h_positive : ∀ n, 0 < a n)
  (h_sum : (∑ i in finset.range 9, a (i + 1)) = 9 / 2) :
  ∃ a_2 a_8, a_2 = a 2 ∧ a_8 = a 8 ∧ (1 / a_2 + 4 / a_8) ≥ 9 := sorry

end min_value_arithmetic_sequence_l0_702


namespace cos_135_eq_neg_sqrt2_div_2_l0_470

theorem cos_135_eq_neg_sqrt2_div_2 : cos (135 * real.pi / 180) = - (real.sqrt 2 / 2) :=
by
  have h1 : 135 = 180 - 45 := by norm_num
  have h2 : cos (180 * real.pi / 180 - 45 * real.pi / 180) = - cos (45 * real.pi / 180) :=
    by rw [cos_sub, cos_pi, cos_pi_div_four, sin_pi, sin_pi_div_four]; norm_num
  have h3 : cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by norm_num
  rw [h1, ←h2, h3]
  norm_num
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_470


namespace inequality_solution_set_l0_299

theorem inequality_solution_set : 
  { x : ℝ | x^2 + x - 2 < 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end inequality_solution_set_l0_299


namespace proof_problem_l0_137

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := f'

lemma f_even (x : ℝ) : f (3 / 2 - 2 * x) = f (3 / 2 + 2 * x) := sorry
lemma g_even (x : ℝ) : g (2 + x) = g (2 - x) := sorry

theorem proof_problem :
  (f (-1) = f 4) ∧ (g (-1 / 2) = 0) :=
by { split; { sorry } }

end proof_problem_l0_137


namespace cos_135_eq_correct_l0_603

noncomputable def cos_135_eq: Prop :=
  real.cos ((135 * real.pi) / 180) = -real.sqrt 2 / 2

theorem cos_135_eq_correct : cos_135_eq :=
  sorry

end cos_135_eq_correct_l0_603


namespace cos_135_eq_correct_l0_602

noncomputable def cos_135_eq: Prop :=
  real.cos ((135 * real.pi) / 180) = -real.sqrt 2 / 2

theorem cos_135_eq_correct : cos_135_eq :=
  sorry

end cos_135_eq_correct_l0_602


namespace positive_difference_l0_894

theorem positive_difference (a b : ℝ) (h₁ : a + b = 10) (h₂ : a^2 - b^2 = 40) : |a - b| = 4 :=
by
  sorry

end positive_difference_l0_894


namespace son_born_after_marriage_l0_207

theorem son_born_after_marriage (L : ℝ) (h1 : L = 83.99999999999996) 
  (h2 : ∀ t : ℝ, t = L * (1/6) + L * (1/12) + L * (1/7))
  (h3 : ∀ s : ℝ, s = L / 2)
  (h4 : ∀ d : ℝ, d = L - 4) :
  let married_age := L * (1/6) + L * (1/12) + L * (1/7),
      son_born_age := L - (L / 2 + 4)
  in son_born_age - married_age = 5 :=
by
  sorry

end son_born_after_marriage_l0_207


namespace solution_to_inequality_system_l0_892

theorem solution_to_inequality_system (x : ℝ) :
  (x + 3 ≥ 2 ∧ (3 * x - 1) / 2 < 4) ↔ -1 ≤ x ∧ x < 3 :=
by
  sorry

end solution_to_inequality_system_l0_892


namespace max_value_of_f_l0_284

-- Define points A and B
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (0, 1)

-- Define the function f(x) as the difference in distances
def dist (p q : ℝ × ℝ) : ℝ := 
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def f (x : ℝ) : ℝ := 
  dist (x, x^2) A - dist (x, x^2) B

-- Theorem statement
theorem max_value_of_f : ∃ x, f(x) = real.sqrt 10 :=
sorry

end max_value_of_f_l0_284


namespace injective_g_restricted_to_interval_l0_982

def g (x : ℝ) : ℝ := (x + 3) ^ 2 - 10

theorem injective_g_restricted_to_interval :
  (∀ x1 x2 : ℝ, x1 ∈ Set.Ici (-3) → x2 ∈ Set.Ici (-3) → g x1 = g x2 → x1 = x2) :=
sorry

end injective_g_restricted_to_interval_l0_982


namespace cos_135_eq_neg_sqrt2_div_2_l0_474

theorem cos_135_eq_neg_sqrt2_div_2 : cos (135 * real.pi / 180) = - (real.sqrt 2 / 2) :=
by
  have h1 : 135 = 180 - 45 := by norm_num
  have h2 : cos (180 * real.pi / 180 - 45 * real.pi / 180) = - cos (45 * real.pi / 180) :=
    by rw [cos_sub, cos_pi, cos_pi_div_four, sin_pi, sin_pi_div_four]; norm_num
  have h3 : cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by norm_num
  rw [h1, ←h2, h3]
  norm_num
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_474


namespace smallest_s_minus_p_l0_666

theorem smallest_s_minus_p : 
  ∃ (p q r s : ℕ), p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 ∧ p < q ∧ q < r ∧ r < s ∧ 
                 (p * q * r * s = nat.factorial 9) ∧ (s - p = 52) := 
by 
  sorry

end smallest_s_minus_p_l0_666


namespace cos_135_eq_neg_sqrt2_div_2_l0_463

theorem cos_135_eq_neg_sqrt2_div_2 : cos (135 * real.pi / 180) = - (real.sqrt 2 / 2) :=
by
  have h1 : 135 = 180 - 45 := by norm_num
  have h2 : cos (180 * real.pi / 180 - 45 * real.pi / 180) = - cos (45 * real.pi / 180) :=
    by rw [cos_sub, cos_pi, cos_pi_div_four, sin_pi, sin_pi_div_four]; norm_num
  have h3 : cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by norm_num
  rw [h1, ←h2, h3]
  norm_num
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_463


namespace cos_135_eq_neg_sqrt2_div_2_l0_592

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  -- Conditions incorporated directly into the requirement due to the nature of trigonometric evaluations.
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_592


namespace compute_expression_l0_663

variables {α : Type*} [linear_order α]

noncomputable def M (x y : α) : α := max x y
noncomputable def m (x y : α) : α := min x y

theorem compute_expression (a b c d e : α) (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e)
  : M (m b (M a d)) (M c (m e d)) = d :=
by sorry

end compute_expression_l0_663


namespace positive_difference_of_two_numbers_l0_901

theorem positive_difference_of_two_numbers 
  (a b : ℝ) 
  (h₁ : a + b = 10) 
  (h₂ : a^2 - b^2 = 40) : 
  |a - b| = 4 := 
sorry

end positive_difference_of_two_numbers_l0_901


namespace parallelogram_area_leq_half_triangle_area_l0_766

-- Definition of a triangle and a parallelogram inside it.
structure Triangle (α : Type) [LinearOrderedField α] :=
(A B C : α × α)

structure Parallelogram (α : Type) [LinearOrderedField α] :=
(P Q R S : α × α)

-- Function to calculate the area of a triangle
def triangle_area {α : Type} [LinearOrderedField α] (T : Triangle α) : α :=
-- Placeholder for the actual area calculation formula
sorry

-- Function to calculate the area of a parallelogram
def parallelogram_area {α : Type} [LinearOrderedField α] (P : Parallelogram α) : α :=
-- Placeholder for the actual area calculation formula
sorry

-- Statement of the problem
theorem parallelogram_area_leq_half_triangle_area {α : Type} [LinearOrderedField α]
(T : Triangle α) (P : Parallelogram α) (inside : P.P.1 < T.A.1 ∧ P.P.2 < T.C.1) : 
  parallelogram_area P ≤ 1 / 2 * triangle_area T :=
sorry

end parallelogram_area_leq_half_triangle_area_l0_766


namespace right_triangle_ratio_maximum_l0_395

theorem right_triangle_ratio_maximum (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = a + b) :
  (a + b + c) / c ≤ 1 + real.sqrt 2 := 
sorry

end right_triangle_ratio_maximum_l0_395


namespace cos_135_eq_neg_sqrt2_div_2_l0_468

theorem cos_135_eq_neg_sqrt2_div_2 : cos (135 * real.pi / 180) = - (real.sqrt 2 / 2) :=
by
  have h1 : 135 = 180 - 45 := by norm_num
  have h2 : cos (180 * real.pi / 180 - 45 * real.pi / 180) = - cos (45 * real.pi / 180) :=
    by rw [cos_sub, cos_pi, cos_pi_div_four, sin_pi, sin_pi_div_four]; norm_num
  have h3 : cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by norm_num
  rw [h1, ←h2, h3]
  norm_num
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_468


namespace quadrilateral_theorem_l0_752

-- Define the geometrical setup
structure Quadrilateral (α : Type*) [ordered_ring α] :=
(A B C D K : α)
(AB BC CD DA BD AK KC AD : α)

variables {α : Type*} [ordered_ring α]

noncomputable def given_setup (Q : Quadrilateral α) (h1 : Q.AB = Q.BC) 
(h2 : ∠ (Q.A Q.K Q.B) + ∠ (Q.B Q.K Q.C) = ∠ (Q.A Q.C Q.D) + ∠ (Q.D Q.A Q.C)) : Prop :=
Q.AK * Q.CD = Q.KC * Q.AD

-- The theorem we'd like to prove
theorem quadrilateral_theorem (Q : Quadrilateral α) (h1 : Q.AB = Q.BC)
(h2 : ∠ (Q.A Q.K Q.B) + ∠ (Q.B Q.K Q.C) = ∠ (Q.A Q.C Q.D) + ∠ (Q.D Q.A Q.C)) : 
Q.AK * Q.CD = Q.KC * Q.AD :=
by 
  sorry

end quadrilateral_theorem_l0_752


namespace vector_magnitude_l0_697

variable (x : ℝ)
def a : ℝ × ℝ := (x, 1)
def b : ℝ × ℝ := (1, -2)

-- Use the orthogonality condition
def ortho_condition : Prop := x - 2 = 0

theorem vector_magnitude (h : ortho_condition) : |(a + b : ℝ × ℝ)| = Real.sqrt 10 := by
  sorry

end vector_magnitude_l0_697


namespace cos_angle_AME_l0_981

open Real

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def dist (p1 p2 : Point3D) : ℝ := sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

def dot_product (v1 v2 : Point3D) : ℝ := v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def magnitude (v : Point3D) : ℝ := sqrt (v.x^2 + v.y^2 + v.z^2)

noncomputable def cos_angle (v1 v2 : Point3D) : ℝ := 
  dot_product v1 v2 / (magnitude v1 * magnitude v2)

def A := Point3D.mk 0 0 0
def B := Point3D.mk 2 0 0
def C := Point3D.mk 2 2 0
def D := Point3D.mk 0 2 0
def E := Point3D.mk 1 1 2
def M := Point3D.mk 1 2 0

def AM := Point3D.mk 1 2 0  -- M - A
def AE := Point3D.mk 1 1 2  -- E - A

theorem cos_angle_AME : cos_angle AM AE = sqrt 30 / 10 := 
  sorry

end cos_angle_AME_l0_981


namespace positive_difference_l0_896

theorem positive_difference (a b : ℝ) (h₁ : a + b = 10) (h₂ : a^2 - b^2 = 40) : |a - b| = 4 :=
by
  sorry

end positive_difference_l0_896


namespace total_wages_l0_986

theorem total_wages (A_days B_days : ℝ) (A_wages : ℝ) (W : ℝ) 
  (h1 : A_days = 10)
  (h2 : B_days = 15)
  (h3 : A_wages = 2100) :
  W = 3500 :=
by sorry

end total_wages_l0_986


namespace max_garden_area_side_length_eq_75_l0_392

noncomputable def max_garden_side_length
  (fence_cost_per_foot : ℝ)
  (total_cost : ℝ)
  (total_wall_length : ℝ) : ℝ :=
  let total_fence_length := total_cost / fence_cost_per_foot
  let f := λ x : ℝ, x * (total_fence_length - 2 * x)
  let x := total_fence_length / 6
  total_fence_length - 2 * x

theorem max_garden_area_side_length_eq_75 :
  max_garden_side_length 10 1500 300 = 75 := by
  sorry

end max_garden_area_side_length_eq_75_l0_392


namespace cos_135_eq_neg_inv_sqrt2_l0_440

theorem cos_135_eq_neg_inv_sqrt2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt2_l0_440


namespace proof_cos_135_degree_l0_565

noncomputable def cos_135_degree_is_negative_sqrt2_over_2 : Prop :=
  real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2

theorem proof_cos_135_degree : cos_135_degree_is_negative_sqrt2_over_2 :=
by
  sorry

end proof_cos_135_degree_l0_565


namespace num_two_fifths_in_twice_one_tenth_l0_166

-- Definitions based on conditions
def twice_one_tenth : ℚ := 2 * (1/10)
def one_fifth : ℚ := 1/5
def two_fifths : ℚ := 2/5

-- Ensure twice one-tenth is correctly identified as one-fifth
lemma twice_one_tenth_is_one_fifth : twice_one_tenth = one_fifth := by
  unfold twice_one_tenth
  unfold one_fifth
  norm_num

-- The statement we need to prove
theorem num_two_fifths_in_twice_one_tenth : (one_fifth / two_fifths) = 1/2 := by
  unfold one_fifth
  unfold two_fifths
  norm_num
  sorry

end num_two_fifths_in_twice_one_tenth_l0_166


namespace find_value_a1_to_a10_l0_106

theorem find_value_a1_to_a10 (a : ℕ → ℕ) : 
  (a 1) + 2 * (a 2) + 3 * (a 3) + 4 * (a 4) + 5 * (a 5) + 6 * (a 6) + 7 * (a 7) + 8 * (a 8) + 9 * (a 9) + 10 * (a 10) = 20 :=
by
  -- Problem statement conditions
  have h: (1 - 2 * x) ^ 10 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7 + a 8 * x^8 + a 9 * x^9 + a 10 * x^10,
  {
    sorry  -- This is provided as a condition
  },
  sorry  -- Full proof would be written here 

#print axioms find_value_a1_to_a10

end find_value_a1_to_a10_l0_106


namespace equal_angles_AXZ_BXZ_l0_432

-- Define the geometric setup and variables involved
variables {Ω ω₁ ω₂ : circle} {A B X Y A₁ B₁ Z : Point}
variables (internally_tangent1 : internally_tangent ω₁ Ω A)
variables (internally_tangent2 : internally_tangent ω₂ Ω B)
variables (intersect_points : ω₁ ∩ ω₂ = {X, Y})
variables (AB_inter_A₁ : line_intersects_circle (line A B) ω₁ {A₁})
variables (AB_inter_B₁ : line_intersects_circle (line A B) ω₂ {B₁})
variables (circle_Z : exists_circle_tangent_to_three ω₁ ω₂ (line A₁ B₁) Z)

-- Prove the statement
theorem equal_angles_AXZ_BXZ :
  ∠AXZ = ∠BXZ :=
by 
  sorry

end equal_angles_AXZ_BXZ_l0_432


namespace exists_perfect_square_sum_l0_841

theorem exists_perfect_square_sum (n : ℕ) (h : n > 2) : ∃ m : ℕ, ∃ k : ℕ, n^2 + m^2 = k^2 :=
by
  sorry

end exists_perfect_square_sum_l0_841


namespace modular_arithmetic_problem_l0_784

theorem modular_arithmetic_problem : 
  ∃ (n : ℤ), 0 ≤ n ∧ n < 29 ∧ 5 * n ≡ 1 [MOD 29] ∧ ((2^n)^3 - 3) % 29 = 10 :=
by
  sorry

end modular_arithmetic_problem_l0_784


namespace sam_distance_proof_l0_240

-- Definitions of conditions
def marguerite_distance : ℝ := 150
def marguerite_time : ℝ := 3
def sam_time : ℝ := 4.5

-- Definition of Marguerite's speed
def marguerite_speed : ℝ := marguerite_distance / marguerite_time

-- Definition to prove that Sam's distance is 225 miles
theorem sam_distance_proof : sam_time * marguerite_speed = 225 := by
  -- No proof is needed, so we use 'sorry'
  sorry

end sam_distance_proof_l0_240


namespace monica_total_savings_l0_807

noncomputable def weekly_savings : ℕ := 15
noncomputable def weeks_to_fill_moneybox : ℕ := 60
noncomputable def num_repeats : ℕ := 5
noncomputable def total_savings (weekly_savings weeks_to_fill_moneybox num_repeats : ℕ) : ℕ :=
  (weekly_savings * weeks_to_fill_moneybox) * num_repeats

theorem monica_total_savings :
  total_savings 15 60 5 = 4500 := by
  sorry

end monica_total_savings_l0_807


namespace arithmetic_b_seq_general_b_seq_formula_sum_a_seq_formula_l0_118

-- Definitions and conditions
def a_seq : ℕ → ℕ
| 0       := 0
| 1       := 3
| (n + 2) := 3 * a_seq (n + 1) + 3^n

def b_seq (n : ℕ) : ℕ :=
  a_seq n / 3^n

def sum_a_seq (n : ℕ) : ℕ :=
  ∑ i in range n, a_seq (i + 1)

-- Statements
theorem arithmetic_b_seq : ∀ n : ℕ, b_seq (n + 1) - b_seq n = 1 / 3 :=
by
  sorry

theorem general_b_seq_formula : ∀ n : ℕ, b_seq n = (n + 2) / 3 :=
by
  sorry

theorem sum_a_seq_formula : ∀ n : ℕ, sum_a_seq n = ((n + 2) * 3^n) / 2 - ((3^n - 3) / 4) :=
by
  sorry

end arithmetic_b_seq_general_b_seq_formula_sum_a_seq_formula_l0_118


namespace train_pass_time_l0_31

-- Assuming conversion factor, length of the train, and speed in km/hr
def conversion_factor := 1000 / 3600
def train_length := 280
def speed_km_hr := 36

-- Defining speed in m/s
def speed_m_s := speed_km_hr * conversion_factor

-- Defining the time to pass a tree
def time_to_pass_tree := train_length / speed_m_s

-- Theorem statement
theorem train_pass_time : time_to_pass_tree = 28 := by
  sorry

end train_pass_time_l0_31


namespace find_a_l0_273

theorem find_a (a : ℝ) (h_pos : 0 < a) (h_neq_one : a ≠ 1) :
  let f := λ x : ℝ, a ^ x in
  (∀ x ∈ set.Icc (1 : ℝ) (2 : ℝ), f x = (if x = 2 then a^2 else if x = 1 then a else f x)) →
  (max (f 1) (f 2) - min (f 1) (f 2) = a / 2) →
  (a = 3 / 2 ∨ a = 1 / 2) :=
by
  sorry

end find_a_l0_273


namespace bus_cost_is_1_50_l0_350

variable (B : ℝ)
variable (C_bus : B)
variable (C_train : B + 6.85)
variable (combined_cost : C_bus + C_train = 9.85)

theorem bus_cost_is_1_50
  (C_bus : B)
  (C_train : B + 6.85)
  (combined_cost : C_bus + C_train = 9.85) :
  C_bus = 1.50 := by
  sorry

end bus_cost_is_1_50_l0_350


namespace pencil_color_change_l0_840

-- Problem statement: Given several children each with a pencil of one of three colors,
-- prove that it is possible to assign pencils initially such that, after some exchanges,
-- no child retains the same pencil color.

theorem pencil_color_change (n : ℕ) (pencils : Fin n → Fin 3) (exchange : Perm (Fin n)) :
  (∀ i, pencils (exchange i) ≠ pencils i) → ∃ initial_assignment : Fin n → Fin 3,
  ∀ i, initial_assignment (exchange i) ≠ initial_assignment i := 
sorry

end pencil_color_change_l0_840


namespace choose_non_werewolf_l0_969

theorem choose_non_werewolf (A B C : Prop) 
  (knight : Prop → Prop) (liar : Prop → Prop) (normal : Prop) -- Definitions of roles
  (A_is_knight : knight A) (B_is_liar : liar B) (C_is_normal : normal)
  (A_truths : ∀ p, knight A = p) (B_lies : ∀ p, liar B = ¬p) -- Behavioral traits
  (question : ∀ p, (knight A → p = tt) ∧ (liar B → p = ff) ∧ (normal → p = tt ∨ p = ff)) :
  (question (A ↔ ¬B → C) → B ≠ werewolf) ∧ (¬(question (A ↔ ¬B → C)) → C ≠ werewolf) :=
by
  sorry

end choose_non_werewolf_l0_969


namespace watch_loss_percentage_l0_34

theorem watch_loss_percentage :
  (CP = 280) →
  (SP' = CP + 0.04 * CP) →
  (SP + 140 = SP') →
  (L = ((CP - SP) / CP) * 100) →
  (L = 46) :=
by
  intros hCP hNewSP hOriginalSP hLossPercentage
  rw hCP at hNewSP
  rw hCP at hOriginalSP
  rw hNewSP at hOriginalSP
  rw hCP at hLossPercentage
  rw hOriginalSP at hLossPercentage
  sorry

end watch_loss_percentage_l0_34


namespace eliminate_alpha_l0_82

theorem eliminate_alpha (α x y : ℝ) (h1 : x = Real.tan α ^ 2) (h2 : y = Real.sin α ^ 2) : 
  x - y = x * y := 
by
  sorry

end eliminate_alpha_l0_82


namespace geometric_sum_omega_l0_63

noncomputable def omega : ℂ := complex.exp (2 * real.pi * complex.I / 19)

theorem geometric_sum_omega :
  (∑ k in finset.range 18, omega ^ (k + 1)) = -1 :=
by
  sorry

end geometric_sum_omega_l0_63


namespace rectangle_ratio_sum_eq_ten_l0_200

-- Define overall configuration of the rectangle.
structure Rectangle :=
  (A B C D E F P Q : Point)
  (AB BC : ℝ)
  (hAB : AB = 8)
  (hBC : BC = 4)
  (BE EF FC FD : ℝ)
  (hBE : BE = 1)
  (hEF : EF = 1)
  (hFC : FC = 1)
  (hFD : FD = 1)

-- Define the necessary calculations and properties.
def calculate_ratio (rect : Rectangle) : ℕ :=
  let BP := 5
  let PQ := 3
  let QD := 2
  BP + PQ + QD

-- Proof statement
theorem rectangle_ratio_sum_eq_ten : ∀ (rect : Rectangle), calculate_ratio(rect) = 10 :=
by
  intro rect
  sorry

end rectangle_ratio_sum_eq_ten_l0_200


namespace positive_difference_of_two_numbers_l0_905

theorem positive_difference_of_two_numbers 
  (a b : ℝ) 
  (h₁ : a + b = 10) 
  (h₂ : a^2 - b^2 = 40) : 
  |a - b| = 4 := 
sorry

end positive_difference_of_two_numbers_l0_905


namespace no_whole_number_solution_l0_67

theorem no_whole_number_solution (n : ℕ) (m : ℤ) (h1 : m = 1) :
  ¬ (n ^ (n / 2 + m) = 12) :=
by
  sorry  -- Proof would go here.

end no_whole_number_solution_l0_67


namespace train_passes_man_in_10_seconds_l0_30

def length_of_train : ℝ := 150
def train_speed : ℝ := 62 * 1000 / 3600 -- converting from km/h to m/s
def man_speed : ℝ := 8 * 1000 / 3600 -- converting from km/h to m/s

def relative_speed : ℝ := train_speed - man_speed

theorem train_passes_man_in_10_seconds :
  length_of_train / relative_speed = 10 :=
by
  sorry

end train_passes_man_in_10_seconds_l0_30


namespace cos_135_eq_neg_inv_sqrt_2_l0_547

theorem cos_135_eq_neg_inv_sqrt_2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 :=
sorry

end cos_135_eq_neg_inv_sqrt_2_l0_547


namespace john_sublets_to_3_people_l0_214

def monthly_income (n : ℕ) : ℕ := 400 * n
def monthly_cost : ℕ := 900
def annual_profit (n : ℕ) : ℕ := 12 * (monthly_income n - monthly_cost)

theorem john_sublets_to_3_people
  (h1 : forall n : ℕ, monthly_income n - monthly_cost > 0)
  (h2 : annual_profit 3 = 3600) :
  3 = 3 := by
  sorry

end john_sublets_to_3_people_l0_214


namespace problem_shiny_pennies_l0_9

theorem problem_shiny_pennies :
  let shiny_count := 5 in
  let dull_count := 3 in
  let total_count := shiny_count + dull_count in
  (∃ a b : ℕ, (Nat.gcd a b = 1) ∧ (a + b = 87) ∧ (a / b) = 31 / 56) := by
sorry

end problem_shiny_pennies_l0_9


namespace cos_135_eq_neg_inv_sqrt_2_l0_558

theorem cos_135_eq_neg_inv_sqrt_2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 :=
sorry

end cos_135_eq_neg_inv_sqrt_2_l0_558


namespace cos_135_eq_neg_sqrt2_div_2_l0_626

theorem cos_135_eq_neg_sqrt2_div_2 : 
  Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_626


namespace find_g_l0_736

open Real

def even (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def odd (g : ℝ → ℝ) := ∀ x, g (-x) = -g x

theorem find_g 
  (f g : ℝ → ℝ) 
  (hf : even f) 
  (hg : odd g)
  (h : ∀ x, f x + g x = exp x) :
  ∀ x, g x = exp x - exp (-x) :=
by
  sorry

end find_g_l0_736


namespace cos_135_eq_neg_inv_sqrt2_l0_439

theorem cos_135_eq_neg_inv_sqrt2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt2_l0_439


namespace min_distance_proof_l0_689

noncomputable def min_distance_squared : ℝ :=
  let a : ℝ := sorry -- some positive real number
  let b : ℝ := ln a / a
  let d : ℝ := sorry -- another real number
  let c : ℝ := d - 2
  in (a - c)^2 + (b - d)^2

theorem min_distance_proof : min_distance_squared = 9 / 2 :=
by
  sorry

end min_distance_proof_l0_689


namespace downstream_distance_l0_771

-- Constants and conditions
def t_downstream := 4 -- hours
def t_upstream := 6 -- hours
def c := 1.0 -- mph

-- Question: Prove the downstream distance is 24 miles
theorem downstream_distance : ∃ v d : ℝ, d = (v + c) * t_downstream ∧ d = 24 := by
  sorry

end downstream_distance_l0_771


namespace length_of_UB_l0_431

theorem length_of_UB (C : ℝ) (r : ℝ) (U A B : ℝ × ℝ) (h1 : C = 16 * π) (h2 : 2 * π * r = C) (h3 : dist U A = r) (h4 : dist U B = r)
  (h5 : dist A B = 2 * r) (h6 : ∠ U A B = π / 4) : dist U B = 8 :=
by
  -- proof to be filled in
  sorry

end length_of_UB_l0_431


namespace trig_identity_proof_l0_425

theorem trig_identity_proof :
  2 * (1 / 2) + (Real.sqrt 3 / 2) * Real.sqrt 3 = 5 / 2 :=
by
  sorry

end trig_identity_proof_l0_425


namespace find_k_values_l0_155

theorem find_k_values (k : ℝ) :
  ∃ a b c : ℝ, 
    a^2 + b^2 = 5 ∧
    2a + c = -2 * (k - 1) ∧
    c * a = 2 ∧
    c = 1 - k ∧
    (x^3 + 2*(k - 1)*x^2 + 9*x + 5*(k - 1) = 0) := sorry

end find_k_values_l0_155


namespace probability_of_5_out_of_7_odd_rolls_l0_333

theorem probability_of_5_out_of_7_odd_rolls :
  let prob_odd := (1 / 2 : ℝ)
  let prob_even := (1 / 2 : ℝ)
  let success_count := 5
  let trial_count := 7
  let combination := nat.choose trial_count success_count
  let prob := combination * prob_odd ^ success_count * prob_even ^ (trial_count - success_count)
  in prob = 21 / 128 := by
  sorry

end probability_of_5_out_of_7_odd_rolls_l0_333


namespace min_value_expression_l0_790

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  8 * a^3 + 6 * b^3 + 27 * c^3 + 9 / (8 * a * b * c) ≥ 18 :=
by
  sorry

end min_value_expression_l0_790


namespace sum_of_roots_of_quadratic_l0_727

theorem sum_of_roots_of_quadratic (m n : ℝ) (h1 : m = 2 * n) (h2 : ∀ x : ℝ, x ^ 2 + m * x + n = 0) :
    m + n = 3 / 2 :=
sorry

end sum_of_roots_of_quadratic_l0_727


namespace cosine_135_eq_neg_sqrt_2_div_2_l0_537

theorem cosine_135_eq_neg_sqrt_2_div_2 (cos180 : Real := -1) (cos90 : Real := 0) (cos45 : Real := Real.sqrt 2 / 2)
    (theta1 : Real := 180) (theta2 : Real := 45) (theta3 : Real := 135) :
    theta3 = theta1 - theta2 ∧ 
    Real.cos theta1 = cos180 ∧ 
    Real.cos theta2 = cos45 → 
    Real.cos theta3 = -cos45 := by
  sorry

end cosine_135_eq_neg_sqrt_2_div_2_l0_537


namespace uncle_bradley_money_l0_326

-- Definitions of the variables and conditions
variables (F H M : ℝ)
variables (h1 : F + H = 13)
variables (h2 : 50 * F = (3 / 10) * M)
variables (h3 : 100 * H = (7 / 10) * M)

-- The theorem statement
theorem uncle_bradley_money : M = 1300 :=
by
  sorry

end uncle_bradley_money_l0_326


namespace cos_135_eq_neg_sqrt_two_div_two_l0_449

theorem cos_135_eq_neg_sqrt_two_div_two : real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt_two_div_two_l0_449


namespace positive_difference_of_numbers_l0_924

theorem positive_difference_of_numbers
  (a b : ℝ)
  (h1 : a + b = 10)
  (h2 : a^2 - b^2 = 40) :
  |a - b| = 4 :=
by 
  sorry

end positive_difference_of_numbers_l0_924


namespace minimal_positive_period_of_f_range_of_g_in_interval_l0_147

noncomputable def f (x : ℝ) : ℝ := 2 * sin x ^ 2 + 2 * sqrt 3 * sin x * sin (x + π / 2)

theorem minimal_positive_period_of_f : 
  (∀ x, f(x + π) = f x) ∧ (∀ p > 0, (∀ x, f(x + p) = f x) → (p ≥ π)) := 
sorry

noncomputable def g (x : ℝ) : ℝ := 2 * sin (2 * x + π / 6) + 3

theorem range_of_g_in_interval : 
  ∀ x ∈ set.Icc (0 : ℝ) (7 * π / 12 : ℝ), 
    3 - sqrt 3 ≤ g x ∧ g x ≤ 5 := 
sorry

end minimal_positive_period_of_f_range_of_g_in_interval_l0_147


namespace distribute_slips_correctly_l0_43

noncomputable def cups_distribution_problem :
  Prop :=
  ∃ (P Q R S T : set ℚ),
  P.sum + Q.sum + R.sum + S.sum + T.sum = 40 ∧
  (∀ s ∈ [P, Q, R, S, T], (∃ n ∈ ℤ, s.sum = n)) ∧
  (list.sum [P.sum, Q.sum, R.sum, S.sum, T.sum] = [9, 10, 11, 12, 13]) ∧
  (\ exists x ∈ T, x = 4) ∧
  (\ exists x ∈ Q, x = 1.5 ∧ x ∉ P ∧ x ∉ R ∧ x ∉ S ∧ x ∉ T ∧ R 3.5)

theorem distribute_slips_correctly : cups_distribution_problem :=
sorry

end distribute_slips_correctly_l0_43


namespace cos_135_eq_neg_inv_sqrt_2_l0_553

theorem cos_135_eq_neg_inv_sqrt_2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 :=
sorry

end cos_135_eq_neg_inv_sqrt_2_l0_553


namespace cos_135_eq_neg_sqrt2_div_2_l0_596

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  -- Conditions incorporated directly into the requirement due to the nature of trigonometric evaluations.
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_596


namespace monotonic_intervals_range_of_k_f_le_0_harmonic_gt_ln_l0_361

noncomputable def f (x k : ℝ) : ℝ := (x - k) / (x + 1)

-- Prove monotonic intervals
theorem monotonic_intervals (k : ℝ) :
  (k ≤ 0 → ∀ x : ℝ, 0 < x → f x k = (x - k) / (x + 1) → by (sorry)) ∧
  (k > 0 → (∀ x : ℝ, 0 < x → f x k = (x - k) / (x + 1) → by (sorry)) ∧
    ∀ x : ℝ, f x k = (x - k) / (x + 1) → by (sorry))) :=
sorry

-- Prove range of k for f(x) <= 0
theorem range_of_k_f_le_0 (k : ℝ) :
  (∀ x : ℝ, f x k <= 0) → 1 ≤ k :=
sorry

-- Prove harmonic series minus log inequality
theorem harmonic_gt_ln (n : ℕ) (h : 1 < n) :
  1 + (1 / 2) + (1 / 3) + ... + (1 / n) > Real.log n :=
sorry

end monotonic_intervals_range_of_k_f_le_0_harmonic_gt_ln_l0_361


namespace minimum_value_of_f_l0_88

noncomputable def f (x : ℝ) : ℝ := (Real.cos (2 * x) + 2 * Real.sin x)

theorem minimum_value_of_f : ∃ x : ℝ, (∀ y : ℝ, f x ≤ f y) ∧ f x = -3 := 
  sorry

end minimum_value_of_f_l0_88


namespace probability_of_odd_numbers_exactly_five_times_in_seven_rolls_l0_328

theorem probability_of_odd_numbers_exactly_five_times_in_seven_rolls :
  (nat.choose 7 5 * (1/2)^5 * (1/2)^2) = (21 / 128) := by
  sorry

end probability_of_odd_numbers_exactly_five_times_in_seven_rolls_l0_328


namespace concrete_needed_l0_377

-- Definitions from the problem conditions
def width_rect := (4 / 3 : ℚ)  -- in yards
def length_rect := (100 / 3 : ℚ)  -- in yards
def thickness_rect := (1 / 9 : ℚ)  -- in yards

def height_trap := (100 / 3 : ℚ)  -- in yards
def bottom_base_trap := (4 / 3 : ℚ)  -- in yards
def top_base_trap := (8 / 3 : ℚ)  -- in yards
def thickness_trap := (1 / 9 : ℚ)  -- in yards

-- Mathematical equivalence of the problem: verify that the total volume equals 12 cubic yards
theorem concrete_needed :
  let V_rectangular := width_rect * length_rect * thickness_rect,
      area_trap := (1 / 2 : ℚ) * (top_base_trap + bottom_base_trap) * height_trap,
      V_trapezoidal := area_trap * thickness_trap,
      V_total := V_rectangular + V_trapezoidal
  in V_total.ceil = 12 := by
  sorry

end concrete_needed_l0_377


namespace distance_PQ_geq_5_l0_818

open Set

variables {P Q: Point} {A O B : Point} {OA OB: Line}
          (angleAOB : Angle A O B) (distancePOA : ℝ) (distancePOB : ℝ)

def lies_on_angle_bisector (P : Point) (angleAOB : Angle A O B) :=
  distance P OA = distance P OB

def distance_to_line (P : Point) (L : Line) : ℝ := sorry -- Assume there's a function that computes this

def on_side (Q : Point) (L : Line) : Prop := sorry -- Assume a predicate for checking a point is on a line side

theorem distance_PQ_geq_5 (angleAOB_bisector : lies_on_angle_bisector P angleAOB)
                          (distancePOA_eq : distance_to_line P OA = 5)
                          (Q_on_OB : on_side Q OB) :
  distance P Q ≥ 5 :=
sorry

end distance_PQ_geq_5_l0_818


namespace cos_135_eq_neg_inv_sqrt2_l0_436

theorem cos_135_eq_neg_inv_sqrt2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt2_l0_436


namespace find_m_l0_162

theorem find_m 
  (m : ℝ)
  (a b : ℝ × ℝ)
  (h1 : a = (1, m))
  (h2 : b = (3, -2))
  (h3 : (1 + 3, m - 2) = (4, m - 2))
  (h4 : (4, m - 2) ⊥ b) :
  m = 8 := by
sorry

end find_m_l0_162


namespace gloves_selection_l0_946

-- Definitions based on the conditions
def num_pairs : ℕ := 5
def selected_gloves : ℕ := 3

-- The main statement
theorem gloves_selection :
  ∃ (n : ℕ), n = 80 :=
by
  use 80
  sorry

end gloves_selection_l0_946


namespace smallest_fourth_lucky_number_l0_390

theorem smallest_fourth_lucky_number :
  ∃ n : ℕ, (∃ a b : ℕ, n = 10 * a + b ∧
             1 ≤ a ∧ a ≤ 9 ∧
             0 ≤ b ∧ b ≤ 9 ∧
             142 + n = 5 * (25 + a + b)) ∧
           n = 38 :=
begin
  -- existential quantifier indicating the smallest n
  use 38,
  -- existential quantifiers indicating the digits a and b
  use 3,
  use 8,
  -- ensuring n = 10a + b
  split,
  { refl }, -- 38 = 10 * 3 + 8
  split,
  { split; norm_num }, -- 3 is a valid digit
  split,
  { split; norm_num }, -- 8 is a valid digit
  -- proving the sum condition
  calc
    142 + 38 = 180 : by norm_num
         ... = 5 * (25 + 3 + 8) : by norm_num,
end

end smallest_fourth_lucky_number_l0_390


namespace find_angle_A_find_length_AM_l0_745

variables {A B C : ℝ}
variables {a b c AM : ℝ}

/-- The conditions of the triangle ABC -/
def conditions (a b c A B C : ℝ) : Prop :=
  (a ≠ 0) ∧ 
  (b ≠ 0) ∧ 
  (c ≠ 0) ∧
  (A > 0) ∧ 
  (A < π) ∧ 
  (B > 0) ∧ 
  (B < π) ∧ 
  (C > 0) ∧ 
  (C < π) ∧ 
  (a * sin B = b * sin A) ∧ 
  (a^2 = b^2 + c^2 - 2 * b * c * cos A) ∧
  (b^2 = a^2 + c^2 - 2 * a * c * cos B) ∧
  (c^2 = a^2 + b^2 - 2 * a * b * cos C) ∧
  ((2 * b - sqrt 3 * c) / (sqrt 3 * a) = (cos C) / (cos A))
  
/-- The conditions for the second part of the proof -/
noncomputable def conditions_part2 (a b c A B C AM : ℝ) : Prop :=
  conditions a b c A B C ∧
  (B = π / 6) ∧
  (1 / 2 * a^2 * sin (2 * π / 3) = 4 * sqrt 3) ∧
  (a = 4) ∧ 
  (c = 4 * sqrt 3) ∧ 
  (AM = sqrt (a^2 + (c / 2)^2 - 2 * a * (c / 2) * cos A))

/-- Statement for the first part proof -/
theorem find_angle_A (h : conditions a b c A B C) : A = π / 6 :=
sorry

/-- Statement for the second part proof -/
theorem find_length_AM (h : conditions_part2 a b c A B C AM) : AM = 2 * sqrt 7 :=
sorry

end find_angle_A_find_length_AM_l0_745


namespace sum_abcd_value_l0_222

theorem sum_abcd_value (a b c d : ℚ) :
  (2 * a + 3 = 2 * b + 5) ∧ 
  (2 * b + 5 = 2 * c + 7) ∧ 
  (2 * c + 7 = 2 * d + 9) ∧ 
  (2 * d + 9 = 2 * (a + b + c + d) + 13) → 
  a + b + c + d = -14 / 3 := 
by
  sorry

end sum_abcd_value_l0_222


namespace find_x_of_parallel_vectors_l0_720

section vector_parallel

variables (x : ℝ)
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (2 * x, -3)
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem find_x_of_parallel_vectors (h : are_parallel vec_a vec_b) : x = -3 / 4 :=
by
  sorry

end vector_parallel

end find_x_of_parallel_vectors_l0_720


namespace total_payment_mr_benson_made_l0_376

noncomputable def general_admission_ticket_cost : ℝ := 40
noncomputable def num_general_admission_tickets : ℕ := 10
noncomputable def num_vip_tickets : ℕ := 3
noncomputable def num_premium_tickets : ℕ := 2
noncomputable def vip_ticket_rate_increase : ℝ := 0.20
noncomputable def premium_ticket_rate_increase : ℝ := 0.50
noncomputable def discount_rate : ℝ := 0.05
noncomputable def threshold_tickets : ℕ := 10

noncomputable def vip_ticket_cost : ℝ := general_admission_ticket_cost * (1 + vip_ticket_rate_increase)
noncomputable def premium_ticket_cost : ℝ := general_admission_ticket_cost * (1 + premium_ticket_rate_increase)

noncomputable def total_general_admission_cost : ℝ := num_general_admission_tickets * general_admission_ticket_cost
noncomputable def total_vip_cost : ℝ := num_vip_tickets * vip_ticket_cost
noncomputable def total_premium_cost : ℝ := num_premium_tickets * premium_ticket_cost

noncomputable def total_tickets : ℕ := num_general_admission_tickets + num_vip_tickets + num_premium_tickets
noncomputable def tickets_exceeding_threshold : ℕ := if total_tickets > threshold_tickets then total_tickets - threshold_tickets else 0

noncomputable def discounted_vip_cost : ℝ := vip_ticket_cost * (1 - discount_rate)
noncomputable def discounted_premium_cost : ℝ := premium_ticket_cost * (1 - discount_rate)

noncomputable def total_discounted_vip_cost : ℝ :=  num_vip_tickets * discounted_vip_cost
noncomputable def total_discounted_premium_cost : ℝ := num_premium_tickets * discounted_premium_cost

noncomputable def total_cost_with_discounts : ℝ := total_general_admission_cost + total_discounted_vip_cost + total_discounted_premium_cost

theorem total_payment_mr_benson_made : total_cost_with_discounts = 650.80 :=
by
  -- Proof is omitted
  sorry

end total_payment_mr_benson_made_l0_376


namespace cos_135_eq_neg_inv_sqrt_2_l0_577

theorem cos_135_eq_neg_inv_sqrt_2 :
  cos (135 * Real.pi / 180) = - (1 / Real.sqrt 2) :=
by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_577


namespace find_second_number_l0_7

theorem find_second_number (x : ℝ) : 217 + x + 0.217 + 2.0017 = 221.2357 → x = 2.017 :=
by
  sorry

end find_second_number_l0_7


namespace distance_PQ_geq_5_l0_819

open Set

variables {P Q: Point} {A O B : Point} {OA OB: Line}
          (angleAOB : Angle A O B) (distancePOA : ℝ) (distancePOB : ℝ)

def lies_on_angle_bisector (P : Point) (angleAOB : Angle A O B) :=
  distance P OA = distance P OB

def distance_to_line (P : Point) (L : Line) : ℝ := sorry -- Assume there's a function that computes this

def on_side (Q : Point) (L : Line) : Prop := sorry -- Assume a predicate for checking a point is on a line side

theorem distance_PQ_geq_5 (angleAOB_bisector : lies_on_angle_bisector P angleAOB)
                          (distancePOA_eq : distance_to_line P OA = 5)
                          (Q_on_OB : on_side Q OB) :
  distance P Q ≥ 5 :=
sorry

end distance_PQ_geq_5_l0_819


namespace cos_135_eq_neg_sqrt2_div_2_l0_464

theorem cos_135_eq_neg_sqrt2_div_2 : cos (135 * real.pi / 180) = - (real.sqrt 2 / 2) :=
by
  have h1 : 135 = 180 - 45 := by norm_num
  have h2 : cos (180 * real.pi / 180 - 45 * real.pi / 180) = - cos (45 * real.pi / 180) :=
    by rw [cos_sub, cos_pi, cos_pi_div_four, sin_pi, sin_pi_div_four]; norm_num
  have h3 : cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by norm_num
  rw [h1, ←h2, h3]
  norm_num
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_464


namespace probability_of_5_out_of_7_odd_rolls_l0_331

theorem probability_of_5_out_of_7_odd_rolls :
  let prob_odd := (1 / 2 : ℝ)
  let prob_even := (1 / 2 : ℝ)
  let success_count := 5
  let trial_count := 7
  let combination := nat.choose trial_count success_count
  let prob := combination * prob_odd ^ success_count * prob_even ^ (trial_count - success_count)
  in prob = 21 / 128 := by
  sorry

end probability_of_5_out_of_7_odd_rolls_l0_331


namespace borgnine_lizards_l0_53

theorem borgnine_lizards (chimps lions tarantulas total_legs : ℕ) (legs_per_chimp legs_per_lion legs_per_tarantula legs_per_lizard lizards : ℕ)
  (H_chimps : chimps = 12)
  (H_lions : lions = 8)
  (H_tarantulas : tarantulas = 125)
  (H_total_legs : total_legs = 1100)
  (H_legs_per_chimp : legs_per_chimp = 4)
  (H_legs_per_lion : legs_per_lion = 4)
  (H_legs_per_tarantula : legs_per_tarantula = 8)
  (H_legs_per_lizard : legs_per_lizard = 4)
  (H_seen_legs : total_legs = (chimps * legs_per_chimp) + (lions * legs_per_lion) + (tarantulas * legs_per_tarantula) + (lizards * legs_per_lizard)) :
  lizards = 5 := 
by
  sorry

end borgnine_lizards_l0_53


namespace tan_product_seven_eq_sqrt_seven_l0_423

noncomputable def tan_product_seven (x : ℝ) : ℝ :=
  Real.tan (π / 7) * Real.tan (2 * π / 7) * Real.tan (3 * π / 7)

theorem tan_product_seven_eq_sqrt_seven :
  tan_product_seven = sqrt 7 :=
  sorry

end tan_product_seven_eq_sqrt_seven_l0_423


namespace pipe_A_fill_time_l0_349

theorem pipe_A_fill_time
  (fill_time_all : ℝ)
  (pipe_B_rate_factor : ℝ)
  (pipe_C_rate_factor : ℝ)
  (combined_fill_time : ℝ)
  (A_rate : ℝ) :
  (pipe_B_rate_factor = 2) ->
  (pipe_C_rate_factor = 4) ->
  (fill_time_all = 5) ->
  (A_rate = (1 / (fill_time_all * (1 + pipe_B_rate_factor + pipe_C_rate_factor)))) ->
  combined_fill_time = 1 / A_rate :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  simp at h4
  have A_rate_is : A_rate = 1 / 35 := h4
  rw A_rate_is
  simp
  sorry

end pipe_A_fill_time_l0_349


namespace proof_cos_135_degree_l0_562

noncomputable def cos_135_degree_is_negative_sqrt2_over_2 : Prop :=
  real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2

theorem proof_cos_135_degree : cos_135_degree_is_negative_sqrt2_over_2 :=
by
  sorry

end proof_cos_135_degree_l0_562


namespace find_x_l0_97

def determinant (a b c d : ℚ) : ℚ := a * d - b * c

theorem find_x (x : ℚ) (h : determinant (2 * x) (-4) x 1 = 18) : x = 3 :=
  sorry

end find_x_l0_97


namespace cos_135_eq_neg_sqrt2_div_2_l0_600

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  -- Conditions incorporated directly into the requirement due to the nature of trigonometric evaluations.
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_600


namespace melanie_total_dimes_l0_805

theorem melanie_total_dimes (d_1 d_2 d_3 : ℕ) (h₁ : d_1 = 19) (h₂ : d_2 = 39) (h₃ : d_3 = 25) : d_1 + d_2 + d_3 = 83 := by
  sorry

end melanie_total_dimes_l0_805


namespace find_largest_p_plus_q_l0_954

noncomputable def largest_possible_p_plus_q : ℝ :=
  let B := (12 : ℝ, 19 : ℝ)
  let C := (23 : ℝ, 20 : ℝ)
  let area_ABC : ℝ := 70
  let slope_median : ℝ := -5 in
  47

theorem find_largest_p_plus_q (p q : ℝ) (hA : (p, q) = A) (h_area : 2 * area_ABC = abs (p * (B.2 - C.2) + B.1 * (C.2 - q) + C.1 * (q - B.2)))
(h_slope : q = -5 * p + 107) : p + q = largest_possible_p_plus_q := 
  sorry

end find_largest_p_plus_q_l0_954


namespace num_solutions_f_f_x_eq_10_l0_732
noncomputable def f : ℝ → ℝ :=
  λ x, if x ≥ -3 then x^2 - 9 else x + 4

theorem num_solutions_f_f_x_eq_10 : 
  {x : ℝ | f (f x) = 10}.finite.to_finset.card = 2 :=
by
  sorry

end num_solutions_f_f_x_eq_10_l0_732


namespace positive_difference_of_two_numbers_l0_916

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l0_916


namespace b_le_1_of_condition_l0_707

noncomputable def f (x : ℝ) : ℝ := |Real.arctan (x - 1)|

theorem b_le_1_of_condition (a b x1 x2 : ℝ)
  (h1 : a ≤ x1) (h2 : x1 ≤ b) (h3 : a ≤ x2) (h4 : x2 ≤ b)
  (hx1x2 : x1 < x2) (hfx : f x1 ≥ f x2) : b ≤ 1 :=
begin
  sorry
end

end b_le_1_of_condition_l0_707


namespace number_of_mowers_l0_29

noncomputable section

def area_larger_meadow (A : ℝ) : ℝ := 2 * A

def team_half_day_work (K a : ℝ) : ℝ := (K * a) / 2

def team_remaining_larger_meadow (K a : ℝ) : ℝ := (K * a) / 2

def half_team_half_day_work (K a : ℝ) : ℝ := (K * a) / 4

def larger_meadow_area_leq_sum (K a A : ℝ) : Prop :=
  team_half_day_work K a + team_remaining_larger_meadow K a = 2 * A

def smaller_meadow_area_left (K a A : ℝ) : ℝ :=
  A - half_team_half_day_work K a

def one_mower_one_day_work_rate (K a : ℝ) : ℝ := (K * a) / 4

def eq_total_mowed_by_team (K a A : ℝ) : Prop :=
  larger_meadow_area_leq_sum K a A ∧ smaller_meadow_area_left K a A = (K * a) / 4

theorem number_of_mowers
  (K a A b : ℝ)
  (h1 : larger_meadow_area_leq_sum K a A)
  (h2 : smaller_meadow_area_left K a A = one_mower_one_day_work_rate K a)
  (h3 : one_mower_one_day_work_rate K a = b)
  (h4 : K * a = 2 * A)
  (h5 : 2 * A = 4 * b)
  : K = 8 :=
  sorry

end number_of_mowers_l0_29


namespace sum_inequality_l0_822

theorem sum_inequality {n k : ℕ} (hn : n > 1) (hk : k > 1) :
  (∑ j in Finset.range (n ^ k + 1) \ Finset.range 2, (1 : ℚ) / j) >
  k * (∑ j in Finset.range (n + 1) \ Finset.range 2, (1 : ℚ) / j) :=
sorry

end sum_inequality_l0_822


namespace cos_135_eq_neg_sqrt_two_div_two_l0_450

theorem cos_135_eq_neg_sqrt_two_div_two : real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt_two_div_two_l0_450


namespace carpet_coverage_percentage_l0_989

variable (l w : ℝ) (floor_area carpet_area : ℝ)

theorem carpet_coverage_percentage 
  (h_carpet_area: carpet_area = l * w) 
  (h_floor_area: floor_area = 180) 
  (hl : l = 4) 
  (hw : w = 9) : 
  carpet_area / floor_area * 100 = 20 := by
  sorry

end carpet_coverage_percentage_l0_989


namespace sum_of_coefficients_is_256_l0_168

theorem sum_of_coefficients_is_256 :
  ∀ (a a1 a2 a3 a4 a5 a6 a7 a8 : ℤ), 
  ((x : ℤ) - a)^8 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7 + a8 * x^8 → 
  a5 = 56 →
  a + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 256 :=
by
  intros
  sorry

end sum_of_coefficients_is_256_l0_168


namespace tan_double_angle_l0_667

theorem tan_double_angle (α : ℝ) (h1 : sin (2 * α) = -sin α) (h2 : α ∈ Ioc (π / 2) π) : tan (2 * α) = sqrt 3 :=
by
  sorry

end tan_double_angle_l0_667


namespace cos_135_eq_neg_sqrt2_div_2_l0_466

theorem cos_135_eq_neg_sqrt2_div_2 : cos (135 * real.pi / 180) = - (real.sqrt 2 / 2) :=
by
  have h1 : 135 = 180 - 45 := by norm_num
  have h2 : cos (180 * real.pi / 180 - 45 * real.pi / 180) = - cos (45 * real.pi / 180) :=
    by rw [cos_sub, cos_pi, cos_pi_div_four, sin_pi, sin_pi_div_four]; norm_num
  have h3 : cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by norm_num
  rw [h1, ←h2, h3]
  norm_num
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_466


namespace parabola_intersects_at_point_l0_72

theorem parabola_intersects_at_point (p q : ℝ) (h : p + q = 2002) :
  (1 : ℝ) ^ 2 + p * 1 + q = 2003 :=
by {
  -- We express \( y(1) \) given the quadratic function \( y = x^2 + px + q \)
  calc
    (1 : ℝ) ^ 2 + p * 1 + q = (1 + p + q) : by simp
    ... = 1 + 2002 : by rw [h]
    ... = 2003 : by norm_num
}

end parabola_intersects_at_point_l0_72


namespace number_made_l0_8

theorem number_made (x y : ℕ) (h1 : x + y = 24) (h2 : x = 11) : 7 * x + 5 * y = 142 := by
  sorry

end number_made_l0_8


namespace find_Spring_Earnings_l0_641

variables (Summer_Earnings Spent_on_Supplies End_Amount Spring_Earnings : ℕ)

-- Given conditions:
def condition1 := Summer_Earnings = 27
def condition2 := Spent_on_Supplies = 5
def condition3 := End_Amount = 24

-- Proof Goal:
theorem find_Spring_Earnings (h1 : condition1) (h2 : condition2) (h3 : condition3) : Spring_Earnings = 2 :=
by
  sorry

end find_Spring_Earnings_l0_641


namespace percent_increase_additive_l0_991

theorem percent_increase_additive (old_value new_value : ℕ) (H_old : old_value = 45) (H_new : new_value = 60) :
  ((new_value - old_value) / old_value) * 100 = 33.33 := 
by 
  sorry

end percent_increase_additive_l0_991


namespace math_problem_l0_141

variable {a n : ℕ}
variable {q : ℝ}
variable (a ≠ 0) (n > 0) (q > 1)
variable (Sn : ℕ → ℝ)

lemma prop1 : (∑ i in range (n + 1), (3 * i^2 - i + 1)) ≠ 3 * n^2 - n + 1 := sorry

lemma prop2 : (∀ n, 0 < (q:ℝ) ^ n) → (∀ n, (q:ℝ) ^ (n + 1) > q ^ n) := sorry

lemma prop3 : (2 : ℝ) ≠ 1 → (∑ i in range (2 + 1), a^i) ≠ (1 - a^2) / (1 - a) := sorry

lemma prop4 : (Sn 9 < 0) ∧ (Sn 10 > 0) → 
  (∀ n, Sn n = (n * (Sn 9 + Sn 10) / 2)) → (∀ k, 0 ≤ k → Sn k < (Sn 5))
  := sorry

theorem math_problem :
  (prop2 q a q > 1) ∧ (prop4 Sn) :=
by
  exact ⟨sorry, sorry⟩

end math_problem_l0_141


namespace vessel_capacity_at_least_eight_l0_33

theorem vessel_capacity_at_least_eight :
    ∀ (A B : ℝ), A = 0.7 ∧ B = 3 →
    ∀ (V : ℝ), V = A + B ∧ V = 3.7 →
    ∀ (total_volume : ℝ), total_volume = 8 →
    ∀ (new_concentration : ℝ), new_concentration = 0.37 →
    ∀ (required_alcohol : ℝ), required_alcohol = total_volume * new_concentration →
    required_alcohol = 2.96 →
    ∀ (vessel_capacity : ℝ), vessel_capacity = total_volume →
    vessel_capacity ≥ 8 :=
begin
    -- proof is not required, thus skipped
    sorry
end

end vessel_capacity_at_least_eight_l0_33


namespace max_pyramid_vertices_on_cube_l0_972

def PyramidOnCube (V : Type) [Vertex V] (faces : List (Set V)) : Prop :=
  -- All vertices of the pyramid lie on the faces of a cube but not on its edges.
  (∀ (v : V), ∃ (f : Set V), v ∈ f ∧ v ∉ edges) ∧
  -- Each face has at least one vertex.
  (∀ (f : Set V), f ∈ faces → ∃ (v : V), v ∈ f)

theorem max_pyramid_vertices_on_cube (V : Type) [Vertex V] (faces : List (Set V)) :
  PyramidOnCube V faces → faces.card = 13 := 
sorry

end max_pyramid_vertices_on_cube_l0_972


namespace cos_135_eq_neg_sqrt2_div_2_l0_623

theorem cos_135_eq_neg_sqrt2_div_2 : 
  Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_623


namespace learning_hours_difference_l0_643

/-- Define the hours Ryan spends on each language. -/
def hours_learned (lang : String) : ℝ :=
  if lang = "English" then 2 else
  if lang = "Chinese" then 5 else
  if lang = "Spanish" then 4 else
  if lang = "French" then 3 else
  if lang = "German" then 1.5 else 0

/-- Prove that Ryan spends 2.5 more hours learning Chinese and French combined
    than he does learning German and Spanish combined. -/
theorem learning_hours_difference :
  hours_learned "Chinese" + hours_learned "French" - (hours_learned "German" + hours_learned "Spanish") = 2.5 :=
by
  sorry

end learning_hours_difference_l0_643


namespace sum_of_altitudes_l0_631

theorem sum_of_altitudes (x y : ℝ) (hline : 10 * x + 8 * y = 80):
  let A := 1 / 2 * 8 * 10
  let hypotenuse := Real.sqrt (8 ^ 2 + 10 ^ 2)
  let third_altitude := 80 / hypotenuse
  let sum_altitudes := 8 + 10 + third_altitude
  sum_altitudes = 18 + 40 / Real.sqrt 41 := by
  sorry

end sum_of_altitudes_l0_631


namespace solve_and_sum_solutions_l0_847

theorem solve_and_sum_solutions :
  (∀ x : ℝ, x^2 > 9 → 
  (∃ S : ℝ, (∀ x : ℝ, (x / 3 + x / Real.sqrt (x^2 - 9) = 35 / 12) → S = 8.75))) :=
sorry

end solve_and_sum_solutions_l0_847


namespace calculate_total_money_l0_380

def gold_coin_value := 80
def silver_coin_value := 45
def bronze_coin_value := 25
def titanium_coin_value := 10
def banknote_value := 50
def coupon_value := 10
def voucher_value := 20

def gold_coins := 7
def silver_coins := 9
def bronze_coins := 12
def titanium_coins := 5
def banknotes := 3
def coupons := 6
def vouchers := 4

def total_gold_value := gold_coins * gold_coin_value
def total_silver_value := silver_coins * silver_coin_value
def total_bronze_value := bronze_coins * bronze_coin_value
def total_titanium_value := titanium_coins * titanium_coin_value
def total_banknote_value := banknotes * banknote_value
def total_coupon_value := coupons * coupon_value
def total_voucher_value := vouchers * voucher_value

def total_without_certificate := total_gold_value + total_silver_value + total_bronze_value + total_titanium_value + total_banknote_value + total_coupon_value + total_voucher_value

def total_gold_and_silver_value := total_gold_value + total_silver_value

def increased_value := (5/100) * total_gold_and_silver_value

def total_with_certificate := total_without_certificate + increased_value

theorem calculate_total_money : total_with_certificate = 1653.25 := by
  sorry

end calculate_total_money_l0_380


namespace suff_but_not_necessary_not_necessary_sufficient_but_not_necessary_condition_l0_792

theorem suff_but_not_necessary {x : ℝ} (h : x > 1) : x^2 + x - 2 > 0 :=
by {
  -- proof step here
  sorry
}

theorem not_necessary {x : ℝ} (h : x^2 + x - 2 > 0) : (x > 1) ∨ (x < -2) :=
by {
  -- proof step here
  sorry
}

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 1 → x^2 + x - 2 > 0) ∧ ¬(x^2 + x - 2 > 0 → x > 1) :=
by {
  split,
  { -- Sufficient condition
    intro h,
    exact suff_but_not_necessary h,
  },
  { -- Not necessary condition
    intro h,
    cases not_necessary h with h1 h2,
    { -- case x > 1
      tauto,
    },
    { -- case x < -2
      exfalso,
      lv, -- insert proof step to show contradiction
      sorry,
    }
  }
}

end suff_but_not_necessary_not_necessary_sufficient_but_not_necessary_condition_l0_792


namespace trig_identity_sin_eq_l0_127

theorem trig_identity_sin_eq (α : ℝ) (h : Real.cos (π / 6 - α) = 1 / 3) : 
  Real.sin (5 * π / 6 - 2 * α) = -7 / 9 := 
by 
  sorry

end trig_identity_sin_eq_l0_127


namespace positive_difference_of_two_numbers_l0_906

theorem positive_difference_of_two_numbers 
  (a b : ℝ) 
  (h₁ : a + b = 10) 
  (h₂ : a^2 - b^2 = 40) : 
  |a - b| = 4 := 
sorry

end positive_difference_of_two_numbers_l0_906


namespace men_finished_race_l0_317

theorem men_finished_race : 
  let total_men := 80
  let tripped := total_men * 1 / 4
  let remaining_after_tripped := total_men - tripped
  let dehydrated := remaining_after_tripped * 2 / 3
  let dehydrated_unfinished := dehydrated * 1 / 5
  let dehydrated_finished := dehydrated - dehydrated_unfinished
  let non_dehydrated_finished := remaining_after_tripped - dehydrated
  in dehydrated_finished + non_dehydrated_finished = 52 :=
by
  let total_men := 80
  let tripped := total_men * 1 / 4
  let remaining_after_tripped := total_men - tripped
  let dehydrated := remaining_after_tripped * 2 / 3
  let dehydrated_unfinished := dehydrated * 1 / 5
  let dehydrated_finished := dehydrated - dehydrated_unfinished
  let non_dehydrated_finished := remaining_after_tripped - dehydrated
  show dehydrated_finished + non_dehydrated_finished = 52
  sorry

end men_finished_race_l0_317


namespace distance_min_value_l0_686

theorem distance_min_value (a b c d : ℝ) 
  (h₁ : |b - (Real.log a) / a| + |c - d + 2| = 0) : 
  (a - c)^2 + (b - d)^2 = 9 / 2 :=
by {
  sorry
}

end distance_min_value_l0_686


namespace polygon_properties_l0_393

-- Define the perimeter and area of a regular pentagon
def perimeter (n : ℕ) (s : ℕ) : ℕ := n * s
noncomputable def pentagon_area (s : ℕ) : ℝ := (1 / 4) * real.sqrt(5 * (5 + 2 * real.sqrt 5)) * (s : ℝ)^2

theorem polygon_properties :
  ∀ (n s : ℕ) (θ : ℝ),
    θ = 72 ∧ s = 6 ∧ (360 / θ).to_nat = 5 →
    perimeter 5 6 = 30 ∧ abs (pentagon_area 6 - 139.482) < 1e-3 :=
by
  intros n s θ h
  rcases h with ⟨hθ, hs, hn⟩
  dsimp [perimeter, pentagon_area]
  dsimp only [real.sqrt, real.sqrt, (Nat.cast (6 * 6) : ℝ)]
  split
  case left =>
    rw [hs, hn]
  case right =>
    rw [real.sqrt, real.sqrt]
  sorry

end polygon_properties_l0_393


namespace cos_135_eq_neg_sqrt2_div_2_l0_627

theorem cos_135_eq_neg_sqrt2_div_2 : 
  Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_627


namespace maximize_pairwise_sums_l0_24

theorem maximize_pairwise_sums :
  ∃ (S : ℕ), 
  let known_sums := [210, 340, 290, 250, 370] in
  let unknown_sums := [x, y, z, w, u] in
  S = 710 ∧
  ∑ x in known_sums, x + ∑ x in unknown_sums, x = 5 * S ∧
  (∀ s ∈ unknown_sums, s = x ∨ s = y ∨ s = z ∨ s = w ∨ s = u) → 
  x + y + z + w + u = 2080 :=
begin
  use 710,
  simp,
  sorry
end

end maximize_pairwise_sums_l0_24


namespace karen_total_nuts_l0_775

variable (x y : ℝ)
variable (hx : x = 0.25)
variable (hy : y = 0.25)

theorem karen_total_nuts : x + y = 0.50 := by
  rw [hx, hy]
  norm_num

end karen_total_nuts_l0_775


namespace positive_difference_of_two_numbers_l0_939

theorem positive_difference_of_two_numbers (x y : ℝ)
  (h1 : x + y = 10) (h2 : x ^ 2 - y ^ 2 = 40) : abs (x - y) = 4 := 
by
  sorry

end positive_difference_of_two_numbers_l0_939


namespace pizza_eaten_after_six_trips_l0_347

theorem pizza_eaten_after_six_trips :
  let a := (1 : ℝ) / 3
  let r := (1 : ℝ) / 3
  let n := 6
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 364 / 729 :=
by
  let a := (1 : ℝ) / 3
  let r := (1 : ℝ) / 3
  let n := 6
  let S_n := a * (1 - r^n) / (1 - r)
  have : S_n = (1 / 3) * (1 - (1 / 3)^6) / (1 - 1 / 3) := by sorry
  have : S_n = 364 / 729 := by sorry
  exact this

end pizza_eaten_after_six_trips_l0_347


namespace cos_square_sum_inequality_part_a_l0_4

theorem cos_square_sum_inequality_part_a (α β γ : Real) :
    cos α ^ 2 + cos β ^ 2 + cos γ ^ 2 = 1 - 2 * cos α * cos β * cos γ →
    cos α ^ 2 + cos β ^ 2 + cos γ ^ 2 ≥ 3 / 4 :=
begin
  sorry
end

end cos_square_sum_inequality_part_a_l0_4


namespace coins_in_first_stack_l0_633

theorem coins_in_first_stack (total_coins : ℕ) (stack_two_coins : ℕ) (h : total_coins = 12) (h2 : stack_two_coins = 8) : (total_coins - stack_two_coins = 4) :=
by {
  have h3 : total_coins - stack_two_coins = 12 - 8, from congrArg (λ x, x - stack_two_coins) h,
  rw [h2] at h3,
  exact h3,
}

end coins_in_first_stack_l0_633


namespace number_in_scientific_notation_l0_274

def scientific_notation_form (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ a ∧ a < 10

theorem number_in_scientific_notation : scientific_notation_form 3.7515 7 ∧ 37515000 = 3.7515 * 10^7 :=
by
  sorry

end number_in_scientific_notation_l0_274


namespace area_of_circle_above_line_l0_653

noncomputable def circle_area_above_line : ℝ :=
  let center := (2, 4) in
  let radius := 2 in
  let circle_eqn := (x - 2)^2 + (y - 4)^2 = 4 in
  let line_eqn := y = 3 in
  if ∀ (p : ℝ × ℝ), circle_eqn p → p.2 > 3 then
    π * radius^2
  else
    0

theorem area_of_circle_above_line : circle_area_above_line = 4 * π :=
  sorry

end area_of_circle_above_line_l0_653


namespace oatmeal_cookie_count_l0_241

theorem oatmeal_cookie_count (TotalBags CookiesPerBag ChocChipCookies : ℕ) :
  TotalBags = 7 → CookiesPerBag = 5 → ChocChipCookies = 33 → (TotalBags * CookiesPerBag - ChocChipCookies = 2) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end oatmeal_cookie_count_l0_241


namespace cosine_of_angle_between_diagonals_l0_388

theorem cosine_of_angle_between_diagonals :
  let a : ℝ × ℝ × ℝ := (3, 2, 1)
  let b : ℝ × ℝ × ℝ := (-1, 2, 2)
  let d1 := (a.1 + b.1, a.2 + b.2, a.3 + b.3)
  let d2 := (a.1 - b.1, a.2 - b.2, a.3 - b.3)
  let dot_product := d1.1 * d2.1 + d1.2 * d2.2 + d1.3 * d2.3
  let mag_d1 := Real.sqrt (d1.1 ^ 2 + d1.2 ^ 2 + d1.3 ^ 2)
  let mag_d2 := Real.sqrt (d2.1 ^ 2 + d2.2 ^ 2 + d2.3 ^ 2)
  Real.cos (Real.acos (dot_product / (mag_d1 * mag_d2))) = 5 / Real.sqrt 493 := 
by
  sorry

end cosine_of_angle_between_diagonals_l0_388


namespace prove_a_plus_t_l0_128

noncomputable def condition1 : Prop := real.sqrt (2 + 2/3) = 2 * real.sqrt (2 / 3)
noncomputable def condition2 : Prop := real.sqrt (3 + 3/8) = 3 * real.sqrt (3 / 8)
noncomputable def condition3 : Prop := real.sqrt (4 + 4/15) = 4 * real.sqrt (4 / 15)
noncomputable def general_condition (n : ℕ) : Prop := real.sqrt (n + 1 + (n + 1) / ((n + 1)^2 - 1)) = (n + 1) * real.sqrt ((n + 1) / ((n + 1)^2 - 1))

theorem prove_a_plus_t :
  let a := 6
  let t := a^2 - 1
  a + t = 41 :=
by
  let ha := 6
  let ht := 35
  have a_def : ha = 6 := rfl
  have t_def : ht = 6^2 - 1 := rfl
  have a_plus_t := ha + ht
  have result : a_plus_t = 41 := by norm_num
  exact result

end prove_a_plus_t_l0_128


namespace a_range_l0_144

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a - 3) * x + 4 * a

theorem a_range (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) ↔ (0 < a ∧ a ≤ 1 / 4) :=
by
  sorry

end a_range_l0_144


namespace find_x0_l0_715

theorem find_x0 : ∃ x0 ∈ set.Icc 0 real.pi, (cos x0 + sin x0) = (sin x0 - cos x0) ∧ x0 = real.pi / 2 :=
by
  sorry

end find_x0_l0_715


namespace true_propositions_l0_224

-- Definitions for the problem
variables {α β : Set Point} -- where α and β are distinct planes
variables {m n : Line} -- where m and n are distinct lines

def proposition1 (α β : Set Point) (m n : Line) : Prop :=
  m ⊥ n → m ⊥ α → n ⊄ α → n ∥ α

def proposition2 (α β : Set Point) (m n : Line) : Prop :=
  m ⊥ n → m ∥ α → n ∥ β → α ⊥ β

def proposition3 (α β : Set Point) (m n : Line) : Prop :=
  α ⊥ β → (α ∩ β = m) → (n ⊂ α) → (n ⊥ m) → n ⊥ β

def proposition4 (α β : Set Point) (m n : Line) : Prop :=
  (n ⊂ α) → (m ⊂ β) → α ∩ β ≠ ⊥ → n ∦ m

-- Main theorem: Propositions 1 and 3 are the true ones
theorem true_propositions :
  proposition1 α β m n ∧ proposition3 α β m n ∧ ¬ proposition2 α β m n ∧ ¬ proposition4 α β m n :=
by {
  sorry -- Proof is omitted as per instructions
}

end true_propositions_l0_224


namespace inscribed_square_ratio_l0_26

theorem inscribed_square_ratio (x y : ℝ)
    (h1 : ∃ t : triangle, t.side_a = 5 ∧ t.side_b = 12 ∧ t.side_c = 13 ∧ ∃ s : square, s.side = x ∧ s.vertex_at_right_angle_vertex t)
    (h2 : ∃ t : triangle, t.side_a = 5 ∧ t.side_b = 12 ∧ t.side_c = 13 ∧ ∃ s : square, s.side = y ∧ s.side_on_hypotenuse t):
    x / y = 5 / 13 := sorry

end inscribed_square_ratio_l0_26


namespace cos_135_eq_neg_sqrt_two_div_two_l0_448

theorem cos_135_eq_neg_sqrt_two_div_two : real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt_two_div_two_l0_448


namespace part_I_part_II_l0_698

open Function

-- Definition of an increasing function on ℝ
def is_increasing (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- Part (Ⅰ)
theorem part_I (f : ℝ → ℝ) (hf : is_increasing f) (a b : ℝ) (hab : a + b ≥ 0) : f(a) + f(b) ≥ f(-a) + f(-b) :=
sorry

-- Part (Ⅱ)
theorem part_II (f : ℝ → ℝ) (hf : is_increasing f) (a b : ℝ) (h : f(a) + f(b) ≥ f(-a) + f(-b)) : a + b ≥ 0 :=
sorry

end part_I_part_II_l0_698


namespace number_of_discount_cards_l0_320

def is_discount_card (card_number : Nat) : Bool :=
  -- Checks if the card number's last four digits contain "6" or "8"
  let last_four := card_number % 10000
  let digits := [last_four / 1000 % 10, last_four / 100 % 10, last_four / 10 % 10, last_four % 10]
  digits.any (λ d => d = 6 ∨ d = 8)

theorem number_of_discount_cards : 
  (Finrange 10000).count is_discount_card = 5904 :=
sorry

end number_of_discount_cards_l0_320


namespace lines_perpendicular_l0_662

theorem lines_perpendicular (b : ℝ) :
  let d1 := ![2, -1, b]
  let d2 := ![-1, 2, 3]
  (d1 ⬝ d2 = 0) → b = 4 / 3 :=
by {
  sorry
}

end lines_perpendicular_l0_662


namespace range_m_max_min_fx_l0_714

def f (x : ℝ) : ℝ := log 3 (9 * x) * log 3 (3 * x)

theorem range_m (x : ℝ) (hx : 1 / 9 ≤ x ∧ x ≤ 9) : -2 ≤ log 3 x ∧ log 3 x ≤ 2 :=
  sorry

theorem max_min_fx (x : ℝ) (hx : 1 / 9 ≤ x ∧ x ≤ 9) :
  ∃! minx : ℝ, x = 3^(-3/2) ∧ (∀ y, f y > f 3^(-3/2)) ∧ f 3^(-3/2) = -(1/4) ∧
  ∃! maxx : ℝ, x = 9 ∧ (∀ y, f y < f 9) ∧ f 9 = 12 :=
  sorry

end range_m_max_min_fx_l0_714


namespace celebration_day_1500_l0_803

def day_of_week_after (n : ℕ) (start_day : String) : String :=
  let days := ["Friday", "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]
  let idx := days.indexOf start_day
  days[(idx + n) % 7]

theorem celebration_day_1500 :
  day_of_week_after 2 "Friday" = "Sunday" :=
sorry

end celebration_day_1500_l0_803


namespace probability_of_5_out_of_7_odd_rolls_l0_332

theorem probability_of_5_out_of_7_odd_rolls :
  let prob_odd := (1 / 2 : ℝ)
  let prob_even := (1 / 2 : ℝ)
  let success_count := 5
  let trial_count := 7
  let combination := nat.choose trial_count success_count
  let prob := combination * prob_odd ^ success_count * prob_even ^ (trial_count - success_count)
  in prob = 21 / 128 := by
  sorry

end probability_of_5_out_of_7_odd_rolls_l0_332


namespace tenth_term_value_l0_677

noncomputable def seq : ℕ → ℝ
| 1     := 2
| 2     := 1
| (n+2) := x where h : seq n - seq (n+1) / seq n = seq (n+1) - x / x := sorry

theorem tenth_term_value :
  seq 10 = 1 / 5 :=
sorry

end tenth_term_value_l0_677


namespace largest_integer_solution_l0_277

theorem largest_integer_solution (x : ℤ) (h : -x ≥ 2 * x + 3) : x ≤ -1 := sorry

end largest_integer_solution_l0_277


namespace master_works_in_10_days_l0_283

noncomputable def master_apprentice_work (x : ℝ) (y : ℝ) : Prop :=
  let master_work_rate := (1 : ℝ) / x
  let apprentice_work_rate := 1 / y
  let combined_work_rate := master_work_rate + apprentice_work_rate
  let half_work_time := 1 / (2 * combined_work_rate)
  (2 * half_work_time + half_work_time = x + 2) ∧ y = x + 5

theorem master_works_in_10_days :
  {x : ℝ // master_apprentice_work x (x + 5)} :=
by
  use 10
  unfold master_apprentice_work
  simp
  split
  · field_simp ; ring
  · rfl
  sorry   -- Detailed actual proof steps would go here

end master_works_in_10_days_l0_283


namespace find_arithmetic_sequence_l0_682

open Nat Real

-- Defining the arithmetic sequence
def arithmetic_sequence (a d : ℝ) := [a - d, a, a + d, a + 2 * d]

-- Conditions on the sequence
def conditions (a d : ℝ) : Prop :=
  (arithmetic_sequence a d).sum = 26 ∧ (a * (a + d)) = 40

-- The possible resulting sequences
def possible_sequences : set (list ℝ) :=
  {[2, 5, 8, 11], [11, 8, 5, 2]}

-- The theorem statement
theorem find_arithmetic_sequence (a d : ℝ) :
  conditions a d →
  arithmetic_sequence a d ∈ possible_sequences :=
by
  sorry

end find_arithmetic_sequence_l0_682


namespace degree_monomial_example_l0_270

variable (a b : ℕ → ℕ) -- Assuming a and b are functions representing the exponentiation

def is_monomial (c : ℤ) (f : ℕ → ℕ) : Prop :=
  ∃ n m : ℕ, f = λ x, if x = 1 then n else if x = 2 then m else 0

noncomputable def degree_of_monomial (c : ℤ) (f : ℕ → ℕ) : ℕ :=
  ∑ i in {1, 2}, f i

theorem degree_monomial_example : is_monomial (-3) (λ i, if i = 1 then 2 else if i = 2 then 1 else 0) →
  degree_of_monomial (-3) (λ i, if i = 1 then 2 else if i = 2 then 1 else 0) = 3 :=
by
  intro h
  simp [degree_of_monomial]
  sorry

end degree_monomial_example_l0_270


namespace cos_135_eq_neg_inv_sqrt_2_l0_526

theorem cos_135_eq_neg_inv_sqrt_2 : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_526


namespace cos_beta_and_2alpha_plus_beta_l0_694

variables {α β : ℝ}

theorem cos_beta_and_2alpha_plus_beta
  (h1 : cos α = 1 / 3)
  (h2 : cos (α + β) = -1 / 3)
  (h3 : 0 < α ∧ α < π / 2)
  (h4 : 0 < β ∧ β < π / 2) :
  cos β = 7 / 9 ∧ 2 * α + β = π :=
by
  sorry

end cos_beta_and_2alpha_plus_beta_l0_694


namespace amount_lent_l0_16

theorem amount_lent (P : ℝ) (annual_interest_A_to_B : ℝ) (annual_interest_B_to_C : ℝ)
    (years : ℕ) (gain_B : ℝ)
    (h1 : annual_interest_A_to_B = 0.10)
    (h2 : annual_interest_B_to_C = 0.12)
    (h3 : years = 3)
    (h4 : gain_B = 210) :
    P = 3500 :=
begin
    sorry
end

end amount_lent_l0_16


namespace range_of_a_l0_142

variable {a b : ℝ}

theorem range_of_a (H : ∀ (b : ℝ), b ≤ 0 → ∀ (x : ℝ), e < x ∧ x ≤ e^2 → a * log x - b * x^2 ≥ x) : a ≥ e^2 / 2 := 
sorry

end range_of_a_l0_142


namespace cos_135_eq_neg_sqrt2_div_2_l0_481

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by 
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_481


namespace part1_part2_part3_l0_151

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x λ : ℝ) : ℝ := λ * (x^2 - 1)

theorem part1 (h : deriv f 1 = deriv (λ x, g x λ) 1) : λ = 1 / 2 := by
  sorry

theorem part2 {x : ℝ} (h1 : λ = 1 / 2) (h2 : 1 ≤ x) : f x ≤ g x λ := by
  sorry

theorem part3 (h : ∀ x ≥ 1, f x ≤ g x λ) : 1 / 2 ≤ λ := by
  sorry

end part1_part2_part3_l0_151


namespace semicircle_circumference_41_12_l0_884

noncomputable def circumference_of_semicircle (length : ℝ) (breadth: ℝ) : ℝ :=
  let perimeter_rectangle := 2 * (length + breadth)
  let side_square := perimeter_rectangle / 4
  let diameter_semicircle := side_square
  let circumference_semicircle := (Real.pi * diameter_semicircle) / 2 + diameter_semicircle
  circumference_semicircle

theorem semicircle_circumference_41_12 :
  circumference_of_semicircle 18 14 ≈ 41.12 :=
by
  sorry

end semicircle_circumference_41_12_l0_884


namespace cos_135_eq_neg_sqrt2_div_2_l0_513

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = - Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_513


namespace winning_strategy_count_l0_68

theorem winning_strategy_count :
  ∃ (count : ℕ), count = 47 ∧ ∀ N, (2 ≤ N ∧ N ≤ 100) → Player1_has_winning_strategy N ↔ N ∈ {even_not_power_of_2_set} :=
by {
  sorry
}

end winning_strategy_count_l0_68


namespace eighth_grade_girls_l0_877

theorem eighth_grade_girls
  (G : ℕ) 
  (boys : ℕ) 
  (h1 : boys = 2 * G - 16) 
  (h2 : G + boys = 68) : 
  G = 28 :=
by
  sorry

end eighth_grade_girls_l0_877


namespace cos_135_degree_l0_502

theorem cos_135_degree : 
  let Q := (Complex.exp (135 * Real.pi / 180 * Complex.I)) in 
  Complex.re Q = -Real.sqrt 2 / 2 :=
by sorry

end cos_135_degree_l0_502


namespace positive_difference_of_numbers_l0_927

theorem positive_difference_of_numbers
  (a b : ℝ)
  (h1 : a + b = 10)
  (h2 : a^2 - b^2 = 40) :
  |a - b| = 4 :=
by 
  sorry

end positive_difference_of_numbers_l0_927


namespace grapefruits_count_l0_995

def citrus_orchards : Prop :=
  ∃ (total lemons oranges limes grapefruits : ℕ),
    (total = 16) ∧ (lemons = 8) ∧ (oranges = lemons / 2) ∧ 
    (limes + grapefruits = total - (lemons + oranges)) ∧ 
    (limes = grapefruits) ∧ (grapefruits = 2)

theorem grapefruits_count : citrus_orchards :=
begin
  unfold citrus_orchards,
  use 16,
  use 8,
  use 4,
  use 2,
  use 2,
  split,
  { refl },
  split,
  { refl },
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { refl },
  refl,
end

end grapefruits_count_l0_995


namespace total_difference_in_cents_l0_58

variable (q : ℕ)

def charles_quarters := 6 * q + 2
def charles_dimes := 3 * q - 2

def richard_quarters := 2 * q + 10
def richard_dimes := 4 * q + 3

def cents_from_quarters (n : ℕ) : ℕ := 25 * n
def cents_from_dimes (n : ℕ) : ℕ := 10 * n

theorem total_difference_in_cents : 
  (cents_from_quarters (charles_quarters q) + cents_from_dimes (charles_dimes q)) - 
  (cents_from_quarters (richard_quarters q) + cents_from_dimes (richard_dimes q)) = 
  90 * q - 250 :=
by
  sorry

end total_difference_in_cents_l0_58


namespace cells_after_3_hours_l0_993

noncomputable def cell_division_problem (t : ℕ) : ℕ :=
  2 ^ (t * 2)

theorem cells_after_3_hours : cell_division_problem 3 = 64 := by
  sorry

end cells_after_3_hours_l0_993


namespace number_of_customers_per_month_l0_327

-- Define the constants and conditions
def price_lettuce_per_head : ℝ := 1
def price_tomato_per_piece : ℝ := 0.5
def num_lettuce_per_customer : ℕ := 2
def num_tomato_per_customer : ℕ := 4
def monthly_sales : ℝ := 2000

-- Calculate the cost per customer
def cost_per_customer : ℝ := 
  (num_lettuce_per_customer * price_lettuce_per_head) + 
  (num_tomato_per_customer * price_tomato_per_piece)

-- Prove the number of customers per month
theorem number_of_customers_per_month : monthly_sales / cost_per_customer = 500 :=
  by
    -- Here, we would write the proof steps
    sorry

end number_of_customers_per_month_l0_327


namespace positive_difference_of_two_numbers_l0_918

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l0_918


namespace obtuse_triangle_contradiction_l0_965

theorem obtuse_triangle_contradiction (A B C : ℝ) (h1 : A + B + C = 180) (h2 : 0 < A) (h3 : 0 < B) (h4 : 0 < C) : 
  (A > 90 ∧ B > 90) → false :=
by
  sorry

end obtuse_triangle_contradiction_l0_965


namespace total_earnings_after_six_months_l0_835

def area_of_farm : ℕ := 20
def trees_per_square_meter : ℕ := 2
def coconuts_per_tree : ℕ := 6
def harvest_interval_months : ℕ := 3
def price_per_coconut : ℕ := 0.5 -- Note: Lean requires specific handling for non-integer values, so we can use a fractional form instead

theorem total_earnings_after_six_months :
  let total_trees := area_of_farm * trees_per_square_meter in
  let total_coconuts := total_trees * coconuts_per_tree in
  let number_of_harvests := 6 / harvest_interval_months in
  let total_coconuts_six_months := total_coconuts * number_of_harvests in
  let earnings := total_coconuts_six_months * price_per_coconut in
  earnings = 240 :=
by {
  sorry
}

end total_earnings_after_six_months_l0_835


namespace cylinder_lateral_surface_area_l0_268

-- Definitions of the conditions
def circumference (c : ℝ) : Prop := c = 5
def height (h : ℝ) : Prop := h = 2

-- The formula for the lateral surface area of the cylinder
def lateral_surface_area (c h : ℝ) : ℝ := c * h

-- The theorem we need to prove
theorem cylinder_lateral_surface_area (c h : ℝ) (Hc : circumference c) (Hh : height h) :
  lateral_surface_area c h = 10 :=
sorry

end cylinder_lateral_surface_area_l0_268


namespace age_difference_l0_15

variable (y m e : ℕ)

theorem age_difference (h1 : m = y + 3) (h2 : e = 3 * y) (h3 : e = 15) : 
  ∃ x, e = y + m + x ∧ x = 2 := by
  sorry

end age_difference_l0_15


namespace seven_pow_k_minus_k_pow_seven_l0_973

theorem seven_pow_k_minus_k_pow_seven (k : ℕ) (h : 21^k ∣ 435961) : 7^k - k^7 = 1 :=
sorry

end seven_pow_k_minus_k_pow_seven_l0_973


namespace parabola_standard_equation_l0_738

def parabola_focus_at_distance (p : ℝ) : Prop :=
  let focus : ℝ × ℝ := (p / 2, 0)
  let hyperbola_asymptote_distance : ℝ := (p * real.sqrt p) / (2 * real.sqrt (p + 8))
  hyperbola_asymptote_distance = (real.sqrt 2 / 4) * p

theorem parabola_standard_equation (p : ℝ) (h : parabola_focus_at_distance p) : 
  y^2 = 16 * x :=
by
  sorry

end parabola_standard_equation_l0_738


namespace greatest_drop_in_june_l0_397

def monthly_changes := [("January", 1.50), ("February", -2.25), ("March", 0.75), ("April", -3.00), ("May", 1.00), ("June", -4.00)]

theorem greatest_drop_in_june : ∀ months : List (String × Float), (months = monthly_changes) → 
  (∃ month : String, 
    month = "June" ∧ 
    ∀ m p, m ≠ "June" → (m, p) ∈ months → p ≥ -4.00) :=
by
  sorry

end greatest_drop_in_june_l0_397


namespace derek_dogs_problem_l0_963

theorem derek_dogs_problem :
  ∀ (dogs_at_7 : ℕ) (car_multiplier : ℕ) (additional_cars : ℕ) (car_to_dog_ratio : ℕ),
    dogs_at_7 = 120 →
    car_multiplier = 4 →
    additional_cars = 350 →
    car_to_dog_ratio = 3 →
    let cars_at_7 := dogs_at_7 / car_multiplier in
    let cars_now := cars_at_7 + additional_cars in
    let dogs_now := cars_now / car_to_dog_ratio in
    dogs_now = 126 :=
by
  intros dogs_at_7 car_multiplier additional_cars car_to_dog_ratio
  intros dogs_at_7_eq car_multiplier_eq additional_cars_eq car_to_dog_ratio_eq
  simp [dogs_at_7_eq, car_multiplier_eq, additional_cars_eq, car_to_dog_ratio_eq]
  let cars_at_7 := 120 / 4
  let cars_now := cars_at_7 + 350
  have : cars_now = 380 := rfl
  let dogs_now := 380 / 3
  show dogs_now = 126 from rfl

end derek_dogs_problem_l0_963


namespace find_digit_sum_l0_654

theorem find_digit_sum (A B X D C Y : ℕ) :
  (A * 100 + B * 10 + X) + (C * 100 + D * 10 + Y) = Y * 1010 + X * 1010 →
  A + D = 6 :=
by
  sorry

end find_digit_sum_l0_654


namespace integral_eq_ln_sub_frac_l0_64

theorem integral_eq_ln_sub_frac (C : ℝ) :
  ∫ (x : ℝ) in set.univ, ((x ^ 3 - 6 * x ^ 2 + 13 * x - 6) / ((x + 2) * (x - 2) ^ 3)) = 
    λ x, ln (abs (x + 2)) - (1 / (2 * (x - 2) ^ 2)) + C :=
by
  sorry

end integral_eq_ln_sub_frac_l0_64


namespace numeric_value_of_BAR_l0_430

variable (b a t c r : ℕ)

-- Conditions from the problem
axiom h1 : b + a + t = 6
axiom h2 : c + a + t = 8
axiom h3 : c + a + r = 12

-- Required to prove
theorem numeric_value_of_BAR : b + a + r = 10 :=
by
  -- Proof goes here
  sorry

end numeric_value_of_BAR_l0_430


namespace cosine_135_eq_neg_sqrt_2_div_2_l0_539

theorem cosine_135_eq_neg_sqrt_2_div_2 (cos180 : Real := -1) (cos90 : Real := 0) (cos45 : Real := Real.sqrt 2 / 2)
    (theta1 : Real := 180) (theta2 : Real := 45) (theta3 : Real := 135) :
    theta3 = theta1 - theta2 ∧ 
    Real.cos theta1 = cos180 ∧ 
    Real.cos theta2 = cos45 → 
    Real.cos theta3 = -cos45 := by
  sorry

end cosine_135_eq_neg_sqrt_2_div_2_l0_539


namespace find_positive_s_l0_665

theorem find_positive_s (s : ℝ) (hs : 0 < s) : |complex.mk (-3) s| = 3 * real.sqrt 5 → s = 6 :=
begin
  sorry
end

end find_positive_s_l0_665


namespace proof_cos_135_degree_l0_567

noncomputable def cos_135_degree_is_negative_sqrt2_over_2 : Prop :=
  real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2

theorem proof_cos_135_degree : cos_135_degree_is_negative_sqrt2_over_2 :=
by
  sorry

end proof_cos_135_degree_l0_567


namespace weekly_allowance_l0_723

variable (A : ℝ)   -- declaring A as a real number

theorem weekly_allowance (h1 : (3/5 * A) + 1/3 * (2/5 * A) + 1 = A) : 
  A = 3.75 :=
sorry

end weekly_allowance_l0_723


namespace cos_135_eq_neg_sqrt2_div_2_l0_590

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  -- Conditions incorporated directly into the requirement due to the nature of trigonometric evaluations.
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_590


namespace product_of_points_l0_59

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 8
  else if nat.prime n then 3
  else if n % 2 = 0 then 1
  else 0

def charlie_rolls := [6, 4, 5, 2, 3]
def dana_rolls := [3, 3, 1, 2]

def total_points (rolls : list ℕ) : ℕ :=
  rolls.map g |>.sum

theorem product_of_points : 
  total_points charlie_rolls * total_points dana_rolls = 437 := 
  sorry

end product_of_points_l0_59


namespace cos_135_eq_neg_sqrt2_div_2_l0_465

theorem cos_135_eq_neg_sqrt2_div_2 : cos (135 * real.pi / 180) = - (real.sqrt 2 / 2) :=
by
  have h1 : 135 = 180 - 45 := by norm_num
  have h2 : cos (180 * real.pi / 180 - 45 * real.pi / 180) = - cos (45 * real.pi / 180) :=
    by rw [cos_sub, cos_pi, cos_pi_div_four, sin_pi, sin_pi_div_four]; norm_num
  have h3 : cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by norm_num
  rw [h1, ←h2, h3]
  norm_num
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_465


namespace number_for_B_expression_l0_876

-- Define the number for A as a variable
variable (a : ℤ)

-- Define the number for B in terms of a
def number_for_B (a : ℤ) : ℤ := 2 * a - 1

-- Statement to prove
theorem number_for_B_expression (a : ℤ) : number_for_B a = 2 * a - 1 := by
  sorry

end number_for_B_expression_l0_876


namespace smallest_positive_e_for_polynomial_l0_638

theorem smallest_positive_e_for_polynomial :
  ∃ a b c d e : ℤ, e = 168 ∧
  (a * x ^ 4 + b * x ^ 3 + c * x ^ 2 + d * x + e = 0) ∧
  (a * (x + 3) * (x - 7) * (x - 8) * (4 * x + 1) = a * x ^ 4 + b * x ^ 3 + c * x ^ 2 + d * x + e) := sorry

end smallest_positive_e_for_polynomial_l0_638


namespace cosine_135_eq_neg_sqrt_2_div_2_l0_531

theorem cosine_135_eq_neg_sqrt_2_div_2 (cos180 : Real := -1) (cos90 : Real := 0) (cos45 : Real := Real.sqrt 2 / 2)
    (theta1 : Real := 180) (theta2 : Real := 45) (theta3 : Real := 135) :
    theta3 = theta1 - theta2 ∧ 
    Real.cos theta1 = cos180 ∧ 
    Real.cos theta2 = cos45 → 
    Real.cos theta3 = -cos45 := by
  sorry

end cosine_135_eq_neg_sqrt_2_div_2_l0_531


namespace cos_135_eq_neg_sqrt2_div_2_l0_471

theorem cos_135_eq_neg_sqrt2_div_2 : cos (135 * real.pi / 180) = - (real.sqrt 2 / 2) :=
by
  have h1 : 135 = 180 - 45 := by norm_num
  have h2 : cos (180 * real.pi / 180 - 45 * real.pi / 180) = - cos (45 * real.pi / 180) :=
    by rw [cos_sub, cos_pi, cos_pi_div_four, sin_pi, sin_pi_div_four]; norm_num
  have h3 : cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by norm_num
  rw [h1, ←h2, h3]
  norm_num
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_471


namespace cos_135_degree_l0_498

theorem cos_135_degree : 
  let Q := (Complex.exp (135 * Real.pi / 180 * Complex.I)) in 
  Complex.re Q = -Real.sqrt 2 / 2 :=
by sorry

end cos_135_degree_l0_498


namespace division_problem_l0_648

theorem division_problem : 160 / (10 + 11 * 2) = 5 := 
  by 
    sorry

end division_problem_l0_648


namespace positive_difference_of_two_numbers_l0_920

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l0_920


namespace positive_difference_of_numbers_l0_922

theorem positive_difference_of_numbers
  (a b : ℝ)
  (h1 : a + b = 10)
  (h2 : a^2 - b^2 = 40) :
  |a - b| = 4 :=
by 
  sorry

end positive_difference_of_numbers_l0_922


namespace max_square_test_plots_l0_379

theorem max_square_test_plots 
  (length : ℕ) (width : ℕ) (fence_available : ℕ) 
  (side_length : ℕ) (num_plots : ℕ) 
  (h_length : length = 30)
  (h_width : width = 60)
  (h_fencing : fence_available = 2500)
  (h_side_length : side_length = 10)
  (h_num_plots : num_plots = 18) :
  (length * width / side_length^2 = num_plots) ∧
  (30 * (60 / side_length - 1) + 60 * (30 / side_length - 1) ≤ fence_available) := 
sorry

end max_square_test_plots_l0_379


namespace fg_of_2_l0_169

def g (x : ℝ) : ℝ := 2 * x^2
def f (x : ℝ) : ℝ := 2 * x - 1

theorem fg_of_2 : f (g 2) = 15 :=
by
  have h1 : g 2 = 8 := by sorry
  have h2 : f 8 = 15 := by sorry
  rw [h1]
  exact h2

end fg_of_2_l0_169


namespace average_percent_increase_per_year_l0_355

def initial_population : ℕ := 175000
def final_population : ℕ := 262500
def period : ℕ := 10

theorem average_percent_increase_per_year : 
  ((final_population - initial_population : ℤ) / period : ℤ.to_float) / initial_population * 100 = 5 := 
by
  sorry

end average_percent_increase_per_year_l0_355


namespace max_sum_abs_diff_l0_233

theorem max_sum_abs_diff (n : ℕ) (hn : n > 0) (x : ℕ → ℝ) (hx : ∀ i, 1 ≤ i → i ≤ n → 0 < x i ∧ x i < 1) :
  ∑ i in Finset.range n, ∑ j in Finset.range i, | x i - x j | ≤ (n^2 / 4) := sorry

end max_sum_abs_diff_l0_233


namespace cos_theta_irrational_l0_410

theorem cos_theta_irrational
  (p q : ℕ) 
  (θ : ℝ)
  (hpq_rel_prime : Nat.coprime p q)
  (h_triangle : ∃ a b c : ℕ, p * θ + q * θ + (π - (p + q) * θ) = π) :
  ¬∃ q : ℚ, cos θ = q := 
sorry

end cos_theta_irrational_l0_410


namespace cos_135_degree_l0_499

theorem cos_135_degree : 
  let Q := (Complex.exp (135 * Real.pi / 180 * Complex.I)) in 
  Complex.re Q = -Real.sqrt 2 / 2 :=
by sorry

end cos_135_degree_l0_499


namespace cos_135_eq_correct_l0_613

noncomputable def cos_135_eq: Prop :=
  real.cos ((135 * real.pi) / 180) = -real.sqrt 2 / 2

theorem cos_135_eq_correct : cos_135_eq :=
  sorry

end cos_135_eq_correct_l0_613


namespace part1_general_term_part2_sum_first_n_terms_l0_126

noncomputable theory

-- Define the sequence a_n and the sums S_n and B_n
def a (n : ℕ) : ℕ := 2 ^ (n - 1)
def S (n : ℕ) : ℕ := ∑ i in range n, a i
def B (n : ℕ) : ℕ := ∑ i in range n, (i + 1) * a (i + 1)

-- Define the conditions
axiom a1_ne_zero : a 1 ≠ 0
axiom main_condition (n : ℕ) (hn : n ≠ 0) : 2 * a n - a 1 = S 1 * S n

-- Prove that a1 = 1, a2 = 2, and the general term formula for the sequence {a_n} is 2^(n-1)
theorem part1_general_term : ∀ n : ℕ, n > 0 → a n = 2 ^ (n - 1) := sorry

-- Prove that the sum of first n terms of the sequence {na_n} is (n-1)2^n+1
theorem part2_sum_first_n_terms : ∀ n : ℕ, B n = (n - 1) * 2^n + 1 := sorry

end part1_general_term_part2_sum_first_n_terms_l0_126


namespace leading_coeff_is_one_over_18_l0_385

noncomputable def leading_coefficient_of_polynomial (f : ℝ[X]) : ℝ := 
  if h : f ≠ 0 ∧ f = (f.derivative) * (f.derivative.derivative) then
    (polynomial.leading_coeff f)
  else
    0

theorem leading_coeff_is_one_over_18 (f : ℝ[X]) (h_nonzero : f ≠ 0) (h_eq : f = (f.derivative) * (f.derivative.derivative)) :
  polynomial.leading_coeff f = 1 / 18 :=
by
  sorry

end leading_coeff_is_one_over_18_l0_385


namespace cos_135_eq_neg_inv_sqrt2_l0_444

theorem cos_135_eq_neg_inv_sqrt2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt2_l0_444


namespace right_triangle_count_l0_812

theorem right_triangle_count :
  let points_y1 := { (x, 1) | x in finset.range 1 201 }
  let points_y6 := { (x, 6) | x in finset.range 1 201 }
  let points := points_y1 ∪ points_y6
  ∃ (n : ℕ), n = 80676 ∧ (∀ p1 p2 p3 ∈ points, is_right_triangle (p1, p2, p3) → n.ways) := sorry

-- Additional supporting definitions might be required for a complete and accurate formalization
-- such as 'is_right_triangle' or 'ways'. These are assumed to exist for this theorem statement.

end right_triangle_count_l0_812


namespace solveEquation_l0_845

theorem solveEquation (x : ℝ) (hx : |x| ≥ 3) : (∃ x₁ x₂ : ℝ, (x₁ ≠ x₂ ∧ (x₁ / 3 + x₁ / Real.sqrt (x₁ ^ 2 - 9) = 35 / 12) ∧ (x₂ / 3 + x₂ / Real.sqrt (x₂ ^ 2 - 9) = 35 / 12)) ∧ x₁ + x₂ = 8.75) :=
sorry

end solveEquation_l0_845


namespace cos_135_eq_neg_sqrt2_div_2_l0_478

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by 
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_478


namespace number_of_paths_from_A_to_B_l0_369

-- Definitions based on the conditions
def top_red_to_blue_ways := 3
def bottom_red_to_blue_ways := 4
def blue_to_green_ways_first_second := 5
def blue_to_green_ways_third_fourth := 7
def green_to_first_orange_ways := 4
def green_to_second_orange_ways := 3
def first_orange_to_B_ways := 2
def second_orange_to_B_ways := 8

-- Theorem statement
theorem number_of_paths_from_A_to_B : (number_of_paths A B = 5376) :=
sorry

end number_of_paths_from_A_to_B_l0_369


namespace range_of_m_l0_671

noncomputable def f (x : ℝ) : ℝ := -x^2
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := 2^x - m

theorem range_of_m (m : ℝ) :
  (∀ x1 ∈ Set.Icc (-1 : ℝ) 3, ∃ x2 ∈ Set.Icc (0 : ℝ) 2, f x1 ≥ g x2 m) ↔ m ≥ 10 := 
by
  sorry

end range_of_m_l0_671


namespace double_neg_five_eq_five_l0_422

theorem double_neg_five_eq_five : -(-5) = 5 := 
sorry

end double_neg_five_eq_five_l0_422


namespace cos_135_degree_l0_497

theorem cos_135_degree : 
  let Q := (Complex.exp (135 * Real.pi / 180 * Complex.I)) in 
  Complex.re Q = -Real.sqrt 2 / 2 :=
by sorry

end cos_135_degree_l0_497


namespace value_of_expression_l0_171

theorem value_of_expression (x : ℝ) (h : x^2 - 3 * x = 4) : 3 * x^2 - 9 * x + 8 = 20 := 
by
  sorry

end value_of_expression_l0_171


namespace collinear_M_N_H_circumcenter_O_on_line_M_N_H_l0_198

-- Defining the problem space
variables (ABC : Type) [triangle ABC]
variables (A B C H M N O : ABC)

-- Conditions
-- Constraint of acute-angled triangle with a specific angle A
axiom acute_angle_triangle : triangle ABC
axiom angle_a_eq_60 : angle A = 60

-- Definition of altitudes intersection
axiom altitudes_intersect_at_H : altitudes_intersection_point ABC = H

-- Definition of points M and N on the sides of the triangle
axiom M_on_side_AB : point_on_side M AB
axiom N_on_side_AC : point_on_side N AC

-- Points being the intersections of perpendicular bisectors of segments BH and CH with AB and AC respectively
axiom M_perpendicular_bisector_BH : perpendicular_bisector BH intersects AB at M
axiom N_perpendicular_bisector_CH : perpendicular_bisector CH intersects AC at N

-- Point O, the circumcenter of the triangle
axiom circumcenter_O : circumcenter ABC = O

-- Proofs to be shown: Collinearity and circumcenter alignment
theorem collinear_M_N_H : collinear M N H := sorry
theorem circumcenter_O_on_line_M_N_H : collinear M N H O := sorry

end collinear_M_N_H_circumcenter_O_on_line_M_N_H_l0_198


namespace find_f_expression_l0_700

noncomputable def a : ℝ := 1 / 2

def log_a_x (x : ℝ) : ℝ := Real.log x / Real.log a

def f (x : ℝ) : ℝ := (1 / 2) ^ x

theorem find_f_expression (h₁ : log_a_x 2 = -1)
  (h₂ : ∀ x : ℝ, f x = (λ y, log_a_x y)⁻¹ x) :
  ∀ x : ℝ, f x = (1 / 2) ^ x := by
  sorry

end find_f_expression_l0_700


namespace line_circle_intersection_probability_l0_696

theorem line_circle_intersection_probability :
  ∀ (k : ℝ), -1 ≤ k ∧ k ≤ 1 →
  let line := λ x, k * x + 3 in
  let circle := λ x y, (x - 2) ^ 2 + (y - 3) ^ 2 = 4 in
  let intersection := λ k, 
    2 * Real.sqrt (4 - (2 * k / Real.sqrt (k ^ 2 + 1)) ^ 2) ≥ 2 * Real.sqrt 3 in
  let probable_k := λ k, -Real.sqrt 3 / 3 ≤ k ∧ k ≤ Real.sqrt 3 / 3 in
    (Real.sqrt 3 / 3 / 2 : ℝ) := by sorry

end line_circle_intersection_probability_l0_696


namespace fraction_days_passed_l0_809

-- Conditions
def total_days : ℕ := 30
def pills_per_day : ℕ := 2
def total_pills : ℕ := total_days * pills_per_day -- 60 pills
def pills_left : ℕ := 12
def pills_taken : ℕ := total_pills - pills_left -- 48 pills
def days_taken : ℕ := pills_taken / pills_per_day -- 24 days

-- Question and answer
theorem fraction_days_passed :
  (days_taken : ℚ) / (total_days : ℚ) = 4 / 5 := 
by
  sorry

end fraction_days_passed_l0_809


namespace proof_a_cardA_eq_b_cardB_l0_231

variables {a b : ℕ} (A B : set ℤ)

def disjoint_sets_condition_1 (A B : set ℤ) : Prop :=
  A ∩ B = ∅

def condition_2 (a b : ℕ) (A B : set ℤ) : Prop :=
  ∀ i : ℤ, i ∈ A ∪ B → (i + a) ∈ A ∨ (i - b) ∈ B

theorem proof_a_cardA_eq_b_cardB
  {a b : ℕ}
  (A B : set ℤ)
  (h1 : finite (A ∪ B))
  (h2 : disjoint_sets_condition_1 A B)
  (h3 : condition_2 a b A B) :
  a * (A.to_finset.card) = b * (B.to_finset.card) :=
sorry

end proof_a_cardA_eq_b_cardB_l0_231


namespace part1_part2_l0_710

-- Function definition f(x)
def f (x : ℝ) : ℝ :=
  2 * (√3) * sin x * cos x + 2 * sin x ^ 2

-- Function definition g(x) after transformations
def g (x : ℝ) : ℝ :=
  2 * cos x + 1

-- Function definition h(x), symmetric to g(x) about x = π/4
def h (x : ℝ) : ℝ :=
  g (π / 2 - x)

-- Prove the required values of x
theorem part1 (x : ℝ) (hx : x ∈ Set.Ioc (-π / 2) π) (hfx : f x = 0) :
  x = -π / 3 ∨ x = 0 ∨ x = 2 * π / 3 :=
  sorry

-- Prove the range of h(x) on the given interval
theorem part2 :
  Set.Ioc (0 : ℝ) 3 = Set.image h (Set.Ioc (-π / 6) (2 * π / 3)) :=
  sorry

end part1_part2_l0_710


namespace EF_tangent_to_circumcircle_CFG_l0_218

variables {A B C D E F G : Type}
variables [Triangle ABC]
variables (D : point) (hD : is_foot_of_altitude A B C D)
variables (E : point) (F : point)
variables (hE : on_segment E A D)
variables (hF : on_segment F B C)
variables (h_ratio : rat_eq (length_segment A E / length_segment D E) (length_segment B F / length_segment C F))
variables (G : point)
variables (hG : on_segment G A F)
variables (h_perp : perp B G A F)

theorem EF_tangent_to_circumcircle_CFG : tangent EF (circumcircle CFG) :=
sorry

end EF_tangent_to_circumcircle_CFG_l0_218


namespace temperature_difference_is_correct_l0_870

theorem temperature_difference_is_correct :
  ∀ (high low : ℤ), high = 11 ∧ low = -1 → high - low = 12 :=
by
  intros high low h
  cases h with h_high h_low
  rw [h_high, h_low]
  sorry

end temperature_difference_is_correct_l0_870


namespace impossible_even_sum_l0_853

theorem impossible_even_sum (n m : ℤ) (h : (n^2 + m^2) % 2 = 1) : (n + m) % 2 ≠ 0 :=
sorry

end impossible_even_sum_l0_853


namespace cos_135_eq_neg_sqrt2_div_2_l0_618

theorem cos_135_eq_neg_sqrt2_div_2 : 
  Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_618


namespace union_cardinality_bound_l0_718

variable {A B C : Finset ℕ}

def cardinality_a : card A = 400 := sorry
def cardinality_b : card B = 300 := sorry
def cardinality_c : card C = 400 := sorry
def cardinality_a_c : card (A ∩ C) = 100 := sorry
def cardinality_a_b : card (A ∩ B) = 50 := sorry

theorem union_cardinality_bound :
  700 ≤ card (A ∪ B ∪ C) ∧ card (A ∪ B ∪ C) ≤ 950 :=
  sorry

end union_cardinality_bound_l0_718


namespace age_difference_l0_976

theorem age_difference (P M Mo : ℕ)
  (h1 : P = 3 * M / 5)
  (h2 : Mo = 5 * M / 3)
  (h3 : P + M + Mo = 196) :
  Mo - P = 64 := 
sorry

end age_difference_l0_976


namespace abs_neg_2_plus_sqrt3_add_tan60_eq_2_l0_360

theorem abs_neg_2_plus_sqrt3_add_tan60_eq_2 :
  abs (-2 + Real.sqrt 3) + Real.tan (Real.pi / 3) = 2 :=
by
  sorry

end abs_neg_2_plus_sqrt3_add_tan60_eq_2_l0_360


namespace cos_135_eq_neg_sqrt2_div_2_l0_587

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  -- Conditions incorporated directly into the requirement due to the nature of trigonometric evaluations.
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_587


namespace positive_difference_of_two_numbers_l0_902

theorem positive_difference_of_two_numbers 
  (a b : ℝ) 
  (h₁ : a + b = 10) 
  (h₂ : a^2 - b^2 = 40) : 
  |a - b| = 4 := 
sorry

end positive_difference_of_two_numbers_l0_902


namespace cos_135_eq_correct_l0_612

noncomputable def cos_135_eq: Prop :=
  real.cos ((135 * real.pi) / 180) = -real.sqrt 2 / 2

theorem cos_135_eq_correct : cos_135_eq :=
  sorry

end cos_135_eq_correct_l0_612


namespace sum_of_perfect_square_divisors_544_l0_351

theorem sum_of_perfect_square_divisors_544 : 
  let n := 544,
      is_divisor (d : ℕ) := d ∣ n,
      is_perfect_square (d : ℕ) := ∃ (k : ℕ), k * k = d in
  ∑ d in {d | is_divisor d ∧ is_perfect_square d}, d = 21 :=
by
  sorry

end sum_of_perfect_square_divisors_544_l0_351


namespace incorrect_statement_C_l0_372

theorem incorrect_statement_C (h : ℕ → ℝ) (t : ℕ → ℝ) (data : ∀ n, 
    (h n = 10 * n + 10) ∧
    (t 0 = 4.23) ∧
    (t 1 = 3.00) ∧
    (t 2 = 2.45) ∧
    (t 3 = 2.13) ∧
    (t 4 = 1.89) ∧
    (t 5 = 1.71) ∧
    (t 6 = 1.59)) :
    ¬ ( ∃ k, (40 ≤ h k) ∧ (h k ≤ 50) ∧ (t k = 2.0) ) :=
begin
  sorry
end

end incorrect_statement_C_l0_372


namespace positive_difference_l0_899

theorem positive_difference (a b : ℝ) (h₁ : a + b = 10) (h₂ : a^2 - b^2 = 40) : |a - b| = 4 :=
by
  sorry

end positive_difference_l0_899


namespace aaron_age_l0_36

variable (A : ℕ)
variable (henry_sister_age : ℕ)
variable (henry_age : ℕ)
variable (combined_age : ℕ)

theorem aaron_age (h1 : henry_sister_age = 3 * A)
                 (h2 : henry_age = 4 * henry_sister_age)
                 (h3 : combined_age = henry_sister_age + henry_age)
                 (h4 : combined_age = 240) : A = 16 := by
  sorry

end aaron_age_l0_36


namespace cos_135_eq_neg_inv_sqrt_2_l0_580

theorem cos_135_eq_neg_inv_sqrt_2 :
  cos (135 * Real.pi / 180) = - (1 / Real.sqrt 2) :=
by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_580


namespace circle_radius_l0_873

variables (a b d : ℝ)

-- Definition of the function radius with the given conditions
def radius (acute angle : Bool) : ℝ :=
  if a = d then
    sqrt(a^2 + b^2)
  else if angle then
    (a / d) * sqrt(a^2 + b^2 - 2 * b * sqrt(a^2 - d^2))
  else
    (a / d) * sqrt(a^2 + b^2 + 2 * b * sqrt(a^2 - d^2))

-- Statement of the theorem to show the radius conditionally
theorem circle_radius (h1 : ∃ (α : ℝ), (2 * a) * 2 = a^2 - d^2 ∧ (2 * b) * 2 = b^2 - d^2) :
         ∀ (α acute obtuse : Bool),
         radius a b d acute = if a = d then
                               sqrt(a^2 + b^2) 
                             else if acute then
                               (a / d) * sqrt(a^2 + b^2 - 2 * b * sqrt(a^2 - d^2))
                             else
                               (a / d) * sqrt(a^2 + b^2 + 2 * b * sqrt(a^2 - d^2)) :=
by {
  -- We acknowledge the proof is required but not providing here
  sorry,
}

end circle_radius_l0_873


namespace graph_of_f_plus_2_is_E_l0_868

noncomputable def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0    then -2 - x
  else if 0 ≤ x ∧ x ≤ 2 then Real.sqrt (4 - (x - 2)^2) - 2
  else if 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2)
  else 0

noncomputable def g (x : ℝ) : ℝ := f(x) + 2

theorem graph_of_f_plus_2_is_E :
  -- A representation of the fact that the graph of g(x) corresponds to option E
  -- To be filled in with the specific properties of graph E when necessary
  sorry

end graph_of_f_plus_2_is_E_l0_868


namespace proof_cos_135_degree_l0_569

noncomputable def cos_135_degree_is_negative_sqrt2_over_2 : Prop :=
  real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2

theorem proof_cos_135_degree : cos_135_degree_is_negative_sqrt2_over_2 :=
by
  sorry

end proof_cos_135_degree_l0_569


namespace find_a5_l0_121

variable (a : ℕ → ℤ) (d : ℤ)

-- Conditions extracted from the given problem
def arithmetic_sequence := ∀ n : ℕ, a (n + 1) = a n + d -- Defining the arithmetic sequence condition

def sum_first_nine_terms := ∑ i in finset.range 9, a i = 81 -- Defining the sum of the first nine terms

def specific_sum_condition := a 5 + a 6 + a 13 = 171 -- Defining the specific condition about terms a_6, a_7, and a_14

-- Theorem to prove the required answer
theorem find_a5 (h1 : arithmetic_sequence a d) (h2 : sum_first_nine_terms a) (h3 : specific_sum_condition a d ) : a 4 = 9 := by
  sorry

end find_a5_l0_121


namespace find_set_M_l0_238

variable (U M : Set ℕ)
variable [DecidableEq ℕ]

-- Universel set U is {1, 3, 5, 7}
def universal_set : Set ℕ := {1, 3, 5, 7}

-- define the complement C_U M
def complement (U M : Set ℕ) : Set ℕ := U \ M

-- M is the set to find such that complement of M in U is {5, 7}
theorem find_set_M (M : Set ℕ) (h : complement universal_set M = {5, 7}) : M = {1, 3} := by
  sorry

end find_set_M_l0_238


namespace total_bad_vegetables_is_correct_l0_57

-- Define the conditions
def carol_carrots := 29
def carol_cucumbers := 15
def carol_tomatoes := 10

def mom_carrots := 16
def mom_cucumbers := 12
def mom_tomatoes := 14

def carol_good_carrots_percentage := 80 / 100
def carol_good_cucumbers_percentage := 95 / 100
def carol_good_tomatoes_percentage := 90 / 100

def mom_good_carrots_percentage := 85 / 100
def mom_good_cucumbers_percentage := 70 / 100
def mom_good_tomatoes_percentage := 75 / 100

-- Calculate the number of good vegetables for Carol and her mom
noncomputable def carol_good_carrots := (carol_carrots * carol_good_carrots_percentage).floor
noncomputable def carol_good_cucumbers := (carol_cucumbers * carol_good_cucumbers_percentage).floor
noncomputable def carol_good_tomatoes := (carol_tomatoes * carol_good_tomatoes_percentage).floor

noncomputable def mom_good_carrots := (mom_carrots * mom_good_carrots_percentage).floor
noncomputable def mom_good_cucumbers := (mom_cucumbers * mom_good_cucumbers_percentage).floor
noncomputable def mom_good_tomatoes := (mom_tomatoes * mom_good_tomatoes_percentage).floor

-- The total number of bad vegetables for Carol and her mom
noncomputable def carol_bad_vegetables := carol_carrots + carol_cucumbers + carol_tomatoes - carol_good_carrots - carol_good_cucumbers - carol_good_tomatoes
noncomputable def mom_bad_vegetables := mom_carrots + mom_cucumbers + mom_tomatoes - mom_good_carrots - mom_good_cucumbers - mom_good_tomatoes

noncomputable def total_bad_vegetables := carol_bad_vegetables + mom_bad_vegetables

-- The theorem to be proved
theorem total_bad_vegetables_is_correct : total_bad_vegetables = 19 := by
  sorry

end total_bad_vegetables_is_correct_l0_57


namespace exists_zero_in_interval_l0_856

noncomputable def f (x : ℝ) : ℝ :=
  3 / x - Real.log x

theorem exists_zero_in_interval : ∃ c ∈ Ioo 2 3, f c = 0 :=
begin
  sorry
end

end exists_zero_in_interval_l0_856


namespace minimum_value_correct_l0_657

-- Define the quadratic polynomial
def quadratic_expression (x y : ℝ) : ℝ :=
  3 * x ^ 2 + 4 * x * y + 2 * y ^ 2 - 6 * x + 8 * y + 10

-- Define the minimum value to be proved
def minimum_value : ℝ := -13 / 5

-- Define the points at which the minimum value occurs
def points (x y : ℝ) : Prop := x = 13 / 5 ∧ y = -12 / 5

-- Main theorem to prove
theorem minimum_value_correct : ∃ x y : ℝ, 
  quadratic_expression x y = minimum_value ∧ points x y :=
by
  sorry

end minimum_value_correct_l0_657


namespace cos_135_eq_neg_inv_sqrt_2_l0_520

theorem cos_135_eq_neg_inv_sqrt_2 : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_520


namespace cos_135_eq_neg_sqrt2_div_2_l0_510

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = - Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l0_510


namespace cos_135_eq_neg_inv_sqrt2_l0_438

theorem cos_135_eq_neg_inv_sqrt2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt2_l0_438


namespace area_of_triangle_KBN_l0_322

theorem area_of_triangle_KBN 
  {A B K N : Type*} 
  {AK AN : ℝ}
  (hAK : AK = Real.sqrt 5)
  (hAN : AN = 2)
  (h_tan : Real.tan (Real.angle K N A) = Real.sqrt 2 / Real.sqrt 3) :
  ∃ (x : ℝ), 
    S (triangle K B N) = (4 * Real.sqrt 6 * (9 - 4 * Real.sqrt 3)) / 33 := sorry

end area_of_triangle_KBN_l0_322


namespace model1_best_fitting_effect_l0_195

-- Definitions for the correlation coefficients of the models
def R1 : ℝ := 0.98
def R2 : ℝ := 0.80
def R3 : ℝ := 0.50
def R4 : ℝ := 0.25

-- Main theorem stating Model 1 has the best fitting effect
theorem model1_best_fitting_effect : |R1| > |R2| ∧ |R1| > |R3| ∧ |R1| > |R4| :=
by sorry

end model1_best_fitting_effect_l0_195


namespace exists_natural_number_satisfies_eq_l0_639

theorem exists_natural_number_satisfies_eq :
  ∃ n : ℕ, (real.sqrt (17 * real.sqrt 5 + 38) ^ (1 / (n:ℝ)) + real.sqrt (17 * real.sqrt 5 - 38) ^ (1 / (n:ℝ))) = 2 * real.sqrt 5 := 
by
  use 3
  sorry

end exists_natural_number_satisfies_eq_l0_639


namespace length_of_AA₁_l0_820

variables (A A₁ B B₁ C C₁ M : Type)
variables (r : ℝ) (center : A) (radius : ℝ)
variables (AA₁ BB₁ CC₁ : A → A → ℝ) (distance : A → A → ℝ)
variables (ratio : ℝ → ℝ → ℝ)

def sphere_radius_11 := radius = 11
def lines_perpendicular (AA₁ BB₁ CC₁ : A → A → ℝ) :=
  ∀ x y z, AA₁ x y = 0 ∧ BB₁ x z = 0 ∧ CC₁ y z = 0

def intersect_at_M (x y z M : A) (AA₁ BB₁ CC₁ : A → A → ℝ) :=
  AA₁ x M = 0 ∧ BB₁ y M = 0 ∧ CC₁ z M = 0

def distance_from_center := distance M center = sqrt 59
def BB₁_length := BB₁ B B₁ = 18
def ratio_CC₁ := ratio (CC₁ C M) (CC₁ C₁ M) = (8 + sqrt 2) / (8 - sqrt 2)

theorem length_of_AA₁ :
  sphere_radius_11 ∧
  lines_perpendicular AA₁ BB₁ CC₁ ∧
  intersect_at_M A B C M AA₁ BB₁ CC₁ ∧
  distance_from_center ∧
  BB₁_length ∧
  ratio_CC₁ 
  → AA₁ A A₁ = 20 :=
by
  sorry

end length_of_AA₁_l0_820


namespace work_completed_in_30_days_l0_254

theorem work_completed_in_30_days (ravi_days : ℕ) (prakash_days : ℕ)
  (h1 : ravi_days = 50) (h2 : prakash_days = 75) : 
  let ravi_rate := (1 / 50 : ℚ)
  let prakash_rate := (1 / 75 : ℚ)
  let combined_rate := ravi_rate + prakash_rate
  let days_to_complete := 1 / combined_rate
  days_to_complete = 30 := by
  sorry

end work_completed_in_30_days_l0_254


namespace binom_26_6_l0_693

theorem binom_26_6 (h₁ : Nat.choose 25 5 = 53130) (h₂ : Nat.choose 25 6 = 177100) :
  Nat.choose 26 6 = 230230 :=
by
  sorry

end binom_26_6_l0_693


namespace seq_solution_l0_676

-- Definitions: Define the sequence {a_n} according to the given conditions
def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 0 ∧ ∀ n ≥ 2, a n - 2 * a (n - 1) = n ^ 2 - 3

-- Main statement: Prove that for all n, the sequence satisfies the derived formula
theorem seq_solution (a : ℕ → ℤ) (h : seq a) : ∀ n, a n = 2 ^ (n + 2) - n ^ 2 - 4 * n - 3 :=
sorry

end seq_solution_l0_676


namespace cos_135_eq_neg_inv_sqrt_2_l0_519

theorem cos_135_eq_neg_inv_sqrt_2 : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l0_519


namespace number_of_women_l0_419

theorem number_of_women (M W : ℕ) (h1 : M = 16) (h2 : ∀ m, m < M → ∃! w, w < W ∧ (danced_with w m)) 
(h3 : ∀ w, w < W → ∃! m1 m2 m3 m4, ∀ i j, i < j → (danced_with w m1 ∧ danced_with w m2 ∧ danced_with w m3 ∧ danced_with w m4) 
∧ dist m1 m2 ∧ dist m1 m3 ∧ dist m1 m4 ∧ dist m2 m3 ∧ dist m2 m4 ∧ dist m3 m4) : W = 8 := 
by {
  sorry
}

end number_of_women_l0_419


namespace cosine_135_eq_neg_sqrt_2_div_2_l0_534

theorem cosine_135_eq_neg_sqrt_2_div_2 (cos180 : Real := -1) (cos90 : Real := 0) (cos45 : Real := Real.sqrt 2 / 2)
    (theta1 : Real := 180) (theta2 : Real := 45) (theta3 : Real := 135) :
    theta3 = theta1 - theta2 ∧ 
    Real.cos theta1 = cos180 ∧ 
    Real.cos theta2 = cos45 → 
    Real.cos theta3 = -cos45 := by
  sorry

end cosine_135_eq_neg_sqrt_2_div_2_l0_534


namespace positive_difference_of_two_numbers_l0_933

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l0_933


namespace monotonically_increasing_iff_l0_866

noncomputable def f (x : ℝ) (a : ℝ) := x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x

theorem monotonically_increasing_iff (a : ℝ) : 
  (∀ x y : ℝ, x < y → f x a ≤ f y a) ↔ (-1 / 3 : ℝ) ≤ a ∧ a ≤ (1 / 3 : ℝ) := 
sorry

end monotonically_increasing_iff_l0_866


namespace pyramid_volume_surface_area_l0_860

variables (a l : ℝ)
noncomputable def V : ℝ := (1/3) * a^2 * (l / Real.sqrt 2)
noncomputable def S : ℝ := a^2 + a * Real.sqrt 2 * l

theorem pyramid_volume_surface_area (a l : ℝ) :
  let V := (1/3) * a^2 * (l / Real.sqrt 2)
  let S := a^2 + a * Real.sqrt 2 * l
  ∃ V S : ℝ, V = (1/3) * a^2 * (l / Real.sqrt 2) ∧ S = a^2 + a * Real.sqrt 2 * l :=
begin
  use [(1/3) * a^2 * (l / Real.sqrt 2), a^2 + a * Real.sqrt 2 * l],
  split; refl,
end

end pyramid_volume_surface_area_l0_860
