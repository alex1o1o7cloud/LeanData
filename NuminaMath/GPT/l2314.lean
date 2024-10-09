import Mathlib

namespace heart_ratio_correct_l2314_231464

def heart (n m : ℕ) : ℕ := n^3 + m^2

theorem heart_ratio_correct : (heart 3 5 : ℚ) / (heart 5 3) = 26 / 67 :=
by
  sorry

end heart_ratio_correct_l2314_231464


namespace checkerboard_probability_not_on_perimeter_l2314_231428

def total_squares : ℕ := 81

def perimeter_squares : ℕ := 32

def non_perimeter_squares : ℕ := total_squares - perimeter_squares

noncomputable def probability_not_on_perimeter : ℚ := non_perimeter_squares / total_squares

theorem checkerboard_probability_not_on_perimeter :
  probability_not_on_perimeter = 49 / 81 :=
by
  sorry

end checkerboard_probability_not_on_perimeter_l2314_231428


namespace a_5_eq_31_l2314_231413

def seq (a : ℕ → ℕ) : Prop :=
  (a 1 = 1) ∧ (∀ n, a (n + 1) = 2 * a n + 1)

theorem a_5_eq_31 (a : ℕ → ℕ) (h : seq a) : a 5 = 31 :=
by
  sorry
 
end a_5_eq_31_l2314_231413


namespace complement_B_in_A_l2314_231442

noncomputable def A : Set ℝ := {x | x < 2}
noncomputable def B : Set ℝ := {x | 1 < x ∧ x < 2}

theorem complement_B_in_A : {x | x ∈ A ∧ x ∉ B} = {x | x ≤ 1} :=
by
  sorry

end complement_B_in_A_l2314_231442


namespace valid_decomposition_2009_l2314_231496

/-- A definition to determine whether a number can be decomposed
    into sums of distinct numbers with repeated digits representation. -/
def decomposable_2009 (n : ℕ) : Prop :=
  ∃ a b c d : ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧
  a = 1111 ∧ b = 777 ∧ c = 66 ∧ d = 55 ∧ a + b + c + d = n

theorem valid_decomposition_2009 :
  decomposable_2009 2009 :=
sorry

end valid_decomposition_2009_l2314_231496


namespace find_13_real_coins_find_15_real_coins_cannot_find_17_real_coins_l2314_231409

noncomputable def board : Type := (Fin 5) × (Fin 5)

def is_counterfeit (c1 : board) (c2 : board) : Prop :=
  (c1.1 = c2.1 ∧ (c1.2 = c2.2 + 1 ∨ c1.2 + 1 = c2.2)) ∨
  (c1.2 = c2.2 ∧ (c1.1 = c2.1 + 1 ∨ c1.1 + 1 = c2.1))

theorem find_13_real_coins (coins : board → ℝ) (c1 c2 : board) :
  (coins c1 < coins (0,0) ∧ coins c2 < coins (0,0)) ∧ is_counterfeit c1 c2 →
  ∃ C : Finset board, C.card = 13 ∧ ∀ c ∈ C, coins c = coins (0,0) :=
sorry

theorem find_15_real_coins (coins : board → ℝ) (c1 c2 : board) :
  (coins c1 < coins (0,0) ∧ coins c2 < coins (0,0)) ∧ is_counterfeit c1 c2 →
  ∃ C : Finset board, C.card = 15 ∧ ∀ c ∈ C, coins c = coins (0,0) :=
sorry

theorem cannot_find_17_real_coins (coins : board → ℝ) (c1 c2 : board) :
  (coins c1 < coins (0,0) ∧ coins c2 < coins (0,0)) ∧ is_counterfeit c1 c2 →
  ¬ (∃ C : Finset board, C.card = 17 ∧ ∀ c ∈ C, coins c = coins (0,0)) :=
sorry

end find_13_real_coins_find_15_real_coins_cannot_find_17_real_coins_l2314_231409


namespace general_term_formula_l2314_231445

-- Define the given sequence as a function
def seq (n : ℕ) : ℤ :=
  match n with
  | 0 => 3
  | n + 1 => if (n % 2 = 0) then 4 * (n + 1) - 1 else -(4 * (n + 1) - 1)

-- Define the proposed general term formula
def a_n (n : ℕ) : ℤ :=
  (-1)^(n+1) * (4 * n - 1)

-- State the theorem that general term of the sequence equals the proposed formula
theorem general_term_formula : ∀ n : ℕ, seq n = a_n n := 
by
  sorry

end general_term_formula_l2314_231445


namespace sum_black_cells_even_l2314_231465

-- Define a rectangular board with cells colored in a chess manner.

structure ChessBoard (m n : ℕ) :=
  (cells : Fin m → Fin n → Int)
  (row_sums_even : ∀ i : Fin m, (Finset.univ.sum (λ j => cells i j)) % 2 = 0)
  (column_sums_even : ∀ j : Fin n, (Finset.univ.sum (λ i => cells i j)) % 2 = 0)

def is_black_cell (i j : ℕ) : Bool :=
  (i + j) % 2 = 0

theorem sum_black_cells_even {m n : ℕ} (B : ChessBoard m n) :
    (Finset.univ.sum (λ (i : Fin m) =>
         Finset.univ.sum (λ (j : Fin n) =>
            if (is_black_cell i.val j.val) then B.cells i j else 0))) % 2 = 0 :=
by
  sorry

end sum_black_cells_even_l2314_231465


namespace nested_fraction_simplifies_l2314_231419

theorem nested_fraction_simplifies : 
  (1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))) = 8 / 21 := 
by 
  sorry

end nested_fraction_simplifies_l2314_231419


namespace length_of_bridge_l2314_231477

/-- A train that is 357 meters long is running at a speed of 42 km/hour. 
    It takes 42.34285714285714 seconds to pass a bridge. 
    Prove that the length of the bridge is 136.7142857142857 meters. -/
theorem length_of_bridge : 
  let train_length := 357 -- meters
  let speed_kmh := 42 -- km/hour
  let passing_time := 42.34285714285714 -- seconds
  let speed_mps := 42 * (1000 / 3600) -- meters/second
  let total_distance := speed_mps * passing_time -- meters
  let bridge_length := total_distance - train_length -- meters
  bridge_length = 136.7142857142857 :=
by
  sorry

end length_of_bridge_l2314_231477


namespace pairs_count_l2314_231485

theorem pairs_count (A B : Set ℕ) (h1 : A ∪ B = {1, 2, 3, 4, 5}) (h2 : 3 ∈ A ∩ B) : 
  Nat.card {p : Set ℕ × Set ℕ | p.1 ∪ p.2 = {1, 2, 3, 4, 5} ∧ 3 ∈ p.1 ∩ p.2} = 81 := by
  sorry

end pairs_count_l2314_231485


namespace find_line_equation_l2314_231481

theorem find_line_equation (k : ℝ) (x y : ℝ) :
  (∀ k, (∃ x y, y = k * x + 1 ∧ x^2 + y^2 - 2 * x - 3 = 0) ↔ x - y + 1 = 0) :=
by
  sorry

end find_line_equation_l2314_231481


namespace center_distance_correct_l2314_231401

noncomputable def ball_diameter : ℝ := 6
noncomputable def ball_radius : ℝ := ball_diameter / 2
noncomputable def R₁ : ℝ := 150
noncomputable def R₂ : ℝ := 50
noncomputable def R₃ : ℝ := 90
noncomputable def R₄ : ℝ := 120
noncomputable def elevation : ℝ := 4

noncomputable def adjusted_R₁ : ℝ := R₁ - ball_radius
noncomputable def adjusted_R₂ : ℝ := R₂ + ball_radius + elevation
noncomputable def adjusted_R₃ : ℝ := R₃ - ball_radius
noncomputable def adjusted_R₄ : ℝ := R₄ - ball_radius

noncomputable def distance_R₁ : ℝ := 1/2 * 2 * Real.pi * adjusted_R₁
noncomputable def distance_R₂ : ℝ := 1/2 * 2 * Real.pi * adjusted_R₂
noncomputable def distance_R₃ : ℝ := 1/2 * 2 * Real.pi * adjusted_R₃
noncomputable def distance_R₄ : ℝ := 1/2 * 2 * Real.pi * adjusted_R₄

noncomputable def total_distance : ℝ := distance_R₁ + distance_R₂ + distance_R₃ + distance_R₄

theorem center_distance_correct : total_distance = 408 * Real.pi := 
  by
  sorry

end center_distance_correct_l2314_231401


namespace polynomial_problem_l2314_231468

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := 2 * x + 4

theorem polynomial_problem (f_nonzero : ∀ x, f x ≠ 0) 
  (h1 : ∀ x, f (g x) = f x * g x)
  (h2 : g 3 = 10)
  (h3 : ∃ a b, g x = a * x + b) :
  g x = 2 * x + 4 :=
sorry

end polynomial_problem_l2314_231468


namespace proof_problem_l2314_231447

-- Definitions needed for conditions
def a := -7 / 4
def b := -2 / 3
def m : ℚ := 1  -- m can be any rational number
def n : ℚ := -m  -- since m and n are opposite numbers

-- Lean statement to prove the given problem
theorem proof_problem : 4 * a / b + 3 * (m + n) = 21 / 2 := by
  -- Definitions ensuring a, b, m, n meet the conditions
  have habs : |a| = 7 / 4 := by sorry
  have brecip : 1 / b = -3 / 2 := by sorry
  have moppos : m + n = 0 := by sorry
  sorry

end proof_problem_l2314_231447


namespace solve_quadratics_l2314_231405

theorem solve_quadratics (x : ℝ) :
  (x^2 - 7 * x - 18 = 0 → x = 9 ∨ x = -2) ∧
  (4 * x^2 + 1 = 4 * x → x = 1/2) :=
by
  sorry

end solve_quadratics_l2314_231405


namespace max_tiles_to_spell_CMWMC_l2314_231412

theorem max_tiles_to_spell_CMWMC {Cs Ms Ws : ℕ} (hC : Cs = 8) (hM : Ms = 8) (hW : Ws = 8) : 
  ∃ (max_draws : ℕ), max_draws = 18 :=
by
  -- Assuming we have 8 C's, 8 M's, and 8 W's in the bag
  sorry

end max_tiles_to_spell_CMWMC_l2314_231412


namespace rectangular_to_cylindrical_l2314_231433

theorem rectangular_to_cylindrical (x y z : ℝ) (r θ : ℝ) (h1 : x = -3) (h2 : y = 4) (h3 : z = 5) (h4 : r = 5) (h5 : θ = Real.pi - Real.arctan (4 / 3)) :
  (r, θ, z) = (5, Real.pi - Real.arctan (4 / 3), 5) :=
by
  sorry

end rectangular_to_cylindrical_l2314_231433


namespace units_digit_sum_l2314_231408

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum
  (h1 : units_digit 13 = 3)
  (h2 : units_digit 41 = 1)
  (h3 : units_digit 27 = 7)
  (h4 : units_digit 34 = 4) :
  units_digit ((13 * 41) + (27 * 34)) = 1 :=
by
  sorry

end units_digit_sum_l2314_231408


namespace mike_picked_12_pears_l2314_231450

theorem mike_picked_12_pears (k_picked k_gave_away k_m_together k_left m_left : ℕ) 
  (hkp : k_picked = 47) 
  (hkg : k_gave_away = 46) 
  (hkt : k_m_together = 13)
  (hkl : k_left = k_picked - k_gave_away) 
  (hlt : k_m_left = k_left + m_left) : 
  m_left = 12 := by
  sorry

end mike_picked_12_pears_l2314_231450


namespace students_arrangement_count_l2314_231494

theorem students_arrangement_count : 
  let total_permutations := Nat.factorial 5
  let a_first_permutations := Nat.factorial 4
  let b_last_permutations := Nat.factorial 4
  let both_permutations := Nat.factorial 3
  total_permutations - a_first_permutations - b_last_permutations + both_permutations = 78 :=
by
  sorry

end students_arrangement_count_l2314_231494


namespace flora_needs_more_daily_l2314_231444

-- Definitions based on conditions
def totalMilk : ℕ := 105   -- Total milk requirement in gallons
def weeks : ℕ := 3         -- Total weeks
def daysInWeek : ℕ := 7    -- Days per week
def floraPlan : ℕ := 3     -- Flora's planned gallons per day

-- Proof statement
theorem flora_needs_more_daily : (totalMilk / (weeks * daysInWeek)) - floraPlan = 2 := 
by
  sorry

end flora_needs_more_daily_l2314_231444


namespace message_spread_in_24_hours_l2314_231434

theorem message_spread_in_24_hours : ∃ T : ℕ, (T = (2^25 - 1)) :=
by 
  let T := 2^24 - 1
  use T
  sorry

end message_spread_in_24_hours_l2314_231434


namespace line_through_ellipse_and_midpoint_l2314_231479

theorem line_through_ellipse_and_midpoint :
  ∃ l : ℝ → ℝ → Prop,
    (∀ (x y : ℝ), l x y ↔ (x + y) = 0) ∧
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      (x₁ + x₂ = 2 ∧ y₁ + y₂ = 1) ∧
      (x₁^2 / 2 + y₁^2 = 1 ∧ x₂^2 / 2 + y₂^2 = 1) ∧
      l x₁ y₁ ∧ l x₂ y₂ ∧
      ∀ (mx my : ℝ), (mx, my) = (1, 0.5) → (mx = (x₁ + x₂) / 2 ∧ my = (y₁ + y₂) / 2))
  := sorry

end line_through_ellipse_and_midpoint_l2314_231479


namespace find_weight_of_a_l2314_231487

theorem find_weight_of_a (A B C D E : ℕ) 
  (h1 : A + B + C = 3 * 84)
  (h2 : A + B + C + D = 4 * 80)
  (h3 : E = D + 3)
  (h4 : B + C + D + E = 4 * 79) : 
  A = 75 := by
  sorry

end find_weight_of_a_l2314_231487


namespace coffee_mix_price_l2314_231439

theorem coffee_mix_price 
  (P : ℝ)
  (pound_2nd : ℝ := 2.45)
  (total_pounds : ℝ := 18)
  (final_price_per_pound : ℝ := 2.30)
  (pounds_each_kind : ℝ := 9) :
  9 * P + 9 * pound_2nd = total_pounds * final_price_per_pound →
  P = 2.15 :=
by
  intros h
  sorry

end coffee_mix_price_l2314_231439


namespace celine_buys_two_laptops_l2314_231495

variable (number_of_laptops : ℕ)
variable (laptop_cost : ℕ := 600)
variable (smartphone_cost : ℕ := 400)
variable (number_of_smartphones : ℕ := 4)
variable (total_money_spent : ℕ := 3000)
variable (change_back : ℕ := 200)

def total_spent : ℕ := total_money_spent - change_back

def cost_of_laptops (n : ℕ) : ℕ := n * laptop_cost

def cost_of_smartphones (n : ℕ) : ℕ := n * smartphone_cost

theorem celine_buys_two_laptops :
  cost_of_laptops number_of_laptops + cost_of_smartphones number_of_smartphones = total_spent →
  number_of_laptops = 2 := by
  sorry

end celine_buys_two_laptops_l2314_231495


namespace diminish_to_divisible_l2314_231461

-- Definitions based on conditions
def LCM (a b : ℕ) : ℕ := Nat.lcm a b
def numbers : List ℕ := [12, 16, 18, 21, 28]
def lcm_numbers : ℕ := List.foldr LCM 1 numbers
def n : ℕ := 1011
def x : ℕ := 3

-- The proof problem statement
theorem diminish_to_divisible :
  ∃ x : ℕ, n - x = lcm_numbers := sorry

end diminish_to_divisible_l2314_231461


namespace total_flowers_l2314_231473

def number_of_pots : ℕ := 141
def flowers_per_pot : ℕ := 71

theorem total_flowers : number_of_pots * flowers_per_pot = 10011 :=
by
  -- formal proof goes here
  sorry

end total_flowers_l2314_231473


namespace andrew_calculation_l2314_231493

theorem andrew_calculation (x y : ℝ) (hx : x ≠ 0) :
  0.4 * 0.5 * x = 0.2 * 0.3 * y → y = (10 / 3) * x :=
by
  sorry

end andrew_calculation_l2314_231493


namespace total_candies_third_set_l2314_231475

-- Variables for the quantities of candies
variables (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)

-- Conditions as stated in the problem
axiom equal_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3
axiom first_set_chocolates_gummy : S1 = M1
axiom first_set_hard_candies : L1 = S1 + 7
axiom second_set_hard_chocolates : L2 = S2
axiom second_set_gummy_candies : M2 = L2 - 15
axiom third_set_no_hard_candies : L3 = 0

-- Theorem to be proven: the number of candies in the third set
theorem total_candies_third_set : 
  L3 + S3 + M3 = 29 :=
by
  sorry

end total_candies_third_set_l2314_231475


namespace complex_addition_result_l2314_231454

theorem complex_addition_result (a b : ℝ) (i : ℂ) 
  (h1 : i * i = -1)
  (h2 : a + b * i = (1 - i) * (2 + i)) : a + b = 2 :=
sorry

end complex_addition_result_l2314_231454


namespace tangerines_in_basket_l2314_231414

/-- Let n be the initial number of tangerines in the basket. -/
theorem tangerines_in_basket
  (n : ℕ)
  (c1 : ∃ m : ℕ, m = 10) -- Minyoung ate 10 tangerines from the basket initially
  (c2 : ∃ k : ℕ, k = 6)  -- An hour later, Minyoung ate 6 more tangerines
  (c3 : n = 10 + 6)      -- The basket was empty after these were eaten
  : n = 16 := sorry

end tangerines_in_basket_l2314_231414


namespace candy_eaten_l2314_231407

/--
Given:
- Faye initially had 47 pieces of candy
- Faye ate x pieces the first night
- Faye's sister gave her 40 more pieces
- Now Faye has 62 pieces of candy

We need to prove:
- Faye ate 25 pieces of candy the first night.
-/
theorem candy_eaten (x : ℕ) (h1 : 47 - x + 40 = 62) : x = 25 :=
by
  sorry

end candy_eaten_l2314_231407


namespace triangle_largest_angle_l2314_231470

theorem triangle_largest_angle (A B C : ℚ) (sinA sinB sinC : ℚ) 
(h_ratio : sinA / sinB = 3 / 5)
(h_ratio2 : sinB / sinC = 5 / 7)
(h_sum : A + B + C = 180) : C = 120 := 
sorry

end triangle_largest_angle_l2314_231470


namespace addition_terms_correct_l2314_231406

def first_seq (n : ℕ) : ℕ := 2 * n + 1
def second_seq (n : ℕ) : ℕ := 5 * n - 1

theorem addition_terms_correct :
  first_seq 10 = 21 ∧ second_seq 10 = 49 ∧
  first_seq 80 = 161 ∧ second_seq 80 = 399 :=
by
  sorry

end addition_terms_correct_l2314_231406


namespace cars_sold_proof_l2314_231458

noncomputable def total_cars_sold : Nat := 300
noncomputable def perc_audi : ℝ := 0.10
noncomputable def perc_toyota : ℝ := 0.15
noncomputable def perc_acura : ℝ := 0.20
noncomputable def perc_honda : ℝ := 0.18

theorem cars_sold_proof : total_cars_sold * (1 - (perc_audi + perc_toyota + perc_acura + perc_honda)) = 111 := by
  sorry

end cars_sold_proof_l2314_231458


namespace length_of_each_brick_l2314_231448

theorem length_of_each_brick (wall_length wall_height wall_thickness : ℝ) (brick_length brick_width brick_height : ℝ) (num_bricks_used : ℝ) 
  (h1 : wall_length = 8) 
  (h2 : wall_height = 6) 
  (h3 : wall_thickness = 0.02) 
  (h4 : brick_length = 0.11) 
  (h5 : brick_width = 0.05) 
  (h6 : brick_height = 0.06) 
  (h7 : num_bricks_used = 2909.090909090909) : 
  brick_length = 0.11 :=
by
  -- variables and assumptions
  have vol_wall : ℝ := wall_length * wall_height * wall_thickness
  have vol_brick : ℝ := brick_length * brick_width * brick_height
  have calc_bricks : ℝ := vol_wall / vol_brick
  -- skipping proof
  sorry

end length_of_each_brick_l2314_231448


namespace find_angle_C_find_sin_A_plus_sin_B_l2314_231432

open Real

namespace TriangleProblem

variables (a b c : ℝ) (A B C : ℝ)

def sides_opposite_angles (a b c : ℝ) (A B C : ℝ) : Prop :=
  c^2 = a^2 + b^2 + a * b

def given_c (c : ℝ) : Prop :=
  c = 4 * sqrt 7

def perimeter (a b c : ℝ) : Prop :=
  a + b + c = 12 + 4 * sqrt 7

theorem find_angle_C (a b c A B C : ℝ)
  (h1 : sides_opposite_angles a b c A B C) : 
  C = 2 * pi / 3 :=
sorry

theorem find_sin_A_plus_sin_B (a b c A B C : ℝ)
  (h1 : sides_opposite_angles a b c A B C)
  (h2 : given_c c)
  (h3 : perimeter a b c) : 
  sin A + sin B = 3 * sqrt 21 / 28 :=
sorry

end TriangleProblem

end find_angle_C_find_sin_A_plus_sin_B_l2314_231432


namespace ashok_average_marks_l2314_231403

theorem ashok_average_marks (avg_6 : ℝ) (marks_6 : ℝ) (total_sub : ℕ) (sub_6 : ℕ)
  (h1 : avg_6 = 75) (h2 : marks_6 = 80) (h3 : total_sub = 6) (h4 : sub_6 = 5) :
  (avg_6 * total_sub - marks_6) / sub_6 = 74 :=
by
  sorry

end ashok_average_marks_l2314_231403


namespace Tom_age_problem_l2314_231423

theorem Tom_age_problem 
  (T : ℝ) 
  (h1 : T = T1 + T2 + T3 + T4) 
  (h2 : T - 3 = 3 * (T - 3 - 3 - 3 - 3)) : 
  T / 3 = 5.5 :=
by 
  -- sorry here to skip the proof
  sorry

end Tom_age_problem_l2314_231423


namespace product_calc_l2314_231411

theorem product_calc : (16 * 0.5 * 4 * 0.125 = 4) :=
by
  sorry

end product_calc_l2314_231411


namespace find_k_l2314_231462

noncomputable section

open Polynomial

-- Define the conditions
variables (h k : Polynomial ℚ)
variables (C : k.eval (-1) = 15) (H : h.comp k = h * k) (nonzero_h : h ≠ 0)

-- The goal is to prove k(x) = x^2 + 21x - 35
theorem find_k : k = X^2 + 21 * X - 35 :=
  by sorry

end find_k_l2314_231462


namespace audio_cassettes_in_first_set_l2314_231497

theorem audio_cassettes_in_first_set (A V : ℝ) (num_audio_cassettes : ℝ) : 
  (V = 300) → (A * num_audio_cassettes + 3 * V = 1110) → (5 * A + 4 * V = 1350) → (A = 30) → (num_audio_cassettes = 7) := 
by
  intros hV hCond1 hCond2 hA
  sorry

end audio_cassettes_in_first_set_l2314_231497


namespace prism_faces_vertices_l2314_231498

theorem prism_faces_vertices {L E F V : ℕ} (hE : E = 21) (hEdges : E = 3 * L) 
    (hF : F = L + 2) (hV : V = L) : F = 9 ∧ V = 7 :=
by
  sorry

end prism_faces_vertices_l2314_231498


namespace seven_distinct_numbers_with_reversed_digits_l2314_231441

theorem seven_distinct_numbers_with_reversed_digits (x y : ℕ) :
  (∃ a b c d e f g : ℕ, 
  (10 * a + b + 18 = 10 * b + a) ∧ (10 * c + d + 18 = 10 * d + c) ∧ 
  (10 * e + f + 18 = 10 * f + e) ∧ (10 * g + y + 18 = 10 * y + g) ∧ 
  a ≠ c ∧ a ≠ e ∧ a ≠ g ∧ 
  c ≠ e ∧ c ≠ g ∧ 
  e ≠ g ∧ 
  (1 ≤ a ∧ a ≤ 9) ∧ (1 ≤ b ∧ b ≤ 9) ∧
  (1 ≤ c ∧ c ≤ 9) ∧ (1 ≤ d ∧ d ≤ 9) ∧
  (1 ≤ e ∧ e <= 9) ∧ (1 ≤ f ∧ f <= 9) ∧
  (1 ≤ g ∧ g <= 9) ∧ (1 ≤ y ∧ y <= 9)) :=
sorry

end seven_distinct_numbers_with_reversed_digits_l2314_231441


namespace find_angle_l2314_231490

theorem find_angle {x : ℝ} (h1 : ∀ i, 1 ≤ i ∧ i ≤ 9 → ∃ x, x > 0) (h2 : 9 * x = 900) : x = 100 :=
  sorry

end find_angle_l2314_231490


namespace mike_typing_time_l2314_231472

-- Definitions based on the given conditions
def original_speed : ℕ := 65
def speed_reduction : ℕ := 20
def document_words : ℕ := 810
def reduced_speed : ℕ := original_speed - speed_reduction

-- The statement to prove
theorem mike_typing_time : (document_words / reduced_speed) = 18 :=
  by
    sorry

end mike_typing_time_l2314_231472


namespace same_percentage_loss_as_profit_l2314_231424

theorem same_percentage_loss_as_profit (CP SP L : ℝ) (h_prof : SP = 1720)
  (h_loss : L = CP - (14.67 / 100) * CP)
  (h_25_prof : 1.25 * CP = 1875) :
  L = 1280 := 
  sorry

end same_percentage_loss_as_profit_l2314_231424


namespace find_time_period_l2314_231489

theorem find_time_period (P r CI : ℝ) (n : ℕ) (A : ℝ) (t : ℝ) 
  (hP : P = 10000)
  (hr : r = 0.15)
  (hCI : CI = 3886.25)
  (hn : n = 1)
  (hA : A = P + CI)
  (h_formula : A = P * (1 + r / n) ^ (n * t)) : 
  t = 2 := 
  sorry

end find_time_period_l2314_231489


namespace negation_of_universal_proposition_l2314_231436

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x ≥ 0) ↔ ∃ x : ℝ, x > 0 ∧ x^2 + x < 0 :=
by sorry

end negation_of_universal_proposition_l2314_231436


namespace total_stairs_climbed_l2314_231452

theorem total_stairs_climbed (samir_stairs veronica_stairs ravi_stairs total_stairs_climbed : ℕ) 
  (h_samir : samir_stairs = 318)
  (h_veronica : veronica_stairs = (318 / 2) + 18)
  (h_ravi : ravi_stairs = (3 * veronica_stairs) / 2) :
  samir_stairs + veronica_stairs + ravi_stairs = total_stairs_climbed ->
  total_stairs_climbed = 761 :=
by
  sorry

end total_stairs_climbed_l2314_231452


namespace problem_statement_l2314_231427

theorem problem_statement : (1021 ^ 1022) % 1023 = 4 := 
by
  sorry

end problem_statement_l2314_231427


namespace track_circumference_l2314_231492

theorem track_circumference (A_speed B_speed : ℝ) (y : ℝ) (c : ℝ)
  (A_initial B_initial : ℝ := 0)
  (B_meeting_distance_A_first_meeting : ℝ := 150)
  (A_meeting_distance_B_second_meeting : ℝ := y - 150)
  (A_second_distance : ℝ := 2 * y - 90)
  (B_second_distance : ℝ := y + 90) 
  (first_meeting_eq : B_meeting_distance_A_first_meeting = 150)
  (second_meeting_eq : A_second_distance + 90 = 2 * y)
  (uniform_speed : A_speed / B_speed = (y + 90)/(2 * y - 90)) :
  c = 2 * y → c = 720 :=
by
  sorry

end track_circumference_l2314_231492


namespace probability_exactly_one_solves_problem_l2314_231431

-- Define the context in which A and B solve the problem with given probabilities.
variables (p1 p2 : ℝ)

-- Define the constraint that the probabilities are between 0 and 1
axiom prob_A_nonneg : 0 ≤ p1
axiom prob_A_le_one : p1 ≤ 1
axiom prob_B_nonneg : 0 ≤ p2
axiom prob_B_le_one : p2 ≤ 1

-- Define the context that A and B solve the problem independently.
axiom A_and_B_independent : true

-- The theorem statement to prove the desired probability of exactly one solving the problem.
theorem probability_exactly_one_solves_problem : (p1 * (1 - p2) + p2 * (1 - p1)) =  p1 * (1 - p2) + p2 * (1 - p1) :=
by
  sorry

end probability_exactly_one_solves_problem_l2314_231431


namespace line_equation_l2314_231435

theorem line_equation {k b : ℝ} 
  (h1 : (∀ x : ℝ, k * x + b = -4 * x + 2023 → k = -4))
  (h2 : b = -5) :
  ∀ x : ℝ, k * x + b = -4 * x - 5 := by
sorry

end line_equation_l2314_231435


namespace inequality_always_true_l2314_231446

theorem inequality_always_true (a : ℝ) :
  (∀ x : ℝ, ax^2 - x + 1 > 0) ↔ a > 1/4 :=
sorry

end inequality_always_true_l2314_231446


namespace units_digit_of_product_l2314_231426

theorem units_digit_of_product : 
  (3 ^ 401 * 7 ^ 402 * 23 ^ 403) % 10 = 9 := 
by
  sorry

end units_digit_of_product_l2314_231426


namespace dishonest_shopkeeper_weight_l2314_231482

noncomputable def weight_used (gain_percent : ℝ) (correct_weight : ℝ) : ℝ :=
  correct_weight / (1 + gain_percent / 100)

theorem dishonest_shopkeeper_weight :
  weight_used 5.263157894736836 1000 = 950 := 
by
  sorry

end dishonest_shopkeeper_weight_l2314_231482


namespace intersection_A_B_l2314_231466

def setA : Set ℝ := { x | |x| < 2 }
def setB : Set ℝ := { x | x^2 - 4 * x + 3 < 0 }
def setC : Set ℝ := { x | 1 < x ∧ x < 2 }

theorem intersection_A_B : setA ∩ setB = setC := by
  sorry

end intersection_A_B_l2314_231466


namespace problem1_problem2_problem3_l2314_231429

def point : Type := (ℝ × ℝ)
def vec (p1 p2 : point) : point := (p2.1 - p1.1, p2.2 - p1.2)

noncomputable def A : point := (-2, 4)
noncomputable def B : point := (3, -1)
noncomputable def C : point := (-3, -4)

noncomputable def a : point := vec A B
noncomputable def b : point := vec B C
noncomputable def c : point := vec C A

-- Problem 1
theorem problem1 : (3 * a.1 + b.1 - 3 * c.1, 3 * a.2 + b.2 - 3 * c.2) = (6, -42) :=
sorry

-- Problem 2
theorem problem2 : ∃ m n : ℝ, a = (m * b.1 + n * c.1, m * b.2 + n * c.2) ∧ m = -1 ∧ n = -1 :=
sorry

-- Helper function for point addition
def add_point (p1 p2 : point) : point := (p1.1 + p2.1, p1.2 + p2.2)
def scale_point (k : ℝ) (p : point) : point := (k * p.1, k * p.2)

-- problem 3
noncomputable def M : point := add_point (scale_point 3 c) C
noncomputable def N : point := add_point (scale_point (-2) b) C

theorem problem3 : M = (0, 20) ∧ N = (9, 2) ∧ vec M N = (9, -18) :=
sorry

end problem1_problem2_problem3_l2314_231429


namespace barrel_tank_ratio_l2314_231471

theorem barrel_tank_ratio
  (B T : ℝ)
  (h1 : (3 / 4) * B = (5 / 8) * T) :
  B / T = 5 / 6 :=
sorry

end barrel_tank_ratio_l2314_231471


namespace polygon_triangle_even_l2314_231459

theorem polygon_triangle_even (n m : ℕ) (h : (3 * m - n) % 2 = 0) : (m + n) % 2 = 0 :=
sorry

noncomputable def number_of_distinct_interior_sides (n m : ℕ) : ℕ :=
(3 * m - n) / 2

noncomputable def number_of_distinct_interior_vertices (n m : ℕ) : ℕ :=
(m - n + 2) / 2

end polygon_triangle_even_l2314_231459


namespace quadratic_inequality_solution_set_l2314_231484

theorem quadratic_inequality_solution_set :
  (∃ x : ℝ, 2 * x + 3 - x^2 > 0) ↔ (-1 < x ∧ x < 3) :=
sorry

end quadratic_inequality_solution_set_l2314_231484


namespace P_sufficient_but_not_necessary_for_Q_l2314_231400

-- Define the propositions P and Q
def P (x : ℝ) : Prop := |x - 2| ≤ 3
def Q (x : ℝ) : Prop := x ≥ -1 ∨ x ≤ 5

-- Define the statement to prove
theorem P_sufficient_but_not_necessary_for_Q :
  (∀ x : ℝ, P x → Q x) ∧ (∃ x : ℝ, Q x ∧ ¬ P x) :=
by
  sorry

end P_sufficient_but_not_necessary_for_Q_l2314_231400


namespace age_difference_ratio_l2314_231457

theorem age_difference_ratio (R J K : ℕ) 
  (h1 : R = J + 8)
  (h2 : R + 2 = 2 * (J + 2))
  (h3 : (R + 2) * (K + 2) = 192) :
  (R - J) / (R - K) = 2 := by
  sorry

end age_difference_ratio_l2314_231457


namespace problem_statement_l2314_231438

def A := {x : ℝ | x * (x - 1) < 0}
def B := {y : ℝ | ∃ x : ℝ, y = x^2}

theorem problem_statement : A ⊆ {y : ℝ | y ≥ 0} :=
sorry

end problem_statement_l2314_231438


namespace donovan_correct_answers_l2314_231455

variable (C : ℝ)
variable (incorrectAnswers : ℝ := 13)
variable (percentageCorrect : ℝ := 0.7292)

theorem donovan_correct_answers :
  (C / (C + incorrectAnswers)) = percentageCorrect → C = 35 := by
  sorry

end donovan_correct_answers_l2314_231455


namespace greatest_value_of_a_plus_b_l2314_231499

-- Definition of the problem conditions
def is_pos_int (n : ℕ) := n > 0

-- Lean statement to prove the greatest possible value of a + b
theorem greatest_value_of_a_plus_b :
  ∃ a b : ℕ, is_pos_int a ∧ is_pos_int b ∧ (1 / (a : ℝ) + 1 / (b : ℝ) = 1 / 9) ∧ a + b = 100 :=
sorry  -- Proof omitted

end greatest_value_of_a_plus_b_l2314_231499


namespace largest_in_set_average_11_l2314_231483

theorem largest_in_set_average_11 :
  ∃ (a_1 a_2 a_3 a_4 a_5 : ℕ), (a_1 < a_2) ∧ (a_2 < a_3) ∧ (a_3 < a_4) ∧ (a_4 < a_5) ∧
  (1 ≤ a_1 ∧ 1 ≤ a_2 ∧ 1 ≤ a_3 ∧ 1 ≤ a_4 ∧ 1 ≤ a_5) ∧
  (a_1 + a_2 + a_3 + a_4 + a_5 = 55) ∧
  (a_5 = 45) := 
sorry

end largest_in_set_average_11_l2314_231483


namespace intersection_of_A_and_B_l2314_231440

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := { y | ∃ x ∈ A, y = x + 1 }

theorem intersection_of_A_and_B :
  A ∩ B = {2, 3, 4} :=
sorry

end intersection_of_A_and_B_l2314_231440


namespace peanuts_in_box_l2314_231456

variable (original_peanuts : Nat)
variable (additional_peanuts : Nat)

theorem peanuts_in_box (h1 : original_peanuts = 4) (h2 : additional_peanuts = 4) :
  original_peanuts + additional_peanuts = 8 := 
by
  sorry

end peanuts_in_box_l2314_231456


namespace fraction_to_decimal_l2314_231422

theorem fraction_to_decimal : (7 / 12 : ℝ) = 0.5833 := by
  sorry

end fraction_to_decimal_l2314_231422


namespace fewest_number_of_students_l2314_231467

def satisfiesCongruences (n : ℕ) : Prop :=
  n % 6 = 3 ∧
  n % 7 = 4 ∧
  n % 8 = 5 ∧
  n % 9 = 2

theorem fewest_number_of_students : ∃ n : ℕ, satisfiesCongruences n ∧ n = 765 :=
by
  have h_ex : ∃ n : ℕ, satisfiesCongruences n := sorry
  obtain ⟨n, hn⟩ := h_ex
  use 765
  have h_correct : satisfiesCongruences 765 := sorry
  exact ⟨h_correct, rfl⟩

end fewest_number_of_students_l2314_231467


namespace sqrt_450_eq_15_sqrt_2_l2314_231469

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l2314_231469


namespace perpendicular_bisector_eqn_l2314_231430

-- Definitions based on given conditions
def C₁ (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ
def C₂ (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

theorem perpendicular_bisector_eqn {ρ θ : ℝ} :
  (∃ A B : ℝ × ℝ,
    A ∈ {p : ℝ × ℝ | ∃ ρ θ, p = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ C₁ ρ θ} ∧
    B ∈ {p : ℝ × ℝ | ∃ ρ θ, p = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ C₂ ρ θ}) →
  ρ * Real.sin θ + ρ * Real.cos θ = 1 :=
sorry

end perpendicular_bisector_eqn_l2314_231430


namespace molecular_weight_constant_l2314_231451

-- Define the molecular weight of bleach
def molecular_weight_bleach (num_moles : Nat) : Nat := 222

-- Theorem stating the molecular weight of any amount of bleach is 222 g/mol
theorem molecular_weight_constant (n : Nat) : molecular_weight_bleach n = 222 :=
by
  sorry

end molecular_weight_constant_l2314_231451


namespace union_sets_l2314_231463

def S : Set ℕ := {0, 1}
def T : Set ℕ := {0, 3}

theorem union_sets : S ∪ T = {0, 1, 3} :=
by
  sorry

end union_sets_l2314_231463


namespace lowest_score_within_two_std_devs_l2314_231402

variable (mean : ℝ) (std_dev : ℝ) (jack_score : ℝ)

def within_two_std_devs (mean : ℝ) (std_dev : ℝ) (score : ℝ) : Prop :=
  score >= mean - 2 * std_dev

theorem lowest_score_within_two_std_devs :
  mean = 60 → std_dev = 10 → within_two_std_devs mean std_dev jack_score → (40 ≤ jack_score) :=
by
  intros h1 h2 h3
  change mean = 60 at h1
  change std_dev = 10 at h2
  sorry

end lowest_score_within_two_std_devs_l2314_231402


namespace fraction_of_cracked_pots_is_2_over_5_l2314_231425

-- Definitions for the problem conditions
def total_pots : ℕ := 80
def price_per_pot : ℕ := 40
def total_revenue : ℕ := 1920

-- Statement to prove the fraction of cracked pots
theorem fraction_of_cracked_pots_is_2_over_5 
  (C : ℕ) 
  (h1 : (total_pots - C) * price_per_pot = total_revenue) : 
  C / total_pots = 2 / 5 :=
by
  sorry

end fraction_of_cracked_pots_is_2_over_5_l2314_231425


namespace salt_cups_l2314_231443

theorem salt_cups (S : ℕ) (h1 : 8 = S + 1) : S = 7 := by
  -- Problem conditions
  -- 1. The recipe calls for 8 cups of sugar.
  -- 2. Mary needs to add 1 more cup of sugar than cups of salt.
  -- This corresponds to h1.

  -- Prove S = 7
  sorry

end salt_cups_l2314_231443


namespace seating_5_out_of_6_around_circle_l2314_231420

def number_of_ways_to_seat_5_out_of_6_in_circle : Nat :=
  Nat.factorial 4

theorem seating_5_out_of_6_around_circle : number_of_ways_to_seat_5_out_of_6_in_circle = 24 :=
by {
  -- proof would be here
  sorry
}

end seating_5_out_of_6_around_circle_l2314_231420


namespace find_hourly_rate_l2314_231449

-- Definitions of conditions in a)
def hourly_rate : ℝ := sorry  -- This is what we will find.
def hours_worked : ℝ := 3
def tip_percentage : ℝ := 0.2
def total_paid : ℝ := 54

-- Functions based on the conditions
def cost_without_tip (rate : ℝ) : ℝ := hours_worked * rate
def tip_amount (rate : ℝ) : ℝ := tip_percentage * (cost_without_tip rate)
def total_cost (rate : ℝ) : ℝ := (cost_without_tip rate) + (tip_amount rate)

-- The goal is to prove that the rate is 15
theorem find_hourly_rate : total_cost 15 = total_paid :=
by
  sorry

end find_hourly_rate_l2314_231449


namespace smallest_integer_for_perfect_square_l2314_231491

theorem smallest_integer_for_perfect_square :
  let y := 2^5 * 3^5 * 4^5 * 5^5 * 6^4 * 7^3 * 8^3 * 9^2
  ∃ z : ℕ, z = 70 ∧ (∃ k : ℕ, y * z = k^2) :=
by
  sorry

end smallest_integer_for_perfect_square_l2314_231491


namespace exists_horizontal_chord_l2314_231410

theorem exists_horizontal_chord (f : ℝ → ℝ) (h_cont : ContinuousOn f (Set.Icc 0 1))
  (h_eq : f 0 = f 1) : ∃ n : ℕ, n ≥ 1 ∧ ∃ x : ℝ, 0 ≤ x ∧ x + 1/n ≤ 1 ∧ f x = f (x + 1/n) :=
by
  sorry

end exists_horizontal_chord_l2314_231410


namespace magic_square_x_value_l2314_231415

theorem magic_square_x_value 
  (a b c d e f g h : ℤ) 
  (h1 : x + b + c = d + e + c)
  (h2 : x + f + e = a + b + d)
  (h3 : x + e + c = a + g + 19)
  (h4 : b + f + e = a + g + 96) 
  (h5 : 19 = b)
  (h6 : 96 = c)
  (h7 : 1 = f)
  (h8 : a + d + x = b + c + f) : 
    x = 200 :=
by
  sorry

end magic_square_x_value_l2314_231415


namespace nth_monomial_is_correct_l2314_231416

-- conditions
def coefficient (n : ℕ) : ℕ := 2 * n - 1
def exponent (n : ℕ) : ℕ := n
def monomial (n : ℕ) : ℕ × ℕ := (coefficient n, exponent n)

-- theorem to prove the nth monomial
theorem nth_monomial_is_correct (n : ℕ) : monomial n = (2 * n - 1, n) := 
by 
    sorry

end nth_monomial_is_correct_l2314_231416


namespace rupert_jumps_more_l2314_231474

theorem rupert_jumps_more (Ronald_jumps Rupert_jumps total_jumps : ℕ)
  (h1 : Ronald_jumps = 157)
  (h2 : total_jumps = 243)
  (h3 : Rupert_jumps + Ronald_jumps = total_jumps) :
  Rupert_jumps - Ronald_jumps = 86 :=
by
  sorry

end rupert_jumps_more_l2314_231474


namespace smallest_number_of_eggs_proof_l2314_231421

noncomputable def smallest_number_of_eggs (c : ℕ) : ℕ := 15 * c - 3

theorem smallest_number_of_eggs_proof :
  ∃ c : ℕ, c ≥ 11 ∧ smallest_number_of_eggs c = 162 ∧ smallest_number_of_eggs c > 150 :=
by
  sorry

end smallest_number_of_eggs_proof_l2314_231421


namespace trigonometric_identity_l2314_231437

open Real

theorem trigonometric_identity
  (x : ℝ)
  (h1 : sin x * cos x = 1 / 8)
  (h2 : π / 4 < x)
  (h3 : x < π / 2) :
  cos x - sin x = - (sqrt 3 / 2) :=
sorry

end trigonometric_identity_l2314_231437


namespace total_initial_yield_l2314_231404

variable (x y z : ℝ)

theorem total_initial_yield (h1 : 0.4 * x + 0.2 * y = 5) 
                           (h2 : 0.4 * y + 0.2 * z = 10) 
                           (h3 : 0.4 * z + 0.2 * x = 9) 
                           : x + y + z = 40 := 
sorry

end total_initial_yield_l2314_231404


namespace find_k_no_solution_l2314_231417

-- Conditions
def vector1 : ℝ × ℝ := (1, 3)
def direction1 : ℝ × ℝ := (5, -8)
def vector2 : ℝ × ℝ := (0, -1)
def direction2 (k : ℝ) : ℝ × ℝ := (-2, k)

-- Statement
theorem find_k_no_solution (k : ℝ) : 
  (∀ t s : ℝ, vector1 + t • direction1 ≠ vector2 + s • direction2 k) ↔ k = 16 / 5 :=
sorry

end find_k_no_solution_l2314_231417


namespace value_of_a_l2314_231480

theorem value_of_a (a : ℝ) (k : ℝ) (hA : -5 = k * 3) (hB : a = k * (-6)) : a = 10 :=
by
  sorry

end value_of_a_l2314_231480


namespace lucas_fence_painting_l2314_231418

-- Define the conditions
def total_time := 60
def time_painting := 12
def rate_per_minute := 1 / total_time

-- State the theorem
theorem lucas_fence_painting :
  let work_done := rate_per_minute * time_painting
  work_done = 1 / 5 :=
by
  -- Proof omitted
  sorry

end lucas_fence_painting_l2314_231418


namespace find_y_l2314_231460

theorem find_y : ∀ (x y : ℤ), x > 0 ∧ y > 0 ∧ x % y = 9 ∧ (x:ℝ) / (y:ℝ) = 96.15 → y = 60 :=
by
  intros x y h
  sorry

end find_y_l2314_231460


namespace questions_two_and_four_equiv_questions_three_and_seven_equiv_l2314_231476

-- Definitions representing conditions about students in classes A and B:
def ClassA (student : Student) : Prop := sorry
def ClassB (student : Student) : Prop := sorry
def taller (x y : Student) : Prop := sorry
def shorter (x y : Student) : Prop := sorry
def tallest (students : Set Student) : Student := sorry
def shortest (students : Set Student) : Student := sorry
def averageHeight (students : Set Student) : ℝ := sorry
def totalHeight (students : Set Student) : ℝ := sorry
def medianHeight (students : Set Student) : ℝ := sorry

-- Equivalence of question 2 and question 4:
theorem questions_two_and_four_equiv (students_A students_B : Set Student) :
  (∀ a ∈ students_A, ∃ b ∈ students_B, taller a b) ↔ 
  (∀ b ∈ students_B, ∃ a ∈ students_A, taller a b) :=
sorry

-- Equivalence of question 3 and question 7:
theorem questions_three_and_seven_equiv (students_A students_B : Set Student) :
  (∀ a ∈ students_A, ∃ b ∈ students_B, shorter b a) ↔ 
  (shorter (shortest students_B) (shortest students_A)) :=
sorry

end questions_two_and_four_equiv_questions_three_and_seven_equiv_l2314_231476


namespace compare_M_N_l2314_231453

variables (a : ℝ)

-- Definitions based on given conditions
def M : ℝ := 2 * a * (a - 2) + 3
def N : ℝ := (a - 1) * (a - 3)

theorem compare_M_N : M a ≥ N a := 
by {
  sorry
}

end compare_M_N_l2314_231453


namespace sufficient_condition_l2314_231488

-- Definitions:
-- 1. Arithmetic sequence with first term a_1 and common difference d
-- 2. Define the sum of the first n terms of the arithmetic sequence

def arithmetic_sequence (a_1 d : ℤ) (n : ℕ) : ℤ := a_1 + n * d

def sum_first_n_terms (a_1 d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a_1 + (n - 1) * d) / 2

-- Conditions given in the problem:
-- Let a_6 = a_1 + 5d
-- Let a_7 = a_1 + 6d
-- Condition p: a_6 + a_7 > 0

def p (a_1 d : ℤ) : Prop := a_1 + 5 * d + a_1 + 6 * d > 0

-- Sum of first 9 terms S_9 and first 3 terms S_3
-- Condition q: S_9 >= S_3

def q (a_1 d : ℤ) : Prop := sum_first_n_terms a_1 d 9 ≥ sum_first_n_terms a_1 d 3

-- The statement to prove:
theorem sufficient_condition (a_1 d : ℤ) : (p a_1 d) -> (q a_1 d) :=
sorry

end sufficient_condition_l2314_231488


namespace no_integer_right_triangle_side_x_l2314_231486

theorem no_integer_right_triangle_side_x :
  ∀ (x : ℤ), (12 + 30 > x ∧ 12 + x > 30 ∧ 30 + x > 12) →
             (12^2 + 30^2 = x^2 ∨ 12^2 + x^2 = 30^2 ∨ 30^2 + x^2 = 12^2) →
             (¬ (∃ x : ℤ, 18 < x ∧ x < 42)) :=
by
  sorry

end no_integer_right_triangle_side_x_l2314_231486


namespace petrol_price_increase_l2314_231478

variable (P C : ℝ)

/- The original price of petrol is P per unit, and the user consumes C units of petrol.
   The new consumption after a 28.57142857142857% reduction is (5/7) * C units.
   The expenditure remains constant, i.e., P * C = P' * (5/7) * C.
-/
theorem petrol_price_increase (h : P * C = (P * (7/5)) * (5/7) * C) :
  (P * (7/5) - P) / P * 100 = 40 :=
by
  sorry

end petrol_price_increase_l2314_231478
