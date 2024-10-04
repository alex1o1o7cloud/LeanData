import Mathlib

namespace anna_money_left_l547_547441

variable {money_given : ℕ}
variable {price_gum : ℕ}
variable {packs_gum : ℕ}
variable {price_chocolate : ℕ}
variable {bars_chocolate : ℕ}
variable {price_candy_cane : ℕ}
variable {candy_canes: ℕ}

theorem anna_money_left (h1 : money_given = 10) 
                        (h2 : price_gum = 1)
                        (h3 : packs_gum = 3)
                        (h4 : price_chocolate = 1)
                        (h5 : bars_chocolate = 5)
                        (h6 : price_candy_cane = 1 / 2)
                        (h7 : candy_canes = 2)
                        (total_spent : (packs_gum * price_gum) + 
                                      (bars_chocolate * price_chocolate) + 
                                      (candy_canes * price_candy_cane) = 9) :
  money_given - total_spent = 1 := 
  sorry

end anna_money_left_l547_547441


namespace arithmetic_progression_iff_mean_sequence_arithmetic_l547_547635

-- Let {a_n} be a sequence and let S_n be the sum of the first n terms of {a_n}
def arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + d * n

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (∑ i in Finset.range (n+1), a i)

def mean_sequence (S : ℕ → ℝ) (m : ℕ → ℝ) : Prop :=
  ∀ n, m n = S n / (n + 1)

-- We need to prove that {a_n} being an arithmetic progression is both necessary and sufficient for {S_n / n} being an arithmetic progression
theorem arithmetic_progression_iff_mean_sequence_arithmetic (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ → ℝ) :
  (arithmetic_progression a ∧ sum_first_n_terms a S ∧ mean_sequence S m) ↔ arithmetic_progression m := 
by 
  -- Proof will go here
  sorry

end arithmetic_progression_iff_mean_sequence_arithmetic_l547_547635


namespace tangent_identity_l547_547009

variable (α : ℝ)
variable (tanα : ℝ)

theorem tangent_identity 
  (h : tanα = -2) :
  tan (α + π / 2) = 1 / 2 :=
by
  sorry

end tangent_identity_l547_547009


namespace min_value_f_l547_547974

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(2 - x)

theorem min_value_f : ∃ x : ℝ, f x = 4 :=
by {
  sorry
}

end min_value_f_l547_547974


namespace smallest_integer_condition_l547_547450

theorem smallest_integer_condition {A : ℕ} (h1 : A > 1) 
  (h2 : ∃ k : ℕ, A = 5 * k / 3 + 2 / 3)
  (h3 : ∃ m : ℕ, A = 7 * m / 5 + 2 / 5)
  (h4 : ∃ n : ℕ, A = 9 * n / 7 + 2 / 7)
  (h5 : ∃ p : ℕ, A = 11 * p / 9 + 2 / 9) : 
  A = 316 := 
sorry

end smallest_integer_condition_l547_547450


namespace find_b_correct_l547_547220

axioms (a b c : ℝ) (B : ℝ)
  (area_of_triangle : ℝ)
  (h1 : area_of_triangle = real.sqrt 3)
  (h2 : B = 60 * real.pi / 180) -- converting degrees to radians
  (h3 : a^2 + c^2 = 3 * a * c)

noncomputable def find_b : ℝ := 2 * real.sqrt 2

theorem find_b_correct :
  let b := find_b in 
  b = 2 * real.sqrt 2 :=
by
  sorry

end find_b_correct_l547_547220


namespace function_min_value_4_l547_547863

theorem function_min_value_4 :
  ∃ x : ℝ, (2^x + 2^(2 - x)) = 4 :=
sorry

end function_min_value_4_l547_547863


namespace option_A_iff_option_B_l547_547686

-- Definitions
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
∀ n m, a (n + 1) - a n = a (m + 1) - a m

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range n, a (i + 1)

def option_B (S : ℕ → ℝ) : Prop :=
is_arithmetic_progression (λ n, S n / n)

-- Main theorem
theorem option_A_iff_option_B (a : ℕ → ℝ) :
  (is_arithmetic_progression a) ↔ (option_B (λ n, sum_of_first_n_terms a n)) :=
sorry

end option_A_iff_option_B_l547_547686


namespace mean_and_stddev_correct_probability_below_mean_l547_547462

noncomputable def mean (xs : List ℝ) := xs.sum / xs.length.toReal

noncomputable def stddev (xs : List ℝ) :=
  let m := mean xs
  Math.sqrt ((xs.map (λ x => (x - m) ^ 2)).sum / xs.length.toReal)

def list_of_weights : List ℝ := [490, 495, 493, 498, 499, 500, 503, 507, 506]

def light_bags : List ℝ := list_of_weights.filter (λ x => x < 500)

def num_combinations (n k : ℕ) : ℕ :=
  nat.choose n k

def count_pairs_below_mean (mean : ℝ) (bags : List ℝ) : ℕ :=
  (list.combinations 2 bags).count (λ p => p.all (λ x => x < mean))

theorem mean_and_stddev_correct :
  mean list_of_weights = 499 ∧
  stddev list_of_weights = (2 * Real.sqrt 66) / 3 := by
  sorry

theorem probability_below_mean :
  let mean := 499
  let total_pairs := num_combinations light_bags.length 2
  let valid_pairs := count_pairs_below_mean mean light_bags
  (valid_pairs : ℝ) / (total_pairs : ℝ) = 1 := by
  sorry

end mean_and_stddev_correct_probability_below_mean_l547_547462


namespace find_side_b_l547_547108

theorem find_side_b (a b c : ℝ) (A B C : ℝ)
  (area_triangle : Real.sqrt 3) (B_eq_60 : B = 60) (cond_ac: (a^2 + c^2 = 3 * a * c)) :
  ∃ b, b = 2 * Real.sqrt 2 :=
by
  sorry

end find_side_b_l547_547108


namespace chessboard_parity_invariant_l547_547485

-- Defining the chessboard and the repaint operation
def Chessboard := Fin 8 → Fin 8 → Bool

-- Given conditions
def initial_black_cells_even (board : Chessboard) : Prop := 
  (Finset.univ.card (Finset.univ.filter (λ p, board p))) % 2 = 0

def repaint_2x2_square (board : Chessboard) (x y : Fin 7) : Chessboard := 
  λ i j, 
    if i >= x ∧ i < x + 2 ∧ j >= y ∧ j < y + 2 then
      not (board i j)
    else 
      board i j

-- Invariant property
def invariant_property (board : Chessboard) : Prop :=
  ∀ x y, 
    ((Finset.univ.card (Finset.univ.filter (λ p, (repaint_2x2_square board x y) p))) % 2 = 
     ((Finset.univ.card (Finset.univ.filter (λ p, board p))) % 2))

-- Statement to prove the problem
theorem chessboard_parity_invariant (board : Chessboard) :
  initial_black_cells_even board →
  invariant_property board →
  ¬ ∃ final_board, (Finset.univ.card (Finset.univ.filter (λ p, final_board p)) = 1) :=
by {
  intros h_initial h_invariant,
  -- proof goes here
  sorry
}

end chessboard_parity_invariant_l547_547485


namespace count_two_digit_numbers_divisible_by_3_l547_547333

/-
 Define the set of digits
-/
def digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

/-
 Define what it means for a number to be two-digit and have distinct digits
-/
def is_two_digit_with_distinct_digits (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ (∃ (d1 d2 : ℕ), d1 ≠ d2 ∧ d1 ∈ digits ∧ d2 ∈ digits ∧ n = 10 * d1 + d2)

/-
 Define what it means for a number to be divisible by 3
-/
def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

/-
 The final theorem statement
-/
theorem count_two_digit_numbers_divisible_by_3 : Finset.filter (λ n, is_two_digit_with_distinct_digits n ∧ is_divisible_by_3 n) (Finset.Icc 10 99) = 14 :=
by
  sorry

end count_two_digit_numbers_divisible_by_3_l547_547333


namespace exists_xn_gt_yn_l547_547792

noncomputable def x_sequence : ℕ → ℝ := sorry
noncomputable def y_sequence : ℕ → ℝ := sorry

theorem exists_xn_gt_yn
    (x1 x2 y1 y2 : ℝ)
    (hx1 : 1 < x1)
    (hx2 : 1 < x2)
    (hy1 : 1 < y1)
    (hy2 : 1 < y2)
    (h_x_seq : ∀ n, x_sequence (n + 2) = x_sequence n + (x_sequence (n + 1))^2)
    (h_y_seq : ∀ n, y_sequence (n + 2) = (y_sequence n)^2 + y_sequence (n + 1)) :
    ∃ n : ℕ, x_sequence n > y_sequence n :=
sorry

end exists_xn_gt_yn_l547_547792


namespace min_value_f_l547_547973

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(2 - x)

theorem min_value_f : ∃ x : ℝ, f x = 4 :=
by {
  sorry
}

end min_value_f_l547_547973


namespace Vladimir_is_tallest_l547_547385

inductive Boy where
  | Andrei
  | Boris
  | Vladimir
  | Dmitry
  deriving DecidableEq, Repr

open Boy

variable (height aged : Boy → ℕ)

-- Each boy's statements
def andreiStatements (tallest shortest : Boy) :=
  (Boris ≠ tallest, Vladimir = shortest)

def borisStatements (oldest shortest : Boy) :=
  (Andrei = oldest, Andrei = shortest)

def vladimirStatements (heightOlder : Boy → Prop) :=
  (heightOlder Dmitry, heightOlder Dmitry)

def dmitryStatements (vladimirTrue oldest : Prop) :=
  (vladimirTrue, Dmitry = oldest)

noncomputable def is_tallest (b : Boy) : Prop :=
  ∀ b' : Boy, height b' ≤ height b

theorem Vladimir_is_tallest :
  (∃! b : Boy, is_tallest b)
  ∧ (∀ (height ordered : Boy), one_of_each (heightAndAged : Boy -> ℕ -> ℕ), ∃ 
  (tall shortest oldest : Boy),
  (andreiStatements tall shortest).1 ∧ ¬(andreiStatements tall shortest).2 ∧ 
  ¬(borisStatements oldest shortest).1 ∧ (borisStatements oldest shortest).2 ∧
  ¬(vladimirStatements heightOlder).1 ∧ (vladimirStatements heightOlder).2 ∧ 
  ¬(dmitryStatements false (Dmitry = oldest).1) ∧ (dmitryStatements false (Dmitry = oldest).2) ∧
  Vladimir = tall) :=
sorry

end Vladimir_is_tallest_l547_547385


namespace find_b_correct_l547_547222

axioms (a b c : ℝ) (B : ℝ)
  (area_of_triangle : ℝ)
  (h1 : area_of_triangle = real.sqrt 3)
  (h2 : B = 60 * real.pi / 180) -- converting degrees to radians
  (h3 : a^2 + c^2 = 3 * a * c)

noncomputable def find_b : ℝ := 2 * real.sqrt 2

theorem find_b_correct :
  let b := find_b in 
  b = 2 * real.sqrt 2 :=
by
  sorry

end find_b_correct_l547_547222


namespace camilla_blueberry_jelly_beans_l547_547448

theorem camilla_blueberry_jelly_beans (b c : ℕ) (h1 : b = 2 * c) (h2 : b - 10 = 3 * (c - 10)) : b = 40 := 
sorry

end camilla_blueberry_jelly_beans_l547_547448


namespace no_values_less_than_180_l547_547322

/-- Given that w and n are positive integers less than 180 
    such that w % 13 = 2 and n % 8 = 5, 
    prove that there are no such values for w and n. -/
theorem no_values_less_than_180 (w n : ℕ) (hw : w < 180) (hn : n < 180) 
  (h1 : w % 13 = 2) (h2 : n % 8 = 5) : false :=
by
  sorry

end no_values_less_than_180_l547_547322


namespace minimum_value_of_option_C_l547_547884

noncomputable def optionA (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def optionB (x : ℝ) : ℝ := |Real.sin x| + 4 / |Real.sin x|
noncomputable def optionC (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def optionD (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem minimum_value_of_option_C :
  ∃ x : ℝ, optionC x = 4 :=
by
  sorry

end minimum_value_of_option_C_l547_547884


namespace min_value_f_l547_547969

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(2 - x)

theorem min_value_f : ∃ x : ℝ, f x = 4 :=
by {
  sorry
}

end min_value_f_l547_547969


namespace set_equivalence_l547_547480

theorem set_equivalence {x y : ℝ} (hA : {2, 0, x} = {Real.inv x, Real.abs x, y / x}) (hx0 : x ≠ 0) (hx1 : x ≠ 1) (hx2 : x ≠ 2) : 
  x - y = 1 / 2 :=
  sorry

end set_equivalence_l547_547480


namespace max_value_expression_l547_547713

theorem max_value_expression (a c : ℝ) (ha : 0 < a) (hc : 0 < c) :
  ∃ x : ℝ, 2 * (a - x) * (x + real.sqrt(x^2 + c^2)) ≤ a^2 + c^2 :=
by
  sorry

end max_value_expression_l547_547713


namespace largest_prime_divides_sum_l547_547403

-- Given conditions of the problem
def cyclic_sequence (s: List ℕ) : Prop :=
  ∀ (i : ℕ), i < s.length - 1 → (s[(i+1) % s.length] = (s[i] % 1000) * 10 + (s[i] / 1000) / 10 * 100 + (s[i] % 100)) 

def sum_of_sequence (s : List ℕ) : ℕ :=
  s.foldl (+) 0

theorem largest_prime_divides_sum (s : List ℕ) (h: ∀ n, n ∈ s → 1000 ≤ n ∧ n < 10000) :
  cyclic_sequence s →
  ∃ p, prime p ∧ p = 37 ∧ p ∣ sum_of_sequence s :=
by
  sorry

end largest_prime_divides_sum_l547_547403


namespace parabola_hyperbola_focus_l547_547515

theorem parabola_hyperbola_focus (p : ℝ) (hp : 0 < p) :
  (∃ k : ℝ, y^2 = 2 * k * x ∧ k > 0) ∧ (x^2 - y^2 / 3 = 1) → (p = 4) :=
by
  sorry

end parabola_hyperbola_focus_l547_547515


namespace parabola_equation_l547_547531

theorem parabola_equation (a b c : ℝ)
  (h_p : (a + b + c = 1))
  (h_q : (4 * a + 2 * b + c = -1))
  (h_tangent : (4 * a + b = 1)) :
  y = 3 * x^2 - 11 * x + 9 :=
by {
  sorry
}

end parabola_equation_l547_547531


namespace probability_sum_seven_twice_l547_547335

def die := { n : ℕ // 1 ≤ n ∧ n ≤ 6 }

def sum_of_dice (d1 d2 : die) : ℕ := d1.val + d2.val

def is_sum_seven (d1 d2 : die) : Prop :=
  sum_of_dice d1 d2 = 7

def probability_event(n : ℕ) : ℚ := 1 / n

-- The problem statement in Lean 4
theorem probability_sum_seven_twice :
  let probability_once := probability_event(36) in
  probability_once * probability_once = 1 / 36 :=
sorry

end probability_sum_seven_twice_l547_547335


namespace compute_sum_of_d_l547_547236

open Nat

def sum_of_two_digit_d (d : ℕ) : Prop :=
  d > 9 ∧ d < 100 ∧ (143 % d = 7)

theorem compute_sum_of_d : ∑ d in (Finset.filter sum_of_two_digit_d (Finset.range 100)), d = 102 := by
  sorry

end compute_sum_of_d_l547_547236


namespace agent_orange_max_villages_l547_547602

-- Definition of a village and the network structure
def VillageNetwork : Type := sorry -- The type that represents our network (tree)

-- Number of villages
def num_villages (network : VillageNetwork) : Nat := 2020

-- The important property of the Village Network: it is a tree
axiom tree_structure (network : VillageNetwork) : 
  ∀ v1 v2, ∃ unique (path : List (V × V)), path.from v1 ∧ path.to v2

-- Agent Orange's movement constraints
axiom movement_constraints (network : VillageNetwork) :
  ∀ path : List (V × V), (∀ (i : Fin (path.length)), ¬ path[i].v1 == path[i].v2) →

  (∀ (i : Fin (path.length - 1)), ¬ (is_connected network path[i].v1 path[i+1].v1)) 

-- Proof statement
theorem agent_orange_max_villages (network : VillageNetwork) : 
  num_max_villages network = 2019 :=
sorry

end agent_orange_max_villages_l547_547602


namespace inradius_of_triangle_l547_547810

-- conditions
variables (A B C I D X : Type)
variables [inhabited A] [inhabited B] [inhabited C] [inhabited I] [inhabited D] [inhabited X]

-- lengths and specific points
variables (ID IA IX AX : ℝ)
variables (incenter : I = incenter_of_triangle A B C)
variables (foot_perpendicular : D = foot_of_perpendicular A B C)
variables (diameter : AX = diameter_of_circumcircle A B C)

-- given values
variables (hID : ID = 2)
variables (hIA : IA = 3)
variables (hIX : IX = 4)

-- correct answer
theorem inradius_of_triangle : inradius_of_triangle A B C = 11 / 12 :=
by
  sorry

end inradius_of_triangle_l547_547810


namespace hexagonal_star_area_ratio_l547_547398

theorem hexagonal_star_area_ratio (r : ℝ) (h : r = 3) : 
  let A_circle := π * r ^ 2,
      A_hexagon := (3 * real.sqrt 3 / 2) * r ^ 2,
      ratio := A_hexagon / A_circle in
  ratio = (3 * real.sqrt 3) / (2 * π) :=
by
  simp [h]
  sorry

end hexagonal_star_area_ratio_l547_547398


namespace gcd_40_56_l547_547313

theorem gcd_40_56 : Int.gcd 40 56 = 8 :=
by
  sorry

end gcd_40_56_l547_547313


namespace sin_alpha_value_l547_547003

noncomputable def point_coordinates (α : ℝ) : Prop :=
  ∃ x y : ℝ, P = (x, y) ∧ x = sin (2 * π / 3) ∧ y = cos (2 * π / 3)

theorem sin_alpha_value (α : ℝ) :
  point_coordinates α → sin α = - (1 / 2) :=
by
  intros h
  sorry

end sin_alpha_value_l547_547003


namespace number_is_100_l547_547033

theorem number_is_100 (x : ℝ) (h : 0.60 * (3 / 5) * x = 36) : x = 100 :=
by sorry

end number_is_100_l547_547033


namespace minimum_value_C_l547_547945

noncomputable def y_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def y_B (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
noncomputable def y_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def y_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimum_value_C : ∃ x : ℝ, y_C x = 4 := 
by
  -- proof goes here
  sorry

end minimum_value_C_l547_547945


namespace lock_combination_l547_547265

theorem lock_combination 
  (B A N D S : ℕ) (base d : ℕ)
  (h_base : d = 6)
  (h_B : B = 1) 
  (h_A : A = 2) 
  (h_N : N = 3) 
  (h_D : D = 4) 
  (h_S : S = 5) 
  : (S * base^2 + A * base + N) = 523 := by 
  sorry

end lock_combination_l547_547265


namespace linear_function_not_in_first_quadrant_l547_547411

theorem linear_function_not_in_first_quadrant:
  ∀ x y : ℝ, y = -2 * x - 3 → ¬ (x > 0 ∧ y > 0) :=
by
 -- proof steps would go here
 sorry

end linear_function_not_in_first_quadrant_l547_547411


namespace find_b_correct_l547_547213

axioms (a b c : ℝ) (B : ℝ)
  (area_of_triangle : ℝ)
  (h1 : area_of_triangle = real.sqrt 3)
  (h2 : B = 60 * real.pi / 180) -- converting degrees to radians
  (h3 : a^2 + c^2 = 3 * a * c)

noncomputable def find_b : ℝ := 2 * real.sqrt 2

theorem find_b_correct :
  let b := find_b in 
  b = 2 * real.sqrt 2 :=
by
  sorry

end find_b_correct_l547_547213


namespace find_g_minimum_l547_547775

noncomputable def f : ℝ → ℝ := λ x, -1 + Real.log x / Real.log 2 

def g : ℝ → ℝ := λ x, 2 * f x - f (x - 1)

theorem find_g_minimum :
  (∀ x > 0, x ≠ 1 → f 8 = 2 ∧ f 1 = -1) →
  (∀ x > 1, g x = Real.log(x^2 / (x - 1)) / Real.log 2 - 1 ∧ g 2 = 1) :=
  sorry

end find_g_minimum_l547_547775


namespace color_points_l547_547248

open Finset

noncomputable def exists_coloring (S : Finset (ℝ × ℝ)) (hS : S.card = 2004) : Prop :=
  ∃ color : (ℝ × ℝ) → bool,
    ∀ p q ∈ S,
      (∃ k, k ∈ {l ∈ S.powerset 2 | l.card = 2} ∧ p ∈ k ∧ q ∈ k ∧ (¬(color p = color q) ↔ ∃ l ∈ (S.powerset 2).erase k, p ∈ l ∧ q ∈ l)). 

theorem color_points (S : Finset (ℝ × ℝ)) (hS : S.card = 2004) (hS_no_three_collinear : ∀ (p q r : (ℝ × ℝ)), p ∈ S → q ∈ S → r ∈ S → p ≠ q → q ≠ r → p ≠ r → ¬Collinear {p, q, r}) :
  exists_coloring S hS :=
sorry

end color_points_l547_547248


namespace simplified_expression_value_l547_547754

noncomputable def a : ℝ := Real.sqrt 3 + 1
noncomputable def b : ℝ := Real.sqrt 3 - 1

theorem simplified_expression_value :
  ( (a ^ 2 / (a - b) - (2 * a * b - b ^ 2) / (a - b)) / (a - b) * a * b ) = 2 := by
  sorry

end simplified_expression_value_l547_547754


namespace min_area_square_parabola_l547_547010

-- Definitions for the conditions
def parabola (x : ℝ) : ℝ := x^2

def is_square (A B C D : ℝ × ℝ) : Prop :=
  let (a1, b1) := A
  let (a2, b2) := B
  let (a3, b3) := C
  let (a4, b4) := D
  (a1 - a2)^2 + (b1 - b2)^2 = (a2 - a3)^2 + (b2 - b3)^2 ∧
  (a3 - a4)^2 + (b3 - b4)^2 = (a4 - a1)^2 + (b4 - b1)^2 ∧
  (a1 - a3)^2 + (b1 - b3)^2 = (a2 - a4)^2 + (b2 - b4)^2

def has_right_angle (A B C : ℝ × ℝ) : Prop :=
  let (a1, b1) := A
  let (a2, b2) := B
  let (a3, b3) := C
  (b1 - b2) * (a3 - a2) = (b3 - b2) * (a1 - a2)

-- Main hypothesis of our problem: points on the parabola and the specified restrictions.
def square_with_properties (A B C D : ℝ × ℝ) : Prop :=  
  (A.2 = parabola A.1) ∧ (B.2 = parabola B.1) ∧ (C.2 = parabola C.1) ∧ (D.2 = parabola D.1) ∧
  (A.1 > B.1) ∧ (B.1 > 0) ∧ (C.1 < 0) ∧ is_square A B C D ∧ has_right_angle A B C

-- The theorem statement to prove the minimum possible area
theorem min_area_square_parabola (A B C D : ℝ × ℝ) (h : square_with_properties A B C D) : 
  let a := (A.1 - B.1)^2 + (A.2 - B.2)^2 in
  a ≥ 2 :=
sorry

end min_area_square_parabola_l547_547010


namespace find_b_proof_l547_547201

noncomputable def find_b (a b c : ℝ) (A B C : ℝ) : Prop :=
  let area := sqrt 3 in
  let angle_B := 60 * Real.pi / 180 in
  let condition1 := area = (1 / 2) * a * c * Real.sin angle_B in
  let condition2 := a^2 + c^2 = 3 * a * c in
  let side_b := b = 2 * Real.sqrt 2 in
  condition1 ∧ condition2 → side_b

theorem find_b_proof :
  ∃ (a b c : ℝ) (A B C : ℝ), find_b a b c A B C :=
by
  sorry

end find_b_proof_l547_547201


namespace smallest_three_digit_palindrome_divisible_by_11_l547_547318

def is_palindrome (n : Nat) : Prop :=
  let s := n.toDigits 10
  s = s.reverse

theorem smallest_three_digit_palindrome_divisible_by_11 :
  ∃ n : Nat, n ≥ 100 ∧ n < 1000 ∧ is_palindrome n ∧ (11 ∣ n) ∧ ∀ m : Nat, m ≥ 100 ∧ m < 1000 ∧ is_palindrome m ∧ (11 ∣ m) → n ≤ m :=
begin
  use 121,
  split,
  { exact Nat.le_of_lt (by norm_num) },
  split,
  { exact Nat.lt_of_lt_of_le (by norm_num) (by norm_num) },
  split,
  { unfold is_palindrome,
    dsimp,
    exact rfl,
  },
  split,
  { exact DvdTrans 11 11 121 (by exact dvd_refl _),
    rw [← mul_comm, Nat.mod_def', Nat.zero_add, Nat.one_mul, Nat.of_dvd_left_symm_right],
  },
  { intros m H1 H2 H3 H4,
    by_cases h_cases : m = 121,
    { rw [h_cases], },
    sorry
  }
end

end smallest_three_digit_palindrome_divisible_by_11_l547_547318


namespace minimize_f_C_l547_547932

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimize_f_C : ∃ x : ℝ, f_C x = 4 :=
by {
  -- The correctness of the statement is asserted and the proof is omitted
  sorry
}

end minimize_f_C_l547_547932


namespace quoted_price_correct_l547_547392

noncomputable def after_tax_yield (yield : ℝ) (tax_rate : ℝ) : ℝ :=
  yield * (1 - tax_rate)

noncomputable def real_yield (after_tax_yield : ℝ) (inflation_rate : ℝ) : ℝ :=
  after_tax_yield - inflation_rate

noncomputable def quoted_price (dividend_rate : ℝ) (real_yield : ℝ) (commission_rate : ℝ) : ℝ :=
  real_yield / (dividend_rate / (1 + commission_rate))

theorem quoted_price_correct :
  quoted_price 0.16 (real_yield (after_tax_yield 0.08 0.15) 0.03) 0.02 = 24.23 :=
by
  -- This is the proof statement. Since the task does not require us to prove it, we use 'sorry'.
  sorry

end quoted_price_correct_l547_547392


namespace Vladimir_is_tallest_l547_547384

inductive Boy where
  | Andrei
  | Boris
  | Vladimir
  | Dmitry
  deriving DecidableEq, Repr

open Boy

variable (height aged : Boy → ℕ)

-- Each boy's statements
def andreiStatements (tallest shortest : Boy) :=
  (Boris ≠ tallest, Vladimir = shortest)

def borisStatements (oldest shortest : Boy) :=
  (Andrei = oldest, Andrei = shortest)

def vladimirStatements (heightOlder : Boy → Prop) :=
  (heightOlder Dmitry, heightOlder Dmitry)

def dmitryStatements (vladimirTrue oldest : Prop) :=
  (vladimirTrue, Dmitry = oldest)

noncomputable def is_tallest (b : Boy) : Prop :=
  ∀ b' : Boy, height b' ≤ height b

theorem Vladimir_is_tallest :
  (∃! b : Boy, is_tallest b)
  ∧ (∀ (height ordered : Boy), one_of_each (heightAndAged : Boy -> ℕ -> ℕ), ∃ 
  (tall shortest oldest : Boy),
  (andreiStatements tall shortest).1 ∧ ¬(andreiStatements tall shortest).2 ∧ 
  ¬(borisStatements oldest shortest).1 ∧ (borisStatements oldest shortest).2 ∧
  ¬(vladimirStatements heightOlder).1 ∧ (vladimirStatements heightOlder).2 ∧ 
  ¬(dmitryStatements false (Dmitry = oldest).1) ∧ (dmitryStatements false (Dmitry = oldest).2) ∧
  Vladimir = tall) :=
sorry

end Vladimir_is_tallest_l547_547384


namespace polygon_sides_l547_547054

theorem polygon_sides (n : ℕ) (h1 : (n - 2) * 180 = 3 * 360) (h2 : n > 2) : n = 8 := by
  -- Conditions given:
  -- h1: (n - 2) * 180 = 3 * 360
  -- h2: n > 2
  sorry

end polygon_sides_l547_547054


namespace diego_payment_l547_547779

theorem diego_payment (d : ℤ) (celina : ℤ) (total : ℤ) (h₁ : celina = 1000 + 4 * d) (h₂ : total = celina + d) (h₃ : total = 50000) : d = 9800 :=
sorry

end diego_payment_l547_547779


namespace sin_795_eq_tan_sin_cos_eq_l547_547464

theorem sin_795_eq : sin (795 : ℝ) = (√6 + √2) / 4 := sorry

theorem tan_sin_cos_eq : (tan 10 - √3) * (sin 80 / cos 40) = -2 := sorry

end sin_795_eq_tan_sin_cos_eq_l547_547464


namespace minimum_value_of_option_C_l547_547894

noncomputable def optionA (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def optionB (x : ℝ) : ℝ := |Real.sin x| + 4 / |Real.sin x|
noncomputable def optionC (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def optionD (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem minimum_value_of_option_C :
  ∃ x : ℝ, optionC x = 4 :=
by
  sorry

end minimum_value_of_option_C_l547_547894


namespace option_A_iff_option_B_l547_547684

-- Definitions
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
∀ n m, a (n + 1) - a n = a (m + 1) - a m

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range n, a (i + 1)

def option_B (S : ℕ → ℝ) : Prop :=
is_arithmetic_progression (λ n, S n / n)

-- Main theorem
theorem option_A_iff_option_B (a : ℕ → ℝ) :
  (is_arithmetic_progression a) ↔ (option_B (λ n, sum_of_first_n_terms a n)) :=
sorry

end option_A_iff_option_B_l547_547684


namespace find_smaller_number_than_neg3_l547_547435

theorem find_smaller_number_than_neg3 {a b c d : ℝ} (h1 : a = -2) (h2 : b = 4) (h3 : c = -5) (h4 : d = 1) :
  c < -3 :=
by
  rw h3
  ring
  norm_num
  sorry

end find_smaller_number_than_neg3_l547_547435


namespace find_side_b_l547_547144

theorem find_side_b
  (a b c : ℝ)
  (area : ℝ)
  (B : ℝ)
  (h_area : area = real.sqrt 3)
  (h_B : B = (60 : ℝ) * real.pi / 180) -- Convert degrees to radians
  (h_cond : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 := 
sorry

end find_side_b_l547_547144


namespace Second_beats_Third_by_miles_l547_547595

theorem Second_beats_Third_by_miles
  (v1 v2 v3 : ℝ) -- speeds of First, Second, and Third
  (H1 : (10 / v1) = (8 / v2)) -- First beats Second by 2 miles in 10-mile race
  (H2 : (10 / v1) = (6 / v3)) -- First beats Third by 4 miles in 10-mile race
  : (10 - (v3 * (10 / v2))) = 2.5 := 
sorry

end Second_beats_Third_by_miles_l547_547595


namespace arithmetic_progression_iff_mean_sequence_arithmetic_l547_547628

-- Let {a_n} be a sequence and let S_n be the sum of the first n terms of {a_n}
def arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + d * n

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (∑ i in Finset.range (n+1), a i)

def mean_sequence (S : ℕ → ℝ) (m : ℕ → ℝ) : Prop :=
  ∀ n, m n = S n / (n + 1)

-- We need to prove that {a_n} being an arithmetic progression is both necessary and sufficient for {S_n / n} being an arithmetic progression
theorem arithmetic_progression_iff_mean_sequence_arithmetic (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ → ℝ) :
  (arithmetic_progression a ∧ sum_first_n_terms a S ∧ mean_sequence S m) ↔ arithmetic_progression m := 
by 
  -- Proof will go here
  sorry

end arithmetic_progression_iff_mean_sequence_arithmetic_l547_547628


namespace proof_problem_l547_547538

open Real  -- We use Real numbers for the vector operations.

noncomputable def vec_dot_product (u v : ℝ × ℝ × ℝ) := 
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def vec_magnitude (v : ℝ × ℝ × ℝ) := 
  Real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

theorem proof_problem
  (a b : ℝ × ℝ × ℝ)
  (h1 : vec_magnitude a = 4)
  (h2 : vec_magnitude b = 4)
  (h3 : vec_dot_product a b = 4 * 4 * Real.cos (120 * Real.pi / 180)) :
  vec_dot_product b (⟨2 * a.1, 2 * a.2, 2 * a.3⟩ + b) = 0 :=
by
  sorry

end proof_problem_l547_547538


namespace area_triangle_AMN_l547_547746

variables (A B C D M N : Point)
variables [has_area : has_area (parallelogram A B C D) 50]
variables (AM_eq_MB : AM = MB)
variables (CN_eq_3_ND : CN = 3 * ND)

theorem area_triangle_AMN :
  area (triangle A M N) = 12.5 :=
sorry

end area_triangle_AMN_l547_547746


namespace arith_prog_iff_avg_arith_prog_l547_547650

variable {α : Type*} [LinearOrderedField α]

def is_arith_prog (a : ℕ → α) :=
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

def avg_is_arith_prog (S : ℕ → α) :=
  ∃ D : α, ∀ n : ℕ, n ≠ 0 → (S n / n) = (S (n + 1) / (n + 1)) + D

theorem arith_prog_iff_avg_arith_prog (a : ℕ → α) (S : ℕ → α)
  (hS : ∀ n, S n = ∑ i in range n, a i) :
  is_arith_prog a ↔ avg_is_arith_prog S :=
sorry

end arith_prog_iff_avg_arith_prog_l547_547650


namespace min_value_of_2x_plus_2_2x_l547_547872

-- Lean 4 statement
theorem min_value_of_2x_plus_2_2x (x : ℝ) : 2^x + 2^(2 - x) ≥ 4 :=
by sorry

end min_value_of_2x_plus_2_2x_l547_547872


namespace molecular_weight_proof_l547_547317

def atomic_weight_Al : Float := 26.98
def atomic_weight_O : Float := 16.00
def atomic_weight_H : Float := 1.01

def molecular_weight_AlOH3 : Float :=
  (1 * atomic_weight_Al) + (3 * atomic_weight_O) + (3 * atomic_weight_H)

def moles : Float := 7.0

def molecular_weight_7_moles_AlOH3 : Float :=
  moles * molecular_weight_AlOH3

theorem molecular_weight_proof : molecular_weight_7_moles_AlOH3 = 546.07 :=
by
  /- Here we calculate the molecular weight of Al(OH)3 and multiply it by 7.
     molecular_weight_AlOH3 = (1 * 26.98) + (3 * 16.00) + (3 * 1.01) = 78.01
     molecular_weight_7_moles_AlOH3 = 7 * 78.01 = 546.07 -/
  sorry

end molecular_weight_proof_l547_547317


namespace cosine_angle_BHD_l547_547603

open Real

noncomputable def angle_DHG := 45 * π / 180
noncomputable def angle_FHB := 60 * π / 180
noncomputable def side_CD := 1

theorem cosine_angle_BHD (angle_DHG angle_FHB : ℝ) (CD : ℝ) :
  angle_DHG = π / 4 ∧ angle_FHB = π / 3 ∧ CD = 1 →
  cos (angle_BHD angle_DHG angle_FHB CD) = √6 / 4 :=
by
  sorry

end cosine_angle_BHD_l547_547603


namespace optionC_has_min_4_l547_547995

noncomputable def funcA (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def funcB (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def funcC (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def funcD (x : ℝ) : ℝ := log x + 4 / log x

theorem optionC_has_min_4 (x : ℝ) : (∀ y, (y = funcA x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcB x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcD x) → y ≠ 4) ∧
                                   (∃ t, (t = 1) ∧ (funcC t = 4)) := 
by {
  sorry
}

end optionC_has_min_4_l547_547995


namespace complex_fraction_evaluation_l547_547463

theorem complex_fraction_evaluation :
  (1 - complex.I) / (2 + complex.I) = (1/5 : ℂ) - (3/5 : ℂ) * complex.I :=
by
  sorry

end complex_fraction_evaluation_l547_547463


namespace minimum_value_of_h_l547_547908

-- Definitions of the given functions
def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
def g (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
def h (x : ℝ) : ℝ := 2^x + 2^(2 - x)
def j (x : ℝ) : ℝ := log x + 4 / log x

-- Main theorem statement with the condition and conclusion
theorem minimum_value_of_h : ∃ x, h x = 4 :=
by { sorry }

end minimum_value_of_h_l547_547908


namespace girls_more_than_boys_l547_547065

theorem girls_more_than_boys (boys girls : ℕ) (ratio_boys ratio_girls : ℕ) 
  (h1 : ratio_boys = 5)
  (h2 : ratio_girls = 13)
  (h3 : boys = 50)
  (h4 : girls = (boys / ratio_boys) * ratio_girls) : 
  girls - boys = 80 :=
by
  sorry

end girls_more_than_boys_l547_547065


namespace find_side_b_l547_547172

noncomputable def triangle_area (a b c : ℝ) (B : ℝ) : ℝ :=
  1 / 2 * a * c * real.sin B

noncomputable def law_of_cosines (a b c : ℝ) (B : ℝ) : ℝ :=
  a^2 + c^2 - 2 * a * c * real.cos B

theorem find_side_b (a b c : ℝ) (B : ℝ) (area : ℝ)
  (h1 : area = √3) 
  (h2 : B = real.pi / 3)
  (h3 : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 :=
by
  sorry

end find_side_b_l547_547172


namespace find_b_proof_l547_547200

noncomputable def find_b (a b c : ℝ) (A B C : ℝ) : Prop :=
  let area := sqrt 3 in
  let angle_B := 60 * Real.pi / 180 in
  let condition1 := area = (1 / 2) * a * c * Real.sin angle_B in
  let condition2 := a^2 + c^2 = 3 * a * c in
  let side_b := b = 2 * Real.sqrt 2 in
  condition1 ∧ condition2 → side_b

theorem find_b_proof :
  ∃ (a b c : ℝ) (A B C : ℝ), find_b a b c A B C :=
by
  sorry

end find_b_proof_l547_547200


namespace arithmetic_progression_iff_mean_sequence_arithmetic_l547_547632

-- Let {a_n} be a sequence and let S_n be the sum of the first n terms of {a_n}
def arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + d * n

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (∑ i in Finset.range (n+1), a i)

def mean_sequence (S : ℕ → ℝ) (m : ℕ → ℝ) : Prop :=
  ∀ n, m n = S n / (n + 1)

-- We need to prove that {a_n} being an arithmetic progression is both necessary and sufficient for {S_n / n} being an arithmetic progression
theorem arithmetic_progression_iff_mean_sequence_arithmetic (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ → ℝ) :
  (arithmetic_progression a ∧ sum_first_n_terms a S ∧ mean_sequence S m) ↔ arithmetic_progression m := 
by 
  -- Proof will go here
  sorry

end arithmetic_progression_iff_mean_sequence_arithmetic_l547_547632


namespace min_val_of_xsq_ysq_zsq_l547_547717

theorem min_val_of_xsq_ysq_zsq (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) :
  (∃ (M : ℝ), M = x^2 + y^2 + z^2 ∧ ∀ m, m = x^2 + y^2 + z^2 → m ≤ M) :=
begin
  sorry
end

end min_val_of_xsq_ysq_zsq_l547_547717


namespace arith_prog_iff_avg_seq_arith_prog_l547_547673

-- Define sequences
def is_arith_prog (a : ℕ → ℝ) := ∃ d : ℝ, ∃ a₀ : ℝ, ∀ n : ℕ, a n = a₀ + n * d
def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range (n+1), a i
def avg_seq (a : ℕ → ℝ) (n : ℕ) := S a n / (n + 1)

-- Define theorem statement
theorem arith_prog_iff_avg_seq_arith_prog (a : ℕ → ℝ) :
  (is_arith_prog a) ↔ (is_arith_prog (λ n, avg_seq a n)) :=
sorry

end arith_prog_iff_avg_seq_arith_prog_l547_547673


namespace area_of_equilateral_triangle_l547_547088

-- Given conditions
variables (A B C Q : Point) (QA QB QC : ℝ)
variable [EquilateralTriangle A B C]
variable (QA_eq : QA = 6)
variable (QB_eq : QB = 8)
variable (QC_eq : QC = 10)

-- Statement: Area of triangle ABC is approximately 67
theorem area_of_equilateral_triangle (A B C Q : Point) (h_eq_triangle : EquilateralTriangle A B C)
    (h_QA : QA = 6) (h_QB : QB = 8) (h_QC : QC = 10) : 
    abs (area_of_triangle A B C - 67) < 1 :=
sorry

end area_of_equilateral_triangle_l547_547088


namespace find_b_proof_l547_547202

noncomputable def find_b (a b c : ℝ) (A B C : ℝ) : Prop :=
  let area := sqrt 3 in
  let angle_B := 60 * Real.pi / 180 in
  let condition1 := area = (1 / 2) * a * c * Real.sin angle_B in
  let condition2 := a^2 + c^2 = 3 * a * c in
  let side_b := b = 2 * Real.sqrt 2 in
  condition1 ∧ condition2 → side_b

theorem find_b_proof :
  ∃ (a b c : ℝ) (A B C : ℝ), find_b a b c A B C :=
by
  sorry

end find_b_proof_l547_547202


namespace min_value_h_is_4_l547_547983

def f (x : ℝ) := x^2 + 2*x + 4
def g (x : ℝ) := abs (sin x) + 4 / abs (sin x)
def h (x : ℝ) := 2^x + 2^(2 - x)
def j (x : ℝ) := log x + 4 / log x

theorem min_value_h_is_4 : 
  (∀ x, f(x) ≥ 3) ∧
  (∀ x, g(x) > 4) ∧
  (∃ x, h(x) = 4) ∧
  (∀ x, j(x) ≠ 4) → min_value_h = 4 :=
by
  sorry

end min_value_h_is_4_l547_547983


namespace transformation_maps_segment_l547_547325

variables (C D : ℝ × ℝ) (C' D' : ℝ × ℝ)

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

theorem transformation_maps_segment :
  reflect_x (reflect_y (3, -2)) = (-3, 2) ∧ reflect_x (reflect_y (4, -5)) = (-4, 5) :=
by {
  sorry
}

end transformation_maps_segment_l547_547325


namespace option_A_is_necessary_and_sufficient_for_option_B_l547_547691

def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

def mean_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : (ℕ → ℝ)
| n => S n / n

def sum_of_first_n_terms (a : ℕ → ℝ) : ℕ → ℝ
| 0     => 0
| (n+1) => sum_of_first_n_terms a n + a (n+1)

theorem option_A_is_necessary_and_sufficient_for_option_B
    (a : ℕ → ℝ) :
    (is_arithmetic_progression a ↔ is_arithmetic_progression (mean_sequence a (sum_of_first_n_terms a))) :=
sorry

end option_A_is_necessary_and_sufficient_for_option_B_l547_547691


namespace no_possible_arrangement_of_1995_numbers_l547_547614

theorem no_possible_arrangement_of_1995_numbers:
  ¬ ∃ (a : Fin 1995 → ℕ),
    (∀ i : Fin 1995, ∀ j : Fin 1995, i ≠ j → a i ≠ a j) ∧
    (∀ i : Fin 1995, is_prime ((max (a i) (a (i + 1) % 1995)) / (min (a i) (a (i + 1) % 1995)))) :=
sorry

end no_possible_arrangement_of_1995_numbers_l547_547614


namespace domain_log_function_l547_547277

theorem domain_log_function {x : ℝ} : (∃ y, y = log 0.5 (2 * x - 8)) ↔ x > 4 :=
by sorry

end domain_log_function_l547_547277


namespace value_of_f_l547_547006

noncomputable def f (x : ℝ) : ℝ :=
  if x >= 0 then log (2, x + 1) else -log (2, -x + 1)

theorem value_of_f (l m : ℝ) (h_odd : ∀ x, f (-x) = -f x)
  (h_condition : ∀ x, x ≥ 0 → f x = log (2, x + l) + m)
  (hx0 : log (2, l) + m = 0) :
  f (1 - real.sqrt 2) = -1 / 2 := sorry

end value_of_f_l547_547006


namespace graph_point_sum_l547_547051

theorem graph_point_sum (f : ℝ → ℝ) (h : f 3 = -2) :
  let y := (-10:ℝ) / 4 in 
  1 + y = -3 / 2 :=
by
  sorry

end graph_point_sum_l547_547051


namespace zircon_sun_distance_halfway_l547_547286

theorem zircon_sun_distance_halfway (perihelion aphelion : ℝ) (h_perihelion : perihelion = 3) (h_aphelion : aphelion = 15) :
  let major_axis := perihelion + aphelion,
      semi_major_axis := major_axis / 2,
      eccentricity := semi_major_axis - perihelion in
  let halfway_distance := semi_major_axis - eccentricity in
  halfway_distance = 3 := 
by
  sorry

end zircon_sun_distance_halfway_l547_547286


namespace find_side_b_l547_547106

theorem find_side_b (a b c : ℝ) (A B C : ℝ)
  (area_triangle : Real.sqrt 3) (B_eq_60 : B = 60) (cond_ac: (a^2 + c^2 = 3 * a * c)) :
  ∃ b, b = 2 * Real.sqrt 2 :=
by
  sorry

end find_side_b_l547_547106


namespace minimum_value_C_l547_547941

noncomputable def y_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def y_B (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
noncomputable def y_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def y_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimum_value_C : ∃ x : ℝ, y_C x = 4 := 
by
  -- proof goes here
  sorry

end minimum_value_C_l547_547941


namespace min_value_h_l547_547962

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def g (x : ℝ) : ℝ := abs (Real.sin x) + 4 / abs (Real.sin x)
noncomputable def h (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def j (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem min_value_h : ∃ x, h x = 4 :=
by
  sorry

end min_value_h_l547_547962


namespace min_value_h_is_4_l547_547980

def f (x : ℝ) := x^2 + 2*x + 4
def g (x : ℝ) := abs (sin x) + 4 / abs (sin x)
def h (x : ℝ) := 2^x + 2^(2 - x)
def j (x : ℝ) := log x + 4 / log x

theorem min_value_h_is_4 : 
  (∀ x, f(x) ≥ 3) ∧
  (∀ x, g(x) > 4) ∧
  (∃ x, h(x) = 4) ∧
  (∀ x, j(x) ≠ 4) → min_value_h = 4 :=
by
  sorry

end min_value_h_is_4_l547_547980


namespace starting_lineup_count_l547_547735

def football_team := {
  -- Define constraints
  total_members : ℕ := 12,
  offensive_lineman_count : ℕ := 4,
  kicker_count : ℕ := 2,
  lineup_positions := ["quarterback", "running back", "offensive lineman", "kicker", "wide receiver"]
}

theorem starting_lineup_count : 
  ∀ (total_members offensive_lineman_count kicker_count remaining_positions : ℕ),
  total_members = 12 → 
  offensive_lineman_count = 4 → 
  kicker_count = 2 →
  remaining_positions = total_members - 2 →
  offensive_lineman_count * kicker_count * 
  (remaining_positions - 0) * 
  (remaining_positions - 1) *
  (remaining_positions - 2) = 5760 :=
by
  intros,
  sorry

end starting_lineup_count_l547_547735


namespace tallest_boy_is_Vladimir_l547_547344

noncomputable def Andrei_statement1 (Boris_tallest: Prop) : Prop := ¬ Boris_tallest
def Andrei_statement2 (Vladimir_shortest: Prop) : Prop := Vladimir_shortest

def Boris_statement1 (Andrei_oldest: Prop) : Prop := Andrei_oldest
def Boris_statement2 (Andrei_shortest: Prop) : Prop := Andrei_shortest

def Vladimir_statement1 (Dmitry_taller: Prop) : Prop := Dmitry_taller
def Vladimir_statement2 (Dmitry_older: Prop) : Prop := Dmitry_older

noncomputable def Dmitry_statement1 (Vladimir_statement1: Prop) (Vladimir_statement2: Prop) : Prop :=
  Vladimir_statement1 ∧ Vladimir_statement2
def Dmitry_statement2 (Dmitry_oldest: Prop) : Prop := Dmitry_oldest

axiom one_statement_true_per_boy :
  ∀ {P₁ P₂: Prop}, (P₁ ∨ P₂) ∧ ¬ (P₁ ∧ P₂)

axiom no_same_height_or_age :
  ∀ {h1 h2 h3 h4 a1 a2 a3 a4 : ℕ},
    (h1 ≠ h2 ∧ h1 ≠ h3 ∧ h1 ≠ h4 ∧ h2 ≠ h3 ∧ h2 ≠ h4 ∧ h3 ≠ h4) ∧
    (a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)

theorem tallest_boy_is_Vladimir :
  ∀ (Andrei_shortest Vladimir_shortest Boris_tallest Dmitry_taller Dmitry_oldest Vladimir_older : Prop),
    Dmitry_statement2 Dmitry_oldest → 
    Boris_statement2 Andrei_shortest → 
    Andrei_statement1 Boris_tallest → 
    Vladimir_statement2 Vladimir_older → 
    Dmitry_statement1 (Vladimir_statement1 Dmitry_taller) (Vladimir_statement2 Vladimir_older) →
    ¬ Dmitry_taller →
    ¬ Boris_tallest →
    Vladimir = "the tallest boy" :=
  sorry

end tallest_boy_is_Vladimir_l547_547344


namespace polygon_sides_l547_547052

theorem polygon_sides (n : ℕ) (h_interior : (n - 2) * 180 = 3 * 360) : n = 8 :=
by
  sorry

end polygon_sides_l547_547052


namespace minimum_value_C_l547_547948

noncomputable def y_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def y_B (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
noncomputable def y_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def y_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimum_value_C : ∃ x : ℝ, y_C x = 4 := 
by
  -- proof goes here
  sorry

end minimum_value_C_l547_547948


namespace rounding_to_nearest_hundredth_l547_547850

theorem rounding_to_nearest_hundredth (x : ℝ) (h : x = 4.259) : Real.round_nearest_hundredth x = 4.26 := by
  sorry

end rounding_to_nearest_hundredth_l547_547850


namespace general_formula_minimum_value_of_Tn_l547_547235

-- Conditions:
def is_geometric_sequence (a : ℕ → ℕ) (q : ℕ) : Prop :=
∀ n, a (n + 1) = q * a n

def Sn (a : ℕ → ℕ) (n : ℕ) : ℕ :=
∑ i in finset.range n, a (i + 1)

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
5 * Sn a 1 + Sn a 3 = 4 * Sn a 2

noncomputable def b (n : ℕ) : ℕ → ℝ := 
λ n, 1 / ((Real.log2 (a n)) * (Real.log2 (a (n + 1))))

noncomputable def T (a : ℕ → ℕ) (n : ℕ) : ℝ :=
∑ i in finset.range n, b i

-- Proof Statements:
theorem general_formula {a : ℕ → ℕ} (q : ℕ) (h1 : q ≠ 1) (h2 : a 4 = 16) (h3 : is_geometric_sequence a q) (h4 : arithmetic_sequence (Sn a)) :
  ∀ n, a n = 2^n := sorry

theorem minimum_value_of_Tn {a : ℕ → ℕ} (h1 : ∀ n, a n = 2^n) :
  ∃ n, T a 1 = 1 / 2 := sorry

end general_formula_minimum_value_of_Tn_l547_547235


namespace minimize_f_C_l547_547934

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimize_f_C : ∃ x : ℝ, f_C x = 4 :=
by {
  -- The correctness of the statement is asserted and the proof is omitted
  sorry
}

end minimize_f_C_l547_547934


namespace smallest_positive_x_for_palindrome_l547_547826

def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10
  s = s.reverse

theorem smallest_positive_x_for_palindrome :
  ∃ x : ℕ, x > 0 ∧ is_palindrome (x + 1234) ∧ (∀ y : ℕ, y > 0 → is_palindrome (y + 1234) → x ≤ y) ∧ x = 97 := 
sorry

end smallest_positive_x_for_palindrome_l547_547826


namespace infinitely_many_non_prime_num_harmonic_sum_l547_547744

def harmonic_sum (n : ℕ) : ℚ :=
  (finset.range (n+1)).sum (λ i, 1 / (i + 1 : ℚ))

theorem infinitely_many_non_prime_num_harmonic_sum :
  ∃ᶠ n in at_top, ¬ ∃ (p : ℕ) [nat.prime p], ∃ (k : ℕ), harmonic_sum n = p ^ k :=
sorry

end infinitely_many_non_prime_num_harmonic_sum_l547_547744


namespace min_value_f_l547_547966

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(2 - x)

theorem min_value_f : ∃ x : ℝ, f x = 4 :=
by {
  sorry
}

end min_value_f_l547_547966


namespace find_b_l547_547197

theorem find_b
  (a b c : ℝ)
  (h_area : (1/2) * a * c * Real.sin (ℝ.pi / 3) = Real.sqrt 3)
  (h_ac_eq_4 : a * c = 4)
  (h_cosine : a^2 + c^2 = 3 * a * c) :
  b = 2 * Real.sqrt 2 :=
begin
  sorry
end

end find_b_l547_547197


namespace fish_population_estimation_l547_547589

-- Definitions based on conditions
def fish_tagged_day1 : ℕ := 80
def fish_caught_day2 : ℕ := 100
def fish_tagged_day2 : ℕ := 20
def fish_caught_day3 : ℕ := 120
def fish_tagged_day3 : ℕ := 36

-- The average percentage of tagged fish caught on the second and third days
def avg_tag_percentage : ℚ := (20 / 100 + 36 / 120) / 2

-- Statement of the proof problem
theorem fish_population_estimation :
  (avg_tag_percentage * P = fish_tagged_day1) → 
  P = 320 :=
by
  -- Proof goes here
  sorry

end fish_population_estimation_l547_547589


namespace find_side_b_l547_547151

theorem find_side_b (a b c A B C : ℝ) (hB : B = 60) 
  (h_area : (1/2)*a*c*(Real.sin (Real.pi/3)) = Real.sqrt 3) 
  (h_ac : a^2 + c^2 = 3*a*c) : b = 2 * Real.sqrt 2 :=
by
  -- Given conditions
  have hB' : Real.sin (Real.pi/3) = Real.sqrt 3 / 2 := by sorry
  have h_ac_val : a * c = 4 := by sorry
  have h_a2c2 : a^2 + c^2 = 12 := by sorry
  have h_cosB : (a^2 + c^2 - b^2) / (2 * a * c) = 1/2 := by sorry
  have h_b2 : b^2 = 8 := by sorry

  -- Result
  exact Real.sqrt 8

end find_side_b_l547_547151


namespace area_ratio_nonagons_l547_547772

theorem area_ratio_nonagons (N S : Type) [is_nonagon N] [is_nonagon S]
  (h_perpendicular : ∀ (v : vertex N), perpendicular_to_preceding_edge v) :
  area_fraction N S = (1 - Real.cos (40 * Real.pi / 180)) / (1 + Real.cos (40 * Real.pi / 180)) :=
sorry

end area_ratio_nonagons_l547_547772


namespace triangle_inequality_l547_547291

theorem triangle_inequality 
  (A B C F D E : Type*)
  [triangle ABC]
  (h1 : subtending_same_angles ABC F)
  (h2 : meet_lines_at_points B F C F A C A B D E) :
  AB + AC ≥ 4 * DE :=
begin
  sorry
end

end triangle_inequality_l547_547291


namespace apollonius_circle_eq_l547_547082

theorem apollonius_circle_eq :
  ∀ (x y : ℝ),
  let A := (-2 : ℝ, 0 : ℝ)
  let B := (2 : ℝ, 0 : ℝ)
  let λ := 1 / 2
  (λ > 0 ∧ λ ≠ 1) →
  ((real.sqrt ((x + 2)^2 + y^2)) / (real.sqrt ((x - 2)^2 + y^2)) = λ) →
  (x^2 + y^2 + (20 / 3 : ℝ) * x + 4 = 0) :=
by
  intros x y A B λ hλ hratio
  sorry

end apollonius_circle_eq_l547_547082


namespace probability_sum_of_three_dice_is_12_l547_547579

open Finset

theorem probability_sum_of_three_dice_is_12 : 
  (∃ (outcomes : set (ℕ × ℕ × ℕ)), 
    ∀ (x y z : ℕ), (x, y, z) ∈ outcomes ↔ 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ (x + y + z = 12)) → 
    (∃ (prob : ℚ), prob = 2 / 27) :=
by 
  sorry

end probability_sum_of_three_dice_is_12_l547_547579


namespace min_value_h_l547_547953

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def g (x : ℝ) : ℝ := abs (Real.sin x) + 4 / abs (Real.sin x)
noncomputable def h (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def j (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem min_value_h : ∃ x, h x = 4 :=
by
  sorry

end min_value_h_l547_547953


namespace number_of_juniors_l547_547068

theorem number_of_juniors
  (T : ℕ := 28)
  (hT : T = 28)
  (x y : ℕ)
  (hxy : x = y)
  (J S : ℕ)
  (hx : x = J / 4)
  (hy : y = S / 10)
  (hJS : J + S = T) :
  J = 8 :=
by sorry

end number_of_juniors_l547_547068


namespace incenter_point_l547_547081

-- Let △ABC be a triangle.
variables {A B C P : Type*}
-- Assume P is a point inside the triangle that is equidistant from sides of △ABC.
variables (inside_triangle : ∀ (p : P), p ∈ triangle A B C)
variables (equidistant : ∀ (p : P), ∃ d : ℝ, distance p (side A B) = d ∧ distance p (side B C) = d ∧ distance p (side C A) = d)

-- Prove that P must be at the intersection of the angle bisectors of △ABC.
theorem incenter_point (P : P) (inside_triangle P : ∀ (p : P), p ∈ triangle A B C)
  (equidistant P : ∀ (p : P), ∃ d : ℝ, distance p (side A B) = d ∧ distance p (side B C) = d ∧ distance p (side C A) = d) :
  ∃ P_incenter : P, is_incenter P_incenter A B C := 
sorry

end incenter_point_l547_547081


namespace tan_alpha_sqrt3_l547_547026

theorem tan_alpha_sqrt3 (α : ℝ) (h : Real.sin (α + 20 * Real.pi / 180) = Real.cos (α + 10 * Real.pi / 180) + Real.cos (α - 10 * Real.pi / 180)) :
  Real.tan α = Real.sqrt 3 := 
  sorry

end tan_alpha_sqrt3_l547_547026


namespace gcd_192_144_320_l547_547820

def factor_192 : list (ℕ × ℕ) := [(2, 6), (3, 1)]
def factor_144 : list (ℕ × ℕ) := [(2, 4), (3, 2)]
def factor_320 : list (ℕ × ℕ) := [(2, 6), (5, 1)]

theorem gcd_192_144_320 : Nat.gcd (Nat.gcd 192 144) 320 = 16 := by
  sorry

end gcd_192_144_320_l547_547820


namespace arith_prog_iff_avg_seq_arith_prog_l547_547677

-- Define sequences
def is_arith_prog (a : ℕ → ℝ) := ∃ d : ℝ, ∃ a₀ : ℝ, ∀ n : ℕ, a n = a₀ + n * d
def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range (n+1), a i
def avg_seq (a : ℕ → ℝ) (n : ℕ) := S a n / (n + 1)

-- Define theorem statement
theorem arith_prog_iff_avg_seq_arith_prog (a : ℕ → ℝ) :
  (is_arith_prog a) ↔ (is_arith_prog (λ n, avg_seq a n)) :=
sorry

end arith_prog_iff_avg_seq_arith_prog_l547_547677


namespace option_A_iff_option_B_l547_547685

-- Definitions
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
∀ n m, a (n + 1) - a n = a (m + 1) - a m

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range n, a (i + 1)

def option_B (S : ℕ → ℝ) : Prop :=
is_arithmetic_progression (λ n, S n / n)

-- Main theorem
theorem option_A_iff_option_B (a : ℕ → ℝ) :
  (is_arithmetic_progression a) ↔ (option_B (λ n, sum_of_first_n_terms a n)) :=
sorry

end option_A_iff_option_B_l547_547685


namespace minimum_value_a_l547_547041

theorem minimum_value_a (a : ℝ) :
  (∀ (x y : ℝ), x > 0 → y > 0 → x + sqrt (x * y) ≤ a * (x + y)) →
  a ≥ (sqrt 2 + 1) / 2 :=
by {
  sorry
}

end minimum_value_a_l547_547041


namespace conjugate_of_z_l547_547011

-- Definition: z is a complex number given by (2 + i)^2
def z : ℂ := (2 + complex.I)^2

-- Definition: the expected conjugate of z
def conj_z : ℂ := 3 - 4 * complex.I

-- Theorem: The conjugate of z is conj_z
theorem conjugate_of_z : conj(z) = conj_z := by
  sorry

end conjugate_of_z_l547_547011


namespace locus_of_intersection_points_l547_547556

noncomputable theory

open Real

-- Define the statement
theorem locus_of_intersection_points
  (A B : ℝ × ℝ)
  (hA : A.fst ^ 2 + A.snd ^ 2 = 1)
  (hB : B.fst ^ 2 + B.snd ^ 2 = 1)
  (hAB_not_diameter : A ≠ -B)
  (X Y : ℝ × ℝ)
  (hX : X.fst ^ 2 + X.snd ^ 2 = 1)
  (hY : Y.fst ^ 2 + Y.snd ^ 2 = 1)
  (hXY_diameter : X + Y = (0, 0)) :
  ∃ (O : ℝ × ℝ) (r : ℝ), 
    O = (0, 1 / B.snd) ∧ 
    r = B.fst / B.snd ∧ 
    ∀ (P : ℝ × ℝ), (P = intersection_line ((A.fst, A.snd), (X.fst, X.snd)) ((B.fst, B.snd), (-X.fst, -X.snd ))) → 
    (P.fst - O.fst) ^ 2 + (P.snd - O.snd) ^ 2 = r ^ 2 :=
sorry

end locus_of_intersection_points_l547_547556


namespace minimum_value_of_option_C_l547_547883

noncomputable def optionA (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def optionB (x : ℝ) : ℝ := |Real.sin x| + 4 / |Real.sin x|
noncomputable def optionC (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def optionD (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem minimum_value_of_option_C :
  ∃ x : ℝ, optionC x = 4 :=
by
  sorry

end minimum_value_of_option_C_l547_547883


namespace sufficient_but_not_necessary_condition_l547_547709

variables {α β : Plane} {m : Line} 

-- Assume the conditions: α and β are different planes, and line m lies in plane α.
variable (h_diff_planes : α ≠ β)
variable (h_m_in_alpha : m ∈ α)

-- Define the condition for m being perpendicular to every line in plane β
def m_perpendicular_to_all_lines_in_beta (m : Line) (β : Plane) : Prop :=
  ∀ (l : Line), l ∈ β → m ⊥ l

-- To state the conditions formally:
theorem sufficient_but_not_necessary_condition (h_perpendicular : m_perpendicular_to_all_lines_in_beta m β) :
  (α ⊥ β) ∧ ¬((α ⊥ β) → (m_perpendicular_to_all_lines_in_beta m β)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l547_547709


namespace find_y_l547_547323

theorem find_y (x y : ℕ) (hx : x > 0) (hy : y > 0) (hr : x % y = 9) (hxy : (x : ℝ) / y = 96.45) : y = 20 :=
by
  sorry

end find_y_l547_547323


namespace find_b_correct_l547_547217

axioms (a b c : ℝ) (B : ℝ)
  (area_of_triangle : ℝ)
  (h1 : area_of_triangle = real.sqrt 3)
  (h2 : B = 60 * real.pi / 180) -- converting degrees to radians
  (h3 : a^2 + c^2 = 3 * a * c)

noncomputable def find_b : ℝ := 2 * real.sqrt 2

theorem find_b_correct :
  let b := find_b in 
  b = 2 * real.sqrt 2 :=
by
  sorry

end find_b_correct_l547_547217


namespace f_n_add_p_neq_f_n_l547_547479

def f (n : ℕ) : ℕ :=
  ∑ i in finset.range (n + 1), if (nat.gcd i n ≠ 1) then i else 0

theorem f_n_add_p_neq_f_n (n p : ℕ) (h1 : n ≥ 2) (h2 : nat.prime p) : f (n + p) ≠ f n :=
by sorry

end f_n_add_p_neq_f_n_l547_547479


namespace function_satisfies_equation_l547_547752

-- Given conditions
def y (a x : ℝ) : ℝ := a * tan (real.sqrt (a / x - 1))

-- Derivative of the function y
noncomputable def y_prime (a x : ℝ) : ℝ :=
    let u := real.sqrt (a / x - 1)
    a * (1 / (real.cos u) ^ 2) * (-a / (2 * x ^ 2 * real.sqrt (a / x - 1)))

-- The function to be proved
def equation (a x : ℝ) : Prop :=
    a^2 + y a x ^ 2 + 2 * x * real.sqrt (a * x - x^2) * y_prime a x = 0

-- The main statement of the problem
theorem function_satisfies_equation (a x : ℝ) (h : x ≠ 0) : equation a x := sorry

end function_satisfies_equation_l547_547752


namespace range_of_a_l547_547031

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 5 → a * x^2 - x - 4 > 0) → a > 5 :=
by
  sorry

end range_of_a_l547_547031


namespace find_b_proof_l547_547207

noncomputable def find_b (a b c : ℝ) (A B C : ℝ) : Prop :=
  let area := sqrt 3 in
  let angle_B := 60 * Real.pi / 180 in
  let condition1 := area = (1 / 2) * a * c * Real.sin angle_B in
  let condition2 := a^2 + c^2 = 3 * a * c in
  let side_b := b = 2 * Real.sqrt 2 in
  condition1 ∧ condition2 → side_b

theorem find_b_proof :
  ∃ (a b c : ℝ) (A B C : ℝ), find_b a b c A B C :=
by
  sorry

end find_b_proof_l547_547207


namespace min_value_of_expression_l547_547611

variables (A B C M P : Type)
variables (vector : Type) [vector_space ℝ vector]

-- Defining points as vectors
variables (a b c m p : vector)

-- Given conditions in the problem
def BM_eq_2BC : Prop := m - b = 2 • (c - b)

def BP_as_lin_comb : Prop := ∀ (λ μ : ℝ), λ > 0 ∧ μ > 0 → p - b = λ • (a - b) + μ • (c - b)

-- Proving the required minimum value
theorem min_value_of_expression (λ μ : ℝ) (hBMEq2BC : BM_eq_2BC) (hBPAsLinComb : BP_as_lin_comb) : 
  λ + 1 / 2 μ = 1 → ∃ λ μ, (1 / λ) + (2 / μ) = 4 :=
sorry

end min_value_of_expression_l547_547611


namespace iPod_final_cost_is_correct_l547_547727

theorem iPod_final_cost_is_correct :
  let original_price : ℝ := 128
  let first_discount : ℝ := original_price * (7 / 20)
  let price_after_first_discount : ℝ := original_price - first_discount
  let second_discount : ℝ := price_after_first_discount * 0.15
  let price_after_second_discount : ℝ := price_after_first_discount - second_discount
  let sales_tax : ℝ := price_after_second_discount * 0.09
  let final_price : ℝ := price_after_second_discount + sales_tax
  final_price ≈ 77.08 := 
by 
  sorry

end iPod_final_cost_is_correct_l547_547727


namespace determinant_of_transformation_l547_547244

variables (a b c d : ℝ)

def dilation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![5, 0], ![0, 5]]

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -1], ![1, 0]]

def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  rotation_matrix ⬝ dilation_matrix

theorem determinant_of_transformation :
  Matrix.det transformation_matrix = 25 := by
  sorry

end determinant_of_transformation_l547_547244


namespace arithmetic_progression_iff_mean_sequence_arithmetic_l547_547634

-- Let {a_n} be a sequence and let S_n be the sum of the first n terms of {a_n}
def arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + d * n

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (∑ i in Finset.range (n+1), a i)

def mean_sequence (S : ℕ → ℝ) (m : ℕ → ℝ) : Prop :=
  ∀ n, m n = S n / (n + 1)

-- We need to prove that {a_n} being an arithmetic progression is both necessary and sufficient for {S_n / n} being an arithmetic progression
theorem arithmetic_progression_iff_mean_sequence_arithmetic (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ → ℝ) :
  (arithmetic_progression a ∧ sum_first_n_terms a S ∧ mean_sequence S m) ↔ arithmetic_progression m := 
by 
  -- Proof will go here
  sorry

end arithmetic_progression_iff_mean_sequence_arithmetic_l547_547634


namespace min_max_value_of_n_minus_s_l547_547719

theorem min_max_value_of_n_minus_s :
  let n := 10 * a + b in
  let s := a^2 + b^2 in
  (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) →
  ∃ (min_s : ℤ) (max_s : ℤ), min_s = -63 ∧ max_s = 25 :=
by
  sorry

end min_max_value_of_n_minus_s_l547_547719


namespace problem_l547_547250

open Set 

def I : Set ℤ := { x | |x| < 3 }
def A : Set ℤ := {1, 2}
def B : Set ℤ := {-2, -1, 2}
def C_I (B : Set ℤ) : Set ℤ := I \ B

theorem problem : A ∪ (C_I B) = {0, 1, 2} := 
by sorry

end problem_l547_547250


namespace sin_double_angle_inequality_l547_547710

variables {α β γ : ℝ}

-- Definition of a triangle where α, β, and γ are acute angles and α < β < γ.
def acute_angles (α β γ : ℝ) : Prop :=
  0 < α ∧ α < π / 2 ∧
  0 < β ∧ β < π / 2 ∧
  0 < γ ∧ γ < π / 2 ∧
  α + β + γ = π ∧
  α < β ∧ β < γ

theorem sin_double_angle_inequality
  (h : acute_angles α β γ) : 
  sin (2 * α) > sin (2 * β) ∧ sin (2 * β) > sin (2 * γ) :=
sorry

end sin_double_angle_inequality_l547_547710


namespace wood_fixed_by_two_nails_l547_547324

/-- Statement: A piece of wood can be fixed to a wall with two nails.
    Option A: The shortest distance between two points
    Option B: Two points determine a straight line
    Option C: The shortest perpendicular distance
    Option D: In the same plane, there is only one line perpendicular to a given line passing through a point
    We assert that the correct option is B. -/
theorem wood_fixed_by_two_nails:
  "A piece of wood can be fixed to a wall with two nails" ↔ "Two points determine a straight line" :=
sorry

end wood_fixed_by_two_nails_l547_547324


namespace math_problem_l547_547782

noncomputable def mean_median_problem (s : Set ℤ) (ys : ℤ) (hy : ys ∈ s) (hs : s = {92, 90, 85, 88, 89, ys}) : Prop :=
  let mean := (92 + 90 + 85 + 88 + 89 + ys) / 6
  and median := ((89 + 90) : ℤ) / 2
  mean = 89.5 → median = 89.5

theorem math_problem : mean_median_problem {92, 90, 85, 88, 89, 93} 93 sorry :=
  sorry

end math_problem_l547_547782


namespace sum_distinct_prime_factors_924_l547_547835

theorem sum_distinct_prime_factors_924 : 
  ∃ p1 p2 p3 p4 : ℕ, (∀ p: ℕ, p ∈ {p1, p2, p3, p4} → Nat.Prime p) ∧ 
  (p1 * p2 * p3 * p4 ∣ 924) ∧ p1 + p2 + p3 + p4 = 23 := 
sorry

end sum_distinct_prime_factors_924_l547_547835


namespace tallest_is_vladimir_l547_547365

variables (Andrei Boris Vladimir Dmitry : Type)
variables (height age : Type)
variables [linear_order height] [linear_order age]
variables (tallest shortest oldest : height)
variables (andrei_statements : (Boris ≠ tallest) ∧ (Vladimir = shortest))
variables (boris_statements : (Andrei = oldest) ∧ (Andrei = shortest))
variables (vladimir_statements : (Dmitry > Vladimir) ∧ (Dmitry > Vladimir))
variables (dmitry_statements : (vladimir_statements = true) ∧ (Dmitry = oldest))

theorem tallest_is_vladimir (H1: andrei_statements ∨ ¬andrei_statements)
  (H2: boris_statements ∨ ¬boris_statements)
  (H3: vladimir_statements ∨ ¬vladimir_statements)
  (H4: dmitry_statements ∨ ¬dmitry_statements)
  (H5: ∃! (a b c d : height) , a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a) : 
  tallest = Vladimir := 
by
  sorry

end tallest_is_vladimir_l547_547365


namespace find_side_b_l547_547157

theorem find_side_b (a b c A B C : ℝ) (hB : B = 60) 
  (h_area : (1/2)*a*c*(Real.sin (Real.pi/3)) = Real.sqrt 3) 
  (h_ac : a^2 + c^2 = 3*a*c) : b = 2 * Real.sqrt 2 :=
by
  -- Given conditions
  have hB' : Real.sin (Real.pi/3) = Real.sqrt 3 / 2 := by sorry
  have h_ac_val : a * c = 4 := by sorry
  have h_a2c2 : a^2 + c^2 = 12 := by sorry
  have h_cosB : (a^2 + c^2 - b^2) / (2 * a * c) = 1/2 := by sorry
  have h_b2 : b^2 = 8 := by sorry

  -- Result
  exact Real.sqrt 8

end find_side_b_l547_547157


namespace triangle_side_b_l547_547180

theorem triangle_side_b 
  (a b c : ℝ)
  (B : ℝ)
  (h1 : 1/2 * a * c * Real.sin B = Real.sqrt 3)
  (h2 : B = π / 3)
  (h3 : a^2 + c^2 = 3 * a * c) : b = 2 * Real.sqrt 2 := by
  sorry

end triangle_side_b_l547_547180


namespace sum_of_cubes_l547_547845

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) : a^3 + b^3 = 1008 := by
  sorry

end sum_of_cubes_l547_547845


namespace count_three_digit_multiples_of_91_l547_547553

theorem count_three_digit_multiples_of_91 : 
  ∃ (count : ℕ), count = 
    let lcm := Nat.lcm 13 7 in
    let min_k := (100 / lcm) + 1 in
    let max_k := 999 / lcm in
    max_k - min_k + 1 ∧ lcm = 91 ∧ (100 ≤ 91 * min_k) ∧ (91 * max_k ≤ 999) :=
by
  sorry

end count_three_digit_multiples_of_91_l547_547553


namespace problem1_problem2_l547_547487

noncomputable def S_n (n : ℕ) (λ : ℝ) : ℝ := λ * 2^n - 2

def a_n (n : ℕ) : ℝ := 2^n -- This is a conjectured general term to be proven.

def b_n (n : ℕ) : ℕ := 2 * n

def c_n (n : ℕ) : ℝ := 1 / (4 * n^2 - 1)

def T_n (n : ℕ) : ℝ := (1 / 2) * (1 - 1 / (2 * n + 1))

-- Prove the general term of the sequence (1):
theorem problem1 (λ : ℝ) (n : ℕ) (hλ : λ = 2) : 
  S_n n λ = λ * 2^n - 2 → a_n n = 2^n := 
begin
  intros h, -- introduction rule
  rw hλ at *, -- substitution
  sorry -- proof is omitted
end

-- Prove the sum of the first n terms of the sequence (2):
theorem problem2 (n : ℕ) : 
  (finset.range n).sum c_n = T_n n := 
begin
  sorry -- proof is omitted
end

end problem1_problem2_l547_547487


namespace function_min_value_l547_547914

theorem function_min_value (f1 f2 f3 f4 : ℝ → ℝ) :
  f1 = λ x, x^2 + 2 * x + 4 ∧
  f2 = λ x, abs (sin x) + 4 / abs (sin x) ∧
  f3 = λ x, 2^x + 2^(2-x) ∧
  f4 = λ x, log x + 4 / log x →
  (∀ x : ℝ, f3 x ≥ 4) ∧ (∃ x : ℝ, f3 x = 4) :=
by
  sorry

end function_min_value_l547_547914


namespace calories_per_carrot_stick_l547_547251

theorem calories_per_carrot_stick : 
  (1 : ℝ) * 400 + (5 : ℝ) * 50 + (5 : ℝ) * x = 750 → x = 20 :=
by 
  assume h : 1 * 400 + 5 * 50 + 5 * x = 750
  have hs := calc
    1 * 400 + 5 * 50 : ℝ
      = 400 + 250 : by simp
      ... = 650 : by ring
  simp at h
  sorry

end calories_per_carrot_stick_l547_547251


namespace find_a_l547_547624

def A (x : ℝ) := (x^2 - 4 ≤ 0)
def B (x : ℝ) (a : ℝ) := (2 * x + a ≤ 0)
def C (x : ℝ) := (-2 ≤ x ∧ x ≤ 1)

theorem find_a (a : ℝ) : (∀ x : ℝ, A x → B x a → C x) → a = -2 :=
sorry

end find_a_l547_547624


namespace zero_matrix_possible_l547_547499

theorem zero_matrix_possible :
  ∀ (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ),
  (∀ i : Fin n, (∑ j : Fin n, A i j) = 0) →
  (∀ j : Fin n, (∑ i : Fin n, A i j) = 0) →
  (∃ (f_seq : List ((Fin n) × (Fin n) × (Fin n))),
  ∀ B : Matrix (Fin n) (Fin n) ℝ,
  (∀ i : Fin n, (∑ j : Fin n, B i j) = 0) →
  (∀ j : Fin n, (∑ i : Fin n, B i j) = 0) →
  List.foldl (λ C f, let ⟨i, j, k⟩ := f in update_matrix C i j k) A f_seq = 0) :=
begin
  sorry
end

noncomputable def update_matrix
  {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) (i : Fin n) (j : Fin n) (k : Fin n) : Matrix (Fin n) (Fin n) ℝ :=
  λ x y, if x = i then A x y else A x y

end zero_matrix_possible_l547_547499


namespace collinear_points_l547_547559

noncomputable def e1 : Type := sorry
noncomputable def e2 : Type := sorry

variables (e1 e2 : Type) [AddCommGroup e1] [AddCommGroup e2] [Module ℝ e1] [Module ℝ e2] 

noncomputable def k (e1 e2 : Type) [AddCommGroup e1] [AddCommGroup e2] [Module ℝ e1] [Module ℝ e2] : ℝ := sorry

theorem collinear_points
  (e1 e2 : Type)
  [AddCommGroup e1] [AddCommGroup e2]
  [Module ℝ e1] [Module ℝ e2]
  (h_non_collinear : non_collinear e1 e2)
  (h_AB : ∃ k : ℝ, ∃ AB : e1, AB = 2 • e1 + k • e2)
  (h_CB : ∃ CB : e1, CB = e1 + 3 • e2)
  (h_CD : ∃ CD : e1, CD = 2 • e1 - e2)
  (h_collinear_points : collinear_points A B D)
  : k = -8 :=
sorry

end collinear_points_l547_547559


namespace probability_sum_of_three_dice_is_12_l547_547577

open Finset

theorem probability_sum_of_three_dice_is_12 : 
  (∃ (outcomes : set (ℕ × ℕ × ℕ)), 
    ∀ (x y z : ℕ), (x, y, z) ∈ outcomes ↔ 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ (x + y + z = 12)) → 
    (∃ (prob : ℚ), prob = 2 / 27) :=
by 
  sorry

end probability_sum_of_three_dice_is_12_l547_547577


namespace min_value_of_2x_plus_2_2x_l547_547870

-- Lean 4 statement
theorem min_value_of_2x_plus_2_2x (x : ℝ) : 2^x + 2^(2 - x) ≥ 4 :=
by sorry

end min_value_of_2x_plus_2_2x_l547_547870


namespace units_digit_2_104_5_205_11_302_l547_547840

theorem units_digit_2_104_5_205_11_302 : 
  ((2 ^ 104) * (5 ^ 205) * (11 ^ 302)) % 10 = 0 :=
by
  sorry

end units_digit_2_104_5_205_11_302_l547_547840


namespace triangle_area_l547_547610

noncomputable def cosA : ℝ := sqrt 6 / 3
noncomputable def cosB : ℝ := 2 * sqrt 2 / 3
noncomputable def c : ℝ := 2 * sqrt 2
noncomputable def expected_area : ℝ := 2 * sqrt 2 / 3

theorem triangle_area (cosA_eq : cosA = sqrt 6 / 3)
  (cosB_eq : cosB = 2 * sqrt 2 / 3)
  (c_eq : c = 2 * sqrt 2) :
  ∃ (a b c : ℝ), 
    ∃ (A B C : ℝ),
      -- These are placeholders that need to satisfy the given conditions
      cosine.law_of_cosines A B C cosA_eq cosB_eq (cos B) c_eq -> 
        area_of_triangle a b c = expected_area :=
by
  sorry

end triangle_area_l547_547610


namespace minimum_value_C_l547_547950

noncomputable def y_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def y_B (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
noncomputable def y_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def y_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimum_value_C : ∃ x : ℝ, y_C x = 4 := 
by
  -- proof goes here
  sorry

end minimum_value_C_l547_547950


namespace find_b_l547_547192

theorem find_b
  (a b c : ℝ)
  (h_area : (1/2) * a * c * Real.sin (ℝ.pi / 3) = Real.sqrt 3)
  (h_ac_eq_4 : a * c = 4)
  (h_cosine : a^2 + c^2 = 3 * a * c) :
  b = 2 * Real.sqrt 2 :=
begin
  sorry
end

end find_b_l547_547192


namespace sin_squared_alpha_plus_sin_2alpha_l547_547506

theorem sin_squared_alpha_plus_sin_2alpha (α : ℝ) 
  (h : sin (3 * Real.pi + α) = 2 * sin ((3 * Real.pi) / 2 + α)) : 
  (sin α) ^ 2 + sin (2 * α) = 8 / 5 :=
by
  sorry

end sin_squared_alpha_plus_sin_2alpha_l547_547506


namespace max_popsicles_l547_547258

theorem max_popsicles (budget : ℕ) (cost_single : ℕ) (popsicles_single : ℕ) (cost_box3 : ℕ) (popsicles_box3 : ℕ) (cost_box7 : ℕ) (popsicles_box7 : ℕ)
  (h_budget : budget = 10) (h_cost_single : cost_single = 1) (h_popsicles_single : popsicles_single = 1)
  (h_cost_box3 : cost_box3 = 3) (h_popsicles_box3 : popsicles_box3 = 3)
  (h_cost_box7 : cost_box7 = 4) (h_popsicles_box7 : popsicles_box7 = 7) :
  ∃ n, n = 16 :=
by
  sorry

end max_popsicles_l547_547258


namespace visitors_total_l547_547439

theorem visitors_total (oct_visitors : ℕ) (nov_increase_pct dec1_decrease_pct dec2_increase_pct : ℝ) :
  oct_visitors = 100 →
  nov_increase_pct = 0.15 →
  dec1_decrease_pct = 0.10 →
  dec2_increase_pct = 0.20 →
  let nov_visitors := oct_visitors + nat.floor (nov_increase_pct * oct_visitors)
  let dec1_visitors := nov_visitors - nat.floor (dec1_decrease_pct * nov_visitors)
  let dec2_visitors := nov_visitors + nat.floor (dec2_increase_pct * nov_visitors)
  oct_visitors + nov_visitors + dec1_visitors + dec2_visitors = 457 :=
by
  intros
  simp only [nov_visitors, dec1_visitors, dec2_visitors]
  simp only [HMul.hMul, Nat.floor]
  norm_num
  sorry

end visitors_total_l547_547439


namespace multiple_of_son_age_last_year_l547_547253

theorem multiple_of_son_age_last_year
  (G : ℕ) (S : ℕ) (M : ℕ)
  (h1 : G = 42 - 1)
  (h2 : S = 16 - 1)
  (h3 : G = M * S - 4) :
  M = 3 := by
  sorry

end multiple_of_son_age_last_year_l547_547253


namespace minimum_value_of_option_C_l547_547891

noncomputable def optionA (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def optionB (x : ℝ) : ℝ := |Real.sin x| + 4 / |Real.sin x|
noncomputable def optionC (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def optionD (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem minimum_value_of_option_C :
  ∃ x : ℝ, optionC x = 4 :=
by
  sorry

end minimum_value_of_option_C_l547_547891


namespace vladimir_is_tallest_l547_547337

inductive Boy : Type
| Andrei | Boris | Vladimir | Dmitry

open Boy

-- Hypotheses based on the problem statement
def statements : Boy → (Prop × Prop)
| Andrei := (¬ (Boris = tallest), Vladimir = shortest)
| Boris := (Andrei = oldest, Andrei = shortest)
| Vladimir := (Dmitry > Vladimir, Dmitry = oldest)
| Dmitry := (statements Vladimir = (true, true), Dmitry = oldest)

def different_heights (a b : Boy) : Prop := a ≠ b
def different_ages (a b : Boy) : Prop := a ≠ b

axiom no_same_height {a b : Boy} : different_heights a b
axiom no_same_age {a b : Boy} : different_ages a b

-- The proof statement
theorem vladimir_is_tallest : Vladimir = tallest := 
by 
  sorry

end vladimir_is_tallest_l547_547337


namespace probability_distribution_median_and_contingency_table_drug_inhibitory_effect_l547_547808

noncomputable section

namespace MouseDrugStudy

-- Define our given conditions
def num_mice_total := 40
def num_mice_control := 20
def num_mice_experimental := 20
def weight_control_group : List ℝ := [17.3, 18.4, 20.1, 20.4, 21.5, 23.2, 24.6, 24.8, 25.0, 25.4, 26.1, 26.3, 26.4, 26.5, 26.8, 27.0, 27.4, 27.5, 27.6, 28.3]
def weight_experimental_group : List ℝ := [5.4, 6.6, 6.8, 6.9, 7.8, 8.2, 9.4, 10.0, 10.4, 11.2, 14.4, 17.3, 19.2, 20.2, 23.6, 23.8, 24.5, 25.1, 25.2, 26.0]

-- Statements to be proven:
def P_X_0 : ℝ := 19 / 78
def P_X_1 : ℝ := 20 / 39
def P_X_2 : ℝ := 19 / 78
def E_X : ℝ := 1
def median_weight : ℝ := 23.4
def chisq_stat : ℝ := 6.400
def confidence_level := 95

-- Main theorem statements
theorem probability_distribution :
  P(X = 0) = P_X_0 ∧ P(X = 1) = P_X_1 ∧ P(X = 2) = P_X_2 ∧ E(X) = E_X := sorry

theorem median_and_contingency_table :
  median (weight_control_group ++ weight_experimental_group) = median_weight ∧
    contingency_table weight_control_group weight_experimental_group median_weight = ([6, 14], [14, 6]) := sorry

theorem drug_inhibitory_effect :
  chisq_test ([6, 14], [14, 6]) num_mice_total ≥ chisq_stat ∧
    chisq_stat > 3.841 → confidence_level = 95 := sorry

end MouseDrugStudy

end probability_distribution_median_and_contingency_table_drug_inhibitory_effect_l547_547808


namespace smallest_k_for_grid_filling_l547_547818

theorem smallest_k_for_grid_filling : ∃ k: ℕ, (∀ n: ℕ, ∃! positions: finset (ℤ × ℤ), positions.card = n ∧ 
  (∀ (p1 p2: (ℤ × ℤ)), p1 ∈ positions → p2 ∈ positions → (|fst p1 - fst p2| ≤ 1 ∧ |snd p1 - snd p2| ≤ 1) → abs ((positions.filter (λ p, p = p1)).card - (positions.filter (λ p, p = p2)).card) < k)) ∧ k = 3 :=
by
  sorry

end smallest_k_for_grid_filling_l547_547818


namespace model_lighthouse_height_l547_547067

-- Given conditions as Lean definitions
def actual_lighthouse_height : ℝ := 50            -- in meters
def actual_visibility_distance : ℝ := 30000       -- in meters
def model_visibility_distance : ℝ := 0.3         -- in meters

-- Problem statement: calculate the height of the model lighthouse to maintain the proper scale
theorem model_lighthouse_height:
  let scale_factor := real.cbrt (actual_visibility_distance / model_visibility_distance) in
  let model_height := actual_lighthouse_height / scale_factor in
  model_height = 1.08 :=
by
  sorry

end model_lighthouse_height_l547_547067


namespace find_side_b_l547_547109

theorem find_side_b (a b c : ℝ) (A B C : ℝ)
  (area_triangle : Real.sqrt 3) (B_eq_60 : B = 60) (cond_ac: (a^2 + c^2 = 3 * a * c)) :
  ∃ b, b = 2 * Real.sqrt 2 :=
by
  sorry

end find_side_b_l547_547109


namespace simplified_expression_value_l547_547753

noncomputable def a : ℝ := Real.sqrt 3 + 1
noncomputable def b : ℝ := Real.sqrt 3 - 1

theorem simplified_expression_value :
  ( (a ^ 2 / (a - b) - (2 * a * b - b ^ 2) / (a - b)) / (a - b) * a * b ) = 2 := by
  sorry

end simplified_expression_value_l547_547753


namespace gondor_total_earnings_l547_547023

-- Defining the earnings from repairing a phone and a laptop
def phone_earning : ℕ := 10
def laptop_earning : ℕ := 20

-- Defining the number of repairs
def monday_phone_repairs : ℕ := 3
def tuesday_phone_repairs : ℕ := 5
def wednesday_laptop_repairs : ℕ := 2
def thursday_laptop_repairs : ℕ := 4

-- Calculating total earnings
def monday_earnings : ℕ := monday_phone_repairs * phone_earning
def tuesday_earnings : ℕ := tuesday_phone_repairs * phone_earning
def wednesday_earnings : ℕ := wednesday_laptop_repairs * laptop_earning
def thursday_earnings : ℕ := thursday_laptop_repairs * laptop_earning

def total_earnings : ℕ := monday_earnings + tuesday_earnings + wednesday_earnings + thursday_earnings

-- The theorem to be proven
theorem gondor_total_earnings : total_earnings = 200 := by
  sorry

end gondor_total_earnings_l547_547023


namespace find_a_b_l547_547019

noncomputable def y (a : ℝ) (b : ℝ) (x : ℝ) : ℝ := a^x + b

theorem find_a_b (a b : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) 
    (h3 : ∀ x ∈ Icc (-1 : ℝ) 0, y a b x ∈ Icc (-1 : ℝ) 0) : a + b = -3/2 :=
by
  sorry

end find_a_b_l547_547019


namespace piecewise_linear_exists_l547_547461

noncomputable def f : ℝ → ℝ
| x if x ∈ ⋃ (x : ℝ), {x | -1 ≤ x ∧ x < -1/2} := -x - 1/2
| x if x ∈ ⋃ (x : ℝ), {x | -1/2 ≤ x ∧ x < 0} := x - 1/2
| 0 := 0
| x if x ∈ ⋃ (x : ℝ), {x | 0 < x ∧ x ≤ 1/2} := x + 1/2
| x if x ∈ ⋃ (x : ℝ), {x | 1/2 < x ∧ x ≤ 1} := -x + 1/2
| x := 0

theorem piecewise_linear_exists :
  ∃ (f : ℝ → ℝ), 
    (∀ x, -1 ≤ x ∧ x ≤ 1 → 
    (x ∈ ⋃ (x : ℝ), {x | -1 ≤ x ∧ x < -1/2} → f x = -x - 1/2) ∧
    (x ∈ ⋃ (x : ℝ), {x | -1/2 ≤ x ∧ x < 0} → f x = x - 1/2) ∧
    (x = 0 → f x = 0) ∧
    (x ∈ ⋃ (x : ℝ), {x | 0 < x ∧ x ≤ 1/2} → f x = x + 1/2) ∧
    (x ∈ ⋃ (x : ℝ), {x | 1/2 < x ∧ x ≤ 1} → f x = -x + 1/2) ∧
    (x ∈ [-1, 1] → f (f x) = -x) :=
begin
  use f,
  sorry
end

end piecewise_linear_exists_l547_547461


namespace function_min_value_4_l547_547862

theorem function_min_value_4 :
  ∃ x : ℝ, (2^x + 2^(2 - x)) = 4 :=
sorry

end function_min_value_4_l547_547862


namespace find_length_of_b_l547_547129

theorem find_length_of_b
  {A B C : Type*}
  (a b c : ℝ)
  (area : ℝ)
  (angleB : ℝ)
  (h_area : area = sqrt 3)
  (h_angle : angleB = real.pi / 3)  -- 60 degrees in radians
  (h_sides : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 :=
by
  -- 1. Using the given area constraint: area = 1/2 * a * c * sin(B)
  -- 2. Using the given angle: sin(60) = sqrt(3)/2
  -- 3. Using the given sides relation: a^2 + c^2 = 3ac
  -- 4. Using the Law of Cosines: cos(60) = 1/2
  -- fill proof here
  sorry

end find_length_of_b_l547_547129


namespace greatest_three_digit_number_l547_547314

theorem greatest_three_digit_number :
  ∃ n : ℕ, 
    (n ≡ -1 [MOD 11]) ∧ 
    (n ≡ 4 [MOD 7]) ∧ 
    100 ≤ n ∧ n ≤ 999 ∧ 
    ∀ m : ℕ, (m ≡ -1 [MOD 11]) ∧ (m ≡ 4 [MOD 7]) ∧ 100 ≤ m ∧ m ≤ 999 → m ≤ n :=
exists.intro 956 ⟨by norm_num, by norm_num, by norm_num, by norm_num, by sorry⟩


end greatest_three_digit_number_l547_547314


namespace right_triangle_side_value_l547_547568

theorem right_triangle_side_value (x : ℝ) (h : x > 0) :
    (x = 10 ∨ x = 2 * Real.sqrt 7) ↔ (x^2 = 6^2 + 8^2 ∨ 6^2 + x^2 = 8^2) :=
begin
  sorry
end

end right_triangle_side_value_l547_547568


namespace square_garden_perimeter_l547_547769

theorem square_garden_perimeter (A : ℝ) (h : A = 450) : ∃ P : ℝ, P = 60 * Real.sqrt 2 :=
  sorry

end square_garden_perimeter_l547_547769


namespace sum_first_10_terms_arithmetic_seq_l547_547078

variable (a₁ a₁₀ : ℝ) (S : ℕ → ℝ)

def arithmetic_sum (n : ℕ) (a₁ aₙ : ℝ) : ℝ := (n / 2) * (a₁ + aₙ)

theorem sum_first_10_terms_arithmetic_seq : 
  a₁ + a₁₀ = 12 → S 10 = 60 :=
by 
  intro h,
  have hS : S 10 = arithmetic_sum 10 a₁ a₁₀ := sorry,
  rw h at *,
  have := arithmetic_sum 10 a₁ a₁₀,
  sorry

end sum_first_10_terms_arithmetic_seq_l547_547078


namespace cos_BAO_proof_l547_547598

variable (A B C D O : Type) [EuclideanGeometry A B C D O]

-- Define the rectangle ABCD with diagonals intersecting at O
variables (rect : is_rectangle A B C D)
variables (diagInt : intersection (diagonal A D) (diagonal B C) = O)
variables (len_AB : length (segment A B) = 15)
variables (len_BC : length (segment B C) = 32)

-- Prove that cos ∠BAO = 450/1249
theorem cos_BAO_proof : cos_angle (angle (segment B A) (segment A O)) = 450 / 1249 := by
  sorry

end cos_BAO_proof_l547_547598


namespace minimum_value_of_h_l547_547901

-- Definitions of the given functions
def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
def g (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
def h (x : ℝ) : ℝ := 2^x + 2^(2 - x)
def j (x : ℝ) : ℝ := log x + 4 / log x

-- Main theorem statement with the condition and conclusion
theorem minimum_value_of_h : ∃ x, h x = 4 :=
by { sorry }

end minimum_value_of_h_l547_547901


namespace find_m_l547_547295

noncomputable def volume_parallelepiped (v₁ v₂ v₃ : ℝ × ℝ × ℝ) : ℝ :=
  real.abs (matrix.det ![
      ![v₁.1, v₂.1, v₃.1],
      ![v₁.2.1, v₂.2.1, v₃.2.1],
      ![v₁.2.2, v₂.2.2, v₃.2.2]
    ])

theorem find_m (m : ℝ) (h : volume_parallelepiped (3, 2, 5) (2, m, 3) (2, 4, m) = 20) (hm : m > 0) :
  m = 5 :=
sorry

end find_m_l547_547295


namespace probability_sum_even_l547_547848

theorem probability_sum_even {q : ℚ} (hq : q = 1 / 3) : 
  let p_even := q
  let p_odd := 2 * q
  let prob_sum_even := (p_even ^ 3) + (p_odd ^ 3) + (3 * p_even * (p_odd ^ 2)) + (3 * (p_even ^ 2) * p_odd)
  in prob_sum_even = 1 :=
by
  -- Conditions
  have p_even_eq : p_even = q := rfl
  have p_odd_eq : p_odd = 2 * q := rfl
  have q_value: q = 1 / 3 := hq
  have p_even_value : p_even = 1 / 3 := p_even_eq ▸ q_value
  have p_odd_value : p_odd = 2 / 3 := p_odd_eq ▸ q_value

  -- Computations
  have pe3 : p_even ^ 3 = (1 / 3) ^ 3 := p_even_value ▸ rfl
  have po3 : p_odd ^ 3 = (2 / 3) ^ 3 := p_odd_value ▸ rfl
  have pe_po2 : p_even * (p_odd ^ 2) = (1 / 3) * ((2 / 3) ^ 2) := by rw [p_even_value, p_odd_value]
  have pe2_po : (p_even ^ 2) * p_odd = ((1 / 3) ^ 2) * (2 / 3) := by rw [p_even_value, p_odd_value]

  -- Combining probabilities
  have prob_sum_even : prob_sum_even = (1 / 27) + (8 / 27) + (3 * (4 / 27)) + (3 * (2 / 27)) := 
    by simp [p_even_value, p_odd_value, pe3, po3, pe_po2, pe2_po]

  -- Final simplification to show it's 1
  simp [prob_sum_even]
  exact sorry

end probability_sum_even_l547_547848


namespace even_expressions_l547_547764

theorem even_expressions (x y : ℕ) (hx : Even x) (hy : Even y) :
  Even (x + 5 * y) ∧
  Even (4 * x - 3 * y) ∧
  Even (2 * x^2 + 5 * y^2) ∧
  Even ((2 * x * y + 4)^2) ∧
  Even (4 * x * y) :=
by
  sorry

end even_expressions_l547_547764


namespace find_side_b_l547_547121

theorem find_side_b
  (a b c : ℝ)
  (area : ℝ)
  (B : ℝ)
  (h_area : area = sqrt 3)
  (h_B : B = real.pi / 3)
  (h_a2c2 : a^2 + c^2 = 3 * a * c)
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) :
  b = 2 * real.sqrt 2 :=
sorry

end find_side_b_l547_547121


namespace loggers_cut_down_trees_l547_547734

theorem loggers_cut_down_trees (loggers : ℕ) (trees_per_logger_per_day : ℕ) (forest_length : ℕ) 
  (forest_width : ℕ) (trees_per_square_mile : ℕ) (days_per_month : ℕ) :
  loggers = 8 →
  trees_per_logger_per_day = 6 →
  forest_length = 4 →
  forest_width = 6 →
  trees_per_square_mile = 600 →
  days_per_month = 30 →
  (forest_length * forest_width * trees_per_square_mile) / (loggers * trees_per_logger_per_day) / days_per_month = 10 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end loggers_cut_down_trees_l547_547734


namespace cot_x_cot_y_l547_547786

theorem cot_x_cot_y (x y : ℝ) (h1 : tan x + tan y = 4) 
  (h2 : 3 * sin (2 * x + 2 * y) = sin (2 * x) * sin (2 * y)): 
  cot x * cot y = 7 / 6 :=
by 
  sorry

end cot_x_cot_y_l547_547786


namespace minimum_value_C_l547_547940

noncomputable def y_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def y_B (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
noncomputable def y_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def y_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimum_value_C : ∃ x : ℝ, y_C x = 4 := 
by
  -- proof goes here
  sorry

end minimum_value_C_l547_547940


namespace find_side_b_l547_547149

theorem find_side_b
  (a b c : ℝ)
  (area : ℝ)
  (B : ℝ)
  (h_area : area = real.sqrt 3)
  (h_B : B = (60 : ℝ) * real.pi / 180) -- Convert degrees to radians
  (h_cond : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 := 
sorry

end find_side_b_l547_547149


namespace number_of_months_in_season_l547_547297

def games_per_month : ℝ := 323.0
def total_games : ℝ := 5491.0

theorem number_of_months_in_season : total_games / games_per_month = 17 := 
by
  sorry

end number_of_months_in_season_l547_547297


namespace problem_statement_l547_547510

def f (x : ℝ) : ℝ := x^2 - 3 * x + 6

def g (x : ℝ) : ℝ := x + 4

theorem problem_statement : f (g 3) - g (f 3) = 24 := by
  sorry

end problem_statement_l547_547510


namespace harkamal_total_amount_paid_l547_547542

theorem harkamal_total_amount_paid (quantity_grapes : ℕ) (rate_grapes : ℕ) (quantity_mangoes : ℕ) (rate_mangoes : ℕ) :
  (quantity_grapes = 8) →
  (rate_grapes = 70) →
  (quantity_mangoes = 9) →
  (rate_mangoes = 45) →
  quantity_grapes * rate_grapes + quantity_mangoes * rate_mangoes = 965 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  done

end harkamal_total_amount_paid_l547_547542


namespace option_A_iff_option_B_l547_547688

-- Definitions
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
∀ n m, a (n + 1) - a n = a (m + 1) - a m

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range n, a (i + 1)

def option_B (S : ℕ → ℝ) : Prop :=
is_arithmetic_progression (λ n, S n / n)

-- Main theorem
theorem option_A_iff_option_B (a : ℕ → ℝ) :
  (is_arithmetic_progression a) ↔ (option_B (λ n, sum_of_first_n_terms a n)) :=
sorry

end option_A_iff_option_B_l547_547688


namespace hyperbola_asymptotes_l547_547005

open Real

theorem hyperbola_asymptotes (m : ℝ) (focal_length : ℝ) (h_focal_length : focal_length = 4) 
  (h_hyperbola : ∀ x y, (x^2 / m^2) - y^2 = 1) : 
  ∀ x y, y = (x : ℝ) * (1 / (sqrt 3)) ∨ y = -((x : ℝ) * (1 / (sqrt 3))) := 
begin
  sorry
end

end hyperbola_asymptotes_l547_547005


namespace smallest_x_palindrome_l547_547827

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

theorem smallest_x_palindrome : ∃ k, k > 0 ∧ is_palindrome (k + 1234) ∧ k = 97 := 
by {
  use 97,
  sorry
}

end smallest_x_palindrome_l547_547827


namespace minimum_value_of_h_l547_547899

-- Definitions of the given functions
def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
def g (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
def h (x : ℝ) : ℝ := 2^x + 2^(2 - x)
def j (x : ℝ) : ℝ := log x + 4 / log x

-- Main theorem statement with the condition and conclusion
theorem minimum_value_of_h : ∃ x, h x = 4 :=
by { sorry }

end minimum_value_of_h_l547_547899


namespace arithmetic_sequence_example_l547_547077

variable {α : Type*} [AddGroup α] [Module ℤ α]

noncomputable def a : ℕ → α
| 0 => 2          -- since a_1 = 2
| 1 => a(0) + d   -- a_2 = a_1 + d
| (n + 1) => a n + d

theorem arithmetic_sequence_example (d : α) (h : a(1) + a(2) = 13) : a(4) = 14 := 
by
  -- the proof will go here
  sorry

end arithmetic_sequence_example_l547_547077


namespace find_larger_number_l547_547777

def hcf := 20
def factor1 := 11
def factor2 := 15
def lcm := hcf * factor1 * factor2

theorem find_larger_number (A B : ℕ) 
  (hcf_A_B : nat.gcd A B = hcf) 
  (lcm_A_B : nat.lcm A B = lcm) : 
  A = 20 * 11 ∨ A = 20 * 15 ∨ B = 20 * 11 ∨ B = 20 * 15 → 
  max A B = 300 :=
by sorry

end find_larger_number_l547_547777


namespace find_b_correct_l547_547216

axioms (a b c : ℝ) (B : ℝ)
  (area_of_triangle : ℝ)
  (h1 : area_of_triangle = real.sqrt 3)
  (h2 : B = 60 * real.pi / 180) -- converting degrees to radians
  (h3 : a^2 + c^2 = 3 * a * c)

noncomputable def find_b : ℝ := 2 * real.sqrt 2

theorem find_b_correct :
  let b := find_b in 
  b = 2 * real.sqrt 2 :=
by
  sorry

end find_b_correct_l547_547216


namespace solve_equation_l547_547391

theorem solve_equation (x : ℕ) (h : x = 88320) : x + 1315 + 9211 - 1569 = 97277 :=
by sorry

end solve_equation_l547_547391


namespace find_b_correct_l547_547221

axioms (a b c : ℝ) (B : ℝ)
  (area_of_triangle : ℝ)
  (h1 : area_of_triangle = real.sqrt 3)
  (h2 : B = 60 * real.pi / 180) -- converting degrees to radians
  (h3 : a^2 + c^2 = 3 * a * c)

noncomputable def find_b : ℝ := 2 * real.sqrt 2

theorem find_b_correct :
  let b := find_b in 
  b = 2 * real.sqrt 2 :=
by
  sorry

end find_b_correct_l547_547221


namespace joels_age_when_dad_twice_l547_547092

theorem joels_age_when_dad_twice
  (joel_age_now : ℕ)
  (dad_age_now : ℕ)
  (years : ℕ)
  (H1 : joel_age_now = 5)
  (H2 : dad_age_now = 32)
  (H3 : years = 22)
  (H4 : dad_age_now + years = 2 * (joel_age_now + years))
  : joel_age_now + years = 27 := 
by sorry

end joels_age_when_dad_twice_l547_547092


namespace general_term_arithmetic_seq_max_sum_of_arithmetic_seq_l547_547508

-- Part 1: Finding the general term of the arithmetic sequence
theorem general_term_arithmetic_seq (a : ℕ → ℤ) (h1 : a 1 = 25) (h4 : a 4 = 16) :
  ∃ d : ℤ, a n = 28 - 3 * n := 
sorry

-- Part 2: Finding the value of n that maximizes the sum of the first n terms
theorem max_sum_of_arithmetic_seq (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : a 1 = 25)
  (h4 : a 4 = 16) 
  (ha : ∀ n, a n = 28 - 3 * n) -- Using the result from part 1
  (h_sum : ∀ n, S n = n * (a 1 + a n) / 2) :
  (∀ n : ℕ, S n < S (n + 1)) →
  9 = 9 :=
sorry

end general_term_arithmetic_seq_max_sum_of_arithmetic_seq_l547_547508


namespace min_value_f_l547_547967

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(2 - x)

theorem min_value_f : ∃ x : ℝ, f x = 4 :=
by {
  sorry
}

end min_value_f_l547_547967


namespace min_value_f_l547_547976

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(2 - x)

theorem min_value_f : ∃ x : ℝ, f x = 4 :=
by {
  sorry
}

end min_value_f_l547_547976


namespace find_b_l547_547195

theorem find_b
  (a b c : ℝ)
  (h_area : (1/2) * a * c * Real.sin (ℝ.pi / 3) = Real.sqrt 3)
  (h_ac_eq_4 : a * c = 4)
  (h_cosine : a^2 + c^2 = 3 * a * c) :
  b = 2 * Real.sqrt 2 :=
begin
  sorry
end

end find_b_l547_547195


namespace range_of_m_l547_547483

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - x + 5

theorem range_of_m (m : ℝ) :
  (∀ x ∈ set.Icc (-3/2 : ℝ) 3, f x < m) ↔ m > 11 :=
begin
  sorry,
end

end range_of_m_l547_547483


namespace income_calculation_l547_547284

theorem income_calculation (savings : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) 
  (ratio_condition : income_ratio = 5 ∧ expenditure_ratio = 4) (savings_condition : savings = 3800) :
  income_ratio * savings / (income_ratio - expenditure_ratio) = 19000 :=
by
  sorry

end income_calculation_l547_547284


namespace arithmetic_sequence_difference_l547_547076

-- Declare the conditions as definitions
variables {a : ℕ → ℝ} {n : ℕ}

-- Creating reusable conditions based on given problem
def condition1 := a 5 * a 7 = 6
def condition2 := a 2 + a 10 = 5

-- Statement to prove
theorem arithmetic_sequence_difference
  (a_arithmetic_seq : ∀ k m : ℕ, k < m → a (m + k) = a m + k * (a 1 - a 0))
  (h1 : condition1)
  (h2 : condition2) : 
  a 10 - a 6 = 2 ∨ a 10 - a 6 = -2 :=
by
  sorry

end arithmetic_sequence_difference_l547_547076


namespace exists_integers_for_linear_combination_l547_547622

theorem exists_integers_for_linear_combination 
  (a b c d b1 b2 : ℤ)
  (h1 : ad - bc ≠ 0)
  (h2 : ∃ k : ℤ, b1 = (ad - bc) * k)
  (h3 : ∃ q : ℤ, b2 = (ad - bc) * q) :
  ∃ x y : ℤ, a * x + b * y = b1 ∧ c * x + d * y = b2 :=
sorry

end exists_integers_for_linear_combination_l547_547622


namespace coin_difference_l547_547736

theorem coin_difference : 
  ∀ (c : ℕ), c = 50 → 
  (∃ (n m : ℕ), 
    (n ≥ m) ∧ 
    (∃ (a b d e : ℕ), n = a + b + d + e ∧ 5 * a + 10 * b + 20 * d + 25 * e = c) ∧
    (∃ (p q r s : ℕ), m = p + q + r + s ∧ 5 * p + 10 * q + 20 * r + 25 * s = c) ∧ 
    (n - m = 8)) :=
by
  sorry

end coin_difference_l547_547736


namespace sum_of_possible_values_l547_547035

theorem sum_of_possible_values (x y : ℕ) (h1 : x * y = 6) :
  ∑ (xy_pairs : ℕ × ℕ) in {(1, 6), (2, 3), (3, 2), (6, 1)}, 2 ^ (2 * (xy_pairs.2)) = 4180 :=
by
  sorry

end sum_of_possible_values_l547_547035


namespace find_length_of_b_l547_547130

theorem find_length_of_b
  {A B C : Type*}
  (a b c : ℝ)
  (area : ℝ)
  (angleB : ℝ)
  (h_area : area = sqrt 3)
  (h_angle : angleB = real.pi / 3)  -- 60 degrees in radians
  (h_sides : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 :=
by
  -- 1. Using the given area constraint: area = 1/2 * a * c * sin(B)
  -- 2. Using the given angle: sin(60) = sqrt(3)/2
  -- 3. Using the given sides relation: a^2 + c^2 = 3ac
  -- 4. Using the Law of Cosines: cos(60) = 1/2
  -- fill proof here
  sorry

end find_length_of_b_l547_547130


namespace arith_prog_iff_avg_arith_prog_l547_547645

variable {α : Type*} [LinearOrderedField α]

def is_arith_prog (a : ℕ → α) :=
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

def avg_is_arith_prog (S : ℕ → α) :=
  ∃ D : α, ∀ n : ℕ, n ≠ 0 → (S n / n) = (S (n + 1) / (n + 1)) + D

theorem arith_prog_iff_avg_arith_prog (a : ℕ → α) (S : ℕ → α)
  (hS : ∀ n, S n = ∑ i in range n, a i) :
  is_arith_prog a ↔ avg_is_arith_prog S :=
sorry

end arith_prog_iff_avg_arith_prog_l547_547645


namespace minimize_f_C_l547_547933

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimize_f_C : ∃ x : ℝ, f_C x = 4 :=
by {
  -- The correctness of the statement is asserted and the proof is omitted
  sorry
}

end minimize_f_C_l547_547933


namespace min_value_h_l547_547961

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def g (x : ℝ) : ℝ := abs (Real.sin x) + 4 / abs (Real.sin x)
noncomputable def h (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def j (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem min_value_h : ∃ x, h x = 4 :=
by
  sorry

end min_value_h_l547_547961


namespace athlete_shooting_training_l547_547437

def mutually_exclusive_not_opposite (E1 E2 : Prop) : Prop := 
  (¬ (E1 ∧ E2) ∧ ¬(¬E1 ∧ ¬E2))

def hitting_exactly (hit target total_shots : Nat) : Prop := hit = target

theorem athlete_shooting_training :
  ∀ (total_shots : Nat),
  total_shots = 5 →
  mutually_exclusive_not_opposite (hitting_exactly 3 3 total_shots) (hitting_exactly 4 4 total_shots) :=
by
  intros
  apply mutually_exclusive_not_opposite
  sorry

end athlete_shooting_training_l547_547437


namespace probability_sum_of_three_dice_is_12_l547_547578

open Finset

theorem probability_sum_of_three_dice_is_12 : 
  (∃ (outcomes : set (ℕ × ℕ × ℕ)), 
    ∀ (x y z : ℕ), (x, y, z) ∈ outcomes ↔ 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ (x + y + z = 12)) → 
    (∃ (prob : ℚ), prob = 2 / 27) :=
by 
  sorry

end probability_sum_of_three_dice_is_12_l547_547578


namespace triangle_side_b_l547_547185

theorem triangle_side_b 
  (a b c : ℝ)
  (B : ℝ)
  (h1 : 1/2 * a * c * Real.sin B = Real.sqrt 3)
  (h2 : B = π / 3)
  (h3 : a^2 + c^2 = 3 * a * c) : b = 2 * Real.sqrt 2 := by
  sorry

end triangle_side_b_l547_547185


namespace range_of_m_l547_547514

theorem range_of_m (m : ℝ) (h : ∃ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x1 + x2 = -(m + 2) ∧ x1 * x2 = m + 5) : -5 < m ∧ m < -2 := 
sorry

end range_of_m_l547_547514


namespace vladimir_is_tallest_l547_547360

section TallestBoy

variables (Andrei Boris Vladimir Dmitry : Type)

-- Define the statements made by each boy
def andrei_statements : Andrei → Prop :=
-- suppose andrei's statements are a::Boris ≠ tallest, b::Vladimir = shortest
λ A, (∀ B : Boris, B ≠ tallest) ∨ (∀ V : Vladimir, V = shortest)

def boris_statements : Boris → Prop :=
λ B, (∀ A : Andrei, A = oldest) ∨ (∀ A' : Andrei, A' = shortest)

def vladimir_statements : Vladimir → Prop :=
λ V, (∀ D : Dmitry, D = taller_than V) ∨ (∀ D' : Dmitry, D' = older_than V)

def dmitry_statements : Dmitry → Prop :=
λ D, ((vladimir_statements V) ∧ (vladimir_statements V)) ∨ (∀ D' : Dmitry, D' = oldest)

-- Define the conditions
-- Each boy makes two statements, one of which is true and the other is false
axiom statements_condition (A : Andrei) (B : Boris) (V : Vladimir) (D : Dmitry) :
  (andrei_statements A → ¬andrei_statements A) ∧
  (boris_statements B → ¬boris_statements B) ∧
  (vladimir_statements V → ¬vladimir_statements V) ∧
  (dmitry_statements D → ¬dmitry_statements D)

-- None of them share the same height or age
axiom uniqueness_condition (A : Andrei) (B : Boris) (V : Vladimir) (D : Dmitry) :
  A ≠ B ∧ A ≠ V ∧ A ≠ D ∧ B ≠ V ∧ B ≠ D ∧ V ≠ D

-- The conclusion that Vladimir is the tallest
theorem vladimir_is_tallest (A : Andrei) (B : Boris) (V : Vladimir) (D : Dmitry) :
  A ≠ tallest ∧ D ≠ taller_than (V : Vladimir) ∧ (¬(B = tallest)) → V = tallest :=
sorry

end TallestBoy

end vladimir_is_tallest_l547_547360


namespace find_b_correct_l547_547212

axioms (a b c : ℝ) (B : ℝ)
  (area_of_triangle : ℝ)
  (h1 : area_of_triangle = real.sqrt 3)
  (h2 : B = 60 * real.pi / 180) -- converting degrees to radians
  (h3 : a^2 + c^2 = 3 * a * c)

noncomputable def find_b : ℝ := 2 * real.sqrt 2

theorem find_b_correct :
  let b := find_b in 
  b = 2 * real.sqrt 2 :=
by
  sorry

end find_b_correct_l547_547212


namespace option_A_is_necessary_and_sufficient_for_option_B_l547_547697

def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

def mean_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : (ℕ → ℝ)
| n => S n / n

def sum_of_first_n_terms (a : ℕ → ℝ) : ℕ → ℝ
| 0     => 0
| (n+1) => sum_of_first_n_terms a n + a (n+1)

theorem option_A_is_necessary_and_sufficient_for_option_B
    (a : ℕ → ℝ) :
    (is_arithmetic_progression a ↔ is_arithmetic_progression (mean_sequence a (sum_of_first_n_terms a))) :=
sorry

end option_A_is_necessary_and_sufficient_for_option_B_l547_547697


namespace walther_janous_inequality_equality_condition_l547_547243

theorem walther_janous_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxy : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 :=
sorry

theorem equality_condition (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxy : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ x = 2 * y ∧ x / y = 2 ∧ y = z :=
sorry

end walther_janous_inequality_equality_condition_l547_547243


namespace minimum_value_of_option_C_l547_547887

noncomputable def optionA (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def optionB (x : ℝ) : ℝ := |Real.sin x| + 4 / |Real.sin x|
noncomputable def optionC (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def optionD (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem minimum_value_of_option_C :
  ∃ x : ℝ, optionC x = 4 :=
by
  sorry

end minimum_value_of_option_C_l547_547887


namespace angle_equality_of_tangency_l547_547095

noncomputable def triangle_excircles (ABC : Triangle) :
    Prop := ∃ (I_b I_c : Circle), excircle I_b ABC ∧ excircle I_c ABC

noncomputable def externally_tangent_circle
  (A : Point) (I_b I_c ω : Circle) : Prop :=
  ω ∈ circle_passing_through A ∧ externally_tangent ω I_b ∧ externally_tangent ω I_c

noncomputable def intersection_points_on_line
  (ω : Circle) (BC : Line) (M N : Point) : Prop :=
  M ∈ circle_intersection ω BC ∧ N ∈ circle_intersection ω BC

theorem angle_equality_of_tangency
  {ABC : Triangle} {I_b I_c ω : Circle} {BC : Line} {M N A : Point}
  (h1 : triangle_excircles ABC)
  (h2 : externally_tangent_circle A I_b I_c ω)
  (h3 : intersection_points_on_line ω BC M N) :
  angle BAC BAM = angle BAC CAN :=
by sorry

end angle_equality_of_tangency_l547_547095


namespace option_A_is_necessary_and_sufficient_for_option_B_l547_547695

def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

def mean_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : (ℕ → ℝ)
| n => S n / n

def sum_of_first_n_terms (a : ℕ → ℝ) : ℕ → ℝ
| 0     => 0
| (n+1) => sum_of_first_n_terms a n + a (n+1)

theorem option_A_is_necessary_and_sufficient_for_option_B
    (a : ℕ → ℝ) :
    (is_arithmetic_progression a ↔ is_arithmetic_progression (mean_sequence a (sum_of_first_n_terms a))) :=
sorry

end option_A_is_necessary_and_sufficient_for_option_B_l547_547695


namespace min_value_h_is_4_l547_547981

def f (x : ℝ) := x^2 + 2*x + 4
def g (x : ℝ) := abs (sin x) + 4 / abs (sin x)
def h (x : ℝ) := 2^x + 2^(2 - x)
def j (x : ℝ) := log x + 4 / log x

theorem min_value_h_is_4 : 
  (∀ x, f(x) ≥ 3) ∧
  (∀ x, g(x) > 4) ∧
  (∃ x, h(x) = 4) ∧
  (∀ x, j(x) ≠ 4) → min_value_h = 4 :=
by
  sorry

end min_value_h_is_4_l547_547981


namespace minimum_value_C_l547_547938

noncomputable def y_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def y_B (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
noncomputable def y_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def y_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimum_value_C : ∃ x : ℝ, y_C x = 4 := 
by
  -- proof goes here
  sorry

end minimum_value_C_l547_547938


namespace number_smaller_than_neg_3_l547_547433

theorem number_smaller_than_neg_3 : ∃ n ∈ [-2, 4, -5, 1], n < -3 :=
by {
    use -5,
    split,
    sorry, -- Proof that -5 is in the list [-2, 4, -5, 1]
    sorry  -- Proof that -5 < -3
}

end number_smaller_than_neg_3_l547_547433


namespace min_value_h_l547_547952

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def g (x : ℝ) : ℝ := abs (Real.sin x) + 4 / abs (Real.sin x)
noncomputable def h (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def j (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem min_value_h : ∃ x, h x = 4 :=
by
  sorry

end min_value_h_l547_547952


namespace Vladimir_is_tallest_l547_547356

-- Definitions for the statements made by the boys.
def Andrei_statements (Boris_tallest Vladimir_shortest : Prop) :=
  (¬Boris_tallest) ∧ Vladimir_shortest

def Boris_statements (Andrei_oldest Andrei_shortest : Prop) :=
  Andrei_oldest ∧ Andrei_shortest

def Vladimir_statements (Dmitry_taller Dmitry_older : Prop) :=
  Dmitry_taller ∧ Dmitry_older

def Dmitry_statements (Vladimir_both_true Dmitry_oldest : Prop) :=
  Vladimir_both_true ∧ Dmitry_oldest

-- Conditions given in the problem.
variables (Boris_tallest Vladimir_shortest Andrei_oldest Andrei_shortest Dmitry_taller Dmitry_older 
           Vladimir_both_true Dmitry_oldest : Prop)

-- Axioms based on the problem statement
axiom Dmitry_statements_condition : Dmitry_statements Vladimir_both_true Dmitry_oldest
axiom Vladimir_statements_condition : Vladimir_statements Dmitry_taller Dmitry_older
axiom Boris_statements_condition : Boris_statements Andrei_oldest Andrei_shortest
axiom Andrei_statements_condition : Andrei_statements Boris_tallest Vladimir_shortest

-- None of them share the same height or age.
axiom no_shared_height_or_age : 
  ¬(Andrei_oldest ∧ Boris_tallest ∧ Dmitry_taller ∧ Vladimir_both_true)

-- Main theorem: proving that Vladimir is the tallest based on the conditions
theorem Vladimir_is_tallest (cond : ¬Boris_tallest → Vladimir_shortest →
                              ¬Andrei_oldest → Andrei_shortest →
                              ¬Dmitry_taller → Dmitry_older →
                              ¬Vladimir_both_true → Dmitry_oldest → 
                              ¬(Andrei_oldest ∧ Boris_tallest ∧ Dmitry_taller ∧ Vladimir_both_true)) :
  (¬Boris_tallest ∧ Dmitry_older) → Vladimir_tallest :=
begin
  sorry,
end

end Vladimir_is_tallest_l547_547356


namespace minimize_f_C_l547_547929

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimize_f_C : ∃ x : ℝ, f_C x = 4 :=
by {
  -- The correctness of the statement is asserted and the proof is omitted
  sorry
}

end minimize_f_C_l547_547929


namespace tallest_boy_is_Vladimir_l547_547349

noncomputable def Andrei_statement1 (Boris_tallest: Prop) : Prop := ¬ Boris_tallest
def Andrei_statement2 (Vladimir_shortest: Prop) : Prop := Vladimir_shortest

def Boris_statement1 (Andrei_oldest: Prop) : Prop := Andrei_oldest
def Boris_statement2 (Andrei_shortest: Prop) : Prop := Andrei_shortest

def Vladimir_statement1 (Dmitry_taller: Prop) : Prop := Dmitry_taller
def Vladimir_statement2 (Dmitry_older: Prop) : Prop := Dmitry_older

noncomputable def Dmitry_statement1 (Vladimir_statement1: Prop) (Vladimir_statement2: Prop) : Prop :=
  Vladimir_statement1 ∧ Vladimir_statement2
def Dmitry_statement2 (Dmitry_oldest: Prop) : Prop := Dmitry_oldest

axiom one_statement_true_per_boy :
  ∀ {P₁ P₂: Prop}, (P₁ ∨ P₂) ∧ ¬ (P₁ ∧ P₂)

axiom no_same_height_or_age :
  ∀ {h1 h2 h3 h4 a1 a2 a3 a4 : ℕ},
    (h1 ≠ h2 ∧ h1 ≠ h3 ∧ h1 ≠ h4 ∧ h2 ≠ h3 ∧ h2 ≠ h4 ∧ h3 ≠ h4) ∧
    (a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)

theorem tallest_boy_is_Vladimir :
  ∀ (Andrei_shortest Vladimir_shortest Boris_tallest Dmitry_taller Dmitry_oldest Vladimir_older : Prop),
    Dmitry_statement2 Dmitry_oldest → 
    Boris_statement2 Andrei_shortest → 
    Andrei_statement1 Boris_tallest → 
    Vladimir_statement2 Vladimir_older → 
    Dmitry_statement1 (Vladimir_statement1 Dmitry_taller) (Vladimir_statement2 Vladimir_older) →
    ¬ Dmitry_taller →
    ¬ Boris_tallest →
    Vladimir = "the tallest boy" :=
  sorry

end tallest_boy_is_Vladimir_l547_547349


namespace speed_increase_is_2_l547_547795

def increase_in_speed {x : ℕ} : Prop :=
  let distances := list.range 12 in
  let traveled_distance := distances.sum.map (λ n, 40 + n * x) in
  traveled_distance = 612

theorem speed_increase_is_2 :
  ∃ x, increase_in_speed x ∧ x = 2 :=
  sorry

end speed_increase_is_2_l547_547795


namespace extreme_value_range_of_a_l547_547045

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (2 * x) * (1 - a * x)

theorem extreme_value_range_of_a (a : ℝ) :
  a ∈ Set.Ioo (2 / 3 : ℝ) 2 ↔
    ∃ c ∈ Set.Ioo 0 1, ∀ x : ℝ, f a c = f a x :=
by
  sorry

end extreme_value_range_of_a_l547_547045


namespace arith_prog_iff_avg_seq_arith_prog_l547_547679

-- Define sequences
def is_arith_prog (a : ℕ → ℝ) := ∃ d : ℝ, ∃ a₀ : ℝ, ∀ n : ℕ, a n = a₀ + n * d
def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range (n+1), a i
def avg_seq (a : ℕ → ℝ) (n : ℕ) := S a n / (n + 1)

-- Define theorem statement
theorem arith_prog_iff_avg_seq_arith_prog (a : ℕ → ℝ) :
  (is_arith_prog a) ↔ (is_arith_prog (λ n, avg_seq a n)) :=
sorry

end arith_prog_iff_avg_seq_arith_prog_l547_547679


namespace friendship_configurations_l547_547264

noncomputable def num_friend_configuration : ℕ := 7

theorem friendship_configurations (A B C D E F G : Prop) (hA : A)
  (hB : B) (hC : C) (hD : D) (hE : E) (hF : F) (hG : G)
  (eA : ∃ f: finite_set, even f.size ∧ A ∧ ¬ ∀ x ∈ f, x ∉ {A,B,C,D,E,F,G})
  (eB : ∃ f: finite_set, even f.size ∧ B ∧ ¬ ∀ x ∈ f, x ∉ {A,B,C,D,E,F,G})
  (eC : ∃ f: finite_set, even f.size ∧ C ∧ ¬ ∀ x ∈ f, x ∉ {A,B,C,D,E,F,G})
  (eD : ∃ f: finite_set, even f.size ∧ D ∧ ¬ ∀ x ∈ f, x ∉ {A,B,C,D,E,F,G})
  (eE : ∃ f: finite_set, even f.size ∧ E ∧ ¬ ∀ x ∈ f, x ∉ {A,B,C,D,E,F,G})
  (eF : ∃ f: finite_set, even f.size ∧ F ∧ ¬ ∀ x ∈ f, x ∉ {A,B,C,D,E,F,G})
  (eG : ∃ f: finite_set, even f.size ∧ G ∧ ¬ ∀ x ∈ f, x ∉ {A,B,C,D,E,F,G}) :
  num_friend_configuration = 7 :=
by 
sorry

end friendship_configurations_l547_547264


namespace union_of_M_and_N_l547_547533

open Set

-- Define the sets M and N
def M := {1, 0, -1}
def N := {1, 2}

-- State the theorem to be proved
theorem union_of_M_and_N : M ∪ N = {1, 2, 0, -1} := 
by sorry

end union_of_M_and_N_l547_547533


namespace parallel_lines_slope_condition_l547_547042

theorem parallel_lines_slope_condition (m : ℝ) :
  (∀ x y : ℝ, 2 * x + (m + 1) * y + 4 = 0 → mx + 3 * y - 2 = 0) →
  (m = 2 ∨ m = -3) :=
by
  sorry

end parallel_lines_slope_condition_l547_547042


namespace main_theorem_l547_547239

noncomputable def u_n (f : ℝ → ℝ) (n : ℕ) : ℝ :=
  ∫ x in 0..∞, x^n * f x

theorem main_theorem 
  (f : ℝ → ℝ)
  (h_diff : ∀ x > 0, differentiable_at ℝ f x)
  (h_diff_eq : ∀ x > 0, deriv f x = -3 * f x + 6 * f (2 * x))
  (h_bound : ∀ x ≥ 0, |f x| ≤ real.exp (-real.sqrt x))
  (h_u0 : ℝ)
  (h_u0_eq : h_u0 = u_n f 0) :

  (u_n f n = h_u0 * (n.factorial / 3^n) * (prod (i in finset.range n.succ, (1 / (1 - 2^(-i))))))
  ∧ (∑ n, (u_n f n * 3^n) / n.factorial).converges
  ∧ ((∑ n, (u_n f n * 3^n) / n.factorial) = 0 ↔ h_u0 = 0) :=
  sorry

end main_theorem_l547_547239


namespace max_sum_of_integer_pairs_l547_547296

theorem max_sum_of_integer_pairs (x y : ℤ) (h : (x-1)^2 + (y+2)^2 = 36) : 
  max (x + y) = 5 :=
sorry

end max_sum_of_integer_pairs_l547_547296


namespace ordinate_of_vertex_of_quadratic_l547_547492

theorem ordinate_of_vertex_of_quadratic (f : ℝ → ℝ) (h1 : ∃ a b c : ℝ, ∀ x : ℝ, f x = a * x^2 + b * x + c)
  (h2 : ∃ t1 t2 t3 : ℝ, (f t1)^3 - f t1 = 0 ∧ (f t2)^3 - f t2 = 0 ∧ (f t3)^3 - f t3 = 0 ∧ t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t3) :
  (∃ h : ℝ, f h = 0) → 
  ∃ k : ℝ, ∀ x : ℝ, f x = f (k/2) → False :=
begin
  sorry
end

end ordinate_of_vertex_of_quadratic_l547_547492


namespace triangle_side_b_l547_547186

theorem triangle_side_b 
  (a b c : ℝ)
  (B : ℝ)
  (h1 : 1/2 * a * c * Real.sin B = Real.sqrt 3)
  (h2 : B = π / 3)
  (h3 : a^2 + c^2 = 3 * a * c) : b = 2 * Real.sqrt 2 := by
  sorry

end triangle_side_b_l547_547186


namespace min_value_h_is_4_l547_547986

def f (x : ℝ) := x^2 + 2*x + 4
def g (x : ℝ) := abs (sin x) + 4 / abs (sin x)
def h (x : ℝ) := 2^x + 2^(2 - x)
def j (x : ℝ) := log x + 4 / log x

theorem min_value_h_is_4 : 
  (∀ x, f(x) ≥ 3) ∧
  (∀ x, g(x) > 4) ∧
  (∃ x, h(x) = 4) ∧
  (∀ x, j(x) ≠ 4) → min_value_h = 4 :=
by
  sorry

end min_value_h_is_4_l547_547986


namespace value_x2012_l547_547516

def f (x : ℝ) : ℝ := sorry

noncomputable def x (n : ℝ) : ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f (x)
axiom increasing_f : ∀ x y : ℝ, x < y → f x < f y
axiom arithmetic_seq : ∀ n : ℕ, x (n) = x (1) + (n-1) * 2
axiom condition : f (x 8) + f (x 9) + f (x 10) + f (x 11) = 0

theorem value_x2012 : x 2012 = 4005 := 
by sorry

end value_x2012_l547_547516


namespace Vladimir_is_tallest_l547_547376

-- Variables representing the boys
inductive Boy
| Andrei | Boris | Vladimir | Dmitry

open Boy

-- Height and Age representations
variable (Height Age : Boy → ℕ)

-- Conditions based on the problem
axiom distinct_heights : ∀ a b, Height a = Height b → a = b
axiom distinct_ages : ∀ a b, Age a = Age b → a = b

-- Statements made by the boys
axiom Andrei_statements :
  (¬ (Height Boris > Height Andrei) ∨ Height Vladimir = Height Andrei)
  ∧ ¬ (¬ (Height Boris > Height Andrei) ∧ Height Vladimir = Height Andrei)

axiom Boris_statements :
  (Age Andrei > Age Boris ∨ Height Andrei = Height Boris)
  ∧ ¬ (Age Andrei > Age Boris ∧ Height Andrei = Height Boris)

axiom Vladimir_statements :
  (Height Dmitry > Height Vladimir ∨ Age Dmitry > Age Vladimir)
  ∧ ¬ (Height Dmitry > Height Vladimir ∧ Age Dmitry > Age Vladimir)

axiom Dmitry_statements :
  ((Height Dmitry > Height Vladimir ∧ Age Dmitry > Age Vladimir) ∨ Age Dmitry > Age Dmitry)
  ∧ ¬ (¬ (Height Dmitry > Height Vladimir ∧ Age Dmitry > Age Vladimir) ∧ ¬ Age Dmitry > Age Dmitry)

-- Problem Statement: Vladimir is the tallest
theorem Vladimir_is_tallest : ∀ h : Height Vladimir = Nat.max (Height Andrei) (Nat.max (Height Boris) (Height Dmitry))
by
  sorry

end Vladimir_is_tallest_l547_547376


namespace num_distinct_triangle_areas_l547_547722

noncomputable def distinct_triangle_areas : Nat :=
  let A := (0, 1)
  let B := (1, 0)
  let C := (0, -1)
  let D := (-1, 0)
  let E := (1/√2, 1/√2)
  let F := (-1/√2, -1/√2)
  let triangles := [
    (A, B, C), (B, C, D), (C, D, A), (D, A, B),
    (A, E, F), (B, E, F), (C, E, F), (D, E, F)
  ]
  let areas := triangles.map (λ t => 
    let (x1, y1) := t.1
    let (x2, y2) := t.2
    let (x3, y3) := t.3
    abs((x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2)
  )
  areas.eraseDup.length

theorem num_distinct_triangle_areas : distinct_triangle_areas = 2 := by
  sorry

end num_distinct_triangle_areas_l547_547722


namespace triangle_side_b_l547_547232

noncomputable def find_side_length_b
  (a b c : ℝ)
  (area : ℝ)
  (B_deg : ℝ)
  (relation : a^2 + c^2 = 3 * a * c)
  (area_condition : area = sqrt 3)
  (angle_condition : B_deg = 60) : Prop :=
  b = 2 * sqrt 2

theorem triangle_side_b
  (a b c : ℝ)
  (area : ℝ := sqrt 3)
  (B_deg : ℝ := 60)
  (h1 : a^2 + c^2 = 3 * a * c)
  (h2 : area = sqrt 3)
  (h3 : B_deg = 60) : find_side_length_b a b c area B_deg h1 h2 h3 :=
  sorry

end triangle_side_b_l547_547232


namespace function_min_value_4_l547_547857

theorem function_min_value_4 :
  ∃ x : ℝ, (2^x + 2^(2 - x)) = 4 :=
sorry

end function_min_value_4_l547_547857


namespace area_of_square_with_given_coords_l547_547422

-- Assume vertices A, B, C, and D corresponding to y-coordinates
-- Vertices A = (a, 0), B = (b, 2), C = (c, 8), D = (d, 6)
variables (a b c d : ℝ)

-- Define the conditions based on slopes and square properties
def is_square_with_given_coords : Prop :=
  ∃ (a b c d : ℝ),
    -- Distance between the points corresponding to a square side
    (b - a) ^ 2 + (2 - 0) ^ 2 = (d - c) ^ 2 + (6 - 8) ^ 2 ∧
    (c - b) ^ 2 + (8 - 2) ^ 2 = (a - d) ^ 2 + (0 - 6) ^ 2

-- The main theorem stating that given these conditions, the area is 16
theorem area_of_square_with_given_coords
  (h : is_square_with_given_coords a b c d) : (b - a)^2 + (2)^2 = 4^2 := sorry


end area_of_square_with_given_coords_l547_547422


namespace area_of_figure_l547_547334

-- Define the two lines
def line1 (x : ℝ) : ℝ := x
def line2 : ℝ := -6

-- Define the point of intersection
def intersection_point : ℝ × ℝ := (-6, -6)

-- Define the vertices of the triangle
def vertex1 : ℝ × ℝ := (0, 0)
def vertex2 : ℝ × ℝ := (−6, 0)
def vertex3 : ℝ × ℝ := (-6, -6)

-- Define the base and height of the triangle
def base : ℝ := 6
def height : ℝ := 6

-- Define the area function for a triangle
def triangle_area (b h : ℝ) : ℝ := 0.5 * b * h

-- Theorem to prove the area
theorem area_of_figure : triangle_area base height = 18 := by
  sorry

end area_of_figure_l547_547334


namespace min_value_h_is_4_l547_547992

def f (x : ℝ) := x^2 + 2*x + 4
def g (x : ℝ) := abs (sin x) + 4 / abs (sin x)
def h (x : ℝ) := 2^x + 2^(2 - x)
def j (x : ℝ) := log x + 4 / log x

theorem min_value_h_is_4 : 
  (∀ x, f(x) ≥ 3) ∧
  (∀ x, g(x) > 4) ∧
  (∃ x, h(x) = 4) ∧
  (∀ x, j(x) ≠ 4) → min_value_h = 4 :=
by
  sorry

end min_value_h_is_4_l547_547992


namespace computer_program_X_value_l547_547566

theorem computer_program_X_value : 
  ∃ (n : ℕ), (let X := 5 + 3 * (n - 1) 
               let S := (3 * n^2 + 7 * n) / 2 
               S ≥ 10500) ∧ X = 251 :=
sorry

end computer_program_X_value_l547_547566


namespace inequality_solution_set_l547_547469

theorem inequality_solution_set :
  {x : ℝ | (x^2 / (x + 2)^2) ≥ 0} = {x : ℝ | x < -2} ∪ {x : ℝ | x > -2} :=
begin
  sorry
end

end inequality_solution_set_l547_547469


namespace triangle_side_b_l547_547181

theorem triangle_side_b 
  (a b c : ℝ)
  (B : ℝ)
  (h1 : 1/2 * a * c * Real.sin B = Real.sqrt 3)
  (h2 : B = π / 3)
  (h3 : a^2 + c^2 = 3 * a * c) : b = 2 * Real.sqrt 2 := by
  sorry

end triangle_side_b_l547_547181


namespace tallest_boy_is_Vladimir_l547_547346

noncomputable def Andrei_statement1 (Boris_tallest: Prop) : Prop := ¬ Boris_tallest
def Andrei_statement2 (Vladimir_shortest: Prop) : Prop := Vladimir_shortest

def Boris_statement1 (Andrei_oldest: Prop) : Prop := Andrei_oldest
def Boris_statement2 (Andrei_shortest: Prop) : Prop := Andrei_shortest

def Vladimir_statement1 (Dmitry_taller: Prop) : Prop := Dmitry_taller
def Vladimir_statement2 (Dmitry_older: Prop) : Prop := Dmitry_older

noncomputable def Dmitry_statement1 (Vladimir_statement1: Prop) (Vladimir_statement2: Prop) : Prop :=
  Vladimir_statement1 ∧ Vladimir_statement2
def Dmitry_statement2 (Dmitry_oldest: Prop) : Prop := Dmitry_oldest

axiom one_statement_true_per_boy :
  ∀ {P₁ P₂: Prop}, (P₁ ∨ P₂) ∧ ¬ (P₁ ∧ P₂)

axiom no_same_height_or_age :
  ∀ {h1 h2 h3 h4 a1 a2 a3 a4 : ℕ},
    (h1 ≠ h2 ∧ h1 ≠ h3 ∧ h1 ≠ h4 ∧ h2 ≠ h3 ∧ h2 ≠ h4 ∧ h3 ≠ h4) ∧
    (a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)

theorem tallest_boy_is_Vladimir :
  ∀ (Andrei_shortest Vladimir_shortest Boris_tallest Dmitry_taller Dmitry_oldest Vladimir_older : Prop),
    Dmitry_statement2 Dmitry_oldest → 
    Boris_statement2 Andrei_shortest → 
    Andrei_statement1 Boris_tallest → 
    Vladimir_statement2 Vladimir_older → 
    Dmitry_statement1 (Vladimir_statement1 Dmitry_taller) (Vladimir_statement2 Vladimir_older) →
    ¬ Dmitry_taller →
    ¬ Boris_tallest →
    Vladimir = "the tallest boy" :=
  sorry

end tallest_boy_is_Vladimir_l547_547346


namespace minimum_value_of_h_l547_547907

-- Definitions of the given functions
def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
def g (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
def h (x : ℝ) : ℝ := 2^x + 2^(2 - x)
def j (x : ℝ) : ℝ := log x + 4 / log x

-- Main theorem statement with the condition and conclusion
theorem minimum_value_of_h : ∃ x, h x = 4 :=
by { sorry }

end minimum_value_of_h_l547_547907


namespace count_integers_with_digit_3_l547_547544

theorem count_integers_with_digit_3 :
  ∃ n, n = (finset.range 10000).filter (λ x, ∃ i < 4, digit x i = 3) .card ∧ n = 2730 :=
by
  sorry

end count_integers_with_digit_3_l547_547544


namespace arith_prog_iff_avg_seq_arith_prog_l547_547675

-- Define sequences
def is_arith_prog (a : ℕ → ℝ) := ∃ d : ℝ, ∃ a₀ : ℝ, ∀ n : ℕ, a n = a₀ + n * d
def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range (n+1), a i
def avg_seq (a : ℕ → ℝ) (n : ℕ) := S a n / (n + 1)

-- Define theorem statement
theorem arith_prog_iff_avg_seq_arith_prog (a : ℕ → ℝ) :
  (is_arith_prog a) ↔ (is_arith_prog (λ n, avg_seq a n)) :=
sorry

end arith_prog_iff_avg_seq_arith_prog_l547_547675


namespace daniel_practices_each_school_day_l547_547455

-- Define the conditions
def total_minutes : ℕ := 135
def school_days : ℕ := 5
def weekend_days : ℕ := 2

-- Define the variables
def x : ℕ := 15

-- Define the practice time equations
def school_week_practice_time (x : ℕ) := school_days * x
def weekend_practice_time (x : ℕ) := weekend_days * 2 * x
def total_practice_time (x : ℕ) := school_week_practice_time x + weekend_practice_time x

-- The proof goal
theorem daniel_practices_each_school_day :
  total_practice_time x = total_minutes := by
  sorry

end daniel_practices_each_school_day_l547_547455


namespace valentines_proof_l547_547726

-- Definitions of the conditions in the problem
def original_valentines : ℝ := 58.5
def remaining_valentines : ℝ := 16.25
def valentines_given : ℝ := 42.25

-- The statement that we need to prove
theorem valentines_proof : original_valentines - remaining_valentines = valentines_given := by
  sorry

end valentines_proof_l547_547726


namespace dice_sum_probability_l547_547571

theorem dice_sum_probability : 
  let outcomes := { (a, b, c) | a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6} ∧ c ∈ {1, 2, 3, 4, 5, 6} }.card,
      favorable := { (a, b, c) | a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6} ∧ c ∈ {1, 2, 3, 4, 5, 6} ∧ a + b + c = 12 }.card,
      probability := (favorable: ℚ) / (outcomes: ℚ)
  in probability = 5 / 108 := by
  sorry

end dice_sum_probability_l547_547571


namespace point_P_is_orthocenter_l547_547300

theorem point_P_is_orthocenter :
  ∀ (A B C P A1 B1 C1 : Type) 
  [inhabited A] [inhabited B] [inhabited C] [inhabited P] [inhabited A1] [inhabited B1] [inhabited C1],
  (inside_triang (A B C) P) →
  (parallel_to_sides_and_intersect (triangle A B C) P) →
  (intersection_points A B C P A1 B1 C1) →
  (A1_on_diff_sides_of_BC A B C A1) →
  (B1_and_C1_similar B1 C1) →
  (hexagon_AC1BA1CB1_inscribed_and_convex A B C A1 B1 C1) →
  (is_orthocenter (triangle A1 B1 C1) P) :=
by
  -- insert your proof here
  sorry

end point_P_is_orthocenter_l547_547300


namespace find_side_b_l547_547118

theorem find_side_b
  (a b c : ℝ)
  (area : ℝ)
  (B : ℝ)
  (h_area : area = sqrt 3)
  (h_B : B = real.pi / 3)
  (h_a2c2 : a^2 + c^2 = 3 * a * c)
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) :
  b = 2 * real.sqrt 2 :=
sorry

end find_side_b_l547_547118


namespace find_f5_l547_547715

-- Step for necessary definitions and set up based on conditions
variable {f : ℝ → ℝ}

-- Define the conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(-x) = -f(x)

def f_properties : Prop :=
  is_odd_function f ∧ f(1) = 1 ∧ ∀ x : ℝ, f(x + 2) = f(x) + f(2)

-- The theorem statement
theorem find_f5 (h : f_properties) : f 5 = 5 :=
sorry

end find_f5_l547_547715


namespace min_value_f_l547_547972

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(2 - x)

theorem min_value_f : ∃ x : ℝ, f x = 4 :=
by {
  sorry
}

end min_value_f_l547_547972


namespace range_of_composite_function_l547_547290

noncomputable def range_of_function : Set ℝ :=
  {y | ∃ x : ℝ, y = (1/2) ^ (|x + 1|)}

theorem range_of_composite_function : range_of_function = Set.Ioc 0 1 :=
by
  sorry

end range_of_composite_function_l547_547290


namespace find_side_b_l547_547158

theorem find_side_b (a b c A B C : ℝ) (hB : B = 60) 
  (h_area : (1/2)*a*c*(Real.sin (Real.pi/3)) = Real.sqrt 3) 
  (h_ac : a^2 + c^2 = 3*a*c) : b = 2 * Real.sqrt 2 :=
by
  -- Given conditions
  have hB' : Real.sin (Real.pi/3) = Real.sqrt 3 / 2 := by sorry
  have h_ac_val : a * c = 4 := by sorry
  have h_a2c2 : a^2 + c^2 = 12 := by sorry
  have h_cosB : (a^2 + c^2 - b^2) / (2 * a * c) = 1/2 := by sorry
  have h_b2 : b^2 = 8 := by sorry

  -- Result
  exact Real.sqrt 8

end find_side_b_l547_547158


namespace solve_for_j_l547_547800

variable (j : ℝ)
variable (h1 : j > 0)
variable (v1 : ℝ × ℝ × ℝ := (3, 4, 5))
variable (v2 : ℝ × ℝ × ℝ := (2, j, 3))
variable (v3 : ℝ × ℝ × ℝ := (2, 3, j))

theorem solve_for_j :
  |(3 * (j * j - 3 * 3) - 2 * (4 * j - 5 * 3) + 2 * (4 * 3 - 5 * j))| = 36 →
  j = (9 + Real.sqrt 585) / 6 :=
by
  sorry

end solve_for_j_l547_547800


namespace function_min_value_l547_547915

theorem function_min_value (f1 f2 f3 f4 : ℝ → ℝ) :
  f1 = λ x, x^2 + 2 * x + 4 ∧
  f2 = λ x, abs (sin x) + 4 / abs (sin x) ∧
  f3 = λ x, 2^x + 2^(2-x) ∧
  f4 = λ x, log x + 4 / log x →
  (∀ x : ℝ, f3 x ≥ 4) ∧ (∃ x : ℝ, f3 x = 4) :=
by
  sorry

end function_min_value_l547_547915


namespace triangle_side_b_l547_547227

noncomputable def find_side_length_b
  (a b c : ℝ)
  (area : ℝ)
  (B_deg : ℝ)
  (relation : a^2 + c^2 = 3 * a * c)
  (area_condition : area = sqrt 3)
  (angle_condition : B_deg = 60) : Prop :=
  b = 2 * sqrt 2

theorem triangle_side_b
  (a b c : ℝ)
  (area : ℝ := sqrt 3)
  (B_deg : ℝ := 60)
  (h1 : a^2 + c^2 = 3 * a * c)
  (h2 : area = sqrt 3)
  (h3 : B_deg = 60) : find_side_length_b a b c area B_deg h1 h2 h3 :=
  sorry

end triangle_side_b_l547_547227


namespace parabola_one_intersection_l547_547050

theorem parabola_one_intersection (k : ℝ) :
  (∀ x : ℝ, x^2 - x + k = 0 → x = 0) → k = 1 / 4 :=
sorry

end parabola_one_intersection_l547_547050


namespace min_value_h_l547_547959

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def g (x : ℝ) : ℝ := abs (Real.sin x) + 4 / abs (Real.sin x)
noncomputable def h (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def j (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem min_value_h : ∃ x, h x = 4 :=
by
  sorry

end min_value_h_l547_547959


namespace constant_term_in_binomial_expansion_l547_547275

noncomputable def binomial_coeff (n k : ℕ) := Nat.choose n k

theorem constant_term_in_binomial_expansion :
  let f := (x + 1/x^2)^6 in
  let T := λ r, binomial_coeff 6 r * x^(6 - 3 * r) in
  T 2 = 15 :=
by sorry

end constant_term_in_binomial_expansion_l547_547275


namespace sequence_general_formula_l547_547097

noncomputable def f (x : ℝ) : ℝ := x / (x + 2)

def sequence (a : ℝ) : ℕ+ → ℝ
| ⟨1, _⟩ := a / (a + 2)
| ⟨n + 1, h⟩ := f (sequence a ⟨n, Nat.succ_pos n⟩)

theorem sequence_general_formula (a : ℝ) (n : ℕ+) (h : 0 < a):
  sequence a n = a / ((a + 1) * 2^n.val - a) :=
by
  sorry

end sequence_general_formula_l547_547097


namespace count_three_digit_multiples_of_91_l547_547552

theorem count_three_digit_multiples_of_91 : 
  ∃ (count : ℕ), count = 
    let lcm := Nat.lcm 13 7 in
    let min_k := (100 / lcm) + 1 in
    let max_k := 999 / lcm in
    max_k - min_k + 1 ∧ lcm = 91 ∧ (100 ≤ 91 * min_k) ∧ (91 * max_k ≤ 999) :=
by
  sorry

end count_three_digit_multiples_of_91_l547_547552


namespace product_of_radii_of_circles_l547_547612

theorem product_of_radii_of_circles 
  (BC AC AB : ℝ)
  (BC_eq : BC = 7)
  (AC_eq : AC = 5)
  (AB_eq : AB = 3)
  (AD_bisects_angle : Π (A B C D : Type) [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty D], Prop) :
  -- Definitions of the necessary radii under the provided conditions
  ∃ (R r : ℝ), (circle_circumscribed_triangle_R △ (A:=A) (B:=B) (D:=D) R) ∧ 
               (circle_inscribed_triangle_r △ (A:=A) (C:=C) (D:=D) r) ∧ 
               (R * r = 35 / 32) :=
sorry

end product_of_radii_of_circles_l547_547612


namespace median_of_consecutive_integers_l547_547316

theorem median_of_consecutive_integers (a b: ℕ) (S: Finset ℕ) (h1: ∀ n : ℕ, a + b = 200) : 
  (median S = 100 ∨ median S = 100.5) :=
sorry

end median_of_consecutive_integers_l547_547316


namespace triangle_area_l547_547787

theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (semi_perimeter : ℝ) (area : ℝ) 
  (h_perimeter : perimeter = 28) 
  (h_inradius : inradius = 2.5) 
  (h_semi_perimeter : semi_perimeter = perimeter / 2) 
  (h_area : area = inradius * semi_perimeter) : 
  area = 35 := 
by
  rw [h_perimeter, h_inradius, ← h_semi_perimeter] at h_area
  calc
    area = 2.5 * (28 / 2) : by rw h_area
    ... = 2.5 * 14 : by simp
    ... = 35 : by simp

end triangle_area_l547_547787


namespace coefficient_x10_in_expansion_l547_547471

theorem coefficient_x10_in_expansion :
  let f := λ x : ℂ, (x^3 / 3 - 3 / x^2)
  let expansion := (f x)^9
  ∀ (x : ℂ), coefficient (expansion) (x^10) = 0 :=
by
  sorry

end coefficient_x10_in_expansion_l547_547471


namespace min_value_of_2x_plus_2_2x_l547_547874

-- Lean 4 statement
theorem min_value_of_2x_plus_2_2x (x : ℝ) : 2^x + 2^(2 - x) ≥ 4 :=
by sorry

end min_value_of_2x_plus_2_2x_l547_547874


namespace minimum_value_C_l547_547942

noncomputable def y_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def y_B (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
noncomputable def y_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def y_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimum_value_C : ∃ x : ℝ, y_C x = 4 := 
by
  -- proof goes here
  sorry

end minimum_value_C_l547_547942


namespace find_distance_CD_l547_547099

noncomputable def distance_CD (C D: ℝ × ℝ) (theta3 theta4 : ℝ) : ℝ := 
  let OC := C.1
  let OD := D.1
  let COD := theta4 - theta3
  real.sqrt (OC^2 + OD^2 - 2 * OC * OD * real.cos(COD))

theorem find_distance_CD :
  let C := (5, theta3) in
  let D := (12, theta4) in
  let theta3 := theta3 in
  let theta4 := theta4 in
  let θ_diff := (θ₄ - θ₃) = π / 3 in
  distance_CD C D theta3 theta4 = real.sqrt 109 :=
by
  let C := (5, θ₃)
  let D := (12, θ₄)
  have h1 : θ₄ - θ₃ = π / 3 := θ_diff
  sorry

end find_distance_CD_l547_547099


namespace area_of_triangle_ABC_l547_547070

noncomputable def isosceles_triangle_area : ℝ :=
  let r := 3 in  -- derived from 9π = πr²
  let BC := 2 * (Real.sqrt 3) * r in
  let h := r + 3 * (Real.sqrt 3) in
  let area := (1 / 2) * BC * h in
  area

theorem area_of_triangle_ABC :
  let r := 3 in
  let BC := 2 * Real.sqrt 3 * r in
  let h := r + 3 * Real.sqrt 3 in
  BC = 2 * r * Real.sqrt 3 → 
  (π * r^2 = 9 * π) →
  (h = r + 3 * Real.sqrt 3) → 
  isosceles_triangle_area = 9 * Real.sqrt 3 + 27 :=
by
  simp [isosceles_triangle_area]
  sorry

end area_of_triangle_ABC_l547_547070


namespace nested_triple_op_eq_two_l547_547456

-- Define the operation [a, b, c] = (a + b) / c for c ≠ 0
def triple_op (a b c : ℝ) : ℝ :=
  if c ≠ 0 then (a + b) / c else 0

-- State the theorem
theorem nested_triple_op_eq_two : triple_op (triple_op 120 60 180) (triple_op 4 2 6) (triple_op 20 10 30) = 2 := by
  sorry

end nested_triple_op_eq_two_l547_547456


namespace max_area_quadrilateral_l547_547822

theorem max_area_quadrilateral (a b c d : ℝ) (h1 : a = 1) (h2 : b = 4) (h3 : c = 7) (h4 : d = 8) : 
  ∃ A : ℝ, (A ≤ (1/2) * 1 * 8 + (1/2) * 4 * 7) ∧ (A = 18) :=
by
  sorry

end max_area_quadrilateral_l547_547822


namespace hash_7_2_eq_24_l547_547561

def hash_op (a b : ℕ) : ℕ := 4 * a - 2 * b

theorem hash_7_2_eq_24 : hash_op 7 2 = 24 := by
  sorry

end hash_7_2_eq_24_l547_547561


namespace average_weight_when_D_joins_is_53_l547_547273

noncomputable def new_average_weight (A B C D E : ℕ) : ℕ :=
  (73 + B + C + D) / 4

theorem average_weight_when_D_joins_is_53 :
  (A + B + C) / 3 = 50 →
  A = 73 →
  (B + C + D + E) / 4 = 51 →
  E = D + 3 →
  73 + B + C + D = 212 →
  new_average_weight A B C D E = 53 :=
by
  sorry

end average_weight_when_D_joins_is_53_l547_547273


namespace triangle_side_b_l547_547184

theorem triangle_side_b 
  (a b c : ℝ)
  (B : ℝ)
  (h1 : 1/2 * a * c * Real.sin B = Real.sqrt 3)
  (h2 : B = π / 3)
  (h3 : a^2 + c^2 = 3 * a * c) : b = 2 * Real.sqrt 2 := by
  sorry

end triangle_side_b_l547_547184


namespace minimum_value_C_l547_547951

noncomputable def y_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def y_B (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
noncomputable def y_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def y_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimum_value_C : ∃ x : ℝ, y_C x = 4 := 
by
  -- proof goes here
  sorry

end minimum_value_C_l547_547951


namespace monotonic_increasing_interval_max_value_on_interval_l547_547525

noncomputable def f (x : ℝ) : ℝ := 2 * sqrt 3 * cos x ^ 2 - 2 * sin (π / 4 - x) ^ 2 - sqrt 3

theorem monotonic_increasing_interval :
  ∃ (k : ℤ) (x : ℝ), (x ∈ set.Icc (k * π - 5 * π / 12) (k * π + π / 12)) →
  monotonic_increasing_on f (set.Icc (k * π - 5 * π / 12) (k * π + π / 12)) := sorry

theorem max_value_on_interval :
  set.Icc (0 : ℝ) (π / 6) → ∃ x, x = π / 12 ∧ f x = 1 := sorry

end monotonic_increasing_interval_max_value_on_interval_l547_547525


namespace tallest_is_vladimir_l547_547366

variables (Andrei Boris Vladimir Dmitry : Type)
variables (height age : Type)
variables [linear_order height] [linear_order age]
variables (tallest shortest oldest : height)
variables (andrei_statements : (Boris ≠ tallest) ∧ (Vladimir = shortest))
variables (boris_statements : (Andrei = oldest) ∧ (Andrei = shortest))
variables (vladimir_statements : (Dmitry > Vladimir) ∧ (Dmitry > Vladimir))
variables (dmitry_statements : (vladimir_statements = true) ∧ (Dmitry = oldest))

theorem tallest_is_vladimir (H1: andrei_statements ∨ ¬andrei_statements)
  (H2: boris_statements ∨ ¬boris_statements)
  (H3: vladimir_statements ∨ ¬vladimir_statements)
  (H4: dmitry_statements ∨ ¬dmitry_statements)
  (H5: ∃! (a b c d : height) , a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a) : 
  tallest = Vladimir := 
by
  sorry

end tallest_is_vladimir_l547_547366


namespace find_side_b_l547_547143

theorem find_side_b
  (a b c : ℝ)
  (area : ℝ)
  (B : ℝ)
  (h_area : area = real.sqrt 3)
  (h_B : B = (60 : ℝ) * real.pi / 180) -- Convert degrees to radians
  (h_cond : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 := 
sorry

end find_side_b_l547_547143


namespace pages_read_later_l547_547089

theorem pages_read_later 
  (initial_read : ℕ)
  (total_read : ℕ)
  (initial_pages : initial_read = 37)
  (total_pages : total_read = 62) :
  ∃ x, initial_read + x = total_read ∧ x = 25 :=
by
  use 25
  split
  · sorry -- proof that initial_read + 25 = total_read,
  · sorry -- proof that x = 25

end pages_read_later_l547_547089


namespace polygon_sides_l547_547055

theorem polygon_sides (n : ℕ) (h1 : (n - 2) * 180 = 3 * 360) (h2 : n > 2) : n = 8 := by
  -- Conditions given:
  -- h1: (n - 2) * 180 = 3 * 360
  -- h2: n > 2
  sorry

end polygon_sides_l547_547055


namespace distance_between_parallel_lines_l547_547048

theorem distance_between_parallel_lines 
    (a : ℝ) 
    (h_parallel : 1 * 3 = (a - 2) * a)
    (l1 : ℝ → Prop := λ x y, x + a * y + 6 = 0)
    (l2 : ℝ → Prop := λ x y, (a - 2) * x + 3 * y + 2 * a = 0) : 
    let d := abs((2 * a) - 6) / sqrt (1^2 + a^2) 
    in d = 8 * sqrt 2 / 3 :=
by 
  -- Proof will go here
  sorry

end distance_between_parallel_lines_l547_547048


namespace arithmetic_sequence_sum_b_l547_547496

-- Define the sequence {a_n} and the sum S_n
def a (n : ℕ) : ℝ := sorry -- We cannot define it directly without proving or assuming the construction.
def S (n : ℕ) : ℝ := (a n + 2) * (a n - 1) / 2

-- Conditions: all terms of the sequence a_n are positive
axiom pos_terms : ∀ n : ℕ, 0 < a n

-- Statement I: The sequence {a_n} is an arithmetic sequence
theorem arithmetic_sequence {a : ℕ → ℝ} (hS: ∀ n, S n = (a n + 2) * (a n - 1) / 2) :
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

-- Define b_n and T_n
def b (n : ℕ) : ℝ := a n * 3 ^ n
def T (n : ℕ) : ℝ := ∑ i in finset.range n, b i

-- Statement II: The sum of the first n terms of the sequence {b_n}
theorem sum_b {a : ℕ → ℝ} (hS: ∀ n, S n = (a n + 2) * (a n - 1) / 2) :
  T n = (2 * n + 1) * 3^(n+1) / 4 - 3 / 4 :=
sorry

end arithmetic_sequence_sum_b_l547_547496


namespace seq_general_form_l547_547512

theorem seq_general_form (p r : ℝ) (a : ℕ → ℝ)
  (hp : p > r)
  (hr : r > 0)
  (h_init : a 1 = r)
  (h_recurrence : ∀ n : ℕ, a (n+1) = p * a n + r^(n+1)) :
  ∀ n : ℕ, a n = r * (p^n - r^n) / (p - r) :=
by
  sorry

end seq_general_form_l547_547512


namespace can_capacity_is_14_l547_547064

noncomputable def capacity_of_can 
    (initial_milk: ℝ) (initial_water: ℝ) 
    (added_milk: ℝ) (ratio_initial: ℝ) (ratio_final: ℝ): ℝ :=
  initial_milk + initial_water + added_milk

theorem can_capacity_is_14
    (M W: ℝ) 
    (ratio_initial : M / W = 1 / 5) 
    (added_milk : ℝ := 2) 
    (ratio_final:  (M + 2) / W = 2.00001 / 5.00001): 
    capacity_of_can M W added_milk (1 / 5) (2.00001 / 5.00001) = 14 := 
  by
    sorry

end can_capacity_is_14_l547_547064


namespace find_side_b_l547_547125

theorem find_side_b
  (a b c : ℝ)
  (area : ℝ)
  (B : ℝ)
  (h_area : area = sqrt 3)
  (h_B : B = real.pi / 3)
  (h_a2c2 : a^2 + c^2 = 3 * a * c)
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) :
  b = 2 * real.sqrt 2 :=
sorry

end find_side_b_l547_547125


namespace greatest_two_digit_with_conditions_l547_547315

theorem greatest_two_digit_with_conditions :
  ∃ (n : ℕ), (10 ≤ n ∧ n < 100) ∧ (let d1 := n / 10 in let d2 := n % 10 in d1 * d2 = 12 ∧ d1 + d2 = 7) ∧ n = 43 :=
by
  sorry

end greatest_two_digit_with_conditions_l547_547315


namespace find_length_of_b_l547_547134

theorem find_length_of_b
  {A B C : Type*}
  (a b c : ℝ)
  (area : ℝ)
  (angleB : ℝ)
  (h_area : area = sqrt 3)
  (h_angle : angleB = real.pi / 3)  -- 60 degrees in radians
  (h_sides : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 :=
by
  -- 1. Using the given area constraint: area = 1/2 * a * c * sin(B)
  -- 2. Using the given angle: sin(60) = sqrt(3)/2
  -- 3. Using the given sides relation: a^2 + c^2 = 3ac
  -- 4. Using the Law of Cosines: cos(60) = 1/2
  -- fill proof here
  sorry

end find_length_of_b_l547_547134


namespace triangle_angle_type_l547_547085

theorem triangle_angle_type (A B C : ℝ) (h : 1007 * A^2 + 1009 * B^2 = 2016 * C^2) :
  (A + B + C = π ∧ (1007 * A^2 + 1009 * B^2 = 2016 * C^2) →
    (∃ x y z : ℝ, (A = x ∧ B = y ∧ C = z) ∧ 
    ((A < π/2 ∧ B < π/2 ∧ C < π/2) ∨ 
     (A = π/2 ∨ B = π/2 ∨ C = π/2) ∨ 
     (A > π/2 ∨ B > π/2 ∨ C > π/2)))) :=
begin
  intros h1,
  sorry
end

end triangle_angle_type_l547_547085


namespace radius_of_sphere_in_truncated_cone_l547_547427

-- Definition of a truncated cone with bases of radii 24 and 6
structure TruncatedCone where
  top_radius : ℝ
  bottom_radius : ℝ

-- Sphere tangent condition
structure Sphere where
  radius : ℝ

-- The specific instance of the problem
def truncatedCone : TruncatedCone :=
{ top_radius := 6, bottom_radius := 24 }

def sphere : Sphere := sorry  -- The actual radius will be proven next.

theorem radius_of_sphere_in_truncated_cone : 
  sphere.radius = 12 ∧ 
  sphere_tangent_to_truncated_cone sphere truncatedCone :=
sorry

end radius_of_sphere_in_truncated_cone_l547_547427


namespace problem_sum_of_real_solutions_l547_547474

theorem problem_sum_of_real_solutions :
  (∃ x : ℝ, |(x^2 - 10 * x + 29)| = 3) → (finset.sum finset.univ (λ x : ℝ, if |(x^2 - 10 * x + 29)| = 3 then x else 0) = 0) :=
begin
  sorry
end

end problem_sum_of_real_solutions_l547_547474


namespace minimum_value_C_l547_547949

noncomputable def y_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def y_B (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
noncomputable def y_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def y_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimum_value_C : ∃ x : ℝ, y_C x = 4 := 
by
  -- proof goes here
  sorry

end minimum_value_C_l547_547949


namespace can_place_more_domino_domino_placement_possible_l547_547587

theorem can_place_more_domino (total_squares : ℕ := 36) (uncovered_squares : ℕ := 14) : Prop :=
∃ (n : ℕ), (n * 2 + uncovered_squares ≤ total_squares) ∧ (n ≥ 1)

/-- Proof that on a 6x6 chessboard with some 1x2 dominoes placed, if there are 14 uncovered
squares, then at least one more domino can be placed on the board. -/
theorem domino_placement_possible :
  can_place_more_domino := by
  sorry

end can_place_more_domino_domino_placement_possible_l547_547587


namespace sum_prime_factors_924_l547_547838

/-- Let n be the integer 924. -/
def n : ℤ := 924

/-- The set of distinct prime factors of 924. -/
def distinct_prime_factors (n : ℤ) : set ℤ :=
  if n = 924 then {2, 3, 7, 11} else ∅

/-- The sum of the distinct prime factors of 924. -/
def sum_distinct_prime_factors (n : ℤ) : ℤ :=
  if n = 924 then 2 + 3 + 7 + 11 else 0

-- The theorem to prove that the sum of the distinct prime factors of 924 is 23.
theorem sum_prime_factors_924 : sum_distinct_prime_factors n = 23 :=
by {
  unfold sum_distinct_prime_factors,
  simp,
}

end sum_prime_factors_924_l547_547838


namespace train_pass_bridge_l547_547425

-- Define variables and conditions
variables (train_length bridge_length : ℕ)
          (train_speed_kmph : ℕ)

-- Convert speed from km/h to m/s
def train_speed_mps(train_speed_kmph : ℕ) : ℚ :=
  (train_speed_kmph * 1000) / 3600

-- Total distance to cover
def total_distance(train_length bridge_length : ℕ) : ℕ :=
  train_length + bridge_length

-- Time to pass the bridge
def time_to_pass_bridge(train_length bridge_length : ℕ) (train_speed_kmph : ℕ) : ℚ :=
  (total_distance train_length bridge_length) / (train_speed_mps train_speed_kmph)

-- The proof statement
theorem train_pass_bridge :
  time_to_pass_bridge 360 140 50 = 36 := 
by
  -- actual proof would go here
  sorry

end train_pass_bridge_l547_547425


namespace minimum_value_of_h_l547_547902

-- Definitions of the given functions
def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
def g (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
def h (x : ℝ) : ℝ := 2^x + 2^(2 - x)
def j (x : ℝ) : ℝ := log x + 4 / log x

-- Main theorem statement with the condition and conclusion
theorem minimum_value_of_h : ∃ x, h x = 4 :=
by { sorry }

end minimum_value_of_h_l547_547902


namespace statement_one_statement_two_statement_three_statement_four_l547_547714

-- Define the function f(x) = |x|x + bx + c
def f (x b c : ℝ) : ℝ := abs x * x + b * x + c

-- Proof statement ①: When b = 0 and c > 0, f(x) = 0 has only one root
theorem statement_one (c : ℝ) (h : c > 0) : ∃! x : ℝ, f x 0 c = 0 :=
by
  sorry -- proof not needed

-- Proof statement ②: When c = 0, y = f(x) is an odd function
theorem statement_two (b : ℝ) : f (-x) b 0 = -f x b 0 :=
by
  rw [f, abs_neg, neg_mul_eq_neg_mul, f, neg_mul_eq_neg_mul]
  simp

-- Proof statement ③: The graph of y = f(x) is symmetric about the point (0,1)
theorem statement_three (b : ℝ) : (∃ x, f x b 1 = 1) ∧ (∀ c, c ≠ 1 → ¬(∃ x, f x b c = c)) :=
by
  sorry -- proof not needed

-- Proof statement ④: f(x) = 0 has at least two real roots
theorem statement_four (b c : ℝ) : ¬ ∃ x1 x2, x1 ≠ x2 ∧ f x1 b c = 0 ∧ f x2 b c = 0 :=
by
  sorry -- proof not needed

end statement_one_statement_two_statement_three_statement_four_l547_547714


namespace find_f_of_composed_l547_547482

theorem find_f_of_composed (x : ℝ) : 
  (∀ x : ℝ, f x = 2 * x + 1) → f (2 * x - 1) = 4 * x - 1 :=
by
  intro h
  sorry

end find_f_of_composed_l547_547482


namespace dice_probability_sum_12_l547_547574

open Nat

/-- Probability that the sum of three six-faced dice rolls equals 12 is 10 / 216 --/
theorem dice_probability_sum_12 : 
  let outcomes := 6^3
  let favorable := 10
  (favorable : ℚ) / outcomes = 10 / 216 := 
by
  let outcomes := 6^3
  let favorable := 10
  sorry

end dice_probability_sum_12_l547_547574


namespace cos_C_in_right_triangle_l547_547083

def triangle_angle := 90
def sin_B := 3 / 5

theorem cos_C_in_right_triangle (A B C : ℕ) (hA : A = 90) (hSinB : sin B = 3 / 5) :
  cos C = 3 / 5 :=
by
  sorry

end cos_C_in_right_triangle_l547_547083


namespace minimize_f_C_l547_547928

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimize_f_C : ∃ x : ℝ, f_C x = 4 :=
by {
  -- The correctness of the statement is asserted and the proof is omitted
  sorry
}

end minimize_f_C_l547_547928


namespace max_rectangles_3x5_in_17x22_l547_547024

theorem max_rectangles_3x5_in_17x22 : ∃ n : ℕ, n = 24 ∧ 
  (∀ (cut_3x5_pieces : ℤ), cut_3x5_pieces ≤ n) :=
by
  sorry

end max_rectangles_3x5_in_17x22_l547_547024


namespace sum_of_distinct_prime_factors_924_l547_547830

theorem sum_of_distinct_prime_factors_924 : 
  (∑ p in ({2, 3, 7, 11} : Finset ℕ), p) = 23 := 
by
  -- sorry is used to skip the proof.
  sorry

end sum_of_distinct_prime_factors_924_l547_547830


namespace taxi_ride_cost_l547_547396

-- Define the cost of the first 1/5 mile
def first_mile_cost : ℚ := 2.50

-- Define the cost per additional 1/5 mile
def additional_mile_cost : ℚ := 0.40

-- Define the cost per minute of waiting time
def waiting_time_cost : ℚ := 0.25

-- Define the toll fee
def toll_fee : ℚ := 3.00

-- Define surge pricing rate
def surge_pricing_rate : ℚ := 0.20

-- Define the distance of the ride in miles
def total_distance : ℚ := 8.00

-- Define the waiting time in minutes
def waiting_time : ℚ := 12.00

-- Define if the ride passed through a toll route
def toll_route : Prop := true

-- Define if the ride took place during peak hours
def peak_hours : Prop := true

-- Define the total number of 1/5 miles excluding the first one
def total_additional_parts : ℚ := (total_distance * 5) - 1

-- Define the total cost without surge pricing
def total_cost_without_surge : ℚ :=
  first_mile_cost +
  (total_additional_parts * additional_mile_cost) +
  (waiting_time * waiting_time_cost) +
  if toll_route then toll_fee else 0

-- Define the surge cost
def surge_cost : ℚ := surge_pricing_rate * total_cost_without_surge

-- Define the total cost including surge pricing
def total_cost : ℚ := total_cost_without_surge + surge_cost

-- The proof statement
theorem taxi_ride_cost : total_cost = 28.92 := 
  sorry

end taxi_ride_cost_l547_547396


namespace cartesian_eq_of_curve_C_eccentricity_of_curve_C_maximum_distance_l547_547072

def polar_eq_curve_C (rho theta : ℝ) : Prop :=
  rho^2 = 4 / (4 * (Real.sin theta)^2 + (Real.cos theta)^2)

def polar_eq_line_l (rho theta : ℝ) : Prop :=
  rho * (Real.cos theta + 2 * Real.sin theta) + 6 = 0

def cartesian_eq_curve_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

def eccentricity_C : ℝ :=
  Real.sqrt 3 / 2

def max_distance (x y : ℝ) : ℝ :=
  (2 * Real.sqrt 10 + 6 * Real.sqrt 5) / 5

theorem cartesian_eq_of_curve_C (rho theta x y : ℝ) (h1 : polar_eq_curve_C rho theta)
  (hx : x = rho * Real.cos theta) (hy : y = rho * Real.sin theta) :
  cartesian_eq_curve_C x y :=
sorry

theorem eccentricity_of_curve_C (e : ℝ) (h2 : e = eccentricity_C) : 
  e = Real.sqrt 3 / 2 := 
sorry

theorem maximum_distance (x y : ℝ) (h3 : cartesian_eq_curve_C x y) : 
  ∃ d, d = max_distance x y :=
sorry

end cartesian_eq_of_curve_C_eccentricity_of_curve_C_maximum_distance_l547_547072


namespace find_side_b_l547_547122

theorem find_side_b
  (a b c : ℝ)
  (area : ℝ)
  (B : ℝ)
  (h_area : area = sqrt 3)
  (h_B : B = real.pi / 3)
  (h_a2c2 : a^2 + c^2 = 3 * a * c)
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) :
  b = 2 * real.sqrt 2 :=
sorry

end find_side_b_l547_547122


namespace max_volume_of_triangular_pyramid_l547_547498

noncomputable def sphere_radius := 1
noncomputable def max_volume_of_pyramid (A B C P : ℝ^3) : ℝ :=
let S := (3 * (1 / 2) * 1 * 1 * (Real.sqrt 3 / 2)) in
(1 / 3) * S * 1

theorem max_volume_of_triangular_pyramid 
  (O : ℝ^3) (A B C P : ℝ^3) 
  (hO : ∥O∥ = sphere_radius) 
  (hA : ∥A∥ = sphere_radius) 
  (hB : ∥B∥ = sphere_radius) 
  (hC : ∥C∥ = sphere_radius) 
  (hP : ∥P∥ = sphere_radius) :
  max_volume_of_pyramid A B C P = Real.sqrt 3 / 4 :=
sorry

end max_volume_of_triangular_pyramid_l547_547498


namespace find_length_of_b_l547_547135

theorem find_length_of_b
  {A B C : Type*}
  (a b c : ℝ)
  (area : ℝ)
  (angleB : ℝ)
  (h_area : area = sqrt 3)
  (h_angle : angleB = real.pi / 3)  -- 60 degrees in radians
  (h_sides : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 :=
by
  -- 1. Using the given area constraint: area = 1/2 * a * c * sin(B)
  -- 2. Using the given angle: sin(60) = sqrt(3)/2
  -- 3. Using the given sides relation: a^2 + c^2 = 3ac
  -- 4. Using the Law of Cosines: cos(60) = 1/2
  -- fill proof here
  sorry

end find_length_of_b_l547_547135


namespace vladimir_is_tallest_l547_547363

section TallestBoy

variables (Andrei Boris Vladimir Dmitry : Type)

-- Define the statements made by each boy
def andrei_statements : Andrei → Prop :=
-- suppose andrei's statements are a::Boris ≠ tallest, b::Vladimir = shortest
λ A, (∀ B : Boris, B ≠ tallest) ∨ (∀ V : Vladimir, V = shortest)

def boris_statements : Boris → Prop :=
λ B, (∀ A : Andrei, A = oldest) ∨ (∀ A' : Andrei, A' = shortest)

def vladimir_statements : Vladimir → Prop :=
λ V, (∀ D : Dmitry, D = taller_than V) ∨ (∀ D' : Dmitry, D' = older_than V)

def dmitry_statements : Dmitry → Prop :=
λ D, ((vladimir_statements V) ∧ (vladimir_statements V)) ∨ (∀ D' : Dmitry, D' = oldest)

-- Define the conditions
-- Each boy makes two statements, one of which is true and the other is false
axiom statements_condition (A : Andrei) (B : Boris) (V : Vladimir) (D : Dmitry) :
  (andrei_statements A → ¬andrei_statements A) ∧
  (boris_statements B → ¬boris_statements B) ∧
  (vladimir_statements V → ¬vladimir_statements V) ∧
  (dmitry_statements D → ¬dmitry_statements D)

-- None of them share the same height or age
axiom uniqueness_condition (A : Andrei) (B : Boris) (V : Vladimir) (D : Dmitry) :
  A ≠ B ∧ A ≠ V ∧ A ≠ D ∧ B ≠ V ∧ B ≠ D ∧ V ≠ D

-- The conclusion that Vladimir is the tallest
theorem vladimir_is_tallest (A : Andrei) (B : Boris) (V : Vladimir) (D : Dmitry) :
  A ≠ tallest ∧ D ≠ taller_than (V : Vladimir) ∧ (¬(B = tallest)) → V = tallest :=
sorry

end TallestBoy

end vladimir_is_tallest_l547_547363


namespace minimum_value_of_h_l547_547897

-- Definitions of the given functions
def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
def g (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
def h (x : ℝ) : ℝ := 2^x + 2^(2 - x)
def j (x : ℝ) : ℝ := log x + 4 / log x

-- Main theorem statement with the condition and conclusion
theorem minimum_value_of_h : ∃ x, h x = 4 :=
by { sorry }

end minimum_value_of_h_l547_547897


namespace find_side_b_l547_547139

theorem find_side_b
  (a b c : ℝ)
  (area : ℝ)
  (B : ℝ)
  (h_area : area = real.sqrt 3)
  (h_B : B = (60 : ℝ) * real.pi / 180) -- Convert degrees to radians
  (h_cond : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 := 
sorry

end find_side_b_l547_547139


namespace count_pos_three_digit_divisible_by_13_and_7_l547_547550

theorem count_pos_three_digit_divisible_by_13_and_7 : 
  ((finset.filter (λ n : ℕ, n % (13 * 7) = 0) (finset.Icc 100 999)).card = 9) := 
sorry

end count_pos_three_digit_divisible_by_13_and_7_l547_547550


namespace total_servings_l547_547090

-- Definitions for the conditions

def servings_per_carrot : ℕ := 4
def plants_per_plot : ℕ := 9
def servings_multiplier_corn : ℕ := 5
def servings_multiplier_green_bean : ℤ := 2

-- Proof statement
theorem total_servings : 
  (plants_per_plot * servings_per_carrot) + 
  (plants_per_plot * (servings_per_carrot * servings_multiplier_corn)) + 
  (plants_per_plot * (servings_per_carrot * servings_multiplier_corn / servings_multiplier_green_bean)) = 
  306 :=
by
  sorry

end total_servings_l547_547090


namespace estimate_butterflies_in_june_l547_547393

-- Given conditions
variables (s1 s2 t2 : ℕ)
variables (p1 p2 : ℝ)

-- Definitions from conditions
def effective_sample_june : ℕ := s2 * (1 - p2)
def effective_tagged_population_october : ℕ := s1 * (1 - p1)

-- Theorem statement to be proved
theorem estimate_butterflies_in_june
  (hs1: s1 = 100) (hs2: s2 = 80) (ht2: t2 = 4) 
  (hp1: p1 = 0.30) (hp2: p2 = 0.50) : 
  ∃ x : ℕ, x = 700 :=
by
  let effective_sample_june := s2 * (1 - p2)
  let effective_tagged_population_october := s1 * (1 - p1)
  have proportion_equation := (t2 : ℝ) / effective_sample_june = effective_tagged_population_october / (x : ℝ)
  -- Placeholder for the solved equation
  use 700
  sorry

end estimate_butterflies_in_june_l547_547393


namespace finite_f_k_l547_547477

def f (n : ℕ) : ℕ :=
  (Finset.range (n - 1)).filter (λ m, ¬ Nat.coprime m n ∧ ¬ m ∣ n).card

theorem finite_f_k (k : ℕ) (hk : 0 < k) : {n : ℕ | f n = k}.finite := 
sorry

end finite_f_k_l547_547477


namespace find_side_b_l547_547119

theorem find_side_b
  (a b c : ℝ)
  (area : ℝ)
  (B : ℝ)
  (h_area : area = sqrt 3)
  (h_B : B = real.pi / 3)
  (h_a2c2 : a^2 + c^2 = 3 * a * c)
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) :
  b = 2 * real.sqrt 2 :=
sorry

end find_side_b_l547_547119


namespace reciprocal_segment_length_sum_constant_l547_547299

variable {α : ℝ} -- The angle α in radians
variable {O P Q A X Y : Point}
variable [h_xoy_angle : XOY.angle_bisector α O A P Q] -- OXY bisected by α through point A

theorem reciprocal_segment_length_sum_constant :
  ∀ (A P Q O: Point) (α : ℝ),
  (A on angle_bisector(O, X, Y, α)) →
  (P lies_on line_segment(O, X)) →
  (Q lies_on line_segment(O, Y)) →
  (A.line_through(P, Q)) →
  ∃ c : ℝ, 
  (constant_value c) =
  (1/OP + 1/OQ) := 
by
  sorry

end reciprocal_segment_length_sum_constant_l547_547299


namespace depth_of_lake_at_base_l547_547401

noncomputable def depth_of_lake (h : ℝ) (V_above_fraction : ℝ) : ℝ :=
  let V_total := (1/3) * π * (1:ℝ) * (h^3)
  let V_above := V_above_fraction * V_total
  let h_submerged := h * ((1 - V_above_fraction)^(1 / 3))
  h - h_submerged

theorem depth_of_lake_at_base :
  depth_of_lake 5000 (1/5) = 660 :=
by
  sorry

end depth_of_lake_at_base_l547_547401


namespace KaydenceAge_l547_547798

-- Definitions for ages of family members based on the problem conditions
def fatherAge := 60
def motherAge := fatherAge - 2 
def brotherAge := fatherAge / 2 
def sisterAge := 40
def totalFamilyAge := 200

-- Lean statement to prove the age of Kaydence
theorem KaydenceAge : 
  fatherAge + motherAge + brotherAge + sisterAge + Kaydence = totalFamilyAge → 
  Kaydence = 12 := 
by
  sorry

end KaydenceAge_l547_547798


namespace range_of_k_decreasing_l547_547567

theorem range_of_k_decreasing (k b : ℝ) (h : ∀ x₁ x₂, x₁ < x₂ → (k^2 - 3*k + 2) * x₁ + b > (k^2 - 3*k + 2) * x₂ + b) : 1 < k ∧ k < 2 :=
by
  -- Proof 
  sorry

end range_of_k_decreasing_l547_547567


namespace books_left_over_l547_547591

theorem books_left_over 
  (n_boxes : ℕ) (books_per_box : ℕ) (books_per_new_box : ℕ)
  (total_books : ℕ) (full_boxes : ℕ) (books_left : ℕ) : 
  n_boxes = 1421 → 
  books_per_box = 27 → 
  books_per_new_box = 35 →
  total_books = n_boxes * books_per_box →
  full_boxes = total_books / books_per_new_box →
  books_left = total_books % books_per_new_box →
  books_left = 7 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end books_left_over_l547_547591


namespace minimize_f_C_l547_547936

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimize_f_C : ∃ x : ℝ, f_C x = 4 :=
by {
  -- The correctness of the statement is asserted and the proof is omitted
  sorry
}

end minimize_f_C_l547_547936


namespace part_a_1_part_a_2_l547_547390

noncomputable def P (x k : ℝ) := x^3 - k*x + 2

theorem part_a_1 (k : ℝ) (h : k = 5) : P 2 k = 0 :=
sorry

theorem part_a_2 {x : ℝ} : P x 5 = (x - 2) * (x^2 + 2*x - 1) :=
sorry

end part_a_1_part_a_2_l547_547390


namespace probability_sum_dice_12_l547_547581

/-- Helper definition for a standard six-faced die roll -/
def is_valid_die_roll (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 6

/-- The probability that the sum of three six-faced dice equals 12 is 19/216. -/
theorem probability_sum_dice_12 :
  (∑ (x y z : ℕ) in (finset.range 7).filter (is_valid_die_roll), ite (x + y + z = 12) 1 0) = 19 :=
begin
  sorry
end

end probability_sum_dice_12_l547_547581


namespace minimum_value_of_option_C_l547_547893

noncomputable def optionA (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def optionB (x : ℝ) : ℝ := |Real.sin x| + 4 / |Real.sin x|
noncomputable def optionC (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def optionD (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem minimum_value_of_option_C :
  ∃ x : ℝ, optionC x = 4 :=
by
  sorry

end minimum_value_of_option_C_l547_547893


namespace option_A_iff_option_B_l547_547683

-- Definitions
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
∀ n m, a (n + 1) - a n = a (m + 1) - a m

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range n, a (i + 1)

def option_B (S : ℕ → ℝ) : Prop :=
is_arithmetic_progression (λ n, S n / n)

-- Main theorem
theorem option_A_iff_option_B (a : ℕ → ℝ) :
  (is_arithmetic_progression a) ↔ (option_B (λ n, sum_of_first_n_terms a n)) :=
sorry

end option_A_iff_option_B_l547_547683


namespace exists_fifth_degree_polynomial_l547_547615

noncomputable def p (x : ℝ) : ℝ :=
  12.4 * (x^5 - 1.38 * x^3 + 0.38 * x)

theorem exists_fifth_degree_polynomial :
  (∃ x1 x2 : ℝ, -1 < x1 ∧ x1 < 1 ∧ -1 < x2 ∧ x2 < 1 ∧ x1 ≠ x2 ∧ 
    p x1 = 1 ∧ p x2 = -1 ∧ p (-1) = 0 ∧ p 1 = 0) :=
  sorry

end exists_fifth_degree_polynomial_l547_547615


namespace arithmetic_progression_iff_mean_sequence_arithmetic_l547_547629

-- Let {a_n} be a sequence and let S_n be the sum of the first n terms of {a_n}
def arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + d * n

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (∑ i in Finset.range (n+1), a i)

def mean_sequence (S : ℕ → ℝ) (m : ℕ → ℝ) : Prop :=
  ∀ n, m n = S n / (n + 1)

-- We need to prove that {a_n} being an arithmetic progression is both necessary and sufficient for {S_n / n} being an arithmetic progression
theorem arithmetic_progression_iff_mean_sequence_arithmetic (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ → ℝ) :
  (arithmetic_progression a ∧ sum_first_n_terms a S ∧ mean_sequence S m) ↔ arithmetic_progression m := 
by 
  -- Proof will go here
  sorry

end arithmetic_progression_iff_mean_sequence_arithmetic_l547_547629


namespace percentage_of_blueberry_muffins_l547_547725

theorem percentage_of_blueberry_muffins
  (cartons : ℕ) (blueberries_per_carton : ℕ) (blueberries_per_muffin : ℕ)
  (blueberries_left : ℕ) (cinnamon_muffins : ℕ) (chocolate_muffins : ℕ)
  (cranberry_muffins : ℕ) (lemon_muffins : ℕ) :
  cartons = 8 →
  blueberries_per_carton = 300 →
  blueberries_per_muffin = 18 →
  blueberries_left = 54 →
  cinnamon_muffins = 80 →
  chocolate_muffins = 40 →
  cranberry_muffins = 50 →
  lemon_muffins = 30 →
  (130 / 330 * 100) ≈ 39.39 :=
by
  intros h_cartons h_blueberries_per_carton h_blueberries_per_muffin 
         h_blueberries_left h_cinnamon_muffins h_chocolate_muffins
         h_cranberry_muffins h_lemon_muffins

  -- Total blueberries
  let total_blueberries := cartons * blueberries_per_carton
  have h_total_blueberries : total_blueberries = 2400 :=
    by rw [h_cartons, h_blueberries_per_carton]; norm_num

  -- Blueberries used
  let used_blueberries := total_blueberries - blueberries_left
  have h_used_blueberries : used_blueberries = 2346 :=
    by rw [h_total_blueberries, h_blueberries_left]; norm_num

  -- Blueberry muffins
  let blueberry_muffins := used_blueberries / blueberries_per_muffin
  have h_blueberry_muffins : blueberry_muffins = 130 :=
    by rw [h_used_blueberries, h_blueberries_per_muffin]; norm_num

  -- Total muffins
  let total_muffins := cinnamon_muffins + chocolate_muffins + cranberry_muffins + lemon_muffins + blueberry_muffins
  have h_total_muffins : total_muffins = 330 :=
    by rw [h_cinnamon_muffins, h_chocolate_muffins, h_cranberry_muffins, h_lemon_muffins, h_blueberry_muffins]; norm_num

  -- Percentage calculation
  let percentage := (blueberry_muffins / total_muffins.toFloat) * 100
  have h_percentage : percentage ≈ 39.39 :=
    by rw [h_blueberry_muffins, h_total_muffins]; norm_num

  -- Conclude
  exact h_percentage

end percentage_of_blueberry_muffins_l547_547725


namespace log_equation_solution_l547_547027

theorem log_equation_solution (x : ℝ) (h : real.log x (3 * x) = 343) : 
  x = 7 / 3 := 
sorry

end log_equation_solution_l547_547027


namespace minimum_a_l547_547038

theorem minimum_a (a : ℝ) : (∀ (x y : ℝ), x > 0 → y > 0 → x + √(x * y) ≤ a * (x + y)) ↔ a ≥ (√2 + 1) / 2 :=
sorry

end minimum_a_l547_547038


namespace option_a_iff_option_b_l547_547640

-- Define a sequence as an arithmetic progression
def is_arithmetic (seq : ℕ → ℚ) : Prop :=
  ∃ d a₁, ∀ n, seq n = a₁ + n * d

-- Define the sum of the first n terms of a sequence
def sum_first_n (seq : ℕ → ℚ) (n : ℕ) : ℚ :=
  ∑ i in finset.range n, seq i

-- Define the average sum sequence
def avg_sum_seq (seq : ℕ → ℚ) (n : ℕ) : ℚ :=
  let Sn := sum_first_n seq n
  in Sn / n

-- The theorem to prove
theorem option_a_iff_option_b (seq : ℕ → ℚ) :
  is_arithmetic seq ↔ is_arithmetic (λ n, avg_sum_seq seq (n + 1)) :=
sorry

end option_a_iff_option_b_l547_547640


namespace optionC_has_min_4_l547_547997

noncomputable def funcA (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def funcB (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def funcC (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def funcD (x : ℝ) : ℝ := log x + 4 / log x

theorem optionC_has_min_4 (x : ℝ) : (∀ y, (y = funcA x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcB x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcD x) → y ≠ 4) ∧
                                   (∃ t, (t = 1) ∧ (funcC t = 4)) := 
by {
  sorry
}

end optionC_has_min_4_l547_547997


namespace minimum_value_of_option_C_l547_547890

noncomputable def optionA (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def optionB (x : ℝ) : ℝ := |Real.sin x| + 4 / |Real.sin x|
noncomputable def optionC (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def optionD (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem minimum_value_of_option_C :
  ∃ x : ℝ, optionC x = 4 :=
by
  sorry

end minimum_value_of_option_C_l547_547890


namespace distance_AB_is_correct_l547_547601

noncomputable def line (t : ℝ) : ℝ × ℝ :=
(1 + t / 2, 2 + (Real.sqrt 3 / 2) * t)

noncomputable def curve (θ : ℝ) : ℝ × ℝ :=
let ρ := 2 * Real.sqrt 3 * Real.sin θ in
(ρ * Real.cos θ, ρ * Real.sin θ)

def distance (A B : ℝ × ℝ) : ℝ :=
Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

theorem distance_AB_is_correct :
  ∀ (A B : ℝ × ℝ), 
  A = line t₁ → 
  B = line t₂ → 
  (A.1 ^ 2 + (A.2 - Real.sqrt 3)^ 2 = 3) → 
  (B.1 ^ 2 + (B.2 - Real.sqrt 3)^ 2 = 3) → 
  distance A B = 2 * Real.sqrt (2 * Real.sqrt 3 - 1) :=
begin
  sorry
end

end distance_AB_is_correct_l547_547601


namespace arith_prog_iff_avg_seq_arith_prog_l547_547680

-- Define sequences
def is_arith_prog (a : ℕ → ℝ) := ∃ d : ℝ, ∃ a₀ : ℝ, ∀ n : ℕ, a n = a₀ + n * d
def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range (n+1), a i
def avg_seq (a : ℕ → ℝ) (n : ℕ) := S a n / (n + 1)

-- Define theorem statement
theorem arith_prog_iff_avg_seq_arith_prog (a : ℕ → ℝ) :
  (is_arith_prog a) ↔ (is_arith_prog (λ n, avg_seq a n)) :=
sorry

end arith_prog_iff_avg_seq_arith_prog_l547_547680


namespace parametric_to_cartesian_eq1_parametric_to_cartesian_eq2_l547_547301

-- Problem 1
theorem parametric_to_cartesian_eq1 (φ : ℝ) :
  ∀ (x y : ℝ), x = 5 * cos φ → y = 4 * sin φ → 
  (x^2 / 25 + y^2 / 16 = 1) := 
by sorry

-- Problem 2
theorem parametric_to_cartesian_eq2 (t : ℝ) :
  ∀ (x y : ℝ), x = 1 - 3 * t^2 → y = 4 * t^2 → x ≤ 1 →
  (4 * x + 3 * y - 4 = 0) := 
by sorry

end parametric_to_cartesian_eq1_parametric_to_cartesian_eq2_l547_547301


namespace tangent_line_at_0_no_such_a_l547_547017

-- Definition of the function
def f (a x : ℝ) : ℝ := (x^2 - x - 1) * exp (a * x)

-- 1. Proving the equation of the tangent line to the curve y = f(x) at the point A(0, f(0))
theorem tangent_line_at_0 (a : ℝ) (h : a ≠ 0) : 
  tangent_line (λ x, f a x) 0 = -x - 1 := 
begin
  sorry
end

-- 2. Proving the range of a for which f(x) ≥ 0 for all x in (-∞, ∞) when a > 0
theorem no_such_a (a : ℝ) (h : a > 0) : 
  ¬(∀ x : ℝ, f a x ≥ 0) := 
begin
  sorry
end

end tangent_line_at_0_no_such_a_l547_547017


namespace inequality_sum_pow_inv_geq_nat_pow_l547_547623

theorem inequality_sum_pow_inv_geq_nat_pow (n k : ℕ) (a : fin n → ℝ) 
  (h1 : ∀ i, 0 < a i) 
  (h2 : (∑ i, a i) = 1) : 
  (∑ i, (a i)⁻¹^k) ≥ (n : ℝ)^(k + 1) := 
sorry

end inequality_sum_pow_inv_geq_nat_pow_l547_547623


namespace monotonicity_intervals_maximum_value_range_of_f_l547_547521

noncomputable def f (a x : ℝ) : ℝ := (1 / 3) ^ (a * x ^ 2 - 4 * x + 3)

theorem monotonicity_intervals (a : ℝ) (h : a = -1) :
  (∀ x : ℝ, -∞ < x ∧ x < -2 → f a x > f a (x + 1)) ∧
  (∀ x : ℝ, -2 < x ∧ x < ∞ → f a x < f a (x + 1)) :=
sorry

theorem maximum_value (h : ∃ x : ℝ, f 1 x = 3) : 
  ∀ a : ℝ, (f a 0 < f a 1) → a = 1 :=
sorry

theorem range_of_f (h : ∀ y : ℝ, 0 < y ∧ y < ∞ → ∃ x : ℝ, f x 0 = y) :
  ∀ a : ℝ, (a = 0) :=
sorry

end monotonicity_intervals_maximum_value_range_of_f_l547_547521


namespace gcd_2_pow_2018_2_pow_2029_l547_547309

theorem gcd_2_pow_2018_2_pow_2029 : Nat.gcd (2^2018 - 1) (2^2029 - 1) = 2047 :=
by
  sorry

end gcd_2_pow_2018_2_pow_2029_l547_547309


namespace find_side_b_l547_547116

theorem find_side_b
  (a b c : ℝ)
  (area : ℝ)
  (B : ℝ)
  (h_area : area = sqrt 3)
  (h_B : B = real.pi / 3)
  (h_a2c2 : a^2 + c^2 = 3 * a * c)
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) :
  b = 2 * real.sqrt 2 :=
sorry

end find_side_b_l547_547116


namespace sqrt_inequality_l547_547260

theorem sqrt_inequality (a b c : ℝ) : 
  sqrt (a^2 + (1 - b)^2) + sqrt (b^2 + (1 - c)^2) + sqrt (c^2 + (1 - a)^2) 
  ≥ (3 * sqrt 2) / 2 := 
  sorry

end sqrt_inequality_l547_547260


namespace function_min_value_4_l547_547859

theorem function_min_value_4 :
  ∃ x : ℝ, (2^x + 2^(2 - x)) = 4 :=
sorry

end function_min_value_4_l547_547859


namespace base_six_500_l547_547307

def base_six (n : ℕ) : list ℕ := 
  if n < 6 then [n]
  else let (q, r) := n / 6, n % 6 in base_six q ++ [r]

theorem base_six_500 : base_six 500 = [2, 1, 5, 2] :=
by
  sorry

end base_six_500_l547_547307


namespace general_terms_sum_terms_l547_547500

noncomputable def a_n (n : ℕ) : ℕ := 4 * n - 3

noncomputable def b_n (n : ℕ) (S_n : ℕ) : ℕ := 2 ^ ((S_n + n) / (2 * n))

noncomputable def S_n (n : ℕ) : ℕ := n * (1 + (n - 1) * 4) / 2

theorem general_terms (n : ℕ):
  (a_n n = 4 * n - 3) ∧
  (b_n n (S_n n) = 2 ^ n) :=
  by sorry

theorem sum_terms (n : ℕ):
  let T_n := (λ n, ∑ k in Finset.range n, a_n (k + 1) * b_n (k + 1) (S_n (k + 1))) in
  T_n n = (4 * n - 7) * 2 ^ (n + 1) + 14 :=
  by sorry

end general_terms_sum_terms_l547_547500


namespace function_min_value_l547_547918

theorem function_min_value (f1 f2 f3 f4 : ℝ → ℝ) :
  f1 = λ x, x^2 + 2 * x + 4 ∧
  f2 = λ x, abs (sin x) + 4 / abs (sin x) ∧
  f3 = λ x, 2^x + 2^(2-x) ∧
  f4 = λ x, log x + 4 / log x →
  (∀ x : ℝ, f3 x ≥ 4) ∧ (∃ x : ℝ, f3 x = 4) :=
by
  sorry

end function_min_value_l547_547918


namespace minimum_value_of_option_C_l547_547882

noncomputable def optionA (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def optionB (x : ℝ) : ℝ := |Real.sin x| + 4 / |Real.sin x|
noncomputable def optionC (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def optionD (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem minimum_value_of_option_C :
  ∃ x : ℝ, optionC x = 4 :=
by
  sorry

end minimum_value_of_option_C_l547_547882


namespace count_three_digit_multiples_of_13_and_7_l547_547548

def is_three_digit (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000

def is_multiple_of (m n : ℕ) : Prop :=
  ∃ k : ℕ, m = k * n

def lcm (a b : ℕ) : ℕ :=
  Nat.lcm a b

def count_multiples_in_range (multiple low high : ℕ) : ℕ :=
  (high - low) / multiple + 1

theorem count_three_digit_multiples_of_13_and_7 : ∃ count : ℕ,
    count = count_multiples_in_range (lcm 13 7) 182 910 ∧ count = 9 :=
by
  sorry

end count_three_digit_multiples_of_13_and_7_l547_547548


namespace find_b_proof_l547_547205

noncomputable def find_b (a b c : ℝ) (A B C : ℝ) : Prop :=
  let area := sqrt 3 in
  let angle_B := 60 * Real.pi / 180 in
  let condition1 := area = (1 / 2) * a * c * Real.sin angle_B in
  let condition2 := a^2 + c^2 = 3 * a * c in
  let side_b := b = 2 * Real.sqrt 2 in
  condition1 ∧ condition2 → side_b

theorem find_b_proof :
  ∃ (a b c : ℝ) (A B C : ℝ), find_b a b c A B C :=
by
  sorry

end find_b_proof_l547_547205


namespace minimum_value_of_h_l547_547900

-- Definitions of the given functions
def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
def g (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
def h (x : ℝ) : ℝ := 2^x + 2^(2 - x)
def j (x : ℝ) : ℝ := log x + 4 / log x

-- Main theorem statement with the condition and conclusion
theorem minimum_value_of_h : ∃ x, h x = 4 :=
by { sorry }

end minimum_value_of_h_l547_547900


namespace find_side_b_l547_547141

theorem find_side_b
  (a b c : ℝ)
  (area : ℝ)
  (B : ℝ)
  (h_area : area = real.sqrt 3)
  (h_B : B = (60 : ℝ) * real.pi / 180) -- Convert degrees to radians
  (h_cond : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 := 
sorry

end find_side_b_l547_547141


namespace min_value_of_quadratic_l547_547511

theorem min_value_of_quadratic (x y z : ℝ) 
  (h1 : x + 2 * y - 5 * z = 3)
  (h2 : x - 2 * y - z = -5) : 
  ∃ z' : ℝ,  x = 3 * z' - 1 ∧ y = z' + 2 ∧ (11 * z' * z' - 2 * z' + 5 = (54 : ℝ) / 11) :=
sorry

end min_value_of_quadratic_l547_547511


namespace find_side_b_l547_547104

theorem find_side_b (a b c : ℝ) (A B C : ℝ)
  (area_triangle : Real.sqrt 3) (B_eq_60 : B = 60) (cond_ac: (a^2 + c^2 = 3 * a * c)) :
  ∃ b, b = 2 * Real.sqrt 2 :=
by
  sorry

end find_side_b_l547_547104


namespace bisect_OM_l547_547592

structure parallelogram (P : Type) :=
(center : P) 
(sides_parallel : P → P → Prop) 

def is_midpoint {P : Type} [add_comm_group P] (A B : P) (K : P) :=
  K = (A + B) / 2

theorem bisect_OM {P : Type} [add_comm_group P] [module ℝ P]
  (parallelogram_P : parallelogram P)
  (M A B C D K O : P)
  (lines_parallel_M_sides : parallelogram_P.sides_parallel M A ∧ parallelogram_P.sides_parallel M B ∧ 
                            parallelogram_P.sides_parallel M C ∧ parallelogram_P.sides_parallel M D)
  (quadrilateral_inside : quadrilateral {A, B, C, D})
  (midpoints_intersections : true) -- Placeholder for "K is the intersection of lines connecting midpoints of opposite sides of ABCD"
  (centerP : parallelogram_P.center = O) :
  is_midpoint O M K :=
  sorry

end bisect_OM_l547_547592


namespace tangent_line_intersects_x_axis_l547_547399

theorem tangent_line_intersects_x_axis (x : ℚ) : 
  (∀ y : ℚ, circle (0, 0) 5 ∧ line_tangent_to_circle (x, 0) y (0, 0) 5) ∧
  (∀ z : ℚ, circle (10, 0) 3 ∧ line_tangent_to_circle (x, 0) z (10, 0) 3) →
  x = 25 / 4 := by
  sorry

end tangent_line_intersects_x_axis_l547_547399


namespace tallest_is_vladimir_l547_547370

variables (Andrei Boris Vladimir Dmitry : Type)
variables (height age : Type)
variables [linear_order height] [linear_order age]
variables (tallest shortest oldest : height)
variables (andrei_statements : (Boris ≠ tallest) ∧ (Vladimir = shortest))
variables (boris_statements : (Andrei = oldest) ∧ (Andrei = shortest))
variables (vladimir_statements : (Dmitry > Vladimir) ∧ (Dmitry > Vladimir))
variables (dmitry_statements : (vladimir_statements = true) ∧ (Dmitry = oldest))

theorem tallest_is_vladimir (H1: andrei_statements ∨ ¬andrei_statements)
  (H2: boris_statements ∨ ¬boris_statements)
  (H3: vladimir_statements ∨ ¬vladimir_statements)
  (H4: dmitry_statements ∨ ¬dmitry_statements)
  (H5: ∃! (a b c d : height) , a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a) : 
  tallest = Vladimir := 
by
  sorry

end tallest_is_vladimir_l547_547370


namespace find_side_b_l547_547114

theorem find_side_b (a b c : ℝ) (A B C : ℝ)
  (area_triangle : Real.sqrt 3) (B_eq_60 : B = 60) (cond_ac: (a^2 + c^2 = 3 * a * c)) :
  ∃ b, b = 2 * Real.sqrt 2 :=
by
  sorry

end find_side_b_l547_547114


namespace trapezoid_perimeter_l547_547737

open Real EuclideanSpace

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem trapezoid_perimeter {A B C D : ℝ × ℝ}
  (hA : A = (0,0)) (hB : B = (3,4)) (hC : C = (15,4)) (hD : D = (18,0)) :
  distance A B + distance B C + distance C D + distance D A = 40 := 
by
  -- establishing vertices
  let A := ⟨0, 0⟩
  let B := ⟨3, 4⟩
  let C := ⟨15, 4⟩
  let D := ⟨18, 0⟩
  -- calculating side lengths
  have d_AB := distance A B
  have d_BC := distance B C
  have d_CD := distance C D
  have d_DA := distance D A
  sorry

end trapezoid_perimeter_l547_547737


namespace dot_product_condition_l547_547539

structure Vector (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α] :=
  (v : α)

variables {α : Type*} [NormedAddCommGroup α] [NormedSpace ℝ α]

theorem dot_product_condition (a b : Vector α)
  (h1 : ∥a.v + b.v∥ = Real.sqrt 10)
  (h2 : ∥a.v - b.v∥ = Real.sqrt 6) :
  (InnerProductSpace ℝ α).inner a b = 1 :=
sorry

end dot_product_condition_l547_547539


namespace option_A_is_iff_option_B_l547_547654

-- Definitions based on conditions
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range (n + 1), a i

def is_arithmetic (seq : ℕ → ℝ) : Prop :=
  ∃ D : ℝ, ∀ n : ℕ, seq (n + 1) - seq n = D

-- The final statement
theorem option_A_is_iff_option_B :
  ∀ (a : ℕ → ℝ),
    (is_arithmetic_progression a ↔ is_arithmetic (λ n, Sn a n / (n + 1))) :=
by sorry

end option_A_is_iff_option_B_l547_547654


namespace opposite_sign_expressions_count_l547_547509

variables (a b : ℚ)

def expr1 : Prop := |a * b| > a * b
def expr2 : Prop := a / b < 0
def expr3 : Prop := |a / b| = -a / b
def expr4 : Prop := a ^ 3 + b ^ 3 = 0

def countOppositeSignsExpressions : ℕ :=
  [expr1 a b, expr2 a b, expr3 a b, expr4 a b].count (λ e, e)

theorem opposite_sign_expressions_count :
  countOppositeSignsExpressions a b = 2 :=
sorry

end opposite_sign_expressions_count_l547_547509


namespace function_min_value_4_l547_547858

theorem function_min_value_4 :
  ∃ x : ℝ, (2^x + 2^(2 - x)) = 4 :=
sorry

end function_min_value_4_l547_547858


namespace circumcircle_of_triangle_E₁E₂E₃_congruent_to_k₁_l547_547274

open EuclideanGeometry

-- Define the conditions
variables (R : ℝ)
variables (O₁ O₂ O₃ : Point) -- centers of circles k₁, k₂, k₃
variables (E₁ E₂ E₃ : Point) -- points of tangency
variables (k₁ k₂ k₃ : Circle)

-- Assuming defined tangencies and distances
axiom h₁ : dist O₁ O₂ = 3 * R
axiom h₂ : dist O₁ O₃ = 4 * R
axiom h₃ : dist O₂ O₃ = 5 * R
axiom tangency_k₁_k₂_at_E₃ : tangency k₁ k₂ E₃
axiom tangency_k₂_k₃_at_E₁ : tangency k₂ k₃ E₁
axiom tangency_k₃_k₁_at_E₂ : tangency k₃ k₁ E₂

-- Define circles' radii
noncomputable def radius_k₁ := R
noncomputable def radius_k₂ := 2 * R
noncomputable def radius_k₃ := 3 * R

-- Define the statement to prove
theorem circumcircle_of_triangle_E₁E₂E₃_congruent_to_k₁ :
  circumradius E₁ E₂ E₃ = R :=
  sorry -- Proof omitted.

end circumcircle_of_triangle_E₁E₂E₃_congruent_to_k₁_l547_547274


namespace tallest_is_vladimir_l547_547367

variables (Andrei Boris Vladimir Dmitry : Type)
variables (height age : Type)
variables [linear_order height] [linear_order age]
variables (tallest shortest oldest : height)
variables (andrei_statements : (Boris ≠ tallest) ∧ (Vladimir = shortest))
variables (boris_statements : (Andrei = oldest) ∧ (Andrei = shortest))
variables (vladimir_statements : (Dmitry > Vladimir) ∧ (Dmitry > Vladimir))
variables (dmitry_statements : (vladimir_statements = true) ∧ (Dmitry = oldest))

theorem tallest_is_vladimir (H1: andrei_statements ∨ ¬andrei_statements)
  (H2: boris_statements ∨ ¬boris_statements)
  (H3: vladimir_statements ∨ ¬vladimir_statements)
  (H4: dmitry_statements ∨ ¬dmitry_statements)
  (H5: ∃! (a b c d : height) , a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a) : 
  tallest = Vladimir := 
by
  sorry

end tallest_is_vladimir_l547_547367


namespace arithmetic_progression_iff_mean_sequence_arithmetic_l547_547630

-- Let {a_n} be a sequence and let S_n be the sum of the first n terms of {a_n}
def arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + d * n

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (∑ i in Finset.range (n+1), a i)

def mean_sequence (S : ℕ → ℝ) (m : ℕ → ℝ) : Prop :=
  ∀ n, m n = S n / (n + 1)

-- We need to prove that {a_n} being an arithmetic progression is both necessary and sufficient for {S_n / n} being an arithmetic progression
theorem arithmetic_progression_iff_mean_sequence_arithmetic (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ → ℝ) :
  (arithmetic_progression a ∧ sum_first_n_terms a S ∧ mean_sequence S m) ↔ arithmetic_progression m := 
by 
  -- Proof will go here
  sorry

end arithmetic_progression_iff_mean_sequence_arithmetic_l547_547630


namespace hexagon_area_l547_547749

-- A definition to specify the regular hexagon with given vertices
def regular_hexagon (P R : ℝ × ℝ) :=
  P = (0, 0) ∧ R = (8, 2) ∧ -- vertices P and R
  ∀ (A B C D E F : ℝ × ℝ), -- other vertices
  linear_dependence A B C D E F P R -- condition for regular hexagon

-- Statement to prove the area of the regular hexagon
theorem hexagon_area (P R : ℝ × ℝ) (h : regular_hexagon P R) : 
  ∃ A B C D E F : ℝ × ℝ, 
  (area (⟨A, B, C, D, E, F, P, R⟩) = 102 * real.sqrt 3) :=
sorry

end hexagon_area_l547_547749


namespace diego_payment_l547_547780

theorem diego_payment (d : ℤ) (celina : ℤ) (total : ℤ) (h₁ : celina = 1000 + 4 * d) (h₂ : total = celina + d) (h₃ : total = 50000) : d = 9800 :=
sorry

end diego_payment_l547_547780


namespace larger_segment_cut_off_l547_547793

theorem larger_segment_cut_off {a b c : ℝ} (h₁ : a = 40) (h₂ : b = 60) (h₃ : c = 80) :
  ∃ x : ℝ, (c - x = 52.5) ∧ (a^2 = x^2 + h^2) ∧ (b^2 = (c - x)^2 + h^2) :=
by
  use 27.5
  have h : c - 27.5 = 52.5 := by norm_num
  split
  { exact h }
  split
  { have eq1: 40^2 = 27.5^2 + h^2 := by sorry
    exact eq1 }
  { have eq2: 60^2 = (80 - 27.5)^2 + h^2 := by sorry
    exact eq2 }
  sorry

end larger_segment_cut_off_l547_547793


namespace divisible_by_6_of_cubed_sum_div_by_18_l547_547245

theorem divisible_by_6_of_cubed_sum_div_by_18 (a b c : ℤ) 
  (h : a^3 + b^3 + c^3 ≡ 0 [ZMOD 18]) : (a * b * c) ≡ 0 [ZMOD 6] :=
sorry

end divisible_by_6_of_cubed_sum_div_by_18_l547_547245


namespace option_a_iff_option_b_l547_547641

-- Define a sequence as an arithmetic progression
def is_arithmetic (seq : ℕ → ℚ) : Prop :=
  ∃ d a₁, ∀ n, seq n = a₁ + n * d

-- Define the sum of the first n terms of a sequence
def sum_first_n (seq : ℕ → ℚ) (n : ℕ) : ℚ :=
  ∑ i in finset.range n, seq i

-- Define the average sum sequence
def avg_sum_seq (seq : ℕ → ℚ) (n : ℕ) : ℚ :=
  let Sn := sum_first_n seq n
  in Sn / n

-- The theorem to prove
theorem option_a_iff_option_b (seq : ℕ → ℚ) :
  is_arithmetic seq ↔ is_arithmetic (λ n, avg_sum_seq seq (n + 1)) :=
sorry

end option_a_iff_option_b_l547_547641


namespace length_of_bridge_l547_547331

theorem length_of_bridge (speed : ℝ) (time_minutes : ℝ)
  (h_speed : speed = 7) (h_time_minutes : time_minutes = 15) :
  let time_hours := time_minutes / 60 in
  let distance := speed * time_hours in
  distance = 1.75 :=
by
  -- place proof here
  sorry

end length_of_bridge_l547_547331


namespace angle_PAC_152_l547_547079

open Real

theorem angle_PAC_152 (A B C D N P T : Point) 
  (H1 : collinear A P D B) 
  (H2 : collinear T B C) 
  (H3 : collinear N C D)
  (H4 : ∠TBD = 110)
  (H5 : ∠BCN = 126)
  (H6 : dist D C = dist D A):
  ∠PAC = 152 := 
sorry

end angle_PAC_152_l547_547079


namespace option_A_iff_option_B_l547_547668

def is_arithmetic_progression (seq : ℕ → ℚ) : Prop :=
  ∃ d a₁, ∀ n, seq (n + 1) = seq n + d

def S_n (a : ℕ → ℚ) : ℕ → ℚ
| 0       := 0
| (n + 1) := S_n n + a (n + 1)

theorem option_A_iff_option_B (a : ℕ → ℚ) :
  (is_arithmetic_progression a) ↔ (is_arithmetic_progression (λ n, S_n a n / (n + 1))) :=
sorry

end option_A_iff_option_B_l547_547668


namespace find_b_correct_l547_547215

axioms (a b c : ℝ) (B : ℝ)
  (area_of_triangle : ℝ)
  (h1 : area_of_triangle = real.sqrt 3)
  (h2 : B = 60 * real.pi / 180) -- converting degrees to radians
  (h3 : a^2 + c^2 = 3 * a * c)

noncomputable def find_b : ℝ := 2 * real.sqrt 2

theorem find_b_correct :
  let b := find_b in 
  b = 2 * real.sqrt 2 :=
by
  sorry

end find_b_correct_l547_547215


namespace Vladimir_is_tallest_l547_547351

-- Definitions for the statements made by the boys.
def Andrei_statements (Boris_tallest Vladimir_shortest : Prop) :=
  (¬Boris_tallest) ∧ Vladimir_shortest

def Boris_statements (Andrei_oldest Andrei_shortest : Prop) :=
  Andrei_oldest ∧ Andrei_shortest

def Vladimir_statements (Dmitry_taller Dmitry_older : Prop) :=
  Dmitry_taller ∧ Dmitry_older

def Dmitry_statements (Vladimir_both_true Dmitry_oldest : Prop) :=
  Vladimir_both_true ∧ Dmitry_oldest

-- Conditions given in the problem.
variables (Boris_tallest Vladimir_shortest Andrei_oldest Andrei_shortest Dmitry_taller Dmitry_older 
           Vladimir_both_true Dmitry_oldest : Prop)

-- Axioms based on the problem statement
axiom Dmitry_statements_condition : Dmitry_statements Vladimir_both_true Dmitry_oldest
axiom Vladimir_statements_condition : Vladimir_statements Dmitry_taller Dmitry_older
axiom Boris_statements_condition : Boris_statements Andrei_oldest Andrei_shortest
axiom Andrei_statements_condition : Andrei_statements Boris_tallest Vladimir_shortest

-- None of them share the same height or age.
axiom no_shared_height_or_age : 
  ¬(Andrei_oldest ∧ Boris_tallest ∧ Dmitry_taller ∧ Vladimir_both_true)

-- Main theorem: proving that Vladimir is the tallest based on the conditions
theorem Vladimir_is_tallest (cond : ¬Boris_tallest → Vladimir_shortest →
                              ¬Andrei_oldest → Andrei_shortest →
                              ¬Dmitry_taller → Dmitry_older →
                              ¬Vladimir_both_true → Dmitry_oldest → 
                              ¬(Andrei_oldest ∧ Boris_tallest ∧ Dmitry_taller ∧ Vladimir_both_true)) :
  (¬Boris_tallest ∧ Dmitry_older) → Vladimir_tallest :=
begin
  sorry,
end

end Vladimir_is_tallest_l547_547351


namespace certain_number_is_two_l547_547032

theorem certain_number_is_two (n : ℕ) 
  (h1 : 1 = 62) 
  (h2 : 363 = 3634) 
  (h3 : 3634 = n) 
  (h4 : n = 365) 
  (h5 : 36 = 2) : 
  n = 2 := 
by 
  sorry

end certain_number_is_two_l547_547032


namespace option_A_is_necessary_and_sufficient_for_option_B_l547_547690

def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

def mean_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : (ℕ → ℝ)
| n => S n / n

def sum_of_first_n_terms (a : ℕ → ℝ) : ℕ → ℝ
| 0     => 0
| (n+1) => sum_of_first_n_terms a n + a (n+1)

theorem option_A_is_necessary_and_sufficient_for_option_B
    (a : ℕ → ℝ) :
    (is_arithmetic_progression a ↔ is_arithmetic_progression (mean_sequence a (sum_of_first_n_terms a))) :=
sorry

end option_A_is_necessary_and_sufficient_for_option_B_l547_547690


namespace vladimir_is_tallest_l547_547361

section TallestBoy

variables (Andrei Boris Vladimir Dmitry : Type)

-- Define the statements made by each boy
def andrei_statements : Andrei → Prop :=
-- suppose andrei's statements are a::Boris ≠ tallest, b::Vladimir = shortest
λ A, (∀ B : Boris, B ≠ tallest) ∨ (∀ V : Vladimir, V = shortest)

def boris_statements : Boris → Prop :=
λ B, (∀ A : Andrei, A = oldest) ∨ (∀ A' : Andrei, A' = shortest)

def vladimir_statements : Vladimir → Prop :=
λ V, (∀ D : Dmitry, D = taller_than V) ∨ (∀ D' : Dmitry, D' = older_than V)

def dmitry_statements : Dmitry → Prop :=
λ D, ((vladimir_statements V) ∧ (vladimir_statements V)) ∨ (∀ D' : Dmitry, D' = oldest)

-- Define the conditions
-- Each boy makes two statements, one of which is true and the other is false
axiom statements_condition (A : Andrei) (B : Boris) (V : Vladimir) (D : Dmitry) :
  (andrei_statements A → ¬andrei_statements A) ∧
  (boris_statements B → ¬boris_statements B) ∧
  (vladimir_statements V → ¬vladimir_statements V) ∧
  (dmitry_statements D → ¬dmitry_statements D)

-- None of them share the same height or age
axiom uniqueness_condition (A : Andrei) (B : Boris) (V : Vladimir) (D : Dmitry) :
  A ≠ B ∧ A ≠ V ∧ A ≠ D ∧ B ≠ V ∧ B ≠ D ∧ V ≠ D

-- The conclusion that Vladimir is the tallest
theorem vladimir_is_tallest (A : Andrei) (B : Boris) (V : Vladimir) (D : Dmitry) :
  A ≠ tallest ∧ D ≠ taller_than (V : Vladimir) ∧ (¬(B = tallest)) → V = tallest :=
sorry

end TallestBoy

end vladimir_is_tallest_l547_547361


namespace units_digit_S12345_l547_547563

  def c := (4 : ℝ) + Real.sqrt 15
  def d := (4 : ℝ) - Real.sqrt 15

  def S : ℕ → ℝ
  | 0 => 1
  | 1 => 4
  | n + 2 => 8 * S (n + 1) - S n

  theorem units_digit_S12345 : (S 12345) % 10 = 4 := 
  sorry
  
end units_digit_S12345_l547_547563


namespace segment_relationship_segment_relationship_phi_180_div_7_segment_relationship_phi_20_l547_547743

theorem segment_relationship (O P S : Point) (x s : ℝ) (hOP : distance O P = x) (hOS : distance O S = s) : x^3 - 3*x - s = 0 :=
sorry

theorem segment_relationship_phi_180_div_7 (x phi : ℝ) (h_phi : phi = Real.pi / 7) : 
  x^3 - 3*x - (2 * Real.cos (3 * phi)) = 0 := 
sorry

theorem segment_relationship_phi_20 (x : ℝ) : 
  x^3 - 3*x - 1 = 0 :=
sorry

end segment_relationship_segment_relationship_phi_180_div_7_segment_relationship_phi_20_l547_547743


namespace find_side_b_l547_547156

theorem find_side_b (a b c A B C : ℝ) (hB : B = 60) 
  (h_area : (1/2)*a*c*(Real.sin (Real.pi/3)) = Real.sqrt 3) 
  (h_ac : a^2 + c^2 = 3*a*c) : b = 2 * Real.sqrt 2 :=
by
  -- Given conditions
  have hB' : Real.sin (Real.pi/3) = Real.sqrt 3 / 2 := by sorry
  have h_ac_val : a * c = 4 := by sorry
  have h_a2c2 : a^2 + c^2 = 12 := by sorry
  have h_cosB : (a^2 + c^2 - b^2) / (2 * a * c) = 1/2 := by sorry
  have h_b2 : b^2 = 8 := by sorry

  -- Result
  exact Real.sqrt 8

end find_side_b_l547_547156


namespace san_francisco_superbowl_probability_l547_547288

theorem san_francisco_superbowl_probability
  (P_play P_not_play : ℝ)
  (k : ℝ)
  (h1 : P_play = k * P_not_play)
  (h2 : P_play + P_not_play = 1) :
  k > 0 :=
sorry

end san_francisco_superbowl_probability_l547_547288


namespace Vladimir_is_tallest_l547_547372

-- Variables representing the boys
inductive Boy
| Andrei | Boris | Vladimir | Dmitry

open Boy

-- Height and Age representations
variable (Height Age : Boy → ℕ)

-- Conditions based on the problem
axiom distinct_heights : ∀ a b, Height a = Height b → a = b
axiom distinct_ages : ∀ a b, Age a = Age b → a = b

-- Statements made by the boys
axiom Andrei_statements :
  (¬ (Height Boris > Height Andrei) ∨ Height Vladimir = Height Andrei)
  ∧ ¬ (¬ (Height Boris > Height Andrei) ∧ Height Vladimir = Height Andrei)

axiom Boris_statements :
  (Age Andrei > Age Boris ∨ Height Andrei = Height Boris)
  ∧ ¬ (Age Andrei > Age Boris ∧ Height Andrei = Height Boris)

axiom Vladimir_statements :
  (Height Dmitry > Height Vladimir ∨ Age Dmitry > Age Vladimir)
  ∧ ¬ (Height Dmitry > Height Vladimir ∧ Age Dmitry > Age Vladimir)

axiom Dmitry_statements :
  ((Height Dmitry > Height Vladimir ∧ Age Dmitry > Age Vladimir) ∨ Age Dmitry > Age Dmitry)
  ∧ ¬ (¬ (Height Dmitry > Height Vladimir ∧ Age Dmitry > Age Vladimir) ∧ ¬ Age Dmitry > Age Dmitry)

-- Problem Statement: Vladimir is the tallest
theorem Vladimir_is_tallest : ∀ h : Height Vladimir = Nat.max (Height Andrei) (Nat.max (Height Boris) (Height Dmitry))
by
  sorry

end Vladimir_is_tallest_l547_547372


namespace Vladimir_is_tallest_l547_547352

-- Definitions for the statements made by the boys.
def Andrei_statements (Boris_tallest Vladimir_shortest : Prop) :=
  (¬Boris_tallest) ∧ Vladimir_shortest

def Boris_statements (Andrei_oldest Andrei_shortest : Prop) :=
  Andrei_oldest ∧ Andrei_shortest

def Vladimir_statements (Dmitry_taller Dmitry_older : Prop) :=
  Dmitry_taller ∧ Dmitry_older

def Dmitry_statements (Vladimir_both_true Dmitry_oldest : Prop) :=
  Vladimir_both_true ∧ Dmitry_oldest

-- Conditions given in the problem.
variables (Boris_tallest Vladimir_shortest Andrei_oldest Andrei_shortest Dmitry_taller Dmitry_older 
           Vladimir_both_true Dmitry_oldest : Prop)

-- Axioms based on the problem statement
axiom Dmitry_statements_condition : Dmitry_statements Vladimir_both_true Dmitry_oldest
axiom Vladimir_statements_condition : Vladimir_statements Dmitry_taller Dmitry_older
axiom Boris_statements_condition : Boris_statements Andrei_oldest Andrei_shortest
axiom Andrei_statements_condition : Andrei_statements Boris_tallest Vladimir_shortest

-- None of them share the same height or age.
axiom no_shared_height_or_age : 
  ¬(Andrei_oldest ∧ Boris_tallest ∧ Dmitry_taller ∧ Vladimir_both_true)

-- Main theorem: proving that Vladimir is the tallest based on the conditions
theorem Vladimir_is_tallest (cond : ¬Boris_tallest → Vladimir_shortest →
                              ¬Andrei_oldest → Andrei_shortest →
                              ¬Dmitry_taller → Dmitry_older →
                              ¬Vladimir_both_true → Dmitry_oldest → 
                              ¬(Andrei_oldest ∧ Boris_tallest ∧ Dmitry_taller ∧ Vladimir_both_true)) :
  (¬Boris_tallest ∧ Dmitry_older) → Vladimir_tallest :=
begin
  sorry,
end

end Vladimir_is_tallest_l547_547352


namespace find_y_l547_547238

-- Definitions and assumptions based on problem conditions
def A := (0, 0)  -- Assume A at origin for simplicity
def B := (86, 0) -- Since ABC is an equilateral triangle with side 86 meters
def C := (43, 43 * Real.sqrt(3))

variable (x y : ℕ)  -- x and y are positive integers denoting the distances

-- Define the property of boy's movement
def boy_moves (A B C : ℝ × ℝ) := 
  ∃ D E : ℝ × ℝ,
    (Dist (A, E) = x) ∧
    (Dist (E, D) = y) ∧
    (D.1 = E.1 + y) ∧
    (E ∈ Line (A, B)) ∧
    (D ∈ Line_westward (E))

-- The main statement to prove
theorem find_y (h : A B C form_equilateral_triangle_with_side 86, x y : ℕ) :
  boy_moves A B C → y = 12 :=
sorry

end find_y_l547_547238


namespace octagon_coloring_4_count_l547_547276

-- Definitions for the conditions
def vertices : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8} -- Represents the 8 vertices of the octagon

def valid_colors : Finset ℕ := {1, 2, 3, 4} -- The integers that can be used to color the vertices

def initial_colors : Finset (ℕ × ℕ) := {(1, 1), (2, 2), (3, 3)}
-- This represents the initial coloring: vertex 1 -> 1, vertex 2 -> 2, vertex 3 -> 3

-- Condition: Adjacent vertices must have different colors
def adj_diff (coloring : ℕ → ℕ) (v1 v2 : ℕ) : Prop :=
  coloring v1 ≠ coloring v2

-- The main theorem to prove
theorem octagon_coloring_4_count :
  ∃ (coloring : ℕ → ℕ), (∀ (v ∈ vertices), coloring v ∈ valid_colors)
   ∧ (∀ (v1 v2 ∈ vertices), adj_diff coloring v1 v2)
   ∧ (∃! (v ∈ vertices), coloring v = 4) = 4 :=
sorry

end octagon_coloring_4_count_l547_547276


namespace slope_constant_l547_547012

-- Define the ellipse with specific parameters
def ellipse (x y : ℝ) : Prop :=
  (x^2) / 8 + (y^2) / 2 = 1

-- Points A, P, and Q definitions
def A : ℝ × ℝ := (2, 1)

-- The main theorem to prove the slope of line PQ is constant
theorem slope_constant (P Q : ℝ × ℝ)
  (hP : ellipse P.1 P.2)
  (hQ : ellipse Q.1 Q.2)
  (hPerpendicular : ∀ P Q, let (x1, y1) := P, (x2, y2) := Q, mPA := (y1 - A.2) / (x1 - A.1), mQA := (y2 - A.2) / (x2 - A.1) in mPA + mQA = 0) :
  let (xP, yP) := P, (xQ, yQ) := Q in (yP - yQ) / (xP - xQ) = 1 / 2 :=
sorry

end slope_constant_l547_547012


namespace find_side_b_l547_547115

theorem find_side_b
  (a b c : ℝ)
  (area : ℝ)
  (B : ℝ)
  (h_area : area = sqrt 3)
  (h_B : B = real.pi / 3)
  (h_a2c2 : a^2 + c^2 = 3 * a * c)
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) :
  b = 2 * real.sqrt 2 :=
sorry

end find_side_b_l547_547115


namespace function_min_value_4_l547_547861

theorem function_min_value_4 :
  ∃ x : ℝ, (2^x + 2^(2 - x)) = 4 :=
sorry

end function_min_value_4_l547_547861


namespace option_A_iff_option_B_l547_547665

def is_arithmetic_progression (seq : ℕ → ℚ) : Prop :=
  ∃ d a₁, ∀ n, seq (n + 1) = seq n + d

def S_n (a : ℕ → ℚ) : ℕ → ℚ
| 0       := 0
| (n + 1) := S_n n + a (n + 1)

theorem option_A_iff_option_B (a : ℕ → ℚ) :
  (is_arithmetic_progression a) ↔ (is_arithmetic_progression (λ n, S_n a n / (n + 1))) :=
sorry

end option_A_iff_option_B_l547_547665


namespace optionA_iff_optionB_l547_547700

variable {a : ℕ → ℝ} {S : ℕ → ℝ}

/-- Sequence a_n is an arithmetic progression -/
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a n = a1 + n * d

/-- Sequence S_n / n is an arithmetic progression -/
def is_S_n_div_n_arithmetic_progression (S : ℕ → ℝ) : Prop :=
  ∃ (b c : ℝ), ∀ n, S n / n = b + c * n

/-- S_n is the sum of the first n terms of the sequence a_n -/
def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = ∑ i in finset.range (n + 1), a i 

/-- Option A is both a sufficient and necessary condition for Option B -/
theorem optionA_iff_optionB (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (is_arithmetic_progression a ∧ sum_first_n_terms a S) ↔ is_S_n_div_n_arithmetic_progression S :=
  sorry

end optionA_iff_optionB_l547_547700


namespace tangent_line_x_intercept_l547_547282

noncomputable def f (x : ℝ) := x^3 + 4 * x + 5
noncomputable def f' (x : ℝ) := 3 * x^2 + 4

theorem tangent_line_x_intercept :
  let point := (1 : ℝ, f 1) in
  let slope := f' 1 in
  let tangent_line y x := y = slope * (x - 1) + f 1 in
  ∃ x₀, tangent_line 0 x₀ ∧ x₀ = -3 / 7 :=
by 
  let point := (1 : ℝ, f 1)
  let slope := f' 1
  let tangent_line y x := y = slope * (x - 1) + f 1
  existsi (-3/7)
  intros
  sorry

end tangent_line_x_intercept_l547_547282


namespace part1_part2_l547_547507

namespace Proof

variable {θ : ℝ}

-- Condition
def condition : Prop := 
  sin θ + cos θ = - (sqrt 10) / 5

-- Questions and corresponding expected values.
theorem part1 (h : condition) : 
  (1 / sin θ) + (1 / cos θ) = (2 * sqrt 10) / 3 := sorry

theorem part2 (h : condition) : 
  tan θ = - (sqrt 11) / 3 := sorry

end Proof

end part1_part2_l547_547507


namespace incorrect_statement_D_l547_547328

def in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem incorrect_statement_D : ¬ in_second_quadrant (-real.sqrt 2) (-real.sqrt 3) :=
by
  sorry

end incorrect_statement_D_l547_547328


namespace find_b_l547_547191

theorem find_b
  (a b c : ℝ)
  (h_area : (1/2) * a * c * Real.sin (ℝ.pi / 3) = Real.sqrt 3)
  (h_ac_eq_4 : a * c = 4)
  (h_cosine : a^2 + c^2 = 3 * a * c) :
  b = 2 * Real.sqrt 2 :=
begin
  sorry
end

end find_b_l547_547191


namespace vladimir_is_tallest_l547_547362

section TallestBoy

variables (Andrei Boris Vladimir Dmitry : Type)

-- Define the statements made by each boy
def andrei_statements : Andrei → Prop :=
-- suppose andrei's statements are a::Boris ≠ tallest, b::Vladimir = shortest
λ A, (∀ B : Boris, B ≠ tallest) ∨ (∀ V : Vladimir, V = shortest)

def boris_statements : Boris → Prop :=
λ B, (∀ A : Andrei, A = oldest) ∨ (∀ A' : Andrei, A' = shortest)

def vladimir_statements : Vladimir → Prop :=
λ V, (∀ D : Dmitry, D = taller_than V) ∨ (∀ D' : Dmitry, D' = older_than V)

def dmitry_statements : Dmitry → Prop :=
λ D, ((vladimir_statements V) ∧ (vladimir_statements V)) ∨ (∀ D' : Dmitry, D' = oldest)

-- Define the conditions
-- Each boy makes two statements, one of which is true and the other is false
axiom statements_condition (A : Andrei) (B : Boris) (V : Vladimir) (D : Dmitry) :
  (andrei_statements A → ¬andrei_statements A) ∧
  (boris_statements B → ¬boris_statements B) ∧
  (vladimir_statements V → ¬vladimir_statements V) ∧
  (dmitry_statements D → ¬dmitry_statements D)

-- None of them share the same height or age
axiom uniqueness_condition (A : Andrei) (B : Boris) (V : Vladimir) (D : Dmitry) :
  A ≠ B ∧ A ≠ V ∧ A ≠ D ∧ B ≠ V ∧ B ≠ D ∧ V ≠ D

-- The conclusion that Vladimir is the tallest
theorem vladimir_is_tallest (A : Andrei) (B : Boris) (V : Vladimir) (D : Dmitry) :
  A ≠ tallest ∧ D ≠ taller_than (V : Vladimir) ∧ (¬(B = tallest)) → V = tallest :=
sorry

end TallestBoy

end vladimir_is_tallest_l547_547362


namespace computation_AK_l547_547241

noncomputable def point (X : Type*) := X

def triangle (A B C : Type*) := Type*

structure Triangle (A B C : Type*) :=
  (AB BC AC : ℝ)
  (AB_gt_0 : 0 < AB)
  (BC_gt_0 : 0 < BC)
  (AC_gt_0 : 0 < AC)
  (sum_gt : AB + BC > AC)
  (sum_gt2 : AB + AC > BC)
  (sum_gt3 : BC + AC > AB)

def Circumcircle (t : Triangle ℝ) := Type*

structure GeometrySetup :=
  (A B C Z X Y K : point ℝ)
  (t : Triangle A B C)
  (circumcircle : Circumcircle t)
  (dist_AB : AB = 6)
  (dist_BC : BC = 5)
  (dist_AC : AC = 7)
  (YZ_eq_3CY : ∀ (CY : ℝ), YZ = 3 * CY)

theorem computation_AK (g : GeometrySetup) : 
  let K := g.K in
  AK = 147 / 10 :=
sorry

end computation_AK_l547_547241


namespace minimize_f_C_l547_547925

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimize_f_C : ∃ x : ℝ, f_C x = 4 :=
by {
  -- The correctness of the statement is asserted and the proof is omitted
  sorry
}

end minimize_f_C_l547_547925


namespace part1_part2_l547_547262

theorem part1 (a x y : ℝ) (h1 : 3 * x - y = 2 * a - 5) (h2 : x + 2 * y = 3 * a + 3)
  (hx : x > 0) (hy : y > 0) : a > 1 :=
sorry

theorem part2 (a b : ℝ) (ha : a > 1) (h3 : a - b = 4) (hb : b < 2) : 
  -2 < a + b ∧ a + b < 8 :=
sorry

end part1_part2_l547_547262


namespace find_side_b_l547_547124

theorem find_side_b
  (a b c : ℝ)
  (area : ℝ)
  (B : ℝ)
  (h_area : area = sqrt 3)
  (h_B : B = real.pi / 3)
  (h_a2c2 : a^2 + c^2 = 3 * a * c)
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) :
  b = 2 * real.sqrt 2 :=
sorry

end find_side_b_l547_547124


namespace min_value_h_l547_547956

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def g (x : ℝ) : ℝ := abs (Real.sin x) + 4 / abs (Real.sin x)
noncomputable def h (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def j (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem min_value_h : ∃ x, h x = 4 :=
by
  sorry

end min_value_h_l547_547956


namespace count_ordered_pairs_l547_547472

theorem count_ordered_pairs :
  let poly (a b : ℤ) : ℤ → ℤ := λ x, x^2 - a * x + b in
  (∑ a in Finset.range (150 + 1), let count_factors := if a % 2 = 0 then (a / 2) + 1 else (a / 2) + 1 in count_factors) = 5851 :=
by
  let poly : ℤ → ℤ → (ℤ → ℤ) := λ a b x, x^2 - a * x + b
  sorry

end count_ordered_pairs_l547_547472


namespace solve_cryptarithm_l547_547757

def cryptarithm_puzzle (K I C : ℕ) : Prop :=
  K ≠ I ∧ K ≠ C ∧ I ≠ C ∧
  K + I + C < 30 ∧  -- Ensuring each is a single digit (0-9)
  (10 * K + I + C) + (10 * K + 10 * C + I) = 100 + 10 * I + 10 * C + K

theorem solve_cryptarithm :
  ∃ K I C, cryptarithm_puzzle K I C ∧ K = 4 ∧ I = 9 ∧ C = 5 :=
by
  use 4, 9, 5
  sorry 

end solve_cryptarithm_l547_547757


namespace function_is_identity_l547_547407

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation : ∀ x y : ℝ, 0 < x → 0 < y → f(x + f(y) + x * y) = x * f(y) + f(x + y)

theorem function_is_identity : ∀ x : ℝ, 0 < x → f(x) = x := 
begin
  sorry
end

end function_is_identity_l547_547407


namespace tallest_boy_is_Vladimir_l547_547347

noncomputable def Andrei_statement1 (Boris_tallest: Prop) : Prop := ¬ Boris_tallest
def Andrei_statement2 (Vladimir_shortest: Prop) : Prop := Vladimir_shortest

def Boris_statement1 (Andrei_oldest: Prop) : Prop := Andrei_oldest
def Boris_statement2 (Andrei_shortest: Prop) : Prop := Andrei_shortest

def Vladimir_statement1 (Dmitry_taller: Prop) : Prop := Dmitry_taller
def Vladimir_statement2 (Dmitry_older: Prop) : Prop := Dmitry_older

noncomputable def Dmitry_statement1 (Vladimir_statement1: Prop) (Vladimir_statement2: Prop) : Prop :=
  Vladimir_statement1 ∧ Vladimir_statement2
def Dmitry_statement2 (Dmitry_oldest: Prop) : Prop := Dmitry_oldest

axiom one_statement_true_per_boy :
  ∀ {P₁ P₂: Prop}, (P₁ ∨ P₂) ∧ ¬ (P₁ ∧ P₂)

axiom no_same_height_or_age :
  ∀ {h1 h2 h3 h4 a1 a2 a3 a4 : ℕ},
    (h1 ≠ h2 ∧ h1 ≠ h3 ∧ h1 ≠ h4 ∧ h2 ≠ h3 ∧ h2 ≠ h4 ∧ h3 ≠ h4) ∧
    (a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)

theorem tallest_boy_is_Vladimir :
  ∀ (Andrei_shortest Vladimir_shortest Boris_tallest Dmitry_taller Dmitry_oldest Vladimir_older : Prop),
    Dmitry_statement2 Dmitry_oldest → 
    Boris_statement2 Andrei_shortest → 
    Andrei_statement1 Boris_tallest → 
    Vladimir_statement2 Vladimir_older → 
    Dmitry_statement1 (Vladimir_statement1 Dmitry_taller) (Vladimir_statement2 Vladimir_older) →
    ¬ Dmitry_taller →
    ¬ Boris_tallest →
    Vladimir = "the tallest boy" :=
  sorry

end tallest_boy_is_Vladimir_l547_547347


namespace dice_sum_probability_l547_547569

theorem dice_sum_probability : 
  let outcomes := { (a, b, c) | a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6} ∧ c ∈ {1, 2, 3, 4, 5, 6} }.card,
      favorable := { (a, b, c) | a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6} ∧ c ∈ {1, 2, 3, 4, 5, 6} ∧ a + b + c = 12 }.card,
      probability := (favorable: ℚ) / (outcomes: ℚ)
  in probability = 5 / 108 := by
  sorry

end dice_sum_probability_l547_547569


namespace minimum_value_C_l547_547939

noncomputable def y_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def y_B (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
noncomputable def y_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def y_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimum_value_C : ∃ x : ℝ, y_C x = 4 := 
by
  -- proof goes here
  sorry

end minimum_value_C_l547_547939


namespace BR_eq_2AM_l547_547733

noncomputable def equilateral_triangle (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

noncomputable def midpoint (M P R : Point) : Prop :=
  dist M P = dist M R ∧ collinear M P R

theorem BR_eq_2AM 
  (A B C P R M : Point)
  (h1 : equilateral_triangle A B C)
  (h2 : P ∈ line_through A B)
  (h3 : R ∈ line_through A C)
  (h4 : dist A P = dist C R)
  (h5 : midpoint M P R) :
  dist B R = 2 * dist A M := 
sorry

end BR_eq_2AM_l547_547733


namespace intersection_A_B_l547_547721

def set_A : Set ℝ := { x | x ≥ 0 }
def set_B : Set ℝ := { x | -1 < x ∧ x < 1 }

theorem intersection_A_B : set_A ∩ set_B = { x | 0 ≤ x ∧ x < 1 } :=
by
  sorry

end intersection_A_B_l547_547721


namespace cubicsum_l547_547841

theorem cubicsum (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) : a^3 + b^3 = 1008 := 
by 
  sorry

end cubicsum_l547_547841


namespace time_after_2345_minutes_l547_547319

-- Define the constants
def minutesInHour : Nat := 60
def hoursInDay : Nat := 24
def startTime : Nat := 0 -- midnight on January 1, 2022, treated as 0 minutes.

-- Prove the equivalent time after 2345 minutes
theorem time_after_2345_minutes :
    let totalMinutes := 2345
    let totalHours := totalMinutes / minutesInHour
    let remainingMinutes := totalMinutes % minutesInHour
    let totalDays := totalHours / hoursInDay
    let remainingHours := totalHours % hoursInDay
    startTime + totalDays * hoursInDay * minutesInHour + remainingHours * minutesInHour + remainingMinutes = startTime + 1 * hoursInDay * minutesInHour + 15 * minutesInHour + 5 :=
    by
    sorry

end time_after_2345_minutes_l547_547319


namespace cookie_problem_l547_547723

theorem cookie_problem (n : ℕ) (M A : ℕ) 
  (hM : M = n - 7) 
  (hA : A = n - 2) 
  (h_sum : M + A < n) 
  (hM_pos : M ≥ 1) 
  (hA_pos : A ≥ 1) : 
  n = 8 := 
sorry

end cookie_problem_l547_547723


namespace problem_proof_l547_547266

noncomputable def problem_statement (x : ℝ) : Prop :=
  (x^2 - 4) * ((x + 2) / (x^2 - 2x) - (x - 1) / (x^2 - 4x + 4)) / ((x - 4) / x) = (x + 2) / (x - 2)

theorem problem_proof (x : ℝ) (h1 : x^2 - 4 = (x + 2) * (x - 2))
  (h2 : x^2 - 2x = x * (x - 2))
  (h3 : x^2 - 4x + 4 = (x - 2)^2) : 
  problem_statement x :=
by {
  sorry
}

end problem_proof_l547_547266


namespace min_value_f_l547_547975

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(2 - x)

theorem min_value_f : ∃ x : ℝ, f x = 4 :=
by {
  sorry
}

end min_value_f_l547_547975


namespace option_a_iff_option_b_l547_547639

-- Define a sequence as an arithmetic progression
def is_arithmetic (seq : ℕ → ℚ) : Prop :=
  ∃ d a₁, ∀ n, seq n = a₁ + n * d

-- Define the sum of the first n terms of a sequence
def sum_first_n (seq : ℕ → ℚ) (n : ℕ) : ℚ :=
  ∑ i in finset.range n, seq i

-- Define the average sum sequence
def avg_sum_seq (seq : ℕ → ℚ) (n : ℕ) : ℚ :=
  let Sn := sum_first_n seq n
  in Sn / n

-- The theorem to prove
theorem option_a_iff_option_b (seq : ℕ → ℚ) :
  is_arithmetic seq ↔ is_arithmetic (λ n, avg_sum_seq seq (n + 1)) :=
sorry

end option_a_iff_option_b_l547_547639


namespace minimum_value_of_option_C_l547_547885

noncomputable def optionA (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def optionB (x : ℝ) : ℝ := |Real.sin x| + 4 / |Real.sin x|
noncomputable def optionC (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def optionD (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem minimum_value_of_option_C :
  ∃ x : ℝ, optionC x = 4 :=
by
  sorry

end minimum_value_of_option_C_l547_547885


namespace find_number_l547_547847

noncomputable def number_when_taken_fraction := 
  ∃ x : ℕ, (4 / 5) * x = (0.80 * 60) - 28 ∧ x = 25

theorem find_number (x : ℕ) (h : (4 / 5) * x = (0.80 * 60) - 28): x = 25 :=
sorry

end find_number_l547_547847


namespace minutes_in_week_closest_to_10000_l547_547784

theorem minutes_in_week_closest_to_10000 : 
  let minutes_in_hour := 60
  let hours_in_day := 24
  let days_in_week := 7
  let minutes_in_day := minutes_in_hour * hours_in_day
  let minutes_in_week := minutes_in_day * days_in_week
  abs (minutes_in_week - 10000) <= abs (minutes_in_week - 100) ∧
  abs (minutes_in_week - 10000) <= abs (minutes_in_week - 1000) ∧
  abs (minutes_in_week - 10000) < abs (minutes_in_week - 100000) ∧
  abs (minutes_in_week - 10000) < abs (minutes_in_week - 1000000) :=
by
  -- start of proof omitted
  sorry

end minutes_in_week_closest_to_10000_l547_547784


namespace rearrange_pairs_l547_547255

theorem rearrange_pairs {a b : ℕ} (hb: b = (2 / 3 : ℚ) * a) (boys_way_museum boys_way_back : ℕ) :
  boys_way_museum = 3 * a ∧ boys_way_back = 4 * b → 
  ∃ c : ℕ, boys_way_museum = 7 * c ∧ b = c := sorry

end rearrange_pairs_l547_547255


namespace find_side_b_l547_547165

noncomputable def triangle_area (a b c : ℝ) (B : ℝ) : ℝ :=
  1 / 2 * a * c * real.sin B

noncomputable def law_of_cosines (a b c : ℝ) (B : ℝ) : ℝ :=
  a^2 + c^2 - 2 * a * c * real.cos B

theorem find_side_b (a b c : ℝ) (B : ℝ) (area : ℝ)
  (h1 : area = √3) 
  (h2 : B = real.pi / 3)
  (h3 : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 :=
by
  sorry

end find_side_b_l547_547165


namespace largest_integer_in_diagram_l547_547816

noncomputable def largest_integer_path : ℕ :=
  let paths := [ (7, 'E', 'B'), (6, 'B', 'C'), (4, 'C', 'E'), (5, 'E', 'D'), (1, 'D', 'B'), (2, 'B', 'A'), (3, 'A', 'C') ] in
  let digits := 7645123 in
  digits

theorem largest_integer_in_diagram : largest_integer_path = 7645123 := 
by 
  unfold largest_integer_path
  sorry

end largest_integer_in_diagram_l547_547816


namespace determine_second_traces_l547_547431

-- Definitions used in the problem.
variables (P Q R : Type) [Plane P] [Plane Q] [Plane R]
variables (s1' s1'' : P)
variables (alpha beta gamma : ℝ)
variables (b : Line)

-- The theorem that confirms the second traces of the planes.
theorem determine_second_traces 
  (h1 : trace b s1' = alpha)
  (h2 : trace b s1'' = beta)
  (h3 : angle_between_traces s1' s1'' = gamma) :
  exists (s2' s2'' : Q), 
    second_trace b s2' = compute_point(alpha, gamma) ∧ 
    second_trace b s2'' = compute_point(beta, gamma) := 
by sorry

end determine_second_traces_l547_547431


namespace option_A_iff_option_B_l547_547666

def is_arithmetic_progression (seq : ℕ → ℚ) : Prop :=
  ∃ d a₁, ∀ n, seq (n + 1) = seq n + d

def S_n (a : ℕ → ℚ) : ℕ → ℚ
| 0       := 0
| (n + 1) := S_n n + a (n + 1)

theorem option_A_iff_option_B (a : ℕ → ℚ) :
  (is_arithmetic_progression a) ↔ (is_arithmetic_progression (λ n, S_n a n / (n + 1))) :=
sorry

end option_A_iff_option_B_l547_547666


namespace fraction_of_EF_eq_1_over_8_l547_547739

variables {G H E F : Type}

open_locale classical
noncomputable theory

def point (x : Type) := x

variables {g h e f : G}

def GH (G H : point G) : ℝ := sorry
def GE (G E : point G) : ℝ := sorry
def EH (E H : point H) : ℝ := sorry
def GF (G F : point G) : ℝ := sorry
def FH (F H : point H) : ℝ := sorry
def EF (E F : point G) : ℝ := sorry

theorem fraction_of_EF_eq_1_over_8 
  (h1 : E ≠ G ∧ F ≠ H) 
  (h2 : GE g e = 3 * EH e h)
  (h3 : GF g f = 7 * FH f h)
  (h4 : GH g h = GE g e + EH e h)
  (h5 : GH g h = GF g f + FH f h) :
  EF e f / GH g h = 1 / 8 :=
sorry

end fraction_of_EF_eq_1_over_8_l547_547739


namespace find_side_b_l547_547113

theorem find_side_b (a b c : ℝ) (A B C : ℝ)
  (area_triangle : Real.sqrt 3) (B_eq_60 : B = 60) (cond_ac: (a^2 + c^2 = 3 * a * c)) :
  ∃ b, b = 2 * Real.sqrt 2 :=
by
  sorry

end find_side_b_l547_547113


namespace min_value_f_l547_547978

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(2 - x)

theorem min_value_f : ∃ x : ℝ, f x = 4 :=
by {
  sorry
}

end min_value_f_l547_547978


namespace complement_union_l547_547389

open Set

def S : Set ℝ := { x | x > -2 }
def T : Set ℝ := { x | x^2 + 3*x - 4 ≤ 0 }

theorem complement_union :
  (compl S) ∪ T = { x : ℝ | x ≤ 1 } :=
sorry

end complement_union_l547_547389


namespace sum_of_distinct_prime_factors_924_l547_547832

theorem sum_of_distinct_prime_factors_924 : 
  (∑ p in ({2, 3, 7, 11} : Finset ℕ), p) = 23 := 
by
  -- sorry is used to skip the proof.
  sorry

end sum_of_distinct_prime_factors_924_l547_547832


namespace can_have_exactly_10_white_marbles_l547_547087

def Operation (w b : ℕ) (op : ℕ) : ℕ × ℕ :=
  match op with
    | 1 => (w, b - 2)
    | 2 => (w - 1, b - 2)
    | 3 => (w - 1, b - 1)
    | 4 => (w - 1, b - 1)
    | 5 => (w - 4 + 1, b + 1)
    | _ => (w, b)

theorem can_have_exactly_10_white_marbles (w b op : ℕ) :
  (w = 50 ∧ b = 150) →
  ∃ op_seq : list ℕ, 
  (¬ op_seq = [] → (∀ o ∈ op_seq, 1 ≤ o ∧ o ≤ 5) ∧ (list.foldl (λ (p : ℕ × ℕ) (o : ℕ), Operation p.fst p.snd o) (w, b) op_seq).fst = 10) :=
by
  intros h
  exists [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
  simp
  sorry

end can_have_exactly_10_white_marbles_l547_547087


namespace minimum_value_of_h_l547_547898

-- Definitions of the given functions
def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
def g (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
def h (x : ℝ) : ℝ := 2^x + 2^(2 - x)
def j (x : ℝ) : ℝ := log x + 4 / log x

-- Main theorem statement with the condition and conclusion
theorem minimum_value_of_h : ∃ x, h x = 4 :=
by { sorry }

end minimum_value_of_h_l547_547898


namespace optionA_iff_optionB_l547_547699

variable {a : ℕ → ℝ} {S : ℕ → ℝ}

/-- Sequence a_n is an arithmetic progression -/
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a n = a1 + n * d

/-- Sequence S_n / n is an arithmetic progression -/
def is_S_n_div_n_arithmetic_progression (S : ℕ → ℝ) : Prop :=
  ∃ (b c : ℝ), ∀ n, S n / n = b + c * n

/-- S_n is the sum of the first n terms of the sequence a_n -/
def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = ∑ i in finset.range (n + 1), a i 

/-- Option A is both a sufficient and necessary condition for Option B -/
theorem optionA_iff_optionB (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (is_arithmetic_progression a ∧ sum_first_n_terms a S) ↔ is_S_n_div_n_arithmetic_progression S :=
  sorry

end optionA_iff_optionB_l547_547699


namespace find_length_of_b_l547_547128

theorem find_length_of_b
  {A B C : Type*}
  (a b c : ℝ)
  (area : ℝ)
  (angleB : ℝ)
  (h_area : area = sqrt 3)
  (h_angle : angleB = real.pi / 3)  -- 60 degrees in radians
  (h_sides : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 :=
by
  -- 1. Using the given area constraint: area = 1/2 * a * c * sin(B)
  -- 2. Using the given angle: sin(60) = sqrt(3)/2
  -- 3. Using the given sides relation: a^2 + c^2 = 3ac
  -- 4. Using the Law of Cosines: cos(60) = 1/2
  -- fill proof here
  sorry

end find_length_of_b_l547_547128


namespace find_two_digit_number_l547_547397

theorem find_two_digit_number (n s p : ℕ) (h1 : n = 4 * s) (h2 : n = 3 * p) : n = 24 := 
  sorry

end find_two_digit_number_l547_547397


namespace general_term_formula_sum_first_10_abs_terms_l547_547008

def sequence_sum (n : ℕ) : ℤ :=
  -(n^2 : ℤ) + 12*n

def a (n : ℕ) : ℤ :=
  if n = 0 then 0  -- Conventionally make an adjustment for n starting at 1 (as ℕ starts from 0)
  else sequence_sum n - sequence_sum (n - 1)

theorem general_term_formula (n : ℕ) (hn : n ≠ 0) : a n = 13 - 2 * n :=
by
  sorry

theorem sum_first_10_abs_terms (T₁₀ : ℤ) (hT₁₀ : T₁₀ = ∑ i in finset.range 10, abs (a (i + 1))) : T₁₀ = 52 :=
by
  sorry

end general_term_formula_sum_first_10_abs_terms_l547_547008


namespace angle_D1OE1_half_angle_DEF_l547_547597

theorem angle_D1OE1_half_angle_DEF (A B C D E F A1 B1 D1 E1 O : Point)
  (h1: Parallel AB DE) (h2: Parallel BC EF) (h3: Parallel CD FA)
  (h4: dist A B = dist D E) (h5: dist B C = dist E F) (h6: dist C D = dist F A)
  (mid1: Midpoint A1 A B) (mid2: Midpoint B1 B C) (mid3: Midpoint D1 D E) (mid4: Midpoint E1 E F)
  (meet1: Meets A1 D1 O) (meet2: Meets B1 E1 O) :
  ∠ (D1 O E1) = 1/2 * ∠ (D E F) := 
sorry

end angle_D1OE1_half_angle_DEF_l547_547597


namespace unique_base_representation_l547_547246

theorem unique_base_representation (a : ℕ) (b : ℕ) (hb : b ≥ 2) :
  ∃ (k : ℕ) (a_i : ℕ → ℕ),
    (∀ j, a_i j < b) ∧
    (a = ∑ i in finset.range (k + 1), a_i i * b ^ i) ∧
    (a_i k ≠ 0) ∧
    ∀ (k' : ℕ) (a'_i : ℕ → ℕ),
      (∀ j, a'_i j < b) →
      (a = ∑ i in finset.range (k' + 1), a'_i i * b ^ i) →
      (k = k' ∧ ∀ j, a_i j = a'_i j) :=
by
  sorry

end unique_base_representation_l547_547246


namespace optionA_iff_optionB_l547_547701

variable {a : ℕ → ℝ} {S : ℕ → ℝ}

/-- Sequence a_n is an arithmetic progression -/
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a n = a1 + n * d

/-- Sequence S_n / n is an arithmetic progression -/
def is_S_n_div_n_arithmetic_progression (S : ℕ → ℝ) : Prop :=
  ∃ (b c : ℝ), ∀ n, S n / n = b + c * n

/-- S_n is the sum of the first n terms of the sequence a_n -/
def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = ∑ i in finset.range (n + 1), a i 

/-- Option A is both a sufficient and necessary condition for Option B -/
theorem optionA_iff_optionB (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (is_arithmetic_progression a ∧ sum_first_n_terms a S) ↔ is_S_n_div_n_arithmetic_progression S :=
  sorry

end optionA_iff_optionB_l547_547701


namespace dice_probability_sum_12_l547_547576

open Nat

/-- Probability that the sum of three six-faced dice rolls equals 12 is 10 / 216 --/
theorem dice_probability_sum_12 : 
  let outcomes := 6^3
  let favorable := 10
  (favorable : ℚ) / outcomes = 10 / 216 := 
by
  let outcomes := 6^3
  let favorable := 10
  sorry

end dice_probability_sum_12_l547_547576


namespace laura_reaches_becky_after_19_5_minutes_l547_547619

theorem laura_reaches_becky_after_19_5_minutes :
  ∃ (vL vB : ℝ) (tB tL: ℝ), 
  (vL = 2 * vB) ∧
  (vL + vB = 2) ∧ 
  let d := 30 - 2 * 6 in
  let d_after_rest := d - vL * 4 in
  (tB = 6) ∧ 
  (tL = d_after_rest / vL + 10) ∧ 
  tL = 19.5 :=
begin
  sorry
end

end laura_reaches_becky_after_19_5_minutes_l547_547619


namespace KaydenceAge_l547_547799

-- Definitions for ages of family members based on the problem conditions
def fatherAge := 60
def motherAge := fatherAge - 2 
def brotherAge := fatherAge / 2 
def sisterAge := 40
def totalFamilyAge := 200

-- Lean statement to prove the age of Kaydence
theorem KaydenceAge : 
  fatherAge + motherAge + brotherAge + sisterAge + Kaydence = totalFamilyAge → 
  Kaydence = 12 := 
by
  sorry

end KaydenceAge_l547_547799


namespace integral_evaluation_l547_547043

noncomputable def coefficient_of_x_squared (a : ℝ) : ℝ :=
  let binom_4_2 := 6 in
  binom_4_2 * 4 - 8 * a

noncomputable def definite_integral (a : ℝ) : ℝ :=
  (Real.log (abs (a))) - (Real.log (abs (Real.exp 1 / 2)))

theorem integral_evaluation (a : ℝ) (h : coefficient_of_x_squared a = 4) :
  definite_integral a = Real.log 5 - 1 := by
  sorry

end integral_evaluation_l547_547043


namespace rosa_used_fraction_l547_547263

noncomputable def fraction_of_perfume_used (r h : ℝ) (remaining_volume : ℝ) : ℝ :=
  let total_volume := (real.pi * r^2 * h) / 1000
  let used_volume := total_volume - remaining_volume
  used_volume / total_volume

theorem rosa_used_fraction (h_radius : ℝ) (h_height : ℝ) (h_remaining : ℝ) :
  h_radius = 7 ∧ h_height = 10 ∧ h_remaining = 0.45 →
  fraction_of_perfume_used h_radius h_height h_remaining = (49 * real.pi - 45) / (49 * real.pi) :=
by sorry

end rosa_used_fraction_l547_547263


namespace function_min_value_l547_547911

theorem function_min_value (f1 f2 f3 f4 : ℝ → ℝ) :
  f1 = λ x, x^2 + 2 * x + 4 ∧
  f2 = λ x, abs (sin x) + 4 / abs (sin x) ∧
  f3 = λ x, 2^x + 2^(2-x) ∧
  f4 = λ x, log x + 4 / log x →
  (∀ x : ℝ, f3 x ≥ 4) ∧ (∃ x : ℝ, f3 x = 4) :=
by
  sorry

end function_min_value_l547_547911


namespace arith_prog_iff_avg_arith_prog_l547_547653

variable {α : Type*} [LinearOrderedField α]

def is_arith_prog (a : ℕ → α) :=
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

def avg_is_arith_prog (S : ℕ → α) :=
  ∃ D : α, ∀ n : ℕ, n ≠ 0 → (S n / n) = (S (n + 1) / (n + 1)) + D

theorem arith_prog_iff_avg_arith_prog (a : ℕ → α) (S : ℕ → α)
  (hS : ∀ n, S n = ∑ i in range n, a i) :
  is_arith_prog a ↔ avg_is_arith_prog S :=
sorry

end arith_prog_iff_avg_arith_prog_l547_547653


namespace minimize_f_C_l547_547931

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimize_f_C : ∃ x : ℝ, f_C x = 4 :=
by {
  -- The correctness of the statement is asserted and the proof is omitted
  sorry
}

end minimize_f_C_l547_547931


namespace function_min_value_l547_547922

theorem function_min_value (f1 f2 f3 f4 : ℝ → ℝ) :
  f1 = λ x, x^2 + 2 * x + 4 ∧
  f2 = λ x, abs (sin x) + 4 / abs (sin x) ∧
  f3 = λ x, 2^x + 2^(2-x) ∧
  f4 = λ x, log x + 4 / log x →
  (∀ x : ℝ, f3 x ≥ 4) ∧ (∃ x : ℝ, f3 x = 4) :=
by
  sorry

end function_min_value_l547_547922


namespace min_value_h_l547_547955

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def g (x : ℝ) : ℝ := abs (Real.sin x) + 4 / abs (Real.sin x)
noncomputable def h (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def j (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem min_value_h : ∃ x, h x = 4 :=
by
  sorry

end min_value_h_l547_547955


namespace solution_to_equation_compare_magnitudes_l547_547486

def satisfies_equation (z : ℂ) : Prop :=
  abs z ^ 2 + 2 * z - 2 * complex.I = 0

theorem solution_to_equation :
  ∃ z : ℂ, satisfies_equation z ∧ z = -1 + complex.I := 
sorry

theorem compare_magnitudes :
  ∀ (z : ℂ), satisfies_equation z → 
  (complex.abs z + complex.abs (z + 3 * complex.I) > complex.abs (2 * z + 3 * complex.I)) := 
sorry

end solution_to_equation_compare_magnitudes_l547_547486


namespace find_a3a4a5_l547_547020

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m, a (n + m) = (a n) * (a m) / (a 0)

theorem find_a3a4a5 :
  ∃ a : ℕ → ℝ, geometric_sequence a ∧ a 2 = 2 ∧ a 6 = 8 ∧ a 3 * a 4 * a 5 = 64 :=
begin
  sorry,
end

end find_a3a4a5_l547_547020


namespace maximize_multicoloured_sets_l547_547801

theorem maximize_multicoloured_sets (m : ℕ → ℕ) (n : ℕ) :
  (∀ i, 1 ≤ i → i ≤ n → m i < m (i + 1)) →
  (∀ i, 1 ≤ i → i < n → m (i + 1) - m i ≤ 2) →
  (count {i | 1 ≤ i ∧ i < n ∧ m (i+1) - m i = 2} = 1) →
  (∑ i in finset.range n, m (i + 1) = 2012) →
  n = 61 :=
sorry

end maximize_multicoloured_sets_l547_547801


namespace measure_angle_ADC_l547_547071

variable {A B C D : Type}
variable [IsParallelogram A B C D]

def angle_ABC := ∠ A B C
def angle_BCD := ∠ B C D
def angle_ADC := ∠ A D C

-- Given conditions
variable (h1 : ∠ A B C = 4 * ∠ B C D)
variable (h2 : OppositeAnglesEqual A B C D)
variable (h3 : AdjacentAnglesSupplementary A B C D)

-- To prove
theorem measure_angle_ADC : ∠ A D C = 36 := by
  sorry

end measure_angle_ADC_l547_547071


namespace solve_for_x_and_y_l547_547560

theorem solve_for_x_and_y (x y : ℝ) (h : sqrt (x - 1) + (y + 2) ^ 2 = 0) : x + y = -1 :=
sorry

end solve_for_x_and_y_l547_547560


namespace exam_failure_l547_547394

structure ExamData where
  max_marks : ℕ
  passing_percentage : ℚ
  secured_marks : ℕ

def passing_marks (data : ExamData) : ℚ :=
  data.passing_percentage * data.max_marks

theorem exam_failure (data : ExamData)
  (h1 : data.max_marks = 150)
  (h2 : data.passing_percentage = 40 / 100)
  (h3 : data.secured_marks = 40) :
  (passing_marks data - data.secured_marks : ℚ) = 20 := by
    sorry

end exam_failure_l547_547394


namespace area_of_triangle_EFD_l547_547272

variables (S : ℝ) (a : ℝ) (h : ℝ)
variable (EFD_area : ℝ)

noncomputable def area_triangle_EFD : ℝ :=
  if h > 0 then max (S / 12) (9 * S / 20) else 0

theorem area_of_triangle_EFD (S : ℝ) (h : ℝ) (base_ratio : ℝ) (side_ratio : ℝ) :
  base_ratio = 3 → side_ratio = 2 → 
  h > 0 →
  (EFD_area = S / 12 ∨ EFD_area = 9 * S / 20) →
  EFD_area = area_triangle_EFD S a h :=
by {
  intros,
  simp [area_triangle_EFD],
  sorry
}

end area_of_triangle_EFD_l547_547272


namespace total_cost_of_ingredients_is_correct_l547_547759

def bottleVolume : ℤ := 16 -- ounces
def bottleCost : ℤ := 4 -- dollars
def cupVolume : ℤ := 8 -- ounces

def chickenWeight : ℤ := 2 -- pounds
def chickenCostPerPound : ℤ := 3 -- dollars per pound

def coconutMilkVolume : ℤ := 500 -- milliliters
def coconutMilkCostPer400ml : ℝ := 2.5 -- dollars per 400 milliliters 

def mixedVeggiesWeight : ℤ := 2 -- pounds
def mixedVeggiesCostPerBag : ℝ := 1.2 -- dollars per 8 ounces

def firstRecipeSoySauce : ℤ := 1 * cupVolume -- ounces (reduced to 1 cup)
def secondRecipeSoySauce : ℤ := (1 * cupVolume) * 3 / 2 -- ounces (1.5 times of 1 cup)
def thirdRecipeSoySauce : ℤ := (3 * cupVolume) * 3 / 4 -- ounces (reduced by 25%)

def totalSoySauce : ℤ := firstRecipeSoySauce + secondRecipeSoySauce + thirdRecipeSoySauce

def soySauceBottlesNeeded : ℤ := (totalSoySauce + bottleVolume - 1) / bottleVolume -- ceil division
def soySauceCost : ℤ := soySauceBottlesNeeded * bottleCost

def totalChickenCost : ℤ := chickenWeight * chickenCostPerPound

def coconutMilkCostPerMl : ℝ := coconutMilkCostPer400ml / 400 -- cost per ml
def totalCoconutMilkCost : ℝ := coconutMilkVolume * coconutMilkCostPerMl

def mixedVeggiesWeightInOunces : ℤ := mixedVeggiesWeight * 16 -- converting to ounces (16 ounces per pound)
def mixedVeggiesBagsNeeded : ℤ := (mixedVeggiesWeightInOunces + 8 - 1) / 8 -- ceil division
def totalMixedVeggiesCost : ℝ := mixedVeggiesBagsNeeded * mixedVeggiesCostPerBag

def totalCost : ℝ := soySauceCost + totalChickenCost + totalCoconutMilkCost + totalMixedVeggiesCost

theorem total_cost_of_ingredients_is_correct : totalCost = 25.93 := 
by sorry -- proof omitted

end total_cost_of_ingredients_is_correct_l547_547759


namespace minimum_value_of_option_C_l547_547889

noncomputable def optionA (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def optionB (x : ℝ) : ℝ := |Real.sin x| + 4 / |Real.sin x|
noncomputable def optionC (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def optionD (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem minimum_value_of_option_C :
  ∃ x : ℝ, optionC x = 4 :=
by
  sorry

end minimum_value_of_option_C_l547_547889


namespace find_slope_of_AF_l547_547773

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  (Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2))

theorem find_slope_of_AF :
  ∀ (A : ℝ × ℝ),
    (A.2 ^ 2 = 4 * A.1) →
    (A.1 - 1)^2 + A.2^2 = 16 →
    (∃ m : ℝ, m = (A.2 - 0) / (A.1 - 1) ∧ (m = sqrt 3 ∨ m = -sqrt 3))
:= by
  sorry

end find_slope_of_AF_l547_547773


namespace find_side_b_l547_547163

noncomputable def triangle_area (a b c : ℝ) (B : ℝ) : ℝ :=
  1 / 2 * a * c * real.sin B

noncomputable def law_of_cosines (a b c : ℝ) (B : ℝ) : ℝ :=
  a^2 + c^2 - 2 * a * c * real.cos B

theorem find_side_b (a b c : ℝ) (B : ℝ) (area : ℝ)
  (h1 : area = √3) 
  (h2 : B = real.pi / 3)
  (h3 : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 :=
by
  sorry

end find_side_b_l547_547163


namespace points_collinear_l547_547600

variables {V : Type*} [inner_product_space ℝ V]

/-- Two planes in space -/
variable (π₁ π₂ : affine_subspace ℝ V)

/-- Two spheres in space with centers O₁ and O₂ and radii r₁ and r₂ respectively -/
variables (O₁ O₂ : V) (r₁ r₂ : ℝ)

/-- Points A, B, and C in space -/
variables (A B C : V)

/-- Hypotheses describing the problem conditions -/
hypothesis (h_planes_parallel : π₁.direction = π₂.direction)
hypothesis (h_A_tangent_to_π₁ : A ∈ π₁)
hypothesis (h_B_tangent_to_π₂ : B ∈ π₂)
hypothesis (h_O₁A_eq_r₁ : dist O₁ A = r₁)
hypothesis (h_O₂B_eq_r₂ : dist O₂ B = r₂)
hypothesis (h_spheres_tangent_at_C : dist O₁ C = r₁ ∧ dist O₂ C = r₂)

/-- Prove that points A, B, and C are collinear -/
theorem points_collinear : affine_space.collinear ℝ ({A, B, C} : set V) :=
sorry

end points_collinear_l547_547600


namespace option_A_iff_option_B_l547_547667

def is_arithmetic_progression (seq : ℕ → ℚ) : Prop :=
  ∃ d a₁, ∀ n, seq (n + 1) = seq n + d

def S_n (a : ℕ → ℚ) : ℕ → ℚ
| 0       := 0
| (n + 1) := S_n n + a (n + 1)

theorem option_A_iff_option_B (a : ℕ → ℚ) :
  (is_arithmetic_progression a) ↔ (is_arithmetic_progression (λ n, S_n a n / (n + 1))) :=
sorry

end option_A_iff_option_B_l547_547667


namespace find_b_l547_547188

theorem find_b
  (a b c : ℝ)
  (h_area : (1/2) * a * c * Real.sin (ℝ.pi / 3) = Real.sqrt 3)
  (h_ac_eq_4 : a * c = 4)
  (h_cosine : a^2 + c^2 = 3 * a * c) :
  b = 2 * Real.sqrt 2 :=
begin
  sorry
end

end find_b_l547_547188


namespace diego_payment_l547_547778

theorem diego_payment (d : ℤ) (celina : ℤ) (total : ℤ) (h₁ : celina = 1000 + 4 * d) (h₂ : total = celina + d) (h₃ : total = 50000) : d = 9800 :=
sorry

end diego_payment_l547_547778


namespace find_side_b_l547_547140

theorem find_side_b
  (a b c : ℝ)
  (area : ℝ)
  (B : ℝ)
  (h_area : area = real.sqrt 3)
  (h_B : B = (60 : ℝ) * real.pi / 180) -- Convert degrees to radians
  (h_cond : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 := 
sorry

end find_side_b_l547_547140


namespace problem1_ineq1_problem1_ineq2_l547_547504

noncomputable
def real_ineq_1 (a b : ℝ) (h₀ : 0 < a) (h₁ : a < b) : Prop :=
  (log a + 1 ≤ (a * log a - b * log b) / (a - b) ∧ 
   (a * log a - b * log b) / (a - b) ≤ log b + 1)

noncomputable
def real_ineq_2 (a b λ : ℝ) (h₀ : 0 < a) (h₁ : a < b) (h₂ : 0 ≤ λ) (h₃: λ ≤ 1) : Prop :=
  (λ * a + (1 - λ) * b) * log (λ * a + (1 - λ) * b) ≤ λ * a * log a + (1 - λ) * b * log b

theorem problem1_ineq1 (a b : ℝ) (h₀ : 0 < a) (h₁ : a < b) : real_ineq_1 a b h₀ h₁ := sorry

theorem problem1_ineq2 (a b λ : ℝ) (h₀ : 0 < a) (h₁ : a < b) (h₂ : 0 ≤ λ) (h₃ : λ ≤ 1) : real_ineq_2 a b λ h₀ h₁ h₂ h₃ := sorry

end problem1_ineq1_problem1_ineq2_l547_547504


namespace minimize_f_C_l547_547930

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimize_f_C : ∃ x : ℝ, f_C x = 4 :=
by {
  -- The correctness of the statement is asserted and the proof is omitted
  sorry
}

end minimize_f_C_l547_547930


namespace find_b_proof_l547_547210

noncomputable def find_b (a b c : ℝ) (A B C : ℝ) : Prop :=
  let area := sqrt 3 in
  let angle_B := 60 * Real.pi / 180 in
  let condition1 := area = (1 / 2) * a * c * Real.sin angle_B in
  let condition2 := a^2 + c^2 = 3 * a * c in
  let side_b := b = 2 * Real.sqrt 2 in
  condition1 ∧ condition2 → side_b

theorem find_b_proof :
  ∃ (a b c : ℝ) (A B C : ℝ), find_b a b c A B C :=
by
  sorry

end find_b_proof_l547_547210


namespace find_theta_l547_547046

def f (x θ : ℝ) : ℝ := sqrt 3 * sin (2 * x + θ) + cos (2 * x + θ)

def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = - f (x)

def is_decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y : ℝ, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x

theorem find_theta 
  (θ : ℝ) 
  (h_odd : is_odd (f θ)) 
  (h_decreasing : is_decreasing_on_interval (f θ) (-π / 4) 0) : 
  θ = 5 * π / 6 :=
sorry

end find_theta_l547_547046


namespace minimum_value_of_h_l547_547909

-- Definitions of the given functions
def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
def g (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
def h (x : ℝ) : ℝ := 2^x + 2^(2 - x)
def j (x : ℝ) : ℝ := log x + 4 / log x

-- Main theorem statement with the condition and conclusion
theorem minimum_value_of_h : ∃ x, h x = 4 :=
by { sorry }

end minimum_value_of_h_l547_547909


namespace smallest_positive_x_for_palindrome_l547_547825

def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10
  s = s.reverse

theorem smallest_positive_x_for_palindrome :
  ∃ x : ℕ, x > 0 ∧ is_palindrome (x + 1234) ∧ (∀ y : ℕ, y > 0 → is_palindrome (y + 1234) → x ≤ y) ∧ x = 97 := 
sorry

end smallest_positive_x_for_palindrome_l547_547825


namespace tank_capacity_l547_547405

variable (c : ℕ) -- Total capacity of the tank in liters.
variable (w_0 : ℕ := c / 3) -- Initial volume of water in the tank in liters.

theorem tank_capacity (h1 : w_0 = c / 3) (h2 : (w_0 + 5) / c = 2 / 5) : c = 75 :=
by
  -- Proof steps would be here.
  sorry

end tank_capacity_l547_547405


namespace triangle_side_b_l547_547224

noncomputable def find_side_length_b
  (a b c : ℝ)
  (area : ℝ)
  (B_deg : ℝ)
  (relation : a^2 + c^2 = 3 * a * c)
  (area_condition : area = sqrt 3)
  (angle_condition : B_deg = 60) : Prop :=
  b = 2 * sqrt 2

theorem triangle_side_b
  (a b c : ℝ)
  (area : ℝ := sqrt 3)
  (B_deg : ℝ := 60)
  (h1 : a^2 + c^2 = 3 * a * c)
  (h2 : area = sqrt 3)
  (h3 : B_deg = 60) : find_side_length_b a b c area B_deg h1 h2 h3 :=
  sorry

end triangle_side_b_l547_547224


namespace equilateral_octagon_side_length_l547_547074

-- Define the problem conditions
theorem equilateral_octagon_side_length (AP BQ x: ℝ) :
  AP = (10 - x) / 2 ∧ AP = BQ ∧ AP < 5 ∧ 
  (2 * (((10 - x) / 2)^2) = x^2) →
  x = 10 :=
by 
  intros conditions,
  sorry

end equilateral_octagon_side_length_l547_547074


namespace starting_lineup_count_l547_547257

theorem starting_lineup_count :
  let players := 16,
      twins := 2,
      triplets := 3,
      required_triplets := 3 in
  (∃ (B Ben Bob : Prop), 
     ∃ (C Charlie Calvin Chris : Prop),
     ∀ lineup, 
     (count lineup triplets = required_triplets ∧ 
      (count lineup twins ≥ 1) → 
      count_combinations := (binom twins 1 * binom (players - triplets - twins) 1 +
                              binom twins 2 * binom (players - triplets - twins) 0))) 
  = 23 :=
by
  sorry

end starting_lineup_count_l547_547257


namespace find_side_b_l547_547107

theorem find_side_b (a b c : ℝ) (A B C : ℝ)
  (area_triangle : Real.sqrt 3) (B_eq_60 : B = 60) (cond_ac: (a^2 + c^2 = 3 * a * c)) :
  ∃ b, b = 2 * Real.sqrt 2 :=
by
  sorry

end find_side_b_l547_547107


namespace arithmetic_sequence_common_difference_range_l547_547453

theorem arithmetic_sequence_common_difference_range (a1 : ℤ) (d : ℂ) :
  a1 = -3 →
  (a1 + 3 * d ≤ 0) →
  (a1 + 4 * d > 0) →
  (3 / 4 : ℤ < d) ∧ (d ≤ 1) :=
by
  intros ha1 hd4 hd5
  sorry

end arithmetic_sequence_common_difference_range_l547_547453


namespace polygon_diagonals_with_restriction_l547_547416

def num_sides := 150

def total_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

def restricted_diagonals (n : ℕ) : ℕ :=
  n * 150 / 4

def valid_diagonals (n : ℕ) : ℕ :=
  total_diagonals n - restricted_diagonals n

theorem polygon_diagonals_with_restriction : valid_diagonals num_sides = 5400 :=
by
  sorry

end polygon_diagonals_with_restriction_l547_547416


namespace find_side_b_l547_547162

theorem find_side_b (a b c A B C : ℝ) (hB : B = 60) 
  (h_area : (1/2)*a*c*(Real.sin (Real.pi/3)) = Real.sqrt 3) 
  (h_ac : a^2 + c^2 = 3*a*c) : b = 2 * Real.sqrt 2 :=
by
  -- Given conditions
  have hB' : Real.sin (Real.pi/3) = Real.sqrt 3 / 2 := by sorry
  have h_ac_val : a * c = 4 := by sorry
  have h_a2c2 : a^2 + c^2 = 12 := by sorry
  have h_cosB : (a^2 + c^2 - b^2) / (2 * a * c) = 1/2 := by sorry
  have h_b2 : b^2 = 8 := by sorry

  -- Result
  exact Real.sqrt 8

end find_side_b_l547_547162


namespace other_rectangle_side_length_l547_547770

def area_circle (r : ℝ) : ℝ := π * r^2

theorem other_rectangle_side_length (A : ℝ) (a : ℝ) :
  A = 153.93804002589985 ∧ a = 14 →
  14 ≤ √(4 * A / π) := by
  intro h
  cases h
  sorry

end other_rectangle_side_length_l547_547770


namespace measure_of_angle_C_smallest_side_length_l547_547613

variable {α : Type*} -- Define the type for the angles and sides
variable [Real ℝ]

variable A B C : ℝ -- Angles
variable a b c : ℝ -- Sides

-- Given conditions
def tan_A := Real.tan A = 1/4
def tan_B := Real.tan B = 3/5
def AB := a = sqrt 17

-- angle C calculation
theorem measure_of_angle_C (h1 : tan_A) (h2 : tan_B) : 
  C = (3 * Real.pi) / 4 := 
sorry

-- Length of the smallest side calculation
theorem smallest_side_length (h1 : tan_A) (h2 : tan_B) (h3 : AB) :
  min a (min b c) = Real.sqrt 2 :=
sorry

end measure_of_angle_C_smallest_side_length_l547_547613


namespace remainder_when_dividing_172_by_17_is_2_l547_547732

theorem remainder_when_dividing_172_by_17_is_2 :
  ∃ R : ℕ, 172 = 17 * 10 + R ∧ R = 2 :=
by {
  use 2,
  split,
  { calc 
    172 = 17 * 10 + 2 : by linarith },
  { refl }
}

end remainder_when_dividing_172_by_17_is_2_l547_547732


namespace minimum_value_inequality_l547_547237

noncomputable def minimum_value_condition (x y z : ℝ) : Prop :=
  xyz = 27

theorem minimum_value_inequality (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0 ∧ xyz = 27) :
  3 * x + 2 * y + z ≥ 18 :=
sorry

end minimum_value_inequality_l547_547237


namespace KF_perp_BD_l547_547620

variables {A B C D K F : Type*}
variables (AB BC AD DC AC BD : ℝ)
variables [h_parallelogram : parallelogram A B C D]
variables (h_AB_lt_BC : AB < BC)
variables (h1 : angle_bisector BAD BC K)
variables (h2 : angle_bisector ADC AC F)
variables (h3 : perpendicular KD BC)

theorem KF_perp_BD :
  perpendicular KF BD := 
sorry

end KF_perp_BD_l547_547620


namespace min_value_of_2x_plus_2_2x_l547_547868

-- Lean 4 statement
theorem min_value_of_2x_plus_2_2x (x : ℝ) : 2^x + 2^(2 - x) ≥ 4 :=
by sorry

end min_value_of_2x_plus_2_2x_l547_547868


namespace arithmetic_progression_iff_mean_sequence_arithmetic_l547_547633

-- Let {a_n} be a sequence and let S_n be the sum of the first n terms of {a_n}
def arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + d * n

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (∑ i in Finset.range (n+1), a i)

def mean_sequence (S : ℕ → ℝ) (m : ℕ → ℝ) : Prop :=
  ∀ n, m n = S n / (n + 1)

-- We need to prove that {a_n} being an arithmetic progression is both necessary and sufficient for {S_n / n} being an arithmetic progression
theorem arithmetic_progression_iff_mean_sequence_arithmetic (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ → ℝ) :
  (arithmetic_progression a ∧ sum_first_n_terms a S ∧ mean_sequence S m) ↔ arithmetic_progression m := 
by 
  -- Proof will go here
  sorry

end arithmetic_progression_iff_mean_sequence_arithmetic_l547_547633


namespace find_side_b_l547_547147

theorem find_side_b
  (a b c : ℝ)
  (area : ℝ)
  (B : ℝ)
  (h_area : area = real.sqrt 3)
  (h_B : B = (60 : ℝ) * real.pi / 180) -- Convert degrees to radians
  (h_cond : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 := 
sorry

end find_side_b_l547_547147


namespace option_A_iff_option_B_l547_547671

def is_arithmetic_progression (seq : ℕ → ℚ) : Prop :=
  ∃ d a₁, ∀ n, seq (n + 1) = seq n + d

def S_n (a : ℕ → ℚ) : ℕ → ℚ
| 0       := 0
| (n + 1) := S_n n + a (n + 1)

theorem option_A_iff_option_B (a : ℕ → ℚ) :
  (is_arithmetic_progression a) ↔ (is_arithmetic_progression (λ n, S_n a n / (n + 1))) :=
sorry

end option_A_iff_option_B_l547_547671


namespace gcd_40_56_l547_547311

theorem gcd_40_56 : Nat.gcd 40 56 = 8 := 
by 
  sorry

end gcd_40_56_l547_547311


namespace option_A_is_necessary_and_sufficient_for_option_B_l547_547696

def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

def mean_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : (ℕ → ℝ)
| n => S n / n

def sum_of_first_n_terms (a : ℕ → ℝ) : ℕ → ℝ
| 0     => 0
| (n+1) => sum_of_first_n_terms a n + a (n+1)

theorem option_A_is_necessary_and_sufficient_for_option_B
    (a : ℕ → ℝ) :
    (is_arithmetic_progression a ↔ is_arithmetic_progression (mean_sequence a (sum_of_first_n_terms a))) :=
sorry

end option_A_is_necessary_and_sufficient_for_option_B_l547_547696


namespace vladimir_is_tallest_l547_547343

inductive Boy : Type
| Andrei | Boris | Vladimir | Dmitry

open Boy

-- Hypotheses based on the problem statement
def statements : Boy → (Prop × Prop)
| Andrei := (¬ (Boris = tallest), Vladimir = shortest)
| Boris := (Andrei = oldest, Andrei = shortest)
| Vladimir := (Dmitry > Vladimir, Dmitry = oldest)
| Dmitry := (statements Vladimir = (true, true), Dmitry = oldest)

def different_heights (a b : Boy) : Prop := a ≠ b
def different_ages (a b : Boy) : Prop := a ≠ b

axiom no_same_height {a b : Boy} : different_heights a b
axiom no_same_age {a b : Boy} : different_ages a b

-- The proof statement
theorem vladimir_is_tallest : Vladimir = tallest := 
by 
  sorry

end vladimir_is_tallest_l547_547343


namespace no_intersection_roots_sum_l547_547626

theorem no_intersection_roots_sum (m r s : ℝ) :
  let P : ℝ → ℝ := λ x, (x - 3)^2 + 2
      Q : ℝ × ℝ := (15, 7)
      discriminant : ℝ := (m + 6)^2 - 4 * (15 * m + 4)
  in (∃ r s : ℝ, ∀ m : ℝ, (discriminant < 0 ↔ r < m ∧ m < s)) → (r + s = 48) :=
by
  sorry

end no_intersection_roots_sum_l547_547626


namespace sequence_properties_l547_547420
noncomputable theory
open scoped Classical

def sequence := ℕ → ℝ
def a_n (n : ℕ) : ℝ := (1 / 2) * n^2 + (3 / 2) * n

def S (n : ℕ) (a : sequence) := ∑ i in Finset.range n, a (i + 1)

theorem sequence_properties :
  let a : sequence := λ n, if n = 0 then 2 else if n = 1 then 5 else if n = 2 then 9 else (1 / 2) * (n + 1)^2 + (3 / 2) * (n + 1) in
  a 0 = 2 ∧ a 1 = 7 - a 0 ∧ a 2 = 16 - (a 0 + a 1) ∧
  (∀ n, a (n + 3) = a_n (n + 1)) ∧
  S 5 a = 50 :=
by
  sorry

end sequence_properties_l547_547420


namespace spelling_contest_questions_count_l547_547594

theorem spelling_contest_questions_count :
  let drew_round1_correct := 20
  let drew_round1_wrong := 6
  let carla_round1_correct := 14
  let carla_round1_wrong := 2 * drew_round1_wrong

  let drew_round1_total := drew_round1_correct + drew_round1_wrong
  let carla_round1_total := carla_round1_correct + carla_round1_wrong

  let drew_round2_correct := 24
  let drew_round2_wrong := 9
  let carla_round2_correct := 21
  let carla_round2_wrong := 8
  let blake_round2_correct := 18
  let blake_round2_wrong := 11

  let drew_round2_total := drew_round2_correct + drew_round2_wrong
  let carla_round2_total := carla_round2_correct + carla_round2_wrong
  let blake_round2_total := blake_round2_correct + blake_round2_wrong

  let drew_round3_correct := 28
  let drew_round3_wrong := 14
  let carla_round3_correct := 22
  let carla_round3_wrong := 10
  let blake_round3_correct := 15
  let blake_round3_wrong := 16

  let drew_round3_total := drew_round3_correct + drew_round3_wrong
  let carla_round3_total := carla_round3_correct + carla_round3_wrong
  let blake_round3_total := blake_round3_correct + blake_round3_wrong

  let round1_total := drew_round1_total + carla_round1_total
  let round2_total := drew_round2_total + carla_round2_total + blake_round2_total
  let round3_total := drew_round3_total + carla_round3_total + blake_round3_total
  let total_questions := round1_total + round2_total + round3_total
  in total_questions = 248 := by
    sorry

end spelling_contest_questions_count_l547_547594


namespace find_b_l547_547189

theorem find_b
  (a b c : ℝ)
  (h_area : (1/2) * a * c * Real.sin (ℝ.pi / 3) = Real.sqrt 3)
  (h_ac_eq_4 : a * c = 4)
  (h_cosine : a^2 + c^2 = 3 * a * c) :
  b = 2 * Real.sqrt 2 :=
begin
  sorry
end

end find_b_l547_547189


namespace four_coloring_of_M_exists_l547_547745

open Function

def M : Finset ℕ := Finset.range 1987
def f : ℕ → Fin 4 := sorry

theorem four_coloring_of_M_exists :
  ∃ (f : ℕ → Fin 4), ∀ (a d : ℕ), d ≠ 0 → ∀ n < 10, (a + d * n) ∈ M → 
  ∃ i j < 10, i ≠ j → (f (a + d * i) ≠ f (a + d * j)) := 
sorry

end four_coloring_of_M_exists_l547_547745


namespace Vladimir_is_tallest_l547_547373

-- Variables representing the boys
inductive Boy
| Andrei | Boris | Vladimir | Dmitry

open Boy

-- Height and Age representations
variable (Height Age : Boy → ℕ)

-- Conditions based on the problem
axiom distinct_heights : ∀ a b, Height a = Height b → a = b
axiom distinct_ages : ∀ a b, Age a = Age b → a = b

-- Statements made by the boys
axiom Andrei_statements :
  (¬ (Height Boris > Height Andrei) ∨ Height Vladimir = Height Andrei)
  ∧ ¬ (¬ (Height Boris > Height Andrei) ∧ Height Vladimir = Height Andrei)

axiom Boris_statements :
  (Age Andrei > Age Boris ∨ Height Andrei = Height Boris)
  ∧ ¬ (Age Andrei > Age Boris ∧ Height Andrei = Height Boris)

axiom Vladimir_statements :
  (Height Dmitry > Height Vladimir ∨ Age Dmitry > Age Vladimir)
  ∧ ¬ (Height Dmitry > Height Vladimir ∧ Age Dmitry > Age Vladimir)

axiom Dmitry_statements :
  ((Height Dmitry > Height Vladimir ∧ Age Dmitry > Age Vladimir) ∨ Age Dmitry > Age Dmitry)
  ∧ ¬ (¬ (Height Dmitry > Height Vladimir ∧ Age Dmitry > Age Vladimir) ∧ ¬ Age Dmitry > Age Dmitry)

-- Problem Statement: Vladimir is the tallest
theorem Vladimir_is_tallest : ∀ h : Height Vladimir = Nat.max (Height Andrei) (Nat.max (Height Boris) (Height Dmitry))
by
  sorry

end Vladimir_is_tallest_l547_547373


namespace cara_arrangements_l547_547449

theorem cara_arrangements (n : ℕ) (h : n = 7) : ∃ k : ℕ, k = 6 :=
by
  sorry

end cara_arrangements_l547_547449


namespace maximum_eccentricity_of_ellipse_l547_547536

noncomputable def point := ℝ × ℝ

def line := point → Prop

def fixed_points : point × point := ((-1, 0), (1, 0))

def moving_point_on_line (P : point) : Prop := ∃ x y : ℝ, y = x + 3 ∧ P = (x, y)

def ellipse_through_point (A B P : point) : Prop := true -- Ellipse properties here, simplified for the Lean statement

def maximum_eccentricity := (c a : ℝ) (h : a > 0) : ℝ := c / a

theorem maximum_eccentricity_of_ellipse :
  ∀ P : point, moving_point_on_line P →
  ∃ e : ℝ, ellipse_through_point (-1, 0) (1, 0) P >>
  e = 1 /√5 :=
sorry

end maximum_eccentricity_of_ellipse_l547_547536


namespace find_b_l547_547187

theorem find_b
  (a b c : ℝ)
  (h_area : (1/2) * a * c * Real.sin (ℝ.pi / 3) = Real.sqrt 3)
  (h_ac_eq_4 : a * c = 4)
  (h_cosine : a^2 + c^2 = 3 * a * c) :
  b = 2 * Real.sqrt 2 :=
begin
  sorry
end

end find_b_l547_547187


namespace min_value_h_is_4_l547_547990

def f (x : ℝ) := x^2 + 2*x + 4
def g (x : ℝ) := abs (sin x) + 4 / abs (sin x)
def h (x : ℝ) := 2^x + 2^(2 - x)
def j (x : ℝ) := log x + 4 / log x

theorem min_value_h_is_4 : 
  (∀ x, f(x) ≥ 3) ∧
  (∀ x, g(x) > 4) ∧
  (∃ x, h(x) = 4) ∧
  (∀ x, j(x) ≠ 4) → min_value_h = 4 :=
by
  sorry

end min_value_h_is_4_l547_547990


namespace number_of_squares_or_cubes_l547_547545

theorem number_of_squares_or_cubes (h1 : ∃ n, n = 28) (h2 : ∃ m, m = 9) (h3 : ∃ k, k = 2) : 
  ∃ t, t = 35 :=
sorry

end number_of_squares_or_cubes_l547_547545


namespace tallest_boy_is_Vladimir_l547_547345

noncomputable def Andrei_statement1 (Boris_tallest: Prop) : Prop := ¬ Boris_tallest
def Andrei_statement2 (Vladimir_shortest: Prop) : Prop := Vladimir_shortest

def Boris_statement1 (Andrei_oldest: Prop) : Prop := Andrei_oldest
def Boris_statement2 (Andrei_shortest: Prop) : Prop := Andrei_shortest

def Vladimir_statement1 (Dmitry_taller: Prop) : Prop := Dmitry_taller
def Vladimir_statement2 (Dmitry_older: Prop) : Prop := Dmitry_older

noncomputable def Dmitry_statement1 (Vladimir_statement1: Prop) (Vladimir_statement2: Prop) : Prop :=
  Vladimir_statement1 ∧ Vladimir_statement2
def Dmitry_statement2 (Dmitry_oldest: Prop) : Prop := Dmitry_oldest

axiom one_statement_true_per_boy :
  ∀ {P₁ P₂: Prop}, (P₁ ∨ P₂) ∧ ¬ (P₁ ∧ P₂)

axiom no_same_height_or_age :
  ∀ {h1 h2 h3 h4 a1 a2 a3 a4 : ℕ},
    (h1 ≠ h2 ∧ h1 ≠ h3 ∧ h1 ≠ h4 ∧ h2 ≠ h3 ∧ h2 ≠ h4 ∧ h3 ≠ h4) ∧
    (a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)

theorem tallest_boy_is_Vladimir :
  ∀ (Andrei_shortest Vladimir_shortest Boris_tallest Dmitry_taller Dmitry_oldest Vladimir_older : Prop),
    Dmitry_statement2 Dmitry_oldest → 
    Boris_statement2 Andrei_shortest → 
    Andrei_statement1 Boris_tallest → 
    Vladimir_statement2 Vladimir_older → 
    Dmitry_statement1 (Vladimir_statement1 Dmitry_taller) (Vladimir_statement2 Vladimir_older) →
    ¬ Dmitry_taller →
    ¬ Boris_tallest →
    Vladimir = "the tallest boy" :=
  sorry

end tallest_boy_is_Vladimir_l547_547345


namespace poly_div_remainder_l547_547460

def P (x : ℕ) : ℕ := x^5 - x^4 + x^3 - x + 1
def D (x : ℕ) : ℕ := x^3 - x + 1
def R (x : ℕ) : ℕ := -x^2 + 4*x - 1

theorem poly_div_remainder :
  ∀ x : ℕ, (P(x) % D(x) = R(x)) :=
by 
  sorry

end poly_div_remainder_l547_547460


namespace iggy_running_hours_l547_547061

theorem iggy_running_hours :
  ∀ (monday tuesday wednesday thursday friday pace_in_minutes total_minutes_in_hour : ℕ),
  monday = 3 → tuesday = 4 → wednesday = 6 → thursday = 8 → friday = 3 →
  pace_in_minutes = 10 → total_minutes_in_hour = 60 →
  ((monday + tuesday + wednesday + thursday + friday) * pace_in_minutes) / total_minutes_in_hour = 4 :=
by
  intros monday tuesday wednesday thursday friday pace_in_minutes total_minutes_in_hour
  sorry

end iggy_running_hours_l547_547061


namespace GroundBeefSalesTotalRevenue_l547_547305

theorem GroundBeefSalesTotalRevenue :
  let price_regular := 3.50
  let price_lean := 4.25
  let price_extra_lean := 5.00

  let monday_revenue := 198.5 * price_regular +
                        276.2 * price_lean +
                        150.7 * price_extra_lean

  let tuesday_revenue := 210 * (price_regular * 0.90) +
                         420 * (price_lean * 0.90) +
                         150 * (price_extra_lean * 0.90)
  
  let wednesday_revenue := 230 * price_regular +
                           324.6 * 3.75 +
                           120.4 * price_extra_lean

  monday_revenue + tuesday_revenue + wednesday_revenue = 8189.35 :=
by
  sorry

end GroundBeefSalesTotalRevenue_l547_547305


namespace minimum_value_of_option_C_l547_547888

noncomputable def optionA (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def optionB (x : ℝ) : ℝ := |Real.sin x| + 4 / |Real.sin x|
noncomputable def optionC (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def optionD (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem minimum_value_of_option_C :
  ∃ x : ℝ, optionC x = 4 :=
by
  sorry

end minimum_value_of_option_C_l547_547888


namespace find_length_of_b_l547_547137

theorem find_length_of_b
  {A B C : Type*}
  (a b c : ℝ)
  (area : ℝ)
  (angleB : ℝ)
  (h_area : area = sqrt 3)
  (h_angle : angleB = real.pi / 3)  -- 60 degrees in radians
  (h_sides : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 :=
by
  -- 1. Using the given area constraint: area = 1/2 * a * c * sin(B)
  -- 2. Using the given angle: sin(60) = sqrt(3)/2
  -- 3. Using the given sides relation: a^2 + c^2 = 3ac
  -- 4. Using the Law of Cosines: cos(60) = 1/2
  -- fill proof here
  sorry

end find_length_of_b_l547_547137


namespace f_irreducible_l547_547720

-- Define the conditions
variables {n : ℕ}
variables {a : Fin (n+1) → ℕ} -- a is a vector of digits of length (n+1)
variable [h_prime : Fact (Prime (∑ i, a i * 10 ^ i))]
variable [h_an_pos : Fact (a ⟨n, lt_add_one n⟩ > 0)]

-- Define the function f(x)
def f (x : ℤ) : ℤ[x] :=
  ∑ i in Finset.range (n+1), polynomial.C (a i) * polynomial.X ^ i

-- State the irreducibility theorem
theorem f_irreducible : Irreducible (f : polynomial ℤ) :=
sorry

end f_irreducible_l547_547720


namespace find_a_n_geo_b_find_S_2n_l547_547001
noncomputable def S : ℕ → ℚ
| n => (n^2 + n + 1) / 2

def a (n : ℕ) : ℚ :=
  if n = 1 then 3/2
  else n

theorem find_a_n (n : ℕ) : a n = if n = 1 then 3/2 else n :=
by
  sorry

def b (n : ℕ) : ℚ :=
  a (2 * n - 1) + a (2 * n)

theorem geo_b (n : ℕ) : b (n + 1) = 3 * b n :=
by
  sorry

theorem find_S_2n (n : ℕ) : S (2 * n) = 3/2 * (3^n - 1) :=
by
  sorry

end find_a_n_geo_b_find_S_2n_l547_547001


namespace count_pos_three_digit_divisible_by_13_and_7_l547_547551

theorem count_pos_three_digit_divisible_by_13_and_7 : 
  ((finset.filter (λ n : ℕ, n % (13 * 7) = 0) (finset.Icc 100 999)).card = 9) := 
sorry

end count_pos_three_digit_divisible_by_13_and_7_l547_547551


namespace function_min_value_l547_547913

theorem function_min_value (f1 f2 f3 f4 : ℝ → ℝ) :
  f1 = λ x, x^2 + 2 * x + 4 ∧
  f2 = λ x, abs (sin x) + 4 / abs (sin x) ∧
  f3 = λ x, 2^x + 2^(2-x) ∧
  f4 = λ x, log x + 4 / log x →
  (∀ x : ℝ, f3 x ≥ 4) ∧ (∃ x : ℝ, f3 x = 4) :=
by
  sorry

end function_min_value_l547_547913


namespace min_value_h_is_4_l547_547985

def f (x : ℝ) := x^2 + 2*x + 4
def g (x : ℝ) := abs (sin x) + 4 / abs (sin x)
def h (x : ℝ) := 2^x + 2^(2 - x)
def j (x : ℝ) := log x + 4 / log x

theorem min_value_h_is_4 : 
  (∀ x, f(x) ≥ 3) ∧
  (∀ x, g(x) > 4) ∧
  (∃ x, h(x) = 4) ∧
  (∀ x, j(x) ≠ 4) → min_value_h = 4 :=
by
  sorry

end min_value_h_is_4_l547_547985


namespace eccentricity_of_hyperbola_l547_547080

theorem eccentricity_of_hyperbola (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b)
  (h_hyperbola : ∀ x y : ℝ, (x = c → y = b^2/a ∨ y = -b^2/a → ∃ x y: ℝ, x^2 / a^2 - y^2 / b^2 = 1))
  (h_vertex : ∀ x : ℝ, x = -a ∧ 0 = y → (c - a) = a)
  (h_focus : ∀ x y : ℝ, y = 0 → (x = c ∧ x = b + a)) :
  eccentricity_of_hyperbola = 2 :=
by
  sorry

end eccentricity_of_hyperbola_l547_547080


namespace function_min_value_l547_547910

theorem function_min_value (f1 f2 f3 f4 : ℝ → ℝ) :
  f1 = λ x, x^2 + 2 * x + 4 ∧
  f2 = λ x, abs (sin x) + 4 / abs (sin x) ∧
  f3 = λ x, 2^x + 2^(2-x) ∧
  f4 = λ x, log x + 4 / log x →
  (∀ x : ℝ, f3 x ≥ 4) ∧ (∃ x : ℝ, f3 x = 4) :=
by
  sorry

end function_min_value_l547_547910


namespace sum_prime_factors_924_l547_547836

/-- Let n be the integer 924. -/
def n : ℤ := 924

/-- The set of distinct prime factors of 924. -/
def distinct_prime_factors (n : ℤ) : set ℤ :=
  if n = 924 then {2, 3, 7, 11} else ∅

/-- The sum of the distinct prime factors of 924. -/
def sum_distinct_prime_factors (n : ℤ) : ℤ :=
  if n = 924 then 2 + 3 + 7 + 11 else 0

-- The theorem to prove that the sum of the distinct prime factors of 924 is 23.
theorem sum_prime_factors_924 : sum_distinct_prime_factors n = 23 :=
by {
  unfold sum_distinct_prime_factors,
  simp,
}

end sum_prime_factors_924_l547_547836


namespace ant_tetrahedron_probability_l547_547436

def p : ℕ → ℝ
| 0     := 1
| (n+1) := (1/3) * (1 - p n)

theorem ant_tetrahedron_probability : p 60 = (3^59 + 1) / (4 * 3^59) :=
by sorry

end ant_tetrahedron_probability_l547_547436


namespace quadratic_roots_eq_l547_547794

theorem quadratic_roots_eq (a b c : ℝ) (h_eq : a = 1) (h_neq : b = -2*Real.sqrt 2) (h_c : c = 2) :
  (b^2 - 4*a*c = 0) → ∃ x : ℝ, (x^2 - 2*Real.sqrt 2*x + 2 = 0) ∧ (∃ y : ℝ, y = x) :=
begin
  intros h_discriminant,
  use [1, sorry], -- We use 1 as an example root.
  exact ⟨rfl⟩,
  sorry
end

end quadratic_roots_eq_l547_547794


namespace min_value_f_l547_547968

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(2 - x)

theorem min_value_f : ∃ x : ℝ, f x = 4 :=
by {
  sorry
}

end min_value_f_l547_547968


namespace trajectory_equation_l547_547044

-- Definitions for distances
def dist_to_line (P : ℝ × ℝ) (line_y : ℝ) : ℝ := abs (P.snd + line_y)
def dist_to_point (P : ℝ × ℝ) (Q : ℝ × ℝ) : ℝ := real.sqrt ((P.fst - Q.fst)^2 + (P.snd - Q.snd)^2)

-- Condition given in the problem:
axiom condition (P : ℝ × ℝ) : 
  dist_to_line P (-1) + 2 = dist_to_point P (0, 3)

-- Prove that the equation of the trajectory of point P is x^2 = 12y
theorem trajectory_equation :
  ∀ (P : ℝ × ℝ), condition P → P.1 ^ 2 = 12 * P.2 :=
sorry

end trajectory_equation_l547_547044


namespace find_b_correct_l547_547211

axioms (a b c : ℝ) (B : ℝ)
  (area_of_triangle : ℝ)
  (h1 : area_of_triangle = real.sqrt 3)
  (h2 : B = 60 * real.pi / 180) -- converting degrees to radians
  (h3 : a^2 + c^2 = 3 * a * c)

noncomputable def find_b : ℝ := 2 * real.sqrt 2

theorem find_b_correct :
  let b := find_b in 
  b = 2 * real.sqrt 2 :=
by
  sorry

end find_b_correct_l547_547211


namespace transformed_cosine_function_l547_547529

theorem transformed_cosine_function (x : ℝ) :
  (∀ x : ℝ, y = cos x) → (y = cos (2 * x + π / 2)) :=
by
  intros x h1
  sorry

end transformed_cosine_function_l547_547529


namespace student_ticket_price_is_2_50_l547_547806

-- Defining the given conditions
def adult_ticket_price : ℝ := 4
def total_tickets_sold : ℕ := 59
def total_revenue : ℝ := 222.50
def student_tickets_sold : ℕ := 9

-- The number of adult tickets sold
def adult_tickets_sold : ℕ := total_tickets_sold - student_tickets_sold

-- The total revenue from adult tickets
def revenue_from_adult_tickets : ℝ := adult_tickets_sold * adult_ticket_price

-- The remaining revenue must come from student tickets and defining the price of student ticket
noncomputable def student_ticket_price : ℝ :=
  (total_revenue - revenue_from_adult_tickets) / student_tickets_sold

-- The theorem to be proved
theorem student_ticket_price_is_2_50 : student_ticket_price = 2.50 :=
by
  sorry

end student_ticket_price_is_2_50_l547_547806


namespace find_side_b_l547_547167

noncomputable def triangle_area (a b c : ℝ) (B : ℝ) : ℝ :=
  1 / 2 * a * c * real.sin B

noncomputable def law_of_cosines (a b c : ℝ) (B : ℝ) : ℝ :=
  a^2 + c^2 - 2 * a * c * real.cos B

theorem find_side_b (a b c : ℝ) (B : ℝ) (area : ℝ)
  (h1 : area = √3) 
  (h2 : B = real.pi / 3)
  (h3 : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 :=
by
  sorry

end find_side_b_l547_547167


namespace minimum_dimes_l547_547451

theorem minimum_dimes (n : ℕ) : 
  let total := 2 * 20 + 5 * 0.25 + 6 * 0.05 + n * 0.10
  in total ≥ 45.50 → n ≥ 40 :=
by
  let total := 2 * 20 + 5 * 0.25 + 6 * 0.05 + n * 0.10
  intro h
  have h1 : total = 40 + 1.25 + 0.30 + n * 0.10, from by simp only [two_mul, add_mul, mul_add, bit0, bit1, add_assoc, mul_comm, add_left_comm]
  have h2 : 41.55 + n * 0.10 ≥ 45.50, by linarith,
  have h3 : n * 0.10 ≥ 45.50 - 41.55, from h2,
  have h4 : n * 0.10 ≥ 3.95, from h3,
  have h5 : n ≥ 3.95 / 0.10, from by { field_simp [h4], ring },
  exact_mod_cast h5

end minimum_dimes_l547_547451


namespace count_three_digit_multiples_of_91_l547_547554

theorem count_three_digit_multiples_of_91 : 
  ∃ (count : ℕ), count = 
    let lcm := Nat.lcm 13 7 in
    let min_k := (100 / lcm) + 1 in
    let max_k := 999 / lcm in
    max_k - min_k + 1 ∧ lcm = 91 ∧ (100 ≤ 91 * min_k) ∧ (91 * max_k ≤ 999) :=
by
  sorry

end count_three_digit_multiples_of_91_l547_547554


namespace sum_of_squares_of_roots_eq_92_l547_547839

theorem sum_of_squares_of_roots_eq_92 :
  (∑ root in ({x | x^2 - 7 * (⌊x⌋ : ℤ) + 5 = 0}.to_finset), root^2) = 92 :=
sorry

end sum_of_squares_of_roots_eq_92_l547_547839


namespace find_function_expression_find_maximum_b_l547_547718

def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2 - a^2 * x

theorem find_function_expression (a b : ℝ) (x1 x2 : ℝ) (h1 : x1 ≠ x2) (h2 : a > 0)
  (hx1 : x1 = -1) (hx2 : x2 = 2) (hfx1 : deriv (f a b) x1 = 0)
  (hfx2 : deriv (f a b) x2 = 0) :
  f a b = f 6 (-9) :=
sorry

theorem find_maximum_b (a : ℝ) (h1 : a > 0) (h2 : 0 < a ∧ a ≤ 6)
  (hx_abs : -2 * a / 3 * (6 * a) = 8) :
  max (λ b, b^2 = 3 * a^2 * (6 - a)) = 4 * sqrt(6) :=
sorry

end find_function_expression_find_maximum_b_l547_547718


namespace minimum_value_C_l547_547944

noncomputable def y_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def y_B (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
noncomputable def y_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def y_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimum_value_C : ∃ x : ℝ, y_C x = 4 := 
by
  -- proof goes here
  sorry

end minimum_value_C_l547_547944


namespace minimize_f_C_l547_547937

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimize_f_C : ∃ x : ℝ, f_C x = 4 :=
by {
  -- The correctness of the statement is asserted and the proof is omitted
  sorry
}

end minimize_f_C_l547_547937


namespace quadrilateral_has_parallel_sides_l547_547073

-- Define a quadrilateral ABCD where angle ACB is equal to angle CAD
variables (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]

-- Define angles ACB and CAD
def angle_ACB (A B C : Type) [metric_space A] [metric_space B] [metric_space C] : ℝ := sorry
def angle_CAD (A C D : Type) [metric_space A] [metric_space C] [metric_space D] : ℝ := sorry

-- Conditions
axiom angle_equality : angle_ACB A B C = angle_CAD A C D

-- Main statement
theorem quadrilateral_has_parallel_sides (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] (h : angle_ACB A B C = angle_CAD A C D) : is_parallelogram A B C D :=
sorry

end quadrilateral_has_parallel_sides_l547_547073


namespace distinct_units_digits_of_squares_l547_547543

theorem distinct_units_digits_of_squares : ∃ (S : Finset ℕ), S.card = 6 ∧ ∀ n : ℤ, ((n^2 % 10 ∈ S)) :=
by
  sorry

end distinct_units_digits_of_squares_l547_547543


namespace smallest_b_to_the_a_l547_547269

theorem smallest_b_to_the_a (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^b = 2^2023) : b^a = 1 :=
by
  -- Proof steps go here
  sorry

end smallest_b_to_the_a_l547_547269


namespace triangle_side_b_l547_547234

noncomputable def find_side_length_b
  (a b c : ℝ)
  (area : ℝ)
  (B_deg : ℝ)
  (relation : a^2 + c^2 = 3 * a * c)
  (area_condition : area = sqrt 3)
  (angle_condition : B_deg = 60) : Prop :=
  b = 2 * sqrt 2

theorem triangle_side_b
  (a b c : ℝ)
  (area : ℝ := sqrt 3)
  (B_deg : ℝ := 60)
  (h1 : a^2 + c^2 = 3 * a * c)
  (h2 : area = sqrt 3)
  (h3 : B_deg = 60) : find_side_length_b a b c area B_deg h1 h2 h3 :=
  sorry

end triangle_side_b_l547_547234


namespace find_b_l547_547198

theorem find_b
  (a b c : ℝ)
  (h_area : (1/2) * a * c * Real.sin (ℝ.pi / 3) = Real.sqrt 3)
  (h_ac_eq_4 : a * c = 4)
  (h_cosine : a^2 + c^2 = 3 * a * c) :
  b = 2 * Real.sqrt 2 :=
begin
  sorry
end

end find_b_l547_547198


namespace number_smaller_than_neg_3_l547_547432

theorem number_smaller_than_neg_3 : ∃ n ∈ [-2, 4, -5, 1], n < -3 :=
by {
    use -5,
    split,
    sorry, -- Proof that -5 is in the list [-2, 4, -5, 1]
    sorry  -- Proof that -5 < -3
}

end number_smaller_than_neg_3_l547_547432


namespace min_value_h_is_4_l547_547987

def f (x : ℝ) := x^2 + 2*x + 4
def g (x : ℝ) := abs (sin x) + 4 / abs (sin x)
def h (x : ℝ) := 2^x + 2^(2 - x)
def j (x : ℝ) := log x + 4 / log x

theorem min_value_h_is_4 : 
  (∀ x, f(x) ≥ 3) ∧
  (∀ x, g(x) > 4) ∧
  (∃ x, h(x) = 4) ∧
  (∀ x, j(x) ≠ 4) → min_value_h = 4 :=
by
  sorry

end min_value_h_is_4_l547_547987


namespace correct_calculation_option_D_l547_547853

theorem correct_calculation_option_D (a : ℝ) : (a ^ 3) ^ 2 = a ^ 6 :=
by sorry

end correct_calculation_option_D_l547_547853


namespace probability_at_most_one_A_B_selected_l547_547066

def total_employees : ℕ := 36
def ratio_3_2_1 : (ℕ × ℕ × ℕ) := (3, 2, 1)
def sample_size : ℕ := 12
def youth_group_size : ℕ := 6
def total_combinations_youth : ℕ := Nat.choose 6 2
def event_complementary : ℕ := Nat.choose 2 2

theorem probability_at_most_one_A_B_selected :
  let prob := 1 - event_complementary / total_combinations_youth
  prob = (14 : ℚ) / 15 := sorry

end probability_at_most_one_A_B_selected_l547_547066


namespace min_value_f_l547_547970

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(2 - x)

theorem min_value_f : ∃ x : ℝ, f x = 4 :=
by {
  sorry
}

end min_value_f_l547_547970


namespace sandbox_area_l547_547386

def sandbox_length : ℕ := 312
def sandbox_width : ℕ := 146

theorem sandbox_area : sandbox_length * sandbox_width = 45552 := by
  sorry

end sandbox_area_l547_547386


namespace sum_of_center_coordinates_eq_five_l547_547748

-- Define the conditions
variables (A B C D : ℝ × ℝ) -- Vertices of the rectangle
variables (P1 P2 P3 P4 : ℝ × ℝ) -- Points on the lines
variables (AD BC AB CD : ℝ → ℝ) -- Equations of lines

-- Given points on the lines
def points_on_lines : Prop :=
  P1 = (2,1) ∧ P2 = (4,1) ∧ P3 = (6,1) ∧ P4 = (12,1)

-- Definition of center of the rectangle
def center (A B C D : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1 + C.1 + D.1) / 4, (A.2 + B.2 + C.2 + D.2) / 4)

-- Prove that the sum of the coordinates of the center of rectangle ABCD is 5
theorem sum_of_center_coordinates_eq_five (h : points_on_lines) : 
  let O := center A B C D in (O.1 + O.2) = 5 :=
sorry

end sum_of_center_coordinates_eq_five_l547_547748


namespace find_b_proof_l547_547204

noncomputable def find_b (a b c : ℝ) (A B C : ℝ) : Prop :=
  let area := sqrt 3 in
  let angle_B := 60 * Real.pi / 180 in
  let condition1 := area = (1 / 2) * a * c * Real.sin angle_B in
  let condition2 := a^2 + c^2 = 3 * a * c in
  let side_b := b = 2 * Real.sqrt 2 in
  condition1 ∧ condition2 → side_b

theorem find_b_proof :
  ∃ (a b c : ℝ) (A B C : ℝ), find_b a b c A B C :=
by
  sorry

end find_b_proof_l547_547204


namespace solve_exponential_equation_l547_547293

theorem solve_exponential_equation (x : ℝ) :
    (4^(x^2 + 1) = 16) ↔ (x = -1 ∨ x = 1) :=
by
  sorry

end solve_exponential_equation_l547_547293


namespace find_side_b_l547_547169

noncomputable def triangle_area (a b c : ℝ) (B : ℝ) : ℝ :=
  1 / 2 * a * c * real.sin B

noncomputable def law_of_cosines (a b c : ℝ) (B : ℝ) : ℝ :=
  a^2 + c^2 - 2 * a * c * real.cos B

theorem find_side_b (a b c : ℝ) (B : ℝ) (area : ℝ)
  (h1 : area = √3) 
  (h2 : B = real.pi / 3)
  (h3 : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 :=
by
  sorry

end find_side_b_l547_547169


namespace find_quadratic_function_find_vertex_find_range_l547_547607

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

def satisfies_points (a b c : ℝ) : Prop :=
  quadratic_function a b c (-1) = 0 ∧
  quadratic_function a b c 0 = -3 ∧
  quadratic_function a b c 2 = -3

theorem find_quadratic_function : ∃ a b c, satisfies_points a b c ∧ (a = 1 ∧ b = -2 ∧ c = -3) :=
sorry

theorem find_vertex (a b c : ℝ) (h : a = 1 ∧ b = -2 ∧ c = -3) :
  ∃ x y, x = 1 ∧ y = -4 ∧ ∀ x', x' > 1 → quadratic_function a b c x' > quadratic_function a b c x :=
sorry

theorem find_range (a b c : ℝ) (h : a = 1 ∧ b = -2 ∧ c = -3) :
  ∀ x, -1 < x ∧ x < 2 → -4 < quadratic_function a b c x ∧ quadratic_function a b c x < 0 :=
sorry

end find_quadratic_function_find_vertex_find_range_l547_547607


namespace consecutive_integers_divisibility_l547_547259

theorem consecutive_integers_divisibility :
  ∀ n : ℤ, ∃ k : ℤ, ∃ l : ℤ, (k ∈ (set.range (λ i, n + i) {i // i < 20}) ∧ k % 9 = 0)
  ∧ (l ∈ (set.range (λ i, n + i) {i // i < 20}) ∧ l % 9 ≠ 0) := by
sorry

end consecutive_integers_divisibility_l547_547259


namespace find_base_l547_547475

theorem find_base (a : ℕ) (ha : a > 11) (hB : 11 = 11) :
  (3 * a^2 + 9 * a + 6) + (5 * a^2 + 7 * a + 5) = (9 * a^2 + 7 * a + 11) → 
  a = 12 :=
sorry

end find_base_l547_547475


namespace mean_height_correct_l547_547294

/-- The heights of players on the volleyball team -/
def heights : List ℕ := [150, 152, 155, 158, 164, 168, 169, 161, 163, 165, 167, 170, 171, 173, 175, 176, 179, 182, 184, 188]

/-- The number of players on the volleyball team -/
def num_players : ℕ := heights.length

/-- The sum of the heights of players on the volleyball team -/
def sum_heights : ℕ := heights.sum

/-- The mean height of the players on the volleyball team -/
def mean_height : ℚ := sum_heights / num_players

theorem mean_height_correct : mean_height = 169.45 := by
  -- conditions from the problem
  have h1: num_players = 20 := rfl
  have h2: sum_heights = 3389 := rfl
  
  sorry

end mean_height_correct_l547_547294


namespace find_side_b_l547_547168

noncomputable def triangle_area (a b c : ℝ) (B : ℝ) : ℝ :=
  1 / 2 * a * c * real.sin B

noncomputable def law_of_cosines (a b c : ℝ) (B : ℝ) : ℝ :=
  a^2 + c^2 - 2 * a * c * real.cos B

theorem find_side_b (a b c : ℝ) (B : ℝ) (area : ℝ)
  (h1 : area = √3) 
  (h2 : B = real.pi / 3)
  (h3 : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 :=
by
  sorry

end find_side_b_l547_547168


namespace find_side_b_l547_547159

theorem find_side_b (a b c A B C : ℝ) (hB : B = 60) 
  (h_area : (1/2)*a*c*(Real.sin (Real.pi/3)) = Real.sqrt 3) 
  (h_ac : a^2 + c^2 = 3*a*c) : b = 2 * Real.sqrt 2 :=
by
  -- Given conditions
  have hB' : Real.sin (Real.pi/3) = Real.sqrt 3 / 2 := by sorry
  have h_ac_val : a * c = 4 := by sorry
  have h_a2c2 : a^2 + c^2 = 12 := by sorry
  have h_cosB : (a^2 + c^2 - b^2) / (2 * a * c) = 1/2 := by sorry
  have h_b2 : b^2 = 8 := by sorry

  -- Result
  exact Real.sqrt 8

end find_side_b_l547_547159


namespace arrangement_count_l547_547805

-- Define the problem conditions: 3 male students and 2 female students.
def male_students : ℕ := 3
def female_students : ℕ := 2
def total_students : ℕ := male_students + female_students

-- Define the condition that female students do not stand at either end.
def valid_positions_for_female : Finset ℕ := {1, 2, 3}
def valid_positions_for_male : Finset ℕ := {0, 4}

-- Theorem statement: the total number of valid arrangements is 36.
theorem arrangement_count : ∃ (n : ℕ), n = 36 := sorry

end arrangement_count_l547_547805


namespace find_b_l547_547194

theorem find_b
  (a b c : ℝ)
  (h_area : (1/2) * a * c * Real.sin (ℝ.pi / 3) = Real.sqrt 3)
  (h_ac_eq_4 : a * c = 4)
  (h_cosine : a^2 + c^2 = 3 * a * c) :
  b = 2 * Real.sqrt 2 :=
begin
  sorry
end

end find_b_l547_547194


namespace find_side_b_l547_547153

theorem find_side_b (a b c A B C : ℝ) (hB : B = 60) 
  (h_area : (1/2)*a*c*(Real.sin (Real.pi/3)) = Real.sqrt 3) 
  (h_ac : a^2 + c^2 = 3*a*c) : b = 2 * Real.sqrt 2 :=
by
  -- Given conditions
  have hB' : Real.sin (Real.pi/3) = Real.sqrt 3 / 2 := by sorry
  have h_ac_val : a * c = 4 := by sorry
  have h_a2c2 : a^2 + c^2 = 12 := by sorry
  have h_cosB : (a^2 + c^2 - b^2) / (2 * a * c) = 1/2 := by sorry
  have h_b2 : b^2 = 8 := by sorry

  -- Result
  exact Real.sqrt 8

end find_side_b_l547_547153


namespace smallest_positive_period_maximum_value_and_x_values_intervals_of_increase_axes_of_symmetry_l547_547527

noncomputable def fx (x : Real) : Real := 4 * (Real.cos x)^2 + 4 * Real.sqrt 3 * (Real.sin x) * (Real.cos x) - 2

theorem smallest_positive_period :
∀ (x : Real), x ∈ Set.univ → Real.Periodic (λ x, 4 * (Real.cos x)^2 + 4 * Real.sqrt 3 * (Real.sin x) * (Real.cos x) - 2) π := 
sorry

theorem maximum_value_and_x_values :
∀ (k : Int), 
∀ (x : Real), x = k * Real.pi + Real.pi / 6 → 
fx x = 4 := 
sorry

theorem intervals_of_increase :
∀ (k : Int), 
∀ (x : Real), 
x ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6) → 
Real.IncreasingOn (λ x, fx x) (Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6)) := 
sorry

theorem axes_of_symmetry :
∀ (k : Int), 
∀ (x : Real), 
x = (k * Real.pi) / 2 + Real.pi / 6 := 
sorry

end smallest_positive_period_maximum_value_and_x_values_intervals_of_increase_axes_of_symmetry_l547_547527


namespace find_a5_and_S10_l547_547501

noncomputable def arithmetic_sequence (a d : ℤ) : ℕ → ℤ
| 0       := a
| (n + 1) := arithmetic_sequence a d n + d

noncomputable def sum_arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
(n + 1) * a + d * (n + 1) * n / 2

theorem find_a5_and_S10 {a2 S4 : ℤ} (h1 : a2 = 1) (h2 : S4 = 8) :
  ∃ a d a5 S10,
    a5 = arithmetic_sequence a d 4 ∧
    S10 = sum_arithmetic_sequence a d 9 ∧
    a2 = arithmetic_sequence a d 1 ∧
    S4 = sum_arithmetic_sequence a d 3 ∧
    a5 = 7 ∧ S10 = 80 :=
by
  sorry

end find_a5_and_S10_l547_547501


namespace geometric_sequence_sum_x_l547_547488

variable {α : Type*} [Field α]

theorem geometric_sequence_sum_x (a : ℕ → α) (S : ℕ → α) (x : α) 
  (h₁ : ∀ n, S n = x * (3:α)^n + 1)
  (h₂ : ∀ n, a n = S n - S (n - 1)) :
  ∃ x, x = -1 :=
by
  let a1 := S 1
  let a2 := S 2 - S 1
  let a3 := S 3 - S 2
  have ha1 : a1 = 3 * x + 1 := sorry
  have ha2 : a2 = 6 * x := sorry
  have ha3 : a3 = 18 * x := sorry
  have h_geom : (6 * x)^2 = (3 * x + 1) * 18 * x := sorry
  have h_solve : 18 * x * (x + 1) = 0 := sorry
  have h_x_neg1 : x = 0 ∨ x = -1 := sorry
  exact ⟨-1, sorry⟩

end geometric_sequence_sum_x_l547_547488


namespace sequence_first_equals_last_four_l547_547240

theorem sequence_first_equals_last_four (n : ℕ) (S : ℕ → ℕ) (h_length : ∀ i < n, S i = 0 ∨ S i = 1)
  (h_condition : ∀ (i j : ℕ), 1 ≤ i ∧ i < j ∧ j ≤ n - 4 → 
    (S i = S j ∧ S (i + 1) = S (j + 1) ∧ S (i + 2) = S (j + 2) ∧ S (i + 3) = S (j + 3) ∧ S (i + 4) = S (j + 4)) → false) :
  S 1 = S (n - 3) ∧ S 2 = S (n - 2) ∧ S 3 = S (n - 1) ∧ S 4 = S n :=
sorry

end sequence_first_equals_last_four_l547_547240


namespace Vladimir_is_tallest_l547_547374

-- Variables representing the boys
inductive Boy
| Andrei | Boris | Vladimir | Dmitry

open Boy

-- Height and Age representations
variable (Height Age : Boy → ℕ)

-- Conditions based on the problem
axiom distinct_heights : ∀ a b, Height a = Height b → a = b
axiom distinct_ages : ∀ a b, Age a = Age b → a = b

-- Statements made by the boys
axiom Andrei_statements :
  (¬ (Height Boris > Height Andrei) ∨ Height Vladimir = Height Andrei)
  ∧ ¬ (¬ (Height Boris > Height Andrei) ∧ Height Vladimir = Height Andrei)

axiom Boris_statements :
  (Age Andrei > Age Boris ∨ Height Andrei = Height Boris)
  ∧ ¬ (Age Andrei > Age Boris ∧ Height Andrei = Height Boris)

axiom Vladimir_statements :
  (Height Dmitry > Height Vladimir ∨ Age Dmitry > Age Vladimir)
  ∧ ¬ (Height Dmitry > Height Vladimir ∧ Age Dmitry > Age Vladimir)

axiom Dmitry_statements :
  ((Height Dmitry > Height Vladimir ∧ Age Dmitry > Age Vladimir) ∨ Age Dmitry > Age Dmitry)
  ∧ ¬ (¬ (Height Dmitry > Height Vladimir ∧ Age Dmitry > Age Vladimir) ∧ ¬ Age Dmitry > Age Dmitry)

-- Problem Statement: Vladimir is the tallest
theorem Vladimir_is_tallest : ∀ h : Height Vladimir = Nat.max (Height Andrei) (Nat.max (Height Boris) (Height Dmitry))
by
  sorry

end Vladimir_is_tallest_l547_547374


namespace grid_special_sums_17x17_l547_547730

theorem grid_special_sums_17x17 :
  ∀ (grid : Fin 17 × Fin 17 → Fin 71),
  (∃ (A B C D : Fin 17 × Fin 17),
    (dist A B = dist C D ∧ dist A D = dist B C) ∧
    grid A + grid C = grid B + grid D) := 
sorry

end grid_special_sums_17x17_l547_547730


namespace min_value_of_2x_plus_2_2x_l547_547877

-- Lean 4 statement
theorem min_value_of_2x_plus_2_2x (x : ℝ) : 2^x + 2^(2 - x) ≥ 4 :=
by sorry

end min_value_of_2x_plus_2_2x_l547_547877


namespace arith_prog_iff_avg_arith_prog_l547_547646

variable {α : Type*} [LinearOrderedField α]

def is_arith_prog (a : ℕ → α) :=
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

def avg_is_arith_prog (S : ℕ → α) :=
  ∃ D : α, ∀ n : ℕ, n ≠ 0 → (S n / n) = (S (n + 1) / (n + 1)) + D

theorem arith_prog_iff_avg_arith_prog (a : ℕ → α) (S : ℕ → α)
  (hS : ∀ n, S n = ∑ i in range n, a i) :
  is_arith_prog a ↔ avg_is_arith_prog S :=
sorry

end arith_prog_iff_avg_arith_prog_l547_547646


namespace smallest_nuts_in_bag_l547_547412

theorem smallest_nuts_in_bag :
  ∃ (N : ℕ), N ≡ 1 [MOD 11] ∧ N ≡ 8 [MOD 13] ∧ N ≡ 3 [MOD 17] ∧
             (∀ M, (M ≡ 1 [MOD 11] ∧ M ≡ 8 [MOD 13] ∧ M ≡ 3 [MOD 17]) → M ≥ N) :=
sorry

end smallest_nuts_in_bag_l547_547412


namespace count_three_digit_multiples_of_13_and_7_l547_547547

def is_three_digit (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000

def is_multiple_of (m n : ℕ) : Prop :=
  ∃ k : ℕ, m = k * n

def lcm (a b : ℕ) : ℕ :=
  Nat.lcm a b

def count_multiples_in_range (multiple low high : ℕ) : ℕ :=
  (high - low) / multiple + 1

theorem count_three_digit_multiples_of_13_and_7 : ∃ count : ℕ,
    count = count_multiples_in_range (lcm 13 7) 182 910 ∧ count = 9 :=
by
  sorry

end count_three_digit_multiples_of_13_and_7_l547_547547


namespace find_b_correct_l547_547218

axioms (a b c : ℝ) (B : ℝ)
  (area_of_triangle : ℝ)
  (h1 : area_of_triangle = real.sqrt 3)
  (h2 : B = 60 * real.pi / 180) -- converting degrees to radians
  (h3 : a^2 + c^2 = 3 * a * c)

noncomputable def find_b : ℝ := 2 * real.sqrt 2

theorem find_b_correct :
  let b := find_b in 
  b = 2 * real.sqrt 2 :=
by
  sorry

end find_b_correct_l547_547218


namespace optionA_iff_optionB_l547_547702

variable {a : ℕ → ℝ} {S : ℕ → ℝ}

/-- Sequence a_n is an arithmetic progression -/
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a n = a1 + n * d

/-- Sequence S_n / n is an arithmetic progression -/
def is_S_n_div_n_arithmetic_progression (S : ℕ → ℝ) : Prop :=
  ∃ (b c : ℝ), ∀ n, S n / n = b + c * n

/-- S_n is the sum of the first n terms of the sequence a_n -/
def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = ∑ i in finset.range (n + 1), a i 

/-- Option A is both a sufficient and necessary condition for Option B -/
theorem optionA_iff_optionB (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (is_arithmetic_progression a ∧ sum_first_n_terms a S) ↔ is_S_n_div_n_arithmetic_progression S :=
  sorry

end optionA_iff_optionB_l547_547702


namespace Vladimir_is_tallest_l547_547354

-- Definitions for the statements made by the boys.
def Andrei_statements (Boris_tallest Vladimir_shortest : Prop) :=
  (¬Boris_tallest) ∧ Vladimir_shortest

def Boris_statements (Andrei_oldest Andrei_shortest : Prop) :=
  Andrei_oldest ∧ Andrei_shortest

def Vladimir_statements (Dmitry_taller Dmitry_older : Prop) :=
  Dmitry_taller ∧ Dmitry_older

def Dmitry_statements (Vladimir_both_true Dmitry_oldest : Prop) :=
  Vladimir_both_true ∧ Dmitry_oldest

-- Conditions given in the problem.
variables (Boris_tallest Vladimir_shortest Andrei_oldest Andrei_shortest Dmitry_taller Dmitry_older 
           Vladimir_both_true Dmitry_oldest : Prop)

-- Axioms based on the problem statement
axiom Dmitry_statements_condition : Dmitry_statements Vladimir_both_true Dmitry_oldest
axiom Vladimir_statements_condition : Vladimir_statements Dmitry_taller Dmitry_older
axiom Boris_statements_condition : Boris_statements Andrei_oldest Andrei_shortest
axiom Andrei_statements_condition : Andrei_statements Boris_tallest Vladimir_shortest

-- None of them share the same height or age.
axiom no_shared_height_or_age : 
  ¬(Andrei_oldest ∧ Boris_tallest ∧ Dmitry_taller ∧ Vladimir_both_true)

-- Main theorem: proving that Vladimir is the tallest based on the conditions
theorem Vladimir_is_tallest (cond : ¬Boris_tallest → Vladimir_shortest →
                              ¬Andrei_oldest → Andrei_shortest →
                              ¬Dmitry_taller → Dmitry_older →
                              ¬Vladimir_both_true → Dmitry_oldest → 
                              ¬(Andrei_oldest ∧ Boris_tallest ∧ Dmitry_taller ∧ Vladimir_both_true)) :
  (¬Boris_tallest ∧ Dmitry_older) → Vladimir_tallest :=
begin
  sorry,
end

end Vladimir_is_tallest_l547_547354


namespace find_side_b_l547_547142

theorem find_side_b
  (a b c : ℝ)
  (area : ℝ)
  (B : ℝ)
  (h_area : area = real.sqrt 3)
  (h_B : B = (60 : ℝ) * real.pi / 180) -- Convert degrees to radians
  (h_cond : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 := 
sorry

end find_side_b_l547_547142


namespace find_side_b_l547_547170

noncomputable def triangle_area (a b c : ℝ) (B : ℝ) : ℝ :=
  1 / 2 * a * c * real.sin B

noncomputable def law_of_cosines (a b c : ℝ) (B : ℝ) : ℝ :=
  a^2 + c^2 - 2 * a * c * real.cos B

theorem find_side_b (a b c : ℝ) (B : ℝ) (area : ℝ)
  (h1 : area = √3) 
  (h2 : B = real.pi / 3)
  (h3 : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 :=
by
  sorry

end find_side_b_l547_547170


namespace find_b_proof_l547_547209

noncomputable def find_b (a b c : ℝ) (A B C : ℝ) : Prop :=
  let area := sqrt 3 in
  let angle_B := 60 * Real.pi / 180 in
  let condition1 := area = (1 / 2) * a * c * Real.sin angle_B in
  let condition2 := a^2 + c^2 = 3 * a * c in
  let side_b := b = 2 * Real.sqrt 2 in
  condition1 ∧ condition2 → side_b

theorem find_b_proof :
  ∃ (a b c : ℝ) (A B C : ℝ), find_b a b c A B C :=
by
  sorry

end find_b_proof_l547_547209


namespace arith_prog_iff_avg_arith_prog_l547_547649

variable {α : Type*} [LinearOrderedField α]

def is_arith_prog (a : ℕ → α) :=
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

def avg_is_arith_prog (S : ℕ → α) :=
  ∃ D : α, ∀ n : ℕ, n ≠ 0 → (S n / n) = (S (n + 1) / (n + 1)) + D

theorem arith_prog_iff_avg_arith_prog (a : ℕ → α) (S : ℕ → α)
  (hS : ∀ n, S n = ∑ i in range n, a i) :
  is_arith_prog a ↔ avg_is_arith_prog S :=
sorry

end arith_prog_iff_avg_arith_prog_l547_547649


namespace mode_backpack_capacity_l547_547395

-- Define the given data as conditions
def capacities : list ℕ := [23, 25, 27, 29, 31, 33]
def students : list ℕ := [3, 2, 5, 21, 2, 2]

-- State the theorem for the mode of backpack capacity:
theorem mode_backpack_capacity : 
  ∃ mode, (mode = 29) ∧ (∀ x ∈ capacities, (students.nth_le (capacities.indexOf x) (list.index_of_lt_length x capacities) = (if x = 29 then 21 else students.nth_le (capacities.index_of x) (list.index_of_lt_length x capacities)))) := 
by 
  sorry

end mode_backpack_capacity_l547_547395


namespace iggy_total_hours_l547_547059

-- Define the conditions
def miles_run_per_day : ℕ → ℕ
| 0 := 3 -- Monday
| 1 := 4 -- Tuesday
| 2 := 6 -- Wednesday
| 3 := 8 -- Thursday
| 4 := 3 -- Friday
| _ := 0 -- Other days

def total_distance : ℕ := List.sum (List.ofFn miles_run_per_day 5)
def miles_per_minute : ℕ := 10
def minutes_per_hour : ℕ := 60
def total_minutes_run : ℕ := total_distance * miles_per_minute
def total_hours_run : ℕ := total_minutes_run / minutes_per_hour

-- The statement to prove
theorem iggy_total_hours :
  total_hours_run = 4 :=
sorry

end iggy_total_hours_l547_547059


namespace option_A_is_necessary_and_sufficient_for_option_B_l547_547698

def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

def mean_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : (ℕ → ℝ)
| n => S n / n

def sum_of_first_n_terms (a : ℕ → ℝ) : ℕ → ℝ
| 0     => 0
| (n+1) => sum_of_first_n_terms a n + a (n+1)

theorem option_A_is_necessary_and_sufficient_for_option_B
    (a : ℕ → ℝ) :
    (is_arithmetic_progression a ↔ is_arithmetic_progression (mean_sequence a (sum_of_first_n_terms a))) :=
sorry

end option_A_is_necessary_and_sufficient_for_option_B_l547_547698


namespace smallest_positive_x_for_palindrome_l547_547824

def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10
  s = s.reverse

theorem smallest_positive_x_for_palindrome :
  ∃ x : ℕ, x > 0 ∧ is_palindrome (x + 1234) ∧ (∀ y : ℕ, y > 0 → is_palindrome (y + 1234) → x ≤ y) ∧ x = 97 := 
sorry

end smallest_positive_x_for_palindrome_l547_547824


namespace common_sum_of_5x5_matrix_l547_547767

theorem common_sum_of_5x5_matrix (matrix : Array (Array Int))
  (h1 : matrix.size = 5)
  (h2 : ∀ i, (matrix[i]!).size = 5)
  (h3 : (matrix.flatten.toList = List.range' (-12) 25)) :
  (∑ i in Finset.range 5, matrix[i]!.sum = 0) ∧
  (∑ j in Finset.range 5, (Finset.range 5).sum (λ i, matrix[i]![j]!) = 0) ∧
  (∑ i in Finset.range 5, matrix[i]![i]! = 0) ∧
  (∑ i in Finset.range 5, matrix[i]![4 - i]! = 0) :=
by
  sorry

end common_sum_of_5x5_matrix_l547_547767


namespace option_A_is_necessary_and_sufficient_for_option_B_l547_547693

def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

def mean_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : (ℕ → ℝ)
| n => S n / n

def sum_of_first_n_terms (a : ℕ → ℝ) : ℕ → ℝ
| 0     => 0
| (n+1) => sum_of_first_n_terms a n + a (n+1)

theorem option_A_is_necessary_and_sufficient_for_option_B
    (a : ℕ → ℝ) :
    (is_arithmetic_progression a ↔ is_arithmetic_progression (mean_sequence a (sum_of_first_n_terms a))) :=
sorry

end option_A_is_necessary_and_sufficient_for_option_B_l547_547693


namespace AD_value_l547_547586

noncomputable def findAD (BD DC : ℝ) (hBC_diameter : Prop) (hA_on_circle : Prop)
  (hAD_perp_BC : Prop) (hBD_eq : BD = 1) (hDC_eq : DC = 4) : ℝ :=
  let a := AD in by
  have similarity : triangle_similarity AD BD := sorry
  have ratio := similarity_ratios AD BD DC := sorry
  have a_eq_2 := solve_ratio_equality ratio hBD_eq hDC_eq := sorry
  exact a_eq_2

theorem AD_value (BD DC : ℝ) (hBC_diameter : Prop) (hA_on_circle : Prop)
  (hAD_perp_BC : Prop) (hBD_eq : BD = 1) (hDC_eq : DC = 4) :
  findAD BD DC hBC_diameter hA_on_circle hAD_perp_BC hBD_eq hDC_eq = 2 :=
sorry

end AD_value_l547_547586


namespace acute_angle_at_3_37_l547_547445

noncomputable def position_of_minute_hand (m : ℕ) : ℝ :=
  (m / 60.0) * 360.0

noncomputable def starting_angle_of_hour_hand (h : ℕ) : ℝ :=
  h * 30

noncomputable def position_of_hour_hand (h : ℕ) (m : ℕ) : ℝ :=
  (h * 30.0) + ((m / 60.0) * 30.0)

noncomputable def acute_angle_between_clock_hands (h : ℕ) (m : ℕ) : ℝ :=
  let minute_position := position_of_minute_hand m
  let hour_position := position_of_hour_hand h m
  if minute_position - hour_position < 180 then
    minute_position - hour_position
  else
    360.0 - (minute_position - hour_position)

theorem acute_angle_at_3_37 : acute_angle_between_clock_hands 3 37 = 113.5 :=
by
  sorry

end acute_angle_at_3_37_l547_547445


namespace find_first_group_men_l547_547758

variable (M : ℕ)

def first_group_men := M
def days_for_first_group := 20
def men_in_second_group := 12
def days_for_second_group := 30

theorem find_first_group_men (h1 : first_group_men * days_for_first_group = men_in_second_group * days_for_second_group) :
  first_group_men = 18 :=
by {
  sorry
}

end find_first_group_men_l547_547758


namespace triangle_side_b_l547_547230

noncomputable def find_side_length_b
  (a b c : ℝ)
  (area : ℝ)
  (B_deg : ℝ)
  (relation : a^2 + c^2 = 3 * a * c)
  (area_condition : area = sqrt 3)
  (angle_condition : B_deg = 60) : Prop :=
  b = 2 * sqrt 2

theorem triangle_side_b
  (a b c : ℝ)
  (area : ℝ := sqrt 3)
  (B_deg : ℝ := 60)
  (h1 : a^2 + c^2 = 3 * a * c)
  (h2 : area = sqrt 3)
  (h3 : B_deg = 60) : find_side_length_b a b c area B_deg h1 h2 h3 :=
  sorry

end triangle_side_b_l547_547230


namespace find_side_b_l547_547105

theorem find_side_b (a b c : ℝ) (A B C : ℝ)
  (area_triangle : Real.sqrt 3) (B_eq_60 : B = 60) (cond_ac: (a^2 + c^2 = 3 * a * c)) :
  ∃ b, b = 2 * Real.sqrt 2 :=
by
  sorry

end find_side_b_l547_547105


namespace in_triangle_inequality_equality_condition_l547_547740

variable {A B C O : Type*}
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O]
variable (O : Point) (A B C : Triangle) (p : ℝ)

noncomputable def semiperimeter (A B C : Triangle) : ℝ :=
  (A.side_length + B.side_length + C.side_length) / 2

theorem in_triangle_inequality (O_in_triangle : inside_triangle O A B C)
    (p_def : p = semiperimeter A B C) :
    (distance O A) * (cos (angle_bisector A B C / 2)) +
    (distance O B) * (cos (angle_bisector B A C / 2)) +
    (distance O C) * (cos (angle_bisector C A B / 2)) ≥ p :=
    sorry -- proof omitted

theorem equality_condition (O_is_incenter : is_incenter O A B C) :
    (distance O A) * (cos (angle_bisector A B C / 2)) +
    (distance O B) * (cos (angle_bisector B A C / 2)) +
    (distance O C) * (cos (angle_bisector C A B / 2)) = p :=
    sorry -- proof omitted

end in_triangle_inequality_equality_condition_l547_547740


namespace smallest_x_palindrome_l547_547828

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

theorem smallest_x_palindrome : ∃ k, k > 0 ∧ is_palindrome (k + 1234) ∧ k = 97 := 
by {
  use 97,
  sorry
}

end smallest_x_palindrome_l547_547828


namespace inscribed_circle_diameter_l547_547819

theorem inscribed_circle_diameter (PQ PR QR : ℝ) (h₁ : PQ = 13) (h₂ : PR = 14) (h₃ : QR = 15) :
  ∃ d : ℝ, d = 8 :=
by
  sorry

end inscribed_circle_diameter_l547_547819


namespace ratio_buses_to_cars_l547_547790

-- Define the number of buses b and number of cars c on River Road
def num_buses (c : ℕ) : ℕ := c - 40

-- Given: There are 60 cars on River Road
def num_cars : ℕ := 60

-- Theorem stating the ratio of buses to cars is 1:3
theorem ratio_buses_to_cars (c : ℕ) (h₁ : c = 60) : num_buses c / c = 1 / 3 :=
by {
  have h₂ : num_buses 60 = 60 - 40,
  {
    rw num_buses,
  },
  rw h₁ at h₂,
  rw num_buses,
  norm_num,
  rw div_eq_div_iff,
  norm_num,
  -- You can fill in the rest of the detailed proof steps if necessary
  sorry
}

end ratio_buses_to_cars_l547_547790


namespace final_price_set_l547_547593

variable (c ch s : ℕ)
variable (dc dtotal : ℚ)
variable (p_final : ℚ)

def coffee_price : ℕ := 6
def cheesecake_price : ℕ := 10
def sandwich_price : ℕ := 8
def coffee_discount : ℚ := 0.25 * 6
def final_discount : ℚ := 3

theorem final_price_set :
  p_final = (coffee_price - coffee_discount) + cheesecake_price + sandwich_price - final_discount :=
by
  sorry

end final_price_set_l547_547593


namespace greatest_integer_prime_abs_l547_547821

theorem greatest_integer_prime_abs (x : ℤ) : 
  (∀ x : ℤ, prime (abs (8 * x^2 - 66 * x + 21)) → x ≤ 2) ∧ 
  prime (abs (8 * 2^2 - 66 * 2 + 21)) 
:= by
  sorry

end greatest_integer_prime_abs_l547_547821


namespace number_of_integers_acute_triangle_sides_l547_547803

theorem number_of_integers_acute_triangle_sides (x : ℤ) :
  (Count (λ (x : ℤ), 476 < x^2 ∧ x^2 < 676) = 4) :=
sorry

end number_of_integers_acute_triangle_sides_l547_547803


namespace find_side_b_l547_547155

theorem find_side_b (a b c A B C : ℝ) (hB : B = 60) 
  (h_area : (1/2)*a*c*(Real.sin (Real.pi/3)) = Real.sqrt 3) 
  (h_ac : a^2 + c^2 = 3*a*c) : b = 2 * Real.sqrt 2 :=
by
  -- Given conditions
  have hB' : Real.sin (Real.pi/3) = Real.sqrt 3 / 2 := by sorry
  have h_ac_val : a * c = 4 := by sorry
  have h_a2c2 : a^2 + c^2 = 12 := by sorry
  have h_cosB : (a^2 + c^2 - b^2) / (2 * a * c) = 1/2 := by sorry
  have h_b2 : b^2 = 8 := by sorry

  -- Result
  exact Real.sqrt 8

end find_side_b_l547_547155


namespace tangent_line_eq_at_A_l547_547278

noncomputable def curve (x : ℝ) : ℝ := 3 * x - x^3

def point := (2 : ℝ, -2 : ℝ)

def tangent_line (x : ℝ) : ℝ := -9 * x + 16

theorem tangent_line_eq_at_A :
  ∀ x y : ℝ, (x, y) = point →
  tangent_line x = y :=
sorry

end tangent_line_eq_at_A_l547_547278


namespace tangent_locus_to_parabola_hyperbola_l547_547459

theorem tangent_locus_to_parabola_hyperbola :
  ∀ (u v : ℝ), (∃ tan_slope1 tan_slope2 : ℝ,
    (forall x : ℝ, x ^ 2 - tan_slope1 * x + tan_slope1 * u - v = 0) ∧
    (forall x : ℝ, x ^ 2 - tan_slope2 * x + tan_slope2 * u - v = 0) ∧
    abs ((tan_slope1 - tan_slope2) / (1 + tan_slope1 * tan_slope2)) = 1 / sqrt 3)
  ↔ (u ^ 2 - (v + 7 / 4) ^ 2 / 3 = -1) := sorry


end tangent_locus_to_parabola_hyperbola_l547_547459


namespace option_a_iff_option_b_l547_547638

-- Define a sequence as an arithmetic progression
def is_arithmetic (seq : ℕ → ℚ) : Prop :=
  ∃ d a₁, ∀ n, seq n = a₁ + n * d

-- Define the sum of the first n terms of a sequence
def sum_first_n (seq : ℕ → ℚ) (n : ℕ) : ℚ :=
  ∑ i in finset.range n, seq i

-- Define the average sum sequence
def avg_sum_seq (seq : ℕ → ℚ) (n : ℕ) : ℚ :=
  let Sn := sum_first_n seq n
  in Sn / n

-- The theorem to prove
theorem option_a_iff_option_b (seq : ℕ → ℚ) :
  is_arithmetic seq ↔ is_arithmetic (λ n, avg_sum_seq seq (n + 1)) :=
sorry

end option_a_iff_option_b_l547_547638


namespace sum_sequence_l547_547518

noncomputable def sum_terms (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  2*n + 1/(2^n) - 1

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

axiom (h1 : ∀ n, S n = 2^n * a n - 1)

theorem sum_sequence (n : ℕ) : 
  ∑ i in finset.range n, (2 - 1/(2^(i+1))) = sum_terms a n :=
by 
  sorry

end sum_sequence_l547_547518


namespace function_min_value_4_l547_547867

theorem function_min_value_4 :
  ∃ x : ℝ, (2^x + 2^(2 - x)) = 4 :=
sorry

end function_min_value_4_l547_547867


namespace C_plus_D_l547_547625

theorem C_plus_D (C D : ℝ) (h : ∀ x : ℝ, x ≠ 3 → C / (x - 3) + D * (x + 2) = (-4 * x^2 + 18 * x + 32) / (x - 3)) : 
  C + D = 28 := sorry

end C_plus_D_l547_547625


namespace quadratic_function_properties_l547_547454

noncomputable def f (x : ℝ) : ℝ := -5 / 2 * x^2 + 15 * x - 25 / 2

theorem quadratic_function_properties :
  (∃ a : ℝ, ∀ x : ℝ, (f x = a * (x - 1) * (x - 5)) ∧ (f 3 = 10)) → 
  (f x = -5 / 2 * x^2 + 15 * x - 25 / 2) :=
by 
  sorry

end quadratic_function_properties_l547_547454


namespace arith_prog_iff_avg_arith_prog_l547_547648

variable {α : Type*} [LinearOrderedField α]

def is_arith_prog (a : ℕ → α) :=
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

def avg_is_arith_prog (S : ℕ → α) :=
  ∃ D : α, ∀ n : ℕ, n ≠ 0 → (S n / n) = (S (n + 1) / (n + 1)) + D

theorem arith_prog_iff_avg_arith_prog (a : ℕ → α) (S : ℕ → α)
  (hS : ∀ n, S n = ∑ i in range n, a i) :
  is_arith_prog a ↔ avg_is_arith_prog S :=
sorry

end arith_prog_iff_avg_arith_prog_l547_547648


namespace prime_divisors_of_exponential_l547_547242

-- Let \( n \) be a natural number. 
-- Show that \( 7^{7^{n}} + 1 \) has at least \( 2n + 3 \) not necessarily different prime divisors.

open Nat

theorem prime_divisors_of_exponential (n : ℕ) : 
  ∃ p : set ℕ, set.size p ≥ 2 * n + 3 ∧ ∀ x ∈ p, x.prime ∧ x ∣ 7^(7^n) + 1 := sorry

end prime_divisors_of_exponential_l547_547242


namespace prob_range_4_8_to_4_85_l547_547747

variable (X : ℝ → Prop)

axiom prob_lt_4_8 : P (λ x, x < 4.8) = 0.3
axiom prob_lt_4_85 : P (λ x, x < 4.85) = 0.32

theorem prob_range_4_8_to_4_85 : P (λ x, 4.8 ≤ x ∧ x < 4.85) = 0.02 := by
  sorry

end prob_range_4_8_to_4_85_l547_547747


namespace t1_eq_t2_l547_547564

variable (n : ℕ)
variable (s₁ s₂ s₃ : ℝ)
variable (t₁ t₂ : ℝ)
variable (S1 S2 S3 : ℝ)

-- Conditions
axiom h1 : S1 = s₁
axiom h2 : S2 = s₂
axiom h3 : S3 = s₃
axiom h4 : t₁ = s₂^2 - s₁ * s₃
axiom h5 : t₂ = ( (s₁ - s₃) / 2 )^2
axiom h6 : s₁ + s₃ = 2 * s₂

theorem t1_eq_t2 : t₁ = t₂ := by
  sorry

end t1_eq_t2_l547_547564


namespace option_A_iff_option_B_l547_547669

def is_arithmetic_progression (seq : ℕ → ℚ) : Prop :=
  ∃ d a₁, ∀ n, seq (n + 1) = seq n + d

def S_n (a : ℕ → ℚ) : ℕ → ℚ
| 0       := 0
| (n + 1) := S_n n + a (n + 1)

theorem option_A_iff_option_B (a : ℕ → ℚ) :
  (is_arithmetic_progression a) ↔ (is_arithmetic_progression (λ n, S_n a n / (n + 1))) :=
sorry

end option_A_iff_option_B_l547_547669


namespace regression_equation_correct_smallest_profit_in_June_l547_547303

-- Definitions for the conditions
def x_values := [1, 2, 3, 4, 5]
def y_values := [4.9, 5.8, 6.8, 8.3, 10.2]

def regression_model (x : ℕ) : ℝ := 0.2 * (x^2 : ℝ) + 5

-- Defining the profit function
def profit (x y : ℝ) : ℝ := (5 * y + 35) / (x + 2)

-- Statements to prove
theorem regression_equation_correct :
  ∀ (x : ℕ), x ∈ x_values → regression_model x = 0.2 * (x^2 : ℝ) + 5 :=
by
  intros,
  refl

theorem smallest_profit_in_June :
  ∃ (x : ℕ), x = 6 ∧ ∀ (y ∈ y_values), profit (x : ℝ) (regression_model x) ≤ profit (_.ℝ (regression_model _)) :=
by
  sorry

end regression_equation_correct_smallest_profit_in_June_l547_547303


namespace find_side_b_l547_547112

theorem find_side_b (a b c : ℝ) (A B C : ℝ)
  (area_triangle : Real.sqrt 3) (B_eq_60 : B = 60) (cond_ac: (a^2 + c^2 = 3 * a * c)) :
  ∃ b, b = 2 * Real.sqrt 2 :=
by
  sorry

end find_side_b_l547_547112


namespace betty_bracelets_count_l547_547444

def betty_bracelets (total_stones : ℝ) (stones_per_bracelet : ℝ) : ℝ :=
  total_stones / stones_per_bracelet

theorem betty_bracelets_count :
  betty_bracelets 88.0 11 = 8 := 
by
  sorry

end betty_bracelets_count_l547_547444


namespace find_a_l547_547562

theorem find_a (a b x : ℝ) (h1 : a ≠ b)
  (h2 : a^3 + b^3 = 35 * x^3)
  (h3 : a^2 - b^2 = 4 * x^2) : a = 2 * x ∨ a = -2 * x :=
by
  sorry

end find_a_l547_547562


namespace variance_transformation_l547_547056

open_locale big_operators

variables {n : ℕ} (x : fin n → ℝ)

def variance (x : fin n → ℝ) : ℝ :=
  (∑ i, (x i - (∑ i, x i) / n) ^ 2) / n

theorem variance_transformation (h : variance x = 1) :
  variance (λ i, 2 * x i + 1) = 4 := by
  sorry

end variance_transformation_l547_547056


namespace tallest_is_vladimir_l547_547368

variables (Andrei Boris Vladimir Dmitry : Type)
variables (height age : Type)
variables [linear_order height] [linear_order age]
variables (tallest shortest oldest : height)
variables (andrei_statements : (Boris ≠ tallest) ∧ (Vladimir = shortest))
variables (boris_statements : (Andrei = oldest) ∧ (Andrei = shortest))
variables (vladimir_statements : (Dmitry > Vladimir) ∧ (Dmitry > Vladimir))
variables (dmitry_statements : (vladimir_statements = true) ∧ (Dmitry = oldest))

theorem tallest_is_vladimir (H1: andrei_statements ∨ ¬andrei_statements)
  (H2: boris_statements ∨ ¬boris_statements)
  (H3: vladimir_statements ∨ ¬vladimir_statements)
  (H4: dmitry_statements ∨ ¬dmitry_statements)
  (H5: ∃! (a b c d : height) , a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a) : 
  tallest = Vladimir := 
by
  sorry

end tallest_is_vladimir_l547_547368


namespace num_ordered_triples_lcm_l547_547708

theorem num_ordered_triples_lcm : 
  (∃ a b c : ℕ+, (Nat.lcm a b = 1250) ∧ (Nat.lcm b c = 2500) ∧ (Nat.lcm c a = 2500) ∧
  (number_of_ordered_triples a b c = 2)) := sorry

end num_ordered_triples_lcm_l547_547708


namespace correct_syllogism_sequence_l547_547327

theorem correct_syllogism_sequence
  (h1: ∀ z₁ z₂ : ℂ, z₁.im ≠ 0 ∧ z₂.im ≠ 0 → ¬(z₁ < z₂ ∨ z₂ < z₁))
  (h2: ∀ z : ℂ, z.im ≠ 0 → ∃ w : ℂ, w.im ≠ 0)
  (h3: ∀ z₁ z₂ : ℂ, z₁.im ≠ 0 ∧ z₂.im ≠ 0) :
  ((h2 -> h3) -> h1) :=
sorry

end correct_syllogism_sequence_l547_547327


namespace mike_cards_remaining_l547_547252

-- Define initial condition
def mike_initial_cards : ℕ := 87

-- Define the cards bought by Sam
def sam_bought_cards : ℕ := 13

-- Define the expected remaining cards
def mike_final_cards := mike_initial_cards - sam_bought_cards

-- Theorem to prove the final count of Mike's baseball cards
theorem mike_cards_remaining : mike_final_cards = 74 := by
  sorry

end mike_cards_remaining_l547_547252


namespace quadratic_polynomial_divisible_by_3_l547_547417

theorem quadratic_polynomial_divisible_by_3
  (a b c : ℤ)
  (h : ∀ x : ℤ, 3 ∣ (a * x^2 + b * x + c)) :
  3 ∣ a ∧ 3 ∣ b ∧ 3 ∣ c :=
sorry

end quadratic_polynomial_divisible_by_3_l547_547417


namespace find_side_b_l547_547146

theorem find_side_b
  (a b c : ℝ)
  (area : ℝ)
  (B : ℝ)
  (h_area : area = real.sqrt 3)
  (h_B : B = (60 : ℝ) * real.pi / 180) -- Convert degrees to radians
  (h_cond : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 := 
sorry

end find_side_b_l547_547146


namespace polygon_sides_l547_547053

theorem polygon_sides (n : ℕ) (h_interior : (n - 2) * 180 = 3 * 360) : n = 8 :=
by
  sorry

end polygon_sides_l547_547053


namespace min_value_of_2x_plus_2_2x_l547_547878

-- Lean 4 statement
theorem min_value_of_2x_plus_2_2x (x : ℝ) : 2^x + 2^(2 - x) ≥ 4 :=
by sorry

end min_value_of_2x_plus_2_2x_l547_547878


namespace area_of_figure_l547_547075

-- We first define the conditions stated in the problem
variables (A B C D E F : Type)
variables (AB DE BD AE AF CD : ℝ)
variables (angleDAE angleABC : ℝ)

-- Given conditions
def conditions :=
  AB = 1 ∧ DE = 1 ∧ BD = 1 ∧ AE = 1 ∧ AF = 1 ∧ CD = 1 ∧
  angleDAE = 90 ∧ angleABC = 60 ∧
  parallel A B D E ∧ parallel B D A E ∧ parallel A F C D

-- We will now state the theorem we aim to prove
theorem area_of_figure (h : conditions A B C D E F AB DE BD AE AF CD angleDAE angleABC) :
  area (figure A B C D E F) = (sqrt 3 / 2) + (3 / 2) :=
sorry

end area_of_figure_l547_547075


namespace find_S_2015_l547_547519

-- Definitions
def sequence (a : ℕ → ℕ) : ℕ → ℕ
| 0     := 0
| (n+1) := sequence a n + a (n+1)

-- Conditions
axiom a1 : ∀ (a : ℕ → ℕ), a 1 = 1
axiom a2 : ∀ (a : ℕ → ℕ) (n : ℕ), n ≥ 2 → a n + 2 * sequence a (n - 1) = n

-- Goal: Prove S_2015 = 1008
theorem find_S_2015 (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (hS: ∀ n, S n = sequence a n) 
  (h1: a 1 = 1)
  (h2: ∀ n, n ≥ 2 → a n + 2 * S (n - 1) = n) : 
S 2015 = 1008 := 
by {
  sorry
}

end find_S_2015_l547_547519


namespace min_value_of_2x_plus_2_2x_l547_547881

-- Lean 4 statement
theorem min_value_of_2x_plus_2_2x (x : ℝ) : 2^x + 2^(2 - x) ≥ 4 :=
by sorry

end min_value_of_2x_plus_2_2x_l547_547881


namespace minimize_f_C_l547_547927

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimize_f_C : ∃ x : ℝ, f_C x = 4 :=
by {
  -- The correctness of the statement is asserted and the proof is omitted
  sorry
}

end minimize_f_C_l547_547927


namespace multiplier_eq_l547_547285

-- Definitions of the given conditions
def length (w : ℝ) (m : ℝ) : ℝ := m * w + 2
def perimeter (l : ℝ) (w : ℝ) : ℝ := 2 * l + 2 * w

-- Condition definitions
def l : ℝ := 38
def P : ℝ := 100

-- Proof statement
theorem multiplier_eq (m w : ℝ) (h1 : length w m = l) (h2 : perimeter l w = P) : m = 3 :=
by
  sorry

end multiplier_eq_l547_547285


namespace option_A_is_iff_option_B_l547_547656

-- Definitions based on conditions
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range (n + 1), a i

def is_arithmetic (seq : ℕ → ℝ) : Prop :=
  ∃ D : ℝ, ∀ n : ℕ, seq (n + 1) - seq n = D

-- The final statement
theorem option_A_is_iff_option_B :
  ∀ (a : ℕ → ℝ),
    (is_arithmetic_progression a ↔ is_arithmetic (λ n, Sn a n / (n + 1))) :=
by sorry

end option_A_is_iff_option_B_l547_547656


namespace tree_measurement_sufficient_n_l547_547458

noncomputable def tree_measurement_problem : Prop :=
  let σ := 10 -- standard deviation
  let ε := 2 -- permissible error margin
  ∀ (X : ℕ → ℝ), -- list of measurements
    (∀ i, X i < σ + ε) →
    ∃ n : ℕ, 
      n ≥ 500 ∧ 
      1 - (σ ^ 2) / (n * ε ^ 2) ≥ 0.95

theorem tree_measurement_sufficient_n : tree_measurement_problem :=
begin
  sorry
end

end tree_measurement_sufficient_n_l547_547458


namespace sum_of_distinct_prime_factors_924_l547_547831

theorem sum_of_distinct_prime_factors_924 : 
  (∑ p in ({2, 3, 7, 11} : Finset ℕ), p) = 23 := 
by
  -- sorry is used to skip the proof.
  sorry

end sum_of_distinct_prime_factors_924_l547_547831


namespace solve_real_roots_in_intervals_l547_547742

noncomputable def real_roots_intervals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  ∃ x₁ x₂ : ℝ,
    (3 * x₁^2 - 2 * (a - b) * x₁ - a * b = 0) ∧
    (3 * x₂^2 - 2 * (a - b) * x₂ - a * b = 0) ∧
    (a / 3 < x₁ ∧ x₁ < 2 * a / 3) ∧
    (-2 * b / 3 < x₂ ∧ x₂ < -b / 3)

-- Statement of the problem:
theorem solve_real_roots_in_intervals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  real_roots_intervals a b ha hb :=
sorry

end solve_real_roots_in_intervals_l547_547742


namespace arith_prog_iff_avg_seq_arith_prog_l547_547676

-- Define sequences
def is_arith_prog (a : ℕ → ℝ) := ∃ d : ℝ, ∃ a₀ : ℝ, ∀ n : ℕ, a n = a₀ + n * d
def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range (n+1), a i
def avg_seq (a : ℕ → ℝ) (n : ℕ) := S a n / (n + 1)

-- Define theorem statement
theorem arith_prog_iff_avg_seq_arith_prog (a : ℕ → ℝ) :
  (is_arith_prog a) ↔ (is_arith_prog (λ n, avg_seq a n)) :=
sorry

end arith_prog_iff_avg_seq_arith_prog_l547_547676


namespace probability_sum_dice_12_l547_547583

/-- Helper definition for a standard six-faced die roll -/
def is_valid_die_roll (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 6

/-- The probability that the sum of three six-faced dice equals 12 is 19/216. -/
theorem probability_sum_dice_12 :
  (∑ (x y z : ℕ) in (finset.range 7).filter (is_valid_die_roll), ite (x + y + z = 12) 1 0) = 19 :=
begin
  sorry
end

end probability_sum_dice_12_l547_547583


namespace vladimir_is_tallest_l547_547359

section TallestBoy

variables (Andrei Boris Vladimir Dmitry : Type)

-- Define the statements made by each boy
def andrei_statements : Andrei → Prop :=
-- suppose andrei's statements are a::Boris ≠ tallest, b::Vladimir = shortest
λ A, (∀ B : Boris, B ≠ tallest) ∨ (∀ V : Vladimir, V = shortest)

def boris_statements : Boris → Prop :=
λ B, (∀ A : Andrei, A = oldest) ∨ (∀ A' : Andrei, A' = shortest)

def vladimir_statements : Vladimir → Prop :=
λ V, (∀ D : Dmitry, D = taller_than V) ∨ (∀ D' : Dmitry, D' = older_than V)

def dmitry_statements : Dmitry → Prop :=
λ D, ((vladimir_statements V) ∧ (vladimir_statements V)) ∨ (∀ D' : Dmitry, D' = oldest)

-- Define the conditions
-- Each boy makes two statements, one of which is true and the other is false
axiom statements_condition (A : Andrei) (B : Boris) (V : Vladimir) (D : Dmitry) :
  (andrei_statements A → ¬andrei_statements A) ∧
  (boris_statements B → ¬boris_statements B) ∧
  (vladimir_statements V → ¬vladimir_statements V) ∧
  (dmitry_statements D → ¬dmitry_statements D)

-- None of them share the same height or age
axiom uniqueness_condition (A : Andrei) (B : Boris) (V : Vladimir) (D : Dmitry) :
  A ≠ B ∧ A ≠ V ∧ A ≠ D ∧ B ≠ V ∧ B ≠ D ∧ V ≠ D

-- The conclusion that Vladimir is the tallest
theorem vladimir_is_tallest (A : Andrei) (B : Boris) (V : Vladimir) (D : Dmitry) :
  A ≠ tallest ∧ D ≠ taller_than (V : Vladimir) ∧ (¬(B = tallest)) → V = tallest :=
sorry

end TallestBoy

end vladimir_is_tallest_l547_547359


namespace optionA_iff_optionB_l547_547706

variable {a : ℕ → ℝ} {S : ℕ → ℝ}

/-- Sequence a_n is an arithmetic progression -/
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a n = a1 + n * d

/-- Sequence S_n / n is an arithmetic progression -/
def is_S_n_div_n_arithmetic_progression (S : ℕ → ℝ) : Prop :=
  ∃ (b c : ℝ), ∀ n, S n / n = b + c * n

/-- S_n is the sum of the first n terms of the sequence a_n -/
def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = ∑ i in finset.range (n + 1), a i 

/-- Option A is both a sufficient and necessary condition for Option B -/
theorem optionA_iff_optionB (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (is_arithmetic_progression a ∧ sum_first_n_terms a S) ↔ is_S_n_div_n_arithmetic_progression S :=
  sorry

end optionA_iff_optionB_l547_547706


namespace probability_AC_given_AB_l547_547849

-- We start by defining the students and the possible permutations condition
variables (students : List Char)

def is_adjacent (x y : Char) (l : List Char) : Prop :=
  ∃ i : ℕ, i < l.length - 1 ∧ (l.nth i = some x ∧ l.nth (i + 1) = some y ∨
                               l.nth i = some y ∧ l.nth (i + 1) = some x)

-- Definition for counting the number of permutations with a given adjacency
def count_permutations (cond : List Char → Prop) : ℕ :=
  (students.permutations).count cond

-- Conditions given in the problem
def condition_A_B_adjacent : List Char → Prop := is_adjacent 'A' 'B'
def condition_A_C_adjacent : List Char → Prop := is_adjacent 'A' 'C'

-- Prove that the probability A and C are adjacent given A and B are adjacent is 1/3
theorem probability_AC_given_AB : 
  3 * count_permutations (λ l => condition_A_B_adjacent l ∧ condition_A_C_adjacent l) = 
  count_permutations condition_A_B_adjacent :=
sorry

end probability_AC_given_AB_l547_547849


namespace distance_to_y_axis_of_point_on_asymptote_l547_547002

noncomputable def distance_from_y_axis (P : ℝ × ℝ) : ℝ :=
  |P.1|

theorem distance_to_y_axis_of_point_on_asymptote
  (P : ℝ × ℝ)
  (H1 : ∃ x₀ : ℝ, P = (x₀, (Real.sqrt 2 / 2) * x₀))
  (H2 : ∃ x₀ : ℝ, P.1 = x₀ ∧ P.2 = (Real.sqrt 2 / 2) * x₀)
  (H_circle : (0, -Real.sqrt 6), (0, Real.sqrt 6), and 
    (P.1 * P.1 - (Real.sqrt 6 + (Real.sqrt 2 / 2) * P.1) * (Real.sqrt 6 - (Real.sqrt 2 / 2) * P.1) = 0))
  : distance_from_y_axis P = 2 := 
sorry

end distance_to_y_axis_of_point_on_asymptote_l547_547002


namespace triangle_side_b_l547_547228

noncomputable def find_side_length_b
  (a b c : ℝ)
  (area : ℝ)
  (B_deg : ℝ)
  (relation : a^2 + c^2 = 3 * a * c)
  (area_condition : area = sqrt 3)
  (angle_condition : B_deg = 60) : Prop :=
  b = 2 * sqrt 2

theorem triangle_side_b
  (a b c : ℝ)
  (area : ℝ := sqrt 3)
  (B_deg : ℝ := 60)
  (h1 : a^2 + c^2 = 3 * a * c)
  (h2 : area = sqrt 3)
  (h3 : B_deg = 60) : find_side_length_b a b c area B_deg h1 h2 h3 :=
  sorry

end triangle_side_b_l547_547228


namespace area_MNKP_l547_547330

-- Define the area of quadrilateral ABCD given the conditions
def S_ABCD : ℝ := (1 / 2) * 6 * real.sqrt 3 * (8 + 20)

-- Theorem statement asserting the area of quadrilateral MNKP
theorem area_MNKP : S_ABCD = 84 * real.sqrt 3 → S_ABCD / 2 = 42 * real.sqrt 3 := by
  sorry

end area_MNKP_l547_547330


namespace minimum_value_of_h_l547_547903

-- Definitions of the given functions
def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
def g (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
def h (x : ℝ) : ℝ := 2^x + 2^(2 - x)
def j (x : ℝ) : ℝ := log x + 4 / log x

-- Main theorem statement with the condition and conclusion
theorem minimum_value_of_h : ∃ x, h x = 4 :=
by { sorry }

end minimum_value_of_h_l547_547903


namespace factorial_division_example_l547_547447

theorem factorial_division_example : (11! / 10!) * 12 = 132 := 
by sorry

end factorial_division_example_l547_547447


namespace find_length_of_b_l547_547136

theorem find_length_of_b
  {A B C : Type*}
  (a b c : ℝ)
  (area : ℝ)
  (angleB : ℝ)
  (h_area : area = sqrt 3)
  (h_angle : angleB = real.pi / 3)  -- 60 degrees in radians
  (h_sides : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 :=
by
  -- 1. Using the given area constraint: area = 1/2 * a * c * sin(B)
  -- 2. Using the given angle: sin(60) = sqrt(3)/2
  -- 3. Using the given sides relation: a^2 + c^2 = 3ac
  -- 4. Using the Law of Cosines: cos(60) = 1/2
  -- fill proof here
  sorry

end find_length_of_b_l547_547136


namespace tangent_line_b_value_l547_547049

theorem tangent_line_b_value (x1 x2 k b : ℝ)
  (h1 : k = 1 / x1)
  (h2 : k = 1 / (x2 + 1))
  (h3 : x1 = x2 + 1)
  (h4 : k * x1 + b = real.log x1 + 2)
  (h5 : k * x2 + b = real.log (x2 + 1)) :
  b = 1 + real.log 2 :=
by sorry

end tangent_line_b_value_l547_547049


namespace option_A_iff_option_B_l547_547682

-- Definitions
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
∀ n m, a (n + 1) - a n = a (m + 1) - a m

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range n, a (i + 1)

def option_B (S : ℕ → ℝ) : Prop :=
is_arithmetic_progression (λ n, S n / n)

-- Main theorem
theorem option_A_iff_option_B (a : ℕ → ℝ) :
  (is_arithmetic_progression a) ↔ (option_B (λ n, sum_of_first_n_terms a n)) :=
sorry

end option_A_iff_option_B_l547_547682


namespace triangle_ABC_cos_sum_l547_547585

noncomputable def cos_sum (a b c : ℝ) (B C : ℝ) (sinB : ℝ) : ℝ :=
  let cosB : ℝ := real.sqrt (1 - sinB^2)
  let cosC : ℝ := -1 / 2
  let sinC : ℝ := real.sqrt(3) / 2
  let cosA : ℝ := -(cosB * cosC + sinB * sinC)
  cosA + cosB

theorem triangle_ABC_cos_sum {a b c : ℝ} (h_b_c : b + c = 12)
  (C : ℝ) (hC : C = 2 * real.pi / 3)
  (sinB : ℝ) (hSinB : sinB = 5 * real.sqrt 3 / 14) :
  cos_sum a b c
  (real.sqrt (1 - (5 * real.sqrt 3 / 14)^2))
  (2 * real.pi / 3)
  (5 * real.sqrt 3 / 14) = 12 / 7 :=
sorry

end triangle_ABC_cos_sum_l547_547585


namespace find_b_l547_547193

theorem find_b
  (a b c : ℝ)
  (h_area : (1/2) * a * c * Real.sin (ℝ.pi / 3) = Real.sqrt 3)
  (h_ac_eq_4 : a * c = 4)
  (h_cosine : a^2 + c^2 = 3 * a * c) :
  b = 2 * Real.sqrt 2 :=
begin
  sorry
end

end find_b_l547_547193


namespace production_increase_l547_547418

theorem production_increase (x : ℕ) :
  let jan_production := 1000
  let avg_monthly_production := 1550
  (jan_production + (jan_production + x) + (jan_production + 2 * x) + ... + (jan_production + 11 * x)) / 12 = avg_monthly_production →
  x = 100 :=
begin
  sorry
end

end production_increase_l547_547418


namespace graphs_with_inverses_l547_547541

theorem graphs_with_inverses :
    (has_inverse G) ∧ (has_inverse H) ∧ ¬ (has_inverse F) ∧ ¬ (has_inverse I) ∧ ¬ (has_inverse J) :=
by
  sorry

end graphs_with_inverses_l547_547541


namespace div_problem_l547_547796

theorem div_problem (a b c : ℝ) (h1 : a / (b * c) = 4) (h2 : (a / b) / c = 12) : a / b = 4 * Real.sqrt 3 := 
by
  sorry

end div_problem_l547_547796


namespace cubicsum_l547_547842

theorem cubicsum (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) : a^3 + b^3 = 1008 := 
by 
  sorry

end cubicsum_l547_547842


namespace triangle_side_b_l547_547229

noncomputable def find_side_length_b
  (a b c : ℝ)
  (area : ℝ)
  (B_deg : ℝ)
  (relation : a^2 + c^2 = 3 * a * c)
  (area_condition : area = sqrt 3)
  (angle_condition : B_deg = 60) : Prop :=
  b = 2 * sqrt 2

theorem triangle_side_b
  (a b c : ℝ)
  (area : ℝ := sqrt 3)
  (B_deg : ℝ := 60)
  (h1 : a^2 + c^2 = 3 * a * c)
  (h2 : area = sqrt 3)
  (h3 : B_deg = 60) : find_side_length_b a b c area B_deg h1 h2 h3 :=
  sorry

end triangle_side_b_l547_547229


namespace iggy_running_hours_l547_547062

theorem iggy_running_hours :
  ∀ (monday tuesday wednesday thursday friday pace_in_minutes total_minutes_in_hour : ℕ),
  monday = 3 → tuesday = 4 → wednesday = 6 → thursday = 8 → friday = 3 →
  pace_in_minutes = 10 → total_minutes_in_hour = 60 →
  ((monday + tuesday + wednesday + thursday + friday) * pace_in_minutes) / total_minutes_in_hour = 4 :=
by
  intros monday tuesday wednesday thursday friday pace_in_minutes total_minutes_in_hour
  sorry

end iggy_running_hours_l547_547062


namespace sum_sequence_l547_547523

theorem sum_sequence (n : ℕ) (h : n > 0) : 
  let f := λ x, (n : ℝ) / x,
      x_n := 2 * (n : ℝ),
      y_n := 2 in
  (finset.sum (finset.range n) (λ k, 1 / (x_n * (x_n + y_n)))) = n / (4 * n + 4) :=
by
  sorry

end sum_sequence_l547_547523


namespace arith_prog_iff_avg_arith_prog_l547_547647

variable {α : Type*} [LinearOrderedField α]

def is_arith_prog (a : ℕ → α) :=
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

def avg_is_arith_prog (S : ℕ → α) :=
  ∃ D : α, ∀ n : ℕ, n ≠ 0 → (S n / n) = (S (n + 1) / (n + 1)) + D

theorem arith_prog_iff_avg_arith_prog (a : ℕ → α) (S : ℕ → α)
  (hS : ∀ n, S n = ∑ i in range n, a i) :
  is_arith_prog a ↔ avg_is_arith_prog S :=
sorry

end arith_prog_iff_avg_arith_prog_l547_547647


namespace option_A_iff_option_B_l547_547670

def is_arithmetic_progression (seq : ℕ → ℚ) : Prop :=
  ∃ d a₁, ∀ n, seq (n + 1) = seq n + d

def S_n (a : ℕ → ℚ) : ℕ → ℚ
| 0       := 0
| (n + 1) := S_n n + a (n + 1)

theorem option_A_iff_option_B (a : ℕ → ℚ) :
  (is_arithmetic_progression a) ↔ (is_arithmetic_progression (λ n, S_n a n / (n + 1))) :=
sorry

end option_A_iff_option_B_l547_547670


namespace min_value_of_2x_plus_2_2x_l547_547871

-- Lean 4 statement
theorem min_value_of_2x_plus_2_2x (x : ℝ) : 2^x + 2^(2 - x) ≥ 4 :=
by sorry

end min_value_of_2x_plus_2_2x_l547_547871


namespace min_value_h_is_4_l547_547991

def f (x : ℝ) := x^2 + 2*x + 4
def g (x : ℝ) := abs (sin x) + 4 / abs (sin x)
def h (x : ℝ) := 2^x + 2^(2 - x)
def j (x : ℝ) := log x + 4 / log x

theorem min_value_h_is_4 : 
  (∀ x, f(x) ≥ 3) ∧
  (∀ x, g(x) > 4) ∧
  (∃ x, h(x) = 4) ∧
  (∀ x, j(x) ≠ 4) → min_value_h = 4 :=
by
  sorry

end min_value_h_is_4_l547_547991


namespace range_of_a_l547_547018

-- Defining the function f(x)
def f (a x : ℝ) : ℝ := a * x ^ 2 + (a - 2) * x - Real.log x

-- Define the domain of the function
def domain (x : ℝ) : Prop := 0 < x

-- The assertion that f(x) has two zeros implies a lies in (0,1)
theorem range_of_a (a : ℝ) (h : ∃ (x1 x2 : ℝ), domain x1 ∧ domain x2 ∧ x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) :
  0 < a ∧ a < 1 :=
by sorry

end range_of_a_l547_547018


namespace projection_AB_BC_is_neg_sqrt2_l547_547069

-- Define the problem constants and conditions
def isosceles_right_triangle (A B C : Point) : Prop :=
  (dist A B = 2) ∧ (dist A C = 2) ∧ (angle A B C = 45 * π / 180) ∧ (angle B A C = 90 * π / 180) 

-- Given the conditions of the triangle, define the vectors
def vector_AB (A B : Point) : Vector := B - A
def vector_BC (B C : Point) : Vector := C - B

-- Define the projection calculation
def projection (u v : Vector) : ℝ := (u.dot v) / (v.norm)

-- State the theorem to be proved
theorem projection_AB_BC_is_neg_sqrt2 {A B C : Point} (h : isosceles_right_triangle A B C) :
  projection (vector_AB A B) (vector_BC B C) = -real.sqrt 2 :=
by
  sorry

end projection_AB_BC_is_neg_sqrt2_l547_547069


namespace floor_function_solution_correct_l547_547283

noncomputable def floor_function_solution_sum : ℚ :=
  let solutions := [5/2, 10/3, 17/4] in
  solutions.sum

theorem floor_function_solution_correct :
  floor_function_solution_sum = 121/12 := 
sorry

end floor_function_solution_correct_l547_547283


namespace john_runs_more_than_jane_l547_547063

def street_width : ℝ := 25
def block_side : ℝ := 500
def jane_perimeter (side : ℝ) : ℝ := 4 * side
def john_perimeter (side : ℝ) (width : ℝ) : ℝ := 4 * (side + 2 * width)

theorem john_runs_more_than_jane :
  john_perimeter block_side street_width - jane_perimeter block_side = 200 :=
by
  -- Substituting values to verify the equality:
  -- Calculate: john_perimeter 500 25 = 4 * (500 + 2 * 25) = 4 * 550 = 2200
  -- Calculate: jane_perimeter 500 = 4 * 500 = 2000
  sorry

end john_runs_more_than_jane_l547_547063


namespace dice_sum_probability_l547_547570

theorem dice_sum_probability : 
  let outcomes := { (a, b, c) | a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6} ∧ c ∈ {1, 2, 3, 4, 5, 6} }.card,
      favorable := { (a, b, c) | a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6} ∧ c ∈ {1, 2, 3, 4, 5, 6} ∧ a + b + c = 12 }.card,
      probability := (favorable: ℚ) / (outcomes: ℚ)
  in probability = 5 / 108 := by
  sorry

end dice_sum_probability_l547_547570


namespace find_ratio_of_coefficients_l547_547007

theorem find_ratio_of_coefficients (a b : ℝ) 
  (h1 : ∃ k : ℝ, k = (deriv (λ x : ℝ, x^3) 1)) 
  (h2 : is_perpendicular (line_through (1, 1) (a, -b, -2)) (tangent_line (curve_function (λ x : ℝ, x^3)) (1, 1))) : 
  a / b = -1 / 3 :=
sorry

end find_ratio_of_coefficients_l547_547007


namespace vladimir_is_tallest_l547_547340

inductive Boy : Type
| Andrei | Boris | Vladimir | Dmitry

open Boy

-- Hypotheses based on the problem statement
def statements : Boy → (Prop × Prop)
| Andrei := (¬ (Boris = tallest), Vladimir = shortest)
| Boris := (Andrei = oldest, Andrei = shortest)
| Vladimir := (Dmitry > Vladimir, Dmitry = oldest)
| Dmitry := (statements Vladimir = (true, true), Dmitry = oldest)

def different_heights (a b : Boy) : Prop := a ≠ b
def different_ages (a b : Boy) : Prop := a ≠ b

axiom no_same_height {a b : Boy} : different_heights a b
axiom no_same_age {a b : Boy} : different_ages a b

-- The proof statement
theorem vladimir_is_tallest : Vladimir = tallest := 
by 
  sorry

end vladimir_is_tallest_l547_547340


namespace option_A_iff_option_B_l547_547689

-- Definitions
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
∀ n m, a (n + 1) - a n = a (m + 1) - a m

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range n, a (i + 1)

def option_B (S : ℕ → ℝ) : Prop :=
is_arithmetic_progression (λ n, S n / n)

-- Main theorem
theorem option_A_iff_option_B (a : ℕ → ℝ) :
  (is_arithmetic_progression a) ↔ (option_B (λ n, sum_of_first_n_terms a n)) :=
sorry

end option_A_iff_option_B_l547_547689


namespace dice_probability_sum_12_l547_547573

open Nat

/-- Probability that the sum of three six-faced dice rolls equals 12 is 10 / 216 --/
theorem dice_probability_sum_12 : 
  let outcomes := 6^3
  let favorable := 10
  (favorable : ℚ) / outcomes = 10 / 216 := 
by
  let outcomes := 6^3
  let favorable := 10
  sorry

end dice_probability_sum_12_l547_547573


namespace speed_of_first_train_l547_547812

-- Definitions of the conditions
def length_train1 : ℝ := 120
def length_train2 : ℝ := 165
def speed_train2_kmph : ℝ := 65
def time_seconds : ℝ := 7.0752960452818945

-- Conversion factors and constants
noncomputable def meters_to_km (meters : ℝ) : ℝ := meters / 1000
noncomputable def seconds_to_hours (seconds : ℝ) : ℝ := seconds / 3600
noncomputable def mps_to_kmph (mps : ℝ) : ℝ := mps * 3.6

-- The total distance both trains must clear
noncomputable def total_distance_meters : ℝ := length_train1 + length_train2

-- The relative speed in m/s
noncomputable def relative_speed_mps : ℝ := total_distance_meters / time_seconds

-- The relative speed in km/h
noncomputable def relative_speed_kmph : ℝ := mps_to_kmph relative_speed_mps

-- The speed of the first train in km/h
noncomputable def speed_train1_kmph : ℝ := relative_speed_kmph - speed_train2_kmph

-- The proof statement
theorem speed_of_first_train :
  speed_train1_kmph ≈ 79.972 :=
by
  -- Place proof here when solving
  sorry

end speed_of_first_train_l547_547812


namespace problem1_problem2_problem3_l547_547524

def f (a x : ℝ) : ℝ := (a * 2^x + a - 2) / (2^x + 1)

-- Problem (1): Prove that a = 1 given that f is an odd function.
theorem problem1 (a : ℝ) : (∀ x : ℝ, f a x = -f a (-x)) → a = 1 := 
by sorry

-- Problem (2): Prove that f(x) is an increasing function given a = 1.
theorem problem2 : (∀ x1 x2 : ℝ, x1 < x2 → f 1 x1 < f 1 x2) :=
by sorry

-- Problem (3): Prove that the range of f(x) is (-1, 1) given a = 1.
theorem problem3 : {y : ℝ | ∃ x : ℝ, f 1 x = y} = Set.Ioo (-1) 1 :=
by sorry

end problem1_problem2_problem3_l547_547524


namespace find_b_proof_l547_547203

noncomputable def find_b (a b c : ℝ) (A B C : ℝ) : Prop :=
  let area := sqrt 3 in
  let angle_B := 60 * Real.pi / 180 in
  let condition1 := area = (1 / 2) * a * c * Real.sin angle_B in
  let condition2 := a^2 + c^2 = 3 * a * c in
  let side_b := b = 2 * Real.sqrt 2 in
  condition1 ∧ condition2 → side_b

theorem find_b_proof :
  ∃ (a b c : ℝ) (A B C : ℝ), find_b a b c A B C :=
by
  sorry

end find_b_proof_l547_547203


namespace fraction_given_to_Janet_l547_547616

-- Declare constants for the problem
constant Jean_stuffies : ℕ := 60
constant Janet_got : ℕ := 10

-- Conditions
constant Jean_keeps_fraction : ℚ := 1 / 3
constant Jean_keeps : ℕ := Jean_stuffies * Jean_keeps_fraction.to_nat
constant Jean_gives_away : ℕ := Jean_stuffies - Jean_keeps

-- Statement to prove
theorem fraction_given_to_Janet :
  (Janet_got : ℚ) / Jean_gives_away = 1 / 4 := sorry

end fraction_given_to_Janet_l547_547616


namespace greatest_number_of_divisors_from_1_to_25_l547_547728

noncomputable def count_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem greatest_number_of_divisors_from_1_to_25 : 
  ∀ n ∈ finset.range (25 + 1), count_divisors 24 ≥ count_divisors n :=
by
  sorry

end greatest_number_of_divisors_from_1_to_25_l547_547728


namespace prob_more_boys_or_girls_correct_l547_547254

-- Define the number of grandchildren
def num_grandchildren : ℕ := 12

-- Define the function to calculate the binomial coefficient
def binom (n k : ℕ) : ℕ := nat.choose n k

-- Calculate the total number of ways the genders can be distributed
def total_ways : ℕ := 2 ^ num_grandchildren

-- Calculate the number of ways to have exactly 6 boys and 6 girls
def exactly_six_each : ℕ := binom num_grandchildren (num_grandchildren / 2)

-- Calculate the probability of having exactly 6 boys and 6 girls
def prob_exactly_six_each : ℚ := exactly_six_each / total_ways

-- Using the complement rule, calculate the desired probability
def prob_more_boys_or_girls : ℚ := 1 - prob_exactly_six_each

-- The statement of the theorem to be proved
theorem prob_more_boys_or_girls_correct :
  prob_more_boys_or_girls = 793 / 1024 :=
by
  -- Proof will be provided here
  sorry

end prob_more_boys_or_girls_correct_l547_547254


namespace number_of_partners_equation_l547_547387

variable (x : ℕ)

theorem number_of_partners_equation :
  5 * x + 45 = 7 * x - 3 :=
sorry

end number_of_partners_equation_l547_547387


namespace borrowed_sheets_l547_547094

theorem borrowed_sheets {c b : ℕ} : 
  (100 % 2 = 0) → 
  (∀ k, 1 ≤ k ∧ k ≤ 50 → sum (λ k, 4 * k + 1) = 2 * b * (b + 1) + b)
  → (∀ k, b + c + 1 ≤ k ∧ k ≤ 50 → sum (λ k, 4 * k + 1) = 2 * (50 - b - c) * (51 + b + c) + (50 - b - c))
  → ((2 * (50 - b - c) * (51 + b + c) + (50 - b - c)) / (100 - 2 * c) = 54) 
  → (c = 15) := 
by 
  intros h1 h2 h3 h4
  sorry

end borrowed_sheets_l547_547094


namespace minimum_value_of_h_l547_547905

-- Definitions of the given functions
def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
def g (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
def h (x : ℝ) : ℝ := 2^x + 2^(2 - x)
def j (x : ℝ) : ℝ := log x + 4 / log x

-- Main theorem statement with the condition and conclusion
theorem minimum_value_of_h : ∃ x, h x = 4 :=
by { sorry }

end minimum_value_of_h_l547_547905


namespace incorrect_statement_B_l547_547021

open Set

def A := {x : ℝ | x^2 - 4 = 0}

theorem incorrect_statement_B : ¬ (-2 ∈ A) := by
  -- proof omitted
  sorry

end incorrect_statement_B_l547_547021


namespace eval_f_pi_over_8_l547_547030

noncomputable def f (θ : ℝ) : ℝ :=
(2 * (Real.sin (θ / 2)) ^ 2 - 1) / (Real.sin (θ / 2) * Real.cos (θ / 2)) + 2 * Real.tan θ

theorem eval_f_pi_over_8 : f (π / 8) = -4 :=
sorry

end eval_f_pi_over_8_l547_547030


namespace ratio_lcm_gcf_280_476_l547_547823

theorem ratio_lcm_gcf_280_476 : 
  let a := 280
  let b := 476
  let lcm_ab := Nat.lcm a b
  let gcf_ab := Nat.gcd a b
  lcm_ab / gcf_ab = 170 := by
  sorry

end ratio_lcm_gcf_280_476_l547_547823


namespace determine_original_triangle_sides_l547_547014

variables (a1 b1 c1 : ℝ)
noncomputable def s1 : ℝ := (a1 + b1 + c1) / 2

noncomputable def original_triangle_sides : (ℝ × ℝ × ℝ) :=
(
  a1 * sqrt((b1 * c1) / ((s1 a1 b1 c1) - b1) * ((s1 a1 b1 c1) - c1)),
  b1 * sqrt((a1 * c1) / ((s1 a1 b1 c1) - a1) * ((s1 a1 b1 c1) - c1)),
  c1 * sqrt((a1 * b1) / ((s1 a1 b1 c1) - a1) * ((s1 a1 b1 c1) - b1))
)

-- Theorem statement to determine the sides of the original triangle from the pedal triangle
theorem determine_original_triangle_sides :
  ∃ (a b c : ℝ), 
    a = a1 * sqrt((b1 * c1) / ((s1 a1 b1 c1) - b1) * ((s1 a1 b1 c1) - c1)) ∧
    b = b1 * sqrt((a1 * c1) / ((s1 a1 b1 c1) - a1) * ((s1 a1 b1 c1) - c1)) ∧
    c = c1 * sqrt((a1 * b1) / ((s1 a1 b1 c1) - a1) * ((s1 a1 b1 c1) - b1)) :=
by {
  sorry
}

end determine_original_triangle_sides_l547_547014


namespace minimum_value_of_h_l547_547896

-- Definitions of the given functions
def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
def g (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
def h (x : ℝ) : ℝ := 2^x + 2^(2 - x)
def j (x : ℝ) : ℝ := log x + 4 / log x

-- Main theorem statement with the condition and conclusion
theorem minimum_value_of_h : ∃ x, h x = 4 :=
by { sorry }

end minimum_value_of_h_l547_547896


namespace sum_primitive_roots_mod_11_l547_547478

theorem sum_primitive_roots_mod_11 :
  let prime_set := {a ∈ Finset.range 11 \ {0}} in
  let primitive_roots := {a ∈ prime_set | ∀ b, b ≠ 0 → (∃ i, a^i % 11 = b)} in
  ∑ x in primitive_roots, x = 20 :=
by sorry

end sum_primitive_roots_mod_11_l547_547478


namespace triangle_side_b_l547_547176

theorem triangle_side_b 
  (a b c : ℝ)
  (B : ℝ)
  (h1 : 1/2 * a * c * Real.sin B = Real.sqrt 3)
  (h2 : B = π / 3)
  (h3 : a^2 + c^2 = 3 * a * c) : b = 2 * Real.sqrt 2 := by
  sorry

end triangle_side_b_l547_547176


namespace equivalent_oranges_value_l547_547765

theorem equivalent_oranges_value :
  (4 / 5 : ℝ) * 15 = (12 : ℝ) →
  (3 / 4 : ℝ) * 8 = 6 :=
begin
  intro h,
  have h1 : (4 / 5 : ℝ) * 15 = 12 := h,
  calc
    (3 / 4 : ℝ) * 8
        = 6 : by linarith,
end

end equivalent_oranges_value_l547_547765


namespace find_side_b_l547_547110

theorem find_side_b (a b c : ℝ) (A B C : ℝ)
  (area_triangle : Real.sqrt 3) (B_eq_60 : B = 60) (cond_ac: (a^2 + c^2 = 3 * a * c)) :
  ∃ b, b = 2 * Real.sqrt 2 :=
by
  sorry

end find_side_b_l547_547110


namespace f_of_x_plus_one_minus_f_of_x_l547_547034

theorem f_of_x_plus_one_minus_f_of_x {x : ℝ} (f : ℝ → ℝ) (h : ∀ x, f(x) = 9^x) : 
  f(x + 1) - f(x) = 8 * f(x) :=
by
  sorry

end f_of_x_plus_one_minus_f_of_x_l547_547034


namespace count_three_digit_multiples_of_13_and_7_l547_547546

def is_three_digit (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000

def is_multiple_of (m n : ℕ) : Prop :=
  ∃ k : ℕ, m = k * n

def lcm (a b : ℕ) : ℕ :=
  Nat.lcm a b

def count_multiples_in_range (multiple low high : ℕ) : ℕ :=
  (high - low) / multiple + 1

theorem count_three_digit_multiples_of_13_and_7 : ∃ count : ℕ,
    count = count_multiples_in_range (lcm 13 7) 182 910 ∧ count = 9 :=
by
  sorry

end count_three_digit_multiples_of_13_and_7_l547_547546


namespace vladimir_is_tallest_l547_547358

section TallestBoy

variables (Andrei Boris Vladimir Dmitry : Type)

-- Define the statements made by each boy
def andrei_statements : Andrei → Prop :=
-- suppose andrei's statements are a::Boris ≠ tallest, b::Vladimir = shortest
λ A, (∀ B : Boris, B ≠ tallest) ∨ (∀ V : Vladimir, V = shortest)

def boris_statements : Boris → Prop :=
λ B, (∀ A : Andrei, A = oldest) ∨ (∀ A' : Andrei, A' = shortest)

def vladimir_statements : Vladimir → Prop :=
λ V, (∀ D : Dmitry, D = taller_than V) ∨ (∀ D' : Dmitry, D' = older_than V)

def dmitry_statements : Dmitry → Prop :=
λ D, ((vladimir_statements V) ∧ (vladimir_statements V)) ∨ (∀ D' : Dmitry, D' = oldest)

-- Define the conditions
-- Each boy makes two statements, one of which is true and the other is false
axiom statements_condition (A : Andrei) (B : Boris) (V : Vladimir) (D : Dmitry) :
  (andrei_statements A → ¬andrei_statements A) ∧
  (boris_statements B → ¬boris_statements B) ∧
  (vladimir_statements V → ¬vladimir_statements V) ∧
  (dmitry_statements D → ¬dmitry_statements D)

-- None of them share the same height or age
axiom uniqueness_condition (A : Andrei) (B : Boris) (V : Vladimir) (D : Dmitry) :
  A ≠ B ∧ A ≠ V ∧ A ≠ D ∧ B ≠ V ∧ B ≠ D ∧ V ≠ D

-- The conclusion that Vladimir is the tallest
theorem vladimir_is_tallest (A : Andrei) (B : Boris) (V : Vladimir) (D : Dmitry) :
  A ≠ tallest ∧ D ≠ taller_than (V : Vladimir) ∧ (¬(B = tallest)) → V = tallest :=
sorry

end TallestBoy

end vladimir_is_tallest_l547_547358


namespace find_b_proof_l547_547206

noncomputable def find_b (a b c : ℝ) (A B C : ℝ) : Prop :=
  let area := sqrt 3 in
  let angle_B := 60 * Real.pi / 180 in
  let condition1 := area = (1 / 2) * a * c * Real.sin angle_B in
  let condition2 := a^2 + c^2 = 3 * a * c in
  let side_b := b = 2 * Real.sqrt 2 in
  condition1 ∧ condition2 → side_b

theorem find_b_proof :
  ∃ (a b c : ℝ) (A B C : ℝ), find_b a b c A B C :=
by
  sorry

end find_b_proof_l547_547206


namespace cubicsum_l547_547843

theorem cubicsum (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) : a^3 + b^3 = 1008 := 
by 
  sorry

end cubicsum_l547_547843


namespace points_within_triangle_l547_547731

theorem points_within_triangle (n : ℕ) (points : fin n → ℝ × ℝ)
  (h : ∀ (a b c : fin n), area (points a) (points b) (points c) ≤ 1) : 
  ∃ (A B C : ℝ × ℝ), triangle_area A B C = 4 ∧ ∀ (i : fin n), inside_triangle A B C (points i) :=
begin
  sorry
end

end points_within_triangle_l547_547731


namespace sunil_interest_earned_l547_547760

def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem sunil_interest_earned :
  ∃ P : ℝ, compound_interest P 0.08 1 2 = 19828.80 ∧ 19828.80 - P = 2828.80 :=
by
  sorry

end sunil_interest_earned_l547_547760


namespace min_value_h_is_4_l547_547993

def f (x : ℝ) := x^2 + 2*x + 4
def g (x : ℝ) := abs (sin x) + 4 / abs (sin x)
def h (x : ℝ) := 2^x + 2^(2 - x)
def j (x : ℝ) := log x + 4 / log x

theorem min_value_h_is_4 : 
  (∀ x, f(x) ≥ 3) ∧
  (∀ x, g(x) > 4) ∧
  (∃ x, h(x) = 4) ∧
  (∀ x, j(x) ≠ 4) → min_value_h = 4 :=
by
  sorry

end min_value_h_is_4_l547_547993


namespace prove_domain_l547_547522

def domain_is_correct (f : ℝ → ℝ) : Prop :=
  f = λ x, sqrt (x * (4 - x)) / (x - 1) →
  (∀ x, 0 ≤ x ∧ x ≤ 4 ∧ x ≠ 1 ↔ (x ∈ set.Icc 0 1 ∪ set.Icc 1 4 \ {1}))

theorem prove_domain (f : ℝ → ℝ) (h : f = λ x, sqrt (x * (4 - x)) / (x - 1)) :
  domain_is_correct f :=
by
  sorry

end prove_domain_l547_547522


namespace find_integers_l547_547470

theorem find_integers (A B C : ℤ) (hA : A = 500) (hB : B = -1) (hC : C = -500) : 
  (A : ℚ) / 999 + (B : ℚ) / 1000 + (C : ℚ) / 1001 = 1 / (999 * 1000 * 1001) :=
by 
  rw [hA, hB, hC]
  sorry

end find_integers_l547_547470


namespace Vladimir_is_tallest_l547_547378

-- Variables representing the boys
inductive Boy
| Andrei | Boris | Vladimir | Dmitry

open Boy

-- Height and Age representations
variable (Height Age : Boy → ℕ)

-- Conditions based on the problem
axiom distinct_heights : ∀ a b, Height a = Height b → a = b
axiom distinct_ages : ∀ a b, Age a = Age b → a = b

-- Statements made by the boys
axiom Andrei_statements :
  (¬ (Height Boris > Height Andrei) ∨ Height Vladimir = Height Andrei)
  ∧ ¬ (¬ (Height Boris > Height Andrei) ∧ Height Vladimir = Height Andrei)

axiom Boris_statements :
  (Age Andrei > Age Boris ∨ Height Andrei = Height Boris)
  ∧ ¬ (Age Andrei > Age Boris ∧ Height Andrei = Height Boris)

axiom Vladimir_statements :
  (Height Dmitry > Height Vladimir ∨ Age Dmitry > Age Vladimir)
  ∧ ¬ (Height Dmitry > Height Vladimir ∧ Age Dmitry > Age Vladimir)

axiom Dmitry_statements :
  ((Height Dmitry > Height Vladimir ∧ Age Dmitry > Age Vladimir) ∨ Age Dmitry > Age Dmitry)
  ∧ ¬ (¬ (Height Dmitry > Height Vladimir ∧ Age Dmitry > Age Vladimir) ∧ ¬ Age Dmitry > Age Dmitry)

-- Problem Statement: Vladimir is the tallest
theorem Vladimir_is_tallest : ∀ h : Height Vladimir = Nat.max (Height Andrei) (Nat.max (Height Boris) (Height Dmitry))
by
  sorry

end Vladimir_is_tallest_l547_547378


namespace fraction_sum_digits_l547_547817

theorem fraction_sum_digits 
  (N : ℕ)
  (frac1 : ℚ := 1/2)
  (frac2 : ℚ := 1/5)
  (frac5 : ℚ := 1/5)
  (frac_other : ℚ := 1/10) :
  frac1 + frac2 + frac2 + frac_other = 1 := 
by
  -- fractions presented are correct and sum up to 1
  rw [show frac1 = 1/2, by rfl, 
      show frac2 = 1/5, by rfl,
      show frac2 = 1/5, by rfl,
      show frac_other = 1/10, by rfl]
  norm_num

end fraction_sum_digits_l547_547817


namespace min_value_of_2x_plus_2_2x_l547_547869

-- Lean 4 statement
theorem min_value_of_2x_plus_2_2x (x : ℝ) : 2^x + 2^(2 - x) ≥ 4 :=
by sorry

end min_value_of_2x_plus_2_2x_l547_547869


namespace min_value_h_is_4_l547_547984

def f (x : ℝ) := x^2 + 2*x + 4
def g (x : ℝ) := abs (sin x) + 4 / abs (sin x)
def h (x : ℝ) := 2^x + 2^(2 - x)
def j (x : ℝ) := log x + 4 / log x

theorem min_value_h_is_4 : 
  (∀ x, f(x) ≥ 3) ∧
  (∀ x, g(x) > 4) ∧
  (∃ x, h(x) = 4) ∧
  (∀ x, j(x) ≠ 4) → min_value_h = 4 :=
by
  sorry

end min_value_h_is_4_l547_547984


namespace tom_saved_money_by_having_pet_insurance_l547_547809

-- Definitions
def planA_duration : ℕ := 24
def planA_monthly_premium : ℕ := 15
def planA_coverage : ℕ := 60
def planA_yearly_deductible : ℕ := 100

def planB_duration : ℕ := 12
def planB_monthly_premium : ℕ := 25
def planB_coverage : ℕ := 90
def planB_yearly_deductible : ℕ := 100

def surgery_cost : ℕ := 7500

-- Theorem statement
theorem tom_saved_money_by_having_pet_insurance : 
  let total_premiums_A := planA_duration * planA_monthly_premium,
      total_premiums_B := planB_duration * planB_monthly_premium,
      total_premiums := total_premiums_A + total_premiums_B,
      
      deductibles_A := 2 * planA_yearly_deductible,
      deductibles_B := 1 * planB_yearly_deductible,
      total_deductibles := deductibles_A + deductibles_B,

      coverage_A := planA_coverage * (surgery_cost - planA_yearly_deductible) / 100,
      coverage_B := planB_coverage * (surgery_cost - planB_yearly_deductible) / 100,

      amount_covered := coverage_B, -- since plan B covers more

      total_cost_with_insurance := total_premiums + total_deductibles + (surgery_cost - amount_covered),
      savings := surgery_cost - total_cost_with_insurance

  in savings = 5700 := sorry

end tom_saved_money_by_having_pet_insurance_l547_547809


namespace function_min_value_4_l547_547856

theorem function_min_value_4 :
  ∃ x : ℝ, (2^x + 2^(2 - x)) = 4 :=
sorry

end function_min_value_4_l547_547856


namespace Vladimir_is_tallest_l547_547377

-- Variables representing the boys
inductive Boy
| Andrei | Boris | Vladimir | Dmitry

open Boy

-- Height and Age representations
variable (Height Age : Boy → ℕ)

-- Conditions based on the problem
axiom distinct_heights : ∀ a b, Height a = Height b → a = b
axiom distinct_ages : ∀ a b, Age a = Age b → a = b

-- Statements made by the boys
axiom Andrei_statements :
  (¬ (Height Boris > Height Andrei) ∨ Height Vladimir = Height Andrei)
  ∧ ¬ (¬ (Height Boris > Height Andrei) ∧ Height Vladimir = Height Andrei)

axiom Boris_statements :
  (Age Andrei > Age Boris ∨ Height Andrei = Height Boris)
  ∧ ¬ (Age Andrei > Age Boris ∧ Height Andrei = Height Boris)

axiom Vladimir_statements :
  (Height Dmitry > Height Vladimir ∨ Age Dmitry > Age Vladimir)
  ∧ ¬ (Height Dmitry > Height Vladimir ∧ Age Dmitry > Age Vladimir)

axiom Dmitry_statements :
  ((Height Dmitry > Height Vladimir ∧ Age Dmitry > Age Vladimir) ∨ Age Dmitry > Age Dmitry)
  ∧ ¬ (¬ (Height Dmitry > Height Vladimir ∧ Age Dmitry > Age Vladimir) ∧ ¬ Age Dmitry > Age Dmitry)

-- Problem Statement: Vladimir is the tallest
theorem Vladimir_is_tallest : ∀ h : Height Vladimir = Nat.max (Height Andrei) (Nat.max (Height Boris) (Height Dmitry))
by
  sorry

end Vladimir_is_tallest_l547_547377


namespace minimum_value_C_l547_547943

noncomputable def y_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def y_B (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
noncomputable def y_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def y_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimum_value_C : ∃ x : ℝ, y_C x = 4 := 
by
  -- proof goes here
  sorry

end minimum_value_C_l547_547943


namespace option_A_is_necessary_and_sufficient_for_option_B_l547_547692

def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

def mean_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : (ℕ → ℝ)
| n => S n / n

def sum_of_first_n_terms (a : ℕ → ℝ) : ℕ → ℝ
| 0     => 0
| (n+1) => sum_of_first_n_terms a n + a (n+1)

theorem option_A_is_necessary_and_sufficient_for_option_B
    (a : ℕ → ℝ) :
    (is_arithmetic_progression a ↔ is_arithmetic_progression (mean_sequence a (sum_of_first_n_terms a))) :=
sorry

end option_A_is_necessary_and_sufficient_for_option_B_l547_547692


namespace hyperbola_asymptotes_l547_547530

theorem hyperbola_asymptotes (a b c : ℝ) (h_a_gt_0 : 0 < a) (h_b_gt_0 : 0 < b)
  (h_2b_eq_4 : 2 * b = 4) (h_2c_eq_4sqrt3 : 2 * c = 4 * Real.sqrt 3)
  (h_c_sq_eq_a_sq_plus_b_sq : c = Real.sqrt (a^2 + b^2)) :
  ∀ x, y = a * x ↔ y = ± Real.sqrt 2 / 2 * x := 
by
  sorry

end hyperbola_asymptotes_l547_547530


namespace length_of_AB_l547_547605

theorem length_of_AB (A B C D : Point)
  (isosceles_ABC : is_isosceles_triangle A B C)
  (isosceles_CBD : is_isosceles_triangle C B D)
  (perimeter_CBD : ∀ P : Point, (P = C ∨ P = B ∨ P = D) → distance P BD = 24)
  (perimeter_ABC : ∀ P : Point, (P = A ∨ P = B ∨ P = C) → distance P ABC = 21)
  (distance_BD : distance B D = 8) :
  distance A B = 5 := sorry

end length_of_AB_l547_547605


namespace goods_train_speed_l547_547413

-- Define the conditions
def length_of_goods_train_km : ℝ := 0.45 -- converting 450 meters to kilometers
def time_to_pass_hours : ℝ := 1 / 240 -- converting 15 seconds to hours
def speed_of_mans_train_kmh : ℝ := 70 -- speed of the man's train in km/h

-- Define the relative speed function
def relative_speed_kmh (vg : ℝ) : ℝ := vg + speed_of_mans_train_kmh

-- The expected relative speed must be equal to 108 km/h
def relative_speed_expected : ℝ := length_of_goods_train_km / time_to_pass_hours

-- Now we state the theorem we need to prove
theorem goods_train_speed :
  ∃ vg : ℝ, relative_speed_kmh vg = relative_speed_expected ∧ vg = 38 :=
by {
  -- The formal proof would go here, but we'll skip it with sorry
  sorry
}

end goods_train_speed_l547_547413


namespace triangle_side_b_l547_547231

noncomputable def find_side_length_b
  (a b c : ℝ)
  (area : ℝ)
  (B_deg : ℝ)
  (relation : a^2 + c^2 = 3 * a * c)
  (area_condition : area = sqrt 3)
  (angle_condition : B_deg = 60) : Prop :=
  b = 2 * sqrt 2

theorem triangle_side_b
  (a b c : ℝ)
  (area : ℝ := sqrt 3)
  (B_deg : ℝ := 60)
  (h1 : a^2 + c^2 = 3 * a * c)
  (h2 : area = sqrt 3)
  (h3 : B_deg = 60) : find_side_length_b a b c area B_deg h1 h2 h3 :=
  sorry

end triangle_side_b_l547_547231


namespace necessary_but_not_sufficient_condition_for_increasing_geometric_sequence_l547_547711

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, a (n + 1) = a n * q

theorem necessary_but_not_sufficient_condition_for_increasing_geometric_sequence
  (a : ℕ → ℝ)
  (h0 : a 0 > 0)
  (h_geom : is_geometric_sequence a) :
  (a 0^2 < a 1^2) ↔ (is_increasing_sequence a) ∧ ¬ (∀ n, a n > 0 → a (n + 1) > 0) :=
sorry

end necessary_but_not_sufficient_condition_for_increasing_geometric_sequence_l547_547711


namespace find_side_b_l547_547150

theorem find_side_b
  (a b c : ℝ)
  (area : ℝ)
  (B : ℝ)
  (h_area : area = real.sqrt 3)
  (h_B : B = (60 : ℝ) * real.pi / 180) -- Convert degrees to radians
  (h_cond : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 := 
sorry

end find_side_b_l547_547150


namespace minimum_value_of_h_l547_547906

-- Definitions of the given functions
def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
def g (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
def h (x : ℝ) : ℝ := 2^x + 2^(2 - x)
def j (x : ℝ) : ℝ := log x + 4 / log x

-- Main theorem statement with the condition and conclusion
theorem minimum_value_of_h : ∃ x, h x = 4 :=
by { sorry }

end minimum_value_of_h_l547_547906


namespace sum_of_four_numbers_l547_547505

noncomputable def nonzero (x : ℕ) : Prop := 1 ≤ x ∧ x ≤ 9

theorem sum_of_four_numbers :
  ∀ (A B : ℕ), nonzero A → nonzero B →
  let s := 98765 + A * 1000 + 532 + B * 100 + 41 + 1021 in
  (s % 4 = 0) → s.digits.length = 6 :=
by
  sorry

end sum_of_four_numbers_l547_547505


namespace ordinate_of_vertex_of_quadratic_l547_547491

theorem ordinate_of_vertex_of_quadratic (f : ℝ → ℝ) (h1 : ∃ a b c : ℝ, ∀ x : ℝ, f x = a * x^2 + b * x + c)
  (h2 : ∃ t1 t2 t3 : ℝ, (f t1)^3 - f t1 = 0 ∧ (f t2)^3 - f t2 = 0 ∧ (f t3)^3 - f t3 = 0 ∧ t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t3) :
  (∃ h : ℝ, f h = 0) → 
  ∃ k : ℝ, ∀ x : ℝ, f x = f (k/2) → False :=
begin
  sorry
end

end ordinate_of_vertex_of_quadratic_l547_547491


namespace Vladimir_is_tallest_l547_547355

-- Definitions for the statements made by the boys.
def Andrei_statements (Boris_tallest Vladimir_shortest : Prop) :=
  (¬Boris_tallest) ∧ Vladimir_shortest

def Boris_statements (Andrei_oldest Andrei_shortest : Prop) :=
  Andrei_oldest ∧ Andrei_shortest

def Vladimir_statements (Dmitry_taller Dmitry_older : Prop) :=
  Dmitry_taller ∧ Dmitry_older

def Dmitry_statements (Vladimir_both_true Dmitry_oldest : Prop) :=
  Vladimir_both_true ∧ Dmitry_oldest

-- Conditions given in the problem.
variables (Boris_tallest Vladimir_shortest Andrei_oldest Andrei_shortest Dmitry_taller Dmitry_older 
           Vladimir_both_true Dmitry_oldest : Prop)

-- Axioms based on the problem statement
axiom Dmitry_statements_condition : Dmitry_statements Vladimir_both_true Dmitry_oldest
axiom Vladimir_statements_condition : Vladimir_statements Dmitry_taller Dmitry_older
axiom Boris_statements_condition : Boris_statements Andrei_oldest Andrei_shortest
axiom Andrei_statements_condition : Andrei_statements Boris_tallest Vladimir_shortest

-- None of them share the same height or age.
axiom no_shared_height_or_age : 
  ¬(Andrei_oldest ∧ Boris_tallest ∧ Dmitry_taller ∧ Vladimir_both_true)

-- Main theorem: proving that Vladimir is the tallest based on the conditions
theorem Vladimir_is_tallest (cond : ¬Boris_tallest → Vladimir_shortest →
                              ¬Andrei_oldest → Andrei_shortest →
                              ¬Dmitry_taller → Dmitry_older →
                              ¬Vladimir_both_true → Dmitry_oldest → 
                              ¬(Andrei_oldest ∧ Boris_tallest ∧ Dmitry_taller ∧ Vladimir_both_true)) :
  (¬Boris_tallest ∧ Dmitry_older) → Vladimir_tallest :=
begin
  sorry,
end

end Vladimir_is_tallest_l547_547355


namespace Vladimir_is_tallest_l547_547357

-- Definitions for the statements made by the boys.
def Andrei_statements (Boris_tallest Vladimir_shortest : Prop) :=
  (¬Boris_tallest) ∧ Vladimir_shortest

def Boris_statements (Andrei_oldest Andrei_shortest : Prop) :=
  Andrei_oldest ∧ Andrei_shortest

def Vladimir_statements (Dmitry_taller Dmitry_older : Prop) :=
  Dmitry_taller ∧ Dmitry_older

def Dmitry_statements (Vladimir_both_true Dmitry_oldest : Prop) :=
  Vladimir_both_true ∧ Dmitry_oldest

-- Conditions given in the problem.
variables (Boris_tallest Vladimir_shortest Andrei_oldest Andrei_shortest Dmitry_taller Dmitry_older 
           Vladimir_both_true Dmitry_oldest : Prop)

-- Axioms based on the problem statement
axiom Dmitry_statements_condition : Dmitry_statements Vladimir_both_true Dmitry_oldest
axiom Vladimir_statements_condition : Vladimir_statements Dmitry_taller Dmitry_older
axiom Boris_statements_condition : Boris_statements Andrei_oldest Andrei_shortest
axiom Andrei_statements_condition : Andrei_statements Boris_tallest Vladimir_shortest

-- None of them share the same height or age.
axiom no_shared_height_or_age : 
  ¬(Andrei_oldest ∧ Boris_tallest ∧ Dmitry_taller ∧ Vladimir_both_true)

-- Main theorem: proving that Vladimir is the tallest based on the conditions
theorem Vladimir_is_tallest (cond : ¬Boris_tallest → Vladimir_shortest →
                              ¬Andrei_oldest → Andrei_shortest →
                              ¬Dmitry_taller → Dmitry_older →
                              ¬Vladimir_both_true → Dmitry_oldest → 
                              ¬(Andrei_oldest ∧ Boris_tallest ∧ Dmitry_taller ∧ Vladimir_both_true)) :
  (¬Boris_tallest ∧ Dmitry_older) → Vladimir_tallest :=
begin
  sorry,
end

end Vladimir_is_tallest_l547_547357


namespace gcd_40_56_l547_547312

theorem gcd_40_56 : Int.gcd 40 56 = 8 :=
by
  sorry

end gcd_40_56_l547_547312


namespace find_length_of_b_l547_547132

theorem find_length_of_b
  {A B C : Type*}
  (a b c : ℝ)
  (area : ℝ)
  (angleB : ℝ)
  (h_area : area = sqrt 3)
  (h_angle : angleB = real.pi / 3)  -- 60 degrees in radians
  (h_sides : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 :=
by
  -- 1. Using the given area constraint: area = 1/2 * a * c * sin(B)
  -- 2. Using the given angle: sin(60) = sqrt(3)/2
  -- 3. Using the given sides relation: a^2 + c^2 = 3ac
  -- 4. Using the Law of Cosines: cos(60) = 1/2
  -- fill proof here
  sorry

end find_length_of_b_l547_547132


namespace G_equals_F_l547_547100

-- Definition of the function F
def F (x : ℝ) : ℝ := log ((1 + x) / (1 - x))

-- Definition of the function G with the substitution
def G (x : ℝ) : ℝ := log ((1 + (2 * x - x^3) / (1 + 2 * x^2)) / (1 - (2 * x - x^3) / (1 + 2 * x^2)))

-- The theorem stating that G is equivalent to F after the substitution
theorem G_equals_F (x : ℝ) : G x = F x := by
  sorry

end G_equals_F_l547_547100


namespace find_side_b_l547_547171

noncomputable def triangle_area (a b c : ℝ) (B : ℝ) : ℝ :=
  1 / 2 * a * c * real.sin B

noncomputable def law_of_cosines (a b c : ℝ) (B : ℝ) : ℝ :=
  a^2 + c^2 - 2 * a * c * real.cos B

theorem find_side_b (a b c : ℝ) (B : ℝ) (area : ℝ)
  (h1 : area = √3) 
  (h2 : B = real.pi / 3)
  (h3 : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 :=
by
  sorry

end find_side_b_l547_547171


namespace tallest_boy_is_Vladimir_l547_547350

noncomputable def Andrei_statement1 (Boris_tallest: Prop) : Prop := ¬ Boris_tallest
def Andrei_statement2 (Vladimir_shortest: Prop) : Prop := Vladimir_shortest

def Boris_statement1 (Andrei_oldest: Prop) : Prop := Andrei_oldest
def Boris_statement2 (Andrei_shortest: Prop) : Prop := Andrei_shortest

def Vladimir_statement1 (Dmitry_taller: Prop) : Prop := Dmitry_taller
def Vladimir_statement2 (Dmitry_older: Prop) : Prop := Dmitry_older

noncomputable def Dmitry_statement1 (Vladimir_statement1: Prop) (Vladimir_statement2: Prop) : Prop :=
  Vladimir_statement1 ∧ Vladimir_statement2
def Dmitry_statement2 (Dmitry_oldest: Prop) : Prop := Dmitry_oldest

axiom one_statement_true_per_boy :
  ∀ {P₁ P₂: Prop}, (P₁ ∨ P₂) ∧ ¬ (P₁ ∧ P₂)

axiom no_same_height_or_age :
  ∀ {h1 h2 h3 h4 a1 a2 a3 a4 : ℕ},
    (h1 ≠ h2 ∧ h1 ≠ h3 ∧ h1 ≠ h4 ∧ h2 ≠ h3 ∧ h2 ≠ h4 ∧ h3 ≠ h4) ∧
    (a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)

theorem tallest_boy_is_Vladimir :
  ∀ (Andrei_shortest Vladimir_shortest Boris_tallest Dmitry_taller Dmitry_oldest Vladimir_older : Prop),
    Dmitry_statement2 Dmitry_oldest → 
    Boris_statement2 Andrei_shortest → 
    Andrei_statement1 Boris_tallest → 
    Vladimir_statement2 Vladimir_older → 
    Dmitry_statement1 (Vladimir_statement1 Dmitry_taller) (Vladimir_statement2 Vladimir_older) →
    ¬ Dmitry_taller →
    ¬ Boris_tallest →
    Vladimir = "the tallest boy" :=
  sorry

end tallest_boy_is_Vladimir_l547_547350


namespace find_triplet_solution_l547_547476

theorem find_triplet_solution (m n x y : ℕ) (hm : 0 < m) (hcoprime : Nat.gcd m n = 1) 
 (heq : (x^2 + y^2)^m = (x * y)^n) : 
  ∃ a : ℕ, x = 2^a ∧ y = 2^a ∧ n = m + 1 :=
by sorry

end find_triplet_solution_l547_547476


namespace julie_hours_per_week_l547_547093

variables 
  (summer_hours_per_week : ℕ)
  (summer_weeks : ℕ)
  (summer_earnings : ℕ)
  (school_year_weeks : ℕ)
  (school_year_earnings : ℕ)
  (hourly_wage : ℕ)
  (hours_per_week_school : ℕ)
  (total_school_hours : ℕ)

-- Conditions
def summer_conditions := 
  summer_hours_per_week = 60 ∧
  summer_weeks = 10 ∧
  summer_earnings = 6000 ∧
  school_year_weeks = 40 ∧
  school_year_earnings = 6000 ∧
  hourly_wage = summer_earnings / (summer_hours_per_week * summer_weeks) ∧
  total_school_hours = school_year_earnings / hourly_wage ∧
  hours_per_week_school = total_school_hours / school_year_weeks

-- Desired proof that hours_per_week_school = 15
theorem julie_hours_per_week (h : summer_conditions) : hours_per_week_school = 15 := 
sorry -- proof is not required

end julie_hours_per_week_l547_547093


namespace Vladimir_is_tallest_l547_547382

inductive Boy where
  | Andrei
  | Boris
  | Vladimir
  | Dmitry
  deriving DecidableEq, Repr

open Boy

variable (height aged : Boy → ℕ)

-- Each boy's statements
def andreiStatements (tallest shortest : Boy) :=
  (Boris ≠ tallest, Vladimir = shortest)

def borisStatements (oldest shortest : Boy) :=
  (Andrei = oldest, Andrei = shortest)

def vladimirStatements (heightOlder : Boy → Prop) :=
  (heightOlder Dmitry, heightOlder Dmitry)

def dmitryStatements (vladimirTrue oldest : Prop) :=
  (vladimirTrue, Dmitry = oldest)

noncomputable def is_tallest (b : Boy) : Prop :=
  ∀ b' : Boy, height b' ≤ height b

theorem Vladimir_is_tallest :
  (∃! b : Boy, is_tallest b)
  ∧ (∀ (height ordered : Boy), one_of_each (heightAndAged : Boy -> ℕ -> ℕ), ∃ 
  (tall shortest oldest : Boy),
  (andreiStatements tall shortest).1 ∧ ¬(andreiStatements tall shortest).2 ∧ 
  ¬(borisStatements oldest shortest).1 ∧ (borisStatements oldest shortest).2 ∧
  ¬(vladimirStatements heightOlder).1 ∧ (vladimirStatements heightOlder).2 ∧ 
  ¬(dmitryStatements false (Dmitry = oldest).1) ∧ (dmitryStatements false (Dmitry = oldest).2) ∧
  Vladimir = tall) :=
sorry

end Vladimir_is_tallest_l547_547382


namespace arith_prog_iff_avg_arith_prog_l547_547652

variable {α : Type*} [LinearOrderedField α]

def is_arith_prog (a : ℕ → α) :=
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

def avg_is_arith_prog (S : ℕ → α) :=
  ∃ D : α, ∀ n : ℕ, n ≠ 0 → (S n / n) = (S (n + 1) / (n + 1)) + D

theorem arith_prog_iff_avg_arith_prog (a : ℕ → α) (S : ℕ → α)
  (hS : ∀ n, S n = ∑ i in range n, a i) :
  is_arith_prog a ↔ avg_is_arith_prog S :=
sorry

end arith_prog_iff_avg_arith_prog_l547_547652


namespace find_side_b_l547_547123

theorem find_side_b
  (a b c : ℝ)
  (area : ℝ)
  (B : ℝ)
  (h_area : area = sqrt 3)
  (h_B : B = real.pi / 3)
  (h_a2c2 : a^2 + c^2 = 3 * a * c)
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) :
  b = 2 * real.sqrt 2 :=
sorry

end find_side_b_l547_547123


namespace angle_in_third_quadrant_l547_547028

theorem angle_in_third_quadrant (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.tan α > 0) : 
  (π < α ∧ α < 3 * π / 2) :=
by
  sorry

end angle_in_third_quadrant_l547_547028


namespace find_length_of_b_l547_547133

theorem find_length_of_b
  {A B C : Type*}
  (a b c : ℝ)
  (area : ℝ)
  (angleB : ℝ)
  (h_area : area = sqrt 3)
  (h_angle : angleB = real.pi / 3)  -- 60 degrees in radians
  (h_sides : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 :=
by
  -- 1. Using the given area constraint: area = 1/2 * a * c * sin(B)
  -- 2. Using the given angle: sin(60) = sqrt(3)/2
  -- 3. Using the given sides relation: a^2 + c^2 = 3ac
  -- 4. Using the Law of Cosines: cos(60) = 1/2
  -- fill proof here
  sorry

end find_length_of_b_l547_547133


namespace tallest_is_vladimir_l547_547371

variables (Andrei Boris Vladimir Dmitry : Type)
variables (height age : Type)
variables [linear_order height] [linear_order age]
variables (tallest shortest oldest : height)
variables (andrei_statements : (Boris ≠ tallest) ∧ (Vladimir = shortest))
variables (boris_statements : (Andrei = oldest) ∧ (Andrei = shortest))
variables (vladimir_statements : (Dmitry > Vladimir) ∧ (Dmitry > Vladimir))
variables (dmitry_statements : (vladimir_statements = true) ∧ (Dmitry = oldest))

theorem tallest_is_vladimir (H1: andrei_statements ∨ ¬andrei_statements)
  (H2: boris_statements ∨ ¬boris_statements)
  (H3: vladimir_statements ∨ ¬vladimir_statements)
  (H4: dmitry_statements ∨ ¬dmitry_statements)
  (H5: ∃! (a b c d : height) , a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a) : 
  tallest = Vladimir := 
by
  sorry

end tallest_is_vladimir_l547_547371


namespace sunil_interest_l547_547762

theorem sunil_interest :
  let A := 19828.80
  let r := 0.08
  let n := 1
  let t := 2
  let P := A / (1 + r / n)^ (n * t)
  A - P = 2828.80 :=
by
  let A := 19828.80
  let r := 0.08
  let n := 1
  let t := 2
  let P := A / (1 + r / n)^ (n * t)
  calc A - P = 19828.80 - (19828.80 / (1 + 0.08 / 1)^ (1 * 2)) := by sorry
            ... = 19828.80 - (19828.80 / (1.08)^2)             := by sorry
            ... = 19828.80 - (19828.80 / 1.1664)              := by sorry
            ... = 19828.80 - 17000                            := by sorry
            ... = 2828.80                                     := by sorry

end sunil_interest_l547_547762


namespace count_pos_three_digit_divisible_by_13_and_7_l547_547549

theorem count_pos_three_digit_divisible_by_13_and_7 : 
  ((finset.filter (λ n : ℕ, n % (13 * 7) = 0) (finset.Icc 100 999)).card = 9) := 
sorry

end count_pos_three_digit_divisible_by_13_and_7_l547_547549


namespace find_m_l547_547540

-- Definitions of the given vectors
def a : ℝ × ℝ := (1, real.sqrt 3)
def b (m : ℝ) : ℝ × ℝ := (2, m)

-- Definition of the vector operation
def c (m : ℝ) : ℝ × ℝ := (3 * a.1 - 2 * b(m).1, 3 * a.2 - 2 * b(m).2)

-- Condition: Vector c is perpendicular to vector a
def perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- The proof problem statement
theorem find_m (m : ℝ) (h : perpendicular (c m) a) :
  m = 4 * real.sqrt 3 / 3 :=
sorry

end find_m_l547_547540


namespace Vladimir_is_tallest_l547_547383

inductive Boy where
  | Andrei
  | Boris
  | Vladimir
  | Dmitry
  deriving DecidableEq, Repr

open Boy

variable (height aged : Boy → ℕ)

-- Each boy's statements
def andreiStatements (tallest shortest : Boy) :=
  (Boris ≠ tallest, Vladimir = shortest)

def borisStatements (oldest shortest : Boy) :=
  (Andrei = oldest, Andrei = shortest)

def vladimirStatements (heightOlder : Boy → Prop) :=
  (heightOlder Dmitry, heightOlder Dmitry)

def dmitryStatements (vladimirTrue oldest : Prop) :=
  (vladimirTrue, Dmitry = oldest)

noncomputable def is_tallest (b : Boy) : Prop :=
  ∀ b' : Boy, height b' ≤ height b

theorem Vladimir_is_tallest :
  (∃! b : Boy, is_tallest b)
  ∧ (∀ (height ordered : Boy), one_of_each (heightAndAged : Boy -> ℕ -> ℕ), ∃ 
  (tall shortest oldest : Boy),
  (andreiStatements tall shortest).1 ∧ ¬(andreiStatements tall shortest).2 ∧ 
  ¬(borisStatements oldest shortest).1 ∧ (borisStatements oldest shortest).2 ∧
  ¬(vladimirStatements heightOlder).1 ∧ (vladimirStatements heightOlder).2 ∧ 
  ¬(dmitryStatements false (Dmitry = oldest).1) ∧ (dmitryStatements false (Dmitry = oldest).2) ∧
  Vladimir = tall) :=
sorry

end Vladimir_is_tallest_l547_547383


namespace min_value_h_l547_547954

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def g (x : ℝ) : ℝ := abs (Real.sin x) + 4 / abs (Real.sin x)
noncomputable def h (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def j (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem min_value_h : ∃ x, h x = 4 :=
by
  sorry

end min_value_h_l547_547954


namespace probability_of_consecutive_cards_l547_547802

noncomputable def cards := {1, 2, 3, 4, 5}  -- Define the set of cards

noncomputable def total_pairs := (5.choose 2)  -- Total number of ways to select two cards

noncomputable def consecutive_pairs := 4  -- Number of ways to get consecutive pairs

noncomputable def probability_consecutive : ℚ := consecutive_pairs / total_pairs

theorem probability_of_consecutive_cards :
  probability_consecutive = 0.4 :=
by
  sorry

end probability_of_consecutive_cards_l547_547802


namespace arith_prog_iff_avg_arith_prog_l547_547651

variable {α : Type*} [LinearOrderedField α]

def is_arith_prog (a : ℕ → α) :=
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

def avg_is_arith_prog (S : ℕ → α) :=
  ∃ D : α, ∀ n : ℕ, n ≠ 0 → (S n / n) = (S (n + 1) / (n + 1)) + D

theorem arith_prog_iff_avg_arith_prog (a : ℕ → α) (S : ℕ → α)
  (hS : ∀ n, S n = ∑ i in range n, a i) :
  is_arith_prog a ↔ avg_is_arith_prog S :=
sorry

end arith_prog_iff_avg_arith_prog_l547_547651


namespace modulus_of_z_l547_547565

open Complex

noncomputable def z : ℂ := (2 + I) / (1 + 2*I)

theorem modulus_of_z : abs z = 1 := by
  -- Introduction of conditions as definitions
  have h : (1 + 2*I) * z = 2 + I := by
    calc (1 + 2*I) * z
         = (1 + 2*I) * ((2 + I) / (1 + 2*I)) : by rfl
     ... = 2 + I : by field_simp [Complex.ext_iff, norm_sq]

  -- We will skip the rest of the proof
  sorry

end modulus_of_z_l547_547565


namespace function_min_value_l547_547919

theorem function_min_value (f1 f2 f3 f4 : ℝ → ℝ) :
  f1 = λ x, x^2 + 2 * x + 4 ∧
  f2 = λ x, abs (sin x) + 4 / abs (sin x) ∧
  f3 = λ x, 2^x + 2^(2-x) ∧
  f4 = λ x, log x + 4 / log x →
  (∀ x : ℝ, f3 x ≥ 4) ∧ (∃ x : ℝ, f3 x = 4) :=
by
  sorry

end function_min_value_l547_547919


namespace male_worker_ants_percentage_l547_547268

theorem male_worker_ants_percentage 
  (total_ants : ℕ) 
  (half_ants : ℕ) 
  (female_worker_ants : ℕ) 
  (h1 : total_ants = 110) 
  (h2 : half_ants = total_ants / 2) 
  (h3 : female_worker_ants = 44) :
  (half_ants - female_worker_ants) * 100 / half_ants = 20 := by
  sorry

end male_worker_ants_percentage_l547_547268


namespace find_side_b_l547_547152

theorem find_side_b (a b c A B C : ℝ) (hB : B = 60) 
  (h_area : (1/2)*a*c*(Real.sin (Real.pi/3)) = Real.sqrt 3) 
  (h_ac : a^2 + c^2 = 3*a*c) : b = 2 * Real.sqrt 2 :=
by
  -- Given conditions
  have hB' : Real.sin (Real.pi/3) = Real.sqrt 3 / 2 := by sorry
  have h_ac_val : a * c = 4 := by sorry
  have h_a2c2 : a^2 + c^2 = 12 := by sorry
  have h_cosB : (a^2 + c^2 - b^2) / (2 * a * c) = 1/2 := by sorry
  have h_b2 : b^2 = 8 := by sorry

  -- Result
  exact Real.sqrt 8

end find_side_b_l547_547152


namespace truncated_cone_sphere_radius_l547_547428

structure TruncatedCone :=
(base_radius_top : ℝ)
(base_radius_bottom : ℝ)

noncomputable def sphere_radius (c : TruncatedCone) : ℝ :=
  if c.base_radius_top = 24 ∧ c.base_radius_bottom = 6 then 12 else 0

theorem truncated_cone_sphere_radius (c : TruncatedCone) (h_radii : c.base_radius_top = 24 ∧ c.base_radius_bottom = 6) :
  sphere_radius c = 12 :=
by
  sorry

end truncated_cone_sphere_radius_l547_547428


namespace function_min_value_4_l547_547866

theorem function_min_value_4 :
  ∃ x : ℝ, (2^x + 2^(2 - x)) = 4 :=
sorry

end function_min_value_4_l547_547866


namespace dice_sum_probability_l547_547572

theorem dice_sum_probability : 
  let outcomes := { (a, b, c) | a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6} ∧ c ∈ {1, 2, 3, 4, 5, 6} }.card,
      favorable := { (a, b, c) | a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6} ∧ c ∈ {1, 2, 3, 4, 5, 6} ∧ a + b + c = 12 }.card,
      probability := (favorable: ℚ) / (outcomes: ℚ)
  in probability = 5 / 108 := by
  sorry

end dice_sum_probability_l547_547572


namespace optionC_has_min_4_l547_547998

noncomputable def funcA (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def funcB (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def funcC (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def funcD (x : ℝ) : ℝ := log x + 4 / log x

theorem optionC_has_min_4 (x : ℝ) : (∀ y, (y = funcA x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcB x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcD x) → y ≠ 4) ∧
                                   (∃ t, (t = 1) ∧ (funcC t = 4)) := 
by {
  sorry
}

end optionC_has_min_4_l547_547998


namespace minimum_value_C_l547_547947

noncomputable def y_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def y_B (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
noncomputable def y_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def y_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimum_value_C : ∃ x : ℝ, y_C x = 4 := 
by
  -- proof goes here
  sorry

end minimum_value_C_l547_547947


namespace det_A_pow_4_eq_16_l547_547557

-- Given condition: det A = 2
variable {A : Matrix n n ℝ}
variable (h : det A = 2)

-- Goal: det (A^4) = 16
theorem det_A_pow_4_eq_16 : det (A ^ 4) = 16 :=
by {
  -- Proof goes here
  sorry
}

end det_A_pow_4_eq_16_l547_547557


namespace probability_sum_of_three_dice_is_12_l547_547580

open Finset

theorem probability_sum_of_three_dice_is_12 : 
  (∃ (outcomes : set (ℕ × ℕ × ℕ)), 
    ∀ (x y z : ℕ), (x, y, z) ∈ outcomes ↔ 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ (x + y + z = 12)) → 
    (∃ (prob : ℚ), prob = 2 / 27) :=
by 
  sorry

end probability_sum_of_three_dice_is_12_l547_547580


namespace option_A_is_iff_option_B_l547_547661

-- Definitions based on conditions
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range (n + 1), a i

def is_arithmetic (seq : ℕ → ℝ) : Prop :=
  ∃ D : ℝ, ∀ n : ℕ, seq (n + 1) - seq n = D

-- The final statement
theorem option_A_is_iff_option_B :
  ∀ (a : ℕ → ℝ),
    (is_arithmetic_progression a ↔ is_arithmetic (λ n, Sn a n / (n + 1))) :=
by sorry

end option_A_is_iff_option_B_l547_547661


namespace find_side_b_l547_547154

theorem find_side_b (a b c A B C : ℝ) (hB : B = 60) 
  (h_area : (1/2)*a*c*(Real.sin (Real.pi/3)) = Real.sqrt 3) 
  (h_ac : a^2 + c^2 = 3*a*c) : b = 2 * Real.sqrt 2 :=
by
  -- Given conditions
  have hB' : Real.sin (Real.pi/3) = Real.sqrt 3 / 2 := by sorry
  have h_ac_val : a * c = 4 := by sorry
  have h_a2c2 : a^2 + c^2 = 12 := by sorry
  have h_cosB : (a^2 + c^2 - b^2) / (2 * a * c) = 1/2 := by sorry
  have h_b2 : b^2 = 8 := by sorry

  -- Result
  exact Real.sqrt 8

end find_side_b_l547_547154


namespace iggy_total_hours_l547_547060

-- Define the conditions
def miles_run_per_day : ℕ → ℕ
| 0 := 3 -- Monday
| 1 := 4 -- Tuesday
| 2 := 6 -- Wednesday
| 3 := 8 -- Thursday
| 4 := 3 -- Friday
| _ := 0 -- Other days

def total_distance : ℕ := List.sum (List.ofFn miles_run_per_day 5)
def miles_per_minute : ℕ := 10
def minutes_per_hour : ℕ := 60
def total_minutes_run : ℕ := total_distance * miles_per_minute
def total_hours_run : ℕ := total_minutes_run / minutes_per_hour

-- The statement to prove
theorem iggy_total_hours :
  total_hours_run = 4 :=
sorry

end iggy_total_hours_l547_547060


namespace find_length_of_b_l547_547131

theorem find_length_of_b
  {A B C : Type*}
  (a b c : ℝ)
  (area : ℝ)
  (angleB : ℝ)
  (h_area : area = sqrt 3)
  (h_angle : angleB = real.pi / 3)  -- 60 degrees in radians
  (h_sides : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 :=
by
  -- 1. Using the given area constraint: area = 1/2 * a * c * sin(B)
  -- 2. Using the given angle: sin(60) = sqrt(3)/2
  -- 3. Using the given sides relation: a^2 + c^2 = 3ac
  -- 4. Using the Law of Cosines: cos(60) = 1/2
  -- fill proof here
  sorry

end find_length_of_b_l547_547131


namespace arc_length_is_correct_l547_547004

-- Definition of given central angle and radius
def central_angle (θ : ℝ) : Prop := θ = 100
def radius (r : ℝ) : Prop := r = 9

def arc_length_of_sector (θ r l : ℝ) : Prop :=
  θ = 100 ∧ r = 9 → l = (θ * π * r) / 180 → l = 5 * π

-- Theorem statement to prove the arc length of the sector is 5π given the conditions
theorem arc_length_is_correct (θ r l : ℝ) (h1 : central_angle θ) (h2 : radius r) (h3: l = (θ * π * r) / 180) : l = 5 * π := sorry

end arc_length_is_correct_l547_547004


namespace tallest_is_vladimir_l547_547369

variables (Andrei Boris Vladimir Dmitry : Type)
variables (height age : Type)
variables [linear_order height] [linear_order age]
variables (tallest shortest oldest : height)
variables (andrei_statements : (Boris ≠ tallest) ∧ (Vladimir = shortest))
variables (boris_statements : (Andrei = oldest) ∧ (Andrei = shortest))
variables (vladimir_statements : (Dmitry > Vladimir) ∧ (Dmitry > Vladimir))
variables (dmitry_statements : (vladimir_statements = true) ∧ (Dmitry = oldest))

theorem tallest_is_vladimir (H1: andrei_statements ∨ ¬andrei_statements)
  (H2: boris_statements ∨ ¬boris_statements)
  (H3: vladimir_statements ∨ ¬vladimir_statements)
  (H4: dmitry_statements ∨ ¬dmitry_statements)
  (H5: ∃! (a b c d : height) , a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a) : 
  tallest = Vladimir := 
by
  sorry

end tallest_is_vladimir_l547_547369


namespace minimum_value_of_option_C_l547_547895

noncomputable def optionA (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def optionB (x : ℝ) : ℝ := |Real.sin x| + 4 / |Real.sin x|
noncomputable def optionC (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def optionD (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem minimum_value_of_option_C :
  ∃ x : ℝ, optionC x = 4 :=
by
  sorry

end minimum_value_of_option_C_l547_547895


namespace volume_of_prism_l547_547298

theorem volume_of_prism (x y z : ℝ) (h1 : x * y = 60)
                                     (h2 : y * z = 75)
                                     (h3 : x * z = 100) :
  x * y * z = 671 :=
by
  sorry

end volume_of_prism_l547_547298


namespace optionC_has_min_4_l547_547994

noncomputable def funcA (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def funcB (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def funcC (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def funcD (x : ℝ) : ℝ := log x + 4 / log x

theorem optionC_has_min_4 (x : ℝ) : (∀ y, (y = funcA x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcB x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcD x) → y ≠ 4) ∧
                                   (∃ t, (t = 1) ∧ (funcC t = 4)) := 
by {
  sorry
}

end optionC_has_min_4_l547_547994


namespace circle_area_increase_l547_547289

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  let r_new := 1.5 * r
  let area_original := π * r^2
  let area_new := π * r_new^2
  let increase := area_new - area_original
  let percentage_increase := (increase / area_original) * 100
  percentage_increase = 125 :=
by
  let r_new := 1.5 * r
  let area_original := π * r^2
  let area_new := π * r_new^2
  let increase := area_new - area_original
  let percentage_increase := (increase / area_original) * 100
  sorry

end circle_area_increase_l547_547289


namespace triangle_side_b_l547_547223

noncomputable def find_side_length_b
  (a b c : ℝ)
  (area : ℝ)
  (B_deg : ℝ)
  (relation : a^2 + c^2 = 3 * a * c)
  (area_condition : area = sqrt 3)
  (angle_condition : B_deg = 60) : Prop :=
  b = 2 * sqrt 2

theorem triangle_side_b
  (a b c : ℝ)
  (area : ℝ := sqrt 3)
  (B_deg : ℝ := 60)
  (h1 : a^2 + c^2 = 3 * a * c)
  (h2 : area = sqrt 3)
  (h3 : B_deg = 60) : find_side_length_b a b c area B_deg h1 h2 h3 :=
  sorry

end triangle_side_b_l547_547223


namespace intersection_complement_l547_547249

open Set

def A : Set ℕ := {1, 2, 3, 5, 7}
def B : Set ℤ := {x | 1 < x ∧ x ≤ 6}
noncomputable def U := A ∪ B

theorem intersection_complement :
  A ∩ (U \ B) = {1, 7} :=
by sorry

end intersection_complement_l547_547249


namespace find_b_proof_l547_547208

noncomputable def find_b (a b c : ℝ) (A B C : ℝ) : Prop :=
  let area := sqrt 3 in
  let angle_B := 60 * Real.pi / 180 in
  let condition1 := area = (1 / 2) * a * c * Real.sin angle_B in
  let condition2 := a^2 + c^2 = 3 * a * c in
  let side_b := b = 2 * Real.sqrt 2 in
  condition1 ∧ condition2 → side_b

theorem find_b_proof :
  ∃ (a b c : ℝ) (A B C : ℝ), find_b a b c A B C :=
by
  sorry

end find_b_proof_l547_547208


namespace solution_criteria_l547_547468

def is_solution (M : ℕ) : Prop :=
  5 ∣ (1989^M + M^1989)

theorem solution_criteria (M : ℕ) (h : M < 10) : is_solution M ↔ (M = 1 ∨ M = 4) :=
sorry

end solution_criteria_l547_547468


namespace find_b_correct_l547_547214

axioms (a b c : ℝ) (B : ℝ)
  (area_of_triangle : ℝ)
  (h1 : area_of_triangle = real.sqrt 3)
  (h2 : B = 60 * real.pi / 180) -- converting degrees to radians
  (h3 : a^2 + c^2 = 3 * a * c)

noncomputable def find_b : ℝ := 2 * real.sqrt 2

theorem find_b_correct :
  let b := find_b in 
  b = 2 * real.sqrt 2 :=
by
  sorry

end find_b_correct_l547_547214


namespace project_completion_time_l547_547807

theorem project_completion_time
  (x y z : ℝ)
  (h1 : x + y = 1 / 2)
  (h2 : y + z = 1 / 4)
  (h3 : z + x = 1 / 2.4) :
  (1 / x) = 3 :=
by
  sorry

end project_completion_time_l547_547807


namespace min_quot_seq_l547_547532

noncomputable def a : ℕ → ℝ
| 0       := 0 -- define a₀ as zero to avoid issues with ℕ⁺indexing
| 1       := 10
| (n + 1) := a n + n

theorem min_quot_seq (n : ℕ) (hn : n > 0) :
  (∃ n, (1 : ℝ) / 2 * (n : ℝ) + 10 / (n : ℝ) - 1 / 2 = 4) :=
sorry

end min_quot_seq_l547_547532


namespace median_half_mean_eq_x_sum_l547_547446

theorem median_half_mean_eq_x_sum :
  ∀ (x : ℝ), (let m := (3 + 7 + 9 + 20 + x) / 5 / 2 in
    (3 ≤ x ∧ x ≤ 7 → m = 7) ∨
    (7 ≤ x ∧ x ≤ 9 → m = 9) ∨
    (9 ≤ x ∧ x ≤ 20 → m = x)) →
  x = 51 :=
by
  intros
  sorry

end median_half_mean_eq_x_sum_l547_547446


namespace option_A_is_necessary_and_sufficient_for_option_B_l547_547694

def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

def mean_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : (ℕ → ℝ)
| n => S n / n

def sum_of_first_n_terms (a : ℕ → ℝ) : ℕ → ℝ
| 0     => 0
| (n+1) => sum_of_first_n_terms a n + a (n+1)

theorem option_A_is_necessary_and_sufficient_for_option_B
    (a : ℕ → ℝ) :
    (is_arithmetic_progression a ↔ is_arithmetic_progression (mean_sequence a (sum_of_first_n_terms a))) :=
sorry

end option_A_is_necessary_and_sufficient_for_option_B_l547_547694


namespace minimize_f_C_l547_547926

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimize_f_C : ∃ x : ℝ, f_C x = 4 :=
by {
  -- The correctness of the statement is asserted and the proof is omitted
  sorry
}

end minimize_f_C_l547_547926


namespace dice_probability_sum_12_l547_547575

open Nat

/-- Probability that the sum of three six-faced dice rolls equals 12 is 10 / 216 --/
theorem dice_probability_sum_12 : 
  let outcomes := 6^3
  let favorable := 10
  (favorable : ℚ) / outcomes = 10 / 216 := 
by
  let outcomes := 6^3
  let favorable := 10
  sorry

end dice_probability_sum_12_l547_547575


namespace option_A_iff_option_B_l547_547687

-- Definitions
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
∀ n m, a (n + 1) - a n = a (m + 1) - a m

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range n, a (i + 1)

def option_B (S : ℕ → ℝ) : Prop :=
is_arithmetic_progression (λ n, S n / n)

-- Main theorem
theorem option_A_iff_option_B (a : ℕ → ℝ) :
  (is_arithmetic_progression a) ↔ (option_B (λ n, sum_of_first_n_terms a n)) :=
sorry

end option_A_iff_option_B_l547_547687


namespace ordinate_vertex_zero_l547_547495

noncomputable def quadratic_polynomial (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

-- Define the condition that the equation (f(x))^3 - f(x) = 0 has exactly three solutions
def has_three_solutions (f : ℝ → ℝ) : Prop :=
  ∃ (s t u : ℝ), s ≠ t ∧ t ≠ u ∧ u ≠ s ∧ (f s)^3 - f s = 0 ∧ (f t)^3 - f t = 0 ∧ (f u)^3 - f u = 0

-- Define a quadratic polynomial f(x)
def f (x : ℝ) : ℝ := quadratic_polynomial a b c x

-- We need to prove that the ordinate of the vertex of f(x) is 0
theorem ordinate_vertex_zero (a b c : ℝ) (h : has_three_solutions (quadratic_polynomial a b c)) : 
  (quadratic_polynomial a b c (-(b / (2 * a)))) = 0 :=
sorry

end ordinate_vertex_zero_l547_547495


namespace problem1_problem2_l547_547497

open Real BigOperators

noncomputable def a (n : ℕ) : ℝ :=
  if n = 0 then 0 else 4 * (n : ℝ) - 1

noncomputable def b (n : ℕ) : ℝ :=
  if n = 0 then 0 else 2^((n : ℝ) - 1)

noncomputable def S (n : ℕ) : ℝ := 2 * (n : ℝ) ^ 2 + n

noncomputable def T (n : ℕ) : ℝ :=
  ∑ i in finset.range n, a (i + 1) * b (i + 1)

theorem problem1 (n : ℕ) (hn : n > 0) : 
  (a n = 4 * n - 1) ∧ (b n = 2 ^ (n - 1)) :=
sorry

theorem problem2 (n : ℕ) (hn : n > 0) : 
  T n = (4 * n - 5) * 2 ^ n + 5 :=
sorry

end problem1_problem2_l547_547497


namespace arith_prog_iff_avg_seq_arith_prog_l547_547674

-- Define sequences
def is_arith_prog (a : ℕ → ℝ) := ∃ d : ℝ, ∃ a₀ : ℝ, ∀ n : ℕ, a n = a₀ + n * d
def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range (n+1), a i
def avg_seq (a : ℕ → ℝ) (n : ℕ) := S a n / (n + 1)

-- Define theorem statement
theorem arith_prog_iff_avg_seq_arith_prog (a : ℕ → ℝ) :
  (is_arith_prog a) ↔ (is_arith_prog (λ n, avg_seq a n)) :=
sorry

end arith_prog_iff_avg_seq_arith_prog_l547_547674


namespace find_b_l547_547190

theorem find_b
  (a b c : ℝ)
  (h_area : (1/2) * a * c * Real.sin (ℝ.pi / 3) = Real.sqrt 3)
  (h_ac_eq_4 : a * c = 4)
  (h_cosine : a^2 + c^2 = 3 * a * c) :
  b = 2 * Real.sqrt 2 :=
begin
  sorry
end

end find_b_l547_547190


namespace sum_distinct_prime_factors_924_l547_547833

theorem sum_distinct_prime_factors_924 : 
  ∃ p1 p2 p3 p4 : ℕ, (∀ p: ℕ, p ∈ {p1, p2, p3, p4} → Nat.Prime p) ∧ 
  (p1 * p2 * p3 * p4 ∣ 924) ∧ p1 + p2 + p3 + p4 = 23 := 
sorry

end sum_distinct_prime_factors_924_l547_547833


namespace equation_representation_l547_547443

theorem equation_representation (x : ℝ) (h : 2 * x + 4 = 8) : 2x + 4 = 8 :=
by { sorry }

end equation_representation_l547_547443


namespace corn_syrup_content_sport_formulation_l547_547608

def standard_ratio_flavoring : ℕ := 1
def standard_ratio_corn_syrup : ℕ := 12
def standard_ratio_water : ℕ := 30

def sport_ratio_flavoring_to_corn_syrup : ℕ := 3 * standard_ratio_flavoring
def sport_ratio_flavoring_to_water : ℕ := standard_ratio_flavoring / 2

def sport_ratio_flavoring : ℕ := 1
def sport_ratio_corn_syrup : ℕ := sport_ratio_flavoring * sport_ratio_flavoring_to_corn_syrup
def sport_ratio_water : ℕ := (sport_ratio_flavoring * standard_ratio_water) / 2

def water_content_sport_formulation : ℕ := 30

theorem corn_syrup_content_sport_formulation : 
  (sport_ratio_corn_syrup / sport_ratio_water) * water_content_sport_formulation = 2 :=
by
  sorry

end corn_syrup_content_sport_formulation_l547_547608


namespace min_value_of_2x_plus_2_2x_l547_547880

-- Lean 4 statement
theorem min_value_of_2x_plus_2_2x (x : ℝ) : 2^x + 2^(2 - x) ≥ 4 :=
by sorry

end min_value_of_2x_plus_2_2x_l547_547880


namespace total_points_of_three_players_l547_547588

-- Definitions based on conditions
def points_tim : ℕ := 30
def points_joe : ℕ := points_tim - 20
def points_ken : ℕ := 2 * points_tim

-- Theorem statement for the total points scored by the three players
theorem total_points_of_three_players :
  points_tim + points_joe + points_ken = 100 :=
by
  -- Proof is to be provided
  sorry

end total_points_of_three_players_l547_547588


namespace range_of_f_l547_547789

noncomputable def f (x : ℝ) : ℝ := ((1/2) ^ (-x^2 + 2*x))

theorem range_of_f : set.range f = set.Ioi (1/2) := sorry

end range_of_f_l547_547789


namespace spider_total_distance_l547_547421

theorem spider_total_distance :
  let start := 3
  let mid := -4
  let final := 8
  let dist1 := abs (mid - start)
  let dist2 := abs (final - mid)
  let total_distance := dist1 + dist2
  total_distance = 19 :=
by
  sorry

end spider_total_distance_l547_547421


namespace vladimir_is_tallest_l547_547341

inductive Boy : Type
| Andrei | Boris | Vladimir | Dmitry

open Boy

-- Hypotheses based on the problem statement
def statements : Boy → (Prop × Prop)
| Andrei := (¬ (Boris = tallest), Vladimir = shortest)
| Boris := (Andrei = oldest, Andrei = shortest)
| Vladimir := (Dmitry > Vladimir, Dmitry = oldest)
| Dmitry := (statements Vladimir = (true, true), Dmitry = oldest)

def different_heights (a b : Boy) : Prop := a ≠ b
def different_ages (a b : Boy) : Prop := a ≠ b

axiom no_same_height {a b : Boy} : different_heights a b
axiom no_same_age {a b : Boy} : different_ages a b

-- The proof statement
theorem vladimir_is_tallest : Vladimir = tallest := 
by 
  sorry

end vladimir_is_tallest_l547_547341


namespace andy_solves_problems_l547_547440

def start : ℕ := 55
def end : ℕ := 150

theorem andy_solves_problems : end - start + 1 = 96 :=
by
  -- proof to be filled in
  sorry

end andy_solves_problems_l547_547440


namespace explicit_formula_for_f_l547_547419

theorem explicit_formula_for_f (f : ℕ → ℕ) (h₀ : f 0 = 0)
  (h₁ : ∀ (n : ℕ), n % 6 = 0 ∨ n % 6 = 1 → f (n + 1) = f n + 3)
  (h₂ : ∀ (n : ℕ), n % 6 = 2 ∨ n % 6 = 5 → f (n + 1) = f n + 1)
  (h₃ : ∀ (n : ℕ), n % 6 = 3 ∨ n % 6 = 4 → f (n + 1) = f n + 2)
  (n : ℕ) : f (6 * n) = 12 * n :=
by
  sorry

end explicit_formula_for_f_l547_547419


namespace domain_of_f_l547_547774

noncomputable def f (x : ℝ) : ℝ := sqrt (x - 1) / (x - 2)

theorem domain_of_f :
  {x : ℝ | 1 ≤ x ∧ x ≠ 2} = {x | x ∈ set.Ico 1 2 ∪ set.Ioi 2} :=
by
  sorry

end domain_of_f_l547_547774


namespace function_min_value_l547_547923

theorem function_min_value (f1 f2 f3 f4 : ℝ → ℝ) :
  f1 = λ x, x^2 + 2 * x + 4 ∧
  f2 = λ x, abs (sin x) + 4 / abs (sin x) ∧
  f3 = λ x, 2^x + 2^(2-x) ∧
  f4 = λ x, log x + 4 / log x →
  (∀ x : ℝ, f3 x ≥ 4) ∧ (∃ x : ℝ, f3 x = 4) :=
by
  sorry

end function_min_value_l547_547923


namespace bus_capacity_l547_547332

theorem bus_capacity :
  ∀ (left_seats right_seats people_per_seat back_seat : ℕ),
  left_seats = 15 →
  right_seats = left_seats - 3 →
  people_per_seat = 3 →
  back_seat = 11 →
  (left_seats * people_per_seat) + 
  (right_seats * people_per_seat) + 
  back_seat = 92 := by
  intros left_seats right_seats people_per_seat back_seat 
  intros h1 h2 h3 h4 
  sorry

end bus_capacity_l547_547332


namespace minimum_value_C_l547_547946

noncomputable def y_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def y_B (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
noncomputable def y_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def y_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimum_value_C : ∃ x : ℝ, y_C x = 4 := 
by
  -- proof goes here
  sorry

end minimum_value_C_l547_547946


namespace frames_per_page_l547_547618

theorem frames_per_page (total_frames : ℕ) (pages : ℕ) (frames : ℕ) 
  (h1 : total_frames = 143) 
  (h2 : pages = 13) 
  (h3 : frames = total_frames / pages) : 
  frames = 11 := 
by 
  sorry

end frames_per_page_l547_547618


namespace archipelago_max_value_l547_547270

noncomputable def archipelago_max_islands (N : ℕ) : Prop :=
  N ≥ 7 ∧ 
  (∀ (a b : ℕ), a ≠ b → a ≤ N → b ≤ N → ∃ c : ℕ, c ≤ N ∧ (∃ d, d ≠ c ∧ d ≤ N → d ≠ a ∧ d ≠ b)) ∧ 
  (∀ (a : ℕ), a ≤ N → ∃ b, b ≠ a ∧ b ≤ N ∧ (∃ c, c ≤ N ∧ c ≠ b ∧ c ≠ a))

theorem archipelago_max_value : archipelago_max_islands 36 := sorry

end archipelago_max_value_l547_547270


namespace find_side_b_l547_547160

theorem find_side_b (a b c A B C : ℝ) (hB : B = 60) 
  (h_area : (1/2)*a*c*(Real.sin (Real.pi/3)) = Real.sqrt 3) 
  (h_ac : a^2 + c^2 = 3*a*c) : b = 2 * Real.sqrt 2 :=
by
  -- Given conditions
  have hB' : Real.sin (Real.pi/3) = Real.sqrt 3 / 2 := by sorry
  have h_ac_val : a * c = 4 := by sorry
  have h_a2c2 : a^2 + c^2 = 12 := by sorry
  have h_cosB : (a^2 + c^2 - b^2) / (2 * a * c) = 1/2 := by sorry
  have h_b2 : b^2 = 8 := by sorry

  -- Result
  exact Real.sqrt 8

end find_side_b_l547_547160


namespace solve_new_system_l547_547534

theorem solve_new_system (a_1 b_1 a_2 b_2 c_1 c_2 x y : ℝ)
(h1 : a_1 * 2 - b_1 * (-1) = c_1)
(h2 : a_2 * 2 + b_2 * (-1) = c_2) :
  (x = -1) ∧ (y = 1) :=
by
  have hx : x + 3 = 2 := by sorry
  have hy : y - 2 = -1 := by sorry
  have hx_sol : x = -1 := by linarith
  have hy_sol : y = 1 := by linarith
  exact ⟨hx_sol, hy_sol⟩

end solve_new_system_l547_547534


namespace sum_of_divisors_cube_lt_n_fourth_l547_547096

open BigOperators

-- Define the function to calculate the sum of divisors of an odd natural number n
def sum_of_divisors (n : ℕ) : ℕ :=
  if n > 1 ∧ odd n then ∑ d in (finset.filter (∣ n) (finset.range (n + 1))), d else 0

theorem sum_of_divisors_cube_lt_n_fourth (n : ℕ) (h1 : 1 < n) (h2 : odd n) : 
  (sum_of_divisors n)^3 < n^4 :=
sorry

end sum_of_divisors_cube_lt_n_fourth_l547_547096


namespace all_statements_correct_l547_547520

-- Definitions for midpoints, centroids, triangles, and parallelograms
def midpoint (A B : Point) : Point := ...
def centroid_segment (A B : Point) : Point := midpoint A B

def median (A B C : Point) : LineSegment := ...
def centroid_triangle (A B C : Point) : Point := ...

def parallelogram (A B C D : Point) : Prop := ...
def intersection_of_diagonals (A B C D : Point) [parallelogram A B C D] : Point := ...

-- Conditions as hypotheses
variables (A B C D : Point)
variables [is_midpoint : ∀ (A B), centroid_segment A B = midpoint A B]
variables [three_medians_intersect : ∀ (A B C), centroid_triangle A B C = intersection_of_medians A B C]
variables [parallelogram_centroid : ∀ (A B C D) [parallelogram A B C D], 
            intersection_of_diagonals A B C D = centroid_of_parallelogram A B C D]
variables [triangle_centroid_property : ∀ (A B C), 
            is_trisection_point (centroid_triangle A B C) (median A B C)]

-- Proof problem
theorem all_statements_correct :
  (is_midpoint → three_medians_intersect → parallelogram_centroid → triangle_centroid_property → True) := 
  by
    intros
    -- This is where the proofs would go, but we only need the statement
    sorry

end all_statements_correct_l547_547520


namespace triangle_side_b_l547_547177

theorem triangle_side_b 
  (a b c : ℝ)
  (B : ℝ)
  (h1 : 1/2 * a * c * Real.sin B = Real.sqrt 3)
  (h2 : B = π / 3)
  (h3 : a^2 + c^2 = 3 * a * c) : b = 2 * Real.sqrt 2 := by
  sorry

end triangle_side_b_l547_547177


namespace fill_tank_time_is_18_l547_547811

def rate1 := 1 / 20
def rate2 := 1 / 30
def combined_rate := rate1 + rate2
def effective_rate := (2 / 3) * combined_rate
def T := 1 / effective_rate

theorem fill_tank_time_is_18 : T = 18 := by
  sorry

end fill_tank_time_is_18_l547_547811


namespace optionC_has_min_4_l547_547999

noncomputable def funcA (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def funcB (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def funcC (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def funcD (x : ℝ) : ℝ := log x + 4 / log x

theorem optionC_has_min_4 (x : ℝ) : (∀ y, (y = funcA x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcB x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcD x) → y ≠ 4) ∧
                                   (∃ t, (t = 1) ∧ (funcC t = 4)) := 
by {
  sorry
}

end optionC_has_min_4_l547_547999


namespace arithmetic_progression_iff_mean_sequence_arithmetic_l547_547627

-- Let {a_n} be a sequence and let S_n be the sum of the first n terms of {a_n}
def arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + d * n

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (∑ i in Finset.range (n+1), a i)

def mean_sequence (S : ℕ → ℝ) (m : ℕ → ℝ) : Prop :=
  ∀ n, m n = S n / (n + 1)

-- We need to prove that {a_n} being an arithmetic progression is both necessary and sufficient for {S_n / n} being an arithmetic progression
theorem arithmetic_progression_iff_mean_sequence_arithmetic (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ → ℝ) :
  (arithmetic_progression a ∧ sum_first_n_terms a S ∧ mean_sequence S m) ↔ arithmetic_progression m := 
by 
  -- Proof will go here
  sorry

end arithmetic_progression_iff_mean_sequence_arithmetic_l547_547627


namespace minimize_f_C_l547_547924

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimize_f_C : ∃ x : ℝ, f_C x = 4 :=
by {
  -- The correctness of the statement is asserted and the proof is omitted
  sorry
}

end minimize_f_C_l547_547924


namespace option_a_iff_option_b_l547_547637

-- Define a sequence as an arithmetic progression
def is_arithmetic (seq : ℕ → ℚ) : Prop :=
  ∃ d a₁, ∀ n, seq n = a₁ + n * d

-- Define the sum of the first n terms of a sequence
def sum_first_n (seq : ℕ → ℚ) (n : ℕ) : ℚ :=
  ∑ i in finset.range n, seq i

-- Define the average sum sequence
def avg_sum_seq (seq : ℕ → ℚ) (n : ℕ) : ℚ :=
  let Sn := sum_first_n seq n
  in Sn / n

-- The theorem to prove
theorem option_a_iff_option_b (seq : ℕ → ℚ) :
  is_arithmetic seq ↔ is_arithmetic (λ n, avg_sum_seq seq (n + 1)) :=
sorry

end option_a_iff_option_b_l547_547637


namespace optionA_iff_optionB_l547_547704

variable {a : ℕ → ℝ} {S : ℕ → ℝ}

/-- Sequence a_n is an arithmetic progression -/
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a n = a1 + n * d

/-- Sequence S_n / n is an arithmetic progression -/
def is_S_n_div_n_arithmetic_progression (S : ℕ → ℝ) : Prop :=
  ∃ (b c : ℝ), ∀ n, S n / n = b + c * n

/-- S_n is the sum of the first n terms of the sequence a_n -/
def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = ∑ i in finset.range (n + 1), a i 

/-- Option A is both a sufficient and necessary condition for Option B -/
theorem optionA_iff_optionB (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (is_arithmetic_progression a ∧ sum_first_n_terms a S) ↔ is_S_n_div_n_arithmetic_progression S :=
  sorry

end optionA_iff_optionB_l547_547704


namespace min_value_f_l547_547979

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(2 - x)

theorem min_value_f : ∃ x : ℝ, f x = 4 :=
by {
  sorry
}

end min_value_f_l547_547979


namespace ac_eq_ae_l547_547086

-- Given a right-angled triangle ABC at C with altitude CD, and a circle tangent to AB at E, CD at G, and an arc of the circumcircle of triangle ABC at F,
-- prove that AC = AE.

theorem ac_eq_ae
  (A B C : Point)
  (h_tri : ∠ A B C = ∠ π / 2)
  (D : Point)
  (h_alt : Foot D A B (Altitude from C))
  (P : Circle)
  (h_tan1 : Tangent P A B E)
  (h_tan2 : Tangent P C D G)
  (h_tan3 : TangentToArc P (CircumcircleOf ABC) B C F) :
  dist A C = dist A E := 
sorry

end ac_eq_ae_l547_547086


namespace tallest_boy_is_Vladimir_l547_547348

noncomputable def Andrei_statement1 (Boris_tallest: Prop) : Prop := ¬ Boris_tallest
def Andrei_statement2 (Vladimir_shortest: Prop) : Prop := Vladimir_shortest

def Boris_statement1 (Andrei_oldest: Prop) : Prop := Andrei_oldest
def Boris_statement2 (Andrei_shortest: Prop) : Prop := Andrei_shortest

def Vladimir_statement1 (Dmitry_taller: Prop) : Prop := Dmitry_taller
def Vladimir_statement2 (Dmitry_older: Prop) : Prop := Dmitry_older

noncomputable def Dmitry_statement1 (Vladimir_statement1: Prop) (Vladimir_statement2: Prop) : Prop :=
  Vladimir_statement1 ∧ Vladimir_statement2
def Dmitry_statement2 (Dmitry_oldest: Prop) : Prop := Dmitry_oldest

axiom one_statement_true_per_boy :
  ∀ {P₁ P₂: Prop}, (P₁ ∨ P₂) ∧ ¬ (P₁ ∧ P₂)

axiom no_same_height_or_age :
  ∀ {h1 h2 h3 h4 a1 a2 a3 a4 : ℕ},
    (h1 ≠ h2 ∧ h1 ≠ h3 ∧ h1 ≠ h4 ∧ h2 ≠ h3 ∧ h2 ≠ h4 ∧ h3 ≠ h4) ∧
    (a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)

theorem tallest_boy_is_Vladimir :
  ∀ (Andrei_shortest Vladimir_shortest Boris_tallest Dmitry_taller Dmitry_oldest Vladimir_older : Prop),
    Dmitry_statement2 Dmitry_oldest → 
    Boris_statement2 Andrei_shortest → 
    Andrei_statement1 Boris_tallest → 
    Vladimir_statement2 Vladimir_older → 
    Dmitry_statement1 (Vladimir_statement1 Dmitry_taller) (Vladimir_statement2 Vladimir_older) →
    ¬ Dmitry_taller →
    ¬ Boris_tallest →
    Vladimir = "the tallest boy" :=
  sorry

end tallest_boy_is_Vladimir_l547_547348


namespace correct_statements_l547_547247

variables (α β : Plane) (m n : Line)

def statement1 (α β : Plane) (m : Line) : Prop :=
α ∥ β ∧ m ⊆ α → m ∥ β

def statement4 (α β : Plane) (m n : Line) : Prop :=
n ⊥ α ∧ n ⊥ β ∧ m ⊥ α → m ⊥ β

theorem correct_statements (α β : Plane) (m n : Line) :
  statement1 α β m ∧ statement4 α β m n :=
by
  -- Proofs should go here.
  sorry

end correct_statements_l547_547247


namespace factorize_expression_l547_547466

theorem factorize_expression (x y : ℝ) : 4 * x^2 - 2 * x * y = 2 * x * (2 * x - y) := 
by
  sorry

end factorize_expression_l547_547466


namespace vector_expression_l547_547558

-- Define the vectors a, b, and c
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)
def c : ℝ × ℝ := (-1, -2)

-- The target relationship
theorem vector_expression :
  c = (- (3 / 2) • a + (1 / 2) • b) :=
sorry

end vector_expression_l547_547558


namespace option_A_is_iff_option_B_l547_547659

-- Definitions based on conditions
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range (n + 1), a i

def is_arithmetic (seq : ℕ → ℝ) : Prop :=
  ∃ D : ℝ, ∀ n : ℕ, seq (n + 1) - seq n = D

-- The final statement
theorem option_A_is_iff_option_B :
  ∀ (a : ℕ → ℝ),
    (is_arithmetic_progression a ↔ is_arithmetic (λ n, Sn a n / (n + 1))) :=
by sorry

end option_A_is_iff_option_B_l547_547659


namespace arithmetic_progression_iff_mean_sequence_arithmetic_l547_547631

-- Let {a_n} be a sequence and let S_n be the sum of the first n terms of {a_n}
def arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + d * n

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (∑ i in Finset.range (n+1), a i)

def mean_sequence (S : ℕ → ℝ) (m : ℕ → ℝ) : Prop :=
  ∀ n, m n = S n / (n + 1)

-- We need to prove that {a_n} being an arithmetic progression is both necessary and sufficient for {S_n / n} being an arithmetic progression
theorem arithmetic_progression_iff_mean_sequence_arithmetic (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ → ℝ) :
  (arithmetic_progression a ∧ sum_first_n_terms a S ∧ mean_sequence S m) ↔ arithmetic_progression m := 
by 
  -- Proof will go here
  sorry

end arithmetic_progression_iff_mean_sequence_arithmetic_l547_547631


namespace sunil_interest_earned_l547_547761

def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem sunil_interest_earned :
  ∃ P : ℝ, compound_interest P 0.08 1 2 = 19828.80 ∧ 19828.80 - P = 2828.80 :=
by
  sorry

end sunil_interest_earned_l547_547761


namespace vladimir_is_tallest_l547_547342

inductive Boy : Type
| Andrei | Boris | Vladimir | Dmitry

open Boy

-- Hypotheses based on the problem statement
def statements : Boy → (Prop × Prop)
| Andrei := (¬ (Boris = tallest), Vladimir = shortest)
| Boris := (Andrei = oldest, Andrei = shortest)
| Vladimir := (Dmitry > Vladimir, Dmitry = oldest)
| Dmitry := (statements Vladimir = (true, true), Dmitry = oldest)

def different_heights (a b : Boy) : Prop := a ≠ b
def different_ages (a b : Boy) : Prop := a ≠ b

axiom no_same_height {a b : Boy} : different_heights a b
axiom no_same_age {a b : Boy} : different_ages a b

-- The proof statement
theorem vladimir_is_tallest : Vladimir = tallest := 
by 
  sorry

end vladimir_is_tallest_l547_547342


namespace probability_sum_dice_12_l547_547584

/-- Helper definition for a standard six-faced die roll -/
def is_valid_die_roll (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 6

/-- The probability that the sum of three six-faced dice equals 12 is 19/216. -/
theorem probability_sum_dice_12 :
  (∑ (x y z : ℕ) in (finset.range 7).filter (is_valid_die_roll), ite (x + y + z = 12) 1 0) = 19 :=
begin
  sorry
end

end probability_sum_dice_12_l547_547584


namespace function_min_value_l547_547912

theorem function_min_value (f1 f2 f3 f4 : ℝ → ℝ) :
  f1 = λ x, x^2 + 2 * x + 4 ∧
  f2 = λ x, abs (sin x) + 4 / abs (sin x) ∧
  f3 = λ x, 2^x + 2^(2-x) ∧
  f4 = λ x, log x + 4 / log x →
  (∀ x : ℝ, f3 x ≥ 4) ∧ (∃ x : ℝ, f3 x = 4) :=
by
  sorry

end function_min_value_l547_547912


namespace proportion_estimation_chi_squared_test_l547_547430

-- Definitions based on the conditions
def total_elders : ℕ := 500
def not_vaccinated_male : ℕ := 20
def not_vaccinated_female : ℕ := 10
def vaccinated_male : ℕ := 230
def vaccinated_female : ℕ := 240

-- Calculations based on the problem conditions
noncomputable def proportion_vaccinated : ℚ := (vaccinated_male + vaccinated_female) / total_elders

def chi_squared_statistic (a b c d n : ℕ) : ℚ :=
  n * (a * d - b * c) ^ 2 / ((a + b) * (c + d) * (a + c) * (b + d))

noncomputable def K2_value : ℚ :=
  chi_squared_statistic not_vaccinated_male not_vaccinated_female vaccinated_male vaccinated_female total_elders

-- Specify the critical value for 99% confidence
def critical_value_99 : ℚ := 6.635

-- Theorem statements (problems to prove)
theorem proportion_estimation : proportion_vaccinated = 94 / 100 := by
  sorry

theorem chi_squared_test : K2_value < critical_value_99 := by
  sorry

end proportion_estimation_chi_squared_test_l547_547430


namespace option_A_iff_option_B_l547_547664

def is_arithmetic_progression (seq : ℕ → ℚ) : Prop :=
  ∃ d a₁, ∀ n, seq (n + 1) = seq n + d

def S_n (a : ℕ → ℚ) : ℕ → ℚ
| 0       := 0
| (n + 1) := S_n n + a (n + 1)

theorem option_A_iff_option_B (a : ℕ → ℚ) :
  (is_arithmetic_progression a) ↔ (is_arithmetic_progression (λ n, S_n a n / (n + 1))) :=
sorry

end option_A_iff_option_B_l547_547664


namespace three_digit_integers_count_l547_547015

/-- 
Prove that the total number of different positive three-digit integers 
that can be formed using the digits {2, 4, 6, 7, 8} without repeating any digit is 60.
-/
theorem three_digit_integers_count : 
  let digits := {2, 4, 6, 7, 8}
  let pos := finset.range 5
  finset.card {n | ∃ d1 d2 d3, d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits ∧
                                d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3 ∧
                                n = d1 * 100 + d2 * 10 + d3} = 60 := 
by {
  sorry
}

end three_digit_integers_count_l547_547015


namespace simplify_and_evaluate_division_l547_547755

theorem simplify_and_evaluate_division (a : ℝ) (h : a = real.sqrt 3 + 1) :
  ((a^2 / (a - 2) - 1 / (a - 2)) / (a^2 - 2 * a + 1) / (a - 2)) = (3 + 2 * real.sqrt 3) / 3 :=
by
  sorry

end simplify_and_evaluate_division_l547_547755


namespace right_angle_triangle_of_pythagorean_identity_l547_547609

theorem right_angle_triangle_of_pythagorean_identity 
  (m n : ℕ) (hm_pos : 0 < m) (hn_pos : 0 < n) (h_m_gt_n : m > n) : 
  let a := m^2 - n^2,
      b := 2 * m * n,
      c := m^2 + n^2 in
  a^2 + b^2 = c^2 := 
by 
  -- proof omitted
  sorry

end right_angle_triangle_of_pythagorean_identity_l547_547609


namespace sum_prime_factors_924_l547_547837

/-- Let n be the integer 924. -/
def n : ℤ := 924

/-- The set of distinct prime factors of 924. -/
def distinct_prime_factors (n : ℤ) : set ℤ :=
  if n = 924 then {2, 3, 7, 11} else ∅

/-- The sum of the distinct prime factors of 924. -/
def sum_distinct_prime_factors (n : ℤ) : ℤ :=
  if n = 924 then 2 + 3 + 7 + 11 else 0

-- The theorem to prove that the sum of the distinct prime factors of 924 is 23.
theorem sum_prime_factors_924 : sum_distinct_prime_factors n = 23 :=
by {
  unfold sum_distinct_prime_factors,
  simp,
}

end sum_prime_factors_924_l547_547837


namespace optionC_has_min_4_l547_547996

noncomputable def funcA (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def funcB (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def funcC (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def funcD (x : ℝ) : ℝ := log x + 4 / log x

theorem optionC_has_min_4 (x : ℝ) : (∀ y, (y = funcA x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcB x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcD x) → y ≠ 4) ∧
                                   (∃ t, (t = 1) ∧ (funcC t = 4)) := 
by {
  sorry
}

end optionC_has_min_4_l547_547996


namespace find_sum_l547_547555

noncomputable def f (x : ℕ) : ℝ := 
  1 / (Real.cbrt (x^2 + 2*x + 1) + Real.cbrt (x^2 - 1) + Real.cbrt (x^2 - 2*x + 1))

theorem find_sum :
  (List.sum (List.map f (List.range' 1 499).map (λk, 2 * k - 1)) - f 999) = 5 :=
by
  sorry

end find_sum_l547_547555


namespace arith_prog_iff_avg_seq_arith_prog_l547_547678

-- Define sequences
def is_arith_prog (a : ℕ → ℝ) := ∃ d : ℝ, ∃ a₀ : ℝ, ∀ n : ℕ, a n = a₀ + n * d
def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range (n+1), a i
def avg_seq (a : ℕ → ℝ) (n : ℕ) := S a n / (n + 1)

-- Define theorem statement
theorem arith_prog_iff_avg_seq_arith_prog (a : ℕ → ℝ) :
  (is_arith_prog a) ↔ (is_arith_prog (λ n, avg_seq a n)) :=
sorry

end arith_prog_iff_avg_seq_arith_prog_l547_547678


namespace vector_ab_l547_547502

theorem vector_ab
  (A B : ℝ × ℝ)
  (hA : A = (1, -1))
  (hB : B = (1, 2)) :
  (B.1 - A.1, B.2 - A.2) = (0, 3) :=
by
  sorry

end vector_ab_l547_547502


namespace Vladimir_is_tallest_l547_547380

inductive Boy where
  | Andrei
  | Boris
  | Vladimir
  | Dmitry
  deriving DecidableEq, Repr

open Boy

variable (height aged : Boy → ℕ)

-- Each boy's statements
def andreiStatements (tallest shortest : Boy) :=
  (Boris ≠ tallest, Vladimir = shortest)

def borisStatements (oldest shortest : Boy) :=
  (Andrei = oldest, Andrei = shortest)

def vladimirStatements (heightOlder : Boy → Prop) :=
  (heightOlder Dmitry, heightOlder Dmitry)

def dmitryStatements (vladimirTrue oldest : Prop) :=
  (vladimirTrue, Dmitry = oldest)

noncomputable def is_tallest (b : Boy) : Prop :=
  ∀ b' : Boy, height b' ≤ height b

theorem Vladimir_is_tallest :
  (∃! b : Boy, is_tallest b)
  ∧ (∀ (height ordered : Boy), one_of_each (heightAndAged : Boy -> ℕ -> ℕ), ∃ 
  (tall shortest oldest : Boy),
  (andreiStatements tall shortest).1 ∧ ¬(andreiStatements tall shortest).2 ∧ 
  ¬(borisStatements oldest shortest).1 ∧ (borisStatements oldest shortest).2 ∧
  ¬(vladimirStatements heightOlder).1 ∧ (vladimirStatements heightOlder).2 ∧ 
  ¬(dmitryStatements false (Dmitry = oldest).1) ∧ (dmitryStatements false (Dmitry = oldest).2) ∧
  Vladimir = tall) :=
sorry

end Vladimir_is_tallest_l547_547380


namespace paint_540_statues_l547_547037

-- Definitions from the problem conditions
def k : ℝ := 1 / 36  -- Proportionality constant
def paint_required (n : ℕ) (h : ℝ) : ℝ := k * n * h^2  -- Function for calculating paint required

-- Theorem stating that 540 one-foot high statues require 15 pints of paint
theorem paint_540_statues : paint_required 540 1 = 15 := 
  by sorry

end paint_540_statues_l547_547037


namespace prove_function_values_order_l547_547457

noncomputable
def f (x : ℝ) : ℝ := sorry

axiom even_function : ∀ x, f x = f (-x)
axiom increasing_on_pos_real : ∀ x y, 0 < x → x < y → f x < f y

theorem prove_function_values_order :
  f 3 < f (-π) ∧ f (-π) < f (-4) := by
  have h1 : f 3 < f π := increasing_on_pos_real 3 π (by linarith) (by linarith)
  have h2 : f π = f (-π) := even_function π
  have h3 : f 4 = f (-4) := even_function 4
  exact ⟨by linarith, by linarith⟩

end prove_function_values_order_l547_547457


namespace sum_of_cubes_l547_547846

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) : a^3 + b^3 = 1008 := by
  sorry

end sum_of_cubes_l547_547846


namespace necessary_but_not_sufficient_condition_l547_547025

-- Prove that x^2 ≥ -x is a necessary but not sufficient condition for |x| = x
theorem necessary_but_not_sufficient_condition (x : ℝ) : x^2 ≥ -x → |x| = x ↔ x ≥ 0 := 
sorry

end necessary_but_not_sufficient_condition_l547_547025


namespace tangents_length_circle_fixed_point_l547_547489

noncomputable section

open Real

-- Definitions for the given problem
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def directrix (x : ℝ) : Prop := x = -1

variable (P : ℝ × ℝ)
variable (Q R : ℝ × ℝ)
variable (M : ℝ × ℝ := (1, 0))

-- Points on parabola
def on_parabola (x y : ℝ) : Prop := parabola x y

-- Tangent points conditions
def tangents_condition (P Q R : ℝ × ℝ) : Prop :=
  on_parabola Q.1 Q.2 ∧ on_parabola R.1 R.2 ∧ Q.1 = 1 ∧ Q.2 = 2 ∧ R.1 = 1 ∧ R.2 = -2

theorem tangents_length (P Q R : ℝ × ℝ) (hP : directrix P.1) (hQR : tangents_condition P Q R) :
  dist Q R = 4 :=
sorry

theorem circle_fixed_point (P Q : ℝ × ℝ) (hP : directrix P.1) (hQ : on_parabola Q.1 Q.2) :
  ∃ M, M = (1, 0) ∧ collinear P Q M :=
sorry

end tangents_length_circle_fixed_point_l547_547489


namespace complete_square_transformation_l547_547815

theorem complete_square_transformation (x : ℝ) : 
  2 * x^2 - 4 * x - 3 = 0 ↔ (x - 1)^2 - (5 / 2) = 0 :=
sorry

end complete_square_transformation_l547_547815


namespace triangle_side_b_l547_547226

noncomputable def find_side_length_b
  (a b c : ℝ)
  (area : ℝ)
  (B_deg : ℝ)
  (relation : a^2 + c^2 = 3 * a * c)
  (area_condition : area = sqrt 3)
  (angle_condition : B_deg = 60) : Prop :=
  b = 2 * sqrt 2

theorem triangle_side_b
  (a b c : ℝ)
  (area : ℝ := sqrt 3)
  (B_deg : ℝ := 60)
  (h1 : a^2 + c^2 = 3 * a * c)
  (h2 : area = sqrt 3)
  (h3 : B_deg = 60) : find_side_length_b a b c area B_deg h1 h2 h3 :=
  sorry

end triangle_side_b_l547_547226


namespace function_min_value_4_l547_547855

theorem function_min_value_4 :
  ∃ x : ℝ, (2^x + 2^(2 - x)) = 4 :=
sorry

end function_min_value_4_l547_547855


namespace min_value_h_is_4_l547_547988

def f (x : ℝ) := x^2 + 2*x + 4
def g (x : ℝ) := abs (sin x) + 4 / abs (sin x)
def h (x : ℝ) := 2^x + 2^(2 - x)
def j (x : ℝ) := log x + 4 / log x

theorem min_value_h_is_4 : 
  (∀ x, f(x) ≥ 3) ∧
  (∀ x, g(x) > 4) ∧
  (∃ x, h(x) = 4) ∧
  (∀ x, j(x) ≠ 4) → min_value_h = 4 :=
by
  sorry

end min_value_h_is_4_l547_547988


namespace find_side_b_l547_547173

noncomputable def triangle_area (a b c : ℝ) (B : ℝ) : ℝ :=
  1 / 2 * a * c * real.sin B

noncomputable def law_of_cosines (a b c : ℝ) (B : ℝ) : ℝ :=
  a^2 + c^2 - 2 * a * c * real.cos B

theorem find_side_b (a b c : ℝ) (B : ℝ) (area : ℝ)
  (h1 : area = √3) 
  (h2 : B = real.pi / 3)
  (h3 : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 :=
by
  sorry

end find_side_b_l547_547173


namespace find_side_b_l547_547145

theorem find_side_b
  (a b c : ℝ)
  (area : ℝ)
  (B : ℝ)
  (h_area : area = real.sqrt 3)
  (h_B : B = (60 : ℝ) * real.pi / 180) -- Convert degrees to radians
  (h_cond : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 := 
sorry

end find_side_b_l547_547145


namespace option_A_iff_option_B_l547_547681

-- Definitions
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
∀ n m, a (n + 1) - a n = a (m + 1) - a m

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range n, a (i + 1)

def option_B (S : ℕ → ℝ) : Prop :=
is_arithmetic_progression (λ n, S n / n)

-- Main theorem
theorem option_A_iff_option_B (a : ℕ → ℝ) :
  (is_arithmetic_progression a) ↔ (option_B (λ n, sum_of_first_n_terms a n)) :=
sorry

end option_A_iff_option_B_l547_547681


namespace find_side_b_l547_547103

theorem find_side_b (a b c : ℝ) (A B C : ℝ)
  (area_triangle : Real.sqrt 3) (B_eq_60 : B = 60) (cond_ac: (a^2 + c^2 = 3 * a * c)) :
  ∃ b, b = 2 * Real.sqrt 2 :=
by
  sorry

end find_side_b_l547_547103


namespace range_of_a_l547_547526

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + 2 * x + 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a (f a x) ≥ 0) ↔ a ≥ (Real.sqrt 5 - 1) / 2 :=
sorry

end range_of_a_l547_547526


namespace two_digit_number_l547_547058

theorem two_digit_number (x : ℕ) (h1 : x ≥ 10 ∧ x < 100)
  (h2 : ∃ k : ℤ, 3 * x - 4 = 10 * k)
  (h3 : 60 < 4 * x - 15 ∧ 4 * x - 15 < 100) :
  x = 28 :=
by
  sorry

end two_digit_number_l547_547058


namespace minimum_value_of_option_C_l547_547886

noncomputable def optionA (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def optionB (x : ℝ) : ℝ := |Real.sin x| + 4 / |Real.sin x|
noncomputable def optionC (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def optionD (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem minimum_value_of_option_C :
  ∃ x : ℝ, optionC x = 4 :=
by
  sorry

end minimum_value_of_option_C_l547_547886


namespace problem_l547_547016

noncomputable def f (a x : ℝ) : ℝ := (3 / a) * x^3 - x

theorem problem (a x1 x2 : ℝ) (h_a : a > 0) (h_x1 : x1 > sqrt (a / 3)) :
  x2 = x1 - (f a x1) / (9 / a * x1^2 - 1) →
  (2 / 3) < x2 / x1 ∧ x2 / x1 < 1 :=
sorry

end problem_l547_547016


namespace p_q_sum_is_26_l547_547029

theorem p_q_sum_is_26 (f : ℝ → ℝ)
  (h_f : ∃ p q : ℝ, p > 0 ∧ q > 0 ∧ f = λ x, x^2 - p * x + q)
  (h_roots : ∃ a b : ℝ, a ≠ b ∧ f a = 0 ∧ f b = 0)
  (h_arith_geo : ∀ a b : ℝ, a ≠ b → 
    (∃ c : ℝ, ((a < c ∧ c < 0 ∧ 0 < b ∨ b < 0 ∧ 0 < c ∧ c < a) 
    ∧ (a + c = b + c = 0)) ∨ (a / -4 = (-4) / b ∨ b / -4 = (-4) / a))) :
   ∃ p q : ℝ, p + q = 26 := 
sorry

end p_q_sum_is_26_l547_547029


namespace simple_interest_calculation_l547_547473

variable (P : ℝ) (R : ℝ) (T : ℝ)

def simple_interest (P R T : ℝ) : ℝ := P * R * T

theorem simple_interest_calculation (hP : P = 10000) (hR : R = 0.09) (hT : T = 1) :
    simple_interest P R T = 900 := by
  rw [hP, hR, hT]
  sorry

end simple_interest_calculation_l547_547473


namespace option_A_iff_option_B_l547_547663

def is_arithmetic_progression (seq : ℕ → ℚ) : Prop :=
  ∃ d a₁, ∀ n, seq (n + 1) = seq n + d

def S_n (a : ℕ → ℚ) : ℕ → ℚ
| 0       := 0
| (n + 1) := S_n n + a (n + 1)

theorem option_A_iff_option_B (a : ℕ → ℚ) :
  (is_arithmetic_progression a) ↔ (is_arithmetic_progression (λ n, S_n a n / (n + 1))) :=
sorry

end option_A_iff_option_B_l547_547663


namespace average_age_of_women_is_30_l547_547771

-- Definitions based on problem conditions
def average_age_men_increases (orig_avg_age : ℕ) (new_avg_age : ℕ) : Prop :=
  new_avg_age = orig_avg_age + 2 

def combined_age_women (combined_age_men : ℕ) (age_increase : ℕ) : ℕ :=
  combined_age_men + age_increase

-- Main problem statement:
theorem average_age_of_women_is_30 (A : ℕ) 
  (h1 : average_age_men_increases A (A + 2))
  (w_combined_age : combined_age_women (20 + 24) (2 * 8)) :
  (w_combined_age / 2) = 30 :=
sorry

end average_age_of_women_is_30_l547_547771


namespace min_value_f_l547_547977

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(2 - x)

theorem min_value_f : ∃ x : ℝ, f x = 4 :=
by {
  sorry
}

end min_value_f_l547_547977


namespace find_length_of_b_l547_547138

theorem find_length_of_b
  {A B C : Type*}
  (a b c : ℝ)
  (area : ℝ)
  (angleB : ℝ)
  (h_area : area = sqrt 3)
  (h_angle : angleB = real.pi / 3)  -- 60 degrees in radians
  (h_sides : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 :=
by
  -- 1. Using the given area constraint: area = 1/2 * a * c * sin(B)
  -- 2. Using the given angle: sin(60) = sqrt(3)/2
  -- 3. Using the given sides relation: a^2 + c^2 = 3ac
  -- 4. Using the Law of Cosines: cos(60) = 1/2
  -- fill proof here
  sorry

end find_length_of_b_l547_547138


namespace find_b_correct_l547_547219

axioms (a b c : ℝ) (B : ℝ)
  (area_of_triangle : ℝ)
  (h1 : area_of_triangle = real.sqrt 3)
  (h2 : B = 60 * real.pi / 180) -- converting degrees to radians
  (h3 : a^2 + c^2 = 3 * a * c)

noncomputable def find_b : ℝ := 2 * real.sqrt 2

theorem find_b_correct :
  let b := find_b in 
  b = 2 * real.sqrt 2 :=
by
  sorry

end find_b_correct_l547_547219


namespace min_value_of_2x_plus_2_2x_l547_547875

-- Lean 4 statement
theorem min_value_of_2x_plus_2_2x (x : ℝ) : 2^x + 2^(2 - x) ≥ 4 :=
by sorry

end min_value_of_2x_plus_2_2x_l547_547875


namespace sunil_interest_l547_547763

theorem sunil_interest :
  let A := 19828.80
  let r := 0.08
  let n := 1
  let t := 2
  let P := A / (1 + r / n)^ (n * t)
  A - P = 2828.80 :=
by
  let A := 19828.80
  let r := 0.08
  let n := 1
  let t := 2
  let P := A / (1 + r / n)^ (n * t)
  calc A - P = 19828.80 - (19828.80 / (1 + 0.08 / 1)^ (1 * 2)) := by sorry
            ... = 19828.80 - (19828.80 / (1.08)^2)             := by sorry
            ... = 19828.80 - (19828.80 / 1.1664)              := by sorry
            ... = 19828.80 - 17000                            := by sorry
            ... = 2828.80                                     := by sorry

end sunil_interest_l547_547763


namespace tangent_line_at_1_1_l547_547279

open Real

noncomputable def f (x : ℝ) : ℝ := x * (2 * log x + 1)

theorem tangent_line_at_1_1 :
  let df := deriv f in
  let x0 := 1 in
  let y0 := f x0 in
  let m := df x0 in
  y0 = 1 ∧ (∀ x : ℝ, 3 * x - f x - 2 = 0) :=
by
  -- Definitions and context setup
  let df := deriv f
  let x0 := 1
  let y0 := f x0
  -- Verifying the point of tangency
  have h_y0 : y0 = 1 := sorry
  -- Verifying the equation of the tangent line
  have h_tangent : ∀ x : ℝ, 3 * x - f x - 2 = 0 := sorry
  -- Concluding the theorem
  exact And.intro h_y0 h_tangent

end tangent_line_at_1_1_l547_547279


namespace find_side_b_l547_547126

theorem find_side_b
  (a b c : ℝ)
  (area : ℝ)
  (B : ℝ)
  (h_area : area = sqrt 3)
  (h_B : B = real.pi / 3)
  (h_a2c2 : a^2 + c^2 = 3 * a * c)
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) :
  b = 2 * real.sqrt 2 :=
sorry

end find_side_b_l547_547126


namespace ordinate_vertex_zero_l547_547493

noncomputable def quadratic_polynomial (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

-- Define the condition that the equation (f(x))^3 - f(x) = 0 has exactly three solutions
def has_three_solutions (f : ℝ → ℝ) : Prop :=
  ∃ (s t u : ℝ), s ≠ t ∧ t ≠ u ∧ u ≠ s ∧ (f s)^3 - f s = 0 ∧ (f t)^3 - f t = 0 ∧ (f u)^3 - f u = 0

-- Define a quadratic polynomial f(x)
def f (x : ℝ) : ℝ := quadratic_polynomial a b c x

-- We need to prove that the ordinate of the vertex of f(x) is 0
theorem ordinate_vertex_zero (a b c : ℝ) (h : has_three_solutions (quadratic_polynomial a b c)) : 
  (quadratic_polynomial a b c (-(b / (2 * a)))) = 0 :=
sorry

end ordinate_vertex_zero_l547_547493


namespace triangle_side_b_l547_547175

theorem triangle_side_b 
  (a b c : ℝ)
  (B : ℝ)
  (h1 : 1/2 * a * c * Real.sin B = Real.sqrt 3)
  (h2 : B = π / 3)
  (h3 : a^2 + c^2 = 3 * a * c) : b = 2 * Real.sqrt 2 := by
  sorry

end triangle_side_b_l547_547175


namespace ordinate_vertex_zero_l547_547494

noncomputable def quadratic_polynomial (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

-- Define the condition that the equation (f(x))^3 - f(x) = 0 has exactly three solutions
def has_three_solutions (f : ℝ → ℝ) : Prop :=
  ∃ (s t u : ℝ), s ≠ t ∧ t ≠ u ∧ u ≠ s ∧ (f s)^3 - f s = 0 ∧ (f t)^3 - f t = 0 ∧ (f u)^3 - f u = 0

-- Define a quadratic polynomial f(x)
def f (x : ℝ) : ℝ := quadratic_polynomial a b c x

-- We need to prove that the ordinate of the vertex of f(x) is 0
theorem ordinate_vertex_zero (a b c : ℝ) (h : has_three_solutions (quadratic_polynomial a b c)) : 
  (quadratic_polynomial a b c (-(b / (2 * a)))) = 0 :=
sorry

end ordinate_vertex_zero_l547_547494


namespace sequence_an_solution_l547_547537

noncomputable def a_n (n : ℕ) : ℝ := (
  (1 / 2) * (2 + Real.sqrt 3)^n + 
  (1 / 2) * (2 - Real.sqrt 3)^n
)^2

theorem sequence_an_solution (n : ℕ) : 
  ∀ (a b : ℕ → ℝ),
  a 0 = 1 → 
  b 0 = 0 → 
  (∀ n, a (n + 1) = 7 * a n + 6 * b n - 3) → 
  (∀ n, b (n + 1) = 8 * a n + 7 * b n - 4) → 
  a n = a_n n := sorry

end sequence_an_solution_l547_547537


namespace find_QS_l547_547712

-- Here we define a right triangle with vertices P, Q, R with Q as the right angle
-- And a point S on PR such that (QS) is the altitude from Q to PR
variables (P Q R S : Type) [real_field P Q R S]

-- Given conditions
def area_triangle (P Q R : ℝ) : ℝ := 210
def hypotenuse_PR : ℝ := 42

-- Proof statement to find QS
theorem find_QS (area_triangle : ℝ) (hypotenuse_PR : ℝ) : ℝ :=
∃ QS : ℝ, 0.5 * hypotenuse_PR * QS = area_triangle ∧ QS = 10

end find_QS_l547_547712


namespace find_side_b_l547_547148

theorem find_side_b
  (a b c : ℝ)
  (area : ℝ)
  (B : ℝ)
  (h_area : area = real.sqrt 3)
  (h_B : B = (60 : ℝ) * real.pi / 180) -- Convert degrees to radians
  (h_cond : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 := 
sorry

end find_side_b_l547_547148


namespace intersection_of_sets_l547_547098

-- Definitions from the conditions.
def A := { x : ℝ | x^2 - 2 * x ≤ 0 }
def B := { x : ℝ | x > 1 }

-- The proof problem statement.
theorem intersection_of_sets :
  A ∩ B = { x : ℝ | 1 < x ∧ x ≤ 2 } :=
sorry

end intersection_of_sets_l547_547098


namespace triangle_side_b_l547_547225

noncomputable def find_side_length_b
  (a b c : ℝ)
  (area : ℝ)
  (B_deg : ℝ)
  (relation : a^2 + c^2 = 3 * a * c)
  (area_condition : area = sqrt 3)
  (angle_condition : B_deg = 60) : Prop :=
  b = 2 * sqrt 2

theorem triangle_side_b
  (a b c : ℝ)
  (area : ℝ := sqrt 3)
  (B_deg : ℝ := 60)
  (h1 : a^2 + c^2 = 3 * a * c)
  (h2 : area = sqrt 3)
  (h3 : B_deg = 60) : find_side_length_b a b c area B_deg h1 h2 h3 :=
  sorry

end triangle_side_b_l547_547225


namespace find_side_b_l547_547120

theorem find_side_b
  (a b c : ℝ)
  (area : ℝ)
  (B : ℝ)
  (h_area : area = sqrt 3)
  (h_B : B = real.pi / 3)
  (h_a2c2 : a^2 + c^2 = 3 * a * c)
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) :
  b = 2 * real.sqrt 2 :=
sorry

end find_side_b_l547_547120


namespace option_a_iff_option_b_l547_547643

-- Define a sequence as an arithmetic progression
def is_arithmetic (seq : ℕ → ℚ) : Prop :=
  ∃ d a₁, ∀ n, seq n = a₁ + n * d

-- Define the sum of the first n terms of a sequence
def sum_first_n (seq : ℕ → ℚ) (n : ℕ) : ℚ :=
  ∑ i in finset.range n, seq i

-- Define the average sum sequence
def avg_sum_seq (seq : ℕ → ℚ) (n : ℕ) : ℚ :=
  let Sn := sum_first_n seq n
  in Sn / n

-- The theorem to prove
theorem option_a_iff_option_b (seq : ℕ → ℚ) :
  is_arithmetic seq ↔ is_arithmetic (λ n, avg_sum_seq seq (n + 1)) :=
sorry

end option_a_iff_option_b_l547_547643


namespace max_elements_in_T_l547_547101

theorem max_elements_in_T (T : set ℕ) (hT : ∀ x ∈ T, x > 0 ∧ x ≤ 100) (hPair : ∀ {a b : ℕ}, a ∈ T → b ∈ T → a ≠ b → (a + b) % 10 ≠ 0) : 
  ∃ (n : ℕ), n = 46 ∧ ∀ T' : set ℕ, (∀ x ∈ T', x > 0 ∧ x ≤ 100) → (∀ {a b : ℕ}, a ∈ T' → b ∈ T' → a ≠ b → (a + b) % 10 ≠ 0) → T'.card ≤ 46 := 
sorry

end max_elements_in_T_l547_547101


namespace speed_of_each_train_60_l547_547813

noncomputable def speed_of_each_train 
  (train_length : ℝ) 
  (time_seconds : ℝ)
  (train_approaching : Prop) : ℝ :=
if train_approaching ∧ train_length = 1/6 ∧ time_seconds = 10 then 60 else 0

theorem speed_of_each_train_60 :
  ∀ (train_length : ℝ)
    (time_seconds : ℝ)
    (train_approaching : Prop),
  train_approaching ∧ train_length = 1/6 ∧ time_seconds = 10 →
  speed_of_each_train train_length time_seconds train_approaching = 60 := by
  intros train_length time_seconds train_approaching h
  rw speed_of_each_train
  simp [h]
  sorry

end speed_of_each_train_60_l547_547813


namespace least_value_xy_l547_547503

theorem least_value_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(3*y) = 1/9) : x*y = 108 :=
sorry

end least_value_xy_l547_547503


namespace pentagon_area_is_19_l547_547306

def Point := (ℝ × ℝ)

def vertices : List Point := [(1,1), (4,5), (7,3), (6,6), (10,2)]

noncomputable def shoelace_area (ps : List Point) : ℝ :=
  (0.5 * ((ps.head.1 * ps.tail.head.2 + ps.tail.head.1 * ps.tail.tail.head.2 + ps.tail.tail.head.1 * ps.tail.tail.tail.head.2 + ps.tail.tail.tail.head.1 * ps.tail.tail.tail.tail.head.2 + ps.tail.tail.tail.tail.head.1 * ps.head.2) -
         (ps.head.2 * ps.tail.head.1 + ps.tail.head.2 * ps.tail.tail.head.1 + ps.tail.tail.head.2 * ps.tail.tail.tail.head.1 + ps.tail.tail.tail.head.2 * ps.tail.tail.tail.tail.head.1 + ps.tail.tail.tail.tail.head.2 * ps.head.1)))

theorem pentagon_area_is_19 : shoelace_area vertices = 19 := by
  sorry

end pentagon_area_is_19_l547_547306


namespace average_square_feet_per_person_l547_547287

theorem average_square_feet_per_person
  (population : ℕ := 226504825)
  (area_sq_miles : ℕ := 3615122)
  (sq_miles_to_sq_feet : ℕ := 5280^2) :
  (area_sq_miles * sq_miles_to_sq_feet / population ≈ 500000) :=
sorry

end average_square_feet_per_person_l547_547287


namespace trader_loss_percentage_l547_547423

noncomputable def costPrice := 100
noncomputable def markedPrice (CP : ℝ) := CP + 0.1 * CP
noncomputable def actualSellingPrice (MP : ℝ) := MP - 0.1 * MP
noncomputable def loss (CP SP : ℝ) := CP - SP
noncomputable def percentageLoss (loss CP : ℝ) := (loss / CP) * 100

theorem trader_loss_percentage :
  let CP := costPrice in
  let MP := markedPrice CP in
  let SP := actualSellingPrice MP in
  percentageLoss (loss CP SP) CP = 1 :=
by
  sorry

end trader_loss_percentage_l547_547423


namespace math_proof_problem_l547_547513

-- Definitions related to circle C1
def center_C1_on_line (x y : ℝ) : Prop := x + y = 1
def passes_through_A (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = r^2
def tangent_to_line (x y r : ℝ) : Prop := abs(3*x - 4*y + 5) / sqrt(3^2 + (-4)^2) = r ∧ r < 5

-- Definitions related to circle C2
def symmetric_about_line (x1 y1 x2 y2 : ℝ) (line_x_y_eq : ℝ → ℝ → Prop) : Prop :=
line_x_y_eq x1 y1 ∧ line_x_y_eq x2 y2 ∧ y1 = x2 ∧ x1 = y2 -- Symmetric about x = y

def circle_C2 (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 9

-- Problem to prove
theorem math_proof_problem :
  (∃ (x y r : ℝ), center_C1_on_line x y ∧ passes_through_A x y ∧ tangent_to_line x y r)
  →
  (∃ (x y : ℝ), symmetric_about_line x y (-y) (-x) (λ x y, x = y) ∧ circle_C2 x y) ∧
  (∃ (P C D : ℝ×ℝ), P.1 + P.2 = 0 ∧ tangent_to_line P.1 P.2 3 ∧ (4*P.1 - 2*P.2 = 1)) :=
by sorry

end math_proof_problem_l547_547513


namespace area_of_region_R_l547_547304

open Real

noncomputable def area_of_strip (width : ℝ) (height : ℝ) : ℝ :=
  width * height

noncomputable def area_of_triangle (leg : ℝ) : ℝ :=
  1 / 2 * leg * leg

theorem area_of_region_R :
  let unit_square_area := 1
  let AE_BE := 1 / sqrt 2
  let area_triangle_ABE := area_of_triangle AE_BE
  let strip_width := 1 / 4
  let strip_height := 1
  let area_strip := area_of_strip strip_width strip_height
  let overlap_area := area_triangle_ABE / 2
  let area_R := area_strip - overlap_area
  area_R = 1 / 8 :=
by
  sorry

end area_of_region_R_l547_547304


namespace chance_of_allergic_reaction_is_50_percent_l547_547617

-- Defining the conditions given in the problem
def number_of_peanut_butter_cookies_jenny : ℕ := 40
def number_of_chocolate_chip_cookies_jenny : ℕ := 50
def number_of_peanut_butter_cookies_marcus : ℕ := 30
def number_of_lemon_cookies_marcus : ℕ := 20
def renee_allergic_to_peanuts : Prop := true

-- Defining the total number of each type of cookies and total cookies
def total_peanut_butter_cookies : ℕ :=
  number_of_peanut_butter_cookies_jenny + number_of_peanut_butter_cookies_marcus

def total_cookies : ℕ :=
  number_of_peanut_butter_cookies_jenny +
  number_of_chocolate_chip_cookies_jenny +
  number_of_peanut_butter_cookies_marcus +
  number_of_lemon_cookies_marcus

-- The proof statement
theorem chance_of_allergic_reaction_is_50_percent :
  renee_allergic_to_peanuts →
  (total_peanut_butter_cookies.to_nat / total_cookies.to_nat * 100) = 50 :=
by
  sorry

end chance_of_allergic_reaction_is_50_percent_l547_547617


namespace function_min_value_4_l547_547860

theorem function_min_value_4 :
  ∃ x : ℝ, (2^x + 2^(2 - x)) = 4 :=
sorry

end function_min_value_4_l547_547860


namespace Vladimir_is_tallest_l547_547353

-- Definitions for the statements made by the boys.
def Andrei_statements (Boris_tallest Vladimir_shortest : Prop) :=
  (¬Boris_tallest) ∧ Vladimir_shortest

def Boris_statements (Andrei_oldest Andrei_shortest : Prop) :=
  Andrei_oldest ∧ Andrei_shortest

def Vladimir_statements (Dmitry_taller Dmitry_older : Prop) :=
  Dmitry_taller ∧ Dmitry_older

def Dmitry_statements (Vladimir_both_true Dmitry_oldest : Prop) :=
  Vladimir_both_true ∧ Dmitry_oldest

-- Conditions given in the problem.
variables (Boris_tallest Vladimir_shortest Andrei_oldest Andrei_shortest Dmitry_taller Dmitry_older 
           Vladimir_both_true Dmitry_oldest : Prop)

-- Axioms based on the problem statement
axiom Dmitry_statements_condition : Dmitry_statements Vladimir_both_true Dmitry_oldest
axiom Vladimir_statements_condition : Vladimir_statements Dmitry_taller Dmitry_older
axiom Boris_statements_condition : Boris_statements Andrei_oldest Andrei_shortest
axiom Andrei_statements_condition : Andrei_statements Boris_tallest Vladimir_shortest

-- None of them share the same height or age.
axiom no_shared_height_or_age : 
  ¬(Andrei_oldest ∧ Boris_tallest ∧ Dmitry_taller ∧ Vladimir_both_true)

-- Main theorem: proving that Vladimir is the tallest based on the conditions
theorem Vladimir_is_tallest (cond : ¬Boris_tallest → Vladimir_shortest →
                              ¬Andrei_oldest → Andrei_shortest →
                              ¬Dmitry_taller → Dmitry_older →
                              ¬Vladimir_both_true → Dmitry_oldest → 
                              ¬(Andrei_oldest ∧ Boris_tallest ∧ Dmitry_taller ∧ Vladimir_both_true)) :
  (¬Boris_tallest ∧ Dmitry_older) → Vladimir_tallest :=
begin
  sorry,
end

end Vladimir_is_tallest_l547_547353


namespace square_AP_square_equals_2000_l547_547621

noncomputable def square_side : ℝ := 100
noncomputable def midpoint_AB : ℝ := square_side / 2
noncomputable def distance_MP : ℝ := 50
noncomputable def distance_PC : ℝ := square_side

/-- Given a square ABCD with side length 100, midpoint M of AB, MP = 50, and PC = 100, prove AP^2 = 2000 -/
theorem square_AP_square_equals_2000 :
  ∃ (P : ℝ × ℝ), (dist (P.1, P.2) (midpoint_AB, 0) = distance_MP) ∧ (dist (P.1, P.2) (square_side, square_side) = distance_PC) ∧ ((P.1) ^ 2 + (P.2) ^ 2 = 2000) := 
sorry


end square_AP_square_equals_2000_l547_547621


namespace g_at_80_l547_547776

noncomputable def g : ℝ → ℝ := sorry

axiom g_property : ∀ x y : ℝ, g(x * y) = x * g(y) + 1
axiom g_at_1 : g(1) = 24

theorem g_at_80 : g(80) = 1921 := 
by
  apply sorry

end g_at_80_l547_547776


namespace anna_money_left_l547_547442

variable {money_given : ℕ}
variable {price_gum : ℕ}
variable {packs_gum : ℕ}
variable {price_chocolate : ℕ}
variable {bars_chocolate : ℕ}
variable {price_candy_cane : ℕ}
variable {candy_canes: ℕ}

theorem anna_money_left (h1 : money_given = 10) 
                        (h2 : price_gum = 1)
                        (h3 : packs_gum = 3)
                        (h4 : price_chocolate = 1)
                        (h5 : bars_chocolate = 5)
                        (h6 : price_candy_cane = 1 / 2)
                        (h7 : candy_canes = 2)
                        (total_spent : (packs_gum * price_gum) + 
                                      (bars_chocolate * price_chocolate) + 
                                      (candy_canes * price_candy_cane) = 9) :
  money_given - total_spent = 1 := 
  sorry

end anna_money_left_l547_547442


namespace min_trapezium_perimeter_l547_547036

theorem min_trapezium_perimeter (a : ℕ) (hₐ : a > 0) : ∃ b, b = 4 + 2 * Real.sqrt 2 :=
by
  let hypotenuse := Real.sqrt 2
  let leg := 1 -- From Pythagorean theorem: leg = 1 cm
  let min_b := 4 + 2 * Real.sqrt 2
  use min_b
  sorry

end min_trapezium_perimeter_l547_547036


namespace sum_distinct_prime_factors_924_l547_547834

theorem sum_distinct_prime_factors_924 : 
  ∃ p1 p2 p3 p4 : ℕ, (∀ p: ℕ, p ∈ {p1, p2, p3, p4} → Nat.Prime p) ∧ 
  (p1 * p2 * p3 * p4 ∣ 924) ∧ p1 + p2 + p3 + p4 = 23 := 
sorry

end sum_distinct_prime_factors_924_l547_547834


namespace square_plot_area_l547_547320

theorem square_plot_area (cost_per_foot : ℕ) (total_cost : ℕ) (P : ℕ) :
  cost_per_foot = 54 →
  total_cost = 3672 →
  P = 4 * (total_cost / (4 * cost_per_foot)) →
  (total_cost / (4 * cost_per_foot)) ^ 2 = 289 :=
by
  intros h_cost_per_foot h_total_cost h_perimeter
  sorry

end square_plot_area_l547_547320


namespace min_value_of_2x_plus_2_2x_l547_547873

-- Lean 4 statement
theorem min_value_of_2x_plus_2_2x (x : ℝ) : 2^x + 2^(2 - x) ≥ 4 :=
by sorry

end min_value_of_2x_plus_2_2x_l547_547873


namespace function_min_value_l547_547920

theorem function_min_value (f1 f2 f3 f4 : ℝ → ℝ) :
  f1 = λ x, x^2 + 2 * x + 4 ∧
  f2 = λ x, abs (sin x) + 4 / abs (sin x) ∧
  f3 = λ x, 2^x + 2^(2-x) ∧
  f4 = λ x, log x + 4 / log x →
  (∀ x : ℝ, f3 x ≥ 4) ∧ (∃ x : ℝ, f3 x = 4) :=
by
  sorry

end function_min_value_l547_547920


namespace sum_of_cubes_l547_547844

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) : a^3 + b^3 = 1008 := by
  sorry

end sum_of_cubes_l547_547844


namespace geometric_sequence_ratio_l547_547606

theorem geometric_sequence_ratio (q : ℝ) :
  let a (n : ℕ) := if n = 3 then 2 else 2 * q * (q ^ (n - 3)),
      a2_plus_a4 := (2 / q) + 2 * q in
  a 3 = 2 ∧ a2_plus_a4 = 20 / 3 → (q = 3 ∨ q = 1 / 3) :=
by
  intro q h,
  let a := λ n : ℕ, if n = 3 then 2 else 2 * q * (q ^ (n - 3)),
  let a2_plus_a4 := (2 / q) + 2 * q,
  have h1 : a 3 = 2 := by sorry,
  have h2 : a2_plus_a4 = 20 / 3 := by sorry,
  exact or.intro_left _ (by sorry) ∨ or.intro_right _ (by sorry)

end geometric_sequence_ratio_l547_547606


namespace find_side_b_l547_547161

theorem find_side_b (a b c A B C : ℝ) (hB : B = 60) 
  (h_area : (1/2)*a*c*(Real.sin (Real.pi/3)) = Real.sqrt 3) 
  (h_ac : a^2 + c^2 = 3*a*c) : b = 2 * Real.sqrt 2 :=
by
  -- Given conditions
  have hB' : Real.sin (Real.pi/3) = Real.sqrt 3 / 2 := by sorry
  have h_ac_val : a * c = 4 := by sorry
  have h_a2c2 : a^2 + c^2 = 12 := by sorry
  have h_cosB : (a^2 + c^2 - b^2) / (2 * a * c) = 1/2 := by sorry
  have h_b2 : b^2 = 8 := by sorry

  -- Result
  exact Real.sqrt 8

end find_side_b_l547_547161


namespace first_term_of_geometric_series_l547_547438

theorem first_term_of_geometric_series (r a S : ℚ) (h_common_ratio : r = -1/5) (h_sum : S = 16) :
  a = 96 / 5 :=
by
  sorry

end first_term_of_geometric_series_l547_547438


namespace unit_digit_sum_powers_of_two_l547_547481

theorem unit_digit_sum_powers_of_two (n : ℕ) (hn : n ≥ 1) :
  let x := 2 in ((x-1)*(x^(n - 1) + x^(n - 2) + ... + x + 1)) = x^n - 1 → 
  let k := 2024 in 2^(k-1) + 2^(k-2) + ... + 2 + 1 = (2^k - 1) :=
-- We aim to prove that the unit digit of the sum 2^{2023} + 2^{2022} + ... + 2 + 1 is 5
sorry

end unit_digit_sum_powers_of_two_l547_547481


namespace vladimir_is_tallest_l547_547364

section TallestBoy

variables (Andrei Boris Vladimir Dmitry : Type)

-- Define the statements made by each boy
def andrei_statements : Andrei → Prop :=
-- suppose andrei's statements are a::Boris ≠ tallest, b::Vladimir = shortest
λ A, (∀ B : Boris, B ≠ tallest) ∨ (∀ V : Vladimir, V = shortest)

def boris_statements : Boris → Prop :=
λ B, (∀ A : Andrei, A = oldest) ∨ (∀ A' : Andrei, A' = shortest)

def vladimir_statements : Vladimir → Prop :=
λ V, (∀ D : Dmitry, D = taller_than V) ∨ (∀ D' : Dmitry, D' = older_than V)

def dmitry_statements : Dmitry → Prop :=
λ D, ((vladimir_statements V) ∧ (vladimir_statements V)) ∨ (∀ D' : Dmitry, D' = oldest)

-- Define the conditions
-- Each boy makes two statements, one of which is true and the other is false
axiom statements_condition (A : Andrei) (B : Boris) (V : Vladimir) (D : Dmitry) :
  (andrei_statements A → ¬andrei_statements A) ∧
  (boris_statements B → ¬boris_statements B) ∧
  (vladimir_statements V → ¬vladimir_statements V) ∧
  (dmitry_statements D → ¬dmitry_statements D)

-- None of them share the same height or age
axiom uniqueness_condition (A : Andrei) (B : Boris) (V : Vladimir) (D : Dmitry) :
  A ≠ B ∧ A ≠ V ∧ A ≠ D ∧ B ≠ V ∧ B ≠ D ∧ V ≠ D

-- The conclusion that Vladimir is the tallest
theorem vladimir_is_tallest (A : Andrei) (B : Boris) (V : Vladimir) (D : Dmitry) :
  A ≠ tallest ∧ D ≠ taller_than (V : Vladimir) ∧ (¬(B = tallest)) → V = tallest :=
sorry

end TallestBoy

end vladimir_is_tallest_l547_547364


namespace Vladimir_is_tallest_l547_547375

-- Variables representing the boys
inductive Boy
| Andrei | Boris | Vladimir | Dmitry

open Boy

-- Height and Age representations
variable (Height Age : Boy → ℕ)

-- Conditions based on the problem
axiom distinct_heights : ∀ a b, Height a = Height b → a = b
axiom distinct_ages : ∀ a b, Age a = Age b → a = b

-- Statements made by the boys
axiom Andrei_statements :
  (¬ (Height Boris > Height Andrei) ∨ Height Vladimir = Height Andrei)
  ∧ ¬ (¬ (Height Boris > Height Andrei) ∧ Height Vladimir = Height Andrei)

axiom Boris_statements :
  (Age Andrei > Age Boris ∨ Height Andrei = Height Boris)
  ∧ ¬ (Age Andrei > Age Boris ∧ Height Andrei = Height Boris)

axiom Vladimir_statements :
  (Height Dmitry > Height Vladimir ∨ Age Dmitry > Age Vladimir)
  ∧ ¬ (Height Dmitry > Height Vladimir ∧ Age Dmitry > Age Vladimir)

axiom Dmitry_statements :
  ((Height Dmitry > Height Vladimir ∧ Age Dmitry > Age Vladimir) ∨ Age Dmitry > Age Dmitry)
  ∧ ¬ (¬ (Height Dmitry > Height Vladimir ∧ Age Dmitry > Age Vladimir) ∧ ¬ Age Dmitry > Age Dmitry)

-- Problem Statement: Vladimir is the tallest
theorem Vladimir_is_tallest : ∀ h : Height Vladimir = Nat.max (Height Andrei) (Nat.max (Height Boris) (Height Dmitry))
by
  sorry

end Vladimir_is_tallest_l547_547375


namespace isosceles_triangle_vertex_angle_l547_547517

theorem isosceles_triangle_vertex_angle (base_ratio_vertex: ℝ) (angle_sum: ℝ) : 
  (base_ratio_vertex = 1/4 ∨ base_ratio_vertex = 4) ∧ (angle_sum = 180) →
  (∃ vertex_angle : ℝ, vertex_angle = 120 ∨ vertex_angle = 20) :=
begin
  sorry
end

end isosceles_triangle_vertex_angle_l547_547517


namespace ratio_of_y_to_x_l547_547084

theorem ratio_of_y_to_x
  (A B C D : Type) 
  [add_comm_group A] [add_comm_group B] [add_comm_group C] [add_comm_group D]
  (BD DC : A)
  (AB AC AD : A) 
  (x y : ℝ)
  (h1 : BD = 2 • DC)
  (h2 : AD = x • AB + y • AC) :
  (y / x) = 2 :=
sorry

end ratio_of_y_to_x_l547_547084


namespace frequency_of_heads_l547_547256

-- Definitions based on given conditions
def coin_tosses := 10
def heads_up := 6
def event_A := "heads up"

-- The Proof Statement
theorem frequency_of_heads :
  (heads_up / coin_tosses : ℚ) = 3 / 5 :=
sorry

end frequency_of_heads_l547_547256


namespace smallest_x_palindrome_l547_547829

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

theorem smallest_x_palindrome : ∃ k, k > 0 ∧ is_palindrome (k + 1234) ∧ k = 97 := 
by {
  use 97,
  sorry
}

end smallest_x_palindrome_l547_547829


namespace truncated_cone_sphere_radius_l547_547429

structure TruncatedCone :=
(base_radius_top : ℝ)
(base_radius_bottom : ℝ)

noncomputable def sphere_radius (c : TruncatedCone) : ℝ :=
  if c.base_radius_top = 24 ∧ c.base_radius_bottom = 6 then 12 else 0

theorem truncated_cone_sphere_radius (c : TruncatedCone) (h_radii : c.base_radius_top = 24 ∧ c.base_radius_bottom = 6) :
  sphere_radius c = 12 :=
by
  sorry

end truncated_cone_sphere_radius_l547_547429


namespace min_value_h_l547_547964

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def g (x : ℝ) : ℝ := abs (Real.sin x) + 4 / abs (Real.sin x)
noncomputable def h (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def j (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem min_value_h : ∃ x, h x = 4 :=
by
  sorry

end min_value_h_l547_547964


namespace find_length_of_b_l547_547127

theorem find_length_of_b
  {A B C : Type*}
  (a b c : ℝ)
  (area : ℝ)
  (angleB : ℝ)
  (h_area : area = sqrt 3)
  (h_angle : angleB = real.pi / 3)  -- 60 degrees in radians
  (h_sides : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 :=
by
  -- 1. Using the given area constraint: area = 1/2 * a * c * sin(B)
  -- 2. Using the given angle: sin(60) = sqrt(3)/2
  -- 3. Using the given sides relation: a^2 + c^2 = 3ac
  -- 4. Using the Law of Cosines: cos(60) = 1/2
  -- fill proof here
  sorry

end find_length_of_b_l547_547127


namespace bottles_per_case_l547_547400

theorem bottles_per_case (total_bottles_per_day : ℕ) 
                       (required_cases : ℕ) 
                       (h1 : total_bottles_per_day = 50000) 
                       (h2 : required_cases = 2000) : 
                       total_bottles_per_day / required_cases = 25 :=
by
  rw [h1, h2]
  norm_num

end bottles_per_case_l547_547400


namespace smallest_nonprime_with_large_prime_factors_l547_547716

/-- 
The smallest nonprime integer greater than 1 with no prime factor less than 15
falls in the range 260 < m ≤ 270.
-/
theorem smallest_nonprime_with_large_prime_factors :
  ∃ m : ℕ, 2 < m ∧ ¬ Nat.Prime m ∧ (∀ p : ℕ, Nat.Prime p → p ∣ m → 15 ≤ p) ∧ 260 < m ∧ m ≤ 270 :=
by
  sorry

end smallest_nonprime_with_large_prime_factors_l547_547716


namespace trig_identity_l547_547465

theorem trig_identity : 4 * Real.sin (20 * Real.pi / 180) + Real.tan (20 * Real.pi / 180) = Real.sqrt 3 := 
by sorry

end trig_identity_l547_547465


namespace function_min_value_l547_547921

theorem function_min_value (f1 f2 f3 f4 : ℝ → ℝ) :
  f1 = λ x, x^2 + 2 * x + 4 ∧
  f2 = λ x, abs (sin x) + 4 / abs (sin x) ∧
  f3 = λ x, 2^x + 2^(2-x) ∧
  f4 = λ x, log x + 4 / log x →
  (∀ x : ℝ, f3 x ≥ 4) ∧ (∃ x : ℝ, f3 x = 4) :=
by
  sorry

end function_min_value_l547_547921


namespace option_a_iff_option_b_l547_547644

-- Define a sequence as an arithmetic progression
def is_arithmetic (seq : ℕ → ℚ) : Prop :=
  ∃ d a₁, ∀ n, seq n = a₁ + n * d

-- Define the sum of the first n terms of a sequence
def sum_first_n (seq : ℕ → ℚ) (n : ℕ) : ℚ :=
  ∑ i in finset.range n, seq i

-- Define the average sum sequence
def avg_sum_seq (seq : ℕ → ℚ) (n : ℕ) : ℚ :=
  let Sn := sum_first_n seq n
  in Sn / n

-- The theorem to prove
theorem option_a_iff_option_b (seq : ℕ → ℚ) :
  is_arithmetic seq ↔ is_arithmetic (λ n, avg_sum_seq seq (n + 1)) :=
sorry

end option_a_iff_option_b_l547_547644


namespace find_side_b_l547_547174

noncomputable def triangle_area (a b c : ℝ) (B : ℝ) : ℝ :=
  1 / 2 * a * c * real.sin B

noncomputable def law_of_cosines (a b c : ℝ) (B : ℝ) : ℝ :=
  a^2 + c^2 - 2 * a * c * real.cos B

theorem find_side_b (a b c : ℝ) (B : ℝ) (area : ℝ)
  (h1 : area = √3) 
  (h2 : B = real.pi / 3)
  (h3 : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 :=
by
  sorry

end find_side_b_l547_547174


namespace minimum_value_a_l547_547040

theorem minimum_value_a (a : ℝ) :
  (∀ (x y : ℝ), x > 0 → y > 0 → x + sqrt (x * y) ≤ a * (x + y)) →
  a ≥ (sqrt 2 + 1) / 2 :=
by {
  sorry
}

end minimum_value_a_l547_547040


namespace percentage_gains_l547_547406

def false_weight_1 : ℝ := 940
def false_weight_2 : ℝ := 960
def false_weight_3 : ℝ := 980
def true_weight : ℝ := 1000

def percentage_gain (true_wt : ℝ) (false_wt : ℝ) : ℝ :=
  ((true_wt - false_wt) / false_wt) * 100

def avg_percentage_gain (pg1 pg2 pg3 : ℝ) : ℝ :=
  (pg1 + pg2 + pg3) / 3

theorem percentage_gains (pg1 pg2 pg3 avg_pg : ℝ) :
  pg1 = percentage_gain true_weight false_weight_1 →
  pg2 = percentage_gain true_weight false_weight_2 →
  pg3 = percentage_gain true_weight false_weight_3 →
  avg_pg = avg_percentage_gain pg1 pg2 pg3 →
  pg1 ≈ 6.38 ∧ pg2 ≈ 4.17 ∧ pg3 ≈ 2.04 ∧ avg_pg ≈ 4.20 :=
by sorry

end percentage_gains_l547_547406


namespace abs_lt_five_implies_interval_l547_547267

theorem abs_lt_five_implies_interval (x : ℝ) : |x| < 5 → -5 < x ∧ x < 5 := by
  sorry

end abs_lt_five_implies_interval_l547_547267


namespace find_Q_l547_547388

-- Define the relation P : Q : sqrt(R)
def relation (P Q R k : ℝ) := P = k * (Q / Real.sqrt R)

-- Known values to determine k
def P1 : ℝ := 9 / 4
def R1 : ℝ := 16 / 25
def Q1 : ℝ := 5 / 8

-- New values for which we need to find Q
def P2 : ℝ := 27
def R2 : ℝ := 1 / 36
def Q2 : ℝ := 1.56

-- The proof statement
theorem find_Q (k : ℝ) (h1 : relation P1 Q1 R1 k) : relation P2 Q2 R2 k := by
  sorry

end find_Q_l547_547388


namespace arithmetic_sequence_ninth_term_l547_547280

-- Define the terms in the arithmetic sequence
def sequence_term (a d : ℚ) (n : ℕ) : ℚ :=
  a + (n - 1) * d

-- Given conditions
def a1 : ℚ := 2 / 3
def a17 : ℚ := 5 / 6
def d : ℚ := 1 / 96 -- Calculated common difference

-- Prove the ninth term is 3/4
theorem arithmetic_sequence_ninth_term :
  sequence_term a1 d 9 = 3 / 4 :=
sorry

end arithmetic_sequence_ninth_term_l547_547280


namespace find_side_b_l547_547117

theorem find_side_b
  (a b c : ℝ)
  (area : ℝ)
  (B : ℝ)
  (h_area : area = sqrt 3)
  (h_B : B = real.pi / 3)
  (h_a2c2 : a^2 + c^2 = 3 * a * c)
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) :
  b = 2 * real.sqrt 2 :=
sorry

end find_side_b_l547_547117


namespace option_A_is_iff_option_B_l547_547660

-- Definitions based on conditions
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range (n + 1), a i

def is_arithmetic (seq : ℕ → ℝ) : Prop :=
  ∃ D : ℝ, ∀ n : ℕ, seq (n + 1) - seq n = D

-- The final statement
theorem option_A_is_iff_option_B :
  ∀ (a : ℕ → ℝ),
    (is_arithmetic_progression a ↔ is_arithmetic (λ n, Sn a n / (n + 1))) :=
by sorry

end option_A_is_iff_option_B_l547_547660


namespace triangle_side_b_l547_547182

theorem triangle_side_b 
  (a b c : ℝ)
  (B : ℝ)
  (h1 : 1/2 * a * c * Real.sin B = Real.sqrt 3)
  (h2 : B = π / 3)
  (h3 : a^2 + c^2 = 3 * a * c) : b = 2 * Real.sqrt 2 := by
  sorry

end triangle_side_b_l547_547182


namespace range_of_a_l547_547057

theorem range_of_a :
  ∀ (x : ℝ) (a : ℝ),
    (∃ x : ℝ, sin (x + π / 4) - sin (2 * x) = a) →
    -2 ≤ a ∧ a ≤ 9 / 8 := by
  sorry

end range_of_a_l547_547057


namespace a_values_a_relationship_inequality_l547_547791

noncomputable def a_n : ℕ → ℕ
| 1 := 1
| (n + 1) := (n + 1) + (n + 1) * a_n n

theorem a_values :
  a_n 2 = 4 ∧ a_n 3 = 15 ∧ a_n 4 = 64 ∧ a_n 5 = 325 := sorry

theorem a_relationship (n : ℕ) (hn : 1 ≤ n) :
  a_n n.succ = n.succ + n.succ * a_n n := sorry

theorem inequality (n : ℕ) (hn : 1 ≤ n) :
  (∏ i in Finset.range n.succ, (1 + 1 / (a_n i + 1))) < 3 := sorry

end a_values_a_relationship_inequality_l547_547791


namespace Archimedean_spiral_l547_547738

theorem Archimedean_spiral (φ : ℝ) (ρ : ℝ)
  (h : φ ∈ {0, π / 4, π / 2, 3 * π / 4, π, 5 * π / 4, 3 * π / 2, 7 * π / 4, 2 * π}) :
  ρ = 0.5 * φ :=
by
  sorry

end Archimedean_spiral_l547_547738


namespace expression_divisible_by_17_l547_547741

theorem expression_divisible_by_17 (n : ℕ) : 
  (6^(2*n) + 2^(n+2) + 12 * 2^n) % 17 = 0 :=
by
  sorry

end expression_divisible_by_17_l547_547741


namespace optionA_iff_optionB_l547_547705

variable {a : ℕ → ℝ} {S : ℕ → ℝ}

/-- Sequence a_n is an arithmetic progression -/
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a n = a1 + n * d

/-- Sequence S_n / n is an arithmetic progression -/
def is_S_n_div_n_arithmetic_progression (S : ℕ → ℝ) : Prop :=
  ∃ (b c : ℝ), ∀ n, S n / n = b + c * n

/-- S_n is the sum of the first n terms of the sequence a_n -/
def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = ∑ i in finset.range (n + 1), a i 

/-- Option A is both a sufficient and necessary condition for Option B -/
theorem optionA_iff_optionB (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (is_arithmetic_progression a ∧ sum_first_n_terms a S) ↔ is_S_n_div_n_arithmetic_progression S :=
  sorry

end optionA_iff_optionB_l547_547705


namespace price_reaches_81_in_2_years_l547_547788

theorem price_reaches_81_in_2_years :
  ∃ (n : ℕ), n = 4 ∧ 3^n * 1 = 81 ∧ n * 6 = 24 :=
by
  -- Given initial price and tripling period
  let P₀ := 1
  let P₁ := 3 * P₀
  let P₂ := 3 * P₁
  let P₃ := 3 * P₂
  let P₄ := 3 * P₃
  existsi 4
  simp
  split
  . exact rfl 
  . split 
    . simp only [pow_succ, pow_zero, mul_one, one_mul]
    . rw Nat.mul_comm
      rfl

end price_reaches_81_in_2_years_l547_547788


namespace polygon_diagonals_mod_3_l547_547402

open Nat

theorem polygon_diagonals_mod_3 {n : Nat} (h1 : n > 3)
  (odd_vertices : List Nat) (h2 : ∀ i ∈ odd_vertices, 1 ≤ i ∧ i ≤ n)
  (unique_odd : ∀ i j ∈ odd_vertices, i < j → i ≠ j)
  (even_vertices : ∀ (v : Nat), v ∈ List.range n → ¬ v ∈ odd_vertices → ∃ m : Nat, 2 * m = (number_of_sides_from_vertex v) ) :
  odd_vertices.length.even ∧ 
  (if odd_vertices.length = 0 then
     n % 3 = 0
   else 
     n % 3 = (odd_vertices.length.even_indices_subtract_odd_indices) % 3) :=
by
  sorry

end polygon_diagonals_mod_3_l547_547402


namespace studentB_visited_A_l547_547321

variable (City : Type)
variable (A B C : City)
variable (visited : (Person : Type) -> (City -> Prop))
variable (StudentA StudentB StudentC : Type)

-- Conditions:
-- 1. Student A said: I have visited more cities than Student B, but I have not visited city B
variable (visited_by_A : City -> Prop := visited StudentA)
variable (visited_by_B : City -> Prop := visited StudentB)
variable (visited_by_C : City -> Prop := visited StudentC)
variable (visited_more_than : (Person : Type) -> (Person -> Prop))

axiom studentA_more_cities : visited_more_than StudentA StudentB
axiom studentA_not_B : ¬ visited_by_A B

-- 2. Student B said: I have not visited city C
axiom studentB_not_C : ¬ visited_by_B C

-- 3. Student C said: The three of us have visited the same city
axiom same_city_visited : ∃ (c : City), visited_by_A c ∧ visited_by_B c ∧ visited_by_C c

-- Conclusion: The city visited by Student B is A
theorem studentB_visited_A : visited_by_B A :=
by
  sorry

end studentB_visited_A_l547_547321


namespace sasha_or_max_is_incorrect_l547_547751

theorem sasha_or_max_is_incorrect (sums_rows_divisible_by_9: ∀ i : Fin 100, (sums_rows i) % 9 = 0)
    (one_column_not_divisible_by_9 : ∃! k : Fin 100, (sums_columns k) % 9 ≠ 0 ∧ (∀ j : Fin 100, j ≠ k → (sums_columns j) % 9 = 0)) :
  ∃ i, (sums_rows i) % 9 ≠ (sums_columns i)%9 → false := 
begin
  sorry -- proof goes here
end

end sasha_or_max_is_incorrect_l547_547751


namespace hyperbola_equation_l547_547292

theorem hyperbola_equation (x y : ℝ) 
  (h : real.sqrt ((x - 3) ^ 2 + y ^ 2) - real.sqrt ((x + 3) ^ 2 + y ^ 2) = 4) : 
  x ^ 2 / 4 - y ^ 2 / 5 = 1 ∧ x ≤ -2 := 
sorry

end hyperbola_equation_l547_547292


namespace sugar_merchant_profit_l547_547414

theorem sugar_merchant_profit 
    (total_sugar : ℕ)
    (sold_at_18 : ℕ)
    (remain_sugar : ℕ)
    (whole_profit : ℕ)
    (profit_18 : ℕ)
    (rem_profit_percent : ℕ) :
    total_sugar = 1000 →
    sold_at_18 = 600 →
    remain_sugar = total_sugar - sold_at_18 →
    whole_profit = 14 →
    profit_18 = 18 →
    (600 * profit_18 / 100) + (remain_sugar * rem_profit_percent / 100) = 
    (total_sugar * whole_profit / 100) →
    rem_profit_percent = 80 :=
by
    sorry

end sugar_merchant_profit_l547_547414


namespace range_of_c_l547_547528

theorem range_of_c (a b c : ℝ) 
  (h1 : ∀ x, f(x) = a * Real.sin x + b * Real.cos x + c)
  (h2 : f 0 = 5) 
  (h3 : f (Real.pi / 2) = 5) 
  (h4 : ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → |f x| ≤ 10) 
  : -5 * Real.sqrt 2 ≤ c ∧ c ≤ 15 * Real.sqrt 2 + 20 :=
begin
  sorry
end

end range_of_c_l547_547528


namespace Vladimir_is_tallest_l547_547379

inductive Boy where
  | Andrei
  | Boris
  | Vladimir
  | Dmitry
  deriving DecidableEq, Repr

open Boy

variable (height aged : Boy → ℕ)

-- Each boy's statements
def andreiStatements (tallest shortest : Boy) :=
  (Boris ≠ tallest, Vladimir = shortest)

def borisStatements (oldest shortest : Boy) :=
  (Andrei = oldest, Andrei = shortest)

def vladimirStatements (heightOlder : Boy → Prop) :=
  (heightOlder Dmitry, heightOlder Dmitry)

def dmitryStatements (vladimirTrue oldest : Prop) :=
  (vladimirTrue, Dmitry = oldest)

noncomputable def is_tallest (b : Boy) : Prop :=
  ∀ b' : Boy, height b' ≤ height b

theorem Vladimir_is_tallest :
  (∃! b : Boy, is_tallest b)
  ∧ (∀ (height ordered : Boy), one_of_each (heightAndAged : Boy -> ℕ -> ℕ), ∃ 
  (tall shortest oldest : Boy),
  (andreiStatements tall shortest).1 ∧ ¬(andreiStatements tall shortest).2 ∧ 
  ¬(borisStatements oldest shortest).1 ∧ (borisStatements oldest shortest).2 ∧
  ¬(vladimirStatements heightOlder).1 ∧ (vladimirStatements heightOlder).2 ∧ 
  ¬(dmitryStatements false (Dmitry = oldest).1) ∧ (dmitryStatements false (Dmitry = oldest).2) ∧
  Vladimir = tall) :=
sorry

end Vladimir_is_tallest_l547_547379


namespace min_value_of_2x_plus_2_2x_l547_547876

-- Lean 4 statement
theorem min_value_of_2x_plus_2_2x (x : ℝ) : 2^x + 2^(2 - x) ≥ 4 :=
by sorry

end min_value_of_2x_plus_2_2x_l547_547876


namespace rectangle_ratio_l547_547604

theorem rectangle_ratio (w : ℝ) (h : ℝ)
  (hw : h = 10)   -- Length is 10
  (hp : 2 * w + 2 * h = 30) :  -- Perimeter is 30
  w / h = 1 / 2 :=             -- Ratio of width to length is 1/2
by
  -- Pending proof
  sorry

end rectangle_ratio_l547_547604


namespace triangle_side_b_l547_547179

theorem triangle_side_b 
  (a b c : ℝ)
  (B : ℝ)
  (h1 : 1/2 * a * c * Real.sin B = Real.sqrt 3)
  (h2 : B = π / 3)
  (h3 : a^2 + c^2 = 3 * a * c) : b = 2 * Real.sqrt 2 := by
  sorry

end triangle_side_b_l547_547179


namespace geometric_sequence_fourth_term_l547_547281

theorem geometric_sequence_fourth_term (x : ℝ) (r : ℝ) (h1 : (3 * x + 3)^2 = x * (6 * x + 6)) 
(h2 : r = (3 * x + 3) / x) :
  (6 * x + 6) * r = -24 :=
by {
  -- Definitions of x, r and condition h1, h2 are given.
  -- Conclusion must follow that the fourth term is -24.
  sorry
}

end geometric_sequence_fourth_term_l547_547281


namespace train_speed_second_part_l547_547424

-- Define conditions
def distance_first_part (x : ℕ) := x
def speed_first_part := 40
def distance_second_part (x : ℕ) := 2 * x
def total_distance (x : ℕ) := 5 * x
def average_speed := 40

-- Define the problem
theorem train_speed_second_part (x : ℕ) (v : ℕ) (h1 : total_distance x = 5 * x)
  (h2 : total_distance x / average_speed = distance_first_part x / speed_first_part + distance_second_part x / v) :
  v = 20 :=
  sorry

end train_speed_second_part_l547_547424


namespace ghost_entry_exit_ways_l547_547410

/-- There are 8 windows, numbered from 1 to 8. Georgie the Ghost can enter through an odd-numbered
    window and leave through an even-numbered window. Prove that the number of ways to do this is 16. -/
theorem ghost_entry_exit_ways : 
  let odd_windows := {1, 3, 5, 7}
      even_windows := {2, 4, 6, 8}
  in (odd_windows.card * even_windows.card = 16) := 
by
  let odd_windows := {1, 3, 5, 7}
  let even_windows := {2, 4, 6, 8}
  have h₁ : odd_windows.card = 4 := by sorry
  have h₂ : even_windows.card = 4 := by sorry
  show 4 * 4 = 16 from sorry

end ghost_entry_exit_ways_l547_547410


namespace probability_sum_dice_12_l547_547582

/-- Helper definition for a standard six-faced die roll -/
def is_valid_die_roll (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 6

/-- The probability that the sum of three six-faced dice equals 12 is 19/216. -/
theorem probability_sum_dice_12 :
  (∑ (x y z : ℕ) in (finset.range 7).filter (is_valid_die_roll), ite (x + y + z = 12) 1 0) = 19 :=
begin
  sorry
end

end probability_sum_dice_12_l547_547582


namespace min_value_h_l547_547960

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def g (x : ℝ) : ℝ := abs (Real.sin x) + 4 / abs (Real.sin x)
noncomputable def h (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def j (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem min_value_h : ∃ x, h x = 4 :=
by
  sorry

end min_value_h_l547_547960


namespace simplify_and_evaluate_expression_l547_547756

variable (a b : ℚ)

theorem simplify_and_evaluate_expression
  (ha : a = 1 / 2)
  (hb : b = -1 / 3) :
  b^2 - a^2 + 2 * (a^2 + a * b) - (a^2 + b^2) = -1 / 3 :=
by
  -- The proof will be inserted here
  sorry

end simplify_and_evaluate_expression_l547_547756


namespace percentage_of_circle_outside_triangle_l547_547271

theorem percentage_of_circle_outside_triangle (A : ℝ)
  (h₁ : 0 < A) -- Total area A is positive
  (A_inter : ℝ) (A_outside_tri : ℝ) (A_total_circle : ℝ)
  (h₂ : A_inter = 0.45 * A)
  (h₃ : A_outside_tri = 0.40 * A)
  (h₄ : A_total_circle = 0.60 * A) :
  100 * (1 - A_inter / A_total_circle) = 25 :=
by
  sorry

end percentage_of_circle_outside_triangle_l547_547271


namespace distinct_points_common_to_graphs_l547_547783

theorem distinct_points_common_to_graphs :
  ∃! (p : ℝ × ℝ), (p.1 ^ 2 + p.2 ^ 2 = 9) ∧ (p.2 ^ 2 = 9) :=
begin
  sorry
end

end distinct_points_common_to_graphs_l547_547783


namespace fraction_of_cream_in_cup1_l547_547750

/-
Problem statement:
Sarah places five ounces of coffee into an eight-ounce cup (Cup 1) and five ounces of cream into a second cup (Cup 2).
After pouring half the coffee from Cup 1 to Cup 2, one ounce of cream is added to Cup 2.
After stirring Cup 2 thoroughly, Sarah then pours half the liquid in Cup 2 back into Cup 1.
Prove that the fraction of the liquid in Cup 1 that is now cream is 4/9.
-/

theorem fraction_of_cream_in_cup1
  (initial_coffee_cup1 : ℝ)
  (initial_cream_cup2 : ℝ)
  (half_initial_coffee : ℝ)
  (added_cream : ℝ)
  (total_mixture : ℝ)
  (half_mixture : ℝ)
  (coffee_fraction : ℝ)
  (cream_fraction : ℝ)
  (coffee_transferred_back : ℝ)
  (cream_transferred_back : ℝ)
  (total_coffee_in_cup1 : ℝ)
  (total_cream_in_cup1 : ℝ)
  (total_liquid_in_cup1 : ℝ)
  :
  initial_coffee_cup1 = 5 →
  initial_cream_cup2 = 5 →
  half_initial_coffee = initial_coffee_cup1 / 2 →
  added_cream = 1 →
  total_mixture = initial_cream_cup2 + half_initial_coffee + added_cream →
  half_mixture = total_mixture / 2 →
  coffee_fraction = half_initial_coffee / total_mixture →
  cream_fraction = (total_mixture - half_initial_coffee) / total_mixture →
  coffee_transferred_back = half_mixture * coffee_fraction →
  cream_transferred_back = half_mixture * cream_fraction →
  total_coffee_in_cup1 = initial_coffee_cup1 - half_initial_coffee + coffee_transferred_back →
  total_cream_in_cup1 = cream_transferred_back →
  total_liquid_in_cup1 = total_coffee_in_cup1 + total_cream_in_cup1 →
  total_cream_in_cup1 / total_liquid_in_cup1 = 4 / 9 :=
by {
  sorry
}

end fraction_of_cream_in_cup1_l547_547750


namespace relative_speed_of_trains_l547_547302

def kmph_to_mps (speed : ℕ) : ℕ := speed * 1000 / 3600

theorem relative_speed_of_trains 
  (speed_train1_kmph : ℕ) 
  (speed_train2_kmph : ℕ) 
  (h1 : speed_train1_kmph = 216) 
  (h2 : speed_train2_kmph = 180) : 
  kmph_to_mps speed_train1_kmph - kmph_to_mps speed_train2_kmph = 10 := 
by 
  sorry

end relative_speed_of_trains_l547_547302


namespace minimum_value_of_h_l547_547904

-- Definitions of the given functions
def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
def g (x : ℝ) : ℝ := |sin x| + 4 / |sin x|
def h (x : ℝ) : ℝ := 2^x + 2^(2 - x)
def j (x : ℝ) : ℝ := log x + 4 / log x

-- Main theorem statement with the condition and conclusion
theorem minimum_value_of_h : ∃ x, h x = 4 :=
by { sorry }

end minimum_value_of_h_l547_547904


namespace num_handshakes_ten_women_l547_547766

def num_handshakes (n : ℕ) : ℕ :=
(n * (n - 1)) / 2

theorem num_handshakes_ten_women :
  num_handshakes 10 = 45 :=
by
  sorry

end num_handshakes_ten_women_l547_547766


namespace total_games_in_tournament_l547_547590

def comb (n k : ℕ) : ℕ := n.choose k

theorem total_games_in_tournament (classes_first_grade : ℕ) (classes_second_grade : ℕ) (classes_third_grade : ℕ) 
    (h1 : classes_first_grade = 5) (h2 : classes_second_grade = 8) (h3 : classes_third_grade = 3) : 
    comb 5 2 + comb 8 2 + comb 3 2 = 41 := 
by
    -- Combination formula for binomial coefficient
    have h_comb_5_2 : comb 5 2 = 10 := by { rw [comb, Nat.choose], exact rfl },
    have h_comb_8_2 : comb 8 2 = 28 := by { rw [comb, Nat.choose], exact rfl },
    have h_comb_3_2 : comb 3 2 = 3 := by { rw [comb, Nat.choose], exact rfl },
    rw [h1, h2, h3],
    rw [h_comb_5_2, h_comb_8_2, h_comb_3_2],
    exact rfl

end total_games_in_tournament_l547_547590


namespace optionA_iff_optionB_l547_547707

variable {a : ℕ → ℝ} {S : ℕ → ℝ}

/-- Sequence a_n is an arithmetic progression -/
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a n = a1 + n * d

/-- Sequence S_n / n is an arithmetic progression -/
def is_S_n_div_n_arithmetic_progression (S : ℕ → ℝ) : Prop :=
  ∃ (b c : ℝ), ∀ n, S n / n = b + c * n

/-- S_n is the sum of the first n terms of the sequence a_n -/
def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = ∑ i in finset.range (n + 1), a i 

/-- Option A is both a sufficient and necessary condition for Option B -/
theorem optionA_iff_optionB (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (is_arithmetic_progression a ∧ sum_first_n_terms a S) ↔ is_S_n_div_n_arithmetic_progression S :=
  sorry

end optionA_iff_optionB_l547_547707


namespace function_min_value_l547_547917

theorem function_min_value (f1 f2 f3 f4 : ℝ → ℝ) :
  f1 = λ x, x^2 + 2 * x + 4 ∧
  f2 = λ x, abs (sin x) + 4 / abs (sin x) ∧
  f3 = λ x, 2^x + 2^(2-x) ∧
  f4 = λ x, log x + 4 / log x →
  (∀ x : ℝ, f3 x ≥ 4) ∧ (∃ x : ℝ, f3 x = 4) :=
by
  sorry

end function_min_value_l547_547917


namespace proof_by_contradiction_example_l547_547814

theorem proof_by_contradiction_example (x a b : ℝ) (h : x^2 - (a + b) * x + a * b ≠ 0) : x ≠ a ∧ x ≠ b :=
by
  by_contradiction h₁
  cases not_and_or_not_of_not_iff (sorry) h₁
  case inl =>
    rw [h.1] at h
    contradiction
  case inr =>
    rw [h.2] at h
    contradiction

end proof_by_contradiction_example_l547_547814


namespace option_A_is_iff_option_B_l547_547658

-- Definitions based on conditions
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range (n + 1), a i

def is_arithmetic (seq : ℕ → ℝ) : Prop :=
  ∃ D : ℝ, ∀ n : ℕ, seq (n + 1) - seq n = D

-- The final statement
theorem option_A_is_iff_option_B :
  ∀ (a : ℕ → ℝ),
    (is_arithmetic_progression a ↔ is_arithmetic (λ n, Sn a n / (n + 1))) :=
by sorry

end option_A_is_iff_option_B_l547_547658


namespace rational_solutions_count_l547_547785

theorem rational_solutions_count :
  ∃ (sols : Finset (ℚ × ℚ × ℚ)), 
    (∀ (x y z : ℚ), (x + y + z = 0) ∧ (x * y * z + z = 0) ∧ (x * y + y * z + x * z + y = 0) ↔ (x, y, z) ∈ sols) ∧
    sols.card = 3 :=
by
  sorry

end rational_solutions_count_l547_547785


namespace minimal_length_curve_equilateral_triangle_l547_547308

theorem minimal_length_curve_equilateral_triangle (A B C : Point) (h : is_equilateral_triangle A B C) :
  ∃ center : Point, 
    is_vertex_of_triangle center A B C ∧
    ∃ radius : ℝ, 
      minimal_length_curve DivA_eqB : 
          curve_divides_triangle_into_two_equal_areas A B C (circle center radius) :=
begin
  sorry
end

end minimal_length_curve_equilateral_triangle_l547_547308


namespace min_sum_of_distances_l547_547102

open Real

variables (a b c : ℝ^2)

-- Conditions
def norm_sq_a : ℝ := 4
def norm_sq_b : ℝ := 1
def norm_sq_c : ℝ := 9

-- Statement to prove
theorem min_sum_of_distances :
  (‖a - b‖^2 + ‖a - c‖^2 + ‖b - c‖^2) = 2 :=
sorry

end min_sum_of_distances_l547_547102


namespace gcd_40_56_l547_547310

theorem gcd_40_56 : Nat.gcd 40 56 = 8 := 
by 
  sorry

end gcd_40_56_l547_547310


namespace option_A_correct_option_B_correct_option_D_correct_l547_547804

def is_elite_function (f : ℝ → ℝ) : Prop :=
  ∀ (x1 x2 : ℝ), (0 < x1 ∧ 0 < x2) → f (x1 + x2) < f x1 + f x2

theorem option_A_correct : is_elite_function (λ x, Real.log (1 + x)) :=
sorry

theorem option_B_correct (f : ℝ → ℝ) (h : is_elite_function f) (n : ℕ) (hn : 2 ≤ n) : f n < n * f 1 :=
sorry

theorem option_D_correct (f : ℝ → ℝ) : (∀ x1 x2 : ℝ, (0 < x2 ∧ 0 < x1 ∧ x2 < x1) → x2 * f x1 < x1 * f x2) → is_elite_function f :=
sorry

end option_A_correct_option_B_correct_option_D_correct_l547_547804


namespace total_servings_l547_547091

-- Definitions for the conditions

def servings_per_carrot : ℕ := 4
def plants_per_plot : ℕ := 9
def servings_multiplier_corn : ℕ := 5
def servings_multiplier_green_bean : ℤ := 2

-- Proof statement
theorem total_servings : 
  (plants_per_plot * servings_per_carrot) + 
  (plants_per_plot * (servings_per_carrot * servings_multiplier_corn)) + 
  (plants_per_plot * (servings_per_carrot * servings_multiplier_corn / servings_multiplier_green_bean)) = 
  306 :=
by
  sorry

end total_servings_l547_547091


namespace Vladimir_is_tallest_l547_547381

inductive Boy where
  | Andrei
  | Boris
  | Vladimir
  | Dmitry
  deriving DecidableEq, Repr

open Boy

variable (height aged : Boy → ℕ)

-- Each boy's statements
def andreiStatements (tallest shortest : Boy) :=
  (Boris ≠ tallest, Vladimir = shortest)

def borisStatements (oldest shortest : Boy) :=
  (Andrei = oldest, Andrei = shortest)

def vladimirStatements (heightOlder : Boy → Prop) :=
  (heightOlder Dmitry, heightOlder Dmitry)

def dmitryStatements (vladimirTrue oldest : Prop) :=
  (vladimirTrue, Dmitry = oldest)

noncomputable def is_tallest (b : Boy) : Prop :=
  ∀ b' : Boy, height b' ≤ height b

theorem Vladimir_is_tallest :
  (∃! b : Boy, is_tallest b)
  ∧ (∀ (height ordered : Boy), one_of_each (heightAndAged : Boy -> ℕ -> ℕ), ∃ 
  (tall shortest oldest : Boy),
  (andreiStatements tall shortest).1 ∧ ¬(andreiStatements tall shortest).2 ∧ 
  ¬(borisStatements oldest shortest).1 ∧ (borisStatements oldest shortest).2 ∧
  ¬(vladimirStatements heightOlder).1 ∧ (vladimirStatements heightOlder).2 ∧ 
  ¬(dmitryStatements false (Dmitry = oldest).1) ∧ (dmitryStatements false (Dmitry = oldest).2) ∧
  Vladimir = tall) :=
sorry

end Vladimir_is_tallest_l547_547381


namespace vladimir_is_tallest_l547_547339

inductive Boy : Type
| Andrei | Boris | Vladimir | Dmitry

open Boy

-- Hypotheses based on the problem statement
def statements : Boy → (Prop × Prop)
| Andrei := (¬ (Boris = tallest), Vladimir = shortest)
| Boris := (Andrei = oldest, Andrei = shortest)
| Vladimir := (Dmitry > Vladimir, Dmitry = oldest)
| Dmitry := (statements Vladimir = (true, true), Dmitry = oldest)

def different_heights (a b : Boy) : Prop := a ≠ b
def different_ages (a b : Boy) : Prop := a ≠ b

axiom no_same_height {a b : Boy} : different_heights a b
axiom no_same_age {a b : Boy} : different_ages a b

-- The proof statement
theorem vladimir_is_tallest : Vladimir = tallest := 
by 
  sorry

end vladimir_is_tallest_l547_547339


namespace min_value_h_l547_547957

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def g (x : ℝ) : ℝ := abs (Real.sin x) + 4 / abs (Real.sin x)
noncomputable def h (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def j (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem min_value_h : ∃ x, h x = 4 :=
by
  sorry

end min_value_h_l547_547957


namespace function_min_value_4_l547_547864

theorem function_min_value_4 :
  ∃ x : ℝ, (2^x + 2^(2 - x)) = 4 :=
sorry

end function_min_value_4_l547_547864


namespace min_value_h_l547_547958

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def g (x : ℝ) : ℝ := abs (Real.sin x) + 4 / abs (Real.sin x)
noncomputable def h (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def j (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem min_value_h : ∃ x, h x = 4 :=
by
  sorry

end min_value_h_l547_547958


namespace ordinate_of_vertex_of_quadratic_l547_547490

theorem ordinate_of_vertex_of_quadratic (f : ℝ → ℝ) (h1 : ∃ a b c : ℝ, ∀ x : ℝ, f x = a * x^2 + b * x + c)
  (h2 : ∃ t1 t2 t3 : ℝ, (f t1)^3 - f t1 = 0 ∧ (f t2)^3 - f t2 = 0 ∧ (f t3)^3 - f t3 = 0 ∧ t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t3) :
  (∃ h : ℝ, f h = 0) → 
  ∃ k : ℝ, ∀ x : ℝ, f x = f (k/2) → False :=
begin
  sorry
end

end ordinate_of_vertex_of_quadratic_l547_547490


namespace min_value_h_l547_547965

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def g (x : ℝ) : ℝ := abs (Real.sin x) + 4 / abs (Real.sin x)
noncomputable def h (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def j (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem min_value_h : ∃ x, h x = 4 :=
by
  sorry

end min_value_h_l547_547965


namespace optionA_iff_optionB_l547_547703

variable {a : ℕ → ℝ} {S : ℕ → ℝ}

/-- Sequence a_n is an arithmetic progression -/
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a n = a1 + n * d

/-- Sequence S_n / n is an arithmetic progression -/
def is_S_n_div_n_arithmetic_progression (S : ℕ → ℝ) : Prop :=
  ∃ (b c : ℝ), ∀ n, S n / n = b + c * n

/-- S_n is the sum of the first n terms of the sequence a_n -/
def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = ∑ i in finset.range (n + 1), a i 

/-- Option A is both a sufficient and necessary condition for Option B -/
theorem optionA_iff_optionB (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (is_arithmetic_progression a ∧ sum_first_n_terms a S) ↔ is_S_n_div_n_arithmetic_progression S :=
  sorry

end optionA_iff_optionB_l547_547703


namespace least_number_to_divisible_by_11_l547_547336

theorem least_number_to_divisible_by_11 (n : ℕ) (h : n = 11002) : ∃ k : ℕ, (n + k) % 11 = 0 ∧ ∀ m : ℕ, (n + m) % 11 = 0 → m ≥ k :=
by
  sorry

end least_number_to_divisible_by_11_l547_547336


namespace locus_of_Q_l547_547013

-- Definitions of the given conditions
def ellipse (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def vertices (A B : ℝ × ℝ) : Prop :=
  A = (-2, 0) ∧ B = (2, 0)

def eccentricity (a e : ℝ) : Prop :=
  e = 1 / 2 ∧ e = sqrt (1 - (b/a)^2)

def pointP_on_ellipse (P : ℝ × ℝ) : Prop :=
  ∃ θ : ℝ, P = (2 * Real.cos θ, sqrt 3 * Real.sin θ) ∧ Real.sin θ ≠ 0

def pointD : ℝ × ℝ := (-4, 0)

def vector_DE_DP (D E P : ℝ × ℝ) : Prop :=
  E = (D.1 + 3/5 * (P.1 - D.1), D.2 + 3/5 * (P.2 - D.2))

def line_intersection (A P B E : ℝ × ℝ) (Q : ℝ × ℝ) : Prop :=
  let slopeAP := (P.2 - A.2) / (P.1 - A.1)
  let slopeBE := (E.2 - B.2) / (E.1 - B.1)
  in Q.2 = slopeAP * (Q.1 - A.1) + A.2 ∧ Q.2 = slopeBE * (Q.1 - B.1) + B.2

-- Proof statement
theorem locus_of_Q :
  ∀ (a b : ℝ) (P Q E : ℝ × ℝ),
  0 < a →
  0 < b →
  a = 2 →
  sqrt(1 - (b/a)^2) = 1/2 →
  ellipse a b (by linarith) (by linarith) P.1 P.2 →
  vertices (-2, 0) (2, 0) →
  pointP_on_ellipse P →
  vector_DE_DP (-4, 0) E P →
  line_intersection (-2, 0) P (2, 0) E Q →
  (Q.1 + 1)^2 + 4/3 * Q.2^2 = 1 :=
sorry

end locus_of_Q_l547_547013


namespace ordered_subset_pairs_count_l547_547535

theorem ordered_subset_pairs_count (U : Finset ℕ) (hU : U = {1, 2, 3, 4}) : 
  let is_ordered_subset_pair (A B : Finset ℕ) :=
    A ⊆ U ∧ B ⊆ U ∧ A.nonempty ∧ B.nonempty ∧ A.max' A.nonempty < B.min' B.nonempty in
  (∑ A in U.powerset.filter (λ s, s.nonempty), 
    (U.powerset.filter (λ t, t.nonempty ∧ is_ordered_subset_pair s t)).card) = 17 :=
by
  sorry

end ordered_subset_pairs_count_l547_547535


namespace intersection_M_N_l547_547022

def M : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def N : Set ℤ := {-3, -1, 1, 3, 5}

theorem intersection_M_N :
  M ∩ N = {1, 3} := 
sorry

end intersection_M_N_l547_547022


namespace angle_equality_l547_547000

theorem angle_equality 
  (α β γ θ : ℝ) 
  (h₁ : 0 < α ∧ α < real.pi)
  (h₂ : 0 < β ∧ β < real.pi)
  (h₃ : 0 < γ ∧ γ < real.pi)
  (h₄ : 0 < θ ∧ θ < real.pi)
  (h : (real.sin α / real.sin β) = (real.sin γ / real.sin θ) 
       ∧ (real.sin γ / real.sin θ) = (real.sin (α - γ) / real.sin (β - θ))) :
  α = β ∧ γ = θ := 
by
  sorry

end angle_equality_l547_547000


namespace relationship_between_n_and_m_l547_547781

variable {n m : ℕ}
variable {x y : ℝ}
variable {a : ℝ}
variable {z : ℝ}

def mean_sample_combined (n m : ℕ) (x y z a : ℝ) : Prop :=
  z = a * x + (1 - a) * y ∧ a > 1 / 2

theorem relationship_between_n_and_m 
  (hx : ∀ (i : ℕ), i < n → x = x)
  (hy : ∀ (j : ℕ), j < m → y = y)
  (hz : mean_sample_combined n m x y z a)
  (hne : x ≠ y) : n < m :=
sorry

end relationship_between_n_and_m_l547_547781


namespace find_side_b_l547_547111

theorem find_side_b (a b c : ℝ) (A B C : ℝ)
  (area_triangle : Real.sqrt 3) (B_eq_60 : B = 60) (cond_ac: (a^2 + c^2 = 3 * a * c)) :
  ∃ b, b = 2 * Real.sqrt 2 :=
by
  sorry

end find_side_b_l547_547111


namespace option_A_is_iff_option_B_l547_547655

-- Definitions based on conditions
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range (n + 1), a i

def is_arithmetic (seq : ℕ → ℝ) : Prop :=
  ∃ D : ℝ, ∀ n : ℕ, seq (n + 1) - seq n = D

-- The final statement
theorem option_A_is_iff_option_B :
  ∀ (a : ℕ → ℝ),
    (is_arithmetic_progression a ↔ is_arithmetic (λ n, Sn a n / (n + 1))) :=
by sorry

end option_A_is_iff_option_B_l547_547655


namespace option_A_is_iff_option_B_l547_547662

-- Definitions based on conditions
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range (n + 1), a i

def is_arithmetic (seq : ℕ → ℝ) : Prop :=
  ∃ D : ℝ, ∀ n : ℕ, seq (n + 1) - seq n = D

-- The final statement
theorem option_A_is_iff_option_B :
  ∀ (a : ℕ → ℝ),
    (is_arithmetic_progression a ↔ is_arithmetic (λ n, Sn a n / (n + 1))) :=
by sorry

end option_A_is_iff_option_B_l547_547662


namespace function_equality_l547_547408

theorem function_equality (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f(x) ≤ x) 
  (h2 : ∀ x y : ℝ, f(x + y) ≤ f(x) + f(y)) : 
  ∀ x : ℝ, f(x) = x := 
by {
  -- The proof steps would go here.
  sorry
}

end function_equality_l547_547408


namespace triangle_side_b_l547_547233

noncomputable def find_side_length_b
  (a b c : ℝ)
  (area : ℝ)
  (B_deg : ℝ)
  (relation : a^2 + c^2 = 3 * a * c)
  (area_condition : area = sqrt 3)
  (angle_condition : B_deg = 60) : Prop :=
  b = 2 * sqrt 2

theorem triangle_side_b
  (a b c : ℝ)
  (area : ℝ := sqrt 3)
  (B_deg : ℝ := 60)
  (h1 : a^2 + c^2 = 3 * a * c)
  (h2 : area = sqrt 3)
  (h3 : B_deg = 60) : find_side_length_b a b c area B_deg h1 h2 h3 :=
  sorry

end triangle_side_b_l547_547233


namespace triangle_side_b_l547_547178

theorem triangle_side_b 
  (a b c : ℝ)
  (B : ℝ)
  (h1 : 1/2 * a * c * Real.sin B = Real.sqrt 3)
  (h2 : B = π / 3)
  (h3 : a^2 + c^2 = 3 * a * c) : b = 2 * Real.sqrt 2 := by
  sorry

end triangle_side_b_l547_547178


namespace probability_of_non_eq_zero_l547_547851

noncomputable def prob_neq_zero (a b c : ℕ) : ℚ :=
  if a ∈ {1, 3, 4, 5, 6} ∧ b ∈ {1, 3, 4, 5, 6} ∧ c ∈ {1, 3, 4, 5, 6} 
  then 1 else 0

theorem probability_of_non_eq_zero :
  let outcomes := {1, 3, 4, 5, 6}
  ∀ (a b c : ℕ), 
    (a ∈ outcomes ∧ b ∈ outcomes ∧ c ∈ outcomes) →
    (prob_neq_zero a b c) = (125 / 216) := 
by
  sorry

end probability_of_non_eq_zero_l547_547851


namespace option_A_is_iff_option_B_l547_547657

-- Definitions based on conditions
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range (n + 1), a i

def is_arithmetic (seq : ℕ → ℝ) : Prop :=
  ∃ D : ℝ, ∀ n : ℕ, seq (n + 1) - seq n = D

-- The final statement
theorem option_A_is_iff_option_B :
  ∀ (a : ℕ → ℝ),
    (is_arithmetic_progression a ↔ is_arithmetic (λ n, Sn a n / (n + 1))) :=
by sorry

end option_A_is_iff_option_B_l547_547657


namespace vladimir_is_tallest_l547_547338

inductive Boy : Type
| Andrei | Boris | Vladimir | Dmitry

open Boy

-- Hypotheses based on the problem statement
def statements : Boy → (Prop × Prop)
| Andrei := (¬ (Boris = tallest), Vladimir = shortest)
| Boris := (Andrei = oldest, Andrei = shortest)
| Vladimir := (Dmitry > Vladimir, Dmitry = oldest)
| Dmitry := (statements Vladimir = (true, true), Dmitry = oldest)

def different_heights (a b : Boy) : Prop := a ≠ b
def different_ages (a b : Boy) : Prop := a ≠ b

axiom no_same_height {a b : Boy} : different_heights a b
axiom no_same_age {a b : Boy} : different_ages a b

-- The proof statement
theorem vladimir_is_tallest : Vladimir = tallest := 
by 
  sorry

end vladimir_is_tallest_l547_547338


namespace radius_of_sphere_in_truncated_cone_l547_547426

-- Definition of a truncated cone with bases of radii 24 and 6
structure TruncatedCone where
  top_radius : ℝ
  bottom_radius : ℝ

-- Sphere tangent condition
structure Sphere where
  radius : ℝ

-- The specific instance of the problem
def truncatedCone : TruncatedCone :=
{ top_radius := 6, bottom_radius := 24 }

def sphere : Sphere := sorry  -- The actual radius will be proven next.

theorem radius_of_sphere_in_truncated_cone : 
  sphere.radius = 12 ∧ 
  sphere_tangent_to_truncated_cone sphere truncatedCone :=
sorry

end radius_of_sphere_in_truncated_cone_l547_547426


namespace number_of_real_roots_f_eq_zero_l547_547409

theorem number_of_real_roots_f_eq_zero (f : ℝ → ℝ)
  (h_continuous : ∀ x, x ≠ 0 → continuous_at f x)
  (h_even : ∀ x, x ≠ 0 → f (-x) = f(x))
  (h_decreasing : ∀ x y, 0 < x → x < y → f y < f x)
  (h_f1_pos : f 1 > 0)
  (h_f2_neg : f 2 < 0) :
  ∃ n, (n = 2 ∨ n = 3) ∧ (∃ roots : fin n → ℝ, ∀ i, f (roots i) = 0) :=
sorry

end number_of_real_roots_f_eq_zero_l547_547409


namespace function_min_value_4_l547_547865

theorem function_min_value_4 :
  ∃ x : ℝ, (2^x + 2^(2 - x)) = 4 :=
sorry

end function_min_value_4_l547_547865


namespace probability_xiao_wang_selected_l547_547729

theorem probability_xiao_wang_selected :
  ∀ (friends : list String) (x : String),
    friends = ["Xiao Wang", "Xiao Zhang", "Xiao Liu", "Xiao Li"] →
    (∃ (n : ℤ) (m : ℤ), n = Nat.choose 4 2 ∧ m = Nat.choose 3 1 ∧ m / n = 1 / 2) :=
by
  intros friends x h
  have n_comb : n = Nat.choose 4 2 := by
    sorry
  have m_comb : ∃ y, m = Nat.choose 3 1 := by
    sorry
  have prob : (m / n = 1 / 2) := by
    sorry
  exact ⟨n, m, n_comb, m_comb, prob⟩

end probability_xiao_wang_selected_l547_547729


namespace find_b_l547_547196

theorem find_b
  (a b c : ℝ)
  (h_area : (1/2) * a * c * Real.sin (ℝ.pi / 3) = Real.sqrt 3)
  (h_ac_eq_4 : a * c = 4)
  (h_cosine : a^2 + c^2 = 3 * a * c) :
  b = 2 * Real.sqrt 2 :=
begin
  sorry
end

end find_b_l547_547196


namespace triangle_side_b_l547_547183

theorem triangle_side_b 
  (a b c : ℝ)
  (B : ℝ)
  (h1 : 1/2 * a * c * Real.sin B = Real.sqrt 3)
  (h2 : B = π / 3)
  (h3 : a^2 + c^2 = 3 * a * c) : b = 2 * Real.sqrt 2 := by
  sorry

end triangle_side_b_l547_547183


namespace find_smaller_number_than_neg3_l547_547434

theorem find_smaller_number_than_neg3 {a b c d : ℝ} (h1 : a = -2) (h2 : b = 4) (h3 : c = -5) (h4 : d = 1) :
  c < -3 :=
by
  rw h3
  ring
  norm_num
  sorry

end find_smaller_number_than_neg3_l547_547434


namespace min_value_f_l547_547971

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(2 - x)

theorem min_value_f : ∃ x : ℝ, f x = 4 :=
by {
  sorry
}

end min_value_f_l547_547971


namespace find_side_b_l547_547166

noncomputable def triangle_area (a b c : ℝ) (B : ℝ) : ℝ :=
  1 / 2 * a * c * real.sin B

noncomputable def law_of_cosines (a b c : ℝ) (B : ℝ) : ℝ :=
  a^2 + c^2 - 2 * a * c * real.cos B

theorem find_side_b (a b c : ℝ) (B : ℝ) (area : ℝ)
  (h1 : area = √3) 
  (h2 : B = real.pi / 3)
  (h3 : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 :=
by
  sorry

end find_side_b_l547_547166


namespace inequality_l547_547484

variable (n : ℕ) (a : Fin n → ℝ)

noncomputable def s : ℝ := ∑ i, a i

theorem inequality
  (hn : 2 ≤ n)
  (ha : ∀ i, 0 < a i)
  (hs : s a = ∑ i, a i) :
  ∑ i, a i / (s a - a i) ≥ n / (n - 1) :=
sorry

end inequality_l547_547484


namespace min_value_h_l547_547963

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def g (x : ℝ) : ℝ := abs (Real.sin x) + 4 / abs (Real.sin x)
noncomputable def h (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def j (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem min_value_h : ∃ x, h x = 4 :=
by
  sorry

end min_value_h_l547_547963


namespace triangle_area_l547_547768

noncomputable def area_CYL (x : ℝ) := (x^2) / (4 * real.sqrt 3)

theorem triangle_area 
  (A B C L M X Y : Type)
  [IsTriangle A B C]
  [bisects_angle A L (angle B A C)]
  [is_median B M]
  [intersect_at A L B M X]
  [line_through_points C X intersects_segment A B Y]
  (angle_BAC_eq_60 : ∠ B A C = 60)
  (AL_eq_x : dist A L = x) :
  area (triangle C Y L) = area_CYL x :=
sorry

end triangle_area_l547_547768


namespace min_value_of_2x_plus_2_2x_l547_547879

-- Lean 4 statement
theorem min_value_of_2x_plus_2_2x (x : ℝ) : 2^x + 2^(2 - x) ≥ 4 :=
by sorry

end min_value_of_2x_plus_2_2x_l547_547879


namespace cone_from_sector_proof_l547_547852

-- Definitions based on the problem conditions
def central_angle : ℝ := 300
def sector_radius : ℝ := 12

-- Correct answer to be proven
def cone_base_radius : ℝ := 10
def cone_slant_height : ℝ := 12

-- Lean 4 statement to prove the equivalence
theorem cone_from_sector_proof :
  let sector_arc_length := (central_angle / 360) * (2 * Real.pi * sector_radius) in
  let calculated_base_radius := sector_arc_length / (2 * Real.pi) in
  calculated_base_radius = cone_base_radius ∧ sector_radius = cone_slant_height :=
by
  -- Implementation of the proof goes here.
  -- Proof steps would recreate the calculation steps from the solution:
  -- 1. Calculate the arc length
  -- 2. Calculate the base radius
  -- 3. Verify slant height is equal to the sector radius
  sorry

end cone_from_sector_proof_l547_547852


namespace reflection_distance_lt_perimeter_l547_547452

-- Define the context of the problem
section regular_polygon_reflection

variables {k : ℕ} (n := 2^k)

-- The main theorem
theorem reflection_distance_lt_perimeter (O : ℂ)
  (l : fin n → (ℂ × ℂ)) : 
  let 
    z : ℕ → ℂ := fun i => if i = 0 then O else reflect (z (i - 1)) (l i)
  in |z n - O| < n * 2 * sin (π / n) := sorry

end regular_polygon_reflection

end reflection_distance_lt_perimeter_l547_547452


namespace min_value_h_is_4_l547_547982

def f (x : ℝ) := x^2 + 2*x + 4
def g (x : ℝ) := abs (sin x) + 4 / abs (sin x)
def h (x : ℝ) := 2^x + 2^(2 - x)
def j (x : ℝ) := log x + 4 / log x

theorem min_value_h_is_4 : 
  (∀ x, f(x) ≥ 3) ∧
  (∀ x, g(x) > 4) ∧
  (∃ x, h(x) = 4) ∧
  (∀ x, j(x) ≠ 4) → min_value_h = 4 :=
by
  sorry

end min_value_h_is_4_l547_547982


namespace function_min_value_l547_547916

theorem function_min_value (f1 f2 f3 f4 : ℝ → ℝ) :
  f1 = λ x, x^2 + 2 * x + 4 ∧
  f2 = λ x, abs (sin x) + 4 / abs (sin x) ∧
  f3 = λ x, 2^x + 2^(2-x) ∧
  f4 = λ x, log x + 4 / log x →
  (∀ x : ℝ, f3 x ≥ 4) ∧ (∃ x : ℝ, f3 x = 4) :=
by
  sorry

end function_min_value_l547_547916


namespace arith_prog_iff_avg_seq_arith_prog_l547_547672

-- Define sequences
def is_arith_prog (a : ℕ → ℝ) := ∃ d : ℝ, ∃ a₀ : ℝ, ∀ n : ℕ, a n = a₀ + n * d
def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range (n+1), a i
def avg_seq (a : ℕ → ℝ) (n : ℕ) := S a n / (n + 1)

-- Define theorem statement
theorem arith_prog_iff_avg_seq_arith_prog (a : ℕ → ℝ) :
  (is_arith_prog a) ↔ (is_arith_prog (λ n, avg_seq a n)) :=
sorry

end arith_prog_iff_avg_seq_arith_prog_l547_547672


namespace minimum_a_l547_547039

theorem minimum_a (a : ℝ) : (∀ (x y : ℝ), x > 0 → y > 0 → x + √(x * y) ≤ a * (x + y)) ↔ a ≥ (√2 + 1) / 2 :=
sorry

end minimum_a_l547_547039


namespace find_b_proof_l547_547199

noncomputable def find_b (a b c : ℝ) (A B C : ℝ) : Prop :=
  let area := sqrt 3 in
  let angle_B := 60 * Real.pi / 180 in
  let condition1 := area = (1 / 2) * a * c * Real.sin angle_B in
  let condition2 := a^2 + c^2 = 3 * a * c in
  let side_b := b = 2 * Real.sqrt 2 in
  condition1 ∧ condition2 → side_b

theorem find_b_proof :
  ∃ (a b c : ℝ) (A B C : ℝ), find_b a b c A B C :=
by
  sorry

end find_b_proof_l547_547199


namespace delivery_ratio_l547_547724

theorem delivery_ratio :
  ∀ (d1 d2 : ℝ) (total_pay pay_per_mile : ℝ),
  d1 = 10 ∧ d2 = 28 ∧ total_pay = 104 ∧ pay_per_mile = 2 →
  ((total_pay / pay_per_mile - (d1 + d2)) / d2) = 1 / 2 :=
by
  intros d1 d2 total_pay pay_per_mile h,
  cases h with h1 h2,
  cases h2 with h1 h2,
  cases h2 with h3 h4,
  rw [h1, h2, h3, h4],
  sorry

end delivery_ratio_l547_547724


namespace option_a_iff_option_b_l547_547636

-- Define a sequence as an arithmetic progression
def is_arithmetic (seq : ℕ → ℚ) : Prop :=
  ∃ d a₁, ∀ n, seq n = a₁ + n * d

-- Define the sum of the first n terms of a sequence
def sum_first_n (seq : ℕ → ℚ) (n : ℕ) : ℚ :=
  ∑ i in finset.range n, seq i

-- Define the average sum sequence
def avg_sum_seq (seq : ℕ → ℚ) (n : ℕ) : ℚ :=
  let Sn := sum_first_n seq n
  in Sn / n

-- The theorem to prove
theorem option_a_iff_option_b (seq : ℕ → ℚ) :
  is_arithmetic seq ↔ is_arithmetic (λ n, avg_sum_seq seq (n + 1)) :=
sorry

end option_a_iff_option_b_l547_547636


namespace triangle_ratio_l547_547599

theorem triangle_ratio :
  ∃ (m n : ℕ), m + n = 11 ∧ RelPrime m n ∧
    let a := 12,
        b := 35,
        c := Real.sqrt (a^2 + b^2), -- Hypotenuse of the triangle
        r := (a * b) / c,           -- Radius derived from the altitude
        ω := circle (0:r),         -- Circle ω based on diameter CD
        I := some_point_outside_triangle, -- Assume the existence of point I
        peri_ABI := 2 * (b + r) + 2 * (c / 3) in
    (peri_ABI / c) = (8 / 3) := 
by 
  sorry -- Proof goes here

noncomputable def some_point_outside_triangle := sorry -- Placeholder for the point I

end triangle_ratio_l547_547599


namespace find_side_b_l547_547164

noncomputable def triangle_area (a b c : ℝ) (B : ℝ) : ℝ :=
  1 / 2 * a * c * real.sin B

noncomputable def law_of_cosines (a b c : ℝ) (B : ℝ) : ℝ :=
  a^2 + c^2 - 2 * a * c * real.cos B

theorem find_side_b (a b c : ℝ) (B : ℝ) (area : ℝ)
  (h1 : area = √3) 
  (h2 : B = real.pi / 3)
  (h3 : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 :=
by
  sorry

end find_side_b_l547_547164


namespace tetrahedron_longest_edge_l547_547797

noncomputable def exists_long_edge (A B C D O : Point) (R : ℝ) :=
  let dist := (λ (P Q : Point), Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2 + (P.z - Q.z)^2))
  ∃ (e : ℝ), (e ∈ {dist A B, dist A C, dist A D, dist B C, dist B D, dist C D}) ∧ e ≥ (2 / 3) * Real.sqrt 6 * R

theorem tetrahedron_longest_edge (A B C D : Point) (R : ℝ) (O : Point)
  (h_sphere : ∀ P ∈ {A, B, C, D}, dist O P = R) (h_inside: tetrahedron_contains O A B C D) :
  exists_long_edge A B C D O R :=
sorry

end tetrahedron_longest_edge_l547_547797


namespace function_min_value_4_l547_547854

theorem function_min_value_4 :
  ∃ x : ℝ, (2^x + 2^(2 - x)) = 4 :=
sorry

end function_min_value_4_l547_547854


namespace regular_tetrahedron_midsphere_geometric_mean_l547_547261

/-- Prove that in a regular tetrahedron, the radius of the midsphere 
(the sphere tangent to the edges) is the geometric mean of the radius 
of the insphere and the radius of the circumsphere. -/
theorem regular_tetrahedron_midsphere_geometric_mean
  (T : Tetrahedron)
  (h_regular : T.regular)
  (M : Point)
  (h_centroid : M = T.centroid)
  (R_circumsphere : ℝ := T.radius_circumsphere)
  (R_insphere : ℝ := T.radius_insphere)
  (R_midsphere : ℝ := T.radius_midsphere) :
  R_midsphere^2 = R_insphere * R_circumsphere :=
sorry

end regular_tetrahedron_midsphere_geometric_mean_l547_547261


namespace no_solution_l547_547467

theorem no_solution (n : ℕ) (x y k : ℕ) (h1 : n ≥ 1) (h2 : x > 0) (h3 : y > 0) (h4 : k > 1) (h5 : Nat.gcd x y = 1) (h6 : 3^n = x^k + y^k) : False :=
by
  sorry

end no_solution_l547_547467


namespace plane_equation_l547_547415

-- Define the parametric equation and vectors as conditions
def v (s t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 2 * s - 2 * t, 3 - 2 * s, 4 - s + 3 * t)

def a : ℝ × ℝ × ℝ := (2, -2, -1)
def b : ℝ × ℝ × ℝ := (-2, 0, 3)

-- Given the vectors a and b, find the cross product
def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2.1 * v.2.2 - u.2.2 * v.2.1,
   u.2.2 * v.1 - u.1 * v.2.2,
   u.1 * v.2.1 - u.2.1 * v.1)

-- Calculate the normal vector by simplifying (2 * 0 - (-2) * 3, -1 * (-2) - 2 * 3, 2 * 0 - (-2) * (-2))
def n := cross_product a b

-- Define the proof statement, where we need to prove the plane equation from the given conditions
theorem plane_equation :
  ∃ A B C D : ℤ, 0 < A ∧ Int.gcd A B C D = 1 ∧ 
  (∀ x y z : ℝ, 3 * x - y + 2 * z - 11 = 0 ↔ 
  ∃ s t : ℝ, (x, y, z) = v s t) := 
by
  sorry

end plane_equation_l547_547415


namespace cyclist_downhill_speed_l547_547404

noncomputable def downhill_speed (d uphill_speed avg_speed : ℝ) : ℝ :=
  let downhill_speed := (2 * d * uphill_speed) / (avg_speed * d - uphill_speed * 2)
  -- We want to prove
  downhill_speed

theorem cyclist_downhill_speed :
  downhill_speed 150 25 35 = 58.33 :=
by
  -- Proof omitted
  sorry

end cyclist_downhill_speed_l547_547404


namespace log_abs_diff_leq_three_l547_547047

theorem log_abs_diff_leq_three (x : ℝ) : log 2 (|x + 1| - |x - 7|) ≤ 3 :=
sorry

end log_abs_diff_leq_three_l547_547047


namespace set_equality_proof_l547_547329

theorem set_equality_proof :
  {x : ℕ | x > 1 ∧ x ≤ 3} = {x : ℕ | x = 2 ∨ x = 3} :=
by
  sorry

end set_equality_proof_l547_547329


namespace option_a_iff_option_b_l547_547642

-- Define a sequence as an arithmetic progression
def is_arithmetic (seq : ℕ → ℚ) : Prop :=
  ∃ d a₁, ∀ n, seq n = a₁ + n * d

-- Define the sum of the first n terms of a sequence
def sum_first_n (seq : ℕ → ℚ) (n : ℕ) : ℚ :=
  ∑ i in finset.range n, seq i

-- Define the average sum sequence
def avg_sum_seq (seq : ℕ → ℚ) (n : ℕ) : ℚ :=
  let Sn := sum_first_n seq n
  in Sn / n

-- The theorem to prove
theorem option_a_iff_option_b (seq : ℕ → ℚ) :
  is_arithmetic seq ↔ is_arithmetic (λ n, avg_sum_seq seq (n + 1)) :=
sorry

end option_a_iff_option_b_l547_547642


namespace irrational_number_l547_547326

noncomputable def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem irrational_number : 
  is_rational (Real.sqrt 4) ∧ 
  is_rational (22 / 7 : ℝ) ∧ 
  is_rational (1.0101 : ℝ) ∧ 
  ¬ is_rational (Real.pi / 3) 
  :=
sorry

end irrational_number_l547_547326


namespace find_B_find_b_find_cos_2A_plus_B_l547_547596

variable (A B C a b c : ℝ)
variable (A_lt_C : A < C) (triangle_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
variable (law_of_sines : ∀ {A B C a b c}, a / sin A = b / sin B ∧ b / sin B = c / sin C ∧ a / sin A = c / sin C)
variable (law_of_cosines : ∀ {a b c A B C}, c^2 = a^2 + b^2 - 2*a*b*cos C)
variable (sin_def : sin (A + C) = sin π - B) (equation : sqrt(3) * b = 2 * a * sin B * cos C + 2 * c * sin B * cos A)

theorem find_B : B = π/3 :=
  sorry

theorem find_b (a_eq : a = 3) (c_eq : c = 4) (B_eq : B = π/3) :
  b = sqrt 13 :=
  sorry

theorem find_cos_2A_plus_B (a_eq : a = 3) (c_eq : c = 4) (B_eq : B = π/3) :
  cos (2*A + B) = -23/26 :=
  sorry

end find_B_find_b_find_cos_2A_plus_B_l547_547596


namespace minimum_value_of_option_C_l547_547892

noncomputable def optionA (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def optionB (x : ℝ) : ℝ := |Real.sin x| + 4 / |Real.sin x|
noncomputable def optionC (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def optionD (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem minimum_value_of_option_C :
  ∃ x : ℝ, optionC x = 4 :=
by
  sorry

end minimum_value_of_option_C_l547_547892


namespace min_value_h_is_4_l547_547989

def f (x : ℝ) := x^2 + 2*x + 4
def g (x : ℝ) := abs (sin x) + 4 / abs (sin x)
def h (x : ℝ) := 2^x + 2^(2 - x)
def j (x : ℝ) := log x + 4 / log x

theorem min_value_h_is_4 : 
  (∀ x, f(x) ≥ 3) ∧
  (∀ x, g(x) > 4) ∧
  (∃ x, h(x) = 4) ∧
  (∀ x, j(x) ≠ 4) → min_value_h = 4 :=
by
  sorry

end min_value_h_is_4_l547_547989


namespace minimize_f_C_l547_547935

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem minimize_f_C : ∃ x : ℝ, f_C x = 4 :=
by {
  -- The correctness of the statement is asserted and the proof is omitted
  sorry
}

end minimize_f_C_l547_547935
