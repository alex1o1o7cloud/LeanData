import Mathlib

namespace count_ordered_triples_l369_369091

theorem count_ordered_triples :
  ∃ (S : Finset (ℝ × ℝ × ℝ)), 4 = S.card ∧
    ∀ x ∈ S, prod.fst (prod.fst x) ≠ 0 ∧ prod.snd (prod.fst x) ≠ 0 ∧ prod.snd x ≠ 0 ∧
              (prod.fst (prod.fst x) * prod.snd (prod.fst x) = 2 * prod.snd x) ∧
              (prod.snd (prod.fst x) * prod.snd x = 2 * prod.fst (prod.fst x)) ∧
              (prod.snd x * prod.fst (prod.fst x) = 2 * prod.snd (prod.fst x)) :=
sorry

end count_ordered_triples_l369_369091


namespace remainder_when_M_divided_by_32_l369_369144

-- Define M as the product of all odd primes less than 32.
def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_M_divided_by_32 :
    M % 32 = 1 := by
  -- We must prove that M % 32 = 1.
  sorry

end remainder_when_M_divided_by_32_l369_369144


namespace regular_polygon_sides_l369_369483

theorem regular_polygon_sides (n : ℕ) (h₁ : ∀ k : ℕ, k = n → 180 * (k - 2) = 140 * k) : n = 9 := by
  have h₂ := h₁ n rfl
  linarith

end regular_polygon_sides_l369_369483


namespace sides_of_regular_polygon_l369_369480

theorem sides_of_regular_polygon (n : ℕ) (h : ∀ (i : ℕ), i < n → (interior_angle i = 140)) :
  n = 9 := 
  sorry

end sides_of_regular_polygon_l369_369480


namespace product_mod_32_l369_369216

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  M % 32 = 17 := by
  sorry

end product_mod_32_l369_369216


namespace sin_function_value_l369_369807

noncomputable def f (x : ℝ) : ℝ := sin(2 * x - 5 * π / 6)

theorem sin_function_value :
  f (-5 * π / 12) = sqrt 3 / 2 := by
  sorry

end sin_function_value_l369_369807


namespace arithmetic_geometric_sequence_l369_369062

theorem arithmetic_geometric_sequence (a b c : ℝ) 
  (a_ne_b : a ≠ b) (b_ne_c : b ≠ c) (a_ne_c : a ≠ c)
  (h1 : 2 * b = a + c)
  (h2 : (a * b)^2 = a * b * c^2)
  (h3 : a + b + c = 15) : a = 20 := 
by 
  sorry

end arithmetic_geometric_sequence_l369_369062


namespace unhappy_snakes_no_skills_no_skills_implies_purple_l369_369402

-- Defining types and predicates
variable {Snake : Type}

-- Predicate definitions
def isPurple (s : Snake) : Prop := sorry
def isHappy (s : Snake) : Prop := sorry
def canAdd (s : Snake) : Prop := sorry
def canSubtract (s : Snake) : Prop := sorry
def isUnhappy (s : Snake) : Prop := sorry

-- Given conditions
axiom totalSnakes : ∀ s : Snake, s ∈ Finset.univ (Finset.range 17)
axiom purpleSnakes : ∀ s, isPurple s → s ∈ Finset.range 5
axiom purple_snakes_unhappy : ∀ s, isPurple s → isUnhappy s
axiom happy_snakes_count : ∀ s, isHappy s → s ∈ Finset.range 7
axiom happy_snakes_skills : ∀ s, isHappy s → canAdd s ∧ canSubtract s
axiom purple_snakes_no_skills : ∀ s, isPurple s → ¬(canAdd s ∨ canSubtract s)

-- Required Proofs
theorem unhappy_snakes_no_skills : ∀ s, isUnhappy s → ¬(canAdd s ∨ canSubtract s) := sorry
theorem no_skills_implies_purple : ∀ s, ¬(canAdd s ∨ canSubtract s) → isPurple s := sorry

end unhappy_snakes_no_skills_no_skills_implies_purple_l369_369402


namespace stratified_sampling_height_group_selection_l369_369469

theorem stratified_sampling_height_group_selection :
  let total_students := 100
  let group1 := 20
  let group2 := 50
  let group3 := 30
  let total_selected := 18
  group1 + group2 + group3 = total_students →
  (group3 : ℝ) / total_students * total_selected = 5.4 →
  round ((group3 : ℝ) / total_students * total_selected) = 3 :=
by
  intros total_students group1 group2 group3 total_selected h1 h2
  sorry

end stratified_sampling_height_group_selection_l369_369469


namespace sin_monotonic_increasing_f_symmetric_axes_find_value_l369_369695

def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi * 5 / 6)

theorem sin_monotonic_increasing (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) :
  Monotone (λ x, f x) (Ioo a b) := 
sorry

theorem f_symmetric_axes (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) 
  (x y : ℝ) (hx : x = a + (b - a) * k) (hy : y = a + (b - a) * (-k)) : 
  f x = f y :=
sorry

theorem find_value :
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
sorry

end sin_monotonic_increasing_f_symmetric_axes_find_value_l369_369695


namespace sin_function_value_l369_369808

noncomputable def f (x : ℝ) : ℝ := sin(2 * x - 5 * π / 6)

theorem sin_function_value :
  f (-5 * π / 12) = sqrt 3 / 2 := by
  sorry

end sin_function_value_l369_369808


namespace max_product_l369_369880

theorem max_product (x y : ℕ) (h1 : 7 * x + 4 * y = 140) : x * y ≤ 168 :=
sorry

end max_product_l369_369880


namespace prime_product_mod_32_l369_369303

open Nat

theorem prime_product_mod_32 : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := List.foldr (· * ·) 1 primes
  M % 32 = 25 :=
by
  sorry

end prime_product_mod_32_l369_369303


namespace num_solutions_eq_3_l369_369320

theorem num_solutions_eq_3 : {p : ℕ × ℕ // 0 < p.1 ∧ 0 < p.2 ∧ (6 / p.1 + 3 / p.2 : ℚ) = 1}.card = 3 := 
sorry

end num_solutions_eq_3_l369_369320


namespace gcd_18_30_l369_369604

-- Define the two numbers
def num1 : ℕ := 18
def num2 : ℕ := 30

-- State the theorem to find the gcd
theorem gcd_18_30 : Nat.gcd num1 num2 = 6 := by
  sorry

end gcd_18_30_l369_369604


namespace gcd_of_18_and_30_l369_369559

-- Define the numbers
def a := 18
def b := 30

-- The main theorem statement
theorem gcd_of_18_and_30 : Nat.gcd a b = 6 := by
  sorry

end gcd_of_18_and_30_l369_369559


namespace remainder_M_mod_32_l369_369262

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_M_mod_32 : M % 32 = 17 :=
by {
  sorry
}

end remainder_M_mod_32_l369_369262


namespace product_of_odd_primes_mod_32_l369_369172

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def M : ℕ := odd_primes_less_than_32.foldr (· * ·) 1

theorem product_of_odd_primes_mod_32 : M % 32 = 17 :=
by
  -- the proof goes here
  sorry

end product_of_odd_primes_mod_32_l369_369172


namespace result_after_2016_operations_l369_369041

-- Define the operations according to conditions
def operation1 := 2^3 + 5^3  -- 133
def operation2 := 1^3 + 3^3 + 3^3  -- 55
def operation3 := 5^3 + 5^3  -- 250
def operation4 := 2^3 + 5^3 + 0^3  -- 133

-- Define the cycle length
def cycle := [operation1, operation2, operation3, operation4]

-- Define a function to get the result of the nth operation
def nth_operation (n : ℕ) : ℕ := cycle[(n % 3) + 1]

theorem result_after_2016_operations : nth_operation 2016 = 250 := 
by
  -- skipping the actual proof
  sorry

end result_after_2016_operations_l369_369041


namespace find_f_of_neg_5_pi_over_12_l369_369732

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - (5 * π / 6))

theorem find_f_of_neg_5_pi_over_12 :
  (∀ x ∈ (set.Ioo (π / 6) (2 * π / 3)), (0 : ℝ) < (f x - f (x - 1))) ∧
  (∀ x, f (π / 6 - x) = f (π / 6 + x)) ∧
  (∀ x, f (2 * π / 3 - x) = f (2 * π / 3 + x)) →
  f (-5 * π / 12) = √3 / 2 :=
by
  sorry

end find_f_of_neg_5_pi_over_12_l369_369732


namespace evaluate_expression_l369_369850

def my_star (A B : ℕ) : ℕ := (A + B) / 2
def my_hash (A B : ℕ) : ℕ := A * B + 1

theorem evaluate_expression : my_hash (my_star 4 6) 5 = 26 := 
by
  sorry

end evaluate_expression_l369_369850


namespace vectors_parallel_l369_369082

noncomputable def log2 := Real.log 2
noncomputable def log23 := Real.log 3 / log2

theorem vectors_parallel (x : ℝ) (h_parallel : (x, -1) ∥ (log23, 1)) : 4^x + 4^(-x) = 82 / 9 := by
  -- conditions: 
  -- Vectors (x, -1) and (log23, 1)
  -- Vectors are parallel
  sorry

end vectors_parallel_l369_369082


namespace gcd_18_30_l369_369577

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := 
by
  sorry

end gcd_18_30_l369_369577


namespace log_3_expression_result_l369_369011

theorem log_3_expression_result :
  ∃ (x : ℝ), 0 < x ∧ x = Real.log 3 (64 + Real.log 3 (64 + Real.log 3 (64 + ...))) ∧ x ≈ 4 :=
by
  sorry

end log_3_expression_result_l369_369011


namespace general_term_formula_l369_369375

section

variable {a : ℕ → ℤ} {S : ℕ → ℤ}

-- Condition: S_n is the sum of the first n terms of the sequence {a_n}
def Sn : ℕ → ℤ
| 0       := 0
| (n + 1) := Sn n + a (n + 1)

-- Condition: (n, S_n) lies on the curve defined by f(x) = x^2 - 4x
axiom Sn_on_curve : ∀ n : ℕ, n > 0 → Sn n = n^2 - 4 * n

-- Theorem: a_n = 2n - 5
theorem general_term_formula (n : ℕ) (hn : n > 0) : a n = 2 * ↑n - 5 :=
sorry

end

end general_term_formula_l369_369375


namespace remainder_of_M_when_divided_by_32_l369_369200

open Nat

def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
def M : ℕ := List.foldl (*) 1 oddPrimes

theorem remainder_of_M_when_divided_by_32 :
  M % 32 = 1 :=
sorry

end remainder_of_M_when_divided_by_32_l369_369200


namespace common_elements_in_S_and_T_l369_369965

-- Definitions based on the problem conditions
def S : Set ℕ := { n | ∃ k, 1 ≤ k ∧ k ≤ 1500 ∧ n = 5 * k }
def T : Set ℕ := { n | ∃ k, 1 ≤ k ∧ k ≤ 1500 ∧ n = 7 * k }

-- The statement of the theorem to be proved:
theorem common_elements_in_S_and_T : (S ∩ T).card = 214 := 
by 
  sorry

end common_elements_in_S_and_T_l369_369965


namespace product_mod_32_is_15_l369_369236

-- Define the odd primes less than 32
def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the product of odd primes
def M : ℕ := List.foldl (· * ·) 1 oddPrimes

-- The theorem to prove the remainder when M is divided by 32 is 15
theorem product_mod_32_is_15 : M % 32 = 15 := by
  sorry

end product_mod_32_is_15_l369_369236


namespace a_plus_b_equals_two_l369_369070

-- The function definition
def f (a b x : ℝ) : ℝ := a * x^2 + (b - 3) * x + 3

-- The property that the function is even: f(-x) = f(x)
def even_function (a b : ℝ) : Prop :=
  ∀ x : ℝ, f(a, b, x) = f(a, b, -x)

-- The domain condition
def domain_condition (a : ℝ) : Prop :=
  (2 * a - 3) + (4 - a) = 0

-- Lean statement to prove
theorem a_plus_b_equals_two (a b : ℝ) (h1 : domain_condition(a)) (h2 : even_function(a, b)) : a + b = 2 :=
sorry

end a_plus_b_equals_two_l369_369070


namespace sin_function_value_l369_369801

noncomputable def f (x : ℝ) : ℝ := sin(2 * x - 5 * π / 6)

theorem sin_function_value :
  f (-5 * π / 12) = sqrt 3 / 2 := by
  sorry

end sin_function_value_l369_369801


namespace gcd_of_18_and_30_l369_369591

-- Define the numbers
def num1 := 18
def num2 := 30

-- State the GCD property
theorem gcd_of_18_and_30 : Nat.gcd num1 num2 = 6 :=
by
  sorry

end gcd_of_18_and_30_l369_369591


namespace gcd_18_30_l369_369586

theorem gcd_18_30: Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l369_369586


namespace product_mod_32_l369_369212

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  M % 32 = 17 := by
  sorry

end product_mod_32_l369_369212


namespace find_value_range_l369_369077

noncomputable def C1_eq (x y : ℝ) : Prop :=
  (x^2 / 2) + y^2 = 1

noncomputable def C2_eq (x y : ℝ) : Prop :=
  y^2 = 4 * x

theorem find_value_range (x y : ℝ) (α : ℝ) (t : ℝ) :
  (C1_eq (sqrt 2 * cos θ) (sin θ)) → 
  (C2_eq (sqrt ((4 * cos θ) / (sin θ)^2)) (4 * cos θ)) →
  (|FA| * |FB| ≠ 0 ∧ |FM| * |FN| ≠ 0) →
  ∀ F_A F_B F_M F_N,
  ∃ r ∈ (0, 1 / 8], 
  r = (|FA| * |FB| / (|FM| * |FN|)) :=
begin
  sorry
end

end find_value_range_l369_369077


namespace find_value_of_f_eq_l369_369766

noncomputable def f (ω ω φ x : ℝ) : ℝ := sin (ω * x + φ)

theorem find_value_of_f_eq :
  ∀ (ω φ : ℝ), 
    (∀ x, x ∈ set.Ioo (π / 6) (2 * π / 3) → (f ω φ x) ≤ (f ω φ (x + 1e-10))) → -- monotonically increasing
    (ω * ∓ (π / 6) + φ) = (φ + ω * (2 * π / 3)) → -- symmetric axes
    f ω φ (-(5 * π / 12)) = sqrt 3 / 2 :=
by
  sorry

end find_value_of_f_eq_l369_369766


namespace remainder_of_product_of_odd_primes_mod_32_l369_369290

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem remainder_of_product_of_odd_primes_mod_32 :
  (product odd_primes_less_than_32) % 32 = 23 :=
by sorry

end remainder_of_product_of_odd_primes_mod_32_l369_369290


namespace alpha_quadratic_eqn_beta_quartic_eqn_l369_369319

noncomputable def alpha : ℝ := Real.cot (Real.pi / 8)
noncomputable def beta : ℝ := Real.csc (Real.pi / 8)

theorem alpha_quadratic_eqn :
  alpha^2 - 2 * alpha - 1 = 0 := 
sorry

theorem beta_quartic_eqn :
  beta^4 - 8 * beta^2 + 8 = 0 :=
sorry

end alpha_quadratic_eqn_beta_quartic_eqn_l369_369319


namespace world_expo_art_arrangement_l369_369525

/-- During the 2010 Shanghai World Expo, a country exhibited 5 pieces of art: 
2 different calligraphy works, 2 different painting works, and 1 iconic architectural design.
These 5 pieces were to be arranged in a row at the exhibition booth, with the requirement that 
the 2 calligraphy works must be adjacent, and the 2 painting works cannot be adjacent.
Prove that the number of different arrangements for exhibiting these 5 pieces of art is 24. -/
theorem world_expo_art_arrangement :
  let pieces := ['C1', 'C2', 'P1', 'P2', 'A'] in
  let units := [['C1', 'C2'], 'P1', 'P2', 'A'] in
  ∃ arrangements, (∀ unit ∈ units, is_adjacent_unit unit arrangements) ∧
  (¬ ∃ pair, pair ∈ [['P1', 'P2']] ∧ is_adjacent_pair pair arrangements) →
  num_arrangements units = 24 :=
sorry

end world_expo_art_arrangement_l369_369525


namespace product_of_odd_primes_mod_32_l369_369173

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def M : ℕ := odd_primes_less_than_32.foldr (· * ·) 1

theorem product_of_odd_primes_mod_32 : M % 32 = 17 :=
by
  -- the proof goes here
  sorry

end product_of_odd_primes_mod_32_l369_369173


namespace product_mod_32_l369_369213

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  M % 32 = 17 := by
  sorry

end product_mod_32_l369_369213


namespace proof_PN_eq_OM_l369_369105

/-- Circle -/
structure Circle (P : Type) :=
  (center : P)
  (radius : ℝ)

-- Midpoint definition in the context of real numbers
def midpoint (x y : ℝ) : ℝ := (x + y) / 2

-- Given definitions
variables {P : Type} [metric_space P] [normed_group P] [normed_space ℝ P]
variables (O : P) -- The center of circle
variables (A B C D M N P : P) -- Points on circle
variables (c : Circle P)

-- Assume chords AC and BD intersect perpendicularly at P
variable intersect_perpendicularly : ∃ P : P, ∃ A B C D : P, 
  dist O A = c.radius ∧ dist O B = c.radius ∧ dist O C = c.radius ∧ dist O D = c.radius ∧
  (∠ A P B = π / 2)

-- Midpoints M and N of AD and BC, respectively
variable midpoint_M_AD : midpoint (dist O A) (dist O D) = M
variable midpoint_N_BC : midpoint (dist O B) (dist O C) = N

theorem proof_PN_eq_OM : dist P N = dist O M :=
sorry

end proof_PN_eq_OM_l369_369105


namespace sum_reciprocal_seq_l369_369437

noncomputable def seq (n : ℕ) : ℝ :=
  if n = 0 then 1
  else if n = 1 then 2.1
  else seq(n+1) * seq(n-1) ^ 2 + 2 * seq(n-1) ^ 2 * seq(n) = seq(n) ^ 3

theorem sum_reciprocal_seq (k : ℕ) : (∑ i in Finset.range (k + 1), 1 / seq i) < 2.016 := by
  sorry

end sum_reciprocal_seq_l369_369437


namespace product_of_odd_primes_mod_32_l369_369225

theorem product_of_odd_primes_mod_32 :
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31) in
  M % 32 = 25 :=
by
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31)
  sorry

end product_of_odd_primes_mod_32_l369_369225


namespace jaiAlai_winner_is_P4_l369_369462

def player := ℕ -- Representing players as natural numbers for simplicity, where 1 represents P1, 2 represents P2, etc.

-- Define the initial conditions and game rules
structure jaiAlaiGame :=
  (players : fin 8)             -- 8 players
  (initial_court : set (fin 2)) -- Starting on the court: P1 (1) and P2 (2)
  (queue : list (fin 6))        -- Initial queue: P3, P4, P5, P6, P7, P8
  (points : fin 37)             -- Total points across all players at the end of the game

-- Define the winning condition
def winning_condition (game : jaiAlaiGame) (winner : player) : Prop :=
  (game.points.val = 36 ∨ game.points.val =37) ∧ winner = 4

-- Define the theorem to be proved
theorem jaiAlai_winner_is_P4 (game : jaiAlaiGame) (winner : player) :
  game.points.val = 37 → winner = 4 :=
by {
  sorry
}

end jaiAlai_winner_is_P4_l369_369462


namespace volume_of_rhombohedron_l369_369484

-- Define the conditions as Lean definitions
def d1 : ℝ := 25 -- First diagonal of the rhombus
def d2 : ℝ := 50 -- Second diagonal of the rhombus
def extrusion_height : ℝ := 20 -- Height of the extrusion

-- Define the proof statement
theorem volume_of_rhombohedron (d1 d2 extrusion_height: ℝ) (h_d1 : d1 = 25) (h_d2 : d2 = 50) (h_height : extrusion_height = 20) :
  let area := (d1 * d2) / 2 in
  let volume := area * extrusion_height in
  volume = 12500 := by
  sorry

end volume_of_rhombohedron_l369_369484


namespace product_of_odd_primes_mod_32_l369_369187

theorem product_of_odd_primes_mod_32 :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  M % 32 = 17 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  exact sorry

end product_of_odd_primes_mod_32_l369_369187


namespace gcd_of_18_and_30_l369_369549

theorem gcd_of_18_and_30 : Nat.gcd 18 30 = 6 :=
by
  sorry

end gcd_of_18_and_30_l369_369549


namespace prime_product_mod_32_l369_369299

open Nat

theorem prime_product_mod_32 : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := List.foldr (· * ·) 1 primes
  M % 32 = 25 :=
by
  sorry

end prime_product_mod_32_l369_369299


namespace find_value_of_f_eq_l369_369758

noncomputable def f (ω ω φ x : ℝ) : ℝ := sin (ω * x + φ)

theorem find_value_of_f_eq :
  ∀ (ω φ : ℝ), 
    (∀ x, x ∈ set.Ioo (π / 6) (2 * π / 3) → (f ω φ x) ≤ (f ω φ (x + 1e-10))) → -- monotonically increasing
    (ω * ∓ (π / 6) + φ) = (φ + ω * (2 * π / 3)) → -- symmetric axes
    f ω φ (-(5 * π / 12)) = sqrt 3 / 2 :=
by
  sorry

end find_value_of_f_eq_l369_369758


namespace number_of_sides_of_polygon_l369_369386

theorem number_of_sides_of_polygon (n : ℕ) : (n - 2) * 180 = 720 → n = 6 :=
by
  sorry

end number_of_sides_of_polygon_l369_369386


namespace tan_C_in_triangle_l369_369941

theorem tan_C_in_triangle (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : Real.tan A = 1) (h₃ : Real.tan B = 2) :
  Real.tan C = 3 :=
sorry

end tan_C_in_triangle_l369_369941


namespace product_of_odd_primes_mod_32_l369_369165

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def M : ℕ := odd_primes_less_than_32.foldr (· * ·) 1

theorem product_of_odd_primes_mod_32 : M % 32 = 17 :=
by
  -- the proof goes here
  sorry

end product_of_odd_primes_mod_32_l369_369165


namespace find_x2_y2_l369_369094

theorem find_x2_y2 (x y : ℝ) (h₁ : (x + y)^2 = 9) (h₂ : x * y = -6) : x^2 + y^2 = 21 := 
by
  sorry

end find_x2_y2_l369_369094


namespace sin_function_value_l369_369798

noncomputable def f (x : ℝ) : ℝ := sin(2 * x - 5 * π / 6)

theorem sin_function_value :
  f (-5 * π / 12) = sqrt 3 / 2 := by
  sorry

end sin_function_value_l369_369798


namespace triangle_side_ratios_l369_369122

theorem triangle_side_ratios (A B C D : Point) (hD : D ∈ line_segment A B) (h1 : dist A B = 2 * dist A C) (h2 : dist A B = 4 * dist A D) :
  dist B C = 2 * dist C D :=
sorry

end triangle_side_ratios_l369_369122


namespace product_of_odd_primes_mod_32_l369_369176

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def M : ℕ := odd_primes_less_than_32.foldr (· * ·) 1

theorem product_of_odd_primes_mod_32 : M % 32 = 17 :=
by
  -- the proof goes here
  sorry

end product_of_odd_primes_mod_32_l369_369176


namespace distance_AB_eq_5_div_2_l369_369447

-- Definitions for lines and points
def l1 (t : ℝ) : ℝ × ℝ := (1 + 3 * t, 2 - 4 * t)

def l2 (x y : ℝ) : Prop := 2 * x - 4 * y = 5

-- Definition of intersection point B
def B : ℝ × ℝ := (5 / 2, 0)

-- Definition of point A
def A : ℝ × ℝ := (1, 2)

-- Calculate the distance |AB|
def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- The statement to prove
theorem distance_AB_eq_5_div_2 :
  distance A B = 5 / 2 := by
  sorry

end distance_AB_eq_5_div_2_l369_369447


namespace theta_calculation_l369_369509

noncomputable def calculate_theta: ℝ := 
let z1 := Complex.exp (Complex.I * (11 * Real.pi / 120));
let z2 := Complex.exp (Complex.I * (31 * Real.pi / 120));
let z3 := Complex.exp (Complex.I * (107 * Real.pi / 120));
let z4 := Complex.exp (Complex.I * (67 * Real.pi / 120));
let z5 := Complex.exp (Complex.I * (47 * Real.pi / 120));
in Complex.arg (z1 + z2 + z3 + z4 + z5)

theorem theta_calculation : calculate_theta = 59 * Real.pi / 120 := by
  sorry

end theta_calculation_l369_369509


namespace gcd_18_30_l369_369601

-- Define the two numbers
def num1 : ℕ := 18
def num2 : ℕ := 30

-- State the theorem to find the gcd
theorem gcd_18_30 : Nat.gcd num1 num2 = 6 := by
  sorry

end gcd_18_30_l369_369601


namespace find_f_neg_5pi_12_l369_369776

-- Conditions
def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

variables {ω φ : ℝ}
def f (x : ℝ) : ℝ := Real.sin (ω * x + φ)

axiom monotonicity_condition : is_monotonically_increasing f (π / 6) (2 * π / 3)
axiom symmetry_condition_1 : f (π / 6) = f (2 * π / 3)
axiom symmetry_condition_2 : f (-π / 6) = f (π / 2)

-- Goal
theorem find_f_neg_5pi_12 : f (- 5 * π / 12) = sqrt 3 / 2 := 
sorry

end find_f_neg_5pi_12_l369_369776


namespace regular_polygon_sides_l369_369477

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 1 ≤ i → i ≤ n → interior_angle i = 140) (h2 : sum_of_interior_angles n = 180 * (n - 2)) :
  n = 9 :=
by
  sorry

end regular_polygon_sides_l369_369477


namespace function_with_prop_M_l369_369100

noncomputable def prop_M (f : ℝ → ℝ) := ∀ x, (deriv (fun x => exp x * f x) x) ≥ 0

theorem function_with_prop_M :
  ∀ f ∈ ({(fun x : ℝ => 2^x): ℝ → ℝ, (fun x : ℝ => x^2): ℝ → ℝ, (fun x: ℝ => 3^(-x)): ℝ → ℝ, (fun x: ℝ => cos x): ℝ → ℝ}),
  f = (fun x => 2^x) ↔ prop_M f :=
by
  intros
  sorry

end function_with_prop_M_l369_369100


namespace xyz_sum_l369_369096

theorem xyz_sum (x y z : ℝ) 
  (h1 : y + z = 17 - 2 * x) 
  (h2 : x + z = 1 - 2 * y) 
  (h3 : x + y = 8 - 2 * z) : 
  x + y + z = 6.5 :=
sorry

end xyz_sum_l369_369096


namespace problem_statement_l369_369756

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - 5 * Real.pi / 6)

theorem problem_statement :
  (∀ (x : ℝ), (∀ a b : ℝ, a < b → x ∈ (Set.Ioc (Real.pi / 6) (2 * Real.pi / 3)) → f(x) < f(b)) → x = (Real.pi / 6) ∨ x = (2 * Real.pi / 3)) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by sorry

end problem_statement_l369_369756


namespace find_value_l369_369706

-- Define the function $f(x)$
def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

-- Define the problem statement in Lean 4
theorem find_value :
  (∀ x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), 0 ≤ 2 * x - 5 * Real.pi / 6 ∧ 2 * x - 5 * Real.pi / 6 ≤ Real.pi) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by
  -- A proper rigorous Lean proof should go here
  sorry

end find_value_l369_369706


namespace gcd_problem_l369_369030

noncomputable def least_reducible_n: ℕ :=
  if h : ∃ n : ℕ, n > 0 ∧ (Nat.gcd (n - 31) (7 * n + 8)) > 1 then
    Classical.choose h
  else
    0

theorem gcd_problem :
  least_reducible_n = 34 :=
sorry

end gcd_problem_l369_369030


namespace geometric_sequence_sum_l369_369067

theorem geometric_sequence_sum (a : ℕ → ℝ) (n : ℕ) 
  (h1 : ∀ m, a (m + 1) = a m * 2)
  (h2 : a 1 + a 4 = 9) 
  (h3 : a 2 * a 3 = 8) : 
  (finset.sum (finset.range n) a) = 2^n - 1 :=
sorry

end geometric_sequence_sum_l369_369067


namespace smallest_positive_period_pi_max_value_one_l369_369823

def f (x : Real) : Real := sqrt 3 * cos x ^ 2 + 1 / 2 * sin (2 * x) - sqrt 3 / 2

theorem smallest_positive_period_pi : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = π :=
by 
  sorry

theorem max_value_one : ∀ x, f x ≤ 1 ∧ ∃ x₀, f x₀ = 1 :=
by 
  sorry

end smallest_positive_period_pi_max_value_one_l369_369823


namespace gcd_of_18_and_30_l369_369563

-- Define the numbers
def a := 18
def b := 30

-- The main theorem statement
theorem gcd_of_18_and_30 : Nat.gcd a b = 6 := by
  sorry

end gcd_of_18_and_30_l369_369563


namespace sin_function_value_l369_369805

noncomputable def f (x : ℝ) : ℝ := sin(2 * x - 5 * π / 6)

theorem sin_function_value :
  f (-5 * π / 12) = sqrt 3 / 2 := by
  sorry

end sin_function_value_l369_369805


namespace max_crosses_in_grid_l369_369343

theorem max_crosses_in_grid : ∀ (n : ℕ), n = 16 → (∃ X : ℕ, X = 30 ∧
  ∀ (i j : ℕ), i < n → j < n → 
    (∀ k, k < n → (i ≠ k → X ≠ k)) ∧ 
    (∀ l, l < n → (j ≠ l → X ≠ l))) :=
by
  sorry

end max_crosses_in_grid_l369_369343


namespace problem_statement_l369_369747

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - 5 * Real.pi / 6)

theorem problem_statement :
  (∀ (x : ℝ), (∀ a b : ℝ, a < b → x ∈ (Set.Ioc (Real.pi / 6) (2 * Real.pi / 3)) → f(x) < f(b)) → x = (Real.pi / 6) ∨ x = (2 * Real.pi / 3)) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by sorry

end problem_statement_l369_369747


namespace mean_score_of_all_students_l369_369996

variable (M : ℕ) (A : ℕ) (m a : ℕ)
variable (h1 : M = 82)
variable (h2 : A = 68)
variable (h3 : m / a = 4 / 5)

theorem mean_score_of_all_students : (133.6 * a) / ((9 / 5) * a) = 74 := by
  have h4 : (m : ℝ) = (4 / 5 : ℝ) * a := by
    sorry
  have h5 : (82 : ℝ) * m = 65.6 * a := by
    sorry
  have h6 : (68 : ℝ) * a := (68 : ℝ) * a := by
    sorry
  have h7 : 65.6 * a + 68 * a = 133.6 * a := by
    sorry
  have h8 : 4 / 5 * a + a = 9 / 5 * a := by
    sorry
  have h9 : (133.6 * a) / ((9 / 5) * a) = 74 := by
    sorry
  exact h9

end mean_score_of_all_students_l369_369996


namespace multiply_658217_99999_l369_369626

theorem multiply_658217_99999 : 658217 * 99999 = 65821034183 := 
by
  sorry

end multiply_658217_99999_l369_369626


namespace regular_polygon_sides_l369_369482

theorem regular_polygon_sides (n : ℕ) (h₁ : ∀ k : ℕ, k = n → 180 * (k - 2) = 140 * k) : n = 9 := by
  have h₂ := h₁ n rfl
  linarith

end regular_polygon_sides_l369_369482


namespace remainder_of_product_of_odd_primes_mod_32_l369_369280

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem remainder_of_product_of_odd_primes_mod_32 :
  (product odd_primes_less_than_32) % 32 = 23 :=
by sorry

end remainder_of_product_of_odd_primes_mod_32_l369_369280


namespace james_works_4_hours_l369_369951

/-- James does chores around the class, where:
- There are 3 bedrooms, 1 living room, and 2 bathrooms to clean.
- The bedrooms each take 20 minutes to clean.
- The living room takes as long as the 3 bedrooms combined.
- The bathroom takes twice as long as the living room.
- He also cleans the outside which takes twice as long as cleaning the house.
- He splits the chores with his 2 siblings who are just as fast as him.

Prove: James works 4 hours. -/
theorem james_works_4_hours : 
  let bedrooms := 3,
      living_room := 1,
      bathrooms := 2,
      clean_bedroom_time := 20,
      clean_living_room_time := 3 * clean_bedroom_time,
      clean_bathroom_time := 2 * clean_living_room_time,
      total_inside_time := bedrooms * clean_bedroom_time + living_room * clean_living_room_time + bathrooms * clean_bathroom_time,
      total_house_clean_time := total_inside_time / 60,
      clean_outside_time := 2 * total_house_clean_time,
      total_clean_time := total_house_clean_time + clean_outside_time,
      james_clean_time := total_clean_time / 3 in
  james_clean_time = 4 := 
by {
  let bedrooms := 3,
  let living_room := 1,
  let bathrooms := 2,
  let clean_bedroom_time := 20,
  let clean_living_room_time := 3 * clean_bedroom_time,
  let clean_bathroom_time := 2 * clean_living_room_time,
  let total_inside_time := bedrooms * clean_bedroom_time + living_room * clean_living_room_time + bathrooms * clean_bathroom_time,
  let total_house_clean_time := total_inside_time / 60,
  let clean_outside_time := 2 * total_house_clean_time,
  let total_clean_time := total_house_clean_time + clean_outside_time,
  let james_clean_time := total_clean_time / 3,
  exact sorry
}

end james_works_4_hours_l369_369951


namespace find_f_value_l369_369815

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

theorem find_f_value :
  f ( -5 * Real.pi / 12 ) = Real.sqrt 3 / 2 :=
by
  sorry

end find_f_value_l369_369815


namespace sin_monotonic_increasing_f_symmetric_axes_find_value_l369_369692

def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi * 5 / 6)

theorem sin_monotonic_increasing (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) :
  Monotone (λ x, f x) (Ioo a b) := 
sorry

theorem f_symmetric_axes (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) 
  (x y : ℝ) (hx : x = a + (b - a) * k) (hy : y = a + (b - a) * (-k)) : 
  f x = f y :=
sorry

theorem find_value :
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
sorry

end sin_monotonic_increasing_f_symmetric_axes_find_value_l369_369692


namespace det_matrix_example_l369_369849

def det_2x2 (a b c d : ℤ) : ℤ := a * d - b * c

theorem det_matrix_example : det_2x2 4 5 2 3 = 2 :=
by
  sorry

end det_matrix_example_l369_369849


namespace tony_correct_score_l369_369325

-- Definitions from conditions
def class_size : ℕ := 20
def initial_average : ℕ := 73
def corrected_average : ℕ := 74
def tony_misread_diff : ℤ := 16

-- Statement of proof problem
theorem tony_correct_score : 
  let initial_total_score := class_size * initial_average in
  let corrected_total_score := class_size * corrected_average in
  let total_score_diff := corrected_total_score - initial_total_score in
  let tony_corrected_score := (total_score_diff + tony_misread_diff) in
  tony_corrected_score = 36 :=
by
  sorry

end tony_correct_score_l369_369325


namespace num_sets_without_perfect_squares_l369_369131

theorem num_sets_without_perfect_squares :
  let S := λ i : ℕ, set_of (λ n : ℕ, 100 * i ≤ n ∧ n < 100 * (i + 1))
  in ((finset.range 1000).filter (λ i, ¬∃ k, (k * k) ∈ S i)).card = 2 := 
by
  let S := λ i : ℕ, set_of (λ n : ℕ, 100 * i ≤ n ∧ n < 100 * (i + 1))
  have h : (∀i : ℕ, ¬∃ k : ℕ, (k * k ≥ 100 * i ∧ k * k < 100 * (i + 1)) ↔ ¬∃ k : ℕ, ∃ i : ℕ, (k * k ≥ 100 * (i : ℕ) ∧ k * k < 100 * (i + 1))) := sorry
  have : ((finset.range 1000).filter (λ i, ¬∃ k, (k * k) ∈ S i)).card = 2 := sorry
  rw this at *
  exact rfl

end num_sets_without_perfect_squares_l369_369131


namespace find_f_neg_5pi_12_l369_369777

-- Conditions
def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

variables {ω φ : ℝ}
def f (x : ℝ) : ℝ := Real.sin (ω * x + φ)

axiom monotonicity_condition : is_monotonically_increasing f (π / 6) (2 * π / 3)
axiom symmetry_condition_1 : f (π / 6) = f (2 * π / 3)
axiom symmetry_condition_2 : f (-π / 6) = f (π / 2)

-- Goal
theorem find_f_neg_5pi_12 : f (- 5 * π / 12) = sqrt 3 / 2 := 
sorry

end find_f_neg_5pi_12_l369_369777


namespace find_a_l369_369646

noncomputable def normalDist (μ σ : ℝ) := sorry

theorem find_a (a : ℝ) (ξ : ℝ → ℝ) (h₁ : ∀ x, ξ x = normalDist 0 a) 
    (h₂ : Probability (ξ 1 > 1) = Probability (ξ (a-3) < a - 3)) : a = 2 :=
sorry

end find_a_l369_369646


namespace remainder_when_M_divided_by_32_l369_369138

-- Define M as the product of all odd primes less than 32.
def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_M_divided_by_32 :
    M % 32 = 1 := by
  -- We must prove that M % 32 = 1.
  sorry

end remainder_when_M_divided_by_32_l369_369138


namespace marc_total_spent_l369_369986

theorem marc_total_spent :
  let cost_model_cars := 20
  let num_model_cars := 5
  let cost_bottles_paint := 10
  let num_bottles_paint := 5
  let cost_paintbrushes := 2
  let num_paintbrushes := 5
  let total_cost := (cost_model_cars * num_model_cars) + (cost_bottles_paint * num_bottles_paint) + (cost_paintbrushes * num_paintbrushes)
  in total_cost = 160 := 
by
  sorry

end marc_total_spent_l369_369986


namespace find_x_l369_369625

theorem find_x (x : ℕ) (h : (85 + 32 / x : ℝ) * x = 9637) : x = 113 :=
sorry

end find_x_l369_369625


namespace percentage_per_annum_l369_369358

-- Definitions based on conditions
def BankersGain (BG : ℝ) : Prop := BG = 360
def BankersDiscount (BD : ℝ) : Prop := BD = 1360
def TimeYears (t : ℝ) : Prop := t = 3

-- Main theorem statement
theorem percentage_per_annum :
  ∃ (r : ℝ), BankersGain 360 ∧ BankersDiscount 1360 ∧ TimeYears 3 →
  r = 100 / 3 := 
by 
  intros BG BD t h, 
  sorry

end percentage_per_annum_l369_369358


namespace largest_tile_size_l369_369431

theorem largest_tile_size
  (length width : ℕ)
  (H1 : length = 378)
  (H2 : width = 595) :
  Nat.gcd length width = 7 :=
by
  sorry

end largest_tile_size_l369_369431


namespace gcd_18_30_l369_369575

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := 
by
  sorry

end gcd_18_30_l369_369575


namespace recurrence_relation_l369_369911

-- Define the function p_nk and prove the recurrence relation
def p (n k : ℕ) : ℝ := sorry

theorem recurrence_relation (n k : ℕ) (h : k < n) : 
  p n k = p (n-1) k - (1 / 2^k) * p (n-k) k + (1 / 2^k) :=
sorry

end recurrence_relation_l369_369911


namespace recurrence_relation_l369_369902

noncomputable def p (n k : ℕ) : ℚ := sorry

theorem recurrence_relation (n k : ℕ) (h : k < n) :
  p n k = p (n - 1) k - (1 / 2^k) * p (n - k) k + (1 / 2^k) :=
by sorry

end recurrence_relation_l369_369902


namespace remainder_of_M_l369_369274

def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_of_M : M % 32 = 31 := by
  -- Proof goes here
  sorry

end remainder_of_M_l369_369274


namespace set_difference_l369_369956

theorem set_difference (a b c : ℕ) (h0 : 0 < a) (h1 : a < c - 1) (h2 : 1 < b) (h3 : b < c) :
  (exists (r : ℕ → ℕ), (∀ k, 0 ≤ k ∧ k ≤ a → (r k ≡ k * b % c) ∧ (0 ≤ r k ∧ r k < c)) ∧ 
  (¬ (∀ k, 0 ≤ k ∧ k ≤ a → r k = k))) :=
by
  sorry

end set_difference_l369_369956


namespace probability_recurrence_relation_l369_369894

theorem probability_recurrence_relation (n k : ℕ) (h : k < n) :
  ∀ (p : ℕ → ℕ → ℝ), p n k = p (n-1) k - (1 / (2:ℝ)^k) * p (n-k) k + 1 / (2:ℝ)^k := 
sorry

end probability_recurrence_relation_l369_369894


namespace find_f_value_l369_369810

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

theorem find_f_value :
  f ( -5 * Real.pi / 12 ) = Real.sqrt 3 / 2 :=
by
  sorry

end find_f_value_l369_369810


namespace gcd_18_30_l369_369584

theorem gcd_18_30: Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l369_369584


namespace find_fx_value_l369_369726

noncomputable def f (x : Real) : Real :=
  sin (2 * x - 5 * Real.pi / 6)

theorem find_fx_value :
  (∀ x, f x = sin (2 * x - 5 * Real.pi / 6)) ∧ 
  (f (Real.pi / 6) = f (2 * Real.pi / 3)) ∧ 
  (∀ x1 x2, (Real.pi / 6 < x1 ∧ x1 < 2 * Real.pi / 3 ∧ x1 < x2 ∧ x2 < 2 * Real.pi / 3) -> f x1 < f x2) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 := by
  sorry

end find_fx_value_l369_369726


namespace smallest_positive_debt_l369_369405

theorem smallest_positive_debt :
 ∃ (D : ℕ), D > 0 ∧ (240 ∣ D) ∧ (180 ∣ D) ∧ (∀ d : ℕ, (d > 0 ∧ (240 ∣ d) ∧ (180 ∣ d)) → D ≤ d) :=
begin
  use 60,
  split,
  { exact nat.zero_lt_bit1 (by simp), },
  split,
  { use (-1), ring, },
  split,
  { use 2, ring, },
  { intros d hd,
    cases hd with hd_pos hd_divs,
    cases hd_divs with hd_div_240 hd_div_180,
    have hd_div_60 : 60 ∣ d,
    { rw ← nat.gcd_eq_right hd_div_240,
      rw ← nat.gcd_eq_left hd_div_180,
      exact (nat.gcd_240_180.symm : nat.gcd 240 180 = 60), },
    exact (nat.le_of_dvd hd_pos hd_div_60), }
end

end smallest_positive_debt_l369_369405


namespace number_of_planes_l369_369116

-- Definitions based on the conditions
def Line (space: Type) := space → space → Prop

variables {space: Type} [MetricSpace space]

-- Given conditions
variable (l1 l2 l3 : Line space)
variable (intersects : ∃ p : space, l1 p p ∧ l2 p p ∧ l3 p p)

-- The theorem stating the conclusion
theorem number_of_planes (h: ∃ p : space, l1 p p ∧ l2 p p ∧ l3 p p) :
  (1 = 1 ∨ 1 = 2 ∨ 1 = 3) ∨ (2 = 1 ∨ 2 = 2 ∨ 2 = 3) ∨ (3 = 1 ∨ 3 = 2 ∨ 3 = 3) := 
sorry

end number_of_planes_l369_369116


namespace remainder_M_mod_32_l369_369249

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_M_mod_32 : M % 32 = 17 :=
by {
  sorry
}

end remainder_M_mod_32_l369_369249


namespace count_numbers_with_digit_sum_5_l369_369433

def digit_sum (n : ℕ) : ℕ :=
  n.digits.rev.sum

theorem count_numbers_with_digit_sum_5 : ∃ n : ℕ, n = 54 ∧ 
  (∀ k : ℕ, k < 7000 → sum_of_digits k = 5 → k ∈ ℕ) := sorry

end count_numbers_with_digit_sum_5_l369_369433


namespace remainder_of_M_when_divided_by_32_l369_369202

open Nat

def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
def M : ℕ := List.foldl (*) 1 oddPrimes

theorem remainder_of_M_when_divided_by_32 :
  M % 32 = 1 :=
sorry

end remainder_of_M_when_divided_by_32_l369_369202


namespace maximum_xy_value_l369_369866

theorem maximum_xy_value :
  ∃ (x y : ℕ), 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by
  sorry

end maximum_xy_value_l369_369866


namespace sum_smallest_largest_mean_l369_369035

-- Definitions based on given conditions
variables (y n : ℤ) (h : even (2*n))

-- Math statement on the sum of smallest and largest odd integers given mean y+1
theorem sum_smallest_largest_mean (y n : ℤ) (H : 2 ∣ 2*n) : 
  (∀ a : ℤ, (n > 0) →  (y + 1 = (1 : ℝ) / (2 * n) * (∑ i in finset.range (2*n), (a + 2*i : ℝ))) → 
  2 * a + 2*(2*n-1) = 2 * y) :=
  sorry


end sum_smallest_largest_mean_l369_369035


namespace sum_of_distances_l369_369916

theorem sum_of_distances (P : ℤ × ℤ) (hP : P = (-1, -2)) :
  abs P.1 + abs P.2 = 3 :=
sorry

end sum_of_distances_l369_369916


namespace value_of_f_neg_5π_over_12_l369_369790

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem value_of_f_neg_5π_over_12 :
  ∀ (ω φ : ℝ), 
    (∀ x y : ℝ, (x < y ∧ x ∈ Ioo (π / 6) (2 * π / 3) ∧ y ∈ Ioo (π / 6) (2 * π / 3)) → f x ω φ < f y ω φ) ∧ 
    f (π / 6) ω φ = f (2 * π / 3) ω φ → 
    f (-5 * π / 12) ω φ = √3 / 2 :=
by
  sorry

end value_of_f_neg_5π_over_12_l369_369790


namespace find_value_of_f_eq_l369_369763

noncomputable def f (ω ω φ x : ℝ) : ℝ := sin (ω * x + φ)

theorem find_value_of_f_eq :
  ∀ (ω φ : ℝ), 
    (∀ x, x ∈ set.Ioo (π / 6) (2 * π / 3) → (f ω φ x) ≤ (f ω φ (x + 1e-10))) → -- monotonically increasing
    (ω * ∓ (π / 6) + φ) = (φ + ω * (2 * π / 3)) → -- symmetric axes
    f ω φ (-(5 * π / 12)) = sqrt 3 / 2 :=
by
  sorry

end find_value_of_f_eq_l369_369763


namespace number_of_values_g100_zero_l369_369971

def g_0 (x : ℝ) : ℝ := x + |x - 50| - |x + 50|

def g : ℕ → ℝ → ℝ
| 0, x     := g_0 x
| (n + 1), x := |g n x| - 2

-- Define a predicate to check when g100(x) = 0
def g100_eq_zero (x : ℝ) : Prop := g 100 x = 0

-- Define a subset of real numbers such that g100(x) = 0
def values_g100_zero : set ℝ := {x : ℝ | g100_eq_zero x}

-- Prove that the size of this set is 103
theorem number_of_values_g100_zero : (values_g100_zero.to_finset.card = 103) := sorry

end number_of_values_g100_zero_l369_369971


namespace remainder_when_divided_by_32_l369_369152

def product_of_odds : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_divided_by_32 : product_of_odds % 32 = 9 := by
  sorry

end remainder_when_divided_by_32_l369_369152


namespace gcd_18_30_l369_369582

theorem gcd_18_30: Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l369_369582


namespace product_of_odd_primes_mod_32_l369_369175

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def M : ℕ := odd_primes_less_than_32.foldr (· * ·) 1

theorem product_of_odd_primes_mod_32 : M % 32 = 17 :=
by
  -- the proof goes here
  sorry

end product_of_odd_primes_mod_32_l369_369175


namespace A_holds_15_l369_369398

def cards : List (ℕ × ℕ) := [(1, 3), (1, 5), (3, 5)]

variables (A_card B_card C_card : ℕ × ℕ)

-- Conditions from the problem
def C_not_35 : Prop := C_card ≠ (3, 5)
def A_says_not_3 (A_card B_card : ℕ × ℕ) : Prop := ¬(A_card.1 = 3 ∧ B_card.1 = 3 ∨ A_card.2 = 3 ∧ B_card.2 = 3)
def B_says_not_1 (B_card C_card : ℕ × ℕ) : Prop := ¬(B_card.1 = 1 ∧ C_card.1 = 1 ∨ B_card.2 = 1 ∧ C_card.2 = 1)

-- Question to prove
theorem A_holds_15 : 
  ∃ (A_card B_card C_card : ℕ × ℕ),
    A_card ∈ cards ∧ B_card ∈ cards ∧ C_card ∈ cards ∧
    A_card ≠ B_card ∧ B_card ≠ C_card ∧ A_card ≠ C_card ∧
    C_not_35 C_card ∧
    A_says_not_3 A_card B_card ∧
    B_says_not_1 B_card C_card ->
    A_card = (1, 5) :=
sorry

end A_holds_15_l369_369398


namespace product_of_odd_primes_mod_32_l369_369229

theorem product_of_odd_primes_mod_32 :
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31) in
  M % 32 = 25 :=
by
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31)
  sorry

end product_of_odd_primes_mod_32_l369_369229


namespace find_f_of_neg_5_pi_over_12_l369_369738

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - (5 * π / 6))

theorem find_f_of_neg_5_pi_over_12 :
  (∀ x ∈ (set.Ioo (π / 6) (2 * π / 3)), (0 : ℝ) < (f x - f (x - 1))) ∧
  (∀ x, f (π / 6 - x) = f (π / 6 + x)) ∧
  (∀ x, f (2 * π / 3 - x) = f (2 * π / 3 + x)) →
  f (-5 * π / 12) = √3 / 2 :=
by
  sorry

end find_f_of_neg_5_pi_over_12_l369_369738


namespace triangle_area_l369_369921

theorem triangle_area (a b c : ℕ) (h : a = 8 ∧ b = 15 ∧ c = 17) :
  let A := if a * a + b * b = c * c then (a * b) / 2 else 0 in
  A = 60 :=
by
  sorry

end triangle_area_l369_369921


namespace new_profit_percentage_l369_369505

theorem new_profit_percentage (P : ℝ) (h1 : 1.10 * P = 990) (h2 : 0.90 * P * (1 + 0.30) = 1053) : 0.30 = 0.30 :=
by sorry

end new_profit_percentage_l369_369505


namespace greatest_xy_value_l369_369874

theorem greatest_xy_value (x y : ℕ) (h1 : 7 * x + 4 * y = 140) (h2 : x > 0) (h3 : y > 0) : 
  xy ≤ 112 :=
by
  sorry

end greatest_xy_value_l369_369874


namespace find_value_of_f_eq_l369_369769

noncomputable def f (ω ω φ x : ℝ) : ℝ := sin (ω * x + φ)

theorem find_value_of_f_eq :
  ∀ (ω φ : ℝ), 
    (∀ x, x ∈ set.Ioo (π / 6) (2 * π / 3) → (f ω φ x) ≤ (f ω φ (x + 1e-10))) → -- monotonically increasing
    (ω * ∓ (π / 6) + φ) = (φ + ω * (2 * π / 3)) → -- symmetric axes
    f ω φ (-(5 * π / 12)) = sqrt 3 / 2 :=
by
  sorry

end find_value_of_f_eq_l369_369769


namespace max_xy_l369_369864

theorem max_xy (x y : ℕ) (h1: 7 * x + 4 * y = 140) : ∃ x y, 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by {
  sorry
}

end max_xy_l369_369864


namespace cake_cost_l369_369324

theorem cake_cost
  (d : ℕ)
  (seven_days_comp : d + 700 = total_compensation_7)
  (total_compensation_7 : ℕ : total_compensation_7 = 7 * daily_compensation)
  (four_days_comp : d + 340 = 4 * daily_compensation)
  (daily_compensation : ℕ : daily_compensation = 120)
  : d = 140 := 
sorry 

end cake_cost_l369_369324


namespace initial_cost_of_car_l369_369127

/--
John made $30,000 doing Uber without factoring in the cost of depreciation for his car.
When he finally traded in the car he got $6,000 back.
His profit from driving Uber was $18,000.
Prove that the initial cost of the car is $24,000.
-/
theorem initial_cost_of_car (profit : ℕ) (trade_in : ℕ) : profit = 18000 → trade_in = 6000 → profit + trade_in = 24000 := by
  intros h_profit h_trade_in
  rw [h_profit, h_trade_in]
  rfl

end initial_cost_of_car_l369_369127


namespace first_digit_base8_of_395_l369_369410

theorem first_digit_base8_of_395 : 
  (∃ d : ℕ, d ∈ {0, 1, 2, 3, 4, 5, 6, 7} ∧ (395 : ℕ) = d * (8 ^ 2) + r ∧ r < (8 ^ 2)) := 
by
  use 6
  split
  {
    simp [nat.mem_set_iff]
  }
  split
  sorry
  

end first_digit_base8_of_395_l369_369410


namespace gcd_18_30_l369_369581

theorem gcd_18_30: Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l369_369581


namespace angle_ECD_eq_angle_FCD_l369_369938

-- Variable declarations
variables {A B C D E F G : Type}
variables [rectABCD : Rectangle ABCD]
variables (angleEAB_eq_angleFAB : angle E A B = angle F A B)
variables (angleFAB_gt_angleBAC : angle F A B > angle B A C)
variables (EF_midpoint_G : midpoint G E F)
variables (G_on_BD : lies_on G B D)

-- Proof statement
theorem angle_ECD_eq_angle_FCD : angle E C D = angle F C D :=
by sorry

end angle_ECD_eq_angle_FCD_l369_369938


namespace find_fx_value_l369_369730

noncomputable def f (x : Real) : Real :=
  sin (2 * x - 5 * Real.pi / 6)

theorem find_fx_value :
  (∀ x, f x = sin (2 * x - 5 * Real.pi / 6)) ∧ 
  (f (Real.pi / 6) = f (2 * Real.pi / 3)) ∧ 
  (∀ x1 x2, (Real.pi / 6 < x1 ∧ x1 < 2 * Real.pi / 3 ∧ x1 < x2 ∧ x2 < 2 * Real.pi / 3) -> f x1 < f x2) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 := by
  sorry

end find_fx_value_l369_369730


namespace product_of_odd_primes_mod_32_l369_369230

theorem product_of_odd_primes_mod_32 :
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31) in
  M % 32 = 25 :=
by
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31)
  sorry

end product_of_odd_primes_mod_32_l369_369230


namespace regular_polygon_sides_l369_369475

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 1 ≤ i → i ≤ n → interior_angle i = 140) (h2 : sum_of_interior_angles n = 180 * (n - 2)) :
  n = 9 :=
by
  sorry

end regular_polygon_sides_l369_369475


namespace unit_vector_opposite_direction_l369_369635

theorem unit_vector_opposite_direction (a : ℝ × ℝ) (a_val : a = (4, 2)) : 
  let neg_a := (-a.1, -a.2) in
  let mag_a := real.sqrt (a.1 * a.1 + a.2 * a.2) in
  let unit_vec := (neg_a.1 / mag_a, neg_a.2 / mag_a) in
  unit_vec = (- (2 * real.sqrt 5) / 5, - (real.sqrt 5) / 5) :=
by
  sorry

end unit_vector_opposite_direction_l369_369635


namespace remainder_of_M_l369_369268

def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_of_M : M % 32 = 31 := by
  -- Proof goes here
  sorry

end remainder_of_M_l369_369268


namespace product_of_odd_primes_mod_32_l369_369233

theorem product_of_odd_primes_mod_32 :
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31) in
  M % 32 = 25 :=
by
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31)
  sorry

end product_of_odd_primes_mod_32_l369_369233


namespace log_3_expression_result_l369_369012

theorem log_3_expression_result :
  ∃ (x : ℝ), 0 < x ∧ x = Real.log 3 (64 + Real.log 3 (64 + Real.log 3 (64 + ...))) ∧ x ≈ 4 :=
by
  sorry

end log_3_expression_result_l369_369012


namespace gcd_of_18_and_30_l369_369554

theorem gcd_of_18_and_30 : Nat.gcd 18 30 = 6 :=
by
  sorry

end gcd_of_18_and_30_l369_369554


namespace sin_function_value_l369_369806

noncomputable def f (x : ℝ) : ℝ := sin(2 * x - 5 * π / 6)

theorem sin_function_value :
  f (-5 * π / 12) = sqrt 3 / 2 := by
  sorry

end sin_function_value_l369_369806


namespace remainder_when_M_divided_by_32_l369_369148

-- Define M as the product of all odd primes less than 32.
def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_M_divided_by_32 :
    M % 32 = 1 := by
  -- We must prove that M % 32 = 1.
  sorry

end remainder_when_M_divided_by_32_l369_369148


namespace find_x2_minus_x1_l369_369317

theorem find_x2_minus_x1 (a x1 x2 d e : ℝ) (h_a : a ≠ 0) (h_d : d ≠ 0) (h_x : x1 ≠ x2) (h_e : e = -d * x1)
  (h_y1 : ∀ x, y1 = a * (x - x1) * (x - x2)) (h_y2 : ∀ x, y2 = d * x + e)
  (h_intersect : ∀ x, y = a * (x - x1) * (x - x2) + (d * x + e)) 
  (h_single_point : ∀ x, y = a * (x - x1)^2) :
  x2 - x1 = d / a :=
sorry

end find_x2_minus_x1_l369_369317


namespace find_f_of_neg_5_pi_over_12_l369_369742

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - (5 * π / 6))

theorem find_f_of_neg_5_pi_over_12 :
  (∀ x ∈ (set.Ioo (π / 6) (2 * π / 3)), (0 : ℝ) < (f x - f (x - 1))) ∧
  (∀ x, f (π / 6 - x) = f (π / 6 + x)) ∧
  (∀ x, f (2 * π / 3 - x) = f (2 * π / 3 + x)) →
  f (-5 * π / 12) = √3 / 2 :=
by
  sorry

end find_f_of_neg_5_pi_over_12_l369_369742


namespace bisection_method_interval_l369_369969

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 5

theorem bisection_method_interval :
  f 1 < 0 ∧ f 2 > 0 ∧ f 1.5 < 0 → ∃ x ∈ Ioo 1.5 2, f x = 0 := by
  sorry

end bisection_method_interval_l369_369969


namespace average_bowling_score_l369_369085

-- Definitions of the scores
def g : ℕ := 120
def m : ℕ := 113
def b : ℕ := 85

-- Theorem statement: The average score is 106
theorem average_bowling_score : (g + m + b) / 3 = 106 := by
  sorry

end average_bowling_score_l369_369085


namespace measure_angle_CAD_is_6_l369_369340

-- Definitions of regular pentagon and equilateral triangle
def regular_pentagon (P : Fin 5 → ℝ×ℝ) : Prop :=
∀ i, ∥P i - P ((i + 1) % 5)∥ = ∥P 0 - P 1∥ ∧
∠(P i, P ((i + 1) % 5), P ((i + 2) % 5)) = 108

def equilateral_triangle (T : Fin 3 → ℝ×ℝ) : Prop :=
∀ i, ∥T i - T ((i + 1) % 3)∥ = ∥T 0 - T 1∥ ∧
∠(T i, T ((i + 1) % 3), T ((i + 2) % 3)) = 60

-- Function to determine proofs for angles
noncomputable def measure_angle (A B C : ℝ×ℝ) : ℝ := sorry

-- Coplanar condition with vertex B shared
def coplanar_shared_vertex (B : ℝ×ℝ) (P : Fin 5 → ℝ×ℝ) (T : Fin 3 → ℝ×ℝ) : Prop :=
B = P 0 ∧ B = T 1 ∧
∃ C : ℝ×ℝ, (∥T 0 - C∥ = ∥C - T 2∥) ∧ regular_pentagon P ∧ equilateral_triangle T

-- The statement to prove
theorem measure_angle_CAD_is_6 {B : ℝ×ℝ} {P : Fin 5 → ℝ×ℝ} {T : Fin 3 → ℝ×ℝ} 
(hPentagon : regular_pentagon P) (hTriangle : equilateral_triangle T) 
(hCoplanar : coplanar_shared_vertex B P T) :
measure_angle (T 0) (sorry) (T 2) = 6 := 
sorry

end measure_angle_CAD_is_6_l369_369340


namespace remainder_of_product_of_odd_primes_mod_32_l369_369285

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem remainder_of_product_of_odd_primes_mod_32 :
  (product odd_primes_less_than_32) % 32 = 23 :=
by sorry

end remainder_of_product_of_odd_primes_mod_32_l369_369285


namespace numberOfSubsets_of_A_l369_369372

def numberOfSubsets (s : Finset ℕ) : ℕ := 2 ^ (Finset.card s)

theorem numberOfSubsets_of_A : 
  numberOfSubsets ({0, 1} : Finset ℕ) = 4 := 
by 
  sorry

end numberOfSubsets_of_A_l369_369372


namespace find_value_of_f_eq_l369_369768

noncomputable def f (ω ω φ x : ℝ) : ℝ := sin (ω * x + φ)

theorem find_value_of_f_eq :
  ∀ (ω φ : ℝ), 
    (∀ x, x ∈ set.Ioo (π / 6) (2 * π / 3) → (f ω φ x) ≤ (f ω φ (x + 1e-10))) → -- monotonically increasing
    (ω * ∓ (π / 6) + φ) = (φ + ω * (2 * π / 3)) → -- symmetric axes
    f ω φ (-(5 * π / 12)) = sqrt 3 / 2 :=
by
  sorry

end find_value_of_f_eq_l369_369768


namespace find_f_neg_5pi_12_l369_369774

-- Conditions
def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

variables {ω φ : ℝ}
def f (x : ℝ) : ℝ := Real.sin (ω * x + φ)

axiom monotonicity_condition : is_monotonically_increasing f (π / 6) (2 * π / 3)
axiom symmetry_condition_1 : f (π / 6) = f (2 * π / 3)
axiom symmetry_condition_2 : f (-π / 6) = f (π / 2)

-- Goal
theorem find_f_neg_5pi_12 : f (- 5 * π / 12) = sqrt 3 / 2 := 
sorry

end find_f_neg_5pi_12_l369_369774


namespace number_of_correct_conclusions_l369_369674

theorem number_of_correct_conclusions : 
  (¬(p ∧ ¬q → p ∧ q)) ∧ 
  (¬(xy = 0 → (x = 0 ∨ y = 0)) ↔ (xy ≠ 0 → (x ≠ 0 ∧ y ≠ 0))) ∧ 
  (¬(∀ x : ℝ, 2^x > 0) ↔ (∃ x : ℝ, 2^x ≤ 0)) → 
  1 :=
by
  sorry

end number_of_correct_conclusions_l369_369674


namespace probability_sum_even_and_greater_than_15_l369_369400

theorem probability_sum_even_and_greater_than_15 (dice_faces : Fin 6 → ℕ) (h_faces : ∀ n, 1 ≤ dice_faces n ∧ dice_faces n ≤ 6) :
  ((∃ x y z : ℕ, x = dice_faces 0 ∧ y = dice_faces 1 ∧ z = dice_faces 2 ∧ (x + y + z) % 2 = 0 ∧ x + y + z > 15) → 
   (Real.toRat (10 / 216) = Rat.ofInt 5 / 108)) :=
sorry

end probability_sum_even_and_greater_than_15_l369_369400


namespace problem1_problem2_l369_369443

-- Problem 1: Prove that the given expression evaluates to 0
theorem problem1 : -real.sqrt 3 + (-5 / 2)^0 + abs (1 - real.sqrt 3) = 0 := 
by sorry

-- Problem 2: Solve the system of linear equations and prove that (1, 2) is a solution
theorem problem2 : ∃ x y : ℝ, (4 * x + 3 * y = 10) ∧ (3 * x + y = 5) ∧ (x = 1) ∧ (y = 2) :=
by sorry

end problem1_problem2_l369_369443


namespace existence_condition_l369_369959

variables {M : Type*} (A B C : Set M)

theorem existence_condition (M : Type*) (A B C : Set M) :
  (A \cap Bᶜ \cap Cᶜ = ∅ ∧ Aᶜ \cap B \cap C = ∅) ↔ ∃ (X : Set M), (X ∪ A) \ B = C :=
by sorry

end existence_condition_l369_369959


namespace length_segment_DE_l369_369113

noncomputable def triangle_equilateral :=
{A B C : Type} [metric_space A] [euclidean_geometry A] 
(side_len_eq : ∀ {x y z : A}, is_equilateral x y z → dist x y = dist y z)

noncomputable def points_on_side_AB :=
{G F H J : Type} [metric_space G] [euclidean_geometry G]
(dist_AG : dist A G = 5) 
(dist_GF : dist G F = 8)
(dist_HJ : dist H J = 12)
(dist_FC : dist F C = 5)

noncomputable def points_on_side_AC_BC :=
{ D E : Type} [metric_space D] [euclidean_geometry D] [metric_space E] [euclidean_geometry E] 
(dist_AD : dist A D = 10) 
(dist_BE : dist B E = 20)
(is_midpoint_D : midpoint A E D)

theorem length_segment_DE
(triangle_equilateral : ∀ {x y z : Type} [metric_space x] [euclidean_geometry x], (dist x y = 30) → (is_equilateral x y z → dist x y = dist y z))
(points_on_side_AB : ∀  {G F H J : Type} [metric_space G] [euclidean_geometry G], 
(dist A G = 5) ∧ (dist G F = 8) ∧ (dist H J = 12) ∧ (dist F C = 5))
(points_on_side_AC_BC : ∀ {D E : Type} [metric_space D] [euclidean_geometry D] [metric_space E] [euclidean_geometry E], 
(dist A D = 10) ∧ (dist B E = 20) ∧ (midpoint A E D)) :
(dist D E = 10) :=
begin
  sorry
end

end length_segment_DE_l369_369113


namespace sin_function_value_l369_369802

noncomputable def f (x : ℝ) : ℝ := sin(2 * x - 5 * π / 6)

theorem sin_function_value :
  f (-5 * π / 12) = sqrt 3 / 2 := by
  sorry

end sin_function_value_l369_369802


namespace find_a_find_sin_cos_diff_l369_369061

variable (θ : ℝ) (a : ℝ)

-- Given conditions
def condition1 : Prop := (sin θ) = (cos (θ + π/2))
def condition2 : Prop := (cos θ) = (sin (θ + π/2))
def roots_condition (a : ℝ) (θ : ℝ) : Prop :=
  let r1 := sin θ
  let r2 := cos θ
  r1 + r2 = 2 * sqrt 2 * a ∧ r1 * r2 = a

-- First part of the problem
theorem find_a (a : ℝ) (θ : ℝ) (h : roots_condition a θ) : 8 * a^2 - 2 * a - 1 = 0 :=
sorry

-- Second part of the problem
theorem find_sin_cos_diff (a : ℝ) (θ : ℝ) (h : roots_condition a θ) (hθ : θ ∈ Ioo (-π/2) 0) :
  a = -1/4 → sin θ - cos θ = -sqrt 6 / 2 :=
sorry

end find_a_find_sin_cos_diff_l369_369061


namespace find_CM_of_trapezoid_l369_369939

noncomputable def trapezoid_CM (AD BC : ℝ) (M : ℝ) : ℝ :=
  if (AD = 12) ∧ (BC = 8) ∧ (M = 2.4)
  then M
  else 0

theorem find_CM_of_trapezoid (trapezoid_ABCD : Type) (AD BC CM : ℝ) (AM_divides_eq_areas : Prop) :
  AD = 12 → BC = 8 → AM_divides_eq_areas → CM = 2.4 := 
by
  intros h1 h2 h3
  have : AD = 12 := h1
  have : BC = 8 := h2
  have : CM = 2.4 := sorry
  exact this

end find_CM_of_trapezoid_l369_369939


namespace max_product_l369_369882

theorem max_product (x y : ℕ) (h1 : 7 * x + 4 * y = 140) : x * y ≤ 168 :=
sorry

end max_product_l369_369882


namespace remainder_when_M_divided_by_32_l369_369140

-- Define M as the product of all odd primes less than 32.
def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_M_divided_by_32 :
    M % 32 = 1 := by
  -- We must prove that M % 32 = 1.
  sorry

end remainder_when_M_divided_by_32_l369_369140


namespace first_digit_base8_of_395_l369_369411

theorem first_digit_base8_of_395 : 
  (∃ d : ℕ, d ∈ {0, 1, 2, 3, 4, 5, 6, 7} ∧ (395 : ℕ) = d * (8 ^ 2) + r ∧ r < (8 ^ 2)) := 
by
  use 6
  split
  {
    simp [nat.mem_set_iff]
  }
  split
  sorry
  

end first_digit_base8_of_395_l369_369411


namespace largest_integer_divides_Q_l369_369516

theorem largest_integer_divides_Q (n : ℤ) :
  ∃ Q : ℤ, Q = (2*n - 3) * (2*n - 1) * (2*n + 1) * (2*n + 3) ∧ ∀ m : ℤ, (m | Q) → m ≤ 3 := 
sorry

end largest_integer_divides_Q_l369_369516


namespace prime_product_mod_32_l369_369300

open Nat

theorem prime_product_mod_32 : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := List.foldr (· * ·) 1 primes
  M % 32 = 25 :=
by
  sorry

end prime_product_mod_32_l369_369300


namespace variance_of_numbers_l369_369037

def mean (l : List ℝ) : ℝ := l.sum / l.length

def variance (l : List ℝ) : ℝ :=
  let μ := mean l
  (l.map (λ x => (x - μ)^2)).sum / l.length

theorem variance_of_numbers :
  variance [8, 8, 9, 10] = 11 / 16 :=
by
  sorry

end variance_of_numbers_l369_369037


namespace smallest_base10_integer_exists_l369_369419

theorem smallest_base10_integer_exists : ∃ (n a b : ℕ), a > 2 ∧ b > 2 ∧ n = 1 * a + 3 ∧ n = 3 * b + 1 ∧ n = 10 := by
  sorry

end smallest_base10_integer_exists_l369_369419


namespace find_fx_value_l369_369728

noncomputable def f (x : Real) : Real :=
  sin (2 * x - 5 * Real.pi / 6)

theorem find_fx_value :
  (∀ x, f x = sin (2 * x - 5 * Real.pi / 6)) ∧ 
  (f (Real.pi / 6) = f (2 * Real.pi / 3)) ∧ 
  (∀ x1 x2, (Real.pi / 6 < x1 ∧ x1 < 2 * Real.pi / 3 ∧ x1 < x2 ∧ x2 < 2 * Real.pi / 3) -> f x1 < f x2) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 := by
  sorry

end find_fx_value_l369_369728


namespace marc_total_spent_l369_369984

theorem marc_total_spent
  (model_cars : ℕ) (cost_model_car : ℕ)
  (paint_bottles : ℕ) (cost_paint_bottle : ℕ)
  (paintbrushes : ℕ) (cost_paintbrush : ℕ) :
  model_cars = 5 →
  cost_model_car = 20 →
  paint_bottles = 5 →
  cost_paint_bottle = 10 →
  paintbrushes = 5 →
  cost_paintbrush = 2 →
  let total_cost := (model_cars * cost_model_car) + (paint_bottles * cost_paint_bottle) + (paintbrushes * cost_paintbrush)
  in total_cost = 160 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  dsimp [Nat.mul, Nat.add]
  sorry

end marc_total_spent_l369_369984


namespace marc_total_spent_l369_369988

theorem marc_total_spent :
  let model_car_cost := 20
  let paint_cost := 10
  let brush_cost := 2
  let model_car_amount := 5
  let paint_amount := 5
  let brush_amount := 5
  let total_cost := model_car_amount * model_car_cost + paint_amount * paint_cost + brush_amount * brush_cost
  in total_cost = 160 :=
by
  sorry

end marc_total_spent_l369_369988


namespace sin_monotonic_increasing_f_symmetric_axes_find_value_l369_369694

def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi * 5 / 6)

theorem sin_monotonic_increasing (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) :
  Monotone (λ x, f x) (Ioo a b) := 
sorry

theorem f_symmetric_axes (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) 
  (x y : ℝ) (hx : x = a + (b - a) * k) (hy : y = a + (b - a) * (-k)) : 
  f x = f y :=
sorry

theorem find_value :
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
sorry

end sin_monotonic_increasing_f_symmetric_axes_find_value_l369_369694


namespace dice_win_properties_l369_369327

theorem dice_win_properties:
  ∃ (A B C : List ℕ),
    A = [1, 4, 4, 4, 4, 4] ∧ 
    B = [2, 2, 2, 5, 5, 5] ∧ 
    C = [3, 3, 3, 3, 3, 6] ∧
    ((Probability (λ a b, a > b) A B > 1/2) ∧
     (Probability (λ b c, b > c) C B > 1/2) ∧
     (Probability (λ c a, a > c) A C > 1/2)) :=
by
  let A := [1, 4, 4, 4, 4, 4]
  let B := [2, 2, 2, 5, 5, 5]
  let C := [3, 3, 3, 3, 3, 6]

  have h1: (Probability (λ a b, a > b) A B > 1/2),
  sorry

  have h2: (Probability (λ b c, b > c) C B > 1/2),
  sorry

  have h3: (Probability (λ c a, a > c) A C > 1/2),
  sorry

  exact ⟨A, B, C, rfl, rfl, rfl, ⟨h1, h2, h3⟩⟩

end dice_win_properties_l369_369327


namespace remainder_M_mod_32_l369_369256

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_M_mod_32 : M % 32 = 17 :=
by {
  sorry
}

end remainder_M_mod_32_l369_369256


namespace find_value_of_f_eq_l369_369764

noncomputable def f (ω ω φ x : ℝ) : ℝ := sin (ω * x + φ)

theorem find_value_of_f_eq :
  ∀ (ω φ : ℝ), 
    (∀ x, x ∈ set.Ioo (π / 6) (2 * π / 3) → (f ω φ x) ≤ (f ω φ (x + 1e-10))) → -- monotonically increasing
    (ω * ∓ (π / 6) + φ) = (φ + ω * (2 * π / 3)) → -- symmetric axes
    f ω φ (-(5 * π / 12)) = sqrt 3 / 2 :=
by
  sorry

end find_value_of_f_eq_l369_369764


namespace no_Hamiltonian_path_2018_grid_l369_369999

theorem no_Hamiltonian_path_2018_grid :
  ¬ ∃ (path : List (Fin 2018 × Fin 2018)),
    (∀ (i j : Fin 2018), (i, j) ∈ path) ∧
    (∀ (i j : Fin 2018) (i' j' : Fin 2018),
         (i, j) ≠ (i', j') → List.Pairwise (λ x y, x.1 ≠ y.1 ∨ x.2 ≠ y.2) path) ∧
    path.head = (⟨0, by decide⟩, ⟨0, by decide⟩) ∧
    path.last = (⟨2017, by decide⟩, ⟨2017, by decide⟩) :=
by
  sorry

end no_Hamiltonian_path_2018_grid_l369_369999


namespace find_fx_value_l369_369724

noncomputable def f (x : Real) : Real :=
  sin (2 * x - 5 * Real.pi / 6)

theorem find_fx_value :
  (∀ x, f x = sin (2 * x - 5 * Real.pi / 6)) ∧ 
  (f (Real.pi / 6) = f (2 * Real.pi / 3)) ∧ 
  (∀ x1 x2, (Real.pi / 6 < x1 ∧ x1 < 2 * Real.pi / 3 ∧ x1 < x2 ∧ x2 < 2 * Real.pi / 3) -> f x1 < f x2) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 := by
  sorry

end find_fx_value_l369_369724


namespace remainder_of_product_of_odd_primes_mod_32_l369_369281

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem remainder_of_product_of_odd_primes_mod_32 :
  (product odd_primes_less_than_32) % 32 = 23 :=
by sorry

end remainder_of_product_of_odd_primes_mod_32_l369_369281


namespace remainder_M_mod_32_l369_369260

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_M_mod_32 : M % 32 = 17 :=
by {
  sorry
}

end remainder_M_mod_32_l369_369260


namespace problem_statement_l369_369753

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - 5 * Real.pi / 6)

theorem problem_statement :
  (∀ (x : ℝ), (∀ a b : ℝ, a < b → x ∈ (Set.Ioc (Real.pi / 6) (2 * Real.pi / 3)) → f(x) < f(b)) → x = (Real.pi / 6) ∨ x = (2 * Real.pi / 3)) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by sorry

end problem_statement_l369_369753


namespace mines_placement_l369_369017

-- Define a grid type
inductive Cell
| mine : Cell
| free : Nat → Cell  -- free : Nat indicates the number of neighboring mines

def countWays (grid : Array (Array Cell)) (m : Nat) : Nat := sorry

def exampleGrid : Array (Array Cell) := 
  #[#[Cell.free 2, Cell.free 3, Cell.mine, Cell.mine],
    #[Cell.mine, Cell.mine, Cell.mine, Cell.mine],
    #[Cell.mine, Cell.mine, Cell.mine, Cell.mine]]

-- Proves that there are 72 ways to place the mines in the grid
theorem mines_placement (grid : Array (Array Cell)) (m : Nat) :
  grid = exampleGrid → (countWays grid m) = 72 :=
by
  -- Assumptions about the grid and m
  assume h1 : grid = exampleGrid,
  assume h2 : m = 8,  -- Assume 8 mines since 3 * 4 grid - 2 free cells
  sorry

end mines_placement_l369_369017


namespace centroid_positions_count_l369_369003

theorem centroid_positions_count :
  let square_side := 12
  let interval_count := 12
  let point_count := (4 * interval_count)
  let possible_centroid_positions := (point_count - 3) ^ 2
  possible_centroid_positions = 1225 :=
by
  let square_side := 12
  let interval_count := 12
  let point_count := (4 * interval_count)
  let possible_centroid_positions := (point_count - 3) ^ 2
  have h : possible_centroid_positions = (35 * 35)
  { sorry }
  rw h
  norm_num

end centroid_positions_count_l369_369003


namespace solve_for_x_l369_369349

theorem solve_for_x (i x : ℂ) (h : i^2 = -1) (eq : 3 - 2 * i * x = 5 + 4 * i * x) : x = i / 3 := 
by
  sorry

end solve_for_x_l369_369349


namespace line_circle_intersection_l369_369370

-- Define the line and circle in Lean
def line_eq (x y : ℝ) : Prop := x + y - 6 = 0
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 5)^2 = 2

-- Define the proof about the intersection
theorem line_circle_intersection :
  (∃ x y : ℝ, line_eq x y ∧ circle_eq x y) ∧
  ∀ (x1 y1 x2 y2 : ℝ), (line_eq x1 y1 ∧ circle_eq x1 y1) → (line_eq x2 y2 ∧ circle_eq x2 y2) → (x1 = x2 ∧ y1 = y2) :=
by {
  sorry
}

end line_circle_intersection_l369_369370


namespace solve_for_x_l369_369350

theorem solve_for_x : ∃ x : ℝ, 10 - 2 * x = 14 ∧ x = -2 :=
by
  use -2
  split
  · show 10 - 2 * (-2) = 14
    ring
  · rfl

end solve_for_x_l369_369350


namespace find_f_value_l369_369816

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

theorem find_f_value :
  f ( -5 * Real.pi / 12 ) = Real.sqrt 3 / 2 :=
by
  sorry

end find_f_value_l369_369816


namespace num_solutions_4_lt_sqrt_3x_lt_6_l369_369009

noncomputable def num_integer_solutions (α : ℝ) (β : ℝ) (γ : ℝ) (δ : ℝ) : ℕ :=
  let num_int_satisfying (p q : ℝ) : ℕ :=
    ((Real.floor q).to_nat) - ((Real.ceil p).to_nat) + 1
  in
  if h : α < δ ∧ γ < β then num_int_satisfying α γ else 0

theorem num_solutions_4_lt_sqrt_3x_lt_6 : num_integer_solutions 16 (36 / 3) 4 6 = 6 := sorry

end num_solutions_4_lt_sqrt_3x_lt_6_l369_369009


namespace probability_recurrence_relation_l369_369890

theorem probability_recurrence_relation (n k : ℕ) (h : k < n) :
  ∀ (p : ℕ → ℕ → ℝ), p n k = p (n-1) k - (1 / (2:ℝ)^k) * p (n-k) k + 1 / (2:ℝ)^k := 
sorry

end probability_recurrence_relation_l369_369890


namespace general_term_sum_first_n_terms_l369_369066

-- Setup any constants or definitions we need
def a (n : ℕ) : ℝ := 2 * (3 : ℝ)^(n - 1)
def b (n : ℕ) : ℝ := real.log 3 (3^n / 2) + real.log 3 (a n)
def S (n : ℕ) : ℝ := (finset.range n).sum (λ k, a (k + 1) + b (k + 1))

-- Problem 1: Finding the general term formula for the sequence {a_n}
theorem general_term (q : ℝ) (h : q > 0) (h1 : a 1 = 2) (h2 : a 3 - a 2 = 12) :
  ∀ n, a n = 2 * 3^(n - 1) := 
sorry

-- Problem 2: Finding the sum of the first n terms, S_n, for the sequence {a_n + b_n}
theorem sum_first_n_terms (n : ℕ) : S n = 3^n - 1 + n^2 := 
sorry


end general_term_sum_first_n_terms_l369_369066


namespace monotonic_intervals_and_extreme_points_l369_369675

noncomputable def f (x a : ℝ) : ℝ := (1 / 2) * x^2 - (a + 1) * x + a * Real.log x

theorem monotonic_intervals_and_extreme_points (a : ℝ) (h : 1 < a) :
  ∃ x1 x2, x1 = 1 ∧ x2 = a ∧ x1 < x2 ∧ f x2 a < - (3 / 2) * x1 :=
by
  sorry

end monotonic_intervals_and_extreme_points_l369_369675


namespace verify_mangleY_l369_369322

noncomputable def mangleY (p q l : Line) (a b c : Angle) : Prop :=
  parallel p q 
  ∧ parallel p l 
  ∧ parallel q l 
  ∧ ∠ on p = a 
  ∧ ∠ on q = b 
  ∧ a = 100 
  ∧ b = 110 
  ∧ c = 70 
  ∧ a - c = 40

theorem verify_mangleY (p q l : Line) (a b c : Angle) :
  mangleY p q l a b c := sorry

end verify_mangleY_l369_369322


namespace prime_product_mod_32_l369_369297

open Nat

theorem prime_product_mod_32 : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := List.foldr (· * ·) 1 primes
  M % 32 = 25 :=
by
  sorry

end prime_product_mod_32_l369_369297


namespace find_f_neg_5pi_12_l369_369778

-- Conditions
def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

variables {ω φ : ℝ}
def f (x : ℝ) : ℝ := Real.sin (ω * x + φ)

axiom monotonicity_condition : is_monotonically_increasing f (π / 6) (2 * π / 3)
axiom symmetry_condition_1 : f (π / 6) = f (2 * π / 3)
axiom symmetry_condition_2 : f (-π / 6) = f (π / 2)

-- Goal
theorem find_f_neg_5pi_12 : f (- 5 * π / 12) = sqrt 3 / 2 := 
sorry

end find_f_neg_5pi_12_l369_369778


namespace count_perfect_squares_diff_of_consecutive_squares_l369_369337

-- Define the notion of a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

-- Define what it means to be the difference of two consecutive perfect squares
def is_diff_of_consecutive_squares (n : ℕ) : Prop :=
  ∃ b : ℕ, n = (b + 1) * (b + 1) - b * b

-- Define what it means to be an odd number
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Prove the main theorem
theorem count_perfect_squares_diff_of_consecutive_squares :
  {x : ℕ | x < 20000 ∧ is_perfect_square x ∧ is_diff_of_consecutive_squares x}.to_finset.card = 70 :=
by
  sorry

end count_perfect_squares_diff_of_consecutive_squares_l369_369337


namespace sin_function_value_l369_369799

noncomputable def f (x : ℝ) : ℝ := sin(2 * x - 5 * π / 6)

theorem sin_function_value :
  f (-5 * π / 12) = sqrt 3 / 2 := by
  sorry

end sin_function_value_l369_369799


namespace train_speed_l369_369491

/--
A train 50 m long passes a platform 100 m long in 10 seconds. 
Prove that the speed of the train is 15 m/sec.
-/
theorem train_speed (length_train length_platform : ℕ) (time : ℕ) (h1 : length_train = 50) (h2 : length_platform = 100) (h3 : time = 10) :
  (length_train + length_platform) / time = 15 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end train_speed_l369_369491


namespace product_of_odd_primes_mod_32_l369_369228

theorem product_of_odd_primes_mod_32 :
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31) in
  M % 32 = 25 :=
by
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31)
  sorry

end product_of_odd_primes_mod_32_l369_369228


namespace find_value_of_f_eq_l369_369759

noncomputable def f (ω ω φ x : ℝ) : ℝ := sin (ω * x + φ)

theorem find_value_of_f_eq :
  ∀ (ω φ : ℝ), 
    (∀ x, x ∈ set.Ioo (π / 6) (2 * π / 3) → (f ω φ x) ≤ (f ω φ (x + 1e-10))) → -- monotonically increasing
    (ω * ∓ (π / 6) + φ) = (φ + ω * (2 * π / 3)) → -- symmetric axes
    f ω φ (-(5 * π / 12)) = sqrt 3 / 2 :=
by
  sorry

end find_value_of_f_eq_l369_369759


namespace maximum_sum_l369_369053

-- Define the arithmetic sequence
def arithmetic_seq (a1 d n : ℕ) : ℕ := a1 + n * d

-- Given conditions
def a1_pos (a1 : ℕ) : Prop := a1 > 0
def cond (a1 d : ℕ) : Prop := 8 * (a1 + 4 * d) = 13 * (a1 + 10 * d)

-- Converting specific terms to check maximum
def term_20 (a1 : ℕ) (d : ℕ) : ℕ := a1 + 19 * d
def term_21 (a1 : ℕ) (d : ℕ) : ℕ := a1 + 20 * d

-- Lean theorem statement equivalent to the problem
theorem maximum_sum (a1 d : ℕ) (h1 : a1_pos a1) (h2 : cond a1 d) : 
  term_20 a1 d > 0 ∧ term_21 a1 d < 0 → (∀ n, a1 + n * d ≤ 20) :=
begin
  sorry
end

end maximum_sum_l369_369053


namespace points_count_l369_369467

theorem points_count :
  let A := (1, 0)
  let line1 := λ x, x = -1
  let parabola P := (P.1^2 / 4 = P.2)
  let distance_to_line P l := abs(P.1 - P.2) / real.sqrt 2
  in  [P | parabola P ∧ distance_to_line P (λ x, x = x) = real.sqrt 2 / 2].length = 3 :=
sorry

end points_count_l369_369467


namespace polygon_sides_sum_720_l369_369379

theorem polygon_sides_sum_720 (n : ℕ) (h1 : (n - 2) * 180 = 720) : n = 6 := by
  sorry

end polygon_sides_sum_720_l369_369379


namespace problem_statement_l369_369748

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - 5 * Real.pi / 6)

theorem problem_statement :
  (∀ (x : ℝ), (∀ a b : ℝ, a < b → x ∈ (Set.Ioc (Real.pi / 6) (2 * Real.pi / 3)) → f(x) < f(b)) → x = (Real.pi / 6) ∨ x = (2 * Real.pi / 3)) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by sorry

end problem_statement_l369_369748


namespace least_multiple_second_smallest_primes_l369_369414

/-- 
  The problem statement given in a way that asks to verify if the result 
  of the least common multiple of the second smallest set of four consecutive primes is 46189.
  Specifically, these four primes are 11, 13, 17, and 19.
--/
theorem least_multiple_second_smallest_primes : 
  let primes := [11, 13, 17, 19] in 
  primes.prod = 46189 :=
by {
  let primes := [11, 13, 17, 19],
  calc primes.prod = 11 * 13 * 17 * 19 : by rfl
              ... = 143 * 17       : by rw [mul_assoc 11 13 17, mul_comm 17 19]
              ... = 2431           : by norm_num
              ... = 2431 * 19      : by rw [mul_assoc 11 13 2431, mul_comm 17 2431]
              ... = 46189          : by norm_num,
  sorry
}

end least_multiple_second_smallest_primes_l369_369414


namespace recurrence_relation_l369_369903

noncomputable def p (n k : ℕ) : ℚ := sorry

theorem recurrence_relation (n k : ℕ) (h : k < n) :
  p n k = p (n - 1) k - (1 / 2^k) * p (n - k) k + (1 / 2^k) :=
by sorry

end recurrence_relation_l369_369903


namespace work_completion_l369_369428

theorem work_completion (rate_B rate_C : ℕ) (h1 : rate_B = 12) (h2 : rate_C = 15) :
  let rate_A := 2 * (1 / rate_B.toReal),
      combined_rate := rate_A + (1 / rate_B.toReal) + (1 / rate_C.toReal),
      days_to_complete := 1 / combined_rate
  in days_to_complete = (60 / 19) :=
by 
  sorry

end work_completion_l369_369428


namespace find_f_neg_5pi_12_l369_369782

-- Conditions
def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

variables {ω φ : ℝ}
def f (x : ℝ) : ℝ := Real.sin (ω * x + φ)

axiom monotonicity_condition : is_monotonically_increasing f (π / 6) (2 * π / 3)
axiom symmetry_condition_1 : f (π / 6) = f (2 * π / 3)
axiom symmetry_condition_2 : f (-π / 6) = f (π / 2)

-- Goal
theorem find_f_neg_5pi_12 : f (- 5 * π / 12) = sqrt 3 / 2 := 
sorry

end find_f_neg_5pi_12_l369_369782


namespace find_q_from_min_y_l369_369518

variables (a p q m : ℝ)
variable (a_nonzero : a ≠ 0)
variable (min_y : ∀ x : ℝ, a*x^2 + p*x + q ≥ m)

theorem find_q_from_min_y :
  q = m + p^2 / (4 * a) :=
sorry

end find_q_from_min_y_l369_369518


namespace jonah_price_per_ring_l369_369954

theorem jonah_price_per_ring
  (n_pineapples : ℕ) (cost_per_pineapple : ℝ) (rings_per_pineapple : ℕ)
  (rings_sold : ℕ) (profit : ℝ) 
  (hc1 : n_pineapples = 6)
  (hc2 : cost_per_pineapple = 3)
  (hc3 : rings_per_pineapple = 12)
  (hc4 : rings_sold = 4)
  (hc5 : profit = 72) :
  (let total_cost := n_pineapples * cost_per_pineapple in
   let total_rings := n_pineapples * rings_per_pineapple in
   let total_revenue := total_cost + profit in
   let sets_sold := total_rings / rings_sold in
   let price_per_set := total_revenue / sets_sold in
   let price_per_ring := price_per_set / rings_sold in
   price_per_ring) = 1.25 :=
by {
  sorry
}

end jonah_price_per_ring_l369_369954


namespace find_fx_value_l369_369720

noncomputable def f (x : Real) : Real :=
  sin (2 * x - 5 * Real.pi / 6)

theorem find_fx_value :
  (∀ x, f x = sin (2 * x - 5 * Real.pi / 6)) ∧ 
  (f (Real.pi / 6) = f (2 * Real.pi / 3)) ∧ 
  (∀ x1 x2, (Real.pi / 6 < x1 ∧ x1 < 2 * Real.pi / 3 ∧ x1 < x2 ∧ x2 < 2 * Real.pi / 3) -> f x1 < f x2) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 := by
  sorry

end find_fx_value_l369_369720


namespace transport_connectivity_l369_369925

-- Define the condition that any two cities are connected by either an air route or a canal.
-- We will formalize this with an inductive type to represent the transport means: AirRoute or Canal.
inductive TransportMeans
| AirRoute : TransportMeans
| Canal : TransportMeans

open TransportMeans

-- Represent cities as a type 'City'
universe u
variable (City : Type u)

-- Connect any two cities by a transport means
variable (connected : City → City → TransportMeans)

-- We want to prove that for any set of cities, 
-- there exists a means of transport such that starting from any city,
-- it is possible to reach any other city using only that means of transport.
theorem transport_connectivity (n : ℕ) (h2 : n ≥ 2) : 
  ∃ (T : TransportMeans), ∀ (c1 c2 : City), connected c1 c2 = T :=
by
  sorry

end transport_connectivity_l369_369925


namespace monotonic_increasing_interval_l369_369367

noncomputable def f (x : ℝ) : ℝ := (1 / 2) ^ (Real.sqrt (2 * x - x ^ 2))

theorem monotonic_increasing_interval :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → 1 ≤ x ∧ x < 2 → ∀ x1 x2, x1 < x2 → f x1 ≤ f x2 :=
by
  sorry

end monotonic_increasing_interval_l369_369367


namespace remainder_when_divided_by_32_l369_369155

def product_of_odds : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_divided_by_32 : product_of_odds % 32 = 9 := by
  sorry

end remainder_when_divided_by_32_l369_369155


namespace max_val_y_l369_369970

noncomputable def f (x : ℝ) : ℝ := 3^(x - 1) + x - 1

theorem max_val_y (a b : ℝ) (ha : 0 ≤ a) (hb : b ≤ 1)
    (h : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = a ∨ f x = b) :
  ∃ z, f z + f⁻¹ z = 2 := by
  sorry

end max_val_y_l369_369970


namespace maximum_xy_value_l369_369868

theorem maximum_xy_value :
  ∃ (x y : ℕ), 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by
  sorry

end maximum_xy_value_l369_369868


namespace range_of_a_l369_369075

noncomputable def f (x : ℝ) : ℝ := x + 4 / x
noncomputable def g (x a : ℝ) : ℝ := 2^x + a

theorem range_of_a (a : ℝ) :
  (∀ x1 ∈ set.Icc (1 / 2 : ℝ) 1, ∃ x2 ∈ set.Icc 2 3, f x1 ≥ g x2 a) → a ≤ 1 :=
by
  sorry

end range_of_a_l369_369075


namespace slower_train_speed_l369_369404

-- Definitions of the constants
def length_train1 : ℝ := 140  -- length of the first train in meters
def length_train2 : ℝ := 210  -- length of the second train in meters
def crossing_time : ℝ := 12.59899208063355  -- time taken to cross each other in seconds
def speed_fast_train_kmh : ℝ := 60  -- speed of the faster train in km/hr

-- Convert speed from km/hr to m/s
def speed_fast_train_ms : ℝ := speed_fast_train_kmh * 1000 / 3600

-- Theorem proving the speed of the slower train
theorem slower_train_speed :
  let total_distance := length_train1 + length_train2 in
  let V_rel := total_distance / crossing_time in
  let V_rel_kmh := V_rel * 3600 / 1000 in
  let speed_slow_train_kmh := V_rel_kmh - speed_fast_train_kmh in
  speed_slow_train_kmh = 39.9788 :=
by
  unfold total_distance V_rel V_rel_kmh speed_slow_train_kmh
  sorry

end slower_train_speed_l369_369404


namespace quilt_block_shading_fraction_l369_369517

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

end quilt_block_shading_fraction_l369_369517


namespace marc_total_spent_l369_369982

theorem marc_total_spent
  (model_cars : ℕ) (cost_model_car : ℕ)
  (paint_bottles : ℕ) (cost_paint_bottle : ℕ)
  (paintbrushes : ℕ) (cost_paintbrush : ℕ) :
  model_cars = 5 →
  cost_model_car = 20 →
  paint_bottles = 5 →
  cost_paint_bottle = 10 →
  paintbrushes = 5 →
  cost_paintbrush = 2 →
  let total_cost := (model_cars * cost_model_car) + (paint_bottles * cost_paint_bottle) + (paintbrushes * cost_paintbrush)
  in total_cost = 160 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  dsimp [Nat.mul, Nat.add]
  sorry

end marc_total_spent_l369_369982


namespace max_product_l369_369881

theorem max_product (x y : ℕ) (h1 : 7 * x + 4 * y = 140) : x * y ≤ 168 :=
sorry

end max_product_l369_369881


namespace part1_part2_l369_369830

variable (a : ℝ)
def line (a : ℝ) : ℝ → ℝ → Prop := λ x y, x + a * y + a - 1 = 0

def point_P := (-2, 1 : ℝ)

-- Part 1
theorem part1 (x y : ℝ) : 
  (line a x y) → 
  ∃ d : ℝ, d = Real.sqrt 13 ∧ line (-2 / 3) x y :=
by sorry

-- Part 2
theorem part2 (x y : ℝ) :
a = 2 → 
line a x y → 
∃ (m n : ℝ), 12 * m + n = 0 ∧ m = 0 ∧ n = 0 :=
by sorry

end part1_part2_l369_369830


namespace greatest_xy_l369_369853

theorem greatest_xy (x y : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 7 * x + 4 * y = 140) : xy ≤ 168 :=
begin
  sorry
end

example : ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ 7 * x + 4 * y = 140 ∧ xy = 168 :=
begin
  use [8, 21],
  split, exact dec_trivial,
  split, exact dec_trivial,
  split, exact dec_trivial,
  exact dec_trivial
end

end greatest_xy_l369_369853


namespace concentric_circumcircles_l369_369650

-- Suppress noncomputable warnings
set_option autoImplicit false

variables {α : Type*} [EuclideanSpace α]

-- Given a triangle ABC with incircle touching sides BC, CA, and AB at points D, E, and F
variables (A B C D E F : α)
-- The midpoints of certain segments
variables (K L M N U V : α)

-- Define conditions: 
-- Triangle incircle touch points D, E, F
-- Midpoints definitions
def incircle_tangency_points : Prop :=
  True  -- This should be defined properly involving tangency properties

def midpoints_def : Prop :=
  True  -- This should be defined properly

-- Main goal: Prove the circumcircles are concentric
theorem concentric_circumcircles :
  incircle_tangency_points A B C D E F →
  midpoints_def A B C K L M N U V →
  let triangle1 := triangle A B C in
  let triangle2 := triangle K L N in
  concentric_circles (circumcircle triangle1) (circumcircle triangle2) := 
begin
  intros,
  sorry
end

end concentric_circumcircles_l369_369650


namespace sequence_formula_l369_369079

-- Definitions of the sequence and conditions
def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  Finset.sum (Finset.range n) a

def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S a n + a n = 2 * n + 1

-- Proposition to prove
theorem sequence_formula (a : ℕ → ℝ) (h : satisfies_condition a) : 
  ∀ n : ℕ, a n = 2 - 1 / 2^n := sorry

end sequence_formula_l369_369079


namespace time_after_1457_minutes_l369_369390

noncomputable def hours_and_minutes (total_minutes : ℕ) : ℕ × ℕ :=
  (total_minutes / 60, total_minutes % 60)

def initial_time : ℕ × ℕ := (15, 0) -- 3:00 p.m. in 24-hour clock

theorem time_after_1457_minutes : hours_and_minutes 1457 = (24, 17) → 
                                   initial_time = (15, 0) →
                                   nat.add (initial_time.1) 24 % 24 = 15 ∧ initial_time.2 + 17 < 60 →
                                   initial_time.1 = 15 ∧ initial_time.2 = 0 → 
                                   (initial_time.1, initial_time.2 + 17) = (15, 17) := 
by 
  intros h1 h2 h3 h4
  sorry

end time_after_1457_minutes_l369_369390


namespace remainder_M_mod_32_l369_369259

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_M_mod_32 : M % 32 = 17 :=
by {
  sorry
}

end remainder_M_mod_32_l369_369259


namespace recurrence_relation_p_series_l369_369906

noncomputable def p_series (n k : ℕ) : ℝ :=
if k < n then (p_series (n - 1) k - (1 / (2 : ℝ)^k) * p_series (n - k) k + (1 / (2 : ℝ)^k))
else 0

-- Statement of the theorem
theorem recurrence_relation_p_series (n k : ℕ) (h : k < n) :
  p_series n k = p_series (n - 1) k - (1 / (2 : ℝ)^k) * p_series (n - k) k + (1 / (2 : ℝ)^k) :=
sorry

end recurrence_relation_p_series_l369_369906


namespace more_volunteers_needed_l369_369995

theorem more_volunteers_needed
    (required_volunteers : ℕ)
    (students_per_class : ℕ)
    (num_classes : ℕ)
    (teacher_volunteers : ℕ)
    (total_volunteers : ℕ) :
    required_volunteers = 50 →
    students_per_class = 5 →
    num_classes = 6 →
    teacher_volunteers = 13 →
    total_volunteers = (students_per_class * num_classes) + teacher_volunteers →
    (required_volunteers - total_volunteers) = 7 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end more_volunteers_needed_l369_369995


namespace gcd_18_30_l369_369567

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := 
by
  sorry

end gcd_18_30_l369_369567


namespace lower_tap_used_earlier_l369_369451

-- Define the conditions given in the problem
def capacity : ℕ := 36
def midway_capacity : ℕ := capacity / 2
def lower_tap_rate : ℕ := 4  -- minutes per litre
def upper_tap_rate : ℕ := 6  -- minutes per litre

def lower_tap_draw (minutes : ℕ) : ℕ := minutes / lower_tap_rate  -- litres drawn by lower tap
def beer_left_after_draw (initial_amount litres_drawn : ℕ) : ℕ := initial_amount - litres_drawn

-- Define the assistant's drawing condition
def assistant_draw_min : ℕ := 16
def assistant_draw_litres : ℕ := lower_tap_draw assistant_draw_min

-- Define proof statement
theorem lower_tap_used_earlier :
  let initial_amount := capacity
  let litres_when_midway := midway_capacity
  let litres_beer_left := beer_left_after_draw initial_amount assistant_draw_litres
  let additional_litres := litres_beer_left - litres_when_midway
  let time_earlier := additional_litres * upper_tap_rate
  time_earlier = 84 := 
by
  sorry

end lower_tap_used_earlier_l369_369451


namespace sum_of_inserted_numbers_eq_12_l369_369947

theorem sum_of_inserted_numbers_eq_12 (a b : ℝ) (r d : ℝ) 
  (h1 : a = 2 * r) 
  (h2 : b = 2 * r^2) 
  (h3 : b = a + d) 
  (h4 : 12 = b + d) : 
  a + b = 12 :=
by
  sorry

end sum_of_inserted_numbers_eq_12_l369_369947


namespace no_integer_solutions_l369_369029

theorem no_integer_solutions (x y z : ℤ) : ¬ (4 * x^2 + 77 * y^2 = 487 * z^2) :=
by {
  -- The proof would go here, but for the generated statement, we leave it as a sorry
  sorry
}

end no_integer_solutions_l369_369029


namespace find_f_value_l369_369820

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

theorem find_f_value :
  f ( -5 * Real.pi / 12 ) = Real.sqrt 3 / 2 :=
by
  sorry

end find_f_value_l369_369820


namespace mr_brown_class_problem_l369_369107

theorem mr_brown_class_problem :
  ∀ (total_students boys girls : ℕ), 
  boys = 3 * (total_students / 7) ∧ 
  girls = 4 * (total_students / 7) ∧ 
  total_students = 56 -> 
  (boys : ℚ) / total_students = 42.86 / 100 ∧ 
  girls = 32 :=
by
  sorry

end mr_brown_class_problem_l369_369107


namespace problem_statement_l369_369681

-- Definition of our function f
def f (x : ℝ) : ℝ := Real.sin (2 * x - (5 * Real.pi / 6))

-- Theorem to prove that f(-5π/12) = √3/2
theorem problem_statement : 
  f(-5 * Real.pi / 12) = sqrt 3 / 2 := 
sorry

end problem_statement_l369_369681


namespace find_f_value_l369_369812

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

theorem find_f_value :
  f ( -5 * Real.pi / 12 ) = Real.sqrt 3 / 2 :=
by
  sorry

end find_f_value_l369_369812


namespace remainder_of_M_l369_369266

def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_of_M : M % 32 = 31 := by
  -- Proof goes here
  sorry

end remainder_of_M_l369_369266


namespace identify_clairvoyant_in_29_questions_l369_369396

-- Define the concept of a person in the room
inductive Person : Type
| clairvoyant (knows_all_birthdates : Bool)
| ordinary (knows_clairvoyant_birthdate : Bool)

-- Define the condition that we have exactly 30 people, among whom one is clairvoyant
def Room := List Person

-- The main theorem to prove:
theorem identify_clairvoyant_in_29_questions (room : Room) (h_length : room.length = 30) :
  ∃ clairvoyant : Person, clairvoyant.knows_all_birthdates = true ∧ 
  ∀ p, p ≠ clairvoyant → ¬p.knows_all_birthdates :=
sorry

end identify_clairvoyant_in_29_questions_l369_369396


namespace prime_product_mod_32_l369_369304

open Nat

theorem prime_product_mod_32 : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := List.foldr (· * ·) 1 primes
  M % 32 = 25 :=
by
  sorry

end prime_product_mod_32_l369_369304


namespace student_B_more_stable_than_A_student_B_more_stable_l369_369488

-- Define students A and B.
structure Student :=
  (average_score : ℝ)
  (variance : ℝ)

-- Given data for both students.
def studentA : Student :=
  { average_score := 90, variance := 51 }

def studentB : Student :=
  { average_score := 90, variance := 12 }

-- The theorem that student B has more stable performance than student A.
theorem student_B_more_stable_than_A (A B : Student) (h_avg : A.average_score = B.average_score) :
  A.variance > B.variance → B.variance < A.variance :=
by
  intro h
  linarith

-- Specific instance of the theorem with given data for students A and B.
theorem student_B_more_stable : studentA.variance > studentB.variance → studentB.variance < studentA.variance :=
  student_B_more_stable_than_A studentA studentB rfl

end student_B_more_stable_than_A_student_B_more_stable_l369_369488


namespace probability_ratio_l369_369348

theorem probability_ratio (a b : ℕ) (h1 : a = 60) (h2 : b = 12)
  (h3 : ∀ n, n ∈ (Finset.range b).image ((λ i, i + 1)) → (∃ c, c = 5))
  (h4 : ∀ k, k = 5) :
  let p := 12 / (Nat.choose 60 5 : ℚ),
      r := 13200 / (Nat.choose 60 5 : ℚ) in
  r / p = 1100 := by
  sorry

end probability_ratio_l369_369348


namespace additional_hours_needed_l369_369530

-- Define the conditions
def speed : ℕ := 5  -- kilometers per hour
def total_distance : ℕ := 30 -- kilometers
def hours_walked : ℕ := 3 -- hours

-- Define the statement to prove
theorem additional_hours_needed : total_distance / speed - hours_walked = 3 := 
by
  sorry

end additional_hours_needed_l369_369530


namespace product_mod_32_l369_369208

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  M % 32 = 17 := by
  sorry

end product_mod_32_l369_369208


namespace gcd_of_18_and_30_l369_369548

theorem gcd_of_18_and_30 : Nat.gcd 18 30 = 6 :=
by
  sorry

end gcd_of_18_and_30_l369_369548


namespace find_value_of_f_eq_l369_369757

noncomputable def f (ω ω φ x : ℝ) : ℝ := sin (ω * x + φ)

theorem find_value_of_f_eq :
  ∀ (ω φ : ℝ), 
    (∀ x, x ∈ set.Ioo (π / 6) (2 * π / 3) → (f ω φ x) ≤ (f ω φ (x + 1e-10))) → -- monotonically increasing
    (ω * ∓ (π / 6) + φ) = (φ + ω * (2 * π / 3)) → -- symmetric axes
    f ω φ (-(5 * π / 12)) = sqrt 3 / 2 :=
by
  sorry

end find_value_of_f_eq_l369_369757


namespace max_product_l369_369879

theorem max_product (x y : ℕ) (h1 : 7 * x + 4 * y = 140) : x * y ≤ 168 :=
sorry

end max_product_l369_369879


namespace ants_in_garden_l369_369472

theorem ants_in_garden :
  ∀ (width length : ℕ) (ant_density : ℕ), 
  width = 120 → 
  length = 160 → 
  ant_density = 4 → 
  100 * 100 * (width * length) * ant_density ≈ 800000000 := 
by
  intros width length ant_density h_width h_length h_density 
  sorry

end ants_in_garden_l369_369472


namespace isosceles_triangle_count_l369_369123

noncomputable def triangle_PQR : Type :=
{ P Q R S T U : ℝ // 
  ∠PQR = 60 ∧ 
  PQ = PR ∧ 
  PS bisects ∠PQR ∧ 
  S on_line PR ∧ 
  T on_line QR ∧ 
  ST ∥ PQ ∧ 
  U on_line PR ∧ 
  TU ∥ PS }

theorem isosceles_triangle_count : ∃ n : ℕ, n = 7 :=
by
  -- Here you would proceed with proving the theorem 
  -- based on the given conditions.
  sorry

end isosceles_triangle_count_l369_369123


namespace min_value_l369_369643

-- Define the conditions
variables (x y : ℝ)
-- Assume x and y are in the positive real numbers
axiom pos_x : 0 < x
axiom pos_y : 0 < y
-- Given equation
axiom eq1 : x + 2 * y = 2 * x * y

-- The goal is to prove that the minimum value of 3x + 4y is 5 + 2sqrt(6)
theorem min_value (x y : ℝ) (pos_x : 0 < x) (pos_y : 0 < y) (eq1 : x + 2 * y = 2 * x * y) : 
  3 * x + 4 * y ≥ 5 + 2 * Real.sqrt 6 := 
sorry

end min_value_l369_369643


namespace sin_function_value_l369_369800

noncomputable def f (x : ℝ) : ℝ := sin(2 * x - 5 * π / 6)

theorem sin_function_value :
  f (-5 * π / 12) = sqrt 3 / 2 := by
  sorry

end sin_function_value_l369_369800


namespace gcd_18_30_is_6_l369_369615

def gcd_18_30 : ℕ :=
  gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 :=
by {
  -- The step here will involve using properties of gcd and prime factorization,
  -- but we are given the result directly for the purpose of this task.
  sorry
}

end gcd_18_30_is_6_l369_369615


namespace James_works_6_hours_l369_369953

variable (bedrooms : ℕ) (living_room : ℕ) (bathrooms : ℕ) (time_per_bedroom : ℕ) 
variable (James_siblings : ℕ)

-- Definitions of given conditions
def time_for_cleaning_house : ℕ :=
  let time_for_bedrooms := bedrooms * time_per_bedroom
  let time_for_living_room := time_for_bedrooms
  let time_for_bathroom := 2 * time_for_living_room
  let total_inside_time :=  time_for_bedrooms + time_for_living_room + 2 * time_for_bathroom
  total_inside_time

def total_time_for_chores : ℕ :=
  let total_inside_time := time_for_cleaning_house bedrooms living_room bathrooms time_per_bedroom
  2 * total_inside_time + total_inside_time

def total_minutes_per_sibling : ℕ :=
  (total_time_for_chores bedrooms living_room bathrooms time_per_bedroom) / (James_siblings + 1)

def total_hours_per_sibling : ℕ :=
  total_minutes_per_sibling bedrooms living_room bathrooms time_per_bedroom James_siblings / 60

-- Prove that James works for 6 hours
theorem James_works_6_hours :
  bedrooms = 3 ∧ living_room = 1 ∧ bathrooms = 2 ∧ time_per_bedroom = 20 ∧ James_siblings = 2 →
  total_hours_per_sibling bedrooms living_room bathrooms time_per_bedroom James_siblings = 6 := by
  intros
  exact sorry

end James_works_6_hours_l369_369953


namespace average_bowling_score_l369_369087

theorem average_bowling_score 
    (gretchen_score : ℕ) (mitzi_score : ℕ) (beth_score : ℕ)
    (gretchen_eq : gretchen_score = 120)
    (mitzi_eq : mitzi_score = 113)
    (beth_eq : beth_score = 85) :
    (gretchen_score + mitzi_score + beth_score) / 3 = 106 := 
by
  sorry

end average_bowling_score_l369_369087


namespace find_fx_value_l369_369718

noncomputable def f (x : Real) : Real :=
  sin (2 * x - 5 * Real.pi / 6)

theorem find_fx_value :
  (∀ x, f x = sin (2 * x - 5 * Real.pi / 6)) ∧ 
  (f (Real.pi / 6) = f (2 * Real.pi / 3)) ∧ 
  (∀ x1 x2, (Real.pi / 6 < x1 ∧ x1 < 2 * Real.pi / 3 ∧ x1 < x2 ∧ x2 < 2 * Real.pi / 3) -> f x1 < f x2) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 := by
  sorry

end find_fx_value_l369_369718


namespace gcd_18_30_l369_369542

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l369_369542


namespace statement1_statement2_statement3_l369_369527

variable (P_W P_Z : ℝ)

/-- The conditions of the problem: -/
def conditions : Prop :=
  P_W = 0.4 ∧ P_Z = 0.2

/-- Proof of the first statement -/
theorem statement1 (h : conditions P_W P_Z) : 
  P_W * P_Z = 0.08 := 
by sorry

/-- Proof of the second statement -/
theorem statement2 (h : conditions P_W P_Z) :
  P_W * (1 - P_Z) + (1 - P_W) * P_Z = 0.44 := 
by sorry

/-- Proof of the third statement -/
theorem statement3 (h : conditions P_W P_Z) :
  1 - P_W * P_Z = 0.92 := 
by sorry

end statement1_statement2_statement3_l369_369527


namespace recurrence_relation_l369_369896

variables {n k : ℕ}

def p : ℕ → ℕ → ℚ := sorry

theorem recurrence_relation (n k : ℕ) (hnk : n ≥ k) :
  p n k = p (n-1) k - (1 / (2^k)) * p (n-k) k + (1 / (2^k)) :=
begin
  sorry
end

end recurrence_relation_l369_369896


namespace field_area_l369_369471

theorem field_area (L W : ℝ) (h1: L = 20) (h2 : 2 * W + L = 41) : L * W = 210 :=
by
  sorry

end field_area_l369_369471


namespace find_f_value_l369_369819

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

theorem find_f_value :
  f ( -5 * Real.pi / 12 ) = Real.sqrt 3 / 2 :=
by
  sorry

end find_f_value_l369_369819


namespace marc_total_spent_l369_369985

theorem marc_total_spent :
  let cost_model_cars := 20
  let num_model_cars := 5
  let cost_bottles_paint := 10
  let num_bottles_paint := 5
  let cost_paintbrushes := 2
  let num_paintbrushes := 5
  let total_cost := (cost_model_cars * num_model_cars) + (cost_bottles_paint * num_bottles_paint) + (cost_paintbrushes * num_paintbrushes)
  in total_cost = 160 := 
by
  sorry

end marc_total_spent_l369_369985


namespace remainder_of_M_when_divided_by_32_l369_369195

open Nat

def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
def M : ℕ := List.foldl (*) 1 oddPrimes

theorem remainder_of_M_when_divided_by_32 :
  M % 32 = 1 :=
sorry

end remainder_of_M_when_divided_by_32_l369_369195


namespace regular_polygon_sides_l369_369476

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 1 ≤ i → i ≤ n → interior_angle i = 140) (h2 : sum_of_interior_angles n = 180 * (n - 2)) :
  n = 9 :=
by
  sorry

end regular_polygon_sides_l369_369476


namespace remainder_of_M_when_divided_by_32_l369_369201

open Nat

def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
def M : ℕ := List.foldl (*) 1 oddPrimes

theorem remainder_of_M_when_divided_by_32 :
  M % 32 = 1 :=
sorry

end remainder_of_M_when_divided_by_32_l369_369201


namespace adult_male_more_bones_than_adult_woman_l369_369927

theorem adult_male_more_bones_than_adult_woman :
  ∀ (total_skeletons adult_women adult_men children woman_bones child_bones total_bones : ℕ)
    (H1 : total_skeletons = 20)
    (H2 : adult_women = 10)
    (H3 : adult_men = 5)
    (H4 : children = 5)
    (H5 : woman_bones = 20)
    (H6 : child_bones = 10)
    (H7 : total_bones = 375),
    let men_bones := total_bones - (adult_women * woman_bones + children * child_bones),
    let per_man_bones := men_bones / adult_men in
    per_man_bones - woman_bones = 5 :=
by
  intros
  let men_bones := total_bones - (adult_women * woman_bones + children * child_bones)
  let per_man_bones := men_bones / adult_men
  have Hmen_bones : men_bones = 125 := sorry
  have Hper_man_bones : per_man_bones = 25 := sorry
  rw [Hmen_bones, Hper_man_bones]
  change 25 - woman_bones = 5
  rw [H5]
  exact sorry

end adult_male_more_bones_than_adult_woman_l369_369927


namespace part_I_part_II_l369_369071

def f (x a : ℝ) : ℝ := |2 * x - a| + a
def g (x : ℝ) : ℝ := |2 * x - 1|

theorem part_I (a : ℝ) (h : a = 3) : { x : ℝ | f x a ≤ 6 } = set.Icc 0 3 := 
sorry

theorem part_II (a : ℝ) : (∀ x : ℝ, f x a + g x ≥ 2 * a^2 - 13) → a ∈ set.Icc (-Real.sqrt 7) 3 :=
sorry

end part_I_part_II_l369_369071


namespace distinct_sequences_TRIANGLE_l369_369841

theorem distinct_sequences_TRIANGLE : 
  let word := ['T', 'R', 'I', 'A', 'N', 'G', 'L', 'E'];
  let first_letter := 'G';
  let last_letter := 'E';
  let remaining_letters := ['T', 'R', 'I', 'A', 'N', 'L'];
  (first_letter ∈ word) ∧ (last_letter ∈ word) →
  (∀ s : List Char, s.head = some first_letter ∧ s.getLast! = last_letter ∧
                   (∀ c : Char, c ∈ s → c ∈ word) ∧ s.length = 5 →
                   List.nodup s) →
  List.length {s | s.head = some first_letter ∧ s.getLast! = last_letter ∧ 
                  (∀ c : Char, c ∈ s → c ∈ word) ∧ s.length = 5 ∧
                  List.nodup s} = 120 :=
by
  sorry

end distinct_sequences_TRIANGLE_l369_369841


namespace relationship_between_m_and_n_l369_369968

variable (a b m n : ℝ)

axiom h1 : a > b
axiom h2 : b > 0
axiom h3 : m = Real.sqrt a - Real.sqrt b
axiom h4 : n = Real.sqrt (a - b)

theorem relationship_between_m_and_n : m < n :=
by
  -- Lean requires 'sorry' to be used as a placeholder for the proof
  sorry

end relationship_between_m_and_n_l369_369968


namespace greatest_xy_l369_369854

theorem greatest_xy (x y : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 7 * x + 4 * y = 140) : xy ≤ 168 :=
begin
  sorry
end

example : ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ 7 * x + 4 * y = 140 ∧ xy = 168 :=
begin
  use [8, 21],
  split, exact dec_trivial,
  split, exact dec_trivial,
  split, exact dec_trivial,
  exact dec_trivial
end

end greatest_xy_l369_369854


namespace max_xy_value_l369_369884

theorem max_xy_value (x y : ℕ) (h : 7 * x + 4 * y = 140) : xy ≤ 168 :=
by sorry

end max_xy_value_l369_369884


namespace average_bowling_score_l369_369086

-- Definitions of the scores
def g : ℕ := 120
def m : ℕ := 113
def b : ℕ := 85

-- Theorem statement: The average score is 106
theorem average_bowling_score : (g + m + b) / 3 = 106 := by
  sorry

end average_bowling_score_l369_369086


namespace total_fence_length_proof_l369_369463

noncomputable def total_length_of_fence (l w r o : ℕ) : ℕ := 
  let p_rectangle := 2 * (l + w) - w
  let c_semicircle := (Real.pi * r).toNat + 2 * r
  p_rectangle + c_semicircle - o

theorem total_fence_length_proof : 
  total_length_of_fence 20 14 7 3 = 73 := 
by
  -- the proof goes here
  sorry

end total_fence_length_proof_l369_369463


namespace hound_catch_hare_l369_369114

theorem hound_catch_hare (d h g : ℕ) (H_d : d = 150) (H_h : h = 7) (H_g : g = 9) :
  ∃ n : ℕ, (g - h) * n = d :=
by
  existsi 75
  rw [H_d, H_h, H_g]
  exact congr_arg (λ x, x * 75) rfl

end hound_catch_hare_l369_369114


namespace two_different_orientations_l369_369393

noncomputable def num_points : ℕ := 100

structure Cube :=
(points : fin num_points → (ℝ × ℝ × ℝ))

def cube_orientations : fin 24 := sorry -- Since there are 24 possible orientations of the cube.

def projection (c : Cube) (o : fin 24) : set (ℝ × ℝ) :=
  sorry -- This would be a function giving the set of projections of the points for a given orientation.

theorem two_different_orientations (c : Cube) :
  ∃ (o1 o2 : fin 24), o1 ≠ o2 ∧ projection c o1 ≠ projection c o2 :=
sorry

end two_different_orientations_l369_369393


namespace find_f_of_neg_5_pi_over_12_l369_369740

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - (5 * π / 6))

theorem find_f_of_neg_5_pi_over_12 :
  (∀ x ∈ (set.Ioo (π / 6) (2 * π / 3)), (0 : ℝ) < (f x - f (x - 1))) ∧
  (∀ x, f (π / 6 - x) = f (π / 6 + x)) ∧
  (∀ x, f (2 * π / 3 - x) = f (2 * π / 3 + x)) →
  f (-5 * π / 12) = √3 / 2 :=
by
  sorry

end find_f_of_neg_5_pi_over_12_l369_369740


namespace points_on_same_circle_l369_369493

variables (O A B C D P M K L N : Type)
variables [circle_vertex O A] [circle_vertex O B] [circle_vertex O C] [circle_vertex O D]
noncomputable def trapezoid (A B C D : Type) := sorry  -- placeholder for defining a trapezoid
noncomputable def midpoint (A B M : Type) := sorry  -- placeholder for midpoint definition
noncomputable def perpendicular_bisector_intersection (A D K L : Type) := sorry  -- placeholder for intersection definition
noncomputable def arc_midpoint (C D P N : Type) := sorry  -- placeholder for arc midpoint definition

theorem points_on_same_circle
  (trapezoid_ABCD : trapezoid A B C D)
  (AC_intersects_BD_at_P : intersects (line_segment A C) (line_segment B D) P)
  (M_midpoint_AB : midpoint A B M)
  (perpendicular_bisector_AD_intersects_at_KL : perpendicular_bisector_intersection A D K L)
  (N_midpoint_arc_CD : arc_midpoint C D P N) :
  cyclic_points K L M N :=
sorry

end points_on_same_circle_l369_369493


namespace marc_total_spent_l369_369983

theorem marc_total_spent
  (model_cars : ℕ) (cost_model_car : ℕ)
  (paint_bottles : ℕ) (cost_paint_bottle : ℕ)
  (paintbrushes : ℕ) (cost_paintbrush : ℕ) :
  model_cars = 5 →
  cost_model_car = 20 →
  paint_bottles = 5 →
  cost_paint_bottle = 10 →
  paintbrushes = 5 →
  cost_paintbrush = 2 →
  let total_cost := (model_cars * cost_model_car) + (paint_bottles * cost_paint_bottle) + (paintbrushes * cost_paintbrush)
  in total_cost = 160 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  dsimp [Nat.mul, Nat.add]
  sorry

end marc_total_spent_l369_369983


namespace remainder_when_divided_by_32_l369_369159

def product_of_odds : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_divided_by_32 : product_of_odds % 32 = 9 := by
  sorry

end remainder_when_divided_by_32_l369_369159


namespace remainder_when_M_divided_by_32_l369_369143

-- Define M as the product of all odd primes less than 32.
def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_M_divided_by_32 :
    M % 32 = 1 := by
  -- We must prove that M % 32 = 1.
  sorry

end remainder_when_M_divided_by_32_l369_369143


namespace sum_of_h_values_l369_369917

variable (f h : ℤ → ℤ)

-- Function definition for f and h
def f_def : ∀ x, 0 ≤ x → f x = f (x + 2) := sorry
def h_def : ∀ x, x < 0 → h x = f x := sorry

-- Symmetry condition for f being odd
def f_odd : ∀ x, f (-x) = -f x := sorry

-- Given value
def f_at_5 : f 5 = 1 := sorry

-- The proof statement we need:
theorem sum_of_h_values :
  h (-2022) + h (-2023) + h (-2024) = -1 :=
sorry

end sum_of_h_values_l369_369917


namespace solve_for_f_1988_l369_369308

noncomputable def f : ℕ+ → ℕ+ := sorry

axiom functional_eq (m n : ℕ+) : f (f m + f n) = m + n

theorem solve_for_f_1988 : f 1988 = 1988 :=
sorry

end solve_for_f_1988_l369_369308


namespace product_of_odd_primes_mod_32_l369_369234

theorem product_of_odd_primes_mod_32 :
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31) in
  M % 32 = 25 :=
by
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31)
  sorry

end product_of_odd_primes_mod_32_l369_369234


namespace sum_of_primitive_roots_l369_369962

-- Given p is an odd prime
def is_odd_prime (p : ℕ) : Prop := Nat.Prime p ∧ p % 2 = 1

-- Define Phi function
def phi (n : ℕ) : ℕ := (n.factorization.keys.map nat.totient).prod

-- Define the mu function (Möbius function)
def mu (n : ℕ) : ℤ := if n = 1 then 1 else if ∃ p ∈ n.factors, 2 ∣ n.factors.count p then 0 else if (n.factors.length % 2 = 0) then 1 else -1

-- Define primitive roots
def is_primitive_root (g p : ℕ) : Prop := Nat.gcd g p = 1 ∧ ∀ d : ℕ, d < p - 1 → Nat.gcd d (p - 1) = 1 → g ^ d ≠ 1 %[ p ]

-- The main theorem
theorem sum_of_primitive_roots (p : ℕ) (h_prime : is_odd_prime p) (g : ℕ → ℕ) (hg : ∀ i, is_primitive_root (g i) p) (hg_range : ∀ i, 1 < g i ∧ g i ≤ p - 1) :
  ∑ i in Finset.range (phi (p - 1)), g i % p = mu (p - 1) % p :=
by
  sorry

end sum_of_primitive_roots_l369_369962


namespace find_PS_length_l369_369115

theorem find_PS_length 
  (PT TR QS QP PQ : ℝ)
  (h1 : PT = 5)
  (h2 : TR = 10)
  (h3 : QS = 16)
  (h4 : QP = 13)
  (h5 : PQ = 7) : 
  PS = Real.sqrt 703 := 
sorry

end find_PS_length_l369_369115


namespace find_fx_value_l369_369723

noncomputable def f (x : Real) : Real :=
  sin (2 * x - 5 * Real.pi / 6)

theorem find_fx_value :
  (∀ x, f x = sin (2 * x - 5 * Real.pi / 6)) ∧ 
  (f (Real.pi / 6) = f (2 * Real.pi / 3)) ∧ 
  (∀ x1 x2, (Real.pi / 6 < x1 ∧ x1 < 2 * Real.pi / 3 ∧ x1 < x2 ∧ x2 < 2 * Real.pi / 3) -> f x1 < f x2) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 := by
  sorry

end find_fx_value_l369_369723


namespace gcd_18_30_l369_369539

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l369_369539


namespace find_a_dividing_area_l369_369978

noncomputable def curve_y (a : ℝ) (x : ℝ) : ℝ := a * x * (x - 2)

def region_D (x y : ℝ) : Prop := (x - 1)^2 + y^2 ≤ 1 ∧ y ≥ 0

theorem find_a_dividing_area :
  (∃ a : ℝ, a < 0 ∧ (∫ x in 0..2, curve_y a x dx = ∫ x in 0..2, (sqrt (1 - (x - 1)^2)) dx / 2)) →
  (∀ a : ℝ, a < 0 → (∫ x in 0..2, curve_y a x dx = frac (3 * pi) 16) → a = - (3 * pi) / 16) :=
by
  -- Proof sketch is omitted
  sorry

end find_a_dividing_area_l369_369978


namespace problem_statement_l369_369680

-- Definition of our function f
def f (x : ℝ) : ℝ := Real.sin (2 * x - (5 * Real.pi / 6))

-- Theorem to prove that f(-5π/12) = √3/2
theorem problem_statement : 
  f(-5 * Real.pi / 12) = sqrt 3 / 2 := 
sorry

end problem_statement_l369_369680


namespace remainder_of_M_l369_369276

def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_of_M : M % 32 = 31 := by
  -- Proof goes here
  sorry

end remainder_of_M_l369_369276


namespace gcd_18_30_is_6_l369_369619

def gcd_18_30 : ℕ :=
  gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 :=
by {
  -- The step here will involve using properties of gcd and prime factorization,
  -- but we are given the result directly for the purpose of this task.
  sorry
}

end gcd_18_30_is_6_l369_369619


namespace problem_statement_l369_369746

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - 5 * Real.pi / 6)

theorem problem_statement :
  (∀ (x : ℝ), (∀ a b : ℝ, a < b → x ∈ (Set.Ioc (Real.pi / 6) (2 * Real.pi / 3)) → f(x) < f(b)) → x = (Real.pi / 6) ∨ x = (2 * Real.pi / 3)) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by sorry

end problem_statement_l369_369746


namespace find_f_l369_369312

-- Define function and its properties
def f (x : ℝ) : ℝ := x^2 + 2 * x * f' 1

-- Define first derivative
def f' (x : ℝ) : ℝ := 2 * x + 2 * f' 1

-- Define the second derivative in terms of the first derivative
def f'' (x : ℝ) : ℝ := 2

theorem find_f''_at_0 (f'1 : ℝ) :
  (2 * f'1 = -4) → f'' (0) = -4 :=
by
  -- Provide necessary steps here
  have h : f' 1 = -2 := by sorry
  rw [h]
  sorry

end find_f_l369_369312


namespace product_of_odd_primes_mod_32_l369_369167

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def M : ℕ := odd_primes_less_than_32.foldr (· * ·) 1

theorem product_of_odd_primes_mod_32 : M % 32 = 17 :=
by
  -- the proof goes here
  sorry

end product_of_odd_primes_mod_32_l369_369167


namespace marc_total_spent_l369_369987

theorem marc_total_spent :
  let cost_model_cars := 20
  let num_model_cars := 5
  let cost_bottles_paint := 10
  let num_bottles_paint := 5
  let cost_paintbrushes := 2
  let num_paintbrushes := 5
  let total_cost := (cost_model_cars * num_model_cars) + (cost_bottles_paint * num_bottles_paint) + (cost_paintbrushes * num_paintbrushes)
  in total_cost = 160 := 
by
  sorry

end marc_total_spent_l369_369987


namespace equation_has_exactly_one_real_solution_l369_369042

-- Definitions for the problem setup
def equation (k : ℝ) (x : ℝ) : Prop := (3 * x + 8) * (x - 6) = -54 + k * x

-- The property that we need to prove
theorem equation_has_exactly_one_real_solution (k : ℝ) :
  (∀ x : ℝ, equation k x → ∃! x : ℝ, equation k x) ↔ k = 6 * Real.sqrt 2 - 10 ∨ k = -6 * Real.sqrt 2 - 10 := 
sorry

end equation_has_exactly_one_real_solution_l369_369042


namespace courtyard_width_l369_369458

theorem courtyard_width (length_m : ℕ) (brick_length_cm brick_width_cm : ℕ) (num_bricks : ℕ) (expected_width_m : ℕ) :
  length_m = 30 →
  brick_length_cm = 20 →
  brick_width_cm = 10 →
  num_bricks = 24000 →
  expected_width_m = 16 →
  (length_m * 100 * (expected_width_m * 100) = num_bricks * (brick_length_cm * brick_width_cm)) :=
by
  intros h_length h_brick_length h_brick_width h_num_bricks h_width
  rw [h_length, h_brick_length, h_brick_width, h_num_bricks, h_width]
  norm_num
  sorry

end courtyard_width_l369_369458


namespace problem_2021_expression_l369_369022

noncomputable def prime_factors (n : ℕ) : List ℕ :=
sorry -- Function to compute the prime factors of a number (definition skipped)

theorem problem_2021_expression :
  ∀ (a b : ℕ), 2021 = (a! / b!) →
  (a = 47 ∧ b = 43) →
  | a - b | = 4 :=
by
  intros a b h_eq h_cond
  have ha : a = 47 := h_cond.1
  have hb : b = 43 := h_cond.2
  rw [ha, hb]
  norm_num
  sorry

end problem_2021_expression_l369_369022


namespace second_factor_of_lcm_l369_369365

def hcf (a b : ℕ) : ℕ := Nat.gcd a b
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem second_factor_of_lcm (A B : ℕ) (hcf_A_B : hcf A B = 23) (larger_number : A = 299) (factor1 : 12 ∣ lcm A B) : 
  ∃ X, lcm A B = 23 * 12 * X ∧ X = 13 :=
by
  sorry

end second_factor_of_lcm_l369_369365


namespace petya_result_l369_369330

theorem petya_result (a b c : ℕ) (ha : a = 3) (hb : b = 4) (hc : c = 5) 
  (right_triangle : a^2 + b^2 = c^2) : 
  let x := 1 in
  let y := (a * b) / (a + b) in
  (x / y).num + (x / y).denom = 19 := 
by
  have h1 : right_triangle := by norm_num,
  have h2 : a = 3 := ha,
  have h3 : b = 4 := hb,
  have h4 : c = 5 := hc,
  let x := 1,
  let y := (a * b) / (a + b),
  have h_ratio := (x / y),
  have h_sum := h_ratio.num + h_ratio.denom,
  show h_sum = 19,
  sorry

end petya_result_l369_369330


namespace math_proof_problem_l369_369829

-- Definition of a and b from the inequality having the solution set [-6, 2]
def values_of_a_and_b (a b : ℝ) : Prop :=
  ∀ x, (∀ (x : ℝ), (abs (x + a) ≤ b) ↔ (x = -6 ∨ x = 2))

-- Main theorem: Perform the tasks (1) and (2) together
theorem math_proof_problem (a b m n : ℝ) (h1 : values_of_a_and_b a b) (h2 : abs (2*m + n) < 1/3) (h3 : abs (m - 4*n) < 1/6) :
  a = 2 ∧ b = 4 ∧ abs n < 2/27 :=
by {
  sorry -- Proof is omitted as per the problem statement
}

end math_proof_problem_l369_369829


namespace remainder_M_mod_32_l369_369258

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_M_mod_32 : M % 32 = 17 :=
by {
  sorry
}

end remainder_M_mod_32_l369_369258


namespace find_a2_general_term_formula_harmonic_sum_inequality_l369_369672

-- Definitions of the sequence and the sum
def seq (n : ℕ) : ℕ := n ^ 2
def sum_seq (n : ℕ) : ℕ := (∑ i in Finset.range n, seq (i + 1))

-- Given conditions
axiom a1 : seq 1 = 1
axiom a2 : ∀ n : ℕ, n > 0 → 2 * (sum_seq n) / n = seq (n + 1) - (1 / 3) * n^2 - n - (2 / 3)

-- Theorems to prove based on given conditions
theorem find_a2 : seq 2 = 4 := sorry

theorem general_term_formula : ∀ n : ℕ, n ≥ 1 → seq n = n^2 := sorry

theorem harmonic_sum_inequality : ∀ n : ℕ, n > 0 → (∑ i in Finset.range n, 1 / seq (i + 1)) < 7/4 := sorry

end find_a2_general_term_formula_harmonic_sum_inequality_l369_369672


namespace recurrence_relation_l369_369901

noncomputable def p (n k : ℕ) : ℚ := sorry

theorem recurrence_relation (n k : ℕ) (h : k < n) :
  p n k = p (n - 1) k - (1 / 2^k) * p (n - k) k + (1 / 2^k) :=
by sorry

end recurrence_relation_l369_369901


namespace expression_increase_fraction_l369_369937

theorem expression_increase_fraction (x y : ℝ) :
  let x' := 1.4 * x
  let y' := 1.4 * y
  let original := x * y^2
  let increased := x' * y'^2
  increased - original = (1744/1000) * original := by
sorry

end expression_increase_fraction_l369_369937


namespace line_through_point_5_4_l369_369935

def is_positive_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k ∧ n > 0

def prime_less_than_10 (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

theorem line_through_point_5_4 : 
  let P : ℕ := 10
  let primes := {2, 3, 5, 7}
  let even_integers := {n | is_positive_even n}
  let point := (5, 4)
  let lines := {line | ∃ a b, line = ((1 / a) * x + (1 / b) * y = 1) ∧ a ∈ primes ∧ b ∈ even_integers}
  ∀ line ∈ lines, line passes_through point →
    count lines = 1 :=
sorry

end line_through_point_5_4_l369_369935


namespace problem_statement_l369_369685

-- Definition of our function f
def f (x : ℝ) : ℝ := Real.sin (2 * x - (5 * Real.pi / 6))

-- Theorem to prove that f(-5π/12) = √3/2
theorem problem_statement : 
  f(-5 * Real.pi / 12) = sqrt 3 / 2 := 
sorry

end problem_statement_l369_369685


namespace find_f_neg_5pi_12_l369_369770

-- Conditions
def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

variables {ω φ : ℝ}
def f (x : ℝ) : ℝ := Real.sin (ω * x + φ)

axiom monotonicity_condition : is_monotonically_increasing f (π / 6) (2 * π / 3)
axiom symmetry_condition_1 : f (π / 6) = f (2 * π / 3)
axiom symmetry_condition_2 : f (-π / 6) = f (π / 2)

-- Goal
theorem find_f_neg_5pi_12 : f (- 5 * π / 12) = sqrt 3 / 2 := 
sorry

end find_f_neg_5pi_12_l369_369770


namespace product_mod_32_is_15_l369_369238

-- Define the odd primes less than 32
def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the product of odd primes
def M : ℕ := List.foldl (· * ·) 1 oddPrimes

-- The theorem to prove the remainder when M is divided by 32 is 15
theorem product_mod_32_is_15 : M % 32 = 15 := by
  sorry

end product_mod_32_is_15_l369_369238


namespace zero_point_range_of_f_x_l369_369918

theorem zero_point_range_of_f_x (a : ℝ) :
  (∃ x ∈ Ioo (-1) 1, x + 2 ^ x + a = 0) ↔ a ∈ Ioo (-3 : ℝ) (1 / 2) :=
sorry

end zero_point_range_of_f_x_l369_369918


namespace range_S₁₂_div_d_l369_369652

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence_sum (a₁ d : α) (n : ℕ) : α :=
  (n * (2 * a₁ + (n - 1) * d)) / 2

theorem range_S₁₂_div_d (a₁ d : α) (h_a₁_pos : a₁ > 0) (h_d_neg : d < 0) 
  (h_max_S_8 : ∀ n, arithmetic_sequence_sum a₁ d n ≤ arithmetic_sequence_sum a₁ d 8) :
  -30 < (arithmetic_sequence_sum a₁ d 12) / d ∧ (arithmetic_sequence_sum a₁ d 12) / d < -18 :=
by
  have h1 : -8 < a₁ / d := by sorry
  have h2 : a₁ / d < -7 := by sorry
  have h3 : (arithmetic_sequence_sum a₁ d 12) / d = 12 * (a₁ / d) + 66 := by sorry
  sorry

end range_S₁₂_div_d_l369_369652


namespace max_xy_l369_369861

theorem max_xy (x y : ℕ) (h1: 7 * x + 4 * y = 140) : ∃ x y, 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by {
  sorry
}

end max_xy_l369_369861


namespace maximize_M_l369_369353

theorem maximize_M (n : ℕ) (a : Fin n → ℝ) (h : ∀ i j : Fin n, i < j → a i < a j)
  (h_pos : ∀ i : Fin n, 0 < a i) (b : Fin n → Fin n) (h_perm : ∀ i : Fin n, b i = i) :
  (∏ i, a i + 1 / a (b i)) = (∏ i, a i + 1 / a i) :=
by sorry

end maximize_M_l369_369353


namespace angle_relation_l369_369120

-- Definitions for the triangle properties and angles.
variables {α : Type*} [LinearOrderedField α]
variables {A B C D E F : α}

-- Definitions stating the properties of the triangles.
def is_isosceles_triangle (a b c : α) : Prop :=
  a = b ∨ b = c ∨ c = a

def triangle_ABC_is_isosceles (AB AC : α) (ABC : α) : Prop :=
  is_isosceles_triangle AB AC ABC

def triangle_DEF_is_isosceles (DE DF : α) (DEF : α) : Prop :=
  is_isosceles_triangle DE DF DEF

-- Condition that gives the specific angle measure in triangle DEF.
def angle_DEF_is_100 (DEF : α) : Prop :=
  DEF = 100

-- The main theorem to prove.
theorem angle_relation (AB AC DE DF DEF a b c : α) :
  triangle_ABC_is_isosceles AB AC (AB + AC) →
  triangle_DEF_is_isosceles DE DF DEF →
  angle_DEF_is_100 DEF →
  a = c :=
by
  -- Assuming the conditions define the angles and state the relationship.
  sorry

end angle_relation_l369_369120


namespace intersection_points_and_sum_l369_369002

def f (x : ℝ) : ℝ := x^2 - 4 * x + 3
def g (x : ℝ) : ℝ := -f x
def h (x : ℝ) : ℝ := f (-x)

theorem intersection_points_and_sum : 
  (∃ a b : ℕ, 
    (∀ x : ℝ, f x = g x → (x = 1 ∨ x = 3)) →
    (∀ x : ℝ, f x = h x → x = 0) → 
    a = 2 ∧ b = 0 ∧ 10 * a + b = 20) := 
begin
  sorry
end

end intersection_points_and_sum_l369_369002


namespace part_i_l369_369448

theorem part_i (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : (a^2 + b^2 + c^2)^2 > 2 * (a^4 + b^4 + c^4)) : 
  a + b > c ∧ a + c > b ∧ b + c > a := sorry

end part_i_l369_369448


namespace remainder_M_mod_32_l369_369252

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_M_mod_32 : M % 32 = 17 :=
by {
  sorry
}

end remainder_M_mod_32_l369_369252


namespace polygon_sides_l369_369383

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 720) : n = 6 :=
by sorry

end polygon_sides_l369_369383


namespace factorial_division_l369_369356

theorem factorial_division :
  (2012! / 2011!) = 2012 := by 
sorry

end factorial_division_l369_369356


namespace gcd_18_30_l369_369534

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l369_369534


namespace janitor_hourly_rate_l369_369468

-- Definitions based on conditions
def janitor_cleaning_time := 8   -- time in hours
def student_cleaning_time := 20  -- time in hours
def student_hourly_rate := 7     -- dollars per hour
def additional_cost := 8         -- additional cost in dollars

-- Statement of the proof problem
theorem janitor_hourly_rate:
  ∃ J : ℝ, 
    let janitor_rate := 1 / janitor_cleaning_time,
        student_rate := 1 / student_cleaning_time,
        combined_rate := janitor_rate + student_rate,
        combined_time := 1 / combined_rate,
        combined_cost := J * combined_time + student_hourly_rate * combined_time in
    janitor_cleaning_time * J = combined_cost + additional_cost ∧ 
    J = 21 := 
by
  sorry

end janitor_hourly_rate_l369_369468


namespace arctan_gt_arccos_l369_369027

theorem arctan_gt_arccos (x : ℝ) (h1 : x ≥ -1) (h2 : x < 0) : arctan x > arccos x :=
sorry

end arctan_gt_arccos_l369_369027


namespace slope_of_l_for_circle_intersection_polar_equation_circle_l369_369019

theorem slope_of_l_for_circle_intersection
  (x y t α : ℝ)
  (hC : (x + 6)^2 + y^2 = 25)
  (h_intersect_AB : ∃ (A B : ℝ), |A - B| = sqrt 10 ∧ (A ≠ B))
  (param_eq_l : ∀ t, x = t * Real.cos α ∧ y = t * Real.sin α) :
  k = ± (Real.sqrt 15 / 3) := 
sorry

theorem polar_equation_circle
  (x y ρ α : ℝ)
  (hC : (x + 6)^2 + y^2 = 25)
  (x_polar : x = ρ * cos α)
  (y_polar : y = ρ * sin α) :
  ρ^2 + 12 * ρ * cos α + 11 = 0 :=
sorry

end slope_of_l_for_circle_intersection_polar_equation_circle_l369_369019


namespace problem_statement_l369_369679

-- Definition of our function f
def f (x : ℝ) : ℝ := Real.sin (2 * x - (5 * Real.pi / 6))

-- Theorem to prove that f(-5π/12) = √3/2
theorem problem_statement : 
  f(-5 * Real.pi / 12) = sqrt 3 / 2 := 
sorry

end problem_statement_l369_369679


namespace integer_between_sqrt2_and_sqrt17_l369_369423

theorem integer_between_sqrt2_and_sqrt17 : ∃ (x : ℤ), (real.sqrt 2 < x) ∧ (x < real.sqrt 17) ∧ (x = 3) :=
by
  sorry

end integer_between_sqrt2_and_sqrt17_l369_369423


namespace gcd_of_18_and_30_l369_369545

theorem gcd_of_18_and_30 : Nat.gcd 18 30 = 6 :=
by
  sorry

end gcd_of_18_and_30_l369_369545


namespace real_part_sum_l369_369507

theorem real_part_sum :
  (1 / 2^2010 * ∑ n in Finset.range 1006, (-3)^n * Nat.choose 2010 (2 * n)) = -1 / 2 := 
by
  sorry

end real_part_sum_l369_369507


namespace gcd_of_18_and_30_l369_369593

-- Define the numbers
def num1 := 18
def num2 := 30

-- State the GCD property
theorem gcd_of_18_and_30 : Nat.gcd num1 num2 = 6 :=
by
  sorry

end gcd_of_18_and_30_l369_369593


namespace remainder_when_M_divided_by_32_l369_369145

-- Define M as the product of all odd primes less than 32.
def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_M_divided_by_32 :
    M % 32 = 1 := by
  -- We must prove that M % 32 = 1.
  sorry

end remainder_when_M_divided_by_32_l369_369145


namespace dihedral_angle_equilateral_triangle_l369_369932

-- A definition of equilateral triangle and properties
structure EquilateralTriangle (A B C : Type) :=
  (side_length : ℝ)
  (eq_length : ∀ {x y z : Type}, x ≠ y → x ≠ z → y ≠ z → x = y ∧ y = z ∧ z = x)
  (perp : Type)

-- Definitions of points (Type here can be replaced by Point or other appropriate type in actual implementation)
def Point := Type

-- Main proof statement
theorem dihedral_angle_equilateral_triangle (A B C D : Point) (a : ℝ)
  (h1 : EquilateralTriangle A B C)
  (h2 : h1.side_length = a)
  (h3 : h1.perp A D)
  (h4 : ∃ (D : Point), D = midpoint B C) :
  size_of_dihedral_angle (B - A - D - C) = 60 :=
sorry

end dihedral_angle_equilateral_triangle_l369_369932


namespace calc_value_l369_369510

theorem calc_value : (3000 * (3000 ^ 2999) * 2 = 2 * 3000 ^ 3000) := 
by
  sorry

end calc_value_l369_369510


namespace remainder_of_product_of_odd_primes_mod_32_l369_369288

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem remainder_of_product_of_odd_primes_mod_32 :
  (product odd_primes_less_than_32) % 32 = 23 :=
by sorry

end remainder_of_product_of_odd_primes_mod_32_l369_369288


namespace pizza_without_toppings_cost_l369_369126

theorem pizza_without_toppings_cost 
  (num_slices : ℕ) (price_per_slice : ℝ) 
  (first_topping_cost : ℝ) (next_two_toppings_cost : ℝ) 
  (rest_toppings_cost : ℝ) (toppings_count : ℕ) 
  (toppings_ordered : list string) : 
  num_slices = 8 →
  price_per_slice = 2 →
  first_topping_cost = 2 →
  next_two_toppings_cost = 1 →
  rest_toppings_cost = 0.5 →
  toppings_count = 7 →
  toppings_ordered = ["pepperoni", "sausage", "ham", "olives", "mushrooms", "bell peppers", "pineapple"] →
  (num_slices * price_per_slice) - (first_topping_cost + next_two_toppings_cost * 2 + rest_toppings_cost * 4) = 10 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end pizza_without_toppings_cost_l369_369126


namespace euler_line_HM_MO_2_concurrent_lines_through_midpoints_nagel_incenter_collinear_l369_369435

open EuclideanGeometry

-- Define the Euler line problem in Lean
theorem euler_line_HM_MO_2
  (ABC : Triangle)
  (M : Point) (H : Point) (O : Point)
  (centroid_ABC : is_centroid ABC M)
  (orthocenter_ABC : is_orthocenter ABC H)
  (circumcenter_ABC : is_circumcenter ABC O) :
  same_line [H, M, O] ∧ (distance H M) / (distance M O) = 2 := 
sorry

-- Define the concurrent lines through midpoints problem in Lean
theorem concurrent_lines_through_midpoints
  (ABC : Triangle)
  (D E F : Point)
  (is_midpoint_D : is_midpoint ((B ABC), (C ABC)) D)
  (is_midpoint_E : is_midpoint ((A ABC), (C ABC)) E)
  (is_midpoint_F : is_midpoint ((A ABC), (B ABC)) F)
  (parallel_to_angle_bisectors : ∀ P ∈ {D, E, F}, ∃ angle_bisector, parallel P angle_bisector) :
  ∃ P, ∀ Q R ∈ {D, E, F}, intersects_at_single_point P [draw_parallel P Q] :=
sorry

-- Define the Nagel point and incenter collinearity problem in Lean
theorem nagel_incenter_collinear
  (ABC : Triangle)
  (M : Point)
  (Z : Point)
  (J : Point)
  (centroid_ABC : is_centroid ABC M)
  (incenter_ABC : is_incenter ABC Z)
  (nagel_point_ABC : is_nagel_point ABC J) :
  same_line [J, M, Z] ∧ (distance J M) / (distance M Z) = 2 := 
sorry

end euler_line_HM_MO_2_concurrent_lines_through_midpoints_nagel_incenter_collinear_l369_369435


namespace carl_first_to_roll_six_l369_369499

-- Definitions based on problem conditions
def prob_six := 1 / 6
def prob_not_six := 5 / 6

-- Define geometric series sum formula for the given context
theorem carl_first_to_roll_six :
  ∑' n : ℕ, (prob_not_six^(3*n+1) * prob_six) = 25 / 91 :=
by
  sorry

end carl_first_to_roll_six_l369_369499


namespace min_k_shirts_consecutive_l369_369111

theorem min_k_shirts_consecutive (w p : ℕ) (h1 : w = 21) (h2 : p = 21) :
  ∃ k : ℕ, ∀ (order : list ℕ), 
    (∀ w, ∃ l r : list ℕ, (l ++ [w] ++ r = order) → 
      (l.count 1 ≥ k ∧ r.count 1 ≥ k → list.consecutive 1 (l ++ r)) ∧
      (l.count 2 ≥ k ∧ r.count 2 ≥ k → list.consecutive 2 (l ++ r))) ∧ 
    k = 10 :=
by
  sorry

end min_k_shirts_consecutive_l369_369111


namespace max_xy_l369_369863

theorem max_xy (x y : ℕ) (h1: 7 * x + 4 * y = 140) : ∃ x y, 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by {
  sorry
}

end max_xy_l369_369863


namespace maximum_xy_value_l369_369869

theorem maximum_xy_value :
  ∃ (x y : ℕ), 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by
  sorry

end maximum_xy_value_l369_369869


namespace greatest_xy_l369_369858

theorem greatest_xy (x y : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 7 * x + 4 * y = 140) : xy ≤ 168 :=
begin
  sorry
end

example : ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ 7 * x + 4 * y = 140 ∧ xy = 168 :=
begin
  use [8, 21],
  split, exact dec_trivial,
  split, exact dec_trivial,
  split, exact dec_trivial,
  exact dec_trivial
end

end greatest_xy_l369_369858


namespace blue_pill_cost_correct_l369_369323

-- Defining the conditions
def num_days : Nat := 21
def total_cost : Nat := 672
def red_pill_cost (blue_pill_cost : Nat) : Nat := blue_pill_cost - 2
def daily_cost (blue_pill_cost : Nat) : Nat := blue_pill_cost + red_pill_cost blue_pill_cost

-- The statement to prove
theorem blue_pill_cost_correct : ∃ (y : Nat), daily_cost y * num_days = total_cost ∧ y = 17 :=
by
  sorry

end blue_pill_cost_correct_l369_369323


namespace find_fx_value_l369_369722

noncomputable def f (x : Real) : Real :=
  sin (2 * x - 5 * Real.pi / 6)

theorem find_fx_value :
  (∀ x, f x = sin (2 * x - 5 * Real.pi / 6)) ∧ 
  (f (Real.pi / 6) = f (2 * Real.pi / 3)) ∧ 
  (∀ x1 x2, (Real.pi / 6 < x1 ∧ x1 < 2 * Real.pi / 3 ∧ x1 < x2 ∧ x2 < 2 * Real.pi / 3) -> f x1 < f x2) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 := by
  sorry

end find_fx_value_l369_369722


namespace download_time_l369_369407

theorem download_time (avg_speed : ℤ) (size_A size_B size_C : ℤ) (gb_to_mb : ℤ) (secs_in_min : ℤ) :
  avg_speed = 30 →
  size_A = 450 →
  size_B = 240 →
  size_C = 120 →
  gb_to_mb = 1000 →
  secs_in_min = 60 →
  ( (size_A * gb_to_mb + size_B * gb_to_mb + size_C * gb_to_mb ) / avg_speed ) / secs_in_min = 450 := by
  intros h_avg h_A h_B h_C h_gb h_secs
  sorry

end download_time_l369_369407


namespace remainder_M_mod_32_l369_369261

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_M_mod_32 : M % 32 = 17 :=
by {
  sorry
}

end remainder_M_mod_32_l369_369261


namespace find_value_l369_369715

-- Define the function $f(x)$
def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

-- Define the problem statement in Lean 4
theorem find_value :
  (∀ x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), 0 ≤ 2 * x - 5 * Real.pi / 6 ∧ 2 * x - 5 * Real.pi / 6 ≤ Real.pi) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by
  -- A proper rigorous Lean proof should go here
  sorry

end find_value_l369_369715


namespace product_of_odd_primes_mod_32_l369_369231

theorem product_of_odd_primes_mod_32 :
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31) in
  M % 32 = 25 :=
by
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31)
  sorry

end product_of_odd_primes_mod_32_l369_369231


namespace reema_interest_payment_l369_369339

-- Definitions from conditions
def P : ℝ := 800
def R : ℝ := 8.888194417315589
def T : ℝ := R -- Time (T) is same as Rate of Interest (R)

-- Simple interest calculation
def SI : ℝ := (P * R * T) / 100

-- Problem statement: Prove that the interest paid is Rs 625.
theorem reema_interest_payment : SI = 625 := by
  sorry

end reema_interest_payment_l369_369339


namespace remainder_when_divided_by_32_l369_369162

def product_of_odds : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_divided_by_32 : product_of_odds % 32 = 9 := by
  sorry

end remainder_when_divided_by_32_l369_369162


namespace value_of_k_l369_369103

theorem value_of_k (x y k : ℤ) (h1 : 2 * x - y = 5 * k + 6) (h2 : 4 * x + 7 * y = k) (h3 : x + y = 2024)
: k = 2023 :=
by
  sorry

end value_of_k_l369_369103


namespace product_divisible_by_12_l369_369057

theorem product_divisible_by_12 (a b c d : ℤ) :
  12 ∣ (b - a) * (c - a) * (d - a) * (d - c) * (d - b) * (c - b) := 
by {
  sorry
}

end product_divisible_by_12_l369_369057


namespace sin_function_value_l369_369797

noncomputable def f (x : ℝ) : ℝ := sin(2 * x - 5 * π / 6)

theorem sin_function_value :
  f (-5 * π / 12) = sqrt 3 / 2 := by
  sorry

end sin_function_value_l369_369797


namespace number_of_good_subsets_l369_369832

-- Define the set S
def S := { i : ℕ | 1 ≤ i ∧ i ≤ 1990 }

-- Define what it means to be a "good subset"
def is_good_subset (A : Set ℕ) : Prop :=
  (A ⊆ S) ∧ (A.card = 31) ∧ (A.sum % 5 = 0)

-- The theorem to prove
theorem number_of_good_subsets :
  {A : Set ℕ | is_good_subset A}.card = (1 / 5 : ℚ) * Nat.choose 1990 31 := sorry

end number_of_good_subsets_l369_369832


namespace f_increasing_solve_inequality_range_of_m_l369_369837

variables {f : ℝ → ℝ} {a b x m : ℝ}

-- We need to state conditions in Lean
-- Condition: \( f(x) \) is an odd function defined on [-1, 1] with \( f(1) = 1 \).
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f(-x) = -f(x)
def f_defined_on_Icc : Prop := ∀ x ∈ (set.Icc (-1 : ℝ) 1), f x = f x
def f_at_1 : Prop := f (1 : ℝ) = 1

-- Condition: For any \(a, b \in [-1, 1]\) and \(a + b \neq 0\), \(\frac{f(a) + f(b)}{a + b} > 0\).
def f_ratio_positive (f : ℝ → ℝ) : Prop :=
  ∀ a b, a ∈ (set.Icc (-1:ℝ) 1) → b ∈ (set.Icc (-1:ℝ) 1) → a + b ≠ 0 → (f(a) + f(b)) / (a + b) > 0

-- Create a context bundling these properties
structure f_properties (f : ℝ → ℝ) : Prop :=
  (odd : is_odd f)
  (defined_on_Icc : f_defined_on_Icc)
  (at_1 : f_at_1)
  (ratio_positive : f_ratio_positive f)

-- Formal Statement of (Ⅰ): Proving f is increasing on [-1, 1]
theorem f_increasing (hf : f_properties f) : ∀ x₁ x₂ ∈ (set.Icc (-1:ℝ) 1), x₁ < x₂ → f x₁ < f x₂ := sorry

-- Formal Statement of (Ⅱ): Solving the inequality f(x + 1/2) < f(1 - x)
theorem solve_inequality (hf : f_properties f) : ∀ (x : ℝ), 0 ≤ x ∧ x < 1/4 ↔ f (x + 1/2) < f (1 - x) := sorry

-- Formal Statement of (Ⅲ): Finding the range of m
theorem range_of_m (hf : f_properties f) : ∀ (m : ℝ), (∀ x ∈ (set.Icc (-1:ℝ) 1), f x ≤ m^2 - 2*m + 1) ↔ m ≤ 0 ∨ m ≥ 2 := sorry

end f_increasing_solve_inequality_range_of_m_l369_369837


namespace gcd_of_18_and_30_l369_369551

theorem gcd_of_18_and_30 : Nat.gcd 18 30 = 6 :=
by
  sorry

end gcd_of_18_and_30_l369_369551


namespace area_of_triangle_OAB_l369_369944

-- Definition of vectors OA and OB
def vec_OA (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, 2 * Real.sin α)
def vec_OB (β : ℝ) : ℝ × ℝ := (5 * Real.cos β, 5 * Real.sin β)

-- Dot product of OA and OB
def dot_product (α β : ℝ) : ℝ := (vec_OA α).1 * (vec_OB β).1 + (vec_OA α).2 * (vec_OB β).2

-- Condition: Dot product is -5
axiom dot_product_condition (α β : ℝ) : dot_product α β = -5

-- Magnitudes of OA and OB
def mag_OA (α : ℝ) : ℝ := Real.sqrt ((vec_OA α).1 ^ 2 + (vec_OA α).2 ^ 2)
def mag_OB (β : ℝ) : ℝ := Real.sqrt ((vec_OB β).1 ^ 2 + (vec_OB β).2 ^ 2)

lemma mag_OA_result (α : ℝ) : mag_OA α = 2 := by
  simp [vec_OA, mag_OA]
  norm_num

lemma mag_OB_result (β : ℝ) : mag_OB β = 5 := by
  simp [vec_OB, mag_OB]
  norm_num

-- Cosine of the angle between OA and OB
def cos_theta (α β : ℝ) : ℝ := dot_product α β / (mag_OA α * mag_OB β)

-- Sin of the angle between OA and OB using the trigonometric identity
def sin_theta (α β : ℝ) : ℝ := Real.sqrt (1 - cos_theta α β ^ 2)

-- Area of triangle OAB
def area_triangle (α β : ℝ) : ℝ := 1 / 2 * mag_OA α * mag_OB β * sin_theta α β

-- The proof problem
theorem area_of_triangle_OAB (α β : ℝ) (h : dot_product α β = -5) : area_triangle α β = 5 * Real.sqrt 3 / 2 := by
  simp [area_triangle, mag_OA_result, mag_OB_result, cos_theta, sin_theta]
  have h1 : dot_product α β / (mag_OA α * mag_OB β) = -1 / 2 := by
    rw [h, mag_OA_result, mag_OB_result]
    norm_num
  rw [h1]
  simp [Real.sqrt]
  norm_num
  sorry

end area_of_triangle_OAB_l369_369944


namespace dodecagon_product_l369_369474

theorem dodecagon_product :
  let center : ℂ := 2 + 1 * complex.i,
      vertex1 : ℂ := 3 + 1 * complex.i,
      polygon := finset.image (λ n, center + vertex1 * complex.exp (2 * π * complex.i * n / 12)) (finset.range 12) in
  (finset.prod polygon id) = -2926 - 3452 * complex.i :=
by
  sorry

end dodecagon_product_l369_369474


namespace volume_space_equals_l369_369485

noncomputable def volume_of_space_inside_sphere_outside_cylinder : ℝ :=
  let r     := 4
  let R     := 7
  let h     := 2 * Real.sqrt 33
  let V_sphere := (4/3) * Real.pi * R^3
  let V_cylinder := Real.pi * (r^2) * h
  (V_sphere - V_cylinder)

theorem volume_space_equals (Y : ℝ) (hY: Y = 841.6 / 3): 
  volume_of_space_inside_sphere_outside_cylinder = Y * Real.pi :=
by 
  rw hY
  sorry

end volume_space_equals_l369_369485


namespace prime_product_mod_32_l369_369294

open Nat

theorem prime_product_mod_32 : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := List.foldr (· * ·) 1 primes
  M % 32 = 25 :=
by
  sorry

end prime_product_mod_32_l369_369294


namespace smallest_n_l369_369310

theorem smallest_n (n : ℕ) (h : n > 1) (h1 : ∀ p, prime p → p ∣ n → n > 1200 * p) :
  n ≥ 3888 :=
sorry

end smallest_n_l369_369310


namespace find_value_l369_369716

-- Define the function $f(x)$
def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

-- Define the problem statement in Lean 4
theorem find_value :
  (∀ x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), 0 ≤ 2 * x - 5 * Real.pi / 6 ∧ 2 * x - 5 * Real.pi / 6 ≤ Real.pi) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by
  -- A proper rigorous Lean proof should go here
  sorry

end find_value_l369_369716


namespace product_of_areas_eq_square_of_volume_l369_369359

theorem product_of_areas_eq_square_of_volume
    (a b c : ℝ)
    (bottom_area : ℝ) (side_area : ℝ) (front_area : ℝ)
    (volume : ℝ)
    (h1 : bottom_area = a * b)
    (h2 : side_area = b * c)
    (h3 : front_area = c * a)
    (h4 : volume = a * b * c) :
    bottom_area * side_area * front_area = volume ^ 2 := by
  -- proof omitted
  sorry

end product_of_areas_eq_square_of_volume_l369_369359


namespace tangent_line_eq_monotonic_intervals_l369_369073

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (1 / 3) * x ^ 3 + m * x ^ 2 + 1
noncomputable def f' (x : ℝ) (m : ℝ) : ℝ := x ^ 2 + 2 * m * x

theorem tangent_line_eq (m : ℝ) (h : f' 1 m = 3) : 
  let f1 := f 1 m
  in let tangent_line := λ (x y : ℝ), 3 * x - 3 * y + 4 = 0
  in tangent_line 1 f1 :=
sorry

theorem monotonic_intervals (m : ℝ) (h : f' 1 m = 3) : 
  let increasing_intervals := (λ x, x < -2) ∨ (λ x, x > 0)
  in let decreasing_intervals := λ x, -2 < x ∧ x < 0
  in ∀ x : ℝ, if x < -2 ∨ x > 0 
     then f' x m > 0 
     else if -2 < x ∧ x < 0 
     then f' x m < 0 
     else true :=
sorry

end tangent_line_eq_monotonic_intervals_l369_369073


namespace gcd_18_30_l369_369602

-- Define the two numbers
def num1 : ℕ := 18
def num2 : ℕ := 30

-- State the theorem to find the gcd
theorem gcd_18_30 : Nat.gcd num1 num2 = 6 := by
  sorry

end gcd_18_30_l369_369602


namespace count_numbers_with_one_prime_digit_l369_369092

-- Definitions of prime and non-prime digits
def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_non_prime_digit (d : ℕ) : Prop :=
  d ≠ 2 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7

-- Definitions of the range and the condition for exactly one prime digit
def in_range (x : ℕ) : Prop :=
  200 ≤ x ∧ x ≤ 600

def count_prime_digits (x : ℕ) : ℕ :=
  (if is_prime_digit (x / 100 % 10) then 1 else 0) +
  (if is_prime_digit (x / 10 % 10) then 1 else 0) +
  (if is_prime_digit (x % 10) then 1 else 0)

def exactly_one_prime_digit (x : ℕ) : Prop :=
  count_prime_digits x = 1

-- Final statement
theorem count_numbers_with_one_prime_digit :
  ∃ (n : ℕ), n = 156 ∧ (∀ x, in_range x → exactly_one_prime_digit x → x ∈ { x : ℕ | in_range x ∧ exactly_one_prime_digit x }.card = n) :=
begin
  sorry
end

end count_numbers_with_one_prime_digit_l369_369092


namespace maximize_volume_of_open_top_container_l369_369406

noncomputable def volume_of_open_top_container (L W h : ℝ) : ℝ :=
  let l := L - 2 * h in
  let w := W - 2 * h in
  l * w * h

theorem maximize_volume_of_open_top_container : 
  ∃ h : ℝ, h = 10 ∧ (∀ x, volume_of_open_top_container 90 48 h ≥ volume_of_open_top_container 90 48 x) :=
sorry

end maximize_volume_of_open_top_container_l369_369406


namespace value_of_f_neg_5π_over_12_l369_369795

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem value_of_f_neg_5π_over_12 :
  ∀ (ω φ : ℝ), 
    (∀ x y : ℝ, (x < y ∧ x ∈ Ioo (π / 6) (2 * π / 3) ∧ y ∈ Ioo (π / 6) (2 * π / 3)) → f x ω φ < f y ω φ) ∧ 
    f (π / 6) ω φ = f (2 * π / 3) ω φ → 
    f (-5 * π / 12) ω φ = √3 / 2 :=
by
  sorry

end value_of_f_neg_5π_over_12_l369_369795


namespace find_other_asymptote_l369_369329

theorem find_other_asymptote
  (asymptote1_eq : ∀ x, y = 2 * x)
  (foci_xcoord : ∀ y, x = 7) :
  ∃ m b, y = m * x + b ↔ m = -1/2 ∧ b = 17.5 :=
sorry

end find_other_asymptote_l369_369329


namespace marc_total_spent_l369_369990

theorem marc_total_spent :
  let model_car_cost := 20
  let paint_cost := 10
  let brush_cost := 2
  let model_car_amount := 5
  let paint_amount := 5
  let brush_amount := 5
  let total_cost := model_car_amount * model_car_cost + paint_amount * paint_cost + brush_amount * brush_cost
  in total_cost = 160 :=
by
  sorry

end marc_total_spent_l369_369990


namespace logarithm_cosine_identity_l369_369627

theorem logarithm_cosine_identity :
  (log (Real.sqrt 2) (Real.cos (20 * Real.pi / 180)) +
   log (Real.sqrt 2) (Real.cos (40 * Real.pi / 180)) +
   log (Real.sqrt 2) (Real.cos (80 * Real.pi / 180)))^2 = 36 :=
by
  -- Proof not required, skipping with sorry
  sorry

end logarithm_cosine_identity_l369_369627


namespace least_whole_number_sub_l369_369417

-- Definitions from conditions
def original_ratio_num : ℕ := 6
def original_ratio_denom : ℕ := 7
def target_ratio_num : ℕ := 16
def target_ratio_denom : ℕ := 21

-- The proof problem
theorem least_whole_number_sub :
  ∃ (x : ℕ), (original_ratio_num - x) / (original_ratio_denom - x) < target_ratio_num / target_ratio_denom ∧
              ∀ (y : ℕ), 
                (original_ratio_num - y) / (original_ratio_denom - y) < target_ratio_num / target_ratio_denom → 
                x ≤ y :=
begin
  sorry
end

end least_whole_number_sub_l369_369417


namespace value_of_f_neg_5π_over_12_l369_369783

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem value_of_f_neg_5π_over_12 :
  ∀ (ω φ : ℝ), 
    (∀ x y : ℝ, (x < y ∧ x ∈ Ioo (π / 6) (2 * π / 3) ∧ y ∈ Ioo (π / 6) (2 * π / 3)) → f x ω φ < f y ω φ) ∧ 
    f (π / 6) ω φ = f (2 * π / 3) ω φ → 
    f (-5 * π / 12) ω φ = √3 / 2 :=
by
  sorry

end value_of_f_neg_5π_over_12_l369_369783


namespace product_mod_32_l369_369209

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  M % 32 = 17 := by
  sorry

end product_mod_32_l369_369209


namespace lines_are_perpendicular_l369_369081

open Real

-- Define lines l1 and l2
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y + 1 = 0
def line2 (x y : ℝ) : Prop := 4 * x - 3 * y + 2 = 0

-- Define slopes
def slope1 : ℝ := -3 / 4
def slope2 : ℝ := 4 / 3

-- The theorem stating that the lines are perpendicular
theorem lines_are_perpendicular : slope1 * slope2 = -1 :=
by sorry

end lines_are_perpendicular_l369_369081


namespace find_value_l369_369717

-- Define the function $f(x)$
def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

-- Define the problem statement in Lean 4
theorem find_value :
  (∀ x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), 0 ≤ 2 * x - 5 * Real.pi / 6 ∧ 2 * x - 5 * Real.pi / 6 ≤ Real.pi) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by
  -- A proper rigorous Lean proof should go here
  sorry

end find_value_l369_369717


namespace trees_died_typhoon_l369_369089

theorem trees_died_typhoon 
  (total_trees_grown : ℕ) 
  (trees_left : ℕ) 
  (total_trees_grown = 17) 
  (trees_left = 12) : 
  total_trees_grown - trees_left = 5 := 
by sorry

end trees_died_typhoon_l369_369089


namespace arithmetic_progression_bounded_sum_l369_369648

noncomputable def a_seq (n : ℕ) : ℝ :=
  if n = 1 then 1 else sorry  -- define as per conditions later

noncomputable def S (n : ℕ) : ℝ := 
  finset.sum (finset.range n) a_seq  -- the sum of the first n terms

lemma seq_first_term : a_seq 1 = 1 := sorry  -- Given condition

lemma seq_n_term (n : ℕ) (hn : 2 ≤ n) : 
  a_seq n = (2 * (S n)^2) / (2 * (S n) - 1) := sorry  -- Given condition for n ≥ 2

theorem arithmetic_progression (n : ℕ) (hn : 2 ≤ n) : 
  (1 / (S (n - 1))) - (1 / (S n)) = 2 := sorry  -- Prove part (1)

theorem bounded_sum (n : ℕ) (hn : 2 ≤ n) : 
  S 1 + (1/2) * S 2 + (1/3) * S 3 + ... + (1/n) * S n < (3/2) := sorry  -- Prove part (2)

end arithmetic_progression_bounded_sum_l369_369648


namespace find_a_solution_l369_369036

open Complex

noncomputable def find_a : Prop := 
  ∃ a : ℂ, ((1 + a * I) / (2 + I) = 1 + 2 * I) ∧ (a = 5 + I)

theorem find_a_solution : find_a := 
  by
    sorry

end find_a_solution_l369_369036


namespace translate_triangle_l369_369362

theorem translate_triangle (A B C A' : (ℝ × ℝ)) (hx_A : A = (2, 1)) (hx_B : B = (4, 3)) 
  (hx_C : C = (0, 2)) (hx_A' : A' = (-1, 5)) : 
  ∃ C' : (ℝ × ℝ), C' = (-3, 6) :=
by 
  sorry

end translate_triangle_l369_369362


namespace fraction_dropped_at_second_station_l369_369492

-- Define the given conditions as constants
def initial_passengers : ℕ := 270
def first_drop_fraction : ℚ := 1/3
def first_increase : ℕ := 280
def second_increase : ℕ := 12
def final_passengers : ℕ := 242

-- Define the fraction of passengers dropped at the second station
def second_drop_fraction := 1/2

-- Formulate the theorem
theorem fraction_dropped_at_second_station :
  let passengers_after_first_drop := initial_passengers - (initial_passengers * first_drop_fraction)
      passengers_after_first_increase := passengers_after_first_drop + first_increase
      passengers_after_second_drop := passengers_after_first_increase - (passengers_after_first_increase * second_drop_fraction)
      passengers_after_second_increase := passengers_after_second_drop + second_increase
  in passengers_after_second_increase = final_passengers :=
by
  let passengers_after_first_drop := initial_passengers - (initial_passengers * first_drop_fraction)
  let passengers_after_first_increase := passengers_after_first_drop + first_increase
  let passengers_after_second_drop := passengers_after_first_increase - (passengers_after_first_increase * second_drop_fraction)
  let passengers_after_second_increase := passengers_after_second_drop + second_increase
  sorry

end fraction_dropped_at_second_station_l369_369492


namespace angle_BCD_l369_369106

theorem angle_BCD (A B C I D: Type)
  (h_triangle: is_isosceles_triangle A B C)
  (h_angle_A: ∠A = 100)
  (h_incenter: is_incenter I A B C)
  (h_BD_BI: dist B D = dist B I):
  ∠BCD = 40 :=
by
  sorry

end angle_BCD_l369_369106


namespace gcd_18_30_l369_369580

theorem gcd_18_30: Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l369_369580


namespace prime_product_mod_32_l369_369296

open Nat

theorem prime_product_mod_32 : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := List.foldr (· * ·) 1 primes
  M % 32 = 25 :=
by
  sorry

end prime_product_mod_32_l369_369296


namespace circle_properties_l369_369644

theorem circle_properties
  (a b : ℝ)
  (h_circle_eq : ∀ (x y : ℝ), x^2 + y^2 - 2 * a * x - 2 * b * y + a^2 + b^2 - 1 = 0)
  (h_center_eq : ∃ p q : ℝ, p = sqrt 3 * a ∧ q = b - sqrt 3 * (a + 1))
  (h_max_dist : ∃ r : ℝ, r = 1 + |sqrt 3 * a + b| / 2 ∧ r = sqrt 3 + 1)
  (h_a_neg : a < 0) :
  a^2 + b^2 = 3 :=
begin
  sorry
end

end circle_properties_l369_369644


namespace max_xy_value_l369_369886

theorem max_xy_value (x y : ℕ) (h : 7 * x + 4 * y = 140) : xy ≤ 168 :=
by sorry

end max_xy_value_l369_369886


namespace product_mod_32_is_15_l369_369240

-- Define the odd primes less than 32
def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the product of odd primes
def M : ℕ := List.foldl (· * ·) 1 oddPrimes

-- The theorem to prove the remainder when M is divided by 32 is 15
theorem product_mod_32_is_15 : M % 32 = 15 := by
  sorry

end product_mod_32_is_15_l369_369240


namespace remainder_of_M_l369_369272

def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_of_M : M % 32 = 31 := by
  -- Proof goes here
  sorry

end remainder_of_M_l369_369272


namespace product_mod_32_is_15_l369_369243

-- Define the odd primes less than 32
def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the product of odd primes
def M : ℕ := List.foldl (· * ·) 1 oddPrimes

-- The theorem to prove the remainder when M is divided by 32 is 15
theorem product_mod_32_is_15 : M % 32 = 15 := by
  sorry

end product_mod_32_is_15_l369_369243


namespace find_principal_l369_369432

theorem find_principal
  (R : ℝ) (T : ℝ) (A : ℝ)
  (hR : R = 0.05)
  (hT : T = 2.4)
  (hA : A = 1680)
  : ∃ P : ℝ, P = 1500 := 
by {
  use 1500,
  -- Proof would go here, but skipped with sorry
  sorry
}

end find_principal_l369_369432


namespace smallest_positive_product_of_given_set_l369_369523

noncomputable def smallest_positive_product (s : Finset ℤ) (n : ℕ) : ℤ :=
  if n = 3 then Finset.min' ((s.pow (Finset.card s)).filter (λ x, x > 0)) sorry else 0

theorem smallest_positive_product_of_given_set :
  smallest_positive_product (Finset.mk [-4, -3, -1, 5, 6] sorry) 3 = 15 :=
sorry

end smallest_positive_product_of_given_set_l369_369523


namespace tan_pi_seven_product_eq_sqrt_seven_l369_369333

theorem tan_pi_seven_product_eq_sqrt_seven :
  (Real.tan (Real.pi / 7)) * (Real.tan (2 * Real.pi / 7)) * (Real.tan (3 * Real.pi / 7)) = Real.sqrt 7 :=
by
  sorry

end tan_pi_seven_product_eq_sqrt_seven_l369_369333


namespace count_perfect_squares_diff_of_consecutive_squares_l369_369338

-- Define the notion of a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

-- Define what it means to be the difference of two consecutive perfect squares
def is_diff_of_consecutive_squares (n : ℕ) : Prop :=
  ∃ b : ℕ, n = (b + 1) * (b + 1) - b * b

-- Define what it means to be an odd number
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Prove the main theorem
theorem count_perfect_squares_diff_of_consecutive_squares :
  {x : ℕ | x < 20000 ∧ is_perfect_square x ∧ is_diff_of_consecutive_squares x}.to_finset.card = 70 :=
by
  sorry

end count_perfect_squares_diff_of_consecutive_squares_l369_369338


namespace fill_time_is_approximately_l369_369344

def box_length : ℝ := 10
def box_width : ℝ := 8
def box_height : ℝ := 4
def fill_rate : ℝ := 6

def box_volume : ℝ := box_length * box_width * box_height
def time_to_fill_box : ℝ := box_volume / fill_rate

theorem fill_time_is_approximately :
  time_to_fill_box ≈ 53.33 := by
  sorry

end fill_time_is_approximately_l369_369344


namespace least_n_for_inequality_l369_369060

theorem least_n_for_inequality (n : ℕ) (h : 1 ≤ n) : (1 / n.toReal - 1 / (n + 1).toReal < 1 / 15) → n = 4 :=
by
  sorry

end least_n_for_inequality_l369_369060


namespace sides_of_regular_polygon_l369_369478

theorem sides_of_regular_polygon (n : ℕ) (h : ∀ (i : ℕ), i < n → (interior_angle i = 140)) :
  n = 9 := 
  sorry

end sides_of_regular_polygon_l369_369478


namespace max_pizzas_l369_369125

theorem max_pizzas (dough_available cheese_available sauce_available pepperoni_available mushroom_available olive_available sausage_available: ℝ)
  (dough_per_pizza cheese_per_pizza sauce_per_pizza toppings_per_pizza: ℝ)
  (total_toppings: ℝ)
  (toppings_per_pizza_sum: total_toppings = pepperoni_available + mushroom_available + olive_available + sausage_available)
  (dough_cond: dough_available = 200)
  (cheese_cond: cheese_available = 20)
  (sauce_cond: sauce_available = 20)
  (pepperoni_cond: pepperoni_available = 15)
  (mushroom_cond: mushroom_available = 5)
  (olive_cond: olive_available = 5)
  (sausage_cond: sausage_available = 10)
  (dough_per_pizza_cond: dough_per_pizza = 1)
  (cheese_per_pizza_cond: cheese_per_pizza = 1/4)
  (sauce_per_pizza_cond: sauce_per_pizza = 1/6)
  (toppings_per_pizza_cond: toppings_per_pizza = 1/3)
  : (min (dough_available / dough_per_pizza) (min (cheese_available / cheese_per_pizza) (min (sauce_available / sauce_per_pizza) (total_toppings / toppings_per_pizza))) = 80) :=
by
  sorry

end max_pizzas_l369_369125


namespace total_area_of_figure_l369_369455

noncomputable def radius_of_circle (d : ℝ) : ℝ := d / 2

noncomputable def area_of_circle (r : ℝ) : ℝ := Real.pi * r ^ 2

def side_length_of_square (d : ℝ) : ℝ := d

def area_of_square (s : ℝ) : ℝ := s ^ 2

noncomputable def total_area (d : ℝ) : ℝ := area_of_square d + area_of_circle (radius_of_circle d)

theorem total_area_of_figure (d : ℝ) (h : d = 6) : total_area d = 36 + 9 * Real.pi :=
by
  -- skipping proof with sorry
  sorry

end total_area_of_figure_l369_369455


namespace product_of_odd_primes_mod_32_l369_369224

theorem product_of_odd_primes_mod_32 :
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31) in
  M % 32 = 25 :=
by
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31)
  sorry

end product_of_odd_primes_mod_32_l369_369224


namespace eliminate_x_l369_369389

theorem eliminate_x (x y : ℝ) (h1 : 6 * x + 2 * y = 4) (h2 : 3 * x - 3 * y = -6) :
  8 * y = 16 :=
by {
  -- Multiply eq2 by 2
  have h2' : 2 * (3 * x - 3 * y) = 2 * (-6) := by ring,
  rw [mul_sub, mul_sub] at h2',
  norm_num at h2',

  -- Subtract h2' from h1
  have h : (6 * x + 2 * y) - (6 * x - 6 * y) = 4 - -12 := by linarith,
  rw [← h2'] at h,
  linarith,
}

end eliminate_x_l369_369389


namespace airplane_altitude_l369_369498

theorem airplane_altitude :
  ∃ (h : ℝ), (Alice_sees_north : ∀ (A : ℝ) (B : ℝ), A - B = 8) ∧
  (Alice_angle : ∀ (h x : ℝ), tan (30 * π / 180) = h / x) ∧
  (Bob_angle : ∀ (h y : ℝ), tan (45 * π / 180) = h / y) ∧
  h ≈ 6.9 :=
by
  sorry

end airplane_altitude_l369_369498


namespace total_time_spent_solving_puzzles_l369_369403

theorem total_time_spent_solving_puzzles :
  let warm_up_puzzle_time := 10 in
  let challenging_puzzle_time := 3 * warm_up_puzzle_time in
  let first_puzzle_set_time := 0.5 * warm_up_puzzle_time in
  let second_puzzle_set_time := 2 * first_puzzle_set_time in
  let third_puzzle_set_time := first_puzzle_set_time + second_puzzle_set_time + 2 in
  let fourth_puzzle_set_time := 1.5 * third_puzzle_set_time in
  warm_up_puzzle_time + 2 * challenging_puzzle_time + 
  first_puzzle_set_time + second_puzzle_set_time + 
  third_puzzle_set_time + fourth_puzzle_set_time = 127.5 :=
by
  sorry

end total_time_spent_solving_puzzles_l369_369403


namespace rectangle_side_ratio_l369_369470

theorem rectangle_side_ratio (a b θ : ℝ) (h1 : tan θ = a / b) (h2 : cos θ = 3 / 5) (h3 : 0 < θ ∧ θ < π / 2) : (a / b)^2 = 16 / 9 :=
by
  sorry

end rectangle_side_ratio_l369_369470


namespace max_k_mono_incr_binom_l369_369634

theorem max_k_mono_incr_binom :
  ∀ (k : ℕ), (k ≤ 11) → 
  (∀ i j : ℕ, 1 ≤ i → i < j → j ≤ k → (Nat.choose 10 (i - 1) < Nat.choose 10 (j - 1))) →
  k = 6 :=
by sorry

end max_k_mono_incr_binom_l369_369634


namespace find_f_of_neg_5_pi_over_12_l369_369734

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - (5 * π / 6))

theorem find_f_of_neg_5_pi_over_12 :
  (∀ x ∈ (set.Ioo (π / 6) (2 * π / 3)), (0 : ℝ) < (f x - f (x - 1))) ∧
  (∀ x, f (π / 6 - x) = f (π / 6 + x)) ∧
  (∀ x, f (2 * π / 3 - x) = f (2 * π / 3 + x)) →
  f (-5 * π / 12) = √3 / 2 :=
by
  sorry

end find_f_of_neg_5_pi_over_12_l369_369734


namespace triangle_side_length_l369_369121

noncomputable def find_c (a b B : ℝ) : ℝ :=
  let c := sqrt (b^2 - a^2 + 2 * a * b * real.cos B)
  c

theorem triangle_side_length (a b B : ℝ) (ha : a = 5) (hb : b = 7) (hB : B = real.pi / 3) :
  find_c a b B = 8 :=
by {
  rw [ha, hb, hB, find_c],
  norm_num,
  sorry
}

end triangle_side_length_l369_369121


namespace product_mod_32_l369_369215

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  M % 32 = 17 := by
  sorry

end product_mod_32_l369_369215


namespace number_of_sides_of_polygon_l369_369387

theorem number_of_sides_of_polygon (n : ℕ) : (n - 2) * 180 = 720 → n = 6 :=
by
  sorry

end number_of_sides_of_polygon_l369_369387


namespace find_f_value_l369_369813

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

theorem find_f_value :
  f ( -5 * Real.pi / 12 ) = Real.sqrt 3 / 2 :=
by
  sorry

end find_f_value_l369_369813


namespace min_distance_parabola_l369_369662

noncomputable def minimum_distance_sum (P : ℝ × ℝ) : ℝ :=
let dist_to_l1 := abs (3 * P.1 - 4 * P.2 + 12) / real.sqrt (3^2 + (-4)^2) in
let dist_to_l2 := abs (P.1 + 2) in
dist_to_l1 + dist_to_l2

theorem min_distance_parabola : ∃ (P : ℝ × ℝ), P.2^2 = 4 * P.1 ∧ minimum_distance_sum P = 4 :=
by {
  use (1, 2),
  split,
  { show (2:ℝ)^2 = 4 * (1:ℝ), by norm_num },
  { show minimum_distance_sum (1, 2) = 4, by {
      dsimp [minimum_distance_sum],
      norm_num,
      rw [abs_of_nonneg, abs_of_nonneg],
      { norm_num },
      { linarith },
      { linarith }
    }
  }
}

end min_distance_parabola_l369_369662


namespace gcd_of_18_and_30_l369_369550

theorem gcd_of_18_and_30 : Nat.gcd 18 30 = 6 :=
by
  sorry

end gcd_of_18_and_30_l369_369550


namespace compute_fractional_exponent_l369_369512

theorem compute_fractional_exponent : (1 / 2) ^ (Real.log 3 / Real.log 2 - 1) = 2 / 3 := by
  sorry

end compute_fractional_exponent_l369_369512


namespace prime_product_mod_32_l369_369292

open Nat

theorem prime_product_mod_32 : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := List.foldr (· * ·) 1 primes
  M % 32 = 25 :=
by
  sorry

end prime_product_mod_32_l369_369292


namespace remainder_of_M_l369_369263

def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_of_M : M % 32 = 31 := by
  -- Proof goes here
  sorry

end remainder_of_M_l369_369263


namespace find_fx_value_l369_369727

noncomputable def f (x : Real) : Real :=
  sin (2 * x - 5 * Real.pi / 6)

theorem find_fx_value :
  (∀ x, f x = sin (2 * x - 5 * Real.pi / 6)) ∧ 
  (f (Real.pi / 6) = f (2 * Real.pi / 3)) ∧ 
  (∀ x1 x2, (Real.pi / 6 < x1 ∧ x1 < 2 * Real.pi / 3 ∧ x1 < x2 ∧ x2 < 2 * Real.pi / 3) -> f x1 < f x2) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 := by
  sorry

end find_fx_value_l369_369727


namespace find_value_l369_369709

-- Define the function $f(x)$
def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

-- Define the problem statement in Lean 4
theorem find_value :
  (∀ x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), 0 ≤ 2 * x - 5 * Real.pi / 6 ∧ 2 * x - 5 * Real.pi / 6 ≤ Real.pi) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by
  -- A proper rigorous Lean proof should go here
  sorry

end find_value_l369_369709


namespace problem_1_problem_2_l369_369074

-- Problem (I)
theorem problem_1 (a : ℝ) (f : ℝ → ℝ) (h_f : ∀ x ∈ set.Ioi 1, f x = log x - a * x)
    (h_decreasing : ∀ x y ∈ set.Ioi 1, x < y → f x ≥ f y) : a ≥ 1 :=
sorry

-- Problem (II)
theorem problem_2 (m x_1 x_2 : ℝ) (h_roots : ∀ x, x = x_1 ∨ x = x_2 ↔ log x + 1 / (2 * x) - m = 0)
    (h_x : 0 < x_1 ∧ x_1 < x_2) : x_1 + x_2 > 1 :=
sorry

end problem_1_problem_2_l369_369074


namespace inclination_angle_range_l369_369658

theorem inclination_angle_range (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 4) :
  ∃ α : ℝ, (∀ α, α = real.arctan (1/a + 1/b) → (π / 4) ≤ α ∧ α < π / 2) :=
sorry

end inclination_angle_range_l369_369658


namespace least_whole_number_subtract_ratio_l369_369415

theorem least_whole_number_subtract_ratio : 
  ∃ x : ℕ, (∀ y : ℕ, y < x → (6 - y) * 3 ≥ (7 - y) * 2.29) ∧
           (6 - x) * 3 < (7 - x) * 2.29 ∧
           y ≤ x :=
begin
  sorry
end

end least_whole_number_subtract_ratio_l369_369415


namespace findStandardEquationOfEllipse_l369_369654

def standardEquationEllipse (a b : ℝ) : Prop := 
  ∀ (x y : ℝ), (x² / a²) + (y² / b²) = 1

def satisfiesCondition (e : ℝ) (A : ℝ × ℝ) (a : ℝ) : Prop := 
  A = (-3, 0) ∧ e = (Real.sqrt 5 / 3) ∧ a = 3

theorem findStandardEquationOfEllipse : 
  ∃ b : ℝ, satisfiesCondition (Real.sqrt 5 / 3) (-3, 0) 3 → standardEquationEllipse 3 b :=
by
  use 2
  intro cond
  sorry

end findStandardEquationOfEllipse_l369_369654


namespace value_of_f_neg_5π_over_12_l369_369794

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem value_of_f_neg_5π_over_12 :
  ∀ (ω φ : ℝ), 
    (∀ x y : ℝ, (x < y ∧ x ∈ Ioo (π / 6) (2 * π / 3) ∧ y ∈ Ioo (π / 6) (2 * π / 3)) → f x ω φ < f y ω φ) ∧ 
    f (π / 6) ω φ = f (2 * π / 3) ω φ → 
    f (-5 * π / 12) ω φ = √3 / 2 :=
by
  sorry

end value_of_f_neg_5π_over_12_l369_369794


namespace min_total_waiting_time_l369_369038

theorem min_total_waiting_time : 
  ∀ (times : List ℕ), times = [4, 5, 6, 8, 10] → 
  (∑ i in range 5, times[i] * (4 - i)) = 84 :=
by
  intro times
  intro h
  rw [h, list.range_eq_healthy]
  have : (∑ i in [0, 1, 2, 3, 4], (λ i, [4, 5, 6, 8, 10][i] * (4 - i))) = 84 :=
    by
      calc
        4 * 4 + 
        5 * 3 + 
        6 * 2 + 
        8 * 1 + 
        10 * 0 = 16 + 15 + 12 + 8 + 0 := rfl
        ... = 51 := rfl
  exact this

print min_total_waiting_time

end min_total_waiting_time_l369_369038


namespace middle_digit_is_zero_l369_369001

noncomputable def N_in_base8 (a b c : ℕ) : ℕ := 512 * a + 64 * b + 8 * c
noncomputable def N_in_base10 (a b c : ℕ) : ℕ := 100 * b + 10 * c + a

theorem middle_digit_is_zero (a b c : ℕ) (h : N_in_base8 a b c = N_in_base10 a b c) :
  b = 0 :=
by 
  sorry

end middle_digit_is_zero_l369_369001


namespace gcd_18_30_l369_369572

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := 
by
  sorry

end gcd_18_30_l369_369572


namespace remainder_when_M_divided_by_32_l369_369139

-- Define M as the product of all odd primes less than 32.
def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_M_divided_by_32 :
    M % 32 = 1 := by
  -- We must prove that M % 32 = 1.
  sorry

end remainder_when_M_divided_by_32_l369_369139


namespace gcd_18_30_l369_369585

theorem gcd_18_30: Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l369_369585


namespace find_f_value_l369_369809

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

theorem find_f_value :
  f ( -5 * Real.pi / 12 ) = Real.sqrt 3 / 2 :=
by
  sorry

end find_f_value_l369_369809


namespace sin_monotonic_increasing_f_symmetric_axes_find_value_l369_369703

def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi * 5 / 6)

theorem sin_monotonic_increasing (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) :
  Monotone (λ x, f x) (Ioo a b) := 
sorry

theorem f_symmetric_axes (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) 
  (x y : ℝ) (hx : x = a + (b - a) * k) (hy : y = a + (b - a) * (-k)) : 
  f x = f y :=
sorry

theorem find_value :
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
sorry

end sin_monotonic_increasing_f_symmetric_axes_find_value_l369_369703


namespace remove_increases_probability_l369_369486

def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}

theorem remove_increases_probability :
  (∀ n ∈ S, n = 7 ∨ n = 8) → (∃ A ⊆ S, A = S \ {n} ∧ n = 7 ∨ n = 8) → (∃ s₁ s₂ ∈ A, s₁ < s₂ ∧ s₁ + s₂ = 15) → 
  sorry

end remove_increases_probability_l369_369486


namespace gcd_18_30_l369_369606

-- Define the two numbers
def num1 : ℕ := 18
def num2 : ℕ := 30

-- State the theorem to find the gcd
theorem gcd_18_30 : Nat.gcd num1 num2 = 6 := by
  sorry

end gcd_18_30_l369_369606


namespace single_elimination_games_l369_369931

theorem single_elimination_games (n : ℕ) (h : n = 512) : ∃ g : ℕ, g = 511 :=
by
  use 511
  rw [←h]
  exact nat.sub_self (succ_pred_eq_of_pos (nat.pos_of_ne_zero (ne_of_eq_of_ne h (ne.symm (ne_of_eq h rfl)))))

end single_elimination_games_l369_369931


namespace remainder_when_divided_by_32_l369_369161

def product_of_odds : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_divided_by_32 : product_of_odds % 32 = 9 := by
  sorry

end remainder_when_divided_by_32_l369_369161


namespace find_x_inversely_as_cube_l369_369392

theorem find_x_inversely_as_cube (x y : ℝ) (h₁ : y = 2) (h₂ : x = 3) (h₃ : y * x^3 = 54)
  (h₄ : y = 18) : x = real.cbrt 3 :=
by
  sorry

end find_x_inversely_as_cube_l369_369392


namespace num_factors_m_eq_378_l369_369975

noncomputable def m : ℕ := 2^5 * 3^6 * 5^2 * 10^3

theorem num_factors_m_eq_378 :
  ∃ n, n = m ∧ (∏ p in n.factorization.support, (n.factorization p + 1)) = 378 :=
by
  sorry

end num_factors_m_eq_378_l369_369975


namespace geometric_sequence_general_term_arithmetic_sequence_sum_l369_369671

variable {n : ℕ}

-- Defining sequences and sums
def S (n : ℕ) : ℕ := sorry
def a (n : ℕ) : ℕ := sorry
def T (n : ℕ) : ℕ := sorry
def b (n : ℕ) : ℕ := sorry

-- Given conditions
axiom h1 : 2 * S n = 3 * a n - 3
axiom h2 : b 1 = a 1
axiom h3 : b 7 = b 1 * b 2
axiom a1_value : a 1 = 3
axiom d_value : ∃ d : ℕ, b 2 = b 1 + d ∧ b 7 = b 1 + 6 * d

theorem geometric_sequence_general_term : a n = 3 ^ n :=
by sorry

theorem arithmetic_sequence_sum : T n = n^2 + 2*n :=
by sorry

end geometric_sequence_general_term_arithmetic_sequence_sum_l369_369671


namespace count_brasil_paths_l369_369843

theorem count_brasil_paths (grid : list (list char)) :
  (count_paths grid 'B' 'R' 'A' 'S' 'I' 'L' 6 1) = 32 :=
  sorry

end count_brasil_paths_l369_369843


namespace num_arithmetic_sequences_l369_369054

theorem num_arithmetic_sequences
  (a d n : ℕ)
  (h1 : n ≥ 3)
  (h2 : n * (2 * a + (n - 1) * d) = 18818) :
  ∃ (S : finset (ℕ × ℕ × ℕ)), 
    S.card = 4 ∧ 
    ∀ (t ∈ S), let (a, d, n) := t in 
      n ≥ 3 ∧ 
      n * (2 * a + (n - 1) * d) = 18818 :=
by
  sorry

end num_arithmetic_sequences_l369_369054


namespace ellipse_proof_problem_l369_369065

section ellipse_problem

variables {a c : ℝ}
variables (M : {x y : ℝ // (x^2) / a^2 + (y^2) / 3 = 1})
variables (F : {x y : ℝ // y = 0 ∧ x = c})
variables (S : {x y : ℝ // M  ≃ symm about line x = c})
variables (P Q : {x y : ℝ // y = 0 ∧ x = 4})
variables (E : {x y : ℝ // y = -Q.2 ∧ x = Q.1})

-- Condition that symmetric graph about the line passes through the origin
def symmetric_graph_through_origin (S : Prop) : Prop :=
  ∃ x y : ℝ, (x = 0 ∧ y = 0) ∧ S 

-- The statement of the problem
theorem ellipse_proof_problem :
  (a = 2 * c) →
  (a^2 = 4) →
  (S → (x^2)/4 + (y^2)/3 = 1) →
  ∀ (k : ℝ), 
  (linear_existence_of_slope (k ≠ 0)) →
  (intersection_of_PQ (x-intersecting y-axis)) →
  ∃ F : {x y : ℝ // y = 0 ∧ x = 1}, (line PE intersects x-axis at F) :=
sorry

end ellipse_problem

end ellipse_proof_problem_l369_369065


namespace remainder_when_divided_by_32_l369_369157

def product_of_odds : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_divided_by_32 : product_of_odds % 32 = 9 := by
  sorry

end remainder_when_divided_by_32_l369_369157


namespace trig_identity_l369_369661

open Real

theorem trig_identity (α : ℝ) (h : 2 * sin α + cos α = 0) : 
  2 * sin α ^ 2 - 3 * sin α * cos α - 5 * cos α ^ 2 = -12 / 5 :=
sorry

end trig_identity_l369_369661


namespace sin_function_value_l369_369804

noncomputable def f (x : ℝ) : ℝ := sin(2 * x - 5 * π / 6)

theorem sin_function_value :
  f (-5 * π / 12) = sqrt 3 / 2 := by
  sorry

end sin_function_value_l369_369804


namespace range_alpha_minus_beta_l369_369848

theorem range_alpha_minus_beta (α β : ℝ) (h1 : -π ≤ α) (h2 : α ≤ β) (h3 : β ≤ π / 2) :
  - (3 * π) / 2 ≤ α - β ∧ α - β ≤ 0 :=
sorry

end range_alpha_minus_beta_l369_369848


namespace kimiko_watched_4_videos_l369_369130

/-- Kimiko's videos. --/
def first_video_length := 120
def second_video_length := 270
def last_two_video_length := 60
def total_time_watched := 510

theorem kimiko_watched_4_videos :
  first_video_length + second_video_length + last_two_video_length + last_two_video_length = total_time_watched → 
  4 = 4 :=
by
  intro h
  sorry

end kimiko_watched_4_videos_l369_369130


namespace cos_eq_half_l369_369391

-- Define the specific problem conditions
def cos_special_angle (x : ℝ) : Prop := 
  (x = - (17 / 3) * Real.pi → Real.cos x = 1 / 2)

-- Define the theorem to be proved
theorem cos_eq_half : cos_special_angle (- (17 / 3) * Real.pi) := 
by 
  intro h
  rw [Real.cos_neg, ← h]
  sorry

end cos_eq_half_l369_369391


namespace product_of_odd_primes_mod_32_l369_369181

theorem product_of_odd_primes_mod_32 :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  M % 32 = 17 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  exact sorry

end product_of_odd_primes_mod_32_l369_369181


namespace remainder_of_product_of_odd_primes_mod_32_l369_369277

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem remainder_of_product_of_odd_primes_mod_32 :
  (product odd_primes_less_than_32) % 32 = 23 :=
by sorry

end remainder_of_product_of_odd_primes_mod_32_l369_369277


namespace range_S_3_l369_369667

variable {α : Type*} [LinearOrderedField α]

-- Define the geometric sequence terms and conditions
def geometric_sequence (a1 : α) (q : α) (n : ℕ) : α := a1 * q^(n - 1)
def a2 : α := 1
def q : α := sorry  -- q > 0 is required but not specifically provided

-- Define the sum of the first 3 terms in the geometric sequence
def S_3 (a1 : α) (q : α) : α := a1 + a2 + a2 * q

-- Proving the range of S_3
theorem range_S_3 (a1 : α) (q > 0) : S_3 a1 q ≥ 3 :=
by
  sorry

end range_S_3_l369_369667


namespace find_x0_l369_369639

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1

theorem find_x0 (x_0 : ℝ) (h : f' x_0 = 2) : x_0 = Real.exp 1 :=
by
  sorry

end find_x0_l369_369639


namespace product_mod_32_l369_369218

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  M % 32 = 17 := by
  sorry

end product_mod_32_l369_369218


namespace find_lambda_l369_369835

variables {V : Type*} [add_comm_group V] [module ℝ V]
variables (a b : V) (λ : ℝ)

-- Assumptions
variables (h1 : ¬ (∃ k : ℝ, a = k • b))
variables (h2 : ∃ k : ℝ, λ • a + b = k • (a - 2 • b))

theorem find_lambda (h1 : ¬ (∃ k : ℝ, a = k • b)) 
  (h2 : ∃ k : ℝ, λ • a + b = k • (a - 2 • b)) :
  λ = -1/2 :=
sorry

end find_lambda_l369_369835


namespace trapezoid_longer_parallel_side_length_l369_369487

theorem trapezoid_longer_parallel_side_length
    (square_side_length : ℝ)
    (h1 : square_side_length = 2)
    (mid_point : ℝ)
    (h2 : mid_point = square_side_length / 2)
    (area_large_square : ℝ)
    (h3 : area_large_square = square_side_length^2)
    (smaller_square_side_length : ℝ)
    (h4 : smaller_square_side_length = mid_point)
    (area_smaller_square : ℝ)
    (h5 : area_smaller_square = smaller_square_side_length^2)
    (trapezoid_area : ℝ)
    (h6 : trapezoid_area = (area_large_square - area_smaller_square) / 2)
    (total_area_outside_smaller_square : ℝ)
    (h7 : total_area_outside_smaller_square = area_large_square - area_smaller_square) :
    ∃ (x: ℝ), trapezoid_area = 1 / 2 * (square_side_length + x) * mid_point ∧ x = 1 := 
begin
  sorry
end

end trapezoid_longer_parallel_side_length_l369_369487


namespace range_of_a_l369_369677

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (3 - a) * x + 1
noncomputable def g (x : ℝ) : ℝ := x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x > 0 ∨ g x > 0) → a ∈ set.Ico 0 9 :=
by
  sorry

end range_of_a_l369_369677


namespace problem_statement_l369_369690

-- Definition of our function f
def f (x : ℝ) : ℝ := Real.sin (2 * x - (5 * Real.pi / 6))

-- Theorem to prove that f(-5π/12) = √3/2
theorem problem_statement : 
  f(-5 * Real.pi / 12) = sqrt 3 / 2 := 
sorry

end problem_statement_l369_369690


namespace find_m_l369_369851

open Int

theorem find_m (c d : ℕ) (h1 : c ≡ 27 [MOD 53]) (h2 : d ≡ 98 [MOD 53]) : 
  ∃ m : ℤ, m ∈ ({150, 151, ..., 201} : Set ℤ) ∧ (c - d : ℤ) ≡ m [MOD 53] :=
by
  sorry

end find_m_l369_369851


namespace line_through_2_1_with_equal_intercepts_l369_369363

noncomputable def line_equation {α : Type} [Field α] (P : α × α) (k : α) : Prop :=
  (∃ a : α, P = (a, a) ∧ (∀ x y : α, P = (x, y) → y = k * x)) ∨
  (∃ b : α, P = (2, 1) ∧ (∀ x y : α, P = (x, y) → x + y = b))

theorem line_through_2_1_with_equal_intercepts :
  line_equation (2 : ℝ, 1 : ℝ) 1/2 :=
by
  sorry

end line_through_2_1_with_equal_intercepts_l369_369363


namespace product_of_odd_primes_mod_32_l369_369185

theorem product_of_odd_primes_mod_32 :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  M % 32 = 17 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  exact sorry

end product_of_odd_primes_mod_32_l369_369185


namespace gcd_18_30_l369_369535

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l369_369535


namespace conjugate_z_eq_i_l369_369673

noncomputable def z : ℂ := (1 - complex.I) / (1 + complex.I)

theorem conjugate_z_eq_i : conj (z) = complex.I :=
by
  sorry

end conjugate_z_eq_i_l369_369673


namespace area_ADP_product_perfect_square_l369_369429

section

variables (S_ABP S_BCP S_CDP S_ADP : ℕ)

-- Part (a)
theorem area_ADP (h : S_ADP = S_ABP * S_CDP / S_BCP) : 
  S_ADP = S_ABP * S_CDP / S_BCP :=
sorry

-- Part (b)
theorem product_perfect_square (h1 : S_ADP = S_ABP * S_CDP / S_BCP) : 
  ∃ k : ℕ, (S_ABP * S_BCP * S_CDP * S_ADP) = k^2 :=
begin
  use S_ABP * S_CDP,
  rw [h1],
  ring,
end

end

end area_ADP_product_perfect_square_l369_369429


namespace equation_solution_count_l369_369093

open Real

theorem equation_solution_count :
  ∃ s : Finset ℝ, (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * π ∧ sin (π / 4 * sin x) = cos (π / 4 * cos x)) ∧ s.card = 4 :=
by
  sorry

end equation_solution_count_l369_369093


namespace gcd_18_30_l369_369543

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l369_369543


namespace find_p_l369_369532

variable {x : ℝ}

def polynomial_eq := 
  (λ d p q, (4 * x^2 - 2 * x + 5 / 2) * (d * x^2 + p * x + q) = 12 * x^4 - 7 * x^3 + 12 * x^2 - 15 / 2 * x + 5)

theorem find_p (d p q : ℝ) (hp : 4 * p - 2 * d = -7)
  (hd : 4 * d = 12) :
  polynomial_eq d p q → p = -1 / 4 :=
by
  sorry

end find_p_l369_369532


namespace product_of_solutions_abs_eq_l369_369032

theorem product_of_solutions_abs_eq (x : ℝ) :
  (∃ x1 x2 : ℝ, |6 * x1 + 2| + 5 = 47 ∧ |6 * x2 + 2| + 5 = 47 ∧ x ≠ x1 ∧ x ≠ x2 ∧ x1 * x2 = -440 / 9) :=
by
  sorry

end product_of_solutions_abs_eq_l369_369032


namespace final_balance_is_103_5_percent_of_initial_l369_369993

/-- Define Megan's initial balance. -/
def initial_balance : ℝ := 125

/-- Define the balance after 25% increase from babysitting. -/
def after_babysitting (balance : ℝ) : ℝ :=
  balance + (balance * 0.25)

/-- Define the balance after 20% decrease from buying shoes. -/
def after_shoes (balance : ℝ) : ℝ :=
  balance - (balance * 0.20)

/-- Define the balance after 15% increase by investing in stocks. -/
def after_stocks (balance : ℝ) : ℝ :=
  balance + (balance * 0.15)

/-- Define the balance after 10% decrease due to medical expenses. -/
def after_medical_expense (balance : ℝ) : ℝ :=
  balance - (balance * 0.10)

/-- Define the final balance. -/
def final_balance : ℝ :=
  let b1 := after_babysitting initial_balance
  let b2 := after_shoes b1
  let b3 := after_stocks b2
  after_medical_expense b3

/-- Prove that the final balance is 103.5% of the initial balance. -/
theorem final_balance_is_103_5_percent_of_initial :
  final_balance / initial_balance = 1.035 :=
by
  unfold final_balance
  unfold initial_balance
  unfold after_babysitting
  unfold after_shoes
  unfold after_stocks
  unfold after_medical_expense
  sorry

end final_balance_is_103_5_percent_of_initial_l369_369993


namespace remainder_when_divided_by_32_l369_369151

def product_of_odds : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_divided_by_32 : product_of_odds % 32 = 9 := by
  sorry

end remainder_when_divided_by_32_l369_369151


namespace find_f_of_neg_5_pi_over_12_l369_369735

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - (5 * π / 6))

theorem find_f_of_neg_5_pi_over_12 :
  (∀ x ∈ (set.Ioo (π / 6) (2 * π / 3)), (0 : ℝ) < (f x - f (x - 1))) ∧
  (∀ x, f (π / 6 - x) = f (π / 6 + x)) ∧
  (∀ x, f (2 * π / 3 - x) = f (2 * π / 3 + x)) →
  f (-5 * π / 12) = √3 / 2 :=
by
  sorry

end find_f_of_neg_5_pi_over_12_l369_369735


namespace find_value_l369_369713

-- Define the function $f(x)$
def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

-- Define the problem statement in Lean 4
theorem find_value :
  (∀ x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), 0 ≤ 2 * x - 5 * Real.pi / 6 ∧ 2 * x - 5 * Real.pi / 6 ≤ Real.pi) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by
  -- A proper rigorous Lean proof should go here
  sorry

end find_value_l369_369713


namespace obtuse_triangle_side_lengths_consecutive_l369_369630

theorem obtuse_triangle_side_lengths_consecutive :
  ∃ (a b c : ℕ), a < b ∧ b < c ∧ a + 1 = b ∧ b + 1 = c ∧
  (a * a + b * b < c * c) ∧ a = 2 ∧ b = 3 ∧ c = 4 :=
by
  let n : ℕ := 3
  use [2, 3, 4]
  split
  case h₀ => exact by decide
  split
  case h₁ => exact by decide
  split
  case h₂ => exact by decide
  split
  case h₃ => exact by decide
  split
  case h₄ => exact dec_trivial
  split
  case h₅ => exact rfl
  case h₆ => exact rfl
  case h₇ => exact rfl
  sorry

end obtuse_triangle_side_lengths_consecutive_l369_369630


namespace prime_product_mod_32_l369_369302

open Nat

theorem prime_product_mod_32 : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := List.foldr (· * ·) 1 primes
  M % 32 = 25 :=
by
  sorry

end prime_product_mod_32_l369_369302


namespace SufficientCondition_NotNecessaryCondition_SufficientButNotNecessary_l369_369656

noncomputable def PointsNotCoplanar (E F G H : Point) : Prop := 
  ¬(coplanar {E, F, G, H})

noncomputable def LinesDoNotIntersect (E F G H : Point) : Prop := 
  ¬(intersect (lineThrough E F) (lineThrough G H))

theorem SufficientCondition (E F G H : Point) :
  PointsNotCoplanar E F G H → LinesDoNotIntersect E F G H :=
sorry

theorem NotNecessaryCondition (E F G H : Point) :
  LinesDoNotIntersect E F G H → PointsNotCoplanar E F G H :=
sorry

theorem SufficientButNotNecessary (E F G H : Point) :
  (PointsNotCoplanar E F G H → LinesDoNotIntersect E F G H) ∧
  ¬(LinesDoNotIntersect E F G H → PointsNotCoplanar E F G H) :=
  by
    apply And.intro
    · exact SufficientCondition E F G H
    · exact NotNecessaryCondition E F G H

end SufficientCondition_NotNecessaryCondition_SufficientButNotNecessary_l369_369656


namespace gcd_of_18_and_30_l369_369594

-- Define the numbers
def num1 := 18
def num2 := 30

-- State the GCD property
theorem gcd_of_18_and_30 : Nat.gcd num1 num2 = 6 :=
by
  sorry

end gcd_of_18_and_30_l369_369594


namespace zoey_correct_percentage_l369_369426

-- Define number of problems in each test and the scores
def test1_problems : ℕ := 30
def test2_problems : ℕ := 50
def test3_problems : ℕ := 20
def test4_problems : ℕ := 40

def score1 : ℝ := 0.85
def score2 : ℝ := 0.75
def score3 : ℝ := 0.65
def score4 : ℝ := 0.95

-- Calculate the number of problems answered correctly in each test
def correct1 : ℝ := score1 * test1_problems
def correct2 : ℝ := score2 * test2_problems
def correct3 : ℝ := score3 * test3_problems
def correct4 : ℝ := score4 * test4_problems

-- Sum the number of problems answered correctly
def total_correct : ℝ := correct1 + correct2 + correct3 + correct4

-- Calculate the total number of problems
def total_problems : ℕ := test1_problems + test2_problems + test3_problems + test4_problems

-- Define the overall percentage of problems answered correctly
def overall_percentage : ℝ := (total_correct / total_problems) * 100

-- Statement to prove
theorem zoey_correct_percentage : overall_percentage = 81.43 := by
  sorry

end zoey_correct_percentage_l369_369426


namespace gcd_of_18_and_30_l369_369595

-- Define the numbers
def num1 := 18
def num2 := 30

-- State the GCD property
theorem gcd_of_18_and_30 : Nat.gcd num1 num2 = 6 :=
by
  sorry

end gcd_of_18_and_30_l369_369595


namespace gcd_binom_div_l369_369332

theorem gcd_binom_div (m n : ℕ) (hm : 1 ≤ m) (hn : m ≤ n) : 
  (gcd m n * nat.factorial n) / (n * (nat.factorial m * nat.factorial (n - m))) ∈ ℤ := by
  sorry

end gcd_binom_div_l369_369332


namespace gcd_18_30_l369_369578

theorem gcd_18_30: Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l369_369578


namespace sum_of_second_largest_and_second_smallest_eq_22_l369_369399

/-- Problem statement:
Given three numbers 10, 11, and 12, prove that the sum of the second largest number
and the second smallest number is 22.
-/
theorem sum_of_second_largest_and_second_smallest_eq_22 (a b c : ℕ) (h1 : a = 10) (h2 : b = 11) (h3 : c = 12) :
  let l := [a, b, c].qsort (≤) in
  (l.nth 1).get_or_else 0 + (l.nth 1).get_or_else 0 = 22 :=
by
  sorry

end sum_of_second_largest_and_second_smallest_eq_22_l369_369399


namespace gcd_18_30_l369_369583

theorem gcd_18_30: Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l369_369583


namespace distinct_points_count_l369_369369

theorem distinct_points_count :
  ∀ x y : ℝ, (x^2 + 4*y^2 = 4) ∧ (4*x^2 - y^2 = 1) ↔ ∃! p : ℝ × ℝ ∈ (set.univ : set (ℝ × ℝ)), (p.fst^2 + 4*p.snd^2 = 4) ∧ (4*p.fst^2 - p.snd^2 = 1) :=
by
  sorry

end distinct_points_count_l369_369369


namespace problem_statement_l369_369683

-- Definition of our function f
def f (x : ℝ) : ℝ := Real.sin (2 * x - (5 * Real.pi / 6))

-- Theorem to prove that f(-5π/12) = √3/2
theorem problem_statement : 
  f(-5 * Real.pi / 12) = sqrt 3 / 2 := 
sorry

end problem_statement_l369_369683


namespace product_of_odd_primes_mod_32_l369_369177

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def M : ℕ := odd_primes_less_than_32.foldr (· * ·) 1

theorem product_of_odd_primes_mod_32 : M % 32 = 17 :=
by
  -- the proof goes here
  sorry

end product_of_odd_primes_mod_32_l369_369177


namespace product_mod_32_is_15_l369_369237

-- Define the odd primes less than 32
def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the product of odd primes
def M : ℕ := List.foldl (· * ·) 1 oddPrimes

-- The theorem to prove the remainder when M is divided by 32 is 15
theorem product_mod_32_is_15 : M % 32 = 15 := by
  sorry

end product_mod_32_is_15_l369_369237


namespace probability_recurrence_relation_l369_369892

theorem probability_recurrence_relation (n k : ℕ) (h : k < n) :
  ∀ (p : ℕ → ℕ → ℝ), p n k = p (n-1) k - (1 / (2:ℝ)^k) * p (n-k) k + 1 / (2:ℝ)^k := 
sorry

end probability_recurrence_relation_l369_369892


namespace range_of_a_l369_369831

def f (m : ℝ) : ℝ → ℝ := λ x, (-2 * m^2 + m + 2) * x^(m + 1)

def y (m a x : ℝ) : ℝ := f(m) x - 4 * (a - 1) * x

theorem range_of_a (m a : ℝ) : 
  (-2 * m^2 + m + 2 = 1) →
  (∀ x, 2 < x ∧ x < 4 → deriv (y m a) x = 0 ∨ deriv (y m a) x ≠ 0) → 
  a ∈ set.Iic 2 ∪ set.Ici 3 := 
sorry

end range_of_a_l369_369831


namespace max_neg_square_in_interval_l369_369316

variable (f : ℝ → ℝ)

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, 0 < x → x < y → f x < f y

noncomputable def neg_square_val (x : ℝ) : ℝ :=
  - (f x) ^ 2

theorem max_neg_square_in_interval : 
  (∀ x_1 x_2 : ℝ, f (x_1 + x_2) = f x_1 + f x_2) →
  f 1 = 2 →
  is_increasing f →
  (∀ x : ℝ, f (-x) = - f x) →
  ∃ b ∈ (Set.Icc (-3) (-2)), 
  ∀ x ∈ (Set.Icc (-3) (-2)), neg_square_val f x ≤ neg_square_val f b ∧ neg_square_val f b = -16 := 
sorry

end max_neg_square_in_interval_l369_369316


namespace product_of_repeating_decimal_l369_369010

noncomputable def repeating_decimal := "0." ++ "356".repeat

theorem product_of_repeating_decimal : 
  let t := 0.356356356... (repeated) in  -- This is a placeholder for the actual repeating decimal construction
  t * 12 = 1424 / 333 :=
by
  -- Definitions based on given conditions
  let t := 356 / 999
  have h1 : ∀ t, 1000 * t = 356 + t, from sorry
  have h2 : t = 356 / 999, from sorry
  have h3 : 12 * t = 4272 / 999, from sorry
  have h4 : (4272 / 999) = (1424 / 333), from sorry
  -- Conclusion
  exact h4

end product_of_repeating_decimal_l369_369010


namespace product_of_odd_primes_mod_32_l369_369186

theorem product_of_odd_primes_mod_32 :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  M % 32 = 17 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  exact sorry

end product_of_odd_primes_mod_32_l369_369186


namespace CQ_perpendicular_BP_l369_369651

open_locale classical

variables {A B C H M P Q : Type} [metric_space A]
variables {triangle_ABC : A}
variables {H_on_AH : H ∈ line A B}
variables {M_on_CM : M ∈ line C A}
variables {P_intersection : same_side_points P A B C}
variables {Q_from_BH_perp_CM : Q ∈ line B H ∧ perpendicular H CM}

def altitude_from_A (A B C : Type) : Type := sorry
def median_from_C (A B C : Type) : Type := sorry
def intersection_of_AH_and_CM (A B C : Type) : Type := sorry
def altitude_from_B (A B C : Type) (H : A) : Type := sorry
def perpendicular_from_point (P : A) (CM : A) : Type := sorry

theorem CQ_perpendicular_BP (A B C H M P Q : Type) 
  (triangle_ABC : A)
  (H_on_AH : H ∈ line A B)
  (M_on_CM : M ∈ line C A)
  (P_intersection : same_side_points P A B C)
  (Q_from_BH_perp_CM : Q ∈ line B H ∧ perpendicular H CM) :
  perpendicular (line Q C) (line P B) :=
sorry

end CQ_perpendicular_BP_l369_369651


namespace value_of_f_neg_5π_over_12_l369_369785

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem value_of_f_neg_5π_over_12 :
  ∀ (ω φ : ℝ), 
    (∀ x y : ℝ, (x < y ∧ x ∈ Ioo (π / 6) (2 * π / 3) ∧ y ∈ Ioo (π / 6) (2 * π / 3)) → f x ω φ < f y ω φ) ∧ 
    f (π / 6) ω φ = f (2 * π / 3) ω φ → 
    f (-5 * π / 12) ω φ = √3 / 2 :=
by
  sorry

end value_of_f_neg_5π_over_12_l369_369785


namespace rational_square_initial_l369_369957

def sequence (x : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, x (n + 1) = 3 * (x n)^3 + x n

def initial_condition (a b : ℕ) (x : ℕ → ℚ) : Prop :=
  x 1 = a / b ∧ (3 ∣ b) = False

def is_square (r : ℚ) : Prop :=
  ∃ q : ℚ, r = q^2

theorem rational_square_initial
  (a b : ℕ)
  (x : ℕ → ℚ)
  (h_seq : sequence x)
  (h_initial : initial_condition a b x)
  (m : ℕ)
  (h_square_m : is_square (x m)) :
  is_square (x 1) :=
  sorry

end rational_square_initial_l369_369957


namespace domain_of_f_l369_369514

noncomputable def f (x : ℝ) := Real.sqrt (7 - Real.sqrt (9 - Real.sqrt (x + 4)))

theorem domain_of_f : {x : ℝ | x + 4 ≥ 0 ∧ x + 4 ≤ 81} = set.Icc (-4) 77 :=
by
  ext x
  simp
  split
  · intro h
    cases h with h1 h2
    split
    · linarith
    · linarith
  · intro h
    cases h with h1 h2
    split
    · linarith
    · linarith

end domain_of_f_l369_369514


namespace problem_statement_l369_369689

-- Definition of our function f
def f (x : ℝ) : ℝ := Real.sin (2 * x - (5 * Real.pi / 6))

-- Theorem to prove that f(-5π/12) = √3/2
theorem problem_statement : 
  f(-5 * Real.pi / 12) = sqrt 3 / 2 := 
sorry

end problem_statement_l369_369689


namespace inappropriate_for_map_measurement_l369_369948

-- Declaration of the problem statement in Lean
theorem inappropriate_for_map_measurement : 
  ¬ (unit_for_map_measurement "Beijing" "Shandong" = "meters") :=
sorry

end inappropriate_for_map_measurement_l369_369948


namespace regular_polygon_sides_l369_369481

theorem regular_polygon_sides (n : ℕ) (h₁ : ∀ k : ℕ, k = n → 180 * (k - 2) = 140 * k) : n = 9 := by
  have h₂ := h₁ n rfl
  linarith

end regular_polygon_sides_l369_369481


namespace choose_pairs_of_fruits_l369_369005

theorem choose_pairs_of_fruits (n : ℕ) (h : n ≥ 2) : (finset.card (finset.pairs (finset.range n))).choose 2 = n * (n - 1) / 2 :=
by 
  sorry

end choose_pairs_of_fruits_l369_369005


namespace sin_monotonic_increasing_f_symmetric_axes_find_value_l369_369699

def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi * 5 / 6)

theorem sin_monotonic_increasing (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) :
  Monotone (λ x, f x) (Ioo a b) := 
sorry

theorem f_symmetric_axes (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) 
  (x y : ℝ) (hx : x = a + (b - a) * k) (hy : y = a + (b - a) * (-k)) : 
  f x = f y :=
sorry

theorem find_value :
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
sorry

end sin_monotonic_increasing_f_symmetric_axes_find_value_l369_369699


namespace gcd_18_30_is_6_l369_369621

def gcd_18_30 : ℕ :=
  gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 :=
by {
  -- The step here will involve using properties of gcd and prime factorization,
  -- but we are given the result directly for the purpose of this task.
  sorry
}

end gcd_18_30_is_6_l369_369621


namespace intersect_complement_A_and_B_l369_369833

noncomputable def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x + 1 < 0}
def B : Set ℝ := {x | x - 3 < 0}

theorem intersect_complement_A_and_B : (Set.compl A ∩ B) = {x | -1 ≤ x ∧ x < 3} := by
  sorry

end intersect_complement_A_and_B_l369_369833


namespace recurrence_relation_p_series_l369_369905

noncomputable def p_series (n k : ℕ) : ℝ :=
if k < n then (p_series (n - 1) k - (1 / (2 : ℝ)^k) * p_series (n - k) k + (1 / (2 : ℝ)^k))
else 0

-- Statement of the theorem
theorem recurrence_relation_p_series (n k : ℕ) (h : k < n) :
  p_series n k = p_series (n - 1) k - (1 / (2 : ℝ)^k) * p_series (n - k) k + (1 / (2 : ℝ)^k) :=
sorry

end recurrence_relation_p_series_l369_369905


namespace necessary_but_not_sufficient_condition_l369_369056

variable (F₁ F₂ M : ℝ × ℝ) (a : ℝ)

def propositionA := abs (dist M F₁ - dist M F₂) = 2 * a

def propositionB :=
  let foci_dist := dist F₁ F₂
  let M₁ := M₁ x := (x, 0) -- assuming some point on the x-axis for simplification
  hyperbola_eqn := (λ (p : ℝ × ℝ), p ≠ M₁) -- assuming a placeholder for hyperbola
  hyperbola_eqn ⟨ M.x, M.y ⟩

theorem necessary_but_not_sufficient_condition :
  ∀ F₁ F₂ M a, propositionB F₁ F₂ M → propositionA F₁ F₂ M ∧ ¬(propositionA F₁ F₂ M → propositionB F₁ F₂ M) :=
by
  sorry

end necessary_but_not_sufficient_condition_l369_369056


namespace intersection_is_correct_l369_369080

open Set

variable {α : Type}

def A : Set ℝ := { x : ℝ | -3 ≤ x ∧ x < 4 }
def B : Set ℝ := { x : ℝ | -2 ≤ x ∧ x ≤ 5 }
def C : Set ℝ := { x : ℝ | -2 ≤ x ∧ x < 4 }

theorem intersection_is_correct : A ∩ B = C :=
by sorry

end intersection_is_correct_l369_369080


namespace trig_function_value_l369_369638

theorem trig_function_value : 
  (∀ x, f (Real.sin x) = Real.cos (2 * x) - 1) → f (Real.cos (Real.pi / 12)) = - Real.sqrt 3 / 2 - 1 :=
by
-- This is the mathematical statement.
sorry

end trig_function_value_l369_369638


namespace remainder_when_M_divided_by_32_l369_369150

-- Define M as the product of all odd primes less than 32.
def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_M_divided_by_32 :
    M % 32 = 1 := by
  -- We must prove that M % 32 = 1.
  sorry

end remainder_when_M_divided_by_32_l369_369150


namespace distance_between_A_and_B_l369_369980

-- Definitions of necessary conditions
def startTime : Nat := 8
def arrivalTime : Nat := 12
def speed : Nat := 8
def breakDuration : Nat := 1

-- Proof statement
theorem distance_between_A_and_B :
  let totalTravelTime := arrivalTime - startTime in
  let actualTravelTime := totalTravelTime - breakDuration in
  speed * actualTravelTime = 24 :=
by
  sorry

end distance_between_A_and_B_l369_369980


namespace find_f_of_neg_5_pi_over_12_l369_369736

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - (5 * π / 6))

theorem find_f_of_neg_5_pi_over_12 :
  (∀ x ∈ (set.Ioo (π / 6) (2 * π / 3)), (0 : ℝ) < (f x - f (x - 1))) ∧
  (∀ x, f (π / 6 - x) = f (π / 6 + x)) ∧
  (∀ x, f (2 * π / 3 - x) = f (2 * π / 3 + x)) →
  f (-5 * π / 12) = √3 / 2 :=
by
  sorry

end find_f_of_neg_5_pi_over_12_l369_369736


namespace math_problem_l369_369422

-- Definitions
def prop_A := ∀ x : ℝ, (fun x => x) = (fun t => t)
def prop_B := (∃ x : ℝ, ¬ (sqrt (x^2) = (sqrt x) ^ 2))
def prop_C := {x : ℝ | (x-2)/x > 0} = Set.Ioo (-∞) 0 ∪ Set.Ioo 2 ∞
def prop_D := ∀ x : ℝ, x ∈ {x : ℝ | sqrt (x + 2) + 1/(4 - x^2) ≠ 0} ↔ (x ∈ (-∞, -2) ∪ (-2, 2) ∪ (2, ∞))

-- Theorem
theorem math_problem : 
  (prop_A = True) ∧ 
  (prop_B = False) ∧ 
  (prop_C = True) ∧ 
  (prop_D = True) := by
  sorry

end math_problem_l369_369422


namespace least_addition_is_107_l369_369449

-- Define palindrome
def is_palindrome (n : ℕ) : Prop :=
  let str := n.toString in str = str.reverse

-- Given number
def given_number : ℕ := 58278

-- Least required number to form a palindrome
def least_palindrome_addition (n : ℕ) : ℕ :=
  if is_palindrome n then 0
  else
    let rec find_palindrome (k : ℕ) : ℕ :=
      if is_palindrome (n + k) then k
      else find_palindrome (k + 1)
    find_palindrome 1

theorem least_addition_is_107 : least_palindrome_addition given_number = 107 := by
  sorry

end least_addition_is_107_l369_369449


namespace sheila_earning_per_hour_l369_369434

theorem sheila_earning_per_hour :
  (252 / ((8 * 3) + (6 * 2)) = 7) := 
by
  -- Prove that sheila earns $7 per hour
  
  sorry

end sheila_earning_per_hour_l369_369434


namespace tan_sum_cos_diff_l369_369641

open Real

noncomputable def alpha : ℝ := sorry
noncomputable def beta : ℝ := sorry

-- Given conditions
axiom h_alpha_beta_pos_pi : alpha ∈ Ioo 0 π
axiom h_roots_eq : ∃ α β : ℝ, tan α = 3 ∧ tan β = 2

-- Question 1 proof statement
theorem tan_sum (h1 : tan alpha + tan beta = 5) (h2 : tan alpha * tan beta = 6) : 
  tan (alpha + beta) = -1 :=
sorry

-- Question 2 proof statement
theorem cos_diff (h1 : tan (alpha + beta) = -1) (h2: Ioo 0 π) (h3: ∃ α β : ℝ, tan α = 3 ∧ tan β = 2) : 
  cos (alpha - beta) =  7 * sqrt 2 / 10 :=
sorry

end tan_sum_cos_diff_l369_369641


namespace arithmetic_seq_diff_50th_term_l369_369501

theorem arithmetic_seq_diff_50th_term :
  ∃ (a : ℕ → ℚ) (L G : ℚ),
    (∀ n, 10 ≤ a n ∧ a n ≤ 100) ∧
    (∑ i in finset.range 200, a i) = 10000 ∧
    L = (101 * 10 + 4900) / 199 ∧
    G = (101 * 90 + 4900) / 199 ∧
    G - L = 8080 / 199 :=
by
  sorry

end arithmetic_seq_diff_50th_term_l369_369501


namespace new_cost_article_decreased_l369_369497

-- Define the actual cost of the article
def actual_cost : ℝ := 775

-- Define the percentage decrease
def decrease_percentage : ℝ := 20

-- Define the new cost calculation considering the decrease
def new_cost (actual : ℝ) (decrease : ℝ) : ℝ :=
  (1 - decrease / 100) * actual

-- The theorem to prove: with the given conditions, the new cost should be 620
theorem new_cost_article_decreased :
  new_cost actual_cost decrease_percentage = 620 :=
by
  sorry

end new_cost_article_decreased_l369_369497


namespace sin_monotonic_increasing_f_symmetric_axes_find_value_l369_369693

def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi * 5 / 6)

theorem sin_monotonic_increasing (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) :
  Monotone (λ x, f x) (Ioo a b) := 
sorry

theorem f_symmetric_axes (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) 
  (x y : ℝ) (hx : x = a + (b - a) * k) (hy : y = a + (b - a) * (-k)) : 
  f x = f y :=
sorry

theorem find_value :
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
sorry

end sin_monotonic_increasing_f_symmetric_axes_find_value_l369_369693


namespace prime_product_mod_32_l369_369295

open Nat

theorem prime_product_mod_32 : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := List.foldr (· * ·) 1 primes
  M % 32 = 25 :=
by
  sorry

end prime_product_mod_32_l369_369295


namespace product_of_odd_primes_mod_32_l369_369222

theorem product_of_odd_primes_mod_32 :
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31) in
  M % 32 = 25 :=
by
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31)
  sorry

end product_of_odd_primes_mod_32_l369_369222


namespace zero_of_f_when_m_is_neg1_monotonicity_of_f_m_gt_neg1_l369_369315

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  x - 1/x - 2 * m * Real.log x

theorem zero_of_f_when_m_is_neg1 : ∃ x > 0, f x (-1) = 0 :=
  by
    use 1
    sorry

theorem monotonicity_of_f_m_gt_neg1 (m : ℝ) (hm : m > -1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → f x m ≤ f y m) ∨
  (∃ a b : ℝ, 0 < a ∧ a < b ∧
    (∀ x : ℝ, 0 < x ∧ x < a → f x m ≤ f a m) ∧
    (∀ x : ℝ, a < x ∧ x < b → f a m ≥ f x m) ∧
    (∀ x : ℝ, b < x → f b m ≤ f x m)) :=
  by
    cases lt_or_le m 1 with
    | inl hlt =>
        left
        intros x y hx hy hxy
        sorry
    | inr hle =>
        right
        use m - Real.sqrt (m^2 - 1), m + Real.sqrt (m^2 - 1)
        sorry

end zero_of_f_when_m_is_neg1_monotonicity_of_f_m_gt_neg1_l369_369315


namespace valid_paths_from_A_to_B_l369_369994

-- Define the grid size and the blocked points
structure Grid :=
  (east_bound : Nat) 
  (south_bound : Nat)
  (blocked_points : List (Nat × Nat))

-- Define the function to calculate the valid paths
def valid_paths : Grid → Nat
  | ⟨5, 3, [(2, 2), (3, 2)]⟩ := 39
  | _ := 0  -- Other scenarios, not specified here, are not our concern

-- The theorem statement
theorem valid_paths_from_A_to_B (g : Grid) (h : g = ⟨5, 3, [(2, 2), (3, 2)]⟩) : 
  valid_paths g = 39 :=
by 
  rw [h]
  exact rfl

end valid_paths_from_A_to_B_l369_369994


namespace gcd_of_18_and_30_l369_369597

-- Define the numbers
def num1 := 18
def num2 := 30

-- State the GCD property
theorem gcd_of_18_and_30 : Nat.gcd num1 num2 = 6 :=
by
  sorry

end gcd_of_18_and_30_l369_369597


namespace angle_between_two_u_neg_v_l369_369668

variables (u v : ℝ^3)
variable (angle_uv : ℝ)
variable (angle_2u_negv : ℝ)

-- Condition: The angle between the non-coplanar 3D vectors u and v is 30 degrees.
def angle_between_vectors (a b : ℝ^3) : ℝ := real.acos ((a • b) / (∥a∥ * ∥b∥))

axiom angle_between_u_v : angle_between_vectors u v = real.pi / 6

-- Question: What is the angle between the vectors 2u and -v?
def two_u : ℝ^3 := 2 • u
def neg_v : ℝ^3 := -v

-- Prove that this angle is 150 degrees
theorem angle_between_two_u_neg_v : angle_between_vectors two_u neg_v = 5 * real.pi / 6 :=
sorry

end angle_between_two_u_neg_v_l369_369668


namespace problem_statement_l369_369744

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - 5 * Real.pi / 6)

theorem problem_statement :
  (∀ (x : ℝ), (∀ a b : ℝ, a < b → x ∈ (Set.Ioc (Real.pi / 6) (2 * Real.pi / 3)) → f(x) < f(b)) → x = (Real.pi / 6) ∨ x = (2 * Real.pi / 3)) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by sorry

end problem_statement_l369_369744


namespace gcd_of_18_and_30_l369_369564

-- Define the numbers
def a := 18
def b := 30

-- The main theorem statement
theorem gcd_of_18_and_30 : Nat.gcd a b = 6 := by
  sorry

end gcd_of_18_and_30_l369_369564


namespace problem_statement_l369_369682

-- Definition of our function f
def f (x : ℝ) : ℝ := Real.sin (2 * x - (5 * Real.pi / 6))

-- Theorem to prove that f(-5π/12) = √3/2
theorem problem_statement : 
  f(-5 * Real.pi / 12) = sqrt 3 / 2 := 
sorry

end problem_statement_l369_369682


namespace max_xy_value_l369_369887

theorem max_xy_value (x y : ℕ) (h : 7 * x + 4 * y = 140) : xy ≤ 168 :=
by sorry

end max_xy_value_l369_369887


namespace product_mod_32_is_15_l369_369242

-- Define the odd primes less than 32
def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the product of odd primes
def M : ℕ := List.foldl (· * ·) 1 oddPrimes

-- The theorem to prove the remainder when M is divided by 32 is 15
theorem product_mod_32_is_15 : M % 32 = 15 := by
  sorry

end product_mod_32_is_15_l369_369242


namespace gcd_of_18_and_30_l369_369557

-- Define the numbers
def a := 18
def b := 30

-- The main theorem statement
theorem gcd_of_18_and_30 : Nat.gcd a b = 6 := by
  sorry

end gcd_of_18_and_30_l369_369557


namespace smallest_initial_number_sum_of_digits_l369_369928

def BernardoOperation (x : ℕ) : ℕ := 2 * x
def SilviaOperation (x : ℕ) : ℕ := x + 100

theorem smallest_initial_number_sum_of_digits :
  ∃ N : ℕ, N = 163 ∧ Nat.digits 10 N.sum = 10 :=
by
  use 163
  split
  -- proof that N = 163 produces a win for Bernardo with sum of digits 10
  sorry

end smallest_initial_number_sum_of_digits_l369_369928


namespace find_fx_value_l369_369721

noncomputable def f (x : Real) : Real :=
  sin (2 * x - 5 * Real.pi / 6)

theorem find_fx_value :
  (∀ x, f x = sin (2 * x - 5 * Real.pi / 6)) ∧ 
  (f (Real.pi / 6) = f (2 * Real.pi / 3)) ∧ 
  (∀ x1 x2, (Real.pi / 6 < x1 ∧ x1 < 2 * Real.pi / 3 ∧ x1 < x2 ∧ x2 < 2 * Real.pi / 3) -> f x1 < f x2) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 := by
  sorry

end find_fx_value_l369_369721


namespace gcd_18_30_l369_369587

theorem gcd_18_30: Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l369_369587


namespace obtuse_triangle_from_groups_l369_369500

-- Define the groups of numbers
def group_A : (ℕ × ℕ × ℕ) := (3, 4, 5)
def group_B : (ℕ × ℕ × ℕ) := (3, 3, 5)
def group_C : (ℕ × ℕ × ℕ) := (4, 4, 5)
def group_D : (ℕ × ℕ × ℕ) := (3, 4, 4)

-- Define a function to check if a triangle is obtuse
def is_obtuse (a b c : ℕ) : Prop :=
  a + b > c ∧ a^2 + b^2 < c^2

theorem obtuse_triangle_from_groups :
  is_obtuse group_A.1 group_A.2 group_A.3 = false ∧
  is_obtuse group_B.1 group_B.2 group_B.3 = true ∧
  is_obtuse group_C.1 group_C.2 group_C.3 = false ∧
  is_obtuse group_D.1 group_D.2 group_D.3 = false :=
by sorry

end obtuse_triangle_from_groups_l369_369500


namespace find_f_neg_5pi_12_l369_369775

-- Conditions
def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

variables {ω φ : ℝ}
def f (x : ℝ) : ℝ := Real.sin (ω * x + φ)

axiom monotonicity_condition : is_monotonically_increasing f (π / 6) (2 * π / 3)
axiom symmetry_condition_1 : f (π / 6) = f (2 * π / 3)
axiom symmetry_condition_2 : f (-π / 6) = f (π / 2)

-- Goal
theorem find_f_neg_5pi_12 : f (- 5 * π / 12) = sqrt 3 / 2 := 
sorry

end find_f_neg_5pi_12_l369_369775


namespace find_f_value_l369_369821

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

theorem find_f_value :
  f ( -5 * Real.pi / 12 ) = Real.sqrt 3 / 2 :=
by
  sorry

end find_f_value_l369_369821


namespace remainder_of_product_of_odd_primes_mod_32_l369_369289

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem remainder_of_product_of_odd_primes_mod_32 :
  (product odd_primes_less_than_32) % 32 = 23 :=
by sorry

end remainder_of_product_of_odd_primes_mod_32_l369_369289


namespace average_bowling_score_l369_369088

theorem average_bowling_score 
    (gretchen_score : ℕ) (mitzi_score : ℕ) (beth_score : ℕ)
    (gretchen_eq : gretchen_score = 120)
    (mitzi_eq : mitzi_score = 113)
    (beth_eq : beth_score = 85) :
    (gretchen_score + mitzi_score + beth_score) / 3 = 106 := 
by
  sorry

end average_bowling_score_l369_369088


namespace range_of_a_l369_369659

def proposition_p (a : ℝ) := a > 1
def proposition_q (a : ℝ) := ∀ x : ℝ, -x^2 + 2*x - 2 ≤ a

theorem range_of_a :
  (∃ a : ℝ, (proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a)) →
  (∃ a : Set.Icc (-1:ℝ) 1, true) :=
begin
  sorry
end

end range_of_a_l369_369659


namespace union_card_ge_165_l369_369135

open Finset

variable (A : Finset ℕ) (A_i : Fin (11) → Finset ℕ)
variable (hA : A.card = 225)
variable (hA_i_card : ∀ i, (A_i i).card = 45)
variable (hA_i_intersect : ∀ i j, i < j → ((A_i i) ∩ (A_i j)).card = 9)

theorem union_card_ge_165 : (Finset.biUnion Finset.univ A_i).card ≥ 165 := by sorry

end union_card_ge_165_l369_369135


namespace math_problem_l369_369119

-- Definitions for the curve C1
def C1_polar_equation (θ : ℝ) : ℝ :=
  2 * sqrt 2 * sin (θ + (π / 4))

-- Definitions for the line L parametric equations
def L_parametric_x (t : ℝ) : ℝ :=
  1 - (sqrt 2 / 2) * t

def L_parametric_y (t : ℝ) : ℝ :=
  (sqrt 2 / 2) * t

-- Definitions for the curve C2 parametric equations
def C2_parametric_x (α : ℝ) : ℝ :=
  3 + sqrt 2 * cos α

def C2_parametric_y (α : ℝ) : ℝ :=
  4 + sqrt 2 * sin α

-- Targets to prove
def rectangular_equation_C1 : Prop :=
  ∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2 ↔ (∃ (θ : ℝ), x = C1_polar_equation θ * cos θ ∧ y = C1_polar_equation θ * sin θ)

def rectangular_equation_L : Prop :=
  ∀ (x y t : ℝ), x + y - 1 = 0 ↔ (x = L_parametric_x t ∧ y = L_parametric_y t)

def min_area_PAB : Prop :=
  ∃ (α : ℝ) (P : ℝ × ℝ), P = (C2_parametric_x α, C2_parametric_y α) → ∀ (A B : ℝ × ℝ), on C1_polar_equation ∧ on L_parametric_x y x, 2 * sqrt 3 ≤ area_of_triangle P A B ∧ area_of_triangle P A B ≤ 4*sqrt 3

theorem math_problem :
  rectangular_equation_C1 ∧ rectangular_equation_L ∧ min_area_PAB :=
by
  sorry

end math_problem_l369_369119


namespace mark_works_hours_per_day_l369_369992

variable (old_wage : ℕ) (raise_percent : ℝ) (days_per_week : ℕ) (old_bills weekly_new_bills leftover : ℕ)
variable (new_wage : ℤ) (total_earning_with_leftover : ℕ)

def wage_after_raise (old_wage : ℕ) (raise_percent : ℝ) := 
  (old_wage : ℝ) + old_wage * (raise_percent / 100.0)

def weekly_earning (new_wage : ℤ) (H : ℕ) (days_per_week : ℕ) := 
  new_wage * (H : ℤ) * (days_per_week : ℤ)

theorem mark_works_hours_per_day
    (old_wage = 40)
    (raise_percent = 5)
    (days_per_week = 5)
    (old_bills = 600)
    (weekly_new_bills = 700)
    (leftover = 980)
    (new_wage = 42)
    (total_earning_with_leftover = 1680) :
    (H : ℕ), (weekly_earning new_wage H days_per_week) = total_earning_with_leftover  :=
by
  sorry

end mark_works_hours_per_day_l369_369992


namespace angle_between_vectors_is_120_degrees_l369_369666

variables (a b : EuclideanSpace ℝ (Fin 3))
          (θ : Real)

-- Given conditions
axiom norm_a : ‖a‖ = Real.sqrt 3
axiom norm_b : ‖b‖ = 2 * Real.sqrt 3
axiom dot_product_ab : inner a b = -3

-- Statement: Prove the angle between a and b is 120 degrees
theorem angle_between_vectors_is_120_degrees
  (h : ∀ θ, inner a b = ‖a‖ * ‖b‖ * Real.cos θ) : 
  θ = 2 * Real.pi / 3 :=
sorry

end angle_between_vectors_is_120_degrees_l369_369666


namespace polygon_sides_sum_720_l369_369380

theorem polygon_sides_sum_720 (n : ℕ) (h1 : (n - 2) * 180 = 720) : n = 6 := by
  sorry

end polygon_sides_sum_720_l369_369380


namespace three_tetrahedra_in_cube_l369_369926

-- Define the edge length of the cube
def cube_edge_length := 1

-- Define the edge length of the tetrahedra
def tetrahedron_edge_length := 1

-- Define the existence of three non-overlapping regular tetrahedra within a cube
theorem three_tetrahedra_in_cube:
  ∃ (tetra1 tetra2 tetra3: {t : set ℝ×ℝ×ℝ | regular_tetrahedron t ∧ edge_length t = tetrahedron_edge_length}),
    (tetra1 ∩ tetra2 = ∅) ∧ (tetra2 ∩ tetra3 = ∅) ∧ (tetra1 ∩ tetra3 = ∅) ∧ 
    (cube_edge_length = 1) ∧ (∀ t ∈ {tetra1, tetra2, tetra3}, ⊆ cube_edge_length) :=
sorry

end three_tetrahedra_in_cube_l369_369926


namespace range_of_omega_for_exactly_three_zeros_in_interval_l369_369069

noncomputable def f (ω x : ℝ) : ℝ := sin(ω * x) - 1

theorem range_of_omega_for_exactly_three_zeros_in_interval
  (ω : ℝ)
  (h_pos : ω > 0)
  (h_zeros : ∀ x ∈ set.Icc (set.Icc 0 (2 * real.pi)).val, f ω x = 0 → x = ω * real.pi / 2 ∨ x = ω * 5 * real.pi / 2 ∨ x = ω * 9 * real.pi / 2):
  (9 / 4) ≤ ω ∧ ω < (13 / 4) :=
sorry

end range_of_omega_for_exactly_three_zeros_in_interval_l369_369069


namespace product_of_odd_primes_mod_32_l369_369166

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def M : ℕ := odd_primes_less_than_32.foldr (· * ·) 1

theorem product_of_odd_primes_mod_32 : M % 32 = 17 :=
by
  -- the proof goes here
  sorry

end product_of_odd_primes_mod_32_l369_369166


namespace find_f_value_l369_369814

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

theorem find_f_value :
  f ( -5 * Real.pi / 12 ) = Real.sqrt 3 / 2 :=
by
  sorry

end find_f_value_l369_369814


namespace pentagon_perimeter_sum_l369_369466

def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def perimeter (points : List (ℝ × ℝ)) : ℝ :=
  (points.zip (points.tail ++ [points.head])).sum (λ ⟨p1, p2⟩, dist p1 p2)

def perimeter_in_form (perimeter : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let a := 4
  let b := 1
  let c := 0
  let d := 2
  (a, b, c, d)

def sum_components (components : ℝ × ℝ × ℝ × ℝ) : ℝ :=
  components.1 + components.2 + components.3 + components.4

theorem pentagon_perimeter_sum :
  let points := [(0,0), (2,0), (3,2), (2,3), (0,2)]
  let perim_form := perimeter_in_form (perimeter points)
  sum_components perim_form = 7 :=
by
  intros
  sorry

end pentagon_perimeter_sum_l369_369466


namespace find_x0_l369_369676

def f (x : ℝ) : ℝ :=
  if x < 1 then x^2 - 2*x - 2 else 2*x - 3

theorem find_x0 (x₀ : ℝ) : f x₀ = 1 ↔ x₀ = -1 ∨ x₀ = 2 := by
  sorry

end find_x0_l369_369676


namespace product_mod_32_is_15_l369_369244

-- Define the odd primes less than 32
def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the product of odd primes
def M : ℕ := List.foldl (· * ·) 1 oddPrimes

-- The theorem to prove the remainder when M is divided by 32 is 15
theorem product_mod_32_is_15 : M % 32 = 15 := by
  sorry

end product_mod_32_is_15_l369_369244


namespace original_example_intended_l369_369110

theorem original_example_intended (x : ℝ) : (3 * x - 4 = x / 3 + 4) → x = 3 :=
by
  sorry

end original_example_intended_l369_369110


namespace no_triangle_tangent_l369_369084

open Real

/-- Given conditions --/
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def C2 (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1 ∧ a > b ∧ b > 0 ∧ (1 / a^2) + (1 / b^2) = 1

theorem no_triangle_tangent (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : (1 : ℝ) / a^2 + 1 / b^2 = 1) :
  ¬∃ (A B C : ℝ × ℝ), 
    (C1 A.1 A.2) ∧ (C1 B.1 B.2) ∧ (C1 C.1 C.2) ∧
    (∃ (l : ℝ) (m : ℝ) (n : ℝ), C2 l m a b ∧ C2 n l a b) :=
by
  sorry

end no_triangle_tangent_l369_369084


namespace product_of_sequence_l369_369020

theorem product_of_sequence : 
  (1 / 2) * (4 / 1) * (1 / 8) * (16 / 1) * (1 / 32) * (64 / 1) * (1 / 128) * (256 / 1) * (1 / 512) * (1024 / 1) * (1 / 2048) * (4096 / 1) = 64 := 
by
  sorry

end product_of_sequence_l369_369020


namespace product_of_odd_primes_mod_32_l369_369191

theorem product_of_odd_primes_mod_32 :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  M % 32 = 17 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  exact sorry

end product_of_odd_primes_mod_32_l369_369191


namespace product_of_odd_primes_mod_32_l369_369179

theorem product_of_odd_primes_mod_32 :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  M % 32 = 17 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  exact sorry

end product_of_odd_primes_mod_32_l369_369179


namespace product_of_odd_primes_mod_32_l369_369171

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def M : ℕ := odd_primes_less_than_32.foldr (· * ·) 1

theorem product_of_odd_primes_mod_32 : M % 32 = 17 :=
by
  -- the proof goes here
  sorry

end product_of_odd_primes_mod_32_l369_369171


namespace walking_speed_proof_l369_369465

-- Definitions based on the problem's conditions
def rest_time_per_period : ℕ := 5
def distance_per_rest : ℕ := 10
def total_distance : ℕ := 50
def total_time : ℕ := 320

-- The man's walking speed
def walking_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

-- The main statement to be proved
theorem walking_speed_proof : 
  walking_speed total_distance ((total_time - ((total_distance / distance_per_rest) * rest_time_per_period)) / 60) = 10 := 
by
  sorry

end walking_speed_proof_l369_369465


namespace suff_not_necessary_condition_l369_369637

def P : Set ℝ := { x | x ^ 2 - 4 * x + 3 ≤ 0 }
def Q : Set ℝ := { x | ∃ y, y = sqrt (x + 1) + sqrt (3 - x) }

theorem suff_not_necessary_condition (x : ℝ) :
  (x ∈ P → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P) :=
by
  -- Proving the necessary and sufficient condition without going into the solution steps
  sorry

end suff_not_necessary_condition_l369_369637


namespace max_xy_l369_369862

theorem max_xy (x y : ℕ) (h1: 7 * x + 4 * y = 140) : ∃ x y, 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by {
  sorry
}

end max_xy_l369_369862


namespace limit_sqrt_sin_exp_l369_369436

noncomputable def limitExpression (x : ℝ) : ℝ := 
  (sqrt (1 + x * sin x) - 1) / (exp (x ^ 2) - 1)

theorem limit_sqrt_sin_exp :
  filter.tendsto limitExpression (nhds 0) (nhds (1 / 2)) :=
sorry

end limit_sqrt_sin_exp_l369_369436


namespace product_of_odd_primes_mod_32_l369_369232

theorem product_of_odd_primes_mod_32 :
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31) in
  M % 32 = 25 :=
by
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31)
  sorry

end product_of_odd_primes_mod_32_l369_369232


namespace problem_statement_l369_369754

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - 5 * Real.pi / 6)

theorem problem_statement :
  (∀ (x : ℝ), (∀ a b : ℝ, a < b → x ∈ (Set.Ioc (Real.pi / 6) (2 * Real.pi / 3)) → f(x) < f(b)) → x = (Real.pi / 6) ∨ x = (2 * Real.pi / 3)) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by sorry

end problem_statement_l369_369754


namespace integer_between_sqrt2_and_sqrt17_l369_369424

theorem integer_between_sqrt2_and_sqrt17 : ∃ (x : ℤ), (real.sqrt 2 < x) ∧ (x < real.sqrt 17) ∧ (x = 3) :=
by
  sorry

end integer_between_sqrt2_and_sqrt17_l369_369424


namespace max_expr_value_l369_369960

-- Definition of the right triangle sides using Euler's formula for primitive Pythagorean triples
def a (m n : ℤ) := m^2 - n^2
def b (m n : ℤ) := 2 * m * n
def c (m n : ℤ) := m^2 + n^2

-- Definition of perimeter and area of the right triangle
def P (m n : ℤ) := a m n + b m n + c m n
def A (m n : ℤ) := (1 / 2) * a m n * b m n

-- Definition of the expression we want to maximize
def expr (m n : ℤ) := (P m n)^2 / A m n

-- The theorem statement: given the conditions on m and n, the maximum value of expr m n is 24
theorem max_expr_value (m n : ℤ) (h1 : m > n) (h2 : n > 0) (h3 : Nat.Coprime m n) (h4 : ¬(Nat.Odd m ∧ Nat.Odd n)) :
  ∃ m n, m > n ∧ n > 0 ∧ Nat.Coprime m n ∧ ¬(Nat.Odd m ∧ Nat.Odd n) ∧ expr m n = 24 :=
by
  sorry

end max_expr_value_l369_369960


namespace remainder_when_M_divided_by_32_l369_369149

-- Define M as the product of all odd primes less than 32.
def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_M_divided_by_32 :
    M % 32 = 1 := by
  -- We must prove that M % 32 = 1.
  sorry

end remainder_when_M_divided_by_32_l369_369149


namespace gcd_18_30_l369_369571

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := 
by
  sorry

end gcd_18_30_l369_369571


namespace gcd_of_18_and_30_l369_369589

-- Define the numbers
def num1 := 18
def num2 := 30

-- State the GCD property
theorem gcd_of_18_and_30 : Nat.gcd num1 num2 = 6 :=
by
  sorry

end gcd_of_18_and_30_l369_369589


namespace x_squared_plus_inverse_squared_l369_369852

theorem x_squared_plus_inverse_squared (x : ℝ) (h : x + 1/x = 5) : x^2 + 1/x^2 = 23 :=
begin
  sorry
end

end x_squared_plus_inverse_squared_l369_369852


namespace find_f_value_l369_369811

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

theorem find_f_value :
  f ( -5 * Real.pi / 12 ) = Real.sqrt 3 / 2 :=
by
  sorry

end find_f_value_l369_369811


namespace final_percentage_of_alcohol_l369_369457

theorem final_percentage_of_alcohol (initial_volume : ℝ) (initial_alcohol_percentage : ℝ)
  (removed_alcohol : ℝ) (added_water : ℝ) :
  initial_volume = 15 → initial_alcohol_percentage = 25 →
  removed_alcohol = 2 → added_water = 3 →
  ( ( (initial_alcohol_percentage / 100 * initial_volume - removed_alcohol) / 
    (initial_volume - removed_alcohol + added_water) ) * 100 = 10.9375) :=
by
  intros
  sorry

end final_percentage_of_alcohol_l369_369457


namespace area_ratio_bisector_l369_369502

namespace Tetrahedron

variables (P A B C D : Type) 
variables (PAD BPA CPA : Set) -- Representing planes
variables [IsTetrahedron P A B C PAD BPA CPA] -- Custom type class to indicate geometric properties

/-- Given a tetrahedron P-ABC, where PAD is the bisecting plane of the dihedral angle B-PA-C and intersects BC at D, 
we need to prove that the ratio of areas S_triangle BDP and S_triangle CDP equals the ratio of areas S_triangle BPA and S_triangle CPA. -/
theorem area_ratio_bisector (tetrahedron : IsTetrahedron P A B C PAD BPA CPA)
  (bisecting_plane : IsBisectingPlane PAD (DihedralAngle B P A C))
  (intersection : IntersectsAt PAD (LineSegment B C) D) :
  (area (Triangle B D P)) / (area (Triangle C D P)) = (area (Triangle B P A)) / (area (Triangle C P A)) :=
sorry

end Tetrahedron

end area_ratio_bisector_l369_369502


namespace similarity_of_triangles_l369_369966

open EuclideanGeometry

variable {Ω : Type*} [MetricSpace Ω] [InnerProductSpace ℝ Ω] [Nonempty Ω]

theorem similarity_of_triangles
  (Γ₁ Γ₂ : Circle Ω)    -- Two circles
  (O₁ O₂ : Ω)          -- Centers of the circles Γ₁ and Γ₂
  (X Y A B: Ω)         -- Points of intersection and additional points on the circles
  (hΓ₁ : Γ₁.center = O₁)
  (hΓ₂ : Γ₂.center = O₂)
  (hX : X ∈ Γ₁ ∧ X ∈ Γ₂)
  (hY : Y ∈ Γ₁ ∧ Y ∈ Γ₂)
  (hAY : LineThrough Y)
  (hAΓ₁ : A ∈ Γ₁)
  (hBΓ₂ : B ∈ Γ₂) :
  ∠ X O₁ O₂ = ∠ X A B → 
  Triangle.similar ⟨X, O₁, O₂⟩ ⟨X, A, B⟩ :=
by
  sorry

end similarity_of_triangles_l369_369966


namespace solve_quadratic_eq_solve_cubic_eq_l369_369446

-- Problem 1: Solve (x-1)^2 = 9
theorem solve_quadratic_eq (x : ℝ) (h : (x - 1) ^ 2 = 9) : x = 4 ∨ x = -2 := 
by 
  sorry

-- Problem 2: Solve (x+3)^3 / 3 - 9 = 0
theorem solve_cubic_eq (x : ℝ) (h : (x + 3) ^ 3 / 3 - 9 = 0) : x = 0 := 
by 
  sorry

end solve_quadratic_eq_solve_cubic_eq_l369_369446


namespace propane_tank_and_burner_cost_l369_369838

theorem propane_tank_and_burner_cost
(Total_money: ℝ)
(Sheet_cost: ℝ)
(Rope_cost: ℝ)
(Helium_cost_per_oz: ℝ)
(Lift_per_oz: ℝ)
(Max_height: ℝ)
(ht: Total_money = 200)
(hs: Sheet_cost = 42)
(hr: Rope_cost = 18)
(hh: Helium_cost_per_oz = 1.50)
(hlo: Lift_per_oz = 113)
(hm: Max_height = 9492)
:
(Total_money - (Sheet_cost + Rope_cost) 
 - (Max_height / Lift_per_oz * Helium_cost_per_oz) 
 = 14) :=
by
  sorry

end propane_tank_and_burner_cost_l369_369838


namespace clea_escalator_time_l369_369949

theorem clea_escalator_time (x y k : ℕ) (h1 : 90 * x = y) (h2 : 30 * (x + k) = y) :
  (y / k) = 45 := by
  sorry

end clea_escalator_time_l369_369949


namespace gcd_of_18_and_30_l369_369553

theorem gcd_of_18_and_30 : Nat.gcd 18 30 = 6 :=
by
  sorry

end gcd_of_18_and_30_l369_369553


namespace gcd_18_30_is_6_l369_369611

def gcd_18_30 : ℕ :=
  gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 :=
by {
  -- The step here will involve using properties of gcd and prime factorization,
  -- but we are given the result directly for the purpose of this task.
  sorry
}

end gcd_18_30_is_6_l369_369611


namespace value_of_f_neg_5π_over_12_l369_369789

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem value_of_f_neg_5π_over_12 :
  ∀ (ω φ : ℝ), 
    (∀ x y : ℝ, (x < y ∧ x ∈ Ioo (π / 6) (2 * π / 3) ∧ y ∈ Ioo (π / 6) (2 * π / 3)) → f x ω φ < f y ω φ) ∧ 
    f (π / 6) ω φ = f (2 * π / 3) ω φ → 
    f (-5 * π / 12) ω φ = √3 / 2 :=
by
  sorry

end value_of_f_neg_5π_over_12_l369_369789


namespace log_computable_in_range_l369_369047

def possible_log_computations : set ℕ := {1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100}

def log_computable (n : ℕ) : Prop :=
  n ∈ possible_log_computations

theorem log_computable_in_range (n : ℕ) (h₁ : 1 ≤ n ∧ n ≤ 100) : log_computable n :=
by sorry

end log_computable_in_range_l369_369047


namespace recurrence_relation_l369_369910

-- Define the function p_nk and prove the recurrence relation
def p (n k : ℕ) : ℝ := sorry

theorem recurrence_relation (n k : ℕ) (h : k < n) : 
  p n k = p (n-1) k - (1 / 2^k) * p (n-k) k + (1 / 2^k) :=
sorry

end recurrence_relation_l369_369910


namespace base_seven_sum_of_product_l369_369508

def base_seven_to_decimal (d1 d0 : ℕ) : ℕ :=
  7 * d1 + d0

def decimal_to_base_seven (n : ℕ) : ℕ × ℕ × ℕ × ℕ :=
  let d3 := n / (7 ^ 3)
  let r3 := n % (7 ^ 3)
  let d2 := r3 / (7 ^ 2)
  let r2 := r3 % (7 ^ 2)
  let d1 := r2 / 7
  let d0 := r2 % 7
  (d3, d2, d1, d0)

def sum_of_base_seven_digits (d3 d2 d1 d0 : ℕ) : ℕ :=
  d3 + d2 + d1 + d0

theorem base_seven_sum_of_product :
  let n1 := base_seven_to_decimal 3 5
  let n2 := base_seven_to_decimal 4 2
  let product := n1 * n2
  let (d3, d2, d1, d0) := decimal_to_base_seven product
  sum_of_base_seven_digits d3 d2 d1 d0 = 18 :=
  by
    sorry

end base_seven_sum_of_product_l369_369508


namespace remainder_of_product_of_odd_primes_mod_32_l369_369284

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem remainder_of_product_of_odd_primes_mod_32 :
  (product odd_primes_less_than_32) % 32 = 23 :=
by sorry

end remainder_of_product_of_odd_primes_mod_32_l369_369284


namespace product_mod_32_is_15_l369_369247

-- Define the odd primes less than 32
def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the product of odd primes
def M : ℕ := List.foldl (· * ·) 1 oddPrimes

-- The theorem to prove the remainder when M is divided by 32 is 15
theorem product_mod_32_is_15 : M % 32 = 15 := by
  sorry

end product_mod_32_is_15_l369_369247


namespace max_product_l369_369877

theorem max_product (x y : ℕ) (h1 : 7 * x + 4 * y = 140) : x * y ≤ 168 :=
sorry

end max_product_l369_369877


namespace closest_integer_sum_logarithms_l369_369028

theorem closest_integer_sum_logarithms (S : ℝ) : 
  is_integer_close (S) (141) ↔
  S = ∑ d in (proper_divisors 1000000), (log 10 d) :=
begin
  sorry
end

end closest_integer_sum_logarithms_l369_369028


namespace product_mod_32_is_15_l369_369239

-- Define the odd primes less than 32
def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the product of odd primes
def M : ℕ := List.foldl (· * ·) 1 oddPrimes

-- The theorem to prove the remainder when M is divided by 32 is 15
theorem product_mod_32_is_15 : M % 32 = 15 := by
  sorry

end product_mod_32_is_15_l369_369239


namespace find_pb_l369_369063

variables (a b : Prop)

-- Known probabilities
def p : Prop → ℝ := sorry -- define the probability function

-- Given conditions
axiom h₁ : p(a) = 2 / 5
axiom h₂ : p(a ∩ b) = 0.16
axiom h_independent : (p(a ∩ b) = p(a) * p(b))

-- The goal
theorem find_pb : p(b) = 0.4 :=
by sorry

end find_pb_l369_369063


namespace find_f_value_l369_369817

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

theorem find_f_value :
  f ( -5 * Real.pi / 12 ) = Real.sqrt 3 / 2 :=
by
  sorry

end find_f_value_l369_369817


namespace remainder_when_divided_by_32_l369_369153

def product_of_odds : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_divided_by_32 : product_of_odds % 32 = 9 := by
  sorry

end remainder_when_divided_by_32_l369_369153


namespace initial_cupcakes_l369_369342

variable (x : ℕ) -- Define x as the number of cupcakes Robin initially made

-- Define the conditions provided in the problem
def cupcakes_sold := 22
def cupcakes_made := 39
def final_cupcakes := 59

-- Formalize the problem statement: Prove that given the conditions, the initial cupcakes equals 42
theorem initial_cupcakes:
  x - cupcakes_sold + cupcakes_made = final_cupcakes → x = 42 := 
by
  -- Placeholder for the proof
  sorry

end initial_cupcakes_l369_369342


namespace prime_product_mod_32_l369_369293

open Nat

theorem prime_product_mod_32 : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := List.foldr (· * ·) 1 primes
  M % 32 = 25 :=
by
  sorry

end prime_product_mod_32_l369_369293


namespace total_rent_paid_l369_369321

theorem total_rent_paid
  (weekly_rent : ℕ) (num_weeks : ℕ) 
  (hrent : weekly_rent = 388)
  (hweeks : num_weeks = 1359) :
  weekly_rent * num_weeks = 527292 := 
by
  sorry

end total_rent_paid_l369_369321


namespace train_speed_is_54_kmh_l369_369490

noncomputable def train_length_m : ℝ := 285
noncomputable def train_length_km : ℝ := train_length_m / 1000
noncomputable def time_seconds : ℝ := 19
noncomputable def time_hours : ℝ := time_seconds / 3600
noncomputable def speed : ℝ := train_length_km / time_hours

theorem train_speed_is_54_kmh :
  speed = 54 := by
sorry

end train_speed_is_54_kmh_l369_369490


namespace remainder_of_M_when_divided_by_32_l369_369206

open Nat

def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
def M : ℕ := List.foldl (*) 1 oddPrimes

theorem remainder_of_M_when_divided_by_32 :
  M % 32 = 1 :=
sorry

end remainder_of_M_when_divided_by_32_l369_369206


namespace least_whole_number_sub_l369_369418

-- Definitions from conditions
def original_ratio_num : ℕ := 6
def original_ratio_denom : ℕ := 7
def target_ratio_num : ℕ := 16
def target_ratio_denom : ℕ := 21

-- The proof problem
theorem least_whole_number_sub :
  ∃ (x : ℕ), (original_ratio_num - x) / (original_ratio_denom - x) < target_ratio_num / target_ratio_denom ∧
              ∀ (y : ℕ), 
                (original_ratio_num - y) / (original_ratio_denom - y) < target_ratio_num / target_ratio_denom → 
                x ≤ y :=
begin
  sorry
end

end least_whole_number_sub_l369_369418


namespace trig_identity_l369_369046

theorem trig_identity (α : ℝ) (h : sin (2 * α) - 2 = 2 * cos (2 * α)) :
  sin α ^ 2 + sin (2 * α) = 1 ∨ sin α ^ 2 + sin (2 * α) = 8 / 5 :=
by
  sorry

end trig_identity_l369_369046


namespace probZ_eq_1_4_l369_369495

noncomputable def probX : ℚ := 1/4
noncomputable def probY : ℚ := 1/3
noncomputable def probW : ℚ := 1/6

theorem probZ_eq_1_4 :
  let probZ : ℚ := 1 - (probX + probY + probW)
  probZ = 1/4 :=
by
  sorry

end probZ_eq_1_4_l369_369495


namespace find_customers_l369_369006

def original_price : ℝ := 600
def discount_rate_1 : ℝ := 0.20
def discount_rate_2 : ℝ := 0.10
def customers_1 : ℚ := 15

-- The discounted costs
def cost_1 (p0 : ℝ) (d : ℝ) : ℝ := p0 * (1 - d)
def cost_2 (p0 : ℝ) (d : ℝ) : ℝ := p0 * (1 - d)

-- Prove the number of customers when discount is reduced to 10%
theorem find_customers :
  let c1 := cost_1 original_price discount_rate_1 in
  let c2 := cost_2 original_price discount_rate_2 in
  let k := customers_1 * c1 in
  let customers_2 := k / c2 in
  customers_2 ≈ 13 :=
by
  sorry

end find_customers_l369_369006


namespace find_constant_c_l369_369915

theorem find_constant_c : ∃ (c : ℝ), (∀ n : ℤ, c * (n:ℝ)^2 ≤ 3600) ∧ (∀ n : ℤ, n ≤ 5) ∧ (c = 144) :=
by
  sorry

end find_constant_c_l369_369915


namespace maximum_xy_value_l369_369867

theorem maximum_xy_value :
  ∃ (x y : ℕ), 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by
  sorry

end maximum_xy_value_l369_369867


namespace remainder_of_product_of_odd_primes_mod_32_l369_369287

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem remainder_of_product_of_odd_primes_mod_32 :
  (product odd_primes_less_than_32) % 32 = 23 :=
by sorry

end remainder_of_product_of_odd_primes_mod_32_l369_369287


namespace find_k_l369_369972

open_locale big_operators

theorem find_k (k : ℝ) (h₁ : 1 < k) (h₂ : ∑' n, (7 * n - 3) / k^n = 2) :
  k = 2 + 3 * real.sqrt 2 / 2 :=
sorry

end find_k_l369_369972


namespace value_of_f_neg_5π_over_12_l369_369784

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem value_of_f_neg_5π_over_12 :
  ∀ (ω φ : ℝ), 
    (∀ x y : ℝ, (x < y ∧ x ∈ Ioo (π / 6) (2 * π / 3) ∧ y ∈ Ioo (π / 6) (2 * π / 3)) → f x ω φ < f y ω φ) ∧ 
    f (π / 6) ω φ = f (2 * π / 3) ω φ → 
    f (-5 * π / 12) ω φ = √3 / 2 :=
by
  sorry

end value_of_f_neg_5π_over_12_l369_369784


namespace gcd_18_30_is_6_l369_369612

def gcd_18_30 : ℕ :=
  gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 :=
by {
  -- The step here will involve using properties of gcd and prime factorization,
  -- but we are given the result directly for the purpose of this task.
  sorry
}

end gcd_18_30_is_6_l369_369612


namespace sin_monotonic_increasing_f_symmetric_axes_find_value_l369_369696

def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi * 5 / 6)

theorem sin_monotonic_increasing (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) :
  Monotone (λ x, f x) (Ioo a b) := 
sorry

theorem f_symmetric_axes (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) 
  (x y : ℝ) (hx : x = a + (b - a) * k) (hy : y = a + (b - a) * (-k)) : 
  f x = f y :=
sorry

theorem find_value :
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
sorry

end sin_monotonic_increasing_f_symmetric_axes_find_value_l369_369696


namespace remainder_when_M_divided_by_32_l369_369147

-- Define M as the product of all odd primes less than 32.
def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_M_divided_by_32 :
    M % 32 = 1 := by
  -- We must prove that M % 32 = 1.
  sorry

end remainder_when_M_divided_by_32_l369_369147


namespace product_of_odd_primes_mod_32_l369_369178

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def M : ℕ := odd_primes_less_than_32.foldr (· * ·) 1

theorem product_of_odd_primes_mod_32 : M % 32 = 17 :=
by
  -- the proof goes here
  sorry

end product_of_odd_primes_mod_32_l369_369178


namespace sum_of_digits_82_l369_369378

def tens_digit (n : ℕ) : ℕ := n / 10
def units_digit (n : ℕ) : ℕ := n % 10
def sum_of_digits (n : ℕ) : ℕ := tens_digit n + units_digit n

theorem sum_of_digits_82 : sum_of_digits 82 = 10 := by
  sorry

end sum_of_digits_82_l369_369378


namespace garden_length_l369_369102

theorem garden_length (P b l : ℕ) (h1 : P = 500) (h2 : b = 100) : l = 150 :=
by
  sorry

end garden_length_l369_369102


namespace shaded_area_correct_l369_369454

noncomputable def shaded_area : ℝ := 
  let radius := 3
  let side := 2
  let sector_area := (120 / 360) * π * (radius ^ 2)
  let triangle_area := (1 / 2) * (radius ^ 2) * (Real.sin (pi / 3))
  sector_area - 2 * triangle_area

theorem shaded_area_correct :
  shaded_area = 3 * π - (9 * Real.sqrt 3 / 2) := by
  sorry

end shaded_area_correct_l369_369454


namespace eccentricity_is_two_l369_369828

noncomputable def eccentricity_of_hyperbola (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b / a = Real.sqrt 3) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  c / a

theorem eccentricity_is_two (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b / a = Real.sqrt 3) : 
  eccentricity_of_hyperbola a b h1 h2 h3 = 2 := 
  sorry

end eccentricity_is_two_l369_369828


namespace gcd_18_30_l369_369537

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l369_369537


namespace remainder_of_M_l369_369270

def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_of_M : M % 32 = 31 := by
  -- Proof goes here
  sorry

end remainder_of_M_l369_369270


namespace triangle_inequality_l369_369943

noncomputable theory

variables {a b c a1 a2 b1 b2 c1 c2 : ℝ} {P1 P2 : Type}

-- Let's assume a, b, c are the lengths of the sides of a triangle
-- And the quantities a1, a2, b1, b2, c1, c2 as defined in the problem
-- We want to prove the inequality
theorem triangle_inequality (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0)
  (ha1 : a1 ≥ 0) (ha2 : a2 ≥ 0) (hb1 : b1 ≥ 0) (hb2 : b2 ≥ 0)
  (hc1 : c1 ≥ 0) (hc2 : c2 ≥ 0) :
  a * a1 * a2 + b * b1 * b2 + c * c1 * c2 ≥ a * b * c :=
by {
  sorry
}

end triangle_inequality_l369_369943


namespace remainder_of_product_of_odd_primes_mod_32_l369_369278

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem remainder_of_product_of_odd_primes_mod_32 :
  (product odd_primes_less_than_32) % 32 = 23 :=
by sorry

end remainder_of_product_of_odd_primes_mod_32_l369_369278


namespace greatest_xy_l369_369855

theorem greatest_xy (x y : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 7 * x + 4 * y = 140) : xy ≤ 168 :=
begin
  sorry
end

example : ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ 7 * x + 4 * y = 140 ∧ xy = 168 :=
begin
  use [8, 21],
  split, exact dec_trivial,
  split, exact dec_trivial,
  split, exact dec_trivial,
  exact dec_trivial
end

end greatest_xy_l369_369855


namespace gcd_18_30_l369_369609

-- Define the two numbers
def num1 : ℕ := 18
def num2 : ℕ := 30

-- State the theorem to find the gcd
theorem gcd_18_30 : Nat.gcd num1 num2 = 6 := by
  sorry

end gcd_18_30_l369_369609


namespace find_number_l369_369840

theorem find_number (x : ℕ) (h : x + 56 = 110) : x = 54 :=
sorry

end find_number_l369_369840


namespace first_digit_base8_rep_395_eq_6_l369_369412

-- Define the base 10 number
def base10_num : ℕ := 395

-- Define the target base
def base : ℕ := 8

-- Define the first digit in base 8 representation for 395 == 6
theorem first_digit_base8_rep_395_eq_6 (base10_num : ℕ) (base : ℕ) : nat.digits base base10_num = (6 :: _) :=
by
  sorry

end first_digit_base8_rep_395_eq_6_l369_369412


namespace gcd_of_18_and_30_l369_369561

-- Define the numbers
def a := 18
def b := 30

-- The main theorem statement
theorem gcd_of_18_and_30 : Nat.gcd a b = 6 := by
  sorry

end gcd_of_18_and_30_l369_369561


namespace greatest_xy_l369_369857

theorem greatest_xy (x y : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 7 * x + 4 * y = 140) : xy ≤ 168 :=
begin
  sorry
end

example : ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ 7 * x + 4 * y = 140 ∧ xy = 168 :=
begin
  use [8, 21],
  split, exact dec_trivial,
  split, exact dec_trivial,
  split, exact dec_trivial,
  exact dec_trivial
end

end greatest_xy_l369_369857


namespace S_5_eq_5_div_11_l369_369649

def a_n (n : ℕ) : ℚ := 1 / ((2 * n - 1) * (2 * n + 1))

def S (n : ℕ) : ℚ := ∑ i in Finset.range n.succ, a_n i

theorem S_5_eq_5_div_11 : S 5 = 5 / 11 :=
by
  sorry

end S_5_eq_5_div_11_l369_369649


namespace calculator_game_result_l369_369007

def calculator_game_sum (N : ℕ) : ℝ :=
 (N - 1)

theorem calculator_game_result (N : ℕ) : ∃ N, 
    (∃ k : ℕ, N = (2:ℝ)^((2^k:ℝ))) ∧ (calculator_game_sum N = N - 1) :=
by
  use 2^(2^50:ℝ)
  split
  {
    use 50
    sorry
  }
  {
    simp [calculator_game_sum]
    sorry
  }

end calculator_game_result_l369_369007


namespace evaluate_expression_l369_369528

theorem evaluate_expression :
  let f : ℝ → ℤ := λ x, int.floor x
  let c : ℝ → ℤ := λ x, int.ceil x
  (f (-6.5) * c 6.5)^2 *
  (f (-5.5) * c 5.5)^2 *
  (f (-4.5) * c 4.5)^2 *
  (f (-3.5) * c 3.5)^2 *
  (f (-2.5) * c 2.5)^2 *
  (f (-1.5) * c 1.5)^2 
  = 53910169600 := by
  sorry

end evaluate_expression_l369_369528


namespace gcd_of_18_and_30_l369_369592

-- Define the numbers
def num1 := 18
def num2 := 30

-- State the GCD property
theorem gcd_of_18_and_30 : Nat.gcd num1 num2 = 6 :=
by
  sorry

end gcd_of_18_and_30_l369_369592


namespace frustum_lateral_surface_area_l369_369373

theorem frustum_lateral_surface_area :
  ∀ (r : ℝ), (r > 0) → (4 * r > 0) → (4 * r > 0) → 
  let r1 := r in 
  let r2 := 4 * r in 
  let h := 4 * r in
  let l := 10 in
  (l = 10) →
  (π * (r1 + r2) * l = 100 * π) :=
begin
  intros r r_pos _ _ r1 r2 h l h_eq l_eq,
  have r_eq_2: r = 2,
  {
    sorry
  },
  rw [r_eq_2] at *,
  have r1_eq_2: r1 = 2,
  {
    sorry
  },
  have r2_eq_8: r2 = 8,
  {
    sorry
  },
  rw [r1_eq_2, r2_eq_8, l_eq],
  ring,
end

end frustum_lateral_surface_area_l369_369373


namespace vertices_after_cut_off_four_corners_l369_369955

-- Definitions for the conditions
def regular_tetrahedron.num_vertices : ℕ := 4

def new_vertices_per_cut : ℕ := 3

def total_vertices_after_cut : ℕ := 
  regular_tetrahedron.num_vertices + regular_tetrahedron.num_vertices * new_vertices_per_cut

-- The theorem to prove the question
theorem vertices_after_cut_off_four_corners :
  total_vertices_after_cut = 12 :=
by
  -- sorry is used to skip the proof steps, as per instructions
  sorry

end vertices_after_cut_off_four_corners_l369_369955


namespace find_value_l369_369712

-- Define the function $f(x)$
def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

-- Define the problem statement in Lean 4
theorem find_value :
  (∀ x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), 0 ≤ 2 * x - 5 * Real.pi / 6 ∧ 2 * x - 5 * Real.pi / 6 ≤ Real.pi) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by
  -- A proper rigorous Lean proof should go here
  sorry

end find_value_l369_369712


namespace product_of_odd_primes_mod_32_l369_369168

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def M : ℕ := odd_primes_less_than_32.foldr (· * ·) 1

theorem product_of_odd_primes_mod_32 : M % 32 = 17 :=
by
  -- the proof goes here
  sorry

end product_of_odd_primes_mod_32_l369_369168


namespace max_xy_value_l369_369885

theorem max_xy_value (x y : ℕ) (h : 7 * x + 4 * y = 140) : xy ≤ 168 :=
by sorry

end max_xy_value_l369_369885


namespace express_308_million_in_scientific_notation_l369_369357

theorem express_308_million_in_scientific_notation :
    (308000000 : ℝ) = 3.08 * (10 ^ 8) :=
by
  sorry

end express_308_million_in_scientific_notation_l369_369357


namespace max_value_frac_c_b_l369_369663

variable (a b c : ℝ)
variable (h_pos : ∀ x, x > 0 → x ∈ {a, b, c})
variable (h_height : ∀ x, x = a / 2 → x ∈ {height to side BC})

theorem max_value_frac_c_b (h : height_to_bc a b c = a / 2) : 
  max (c / b) = sqrt 5 :=
sorry

end max_value_frac_c_b_l369_369663


namespace gcd_of_18_and_30_l369_369547

theorem gcd_of_18_and_30 : Nat.gcd 18 30 = 6 :=
by
  sorry

end gcd_of_18_and_30_l369_369547


namespace number_of_sides_of_polygon_l369_369385

theorem number_of_sides_of_polygon (n : ℕ) : (n - 2) * 180 = 720 → n = 6 :=
by
  sorry

end number_of_sides_of_polygon_l369_369385


namespace trajectory_of_M_is_line_segment_l369_369834

variables (F1 F2 M : Point)
variables (distance : Point → Point → Real)

-- Define the conditions
def F1F2_distance_condition : Prop :=
  distance F1 F2 = 4

def MF1_MF2_sum_condition : Prop :=
  distance M F1 + distance M F2 = 4

-- Define the goal
theorem trajectory_of_M_is_line_segment
  (cond1 : F1F2_distance_condition)
  (cond2 : MF1_MF2_sum_condition) :
  lies_on_line_segment M F1 F2 := sorry

end trajectory_of_M_is_line_segment_l369_369834


namespace g_at_three_l369_369354

noncomputable def g : ℝ → ℝ := sorry

axiom g_functional_eq (x y : ℝ) : g (x + y) = g x * g y
axiom g_nonzero_at_zero : g 0 ≠ 0
axiom g_at_one : g 1 = 2

theorem g_at_three : g 3 = 8 := sorry

end g_at_three_l369_369354


namespace trajectory_of_point_M_l369_369064

theorem trajectory_of_point_M :
  ∀ (x y : ℝ),
  (∀ M : ℝ × ℝ, M = (x, y) →
  (⁅((x - 5)^2 + y^2)⁆.sqrt / |x - 9 / 5| = 5 / 3)) →
  x^2 / 9 - y^2 / 16 = 1 :=
by
  sorry

end trajectory_of_point_M_l369_369064


namespace eccentricity_of_ellipse_l369_369920

def ellipse_equation (x y : ℝ) (m : ℝ) : Prop := x^2 / 4 + y^2 / m = 1
def sum_of_distances_to_foci (x y : ℝ) (m : ℝ) : Prop := 
  ∃ fx₁ fy₁ fx₂ fy₂ : ℝ, (ellipse_equation x y m) ∧ 
  (dist (x, y) (fx₁, fy₁) + dist (x, y) (fx₂, fy₂) = m - 3)

theorem eccentricity_of_ellipse (m : ℝ) (h₁ : 4 < m) 
  (h₂ : sum_of_distances_to_foci x y m) : 
  ∃ e : ℝ, e = √5 / 3 :=
sorry

end eccentricity_of_ellipse_l369_369920


namespace max_mutually_exclusive_number_l369_369098

def F (m : ℕ) : ℕ := 
  let unit_digit := m % 10
  let m_prime := m / 10
  in m_prime - unit_digit

def G (m : ℕ) : ℕ := 
  let tens_digit := (m / 10) % 10
  let unit_digit := m % 10
  in tens_digit - unit_digit

def is_mutually_exclusive (m : ℕ) : Prop := 
  let d1 := m / 100
  let d2 := (m / 10) % 10
  let d3 := m % 10
  in d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3 ∧ d1 ≠ 0 ∧ d2 ≠ 0 ∧ d3 ≠ 0

def special_condition (x y : ℕ) : Prop := 
  let m := 20 * (5 * x + 1) + 2 * y
  in (1 ≤ x ∧ x ≤ 9) ∧ (1 ≤ y ∧ y ≤ 9) ∧ 
     is_mutually_exclusive m ∧ 
     (F m) % (G m) = 0 ∧ 
     (F m) / (G m) % 13 = 0

theorem max_mutually_exclusive_number : 
  ∃ (m : ℕ), 
  ∃ (x y : ℕ), 
    special_condition x y ∧ 
    m = 20 * (5 * x + 1) + 2 * y ∧ 
    m = 932 :=
by {
  sorry
}

end max_mutually_exclusive_number_l369_369098


namespace product_mod_32_l369_369214

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  M % 32 = 17 := by
  sorry

end product_mod_32_l369_369214


namespace perfect_squares_less_than_20000_representable_l369_369335

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define the difference of two consecutive perfect squares
def consecutive_difference (b : ℕ) : ℕ :=
  (b + 1) ^ 2 - b ^ 2

-- Define the condition under which the perfect square is less than 20000
def less_than_20000 (n : ℕ) : Prop :=
  n < 20000

-- Define the main problem statement using the above definitions
theorem perfect_squares_less_than_20000_representable :
  ∃ count : ℕ, (∀ n : ℕ, (is_perfect_square n) ∧ (less_than_20000 n) →
  ∃ b : ℕ, n = consecutive_difference b) ∧ count = 69 :=
sorry

end perfect_squares_less_than_20000_representable_l369_369335


namespace sum_of_possible_m_values_l369_369664

theorem sum_of_possible_m_values :
  (∑ m in (Finset.filter (λ m : ℤ, 0 < 3 * m ∧ 3 * m < 27) (Finset.Icc 1 8)).val.to_finset, (m : ℤ)) = 36 
:= sorry

end sum_of_possible_m_values_l369_369664


namespace problem_statement_l369_369691

-- Definition of our function f
def f (x : ℝ) : ℝ := Real.sin (2 * x - (5 * Real.pi / 6))

-- Theorem to prove that f(-5π/12) = √3/2
theorem problem_statement : 
  f(-5 * Real.pi / 12) = sqrt 3 / 2 := 
sorry

end problem_statement_l369_369691


namespace hyperbola_eccentricity_l369_369827

theorem hyperbola_eccentricity (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_asymptote : b = sqrt 3 * a) :
  let c := sqrt (a^2 + b^2)
  let e := c / a
  e = 2 :=
by
  have h1 : b = sqrt 3 * a := h_asymptote
  have h2 : c = sqrt (a^2 + b^2) := rfl
  have h3 : e = c / a := rfl
  sorry

end hyperbola_eccentricity_l369_369827


namespace value_of_f_neg_5π_over_12_l369_369787

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem value_of_f_neg_5π_over_12 :
  ∀ (ω φ : ℝ), 
    (∀ x y : ℝ, (x < y ∧ x ∈ Ioo (π / 6) (2 * π / 3) ∧ y ∈ Ioo (π / 6) (2 * π / 3)) → f x ω φ < f y ω φ) ∧ 
    f (π / 6) ω φ = f (2 * π / 3) ω φ → 
    f (-5 * π / 12) ω φ = √3 / 2 :=
by
  sorry

end value_of_f_neg_5π_over_12_l369_369787


namespace gcd_18_30_l369_369568

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := 
by
  sorry

end gcd_18_30_l369_369568


namespace find_value_of_f_eq_l369_369760

noncomputable def f (ω ω φ x : ℝ) : ℝ := sin (ω * x + φ)

theorem find_value_of_f_eq :
  ∀ (ω φ : ℝ), 
    (∀ x, x ∈ set.Ioo (π / 6) (2 * π / 3) → (f ω φ x) ≤ (f ω φ (x + 1e-10))) → -- monotonically increasing
    (ω * ∓ (π / 6) + φ) = (φ + ω * (2 * π / 3)) → -- symmetric axes
    f ω φ (-(5 * π / 12)) = sqrt 3 / 2 :=
by
  sorry

end find_value_of_f_eq_l369_369760


namespace max_xy_l369_369859

theorem max_xy (x y : ℕ) (h1: 7 * x + 4 * y = 140) : ∃ x y, 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by {
  sorry
}

end max_xy_l369_369859


namespace gcd_18_30_is_6_l369_369617

def gcd_18_30 : ℕ :=
  gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 :=
by {
  -- The step here will involve using properties of gcd and prime factorization,
  -- but we are given the result directly for the purpose of this task.
  sorry
}

end gcd_18_30_is_6_l369_369617


namespace orthocenter_incircle_center_l369_369976

theorem orthocenter_incircle_center (ABCD : CyclicQuadrilateral) 
(K : IntersectionPoint (LineSegment ABCD.AB) (LineSegment ABCD.CD))
(L : IntersectionPoint (LineSegment ABCD.AD) (LineSegment ABCD.BC)) :
    Orthocenter (Triangle (LineSegment K.L) (LineSegment ABCD.AC) (LineSegment ABCD.BD)) = Incircle Ctr ABCD := 
sorry 

end orthocenter_incircle_center_l369_369976


namespace tree_count_l369_369929

theorem tree_count (total_trees : ℕ) (pine_percentage oak_percentage maple_percentage : ℝ)
  (h_total : total_trees = 3250)
  (h_pine : pine_percentage = 0.45)
  (h_oak : oak_percentage = 0.25)
  (h_maple : maple_percentage = 0.12) :
  let num_pine := Int.ofNat (floor (pine_percentage * total_trees + 0.5))
  let num_oak := Int.ofNat (floor (oak_percentage * total_trees + 0.5))
  let num_maple := Int.ofNat (floor (maple_percentage * total_trees + 0.5))
  let num_fruit := Int.ofNat (floor ((1 - (pine_percentage + oak_percentage + maple_percentage)) * total_trees + 0.5))
  num_pine = 1463 ∧ num_oak = 813 ∧ num_maple = 390 ∧ num_fruit = 585 := by
  sorry

end tree_count_l369_369929


namespace scientific_notation_1p2_billion_l369_369998

theorem scientific_notation_1p2_billion :
  (1_200_000_000: ℝ) = 1.2 * 10^8 :=
sorry

end scientific_notation_1p2_billion_l369_369998


namespace max_f_l369_369974

noncomputable def S_n (n : ℕ) : ℚ :=
  n * (n + 1) / 2

noncomputable def f (n : ℕ) : ℚ :=
  S_n n / ((n + 32) * S_n (n + 1))

theorem max_f (n : ℕ) : f n ≤ 1 / 50 := sorry

-- Verify the bound is achieved for n = 8
example : f 8 = 1 / 50 := by
  unfold f S_n
  norm_num

end max_f_l369_369974


namespace remainder_when_M_divided_by_32_l369_369141

-- Define M as the product of all odd primes less than 32.
def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_M_divided_by_32 :
    M % 32 = 1 := by
  -- We must prove that M % 32 = 1.
  sorry

end remainder_when_M_divided_by_32_l369_369141


namespace product_mod_32_is_15_l369_369235

-- Define the odd primes less than 32
def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the product of odd primes
def M : ℕ := List.foldl (· * ·) 1 oddPrimes

-- The theorem to prove the remainder when M is divided by 32 is 15
theorem product_mod_32_is_15 : M % 32 = 15 := by
  sorry

end product_mod_32_is_15_l369_369235


namespace second_number_more_than_first_l369_369922

-- Definitions of A and B based on the given ratio
def A : ℚ := 7 / 56
def B : ℚ := 8 / 56

-- Proof statement
theorem second_number_more_than_first : ((B - A) / A) * 100 = 100 / 7 :=
by
  -- skipped the proof
  sorry

end second_number_more_than_first_l369_369922


namespace arun_weight_lower_limit_l369_369924

variable {W B : ℝ}

theorem arun_weight_lower_limit
  (h1 : 64 < W ∧ W < 72)
  (h2 : B < W ∧ W < 70)
  (h3 : W ≤ 67)
  (h4 : (64 + 67) / 2 = 66) :
  64 < B :=
by sorry

end arun_weight_lower_limit_l369_369924


namespace intersection_points_exactly_four_l369_369521

noncomputable def intersect_points (A : ℝ) (hA : 0 < A) : ℕ :=
  (let f : ℝ → ℝ := λ x, A * x^2 in
   let g : ℝ → ℝ := λ y, y^2 + 5 - 6 * y in
   let h : ℝ → ℝ := λ x, x^2 + 5 / A - 6 in
   4)

theorem intersection_points_exactly_four (A : ℝ) (hA : 0 < A) :
  intersect_points A hA = 4 :=
  sorry

end intersection_points_exactly_four_l369_369521


namespace greatest_xy_value_l369_369876

theorem greatest_xy_value (x y : ℕ) (h1 : 7 * x + 4 * y = 140) (h2 : x > 0) (h3 : y > 0) : 
  xy ≤ 112 :=
by
  sorry

end greatest_xy_value_l369_369876


namespace product_of_odd_primes_mod_32_l369_369183

theorem product_of_odd_primes_mod_32 :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  M % 32 = 17 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  exact sorry

end product_of_odd_primes_mod_32_l369_369183


namespace gcd_18_30_l369_369588

theorem gcd_18_30: Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l369_369588


namespace value_of_f_neg_5π_over_12_l369_369786

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem value_of_f_neg_5π_over_12 :
  ∀ (ω φ : ℝ), 
    (∀ x y : ℝ, (x < y ∧ x ∈ Ioo (π / 6) (2 * π / 3) ∧ y ∈ Ioo (π / 6) (2 * π / 3)) → f x ω φ < f y ω φ) ∧ 
    f (π / 6) ω φ = f (2 * π / 3) ω φ → 
    f (-5 * π / 12) ω φ = √3 / 2 :=
by
  sorry

end value_of_f_neg_5π_over_12_l369_369786


namespace product_of_odd_primes_mod_32_l369_369174

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def M : ℕ := odd_primes_less_than_32.foldr (· * ·) 1

theorem product_of_odd_primes_mod_32 : M % 32 = 17 :=
by
  -- the proof goes here
  sorry

end product_of_odd_primes_mod_32_l369_369174


namespace sum_of_divisors_divisible_by_24_l369_369133

theorem sum_of_divisors_divisible_by_24 (n : ℕ) (h : 24 ∣ n + 1) :
  24 ∣ ∑ d in (Finset.divisors n), d := 
sorry

end sum_of_divisors_divisible_by_24_l369_369133


namespace function_relation_l369_369314

def f (x : ℝ) (c : ℝ) : ℝ := x^2 + 4*x + c

theorem function_relation (c : ℝ) :
  f 1 c > f 0 c ∧ f 0 c > f (-2) c := by
  sorry

end function_relation_l369_369314


namespace johny_distance_l369_369129

noncomputable def distance_south : ℕ := 40
variable (E : ℕ)
noncomputable def distance_east : ℕ := E
noncomputable def distance_north (E : ℕ) : ℕ := 2 * E
noncomputable def total_distance (E : ℕ) : ℕ := distance_south + distance_east E + distance_north E

theorem johny_distance :
  ∀ E : ℕ, total_distance E = 220 → E - distance_south = 20 :=
by
  intro E
  intro h
  rw [total_distance, distance_north, distance_east, distance_south] at h
  sorry

end johny_distance_l369_369129


namespace f_equals_fibonacci_l369_369961

-- Defining the function f(n)
noncomputable def f : ℕ → ℕ
| 0     := 0 -- We define f(0) as 0 since it is beyond the scope of positive integers.
| 1     := 1 -- Initial condition f(1) = 1.
| 2     := 1 -- Initial condition f(2) = 1.
| (n+3) := f (n+2) + f (n+1)

-- Define the expected Fibonacci function
noncomputable def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n+1) + fib n

theorem f_equals_fibonacci (n : ℕ) : f n = fib n :=
sorry

end f_equals_fibonacci_l369_369961


namespace remainder_when_M_divided_by_32_l369_369146

-- Define M as the product of all odd primes less than 32.
def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_M_divided_by_32 :
    M % 32 = 1 := by
  -- We must prove that M % 32 = 1.
  sorry

end remainder_when_M_divided_by_32_l369_369146


namespace product_of_odd_primes_mod_32_l369_369227

theorem product_of_odd_primes_mod_32 :
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31) in
  M % 32 = 25 :=
by
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31)
  sorry

end product_of_odd_primes_mod_32_l369_369227


namespace remainder_of_M_when_divided_by_32_l369_369204

open Nat

def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
def M : ℕ := List.foldl (*) 1 oddPrimes

theorem remainder_of_M_when_divided_by_32 :
  M % 32 = 1 :=
sorry

end remainder_of_M_when_divided_by_32_l369_369204


namespace perfect_squares_less_than_20000_representable_l369_369336

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define the difference of two consecutive perfect squares
def consecutive_difference (b : ℕ) : ℕ :=
  (b + 1) ^ 2 - b ^ 2

-- Define the condition under which the perfect square is less than 20000
def less_than_20000 (n : ℕ) : Prop :=
  n < 20000

-- Define the main problem statement using the above definitions
theorem perfect_squares_less_than_20000_representable :
  ∃ count : ℕ, (∀ n : ℕ, (is_perfect_square n) ∧ (less_than_20000 n) →
  ∃ b : ℕ, n = consecutive_difference b) ∧ count = 69 :=
sorry

end perfect_squares_less_than_20000_representable_l369_369336


namespace students_algebra_or_drafting_not_both_not_geography_l369_369024

variables (A D G : Finset ℕ)
-- Condition 1: Fifteen students are taking both algebra and drafting
variable (h1 : (A ∩ D).card = 15)
-- Condition 2: There are 30 students taking algebra
variable (h2 : A.card = 30)
-- Condition 3: There are 12 students taking drafting only
variable (h3 : (D \ A).card = 12)
-- Condition 4: There are eight students taking a geography class
variable (h4 : G.card = 8)
-- Condition 5: Two students are also taking both algebra and drafting and geography
variable (h5 : ((A ∩ D) ∩ G).card = 2)

-- Question: Prove the final count of students taking algebra or drafting but not both, and not taking geography is 25
theorem students_algebra_or_drafting_not_both_not_geography :
  ((A \ D) ∪ (D \ A)).card - ((A ∩ D) ∩ G).card = 25 :=
by
  sorry

end students_algebra_or_drafting_not_both_not_geography_l369_369024


namespace gcd_18_30_l369_369544

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l369_369544


namespace number_of_boxes_of_nectarines_l369_369394

namespace ProofProblem

/-- Define the given conditions: -/
def crates : Nat := 12
def oranges_per_crate : Nat := 150
def nectarines_per_box : Nat := 30
def total_fruit : Nat := 2280

/-- Define the number of oranges: -/
def total_oranges : Nat := crates * oranges_per_crate

/-- Calculate the number of nectarines: -/
def total_nectarines : Nat := total_fruit - total_oranges

/-- Calculate the number of boxes of nectarines: -/
def boxes_of_nectarines : Nat := total_nectarines / nectarines_per_box

-- Theorem stating that given the conditions, the number of boxes of nectarines is 16.
theorem number_of_boxes_of_nectarines :
  boxes_of_nectarines = 16 := by
  sorry

end ProofProblem

end number_of_boxes_of_nectarines_l369_369394


namespace johns_score_unique_l369_369128

theorem johns_score_unique (s c w : ℕ) (h1 : s > 70) (h2 : s = 5 * c - 2 * w) 
  (h3 : ∀ s', s' > 70 → s' < s → (¬ ∃ c' w', s' = 5 * c' - 2 * w' ∧ c ∈ ℕ)) : s = 71 :=
by
  sorry

end johns_score_unique_l369_369128


namespace remainder_M_mod_32_l369_369257

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_M_mod_32 : M % 32 = 17 :=
by {
  sorry
}

end remainder_M_mod_32_l369_369257


namespace chloe_sold_strawberries_l369_369511

noncomputable section

def cost_per_dozen : ℕ := 50
def sale_price_per_half_dozen : ℕ := 30
def total_profit : ℕ := 500
def profit_per_half_dozen := sale_price_per_half_dozen - (cost_per_dozen / 2)
def half_dozens_sold := total_profit / profit_per_half_dozen

theorem chloe_sold_strawberries : half_dozens_sold / 2 = 50 :=
by
  -- proof would go here
  sorry

end chloe_sold_strawberries_l369_369511


namespace centroids_perpendicular_orthocenters_l369_369136

section Geometry

variables {A B C D E : Type} [ConvexQuadrilateral A B C D E]

open EuclideanGeometry

theorem centroids_perpendicular_orthocenters (hE : E = intersection_of_diagonals A B C D) :
  let S1 := centroid A E B,
      S2 := centroid C E D,
      M1 := orthocenter B E C,
      M2 := orthocenter D E A in
  ∃ S1 S2 M1 M2, line S1 S2 ⊥ line M1 M2 :=  
sorry

end Geometry

end centroids_perpendicular_orthocenters_l369_369136


namespace range_of_function_l369_369000

noncomputable def range_of_y : Set ℝ :=
  {y | ∃ x : ℝ, y = |x + 5| - |x - 3|}

theorem range_of_function : range_of_y = Set.Icc (-2) 12 :=
by
  sorry

end range_of_function_l369_369000


namespace smallest_t_eq_3_over_4_l369_369034

theorem smallest_t_eq_3_over_4 (t : ℝ) :
  (∀ t : ℝ,
    (16 * t^3 - 49 * t^2 + 35 * t - 6) / (4 * t - 3) + 7 * t = 8 * t - 2 → t >= (3 / 4)) ∧
  (∃ t₀ : ℝ, (16 * t₀^3 - 49 * t₀^2 + 35 * t₀ - 6) / (4 * t₀ - 3) + 7 * t₀ = 8 * t₀ - 2 ∧ t₀ = (3 / 4)) :=
sorry

end smallest_t_eq_3_over_4_l369_369034


namespace sin_monotonic_increasing_f_symmetric_axes_find_value_l369_369698

def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi * 5 / 6)

theorem sin_monotonic_increasing (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) :
  Monotone (λ x, f x) (Ioo a b) := 
sorry

theorem f_symmetric_axes (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) 
  (x y : ℝ) (hx : x = a + (b - a) * k) (hy : y = a + (b - a) * (-k)) : 
  f x = f y :=
sorry

theorem find_value :
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
sorry

end sin_monotonic_increasing_f_symmetric_axes_find_value_l369_369698


namespace sequence_inequality_l369_369439

def F : ℕ → ℕ
| 0 => 1
| 1 => 1
| 2 => 2
| (n+2) => F (n+1) + F n

theorem sequence_inequality (n : ℕ) :
  (F (n+1) : ℝ)^(1 / n) ≥ 1 + 1 / ((F n : ℝ)^(1 / n)) :=
by
  sorry

end sequence_inequality_l369_369439


namespace product_of_odd_primes_mod_32_l369_369188

theorem product_of_odd_primes_mod_32 :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  M % 32 = 17 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  exact sorry

end product_of_odd_primes_mod_32_l369_369188


namespace sum_of_x_coordinates_l369_369361

def line1 (x : ℝ) : ℝ := -3 * x - 5
def line2 (x : ℝ) : ℝ := 2 * x - 3

def has_x_intersect (line : ℝ → ℝ) (y : ℝ) : Prop := ∃ x : ℝ, line x = y

theorem sum_of_x_coordinates :
  (∃ x1 x2 : ℝ, line1 x1 = 2.2 ∧ line2 x2 = 2.2 ∧ x1 + x2 = 0.2) :=
  sorry

end sum_of_x_coordinates_l369_369361


namespace sin_monotonic_increasing_f_symmetric_axes_find_value_l369_369702

def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi * 5 / 6)

theorem sin_monotonic_increasing (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) :
  Monotone (λ x, f x) (Ioo a b) := 
sorry

theorem f_symmetric_axes (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) 
  (x y : ℝ) (hx : x = a + (b - a) * k) (hy : y = a + (b - a) * (-k)) : 
  f x = f y :=
sorry

theorem find_value :
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
sorry

end sin_monotonic_increasing_f_symmetric_axes_find_value_l369_369702


namespace expand_and_solve_solve_quadratic_l369_369021

theorem expand_and_solve (x : ℝ) :
  6 * (x - 3) * (x + 5) = 6 * x^2 + 12 * x - 90 :=
by sorry

theorem solve_quadratic (x : ℝ) :
  6 * x^2 + 12 * x - 90 = 0 ↔ x = -5 ∨ x = 3 :=
by sorry

end expand_and_solve_solve_quadratic_l369_369021


namespace find_f_of_neg_5_pi_over_12_l369_369733

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - (5 * π / 6))

theorem find_f_of_neg_5_pi_over_12 :
  (∀ x ∈ (set.Ioo (π / 6) (2 * π / 3)), (0 : ℝ) < (f x - f (x - 1))) ∧
  (∀ x, f (π / 6 - x) = f (π / 6 + x)) ∧
  (∀ x, f (2 * π / 3 - x) = f (2 * π / 3 + x)) →
  f (-5 * π / 12) = √3 / 2 :=
by
  sorry

end find_f_of_neg_5_pi_over_12_l369_369733


namespace sin_function_value_l369_369803

noncomputable def f (x : ℝ) : ℝ := sin(2 * x - 5 * π / 6)

theorem sin_function_value :
  f (-5 * π / 12) = sqrt 3 / 2 := by
  sorry

end sin_function_value_l369_369803


namespace range_of_a_l369_369977

noncomputable def satisfies_system (a b c : ℝ) : Prop :=
  (a^2 - b * c - 8 * a + 7 = 0) ∧ (b^2 + c^2 + b * c - 6 * a + 6 = 0)

theorem range_of_a (a b c : ℝ) 
  (h : satisfies_system a b c) : 1 ≤ a ∧ a ≤ 9 :=
by
  sorry

end range_of_a_l369_369977


namespace average_next_seven_consecutive_is_correct_l369_369347

-- Define the sum of seven consecutive integers starting at x.
def sum_seven_consecutive_integers (x : ℕ) : ℕ := 7 * x + 21

-- Define the next sequence of seven integers starting from y + 1.
def average_next_seven_consecutive_integers (x : ℕ) : ℕ :=
  let y := sum_seven_consecutive_integers x
  let start := y + 1
  (start + (start + 1) + (start + 2) + (start + 3) + (start + 4) + (start + 5) + (start + 6)) / 7

-- Problem statement
theorem average_next_seven_consecutive_is_correct (x : ℕ) : 
  average_next_seven_consecutive_integers x = 7 * x + 25 :=
by
  sorry

end average_next_seven_consecutive_is_correct_l369_369347


namespace find_leg_of_45_45_90_triangle_l369_369531

theorem find_leg_of_45_45_90_triangle (BC : ℝ) (hBC : BC = 18 * Real.sqrt 2) : 
  ∀ (A B C : ℝ), 
  ∃ (AB : ℝ), 
  ∠BCA = 45 ∧ ∠BAC = 45 ∧ ∠ABC = 90 ∧ 
  BC = (AB * Real.sqrt 2) ∧ 
  AB = 18 := 
by
  -- sorry is used to skip the proof
  sorry

end find_leg_of_45_45_90_triangle_l369_369531


namespace gcd_18_30_l369_369536

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l369_369536


namespace max_triplets_l369_369395

-- Define conditions as Lean functions and predicates
def points := 1955

def is_valid_triplet (triplets : Finset (Finset ℕ)) : Prop :=
  ∀ t1 t2 ∈ triplets, t1 ≠ t2 → ∃! x, x ∈ t1 ∧ x ∈ t2

-- Define the main theorem statement using the conditions
theorem max_triplets (triplets : Finset (Finset ℕ)) : 
  triplets.card = 977 ∧
  ∀ t ∈ triplets, t.card = 3 ∧ is_valid_triplet triplets :=
sorry

end max_triplets_l369_369395


namespace greatest_xy_value_l369_369872

theorem greatest_xy_value (x y : ℕ) (h1 : 7 * x + 4 * y = 140) (h2 : x > 0) (h3 : y > 0) : 
  xy ≤ 112 :=
by
  sorry

end greatest_xy_value_l369_369872


namespace gcd_18_30_l369_369600

-- Define the two numbers
def num1 : ℕ := 18
def num2 : ℕ := 30

-- State the theorem to find the gcd
theorem gcd_18_30 : Nat.gcd num1 num2 = 6 := by
  sorry

end gcd_18_30_l369_369600


namespace greatest_xy_l369_369856

theorem greatest_xy (x y : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 7 * x + 4 * y = 140) : xy ≤ 168 :=
begin
  sorry
end

example : ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ 7 * x + 4 * y = 140 ∧ xy = 168 :=
begin
  use [8, 21],
  split, exact dec_trivial,
  split, exact dec_trivial,
  split, exact dec_trivial,
  exact dec_trivial
end

end greatest_xy_l369_369856


namespace margo_total_distance_l369_369991

theorem margo_total_distance
  (time_to_friend : ℝ)
  (time_to_home : ℝ)
  (average_rate : ℝ)
  (total_time : time_to_friend + time_to_home = 40) 
  (rate_in_hours : average_rate = 3) : 
  (total_distance : ℝ) := 
  let time_hours : ℝ := (time_to_friend + time_to_home) / 60 
  in let total_distance := average_rate * time_hours 
  in total_distance = 2 := 
sorry

end margo_total_distance_l369_369991


namespace sides_of_regular_polygon_l369_369479

theorem sides_of_regular_polygon (n : ℕ) (h : ∀ (i : ℕ), i < n → (interior_angle i = 140)) :
  n = 9 := 
  sorry

end sides_of_regular_polygon_l369_369479


namespace product_of_odd_primes_mod_32_l369_369192

theorem product_of_odd_primes_mod_32 :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  M % 32 = 17 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  exact sorry

end product_of_odd_primes_mod_32_l369_369192


namespace area_triangle_BCD_equals_l369_369963

-- Definitions of the points based on given conditions
def A := (0: ℝ, 0: ℝ, 0: ℝ)
def B := (3: ℝ, 0: ℝ, 0: ℝ)
def C := (0: ℝ, 4: ℝ, 0: ℝ)
def D := (0: ℝ, 0: ℝ, 5: ℝ)

-- Area of triangle ABC given
def x := 6

noncomputable def area_BCD : ℝ :=
  let BC := (C.1 - B.1, C.2 - B.2, C.3 - B.3) in
  let BD := (D.1 - B.1, D.2 - B.2, D.3 - B.3) in
  let cross_prod := (BC.2 * BD.3 - BC.3 * BD.2, BC.3 * BD.1 - BC.1 * BD.3, BC.1 * BD.2 - BC.2 * BD.1) in
  (0.5 * real.sqrt (cross_prod.1^2 + cross_prod.2^2 + cross_prod.3^2))

theorem area_triangle_BCD_equals : area_BCD = real.sqrt 769 / 2 :=
by sorry

end area_triangle_BCD_equals_l369_369963


namespace remainder_when_divided_by_32_l369_369154

def product_of_odds : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_divided_by_32 : product_of_odds % 32 = 9 := by
  sorry

end remainder_when_divided_by_32_l369_369154


namespace gcd_of_18_and_30_l369_369562

-- Define the numbers
def a := 18
def b := 30

-- The main theorem statement
theorem gcd_of_18_and_30 : Nat.gcd a b = 6 := by
  sorry

end gcd_of_18_and_30_l369_369562


namespace daffodil_bulb_cost_l369_369425

theorem daffodil_bulb_cost :
  let total_bulbs := 55
  let crocus_cost := 0.35
  let total_budget := 29.15
  let num_crocus_bulbs := 22
  let total_crocus_cost := num_crocus_bulbs * crocus_cost
  let remaining_budget := total_budget - total_crocus_cost
  let num_daffodil_bulbs := total_bulbs - num_crocus_bulbs
  remaining_budget / num_daffodil_bulbs = 0.65 := 
by
  -- proof to be filled in
  sorry

end daffodil_bulb_cost_l369_369425


namespace gcd_18_30_l369_369569

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := 
by
  sorry

end gcd_18_30_l369_369569


namespace sum_of_inradii_eq_height_l369_369506

variables (a b c h b1 a1 : ℝ)
variables (r r1 r2 : ℝ)

-- Assume CH is the height of the right-angled triangle ABC from the vertex of the right angle.
-- r, r1, r2 are the radii of the incircles of triangles ABC, AHC, and BHC respectively.
-- Given definitions:
-- BC = a
-- AC = b
-- AB = c
-- AH = b1
-- BH = a1
-- CH = h

-- Formulas for the radii of the respective triangles:
-- r : radius of incircle of triangle ABC = (a + b - h) / 2
-- r1 : radius of incircle of triangle AHC = (h + b1 - b) / 2
-- r2 : radius of incircle of triangle BHC = (h + a1 - a) / 2

theorem sum_of_inradii_eq_height 
  (H₁ : r = (a + b - h) / 2)
  (H₂ : r1 = (h + b1 - b) / 2) 
  (H₃ : r2 = (h + a1 - a) / 2) 
  (H₄ : b1 = b - h) 
  (H₅ : a1 = a - h) : 
  r + r1 + r2 = h :=
by
  sorry

end sum_of_inradii_eq_height_l369_369506


namespace remainder_of_product_of_odd_primes_mod_32_l369_369286

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem remainder_of_product_of_odd_primes_mod_32 :
  (product odd_primes_less_than_32) % 32 = 23 :=
by sorry

end remainder_of_product_of_odd_primes_mod_32_l369_369286


namespace gcd_of_18_and_30_l369_369555

theorem gcd_of_18_and_30 : Nat.gcd 18 30 = 6 :=
by
  sorry

end gcd_of_18_and_30_l369_369555


namespace product_mod_32_l369_369219

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  M % 32 = 17 := by
  sorry

end product_mod_32_l369_369219


namespace gcd_18_30_is_6_l369_369620

def gcd_18_30 : ℕ :=
  gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 :=
by {
  -- The step here will involve using properties of gcd and prime factorization,
  -- but we are given the result directly for the purpose of this task.
  sorry
}

end gcd_18_30_is_6_l369_369620


namespace find_expression_for_a_n_l369_369050

noncomputable def seq (n : ℕ) : ℕ := sorry
def sumFirstN (n : ℕ) : ℕ := sorry

theorem find_expression_for_a_n (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (h_pos : ∀ n, 0 < a n)
  (h_arith_seq : ∀ n, S n + 1 = 2 * a n) :
  ∀ n, a n = 2^(n-1) :=
sorry

end find_expression_for_a_n_l369_369050


namespace gcd_of_18_and_30_l369_369599

-- Define the numbers
def num1 := 18
def num2 := 30

-- State the GCD property
theorem gcd_of_18_and_30 : Nat.gcd num1 num2 = 6 :=
by
  sorry

end gcd_of_18_and_30_l369_369599


namespace fraction_value_l369_369640

theorem fraction_value
  (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (x^2 - 2*x + 1) / (x^2 - 1) = 1 - Real.sqrt 2 :=
by
  sorry

end fraction_value_l369_369640


namespace sin_monotonic_increasing_f_symmetric_axes_find_value_l369_369704

def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi * 5 / 6)

theorem sin_monotonic_increasing (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) :
  Monotone (λ x, f x) (Ioo a b) := 
sorry

theorem f_symmetric_axes (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) 
  (x y : ℝ) (hx : x = a + (b - a) * k) (hy : y = a + (b - a) * (-k)) : 
  f x = f y :=
sorry

theorem find_value :
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
sorry

end sin_monotonic_increasing_f_symmetric_axes_find_value_l369_369704


namespace gcd_of_18_and_30_l369_369552

theorem gcd_of_18_and_30 : Nat.gcd 18 30 = 6 :=
by
  sorry

end gcd_of_18_and_30_l369_369552


namespace recurrence_relation_l369_369913

-- Define the function p_nk and prove the recurrence relation
def p (n k : ℕ) : ℝ := sorry

theorem recurrence_relation (n k : ℕ) (h : k < n) : 
  p n k = p (n-1) k - (1 / 2^k) * p (n-k) k + (1 / 2^k) :=
sorry

end recurrence_relation_l369_369913


namespace problem_statement_l369_369688

-- Definition of our function f
def f (x : ℝ) : ℝ := Real.sin (2 * x - (5 * Real.pi / 6))

-- Theorem to prove that f(-5π/12) = √3/2
theorem problem_statement : 
  f(-5 * Real.pi / 12) = sqrt 3 / 2 := 
sorry

end problem_statement_l369_369688


namespace find_f_neg_5pi_12_l369_369772

-- Conditions
def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

variables {ω φ : ℝ}
def f (x : ℝ) : ℝ := Real.sin (ω * x + φ)

axiom monotonicity_condition : is_monotonically_increasing f (π / 6) (2 * π / 3)
axiom symmetry_condition_1 : f (π / 6) = f (2 * π / 3)
axiom symmetry_condition_2 : f (-π / 6) = f (π / 2)

-- Goal
theorem find_f_neg_5pi_12 : f (- 5 * π / 12) = sqrt 3 / 2 := 
sorry

end find_f_neg_5pi_12_l369_369772


namespace larger_number_is_37_l369_369388

-- Defining the conditions
def sum_of_two_numbers (a b : ℕ) : Prop := a + b = 62
def one_is_12_more (a b : ℕ) : Prop := a = b + 12

-- Proof statement
theorem larger_number_is_37 (a b : ℕ) (h₁ : sum_of_two_numbers a b) (h₂ : one_is_12_more a b) : a = 37 :=
by
  sorry

end larger_number_is_37_l369_369388


namespace matrix_multiplication_correct_l369_369513

def mat1 : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, 0], ![5, -3]]

def mat2 : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![8, -2], ![1, 1]]

def result : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![16, -4], ![37, -13]]

theorem matrix_multiplication_correct :
  mat1 ⬝ mat2 = result := by
  sorry

end matrix_multiplication_correct_l369_369513


namespace smallest_value_of_3a_plus_1_l369_369847

theorem smallest_value_of_3a_plus_1 (a : ℚ) (h : 8 * a^2 + 6 * a + 2 = 2) : 3 * a + 1 = -5/4 :=
by
  sorry

end smallest_value_of_3a_plus_1_l369_369847


namespace log3_seq_limit_value_l369_369014

theorem log3_seq_limit_value (x : ℝ) (h : x = Real.log 3 (64 + x)) : x ≈ 3.8 :=
sorry

end log3_seq_limit_value_l369_369014


namespace dino_second_gig_hourly_rate_l369_369016

theorem dino_second_gig_hourly_rate (h1 : 20 * 10 = 200)
  (h2 : 5 * 40 = 200) (h3 : 500 + 500 = 1000) : 
  let total_income := 1000 
  let income_first_gig := 200 
  let income_third_gig := 200 
  let income_second_gig := total_income - income_first_gig - income_third_gig 
  let hours_second_gig := 30 
  let hourly_rate := income_second_gig / hours_second_gig 
  hourly_rate = 20 := 
by 
  sorry

end dino_second_gig_hourly_rate_l369_369016


namespace sin_function_value_l369_369796

noncomputable def f (x : ℝ) : ℝ := sin(2 * x - 5 * π / 6)

theorem sin_function_value :
  f (-5 * π / 12) = sqrt 3 / 2 := by
  sorry

end sin_function_value_l369_369796


namespace remainder_M_mod_32_l369_369253

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_M_mod_32 : M % 32 = 17 :=
by {
  sorry
}

end remainder_M_mod_32_l369_369253


namespace remainder_of_M_when_divided_by_32_l369_369194

open Nat

def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
def M : ℕ := List.foldl (*) 1 oddPrimes

theorem remainder_of_M_when_divided_by_32 :
  M % 32 = 1 :=
sorry

end remainder_of_M_when_divided_by_32_l369_369194


namespace gcd_18_30_is_6_l369_369613

def gcd_18_30 : ℕ :=
  gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 :=
by {
  -- The step here will involve using properties of gcd and prime factorization,
  -- but we are given the result directly for the purpose of this task.
  sorry
}

end gcd_18_30_is_6_l369_369613


namespace system_of_equations_l369_369936

variables (x y : ℝ)

def cond1 : Prop := 4 * x + 6 * y = 48
def cond2 : Prop := 3 * x + 5 * y = 38

theorem system_of_equations (h1 : cond1) (h2 : cond2) :
  (4 * x + 6 * y = 48) ∧ (3 * x + 5 * y = 38) :=
by
  simp [h1, h2]
  sorry

end system_of_equations_l369_369936


namespace remainder_when_divided_by_32_l369_369160

def product_of_odds : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_divided_by_32 : product_of_odds % 32 = 9 := by
  sorry

end remainder_when_divided_by_32_l369_369160


namespace triangle_formation_l369_369112

theorem triangle_formation (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h₄ : c^2 = a^2 + b^2 + a * b) : 
  a + b > c ∧ a + c > b ∧ c + (a + b) > a :=
by
  sorry

end triangle_formation_l369_369112


namespace find_fx_value_l369_369719

noncomputable def f (x : Real) : Real :=
  sin (2 * x - 5 * Real.pi / 6)

theorem find_fx_value :
  (∀ x, f x = sin (2 * x - 5 * Real.pi / 6)) ∧ 
  (f (Real.pi / 6) = f (2 * Real.pi / 3)) ∧ 
  (∀ x1 x2, (Real.pi / 6 < x1 ∧ x1 < 2 * Real.pi / 3 ∧ x1 < x2 ∧ x2 < 2 * Real.pi / 3) -> f x1 < f x2) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 := by
  sorry

end find_fx_value_l369_369719


namespace find_journalist_l369_369979

axiom profession_different : ∀ (LZ ZB WD : Prop), LZ ≠ ZB ∧ ZB ≠ WD ∧ WD ≠ LZ
axiom only_one_journalist : ∀ (LZ JZ WD : Prop), (LZ ∨ JZ ∨ WD) ∧ ¬ (LZ ∧ JZ) ∧ ¬ (JZ ∧ WD) ∧ ¬ (WD ∧ LZ)

-- Statements
axiom LZ_statement : Prop
axiom ZB_statement : Prop
axiom WD_statement : Prop

-- The actual statements they made
axiom LZ_claim : LZ_statement ↔ LZ
axiom ZB_claim : ZB_statement ↔ ¬ ZB
axiom WD_claim : WD_statement ↔ ¬ LZ

-- Only one statement is true
axiom one_true_statement : (LZ_statement ∨ ZB_statement ∨ WD_statement) ∧
                            ¬(LZ_statement ∧ ZB_statement) ∧
                            ¬(ZB_statement ∧ WD_statement) ∧
                            ¬(WD_statement ∧ LZ_statement)

theorem find_journalist : ∀ (LZ JZ WD : Prop), ZB = JZ :=
begin
  sorry
end

end find_journalist_l369_369979


namespace value_of_f_neg_5π_over_12_l369_369791

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem value_of_f_neg_5π_over_12 :
  ∀ (ω φ : ℝ), 
    (∀ x y : ℝ, (x < y ∧ x ∈ Ioo (π / 6) (2 * π / 3) ∧ y ∈ Ioo (π / 6) (2 * π / 3)) → f x ω φ < f y ω φ) ∧ 
    f (π / 6) ω φ = f (2 * π / 3) ω φ → 
    f (-5 * π / 12) ω φ = √3 / 2 :=
by
  sorry

end value_of_f_neg_5π_over_12_l369_369791


namespace prime_product_mod_32_l369_369301

open Nat

theorem prime_product_mod_32 : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := List.foldr (· * ·) 1 primes
  M % 32 = 25 :=
by
  sorry

end prime_product_mod_32_l369_369301


namespace max_elements_in_A_l369_369309

theorem max_elements_in_A (n : ℕ) (h : 0 < n) :
  ∃ (A : set (set (fin n))), (∀ a b ∈ A, a ⊆ b → a = b) ∧ (A.card = nat.choose n (n / 2)) :=
by
  sorry

end max_elements_in_A_l369_369309


namespace unit_digit_G_1000_l369_369008

def G (n : ℕ) : ℕ := 3^(3^n) + 1

theorem unit_digit_G_1000 : (G 1000) % 10 = 2 :=
by
  sorry

end unit_digit_G_1000_l369_369008


namespace maximum_xy_value_l369_369865

theorem maximum_xy_value :
  ∃ (x y : ℕ), 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by
  sorry

end maximum_xy_value_l369_369865


namespace vector_relation_l369_369058

noncomputable theory

open_locale classical

-- Define the points and vectors
variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables (O A B C D : V)

-- Conditions given in the problem:
def midpoint (B C D : V) := 2 • D = B + C
def condition1 (O A B C : V) := 4 • O + B + C = (0 : V)

-- Given the assumptions
variables (h1 : midpoint B C D)
variables (h2 : 4 • (O - A) + (O - B) + (O - C) = 0)

-- Statement to be proved
theorem vector_relation : 2 • (A - O) = D - O :=
sorry

end vector_relation_l369_369058


namespace circle_problem_l369_369068

theorem circle_problem 
  (x y : ℝ)
  (h : x^2 + 8*x - 10*y = 10 - y^2 + 6*x) :
  let a := -1
  let b := 5
  let r := 6
  a + b + r = 10 :=
by sorry

end circle_problem_l369_369068


namespace gcd_of_18_and_30_l369_369546

theorem gcd_of_18_and_30 : Nat.gcd 18 30 = 6 :=
by
  sorry

end gcd_of_18_and_30_l369_369546


namespace g_domain_l369_369520

noncomputable def g (x : ℝ) : ℝ := Real.tan (Real.arcsin (x ^ 3))

theorem g_domain : {x : ℝ | -1 < x ∧ x < 1} = Set {x | ∃ y, g x = y} :=
by
  sorry

end g_domain_l369_369520


namespace balls_in_boxes_l369_369334

-- Define the conditions
def num_balls : ℕ := 3
def num_boxes : ℕ := 4

-- Define the problem
theorem balls_in_boxes : (num_boxes ^ num_balls) = 64 :=
by
  -- We acknowledge that we are skipping the proof details here
  sorry

end balls_in_boxes_l369_369334


namespace find_digits_sum_l369_369460

def digit_sum_valid (x y : ℕ) (h_x : 0 ≤ x ∧ x ≤ 9) (h_y : 0 ≤ y ∧ y ≤ 9) (h_diff : x ≠ y) : Prop :=
    (15.2 + 1.52 + 0.15 * x + y * 0.128 = 20) ∧ (x + y = 5)

theorem find_digits_sum :
  ∃ (x y : ℕ), digit_sum_valid x y (⟨Nat.zero_le 10, by decide⟩) (⟨Nat.zero_le 10, by decide⟩) (by decide) :=
sorry

end find_digits_sum_l369_369460


namespace sin_monotonic_increasing_f_symmetric_axes_find_value_l369_369700

def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi * 5 / 6)

theorem sin_monotonic_increasing (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) :
  Monotone (λ x, f x) (Ioo a b) := 
sorry

theorem f_symmetric_axes (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) 
  (x y : ℝ) (hx : x = a + (b - a) * k) (hy : y = a + (b - a) * (-k)) : 
  f x = f y :=
sorry

theorem find_value :
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
sorry

end sin_monotonic_increasing_f_symmetric_axes_find_value_l369_369700


namespace remainder_of_M_when_divided_by_32_l369_369193

open Nat

def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
def M : ℕ := List.foldl (*) 1 oddPrimes

theorem remainder_of_M_when_divided_by_32 :
  M % 32 = 1 :=
sorry

end remainder_of_M_when_divided_by_32_l369_369193


namespace product_of_odd_primes_mod_32_l369_369182

theorem product_of_odd_primes_mod_32 :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  M % 32 = 17 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  exact sorry

end product_of_odd_primes_mod_32_l369_369182


namespace find_f_neg_5pi_12_l369_369781

-- Conditions
def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

variables {ω φ : ℝ}
def f (x : ℝ) : ℝ := Real.sin (ω * x + φ)

axiom monotonicity_condition : is_monotonically_increasing f (π / 6) (2 * π / 3)
axiom symmetry_condition_1 : f (π / 6) = f (2 * π / 3)
axiom symmetry_condition_2 : f (-π / 6) = f (π / 2)

-- Goal
theorem find_f_neg_5pi_12 : f (- 5 * π / 12) = sqrt 3 / 2 := 
sorry

end find_f_neg_5pi_12_l369_369781


namespace calculate_expression_l369_369420

theorem calculate_expression :
  15^2 + 2 * 15 * 5 + 5^2 + 5^3 = 525 := 
sorry

end calculate_expression_l369_369420


namespace max_maximal_subsets_l369_369889

-- Define the concept of a maximal subset
def is_maximal_subset (S : Finset (ℝ × ℝ)) (T : Finset (ℝ × ℝ)) : Prop :=
  T ⊆ S ∧ ∀ (T' : Finset (ℝ × ℝ)), T' ⊆ S → (T ≠ T' → (T.sum (λ v, v) = T'.sum (λ v, v) → T.sum (λ v, v).norm < T'.sum (λ v, v).norm))

-- Main theorem statement
theorem max_maximal_subsets (S : Finset (ℝ × ℝ)) (n : ℕ) (hS : S.card = n) :
  (Finset.univ.filter (is_maximal_subset S)).card ≤ 2 * n :=
sorry

end max_maximal_subsets_l369_369889


namespace diagonal_of_larger_tv_l369_369376

-- Define the diagonal and areas of the televisions
def diagonal_19 : ℝ := 19
def area_19 : ℝ := (diagonal_19 / Real.sqrt 2) ^ 2
def area_larger : ℝ := area_19 + 40
def diagonal_larger : ℝ := Real.sqrt (2 * area_larger)

-- State the theorem to prove
theorem diagonal_of_larger_tv : diagonal_larger = 21 :=
by
  have h1 : area_19 = (diagonal_19 / Real.sqrt 2) ^ 2 := rfl
  have h2 : area_larger = area_19 + 40 := rfl
  have h3 : diagonal_larger = Real.sqrt (2 * area_larger) := rfl
  have h4 : diagonal_19 = 19 := rfl
  sorry

end diagonal_of_larger_tv_l369_369376


namespace exists_white_cell_never_colored_l369_369946

-- Definition of the grid and initial conditions
def grid := ℤ × ℤ  -- A grid is represented by integer coordinates

-- A finite set of initially black cells
def initially_black_cells (S : set grid) : Prop :=
  set.finite S

-- Definition of polygon M
def polygon_M (M : set grid) : Prop :=
  ∃ n, 1 < n ∧ finite M ∧ M.card = n  -- M covers more than one cell, it's finite.

-- Definition of the shifting rule
def shift (M : set grid) (d : grid) : set grid :=
  {p | ∃ q ∈ M, p = (q.1 + d.1, q.2 + d.2)}  -- Shifting M by the vector d

-- Shifting and coloring rule
def shift_and_color (S : set grid) (M : set grid) (pos : grid) : set grid :=
  if ∃ w ∈ shift M pos, w ∉ S then S ∪ {w} else S  -- "pos" is the vector by which M is shifted

theorem exists_white_cell_never_colored (S : set grid) (M : set grid) (pos : grid) :
  initially_black_cells S → polygon_M M → 
  ∃ w ∈ shift M pos, w ∉ S ∧ ∀ pos', ∃ w' ∈ shift M pos', w' ≠ w :=
sorry

end exists_white_cell_never_colored_l369_369946


namespace james_works_4_hours_l369_369950

/-- James does chores around the class, where:
- There are 3 bedrooms, 1 living room, and 2 bathrooms to clean.
- The bedrooms each take 20 minutes to clean.
- The living room takes as long as the 3 bedrooms combined.
- The bathroom takes twice as long as the living room.
- He also cleans the outside which takes twice as long as cleaning the house.
- He splits the chores with his 2 siblings who are just as fast as him.

Prove: James works 4 hours. -/
theorem james_works_4_hours : 
  let bedrooms := 3,
      living_room := 1,
      bathrooms := 2,
      clean_bedroom_time := 20,
      clean_living_room_time := 3 * clean_bedroom_time,
      clean_bathroom_time := 2 * clean_living_room_time,
      total_inside_time := bedrooms * clean_bedroom_time + living_room * clean_living_room_time + bathrooms * clean_bathroom_time,
      total_house_clean_time := total_inside_time / 60,
      clean_outside_time := 2 * total_house_clean_time,
      total_clean_time := total_house_clean_time + clean_outside_time,
      james_clean_time := total_clean_time / 3 in
  james_clean_time = 4 := 
by {
  let bedrooms := 3,
  let living_room := 1,
  let bathrooms := 2,
  let clean_bedroom_time := 20,
  let clean_living_room_time := 3 * clean_bedroom_time,
  let clean_bathroom_time := 2 * clean_living_room_time,
  let total_inside_time := bedrooms * clean_bedroom_time + living_room * clean_living_room_time + bathrooms * clean_bathroom_time,
  let total_house_clean_time := total_inside_time / 60,
  let clean_outside_time := 2 * total_house_clean_time,
  let total_clean_time := total_house_clean_time + clean_outside_time,
  let james_clean_time := total_clean_time / 3,
  exact sorry
}

end james_works_4_hours_l369_369950


namespace gcd_18_30_l369_369541

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l369_369541


namespace hamburgers_sold_in_winter_l369_369450

theorem hamburgers_sold_in_winter 
  (fall_sales : ℕ) (spring_sales : ℕ) (summer_sales : ℕ) (total_sales : ℕ) 
  (fall_percentage : Nat)
  (H1 : fall_sales = 4) 
  (H2 : fall_percentage = 25)
  (H3 : fall_sales = fall_percentage * total_sales / 100) 
  (H4 : spring_sales = 4.5) 
  (H5 : summer_sales = 5) 
  (total_sales_eqn : total_sales = spring_sales + summer_sales + fall_sales + winter_sales) :
  winter_sales = 2.5 :=
by
  sorry

end hamburgers_sold_in_winter_l369_369450


namespace perpendicular_line_through_point_l369_369533

noncomputable def is_perpendicular (m₁ m₂ : ℝ) : Prop :=
  m₁ * m₂ = -1

theorem perpendicular_line_through_point
  (line : ℝ → ℝ)
  (P : ℝ × ℝ)
  (h_line_eq : ∀ x, line x = 3 * x + 8)
  (hP : P = (2,1)) :
  ∃ a b c : ℝ, a * (P.1) + b * (P.2) + c = 0 ∧ is_perpendicular 3 (-b / a) ∧ a * 1 + b * 3 + c = 0 :=
sorry

end perpendicular_line_through_point_l369_369533


namespace sin_monotonic_increasing_f_symmetric_axes_find_value_l369_369701

def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi * 5 / 6)

theorem sin_monotonic_increasing (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) :
  Monotone (λ x, f x) (Ioo a b) := 
sorry

theorem f_symmetric_axes (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) 
  (x y : ℝ) (hx : x = a + (b - a) * k) (hy : y = a + (b - a) * (-k)) : 
  f x = f y :=
sorry

theorem find_value :
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
sorry

end sin_monotonic_increasing_f_symmetric_axes_find_value_l369_369701


namespace probability_of_getting_specific_clothing_combination_l369_369845

def total_articles := 21

def ways_to_choose_4_articles : ℕ := Nat.choose total_articles 4

def ways_to_choose_2_shirts_from_6 : ℕ := Nat.choose 6 2

def ways_to_choose_1_pair_of_shorts_from_7 : ℕ := Nat.choose 7 1

def ways_to_choose_1_pair_of_socks_from_8 : ℕ := Nat.choose 8 1

def favorable_outcomes := 
  ways_to_choose_2_shirts_from_6 * 
  ways_to_choose_1_pair_of_shorts_from_7 * 
  ways_to_choose_1_pair_of_socks_from_8

def probability := (favorable_outcomes : ℚ) / (ways_to_choose_4_articles : ℚ)

theorem probability_of_getting_specific_clothing_combination : 
  probability = 56 / 399 := by
  sorry

end probability_of_getting_specific_clothing_combination_l369_369845


namespace find_value_of_f_eq_l369_369765

noncomputable def f (ω ω φ x : ℝ) : ℝ := sin (ω * x + φ)

theorem find_value_of_f_eq :
  ∀ (ω φ : ℝ), 
    (∀ x, x ∈ set.Ioo (π / 6) (2 * π / 3) → (f ω φ x) ≤ (f ω φ (x + 1e-10))) → -- monotonically increasing
    (ω * ∓ (π / 6) + φ) = (φ + ω * (2 * π / 3)) → -- symmetric axes
    f ω φ (-(5 * π / 12)) = sqrt 3 / 2 :=
by
  sorry

end find_value_of_f_eq_l369_369765


namespace find_value_of_f_eq_l369_369761

noncomputable def f (ω ω φ x : ℝ) : ℝ := sin (ω * x + φ)

theorem find_value_of_f_eq :
  ∀ (ω φ : ℝ), 
    (∀ x, x ∈ set.Ioo (π / 6) (2 * π / 3) → (f ω φ x) ≤ (f ω φ (x + 1e-10))) → -- monotonically increasing
    (ω * ∓ (π / 6) + φ) = (φ + ω * (2 * π / 3)) → -- symmetric axes
    f ω φ (-(5 * π / 12)) = sqrt 3 / 2 :=
by
  sorry

end find_value_of_f_eq_l369_369761


namespace marc_total_spent_l369_369989

theorem marc_total_spent :
  let model_car_cost := 20
  let paint_cost := 10
  let brush_cost := 2
  let model_car_amount := 5
  let paint_amount := 5
  let brush_amount := 5
  let total_cost := model_car_amount * model_car_cost + paint_amount * paint_cost + brush_amount * brush_cost
  in total_cost = 160 :=
by
  sorry

end marc_total_spent_l369_369989


namespace find_fx_value_l369_369725

noncomputable def f (x : Real) : Real :=
  sin (2 * x - 5 * Real.pi / 6)

theorem find_fx_value :
  (∀ x, f x = sin (2 * x - 5 * Real.pi / 6)) ∧ 
  (f (Real.pi / 6) = f (2 * Real.pi / 3)) ∧ 
  (∀ x1 x2, (Real.pi / 6 < x1 ∧ x1 < 2 * Real.pi / 3 ∧ x1 < x2 ∧ x2 < 2 * Real.pi / 3) -> f x1 < f x2) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 := by
  sorry

end find_fx_value_l369_369725


namespace part_a_part_b_part_b_lambda_l369_369440

variable {ABC : Type} -- Triangle ABC
variable {M : Type} -- Point M inside the triangle

-- Conditions for Part (a)
noncomputable def sqrt_Ra_sqrt_Rb_sqrt_Rc_geq_sqrt2_sqrt_da_sqrt_db_sqrt_dc (R_a R_b R_c d_a d_b d_c : ℝ) : Prop :=
  sqrt R_a + sqrt R_b + sqrt R_c ≥ sqrt 2 * (sqrt d_a + sqrt d_b + sqrt d_c)

-- Statement for Part (a)
theorem part_a (ABC : Type) (M : Type) (R_a R_b R_c d_a d_b d_c : ℝ) :
  sqrt_Ra_sqrt_Rb_sqrt_Rc_geq_sqrt2_sqrt_da_sqrt_db_sqrt_dc R_a R_b R_c d_a d_b d_c :=
by sorry

-- Conditions for Part (b)
noncomputable def Ra2_Rb2_Rc2_gt_2_da2_db2_dc2 (R_a R_b R_c d_a d_b d_c : ℝ) : Prop :=
  R_a^2 + R_b^2 + R_c^2 > 2 * (d_a^2 + d_b^2 + d_c^2)

-- Additional check for existence of lambda
noncomputable def existence_lambda_gt_2 (R_a R_b R_c d_a d_b d_c : ℝ) : Prop :=
  ∃ (λ : ℝ), λ > 2 ∧ R_a^2 + R_b^2 + R_c^2 > λ * (d_a^2 + d_b^2 + d_c^2)

-- Statement for Part (b)
theorem part_b (ABC : Type) (M : Type) (R_a R_b R_c d_a d_b d_c : ℝ) :
  Ra2_Rb2_Rc2_gt_2_da2_db2_dc2 R_a R_b R_c d_a d_b d_c :=
by sorry

-- Existence of λ
theorem part_b_lambda (ABC : Type) (M : Type) (R_a R_b R_c d_a d_b d_c : ℝ) :
  existence_lambda_gt_2 R_a R_b R_c d_a d_b d_c :=
by sorry

end part_a_part_b_part_b_lambda_l369_369440


namespace log2_f2_eq_half_l369_369076

theorem log2_f2_eq_half (f : ℝ → ℝ) (h1 : ∀ x, f x = x^n)
  (h2 : f (1/2) = (√2/2)) : log 2 (f 2) = (1 / 2) :=
by
  sorry

end log2_f2_eq_half_l369_369076


namespace remainder_of_M_l369_369267

def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_of_M : M % 32 = 31 := by
  -- Proof goes here
  sorry

end remainder_of_M_l369_369267


namespace greatest_xy_value_l369_369875

theorem greatest_xy_value (x y : ℕ) (h1 : 7 * x + 4 * y = 140) (h2 : x > 0) (h3 : y > 0) : 
  xy ≤ 112 :=
by
  sorry

end greatest_xy_value_l369_369875


namespace find_f_neg_5pi_12_l369_369773

-- Conditions
def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

variables {ω φ : ℝ}
def f (x : ℝ) : ℝ := Real.sin (ω * x + φ)

axiom monotonicity_condition : is_monotonically_increasing f (π / 6) (2 * π / 3)
axiom symmetry_condition_1 : f (π / 6) = f (2 * π / 3)
axiom symmetry_condition_2 : f (-π / 6) = f (π / 2)

-- Goal
theorem find_f_neg_5pi_12 : f (- 5 * π / 12) = sqrt 3 / 2 := 
sorry

end find_f_neg_5pi_12_l369_369773


namespace num_ways_to_cut_shape_l369_369118

theorem num_ways_to_cut_shape : 
  ∃ n : ℕ, n = 10 ∧ (n = count_partition_ways shape_17_cells) :=
begin
  -- shape_17_cells represents the given 17 cells figure.
  sorry
end

end num_ways_to_cut_shape_l369_369118


namespace find_value_l369_369707

-- Define the function $f(x)$
def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

-- Define the problem statement in Lean 4
theorem find_value :
  (∀ x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), 0 ≤ 2 * x - 5 * Real.pi / 6 ∧ 2 * x - 5 * Real.pi / 6 ≤ Real.pi) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by
  -- A proper rigorous Lean proof should go here
  sorry

end find_value_l369_369707


namespace find_f_neg_5pi_12_l369_369780

-- Conditions
def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

variables {ω φ : ℝ}
def f (x : ℝ) : ℝ := Real.sin (ω * x + φ)

axiom monotonicity_condition : is_monotonically_increasing f (π / 6) (2 * π / 3)
axiom symmetry_condition_1 : f (π / 6) = f (2 * π / 3)
axiom symmetry_condition_2 : f (-π / 6) = f (π / 2)

-- Goal
theorem find_f_neg_5pi_12 : f (- 5 * π / 12) = sqrt 3 / 2 := 
sorry

end find_f_neg_5pi_12_l369_369780


namespace both_parents_single_is_sufficient_but_not_necessary_l369_369934

-- Definition of dominant and recessive genes
def dominant_allele := 'A'
def recessive_allele := 'a'

-- Genotype and phenotype definitions
def has_single_eyelids (genotype : list char) : Prop :=
  genotype = [recessive_allele, recessive_allele]

def has_double_eyelids (genotype : list char) : Prop :=
  genotype = [dominant_allele, dominant_allele] ∨ genotype = [dominant_allele, recessive_allele] ∨ genotype = [recessive_allele, dominant_allele]

-- Conditions given in the problem
variable (parent1_genotype parent2_genotype : list char)
variable (child_genotype : list char)
variable (parents_genotype_single : has_single_eyelids parent1_genotype ∧ has_single_eyelids parent2_genotype)
variable (child_inherits_genes : child_genotype = if (list.head parent1_genotype = dominant_allele) then [dominant_allele] else [recessive_allele] ++ if (list.head parent2_genotype = dominant_allele) then [dominant_allele] else [recessive_allele])

-- Proof problem statement
theorem both_parents_single_is_sufficient_but_not_necessary :
  (has_single_eyelids parent1_genotype ∧ has_single_eyelids parent2_genotype → has_single_eyelids child_genotype)
  ∧ (¬(has_single_eyelids parent1_genotype ∧ has_single_eyelids parent2_genotype) ∧ has_single_eyelids child_genotype) :=
sorry

end both_parents_single_is_sufficient_but_not_necessary_l369_369934


namespace remainder_M_mod_32_l369_369255

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_M_mod_32 : M % 32 = 17 :=
by {
  sorry
}

end remainder_M_mod_32_l369_369255


namespace solution_set_inequality_x0_1_solution_set_inequality_x0_half_l369_369678

noncomputable def f (x : ℝ) : ℝ := abs (Real.log x)

theorem solution_set_inequality_x0_1 : 
  ∀ (c : ℝ), (∀ x, 0 < x → f x - f 1 ≥ c * (x - 1)) ↔ c ∈ Set.Icc (-1) 1 := 
by
  sorry

theorem solution_set_inequality_x0_half : 
  ∀ (c : ℝ), (∀ x, 0 < x → f x - f (1 / 2) ≥ c * (x - 1 / 2)) ↔ c = -2 :=
by
  sorry

end solution_set_inequality_x0_1_solution_set_inequality_x0_half_l369_369678


namespace square_area_from_diagonal_l369_369409

theorem square_area_from_diagonal (d : ℝ) (h : d = 10) : ∃ (A : ℝ), A = 50 := by
  let s := Real.sqrt (d^2 / 2)
  have hs : s^2 = 50 := by
    calc
      s^2 = (Real.sqrt (d^2 / 2))^2 : by rfl
      ... = d^2 / 2 : Real.sqrt_sq (by linarith)
      ... = 100 / 2 : by rw [h]; norm_num
      ... = 50 : by norm_num
  use s^2
  rwa [hs]

end square_area_from_diagonal_l369_369409


namespace find_a7_l369_369967

variable {a : ℕ → ℝ} {d : ℝ}

-- Conditions on the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Geometric sequence condition
def forms_geometric_sequence (a4 a5 a7 : ℝ) : Prop :=
  a5^2 = a4 * a7

-- Sum of the first n terms of an arithmetic sequence
def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

-- Main statement
theorem find_a7 (h_arith: is_arithmetic_sequence a d) (hd : d ≠ 0)
  (h_geo: forms_geometric_sequence (a 4) (a 5) (a 7))
  (h_sum11 : sum_of_first_n_terms a 11 = 66) : 
  a 7 = 8 :=
by
  sorry

end find_a7_l369_369967


namespace max_xy_l369_369860

theorem max_xy (x y : ℕ) (h1: 7 * x + 4 * y = 140) : ∃ x y, 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by {
  sorry
}

end max_xy_l369_369860


namespace problem_statement_l369_369687

-- Definition of our function f
def f (x : ℝ) : ℝ := Real.sin (2 * x - (5 * Real.pi / 6))

-- Theorem to prove that f(-5π/12) = √3/2
theorem problem_statement : 
  f(-5 * Real.pi / 12) = sqrt 3 / 2 := 
sorry

end problem_statement_l369_369687


namespace remainder_of_product_of_odd_primes_mod_32_l369_369283

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem remainder_of_product_of_odd_primes_mod_32 :
  (product odd_primes_less_than_32) % 32 = 23 :=
by sorry

end remainder_of_product_of_odd_primes_mod_32_l369_369283


namespace part1_expr_calc_part2_inequalities_l369_369444

-- Calculating the value of the given expression
theorem part1_expr_calc : 
  (-1 : ℤ) ^ 2023 + (18 : ℚ) / ((-1/3 : ℚ) ^ (-2 : ℤ)) - (Real.sqrt (3/2 : ℚ) * Real.sqrt (6 : ℕ)) = -2 :=
  sorry

-- Solving the system of inequalities
theorem part2_inequalities : 
  ∀ x : ℤ, 4 - 3 * x > - x ∧ 1 + x / 2 > -1/3 ↔ x ∈ {-2, -1, 0, 1} :=
  sorry

end part1_expr_calc_part2_inequalities_l369_369444


namespace S_13_constant_l369_369052

variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (a_1 : ℝ) (d : ℝ)

-- condition: Arithmetic sequence definition
def is_arithmetic_sequence (a : ℕ → ℝ) (a_1 d : ℝ) : Prop :=
  ∀ n, a n = a_1 + n * d

-- condition: Sum of first n terms of arithmetic sequence
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 0 + a (n - 1)) / 2

-- condition: Given S_4 + a_{25} = 5
axiom given_condition (a : ℕ → ℝ) (S : ℕ → ℝ) (a_1 d : ℝ) : 
  sum_first_n_terms a 4 + a 25 = 5

-- Prove that S_{13} is a constant value
theorem S_13_constant (a : ℕ → ℝ) (S : ℕ → ℝ) (a_1 d : ℝ) 
  [is_arithmetic_sequence a a_1 d] [given_condition a S a_1 d] : 
  S 13 = 13 := 
sorry

end S_13_constant_l369_369052


namespace number_of_proper_subsets_l369_369371

open Set

def properSubsetsCount (S : Set ℕ) : ℕ :=
2 ^ S.size - 1

theorem number_of_proper_subsets :
  let S := { x : ℕ | -1 ≤ log 10 (1 / x : ℝ) ∧ log 10 (1 / (x : ℝ)) < -1/2 }
  (properSubsetsCount S) = (2 ^ 90 - 1) := by
suffices h : S = { x : ℕ | 10 ≤ x ∧ x < 100 }
· simp [properSubsetsCount, card_eq_fintype_card, Fintype.card_fin, h]
sorry

end number_of_proper_subsets_l369_369371


namespace cotangent_identity_l369_369629

variables {A B C : Type} [AddGroup A] [AddGroup B] [AddGroup C]

-- Definitions of the triangle
def triangle (a b c : ℝ) := a < b ∧ b < c

-- Definitions of the medians
def median_sides (a b c sa sb sc : ℝ) : Prop :=
  True -- Placeholder for actual median definitions

-- Definitions of the angles δ, ε and ζ
def median_angles (δ ε ζ : ℝ) : Prop :=
  True -- Placeholder for actual angle definitions

-- The proof statement
theorem cotangent_identity
  (a b c sa sb sc δ ε ζ : ℝ)
  (h1 : triangle a b c)
  (h2 : median_sides a b c sa sb sc)
  (h3 : median_angles δ ε ζ) :
  real.cot ε = real.cot δ + real.cot ζ :=
sorry

end cotangent_identity_l369_369629


namespace right_triangle_largest_angle_l369_369930

theorem right_triangle_largest_angle (a b : ℝ) (h1 : a = 3/5 * 90) (h2 : b = 2/5 * 90) (right_angle : a + b = 90) : ∃ γ, γ = 90 :=
by 
  use 90
  exact right_angle
  sorry

end right_triangle_largest_angle_l369_369930


namespace recurrence_relation_p_series_l369_369907

noncomputable def p_series (n k : ℕ) : ℝ :=
if k < n then (p_series (n - 1) k - (1 / (2 : ℝ)^k) * p_series (n - k) k + (1 / (2 : ℝ)^k))
else 0

-- Statement of the theorem
theorem recurrence_relation_p_series (n k : ℕ) (h : k < n) :
  p_series n k = p_series (n - 1) k - (1 / (2 : ℝ)^k) * p_series (n - k) k + (1 / (2 : ℝ)^k) :=
sorry

end recurrence_relation_p_series_l369_369907


namespace loss_percentage_l369_369844

/-
Books Problem:
Determine the loss percentage on the first book given:
1. The cost of the first book (C1) is Rs. 280.
2. The total cost of two books is Rs. 480.
3. The second book is sold at a gain of 19%.
4. Both books are sold at the same price.
-/

theorem loss_percentage (C1 C2 SP1 SP2 : ℝ) (h1 : C1 = 280)
  (h2 : C1 + C2 = 480) (h3 : SP2 = C2 * 1.19) (h4 : SP1 = SP2) : 
  (C1 - SP1) / C1 * 100 = 15 := 
by
  sorry

end loss_percentage_l369_369844


namespace farm_growing_corn_in_two_fields_l369_369461

noncomputable theory

def num_fields (rows_field1 rows_field2 cobs_per_row total_cobs : ℕ) : ℕ :=
  if (rows_field1 * cobs_per_row + rows_field2 * cobs_per_row) = total_cobs then 2 else sorry

theorem farm_growing_corn_in_two_fields :
  num_fields 13 16 4 116 = 2 :=
by
  unfold num_fields
  simp
  sorry

end farm_growing_corn_in_two_fields_l369_369461


namespace area_BDE_l369_369331

noncomputable def points_in_space : Type := sorry

noncomputable def distances (A B C D E : points_in_space) : Prop :=
  dist A B = 3 ∧ dist B C = 3 ∧ dist C D = 3 ∧ dist D E = 3 ∧ dist E A = 3

noncomputable def angles (A B C D E : points_in_space) : Prop :=
  angle A B C = 90 ∧ angle C D E = 90 ∧ angle D E A = 120

noncomputable def plane_perpendicular (A B C D E : points_in_space) : Prop :=
  plane A B C perp_line D E

noncomputable def area_triangle (B D E : points_in_space) : ℝ :=
  (dist B D) * (dist B E) / 2

theorem area_BDE (A B C D E : points_in_space)
  (h1 : distances A B C D E)
  (h2 : angles A B C D E)
  (h3 : plane_perpendicular A B C D E) :
  area_triangle B D E = 9 * sqrt 2 / 2 := sorry

end area_BDE_l369_369331


namespace tax_refund_three_years_l369_369408

-- Define the conditions and the question
def annual_max_deduction : ℝ := 200_000
def tax_rate : ℝ := 0.13
def years: ℝ := 3

-- Define the calculation of refund
def annual_tax_refund (annual_max_deduction : ℝ) (tax_rate : ℝ) : ℝ :=
  annual_max_deduction * tax_rate

def total_tax_refund (annual_tax_refund : ℝ) (years : ℝ) : ℝ :=
  annual_tax_refund * years

-- The statement to prove
theorem tax_refund_three_years : total_tax_refund (annual_tax_refund annual_max_deduction tax_rate) years = 78_000 :=
  sorry

end tax_refund_three_years_l369_369408


namespace greatest_xy_value_l369_369873

theorem greatest_xy_value (x y : ℕ) (h1 : 7 * x + 4 * y = 140) (h2 : x > 0) (h3 : y > 0) : 
  xy ≤ 112 :=
by
  sorry

end greatest_xy_value_l369_369873


namespace max_product_l369_369878

theorem max_product (x y : ℕ) (h1 : 7 * x + 4 * y = 140) : x * y ≤ 168 :=
sorry

end max_product_l369_369878


namespace problem_statement_l369_369745

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - 5 * Real.pi / 6)

theorem problem_statement :
  (∀ (x : ℝ), (∀ a b : ℝ, a < b → x ∈ (Set.Ioc (Real.pi / 6) (2 * Real.pi / 3)) → f(x) < f(b)) → x = (Real.pi / 6) ∨ x = (2 * Real.pi / 3)) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by sorry

end problem_statement_l369_369745


namespace smallest_n_l369_369624

-- Define the conditions as predicates
def condition1 (n : ℕ) : Prop := (n + 2018) % 2020 = 0
def condition2 (n : ℕ) : Prop := (n + 2020) % 2018 = 0

-- The main theorem statement using these conditions
theorem smallest_n (n : ℕ) : 
  (∃ n, condition1 n ∧ condition2 n ∧ (∀ m, condition1 m ∧ condition2 m → n ≤ m)) ↔ n = 2030102 := 
by 
    sorry

end smallest_n_l369_369624


namespace gcd_18_30_l369_369608

-- Define the two numbers
def num1 : ℕ := 18
def num2 : ℕ := 30

-- State the theorem to find the gcd
theorem gcd_18_30 : Nat.gcd num1 num2 = 6 := by
  sorry

end gcd_18_30_l369_369608


namespace prove_R_m_formula_l369_369973

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
nat.choose n k

def R_m (n m M : ℕ) : ℝ :=
let C_M_n := binomial_coefficient M n in
let sum_part := ∑ k in finset.range (M - m + 1), 
  (-1)^k * binomial_coefficient (M - m) k * (1 - (m + k) / M)^n in
C_M_n * sum_part

theorem prove_R_m_formula {n m M : ℕ} (h1 : M ≥ m) :
  R_m n m M = binomial_coefficient M n * 
    ∑ k in finset.range (M - m + 1), 
      (-1)^k * binomial_coefficient (M - m) k * (1 - (m + k) / M)^n :=
sorry

end prove_R_m_formula_l369_369973


namespace James_works_6_hours_l369_369952

variable (bedrooms : ℕ) (living_room : ℕ) (bathrooms : ℕ) (time_per_bedroom : ℕ) 
variable (James_siblings : ℕ)

-- Definitions of given conditions
def time_for_cleaning_house : ℕ :=
  let time_for_bedrooms := bedrooms * time_per_bedroom
  let time_for_living_room := time_for_bedrooms
  let time_for_bathroom := 2 * time_for_living_room
  let total_inside_time :=  time_for_bedrooms + time_for_living_room + 2 * time_for_bathroom
  total_inside_time

def total_time_for_chores : ℕ :=
  let total_inside_time := time_for_cleaning_house bedrooms living_room bathrooms time_per_bedroom
  2 * total_inside_time + total_inside_time

def total_minutes_per_sibling : ℕ :=
  (total_time_for_chores bedrooms living_room bathrooms time_per_bedroom) / (James_siblings + 1)

def total_hours_per_sibling : ℕ :=
  total_minutes_per_sibling bedrooms living_room bathrooms time_per_bedroom James_siblings / 60

-- Prove that James works for 6 hours
theorem James_works_6_hours :
  bedrooms = 3 ∧ living_room = 1 ∧ bathrooms = 2 ∧ time_per_bedroom = 20 ∧ James_siblings = 2 →
  total_hours_per_sibling bedrooms living_room bathrooms time_per_bedroom James_siblings = 6 := by
  intros
  exact sorry

end James_works_6_hours_l369_369952


namespace remainder_M_mod_32_l369_369254

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_M_mod_32 : M % 32 = 17 :=
by {
  sorry
}

end remainder_M_mod_32_l369_369254


namespace find_value_l369_369708

-- Define the function $f(x)$
def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

-- Define the problem statement in Lean 4
theorem find_value :
  (∀ x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), 0 ≤ 2 * x - 5 * Real.pi / 6 ∧ 2 * x - 5 * Real.pi / 6 ≤ Real.pi) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by
  -- A proper rigorous Lean proof should go here
  sorry

end find_value_l369_369708


namespace part1_integer_solutions_l369_369445

def integerSolutionsInequality : set ℤ :=
  { x | -2 ≤ (1 + 2 * x) / 3 ∧ (1 + 2 * x) / 3 ≤ 2 }

theorem part1_integer_solutions (x : ℤ) :
  x ∈ integerSolutionsInequality ↔ x = -3 ∨ x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2 := sorry

end part1_integer_solutions_l369_369445


namespace distance_traveled_l369_369441

-- Define constants for speed and time
def speed : ℝ := 60
def time : ℝ := 5

-- Define the expected distance
def expected_distance : ℝ := 300

-- Theorem statement
theorem distance_traveled : speed * time = expected_distance :=
by
  sorry

end distance_traveled_l369_369441


namespace gcd_18_30_l369_369610

-- Define the two numbers
def num1 : ℕ := 18
def num2 : ℕ := 30

-- State the theorem to find the gcd
theorem gcd_18_30 : Nat.gcd num1 num2 = 6 := by
  sorry

end gcd_18_30_l369_369610


namespace gcd_18_30_l369_369607

-- Define the two numbers
def num1 : ℕ := 18
def num2 : ℕ := 30

-- State the theorem to find the gcd
theorem gcd_18_30 : Nat.gcd num1 num2 = 6 := by
  sorry

end gcd_18_30_l369_369607


namespace triangle_congruence_and_nine_point_circle_coincidence_l369_369945

-- Definitions based on given conditions
variable {A B C O H A1 B1 C1 : Type}

-- Assume the properties of the points involved and triangle properties
variables [Triangle ABC]
variables (Circumcenter O ABC) (Orthocenter H ABC)
variables (Circumcenter A1 (Triangle.mk C H B)) (Circumcenter B1 (Triangle.mk C H A)) (Circumcenter C1 (Triangle.mk A H B))

-- The main theorem statements
theorem triangle_congruence_and_nine_point_circle_coincidence :
  (Congruent (Triangle.mk A B C) (Triangle.mk A1 B1 C1)) ∧ (NinePointCircleCoincidence (Triangle.mk A B C) (Triangle.mk A1 B1 C1)) := 
sorry

end triangle_congruence_and_nine_point_circle_coincidence_l369_369945


namespace gcd_18_30_l369_369573

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := 
by
  sorry

end gcd_18_30_l369_369573


namespace sum_rel_prime_l369_369427

def product_series (n : ℕ) : ℚ :=
  ∏ k in finset.range n + 1, (1 + 1 / (1 + 2^k))

def f (n : ℕ) : ℚ :=
  (2 ^ (n + 1)) / (2 ^ n + 1)

theorem sum_rel_prime (n : ℕ) (prod_eq : product_series n = f n) (h : nat.coprime (2^11) (2^10 + 1)) :
  (2 ^ (n + 1)) + (2 ^ n + 1) = 3073 := by
  have prod_val : product_series 10 = 2^11 / (2^10 + 1) := prod_eq
  rw [prod_val]
  simp
  sorry

end sum_rel_prime_l369_369427


namespace recurrence_relation_l369_369897

variables {n k : ℕ}

def p : ℕ → ℕ → ℚ := sorry

theorem recurrence_relation (n k : ℕ) (hnk : n ≥ k) :
  p n k = p (n-1) k - (1 / (2^k)) * p (n-k) k + (1 / (2^k)) :=
begin
  sorry
end

end recurrence_relation_l369_369897


namespace find_f_of_neg_5_pi_over_12_l369_369737

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - (5 * π / 6))

theorem find_f_of_neg_5_pi_over_12 :
  (∀ x ∈ (set.Ioo (π / 6) (2 * π / 3)), (0 : ℝ) < (f x - f (x - 1))) ∧
  (∀ x, f (π / 6 - x) = f (π / 6 + x)) ∧
  (∀ x, f (2 * π / 3 - x) = f (2 * π / 3 + x)) →
  f (-5 * π / 12) = √3 / 2 :=
by
  sorry

end find_f_of_neg_5_pi_over_12_l369_369737


namespace f_is_even_l369_369364

noncomputable def f (x : ℝ) : ℝ := x ^ 2

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := 
by
  intros x
  sorry

end f_is_even_l369_369364


namespace RobertoSkippingRatePerHour_l369_369341

namespace SkippingProblem

-- Definitions based on the conditions
def valerieSkipRatePerMinute : ℕ := 80
def totalSkipsInFifteenMinutes : ℕ := 2250

-- Theorem statement
theorem RobertoSkippingRatePerHour : 
  (let valerieSkips := valerieSkipRatePerMinute * 15 in
   let robertoSkipsInFifteenMinutes := totalSkipsInFifteenMinutes - valerieSkips in
   let robertoSkipsInAnHour := robertoSkipsInFifteenMinutes * 4 in
   robertoSkipsInAnHour = 4200) := 
  by
    sorry

end SkippingProblem

end RobertoSkippingRatePerHour_l369_369341


namespace homothety_center_similarity_circle_l369_369964

-- Conditions
variables {F₁ F₂ F₃ : Type} [figure F₁] [figure F₂] [figure F₃]

variables {A₁ B₁ C₁ A₂ B₂ C₂ A₃ B₃ C₃ : Type}
[h₁ : segment_prop A₁ B₁ C₁ F₁] [h₂ : segment_prop A₂ B₂ C₂ F₂]
[h₃ : segment_prop A₃ B₃ C₃ F₃]

-- Definitions
def similar (P Q R : Type) : Prop := sorry
def circumcircle (P Q R : Type) : Type := sorry
def segment_prop (A B C : Type) (F : Type) : Prop := sorry

-- Theorem statement
theorem homothety_center_similarity_circle :
  similar (A₁ B₁) (A₂ B₂) (A₃ B₃) ∧
  similar (A₁ C₁) (A₂ C₂) (A₃ C₃) →
  (∃ O, (homothety_center (A₁ B₁) (A₁ C₁) = O) ∧
        (O ∈ circumcircle (F₁ F₂ F₃))) := sorry


end homothety_center_similarity_circle_l369_369964


namespace math_problem_l369_369636

noncomputable def a := log (9 : ℝ) 45
noncomputable def b := log (9 : ℝ) 5

theorem math_problem : a - b = 1 :=
sorry

end math_problem_l369_369636


namespace Bernoulli_expected_value_l369_369313

variable {p : ℝ} {X : ℕ → ℝ}

noncomputable def Bernoulli (p : ℝ) (k : ℕ) : ℝ :=
  if k = 0 then (1 - p) else p

theorem Bernoulli_expected_value (h₀ : 0 < p) (h₁ : p < 1) :
  E(X) = ∑ k in {0, 1}, Bernoulli p k * k = 1 - p :=
by
  sorry

end Bernoulli_expected_value_l369_369313


namespace recurrence_relation_l369_369900

noncomputable def p (n k : ℕ) : ℚ := sorry

theorem recurrence_relation (n k : ℕ) (h : k < n) :
  p n k = p (n - 1) k - (1 / 2^k) * p (n - k) k + (1 / 2^k) :=
by sorry

end recurrence_relation_l369_369900


namespace train_cross_time_l369_369489

theorem train_cross_time (train_length : ℝ) (platform_length : ℝ) (speed_kmph : ℝ) : 
  train_length = 120 → 
  platform_length = 130.02 → 
  speed_kmph = 60 → 
  (train_length + platform_length) / (speed_kmph * 1000 / 3600) = 15 :=
by
  intros h_train h_platform h_speed
  rw [h_train, h_platform, h_speed]
  have total_distance : ℝ := 120 + 130.02
  have speed_mps : ℝ := 60 * 1000 / 3600
  have h1 : (total_distance / speed_mps) = 15
  {
    calc
      (250.02 / (60 * 1000 / 3600)) = (250.02 / (60 * (5 / 18))) : rfl
      ... = (250.02 / 16.67) : by norm_num
      ... = 15 : by norm_num
  }
  exact h1

end train_cross_time_l369_369489


namespace find_f_of_2x1_eq_6x_minus_2_l369_369826

variable (f : ℝ → ℝ)

axiom h : ∀ x : ℝ, f(2*x + 1) = 6*x - 2

theorem find_f_of_2x1_eq_6x_minus_2 : ∀ x : ℝ, f(x) = 3*x - 5 :=
begin
  intro x,
  have h1 : f (2 * (x / 2 - 1 / 2 + 1 / 2)) = 6 * (x / 2 - 1 / 2 + 1 / 2) - 2,
    calc f (2 * (x / 2 - 1 / 2 + 1 / 2)) = f (2 * x / 2) : by rw [mul_sub, mul_add, mul_one, sub_add_cancel, add_left_comm, sub_add_cancel, mul_one, add_left_comm, div_mul_cancel, by norm_num1]
    ... = 6 * (x / 2 - 1 / 2 + 1 / 2) - 2 : by rw h,
  rw [mul_sub, mul_add, mul_one, sub_add_cancel, add_left_comm, sub_add_cancel, mul_one, add_left_comm, div_mul_cancel] at h1,
  exact h1,
end

end find_f_of_2x1_eq_6x_minus_2_l369_369826


namespace area_of_sector_l369_369669

theorem area_of_sector {R θ: ℝ} (hR: R = 2) (hθ: θ = (2 * Real.pi) / 3) :
  (1 / 2) * R^2 * θ = (4 / 3) * Real.pi :=
by
  simp [hR, hθ]
  norm_num
  linarith

end area_of_sector_l369_369669


namespace sum_distances_l369_369836

theorem sum_distances (r a : ℝ) (h1 : 2 * a < r / 2) 
  (M N : ℝ × ℝ) (dM dN : ℝ) (hM : (dM / |(M.fst - (r + a))^) = 1) 
  (hN : (dN / |(N.fst - (r + a))^) = 1) 
  (h_dist1 : ∀ x, sqrt ((x - (r + a))^2 + hM * dM^2) = r) 
  (h_dist2 : ∀ x, sqrt ((x - (r + a))^2 + hN * dN^2) = r) :
  |(M.fst - (r + a))| + |(N.fst - (r + a))| = 2 * r := 
by sorry

end sum_distances_l369_369836


namespace line_parabola_intersect_at_one_point_l369_369366

theorem line_parabola_intersect_at_one_point (a : ℝ) :
  (y = ax - 6) → (y = x^2 + 4x + 3) → ∃ x : ℝ, (x^2 + (4-a)x + 9 = 0) →
  (∃! x, (x^2 + (4-a)x + 9 = 0)) ↔ (a = -2 ∨ a = 10) :=
by
  intro line_eq parabola_eq unique_x
  sorry

end line_parabola_intersect_at_one_point_l369_369366


namespace gcd_of_18_and_30_l369_369598

-- Define the numbers
def num1 := 18
def num2 := 30

-- State the GCD property
theorem gcd_of_18_and_30 : Nat.gcd num1 num2 = 6 :=
by
  sorry

end gcd_of_18_and_30_l369_369598


namespace find_f_value_l369_369818

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

theorem find_f_value :
  f ( -5 * Real.pi / 12 ) = Real.sqrt 3 / 2 :=
by
  sorry

end find_f_value_l369_369818


namespace total_area_of_field_l369_369464

noncomputable def total_field_area (A1 A2 : ℝ) : ℝ := A1 + A2

theorem total_area_of_field :
  ∀ (A1 A2 : ℝ),
    A1 = 405 ∧ (A2 - A1 = (1/5) * ((A1 + A2) / 2)) →
    total_field_area A1 A2 = 900 :=
by
  intros A1 A2 h
  sorry

end total_area_of_field_l369_369464


namespace proof_problem_l369_369044

def math_problem_conditions (alpha beta : ℝ) :=
  (0 < alpha ∧ alpha < (π / 2)) ∧
  ((π / 2) < beta ∧ beta < π) ∧
  (cos beta = -1 / 3) ∧
  (sin (alpha + beta) = (4 - sqrt 2) / 6)

theorem proof_problem (alpha beta : ℝ) (h : math_problem_conditions alpha beta) :
  tan (2 * beta) = (4 * sqrt 2) / 7 ∧ alpha = π / 4 :=
by
  sorry

end proof_problem_l369_369044


namespace product_of_odd_primes_mod_32_l369_369226

theorem product_of_odd_primes_mod_32 :
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31) in
  M % 32 = 25 :=
by
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31)
  sorry

end product_of_odd_primes_mod_32_l369_369226


namespace find_f_neg_5pi_12_l369_369779

-- Conditions
def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

variables {ω φ : ℝ}
def f (x : ℝ) : ℝ := Real.sin (ω * x + φ)

axiom monotonicity_condition : is_monotonically_increasing f (π / 6) (2 * π / 3)
axiom symmetry_condition_1 : f (π / 6) = f (2 * π / 3)
axiom symmetry_condition_2 : f (-π / 6) = f (π / 2)

-- Goal
theorem find_f_neg_5pi_12 : f (- 5 * π / 12) = sqrt 3 / 2 := 
sorry

end find_f_neg_5pi_12_l369_369779


namespace find_value_l369_369711

-- Define the function $f(x)$
def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

-- Define the problem statement in Lean 4
theorem find_value :
  (∀ x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), 0 ≤ 2 * x - 5 * Real.pi / 6 ∧ 2 * x - 5 * Real.pi / 6 ≤ Real.pi) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by
  -- A proper rigorous Lean proof should go here
  sorry

end find_value_l369_369711


namespace product_of_odd_primes_mod_32_l369_369221

theorem product_of_odd_primes_mod_32 :
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31) in
  M % 32 = 25 :=
by
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31)
  sorry

end product_of_odd_primes_mod_32_l369_369221


namespace greatest_xy_value_l369_369871

theorem greatest_xy_value (x y : ℕ) (h1 : 7 * x + 4 * y = 140) (h2 : x > 0) (h3 : y > 0) : 
  xy ≤ 112 :=
by
  sorry

end greatest_xy_value_l369_369871


namespace prove_n_eq_23_l369_369134

theorem prove_n_eq_23 
  (n : ℕ) (k : ℕ) 
  (h1 : odd n) 
  (h2 : n > 11) 
  (h3 : k ≥ 6) 
  (h4 : n = 2 * k - 1) 
  (T : set (vector bool n)) 
  (d : vector bool n → vector bool n → ℕ) 
  (S : set (vector bool n)) 
  (h5 : ∀ x y : vector bool n, d x y = (x.zip y).countp (λ (p : bool × bool), p.fst ≠ p.snd)) 
  (h6 : S ⊆ T) 
  (h7 : ∃ S : set (vector bool n), S ⊆ T ∧ |S| = 2 ^ k ∧ ∀ x ∈ T, ∃ y ∈ S, d x y ≤ 3) : 
  n = 23 := 
sorry

end prove_n_eq_23_l369_369134


namespace remainder_of_M_l369_369273

def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_of_M : M % 32 = 31 := by
  -- Proof goes here
  sorry

end remainder_of_M_l369_369273


namespace product_mod_32_l369_369210

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  M % 32 = 17 := by
  sorry

end product_mod_32_l369_369210


namespace FM_tangent_to_EMB_circumcircle_l369_369958

-- The following definitions are directly derived from the problem conditions
variables {A B C D E F M : Type}
variables (BC AC : ℝ) [hAC: AC > BC]

-- Define the entire problem as a Lean proposition
theorem FM_tangent_to_EMB_circumcircle (hABC : ∀ M : Type, ∀ hM, hABC : ∀ M : Type, ∀ hM, 
  is_acute_triangle ABC) 
  (hM : midpoint M A B) 
  (hCD : altitude C D A B) 
  (hE : perpendicular AE CM) 
  (hF : midpoint F C D) :
  tangent FM (circumcircle E M B) :=
sorry

end FM_tangent_to_EMB_circumcircle_l369_369958


namespace max_load_per_truck_l369_369352

-- Definitions based on given conditions
def num_trucks : ℕ := 3
def total_boxes : ℕ := 240
def lighter_box_weight : ℕ := 10
def heavier_box_weight : ℕ := 40

-- Proof problem statement
theorem max_load_per_truck :
  (total_boxes / 2) * lighter_box_weight + (total_boxes / 2) * heavier_box_weight = 6000 →
  6000 / num_trucks = 2000 :=
by sorry

end max_load_per_truck_l369_369352


namespace three_x_power_x_l369_369846

theorem three_x_power_x (x : ℝ) (h : 8^x - 8^(x-1) = 56) : (3 * x)^x = 36 :=
by
  sorry

end three_x_power_x_l369_369846


namespace find_value_of_f_eq_l369_369767

noncomputable def f (ω ω φ x : ℝ) : ℝ := sin (ω * x + φ)

theorem find_value_of_f_eq :
  ∀ (ω φ : ℝ), 
    (∀ x, x ∈ set.Ioo (π / 6) (2 * π / 3) → (f ω φ x) ≤ (f ω φ (x + 1e-10))) → -- monotonically increasing
    (ω * ∓ (π / 6) + φ) = (φ + ω * (2 * π / 3)) → -- symmetric axes
    f ω φ (-(5 * π / 12)) = sqrt 3 / 2 :=
by
  sorry

end find_value_of_f_eq_l369_369767


namespace recurrence_relation_l369_369912

-- Define the function p_nk and prove the recurrence relation
def p (n k : ℕ) : ℝ := sorry

theorem recurrence_relation (n k : ℕ) (h : k < n) : 
  p n k = p (n-1) k - (1 / 2^k) * p (n-k) k + (1 / 2^k) :=
sorry

end recurrence_relation_l369_369912


namespace find_f_of_neg_5_pi_over_12_l369_369741

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - (5 * π / 6))

theorem find_f_of_neg_5_pi_over_12 :
  (∀ x ∈ (set.Ioo (π / 6) (2 * π / 3)), (0 : ℝ) < (f x - f (x - 1))) ∧
  (∀ x, f (π / 6 - x) = f (π / 6 + x)) ∧
  (∀ x, f (2 * π / 3 - x) = f (2 * π / 3 + x)) →
  f (-5 * π / 12) = √3 / 2 :=
by
  sorry

end find_f_of_neg_5_pi_over_12_l369_369741


namespace half_angle_tangent_inequality_l369_369108

theorem half_angle_tangent_inequality 
  {A B C : ℝ} (h : A + B + C = real.pi) 
  : real.tan (B / 2) * real.tan (C / 2) ≤ (1 - real.sin (A / 2))^2 / (real.cos (A / 2))^2 := 
begin
  sorry
end

end half_angle_tangent_inequality_l369_369108


namespace product_of_odd_primes_mod_32_l369_369169

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def M : ℕ := odd_primes_less_than_32.foldr (· * ·) 1

theorem product_of_odd_primes_mod_32 : M % 32 = 17 :=
by
  -- the proof goes here
  sorry

end product_of_odd_primes_mod_32_l369_369169


namespace recurrence_relation_l369_369914

-- Define the function p_nk and prove the recurrence relation
def p (n k : ℕ) : ℝ := sorry

theorem recurrence_relation (n k : ℕ) (h : k < n) : 
  p n k = p (n-1) k - (1 / 2^k) * p (n-k) k + (1 / 2^k) :=
sorry

end recurrence_relation_l369_369914


namespace typists_initial_group_l369_369099

theorem typists_initial_group
  (T : ℕ) 
  (h1 : 0 < T) 
  (h2 : T * (240 / 40 * 20) = 2400) : T = 10 :=
by
  sorry

end typists_initial_group_l369_369099


namespace find_x_when_y_equals_2_l369_369097

theorem find_x_when_y_equals_2 (x : ℚ) (y : ℚ) : 
  y = (1 / (4 * x + 2)) ∧ y = 2 -> x = -3 / 8 := 
by 
  sorry

end find_x_when_y_equals_2_l369_369097


namespace maximum_n_for_positive_sum_l369_369059

variable {a : ℕ → ℝ} -- a_n denotes the nth term of the arithmetic sequence

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ (n : ℕ), a (n + 1) = a n + d

theorem maximum_n_for_positive_sum (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d →
  (a 1 + a 18) > 0 →
  (a 10) < 0 →
  ∃ (n : ℕ), (∀ m, m > n → (∑ i in Finset.range (m + 1), a i) ≤ 0) ∧ (∑ i in Finset.range (n + 1), a i) > 0 ∧ n = 18 :=
by
  sorry

end maximum_n_for_positive_sum_l369_369059


namespace polynomial_remainder_l369_369033

theorem polynomial_remainder (x : ℝ) : 
  let f := 3 * x^6 - x^5 + 2 * x^3 - 8,
      g := x^2 + 3 * x + 2
  in ∃ q : ℝ → ℝ, ∃ a b : ℝ, f = g * q + a * x + b ∧ a = -206 ∧ b = -212 :=
by 
  let f := 3 * x^6 - x^5 + 2 * x^3 - 8
  let g := x^2 + 3 * x + 2
  use polynomial.eval (λ q, q) -- placeholder for q(x)
  use -206
  use -212
  split
  · sorry
  · exact ⟨rfl, rfl⟩

end polynomial_remainder_l369_369033


namespace remainder_of_M_l369_369264

def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_of_M : M % 32 = 31 := by
  -- Proof goes here
  sorry

end remainder_of_M_l369_369264


namespace value_of_f_neg_5π_over_12_l369_369788

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem value_of_f_neg_5π_over_12 :
  ∀ (ω φ : ℝ), 
    (∀ x y : ℝ, (x < y ∧ x ∈ Ioo (π / 6) (2 * π / 3) ∧ y ∈ Ioo (π / 6) (2 * π / 3)) → f x ω φ < f y ω φ) ∧ 
    f (π / 6) ω φ = f (2 * π / 3) ω φ → 
    f (-5 * π / 12) ω φ = √3 / 2 :=
by
  sorry

end value_of_f_neg_5π_over_12_l369_369788


namespace log3_seq_limit_value_l369_369013

theorem log3_seq_limit_value (x : ℝ) (h : x = Real.log 3 (64 + x)) : x ≈ 3.8 :=
sorry

end log3_seq_limit_value_l369_369013


namespace probability_recurrence_relation_l369_369891

theorem probability_recurrence_relation (n k : ℕ) (h : k < n) :
  ∀ (p : ℕ → ℕ → ℝ), p n k = p (n-1) k - (1 / (2:ℝ)^k) * p (n-k) k + 1 / (2:ℝ)^k := 
sorry

end probability_recurrence_relation_l369_369891


namespace solution_set_l369_369377

theorem solution_set (x : ℝ) :
  (x + 8 < 4 * x - 1 ∧ (1 / 2) * x ≥ 4 - (3 / 2) * x) → x > 3 :=
by
  intros h
  cases h with h1 h2
  sorry

end solution_set_l369_369377


namespace remainder_when_divided_by_32_l369_369164

def product_of_odds : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_divided_by_32 : product_of_odds % 32 = 9 := by
  sorry

end remainder_when_divided_by_32_l369_369164


namespace find_f_of_neg_5_pi_over_12_l369_369731

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - (5 * π / 6))

theorem find_f_of_neg_5_pi_over_12 :
  (∀ x ∈ (set.Ioo (π / 6) (2 * π / 3)), (0 : ℝ) < (f x - f (x - 1))) ∧
  (∀ x, f (π / 6 - x) = f (π / 6 + x)) ∧
  (∀ x, f (2 * π / 3 - x) = f (2 * π / 3 + x)) →
  f (-5 * π / 12) = √3 / 2 :=
by
  sorry

end find_f_of_neg_5_pi_over_12_l369_369731


namespace remainder_when_M_divided_by_32_l369_369142

-- Define M as the product of all odd primes less than 32.
def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_M_divided_by_32 :
    M % 32 = 1 := by
  -- We must prove that M % 32 = 1.
  sorry

end remainder_when_M_divided_by_32_l369_369142


namespace remainder_M_mod_32_l369_369251

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_M_mod_32 : M % 32 = 17 :=
by {
  sorry
}

end remainder_M_mod_32_l369_369251


namespace find_PC_length_l369_369940

def triangle : Type := ℝ × ℝ × ℝ

variables (A B C : triangle) (P : triangle)
variables (PA PB PC : ℝ)

-- Conditions
axiom angle_ABC_right : ∠ B C = 90
axiom PA_length : PA = 12
axiom PB_length : PB = 8
axiom angles_equal: ∠ APB = 120 ∧ ∠ BPC = 120 ∧ ∠ CPA = 120

-- The theorem to prove
theorem find_PC_length : PC = 16 :=
by {
  sorry
}

end find_PC_length_l369_369940


namespace problem_statement_l369_369750

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - 5 * Real.pi / 6)

theorem problem_statement :
  (∀ (x : ℝ), (∀ a b : ℝ, a < b → x ∈ (Set.Ioc (Real.pi / 6) (2 * Real.pi / 3)) → f(x) < f(b)) → x = (Real.pi / 6) ∨ x = (2 * Real.pi / 3)) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by sorry

end problem_statement_l369_369750


namespace find_value_of_f_eq_l369_369762

noncomputable def f (ω ω φ x : ℝ) : ℝ := sin (ω * x + φ)

theorem find_value_of_f_eq :
  ∀ (ω φ : ℝ), 
    (∀ x, x ∈ set.Ioo (π / 6) (2 * π / 3) → (f ω φ x) ≤ (f ω φ (x + 1e-10))) → -- monotonically increasing
    (ω * ∓ (π / 6) + φ) = (φ + ω * (2 * π / 3)) → -- symmetric axes
    f ω φ (-(5 * π / 12)) = sqrt 3 / 2 :=
by
  sorry

end find_value_of_f_eq_l369_369762


namespace gcd_18_30_is_6_l369_369616

def gcd_18_30 : ℕ :=
  gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 :=
by {
  -- The step here will involve using properties of gcd and prime factorization,
  -- but we are given the result directly for the purpose of this task.
  sorry
}

end gcd_18_30_is_6_l369_369616


namespace max_value_f_l369_369824

noncomputable def f (x : ℝ) : ℝ := x - real.sqrt 2 * real.sin x

theorem max_value_f :
  ∃ x ∈ set.Icc 0 real.pi, ∀ y ∈ set.Icc 0 real.pi, f y ≤ f x ∧ f x = real.pi :=
sorry

end max_value_f_l369_369824


namespace two_digit_number_eq_27_l369_369494

theorem two_digit_number_eq_27 (A : ℕ) (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 9) (hy : 0 ≤ y ∧ y ≤ 9)
    (h : A = 10 * x + y) (hcond : A = 3 * (x + y)) : A = 27 :=
by
  sorry

end two_digit_number_eq_27_l369_369494


namespace gcd_of_18_and_30_l369_369558

-- Define the numbers
def a := 18
def b := 30

-- The main theorem statement
theorem gcd_of_18_and_30 : Nat.gcd a b = 6 := by
  sorry

end gcd_of_18_and_30_l369_369558


namespace freddy_travel_time_l369_369018

theorem freddy_travel_time (dist_A_B : ℝ) (time_Eddy : ℝ) (dist_A_C : ℝ) (speed_ratio : ℝ) (travel_time_Freddy : ℝ) :
  dist_A_B = 540 ∧ time_Eddy = 3 ∧ dist_A_C = 300 ∧ speed_ratio = 2.4 →
  travel_time_Freddy = dist_A_C / (dist_A_B / time_Eddy / speed_ratio) :=
  sorry

end freddy_travel_time_l369_369018


namespace find_value_l369_369705

-- Define the function $f(x)$
def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

-- Define the problem statement in Lean 4
theorem find_value :
  (∀ x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), 0 ≤ 2 * x - 5 * Real.pi / 6 ∧ 2 * x - 5 * Real.pi / 6 ≤ Real.pi) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by
  -- A proper rigorous Lean proof should go here
  sorry

end find_value_l369_369705


namespace theta_value_l369_369368

noncomputable def complexSumExpression : ℂ :=
  complex.exp (11 * real.pi * complex.I / 60) +
  complex.exp (21 * real.pi * complex.I / 60) +
  complex.exp (31 * real.pi * complex.I / 60) +
  complex.exp (41 * real.pi * complex.I / 60) +
  complex.exp (51 * real.pi * complex.I / 60) +
  complex.exp (59 * real.pi * complex.I / 60)

theorem theta_value (θ: ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * real.pi) :
  complexSumExpression = complex.abs complexSumExpression * complex.exp (complex.I * θ) →
  θ = 31 * real.pi / 60 :=
sorry

end theta_value_l369_369368


namespace journeymen_percentage_correct_l369_369997

def total_employees : ℕ := 20210
def journeymen_fraction : ℚ := 2 / 7
def laid_off_fraction : ℚ := 1 / 2

def initial_journeymen : ℚ := journeymen_fraction * total_employees
def laid_off_journeymen : ℚ := initial_journeymen * laid_off_fraction
def remaining_journeymen : ℚ := initial_journeymen - laid_off_journeymen
def remaining_employees : ℚ := total_employees - laid_off_journeymen
def percentage_journeymen : ℚ := (remaining_journeymen / remaining_employees) * 100

theorem journeymen_percentage_correct :
  percentage_journeymen ≈ 16.67 := 
sorry

end journeymen_percentage_correct_l369_369997


namespace recurrence_relation_l369_369899

variables {n k : ℕ}

def p : ℕ → ℕ → ℚ := sorry

theorem recurrence_relation (n k : ℕ) (hnk : n ≥ k) :
  p n k = p (n-1) k - (1 / (2^k)) * p (n-k) k + (1 / (2^k)) :=
begin
  sorry
end

end recurrence_relation_l369_369899


namespace remainder_M_mod_32_l369_369250

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_M_mod_32 : M % 32 = 17 :=
by {
  sorry
}

end remainder_M_mod_32_l369_369250


namespace shirts_before_buying_l369_369345

-- Define the conditions
variable (new_shirts : ℕ)
variable (total_shirts : ℕ)

-- Define the statement where we need to prove the number of shirts Sarah had before buying the new ones
theorem shirts_before_buying (h₁ : new_shirts = 8) (h₂ : total_shirts = 17) : total_shirts - new_shirts = 9 :=
by
  -- Proof goes here
  sorry

end shirts_before_buying_l369_369345


namespace remainder_when_M_divided_by_32_l369_369137

-- Define M as the product of all odd primes less than 32.
def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_M_divided_by_32 :
    M % 32 = 1 := by
  -- We must prove that M % 32 = 1.
  sorry

end remainder_when_M_divided_by_32_l369_369137


namespace remainder_of_M_l369_369265

def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_of_M : M % 32 = 31 := by
  -- Proof goes here
  sorry

end remainder_of_M_l369_369265


namespace remainder_when_divided_by_32_l369_369163

def product_of_odds : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_divided_by_32 : product_of_odds % 32 = 9 := by
  sorry

end remainder_when_divided_by_32_l369_369163


namespace prime_product_mod_32_l369_369291

open Nat

theorem prime_product_mod_32 : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := List.foldr (· * ·) 1 primes
  M % 32 = 25 :=
by
  sorry

end prime_product_mod_32_l369_369291


namespace bowling_ball_weight_l369_369355

theorem bowling_ball_weight :
  (∃ (weight_of_bowling_ball : ℝ) (weight_of_kayak : ℝ),
    (∀ k, weight_of_kayak = 35) ∧
    (forall w, 10 * w = 4 * weight_of_kayak) →
    weight_of_bowling_ball = 14) :=
by
  sorry

end bowling_ball_weight_l369_369355


namespace extremum_range_a_l369_369101

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - a * x^2 + x

theorem extremum_range_a :
  (∀ x : ℝ, -1 < x ∧ x < 0 → (f a x = 0 → ∃ x0 : ℝ, f a x0 = 0 ∧ -1 < x0 ∧ x0 < 0)) →
  a < -1/5 ∨ a = -1 :=
sorry

end extremum_range_a_l369_369101


namespace evaluate_product_to_permutation_l369_369045

/- The mathematical problem statement in Lean 4 -/
theorem evaluate_product_to_permutation (n : ℕ) (h1 : 0 < n) (h2 : n < 20) :
  (finset.prod (finset.range 81) (λ k, 20 + k - n)) = nat.desc_factorial (100 - n) 81 :=
sorry

end evaluate_product_to_permutation_l369_369045


namespace min_sum_inequality_l369_369039

noncomputable def min_sum (n : ℕ) (a : ℕ → ℝ) : ℝ :=
a 0 + a 1 + ... + a n

theorem min_sum_inequality (n : ℕ) (a : ℕ → ℝ) (F : ℕ → ℝ) :
  (n ≥ 2) →
  (a 0 = 1) →
  (∀ i, (0 ≤ i ∧ i ≤ n-2) → a i ≤ a (i+1) + a (i+2)) →
  (∀ i, 0 ≤ a i) →
  min_sum n a = (F (n+2) - 1) / F n :=
begin
  sorry
end

end min_sum_inequality_l369_369039


namespace area_of_square_l369_369453

theorem area_of_square (x y : ℝ) :
  2 * x^2 = -2 * y^2 + 8 * x - 12 * y + 40 →
  ∃ (s : ℝ), s = 2 * real.sqrt 33 ∧ s^2 = 132 :=
by
  sorry

end area_of_square_l369_369453


namespace majors_selection_l369_369526

theorem majors_selection (majors : Finset ℕ) (A B : ℕ) (h : A ∈ majors) (h' : B ∈ majors) (h_card : majors.card = 7) :
  let S := majors.erase A;
  let T := majors.erase B;
  (majors.card.choose 3 - (2.choose 2) * (5.choose 1)) * 3.factorial = 180 :=
by
  sorry

end majors_selection_l369_369526


namespace remainder_of_product_of_odd_primes_mod_32_l369_369282

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem remainder_of_product_of_odd_primes_mod_32 :
  (product odd_primes_less_than_32) % 32 = 23 :=
by sorry

end remainder_of_product_of_odd_primes_mod_32_l369_369282


namespace remainder_when_divided_by_32_l369_369158

def product_of_odds : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_divided_by_32 : product_of_odds % 32 = 9 := by
  sorry

end remainder_when_divided_by_32_l369_369158


namespace remainder_of_M_when_divided_by_32_l369_369198

open Nat

def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
def M : ℕ := List.foldl (*) 1 oddPrimes

theorem remainder_of_M_when_divided_by_32 :
  M % 32 = 1 :=
sorry

end remainder_of_M_when_divided_by_32_l369_369198


namespace value_of_f_neg_5π_over_12_l369_369792

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem value_of_f_neg_5π_over_12 :
  ∀ (ω φ : ℝ), 
    (∀ x y : ℝ, (x < y ∧ x ∈ Ioo (π / 6) (2 * π / 3) ∧ y ∈ Ioo (π / 6) (2 * π / 3)) → f x ω φ < f y ω φ) ∧ 
    f (π / 6) ω φ = f (2 * π / 3) ω φ → 
    f (-5 * π / 12) ω φ = √3 / 2 :=
by
  sorry

end value_of_f_neg_5π_over_12_l369_369792


namespace lucy_run_base10_eq_1878_l369_369401

-- Define a function to convert a base-8 numeral to base-10.
def base8_to_base10 (n: Nat) : Nat :=
  (3 * 8^3) + (5 * 8^2) + (2 * 8^1) + (6 * 8^0)

-- Define the base-8 number.
def lucy_run (n : Nat) : Nat := n

-- Prove that the base-10 equivalent of the base-8 number 3526 is 1878.
theorem lucy_run_base10_eq_1878 : base8_to_base10 (lucy_run 3526) = 1878 :=
by
  -- This is a placeholder for the actual proof.
  sorry

end lucy_run_base10_eq_1878_l369_369401


namespace problem1_problem2_problem3_l369_369642

section

variable (k : ℝ) (A : Set ℝ)

/- Problem 1 -/
theorem problem1 (h₁ : A = {x | k * x^2 - 2 * x + 6 * k < 0} ∧ k ≠ 0) (h₂ : A ⊆ set.Ioo 2 3) : 
  k ≥ (2 / 5) := 
sorry

/- Problem 2 -/
theorem problem2 (h₁ : A = {x | k * x^2 - 2 * x + 6 * k < 0} ∧ k ≠ 0) (h₂ : set.Ioo 2 3 ⊆ A) : 
  k ≤ (2 / 5) := 
sorry

/- Problem 3 -/
theorem problem3 (h₁ : A = {x | k * x^2 - 2 * x + 6 * k < 0} ∧ k ≠ 0) (h₂ : (A ∩ set.Ioo 2 3).Nonempty) : 
  k < (Real.sqrt 6 / 6) := 
sorry

end

end problem1_problem2_problem3_l369_369642


namespace product_mod_32_l369_369211

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  M % 32 = 17 := by
  sorry

end product_mod_32_l369_369211


namespace product_mod_32_is_15_l369_369248

-- Define the odd primes less than 32
def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the product of odd primes
def M : ℕ := List.foldl (· * ·) 1 oddPrimes

-- The theorem to prove the remainder when M is divided by 32 is 15
theorem product_mod_32_is_15 : M % 32 = 15 := by
  sorry

end product_mod_32_is_15_l369_369248


namespace find_value_l369_369710

-- Define the function $f(x)$
def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

-- Define the problem statement in Lean 4
theorem find_value :
  (∀ x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), 0 ≤ 2 * x - 5 * Real.pi / 6 ∧ 2 * x - 5 * Real.pi / 6 ≤ Real.pi) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by
  -- A proper rigorous Lean proof should go here
  sorry

end find_value_l369_369710


namespace probability_recurrence_relation_l369_369893

theorem probability_recurrence_relation (n k : ℕ) (h : k < n) :
  ∀ (p : ℕ → ℕ → ℝ), p n k = p (n-1) k - (1 / (2:ℝ)^k) * p (n-k) k + 1 / (2:ℝ)^k := 
sorry

end probability_recurrence_relation_l369_369893


namespace factorize_expression_l369_369023

theorem factorize_expression (a : ℝ) : a^2 + 2*a + 1 = (a + 1)^2 :=
by
  sorry

end factorize_expression_l369_369023


namespace find_A_l369_369519

def diamond (A B : ℝ) : ℝ := 5 * A + 3 * B + 7

theorem find_A (A : ℝ) (h : diamond A 5 = 82) : A = 12 :=
by
  unfold diamond at h
  sorry

end find_A_l369_369519


namespace gcd_18_30_l369_369574

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := 
by
  sorry

end gcd_18_30_l369_369574


namespace perfect_square_l369_369421

theorem perfect_square (a b : ℝ) : a^2 + 2 * a * b + b^2 = (a + b)^2 := by
  sorry

end perfect_square_l369_369421


namespace find_a_n_expression_l369_369647

variable {a_n : ℕ → ℕ}
variable (S : ℕ → ℕ)

def S_n_property (n : ℕ) : Prop := S n = 2 * a_n n - 4

theorem find_a_n_expression (n : ℕ) (h : ∀ n, S_n_property S a_n n) : a_n n = 2^(n+1) := by
  sorry

end find_a_n_expression_l369_369647


namespace second_container_clay_l369_369456

theorem second_container_clay :
  let h1 := 3
  let w1 := 5
  let l1 := 7
  let clay1 := 105
  let h2 := 3 * h1
  let w2 := 2 * w1
  let l2 := l1
  let V1 := h1 * w1 * l1
  let V2 := h2 * w2 * l2
  V1 = clay1 →
  V2 = 6 * V1 →
  V2 = 630 :=
by
  intros
  sorry

end second_container_clay_l369_369456


namespace product_of_odd_primes_mod_32_l369_369170

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def M : ℕ := odd_primes_less_than_32.foldr (· * ·) 1

theorem product_of_odd_primes_mod_32 : M % 32 = 17 :=
by
  -- the proof goes here
  sorry

end product_of_odd_primes_mod_32_l369_369170


namespace polygon_sides_l369_369384

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 720) : n = 6 :=
by sorry

end polygon_sides_l369_369384


namespace find_possible_numbers_l369_369438

-- Definitions derived from the conditions
def is_valid_column (top mid bot : ℕ) : Prop :=
  top + mid = 2 * bot

def valid_sum (numbers : list ℕ) : Prop :=
  numbers.sum = 78

def filled_table_with_condition (table : list (list (option ℕ))) : Prop :=
  let column1 := (table[0][0], table[1][0], table[2][0])
  let column2 := (table[0][1], table[1][1], table[2][1])
  let column3 := (table[0][2], table[1][2], table[2][2])
  let column4 := (table[0][3], table[1][3], table[2][3])
  column1.1.is_some ∧ column1.2.is_some ∧ column1.3.is_some ∧
  column2.1.is_some ∧ column2.2.is_some ∧ column2.3.is_some ∧
  column3.1.is_some ∧ column3.2.is_some ∧ column3.3.is_some ∧
  column4.1.is_some ∧ column4.2.is_some ∧ column4.3.is_some ∧
  is_valid_column column1.1.get column1.2.get column1.3.get ∧
  is_valid_column column2.1.get column2.2.get column2.3.get ∧
  is_valid_column column3.1.get column3.2.get column3.3.get ∧
  is_valid_column column4.1.get column4.2.get column4.3.get

theorem find_possible_numbers (table : list (list (option ℕ))) :
  (∃ (s : ℕ), (s = 2 ∨ s = 11) ∧ 
              filled_table_with_condition table ∧
              valid_sum (table.bind id).map (option.get_or_else 0)) :=
sorry

end find_possible_numbers_l369_369438


namespace henry_distance_from_starting_point_l369_369839

theorem henry_distance_from_starting_point :
  let north_distance := 12
  let east_distance := 30
  let south_distance := 18
  let net_south := south_distance - north_distance
  let distance := Real.sqrt (net_south^2 + east_distance^2)
  distance = 2 * Real.sqrt 234 :=
by
  let north_distance := 12
  let east_distance := 30
  let south_distance := 18
  let net_south := south_distance - north_distance
  let distance := Real.sqrt (net_south^2 + east_distance^2)
  show distance = 2 * Real.sqrt 234
  sorry

end henry_distance_from_starting_point_l369_369839


namespace product_of_odd_primes_mod_32_l369_369184

theorem product_of_odd_primes_mod_32 :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  M % 32 = 17 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  exact sorry

end product_of_odd_primes_mod_32_l369_369184


namespace vulgar_fraction_of_decimal_l369_369430

theorem vulgar_fraction_of_decimal :
  (0.34 : ℚ) = 17 / 50 :=
by 
  norm_num

end vulgar_fraction_of_decimal_l369_369430


namespace hexagon_diagonals_count_l369_369090

theorem hexagon_diagonals_count : 6.choose 2 - 6 = 9 := by
  -- there are 6 vertices in a hexagon
  -- diagonals are lines connecting non-adjacent vertices
  -- the formula for the number of diagonals in a polygon of n sides is n choose 2 - n
  sorry

end hexagon_diagonals_count_l369_369090


namespace range_of_f2_l369_369072

-- Define the function
def f (x : ℝ) (m : ℝ) : ℝ := 3 * x^2 + m * x + 2

-- Define the condition that f is increasing on the interval [1, +∞)
def increasing_on_interval (m : ℝ) : Prop := ∀ x y : ℝ, 1 ≤ x → x ≤ y → f(x, m) ≤ f(y, m)

-- Prove that given the function f is increasing on [1, +∞), the range of f(2) is [2, +∞)
theorem range_of_f2 (m : ℝ) (h : increasing_on_interval m) : 2 ≤ f(2, m) :=
by
  -- The proof is omitted
  sorry

end range_of_f2_l369_369072


namespace total_number_of_coins_is_324_l369_369397

noncomputable def total_coins (total_sum : ℕ) (coins_20p : ℕ) (coins_25p_value : ℕ) : ℕ :=
    coins_20p + (coins_25p_value / 25)

theorem total_number_of_coins_is_324 (h_sum: 7100 = 71 * 100) (h_coins_20p: 200 * 20 = 4000) :
  total_coins 7100 200 3100 = 324 := by
  sorry

end total_number_of_coins_is_324_l369_369397


namespace product_mod_32_l369_369207

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  M % 32 = 17 := by
  sorry

end product_mod_32_l369_369207


namespace remainder_of_M_l369_369271

def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_of_M : M % 32 = 31 := by
  -- Proof goes here
  sorry

end remainder_of_M_l369_369271


namespace sin_monotonic_increasing_f_symmetric_axes_find_value_l369_369697

def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi * 5 / 6)

theorem sin_monotonic_increasing (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) :
  Monotone (λ x, f x) (Ioo a b) := 
sorry

theorem f_symmetric_axes (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) 
  (x y : ℝ) (hx : x = a + (b - a) * k) (hy : y = a + (b - a) * (-k)) : 
  f x = f y :=
sorry

theorem find_value :
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
sorry

end sin_monotonic_increasing_f_symmetric_axes_find_value_l369_369697


namespace find_fx_value_l369_369729

noncomputable def f (x : Real) : Real :=
  sin (2 * x - 5 * Real.pi / 6)

theorem find_fx_value :
  (∀ x, f x = sin (2 * x - 5 * Real.pi / 6)) ∧ 
  (f (Real.pi / 6) = f (2 * Real.pi / 3)) ∧ 
  (∀ x1 x2, (Real.pi / 6 < x1 ∧ x1 < 2 * Real.pi / 3 ∧ x1 < x2 ∧ x2 < 2 * Real.pi / 3) -> f x1 < f x2) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 := by
  sorry

end find_fx_value_l369_369729


namespace remainder_of_M_when_divided_by_32_l369_369196

open Nat

def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
def M : ℕ := List.foldl (*) 1 oddPrimes

theorem remainder_of_M_when_divided_by_32 :
  M % 32 = 1 :=
sorry

end remainder_of_M_when_divided_by_32_l369_369196


namespace fraction_of_students_playing_woodwind_or_brass_instruments_l369_369529

theorem fraction_of_students_playing_woodwind_or_brass_instruments (x : ℕ) :
  let woodwind_last_year := (1 / 2 : ℚ) * x,
      brass_last_year := (2 / 5 : ℚ) * x,
      percussion_last_year := x - woodwind_last_year - brass_last_year,
      woodwind_this_year := (1 / 2 : ℚ) * woodwind_last_year,
      brass_this_year := (3 / 4 : ℚ) * brass_last_year,
      fraction_woodwind_and_brass_this_year := (woodwind_this_year + brass_this_year) / x
  in fraction_woodwind_and_brass_this_year = 11 / 20 :=
by
  sorry

end fraction_of_students_playing_woodwind_or_brass_instruments_l369_369529


namespace isosceles_trapezoid_diagonal_l369_369933

noncomputable def isosceles_trapezoid_diagonal_length : ℕ :=
  let h := Real.sqrt ((12 : ℝ) ^ 2 - (6 : ℝ) ^ 2)
  let d := Real.sqrt ((6 : ℝ) ^ 2 + h ^ 2)
  d.toNat

theorem isosceles_trapezoid_diagonal (AB CD AD : ℕ) (h : AB = 25 ∧ CD = 13 ∧ AD = 12) :
  isosceles_trapezoid_diagonal_length = 12 := by
  sorry

end isosceles_trapezoid_diagonal_l369_369933


namespace first_digit_base8_rep_395_eq_6_l369_369413

-- Define the base 10 number
def base10_num : ℕ := 395

-- Define the target base
def base : ℕ := 8

-- Define the first digit in base 8 representation for 395 == 6
theorem first_digit_base8_rep_395_eq_6 (base10_num : ℕ) (base : ℕ) : nat.digits base base10_num = (6 :: _) :=
by
  sorry

end first_digit_base8_rep_395_eq_6_l369_369413


namespace problem_statement_l369_369752

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - 5 * Real.pi / 6)

theorem problem_statement :
  (∀ (x : ℝ), (∀ a b : ℝ, a < b → x ∈ (Set.Ioc (Real.pi / 6) (2 * Real.pi / 3)) → f(x) < f(b)) → x = (Real.pi / 6) ∨ x = (2 * Real.pi / 3)) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by sorry

end problem_statement_l369_369752


namespace sqrt_fraction_eq_l369_369515

theorem sqrt_fraction_eq : 
  sqrt (1 / 4 + 1 / 9) = (sqrt 13) / 6 := 
by
  sorry

end sqrt_fraction_eq_l369_369515


namespace min_average_value_l369_369503

noncomputable def is_valid_grid (grid : ℕ × ℕ → ℕ) : Prop :=
  ∀ i j, i ≤ 3 ∧ j ≤ 3 → 
    (grid (i, j) + grid (i + 1, j) + grid (i, j + 1) + grid (i + 1, j + 1)) = 400

theorem min_average_value (grid : ℕ × ℕ → ℕ) (h : is_valid_grid grid) :
  (∑ i in finRange 5, ∑ j in finRange 5, grid (i, j)) / 25 = 64 := 
sorry

end min_average_value_l369_369503


namespace poly_roots_l369_369623

-- Define the polynomial
def poly: Polynomial ℝ := 8 * X^5 + 26 * X^4 - 74 * X^3 + 40 * X^2

-- Noncomputable declaration needed for roots involving square roots
noncomputable def roots : List ℝ :=
  [0, 0, 1, (-34 + 2 * Real.sqrt 609) / 16, (-34 - 2 * Real.sqrt 609) / 16]

-- The theorem stating that these are indeed the roots of the given polynomial
theorem poly_roots : (Polynomial.eval₂ Polynomial.C id poly) roots = 0 :=
by
  -- we would provide a proof here
  sorry

end poly_roots_l369_369623


namespace remainder_when_divided_by_32_l369_369156

def product_of_odds : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_divided_by_32 : product_of_odds % 32 = 9 := by
  sorry

end remainder_when_divided_by_32_l369_369156


namespace gcd_18_30_is_6_l369_369614

def gcd_18_30 : ℕ :=
  gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 :=
by {
  -- The step here will involve using properties of gcd and prime factorization,
  -- but we are given the result directly for the purpose of this task.
  sorry
}

end gcd_18_30_is_6_l369_369614


namespace polygon_sides_l369_369382

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 720) : n = 6 :=
by sorry

end polygon_sides_l369_369382


namespace find_f_of_neg_5_pi_over_12_l369_369739

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - (5 * π / 6))

theorem find_f_of_neg_5_pi_over_12 :
  (∀ x ∈ (set.Ioo (π / 6) (2 * π / 3)), (0 : ℝ) < (f x - f (x - 1))) ∧
  (∀ x, f (π / 6 - x) = f (π / 6 + x)) ∧
  (∀ x, f (2 * π / 3 - x) = f (2 * π / 3 + x)) →
  f (-5 * π / 12) = √3 / 2 :=
by
  sorry

end find_f_of_neg_5_pi_over_12_l369_369739


namespace parabola_problem_l369_369645

theorem parabola_problem
  (M : ℝ × ℝ)
  (C : ℝ → ℝ → Prop)
  (directrix : ℝ → Prop)
  (N : ℝ × ℝ)
  (A : ℝ × ℝ)
  (p : ℝ)
  (t : ℝ) :
  M = (1, 1/2) →
  C y x ↔ y^2 = 2 * p * x →
  directrix x ↔ x = -p/2 →
  dist M (1, -p/2) = 5/4 →
  N = (t, 2) →
  C 2 t →
  (p = 1/2 ∧ t = 4) ∧
  ∃ P Q : ℝ × ℝ, 
    (C P.snd P.fst ∧ C Q.snd Q.fst) ∧
    ((P = (x1, -2 * x1 + 1)) ∨ (Q = (x2, -2 * x2 + 1))) ∧ 
    dist P Q = (3/4) * real.sqrt(5) :=
by
  sorry

end parabola_problem_l369_369645


namespace recurrence_relation_p_series_l369_369908

noncomputable def p_series (n k : ℕ) : ℝ :=
if k < n then (p_series (n - 1) k - (1 / (2 : ℝ)^k) * p_series (n - k) k + (1 / (2 : ℝ)^k))
else 0

-- Statement of the theorem
theorem recurrence_relation_p_series (n k : ℕ) (h : k < n) :
  p_series n k = p_series (n - 1) k - (1 / (2 : ℝ)^k) * p_series (n - k) k + (1 / (2 : ℝ)^k) :=
sorry

end recurrence_relation_p_series_l369_369908


namespace library_books_new_releases_l369_369104

theorem library_books_new_releases (P Q R S : Prop) 
  (h : ¬P) 
  (P_iff_Q : P ↔ Q)
  (Q_implies_R : Q → R)
  (S_iff_notP : S ↔ ¬P) : 
  Q ∧ S := by 
  sorry

end library_books_new_releases_l369_369104


namespace product_mod_32_is_15_l369_369241

-- Define the odd primes less than 32
def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the product of odd primes
def M : ℕ := List.foldl (· * ·) 1 oddPrimes

-- The theorem to prove the remainder when M is divided by 32 is 15
theorem product_mod_32_is_15 : M % 32 = 15 := by
  sorry

end product_mod_32_is_15_l369_369241


namespace false_proposition_D_l369_369504

noncomputable def l : Type := sorry

noncomputable def α : Type := sorry

noncomputable def β : Type := sorry

noncomputable def γ : Type := sorry

axiom alpha_perpendicular_beta : α ⊥ β

axiom alpha_not_perpendicular_beta (A_internal α β : Type) : ¬(α ⊥ β) → ¬(exists l, l ∈ α ∧ l ⊥ β)

axiom alpha_perpendicular_gamma_and_beta_perpendicular_gamma : α ⊥ γ ∧ β ⊥ γ

axiom alpha_intersect_beta_is_l (A_internal α β : Type) : α ∩ β = l

axiom perpendicularity (α β γ : Type) (l : α) (h1: α ⊥ β) (h2: α ⊥ γ) : l ⊥ γ

axiom angle_condition (α β : Type) (l : α) : ¬(α_perpendicular_beta α β) → ¬(angle_between l α ∧ angle_between l β are_complementary)

theorem false_proposition_D : ∃ (D : Type), 
  (if α ⊥ β ∧ l ∈ α ∧ l ∈ β then ¬ (angle_between l α ∧ angle_between l β are_complementary) else true) :=
by
  sorry

end false_proposition_D_l369_369504


namespace gcd_18_30_l369_369579

theorem gcd_18_30: Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l369_369579


namespace isosceles_triangle_dot_product_range_l369_369055

theorem isosceles_triangle_dot_product_range
  {O A B : Type}
  [Nonempty O]
  [Nonempty A]
  [Nonempty B]
  (OA OB : O → ℝ^3)
  (length_OA : ∥OA∥ = 2)
  (length_OB : ∥OB∥ = 2)
  (triangle_isosceles : ∃ O A B : ℝ^3, ∥OA∥ = ∥OB∥ ∧ OA + OB ≥ (sqrt 3 / 3 : ℝ) * (∥OA - OB∥)) :
  ∀ OA OB : ℝ^3,
  -2 ≤ OA ⋅ OB ∧ OA ⋅ OB < 4 :=
by
  sorry

end isosceles_triangle_dot_product_range_l369_369055


namespace product_of_odd_primes_mod_32_l369_369180

theorem product_of_odd_primes_mod_32 :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  M % 32 = 17 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  exact sorry

end product_of_odd_primes_mod_32_l369_369180


namespace functional_eq_solution_l369_369026

theorem functional_eq_solution (f : ℕ → ℕ) :
  (∀ m n, f (m * n) + f (m + n) = f m * f n + 1) →
  (f = (λ n, 1) ∨ f = (λ n, n + 1)) :=
begin
  intro h,
  sorry,
end

end functional_eq_solution_l369_369026


namespace product_of_odd_primes_mod_32_l369_369223

theorem product_of_odd_primes_mod_32 :
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31) in
  M % 32 = 25 :=
by
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31)
  sorry

end product_of_odd_primes_mod_32_l369_369223


namespace min_prime_factor_sum_l369_369305

theorem min_prime_factor_sum (x y a b c d : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : 5 * x^7 = 13 * y^11)
  (h4 : x = 13^6 * 5^7) (h5 : a = 13) (h6 : b = 5) (h7 : c = 6) (h8 : d = 7) : 
  a + b + c + d = 31 :=
by
  sorry

end min_prime_factor_sum_l369_369305


namespace product_mod_32_l369_369220

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  M % 32 = 17 := by
  sorry

end product_mod_32_l369_369220


namespace zeros_indeterminate_in_interval_l369_369040

noncomputable def f : ℝ → ℝ := sorry

variables (a b : ℝ) (ha : a < b) (hf : f a * f b < 0)

-- The theorem statement
theorem zeros_indeterminate_in_interval :
  (∀ (f : ℝ → ℝ), f a * f b < 0 → (∃ (x : ℝ), a < x ∧ x < b ∧ f x = 0) ∨ (∀ (x : ℝ), a < x ∧ x < b → f x ≠ 0) ∨ (∃ (x1 x2 : ℝ), a < x1 ∧ x1 < x2 ∧ x2 < b ∧ f x1 = 0 ∧ f x2 = 0)) :=
by sorry

end zeros_indeterminate_in_interval_l369_369040


namespace bases_linear_combination_unique_l369_369095

variables {R : Type*} [Field R]
variables (e1 e2 : R → R) (m n : R)

-- Given conditions
def bases_in_plane (e1 e2 : R → R)  := 
  ∀ v : R → R, ∃ a b : R, v = a • e1 + b • e2

def not_collinear (e1 e2 : R → R) := 
  ¬ ∃ (λ : R), e1 = λ • e2

theorem bases_linear_combination_unique
  (h1 : bases_in_plane e1 e2)
  (h2 : not_collinear e1 e2):
  m • e1 + n • e2 = 0 -> m = 0 ∧ n = 0 :=
sorry

end bases_linear_combination_unique_l369_369095


namespace no_such_function_exists_l369_369524

theorem no_such_function_exists : ¬ ∃ f : ℕ → ℕ, ∀ n > 2, f (f (n - 1)) = f (n + 1) - f n :=
by {
  sorry
}

end no_such_function_exists_l369_369524


namespace problem_statement_l369_369049

noncomputable def sequence_a (n : ℕ) : ℕ := 2*n

noncomputable def sum_first_n_terms (n : ℕ) : ℕ :=
  finset.sum (finset.range n) (λ i, sequence_a (i+1))

noncomputable def graph_condition (n : ℕ) : Prop :=
  let Sn := sum_first_n_terms n in
  (n : ℝ) + (sequence_a n / (2 * n)).toReal = (Sn / n).toReal

noncomputable def group_sum (start len : ℕ) : ℕ :=
  finset.sum (finset.range len) (λ i, sequence_a (start + i))

noncomputable def sequence_b (n : ℕ) : ℕ :=
  let cycle := (n-1) / 4 + 1 in
  let group := (n-1) % 4 + 1 in
  match group with
  | 1 => group_sum (2 * (cycle - 1) * 4 + 1) 1
  | 2 => group_sum (2 * (cycle - 1) * 4 + 2) 2
  | 3 => group_sum (2 * (cycle - 1) * 4 + 4) 3
  | 4 => group_sum (2 * (cycle - 1) * 4 + 7) 4
  | _ => 0
  end

theorem problem_statement :
  (graph_condition 1) ∧ (graph_condition 2) ∧ (graph_condition 3) ∧
  (∀ n : ℕ, n >= 1 → (graph_condition n) → (graph_condition (n + 1))) ∧ 
  sequence_b 2018 - sequence_b 1314 = 7040 :=
by sorry

end problem_statement_l369_369049


namespace recurrence_relation_l369_369904

noncomputable def p (n k : ℕ) : ℚ := sorry

theorem recurrence_relation (n k : ℕ) (h : k < n) :
  p n k = p (n - 1) k - (1 / 2^k) * p (n - k) k + (1 / 2^k) :=
by sorry

end recurrence_relation_l369_369904


namespace range_of_a_l369_369078

theorem range_of_a (a : ℝ) (p : ∀ x ∈ Icc (0 : ℝ) 1, a ≥ Real.exp x) 
  (q : ∃ x₀ : ℝ, x₀^2 + 4 * x₀ + a = 0) : e ≤ a ∧ a ≤ 4 := by
  sorry

end range_of_a_l369_369078


namespace find_number_l369_369031

-- Define the number x that satisfies the given condition
theorem find_number (x : ℤ) (h : x + 12 - 27 = 24) : x = 39 :=
by {
  -- This is where the proof steps will go, but we'll use sorry to indicate it's incomplete
  sorry
}

end find_number_l369_369031


namespace calculation_result_poly_step1_correct_simplification_l369_369442

-- Part 1: Calculation
theorem calculation_result :
  -(-1)^4 + sqrt 12 * (sqrt 3) / 3 - (-1 / 2)^(-3) = 9 := sorry

-- Part 2: Polynomial Operations

variables (x y : ℝ)

-- Step 1: Use of difference of squares formula and perfect square formula
theorem poly_step1 :
  ((2*x + y)*(2*x - y) - (2*x - 3*y)^2) = 12*x*y - 10*y^2 := sorry

-- Step 2: Correct Simplification Result
theorem correct_simplification :
  (12*x*y - 10*y^2) * (-2*y) = -24*x*y^2 + 20*y^3 := sorry

end calculation_result_poly_step1_correct_simplification_l369_369442


namespace find_d_l369_369374

theorem find_d (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + 1 = d + real.sqrt (a + b + c - d)) :
  d = 5 / 4 := 
sorry

end find_d_l369_369374


namespace equation_of_ellipse_equation_of_line_AB_l369_369653

-- Step 1: Given conditions for the ellipse and related hyperbola.
def condition_eccentricity (a b c : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ c / a = Real.sqrt 2 / 2

def condition_distance_focus_asymptote (c : ℝ) : Prop :=
  abs c / Real.sqrt (1 + 2) = Real.sqrt 3 / 3

-- Step 2: Given conditions for the line AB.
def condition_line_A_B (k m : ℝ) : Prop :=
  k < 0 ∧ m^2 = 4 / 5 * (1 + k^2) ∧
  ∃ (x1 x2 y1 y2 : ℝ), 
  (1 + 2 * k^2) * x1^2 + 4 * k * m * x1 + 2 * m^2 - 2 = 0 ∧ 
  (1 + 2 * k^2) * x2^2 + 4 * k * m * x2 + 2 * m^2 - 2 = 0 ∧
  x1 + x2 = -4 * k * m / (1 + 2*k^2) ∧ 
  x1 * x2 = (2 * m^2 - 2) / (1 + 2*k^2)

def condition_circle_passes_F2 (x1 x2 k m : ℝ) : Prop :=
  (1 + k^2) * x1 * x2 + (k * m - 1) * (x1 + x2) + m^2 + 1 = 0

noncomputable def problem_data : Prop :=
  ∃ (a b c k m x1 x2 : ℝ),
    condition_eccentricity a b c ∧
    condition_distance_focus_asymptote c ∧
    condition_line_A_B k m ∧
    condition_circle_passes_F2 x1 x2 k m

-- Step 3: Statements to be proven.
theorem equation_of_ellipse : problem_data → 
  ∃ (a b : ℝ), a = Real.sqrt 2 ∧ b = 1 ∧ ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 2 + y^2 = 1) :=
by sorry

theorem equation_of_line_AB : problem_data → 
  ∃ (k m : ℝ), m = 1 ∧ k = -1/2 ∧ ∀ x y : ℝ, (y = k * x + m) ↔ (y = -0.5 * x + 1) :=
by sorry

end equation_of_ellipse_equation_of_line_AB_l369_369653


namespace variance_of_data_set_l369_369051

theorem variance_of_data_set :
  ∃ (x : ℝ), (x = 32) ∧
    let data := [23, 28, 30, x, 34, 39] in
    -- data is arranged in ascending order by definition of list
    -- median is 31
    (∀ (s : list ℝ), s = data →
      (s.sorted.nth (s.length / 2 - 1) + s.sorted.nth (s.length / 2)) / 2 = 31) →
    -- variance calculation
    let mean := (23 + 28 + 30 + x + 34 + 39) / 6 in
    (variance data mean = 74 / 3) :=
by
  sorry

end variance_of_data_set_l369_369051


namespace remainder_of_M_when_divided_by_32_l369_369203

open Nat

def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
def M : ℕ := List.foldl (*) 1 oddPrimes

theorem remainder_of_M_when_divided_by_32 :
  M % 32 = 1 :=
sorry

end remainder_of_M_when_divided_by_32_l369_369203


namespace find_value_l369_369714

-- Define the function $f(x)$
def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

-- Define the problem statement in Lean 4
theorem find_value :
  (∀ x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), 0 ≤ 2 * x - 5 * Real.pi / 6 ∧ 2 * x - 5 * Real.pi / 6 ≤ Real.pi) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by
  -- A proper rigorous Lean proof should go here
  sorry

end find_value_l369_369714


namespace find_f_of_neg_5_pi_over_12_l369_369743

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - (5 * π / 6))

theorem find_f_of_neg_5_pi_over_12 :
  (∀ x ∈ (set.Ioo (π / 6) (2 * π / 3)), (0 : ℝ) < (f x - f (x - 1))) ∧
  (∀ x, f (π / 6 - x) = f (π / 6 + x)) ∧
  (∀ x, f (2 * π / 3 - x) = f (2 * π / 3 + x)) →
  f (-5 * π / 12) = √3 / 2 :=
by
  sorry

end find_f_of_neg_5_pi_over_12_l369_369743


namespace magical_stack_number_of_cards_l369_369360

theorem magical_stack_number_of_cards (n : ℕ) (A B : list ℕ) (m : ℕ) (h: A.length = n ∧ 
                                    B.length = n ∧ 
                                    A = list.range' 1 n ∧ 
                                    B = list.range' (n + 1) (2 * n)) : 
                                    (exists! x, x ∈ A ∧ odd x ∧ 
                                    x = A.index_of 197 ∨ exists! y, y ∈ B ∧ even y ∧ 
                                    y = B.index_of (n + 197)) → 2 * n = 394 :=
by 
  sorry

end magical_stack_number_of_cards_l369_369360


namespace geometric_sequence_term_formula_l369_369670

theorem geometric_sequence_term_formula (a n : ℕ) (a_seq : ℕ → ℕ)
  (h1 : a_seq 0 = a - 1) (h2 : a_seq 1 = a + 1) (h3 : a_seq 2 = a + 4)
  (geometric_seq : ∀ n, a_seq (n + 1) = a_seq n * ((a_seq 1) / (a_seq 0))) :
  a = 5 ∧ a_seq n = 4 * (3 / 2) ^ (n - 1) :=
by
  sorry

end geometric_sequence_term_formula_l369_369670


namespace product_of_odd_primes_mod_32_l369_369190

theorem product_of_odd_primes_mod_32 :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  M % 32 = 17 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  exact sorry

end product_of_odd_primes_mod_32_l369_369190


namespace gcd_18_30_l369_369570

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := 
by
  sorry

end gcd_18_30_l369_369570


namespace gcd_of_18_and_30_l369_369596

-- Define the numbers
def num1 := 18
def num2 := 30

-- State the GCD property
theorem gcd_of_18_and_30 : Nat.gcd num1 num2 = 6 :=
by
  sorry

end gcd_of_18_and_30_l369_369596


namespace exists_distinct_permutations_divisible_l369_369311

open List

theorem exists_distinct_permutations_divisible (n : ℕ) (hn : n > 1 ∧ Odd n) 
  (k : Fin n → ℤ) :
  ∃ (b c : Perm (Fin n)), b ≠ c ∧ (∑ i : Fin n, k i * b i - ∑ i : Fin n, k i * c i) % Nat.factorial n = 0 :=
by
  sorry

end exists_distinct_permutations_divisible_l369_369311


namespace problem_statement_l369_369686

-- Definition of our function f
def f (x : ℝ) : ℝ := Real.sin (2 * x - (5 * Real.pi / 6))

-- Theorem to prove that f(-5π/12) = √3/2
theorem problem_statement : 
  f(-5 * Real.pi / 12) = sqrt 3 / 2 := 
sorry

end problem_statement_l369_369686


namespace gcd_of_18_and_30_l369_369560

-- Define the numbers
def a := 18
def b := 30

-- The main theorem statement
theorem gcd_of_18_and_30 : Nat.gcd a b = 6 := by
  sorry

end gcd_of_18_and_30_l369_369560


namespace problem_statement_l369_369751

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - 5 * Real.pi / 6)

theorem problem_statement :
  (∀ (x : ℝ), (∀ a b : ℝ, a < b → x ∈ (Set.Ioc (Real.pi / 6) (2 * Real.pi / 3)) → f(x) < f(b)) → x = (Real.pi / 6) ∨ x = (2 * Real.pi / 3)) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by sorry

end problem_statement_l369_369751


namespace quadratic_inequality_solution_l369_369351

theorem quadratic_inequality_solution (x : ℝ) : (-x^2 + 5 * x - 4 < 0) ↔ (1 < x ∧ x < 4) :=
sorry

end quadratic_inequality_solution_l369_369351


namespace remainder_of_M_when_divided_by_32_l369_369199

open Nat

def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
def M : ℕ := List.foldl (*) 1 oddPrimes

theorem remainder_of_M_when_divided_by_32 :
  M % 32 = 1 :=
sorry

end remainder_of_M_when_divided_by_32_l369_369199


namespace remainder_of_M_l369_369275

def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_of_M : M % 32 = 31 := by
  -- Proof goes here
  sorry

end remainder_of_M_l369_369275


namespace paper_folding_holes_l369_369473

def folded_paper_holes (folds: Nat) (holes: Nat) : Nat :=
  match folds with
  | 0 => holes
  | n+1 => 2 * folded_paper_holes n holes

theorem paper_folding_holes : folded_paper_holes 3 1 = 8 :=
by
  -- sorry to skip the proof
  sorry

end paper_folding_holes_l369_369473


namespace problem_statement_l369_369684

-- Definition of our function f
def f (x : ℝ) : ℝ := Real.sin (2 * x - (5 * Real.pi / 6))

-- Theorem to prove that f(-5π/12) = √3/2
theorem problem_statement : 
  f(-5 * Real.pi / 12) = sqrt 3 / 2 := 
sorry

end problem_statement_l369_369684


namespace Triangle_l369_369318

-- Define the conditions and properties given in the problem
structure Triangle :=
  (A B C : Point)
  (perimeter : Real)

def triangle_ABC : Triangle :=
  { A := Point.mk 0 0,
    B := Point.mk 1 0,
    C := Point.mk 0.5 (sqrt 3 / 2 ),
    perimeter := 3 + 2 * sqrt 3 }

-- The theorem to prove the Triangle is equilateral under given conditions
theorem Triangle.is_equilateral (T : Triangle)
  (hP : T.perimeter = 3 + 2 * sqrt 3)
  (hLattice : ∀ P : Point, P ∈ T → is_lattice_point P) :
  is_equilateral T :=
sorry

end Triangle_l369_369318


namespace find_P_Q_l369_369025

theorem find_P_Q : 
    ∃ P Q : ℚ, (∀ x : ℚ, x ≠ 10 → x ≠ -4 → (3 * x + 4) / (x^2 - 6 * x - 40) = P / (x - 10) + Q / (x + 4)) ∧ 
                P = 17 / 7 ∧ Q = 4 / 7 :=
by 
  use 17 / 7, 4 / 7
  split 
  . sorry 
  . split
    . rfl 
    . rfl

end find_P_Q_l369_369025


namespace problem_statement_l369_369755

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - 5 * Real.pi / 6)

theorem problem_statement :
  (∀ (x : ℝ), (∀ a b : ℝ, a < b → x ∈ (Set.Ioc (Real.pi / 6) (2 * Real.pi / 3)) → f(x) < f(b)) → x = (Real.pi / 6) ∨ x = (2 * Real.pi / 3)) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by sorry

end problem_statement_l369_369755


namespace savings_deficit_l369_369124

theorem savings_deficit 
  (income : ℝ) (expenditure_ratio : ℕ) (income_ratio : ℕ) 
  (tax_rate : ℝ) (rent : ℝ) (utilities : ℝ) 
  (monthly_income : ℝ) 
  (h_income_ratio : income_ratio = 9) 
  (h_expenditure_ratio : expenditure_ratio = 8) 
  (h_monthly_income : monthly_income = 36000) 
  (h_tax_rate : tax_rate = 0.10)
  (h_rent : rent = 5000)
  (h_utilities : utilities = 2000) :
  let one_part := monthly_income / income_ratio in
  let total_expenditure := expenditure_ratio * one_part in 
  let tax := tax_rate * monthly_income in
  let remaining_income := monthly_income - tax in
  let total_monthly_costs := rent + utilities + total_expenditure in
  let savings := remaining_income - total_monthly_costs in
  savings = -6600 :=
by
  sorry

end savings_deficit_l369_369124


namespace max_lg_sum_eq_one_min_inv_sum_eq_specific_value_l369_369665

theorem max_lg_sum_eq_one {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 2 * x + 5 * y = 20) :
  ∀ u, u = Real.log x + Real.log y → u ≤ 1 :=
sorry

theorem min_inv_sum_eq_specific_value {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 2 * x + 5 * y = 20) :
  ∀ v, v = (1 / x) + (1 / y) → v ≥ (7 + 2 * Real.sqrt 10) / 20 :=
sorry

end max_lg_sum_eq_one_min_inv_sum_eq_specific_value_l369_369665


namespace gcd_of_18_and_30_l369_369590

-- Define the numbers
def num1 := 18
def num2 := 30

-- State the GCD property
theorem gcd_of_18_and_30 : Nat.gcd num1 num2 = 6 :=
by
  sorry

end gcd_of_18_and_30_l369_369590


namespace measure_of_angle_A_find_length_of_b_l369_369923

open Real

noncomputable def tan (θ : ℝ) : ℝ := sin θ / cos θ

variables {A B C : ℝ} {a b c : ℝ} [triangle_ABC : Triangle ABC]

/-- Problem 1 -/
theorem measure_of_angle_A 
  (h_tanB : tan B = 2) 
  (h_tanC : tan C = 3) : A = π / 4 :=
sorry

/-- Problem 2 -/
theorem find_length_of_b 
  (h_tanB : tan B = 2) 
  (h_tanC : tan C = 3) 
  (h_c : c = 3) : b = 2 * sqrt 2 :=
sorry

end measure_of_angle_A_find_length_of_b_l369_369923


namespace remainder_of_M_l369_369269

def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_of_M : M % 32 = 31 := by
  -- Proof goes here
  sorry

end remainder_of_M_l369_369269


namespace remainder_of_product_of_odd_primes_mod_32_l369_369279

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem remainder_of_product_of_odd_primes_mod_32 :
  (product odd_primes_less_than_32) % 32 = 23 :=
by sorry

end remainder_of_product_of_odd_primes_mod_32_l369_369279


namespace problem_statement_l369_369749

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - 5 * Real.pi / 6)

theorem problem_statement :
  (∀ (x : ℝ), (∀ a b : ℝ, a < b → x ∈ (Set.Ioc (Real.pi / 6) (2 * Real.pi / 3)) → f(x) < f(b)) → x = (Real.pi / 6) ∨ x = (2 * Real.pi / 3)) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by sorry

end problem_statement_l369_369749


namespace product_mod_32_l369_369217

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  M % 32 = 17 := by
  sorry

end product_mod_32_l369_369217


namespace factors_of_M_l369_369307

theorem factors_of_M :
  let M := 57^6 + 6 * 57^5 + 15 * 57^4 + 20 * 57^3 + 15 * 57^2 + 6 * 57 + 1
  in nat.factors_count M = 49 :=
by
  let M := 57^6 + 6 * 57^5 + 15 * 57^4 + 20 * 57^3 + 15 * 57^2 + 6 * 57 + 1
  sorry

end factors_of_M_l369_369307


namespace recurrence_relation_l369_369895

variables {n k : ℕ}

def p : ℕ → ℕ → ℚ := sorry

theorem recurrence_relation (n k : ℕ) (hnk : n ≥ k) :
  p n k = p (n-1) k - (1 / (2^k)) * p (n-k) k + (1 / (2^k)) :=
begin
  sorry
end

end recurrence_relation_l369_369895


namespace phase_shift_of_sine_l369_369522

theorem phase_shift_of_sine {B C : ℝ} (hB : B = 5) (hC : C = 2 * real.pi) :
  (C / B) = (2 * real.pi / 5) :=
by 
  rw [hB, hC]
  sorry

end phase_shift_of_sine_l369_369522


namespace least_whole_number_subtract_ratio_l369_369416

theorem least_whole_number_subtract_ratio : 
  ∃ x : ℕ, (∀ y : ℕ, y < x → (6 - y) * 3 ≥ (7 - y) * 2.29) ∧
           (6 - x) * 3 < (7 - x) * 2.29 ∧
           y ≤ x :=
begin
  sorry
end

end least_whole_number_subtract_ratio_l369_369416


namespace bus_trip_distance_l369_369452

-- Defining the problem variables
variables (x D : ℝ) -- x: speed in mph, D: total distance in miles

-- Main theorem stating the problem
theorem bus_trip_distance
  (h1 : 0 < x) -- speed of the bus is positive
  (h2 : (2 * x + 3 * (D - 2 * x) / (2 / 3 * x) / 2 + 0.75) - 2 - 4 = 0)
  -- The first scenario summarising the travel and delays
  (h3 : ((2 * x + 120) / x + 3 * (D - (2 * x + 120)) / (2 / 3 * x) / 2 + 0.75) - 3 = 0)
  -- The second scenario summarising the travel and delays; accident 120 miles further down
  : D = 720 := sorry

end bus_trip_distance_l369_369452


namespace correct_statements_l369_369822

def f (x : ℝ) : ℝ := sqrt 2 * sin (2 * x - π / 4)

theorem correct_statements :
  ( (∀ x, f(x) = sqrt 2 * cos (2 * (x - 3 * π / 8))) ∨
    (∀ x₁ x₂, f(x₁) * f(x₂) = -2 → |x₁ - x₂| = π) ∨
    (∀ x, f (x + 5 * π / 8) = f (-x + 5 * π / 8)) ∨
    (∀ x, 0 < x ∧ x < π / 4 → -π / 2 < 2 * x - π / 4 ∧ 2 * x - π / 4 < π / 4 ∧
      monotone_on (λ x, f x) (set.Ioo 0 (π / 4))) ) ↔ 
  ((∀ x, f(x) = sqrt 2 * cos (2 * (x - 3 * π / 8))) ∨ 
    (∀ x, 0 < x ∧ x < π / 4 → -π / 2 < 2 * x - π / 4 ∧ 2 * x - π / 4 < π / 4 ∧
        monotone_on (λ x, f x) (set.Ioo 0 (π / 4)) ))
:=
by
  sorry

end correct_statements_l369_369822


namespace gcd_18_30_l369_369540

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l369_369540


namespace find_n_l369_369622

theorem find_n : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 180 ∧ (real.cos (n * real.pi / 180) = real.cos (317 * real.pi / 180)) ∧ n = 43 :=
begin
  sorry
end

end find_n_l369_369622


namespace prime_product_mod_32_l369_369298

open Nat

theorem prime_product_mod_32 : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := List.foldr (· * ·) 1 primes
  M % 32 = 25 :=
by
  sorry

end prime_product_mod_32_l369_369298


namespace min_sum_xy_l369_369657

theorem min_sum_xy (x y : ℕ) (hx : x ≠ y) (pos_x : 0 < x) (pos_y : 0 < y)
  (h : (1 : ℚ) / x + 1 / y = 1 / 12) : x + y = 49 :=
sorry

end min_sum_xy_l369_369657


namespace sin_squared_minus_sin_double_l369_369825

-- From the problem:
variables {a : ℝ} (α : ℝ)
def P : ℝ × ℝ := (2, 3)

-- Conditions:
-- a > 0 and a ≠ 1
axiom a_pos : a > 0
axiom a_neq_one : a ≠ 1
-- P(2, 3) derived from y = log_a(x-1) + 3
axiom P_is_on_graph : ∃ b : ℝ, b = log a (2 - 1) + 3 ∧ P = (2, b)

-- Coordinates and related distances:
def x := (P : ℝ × ℝ).1
def y := (P : ℝ × ℝ).2
def r := real.sqrt (x^2 + y^2)

-- Sine and cosine definitions based on the point P:
noncomputable def sin_alpha := y / r
noncomputable def cos_alpha := x / r

-- The trigonometric identity and calculation part:
theorem sin_squared_minus_sin_double : sin_alpha α ^ 2 - 2 * sin_alpha α * cos_alpha α = -3 / 13 := by
  have cos_alpha_def : cos_alpha α = 2 / real.sqrt 13 := by sorry
  have sin_alpha_def : sin_alpha α = 3 / real.sqrt 13 := by sorry
  have sin_2alpha : 2 * sin_alpha α * cos_alpha α = 12 / 13 := by sorry
  calc
    sin_alpha α ^ 2 - 2 * sin_alpha α * cos_alpha α
      = (3 / real.sqrt 13) ^ 2 - 12 / 13 : by
      rw [sin_alpha_def, cos_alpha_def, sin_2alpha]
      sorry
    ... = –3 / 13 : by sorry

end sin_squared_minus_sin_double_l369_369825


namespace total_surface_area_of_rotated_triangle_eq_36pi_l369_369942

noncomputable def total_surface_area_cone : ℝ :=
  let AB := 3
  let BC := 4
  let AC := 5
  let r := BC
  let l := AC
  π * r * (r + l)

theorem total_surface_area_of_rotated_triangle_eq_36pi
  (AB BC AC : ℝ) (hAB : AB = 3) (hBC : BC = 4) (hAC : AC = 5) :
  total_surface_area_cone = 36 * π :=
by
  rw [total_surface_area_cone, hAB, hBC, hAC]
  sorry

end total_surface_area_of_rotated_triangle_eq_36pi_l369_369942


namespace gcd_of_18_and_30_l369_369556

-- Define the numbers
def a := 18
def b := 30

-- The main theorem statement
theorem gcd_of_18_and_30 : Nat.gcd a b = 6 := by
  sorry

end gcd_of_18_and_30_l369_369556


namespace recurrence_relation_l369_369898

variables {n k : ℕ}

def p : ℕ → ℕ → ℚ := sorry

theorem recurrence_relation (n k : ℕ) (hnk : n ≥ k) :
  p n k = p (n-1) k - (1 / (2^k)) * p (n-k) k + (1 / (2^k)) :=
begin
  sorry
end

end recurrence_relation_l369_369898


namespace gcd_18_30_l369_369576

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := 
by
  sorry

end gcd_18_30_l369_369576


namespace intersection_of_A_and_B_l369_369660

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := { x | ∃ m : ℕ, x = 2 * m }

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := 
by sorry

end intersection_of_A_and_B_l369_369660


namespace sum_first_10_terms_possible_values_x_exists_k_l369_369655

-- Conditions for the sequence
def sequence (a : ℕ → ℕ) := ∀ n : ℕ, a (n + 2) = |a (n + 1) - a n|

-- Question 1: Prove the sum of the first 10 terms is 8
theorem sum_first_10_terms (a : ℕ → ℕ) (h : sequence a) (h1 : a 1 = 1) (h2 : a 2 = 2) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = 8 :=
sorry

-- Question 2: Possible values of x for the sequence to contain exactly 100 zeros in first 2017 terms
theorem possible_values_x (x : ℤ) (a : ℕ → ℤ) (h : ∀ n : ℕ, a (n + 2) = |a (n + 1) - a n|)
  (h1 : a 1 = 1) (h2 : a 2 = x) (hx0 : (finset.range 2017).filter (λ n, a n = 0).card = 100) :
  x = 1144 ∨ x = 1145 ∨ x = -1141 ∨ x = -1140 :=
sorry

-- Question 3: Prove there exists k such that 0 ≤ a_k < 1
theorem exists_k (a : ℕ → ℕ) (h : sequence a) : 
  ∃ k : ℕ, 0 ≤ a k ∧ a k < 1 :=
sorry

end sum_first_10_terms_possible_values_x_exists_k_l369_369655


namespace symmetric_line_equation_l369_369919

noncomputable def line_symmetric_equation (l1 l2 l3 : ℝ → ℝ → Prop) : ℝ → ℝ → Prop :=
  ∀ x y, l2 (2, 2) (x, y) → l1 (1, 0) (4, 3) → l3 x y

def given_line1 (x y : ℝ) : Prop := 2 * x - y - 2 = 0
def given_line2 (x y : ℝ) : Prop := x + y - 4 = 0 
def resulting_line (x y : ℝ) : Prop := x - 2 * y + 2 = 0

theorem symmetric_line_equation :
  line_symmetric_equation given_line1 given_line2 resulting_line :=
by
  sorry

end symmetric_line_equation_l369_369919


namespace move_coins_to_single_tray_l369_369048

/-- Given n trays and m coins with n, m > 3,
    prove that it is possible to move all coins to a single tray
    by taking one coin from two trays and placing them on a third tray. -/
theorem move_coins_to_single_tray (n m : ℕ) (hn : n > 3) (hm : m > 3) :
  ∃ t : ℕ, t = m ∧ (∃ k, k < n ∧ ∀ i : ℕ, i ≠ k → 0) :=
sorry

end move_coins_to_single_tray_l369_369048


namespace polynomial_identity_or_perfect_power_l369_369132

open Int Polynomial

def isPerfectPower (n : ℤ) : Prop :=
  ∃ (m k : ℤ), k ≥ 2 ∧ m^k = n

def isPerfectPowerPoly (P : Polynomial ℤ) : Prop :=
  ∃ (Q : Polynomial ℤ) (k : ℤ), k ≥ 2 ∧ Q^k = P

theorem polynomial_identity_or_perfect_power (P : Polynomial ℤ) 
  (h_non_const : P.degree > 0)
  (h_condition : ∀ n : ℤ, isPerfectPower n → isPerfectPower (P.eval n)) :
  P = Polynomial.X ∨ isPerfectPowerPoly P :=
  sorry

end polynomial_identity_or_perfect_power_l369_369132


namespace julia_mean_score_l369_369043

theorem julia_mean_score (s1 s2 s3 s4 s5 s6 s7 s8 : ℝ) (h_mean : ℝ) (h_count : ℕ) (total_sum : ℝ) :
  s1 = 88 → s2 = 90 → s3 = 92 → s4 = 94 → s5 = 95 → s6 = 97 → s7 = 98 → s8 = 99 →
  h_mean = 94 →
  h_count = 4 →
  total_sum = 753 →
  let henry_total := h_mean * h_count in
  let julia_total := total_sum - henry_total in
  let j_mean := julia_total / 4 in
  j_mean = 94.25 :=
by {
  intros,
  sorry
}

end julia_mean_score_l369_369043


namespace recurrence_relation_p_series_l369_369909

noncomputable def p_series (n k : ℕ) : ℝ :=
if k < n then (p_series (n - 1) k - (1 / (2 : ℝ)^k) * p_series (n - k) k + (1 / (2 : ℝ)^k))
else 0

-- Statement of the theorem
theorem recurrence_relation_p_series (n k : ℕ) (h : k < n) :
  p_series n k = p_series (n - 1) k - (1 / (2 : ℝ)^k) * p_series (n - k) k + (1 / (2 : ℝ)^k) :=
sorry

end recurrence_relation_p_series_l369_369909


namespace product_mod_32_is_15_l369_369245

-- Define the odd primes less than 32
def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the product of odd primes
def M : ℕ := List.foldl (· * ·) 1 oddPrimes

-- The theorem to prove the remainder when M is divided by 32 is 15
theorem product_mod_32_is_15 : M % 32 = 15 := by
  sorry

end product_mod_32_is_15_l369_369245


namespace vectors_perpendicular_l369_369083

variables {α β : ℝ}

def vector_a : ℝ × ℝ := (Real.cos α, Real.sin α)
def vector_b : ℝ × ℝ := (Real.cos β, Real.sin β)

theorem vectors_perpendicular :
  (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2) 
    • (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2) = 0 := by
  sorry

end vectors_perpendicular_l369_369083


namespace maximum_xy_value_l369_369870

theorem maximum_xy_value :
  ∃ (x y : ℕ), 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by
  sorry

end maximum_xy_value_l369_369870


namespace product_of_odd_primes_mod_32_l369_369189

theorem product_of_odd_primes_mod_32 :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  M % 32 = 17 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  exact sorry

end product_of_odd_primes_mod_32_l369_369189


namespace find_f_neg_5pi_12_l369_369771

-- Conditions
def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

variables {ω φ : ℝ}
def f (x : ℝ) : ℝ := Real.sin (ω * x + φ)

axiom monotonicity_condition : is_monotonically_increasing f (π / 6) (2 * π / 3)
axiom symmetry_condition_1 : f (π / 6) = f (2 * π / 3)
axiom symmetry_condition_2 : f (-π / 6) = f (π / 2)

-- Goal
theorem find_f_neg_5pi_12 : f (- 5 * π / 12) = sqrt 3 / 2 := 
sorry

end find_f_neg_5pi_12_l369_369771


namespace product_mod_32_is_15_l369_369246

-- Define the odd primes less than 32
def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the product of odd primes
def M : ℕ := List.foldl (· * ·) 1 oddPrimes

-- The theorem to prove the remainder when M is divided by 32 is 15
theorem product_mod_32_is_15 : M % 32 = 15 := by
  sorry

end product_mod_32_is_15_l369_369246


namespace find_counterexample_l369_369628

-- Define the given cards
inductive Card
| Y | R | two | three | five

-- Define prime property
def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5

-- Define consonant property
def is_consonant : Card → Prop
| Card.Y := true
| Card.R := true
| _      := false

-- Define the proof statement
theorem find_counterexample : 
  ∃ card : Card, is_consonant card ∧ ¬ is_prime 2 :=
by
  existsi Card.Y
  simp
  sorry

end find_counterexample_l369_369628


namespace george_hourly_rate_l369_369633

theorem george_hourly_rate (total_hours : ℕ) (total_amount : ℕ) (h1 : total_hours = 7 + 2)
  (h2 : total_amount = 45) : 
  total_amount / total_hours = 5 := 
by sorry

end george_hourly_rate_l369_369633


namespace gcd_18_30_is_6_l369_369618

def gcd_18_30 : ℕ :=
  gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 :=
by {
  -- The step here will involve using properties of gcd and prime factorization,
  -- but we are given the result directly for the purpose of this task.
  sorry
}

end gcd_18_30_is_6_l369_369618


namespace maximum_value_product_cube_expression_l369_369306

theorem maximum_value_product_cube_expression (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) :
  (x^3 - x * y^2 + y^3) * (x^3 - x^2 * z + z^3) * (y^3 - y^2 * z + z^3) ≤ 1 :=
sorry

end maximum_value_product_cube_expression_l369_369306


namespace ratio_of_boys_l369_369109

theorem ratio_of_boys (p : ℝ) (h : p = (3 / 5) * (1 - p)) 
  : p = 3 / 8 := 
by
  sorry

end ratio_of_boys_l369_369109


namespace polygon_sides_sum_720_l369_369381

theorem polygon_sides_sum_720 (n : ℕ) (h1 : (n - 2) * 180 = 720) : n = 6 := by
  sorry

end polygon_sides_sum_720_l369_369381


namespace align_two_pieces_l369_369328

theorem align_two_pieces (
  (A B C D : ℤ × ℤ) : 
  ∃ f : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ), 
  (∀ X Y : ℤ × ℤ, f X Y = (X.1 - Y.1, X.2 - Y.2) ∨ f X Y = (Y.1 - X.1, Y.2 - X.2)) ∧ 
  ∀ U V : ℤ × ℤ, U = f U' V' → V = f U V → U = V)
  --> ∃ moves : list ((ℤ × ℤ) × (ℤ × ℤ)), 
          (∀ m : (ℤ × ℤ) × (ℤ × ℤ), m ∈ moves → 
              ∃ X Y, m = (X, f X Y)) ∧ 
          (list.nth moves (list.length moves - 1)).fst = (predetermined_1 : ℤ × ℤ) ∧ 
          (list.nth moves (list.length moves - 1)).snd = (predetermined_2 : ℤ × ℤ)
) : 
  ∃ moves : list ((ℤ × ℤ) × (ℤ × ℤ)), 
    (∀ m : (ℤ × ℤ) × (ℤ × ℤ), 
        m.1 = predetermined_1 ∧ m.2 = predetermined_2).

end align_two_pieces_l369_369328


namespace sarah_trips_to_fill_barrel_l369_369346

noncomputable def volume_cylindrical_barrel (radius_barrel : ℝ) (height_barrel : ℝ) : ℝ :=
  π * radius_barrel^2 * height_barrel

noncomputable def volume_hemisphere_bucket (radius_bucket : ℝ) : ℝ :=
  (2 / 3) * π * radius_bucket^3

noncomputable def trips_to_fill_barrel (radius_barrel : ℝ) (height_barrel : ℝ) (radius_bucket : ℝ) : ℕ :=
  let volume_barrel := volume_cylindrical_barrel radius_barrel height_barrel
  let volume_bucket := volume_hemisphere_bucket radius_bucket
  let trips := volume_barrel / volume_bucket
  nat.ceil trips

theorem sarah_trips_to_fill_barrel : trips_to_fill_barrel 5 10 4 = 6 := by
  sorry

end sarah_trips_to_fill_barrel_l369_369346


namespace remainder_of_M_when_divided_by_32_l369_369205

open Nat

def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
def M : ℕ := List.foldl (*) 1 oddPrimes

theorem remainder_of_M_when_divided_by_32 :
  M % 32 = 1 :=
sorry

end remainder_of_M_when_divided_by_32_l369_369205


namespace probability_one_letter_from_each_l369_369496

theorem probability_one_letter_from_each
  (total_cards : ℕ)
  (adam_cards : ℕ)
  (brian_cards : ℕ)
  (h1 : total_cards = 12)
  (h2 : adam_cards = 4)
  (h3 : brian_cards = 6)
  : (4/12 * 6/11) + (6/12 * 4/11) = 4/11 := by
  sorry

end probability_one_letter_from_each_l369_369496


namespace gcd_18_30_l369_369605

-- Define the two numbers
def num1 : ℕ := 18
def num2 : ℕ := 30

-- State the theorem to find the gcd
theorem gcd_18_30 : Nat.gcd num1 num2 = 6 := by
  sorry

end gcd_18_30_l369_369605


namespace gcd_18_30_l369_369538

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l369_369538


namespace min_coord_of_quadratic_l369_369015

variables {p q : ℝ} (hp : 0 < p) (hq : 0 < q)

/-- The x-coordinate of the minimum value of f(x) = x^2 + px + qx is - (p + q) / 2, given p, q > 0 -/
theorem min_coord_of_quadratic (hp : p > 0) (hq : q > 0) : 
  let f := λ x : ℝ, x^2 + p * x + q * x in
  exists x_min, ∀ x, f x_min ≤ f x ∧ x_min = -(p + q) / 2 :=
by
  sorry

end min_coord_of_quadratic_l369_369015


namespace greatest_n_less_than_1000_l369_369631

noncomputable def h (x : ℕ) : ℕ :=
  if even x then (max (λ j : ℕ, if 3^j ∣ x then some (3^j) else none)).get_or_else 1 else 1

noncomputable def T (n : ℕ) : ℕ :=
  (Finset.range (3^n)).sum (λ k, h (3 * (k + 1)))

theorem greatest_n_less_than_1000 (n : ℕ) :
  n < 1000 → (∀ m, m < 1000 → T m ≠ T m) → T 899 =
sorry

end greatest_n_less_than_1000_l369_369631


namespace quarters_left_after_purchase_l369_369981

theorem quarters_left_after_purchase (quarters : ℕ) (cost_per_quarter : ℝ) (total_cost : ℝ) :
  quarters = 375 ∧ cost_per_quarter = 0.25 ∧ total_cost = 42.63 →
  let quarters_needed := ⌊total_cost / cost_per_quarter⌋ in
  let remaining_quarters := quarters - quarters_needed in
  remaining_quarters = 205 :=
by
  intros h
  obtain ⟨h₁, h₂, h₃⟩ := h
  let q_needed := ⌊h₃ / h₂⌋
  have q_needed_val : q_needed = 170 := by sorry
  let r_quarters := h₁ - q_needed
  have r_quarters_val : r_quarters = 205 := by
    rw [q_needed_val]
    exact Nat.sub_self 205
  exact r_quarters_val

end quarters_left_after_purchase_l369_369981


namespace number_of_sedans_total_vehicles_l369_369459

-- Definitions of the conditions
def sales_ratio_sports_to_sedans : ℕ × ℕ := (3, 5)
def predicted_sports_sales : ℕ := 36

-- Theorem for the number of sedans
theorem number_of_sedans (sales_ratio : ℕ × ℕ) (predicted_sports : ℕ) : 
  sales_ratio = (3, 5) → predicted_sports = 36 → (5 * predicted_sports) / 3 = 6 :=
by
  intros h_ratio h_predicted
  rw [h_ratio, h_predicted]
  sorry

-- Theorem for the total number of vehicles
theorem total_vehicles (sales_ratio : ℕ × ℕ) (predicted_sports : ℕ) : 
  sales_ratio = (3, 5) → predicted_sports = 36 → ((5 * predicted_sports) / 3) + predicted_sports = 96 :=
by
  intros h_ratio h_predicted
  rw [h_ratio, h_predicted]
  sorry

end number_of_sedans_total_vehicles_l369_369459


namespace map_distance_ram_location_l369_369326

-- Definitions for given conditions
def map_distance_between_mountains := 312 -- inches
def actual_distance_between_mountains := (136: ℝ) -- km
def actual_distance_from_base := (18.307692307692307: ℝ) -- km

-- The theorem to be proved: the map distance of Ram's location from the base of the mountain is approximately 1065.75 inches
theorem map_distance_ram_location : 
  let scale := actual_distance_between_mountains * 1000 / (map_distance_between_mountains * 0.0254) in
  let map_distance := actual_distance_from_base * 1000 / scale in
  map_distance ≈ 1065.75 := 
by
  sorry

end map_distance_ram_location_l369_369326


namespace gcd_of_18_and_30_l369_369566

-- Define the numbers
def a := 18
def b := 30

-- The main theorem statement
theorem gcd_of_18_and_30 : Nat.gcd a b = 6 := by
  sorry

end gcd_of_18_and_30_l369_369566


namespace sum_of_squares_max_l369_369632

def quadratic_eq (a : ℝ) : Polynomial ℝ := 
  Polynomial.C (2 * a^2 + 4 * a + 3) + Polynomial.C (2 * a) * Polynomial.X + Polynomial.C 1 * Polynomial.X^2

theorem sum_of_squares_max (a : ℝ) (x1 x2 : ℝ) :
  a ∈ set.Icc (-3 : ℝ) (-1 : ℝ) ∧
  (quadratic_eq a).roots = {x1, x2} →
  let sum_of_squares := x1^2 + x2^2 in
  sum_of_squares ≤ 18 ∧ 
  (a = -3 → sum_of_squares = 18) :=
sorry

end sum_of_squares_max_l369_369632


namespace value_of_f_neg_5π_over_12_l369_369793

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem value_of_f_neg_5π_over_12 :
  ∀ (ω φ : ℝ), 
    (∀ x y : ℝ, (x < y ∧ x ∈ Ioo (π / 6) (2 * π / 3) ∧ y ∈ Ioo (π / 6) (2 * π / 3)) → f x ω φ < f y ω φ) ∧ 
    f (π / 6) ω φ = f (2 * π / 3) ω φ → 
    f (-5 * π / 12) ω φ = √3 / 2 :=
by
  sorry

end value_of_f_neg_5π_over_12_l369_369793


namespace ten_digit_numbers_with_digit_sum_l369_369842

noncomputable def num_ten_digit_numbers_with_digit_sum (n : ℕ) : ℕ :=
  ∑ i in (finset.range 10), if i = n then 1 else if 0 ≤ n - i ≤ 8 then (finset.range 10).card.choose (n - i) else 0

theorem ten_digit_numbers_with_digit_sum :
  num_ten_digit_numbers_with_digit_sum 2 = 46 ∧
  num_ten_digit_numbers_with_digit_sum 3 = 166 ∧
  num_ten_digit_numbers_with_digit_sum 4 = 361 :=
by
  sorry

end ten_digit_numbers_with_digit_sum_l369_369842


namespace gcd_18_30_l369_369603

-- Define the two numbers
def num1 : ℕ := 18
def num2 : ℕ := 30

-- State the theorem to find the gcd
theorem gcd_18_30 : Nat.gcd num1 num2 = 6 := by
  sorry

end gcd_18_30_l369_369603


namespace exists_polynomial_perfect_powers_l369_369004

theorem exists_polynomial_perfect_powers (m n : ℕ) (hm : 2 ≤ m) (hn : 1 ≤ n) :
  ∃ P : Polynomial ℤ, P.degree = n ∧ ∀ x ∈ Finset.range (n + 1), ∃ k : ℕ, P.eval x = m ^ k :=
by { sorry }

end exists_polynomial_perfect_powers_l369_369004


namespace max_xy_value_l369_369888

theorem max_xy_value (x y : ℕ) (h : 7 * x + 4 * y = 140) : xy ≤ 168 :=
by sorry

end max_xy_value_l369_369888


namespace max_xy_value_l369_369883

theorem max_xy_value (x y : ℕ) (h : 7 * x + 4 * y = 140) : xy ≤ 168 :=
by sorry

end max_xy_value_l369_369883


namespace remainder_of_M_when_divided_by_32_l369_369197

open Nat

def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
def M : ℕ := List.foldl (*) 1 oddPrimes

theorem remainder_of_M_when_divided_by_32 :
  M % 32 = 1 :=
sorry

end remainder_of_M_when_divided_by_32_l369_369197


namespace gcd_of_18_and_30_l369_369565

-- Define the numbers
def a := 18
def b := 30

-- The main theorem statement
theorem gcd_of_18_and_30 : Nat.gcd a b = 6 := by
  sorry

end gcd_of_18_and_30_l369_369565


namespace max_k_value_l369_369117

def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8 * x + 15 = 0

def line_L (k x y : ℝ) : Prop := y = k * x - 2

def intersects_with_circle (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 4

theorem max_k_value : ∀ (k : ℝ),
  (∃ (x y : ℝ), line_L k x y ∧ intersects_with_circle x y) → 
  (∀ k, 0 ≤ k ∧ k ≤ 4 / 3).

end max_k_value_l369_369117
