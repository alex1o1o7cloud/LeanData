import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Arithmetic.Basic
import Mathlib.Algebra.GCDMonoid.Basic
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Order.Basic
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Gcd.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set
import Mathlib.Data.Set.Basic
import Mathlib.NumberTheory.Cyclotomic.Basic
import Mathlib.NumberTheory.PythagoreanTriples
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.Stats
import Mathlib.Real
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

namespace sin_180_degree_l530_530635

theorem sin_180_degree : Real.sin (Real.pi) = 0 := by sorry

end sin_180_degree_l530_530635


namespace remainder_of_a2020_mod_22_l530_530338

def sequence (n : ℕ) : ℕ :=
  if n = 1 then 3
  else if n = 2 then 3
  else sequence (n-1) * sequence (n-2) - sequence (n-1) - sequence (n-2) + 2

theorem remainder_of_a2020_mod_22 : sequence 2020 % 22 = 11 := by
  sorry

end remainder_of_a2020_mod_22_l530_530338


namespace cricket_player_average_l530_530116

theorem cricket_player_average (A : ℕ)
  (H1 : 10 * A + 62 = 11 * (A + 4)) : A = 18 :=
by {
  sorry -- The proof itself
}

end cricket_player_average_l530_530116


namespace carls_sequence_l530_530610

def sequence (a : ℕ) (b : ℕ → ℕ) (n : ℕ) : ℕ :=
  if h : n = 0 then a else (a * (List.product (List.map b (List.range n))))

theorem carls_sequence :
  ∃ (a : ℕ) (b : ℕ → ℕ),
  (∀ n, 1 ≤ b n) ∧
  (a, b) ≠ (0, λ _, 1) ∧
  (sequence a b 9 > 600) ∧
  (sequence a b 9 < 1000) ∧
  (sequence a b 9 = 768) := sorry

end carls_sequence_l530_530610


namespace sum_of_ages_correct_l530_530367

noncomputable def sum_of_ages := sorry

theorem sum_of_ages_correct (P S : ℕ) (h1 : P = S + 8) (h2 : P + 6 = 3 * (S - 2)) : 
  P + S = 28 :=
sorry

end sum_of_ages_correct_l530_530367


namespace solve_for_a_and_b_l530_530266

theorem solve_for_a_and_b (a b : ℤ) (h1 : 5 + a = 6 - b) (h2 : 6 + b = 9 + a) : 5 - a = 6 := 
sorry

end solve_for_a_and_b_l530_530266


namespace min_max_value_monotonic_range_l530_530243

noncomputable def f (a x : ℝ) : ℝ := x^2 + 2 * a * x + 2

-- Problem Ⅰ
theorem min_max_value (a : ℝ) (h_a : a = -1) : 
  (∀ x : ℝ, x ∈ Icc (-5 : ℝ) (5 : ℝ) → f a x ≥ 1) ∧
  (∃ x : ℝ, x ∈ Icc (-5 : ℝ) (5 : ℝ) ∧ f a x = 1) ∧
  (∀ x : ℝ, x ∈ Icc (-5 : ℝ) (5 : ℝ) → f a x ≤ 37) ∧
  (∃ x : ℝ, x ∈ Icc (-5 : ℝ) (5 : ℝ) ∧ f a x = 37) := 
sorry

-- Problem Ⅱ
theorem monotonic_range :
  (∀ a : ℝ, (∀ x1 x2 : ℝ, x1 ∈ Icc (-5 : ℝ) (5 : ℝ) → x2 ∈ Icc (-5 : ℝ) (5 : ℝ) → x1 ≤ x2 → f a x1 ≤ f a x2) ∨
  (∀ x1 x2 : ℝ, x1 ∈ Icc (-5 : ℝ) (5 : ℝ) → x2 ∈ Icc (-5 : ℝ) (5 : ℝ) → x1 ≤ x2 → f a x1 ≥ f a x2)) ↔
  (a ∈ Icc (-∞ : ℝ) (-5 : ℝ) ∨ a ∈ Icc (5 : ℝ) (∞ : ℝ)) :=
sorry

end min_max_value_monotonic_range_l530_530243


namespace max_min_expression_l530_530201

theorem max_min_expression :
  ∀ (x y z : ℝ), 
  (x ≥ 0) → (y ≥ 0) → (z ≥ 0) → 
  (x - 2 * y - 3 * z = -10) → 
  (x + 2 * y + z = 6) → 
  let A := 1.5 * x + y - z in 
  (A ≤ 0) ∧ (A ≥ -1) :=
by
  intro x y z h_x h_y h_z h_eq1 h_eq2 A_def
  have key := calc
    A = 1.5 * x + y - z : by simp [A_def]
      ... ≤ sorry -- this is where the detailed steps to prove the constraints would go
  exact ⟨sorry, sorry⟩ -- completing the placeholders for (A ≤ 0) and (A ≥ -1)

end max_min_expression_l530_530201


namespace determine_k_l530_530883

-- Define the incenter I of triangle ABC, arbitrary point P, and required distances.
variables {A B C P : Type} 
variables [inner_product_space ℝ A] [inner_product_space ℝ B]
variables (a b c : ℝ)
variables (p a_vec b_vec c_vec i : A)

-- Define the position of the incenter.
def incenter (a_vec b_vec c_vec : A) (a b c : ℝ) : A :=
  (a • a_vec + b • b_vec + c • c_vec) / (a + b + c)

-- Specify that i is the incenter
def i_ (a_vec b_vec c_vec : A) (a b c : ℝ) : A := incenter a_vec b_vec c_vec a b c

-- Define squared distances for given points.
def squared_distance (x y : A) : ℝ := ∥x - y∥^2

#check squared_distance p a_vec
#check squared_distance p b_vec
#check squared_distance p c_vec
#check squared_distance i_ a_vec
#check squared_distance i_ b_vec
#check squared_distance i_ c_vec

theorem determine_k (a b c : ℝ) (A B C P : A) (i := incenter A B C a b c) :
  squared_distance P A + squared_distance P B + squared_distance P C = 
  3 * squared_distance P i + squared_distance i A + squared_distance i B + squared_distance i C :=
sorry

end determine_k_l530_530883


namespace halved_area_of_T3_l530_530439

theorem halved_area_of_T3 (T1_area : ℝ) (h1 : T1_area = 36) : 
  let T2_side := (2 / 3) * real.sqrt T1_area,
      T3_side := (2 / 3) * T2_side in 
  (T3_side ^ 2 / 2) = 32 / 9 :=
by
  let T2_side := (2 / 3) * real.sqrt T1_area,
      T3_side := (2 / 3) * T2_side
  calc (T3_side ^ 2 / 2) = ((2 / 3 * real.sqrt T1_area * / (1 / 3)) ^ 2 / 2) : by sorry
                     ... = 32 / 9 : by sorry

end halved_area_of_T3_l530_530439


namespace planes_meet_in_50_minutes_l530_530906

noncomputable def time_to_meet (d : ℕ) (vA vB : ℕ) : ℚ :=
  d / (vA + vB : ℚ)

theorem planes_meet_in_50_minutes
  (d : ℕ) (vA vB : ℕ)
  (h_d : d = 500) (h_vA : vA = 240) (h_vB : vB = 360) :
  (time_to_meet d vA vB * 60 : ℚ) = 50 := by
  sorry

end planes_meet_in_50_minutes_l530_530906


namespace triangle_side_difference_l530_530866

theorem triangle_side_difference : 
    ∀ y : ℕ, (2 < y) → (y < 16) → 
    (max (finset.range 16) ∈ set_of (λ y, 2 < y ∧ y < 16)) - 
    (min (finset.range 16) ∈ set_of (λ y, 2 < y ∧ y < 16)) = 12 :=
sorry

end triangle_side_difference_l530_530866


namespace intersection_A_B_range_of_a_l530_530229

-- Problem 1: Prove the intersection of A and B when a = 4
theorem intersection_A_B (a : ℝ) (h : a = 4) :
  { x : ℝ | 5 ≤ x ∧ x ≤ 7 } ∩ { x : ℝ | x ≤ 3 ∨ 5 < x} = {6, 7} :=
by sorry

-- Problem 2: Prove the range of values for a such that A ⊆ B
theorem range_of_a :
  { a : ℝ | (a < 2) ∨ (a > 4) } :=
by sorry

end intersection_A_B_range_of_a_l530_530229


namespace total_packs_of_groceries_is_14_l530_530350

-- Define the number of packs of cookies
def packs_of_cookies : Nat := 2

-- Define the number of packs of cakes
def packs_of_cakes : Nat := 12

-- Define the total packs of groceries as the sum of packs of cookies and cakes
def total_packs_of_groceries : Nat := packs_of_cookies + packs_of_cakes

-- The theorem which states that the total packs of groceries is 14
theorem total_packs_of_groceries_is_14 : total_packs_of_groceries = 14 := by
  -- this is where the proof would go
  sorry

end total_packs_of_groceries_is_14_l530_530350


namespace topsoil_cost_l530_530049

theorem topsoil_cost 
  (cost_per_cubic_foot : ℝ)
  (cubic_yards_to_cubic_feet : ℝ)
  (cubic_yards : ℝ) :
  cost_per_cubic_foot = 8 →
  cubic_yards_to_cubic_feet = 27 →
  cubic_yards = 8 →
  (cubic_yards * cubic_yards_to_cubic_feet * cost_per_cubic_foot) = 1728 :=
begin
  intros h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
end

end topsoil_cost_l530_530049


namespace cos_triple_angle_l530_530838

theorem cos_triple_angle
  (θ : ℝ)
  (h : Real.cos θ = 1/3) :
  Real.cos (3 * θ) = -23 / 27 :=
by
  sorry

end cos_triple_angle_l530_530838


namespace average_of_other_two_numbers_l530_530011

theorem average_of_other_two_numbers 
  (a b c d : ℕ) 
  (h_diff : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a + b + c + d = 28) 
  (h_max_diff : (max a (max b (max c d)) - min a (min b (min c d)))) 
  : (3 + 4) / 2 = 3.5 := 
begin
  -- sorry as proof steps are omitted as per instructions
  sorry
end

end average_of_other_two_numbers_l530_530011


namespace probability_double_head_l530_530421

/-- There are three types of coins:
  - Coin 1: one head and one tail
  - Coin 2: two heads
  - Coin 3: two tails
  A coin is randomly selected and flipped, resulting in heads. 
  Prove that the probability that the coin is Coin 2 (the double head coin)
  is 2/3. -/
theorem probability_double_head (h : true) : 
  let Coin1 : ℕ := 1,
      Coin2 : ℕ := 2,
      Coin3 : ℕ := 0,
      totalHeads : ℕ := Coin1 + Coin2 + Coin3
  in 
  let p_Coin1 := 1 / 3,
      p_Coin2 := 1 / 3,
      p_Coin3 := 1 / 3,
      p_Heads_given_Coin1 := 1 / 2,
      p_Heads_given_Coin2 := 1,
      p_Heads_given_Coin3 := 0,
      p_Heads := (p_Heads_given_Coin1 * p_Coin1) + (p_Heads_given_Coin2 * p_Coin2) + (p_Heads_given_Coin3 * p_Coin3)
  in (p_Heads_given_Coin2 * p_Coin2) / p_Heads = 2 / 3 :=
by
  sorry

end probability_double_head_l530_530421


namespace largest_n_factorial_product_l530_530189

theorem largest_n_factorial_product :
  ∃ n : ℕ, (∀ a : ℕ, (n > 0) → (n! = (∏ k in finset.range (n - 4 + a), k + 1))) → n = 4 :=
begin
  sorry
end

end largest_n_factorial_product_l530_530189


namespace probability_one_common_course_l530_530976

theorem probability_one_common_course :
  ∃ p : ℚ, 
  p = 2 / 3 ∧ 
  ∑ (x in Finset.powersetLen 2 (Finset.range 4)), 
    1 = 6 ∧
  ∑ (x in Finset.powersetLen 2 (Finset.range 4)), 
    1 = 6 := 
sorry

end probability_one_common_course_l530_530976


namespace letter_150th_in_pattern_l530_530490

def repeating_sequence := "XYZ"

def letter_at_position (n : ℕ) : char :=
  let seq := repeating_sequence.to_list
  seq.get! ((n - 1) % seq.length)

theorem letter_150th_in_pattern : letter_at_position 150 = 'Z' :=
by sorry

end letter_150th_in_pattern_l530_530490


namespace cuboid_total_edge_length_cuboid_surface_area_l530_530699

variables (a b c : ℝ)

theorem cuboid_total_edge_length : 4 * (a + b + c) = 4 * (a + b + c) := 
by
  sorry

theorem cuboid_surface_area : 2 * (a * b + b * c + a * c) = 2 * (a * b + b * c + a * c) := 
by
  sorry

end cuboid_total_edge_length_cuboid_surface_area_l530_530699


namespace parallelogram_area_l530_530091

theorem parallelogram_area (base height : ℝ) (h_base : base = 24) (h_height : height = 10) :
  base * height = 240 :=
by
  rw [h_base, h_height]
  norm_num

end parallelogram_area_l530_530091


namespace orchids_reduction_l530_530670

noncomputable def year_orchids_less_than_5_percent (N0 : ℕ) : ℕ :=
  let n : ℕ := nat_ceil (log (0.05) / log (0.7)) in
  2010 + n

theorem orchids_reduction (N0 : ℕ) :
  year_orchids_less_than_5_percent N0 = 2019 :=
by
  let n := nat_ceil (log (0.05) / log (0.7))
  have hn : n = 9 := sorry
  show 2010 + n = 2019, by rw [hn]; refl

end orchids_reduction_l530_530670


namespace a_4_value_geometric_seq_property_general_formula_l530_530342

-- Given conditions
def a (n : ℕ) : ℚ := if n = 1 then 1 else if n = 2 then 3/2 else if n = 3 then 5/4 else 0

def S : ℕ → ℚ
| 0     := 0
| (n+1) := S n + a (n+1)

axiom sum_recurrence (n : ℕ) (h : n ≥ 2) : 4 * S (n + 2) + 5 * S n = 8 * S (n + 1) + S (n - 1)

-- Proof statements
theorem a_4_value : a 4 = 7/8 := sorry

theorem geometric_seq_property : ∀ n : ℕ, n ≥ 1 → ∃ r : ℚ, r = 1 / 2 ∧ (a (n+1) - 1/2 * a n) = r ^ n := sorry

theorem general_formula (n : ℕ) : a n = (2 * n - 1) / (2 ^ (n - 1)) := sorry

end a_4_value_geometric_seq_property_general_formula_l530_530342


namespace nth_150th_letter_in_XYZ_l530_530485

def pattern : List Char := ['X', 'Y', 'Z']

def nth_letter (n : Nat) : Char :=
  pattern[(n - 1) % pattern.length]

theorem nth_150th_letter_in_XYZ :
  nth_letter 150 = 'Z' :=
by
  sorry

end nth_150th_letter_in_XYZ_l530_530485


namespace smallest_positive_integer_divisible_conditions_l530_530077

theorem smallest_positive_integer_divisible_conditions : 
  ∃ (M : ℕ), (M % 4 = 3) ∧ 
             (M % 5 = 4) ∧ 
             (M % 6 = 5) ∧ 
             (M % 7 = 6) ∧ 
             (∀ N : ℕ, ((N % 4 = 3) ∧ 
                        (N % 5 = 4) ∧ 
                        (N % 6 = 5) ∧ 
                        (N % 7 = 6)) → 
                        N ≥ M) :=
begin
  use 419,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  intros N hN,
  cases hN with hN1 hN23,
  cases hN23 with hN2 hN3,
  cases hN3 with hN4 hN5,
  sorry,
end

end smallest_positive_integer_divisible_conditions_l530_530077


namespace minimum_value_l530_530074

-- Given conditions
variables (a b c d : ℝ)
variables (h_a : a > 0) (h_b : b = 0) (h_a_eq : a = 1)

-- Define the function
def f (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- The statement to prove
theorem minimum_value (h_c : c = 0) : ∃ x : ℝ, f a b c d x = d :=
by
  -- Given the conditions a=1, b=0, and c=0, we need to show that the minimum value is d
  sorry

end minimum_value_l530_530074


namespace no_apples_info_l530_530122

theorem no_apples_info (r d : ℕ) (condition1 : r = 79) (condition2 : d = 53) (condition3 : r = d + 26) : 
  ∀ a : ℕ, (a = a) → false :=
by
  intro a h
  sorry

end no_apples_info_l530_530122


namespace sin_180_eq_zero_l530_530621

theorem sin_180_eq_zero : Real.sin (180 * Real.pi / 180) = 0 := by
  -- Simplifying the angle, 180 degrees = π radians
  let angle := 180 * Real.pi / 180
  have h : angle = Real.pi := by
    simp [angle, Real.pi]
  rw h
  -- From the unit circle, we know the sine of angle π is 0
  exact Real.sin_pi

end sin_180_eq_zero_l530_530621


namespace final_price_after_reductions_l530_530571

variable original_price : ℝ := 20
variable first_reduction_rate : ℝ := 0.2
variable second_reduction_rate : ℝ := 0.5

theorem final_price_after_reductions : 
  (original_price * (1 - first_reduction_rate)) * (1 - second_reduction_rate) = 8 := by
  sorry

end final_price_after_reductions_l530_530571


namespace base12_div_remainder_9_l530_530994

def base12_to_base10 (d0 d1 d2 d3: ℕ) : ℕ :=
  d0 * 12^3 + d1 * 12^2 + d2 * 12^1 + d3 * 12^0

theorem base12_div_remainder_9 : 
  let n := base12_to_base10 1 7 4 2 in
  n % 9 = 3 :=
by
  let n := base12_to_base10 1 7 4 2
  have : n = 2786 := rfl
  have : 2786 % 9 = 3 := by decide
  exact this

end base12_div_remainder_9_l530_530994


namespace length_of_first_square_flag_l530_530607

theorem length_of_first_square_flag
  (x : ℝ)
  (h1x : x * 5 + 10 * 7 + 5 * 5 = 15 * 9) : 
  x = 8 :=
by
  sorry

end length_of_first_square_flag_l530_530607


namespace letter_150_is_Z_l530_530503

/-- Definition of the repeating pattern "XYZ" -/
def pattern : List Char := ['X', 'Y', 'Z']

/-- The repeating pattern has a length of 3 -/
def pattern_length : ℕ := 3

/-- Calculate the 150th letter in the repeating pattern "XYZ" -/
def nth_letter_in_pattern (n : ℕ) : Char :=
  let m := n % pattern_length
  if m = 0 then pattern[2] else pattern[m - 1]

/-- Prove that the 150th letter in the pattern "XYZ" is 'Z' -/
theorem letter_150_is_Z : nth_letter_in_pattern 150 = 'Z' :=
by
  sorry

end letter_150_is_Z_l530_530503


namespace dice_same_color_probability_l530_530830

theorem dice_same_color_probability :
  let p := (4 / 20)^2 + (7 / 20)^2 + (8 / 20)^2 + (1 / 20)^2 in p = 13 / 40 :=
by
  let p := (4/20)^(2 : ℕ) + (7/20)^(2 : ℕ) + (8/20)^(2 : ℕ) + (1/20)^(2 : ℕ)
  have h : p = 13 / 40 := sorry
  exact h

end dice_same_color_probability_l530_530830


namespace statement_E_incorrect_l530_530837

-- Defining the necessary conditions
variables (b x y : ℝ)
variable (h₁ : b > 0)
variable (h₂ : b ≠ 1)
variable (h₃ : y = Real.logb b x)

-- Stating the incorrectness of statement (E)
theorem statement_E_incorrect (h₄ : x = -1) : ¬ Real.logb b (-1) ∈ ℝ := 
sorry

end statement_E_incorrect_l530_530837


namespace general_term_l530_530305

noncomputable def sequence (a : Nat → ℝ) : Prop :=
  a 1 = 6 ∧ ∀ n : Nat, n ≥ 1 → a (n + 1) / a n = (n + 3) / n

theorem general_term (a : Nat → ℝ) (h : sequence a) : 
  ∀ n, a n = n * (n + 1) * (n + 2) := 
by sorry

end general_term_l530_530305


namespace triangle_acute_angle_exists_l530_530226

theorem triangle_acute_angle_exists (a b c d e : ℝ)
  (h_abc : a + b > c ∧ a + c > b ∧ b + c > a)
  (h_abd : a + b > d ∧ a + d > b ∧ b + d > a)
  (h_abe : a + b > e ∧ a + e > b ∧ b + e > a)
  (h_acd : a + c > d ∧ a + d > c ∧ c + d > a)
  (h_ace : a + c > e ∧ a + e > c ∧ c + e > a)
  (h_ade : a + d > e ∧ a + e > d ∧ d + e > a)
  (h_bcd : b + c > d ∧ b + d > c ∧ c + d > b)
  (h_bce : b + c > e ∧ b + e > c ∧ c + e > b)
  (h_bde : b + d > e ∧ b + e > d ∧ d + e > b)
  (h_cde : c + d > e ∧ c + e > d ∧ d + e > c) :
  ∃ x y z, (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧
           (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧
           (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧
           x ≠ y ∧ y ≠ z ∧ z ≠ x ∧
           x + y > z ∧ x + z > y ∧ y + z > x ∧
           (x * x + y * y > z * z ∧ y * y + z * z > x * x ∧ z * z + x * x > y * y) := 
sorry

end triangle_acute_angle_exists_l530_530226


namespace letter_150th_in_pattern_l530_530496

def repeating_sequence := "XYZ"

def letter_at_position (n : ℕ) : char :=
  let seq := repeating_sequence.to_list
  seq.get! ((n - 1) % seq.length)

theorem letter_150th_in_pattern : letter_at_position 150 = 'Z' :=
by sorry

end letter_150th_in_pattern_l530_530496


namespace map_distance_l530_530957

noncomputable def map_scale_distance (actual_distance_km : ℕ) (scale : ℕ) : ℕ :=
  let actual_distance_cm := actual_distance_km * 100000;  -- conversion from kilometers to centimeters
  actual_distance_cm / scale

theorem map_distance (d_km : ℕ) (scale : ℕ) (h1 : d_km = 500) (h2 : scale = 8000000) :
  map_scale_distance d_km scale = 625 :=
by
  rw [h1, h2]
  dsimp [map_scale_distance]
  norm_num
  sorry

end map_distance_l530_530957


namespace convex_polygon_diagonals_l530_530783

theorem convex_polygon_diagonals (n : ℕ) (h : n = 30) : 
  let total_pairs := (n * (n - 1)) / 2 in (total_pairs - n) = 405 := 
by
  simp only [total_pairs]
  sorry

end convex_polygon_diagonals_l530_530783


namespace book_pricing_and_min_cost_l530_530551

-- Define the conditions
def price_relation (a : ℝ) (ps_price : ℝ) : Prop :=
  ps_price = 1.2 * a

def book_count_relation (a : ℝ) (lit_count ps_count : ℕ) : Prop :=
  lit_count = 1200 / a ∧ ps_count = 1200 / (1.2 * a) ∧ lit_count - ps_count = 10

def min_cost_condition (x : ℕ) : Prop :=
  x ≤ 600

def total_cost (x : ℕ) : ℝ :=
  20 * x + 24 * (1000 - x)

-- The theorem combining all parts
theorem book_pricing_and_min_cost:
  ∃ (a : ℝ) (ps_price : ℝ) (lit_count ps_count : ℕ),
    price_relation a ps_price ∧
    book_count_relation a lit_count ps_count ∧
    a = 20 ∧ ps_price = 24 ∧
    (∀ (x : ℕ), min_cost_condition x → total_cost x ≥ 21600) ∧
    (total_cost 600 = 21600) :=
by
  sorry

end book_pricing_and_min_cost_l530_530551


namespace count_valid_four_digit_sequences_l530_530202

open Nat

-- Define a function to check if the list of digits forms an arithmetic sequence
def is_arithmetic_sequence : List ℕ → ℕ → Prop
  | [], _ => true
  | [x], _ => true
  | x::y::ys, d => if y - x = d then is_arithmetic_sequence (y::ys) d else false

-- Define the main condition for the arithmetic sequence of four digits in increasing order
def is_valid_four_digit_sequence (a d : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 6 ∧ (d : ℕ) ≤ (9 - a) / 3 ∧ is_arithmetic_sequence [a, a+d, a+2*d, a+3*d] d

-- Define a function to generate the valid sequences
def generate_valid_sequences : List (ℕ × ℕ) :=
  (List.range' 1 6).bind (λ a => (List.range' 1 (min 3 ((9 - a + 2) / 3))).map (λ d => (a, d)))

-- Filter the sequences that meet the criteria
def valid_sequences : List (ℕ × ℕ) :=
  generate_valid_sequences.filter (λ ⟨a, d⟩ => is_valid_four_digit_sequence a d)

-- Define the theorem to count the number of valid sequences
theorem count_valid_four_digit_sequences : valid_sequences.length = 9 :=
  sorry

end count_valid_four_digit_sequences_l530_530202


namespace points_cocyclic_l530_530333

open EuclideanGeometry

-- Define the elements and conditions
variables {A B C P Q R S : Point}

-- The triangle is acute-angled
variable (h_tri : acute_triangle A B C)

-- Perpendicular conditions
variable (h1 : ∃ P Q, (⊥ AC B) ∧ (circle (AC-diameter) B ⊜ {P, Q}))
variable (h2 : ∃ R S, (⊥ AB C) ∧ (circle (AB-diameter) C ⊜ {R, S}))

-- Proof to show cocyclic property
theorem points_cocyclic
  (A B C P Q R S : Point)
  (h_tri : acute_triangle A B C)
  (h1 : ∃ P Q, (⊥ AC B) ∧ (circle (AC-diameter) B ⊜ {P, Q}))
  (h2 : ∃ R S, (⊥ AB C) ∧ (circle (AB-diameter) C ⊜ {R, S})) :
  cocyclic {P, Q, R, S} :=
by sorry

end points_cocyclic_l530_530333


namespace significant_improvement_l530_530549

def indicator_data_old : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def indicator_data_new : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def sample_mean (data : List ℝ) : ℝ := (data.sum / data.length.toReal)

def sample_variance (data : List ℝ) : ℝ := 
  let mean := sample_mean data
  (data.map (λ x => (x - mean) ^ 2)).sum / data.length.toReal

-- Given means and variances directly
def x̄ := 10
def ȳ := 10.3
def s1_sq := 0.036
def s2_sq := 0.04

theorem significant_improvement :
  10.3 - 10 ≥ 2 * Real.sqrt((0.036 + 0.04) / 10) :=
by sorry

end significant_improvement_l530_530549


namespace convex_polygon_diagonals_l530_530785

theorem convex_polygon_diagonals (n : ℕ) (h : n = 30) : 
  let total_pairs := (n * (n - 1)) / 2 in (total_pairs - n) = 405 := 
by
  simp only [total_pairs]
  sorry

end convex_polygon_diagonals_l530_530785


namespace value_of_x_squared_minus_y_squared_l530_530272

theorem value_of_x_squared_minus_y_squared (x y : ℝ) 
  (h₁ : x + y = 20) 
  (h₂ : x - y = 6) :
  x^2 - y^2 = 120 := 
by 
  sorry

end value_of_x_squared_minus_y_squared_l530_530272


namespace door_obstruction_l530_530419

def obstructed_minutes_per_day : Nat := 522

theorem door_obstruction
    (minute_hand_obstructs : ∀ (h m : ℕ), m ∈ {9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21} → door_obstructed h m)
    (hour_hand_obstructs : ∀ (h : ℕ) (m : ℕ), h ∈ {1, 2, 3, 4} ∨ (h = 1 ∧ m ∈ {48..59}) ∨ (h = 4 ∧ m ∈ {0..12}) → door_obstructed h m)
  : minutes_door_obstructed_per_day = obstructed_minutes_per_day :=
by 
  sorry

end door_obstruction_l530_530419


namespace part1_part2_l530_530734

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (x - 3)^2

theorem part1 (a b : ℝ) (h_tangent : ∀ x y, x + y + b = 0) :
  a = 3 ∧ b = -5 := sorry

theorem part2 (a : ℝ) (h_critical : ∃ x1 x2, x1 ≠ x2 ∧ f' a x1 = 0 ∧ f' a x2 = 0) :
  0 < a ∧ a < 9 / 2 := sorry

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ :=
  (2 * x^2 - 6 * x + a) / x

end part1_part2_l530_530734


namespace proof_problem_l530_530726

noncomputable def f (x : ℝ) := 2 * Real.sin (π * x / 6 + π / 3)

theorem proof_problem :
  (∃ A B : ℝ × ℝ, A = (1, 2) ∧ B = (5, -1) ∧
    let OA := (A.1, A.2) in
    let OB := (B.1, B.2) in
    OA.1 * OB.1 + OA.2 * OB.2 = 3) ∧
  (∃ α β : ℝ, Real.tan α = 2 ∧ Real.tan β = -1 / 5 ∧ Real.tan (α - 2 * β) = 29 / 2) :=
sorry

end proof_problem_l530_530726


namespace perimeter_of_triangle_l530_530741

noncomputable def triangle_perimeter (A B C : ℝ) (r R : ℝ) (α : ℝ) : ℝ :=
  if valid (r, R, α) then (find_perimeter r R α) else 0

theorem perimeter_of_triangle
  (r R : ℝ)
  (α : ℝ)
  (hα : α = Real.arccos (2 / 3))
  (h_rR : r * R = 20)
  : triangle_perimeter ABC r R α = 10 * Real.sqrt 5 := sorry

end perimeter_of_triangle_l530_530741


namespace div_iff_div_l530_530898

theorem div_iff_div {a b : ℤ} : (29 ∣ (3 * a + 2 * b)) ↔ (29 ∣ (11 * a + 17 * b)) := 
by sorry

end div_iff_div_l530_530898


namespace angle_EDH_eq_FDH_l530_530876

variables {A B C D H E F : Type*}
variables [IsAcuteTriangle A B C] [AltitudeFromTo A C B] [OnLine H A D] [Intersection BH AC E] [Intersection CH AB F]

theorem angle_EDH_eq_FDH :
  ∀ (A B C D H E F : Point) (altitude_AD : Altitude A B C) (on_AD : OnLine H A D) (intersect_BH_AC : Intersection BH AC E)
  (intersect_CH_AB : Intersection CH AB F),
  ∠ E D H = ∠ F D H := sorry

end angle_EDH_eq_FDH_l530_530876


namespace area_of_triangle_l530_530867

theorem area_of_triangle 
  (a b : ℝ) 
  (C : ℝ) 
  (h_a : a = 1) 
  (h_b : b = sqrt 3) 
  (h_C : C = 30 * real.pi / 180) : 
  1 / 2 * a * b * real.sin C = sqrt 3 / 4 :=
by 
  simp [h_a, h_b, h_C, real.sin_pi_div_two]
  sorry

end area_of_triangle_l530_530867


namespace convex_polygon_diagonals_l530_530815

theorem convex_polygon_diagonals (n : ℕ) (h : n = 30) : 
  let V := n in
  let total_pairs := V * (V - 1) / 2 in
  let adjacent_pairs := n in
  let diagonals := total_pairs - adjacent_pairs in
  diagonals = 405 :=
  by
  -- number of sides is 30
  have hn : n = 30 := h
  -- calculate the total pairs
  have total_pairs_calc : total_pairs = 30 * 29 / 2 := by sorry
  -- calculate the adjacent pairs
  have adjacent_pairs_calc : adjacent_pairs = 30 := by sorry
  -- calculate the diagonals
  have diagonals_calc : diagonals = (30 * 29 / 2) - 30 := by sorry
  -- proved statement
  show 405 = 405, by rfl

end convex_polygon_diagonals_l530_530815


namespace wendy_ran_miles_l530_530978

theorem wendy_ran_miles :
  ∀ (walked ran : ℝ), walked = 9.17 → ran = walked + 10.67 → ran = 19.84 :=
by
  intros walked ran hwalked hran
  rw [hwalked] at hran
  rw [add_comm] at hran
  exact hran

end wendy_ran_miles_l530_530978


namespace vector_b_norm_range_l530_530261

variable (a b : ℝ × ℝ)
variable (norm_a : ‖a‖ = 1)
variable (norm_sum : ‖a + b‖ = 2)

theorem vector_b_norm_range : 1 ≤ ‖b‖ ∧ ‖b‖ ≤ 3 :=
sorry

end vector_b_norm_range_l530_530261


namespace ali_seashells_final_count_l530_530589

theorem ali_seashells_final_count :
  385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25) 
  - (2 / 3) * (385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)) 
  - 0.10 * ((385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)) 
  - (2 / 3) * (385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25))) 
  - (1 / 4) * ((385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)) 
  - (2 / 3) * (385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)) 
  - 0.10 * ((385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)) 
  - (2 / 3) * (385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)))) = 82.485 :=
sorry

end ali_seashells_final_count_l530_530589


namespace golden_section_AC_length_l530_530280

-- Definitions based on the conditions
def golden_section_point (A B C : ℝ) := C / (B - C) = (B - C) / (A - B)
def length_AB : ℝ := 8
def longer_segment_fraction := (Real.sqrt 5 - 1) / 2

-- Statement of the theorem to be proved
theorem golden_section_AC_length : 
  ∃ C : ℝ, 
    golden_section_point 8 0 C ∧ 
    C = 8 * ((Real.sqrt 5 - 1) / 2) := by
  sorry

end golden_section_AC_length_l530_530280


namespace find_sum_of_vars_l530_530254

-- Definitions of the quadratic polynomials
def quadratic1 (x : ℝ) : ℝ := 2 * x^2 + 8 * x + 11
def quadratic2 (y : ℝ) : ℝ := y^2 - 10 * y + 29
def quadratic3 (z : ℝ) : ℝ := 3 * z^2 - 18 * z + 32

-- Theorem statement
theorem find_sum_of_vars (x y z : ℝ) :
  quadratic1 x * quadratic2 y * quadratic3 z ≤ 60 → x + y - z = 0 :=
by 
-- here we would complete the proof steps
sorry

end find_sum_of_vars_l530_530254


namespace group_weight_problem_l530_530388

theorem group_weight_problem (n : ℕ) (avg_weight_increase : ℕ) (weight_diff : ℕ) (total_weight_increase : ℕ) 
  (h1 : avg_weight_increase = 3) (h2 : weight_diff = 75 - 45) (h3 : total_weight_increase = avg_weight_increase * n)
  (h4 : total_weight_increase = weight_diff) : n = 10 := by
  sorry

end group_weight_problem_l530_530388


namespace projection_vector_coordinates_l530_530231

variable (a : ℝ × ℝ × ℝ) (b : ℝ × ℝ × ℝ)

theorem projection_vector_coordinates :
  a = (1, 0, 1) →
  b = (2, 2, 1) →
  let proj := ((a.1 * b.1 + a.2 * b.2 + a.3 * b.3) / (b.1 * b.1 + b.2 * b.2 + b.3 * b.3)) • b in
  proj = (2/3, 2/3, 1/3) :=
by 
  intros ha hb
  rw [ha, hb]
  have h_dot : (1 * 2 + 0 * 2 + 1 * 1 : ℝ) = 3 := by norm_num
  have h_mag_sq : (2 * 2 + 2 * 2 + 1 * 1 : ℝ) = 9 := by norm_num
  have h_scalar : (3 : ℝ) / 9 = 1 / 3 := by norm_num
  have proj_eq : ((1 / 3 : ℝ) • (2, 2, 1)) = (2 / 3, 2 / 3, 1 / 3) := by norm_num
  exact proj_eq

end projection_vector_coordinates_l530_530231


namespace symmetric_point_of_A_l530_530390

theorem symmetric_point_of_A (a b : ℝ) 
  (h1 : 2 * a - 4 * b + 9 = 0) 
  (h2 : ∃ t : ℝ, (a, b) = (1 - 4 * t, 4 + 2 * t)) : 
  (a, b) = (1, 4) :=
sorry

end symmetric_point_of_A_l530_530390


namespace sequence_sum_l530_530020

noncomputable def a (n : ℕ) : ℝ := n * Real.cos (n * Real.pi / 2)

noncomputable def S : ℕ → ℝ
| 0     => 0
| (n+1) => S n + a (n+1)

theorem sequence_sum : S 2017 = 1008 :=
by
  sorry

end sequence_sum_l530_530020


namespace probability_heads_given_heads_l530_530425

-- Definitions based on the conditions
inductive Coin
| coin1 | coin2 | coin3

open Coin

def P (event : Set Coin) : ℝ := 
  if event = {coin1} then 1/3
  else if event = {coin2} then 1/3
  else if event = {coin3} then 1/3
  else 0

def event_heads (coin : Coin) : bool :=
  match coin with
  | coin1 => true
  | coin2 => true
  | coin3 => false

-- Probability of showing heads
def P_heads_given_coin (coin : Coin) : ℝ :=
  if event_heads coin then 1 else 0

-- Total probability of getting heads
def P_heads : ℝ := 
  P {coin1} * (if event_heads coin1 then 1 else 0) +
  P {coin2} * (if event_heads coin2 then 1 else 0) +
  P {coin3} * (if event_heads coin3 then 0 else 0)

-- Using Bayes' Theorem to find probability that the coin showing heads is coin2
def P_coin2_given_heads : ℝ :=
  (P_heads_given_coin coin2 * P {coin2}) / P_heads

-- Defining the theorem statement
theorem probability_heads_given_heads : P_coin2_given_heads = 2 / 3 := by
  sorry

end probability_heads_given_heads_l530_530425


namespace fractional_part_implication_l530_530339

-- Definition of the fractional part function
def fractional_part (x : ℝ) : ℝ := x - ⌊x⌋

theorem fractional_part_implication (a : ℝ) (n : ℕ) (h0 : a ≠ 0)
    (h1 : fractional_part a + fractional_part (1 / a) = 1) :
    fractional_part (a ^ n) + fractional_part (1 / (a ^ n)) = 1 :=
sorry

end fractional_part_implication_l530_530339


namespace workshop_processing_equation_l530_530565

noncomputable def process_equation (x : ℝ) : Prop :=
  (4000 / x - 4200 / (1.5 * x) = 3)

theorem workshop_processing_equation (x : ℝ) (hx : x > 0) :
  process_equation x :=
by
  sorry

end workshop_processing_equation_l530_530565


namespace set_inter_complement_l530_530326

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem set_inter_complement :
  U = {1, 2, 3, 4, 5, 6, 7} ∧ A = {1, 2, 3, 4} ∧ B = {3, 5, 6} →
  A ∩ (U \ B) = {1, 2, 4} :=
by
  sorry

end set_inter_complement_l530_530326


namespace total_weight_of_remaining_chocolate_eggs_l530_530353

-- Define the conditions as required:
def marie_makes_12_chocolate_eggs : ℕ := 12
def weight_of_each_egg : ℕ := 10
def number_of_boxes : ℕ := 4
def boxes_discarded : ℕ := 1

-- Define the proof statement
theorem total_weight_of_remaining_chocolate_eggs :
  let eggs_per_box := marie_makes_12_chocolate_eggs / number_of_boxes,
      remaining_boxes := number_of_boxes - boxes_discarded,
      remaining_eggs := remaining_boxes * eggs_per_box,
      total_weight := remaining_eggs * weight_of_each_egg
  in total_weight = 90 := by
  sorry

end total_weight_of_remaining_chocolate_eggs_l530_530353


namespace find_150th_letter_in_pattern_l530_530447

/--
Given the repeating pattern "XYZ" with a cycle length of 3,
prove that the 150th letter in this pattern is 'Z'.
-/
theorem find_150th_letter_in_pattern : 
  let pattern := "XYZ"
  let cycle_length := String.length pattern
in (150 % cycle_length = 0) → "Z" := 
sorry

end find_150th_letter_in_pattern_l530_530447


namespace letter_at_position_150_l530_530481

theorem letter_at_position_150 : 
  (∀ n, n > 0 → ∃ i, i ∈ {1, 2, 3} ∧ "XYZ".to_list[i-1] = "XYZ".to_list[(n - 1) % 3]) →
  ("XYZ".to_list[(150 - 1) % 3] = 'Z') :=
by
  sorry

end letter_at_position_150_l530_530481


namespace quadratic_transformation_l530_530406

theorem quadratic_transformation (p q r : ℝ) :
  (∀ x : ℝ, p * x^2 + q * x + r = 3 * (x - 5)^2 + 15) →
  (∀ x : ℝ, 4 * p * x^2 + 4 * q * x + 4 * r = 12 * (x - 5)^2 + 60) :=
by
  intro h
  exact sorry

end quadratic_transformation_l530_530406


namespace pages_of_dictionary_l530_530973

theorem pages_of_dictionary (h : ∃ pages, count_digit_occurrences 1 pages = 1988) : pages = 3144 :=
sorry

end pages_of_dictionary_l530_530973


namespace dot_product_AD_BC_l530_530297

-- Definitions for vector operations and point relations
variables (A B C D: Type) [InnerProductSpace ℝ A] 
variables (a b c d : A) -- points corresponding to A, B, C, D 

-- Conditions 
axiom lengthBC : InnerProductSpace.norm (b - c) = 6
axiom D_relation : b - d = 2 • (d - c)

-- Theorem statement
theorem dot_product_AD_BC : inner ((a - d) : A) ((b - c) : A) = 6 := 
sorry

end dot_product_AD_BC_l530_530297


namespace proof_problem_l530_530692

theorem proof_problem 
  (a b : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) : 
  |a / b + b / a| ≥ 2 := 
sorry

end proof_problem_l530_530692


namespace nth_150th_letter_in_XYZ_l530_530484

def pattern : List Char := ['X', 'Y', 'Z']

def nth_letter (n : Nat) : Char :=
  pattern[(n - 1) % pattern.length]

theorem nth_150th_letter_in_XYZ :
  nth_letter 150 = 'Z' :=
by
  sorry

end nth_150th_letter_in_XYZ_l530_530484


namespace negation_of_universal_l530_530400

variable (P : ℝ → Prop)
def pos (x : ℝ) : Prop := x > 0
def gte_zero (x : ℝ) : Prop := x^2 - x ≥ 0
def lt_zero (x : ℝ) : Prop := x^2 - x < 0

theorem negation_of_universal :
  ¬ (∀ x, pos x → gte_zero x) ↔ ∃ x, pos x ∧ lt_zero x := by
  sorry

end negation_of_universal_l530_530400


namespace sin_180_degree_l530_530630

theorem sin_180_degree : Real.sin (Real.pi) = 0 := by sorry

end sin_180_degree_l530_530630


namespace billy_tickets_l530_530603

theorem billy_tickets (ferris_wheel_rides bumper_car_rides rides_per_ride total_tickets : ℕ) 
  (h1 : ferris_wheel_rides = 7)
  (h2 : bumper_car_rides = 3)
  (h3 : rides_per_ride = 5)
  (h4 : total_tickets = (ferris_wheel_rides + bumper_car_rides) * rides_per_ride) :
  total_tickets = 50 := 
by 
  sorry

end billy_tickets_l530_530603


namespace sin_180_degrees_l530_530638

theorem sin_180_degrees : Real.sin (180 * Real.pi / 180) = 0 := 
by
  sorry

end sin_180_degrees_l530_530638


namespace arithmetic_sequence_seventy_fifth_term_l530_530391

theorem arithmetic_sequence_seventy_fifth_term:
  ∀ (a₁ a₂ d : ℕ), a₁ = 3 → a₂ = 51 → a₂ = a₁ + 24 * d → (3 + 74 * d) = 151 := by
  sorry

end arithmetic_sequence_seventy_fifth_term_l530_530391


namespace largest_n_lemma_l530_530190
noncomputable def largest_n (n b: ℕ) : Prop := 
  n! = ((n-4) + b)!/(b!)

theorem largest_n_lemma : ∀ n: ℕ, ∀ b: ℕ, b ≥ 4 → (largest_n n b → n = 1) := 
  by
  intros n b hb h
  sorry

end largest_n_lemma_l530_530190


namespace inequality_solution_set_quadratic_inequality_range_l530_530099

theorem inequality_solution_set (x : ℝ) :
  (9 / (x + 4) ≤ 2) ↔ (x < -4) ∨ (x ≥ 1 / 2) := 
sorry

theorem quadratic_inequality_range (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + k^2 - 1 > 0) ↔ (k > Real.sqrt 2 ∨ k < -Real.sqrt 2) :=
sorry

end inequality_solution_set_quadratic_inequality_range_l530_530099


namespace find_equation_of_line_l530_530222

-- Define the given lines
def l1 (x y : ℝ) : Prop := x + 2 * y - 5 = 0
def l2 (x y : ℝ) : Prop := x + 2 * y - 3 = 0
def l3 (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the point P
def P : ℝ × ℝ := (-1, 1)

-- Define the midpoint condition
def midpoint (M M1 M2 : ℝ × ℝ) : Prop := ∀x y:ℝ, 
    M = (x, y) →
    M1 = (l1.1, l1.2) →
    M2 = (l2.1, l2.2) →
    l3 x y

-- Define the intersection of l with l1 and l2
def intersects (l l1 l2 : ℝ × ℝ → Prop) : Prop := 
  ∃ x y : ℝ, l1 x y ∧ l x y ∧ ∃ x' y' : ℝ, l2 x' y' ∧ l x' y' 

-- Final statement
theorem find_equation_of_line (l : ℝ × ℝ → Prop) :
  (∃ x y : ℝ, l x y) ∧
  (∃ x y : ℝ, midpoint (2, 1) (x, y) (-1, 1)) →
  (l = (fun (p : ℝ × ℝ) => p.2 = 1)) :=
by
  sorry

end find_equation_of_line_l530_530222


namespace diagonal_count_of_convex_polygon_30_sides_l530_530794
-- Importing the entire Mathlib library for necessary mathematical constructs

theorem diagonal_count_of_convex_polygon_30_sides : 
  let n := 30 in
  let num_diagonals := (n * (n - 3)) / 2 in
  num_diagonals = 405 := 
by {
  let n := 30
  let num_diagonals := (n * (n - 3)) / 2
  show num_diagonals = 405,
  sorry
}

end diagonal_count_of_convex_polygon_30_sides_l530_530794


namespace option_B_is_not_polynomial_l530_530998

-- Define what constitutes a polynomial
def is_polynomial (expr : String) : Prop :=
  match expr with
  | "-26m" => True
  | "3m+5n" => True
  | "0" => True
  | _ => False

-- Given expressions
def expr_A := "-26m"
def expr_B := "m-n=1"
def expr_C := "3m+5n"
def expr_D := "0"

-- The Lean statement confirming option B is not a polynomial
theorem option_B_is_not_polynomial : ¬is_polynomial expr_B :=
by
  -- Since this statement requires a proof, we use 'sorry' as a placeholder.
  sorry

end option_B_is_not_polynomial_l530_530998


namespace typists_problem_l530_530420

variable (t_a t_b t_c : ℝ) -- Define variables for the completion times of A, B, and C
variable (eff_a eff_b eff_c : ℝ) -- Define variables for efficiencies of A, B, and C
variable (total_time : ℝ) -- Define a variable for the total time

-- Conditions based on given problem
def condition1 : Prop := t_a = t_b + 5
def condition2 : Prop := 4 * eff_a = 3 * eff_b
def condition3 : Prop := eff_c = 2 * eff_a

-- Work efficiencies definitions
def efficiency_A : Prop := eff_a = 1 / t_a
def efficiency_B : Prop := eff_b = 1 / (t_a - 5)
def efficiency_C : Prop := eff_c = 1 / 10 -- from C being twice as efficient as A

-- Formalize the question
def total_time_proof : Prop :=
  condition1 ∧ condition2 ∧ condition3 ∧
  efficiency_A ∧ efficiency_B ∧ efficiency_C →
  total_time = 14 + 1/6

-- Lean statement for the proof problem requiring solution to be skipped
theorem typists_problem : total_time_proof t_a t_b t_c eff_a eff_b eff_c total_time :=
by
  -- Providing the conditions
  intros h,
  cases h with c1 c_rest,
  cases c_rest with c2 c3,
  cases c3 with eA eBC,
  cases eBC with eB eC,
  
  -- Use the conditions to derive the solution, skipping steps with sorry
  sorry -- The actual proof is skipped

end typists_problem_l530_530420


namespace deductive_reasoning_chain_l530_530404

-- Define the conditions as hypotheses
variables 
  (names_not_correct : Prop)
  (language_not_in_accordance_with_truth : Prop)
  (things_not_accomplished : Prop)
  (rituals_and_music_not_flourished : Prop)
  (punishments_not_properly_imposed : Prop)
  (people_do_not_know_how_to_move : Prop)

-- The sequence of implications
hypothesis h1 : names_not_correct → language_not_in_accordance_with_truth
hypothesis h2 : language_not_in_accordance_with_truth → things_not_accomplished
hypothesis h3 : things_not_accomplished → rituals_and_music_not_flourished
hypothesis h4 : rituals_and_music_not_flourished → punishments_not_properly_imposed
hypothesis h5 : punishments_not_properly_imposed → people_do_not_know_how_to_move

-- The conclusion we want to prove
theorem deductive_reasoning_chain : names_not_correct → people_do_not_know_how_to_move :=
by {
  intros h,
  apply h5,
  apply h4,
  apply h3,
  apply h2,
  apply h1,
  assumption,
}

end deductive_reasoning_chain_l530_530404


namespace car_speed_calculation_l530_530545

noncomputable def car_speed_next_160_km
  (d1 : ℝ) (s1 : ℝ) (d_total : ℝ) (v_avg : ℝ) : ℝ :=
  let t_total := d_total / v_avg in
  let t1 := d1 / s1 in
  let t2 := t_total - t1 in
  (d_total - d1) / t2

theorem car_speed_calculation
  (d1 d2 s1 d_total : ℝ)
  (h1 : d_total = d1 + d2)
  (v_avg : ℝ)
  (h2 : v_avg = 71.11111111111111)
  (h3 : d1 = 160)
  (h4 : s1 = 64)
  (h5 : d_total = 320)
  : car_speed_next_160_km d1 s1 d_total v_avg = 80 := sorry

end car_speed_calculation_l530_530545


namespace tan_value_l530_530695

theorem tan_value (x : ℝ) (hx : x ∈ Set.Ioo (-π / 2) 0) (hcos : Real.cos x = 4 / 5) : Real.tan x = -3 / 4 :=
sorry

end tan_value_l530_530695


namespace symmetric_complex_division_l530_530716

theorem symmetric_complex_division :
  (∀ (z1 z2 : ℂ), z1 = 3 - (1 : ℂ) * Complex.I ∧ z2 = -(Complex.re z1) + (Complex.im z1) * Complex.I 
   → (z1 / z2) = -4/5 + (3/5) * Complex.I) := sorry

end symmetric_complex_division_l530_530716


namespace circumference_of_cone_base_l530_530566

theorem circumference_of_cone_base (r : ℝ) (sector_angle : ℝ) (A B : ℝ) 
  (h_radius : r = 5) (h_angle : sector_angle = 240) (hAB : A = B) : 
  (2 * real.pi * r * sector_angle / 360 = 20 * real.pi / 3) := 
  by 
    sorry

end circumference_of_cone_base_l530_530566


namespace find_150th_letter_l530_530473

def pattern : List Char := ['X', 'Y', 'Z']

def position (N : ℕ) (pattern_length : ℕ) : ℕ :=
  if N % pattern_length = 0 then pattern_length else N % pattern_length

theorem find_150th_letter : 
  let pattern_length := 3 in (position 150 pattern_length = 3) ∧ (pattern.drop (position 150 pattern_length - 1)).head = 'Z' :=
by
  sorry

end find_150th_letter_l530_530473


namespace exists_N_with_N_and_N2_ending_same_l530_530174

theorem exists_N_with_N_and_N2_ending_same : 
  ∃ (N : ℕ), (N > 0) ∧ (N % 100000 = (N*N) % 100000) ∧ (N / 10000 ≠ 0) := sorry

end exists_N_with_N_and_N2_ending_same_l530_530174


namespace find_area_of_smaller_circle_l530_530433

-- Variables for the radii of the circles
variables (r : ℝ) (R : ℝ)

-- Definitions of the conditions
def is_tangent_externally (c1 c2 : ℝ) : Prop := c1 + r = c2
def is_common_tangent_line (x y : ℝ) : Prop := x = 8 ∧ y = 8
def radius_relation (r R : ℝ) : Prop := R = 3 * r

-- Theorem statement for the problem
theorem find_area_of_smaller_circle (r : ℝ) (R : ℝ) (h1 : is_tangent_externally R r) (h2 : ∀ (x y : ℝ), is_common_tangent_line x y) (h3 : radius_relation r R) : 
  ∀ (A : ℝ), A = π * r ^ 2 → A = 16 * π :=
by
  simp only [is_tangent_externally, is_common_tangent_line, radius_relation] at *,
  sorry

end find_area_of_smaller_circle_l530_530433


namespace diagonals_in_convex_polygon_with_30_sides_l530_530768

theorem diagonals_in_convex_polygon_with_30_sides : 
  let n := 30 in
  ∑ i j in finset.range n, (i + 1 - 1) * (j + 1 - 1) * (nat.choose 30 2 - 30) / 2 = 202 := 
by
  sorry

end diagonals_in_convex_polygon_with_30_sides_l530_530768


namespace odd_positive_integer_minus_twenty_l530_530506

theorem odd_positive_integer_minus_twenty (x : ℕ) (h : x = 53) : (2 * x - 1) - 20 = 85 := by
  subst h
  rfl

end odd_positive_integer_minus_twenty_l530_530506


namespace monotonic_increasing_interval_l530_530729

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem monotonic_increasing_interval :
  MonotonicOn g (Icc (-5 * Real.pi / 12) (-Real.pi / 6)) :=
sorry

end monotonic_increasing_interval_l530_530729


namespace alex_minimum_additional_coins_l530_530588

theorem alex_minimum_additional_coins (friends coins : ℕ) (h_friends : friends = 15) (h_coins : coins = 105) : 
  ∃ add_coins, add_coins = (∑ i in range (friends + 1), i) - coins :=
by
  sorry

end alex_minimum_additional_coins_l530_530588


namespace tan_of_A_sin_of_B_l530_530849

theorem tan_of_A (A : ℝ) (h1 : sin (A + π / 6) = 2 * cos A) : tan A = sqrt 3 := 
sorry

theorem sin_of_B (A B : ℝ) 
  (hA : A = π / 3)
  (hB1 : B ∈ 0..(π / 3))
  (hB2 : sin (A - B) = 3 / 5) : sin B = (4 * sqrt 3 - 3) / 10 := 
sorry

end tan_of_A_sin_of_B_l530_530849


namespace find_150th_letter_l530_530463

theorem find_150th_letter :
  let pattern := ['X', 'Y', 'Z']
  150 % 3 = 0 -> pattern[(150 % 3 + 2) % 3] = 'Z' :=
begin
  intros pattern h,
  simp at *,
  exact rfl,
end

end find_150th_letter_l530_530463


namespace closed_curve_in_hemisphere_l530_530540

open Set

noncomputable def unit_sphere : Set (ℝ^3) :=
  { p | p.1^2 + p.2^2 + p.3^2 = 1 }

noncomputable def is_closed_curve (c : ℝ → ℝ^3) : Prop :=
  c 0 = c 1

noncomputable def length_of_curve (c : ℝ → ℝ^3) (a b : ℝ) : ℝ :=
  ℝ

theorem closed_curve_in_hemisphere {c : ℝ → ℝ^3} (h₁ : is_closed_curve c)
    (h₂ : ∀ t, c t ∈ unit_sphere) (h₃ : length_of_curve c 0 1 < 2 * real.pi) :
    ∃ H : Set (ℝ^3), (unit_sphere ∩ hsphere H) ∧ (∀ t, c t ∈ H) := sorry

end closed_curve_in_hemisphere_l530_530540


namespace proposition_2_proposition_4_l530_530260

-- Definitions based on initial conditions
variables (m n : Line) (α β : Plane)
variable [non_overlapping_m_n : m ≠ n ∧ ¬(m ∩ n).nonempty]
variable [non_overlapping_α_β : α ≠ β ∧ ¬(α ∩ β).nonempty]

-- Proposition ②: n ⊥ α and m ⊥ β and m ∥ n implies α ∥ β
theorem proposition_2 (n_perp_α : n ⊥ α) (m_perp_β : m ⊥ β) (m_par_n : m ∥ n) : α ∥ β := sorry

-- Proposition ④: α ⊥ β and α ∩ β = m and n ⊆ β and n ⊥ m implies n ⊥ α
theorem proposition_4 (α_perp_β : α ⊥ β) (α_cap_β_eq_m : α ∩ β = m) (n_in_β : n ⊆ β) (n_perp_m : n ⊥ m) : n ⊥ α := sorry

end proposition_2_proposition_4_l530_530260


namespace number_of_diagonals_in_convex_polygon_l530_530754

/-- 
A theorem stating that the number of diagonals 
in a convex polygon with 30 sides is equal to 405.
-/
theorem number_of_diagonals_in_convex_polygon (n : ℕ) (h : n = 30) (convex : True) : (n * (n - 3)) / 2 = 405 := 
by 
  rw h
  norm_num
  done

end number_of_diagonals_in_convex_polygon_l530_530754


namespace bookmarks_per_day_l530_530171

theorem bookmarks_per_day (pages_now : ℕ) (pages_end_march : ℕ) (days_in_march : ℕ) (pages_added : ℕ) (pages_per_day : ℕ)
  (h1 : pages_now = 400)
  (h2 : pages_end_march = 1330)
  (h3 : days_in_march = 31)
  (h4 : pages_added = pages_end_march - pages_now)
  (h5 : pages_per_day = pages_added / days_in_march) :
  pages_per_day = 30 := sorry

end bookmarks_per_day_l530_530171


namespace sin_180_degree_l530_530631

theorem sin_180_degree : Real.sin (Real.pi) = 0 := by sorry

end sin_180_degree_l530_530631


namespace largest_n_factorial_product_l530_530187

theorem largest_n_factorial_product :
  ∃ n : ℕ, (∀ a : ℕ, (n > 0) → (n! = (∏ k in finset.range (n - 4 + a), k + 1))) → n = 4 :=
begin
  sorry
end

end largest_n_factorial_product_l530_530187


namespace george_and_henry_second_meeting_l530_530689

def george_and_henry_second_meeting_time : ℝ :=
  5.4

theorem george_and_henry_second_meeting
  (pool_length : ℝ)
  (george_start : ℝ)
  (henry_start : ℝ)
  (meeting_time_initial : ℝ)
  (george_meet_center : bool)
  (henry_meet_center : bool)
  (no_turn_time_lost : bool)
  (george_speed : ℝ)
  (henry_speed : ℝ)
  (total_time : ℝ) :
  george_meet_center = tt →
  henry_meet_center = tt →
  no_turn_time_lost = tt →
  pool_length = 50 →
  george_start = 0 →
  henry_start = 1 →
  meeting_time_initial = 3 →
  george_speed = (pool_length / 2) / meeting_time_initial →
  henry_speed = (pool_length / 2) / (meeting_time_initial - (henry_start - george_start)) →
  total_time = 5.4 :=
by
  intros
  sorry

end george_and_henry_second_meeting_l530_530689


namespace minimum_value_value_of_f_a_l530_530250

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + π / 3)

theorem minimum_value (x : ℝ) (h : x ∈ Icc (-π/6) 0) :
  4 * f(x) + 1/f(x) = 4 ↔ x = -π/12 := by
  sorry

theorem value_of_f_a (a : ℝ) (h₁ : a ∈ Icc (-π/2) 0) (h₂ : f(a/2 + π/3) = √5 / 5) :
  f(a) = (3 * √3 - 4) / 10 := by
  sorry

end minimum_value_value_of_f_a_l530_530250


namespace petya_wins_l530_530045

theorem petya_wins (n : ℕ) : n = 111 → (∀ k : ℕ, 1 ≤ k ∧ k ≤ 9 → ∃ x : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ (n - k - x) % 10 = 0) → wins_optimal_play := sorry

end petya_wins_l530_530045


namespace solve_fraction_equation_l530_530926

theorem solve_fraction_equation (x : ℝ) (h : x ≠ 1) : (3 * x - 1) / (4 * x - 4) = 2 / 3 → x = -5 :=
by
  intro h_eq
  sorry

end solve_fraction_equation_l530_530926


namespace alternating_draws_probability_l530_530109

theorem alternating_draws_probability :
    (box : Box) → 
    (black_balls white_balls : ℕ) → 
    (black_balls = 5) → 
    (white_balls = 4) → 
    (∃ pattern : list Color,
      pattern = [Black, White, Black, White, Black, White, Black, White, Black]) →
    (probability : ℚ) → 
    (probability = 1 / Nat.choose 9 5) → 
    (∑ (s : finset (list Color)), 
      if ∃ pattern, s.val = pattern 
      then 1 else (0 : ℚ)) / (Nat.choose 9 5) = 1 / 126 :=
begin
sorry
end

end alternating_draws_probability_l530_530109


namespace R_depends_on_d_and_n_l530_530889

theorem R_depends_on_d_and_n (a d n : ℕ) : 
  let s1 := (n * (2 * a + (n - 1) * d)) / 2
          s2 := (3 * n * (2 * a + (3 * n - 1) * d)) / 2
          s3 := (5 * n * (2 * a + (5 * n - 1) * d)) / 2
    in 2 * s3 - 3 * s2 + s1 = 5 * n * d :=
by
  sorry

end R_depends_on_d_and_n_l530_530889


namespace xiao_cong_fee_xiao_ming_fee_l530_530546

def water_fee (x : ℕ) : ℕ :=
  if x ≤ 8 then 2 * x
  else if x ≤ 12 then 2 * 8 + 3 * (x - 8)
  else 2 * 8 + 3 * 4 + 5 * (x - 12)

-- Proving the first case for Xiao Cong's household
theorem xiao_cong_fee : water_fee 10 = 22 :=
by
  simp [water_fee]
  sorry

-- Proving the second case for Xiao Ming's household
theorem xiao_ming_fee (a : ℕ) (h : 12 < a) : water_fee a = 5 * a - 32 :=
by
  simp [water_fee]
  sorry

end xiao_cong_fee_xiao_ming_fee_l530_530546


namespace vicente_total_spent_l530_530064

def kilograms_of_rice := 5
def cost_per_kilogram_of_rice := 2
def pounds_of_meat := 3
def cost_per_pound_of_meat := 5

def total_spent := kilograms_of_rice * cost_per_kilogram_of_rice + pounds_of_meat * cost_per_pound_of_meat

theorem vicente_total_spent : total_spent = 25 := 
by
  sorry -- Proof would go here

end vicente_total_spent_l530_530064


namespace find_N_l530_530437

def is_divisible_by (n m : Nat) : Prop := m % n = 0

theorem find_N (N : ℕ) : 
  (N = 5 ∨ N = 10 ∨ N = 11 ∨ N = 55 ∨ N = 110) ∧ 
  ((is_divisible_by 5 N + is_divisible_by 11 N + (N < 10) + is_divisible_by 55 N) = 2) → 
  N = 5 :=
by
  sorry

end find_N_l530_530437


namespace ratio_milk_water_in_first_vessel_l530_530977

noncomputable def volumes_ratio (V : ℝ) := 
  (volume_first : ℝ) (volume_second : ℝ) (ratio_milk_water_first : ℝ) (ratio_milk_water_second : ℝ) : Prop :=
  (volume_first / volume_second = 3 / 5) ∧
  (milk_first : ℝ = V / 3) ∧ (water_first : ℝ = 2 * (V / 3)) ∧
  (milk_second : ℝ = 3 * (5 / 5)) ∧ (water_second : ℝ = 2 * (5 / 5)) ∧
  ((milk_first + milk_second) / (water_first + water_second) = 1)

theorem ratio_milk_water_in_first_vessel :
  ∀ (V : ℝ) (volume_first volume_second ratio_milk_water_first ratio_milk_water_second : ℝ),
  volumes_ratio V volume_first volume_second ratio_milk_water_first ratio_milk_water_second →
  (milk_first : ℝ = V / 3) ∧ (water_first : ℝ = 2 * (V / 3)) :=
sorry

end ratio_milk_water_in_first_vessel_l530_530977


namespace carole_wins_if_and_only_if_n_is_odd_l530_530611

noncomputable def game_winner (n : ℕ) : Prop :=
  if n % 2 = 1 then "Carole" else "Leo"

theorem carole_wins_if_and_only_if_n_is_odd (n : ℕ) (h : n > 10) :
  game_winner(n) = "Carole" ↔ n % 2 = 1 :=
sorry

end carole_wins_if_and_only_if_n_is_odd_l530_530611


namespace alan_has_5_20_cent_coins_l530_530139

theorem alan_has_5_20_cent_coins
  (a b c : ℕ)
  (h1 : a + b + c = 20)
  (h2 : ((400 - 15 * a - 10 * b) / 5) + 1 = 24) :
  c = 5 :=
by
  sorry

end alan_has_5_20_cent_coins_l530_530139


namespace cost_of_8_cubic_yards_topsoil_l530_530056

def cubic_yards_to_cubic_feet (yd³ : ℕ) : ℕ := 27 * yd³

def cost_of_topsoil (cubic_feet : ℕ) (cost_per_cubic_foot : ℕ) : ℕ := cubic_feet * cost_per_cubic_foot

theorem cost_of_8_cubic_yards_topsoil :
  cost_of_topsoil (cubic_yards_to_cubic_feet 8) 8 = 1728 :=
by
  sorry

end cost_of_8_cubic_yards_topsoil_l530_530056


namespace rhombus_shorter_diagonal_l530_530947

theorem rhombus_shorter_diagonal (d1 d2 : ℝ) (area : ℝ) (h1 : d2 = 20) (h2 : area = 120) (h3 : area = (d1 * d2) / 2) : d1 = 12 :=
by 
  sorry

end rhombus_shorter_diagonal_l530_530947


namespace inclination_angle_of_line_l530_530840

noncomputable def slope_of_line (x1 y1 x2 y2 : ℝ) : ℝ :=
  (y2 - y1) / (x2 - x1)

theorem inclination_angle_of_line {α : ℝ} :
  let A := (1 : ℝ, 0 : ℝ)
  let B := (2 : ℝ, Real.sqrt 3)
  let k := slope_of_line A.1 A.2 B.1 B.2
  α ∈ Set.Ico 0 Real.pi ∧ α ≠ Real.pi / 2 ∧ Real.tan α = k →
  α = Real.pi / 3 :=
by
  intros
  sorry

end inclination_angle_of_line_l530_530840


namespace find_150th_letter_l530_530456
open Nat

def repeating_sequence := "XYZ"

def length_repeating_sequence := 3

theorem find_150th_letter : (150 % length_repeating_sequence == 0) → repeating_sequence[(length_repeating_sequence - 1) % length_repeating_sequence] = 'Z' := 
by
  sorry

end find_150th_letter_l530_530456


namespace find_AC_l530_530277

def φ : ℝ := (Real.sqrt 5 + 1) / 2

def golden_section (AB AC : ℝ) : Prop :=
  (AB / AC = AC / (AB - AC))

theorem find_AC (AB : ℝ) (AC : ℝ) (h1 : AB = 8) (h2 : AC > (AB - AC)) (h3 : golden_section AB AC) :
  AC = 4 * (Real.sqrt 5 - 1) :=
sorry

end find_AC_l530_530277


namespace dress_designs_count_l530_530555

theorem dress_designs_count :
  let colors := 5
  let patterns := 5
  let materials := 2
  (colors * patterns * materials) = 50 :=
by
  let colors := 5
  let patterns := 5
  let materials := 2
  show colors * patterns * materials = 50
  calc
    colors * patterns * materials
        = 5 * 5 * 2   : by rfl
    ... = 25 * 2      : by rfl
    ... = 50          : by rfl

end dress_designs_count_l530_530555


namespace diagonals_in_30_sided_polygon_l530_530823

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem diagonals_in_30_sided_polygon :
  number_of_diagonals 30 = 405 :=
by sorry

end diagonals_in_30_sided_polygon_l530_530823


namespace rectangle_tiling_l530_530373

theorem rectangle_tiling (n m : ℕ) (h_n : n > 1) : 
  ∃ top_layer bottom_layer : list (ℕ × ℕ), 
  (∀ tile ∈ top_layer, tile.fst = 1 ∧ tile.snd = 2) ∧ 
  (∀ tile ∈ bottom_layer, tile.fst = 1 ∧ tile.snd = 2) ∧ 
  (∀ tile1 ∈ top_layer, ∀ tile2 ∈ bottom_layer, tile1 ≠ tile2) :=
sorry

end rectangle_tiling_l530_530373


namespace ring_area_and_circumference_l530_530434

noncomputable def area_of_ring (r1 r2 : ℝ) : ℝ :=
  π * (r1 ^ 2 - r2 ^ 2)

noncomputable def circumference_of_circle (r : ℝ) : ℝ :=
  2 * π * r

theorem ring_area_and_circumference :
  area_of_ring 12 7 = 95 * π ∧ circumference_of_circle 12 = 24 * π :=
by
  sorry

end ring_area_and_circumference_l530_530434


namespace logical_relationship_l530_530720

-- Definitions of the given conditions in the problem
def circle_C (x y r : ℝ) : Prop := (x - 1) ^ 2 + y ^ 2 = r ^ 2
def line (x y : ℝ) : Prop := x - sqrt 3 * y + 3 = 0
def p (r : ℝ) : Prop := 0 < r ∧ r < 3
def q (x y r : ℝ) : Prop := circle_C x y r ∧ ∃ x1 y1 x2 y2, 
  (x1, y1), (x2, y2) ∈ C ∧ (dist (x1, y1) line = 1) ∧ (dist (x2, y2) line = 1) ∧ (x1, y1) ≠ (x2, y2)

-- Proving the logical relationship between conditions q and p
theorem logical_relationship (r : ℝ) : p r ↔ q r :=
by
  sorry

end logical_relationship_l530_530720


namespace ratio_PR_to_ST_l530_530908

noncomputable def distance {A B : Type} [Add A] (a b : A) : A := a + b

theorem ratio_PR_to_ST
  (P Q R S T : ℝ)
  (PQ QR RS ST PT : ℝ)
  (h1 : PQ = 3)
  (h2 : QR = 6)
  (h3 : RS = 4)
  (h4 : ST = 10)
  (h5 : PT = 30) :
  (distance PQ QR / ST) = 9 / 10 :=
by 
  rw [←h1, ←h2, ←h4]
  have hPR : PQ + QR = 9 := by linarith
  exact hPR.symm ▸ sorry

end ratio_PR_to_ST_l530_530908


namespace sin_180_degree_l530_530632

theorem sin_180_degree : Real.sin (Real.pi) = 0 := by sorry

end sin_180_degree_l530_530632


namespace largest_median_of_given_numbers_l530_530711

theorem largest_median_of_given_numbers
  (L : List ℕ)
  (hL1 : L.length = 10)
  (hL2 : Multiset.Of (L.toMultiset) = {6, 7, 2, 4, 8, 5} ∪ {x | x > 8 ∧ x < 13}):
  ∃ m, m = 7.5 ∧ m = ((L.nth_le 4 sorry) + (L.nth_le 5 sorry)) / 2 := 
sorry

end largest_median_of_given_numbers_l530_530711


namespace total_weight_of_remaining_eggs_is_correct_l530_530356

-- Define the initial conditions and the question as Lean definitions
def total_eggs : Nat := 12
def weight_per_egg : Nat := 10
def num_boxes : Nat := 4
def melted_boxes : Nat := 1

-- Calculate the total weight of the eggs
def total_weight : Nat := total_eggs * weight_per_egg

-- Calculate the number of eggs per box
def eggs_per_box : Nat := total_eggs / num_boxes

-- Calculate the weight per box
def weight_per_box : Nat := eggs_per_box * weight_per_egg

-- Calculate the number of remaining boxes after one is tossed out
def remaining_boxes : Nat := num_boxes - melted_boxes

-- Calculate the total weight of the remaining chocolate eggs
def remaining_weight : Nat := remaining_boxes * weight_per_box

-- The proof task
theorem total_weight_of_remaining_eggs_is_correct : remaining_weight = 90 := by
  sorry

end total_weight_of_remaining_eggs_is_correct_l530_530356


namespace cubic_inequality_l530_530891

theorem cubic_inequality (a b c : ℝ) (P : ℝ → ℝ) (hP : P = λ x, x^3 + a * x^2 + b * x + c)
  (hroots : ∃ α β γ : ℝ, α ≠ 0 ∧ β ≠ 0 ∧ γ ≠ 0 ∧ (P α = 0 ∧ P β = 0 ∧ P γ = 0)) :
  6 * a^3 + 10 * (a^2 - 2 * b)^(3 / 2) - 12 * a * b ≥ 27 * c :=
sorry

end cubic_inequality_l530_530891


namespace cars_between_15000_and_20000_l530_530148

theorem cars_between_15000_and_20000 
  (total_cars : ℕ)
  (less_than_15000_ratio : ℝ)
  (more_than_20000_ratio : ℝ)
  : less_than_15000_ratio = 0.15 → 
    more_than_20000_ratio = 0.40 → 
    total_cars = 3000 → 
    ∃ (cars_between : ℕ),
      cars_between = total_cars - (less_than_15000_ratio * total_cars + more_than_20000_ratio * total_cars) ∧ 
      cars_between = 1350 :=
by
  sorry

end cars_between_15000_and_20000_l530_530148


namespace total_cost_of_topsoil_l530_530053

-- Definitions
def cost_per_cubic_foot : ℝ := 8
def cubic_yard_to_cubic_foot : ℝ := 27
def volume_in_cubic_yards : ℕ := 8

-- The total cost of 8 cubic yards of topsoil
theorem total_cost_of_topsoil : volume_in_cubic_yards * cubic_yard_to_cubic_foot * cost_per_cubic_foot = 1728 := by
  sorry

end total_cost_of_topsoil_l530_530053


namespace smallest_common_multiple_8_6_l530_530512

theorem smallest_common_multiple_8_6 : ∃ n : ℕ, n > 0 ∧ (8 ∣ n) ∧ (6 ∣ n) ∧ ∀ m : ℕ, m > 0 ∧ (8 ∣ m) ∧ (6 ∣ m) → n ≤ m :=
begin
  use 24,
  split,
  { exact zero_lt_24, },
  split,
  { exact dvd.intro 3 rfl, },
  split,
  { exact dvd.intro 4 rfl, },
  intros m hm h8 h6,
  -- actual proof here
  sorry
end

end smallest_common_multiple_8_6_l530_530512


namespace inequality_transitive_l530_530999

theorem inequality_transitive (a b c : ℝ) : a * c^2 > b * c^2 → a > b :=
sorry

end inequality_transitive_l530_530999


namespace smallest_a_l530_530930

theorem smallest_a (a b c : ℚ)
  (h1 : a > 0)
  (h2 : b = -2 * a / 3)
  (h3 : c = a / 9 - 5 / 9)
  (h4 : (a + b + c).den = 1) : a = 5 / 4 :=
by
  sorry

end smallest_a_l530_530930


namespace ratio_second_to_first_l530_530046

-- Definitions based on given conditions
def M1 : ℕ := 3  -- First cat's meows per minute
axiom M2 : ℕ     -- Second cat's meows per minute (to be determined)
def M3 : ℕ := M2 / 3  -- Third cat's meows per minute

-- Total meows in 5 minutes
axiom total_meows : 5 * (M1 + M2 + M3) = 55

-- The theorem stating that the ratio of the second cat's meows to the first cat's meows is 2:1
theorem ratio_second_to_first : M2 / M1 = 2 := by
  sorry

end ratio_second_to_first_l530_530046


namespace smallest_common_multiple_l530_530509

theorem smallest_common_multiple (n : ℕ) (h1 : n > 0) (h2 : 8 ∣ n) (h3 : 6 ∣ n) : n = 24 :=
by sorry

end smallest_common_multiple_l530_530509


namespace loss_denotation_l530_530275

theorem loss_denotation : 
  (profit : ℤ) (p_amount : profit = 500000) (loss : ℤ) (l_amount : loss = -200000) : 
  loss = -200000 := sorry

end loss_denotation_l530_530275


namespace sin_180_degree_l530_530634

theorem sin_180_degree : Real.sin (Real.pi) = 0 := by sorry

end sin_180_degree_l530_530634


namespace root_properties_of_P_l530_530948

noncomputable def P (x : ℝ) : ℝ := x^6 - 3*x^5 - 6*x^3 - x + 8

theorem root_properties_of_P :
  (∀ x < 0, P x > 0) ∧ ∃ x > 0, P x = 0 :=
by
  have h1 : ∀ x < 0, x^6 > 0 := λ x hx, pow_pos hx 6
  have h2 : ∀ x < 0, -3*x^5 > 0 := λ x hx, by linarith [pow_odd_pos_of_neg hx 5]
  have h3 : ∀ x < 0, -6*x^3 > 0 := λ x hx, by linarith [pow_odd_pos_of_neg hx 3]
  have h4 : ∀ x < 0, -x > 0 := λ x hx, by linarith
  have h5 : ∀ x < 0, P x > 0 := λ x hx, by linarith [h1 x hx, h2 x hx, h3 x hx, h4 x hx]
  split
  · intro x hx
    exact h5 x hx
  · apply Exists.intro
    use 0 < 1
    existsi 1
    calc P (1 : ℝ) = (1 : ℝ)^6 - 3*(1 : ℝ)^5 - 6*(1 : ℝ)^3 - 1 + 8 : rfl
    ... = 1 - 3 - 6 - 1 + 8 : by norm_num
    ... = -1 : by norm_num
  sorry -- Placeholder for completing the proof of the existence of positive roots

end root_properties_of_P_l530_530948


namespace diagonals_in_convex_polygon_with_30_sides_l530_530765

theorem diagonals_in_convex_polygon_with_30_sides : 
  let n := 30 in
  ∑ i j in finset.range n, (i + 1 - 1) * (j + 1 - 1) * (nat.choose 30 2 - 30) / 2 = 202 := 
by
  sorry

end diagonals_in_convex_polygon_with_30_sides_l530_530765


namespace gcd_98_63_l530_530393

-- The statement of the problem in Lean 4
theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end gcd_98_63_l530_530393


namespace number_of_diagonals_in_convex_polygon_l530_530746

/-- 
A theorem stating that the number of diagonals 
in a convex polygon with 30 sides is equal to 405.
-/
theorem number_of_diagonals_in_convex_polygon (n : ℕ) (h : n = 30) (convex : True) : (n * (n - 3)) / 2 = 405 := 
by 
  rw h
  norm_num
  done

end number_of_diagonals_in_convex_polygon_l530_530746


namespace part1_part2_part3_l530_530248

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

theorem part1 :
  f(x:ℝ) = 2 * Real.sin (2 * x + Real.pi / 6) := 
sorry

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3)

theorem part2 (a α β : ℝ) (hαβ : α ≠ β) (hα_bounds : α ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3)) (hβ_bounds : β ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3)) : 
  ∃ α β, g(α) = a ∧ g(β) = a ∧ α + β = 7 * Real.pi / 6 :=
sorry

theorem part3 : 
  ∃ a ∈ Set.Ioc (-2) (-Real.sqrt 3), ∀ x, g(x) = a ↔ x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3) :=
sorry

end part1_part2_part3_l530_530248


namespace simplify_expression_l530_530925

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) : 
  (5 / (4 * x^(-4))) * (2 * x^3 / 3) = (5 * x^7 / 6) :=
by
  sorry

end simplify_expression_l530_530925


namespace geometric_prog_common_ratio_one_l530_530843

variable {x y z : ℝ}
variable {r : ℝ}

theorem geometric_prog_common_ratio_one
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x)
  (hgeom : ∃ a : ℝ, a = x * (y - z) ∧ a * r = y * (z - x) ∧ a * r^2 = z * (x - y))
  (hprod : (x * (y - z)) * (y * (z - x)) * (z * (x - y)) * r^3 = (y * (z - x))^2) : 
  r = 1 := sorry

end geometric_prog_common_ratio_one_l530_530843


namespace find_150th_letter_in_pattern_l530_530446

/--
Given the repeating pattern "XYZ" with a cycle length of 3,
prove that the 150th letter in this pattern is 'Z'.
-/
theorem find_150th_letter_in_pattern : 
  let pattern := "XYZ"
  let cycle_length := String.length pattern
in (150 % cycle_length = 0) → "Z" := 
sorry

end find_150th_letter_in_pattern_l530_530446


namespace midpoint_equidistant_from_AD_l530_530573

variables {A B C D B' C' O1 O2 P Q R S : Type*}
variables [EuclideanGeometry A B C D B' C' O1 O2 P Q R S]

-- Conditions
def trapezoid (A B C D : Type*) : Prop := parallel B C A D
def reflection (X Y : Type*) (L : line) : Type* := R Y X L
def circumcenter (O : Type*) (Δ : triangle) : Prop := O.circumcenter Δ
def midpoint (M : Type*) (X Y : Type*) : Prop := M.midpoint X Y

-- Problem statement
theorem midpoint_equidistant_from_AD (h1 : trapezoid A B C D)
(h2 : reflection B B' line_CD)
(h3 : reflection C C' line_AB)
(h4 : circumcenter O1 (triangle A B C'))
(h5 : circumcenter O2 (triangle B' C D))
(h6 : midpoint P O1 O2)
(h7 : midpoint Q M N)
(h8 : midpoint R A B)
(h9 : midpoint S C D) :
distance P A = distance P D :=
sorry

end midpoint_equidistant_from_AD_l530_530573


namespace find_f_neg1_plus_f_7_l530_530735

-- Given a function f : ℝ → ℝ
axiom f : ℝ → ℝ

-- f satisfies the property of an even function
axiom even_f : ∀ x : ℝ, f (-x) = f x

-- f satisfies the periodicity of period 2
axiom periodic_f : ∀ x : ℝ, f (x + 2) = f x

-- Also, we are given that f(1) = 1
axiom f_one : f 1 = 1

-- We need to prove that f(-1) + f(7) = 2
theorem find_f_neg1_plus_f_7 : f (-1) + f 7 = 2 :=
by
  sorry

end find_f_neg1_plus_f_7_l530_530735


namespace necessary_but_not_sufficient_condition_for_purely_imaginary_l530_530536

theorem necessary_but_not_sufficient_condition_for_purely_imaginary (m : ℂ) :
  (1 - m^2 + (1 + m) * Complex.I = 0 → m = 1) ∧ 
  ((1 - m^2 + (1 + m) * Complex.I = 0 ↔ m = 1) = false) := by
  sorry

end necessary_but_not_sufficient_condition_for_purely_imaginary_l530_530536


namespace sin_180_is_zero_l530_530645

noncomputable def sin_180_eq_zero : Prop :=
  let point_zero_deg := (1, 0)
  let point_180_deg := (-1, 0)
  let y_coord_of_180_deg := 0
  sin 180 = y_coord_of_180_deg

theorem sin_180_is_zero : sin_180_eq_zero :=
  sorry

end sin_180_is_zero_l530_530645


namespace invariant_expression_l530_530340

variable {α β : Real}
variable {a b : Real}
variable {ABCD : Type}

def tetrahedron (A B C D : ABCD) : Prop :=
  ∃ (AB CD : Real) (AB_eq_a : AB = a) (CD_eq_b : CD = b)
    (dihedral_angle_AB : Real) (dihedral_angle_CD : Real)
    (dihedral_angle_AB_eq_α : dihedral_angle_AB = α)
    (dihedral_angle_CD_eq_β : dihedral_angle_CD = β), true

theorem invariant_expression (A B C D : ABCD) :
  tetrahedron A B C D →
  a^2 + b^2 + 2*a*b*Real.cot α * Real.cot β =
    a^2 + b^2 + 2*a*b*Real.cot α * Real.cot β :=
by
  intro h
  sorry

end invariant_expression_l530_530340


namespace identifyEvenFunction_l530_530593

-- Definition of even function
def isEvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Definition of the given functions
def fA : ℝ → ℝ := λ x, Real.log x
def fB : ℝ → ℝ := λ x, x^2
def fC : ℝ → ℝ := λ x, x^3
def fD : ℝ → ℝ := λ x, x + 1

-- The theorem stating the problem
theorem identifyEvenFunction :
  isEvenFunction fB ∧ ¬ isEvenFunction fA ∧ ¬ isEvenFunction fC ∧ ¬ isEvenFunction fD :=
by {
  sorry
}

end identifyEvenFunction_l530_530593


namespace find_complex_number_l530_530698

open Complex

theorem find_complex_number
  (z : ℂ)
  (h : abs z - conj z = 2 - 4 * I) :
  z = 3 - 4 * I :=
sorry

end find_complex_number_l530_530698


namespace complex_fraction_simplification_l530_530616

theorem complex_fraction_simplification :
  (∏ n in (finset.range 1007).map (λ n, 2 * n + 1), ⌊(↑((2 * n + 1) : ℚ) ^ (1/3 : ℚ))⌋) / 
  (∏ n in (finset.range 1007), ⌊(↑((2 * n + 2) : ℚ) ^ (1/3 : ℚ))⌋) =
  (12 / 35 : ℚ) := by
  sorry

end complex_fraction_simplification_l530_530616


namespace determine_n_l530_530167

theorem determine_n :
  (∃ n : ℝ, 10^n = 10^(-7) * (sqrt (10^(85) / 0.0001))) → n = 37.5 :=
by
  sorry

end determine_n_l530_530167


namespace hostel_initial_men_l530_530124

theorem hostel_initial_men (M P : ℕ) (h : P / (32 * M) = P / (40 * (M - 50))) : M = 250 :=
by {
  have h1 : 32 * M = 40 * (M - 50), from (div_eq_div_iff) h,
  calc
    32 * M     = 40 * (M - 50)       : h1
           ... =  40 * M - 2000       : by ring
           ... = 32 * M + 8 * M - 2000 : by ring
           ... = 32 * M + 8 * (M - 250) :  by ring,
  sorry
}

end hostel_initial_men_l530_530124


namespace largest_n_product_consecutive_integers_l530_530194

theorem largest_n_product_consecutive_integers : ∃ (n : ℕ), (∀ (x : ℕ), n! = (list.Ico x (x + n - 4)).prod) ∧ n = 4 := 
by
  sorry

end largest_n_product_consecutive_integers_l530_530194


namespace sugar_consumption_reduction_l530_530847

theorem sugar_consumption_reduction :
  ∀ (X : ℝ), let initial_price := 6
             let new_price := 7.5
             let initial_expenditure := initial_price * X
             let new_expenditure := new_price * (0.8 * X)
             initial_expenditure = new_expenditure →
             ((X - 0.8 * X) / X * 100) = 20 :=
  by
    intros
    unfold initial_price new_price initial_expenditure new_expenditure
    sorry

end sugar_consumption_reduction_l530_530847


namespace pages_of_dictionary_l530_530972

theorem pages_of_dictionary (h : ∃ pages, count_digit_occurrences 1 pages = 1988) : pages = 3144 :=
sorry

end pages_of_dictionary_l530_530972


namespace find_150th_letter_l530_530457
open Nat

def repeating_sequence := "XYZ"

def length_repeating_sequence := 3

theorem find_150th_letter : (150 % length_repeating_sequence == 0) → repeating_sequence[(length_repeating_sequence - 1) % length_repeating_sequence] = 'Z' := 
by
  sorry

end find_150th_letter_l530_530457


namespace sum_of_present_ages_l530_530316

def Jed_age_future (current_Jed: ℕ) (years: ℕ) : ℕ := 
  current_Jed + years

def Matt_age (current_Jed: ℕ) : ℕ := 
  current_Jed - 10

def sum_ages (jed_age: ℕ) (matt_age: ℕ) : ℕ := 
  jed_age + matt_age

theorem sum_of_present_ages :
  ∃ jed_curr_age matt_curr_age : ℕ, 
  (Jed_age_future jed_curr_age 10 = 25) ∧ 
  (jed_curr_age = matt_curr_age + 10) ∧ 
  (sum_ages jed_curr_age matt_curr_age = 20) :=
sorry

end sum_of_present_ages_l530_530316


namespace selling_price_is_1260_l530_530378

-- Definitions based on conditions
def purchase_price : ℕ := 900
def repair_cost : ℕ := 300
def gain_percent : ℕ := 5 -- percentage as a natural number

-- Known variables
def total_cost : ℕ := purchase_price + repair_cost
def gain_amount : ℕ := (gain_percent * total_cost) / 100
def selling_price : ℕ := total_cost + gain_amount

-- The theorem we want to prove
theorem selling_price_is_1260 : selling_price = 1260 := by
  sorry

end selling_price_is_1260_l530_530378


namespace area_of_enclosed_shape_l530_530934

theorem area_of_enclosed_shape :
  let f := fun x : ℝ => 1
  let g := fun x : ℝ => x^2
  (∫ x in -1..1, f x - g x) = 4 / 3 :=
by
  sorry

end area_of_enclosed_shape_l530_530934


namespace none_of_these_formulas_correspond_to_values_l530_530240

theorem none_of_these_formulas_correspond_to_values : ¬ (∃ f : ℕ → ℕ, 
  (∀ x, x = 1 → f x = 7) ∧
  (∀ x, x = 2 → f x = 17) ∧
  (∀ x, x = 3 → f x = 31) ∧
  (∀ x, x = 4 → f x = 49) ∧
  (∀ x, x = 5 → f x = 71) ∧
  (
    f = λ x, x^3 + 3 * x + 3 ∨
    f = λ x, x^2 + 4 * x + 3 ∨
    f = λ x, x^3 + x^2 + 2 * x + 1 ∨
    f = λ x, 2 * x^3 - x + 5
  ))
:=
by
  dsimp
  sorry

end none_of_these_formulas_correspond_to_values_l530_530240


namespace find_150th_letter_in_pattern_l530_530445

/--
Given the repeating pattern "XYZ" with a cycle length of 3,
prove that the 150th letter in this pattern is 'Z'.
-/
theorem find_150th_letter_in_pattern : 
  let pattern := "XYZ"
  let cycle_length := String.length pattern
in (150 % cycle_length = 0) → "Z" := 
sorry

end find_150th_letter_in_pattern_l530_530445


namespace alan_total_cost_is_84_l530_530579

def num_dark_cds : ℕ := 2
def num_avn_cds : ℕ := 1
def num_90s_cds : ℕ := 5
def price_avn_cd : ℕ := 12 -- in dollars
def price_dark_cd : ℕ := price_avn_cd * 2
def total_cost_other_cds : ℕ := num_dark_cds * price_dark_cd + num_avn_cds * price_avn_cd
def price_90s_cds : ℕ := ((40 : ℕ) * total_cost_other_cds) / 100
def total_cost_all_products : ℕ := num_dark_cds * price_dark_cd + num_avn_cds * price_avn_cd + price_90s_cds

theorem alan_total_cost_is_84 : total_cost_all_products = 84 := by
  sorry

end alan_total_cost_is_84_l530_530579


namespace digit_ten_thousandths_five_eighths_l530_530507

/-- Theorem: The digit in the ten-thousandths place of the decimal equivalent of 5/8 is 0. -/
theorem digit_ten_thousandths_five_eighths :
  let decimal_equivalent := rat.to_decimal 4 (5 / 8)
  in decimal_equivalent.to_string.drop 5.head = '0' :=
by
  sorry

end digit_ten_thousandths_five_eighths_l530_530507


namespace total_number_of_gifts_l530_530561

/-- Number of gifts calculation, given the distribution conditions with certain children -/
theorem total_number_of_gifts
  (n : ℕ) -- the total number of children
  (h1 : 2 * 4 + (n - 2) * 3 + 11 = 3 * n + 13) -- first scenario equation
  (h2 : 4 * 3 + (n - 4) * 6 + 10 = 6 * n - 2) -- second scenario equation
  : 3 * n + 13 = 28 := 
by 
  sorry

end total_number_of_gifts_l530_530561


namespace smallest_common_multiple_8_6_l530_530520

theorem smallest_common_multiple_8_6 : 
  ∃ n : ℕ, n > 0 ∧ (n % 8 = 0) ∧ (n % 6 = 0) ∧ ∀ m : ℕ, m > 0 ∧ (m % 8 = 0) ∧ (m % 6 = 0) → m ≥ n :=
begin
  use 24,
  split,
  { norm_num }, -- 24 > 0
  split,
  { norm_num }, -- 24 % 8 = 0
  split,
  { norm_num }, -- 24 % 6 = 0
  { intros m hm,
    cases hm with hp8 hp6,
    norm_num at hp8 hp6,
    sorry -- Prove that 24 is the smallest such number
  }
end

end smallest_common_multiple_8_6_l530_530520


namespace Sn_formula_l530_530224

-- Define the sequence and the conditions
def a1 : ℝ := -2/3

def S (n : ℕ) : ℝ
| 0       := a1
| (n + 1) := -1 / (S n + 2)

-- Prove the conjecture
theorem Sn_formula (n : ℕ) : S n = - (n + 1) / (n + 2) :=
by
  induction n with
  | zero =>
    unfold S a1
    norm_num
  | succ n ih =>
    unfold S
    rw [ih]
    field_simp
    sorry

end Sn_formula_l530_530224


namespace second_book_pages_l530_530146

-- Define the conditions
def total_pages : ℕ := 800
def first_book_pages : ℕ := 500
def percent_read_first_book : ℝ := 0.80 
def fraction_read_second_book : ℝ := 1 / 5
def remaining_pages : ℕ := 200 

-- Define the number of pages read from the first book
def pages_read_first_book : ℕ := percent_read_first_book * first_book_pages

-- Define the total number of pages in the second book as P
variable (P : ℕ)

-- Define the number of pages read from the second book
def pages_read_second_book : ℕ := (fraction_read_second_book * P).toNat

-- Define the main statement to be proved
theorem second_book_pages :
  pages_read_first_book + pages_read_second_book = total_pages - remaining_pages → P = 1000 := 
by
  sorry

end second_book_pages_l530_530146


namespace largest_divisor_l530_530982

theorem largest_divisor (n : ℕ) (hn : Even n) : ∃ k, ∀ n, Even n → k ∣ (n * (n+2) * (n+4) * (n+6) * (n+8)) ∧ (∀ m, (∀ n, Even n → m ∣ (n * (n+2) * (n+4) * (n+6) * (n+8))) → m ≤ k) :=
by
  use 96
  { sorry }

end largest_divisor_l530_530982


namespace divisible_by_24_l530_530923

theorem divisible_by_24 (n : ℕ) (hn : n > 0) : 24 ∣ n * (n + 2) * (5 * n - 1) * (5 * n + 1) := 
by sorry

end divisible_by_24_l530_530923


namespace diagonals_of_30_sided_polygon_l530_530806

theorem diagonals_of_30_sided_polygon : 
  ∀ (n : ℕ), n = 30 → (n * (n - 3)) / 2 = 405 := 
by
  intro n h
  rw h
  simp
  sorry

end diagonals_of_30_sided_polygon_l530_530806


namespace cars_cost_between_15000_and_20000_l530_530151

theorem cars_cost_between_15000_and_20000 (total_cars : ℕ) (p1 p2 : ℕ) :
    total_cars = 3000 → 
    p1 = 15 → 
    p2 = 40 → 
    (p1 * total_cars / 100 + p2 * total_cars / 100 + x = total_cars) → 
    x = 1350 :=
by
  intro h_total
  intro h_p1
  intro h_p2
  intro h_eq
  sorry

end cars_cost_between_15000_and_20000_l530_530151


namespace probability_valid_combination_l530_530968

def is_valid_combination (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a + b = c ∨ a + c = b ∨ b + c = a)

def count_valid_combinations (s : Finset ℕ) : ℕ :=
  (s.card == 3) ∧ ∃ (a b c : ℕ), {a, b, c} ⊆ s ∧ is_valid_combination a b c

theorem probability_valid_combination : probability (event (λ (s : Finset ℕ), count_valid_combinations s)) = 1/4 := by
  sorry

end probability_valid_combination_l530_530968


namespace sin_180_eq_0_l530_530648

theorem sin_180_eq_0 : Real.sin (180 * Real.pi / 180) = 0 :=
by
  sorry

end sin_180_eq_0_l530_530648


namespace gcd_of_sum_and_sum_of_squares_l530_530893

theorem gcd_of_sum_and_sum_of_squares {a b : ℕ} (h : Nat.gcd a b = 1) :
  Nat.gcd (a + b) (a^2 + b^2) = 1 ∨ Nat.gcd (a + b) (a^2 + b^2) = 2 :=
sorry

end gcd_of_sum_and_sum_of_squares_l530_530893


namespace letter_150th_in_pattern_l530_530492

def repeating_sequence := "XYZ"

def letter_at_position (n : ℕ) : char :=
  let seq := repeating_sequence.to_list
  seq.get! ((n - 1) % seq.length)

theorem letter_150th_in_pattern : letter_at_position 150 = 'Z' :=
by sorry

end letter_150th_in_pattern_l530_530492


namespace part_1_part_2_l530_530854

variables {A B C a b c : ℝ}
variables (h_condition1 : ∠A < π / 2)
variables (h_condition2 : ∠B < π / 2)
variables (h_condition3 : ∠C < π / 2)
variables (h_perpendicular : (a + c) * (a - c) + a * b = 0)

theorem part_1 (h_acute_triangle : ∠A + ∠B + ∠C = π) : ∠C = 2 * ∠A :=
sorry

theorem part_2 (h_acute_triangle : ∠A + ∠B + ∠C = π) : 
  ∃ (range : Set ℝ), range = {x | 3 < x ∧ x < 10 / 3} ∧
  (frac_b_a := b / a) + (2 * a / c) ^ 2 ∈ range :=
sorry

end part_1_part_2_l530_530854


namespace problem_statement_l530_530000

namespace ProofProblem

variable (t : ℚ) (y : ℚ)

/-- Given equations and condition, we want to prove y = 21 / 2 -/
theorem problem_statement (h1 : x = 3 - 2 * t) (h2 : y = 3 * t + 6) (h3 : x = 0) : y = 21 / 2 :=
by sorry

end ProofProblem

end problem_statement_l530_530000


namespace coin_problem_l530_530429

noncomputable def P_flip_heads_is_heads (coin : ℕ → bool → Prop) : ℚ :=
if coin 2 tt then 2 / 3 else 0

theorem coin_problem (coin : ℕ → bool → Prop)
  (h1 : ∀ coin_num, coin coin_num tt = (coin_num = 1 ∨ coin_num = 2))
  (h2 : ∀ coin_num, coin coin_num ff = (coin_num = 1 ∨ coin_num = 3)):
  P_flip_heads_is_heads coin = 2 / 3 :=
by
  sorry

end coin_problem_l530_530429


namespace find_b_constants_l530_530673

theorem find_b_constants :
  (∀ (θ : ℝ), cos(θ)^3 = (3 / 4) * cos(θ) + 0 * cos(2 * θ) + (1 / 4) * cos(3 * θ)) →
  (3 / 4)^2 + 0^2 + (1 / 4)^2 = 5 / 8 :=
by {
  intro h,
  sorry
}

end find_b_constants_l530_530673


namespace ratio_Pat_Mark_l530_530366

-- Total hours charged by all three
def total_hours (P K M : ℕ) : Prop :=
  P + K + M = 144

-- Pat charged twice as much time as Kate
def pat_hours (P K : ℕ) : Prop :=
  P = 2 * K

-- Mark charged 80 hours more than Kate
def mark_hours (M K : ℕ) : Prop :=
  M = K + 80

-- The ratio of Pat's hours to Mark's hours
def ratio (P M : ℕ) : ℚ :=
  (P : ℚ) / (M : ℚ)

theorem ratio_Pat_Mark (P K M : ℕ)
  (h1 : total_hours P K M)
  (h2 : pat_hours P K)
  (h3 : mark_hours M K) :
  ratio P M = (1 : ℚ) / (3 : ℚ) :=
by
  sorry

end ratio_Pat_Mark_l530_530366


namespace sine_180_eq_zero_l530_530628

theorem sine_180_eq_zero :
  sin (180 : ℝ) = 0 :=
sorry

end sine_180_eq_zero_l530_530628


namespace total_distance_l530_530358

def distance (a b : ℝ × ℝ) : ℝ :=
  real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

theorem total_distance :
  distance (3, -4) (0, 0) + distance (0, 0) (-5, 7) = 5 + real.sqrt 74 :=
by
  sorry

end total_distance_l530_530358


namespace convert_20121_base3_to_base10_l530_530070

/- Define the base conversion function for base 3 to base 10 -/
def base3_to_base10 (d4 d3 d2 d1 d0 : ℕ) :=
  d4 * 3^4 + d3 * 3^3 + d2 * 3^2 + d1 * 3^1 + d0 * 3^0

/- Define the specific number in base 3 -/
def num20121_base3 := (2, 0, 1, 2, 1)

/- The theorem stating the equivalence of the base 3 number 20121_3 to its base 10 equivalent -/
theorem convert_20121_base3_to_base10 :
  base3_to_base10 2 0 1 2 1 = 178 :=
by
  sorry

end convert_20121_base3_to_base10_l530_530070


namespace problem_solution_l530_530322

noncomputable def S : ℝ :=
  1 / (4 - Real.sqrt 15) + 1 / (Real.sqrt 15 - Real.sqrt 14) - 1 / (Real.sqrt 14 - Real.sqrt 13) + 1 / (Real.sqrt 13 - 3)

theorem problem_solution : S = 7 := 
by
  sorry

end problem_solution_l530_530322


namespace original_area_is_150_l530_530962

-- Definitions from the problem conditions
def width : ℝ := 10
def new_perimeter : ℝ := 60
def ratio : ℝ := 4 / 3

-- The rectangle with original length L and original area A
variable (L : ℝ) (A : ℝ)

-- Definitions from the new rectangle conditions
def new_length (L' : ℝ) : Prop := 2 * L' + 2 * width = new_perimeter
def new_area_condition (A' : ℝ) : Prop := A' = ratio * A

-- The new length and new area
def new_length_value := 20

-- The new area given the new length and original width
def original_area : Prop :=
  ∃ A, width = 10 ∧ new_perimeter = 60 ∧ (∃ L', new_length L' ∧ L' = new_length_value) ∧ (∃ A', new_area_condition A' ∧ A' = L' * width) ∧ A = 150

-- The theorem to prove
theorem original_area_is_150 : original_area := by
  unfold original_area
  sorry

end original_area_is_150_l530_530962


namespace probability_first_card_greater_l530_530687

theorem probability_first_card_greater :
  let cards : List ℕ := [1, 2, 3, 4, 5]
  let draw_and_replace (lst : List ℕ) : List (ℕ × ℕ) :=
    List.bind lst (λ x, List.map (λ y, (x, y)) lst)
  let total_events := draw_and_replace cards
  let favorable_events := total_events.filter (λ pair, pair.fst > pair.snd)
  (favorable_events.length : ℚ) / (total_events.length : ℚ) = 2 / 5 :=
by
  sorry

end probability_first_card_greater_l530_530687


namespace product_of_distances_equal_l530_530552

-- Definitions of points on the circle and distance
variables {k : Type*} [Field k]

structure Point (k : Type*) :=
(x : k)
(y : k)

-- Assuming the distance from a point to a line
noncomputable def distance (M : Point k) (L : lin k) : k := sorry

theorem product_of_distances_equal
  (C : set (Point k)) -- the circle
  (R : k) -- radius of the circle
  (M M1 M2 M3 M4 : Point k) 
  (hM_on_C : M ∈ C)
  (hM1_on_C : M1 ∈ C)
  (hM2_on_C : M2 ∈ C)
  (hM3_on_C : M3 ∈ C)
  (hM4_on_C : M4 ∈ C)
  : distance M (line_through M1 M2) * distance M (line_through M3 M4) =
    distance M (line_through M1 M3) * distance M (line_through M2 M4) := sorry

end product_of_distances_equal_l530_530552


namespace gray_eyed_black_haired_students_l530_530668

theorem gray_eyed_black_haired_students :
  ∀ (students : ℕ)
    (green_eyed_red_haired : ℕ)
    (black_haired : ℕ)
    (gray_eyed : ℕ),
    students = 60 →
    green_eyed_red_haired = 20 →
    black_haired = 40 →
    gray_eyed = 25 →
    (gray_eyed - (students - black_haired - green_eyed_red_haired)) = 25 := by
  intros students green_eyed_red_haired black_haired gray_eyed
  intros h_students h_green h_black h_gray
  sorry

end gray_eyed_black_haired_students_l530_530668


namespace ratio_of_pictures_l530_530209

theorem ratio_of_pictures (totalPics soldPics : ℕ) (h_total : totalPics = 153) (h_sold : soldPics = 72) :
  let stillHasPics := totalPics - soldPics in
  let gcdVal := Nat.gcd stillHasPics soldPics in
  stillHasPics / gcdVal = 9 ∧ soldPics / gcdVal = 8 := 
by
  have h_stillHasPics : stillHasPics = totalPics - soldPics := rfl
  have h_gcdVal : gcdVal = Nat.gcd stillHasPics soldPics := rfl
  sorry

end ratio_of_pictures_l530_530209


namespace ellipse_equation_l530_530707

theorem ellipse_equation :
  ∃ a b : ℝ, 0 < b ∧ b < a ∧
  (∀ x y, (x^2 / a^2 + y^2 / b^2 = 1) → 
  (∃ c : ℝ, c = 6 ∧ b = 3 ∧ a = 6 ∧
    (∀ (p f1 f2 : ℝ × ℝ), 
      (f1 = (-c, 0) ∧ f2 = (c, 0)) ∧ 
      (p = (x, y)) → (dist p f1 + dist p f2 = 12)))) :=
begin
  sorry
end

end ellipse_equation_l530_530707


namespace least_prime_factor_of_5_pow_5_minus_5_pow_3_l530_530985

theorem least_prime_factor_of_5_pow_5_minus_5_pow_3 :
  Nat.least_prime_factor (5^5 - 5^3) = 2 :=
by
  sorry

end least_prime_factor_of_5_pow_5_minus_5_pow_3_l530_530985


namespace cost_of_8_cubic_yards_topsoil_l530_530057

def cubic_yards_to_cubic_feet (yd³ : ℕ) : ℕ := 27 * yd³

def cost_of_topsoil (cubic_feet : ℕ) (cost_per_cubic_foot : ℕ) : ℕ := cubic_feet * cost_per_cubic_foot

theorem cost_of_8_cubic_yards_topsoil :
  cost_of_topsoil (cubic_yards_to_cubic_feet 8) 8 = 1728 :=
by
  sorry

end cost_of_8_cubic_yards_topsoil_l530_530057


namespace smallest_common_multiple_8_6_l530_530521

theorem smallest_common_multiple_8_6 : 
  ∃ n : ℕ, n > 0 ∧ (n % 8 = 0) ∧ (n % 6 = 0) ∧ ∀ m : ℕ, m > 0 ∧ (m % 8 = 0) ∧ (m % 6 = 0) → m ≥ n :=
begin
  use 24,
  split,
  { norm_num }, -- 24 > 0
  split,
  { norm_num }, -- 24 % 8 = 0
  split,
  { norm_num }, -- 24 % 6 = 0
  { intros m hm,
    cases hm with hp8 hp6,
    norm_num at hp8 hp6,
    sorry -- Prove that 24 is the smallest such number
  }
end

end smallest_common_multiple_8_6_l530_530521


namespace chord_square_length_eq_512_l530_530155

open Real

/-
The conditions are:
1. The radii of two smaller circles are 4 and 8.
2. These circles are externally tangent to each other.
3. Both smaller circles are internally tangent to a larger circle with radius 12.
4. A common external tangent to the two smaller circles serves as a chord of the larger circle.
-/

noncomputable def radius_small1 : ℝ := 4
noncomputable def radius_small2 : ℝ := 8
noncomputable def radius_large : ℝ := 12

/-- Show that the square of the length of the chord formed by the common external tangent of two smaller circles 
which are externally tangent to each other and internally tangent to a larger circle is 512. -/
theorem chord_square_length_eq_512 : ∃ (PQ : ℝ), PQ^2 = 512 := by
  sorry

end chord_square_length_eq_512_l530_530155


namespace nth_150th_letter_in_XYZ_l530_530486

def pattern : List Char := ['X', 'Y', 'Z']

def nth_letter (n : Nat) : Char :=
  pattern[(n - 1) % pattern.length]

theorem nth_150th_letter_in_XYZ :
  nth_letter 150 = 'Z' :=
by
  sorry

end nth_150th_letter_in_XYZ_l530_530486


namespace enclosed_area_of_curve_l530_530942

-- Define the hexagon side length
def side_length : ℝ := 2

-- Define the number of arcs and their lengths
def number_of_arcs : ℕ := 9
def arc_length : ℝ := (2 * Real.pi) / 3

-- The radius of the circular arcs based on the given arc length and full circumference.
def radius : ℝ := 1

-- Compute the area of the regular hexagon with the given side length
def hexagon_area (s : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * (s ^ 2)

-- The total area of the sectors formed by the arcs (we know there are 9 arcs)
def total_sector_area (r : ℝ) : ℝ := 9 * (1 / 2) * r^2 * (2 * Real.pi / 3)

-- The final theorem statement
theorem enclosed_area_of_curve :
  let r := radius in
  let s := side_length in
  let hex_area := hexagon_area s in
  let sector_area := total_sector_area r in
  hex_area - sector_area = π + (6 * Real.sqrt 3) :=
by
  sorry

end enclosed_area_of_curve_l530_530942


namespace polynomial_of_odd_degree_has_real_root_l530_530924

theorem polynomial_of_odd_degree_has_real_root {P : ℝ[X]} (h_odd : odd (nat_degree P)) : ∃ x : ℝ, P.eval x = 0 :=
sorry

end polynomial_of_odd_degree_has_real_root_l530_530924


namespace amare_needs_more_fabric_l530_530591

theorem amare_needs_more_fabric :
  let first_two_dresses_in_feet := 2 * 5.5 * 3
  let next_two_dresses_in_feet := 2 * 6 * 3
  let last_two_dresses_in_feet := 2 * 6.5 * 3
  let total_fabric_needed := first_two_dresses_in_feet + next_two_dresses_in_feet + last_two_dresses_in_feet
  let fabric_amare_has := 10
  total_fabric_needed - fabric_amare_has = 98 :=
by {
  sorry
}

end amare_needs_more_fabric_l530_530591


namespace inequality_holds_l530_530697

noncomputable def sum_to (f : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i, f (i + 1))

theorem inequality_holds (t : ℝ) (ht : 0 ≤ t ∧ t ≤ 1) (n : ℕ) 
  (x : Fin n.succ → ℝ) (hx : ∀ i j, i ≤ j → 1 ≥ x i ∧ x i ≥ x j ∧ x j > 0) : 
  (1 + sum_to (λ i, x ⟨i - 1, sorry⟩) n.succ) ^ t ≤ 
  1 + sum_to (λ i, i ^ (t - 1) * (x ⟨i - 1, sorry⟩) ^ t) n.succ :=  
sorry

end inequality_holds_l530_530697


namespace angle_difference_l530_530094

-- Conditions Definitions
variable (A B C X Y : Type)
variable [DecidableEq A]
variable [DecidableEq B]
variable [DecidableEq C]
variable [DecidableEq X]
variable [DecidableEq Y]

-- Variables representing points and lines
variable (P Q R : A)
variable (l1 l2 : B)
variable (XY : ℝ)
variable (ABC : Triangle)
variable (bisectorB : Line)

-- Given conditions
variable (parallel_l1_BC : parallel l1 BC)
variable (parallel_l2_AB : parallel l2 AB)
variable (bisector_intersection_XY : bisectorB ∩ l1 = X ∧ bisectorB ∩ l2 = Y)
variable (XY_eq_AC : XY = AC)

-- Final theorem to prove
theorem angle_difference (h : ABC ∈ Triangle)
  (h1 : parallel l1 BC)
  (h2 : parallel l2 AB)
  (h3 : bisectorB ∩ l1 = X ∧ bisectorB ∩ l2 = Y)
  (h4 : XY = AC) :
  ∠A - ∠C = 60 :=
sorry

end angle_difference_l530_530094


namespace smallest_n_divisible_by_5_l530_530205

def is_not_divisible_by_5 (x : ℤ) : Prop :=
  ¬ (x % 5 = 0)

def avg_is_integer (xs : List ℤ) : Prop :=
  (List.sum xs) % 5 = 0

theorem smallest_n_divisible_by_5 (n : ℕ) (h1 : n > 1980)
  (h2 : ∀ x ∈ List.range n, is_not_divisible_by_5 x)
  : n = 1985 :=
by
  -- The proof would go here
  sorry

end smallest_n_divisible_by_5_l530_530205


namespace largest_n_factorial_product_l530_530183

theorem largest_n_factorial_product : ∃ n : ℕ, (∀ m : ℕ, m ≥ n → ¬ (m! = ∏ i in (range (m - 4)).map (λ k, k + 1 + m), (1 : ℕ))) ∧ n = 1 :=
sorry

end largest_n_factorial_product_l530_530183


namespace count_obtuse_triangle_values_l530_530032

theorem count_obtuse_triangle_values :
  {k : ℕ // k > 0 ∧ (5 + 12 > k) ∧ (5 + k > 12) ∧ (12 + k > 5) ∧ (k = 8 ∨ k = 9 ∨ k = 10 ∨ k = 14 ∨ k = 15 ∨ k = 16)}.card = 6 := 
by
  sorry

end count_obtuse_triangle_values_l530_530032


namespace second_cannibal_wins_l530_530939

noncomputable def cannibal_can_consume_opponent 
  (n : ℕ) 
  (position : ℕ × ℕ) 
  (move : (ℕ × ℕ) → (ℕ × ℕ) -> Prop) 
  (start_pos1 start_pos2 : ℕ × ℕ) 
  (opponent_consumed : Prop) : Prop :=
∀ (k : ℕ), 
  (1 ≤ position.1 ∧ position.1 ≤ n) ∧ 
  (1 ≤ position.2 ∧ position.2 ≤ n) ∧
  (move position start_pos1 ∨ move position start_pos2) ∧ 
  starts_opposite_corner start_pos1 start_pos2 

theorem second_cannibal_wins 
  (n : ℕ) 
  (start_pos1 start_pos2 : ℕ × ℕ) 
  (move_like_king : ∀ (pos1 pos2 : ℕ × ℕ), (move pos1 pos2 -> bool))
  (starts_opposite_corner : ℕ × ℕ -> ℕ × ℕ -> bool) : 
  cannibal_can_consume_opponent n (start_pos1, start_pos2) (λ pos1 pos2, if move_like_king pos1 pos2 then true else false) start_pos1 start_pos2 True :=
sorry

end second_cannibal_wins_l530_530939


namespace cafeteria_earnings_l530_530966

def initial_apples := 80
def initial_oranges := 60
def initial_bananas := 40

def remaining_apples := 25
def remaining_oranges := 15
def remaining_bananas := 5

def apple_cost := 1.20
def orange_cost := 0.75
def banana_cost := 0.55

def sold_apples := initial_apples - remaining_apples
def sold_oranges := initial_oranges - remaining_oranges
def sold_bananas := initial_bananas - remaining_bananas

def earnings_apples := sold_apples * apple_cost
def earnings_oranges := sold_oranges * orange_cost
def earnings_bananas := sold_bananas * banana_cost

def total_earnings := earnings_apples + earnings_oranges + earnings_bananas

theorem cafeteria_earnings :
  total_earnings = 119.00 :=
by sorry

end cafeteria_earnings_l530_530966


namespace geom_seq_sum_l530_530861

theorem geom_seq_sum {a : ℕ → ℝ} (q : ℝ) (h1 : a 0 + a 1 + a 2 = 2)
    (h2 : a 3 + a 4 + a 5 = 16)
    (h_geom : ∀ n, a (n + 1) = q * a n) :
  a 6 + a 7 + a 8 = 128 :=
sorry

end geom_seq_sum_l530_530861


namespace find_numbers_l530_530411

theorem find_numbers (x y : ℤ) (h1 : x + y = 18) (h2 : x - y = 24) : x = 21 ∧ y = -3 :=
by
  sorry

end find_numbers_l530_530411


namespace number_of_diagonals_30_sides_l530_530774

def number_of_diagonals (n : ℕ) : ℕ :=
  nat.choose n 2 - n

theorem number_of_diagonals_30_sides :
  number_of_diagonals 30 = 405 :=
by {
  rw [number_of_diagonals, nat.choose, nat.factorial, nat.factorial, nat.factorial],
  -- The proof would proceed to simplify the combination and subtraction, but we use sorry to skip details.
  sorry,
}

end number_of_diagonals_30_sides_l530_530774


namespace certain_number_divides_expression_l530_530681

theorem certain_number_divides_expression : 
  ∃ m : ℕ, (∃ n : ℕ, n = 6 ∧ m ∣ (11 * n - 1)) ∧ m = 65 := 
by
  sorry

end certain_number_divides_expression_l530_530681


namespace inradii_exradii_relation_l530_530351

-- Let's define the necessary variables
variable (A B C M : Point)
-- Define the radii of the incircles and excircles
variable (r r1 r2 q q1 q2 : ℝ)

-- Define the conditions stated
axiom (hM : M ∈ Segment A B)
axiom (hr : r = inradius (Triangle.mk A B C))
axiom (hr1 : r1 = inradius (Triangle.mk A M C))
axiom (hr2 : r2 = inradius (Triangle.mk B M C))
axiom (hq : q = exradius_opposite_C (Triangle.mk A B C))
axiom (hq1 : q1 = exradius_opposite_CA (Triangle.mk A M C))
axiom (hq2 : q2 = exradius_opposite_CB (Triangle.mk B M C))

-- The problem statement
theorem inradii_exradii_relation : r1 * r2 * q = r * q1 * q2 := sorry

end inradii_exradii_relation_l530_530351


namespace all_d_zero_l530_530902

def d (n m : ℕ) : ℤ := sorry -- or some explicit initial definition

theorem all_d_zero (n m : ℕ) (h₁ : n ≥ 0) (h₂ : 0 ≤ m) (h₃ : m ≤ n) :
  (m = 0 ∨ m = n → d n m = 0) ∧
  (0 < m ∧ m < n → m * d n m = m * d (n - 1) m + (2 * n - m) * d (n - 1) (m - 1))
:=
  sorry

end all_d_zero_l530_530902


namespace diagonals_in_30_sided_polygon_l530_530821

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem diagonals_in_30_sided_polygon :
  number_of_diagonals 30 = 405 :=
by sorry

end diagonals_in_30_sided_polygon_l530_530821


namespace find_150th_letter_in_pattern_l530_530442

/--
Given the repeating pattern "XYZ" with a cycle length of 3,
prove that the 150th letter in this pattern is 'Z'.
-/
theorem find_150th_letter_in_pattern : 
  let pattern := "XYZ"
  let cycle_length := String.length pattern
in (150 % cycle_length = 0) → "Z" := 
sorry

end find_150th_letter_in_pattern_l530_530442


namespace triangle_problem_solution_l530_530288

noncomputable def A : ℝ := sorry -- The angle A can't be directly derived without proof steps.
noncomputable def B : ℝ := sorry -- The angle B can't be directly derived without proof steps.
noncomputable def C : ℝ := sorry -- The angle C can't be directly derived without proof steps.

def a : ℝ := Real.sqrt 5
def b : ℝ := 3
def sin_C : ℝ := 2 * Real.sin A

def expected_c : ℝ := 2 * Real.sqrt 5
def expected_sin_2A_minus_pi4 : ℝ := Real.sqrt 2 / 10

theorem triangle_problem_solution :
  let c := (sin_C / Real.sin A) * a in
  c = expected_c ∧ 
  let cos_A := (c^2 + b^2 - a^2) / (2 * b * c) in
  let sin_A := Real.sqrt (1 - cos_A^2) in
  let sin_2A := 2 * sin_A * cos_A in
  let cos_2A := cos_A^2 - sin_A^2 in
  let sin_2A_minus_pi4 := sin_2A * Real.cos (π / 4) - cos_2A * Real.sin (π / 4) in
  sin_2A_minus_pi4 = expected_sin_2A_minus_pi4 :=
by 
  sorry -- Proof skipped

end triangle_problem_solution_l530_530288


namespace largest_n_factorial_product_l530_530186

theorem largest_n_factorial_product :
  ∃ n : ℕ, (∀ a : ℕ, (n > 0) → (n! = (∏ k in finset.range (n - 4 + a), k + 1))) → n = 4 :=
begin
  sorry
end

end largest_n_factorial_product_l530_530186


namespace intersection_is_14_l530_530257

open Set

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {y | ∃ x ∈ A, y = 3 * x - 2}

theorem intersection_is_14 : A ∩ B = {1, 4} := 
by sorry

end intersection_is_14_l530_530257


namespace smallest_positive_root_floor_l530_530329

noncomputable def g (x : ℝ) := 3 * Real.sin x - Real.cos x + 2 * Real.tan x

theorem smallest_positive_root_floor :
  let s := Inf {x : ℝ | 0 < x ∧ g x = 0} in
  floor s = 3 :=
by
  sorry

end smallest_positive_root_floor_l530_530329


namespace tate_education_ratio_l530_530005

theorem tate_education_ratio
  (n : ℕ)
  (m : ℕ)
  (h1 : n > 1)
  (h2 : (n - 1) + m * (n - 1) = 12)
  (h3 : n = 4) :
  (m * (n - 1)) / (n - 1) = 3 := 
by 
  sorry

end tate_education_ratio_l530_530005


namespace smallest_positive_integer_l530_530524

theorem smallest_positive_integer (x : ℕ) (hx_pos : x > 0) (h : x < 15) : x = 1 :=
by
  sorry

end smallest_positive_integer_l530_530524


namespace circumference_approx_l530_530300

noncomputable def cylinder_height : ℝ := 40 / 3

noncomputable def cylinder_volume : ℝ := 2000 * 1.62

noncomputable def pi_approx : ℝ := 3

theorem circumference_approx (h : ℝ) (V : ℝ) (π : ℝ) (r : ℝ) :
  h = cylinder_height → V = cylinder_volume → π = pi_approx → r = 9 → (2 * π * r ≈ 54) := 
by
  intros h_eq V_eq π_eq r_eq
  sorry

end circumference_approx_l530_530300


namespace product_multiple_of_4_probability_l530_530162

-- Setting up the sample spaces
def X : set ℕ := {2, 4, 6, 8, 10}
def Y : set ℕ := {3, 5, 9}

-- Defining the condition for a product to be a multiple of 4
def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

-- Counting the favorable outcomes (products that are multiples of 4)
def favorable_outcomes : ℤ := (X.filter (fun x => is_multiple_of_4 x)).card * Y.card

-- Counting the total possible outcomes
def total_outcomes : ℤ := X.card * Y.card

-- Calculating the probability as a fraction
def probability : ℚ := favorable_outcomes / total_outcomes

-- Statement of the problem to prove the probability is 2/5
theorem product_multiple_of_4_probability : probability = (2 / 5 : ℚ) :=
  by
  sorry

end product_multiple_of_4_probability_l530_530162


namespace decreasing_sqrt_radicand_l530_530675

noncomputable def is_decreasing_on (f : ℝ → ℝ) (s : set ℝ) : Prop :=
∀ ⦃a b⦄, a ∈ s → b ∈ s → a < b → f b ≤ f a

noncomputable def radicand (x : ℝ) : ℝ := -x^2 + 2*x + 3

theorem decreasing_sqrt_radicand : 
  open_interval (-1 : ℝ) (3 : ℝ) → 
  (∀ x, -1 ≤ x ∧ x ≤ 3 → radicand x ≥ 0) → 
  is_decreasing_on (λ x, real.sqrt (radicand x)) (set.Icc 1 3) :=
by
  sorry

end decreasing_sqrt_radicand_l530_530675


namespace sin_180_eq_zero_l530_530619

theorem sin_180_eq_zero : Real.sin (180 * Real.pi / 180) = 0 := by
  -- Simplifying the angle, 180 degrees = π radians
  let angle := 180 * Real.pi / 180
  have h : angle = Real.pi := by
    simp [angle, Real.pi]
  rw h
  -- From the unit circle, we know the sine of angle π is 0
  exact Real.sin_pi

end sin_180_eq_zero_l530_530619


namespace letter_150_is_Z_l530_530505

/-- Definition of the repeating pattern "XYZ" -/
def pattern : List Char := ['X', 'Y', 'Z']

/-- The repeating pattern has a length of 3 -/
def pattern_length : ℕ := 3

/-- Calculate the 150th letter in the repeating pattern "XYZ" -/
def nth_letter_in_pattern (n : ℕ) : Char :=
  let m := n % pattern_length
  if m = 0 then pattern[2] else pattern[m - 1]

/-- Prove that the 150th letter in the pattern "XYZ" is 'Z' -/
theorem letter_150_is_Z : nth_letter_in_pattern 150 = 'Z' :=
by
  sorry

end letter_150_is_Z_l530_530505


namespace find_y_l530_530323

noncomputable def x : ℝ := (4 / 25)^(1 / 3)

theorem find_y (y : ℝ) (h1 : 0 < y) (h2 : y < x) (h3 : x^x = y^y) : y = (32 / 3125)^(1 / 3) :=
sorry

end find_y_l530_530323


namespace remaining_space_after_backup_l530_530360

theorem remaining_space_after_backup (total_space : ℕ) (file_space : ℕ) (h1 : total_space = 28) (h2 : file_space = 26) : total_space - file_space = 2 :=
by
  rw [h1, h2]
  simp
  sorry

end remaining_space_after_backup_l530_530360


namespace smallest_common_multiple_8_6_l530_530515

theorem smallest_common_multiple_8_6 : ∃ n : ℕ, n > 0 ∧ (8 ∣ n) ∧ (6 ∣ n) ∧ ∀ m : ℕ, m > 0 ∧ (8 ∣ m) ∧ (6 ∣ m) → n ≤ m :=
begin
  use 24,
  split,
  { exact zero_lt_24, },
  split,
  { exact dvd.intro 3 rfl, },
  split,
  { exact dvd.intro 4 rfl, },
  intros m hm h8 h6,
  -- actual proof here
  sorry
end

end smallest_common_multiple_8_6_l530_530515


namespace complement_intersection_l530_530230

open Set

def A := {x : ℝ | x^2 - x - 12 ≤ 0}
def B := {x : ℝ | sqrt x ≤ 2}
def C := {x : ℝ | abs (x + 1) ≤ 2}

theorem complement_intersection (x : ℝ) : x ∈ A \ (B ∩ C) ↔ x ∈ ((Icc (-3 : ℝ) 0).Ioo ∪ (Ioo 1 4)) :=
by {
  sorry,
}

end complement_intersection_l530_530230


namespace topsoil_cost_l530_530050

theorem topsoil_cost 
  (cost_per_cubic_foot : ℝ)
  (cubic_yards_to_cubic_feet : ℝ)
  (cubic_yards : ℝ) :
  cost_per_cubic_foot = 8 →
  cubic_yards_to_cubic_feet = 27 →
  cubic_yards = 8 →
  (cubic_yards * cubic_yards_to_cubic_feet * cost_per_cubic_foot) = 1728 :=
begin
  intros h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
end

end topsoil_cost_l530_530050


namespace nth_150th_letter_in_XYZ_l530_530487

def pattern : List Char := ['X', 'Y', 'Z']

def nth_letter (n : Nat) : Char :=
  pattern[(n - 1) % pattern.length]

theorem nth_150th_letter_in_XYZ :
  nth_letter 150 = 'Z' :=
by
  sorry

end nth_150th_letter_in_XYZ_l530_530487


namespace area_of_enclosed_shape_l530_530933

theorem area_of_enclosed_shape :
  let f := fun x : ℝ => 1
  let g := fun x : ℝ => x^2
  (∫ x in -1..1, f x - g x) = 4 / 3 :=
by
  sorry

end area_of_enclosed_shape_l530_530933


namespace subsets_count_l530_530738

-- Define the set S
def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Define the property of subset T of S having at least 2 elements and abs difference condition
def valid_subset (T : Set ℕ) : Prop :=
  2 ≤ T.size ∧ (∀ {a b : ℕ}, a ∈ T → b ∈ T → a ≠ b → |a - b| > 1)

-- Define the number of valid subsets of S
noncomputable def number_of_valid_subsets : ℕ :=
  Set.card { T : Set ℕ | T ⊆ S ∧ valid_subset T }

-- State the theorem to prove
theorem subsets_count : number_of_valid_subsets = 133 :=
  sorry

end subsets_count_l530_530738


namespace perpendicular_distance_R_to_SPQ_plane_is_4_l530_530543

structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def SPQ_Plane (S P Q : Point3D) : ℝ × ℝ × ℝ :=
  let SP := (P.x - S.x, P.y - S.y, P.z - S.z)
  let SQ := (Q.x - S.x, Q.y - S.y, Q.z - S.z)
  let normal := (SP.2 * SQ.3 - SP.3 * SQ.2, SP.3 * SQ.1 - SP.1 * SQ.3, SP.1 * SQ.2 - SP.2 * SQ.1)
  normal

def distance_to_plane (R : Point3D) (a b c : ℝ): ℝ :=
  let d := 0
  |a * R.x + b * R.y + c * R.z + d| / sqrt (a ^ 2 + b ^ 2 + c ^ 2)

def S : Point3D := ⟨0, 0, 0⟩
def P : Point3D := ⟨6, 0, 0⟩
def Q : Point3D := ⟨0, 5, 0⟩
def R : Point3D := ⟨0, 0, 4⟩

theorem perpendicular_distance_R_to_SPQ_plane_is_4 :
  let (a, b, c) := SPQ_Plane S P Q
  distance_to_plane R a b c = 4 :=
by sorry

end perpendicular_distance_R_to_SPQ_plane_is_4_l530_530543


namespace problem1_problem2_problem3_problem4_l530_530609

-- Statement for problem 1
theorem problem1 : -12 + (-6) - (-28) = 10 :=
  by sorry

-- Statement for problem 2
theorem problem2 : (-8 / 5) * (15 / 4) / (-9) = 2 / 3 :=
  by sorry

-- Statement for problem 3
theorem problem3 : (-3 / 16 - 7 / 24 + 5 / 6) * (-48) = -17 :=
  by sorry

-- Statement for problem 4
theorem problem4 : -3^2 + (7 / 8 - 1) * (-2)^2 = -9.5 :=
  by sorry

end problem1_problem2_problem3_problem4_l530_530609


namespace optimal_voltage_minimizes_cost_l530_530405

section LightBulbOptimization

-- Definitions from conditions
def E (U : ℝ) : ℝ := 2 * 10^6 * 2^(-U / 10)
def G (U : ℝ) : ℝ := 4 * 10^(-5) * (U / 1) ^ 1.7
def R : ℝ := 200
def energy_price : ℝ := 2.1
def bulb_price : ℝ := 8

-- Prove that the optimal voltage minimizing the cost is approximately 138.5V
theorem optimal_voltage_minimizes_cost :
    let α : ℝ := 42.5
    let β : ℝ := 0.372
    let f (U : ℝ) : ℝ := α * U^(-1.7) * (1 + β * 2^(0.1 * U) * U^(-2))
in ∃ (Umin : ℝ), Umin ≈ 138.5 ∧ (∀ U : ℝ, U > 0 → f Umin ≤ f U) :=
sorry

end LightBulbOptimization

end optimal_voltage_minimizes_cost_l530_530405


namespace significant_improvement_l530_530550

def indicator_data_old : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def indicator_data_new : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def sample_mean (data : List ℝ) : ℝ := (data.sum / data.length.toReal)

def sample_variance (data : List ℝ) : ℝ := 
  let mean := sample_mean data
  (data.map (λ x => (x - mean) ^ 2)).sum / data.length.toReal

-- Given means and variances directly
def x̄ := 10
def ȳ := 10.3
def s1_sq := 0.036
def s2_sq := 0.04

theorem significant_improvement :
  10.3 - 10 ≥ 2 * Real.sqrt((0.036 + 0.04) / 10) :=
by sorry

end significant_improvement_l530_530550


namespace evaluate_expression1_evaluate_expression2_l530_530170

theorem evaluate_expression1 : (9 / 4) ^ (1 / 2) + (8 / 27) ^ (-1 / 3) = 3 := 
by 
  sorry

theorem evaluate_expression2 : log 4 12 - log 4 3 = 1 := 
by 
  sorry

end evaluate_expression1_evaluate_expression2_l530_530170


namespace find_150th_letter_l530_530470

def pattern : List Char := ['X', 'Y', 'Z']

def position (N : ℕ) (pattern_length : ℕ) : ℕ :=
  if N % pattern_length = 0 then pattern_length else N % pattern_length

theorem find_150th_letter : 
  let pattern_length := 3 in (position 150 pattern_length = 3) ∧ (pattern.drop (position 150 pattern_length - 1)).head = 'Z' :=
by
  sorry

end find_150th_letter_l530_530470


namespace total_collection_l530_530559

theorem total_collection (n : ℕ) (c : ℕ) (h1 : n = 68) (h2 : c = 68) : ℝ :=
  let total_paise := n * c
  let total_rupees := (total_paise : ℝ) / 100
  have h3 : total_paise = 4624 := by sorry
  have h4 : total_rupees = 46.24 := by sorry
  h4

end total_collection_l530_530559


namespace subset_S_l530_530307

def is_adjacent (P Q : ℤ × ℤ × ℤ) : Prop :=
  let (x, y, z) := P in
  let (u, v, w) := Q in
  Int.natAbs (x - u) + Int.natAbs (y - v) + Int.natAbs (z - w) = 1

def S : set (ℤ × ℤ × ℤ) :=
  {P : ℤ × ℤ × ℤ | 7 ∣ (P.1 + 2 * P.2 + 3 * P.3)}

theorem subset_S (P : ℤ × ℤ × ℤ) :
  ∃ (S : set (ℤ × ℤ × ℤ)), (∀ P ∈ T, is_adjacent P Q → Q ∈ S ∨ P ∈ S) :=
sorry

end subset_S_l530_530307


namespace area_of_shaded_region_l530_530030

theorem area_of_shaded_region (d : ℝ) (n : ℕ) (s : ℝ) (area_total : ℝ) :
  d = 10 ∧ n = 25 ∧ s = d / Real.sqrt 2 ∧ area_total = n * s ^ 2 → area_total = 50 :=
by
  intro h
  cases h with hd hnss
  cases hnss with hn hss
  cases hss with hs h_area
  rw [hd, hs] at h_area
  simp [Real.sqrt] at h_area
  assumption

end area_of_shaded_region_l530_530030


namespace xy_square_value_l530_530834

theorem xy_square_value (x y : ℝ) (h1 : x * (x + y) = 24) (h2 : y * (x + y) = 72) : (x + y)^2 = 96 :=
by
  sorry

end xy_square_value_l530_530834


namespace number_of_diagonals_in_convex_polygon_l530_530747

/-- 
A theorem stating that the number of diagonals 
in a convex polygon with 30 sides is equal to 405.
-/
theorem number_of_diagonals_in_convex_polygon (n : ℕ) (h : n = 30) (convex : True) : (n * (n - 3)) / 2 = 405 := 
by 
  rw h
  norm_num
  done

end number_of_diagonals_in_convex_polygon_l530_530747


namespace alan_total_cost_l530_530584

theorem alan_total_cost :
  let price_AVN_CD := 12 in
  let price_The_Dark_CD := price_AVN_CD * 2 in
  let total_cost_The_Dark_CDs := 2 * price_The_Dark_CD in
  let total_cost_before_90s_CDs := price_AVN_CD + total_cost_The_Dark_CDs in
  let cost_90s_CDs := 0.4 * total_cost_before_90s_CDs in
  let total_cost := total_cost_before_90s_CDs + cost_90s_CDs in
  total_cost = 84 :=
by
  let price_AVN_CD := 12
  let price_The_Dark_CD := price_AVN_CD * 2
  let total_cost_The_Dark_CDs := 2 * price_The_Dark_CD
  let total_cost_before_90s_CDs := price_AVN_CD + total_cost_The_Dark_CDs
  let cost_90s_CDs := 0.4 * total_cost_before_90s_CDs
  let total_cost := total_cost_before_90s_CDs + cost_90s_CDs
  show total_cost = 84, from sorry

end alan_total_cost_l530_530584


namespace outer_not_necessarily_greater_l530_530061

noncomputable def edge_length (A B : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2 + (A.3 - B.3)^2)

structure Tetrahedron :=
  (A B C D : ℝ × ℝ × ℝ)
  (D_inside : ∃ λ μ ν, λ + μ + ν ≤ 1 ∧ λ ≥ 0 ∧ μ ≥ 0 ∧ ν ≥ 0 ∧
                        D.1 = λ * A.1 + μ * B.1 + ν * C.1 ∧
                        D.2 = λ * A.2 + μ * B.2 + ν * C.2 ∧
                        D.3 = λ * A.3 + μ * B.3 + ν * C.3)

def sum_of_edge_lengths (T : Tetrahedron) : ℝ :=
  edge_length T.A T.B + edge_length T.B T.C + edge_length T.C T.A +
  edge_length T.A T.D + edge_length T.B T.D + edge_length T.C T.D

theorem outer_not_necessarily_greater 
  (ABCD_inner : Tetrahedron) (ABCD_outer : Tetrahedron) (E : ℝ × ℝ × ℝ) 
  (H : E.1 = ABD_inner.D.1 ∧ E.2 = ABD_inner.D.2 ∧ E.3 = ABD_inner.D.3):
  ¬ (sum_of_edge_lengths ABCD_outer > sum_of_edge_lengths ABCD_inner) := sorry

end outer_not_necessarily_greater_l530_530061


namespace shape_described_by_spherical_coordinates_l530_530208

theorem shape_described_by_spherical_coordinates (c : ℝ) (hc : c > 0) :
  (∃ (shape : string), shape = "Cone" ∧ ∀ (ρ φ : ℝ), ρ = c * real.sin φ → shape = "Cone") :=
by
  existsi "Cone"
  split
  { refl }
  { intros ρ φ h
    sorry }

end shape_described_by_spherical_coordinates_l530_530208


namespace find_percentage_decrease_l530_530133

-- Define the conditions
def first_quarter_sales : ℝ := 1.0
def model_A_first_quarter_sales : ℝ := 0.56 * first_quarter_sales
def model_A_second_quarter_sales : ℝ := model_A_first_quarter_sales * 1.23
def total_sales_second_quarter : ℝ := first_quarter_sales * 1.12
def unknown_decrease : ℝ := 1 - a / 100

-- Proof problem statement
theorem find_percentage_decrease (a : ℝ) :
  0.56 * 1.23 + (1 - 0.56) * (1 - a / 100) = 1.12 → a = 2 :=
by
  sorry

end find_percentage_decrease_l530_530133


namespace elberta_amount_l530_530263

theorem elberta_amount (grannySmith_amount : ℝ) (Anjou_factor : ℝ) (extra_amount : ℝ) :
  grannySmith_amount = 45 →
  Anjou_factor = 1 / 4 →
  extra_amount = 4 →
  (extra_amount + Anjou_factor * grannySmith_amount) = 15.25 :=
by
  intros h_grannySmith h_AnjouFactor h_extraAmount
  sorry

end elberta_amount_l530_530263


namespace letter_150_is_Z_l530_530498

/-- Definition of the repeating pattern "XYZ" -/
def pattern : List Char := ['X', 'Y', 'Z']

/-- The repeating pattern has a length of 3 -/
def pattern_length : ℕ := 3

/-- Calculate the 150th letter in the repeating pattern "XYZ" -/
def nth_letter_in_pattern (n : ℕ) : Char :=
  let m := n % pattern_length
  if m = 0 then pattern[2] else pattern[m - 1]

/-- Prove that the 150th letter in the pattern "XYZ" is 'Z' -/
theorem letter_150_is_Z : nth_letter_in_pattern 150 = 'Z' :=
by
  sorry

end letter_150_is_Z_l530_530498


namespace diagonals_in_30_sided_polygon_l530_530824

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem diagonals_in_30_sided_polygon :
  number_of_diagonals 30 = 405 :=
by sorry

end diagonals_in_30_sided_polygon_l530_530824


namespace inequality_semi_perimeter_l530_530911

variables {R r p : Real}

theorem inequality_semi_perimeter (h1 : 0 < R) (h2 : 0 < r) (h3 : 0 < p) :
  16 * R * r - 5 * r^2 ≤ p^2 ∧ p^2 ≤ 4 * R^2 + 4 * R * r + 3 * r^2 :=
sorry

end inequality_semi_perimeter_l530_530911


namespace shortest_distance_exp_graph_to_line_l530_530886

open Real

theorem shortest_distance_exp_graph_to_line :
  ∀ (P : ℝ × ℝ), P.2 = exp P.1 → 
  ∃ Q : ℝ × ℝ, Q = (0, 1) ∧ dist P (1, 1) = sqrt 2 / 2 := 
by
  sorry

end shortest_distance_exp_graph_to_line_l530_530886


namespace general_formula_l530_530256

-- Define the sequence term a_n
def sequence_term (n : ℕ) : ℚ :=
  if h : n = 0 then 1
  else (2 * n - 1 : ℚ) / (n * n)

-- State the theorem for the general formula of the nth term
theorem general_formula (n : ℕ) (hn : n ≠ 0) : 
  sequence_term n = (2 * n - 1 : ℚ) / (n * n) :=
by sorry

end general_formula_l530_530256


namespace letter_at_position_150_l530_530478

theorem letter_at_position_150 : 
  (∀ n, n > 0 → ∃ i, i ∈ {1, 2, 3} ∧ "XYZ".to_list[i-1] = "XYZ".to_list[(n - 1) % 3]) →
  ("XYZ".to_list[(150 - 1) % 3] = 'Z') :=
by
  sorry

end letter_at_position_150_l530_530478


namespace bottle_R_cost_l530_530153

-- Definitions based on conditions
def capsules_R := 250
def capsules_T := 100
def cost_T := 3.0
def difference_per_capsule := 0.005

-- Define the cost per capsule for each bottle
def cost_per_capsule_R (R : ℝ) := R / capsules_R
def cost_per_capsule_T := cost_T / capsules_T

-- Define the main theorem
theorem bottle_R_cost (R : ℝ) (h : cost_per_capsule_R R - cost_per_capsule_T = difference_per_capsule) : R = 8.75 := 
by
  sorry

end bottle_R_cost_l530_530153


namespace find_150th_letter_l530_530465

theorem find_150th_letter :
  let pattern := ['X', 'Y', 'Z']
  150 % 3 = 0 -> pattern[(150 % 3 + 2) % 3] = 'Z' :=
begin
  intros pattern h,
  simp at *,
  exact rfl,
end

end find_150th_letter_l530_530465


namespace sequence_not_periodic_with_period_2_l530_530026

def sequence (a : ℝ) : ℕ → ℝ
| 0       := a
| (n + 1) := 2 * sequence a n * (1 - sequence a n)

theorem sequence_not_periodic_with_period_2 (a : ℝ) (h : 0 ≤ a ∧ a ≤ 1) : 
  ¬ (∀ n, sequence a (n + 2) = sequence a n) :=
sorry

end sequence_not_periodic_with_period_2_l530_530026


namespace find_150th_letter_l530_530468

def pattern : List Char := ['X', 'Y', 'Z']

def position (N : ℕ) (pattern_length : ℕ) : ℕ :=
  if N % pattern_length = 0 then pattern_length else N % pattern_length

theorem find_150th_letter : 
  let pattern_length := 3 in (position 150 pattern_length = 3) ∧ (pattern.drop (position 150 pattern_length - 1)).head = 'Z' :=
by
  sorry

end find_150th_letter_l530_530468


namespace percentage_of_apples_is_50_l530_530108

-- Definitions based on the conditions
def initial_apples : ℕ := 10
def initial_oranges : ℕ := 23
def oranges_removed : ℕ := 13

-- Final percentage calculation after removing 13 oranges
def percentage_apples (apples oranges_removed : ℕ) :=
  let total_initial := initial_apples + initial_oranges
  let oranges_left := initial_oranges - oranges_removed
  let total_after_removal := initial_apples + oranges_left
  (initial_apples * 100) / total_after_removal

-- The theorem to be proved
theorem percentage_of_apples_is_50 : percentage_apples initial_apples oranges_removed = 50 := by
  sorry

end percentage_of_apples_is_50_l530_530108


namespace min_formula_l530_530544

theorem min_formula (a b : ℝ) : 
  min a b = (a + b - Real.sqrt ((a - b) ^ 2)) / 2 :=
by
  sorry

end min_formula_l530_530544


namespace find_150th_letter_l530_530466

def pattern : List Char := ['X', 'Y', 'Z']

def position (N : ℕ) (pattern_length : ℕ) : ℕ :=
  if N % pattern_length = 0 then pattern_length else N % pattern_length

theorem find_150th_letter : 
  let pattern_length := 3 in (position 150 pattern_length = 3) ∧ (pattern.drop (position 150 pattern_length - 1)).head = 'Z' :=
by
  sorry

end find_150th_letter_l530_530466


namespace mary_stickers_problem_l530_530359

theorem mary_stickers_problem:
  ∀ (total_stickers : ℕ) (friends : ℕ) (stickers_per_friend : ℕ) (stickers_per_other : ℕ) (leftover_stickers : ℕ) (total_students : ℕ), 
  total_stickers = 50 → 
  friends = 5 → 
  stickers_per_friend = 4 → 
  stickers_per_other = 2 → 
  leftover_stickers = 8 → 
  total_students = friends + ℕ.div (total_stickers - friends * stickers_per_friend - leftover_stickers) stickers_per_other + 1 →
  total_students = 17 :=
begin
  intros total_stickers friends stickers_per_friend stickers_per_other leftover_stickers total_students,
  intros ht hf hsfo hspo hls hts,
  have calculate_stickers := (total_stickers - friends * stickers_per_friend - leftover_stickers) / stickers_per_other,
  rw ←calculate_stickers at hts,
  exact hts,
end

end mary_stickers_problem_l530_530359


namespace S_value_l530_530836

noncomputable def S : ℝ := (-95 + 3 * Real.sqrt 1505) / 10

theorem S_value 
  (S_pos : S > 0)
  (series_eq : (∑ k in Finset.range 21, 1 / ((S + k) * (S + k - 1))) = 1 - 1 / (R : ℝ)) :
  S = ((-95 + 3 * Real.sqrt 1505) / 10) :=
by
  -- Proof omitted
  sorry

end S_value_l530_530836


namespace sin_180_degree_l530_530633

theorem sin_180_degree : Real.sin (Real.pi) = 0 := by sorry

end sin_180_degree_l530_530633


namespace diagonals_in_30_sided_polygon_l530_530818

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem diagonals_in_30_sided_polygon :
  number_of_diagonals 30 = 405 :=
by sorry

end diagonals_in_30_sided_polygon_l530_530818


namespace triangle_converges_to_equilateral_l530_530220

open Real

theorem triangle_converges_to_equilateral 
  (O : Point) (A B C : Point) 
  (h1 : inscribed_triangle O A B C)
  (midpoint_of_arc : ∀ l m : Point, Point)
  (A1 B1 C1 : Point) (hA1 : A1 = midpoint_of_arc B C)
  (hB1 : B1 = midpoint_of_arc C A)
  (hC1 : C1 = midpoint_of_arc A B) 
  (sequence_of_midpoints : ℕ → Point → Point)
  (An Bn Cn : ℕ → Point)
  (hAn : ∀ n, An (n + 1) = midpoint_of_arc (Bn n) (Cn n))
  (hBn : ∀ n, Bn (n + 1) = midpoint_of_arc (Cn n) (An n))
  (hCn : ∀ n, Cn (n + 1) = midpoint_of_arc (An n) (Bn n)) :
    ∀ ε > 0, ∃ N, ∀ n ≥ N, 
      abs (angle O (An n)) - (π / 3) < ε ∧
      abs (angle O (Bn n)) - (π / 3) < ε ∧
      abs (angle O (Cn n)) - (π / 3) < ε :=
sorry

end triangle_converges_to_equilateral_l530_530220


namespace karan_borrowed_amount_l530_530532

theorem karan_borrowed_amount :
  ∃ P : ℝ, P > 0 ∧ (let R := 6 / 100 in let T := 9 in let A := 8210 in A = P + P * R * T) ∧ P ≈ 5331.17 :=
by
  use 8210 / 1.54
  split
  . linarith
  sorry

end karan_borrowed_amount_l530_530532


namespace find_length_of_side_a_l530_530287

variable {A : ℝ} -- angle A in radians
variable {b c a : ℝ} -- lengths of sides b, c, a
variable {S : ℝ} -- area of the triangle

def deg_to_rad (d : ℝ) := d * (real.pi / 180)

theorem find_length_of_side_a 
  (hA : A = deg_to_rad 60)
  (hb : b = 8)
  (hS : S = 12 * real.sqrt 3) :
  a = 2 * real.sqrt 13 := by
{
  -- Implement the proof here
  sorry
}

end find_length_of_side_a_l530_530287


namespace brick_length_is_50_l530_530112

theorem brick_length_is_50
  (x : ℝ)
  (brick_volume_eq : x * 11.25 * 6 * 3200 = 800 * 600 * 22.5) :
  x = 50 :=
by
  sorry

end brick_length_is_50_l530_530112


namespace diagonals_in_30_sided_polygon_l530_530819

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem diagonals_in_30_sided_polygon :
  number_of_diagonals 30 = 405 :=
by sorry

end diagonals_in_30_sided_polygon_l530_530819


namespace storage_room_solution_l530_530306

noncomputable def storage_room_problem : Prop :=
  let sugar := 6000 -- pounds of sugar
  let flour := 2 / 5 * sugar -- flour is 2/5 of sugar
  let baking_soda := (2400 / 8 - 60) -- given flour / (x + 60) = 8/1
  (flour / baking_soda) = 10

theorem storage_room_solution : storage_room_problem :=
by
  apply eq.refl

end storage_room_solution_l530_530306


namespace find_150th_letter_l530_530454
open Nat

def repeating_sequence := "XYZ"

def length_repeating_sequence := 3

theorem find_150th_letter : (150 % length_repeating_sequence == 0) → repeating_sequence[(length_repeating_sequence - 1) % length_repeating_sequence] = 'Z' := 
by
  sorry

end find_150th_letter_l530_530454


namespace sin_180_degrees_l530_530639

theorem sin_180_degrees : Real.sin (180 * Real.pi / 180) = 0 := 
by
  sorry

end sin_180_degrees_l530_530639


namespace eval_expression_l530_530669

theorem eval_expression : (2^2 - 2) - (3^2 - 3) + (4^2 - 4) - (5^2 - 5) + (6^2 - 6) = 18 :=
by
  sorry

end eval_expression_l530_530669


namespace find_value_of_f_6_l530_530165

def f : ℕ → ℤ
| 1       := 1
| 2       := 2
| (n+3)   := f (n+2) - f (n+1) + (n+3)^2

theorem find_value_of_f_6 : f 6 = 51 := by
  sorry

end find_value_of_f_6_l530_530165


namespace find_integer_l530_530180

theorem find_integer (n : ℤ) (h1 : 10 ≤ n) (h2 : n ≤ 15) (h3 : n ≡ 12345 [MOD 7]) : n = 11 :=
sorry

end find_integer_l530_530180


namespace jack_page_count_l530_530313

theorem jack_page_count (pages_per_booklet : ℕ) (number_of_booklets : ℕ) 
  (h1 : pages_per_booklet = 13) (h2 : number_of_booklets = 67) : 
  pages_per_booklet * number_of_booklets = 871 :=
by {
  rw [h1, h2],
  exact Nat.mul_eq_mul_left_iff.2 (or.intro_left _ rfl),
}

end jack_page_count_l530_530313


namespace isosceles_triangle_base_angle_l530_530905

theorem isosceles_triangle_base_angle (A B C : ℝ) (h_sum : A + B + C = 180) (h_iso : B = C) (h_one_angle : A = 80) : B = 50 :=
sorry

end isosceles_triangle_base_angle_l530_530905


namespace repeating_decimal_fraction_sum_is_839_l530_530401

noncomputable def repeating_decimal_to_fraction (x : ℝ) (hx : x = 7.47474747) : ℚ :=
  let n := 740
  let d := 99
  have h : 99*x = 740 := sorry
  have fraction : ℚ := ⟨n, d, sorry⟩  -- Here, ⟨n, d, sorry⟩ is a numerator-over-denominator fraction representation.
  fraction

theorem repeating_decimal_fraction_sum_is_839 :
  let sum_num_denom : ℚ := repeating_decimal_to_fraction 7.47474747 rfl 
  sum_num_denom.num + sum_num_denom.denom = 839 :=
sorry

end repeating_decimal_fraction_sum_is_839_l530_530401


namespace find_a_and_b_find_m_l530_530856

-- Define the main problem conditions and the variables
def parabola_condition_1 (a b : ℝ) : Prop :=
  16 * a - 4 * b - 4 = 0

-- The axis of symmetry condition
def parabola_condition_2 (a b : ℝ) : Prop :=
  - b / (2 * a) = -1

-- Point B, where x = 0
def point_B (a b : ℝ) : ℝ :=
  a * 0^2 + b * 0 - 4

-- Line AB moved down by m units intersects the parabola at one point condition
def line_parabola_intersect_once (a b m : ℝ) : Prop :=
  (2)^2 - 4 * (1/2) * m = 0

-- Prove that given the conditions, a and b have specific values
theorem find_a_and_b : ∃ a b : ℝ, parabola_condition_1 a b ∧ parabola_condition_2 a b ∧ a = 1/2 ∧ b = 1 :=
by {
  use (1/2), 
  use 1,
  unfold parabola_condition_1 parabola_condition_2,
  simp,
  sorry
}

-- Prove the value of m
theorem find_m (a b : ℝ) (h₁ : a = 1/2) (h₂ : b = 1) : ∃ m > 0, line_parabola_intersect_once a b m ∧ m = 2 :=
by {
  use 2,
  split,
  {
    norm_num,
  },
  {
    unfold line_parabola_intersect_once,
    simp [h₁, h₂],
    sorry
  }
}

end find_a_and_b_find_m_l530_530856


namespace range_of_m_for_increasing_f_l530_530249

noncomputable def f (m x : ℝ) : ℝ := (1 / 3) * x^3 - m * x^2 - 3 * m^2 * x + 1

theorem range_of_m_for_increasing_f :
  { m : ℝ | ∀ x ∈ set.Ioo 1 2, f m x > f m (x - 1) } = set.Icc (-1) (1 / 3) :=
sorry

end range_of_m_for_increasing_f_l530_530249


namespace function_with_real_domain_and_range_l530_530233

variable (a : ℝ)
def f_A (x : ℝ) : ℝ := x^2 + a
def f_B (x : ℝ) : ℝ := a * x^2 + 1
def f_C (x : ℝ) : ℝ := a * x^2 + x + 1
def f_D (x : ℝ) : ℝ := x^2 + a * x + 1

theorem function_with_real_domain_and_range :
  (f_C(a) = λ x : ℝ, a * x^2 + x + 1) ∧ ((∀ x : ℝ, x ≥ 0 → ∀ y : ℝ, x^2 + a = y → y ≠ ℝ) ∧ 
  (∀ x : ℝ, x ≥ 0 → ∀ y : ℝ, a * x^2 + 1 = y → y ≠ ℝ) ∧ 
  (∀ x : ℝ, x ≥ 0 → ∀ y : ℝ, x^2 + a * x + 1 = y → y ≠ ℝ)) → 
  (∃ g : ℝ → ℝ, ∀ x y : ℝ, (g x = g y ↔ x = y) ∧ (g x = f_C a x)) :=
by
  sorry

end function_with_real_domain_and_range_l530_530233


namespace area_of_right_triangle_with_30_deg_l530_530949

-- definitions and given conditions
def hypotenuse : Real := 12
def angle : Real := 30
def right_triangle (a b c : Real) : Prop := a^2 + b^2 = c^2

-- properties specific to our 30-60-90 triangle
def is_30_60_90_triangle (short long hyp : Real) : Prop :=
  long = short * Real.sqrt 3 ∧ hyp = short * 2

-- the triangle's base and height
def base : Real := 6
def height : Real := base * Real.sqrt 3

-- the area of the triangle
def area (b h : Real) : Real := 1 / 2 * b * h

-- the main theorem we want to prove
theorem area_of_right_triangle_with_30_deg : right_triangle base height hypotenuse ∧ angle = 30 → area base height = 18 * Real.sqrt 3 :=
by
  sorry

end area_of_right_triangle_with_30_deg_l530_530949


namespace boundary_value_problem_solution_l530_530438

open Real

def y (x : ℝ) : ℝ := (sinh x) / (sinh 1) - x

theorem boundary_value_problem_solution :
  (∀ (x: ℝ), (deriv^[2] y x) - y x = x) ∧ (y 0 = 0) ∧ (y 1 = 0) :=
by
  -- Proof skipped as per the guidelines
  sorry

end boundary_value_problem_solution_l530_530438


namespace determine_c_l530_530606

theorem determine_c (c d : ℝ) (hc : c < 0) (hd : d > 0) (hamp : ∀ x, y = c * Real.cos (d * x) → |y| ≤ 3) :
  c = -3 :=
sorry

end determine_c_l530_530606


namespace proposition_1_proposition_2_proposition_3_proposition_4_only_correct_proposition_l530_530592

section

variable (f : ℝ → ℝ)

def even_function := ∀ x : ℝ, f (-x) = f x
def odd_function := ∀ x : ℝ, f (-x) = -f x

theorem proposition_1 : ¬(∀ f : ℝ → ℝ, even_function f → ∃ y, ∃ x, f y = x * 1) := sorry
theorem proposition_2 : ¬(∀ f : ℝ → ℝ, odd_function f → f 0 = 0) := sorry
theorem proposition_3 : (∀ f : ℝ → ℝ, even_function f → ∀ x, f x = f (-x)) := sorry
theorem proposition_4 : ¬(∀ f : ℝ → ℝ, (even_function f ∧ odd_function f) → ∀ x, f x = 0) := sorry

theorem only_correct_proposition :
  proposition_1 = False ∧ proposition_2 = False ∧ proposition_3 = True ∧ proposition_4 = False :=
  by 
  sorry

end

end proposition_1_proposition_2_proposition_3_proposition_4_only_correct_proposition_l530_530592


namespace probability_second_student_male_l530_530575

variable {α : Type} [Fintype α] [DecidableEq α]

/-- There are 2 male students (m) and 2 female students (f). -/
def students : Finset α := {1, 2, 3, 4}

/-- The probability that the second student to leave is a male student is 1/6. -/
theorem probability_second_student_male (m1 m2 f1 f2 : α) (hm : m1 ≠ m2) (hf : f1 ≠ f2)
  (h : {m1, m2, f1, f2} = students) :
  (∃ seq : List α, seq.perm [m1, m2, f1, f2] ∧ seq.nth 1 = some m1 ∨ seq.nth 1 = some m2) ->
  nat.cast (#(students.filter (λ x, x = m1 ∨ x = m2))) / nat.cast (students.card) = (1 : ℚ) / 6 :=
by
  intro h_seq
  sorry

end probability_second_student_male_l530_530575


namespace trisect_angle_in_regular_pentagon_ratio_PT_area_triangle_XSR_l530_530089

-- Definition of trisection of an angle in a regular pentagon
theorem trisect_angle_in_regular_pentagon (P Q R S T : Point) (hreg: RegularPentagon P Q R S T) :
  ∃ A B C, ∠ Q P R = 36 ∧ ∠ R P S = 36 ∧ ∠ S P Q = 36 :=
sorry

-- Ratio after folding a regular pentagon
theorem ratio_PT'_TR' (P Q R S T T' : Point) (hreg: RegularPentagon P Q R S T) (hfold: FoldedAlongDiagonal P S T T') :
  ∃ a b c : ℕ, PT'/T'R = (a + √b)/c ∧ (a = 0 ∧ b = 5 ∧ c = 2) :=
sorry

-- Area of triangle XSR after folding in a regular pentagon
theorem area_triangle_XSR (P Q R S T T' Q' X : Point) (hreg: RegularPentagon P Q R S T) (hfold1: FoldedAlongDiagonal P S T T') (hfold2: FoldedAlongDiagonal P R Q Q') :
  ∃ a b c : ℕ, area X S R = (a + √b)/c ∧ (a = 1 ∧ b = 5 ∧ c = 6) :=
sorry

end trisect_angle_in_regular_pentagon_ratio_PT_area_triangle_XSR_l530_530089


namespace john_pays_2010_dollars_l530_530873

-- Define the main problem as the number of ways to pay 2010$ using 2, 5, and 10$ notes.
theorem john_pays_2010_dollars :
  ∃ (count : ℕ), count = 20503 ∧
  ∀ (x y z : ℕ), (2 * x + 5 * y + 10 * z = 2010) → (x % 5 = 0) → (y % 2 = 0) → count = 20503 :=
by sorry

end john_pays_2010_dollars_l530_530873


namespace letter_at_position_150_l530_530477

theorem letter_at_position_150 : 
  (∀ n, n > 0 → ∃ i, i ∈ {1, 2, 3} ∧ "XYZ".to_list[i-1] = "XYZ".to_list[(n - 1) % 3]) →
  ("XYZ".to_list[(150 - 1) % 3] = 'Z') :=
by
  sorry

end letter_at_position_150_l530_530477


namespace total_arrangements_l530_530560

-- Define the conditions
def volunteers : ℕ := 3
def elderly_people : ℕ := 2

-- Define the constraints
def elderly_next_to_each_other (row : List ℕ) : Prop :=
  ∃ i, row.nth i = some 1 ∧ row.nth (i + 1) = some 2 ∨ row.nth i = some 2 ∧ row.nth (i + 1) = some 1

def elderly_not_at_ends (row : List ℕ) : Prop :=
  row.head ≠ some 1 ∧ row.head ≠ some 2 ∧ row.last ≠ some 1 ∧ row.last ≠ some 2

-- Define the arrangement of volunteers and elderly people
def arrangement := List.range (volunteers + elderly_people)

-- Main theorem statement
theorem total_arrangements : 
  (∃ (row : List ℕ), row.perm arrangement ∧ elderly_next_to_each_other row ∧ elderly_not_at_ends row) → 
  24 :=
by
  sorry

end total_arrangements_l530_530560


namespace eval_f_at_log5_l530_530225

noncomputable def f : ℝ → ℝ := sorry

theorem eval_f_at_log5 :
  (∀ x, f x = f (-x)) →
  (∀ x, f (x + 1) = f (x - 1)) →
  (∀ x, x ∈ set.Icc (-1:ℝ) 0 → f x = 3^x + 1) →
  f (Real.log 5) = 4 / 3 :=
sorry

end eval_f_at_log5_l530_530225


namespace fewer_puzzles_sold_l530_530007

theorem fewer_puzzles_sold (a b : ℕ) (h1 : a = 45) (h2 : b = 36) : a - b = 9 :=
by
  rw [h1, h2]
  norm_num
  sorry

end fewer_puzzles_sold_l530_530007


namespace fry_sausage_time_l530_530321

variable (time_per_sausage : ℕ)

noncomputable def time_for_sausages (sausages : ℕ) (tps : ℕ) : ℕ :=
  sausages * tps

noncomputable def time_for_eggs (eggs : ℕ) (minutes_per_egg : ℕ) : ℕ :=
  eggs * minutes_per_egg

noncomputable def total_time (time_sausages : ℕ) (time_eggs : ℕ) : ℕ :=
  time_sausages + time_eggs

theorem fry_sausage_time :
  let sausages := 3
  let eggs := 6
  let minutes_per_egg := 4
  let total_time_taken := 39
  total_time (time_for_sausages sausages time_per_sausage) (time_for_eggs eggs minutes_per_egg) = total_time_taken
  → time_per_sausage = 5 := by
  sorry

end fry_sausage_time_l530_530321


namespace sum_floor_parity_l530_530212

theorem sum_floor_parity (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  (∑ i in Finset.range (m * n), (-1) ^ (i / m + i / n) = 0) ↔ ((m % 2 = 0 ∧ n % 2 = 1) ∨ (m % 2 = 1 ∧ n % 2 = 0)) :=
by sorry

end sum_floor_parity_l530_530212


namespace largest_n_product_consecutive_integers_l530_530197

theorem largest_n_product_consecutive_integers : ∃ (n : ℕ), (∀ (x : ℕ), n! = (list.Ico x (x + n - 4)).prod) ∧ n = 4 := 
by
  sorry

end largest_n_product_consecutive_integers_l530_530197


namespace log2_T_equals_1005_l530_530325

-- Define T as the sum of all the real coefficients of (1 + ix)^2011
noncomputable def T : ℂ := (1 : ℂ) + Complex.I * x

-- Theorem statement
theorem log2_T_equals_1005 : ∀ x : ℝ, log 2 (T.re) = 1005 := by
  sorry

end log2_T_equals_1005_l530_530325


namespace min_cost_proof_l530_530920

-- Define the costs and servings for each ingredient
def pasta_cost : ℝ := 1.12
def pasta_servings_per_box : ℕ := 5

def meatballs_cost : ℝ := 5.24
def meatballs_servings_per_pack : ℕ := 4

def tomato_sauce_cost : ℝ := 2.31
def tomato_sauce_servings_per_jar : ℕ := 5

def tomatoes_cost : ℝ := 1.47
def tomatoes_servings_per_pack : ℕ := 4

def lettuce_cost : ℝ := 0.97
def lettuce_servings_per_head : ℕ := 6

def olives_cost : ℝ := 2.10
def olives_servings_per_jar : ℕ := 8

def cheese_cost : ℝ := 2.70
def cheese_servings_per_block : ℕ := 7

-- Define the number of people to serve
def number_of_people : ℕ := 8

-- The total cost calculated
def total_cost : ℝ := 
  (2 * pasta_cost) +
  (2 * meatballs_cost) +
  (2 * tomato_sauce_cost) +
  (2 * tomatoes_cost) +
  (2 * lettuce_cost) +
  (1 * olives_cost) +
  (2 * cheese_cost)

-- The minimum total cost
def min_total_cost : ℝ := 29.72

theorem min_cost_proof : total_cost = min_total_cost :=
by sorry

end min_cost_proof_l530_530920


namespace reflex_angle_at_G_correct_l530_530686

noncomputable def reflex_angle_at_G
    (B A E L G : Type)
    (on_line : B = A ∨ A = E ∨ E = L) 
    (off_line : ¬(G = B ∨ G = A ∨ G = E ∨ G = L))
    (angle_BAG : ℝ)
    (angle_GEL : ℝ)
    (h1 : angle_BAG = 120)
    (h2 : angle_GEL = 80)
    : ℝ :=
  360 - (180 - (180 - angle_BAG) - (180 - angle_GEL))

theorem reflex_angle_at_G_correct :
    (∀ (B A E L G : Type)
    (on_line : B = A ∨ A = E ∨ E = L) 
    (off_line : ¬(G = B ∨ G = A ∨ G = E ∨ G = L))
    (angle_BAG : ℝ)
    (angle_GEL : ℝ)
    (h1 : angle_BAG = 120)
    (h2 : angle_GEL = 80),
    reflex_angle_at_G B A E L G on_line off_line angle_BAG angle_GEL h1 h2 = 340) := sorry

end reflex_angle_at_G_correct_l530_530686


namespace convex_polygon_diagonals_l530_530814

theorem convex_polygon_diagonals (n : ℕ) (h : n = 30) : 
  let V := n in
  let total_pairs := V * (V - 1) / 2 in
  let adjacent_pairs := n in
  let diagonals := total_pairs - adjacent_pairs in
  diagonals = 405 :=
  by
  -- number of sides is 30
  have hn : n = 30 := h
  -- calculate the total pairs
  have total_pairs_calc : total_pairs = 30 * 29 / 2 := by sorry
  -- calculate the adjacent pairs
  have adjacent_pairs_calc : adjacent_pairs = 30 := by sorry
  -- calculate the diagonals
  have diagonals_calc : diagonals = (30 * 29 / 2) - 30 := by sorry
  -- proved statement
  show 405 = 405, by rfl

end convex_polygon_diagonals_l530_530814


namespace cars_cost_between_15000_and_20000_l530_530150

theorem cars_cost_between_15000_and_20000 (total_cars : ℕ) (p1 p2 : ℕ) :
    total_cars = 3000 → 
    p1 = 15 → 
    p2 = 40 → 
    (p1 * total_cars / 100 + p2 * total_cars / 100 + x = total_cars) → 
    x = 1350 :=
by
  intro h_total
  intro h_p1
  intro h_p2
  intro h_eq
  sorry

end cars_cost_between_15000_and_20000_l530_530150


namespace convex_polygon_diagonals_l530_530816

theorem convex_polygon_diagonals (n : ℕ) (h : n = 30) : 
  let V := n in
  let total_pairs := V * (V - 1) / 2 in
  let adjacent_pairs := n in
  let diagonals := total_pairs - adjacent_pairs in
  diagonals = 405 :=
  by
  -- number of sides is 30
  have hn : n = 30 := h
  -- calculate the total pairs
  have total_pairs_calc : total_pairs = 30 * 29 / 2 := by sorry
  -- calculate the adjacent pairs
  have adjacent_pairs_calc : adjacent_pairs = 30 := by sorry
  -- calculate the diagonals
  have diagonals_calc : diagonals = (30 * 29 / 2) - 30 := by sorry
  -- proved statement
  show 405 = 405, by rfl

end convex_polygon_diagonals_l530_530816


namespace range_of_a_l530_530841

-- Definitions for the conditions: 'a > 0' and curves having a common point.
def has_common_point (a : ℝ) : Prop :=
  ∃ x : ℝ, (0 < x) ∧ (e^x = a * x^2)

-- Theorem statement: Prove that if a > 0 and the curves have common points, then a >= e^2 / 4.
theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : has_common_point a) : a ≥ e^2 / 4 :=
  sorry

end range_of_a_l530_530841


namespace direction_reversal_l530_530291

def opposite_direction (direction : String) : String :=
  match direction with
  | "south by east" => "north by west"
  | _ => "undefined"

theorem direction_reversal :
  ∀ (A B : Type) (d : String), (d = "south by east") → (opposite_direction d = "north by west") :=
by {
  intros A B d h,
  simp [opposite_direction],
  exact h
}

end direction_reversal_l530_530291


namespace number_of_diagonals_in_convex_polygon_l530_530752

/-- 
A theorem stating that the number of diagonals 
in a convex polygon with 30 sides is equal to 405.
-/
theorem number_of_diagonals_in_convex_polygon (n : ℕ) (h : n = 30) (convex : True) : (n * (n - 3)) / 2 = 405 := 
by 
  rw h
  norm_num
  done

end number_of_diagonals_in_convex_polygon_l530_530752


namespace sine_180_eq_zero_l530_530626

theorem sine_180_eq_zero :
  sin (180 : ℝ) = 0 :=
sorry

end sine_180_eq_zero_l530_530626


namespace find_x_value_l530_530029

theorem find_x_value (x : ℝ) (hx : x > 0) : 
  x^{log (x) / log 10} = x^4 / 10000 → x = 100 :=
by
  sorry

end find_x_value_l530_530029


namespace textile_firm_looms_l530_530136

theorem textile_firm_looms
  (sales_val : ℝ)
  (manu_exp : ℝ)
  (estab_charges : ℝ)
  (profit_decrease : ℝ)
  (L : ℝ)
  (h_sales : sales_val = 500000)
  (h_manu_exp : manu_exp = 150000)
  (h_estab_charges : estab_charges = 75000)
  (h_profit_decrease : profit_decrease = 7000)
  (hem_equal_contrib : ∀ l : ℝ, l > 0 →
    (l = sales_val / (sales_val / L) - manu_exp / (manu_exp / L)))
  : L = 50 := 
by
  sorry

end textile_firm_looms_l530_530136


namespace smallest_cards_guarantee_trick_l530_530534

theorem smallest_cards_guarantee_trick (colors : ℕ) (n : ℕ) (h_colors : colors = 2017) :
  (n > 1) → (n = colors + 1) → (∃ strat : list (ℕ → ℕ), ∀ table : list ℕ,
  length table = n → ∃ flip : list ℕ, (flip.length = n - 1) ∧ 
  (∃ guess : ℕ, guess ∈ table ∧ ∀ color, guess = color)) :=
by
  sorry

noncomputable def predefined_strategy : list (ℕ → ℕ) := sorry

end smallest_cards_guarantee_trick_l530_530534


namespace nonWhiteHomesWithoutFireplace_l530_530413

-- Definitions based on the conditions
def totalHomes : ℕ := 400
def whiteHomes (h : ℕ) : ℕ := h / 4
def nonWhiteHomes (h w : ℕ) : ℕ := h - w
def nonWhiteHomesWithFireplace (nh : ℕ) : ℕ := nh / 5

-- Theorem statement to prove the required result
theorem nonWhiteHomesWithoutFireplace : 
  let h := totalHomes
  let w := whiteHomes h
  let nh := nonWhiteHomes h w
  let nf := nonWhiteHomesWithFireplace nh
  nh - nf = 240 :=
by
  let h := totalHomes
  let w := whiteHomes h
  let nh := nonWhiteHomes h w
  let nf := nonWhiteHomesWithFireplace nh
  show nh - nf = 240
  sorry

end nonWhiteHomesWithoutFireplace_l530_530413


namespace range_of_a_l530_530410

noncomputable def p : set ℝ := {x | x < 1} ∪ {x | 2 < x}
noncomputable def q (a : ℝ) : set ℝ := {x | (x - 1) * (x + a) > 0}

theorem range_of_a {a : ℝ} : (∀ x, x ∈ p → x ∈ q a) ∧ ¬ (∀ x, x ∈ q a → x ∈ p) ↔ a ∈ set.Ioc (-2) (-1) := by
  sorry

end range_of_a_l530_530410


namespace x_eq_3_minus_2t_and_y_eq_3t_plus_6_l530_530002

theorem x_eq_3_minus_2t_and_y_eq_3t_plus_6 (t : ℝ) (x : ℝ) (y : ℝ) : x = 3 - 2 * t → y = 3 * t + 6 → x = 0 → y = 10.5 :=
by
  sorry

end x_eq_3_minus_2t_and_y_eq_3t_plus_6_l530_530002


namespace letter_at_position_150_l530_530480

theorem letter_at_position_150 : 
  (∀ n, n > 0 → ∃ i, i ∈ {1, 2, 3} ∧ "XYZ".to_list[i-1] = "XYZ".to_list[(n - 1) % 3]) →
  ("XYZ".to_list[(150 - 1) % 3] = 'Z') :=
by
  sorry

end letter_at_position_150_l530_530480


namespace max_value_of_f_min_value_of_f_l530_530199

noncomputable def f (x : ℝ) : ℝ := x^3 - 12 * x

def max_value_of_f_on_interval : ℝ := 16
def min_value_of_f_on_interval : ℝ := -16

theorem max_value_of_f (h₁ : -3 ≤ 3) : 
  ∃ x ∈ set.Icc (-3 : ℝ) 3, f x = max_value_of_f_on_interval := sorry

theorem min_value_of_f (h₂ : -3 ≤ 3) : 
  ∃ x ∈ set.Icc (-3 : ℝ) 3, f x = min_value_of_f_on_interval := sorry

end max_value_of_f_min_value_of_f_l530_530199


namespace convex_polygon_diagonals_l530_530813

theorem convex_polygon_diagonals (n : ℕ) (h : n = 30) : 
  let V := n in
  let total_pairs := V * (V - 1) / 2 in
  let adjacent_pairs := n in
  let diagonals := total_pairs - adjacent_pairs in
  diagonals = 405 :=
  by
  -- number of sides is 30
  have hn : n = 30 := h
  -- calculate the total pairs
  have total_pairs_calc : total_pairs = 30 * 29 / 2 := by sorry
  -- calculate the adjacent pairs
  have adjacent_pairs_calc : adjacent_pairs = 30 := by sorry
  -- calculate the diagonals
  have diagonals_calc : diagonals = (30 * 29 / 2) - 30 := by sorry
  -- proved statement
  show 405 = 405, by rfl

end convex_polygon_diagonals_l530_530813


namespace rhombus_area_l530_530283

theorem rhombus_area (a b : ℝ) (h : (a - 1) ^ 2 + Real.sqrt (b - 4) = 0) : (1 / 2) * a * b = 2 := by
  sorry

end rhombus_area_l530_530283


namespace is_increasing_function_on_R_l530_530019

theorem is_increasing_function_on_R : ∀ x : ℝ, (1 - cos x) ≥ 0 :=
by
  intro x
  have h_cos_bounds : -1 ≤ cos x ∧ cos x ≤ 1 := ⟨cos_bounded_below x, cos_bounded_above x⟩
  sorry

end is_increasing_function_on_R_l530_530019


namespace candy_problem_l530_530349

theorem candy_problem (N S a : ℕ) (h1 : a = S - a - 7) (h2 : a > 1) : S = 21 := 
sorry

end candy_problem_l530_530349


namespace arrange_abc_l530_530888

def a : ℝ := Real.log 2
def b : ℝ := Real.sqrt 2
def c : ℝ := Real.cos (3 * Real.pi / 4)

theorem arrange_abc : c < a ∧ a < b := by
  sorry

end arrange_abc_l530_530888


namespace length_of_platform_l530_530121

theorem length_of_platform {v : ℝ} {t : ℝ} {lₜ : ℝ} : 
  v = 72 * 5 / 18 → 
  t = 26 → 
  lₜ = 270 →
  ∃ lₚ, lₚ + lₜ = v * t ∧ lₚ = 250 :=
by
  intros hv ht hlₜ
  have h₁ : v = 20 := by sorry
  have h₂ : t = 26 := ht
  have h₃ : lₜ = 270 := hlₜ
  have h₄ : 520 = 20 * 26 := by sorry
  use 250
  split
  . rw [←h₃, h₄]; sorry
  . exact by sorry

end length_of_platform_l530_530121


namespace letter_150th_in_pattern_l530_530491

def repeating_sequence := "XYZ"

def letter_at_position (n : ℕ) : char :=
  let seq := repeating_sequence.to_list
  seq.get! ((n - 1) % seq.length)

theorem letter_150th_in_pattern : letter_at_position 150 = 'Z' :=
by sorry

end letter_150th_in_pattern_l530_530491


namespace least_num_subtracted_l530_530080

theorem least_num_subtracted 
  (n : ℕ) 
  (h1 : n = 642) 
  (rem_cond : ∀ k, (k = 638) → n - k = 4): 
  n - 638 = 4 := 
by sorry

end least_num_subtracted_l530_530080


namespace find_150th_letter_l530_530453
open Nat

def repeating_sequence := "XYZ"

def length_repeating_sequence := 3

theorem find_150th_letter : (150 % length_repeating_sequence == 0) → repeating_sequence[(length_repeating_sequence - 1) % length_repeating_sequence] = 'Z' := 
by
  sorry

end find_150th_letter_l530_530453


namespace other_root_of_quadratic_l530_530365

theorem other_root_of_quadratic :
  ∀ {c : ℝ}, (6 * (-1 / 2) ^ 2 + c * (-1 / 2) = -3) → ∃ q : ℝ, (6 * q ^ 2 + c * q + 3 = 0) ∧ q = -1 :=
by
  intros c h
  have h1 : 6 * q ^ 2 + c * q + 3 = 0 := sorry
  use -1
  simp [h1]
  sorry

end other_root_of_quadratic_l530_530365


namespace dino_remaining_balance_is_4650_l530_530665

def gigA_hours : Nat := 20
def gigA_rate : Nat := 10

def gigB_hours : Nat := 30
def gigB_rate : Nat := 20

def gigC_hours : Nat := 5
def gigC_rate : Nat := 40

def gigD_hours : Nat := 15
def gigD_rate : Nat := 25

def gigE_hours : Nat := 10
def gigE_rate : Nat := 30

def january_expense : Nat := 500
def february_expense : Nat := 550
def march_expense : Nat := 520
def april_expense : Nat := 480

theorem dino_remaining_balance_is_4650 :
  let gigA_earnings := gigA_hours * gigA_rate
  let gigB_earnings := gigB_hours * gigB_rate
  let gigC_earnings := gigC_hours * gigC_rate
  let gigD_earnings := gigD_hours * gigD_rate
  let gigE_earnings := gigE_hours * gigE_rate

  let total_monthly_earnings := gigA_earnings + gigB_earnings + gigC_earnings + gigD_earnings + gigE_earnings

  let total_expenses := january_expense + february_expense + march_expense + april_expense

  let total_earnings_four_months := total_monthly_earnings * 4

  total_earnings_four_months - total_expenses = 4650 :=
by {
  sorry
}

end dino_remaining_balance_is_4650_l530_530665


namespace find_150th_letter_l530_530464

theorem find_150th_letter :
  let pattern := ['X', 'Y', 'Z']
  150 % 3 = 0 -> pattern[(150 % 3 + 2) % 3] = 'Z' :=
begin
  intros pattern h,
  simp at *,
  exact rfl,
end

end find_150th_letter_l530_530464


namespace find_middle_number_l530_530412

namespace Problem

-- Define the three numbers x, y, z
variables (x y z : ℕ)

-- Given conditions from the problem
def condition1 (h1 : x + y = 18) := x + y = 18
def condition2 (h2 : x + z = 23) := x + z = 23
def condition3 (h3 : y + z = 27) := y + z = 27
def condition4 (h4 : x < y ∧ y < z) := x < y ∧ y < z

-- Statement to prove:
theorem find_middle_number (h1 : x + y = 18) (h2 : x + z = 23) (h3 : y + z = 27) (h4 : x < y ∧ y < z) : 
  y = 11 :=
by
  sorry

end Problem

end find_middle_number_l530_530412


namespace smallest_common_multiple_of_8_and_6_l530_530519

theorem smallest_common_multiple_of_8_and_6 : ∃ n : ℕ, n > 0 ∧ (8 ∣ n) ∧ (6 ∣ n) ∧ ∀ m : ℕ, (m > 0 ∧ (8 ∣ m) ∧ (6 ∣ m)) → n ≤ m :=
by
  sorry

end smallest_common_multiple_of_8_and_6_l530_530519


namespace circumcenter_area_comparison_l530_530878

theorem circumcenter_area_comparison 
  (A B C M : Point) 
  (angle_AMB : ∠ A M B = 150)
  (angle_BMC : ∠ B M C = 120)
  (circumcenter_AMB : Point)
  (circumcenter_BMC : Point)
  (circumcenter_CMA : Point)
  (P : circumcenter_AMB = circumcenter (triangle A M B))
  (Q : circumcenter_BMC = circumcenter (triangle B M C))
  (R : circumcenter_CMA = circumcenter (triangle C M A)) :
  area (triangle circumcenter_AMB circumcenter_BMC circumcenter_CMA) ≥ area (triangle A B C) :=
by
  sorry

end circumcenter_area_comparison_l530_530878


namespace intersecting_chords_theorem_l530_530013

-- Definitions for conditions given in the problem
variables {O P A B C D : Type}
variables [MetricSpace O] [MetricSpace P] [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
-- P is the midpoint of AB
def midpoint (P A B : Type) := dist A P = dist B P ∧ dist A P + dist B P = dist A B

-- Main theorem statement
theorem intersecting_chords_theorem (h_midpoint : midpoint P A B) 
                                   (h_AB_length : dist A B = 12)
                                   (h_PC_length : dist P C = 4)
                                   : ∃ (PD : ℝ), PD = 9 :=
by {
  sorry
}

end intersecting_chords_theorem_l530_530013


namespace emails_in_evening_l530_530314

theorem emails_in_evening (emails_morning emails_afternoon emails_evening emails_morning_and_evening : ℕ)
  (h_morning : emails_morning = 3)
  (h_afternoon : emails_afternoon = 4)
  (h_morning_and_evening : emails_morning_and_evening = 11):
  emails_evening = 8 :=
by
  -- 3 emails in the morning
  have h_morning : emails_morning = 3 := by assumption
  -- 11 emails are received in the morning and evening combined
  have h_morning_and_evening : emails_morning_and_evening = 11 := by assumption
  -- Evening emails calculation
  have h_evening : emails_evening = emails_morning_and_evening - emails_morning := by simp [h_morning, h_morning_and_evening]
  -- Prove emails in the evening is 8
  show emails_evening = 8 from by simp [h_evening]
  sorry

end emails_in_evening_l530_530314


namespace original_mean_proof_l530_530021

-- Define the initial conditions
variables {n : ℕ} (mean updated_mean decrement : ℝ)
def conditions : Prop := n = 50 ∧ updated_mean = 194 ∧ decrement = 6

-- Formalize the proof problem
theorem original_mean_proof (h : conditions n mean updated_mean decrement) : mean = 200 :=
by
  sorry

end original_mean_proof_l530_530021


namespace checkerboard_sum_l530_530114

def f (i j : ℕ) : ℕ := 13 * (i - 1) + j
def g (i j : ℕ) : ℕ := 11 * (j - 1) + i

theorem checkerboard_sum :
  ( ∑ i in {i | ∃ j, 1 ≤ i ∧ i ≤ 11 ∧ 1 ≤ j ∧ j ≤ 13 ∧ f i j = g i j}, ∑ j in {j | f i j = g i j}, f i j ) = 203 :=
by
  sorry

end checkerboard_sum_l530_530114


namespace letter_150_is_Z_l530_530502

/-- Definition of the repeating pattern "XYZ" -/
def pattern : List Char := ['X', 'Y', 'Z']

/-- The repeating pattern has a length of 3 -/
def pattern_length : ℕ := 3

/-- Calculate the 150th letter in the repeating pattern "XYZ" -/
def nth_letter_in_pattern (n : ℕ) : Char :=
  let m := n % pattern_length
  if m = 0 then pattern[2] else pattern[m - 1]

/-- Prove that the 150th letter in the pattern "XYZ" is 'Z' -/
theorem letter_150_is_Z : nth_letter_in_pattern 150 = 'Z' :=
by
  sorry

end letter_150_is_Z_l530_530502


namespace diagonal_count_of_convex_polygon_30_sides_l530_530792
-- Importing the entire Mathlib library for necessary mathematical constructs

theorem diagonal_count_of_convex_polygon_30_sides : 
  let n := 30 in
  let num_diagonals := (n * (n - 3)) / 2 in
  num_diagonals = 405 := 
by {
  let n := 30
  let num_diagonals := (n * (n - 3)) / 2
  show num_diagonals = 405,
  sorry
}

end diagonal_count_of_convex_polygon_30_sides_l530_530792


namespace length_of_XY_l530_530303

theorem length_of_XY (OY_radius : ℝ) (OX_segment : ℝ)
(OA_OB_radius : ℝ) (angle_AO_B : ℝ) (cos_angle_OAX : ℝ) (sin_angle_OAX : ℝ) :
  OY_radius = 15 →
  OY_radius = OA_OB_radius →
  angle_AO_B = 45 →
  cos_angle_OAX ≈ 0.8321 →
  sin_angle_OAX ≈ 0.5547 →
  OX_segment = OA_OB_radius * sin_angle_OAX →
  ∃ XY : ℝ, XY = OY_radius - OX_segment ∧ XY ≈ 6.68 :=
by
  intros hOY hOA_OB hAO_B hcos_OAX hsin_OAX hOX
  use OY_radius - OX_segment
  split
  · sorry
  · exact hOY.symm ▸ hOX ▸ sorry

end length_of_XY_l530_530303


namespace alan_total_cost_is_84_l530_530577

noncomputable def price_AVN : ℝ := 12
noncomputable def multiplier : ℝ := 2
noncomputable def count_Dark : ℕ := 2
noncomputable def count_AVN : ℕ := 1
noncomputable def count_90s : ℕ := 5
noncomputable def percentage_90s : ℝ := 0.40

def main_theorem : Prop :=
  let price_Dark := price_AVN * multiplier in
  let total_cost_Dark := price_Dark * count_Dark in
  let total_cost_AVN := price_AVN * count_AVN in
  let total_cost_other := total_cost_Dark + total_cost_AVN in
  let cost_90s := percentage_90s * total_cost_other in
  let total_cost := total_cost_other + cost_90s in
  total_cost = 84

theorem alan_total_cost_is_84 : main_theorem :=
  sorry

end alan_total_cost_is_84_l530_530577


namespace diagonals_in_30_sided_polygon_l530_530826

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem diagonals_in_30_sided_polygon :
  number_of_diagonals 30 = 405 :=
by sorry

end diagonals_in_30_sided_polygon_l530_530826


namespace meaningful_sqrt_condition_l530_530048

theorem meaningful_sqrt_condition (x : ℝ) : (2 * x - 1 ≥ 0) ↔ (x ≥ 1 / 2) :=
by
  sorry

end meaningful_sqrt_condition_l530_530048


namespace most_stable_student_l530_530858

-- Define the variances for the four students
def variance_A (SA2 : ℝ) : Prop := SA2 = 0.15
def variance_B (SB2 : ℝ) : Prop := SB2 = 0.32
def variance_C (SC2 : ℝ) : Prop := SC2 = 0.5
def variance_D (SD2 : ℝ) : Prop := SD2 = 0.25

-- Theorem proving that the most stable student is A
theorem most_stable_student {SA2 SB2 SC2 SD2 : ℝ} 
  (hA : variance_A SA2) 
  (hB : variance_B SB2)
  (hC : variance_C SC2)
  (hD : variance_D SD2) : 
  SA2 < SB2 ∧ SA2 < SC2 ∧ SA2 < SD2 :=
by
  rw [variance_A, variance_B, variance_C, variance_D] at *
  sorry

end most_stable_student_l530_530858


namespace allen_reading_days_l530_530142

theorem allen_reading_days (pages_per_day : ℕ) (total_pages : ℕ) 
  (h1 : pages_per_day = 10) (h2 : total_pages = 120) : 
  (total_pages / pages_per_day) = 12 := by
  sorry

end allen_reading_days_l530_530142


namespace pages_of_dictionary_l530_530970

theorem pages_of_dictionary (total_ones : ℕ) (h : total_ones = 1988) : ∃ n : ℕ, n = 3152 ∧ -- total page count in dictionary
  (let count_ones_in_pos : ℕ → ℕ := λ x, -- function that counts how many times '1' appears in the number
    nat.digits 10 x |>.count 1 in
  list.sum (list.map count_ones_in_pos (list.range (n + 1))) = total_ones) :=
sorry -- Proof to be filled in

end pages_of_dictionary_l530_530970


namespace triangle_count_bound_construction_exists_l530_530690

theorem triangle_count_bound (n k : ℕ) (h_n_gt_3 : n > 3) (h_n_ge_k : ∃(k : ℕ), ∃(points : list (ℝ × ℝ)), points.length = n ∧ valid_convex_polygon points ∧ ∃(triangles : list (ℕ × ℕ × ℕ)), triangles.length = k ∧ (∀ t ∈ triangles, is_regular_triangle t points ∧ side_length_is_1 t points)) : k < (2 * n) / 3 :=
sorry

theorem construction_exists (n : ℕ) (h_n_gt_333 : n > 333) : ∃(points : list (ℝ × ℝ)), points.length = n ∧ valid_convex_polygon points ∧ ∃(triangles : list (ℕ × ℕ × ℕ)), triangles.length > 2 * n / 3 :=
sorry

-- Definitions for validity checks
def valid_convex_polygon (points : list (ℝ × ℝ)) : Prop :=
  sorry -- Define what makes a list of points a valid convex polygon

def is_regular_triangle (triangle : ℕ × ℕ × ℕ) (points : list (ℝ × ℝ)) : Prop :=
  sorry -- Define what makes a triple of indices form a regular triangle among points

def side_length_is_1 (triangle : ℕ × ℕ × ℕ) (points : list (ℝ × ℝ)) : Prop :=
  sorry -- Define the property that the side length of the triangle is 1

end triangle_count_bound_construction_exists_l530_530690


namespace problem_1_problem_2_l530_530744

-- Definition of the vectors and function
def vec_a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, Real.sin x)
def vec_b (x : ℝ) : ℝ × ℝ := (Real.cos x, 2 * Real.sqrt 3 * Real.cos x)
def f (x : ℝ) : ℝ := (vec_a x).fst * (vec_b x).fst + (vec_a x).snd * (vec_b x).snd + 1

-- Interval of monotonically increasing function
def increasing_intervals (k : ℤ) : Set ℝ := Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6)

-- Definition to capture the maximum area of the triangle
def triangle_area (a b c : ℝ) (A : ℝ) := (1 / 2) * b * c * Real.sin A

-- Formal statements of the problems
theorem problem_1 (k : ℤ) : 
  ∀ x : ℝ,
    (k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6) → 
    (∀ y : ℝ, y ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6) → 
      deriv f y > 0) :=
sorry  -- Proof omitted

theorem problem_2 (b c : ℝ) (A : ℝ) (h1 : 1 = b * c * Real.sin (A)) (h2 : 0 < A ∧ A < π) : 
  ∃ S : ℝ, S = triangle_area 1 b c (π / 6) ∧ S = sqrt 3 / 4 :=
sorry  -- Proof omitted

end problem_1_problem_2_l530_530744


namespace least_number_to_add_l530_530525

theorem least_number_to_add (n : ℕ) (d : ℕ) (h1 : n = 907223) (h2 : d = 577) : (d - (n % d) = 518) := 
by
  rw [h1, h2]
  sorry

end least_number_to_add_l530_530525


namespace find_omega_l530_530725

noncomputable def f (ω x : ℝ) := sqrt 3 * Real.sin (ω * x) + Real.cos (ω * x)

theorem find_omega (ω x₁ x₂ : ℝ) (h_ω : ω > 0) (hx₁ : f ω x₁ = -2) (hx₂ : f ω x₂ = 0) (h_dist : abs (x₁ - x₂) = π) : ω = 1 / 2 :=
sorry

end find_omega_l530_530725


namespace smallest_positive_period_f_max_min_f_interval_l530_530247

def f (x : ℝ) : ℝ :=
  1 - 2 * (Real.sin (x + π / 8)) * (Real.sin (x + π / 8) - Real.cos (x + π / 8))

theorem smallest_positive_period_f :
  ∃ T > 0, ∀ x, f (x + T) = f x := by
  existsi π
  intros
  sorry 

theorem max_min_f_interval :
  ∃ (max min : ℝ), max = √2 ∧ min = -1 ∧
  ∀ x ∈ Set.Icc (-(π / 2)) 0, f (x + π / 8) ≤ max ∧ f (x + π / 8) ≥ min := by
  existsi √2
  existsi -1
  intros
  sorry

end smallest_positive_period_f_max_min_f_interval_l530_530247


namespace minimum_yellow_marbles_l530_530140

theorem minimum_yellow_marbles :
  ∀ (n y : ℕ), 
  (3 ∣ n) ∧ (4 ∣ n) ∧ 
  (9 + y + 2 * y ≤ n) ∧ 
  (n = n / 3 + n / 4 + 9 + y + 2 * y) → 
  y = 4 :=
by
  sorry

end minimum_yellow_marbles_l530_530140


namespace letter_150th_in_pattern_l530_530495

def repeating_sequence := "XYZ"

def letter_at_position (n : ℕ) : char :=
  let seq := repeating_sequence.to_list
  seq.get! ((n - 1) % seq.length)

theorem letter_150th_in_pattern : letter_at_position 150 = 'Z' :=
by sorry

end letter_150th_in_pattern_l530_530495


namespace negation_proof_l530_530952

theorem negation_proof :
  (¬ (∃ x0 : ℝ, x0 > 1 ∧ x0^2 - x0 + 2016 > 0)) ↔
  (∀ (x : ℝ), x > 1 → x^2 - x + 2016 ≤ 0) :=
begin
  sorry
end

end negation_proof_l530_530952


namespace diagonals_of_30_sided_polygon_l530_530807

theorem diagonals_of_30_sided_polygon : 
  ∀ (n : ℕ), n = 30 → (n * (n - 3)) / 2 = 405 := 
by
  intro n h
  rw h
  simp
  sorry

end diagonals_of_30_sided_polygon_l530_530807


namespace number_of_incorrect_conclusions_l530_530017

theorem number_of_incorrect_conclusions :
  let C₁ := (∀ k : ℝ, 5 < k ∧ k < 8 → (k - 5) ≠ (8 - k))
  let C₂ := (let e₁_y := 4; e₂_x := 4 in
            ∀ c : Type, (0, e₁_y) ≠ (e₂_x, 0))
  let C₃ := (∀ (a b c M F₁ F₂ : ℝ), c = 4 ∧ || M - F₁ || = 5 → || M - F₂ || ∈ {9})
  let C₄ := (∀ (a b : ℝ), (- (b^2 / a^2) = - 1 / 3) → (sqrt (6) / 3 = (sqrt (1 - (b^2 / a^2)))))
 in
  let incorrect_conclusions := C₁ → true ∧ C₂ → true ∧ C₃ → true ∧ C₄ → false in
  ∃ n, n = 3 := by
  sorry

end number_of_incorrect_conclusions_l530_530017


namespace switches_not_infinitely_repeated_l530_530068

noncomputable def switches_infinitely_repeated (k : ℕ) : Prop :=
∀ (s : ℕ → fin 4), 
  (∃ f : ℕ, ∀ m, ∃ n > m, ∀ i < k-2, s n = 1 → s (n+1) = 2 → s (n+2) = 3 → s (n+3) = 0) → 
  false

theorem switches_not_infinitely_repeated (k : ℕ) : 
  (∀ (s : ℕ → fin 4), 
    (∃ f : ℕ, ∀ m, ∃ n > m, ∀ i < k-2, s n = 1 → s (n+1) = 2 → s (n+2) = 3 → s (n+3) = 0) → 
    false) 
:= 
sorry

end switches_not_infinitely_repeated_l530_530068


namespace volume_of_cone_l530_530395

noncomputable def coneVolume (S : ℝ) : ℝ :=
  let r := sqrt (S / (3 * π))
  let l := sqrt (3 * S / π)
  let h := 2 * sqrt (2 * S / (3 * π))
  (1 / 3) * π * r^2 * h

theorem volume_of_cone (S : ℝ) (hS : S > 0) :
  coneVolume S = (2 * S * sqrt (6 * π * S)) / (27 * π) :=
by
  sorry

end volume_of_cone_l530_530395


namespace horseshoes_riding_school_l530_530106

theorem horseshoes_riding_school (iron_total : ℕ) (iron_per_horseshoe : ℕ)
    (farms : ℕ) (horses_per_farm : ℕ) 
    (stables : ℕ) (horses_per_stable : ℕ)
    (hooves_per_horse : ℕ) :
    iron_total = 400 →
    iron_per_horseshoe = 2 →
    farms = 2 →
    horses_per_farm = 2 →
    stables = 2 →
    horses_per_stable = 5 →
    hooves_per_horse = 4 →
    36 = (iron_total - 
        ((farms * horses_per_farm * hooves_per_horse + 
          stables * horses_per_stable * hooves_per_horse) * iron_per_horseshoe)) / 
        iron_per_horseshoe / hooves_per_horse :=
by
    intros h1 h2 h3 h4 h5 h6 h7
    rw [h1, h2, h3, h4, h5, h6, h7]
    simp
    sorry

end horseshoes_riding_school_l530_530106


namespace prize_money_and_fund_distribution_l530_530129

def annual_interest_rate : ℝ := 0.0624
def rate_per_category : ℝ := 0.0624 / 2 / 6
def initial_fund : ℝ := 21000
def growth_rate : ℝ := 1 + 0.0312

def a (n : ℕ) : ℝ := initial_fund * growth_rate ^ (n - 1)

theorem prize_money_and_fund_distribution :
  (∀ n : ℕ, 0 < n → a n = initial_fund * growth_rate ^ (n - 1)) ∧
  (let prize_money_2011 := rate_per_category * a 11 in prize_money_2011 ≤ 1.5) ∧
  (let sum_prize_money := (3.12 / 100) * (initial_fund * ((growth_rate ^ 10 - 1) / (growth_rate - 1))) 
   in sum_prize_money = 7560) := by
    sorry

end prize_money_and_fund_distribution_l530_530129


namespace measure_of_angle_F_l530_530309

-- defining our variables
variables {D E F : ℝ}

-- defining our conditions
def is_isosceles (D E F : ℝ) : Prop := D = E
def angles_sum (D E F : ℝ) : Prop := D + E + F = 180
def angle_F_property (D E : ℝ) : Prop := ∀ F, F = D + 40

-- the main theorem
theorem measure_of_angle_F (D E : ℝ) (h_iso : is_isosceles D E F) (h_sum : angles_sum D E F) (h_F : angle_F_property D E  F) : F = 86.67 :=
by {
  sorry -- proof omitted
}

end measure_of_angle_F_l530_530309


namespace max_volume_pyramid_l530_530938

noncomputable def maxVolumePyramid (a : ℝ) (h : ℝ) : ℝ :=
  (1 / 3) * (a ^ 2) * h

theorem max_volume_pyramid :
  ∃ a h : ℝ, 
    (∀ a b : ℝ, maxVolumePyramid a b = (1 / 3) * (a ^ 2) * b)
    ∧ (a > 0) 
    ∧ (h > 0) 
    ∧ h = sqrt ((5 - a * sqrt 2) / 2 ^ 2 - ((a * sqrt 2) / 2) ^ 2)
    ∧ 2 * h + a * sqrt 2 = 5  
    ∧ maxVolumePyramid a h = sqrt 5 / 3 
:= 
  sorry

end max_volume_pyramid_l530_530938


namespace triangle_with_ratio_is_right_triangle_l530_530284

/-- If the ratio of the interior angles of a triangle is 1:2:3, then the triangle is a right triangle. -/
theorem triangle_with_ratio_is_right_triangle (x : ℝ) (h : x + 2*x + 3*x = 180) : 
  3*x = 90 :=
sorry

end triangle_with_ratio_is_right_triangle_l530_530284


namespace vicentes_total_cost_l530_530062

def total_cost (rice_bought cost_per_kg_rice meat_bought cost_per_lb_meat : Nat) : Nat :=
  (rice_bought * cost_per_kg_rice) + (meat_bought * cost_per_lb_meat)

theorem vicentes_total_cost :
  let rice_bought := 5
  let cost_per_kg_rice := 2
  let meat_bought := 3
  let cost_per_lb_meat := 5
  total_cost rice_bought cost_per_kg_rice meat_bought cost_per_lb_meat = 25 :=
by
  intros
  sorry

end vicentes_total_cost_l530_530062


namespace problem_l530_530242

noncomputable def f (x : ℝ) : ℝ := Real.exp (x^2 + 2 * x)
noncomputable def a : ℝ := Real.log (1/5)
noncomputable def b : ℝ := Real.log (1/3) / Real.log (1/2)
noncomputable def c : ℝ := (1/3)^(0.5)

theorem problem (f a b c: ℝ) : 
  (f(b) < f(a) < f(c)) :=
by
  -- Proof to be filled in
  sorry

end problem_l530_530242


namespace find_c_solution_l530_530172

theorem find_c_solution {c : ℚ} 
  (h₁ : ∃ x : ℤ, 2 * (x : ℚ)^2 + 17 * x - 55 = 0 ∧ x = ⌊c⌋)
  (h₂ : ∃ x : ℚ, 6 * x^2 - 23 * x + 7 = 0 ∧ 0 ≤ x ∧ x < 1 ∧ x = c - ⌊c⌋) :
  c = -32 / 3 :=
by
  sorry

end find_c_solution_l530_530172


namespace enclosed_area_of_curve_l530_530941

-- Define the hexagon side length
def side_length : ℝ := 2

-- Define the number of arcs and their lengths
def number_of_arcs : ℕ := 9
def arc_length : ℝ := (2 * Real.pi) / 3

-- The radius of the circular arcs based on the given arc length and full circumference.
def radius : ℝ := 1

-- Compute the area of the regular hexagon with the given side length
def hexagon_area (s : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * (s ^ 2)

-- The total area of the sectors formed by the arcs (we know there are 9 arcs)
def total_sector_area (r : ℝ) : ℝ := 9 * (1 / 2) * r^2 * (2 * Real.pi / 3)

-- The final theorem statement
theorem enclosed_area_of_curve :
  let r := radius in
  let s := side_length in
  let hex_area := hexagon_area s in
  let sector_area := total_sector_area r in
  hex_area - sector_area = π + (6 * Real.sqrt 3) :=
by
  sorry

end enclosed_area_of_curve_l530_530941


namespace complement_of_A_in_U_l530_530740

open Set

-- Define the universal set U and set A based on the conditions
def U : Set ℝ := {x | x^2 ≥ 1}
def A : Set ℝ := {x | 1 < x ∧ x ≤ 2}

-- Define the complement of A in U
def complement_U_A : Set ℝ := {x | x ≤ -1 ∨ x = 1 ∨ x > 2}

-- The problematic statement: Prove that the complement of A in U is complement_U_A
theorem complement_of_A_in_U : compl U A = complement_U_A := by
  sorry

end complement_of_A_in_U_l530_530740


namespace number_of_diagonals_30_sides_l530_530778

def number_of_diagonals (n : ℕ) : ℕ :=
  nat.choose n 2 - n

theorem number_of_diagonals_30_sides :
  number_of_diagonals 30 = 405 :=
by {
  rw [number_of_diagonals, nat.choose, nat.factorial, nat.factorial, nat.factorial],
  -- The proof would proceed to simplify the combination and subtraction, but we use sorry to skip details.
  sorry,
}

end number_of_diagonals_30_sides_l530_530778


namespace spadesuit_eval_l530_530684

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spadesuit_eval : spadesuit 2 (spadesuit 3 (spadesuit 1 2)) = 4 := 
by
  sorry

end spadesuit_eval_l530_530684


namespace shaded_area_percentage_l530_530435

/-- Two congruent squares, each with side length 18, overlap to form a 20 by 18 rectangle.
    The problem is to show that 80% of the area of the rectangle is shaded. -/
theorem shaded_area_percentage (a b s : ℕ) (h₁ : a = 20) (h₂ : b = 18) (h₃ : s = 18) :
  let A_rectangle := a * b,
      overlap_length := 2 * s - a,
      A_shaded := overlap_length * b in
  (A_shaded * 100) / A_rectangle = 80 :=
by
  -- Place the necessary proof steps here
  sorry

end shaded_area_percentage_l530_530435


namespace vicente_total_spent_l530_530065

def kilograms_of_rice := 5
def cost_per_kilogram_of_rice := 2
def pounds_of_meat := 3
def cost_per_pound_of_meat := 5

def total_spent := kilograms_of_rice * cost_per_kilogram_of_rice + pounds_of_meat * cost_per_pound_of_meat

theorem vicente_total_spent : total_spent = 25 := 
by
  sorry -- Proof would go here

end vicente_total_spent_l530_530065


namespace find_150th_letter_in_pattern_l530_530443

/--
Given the repeating pattern "XYZ" with a cycle length of 3,
prove that the 150th letter in this pattern is 'Z'.
-/
theorem find_150th_letter_in_pattern : 
  let pattern := "XYZ"
  let cycle_length := String.length pattern
in (150 % cycle_length = 0) → "Z" := 
sorry

end find_150th_letter_in_pattern_l530_530443


namespace time_after_1750_minutes_is_1_10_pm_l530_530946

def add_minutes_to_time (hours : Nat) (minutes : Nat) : Nat × Nat :=
  let total_minutes := hours * 60 + minutes
  (total_minutes / 60, total_minutes % 60)

def time_after_1750_minutes (current_hour : Nat) (current_minute : Nat) : Nat × Nat :=
  let (new_hour, new_minute) := add_minutes_to_time current_hour current_minute
  let final_hour := (new_hour + 1750 / 60) % 24
  let final_minute := (new_minute + 1750 % 60) % 60
  (final_hour, final_minute)

theorem time_after_1750_minutes_is_1_10_pm : 
  time_after_1750_minutes 8 0 = (13, 10) :=
by {
  sorry
}

end time_after_1750_minutes_is_1_10_pm_l530_530946


namespace sine_180_eq_zero_l530_530629

theorem sine_180_eq_zero :
  sin (180 : ℝ) = 0 :=
sorry

end sine_180_eq_zero_l530_530629


namespace sum_of_coordinates_l530_530304

def friendly_point (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.2 - 1, -P.1 - 1)

theorem sum_of_coordinates (x y : ℝ) (A_2023 : ℝ × ℝ) (h : A_2023 = (-3, -2)) :
  (x, y) = (1, 2) → x + y = 3 :=
by
  intros h_eq
  rw [h_eq]
  simp
  exact h_eq

end sum_of_coordinates_l530_530304


namespace geometry_problem_l530_530857

theorem geometry_problem :
  (∀ θ : ℝ, let x := 1 + Real.cos θ, let y := 2 + Real.sin θ in (x - 1)^2 + (y - 2)^2 = 1) ∧
  (∀ θ : ℝ, let x := -2 in ρ * Real.cos θ = -2 → x = -2) ∧
  (let x := (3 / 2), let y := (3 / 2), polar_r := Real.sqrt (1.5 ^ 2 + 1.5 ^ 2), polar_θ := Real.pi / 4 in
   (x = y ∧ polar_r = (3 / 2) * Real.sqrt 2 ∧ polar_θ = Real.pi / 4)) := sorry

end geometry_problem_l530_530857


namespace alex_minimum_additional_coins_l530_530587

theorem alex_minimum_additional_coins (friends coins : ℕ) (h_friends : friends = 15) (h_coins : coins = 105) : 
  ∃ add_coins, add_coins = (∑ i in range (friends + 1), i) - coins :=
by
  sorry

end alex_minimum_additional_coins_l530_530587


namespace midpoint_of_KL_lies_on_circumcircle_l530_530096

-- Definitions of points K and L given conditions
variables {A B C K L M : Type*}
variables {circumcircle : Set Point}

-- Basic properties and conditions of points K and L
variable (triangle : Triangle A B C)
variables (internal_angle_bisector_B : Line (AngleBisector B))
variables (external_angle_bisector_C : Line (ExternalAngleBisector C))
variables (internal_angle_bisector_C : Line (AngleBisector C))
variables (external_angle_bisector_B : Line (ExternalAngleBisector B))

-- Conditions on points K and L
noncomputable def intersection_K : Point := intersection internal_angle_bisector_B external_angle_bisector_C
noncomputable def intersection_L : Point := intersection internal_angle_bisector_C external_angle_bisector_B

-- Midpoint of K and L
noncomputable def midpoint_M (K L : Point) : Point := midpoint K L

-- Circumcircle property statement
theorem midpoint_of_KL_lies_on_circumcircle :
  (midpoint_M K L) ∈ circumcircle (triangle_circumcircle A B C) :=
by sorry

end midpoint_of_KL_lies_on_circumcircle_l530_530096


namespace length_of_AD_l530_530127

theorem length_of_AD
  (R : ℝ) (A B C D : Point) (h_circle : ∀ P, dist P O = R) 
  (h_AB : dist A B = 100)
  (h_BC : dist B C = 150)
  (h_CD : dist C D = 100)
  (h_area : area_ABCD A B C D = 7500 * real.sqrt 3) :
  dist A D = 150 := 
sorry

end length_of_AD_l530_530127


namespace problem_solution_l530_530097

theorem problem_solution :
  ((8 * 2.25 - 5 * 0.85) / 2.5 + (3 / 5 * 1.5 - 7 / 8 * 0.35) / 1.25) = 5.975 :=
by
  sorry

end problem_solution_l530_530097


namespace tangent_parallel_to_given_line_l530_530252

theorem tangent_parallel_to_given_line (a : ℝ) : 
  let y := λ x : ℝ => x^2 + a / x
  let y' := λ x : ℝ => (deriv y) x
  y' 1 = 2 
  → a = 0 := by
  -- y'(1) is the derivative of y at x=1
  sorry

end tangent_parallel_to_given_line_l530_530252


namespace smallest_common_multiple_8_6_l530_530522

theorem smallest_common_multiple_8_6 : 
  ∃ n : ℕ, n > 0 ∧ (n % 8 = 0) ∧ (n % 6 = 0) ∧ ∀ m : ℕ, m > 0 ∧ (m % 8 = 0) ∧ (m % 6 = 0) → m ≥ n :=
begin
  use 24,
  split,
  { norm_num }, -- 24 > 0
  split,
  { norm_num }, -- 24 % 8 = 0
  split,
  { norm_num }, -- 24 % 6 = 0
  { intros m hm,
    cases hm with hp8 hp6,
    norm_num at hp8 hp6,
    sorry -- Prove that 24 is the smallest such number
  }
end

end smallest_common_multiple_8_6_l530_530522


namespace probability_heads_given_heads_l530_530424

-- Definitions based on the conditions
inductive Coin
| coin1 | coin2 | coin3

open Coin

def P (event : Set Coin) : ℝ := 
  if event = {coin1} then 1/3
  else if event = {coin2} then 1/3
  else if event = {coin3} then 1/3
  else 0

def event_heads (coin : Coin) : bool :=
  match coin with
  | coin1 => true
  | coin2 => true
  | coin3 => false

-- Probability of showing heads
def P_heads_given_coin (coin : Coin) : ℝ :=
  if event_heads coin then 1 else 0

-- Total probability of getting heads
def P_heads : ℝ := 
  P {coin1} * (if event_heads coin1 then 1 else 0) +
  P {coin2} * (if event_heads coin2 then 1 else 0) +
  P {coin3} * (if event_heads coin3 then 0 else 0)

-- Using Bayes' Theorem to find probability that the coin showing heads is coin2
def P_coin2_given_heads : ℝ :=
  (P_heads_given_coin coin2 * P {coin2}) / P_heads

-- Defining the theorem statement
theorem probability_heads_given_heads : P_coin2_given_heads = 2 / 3 := by
  sorry

end probability_heads_given_heads_l530_530424


namespace no_nines_in_product_l530_530403

theorem no_nines_in_product : 
  let a := 123456789
  let b := 999999999
  let product := a * b
  number_of_nine_digits product = 0 :=
by
  sorry

def number_of_nine_digits (n : ℕ) : ℕ :=
  n.to_digits.filter (λ d => d = 9).length

end no_nines_in_product_l530_530403


namespace perfect_square_probability_l530_530688

def set_of_cards : set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

noncomputable def draw_two_cards (s : set ℕ) : set (ℕ × ℕ) :=
{p | p.1 ∈ s ∧ p.2 ∈ s ∧ p.1 < p.2}

theorem perfect_square_probability :
  let outcomes := draw_two_cards set_of_cards in
  let favorable := {p ∈ outcomes | is_perfect_square (p.1 * p.2)} in
  (favorable.card : ℚ) / (outcomes.card : ℚ) = 1 / 9 :=
by
  sorry

end perfect_square_probability_l530_530688


namespace blue_apples_l530_530265

theorem blue_apples (B : ℕ) (h : (12 / 5) * B = 12) : B = 5 :=
by
  sorry

end blue_apples_l530_530265


namespace find_ratio_BG_GA_l530_530299

variables {A B C E G Q : Type}
variables [points : Type]

def BG_over_GA (BQ QE GQ QA : ℕ) [BQ = 2] [QE = 1] [GQ = 3] [QA = 4] : Prop :=
  (BG / GA) = 3 / 4

theorem find_ratio_BG_GA (E_in_AC : E ∈ line AC) (G_in_AB : G ∈ line AB)
  (Q_on_BE_AG : ∃ Q, Q ∈ line BE ∧ Q ∈ line AG) :
  BG_over_GA 2 1 3 4 := by
sorry

end find_ratio_BG_GA_l530_530299


namespace find_x_l530_530274

theorem find_x (x : ℚ) (h : (3 * x - 7) / 4 = 15) : x = 67 / 3 :=
sorry

end find_x_l530_530274


namespace constant_term_expansion_l530_530660

-- Definition of the binomial term
noncomputable def binomial_term (n k : ℕ) (a b : ℝ) : ℝ := nat.choose n k * a^(n-k) * b^k

-- Define the specific problem parameters
def general_term (r : ℕ) : ℝ := binomial_term 6 r (1:ℝ) (-1) * (1/x)^(6-r) * (x^(1/2))^r

-- Main statement proving the specific example
theorem constant_term_expansion: general_term 4 = 15 := by
  sorry

end constant_term_expansion_l530_530660


namespace fold_paper_point_match_l530_530570

-- Definitions from the conditions
def point_A := (0 : ℝ, 5 : ℝ)
def point_B := (5 : ℝ, 0 : ℝ)
def point_C := (8 : ℝ, 4 : ℝ)

variable (m n : ℝ)

-- Theorem statement
theorem fold_paper_point_match :
  let mid_point := ((point_A.1 + point_B.1) / 2, (point_A.2 + point_B.2) / 2)
  let perp_bisector := fun x : ℝ => x
  let match_point := (m, n)
  (point_A.1 - point_B.1)^2 + (point_A.2 - point_B.2)^2 = (m-8)^2 + (n-4)^2 ∧
  (m + 8) / 2 = (n + 4) / 2 →
  m + n = 12 :=
begin
  sorry
end

end fold_paper_point_match_l530_530570


namespace check_point_on_graph_l530_530594

def lies_on_graph (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  f p.1 = p.2

def linear_function := λ x : ℝ, -3 * x + 3

theorem check_point_on_graph :
  lies_on_graph linear_function (-2, 9) :=
by {
  -- Start the proof by unfolding the lie_on_graph definition and substituting the x-value
  unfold lies_on_graph,
  -- Perform the calculation for verification
  calc
  linear_function (-2) = -3 * (-2) + 3 : by rfl
                   ... = 9             : by norm_num,
}

end check_point_on_graph_l530_530594


namespace perfect_square_in_product_set_l530_530022

theorem perfect_square_in_product_set :
  ∃ (S : Finset ℕ), S.card ≤ 1986 ∧ (∀ (n ∈ S), n ≠ 0) ∧
  (∃ k : ℕ, k ≠ 0 ∧ (S.prod id = k^2)) :=
by
  sorry

end perfect_square_in_product_set_l530_530022


namespace regular_ngon_diagonal_difference_side_length_eq_nine_l530_530014

theorem regular_ngon_diagonal_difference_side_length_eq_nine (n : ℕ) (h : n ≥ 3) :
  (∃ (side_length longest_diagonal shortest_diagonal : ℝ), 
     regular_ngon n ∧
     side_length = diagonal_length n ∧
     longest_diagonal = diagonal_length n ∧
     shortest_diagonal = diagonal_length n ∧
     side_length = longest_diagonal - shortest_diagonal) →
  n = 9 := 
by
  sorry -- Proof goes here

end regular_ngon_diagonal_difference_side_length_eq_nine_l530_530014


namespace tetrahedron_non_coplanar_points_l530_530135

theorem tetrahedron_non_coplanar_points :
  let vertices := 4
  let midpoints_of_edges := 6
  let total_points := vertices + midpoints_of_edges
  let total_selections := nat.choose total_points 4
  let coplanar_on_same_face := 4 * nat.choose 6 4
  let coplanar_with_opposite_edge_midpoint := 6
  let coplanar_parallelogram := 3
  let coplanar_total := coplanar_on_same_face + coplanar_with_opposite_edge_midpoint + coplanar_parallelogram
  (total_selections - coplanar_total) = 141
  := by
    sorry

end tetrahedron_non_coplanar_points_l530_530135


namespace convex_polygon_diagonals_l530_530784

theorem convex_polygon_diagonals (n : ℕ) (h : n = 30) : 
  let total_pairs := (n * (n - 1)) / 2 in (total_pairs - n) = 405 := 
by
  simp only [total_pairs]
  sorry

end convex_polygon_diagonals_l530_530784


namespace sin_x_inequality_l530_530960

theorem sin_x_inequality (m : ℝ) : 
  (∃ x, -1 ≤ sin x ∧ sin x ≤ 1 ∧ sin x = 1 - m) ↔ (0 ≤ m ∧ m ≤ 2) := 
by 
  sorry

end sin_x_inequality_l530_530960


namespace intersection_M_N_l530_530922

def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 4}

theorem intersection_M_N : M ∩ N = {x | 1 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_M_N_l530_530922


namespace train_speed_l530_530060

theorem train_speed (length1 length2 speed2 : ℝ) (time_seconds speed1 : ℝ)
    (h_length1 : length1 = 111)
    (h_length2 : length2 = 165)
    (h_speed2 : speed2 = 90)
    (h_time : time_seconds = 6.623470122390208)
    (h_speed1 : speed1 = 60) :
    (length1 / 1000.0) + (length2 / 1000.0) / (time_seconds / 3600) = speed1 + speed2 :=
by
  sorry

end train_speed_l530_530060


namespace find_angle_A_max_area_of_triangle_l530_530289

noncomputable def m : ℝ × ℝ := (real.cos A, real.cos B)
noncomputable def n : ℝ × ℝ := (b - 2 * c, a)

theorem find_angle_A (A B : ℝ) (a b c : ℝ) (h : m.1 * n.1 + m.2 * n.2 = 0) : A = π / 3 :=
sorry

theorem max_area_of_triangle (A B C : ℝ) (a b c : ℝ) (hA : A = π / 3) (ha : a = 3) : 
  real.abs (a * b * real.sin C / 2) ≤ 9 * real.sqrt 3 / 4 :=
sorry

end find_angle_A_max_area_of_triangle_l530_530289


namespace convex_polygon_diagonals_l530_530812

theorem convex_polygon_diagonals (n : ℕ) (h : n = 30) : 
  let V := n in
  let total_pairs := V * (V - 1) / 2 in
  let adjacent_pairs := n in
  let diagonals := total_pairs - adjacent_pairs in
  diagonals = 405 :=
  by
  -- number of sides is 30
  have hn : n = 30 := h
  -- calculate the total pairs
  have total_pairs_calc : total_pairs = 30 * 29 / 2 := by sorry
  -- calculate the adjacent pairs
  have adjacent_pairs_calc : adjacent_pairs = 30 := by sorry
  -- calculate the diagonals
  have diagonals_calc : diagonals = (30 * 29 / 2) - 30 := by sorry
  -- proved statement
  show 405 = 405, by rfl

end convex_polygon_diagonals_l530_530812


namespace range_of_x_l530_530715

def f (x : ℝ) : ℝ := log(x + 1) / log(2) + 2^x - 1

theorem range_of_x 
  (hx : ∀ x : ℝ, f (-x) = -f x)
  (h_fx_expr : ∀ x : ℝ, x ≥ 0 → f x = log(x + 1) / log(2) + 2^x - 1) : 
  {x : ℝ | f(x^2 - 3 * x - 1) + 9 < 0} = {x : ℝ | -1 < x ∧ x < 2} :=
sorry

end range_of_x_l530_530715


namespace solve_for_z_l530_530383

theorem solve_for_z (i : ℂ) (z : ℂ) (h : 3 - 5 * i * z = -2 + 5 * i * z) (h_i : i^2 = -1) :
  z = -i / 2 :=
by {
  sorry
}

end solve_for_z_l530_530383


namespace find_min_A_value_l530_530526

def f (A : ℝ) : ℝ := Real.sin (A / 2) - Real.sqrt 3 * Real.cos (A / 2)

theorem find_min_A_value :
  ∀ (A : ℝ), 
    (let min_A := 
      let k := 0 in 
        660 + k * 720 in 
        A ≠ min_A)
        → 
        (A ≠ 300) → 
        A ≠ 60 → 
        A ≠ 120 → 
        A ≠ 0 → 
        ∀ x ∈ {A | f A}, 
          A ∉ {-180, 60, 120, 0}
        (E) None of these.
  ) :=
by sorry

end find_min_A_value_l530_530526


namespace parallel_statements_l530_530596

-- Definitions for lines and planes
variables {Line Plane : Type}

-- Parallelism between a line and another line or a plane
variables (a b : Line) (α : Plane)

-- Parallel relationship assertions
axiom parallel_lines (l1 l2 : Line) : Prop -- l1 is parallel to l2
axiom line_in_plane (l : Line) (p : Plane) : Prop -- line l is in plane p
axiom parallel_line_plane (l : Line) (p : Plane) : Prop -- line l is parallel to plane p

-- Problem statement
theorem parallel_statements :
  (parallel_lines a b ∧ line_in_plane b α → parallel_line_plane a α) ∧
  (parallel_lines a b ∧ parallel_line_plane a α → parallel_line_plane b α) :=
sorry

end parallel_statements_l530_530596


namespace x_eq_3_minus_2t_and_y_eq_3t_plus_6_l530_530001

theorem x_eq_3_minus_2t_and_y_eq_3t_plus_6 (t : ℝ) (x : ℝ) (y : ℝ) : x = 3 - 2 * t → y = 3 * t + 6 → x = 0 → y = 10.5 :=
by
  sorry

end x_eq_3_minus_2t_and_y_eq_3t_plus_6_l530_530001


namespace christian_initial_savings_l530_530614

theorem christian_initial_savings (christian_earnings sue_initial sue_earnings total_needed remaining_needed: ℝ) 
  (h1: christian_earnings = 4 * 5)
  (h2: sue_initial = 7)
  (h3: sue_earnings = 6 * 2)
  (h4: total_needed = 50)
  (h5: remaining_needed = 6) : (∃ initial_savings, initial_savings = 5) :=
by
  use 5
  sorry

end christian_initial_savings_l530_530614


namespace max_shooters_at_same_person_l530_530123

theorem max_shooters_at_same_person (n : ℕ) (h_distinct : ∀ i j : fin n, i ≠ j → dist i j ≠ dist j i) : n ≤ 5 := 
sorry

end max_shooters_at_same_person_l530_530123


namespace total_weight_of_remaining_eggs_is_correct_l530_530357

-- Define the initial conditions and the question as Lean definitions
def total_eggs : Nat := 12
def weight_per_egg : Nat := 10
def num_boxes : Nat := 4
def melted_boxes : Nat := 1

-- Calculate the total weight of the eggs
def total_weight : Nat := total_eggs * weight_per_egg

-- Calculate the number of eggs per box
def eggs_per_box : Nat := total_eggs / num_boxes

-- Calculate the weight per box
def weight_per_box : Nat := eggs_per_box * weight_per_egg

-- Calculate the number of remaining boxes after one is tossed out
def remaining_boxes : Nat := num_boxes - melted_boxes

-- Calculate the total weight of the remaining chocolate eggs
def remaining_weight : Nat := remaining_boxes * weight_per_box

-- The proof task
theorem total_weight_of_remaining_eggs_is_correct : remaining_weight = 90 := by
  sorry

end total_weight_of_remaining_eggs_is_correct_l530_530357


namespace sin_180_eq_0_l530_530650

theorem sin_180_eq_0 : Real.sin (180 * Real.pi / 180) = 0 :=
by
  sorry

end sin_180_eq_0_l530_530650


namespace quadratic_unique_solution_l530_530956

theorem quadratic_unique_solution (a c : ℝ) :
  (∀ x : ℝ, a * x^2 + 36 * x + c = 0 ↔ x = (-36) / (2*a))  -- The quadratic equation has exactly one solution
  → a + c = 37  -- Given condition
  → a < c      -- Given condition
  → (a, c) = ( (37 - Real.sqrt 73) / 2, (37 + Real.sqrt 73) / 2 ) :=  -- Correct answer
by
  sorry

end quadratic_unique_solution_l530_530956


namespace cone_tangent_min_lateral_area_l530_530414

/-- 
Given a cone with volume π / 6, prove that when the lateral area of the cone is minimized,
the tangent of the angle between the slant height and the base is sqrt(2).
-/
theorem cone_tangent_min_lateral_area :
  ∀ (r h l : ℝ), (π / 6 = (1 / 3) * π * r^2 * h) →
    (h = 1 / (2 * r^2)) →
    (l = Real.sqrt (r^2 + h^2)) →
    ((π * r * l) ≥ (3 / 4 * π)) →
    (r = Real.sqrt (2) / 2) →
    (h / r = Real.sqrt (2)) :=
by
  intro r h l V_cond h_cond l_def min_lateral_area r_val
  -- Proof steps go here (omitted as per the instruction)
  sorry

end cone_tangent_min_lateral_area_l530_530414


namespace incorrect_conclusions_l530_530214

theorem incorrect_conclusions
  (h1 : ∃ (y x : ℝ), (¬∃ a b : ℝ, a < 0 ∧ y = a * x + b) ∧ ∃ a b : ℝ, y = 2.347 * x - 6.423)
  (h2 : ∃ (y x : ℝ), (∃ a b : ℝ, a < 0 ∧ y = a * x + b) ∧ y = -3.476 * x + 5.648)
  (h3 : ∃ (y x : ℝ), (∃ a b : ℝ, a > 0 ∧ y = a * x + b) ∧ y = 5.437 * x + 8.493)
  (h4 : ∃ (y x : ℝ), (¬∃ a b : ℝ, a > 0 ∧ y = a * x + b) ∧ y = -4.326 * x - 4.578) :
  (∃ (y x : ℝ), y = 2.347 * x - 6.423 ∧ (¬∃ a b : ℝ, a < 0 ∧ y = a * x + b)) ∧
  (∃ (y x : ℝ), y = -4.326 * x - 4.578 ∧ (¬∃ a b : ℝ, a > 0 ∧ y = a * x + b)) :=
by {
  sorry
}

end incorrect_conclusions_l530_530214


namespace param_line_segment_l530_530951

theorem param_line_segment:
  ∃ (p q r s : ℝ), 
      (0 ≤ t ∧ t ≤ 1 → (q, s) = (1, -3) ∧ (p + q, r + s) = (4, 6)) →
      (p^2 + q^2 + r^2 + s^2 = 100) :=
by 
  have hq : q = 1 := sorry,
  have hs : s = -3 := sorry,
  have hp : p = 3 := sorry,
  have hr : r = 9 := sorry,
  exact ⟨3, 1, 9, -3, by simp [hp, hq, hr, hs]⟩.

end param_line_segment_l530_530951


namespace DE_bisects_BC_l530_530940

noncomputable def midpoint (A B : Point) : Point := sorry
noncomputable def bisect {A B C : Point} : Prop := C = midpoint A B
noncomputable def perpendicular (A B C : Point) : Prop := sorry

variable (O A B C D E M N : Point)
variable (circle : Circle)
variable [circle : HasCenter circle O]

-- Given conditions
axiom h1 : perpendicular C D (diameter circle)
axiom h2 : midpoint O C = E

-- Question: Prove that chord DE bisects chord BC
theorem DE_bisects_BC (h1 h2 : Prop) : bisect (intersection (chord D E) (chord B C)) B C :=
sorry

end DE_bisects_BC_l530_530940


namespace exists_valid_arrangement_l530_530369

def valid_arrangement (arrangement : ℕ → ℕ) : Prop :=
  arrangement 1 + arrangement 7 +  arrangement 4 = 12 ∧
  arrangement 4 + arrangement 1 +  arrangement 3 = 12 ∧ 
  arrangement 2 + arrangement 6 +  arrangement 5 = 12 ∧
  arrangement 5 + arrangement 7 +  arrangement 3 = 12 ∧
  arrangement 1 + arrangement 2 +  arrangement 6 = 12 

theorem exists_valid_arrangement: 
  ∃ (arrangement : ℕ → ℕ), 
  (∀ n, n ∈ {1, 2, 3, 4, 5, 6, 7} → arrangement n ∈ {1, 2, 3, 4, 5, 6, 7}) ∧ 
  (∀ m n, m ≠ n → arrangement m ≠ arrangement n) ∧ 
  valid_arrangement arrangement := 
sorry

end exists_valid_arrangement_l530_530369


namespace find_150th_letter_l530_530469

def pattern : List Char := ['X', 'Y', 'Z']

def position (N : ℕ) (pattern_length : ℕ) : ℕ :=
  if N % pattern_length = 0 then pattern_length else N % pattern_length

theorem find_150th_letter : 
  let pattern_length := 3 in (position 150 pattern_length = 3) ∧ (pattern.drop (position 150 pattern_length - 1)).head = 'Z' :=
by
  sorry

end find_150th_letter_l530_530469


namespace shaded_triangle_area_l530_530950

/--
The large equilateral triangle shown consists of 36 smaller equilateral triangles.
Each of the smaller equilateral triangles has an area of 10 cm². 
The area of the shaded triangle is K cm².
Prove that K = 110 cm².
-/
theorem shaded_triangle_area 
  (n : ℕ) (area_small : ℕ) (area_total : ℕ) (K : ℕ)
  (H1 : n = 36)
  (H2 : area_small = 10)
  (H3 : area_total = n * area_small)
  (H4 : K = 110)
: K = 110 :=
by
  -- Adding 'sorry' indicating missing proof steps.
  sorry

end shaded_triangle_area_l530_530950


namespace sum_of_perimeters_of_triangle_is_168_l530_530907

-- Define the problem conditions
def AB : ℕ := 7
def BC : ℕ := 17
variable (x y : ℕ)  -- AD, CD and BD
axiom h1 : x = AD := rfl
axiom h2 : x = CD := rfl
axiom h3 : AD > 0
axiom h4 : BD > 0
axiom h5 : x * x - y * y = 95

-- Define the resulting sum
def S : ℕ := 168

-- Prove the sum of perimeters is 168 given conditions
theorem sum_of_perimeters_of_triangle_is_168 (A B C D : Type) [AddSemigroup A] [DivIdempotent A] :
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ (x*x - y*y = 95) ∧ S = 168 := by
  sorry

end sum_of_perimeters_of_triangle_is_168_l530_530907


namespace lcm_is_one_of_numbers_gcd_is_one_of_numbers_l530_530082

-- Definitions and conditions identified in part A
def is_prime_factorization (n : ℕ) (f : ℕ → ℕ) : Prop :=
  ∃ (p : ℕ → ℕ), (f = p) ∧ (∀ n, nat.prime n → p n = f n)

def include_all_prime_factors (a b : ℕ) : Prop :=
  ∀ (p : ℕ → ℕ) (e : ℕ), is_prime_factorization a p → is_prime_factorization b (λ p, nat.max (p a) e)

def is_divisor (a b : ℕ) : Prop :=
  b % a = 0

-- Proof problems for LCM and GCD
theorem lcm_is_one_of_numbers (a b : ℕ) (h₁ : include_all_prime_factors a b) : 
  nat.lcm a b = b :=
sorry

theorem gcd_is_one_of_numbers (a b : ℕ) (h₂ : is_divisor b a) : 
  nat.gcd a b = b :=
sorry

end lcm_is_one_of_numbers_gcd_is_one_of_numbers_l530_530082


namespace topsoil_cost_l530_530051

theorem topsoil_cost 
  (cost_per_cubic_foot : ℝ)
  (cubic_yards_to_cubic_feet : ℝ)
  (cubic_yards : ℝ) :
  cost_per_cubic_foot = 8 →
  cubic_yards_to_cubic_feet = 27 →
  cubic_yards = 8 →
  (cubic_yards * cubic_yards_to_cubic_feet * cost_per_cubic_foot) = 1728 :=
begin
  intros h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
end

end topsoil_cost_l530_530051


namespace number_of_diagonals_in_30_sided_polygon_l530_530760

-- Definition of a convex polygon with 30 sides
def convex_polygon_sides := 30

-- The function to calculate the number of diagonals in a convex polygon with n sides
noncomputable def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Prove that a convex polygon with 30 sides has 405 diagonals
theorem number_of_diagonals_in_30_sided_polygon :
  num_diagonals convex_polygon_sides = 405 :=
by
  calc
    num_diagonals 30 = 30 * (30 - 3) / 2 : rfl
    ... = 30 * 27 / 2 : rfl
    ... = 810 / 2 : rfl
    ... = 405 : rfl

end number_of_diagonals_in_30_sided_polygon_l530_530760


namespace number_of_diagonals_in_30_sided_polygon_l530_530763

-- Definition of a convex polygon with 30 sides
def convex_polygon_sides := 30

-- The function to calculate the number of diagonals in a convex polygon with n sides
noncomputable def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Prove that a convex polygon with 30 sides has 405 diagonals
theorem number_of_diagonals_in_30_sided_polygon :
  num_diagonals convex_polygon_sides = 405 :=
by
  calc
    num_diagonals 30 = 30 * (30 - 3) / 2 : rfl
    ... = 30 * 27 / 2 : rfl
    ... = 810 / 2 : rfl
    ... = 405 : rfl

end number_of_diagonals_in_30_sided_polygon_l530_530763


namespace area_relationship_l530_530568

theorem area_relationship (x β : ℝ) (hβ : 0.60 * x^2 = β) : α = (4 / 3) * β :=
by
  -- conditions and goal are stated
  let α := 0.80 * x^2
  sorry

end area_relationship_l530_530568


namespace find_150th_letter_l530_530459

theorem find_150th_letter :
  let pattern := ['X', 'Y', 'Z']
  150 % 3 = 0 -> pattern[(150 % 3 + 2) % 3] = 'Z' :=
begin
  intros pattern h,
  simp at *,
  exact rfl,
end

end find_150th_letter_l530_530459


namespace complex_real_eq_implies_b_value_l530_530285

theorem complex_real_eq_implies_b_value (b : ℝ) :
  (complex.re ( (3 - b * complex.I) / (2 + complex.I) ) = 
   complex.im ( (3 - b * complex.I) / (2 + complex.I) )) →
  b = -9 :=
by
  sorry

end complex_real_eq_implies_b_value_l530_530285


namespace plus_signs_count_l530_530904

theorem plus_signs_count (n : ℕ) (h : n ≥ 4) (chart : ℕ → ℕ → Prop) :
  (∀ (i j : ℕ), i = j → chart i j = true) ∧ 
  (∀ (i j : ℕ), i ≠ j → chart i j = false) →
  (∃ (f : ℕ → ℕ → ℕ → ℕ → Prop), (∀ k : ℕ, k ≥ 4 →
    ∀ i j : ℕ,
    (chart i j = ¬(chart i j)) ∨ 
    (chart i j = (¬chart (f k i j i j)))) →
    (∑ (i j : ℕ) in finset.range n ×ˢ finset.range n, (if chart i j then 1 else 0) ≥ n)) :=
sorry

end plus_signs_count_l530_530904


namespace max_value_p_l530_530732

noncomputable theory

def f (x : ℝ) : ℝ := Real.exp x

theorem max_value_p (m n p : ℝ) 
  (h1 : ∀ m n : ℝ, f (m + n) = f m + f n)
  (h2 : ∀ m n p : ℝ, f (m + n + p) = f m + f n + f p) : 
  p = 2 * Real.log 2 - Real.log 3 := 
sorry

end max_value_p_l530_530732


namespace QR_eq_RT_l530_530877

-- Definitions of the given conditions
structure Circle (Point : Type) :=
(center : Point)
(radius : ℝ)

variables {Point : Type} [metric_space Point] [normed_space ℝ Point]

def Circle.externally_tangent (C1 C2 : Circle Point) (S : Point) : Prop :=
  dist C1.center C2.center = C1.radius + C2.radius ∧ (dist C1.center S = C1.radius ∧ dist C2.center S = C2.radius)

def tangent_line (C : Circle Point) (P : Point) : Prop :=
  dist C.center P = C.radius
  
noncomputable def diameter (C : Circle Point) (Q T : Point) : Prop :=
  dist Q T = 2 * C.radius ∧ dist C.center Q = C.radius ∧ dist C.center T = C.radius

-- The main theorem
theorem QR_eq_RT (C1 C2 : Circle Point) (S P Q T R : Point)
  (h1 : Circle.externally_tangent C1 C2 S) 
  (h2 : C1.radius = r) 
  (h3 : C2.radius = 3 * r)
  (h4 : tangent_line C1 P ∧ tangent_line C2 Q)
  (h5 : diameter C2 Q T)
  (h6 : ∃ R, R ∈ line_segment S T ∧ is_angle_bisector (∠S Q T) (line_segment R T)) :
  dist Q R = dist R T :=
begin
  sorry
end

end QR_eq_RT_l530_530877


namespace prime_pairs_l530_530672

def is_prime (n : ℕ) : Prop := nat.prime n

theorem prime_pairs (p q : ℕ) (h₀ : is_prime p) (h₁ : is_prime q) (h₂ : is_prime (p * q + p - 6)) :
  (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) :=
sorry

end prime_pairs_l530_530672


namespace number_of_5_letter_words_with_at_least_one_consonant_l530_530264

def total_letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'I'}
def vowels : Finset Char := {'A', 'E', 'I'}
def consonants : Finset Char := total_letters \ vowels

def total_5_letter_words : ℕ := (total_letters.card) ^ 5
def all_vowel_5_letter_words : ℕ := (vowels.card) ^ 5

theorem number_of_5_letter_words_with_at_least_one_consonant :
  total_5_letter_words - all_vowel_5_letter_words = 32525 := by
  have total : total_5_letter_words = 8^5 := by rfl
  have all_vowel : all_vowel_5_letter_words = 3^5 := by rfl
  have total_calc : 8^5 = 32768 := by norm_num
  have all_vowel_calc : 3^5 = 243 := by norm_num
  rw [total, all_vowel, total_calc, all_vowel_calc]
  norm_num
  sorry

end number_of_5_letter_words_with_at_least_one_consonant_l530_530264


namespace problem_part1_problem_part2_l530_530244

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := abs (a - x)

def setA (a : ℝ) : Set ℝ := {x | f a (2 * x - 3 / 2) > 2 * f a (x + 2) + 2}

theorem problem_part1 {a : ℝ} (h : a = 3 / 2) : setA a = {x | x < 0} := by
  sorry

theorem problem_part2 {a : ℝ} (h : a = 3 / 2) (x0 : ℝ) (hx0 : x0 ∈ setA a) (x : ℝ) : 
    f a (x0 * x) ≥ x0 * f a x + f a (a * x0) := by
  sorry

end problem_part1_problem_part2_l530_530244


namespace number_of_diagonals_in_convex_polygon_l530_530748

/-- 
A theorem stating that the number of diagonals 
in a convex polygon with 30 sides is equal to 405.
-/
theorem number_of_diagonals_in_convex_polygon (n : ℕ) (h : n = 30) (convex : True) : (n * (n - 3)) / 2 = 405 := 
by 
  rw h
  norm_num
  done

end number_of_diagonals_in_convex_polygon_l530_530748


namespace solve_for_a_l530_530164

def op (a b : ℝ) : ℝ := 3 * a - 2 * b ^ 2

theorem solve_for_a (a : ℝ) : op a 3 = 15 → a = 11 :=
by
  intro h
  rw [op] at h
  sorry

end solve_for_a_l530_530164


namespace problem_solution_set_l530_530238

-- Definitions and conditions according to the given problem
def odd_function_domain := {x : ℝ | x ≠ 0}
def function_condition1 (f : ℝ → ℝ) (x : ℝ) : Prop := x > 0 → deriv f x < (3 * f x) / x
def function_condition2 (f : ℝ → ℝ) : Prop := f 1 = 1 / 2
def function_condition3 (f : ℝ → ℝ) : Prop := ∀ x, f (2 * x) = 2 * f x

-- Main proof statement
theorem problem_solution_set (f : ℝ → ℝ)
  (odd_function : ∀ x, f (-x) = -f x)
  (dom : ∀ x, x ∈ odd_function_domain → f x ≠ 0)
  (cond1 : ∀ x, function_condition1 f x)
  (cond2 : function_condition2 f)
  (cond3 : function_condition3 f) :
  {x : ℝ | f x / (4 * x) < 2 * x^2} = {x : ℝ | x < -1 / 4} ∪ {x : ℝ | x > 1 / 4} :=
sorry

end problem_solution_set_l530_530238


namespace f_at_neg_3_l530_530217

def f (x : Int) : Int :=
  if x >= 6 then
    x - 5
  else
    f (x + 2)

theorem f_at_neg_3 : f (-3) = 2 := by
  sorry

end f_at_neg_3_l530_530217


namespace complement_intersection_l530_530343

open Set

variable (U M N : Set ℕ)
variable (H₁ : U = {1, 2, 3, 4, 5, 6})
variable (H₂ : M = {1, 2, 3, 5})
variable (H₃ : N = {1, 3, 4, 6})

theorem complement_intersection :
  (U \ (M ∩ N)) = {2, 4, 5, 6} :=
by
  sorry

end complement_intersection_l530_530343


namespace b_dot_c_equals_three_l530_530270

noncomputable def vectors := ℝ^3
variables (a b c : vectors)

-- Given conditions
variables (ha : ∥a∥ = 1)
variables (hb : ∥b∥ = 1)
variables (hab : ∥a + b∥ = 2)
variables (h : c - 2 • a - b = 4 • (a × b))

-- Statement to prove
theorem b_dot_c_equals_three : b ⬝ c = 3 :=
by sorry

end b_dot_c_equals_three_l530_530270


namespace calculate_expression_l530_530154

theorem calculate_expression : (Real.sqrt 4) + abs (3 - Real.pi) + (1 / 3)⁻¹ = 2 + Real.pi :=
by 
  sorry

end calculate_expression_l530_530154


namespace intersecting_segments_l530_530868

theorem intersecting_segments (n : ℕ) (l : ℝ) (segments : list (ℝ × ℝ)) 
  (h_segments_len : ∀ s ∈ segments, s.2 - s.1 = 1) 
  (h_segments_count : segments.length = 4 * n)
  (h_in_circle : ∀ s ∈ segments, dist (s.1, 0) (0, 0) ≤ n ∧ dist (s.2, 0) (0, 0) ≤ n) :
  ∃ l' : ℝ, (l' = l ∨ l' = -1 / l) ∧ (∃ s₁ s₂ ∈ segments, s₁ ≠ s₂ ∧ ∃ x, x ∈ s₁ ∩ s₂) :=
sorry

end intersecting_segments_l530_530868


namespace number_of_diagonals_in_convex_polygon_l530_530753

/-- 
A theorem stating that the number of diagonals 
in a convex polygon with 30 sides is equal to 405.
-/
theorem number_of_diagonals_in_convex_polygon (n : ℕ) (h : n = 30) (convex : True) : (n * (n - 3)) / 2 = 405 := 
by 
  rw h
  norm_num
  done

end number_of_diagonals_in_convex_polygon_l530_530753


namespace prime_iff_good_fractions_l530_530701

def isGoodFraction (n : ℕ) (a b : ℕ) : Prop := a > 0 ∧ b > 0 ∧ (a + b = n)

def canBeExpressedUsingGoodFractions (n : ℕ) (a b : ℕ) : Prop :=
  ∃ (expressedFraction : ℕ → ℕ → Prop), expressedFraction a b ∧
  ∀ x y, expressedFraction x y → isGoodFraction n x y

theorem prime_iff_good_fractions {n : ℕ} (hn : n > 1) :
  Prime n ↔
    ∀ a b : ℕ, b < n → (a > 0 ∧ b > 0) → canBeExpressedUsingGoodFractions n a b :=
sorry

end prime_iff_good_fractions_l530_530701


namespace dot_product_AD_BC_l530_530298

-- Definitions for vector operations and point relations
variables (A B C D: Type) [InnerProductSpace ℝ A] 
variables (a b c d : A) -- points corresponding to A, B, C, D 

-- Conditions 
axiom lengthBC : InnerProductSpace.norm (b - c) = 6
axiom D_relation : b - d = 2 • (d - c)

-- Theorem statement
theorem dot_product_AD_BC : inner ((a - d) : A) ((b - c) : A) = 6 := 
sorry

end dot_product_AD_BC_l530_530298


namespace remainder_of_polynomial_mod_60_l530_530833

theorem remainder_of_polynomial_mod_60 (k : ℤ) :
  let n := 60 * k - 3 in (n^3 + 2 * n^2 + 3 * n + 4) % 60 = 46 := by
  sorry

end remainder_of_polynomial_mod_60_l530_530833


namespace predict_sales_volume_correct_l530_530008

-- Define the data points
def data_points : List (ℕ × ℕ) := [(4, 49), (2, 26), (3, 39), (5, 54)]

-- Define the regression slope
def regression_slope : ℝ := 9.4

-- Define the mean of x-values
def mean_x : ℝ := (4 + 2 + 3 + 5) / 4

-- Define the mean of y-values
def mean_y : ℝ := (49 + 26 + 39 + 54) / 4

-- The regression line passes through the mean point
def intercept : ℝ := mean_y - regression_slope * mean_x

-- Define the regression equation
noncomputable def regression_eq : ℝ → ℝ := λ x, regression_slope * x + intercept

-- The advertising cost prediction
def advertising_cost : ℝ := 6

-- Predicted sales volume when the advertising cost is 6 million yuan
def predicted_sales_volume : ℝ := regression_eq advertising_cost

-- The target predicted sales volume
def target_sales_volume : ℝ := 65.5

-- The theorem statement to prove that predicted sales volume is equal to the target sales volume
theorem predict_sales_volume_correct :
  predicted_sales_volume = target_sales_volume := sorry

end predict_sales_volume_correct_l530_530008


namespace prove_angles_equal_l530_530909

-- Define the point locations
def Point := (ℝ × ℝ)

-- Define a square structure and the midpoints
structure Square :=
  (A B C D E F P : Point)
  (is_square : true) -- Placeholder condition stating the shape is a square
  (E_midpoint : E = ((C.1 + B.1) / 2, (C.2 + B.2) / 2))
  (F_midpoint : F = ((D.1 + C.1) / 2, (D.2 + C.2) / 2))
  (P_intersection : ∃ p : Point, lines_intersect (A, E) (B, F) = p ∧ p = P)

-- Statement to prove
theorem prove_angles_equal (sq : Square) : 
  ∠ sq.P sq.D sq.A = ∠ sq.A sq.E sq.D :=
sorry

end prove_angles_equal_l530_530909


namespace expected_adjacent_red_pairs_l530_530118

noncomputable def number_of_red_pairs_in_circle (total_cards : ℕ) (red_cards : ℕ) : ℚ :=
let pairs := total_cards in
let prob_card_red := (red_cards : ℚ) / (total_cards : ℚ) in
let prob_next_red_given_red := (red_cards - 1 : ℚ) / (total_cards - 1 : ℚ) in
let prob_adjacent_red := prob_card_red * prob_next_red_given_red in
pairs * prob_adjacent_red

theorem expected_adjacent_red_pairs :
  number_of_red_pairs_in_circle 51 17 = 464 / 85 := by
  sorry

end expected_adjacent_red_pairs_l530_530118


namespace coin_toss_probability_l530_530084

theorem coin_toss_probability :
  (∀ n : ℕ, 0 ≤ n → n ≤ 10 → (∀ m : ℕ, 0 ≤ m → m = 10 → 
  (∀ k : ℕ, k = 9 → 
  (∀ i : ℕ, 0 ≤ i → i = 10 → ∃ p : ℝ, p = 1/2 → 
  (∃ q : ℝ, q = 1/2 → q = p))))) := 
sorry

end coin_toss_probability_l530_530084


namespace distance_reflection_l530_530296

def euclidean_distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ := 
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

def M : ℝ × ℝ × ℝ := (2, -1, 6)
def M' : ℝ × ℝ × ℝ := (2, 1, -6)

theorem distance_reflection (M : ℝ × ℝ × ℝ) (M' : ℝ × ℝ × ℝ)
  (hx : M.1 = M'.1) (hy : M.2 = -M'.2) (hz : M.3 = -M'.3) :
  euclidean_distance M M' = 2 * real.sqrt 37 := by sorry

end distance_reflection_l530_530296


namespace diagonals_of_30_sided_polygon_l530_530803

theorem diagonals_of_30_sided_polygon : 
  ∀ (n : ℕ), n = 30 → (n * (n - 3)) / 2 = 405 := 
by
  intro n h
  rw h
  simp
  sorry

end diagonals_of_30_sided_polygon_l530_530803


namespace riding_school_horseshoes_l530_530104

def horseshoes_needed_per_horse : ℕ := 4
def horses_per_farm : ℕ := 2
def horses_per_stable : ℕ := 5
def farms : ℕ := 2
def stables : ℕ := 2
def iron_available : ℕ := 400
def iron_per_horseshoe : ℕ := 2

theorem riding_school_horseshoes : 
  let total_horseshoes := iron_available / iron_per_horseshoe in
  let total_farm_horses := farms * horses_per_farm in
  let total_stable_horses := stables * horses_per_stable in
  let total_horses := total_farm_horses + total_stable_horses in
  let horseshoes_needed := total_horses * horseshoes_needed_per_horse in
  let horseshoes_left := total_horseshoes - horseshoes_needed in
  horseshoes_left / horseshoes_needed_per_horse = 36 :=
by
  sorry

end riding_school_horseshoes_l530_530104


namespace part1_part2_l530_530727

def f (x : ℝ) : ℝ := Real.cos x + (1/2) * x^2 - 1

theorem part1 :
  ∃ x0, (∀ x, f x0 ≤ f x) ∧ f x0 = 0 :=
by
  sorry

theorem part2 : 
  ∑ i in Finset.range 2023, Real.sin (i.succ / 2^i.succ) < 2 :=
by
  sorry

end part1_part2_l530_530727


namespace smallest_period_and_intervals_of_monotonic_decrease_area_of_triangle_ABC_l530_530722

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3

noncomputable def g (x : ℝ) : ℝ := f (-2 * x) + 1

theorem smallest_period_and_intervals_of_monotonic_decrease :
  (∃ T > 0, ∀ x, g (x + T) = g x) ∧
  (∀ k : ℤ, ∀ x, (1/2 * k * Real.pi - Real.pi / 24) ≤ x ∧ x ≤ (1/2 * k * Real.pi + 5 * Real.pi / 24) →
    (∀ y, y ∈ (1/2 * k * Real.pi - Real.pi / 24, 1/2 * k * Real.pi + 5 * Real.pi / 24) → g y < g x)) :=
sorry

variables {a b c A B C : ℝ}

axiom sides_and_angles (hA : A/2 - Real.pi/6 = Real.pi/2) (ha : a = 8) (hb_plus_c : Real.sin B + Real.sin C = 13 * Real.sqrt 3 / 16) : b + c = 13

theorem area_of_triangle_ABC (hA : f (A / 2 - Real.pi / 6) = Real.sqrt 3) (ha : a = 8) 
  (hb_plus_c : Real.sin B + Real.sin C = 13 * Real.sqrt 3 / 16) :
  1/2 * b * c * Real.sin A = 35 * Real.sqrt 3 / 4 :=
sorry

end smallest_period_and_intervals_of_monotonic_decrease_area_of_triangle_ABC_l530_530722


namespace base_angle_isosceles_triangle_l530_530276

theorem base_angle_isosceles_triangle (T : Type) [EuclideanGeometry T] (A B C : T) (hIso : IsoscelesTriangle A B C) (hAngle : ∠BAC = 80) :
  ∠ABC ∈ {80, 50} :=
by
  sorry

end base_angle_isosceles_triangle_l530_530276


namespace diagonals_in_convex_polygon_with_30_sides_l530_530764

theorem diagonals_in_convex_polygon_with_30_sides : 
  let n := 30 in
  ∑ i j in finset.range n, (i + 1 - 1) * (j + 1 - 1) * (nat.choose 30 2 - 30) / 2 = 202 := 
by
  sorry

end diagonals_in_convex_polygon_with_30_sides_l530_530764


namespace number_of_diagonals_30_sides_l530_530780

def number_of_diagonals (n : ℕ) : ℕ :=
  nat.choose n 2 - n

theorem number_of_diagonals_30_sides :
  number_of_diagonals 30 = 405 :=
by {
  rw [number_of_diagonals, nat.choose, nat.factorial, nat.factorial, nat.factorial],
  -- The proof would proceed to simplify the combination and subtraction, but we use sorry to skip details.
  sorry,
}

end number_of_diagonals_30_sides_l530_530780


namespace alan_total_cost_l530_530583

theorem alan_total_cost :
  let price_AVN_CD := 12 in
  let price_The_Dark_CD := price_AVN_CD * 2 in
  let total_cost_The_Dark_CDs := 2 * price_The_Dark_CD in
  let total_cost_before_90s_CDs := price_AVN_CD + total_cost_The_Dark_CDs in
  let cost_90s_CDs := 0.4 * total_cost_before_90s_CDs in
  let total_cost := total_cost_before_90s_CDs + cost_90s_CDs in
  total_cost = 84 :=
by
  let price_AVN_CD := 12
  let price_The_Dark_CD := price_AVN_CD * 2
  let total_cost_The_Dark_CDs := 2 * price_The_Dark_CD
  let total_cost_before_90s_CDs := price_AVN_CD + total_cost_The_Dark_CDs
  let cost_90s_CDs := 0.4 * total_cost_before_90s_CDs
  let total_cost := total_cost_before_90s_CDs + cost_90s_CDs
  show total_cost = 84, from sorry

end alan_total_cost_l530_530583


namespace probability_sum_is_five_probability_in_region_Ω_l530_530919

-- Definitions
def outcomes := {(x, y) | x ∈ {1, 2, 3, 4, 5, 6} ∧ y ∈ {1, 2, 3, 4, 5, 6}}

-- Probability question: Probability that the sum of the two numbers is 5
def event_A := {p ∈ outcomes | p.1 + p.2 = 5}
def P_A := (event_A).card.toReal / (outcomes).card.toReal

theorem probability_sum_is_five :
  P_A = 1 / 9 :=
sorry

-- Region defined by conditions for second probability question
def region_Ω := {(x, y) | x ∈ {1, 2, 3, 4, 5, 6} ∧ y ∈ {1, 2, 3, 4, 5, 6} ∧ x > 0 ∧ y > 0 ∧ x - y - 2 > 0}

-- Probability question: Probability point is in region Ω
def event_B := {(x, y) | (x, y) ∈ region_Ω}
def P_B := (event_B).card.toReal / (outcomes).card.toReal

theorem probability_in_region_Ω :
  P_B = 1 / 6 :=
sorry

end probability_sum_is_five_probability_in_region_Ω_l530_530919


namespace min_value_a_l530_530709

noncomputable def ellipse_min_a (a b c : ℝ) (F1 F2 P G : ℝ × ℝ) (λ : ℝ) : Prop :=
  let ellipse_eq := (P.1^2) / (a^2) + (P.2^2) / (b^2) = 1 in
  let foci_positions := F1 = (-c, 0) ∧ F2 = (c, 0) in
  let positive_half_y := G.1 = 0 ∧ G.2 > 0 in
  let circumcenter := True in /- circumcenter condition is abstract, considered as true -/
  let vector_eq := (G.1 - F1.1, G.2 - F1.2) + (G.1 - F2.1, G.2 - F2.2) + λ * (G.1 - P.1, G.2 - P.2) = (0, 0) in
  let area_triangle := 1/2 * b * 2 * c = 8 in
  let min_a_condition := a ≥ 4 in
  ellipse_eq ∧ foci_positions ∧ positive_half_y ∧ circumcenter ∧ vector_eq ∧ area_triangle → min_a_condition

theorem min_value_a (a b c : ℝ) (F1 F2 P G : ℝ × ℝ) (λ : ℝ) (h : ellipse_min_a a b c F1 F2 P G λ) : a ≥ 4 :=
sorry

end min_value_a_l530_530709


namespace neither_factorial_tail_nor_double_factorial_tail_l530_530658

def is_factorial_tail (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 0 ∧ (λ m, (m / 5) + (m / 25) + (m / 125) + (m / 625) + ...) m = n

def is_double_factorial_tail (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 0 ∧ (λ m, (m / 5) + (m / 25) + (m / 125) + (m / 625) + ...) m = 2 * n

theorem neither_factorial_tail_nor_double_factorial_tail (N : ℕ) (hN : N < 1000) :
  (¬ is_factorial_tail N ∧ ¬ is_double_factorial_tail N) → N = 1 :=
sorry

end neither_factorial_tail_nor_double_factorial_tail_l530_530658


namespace red_blood_cell_scientific_notation_l530_530916

theorem red_blood_cell_scientific_notation :
  (0.0000078 : ℝ) = 7.8 * 10^(-6) :=
sorry

end red_blood_cell_scientific_notation_l530_530916


namespace find_matrix_X_l530_530198

theorem find_matrix_X :
  ∀ (p q r s t u v w x : ℚ),
  let X := ![![p, q, r], ![s, t, u], ![v, w, x]],
      A := ![![2, -1, 0], ![-3, 5, 0], ![0, 0, 2]],
      I := ![![1, 0, 0], ![0, 1, 0], ![0, 0, 1]])
  in X ⬝ A = I →
     X = ![![5/7, 1/7, 0], ![3/7, 2/7, 0], ![0, 0, 1/2]] :=
by
  sorry

end find_matrix_X_l530_530198


namespace total_stoppage_time_correct_l530_530110

def stoppage_times : List ℕ := [5, 8, 10]      -- Rest points (5 minutes, 8 minutes, 10 minutes)
def school_zone_time : ℕ := 2 * 5             -- Two school zones, each 5 minutes
def construction_site_time : ℕ := 15          -- Construction site, 15 minutes

def total_stoppage_time : ℕ := List.sum stoppage_times + school_zone_time + construction_site_time

theorem total_stoppage_time_correct : total_stoppage_time = 48 := by
  sorry

end total_stoppage_time_correct_l530_530110


namespace find_f_sqrt_two_l530_530733

theorem find_f_sqrt_two (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x - 1) = x^2 - 2 * x) : f (real.sqrt 2) = 1 := 
by
  sorry

end find_f_sqrt_two_l530_530733


namespace num_ways_pay_l530_530874

theorem num_ways_pay : 
  let n : ℕ := 2010,
      num_solutions : ℕ := (Nat.choose (201 + 2) 2)
  in ∑ x y z in { (x' : ℕ) }, 2 * x' + 5 * y' + 10 * z' = 2010 = num_solutions :=
by
  sorry

end num_ways_pay_l530_530874


namespace initial_oranges_count_l530_530389

theorem initial_oranges_count (O : ℕ) (H1 : 14 / (14 + (O - 14)) = 0.7) : O = 20 := 
by
  sorry

end initial_oranges_count_l530_530389


namespace find_total_boys_l530_530853

noncomputable def total_boys (B : ℕ) : Prop :=
  let percentage_other := 100 - (44 + 28 + 10) in
  let boys_other := B * percentage_other / 100 in
  boys_other = 153

theorem find_total_boys : ∃ B : ℕ, total_boys B ∧ B = 850 :=
by {
  use 850,
  unfold total_boys,
  -- By calculations, we have:
  -- percentage_other = 18
  -- boys_other = 850 * 18 / 100 = 153
  sorry
}

end find_total_boys_l530_530853


namespace tank_a_height_l530_530003

theorem tank_a_height (h_B : ℝ) (C_A C_B : ℝ) (V_A : ℝ → ℝ) (V_B : ℝ) :
  C_A = 4 ∧ C_B = 10 ∧ h_B = 8 ∧ (∀ h_A : ℝ, V_A h_A = 0.10000000000000002 * V_B) →
  ∃ h_A : ℝ, h_A = 5 :=
by sorry

end tank_a_height_l530_530003


namespace sam_catches_up_alice_in_30_minutes_l530_530590

def alice_speed : ℝ := 10
def sam_speed : ℝ := 16
def initial_distance : ℝ := 3

theorem sam_catches_up_alice_in_30_minutes :
  let relative_speed := sam_speed - alice_speed,
      time_hours := initial_distance / relative_speed,
      time_minutes := time_hours * 60
  in time_minutes = 30 :=
by
  sorry

end sam_catches_up_alice_in_30_minutes_l530_530590


namespace sine_180_eq_zero_l530_530625

theorem sine_180_eq_zero :
  sin (180 : ℝ) = 0 :=
sorry

end sine_180_eq_zero_l530_530625


namespace max_slope_OM_l530_530885

-- Define the problem-specific conditions
variables (p : ℝ) (t : ℝ) (h_p : p > 0)

-- Define the point P on the parabola
def P := (2 * p * t, 2 * p * t^2)

-- Define the focus F of the parabola
def F := (p / 2, 0)

-- Define the point M according to the given condition |PM| = 2|MF|
def M := (2 * p / 3 * t^2 + p / 3, 2 * p / 3 * t)

-- Calculate the slope of the line OM
def slope_OM := (2 * p * t / 3) / (2 * p / 3 * t^2 + p / 3)

-- Prove the maximum slope is sqrt(2)/2
theorem max_slope_OM : 
    ∃ (t : ℝ), slope_OM p t = (Real.sqrt 2) / 2 :=
begin
    sorry
end

end max_slope_OM_l530_530885


namespace horseshoes_riding_school_l530_530105

theorem horseshoes_riding_school (iron_total : ℕ) (iron_per_horseshoe : ℕ)
    (farms : ℕ) (horses_per_farm : ℕ) 
    (stables : ℕ) (horses_per_stable : ℕ)
    (hooves_per_horse : ℕ) :
    iron_total = 400 →
    iron_per_horseshoe = 2 →
    farms = 2 →
    horses_per_farm = 2 →
    stables = 2 →
    horses_per_stable = 5 →
    hooves_per_horse = 4 →
    36 = (iron_total - 
        ((farms * horses_per_farm * hooves_per_horse + 
          stables * horses_per_stable * hooves_per_horse) * iron_per_horseshoe)) / 
        iron_per_horseshoe / hooves_per_horse :=
by
    intros h1 h2 h3 h4 h5 h6 h7
    rw [h1, h2, h3, h4, h5, h6, h7]
    simp
    sorry

end horseshoes_riding_school_l530_530105


namespace ratio_length_to_breadth_l530_530396

theorem ratio_length_to_breadth (b l : ℕ) (A : ℕ) (h1 : b = 30) (h2 : A = 2700) (h3 : A = l * b) :
  l / b = 3 :=
by sorry

end ratio_length_to_breadth_l530_530396


namespace find_valid_pairs_l530_530671

def is_integer (n : ℤ) : Prop := ∃ k : ℤ, n = k

noncomputable def valid_pairs (a b : ℕ) :=
a > 0 ∧ b > 0 ∧ 
is_integer (a^2 + b : ℤ) / (b^2 - a : ℤ) ∧
is_integer (b^2 + a : ℤ) / (a^2 - b : ℤ)

theorem find_valid_pairs :
  {ab : (ℕ × ℕ) | valid_pairs ab.1 ab.2} = {⟨2, 2⟩, ⟨3, 3⟩, ⟨1, 2⟩, ⟨2, 3⟩} :=
sorry

end find_valid_pairs_l530_530671


namespace well_depth_l530_530043

theorem well_depth (e x a b c d : ℝ)
  (h1 : x = 2 * a + b)
  (h2 : x = 3 * b + c)
  (h3 : x = 4 * c + d)
  (h4 : x = 5 * d + e)
  (h5 : x = 6 * e + a) :
  x = 721 / 76 * e ∧
  a = 265 / 76 * e ∧
  b = 191 / 76 * e ∧
  c = 37 / 19 * e ∧
  d = 129 / 76 * e :=
sorry

end well_depth_l530_530043


namespace exists_perpendicular_in_plane_l530_530702

theorem exists_perpendicular_in_plane (a : ℝ → aff ℝ^3) (α : affine_subspace ℝ ℝ^3) :
  ∃ b : ℝ → aff ℝ^3, b ∈ α ∧ perp a b :=
sorry

end exists_perpendicular_in_plane_l530_530702


namespace ellipsoid_volume_in_box_l530_530871

noncomputable def internalEllipsoidVolume (box_length box_width box_height wall_thickness : ℝ) : ℝ :=
  let internal_length := box_length - 2 * wall_thickness
  let internal_width  := box_width - 2 * wall_thickness
  let internal_height := box_height - 2 * wall_thickness
  let semi_major_axis := internal_length / 2
  let semi_minor_axis_1 := internal_width / 2
  let semi_minor_axis_2 := internal_height / 2
  let volume_cubic_inches := (4 / 3) * Real.pi * semi_major_axis * semi_minor_axis_1 * semi_minor_axis_2
  let cubic_inches_to_cubic_feet := 1 / 1728
  let volume_cubic_feet := volume_cubic_inches * cubic_inches_to_cubic_feet
  volume_cubic_feet

theorem ellipsoid_volume_in_box : 
  internalEllipsoidVolume 26 26 14 1 ≈ 2.0944 := 
by 
  sorry

end ellipsoid_volume_in_box_l530_530871


namespace find_150th_letter_in_pattern_l530_530448

/--
Given the repeating pattern "XYZ" with a cycle length of 3,
prove that the 150th letter in this pattern is 'Z'.
-/
theorem find_150th_letter_in_pattern : 
  let pattern := "XYZ"
  let cycle_length := String.length pattern
in (150 % cycle_length = 0) → "Z" := 
sorry

end find_150th_letter_in_pattern_l530_530448


namespace sum_of_three_equal_expressions_l530_530859

-- Definitions of variables and conditions
variables (a b c d e f g h i S : ℤ)
variable (ha : a = 4)
variable (hg : g = 13)
variable (hh : h = 6)
variable (heq1 : a + b + c + d = S)
variable (heq2 : d + e + f + g = S)
variable (heq3 : g + h + i = S)

-- Main statement we want to prove
theorem sum_of_three_equal_expressions : S = 19 + i :=
by
  -- substitution steps and equality reasoning would be carried out here
  sorry

end sum_of_three_equal_expressions_l530_530859


namespace convex_polygon_diagonals_l530_530782

theorem convex_polygon_diagonals (n : ℕ) (h : n = 30) : 
  let total_pairs := (n * (n - 1)) / 2 in (total_pairs - n) = 405 := 
by
  simp only [total_pairs]
  sorry

end convex_polygon_diagonals_l530_530782


namespace coefficient_of_x2_in_expansion_l530_530860

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
if k > n then 0 else Nat.choose n k

theorem coefficient_of_x2_in_expansion :
  let a := 1
  let b := 2
  let n := 10
  binomial_coefficient n 2 * (b^2) = 180 :=
by
  let a := 1
  let b := 2
  let n := 10
  have h1 : binomial_coefficient n 2 = Nat.choose 10 2 := rfl
  have h2 : b^2 = 4 := rfl
  sorry

end coefficient_of_x2_in_expansion_l530_530860


namespace simplify_expression_l530_530992

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x^(-2) + y^(-2))^(-1) = (x^2 * y^2) / (x^2 + y^2) :=
by
  sorry

end simplify_expression_l530_530992


namespace letter_150_is_Z_l530_530504

/-- Definition of the repeating pattern "XYZ" -/
def pattern : List Char := ['X', 'Y', 'Z']

/-- The repeating pattern has a length of 3 -/
def pattern_length : ℕ := 3

/-- Calculate the 150th letter in the repeating pattern "XYZ" -/
def nth_letter_in_pattern (n : ℕ) : Char :=
  let m := n % pattern_length
  if m = 0 then pattern[2] else pattern[m - 1]

/-- Prove that the 150th letter in the pattern "XYZ" is 'Z' -/
theorem letter_150_is_Z : nth_letter_in_pattern 150 = 'Z' :=
by
  sorry

end letter_150_is_Z_l530_530504


namespace sequence_problems_l530_530232

theorem sequence_problems
  (a : ℕ → ℤ)
  (b : ℕ → ℤ)
  (A : ℕ → ℤ := λ n, (-2 : ℤ) * n + 3)
  (B : ℕ → ℤ := λ n, (2 : ℤ)^n + (2 : ℤ) * n - 3)
  (S : ℕ → ℤ := λ n, 2^(n + 1) + n^2 - 2 * n - 2) :
  (a 1 = 1) →
  (a 4 = -5) →
  (b 1 = 1) →
  (b 4 = 21) →
  (∀ n : ℕ, a n + b n = 2^(n : ℤ)) →
  (∀ n, a n = A n) ∧ (∀ n, b n = B n) ∧ (∀ n, ∑ i in Finset.range n, b (i + 1) = S n) :=
by
  intros h1 h2 h3 h4 h5
  split
  sorry
  split
  sorry
  sorry

end sequence_problems_l530_530232


namespace find_150th_letter_l530_530451
open Nat

def repeating_sequence := "XYZ"

def length_repeating_sequence := 3

theorem find_150th_letter : (150 % length_repeating_sequence == 0) → repeating_sequence[(length_repeating_sequence - 1) % length_repeating_sequence] = 'Z' := 
by
  sorry

end find_150th_letter_l530_530451


namespace correctness_statement_l530_530227

-- Given points A, B, C are on the specific parabola
variable (a c x1 x2 x3 y1 y2 y3 : ℝ)
variable (ha : a < 0) -- a < 0 since the parabola opens upwards
variable (hA : y1 = - (a / 4) * x1^2 + a * x1 + c)
variable (hB : y2 = a + c) -- B is the vertex
variable (hC : y3 = - (a / 4) * x3^2 + a * x3 + c)
variable (hOrder : y1 > y3 ∧ y3 ≥ y2)

theorem correctness_statement : abs (x1 - x2) > abs (x3 - x2) :=
sorry

end correctness_statement_l530_530227


namespace value_of_y_l530_530168

theorem value_of_y : ∃ y : ℤ, 3 * y - 6 = | -20 + 5 | ∧ y = 7 := by
  use 7
  rw Int.abs_of_nonneg (by norm_num)
  norm_num
  sorry

end value_of_y_l530_530168


namespace sin_180_eq_0_l530_530652

theorem sin_180_eq_0 : Real.sin (180 * Real.pi / 180) = 0 :=
by
  sorry

end sin_180_eq_0_l530_530652


namespace number_of_diagonals_in_30_sided_polygon_l530_530758

-- Definition of a convex polygon with 30 sides
def convex_polygon_sides := 30

-- The function to calculate the number of diagonals in a convex polygon with n sides
noncomputable def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Prove that a convex polygon with 30 sides has 405 diagonals
theorem number_of_diagonals_in_30_sided_polygon :
  num_diagonals convex_polygon_sides = 405 :=
by
  calc
    num_diagonals 30 = 30 * (30 - 3) / 2 : rfl
    ... = 30 * 27 / 2 : rfl
    ... = 810 / 2 : rfl
    ... = 405 : rfl

end number_of_diagonals_in_30_sided_polygon_l530_530758


namespace angle_fold_paper_l530_530128

theorem angle_fold_paper
  (AB CD: Type)
  (D C F E: AB)
  (AB: line AB)
  (angle1: Real := 22)
  (H1: is_right_angle (angle D E F))
  (H2: angle1 = 22) : 
  angle D F E = 44 :=
by 
  sorry

end angle_fold_paper_l530_530128


namespace perfect_cubes_as_diff_consecutive_squares_l530_530828

theorem perfect_cubes_as_diff_consecutive_squares :
  { c : ℕ // c^3 < 20000 ∧ (∃ b : ℕ, c^3 = 2*b + 1) }.to_finset.card = 14 := by
sorry

end perfect_cubes_as_diff_consecutive_squares_l530_530828


namespace angle_value_l530_530705

theorem angle_value (α : ℝ) (h1 : 0 ≤ α) (h2 : α < 360) 
(h3 : (Real.sin 215 * π / 180, Real.cos 215 * π / 180) = (Real.sin α, Real.cos α)) :
α = 235 :=
sorry

end angle_value_l530_530705


namespace sin_180_is_zero_l530_530644

noncomputable def sin_180_eq_zero : Prop :=
  let point_zero_deg := (1, 0)
  let point_180_deg := (-1, 0)
  let y_coord_of_180_deg := 0
  sin 180 = y_coord_of_180_deg

theorem sin_180_is_zero : sin_180_eq_zero :=
  sorry

end sin_180_is_zero_l530_530644


namespace term_containing_x3_is_seventh_term_l530_530037

noncomputable def binomial_expansion (n : ℕ) (k : ℕ) (a b : ℝ) : ℝ :=
  Nat.choose n k * a^(n - k) * b^k 

theorem term_containing_x3_is_seventh_term 
  (n : ℕ) (a b : ℝ) (r : ℕ) (x : ℝ) :
  n = 16 ∧ a = sqrt x ∧ b = 2 / cbrt x ∧ binomial_expansion n r a b * x^(n - r) * x^(-5 * r / 6) = x^3 →
  r = 6 :=
sorry

end term_containing_x3_is_seventh_term_l530_530037


namespace normal_distribution_probability_l530_530719

noncomputable def standard_normal : ProbabilityDistribution := sorry -- Placeholder for the standard normal distribution

theorem normal_distribution_probability :
  let ξ := standard_normal in
  (∀ x : ℝ, CDF ξ x = if x ≤ 1 then 0.8413 else sorry) →
  P(ξ > 1) = 0.1587 →
  P(-1 < ξ ∧ ξ ≤ 0) = 0.3413 :=
sorry

end normal_distribution_probability_l530_530719


namespace sum_of_even_theta_angles_l530_530963

theorem sum_of_even_theta_angles 
  (n : ℕ)
  (z : ℂ)
  (hz : z^35 - z^7 - 1 = 0)
  (h_abs_z : abs z = 1)
  (theta : Fin (2 * n) → ℝ)
  (h_theta_form : ∀ m : Fin (2 * n), z^(m : ℕ) = complex.exp (complex.I * theta m) )
  (h_theta_range : ∀ i j : Fin (2 * n), i < j → theta i < theta j)
  (h_theta_mod : ∀ m : Fin (2 * n), 0 ≤ theta m ∧ theta m < 360) :
  theta 1 + theta 3 + theta 5 + ⋯ + theta (2 * n - 1) = 925.714 :=
sorry

end sum_of_even_theta_angles_l530_530963


namespace max_value_M_l530_530259

open Real

theorem max_value_M :
  ∃ M : ℝ, ∀ x y z u : ℝ, 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < u ∧ z ≥ y ∧ (x - 2 * y = z - 2 * u) ∧ (2 * y * z = u * x) →
  M ≤ z / y ∧ M = 6 + 4 * sqrt 2 := 
  sorry

end max_value_M_l530_530259


namespace BD_condition_l530_530440

noncomputable def find_BD (A B C D B' C' : Point) (AB AC BC : ℝ) (hab : distance A B = 9) (hac : distance A C = 10) (hbc : distance B C = 12) (hD : PointOnLine D B C) (hBb' : Reflected B A D = B') (hCc' : Reflected C A D = C') (h_par : LinesParallel BC' B'C) : Prop :=
  distance B D = 6

theorem BD_condition {A B C D B' C' : Point} (hab : distance A B = 9) (hac : distance A C = 10) (hbc : distance B C = 12) (hD : PointOnLine D B C) (hBb' : Reflected B A D = B') (hCc' : Reflected C A D = C') (h_par : LinesParallel BC' B'C) : find_BD A B C D B' C' 9 hab hac hbc hD hBb' hCc' h_par :=
sorry

end BD_condition_l530_530440


namespace distance_between_points_l530_530674

theorem distance_between_points :
  let p1 := (3 : ℝ, -2 : ℝ, 5 : ℝ)
  let p2 := (-1 : ℝ, 4 : ℝ, 2 : ℝ)
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2 + (p2.3 - p1.3) ^ 2)
  = real.sqrt 61 :=
by
  sorry

end distance_between_points_l530_530674


namespace symmetric_points_l530_530371

theorem symmetric_points (a b : ℝ) (h1 : 2 * a + 1 = -1) (h2 : 4 = -(3 * b - 1)) :
  2 * a + b = -3 := 
sorry

end symmetric_points_l530_530371


namespace min_value_expr_sum_of_squares_inequality_l530_530713

-- Given conditions
variables (a b : ℝ)
variables (ha : a > 0) (hb : b > 0) (hab : a + b = 2)

-- Problem (1): Prove minimum value of (2 / a + 8 / b) is 9
theorem min_value_expr : ∃ a b, a > 0 ∧ b > 0 ∧ a + b = 2 ∧ ((2 / a) + (8 / b) = 9) := sorry

-- Problem (2): Prove a^2 + b^2 ≥ 2
theorem sum_of_squares_inequality : a^2 + b^2 ≥ 2 :=
by { sorry }

end min_value_expr_sum_of_squares_inequality_l530_530713


namespace exactly_two_tetrahedra_l530_530969

-- Define the initial conditions: three non-coplanar segments intersecting at O, dividing each in half.
variables {O : Type} [Point O]
variables (S1 S2 S3 : Segment O) 

-- Assuming the segments are non-coplanar and intersect at O and O halves each
axiom non_coplanar : ¬ coplanar S1 S2 S3
axiom intersection_at_O : intersects_at S1 S2 S3 O
axiom divide_in_half : ∀ (S : Segment O), midpoint O S

-- The statement we want to prove
theorem exactly_two_tetrahedra : ∃ (T1 T2 : Tetrahedron O), connects_midpoints_opposite_edges S1 S2 S3 T1 ∧ connects_midpoints_opposite_edges S1 S2 S3 T2 ∧ T1 ≠ T2 :=
begin
  sorry
end

end exactly_two_tetrahedra_l530_530969


namespace inverse_of_matrix_A_l530_530173

theorem inverse_of_matrix_A :
  let A := ![
    ![2, 5, 6],
    ![1, 2, 5],
    ![1, 2, 3]
  ]
  A⁻¹ = ![
    ![-2, 3/2, 13/2],
    ![1, 0, 2],
    ![0, -1/2, -1/2]
  ] :=
by
  let A := ![
    ![2, 5, 6],
    ![1, 2, 5],
    ![1, 2, 3]
  ]
  let A_inv := ![
    ![-2, 3/2, 13/2],
    ![1, 0, 2],
    ![0, -1/2, -1/2]
  ]
  -- Proof would be provided here
  sorry

end inverse_of_matrix_A_l530_530173


namespace neil_final_num_3_l530_530869

-- Assuming the conditions in the problem as definitions
noncomputable def die_prob := (1/3 : ℚ)

noncomputable def prob_neil_final_3 (prob_jerry_roll_1 : ℚ) (prob_jerry_roll_2 : ℚ) (prob_jerry_roll_3 : ℚ) :=
  (prob_jerry_roll_1 * die_prob) + (prob_jerry_roll_2 * (1/2)) + (prob_jerry_roll_3 * 1)

theorem neil_final_num_3 :
  let prob_jerry_roll := die_prob in
  (prob_jerry_roll * die_prob) + (prob_jerry_roll * (1/2)) + (prob_jerry_roll * 1) = 11/18 :=
by
  let prob_jerry_roll := die_prob
  let partial_prob_1 := prob_jerry_roll * die_prob
  let partial_prob_2 := prob_jerry_roll * (1/2)
  let partial_prob_3 := prob_jerry_roll * 1
  let total_prob := partial_prob_1 + partial_prob_2 + partial_prob_3
  show total_prob = 11/18
  sorry

end neil_final_num_3_l530_530869


namespace maximum_value_frac_l530_530714

/-- For all positive real numbers x, y, and z, the maximum value of (xy + yz) / (x^2 + y^2 + z^2) is sqrt(2)/2 -/
theorem maximum_value_frac (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (xy + yz) / (x^2 + y^2 + z^2) ≤ sqrt(2)/2 :=
sorry

end maximum_value_frac_l530_530714


namespace line_fixed_point_l530_530562

theorem line_fixed_point (m : ℝ) : ∃ x y, (∀ m, y = m * x + (2 * m + 1)) ↔ (x = -2 ∧ y = 1) :=
by
  sorry

end line_fixed_point_l530_530562


namespace sum_of_squares_of_solutions_l530_530206

theorem sum_of_squares_of_solutions :
  let a := (1 : ℚ) / 2010 
  let b := (1 : ℚ) / 1005
  ∑ (s : ℚ) in { x | abs (x ^ 2 - x + a) = a }.to_finset, s ^ 2 = (2008 : ℚ) / 1005 :=
by
  sorry

end sum_of_squares_of_solutions_l530_530206


namespace find_150th_letter_l530_530452
open Nat

def repeating_sequence := "XYZ"

def length_repeating_sequence := 3

theorem find_150th_letter : (150 % length_repeating_sequence == 0) → repeating_sequence[(length_repeating_sequence - 1) % length_repeating_sequence] = 'Z' := 
by
  sorry

end find_150th_letter_l530_530452


namespace base_h_addition_eq_l530_530662

theorem base_h_addition_eq (h : ℕ) :
  let n1 := 7 * h^3 + 3 * h^2 + 6 * h + 4
  let n2 := 8 * h^3 + 4 * h^2 + 2 * h + 1
  let sum := 1 * h^4 + 7 * h^3 + 2 * h^2 + 8 * h + 5
  n1 + n2 = sum → h = 8 :=
by
  intros n1 n2 sum h_eq
  sorry

end base_h_addition_eq_l530_530662


namespace num_elements_with_8_as_first_digit_l530_530336

noncomputable def S : set ℕ := {k | ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 3000 ∧ 8^n = k}

theorem num_elements_with_8_as_first_digit : 
  (∃ f : ℕ → ℕ, ∀ k ∈ S, f k = 291 ∧ first_digit (8^k) = 8) := 
sorry

end num_elements_with_8_as_first_digit_l530_530336


namespace sin_180_eq_0_l530_530653

theorem sin_180_eq_0 : Real.sin (180 * Real.pi / 180) = 0 :=
by
  sorry

end sin_180_eq_0_l530_530653


namespace regular_polygon_interior_angle_sum_l530_530035

theorem regular_polygon_interior_angle_sum (n : ℕ) :
  (n - 2) * 180 = 4 * 360 → n = 10 ∧ (n - 2) * 180 / n = 144 :=
by
  intro h
  have hn : n = 10,
  { 
    calc
      (n - 2) * 180 = 4 * 360 : h
      (n - 2) * 180 = 1440 : by ring
      n - 2 = 8 : by linarith
      n = 8 + 2 : by ring
      n = 10 : by linarith },
  use hn,
  have h2 : (10 - 2) * 180 / 10 = 144,
  { 
    calc (10 - 2) * 180 / 10 = 8 * 180 / 10 : by ring
    = 144 : by norm_num },
  use h2,
  sorry

end regular_polygon_interior_angle_sum_l530_530035


namespace start_page_day2_correct_l530_530107

variables (total_pages : ℕ) (percentage_read_day1 : ℝ) (start_page_day2 : ℕ)

theorem start_page_day2_correct
  (h1 : total_pages = 200)
  (h2 : percentage_read_day1 = 0.2)
  : start_page_day2 = total_pages * percentage_read_day1 + 1 :=
by
  sorry

end start_page_day2_correct_l530_530107


namespace sin_cos_60_degrees_l530_530654

theorem sin_cos_60_degrees :
  sin (60 * Real.pi / 180) = (Real.sqrt 3 / 2) ∧ cos (60 * Real.pi / 180) = (1 / 2) :=
by
  -- proof omitted
  sorry

end sin_cos_60_degrees_l530_530654


namespace largest_n_lemma_l530_530193
noncomputable def largest_n (n b: ℕ) : Prop := 
  n! = ((n-4) + b)!/(b!)

theorem largest_n_lemma : ∀ n: ℕ, ∀ b: ℕ, b ≥ 4 → (largest_n n b → n = 1) := 
  by
  intros n b hb h
  sorry

end largest_n_lemma_l530_530193


namespace decryption_probabilities_l530_530436

-- Definitions based on conditions
variables {A B : Type}
variables (p1 p2 : ℝ)
variables (X Y : ℕ)

-- Assumptions and statement of proof
theorem decryption_probabilities (h1 : 0 < p1) (h2 : p1 < p2) (h3 : p2 < 1/2)
  (hX : X ~ (Binomial 3 p1)) (hY : Y ~ (Binomial 3 p2)) :
  (3 * p1 < 3 * p2 ∧ 3 * p1 * (1 - p1) < 3 * p2 * (1 - p2)) :=
by
  -- Insert formal proof steps here
  sorry

end decryption_probabilities_l530_530436


namespace decimal_digits_first_three_l530_530979

def first_three_digits_decimal (x : ℝ) : ℕ :=
  let fractional_part := x - x.toInt
  let shifted := fractional_part * 1000
  shifted.toInt

theorem decimal_digits_first_three
  (x : ℝ)
  (h : x = (10 ^ 2003 + 1) ^ (11 / 9))
  : first_three_digits_decimal x = 222 :=
by
  sorry

end decimal_digits_first_three_l530_530979


namespace sin_180_eq_0_l530_530649

theorem sin_180_eq_0 : Real.sin (180 * Real.pi / 180) = 0 :=
by
  sorry

end sin_180_eq_0_l530_530649


namespace solve_for_a_and_b_l530_530267

theorem solve_for_a_and_b (a b : ℤ) (h1 : 5 + a = 6 - b) (h2 : 6 + b = 9 + a) : 5 - a = 6 := 
sorry

end solve_for_a_and_b_l530_530267


namespace XY_XZ_squared_l530_530310

-- Defining the setup of the triangle.
variables {X Y Z M : Type*}
variables {YZ_length XM_length : ℝ}
variable (is_median : ∀ {Y Z M: ℝ} (YZ := 10) (XM := 7) (M := (Y + Z) / 2))

-- We want to prove the given condition.
theorem XY_XZ_squared (YZ_length = 10) (XM_length = 7) :
  M = midpoint Y Z → XY^2 + XZ^2 = 148 :=
begin
  sorry
end

end XY_XZ_squared_l530_530310


namespace number_of_diagonals_in_30_sided_polygon_l530_530755

-- Definition of a convex polygon with 30 sides
def convex_polygon_sides := 30

-- The function to calculate the number of diagonals in a convex polygon with n sides
noncomputable def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Prove that a convex polygon with 30 sides has 405 diagonals
theorem number_of_diagonals_in_30_sided_polygon :
  num_diagonals convex_polygon_sides = 405 :=
by
  calc
    num_diagonals 30 = 30 * (30 - 3) / 2 : rfl
    ... = 30 * 27 / 2 : rfl
    ... = 810 / 2 : rfl
    ... = 405 : rfl

end number_of_diagonals_in_30_sided_polygon_l530_530755


namespace diagonals_of_30_sided_polygon_l530_530802

theorem diagonals_of_30_sided_polygon : 
  ∀ (n : ℕ), n = 30 → (n * (n - 3)) / 2 = 405 := 
by
  intro n h
  rw h
  simp
  sorry

end diagonals_of_30_sided_polygon_l530_530802


namespace simplify_trig_expr_l530_530381

theorem simplify_trig_expr (B : ℝ) (h : sin B ≠ 0 ∧ cos B ≠ 0) :
  (2 + 2 * (cos B / sin B) - 3 * (1 / sin B)) * (3 + (sin B / cos B) + 2 * (1 / cos B)) = 2 :=
by
  -- Additional helper conditions to assist Lean in calculations
  have h1 : sin B ≠ 0 := h.1,
  have h2 : cos B ≠ 0 := h.2,
  have h3 : sin B^2 + cos B^2 = 1 := by sorry, 
  sorry

end simplify_trig_expr_l530_530381


namespace storks_more_than_birds_l530_530542

def initial_birds := 2
def additional_birds := 3
def total_birds := initial_birds + additional_birds
def storks := 6
def difference := storks - total_birds

theorem storks_more_than_birds : difference = 1 :=
by
  sorry

end storks_more_than_birds_l530_530542


namespace length_of_chord_AB_l530_530863

noncomputable def circle_polar (ρ θ : ℝ) : Prop :=
  ρ = 4 * Real.sqrt 2 * Real.cos (θ - Real.pi / 4)

noncomputable def line_parametric (t : ℝ) : ℝ × ℝ :=
  (t + 1, t - 1)

noncomputable def circle_cartesian (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 4*y = 0

noncomputable def chord_length (AB : ℝ) : Prop :=
  AB = 2 * Real.sqrt 6

theorem length_of_chord_AB :
  (∃ ρ θ, circle_polar ρ θ) ∧
  (∃ t, (line_parametric t).fst = line_parametric t.snd) →
  chord_length (2 * Real.sqrt 6) :=
by
  sorry

end length_of_chord_AB_l530_530863


namespace sum_of_remainders_eq_24_l530_530996

theorem sum_of_remainders_eq_24 (a b c : ℕ) 
  (h1 : a % 30 = 13) (h2 : b % 30 = 19) (h3 : c % 30 = 22) :
  (a + b + c) % 30 = 24 :=
by
  sorry

end sum_of_remainders_eq_24_l530_530996


namespace cos_identity_solution_l530_530927

theorem cos_identity_solution (x : ℝ) (n : ℤ) :  
  (8 * (cos x) ^ 5 - 5 * cos x - 2 * cos (3 * x) = 1) → (∃ n : ℤ, x = 2 * π * n) :=
by
  sorry

end cos_identity_solution_l530_530927


namespace least_prime_factor_of_expression_l530_530983

theorem least_prime_factor_of_expression : ∃ p : ℕ, Prime p ∧ p = Nat.min_factor (5^5 - 5^3) ∧ p = 2 := by
  sorry

end least_prime_factor_of_expression_l530_530983


namespace solve_geometric_sequence_l530_530239

noncomputable def general_formula (a : ℕ → ℝ) := ∀ n : ℕ, a n = Real.exp (-3 * n + 13)

noncomputable def Sn_formula (b : ℕ → ℝ) (n : ℕ) := ∑ i in Finset.range n, b i = n

noncomputable def Tn_formula (c : ℕ → ℝ) (T : ℕ → ℝ) :=
  ∀ n : ℕ, T n = ∑ i in Finset.range n, c i 

theorem solve_geometric_sequence {a : ℕ → ℝ} (k : ℝ) (hk : k > 2 * Real.sqrt Real.exp 2)
    (h_geom : ∀ n m : ℕ, a n * a m = a (n + m)) (ha4 : a 4 = Real.exp 1)
    (ha2_ha7 : ∀ x : ℝ, Real.exp 1 * x ^ 2 + k * x + 1 = 0 → x = a 2 ∨ x = a 7) :
  general_formula a ∧ (∃ n, Sn_formula (λ n, Real.log (a n)) n = n ∧ n = 7) ∧ 
  (∃ m, Tn_formula (λ n, (Real.log (a n)) * (Real.log (a (n + 1))) * 
                       (Real.log (a (n + 2)))) = m ∧ T m = 310 ∧ m = 4) := 
sorry

end solve_geometric_sequence_l530_530239


namespace diagonals_in_30_sided_polygon_l530_530822

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem diagonals_in_30_sided_polygon :
  number_of_diagonals 30 = 405 :=
by sorry

end diagonals_in_30_sided_polygon_l530_530822


namespace smallest_common_multiple_l530_530508

theorem smallest_common_multiple (n : ℕ) (h1 : n > 0) (h2 : 8 ∣ n) (h3 : 6 ∣ n) : n = 24 :=
by sorry

end smallest_common_multiple_l530_530508


namespace initially_had_8_l530_530379

-- Define the number of puppies given away
def given_away : ℕ := 4

-- Define the number of puppies still with Sandy
def still_has : ℕ := 4

-- Define the total number of puppies initially
def initially_had (x y : ℕ) : ℕ := x + y

-- Prove that the number of puppies Sandy's dog had initially equals 8
theorem initially_had_8 : initially_had given_away still_has = 8 :=
by sorry

end initially_had_8_l530_530379


namespace number_of_non_similar_regular_pointed_stars_l530_530157

theorem number_of_non_similar_regular_pointed_stars (a b : ℕ) (h_prime_a : Nat.Prime a) (h_prime_b : Nat.Prime b) (h_distinct : a ≠ b) (h_order : a < b):
  let p := a * b in
  (p - a - b + 1) / 2 = (∑ n in nondegenerate_stars p, 1) := by
    sorry

end number_of_non_similar_regular_pointed_stars_l530_530157


namespace intersection_of_segments_l530_530093

theorem intersection_of_segments (A : ℕ → set ℝ) (h : ∀ i, ∃ S : set (set ℝ), (finite S) ∧ (S.card = 100) ∧ (A i = ⋃₀ S)) :
  ∃ S : set (set ℝ), finite S ∧ (S.card ≤ 9901) ∧ (⋂ i in finset.range 100, A i = ⋃₀ S) :=
by
  sorry

end intersection_of_segments_l530_530093


namespace initial_machines_l530_530113

theorem initial_machines (x : ℕ) (N : ℕ)
  (h1 : N * R = x / 10)
  (h2 : 50 * R = 5x / 6)
  (hR : R = x / 60) :
  N = 6 := 
sorry

end initial_machines_l530_530113


namespace product_of_f_and_g_l530_530219

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x * (x + 1))
noncomputable def g (x : ℝ) : ℝ := 1 / real.sqrt x

theorem product_of_f_and_g (x : ℝ) (h1 : x > 0): 
  f(x) * g(x) = real.sqrt(x + 1) := 
by sorry

end product_of_f_and_g_l530_530219


namespace multiples_of_3_or_11_but_not_both_l530_530829

theorem multiples_of_3_or_11_but_not_both (n : ℕ) (h : n < 200) : 
  ∑ i in finset.range 200, 
    (ite (i % 3 = 0 ∧ i % 11 ≠ 0 ∨ i % 11 = 0 ∧ i % 3 ≠ 0) 1 0) = 72 :=
by sorry

end multiples_of_3_or_11_but_not_both_l530_530829


namespace supplementary_angles_equal_l530_530917

axiom supplementary (a b : ℝ) : Prop := a + b = 180

theorem supplementary_angles_equal (a b c : ℝ) (h₁ : supplementary a c) (h₂ : supplementary b c) : a = b := 
sorry

end supplementary_angles_equal_l530_530917


namespace fourth_root_of_81_is_3_l530_530025

noncomputable theory

def fourth_root_of_81 : ℝ := real.root 81 4

theorem fourth_root_of_81_is_3 : fourth_root_of_81 = 3 :=
by {
  unfold fourth_root_of_81,
  sorry,
}

end fourth_root_of_81_is_3_l530_530025


namespace rectangular_solid_height_l530_530079

theorem rectangular_solid_height (l w SA : ℕ) (h : ℕ) (h1 : l = 6) (h2 : w = 5) (h3 : SA = 104)
  (h_formula : SA = 2 * l * w + 2 * l * h + 2 * w * h) : h = 2 := by
  subst h1
  subst h2
  subst h3
  rw [h_formula]
  sorry

end rectangular_solid_height_l530_530079


namespace number_of_diagonals_30_sides_l530_530777

def number_of_diagonals (n : ℕ) : ℕ :=
  nat.choose n 2 - n

theorem number_of_diagonals_30_sides :
  number_of_diagonals 30 = 405 :=
by {
  rw [number_of_diagonals, nat.choose, nat.factorial, nat.factorial, nat.factorial],
  -- The proof would proceed to simplify the combination and subtraction, but we use sorry to skip details.
  sorry,
}

end number_of_diagonals_30_sides_l530_530777


namespace product_of_roots_eq_neg25_l530_530678

theorem product_of_roots_eq_neg25 : 
  ∀ (x : ℝ), 24 * x^2 + 36 * x - 600 = 0 → x * (x - ((-36 - 24 * x)/24)) = -25 :=
by
  sorry

end product_of_roots_eq_neg25_l530_530678


namespace sin_180_is_zero_l530_530647

noncomputable def sin_180_eq_zero : Prop :=
  let point_zero_deg := (1, 0)
  let point_180_deg := (-1, 0)
  let y_coord_of_180_deg := 0
  sin 180 = y_coord_of_180_deg

theorem sin_180_is_zero : sin_180_eq_zero :=
  sorry

end sin_180_is_zero_l530_530647


namespace minimum_value_of_f_l530_530835

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x^2 + a*x - 1) * real.exp (x - 1)

theorem minimum_value_of_f 
  (a : ℝ)
  (h : ∀ x, deriv (λ x, f x a) x = 0 ↔ x = -2) :
  ∀ x, f x a = -1 :=
sorry

end minimum_value_of_f_l530_530835


namespace ellipse_equation_correct_l530_530706

-- Definitions from the conditions
variables {x y a b : ℝ}
variables {F1 F2 : ℝ}
variables (C : ℝ → ℝ → ℝ → Prop)

-- Condition 1
def ellipse_equation (a b : ℝ) (h₁ : a > b > 0) :=
  ∀ x y, C x y ↔ (x^2 / a^2 + y^2 / b^2 = 1)

-- Condition 2
def eccentricity (a c : ℝ) (e : ℝ) := e = c / a

-- Condition 3
def line_through_focus (F2 : ℝ) := ∀ l, l ∩ {F2} ≠ ∅

-- Condition 4
def triangle_perimeter (A B F1 F2 : ℝ) (P : ℝ) :=
  ∀ P, P = 4 * √3

-- The theorem to be proven
theorem ellipse_equation_correct (h₁ : a = √3) (h₂ : b = √2) (h₃ : b^2 = a^2 - 1)
  (h₄ : a * eccentricity = 1) :
  ∀ x y, C x y ↔ (x^2 / 3 + y^2 / 2 = 1) :=
by sorry

end ellipse_equation_correct_l530_530706


namespace enclosed_area_of_curve_is_correct_l530_530943

noncomputable def area_enclosed_by_curve (arc_length : ℝ) (side_length : ℝ) : ℝ := sorry

theorem enclosed_area_of_curve_is_correct :
  area_enclosed_by_curve (2 * pi / 3) 2 = pi + 6 * real.sqrt 3 :=
by sorry

end enclosed_area_of_curve_is_correct_l530_530943


namespace number_of_valid_integer_pairs_l530_530724

def f (x : ℝ) : ℝ := x^2 + 2 * |x| - 15

def valid_domain (a b : ℤ) : Prop :=
  ∀ x, (a : ℝ) ≤ x ∧ x ≤ (b : ℝ) → (f x ∈ Icc (-15) 0)

theorem number_of_valid_integer_pairs : 
  ∃ (pairs : finset (ℤ × ℤ)), (∀ p ∈ pairs, valid_domain p.1 p.2) ∧ pairs.card = 7 :=
by {
  -- Proof goes here.
  sorry
}

end number_of_valid_integer_pairs_l530_530724


namespace solve_quadratic_l530_530385

theorem solve_quadratic : ∀ x, x^2 - 4 * x + 3 = 0 ↔ x = 3 ∨ x = 1 := 
by
  sorry

end solve_quadratic_l530_530385


namespace find_seq_a_n_l530_530027

-- Define the conditions and sequences
axiom seqs (a_n b_n : ℕ → ℝ) (a₀ d p₀ q : ℝ) : Prop :=
  (∀ n, a_n n + b_n n = a₀ + (d * n)) ∧ -- arithmetic sequence condition
  (∀ n, a_n n - b_n n = p₀ * (q ^ (2 * n))) ∧ -- geometric sequence condition
  (∀ n, q ≠ 0 ∧ q ≠ 1 ∧ q ≠ -1) -- q not equal to ±1 or 0

-- Define the limit condition for c_n
axiom c_n_lim (a_n b_n : ℕ → ℝ) : Prop :=
  is_limit (λ n, a_n n ^ 2 + b_n n ^ 2) 2

-- The proof problem: prove the sequences a_n based on given conditions
theorem find_seq_a_n (a_n b_n : ℕ → ℝ) (a₀ d p₀ q : ℝ) 
  (h_seqs : seqs a_n b_n a₀ d p₀ q) 
  (h_lim : c_n_lim a_n b_n) :
  (∀ n : ℕ, a_n n = 1 + (p₀ / 2) * (q ^ (2 * n)) ∨ a_n n = (p₀ / 2) * (q ^ (2 * n)) - 1) :=
  sorry

end find_seq_a_n_l530_530027


namespace number_of_diagonals_in_convex_polygon_l530_530750

/-- 
A theorem stating that the number of diagonals 
in a convex polygon with 30 sides is equal to 405.
-/
theorem number_of_diagonals_in_convex_polygon (n : ℕ) (h : n = 30) (convex : True) : (n * (n - 3)) / 2 = 405 := 
by 
  rw h
  norm_num
  done

end number_of_diagonals_in_convex_polygon_l530_530750


namespace smallest_positive_period_l530_530241

-- Define a function as described in the problem
def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem smallest_positive_period (A ω φ : ℝ) :
  A > 0 → ω > 0 →
  (∀ x y : ℝ, x ∈ Set.Icc (π / 6) (π / 2) ∧ y ∈ Set.Icc (π / 6) (π / 2) → 
                x < y → f A ω φ x < f A ω φ y) →
  f A ω φ (π / 2) = f A ω φ (2 * π / 3) →
  f A ω φ (π / 2) = -f A ω φ (π / 6) →
  ∃ T > 0, ∀ x : ℝ, f A ω φ (x + T) = f A ω φ x ∧ T = π :=
by
  intros hA hω hmono hsym1 hsym2
  -- The proof steps would go here
  sorry

end smallest_positive_period_l530_530241


namespace sin_180_is_zero_l530_530643

noncomputable def sin_180_eq_zero : Prop :=
  let point_zero_deg := (1, 0)
  let point_180_deg := (-1, 0)
  let y_coord_of_180_deg := 0
  sin 180 = y_coord_of_180_deg

theorem sin_180_is_zero : sin_180_eq_zero :=
  sorry

end sin_180_is_zero_l530_530643


namespace smallest_n_satisfies_condition_l530_530680

-- Define predicate to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m ^ 2 = n

-- Define the main problem as a Lean statement
theorem smallest_n_satisfies_condition : 
  ∀ n : ℕ, n = 7 → (∀ (k : ℕ), k ≥ n → ∀ (x : ℕ), x ∈ (finset.range (k + 1)).filter (λ n, n > 0) → 
  ∃ (y : ℕ), y ∈ (finset.range (k + 1)).filter (λ n, n > 0) ∧ y ≠ x ∧ is_perfect_square (x + y)) :=
by
  sorry

end smallest_n_satisfies_condition_l530_530680


namespace cyclic_sum_sqrt_leq_2sqrt3_l530_530894

variable {a b c d : ℝ}

def condition : Prop :=
  a^2 + b^2 + c^2 + d^2 = 1

theorem cyclic_sum_sqrt_leq_2sqrt3 (h : condition) : 
  (sqrt (1 - a * b) + sqrt (1 - b * c) + sqrt (1 - c * d) + sqrt (1 - d * a) + sqrt (1 - a * c) + sqrt (1 - b * d)) <= 2 * sqrt 3 := 
  sorry

end cyclic_sum_sqrt_leq_2sqrt3_l530_530894


namespace number_of_diagonals_in_30_sided_polygon_l530_530762

-- Definition of a convex polygon with 30 sides
def convex_polygon_sides := 30

-- The function to calculate the number of diagonals in a convex polygon with n sides
noncomputable def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Prove that a convex polygon with 30 sides has 405 diagonals
theorem number_of_diagonals_in_30_sided_polygon :
  num_diagonals convex_polygon_sides = 405 :=
by
  calc
    num_diagonals 30 = 30 * (30 - 3) / 2 : rfl
    ... = 30 * 27 / 2 : rfl
    ... = 810 / 2 : rfl
    ... = 405 : rfl

end number_of_diagonals_in_30_sided_polygon_l530_530762


namespace number_of_correct_propositions_l530_530144

theorem number_of_correct_propositions :
  (¬(∀ a b c : ℝ, a > b → ac^2 > bc^2) ∧
  ∀ a b c : ℝ, a ≥ b → ac^2 ≥ bc^2 ∧
  ∀ a b c : ℝ, c ≠ 0 → (a / c > b / c) → ac > bc ∧
  ∀ a b c : ℝ, c ≠ 0 → (a / c ≥ b / c) → ac ≥ bc ∧
  ∀ a b c : ℝ, a > b → ac > bc → c > 0 ∧
  ∀ a b c : ℝ, a ≥ b → ac ≥ bc → c ≥ 0) :=
  sorry

end number_of_correct_propositions_l530_530144


namespace determine_lines_by_points_l530_530059

theorem determine_lines_by_points (n : ℕ) (h : (∑ k in Finset.range n, k) ≤ 28) : n = 8 :=
sorry

end determine_lines_by_points_l530_530059


namespace sin_greater_not_obtuse_non_existence_not_necessarily_isosceles_l530_530308

-- Statement 1: If A > B, then sin A > sin B
theorem sin_greater (A B : ℝ) (h : A > B) : sin A > sin B := sorry

-- Statement 2: The triangle with sides a=4, b=5, c=6 is not obtuse
theorem not_obtuse (a b c : ℝ) (h1 : a = 4) (h2 : b = 5) (h3 : c = 6) : ¬(c^2 > a^2 + b^2) := sorry

-- Statement 3: The triangle with sides a=5, b=10, angle A=π/4 does not exist
theorem non_existence (a b A : ℝ) (h1 : a = 5) (h2 : b = 10) (h3 : A = π / 4) : false := sorry

-- Statement 4: A triangle with a cos A = b cos B is not necessarily isosceles
theorem not_necessarily_isosceles (a b A B : ℝ) (h : a * cos A = b * cos B) : ¬(a = b) ∨ ¬(A = B) := sorry

end sin_greater_not_obtuse_non_existence_not_necessarily_isosceles_l530_530308


namespace angle_340_is_fourth_quadrant_l530_530100

def angle_quadrant (θ : ℝ) : ℕ :=
  if 0 ≤ θ ∧ θ < 90 then 1
  else if 90 ≤ θ ∧ θ < 180 then 2
  else if 180 ≤ θ ∧ θ < 270 then 3
  else if 270 ≤ θ ∧ θ < 360 then 4
  else (θ / 360).floor.toNat % 4 + 1

theorem angle_340_is_fourth_quadrant : angle_quadrant 340 = 4 := 
by 
  sorry

end angle_340_is_fourth_quadrant_l530_530100


namespace jake_balloons_bought_l530_530141

theorem jake_balloons_bought (B : ℕ) (h : 6 = (2 + B) + 1) : B = 3 :=
by
  -- proof omitted
  sorry

end jake_balloons_bought_l530_530141


namespace area_rectangle_in_triangle_le_half_l530_530914

theorem area_rectangle_in_triangle_le_half (ABC : Triangle) (R : Rectangle) (h_inscribed : Inscribed R ABC) :
  area R ≤ 1 / 2 * area ABC :=
sorry

end area_rectangle_in_triangle_le_half_l530_530914


namespace sin_180_eq_zero_l530_530618

theorem sin_180_eq_zero : Real.sin (180 * Real.pi / 180) = 0 := by
  -- Simplifying the angle, 180 degrees = π radians
  let angle := 180 * Real.pi / 180
  have h : angle = Real.pi := by
    simp [angle, Real.pi]
  rw h
  -- From the unit circle, we know the sine of angle π is 0
  exact Real.sin_pi

end sin_180_eq_zero_l530_530618


namespace largest_n_product_consecutive_integers_l530_530195

theorem largest_n_product_consecutive_integers : ∃ (n : ℕ), (∀ (x : ℕ), n! = (list.Ico x (x + n - 4)).prod) ∧ n = 4 := 
by
  sorry

end largest_n_product_consecutive_integers_l530_530195


namespace letter_150th_in_pattern_l530_530494

def repeating_sequence := "XYZ"

def letter_at_position (n : ℕ) : char :=
  let seq := repeating_sequence.to_list
  seq.get! ((n - 1) % seq.length)

theorem letter_150th_in_pattern : letter_at_position 150 = 'Z' :=
by sorry

end letter_150th_in_pattern_l530_530494


namespace f_g_relationship_l530_530831

def f (x : ℝ) : ℝ := 3 * x ^ 2 - x + 1
def g (x : ℝ) : ℝ := 2 * x ^ 2 + x - 1

theorem f_g_relationship (x : ℝ) : f x > g x :=
by
  -- proof goes here
  sorry

end f_g_relationship_l530_530831


namespace total_newspapers_collected_l530_530613

-- Definitions based on the conditions
def Chris_collected : ℕ := 42
def Lily_collected : ℕ := 23

-- The proof statement
theorem total_newspapers_collected :
  Chris_collected + Lily_collected = 65 := by
  sorry

end total_newspapers_collected_l530_530613


namespace least_prime_factor_of_5_pow_5_minus_5_pow_3_l530_530986

theorem least_prime_factor_of_5_pow_5_minus_5_pow_3 :
  Nat.least_prime_factor (5^5 - 5^3) = 2 :=
by
  sorry

end least_prime_factor_of_5_pow_5_minus_5_pow_3_l530_530986


namespace sequence_abs_sum_l530_530255

theorem sequence_abs_sum : 
  let a_sequence : ℕ → ℤ := λ n => -60 + 3 * n
  in |a_sequence 0| + |a_sequence 1| + ... + |a_sequence 29| = 765 :=
by
  sorry

end sequence_abs_sum_l530_530255


namespace find_x_in_triangle_l530_530311

theorem find_x_in_triangle (y z : ℝ) (cos_Y_minus_Z : ℝ) (h1 : y = 7) (h2 : z = 6) (h3 : cos_Y_minus_Z = 1 / 2) : 
    ∃ x : ℝ, x = Real.sqrt 73 :=
by
  existsi Real.sqrt 73
  sorry

end find_x_in_triangle_l530_530311


namespace diagonal_count_of_convex_polygon_30_sides_l530_530795
-- Importing the entire Mathlib library for necessary mathematical constructs

theorem diagonal_count_of_convex_polygon_30_sides : 
  let n := 30 in
  let num_diagonals := (n * (n - 3)) / 2 in
  num_diagonals = 405 := 
by {
  let n := 30
  let num_diagonals := (n * (n - 3)) / 2
  show num_diagonals = 405,
  sorry
}

end diagonal_count_of_convex_polygon_30_sides_l530_530795


namespace trig_identity_simplification_l530_530215

theorem trig_identity_simplification
  (α : ℝ)
  (h1 : sin (2 * α) = 3 / 5)
  (h2 : α ∈ set.Ioo (π / 4) (π / 2))
  : sin (2 * α + π / 4) + 2 * cos (π / 4) * (cos α) ^ 2 = 0 :=
begin
  sorry
end

end trig_identity_simplification_l530_530215


namespace probability_double_head_l530_530422

/-- There are three types of coins:
  - Coin 1: one head and one tail
  - Coin 2: two heads
  - Coin 3: two tails
  A coin is randomly selected and flipped, resulting in heads. 
  Prove that the probability that the coin is Coin 2 (the double head coin)
  is 2/3. -/
theorem probability_double_head (h : true) : 
  let Coin1 : ℕ := 1,
      Coin2 : ℕ := 2,
      Coin3 : ℕ := 0,
      totalHeads : ℕ := Coin1 + Coin2 + Coin3
  in 
  let p_Coin1 := 1 / 3,
      p_Coin2 := 1 / 3,
      p_Coin3 := 1 / 3,
      p_Heads_given_Coin1 := 1 / 2,
      p_Heads_given_Coin2 := 1,
      p_Heads_given_Coin3 := 0,
      p_Heads := (p_Heads_given_Coin1 * p_Coin1) + (p_Heads_given_Coin2 * p_Coin2) + (p_Heads_given_Coin3 * p_Coin3)
  in (p_Heads_given_Coin2 * p_Coin2) / p_Heads = 2 / 3 :=
by
  sorry

end probability_double_head_l530_530422


namespace proof_problem_one_proof_problem_two_l530_530742

noncomputable def condition_one :=
  ∃ (a b : ℝ), (∀ (x y : ℝ), ax - by + 4 = 0 → (a - 1)x + y + b = 0 → 
  (l₁ (-3, -1)) ∧ (a(a-1) - b = 0)) ∧ (a = 2 ∧ b = 2)

noncomputable def condition_two := 
  ∃ (a b : ℝ), (∀ (x y : ℝ), (ax - by + 4 = 0) → ((a - 1)x + y + b = 0) → 
  (parallel l₁ l₂) ∧ (eq_dist_origin l₁ l₂)) 
  ∧ ((a = 2 ∧ b = -2) ∨ (a = 2/3 ∧ b = 2))

theorem proof_problem_one : condition_one :=
sorry

theorem proof_problem_two : condition_two :=
sorry

end proof_problem_one_proof_problem_two_l530_530742


namespace work_together_time_l530_530563

theorem work_together_time (man_days : ℝ) (son_days : ℝ)
  (h_man : man_days = 5) (h_son : son_days = 7.5) :
  (1 / (1 / man_days + 1 / son_days)) = 3 :=
by
  -- Given the constraints, prove the result
  rw [h_man, h_son]
  sorry

end work_together_time_l530_530563


namespace range_of_m_if_p_implies_q_l530_530228

def proposition_p (m : ℝ) : Prop := (0 < m) ∧ (m < 1 / 3)
def proposition_q (m : ℝ) : Prop := (0 < m) ∧ (m < 15)

theorem range_of_m_if_p_implies_q :
  (∀ (m : ℝ), proposition_p m → proposition_q m) → (∀ (m : ℝ), 1 / 3 ≤ m ∧ m < 15) :=
begin
  sorry,
end

end range_of_m_if_p_implies_q_l530_530228


namespace diagonals_of_30_sided_polygon_l530_530800

theorem diagonals_of_30_sided_polygon : 
  ∀ (n : ℕ), n = 30 → (n * (n - 3)) / 2 = 405 := 
by
  intro n h
  rw h
  simp
  sorry

end diagonals_of_30_sided_polygon_l530_530800


namespace circumcircles_concur_l530_530161

variables {A B C M N P : Type} [EuclideanGeometry]

/-- Let \(A, B, C\) be points forming a triangle. Let \( M \) be a point on segment \( AB \),
and \( N \) be a point on segment \( BC \), and \( P \) be a point on segment \( CA \).
We need to prove that the circumcircles of triangles \( PAM \), \( MBN \), and \( NCP \)
concur (meet at a single point). -/

theorem circumcircles_concur
  (A B C M N P : Point)
  (hM_AB : M ∈ segment A B)
  (hN_BC : N ∈ segment B C)
  (hP_CA : P ∈ segment C A) :
  intersect_at_single_point (circumcircle A P M) (circumcircle B M N) (circumcircle C N P) :=
sorry

end circumcircles_concur_l530_530161


namespace volume_ratio_l530_530407

-- Definitions of the conditions
def radius_sphere (p : ℝ) : ℝ := 3 * p
def radius_hemisphere (p : ℝ) : ℝ := p

-- Definition of the volumes based on the radii
def volume_sphere (p : ℝ) : ℝ := (4/3) * Real.pi * (radius_sphere p)^3
def volume_hemisphere (p : ℝ) : ℝ := (1/2) * (4/3) * Real.pi * (radius_hemisphere p)^3

-- Assert the ratio of the volumes is 54
theorem volume_ratio (p : ℝ) : 
  (volume_sphere p) / (volume_hemisphere p) = 54 :=
by
  sorry

end volume_ratio_l530_530407


namespace volume_of_prism_l530_530918

theorem volume_of_prism (AB AC : ℝ) (h₁ : AB = √2) (h₂ : AC = √2) (height : ℝ) (h₃ : height = 3) : 
  let area_of_base := 1 / 2 * AB * AC in
  let volume := area_of_base * height in
  volume = 3 :=
by
  sorry

end volume_of_prism_l530_530918


namespace diameter_of_circle_l530_530899

open Real

variables {O A D E B C : Point} {r a b : ℝ}
variables (h_circle : Circle O r) (h_A_outside : A ∉ h_circle) 
variables (h_tangents : Tangent h_circle A D AE)
variables (h_AD : dist A D = r) (h_AE : dist A E = r)
variables (h_line_through_O : LineThrough A O ∈ [B, C])
variables (h_AB : dist A B = a) (h_AC : dist A C = b)

theorem diameter_of_circle (h_neq : a ≠ b) : diameter h_circle = a + b :=
sorry

end diameter_of_circle_l530_530899


namespace smallest_integer_switch_add_l530_530989

theorem smallest_integer_switch_add (a b: ℕ) (h1: n = 10 * a + b) 
  (h2: 3 * n = 10 * b + a + 5)
  (h3: 0 ≤ b) (h4: b < 10) (h5: 1 ≤ a) (h6: a < 10): n = 47 :=
by
  sorry

end smallest_integer_switch_add_l530_530989


namespace convex_polygon_diagonals_l530_530810

theorem convex_polygon_diagonals (n : ℕ) (h : n = 30) : 
  let V := n in
  let total_pairs := V * (V - 1) / 2 in
  let adjacent_pairs := n in
  let diagonals := total_pairs - adjacent_pairs in
  diagonals = 405 :=
  by
  -- number of sides is 30
  have hn : n = 30 := h
  -- calculate the total pairs
  have total_pairs_calc : total_pairs = 30 * 29 / 2 := by sorry
  -- calculate the adjacent pairs
  have adjacent_pairs_calc : adjacent_pairs = 30 := by sorry
  -- calculate the diagonals
  have diagonals_calc : diagonals = (30 * 29 / 2) - 30 := by sorry
  -- proved statement
  show 405 = 405, by rfl

end convex_polygon_diagonals_l530_530810


namespace diagonals_of_30_sided_polygon_l530_530805

theorem diagonals_of_30_sided_polygon : 
  ∀ (n : ℕ), n = 30 → (n * (n - 3)) / 2 = 405 := 
by
  intro n h
  rw h
  simp
  sorry

end diagonals_of_30_sided_polygon_l530_530805


namespace max_weight_American_l530_530006

noncomputable def max_weight_of_American_swallow (A E : ℕ) : Prop :=
A = 5 ∧ 2 * E + E = 90 ∧ 60 * A + 60 * 2 * A = 600

theorem max_weight_American (A E : ℕ) : max_weight_of_American_swallow A E :=
by
  sorry

end max_weight_American_l530_530006


namespace probability_walk_450_feet_or_less_l530_530657

theorem probability_walk_450_feet_or_less 
  (gates : List ℕ) (initial_gate new_gate : ℕ) 
  (n : ℕ) (dist_between_adjacent_gates : ℕ) 
  (valid_gates : gates.length = n)
  (distance : dist_between_adjacent_gates = 90) :
  n = 15 → 
  (initial_gate ∈ gates ∧ new_gate ∈ gates) → 
  ∃ (m1 m2 : ℕ), m1 = 59 ∧ m2 = 105 ∧ gcd m1 m2 = 1 ∧ 
  (∃ probability : ℚ, probability = (59 / 105 : ℚ) ∧ 
  (∃ sum_m1_m2 : ℕ, sum_m1_m2 = m1 + m2 ∧ sum_m1_m2 = 164)) :=
by
  sorry

end probability_walk_450_feet_or_less_l530_530657


namespace largest_angle_of_triangle_l530_530223

variable {α : Type*}

-- Definitions of the sequence and the sum of the sequence
def Sn : ℕ → ℕ
| n := n^2

def a_n (n : ℕ) : ℕ := Sn n - Sn (n - 1)

-- Definition of the triangle side lengths based on the sequence terms
def side_lengths : ℕ × ℕ × ℕ := (a_n 2, a_n 3, a_n 4)

theorem largest_angle_of_triangle :
  let ⟨a, b, c⟩ := side_lengths in a = 3 ∧ b = 5 ∧ c = 7 → ∃ θ, 0 < θ ∧ θ < 180 ∧ cos θ = -1/2 ∧ θ = 120 :=
begin
  sorry,
end

end largest_angle_of_triangle_l530_530223


namespace solve_for_x_l530_530384

theorem solve_for_x (x : ℝ) (h : sqrt x + 4 * sqrt (x^2 + 9 * x) + sqrt (x + 9) = 45 - 2 * x) : x = 81 / 16 := 
sorry

end solve_for_x_l530_530384


namespace sequence_100_eq_14_l530_530980

def sequence (n : ℕ) : ℕ :=
  let k := n - (n * (n - 1)) / 2 in k

theorem sequence_100_eq_14 : sequence 100 = 14 := 
by sorry

end sequence_100_eq_14_l530_530980


namespace sin_180_is_zero_l530_530646

noncomputable def sin_180_eq_zero : Prop :=
  let point_zero_deg := (1, 0)
  let point_180_deg := (-1, 0)
  let y_coord_of_180_deg := 0
  sin 180 = y_coord_of_180_deg

theorem sin_180_is_zero : sin_180_eq_zero :=
  sorry

end sin_180_is_zero_l530_530646


namespace number_of_diagonals_30_sides_l530_530781

def number_of_diagonals (n : ℕ) : ℕ :=
  nat.choose n 2 - n

theorem number_of_diagonals_30_sides :
  number_of_diagonals 30 = 405 :=
by {
  rw [number_of_diagonals, nat.choose, nat.factorial, nat.factorial, nat.factorial],
  -- The proof would proceed to simplify the combination and subtraction, but we use sorry to skip details.
  sorry,
}

end number_of_diagonals_30_sides_l530_530781


namespace diagonals_in_convex_polygon_with_30_sides_l530_530770

theorem diagonals_in_convex_polygon_with_30_sides : 
  let n := 30 in
  ∑ i j in finset.range n, (i + 1 - 1) * (j + 1 - 1) * (nat.choose 30 2 - 30) / 2 = 202 := 
by
  sorry

end diagonals_in_convex_polygon_with_30_sides_l530_530770


namespace convex_polygon_diagonals_l530_530787

theorem convex_polygon_diagonals (n : ℕ) (h : n = 30) : 
  let total_pairs := (n * (n - 1)) / 2 in (total_pairs - n) = 405 := 
by
  simp only [total_pairs]
  sorry

end convex_polygon_diagonals_l530_530787


namespace smallest_common_multiple_l530_530511

theorem smallest_common_multiple (n : ℕ) (h1 : n > 0) (h2 : 8 ∣ n) (h3 : 6 ∣ n) : n = 24 :=
by sorry

end smallest_common_multiple_l530_530511


namespace find_x_l530_530092

variable (x : ℝ)
variable (h : 0.3 * 100 = 0.5 * x + 10)

theorem find_x : x = 40 :=
by
  sorry

end find_x_l530_530092


namespace number_of_diagonals_in_convex_polygon_l530_530749

/-- 
A theorem stating that the number of diagonals 
in a convex polygon with 30 sides is equal to 405.
-/
theorem number_of_diagonals_in_convex_polygon (n : ℕ) (h : n = 30) (convex : True) : (n * (n - 3)) / 2 = 405 := 
by 
  rw h
  norm_num
  done

end number_of_diagonals_in_convex_polygon_l530_530749


namespace number_of_diagonals_30_sides_l530_530779

def number_of_diagonals (n : ℕ) : ℕ :=
  nat.choose n 2 - n

theorem number_of_diagonals_30_sides :
  number_of_diagonals 30 = 405 :=
by {
  rw [number_of_diagonals, nat.choose, nat.factorial, nat.factorial, nat.factorial],
  -- The proof would proceed to simplify the combination and subtraction, but we use sorry to skip details.
  sorry,
}

end number_of_diagonals_30_sides_l530_530779


namespace log_base3_850_consecutive_sum_l530_530039

theorem log_base3_850_consecutive_sum :
  (∃ c d : ℤ, c + 1 = d ∧ 6 < log 3 850 ∧ log 3 850 < 7 ∧ c + d = 13) := sorry

end log_base3_850_consecutive_sum_l530_530039


namespace minimum_area_l530_530935

noncomputable def area_enclosed (t : ℝ) : ℝ :=
  ∫ x in 0..t, (t^2 - x^2) + ∫ x in t..1, (x^2 - t^2)

theorem minimum_area : (∃ t ∈ Ioo 0 1, area_enclosed t = 1 / 4) :=
by
  sorry

end minimum_area_l530_530935


namespace equilateral_triangle_lines_l530_530656

-- Define the properties of an equilateral triangle
structure EquilateralTriangle :=
(sides_length : ℝ) -- All sides are of equal length
(angle : ℝ := 60)  -- All internal angles are 60 degrees

-- Define the concept that altitudes, medians, and angle bisectors coincide
structure CoincidingLines (T : EquilateralTriangle) :=
(altitude : T.angle = 60)
(median : T.angle = 60)
(angle_bisector : T.angle = 60)

-- Define a statement that proves the number of distinct lines in the equilateral triangle
theorem equilateral_triangle_lines (T : EquilateralTriangle) (L : CoincidingLines T) :  
  -- The total number of distinct lines consisting of altitudes, medians, and angle bisectors
  (3 = 3) :=
by
  sorry

end equilateral_triangle_lines_l530_530656


namespace Patty_pedometer_steps_l530_530910

theorem Patty_pedometer_steps :
  ∀ (resets : ℕ) (total_steps_last_day : ℕ) (steps_per_mile : ℕ),
  resets = 50 →
  total_steps_last_day = 45000 →
  steps_per_mile = 1650 →
  (90000 * resets + total_steps_last_day) / steps_per_mile = 2755 :=
by
  intros resets total_steps_last_day steps_per_mile h1 h2 h3
  have total_steps : ℕ := 90000 * resets + total_steps_last_day
  have total_miles : ℕ := total_steps / steps_per_mile
  exact eq_of_bounded_eq 10 total_miles 2755

end Patty_pedometer_steps_l530_530910


namespace nth_150th_letter_in_XYZ_l530_530488

def pattern : List Char := ['X', 'Y', 'Z']

def nth_letter (n : Nat) : Char :=
  pattern[(n - 1) % pattern.length]

theorem nth_150th_letter_in_XYZ :
  nth_letter 150 = 'Z' :=
by
  sorry

end nth_150th_letter_in_XYZ_l530_530488


namespace parallelogram_area_l530_530009

theorem parallelogram_area (EH_length EG_length : ℝ) (H_height_formula : ∀ (x : ℝ), x = EG_length / 2) :
  EH_length = 10 → EG_length = 8 → ∃ (area : ℝ), area = EH_length * (EG_length / 2) ∧ area = 40 :=
by
  intros hEH hEG
  use EH_length * (EG_length / 2)
  split
  sorry
  sorry

end parallelogram_area_l530_530009


namespace quadratic_unique_solution_l530_530955

theorem quadratic_unique_solution (a c : ℝ) :
  (∀ x : ℝ, a * x^2 + 36 * x + c = 0 ↔ x = (-36) / (2*a))  -- The quadratic equation has exactly one solution
  → a + c = 37  -- Given condition
  → a < c      -- Given condition
  → (a, c) = ( (37 - Real.sqrt 73) / 2, (37 + Real.sqrt 73) / 2 ) :=  -- Correct answer
by
  sorry

end quadratic_unique_solution_l530_530955


namespace base12_div_remainder_9_l530_530995

def base12_to_base10 (d0 d1 d2 d3: ℕ) : ℕ :=
  d0 * 12^3 + d1 * 12^2 + d2 * 12^1 + d3 * 12^0

theorem base12_div_remainder_9 : 
  let n := base12_to_base10 1 7 4 2 in
  n % 9 = 3 :=
by
  let n := base12_to_base10 1 7 4 2
  have : n = 2786 := rfl
  have : 2786 % 9 = 3 := by decide
  exact this

end base12_div_remainder_9_l530_530995


namespace solve_y_l530_530663

theorem solve_y (x y : ℤ) (h₁ : x = 3) (h₂ : x^3 - x - 2 = y + 2) : y = 20 :=
by
  -- Proof goes here
  sorry

end solve_y_l530_530663


namespace percentage_of_singles_l530_530666

noncomputable def total_hits : ℕ := 45
noncomputable def home_runs : ℕ := 2
noncomputable def triples : ℕ := 2
noncomputable def doubles : ℕ := 8

noncomputable def singles : ℕ := total_hits - (home_runs + triples + doubles)

theorem percentage_of_singles :
  (singles.to_nat * 100) / total_hits.to_nat = 73 :=
by
  sorry

end percentage_of_singles_l530_530666


namespace Walter_bus_time_l530_530066

theorem Walter_bus_time :
  let start_time := 7 * 60 + 30 -- 7:30 a.m. in minutes
  let end_time := 16 * 60 + 15 -- 4:15 p.m. in minutes
  let away_time := end_time - start_time -- total time away from home in minutes
  let classes_time := 7 * 45 -- 7 classes 45 minutes each
  let lunch_time := 40 -- lunch time in minutes
  let additional_school_time := 1.5 * 60 -- additional time at school in minutes
  let school_time := classes_time + lunch_time + additional_school_time -- total school activities time
  (away_time - school_time) = 80 :=
by
  sorry

end Walter_bus_time_l530_530066


namespace modulus_of_z_is_10_l530_530330

def i : ℂ := complex.I

def z : ℂ := (3 - i) * (1 + 3 * i)

theorem modulus_of_z_is_10 : complex.abs z = 10 := by
  sorry

end modulus_of_z_is_10_l530_530330


namespace part_I_part_II_l530_530730

-- Define the function f(x)
def f (x : Real) : Real := Math.sin (x / 2) * Math.cos (x / 2) + Math.sin (x / 2) ^ 2

-- Question I: Prove that f(pi/3) = (1 + sqrt(3)) / 4
theorem part_I : f (Real.pi / 3) = (1 + Real.sqrt 3) / 4 :=
by sorry

-- Question II: Prove the range of f(x) on (-pi/3, pi/2] is [ (1 - sqrt(2)) / 2, 1 ]
theorem part_II : Set.range (fun x => f x) ∩ Set.Icc (-Real.pi / 3) (Real.pi / 2) = Set.Icc ((1 - Real.sqrt 2) / 2) 1 :=
by sorry

end part_I_part_II_l530_530730


namespace largest_n_factorial_product_l530_530182

theorem largest_n_factorial_product : ∃ n : ℕ, (∀ m : ℕ, m ≥ n → ¬ (m! = ∏ i in (range (m - 4)).map (λ k, k + 1 + m), (1 : ℕ))) ∧ n = 1 :=
sorry

end largest_n_factorial_product_l530_530182


namespace sum_un_eq_10_10_l530_530441

noncomputable def u0 : ℝ × ℝ := (2, 2)
noncomputable def z0 : ℝ × ℝ := (3, 1)

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let norm_sq := v.1 ^ 2 + v.2 ^ 2
  (dot_product / norm_sq) * v

noncomputable def u (n : ℕ) : ℝ × ℝ :=
  match n with
  | 0 => u0
  | n + 1 => proj (z n) u0

noncomputable def z (n : ℕ) : ℝ × ℝ :=
  match n with
  | 0 => z0
  | n + 1 => proj (u n) z0

theorem sum_un_eq_10_10 : (∑ n in Finset.range 1000, u n) = (10, 10) := 
  sorry

end sum_un_eq_10_10_l530_530441


namespace Jorge_Giuliana_cakes_l530_530319

theorem Jorge_Giuliana_cakes (C : ℕ) :
  (2 * 7 + 2 * C + 2 * 30 = 110) → (C = 18) :=
by
  sorry

end Jorge_Giuliana_cakes_l530_530319


namespace part_I_part_II_l530_530251

noncomputable def f (x : ℝ) := x^3 - 3 * x^2 - 3 * x + 2
noncomputable def g (x : ℝ) (a : ℝ) := (3/2) * x^2 - 9 * x + a + 2

theorem part_I :
  ∀ {b c d : ℝ} (f0 : f 0 = 2) (fM : f (-1) = 1) (f_deriv_M : deriv f (-1) = 6),
    f x = x^3 - 3 * x^2 - 3 * x + 2 := 
begin
  intros b c d f0 fM f_deriv_M,
  sorry
end

theorem part_II :
  ∀ {a : ℝ}, (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ g x1 a = f x1 ∧ g x2 a = f x2 ∧ g x3 a = f x3) →
    2 < a ∧ a < 5/2 :=
begin
  intros a h,
  sorry
end

end part_I_part_II_l530_530251


namespace trivia_team_division_l530_530044

theorem trivia_team_division :
  ∀ (total_students not_picked group_size : ℕ),
    total_students = 65 →
    not_picked = 17 →
    group_size = 6 →
    (total_students - not_picked) / group_size = 8 :=
by
  intros total_students not_picked group_size ht hn hg
  rw [ht, hn, hg]
  sorry

end trivia_team_division_l530_530044


namespace eugene_initial_pencils_l530_530169

theorem eugene_initial_pencils (e given left : ℕ) (h1 : given = 6) (h2 : left = 45) (h3 : e = given + left) : e = 51 := by
  sorry

end eugene_initial_pencils_l530_530169


namespace road_length_l530_530597

theorem road_length 
  (D : ℕ) (N1 : ℕ) (t : ℕ) (d1 : ℝ) (N_extra : ℝ) 
  (h1 : D = 300) (h2 : N1 = 35) (h3 : t = 100) (h4 : d1 = 2.5) (h5 : N_extra = 52.5) : 
  ∃ L : ℝ, L = 3 := 
by {
  sorry
}

end road_length_l530_530597


namespace letter_150_is_Z_l530_530499

/-- Definition of the repeating pattern "XYZ" -/
def pattern : List Char := ['X', 'Y', 'Z']

/-- The repeating pattern has a length of 3 -/
def pattern_length : ℕ := 3

/-- Calculate the 150th letter in the repeating pattern "XYZ" -/
def nth_letter_in_pattern (n : ℕ) : Char :=
  let m := n % pattern_length
  if m = 0 then pattern[2] else pattern[m - 1]

/-- Prove that the 150th letter in the pattern "XYZ" is 'Z' -/
theorem letter_150_is_Z : nth_letter_in_pattern 150 = 'Z' :=
by
  sorry

end letter_150_is_Z_l530_530499


namespace no_real_root_greater_than_one_l530_530376

theorem no_real_root_greater_than_one 
  (n : ℕ)
  (a : Fin n.succ → ℝ)
  (hconds : ∀ j : Fin (n.succ), (∑ i in Finset.range j.succ, a ⟨i, Nat.lt_succ_self _⟩) ≥ 0)
  (x : ℝ)
  (hx : x > 1) :
  (∑ i in Finset.range (n + 1), a i * x^(n - i)) ≠ 0 :=
sorry

end no_real_root_greater_than_one_l530_530376


namespace sum_of_divisors_l530_530913

theorem sum_of_divisors (n : ℕ) (x : ℕ) (hx : x ≤ n!) :
  ∃ (m : ℕ) (terms : Fin m → ℕ),
    (m ≤ n) ∧
    (∀ i j, i ≠ j → terms i ≠ terms j) ∧ 
    (∀ i, terms i ∣ n!) ∧ 
    (Finset.univ.sum (λ i, terms i) = x) :=
sorry

end sum_of_divisors_l530_530913


namespace green_ball_probability_l530_530967

/-
  There are four containers:
  - Container A holds 5 red balls and 7 green balls.
  - Container B holds 7 red balls and 3 green balls.
  - Container C holds 8 red balls and 2 green balls.
  - Container D holds 4 red balls and 6 green balls.
  The probability of choosing containers A, B, C, and D is 1/4 each.
-/

def prob_A : ℚ := 1 / 4
def prob_B : ℚ := 1 / 4
def prob_C : ℚ := 1 / 4
def prob_D : ℚ := 1 / 4

def prob_Given_A : ℚ := 7 / 12
def prob_Given_B : ℚ := 3 / 10
def prob_Given_C : ℚ := 1 / 5
def prob_Given_D : ℚ := 3 / 5

def total_prob_green : ℚ :=
  prob_A * prob_Given_A + prob_B * prob_Given_B +
  prob_C * prob_Given_C + prob_D * prob_Given_D

theorem green_ball_probability : total_prob_green = 101 / 240 := 
by
  -- here would normally be the proof steps, but we use sorry to skip it.
  sorry

end green_ball_probability_l530_530967


namespace right_triangle_hypotenuse_l530_530130

theorem right_triangle_hypotenuse (a b c : ℝ) 
  (h₁ : a + b + c = 40) 
  (h₂ : a * b = 60) 
  (h₃ : a^2 + b^2 = c^2) : c = 18.5 := 
by 
  sorry

end right_triangle_hypotenuse_l530_530130


namespace wickets_in_last_match_l530_530125

noncomputable def last_match_wickets 
  (W : ℕ) -- number of wickets before last match, given W ≈ 25
  (avg_before : ℝ := 12.4) -- average before the last match
  (avg_decrease : ℝ := 0.4) -- decrease in average after the match
  (runs_last_match : ℕ := 26) -- runs given in the last match
  : ℕ := 3 -- number of wickets taken in the last match (to be proven)

theorem wickets_in_last_match (W : ℕ) (h1 : avg_before = 12.4) (h2 : avg_decrease = 0.4) (h3 : runs_last_match = 26) (h4 : W ≈ 25) : 
  last_match_wickets W 12.4 0.4 26 = 3 :=
begin
  sorry
end

end wickets_in_last_match_l530_530125


namespace range_of_m_l530_530694

theorem range_of_m (m : ℝ) (h₁ : ∀ x : ℝ, -x^2 + 7*x + 8 ≥ 0 → x^2 - 7*x - 8 ≤ 0)
  (h₂ : ∀ x : ℝ, x^2 - 2*x + 1 - 4*m^2 ≤ 0 → 1 - 2*m ≤ x ∧ x ≤ 1 + 2*m)
  (not_p_sufficient_for_not_q : ∀ x : ℝ, ¬(-x^2 + 7*x + 8 ≥ 0) → ¬(x^2 - 2*x + 1 - 4*m^2 ≤ 0))
  (suff_non_necess : ∀ x : ℝ, (x^2 - 2*x + 1 - 4*m^2 ≤ 0) → ¬(x^2 - 7*x - 8 ≤ 0))
  : 0 < m ∧ m ≤ 1 := sorry

end range_of_m_l530_530694


namespace convex_polygon_diagonals_l530_530809

theorem convex_polygon_diagonals (n : ℕ) (h : n = 30) : 
  let V := n in
  let total_pairs := V * (V - 1) / 2 in
  let adjacent_pairs := n in
  let diagonals := total_pairs - adjacent_pairs in
  diagonals = 405 :=
  by
  -- number of sides is 30
  have hn : n = 30 := h
  -- calculate the total pairs
  have total_pairs_calc : total_pairs = 30 * 29 / 2 := by sorry
  -- calculate the adjacent pairs
  have adjacent_pairs_calc : adjacent_pairs = 30 := by sorry
  -- calculate the diagonals
  have diagonals_calc : diagonals = (30 * 29 / 2) - 30 := by sorry
  -- proved statement
  show 405 = 405, by rfl

end convex_polygon_diagonals_l530_530809


namespace liam_book_pages_l530_530345

theorem liam_book_pages
  (h1 : ∀ i ∈ {1, 2, 3}, liam_reads i = 45)
  (h2 : ∀ i ∈ {4, 5, 6}, liam_reads i = 50)
  (h3 : liam_reads 7 = 25) :
  (∑ i in {1, 2, 3, 4, 5, 6, 7}, liam_reads i) = 310 := 
by
  sorry

end liam_book_pages_l530_530345


namespace number_of_diagonals_30_sides_l530_530775

def number_of_diagonals (n : ℕ) : ℕ :=
  nat.choose n 2 - n

theorem number_of_diagonals_30_sides :
  number_of_diagonals 30 = 405 :=
by {
  rw [number_of_diagonals, nat.choose, nat.factorial, nat.factorial, nat.factorial],
  -- The proof would proceed to simplify the combination and subtraction, but we use sorry to skip details.
  sorry,
}

end number_of_diagonals_30_sides_l530_530775


namespace handshake_count_l530_530417

theorem handshake_count (num_companies : ℕ) (num_representatives : ℕ) 
  (total_handshakes : ℕ) (h1 : num_companies = 5) (h2 : num_representatives = 5)
  (h3 : total_handshakes = (num_companies * num_representatives * 
   (num_companies * num_representatives - 1 - (num_representatives - 1)) / 2)) :
  total_handshakes = 250 :=
by
  rw [h1, h2] at h3
  exact h3

end handshake_count_l530_530417


namespace smallest_n_for_subsets_l530_530337

open Set

theorem smallest_n_for_subsets (X : Set ℕ) (hX : X.card = 56) :
  ∃ (n : ℕ), (∀ (S : Finset (Set X)) (hS : S.card = 15),
    (∀ T : Finset (Set X), T ⊆ S → T.card = 7 →
      (T.biUnion id).card ≥ n) →
    ∃ A B C ∈ S, (A ∩ B ∩ C).nonempty) → n = 41 :=
by
  sorry

end smallest_n_for_subsets_l530_530337


namespace abs_2023_eq_2023_l530_530539

theorem abs_2023_eq_2023 : abs 2023 = 2023 := by
  sorry

end abs_2023_eq_2023_l530_530539


namespace mod_last_digit_l530_530912

theorem mod_last_digit (N : ℕ) (a b : ℕ) (h : N = 10 * a + b) (hb : b < 10) : 
  N % 10 = b ∧ N % 2 = b % 2 ∧ N % 5 = b % 5 :=
by
  sorry

end mod_last_digit_l530_530912


namespace coin_problem_l530_530427

noncomputable def P_flip_heads_is_heads (coin : ℕ → bool → Prop) : ℚ :=
if coin 2 tt then 2 / 3 else 0

theorem coin_problem (coin : ℕ → bool → Prop)
  (h1 : ∀ coin_num, coin coin_num tt = (coin_num = 1 ∨ coin_num = 2))
  (h2 : ∀ coin_num, coin coin_num ff = (coin_num = 1 ∨ coin_num = 3)):
  P_flip_heads_is_heads coin = 2 / 3 :=
by
  sorry

end coin_problem_l530_530427


namespace train_passes_jogger_in_34_seconds_l530_530529

noncomputable def km_per_hr_to_m_per_s (speed_km_per_hr : ℝ) : ℝ := (speed_km_per_hr * 1000) / 3600

theorem train_passes_jogger_in_34_seconds :
  ∀ (v_j v_t : ℝ) (d_head_start train_length : ℝ),
    v_j = 9 →
    v_t = 45 →
    d_head_start = 240 →
    train_length = 100 →
    let v_j_m_s := km_per_hr_to_m_per_s v_j in
    let v_t_m_s := km_per_hr_to_m_per_s v_t in
    let relative_speed := v_t_m_s - v_j_m_s in
    let total_distance := d_head_start + train_length in
    total_distance / relative_speed = 34 :=
begin
  intros v_j v_t d_head_start train_length h1 h2 h3 h4,
  simp [km_per_hr_to_m_per_s] at *,
  sorry
end

end train_passes_jogger_in_34_seconds_l530_530529


namespace ratio_JL_JM_l530_530915

variables (JL JM : ℝ) (areaJK: ℝ) (areaNOPQ: ℝ)

-- Given conditions
axiom area_share_RECT_SQ : areaJK * 0.3 = areaNOPQ * 0.4
axiom area_SQ: areaNOPQ = (JM))^2
axiom area_RECT: areaJK = JL * JM

-- Prove
theorem ratio_JL_JM : (JL / JM) = (4/3) :=
by 
  sorry

end ratio_JL_JM_l530_530915


namespace intersect_A_B_l530_530258

def A : Set ℝ := {x | log x / log 2 < 1}
def B : Set ℝ := {y | ∃ x, y = 2^x ∧ x ∈ A}

theorem intersect_A_B :
  A ∩ B = {y | 1 < y ∧ y < 2} :=
sorry

end intersect_A_B_l530_530258


namespace kate_money_left_l530_530320

def kate_savings_march := 27
def kate_savings_april := 13
def kate_savings_may := 28
def kate_expenditure_keyboard := 49
def kate_expenditure_mouse := 5

def total_savings := kate_savings_march + kate_savings_april + kate_savings_may
def total_expenditure := kate_expenditure_keyboard + kate_expenditure_mouse
def money_left := total_savings - total_expenditure

-- Prove that Kate has $14 left
theorem kate_money_left : money_left = 14 := 
by 
  sorry

end kate_money_left_l530_530320


namespace nth_150th_letter_in_XYZ_l530_530483

def pattern : List Char := ['X', 'Y', 'Z']

def nth_letter (n : Nat) : Char :=
  pattern[(n - 1) % pattern.length]

theorem nth_150th_letter_in_XYZ :
  nth_letter 150 = 'Z' :=
by
  sorry

end nth_150th_letter_in_XYZ_l530_530483


namespace area_of_equilateral_triangle_l530_530293

theorem area_of_equilateral_triangle (a : ℝ) (h_eq : a = 10) :
  ∃ (area : ℝ), area = 25 * real.sqrt 3 :=
sorry

end area_of_equilateral_triangle_l530_530293


namespace diagonals_of_30_sided_polygon_l530_530808

theorem diagonals_of_30_sided_polygon : 
  ∀ (n : ℕ), n = 30 → (n * (n - 3)) / 2 = 405 := 
by
  intro n h
  rw h
  simp
  sorry

end diagonals_of_30_sided_polygon_l530_530808


namespace new_device_significant_improvement_l530_530547

def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

def variance (data : List ℝ) (mean : ℝ) : ℝ :=
  (data.foldl (λ acc x => acc + (x - mean) ^ 2) 0) / data.length

def significantImprovement (oldData newData : List ℝ) : Prop :=
  let x := mean oldData
  let y := mean newData
  let s1_squared := variance oldData x
  let s2_squared := variance newData y
  y - x ≥ 2 * Real.sqrt ((s1_squared + s2_squared) / oldData.length)

theorem new_device_significant_improvement :
  significantImprovement [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
                         [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5] :=
by
  sorry

end new_device_significant_improvement_l530_530547


namespace candy_problem_l530_530348

theorem candy_problem (N S a : ℕ) (h1 : a = S - a - 7) (h2 : a > 1) : S = 21 := 
sorry

end candy_problem_l530_530348


namespace coefficient_x3_in_expansion_l530_530945

theorem coefficient_x3_in_expansion :
  ∀ (x : ℝ), polynomial.coeff (polynomial.mul (polynomial.C 2 * polynomial.X + 1) (polynomial.C 1 - polynomial.X)^5) 3 = -10 :=
by
  intro x
  -- The proof would go here
  sorry

end coefficient_x3_in_expansion_l530_530945


namespace diagonals_bisect_each_other_l530_530086

-- Definitions of Rectangle, Rhombus, and Square with their properties
structure Rectangle (α : Type*) [OrderedSemiring α] :=
  (a b : α)
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (diagonals_bisect_each_other : True)

structure Rhombus (α : Type*) [OrderedSemiring α] :=
  (a : α)
  (a_pos : 0 < a)
  (diagonals_bisect_each_other : True)

structure Square (α : Type*) [OrderedSemiring α] :=
  (a : α)
  (a_pos : 0 < a)
  (diagonals_bisect_each_other : True)

-- Proof problem: Given a shape that can be a rectangle, rhombus, or square, prove that the diagonals bisect each other.
theorem diagonals_bisect_each_other (α : Type*) [OrderedSemiring α] 
  (r : Rectangle α ∨ Rhombus α ∨ Square α) : True :=
by
  cases r with
  | inl rect => exact rect.diagonals_bisect_each_other
  | inr rhom_sq => 
    cases rhom_sq with
    | inl rhom => exact rhom.diagonals_bisect_each_other
    | inr sq => exact sq.diagonals_bisect_each_other
  sorry -- Proof not required as per instructions

end diagonals_bisect_each_other_l530_530086


namespace lcm_36_225_l530_530676

theorem lcm_36_225 : Nat.lcm 36 225 = 900 := by
  -- Defining the factorizations as given
  let fact_36 : 36 = 2^2 * 3^2 := by rfl
  let fact_225 : 225 = 3^2 * 5^2 := by rfl

  -- Indicating what LCM we need to prove
  show Nat.lcm 36 225 = 900

  -- Proof (skipped)
  sorry

end lcm_36_225_l530_530676


namespace sum_of_interior_angles_l530_530282

theorem sum_of_interior_angles (n : ℕ) (h : n - 3 = 7) : (n - 2) * 180 = 1440 :=
by {
  have h1 : n = 10 := by linarith,
  rw h1,
  norm_num,
}

end sum_of_interior_angles_l530_530282


namespace additional_coins_needed_l530_530585

def num_friends : Nat := 15
def current_coins : Nat := 105

def total_coins_needed (n : Nat) : Nat :=
  n * (n + 1) / 2
  
theorem additional_coins_needed :
  let coins_needed := total_coins_needed num_friends
  let additional_coins := coins_needed - current_coins
  additional_coins = 15 :=
by
  sorry

end additional_coins_needed_l530_530585


namespace conical_tower_depth_l530_530115

noncomputable def depth_of_water (height_tower : ℝ) (volume_fraction_above_water : ℝ) : ℝ :=
  let submerged_volume_fraction := 1 - volume_fraction_above_water
  let submerged_height_fraction := real.cbrt submerged_volume_fraction
  let submerged_height := submerged_height_fraction * height_tower
  height_tower - submerged_height

theorem conical_tower_depth :
  depth_of_water 10000 (1 / 4) = 905 :=
by
  sorry

end conical_tower_depth_l530_530115


namespace sum_of_roots_3x2_minus_12x_plus_12_eq_4_l530_530990

def sum_of_roots_quadratic (a b : ℚ) (h : a ≠ 0) : ℚ := -b / a

theorem sum_of_roots_3x2_minus_12x_plus_12_eq_4 :
  sum_of_roots_quadratic 3 (-12) (by norm_num) = 4 :=
sorry

end sum_of_roots_3x2_minus_12x_plus_12_eq_4_l530_530990


namespace solve_eq1_solve_eq2_l530_530928

-- Definition of the first problem
def eq1 (x : ℝ) : Prop := x^2 - 5 = 0

-- Statement for the first problem that solves eq1 and finds the roots
theorem solve_eq1 (x : ℝ) : eq1 x → x = sqrt 5 ∨ x = -sqrt 5 := sorry

-- Definition of the second problem
def eq2 (x : ℝ) : Prop := x^2 + 2*x - 5 = 0

-- Statement for the second problem that solves eq2 by completing the square
theorem solve_eq2 (x : ℝ) : eq2 x → 
  x = -1 + sqrt 6 ∨ x = -1 - sqrt 6 := sorry

end solve_eq1_solve_eq2_l530_530928


namespace alan_total_cost_is_84_l530_530581

def num_dark_cds : ℕ := 2
def num_avn_cds : ℕ := 1
def num_90s_cds : ℕ := 5
def price_avn_cd : ℕ := 12 -- in dollars
def price_dark_cd : ℕ := price_avn_cd * 2
def total_cost_other_cds : ℕ := num_dark_cds * price_dark_cd + num_avn_cds * price_avn_cd
def price_90s_cds : ℕ := ((40 : ℕ) * total_cost_other_cds) / 100
def total_cost_all_products : ℕ := num_dark_cds * price_dark_cd + num_avn_cds * price_avn_cd + price_90s_cds

theorem alan_total_cost_is_84 : total_cost_all_products = 84 := by
  sorry

end alan_total_cost_is_84_l530_530581


namespace solve_for_five_minus_a_l530_530268

theorem solve_for_five_minus_a (a b : ℤ) 
  (h1 : 5 + a = 6 - b)
  (h2 : 6 + b = 9 + a) : 
  5 - a = 6 := 
by 
  sorry

end solve_for_five_minus_a_l530_530268


namespace count_integers_containing_zero_up_to_2050_l530_530659

def contains_digit_zero (n : ℕ) : Prop :=
  n.digits 10 ∈ (0 :: List.nil)

def count_integers_containing_zero (n : ℕ) : ℕ :=
  (List.range' 1 n).countp contains_digit_zero

theorem count_integers_containing_zero_up_to_2050 :
  count_integers_containing_zero 2051 = 502 :=
by
  sorry

end count_integers_containing_zero_up_to_2050_l530_530659


namespace distance_between_vertices_of_hyperbola_l530_530179

theorem distance_between_vertices_of_hyperbola : 
  ∀ (x y : ℝ),
  (y^2 / 36 - x^2 / 16 = 1) → 
  distance_between_vertices 36 12 :=
begin
  sorry,
end

end distance_between_vertices_of_hyperbola_l530_530179


namespace diagonals_in_convex_polygon_with_30_sides_l530_530767

theorem diagonals_in_convex_polygon_with_30_sides : 
  let n := 30 in
  ∑ i j in finset.range n, (i + 1 - 1) * (j + 1 - 1) * (nat.choose 30 2 - 30) / 2 = 202 := 
by
  sorry

end diagonals_in_convex_polygon_with_30_sides_l530_530767


namespace initial_bitcoins_l530_530921

variable (S0 P0 A0 : ℕ)

def transactions : Prop :=
    let S1 := S0 - P0
    let P1 := 2 * P0
    let A1 := A0
    let S2 := S1 - A0
    let P2 := P1
    let A2 := 2 * A0
    let S3 := 2 * S2
    let P3 := P2 - S2 - A2
    let A3 := 4 * A0
    let S4 := S3 + S3
    let P4 := P3 + S3
    let A4 := A3 - S3 - P3
    S4 = 8 ∧ P4 = 8 ∧ A4 = 8

theorem initial_bitcoins :
  S0 = 13 ∧ P0 = 7 ∧ A0 = 4 → transactions S0 P0 A0 :=
by
  intro h
  cases h with hS0 hPA
  cases hPA with hP0 hA0
  simp [transactions, hS0, hP0, hA0]
  sorry

end initial_bitcoins_l530_530921


namespace fixed_point_of_quadratic_l530_530210

theorem fixed_point_of_quadratic (k : ℝ) : 
    ∃ (a b : ℝ), (∀ k : ℝ, 9 * a^2 + 3 * k * a - 5 * k = b) ∧ (a = 5) ∧ (b = 225) :=
by
  use [5, 225]
  split
  · intros k
    simp [mul_assoc]
    linarith
  · constructor <;> rfl

end fixed_point_of_quadratic_l530_530210


namespace problem_statement_l530_530221
noncomputable theory
open Real

theorem problem_statement (f : ℝ → ℝ) (h_deriv : ∀ x : ℝ, f x < (deriv f x)) :
  f 1 > exp 1 * f 0 ∧ f 2023 > exp 2023 * f 0 :=
by
  sorry

end problem_statement_l530_530221


namespace sine_180_eq_zero_l530_530624

theorem sine_180_eq_zero :
  sin (180 : ℝ) = 0 :=
sorry

end sine_180_eq_zero_l530_530624


namespace least_prime_factor_of_expression_l530_530984

theorem least_prime_factor_of_expression : ∃ p : ℕ, Prime p ∧ p = Nat.min_factor (5^5 - 5^3) ∧ p = 2 := by
  sorry

end least_prime_factor_of_expression_l530_530984


namespace normal_price_of_4oz_package_l530_530137

-- Define the conditions as Lean variables and constants
variables (P : ℝ) -- Normal price of each 4 oz package of butter
                  (P₈ : ℝ) -- Price of 8 oz package of butter
                  (P₁₆ : ℝ) -- Price of 16 oz package of butter
                  (total_price : ℝ) -- Total minimum price for 16 oz butter

-- Define the conditions in Lean statements
def condition_1 : P₁₆ = 7 := sorry
def condition_2 : P₈ = 4 := sorry
def condition_3 : ∃ coupon_discount, coupon_discount = 0.5 := sorry
def condition_4 : (total_weight : ℝ) := 16
def condition_5 : total_price = 6 := sorry

-- The total price consists of an 8 oz package and two 4 oz packages at a discount
noncomputable def cost_equation : Prop :=
  (P₈ + 2 * (P * coupon_discount) = total_price)

-- The proof statement
theorem normal_price_of_4oz_package (h1 : P₁₆ = 7)
                                     (h2 : P₈ = 4)
                                     (h3 : coupon_discount = 0.5)
                                     (h4 : total_weight = 16)
                                     (h5 : total_price = 6)
                                     (h6 : cost_equation) :
  P = 2 :=
by sorry

end normal_price_of_4oz_package_l530_530137


namespace limit_expression_l530_530335

noncomputable def f (x y : ℝ) : ℝ :=
  ∑ m n in ({mn : ℕ × ℕ | 0 < mn.1 ∧ 0 < mn.2 ∧ mn.1 ≥ mn.2 / 2 ∧ mn.2 ≥ mn.1 / 2}), 
    x ^ m * y ^ n

theorem limit_expression (h : ∀ ε > 0, ∃ δ > 0, ∀ {x y : ℝ}, 0 < |x - 1| < δ ∧ 0 < |y - 1| < δ → 
        |(1 - x * y ^ 2) * (1 - x ^ 2 * y) * f x y - 3| < ε) :
  (∀ (ε > 0), ∃ (δ > 0), ∀ (x y : ℝ), 0 < |x - 1| < δ ∧ 0 < |y - 1| < δ → 
    |(1 - x * y ^ 2) * (1 - x ^ 2 * y) * f x y - 3| < ε) :=
by
  sorry

end limit_expression_l530_530335


namespace option_B_is_incorrect_l530_530145

theorem option_B_is_incorrect
    (hyp1 : ∀ {p1 p2 : Point}, centrally_symmetric p1 p2 → congruent p1 p2)
    (hyp2 : ∀ {p1 p2 : Point} {center : Point}, centrally_symmetric_about center p1 p2 → center = midpoint p1 p2)
    (hyp3 : ∀ {s : Shape} {center : Point}, centrally_symmetric_about center s s' → rotate 180 center s = s')
    : ¬ (∀ p1 p2 : Point, centrally_symmetric p1 p2 → Line.connects_symmetric_points p1 p2 = axis_of_symmetry) :=
sorry

end option_B_is_incorrect_l530_530145


namespace number_of_diagonals_30_sides_l530_530776

def number_of_diagonals (n : ℕ) : ℕ :=
  nat.choose n 2 - n

theorem number_of_diagonals_30_sides :
  number_of_diagonals 30 = 405 :=
by {
  rw [number_of_diagonals, nat.choose, nat.factorial, nat.factorial, nat.factorial],
  -- The proof would proceed to simplify the combination and subtraction, but we use sorry to skip details.
  sorry,
}

end number_of_diagonals_30_sides_l530_530776


namespace monotonic_increase_interval_l530_530018

noncomputable def interval_of_increase {φ : ℝ} (f : ℝ → ℝ) := {x : ℝ | ∀ k : ℤ, f x = sin (2 * x + φ)}

theorem monotonic_increase_interval 
  (φ : ℝ) 
  (f : ℝ → ℝ) 
  (h₁ : ∀ x : ℝ, x > 0 → f x ≤ |f (π / 6)|) 
  (h₂ : f (π / 2) > f π) 
  (interval_of_increase f) :
  interval_of_increase f = {x : ℝ | ∃ k : ℤ, (π / 6 + k * π ≤ x ∧ x ≤ 2 * π / 3 + k * π)} :=
sorry

end monotonic_increase_interval_l530_530018


namespace outOfPocketCost_l530_530430

noncomputable def visitCost : ℝ := 300
noncomputable def castCost : ℝ := 200
noncomputable def insuranceCoverage : ℝ := 0.60

theorem outOfPocketCost : (visitCost + castCost - (visitCost + castCost) * insuranceCoverage) = 200 := by
  sorry

end outOfPocketCost_l530_530430


namespace exists_prime_divisor_coprime_infinitely_many_primes_of_form_l530_530895

-- Definition of the problem conditions in Lean
def is_prime (n : ℕ) : Prop := nat.prime n

noncomputable def polynomial_f (p : ℕ) (x : ℕ) : ℕ :=
  (finset.range p).sum (λ i, x ^ i)

theorem exists_prime_divisor_coprime (p : ℕ) (m : ℕ) (h_prime_p : is_prime p) (h_divisible_m : p ∣ m) :
  ∃ q, is_prime q ∧ q ∣ polynomial_f p m ∧ nat.coprime q (m * (m-1)) :=
sorry

theorem infinitely_many_primes_of_form (p : ℕ) (h_prime_p : is_prime p) :
  ∃^∞ n : ℕ, is_prime (p * n + 1) :=
sorry

end exists_prime_divisor_coprime_infinitely_many_primes_of_form_l530_530895


namespace length_OD1_l530_530312

-- Definitions
variables (A B C D A1 B1 C1 D1 O : Type) -- points in space
variables (distance : Type → Type → ℝ)

def is_center_of_sphere (O : Type) (sphere_center : Type) (radius : ℝ) : Prop :=
distance O sphere_center = radius

def intersects_face (sphere_center : Type) (face_center : Type) (circle_radius : ℝ) : Prop :=
distance sphere_center face_center = circle_radius

-- Given conditions
axiom cube_structure : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ A1 ≠ B1 ∧ B1 ≠ C1 ∧ C1 ≠ D1 ∧ D1 ≠ A1 ∧ A ≠ A1
axiom sphere_center_O : is_center_of_sphere O O 10
axiom intersection_A_O : intersects_face O A 1
axiom intersection_A1_O : intersects_face O A1 1
axiom intersection_C_O : intersects_face O C 3

-- Prove the length of the segment OD1
theorem length_OD1 : distance O D1 = 17 :=
sorry

end length_OD1_l530_530312


namespace problem_statement_l530_530736

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then 2 * x^3 else
if x < (π / 2) then -Real.sin x else
0

theorem problem_statement : f (f (π / 6)) = -1 / 4 := by
  sorry

end problem_statement_l530_530736


namespace modulus_of_z_l530_530721

open Complex

-- Definition of the complex number and the modulus property
def z := (1 - 2 * Complex.i) ^ 2 / (2 + Complex.i)

theorem modulus_of_z : Complex.abs z = Real.sqrt 5 := by
  sorry

end modulus_of_z_l530_530721


namespace alan_total_cost_is_84_l530_530578

noncomputable def price_AVN : ℝ := 12
noncomputable def multiplier : ℝ := 2
noncomputable def count_Dark : ℕ := 2
noncomputable def count_AVN : ℕ := 1
noncomputable def count_90s : ℕ := 5
noncomputable def percentage_90s : ℝ := 0.40

def main_theorem : Prop :=
  let price_Dark := price_AVN * multiplier in
  let total_cost_Dark := price_Dark * count_Dark in
  let total_cost_AVN := price_AVN * count_AVN in
  let total_cost_other := total_cost_Dark + total_cost_AVN in
  let cost_90s := percentage_90s * total_cost_other in
  let total_cost := total_cost_other + cost_90s in
  total_cost = 84

theorem alan_total_cost_is_84 : main_theorem :=
  sorry

end alan_total_cost_is_84_l530_530578


namespace convex_polygon_diagonals_l530_530789

theorem convex_polygon_diagonals (n : ℕ) (h : n = 30) : 
  let total_pairs := (n * (n - 1)) / 2 in (total_pairs - n) = 405 := 
by
  simp only [total_pairs]
  sorry

end convex_polygon_diagonals_l530_530789


namespace equal_areas_of_quadrilaterals_l530_530147

variables {A B C D E K O : Type*} [add_comm_group A] [vector_space ℝ A]
variables [affine_space A E]

/-- Given points A, B, C, D forming a parallelogram, 
    E on the extension of AB, and K on the extension of AD,
    with lines BK and DE intersecting at O -/
def parallelogram (A B C D E K O : Type*) [affine_space A E] := 
-- define points making a parallelogram
affine_space.parallel A (line B C) D E ∧
affine_space.between A (line D E) B K ∧
affine_space.intersects (line B K) (line D E) O

theorem equal_areas_of_quadrilaterals
  (h : parallelogram A B C D E K O)
  : area (quadrilateral A D O B) = area (quadrilateral E C K O) :=
sorry

end equal_areas_of_quadrilaterals_l530_530147


namespace avg_decreased_by_one_l530_530937

noncomputable def avg_decrease (n : ℕ) (average_initial : ℝ) (obs_new : ℝ) : ℝ :=
  (n * average_initial + obs_new) / (n + 1)

theorem avg_decreased_by_one (init_avg : ℝ) (obs_new : ℝ) (num_obs : ℕ)
  (h₁ : num_obs = 6)
  (h₂ : init_avg = 12)
  (h₃ : obs_new = 5) :
  init_avg - avg_decrease num_obs init_avg obs_new = 1 :=
by
  sorry

end avg_decreased_by_one_l530_530937


namespace find_center_coordinates_l530_530700

theorem find_center_coordinates (a : ℝ) (h : a > 0) : 
  let l1 := λ x y : ℝ, 3*x - y = 0 -- line l1 passing through origin
  let l2 := λ x y : ℝ, x + 3*y + 1 = 0 -- line l2
  let C := (λ x y : ℝ, x^2 + y^2 - 2*a*x - 2*a*y = 1 - 2*a^2) -- circle equation
  let distance := (|3*a - a| / sqrt (9 + 1)) -- distance from center to l1
  distance = 1 * sqrt 2 / 2 -> (a = sqrt 5 / 2) := 
begin
  sorry
end

end find_center_coordinates_l530_530700


namespace additional_coins_needed_l530_530586

def num_friends : Nat := 15
def current_coins : Nat := 105

def total_coins_needed (n : Nat) : Nat :=
  n * (n + 1) / 2
  
theorem additional_coins_needed :
  let coins_needed := total_coins_needed num_friends
  let additional_coins := coins_needed - current_coins
  additional_coins = 15 :=
by
  sorry

end additional_coins_needed_l530_530586


namespace find_lowest_score_l530_530936

-- Define the mean of 15 scores
def mean_15_scores (sum_15 : ℕ) : ℕ := sum_15 / 15

-- Define the mean of 13 remaining scores
def mean_13_remaining_scores (sum_13 : ℕ) : ℕ := sum_13 / 13

-- Given data
axiom sum_15 : ℕ
axiom sum_13 : ℕ
axiom highest_score : ℕ := 95

-- Given conditions
axiom h1 : mean_15_scores sum_15 = 75
axiom h2 : mean_13_remaining_scores sum_13 = 78

-- Theorem to prove the lowest score
theorem find_lowest_score : ∃ lowest_score : ℕ, lowest_score = 16 :=
  by
  sorry

end find_lowest_score_l530_530936


namespace sequence_follows_pattern_l530_530361

-- Define the sequence
def sequence : List ℕ := [2, 3, 5, 8, 12, 17, 23]

-- Define the pattern condition function
def follows_pattern (seq : List ℕ) : Prop :=
  seq.length ≥ 2 ∧ ∀ i, i < seq.length - 1 → seq.get i + (i + 1) = seq.get (i + 1)

-- The theorem statement
theorem sequence_follows_pattern :
  follows_pattern sequence :=
by
  -- The proof logic will go here, skipped for now
  sorry

end sequence_follows_pattern_l530_530361


namespace sea_horses_count_l530_530602

theorem sea_horses_count (S P : ℕ) 
  (h1 : S / P = 5 / 11) 
  (h2 : P = S + 85) 
  : S = 70 := sorry

end sea_horses_count_l530_530602


namespace smallest_common_multiple_of_8_and_6_l530_530516

theorem smallest_common_multiple_of_8_and_6 : ∃ n : ℕ, n > 0 ∧ (8 ∣ n) ∧ (6 ∣ n) ∧ ∀ m : ℕ, (m > 0 ∧ (8 ∣ m) ∧ (6 ∣ m)) → n ≤ m :=
by
  sorry

end smallest_common_multiple_of_8_and_6_l530_530516


namespace sin_180_eq_0_l530_530651

theorem sin_180_eq_0 : Real.sin (180 * Real.pi / 180) = 0 :=
by
  sorry

end sin_180_eq_0_l530_530651


namespace concatenated_number_mod_49_l530_530884

-- Define M as the concatenated number formed by writing the integers from 1 to 48 in order
def M : ℤ := 
  let digits := (List.range 48).map (λ n => n + 1).joinDigits
  digits

theorem concatenated_number_mod_49 : M % 49 = 0 := by
  sorry

end concatenated_number_mod_49_l530_530884


namespace distance_to_origin_eq_sqrt2_l530_530331

def complex_number : ℂ := 2 * complex.I / (1 - complex.I)

theorem distance_to_origin_eq_sqrt2 : complex.abs complex_number = real.sqrt 2 := by
  -- proof steps here
  sorry

end distance_to_origin_eq_sqrt2_l530_530331


namespace vicentes_total_cost_l530_530063

def total_cost (rice_bought cost_per_kg_rice meat_bought cost_per_lb_meat : Nat) : Nat :=
  (rice_bought * cost_per_kg_rice) + (meat_bought * cost_per_lb_meat)

theorem vicentes_total_cost :
  let rice_bought := 5
  let cost_per_kg_rice := 2
  let meat_bought := 3
  let cost_per_lb_meat := 5
  total_cost rice_bought cost_per_kg_rice meat_bought cost_per_lb_meat = 25 :=
by
  intros
  sorry

end vicentes_total_cost_l530_530063


namespace conditional_expectation_convergence_l530_530327

-- Define the problem using Lean 4 syntax

variables {Ω : Type} {F : Type} [measurable_space Ω] [probability_space Ω] 
variables (ξ ξn : ℕ → Ω → ℝ) (ℱ : measurable_space Ω) (G : measurable_space Ω)
variables (p : ℝ) (h1 : 1 ≤ p)
variables (h2 : ∀ n, measurable (ξn n)) (h3 : measurable ξ)
variables (h4 : ξn ⟶p ξ) -- This denotes that ξn converges to ξ in L^p

-- The problem statement in Lean 4
theorem conditional_expectation_convergence {Ω : Type} {F : Type} 
[measurable_space Ω] [probability_space Ω] (ξ ξn : ℕ → Ω → ℝ) 
(G : measurable_space Ω) (p : ℝ) [hp: 1 ≤ p]
(h_convergence : ∀ n, measurable (ξn n)) (hmeas : measurable ξ)
(h_Lp : ξn ⟶p ξ) : 
ξn ⟶p (λ ω, E[ξ ω | G]) := 
begin
  sorry
end

end conditional_expectation_convergence_l530_530327


namespace find_150th_letter_l530_530467

def pattern : List Char := ['X', 'Y', 'Z']

def position (N : ℕ) (pattern_length : ℕ) : ℕ :=
  if N % pattern_length = 0 then pattern_length else N % pattern_length

theorem find_150th_letter : 
  let pattern_length := 3 in (position 150 pattern_length = 3) ∧ (pattern.drop (position 150 pattern_length - 1)).head = 'Z' :=
by
  sorry

end find_150th_letter_l530_530467


namespace problem_equiv_l530_530271

theorem problem_equiv {△ ✓ ○ □ : ℕ} 
  (h1 : △ + △ = ✓) 
  (h2 : ○ = □ + □) 
  (h3 : △ = ○ + ○ + ○ + ○) : 
  ✓ ÷ □ = 16 :=
sorry

end problem_equiv_l530_530271


namespace number_of_diagonals_in_30_sided_polygon_l530_530761

-- Definition of a convex polygon with 30 sides
def convex_polygon_sides := 30

-- The function to calculate the number of diagonals in a convex polygon with n sides
noncomputable def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Prove that a convex polygon with 30 sides has 405 diagonals
theorem number_of_diagonals_in_30_sided_polygon :
  num_diagonals convex_polygon_sides = 405 :=
by
  calc
    num_diagonals 30 = 30 * (30 - 3) / 2 : rfl
    ... = 30 * 27 / 2 : rfl
    ... = 810 / 2 : rfl
    ... = 405 : rfl

end number_of_diagonals_in_30_sided_polygon_l530_530761


namespace monotonicity_of_f_min_value_of_m_l530_530731

-- Definitions
def f (x a : ℝ) : ℝ := Real.exp (a * x) - a * x - 1

noncomputable def factorial (n : ℕ) : ℝ := (Nat.factorial n).toReal

-- Monotonicity of f(x)
theorem monotonicity_of_f (a : ℝ) :
  (a = 0 → ¬ ∃ I, ∀ x ∈ I, strict_mono_on (λ x, f x a) I)
  ∧ (a ≠ 0 → ∃ I₁ I₂, I₁ = Icc 0 (∞:ℝ) ∧ I₂ = Icc (⨵:ℝ) 0 ∧ 
       strict_anti_on (λ x, f x a) I₂ ∧ strict_mono_on (λ x, f x a) I₁) :=
sorry

-- Minimum value of m
theorem min_value_of_m (m : ℤ) (h : ∀ n: ℕ, 2 ≤ n → (factorial n) ^ (2 / (n * (n - 1) : ℝ)) < m) : 
  m = 3 :=
sorry

end monotonicity_of_f_min_value_of_m_l530_530731


namespace handshake_count_l530_530415

-- Defining the conditions
def number_of_companies : ℕ := 5
def representatives_per_company : ℕ := 5
def total_participants : ℕ := number_of_companies * representatives_per_company

-- Defining the number of handshakes each person makes
def handshakes_per_person : ℕ := total_participants - 1 - (representatives_per_company - 1)

-- Defining the total number of handshakes
def total_handshakes : ℕ := (total_participants * handshakes_per_person) / 2

theorem handshake_count :
  total_handshakes = 250 :=
by
  sorry

end handshake_count_l530_530415


namespace min_distance_points_l530_530737

noncomputable def parabola (p : ℝ) : set (ℝ × ℝ) :=
  { Q | ∃ (x y : ℝ), y^2 = 2 * p * x ∧ Q = (x, y) }

def point_P (p : ℝ) : ℝ × ℝ := (3 * p, 0)

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem min_distance_points (p : ℝ) (Q : ℝ × ℝ) :
  Q ∈ parabola p → distance (point_P p) Q = real.sqrt (5 * p^2) →
  (Q = (2 * p, 2 * p) ∨ Q = (2 * p, -2 * p)) :=
by
  sorry

end min_distance_points_l530_530737


namespace maximize_profit_l530_530038

def C (x : ℝ) : ℝ := 300 + (1/12)*x^3 - 5*x^2 + 170*x

def R (x : ℝ) : ℝ := 134 * x

def L (x : ℝ) : ℝ := R x - C x

theorem maximize_profit : ∃ x : ℝ, 0 ≤ x ∧ x = 36 ∧ L x = 996 := by
    sorry

end maximize_profit_l530_530038


namespace expression_of_f_f_increasing_on_interval_inequality_solution_l530_530246

noncomputable def f (x : ℝ) : ℝ := (x / (1 + x^2))

-- 1. Proving f(x) is the given function
theorem expression_of_f (x : ℝ) (h₁ : f x = (a*x + b) / (1 + x^2)) (h₂ : (∀ x, f (-x) = -f x)) (h₃ : f (1/2) = 2/5) :
  f x = x / (1 + x^2) :=
sorry

-- 2. Prove f(x) is increasing on (-1,1)
theorem f_increasing_on_interval {x₁ x₂ : ℝ} (h₁ : -1 < x₁ ∧ x₁ < 1) (h₂ : -1 < x₂ ∧ x₂ < 1) (h₃ : x₁ < x₂) :
  f x₁ < f x₂ :=
sorry

-- 3. Solve the inequality f(t-1) + f(t) < 0 on (0, 1/2)
theorem inequality_solution (t : ℝ) (h₁ : 0 < t) (h₂ : t < 1/2) :
  f (t - 1) + f t < 0 :=
sorry

end expression_of_f_f_increasing_on_interval_inequality_solution_l530_530246


namespace eq_sqrt3_power_eq_mul_power_l530_530693

-- Problem statement
theorem eq_sqrt3_power_eq_mul_power (m n a : ℝ) (hm : m > 0) (hn : n > 0) (ha : a > 0) (h₁ : a ≠ 1) :
  real.cbrt (m ^ 4 * n ^ 4) = (m * n) ^ (4/3) :=
by sorry

end eq_sqrt3_power_eq_mul_power_l530_530693


namespace find_150th_letter_l530_530472

def pattern : List Char := ['X', 'Y', 'Z']

def position (N : ℕ) (pattern_length : ℕ) : ℕ :=
  if N % pattern_length = 0 then pattern_length else N % pattern_length

theorem find_150th_letter : 
  let pattern_length := 3 in (position 150 pattern_length = 3) ∧ (pattern.drop (position 150 pattern_length - 1)).head = 'Z' :=
by
  sorry

end find_150th_letter_l530_530472


namespace find_x_l530_530069

variable (a b c d e f g h x : ℤ)

def cell_relationships (a b c d e f g h x : ℤ) : Prop :=
  (a = 10) ∧
  (h = 3) ∧
  (a = 10 + b) ∧
  (b = c + a) ∧
  (c = b + d) ∧
  (d = c + h) ∧
  (e = 10 + f) ∧
  (f = e + g) ∧
  (g = d + h) ∧
  (h = g + x)

theorem find_x : cell_relationships a b c d e f g h x → x = 7 :=
sorry

end find_x_l530_530069


namespace largest_n_lemma_l530_530192
noncomputable def largest_n (n b: ℕ) : Prop := 
  n! = ((n-4) + b)!/(b!)

theorem largest_n_lemma : ∀ n: ℕ, ∀ b: ℕ, b ≥ 4 → (largest_n n b → n = 1) := 
  by
  intros n b hb h
  sorry

end largest_n_lemma_l530_530192


namespace find_scalars_l530_530040

noncomputable def vector_a : ℝ × ℝ × ℝ := (2, -1, 2)
noncomputable def vector_b : ℝ × ℝ × ℝ := (1, 2, -2)
noncomputable def vector_c : ℝ × ℝ × ℝ := (-2, 2, 1)
noncomputable def target_vector : ℝ × ℝ × ℝ := (3, -1, 9)

theorem find_scalars (p q r : ℝ) :
  target_vector = (p, q, r) • (vector_a, vector_b, vector_c) →
  (p, q, r) = (25/9, -17/9, 1/9) :=
sorry

end find_scalars_l530_530040


namespace find_f_six_minus_a_l530_530245

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then 2^(x-2) - 2 else -Real.logb 2 (x + 1)

variable (a : ℝ)
axiom h : f a = -3

theorem find_f_six_minus_a : f (6 - a) = - 15 / 8 :=
by
  sorry

end find_f_six_minus_a_l530_530245


namespace calculation_error_in_relay_game_l530_530083

theorem calculation_error_in_relay_game :
  (∀ x : ℝ, x ≠ 1 → x ≠ -1 → (3 * (x + 1) = (x + 1) * (x - 1) - x * (x - 1)) ∧
   (3x + 3 = x^2 + 1 - x^2 + x → false)) :=
by
  intros x h1 h2
  have h_eq : 3 * (x + 1) = (x + 1) * (x - 1) - x * (x - 1) := by sorry
  have h_ineq : 3x + 3 = x^2 + 1 - x^2 + x → false := by sorry
  exact ⟨h_eq, h_ineq⟩

end calculation_error_in_relay_game_l530_530083


namespace solve_proof_problem_l530_530177

def product_of_digits (k : ℕ) : ℕ := k.digits 10).prod

theorem solve_proof_problem (k : ℕ) (h_pos : k > 0) :
product_of_digits k = (25*k - 211 * 8) / 8 → k = 72 ∨ k = 88 :=
sorry

end solve_proof_problem_l530_530177


namespace value_of_expression_l530_530558

variable {a b c d : ℝ}

-- Declare the function g(x)
def g (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- Conditions
def passes_through_points : Prop :=
  g 1 = 4 ∧ g 0 = -2 ∧ g (-1) = -8
  
-- Prove the value of 4a - 2b + c - 3d
theorem value_of_expression (h : passes_through_points) : 4 * a - 2 * b + c - 3 * d = 48 :=
by
  sorry

end value_of_expression_l530_530558


namespace correct_propositions_l530_530595

theorem correct_propositions :
  (∀ (n : ℕ) (y : ℕ → ℝ) (y_hat : ℕ → ℝ) (y_bar : ℝ),
    (1 - (∑ i in Finset.range n, (y i - y_hat i)^2) / (∑ i in Finset.range n, (y i - y_bar)^2)) does not indicate worse fitting when it is larger) →
  (∀ (f : ℝ → ℝ) (x0 : ℝ), differentiable ℝ f → has_extremum_at f x0 → deriv f x0 = 0) →
  (inductive_reasoning : reasoning from specific to general) →
  (deductive_reasoning : reasoning from general to specific) →
  ((synthetic_method : proves by seeking the cause from the effect) ∧
   (analytic_method : proves by seeking the cause from the result)) →
  true :=
by
  intros h1 h2 h3 h4 h5
  sorry

end correct_propositions_l530_530595


namespace distance_A_OF_l530_530286

-- Given triangle ABC with angle bisector, right angles, and specific conditions
variables (A B C D E F O : Point)
variables (AE_40 : length (segment A E) = 40)
variables (AB_AC_34 : length (segment A B) / length (segment A C) = 3 / 4)
variables (angle_ABD_90 : angle (line A B) (line B D) = 90)
variables (angle_ACE_90 : angle (line A C) (line C E) = 90)
variables (F_intersection_AE_BC : ∃ F : Point, lies_on F (line A E) ∧ lies_on F (line B C))
variables (O_circumcenter_AFC : is_circumcenter O (triangle A F C))
variables (BD_bisects_EF : is_bisector (segment B D) (segment E F))

-- Prove the perpendicular distance from A to OF is 10√3
theorem distance_A_OF :
  let d := perpendicular_distance (point A) (line O F) in
  d = 10 * sqrt 3 :=
sorry

end distance_A_OF_l530_530286


namespace number_of_diagonals_in_30_sided_polygon_l530_530757

-- Definition of a convex polygon with 30 sides
def convex_polygon_sides := 30

-- The function to calculate the number of diagonals in a convex polygon with n sides
noncomputable def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Prove that a convex polygon with 30 sides has 405 diagonals
theorem number_of_diagonals_in_30_sided_polygon :
  num_diagonals convex_polygon_sides = 405 :=
by
  calc
    num_diagonals 30 = 30 * (30 - 3) / 2 : rfl
    ... = 30 * 27 / 2 : rfl
    ... = 810 / 2 : rfl
    ... = 405 : rfl

end number_of_diagonals_in_30_sided_polygon_l530_530757


namespace compute_expression_l530_530617

theorem compute_expression (x : ℝ) (h : x = 8) : 
  (x^6 - 64 * x^3 + 1024) / (x^3 - 16) = 480 :=
by
  rw [h]
  sorry

end compute_expression_l530_530617


namespace inequality_proof_l530_530710

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 3) :
  (a^2 + 9) / (2*a^2 + (b+c)^2) + (b^2 + 9) / (2*b^2 + (c+a)^2) + (c^2 + 9) / (2*c^2 + (a+b)^2) ≤ 5 :=
by
  sorry

end inequality_proof_l530_530710


namespace problem_inequality_l530_530892

theorem problem_inequality (n : ℕ) (a : ℕ → ℝ) (h_pos : ∀ i, 0 < a i) (h_n_ge_two : n ≥ 2) (h_sum_eq : (finset.range n).sum (λ i, a i) = S) :
  (finset.range n).sum (λ i, a i / (S - a i)) ≥ n / (n - 1) := 
begin
  sorry,
end

end problem_inequality_l530_530892


namespace total_weight_of_remaining_eggs_is_correct_l530_530355

-- Define the initial conditions and the question as Lean definitions
def total_eggs : Nat := 12
def weight_per_egg : Nat := 10
def num_boxes : Nat := 4
def melted_boxes : Nat := 1

-- Calculate the total weight of the eggs
def total_weight : Nat := total_eggs * weight_per_egg

-- Calculate the number of eggs per box
def eggs_per_box : Nat := total_eggs / num_boxes

-- Calculate the weight per box
def weight_per_box : Nat := eggs_per_box * weight_per_egg

-- Calculate the number of remaining boxes after one is tossed out
def remaining_boxes : Nat := num_boxes - melted_boxes

-- Calculate the total weight of the remaining chocolate eggs
def remaining_weight : Nat := remaining_boxes * weight_per_box

-- The proof task
theorem total_weight_of_remaining_eggs_is_correct : remaining_weight = 90 := by
  sorry

end total_weight_of_remaining_eggs_is_correct_l530_530355


namespace max_distance_is_15_l530_530612

noncomputable def max_distance_between_cars (v_A v_B: ℝ) (a: ℝ) (D: ℝ) : ℝ :=
  if v_A > v_B ∧ D = a + 60 then (a * (1 - a / 60)) else 0

theorem max_distance_is_15 (v_A v_B: ℝ) (a: ℝ) (D: ℝ) :
  v_A > v_B ∧ D = a + 60 → max_distance_between_cars v_A v_B a D = 15 :=
by
  sorry

end max_distance_is_15_l530_530612


namespace find_150th_letter_l530_530455
open Nat

def repeating_sequence := "XYZ"

def length_repeating_sequence := 3

theorem find_150th_letter : (150 % length_repeating_sequence == 0) → repeating_sequence[(length_repeating_sequence - 1) % length_repeating_sequence] = 'Z' := 
by
  sorry

end find_150th_letter_l530_530455


namespace find_integer_l530_530181

theorem find_integer (n : ℤ) (h1 : 10 ≤ n) (h2 : n ≤ 15) (h3 : n ≡ 12345 [MOD 7]) : n = 11 :=
sorry

end find_integer_l530_530181


namespace find_ac_pair_l530_530953

theorem find_ac_pair (a c : ℤ) (h1 : a + c = 37) (h2 : a < c) (h3 : 36^2 - 4 * a * c = 0) : a = 12 ∧ c = 25 :=
by
  sorry

end find_ac_pair_l530_530953


namespace prob_one_head_one_tail_l530_530974

theorem prob_one_head_one_tail (h1 h2 : bool) (H : h1 = tt ∨ h1 = ff) (T : h2 = tt ∨ h2 = ff):
  (h1 = tt ∧ h2 = ff) ∨ (h1 = ff ∧ h2 = tt) →
  real := 1 / 2 :=
by
  sorry

end prob_one_head_one_tail_l530_530974


namespace union_complement_eq_l530_530739

open Set

def U : Set ℕ := {1, 2, 3}
def A : Set ℕ := {1}
def B : Set ℕ := {1, 2}

theorem union_complement_eq : A ∪ (U \ B) = {1, 3} := by
  sorry

end union_complement_eq_l530_530739


namespace distinguishable_cube_colorings_l530_530117

theorem distinguishable_cube_colorings : ∃ n : ℕ, n = 92 ∧ 
  let faces := 6 in
  let colors := 4 in
  n = count_distinguishable_colorings faces colors :=
sorry

def count_distinguishable_colorings (faces colors : ℕ) : ℕ :=
  -- Function to calculate the number of distinguishable colorings
  sorry

end distinguishable_cube_colorings_l530_530117


namespace count_triply_divisible_by_6_l530_530126

/-- A function to check if a number is triply. -/
def is_triply (n : Nat) : Prop :=
  let digits := n.digits 10
  ∃ (a b c : Nat), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
    digits = [a, b, c, a, b, c]

/-- A function to check if a number is divisible by 6. -/
def divisible_by_6 (n : Nat) : Prop :=
  n % 2 = 0 ∧ (n.digits 10).sum % 3 = 0

/-- The main statement to prove. -/
theorem count_triply_divisible_by_6 : 
  ∃ (count : Nat), count = 3 ∧ 
  count = (Finset.range 1000000).filter (λ n, n ≥ 100000 ∧ n < 1000000 ∧ is_triply n ∧ divisible_by_6 n).card :=
by 
  sorry

end count_triply_divisible_by_6_l530_530126


namespace car_distance_traveled_l530_530111

theorem car_distance_traveled (d : ℝ)
  (h_avg_speed : 84.70588235294117 = 320 / ((d / 90) + (d / 80))) :
  d = 160 :=
by
  sorry

end car_distance_traveled_l530_530111


namespace sin_180_eq_zero_l530_530622

theorem sin_180_eq_zero : Real.sin (180 * Real.pi / 180) = 0 := by
  -- Simplifying the angle, 180 degrees = π radians
  let angle := 180 * Real.pi / 180
  have h : angle = Real.pi := by
    simp [angle, Real.pi]
  rw h
  -- From the unit circle, we know the sine of angle π is 0
  exact Real.sin_pi

end sin_180_eq_zero_l530_530622


namespace alan_total_cost_is_84_l530_530580

def num_dark_cds : ℕ := 2
def num_avn_cds : ℕ := 1
def num_90s_cds : ℕ := 5
def price_avn_cd : ℕ := 12 -- in dollars
def price_dark_cd : ℕ := price_avn_cd * 2
def total_cost_other_cds : ℕ := num_dark_cds * price_dark_cd + num_avn_cds * price_avn_cd
def price_90s_cds : ℕ := ((40 : ℕ) * total_cost_other_cds) / 100
def total_cost_all_products : ℕ := num_dark_cds * price_dark_cd + num_avn_cds * price_avn_cd + price_90s_cds

theorem alan_total_cost_is_84 : total_cost_all_products = 84 := by
  sorry

end alan_total_cost_is_84_l530_530580


namespace quadrilateral_rhombus_condition_l530_530143

variables (A B C D : Type) [decidable_eq A] [decidable_eq B] [decidable_eq C] [decidable_eq D]

def is_quadrilateral (A B C D : Type) : Prop :=
  true -- a placeholder definition to represent a quadrilateral

def is_rhombus (A B C D : Type) : Prop :=
  true -- a placeholder definition to represent a rhombus

def is_perpendicular (AC BD : Type) : Prop :=
  true -- a placeholder definition to represent perpendicular lines

-- We state that given AC is perpendicular to BD, ABCD is a rhombus
theorem quadrilateral_rhombus_condition (h : is_perpendicular AC BD) : is_rhombus A B C D :=
sorry

end quadrilateral_rhombus_condition_l530_530143


namespace handshake_count_l530_530418

theorem handshake_count (num_companies : ℕ) (num_representatives : ℕ) 
  (total_handshakes : ℕ) (h1 : num_companies = 5) (h2 : num_representatives = 5)
  (h3 : total_handshakes = (num_companies * num_representatives * 
   (num_companies * num_representatives - 1 - (num_representatives - 1)) / 2)) :
  total_handshakes = 250 :=
by
  rw [h1, h2] at h3
  exact h3

end handshake_count_l530_530418


namespace find_150th_letter_l530_530471

def pattern : List Char := ['X', 'Y', 'Z']

def position (N : ℕ) (pattern_length : ℕ) : ℕ :=
  if N % pattern_length = 0 then pattern_length else N % pattern_length

theorem find_150th_letter : 
  let pattern_length := 3 in (position 150 pattern_length = 3) ∧ (pattern.drop (position 150 pattern_length - 1)).head = 'Z' :=
by
  sorry

end find_150th_letter_l530_530471


namespace percentage_attended_picnic_l530_530290

def total_employees : ℕ := 100
def percentage_men : ℝ := 0.45
def percentage_women : ℝ := 1 - percentage_men
def percentage_men_at_picnic : ℝ := 0.20
def percentage_women_at_picnic : ℝ := 0.40

theorem percentage_attended_picnic : 
  (9 / 100 + 22 / 100) * 100 = 31 :=
by
  have total_men : ℕ := percentage_men * total_employees
  have total_women : ℕ := percentage_women * total_employees
  have men_at_picnic : ℕ := percentage_men_at_picnic * total_men
  have women_at_picnic : ℕ := percentage_women_at_picnic * total_women
  have total_at_picnic : ℕ := men_at_picnic + women_at_picnic
  have result : ℝ := (total_at_picnic.toRat / total_employees.toRat) * 100
  exact (31 : ℝ)

end percentage_attended_picnic_l530_530290


namespace find_AC_l530_530278

def φ : ℝ := (Real.sqrt 5 + 1) / 2

def golden_section (AB AC : ℝ) : Prop :=
  (AB / AC = AC / (AB - AC))

theorem find_AC (AB : ℝ) (AC : ℝ) (h1 : AB = 8) (h2 : AC > (AB - AC)) (h3 : golden_section AB AC) :
  AC = 4 * (Real.sqrt 5 - 1) :=
sorry

end find_AC_l530_530278


namespace problem1_l530_530095

theorem problem1 (α : ℝ) (h : Real.tan (π / 4 + α) = 1 / 2) :
  (Real.sin (2 * α) - Real.cos α ^ 2) / (1 + Real.cos (2 * α)) = -5 / 6 := 
  sorry

end problem1_l530_530095


namespace sin_eq_implies_necessary_but_not_sufficient_l530_530537

theorem sin_eq_implies_necessary_but_not_sufficient (A B : ℝ) : 
  (sin A = sin B) ↔ (A = B + 2 * Int.pi * n) :=
begin
  sorry 
end

end sin_eq_implies_necessary_but_not_sufficient_l530_530537


namespace equilateral_triangle_area_half_side_l530_530387

noncomputable def equilateral_triangle_perimeter (s : ℝ) (s_pos : 0 < s) : ℝ :=
  3 * s

theorem equilateral_triangle_area_half_side (s : ℝ) (s_pos : s ≠ 0) :
  (s^2 * Real.sqrt 3) / 4 = s / 2 → equilateral_triangle_perimeter s s_pos = 2 * Real.sqrt 3 :=
by
  intro h
  sorry

end equilateral_triangle_area_half_side_l530_530387


namespace find_m_value_find_lambda_value_l530_530862

noncomputable def m_solution1 (a : ℝ) (P : ℝ × ℝ) : ℝ :=
-1 / 2

noncomputable def m_solution2 (a : ℝ) (P : ℝ × ℝ) : ℝ :=
-3 / 2

theorem find_m_value (a : ℝ) (h : 0 < a)
    (ecc : ℝ := sqrt 2 / 2)
    (P : ℝ × ℝ)
    (m : ℝ)
    (hP : ((P.2 / (P.1 - a)) * (P.2 / (P.1 + a))) = m)
    (hv : m * P.1^2 - P.2^2 = m * a^2) :
  m = m_solution1 a P ∨ m = m_solution2 a P :=
sorry

theorem find_lambda_value (a : ℝ) (A B M : ℝ × ℝ) (λ : ℝ)
  (hA : A = (0, -sqrt 2 / 2 * a) ∨ A = (2 * sqrt 2 / 3 * a, sqrt 2 / 6 * a))
  (hB : B = A ∨ B = (0, -sqrt 2 / 2 * a) ∨ B = (2 * sqrt 2 / 3 * a, sqrt 2 / 6 * a))
  (hM : M = (λ * A.1 + (1 - λ) * B.1, λ * A.2 + (1 - λ) * B.2))
  (hlellipse : (A.1 ≠ B.1 ∧
    A.2 ≠ B.2 ∧ 
    M.1^2 + 2 * M.2^2 = a^2)) :
  λ = 0 ∨ λ = 2 / 3 :=
sorry

end find_m_value_find_lambda_value_l530_530862


namespace row_arith_seq_impossible_l530_530655

def a (i j : ℕ) : ℕ := (i^2 + j) * (i + j^2)

theorem row_arith_seq_impossible :
  ∀ M : Matrix (Fin 7) (Fin 7) ℕ,
    (∀ i j, M i j = a i j) →
    ¬ ∃ M' : Matrix (Fin 7) (Fin 7) ℕ,
      (∀ i, ∃ p d : ℕ, ∀ j, M' i j = p + d * j) ∧
      (∀ i j, M i j = M' i j + arbitrary_sequence i j) := 
sorry

attribute [irreducible] a

end row_arith_seq_impossible_l530_530655


namespace num_ways_pay_l530_530875

theorem num_ways_pay : 
  let n : ℕ := 2010,
      num_solutions : ℕ := (Nat.choose (201 + 2) 2)
  in ∑ x y z in { (x' : ℕ) }, 2 * x' + 5 * y' + 10 * z' = 2010 = num_solutions :=
by
  sorry

end num_ways_pay_l530_530875


namespace largest_n_factorial_product_l530_530188

theorem largest_n_factorial_product :
  ∃ n : ℕ, (∀ a : ℕ, (n > 0) → (n! = (∏ k in finset.range (n - 4 + a), k + 1))) → n = 4 :=
begin
  sorry
end

end largest_n_factorial_product_l530_530188


namespace solution_set_of_inequality_l530_530034

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 2 * x + 3 ≥ 0 } = {x : ℝ | -1 ≤ x ∧ x ≤ 3 } :=
sorry

end solution_set_of_inequality_l530_530034


namespace problem_statement_l530_530879

theorem problem_statement :
  let N := (setOf (λ (x : ℤ × ℤ), 4 * x.1 ^ 2 + 9 * x.2 ^ 2 ≤ 1000000000)).card
  let a := N.digits 10 0
  let b := N.digits 10 1
  in 10 * a + b = 52 :=
by
  sorry

end problem_statement_l530_530879


namespace joan_change_received_l530_530870

/-- Definition of the cat toy cost -/
def cat_toy_cost : ℝ := 8.77

/-- Definition of the cage cost -/
def cage_cost : ℝ := 10.97

/-- Definition of the total cost -/
def total_cost : ℝ := cat_toy_cost + cage_cost

/-- Definition of the payment amount -/
def payment : ℝ := 20.00

/-- Definition of the change received -/
def change_received : ℝ := payment - total_cost

/-- Statement proving that Joan received $0.26 in change -/
theorem joan_change_received : change_received = 0.26 := by
  sorry

end joan_change_received_l530_530870


namespace number_of_diagonals_in_30_sided_polygon_l530_530756

-- Definition of a convex polygon with 30 sides
def convex_polygon_sides := 30

-- The function to calculate the number of diagonals in a convex polygon with n sides
noncomputable def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Prove that a convex polygon with 30 sides has 405 diagonals
theorem number_of_diagonals_in_30_sided_polygon :
  num_diagonals convex_polygon_sides = 405 :=
by
  calc
    num_diagonals 30 = 30 * (30 - 3) / 2 : rfl
    ... = 30 * 27 / 2 : rfl
    ... = 810 / 2 : rfl
    ... = 405 : rfl

end number_of_diagonals_in_30_sided_polygon_l530_530756


namespace probability_of_multiple_of_3_l530_530102

-- For handling probability
noncomputable def probability {α : Type*} [Fintype α] (s : Finset α) : ℚ :=
  (s.card : ℚ) / (Fintype.card α)

-- Definitions to match conditions
def digits : Finset ℕ := {1, 2, 3, 4, 5}
def draws : Finset (Finset ℕ) := digits.powerset.filter (λ s, s.card = 4)

-- Function to determine if a set of digits sums to a multiple of 3
def sum_to_multiple_of_3 (s : Finset ℕ) : Prop :=
  s.sum id % 3 = 0

-- Set of draws whose sum of digits is a multiple of 3
def valid_draws := draws.filter sum_to_multiple_of_3

-- Total number of ways to order four digits from the draw
def total_permuted_draws : Finset (List ℕ) :=
  draws.bUnion (λ s, s.permutations)

-- Number of valid (i.e., multiple of 3) four-digit numbers
def valid_permuted_draws : Finset (List ℕ) :=
  valid_draws.bUnion (λ s, s.permutations)

-- The theorem to be proved
theorem probability_of_multiple_of_3 : probability valid_permuted_draws total_permuted_draws = 1 / 5 := by
  sorry

end probability_of_multiple_of_3_l530_530102


namespace smallest_common_multiple_8_6_l530_530514

theorem smallest_common_multiple_8_6 : ∃ n : ℕ, n > 0 ∧ (8 ∣ n) ∧ (6 ∣ n) ∧ ∀ m : ℕ, m > 0 ∧ (8 ∣ m) ∧ (6 ∣ m) → n ≤ m :=
begin
  use 24,
  split,
  { exact zero_lt_24, },
  split,
  { exact dvd.intro 3 rfl, },
  split,
  { exact dvd.intro 4 rfl, },
  intros m hm h8 h6,
  -- actual proof here
  sorry
end

end smallest_common_multiple_8_6_l530_530514


namespace Lagrange_interpolation_poly_l530_530535

noncomputable def Lagrange_interpolation (P : ℝ → ℝ) : Prop :=
  P (-1) = -11 ∧ P (1) = -3 ∧ P (2) = 1 ∧ P (3) = 13

theorem Lagrange_interpolation_poly :
  ∃ P : ℝ → ℝ, Lagrange_interpolation P ∧ ∀ x, P x = x^3 - 2*x^2 + 3*x - 5 :=
by
  sorry

end Lagrange_interpolation_poly_l530_530535


namespace angle_between_vectors_l530_530745

variable {V : Type*} [InnerProductSpace ℝ V]

variables (a b : V)
variables (ha : ∥a∥ = 2)
variables (hb : ∥b∥ = 1)
variables (h : ⟪b, b - a⟫ = 2)

theorem angle_between_vectors (a b : V) (ha : ∥a∥ = 2) (hb : ∥b∥ = 1) (h : ⟪b, b - a⟫ = 2) :
  real.angle a b = 2 * real.pi / 3 :=
sorry

end angle_between_vectors_l530_530745


namespace modulus_z_l530_530235

noncomputable def z : ℂ := (7 - complex.i) / (1 + complex.i)

theorem modulus_z : complex.abs z = 5 := by sorry

end modulus_z_l530_530235


namespace diagonal_count_of_convex_polygon_30_sides_l530_530797
-- Importing the entire Mathlib library for necessary mathematical constructs

theorem diagonal_count_of_convex_polygon_30_sides : 
  let n := 30 in
  let num_diagonals := (n * (n - 3)) / 2 in
  num_diagonals = 405 := 
by {
  let n := 30
  let num_diagonals := (n * (n - 3)) / 2
  show num_diagonals = 405,
  sorry
}

end diagonal_count_of_convex_polygon_30_sides_l530_530797


namespace collinear_I_D_K_l530_530394

variables {A B C I K L M N D B₁: Type} 

-- Conditions
variables [IsIncircleCenter I A B C]
variables [IsIncircleTouchPoint K A C I]
variables [IsIncircleTouchPoint L A B I]
variables [IsIncircleTouchPoint M B C I]
variables [IsMedian B B₁ A C]
variables [IsSegment MN M N]
variables [IsIntersection D BB₁ MN]

-- Proof goal
theorem collinear_I_D_K :
  Collinear I D K := 
sorry

end collinear_I_D_K_l530_530394


namespace base_three_to_base_ten_l530_530072

theorem base_three_to_base_ten (n : ℕ) (h : n = 20121) : 
  let digits := [1, 2, 0, 1, 2] in
  let base := 3 in
  ∑ i in finset.range digits.length, (digits[i] * base^i) = 178 :=
by sorry

end base_three_to_base_ten_l530_530072


namespace probability_double_head_l530_530423

/-- There are three types of coins:
  - Coin 1: one head and one tail
  - Coin 2: two heads
  - Coin 3: two tails
  A coin is randomly selected and flipped, resulting in heads. 
  Prove that the probability that the coin is Coin 2 (the double head coin)
  is 2/3. -/
theorem probability_double_head (h : true) : 
  let Coin1 : ℕ := 1,
      Coin2 : ℕ := 2,
      Coin3 : ℕ := 0,
      totalHeads : ℕ := Coin1 + Coin2 + Coin3
  in 
  let p_Coin1 := 1 / 3,
      p_Coin2 := 1 / 3,
      p_Coin3 := 1 / 3,
      p_Heads_given_Coin1 := 1 / 2,
      p_Heads_given_Coin2 := 1,
      p_Heads_given_Coin3 := 0,
      p_Heads := (p_Heads_given_Coin1 * p_Coin1) + (p_Heads_given_Coin2 * p_Coin2) + (p_Heads_given_Coin3 * p_Coin3)
  in (p_Heads_given_Coin2 * p_Coin2) / p_Heads = 2 / 3 :=
by
  sorry

end probability_double_head_l530_530423


namespace letter_150th_in_pattern_l530_530493

def repeating_sequence := "XYZ"

def letter_at_position (n : ℕ) : char :=
  let seq := repeating_sequence.to_list
  seq.get! ((n - 1) % seq.length)

theorem letter_150th_in_pattern : letter_at_position 150 = 'Z' :=
by sorry

end letter_150th_in_pattern_l530_530493


namespace simulate_die_toss_l530_530368

/-- 
Design a random simulation experiment for tossing a die to get an odd or even number. 
One accepted method is to use a calculator to generate a random number between 0 and 1.
-/
theorem simulate_die_toss :
  ∃ (f : ℝ → ℕ), (∀ r : ℝ, 0 ≤ r ∧ r < 0.5 → f r = 1) ∧ 
  (∀ r : ℝ, 0.5 ≤ r ∧ r ≤ 1 → f r = 2) :=
begin
  -- Simulation experiment generator function
  have f : ℝ → ℕ := λ r, if r < 0.5 then 1 else 2,
  -- Prove that 'f' satisfies the conditions for odds and evens
  use f,
  split,
  { intros r hr,
    simp only [hr, if_true],
    linarith },
  { intros r hr,
    simp only [hr, if_false],
    linarith },
  sorry
end

end simulate_die_toss_l530_530368


namespace problem_solution_l530_530262

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem problem_solution (t : ℝ) (h : ∀ (k : ℝ), (t - 2 + 2 * 1 = k * 1) ∧ (3 + 2 * -1 = k * -1)) :
  vector_magnitude ((t - 2, 3) - (1, -1)) = 4 * real.sqrt 2 :=
by
  let a := (t - 2, 3)
  let b := (1, -1)
  let ab := (a.1 - b.1, a.2 - b.2)
  have h1 : t = -1 := sorry
  rw h1
  rw [sub_eq_add_neg, add_assoc, add_neg_cancel_right]
  unfold vector_magnitude
  have : (a.1 - b.1) = -4 := by norm_num
  have : (a.2 - b.2) = 4 := by norm_num
  rw this
  norm_num
  exact rfl

end problem_solution_l530_530262


namespace candies_total_l530_530347

theorem candies_total (N a S : ℕ) (h1 : S = 2 * a + 7) (h2 : S = N * a) (h3 : a > 1) (h4 : N = 3) : S = 21 := 
sorry

end candies_total_l530_530347


namespace ratio_of_boys_l530_530292

theorem ratio_of_boys (p : ℝ) (h : p = (3/4) * (1 - p)) : 
  p = 3 / 7 := 
by 
  sorry

end ratio_of_boys_l530_530292


namespace candies_total_l530_530346

theorem candies_total (N a S : ℕ) (h1 : S = 2 * a + 7) (h2 : S = N * a) (h3 : a > 1) (h4 : N = 3) : S = 21 := 
sorry

end candies_total_l530_530346


namespace scissors_total_l530_530965

theorem scissors_total (original_scissors : ℕ) (added_scissors : ℕ) (total_scissors : ℕ) 
  (h1 : original_scissors = 39)
  (h2 : added_scissors = 13)
  (h3 : total_scissors = original_scissors + added_scissors) : total_scissors = 52 :=
by
  rw [h1, h2] at h3
  exact h3

end scissors_total_l530_530965


namespace number_of_paths_l530_530159

theorem number_of_paths (m n : ℕ) : 
  (finset.card {p : list (ℕ × ℕ) | path_from (0, 0) p (m, n) (λ ⟨x,y⟩, (x + 1 = y ∨ x = y + 1))} = nat.choose (m+n) n) :=
sorry

end number_of_paths_l530_530159


namespace problem_1_problem_2_l530_530332

-- Definitions for problem (1)
def p (x a : ℝ) := x^2 - 4 * a * x + 3 * a^2 < 0 ∧ a > 0
def q (x : ℝ) := x^2 - x - 6 ≤ 0 ∧ x^2 + 3 * x - 10 > 0

-- Statement for problem (1)
theorem problem_1 (a : ℝ) (h : p 1 a ∧ q x) : 2 < x ∧ x < 3 :=
by 
  sorry

-- Definitions for problem (2)
def neg_p (x a : ℝ) := ¬ (x^2 - 4 * a * x + 3 * a^2 < 0 ∧ a > 0)
def neg_q (x : ℝ) := ¬ (x^2 - x - 6 ≤ 0 ∧ x^2 + 3 * x - 10 > 0)

-- Statement for problem (2)
theorem problem_2 (a : ℝ) (h : ∀ x, neg_p x a → neg_q x ∧ ¬ (neg_q x → neg_p x a)) : 1 < a ∧ a ≤ 2 :=
by 
  sorry

end problem_1_problem_2_l530_530332


namespace find_angle_and_side_l530_530848

theorem find_angle_and_side (B: ℝ) (c: ℝ) (a: ℝ) (b: ℝ) (h1: sqrt (3) * sin (2 * B) = 2 * (sin B)^2) (h2: a = 4) (h3: b = 2 * sqrt (7)) : 
  B = pi / 3 ∧ c = 6 := 
  by sorry

end find_angle_and_side_l530_530848


namespace sin_180_degrees_l530_530637

theorem sin_180_degrees : Real.sin (180 * Real.pi / 180) = 0 := 
by
  sorry

end sin_180_degrees_l530_530637


namespace problem_l530_530839

theorem problem (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 10) : a^2 + b^2 = 29 :=
by
  sorry

end problem_l530_530839


namespace min_convex_number_l530_530028

noncomputable def minimum_convex_sets (A B C : ℝ × ℝ) : ℕ :=
  if A ≠ B ∧ B ≠ C ∧ C ≠ A then 3 else 4

theorem min_convex_number (A B C : ℝ × ℝ) (h : A ≠ B ∧ B ≠ C ∧ C ≠ A) :
  minimum_convex_sets A B C = 3 :=
by 
  sorry

end min_convex_number_l530_530028


namespace diagonals_in_convex_polygon_with_30_sides_l530_530766

theorem diagonals_in_convex_polygon_with_30_sides : 
  let n := 30 in
  ∑ i j in finset.range n, (i + 1 - 1) * (j + 1 - 1) * (nat.choose 30 2 - 30) / 2 = 202 := 
by
  sorry

end diagonals_in_convex_polygon_with_30_sides_l530_530766


namespace cone_volume_l530_530703

theorem cone_volume (SA SB : ℝ) (angle_SA_base : ℝ) (area_triangle_SAB : ℝ) :
  angle_SA_base = 30 → SB = SA → area_triangle_SAB = 8 →
  let r := 2 * Real.sqrt 3
  let h := 2
  let volume := (1 / 3) * Real.pi * r^2 * h
  volume = 8 * Real.pi :=
by
  intros hangle_SA_base hSB harea_triangle_SAB
  have hSA := by
    have h : (1/2) * SA^2 = 8 := harea_triangle_SAB
    sorry
  sorry

end cone_volume_l530_530703


namespace find_150th_letter_l530_530458

theorem find_150th_letter :
  let pattern := ['X', 'Y', 'Z']
  150 % 3 = 0 -> pattern[(150 % 3 + 2) % 3] = 'Z' :=
begin
  intros pattern h,
  simp at *,
  exact rfl,
end

end find_150th_letter_l530_530458


namespace train_b_speed_is_202_5_l530_530533

noncomputable def speed_of_train_b (vb : ℝ) : Prop :=
  let t := vb / (vb + 360) in
  810 = 4 * vb

theorem train_b_speed_is_202_5 : 
  ∃ vb : ℝ, speed_of_train_b vb ∧ vb = 202.5 :=
by
  use 202.5
  unfold speed_of_train_b
  sorry

end train_b_speed_is_202_5_l530_530533


namespace value_of_expression_l530_530696

theorem value_of_expression (m n : ℤ) (hm : |m| = 3) (hn : |n| = 2) (hmn : m < n) :
  m^2 + m * n + n^2 = 7 ∨ m^2 + m * n + n^2 = 19 := by
  sorry

end value_of_expression_l530_530696


namespace problem1_problem2_l530_530608

variables (a b : ℝ)

theorem problem1 : ((a^2)^3 / (-a)^2) = a^4 :=
sorry

theorem problem2 : ((a + 2 * b) * (a + b) - 3 * a * (a + b)) = -2 * a^2 + 2 * b^2 :=
sorry

end problem1_problem2_l530_530608


namespace diagonal_count_of_convex_polygon_30_sides_l530_530798
-- Importing the entire Mathlib library for necessary mathematical constructs

theorem diagonal_count_of_convex_polygon_30_sides : 
  let n := 30 in
  let num_diagonals := (n * (n - 3)) / 2 in
  num_diagonals = 405 := 
by {
  let n := 30
  let num_diagonals := (n * (n - 3)) / 2
  show num_diagonals = 405,
  sorry
}

end diagonal_count_of_convex_polygon_30_sides_l530_530798


namespace multiples_of_7_in_q_l530_530088

-- Conditions
variables (a b : ℤ)
def multiple_of_14 (x : ℤ) : Prop := ∃ k : ℤ, x = 14 * k
def q : set ℤ := {x | a ≤ x ∧ x ≤ b}

-- Given definitions
variables (ha : multiple_of_14 a) (hb : multiple_of_14 b)
def count_multiples_of (n : ℤ) (s : set ℤ) : ℕ := s.count (λ x, ∃ k : ℤ, x = n * k)

-- Given condition
variables (hmul14 : count_multiples_of 14 q = 9)

-- Proof problem
theorem multiples_of_7_in_q : count_multiples_of 7 q = 18 :=
by sorry

end multiples_of_7_in_q_l530_530088


namespace part1_proof_l530_530341

variable (α β t x1 x2 : ℝ)

-- Conditions
def quadratic_roots := 2 * α ^ 2 - t * α - 2 = 0 ∧ 2 * β ^ 2 - t * β - 2 = 0
def roots_relation := α + β = t / 2 ∧ α * β = -1
def points_in_interval := α < β ∧ α ≤ x1 ∧ x1 ≤ β ∧ α ≤ x2 ∧ x2 ≤ β ∧ x1 ≠ x2

-- Proof of Part 1
theorem part1_proof (h1 : quadratic_roots α β t) (h2 : roots_relation α β t)
                    (h3 : points_in_interval α β x1 x2) : 
                    4 * x1 * x2 - t * (x1 + x2) - 4 < 0 := 
sorry

end part1_proof_l530_530341


namespace max_area_ABCD_l530_530881

theorem max_area_ABCD
  (A B C D : Point)
  (hABCD_convex : ConvexQuadrilateral A B C D)
  (hBC : dist B C = 3)
  (hCD : dist C D = 4)
  (hABD_right : RightTriangle A B D)
  (h_centroids : RightTriangleCenters (centroid A B C) (centroid B C D) (centroid A C D)) :
  ∃ x y, area (Quadrilateral A B C D) = (1 / 2) * x * y + 6 :=
sorry

end max_area_ABCD_l530_530881


namespace multiples_of_eight_l530_530392

open Real

noncomputable def f (n : ℕ) : ℝ := (1 / sqrt 5) * ((1 + sqrt 5) / 2)^n - (1 / sqrt 5) * ((1 - sqrt 5) / 2)^n

noncomputable def S (n : ℕ) : ℝ := ∑ k in Finset.range (n + 1), (Nat.choose n k) * f k

theorem multiples_of_eight (n : ℕ) : 8 ∣ S n ↔ 3 ∣ n :=
sorry

end multiples_of_eight_l530_530392


namespace letter_at_position_150_l530_530479

theorem letter_at_position_150 : 
  (∀ n, n > 0 → ∃ i, i ∈ {1, 2, 3} ∧ "XYZ".to_list[i-1] = "XYZ".to_list[(n - 1) % 3]) →
  ("XYZ".to_list[(150 - 1) % 3] = 'Z') :=
by
  sorry

end letter_at_position_150_l530_530479


namespace probability_exactly_two_singers_same_province_l530_530557

-- Defining the number of provinces and number of singers per province
def num_provinces : ℕ := 6
def singers_per_province : ℕ := 2

-- Total number of singers
def num_singers : ℕ := num_provinces * singers_per_province

-- Define the total number of ways to choose 4 winners from 12 contestants
def total_combinations : ℕ := Nat.choose num_singers 4

-- Define the number of favorable ways to select exactly two singers from the same province and two from two other provinces
def favorable_combinations : ℕ := 
  (Nat.choose num_provinces 1) *  -- Choose one province for the pair
  (Nat.choose (num_provinces - 1) 2) *  -- Choose two remaining provinces
  (Nat.choose singers_per_province 1) *
  (Nat.choose singers_per_province 1)

-- Calculate the probability
def probability : ℚ := favorable_combinations / total_combinations

-- Stating the theorem to be proved
theorem probability_exactly_two_singers_same_province : probability = 16 / 33 :=
by
  sorry

end probability_exactly_two_singers_same_province_l530_530557


namespace maximum_prime_count_at_k_eq_1_l530_530176

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes (l : List ℕ) : ℕ :=
  l.countp is_prime

def max_primes_in_sequence (k : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 0 → n ≠ 1 → 
  count_primes (List.range' (k+1) 10) ≥ 
  count_primes (List.range' (n+1) 10)

theorem maximum_prime_count_at_k_eq_1 : 
  ∃ k : ℕ, k = 1 ∧ max_primes_in_sequence k :=
sorry

end maximum_prime_count_at_k_eq_1_l530_530176


namespace simplify_expression_l530_530380

variable (a : ℝ)

#check eq.subst

-- Define the conditions
def condition (a : ℝ) : Prop := a = real.sqrt 3 + 1/2

-- State the problem
theorem simplify_expression : condition a → 
  (a - real.sqrt 3) * (a + real.sqrt 3) - a * (a - 6) = 6 * real.sqrt 3 :=
by
  intro ha
  rw [condition] at ha
  rw ha
  sorry

end simplify_expression_l530_530380


namespace jo_climb_ways_l530_530317

-- Define the function g
def g : ℕ → ℕ
| 0       := 1
| 1       := 0
| 2       := 1
| 3       := 1
| (n + 1) := g (n - 1) + g (n - 3)

-- The theorem to prove the number of ways Jo can climb 8 stairs is 4
theorem jo_climb_ways : g 8 = 4 := 
sorry

end jo_climb_ways_l530_530317


namespace compare_xyz_l530_530691

noncomputable def x : ℝ := 5 ^ (Real.log 3.4 / Real.log 2)
noncomputable def y : ℝ := 5 ^ ((Real.log 3.6 / Real.log 2) / 2)
noncomputable def z : ℝ := 5 ^ (-Real.log (0.3) / Real.log 3)

theorem compare_xyz : y < z ∧ z < x := by
  have h1 : Real.log 3.4 / Real.log 2 > (-Real.log 0.3 / Real.log 3), by sorry
  have h2 : (-Real.log 0.3 / Real.log 3) > ((Real.log 3.6 / Real.log 2) / 2), by sorry
  have h3 : (5 : ℝ) > 1, by norm_num
  exact ⟨by apply Real.rpow_lt_rpow h3 (ne_of_gt h1) sorry, by apply Real.rpow_lt_rpow h3 (ne_of_gt h2) sorry⟩

end compare_xyz_l530_530691


namespace letter_at_position_150_l530_530476

theorem letter_at_position_150 : 
  (∀ n, n > 0 → ∃ i, i ∈ {1, 2, 3} ∧ "XYZ".to_list[i-1] = "XYZ".to_list[(n - 1) % 3]) →
  ("XYZ".to_list[(150 - 1) % 3] = 'Z') :=
by
  sorry

end letter_at_position_150_l530_530476


namespace painter_earnings_l530_530134

noncomputable theory
open_locale big_operators

variables (n_s n_n : ℕ → ℕ) (digit_cost : ℕ → ℕ) (south_total north_total : ℕ)

def south_addresses : ℕ → ℕ :=
  λ n, 5 + (n - 1) * 7

def north_addresses : ℕ → ℕ :=
  λ n, 7 + (n - 1) * 8

def count_digits : ℕ → ℕ
| n := if n < 10 then 1 else if n < 100 then 2 else 3

def total_digit_cost (addr_fn : ℕ → ℕ) (n : ℕ) : ℕ :=
  (finset.range n).sum (λ i, count_digits (addr_fn (i + 1)))

theorem painter_earnings : 
  total_digit_cost south_addresses 25 + total_digit_cost north_addresses 25 = 125 :=
by sorry

end painter_earnings_l530_530134


namespace golden_section_AC_length_l530_530279

-- Definitions based on the conditions
def golden_section_point (A B C : ℝ) := C / (B - C) = (B - C) / (A - B)
def length_AB : ℝ := 8
def longer_segment_fraction := (Real.sqrt 5 - 1) / 2

-- Statement of the theorem to be proved
theorem golden_section_AC_length : 
  ∃ C : ℝ, 
    golden_section_point 8 0 C ∧ 
    C = 8 * ((Real.sqrt 5 - 1) / 2) := by
  sorry

end golden_section_AC_length_l530_530279


namespace circle_sum_positive_l530_530538
open Nat

def sum_n_consecutive (a : ℕ → ℝ) (i n : ℕ) (N : ℕ) : ℝ :=
  ∑ j in i..i+n-1, a (j % N)

theorem circle_sum_positive 
  (n : ℕ) (a : ℕ → ℝ) (h_length : ∀ i, a (i + 2 * n) = a i)
  (h_sum : (∑ k in 0..2 * n - 1, a k) > 0) :
  ∃ i < (2 * n), sum_n_consecutive a i n (2 * n) > 0 ∧ sum_n_consecutive a ((i + n) % (2 * n)) n (2 * n) > 0 :=
sorry

end circle_sum_positive_l530_530538


namespace number_of_diagonals_in_convex_polygon_l530_530751

/-- 
A theorem stating that the number of diagonals 
in a convex polygon with 30 sides is equal to 405.
-/
theorem number_of_diagonals_in_convex_polygon (n : ℕ) (h : n = 30) (convex : True) : (n * (n - 3)) / 2 = 405 := 
by 
  rw h
  norm_num
  done

end number_of_diagonals_in_convex_polygon_l530_530751


namespace convex_polygon_diagonals_l530_530790

theorem convex_polygon_diagonals (n : ℕ) (h : n = 30) : 
  let total_pairs := (n * (n - 1)) / 2 in (total_pairs - n) = 405 := 
by
  simp only [total_pairs]
  sorry

end convex_polygon_diagonals_l530_530790


namespace symmetric_point_of_P_l530_530302

-- Define a point in the Cartesian coordinate system
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define central symmetry with respect to the origin
def symmetric (p : Point) : Point :=
  { x := -p.x, y := -p.y }

-- Given point P with coordinates (1, -2)
def P : Point := { x := 1, y := -2 }

-- The theorem to be proved: the symmetric point of P is (-1, 2)
theorem symmetric_point_of_P :
  symmetric P = { x := -1, y := 2 } :=
by
  -- Proof is omitted.
  sorry

end symmetric_point_of_P_l530_530302


namespace diagonals_in_convex_polygon_with_30_sides_l530_530772

theorem diagonals_in_convex_polygon_with_30_sides : 
  let n := 30 in
  ∑ i j in finset.range n, (i + 1 - 1) * (j + 1 - 1) * (nat.choose 30 2 - 30) / 2 = 202 := 
by
  sorry

end diagonals_in_convex_polygon_with_30_sides_l530_530772


namespace calculate_solution_volume_l530_530042

theorem calculate_solution_volume (V : ℝ) (h : 0.35 * V = 1.4) : V = 4 :=
sorry

end calculate_solution_volume_l530_530042


namespace smallest_common_multiple_l530_530510

theorem smallest_common_multiple (n : ℕ) (h1 : n > 0) (h2 : 8 ∣ n) (h3 : 6 ∣ n) : n = 24 :=
by sorry

end smallest_common_multiple_l530_530510


namespace ellipse_foci_coordinates_l530_530166

/-- Define the parameters for the ellipse. -/
def ellipse_eq (x y : ℝ) : Prop := (x^2) / 25 + (y^2) / 169 = 1

/-- Prove the coordinates of the foci of the given ellipse. -/
theorem ellipse_foci_coordinates :
  (∀ (x y : ℝ), ellipse_eq x y → False) →
  ∃ (c : ℝ), c = 12 ∧ 
  ((0, c) = (0, 12) ∧ (0, -c) = (0, -12)) := 
by
  sorry

end ellipse_foci_coordinates_l530_530166


namespace each_team_plays_each_other_n_times_l530_530964

variable (teams games : ℕ)
variable (n : ℕ)

-- Conditions
axiom H1 : teams = 12
axiom H2 : games = 66

-- Problem statement
theorem each_team_plays_each_other_n_times :
  games = (teams * (teams - 1) * n) / 2 → 
  n = 2 := 
  by
  intro h
  rw [H1, H2] at h
  sorry

end each_team_plays_each_other_n_times_l530_530964


namespace max_profit_l530_530294

def fixed_cost := 14000
def variable_cost_per_unit := 210

def sales_volume (x : ℝ) : ℝ :=
 if x ≤ 400 then (1 / 625) * x^2 else 256

def selling_price_per_unit (x : ℝ) : ℝ :=
 if x ≤ 400 then (-5 / 8) * x + 750 else 500

def total_cost (x : ℝ) : ℝ :=
 fixed_cost + variable_cost_per_unit * x

def profit (x : ℝ) : ℝ :=
 if x ≤ 400 then
   sales_volume x * selling_price_per_unit x - total_cost x
 else
   sales_volume x * selling_price_per_unit x - total_cost x

theorem max_profit : ∃ x : ℝ, x = 400 ∧ profit x = 30000 := 
 by sorry

end max_profit_l530_530294


namespace university_committee_count_l530_530152

theorem university_committee_count :
  let departments := ["math", "stats", "cs"],
      males_per_dept := 3,
      females_per_dept := 3,
      total_males := 3 * 3,
      total_females := 3 * 3,
      total_committee_men := 4,
      total_committee_women := 2,
      total_committee_size := total_committee_men + total_committee_women,
      choose := Nat.choose in
  (departments.choose 2).length = 3
  ∧ (choose 3 2 = 3)
  ∧ (choose 3 1 = 3)
  ∧ (choose 3 3 = 1)
  ∧ ((3 * 3 * 3 * 3 * 2) + (3 * 9 * 9) = 891) :=
by
  sorry

end university_committee_count_l530_530152


namespace convex_polygon_diagonals_l530_530788

theorem convex_polygon_diagonals (n : ℕ) (h : n = 30) : 
  let total_pairs := (n * (n - 1)) / 2 in (total_pairs - n) = 405 := 
by
  simp only [total_pairs]
  sorry

end convex_polygon_diagonals_l530_530788


namespace revenue_loss_l530_530667

-- Defining constants that express the conditions
def daily_loss : ℕ := 5000
def closed_days_per_year : ℕ := 3
def years_of_activity : ℕ := 6

-- Calculate yearly loss
def yearly_loss : ℕ := daily_loss * closed_days_per_year

-- Calculate the total loss over the given timeframe
def total_loss : ℕ := yearly_loss * years_of_activity

-- Lean theorem stating the revenue lost over the given period
theorem revenue_loss (daily_loss = 5000) (closed_days_per_year = 3) (years_of_activity = 6) : 
  total_loss = 90000 :=
by simp [daily_loss, closed_days_per_year, years_of_activity, yearly_loss, total_loss]; sorry

end revenue_loss_l530_530667


namespace baron_not_boasting_l530_530605

noncomputable def is_isosceles_triangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist A C ∨ dist B C = dist A B ∨ dist C A = dist B C

theorem baron_not_boasting (A B C : ℝ × ℝ) (h_isosceles : is_isosceles_triangle A B C) :
∃ D E F G H : ℝ × ℝ, 
  ∀ (P Q : ℝ × ℝ), 
    (P = D ∨ P = E ∨ P = F ∨ P = G ∨ P = H) ∧ 
    (Q = D ∨ Q = E ∨ Q = F ∨ Q = G ∨ Q = H) ∧ 
    is_isosceles_triangle P Q :=
sorry

end baron_not_boasting_l530_530605


namespace smallest_prime_factor_of_1879_l530_530988

theorem smallest_prime_factor_of_1879 :
  ¬ (1879 % 2 = 0) →
  ¬ (1879 % 3 = 0) →
  ¬ (1879 % 5 = 0) →
  ¬ (1879 % 7 = 0) →
  ¬ (1879 % 11 = 0) →
  ¬ (1879 % 13 = 0) →
  1879 % 17 = 0 →
  ∀ p : ℕ, (prime p ∧ p ∣ 1879) → 17 ≤ p := by
  sorry

end smallest_prime_factor_of_1879_l530_530988


namespace alan_total_cost_l530_530582

theorem alan_total_cost :
  let price_AVN_CD := 12 in
  let price_The_Dark_CD := price_AVN_CD * 2 in
  let total_cost_The_Dark_CDs := 2 * price_The_Dark_CD in
  let total_cost_before_90s_CDs := price_AVN_CD + total_cost_The_Dark_CDs in
  let cost_90s_CDs := 0.4 * total_cost_before_90s_CDs in
  let total_cost := total_cost_before_90s_CDs + cost_90s_CDs in
  total_cost = 84 :=
by
  let price_AVN_CD := 12
  let price_The_Dark_CD := price_AVN_CD * 2
  let total_cost_The_Dark_CDs := 2 * price_The_Dark_CD
  let total_cost_before_90s_CDs := price_AVN_CD + total_cost_The_Dark_CDs
  let cost_90s_CDs := 0.4 * total_cost_before_90s_CDs
  let total_cost := total_cost_before_90s_CDs + cost_90s_CDs
  show total_cost = 84, from sorry

end alan_total_cost_l530_530582


namespace cost_of_8_cubic_yards_topsoil_l530_530055

def cubic_yards_to_cubic_feet (yd³ : ℕ) : ℕ := 27 * yd³

def cost_of_topsoil (cubic_feet : ℕ) (cost_per_cubic_foot : ℕ) : ℕ := cubic_feet * cost_per_cubic_foot

theorem cost_of_8_cubic_yards_topsoil :
  cost_of_topsoil (cubic_yards_to_cubic_feet 8) 8 = 1728 :=
by
  sorry

end cost_of_8_cubic_yards_topsoil_l530_530055


namespace solve_for_rs_l530_530890

noncomputable def circle1 : Type :=
{ center := (-4, 10),
  radius := 11 }

noncomputable def circle2 : Type :=
{ center := (4, 10),
  radius := 7 }

theorem solve_for_rs :
  (let b := sqrt (5/16) in 
   let n := sqrt (5/16) in
   ∃ r s : Nat, r + s = 21 ∧ n^2 = r / s) :=
sorry

end solve_for_rs_l530_530890


namespace find_150th_letter_l530_530460

theorem find_150th_letter :
  let pattern := ['X', 'Y', 'Z']
  150 % 3 = 0 -> pattern[(150 % 3 + 2) % 3] = 'Z' :=
begin
  intros pattern h,
  simp at *,
  exact rfl,
end

end find_150th_letter_l530_530460


namespace geometric_series_sum_l530_530078

theorem geometric_series_sum : 
  let a := 2
  let r := 2
  let n := 13
  have end_term : (a * r^(n - 1)) = 8192, by
  -- Given conditions
  (2 * 2^(13 - 1)) = 8192 := rfl
  multiply lhs by 2
  (2 * 2^(12 + 1)) = 8192 := rfl
  8192 = 8192 := rfl
  
  calc
     Σ_{k=0}^{n-1} a * r^k 
     = Σ_{k=0}^{12} 2 * 2^k : rfl
     = 2 * Σ_{k=0}^{12} 2^k 
     = 2 * (2^(12+1) - 1) / (2 - 1) : geom_sum_closed_form _
     = 2 * (2^13 - 1)
     = 2 * (8192 - 1)
     = 2 * 8191,
   sum of the series is 16382

end geometric_series_sum_l530_530078


namespace largest_consecutive_useful_numbers_l530_530344

def isUseful (n : ℕ) : Prop :=
  let digits := (nat.digits 10 n).erase_dup
  n > 0 ∧
  (¬ digits.contains 0) ∧
  (digits.length = (nat.digits 10 n).length) ∧
  ((nat.digits 10 n).prod % (nat.digits 10 n).sum = 0)

theorem largest_consecutive_useful_numbers :
  isUseful 9875213 ∧ isUseful 9875214 ∧
  (∀ m n : ℕ, m > 9875214 → n = m + 1 → ¬(isUseful m ∧ isUseful n)) :=
by sorry

end largest_consecutive_useful_numbers_l530_530344


namespace area_of_trapezoid_l530_530991

theorem area_of_trapezoid {E F G H : ℝ × ℝ} 
  (hE : E = (2, -3)) (hF : F = (2, 2)) (hG : G = (7, 8)) (hH : H = (7, 0)) :
  let b1 := abs(E.2 - F.2),
      b2 := abs(G.2 - H.2),
      h := abs(G.1 - E.1),
      area := (1 / 2 : ℝ) * (b1 + b2) * h in
  area = 32.5 :=
by sorry

end area_of_trapezoid_l530_530991


namespace ratio_spherical_segment_to_sphere_volume_l530_530204

theorem ratio_spherical_segment_to_sphere_volume
  (R : ℝ) (α : ℝ) :
  let H := 2 * R * (Real.sin (α / 4))^2 in
  let V_seg := Real.pi * H^2 * (R - H / 3) in
  let V_sphere := (4 * Real.pi * R^3) / 3 in
  V_seg / V_sphere = (Real.sin (α / 4))^4 * (2 + Real.cos (α / 2)) :=
by
  sorry

end ratio_spherical_segment_to_sphere_volume_l530_530204


namespace sum_of_first_1500_terms_l530_530661

-- Definition of the sequence described in the problem
def sequence : ℕ → ℕ
| 0 => 1
| n => if h : ∃ k, n = (k+1) * (k + 2) / 2 - 1 then 1 else 2

-- Sum of the first n terms of the sequence
def sum_sequence_up_to (n : ℕ) : ℕ :=
  (Finset.range n).sum sequence

-- The theorem stating the sum of the first 1500 terms
theorem sum_of_first_1500_terms : sum_sequence_up_to 1500 = 2946 := by sorry

end sum_of_first_1500_terms_l530_530661


namespace improper_fractions_count_l530_530997

open Finset

def is_improper_fraction (num den : ℕ) : Prop :=
  num >= den

def count_improper_fractions (s : Finset ℕ) : ℕ :=
  (s.product s).count (λ p => p.1 ≠ p.2 ∧ is_improper_fraction p.1 p.2)

theorem improper_fractions_count :
  count_improper_fractions {3, 5, 7, 11, 13, 17} = 15 :=
by
  sorry

end improper_fractions_count_l530_530997


namespace solve_for_x_l530_530382

theorem solve_for_x :
  8 * 216^(1/3) - 4 * (216 / 216^(2/3)) = 12 + 2 * 216^(1/3) ∧
  216 ≥ sqrt 144 :=
by
  sorry

end solve_for_x_l530_530382


namespace islands_not_connected_by_bridges_for_infinitely_many_primes_l530_530211

open Nat

theorem islands_not_connected_by_bridges_for_infinitely_many_primes :
  ∃ᶠ p in at_top, ∃ n m : ℕ, n ≠ m ∧ ¬(p ∣ (n^2 - m + 1) * (m^2 - n + 1)) :=
sorry

end islands_not_connected_by_bridges_for_infinitely_many_primes_l530_530211


namespace origin_moves_correctly_l530_530119

noncomputable def dilation_distance_moved (B B' : ℝ × ℝ) (r r' : ℝ) (O : ℝ × ℝ) : ℝ :=
  let (x, y) := (-6: ℝ, -9: ℝ) -- Center of dilation (already solved)
  let initial_distance := real.sqrt ((-6: ℝ) ^ 2 + (-9: ℝ) ^ 2)
  let dilation_factor := r' / r
  let final_distance := dilation_factor * initial_distance
  final_distance - initial_distance

theorem origin_moves_correctly :
  dilation_distance_moved (3, 3) (7, 9) 3 4.5 (0, 0) = 1.5 * real.sqrt 13 :=
by
  -- Placeholder to eventually contain the correct proof steps.
  sorry

end origin_moves_correctly_l530_530119


namespace train_speed_l530_530572

theorem train_speed (l_train l_bridge : ℕ) (t : ℕ) (h1 : l_train = 135) (h2 : l_bridge = 240) (h3 : t = 30) : 
  (l_train + l_bridge) / t * 3.6 = 45 := 
by {
  sorry
}

end train_speed_l530_530572


namespace polyhedron_vertex_coloring_l530_530067

theorem polyhedron_vertex_coloring (V : Type) [Fintype V] (E : V → V → Prop) [Symmetric E] [Irreflexive E] (red_blue_coloring : V → Prop) :
  (∃ v : V, red_blue_coloring v ∧ (∃! face : {f : set V // ∃ (l : Fintype (Subtype {v // red_blue_coloring v})), finite l}, ∀ u ∈ face.1, red_blue_coloring u)) →
  (∃ v : V, ¬red_blue_coloring v → ∃ le5_edges : Fintype (Subtype {v' // E v v'}), fincard le5_edges ≤ 5) ∨
  (∃ v : V, red_blue_coloring v → ∃ eq3_edges : Fintype (Subtype {v' // E v v'}), fincard eq3_edges = 3) :=
sorry

end polyhedron_vertex_coloring_l530_530067


namespace length_segment_OS_l530_530554

-- Definitions of given conditions
def Circle (center : Point) (radius : ℝ) := ∀ (p : Point), (dist center p = radius ↔ OnCircle p)
axiom O_center : Point
axiom P_center : Point
axiom Q_point : Point
axiom T_point : Point
axiom S_point : Point
axiom O_radius : dist O_center Q_point = 10 -- Radius of circle O is 10 units
axiom P_radius : dist P_center Q_point = 5 -- Radius of circle P is 5 units
axiom tangent_point : dist O_center P_center = 15 -- The circles O and P are tangent at Q (OP = OQ + PQ)
axiom common_tangent : tangent T_point S_point O_center P_center -- Segment TS is the common external tangent

-- The theorem statement to be proved
theorem length_segment_OS : dist O_center S_point = 10 * Real.sqrt 3 := 
sorry

end length_segment_OS_l530_530554


namespace find_principal_l530_530677

theorem find_principal
  (R : ℝ) (hR : R = 0.05)
  (I : ℝ) (hI : I = 0.02)
  (A : ℝ) (hA : A = 1120)
  (n : ℕ) (hn : n = 6)
  (R' : ℝ) (hR' : R' = ((1 + R) / (1 + I)) - 1) :
  P = 938.14 :=
by
  have compound_interest_formula := A / (1 + R')^n
  sorry

end find_principal_l530_530677


namespace javier_pumpkin_ravioli_count_l530_530315

/-- The meat ravioli weighs 1.5 ounces each, the pumpkin ravioli weighs 1.25 ounces each, 
and the cheese ravioli weighs 1 ounce each. Given Javier eats 5 meat ravioli, 
4 cheese ravioli and his brother eats 12 pumpkin ravioli.
The winner ate a total of 15 ounces. 
Prove that Javier ate 2 pumpkin ravioli. -/
theorem javier_pumpkin_ravioli_count :
  let meat_ravioli_weight := 1.5
  let pumpkin_ravioli_weight := 1.25
  let cheese_ravioli_weight := 1.0
  let meat_ravioli_count := 5
  let cheese_ravioli_count := 4
  let brother_pumpkin_ravioli_count := 12
  let total_weight_winner := 15
  ∃ (javier_pumpkin_ravioli_count : ℕ), 
  (javier_pumpkin_ravioli_count * pumpkin_ravioli_weight + 
   (meat_ravioli_count * meat_ravioli_weight) + 
   (cheese_ravioli_count * cheese_ravioli_weight) = total_weight_winner)
  ∧ javier_pumpkin_ravioli_count = 2 :=
by
  sorry

end javier_pumpkin_ravioli_count_l530_530315


namespace smallest_common_multiple_of_8_and_6_l530_530518

theorem smallest_common_multiple_of_8_and_6 : ∃ n : ℕ, n > 0 ∧ (8 ∣ n) ∧ (6 ∣ n) ∧ ∀ m : ℕ, (m > 0 ∧ (8 ∣ m) ∧ (6 ∣ m)) → n ≤ m :=
by
  sorry

end smallest_common_multiple_of_8_and_6_l530_530518


namespace exists_sequence_l530_530880

def t (n : ℕ) : ℕ := 
  (nat.digits 2 n).sum

theorem exists_sequence (k : ℕ) (h : k ≥ 2) : 
  ∃ (a : ℕ → ℕ), (∀ m, a m ≥ 3 ∧ Odd (a m) ∧ t (list.prod (list.of_fn (λ i, a (i + 1)) (m + 1))) = k) := 
sorry

end exists_sequence_l530_530880


namespace larger_number_is_2997_l530_530015

theorem larger_number_is_2997 (L S : ℕ) (h1 : L - S = 2500) (h2 : L = 6 * S + 15) : L = 2997 := 
by
  sorry

end larger_number_is_2997_l530_530015


namespace f_is_monotonically_decreasing_f_is_odd_function_iff_a_eq_neg_one_l530_530685

-- Define the function f
def f (a : ℝ) (x : ℝ) := a + 2 / (2 ^ x + 1)

-- 1. Prove that f(x) is a monotonically decreasing function on ℝ
theorem f_is_monotonically_decreasing (a : ℝ) : ∀ (x₁ x₂ : ℝ), x₁ < x₂ → f a x₁ > f a x₂ := sorry

-- 2. Prove that there exists a real number a such that f(x) is an odd function, and find the value of a.
theorem f_is_odd_function_iff_a_eq_neg_one : ∃! a : ℝ, ∀ x : ℝ, f a (-x) = -f a x ↔ a = -1 :=
begin
  use -1,
  intro x,
  split,
  { rintro rfl, -- Proof that f(x) is an odd function implies a = -1
    unfold f,
    simp, },
  { -- Proof that a = -1 implies f(x) is an odd function
    intro h,
    rw h,
    unfold f,
    simp, }
end

end f_is_monotonically_decreasing_f_is_odd_function_iff_a_eq_neg_one_l530_530685


namespace diagonal_count_of_convex_polygon_30_sides_l530_530799
-- Importing the entire Mathlib library for necessary mathematical constructs

theorem diagonal_count_of_convex_polygon_30_sides : 
  let n := 30 in
  let num_diagonals := (n * (n - 3)) / 2 in
  num_diagonals = 405 := 
by {
  let n := 30
  let num_diagonals := (n * (n - 3)) / 2
  show num_diagonals = 405,
  sorry
}

end diagonal_count_of_convex_polygon_30_sides_l530_530799


namespace number_of_diagonals_in_30_sided_polygon_l530_530759

-- Definition of a convex polygon with 30 sides
def convex_polygon_sides := 30

-- The function to calculate the number of diagonals in a convex polygon with n sides
noncomputable def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Prove that a convex polygon with 30 sides has 405 diagonals
theorem number_of_diagonals_in_30_sided_polygon :
  num_diagonals convex_polygon_sides = 405 :=
by
  calc
    num_diagonals 30 = 30 * (30 - 3) / 2 : rfl
    ... = 30 * 27 / 2 : rfl
    ... = 810 / 2 : rfl
    ... = 405 : rfl

end number_of_diagonals_in_30_sided_polygon_l530_530759


namespace sine_180_eq_zero_l530_530627

theorem sine_180_eq_zero :
  sin (180 : ℝ) = 0 :=
sorry

end sine_180_eq_zero_l530_530627


namespace number_of_functions_l530_530203

def satisfies_condition (f : ℤ → ℤ) (k : ℤ) :=
  ∀ a b : ℤ, k * f(a + b) + f(a * b) = f(a) * f(b) + k

theorem number_of_functions (k : ℤ) (hk : k ≠ 0) :
  ∃ (f : ℤ → ℤ), satisfies_condition f k :=
sorry

end number_of_functions_l530_530203


namespace polynomial_divisibility_l530_530213

theorem polynomial_divisibility (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (Polynomial.geom_series a ℕ).is_dvd (Polynomial.geom_series (a * b) ℕ) ↔ Nat.gcd (a + 1) b = 1 :=
sorry

end polynomial_divisibility_l530_530213


namespace subset_intersect_nonempty_l530_530900

variable {α : Type*}
variable (S : Finset α)
variable (A : Finset (Finset α))
variable [Fintype α]
variable [DecidableEq α]

def conditions (n k : ℕ) : Prop :=
  let S' := Finset.card S = n in
  let Ak := ∀ A1 A2 ∈ A, A1 ≠ A2 → (A1 ∩ A2).nonempty in
  let NoOther := ∀ B ∉ A, ¬∀ A' ∈ A, (A' ∩ B).nonempty in
  S' ∧ Ak ∧ NoOther

theorem subset_intersect_nonempty (n k : ℕ) (S : Finset α) (A : Finset (Finset α)) 
  [Fintype α] [DecidableEq α] (hk : conditions S A n k) :
  k = 2^(n-1) := 
sorry -- the proof goes here

end subset_intersect_nonempty_l530_530900


namespace centroids_quadrilateral_intersection_l530_530004

theorem centroids_quadrilateral_intersection (
  A B C D : Point
  (h_convex : convex_quadrilateral A B C D)
  (G1 := centroid_triangle A B C)
  (G2 := centroid_triangle B C D)
  (G3 := centroid_triangle C D A)
  (G4 := centroid_triangle D A B)
) :
  let K := G1,
      L := G2,
      M := G3,
      N := G4,
      AB_CD_mid := midpoint (midpoint A B) (midpoint C D),
      BC_DA_mid := midpoint (midpoint B C) (midpoint D A),
      KL_MN_mid := midpoint (midpoint K L) (midpoint M N),
      LM_NK_mid := midpoint (midpoint L M) (midpoint N K)
  in AB_CD_mid = BC_DA_mid ∧ KL_MN_mid = LM_NK_mid :=
by {
  sorry
}

end centroids_quadrilateral_intersection_l530_530004


namespace inscribed_circle_radius_is_correct_l530_530553

noncomputable def radius_of_inscribed_circle (base height : ℝ) : ℝ := sorry

theorem inscribed_circle_radius_is_correct :
  radius_of_inscribed_circle 20 24 = 120 / 13 := sorry

end inscribed_circle_radius_is_correct_l530_530553


namespace smallest_common_multiple_of_8_and_6_l530_530517

theorem smallest_common_multiple_of_8_and_6 : ∃ n : ℕ, n > 0 ∧ (8 ∣ n) ∧ (6 ∣ n) ∧ ∀ m : ℕ, (m > 0 ∧ (8 ∣ m) ∧ (6 ∣ m)) → n ≤ m :=
by
  sorry

end smallest_common_multiple_of_8_and_6_l530_530517


namespace no_values_of_t_l530_530664

theorem no_values_of_t (t : ℂ) : sqrt(49 - t^2) + 7 = 0 → false := 
begin
  sorry
end

end no_values_of_t_l530_530664


namespace letter_150_is_Z_l530_530500

/-- Definition of the repeating pattern "XYZ" -/
def pattern : List Char := ['X', 'Y', 'Z']

/-- The repeating pattern has a length of 3 -/
def pattern_length : ℕ := 3

/-- Calculate the 150th letter in the repeating pattern "XYZ" -/
def nth_letter_in_pattern (n : ℕ) : Char :=
  let m := n % pattern_length
  if m = 0 then pattern[2] else pattern[m - 1]

/-- Prove that the 150th letter in the pattern "XYZ" is 'Z' -/
theorem letter_150_is_Z : nth_letter_in_pattern 150 = 'Z' :=
by
  sorry

end letter_150_is_Z_l530_530500


namespace diagonals_in_convex_polygon_with_30_sides_l530_530771

theorem diagonals_in_convex_polygon_with_30_sides : 
  let n := 30 in
  ∑ i j in finset.range n, (i + 1 - 1) * (j + 1 - 1) * (nat.choose 30 2 - 30) / 2 = 202 := 
by
  sorry

end diagonals_in_convex_polygon_with_30_sides_l530_530771


namespace diagonal_count_of_convex_polygon_30_sides_l530_530796
-- Importing the entire Mathlib library for necessary mathematical constructs

theorem diagonal_count_of_convex_polygon_30_sides : 
  let n := 30 in
  let num_diagonals := (n * (n - 3)) / 2 in
  num_diagonals = 405 := 
by {
  let n := 30
  let num_diagonals := (n * (n - 3)) / 2
  show num_diagonals = 405,
  sorry
}

end diagonal_count_of_convex_polygon_30_sides_l530_530796


namespace base_three_to_base_ten_l530_530073

theorem base_three_to_base_ten (n : ℕ) (h : n = 20121) : 
  let digits := [1, 2, 0, 1, 2] in
  let base := 3 in
  ∑ i in finset.range digits.length, (digits[i] * base^i) = 178 :=
by sorry

end base_three_to_base_ten_l530_530073


namespace peter_fraction_equiv_l530_530932

def fraction_pizza_peter_ate (total_slices : ℕ) (slices_ate_alone : ℕ) (shared_slices_brother : ℚ) (shared_slices_sister : ℚ) : ℚ :=
  (slices_ate_alone / total_slices) + (shared_slices_brother / total_slices) + (shared_slices_sister / total_slices)

theorem peter_fraction_equiv :
  fraction_pizza_peter_ate 16 3 (1/2) (1/2) = 1/4 :=
by
  sorry

end peter_fraction_equiv_l530_530932


namespace total_weight_of_remaining_chocolate_eggs_l530_530354

-- Define the conditions as required:
def marie_makes_12_chocolate_eggs : ℕ := 12
def weight_of_each_egg : ℕ := 10
def number_of_boxes : ℕ := 4
def boxes_discarded : ℕ := 1

-- Define the proof statement
theorem total_weight_of_remaining_chocolate_eggs :
  let eggs_per_box := marie_makes_12_chocolate_eggs / number_of_boxes,
      remaining_boxes := number_of_boxes - boxes_discarded,
      remaining_eggs := remaining_boxes * eggs_per_box,
      total_weight := remaining_eggs * weight_of_each_egg
  in total_weight = 90 := by
  sorry

end total_weight_of_remaining_chocolate_eggs_l530_530354


namespace conjugate_complex_square_l530_530216

noncomputable def imaginaryUnit : ℂ := complex.I

def are_conjugate_complex (z1 z2 : ℂ) : Prop :=
  z2 = complex.conj z1

theorem conjugate_complex_square (a b : ℝ)
  (h1 : are_conjugate_complex (a - imaginaryUnit) (2 + b * imaginaryUnit)) :
  (a + b * imaginaryUnit) ^ 2 = 3 + 4 * imaginaryUnit := by
  sorry

end conjugate_complex_square_l530_530216


namespace race_distance_l530_530295

-- Define the conditions as separate hypotheses
variables (D : ℝ) -- Distance of the race
variables (A_speed : ℝ) (B_speed : ℝ)
variables (Beats_by : ℝ)

-- Hypotheses from the problem conditions
def racing_conditions : Prop :=
  (A_speed = D / 36) ∧
  (B_speed = D / 45) ∧
  (D - (36 * B_speed) = 22)

-- Statement to prove that D is 110 under the given conditions
theorem race_distance (h : racing_conditions D A_speed B_speed Beats_by) : D = 110 :=
by
  sorry

end race_distance_l530_530295


namespace maximum_value_permutation_sum_l530_530075

theorem maximum_value_permutation_sum :
  ∀ (x : Fin 63 → ℕ), (∀ i, 1 ≤ x i ∧ x i ≤ 63) →
  (Injective x) →
  (∑ i : Fin 63, |(x i - (i + 1))|) ≤ 1984 :=
by 
  intros x hx hinj;
  sorry

end maximum_value_permutation_sum_l530_530075


namespace distance_to_gym_l530_530599

theorem distance_to_gym (v d : ℝ) (h_walked_200_m: 200 / v > 0) (h_double_speed: 2 * v = 2) (h_time_diff: 200 / v - d / (2 * v) = 50) : d = 300 :=
by sorry

end distance_to_gym_l530_530599


namespace Jon_needs_to_wash_20_pairs_of_pants_l530_530318

theorem Jon_needs_to_wash_20_pairs_of_pants
  (machine_capacity : ℕ)
  (shirts_per_pound : ℕ)
  (pants_per_pound : ℕ)
  (num_shirts : ℕ)
  (num_loads : ℕ)
  (total_pounds : ℕ)
  (weight_of_shirts : ℕ)
  (remaining_weight : ℕ)
  (num_pairs_of_pants : ℕ) :
  machine_capacity = 5 →
  shirts_per_pound = 4 →
  pants_per_pound = 2 →
  num_shirts = 20 →
  num_loads = 3 →
  total_pounds = num_loads * machine_capacity →
  weight_of_shirts = num_shirts / shirts_per_pound →
  remaining_weight = total_pounds - weight_of_shirts →
  num_pairs_of_pants = remaining_weight * pants_per_pound →
  num_pairs_of_pants = 20 :=
by
  intros _ _ _ _ _ _ _ _ _
  sorry

end Jon_needs_to_wash_20_pairs_of_pants_l530_530318


namespace alan_total_cost_is_84_l530_530576

noncomputable def price_AVN : ℝ := 12
noncomputable def multiplier : ℝ := 2
noncomputable def count_Dark : ℕ := 2
noncomputable def count_AVN : ℕ := 1
noncomputable def count_90s : ℕ := 5
noncomputable def percentage_90s : ℝ := 0.40

def main_theorem : Prop :=
  let price_Dark := price_AVN * multiplier in
  let total_cost_Dark := price_Dark * count_Dark in
  let total_cost_AVN := price_AVN * count_AVN in
  let total_cost_other := total_cost_Dark + total_cost_AVN in
  let cost_90s := percentage_90s * total_cost_other in
  let total_cost := total_cost_other + cost_90s in
  total_cost = 84

theorem alan_total_cost_is_84 : main_theorem :=
  sorry

end alan_total_cost_is_84_l530_530576


namespace incenter_intersection_l530_530364

noncomputable def incenter (A B C : Point) : Point := sorry

theorem incenter_intersection
  {A B C D E P : Point}
  (hD : D ∈ Line A B)
  (hE : E ∈ Line A C)
  (hDB : dist D B = dist B C)
  (hEC : dist C E = dist B C)
  (hP : ∃ (e f : Line), e.contains B ∧ e.contains E ∧ f.contains C ∧ f.contains D ∧ e.intersects f = {P})
  (I : Point) (hI : I = incenter A B C) :
  (Circle (B P D)).contains I ∧ (Circle (C P E)).contains I := sorry

end incenter_intersection_l530_530364


namespace length_MN_in_terms_of_AB_and_CD_l530_530023

variables {A B C D P Q M N : Type}
           [field A] [field B] [field C] [field D] [field P] [field Q] [field M] [field N]
           [AB CD : Type] [parallel: (AB) -> (CD) -> Prop]


def points_on_line (AB : Type) [field AB] (P : AB) (PB: AB): Type := sorry
def lines_intersection (AQ: Type) (DP: Type): (AQ) -> (DP) -> Type := sorry
def lines_intersection2 (PC: Type) (QB: Type): (PC) -> (QB) -> Type := sorry

theorem length_MN_in_terms_of_AB_and_CD
  (AB CD : Type) [field AB] [field CD] (P Q : Type) 
  (parallel : (AB) -> (CD) -> Prop) (h1 : parallel (lines_intersection A B) (lines_intersection C D))
  (h2 : ∃ k : ℚ, ∀ AP PB DQ CQ : ℚ, AP / PB = k ∧ DQ / CQ = k)
  (AP PB DQ CQ : ℚ) (h3 : P = points_on_line AB P PB)
  (M : (lines_intersection A Q) -> (lines_intersection D P) -> Type)
  (N : (lines_intersection2 P C) -> (lines_intersection2 Q B) -> Type):
  MN (MN_length : ℚ) := 
  begin
    sorry
  end :=
begin
  sorry
end

end length_MN_in_terms_of_AB_and_CD_l530_530023


namespace pages_of_dictionary_l530_530971

theorem pages_of_dictionary (total_ones : ℕ) (h : total_ones = 1988) : ∃ n : ℕ, n = 3152 ∧ -- total page count in dictionary
  (let count_ones_in_pos : ℕ → ℕ := λ x, -- function that counts how many times '1' appears in the number
    nat.digits 10 x |>.count 1 in
  list.sum (list.map count_ones_in_pos (list.range (n + 1))) = total_ones) :=
sorry -- Proof to be filled in

end pages_of_dictionary_l530_530971


namespace smallest_n_l530_530679

theorem smallest_n (n : ℕ) (h₁ : n > 2016) (h₂ : n % 4 = 0) : 
  ¬(1^n + 2^n + 3^n + 4^n) % 10 = 0 → n = 2020 :=
by
  sorry

end smallest_n_l530_530679


namespace inverse_proportion_point_l530_530844

theorem inverse_proportion_point (k : ℝ) (x1 y1 x2 y2 : ℝ)
  (h1 : y1 = k / x1) 
  (h2 : x1 = -2) 
  (h3 : y1 = 3)
  (h4 : x2 = 2) :
  y2 = -3 := 
by
  -- proof will be provided here
  sorry

end inverse_proportion_point_l530_530844


namespace find_incorrect_value_l530_530399

theorem find_incorrect_value (n : ℕ) (mean_initial mean_correct : ℕ) (wrongly_copied correct_value incorrect_value : ℕ) 
  (h1 : n = 30) 
  (h2 : mean_initial = 150) 
  (h3 : mean_correct = 151) 
  (h4 : correct_value = 165) 
  (h5 : n * mean_initial = 4500) 
  (h6 : n * mean_correct = 4530) 
  (h7 : n * mean_correct - n * mean_initial = 30) 
  (h8 : correct_value - (n * mean_correct - n * mean_initial) = incorrect_value) : 
  incorrect_value = 135 :=
by
  sorry

end find_incorrect_value_l530_530399


namespace convex_polygon_diagonals_l530_530811

theorem convex_polygon_diagonals (n : ℕ) (h : n = 30) : 
  let V := n in
  let total_pairs := V * (V - 1) / 2 in
  let adjacent_pairs := n in
  let diagonals := total_pairs - adjacent_pairs in
  diagonals = 405 :=
  by
  -- number of sides is 30
  have hn : n = 30 := h
  -- calculate the total pairs
  have total_pairs_calc : total_pairs = 30 * 29 / 2 := by sorry
  -- calculate the adjacent pairs
  have adjacent_pairs_calc : adjacent_pairs = 30 := by sorry
  -- calculate the diagonals
  have diagonals_calc : diagonals = (30 * 29 / 2) - 30 := by sorry
  -- proved statement
  show 405 = 405, by rfl

end convex_polygon_diagonals_l530_530811


namespace problem_statement_l530_530864

open Real

-- Step 1: Parametric equations of curve C
def parametric_eqs_curve_C (α : ℝ) : ℝ × ℝ := 
  (2 + cos α, 3 + sin α)

-- Step 2: Polar coordinates of point A
def polar_coords_A : ℝ × ℝ := (3, π / 2)

-- Step 3: Polar equation of curve C
def polar_eq_curve_C (ρ θ : ℝ) : Prop := 
  ρ^2 - 4 * ρ * cos θ - 6 * ρ * sin θ + 12 = 0

-- Step 4: Ratio |ON| / |AM|
def ratio_ON_AM (C M N O A : ℝ × ℝ) : ℝ :=
  let ON := distance O N
  let AM := distance A M
  ON / AM

-- Define the problem statement as the theorem
theorem problem_statement :
  ∃ (C : ℝ × ℝ) (O : ℝ × ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ),
  ∀ (α ρ θ : ℝ),
    (parametric_eqs_curve_C α = C) →
    (polar_coords_A = (3, π / 2)) →
    (polar_eq_curve_C ρ θ) →
    (ratio_ON_AM C M N O (0, 3) = 2) :=
by
  -- Given that the equations for curve C, point A, polar coordinates, etc., are definitions,
  -- and the goal is to work towards proving the ratio is 2 under these setups.
  sorry

end problem_statement_l530_530864


namespace p_q_sum_l530_530402

noncomputable def sum_p_q : ℕ :=
  let totalArrangements := fact 10
  let acceptableArrangements := 302400
  let probability := (acceptableArrangements : ℚ) / (totalArrangements : ℚ)
  let reduced_fraction := (probability.num, probability.denom)  -- finding the numerators and denominators in reduced form
  reduced_fraction.1 + reduced_fraction.2

theorem p_q_sum : sum_p_q = 13 :=
  by sorry

end p_q_sum_l530_530402


namespace find_150th_letter_in_pattern_l530_530444

/--
Given the repeating pattern "XYZ" with a cycle length of 3,
prove that the 150th letter in this pattern is 'Z'.
-/
theorem find_150th_letter_in_pattern : 
  let pattern := "XYZ"
  let cycle_length := String.length pattern
in (150 % cycle_length = 0) → "Z" := 
sorry

end find_150th_letter_in_pattern_l530_530444


namespace problem_solution_l530_530993

noncomputable def f (x : ℝ) := x / (|x| + 1)

def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

def range_of_f (f : ℝ → ℝ) (s : set ℝ) := ∀ y, y ∈ s ↔ ∃ x, f x = y

def increasing_function (f : ℝ → ℝ) := ∀ x y, x < y → f x ≤ f y

def exists_no_real_k (f g : ℝ → ℝ) := ∀ k, ¬ ∀ x, g x ≠ 0

theorem problem_solution :
  odd_function f ∧
  range_of_f f (set.Ioo (-1 : ℝ) 1) ∧
  increasing_function f ∧
  exists_no_real_k f (λ x, f x - k * x - k) :=
by
  sorry

end problem_solution_l530_530993


namespace diagonals_in_30_sided_polygon_l530_530825

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem diagonals_in_30_sided_polygon :
  number_of_diagonals 30 = 405 :=
by sorry

end diagonals_in_30_sided_polygon_l530_530825


namespace present_value_correct_l530_530961

noncomputable def present_value (selling_price : ℝ) (profit : ℝ) (depreciation_rate : ℝ) (years : ℕ) : ℝ :=
  (selling_price - profit) / (depreciation_rate ^ years)

theorem present_value_correct :
  present_value 115260 24000 0.78 2 ≈ 150000 := by
  sorry

end present_value_correct_l530_530961


namespace sin_180_is_zero_l530_530642

noncomputable def sin_180_eq_zero : Prop :=
  let point_zero_deg := (1, 0)
  let point_180_deg := (-1, 0)
  let y_coord_of_180_deg := 0
  sin 180 = y_coord_of_180_deg

theorem sin_180_is_zero : sin_180_eq_zero :=
  sorry

end sin_180_is_zero_l530_530642


namespace length_YD_l530_530432

def TriangleXYZ : Type :=
  { XY YZ XZ : ℕ // XY = 6 ∧ YZ = 8 ∧ XZ = 10 }

def bugs_meet_at_D (t : TriangleXYZ) : ℕ :=
  let XY := t.1.1
  let YZ := t.1.2
  let XZ := t.1.3
  let perimeter := XY + YZ + XZ
  let half_perimeter := perimeter / 2
  half_perimeter - XY

theorem length_YD (t : TriangleXYZ) : bugs_meet_at_D t = 6 :=
  sorry

end length_YD_l530_530432


namespace problem_statement_l530_530301

-- Define the problem conditions
def m : ℝ := 2 * Real.sin (Real.pi * 18 / 180)
def n : ℝ := 4 - m^2

-- Define the expression to be evaluated
def expr : ℝ := m * Real.sqrt n / (2 * Real.cos (Real.pi * 27 / 180)^2 - 1)

-- Prove the expression equals 2
theorem problem_statement : expr = 2 :=
by
  -- proof will be placed here
  sorry

end problem_statement_l530_530301


namespace largest_n_factorial_product_l530_530184

theorem largest_n_factorial_product : ∃ n : ℕ, (∀ m : ℕ, m ≥ n → ¬ (m! = ∏ i in (range (m - 4)).map (λ k, k + 1 + m), (1 : ℕ))) ∧ n = 1 :=
sorry

end largest_n_factorial_product_l530_530184


namespace lambda_less_than_37_over_8_l530_530845

theorem lambda_less_than_37_over_8 (λ : ℝ) (h : ∀ n : ℕ, 0 < n → 2 * n^2 - n - 3 < (5 - λ) * (n + 1) * 2^n) : λ < 37 / 8 :=
sorry

end lambda_less_than_37_over_8_l530_530845


namespace machine_work_hours_l530_530901

theorem machine_work_hours (A B : ℝ) (x : ℝ) (hA : A = 1 / 8) (hB : B = A / 4)
  (hB_rate : B = 1 / 32) (B_time : B * 8 = 1 - x / 8) : x = 6 :=
by
  sorry

end machine_work_hours_l530_530901


namespace angles_of_triangle_KHM_l530_530370

open Real
open BigOperators

-- Definitions of points and lengths
variable (A B C K H M: Point)
variable (alpha: ℝ)

-- Conditions
axiom B_between_A_and_C: collinear A B C ∧ (B ∈ segment A C)
axiom AK_eq_KB: dist A K = dist K B
axiom BH_eq_HC: dist B H = dist H C
axiom angle_AKB_eq_alpha: ∠ A K B = α
axiom angle_BHC_eq_pi_minus_alpha: ∠ B H C = π - α

-- Midpoint condition
axiom M_is_midpoint_AC: M = midpoint A C

-- Goal: angles of triangle K H M
theorem angles_of_triangle_KHM:
  angles_of_triangle K H M = [90, α / 2, 90 - α / 2] := by
  sorry

end angles_of_triangle_KHM_l530_530370


namespace correct_exponent_calculation_l530_530527

theorem correct_exponent_calculation (x : ℝ) : (-x^3)^4 = x^12 := 
by sorry

end correct_exponent_calculation_l530_530527


namespace sufficient_but_not_necessary_l530_530712

def quadratic_real_roots (a : ℝ) : Prop :=
  (∃ x : ℝ, x^2 - 2 * x + a = 0)

theorem sufficient_but_not_necessary (a : ℝ) :
  (quadratic_real_roots 1) ∧ (∀ a > 1, ¬ quadratic_real_roots a) :=
sorry

end sufficient_but_not_necessary_l530_530712


namespace f_f_1_l530_530723

def f (x : ℝ) : ℝ :=
if x > 0 then 2^x - 4 else 2 * x

theorem f_f_1 : f (f 1) = -4 := by
  sorry

end f_f_1_l530_530723


namespace find_X_l530_530683

theorem find_X (X : ℝ) 
  (h : 2 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * X)) = 1600.0000000000002) : 
  X = 1.25 := 
sorry

end find_X_l530_530683


namespace distance_from_point_to_line_l530_530717

theorem distance_from_point_to_line (m : ℝ) 
  (h : dist (3, m) (1, 1, -4) = Real.sqrt 2) : 
  m = 3 ∨ m = -1 := 
by
  sorry

end distance_from_point_to_line_l530_530717


namespace axis_of_symmetry_shifted_sine_function_l530_530975

theorem axis_of_symmetry_shifted_sine_function :
  ∀ x : ℝ, 
    (f x = sin (2 * x - π / 6)) →
    (g x = f (x - π / 12)) →
    x = 5 * π / 12 :=
by
  intros x f_eq g_eq
  sorry

end axis_of_symmetry_shifted_sine_function_l530_530975


namespace find_removed_number_l530_530081

theorem find_removed_number (l : List ℕ) (avg : ℕ) (h1 : l = [1,2,3,4,5,6,7,8,9,10,11]) 
  (h2 : avg = 61) : ∃ (n : ℕ), (list.sum l - n) / 10 = 6.1 :=
by
  use 5
  rw [h1, sum_list, h2]
  sorry -- skip the actual proof steps

def sum_list : List ℕ := [1,2,3,4,5,6,7,8,9,10,11]

def list_sum_to_66 : list.sum sum_list = 66 := sorry -- proving List sum from list's sum definition

end find_removed_number_l530_530081


namespace solve_for_five_minus_a_l530_530269

theorem solve_for_five_minus_a (a b : ℤ) 
  (h1 : 5 + a = 6 - b)
  (h2 : 6 + b = 9 + a) : 
  5 - a = 6 := 
by 
  sorry

end solve_for_five_minus_a_l530_530269


namespace convex_polygon_diagonals_l530_530786

theorem convex_polygon_diagonals (n : ℕ) (h : n = 30) : 
  let total_pairs := (n * (n - 1)) / 2 in (total_pairs - n) = 405 := 
by
  simp only [total_pairs]
  sorry

end convex_polygon_diagonals_l530_530786


namespace find_150th_letter_in_pattern_l530_530449

/--
Given the repeating pattern "XYZ" with a cycle length of 3,
prove that the 150th letter in this pattern is 'Z'.
-/
theorem find_150th_letter_in_pattern : 
  let pattern := "XYZ"
  let cycle_length := String.length pattern
in (150 % cycle_length = 0) → "Z" := 
sorry

end find_150th_letter_in_pattern_l530_530449


namespace watch_cost_l530_530163

-- Definitions based on conditions
def initial_money : ℤ := 1
def money_from_david : ℤ := 12
def money_needed : ℤ := 7

-- Indicating the total money Evan has after receiving money from David
def total_money := initial_money + money_from_david

-- The cost of the watch based on total money Evan has and additional money needed
def cost_of_watch := total_money + money_needed

-- Proving the cost of the watch
theorem watch_cost : cost_of_watch = 20 := by
  -- We are skipping the proof steps here
  sorry

end watch_cost_l530_530163


namespace convex_polygon_diagonals_l530_530817

theorem convex_polygon_diagonals (n : ℕ) (h : n = 30) : 
  let V := n in
  let total_pairs := V * (V - 1) / 2 in
  let adjacent_pairs := n in
  let diagonals := total_pairs - adjacent_pairs in
  diagonals = 405 :=
  by
  -- number of sides is 30
  have hn : n = 30 := h
  -- calculate the total pairs
  have total_pairs_calc : total_pairs = 30 * 29 / 2 := by sorry
  -- calculate the adjacent pairs
  have adjacent_pairs_calc : adjacent_pairs = 30 := by sorry
  -- calculate the diagonals
  have diagonals_calc : diagonals = (30 * 29 / 2) - 30 := by sorry
  -- proved statement
  show 405 = 405, by rfl

end convex_polygon_diagonals_l530_530817


namespace probability_of_event_A_l530_530850

-- Definitions based on conditions
def card_label := {n : ℕ // n ≥ 1 ∧ n ≤ 6}
def is_odd (a : card_label) : Prop := a.val % 2 = 1
def is_even (a : card_label) : Prop := a.val % 2 = 0

-- The event we are interested in
def event_A : Prop := 
  ∃ a b : card_label, 
  is_odd a ∧ is_even b ∧ b.val > a.val

-- Calculation of the probability
theorem probability_of_event_A : 
  ∑ (a : card_label) in {a : card_label | is_odd a}.toFinset, 
    (1 / 6 : ℚ) * ∑ (b : card_label) in {b : card_label | is_even b ∧ b.val > a.val}.toFinset, ((1 : ℚ) / (({b : card_label | b.val > a.val}.toFinset.card : ℚ))) = (17 / 45 : ℚ) := 
sorry

end probability_of_event_A_l530_530850


namespace range_of_k_l530_530236

theorem range_of_k 
  (h : ∀ x : ℝ, x = 1 → k^2 * x^2 - 6 * k * x + 8 ≥ 0) :
  k ≥ 4 ∨ k ≤ 2 := by
sorry

end range_of_k_l530_530236


namespace smallest_value_l530_530273

theorem smallest_value (x : ℝ) (h : 0 < x ∧ x < 1) : 
  (∀ y ∈ ({x, x ^ 3, 3 * x, x ^ (1 / 3), 1 / (x + 1)} : set ℝ), y ≥ x ^ 3) :=
by
  sorry

end smallest_value_l530_530273


namespace count_integers_satisfying_inequality_l530_530827

theorem count_integers_satisfying_inequality : 
  { n : ℤ | -15 ≤ n ∧ n ≤ 10 ∧ (n - 1) * (n + 3) * (n + 7) < 0 }.card = 11 := 
sorry

end count_integers_satisfying_inequality_l530_530827


namespace sin_sum_cos_product_tan_sum_tan_product_l530_530375

theorem sin_sum_cos_product
  (A B C : ℝ)
  (h : A + B + C = π) : 
  (Real.sin A + Real.sin B + Real.sin C = 4 * Real.cos (A / 2) * Real.cos (B / 2) * Real.cos (C / 2)) :=
sorry

theorem tan_sum_tan_product
  (A B C : ℝ)
  (h : A + B + C = π) :
  (Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C) := 
sorry

end sin_sum_cos_product_tan_sum_tan_product_l530_530375


namespace lineup_count_l530_530156

theorem lineup_count (n k : ℕ) (h : n = 13) (k_eq : k = 4) : (n.choose k) = 715 := by
  sorry

end lineup_count_l530_530156


namespace more_polygons_in_first_group_l530_530903

-- Define the points A_1, A_2, ..., A_16
def points : Finset ℕ := Finset.range 16

-- Define the condition of a convex polygon using these points
def is_convex_polygon (s : Finset ℕ) : Prop :=
  3 ≤ s.card ∧ s.card ≤ 16

-- Define groups based on inclusion of A1, which in Lean we can represent as 0
def group1 : Finset (Finset ℕ) :=
  {s ∈ Finset.powerset points | 0 ∈ s ∧ is_convex_polygon s}
def group2 : Finset (Finset ℕ) :=
  {s ∈ Finset.powerset points | 0 ∉ s ∧ is_convex_polygon s}

-- Theorem statement
theorem more_polygons_in_first_group :
  group1.card > group2.card :=
sorry

end more_polygons_in_first_group_l530_530903


namespace smallest_k_prod_is_integer_l530_530408

open BigOperators

/-- The recursively defined sequence (a_n) -/
noncomputable def a : ℕ → ℝ
| 0         := 1
| 1         := real.rpow 3 (1/17)
| (n + 2) := a (n + 1) * (a n ^ 3)

theorem smallest_k_prod_is_integer
  (h : ∀ k: ℕ, 0 < k → a 1 * a 2 ... * a k ∈ ℤ → k = 19) : 
  ∃ k: ℕ, k = 19 ∧ ((∏ i in finset.range (k + 1), a i).val ∈ ℤ) :=
begin
  sorry
end

end smallest_k_prod_is_integer_l530_530408


namespace problem_solution_l530_530598

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def functional_identity (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x + 2) = f (x + 2)

noncomputable def given_function : ℝ → ℝ :=
  λ x, x * real.log (x + 1)

theorem problem_solution (f : ℝ → ℝ) 
  (h1 : even_function f)
  (h2 : functional_identity f)
  (h3 : ∀ x, 0 ≤ x ∧ x ≤ 2 → f x = given_function x) :
  (f (-1) = real.log 2) ∧
  (∀ x, f (x + 4) = f x) := 
by sorry

end problem_solution_l530_530598


namespace largest_n_product_consecutive_integers_l530_530196

theorem largest_n_product_consecutive_integers : ∃ (n : ℕ), (∀ (x : ℕ), n! = (list.Ico x (x + n - 4)).prod) ∧ n = 4 := 
by
  sorry

end largest_n_product_consecutive_integers_l530_530196


namespace circle_y_coords_sum_l530_530615

theorem circle_y_coords_sum (x y : ℝ) (hc : (x + 3)^2 + (y - 5)^2 = 64) (hx : x = 0) : y = 5 + Real.sqrt 55 ∨ y = 5 - Real.sqrt 55 → (5 + Real.sqrt 55) + (5 - Real.sqrt 55) = 10 := 
by
  intros
  sorry

end circle_y_coords_sum_l530_530615


namespace quadratic_integer_coefficients_l530_530896

variable {R : Type*} [CommRing R] [CharZero R]

noncomputable def P (a b x : R) : R := x^2 + a * x + b

theorem quadratic_integer_coefficients 
  (a b : ℚ) 
  (h1 : (P a b 0)^2 ∈ ℤ) 
  (h2 : (P a b 1)^2 ∈ ℤ) 
  (h3 : (P a b 2)^2 ∈ ℤ) 
  (hb : b ≠ 2) 
  : a ∈ ℤ ∧ b ∈ ℤ :=
by
  sorry

end quadratic_integer_coefficients_l530_530896


namespace total_cost_of_topsoil_l530_530054

-- Definitions
def cost_per_cubic_foot : ℝ := 8
def cubic_yard_to_cubic_foot : ℝ := 27
def volume_in_cubic_yards : ℕ := 8

-- The total cost of 8 cubic yards of topsoil
theorem total_cost_of_topsoil : volume_in_cubic_yards * cubic_yard_to_cubic_foot * cost_per_cubic_foot = 1728 := by
  sorry

end total_cost_of_topsoil_l530_530054


namespace number_of_diagonals_30_sides_l530_530773

def number_of_diagonals (n : ℕ) : ℕ :=
  nat.choose n 2 - n

theorem number_of_diagonals_30_sides :
  number_of_diagonals 30 = 405 :=
by {
  rw [number_of_diagonals, nat.choose, nat.factorial, nat.factorial, nat.factorial],
  -- The proof would proceed to simplify the combination and subtraction, but we use sorry to skip details.
  sorry,
}

end number_of_diagonals_30_sides_l530_530773


namespace sum_of_solutions_eq_l530_530682

noncomputable def sum_of_real_solutions (f : ℝ → ℝ) (g : ℝ → ℝ) (h : ℝ → ℝ) : ℝ :=
  ∑ x in (f = g).roots.filter (λ x : ℝ, h x ≠ 0), x

theorem sum_of_solutions_eq : 
  let f := λ x : ℝ, (x - 3) 
  let g := λ x : ℝ, (x - 8)
  let h := λ x : ℝ, (x^2 + 5x + 2) * (x^2 - 15x)
  sum_of_real_solutions f g h = 20 / 3 :=
by
  sorry

end sum_of_solutions_eq_l530_530682


namespace minimum_value_ineq_l530_530897

variable {a b c : ℝ}

theorem minimum_value_ineq (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2 * a + 1) * (b^2 + 2 * b + 1) * (c^2 + 2 * c + 1) / (a * b * c) ≥ 64 :=
sorry

end minimum_value_ineq_l530_530897


namespace tangent_line_exists_function_A_tangent_line_exists_function_C_l530_530846

-- Conditions for the functions
def function_A (x : ℝ) : ℝ := x^3 + 2 * x^2 + 8
def function_C (x : ℝ) : ℝ := x * Real.exp x

-- Statement of the proof problem
theorem tangent_line_exists_function_A :
  ∃ x₀ b : ℝ, (deriv function_A x₀ = 1/2) ∧ (function_A x₀ = 1/2 * x₀ + b) :=
sorry

theorem tangent_line_exists_function_C :
  ∃ x₀ b : ℝ, (deriv function_C x₀ = 1/2) ∧ (function_C x₀ = 1/2 * x₀ + b) :=
sorry

end tangent_line_exists_function_A_tangent_line_exists_function_C_l530_530846


namespace top_three_ranking_l530_530138

def Person : Type :=
| Xiaojun
| Xiaomin
| Xiaole

def Rank : Type :=
| First
| Second
| Third

open Person
open Rank

def teacher_guess_1 (ranking : Person → Rank) : Prop := ranking Xiaojun = First
def teacher_guess_2 (ranking : Person → Rank) : Prop := ranking Xiaomin ≠ First
def teacher_guess_3 (ranking : Person → Rank) : Prop := ranking Xiaole ≠ Third

def exactly_one_correct (ranking : Person → Rank) : Prop :=
  (teacher_guess_1 ranking ∧ ¬teacher_guess_2 ranking ∧ ¬teacher_guess_3 ranking) ∨
  (¬teacher_guess_1 ranking ∧ teacher_guess_2 ranking ∧ ¬teacher_guess_3 ranking) ∨
  (¬teacher_guess_1 ranking ∧ ¬teacher_guess_2 ranking ∧ teacher_guess_3 ranking)

theorem top_three_ranking :
  ∃ (ranking : Person → Rank), exactly_one_correct ranking ∧
  ranking Xiaomin = First ∧
  ranking Xiaole = Second ∧
  ranking Xiaojun = Third :=
by
  sorry

end top_three_ranking_l530_530138


namespace determine_b2_possible_values_l530_530131

-- Definition of the sequence conditions
def b_n (n : ℕ) : ℕ := sorry

-- Conditions
def b1 : ℕ := 1000
def b2 : ℕ := sorry  -- We need the total count of valid b2 values
def b1004 : ℕ := 2

-- Main theorem statement
theorem determine_b2_possible_values :
  b1 = 1000 ∧ b1004 = 2 ∧
  (∀ n : ℕ, b_n (n + 2) = |b_n (n + 1) - b_n n|) →
  (∃ count : ℕ, count = 374 ∧ 
    ∃ b2_values : finset ℕ, b2_values.card = count ∧ 
    (∀ b2_val ∈ b2_values, b2_val < 1000 ∧ b2_val % 2 = 0 ∧ gcd (b1, b2_val) = 2)) :=
sorry

end determine_b2_possible_values_l530_530131


namespace square_perimeter_from_area_l530_530569

def square_area (s : ℝ) : ℝ := s * s -- Definition of the area of a square based on its side length.
def square_perimeter (s : ℝ) : ℝ := 4 * s -- Definition of the perimeter of a square based on its side length.

theorem square_perimeter_from_area (s : ℝ) (h : square_area s = 900) : square_perimeter s = 120 :=
by {
  sorry -- Placeholder for the proof.
}

end square_perimeter_from_area_l530_530569


namespace sin_180_degrees_l530_530640

theorem sin_180_degrees : Real.sin (180 * Real.pi / 180) = 0 := 
by
  sorry

end sin_180_degrees_l530_530640


namespace nth_150th_letter_in_XYZ_l530_530489

def pattern : List Char := ['X', 'Y', 'Z']

def nth_letter (n : Nat) : Char :=
  pattern[(n - 1) % pattern.length]

theorem nth_150th_letter_in_XYZ :
  nth_letter 150 = 'Z' :=
by
  sorry

end nth_150th_letter_in_XYZ_l530_530489


namespace diagonals_in_30_sided_polygon_l530_530820

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem diagonals_in_30_sided_polygon :
  number_of_diagonals 30 = 405 :=
by sorry

end diagonals_in_30_sided_polygon_l530_530820


namespace program_arrangement_possible_l530_530852

theorem program_arrangement_possible (initial_programs : ℕ) (additional_programs : ℕ) 
  (h_initial: initial_programs = 6) (h_additional: additional_programs = 2) : 
  ∃ arrangements, arrangements = 56 :=
by
  sorry

end program_arrangement_possible_l530_530852


namespace diagonals_of_30_sided_polygon_l530_530804

theorem diagonals_of_30_sided_polygon : 
  ∀ (n : ℕ), n = 30 → (n * (n - 3)) / 2 = 405 := 
by
  intro n h
  rw h
  simp
  sorry

end diagonals_of_30_sided_polygon_l530_530804


namespace angle_between_vectors_l530_530743

variables (a b : EuclideanSpace ℝ (Fin 2)) 
          (theta : ℝ)

-- Given conditions
def dot_product_condition : Prop := (a ⬝ b) = -1
def magnitude_condition_a : Prop := ‖a‖ = 2
def magnitude_condition_b : Prop := ‖b‖ = 1

-- Statement to be proved
theorem angle_between_vectors (h1 : dot_product_condition a b)
                               (h2 : magnitude_condition_a a)
                               (h3 : magnitude_condition_b b) :
  θ = (2 * Real.pi) / 3 := by
  sorry

end angle_between_vectors_l530_530743


namespace new_device_significant_improvement_l530_530548

def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

def variance (data : List ℝ) (mean : ℝ) : ℝ :=
  (data.foldl (λ acc x => acc + (x - mean) ^ 2) 0) / data.length

def significantImprovement (oldData newData : List ℝ) : Prop :=
  let x := mean oldData
  let y := mean newData
  let s1_squared := variance oldData x
  let s2_squared := variance newData y
  y - x ≥ 2 * Real.sqrt ((s1_squared + s2_squared) / oldData.length)

theorem new_device_significant_improvement :
  significantImprovement [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
                         [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5] :=
by
  sorry

end new_device_significant_improvement_l530_530548


namespace perpendicular_line_through_point_l530_530016

theorem perpendicular_line_through_point (x y : ℝ) (c : ℝ) (P : ℝ × ℝ) :
  P = (-1, 2) →
  (∀ x y c : ℝ, (2*x - y + c = 0) ↔ (x+2*y-1=0) → (x+2*y-1=0)) →
  ∃ c : ℝ, 2*(-1) - 2 + c = 0 ∧ (2*x - y + c = 0) :=
by
  sorry

end perpendicular_line_through_point_l530_530016


namespace largest_n_lemma_l530_530191
noncomputable def largest_n (n b: ℕ) : Prop := 
  n! = ((n-4) + b)!/(b!)

theorem largest_n_lemma : ∀ n: ℕ, ∀ b: ℕ, b ≥ 4 → (largest_n n b → n = 1) := 
  by
  intros n b hb h
  sorry

end largest_n_lemma_l530_530191


namespace find_value_of_p_l530_530865

theorem find_value_of_p (p q r s t u v w : ℤ)
  (h1 : r + s = -2)
  (h2 : s + (-2) = 5)
  (h3 : t + u = 5)
  (h4 : u + v = 3)
  (h5 : v + w = 8)
  (h6 : w + t = 3)
  (h7 : q + r = s)
  (h8 : p + q = r) :
  p = -25 := by
  -- proof skipped
  sorry

end find_value_of_p_l530_530865


namespace representation_of_n_digit_numbers_l530_530374

-- Define uniform numbers
def is_uniform (a : ℕ) : Prop :=
  let digits := a.digits 10 in
  ∀ d ∈ digits, d = digits.head'.getD 0

-- Theorem to prove
theorem representation_of_n_digit_numbers (n : ℕ) (a : ℕ) (h : a < 10 ^ n) :
  ∃ k (u : ℕ → ℕ), k ≤ n + 1 ∧ (∀ i, i < k → is_uniform (u i)) ∧ a = ∑ i in Finset.range k, u i :=
begin
  sorry
end

end representation_of_n_digit_numbers_l530_530374


namespace range_of_f_l530_530832

variable {a b c : ℝ} (h_a : a > 0)

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem range_of_f : 
  (0 ≤ x ∧ x ≤ 1) → 
  (h1 : -2 * a ≤ b ∧ b ≤ 0 → Set.range f = Set.Icc ((-b^2) / (4 * a) + c) (a + b + c)) ∧ 
  (h2 : (b < -2 * a ∨ b > 0) → Set.range f = Set.Icc c (a + b + c)) :=
by sorry

end range_of_f_l530_530832


namespace length_and_slope_for_P_on_C_max_min_distance_MQ_l530_530237

-- Define the circle C
def is_on_circle_C (M : ℝ × ℝ) : Prop :=
  (M.1)^2 + (M.2)^2 - 4*M.1 - 14*M.2 + 45 = 0

-- Define point Q
def Q : ℝ × ℝ := (-2, 3)

-- Define circle center and radius
def circle_center : ℝ × ℝ := (2, 7)
def circle_radius : ℝ := 2 * Real.sqrt 2

-- Define distance function
def dist (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Length and slope for P(a, a+1) on the circle
theorem length_and_slope_for_P_on_C (a : ℝ) (P : ℝ × ℝ) (PQ_length : ℝ) (PQ_slope : ℝ) :
  is_on_circle_C (P.1, P.2) ∧ P = (4, 5) →
  (PQ_length = dist P Q) ∧ (PQ_slope = (Q.2 - P.2) / (Q.1 - P.1)) →
  PQ_length = 2 * Real.sqrt 10 ∧ PQ_slope = 1/3 :=
sorry

-- Maximum and minimum values of |MQ|
theorem max_min_distance_MQ (M Q : ℝ × ℝ) :
  is_on_circle_C M →
  dist Q circle_center = 4 * Real.sqrt 2 →
  ∃ MQ_max MQ_min, MQ_max = 6 * Real.sqrt 2 ∧ MQ_min = 2 * Real.sqrt 2 :=
sorry

end length_and_slope_for_P_on_C_max_min_distance_MQ_l530_530237


namespace identify_stolen_bag_with_two_weighings_l530_530132

-- Definition of the weights of the nine bags
def weights : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

-- Statement of the problem: Using two weighings on a balance scale without weights,
-- prove that it is possible to identify the specific bag from which the treasure was stolen.
theorem identify_stolen_bag_with_two_weighings (stolen_bag : {n // n < 9}) :
  ∃ (group1 group2 : List ℕ), group1 ≠ group2 ∧ (group1.sum = 11 ∨ group1.sum = 15) ∧ (group2.sum = 11 ∨ group2.sum = 15) →
  ∃ (b1 b2 b3 : ℕ), b1 ≠ b2 ∧ b1 ≠ b3 ∧ b2 ≠ b3 ∧ b1 + b2 + b3 = 6 ∧ (b1 + b2 = 11 ∨ b1 + b2 = 15) := sorry

end identify_stolen_bag_with_two_weighings_l530_530132


namespace probability_at_least_one_alarm_trigger_l530_530098

-- Definitions for the conditions
def prob_trigger (p : ℝ) : set (ℕ → Prop) := {s | s = λ n, n = 1 ∧ s n = true}
def event_space := {s | s = λ n : ℕ, n = 1}
def prob_event (a b : ℝ) := a * b

-- Main theorem statement
theorem probability_at_least_one_alarm_trigger :
  let p := 0.4 in
  let q := 1 - p in
  let two_indep_alarms := 2 • (prob_trigger p) + (prob_event p p) in
  (∃ (prob_one_alarm : ℝ), prob_one_alarm = 2 * p * q) ∧ (∃ (prob_both_alarms : ℝ), prob_both_alarms = p * p) ∧
  ((2 * p * q) + (p * p) = 0.64) :=
by 
  sorry

end probability_at_least_one_alarm_trigger_l530_530098


namespace find_m_repeated_root_l530_530718

theorem find_m_repeated_root (m : ℝ) :
  (∃ x : ℝ, (x - 1) ≠ 0 ∧ (m - 1) - x = 0) → m = 2 :=
by
  sorry

end find_m_repeated_root_l530_530718


namespace total_weight_of_remaining_chocolate_eggs_l530_530352

-- Define the conditions as required:
def marie_makes_12_chocolate_eggs : ℕ := 12
def weight_of_each_egg : ℕ := 10
def number_of_boxes : ℕ := 4
def boxes_discarded : ℕ := 1

-- Define the proof statement
theorem total_weight_of_remaining_chocolate_eggs :
  let eggs_per_box := marie_makes_12_chocolate_eggs / number_of_boxes,
      remaining_boxes := number_of_boxes - boxes_discarded,
      remaining_eggs := remaining_boxes * eggs_per_box,
      total_weight := remaining_eggs * weight_of_each_egg
  in total_weight = 90 := by
  sorry

end total_weight_of_remaining_chocolate_eggs_l530_530352


namespace maximize_expected_profit_l530_530556

open Classical

-- Define the relevant variables and conditions
variables (k : ℕ) (w : ℝ)
-- Define the probability of not being intercepted after n attempts
def prob_no_interception (n : ℕ) : ℝ := (1 - 1 / k.to_real) ^ n

-- Define the expected profit function
def expected_profit (n : ℕ) : ℝ := w * n.to_real * prob_no_interception k n

theorem maximize_expected_profit (k : ℕ) (w : ℝ) (hk : k > 0) : 
  ∃ n, n = k - 1 ∧ ∀ m, expected_profit k w m ≤ expected_profit k w (k - 1) :=
by
  sorry

end maximize_expected_profit_l530_530556


namespace measure_of_angle_A_maximum_area_triangle_l530_530855

variables {A B C : ℝ} {a b c : ℝ}

-- Problem 1: Prove the measure of angle A
theorem measure_of_angle_A (h : (b^2 - a^2 - c^2) * sin A * cos A = a * c * cos (A + C)) (hAcute : 0 < A ∧ A < π / 2)
  (hCosLaw : a^2 = b^2 + c^2 - 2 * b * c * cos A) : A = π / 4 :=
sorry

-- Problem 2: Prove the maximum area of triangle ABC given a = sqrt 2
theorem maximum_area_triangle (a_sq : a = sqrt 2) (h : 0 < A ∧ A < π / 2) :
  let A := π / 4 in (∀ b c : ℝ, (b * c * (sqrt 2 / 2)) / 2 ≤ (sqrt 2 + 1) / 2) :=
sorry

end measure_of_angle_A_maximum_area_triangle_l530_530855


namespace sums_equal_l530_530362

theorem sums_equal :
  let S1 := 1 + 22 + 333 + 4444 + 55555 + 666666 + 7777777 + 88888888 + 999999999
  let S2 := 9 + 98 + 987 + 9876 + 98765 + 987654 + 9876543 + 98765432 + 987654321 
  S1 = S2 := 
by {
  let S1 := 1 + 22 + 333 + 4444 + 55555 + 666666 + 7777777 + 88888888 + 999999999
  let S2 := 9 + 98 + 987 + 9876 + 98765 + 987654 + 9876543 + 98765432 + 987654321 
  have : S1 = 1097393685 := sorry,
  have : S2 = 1097393685 := sorry,
  rw [← this, ← this_1],
}

end sums_equal_l530_530362


namespace max_value_of_a_b_c_l530_530234

theorem max_value_of_a_b_c (a b c : ℤ) (h1 : a + b = 2006) (h2 : c - a = 2005) (h3 : a < b) : 
  a + b + c = 5013 :=
sorry

end max_value_of_a_b_c_l530_530234


namespace negative_product_probability_l530_530531

noncomputable def m : Set ℤ := { -6, -5, -4, -3, -2 }

noncomputable def t : Set ℤ := { -3, -2, -1, 0, 1, 2, 3, 4 }

theorem negative_product_probability :
  let num_pairs := Set.prod m t
  let negative_pairs := { p : ℤ × ℤ | p ∈ num_pairs ∧ p.fst * p.snd < 0 }
  let probability := (Set.card negative_pairs : ℚ) / (Set.card num_pairs : ℚ)
  probability = 5 / 8 :=
by
  sorry

end negative_product_probability_l530_530531


namespace number_of_B_students_l530_530851

theorem number_of_B_students (x : ℝ) (h1 : 0.8 * x + x + 1.2 * x = 40) : x = 13 :=
  sorry

end number_of_B_students_l530_530851


namespace house_cost_l530_530047

theorem house_cost (land_acres : ℕ) (land_cost_per_acre : ℕ)
  (num_cows : ℕ) (cost_per_cow : ℕ)
  (num_chickens : ℕ) (cost_per_chicken : ℕ)
  (installation_hours : ℕ) (cost_per_hour : ℕ)
  (equipment_cost : ℕ) (total_cost : ℕ)
  (h1 : land_acres = 30)
  (h2 : land_cost_per_acre = 20)
  (h3 : num_cows = 20)
  (h4 : cost_per_cow = 1000)
  (h5 : num_chickens = 100)
  (h6 : cost_per_chicken = 5)
  (h7 : installation_hours = 6)
  (h8 : cost_per_hour = 100)
  (h9 : equipment_cost = 6000)
  (h10 : total_cost = 147700) :
  let land_cost := land_acres * land_cost_per_acre,
      cows_cost := num_cows * cost_per_cow,
      chickens_cost := num_chickens * cost_per_chicken,
      solar_install_cost := installation_hours * cost_per_hour,
      solar_total_cost := solar_install_cost + equipment_cost,
      other_total_cost := land_cost + cows_cost + chickens_cost + solar_total_cost,
      house_cost := total_cost - other_total_cost
  in house_cost = 120000 :=
by
  sorry

end house_cost_l530_530047


namespace second_hand_travel_distance_l530_530958

-- Define the conditions
def radius_of_second_hand : ℝ := 8
def time_in_minutes : ℕ := 45

-- Define the expected result
def expected_distance : ℝ := 720 * Real.pi

-- Define the Lean statement to prove
theorem second_hand_travel_distance : (45 * 2 * Real.pi * 8) = 720 * Real.pi :=
by
  sorry

end second_hand_travel_distance_l530_530958


namespace riding_school_horseshoes_l530_530103

def horseshoes_needed_per_horse : ℕ := 4
def horses_per_farm : ℕ := 2
def horses_per_stable : ℕ := 5
def farms : ℕ := 2
def stables : ℕ := 2
def iron_available : ℕ := 400
def iron_per_horseshoe : ℕ := 2

theorem riding_school_horseshoes : 
  let total_horseshoes := iron_available / iron_per_horseshoe in
  let total_farm_horses := farms * horses_per_farm in
  let total_stable_horses := stables * horses_per_stable in
  let total_horses := total_farm_horses + total_stable_horses in
  let horseshoes_needed := total_horses * horseshoes_needed_per_horse in
  let horseshoes_left := total_horseshoes - horseshoes_needed in
  horseshoes_left / horseshoes_needed_per_horse = 36 :=
by
  sorry

end riding_school_horseshoes_l530_530103


namespace letter_at_position_150_l530_530475

theorem letter_at_position_150 : 
  (∀ n, n > 0 → ∃ i, i ∈ {1, 2, 3} ∧ "XYZ".to_list[i-1] = "XYZ".to_list[(n - 1) % 3]) →
  ("XYZ".to_list[(150 - 1) % 3] = 'Z') :=
by
  sorry

end letter_at_position_150_l530_530475


namespace sin_180_eq_zero_l530_530623

theorem sin_180_eq_zero : Real.sin (180 * Real.pi / 180) = 0 := by
  -- Simplifying the angle, 180 degrees = π radians
  let angle := 180 * Real.pi / 180
  have h : angle = Real.pi := by
    simp [angle, Real.pi]
  rw h
  -- From the unit circle, we know the sine of angle π is 0
  exact Real.sin_pi

end sin_180_eq_zero_l530_530623


namespace diagonals_in_convex_polygon_with_30_sides_l530_530769

theorem diagonals_in_convex_polygon_with_30_sides : 
  let n := 30 in
  ∑ i j in finset.range n, (i + 1 - 1) * (j + 1 - 1) * (nat.choose 30 2 - 30) / 2 = 202 := 
by
  sorry

end diagonals_in_convex_polygon_with_30_sides_l530_530769


namespace maximum_value_expression_l530_530200

def max_expression_value (l : List ℕ) : ℕ :=
  List.foldr (λ x₁ x₂ => abs (x₁ - x₂)) 0 l

theorem maximum_value_expression :
  ∀ (l : List ℕ), 
  (∀ x ∈ l, x ∈ Finset.range 1 1991) →
  l.nodup →
  l.length = 1990 →
  max_expression_value l ≤ 1989 :=
by
  intros l h1 h2 h3
  sorry

end maximum_value_expression_l530_530200


namespace functional_inequality_l530_530175

theorem functional_inequality (f : ℝ → ℝ) (C : ℝ) (hC : 0 < C) :
  (∀ x y : ℝ, 0 < x → 0 < y → f x = C / x^2 → f y = C / y^2 →
    (f x / y^2 - f y / x^2 ≤ (1 / x - 1 / y) ^ 2)) :=
begin
  sorry
end

end functional_inequality_l530_530175


namespace arrangement_count_l530_530601

open Nat

noncomputable def combinations (n k : ℕ) : ℕ :=
  n.choose k

theorem arrangement_count :
  let saturday_combinations := combinations 7 3
  let sunday_combinations := combinations 4 3
  saturday_combinations * sunday_combinations = 140 := by
  let saturday_combinations := combinations 7 3
  let sunday_combinations := combinations 4 3
  calc
    saturday_combinations * sunday_combinations
      = 35 * 4 : by rfl  -- since combinations 7 3 = 35 and combinations 4 3 = 4
      ... = 140 : by rfl

end arrangement_count_l530_530601


namespace smallest_common_multiple_8_6_l530_530513

theorem smallest_common_multiple_8_6 : ∃ n : ℕ, n > 0 ∧ (8 ∣ n) ∧ (6 ∣ n) ∧ ∀ m : ℕ, m > 0 ∧ (8 ∣ m) ∧ (6 ∣ m) → n ≤ m :=
begin
  use 24,
  split,
  { exact zero_lt_24, },
  split,
  { exact dvd.intro 3 rfl, },
  split,
  { exact dvd.intro 4 rfl, },
  intros m hm h8 h6,
  -- actual proof here
  sorry
end

end smallest_common_multiple_8_6_l530_530513


namespace diagonal_count_of_convex_polygon_30_sides_l530_530793
-- Importing the entire Mathlib library for necessary mathematical constructs

theorem diagonal_count_of_convex_polygon_30_sides : 
  let n := 30 in
  let num_diagonals := (n * (n - 3)) / 2 in
  num_diagonals = 405 := 
by {
  let n := 30
  let num_diagonals := (n * (n - 3)) / 2
  show num_diagonals = 405,
  sorry
}

end diagonal_count_of_convex_polygon_30_sides_l530_530793


namespace letter_150th_in_pattern_l530_530497

def repeating_sequence := "XYZ"

def letter_at_position (n : ℕ) : char :=
  let seq := repeating_sequence.to_list
  seq.get! ((n - 1) % seq.length)

theorem letter_150th_in_pattern : letter_at_position 150 = 'Z' :=
by sorry

end letter_150th_in_pattern_l530_530497


namespace total_cost_of_topsoil_l530_530052

-- Definitions
def cost_per_cubic_foot : ℝ := 8
def cubic_yard_to_cubic_foot : ℝ := 27
def volume_in_cubic_yards : ℕ := 8

-- The total cost of 8 cubic yards of topsoil
theorem total_cost_of_topsoil : volume_in_cubic_yards * cubic_yard_to_cubic_foot * cost_per_cubic_foot = 1728 := by
  sorry

end total_cost_of_topsoil_l530_530052


namespace john_pays_2010_dollars_l530_530872

-- Define the main problem as the number of ways to pay 2010$ using 2, 5, and 10$ notes.
theorem john_pays_2010_dollars :
  ∃ (count : ℕ), count = 20503 ∧
  ∀ (x y z : ℕ), (2 * x + 5 * y + 10 * z = 2010) → (x % 5 = 0) → (y % 2 = 0) → count = 20503 :=
by sorry

end john_pays_2010_dollars_l530_530872


namespace perpendicular_slope_l530_530076

theorem perpendicular_slope (x1 y1 x2 y2 : ℤ) (h1 : x1 = 3) (h2 : y1 = -7) (h3 : x2 = -5) (h4 : y2 = -1) :
  let slope := (y2 - y1) / (x2 - x1) in
  let perp_slope := -1 / slope in
  perp_slope = 4 / 3 :=
by
  -- Placeholder for the actual proof
  sorry

end perpendicular_slope_l530_530076


namespace cars_between_15000_and_20000_l530_530149

theorem cars_between_15000_and_20000 
  (total_cars : ℕ)
  (less_than_15000_ratio : ℝ)
  (more_than_20000_ratio : ℝ)
  : less_than_15000_ratio = 0.15 → 
    more_than_20000_ratio = 0.40 → 
    total_cars = 3000 → 
    ∃ (cars_between : ℕ),
      cars_between = total_cars - (less_than_15000_ratio * total_cars + more_than_20000_ratio * total_cars) ∧ 
      cars_between = 1350 :=
by
  sorry

end cars_between_15000_and_20000_l530_530149


namespace find_150th_letter_l530_530461

theorem find_150th_letter :
  let pattern := ['X', 'Y', 'Z']
  150 % 3 = 0 -> pattern[(150 % 3 + 2) % 3] = 'Z' :=
begin
  intros pattern h,
  simp at *,
  exact rfl,
end

end find_150th_letter_l530_530461


namespace problem1_problem2_l530_530253

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := Real.sin x ^ 4
def g (x : ℝ) : ℝ := f x + f (Real.pi / 2 - x)

-- Problem 1: Maximum and minimum values of g(x) in the interval [π/6, 3π/8]
theorem problem1 :
  (∀ x ∈ Set.Icc (Real.pi / 6) (3 * Real.pi / 8), g x ≤ 3 / 4) ∧ 
  (∃ x ∈ Set.Icc (Real.pi / 6) (3 * Real.pi / 8), g x = 3 / 4) ∧
  (∀ x ∈ Set.Icc (Real.pi / 6) (3 * Real.pi / 8), g x ≥ 1 / 2) ∧ 
  (∃ x ∈ Set.Icc (Real.pi / 6) (3 * Real.pi / 8), g x = 1 / 2) := sorry

-- Problem 2: Sum of f(kπ/180) for k = 1 to 89
theorem problem2 :
  ∑ k in Finset.range 89 | (k + 1), f (↑k * Real.pi / 180) = 133 / 4 := sorry

end problem1_problem2_l530_530253


namespace nth_150th_letter_in_XYZ_l530_530482

def pattern : List Char := ['X', 'Y', 'Z']

def nth_letter (n : Nat) : Char :=
  pattern[(n - 1) % pattern.length]

theorem nth_150th_letter_in_XYZ :
  nth_letter 150 = 'Z' :=
by
  sorry

end nth_150th_letter_in_XYZ_l530_530482


namespace find_a_l530_530036

theorem find_a (a : ℝ) (h_pos : 0 < a) 
(h : a + a^2 = 6) : a = 2 :=
sorry

end find_a_l530_530036


namespace line_intersects_x_axis_at_point_l530_530604

theorem line_intersects_x_axis_at_point :
  (∃ x, 5 * 0 - 2 * x = 10) ↔ (x = -5) ∧ (∃ x, 5 * y - 2 * x = 10 ∧ y = 0) :=
by
  sorry

end line_intersects_x_axis_at_point_l530_530604


namespace shaded_area_is_square_l530_530033

noncomputable def length_of_squares : ℝ := 1

structure Point :=
  (x : ℝ)
  (y : ℝ)

structure RightTriangle :=
  (a b c : ℝ) ( -- sides
  h1 : (a ^ 2) + (b ^ 2) = (c ^ 2) -- right triangle condition
)

def triangle1 : RightTriangle :=
  { a := length_of_squares,
    b := length_of_squares,
    c := (length_of_squares * (2: ℝ)^(1/2)),
    h1 := by { -- proof of the right triangle condition
      have ha : (length_of_squares) ^ 2 = 1 := by norm_num,
      have hb : (length_of_squares) ^ 2 = 1 := by norm_num,
      have hc : (length_of_squares * (2: ℝ)^(1/2)) ^ 2 = 2 := by norm_num,
      rw [ha, hb, hc],
      norm_num }
  }

def triangle2 : RightTriangle :=
  { a := length_of_squares,
    b := length_of_squares,
    c := (length_of_squares * (2: ℝ)^(1/2)),
    h1 := by { -- proof of the right triangle condition
      have ha : (length_of_squares) ^ 2 = 1 := by norm_num,
      have hb : (length_of_squares) ^ 2 = 1 := by norm_num,
      have hc : (length_of_squares * (2: ℝ)^(1/2)) ^ 2 = 2 := by norm_num,
      rw [ha, hb, hc],
      norm_num }
  }

theorem shaded_area_is_square :
  ∃ (shaded_area : ℝ), shaded_area = 1 :=
begin
  use 1,
  -- we assume that the congruent triangles constitute the shaded area of 1 square of side length 1m
  sorry
end

end shaded_area_is_square_l530_530033


namespace distance_between_points_l530_530178

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_points :
  distance 2 5 5 1 = 5 :=
by
  sorry

end distance_between_points_l530_530178


namespace line_intersects_circle_l530_530397

noncomputable def line_eqn (x y : ℝ) : Prop :=
  y = x - 1

theorem line_intersects_circle (a : ℝ)
  (H : a < 3)
  (circle : ∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + a = 0)
  (midpoint : (0, 1) ∈ (set_of (λ p : ℝ × ℝ, line_eqn p.1 p.2))) :
  ∀ x y : ℝ, line_eqn x y ↔ y = x - 1 := sorry

end line_intersects_circle_l530_530397


namespace shaded_region_area_l530_530567

-- Definitions based on given conditions
def num_squares : ℕ := 25
def diagonal_length : ℝ := 10
def squares_in_large_square : ℕ := 16

-- The area of the entire shaded region
def area_of_shaded_region : ℝ := 78.125

-- Theorem to prove 
theorem shaded_region_area 
  (num_squares : ℕ) 
  (diagonal_length : ℝ) 
  (squares_in_large_square : ℕ) : 
  (num_squares = 25) → 
  (diagonal_length = 10) → 
  (squares_in_large_square = 16) → 
  area_of_shaded_region = 78.125 := 
by {
  sorry -- proof to be filled
}

end shaded_region_area_l530_530567


namespace investment_time_ratio_l530_530024

theorem investment_time_ratio (x t : ℕ) (h_inv : 7 * x = t * 5) (h_prof_ratio : 7 / 10 = 70 / (5 * t)) : 
  t = 20 := sorry

end investment_time_ratio_l530_530024


namespace two_squares_share_two_vertices_l530_530887

-- Define the isosceles right triangle with ∠B = 90°
structure IsoscelesRightTriangle (A B C : Type) :=
(angle_B : angle A B C = 90)

-- Define what it means to share exactly two vertices with two given points
def share_two_vertices_with (pts1 pts2: Set Point) : Prop :=
    pts1.card = 4 ∧ pts2.card = 4 ∧ 
    (pts1 ∩ pts2).card = 2

-- Define the problem's statement
theorem two_squares_share_two_vertices
    (A B C : Point) (h_triangle : IsoscelesRightTriangle A B C) :
    ∃ pts1 pts2 : Set Point, share_two_vertices_with pts1 {A, B, C} ∧ 
    share_two_vertices_with pts2 {A, B, C} ∧ 
    pts1 ≠ pts2 :=
sorry

end two_squares_share_two_vertices_l530_530887


namespace zoo_lineup_arrangements_l530_530087

theorem zoo_lineup_arrangements : 
  let fathers_at_ends := 2,
      total_positions := 3,
      middle_positions := factorial total_positions,
      two_children_together := 2,
      total_arrangements := fathers_at_ends * middle_positions * two_children_together
  in total_arrangements = 24 :=
by
  let fathers_at_ends := 2
  let total_positions := 3
  let middle_positions := factorial total_positions
  let two_children_together := 2
  let total_arrangements := fathers_at_ends * middle_positions * two_children_together
  sorry

end zoo_lineup_arrangements_l530_530087


namespace smallest_common_multiple_8_6_l530_530523

theorem smallest_common_multiple_8_6 : 
  ∃ n : ℕ, n > 0 ∧ (n % 8 = 0) ∧ (n % 6 = 0) ∧ ∀ m : ℕ, m > 0 ∧ (m % 8 = 0) ∧ (m % 6 = 0) → m ≥ n :=
begin
  use 24,
  split,
  { norm_num }, -- 24 > 0
  split,
  { norm_num }, -- 24 % 8 = 0
  split,
  { norm_num }, -- 24 % 6 = 0
  { intros m hm,
    cases hm with hp8 hp6,
    norm_num at hp8 hp6,
    sorry -- Prove that 24 is the smallest such number
  }
end

end smallest_common_multiple_8_6_l530_530523


namespace max_distinct_quadrilaterals_max_distinct_quadrilaterals_is_six_l530_530334

-- Define the sequence of convex quadrilaterals
variable {Quadrilateral : Type}

-- Define convexity condition
variable (isConvex : Quadrilateral → Prop)

-- Define transformation condition: cut along a diagonal, flip, glue
variable (transform : Quadrilateral → Quadrilateral)

-- Define the sequence of quadrilaterals
def sequence (n : ℕ) : Quadrilateral
| 0 := sorry  -- Initial quadrilateral
| (n+1) := transform (sequence n)

-- Define the problem of counting distinct quadrilaterals
theorem max_distinct_quadrilaterals (P : ℕ → Quadrilateral) 
  (h1 : ∀ n, isConvex (P n))
  (h2 : ∀ n, P (n+1) = transform (P n)) : 
  ∃ N, ∀ m n, m < N → n < N → m ≠ n → P m ≠ P n :=
sorry

-- Statement showing the maximum number is 6
theorem max_distinct_quadrilaterals_is_six : 
  ∃ N, (∀ P, 
    (∀ n, isConvex (P n)) → 
    (∀ n, P (n+1) = transform (P n)) → 
    (∀ m n, m < N → n < N → m ≠ n → P m ≠ P n)) ∧ N = 6 :=
sorry

end max_distinct_quadrilaterals_max_distinct_quadrilaterals_is_six_l530_530334


namespace sin_alpha_value_l530_530541

variable (α : ℝ)
variable (h_cos : cos α = -4/5)
variable (h_quadrant : π < α ∧ α < 3*π/2)

theorem sin_alpha_value : sin α = -3/5 :=
sorry

end sin_alpha_value_l530_530541


namespace point_lies_on_line_l530_530398

variables (x y z x0 y0 z0 a b c t : ℝ)

def is_collinear (u v : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2, k * v.3)

-- Condition definitions
def point_on_line (M M0 : ℝ × ℝ × ℝ) (m : ℝ × ℝ × ℝ) : Prop :=
  ∃ t : ℝ, (M.1, M.2, M.3) = (M0.1 + t * m.1, M0.2 + t * m.2, M0.3 + t * m.3)

def non_zero_vector (m : ℝ × ℝ × ℝ) : Prop :=
  m ≠ (0, 0, 0)

theorem point_lies_on_line (M0 M : ℝ × ℝ × ℝ) (m : ℝ × ℝ × ℝ) 
(hv : non_zero_vector m) : 
  point_on_line M M0 m ↔ (M.1 - M0.1) / m.1 = (M.2 - M0.2) / m.2 ∧ (M.1 - M0.1) / m.1 = (M.3 - M0.3) / m.3 :=
sorry

end point_lies_on_line_l530_530398


namespace largest_n_factorial_product_l530_530185

theorem largest_n_factorial_product : ∃ n : ℕ, (∀ m : ℕ, m ≥ n → ¬ (m! = ∏ i in (range (m - 4)).map (λ k, k + 1 + m), (1 : ℕ))) ∧ n = 1 :=
sorry

end largest_n_factorial_product_l530_530185


namespace find_ac_pair_l530_530954

theorem find_ac_pair (a c : ℤ) (h1 : a + c = 37) (h2 : a < c) (h3 : 36^2 - 4 * a * c = 0) : a = 12 ∧ c = 25 :=
by
  sorry

end find_ac_pair_l530_530954


namespace triangle_ADN_area_l530_530324

-- Define the triangle ABC with area 1
axiom TriangleABC (A B C : Type*) (abc_area : ℝ) (H1 : abc_area = 1)

-- Define the midpoint P of side BC
axiom MidpointP (B C P : Type*) (H2 : P = midpoint B C)

-- Define points M and N on AB and AC respectively
axiom PointsMN (A B C M N : Type*) 
  (H3 : M ∈ segment A B) (H4 : N ∈ segment A C) 
  (H5 : dist A M = 2 * dist M B) (H6 : dist C N = 2 * dist A N)

-- Define the intersection point D of lines AP and MN
axiom IntersectionD (A P M N D : Type*) (H7 : D ∈ line_through A P)
  (H8 : D ∈ line_through M N)

-- Define the theorem to find the area of triangle ADN
theorem triangle_ADN_area {A B C P M N D : Type*}
  (H1 : abc_area = 1) (H2 : P = midpoint B C)
  (H3 : M ∈ segment A B) (H4 : N ∈ segment A C)
  (H5 : dist A M = 2 * dist M B) (H6 : dist C N = 2 * dist A N)
  (H7 : D ∈ line_through A P) (H8 : D ∈ line_through M N) :
  triangle_area A D N = 2 / 27 :=
by sorry

end triangle_ADN_area_l530_530324


namespace minimize_F_l530_530207

variables {a b : ℝ} (f : ℝ → ℝ)
-- Conditions:
-- 1. f'(x) > 0 for a <= x <= b
-- 2. F(x) = ∫ a to b |f(t) - f(x)| dt

noncomputable def F (x : ℝ) : ℝ := ∫ t in Set.Icc a b, |f t - f x|

theorem minimize_F (h : ∀ x, a ≤ x ∧ x ≤ b → deriv f x > 0) :
  ∃ x, (x = (a + b) / 2) ∧ ∀ y, F f y ≥ F f x :=
begin
  sorry
end

end minimize_F_l530_530207


namespace min_area_of_square_l530_530160

noncomputable def minimum_square_area (p q r s : ℤ) : ℂ :=
  let f : ℂ → ℂ := λ x, x^4 + (p : ℂ) * x^3 + (q : ℂ) * x^2 + (r : ℂ) * x + (s : ℂ)
  let roots := {z : ℂ | f z = 0}
  let vertices := roots.map (λ z, z)
  let square_area := (Complex.abs (vertices[1] - vertices[0]))^2
  2

theorem min_area_of_square {p q r s : ℤ} (h : ∃ z : ℂ, z^4 + (p : ℂ) * z^3 + (q : ℂ) * z^2 + (r : ℂ) * z + (s : ℂ) = 0) :
  minimum_square_area p q r s = 2 :=
sorry

end min_area_of_square_l530_530160


namespace john_weekly_earnings_increase_l530_530530

theorem john_weekly_earnings_increase :
  let original_amount := 60
  let new_amount := 100
  let increase := (new_amount - original_amount) / original_amount * 100
  increase = 66.67 :=
by
  let original_amount := 60
  let new_amount := 100
  let increase := (new_amount - original_amount) / original_amount * 100
  have h1 : increase = 66.67, from sorry
  exact h1

end john_weekly_earnings_increase_l530_530530


namespace probability_at_least_one_interested_l530_530090

def total_members : ℕ := 20
def interested_ratio : ℚ := 4 / 5
def interested_members : ℕ := interested_ratio * total_members
def not_interested_members : ℕ := total_members - interested_members

theorem probability_at_least_one_interested :
  let prob_first_not_interested := (not_interested_members : ℚ) / total_members
  let prob_second_not_interested := ((not_interested_members - 1 : ℚ) / (total_members - 1))
  let prob_both_not_interested := prob_first_not_interested * prob_second_not_interested
  let prob_at_least_one_interested := 1 - prob_both_not_interested
  prob_at_least_one_interested = 92 / 95 :=
by
  sorry

end probability_at_least_one_interested_l530_530090


namespace multiple_of_average_speed_l530_530012

theorem multiple_of_average_speed 
  (average_speed : ℝ)
  (hours : ℝ)
  (total_distance : ℝ)
  (goal_distance : ℝ) :
  average_speed = 66 → hours = 4 → goal_distance = 528 →
  total_distance = average_speed * hours →
  m * average_speed * hours = goal_distance → 
  m = 2 :=
by
  intros avg_speed_is_66 hrs_is_4 goal_dist_is_528 total_dist_calc goal_dist_calc
  have h1 : total_distance = 66 * 4 := by
    rw [avg_speed_is_66, hrs_is_4]
  have h2 : total_distance = 264 := by
    linarith
  have h3 : 528 = m * 264 := by
    rw [goal_dist_is_528, h2, goal_dist_calc]
  have h4 : m = 528 / 264 := by
    linarith
  linarith

end multiple_of_average_speed_l530_530012


namespace sum_of_19th_set_is_29572_l530_530704

-- Define the sequence rules and properties.
noncomputable def first_element_of_set : ℕ → ℕ
| 0     := 1
| (n+1) := first_element_of_set n + 2 * (n + 1) - 1

def set_elements (n : ℕ) : List ℕ := 
List.range (n+1) |>.map (λ k => first_element_of_set n + 2 * k)

def sum_of_set (n : ℕ) : ℕ :=
(set_elements n).sum

-- Proving the specific case for \( \tilde{S}_{19} \)
theorem sum_of_19th_set_is_29572 : sum_of_set 19 = 29572 :=
by
  -- proof steps can be added here
  sorry

end sum_of_19th_set_is_29572_l530_530704


namespace probability_heads_given_heads_l530_530426

-- Definitions based on the conditions
inductive Coin
| coin1 | coin2 | coin3

open Coin

def P (event : Set Coin) : ℝ := 
  if event = {coin1} then 1/3
  else if event = {coin2} then 1/3
  else if event = {coin3} then 1/3
  else 0

def event_heads (coin : Coin) : bool :=
  match coin with
  | coin1 => true
  | coin2 => true
  | coin3 => false

-- Probability of showing heads
def P_heads_given_coin (coin : Coin) : ℝ :=
  if event_heads coin then 1 else 0

-- Total probability of getting heads
def P_heads : ℝ := 
  P {coin1} * (if event_heads coin1 then 1 else 0) +
  P {coin2} * (if event_heads coin2 then 1 else 0) +
  P {coin3} * (if event_heads coin3 then 0 else 0)

-- Using Bayes' Theorem to find probability that the coin showing heads is coin2
def P_coin2_given_heads : ℝ :=
  (P_heads_given_coin coin2 * P {coin2}) / P_heads

-- Defining the theorem statement
theorem probability_heads_given_heads : P_coin2_given_heads = 2 / 3 := by
  sorry

end probability_heads_given_heads_l530_530426


namespace smaller_octagon_area_ratio_l530_530882

theorem smaller_octagon_area_ratio (ABCDEFH: regular_octagon) :
  let I J K L M N O P be midpoints of sides AB BC CD DE EF FG GH HA respectively,
  let smaller_octagon be the octagon bounded by segments AI BJ CK DL EM FN GO HP,
  let area_ratio := (area smaller_octagon) / (area ABCDEFH)
  let m n be relatively prime positive integers such that area_ratio == m / n,
  m + n = 7 :=
begin
  sorry
end

end smaller_octagon_area_ratio_l530_530882


namespace find_sum_l530_530409

variable (P T R : ℝ)

def simple_interest (P T R : ℝ) : ℝ := (P * T * R) / 100
def true_discount (P T R : ℝ) : ℝ := (P * T * R) / (100 + (T * R))

theorem find_sum
  (h1: simple_interest P T R = 85)
  (h2: true_discount P T R = 80) :
  P = 1360 :=
sorry

end find_sum_l530_530409


namespace part_1_part_2_l530_530708

open Real

noncomputable def slope_of_line_passing_through (P : ℝ × ℝ) (m : ℝ) : ℝ × ℝ → Prop :=
λ Q, P.2 + P.1 * m = Q.2

def conjugate_pair_P (P : ℝ × ℝ) (λ : ℝ) (l1 l2 : ℝ → ℝ) : Prop :=
∃ k1 k2, (∀ x, l1 x = k1 * x ∧ l2 x = k2 * x) ∧ k1 * k2 = λ

def conjugate_pair_O_neg3 (l1 l2 : ℝ → ℝ) : Prop :=
conjugate_pair_P (0, 0) (-3) l1 l2

def conjugate_pair_Q_neg2 (l1 l2 : ℝ → ℝ) : Prop :=
conjugate_pair_P (-1, -sqrt 2) (-2) l1 l2

theorem part_1 :
  (conjugate_pair_O_neg3 (λ x, 2 * x) l2) → (∀ x, l2 x = -3 / 2 * x) :=
sorry

theorem part_2 :
  ∀ (l1 l2 : ℝ → ℝ),
  (conjugate_pair_Q_neg2 l1 l2) →
  (∀ m1 m2 : ℝ, (∀ x, l1 x = m1 * (x + 1) - sqrt 2) ∧ (∀ x, l2 x = m2 * (x + 1) - sqrt 2)) →
  ∃ d1 d2 : ℝ, 
  (d1 * d2 = sqrt 2 ∨ d1 * d2 < sqrt 2) ∧ 0 ≤ d1 * d2 ∧ d1 * d2 < sqrt 2 :=
sorry

end part_1_part_2_l530_530708


namespace all_rationals_on_number_line_l530_530528

theorem all_rationals_on_number_line :
  ∀ q : ℚ, ∃ p : ℝ, p = ↑q :=
by
  sorry

end all_rationals_on_number_line_l530_530528


namespace coin_problem_l530_530428

noncomputable def P_flip_heads_is_heads (coin : ℕ → bool → Prop) : ℚ :=
if coin 2 tt then 2 / 3 else 0

theorem coin_problem (coin : ℕ → bool → Prop)
  (h1 : ∀ coin_num, coin coin_num tt = (coin_num = 1 ∨ coin_num = 2))
  (h2 : ∀ coin_num, coin coin_num ff = (coin_num = 1 ∨ coin_num = 3)):
  P_flip_heads_is_heads coin = 2 / 3 :=
by
  sorry

end coin_problem_l530_530428


namespace volume_of_soup_in_hemisphere_half_height_l530_530041

theorem volume_of_soup_in_hemisphere_half_height 
  (V_hemisphere : ℝ)
  (hV_hemisphere : V_hemisphere = 8)
  (V_cap : ℝ) :
  V_cap = 2.5 :=
sorry

end volume_of_soup_in_hemisphere_half_height_l530_530041


namespace find_150th_letter_l530_530450
open Nat

def repeating_sequence := "XYZ"

def length_repeating_sequence := 3

theorem find_150th_letter : (150 % length_repeating_sequence == 0) → repeating_sequence[(length_repeating_sequence - 1) % length_repeating_sequence] = 'Z' := 
by
  sorry

end find_150th_letter_l530_530450


namespace max_possible_min_dist_deg_l530_530929

-- Let m and n be relatively prime positive integers
def relatively_prime (m n : ℕ) : Prop := Nat.gcd m n = 1

-- Define the maximum possible value of the minimum distance d between vertices of m-gon and n-gon
def max_min_distance (m n : ℕ) [Fact (relatively_prime m n)] : ℚ :=
  180 / (m * n)

-- The theorem statement asserting the maximum possible value of d given the conditions
theorem max_possible_min_dist_deg (m n : ℕ) [h_rel : Fact (relatively_prime m n)] :
  ∃ d, d = 180 / (m * n) :=
begin
  use max_min_distance m n,
  exact sorry
end

end max_possible_min_dist_deg_l530_530929


namespace letter_at_position_150_l530_530474

theorem letter_at_position_150 : 
  (∀ n, n > 0 → ∃ i, i ∈ {1, 2, 3} ∧ "XYZ".to_list[i-1] = "XYZ".to_list[(n - 1) % 3]) →
  ("XYZ".to_list[(150 - 1) % 3] = 'Z') :=
by
  sorry

end letter_at_position_150_l530_530474


namespace diagonal_count_of_convex_polygon_30_sides_l530_530791
-- Importing the entire Mathlib library for necessary mathematical constructs

theorem diagonal_count_of_convex_polygon_30_sides : 
  let n := 30 in
  let num_diagonals := (n * (n - 3)) / 2 in
  num_diagonals = 405 := 
by {
  let n := 30
  let num_diagonals := (n * (n - 3)) / 2
  show num_diagonals = 405,
  sorry
}

end diagonal_count_of_convex_polygon_30_sides_l530_530791


namespace letter_150_is_Z_l530_530501

/-- Definition of the repeating pattern "XYZ" -/
def pattern : List Char := ['X', 'Y', 'Z']

/-- The repeating pattern has a length of 3 -/
def pattern_length : ℕ := 3

/-- Calculate the 150th letter in the repeating pattern "XYZ" -/
def nth_letter_in_pattern (n : ℕ) : Char :=
  let m := n % pattern_length
  if m = 0 then pattern[2] else pattern[m - 1]

/-- Prove that the 150th letter in the pattern "XYZ" is 'Z' -/
theorem letter_150_is_Z : nth_letter_in_pattern 150 = 'Z' :=
by
  sorry

end letter_150_is_Z_l530_530501


namespace preserve_area_needed_l530_530574

theorem preserve_area_needed 
  (current_rhinoceroses : ℕ := 8000)
  (watering_area : ℕ := 10000)
  (grazing_area_per_rhino : ℕ := 100)
  (expected_increase_percent : ℕ := 10) :
  (1000 * (current_rhinoceroses * (1 + expected_increase_percent / 100) / 1 * grazing_area_per_rhino + watering_area) = 890 :=
by
  sorry

end preserve_area_needed_l530_530574


namespace convert_20121_base3_to_base10_l530_530071

/- Define the base conversion function for base 3 to base 10 -/
def base3_to_base10 (d4 d3 d2 d1 d0 : ℕ) :=
  d4 * 3^4 + d3 * 3^3 + d2 * 3^2 + d1 * 3^1 + d0 * 3^0

/- Define the specific number in base 3 -/
def num20121_base3 := (2, 0, 1, 2, 1)

/- The theorem stating the equivalence of the base 3 number 20121_3 to its base 10 equivalent -/
theorem convert_20121_base3_to_base10 :
  base3_to_base10 2 0 1 2 1 = 178 :=
by
  sorry

end convert_20121_base3_to_base10_l530_530071


namespace first_car_speed_l530_530058

theorem first_car_speed
  (highway_length : ℝ)
  (second_car_speed : ℝ)
  (meeting_time : ℝ)
  (D1 D2 : ℝ) :
  highway_length = 45 → second_car_speed = 16 → meeting_time = 1.5 → D2 = second_car_speed * meeting_time → D1 + D2 = highway_length → D1 = 14 * meeting_time :=
by
  intros h_highway h_speed h_time h_D2 h_sum
  sorry

end first_car_speed_l530_530058


namespace find_150th_letter_l530_530462

theorem find_150th_letter :
  let pattern := ['X', 'Y', 'Z']
  150 % 3 = 0 -> pattern[(150 % 3 + 2) % 3] = 'Z' :=
begin
  intros pattern h,
  simp at *,
  exact rfl,
end

end find_150th_letter_l530_530462


namespace semicircles_problem_l530_530158

-- Define the problem in Lean
theorem semicircles_problem 
  (D : ℝ) -- Diameter of the large semicircle
  (N : ℕ) -- Number of small semicircles
  (r : ℝ) -- Radius of each small semicircle
  (H1 : D = 2 * N * r) -- Combined diameter of small semicircles is equal to the large semicircle's diameter
  (H2 : (N * (π * r^2 / 2)) / ((π * (N * r)^2 / 2) - (N * (π * r^2 / 2))) = 1 / 10) -- Ratio of areas condition
  : N = 11 :=
   sorry -- Proof to be filled in later

end semicircles_problem_l530_530158


namespace Anne_is_15_pounds_heavier_l530_530600

def Anne_weight : ℕ := 67
def Douglas_weight : ℕ := 52

theorem Anne_is_15_pounds_heavier : Anne_weight - Douglas_weight = 15 := by
  sorry

end Anne_is_15_pounds_heavier_l530_530600


namespace symmetric_points_l530_530372

theorem symmetric_points (a b : ℝ) (h1 : 2 * a + 1 = -1) (h2 : 4 = -(3 * b - 1)) :
  2 * a + b = -3 := 
sorry

end symmetric_points_l530_530372


namespace fertilizer_production_l530_530120

theorem fertilizer_production (daily_production : ℕ) (days : ℕ) (total_production : ℕ) 
  (h1 : daily_production = 105) 
  (h2 : days = 24) 
  (h3 : total_production = daily_production * days) : 
  total_production = 2520 := 
  by 
  -- skipping the proof
  sorry

end fertilizer_production_l530_530120


namespace sum_f_values_l530_530728

def f (x : ℝ) : ℝ := (1 / x) + (x ^ (1 / 3)) + 2

theorem sum_f_values : 
  (∑ i in (-26 : ℤ)..(-1 : ℤ), f (i : ℝ)) +
  (∑ i in (1 : ℤ)..(27 : ℤ), f (i : ℝ)) = 2944 / 27 := by
  sorry

end sum_f_values_l530_530728


namespace cost_prices_max_profit_find_m_l530_530931

-- Part 1
theorem cost_prices (x y: ℕ) (h1 : 40 * x + 30 * y = 5000) (h2 : 10 * x + 50 * y = 3800) : 
  x = 80 ∧ y = 60 :=
sorry

-- Part 2
theorem max_profit (a: ℕ) (h1 : 70 ≤ a ∧ a ≤ 75) : 
  (20 * a + 6000) ≤ 7500 :=
sorry

-- Part 3
theorem find_m (m : ℝ) (h1 : 4 < m ∧ m < 8) (h2 : (20 - 5 * m) * 70 + 6000 = 5720) : 
  m = 4.8 :=
sorry

end cost_prices_max_profit_find_m_l530_530931


namespace sin_180_eq_zero_l530_530620

theorem sin_180_eq_zero : Real.sin (180 * Real.pi / 180) = 0 := by
  -- Simplifying the angle, 180 degrees = π radians
  let angle := 180 * Real.pi / 180
  have h : angle = Real.pi := by
    simp [angle, Real.pi]
  rw h
  -- From the unit circle, we know the sine of angle π is 0
  exact Real.sin_pi

end sin_180_eq_zero_l530_530620


namespace min_positive_difference_of_composite_sum_96_l530_530101

-- Definition of a composite number
def is_composite (n : ℕ) : Prop :=
  ∃ p1 p2 : ℕ, 1 < p1 ∧ 1 < p2 ∧ n = p1 * p2

-- The theorem to prove
theorem min_positive_difference_of_composite_sum_96 :
  ∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ a + b = 96 ∧ a ≠ b ∧ abs (a - b) = 4 :=
sorry

end min_positive_difference_of_composite_sum_96_l530_530101


namespace collinear_midpoints_l530_530031

theorem collinear_midpoints
  (A B C C1 C2 A1 A2 B1 B2 : Point)
  (u v w : ℝ)
  (h1 : C1 ∈ segment A B)
  (h2 : C2 ∈ line_through A B)
  (h3 : A1 ∈ segment B C)
  (h4 : A2 ∈ segment B C)
  (h5 : B1 ∈ segment C A)
  (h6 : B2 ∈ segment C A)
  (h7 : (distance A C1) / (distance C1 B) = u / v)
  (h8 : (distance A C2) / (distance C2 B) = u / v)
  (h9 : (distance B A1) / (distance A1 C) = v / w)
  (h10 : (distance B A2) / (distance A2 C) = v / w)
  (h11 : (distance C B1) / (distance B1 A) = w / u)
  (h12 : (distance C B2) / (distance B2 A) = w / u) :
  collinear [(midpoint A1 A2), (midpoint B1 B2), (midpoint C1 C2)] := sorry

end collinear_midpoints_l530_530031


namespace no_nat_solutions_l530_530377

theorem no_nat_solutions (x y z : ℕ) : x^2 + y^2 + z^2 ≠ 2 * x * y * z :=
sorry

end no_nat_solutions_l530_530377


namespace cost_price_eq_selling_price_25_l530_530842

theorem cost_price_eq_selling_price_25 (C S : ℝ) (h1 : ∃ (X : ℝ), X * C = 25 * S) (h2 : S = 2 * C) : 
  ∃ (X : ℝ), X = 50 :=
by
  obtain ⟨X, hX⟩ := h1
  use X
  have h : X * C = 25 * (2 * C),
  {
    rw h2 at hX,
    exact hX,
  }
  calc
    X = 50 : by sorry

end cost_price_eq_selling_price_25_l530_530842


namespace enclosed_area_of_curve_is_correct_l530_530944

noncomputable def area_enclosed_by_curve (arc_length : ℝ) (side_length : ℝ) : ℝ := sorry

theorem enclosed_area_of_curve_is_correct :
  area_enclosed_by_curve (2 * pi / 3) 2 = pi + 6 * real.sqrt 3 :=
by sorry

end enclosed_area_of_curve_is_correct_l530_530944


namespace handshake_count_l530_530416

-- Defining the conditions
def number_of_companies : ℕ := 5
def representatives_per_company : ℕ := 5
def total_participants : ℕ := number_of_companies * representatives_per_company

-- Defining the number of handshakes each person makes
def handshakes_per_person : ℕ := total_participants - 1 - (representatives_per_company - 1)

-- Defining the total number of handshakes
def total_handshakes : ℕ := (total_participants * handshakes_per_person) / 2

theorem handshake_count :
  total_handshakes = 250 :=
by
  sorry

end handshake_count_l530_530416


namespace thirteenth_digit_l530_530981

-- Let's define the fractions and their decimal equivalents
def f1 : ℚ := 1 / 8
def f2 : ℚ := 1 / 11

-- Defining a function to obtain the nth digit after the decimal point
def nth_decimal_digit (q : ℚ) (n : ℕ) : ℕ :=
  let decimal_str := q.to_string.split_on '.'[1]
  let digit_strs := decimal_str.foldl (++) ""
  let digit_index := (n - 1) % digit_str.size
  digit_strs.get_digit digit_index.to_nat

-- The required theorem which states the 13th digit after the decimal point
theorem thirteenth_digit : nth_decimal_digit (f1 + f2) 13 = 9 :=
  sorry

end thirteenth_digit_l530_530981


namespace proposition_p_l530_530218

open Real

def f (x : ℝ) := -x + sin x

theorem proposition_p : ∀ x ∈ Ioo 0 (π/2), f x < 0 := 
by
  -- proof would go here
  sorry

end proposition_p_l530_530218


namespace sin_180_degrees_l530_530641

theorem sin_180_degrees : Real.sin (180 * Real.pi / 180) = 0 := 
by
  sorry

end sin_180_degrees_l530_530641


namespace trent_walks_to_bus_stop_l530_530431

theorem trent_walks_to_bus_stop (x : ℕ) (h1 : 2 * (x + 7) = 22) : x = 4 :=
sorry

end trent_walks_to_bus_stop_l530_530431


namespace area_of_closed_figure_l530_530010

theorem area_of_closed_figure : 
  ∫ x in 0..1, (x - x^2) = 1 / 6 := 
sorry

end area_of_closed_figure_l530_530010


namespace parabola_focus_distance_proof_l530_530281

noncomputable def parabola_focus_distance : Prop :=
  ∃ t : ℝ, 
  let P := (3, 4*t) in
  let F := (1, 0) in -- Focus of the parabola y^2 = 4x is at (1, 0)
  dist P F = 4

theorem parabola_focus_distance_proof : parabola_focus_distance :=
  sorry

end parabola_focus_distance_proof_l530_530281


namespace conclusion_1_conclusion_2_conclusion_4_correct_conclusions_l530_530328

def operation (a b : ℝ) : ℝ := (a + b)^2 - (a - b)^2

theorem conclusion_1 (a b : ℝ) : operation a b = 0 -> (a = 0 ∨ b = 0) :=
sorry

theorem conclusion_2 (a b c : ℝ) : operation a (b + c) = operation a b + operation a c :=
sorry

theorem conclusion_4 (a b : ℝ) (h : 2 * (a + b) = perimeter) : a = b -> ∀ (a b : ℝ), operation a b = 4 * a * b :=
sorry

theorem correct_conclusions :
  conclusion_1 ∧
  conclusion_2 ∧
  conclusion_4 :=
sorry

end conclusion_1_conclusion_2_conclusion_4_correct_conclusions_l530_530328


namespace correct_statement_l530_530085

def complementary_events_mutually_exclusive (A B : Prop) :=
  ∀ (h1 : A ≠ B), A ∧ ¬ B ∧ (A ∨ B)

theorem correct_statement :
  let A := "Complementary events are also mutually exclusive events."
  let B := "The probability of a certain event occurring is 1.1."
  let C := "Two events that cannot occur at the same time are two complementary events."
  let D := "The probability of a certain event occurring changes with the number of experiments."
  A = "Complementary events are also mutually exclusive events." :=
by
  have h1 : A ≠ B := sorry
  have h2 : A ∧ ¬ B ∧ (A ∨ B) := sorry
  exact sorry

end correct_statement_l530_530085


namespace distance_is_absolute_value_l530_530363

noncomputable def distance_to_origin (x : ℝ) : ℝ := |x|

theorem distance_is_absolute_value (x : ℝ) : distance_to_origin x = |x| :=
by
  sorry

end distance_is_absolute_value_l530_530363


namespace probability_six_distinct_numbers_l530_530987

theorem probability_six_distinct_numbers (eight_sided_dice : Finset ℕ) (h : eight_sided_dice.card = 8) : 
  (∃ dice_rolls : Finset (fin 8) → ℕ, dice_rolls.card = 6) → 
  (probability : ℚ) = 315 / 4096 :=
by
  sorry

end probability_six_distinct_numbers_l530_530987


namespace solve_quad_l530_530959

theorem solve_quad (x : ℝ) : 3 * x^2 = real.sqrt 3 * x ↔ (x = 0 ∨ x = real.sqrt 3 / 3) :=
by
  sorry

end solve_quad_l530_530959


namespace polygon_internal_angle_twice_external_l530_530564

theorem polygon_internal_angle_twice_external (n : ℕ) 
  (h1 : ∀ (p : Polygon), Polygon.sum_external_angles p = 360)
  (h2 : Polygon.sum_internal_angles n = (n - 2) * 180) 
  (h3 : Polygon.sum_internal_angles n = 2 * (Polygon.sum_external_angles p)) :
  n = 6 :=
by sorry

end polygon_internal_angle_twice_external_l530_530564


namespace sin_180_degrees_l530_530636

theorem sin_180_degrees : Real.sin (180 * Real.pi / 180) = 0 := 
by
  sorry

end sin_180_degrees_l530_530636


namespace break_even_point_l530_530386

/-- Conditions of the problem -/
def fixed_costs : ℝ := 10410
def variable_cost_per_unit : ℝ := 2.65
def selling_price_per_unit : ℝ := 20

/-- The mathematically equivalent proof problem / statement -/
theorem break_even_point :
  fixed_costs / (selling_price_per_unit - variable_cost_per_unit) = 600 := 
by
  -- Proof to be filled in
  sorry

end break_even_point_l530_530386


namespace diagonals_of_30_sided_polygon_l530_530801

theorem diagonals_of_30_sided_polygon : 
  ∀ (n : ℕ), n = 30 → (n * (n - 3)) / 2 = 405 := 
by
  intro n h
  rw h
  simp
  sorry

end diagonals_of_30_sided_polygon_l530_530801
